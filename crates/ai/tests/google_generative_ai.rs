//! The Google Generative AI adapter suite, ported from
//! `packages/ai/test/google-raw-stop-reason.test.ts`,
//! `packages/ai/test/google-thinking-level-map.test.ts` (the Generative AI
//! sections; the Vertex sections ride with the Vertex adapter), and
//! `packages/ai/test/google-shared-retry.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: the `vi.mock` of `@google/genai` becomes
//! `MockHttpClient` routes over the transport seam — the SDK boundary the
//! tests captured (constructor config, streamed chunks) moves to the recorded
//! request headers and the SSE body. The capture helpers assert the captured
//! payload directly: upstream's throw-to-capture trick ("payload captured")
//! has no hook-throwing counterpart.

#![expect(
    clippy::expect_used,
    reason = "the tests pin adapter outcomes; an unexpected shape panics the test by design"
)]

use std::collections::BTreeMap;

use pi_ai::api::google_generative_ai::{
    GoogleStreamOptions, get_google_budget, stream as stream_google, stream_simple,
};
use pi_ai::api::google_shared::{ResolvedGoogleThinkingLevel, resolve_google_thinking_level};
use pi_ai::http::mock::MockResponse;
use pi_ai::types::{
    Api, Context, Message, Model, ModelThinkingLevel, Modality, ProviderId, SimpleStreamOptions,
    ThinkingBudgets, ThinkingLevel, ThinkingLevelMap, ToolCall, UserContent, UserMessage,
};
use pi_ai::utils::pi_user_agent::get_pi_user_agent;
use serde_json::json;

mod common;

/// The user-message-only context the suite runs.
fn context() -> Context {
    Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("hello".to_owned()),
            timestamp: 0,
        })],
        tools: None,
    }
}

/// The test Google model with a thinking map, upstream's `googleModel`.
fn google_model(id: &str, thinking_level_map: ThinkingLevelMap) -> Model {
    Model {
        id: id.to_owned(),
        name: id.to_owned(),
        api: Api::from("google-generative-ai"),
        provider: ProviderId::from("test-google"),
        base_url: "https://example.invalid/v1beta".to_owned(),
        reasoning: true,
        thinking_level_map: Some(thinking_level_map),
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 128_000,
        max_tokens: 4_096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// One streamed chunk as a data-only SSE response body, the shape the
/// `@google/genai` mock yields upstream.
fn sse_data_response(status: u16, chunk: &str) -> MockResponse {
    MockResponse::status(status)
        .with_header("content-type", "text/event-stream")
        .with_body(format!("data: {chunk}\n\n"))
}

/// Mount the mock's single streamed chunk, upstream's `generateContentStream`
/// async generator.
fn mount_google_stream(mock: &pi_ai::http::mock::MockHttpClient, chunk: &str) {
    let response = sse_data_response(200, chunk);
    mock.on(|request| request.url.contains(":streamGenerateContent"))
        .respond(response);
}

/// The terminal chunk the raw-stop-reason suite yields, with an optional
/// function-call part, upstream's mock fixture.
fn raw_stop_chunk(finish_reason: &str, include_function_call: bool) -> String {
    let mut candidate = json!({ "finishReason": finish_reason });
    if include_function_call {
        candidate["content"] = json!({
            "parts": [{
                "functionCall": {
                    "id": "call-1",
                    "name": "echo",
                    "args": { "value": "truncated" },
                }
            }]
        });
    }
    json!({
        "responseId": "google-response-id",
        "candidates": [candidate],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 0,
            "totalTokenCount": 1,
        },
    })
    .to_string()
}

fn gemini_options(mock: &pi_ai::http::mock::MockHttpClient) -> GoogleStreamOptions {
    GoogleStreamOptions {
        transport_options: common::mock_transport(mock),
        api_key: Some("test-api-key".to_owned()),
        ..GoogleStreamOptions::default()
    }
}

/// The catalog gemini-2.5-flash model, upstream's `getModel("google", ...)`.
fn catalog_gemini() -> Model {
    common::builtin_model("google", "gemini-2.5-flash")
}

// --- upstream google-raw-stop-reason.test.ts (Generative AI sections) ---

/// An unmapped finish reason surfaces as an error stop with the raw wire
/// value preserved and the exact provider-stopped wording.
#[tokio::test]
async fn preserves_raw_gemini_finish_reasons_for_errors() {
    let mock = pi_ai::http::mock::MockHttpClient::new();
    mount_google_stream(&mock, &raw_stop_chunk("MALFORMED_FUNCTION_CALL", false));
    let model = catalog_gemini();
    let options = gemini_options(&mock);

    let message = stream_google(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Error);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("MALFORMED_FUNCTION_CALL"));
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider stopped with: MALFORMED_FUNCTION_CALL")
    );
}

/// `MAX_TOKENS` with a tool call stays `length`: the tool-use upgrade only
/// applies to `STOP`.
#[tokio::test]
async fn preserves_max_tokens_with_a_tool_call_as_length() {
    let mock = pi_ai::http::mock::MockHttpClient::new();
    mount_google_stream(&mock, &raw_stop_chunk("MAX_TOKENS", true));
    let model = catalog_gemini();
    let options = gemini_options(&mock);

    let message = stream_google(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Length);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("MAX_TOKENS"));
    assert!(message
        .content
        .iter()
        .any(|block| matches!(block, pi_ai::types::AssistantBlock::ToolCall(_))));
}

/// `STOP` with a tool call upgrades to `toolUse`.
#[tokio::test]
async fn maps_stop_with_a_tool_call_to_tool_use() {
    let mock = pi_ai::http::mock::MockHttpClient::new();
    mount_google_stream(&mock, &raw_stop_chunk("STOP", true));
    let model = catalog_gemini();
    let options = gemini_options(&mock);

    let message = stream_google(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::ToolUse);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("STOP"));
    let Some(pi_ai::types::AssistantBlock::ToolCall(ToolCall { id, .. })) =
        message.content.iter().find(|block| matches!(block, pi_ai::types::AssistantBlock::ToolCall(_)))
    else {
        panic!("expected a tool call block");
    };
    assert_eq!(id, "call-1");
}

/// The default `User-Agent` is pi's own value overriding the SDK default.
#[tokio::test]
async fn uses_pi_user_agent_by_default() {
    let mock = pi_ai::http::mock::MockHttpClient::new();
    mount_google_stream(&mock, &raw_stop_chunk("STOP", false));
    let model = catalog_gemini();
    let options = gemini_options(&mock);

    let _ = stream_google(&model, &context(), Some(&options)).result().await;

    assert_eq!(
        common::recorded_header(&mock, "User-Agent").as_deref(),
        Some(get_pi_user_agent().as_str())
    );
}

/// An explicit caller `User-Agent` overrides the default.
#[tokio::test]
async fn lets_explicit_headers_override_the_default_user_agent() {
    let mock = pi_ai::http::mock::MockHttpClient::new();
    mount_google_stream(&mock, &raw_stop_chunk("STOP", false));
    let model = catalog_gemini();
    let mut options = gemini_options(&mock);
    options.headers = Some(BTreeMap::from([(
        "User-Agent".to_owned(),
        Some("custom-agent".to_owned()),
    )]));

    let _ = stream_google(&model, &context(), Some(&options)).result().await;

    assert_eq!(
        common::recorded_header(&mock, "User-Agent").as_deref(),
        Some("custom-agent")
    );
}

// --- upstream google-thinking-level-map.test.ts (Generative AI sections) ---

/// The level resolver accepts every pi level and lowercase/uppercase mapped
/// values, and rejects unknown mappings with upstream's wording.
#[test]
fn exhaustively_resolves_supported_logical_levels_and_mapping_values() {
    let model = google_model("gemini-3.7-flash", ThinkingLevelMap::new());
    assert_eq!(
        resolve_google_thinking_level(&model, ModelThinkingLevel::Off)
            .expect("off"),
        ResolvedGoogleThinkingLevel::High
    );
    assert_eq!(
        resolve_google_thinking_level(&model, ModelThinkingLevel::Minimal)
            .expect("minimal"),
        ResolvedGoogleThinkingLevel::Minimal
    );
    assert_eq!(
        resolve_google_thinking_level(&model, ModelThinkingLevel::Low).expect("low"),
        ResolvedGoogleThinkingLevel::Low
    );
    assert_eq!(
        resolve_google_thinking_level(&model, ModelThinkingLevel::Medium)
            .expect("medium"),
        ResolvedGoogleThinkingLevel::Medium
    );
    assert_eq!(
        resolve_google_thinking_level(&model, ModelThinkingLevel::High).expect("high"),
        ResolvedGoogleThinkingLevel::High
    );

    for mapped in ["minimal", "low", "medium", "high", "MINIMAL", "LOW", "MEDIUM", "HIGH"] {
        let model = model_with_map("gemini-3.7-flash", &[
            (ModelThinkingLevel::High, mapped),
            (ModelThinkingLevel::Xhigh, mapped),
            (ModelThinkingLevel::Max, mapped),
        ]);
        let expected = match mapped.to_lowercase().as_str() {
            "minimal" => ResolvedGoogleThinkingLevel::Minimal,
            "low" => ResolvedGoogleThinkingLevel::Low,
            "medium" => ResolvedGoogleThinkingLevel::Medium,
            _ => ResolvedGoogleThinkingLevel::High,
        };
        for level in [ModelThinkingLevel::High, ModelThinkingLevel::Xhigh, ModelThinkingLevel::Max] {
            assert_eq!(
                resolve_google_thinking_level(&model, level).expect("resolved"),
                expected,
                "{mapped} at {level:?}"
            );
        }
    }

    let invalid = model_with_map("gemini-3.7-flash", &[(ModelThinkingLevel::Xhigh, "extreme")]);
    assert_eq!(
        resolve_google_thinking_level(&invalid, ModelThinkingLevel::Xhigh),
        Err("Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: xhigh -> extreme"
            .to_owned())
    );
    assert_eq!(
        resolve_google_thinking_level(&model, ModelThinkingLevel::Max),
        Err("Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: max -> undefined"
            .to_owned())
    );
}

fn model_with_map(id: &str, entries: &[(ModelThinkingLevel, &str)]) -> Model {
    let mut map = ThinkingLevelMap::new();
    for (level, value) in entries {
        map.insert(*level, Some((*value).to_owned()));
    }
    thinking_map_model(id, map)
}

/// The thinking-map model fixture, upstream's `googleModel`.
fn thinking_map_model(id: &str, thinking_level_map: ThinkingLevelMap) -> Model {
    Model {
        id: id.to_owned(),
        name: id.to_owned(),
        api: Api::from("google-generative-ai"),
        provider: ProviderId::from("test-google"),
        base_url: "https://example.invalid/v1beta".to_owned(),
        reasoning: true,
        thinking_level_map: Some(thinking_level_map),
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 128_000,
        max_tokens: 4_096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// Capture the `streamSimple` payload through the hook and settle the stream,
/// the Rust encoding of upstream's throw-to-capture helper: the hook records
/// and keeps the payload, and the empty mock fails the request.
async fn capture_payload(
    model: &Model,
    reasoning: ThinkingLevel,
    thinking_budgets: Option<ThinkingBudgets>,
) -> (serde_json::Value, pi_ai::types::AssistantMessage) {
    let mock = pi_ai::http::mock::MockHttpClient::new();
    let (on_payload, captured) = common::payload_capture();
    let mut transport = common::mock_transport(&mock);
    transport.on_payload = Some(on_payload);
    let options = SimpleStreamOptions {
        transport_options: transport,
        api_key: Some("test".to_owned()),
        reasoning: Some(reasoning),
        thinking_budgets,
        ..SimpleStreamOptions::default()
    };

    let stream = stream_simple(&model, &context(), Some(&options));
    let message = stream.result().await;
    (common::captured_payload(&captured), message)
}

/// `xhigh` and `max` map through the model map to the provider-native level.
#[tokio::test]
async fn maps_generative_ai_extended_levels_to_a_supported_level() {
    let map = ThinkingLevelMap::from([
        (ModelThinkingLevel::Xhigh, Some("high".to_owned())),
        (ModelThinkingLevel::Max, Some("high".to_owned())),
    ]);
    let model = thinking_map_model("gemini-3.7-flash", map);
    for reasoning in [ThinkingLevel::Xhigh, ThinkingLevel::Max] {
        let (payload, _) = capture_payload(&model, reasoning, None).await;
        assert_eq!(
            payload["config"]["thinkingConfig"]["thinkingLevel"],
            json!("HIGH"),
            "{reasoning:?}"
        );
        assert_eq!(
            payload["config"]["thinkingConfig"]["includeThoughts"],
            json!(true)
        );
    }
}

/// Uppercase provider map values resolve case-insensitively.
#[tokio::test]
async fn honors_uppercase_provider_values_for_standard_levels() {
    let model = model_with_map("gemini-3.7-flash", &[(ModelThinkingLevel::High, "LOW")]);
    let (payload, _) = capture_payload(&model, ThinkingLevel::High, None).await;

    assert_eq!(
        payload["config"]["thinkingConfig"]["thinkingLevel"],
        json!("LOW")
    );
}

/// A custom budget for the mapped level rides as `thinkingBudget`.
#[tokio::test]
async fn uses_mapped_levels_for_token_budgets() {
    let model = model_with_map("gemini-2.5-flash", &[(ModelThinkingLevel::Xhigh, "high")]);
    let (payload, _) = capture_payload(
        &model,
        ThinkingLevel::Xhigh,
        Some(ThinkingBudgets {
            high: Some(1234),
            ..ThinkingBudgets::default()
        }),
    )
    .await;

    assert_eq!(
        payload["config"]["thinkingConfig"]["thinkingBudget"],
        json!(1234)
    );
}

// --- upstream google-shared-retry.test.ts ---

/// A retryable status retries once when `maxRetries` is 1, upstream's
/// "retries a headers-less SDK error with a retryable status": the
/// headers-less normalization vanishes because the port's errors always
/// carry headers.
#[tokio::test(start_paused = true)]
async fn retries_a_retryable_error_with_a_max_retry_setting() {
    let mock = pi_ai::http::mock::MockHttpClient::new();
    mock.on(|request| request.url.contains(":streamGenerateContent"))
        .respond_sequence(vec![
            MockResponse::status(429).with_body("{}"),
            sse_data_response(200, &raw_stop_chunk("STOP", false)),
        ]);
    let model = catalog_gemini();
    let mut options = gemini_options(&mock);
    options.max_retries = Some(1);

    let message = stream_google(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    assert_eq!(mock.request_count(), 2);
}

/// Without `maxRetries` a 429 surfaces as the folded-body error after one
/// attempt.
#[tokio::test]
async fn does_not_retry_when_the_max_retry_setting_is_unset() {
    let mock = pi_ai::http::mock::MockHttpClient::new();
    mock.on(|request| request.url.contains(":streamGenerateContent"))
        .respond(MockResponse::status(429).with_body("{}"));
    let model = catalog_gemini();
    let options = gemini_options(&mock);

    let message = stream_google(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("{}"));
    assert_eq!(mock.request_count(), 1);
}

/// A 400 never retries, upstream's non-retryable status case.
#[tokio::test]
async fn does_not_retry_a_non_retryable_status() {
    let mock = pi_ai::http::mock::MockHttpClient::new();
    mock.on(|request| request.url.contains(":streamGenerateContent"))
        .respond(MockResponse::status(400).with_body("{}"));
    let model = catalog_gemini();
    let mut options = gemini_options(&mock);
    options.max_retries = Some(2);

    let message = stream_google(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Error);
    assert_eq!(mock.request_count(), 1);
}

// --- shared helpers ---

/// The dynamic-budget fallback, upstream's `getGoogleBudget` fallthrough.
#[test]
fn budgets_default_to_dynamic_outside_the_catalog_entries() {
    let model = thinking_map_model("gemini-2.0-flash", ThinkingLevelMap::new());
    assert_eq!(
        get_google_budget(&model, ResolvedGoogleThinkingLevel::Medium, None),
        -1
    );
}
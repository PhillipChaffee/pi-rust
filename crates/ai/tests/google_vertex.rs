//! The Google Vertex adapter suite, ported from
//! `packages/ai/test/google-vertex-api-key-resolution.test.ts`,
//! `packages/ai/test/google-raw-stop-reason.test.ts` (the Vertex sections),
//! and `packages/ai/test/google-thinking-level-map.test.ts` (the Vertex
//! sections) at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: upstream asserts on the mocked SDK client
//! construction; here the same decisions surface as the recorded request's
//! URL shape and headers over the transport seam, with the ADC metadata
//! server mocked so the fallback chain resolves hermetically. The capture
//! helpers assert the captured payload directly, without upstream's
//! throw-to-capture error-message check.

#![expect(
    clippy::expect_used,
    reason = "the tests pin adapter outcomes; an unexpected shape panics the test by design"
)]

use std::collections::BTreeMap;

use pi_ai::api::google_shared::{ResolvedGoogleThinkingLevel, resolve_google_thinking_level};
use pi_ai::api::google_vertex::{GoogleVertexStreamOptions, stream as stream_vertex, stream_simple};
use pi_ai::http::mock::{MockHttpClient, MockResponse};
use pi_ai::types::{
    Api, Context, Message, Model, ModelThinkingLevel, Modality, ProviderId, SimpleStreamOptions,
    ThinkingBudgets, ThinkingLevel, ThinkingLevelMap, UserContent, UserMessage,
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

/// The catalog gemini-3-flash-preview Vertex model, upstream's
/// `getModel("google-vertex", ...)`.
fn catalog_vertex() -> Model {
    common::builtin_model("google-vertex", "gemini-3-flash-preview")
}

/// The test Vertex model with a thinking map, upstream's `vertexModel`.
fn vertex_model(id: &str, thinking_level_map: ThinkingLevelMap) -> Model {
    Model {
        id: id.to_owned(),
        name: id.to_owned(),
        api: Api::from("google-vertex"),
        provider: ProviderId::from("test-vertex"),
        base_url: "https://example.invalid/v1".to_owned(),
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

/// One streamed chunk as a data-only SSE response body.
fn sse_data_response(status: u16, chunk: &str) -> MockResponse {
    MockResponse::status(status)
        .with_header("content-type", "text/event-stream")
        .with_body(format!("data: {chunk}\n\n"))
}

/// The terminal chunk the raw-stop-reason suite yields.
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
        "responseId": "vertex-response-id",
        "candidates": [candidate],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2,
        },
    })
    .to_string()
}

fn mount_vertex_stream(mock: &MockHttpClient, chunk: &str) {
    let response = sse_data_response(200, chunk);
    mock.on(|request| request.url.contains(":streamGenerateContent"))
        .respond(response);
}

/// Mount the ADC metadata-server token endpoint so the fallback chain
/// resolves hermetically even on machines with real gcloud credentials.
fn mount_metadata_server(mock: &MockHttpClient) {
    mock.on(|request| request.url.contains("169.254.169.254"))
        .respond(MockResponse::status(200).with_body(
            r#"{"access_token": "metadata-token", "expires_in": 3600}"#,
        ));
}

/// Point `GOOGLE_APPLICATION_CREDENTIALS` at a path that does not exist so
/// the ADC chain reaches the (mocked) metadata server deterministically.
fn env_without_credentials_file() -> std::collections::BTreeMap<String, String> {
    std::collections::BTreeMap::from([(
        "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
        "/nonexistent/vertex-credentials.json".to_owned(),
    )])
}

fn express_options(mock: &MockHttpClient) -> GoogleVertexStreamOptions {
    GoogleVertexStreamOptions {
        transport_options: common::mock_transport(mock),
        api_key: Some("AIzaSyExampleRealisticLookingApiKey123456".to_owned()),
        env: Some(env_without_credentials_file()),
        ..GoogleVertexStreamOptions::default()
    }
}

fn adc_options(mock: &MockHttpClient) -> GoogleVertexStreamOptions {
    let mut options = GoogleVertexStreamOptions {
        transport_options: common::mock_transport(mock),
        env: Some(env_without_credentials_file()),
        ..GoogleVertexStreamOptions::default()
    };
    options.project = Some("test-project".to_owned());
    options.location = Some("us-central1".to_owned());
    options
}

/// The stream request the mock recorded, the URL-shape assertions the
/// SDK-client-config assertions became.
async fn settle_stream(
    model: &Model,
    context: &Context,
    options: &GoogleVertexStreamOptions,
) -> pi_ai::types::AssistantMessage {
    let stream = stream_vertex(model, context, Some(options));
    stream.result().await
}

// --- upstream google-vertex-api-key-resolution.test.ts ---

/// The header of the n-th recorded request; ADC flows record the token
/// request before the stream request, so the stream is not the first.
fn recorded_header_at(mock: &MockHttpClient, index: usize, name: &str) -> Option<String> {
    mock.recorded()[index]
        .headers
        .iter()
        .find(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
}

/// A real API key rides the express path: the plain aiplatform host, the
/// `publishers/google` model path without a project prefix, and the API-key
/// header.
#[tokio::test]
async fn uses_the_api_key_client_for_real_api_keys() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let model = catalog_vertex();
    let options = express_options(&mock);

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    let request = &mock.recorded()[0];
    assert_eq!(
        request.url,
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    );
    assert_eq!(
        common::recorded_header(&mock, "x-goog-api-key").as_deref(),
        Some("AIzaSyExampleRealisticLookingApiKey123456")
    );
    assert!(recorded_header_at(&mock, 0, "authorization").is_none());
}

/// The ambient-auth placeholder falls through to ADC: the request carries
/// the project/location prefix and a bearer token, no API-key header.
#[tokio::test]
async fn falls_back_to_adc_when_the_api_key_is_a_placeholder_marker() {
    let mock = MockHttpClient::new();
    mount_metadata_server(&mock);
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let model = catalog_vertex();
    let mut options = adc_options(&mock);
    options.api_key = Some("<authenticated>".to_owned());

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    let request = &mock.recorded()[1];
    assert_eq!(
        request.url,
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    );
    assert_eq!(
        recorded_header_at(&mock, 1, "authorization").as_deref(),
        Some("Bearer metadata-token")
    );
    assert!(common::recorded_header(&mock, "x-goog-api-key").is_none());
}

/// The `gcp-vertex-credentials` marker falls through to ADC the same way.
#[tokio::test]
async fn falls_back_to_adc_when_the_api_key_is_the_credentials_marker() {
    let mock = MockHttpClient::new();
    mount_metadata_server(&mock);
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let model = catalog_vertex();
    let mut options = adc_options(&mock);
    options.api_key = Some("gcp-vertex-credentials".to_owned());

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    assert!(common::recorded_header(&mock, "x-goog-api-key").is_none());
    assert_eq!(
        recorded_header_at(&mock, 1, "authorization").as_deref(),
        Some("Bearer metadata-token")
    );
}

/// A generated catalog base URL carrying the `{location}` placeholder is
/// never forwarded: the request uses the default host shape.
#[tokio::test]
async fn does_not_forward_generated_vertex_base_url_placeholders() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let mut model = catalog_vertex();
    model.base_url = "https://{location}-aiplatform.googleapis.com/v1".to_owned();
    let options = express_options(&mock);

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    assert_eq!(
        mock.recorded()[0].url,
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    );
}

/// An explicit caller `User-Agent` overrides the default.
#[tokio::test]
async fn lets_explicit_headers_override_the_default_user_agent() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let model = catalog_vertex();
    let mut options = express_options(&mock);
    options.headers = Some(BTreeMap::from([(
        "User-Agent".to_owned(),
        Some("custom-agent".to_owned()),
    )]));

    let _ = settle_stream(&model, &context(), &options).await;

    assert_eq!(
        common::recorded_header(&mock, "User-Agent").as_deref(),
        Some("custom-agent")
    );
}

/// A custom base URL forwards verbatim with the COLLECTION scope: no project
/// prefix, and the version segment rides because the URL lacks one.
#[tokio::test]
async fn forwards_custom_base_urls_to_the_clients() {
    for build_options in [build_express, build_adc] {
        let mock = MockHttpClient::new();
        mount_metadata_server(&mock);
        mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
        let mut model = catalog_vertex();
        model.base_url = "https://proxy.example.com".to_owned();
        let options = build_options(&mock);

        let message = settle_stream(&model, &context(), &options).await;

        assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
        assert_eq!(
            mock.recorded().last().expect("request").url,
            "https://proxy.example.com/v1/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
        );
    }
}

fn build_express(mock: &MockHttpClient) -> GoogleVertexStreamOptions {
    express_options(mock)
}

fn build_adc(mock: &MockHttpClient) -> GoogleVertexStreamOptions {
    let mut options = adc_options(mock);
    options.api_key = Some("<authenticated>".to_owned());
    options
}

/// When the custom base URL already carries a version segment, no `v1` is
/// inserted.
#[tokio::test]
async fn does_not_append_api_version_when_the_base_url_includes_one() {
    let mock = MockHttpClient::new();
    mount_metadata_server(&mock);
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", false));
    let mut model = catalog_vertex();
    model.base_url = "https://proxy.example.com/v1/projects/test-project/locations/global".to_owned();
    let mut options = build_adc(&mock);

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Stop);
    assert_eq!(
        mock.recorded().last().expect("request").url,
        "https://proxy.example.com/v1/projects/test-project/locations/global/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
    );
}

/// Missing project or location fail the stream with the upstream wording.
#[tokio::test]
async fn requires_a_project_and_location_for_adc() {
    let mock = MockHttpClient::new();
    let mut options = GoogleVertexStreamOptions {
        transport_options: common::mock_transport(&mock),
        env: Some(env_without_credentials_file()),
        ..GoogleVertexStreamOptions::default()
    };
    options.api_key = Some("gcp-vertex-credentials".to_owned());
    let model = catalog_vertex();

    let message = settle_stream(&model, &context(), &options).await;
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("Vertex AI requires a project ID")),
        "got: {:?}",
        message.error_message
    );

    options.project = Some("test-project".to_owned());
    let message = settle_stream(&model, &context(), &options).await;
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("Vertex AI requires a location")),
        "got: {:?}",
        message.error_message
    );
}

// --- upstream google-raw-stop-reason.test.ts (Vertex sections) ---

/// An unmapped finish reason surfaces as an error stop with the raw wire
/// value preserved and the exact provider-stopped wording.
#[tokio::test]
async fn preserves_raw_vertex_finish_reasons_for_errors() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("SAFETY", false));
    let model = catalog_vertex();
    let mut options = express_options(&mock);
    options.project = Some("test-project".to_owned());
    options.location = Some("us-central1".to_owned());

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Error);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("SAFETY"));
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider stopped with: SAFETY")
    );
}

/// `MAX_TOKENS` with a tool call stays `length` on Vertex too.
#[tokio::test]
async fn preserves_max_tokens_with_a_tool_call_as_length_on_vertex() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("MAX_TOKENS", true));
    let model = catalog_vertex();
    let options = express_options(&mock);

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::Length);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("MAX_TOKENS"));
    assert!(message
        .content
        .iter()
        .any(|block| matches!(block, pi_ai::types::AssistantBlock::ToolCall(_))));
}

/// `STOP` with a tool call upgrades to `toolUse` on Vertex too.
#[tokio::test]
async fn maps_stop_with_a_tool_call_to_tool_use_on_vertex() {
    let mock = MockHttpClient::new();
    mount_vertex_stream(&mock, &raw_stop_chunk("STOP", true));
    let model = catalog_vertex();
    let options = express_options(&mock);

    let message = settle_stream(&model, &context(), &options).await;

    assert_eq!(message.stop_reason, pi_ai::types::StopReason::ToolUse);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("STOP"));
}

// --- upstream google-thinking-level-map.test.ts (Vertex sections) ---

async fn capture_vertex_payload(
    model: &Model,
    reasoning: ThinkingLevel,
    thinking_budgets: Option<ThinkingBudgets>,
) -> serde_json::Value {
    let mock = MockHttpClient::new();
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

    let stream = stream_simple(model, &context(), Some(&options));
    let _message = stream.result().await;
    common::captured_payload(&captured)
}

/// `xhigh` maps through the model map to the provider-native level.
#[tokio::test]
async fn maps_vertex_extended_levels() {
    let map = ThinkingLevelMap::from([(ModelThinkingLevel::Xhigh, Some("high".to_owned()))]);
    let model = vertex_model("gemini-3.7-flash", map);
    let payload = capture_vertex_payload(&model, ThinkingLevel::Xhigh, None).await;

    assert_eq!(
        payload["config"]["thinkingConfig"]["thinkingLevel"],
        json!("HIGH")
    );
    assert_eq!(
        payload["config"]["thinkingConfig"]["includeThoughts"],
        json!(true)
    );
}

/// A custom budget for the mapped level rides as `thinkingBudget`.
#[tokio::test]
async fn uses_mapped_vertex_levels_for_token_budgets() {
    let map = ThinkingLevelMap::from([(ModelThinkingLevel::Max, Some("high".to_owned()))]);
    let model = vertex_model("gemini-2.5-flash", map);
    let payload = capture_vertex_payload(
        &model,
        ThinkingLevel::Max,
        Some(ThinkingBudgets {
            high: Some(4321),
            ..ThinkingBudgets::default()
        }),
    )
    .await;

    assert_eq!(
        payload["config"]["thinkingConfig"]["thinkingBudget"],
        json!(4321)
    );
}

/// The resolver accepts the mapped value set the Vertex suite exercises.
#[test]
fn resolves_vertex_thinking_levels() {
    let map = ThinkingLevelMap::from([(ModelThinkingLevel::Xhigh, Some("high".to_owned()))]);
    let model = vertex_model("gemini-3.7-flash", map);
    assert_eq!(
        resolve_google_thinking_level(&model, ModelThinkingLevel::Xhigh)
            .expect("resolved"),
        ResolvedGoogleThinkingLevel::High
    );
}
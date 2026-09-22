//! The Mistral HTTP transport suite, ported from
//! `packages/ai/test/mistral-http-transport.test.ts`,
//! `packages/ai/test/mistral-reasoning-mode.test.ts`,
//! `packages/ai/test/mistral-raw-stop-reason.test.ts`, and
//! `packages/ai/test/mistral-tool-schema.test.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: the custom-fetch seam is the transport mock; the
//! symbol-key stripping assertion is statically upheld (serde JSON values
//! carry no symbol keys); the never-producing body becomes a paused-clock
//! handler that the request timeout outruns.

#![expect(
    clippy::expect_used,
    reason = "the tests pin transport outcomes; an unexpected shape panics the test by design"
)]

use std::sync::Arc;

use bytes::Bytes;
use pi_ai::api::mistral_conversations::{
    MistralPromptMode, MistralReasoningEffort, MistralStreamOptions, MistralToolChoice,
    stream as stream_mistral, stream_simple,
};
use pi_ai::http::mock::{MockHttpClient, MockResponse};
use pi_ai::types::{
    Api, AssistantBlock, ConstrainedSamplingConfig, Context, Message, Modality, Model, OnPayload,
    ProviderId, SimpleStreamOptions, StopReason, Strictness, TextContent, ThinkingContent,
    ThinkingLevel, Tool, ToolCall, UserBlock, UserContent, UserMessage,
};
use serde_json::{Value, json};
use std::fmt::Write as _;

mod common;

/// The context the transport tests run, upstream's system+user+image shape.
fn transport_context() -> Context {
    Context {
        system_prompt: Some("Be precise".to_owned()),
        messages: vec![Message::User(UserMessage {
            content: UserContent::Blocks(vec![
                UserBlock::Text(common::text_block("describe")),
                UserBlock::Image(pi_ai::types::ImageContent {
                    data: "aGVsbG8=".to_owned(),
                    mime_type: "image/png".to_owned(),
                }),
            ]),
            timestamp: 1,
        })],
        tools: Some(vec![Tool {
            name: "lookup".to_owned(),
            description: "lookup".to_owned(),
            parameters: json!({ "type": "object", "properties": { "query": { "type": "string" } } }),
            constrained_sampling: None,
        }]),
    }
}

/// One terminal SSE frame with usage, upstream's `createTerminalEvent`.
fn terminal_chunk(finish_reason: &str) -> String {
    json!({
        "id": "mistral-response-id",
        "model": "mistral-large-latest",
        "choices": [{ "index": 0, "finish_reason": finish_reason, "delta": {} }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
    })
    .to_string()
}

/// Mount the SSE frames plus the `[DONE]` terminator, upstream's
/// `createSseResponse`.
fn mount_mistral_stream(mock: &MockHttpClient, events: &[String]) {
    let mut body = String::new();
    for event in events {
        let _ = write!(body, "data: {event}\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    mock.on(|request| request.url.contains("/v1/chat/completions"))
        .respond(
            MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_header("x-request-id", "request-1")
                .with_body(body),
        );
}

fn transport_options(mock: &MockHttpClient) -> MistralStreamOptions {
    MistralStreamOptions {
        transport_options: common::mock_transport(mock),
        api_key: Some("secret".to_owned()),
        ..MistralStreamOptions::default()
    }
}

/// A tool choice forced to the lookup function, upstream's object form.
fn forced_lookup() -> MistralToolChoice {
    MistralToolChoice::Function {
        name: "lookup".to_owned(),
    }
}

// --- upstream mistral-http-transport.test.ts ---

/// The SDK-style payload serializes to Mistral's `snake_case` wire format with
/// the affinity and override headers, upstream's first transport test.
#[expect(
    clippy::too_many_lines,
    reason = "the transport assertion walks every remapped field in one stream"
)]
#[tokio::test(start_paused = true)]
async fn serializes_sdk_style_payloads_to_the_mistral_wire_format() {
    let mock = MockHttpClient::new();
    mount_mistral_stream(&mock, &[terminal_chunk("stop")]);
    let model = common::builtin_model("mistral", "mistral-large-latest");

    // The hook records the camelCase payload and enriches it with the
    // fields the wire conversion must remap, upstream's onPayload fixture.
    let (on_payload, captured) = capture_and_extend();
    let mut options = transport_options_with(&mock, "secret", Some(on_payload));
    options.headers = Some(std::collections::BTreeMap::from([(
        "x-custom".to_owned(),
        Some("value".to_owned()),
    )]));
    options.max_tokens = Some(123);
    options.prompt_mode = Some(MistralPromptMode::Reasoning);
    options.reasoning_effort = Some(MistralReasoningEffort::High);
    options.tool_choice = Some(forced_lookup());
    options.session_id = Some("session-1".to_owned());
    let responses: Arc<std::sync::Mutex<Vec<pi_ai::types::ProviderResponse>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let slot = Arc::clone(&responses);
    options.transport_options.on_response =
        Some(pi_ai::types::OnResponse::new(move |response, _model| {
            slot.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(response);
            Box::pin(async {})
        }));

    let message = stream_mistral(&model, &transport_context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    let request = &mock.recorded()[0];
    assert_eq!(request.url, "https://api.mistral.ai/v1/chat/completions");
    assert_eq!(
        common::recorded_header(&mock, "authorization").as_deref(),
        Some("Bearer secret")
    );
    assert_eq!(
        common::recorded_header(&mock, "accept").as_deref(),
        Some("text/event-stream")
    );
    assert_eq!(
        common::recorded_header(&mock, "x-affinity").as_deref(),
        Some("session-1")
    );
    assert_eq!(
        common::recorded_header(&mock, "x-custom").as_deref(),
        Some("value")
    );
    assert_eq!(
        common::recorded_header(&mock, "User-Agent").as_deref(),
        Some(pi_ai::utils::pi_user_agent::get_pi_user_agent().as_str())
    );

    let payload = common::captured_payload(&captured);
    assert_eq!(payload["maxTokens"], json!(123));
    assert_eq!(payload["promptMode"], json!("reasoning"));
    assert_eq!(payload["promptCacheKey"], json!("session-1"));

    let observed = responses
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(observed.len(), 1);
    assert_eq!(observed[0].status, 200);
    assert_eq!(
        observed[0].headers.get("x-request-id").map(String::as_str),
        Some("request-1")
    );

    let body = common::recorded_body(&mock);
    assert_eq!(body["max_tokens"], json!(123));
    assert_eq!(body["prompt_mode"], json!("reasoning"));
    assert_eq!(body["reasoning_effort"], json!("high"));
    assert_eq!(
        body["tool_choice"],
        json!({ "type": "function", "function": { "name": "lookup" } })
    );
    assert_eq!(body["prompt_cache_key"], json!("session-1"));
    assert_eq!(body["top_p"], json!(0.9));
    assert_eq!(body["random_seed"], json!(42));
    assert_eq!(body["presence_penalty"], json!(0.1));
    assert_eq!(body["frequency_penalty"], json!(0.2));
    assert_eq!(body["parallel_tool_calls"], json!(true));
    assert_eq!(body["safe_prompt"], json!(true));
    assert_eq!(
        body["response_format"],
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "schema": {
                    "type": "object",
                    "properties": { "maxTokens": { "type": "number" } },
                },
            }
        })
    );
    assert_eq!(
        body["messages"],
        json!([
            { "role": "system", "content": "Be precise" },
            { "role": "user", "content": [
                { "type": "text", "text": "describe" },
                { "type": "image_url", "image_url": "data:image/png;base64,aGVsbG8=" },
            ] },
        ])
    );
}

fn transport_options_with(
    mock: &MockHttpClient,
    api_key: &str,
    on_payload: Option<OnPayload>,
) -> MistralStreamOptions {
    MistralStreamOptions {
        transport_options: pi_ai::types::TransportOptions {
            http_client: Some(Arc::new(mock.clone())),
            on_payload,
            ..pi_ai::types::TransportOptions::default()
        },
        api_key: Some(api_key.to_owned()),
        ..MistralStreamOptions::default()
    }
}

/// Record and enrich the payload, returning it as the replacement so the
/// extra camelCase fields ride through the wire conversion, upstream's
/// onPayload fixture shape.
fn capture_and_extend() -> (OnPayload, common::CapturedPayload) {
    let captured = Arc::new(std::sync::Mutex::new(None));
    let slot = Arc::clone(&captured);
    let hook = OnPayload::new(move |mut payload, _model| {
        if let Some(object) = payload.as_object_mut() {
            object.insert("topP".to_owned(), json!(0.9));
            object.insert("randomSeed".to_owned(), json!(42));
            object.insert(
                "responseFormat".to_owned(),
                json!({
                    "type": "json_schema",
                    "jsonSchema": {
                        "name": "result",
                        "schemaDefinition": {
                            "type": "object",
                            "properties": { "maxTokens": { "type": "number" } },
                        },
                    },
                }),
            );
            object.insert("presencePenalty".to_owned(), json!(0.1));
            object.insert("frequencyPenalty".to_owned(), json!(0.2));
            object.insert("parallelToolCalls".to_owned(), json!(true));
            object.insert("safePrompt".to_owned(), json!(true));
        }
        *slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(payload.clone());
        Box::pin(async move { Some(payload) })
    });
    (hook, captured)
}

// --- upstream mistral-http-transport.test.ts, replay serialization ---

/// Assistant thinking, tool calls, and tool results replay in Mistral's
/// wire shapes with `prefix: false` always present.
#[tokio::test]
async fn serializes_assistant_thinking_tool_calls_and_tool_results_for_replay() {
    let mock = MockHttpClient::new();
    mount_mistral_stream(&mock, &[terminal_chunk("stop")]);
    let model = common::builtin_model("mistral", "mistral-large-latest");

    let mut arguments = serde_json::Map::new();
    arguments.insert("query".to_owned(), json!("pi"));
    let mut assistant = common::assistant_message_with_content(
        "mistral-conversations",
        "mistral",
        &model.id,
        vec![
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "reason".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantBlock::Text(common::text_block("answer")),
            AssistantBlock::ToolCall(ToolCall {
                id: "abc123456".to_owned(),
                name: "lookup".to_owned(),
                arguments,
                thought_signature: None,
                namespace: None,
            }),
        ],
    );
    assistant.provider = ProviderId::from("mistral");
    let context = Context {
        system_prompt: None,
        messages: vec![
            Message::Assistant(assistant),
            Message::ToolResult(pi_ai::types::ToolResultMessage {
                tool_call_id: "abc123456".to_owned(),
                tool_name: "lookup".to_owned(),
                content: vec![
                    pi_ai::types::ToolResultBlock::Text(common::text_block("found")),
                    pi_ai::types::ToolResultBlock::Image(pi_ai::types::ImageContent {
                        data: "aGVsbG8=".to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: false,
                timestamp: 1,
            }),
        ],
        tools: None,
    };
    let options = transport_options(&mock);

    let message = stream_mistral(&model, &context, Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    let body = common::recorded_body(&mock);
    assert_eq!(
        body["messages"],
        json!([
            {
                "role": "assistant",
                "prefix": false,
                "content": [
                    { "type": "thinking", "thinking": [{ "type": "text", "text": "reason" }] },
                    { "type": "text", "text": "answer" },
                ],
                "tool_calls": [
                    {
                        "id": "abc123456",
                        "type": "function",
                        "function": { "name": "lookup", "arguments": "{\"query\":\"pi\"}" },
                        "index": 0,
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "abc123456",
                "name": "lookup",
                "content": [
                    { "type": "text", "text": "found" },
                    { "type": "image_url", "image_url": "data:image/png;base64,aGVsbG8=" },
                ],
            },
        ])
    );
}

// --- upstream mistral-http-transport.test.ts, stream parsing ---

/// The four-frame stream: thinking, text, fragmented tool call, then the
/// terminal event carrying the finish reason and cached-token usage.
#[tokio::test]
async fn parses_native_thinking_text_fragmented_tool_calls_and_cached_token_usage() {
    let mock = MockHttpClient::new();
    let events = [
        json!({
            "id": "response-1",
            "choices": [{ "index": 0, "finish_reason": Value::Null, "delta": {
                "content": [{ "type": "thinking", "thinking": [{ "type": "text", "text": "reason" }] }],
            }}],
        })
        .to_string(),
        json!({
            "id": "response-1",
            "choices": [{ "index": 0, "finish_reason": Value::Null, "delta": {
                "content": [{ "type": "text", "text": "answer" }],
            }}],
        })
        .to_string(),
        json!({
            "id": "response-1",
            "choices": [{ "index": 0, "finish_reason": Value::Null, "delta": {
                "tool_calls": [{ "id": "abc123456", "index": 0, "function": {
                    "name": "lookup", "arguments": "{\"query\":",
                }}],
            }}],
        })
        .to_string(),
        json!({
            "choices": [{ "index": 0, "finish_reason": "tool_calls", "delta": {
                "tool_calls": [{ "index": 0, "function": { "name": "", "arguments": "\"pi\"}" } }],
            }}],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
                "prompt_tokens_details": { "cached_tokens": 3 },
            },
        })
        .to_string(),
    ];
    mount_mistral_stream(&mock, &events);
    let model = common::builtin_model("mistral", "mistral-large-latest");
    let options = transport_options(&mock);

    let message = stream_mistral(&model, &transport_context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::ToolUse);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("tool_calls"));
    assert_eq!(message.response_id.as_deref(), Some("response-1"));
    assert_eq!(
        message.content,
        vec![
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "reason".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantBlock::Text(TextContent {
                text: "answer".to_owned(),
                text_signature: None,
            }),
            AssistantBlock::ToolCall(ToolCall {
                id: "abc123456".to_owned(),
                name: "lookup".to_owned(),
                arguments: {
                    let mut args = serde_json::Map::new();
                    args.insert("query".to_owned(), json!("pi"));
                    args
                },
                thought_signature: None,
                namespace: None,
            }),
        ]
    );
    assert_eq!(message.usage.input, 7);
    assert_eq!(message.usage.output, 4);
    assert_eq!(message.usage.cache_read, 3);
    assert_eq!(message.usage.total_tokens, 14);
}

/// Byte-at-a-time SSE delivery decodes multi-byte UTF-8 split across
/// transport chunks, upstream's bytewise fixture.
#[tokio::test]
async fn parses_sse_and_utf8_sequences_split_across_transport_chunks() {
    let mock = MockHttpClient::new();
    let event = json!({
        "id": "response-utf8",
        "choices": [{ "index": 0, "finish_reason": Value::Null, "delta": {
            "content": "héllo 🌍",
        }}],
    })
    .to_string();
    let body = format!("data: {event}\n\ndata: [DONE]\n\n");
    let chunks: Vec<Bytes> = body
        .as_bytes()
        .iter()
        .map(|byte| Bytes::from(vec![*byte]))
        .collect();
    mock.on(|request| request.url.contains("/v1/chat/completions"))
        .respond(
            MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_chunks(chunks),
        );
    let model = common::builtin_model("mistral", "mistral-large-latest");
    let options = transport_options(&mock);

    let message = stream_mistral(&model, &transport_context(), Some(&options))
        .result()
        .await;

    let Some(AssistantBlock::Text(block)) = message.content.first() else {
        unreachable!("expected a text block");
    };
    assert_eq!(block.text, "héllo 🌍");
}

/// Case-insensitive header overrides delete defaults, and an explicit
/// (even null) affinity override suppresses the automatic one.
#[tokio::test]
async fn honors_case_insensitive_header_overrides_and_affinity_suppression() {
    let mock = MockHttpClient::new();
    mount_mistral_stream(&mock, &[terminal_chunk("stop")]);
    let mut model = common::builtin_model("mistral", "mistral-large-latest");
    model.headers = Some(std::collections::BTreeMap::from([
        ("Authorization".to_owned(), "Bearer model-key".to_owned()),
        ("X-Affinity".to_owned(), "model-affinity".to_owned()),
    ]));
    let mut options = transport_options_with(&mock, "secret", None);
    options.headers = Some(std::collections::BTreeMap::from([
        ("authorization".to_owned(), None),
        ("x-affinity".to_owned(), None),
        ("User-Agent".to_owned(), Some("custom-agent".to_owned())),
    ]));
    options.session_id = Some("automatic-affinity".to_owned());

    let message = stream_mistral(&model, &transport_context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert!(common::recorded_header(&mock, "authorization").is_none());
    assert!(common::recorded_header(&mock, "x-affinity").is_none());
    assert_eq!(
        common::recorded_header(&mock, "User-Agent").as_deref(),
        Some("custom-agent")
    );
}

/// An already-cancelled signal aborts the request, upstream's immediate
/// abort; the wording mirrors the pre-dispatch throw.
#[tokio::test]
async fn aborted_requests_set_the_aborted_stop_reason() {
    let mock = MockHttpClient::new();
    mount_mistral_stream(&mock, &[terminal_chunk("stop")]);
    let token = tokio_util::sync::CancellationToken::new();
    token.cancel();
    let model = common::builtin_model("mistral", "mistral-large-latest");
    let mut options = transport_options(&mock);
    options.transport_options.signal = Some(token);

    let message = stream_mistral(&model, &transport_context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Aborted);
}

/// The request timeout surfaces as an error carrying the timeout wording,
/// upstream's `AbortSignal.timeout` fixture.
#[tokio::test(start_paused = true)]
async fn applies_the_request_timeout_while_waiting_for_a_chunk() {
    use std::time::Duration;

    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/v1/chat/completions"))
        .respond_fn(move |_request| async move {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Ok(MockResponse::status(200))
        });
    let model = common::builtin_model("mistral", "mistral-large-latest");
    let mut options = transport_options(&mock);
    options.timeout_ms = Some(5);

    let message = stream_mistral(&model, &transport_context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("timeout")),
        "got: {:?}",
        message.error_message
    );
}

/// HTTP failures carry the status and the verbatim body, upstream's
/// gateway error fixture.
#[tokio::test]
async fn preserves_http_status_and_response_bodies_in_errors() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/v1/chat/completions"))
        .respond(MockResponse::status(403).with_body(r#"{"message":"blocked by gateway"}"#));
    let model = common::builtin_model("mistral", "mistral-large-latest");
    let options = transport_options(&mock);

    let message = stream_mistral(&model, &transport_context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some(r#"Mistral API error (403): {"message":"blocked by gateway"}"#)
    );
}

// --- upstream mistral-reasoning-mode.test.ts ---

/// The reasoning-mode model fixture, upstream's hand-built models with the
/// guaranteed-refusal base URL.
fn reasoning_model(id: &str, reasoning: bool) -> Model {
    Model {
        id: id.to_owned(),
        name: id.to_owned(),
        api: Api::from("mistral-conversations"),
        provider: ProviderId::from("test-mistral"),
        base_url: "http://127.0.0.1:9".to_owned(),
        reasoning,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 128_000,
        max_tokens: 16_384,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// Capture the payload through the hook and settle the stream; the refusal
/// base URL fails the request after capture.
async fn capture_reasoning_payload(
    model: &Model,
    reasoning: Option<ThinkingLevel>,
) -> (Value, pi_ai::types::AssistantMessage) {
    let mock = MockHttpClient::new();
    let (on_payload, captured) = common::payload_capture();
    let mut transport = common::mock_transport(&mock);
    transport.on_payload = Some(on_payload);
    let options = SimpleStreamOptions {
        transport_options: transport,
        api_key: Some("fake-key".to_owned()),
        reasoning,
        ..SimpleStreamOptions::default()
    };
    let stream = stream_simple(model, &reasoning_context(), Some(&options));
    let message = stream.result().await;
    (common::captured_payload(&captured), message)
}

fn reasoning_context() -> Context {
    Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("hi".to_owned()),
            timestamp: 1,
        })],
        tools: None,
    }
}

#[tokio::test]
async fn uses_reasoning_effort_for_mistral_small_4() {
    let model = reasoning_model("mistral-small-2603", true);
    let (payload, _) = capture_reasoning_payload(&model, Some(ThinkingLevel::Medium)).await;
    assert_eq!(payload["reasoningEffort"], json!("high"));
    assert!(payload.get("promptMode").is_none());
}

#[tokio::test]
async fn omits_reasoning_controls_when_thinking_is_off() {
    let model = reasoning_model("mistral-small-2603", true);
    let (payload, _) = capture_reasoning_payload(&model, None).await;
    assert!(payload.get("reasoningEffort").is_none());
    assert!(payload.get("promptMode").is_none());
}

#[tokio::test]
async fn uses_prompt_mode_for_magistral_reasoning_models() {
    let model = reasoning_model("magistral-medium-latest", true);
    let (payload, _) = capture_reasoning_payload(&model, Some(ThinkingLevel::Medium)).await;
    assert_eq!(payload["promptMode"], json!("reasoning"));
    assert!(payload.get("reasoningEffort").is_none());
}

/// The GLM-5.2 regression: Mistral-hosted GLM ignores `prompt_mode`, so it
/// rides the `reasoning_effort` control.
#[tokio::test]
async fn uses_reasoning_effort_for_zai_glm() {
    let model = reasoning_model("zai-glm-5-2", true);
    let (payload, _) = capture_reasoning_payload(&model, Some(ThinkingLevel::High)).await;
    assert_eq!(payload["reasoningEffort"], json!("high"));
    assert!(payload.get("promptMode").is_none());
}

#[tokio::test]
async fn uses_reasoning_effort_for_mistral_medium_aliases() {
    for id in ["mistral-medium-2604", "mistral-medium-latest"] {
        let model = reasoning_model(id, true);
        let (payload, _) = capture_reasoning_payload(&model, Some(ThinkingLevel::High)).await;
        assert_eq!(payload["reasoningEffort"], json!("high"), "{id}");
        assert!(payload.get("promptMode").is_none(), "{id}");
    }
}

/// A non-reasoning Medium model omits both controls despite the prefix
/// match, upstream's 2505 fixture.
#[tokio::test]
async fn omits_reasoning_controls_for_non_reasoning_medium_models() {
    let model = reasoning_model("mistral-medium-2505", false);
    let (payload, _) = capture_reasoning_payload(&model, Some(ThinkingLevel::Medium)).await;
    assert!(payload.get("reasoningEffort").is_none());
    assert!(payload.get("promptMode").is_none());
}

#[tokio::test]
async fn uses_the_session_id_as_prompt_cache_key() {
    let mock = MockHttpClient::new();
    let (on_payload, captured) = common::payload_capture();
    let mut transport = common::mock_transport(&mock);
    transport.on_payload = Some(on_payload);
    let options = SimpleStreamOptions {
        transport_options: transport,
        api_key: Some("fake-key".to_owned()),
        session_id: Some("session-123".to_owned()),
        ..SimpleStreamOptions::default()
    };
    let model = reasoning_model("devstral-medium-latest", false);
    let stream = stream_simple(&model, &reasoning_context(), Some(&options));
    let _message = stream.result().await;

    let payload = common::captured_payload(&captured);
    assert_eq!(payload["promptCacheKey"], json!("session-123"));
}

#[tokio::test]
async fn omits_the_prompt_cache_key_when_cache_retention_is_disabled() {
    let mock = MockHttpClient::new();
    let (on_payload, captured) = common::payload_capture();
    let mut transport = common::mock_transport(&mock);
    transport.on_payload = Some(on_payload);
    let options = SimpleStreamOptions {
        transport_options: transport,
        api_key: Some("fake-key".to_owned()),
        session_id: Some("session-123".to_owned()),
        cache_retention: Some(pi_ai::types::CacheRetention::None),
        ..SimpleStreamOptions::default()
    };
    let model = reasoning_model("devstral-medium-latest", false);
    let stream = stream_simple(&model, &reasoning_context(), Some(&options));
    let _message = stream.result().await;

    let payload = common::captured_payload(&captured);
    assert!(payload.get("promptCacheKey").is_none());
}

// --- upstream mistral-raw-stop-reason.test.ts ---

fn stop_reason_chunk(finish_reason: &str) -> String {
    json!({
        "choices": [{ "index": 0, "finish_reason": finish_reason, "delta": {} }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1 },
    })
    .to_string()
}

/// Successful stops preserve the raw finish reason with no error message.
#[tokio::test]
async fn preserves_raw_mistral_finish_reasons_for_successful_stops() {
    let mock = MockHttpClient::new();
    mount_mistral_stream(&mock, &[stop_reason_chunk("stop")]);
    let model = common::builtin_model("mistral", "devstral-medium-latest");
    let options = transport_options(&mock);

    let message = stream_mistral(&model, &reasoning_context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("stop"));
    assert_eq!(message.error_message, None);
}

/// Provider error stops carry the exact provider-stopped wording.
#[tokio::test]
async fn preserves_raw_mistral_finish_reasons_for_provider_error_stops() {
    let mock = MockHttpClient::new();
    mount_mistral_stream(&mock, &[stop_reason_chunk("error")]);
    let model = common::builtin_model("mistral", "devstral-medium-latest");
    let options = transport_options(&mock);

    let message = stream_mistral(&model, &reasoning_context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("error"));
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider stopped with: error")
    );
}

/// Unknown finish reasons are provider errors carrying the raw value.
#[tokio::test]
async fn treats_unknown_mistral_finish_reasons_as_provider_error_stops() {
    let mock = MockHttpClient::new();
    mount_mistral_stream(&mock, &[stop_reason_chunk("unmapped_error")]);
    let model = common::builtin_model("mistral", "devstral-medium-latest");
    let options = transport_options(&mock);

    let message = stream_mistral(&model, &reasoning_context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("unmapped_error"));
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider stopped with: unmapped_error")
    );
}

// --- upstream mistral-tool-schema.test.ts ---

/// The strict tool schema serializes with `strict: true` and the fully
/// strict-converted parameter document; the symbol-key stripping the test
/// pins is statically upheld in Rust.
#[tokio::test]
async fn strict_tool_schemas_serialize_with_the_strict_flag() {
    let mock = MockHttpClient::new();
    let (on_payload, captured) = common::payload_capture();
    let mut transport = common::mock_transport(&mock);
    transport.on_payload = Some(on_payload);
    let options = SimpleStreamOptions {
        transport_options: transport,
        api_key: Some("fake-key".to_owned()),
        ..SimpleStreamOptions::default()
    };
    let mut model = reasoning_model("devstral-medium-latest", false);
    model.base_url = "http://127.0.0.1:9".to_owned();
    let mut context = reasoning_context();
    context.tools = Some(vec![Tool {
        name: "inspect_schema".to_owned(),
        description: "inspect".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                },
            },
        }),
        constrained_sampling: Some(pi_ai::types::ConstrainedSamplingSetting::Config(
            ConstrainedSamplingConfig::JsonSchema {
                strict: Strictness::Require,
            },
        )),
    }]);

    let stream = stream_simple(&model, &context, Some(&options));
    let message = stream.result().await;

    let payload = common::captured_payload(&captured);
    let tools = payload["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["strict"], json!(true));
    assert_eq!(
        tools[0]["function"]["parameters"],
        json!({
            "type": "object",
            "properties": {
                "nested": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": { "value": { "anyOf": [
                                { "type": "string" },
                                { "type": "null" },
                            ]}},
                            "required": ["value"],
                            "additionalProperties": false,
                        },
                        { "type": "null" },
                    ],
                },
            },
            "required": ["nested"],
            "additionalProperties": false,
        })
    );
    // The schema passed the provider-side validation shape: the failure is
    // the refusal port, never a schema rejection.
    assert!(
        !message
            .error_message
            .as_deref()
            .unwrap_or_default()
            .contains("Input validation failed")
    );
}

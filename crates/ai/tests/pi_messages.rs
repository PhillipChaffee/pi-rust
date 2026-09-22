//! The pi-messages suite, ported from `packages/ai/test/pi-messages.test.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: the loopback `node:http` responder becomes
//! `MockHttpClient` routes; the registration assertions ("registered as a
//! builtin api provider", "known api usable on models") ride with the
//! provider-registry suite, which already walks every constructor and the
//! `KnownApi` vocabulary. The `debug` flag rides the adapter-local options,
//! so the debug test uses the full-fidelity `stream` entry.

#![expect(
    clippy::expect_used,
    reason = "the tests pin adapter outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::api::pi_messages::{
    PiMessagesStreamOptions, PiMessagesToolChoice, stream as stream_pi_messages,
};
use pi_ai::http::mock::{MockHttpClient, MockResponse};
use pi_ai::types::{
    Api, AssistantBlock, Context, Message, Modality, Model, ProviderId, StopReason, TextContent,
    ToolCall, UserContent, UserMessage,
};
use serde_json::{Value, json};
use std::fmt::Write as _;

mod common;

/// The catalog-driven model fixture, upstream's `createModel`.
fn pi_messages_model(base_url: &str) -> Model {
    Model {
        id: "auto".to_owned(),
        name: "Radius Auto".to_owned(),
        api: Api::from("pi-messages"),
        provider: ProviderId::from("radius"),
        base_url: base_url.to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost {
            rates: pi_ai::types::ModelCostRates {
                input: 1.0,
                output: 2.0,
                cache_read: 0.1,
                cache_write: 0.2,
            },
            tiers: None,
        },
        context_window: 128_000,
        max_tokens: 16_384,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn context() -> Context {
    Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("Hello".to_owned()),
            timestamp: 1,
        })],
        tools: None,
    }
}

/// The usage fixture the wire returns verbatim, upstream's `usage`.
fn wire_usage() -> Value {
    json!({
        "input": 10,
        "output": 5,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 15,
        "cost": { "input": 0.1, "output": 0.2, "cacheRead": 0, "cacheWrite": 0, "total": 0.3 },
    })
}

/// Mount the SSE frames, upstream's `startServer` 200 path: each event one
/// `data:` frame, plus the `[DONE]` terminator the parser skips.
fn mount_events(mock: &MockHttpClient, events: &[Value]) {
    let mut body = String::new();
    for event in events {
        let _ = write!(body, "data: {event}\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    mock.on(|request| request.url.contains("/messages"))
        .respond(
            MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_body(body),
        );
}

fn options(mock: &MockHttpClient, api_key: &str) -> PiMessagesStreamOptions {
    PiMessagesStreamOptions {
        transport_options: common::mock_transport(mock),
        api_key: Some(api_key.to_owned()),
        ..PiMessagesStreamOptions::default()
    }
}

/// The terminal `done` frame with the usage fixture and optional extras.
fn done_frame(reason: &str, extra: &Value) -> Value {
    let mut frame = json!({
        "type": "done",
        "reason": reason,
        "usage": wire_usage(),
    });
    if let Some(object) = extra.as_object() {
        for (key, value) in object {
            frame[key] = value.clone();
        }
    }
    frame
}

/// The full generation stream: text block, tool call, terminal done,
/// upstream's first test fixture.
fn text_and_toolcall_events() -> Vec<Value> {
    vec![
        json!({ "type": "start" }),
        json!({ "type": "text_start", "contentIndex": 0 }),
        json!({ "type": "text_delta", "contentIndex": 0, "delta": "Hel" }),
        json!({ "type": "text_delta", "contentIndex": 0, "delta": "Hello" }),
        json!({ "type": "text_end", "contentIndex": 0, "content": "Hello" }),
        json!({ "type": "toolcall_start", "contentIndex": 1, "id": "call_1", "toolName": "read" }),
        json!({ "type": "toolcall_delta", "contentIndex": 1, "delta": "{\"path\":" }),
        json!({ "type": "toolcall_delta", "contentIndex": 1, "delta": "\"a.txt\"}" }),
        json!({
            "type": "toolcall_end",
            "contentIndex": 1,
            "toolCall": { "type": "toolCall", "id": "call_1", "name": "read", "arguments": { "path": "a.txt" } },
        }),
        json!({
            "type": "done",
            "reason": "toolUse",
            "usage": wire_usage(),
            "responseId": "resp_1",
            "providerThinkingLevel": "high",
        }),
    ]
}

// --- upstream pi-messages.test.ts, first test ---

/// Text and tool calls stream, the terminal message resolves with the
/// wire-verbatim usage, and the request carries the serialized context.
#[tokio::test]
async fn streams_text_and_tool_calls_and_resolves_the_terminal_message() {
    let mock = MockHttpClient::new();
    mount_events(&mock, &text_and_toolcall_events());
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let mut options = options(&mock, "test-key");
    options.session_id = Some("session-1".to_owned());
    options.tool_choice = Some(PiMessagesToolChoice::Auto);
    options.max_tokens = Some(100);
    options.headers = Some(std::collections::BTreeMap::from([(
        "x-custom".to_owned(),
        Some("1".to_owned()),
    )]));

    let message = stream_pi_messages(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::ToolUse);
    assert_eq!(message.response_id.as_deref(), Some("resp_1"));
    assert_eq!(message.provider_thinking_level.as_deref(), Some("high"));
    assert_eq!(message.model, "auto");
    assert_eq!(message.provider, ProviderId::from("radius"));
    assert_eq!(
        message.content,
        vec![
            AssistantBlock::Text(TextContent {
                text: "Hello".to_owned(),
                text_signature: None,
            }),
            AssistantBlock::ToolCall(ToolCall {
                id: "call_1".to_owned(),
                name: "read".to_owned(),
                arguments: {
                    let mut args = serde_json::Map::new();
                    args.insert("path".to_owned(), json!("a.txt"));
                    args
                },
                thought_signature: None,
                namespace: None,
            }),
        ]
    );
    assert_eq!(message.usage.input, 10);
    assert_eq!(message.usage.output, 5);
    assert_eq!(message.usage.total_tokens, 15);
    assert!(
        (message.usage.cost.total - 0.3).abs() < 1e-9,
        "cost total: {}",
        message.usage.cost.total
    );

    let request = &mock.recorded()[0];
    assert!(request.url.ends_with("/v1/messages"));
    assert_eq!(
        common::recorded_header(&mock, "authorization").as_deref(),
        Some("Bearer test-key")
    );
    assert_eq!(
        common::recorded_header(&mock, "x-custom").as_deref(),
        Some("1")
    );
    let body = common::recorded_body(&mock);
    assert_eq!(body["model"], json!("auto"));
    assert_eq!(
        body["context"],
        serde_json::to_value(context()).expect("context")
    );
    assert_eq!(body["options"]["maxTokens"], json!(100));
    assert_eq!(body["options"]["sessionId"], json!("session-1"));
    assert_eq!(body["options"]["toolChoice"], json!("auto"));
    assert!(body["options"].get("temperature").is_none());
    assert!(body["options"].get("cacheRetention").is_none());
}

// --- upstream pi-messages.test.ts, debug mode ---

/// `debug: true` appends the query parameter and the response hook sees the
/// gateway's routing headers, upstream's second test.
#[tokio::test]
async fn appends_debug_and_reports_response_headers_via_on_response() {
    let mock = MockHttpClient::new();
    let responses: std::sync::Arc<std::sync::Mutex<Vec<pi_ai::types::ProviderResponse>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    mount_events(&mock, &[done_frame("stop", &Value::Null)]);
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let mut options = options(&mock, "test-key");
    options.debug = true;
    let slot = std::sync::Arc::clone(&responses);
    options.transport_options.on_response =
        Some(pi_ai::types::OnResponse::new(move |response, _model| {
            slot.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(response);
            Box::pin(async {})
        }));

    let message = stream_pi_messages(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert!(mock.recorded()[0].url.contains("/messages?debug=1"));
    let observed = responses
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(observed.len(), 1);
    assert_eq!(observed[0].status, 200);
}

// --- upstream pi-messages.test.ts, error surfaces ---

/// Non-OK responses surface with the backend message, code, and the
/// response-failure diagnostic.
#[tokio::test]
async fn surfaces_backend_error_responses_with_diagnostics() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/messages"))
        .respond(
            MockResponse::status(401)
                .with_body(r#"{"error":{"message":"Token expired","code":"unauthorized"}}"#),
        );
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let options = options(&mock, "test-key");

    let message = stream_pi_messages(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    let error_message = message.error_message.as_deref().unwrap_or_default();
    assert!(error_message.contains("401"), "{error_message}");
    assert!(error_message.contains("Token expired"), "{error_message}");
    assert!(error_message.contains("unauthorized"), "{error_message}");
    let diagnostics = message.diagnostics.as_ref().expect("diagnostics");
    assert_eq!(diagnostics[0].kind, "pi_messages_response_failure");
    assert_eq!(
        diagnostics[0]
            .details
            .as_ref()
            .and_then(|details| details.get("status"))
            .cloned(),
        Some(json!(401))
    );
}

/// Server-sent error events settle the stream with the failing message.
#[tokio::test]
async fn propagates_server_sent_error_events() {
    let mock = MockHttpClient::new();
    let frame = json!({
        "type": "error",
        "reason": "error",
        "usage": wire_usage(),
        "errorMessage": "Upstream failed",
    });
    mount_events(&mock, &[frame]);
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let options = options(&mock, "test-key");

    let message = stream_pi_messages(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("Upstream failed"));
    assert_eq!(message.usage.input, 10);
    assert_eq!(message.usage.total_tokens, 15);
}

/// A missing key settles the stream with the no-key wording.
#[tokio::test]
async fn errors_when_no_api_key_is_provided() {
    let mock = MockHttpClient::new();
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let options = PiMessagesStreamOptions {
        transport_options: common::mock_transport(&mock),
        ..PiMessagesStreamOptions::default()
    };

    let message = stream_pi_messages(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("No API key provided")),
        "{:?}",
        message.error_message
    );
}

/// A stream that ends without a terminal wire event settles with the
/// trailing-event wording.
#[tokio::test]
async fn errors_when_the_stream_ends_without_a_terminal_event() {
    let mock = MockHttpClient::new();
    mount_events(
        &mock,
        &[
            json!({ "type": "start" }),
            json!({ "type": "text_start", "contentIndex": 0 }),
            json!({ "type": "text_delta", "contentIndex": 0, "delta": "partial" }),
        ],
    );
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let options = options(&mock, "test-key");

    let message = stream_pi_messages(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("stream ended without a terminal event")),
        "{:?}",
        message.error_message
    );
}

// --- the simple entry, the tool-choice forms, and the payload options ---

/// The simple entry maps the shared tool choices onto the two wire strings
/// and rides the reasoning level.
#[tokio::test]
async fn streams_the_simple_entry_with_the_shared_tool_choices() {
    for (choice, wire) in [
        (pi_ai::types::ToolChoice::Auto, json!("auto")),
        (pi_ai::types::ToolChoice::None, json!("none")),
    ] {
        let mock = MockHttpClient::new();
        mount_events(&mock, &[done_frame("stop", &Value::Null)]);
        let model = pi_messages_model("http://127.0.0.1:9/v1");
        let options = pi_ai::types::SimpleStreamOptions {
            transport_options: common::mock_transport(&mock),
            api_key: Some("test-key".to_owned()),
            tool_choice: Some(choice),
            reasoning: Some(pi_ai::types::ThinkingLevel::High),
            ..pi_ai::types::SimpleStreamOptions::default()
        };

        let message = pi_ai::api::pi_messages::stream_simple(&model, &context(), Some(&options))
            .result()
            .await;

        assert_eq!(message.stop_reason, StopReason::Stop);
        let body = common::recorded_body(&mock);
        assert_eq!(body["options"]["toolChoice"], wire, "{wire}");
        assert!(body["options"]["reasoning"].is_string(), "{body}");
    }
}

/// The pi-messages-only tool-choice forms ride verbatim: `required` and the
/// forced-function shape.
#[tokio::test]
async fn maps_the_required_and_function_tool_choices_to_the_wire() {
    for (choice, expected) in [
        (PiMessagesToolChoice::Required, json!("required")),
        (
            PiMessagesToolChoice::Function {
                name: "read".to_owned(),
            },
            json!({ "type": "function", "function": { "name": "read" } }),
        ),
        (PiMessagesToolChoice::Auto, json!("auto")),
        (PiMessagesToolChoice::None, json!("none")),
    ] {
        let mock = MockHttpClient::new();
        mount_events(&mock, &[done_frame("stop", &Value::Null)]);
        let model = pi_messages_model("http://127.0.0.1:9/v1");
        let mut options = options(&mock, "test-key");
        options.tool_choice = Some(choice);

        let message = stream_pi_messages(&model, &context(), Some(&options))
            .result()
            .await;

        assert_eq!(message.stop_reason, StopReason::Stop);
        let body = common::recorded_body(&mock);
        assert_eq!(body["options"]["toolChoice"], expected, "{expected}");
    }
}

/// The cache-retention preference: the explicit option wins, the legacy env
/// opt-in maps `long`, and anything else falls back to the backend defaults.
#[tokio::test]
async fn carries_the_cache_retention_preference() {
    let explicit = MockHttpClient::new();
    mount_events(&explicit, &[done_frame("stop", &Value::Null)]);
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let mut explicit_options = options(&explicit, "test-key");
    explicit_options.cache_retention = Some(pi_ai::types::CacheRetention::Long);
    let message = stream_pi_messages(&model, &context(), Some(&explicit_options))
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(
        common::recorded_body(&explicit)["options"]["cacheRetention"],
        json!("long")
    );

    let env_opt_in = MockHttpClient::new();
    mount_events(&env_opt_in, &[done_frame("stop", &Value::Null)]);
    let mut env_options = options(&env_opt_in, "test-key");
    env_options.env = Some(std::collections::BTreeMap::from([(
        "PI_CACHE_RETENTION".to_owned(),
        "long".to_owned(),
    )]));
    let message = stream_pi_messages(&model, &context(), Some(&env_options))
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(
        common::recorded_body(&env_opt_in)["options"]["cacheRetention"],
        json!("long")
    );

    let other_env = MockHttpClient::new();
    mount_events(&other_env, &[done_frame("stop", &Value::Null)]);
    let mut other_options = options(&other_env, "test-key");
    other_options.env = Some(std::collections::BTreeMap::from([(
        "PI_CACHE_RETENTION".to_owned(),
        "short".to_owned(),
    )]));
    let message = stream_pi_messages(&model, &context(), Some(&other_options))
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert!(
        common::recorded_body(&other_env)["options"]
            .get("cacheRetention")
            .is_none(),
        "a non-long env value falls back to the backend defaults"
    );
}

/// The response-error body fallback: a body that is not an error object
/// rides verbatim (truncated at 8192 chars with the ellipsis), and the
/// non-string `error` member counts as absent.
#[tokio::test]
async fn falls_back_to_the_raw_body_when_the_error_body_is_not_an_error_object() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/messages"))
        .respond(MockResponse::status(500).with_body("plain gateway text"));
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let gateway_options = options(&mock, "test-key");

    let message = stream_pi_messages(&model, &context(), Some(&gateway_options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    let error_message = message.error_message.as_deref().unwrap_or_default();
    assert!(error_message.contains("500"), "{error_message}");
    assert!(
        error_message.contains("plain gateway text"),
        "{error_message}"
    );
    let diagnostics = message.diagnostics.as_ref().expect("diagnostics");
    assert_eq!(
        diagnostics[0]
            .details
            .as_ref()
            .and_then(|details| details.get("body"))
            .cloned(),
        Some(json!("plain gateway text")),
        "the raw body rides the diagnostic"
    );
    assert!(
        diagnostics[0]
            .details
            .as_ref()
            .and_then(|d| d.get("error"))
            .is_none()
    );

    // An `error` member that is not an object counts as absent.
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/messages"))
        .respond(MockResponse::status(502).with_body(r#"{"error": "string form"}"#));
    let string_error_options = options(&mock, "test-key");
    let message = stream_pi_messages(&model, &context(), Some(&string_error_options))
        .result()
        .await;
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("string form"))
    );

    // Over-long bodies truncate with the trailing ellipsis.
    let long_body = "x".repeat(9000);
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/messages"))
        .respond(MockResponse::status(503).with_body(long_body.clone()));
    let long_options = options(&mock, "test-key");
    let message = stream_pi_messages(&model, &context(), Some(&long_options))
        .result()
        .await;
    let diagnostics = message.diagnostics.as_ref().expect("diagnostics");
    let body = diagnostics[0]
        .details
        .as_ref()
        .and_then(|details| details.get("body"))
        .and_then(Value::as_str)
        .expect("the truncated body");
    assert_eq!(body.chars().count(), 8193, "8192 chars plus the ellipsis");
    assert!(body.ends_with('…'));
    assert!(body.starts_with("xxxx"));
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| text.contains("503: ")),
        "the message carries the raw body, only the diagnostic truncates"
    );
}

/// The gateway rewrite impact rides as the `pi_messages_rewrite` diagnostic.
#[tokio::test]
async fn carries_the_rewrite_diagnostic() {
    let mock = MockHttpClient::new();
    let rewrite = json!({
        "policyId": "redact-secrets",
        "policyVersion": 3,
        "changed": true,
        "tokenCountChange": -12,
        "messageCountChange": 0,
        "systemPromptChanged": true,
    });
    let done = done_frame("stop", &json!({ "rewrite": rewrite.clone() }));
    mount_events(&mock, &[done]);
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let options = options(&mock, "test-key");

    let message = stream_pi_messages(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    let diagnostics = message.diagnostics.as_ref().expect("diagnostics");
    assert_eq!(diagnostics[0].kind, "pi_messages_rewrite");
    let details = diagnostics[0].details.as_ref().expect("the details");
    assert_eq!(details.get("policyId"), Some(&json!("redact-secrets")));
    assert_eq!(details.get("policyVersion"), Some(&json!(3)));
    assert_eq!(details.get("changed"), Some(&json!(true)));
    assert_eq!(details.get("tokenCountChange"), Some(&json!(-12)));
    assert_eq!(details.get("messageCountChange"), Some(&json!(0)));
    assert_eq!(details.get("systemPromptChanged"), Some(&json!(true)));
}

/// Content indices past the end fill with empty text placeholders, a shape
/// only malformed streams produce, upstream's array-assignment semantics.
#[tokio::test]
async fn fills_content_holes_with_placeholders() {
    let mock = MockHttpClient::new();
    mount_events(
        &mock,
        &[
            json!({ "type": "text_start", "contentIndex": 2 }),
            json!({ "type": "text_delta", "contentIndex": 2, "delta": "late" }),
            json!({ "type": "text_end", "contentIndex": 2, "content": "Hello" }),
            done_frame("stop", &Value::Null),
        ],
    );
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let options = options(&mock, "test-key");

    let message = stream_pi_messages(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(
        message.content,
        vec![
            AssistantBlock::Text(TextContent {
                text: String::new(),
                text_signature: None
            }),
            AssistantBlock::Text(TextContent {
                text: String::new(),
                text_signature: None
            }),
            AssistantBlock::Text(TextContent {
                text: "Hello".to_owned(),
                text_signature: None,
            }),
        ]
    );
}

/// A tool-call close without an open settles the stream with the sequence
/// error, upstream's Object.assign crash into the catch path.
#[tokio::test]
async fn errors_when_a_toolcall_closes_without_an_open() {
    let mock = MockHttpClient::new();
    mount_events(
        &mock,
        &[
            json!({
                "type": "toolcall_end",
                "contentIndex": 0,
                "toolCall": { "type": "toolCall", "id": "call_1", "name": "read", "arguments": {} },
            }),
            done_frame("stop", &Value::Null),
        ],
    );
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let options = options(&mock, "test-key");

    let message = stream_pi_messages(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Invalid pi-messages event sequence")
    );
}

/// The thinking blocks stream through with their signatures and the
/// redacted flag.
#[tokio::test]
async fn carries_the_thinking_blocks_with_signatures() {
    let mock = MockHttpClient::new();
    mount_events(
        &mock,
        &[
            json!({ "type": "start" }),
            json!({ "type": "thinking_start", "contentIndex": 0 }),
            json!({ "type": "thinking_delta", "contentIndex": 0, "delta": "pon" }),
            json!({ "type": "thinking_delta", "contentIndex": 0, "delta": "der" }),
            json!({
                "type": "thinking_end",
                "contentIndex": 0,
                "content": "ponder",
                "contentSignature": "sig-1",
            }),
            json!({
                "type": "thinking_start",
                "contentIndex": 1,
            }),
            json!({
                "type": "thinking_end",
                "contentIndex": 1,
                "content": "[redacted]",
                "redacted": true,
            }),
            done_frame("stop", &Value::Null),
        ],
    );
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let options = options(&mock, "test-key");

    let message = stream_pi_messages(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(
        message.content,
        vec![
            AssistantBlock::Thinking(pi_ai::types::ThinkingContent {
                thinking: "ponder".to_owned(),
                thinking_signature: Some("sig-1".to_owned()),
                redacted: None,
            }),
            AssistantBlock::Thinking(pi_ai::types::ThinkingContent {
                thinking: "[redacted]".to_owned(),
                thinking_signature: None,
                redacted: Some(true),
            }),
        ]
    );
}

/// A frame that does not parse settles the stream with the parse failure
/// before any converter work, upstream's JSON.parse throw.
#[tokio::test]
async fn errors_when_an_sse_frame_is_not_valid_json() {
    let mock = MockHttpClient::new();
    let mut body = String::new();
    let _ = writeln!(body, "data: not json at all\n\n");
    body.push_str("data: [DONE]\n\n");
    mock.on(|request| request.url.contains("/messages"))
        .respond(MockResponse::status(200).with_body(body));
    let model = pi_messages_model("http://127.0.0.1:9/v1");
    let options = options(&mock, "test-key");

    let message = stream_pi_messages(&model, &context(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|text| !text.is_empty()),
        "the parse failure surfaces: {:?}",
        message.error_message
    );
}

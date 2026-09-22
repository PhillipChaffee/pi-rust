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
    Api, AssistantBlock, Context, Message, Model, Modality, ProviderId, StopReason, TextContent,
    ToolCall, UserContent, UserMessage,
};
use serde_json::{Value, json};

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
    let mut body = events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>();
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
fn done_frame(reason: &str, extra: Value) -> Value {
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
    assert_eq!(message.usage.cost.total, 0.3);

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
    mount_events(&mock, &[done_frame("stop", Value::Null)]);
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
            MockResponse::status(401).with_body(
                r#"{"error":{"message":"Token expired","code":"unauthorized"}}"#,
            ),
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
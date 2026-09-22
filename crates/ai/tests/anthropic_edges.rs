//! The Anthropic Messages seam edges the upstream suites leave implicit:
//! transport-error mapping, SDK error message shapes, malformed event
//! surfaces, stream-lifecycle failures, header-assembly branches, and the
//! payload-conversion corners. Behaviors upstream exercises at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`; tests pinning port seams the
//! upstream suites reach only implicitly are marked in their doc comments.

#![expect(
    clippy::expect_used,
    reason = "the tests pin stream outcomes; an unexpected shape panics the test by design"
)]

use std::sync::Arc;

use pi_ai::api::anthropic_messages::{
    AnthropicEffort, AnthropicStreamOptions, AnthropicToolChoice, stream as stream_anthropic,
    stream_simple,
};
use pi_ai::http::{MockBody, MockHttpClient, MockResponse};
use pi_ai::types::{
    AssistantBlock, CacheRetention, ConstrainedSamplingConfig, ConstrainedSamplingSetting, Context,
    Message, Modality, Model, ModelCompat, OnResponse, StopReason, Strictness, TextContent,
    ThinkingContent, ThinkingLevel, Tool, ToolCall, ToolChoice, ToolResultBlock, TransportOptions,
};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

mod common;
use common::{
    anthropic_mock, anthropic_mock_with, assistant_message_with_content, mock_transport,
    recorded_header, user_message_now,
};

fn done_context() -> Context {
    Context {
        messages: vec![user_message_now("Hello")],
        ..Context::default()
    }
}

fn keyed_options(mock: &MockHttpClient) -> AnthropicStreamOptions {
    common::keyed_anthropic_options(mock)
}

/// Capture the request payload through the on-payload hook while the mock
/// answers the request, so payload and settled message assert together.
async fn capture_params(
    model: &Model,
    context: &Context,
    mut options: AnthropicStreamOptions,
) -> (Value, pi_ai::types::AssistantMessage) {
    let (hook, captured) = common::payload_capture();
    options.transport_options.on_payload = Some(hook);
    let result = stream_anthropic(model, context, Some(&options))
        .result()
        .await;
    (common::captured_payload(&captured), result)
}

async fn capture_simple_params(
    model: &Model,
    context: &Context,
    mut options: pi_ai::types::SimpleStreamOptions,
) -> Value {
    let (hook, captured) = common::payload_capture();
    options.transport_options.on_payload = Some(hook);
    let _ = stream_simple(model, context, Some(&options)).result().await;
    common::captured_payload(&captured)
}

/// Stream the given events against a fresh mock and capture the request
/// payload, the shape the payload-only edge tests use.
async fn capture_done_events(
    model: &Model,
    context: &Context,
    events: &[(&'static str, String)],
) -> (Value, pi_ai::types::AssistantMessage) {
    capture_events_with(model, context, events, keyed_stream_options()).await
}

/// The keyed request options without a transport: the capture helpers mount
/// the mock.
fn keyed_stream_options() -> AnthropicStreamOptions {
    AnthropicStreamOptions {
        api_key: Some("test-key".to_owned()),
        ..AnthropicStreamOptions::default()
    }
}

/// Stream the given events with explicit options and capture the payload;
/// options without a transport ride the fresh mock.
async fn capture_events_with(
    model: &Model,
    context: &Context,
    events: &[(&'static str, String)],
    mut options: AnthropicStreamOptions,
) -> (Value, pi_ai::types::AssistantMessage) {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, events);
    if options.transport_options.http_client.is_none() {
        options.transport_options = mock_transport(&mock);
    }
    capture_params(model, context, options).await
}

/// The tool-marker context the deferred-reference suites send: a user turn,
/// a tool result loading the given names, and the given tools.
fn tool_marker_context(
    added_tool_names: Option<Vec<String>>,
    content: Vec<ToolResultBlock>,
    tools: Vec<Tool>,
) -> Context {
    Context {
        messages: vec![
            user_message_now("use the tool"),
            common::tool_result_message("call_1", content, added_tool_names),
        ],
        tools: Some(tools),
        ..Context::default()
    }
}

/// The plain text tool result the reference suites load.
fn loaded_result() -> ToolResultBlock {
    ToolResultBlock::Text(TextContent {
        text: "loaded".to_owned(),
        text_signature: None,
    })
}

// ---------------------------------------------------------------------------
// Transport-error mapping (provider_error_from_http)
// ---------------------------------------------------------------------------

/// Port-added: the seam's `Timeout` and `InvalidUrl` arms keep their
/// transport wordings; the Transport arm rides the no-route mock in the
/// conformance suites.
#[tokio::test]
async fn transport_error_arms_surface_their_seam_messages() {
    let cases: [(pi_ai::http::HttpError, &str); 2] = [
        (pi_ai::http::HttpError::Timeout, "request timed out"),
        (
            pi_ai::http::HttpError::InvalidUrl("not a url".to_owned()),
            "invalid URL: not a url",
        ),
    ];
    for (error, expected) in cases {
        let mock = MockHttpClient::new();
        mock.on(|request| request.url.contains("/v1/messages"))
            .respond_fn(move |_request| {
                let error = error.clone();
                async move { Err(error) }
            });
        let options = keyed_options(&mock);

        let result = stream_anthropic(&common::anthropic_model(), &done_context(), Some(&options))
            .result()
            .await;

        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some(expected));
    }
}

/// A pre-cancelled signal aborts before dispatch: the stream stops with
/// `Aborted` and the abort wording, upstream's `AbortError` mapping.
#[tokio::test]
async fn aborted_requests_set_the_aborted_stop_reason() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let token = CancellationToken::new();
    token.cancel();
    let model = common::anthropic_model();
    let options = AnthropicStreamOptions {
        transport_options: TransportOptions {
            http_client: Some(Arc::new(mock.clone())),
            signal: Some(token),
            ..TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        ..AnthropicStreamOptions::default()
    };

    let result = stream_anthropic(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq!(result.error_message.as_deref(), Some("Request aborted"));
}

// ---------------------------------------------------------------------------
// SDK error message shapes
// ---------------------------------------------------------------------------

/// Port-added: the raw-body and missing-body variants of the SDK's
/// `{status} {body}` error message.
#[tokio::test]
async fn sdk_error_messages_cover_raw_and_missing_bodies() {
    let cases: [(u16, Option<&str>, &str); 2] = [
        (502, Some("Bad Gateway"), "502 Bad Gateway"),
        (503, None, "503 status code (no body)"),
    ];
    for (status, body, expected) in cases {
        let mock = MockHttpClient::new();
        mock.on(|request| request.url.contains("/v1/messages"))
            .respond(body.map_or_else(
                || MockResponse::status(status),
                |body| MockResponse::status(status).with_body(body),
            ));
        let model = common::anthropic_model();
        let options = keyed_options(&mock);

        let result = stream_anthropic(&model, &done_context(), Some(&options))
            .result()
            .await;

        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some(expected));
    }
}

// ---------------------------------------------------------------------------
// Malformed event surfaces
// ---------------------------------------------------------------------------

/// An `event: error` frame fails the stream with its data, upstream's
/// mid-stream error propagation.
#[tokio::test]
async fn sse_error_events_fail_the_stream_with_their_data() {
    let mock = anthropic_mock(&[("error", "boom".to_owned())]);
    let model = common::anthropic_model();
    let options = keyed_options(&mock);

    let result = stream_anthropic(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.error_message.as_deref(), Some("boom"));
}

/// Stream the built events with the standard model and context and settle
/// the result, the shape the lifecycle tests share.
async fn stream_done(events: Vec<(&'static str, String)>) -> pi_ai::types::AssistantMessage {
    let mock = anthropic_mock(&events);
    stream_anthropic(
        &common::anthropic_model(),
        &done_context(),
        Some(&keyed_options(&mock)),
    )
    .result()
    .await
}

/// Port-added: nameless SSE frames (data-only) skip the event loop, the
/// proxy-heartbeat shape the decoder yields with no event name.
#[tokio::test]
async fn unnamed_sse_events_are_skipped() {
    let mock = MockHttpClient::new();
    let done = common::minimal_anthropic_done();
    let pairs = common::sse_pairs(&done);
    let tail = match pi_ai::http::sse_response(200, &pairs).body {
        MockBody::Bytes(body) => String::from_utf8_lossy(&body).into_owned(),
        MockBody::Chunks(chunks) => chunks
            .iter()
            .map(|chunk| String::from_utf8_lossy(chunk).into_owned())
            .collect::<String>(),
        MockBody::Empty => String::new(),
    };
    let body = format!("data: {{\"type\":\"ping\"}}\n\n{tail}");
    mock.on(|request| request.url.contains("/v1/messages"))
        .respond(
            MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_body(body),
        );
    let model = common::anthropic_model();
    let options = keyed_options(&mock);

    let result = stream_anthropic(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.error_message, None);
}

/// Port-added: an event whose JSON parses but fails the typed shape fails
/// with the parse message naming the event.
#[tokio::test]
async fn typed_deser_failures_carry_the_parse_message() {
    let mock = anthropic_mock(&[(
        "content_block_start",
        json!({ "type": "content_block_start", "index": 0 }).to_string(),
    )]);
    let model = common::anthropic_model();
    let options = keyed_options(&mock);

    let result = stream_anthropic(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the parse error");
    assert!(
        message.starts_with("Could not parse Anthropic SSE event content_block_start"),
        "got: {message}"
    );
    assert!(message.contains("raw="), "got: {message}");
}

// ---------------------------------------------------------------------------
// Stream-lifecycle failures
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streams_ending_before_message_stop_fail() {
    let mock = anthropic_mock(&[(
        "message_start",
        json!({
            "type": "message_start",
            "message": { "id": "msg_truncated", "usage": { "input_tokens": 1, "output_tokens": 0 } },
        })
        .to_string(),
    )]);
    let model = common::anthropic_model();
    let options = keyed_options(&mock);

    let result = stream_anthropic(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Anthropic stream ended before message_stop")
    );
}

#[tokio::test]
async fn streams_without_a_stop_reason_fail() {
    let result = stream_done(vec![
        common::message_start_event(
            "msg_quiet",
            json!({ "input_tokens": 1, "output_tokens": 0 }),
        ),
        common::message_stop_event(),
    ])
    .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Anthropic stream ended without a stop reason")
    );
}

#[tokio::test]
async fn unknown_stop_reasons_fail_the_stream() {
    let result = stream_done(common::anthropic_done_run(
        "msg_test",
        vec![],
        "banana",
        json!({ "input_tokens": 1, "output_tokens": 1 }),
    ))
    .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Unhandled stop reason: banana")
    );
    assert_eq!(result.raw_stop_reason.as_deref(), Some("banana"));
}

/// Anthropic reports reasoning tokens as a subset of output tokens; the
/// breakdown rides `usage.reasoning`.
#[tokio::test]
async fn reasoning_tokens_ride_the_usage_breakdown() {
    let mock = anthropic_mock(&[
        common::message_start_event("msg_test", json!({ "input_tokens": 1, "output_tokens": 0 })),
        common::message_delta_event(
            json!({ "stop_reason": "end_turn" }),
            Some(
                json!({ "input_tokens": 1, "output_tokens": 30, "output_tokens_details": { "thinking_tokens": 7 }, }),
            ),
        ),
        common::message_stop_event(),
    ]);
    let model = common::anthropic_model();
    let options = keyed_options(&mock);

    let result = stream_anthropic(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.usage.reasoning, Some(7));
}

// ---------------------------------------------------------------------------
// Content-block edges
// ---------------------------------------------------------------------------

#[tokio::test]
async fn redacted_thinking_streams_as_the_opaque_payload_and_replays_verbatim() {
    let events = vec![
        common::message_start_event(
            "msg_redacted",
            json!({ "input_tokens": 1, "output_tokens": 0 }),
        ),
        common::block_start_event(
            0,
            json!({ "type": "redacted_thinking", "data": "enc-payload" }),
        ),
        common::block_stop_event(0),
        common::message_delta_event(
            json!({ "stop_reason": "end_turn" }),
            Some(json!({ "input_tokens": 1, "output_tokens": 1 })),
        ),
        common::message_stop_event(),
    ];
    let result = stream_done(events).await;

    assert_eq!(
        result.content,
        vec![AssistantBlock::Thinking(ThinkingContent {
            thinking: "[Reasoning redacted]".to_owned(),
            thinking_signature: Some("enc-payload".to_owned()),
            redacted: Some(true),
        })],
    );

    // The redacted block replays as the wire's opaque payload.
    let (payload, _) = capture_done_events(
        &common::anthropic_model(),
        &Context {
            messages: vec![Message::Assistant(result)],
            ..Context::default()
        },
        &common::minimal_anthropic_done(),
    )
    .await;
    assert_eq!(
        payload["messages"][0]["content"][0],
        json!({ "type": "redacted_thinking", "data": "enc-payload" }),
    );
}

#[tokio::test]
async fn deltas_for_unknown_blocks_or_mismatched_types_are_ignored() {
    let result = stream_done(common::anthropic_done_run(
        "msg_test",
        vec![
            common::block_start_event(0, json!({ "type": "text", "text": "" })),
            common::block_delta_event(7, json!({ "type": "text_delta", "text": "ghost" })),
            common::block_delta_event(0, json!({ "type": "thinking_delta", "thinking": "nope" })),
            common::block_delta_event(0, json!({ "type": "text_delta", "text": "Hello" })),
            common::block_stop_event(0),
        ],
        "end_turn",
        json!({ "input_tokens": 1, "output_tokens": 1 }),
    ))
    .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(
        result.content,
        vec![AssistantBlock::Text(TextContent {
            text: "Hello".to_owned(),
            text_signature: None,
        })],
    );
}

/// A fallback `message_start` serving model reprices from the fallback's
/// locally recorded rates.
#[tokio::test]
async fn a_serving_model_fallback_reprices_from_the_fallback_entry() {
    let mock = anthropic_mock(&[
        common::message_start_from_serving_model(
            "claude-fallback",
            "msg_fallback",
            json!({ "input_tokens": 10, "output_tokens": 2 }),
        ),
        common::message_delta_event(
            json!({ "stop_reason": "end_turn" }),
            Some(json!({ "input_tokens": 10, "output_tokens": 2 })),
        ),
        common::message_stop_event(),
    ]);
    let model = Model {
        compat: Some(
            serde_json::from_value::<ModelCompat>(json!({
                "allowedFallbackModels": [
                    {
                        "provider": "anthropic",
                        "model": "claude-fallback",
                        "cost": { "input": 1.0, "output": 5.0, "cacheRead": 0.0, "cacheWrite": 0.0 },
                    },
                ],
            }))
            .expect("compat map"),
        ),
        ..common::anthropic_model()
    };
    let options = keyed_options(&mock);

    let result = stream_anthropic(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.model, "claude-fallback");
    // The requested model's own rates are zero, so these numbers can only
    // come from the fallback entry.
    assert!(
        (result.usage.cost.input - 1.0e-5).abs() < 1e-12,
        "got: {:?}",
        result.usage.cost
    );
    assert!(
        (result.usage.cost.output - 1.0e-5).abs() < 1e-12,
        "got: {:?}",
        result.usage.cost
    );
}

// ---------------------------------------------------------------------------
// Header assembly
// ---------------------------------------------------------------------------

/// Port-added: the session-affinity header follows the compat format,
/// `x-session-affinity` by default and `x-session-id` for OpenRouter
/// endpoints.
#[tokio::test]
async fn session_affinity_headers_follow_the_compat_format() {
    let cases: [(Value, &str); 2] = [
        (
            json!({ "sendSessionAffinityHeaders": true }),
            "x-session-affinity",
        ),
        (
            json!({ "sendSessionAffinityHeaders": true, "sessionAffinityFormat": "openrouter" }),
            "x-session-id",
        ),
    ];
    for (compat, header) in cases {
        let mock = MockHttpClient::new();
        anthropic_mock_with(&mock, &common::minimal_anthropic_done());
        let model = Model {
            compat: Some(serde_json::from_value::<ModelCompat>(compat).expect("compat map")),
            ..common::anthropic_model()
        };
        let options = AnthropicStreamOptions {
            transport_options: mock_transport(&mock),
            api_key: Some("test-key".to_owned()),
            session_id: Some("sess-1".to_owned()),
            ..AnthropicStreamOptions::default()
        };

        let _ = stream_anthropic(&model, &done_context(), Some(&options))
            .result()
            .await;

        assert_eq!(
            recorded_header(&mock, header).as_deref(),
            Some("sess-1"),
            "header: {header}"
        );
    }
}

#[tokio::test]
async fn model_headers_merge_into_the_request() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let model = Model {
        headers: Some(
            std::iter::once(("x-model-tenant".to_owned(), "tenant-1".to_owned())).collect(),
        ),
        ..common::anthropic_model()
    };
    let options = keyed_options(&mock);

    let _ = stream_anthropic(&model, &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(
        recorded_header(&mock, "x-model-tenant").as_deref(),
        Some("tenant-1")
    );
}

/// The response hook observes the settled status, upstream's `onResponse`.
#[tokio::test]
async fn the_response_hook_observes_the_response_status() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let slot = Arc::clone(&observed);
    let options = AnthropicStreamOptions {
        transport_options: TransportOptions {
            http_client: Some(Arc::new(mock.clone())),
            on_response: Some(OnResponse::new(move |response, _model| {
                slot.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(response.status);
                Box::pin(async {})
            })),
            ..TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        ..AnthropicStreamOptions::default()
    };

    let result = stream_anthropic(&common::anthropic_model(), &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(
        *observed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        [200]
    );
}

// ---------------------------------------------------------------------------
// Mid-stream seam errors
// ---------------------------------------------------------------------------

/// A canned-seam client: the request always resolves 200 and the body
/// streams the given chunks, the shape the SSE error mapping needs.
#[derive(Debug)]
struct SeamedClient {
    chunks: Vec<Result<bytes::Bytes, pi_ai::http::HttpError>>,
}

impl pi_ai::http::HttpClient for SeamedClient {
    fn execute(
        &self,
        _request: pi_ai::http::HttpRequest,
    ) -> pi_ai::http::BoxHttpFuture<Result<pi_ai::http::HttpResponse, pi_ai::http::HttpError>> {
        let response = pi_ai::http::HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: pi_ai::http::HttpByteStream::from_chunks(self.chunks.clone()),
        };
        Box::pin(async move { Ok(response) })
    }
}

fn sse_frame(event: &str, data: &Value) -> bytes::Bytes {
    bytes::Bytes::from(format!("event: {event}\ndata: {data}\n"))
}

fn message_start_frame() -> bytes::Bytes {
    sse_frame(
        "message_start",
        &json!({
            "type": "message_start",
            "message": { "id": "msg_seamed", "usage": { "input_tokens": 1, "output_tokens": 0 } },
        }),
    )
}

fn seamed_options(
    chunks: Vec<Result<bytes::Bytes, pi_ai::http::HttpError>>,
) -> AnthropicStreamOptions {
    AnthropicStreamOptions {
        transport_options: TransportOptions {
            http_client: Some(Arc::new(SeamedClient { chunks })),
            ..TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        ..AnthropicStreamOptions::default()
    }
}

/// Port-added: a transport failure mid-SSE carries the transport's wording,
/// and a mid-stream abort carries the abort wording.
#[tokio::test]
async fn mid_stream_errors_carry_their_wording() {
    let cases: [(pi_ai::http::HttpError, &str); 2] = [
        (
            pi_ai::http::HttpError::Transport("socket reset".to_owned()),
            "socket reset",
        ),
        (pi_ai::http::HttpError::Aborted, "Request was aborted"),
    ];
    for (error, expected) in cases {
        let options = seamed_options(vec![Ok(message_start_frame()), Err(error)]);
        let result = stream_anthropic(&common::anthropic_model(), &done_context(), Some(&options))
            .result()
            .await;

        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.error_message.as_deref(), Some(expected));
    }
}

/// A fallback before any output is a no-op: the stream continues, upstream's
/// fallback-after-output guard's other arm.
#[tokio::test]
async fn a_fallback_before_output_is_a_no_op() {
    let mock = anthropic_mock(&[
        common::message_start_event(
            "msg_fb_first",
            json!({ "input_tokens": 1, "output_tokens": 0 }),
        ),
        common::block_start_event(
            0,
            json!({ "type": "fallback", "from": { "model": "claude-test" } }),
        ),
        common::block_start_event(1, json!({ "type": "text", "text": "After the switch" })),
        common::block_stop_event(1),
        common::block_stop_event(9),
        common::message_delta_event(
            json!({ "stop_reason": "end_turn" }),
            Some(json!({ "input_tokens": 1, "output_tokens": 1 })),
        ),
        common::message_stop_event(),
    ]);
    let options = keyed_options(&mock);

    let result = stream_anthropic(&common::anthropic_model(), &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(
        result.content,
        vec![AssistantBlock::Text(TextContent {
            text: "After the switch".to_owned(),
            text_signature: None,
        })],
    );
}

/// A `message_delta` without a stop reason is a usage-only update; the stop
/// reason rides a later delta.
#[tokio::test]
async fn usage_only_message_deltas_update_usage_without_touching_the_stop_reason() {
    let mock = anthropic_mock(&[
        common::message_start_event("msg_test", json!({ "input_tokens": 1, "output_tokens": 0 })),
        common::message_delta_event(
            json!({}),
            Some(json!({ "output_tokens": 30, "output_tokens_details": { "thinking_tokens": 7 } })),
        ),
        common::message_delta_event(
            json!({ "stop_reason": "end_turn" }),
            Some(json!({ "output_tokens": 31 })),
        ),
        common::message_stop_event(),
    ]);
    let options = keyed_options(&mock);

    let result = stream_anthropic(&common::anthropic_model(), &done_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.usage.reasoning, Some(7));
    assert_eq!(result.usage.output, 31);
}

// ---------------------------------------------------------------------------
// Payload-conversion corners
// ---------------------------------------------------------------------------

/// Port-added: `cache_retention: long` carries the 1h TTL, driven both by
/// the request and by `PI_CACHE_RETENTION` in the provider env.
#[tokio::test]
async fn long_cache_retention_carries_the_one_hour_ttl() {
    for cache_retention in [Some(CacheRetention::Long), None] {
        let mock = MockHttpClient::new();
        anthropic_mock_with(&mock, &common::minimal_anthropic_done());
        let options = AnthropicStreamOptions {
            transport_options: mock_transport(&mock),
            api_key: Some("test-key".to_owned()),
            cache_retention,
            env: Some(
                std::iter::once(("PI_CACHE_RETENTION".to_owned(), "long".to_owned())).collect(),
            ),
            ..AnthropicStreamOptions::default()
        };
        let context = Context {
            system_prompt: Some("System prompt.".to_owned()),
            messages: vec![user_message_now("Hello")],
            ..Context::default()
        };

        let (payload, _) = capture_params(&common::anthropic_model(), &context, options).await;

        assert_eq!(
            payload["system"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" }),
        );
    }
}

/// Port-added: `thinking_display: omitted` rides the managed-effort thinking
/// block.
#[tokio::test]
async fn thinking_display_omitted_rides_the_managed_thinking_block() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let model = Model {
        compat: Some(
            serde_json::from_value::<ModelCompat>(json!({ "supportsMidConvoEffort": true }))
                .expect("compat map"),
        ),
        ..common::anthropic_model()
    };
    let options = AnthropicStreamOptions {
        transport_options: mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        thinking_display: Some(pi_ai::api::anthropic_messages::AnthropicThinkingDisplay::Omitted),
        ..AnthropicStreamOptions::default()
    };

    let (payload, _) = capture_params(&model, &done_context(), options).await;

    assert_eq!(payload["thinking"]["display"], json!("omitted"));
    assert_eq!(
        payload["thinking"]["block_binding"],
        json!({ "prefix_mismatch_behavior": "drop_block" }),
    );
}

/// Port-added: `supportsToolReferences` defaults off outside the first-party
/// Claude 4.5+ families — non-claude ids, unmapped suffixes, and date-like
/// digit runs all ride plain content.
#[tokio::test]
async fn tool_references_default_off_outside_first_party_claude_families() {
    let ids = [
        "custom-model",
        "claude-opus-vision",
        "claude-opus-99999999999",
    ];
    for id in ids {
        let model = Model {
            id: id.to_owned(),
            ..common::anthropic_model()
        };
        let context = tool_marker_context(
            Some(vec!["late_tool".to_owned()]),
            vec![loaded_result()],
            vec![common::lookup_tool(), common::late_tool()],
        );

        let (payload, _) =
            capture_done_events(&model, &context, &common::minimal_anthropic_done()).await;

        let last_turn = &payload["messages"]
            .as_array()
            .expect("messages")
            .last()
            .expect("the tool-result turn")["content"];
        assert_eq!(last_turn[0]["content"], json!("loaded"));
        // The tool definitions ride immediately; nothing defers.
        let tools = payload["tools"].as_array().expect("tools");
        assert!(
            tools.iter().all(|tool| tool.get("defer_loading").is_none()),
            "{id}: no tool defers"
        );
    }
}

/// Port-added: when every tool defers, the set promotes to immediate — the
/// request always carries at least the deferred definitions.
#[tokio::test]
async fn an_all_deferred_tool_set_promotes_to_immediate() {
    let model = Model {
        id: "claude-opus-4-8".to_owned(),
        ..common::anthropic_model()
    };
    let context = tool_marker_context(
        Some(vec!["lookup".to_owned()]),
        vec![loaded_result()],
        vec![common::lookup_tool()],
    );

    let (payload, _) =
        capture_done_events(&model, &context, &common::minimal_anthropic_done()).await;

    let tools = payload["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 1);
    assert!(tools[0].get("defer_loading").is_none());
    // Without a deferred set the load marker rides as ordinary content.
    let last_turn = &payload["messages"]
        .as_array()
        .expect("messages")
        .last()
        .expect("the tool-result turn")["content"];
    assert_eq!(last_turn[0]["content"], json!("loaded"));
}

/// Port-added: a same-model signed thinking block whose signature is only
/// whitespace survives the rewrite and drops at the wire conversion.
#[tokio::test]
async fn whitespace_signed_empty_thinking_drops_at_the_wire_conversion() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let context = Context {
        messages: vec![
            user_message_now("hello"),
            Message::Assistant(assistant_message_with_content(
                "anthropic-messages",
                "anthropic",
                "claude-test",
                vec![AssistantBlock::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: Some("   ".to_owned()),
                    redacted: None,
                })],
            )),
        ],
        ..Context::default()
    };

    let (payload, _) =
        capture_params(&common::anthropic_model(), &context, keyed_options(&mock)).await;

    let messages = payload["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
}

#[tokio::test]
async fn thinking_without_a_budget_sends_the_default_budget_tokens() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let mut model = common::anthropic_model();
    model.reasoning = true;
    let options = AnthropicStreamOptions {
        transport_options: mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        thinking_enabled: Some(true),
        ..AnthropicStreamOptions::default()
    };

    let (payload, _) = capture_params(&model, &done_context(), options).await;

    assert_eq!(payload["thinking"]["budget_tokens"], json!(1024));
}

/// Port-added: effort falls back to `low`/`high` when the model carries no
/// `thinkingLevelMap`.
#[tokio::test]
async fn effort_falls_back_to_low_and_high_without_a_map() {
    let levels: [(ThinkingLevel, &str); 2] = [
        (ThinkingLevel::Minimal, "low"),
        (ThinkingLevel::Max, "high"),
    ];
    for (level, effort) in levels {
        let mock = MockHttpClient::new();
        anthropic_mock_with(&mock, &common::minimal_anthropic_done());
        let mut model = common::anthropic_model();
        model.reasoning = true;
        model.compat = Some(
            serde_json::from_value::<ModelCompat>(json!({ "forceAdaptiveThinking": true }))
                .expect("compat map"),
        );
        let options = pi_ai::types::SimpleStreamOptions {
            transport_options: mock_transport(&mock),
            api_key: Some("test-key".to_owned()),
            reasoning: Some(level),
            ..pi_ai::types::SimpleStreamOptions::default()
        };

        let payload = capture_simple_params(&model, &done_context(), options).await;

        assert_eq!(payload["output_config"], json!({ "effort": effort }));
    }
}

/// Port-added: a `thinkingLevelMap` entry naming a native effort spelling
/// wins over the fallback, for the low level too.
#[tokio::test]
async fn thinking_level_maps_drive_the_low_effort_too() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let mut model = common::anthropic_model();
    model.reasoning = true;
    model.compat = Some(
        serde_json::from_value::<ModelCompat>(json!({ "forceAdaptiveThinking": true }))
            .expect("compat map"),
    );
    model.thinking_level_map = Some(
        std::iter::once((
            pi_ai::types::ModelThinkingLevel::Low,
            Some("medium".to_owned()),
        ))
        .collect(),
    );
    let options = pi_ai::types::SimpleStreamOptions {
        transport_options: mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        reasoning: Some(ThinkingLevel::Low),
        ..pi_ai::types::SimpleStreamOptions::default()
    };

    let payload = capture_simple_params(&model, &done_context(), options).await;

    assert_eq!(payload["output_config"], json!({ "effort": "medium" }));
}

#[tokio::test]
async fn tool_choice_variants_map_to_the_wire() {
    let cases: [(Option<AnthropicToolChoice>, Value); 3] = [
        (Some(AnthropicToolChoice::Any), json!({ "type": "any" })),
        (Some(AnthropicToolChoice::None), json!({ "type": "none" })),
        (
            Some(AnthropicToolChoice::Tool {
                name: "lookup".to_owned(),
            }),
            json!({ "type": "tool", "name": "lookup" }),
        ),
    ];
    for (choice, expected) in cases {
        let mock = MockHttpClient::new();
        anthropic_mock_with(&mock, &common::minimal_anthropic_done());
        let options = AnthropicStreamOptions {
            transport_options: mock_transport(&mock),
            api_key: Some("test-key".to_owned()),
            tool_choice: choice,
            ..AnthropicStreamOptions::default()
        };

        let (payload, _) =
            capture_params(&common::anthropic_model(), &done_context(), options).await;

        assert_eq!(payload["tool_choice"], expected);
    }

    // The neutral pi-level choice maps through the simple entry.
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let options = pi_ai::types::SimpleStreamOptions {
        transport_options: mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        tool_choice: Some(ToolChoice::None),
        ..pi_ai::types::SimpleStreamOptions::default()
    };
    let payload = capture_simple_params(&common::anthropic_model(), &done_context(), options).await;
    assert_eq!(payload["tool_choice"], json!({ "type": "none" }));
}

#[tokio::test]
async fn cache_retention_none_drops_the_cache_control_marker() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let options = AnthropicStreamOptions {
        transport_options: mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        cache_retention: Some(CacheRetention::None),
        ..AnthropicStreamOptions::default()
    };
    let context = Context {
        system_prompt: Some("System prompt.".to_owned()),
        messages: vec![user_message_now("Hello")],
        ..Context::default()
    };

    let (payload, _) = capture_params(&common::anthropic_model(), &context, options).await;

    assert!(payload["system"][0].get("cache_control").is_none());
}

#[tokio::test]
async fn string_user_content_gains_the_cache_control_block() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let options = AnthropicStreamOptions {
        transport_options: mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        cache_retention: Some(CacheRetention::Short),
        ..AnthropicStreamOptions::default()
    };

    let (payload, _) = capture_params(&common::anthropic_model(), &done_context(), options).await;

    let last = payload["messages"]
        .as_array()
        .expect("messages")
        .last()
        .expect("the user turn");
    assert!(last["content"].is_array());
    assert_eq!(last["content"][0]["type"], "text");
    assert!(last["content"][0].get("cache_control").is_some());
}

#[tokio::test]
async fn empty_user_and_assistant_turns_drop_out_of_the_request() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let context = Context {
        messages: vec![
            Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Blocks(vec![pi_ai::types::UserBlock::Text(
                    TextContent {
                        text: "   ".to_owned(),
                        text_signature: None,
                    },
                )]),
                timestamp: 1,
            }),
            Message::Assistant(assistant_message_with_content(
                "anthropic-messages",
                "anthropic",
                "claude-test",
                vec![
                    AssistantBlock::Text(TextContent {
                        text: String::new(),
                        text_signature: None,
                    }),
                    AssistantBlock::Thinking(ThinkingContent {
                        thinking: String::new(),
                        thinking_signature: None,
                        redacted: None,
                    }),
                ],
            )),
            user_message_now("Hello"),
        ],
        ..Context::default()
    };

    let (payload, _) =
        capture_params(&common::anthropic_model(), &context, keyed_options(&mock)).await;

    let messages = payload["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
}

/// A tool call in history normalizes its id to Anthropic's pattern, and an
/// OAuth history turn renames the call to CC casing.
#[tokio::test]
async fn history_tool_calls_normalize_ids_and_cc_names() {
    let tool_call = ToolCall {
        id: "call 1/a".to_owned(),
        name: "lookup".to_owned(),
        arguments: serde_json::Map::new(),
        thought_signature: None,
        namespace: None,
    };
    let context = Context {
        messages: vec![
            user_message_now("use the tool"),
            Message::Assistant(assistant_message_with_content(
                "openai-completions",
                "other",
                "gpt-x",
                vec![AssistantBlock::ToolCall(tool_call)],
            )),
            common::tool_result_message(
                "call 1/a",
                vec![ToolResultBlock::Text(TextContent {
                    text: "ok".to_owned(),
                    text_signature: None,
                })],
                None,
            ),
        ],
        ..Context::default()
    };

    let (payload, _) = capture_params(
        &common::anthropic_model(),
        &context,
        keyed_options(&MockHttpClient::new()),
    )
    .await;

    let messages = payload["messages"].as_array().expect("messages");
    let assistant = &messages[1];
    assert_eq!(assistant["content"][0]["type"], "tool_use");
    assert_eq!(assistant["content"][0]["id"], json!("call_1_a"));
    assert_eq!(assistant["content"][0]["name"], json!("lookup"));
    let tool_result = &messages[2];
    assert_eq!(tool_result["content"][0]["tool_use_id"], json!("call_1_a"));

    // OAuth renames the outbound call to CC casing.
    let oauth_call = ToolCall {
        id: "toolu_1".to_owned(),
        name: "todowrite".to_owned(),
        arguments: serde_json::Map::new(),
        thought_signature: None,
        namespace: None,
    };
    let oauth_context = Context {
        messages: vec![
            user_message_now("add a todo"),
            Message::Assistant(assistant_message_with_content(
                "openai-completions",
                "other",
                "gpt-x",
                vec![AssistantBlock::ToolCall(oauth_call)],
            )),
        ],
        ..Context::default()
    };
    let oauth_options = AnthropicStreamOptions {
        api_key: Some("sk-ant-oat-token".to_owned()),
        ..keyed_options(&MockHttpClient::new())
    };
    let (payload, _) =
        capture_params(&common::anthropic_model(), &oauth_context, oauth_options).await;
    let assistant = &payload["messages"].as_array().expect("messages")[1];
    assert_eq!(assistant["content"][0]["name"], json!("TodoWrite"));
}

/// An OAuth inbound tool call maps back unchanged when the context carries
/// no tools to look the name up in, upstream's `fromClaudeCodeName` guard.
#[tokio::test]
async fn oauth_tool_calls_without_context_tools_keep_the_wire_name() {
    let events = [
        common::message_start_event("msg_test", json!({ "input_tokens": 1, "output_tokens": 0 })),
        common::block_start_event(
            0,
            json!({ "type": "tool_use", "id": "toolu_1", "name": "CustomTool", "input": {} }),
        ),
        common::block_stop_event(0),
        common::message_delta_event(
            json!({ "stop_reason": "tool_use" }),
            Some(json!({ "input_tokens": 1, "output_tokens": 1 })),
        ),
        common::message_stop_event(),
    ];
    for tools in [None, Some(Vec::new())] {
        let mock = anthropic_mock(&events);
        let options = AnthropicStreamOptions {
            transport_options: mock_transport(&mock),
            api_key: Some("sk-ant-oat-token".to_owned()),
            ..AnthropicStreamOptions::default()
        };
        let context = Context {
            tools,
            ..done_context()
        };

        let result = stream_anthropic(&common::anthropic_model(), &context, Some(&options))
            .result()
            .await;

        let tool_call = result.content.iter().find_map(|block| match block {
            AssistantBlock::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        });
        assert_eq!(tool_call.expect("tool call").name, "CustomTool");
    }
}

/// A `require`-strict tool whose schema leaves the subset fails the request
/// before dispatch, upstream's `resolveJsonSchemaStrictSampling` rejection.
#[tokio::test]
async fn require_strict_tools_with_unsupported_schemas_fail_the_request() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let model = Model {
        compat: Some(
            serde_json::from_value::<ModelCompat>(json!({ "supportsStrictTools": true }))
                .expect("compat map"),
        ),
        ..common::anthropic_model()
    };
    let tool = Tool {
        name: "strict_tool".to_owned(),
        description: "Strict tool.".to_owned(),
        parameters: json!({
            "type": "object",
            "allOf": [{ "type": "object", "properties": {} }],
        }),
        constrained_sampling: Some(ConstrainedSamplingSetting::Config(
            ConstrainedSamplingConfig::JsonSchema {
                strict: Strictness::Require,
            },
        )),
    };
    let context = Context {
        messages: vec![user_message_now("Hello")],
        tools: Some(vec![tool]),
        ..Context::default()
    };
    let options = keyed_options(&mock);

    let result = stream_anthropic(&model, &context, Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the strict rejection");
    assert!(
        message.contains("requires JSON-schema constrained sampling"),
        "got: {message}"
    );
}

// ---------------------------------------------------------------------------
// Deferred tool references
// ---------------------------------------------------------------------------

/// Port-added: a load marker naming a tool outside the context's set rides
/// as ordinary content, the `deferredToolNames` guard.
#[tokio::test]
async fn tool_references_skip_names_outside_the_tool_set() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let model = Model {
        id: "claude-opus-4-8".to_owned(),
        ..common::anthropic_model()
    };
    let context = Context {
        messages: vec![
            user_message_now("use the tool"),
            common::tool_result_message(
                "call_1",
                vec![ToolResultBlock::Text(TextContent {
                    text: "loaded".to_owned(),
                    text_signature: None,
                })],
                Some(vec!["ghost".to_owned()]),
            ),
        ],
        tools: Some(vec![common::lookup_tool()]),
        ..Context::default()
    };

    let (payload, _) = capture_params(&model, &context, keyed_options(&mock)).await;

    let last_turn = &payload["messages"]
        .as_array()
        .expect("messages")
        .last()
        .expect("the tool-result turn")["content"];
    assert_eq!(last_turn[0]["content"], json!("loaded"));
}

/// Port-added: an OAuth reference result names the tool in CC casing.
#[tokio::test]
async fn oauth_tool_references_carry_cc_tool_names() {
    let model = Model {
        id: "claude-opus-4-8".to_owned(),
        ..common::anthropic_model()
    };
    let context = tool_marker_context(
        Some(vec!["todowrite".to_owned()]),
        vec![loaded_result()],
        vec![common::lookup_tool(), common::todo_tool()],
    );

    let (payload, _) = capture_events_with(
        &model,
        &context,
        &common::minimal_anthropic_done(),
        AnthropicStreamOptions {
            api_key: Some("sk-ant-oat-token".to_owned()),
            ..AnthropicStreamOptions::default()
        },
    )
    .await;

    let last_turn = &payload["messages"]
        .as_array()
        .expect("messages")
        .last()
        .expect("the tool-result turn")["content"];
    assert_eq!(
        last_turn[0],
        json!({ "type": "tool_result", "tool_use_id": "call_1", "content": [
            { "type": "tool_reference", "tool_name": "TodoWrite" },
        ], "is_error": false }),
    );
    // The ordinary content rides as the sibling block.
    assert_eq!(last_turn[1]["type"], "text");
    assert_eq!(last_turn[1]["text"], json!("loaded"));
}

/// Port-added: reference-bearing results displace array content into
/// sibling blocks after the `tool_result`.
#[tokio::test]
async fn reference_results_displace_array_content_into_siblings() {
    let model = Model {
        id: "claude-opus-4-8".to_owned(),
        input: vec![Modality::Text, Modality::Image],
        ..common::anthropic_model()
    };
    let context = tool_marker_context(
        Some(vec!["late_tool".to_owned()]),
        vec![
            loaded_result(),
            ToolResultBlock::Image(pi_ai::types::ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ],
        vec![common::lookup_tool(), common::late_tool()],
    );

    let (payload, _) =
        capture_done_events(&model, &context, &common::minimal_anthropic_done()).await;

    let last_turn = &payload["messages"]
        .as_array()
        .expect("messages")
        .last()
        .expect("the tool-result turn")["content"];
    assert_eq!(last_turn[0]["type"], "tool_result");
    assert_eq!(last_turn[0]["content"][0]["type"], "tool_reference");
    assert_eq!(last_turn[1]["type"], "text");
    assert_eq!(last_turn[2]["type"], "image");
}

// ---------------------------------------------------------------------------
// Entries the registry reaches
// ---------------------------------------------------------------------------

#[test]
fn effort_display_and_parse_cover_the_vocabulary() {
    let spellings = [
        (AnthropicEffort::Low, "low"),
        (AnthropicEffort::Medium, "medium"),
        (AnthropicEffort::High, "high"),
        (AnthropicEffort::Xhigh, "xhigh"),
        (AnthropicEffort::Max, "max"),
    ];
    for (effort, spelling) in spellings {
        assert_eq!(effort.to_string(), spelling);
        assert_eq!(AnthropicEffort::try_from(spelling), Ok(effort));
    }
    assert!(AnthropicEffort::try_from("bogus").is_err());
}

/// Port-added: `stream_simple` without any credential fails through the
/// setup-error stream instead of dispatching.
#[tokio::test]
async fn stream_simple_without_a_credential_fails_before_dispatch() {
    let mock = MockHttpClient::new();
    let model = common::anthropic_model();
    let context = done_context();

    let result = stream_simple(&model, &context, None).result().await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("No API key for provider: anthropic")
    );
    assert_eq!(mock.recorded().len(), 0);
}

/// Port-added: the registry's uniform dispatch reaches the Anthropic
/// streams both ways.
#[tokio::test]
async fn the_registry_dispatch_reaches_the_anthropic_streams() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let model = common::anthropic_model();
    let context = done_context();
    let streams = pi_ai::api::anthropic_messages();

    let stream_result = streams
        .stream(
            &model,
            &context,
            Some(&pi_ai::types::StreamOptions {
                transport_options: mock_transport(&mock),
                api_key: Some("test-key".to_owned()),
                ..pi_ai::types::StreamOptions::default()
            }),
        )
        .result()
        .await;
    assert_eq!(stream_result.stop_reason, StopReason::Stop);

    let simple_result = streams
        .stream_simple(
            &model,
            &context,
            Some(&pi_ai::types::SimpleStreamOptions {
                transport_options: mock_transport(&mock),
                api_key: Some("test-key".to_owned()),
                ..pi_ai::types::SimpleStreamOptions::default()
            }),
        )
        .result()
        .await;
    assert_eq!(simple_result.stop_reason, StopReason::Stop);
}

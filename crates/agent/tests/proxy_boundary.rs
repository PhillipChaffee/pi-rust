//! Boundary suite for the proxy port: the branches upstream's
//! `test/proxy.test.ts` leaves untested, bound against the ported
//! restatements at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` — the
//! non-ok status messages, the request the proxy sends, the abort surface,
//! the line framing's skip and parse-failure arms, the text/thinking
//! translation arms, the streamed tool-call argument assembly, the
//! mismatch throws, the silently dropped `toolcall_end`, the sparse-index
//! filler, and the `done` arm's usage replacement.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod common;

use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use bytes::Bytes;
use common::empty_proxy_context;
use common::empty_proxy_usage;
use common::proxy_data_lines;
use common::proxy_done;
use common::proxy_event_type;
use common::proxy_model;
use common::proxy_options_with_body;
use common::proxy_toolcall_end;
use common::run_proxy;
use futures_core::Stream;
use pi_agent_core::ProxyAssistantMessageEvent;
use pi_agent_core::ProxyStreamOptions;
use pi_agent_core::stream_proxy;
use pi_ai::http::client::BoxHttpFuture;
use pi_ai::http::client::HttpByteStream;
use pi_ai::http::client::HttpClient;
use pi_ai::http::client::HttpError;
use pi_ai::http::client::HttpRequest;
use pi_ai::http::client::HttpResponse;
use pi_ai::http::mock::MockHttpClient;
use pi_ai::http::mock::MockResponse;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::types::ThinkingContent;
use pi_ai::types::ThinkingLevel;
use pi_ai::types::Transport;
use pi_ai::types::Usage;
use serde_json::json;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// Mount a mock client answering `response` on the proxy URL, returning the
/// client for the recorded-request assertions.
fn mounted_mock(response: MockResponse) -> MockHttpClient {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == "https://proxy.example.com/api/stream")
        .respond(response);
    mock
}

/// Proxy options sending through `client`.
fn options_with(client: Arc<dyn HttpClient>) -> ProxyStreamOptions {
    ProxyStreamOptions {
        http_client: Some(client),
        ..ProxyStreamOptions::new("test-token", "https://proxy.example.com")
    }
}

/// The failure message a drained stream ended on, the settled result's own
/// error message.
fn failure_message(result: &AssistantMessage) -> String {
    result
        .error_message
        .clone()
        .expect("the failure carries its message")
}

/// Run the proxy against `options` and assert the run ends on an error
/// event carrying `expected` with `reason`; returns the settled message.
async fn error_run(
    options: ProxyStreamOptions,
    expected: &str,
    reason: StopReason,
) -> AssistantMessage {
    let (_, result) = run_proxy(options).await;
    assert_eq!(result.stop_reason, reason);
    assert_eq!(failure_message(&result), expected);
    result
}

/// Mount a canned non-ok status and assert the error message it produces,
/// the three status tests' shared shape.
async fn status_error(status: u16, body: &str, expected: &str) {
    let mock = mounted_mock(MockResponse::status(status).with_body(body.to_owned()));
    error_run(options_with(Arc::new(mock)), expected, StopReason::Error).await;
}

/// Run the proxy and return the recorded request body parsed; the
/// request-shape tests share the drain-then-inspect tail.
async fn recorded_body(mock: &MockHttpClient, options: ProxyStreamOptions) -> serde_json::Value {
    let _ = run_proxy(options).await;
    let recorded = mock.recorded();
    assert_eq!(recorded.len(), 1);
    serde_json::from_slice(recorded[0].body.as_ref().expect("a request body"))
        .expect("the body parses")
}

#[tokio::test]
async fn a_non_ok_status_with_an_error_body_reports_the_body_error() {
    status_error(
        404,
        r#"{"error": "model not found"}"#,
        "Proxy error: model not found",
    )
    .await;
}

#[tokio::test]
async fn a_non_ok_status_with_json_lacking_an_error_reports_the_status() {
    // The seam carries no statusText; the IANA reason phrase stands in,
    // the Copilot OAuth helper's restatement.
    status_error(
        500,
        r#"{"detail": "boom"}"#,
        "Proxy error: 500 Internal Server Error",
    )
    .await;
}

#[tokio::test]
async fn a_non_ok_status_with_a_non_json_body_reports_the_status() {
    status_error(503, "service down", "Proxy error: 503 Service Unavailable").await;
}

#[tokio::test]
async fn the_request_carries_the_wire_shape_the_proxy_server_parses() {
    let mock = mounted_mock(MockResponse::status(200).with_body(proxy_data_lines(&[
        ProxyAssistantMessageEvent::Start,
        proxy_done(StopReason::Stop, None),
    ])));
    let body = recorded_body(&mock, options_with(Arc::new(mock.clone()))).await;

    let recorded = mock.recorded();
    let request = &recorded[0];
    assert!(matches!(request.method, pi_ai::http::HttpMethod::Post));
    assert_eq!(request.url, "https://proxy.example.com/api/stream");
    assert_eq!(
        request.headers,
        vec![
            ("Authorization".to_owned(), "Bearer test-token".to_owned()),
            ("Content-Type".to_owned(), "application/json".to_owned()),
        ]
    );
    assert_eq!(body["model"]["id"], "gpt-5.4");
    assert_eq!(body["model"]["baseUrl"], "https://api.openai.com/v1");
    assert_eq!(body["context"]["systemPrompt"], "");
    // Absent fields drop the way JSON.stringify drops undefined properties.
    assert_eq!(body["options"], json!({}));
}

#[tokio::test]
async fn the_request_body_options_carry_the_set_fields_under_wire_names() {
    let mock = mounted_mock(MockResponse::status(200).with_body(proxy_data_lines(&[
        ProxyAssistantMessageEvent::Start,
        proxy_done(StopReason::Stop, None),
    ])));
    let options = ProxyStreamOptions {
        temperature: Some(0.5),
        max_tokens: Some(1024),
        reasoning: Some(ThinkingLevel::Low),
        transport: Some(Transport::Sse),
        session_id: Some("session-1".to_owned()),
        max_retry_delay_ms: Some(250),
        ..options_with(Arc::new(mock.clone()))
    };

    let body = recorded_body(&mock, options).await;

    assert_eq!(
        body["options"],
        json!({
            "temperature": 0.5,
            "maxTokens": 1024,
            "reasoning": "low",
            "transport": "sse",
            "sessionId": "session-1",
            "maxRetryDelayMs": 250,
        })
    );
}

#[tokio::test]
async fn a_cancelled_signal_before_the_request_sends_nothing_and_reports_aborted() {
    let mock = mounted_mock(MockResponse::status(200).with_body("data: {}\n\n"));
    let token = CancellationToken::new();
    token.cancel();
    let options = ProxyStreamOptions {
        signal: Some(token),
        ..options_with(Arc::new(mock.clone()))
    };

    let (events, result) = run_proxy(options).await;

    assert_eq!(mock.request_count(), 0);
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error { .. })
    ));
    assert_eq!(result.stop_reason, StopReason::Aborted);
    // The seam's abort surfaces as its canonical message, upstream's
    // undici AbortError text restated to the port's seam vocabulary.
    assert_eq!(failure_message(&result), "The operation was aborted");
}

/// A client that answers with a canned single-chunk body regardless of the
/// request's cancellation — the harness the read-loop's aborted checks
/// need: the seam's clients surface cancellation as a read error, so the
/// checks' window (a chunk arriving while the signal is cancelled) needs a
/// client that yields the chunk anyway. The optional gate suspends the body
/// after its chunk so a test can cancel between the last chunk and the
/// loop's exit.
#[derive(Debug)]
struct CannedBodyClient {
    body: String,
    gate: Mutex<Option<oneshot::Receiver<()>>>,
}

impl CannedBodyClient {
    const fn ungated(body: String) -> Self {
        Self {
            body,
            gate: Mutex::new(None),
        }
    }

    const fn gated(body: String, gate: oneshot::Receiver<()>) -> Self {
        Self {
            body,
            gate: Mutex::new(Some(gate)),
        }
    }
}

impl HttpClient for CannedBodyClient {
    fn execute(&self, _request: HttpRequest) -> BoxHttpFuture<Result<HttpResponse, HttpError>> {
        let chunk = Bytes::from(self.body.clone());
        let gate = self.gate.lock().expect("gate lock").take();
        Box::pin(async move {
            Ok(HttpResponse {
                status: 200,
                headers: Vec::new(),
                body: HttpByteStream::new(GatedBody {
                    chunk: Some(chunk),
                    gate,
                }),
            })
        })
    }
}

/// A one-chunk body that suspends after the chunk until its gate releases,
/// then ends — the suspension is where the test cancels, landing the
/// cancellation between the last chunk read and the loop's exit.
#[derive(Debug)]
struct GatedBody {
    chunk: Option<Bytes>,
    gate: Option<oneshot::Receiver<()>>,
}

impl Stream for GatedBody {
    type Item = Result<Bytes, HttpError>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if let Some(chunk) = self.chunk.take() {
            return std::task::Poll::Ready(Some(Ok(chunk)));
        }
        let Some(gate) = self.gate.as_mut() else {
            return std::task::Poll::Ready(None);
        };
        match Pin::new(gate).poll(cx) {
            std::task::Poll::Ready(_) => std::task::Poll::Ready(None),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

#[tokio::test]
async fn a_chunk_read_while_the_signal_is_cancelled_reports_the_read_loop_abort() {
    let token = CancellationToken::new();
    token.cancel();
    let options = ProxyStreamOptions {
        signal: Some(token),
        ..options_with(Arc::new(CannedBodyClient::ungated(proxy_data_lines(&[
            ProxyAssistantMessageEvent::Start,
        ]))))
    };

    let (events, result) = run_proxy(options).await;

    // The check fires before the chunk's lines are processed: no start
    // event reaches the stream.
    let types: Vec<String> = events.iter().map(proxy_event_type).collect();
    assert_eq!(types, ["error"]);
    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq!(failure_message(&result), "Request aborted by user");
}

#[tokio::test]
async fn cancellation_between_the_last_chunk_and_the_loop_exit_reports_the_read_loop_abort() {
    let (gate_tx, gate_rx) = oneshot::channel::<()>();
    let token = CancellationToken::new();
    let options = ProxyStreamOptions {
        signal: Some(token.clone()),
        ..options_with(Arc::new(CannedBodyClient::gated(
            proxy_data_lines(&[ProxyAssistantMessageEvent::Start]),
            gate_rx,
        )))
    };

    let stream = stream_proxy(&proxy_model(), &empty_proxy_context(), &options);
    // The start event's arrival leaves the read loop suspended on the gate.
    let first = stream.next().await;
    assert!(
        matches!(first, Some(AssistantMessageEvent::Start { .. })),
        "the first event is the start"
    );
    token.cancel();
    let _ = gate_tx.send(());
    let mut events = vec![first.expect("a first event")];
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    let result = stream.result().await;

    let types: Vec<String> = events.iter().map(proxy_event_type).collect();
    assert_eq!(types, ["start", "error"]);
    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq!(failure_message(&result), "Request aborted by user");
}

#[tokio::test]
async fn a_malformed_data_line_reports_the_parse_failure() {
    // JSON.parse's SyntaxError message restates as the serde one: `{` at
    // the line start promises an object, so the non-string key fails there.
    error_run(
        proxy_options_with_body("data: {oops\n\n".to_owned()),
        "key must be a string at line 1 column 2",
        StopReason::Error,
    )
    .await;
}

#[tokio::test]
async fn lines_without_the_data_prefix_and_empty_payloads_are_skipped() {
    let body = "event: x\n: keep-alive\n\ndata: \n\n".to_owned()
        + &format!(
            "data: {}\n\n",
            serde_json::to_string(&ProxyAssistantMessageEvent::Done {
                reason: StopReason::Stop,
                usage: empty_proxy_usage(),
                provider_thinking_level: None,
            })
            .expect("a serializable proxy event")
        );
    let options = proxy_options_with_body(body);

    let (events, result) = run_proxy(options).await;

    let types: Vec<String> = events.iter().map(proxy_event_type).collect();
    assert_eq!(types, ["done"]);
    assert_eq!(result.stop_reason, StopReason::Stop);
}

#[tokio::test]
async fn the_text_and_thinking_arms_translate_with_signatures_and_authoritative_ends() {
    let proxy_events = vec![
        ProxyAssistantMessageEvent::Start,
        ProxyAssistantMessageEvent::ThinkingStart { content_index: 0 },
        ProxyAssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta: "step".to_owned(),
        },
        ProxyAssistantMessageEvent::ThinkingEnd {
            content_index: 0,
            content_signature: Some("sig-t".to_owned()),
        },
        ProxyAssistantMessageEvent::TextStart { content_index: 1 },
        ProxyAssistantMessageEvent::TextDelta {
            content_index: 1,
            delta: "hel".to_owned(),
        },
        ProxyAssistantMessageEvent::TextDelta {
            content_index: 1,
            delta: "lo".to_owned(),
        },
        ProxyAssistantMessageEvent::TextEnd {
            content_index: 1,
            content_signature: Some("sig-x".to_owned()),
        },
        proxy_done(StopReason::Stop, None),
    ];
    let options = proxy_options_with_body(proxy_data_lines(&proxy_events));

    let (events, result) = run_proxy(options).await;

    let types: Vec<String> = events.iter().map(proxy_event_type).collect();
    assert_eq!(
        types,
        [
            "start",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "text_start",
            "text_delta",
            "text_delta",
            "text_end",
            "done",
        ]
    );
    // The authoritative ends carry the full accumulated text.
    let thinking_end = events.iter().find_map(|event| match event {
        AssistantMessageEvent::ThinkingEnd { content, .. } => Some(content.clone()),
        _ => None,
    });
    assert_eq!(thinking_end.as_deref(), Some("step"));
    let text_end = events.iter().find_map(|event| match event {
        AssistantMessageEvent::TextEnd { content, .. } => Some(content.clone()),
        _ => None,
    });
    assert_eq!(text_end.as_deref(), Some("hello"));
    // The final message carries the signatures and the accumulated blocks.
    assert_eq!(
        result.content,
        vec![
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "step".to_owned(),
                thinking_signature: Some("sig-t".to_owned()),
                redacted: None,
            }),
            AssistantBlock::Text(TextContent {
                text: "hello".to_owned(),
                text_signature: Some("sig-x".to_owned()),
            }),
        ]
    );
}

#[tokio::test]
async fn streamed_tool_call_arguments_assemble_across_split_deltas() {
    let proxy_events = vec![
        ProxyAssistantMessageEvent::Start,
        ProxyAssistantMessageEvent::ToolcallStart {
            content_index: 0,
            id: "call_1".to_owned(),
            tool_name: "lookup".to_owned(),
        },
        ProxyAssistantMessageEvent::ToolcallDelta {
            content_index: 0,
            delta: r#"{"value":"hel"#.to_owned(),
        },
        ProxyAssistantMessageEvent::ToolcallDelta {
            content_index: 0,
            delta: r#"lo"}"#.to_owned(),
        },
        proxy_toolcall_end(
            0,
            json!({
                "type": "toolCall",
                "id": "call_1",
                "name": "lookup",
                "arguments": { "value": "hello", "extra": true },
                "thoughtSignature": "ts-1",
            }),
        ),
        proxy_done(StopReason::ToolUse, None),
    ];
    let options = proxy_options_with_body(proxy_data_lines(&proxy_events));

    let (events, result) = run_proxy(options).await;

    // Each delta re-parses the accumulated JSON into the call's arguments.
    let deltas: Vec<serde_json::Map<String, serde_json::Value>> = events
        .iter()
        .filter_map(|event| match event {
            AssistantMessageEvent::ToolcallDelta { partial, .. } => {
                Some(match &partial.content[0] {
                    AssistantBlock::ToolCall(call) => call.arguments.clone(),
                    _ => panic!("a tool call block"),
                })
            }
            _ => None,
        })
        .collect();
    // The partial-JSON salvage completes the first delta's unterminated
    // string, so the live arguments grow with each delta.
    assert_eq!(
        deltas[0].get("value").and_then(serde_json::Value::as_str),
        Some("hel")
    );
    assert_eq!(
        deltas[1].get("value").and_then(serde_json::Value::as_str),
        Some("hello")
    );
    // The wire call replaces the accumulated one whole, carrying the
    // thought signature the start never set.
    let AssistantBlock::ToolCall(call) = &result.content[0] else {
        panic!("a tool call block");
    };
    assert_eq!(call.thought_signature.as_deref(), Some("ts-1"));
    assert_eq!(
        call.arguments
            .get("extra")
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
}

/// The events one mismatch run scripts: start, a block opened at 0, then a
/// delta naming the wrong kind for that slot, upstream's three throw arms.
fn mismatch_options(
    opened: ProxyAssistantMessageEvent,
    wrong_delta: ProxyAssistantMessageEvent,
) -> ProxyStreamOptions {
    proxy_options_with_body(proxy_data_lines(&[
        ProxyAssistantMessageEvent::Start,
        opened,
        wrong_delta,
    ]))
}

#[tokio::test]
async fn delta_and_end_events_on_the_wrong_block_kind_error_verbatim() {
    error_run(
        mismatch_options(
            ProxyAssistantMessageEvent::TextStart { content_index: 0 },
            ProxyAssistantMessageEvent::ThinkingDelta {
                content_index: 0,
                delta: "step".to_owned(),
            },
        ),
        "Received thinking_delta for non-thinking content",
        StopReason::Error,
    )
    .await;

    error_run(
        mismatch_options(
            ProxyAssistantMessageEvent::ThinkingStart { content_index: 0 },
            ProxyAssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "text".to_owned(),
            },
        ),
        "Received text_delta for non-text content",
        StopReason::Error,
    )
    .await;

    error_run(
        mismatch_options(
            ProxyAssistantMessageEvent::TextStart { content_index: 0 },
            ProxyAssistantMessageEvent::ToolcallDelta {
                content_index: 0,
                delta: "{}".to_owned(),
            },
        ),
        "Received toolcall_delta for non-toolCall content",
        StopReason::Error,
    )
    .await;
}

#[tokio::test]
async fn a_toolcall_end_on_a_non_tool_call_slot_is_dropped_silently() {
    let options = proxy_options_with_body(proxy_data_lines(&[
        ProxyAssistantMessageEvent::Start,
        ProxyAssistantMessageEvent::TextStart { content_index: 0 },
        proxy_toolcall_end(
            0,
            json!({
                "type": "toolCall",
                "id": "call_1",
                "name": "lookup",
                "arguments": {},
            }),
        ),
        proxy_done(StopReason::Stop, None),
    ]));

    let (events, result) = run_proxy(options).await;

    let types: Vec<String> = events.iter().map(proxy_event_type).collect();
    assert_eq!(types, ["start", "text_start", "done"]);
    // The text block stands; the wire call never reached it.
    assert!(matches!(result.content[0], AssistantBlock::Text(_)));
}

#[tokio::test]
async fn a_content_index_past_the_end_fills_the_gap_with_empty_text_blocks() {
    let options = proxy_options_with_body(proxy_data_lines(&[
        ProxyAssistantMessageEvent::Start,
        ProxyAssistantMessageEvent::TextStart { content_index: 2 },
        ProxyAssistantMessageEvent::TextDelta {
            content_index: 2,
            delta: "late".to_owned(),
        },
        proxy_done(StopReason::Stop, None),
    ]));

    let (events, result) = run_proxy(options).await;

    let types: Vec<String> = events.iter().map(proxy_event_type).collect();
    assert_eq!(types, ["start", "text_start", "text_delta", "done"]);
    assert_eq!(result.content.len(), 3);
    let AssistantBlock::Text(empty_gap) = &result.content[0] else {
        panic!("a text filler block");
    };
    assert_eq!(empty_gap.text, "");
    let AssistantBlock::Text(assigned) = &result.content[2] else {
        panic!("the assigned block");
    };
    assert_eq!(assigned.text, "late");
}

#[tokio::test]
async fn the_done_event_replaces_the_accumulated_usage_and_thinking_level() {
    let usage: Usage = serde_json::from_value(json!({
        "input": 5, "output": 2, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 7,
        "cost": { "input": 0.1, "output": 0.2, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.3 },
    }))
    .expect("a well-formed usage");
    let options = proxy_options_with_body(proxy_data_lines(&[
        ProxyAssistantMessageEvent::Start,
        ProxyAssistantMessageEvent::Done {
            reason: StopReason::Length,
            usage,
            provider_thinking_level: Some("max".to_owned()),
        },
    ]));

    let (_, result) = run_proxy(options).await;

    assert_eq!(result.stop_reason, StopReason::Length);
    assert_eq!(result.usage, usage);
    assert_eq!(result.provider_thinking_level.as_deref(), Some("max"));
}

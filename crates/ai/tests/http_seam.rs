//! The HttpClient seam, the shared SSE decoder, and the seam-mock harness,
//! carrying the upstream raw-fetch patterns (`test/fetch-option.test.ts`) and
//! the SSE framing suite the substrate must satisfy
//! (`test/anthropic-sse-parsing.test.ts`'s body-building pattern), plus the
//! chunk-boundary hazard the protocol survey flags.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected shape panics by design"
)]

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use pi_ai::http::{
    HttpByteStream, HttpClient, HttpError, HttpMethod, HttpRequest, HttpResponse, MockBody,
    MockHttpClient, MockResponse, ServerSentEvent, SseDecoder, default_http_client, json_response,
    sse_response,
};
use pi_ai::types::{Model, TransportOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

fn static_response(response: String) -> &'static [u8] {
    Box::leak(response.into_bytes().into_boxed_slice())
}

/// A minimal `HTTP/1.1` loopback server: one connection, one response written
/// raw, then the socket closes so the client sees a clean end of body.
async fn spawn_http_server(response: &'static [u8]) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback bind");
    let address = listener
        .local_addr()
        .expect("loopback local addr")
        .to_string();

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("loopback accept");
        // Drain the request head and any body the client sends: this server
        // never inspects requests, it only answers once.
        let mut scratch = [0u8; 8192];
        let _ = socket.read(&mut scratch).await;
        socket
            .write_all(response)
            .await
            .expect("loopback response write");
        socket.shutdown().await.expect("loopback shutdown");
    });

    format!("http://{address}")
}

/// A loopback server that answers every connection with one response, for
/// tests that open several requests.
async fn spawn_http_server_multi(response: &'static [u8]) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback bind");
    let address = listener
        .local_addr()
        .expect("loopback local addr")
        .to_string();

    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let response = response;
            tokio::spawn(async move {
                let mut scratch = [0u8; 8192];
                let _ = socket.read(&mut scratch).await;
                let _ = socket.write_all(response).await;
                let _ = socket.shutdown().await;
            });
        }
    });

    format!("http://{address}")
}

/// The SSE byte payload upstream's `createSseResponse` helper builds for a
/// list of event/data pairs.
fn sse_bytes(events: &[(&str, &str)]) -> Vec<u8> {
    events
        .iter()
        .map(|(event, data)| format!("event: {event}\ndata: {data}\n"))
        .collect::<Vec<String>>()
        .join("\n")
        .into_bytes()
}

fn request(url: &str) -> HttpRequest {
    HttpRequest {
        method: HttpMethod::Get,
        url: url.to_owned(),
        headers: Vec::new(),
        body: None,
        timeout_ms: None,
        signal: CancellationToken::new(),
    }
}

fn test_model() -> Model {
    serde_json::from_value(serde_json::json!({
        "id": "test-model",
        "name": "Test Model",
        "api": "anthropic-messages",
        "provider": "test-provider",
        "baseUrl": "https://upstream.test/v1",
        "reasoning": false,
        "input": ["text"],
        "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0 },
        "contextWindow": 10_000,
        "maxTokens": 1_000
    }))
    .expect("model fixture parses")
}

// ---------------------------------------------------------------------------
// The seam: fetch-option semantics
// ---------------------------------------------------------------------------

#[tokio::test]
async fn injected_mock_serves_the_request_and_records_it() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url == "https://upstream.test/v1/messages")
        .respond(sse_response(200, &[("message_start", "{}")]));

    let response = mock
        .execute(HttpRequest {
            method: HttpMethod::Post,
            url: "https://upstream.test/v1/messages".to_owned(),
            headers: vec![("x-api-key".to_owned(), "test-key".to_owned())],
            body: Some(Bytes::from_static(b"{\"stream\":true}")),
            timeout_ms: None,
            signal: CancellationToken::new(),
        })
        .await
        .expect("mock route answers");

    assert_eq!(response.status, 200);
    assert_eq!(
        response
            .headers
            .iter()
            .find(|(name, _)| name == "content-type")
            .expect("sse content type"),
        &("content-type".to_owned(), "text/event-stream".to_owned())
    );

    let served = mock.recorded();
    assert_eq!(served.len(), 1);
    assert_eq!(served[0].method, HttpMethod::Post);
    assert_eq!(served[0].url, "https://upstream.test/v1/messages");
    assert_eq!(served[0].body.as_deref(), Some(&b"{\"stream\":true}"[..]));
}

#[tokio::test]
async fn no_route_matched_fails_like_ambient_fetch_must_not_be_called() {
    let mock = MockHttpClient::new();
    let error = mock
        .execute(request("https://upstream.test/v1/messages"))
        .await
        .expect_err("an unmatched mock rejects");

    match error {
        HttpError::Transport(message) => {
            assert!(message.contains("no mock route matched"), "{message}");
        }
        other => panic!("expected a transport error, got {other:?}"),
    }
}

#[tokio::test]
async fn response_sequence_answers_in_order_then_exhausts() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/retry"))
        .respond_sequence(vec![
            MockResponse::status(500).with_body(b"{\"error\":true}".to_vec()),
            MockResponse::status(200).with_body(b"ok".to_vec()),
        ]);

    let first = mock
        .execute(request("https://upstream.test/retry"))
        .await
        .expect("first");
    let second = mock
        .execute(request("https://upstream.test/retry"))
        .await
        .expect("second");
    let third = mock
        .execute(request("https://upstream.test/retry"))
        .await
        .expect_err("exhausted sequence rejects");

    assert_eq!(first.status, 500);
    assert_eq!(second.status, 200);
    match third {
        HttpError::Transport(message) => {
            assert!(message.contains("sequence is exhausted"), "{message}");
        }
        other => panic!("expected transport error, got {other:?}"),
    }
}

#[tokio::test]
async fn respond_fn_sees_the_request_and_can_fail_like_a_dead_server() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/dead"))
        .respond_fn(|_request| async {
            Err::<MockResponse, HttpError>(HttpError::Transport("fetch failed".to_owned()))
        });

    let error = mock
        .execute(request("https://upstream.test/dead"))
        .await
        .expect_err("handler failure surfaces");

    assert_eq!(error, HttpError::Transport("fetch failed".to_owned()));
    assert_eq!(mock.request_count(), 1);
}

#[tokio::test]
async fn abort_fails_reads_of_a_mock_body_mid_stream() {
    let token = CancellationToken::new();
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/live")).respond(
        MockResponse::status(200).with_chunks([
            b"event: a\ndata: 1\n".as_slice(),
            b"event: b\ndata: 2\n".as_slice(),
        ]),
    );

    let mut request = request("https://upstream.test/live");
    request.signal = token.clone();
    let mut body = mock.execute(request).await.expect("live response").body;
    body.next_chunk().await.expect("chunk one");
    token.cancel();
    let error = body.next_chunk().await.expect_err("abort mid-stream");
    assert_eq!(error, HttpError::Aborted);
}

#[test]
fn sse_response_bytes_match_upstreams_helper_shape() {
    let response = sse_response(
        200,
        &[("message_start", "{\"a\":1}"), ("message_stop", "{}")],
    );
    let MockBody::Bytes(body) = response.body else {
        panic!("sse_response carries whole bytes");
    };
    assert_eq!(
        body.as_ref(),
        &sse_bytes(&[("message_start", "{\"a\":1}"), ("message_stop", "{}")])
    );
    assert_eq!(response.status, 200);
    assert!(
        response
            .headers
            .iter()
            .any(|(name, value)| name == "content-type" && value == "text/event-stream")
    );
}

// ---------------------------------------------------------------------------
// TransportOptions: the fetch/signal/callback bundle on the option structs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn transport_options_resolve_the_injected_client_over_the_default() {
    use std::sync::Arc;

    let injected: Arc<dyn HttpClient> = Arc::new(MockHttpClient::new());

    let options = TransportOptions::default();
    assert!(Arc::ptr_eq(&options.client(), &default_http_client()));

    let owned = TransportOptions {
        http_client: Some(injected.clone()),
        ..TransportOptions::default()
    };
    assert!(Arc::ptr_eq(&owned.client(), &injected));
}

#[test]
fn transport_options_signal_defaults_to_an_uncancelled_token() {
    let options = TransportOptions::default();
    assert!(!options.signal().is_cancelled());

    let token = CancellationToken::new();
    let owned = TransportOptions {
        signal: Some(token.clone()),
        ..TransportOptions::default()
    };
    assert!(!owned.signal().is_cancelled());
    token.cancel();
    assert!(owned.signal().is_cancelled());
}

#[tokio::test]
async fn payload_and_response_hooks_round_trip() {
    use pi_ai::types::{OnPayload, OnResponse, ProviderRequestOptions};

    let captured = Arc::new(std::sync::Mutex::new(Vec::<serde_json::Value>::new()));
    let hook_payload = {
        let captured = Arc::clone(&captured);
        OnPayload::new(move |payload, _model| {
            let captured = Arc::clone(&captured);
            Box::pin(async move {
                captured
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(payload.clone());
                Some(serde_json::json!({"replaced": true}))
            })
        })
    };
    let responses_seen = Arc::new(std::sync::Mutex::new(Vec::<u16>::new()));
    let hook_response = {
        let responses_seen = Arc::clone(&responses_seen);
        OnResponse::new(move |response, _model| {
            let responses_seen = Arc::clone(&responses_seen);
            Box::pin(async move {
                responses_seen
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(response.status);
            })
        })
    };

    let mut options = ProviderRequestOptions::default();
    options.transport_options.on_payload = Some(hook_payload);
    options.transport_options.on_response = Some(hook_response);

    let payload_hook = options
        .transport_options
        .on_payload
        .as_ref()
        .expect("hook present");
    let replaced = payload_hook
        .call(serde_json::json!({"original": 1}), test_model())
        .await;
    assert_eq!(replaced, Some(serde_json::json!({"replaced": true})));

    let response_hook = options
        .transport_options
        .on_response
        .as_ref()
        .expect("hook present");
    response_hook
        .call(
            pi_ai::types::ProviderResponse {
                status: 200,
                headers: BTreeMap::default(),
            },
            test_model(),
        )
        .await;
    assert_eq!(
        responses_seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone(),
        vec![200]
    );
    assert_eq!(
        captured
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone(),
        vec![serde_json::json!({"original": 1})]
    );
    assert_eq!(
        format!("{:?}", options.transport_options.on_payload),
        "Some(OnPayload(..))"
    );
    assert_eq!(
        format!("{:?}", options.transport_options.on_response),
        "Some(OnResponse(..))"
    );
}

// ---------------------------------------------------------------------------
// The SSE decoder: upstream framing semantics
// ---------------------------------------------------------------------------

fn decode_all(body: &[u8]) -> Vec<ServerSentEvent> {
    let mut decoder = SseDecoder::default();
    let mut events = decoder.feed(body);
    events.extend(decoder.finish());
    events
}

#[test]
fn decodes_event_and_data_fields_with_multi_line_data_joined() {
    let events = decode_all(b"event: message_start\ndata: {\"a\":\ndata:1}\n\n");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event.as_deref(), Some("message_start"));
    assert_eq!(events[0].data, "{\"a\":\n1}");
    assert_eq!(
        events[0].raw,
        vec![
            "event: message_start".to_owned(),
            "data: {\"a\":".to_owned(),
            "data:1}".to_owned(),
        ]
    );
}

#[test]
fn comment_lines_and_unknown_fields_are_ignored_but_recorded() {
    let events = decode_all(b": keep-alive\ndata: one\nid: 42\nevent: ping\n\n");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event.as_deref(), Some("ping"));
    assert_eq!(events[0].data, "one");
    assert_eq!(
        events[0].raw,
        vec![
            ": keep-alive".to_owned(),
            "data: one".to_owned(),
            "id: 42".to_owned(),
            "event: ping".to_owned(),
        ]
    );
}

#[test]
fn data_only_frames_carry_no_event_and_value_space_stripping_follows_the_wire() {
    let events = decode_all(b"data: {\"a\":1}\ndata: [DONE]\n\n");
    // Two data lines without a blank line join into one event, per the wire
    // grammar and upstream's flush.
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event, None);
    assert_eq!(events[0].data, "{\"a\":1}\n[DONE]");

    // One space after the colon is stripped; two survive as one.
    let spaced = decode_all(b"data:  x\n\n");
    assert_eq!(spaced[0].data, " x");

    // A field with no colon carries an empty value.
    let bare = decode_all(b"data\n\n");
    assert_eq!(bare.len(), 1);
    assert_eq!(bare[0].data, "");
}

#[test]
fn crlf_and_bare_cr_line_breaks_frame_the_same_events() {
    let crlf = decode_all(b"event: a\r\ndata: 1\r\n\r\nevent: b\r\ndata: 2\r\n\r\n");
    assert_eq!(crlf.len(), 2);
    assert_eq!(crlf[0].data, "1");
    assert_eq!(crlf[1].data, "2");

    let cr = decode_all(b"event: a\rdata: 1\r\rdata: 2\r");
    // Each bare CR ends a line, so the empty line between the data lines
    // flushes the first event: two events total.
    assert_eq!(cr.len(), 2);
    assert_eq!(cr[0].data, "1");
    assert_eq!(cr[1].data, "2");
}

#[test]
fn an_event_without_data_flushes_at_the_blank_line() {
    let events = decode_all(b"event: ping\n\n");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event.as_deref(), Some("ping"));
    assert_eq!(events[0].data, "");
}

#[test]
fn end_of_stream_flushes_the_residual_partial_line_and_trailing_event() {
    // No trailing blank line: the last event flushes at finish().
    let events = decode_all(b"event: message_stop\ndata: {}");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event.as_deref(), Some("message_stop"));

    // A partial line with no terminator still decodes at finish().
    let tail = decode_all(b"data: ok");
    assert_eq!(tail.len(), 1);
    assert_eq!(tail[0].data, "ok");
}

#[test]
fn multi_byte_utf8_split_across_chunks_decodes_whole_code_points() {
    let payload = "data: héllo wörld ✓\n\n".as_bytes();
    let mut decoder = SseDecoder::default();
    let mut events = Vec::new();
    // Feed one byte at a time; é, ö, and ✓ are multi-byte code points.
    for byte in payload {
        events.extend(decoder.feed(std::slice::from_ref(byte)));
    }
    events.extend(decoder.finish());
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, "héllo wörld ✓");
}

#[test]
fn feeding_byte_by_byte_across_every_boundary_yields_the_whole_body_events() {
    let payload = sse_bytes(&[
        ("message_start", "{\"type\":\"message_start\"}"),
        (
            "content_block_delta",
            "{\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}",
        ),
        ("message_stop", "{\"type\":\"message_stop\"}"),
    ]);
    let expected = decode_all(&payload);

    // Every split point, one chunk each side.
    for split in 0..payload.len() {
        let mut decoder = SseDecoder::default();
        let mut events = decoder.feed(&payload[..split]);
        events.extend(decoder.feed(&payload[split..]));
        events.extend(decoder.finish());
        assert_eq!(events, expected, "split at {split}");
    }

    // Byte-at-a-time feeding matches too.
    let mut decoder = SseDecoder::default();
    let mut events = Vec::new();
    for byte in &payload {
        events.extend(decoder.feed(std::slice::from_ref(byte)));
    }
    events.extend(decoder.finish());
    assert_eq!(events, expected);
}

#[test]
fn invalid_utf8_in_a_line_becomes_replacement_characters_like_textdecoder() {
    let events = decode_all(b"data: \xff\xfe\n\n");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, "\u{fffd}\u{fffd}");
}

#[tokio::test]
async fn sse_stream_decodes_off_chunked_mock_bodies_and_reports_body_errors() {
    let mock = MockHttpClient::new();
    let payload = sse_bytes(&[("ping", "{}"), ("ping", "{}")]);
    let chunks: Vec<Vec<u8>> = payload.chunks(7).map(<[u8]>::to_vec).collect();
    mock.on(|request| request.url.contains("/sse"))
        .respond(MockResponse::status(200).with_chunks(chunks));

    let response = mock
        .execute(request("https://upstream.test/sse"))
        .await
        .expect("route answers");
    let events = pi_ai::http::collect_sse(response.body)
        .await
        .expect("chunk boundaries never break the decoder");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].event.as_deref(), Some("ping"));
    assert_eq!(events[1].event.as_deref(), Some("ping"));

    // A body error terminates the stream with that error.
    let error_stream = HttpByteStream::from_chunks(vec![
        Ok(Bytes::from_static(b"data: 1\n")),
        Err(HttpError::Transport("connection reset".to_owned())),
    ]);
    let result = pi_ai::http::collect_sse(error_stream).await;
    assert_eq!(
        result.expect_err("error propagates"),
        HttpError::Transport("connection reset".to_owned())
    );
}

#[tokio::test]
async fn sse_stream_reports_abort_mid_frame() {
    let token = CancellationToken::new();
    let mock = MockHttpClient::new();
    mock.on(|_| true).respond(
        MockResponse::status(200)
            .with_chunks([b"event: a\ndata: {\"partial".to_vec(), b"\n\n".to_vec()]),
    );
    let mut request = request("https://upstream.test/abort");
    request.signal = token.clone();
    let mut body = mock.execute(request).await.expect("response").body;
    body.next_chunk().await.expect("first chunk");
    token.cancel();
    let error = pi_ai::http::collect_sse(body).await.expect_err("aborted");
    assert_eq!(error, HttpError::Aborted);
}

// ---------------------------------------------------------------------------
// Timeouts and aborts against the injected (tokio) clock
// ---------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn request_timeout_expires_against_the_paused_clock() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/slow"))
        .respond_fn(|_request| async {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            Ok(MockResponse::status(200).with_body(b"late".to_vec()))
        });

    let mut request = request("https://upstream.test/slow");
    request.timeout_ms = Some(1_000);
    let runner = tokio::spawn(mock.execute(request));
    tokio::time::advance(std::time::Duration::from_secs(30)).await;
    let result = runner.await.expect("task completes");

    match result {
        Err(HttpError::Timeout) => {}
        other => panic!("expected timeout, got {other:?}"),
    }
}

#[tokio::test(start_paused = true)]
async fn abort_beats_a_pending_timeout() {
    let token = CancellationToken::new();
    let mock = MockHttpClient::new();
    mock.on(|_| true).respond_fn(|_request| async {
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        Ok::<_, HttpError>(MockResponse::status(200))
    });

    let mut request = request("https://upstream.test/slow");
    request.timeout_ms = Some(10_000);
    request.signal = token.clone();

    let runner = tokio::spawn(mock.execute(request));
    tokio::time::advance(std::time::Duration::from_millis(100)).await;
    token.cancel();
    let result = runner
        .await
        .expect("task completes")
        .expect_err("abort wins");
    assert_eq!(result, HttpError::Aborted);
}

// ---------------------------------------------------------------------------
// The process-default reqwest client against a real loopback socket
// ---------------------------------------------------------------------------

#[tokio::test]
async fn default_client_serves_headers_body_and_status_over_the_wire() {
    let url = spawn_http_server(static_response(
        "HTTP/1.1 201 Created\r\ncontent-type: text/plain\r\nx-dup: a\r\nx-dup: b\r\nconnection: close\r\n\r\nhello seam".to_owned(),
    ))
    .await;

    let response = default_http_client()
        .execute(HttpRequest {
            method: HttpMethod::Get,
            url,
            headers: vec![("x-probe".to_owned(), "pi".to_owned())],
            body: None,
            timeout_ms: None,
            signal: CancellationToken::new(),
        })
        .await
        .expect("loopback answers");

    assert_eq!(response.status, 201);
    assert!(
        response
            .headers
            .iter()
            .any(|(name, value)| name == "content-type" && value == "text/plain")
    );
    let mut body = Vec::new();
    let mut stream = response.body;
    while let Some(chunk) = stream.next_chunk().await.expect("body chunk") {
        body.extend_from_slice(&chunk);
    }
    assert_eq!(body, b"hello seam");
}

#[tokio::test]
async fn reqwest_streams_sse_across_real_wire_chunks() {
    let payload = sse_bytes(&[
        ("message_start", "{\"type\":\"message_start\"}"),
        ("content_block_delta", "{\"delta\":{\"text\":\"wire\"}}"),
        ("message_stop", "{}"),
    ]);
    let (first, second) = payload.split_at(payload.len() / 3);
    let url = spawn_http_server(static_response(format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n{}{}",
        std::str::from_utf8(first).expect("sse ascii"),
        std::str::from_utf8(second).expect("sse ascii"),
    )))
    .await;

    let response = default_http_client()
        .execute(request(&url))
        .await
        .expect("response");
    let events = pi_ai::http::collect_sse(response.body)
        .await
        .expect("decodes across transport chunk boundaries");
    assert_eq!(events.len(), 3);
    assert_eq!(events[2].event.as_deref(), Some("message_stop"));
}

#[tokio::test]
async fn error_statuses_resolve_like_fetch_with_their_json_body() {
    let mock = MockHttpClient::new();
    mock.on(|request| request.method == HttpMethod::Post)
        .respond(json_response(
            401,
            &serde_json::json!({"error": {"message": "nope"}}),
        ));

    let response = mock
        .execute(HttpRequest {
            method: HttpMethod::Post,
            url: "https://upstream.test/v1/messages".to_owned(),
            headers: vec![("authorization".to_owned(), "Bearer k".to_owned())],
            body: Some(Bytes::from_static(b"{}")),
            timeout_ms: None,
            signal: CancellationToken::new(),
        })
        .await
        .expect("401 resolves like fetch");

    assert_eq!(response.status, 401);
    let mut text = Vec::new();
    let mut body = response.body;
    while let Some(chunk) = body.next_chunk().await.expect("body") {
        text.extend_from_slice(&chunk);
    }
    let parsed: serde_json::Value = serde_json::from_slice(&text).expect("json body");
    assert_eq!(parsed["error"]["message"], "nope");
}

#[tokio::test(start_paused = true)]
async fn reqwest_request_timeout_fires_on_a_silent_socket() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    tokio::spawn(async move {
        // Accept and hold the socket open without writing a response.
        let (_socket, _) = listener.accept().await.expect("accept");
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
    });

    let mut request = request(&format!("http://{address}/"));
    request.timeout_ms = Some(2_000);
    let runner = tokio::spawn(async move { default_http_client().execute(request).await });
    tokio::time::advance(std::time::Duration::from_secs(60)).await;
    let result = runner.await.expect("task ends");

    match result {
        Err(HttpError::Timeout) => {}
        other => panic!("expected timeout, got {other:?}"),
    }
}

#[tokio::test]
async fn reqwest_body_reads_fail_aborted_when_the_request_cancels() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let _ = socket.read(&mut [0u8; 4096]).await;
        let head = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\nevent: a\ndata: 1\n";
        let _ = socket.write_all(head).await;
        // Hold the socket open; the abort must not wait for it.
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
    });

    let token = CancellationToken::new();
    let mut request = request(&format!("http://{address}/"));
    request.signal = token.clone();
    let mut body = default_http_client()
        .execute(request)
        .await
        .expect("headers arrive")
        .body;
    // Read whatever arrives, then cancel and require the next read to fail.
    let _ = body.next_chunk().await;
    token.cancel();
    let error = body.next_chunk().await.expect_err("abort mid-body");
    assert_eq!(error, HttpError::Aborted);
}
// ---------------------------------------------------------------------------
// Seam surface: the shapes adapters and mocks rely on
// ---------------------------------------------------------------------------

#[test]
fn http_method_spells_every_variant_the_wire_expects() {
    use pi_ai::http::HttpMethod;

    assert_eq!(HttpMethod::Get.as_str(), "GET");
    assert_eq!(HttpMethod::Post.as_str(), "POST");
    assert_eq!(HttpMethod::Put.as_str(), "PUT");
    assert_eq!(HttpMethod::Delete.as_str(), "DELETE");
    assert_eq!(HttpMethod::Patch.as_str(), "PATCH");
    assert_eq!(HttpMethod::Head.as_str(), "HEAD");
    assert_eq!(HttpMethod::Custom("purge".to_owned()).as_str(), "purge");

    assert_eq!(HttpMethod::Post.to_string(), "POST");
    assert_eq!(HttpMethod::Custom("purge".to_owned()).to_string(), "purge");
    // Clone and Debug ride every request through the seam.
    let method = HttpMethod::Custom("purge".to_owned());
    assert_eq!(method.clone(), method);
    assert!(format!("{method:?}").contains("purge"));
}

#[test]
fn http_errors_display_the_messages_the_retry_classifiers_match_on() {
    let cases = [
        (HttpError::Aborted, "The operation was aborted"),
        (HttpError::Timeout, "request timed out"),
        (
            HttpError::Transport("fetch failed".to_owned()),
            "fetch failed",
        ),
        (
            HttpError::InvalidUrl("not a url".to_owned()),
            "invalid URL: not a url",
        ),
    ];
    for (error, message) in cases {
        assert_eq!(error.to_string(), message);
        // The seam's errors participate in error handling as real errors.
        let _: &dyn std::error::Error = &error;
    }
}

#[test]
fn seam_shapes_clone_and_debug_like_their_callers_need() {
    let request = request("https://upstream.test/v1");
    let cloned = request.clone();
    assert_eq!(cloned.url, request.url);

    let byte_stream = HttpByteStream::from_chunks(vec![Ok(Bytes::new())]);
    assert_eq!(format!("{byte_stream:?}"), "HttpByteStream(..)");

    let response = HttpResponse {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: byte_stream_fixture(),
    };
    let debugged = format!("{response:?}");
    assert!(debugged.contains("200"), "{debugged}");
    assert!(debugged.contains("event-stream"), "{debugged}");
}

fn byte_stream_fixture() -> HttpByteStream {
    HttpByteStream::from_chunks(vec![Ok(Bytes::from_static(b"chunk"))])
}

#[tokio::test]
async fn reqwest_rejects_invalid_headers_and_methods_with_transport_errors() {
    let client = pi_ai::http::ReqwestHttpClient::new().expect("default build");

    let mut invalid_header = request("https://upstream.test/v1");
    invalid_header.headers = vec![("bad header name".to_owned(), "value".to_owned())];
    let error = client
        .execute(invalid_header)
        .await
        .expect_err("invalid header rejects");
    assert!(
        matches!(error, HttpError::Transport(message) if message.contains("invalid header name"))
    );

    let mut invalid_value = request("https://upstream.test/v1");
    invalid_value.headers = vec![("x-api-key".to_owned(), "bad\nvalue".to_owned())];
    let error = client
        .execute(invalid_value)
        .await
        .expect_err("invalid value");
    assert!(
        matches!(error, HttpError::Transport(message) if message.contains("invalid header value"))
    );

    let mut invalid_method = request("https://upstream.test/v1");
    invalid_method.method = HttpMethod::Custom("not a method".to_owned());
    let error = client
        .execute(invalid_method)
        .await
        .expect_err("invalid method");
    assert!(matches!(error, HttpError::Transport(message) if message.contains("invalid method")));
}

// The stalled body cannot run under the paused clock: tokio auto-advances
// pending timers whenever every task blocks, firing the request timeout
// before the loopback writes its headers. Real time bounds this test at the
// one-second timeout.
#[tokio::test]
async fn reqwest_total_timeout_covers_a_stalled_body() {
    // Headers arrive, the body stalls: the total request timeout surfaces on
    // the next body read, the SDK semantics for streaming responses.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let _ = socket.read(&mut [0u8; 4096]).await;
        let head = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\nevent: a\ndata: 1\n";
        let _ = socket.write_all(head).await;
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
    });

    let mut request = request(&format!("http://{address}/"));
    request.timeout_ms = Some(1_000);
    let mut body = default_http_client()
        .execute(request)
        .await
        .expect("headers arrive before the timeout")
        .body;
    let first = body.next_chunk().await.expect("body begins");
    assert!(first.is_some());

    let result = body.next_chunk().await.expect_err("stalled body times out");
    assert_eq!(result, HttpError::Timeout);
}

#[tokio::test]
async fn reqwest_serves_every_method_variant_the_seam_carries() {
    let url = spawn_http_server_multi(static_response(
        "HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok".to_owned(),
    ))
    .await;

    for method in [
        HttpMethod::Put,
        HttpMethod::Delete,
        HttpMethod::Patch,
        HttpMethod::Head,
        HttpMethod::Custom("purge".to_owned()),
    ] {
        let mut request = request(&url);
        request.method = method.clone();
        let response = default_http_client()
            .execute(request)
            .await
            .unwrap_or_else(|error| panic!("{method} failed: {error}"));
        assert_eq!(response.status, 200, "{method}");
    }
}

#[tokio::test]
async fn http_byte_stream_polls_through_its_stream_impl() {
    use futures_util::StreamExt;

    let mut stream = HttpByteStream::from_chunks(vec![
        Ok(Bytes::from_static(b"one")),
        Ok(Bytes::from_static(b"two")),
    ]);
    let mut collected = Vec::new();
    while let Some(chunk) = stream.next().await {
        collected.extend_from_slice(chunk.expect("chunk").as_ref());
    }
    assert_eq!(collected, b"onetwo");
}

#[tokio::test]
async fn a_mock_response_without_a_body_reads_as_end_of_stream() {
    let mock = MockHttpClient::new();
    mock.on(|_| true).respond(MockResponse::status(204));
    let mut response = mock
        .execute(request("https://upstream.test/empty"))
        .await
        .expect("no-content response");
    assert_eq!(response.body.next_chunk().await.expect("empty body"), None);
}

#[test]
fn mock_builder_debug_prints_without_leaking_matchers() {
    let mock = MockHttpClient::new();
    let builder = mock.on(|_| true);
    assert_eq!(format!("{builder:?}"), "MockRouteBuilder(..)");
    builder.respond(MockResponse::status(200));
}

#[tokio::test]
async fn a_pre_cancelled_token_rejects_before_any_route_runs() {
    let token = CancellationToken::new();
    let mock = MockHttpClient::new();
    mock.on(|_| true).respond(MockResponse::status(200));

    let mut request = request("https://upstream.test/v1");
    request.signal = token.clone();
    token.cancel();
    let error = mock
        .execute(request)
        .await
        .expect_err("pre-cancelled rejects");
    assert_eq!(error, HttpError::Aborted);
    assert_eq!(mock.request_count(), 0, "nothing ran");
}

#[tokio::test]
async fn reqwest_carries_a_post_body_to_the_wire() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    let (body_tx, body_rx) = tokio::sync::oneshot::channel::<Vec<u8>>();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        // The client holds the connection open awaiting a response, so read
        // until the request head and body have landed rather than to EOF.
        let mut buffer = Vec::new();
        let mut scratch = [0u8; 1024];
        loop {
            let read = socket.read(&mut scratch).await.unwrap_or(0);
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&scratch[..read]);
            if buffer.ends_with(b"\"stream\":true}".as_slice()) {
                break;
            }
        }
        let _ = body_tx.send(buffer);
        let _ = socket
            .write_all(b"HTTP/1.1 204 No Content\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
            .await;
        let _ = socket.shutdown().await;
    });

    let response = default_http_client()
        .execute(HttpRequest {
            method: HttpMethod::Post,
            url: format!("http://{address}/v1/messages"),
            headers: vec![("content-type".to_owned(), "application/json".to_owned())],
            body: Some(Bytes::from_static(b"{\"stream\":true}")),
            timeout_ms: Some(5_000),
            signal: CancellationToken::new(),
        })
        .await
        .expect("the server answers");
    assert_eq!(response.status, 204);

    let sent = body_rx.await.expect("request bytes reach the server");
    let text = String::from_utf8_lossy(&sent).to_string();
    assert!(text.contains("POST /v1/messages"), "{text}");
    assert!(text.contains("{\"stream\":true}"), "{text}");
}

#[tokio::test]
async fn reqwest_rejects_a_pre_cancelled_token_without_sending() {
    let token = CancellationToken::new();
    token.cancel();
    let mut request = request("https://upstream.test/v1");
    request.signal = token;
    let error = default_http_client()
        .execute(request)
        .await
        .expect_err("pre-cancelled request rejects");
    assert_eq!(error, HttpError::Aborted);
}

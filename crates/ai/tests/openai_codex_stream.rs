//! The OpenAI Codex Responses wire-API stream suite, ported 1:1 from
//! `packages/ai/test/openai-codex-stream.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Seam adaptations over the vitest originals, each standing in for a `vi.*`
//! stub the JS runtime offered:
//!
//! - `vi.stubGlobal("WebSocket", MockWebSocket)` rides
//!   `openai_codex_responses::set_websocket_transport` with the crate's
//!   `MockWebSocketTransport`; the peer side plays the upstream
//!   `MockWebSocket` server. The mock's JS `WebSocket.onerror` event ports
//!   to a server-side close, the seam's transport-failure shape.
//! - `vi.stubGlobal("fetch", ...)` rides the `TransportOptions::http_client`
//!   seam. The github release/prompt routes the upstream fetch mocks answer
//!   are vestigial — the module under test never fetches them — so the port
//!   mounts only the `/codex/responses` route; an unmounted mock fetch fails
//!   loudly the way upstream's fallback throws.
//! - `vi.useFakeTimers` rides `#[tokio::test(start_paused = true)]`. Upstream
//!   pins retry delays by spying on `setTimeout`; the port measures the
//!   virtual clock between mock requests, the same fact read off the timer
//!   the paused clock actually fires.
//! - `vi.setSystemTime` has no Rust seam (`now_ms` reads the wall clock), so
//!   the connection-age case it drives is not ported; the HTTP-date
//!   `Retry-After` case builds its date from `now_ms()` at handler time so
//!   the 45s distance survives unmocked.
//! - The per-test `PI_CODING_AGENT_DIR` temp-dir setup is dropped: the Rust
//!   stream path reads no environment.
//! - The mock's zstd SSE request bodies decode through a hand parse of the
//!   stored-block frame the module emits — the pinned dependency set has no
//!   zstd crate — the test-side port of node's `zstdDecompressSync`.
//! - Upstream's `buildSSEPayload` carries the `end_turn` and
//!   `incomplete_details: null` keys its inline SSE fixtures omit;
//!   `JSON.stringify` drops the undefined one and the wire reads both
//!   spellings the same.
//!
//! Not ported from the upstream suite: the connection-age case (needs
//! `vi.setSystemTime`; no Rust seam without a module change). The module's
//! usage-limit catch guard (`!message.contains("usage limit")`) has no case
//! in the pinned upstream suite — every retry case there rides the
//! 429/`Retry-After` paths — so there is no 1:1 source case to port.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected shape panics by design"
)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use bytes::Bytes;
use pi_ai::api::openai_codex_responses::{self, CodexToolChoice, OpenAiCodexResponsesOptions};
use pi_ai::http::mock::RecordedRequest;
use pi_ai::http::{
    BoxHttpFuture, HttpByteStream, HttpClient, HttpError, HttpRequest, HttpResponse,
    MockHttpClient, MockResponse, MockWebSocketPeer, MockWebSocketTransport, WebSocketConnection,
    WebSocketError, WebSocketMessage, WebSocketOutbound, WebSocketRequest, WebSocketTransport,
    json_response,
};
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, AssistantMessageEvent, BoxedFuture, CacheRetention,
    ConstrainedSamplingConfig, ConstrainedSamplingSetting, Context, GrammarFormat, Message,
    Modality, Model, ModelCompat, ModelCost, ModelCostRates, ModelThinkingLevel, ProviderId,
    SimpleStreamOptions, StopReason, Strictness, ThinkingLevel, Tool, ToolResultBlock,
    ToolResultMessage, Transport, TransportOptions, UserContent, UserMessage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_ai::utils::pi_user_agent::get_pi_user_agent;
use serde_json::{Value, json};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

mod common;
use common::{
    drain_and_settle, mock_transport, payload_capture, user_message_at, user_message_now,
};

// ---------------------------------------------------------------------------
// Suite lifecycle
// ---------------------------------------------------------------------------

/// Hold the suite lock and carry the `afterEach` cleanup. The vitest file
/// runs its cases sequentially and unstub-all-globals plus
/// `closeOpenAICodexWebSocketSessions` between them; the module's transport
/// override, session cache, and debug stats are process-global here, so both
/// edges of every test reset them under one suite mutex.
fn suite() -> SuiteGuard {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_codex_global_state();
    SuiteGuard { _lock: guard }
}

struct SuiteGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl Drop for SuiteGuard {
    fn drop(&mut self) {
        reset_codex_global_state();
    }
}

fn reset_codex_global_state() {
    openai_codex_responses::set_websocket_transport(None);
    openai_codex_responses::close_openai_codex_websocket_sessions(None);
    openai_codex_responses::reset_openai_codex_websocket_debug_stats(None);
}

/// Wire the codex WebSocket mock harness: the suite lock, the account token,
/// the registered mock transport, the unmounted fetch, and the standard
/// model, the prefix every transport case repeats.
fn ws_suite() -> (
    SuiteGuard,
    String,
    Arc<MockWebSocketTransport>,
    MockHttpClient,
    Model,
) {
    let guard = suite();
    let transport = Arc::new(MockWebSocketTransport::new());
    openai_codex_responses::set_websocket_transport(Some(transport.clone()));
    (
        guard,
        mock_token("acc_test"),
        transport,
        MockHttpClient::new(),
        codex_model("gpt-5.1-codex", "GPT-5.1 Codex"),
    )
}

/// The harness of [`ws_suite`] with the SSE fallback route mounted — the
/// prefix every fallback-riding transport case repeats.
fn ws_fallback_suite() -> (
    SuiteGuard,
    String,
    Arc<MockWebSocketTransport>,
    MockHttpClient,
    Model,
) {
    let (guard, token, transport, fetch, model) = ws_suite();
    mount_sse_route(&fetch, hello_sse_body());
    (guard, token, transport, fetch, model)
}

/// The uncached-WebSocket options, the shape the frame-shape cases send.
fn plain_ws_options(fetch: &MockHttpClient, token: &str) -> OpenAiCodexResponsesOptions {
    let mut options = websocket_options(fetch, token, Transport::Websocket);
    options.cache_retention = Some(CacheRetention::None);
    options
}

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// The three-part JWT the account id rides, upstream's `mockToken`.
fn mock_token(account_id: &str) -> String {
    let payload =
        json!({ "https://api.openai.com/auth": { "chatgpt_account_id": account_id } }).to_string();
    format!("aaa.{}.bbb", base64_encode(payload.as_bytes()))
}

/// Standard base64 with padding, the form `Buffer.toString("base64")` emits
/// that the module's lenient `decode_base64_segment` reads.
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in input.chunks(3) {
        let bytes = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let packed = (u32::from(bytes[0]) << 16) | (u32::from(bytes[1]) << 8) | u32::from(bytes[2]);
        out.push(ALPHABET[(packed >> 18) as usize & 63] as char);
        out.push(ALPHABET[(packed >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(packed >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[packed as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// The codex-shaped model the cases stream against, upstream's inline
/// `Model<"openai-codex-responses">` fixtures.
fn codex_model(id: &str, name: &str) -> Model {
    Model {
        id: id.to_owned(),
        name: name.to_owned(),
        api: Api::from("openai-codex-responses"),
        provider: ProviderId::from("openai-codex"),
        base_url: "https://chatgpt.com/backend-api".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: ModelCost {
            rates: ModelCostRates::default(),
            tiers: None,
        },
        context_window: 400_000,
        max_tokens: 128_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The "Say hello" context most SSE cases send, upstream's inline `Context`
/// fixtures with `timestamp: Date.now()`.
fn codex_context_now() -> Context {
    Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![user_message_now("Say hello")],
        tools: None,
    }
}

/// The "Say hello" context the WebSocket cases send with the fixed
/// `timestamp: 1` the upstream websocket fixtures carry.
fn codex_context_at(timestamp: i64) -> Context {
    Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![user_message_at("Say hello", timestamp)],
        tools: None,
    }
}

/// The empty-context shape the websocket-cache cases send, upstream's
/// `{ systemPrompt: "", messages: [] }`.
const fn codex_context_empty() -> Context {
    Context {
        system_prompt: Some(String::new()),
        messages: Vec::new(),
        tools: None,
    }
}

/// The keyed SSE options the fetch-stub cases send.
fn sse_options(mock: &MockHttpClient, token: &str) -> OpenAiCodexResponsesOptions {
    OpenAiCodexResponsesOptions {
        transport_options: mock_transport(mock),
        api_key: Some(token.to_owned()),
        transport: Some(Transport::Sse),
        ..OpenAiCodexResponsesOptions::default()
    }
}

/// The WebSocket-transport options: an unmounted mock fetch rides the
/// `http_client` seam so any HTTP dispatch fails loudly, upstream's
/// `"unexpected fetch"` 500 stub.
fn websocket_options(
    fetch: &MockHttpClient,
    token: &str,
    transport: Transport,
) -> OpenAiCodexResponsesOptions {
    OpenAiCodexResponsesOptions {
        transport_options: TransportOptions {
            http_client: Some(Arc::new(fetch.clone())),
            ..TransportOptions::default()
        },
        api_key: Some(token.to_owned()),
        transport: Some(transport),
        ..OpenAiCodexResponsesOptions::default()
    }
}

/// The cached-turn options: the `WebSocketCached` transport with the long
/// retention and the named session, the shape every cached-turn case sends.
fn cached_ws_options(
    fetch: &MockHttpClient,
    token: &str,
    session: &str,
) -> OpenAiCodexResponsesOptions {
    let mut options = websocket_options(fetch, token, Transport::WebsocketCached);
    options.cache_retention = Some(CacheRetention::Long);
    options.session_id = Some(session.to_owned());
    options
}

/// Stream the hello frames with the given extras and settle the run, the
/// mock mounted for the request-count assertions, the shape the
/// injected-frame cases share.
async fn framed_run(token: &str, extra: &[Value]) -> (AssistantMessage, MockHttpClient) {
    let mock = MockHttpClient::new();
    let mut frames = hello_frames();
    frames.extend(extra.iter().cloned());
    mount_sse_route(&mock, sse_body(&frames, false));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = sse_options(&mock, token);
    (sse_run(&model, &options).await, mock)
}

/// Stream one uncached WebSocket turn and take the accepted peer for the
/// peer-side assertions, the shape the frame-shape cases share.
async fn plain_turn(
    transport: &MockWebSocketTransport,
    model: &Model,
    options: &OpenAiCodexResponsesOptions,
) -> (AssistantMessageEventStream, MockWebSocketPeer) {
    let stream = openai_codex_responses::stream(model, &codex_context_at(1), Some(options));
    let mut peer = next_peer(transport).await;
    let _ = next_sent_body(&mut peer).await;
    (stream, peer)
}

/// Settle the hello stream on a spawned task, the shape the paused-clock
/// retry cases need: the result waiter registers before the stream can
/// complete, upstream's `result()` promise racing the retry clock.
fn spawned_sse_run(
    model: &Model,
    options: &OpenAiCodexResponsesOptions,
) -> tokio::task::JoinHandle<AssistantMessage> {
    let stream = openai_codex_responses::stream(model, &codex_context_now(), Some(options));
    tokio::spawn(async move { drain_and_settle(&stream).await })
}

/// The SSE options over the never-closing body client, the shape the
/// open-body terminal cases send.
fn open_body_options(token: &str, payload: Bytes) -> OpenAiCodexResponsesOptions {
    OpenAiCodexResponsesOptions {
        transport_options: TransportOptions {
            http_client: Some(Arc::new(OpenEndedClient { payload })),
            ..TransportOptions::default()
        },
        api_key: Some(token.to_owned()),
        transport: Some(Transport::Sse),
        ..OpenAiCodexResponsesOptions::default()
    }
}

/// Mount the SSE route the fetch-stub cases answer with.
fn mount_sse_route(mock: &MockHttpClient, body: String) {
    mock.on(|request| request.url.contains("/codex/responses"))
        .respond(
            MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_body(body),
        );
}

/// Stream the hello context over SSE and settle the final message, the
/// one-line form the fetch-stub cases share, upstream's `result()` await.
async fn sse_run(model: &Model, options: &OpenAiCodexResponsesOptions) -> AssistantMessage {
    drain_and_settle(&openai_codex_responses::stream(
        model,
        &codex_context_now(),
        Some(options),
    ))
    .await
}

/// Stream one cached-WebSocket turn: take the accepted peer, replay its
/// request frame, answer with the completed response, and settle,
/// upstream's per-turn `processWebSocketStream` call.
async fn cached_turn(
    transport: &MockWebSocketTransport,
    model: &Model,
    options: &OpenAiCodexResponsesOptions,
    timestamp: i64,
    response_id: usize,
) -> AssistantMessage {
    let stream = openai_codex_responses::stream(model, &codex_context_at(timestamp), Some(options));
    let mut peer = next_peer(transport).await;
    let _ = next_sent_body(&mut peer).await;
    peer_sends(&peer, &[completed_response(response_id)]);
    drain_and_settle(&stream).await
}

// ---------------------------------------------------------------------------
// SSE body fixtures
// ---------------------------------------------------------------------------

/// One `data:` frame per event, the body shape every SSE fixture joins.
fn sse_body(frames: &[Value], trailing_done: bool) -> String {
    let mut body: Vec<String> = frames
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect();
    if trailing_done {
        body.push("data: [DONE]\n\n".to_owned());
    }
    body.concat()
}

/// The message-open frame the codex SSE fixtures stream first.
fn message_added() -> Value {
    json!({
        "type": "response.output_item.added",
        "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
    })
}

/// The part-open frame the delta events ride behind.
fn part_added() -> Value {
    json!({ "type": "response.content_part.added", "part": { "type": "output_text", "text": "" } })
}

/// The text delta the SSE body streams.
fn text_delta(delta: &str) -> Value {
    json!({ "type": "response.output_text.delta", "delta": delta })
}

/// The authoritative message close, the item status the finish reads.
fn message_done(id: &str, text: &str) -> Value {
    json!({
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "id": id,
            "role": "assistant",
            "status": "completed",
            "content": [{ "type": "output_text", "text": text }],
        },
    })
}

/// The hello-run frames: the message opens, the part opens, the "Hello"
/// delta lands, and the message closes authoritatively — the four events
/// every completed codex SSE fixture repeats ahead of its terminal.
fn hello_frames() -> Vec<Value> {
    vec![
        message_added(),
        part_added(),
        text_delta("Hello"),
        message_done("msg_1", "Hello"),
    ]
}

/// The completed-terminal response the inline fixtures and
/// `buildSSEPayload({ status: "completed" })` both stream.
fn completed_terminal(end_turn: Option<bool>, service_tier: Option<&str>) -> Value {
    let mut response = json!({
        "status": "completed",
        "incomplete_details": Value::Null,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
            "input_tokens_details": { "cached_tokens": 0 },
        },
    });
    if let Some(end_turn) = end_turn {
        response["end_turn"] = json!(end_turn);
    }
    if let Some(service_tier) = service_tier {
        response["service_tier"] = json!(service_tier);
    }
    json!({ "type": "response.completed", "response": response })
}

/// The completed hello run most SSE cases stream.
fn hello_sse_body() -> String {
    let mut frames = hello_frames();
    frames.push(completed_terminal(None, None));
    sse_body(&frames, false)
}

/// The service-tier terminal the pricing cases stream: a `"default"` echo
/// over 1M/1M usage.
fn service_tier_sse_body() -> String {
    let mut frames = hello_frames();
    frames.push(json!({
        "type": "response.completed",
        "response": {
            "status": "completed",
            "service_tier": "default",
            "usage": {
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "total_tokens": 2_000_000,
                "input_tokens_details": { "cached_tokens": 0 },
            },
        },
    }));
    sse_body(&frames, false)
}

/// Upstream's `buildSSEPayload`: the hello run whose terminal carries the
/// given status, the `data: [DONE]` sentinel when `include_done` holds, and
/// the `end_turn` field only when defined, exactly like `JSON.stringify`
/// drops the undefined key.
fn build_sse_payload(status: &str, include_done: bool, end_turn: Option<bool>) -> String {
    let (terminal_type, incomplete_details) = if status == "incomplete" {
        (
            "response.incomplete",
            json!({ "reason": "max_output_tokens" }),
        )
    } else {
        ("response.completed", Value::Null)
    };
    let mut terminal_response = json!({
        "status": status,
        "incomplete_details": incomplete_details,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
            "input_tokens_details": { "cached_tokens": 0 },
        },
    });
    if let Some(end_turn) = end_turn {
        terminal_response["end_turn"] = json!(end_turn);
    }
    let mut frames = hello_frames();
    frames.push(json!({ "type": terminal_type, "response": terminal_response }));
    if include_done {
        frames.push(json!("[DONE]"));
    }
    sse_body(&frames, false)
}

// ---------------------------------------------------------------------------
// Request-capture and event helpers
// ---------------------------------------------------------------------------

/// Read a recorded request header case-insensitively, like `Headers.get`.
fn request_header<'a>(request: &'a RecordedRequest, name: &str) -> Option<&'a str> {
    request
        .headers
        .iter()
        .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

/// Decode an SSE request body, upstream's `decodeCodexRequestBody`: the
/// stored-block zstd frame the module emits decodes by hand where node rides
/// `zstdDecompressSync`; anything else is JSON.
fn decode_codex_request_body(body: Option<&Bytes>) -> Option<Value> {
    let body = body?;
    if body.len() < 9 || body[..4] != [0x28, 0xB5, 0x2F, 0xFD] {
        return serde_json::from_slice(body).ok();
    }
    // Frame header: magic, the single-segment descriptor, the 4-byte frame
    // content size, then raw blocks whose 3-byte headers carry the size.
    let mut offset = 9;
    let mut content = Vec::new();
    loop {
        let header = u32::from(body[offset])
            | (u32::from(body[offset + 1]) << 8)
            | (u32::from(body[offset + 2]) << 16);
        let last = header & 1 == 1;
        let size = usize::try_from((header >> 3) & 0x1F_FFFF).ok()?;
        offset += 3;
        let end = offset.checked_add(size)?;
        content.extend_from_slice(body.get(offset..end)?);
        offset = end;
        if last {
            break;
        }
    }
    serde_json::from_slice(&content).ok()
}

/// The decoded JSON body of the recorded SSE request at `index`.
fn recorded_codex_body(mock: &MockHttpClient, index: usize) -> Value {
    let request = &mock.recorded()[index];
    decode_codex_request_body(request.body.as_ref()).expect("the recorded request body decodes")
}

/// The text of an assistant message's first text block, upstream's
/// `content.find((c) => c.type === "text")?.text`.
fn message_text(message: &AssistantMessage) -> Option<&str> {
    message.content.iter().find_map(|block| match block {
        AssistantBlock::Text(text) => Some(text.text.as_str()),
        _ => None,
    })
}

/// The event's wire tag, upstream's `event.type` collection: text deltas
/// carry their payload the way the abort case asserts.
fn event_tag(event: &AssistantMessageEvent) -> String {
    match event {
        AssistantMessageEvent::Start { .. } => "start".to_owned(),
        AssistantMessageEvent::TextStart { .. } => "text_start".to_owned(),
        AssistantMessageEvent::TextDelta { delta, .. } => format!("text_delta:{delta}"),
        AssistantMessageEvent::TextEnd { .. } => "text_end".to_owned(),
        AssistantMessageEvent::ThinkingStart { .. } => "thinking_start".to_owned(),
        AssistantMessageEvent::ThinkingDelta { delta, .. } => format!("thinking_delta:{delta}"),
        AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end".to_owned(),
        AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start".to_owned(),
        AssistantMessageEvent::ToolcallDelta { delta, .. } => format!("toolcall_delta:{delta}"),
        AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end".to_owned(),
        AssistantMessageEvent::Done { .. } => "done".to_owned(),
        AssistantMessageEvent::Error { .. } => "error".to_owned(),
    }
}

/// Drain the event stream collecting wire tags and settle the final message,
/// upstream's `for await (const event of ...)` + `result()` pair.
async fn drain_tags(stream: &AssistantMessageEventStream) -> (Vec<String>, AssistantMessage) {
    let (tags, message) = tokio::join!(
        async {
            let mut tags = Vec::new();
            while let Some(event) = stream.next().await {
                tags.push(event_tag(&event));
            }
            tags
        },
        stream.result(),
    );
    (tags, message)
}

// ---------------------------------------------------------------------------
// Seam clients
// ---------------------------------------------------------------------------

/// The SSE response whose body never sends EOF, upstream's `ReadableStream`
/// whose start enqueues without closing.
struct OpenEndedClient {
    payload: Bytes,
}

impl std::fmt::Debug for OpenEndedClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OpenEndedClient(..)")
    }
}

impl HttpClient for OpenEndedClient {
    fn execute(&self, _request: HttpRequest) -> BoxHttpFuture<Result<HttpResponse, HttpError>> {
        let payload = self.payload.clone();
        Box::pin(async move {
            Ok(HttpResponse {
                status: 200,
                headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
                body: HttpByteStream::new(futures_util::stream::unfold(
                    Some(payload),
                    |payload: Option<Bytes>| async move {
                        match payload {
                            Some(bytes) => Some((Ok(bytes), None)),
                            None => std::future::pending().await,
                        }
                    },
                )),
            })
        })
    }
}

/// The phases of the abort-test body, upstream's `ReadableStream` whose start
/// enqueues on timers and whose cancel ends delivery.
enum TimedPhase {
    /// The first chunk rides at once, like the start hook's enqueue.
    First(Vec<Bytes>),
    /// Later chunks wait the gap, racing the request's cancellation.
    Waiting(Vec<Bytes>),
    /// The abort ended delivery; the body hangs.
    Aborted,
}

/// The abort-test client: the first chunk at once, then the later chunks
/// after the gap — unless the request cancels first, which fails the read
/// aborted, the port of `ReadableStream.cancel` ending delivery.
struct TimedChunksClient {
    chunks: Vec<Bytes>,
    gap_ms: u64,
}

impl std::fmt::Debug for TimedChunksClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TimedChunksClient(..)")
    }
}

impl HttpClient for TimedChunksClient {
    fn execute(&self, request: HttpRequest) -> BoxHttpFuture<Result<HttpResponse, HttpError>> {
        let gap = Duration::from_millis(self.gap_ms);
        // The cancellation token rides the unfold state: each poll takes it
        // out and hands it back, so the closure stays callable.
        let init = (TimedPhase::First(self.chunks.clone()), request.signal);
        Box::pin(async move {
            Ok(HttpResponse {
                status: 200,
                headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
                body: HttpByteStream::new(futures_util::stream::unfold(
                    init,
                    move |(phase, signal)| async move {
                        match phase {
                            TimedPhase::First(mut chunks) => {
                                let first = chunks.remove(0);
                                Some((Ok(first), (TimedPhase::Waiting(chunks), signal)))
                            }
                            TimedPhase::Waiting(mut chunks) => {
                                if chunks.is_empty() {
                                    return std::future::pending().await;
                                }
                                tokio::select! {
                                    () = tokio::time::sleep(gap) => {
                                        let next = chunks.remove(0);
                                        Some((Ok(next), (TimedPhase::Waiting(chunks), signal)))
                                    }
                                    () = signal.cancelled() => {
                                        Some((Err(HttpError::Aborted), (TimedPhase::Aborted, signal)))
                                    }
                                }
                            }
                            TimedPhase::Aborted => std::future::pending().await,
                        }
                    },
                )),
            })
        })
    }
}

// ---------------------------------------------------------------------------
// WebSocket peer helpers
// ---------------------------------------------------------------------------

/// Take the next accepted mock connection, yielding until the connect under
/// test registers it.
async fn next_peer(transport: &MockWebSocketTransport) -> MockWebSocketPeer {
    loop {
        if let Some(peer) = transport.next_peer() {
            return peer;
        }
        tokio::task::yield_now().await;
    }
}

/// The request frame the client side sent, decoded from the text message.
async fn next_sent_body(peer: &mut MockWebSocketPeer) -> Value {
    let sent = peer
        .next_sent()
        .await
        .expect("the connection sends a request frame");
    let WebSocketOutbound::Message(WebSocketMessage::Text(text)) = sent else {
        panic!("expected a text send, got {sent:?}");
    };
    serde_json::from_str(&text).expect("the frame body is JSON")
}

/// Push the peer's reply messages, the mock's `dispatch("message", ...)`.
fn peer_sends(peer: &MockWebSocketPeer, events: &[Value]) {
    for event in events {
        peer.send_text(event.to_string())
            .expect("push to the client side");
    }
}

/// The four-event success run the continuation-recovery cases replay:
/// created, added, done with the text, completed with usage 5/3/8.
fn recovery_success_events(response_id: &str, message_id: &str, text: &str) -> Vec<Value> {
    vec![
        json!({ "type": "response.created", "response": { "id": response_id } }),
        json!({ "type": "response.output_item.added", "output_index": 0,
            "item": { "type": "message", "id": message_id, "role": "assistant", "status": "in_progress", "content": [] } }),
        json!({ "type": "response.output_item.done", "output_index": 0,
            "item": { "type": "message", "id": message_id, "role": "assistant", "status": "completed",
                "content": [{ "type": "output_text", "text": text }] } }),
        json!({ "type": "response.completed",
            "response": { "id": response_id, "status": "completed",
                "usage": { "input_tokens": 5, "output_tokens": 3, "total_tokens": 8 } } }),
    ]
}

/// The completed response the plain success replies send.
fn completed_response(response_id: usize) -> Value {
    json!({
        "type": "response.completed",
        "response": {
            "id": format!("resp_{response_id}"),
            "status": "completed",
            "usage": { "input_tokens": 5, "output_tokens": 3, "total_tokens": 8 },
        },
    })
}

// ===========================================================================
// SSE-transport cases
// ===========================================================================

/// Streams SSE responses into `AssistantMessageEventStream`: the event ride,
/// the forced header set, and the settled text.
#[tokio::test]
async fn streams_sse_responses_into_assistant_message_event_stream() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = sse_options(&mock, &token);

    let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
    let mut saw_text_delta = false;
    let mut saw_done = false;
    while let Some(event) = stream.next().await {
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            saw_text_delta = true;
        }
        if let AssistantMessageEvent::Done { message, .. } = event {
            saw_done = true;
            assert_eq!(message_text(&message), Some("Hello"));
        }
    }

    assert!(saw_text_delta);
    assert!(saw_done);
    assert_eq!(mock.request_count(), 1);
    let request = &mock.recorded()[0];
    assert_eq!(
        request_header(request, "Authorization"),
        Some(format!("Bearer {token}").as_str())
    );
    assert_eq!(
        request_header(request, "chatgpt-account-id"),
        Some("acc_test")
    );
    assert_eq!(
        request_header(request, "OpenAI-Beta"),
        Some("responses=experimental")
    );
    assert_eq!(request_header(request, "originator"), Some("pi"));
    assert_eq!(
        request_header(request, "User-Agent"),
        Some(get_pi_user_agent().as_str())
    );
    assert_eq!(request_header(request, "accept"), Some("text/event-stream"));
    assert!(request_header(request, "x-api-key").is_none());
}

/// Regression test for <https://github.com/earendil-works/pi/issues/9047>: a
/// terminal SSE event without a trailing blank line still completes.
#[tokio::test]
async fn processes_a_terminal_sse_event_without_a_trailing_blank_line() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let body = build_sse_payload("completed", false, None);
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, body.trim_end().to_owned());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = sse_options(&mock, &token);
    let context = codex_context_now();

    let result = drain_and_settle(&openai_codex_responses::stream(
        &model,
        &context,
        Some(&options),
    ))
    .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(message_text(&result), Some("Hello"));
}

/// The terminal event completes the stream even though the body never sends
/// EOF; upstream's `Promise.race` guard rides a real 1s deadline here.
#[tokio::test]
async fn completes_after_response_completed_even_when_the_sse_body_stays_open() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let payload = Bytes::from(build_sse_payload("completed", true, Some(false)));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = open_body_options(&token, payload);

    let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
    let result = tokio::time::timeout(Duration::from_millis(1000), drain_and_settle(&stream))
        .await
        .expect("Timed out waiting for completed SSE stream");

    assert_eq!(message_text(&result), Some("Hello"));
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.end_turn, Some(false));
}

/// The `response.incomplete` terminal maps to the `length` stop reason under
/// the same open-body guard.
#[tokio::test]
async fn maps_response_incomplete_to_stop_reason_length_even_when_the_sse_body_stays_open() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let payload = Bytes::from(build_sse_payload("incomplete", false, None));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = open_body_options(&token, payload);

    let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
    let result = tokio::time::timeout(Duration::from_millis(1000), drain_and_settle(&stream))
        .await
        .expect("Timed out waiting for incomplete SSE stream");

    assert_eq!(message_text(&result), Some("Hello"));
    assert_eq!(result.stop_reason, StopReason::Length);
}

/// The header timeout aborts the fetch when response headers never arrive,
/// surfacing the exact message the timeout wiring composes.
#[tokio::test(start_paused = true)]
async fn aborts_sse_fetch_after_the_configured_http_timeout_when_response_headers_do_not_arrive() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/codex/responses"))
        .respond_fn(|_request| async {
            // The headers never arrive; the mock's request timeout races this
            // hang the way the real transport races a pending response.
            tokio::time::sleep(Duration::from_secs(30)).await;
            Err(HttpError::Timeout)
        });
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.timeout_ms = Some(10);

    let result = sse_run(&model, &options).await;

    assert_eq!(mock.request_count(), 1);
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Codex SSE response headers timed out after 10ms")
    );
}

/// Aborting after the first delta stops the read before the queued "two"
/// delta can ride out; the abort-shaped body failure is the observable form
/// of the upstream `ReadableStream`'s cancel hook.
#[tokio::test(start_paused = true)]
async fn aborts_sse_body_reads_after_response_headers_arrive() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let first = sse_body(&[message_added(), part_added(), text_delta("one")], false);
    let second = sse_body(&[text_delta("two")], false);
    let terminal = sse_body(
        &[
            message_done("msg_1", "onetwo"),
            completed_terminal(None, None),
        ],
        false,
    );
    let cancel = CancellationToken::new();
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = OpenAiCodexResponsesOptions {
        transport_options: TransportOptions {
            http_client: Some(Arc::new(TimedChunksClient {
                chunks: vec![
                    Bytes::from(first),
                    Bytes::from(second),
                    Bytes::from(terminal),
                ],
                gap_ms: 10,
            })),
            signal: Some(cancel.clone()),
            ..TransportOptions::default()
        },
        api_key: Some(token),
        transport: Some(Transport::Sse),
        ..OpenAiCodexResponsesOptions::default()
    };

    let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
    let (events, result) = tokio::join!(
        async {
            let mut events = Vec::new();
            while let Some(event) = stream.next().await {
                let tag = event_tag(&event);
                if tag == "text_delta:one" {
                    cancel.cancel();
                }
                events.push(tag);
            }
            events
        },
        stream.result(),
    );

    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq!(result.error_message.as_deref(), Some("Request was aborted"));
    assert!(events.contains(&"text_delta:one".to_owned()));
    assert!(!events.contains(&"text_delta:two".to_owned()));
}

/// A provided session id rides the `session-id`/`x-client-request-id` header
/// pair and the request body's `prompt_cache_key`.
#[tokio::test]
async fn sets_session_id_x_client_request_id_headers_and_prompt_cache_key_when_session_id_is_provided()
 {
    let _guard = suite();
    let token = mock_token("acc_test");
    let session_id = "test-session-123";
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.session_id = Some(session_id.to_owned());

    let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let request = &mock.recorded()[0];
    assert_eq!(request_header(request, "session-id"), Some(session_id));
    assert!(request_header(request, "session_id").is_none());
    assert_eq!(
        request_header(request, "x-client-request-id"),
        Some(session_id)
    );
    let body = decode_codex_request_body(request.body.as_ref()).expect("the request body decodes");
    assert_eq!(body.get("prompt_cache_key"), Some(&json!(session_id)));
}

/// `cacheRetention: "none"` drops the session-affinity headers and the
/// `prompt_cache_key` even with a session id provided.
#[tokio::test]
async fn omits_sse_cache_affinity_when_cache_retention_is_none() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.cache_retention = Some(CacheRetention::None);
    options.session_id = Some("one-off-summary".to_owned());

    let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let request = &mock.recorded()[0];
    assert!(request_header(request, "session-id").is_none());
    assert!(request_header(request, "x-client-request-id").is_none());
    let body = decode_codex_request_body(request.body.as_ref()).expect("the request body decodes");
    assert_eq!(body.get("prompt_cache_key"), None);
}

/// The `onPayload` hook sees the `prompt_cache_key` clamped to OpenAI's
/// 64-character limit.
#[tokio::test]
async fn clamps_prompt_cache_key_to_openai_s_64_character_limit() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let session_id = "x".repeat(67);
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let (on_payload, captured) = payload_capture();
    let mut options = sse_options(&mock, &token);
    options.session_id = Some(session_id);
    options.transport_options.on_payload = Some(on_payload);

    let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let payload = common::captured_payload(&captured);
    assert_eq!(
        payload.get("prompt_cache_key"),
        Some(&json!("x".repeat(64)))
    );
}

/// The `session-id` and `x-client-request-id` headers clamp to 64 characters
/// with the session id.
#[tokio::test]
async fn clamps_codex_session_id_header_to_64_characters() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let session_id = "x".repeat(67);
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.session_id = Some(session_id);

    let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let request = &mock.recorded()[0];
    assert_eq!(
        request_header(request, "session-id"),
        Some("x".repeat(64).as_str())
    );
    assert_eq!(
        request_header(request, "x-client-request-id"),
        Some("x".repeat(64).as_str())
    );
}

/// `streamSimple` preserves the model's `xhigh` thinking-level mapping.
#[tokio::test]
async fn preserves_gpt_5_5_xhigh_reasoning_effort_from_simple_options() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let mut model = codex_model("gpt-5.5", "GPT-5.5");
    model.thinking_level_map = Some(BTreeMap::from([(
        ModelThinkingLevel::Xhigh,
        Some("xhigh".to_owned()),
    )]));
    let options = SimpleStreamOptions {
        transport_options: mock_transport(&mock),
        api_key: Some(token),
        reasoning: Some(ThinkingLevel::Xhigh),
        transport: Some(Transport::Sse),
        ..SimpleStreamOptions::default()
    };

    let stream =
        openai_codex_responses::stream_simple(&model, &codex_context_now(), Some(&options));
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let body = recorded_codex_body(&mock, 0);
    assert_eq!(
        body.get("reasoning"),
        Some(&json!({ "effort": "xhigh", "summary": "auto" }))
    );
}

/// An explicit `required` tool choice rides the request body.
#[tokio::test]
async fn forwards_required_tool_choice() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.5", "GPT-5.5");
    let context = Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("Do not call ping. Respond with text instead.".to_owned()),
            timestamp: 1,
        })],
        tools: Some(vec![Tool {
            name: "ping".to_owned(),
            description: "Ping".to_owned(),
            parameters: json!({ "type": "object", "properties": { "value": { "type": "string" } } }),
            constrained_sampling: None,
        }]),
    };
    let mut options = sse_options(&mock, &token);
    options.tool_choice = Some(CodexToolChoice::Required);

    let stream = openai_codex_responses::stream(&model, &context, Some(&options));
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let body = recorded_codex_body(&mock, 0);
    assert_eq!(body.get("tool_choice"), Some(&json!("required")));
}

/// Codex strict mode is set explicitly per tool: the disabled constrained
/// sampling carries `strict: null`, the prefer config carries `strict: true`.
#[tokio::test]
async fn sets_codex_strict_mode_explicitly_and_honors_constrained_sampling() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.5", "GPT-5.5");
    let context = Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("Use a tool".to_owned()),
            timestamp: 1,
        })],
        tools: Some(vec![
            Tool {
                name: "optional".to_owned(),
                description: "Optional constrained sampling".to_owned(),
                parameters: json!({ "type": "object", "properties": { "value": { "type": "string" } } }),
                constrained_sampling: Some(ConstrainedSamplingSetting::Disabled(false)),
            },
            Tool {
                name: "strict".to_owned(),
                description: "Strict constrained sampling".to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"],
                    "additionalProperties": false,
                }),
                constrained_sampling: Some(ConstrainedSamplingSetting::Config(
                    ConstrainedSamplingConfig::JsonSchema {
                        strict: Strictness::Prefer,
                    },
                )),
            },
        ]),
    };
    let (on_payload, captured) = payload_capture();
    let mut options = sse_options(&mock, &token);
    options.transport_options.on_payload = Some(on_payload);

    let stream = openai_codex_responses::stream(&model, &context, Some(&options));
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let payload = common::captured_payload(&captured);
    let tools = payload
        .get("tools")
        .and_then(Value::as_array)
        .expect("the request carries the tools");
    assert_eq!(tools[0].get("type"), Some(&json!("function")));
    assert_eq!(tools[0].get("name"), Some(&json!("optional")));
    assert_eq!(tools[0].get("strict"), Some(&Value::Null));
    assert_eq!(tools[1].get("type"), Some(&json!("function")));
    assert_eq!(tools[1].get("name"), Some(&json!("strict")));
    assert_eq!(tools[1].get("strict"), Some(&json!(true)));
}

/// The `minimal` reasoning effort clamps through the model's
/// `thinkingLevelMap` on every codex model id the upstream suite names.
#[tokio::test]
async fn clamps_minimal_reasoning_effort_to_low() {
    let _guard = suite();
    for model_id in ["gpt-5.3-codex", "gpt-5.4", "gpt-5.5"] {
        let token = mock_token("acc_test");
        let mock = MockHttpClient::new();
        mount_sse_route(&mock, hello_sse_body());
        let mut model = codex_model(model_id, model_id);
        model.thinking_level_map = Some(BTreeMap::from([(
            ModelThinkingLevel::Minimal,
            Some("low".to_owned()),
        )]));
        let mut options = sse_options(&mock, &token);
        options.reasoning_effort = Some(ModelThinkingLevel::Minimal);

        let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
        let result = drain_and_settle(&stream).await;
        assert_eq!(result.stop_reason, StopReason::Stop);

        let body = recorded_codex_body(&mock, 0);
        assert_eq!(
            body.get("reasoning"),
            Some(&json!({ "effort": "low", "summary": "auto" })),
            "model {model_id}"
        );
    }
}

/// The client-sent service tier prices the usage when the response echoes
/// `"default"`, upstream's four-tier pricing matrix.
#[tokio::test]
async fn uses_the_client_sent_service_tier_when_codex_echoes_default() {
    let _guard = suite();
    let cases: [(&str, &str, f64); 4] = [
        ("gpt-5.1-codex", "flex", 0.5),
        ("gpt-5.1-codex", "priority", 2.0),
        ("gpt-5.5", "flex", 0.5),
        ("gpt-5.5", "priority", 2.5),
    ];
    for (model_id, service_tier, multiplier) in cases {
        let token = mock_token("acc_test");
        let mock = MockHttpClient::new();
        mount_sse_route(&mock, service_tier_sse_body());
        let name = if model_id == "gpt-5.5" {
            "GPT-5.5"
        } else {
            "GPT-5.1 Codex"
        };
        let mut model = codex_model(model_id, name);
        model.cost.rates = ModelCostRates {
            input: 1.0,
            output: 2.0,
            cache_read: 0.0,
            cache_write: 0.0,
        };
        let mut options = sse_options(&mock, &token);
        options.service_tier = Some(service_tier.to_owned());

        let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
        let result = drain_and_settle(&stream).await;
        assert_eq!(result.stop_reason, StopReason::Stop, "model {model_id}");

        // The rates divide by 1e6 before the token multiply, the Rust order
        // of `calculate_cost`, so the comparisons ride a float tolerance.
        let expected_input = multiplier;
        let expected_output = 2.0 * multiplier;
        let expected_total = 3.0 * multiplier;
        let cost = &result.usage.cost;
        assert!(
            (cost.input - expected_input).abs() < 1e-9,
            "model {model_id} input: {}",
            cost.input
        );
        assert!(
            (cost.output - expected_output).abs() < 1e-9,
            "model {model_id} output: {}",
            cost.output
        );
        assert!(
            (cost.total - expected_total).abs() < 1e-9,
            "model {model_id} total: {}",
            cost.total
        );
    }
}

/// No session id means no session-affinity headers.
#[tokio::test]
async fn does_not_set_session_id_headers_when_session_id_is_not_provided() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = sse_options(&mock, &token);

    let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let request = &mock.recorded()[0];
    assert!(request_header(request, "session-id").is_none());
    assert!(request_header(request, "session_id").is_none());
    assert!(request_header(request, "x-client-request-id").is_none());
}

// ===========================================================================
// WebSocket-transport cases
// ===========================================================================

/// The connect request's handshake header value, spelled exactly as the
/// transport received it (the record `connectWebSocket` builds).
fn connect_header<'a>(connect: &'a WebSocketRequest, name: &str) -> Option<&'a str> {
    connect
        .headers
        .iter()
        .find(|(candidate, _)| candidate == name)
        .map(|(_, value)| value.as_str())
}

/// `streamSimple`'s `auto` transport rides the cached WebSocket context: the
/// response frame sends once, the handshake carries the session pair, and no
/// fetch fires.
#[tokio::test]
async fn forwards_auto_transport_from_stream_simple_options_and_uses_cached_websocket_context() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let options = SimpleStreamOptions {
        transport_options: TransportOptions {
            http_client: Some(Arc::new(fetch.clone())),
            ..TransportOptions::default()
        },
        api_key: Some(token),
        transport: Some(Transport::Auto),
        session_id: Some("session-auto".to_owned()),
        ..SimpleStreamOptions::default()
    };

    let stream =
        openai_codex_responses::stream_simple(&model, &codex_context_at(1), Some(&options));
    let mut peer = next_peer(&transport).await;
    let sent = next_sent_body(&mut peer).await;
    peer_sends(
        &peer,
        &[
            message_added(),
            part_added(),
            text_delta("Hello"),
            message_done("msg_1", "Hello"),
            completed_terminal(Some(false), None),
        ],
    );
    let result = drain_and_settle(&stream).await;

    assert_eq!(result.end_turn, Some(false));
    assert_eq!(sent.get("type"), Some(&json!("response.create")));
    let connect = peer.connect_request();
    assert_eq!(connect_header(connect, "session-id"), Some("session-auto"));
    assert!(connect_header(connect, "session_id").is_none());
    assert_eq!(
        connect_header(connect, "x-client-request-id"),
        Some("session-auto")
    );
    assert_eq!(fetch.request_count(), 0);
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("session-auto")
        .expect("the session records stats");
    assert_eq!(stats.cached_context_requests, 1);
    assert_eq!(stats.full_context_requests, 1);
}

/// Rotating accounts must not reuse a socket authenticated by another
/// account (#7284): the cache keys on the session and the account id.
#[tokio::test]
async fn scopes_cached_websockets_to_the_authenticated_account() {
    let _guard = suite();
    let transport = Arc::new(MockWebSocketTransport::new());
    openai_codex_responses::set_websocket_transport(Some(transport.clone()));
    let fetch = MockHttpClient::new();
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let context = codex_context_empty();
    let turn_options = |account: &str| {
        let mut options =
            websocket_options(&fetch, &mock_token(account), Transport::WebsocketCached);
        options.session_id = Some("shared-session".to_owned());
        options
    };

    // Turn 1 (account-a) connects and caches its socket.
    let stream = openai_codex_responses::stream(&model, &context, Some(&turn_options("account-a")));
    let mut first_peer = next_peer(&transport).await;
    let first_sent = next_sent_body(&mut first_peer).await;
    let _ = first_sent;
    peer_sends(&first_peer, &[completed_response(1)]);
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    // Turn 2 (account-b) must connect fresh: no cached socket for its
    // account.
    let stream = openai_codex_responses::stream(&model, &context, Some(&turn_options("account-b")));
    let mut second_peer = next_peer(&transport).await;
    let second_sent = next_sent_body(&mut second_peer).await;
    let _ = second_sent;
    peer_sends(&second_peer, &[completed_response(2)]);
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    // Turn 3 (account-a) reuses the first socket; the frame rides it.
    let stream = openai_codex_responses::stream(&model, &context, Some(&turn_options("account-a")));
    let third_sent = next_sent_body(&mut first_peer).await;
    let _ = third_sent;
    peer_sends(&first_peer, &[completed_response(3)]);
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    assert_eq!(
        connect_header(first_peer.connect_request(), "chatgpt-account-id"),
        Some("account-a")
    );
    assert_eq!(
        connect_header(second_peer.connect_request(), "chatgpt-account-id"),
        Some("account-b")
    );
    assert_eq!(
        connect_header(first_peer.connect_request(), "authorization"),
        Some(format!("Bearer {}", mock_token("account-a")).as_str())
    );
    assert_eq!(
        connect_header(second_peer.connect_request(), "authorization"),
        Some(format!("Bearer {}", mock_token("account-b")).as_str())
    );
    // Turn 3 reused the cached socket, so exactly the two connects above
    // happened and no peer is left un-served.
    assert!(
        transport.next_peer().is_none(),
        "exactly two connects served"
    );
    assert_eq!(fetch.request_count(), 0);
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("shared-session")
        .expect("the session records stats");
    assert_eq!(stats.connections_created, 2);
    assert_eq!(stats.connections_reused, 1);
}

/// `cacheRetention: "none"` closes every one-shot socket: two turns, two
/// connects, two closes, and no session stats.
#[tokio::test]
async fn closes_one_shot_websockets_when_cache_retention_is_none() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let context = codex_context_at(1);
    let options = || {
        let mut options = websocket_options(&fetch, &token, Transport::Auto);
        options.cache_retention = Some(CacheRetention::None);
        options.session_id = Some("one-off-summary".to_owned());
        options
    };

    let mut closes = 0;
    for turn in 1..=2 {
        let stream = openai_codex_responses::stream(&model, &context, Some(&options()));
        let mut peer = next_peer(&transport).await;
        let sent = next_sent_body(&mut peer).await;
        assert_eq!(sent.get("prompt_cache_key"), None, "turn {turn}");
        peer_sends(&peer, &[completed_response(turn)]);
        let result = drain_and_settle(&stream).await;
        assert_eq!(result.stop_reason, StopReason::Stop, "turn {turn}");
        let outbound = peer
            .next_sent()
            .await
            .expect("the release closes the one-shot socket");
        assert!(matches!(outbound, WebSocketOutbound::Close { .. }));
        closes += 1;
    }

    assert_eq!(closes, 2);
    // Both one-shot sockets connected and were served; nothing is left.
    assert!(
        transport.next_peer().is_none(),
        "exactly two connects served"
    );
    assert!(
        openai_codex_responses::get_openai_codex_websocket_debug_stats("one-off-summary").is_none()
    );
    assert_eq!(fetch.request_count(), 0);
}

/// A connect that does not open before the connect-handshake timeout falls
/// back to SSE and records the failure shape.
#[tokio::test(start_paused = true)]
async fn falls_back_to_sse_when_websocket_connect_does_not_open_before_the_connect_timeout() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let transport = Arc::new(MockWebSocketTransport::new().with_connect_delay_ms(30_000));
    openai_codex_responses::set_websocket_transport(Some(transport.clone()));
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = websocket_options(&mock, &token, Transport::Auto);
    options.session_id = Some("ws-connect-timeout".to_owned());
    options.timeout_ms = Some(300_000);
    options.websocket_connect_timeout_ms = Some(50);

    let stream = openai_codex_responses::stream(&model, &codex_context_at(1), Some(&options));
    tokio::time::advance(Duration::from_millis(50)).await;
    let result = drain_and_settle(&stream).await;

    assert_eq!(message_text(&result), Some("Hello"));
    assert_eq!(mock.request_count(), 1);
    let stats =
        openai_codex_responses::get_openai_codex_websocket_debug_stats("ws-connect-timeout")
            .expect("the session records stats");
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(stats.sse_fallbacks, 1);
    assert_eq!(stats.websocket_fallback_active, Some(true));
    assert_eq!(
        stats.last_web_socket_error.as_deref(),
        Some("WebSocket connect timeout after 50ms")
    );
}

/// A full backend connection pool (`websocket_connection_limit_reached`)
/// spends one fresh reconnect before any fallback.
#[tokio::test]
async fn reconnects_once_when_the_websocket_connection_limit_is_reached_before_output_starts() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let options = websocket_options(&fetch, &token, Transport::Auto);

    let stream = openai_codex_responses::stream(&model, &codex_context_empty(), Some(&options));
    let mut first_peer = next_peer(&transport).await;
    let first_sent = next_sent_body(&mut first_peer).await;
    let _ = first_sent;
    peer_sends(
        &first_peer,
        &[json!({ "type": "error", "error": { "code": "websocket_connection_limit_reached" } })],
    );
    let mut second_peer = next_peer(&transport).await;
    let second_sent = next_sent_body(&mut second_peer).await;
    let _ = second_sent;
    peer_sends(&second_peer, &[completed_response(1)]);
    let result = drain_and_settle(&stream).await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    // Exactly two connects happened and both peers were served.
    assert!(
        transport.next_peer().is_none(),
        "exactly two connects served"
    );
    assert_eq!(fetch.request_count(), 0);
}

/// A WebSocket idle before the first event falls back to SSE.
#[tokio::test(start_paused = true)]
async fn falls_back_to_sse_when_a_websocket_is_idle_before_the_first_event() {
    let (guard, token, transport, fetch, model) = ws_fallback_suite();
    let _guard = guard;
    let mut options = websocket_options(&fetch, &token, Transport::Auto);
    options.session_id = Some("ws-idle-before-start".to_owned());
    options.timeout_ms = Some(50);

    let stream = openai_codex_responses::stream(&model, &codex_context_at(1), Some(&options));
    let mut peer = next_peer(&transport).await;
    let sent = next_sent_body(&mut peer).await;
    assert_eq!(sent.get("type"), Some(&json!("response.create")));
    tokio::time::advance(Duration::from_millis(50)).await;
    let result = drain_and_settle(&stream).await;

    assert_eq!(message_text(&result), Some("Hello"));
    assert_eq!(fetch.request_count(), 1);
    let stats =
        openai_codex_responses::get_openai_codex_websocket_debug_stats("ws-idle-before-start")
            .expect("the session records stats");
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(stats.sse_fallbacks, 1);
    assert_eq!(stats.websocket_fallback_active, Some(true));
}

/// A WebSocket idle after the stream started is an error, not a fallback:
/// events already rode out.
#[tokio::test(start_paused = true)]
async fn errors_when_a_websocket_is_idle_after_the_stream_started() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let mut options = websocket_options(&fetch, &token, Transport::Auto);
    options.timeout_ms = Some(50);

    let stream = openai_codex_responses::stream(&model, &codex_context_at(1), Some(&options));
    let mut peer = next_peer(&transport).await;
    let sent = next_sent_body(&mut peer).await;
    let _ = sent;
    peer_sends(
        &peer,
        &[json!({
            "type": "response.output_item.added",
            "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
        })],
    );
    // One event confirms the first frame rode out, so the idle that follows
    // is the after-start failure shape.
    let _start = stream.next().await.expect("the stream opens");
    tokio::time::advance(Duration::from_millis(50)).await;
    let result = drain_and_settle(&stream).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("WebSocket idle timeout after 50ms")
    );
    assert_eq!(fetch.request_count(), 0);
}

// ===========================================================================
// Cached-context continuation cases
// ===========================================================================

/// The `websocket-cached` transport sends only the input delta on a matching
/// continuation: the second turn's frame carries `previous_response_id` and
/// the two new items alone.
#[expect(
    clippy::too_many_lines,
    reason = "the test replays upstream's two-turn mock sequence in one body; splitting the fixtures out would hide the 1:1 mapping"
)]
#[tokio::test]
async fn sends_only_response_input_deltas_in_websocket_cached_mode() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let transport = Arc::new(MockWebSocketTransport::new());
    openai_codex_responses::set_websocket_transport(Some(transport.clone()));
    let fetch = MockHttpClient::new();
    let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    model.compat = Some(ModelCompat {
        supports_openai_grammar_tools: Some(true),
        ..ModelCompat::default()
    });
    let tool = Tool {
        name: "sample_tool".to_owned(),
        description: "Sample tool".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "payload": { "type": "string" } },
            "required": ["payload"],
        }),
        constrained_sampling: Some(ConstrainedSamplingSetting::Config(
            ConstrainedSamplingConfig::Grammar {
                variants: BTreeMap::from([(
                    GrammarFormat::OpenaiLark,
                    "start: /[a-z]+/".to_owned(),
                )]),
            },
        )),
    };
    let first_context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![user_message_at("Use the tool", 1)],
        tools: Some(vec![tool]),
    };
    let options = || {
        let mut options = websocket_options(&fetch, &token, Transport::WebsocketCached);
        options.session_id = Some("session-1".to_owned());
        options
    };

    let stream = openai_codex_responses::stream(&model, &first_context, Some(&options()));
    let mut peer = next_peer(&transport).await;
    let first_sent = next_sent_body(&mut peer).await;
    peer_sends(
        &peer,
        &[
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({
                "type": "response.output_item.added",
                "item": { "type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "sample_tool", "input": "" },
            }),
            json!({ "type": "response.custom_tool_call_input.delta", "item_id": "ctc_1", "delta": "abc" }),
            json!({ "type": "response.custom_tool_call_input.done", "item_id": "ctc_1", "input": "abc" }),
            json!({
                "type": "response.output_item.done",
                "item": { "type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "sample_tool", "input": "abc" },
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "total_tokens": 8,
                        "input_tokens_details": { "cached_tokens": 0 },
                    },
                },
            }),
        ],
    );
    let first = drain_and_settle(&stream).await;

    let second_context = Context {
        system_prompt: first_context.system_prompt.clone(),
        messages: vec![
            first_context.messages[0].clone(),
            Message::Assistant(first.clone()),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "call_1|ctc_1".to_owned(),
                tool_name: "sample_tool".to_owned(),
                content: vec![ToolResultBlock::Text(pi_ai::types::TextContent {
                    text: "real result".to_owned(),
                    text_signature: None,
                })],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: false,
                timestamp: 2,
            }),
            user_message_at("Now finish", 3),
        ],
        tools: first_context.tools.clone(),
    };
    let stream = openai_codex_responses::stream(&model, &second_context, Some(&options()));
    let second_sent = next_sent_body(&mut peer).await;
    peer_sends(
        &peer,
        &[
            json!({ "type": "response.created", "response": { "id": "resp_2" } }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_2",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "total_tokens": 8,
                        "input_tokens_details": { "cached_tokens": 0 },
                    },
                },
            }),
        ],
    );
    let _second = drain_and_settle(&stream).await;

    assert_eq!(first_sent.get("store"), Some(&json!(false)));
    assert_eq!(first_sent.get("previous_response_id"), None);
    assert_eq!(
        first_sent.get("input"),
        Some(&json!([
            { "role": "user", "content": [{ "type": "input_text", "text": "Use the tool" }] },
        ]))
    );
    assert_eq!(second_sent.get("store"), Some(&json!(false)));
    assert_eq!(
        second_sent.get("previous_response_id"),
        Some(&json!("resp_1"))
    );
    assert_eq!(
        second_sent.get("input"),
        Some(&json!([
            { "type": "custom_tool_call_output", "call_id": "call_1", "output": "real result" },
            { "role": "user", "content": [{ "type": "input_text", "text": "Now finish" }] },
        ]))
    );
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("session-1")
        .expect("the session records stats");
    assert_eq!(stats.requests, 2);
    assert_eq!(stats.connections_created, 1);
    assert_eq!(stats.connections_reused, 1);
    assert_eq!(stats.cached_context_requests, 2);
    assert_eq!(stats.store_true_requests, 0);
    assert_eq!(stats.full_context_requests, 1);
    assert_eq!(stats.delta_requests, 1);
    assert_eq!(stats.last_delta_input_items, Some(2));
    assert_eq!(stats.last_previous_response_id.as_deref(), Some("resp_1"));
    assert_eq!(fetch.request_count(), 0);
}

/// Which recovery path the missing-continuation regression rides.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Recovery {
    /// The retry lands on a fresh socket.
    Websocket,
    /// The retry socket dies and the full input re-streams over SSE.
    Sse,
}

/// `recovers a missing cached websocket continuation via <transport>`: the
/// delta request fails with `previous_response_not_found`, the retry sends
/// the full input, and the recovery transport decides the outcome.
#[expect(
    clippy::too_many_lines,
    reason = "the test replays upstream's three-send recovery state machine in one body; splitting the fixtures out would hide the 1:1 shape"
)]
async fn recover_missing_cached_websocket_continuation(recovery: Recovery) {
    let _guard = suite();
    let token = mock_token("acc_test");
    let transport = Arc::new(MockWebSocketTransport::new());
    openai_codex_responses::set_websocket_transport(Some(transport.clone()));
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, build_sse_payload("completed", false, None));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let session_id = format!(
        "missing-continuation-{}",
        if recovery == Recovery::Sse {
            "sse"
        } else {
            "websocket"
        }
    );
    let first_context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![user_message_at("Say hello", 1)],
        tools: None,
    };
    let options = || {
        let mut options = websocket_options(&mock, &token, Transport::WebsocketCached);
        options.session_id = Some(session_id.clone());
        options
    };

    // Turn 1 succeeds on connection one.
    let stream = openai_codex_responses::stream(&model, &first_context, Some(&options()));
    let mut first_peer = next_peer(&transport).await;
    let first_sent = next_sent_body(&mut first_peer).await;
    let _ = first_sent;
    peer_sends(
        &first_peer,
        &recovery_success_events("resp_1", "msg_1", "Hello"),
    );
    let first = drain_and_settle(&stream).await;

    // Turn 2 rides the reused socket as a delta, which the backend cannot
    // continue.
    let second_context = Context {
        system_prompt: first_context.system_prompt.clone(),
        messages: vec![
            first_context.messages[0].clone(),
            Message::Assistant(first.clone()),
            user_message_at("Now finish", 2),
        ],
        tools: None,
    };
    let stream = openai_codex_responses::stream(&model, &second_context, Some(&options()));
    let second_sent = next_sent_body(&mut first_peer).await;
    peer_sends(
        &first_peer,
        &[
            json!({
                "type": "codex.rate_limits",
                "plan_type": "plus",
                "rate_limits": {
                    "allowed": true,
                    "limit_reached": false,
                    "primary": {
                        "used_percent": 7,
                        "window_minutes": 10080,
                        "reset_after_seconds": 556_112,
                        "reset_at": 1_785_269_351,
                    },
                    "secondary": Value::Null,
                },
                "code_review_rate_limits": Value::Null,
                "additional_rate_limits": Value::Null,
                "credits": { "has_credits": false, "unlimited": false, "balance": "0" },
                "promo": Value::Null,
            }),
            json!({
                "type": "error",
                "status": 400,
                "error": {
                    "code": "previous_response_not_found",
                    "message": "Previous response with id 'resp_1' not found.",
                    "param": "previous_response_id",
                },
            }),
        ],
    );

    // The retry reconnects; the fresh socket either recovers or fails into
    // the SSE fallback.
    let mut second_peer = next_peer(&transport).await;
    let third_sent = next_sent_body(&mut second_peer).await;
    if recovery == Recovery::Sse {
        second_peer
            .close(1008, "retry websocket failed")
            .expect("server closes");
    } else {
        peer_sends(
            &second_peer,
            &recovery_success_events("resp_2", "msg_2", "Recovered"),
        );
    }
    let (tags, second) = drain_tags(&stream).await;

    assert_eq!(second.stop_reason, StopReason::Stop);
    assert_eq!(
        message_text(&second),
        Some(if recovery == Recovery::Sse {
            "Hello"
        } else {
            "Recovered"
        })
    );
    assert_eq!(tags.iter().filter(|tag| tag.as_str() == "start").count(), 1);
    assert!(!tags.contains(&"error".to_owned()));
    // Exactly two connects happened and both peers were served.
    assert!(
        transport.next_peer().is_none(),
        "exactly two connects served"
    );
    // Send 1 rode connection 1, send 2 the reused socket, send 3 the fresh
    // connection.
    assert_eq!(
        second_sent.get("previous_response_id"),
        Some(&json!("resp_1"))
    );
    assert_eq!(
        second_sent.get("input"),
        Some(&json!([
            { "role": "user", "content": [{ "type": "input_text", "text": "Now finish" }] },
        ]))
    );
    assert_eq!(third_sent.get("previous_response_id"), None);
    let third_input = third_sent
        .get("input")
        .and_then(Value::as_array)
        .expect("the full request carries its input");
    assert_eq!(third_input.len(), 3);
    assert_eq!(
        third_input.last(),
        Some(&json!({
            "role": "user",
            "content": [{ "type": "input_text", "text": "Now finish" }],
        }))
    );
    assert_eq!(mock.request_count(), usize::from(recovery == Recovery::Sse));
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats(&session_id)
        .expect("the session records stats");
    assert_eq!(stats.requests, 3);
    assert_eq!(stats.connections_created, 2);
    assert_eq!(stats.connections_reused, 1);
    assert_eq!(stats.full_context_requests, 2);
    assert_eq!(stats.delta_requests, 1);
    assert_eq!(
        stats.websocket_failures,
        u64::from(recovery == Recovery::Sse)
    );
    assert_eq!(stats.sse_fallbacks, u64::from(recovery == Recovery::Sse));
}

/// The fresh socket answers the retried full request itself.
#[tokio::test]
async fn recovers_a_missing_cached_websocket_continuation_via_websocket() {
    recover_missing_cached_websocket_continuation(Recovery::Websocket).await;
}

/// The fresh socket fails and the full context re-streams over SSE.
#[tokio::test]
async fn recovers_a_missing_cached_websocket_continuation_via_sse() {
    recover_missing_cached_websocket_continuation(Recovery::Sse).await;
}

// ===========================================================================
// SSE retry cases
// ===========================================================================

/// The IMF-fixdate `toUTCString` emits that `Date.parse` reads back, the
/// form the module's HTTP-date `Retry-After` distance parses.
fn http_date(ms: i64) -> String {
    const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];

    fn civil_from_days(days: i64) -> (i64, u32, u32) {
        let z = days + 719_468;
        let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
        let year = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = mp + if mp < 10 { 3 } else { -9 };
        let year = if month <= 2 { year + 1 } else { year };
        (
            year,
            u32::try_from(month).expect("month 1..=12"),
            u32::try_from(day).expect("day"),
        )
    }

    let days = ms.div_euclid(86_400_000);
    let seconds_of_day = ms.rem_euclid(86_400_000) / 1_000;
    let (year, month, day) = civil_from_days(days);
    let weekday = WEEKDAYS[usize::try_from(days.rem_euclid(7) + 4).expect("weekday in range") % 7];
    format!(
        "{weekday}, {day:02} {month} {year} {hour:02}:{minute:02}:{second:02} GMT",
        day = day,
        month = MONTHS[(month - 1) as usize],
        year = year,
        hour = seconds_of_day / 3_600,
        minute = (seconds_of_day % 3_600) / 60,
        second = seconds_of_day % 60,
    )
}

/// How the rate-limited first attempt answers, upstream's three
/// `Retry-After` header spellings.
#[derive(Clone)]
enum RetryHeaderCase {
    /// `retry-after-ms` in milliseconds.
    Millis(&'static str),
    /// `retry-after` in seconds.
    Seconds(&'static str),
    /// An IMF-fixdate 45s ahead of the wall clock, built at request time.
    HttpDate,
}

impl RetryHeaderCase {
    /// The header pairs the rate-limited response carries.
    fn headers(&self) -> Vec<(&'static str, String)> {
        let mut headers = vec![("content-type", "application/json".to_owned())];
        match self {
            Self::Millis(value) => headers.push(("retry-after-ms", (*value).to_owned())),
            Self::Seconds(value) => headers.push(("retry-after", (*value).to_owned())),
            Self::HttpDate => headers.push(("retry-after", http_date(now_ms() + 45_000))),
        }
        headers
    }

    /// The virtual-clock delay the retry must wait, upstream's
    /// `setTimeout` spy reading.
    const fn expected_delay_ms(&self) -> u64 {
        match self {
            Self::Millis(_) => 1_500,
            Self::Seconds(_) => 60_000,
            // The distance reads the wall clock at response time, so the
            // pinned window rides a small tolerance instead of one
            // millisecond.
            Self::HttpDate => 45_000,
        }
    }
}

use pi_ai::auth::resolve::now_ms;

/// The rate-limited-then-success mock the SSE retry cases mount: request
/// one answers 429 with the case's headers, later ones stream the SSE run.
/// Each request signals the shared channel so the paused-clock test can
/// measure the virtual delay between them.
fn rate_limited_sse_mock(
    headers: RetryHeaderCase,
    sse: String,
) -> (MockHttpClient, tokio::sync::mpsc::UnboundedReceiver<()>) {
    let mock = MockHttpClient::new();
    let (signal, receiver) = tokio::sync::mpsc::unbounded_channel();
    let served = Arc::new(AtomicUsize::new(0));
    mock.on(move |request| request.url.contains("/codex/responses"))
        .respond_fn(move |_request| {
            let served = Arc::clone(&served);
            let signal = signal.clone();
            let headers = headers.clone();
            let sse = sse.clone();
            async move {
                let index = served.fetch_add(1, Ordering::AcqRel) + 1;
                let _ = signal.send(());
                if index == 1 {
                    let mut response = json_response(
                        429,
                        &json!({ "error": { "code": "rate_limit_exceeded", "message": "rate limited" } }),
                    );
                    for (name, value) in headers.headers() {
                        response = response.with_header(name, value);
                    }
                    return Ok(response);
                }
                Ok(
                    MockResponse::status(200)
                        .with_header("content-type", "text/event-stream")
                        .with_body(sse),
                )
            }
        });
    (mock, receiver)
}

/// The server-requested `Retry-After` header drives the retry delay on every
/// spelling the parser reads.
#[tokio::test(start_paused = true)]
async fn uses_retry_after_headers_for_sse_retries() {
    let _guard = suite();
    let cases = [
        ("retry-after-ms", RetryHeaderCase::Millis("1500")),
        ("retry-after seconds", RetryHeaderCase::Seconds("60")),
        ("retry-after HTTP date", RetryHeaderCase::HttpDate),
    ];
    for (name, headers) in cases {
        let token = mock_token("acc_test");
        let (mock, mut requests) = rate_limited_sse_mock(headers.clone(), hello_sse_body());
        let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
        let mut options = sse_options(&mock, &token);
        options.max_retries = Some(1);

        let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
        // The settle future registers its result waiter before the stream can
        // complete: the retry stream needs no cooperation from this task, so
        // it may finish between the request signals.
        let settle = tokio::spawn({
            let stream = stream.clone();
            async move { drain_and_settle(&stream).await }
        });
        // The first attempt lands, then the paused clock advances exactly the
        // server-requested delay before the retry fires.
        requests.recv().await.expect("the first request arrives");
        let before = Instant::now();
        requests.recv().await.expect("the retry request fires");
        let elapsed = (Instant::now() - before).as_millis();
        let expected = headers.expected_delay_ms();
        match &headers {
            RetryHeaderCase::HttpDate => {
                assert!(
                    (44_000..=46_000).contains(&elapsed),
                    "{name}: the virtual delay {elapsed}ms pins the 45s window"
                );
            }
            _ => assert_eq!(elapsed, u128::from(expected), "{name}"),
        }
        let result = settle.await.expect("the settle task");

        assert_eq!(message_text(&result), Some("Hello"), "{name}");
        assert_eq!(mock.request_count(), 2, "{name}");
    }
}

/// A retry delay above the caller's cap fails immediately with the requested
/// delay in the message, for both retryable statuses.
#[tokio::test]
async fn fails_immediately_when_a_retry_delay_exceeds_the_limit() {
    let _guard = suite();
    for status in [429u16, 503] {
        let token = mock_token("acc_test");
        let mock = MockHttpClient::new();
        mock.on(|request| request.url.contains("/codex/responses"))
            .respond(
                json_response(
                    status,
                    &json!({ "error": { "code": "temporarily_unavailable", "message": "retry later" } }),
                )
                .with_header("retry-after", "2"),
            );
        let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
        let mut options = sse_options(&mock, &token);
        options.max_retries = Some(3);
        options.max_retry_delay_ms = Some(1000);

        let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
        let result = drain_and_settle(&stream).await;

        assert_eq!(result.stop_reason, StopReason::Error, "status {status}");
        assert_eq!(
            result.error_message.as_deref(),
            Some("Server requested 2s retry delay (max: 1s)")
        );
        assert_eq!(mock.request_count(), 1, "status {status}");
    }
}

/// The SSE request bodies ride zstd frames, decoded back byte-exact.
#[tokio::test]
async fn zstd_compresses_sse_request_bodies() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let large_text = "compress me ".repeat(400);
    let options = sse_options(&mock, &token);

    let first_context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![user_message_at(&large_text, 1)],
        tools: None,
    };
    let stream = openai_codex_responses::stream(&model, &first_context, Some(&options));
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let first_request = &mock.recorded()[0];
    assert_eq!(
        request_header(first_request, "content-encoding"),
        Some("zstd")
    );
    assert!(first_request.body.is_some());
    let decoded = decode_codex_request_body(first_request.body.as_ref())
        .expect("the large request body decodes");
    assert_eq!(
        decoded_input_text(&decoded),
        Some(large_text.as_str()),
        "the stored-block frame round-trips"
    );

    let second_context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![user_message_at("hi", 1)],
        tools: None,
    };
    let stream = openai_codex_responses::stream(&model, &second_context, Some(&options));
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    let second_request = &mock.recorded()[1];
    assert_eq!(
        request_header(second_request, "content-encoding"),
        Some("zstd")
    );
    assert!(second_request.body.is_some());
}

/// The first input item's first text block, the shape the zstd round-trip
/// assertion reads.
fn decoded_input_text(body: &Value) -> Option<&str> {
    body.get("input")
        .and_then(Value::as_array)
        .and_then(|input| input.first())
        .and_then(|item| item.get("content"))
        .and_then(Value::as_array)
        .and_then(|content| content.first())
        .and_then(|block| block.get("text"))
        .and_then(Value::as_str)
}

/// Repeated SSE retries without `Retry-After` headers climb the exact
/// `1000 * 2^n` backoff steps.
#[tokio::test(start_paused = true)]
async fn uses_exponential_backoff_across_repeated_sse_retries_without_retry_headers() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    let sse = hello_sse_body();
    let served = Arc::new(AtomicUsize::new(0));
    let (signal, mut attempts) = tokio::sync::mpsc::unbounded_channel();
    mock.on(move |request| request.url.contains("/codex/responses"))
        .respond_fn(move |_request| {
            let served = Arc::clone(&served);
            let signal = signal.clone();
            let sse = sse.clone();
            async move {
                let index = served.fetch_add(1, Ordering::AcqRel) + 1;
                let _ = signal.send(());
                if index <= 3 {
                    // No `Retry-After` header: the retry climbs the
                    // exponential schedule.
                    return Ok(json_response(
                        429,
                        &json!({ "error": { "code": "rate_limit_exceeded", "message": "rate limited" } }),
                    ));
                }
                Ok(MockResponse::status(200)
                    .with_header("content-type", "text/event-stream")
                    .with_body(sse))
            }
        });
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.max_retries = Some(3);
    let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
    // The settle future registers its result waiter before the stream can
    // complete: the retried stream needs no cooperation from this task.
    let settle = tokio::spawn({
        let stream = stream.clone();
        async move { drain_and_settle(&stream).await }
    });

    // Each retry fires at the exact exponential step; the virtual clock
    // measures the gap the paused timer actually waited.
    let mut delays = Vec::new();
    let mut previous = Instant::now();
    for _ in 0..4 {
        attempts.recv().await.expect("the next attempt fires");
        let now = Instant::now();
        delays.push((now - previous).as_millis());
        previous = now;
    }
    let result = settle.await.expect("the settle task");

    // The first measurement spans the setup polls (zero virtual time); the
    // three retry steps carry the pinned schedule.
    assert_eq!(delays[1], 1_000, "the first backoff step");
    assert_eq!(delays[2], 2_000, "the second backoff step");
    assert_eq!(delays[3], 4_000, "the third backoff step");
    assert_eq!(mock.request_count(), 4);
    assert_eq!(message_text(&result), Some("Hello"));
}
// ---------------------------------------------------------------------------
// Port-added: WebSocket pump edges, friendly error parsing, and request
// fields the pinned upstream suite reaches only implicitly
// ---------------------------------------------------------------------------

/// Port-added: WebSocket events ride binary frames the same way text frames
/// do — the pump decodes the bytes and maps the event.
#[tokio::test]
async fn delivers_websocket_events_through_binary_frames() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let options = plain_ws_options(&fetch, &token);

    let (stream, peer) = plain_turn(&transport, &model, &options).await;
    let completed = serde_json::to_vec(&completed_response(1)).expect("the terminal bytes");
    peer.send_binary(Bytes::from(completed))
        .expect("the peer delivers the binary frame");
    let result = drain_and_settle(&stream).await;

    assert_eq!(result.stop_reason, StopReason::Stop);
}

/// Port-added: a WebSocket message that is not JSON fails the stream with
/// the protocol message.
#[tokio::test]
async fn an_invalid_websocket_json_message_fails_the_stream() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let options = plain_ws_options(&fetch, &token);

    let (stream, peer) = plain_turn(&transport, &model, &options).await;
    peer.send_text("this is not json")
        .expect("the peer delivers the frame");
    let result = drain_and_settle(&stream).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the protocol failure");
    assert![
        message.contains("Invalid Codex WebSocket JSON"),
        "{message}"
    ];
}

/// Port-added: a WebSocket failure after the stream started is final — the
/// dropped peer surfaces as the 1006 close wording, not an SSE fallback.
#[tokio::test]
async fn a_websocket_transport_error_after_the_stream_started_fails_the_stream() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let options = plain_ws_options(&fetch, &token);

    let (stream, peer) = plain_turn(&transport, &model, &options).await;
    // One event confirms the frame rode out, so the failure that follows is
    // final instead of falling back to SSE.
    peer_sends(
        &peer,
        &[json!({
            "type": "response.output_item.added",
            "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
        })],
    );
    let _start = stream.next().await.expect("the stream opens");
    drop(peer);
    let result = drain_and_settle(&stream).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the transport failure");
    assert![message.contains("1006 mock websocket dropped"), "{message}"];
}

/// Port-added: a server close before any output fails with the close code
/// and reason; the message-too-big close code spells its own wording.
#[tokio::test]
async fn a_websocket_close_before_output_fails_with_the_close_reason() {
    for (code, reason, needle) in [
        (4000u16, "backend restarting", "backend restarting"),
        // The too-big close code with an empty reason spells its own wording.
        (1009, "", "message too big"),
    ] {
        let _guard = suite();
        let token = mock_token("acc_test");
        let transport = Arc::new(MockWebSocketTransport::new());
        openai_codex_responses::set_websocket_transport(Some(transport.clone()));
        let fetch = MockHttpClient::new();
        let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
        let mut options = websocket_options(&fetch, &token, Transport::Websocket);
        options.cache_retention = Some(CacheRetention::None);

        let stream = openai_codex_responses::stream(&model, &codex_context_at(1), Some(&options));
        let mut peer = next_peer(&transport).await;
        let _ = next_sent_body(&mut peer).await;
        peer_sends(
            &peer,
            &[json!({
                "type": "response.output_item.added",
                "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
            })],
        );
        let _start = stream.next().await.expect("the stream opens");
        peer.close(code, reason).expect("the peer closes");
        let result = drain_and_settle(&stream).await;

        assert_eq!(result.stop_reason, StopReason::Error, "code {code}");
        let message = result.error_message.expect("the close failure");
        assert![message.contains(needle), "code {code}: {message}"];
    }
}

/// Port-added: a WebSocket read aborted by the request signal reports the
/// abort failure.
#[tokio::test(start_paused = true)]
async fn an_aborted_websocket_read_reports_the_abort() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let signal = CancellationToken::new();
    let mut options = plain_ws_options(&fetch, &token);
    options.transport_options.signal = Some(signal);

    let (stream, _peer) = plain_turn(&transport, &model, &options).await;
    let settle = tokio::spawn({
        let stream = stream.clone();
        async move { drain_and_settle(&stream).await }
    });
    // The read is parked on the silent peer; the cancel ends it aborted.
    tokio::time::advance(Duration::from_millis(10)).await;
    if let Some(signal) = options.transport_options.signal.as_ref() {
        signal.cancel();
    }
    let result = settle.await.expect("the settle task");

    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq!(result.error_message.as_deref(), Some("Request was aborted"));
}

/// Port-added: the 429 usage-limit error spells the friendly ChatGPT message
/// with the plan name and the try-again window.
#[tokio::test]
async fn the_usage_limit_error_spells_the_friendly_message() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    let resets_at = (now_ms() + 120_000) / 1000;
    mock.on(|request| request.url.contains("/codex/responses"))
        .respond(json_response(
            429,
            &json!({
                "error": {
                    "code": "usage_limit_reached",
                    "message": "You've hit your usage limit",
                    "plan_type": "pro",
                    "resets_at": resets_at,
                }
            }),
        ));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.max_retries = Some(0);

    let result = sse_run(&model, &options).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the friendly message");
    assert![
        message.contains("You have hit your ChatGPT usage limit"),
        "{message}"
    ];
    assert![message.contains("(pro plan)"), "{message}"];
    assert![message.contains("Try again in ~2 min."), "{message}"];
}

/// Port-added: the error-body parser's remaining shapes — an expired reset
/// window, plain error bodies, non-JSON bodies, and empty bodies — spell
/// their messages.
#[tokio::test]
async fn the_error_body_parser_spells_its_remaining_shapes() {
    for (status, body, needle) in [
        (
            429u16,
            json!({"error": {"code": "usage_limit_reached", "plan_type": "free", "resets_at": 1}}),
            "You have hit your ChatGPT usage limit (free plan).",
        ),
        (
            500,
            json!({"error": {"message": "backend exploded", "type": "server_error"}}),
            "backend exploded",
        ),
        (500, json!({"not_an_error": true}), "not_an_error"),
        (
            500,
            Value::String("plain text body".to_owned()),
            "plain text body",
        ),
    ] {
        let _guard = suite();
        let token = mock_token("acc_test");
        let mock = MockHttpClient::new();
        mock.on(move |request| request.url.contains("/codex/responses"))
            .respond(json_response(status, &body));
        let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
        let mut options = sse_options(&mock, &token);
        options.max_retries = Some(0);

        let result = sse_run(&model, &options).await;

        assert_eq!(result.stop_reason, StopReason::Error, "{body}");
        let message = result.error_message.expect("the failure message");
        assert![message.contains(needle), "{body} -> {message}"];
    }

    // An empty body spells the bare "Request failed" message.
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/codex/responses"))
        .respond(MockResponse::status(502).with_body(""));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.max_retries = Some(0);
    let result = sse_run(&model, &options).await;
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.error_message.as_deref(), Some("Request failed"));
}

/// Port-added: a `response.failed` terminal fails with the error's code and
/// message.
#[tokio::test]
async fn a_response_failed_event_fails_with_the_code_and_message() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    let mut frames = hello_frames();
    frames.push(json!({
        "type": "response.failed",
        "response": {"error": {"message": "boom", "code": "srv_err"}}
    }));
    mount_sse_route(&mock, sse_body(&frames, false));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = sse_options(&mock, &token);

    let result = sse_run(&model, &options).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the failure");
    assert![message.contains("boom"), "{message}"];
}

/// Port-added: invalid SSE JSON fails the stream with the protocol message.
#[tokio::test]
async fn an_invalid_codex_sse_json_fails_the_stream() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, "data: this is not json\n\n".to_owned());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = sse_options(&mock, &token);

    let result = sse_run(&model, &options).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the protocol failure");
    assert![message.contains("Invalid Codex SSE JSON"), "{message}"];
}

/// Port-added: the request fields the codex body builders gate —
/// temperature, verbosity, both deferred-tools modes, and the auto tool
/// choice — ride the request body or headers as the wire spells them.
#[tokio::test]
async fn the_codex_request_fields_ride_their_request_shapes() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let fetch = MockHttpClient::new();
    let transport = Arc::new(MockWebSocketTransport::new());
    openai_codex_responses::set_websocket_transport(Some(transport.clone()));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = plain_ws_options(&fetch, &token);
    options.temperature = Some(0.4);
    options.text_verbosity = Some("low".to_owned());
    options.tool_choice = Some(CodexToolChoice::Auto);
    options.session_id = Some("codex-fields".to_owned());

    let stream = openai_codex_responses::stream(&model, &codex_context_at(1), Some(&options));
    let mut peer = next_peer(&transport).await;
    let body = next_sent_body(&mut peer).await;
    peer_sends(&peer, &[completed_response(1)]);
    let result = drain_and_settle(&stream).await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(body.get("temperature"), Some(&json!(0.4)));
    assert_eq!(body["text"]["verbosity"], json!("low"));
    assert_eq!(body.get("tool_choice"), Some(&json!("auto")));
} // ---------------------------------------------------------------------------
// Port-added: request-build edges, token parsing, retry exhaustion, and the
// cached-connection lifecycle
// ---------------------------------------------------------------------------

/// Port-added: caller headers merge over the model's, `None` values suppress
/// a default, and the forced auth headers win over both.
#[tokio::test]
async fn the_codex_header_merges_follow_the_wire_precedence() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    model.headers = Some(
        [
            ("x-init-header".to_owned(), "init".to_owned()),
            ("x-suppressed".to_owned(), "model-value".to_owned()),
        ]
        .into_iter()
        .collect(),
    );
    let mut options = sse_options(&mock, &token);
    options.headers = Some(
        [
            ("x-init-header".to_owned(), Some("caller".to_owned())),
            ("x-suppressed".to_owned(), None),
            ("x-additional".to_owned(), Some("extra".to_owned())),
        ]
        .into_iter()
        .collect(),
    );

    let result = sse_run(&model, &options).await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    let request = &mock.recorded()[0];
    let header = |name: &str| {
        request
            .headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    };
    assert_eq!(header("x-init-header").as_deref(), Some("caller"));
    assert_eq!(header("x-suppressed"), None);
    assert_eq!(header("x-additional").as_deref(), Some("extra"));
}

/// Port-added: a connect header name that is not a valid token fails the
/// WebSocket connect; the failure surfaces through the fallback machine.
#[tokio::test]
async fn an_invalid_websocket_header_name_fails_the_connect() {
    let (guard, token, _transport, fetch, model) = ws_fallback_suite();
    let _guard = guard;
    let mut options = websocket_options(&fetch, &token, Transport::Websocket);
    options.headers =
        Some(std::iter::once(("bad header\nname".to_owned(), Some("v".to_owned()))).collect());

    let result = drain_and_settle(&openai_codex_responses::stream(
        &model,
        &codex_context_at(1),
        Some(&options),
    ))
    .await;

    // The connect-time validation failure rides the fallback machine; the
    // SSE route serves the run.
    assert_eq!(message_text(&result), Some("Hello"));
    assert_eq!(fetch.request_count(), 1);
}

/// Port-added: a base URL ending in `/codex` keeps its path and the SSE URL;
/// the WebSocket URL swaps the scheme; an `http` base maps to `ws`.
#[tokio::test]
async fn the_codex_urls_normalize_their_schemes_and_paths() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    model.base_url = "https://codex.example/backend/codex/".to_owned();
    let options = sse_options(&mock, &token);

    let result = sse_run(&model, &options).await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(
        mock.recorded()[0].url,
        "https://codex.example/backend/codex/responses"
    );

    // An http base URL maps to ws for the WebSocket transport.
    let transport = Arc::new(MockWebSocketTransport::new());
    openai_codex_responses::set_websocket_transport(Some(transport.clone()));
    let fetch = MockHttpClient::new();
    let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    model.base_url = "http://codex.example/backend".to_owned();
    let options = plain_ws_options(&fetch, &token);

    let (stream, peer) = plain_turn(&transport, &model, &options).await;
    peer_sends(&peer, &[completed_response(1)]);
    let result = drain_and_settle(&stream).await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    let connect = peer.connect_request();
    assert![
        connect
            .url
            .starts_with("ws://codex.example/backend/codex/responses"),
        "{}",
        connect.url
    ];
}

/// Port-added: the deferred-tools compat flags split the tool list into the
/// wire's deferred shapes.
#[tokio::test]
async fn the_deferred_tools_flags_split_the_tool_set() {
    for flag in ["supports_additional_tools", "supports_tool_search"] {
        let _guard = suite();
        let token = mock_token("acc_test");
        let mock = MockHttpClient::new();
        mount_sse_route(&mock, hello_sse_body());
        let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
        model.compat = Some(ModelCompat {
            supports_additional_tools: Some(flag == "supports_additional_tools"),
            supports_tool_search: Some(flag == "supports_tool_search"),
            ..ModelCompat::default()
        });
        let context = Context {
            system_prompt: Some("You are a helpful assistant.".to_owned()),
            messages: vec![
                Message::Assistant(common::tool_call_assistant_message(
                    "openai-codex-responses",
                    "openai-codex",
                    "gpt-5.1-codex",
                    pi_ai::types::ToolCall {
                        id: "call_1".to_owned(),
                        name: "lookup".to_owned(),
                        arguments: serde_json::Map::new(),
                        thought_signature: None,
                        namespace: None,
                    },
                )),
                common::tool_result_message(
                    "call_1",
                    vec![ToolResultBlock::Text(common::text_block("done"))],
                    Some(vec!["late_tool".to_owned()]),
                ),
                user_message_now("Say hello"),
            ],
            tools: Some(vec![common::lookup_tool(), common::late_tool()]),
        };
        let options = sse_options(&mock, &token);

        let result = drain_and_settle(&openai_codex_responses::stream(
            &model,
            &context,
            Some(&options),
        ))
        .await;

        assert_eq!(result.stop_reason, StopReason::Stop, "{flag}");
        let body = recorded_codex_body(&mock, 0);
        let names: Vec<Value> = body["tools"]
            .as_array()
            .expect("the tools array")
            .iter()
            .map(|tool| tool["name"].clone())
            .collect();
        assert_eq![names, vec![json!("lookup")], "{flag}"];
    }
}

/// Port-added: an unparsable auth token fails the stream with the generic
/// account-id message.
#[tokio::test]
async fn an_unparsable_token_fails_the_account_extraction() {
    let _guard = suite();
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, "not-a-jwt");
    options.max_retries = Some(0);

    let result = sse_run(&model, &options).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Failed to extract accountId from token")
    );
    assert_eq!(mock.request_count(), 0);
}

/// Port-added: token payloads decode through the lenient base64 — both
/// alphabet spellings, missing padding, and invalid characters skipped.
#[tokio::test]
async fn the_token_payload_decodes_through_the_lenient_base64() {
    let _guard = suite();
    // The U+FFFF tail makes the payload segment's standard-alphabet base64
    // carry '+', '/', and '=' padding; the lenient decoder reads them.
    let token = mock_token("a\u{FFFF}");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = sse_options(&mock, &token);

    let result = sse_run(&model, &options).await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    let request = &mock.recorded()[0];
    let account = request
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("chatgpt-account-id"))
        .map(|(_, value)| value.clone())
        .expect("the account id header");
    assert_eq!(account, "a\u{FFFF}");
}

/// Port-added: a retryable 5xx exhausts the retry budget and surfaces the
/// last attempt's error message.
#[tokio::test(start_paused = true)]
async fn a_retryable_error_exhausts_the_budget_and_fails_after_retries() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    let served = Arc::new(AtomicUsize::new(0));
    let served_for_body = Arc::clone(&served);
    mock.on(move |_request| {
        served.fetch_add(1, Ordering::AcqRel);
        true
    })
    .respond_fn(move |_request| {
        let served = Arc::clone(&served_for_body);
        async move {
            served.fetch_add(1, Ordering::AcqRel);
            Ok(json_response(
                503,
                &json!({"error": {"message": "overloaded"}}),
            ))
        }
    });
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.max_retries = Some(2);

    let result = sse_run(&model, &options).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.error_message.as_deref(), Some("overloaded"));
    assert_eq!(mock.request_count(), 3);
}

/// Port-added: an abort during the exponential backoff sleep ends the retry
/// loop aborted.
#[tokio::test(start_paused = true)]
async fn an_abort_during_the_backoff_sleep_ends_aborted() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let signal = CancellationToken::new();
    let mock = MockHttpClient::new();
    let (arrived_tx, mut first_request) = tokio::sync::mpsc::unbounded_channel();
    mock.on(move |request| {
        let _ = arrived_tx.send(());
        request.url.contains("/codex/responses")
    })
    .respond(json_response(
        503,
        &json!({"error": {"message": "unavailable"}}),
    ));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.max_retries = Some(3);
    options.transport_options.signal = Some(signal.clone());

    let settle = spawned_sse_run(&model, &options);
    // The first attempt lands, the backoff sleep starts, and the cancel ends
    // it aborted.
    first_request
        .recv()
        .await
        .expect("the first request arrives");
    signal.cancel();
    let result = settle.await.expect("the settle task");

    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq!(result.error_message.as_deref(), Some("Request was aborted"));
    assert_eq!(mock.request_count(), 1);
}

/// Port-added: a `Retry-After` header with an unparsable value reads as
/// absent, and the exponential backoff between retries fires.
#[tokio::test(start_paused = true)]
async fn the_retry_after_header_reads_blank_and_unparsable_values_as_absent() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let (mock, mut requests) =
        rate_limited_sse_mock(RetryHeaderCase::Seconds("x"), hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.max_retries = Some(1);

    let settle = spawned_sse_run(&model, &options);
    requests.recv().await.expect("the first request arrives");
    let before = Instant::now();
    requests.recv().await.expect("the retry fires");
    // An unparsable header falls back to the exponential backoff.
    assert_eq!((Instant::now() - before).as_millis(), 1_000);
    let result = settle.await.expect("the settle task");
    assert_eq!(message_text(&result), Some("Hello"));
}

/// Port-added: the idle-expiry task closes a released cached connection and
/// drops its cache entry after the idle window.
#[tokio::test(start_paused = true)]
async fn the_idle_expiry_closes_a_released_cached_connection() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let options = cached_ws_options(&fetch, &token, "idle-expiry");

    let (stream, mut peer) = plain_turn(&transport, &model, &options).await;
    peer_sends(&peer, &[completed_response(1)]);
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    // The released connection stays cached until the idle window elapses.
    assert!(
        openai_codex_responses::get_openai_codex_websocket_debug_stats("idle-expiry").is_some()
    );
    tokio::time::advance(Duration::from_secs(300)).await;
    let outbound = peer
        .next_sent()
        .await
        .expect("the idle expiry closes the socket");
    let WebSocketOutbound::Close { code, reason } = outbound else {
        panic!("expected the idle close, got {outbound:?}");
    };
    assert_eq![code, 1000];
    assert_eq![reason, "idle_timeout"];
}

/// Port-added: a failed WebSocket send is final for the attempt, surfaces
/// through the fallback machine, and records the failure in the stats.
#[tokio::test]
async fn a_websocket_send_failure_falls_back_and_records_the_failure() {
    let (guard, token, transport, fetch, model) = ws_fallback_suite();
    let _guard = guard;
    let options = cached_ws_options(&fetch, &token, "send-failure");

    let stream = openai_codex_responses::stream(&model, &codex_context_at(1), Some(&options));
    let peer = next_peer(&transport).await;
    // The peer drops before the client's send resolves: the send fails the
    // attempt and the SSE fallback serves the run.
    drop(peer);
    let result = drain_and_settle(&stream).await;

    assert_eq!(message_text(&result), Some("Hello"));
    assert_eq!(fetch.request_count(), 1);
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("send-failure")
        .expect("the session records stats");
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(stats.sse_fallbacks, 1);
}

/// Port-added: a cached attempt that fails before any output clears the
/// cached entry — the next turn reconnects instead of reusing.
#[tokio::test]
async fn a_failed_cached_attempt_clears_its_cache_entry() {
    let (guard, token, transport, fetch, model) = ws_fallback_suite();
    let _guard = guard;
    let options = cached_ws_options(&fetch, &token, "failed-attempt");

    let (stream, mut peer) = plain_turn(&transport, &model, &options).await;
    peer.close(4000, "backend restarting")
        .expect("the peer closes");
    let result = drain_and_settle(&stream).await;
    assert_eq!(message_text(&result), Some("Hello"));

    // The failed attempt released its slot: the client closes with the
    // release wording and the cache entry drops.
    let outbound = peer
        .next_sent()
        .await
        .expect("the release closes the failed connection");
    let WebSocketOutbound::Close { code, reason } = outbound else {
        panic!("expected the release close, got {outbound:?}");
    };
    assert_eq![code, 1000];
    assert_eq![reason, "done"];
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("failed-attempt")
        .expect("the session records stats");
    assert_eq!(stats.connections_created, 1);
    assert_eq!(stats.connections_reused, 0);
}
// ---------------------------------------------------------------------------
// Port-added: codex event normalization, retry classification edges, session
// fallback disabling, and the request-body edges
// ---------------------------------------------------------------------------

/// Port-added: an unknown terminal status drops from the response; a `null`
/// response rides and any other shape drops.
#[tokio::test]
async fn the_codex_terminal_status_normalizes_like_the_wire() {
    let _guard = suite();
    let token = mock_token("acc_test");

    // An unknown status drops so the shared processor reads the bare stop.
    let mock = MockHttpClient::new();
    let mut frames = hello_frames();
    frames.push(json!({
        "type": "response.completed",
        "response": {"id": "resp_t", "status": "weird_status", "usage": {"input_tokens": 5}},
    }));
    mount_sse_route(&mock, sse_body(&frames, false));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = sse_options(&mock, &token);
    let result = sse_run(&model, &options).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    // A null response rides; a string response drops.
    for response in [Value::Null, json!("a string")] {
        let token = mock_token("acc_test");
        let mock = MockHttpClient::new();
        let mut frames = hello_frames();
        frames.push(json!({ "type": "response.completed", "response": response }));
        mount_sse_route(&mock, sse_body(&frames, false));
        let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
        let options = sse_options(&mock, &token);
        let result = sse_run(&model, &options).await;
        assert_eq!(result.stop_reason, StopReason::Stop, "{response}");
    }
}

/// Port-added: a frame without a `type` field rides through as skipped.
#[tokio::test]
async fn a_codex_event_without_a_type_skips() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    let mut frames = hello_frames();
    frames.push(json!({"no_type_here": true}));
    frames.push(completed_terminal(None, None));
    mount_sse_route(&mock, sse_body(&frames, false));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = sse_options(&mock, &token);
    let result = sse_run(&model, &options).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
}

/// Port-added: the reasoning field reads the off map — the mapped off entry,
/// the null marker dropping the field, the request-less off fallback spelling
/// effort alone, and the empty-summary alias riding `auto`. Upstream's
/// `buildRequestBody` guards both branches against a null effort
/// (`if (effort !== null)`, `model.thinkingLevelMap?.off !== null`), so the
/// null off marker leaves the whole `reasoning` field off the wire rather
/// than sending a null effort, and the request-less branch's object literal
/// carries no summary.
#[tokio::test]
async fn the_codex_reasoning_field_reads_the_off_map() {
    let map: Option<Value> = None;
    for (map, effort, summary, expected) in [
        (
            map,
            Some(ModelThinkingLevel::Off),
            None,
            Some(json!({"effort": "none", "summary": "auto"})),
        ),
        (
            Some(json!({"off": "low"})),
            Some(ModelThinkingLevel::Off),
            None,
            Some(json!({"effort": "low", "summary": "auto"})),
        ),
        (
            Some(json!({"off": null})),
            Some(ModelThinkingLevel::Off),
            None,
            None,
        ),
        (Some(json!({"off": null})), None, None, None),
        (None, None, None, Some(json!({"effort": "none"}))),
        (
            Some(json!({"off": "low"})),
            None,
            None,
            Some(json!({"effort": "low"})),
        ),
        (
            None,
            Some(ModelThinkingLevel::Off),
            Some(""),
            Some(json!({"effort": "none", "summary": "auto"})),
        ),
    ] {
        let _guard = suite();
        let token = mock_token("acc_test");
        let mock = MockHttpClient::new();
        mount_sse_route(&mock, hello_sse_body());
        let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
        model.thinking_level_map = map
            .as_ref()
            .map(|map| serde_json::from_value(map.clone()).expect("the level map"));
        let mut options = sse_options(&mock, &token);
        options.reasoning_effort = effort;
        options.reasoning_summary = summary.map(str::to_owned);
        let _ = sse_run(&model, &options).await;
        let body = recorded_codex_body(&mock, 0);
        assert_eq!(body.get("reasoning").cloned(), expected, "{map:?}");
    }
}

/// Port-added: a terminal rate-limit 429 fails without retrying; a
/// connection-refused 400 retries through the error-text pattern.
#[tokio::test(start_paused = true)]
async fn the_retry_classification_reads_its_patterns() {
    // A 429 whose text names a spent subscription does not retry. The
    // phase-1 guard is dropped before the phase-2 suite() call — the mutex
    // is non-reentrant and shadowing keeps the first guard alive to scope
    // end, so the release must be explicit.
    let phase_one_guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/codex/responses"))
        .respond(json_response(
            429,
            &json!({"error": {"message": "Monthly usage limit reached"}}),
        ));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.max_retries = Some(2);
    let result = sse_run(&model, &options).await;
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(mock.request_count(), 1);

    // A 400 whose text matches the transient pattern retries.
    drop(phase_one_guard);
    let _guard = suite();
    let token = mock_token("acc_test");
    let (mock, mut requests) = connection_refused_mock(hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.max_retries = Some(1);
    let settle = spawned_sse_run(&model, &options);
    requests.recv().await.expect("the first request");
    let before = Instant::now();
    tokio::time::advance(Duration::from_secs(2)).await;
    requests.recv().await.expect("the retry fires");
    let _ = before;
    let result = settle.await.expect("the settle task");
    assert_eq!(message_text(&result), Some("Hello"));
    assert_eq!(mock.request_count(), 2);
}

/// A two-response mock: request one answers 400 with the transient wording,
/// later ones stream the SSE run. Signals each request on the channel.
fn connection_refused_mock(
    sse: String,
) -> (MockHttpClient, tokio::sync::mpsc::UnboundedReceiver<()>) {
    let mock = MockHttpClient::new();
    let (signal, receiver) = tokio::sync::mpsc::unbounded_channel();
    let served = Arc::new(AtomicUsize::new(0));
    let served_for_body = Arc::clone(&served);
    mock.on(move |request| {
        let _ = signal.send(());
        request.url.contains("/codex/responses")
    })
    .respond_fn(move |_request| {
        let served = Arc::clone(&served_for_body);
        let sse = sse.clone();
        async move {
            let index = served.fetch_add(1, Ordering::AcqRel) + 1;
            if index == 1 {
                return Ok(json_response(
                    400,
                    &json!({"error": {"message": "connection refused"}}),
                ));
            }
            Ok(MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_body(sse))
        }
    });
    (mock, receiver)
}

/// Port-added: a blank or unparsable `Retry-After` header reads as absent and
/// the exponential backoff fires; malformed HTTP dates parse as absent.
#[tokio::test(start_paused = true)]
async fn the_retry_after_header_reads_bad_spellings_as_absent() {
    for (header_value, name) in [
        ("", "blank"),
        ("garbage", "junk"),
        ("Mon, 15 Jun 2026 99:00:00 GMT", "bad hour"),
    ] {
        let _guard = suite();
        let token = mock_token("acc_test");
        let (mock, mut requests) = retry_after_mock(header_value, hello_sse_body());
        let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
        let mut options = sse_options(&mock, &token);
        options.max_retries = Some(1);
        let stream = openai_codex_responses::stream(&model, &codex_context_now(), Some(&options));
        let settle = tokio::spawn({
            let stream = stream.clone();
            async move { drain_and_settle(&stream).await }
        });
        requests.recv().await.expect("the first request arrives");
        let before = Instant::now();
        requests.recv().await.expect("the retry fires");
        // Every bad spelling falls back to the exponential backoff.
        assert_eq!((Instant::now() - before).as_millis(), 1_000, "{name}");
        let result = settle.await.expect("the settle task");
        assert_eq!(message_text(&result), Some("Hello"), "{name}");
    }
}

/// A two-response mock whose first answer carries the given `retry-after`
/// header and the later ones stream the SSE run, signalling each request.
fn retry_after_mock(
    header_value: &'static str,
    sse: String,
) -> (MockHttpClient, tokio::sync::mpsc::UnboundedReceiver<()>) {
    let mock = MockHttpClient::new();
    let (signal, receiver) = tokio::sync::mpsc::unbounded_channel();
    let served = Arc::new(AtomicUsize::new(0));
    let served_for_body = Arc::clone(&served);
    mock.on(move |request| {
        let _ = signal.send(());
        request.url.contains("/codex/responses")
    })
    .respond_fn(move |_request| {
        let served = Arc::clone(&served_for_body);
        let sse = sse.clone();
        async move {
            let index = served.fetch_add(1, Ordering::AcqRel) + 1;
            if index == 1 {
                return Ok(MockResponse::status(500)
                    .with_header("retry-after", header_value)
                    .with_body(r#"{"error":{"message":"overloaded"}}"#));
            }
            Ok(MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_body(sse))
        }
    });
    (mock, receiver)
}

/// Port-added: a base URL already carrying `/codex/responses` stays put, and
/// a `ws` scheme rides unchanged into the WebSocket URL.
#[tokio::test]
async fn the_codex_urls_accept_their_edge_spellings() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    model.base_url = "https://codex.example/backend/codex/responses".to_owned();
    let options = sse_options(&mock, &token);
    let result = sse_run(&model, &options).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(
        mock.recorded()[0].url,
        "https://codex.example/backend/codex/responses"
    );

    // A ws:// base rides into the WebSocket URL unchanged.
    let transport = Arc::new(MockWebSocketTransport::new());
    openai_codex_responses::set_websocket_transport(Some(transport.clone()));
    let fetch = MockHttpClient::new();
    let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    model.base_url = "ws://codex.example/backend".to_owned();
    let options = plain_ws_options(&fetch, &token);
    let (stream, peer) = plain_turn(&transport, &model, &options).await;
    peer_sends(&peer, &[completed_response(1)]);
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    let connect = peer.connect_request();
    assert_eq![connect.url, "ws://codex.example/backend/codex/responses"];
}

/// Port-added: an explicit close drops a session's cached connections but
/// keeps its stats entry — upstream's `closeOpenAICodexWebSocketSessions`
/// closes sockets and clears the session cache only, the stats outlive both
/// the settle and the close (`getOpenAICodexWebSocketDebugStats` still
/// returns them), and `resetOpenAICodexWebSocketDebugStats` is the entry's
/// only drop path.
#[tokio::test]
async fn closing_a_session_drops_its_connections_but_keeps_its_stats() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let options = cached_ws_options(&fetch, &token, "close-me");

    let _result = cached_turn(&transport, &model, &options, 1, 1).await;
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("close-me")
        .expect("the settle keeps the session's stats entry");
    assert_eq!(stats.connections_created, 1);

    // The close drops the cached connection, so the session's next turn
    // opens a fresh socket; the stats entry rides on.
    openai_codex_responses::close_openai_codex_websocket_sessions(Some("close-me"));
    let _result = cached_turn(&transport, &model, &options, 2, 2).await;
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("close-me")
        .expect("close leaves the stats entry");
    assert_eq!(stats.connections_created, 2);
    assert_eq!(stats.connections_reused, 0);

    // The debug-stats reset drops the entry.
    openai_codex_responses::reset_openai_codex_websocket_debug_stats(Some("close-me"));
    assert!(openai_codex_responses::get_openai_codex_websocket_debug_stats("close-me").is_none());
    openai_codex_responses::close_openai_codex_websocket_sessions(None);
}

/// Port-added: a pre-cancelled signal aborts the codex stream before any
/// dispatch, through both the setup check and the retry loop.
#[tokio::test]
async fn a_pre_cancelled_signal_aborts_the_codex_stream() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let signal = CancellationToken::new();
    signal.cancel();
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.transport_options.signal = Some(signal);

    let result = sse_run(&model, &options).await;
    assert_eq!(result.stop_reason, StopReason::Aborted);
    assert_eq![result.error_message.as_deref(), Some("Request was aborted")];
    assert_eq![mock.request_count(), 0];
}

/// Port-added: the payload hook replaces the codex request body, and a
/// hook-swapped `store: true` books the store-true stat.
#[tokio::test]
async fn the_codex_payload_hook_replaces_the_body_and_books_the_store_stat() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    model.base_url = "https://chatgpt.com/backend-api/codex".to_owned();
    let mut options = sse_options(&mock, &token);
    options.session_id = Some("hooked-body".to_owned());
    let hook: pi_ai::types::OnPayload = pi_ai::types::OnPayload::new(|mut payload, _model| {
        if let Value::Object(ref mut body) = payload {
            body.insert("store".to_owned(), json!(true));
        }
        Box::pin(async { Some(payload) })
    });
    options.transport_options.on_payload = Some(hook);
    let _ = sse_run(&model, &options).await;
    let body = recorded_codex_body(&mock, 0);
    assert_eq!(body["store"], json!(true));
}

/// Port-added: once a session fell back to SSE the WebSocket stays disabled
/// for that session's later streams.
#[tokio::test]
async fn the_session_fallback_disables_the_websocket_for_later_streams() {
    let (guard, token, transport, fetch, model) = ws_fallback_suite();
    let _guard = guard;
    let options = cached_ws_options(&fetch, &token, "fallback-lock");

    // Turn one falls back: the peer closes before any output.
    let (stream, peer) = plain_turn(&transport, &model, &options).await;
    peer.close(4000, "backend restarting")
        .expect("the peer closes");
    let result = drain_and_settle(&stream).await;
    assert_eq!(message_text(&result), Some("Hello"));

    // Turn two goes straight to SSE: no further connect happens.
    let stream = openai_codex_responses::stream(&model, &codex_context_at(2), Some(&options));
    let result = drain_and_settle(&stream).await;
    assert_eq!(message_text(&result), Some("Hello"));
    assert!(transport.next_peer().is_none());
    assert_eq!(fetch.request_count(), 2);
}

/// Port-added: a cached continuation resets when the body no longer matches
/// — the second turn sends in full again.
#[tokio::test]
async fn the_cached_continuation_resets_when_the_body_changes() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let options = cached_ws_options(&fetch, &token, "body-changed");

    let (stream, mut peer) = plain_turn(&transport, &model, &options).await;
    peer_sends(&peer, &[completed_response(1)]);
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);

    // A different system prompt changes the request body, so the delta
    // lookup drops the continuation and sends in full.
    let changed = Context {
        system_prompt: Some("A different persona.".to_owned()),
        messages: vec![user_message_at("Now finish", 2)],
        tools: None,
    };
    let stream = openai_codex_responses::stream(&model, &changed, Some(&options));
    // The cached connection is reused; the changed body sends in full.
    let body = next_sent_body(&mut peer).await;
    peer_sends(&peer, &[completed_response(2)]);
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert![body.get("previous_response_id").is_none(), "{}", body];
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("body-changed")
        .expect("the session records stats");
    assert_eq!(stats.full_context_requests, 2);
    assert_eq!(stats.connections_reused, 1);
}

/// Port-added: codex `stream_simple` maps the tool choices, and a stream
/// without any api key fails with the provider-named setup error.
#[tokio::test]
async fn codex_simple_options_and_the_setup_error_surface() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let options = SimpleStreamOptions {
        transport_options: TransportOptions {
            http_client: Some(Arc::new(fetch.clone())),
            ..TransportOptions::default()
        },
        api_key: Some(token.clone()),
        tool_choice: Some(pi_ai::types::ToolChoice::Auto),
        ..SimpleStreamOptions::default()
    };
    let stream =
        openai_codex_responses::stream_simple(&model, &codex_context_at(1), Some(&options));
    let mut peer = next_peer(&transport).await;
    let body = next_sent_body(&mut peer).await;
    peer_sends(&peer, &[completed_response(1)]);
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(body["tool_choice"], json!("auto"));

    // The `none` choice rides the wire's "none" spelling.
    let none_options = SimpleStreamOptions {
        transport_options: TransportOptions {
            http_client: Some(Arc::new(fetch.clone())),
            ..TransportOptions::default()
        },
        api_key: Some(token.clone()),
        tool_choice: Some(pi_ai::types::ToolChoice::None),
        ..SimpleStreamOptions::default()
    };
    let stream =
        openai_codex_responses::stream_simple(&model, &codex_context_at(2), Some(&none_options));
    let mut peer = next_peer(&transport).await;
    let body = next_sent_body(&mut peer).await;
    peer_sends(&peer, &[completed_response(3)]);
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(body["tool_choice"], json!("none"));

    // No credential: the setup error stream reports the provider.
    let result = drain_and_settle(&openai_codex_responses::stream_simple(
        &model,
        &codex_context_at(3),
        Some(&SimpleStreamOptions::default()),
    ))
    .await;
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("No API key for provider: openai-codex")
    );
}

/// Port-added: an SSE body that fails mid-read fails the codex stream after
/// its events rode out.
#[tokio::test]
async fn a_mid_stream_sse_read_failure_fails_the_codex_stream() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let frames = serde_json::to_string(&json!({
        "type": "response.output_item.added",
        "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
    }))
    .expect("the frame");
    let first = Bytes::from(format!("data: {frames}\n\n"));
    let client = SeamedAfterFrameClient {
        first,
        then: HttpError::Transport("dead server".to_owned()),
    };
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = OpenAiCodexResponsesOptions {
        transport_options: TransportOptions {
            http_client: Some(Arc::new(client)),
            ..TransportOptions::default()
        },
        api_key: Some(token),
        transport: Some(Transport::Sse),
        ..OpenAiCodexResponsesOptions::default()
    };

    let result = sse_run(&model, &options).await;
    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the read failure");
    assert![message.contains("dead server"), "{message}"];
}

#[derive(Debug)]
struct SeamedAfterFrameClient {
    first: Bytes,
    then: HttpError,
}

impl HttpClient for SeamedAfterFrameClient {
    fn execute(&self, _request: HttpRequest) -> BoxHttpFuture<Result<HttpResponse, HttpError>> {
        let chunks = vec![Ok(self.first.clone()), Err(self.then.clone())];
        let response = HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: HttpByteStream::from_chunks(chunks),
        };
        Box::pin(async move { Ok(response) })
    }
}

/// Port-added: a base URL the URL parser rejects fails the codex WebSocket
/// URL resolution, upstream's `resolveCodexWebSocketUrl` throw shape.
#[tokio::test]
async fn an_unparsable_codex_websocket_base_url_fails_the_stream() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let fetch = MockHttpClient::new();
    let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    model.base_url = "not a url".to_owned();
    let options = websocket_options(&fetch, &token, Transport::Websocket);

    let result = sse_run(&model, &options).await;
    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the URL failure");
    assert![message.contains("Invalid URL"), "{message}"];
}

/// The connect-aborting transport: every handshake fails with the abort
/// error, upstream's pre-aborted `WebSocket` constructor.
#[derive(Debug)]
struct AbortConnectTransport;

impl WebSocketTransport for AbortConnectTransport {
    fn connect(
        &self,
        _request: WebSocketRequest,
    ) -> BoxHttpFuture<Result<Box<dyn WebSocketConnection>, WebSocketError>> {
        Box::pin(async move { Err(WebSocketError::Aborted) })
    }
}

/// Port-added: an aborted WebSocket handshake records the failure and the
/// codex stream falls back to SSE, upstream's `connectWebSocket` catch
/// mapping plus the pre-start fallback machine.
#[tokio::test]
async fn an_aborted_websocket_connect_surfaces_the_abort() {
    let _guard = suite();
    let token = mock_token("acc_test");
    openai_codex_responses::set_websocket_transport(Some(Arc::new(AbortConnectTransport)));
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    model.base_url = "http://codex.example/backend".to_owned();
    let mut options = sse_options(&mock, &token);
    options.transport = Some(Transport::Websocket);
    options.cache_retention = Some(CacheRetention::Long);
    options.session_id = Some("aborted-connect".to_owned());

    let result = sse_run(&model, &options).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("aborted-connect")
        .expect("the fallback records the failure");
    assert![
        stats
            .last_web_socket_error
            .as_deref()
            .unwrap_or("")
            .contains("aborted")
    ];
}

/// The broken-connection transport: the handshake opens, and the first read
/// fails with the stored transport error.
#[derive(Debug)]
struct BrokenReadConnection;

impl WebSocketConnection for BrokenReadConnection {
    fn send(&mut self, _message: WebSocketMessage) -> BoxedFuture<'_, Result<(), WebSocketError>> {
        Box::pin(async move { Ok(()) })
    }

    fn next_message(&mut self) -> BoxedFuture<'_, Result<WebSocketMessage, WebSocketError>> {
        Box::pin(async move { Err(WebSocketError::Transport("line dropped".to_owned())) })
    }

    fn close(
        &mut self,
        _code: u16,
        _reason: String,
    ) -> BoxedFuture<'_, Result<(), WebSocketError>> {
        Box::pin(async move { Ok(()) })
    }
}

/// The transport whose handshakes succeed and whose reads fail.
#[derive(Debug)]
struct BrokenReadTransport;

impl WebSocketTransport for BrokenReadTransport {
    fn connect(
        &self,
        _request: WebSocketRequest,
    ) -> BoxHttpFuture<Result<Box<dyn WebSocketConnection>, WebSocketError>> {
        let connection: Box<dyn WebSocketConnection> = Box::new(BrokenReadConnection);
        Box::pin(async move { Ok(connection) })
    }
}

/// Port-added: a WebSocket read failure that is neither a peer close nor an
/// abort fails the codex stream with the transport's wording, upstream's
/// `parseWebSocket` catch-all arm.
#[tokio::test]
async fn a_websocket_transport_read_error_fails_the_codex_stream() {
    let _guard = suite();
    let token = mock_token("acc_test");
    openai_codex_responses::set_websocket_transport(Some(Arc::new(BrokenReadTransport)));
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    model.base_url = "http://codex.example/backend".to_owned();
    let mut options = sse_options(&mock, &token);
    options.transport = Some(Transport::Websocket);
    options.cache_retention = Some(CacheRetention::Long);
    options.session_id = Some("broken-read".to_owned());

    let result = sse_run(&model, &options).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("broken-read")
        .expect("the fallback records the failure");
    assert![
        stats
            .last_web_socket_error
            .as_deref()
            .unwrap_or("")
            .contains("line dropped")
    ];
}

/// Port-added: a peer close with the message-too-big code and an empty
/// reason records the default-reason failure and the codex stream falls
/// back to SSE, upstream's `extractWebSocketCloseError`.
#[tokio::test]
async fn a_peer_close_with_the_too_big_code_names_it() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let transport = Arc::new(MockWebSocketTransport::new());
    openai_codex_responses::set_websocket_transport(Some(transport.clone()));
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, hello_sse_body());
    let mut model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    model.base_url = "http://codex.example/backend".to_owned();
    let mut options = sse_options(&mock, &token);
    options.transport = Some(Transport::Websocket);
    options.cache_retention = Some(CacheRetention::Long);
    options.session_id = Some("too-big-close".to_owned());

    let (stream, peer) = plain_turn(&transport, &model, &options).await;
    peer.close(1009, "").expect("the peer closes");
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("too-big-close")
        .expect("the fallback records the failure");
    assert![
        stats
            .last_web_socket_error
            .as_deref()
            .unwrap_or("")
            .contains("1009 message too big")
    ];
}

/// Port-added: a 429 whose `Retry-After` names an IMF fixdate sleeps the
/// wall-clock distance, upstream's `Date.parse` branch of the retry delay.
#[tokio::test(start_paused = true)]
async fn a_codex_429_with_an_http_date_retry_after_rides_its_delay() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/codex/responses"))
        .respond_sequence(vec![
            MockResponse::status(429)
                .with_header("retry-after", "Wed, 21 Oct 2026 07:28:00 GMT")
                .with_body(json!({"error": {"message": "slow down"}}).to_string()),
            MockResponse::status(200)
                .with_header("content-type", "text/event-stream")
                .with_body(hello_sse_body()),
        ]);
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let mut options = sse_options(&mock, &token);
    options.max_retries = Some(1);
    // The far-future date waits far past the default cap; a disabled cap
    // (`max: 0`) lets the parsed delay ride.
    options.max_retry_delay_ms = Some(0);

    let settle = spawned_sse_run(&model, &options);
    // The date parse waits until the fixed date; a broad advance covers it.
    tokio::time::advance(Duration::from_hours(2160)).await;
    let result = settle.await.expect("the settle task");
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(mock.request_count(), 2);
}

/// Port-added: a codex SSE run that ends without a terminal event reports
/// the missing terminal, upstream's `processResponsesStream` EOF check
/// riding the mapped pump's clean end.
#[tokio::test]
async fn a_codex_sse_run_without_a_terminal_reports_the_missing_stop() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let mock = MockHttpClient::new();
    mount_sse_route(&mock, sse_body(&hello_frames(), false));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = sse_options(&mock, &token);

    let result = sse_run(&model, &options).await;
    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the missing terminal");
    assert![message.contains("terminal response event"), "{message}"];
}

/// Port-added: a completed terminal carrying the failed status settles the
/// error stop and the unknown-error wording, upstream's status mapping plus
/// `assertSuccessfulOutput`'s failure arm.
#[tokio::test]
async fn a_codex_failed_terminal_surfaces_the_unknown_error() {
    let _guard = suite();
    let token = mock_token("acc_test");
    let (result, _mock) = framed_run(
        &token,
        &[json!({
            "type": "response.completed",
            "response": {
                "status": "failed",
                "incomplete_details": Value::Null,
                "usage": { "input_tokens": 5, "output_tokens": 3, "total_tokens": 8 },
            },
        })],
    )
    .await;
    assert_eq![result.stop_reason, StopReason::Error];
    let message = result.error_message.expect("the assertion failure");
    assert![message.contains("unknown error"), "{message}"];
}

/// Port-added: a codex error event without a message or code names the raw
/// event JSON, and a response.failed terminal without an error names the
/// default wording, upstream's `mapCodexEvents` detail fallbacks.
#[tokio::test]
async fn a_codex_error_event_without_details_names_the_wire_shape() {
    let _guard = suite();
    let token = mock_token("acc_test");

    // An error event without details: the detail rides the event's JSON.
    let (result, _mock) = framed_run(&token, &[json!({ "type": "error" })]).await;
    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the error event");
    assert![message.contains("Codex error:"), "{message}"];
    assert![message.contains("\"type\":\"error\""), "{message}"];

    // A response.failed terminal without an error object: the default name.
    let mock = MockHttpClient::new();
    let mut frames = hello_frames();
    frames.push(json!({
        "type": "response.failed",
        "response": { "status": "failed" },
    }));
    mount_sse_route(&mock, sse_body(&frames, false));
    let model = codex_model("gpt-5.1-codex", "GPT-5.1 Codex");
    let options = sse_options(&mock, &token);
    let result = sse_run(&model, &options).await;
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq![
        result.error_message.as_deref(),
        Some("Codex response failed")
    ];
}

/// Port-added: an idle-timed release closes its connection five minutes
/// later — the spawned expiry task closes the idle socket, a session the
/// close already removed exits early, and a busy release's timer schedules
/// from the release only. Upstream's `scheduleSessionWebSocketExpiry`.
#[tokio::test(start_paused = true)]
async fn the_codex_idle_expiry_closes_the_released_connection() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let options = cached_ws_options(&fetch, &token, "idle-expiry");

    // Turn one settles and releases its connection into the cache with the
    // idle timer armed.
    let _result = cached_turn(&transport, &model, &options, 1, 1).await;

    // The idle timer closes the cached connection; the session's next turn
    // opens a fresh socket.
    tokio::time::advance(Duration::from_secs(5 * 60 + 5)).await;
    let stream = openai_codex_responses::stream(&model, &codex_context_at(2), Some(&options));
    let mut peer = next_peer(&transport).await;
    let body = next_sent_body(&mut peer).await;
    assert![body.get("previous_response_id").is_none(), "{}", body];
    peer_sends(&peer, &[completed_response(2)]);
    let result = drain_and_settle(&stream).await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("idle-expiry")
        .expect("the session records stats");
    assert_eq![stats.connections_created, 2];

    // A timer firing after the session's entries are gone exits early.
    openai_codex_responses::close_openai_codex_websocket_sessions(Some("idle-expiry"));
    tokio::time::advance(Duration::from_secs(5 * 60 + 5)).await;
    openai_codex_responses::close_openai_codex_websocket_sessions(None);
}

/// Port-added: the session-resource cleanup closes a session's cached
/// connections — the cleanup the codex state registers on its first lock,
/// upstream's `registerSessionResourceCleanup(closeOpenAICodexWebSocketSessions)`.
#[tokio::test]
async fn the_codex_session_cleanup_closes_its_sockets() {
    let (guard, token, transport, fetch, model) = ws_suite();
    let _guard = guard;
    let options = cached_ws_options(&fetch, &token, "cleanup-me");

    let _result = cached_turn(&transport, &model, &options, 1, 1).await;

    // The session teardown runs the registered cleanup: the cached socket
    // closes and the session's next turn reconnects.
    pi_ai::session_resources::cleanup_session_resources(Some("cleanup-me"))
        .expect("the cleanup runs");
    let _result = cached_turn(&transport, &model, &options, 2, 2).await;
    let stats = openai_codex_responses::get_openai_codex_websocket_debug_stats("cleanup-me")
        .expect("the session records stats");
    assert_eq![stats.connections_created, 2];
}

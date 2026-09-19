//! The seam-level mocking harness the wire-API test suites use where upstream
//! stubs `globalThis.fetch` (`vi.stubGlobal("fetch")`) and `globalThis.WebSocket`.
//!
//! Public so integration tests in `tests/` and later port tickets can share
//! it; nothing here ships in a provider request path.

use std::future::Future;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::http::client::{
    BoxHttpFuture, HttpByteStream, HttpClient, HttpError, HttpRequest, HttpResponse,
};
use crate::http::websocket::{
    WebSocketConnection, WebSocketError, WebSocketMessage, WebSocketOutbound, WebSocketRequest,
    WebSocketTransport,
};
use crate::types::BoxedFuture;

/// A canned HTTP response a mock route answers with.
#[derive(Clone, Debug)]
pub struct MockResponse {
    /// The response status code.
    pub status: u16,
    /// The response headers, duplicates and order preserved.
    pub headers: Vec<(String, String)>,
    /// The response body.
    pub body: MockBody,
}

/// A canned response body: whole bytes, or an explicit chunk sequence the
/// stream yields verbatim — chunk-boundary tests split payloads here.
#[derive(Clone, Debug)]
pub enum MockBody {
    /// No body.
    Empty,
    /// One chunk with the whole body.
    Bytes(Bytes),
    /// The chunks the stream yields in order.
    Chunks(Vec<Bytes>),
}

impl MockResponse {
    /// A response with the given status and no body.
    #[must_use]
    pub const fn status(status: u16) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: MockBody::Empty,
        }
    }

    /// Add a response header; duplicates append, matching the wire.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Set a whole-body response.
    #[must_use]
    pub fn with_body(mut self, body: impl Into<Bytes>) -> Self {
        self.body = MockBody::Bytes(body.into());
        self
    }

    /// Set an explicit chunk sequence the stream yields verbatim.
    #[must_use]
    pub fn with_chunks<I, B>(mut self, chunks: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: Into<Bytes>,
    {
        self.body = MockBody::Chunks(chunks.into_iter().map(Into::into).collect());
        self
    }
}

/// A JSON body response, the shape `vi.stubGlobal("fetch")` tests hand back
/// for error statuses.
#[must_use]
pub fn json_response(status: u16, body: &serde_json::Value) -> MockResponse {
    MockResponse::status(status)
        .with_header("content-type", "application/json")
        .with_body(body.to_string())
}

/// A `text/event-stream` response over framed events, the port of upstream's
/// `createSseResponse` helper (`test/anthropic-sse-parsing.test.ts`).
///
/// Each pair becomes an `event:`/`data:` block, blocks joined with a blank
/// line.
#[must_use]
pub fn sse_response(status: u16, events: &[(&str, &str)]) -> MockResponse {
    let body = events
        .iter()
        .map(|(event, data)| format!("event: {event}\ndata: {data}\n"))
        .collect::<Vec<String>>()
        .join("\n");
    MockResponse::status(status)
        .with_header("content-type", "text/event-stream")
        .with_body(body)
}

/// The matcher a route answers with: [`Single`] always, [`Sequence`] in mount
/// order until exhausted, [`Handler`] computing per request.
enum Responder {
    Single(MockResponse),
    Sequence(Vec<MockResponse>),
    Handler(HandlerArc),
}

/// A handler route's per-request responder.
type HandlerFn =
    dyn Fn(&RecordedRequest) -> BoxHttpFuture<Result<MockResponse, HttpError>> + Send + Sync;

/// A shared handler responder.
type HandlerArc = Arc<HandlerFn>;

/// One mounted route: a matcher plus what it answers with.
struct MockRoute {
    matches: Box<dyn Fn(&HttpRequest) -> bool + Send + Sync>,
    responder: Mutex<Responder>,
}

/// What one seam request carried, the snapshot assertions read back.
#[derive(Clone, Debug)]
pub struct RecordedRequest {
    /// The request method.
    pub method: crate::http::client::HttpMethod,
    /// The request URL.
    pub url: String,
    /// The request headers.
    pub headers: Vec<(String, String)>,
    /// The request body.
    pub body: Option<Bytes>,
}

/// The seam-level HTTP client mock: routes answering canned responses, plus
/// the record of every request it has served.
#[derive(Clone)]
pub struct MockHttpClient(Arc<MockInner>);

impl std::fmt::Debug for MockHttpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockHttpClient")
            .field("served", &self.request_count())
            .finish_non_exhaustive()
    }
}

struct MockInner {
    routes: Mutex<Vec<Arc<MockRoute>>>,
    recorded: Mutex<Vec<RecordedRequest>>,
}

/// A route under construction, mounted by its `respond*` call.
pub struct MockRouteBuilder<'a> {
    client: &'a MockHttpClient,
    matches: Box<dyn Fn(&HttpRequest) -> bool + Send + Sync>,
}

impl std::fmt::Debug for MockRouteBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MockRouteBuilder(..)")
    }
}

impl MockHttpClient {
    /// An empty mock: every request fails with a no-route-matched transport
    /// error, the "ambient fetch must not be called" guard upstream's
    /// fallback fetch throws.
    #[must_use]
    pub fn new() -> Self {
        Self(Arc::new(MockInner {
            routes: Mutex::new(Vec::new()),
            recorded: Mutex::new(Vec::new()),
        }))
    }

    /// Start a route: when the matcher holds for a request, the response
    /// mounted next answers it.
    pub fn on(
        &self,
        matches: impl Fn(&HttpRequest) -> bool + Send + Sync + 'static,
    ) -> MockRouteBuilder<'_> {
        MockRouteBuilder {
            client: self,
            matches: Box::new(matches),
        }
    }

    /// The requests this mock has served so far, oldest first.
    #[must_use]
    pub fn recorded(&self) -> Vec<RecordedRequest> {
        self.0
            .recorded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// How many requests this mock has served.
    #[must_use]
    pub fn request_count(&self) -> usize {
        self.0
            .recorded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    fn mount(
        &self,
        matches: Box<dyn Fn(&HttpRequest) -> bool + Send + Sync>,
        responder: Responder,
    ) {
        self.0
            .routes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Arc::new(MockRoute {
                matches,
                responder: Mutex::new(responder),
            }));
    }
}

impl Default for MockHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl MockRouteBuilder<'_> {
    /// Mount the route to always answer with this response.
    pub fn respond(self, response: MockResponse) {
        self.client.mount(self.matches, Responder::Single(response));
    }

    /// Mount the route to answer successive requests with successive
    /// responses; a request past the end of the sequence fails.
    pub fn respond_sequence(self, responses: Vec<MockResponse>) {
        self.client
            .mount(self.matches, Responder::Sequence(responses));
    }

    /// Mount the route to compute each response — the seam-level shape of a
    /// `vi.fn(fetch)` whose implementation inspects the request; sleep inside
    /// the handler with `tokio::time::sleep` to exercise timeouts against a
    /// paused clock.
    pub fn respond_fn<F, Fut>(self, handler: F)
    where
        F: Fn(&RecordedRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<MockResponse, HttpError>> + Send + 'static,
    {
        let responder: HandlerArc = Arc::new(move |request| {
            let future = handler(request);
            Box::pin(future)
        });
        self.client
            .mount(self.matches, Responder::Handler(responder));
    }
}

/// Yield a mock body's chunks in order, racing the request's cancellation:
/// an aborted request fails every read, and the chunks yield then end the
/// stream. Handlers model live-stream hangs and mid-stream aborts with
/// `tokio::time::sleep` and the cancellation token.
fn mock_body_stream(body: MockBody, signal: CancellationToken) -> HttpByteStream {
    let chunks: Vec<Result<Bytes, HttpError>> = match body {
        MockBody::Empty => Vec::new(),
        MockBody::Bytes(chunk) => vec![Ok(chunk)],
        MockBody::Chunks(chunks) => chunks.into_iter().map(Ok).collect(),
    };
    let mut chunks = chunks.into_iter();
    HttpByteStream::new(futures_util::stream::poll_fn(move |_cx| {
        if signal.is_cancelled() {
            return std::task::Poll::Ready(Some(Err(HttpError::Aborted)));
        }
        chunks.next().map_or(std::task::Poll::Ready(None), |chunk| {
            std::task::Poll::Ready(Some(chunk))
        })
    }))
}

/// A matched route's answer detached from the route locks: a canned
/// response ready to return, or a handler to run — handlers await after the
/// locks drop, keeping the execute future Send.
enum Detached {
    Ready(Result<MockResponse, HttpError>),
    Handler(HandlerArc),
}

impl HttpClient for MockHttpClient {
    fn execute(&self, request: HttpRequest) -> BoxHttpFuture<Result<HttpResponse, HttpError>> {
        let inner = self.0.clone();
        Box::pin(async move {
            if request.signal.is_cancelled() {
                return Err(HttpError::Aborted);
            }

            let recorded = RecordedRequest {
                method: request.method.clone(),
                url: request.url.clone(),
                headers: request.headers.clone(),
                body: request.body.clone(),
            };
            inner
                .recorded
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(recorded.clone());

            let detached = {
                let routes = inner
                    .routes
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let route = Arc::clone(
                    routes
                        .iter()
                        .find(|route| (route.matches)(&request))
                        .ok_or_else(|| {
                            HttpError::Transport(format!(
                                "no mock route matched {} {}",
                                request.method, request.url
                            ))
                        })?,
                );
                drop(routes);
                let mut responder = route
                    .responder
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match &mut *responder {
                    Responder::Single(response) => Detached::Ready(Ok(response.clone())),
                    Responder::Sequence(responses) => {
                        if responses.is_empty() {
                            Detached::Ready(Err(HttpError::Transport(
                                "mock route matched but its response sequence is exhausted"
                                    .to_owned(),
                            )))
                        } else {
                            Detached::Ready(Ok(responses.remove(0)))
                        }
                    }
                    Responder::Handler(handler) => Detached::Handler(Arc::clone(handler)),
                }
            };

            let response = match detached {
                Detached::Ready(response) => response?,
                Detached::Handler(handler) => {
                    // A handler models a slow server: the request timeout and
                    // the cancellation race it the way the real transport
                    // would race a pending response.
                    let timeout = tokio::time::sleep(std::time::Duration::from_millis(
                        request.timeout_ms.unwrap_or(u64::MAX),
                    ));
                    tokio::select! {
                        () = request.signal.cancelled() => return Err(HttpError::Aborted),
                        () = timeout => return Err(HttpError::Timeout),
                        outcome = handler(&recorded) => outcome?,
                    }
                }
            };
            Ok(HttpResponse {
                status: response.status,
                headers: response.headers,
                body: mock_body_stream(response.body, request.signal),
            })
        })
    }
}
/// What crosses from a mock WebSocket connection's client side to its server
/// peer.
enum InboundFrame {
    /// A text or binary message the client sent.
    Message(WebSocketMessage),
    /// The client closed with the wire's status code and reason.
    Close { code: u16, reason: String },
}

/// One mock WebSocket connection's server side holds: the messages the
/// client sent and the channel to push client-bound messages through.
#[derive(Debug)]
struct MockPeerHalves {
    outbound: mpsc::Receiver<WebSocketOutbound>,
    inbound: mpsc::Sender<InboundFrame>,
}

/// One mock WebSocket connection's server peer, the port of the event
/// listeners upstream's `MockWebSocket` test doubles expose.
///
/// The test pushes messages and closes through it while the code under test
/// drives the client connection.
#[derive(Debug)]
pub struct MockWebSocketPeer {
    connect: WebSocketRequest,
    outbound: mpsc::Receiver<WebSocketOutbound>,
    inbound: mpsc::Sender<InboundFrame>,
}

impl MockWebSocketPeer {
    /// The connect request the client side issued — handshake headers the
    /// wire API set are asserted through it.
    #[must_use]
    pub const fn connect_request(&self) -> &WebSocketRequest {
        &self.connect
    }

    /// The next message the client side sent, `None` once it dropped.
    pub async fn next_sent(&mut self) -> Option<WebSocketOutbound> {
        self.outbound.recv().await
    }

    /// Push a text message to the client side.
    ///
    /// # Errors
    /// Returns a [`WebSocketError::Transport`] once the client side dropped.
    pub fn send_text(&self, text: impl Into<String>) -> Result<(), WebSocketError> {
        self.inbound
            .try_send(InboundFrame::Message(WebSocketMessage::Text(text.into())))
            .map_err(|_| WebSocketError::Transport("mock websocket client dropped".to_owned()))
    }

    /// Push a binary message to the client side.
    ///
    /// # Errors
    /// Returns a [`WebSocketError::Transport`] once the client side dropped.
    pub fn send_binary(&self, data: Bytes) -> Result<(), WebSocketError> {
        self.inbound
            .try_send(InboundFrame::Message(WebSocketMessage::Binary(data)))
            .map_err(|_| WebSocketError::Transport("mock websocket client dropped".to_owned()))
    }

    /// Close from the server side; the client's next read fails with
    /// [`WebSocketError::Closed`] carrying this code and reason.
    ///
    /// # Errors
    /// Returns a [`WebSocketError::Transport`] once the client side dropped.
    pub fn close(&self, code: u16, reason: impl Into<String>) -> Result<(), WebSocketError> {
        self.inbound
            .try_send(InboundFrame::Close {
                code,
                reason: reason.into(),
            })
            .map_err(|_| WebSocketError::Transport("mock websocket client dropped".to_owned()))
    }
}

/// The mock [`WebSocketTransport`], the port of stubbed
/// `globalThis.WebSocket` constructors in the Codex stream tests.
///
/// Connects resolve into client connections whose server peers the test
/// takes with [`MockWebSocketTransport::next_peer`]; `connects` records
/// every connect request for header assertions.
#[derive(Debug)]
pub struct MockWebSocketTransport {
    connect_delay_ms: Option<u64>,
    connects: Arc<Mutex<Vec<WebSocketRequest>>>,
    peers: Arc<Mutex<Vec<MockPeerHalves>>>,
}

impl MockWebSocketTransport {
    /// A transport with no connect delay and no peers yet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            connect_delay_ms: None,
            connects: Arc::new(Mutex::new(Vec::new())),
            peers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Delay every connect by `ms` under `tokio::time::sleep`, and enforce
    /// the request's connect-handshake timeout the real transport does, so
    /// paused-clock tests can expire the handshake; a delayed connect still
    /// races the request's cancellation token.
    #[must_use]
    pub const fn with_connect_delay_ms(mut self, ms: u64) -> Self {
        self.connect_delay_ms = Some(ms);
        self
    }

    /// The connect requests this transport has accepted, oldest first.
    #[must_use]
    pub fn connects(&self) -> Vec<WebSocketRequest> {
        self.connects
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Take the server peer of the next accepted connect, oldest first.
    #[must_use]
    pub fn next_peer(&self) -> Option<MockWebSocketPeer> {
        let mut peers = self
            .peers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if peers.is_empty() {
            return None;
        }
        let connect = self
            .connects
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(0);
        let halves = peers.remove(0);
        drop(peers);
        Some(MockWebSocketPeer {
            connect,
            outbound: halves.outbound,
            inbound: halves.inbound,
        })
    }
}

impl Default for MockWebSocketTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSocketTransport for MockWebSocketTransport {
    fn connect(
        &self,
        request: WebSocketRequest,
    ) -> BoxHttpFuture<Result<Box<dyn WebSocketConnection>, WebSocketError>> {
        let connects = self.connects.clone();
        let peers = self.peers.clone();
        let connect_delay_ms = self.connect_delay_ms;
        Box::pin(async move {
            if request.signal.is_cancelled() {
                return Err(WebSocketError::Aborted);
            }
            if let Some(delay_ms) = connect_delay_ms {
                let connect_timeout_ms = request
                    .connect_timeout_ms
                    .unwrap_or(crate::http::websocket::DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS);
                tokio::select! {
                    () = request.signal.cancelled() => return Err(WebSocketError::Aborted),
                    () = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => {}
                    () = tokio::time::sleep(std::time::Duration::from_millis(connect_timeout_ms)) => {
                        return Err(WebSocketError::Timeout(connect_timeout_ms));
                    }
                }
            }

            let (client_tx, server_rx) = mpsc::channel::<WebSocketOutbound>(64);
            let (server_tx, client_rx) = mpsc::channel::<InboundFrame>(64);
            connects
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(request.clone());
            peers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(MockPeerHalves {
                    outbound: server_rx,
                    inbound: server_tx,
                });

            let connection: Box<dyn WebSocketConnection> = Box::new(MockWebSocketConnection {
                request,
                outbound: client_tx,
                inbound: client_rx,
                closed: false,
            });
            Ok(connection)
        })
    }
}

/// The client-side connection of a mock WebSocket pair.
#[derive(Debug)]
struct MockWebSocketConnection {
    request: WebSocketRequest,
    outbound: mpsc::Sender<WebSocketOutbound>,
    inbound: mpsc::Receiver<InboundFrame>,
    closed: bool,
}

impl WebSocketConnection for MockWebSocketConnection {
    fn send(&mut self, message: WebSocketMessage) -> BoxedFuture<'_, Result<(), WebSocketError>> {
        Box::pin(async move {
            if self.closed || self.outbound.is_closed() {
                return Err(WebSocketError::closed(1006, "mock websocket closed", false));
            }
            tokio::select! {
                () = self.request.signal.cancelled() => Err(WebSocketError::Aborted),
                result = self.outbound.send(WebSocketOutbound::Message(message)) => result
                    .map_err(|_| WebSocketError::closed(1006, "mock websocket closed", false)),
            }
        })
    }

    fn next_message(&mut self) -> BoxedFuture<'_, Result<WebSocketMessage, WebSocketError>> {
        Box::pin(async move {
            tokio::select! {
                () = self.request.signal.cancelled() => Err(WebSocketError::Aborted),
                frame = self.inbound.recv() => match frame {
                    Some(InboundFrame::Message(message)) => Ok(message),
                    Some(InboundFrame::Close { code, reason }) => {
                        self.closed = true;
                        Err(WebSocketError::closed(code, reason, true))
                    }
                    None => Err(WebSocketError::closed(1006, "mock websocket dropped", false)),
                },
            }
        })
    }

    fn close(&mut self, code: u16, reason: String) -> BoxedFuture<'_, Result<(), WebSocketError>> {
        Box::pin(async move {
            self.closed = true;
            self.outbound
                .send(WebSocketOutbound::Close { code, reason })
                .await
                .map_err(|_| WebSocketError::closed(1006, "mock websocket closed", false))
        })
    }
}

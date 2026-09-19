//! The HTTP seam and streaming substrate.
//!
//! [`HttpClient`] is the trait every wire-API adapter sends provider requests
//! through; the SSE decoder rides its byte streams; the WebSocket transport
//! carries the Codex wire API; and the seam-level mocks replace the
//! `globalThis.fetch` and `globalThis.WebSocket` stubs upstream tests use.
//!
//! The seam ports upstream's injectable `fetch`
//! (`packages/ai/src/types.ts:115`): `ProviderRequestOptions` carries an
//! optional client, the process default is reqwest 0.12 + rustls per the
//! stack decision, and one shared SSE decoder replaces the per-module framing
//! loops upstream repeats. The injected clock is tokio's — every timeout and
//! backoff sleep here runs on `tokio::time::sleep`, so tests pause the clock
//! with `tokio::time::pause` and the fake-timer suites drive real code.

use std::sync::{Arc, OnceLock};

pub mod client;
pub mod mock;
pub mod reqwest_client;
pub mod sse;
pub mod websocket;

pub use client::{
    BoxHttpFuture, HttpByteStream, HttpClient, HttpError, HttpMethod, HttpRequest, HttpResponse,
};
pub use mock::{
    MockBody, MockHttpClient, MockResponse, MockWebSocketPeer, MockWebSocketTransport,
    json_response, sse_response,
};
pub use reqwest_client::ReqwestHttpClient;
pub use sse::{ServerSentEvent, SseDecoder, SseStream, collect_sse};
pub use websocket::{
    TungsteniteWebSocket, WebSocketConnection, WebSocketError, WebSocketMessage, WebSocketOutbound,
    WebSocketRequest, WebSocketTransport,
};

static DEFAULT_HTTP_CLIENT: OnceLock<Arc<dyn HttpClient>> = OnceLock::new();
static DEFAULT_WEBSOCKET_TRANSPORT: OnceLock<Arc<dyn WebSocketTransport>> = OnceLock::new();

/// The process-default client, the port of ambient `globalThis.fetch`: the
/// reqwest 0.12 + rustls [`ReqwestHttpClient`], shared so connection pooling
/// matches the one global object upstream builds on.
///
/// # Panics
/// Panics when the client's TLS or proxy configuration cannot initialize —
/// every provider request rides this client, so there is no usable
/// degradation.
#[must_use]
pub fn default_http_client() -> Arc<dyn HttpClient> {
    #[expect(
        clippy::expect_used,
        reason = "the process-default client cannot recover from a failed build; first-use failure with the builder's message beats silent degradation"
    )]
    DEFAULT_HTTP_CLIENT
        .get_or_init(|| {
            Arc::new(
                ReqwestHttpClient::new().expect("the process-default reqwest client must build"),
            )
        })
        .clone()
}

/// The process-default WebSocket transport, the port of ambient
/// `globalThis.WebSocket`: the tokio-tungstenite [`TungsteniteWebSocket`].
#[must_use]
pub fn default_websocket_transport() -> Arc<dyn WebSocketTransport> {
    DEFAULT_WEBSOCKET_TRANSPORT
        .get_or_init(|| Arc::new(TungsteniteWebSocket::new()))
        .clone()
}

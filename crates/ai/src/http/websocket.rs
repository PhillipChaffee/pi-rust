//! The WebSocket transport seam, and the tokio-tungstenite implementation
//! the stack decision pins for it.
//!
//! The seam is upstream's `globalThis.WebSocket` injectability for the Codex
//! wire API (`packages/ai/src/api/openai-codex-responses.ts:992`).
//!
//! Porting restatements: Node's WebSocket close model (message events until a
//! close event carrying code and reason) ports to the [`WebSocketError::Closed`]
//! variant a read fails with once the peer closes; and proxy environment
//! handling stays out of the transport — upstream resolves proxy URLs for
//! WebSocket connects only on Bun (`node-http-proxy.ts`), and Node's own
//! `WebSocket` connects directly, so Rust pi does too.

use futures_util::StreamExt;
use tokio_tungstenite::tungstenite;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_util::sync::CancellationToken;

use crate::http::client::BoxHttpFuture;
use crate::types::BoxedFuture;

/// The WebSocket connect timeout when a request carries none, upstream's
/// `DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS` (`openai-codex-responses.ts:50`).
pub const DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS: u64 = 15_000;

/// A WebSocket connect the transport performs, the argument pair of
/// `new WebSocket(url, { headers })`.
#[derive(Clone, Debug)]
pub struct WebSocketRequest {
    /// The WebSocket URL (`ws:` or `wss:`).
    pub url: String,
    /// Handshake headers. Transport-controlled headers — `Host`, `Upgrade`,
    /// `Connection`, and the `Sec-WebSocket-*` fields — must not appear here.
    pub headers: Vec<(String, String)>,
    /// Connect-handshake timeout in milliseconds; covers the open handshake
    /// only, stream idleness afterwards rides the request timeout upstream.
    /// `None` uses [`DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS`].
    pub connect_timeout_ms: Option<u64>,
    /// Cancellation for the handshake, the reads, and the writes.
    pub signal: CancellationToken,
}

/// One WebSocket message, the payload of a `message` event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WebSocketMessage {
    /// A text message.
    Text(String),
    /// A binary message.
    Binary(bytes::Bytes),
}

/// A message a connection sent, the shape tests read off mock peers and
/// loopback servers: [`WebSocketOutbound::Message`] for sends,
/// [`WebSocketOutbound::Close`] for `close(code, reason)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WebSocketOutbound {
    /// A text or binary send.
    Message(WebSocketMessage),
    /// A close with the wire's status code and reason.
    Close {
        /// The close status code.
        code: u16,
        /// The close reason text.
        reason: String,
    },
}

/// Why the transport failed, mirroring the error shapes the Codex wire API
/// retries on: connect failures, an aborted handshake, a timed-out
/// handshake, and the close event's code/reason pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebSocketError {
    /// The request's cancellation token aborted before or during the
    /// connection, upstream's abort listener on the handshake.
    Aborted,
    /// The connect handshake did not complete in time, upstream's
    /// `WebSocket connect timeout after {ms}ms` rejection.
    Timeout(u64),
    /// The peer closed, upstream's `close` event's code and reason;
    /// `was_clean` mirrors `WebSocketCloseError.wasClean`.
    Closed {
        /// The close status code.
        code: u16,
        /// The close reason text.
        reason: String,
        /// Whether the close handshake completed.
        was_clean: bool,
    },
    /// Transport failure — DNS, TCP, TLS, or a failed handshake; the message
    /// mirrors what the transport reported, the shape the retry classifiers
    /// match on.
    Transport(String),
}

impl WebSocketError {
    /// The close of a connected peer that reported its own code and reason.
    #[must_use]
    pub fn closed(code: u16, reason: impl Into<String>, was_clean: bool) -> Self {
        Self::Closed {
            code,
            reason: reason.into(),
            was_clean,
        }
    }
}

impl std::fmt::Display for WebSocketError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Aborted => f.write_str(crate::utils::abort::AbortError::MESSAGE),
            Self::Timeout(ms) => write!(f, "WebSocket connect timeout after {ms}ms"),
            Self::Closed { code, reason, .. } => {
                write!(f, "websocket closed ({code}): {reason}")
            }
            Self::Transport(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for WebSocketError {}

/// A connected WebSocket, the seam's port of `WebSocketLike`
/// (`openai-codex-responses.ts`).
pub trait WebSocketConnection: Send + std::fmt::Debug {
    /// Send one message.
    ///
    /// # Errors
    /// Returns [`WebSocketError::Closed`] once either side has closed and
    /// [`WebSocketError::Aborted`] when the request's token aborts.
    fn send(&mut self, message: WebSocketMessage) -> BoxedFuture<'_, Result<(), WebSocketError>>;

    /// Read the next message from the peer.
    ///
    /// # Errors
    /// Returns [`WebSocketError::Closed`] once the peer closes — the close
    /// handshake's reply is sent best-effort — and
    /// [`WebSocketError::Aborted`] or [`WebSocketError::Transport`] when the
    /// connection fails another way.
    fn next_message(&mut self) -> BoxedFuture<'_, Result<WebSocketMessage, WebSocketError>>;

    /// Close with a status code and reason, sending the close frame.
    ///
    /// # Errors
    /// Returns a transport error when the close frame cannot be sent; the
    /// peer's closing answer is observed through [`Self::next_message`].
    fn close(&mut self, code: u16, reason: String) -> BoxedFuture<'_, Result<(), WebSocketError>>;
}

/// The transport that opens WebSocket connections, upstream's injectable
/// `WebSocket` constructor; tests inject mocks over the same seam.
pub trait WebSocketTransport: std::fmt::Debug + Send + Sync {
    /// Open one connection.
    ///
    /// # Errors
    /// Returns [`WebSocketError::Aborted`], [`WebSocketError::Timeout`], or
    /// [`WebSocketError::Transport`] when the handshake fails.
    fn connect(
        &self,
        request: WebSocketRequest,
    ) -> BoxHttpFuture<Result<Box<dyn WebSocketConnection>, WebSocketError>>;
}

/// The tokio-tungstenite-backed [`WebSocketTransport`], the port of Node
/// 22+'s ambient `WebSocket` constructor.
#[derive(Debug, Clone, Default)]
pub struct TungsteniteWebSocket;

impl TungsteniteWebSocket {
    /// Build the transport; TLS initializes per connection, so construction
    /// cannot fail.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Convert wire header pairs into the handshake request's header map,
/// preserving duplicates and order.
///
/// # Errors
/// Returns [`WebSocketError::Transport`] when a pair is not a valid HTTP
/// header.
fn handshake_headers(
    request: &mut tungstenite::handshake::client::Request,
    pairs: &[(String, String)],
) -> Result<(), WebSocketError> {
    use tungstenite::http::{HeaderName, HeaderValue};
    for (name, value) in pairs {
        let name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
            WebSocketError::Transport(format!("invalid header name {name:?}: {error}"))
        })?;
        let value = HeaderValue::from_str(value).map_err(|error| {
            WebSocketError::Transport(format!("invalid header value {value:?}: {error}"))
        })?;
        request.headers_mut().append(name, value);
    }
    Ok(())
}

/// Map a tungstenite failure to the seam's error vocabulary.
fn transport_error(error: &tungstenite::Error) -> WebSocketError {
    match error {
        tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed => {
            WebSocketError::closed(1005, "", true)
        }
        tungstenite::Error::Io(io) => {
            WebSocketError::Transport(format!("websocket transport error: {io}"))
        }
        tungstenite::Error::Tls(tls) => {
            WebSocketError::Transport(format!("websocket tls error: {tls}"))
        }
        tungstenite::Error::Url(url) => {
            WebSocketError::Transport(format!("invalid websocket URL: {url}"))
        }
        tungstenite::Error::Http(response) => WebSocketError::Transport(format!(
            "websocket handshake failed with status {}",
            response.status().as_u16()
        )),
        tungstenite::Error::HttpFormat(error) => {
            WebSocketError::Transport(format!("websocket handshake malformed: {error}"))
        }
        other => WebSocketError::Transport(format!("websocket error: {other}")),
    }
}

/// The tokio-tungstenite-backed [`WebSocketConnection`].
#[derive(Debug)]
struct TungsteniteConnection {
    socket: tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    signal: CancellationToken,
}

impl WebSocketConnection for TungsteniteConnection {
    fn send(&mut self, message: WebSocketMessage) -> BoxedFuture<'_, Result<(), WebSocketError>> {
        Box::pin(async move {
            let wire_message = match message {
                WebSocketMessage::Text(text) => tungstenite::Message::text(text),
                WebSocketMessage::Binary(data) => tungstenite::Message::binary(data),
            };
            futures_util::SinkExt::send(&mut self.socket, wire_message)
                .await
                .map_err(|error| transport_error(&error))
        })
    }

    fn next_message(&mut self) -> BoxedFuture<'_, Result<WebSocketMessage, WebSocketError>> {
        Box::pin(async move {
            if self.signal.is_cancelled() {
                return Err(WebSocketError::Aborted);
            }
            tokio::select! {
                () = self.signal.cancelled() => Err(WebSocketError::Aborted),
                message = self.socket.next() => match message {
                    Some(Ok(tungstenite::Message::Text(text))) => {
                        Ok(WebSocketMessage::Text(text.to_string()))
                    }
                    Some(Ok(tungstenite::Message::Binary(data))) => {
                        Ok(WebSocketMessage::Binary(bytes::Bytes::from(data.to_vec())))
                    }
                    // Ping/pong frames are answered by tungstenite and carry
                    // no payload the wire API reads.
                    Some(Ok(
                        tungstenite::Message::Ping(_)
                            | tungstenite::Message::Pong(_)
                            | tungstenite::Message::Frame(_),
                    )) => Box::pin(self.next_message()).await,
                    Some(Ok(tungstenite::Message::Close(frame))) => {
                        // The close event upstream carries code/reason on;
                        // answer the handshake best-effort so the peer sees a
                        // clean close even when Rust pi is done reading.
                        let _ = futures_util::SinkExt::send(
                            &mut self.socket,
                            tungstenite::Message::Close(None),
                        )
                        .await;
                        match frame {
                            Some(close) => Err(WebSocketError::closed(
                                u16::from(close.code),
                                close.reason.to_string(),
                                true,
                            )),
                            None => Err(WebSocketError::closed(1005, "", true)),
                        }
                    }
                    Some(Err(error)) => Err(transport_error(&error)),
                    None => Err(WebSocketError::closed(1005, "", true)),
                },
            }
        })
    }

    fn close(&mut self, code: u16, reason: String) -> BoxedFuture<'_, Result<(), WebSocketError>> {
        Box::pin(async move {
            let frame = tungstenite::protocol::CloseFrame {
                code: tungstenite::protocol::frame::coding::CloseCode::from(code),
                reason: reason.as_str().into(),
            };
            futures_util::SinkExt::send(&mut self.socket, tungstenite::Message::Close(Some(frame)))
                .await
                .map_err(|error| transport_error(&error))
        })
    }
}

impl WebSocketTransport for TungsteniteWebSocket {
    fn connect(
        &self,
        request: WebSocketRequest,
    ) -> BoxHttpFuture<Result<Box<dyn WebSocketConnection>, WebSocketError>> {
        Box::pin(async move {
            if request.signal.is_cancelled() {
                return Err(WebSocketError::Aborted);
            }

            let mut handshake = request
                .url
                .as_str()
                .into_client_request()
                .map_err(|error| {
                    WebSocketError::Transport(format!("invalid websocket URL: {error}"))
                })?;
            handshake_headers(&mut handshake, &request.headers)?;

            let connect_timeout_ms = request
                .connect_timeout_ms
                .unwrap_or(DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS);
            let connect = tokio_tungstenite::connect_async(handshake);
            let timeout = tokio::time::sleep(std::time::Duration::from_millis(connect_timeout_ms));

            let (socket, _response) = tokio::select! {
                () = request.signal.cancelled() => return Err(WebSocketError::Aborted),
                () = timeout => return Err(WebSocketError::Timeout(connect_timeout_ms)),
                result = connect => result.map_err(|error| transport_error(&error))?,
            };

            let connection: Box<dyn WebSocketConnection> = Box::new(TungsteniteConnection {
                socket,
                signal: request.signal,
            });
            Ok(connection)
        })
    }
}
#[cfg(test)]
mod error_mapping_tests {
    #![expect(
        clippy::expect_used,
        reason = "the tests pin outcomes; an unexpected result panics the test by design"
    )]

    use super::*;

    fn map(error: &tungstenite::Error) -> WebSocketError {
        transport_error(error)
    }

    #[test]
    fn closed_socket_states_map_to_the_clean_close_error() {
        for error in [
            tungstenite::Error::ConnectionClosed,
            tungstenite::Error::AlreadyClosed,
        ] {
            assert_eq!(
                map(&error),
                WebSocketError::Closed {
                    code: 1005,
                    reason: String::new(),
                    was_clean: true,
                }
            );
        }
    }

    #[test]
    fn transport_failures_carry_the_wording_the_retry_classifiers_match() {
        let io = std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused");
        let mapped = map(&tungstenite::Error::Io(io));
        assert!(
            matches!(mapped, WebSocketError::Transport(message) if message.contains("refused"))
        );

        let mapped = map(&tungstenite::Error::Url(
            tungstenite::error::UrlError::UnsupportedUrlScheme,
        ));
        assert!(matches!(
            mapped,
            WebSocketError::Transport(message) if message.contains("invalid websocket URL")
        ));

        let response = tungstenite::http::Response::builder()
            .status(401)
            .body(Some(Vec::new()))
            .expect("response builds");
        let mapped = map(&tungstenite::Error::Http(Box::new(response)));
        assert!(matches!(mapped, WebSocketError::Transport(message) if message.contains("401")));

        let mapped = map(&tungstenite::Error::Protocol(
            tungstenite::error::ProtocolError::SendAfterClosing,
        ));
        assert!(
            matches!(mapped, WebSocketError::Transport(message) if message.contains("protocol"))
        );

        // The catch-all carries tungstenite's own wording behind the seam's
        // prefix, the shape the retry classifiers substring-match on.
        let mapped = map(&tungstenite::Error::Capacity(
            tungstenite::error::CapacityError::MessageTooLong {
                size: 5,
                max_size: 5,
            },
        ));
        assert!(
            matches!(mapped, WebSocketError::Transport(message) if message.contains("websocket error"))
        );
    }

    #[test]
    fn handshake_headers_reject_malformed_pairs_with_transport_errors() {
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;

        let mut request = "ws://upstream.test/ws"
            .into_client_request()
            .expect("well-formed url");
        let error = handshake_headers(
            &mut request,
            &[("bad header".to_owned(), "value".to_owned())],
        )
        .expect_err("bad header name rejects");
        assert!(matches!(
            error,
            WebSocketError::Transport(message) if message.contains("invalid header name")
        ));

        let error = handshake_headers(
            &mut request,
            &[("x-bad".to_owned(), "bad\nvalue".to_owned())],
        )
        .expect_err("bad header value rejects");
        assert!(matches!(
            error,
            WebSocketError::Transport(message) if message.contains("invalid header value")
        ));
    }

    #[test]
    fn close_errors_display_the_shapes_the_codex_retries_match() {
        assert_eq!(
            WebSocketError::closed(1006, "abnormal closure", false).to_string(),
            "websocket closed (1006): abnormal closure"
        );
        assert_eq!(
            WebSocketError::Timeout(15_000).to_string(),
            "WebSocket connect timeout after 15000ms"
        );
        assert_eq!(
            WebSocketError::Aborted.to_string(),
            crate::utils::abort::AbortError::MESSAGE
        );
    }
}

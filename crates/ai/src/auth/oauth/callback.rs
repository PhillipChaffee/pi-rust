//! The loopback HTTP callback server the browser-based OAuth flows listen on.
//!
//! The shared core of upstream's per-flow `node:http` servers
//! (`anthropic.ts`, `openai-codex.ts`, `openrouter.ts`, `radius.ts`).
//!
//! Upstream builds a fresh `node:http` server per flow with the same shape:
//! route on the callback path, run the flow's own checks, send the success
//! or error page, and settle a waiter the login races against. This module
//! carries the HTTP mechanics — accept, read one request, respond, close —
//! and each flow keeps its routing and settle semantics.
//!
//! Hand-rolled HTTP/1.1 responder, not a server framework: the callbacks
//! receive one `GET` from a browser per connection, which the server reads
//! to the header terminator and answers with a fixed Content-Length body —
//! the entire protocol surface `createServer` serves here.

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use super::auth_error;
use crate::auth::types::AuthError;

/// One parsed callback request, the slice of the HTTP request the flows
/// route on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallbackRequest {
    /// The request path, no query string.
    pub path: String,
    /// The query parameters in wire order.
    pub query: Vec<(String, String)>,
}

impl CallbackRequest {
    /// The first value of a query parameter.
    #[must_use]
    pub fn query_value(&self, name: &str) -> Option<&str> {
        self.query
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

/// The page a handler sends back for one request.
#[derive(Clone, Debug)]
pub struct CallbackResponse {
    /// The HTTP status code.
    pub status: u16,
    /// The HTML body, the success or error page.
    pub html: String,
}

/// A handler the callback server consults per request. The handler settles
/// its flow's wait state itself; the server only moves bytes.
pub type CallbackHandler = Arc<
    dyn Fn(CallbackRequest) -> crate::types::BoxedFuture<'static, CallbackResponse> + Send + Sync,
>;

/// A running loopback server: route requests through the handler, stop
/// accepting on [`OAuthCallbackServer::close`].
pub struct OAuthCallbackServer {
    local_addr: SocketAddr,
    shutdown: tokio_util::sync::CancellationToken,
}

impl std::fmt::Debug for OAuthCallbackServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthCallbackServer")
            .finish_non_exhaustive()
    }
}

impl OAuthCallbackServer {
    /// Start listening on `host:port`. Port `0` binds an ephemeral port.
    ///
    /// # Errors
    /// Rejects when the bind fails — upstream's listen error, surfaced
    /// before login hands a URL to the user.
    pub async fn bind(host: &str, port: u16, handler: CallbackHandler) -> Result<Self, AuthError> {
        let listener = TcpListener::bind((host, port)).await.map_err(|error| {
            auth_error(format!(
                "could not bind the OAuth callback server on {host}:{port}: {error}"
            ))
        })?;
        let local_addr = listener.local_addr().map_err(|error| {
            auth_error(format!(
                "could not determine the OAuth callback port: {error}"
            ))
        })?;
        let shutdown = tokio_util::sync::CancellationToken::new();
        tokio::spawn(accept_loop(listener, handler, shutdown.child_token()));
        Ok(Self {
            local_addr,
            shutdown,
        })
    }

    /// The bound address, the port the authorize URL points at.
    ///
    /// # Errors
    /// Rejects when the local address is unavailable — upstream's "Could
    /// not determine the callback port".
    pub const fn local_addr(&self) -> Result<SocketAddr, AuthError> {
        Ok(self.local_addr)
    }

    /// Stop accepting, upstream's `server.close()`; an in-flight request
    /// already being answered finishes first.
    pub fn close(&self) {
        self.shutdown.cancel();
    }
}

impl Drop for OAuthCallbackServer {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

async fn accept_loop(
    listener: TcpListener,
    handler: CallbackHandler,
    shutdown: tokio_util::sync::CancellationToken,
) {
    loop {
        tokio::select! {
            () = shutdown.cancelled() => break,
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    let handler = handler.clone();
                    let connection_shutdown = shutdown.child_token();
                    tokio::spawn(serve_connection(stream, handler, connection_shutdown));
                }
                Err(_) => break,
            },
        }
    }
}

/// Read one request to its header terminator, route it, and answer with a
/// `Connection: close` response — one request per connection.
async fn serve_connection(
    mut stream: TcpStream,
    handler: CallbackHandler,
    shutdown: tokio_util::sync::CancellationToken,
) {
    let outcome = tokio::select! {
        () = shutdown.cancelled() => return,
        read = read_request(&mut stream) => match read {
            Ok(Some(request)) => Ok(handler(request).await),
            Ok(None) => return,
            Err(_) => Err(CallbackResponse {
                status: 500,
                html: "Internal error".to_owned(),
            }),
        },
    };
    let (status, html) = match outcome {
        Ok(response) | Err(response) => (response.status, response.html),
    };
    let _ = write_response(&mut stream, status, &html).await;
    let _ = stream.shutdown().await;
}

/// Read one HTTP/1.1 request line and its headers, parsing only what the
/// flows route on: the path and query. `None` on EOF before any bytes.
async fn read_request(stream: &mut TcpStream) -> Result<Option<CallbackRequest>, std::io::Error> {
    let mut buffer = Vec::with_capacity(1024);
    let mut chunk = [0_u8; 1024];
    loop {
        let read = match stream.read(&mut chunk).await? {
            0 => {
                return Ok(if buffer.is_empty() {
                    None
                } else {
                    Some(parse_request(&buffer)?)
                });
            }
            read => read,
        };
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
            return Some(parse_request(&buffer)).transpose();
        }
        if buffer.len() > 16 * 1024 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "oversized request",
            ));
        }
    }
}

/// Parse the request target of one buffered request, upstream's
/// `new URL(req.url || "", "http://localhost")`.
fn parse_request(buffer: &[u8]) -> Result<CallbackRequest, std::io::Error> {
    let text = std::str::from_utf8(buffer)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "non-UTF-8 request"))?;
    let request_line = text
        .split("\r\n")
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "empty request"))?;
    let target = request_line.split(' ').nth(1).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "malformed request line")
    })?;
    let url = url::Url::parse(&format!("http://localhost{target}")).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "malformed request target")
    })?;
    Ok(CallbackRequest {
        path: url.path().to_owned(),
        query: url
            .query_pairs()
            .map(|(key, value)| (key.into_owned(), value.into_owned()))
            .collect(),
    })
}

/// Write one HTTP/1.1 response with the page as the body.
async fn write_response(stream: &mut TcpStream, status: u16, html: &str) -> std::io::Result<()> {
    let reason = match status {
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        _ => "OK",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: text/html; charset=utf-8\r\ncontent-length: {}\r\ncache-control: no-store\r\nconnection: close\r\n\r\n{html}",
        html.len()
    );
    stream.write_all(response.as_bytes()).await
}

/// The wait one login races its callback against, upstream's
/// `waitForCode`/`waitForCredential` promise.
///
/// One settle, later settles ignored. The settle carries [`None`] to hand
/// the login over to manual entry, upstream's `cancelWait` semantics.
pub struct WaitCell<T> {
    slot: Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Sender<Option<T>>>>>,
}

impl<T> Default for WaitCell<T> {
    fn default() -> Self {
        Self {
            slot: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }
}

impl<T> Clone for WaitCell<T> {
    fn clone(&self) -> Self {
        Self {
            slot: self.slot.clone(),
        }
    }
}

impl<T> std::fmt::Debug for WaitCell<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WaitCell(..)")
    }
}

impl<T> WaitCell<T> {
    /// Prepare the waiter; the receiver resolves once a settle lands.
    #[must_use]
    pub fn new() -> (Self, tokio::sync::oneshot::Receiver<Option<T>>) {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        (
            Self {
                slot: Arc::new(tokio::sync::Mutex::new(Some(sender))),
            },
            receiver,
        )
    }

    /// Settle the wait with a value; later settles are ignored. `None`
    /// cancels the wait — upstream's `cancelWait` handing the login over to
    /// manual code entry.
    pub async fn settle(&self, value: Option<T>) {
        let sender = self.slot.lock().await.take();
        if let Some(sender) = sender {
            let _ = sender.send(value);
        }
    }
}

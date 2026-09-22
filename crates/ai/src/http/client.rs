//! The HTTP seam core: the request/response shapes every wire-API adapter
//! builds on, upstream's `FetchFunction` injectability
//! (`packages/ai/src/types.ts:115`) expressed as a trait.

use std::pin::Pin;

use bytes::Bytes;
use futures_core::Stream;
use tokio_util::sync::CancellationToken;

use crate::types::BoxedFuture;

/// The future type the seam's owned-input methods return: the crate's
/// [`BoxedFuture`] pinned to `'static`.
pub type BoxHttpFuture<T> = BoxedFuture<'static, T>;

/// An HTTP method, upstream's `init.method` string of `RequestInit`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HttpMethod {
    /// GET.
    Get,
    /// POST.
    Post,
    /// PUT.
    Put,
    /// DELETE.
    Delete,
    /// PATCH.
    Patch,
    /// HEAD.
    Head,
    /// Any other method, upstream passes method strings through verbatim.
    Custom(String),
}

impl HttpMethod {
    /// The method as the wire spells it.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Patch => "PATCH",
            Self::Head => "HEAD",
            Self::Custom(method) => method,
        }
    }
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A provider request the seam sends, the argument pair of
/// `globalThis.fetch(input, init)` flattened into one struct.
///
/// Headers keep duplicates and wire order; the response mirrors the same
/// shape, so adapters that count set-cookie entries or read repeated headers
/// see what the wire sent.
#[derive(Clone, Debug)]
pub struct HttpRequest {
    /// The request method.
    pub method: HttpMethod,
    /// The absolute request URL.
    pub url: String,
    /// The request headers, duplicates and order preserved.
    pub headers: Vec<(String, String)>,
    /// The request body, when the method carries one.
    pub body: Option<Bytes>,
    /// Total request timeout in milliseconds, upstream's `timeoutMs` as the
    /// SDKs apply it: from request start until the response body finishes.
    /// `None` means unbounded.
    pub timeout_ms: Option<u64>,
    /// Cancellation for the request and its streamed body, upstream's
    /// `init.signal`.
    pub signal: CancellationToken,
}

/// A streamed byte body, the seam's port of `Response.body`
/// (`ReadableStream<Uint8Array>`).
///
/// Chunk boundaries carry no framing guarantees: chunks split wherever the
/// transport happened to cut the bytes, so every consumer must decode across
/// boundaries (the SSE decoder does). Reads race the request's cancellation
/// token, so aborting the request ends the stream with [`HttpError::Aborted`].
pub struct HttpByteStream(Pin<Box<dyn Stream<Item = Result<Bytes, HttpError>> + Send>>);

impl HttpByteStream {
    /// Wrap a stream of byte chunks.
    #[must_use]
    pub fn new(stream: impl Stream<Item = Result<Bytes, HttpError>> + Send + 'static) -> Self {
        Self(Box::pin(stream))
    }

    /// Build a stream that yields the given items in order — canned responses
    /// for tests and mocks. An `Err` item terminates the stream.
    #[must_use]
    pub fn from_chunks(chunks: Vec<Result<Bytes, HttpError>>) -> Self {
        Self(Box::pin(futures_util::stream::iter(chunks)))
    }

    /// Read the next chunk, `None` at end of stream.
    ///
    /// # Errors
    /// Returns the stream's own error: an aborted request surfaces as
    /// [`HttpError::Aborted`], a dropped connection as [`HttpError::Transport`].
    pub async fn next_chunk(&mut self) -> Result<Option<Bytes>, HttpError> {
        match futures_util::StreamExt::next(&mut self.0).await {
            Some(Ok(chunk)) => Ok(Some(chunk)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        }
    }
}

impl Stream for HttpByteStream {
    type Item = Result<Bytes, HttpError>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.0.as_mut().poll_next(cx)
    }
}

impl std::fmt::Debug for HttpByteStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HttpByteStream(..)")
    }
}

/// A provider HTTP response with a streaming body, the seam's port of
/// `Response`.
pub struct HttpResponse {
    /// The HTTP status code; the seam resolves non-2xx responses normally,
    /// like `fetch` does — adapters own status handling.
    pub status: u16,
    /// The response headers, duplicates and wire order preserved.
    pub headers: Vec<(String, String)>,
    /// The response body stream.
    pub body: HttpByteStream,
}

impl std::fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .field("body", &self.body)
            .finish()
    }
}

/// Why a seam request or body read failed.
///
/// HTTP error statuses are not errors here — `fetch` resolves on 4xx/5xx and
/// the adapters translate them; these variants are what `fetch` itself
/// rejects with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpError {
    /// The request's cancellation token aborted before or during the request,
    /// upstream's `AbortError`.
    Aborted,
    /// The total request timeout expired, upstream's SDK timeout rejection.
    Timeout,
    /// Connection, DNS, TLS, or other transport failure; the message mirrors
    /// what the underlying transport reported, the shape the retry
    /// classifiers match on.
    Transport(String),
    /// The request URL does not parse.
    InvalidUrl(String),
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Aborted => f.write_str(crate::utils::abort::AbortError::MESSAGE),
            Self::Timeout => f.write_str("request timed out"),
            Self::Transport(message) => f.write_str(message),
            Self::InvalidUrl(url) => write!(f, "invalid URL: {url}"),
        }
    }
}

impl std::error::Error for HttpError {}

/// The seam every wire-API adapter sends provider requests through,
/// upstream's injectable `fetch` (`packages/ai/src/types.ts:115`).
///
/// The process default — [`crate::http::default_http_client`] — is the
/// reqwest 0.12 + rustls client the stack decision pins; tests inject mocks
/// over the same seam where upstream stubbed `globalThis.fetch`.
pub trait HttpClient: std::fmt::Debug + Send + Sync {
    /// Send one request, resolving with the response once its headers arrive;
    /// the body streams afterwards. Cancellation and timeouts surface as
    /// [`HttpError`] variants, never as `Ok` error statuses.
    fn execute(&self, request: HttpRequest) -> BoxHttpFuture<Result<HttpResponse, HttpError>>;
}

/// Read a response (or error) body to text, the helper the adapters use when
/// an error body or a non-streaming payload must be inspected before use.
/// The lossy UTF-8 decode mirrors the JS adapters reading `response.text()`.
pub async fn read_body_text(
    mut body: HttpByteStream,
) -> Result<String, HttpError> {
    let mut text = String::new();
    while let Some(chunk) = body.next_chunk().await? {
        text.push_str(&String::from_utf8_lossy(&chunk));
    }
    Ok(text)
}

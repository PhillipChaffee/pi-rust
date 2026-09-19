//! The process-default [`HttpClient`]: reqwest 0.12 + rustls, the stack
//! decision's wire client behind the seam.

use std::task::Poll;

use bytes::Bytes;
use futures_core::Stream;
use tokio_util::sync::CancellationToken;

use crate::http::client::{
    BoxHttpFuture, HttpByteStream, HttpClient, HttpError, HttpMethod, HttpRequest, HttpResponse,
};
use crate::utils::abort::{RaceError, race_with_abort_signal};

/// The reqwest-backed [`HttpClient`], the port of `globalThis.fetch` as the
/// ambient default (`packages/ai/src/types.ts:115`).
#[derive(Debug, Clone)]
pub struct ReqwestHttpClient {
    client: reqwest::Client,
}

impl ReqwestHttpClient {
    /// Build the client with rustls, webpki trust roots, and env proxy
    /// support.
    ///
    /// # Errors
    /// Returns the builder's error when the TLS backend or the proxy
    /// configuration the environment describes cannot initialize.
    pub fn new() -> Result<Self, reqwest::Error> {
        Ok(Self {
            client: reqwest::Client::builder().build()?,
        })
    }
}

impl Default for ReqwestHttpClient {
    fn default() -> Self {
        // The process default cannot hand a build failure to a caller — every
        // provider request rides this client — so a broken TLS or proxy
        // environment must fail loudly here rather than degrade silently.
        #[expect(
            clippy::expect_used,
            reason = "the process-default client cannot recover from a failed build; first-use failure with the builder's message beats silent degradation"
        )]
        Self::new().expect("the process-default reqwest client must build")
    }
}

/// Map a reqwest failure to the seam's error vocabulary.
fn transport_error(error: &reqwest::Error) -> HttpError {
    if error.is_timeout() {
        HttpError::Timeout
    } else {
        HttpError::Transport(error.to_string())
    }
}

/// Race a body read against the request's cancellation token, upstream's
/// `signal` aborting body reads.
fn raced_body(
    body: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    signal: CancellationToken,
) -> HttpByteStream {
    let mut body = Box::pin(body);
    let mut cancelled = Box::pin(signal.cancelled_owned());
    HttpByteStream::new(futures_util::stream::poll_fn(move |cx| {
        match body.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => Poll::Ready(Some(Ok(chunk))),
            Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(transport_error(&error)))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => {
                // Register the owned cancellation future so an abort during an
                // idle read wakes this poll instead of waiting for the
                // transport to move.
                if cancelled.as_mut().poll(cx) == Poll::Ready(()) {
                    Poll::Ready(Some(Err(HttpError::Aborted)))
                } else {
                    Poll::Pending
                }
            }
        }
    }))
}

/// Convert the wire header pairs to reqwest's map, preserving duplicates and
/// order.
///
/// # Errors
/// Returns [`HttpError::Transport`] when a pair is not a valid HTTP header.
fn header_map(pairs: &[(String, String)]) -> Result<reqwest::header::HeaderMap, HttpError> {
    let mut headers = reqwest::header::HeaderMap::new();
    for (name, value) in pairs {
        let name = reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
            HttpError::Transport(format!("invalid header name {name:?}: {error}"))
        })?;
        let value = reqwest::header::HeaderValue::from_str(value).map_err(|error| {
            HttpError::Transport(format!("invalid header value {value:?}: {error}"))
        })?;
        headers.append(name, value);
    }
    Ok(headers)
}

impl HttpClient for ReqwestHttpClient {
    fn execute(&self, request: HttpRequest) -> BoxHttpFuture<Result<HttpResponse, HttpError>> {
        let client = self.client.clone();
        Box::pin(async move {
            if request.signal.is_cancelled() {
                return Err(HttpError::Aborted);
            }

            let method = match &request.method {
                HttpMethod::Get => reqwest::Method::GET,
                HttpMethod::Post => reqwest::Method::POST,
                HttpMethod::Put => reqwest::Method::PUT,
                HttpMethod::Delete => reqwest::Method::DELETE,
                HttpMethod::Patch => reqwest::Method::PATCH,
                HttpMethod::Head => reqwest::Method::HEAD,
                HttpMethod::Custom(method) => reqwest::Method::from_bytes(method.as_bytes())
                    .map_err(|error| {
                        HttpError::Transport(format!("invalid method {method:?}: {error}"))
                    })?,
            };

            let mut request_builder = client
                .request(method, &request.url)
                .headers(header_map(&request.headers)?);
            if let Some(timeout_ms) = request.timeout_ms {
                request_builder =
                    request_builder.timeout(std::time::Duration::from_millis(timeout_ms));
            }
            if let Some(body) = request.body {
                request_builder = request_builder.body(body.to_vec());
            }

            let response = race_with_abort_signal(request_builder.send(), &request.signal)
                .await
                .map_err(|race| match race {
                    RaceError::Aborted(_) => HttpError::Aborted,
                    RaceError::Operation(error) => transport_error(&error),
                })?;

            let status = response.status().as_u16();
            let headers = response
                .headers()
                .iter()
                .map(|(name, value)| {
                    (
                        name.as_str().to_owned(),
                        String::from_utf8_lossy(value.as_bytes()).into_owned(),
                    )
                })
                .collect::<Vec<(String, String)>>();
            let body = raced_body(response.bytes_stream(), request.signal);

            Ok(HttpResponse {
                status,
                headers,
                body,
            })
        })
    }
}

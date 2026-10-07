//! Management HTTP with a bounded immediate retry, upstream's
//! `src/utils/management-http.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! This is intentionally a transport-level helper for idempotent management
//! requests (version checks, catalogs, and downloads). It must not be used
//! for agent/model operations: those can fail after the HTTP request starts
//! and are retried by their semantic caller instead.
//!
//! Porting restatements: ambient `fetch` rides the crate's [`HttpClient`]
//! seam (the caller passes the process default or an injected mock), and the
//! `AbortSignal` family ports to [`CancellationToken`]s raced through
//! `tokio::select!` — the parent signal and the total budget are terminal,
//! the per-attempt budget aborts only the current attempt so a hung
//! connection can be retried.

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use pi_ai::http::client::{HttpMethod, HttpRequest};
use pi_ai::http::{HttpClient, HttpError, HttpResponse};

/// The retry options, upstream's `FetchRetryOptions`.
#[derive(Clone, Debug, Default)]
pub struct FetchRetryOptions {
    /// Number of additional attempts after the initial request. Default: 2.
    pub max_retries: Option<u32>,
    /// Retry transient HTTP responses as well as transport failures.
    /// Default: true.
    pub retry_on_status: Option<bool>,
    /// Overall time budget shared by all attempts.
    pub timeout_ms: Option<u64>,
    /// Per-attempt timeout. A new budget is created for every attempt.
    pub attempt_timeout_ms: Option<u64>,
}

/// The statuses a retry follows, upstream's `RETRYABLE_STATUS_CODES`.
const RETRYABLE_STATUS_CODES: [u16; 7] = [408, 425, 429, 500, 502, 503, 504];

/// Fetch a management HTTP resource with a bounded immediate retry,
/// upstream's `fetchWithRetry`.
///
/// Caller cancellation and `timeout_ms` are terminal. `attempt_timeout_ms`
/// aborts only the current attempt so a hung connection can be retried.
///
/// # Errors
/// The last attempt's transport failure, the parent cancellation, or the
/// total budget's expiry.
pub async fn fetch_with_retry(
    client: &Arc<dyn HttpClient>,
    url: &str,
    headers: Vec<(String, String)>,
    signal: CancellationToken,
    options: FetchRetryOptions,
) -> Result<HttpResponse, HttpError> {
    let max_retries = options.max_retries.unwrap_or(2);
    let retry_on_status = options.retry_on_status.unwrap_or(true);
    let timeout_deadline = options
        .timeout_ms
        .filter(|timeout_ms| *timeout_ms > 0)
        .map(|timeout_ms| tokio::time::Instant::now() + Duration::from_millis(timeout_ms));
    let attempt_timeout_ms = options
        .attempt_timeout_ms
        .filter(|attempt_timeout_ms| *attempt_timeout_ms > 0);

    for attempt in 0u32.. {
        if signal.is_cancelled() {
            return Err(HttpError::Aborted);
        }
        if timeout_deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
            return Err(HttpError::Timeout);
        }
        let attempt_deadline = attempt_timeout_ms.map(|attempt_timeout_ms| {
            tokio::time::Instant::now() + Duration::from_millis(attempt_timeout_ms)
        });
        let request = HttpRequest {
            method: HttpMethod::Get,
            url: url.to_owned(),
            headers: headers.clone(),
            body: None,
            timeout_ms: None,
            signal: signal.clone(),
        };
        let response = tokio::select! {
            () = signal.cancelled() => Err(HttpError::Aborted),
            () = async {
                match timeout_deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending().await,
                }
            } => Err(HttpError::Timeout),
            () = async {
                match attempt_deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending().await,
                }
            } => Err(HttpError::Timeout),
            response = client.execute(request) => response,
        };

        match response {
            Ok(response) => {
                let should_retry =
                    retry_on_status && RETRYABLE_STATUS_CODES.contains(&response.status);
                if should_retry && attempt < max_retries {
                    continue;
                }
                return Ok(response);
            }
            Err(error) => {
                // Caller cancellation and the total budget are terminal; the
                // attempt budget is not — its expiry retries.
                if signal.is_cancelled() {
                    return Err(error);
                }
                if timeout_deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline)
                {
                    return Err(HttpError::Timeout);
                }
                if attempt >= max_retries {
                    return Err(error);
                }
            }
        }
    }
    unreachable!("the retry loop always returns")
}

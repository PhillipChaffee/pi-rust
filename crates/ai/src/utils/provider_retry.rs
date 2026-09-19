//! Provider-request retries, ported from
//! `packages/ai/src/utils/provider-retry.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Reproduces the retry behavior of the pinned OpenAI and Anthropic SDKs
//! while making their backoff sleep interruptible: their built-in retry
//! timers ignore the request's abort signal, so callers invoke the client
//! with zero retries and wrap the request with [`retry_provider_request`].
//! Provider-requested delays above `maxRetryDelayMs` fail immediately
//! (60 seconds by default); a cap of zero disables the limit.
//!
//! Porting restatements: fake timers become the tokio paused clock, and the
//! jitter source is injectable for deterministic tests. The HTTP-date form
//! of the `retry-after` header is not parsed here — the client seam that
//! lands with the HttpClient child maps its response errors onto
//! [`ProviderRequestError`] and carries the date-parsed delay, since no
//! upstream test exercises it.

use std::collections::BTreeMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

/// The delay cap a server-requested retry wait may not exceed, upstream's
/// `DEFAULT_MAX_RETRY_DELAY_MS`.
pub const DEFAULT_MAX_RETRY_DELAY_MS: u64 = 60_000;

/// The error surface [`retry_provider_request`] retries on: the HTTP status
/// and headers the provider sent with a failure, plus its message. The
/// client seam maps its concrete errors onto this shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderRequestError {
    /// The failure description.
    pub message: String,
    /// The HTTP status, when the failure carried one.
    pub status: Option<u16>,
    /// The response headers, when the failure carried any.
    pub headers: Option<BTreeMap<String, String>>,
}

impl ProviderRequestError {
    /// A provider error with the given status and headers.
    #[must_use]
    pub fn new(
        status: Option<u16>,
        headers: Option<BTreeMap<String, String>>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            message: message.into(),
            status,
            headers,
        }
    }

    /// The abort failure upstream's `createAbortError` produces: name
    /// `AbortError`, message `Request aborted`.
    #[must_use]
    pub fn aborted() -> Self {
        Self::new(None, None, "Request aborted")
    }

    /// Whether this error is the abort failure.
    #[must_use]
    pub fn is_abort(&self) -> bool {
        self.status.is_none() && self.headers.is_none() && self.message == "Request aborted"
    }

    /// Read a header case-insensitively, like `Headers.get`.
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.as_ref().and_then(|headers| {
            headers
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, value)| value.as_str())
        })
    }
}

impl std::fmt::Display for ProviderRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProviderRequestError {}

impl std::fmt::Debug for ProviderRetryOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderRetryOptions")
            .field("max_retries", &self.max_retries)
            .field("max_retry_delay_ms", &self.max_retry_delay_ms)
            .field("signal", &self.signal)
            .finish_non_exhaustive()
    }
}

/// The options of [`retry_provider_request`].
#[derive(Clone, Default)]
pub struct ProviderRetryOptions {
    /// Retry attempts after the initial request. Default: 0.
    pub max_retries: u32,
    /// Provider-requested delays above this fail immediately; 0 disables
    /// the limit. Default: 60 seconds.
    pub max_retry_delay_ms: Option<u64>,
    /// The request's cancellation token.
    pub signal: Option<CancellationToken>,
    /// The jitter source of the exponential backoff, upstream's
    /// `Math.random`; injectable for deterministic tests. Default: system
    /// randomness.
    pub random: Option<Arc<dyn Fn() -> f64 + Send + Sync>>,
}

/// The default jitter source, a uniform `[0, 1)` sample.
fn system_random() -> f64 {
    let mut bytes = [0u8; 4];
    let fill = getrandom::fill(&mut bytes);
    if fill.is_err() {
        return 0.0;
    }
    let sample = u32::from_le_bytes(bytes);
    f64::from(sample) / f64::from(u32::MAX)
}

/// Mirror the retry behavior of the pinned OpenAI and Anthropic SDKs.
///
/// The backoff sleep is interruptible. Each retry is a fresh request, so the
/// SDK's own retry-count header stays zero. Errors the provider marks
/// non-retryable, exhausted retries, and aborts surface immediately.
///
/// # Errors
/// Returns the provider's error when the request fails without a retryable
/// shape or the retry budget is spent, and the abort-shaped error when the
/// signal cancels the request or its backoff.
pub async fn retry_provider_request<T, F, Fut>(
    mut request: F,
    options: &ProviderRetryOptions,
) -> Result<T, ProviderRequestError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, ProviderRequestError>>,
{
    let mut retries_remaining = options.max_retries;

    loop {
        let outcome = request().await;
        let error = match outcome {
            Ok(value) => return Ok(value),
            Err(error) => error,
        };

        if options
            .signal
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(ProviderRequestError::aborted());
        }
        if retries_remaining == 0 || !is_retryable_provider_error(&error) {
            return Err(error);
        }

        let retry_index = options.max_retries - retries_remaining;
        retries_remaining -= 1;
        let delay_ms = get_retry_delay_ms(
            &error,
            retry_index,
            options.max_retry_delay_ms,
            jitter_of(options),
        )?;
        abortable_sleep(delay_ms, options.signal.as_ref()).await?;
    }
}

/// Sleep the delay, stopping early when the signal aborts.
///
/// # Errors
/// Returns the abort-shaped error when the signal cancels during the sleep.
async fn abortable_sleep(
    delay_ms: f64,
    signal: Option<&CancellationToken>,
) -> Result<(), ProviderRequestError> {
    // The delay is clamped non-negative and its sub-millisecond remainder
    // truncates, matching the JS scheduler the timeout ports.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the delay is clamped to a non-negative millisecond wait before the cast"
    )]
    let delay = std::time::Duration::from_millis(delay_ms.max(0.0) as u64);
    let Some(signal) = signal else {
        tokio::time::sleep(delay).await;
        return Ok(());
    };
    tokio::select! {
        () = signal.cancelled() => Err(ProviderRequestError::aborted()),
        () = tokio::time::sleep(delay) => Ok(()),
    }
}

fn jitter_of(options: &ProviderRetryOptions) -> Box<dyn Fn() -> f64 + '_> {
    match &options.random {
        Some(random) => {
            let random = random.clone();
            Box::new(move || random())
        }
        None => Box::new(system_random),
    }
}

/// The pinned OpenAI/Anthropic SDK retry policy.
///
/// The `x-should-retry` header decides when present; a status-less failure is
/// transient; 408, 409, 429, and 5xx are retryable. Review when either SDK is
/// upgraded.
#[must_use]
pub fn is_retryable_provider_error(error: &ProviderRequestError) -> bool {
    match error.header("x-should-retry") {
        Some("true") => return true,
        Some("false") => return false,
        _ => {}
    }
    match error.status {
        None | Some(408 | 409 | 429) => true,
        Some(status) => status >= 500,
    }
}

/// The server-requested delay: `retry-after-ms`, then `retry-after`
/// seconds, then the SDKs' exponential schedule with jitter. Delays above
/// the cap fail with the requested delay in the message so higher-level
/// retry logic can surface them.
///
/// # Errors
/// Returns the limit-failure error when the server requests a delay above
/// the cap and the cap is enabled.
fn get_retry_delay_ms(
    error: &ProviderRequestError,
    retry_index: u32,
    max_retry_delay_ms: Option<u64>,
    random: impl Fn() -> f64,
) -> Result<f64, ProviderRequestError> {
    if let Some(retry_after_ms) = error
        .header("retry-after-ms")
        .and_then(|value| value.parse::<f64>().ok())
    {
        return validate_server_retry_delay_ms(retry_after_ms, max_retry_delay_ms, &error.message);
    }

    if let Some(retry_after) = error
        .header("retry-after")
        .and_then(|value| value.parse::<f64>().ok())
    {
        return validate_server_retry_delay_ms(
            retry_after * 1000.0,
            max_retry_delay_ms,
            &error.message,
        );
    }

    let exponential_delay =
        (0.5 * 2f64.powi(i32::try_from(retry_index).unwrap_or(i32::MAX))).min(8.0) * 1000.0;
    Ok(exponential_delay * random().mul_add(-0.25, 1.0))
}

fn validate_server_retry_delay_ms(
    delay_ms: f64,
    max_retry_delay_ms: Option<u64>,
    message: &str,
) -> Result<f64, ProviderRequestError> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "the delay cap enters the JS number math that compares server-requested delays"
    )]
    let max_delay_ms = max_retry_delay_ms.unwrap_or(DEFAULT_MAX_RETRY_DELAY_MS) as f64;
    if max_delay_ms > 0.0 && delay_ms > max_delay_ms {
        let requested_seconds = (delay_ms / 1000.0).ceil();
        let max_seconds = (max_delay_ms / 1000.0).ceil();
        return Err(ProviderRequestError::new(
            None,
            None,
            format!(
                "Server requested {requested_seconds}s retry delay (max: {max_seconds}s). {message}"
            ),
        ));
    }
    Ok(delay_ms)
}

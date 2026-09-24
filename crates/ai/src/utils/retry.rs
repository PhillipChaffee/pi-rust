//! Assistant-call retries, ported from
//! `packages/ai/src/utils/retry.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Retry policy: bounded attempts with exponential backoff
//! (`baseDelayMs * 2^(attempt-1)`), `maxAgentDelayMs` capping each computed
//! delay (default 60 seconds). Mirrors `settings.retry` (`enabled`,
//! `maxRetries`, `baseDelayMs`, `maxAgentDelayMs`) in the coding agent; the
//! classifier and the policy-driven retry loop live together here so the SDK
//! and other callers can reuse them.
//!
//! Porting restatements: fake timers become the tokio clock, and the
//! optionally-async callbacks collapse to sync closures — the loop awaits
//! each before proceeding, so ordering is the observable contract.

use std::sync::{Arc, LazyLock};

use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::types::{AssistantMessage, StopReason};

/// Subscription and account limits that look like throttles but must never
/// retry, upstream's `NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN`.
#[expect(
    clippy::expect_used,
    reason = "the joined pattern is a compile-time constant; a failure is a programming error, not a runtime condition"
)]
static NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        "(?i){}",
        [
            // OpenCode Go/free-tier limits returned as 429 JSON error types
            // by OpenCode's Zen API. These are subscription/account limits,
            // not transient throttles.
            "GoUsageLimitError",
            "FreeUsageLimitError",
            // OpenCode Go subscription-limit text asks users to enable
            // available-balance usage after rolling/weekly/monthly limits
            // are reached.
            "Monthly usage limit reached",
            "available balance",
            // Generic quota/budget/billing exhaustion. `insufficient_quota`
            // is OpenAI's quota/billing error code; the other strings cover
            // common gateway wording.
            "insufficient_quota",
            "out of budget",
            "quota exceeded",
            "billing",
        ]
        .join("|")
    ))
    .expect("the joined limit pattern is a valid regex")
});

/// The transient provider, transport, and stream failures worth retrying,
/// upstream's `RETRYABLE_PROVIDER_ERROR_PATTERN` alternatives.
const RETRYABLE_PROVIDER_ERROR_ALTERNATIVES: &[&str] = &[
    // Generic provider load, HTTP status, and server-side transient failures.
    "overloaded",
    "rate.?limit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "524",
    "service.?unavailable",
    "server.?error",
    "internal.?error",
    // Wrapper/provider text for transient upstream failures, including
    // OpenRouter "Provider returned error" responses (pi #2264).
    "provider.?returned.?error",
    "exceeded request buffer limit while retrying upstream",
    // Network, proxy, and fetch transport failures. This includes OpenAI
    // Codex raw-fetch failures such as "upstream connect", "connection
    // refused", and "reset before headers" (pi #733), plus OpenRouter
    // connection drops (pi #3317).
    "network.?error",
    "connection.?error",
    "connection.?refused",
    "connection.?lost",
    "other side closed",
    "fetch failed",
    "getaddrinfo",
    "ENOTFOUND",
    "EAI_AGAIN",
    "upstream.?connect",
    "reset before headers",
    "socket hang up",
    "socket connection was closed",
    "timed? out",
    "timeout",
    "terminated",
    // WebSocket transports can report close/error text instead of
    // HTTP/fetch text.
    "websocket.?closed",
    "websocket.?error",
    // Premature stream endings from SDKs and transports. Anthropic can throw
    // "stream ended without ..." and "Anthropic stream ended before
    // message_stop" (pi #4433); Bedrock/Smithy can throw an HTTP/2
    // no-response error (pi #3594).
    "ended without",
    "stream ended before message_stop",
    "stream ended before a terminal response event",
    "http2 request did not get a response",
    // Provider-requested retry delay cap failures should flow through the
    // outer retry policy so callers can surface/abort the backoff (pi #1123).
    "retry delay",
    // Explicit retry guidance emitted mid-stream by OpenAI Responses and
    // Bedrock stream exceptions (pi #6019).
    "you can retry your request",
    "try your request again",
    "please retry your request",
    // gRPC based providers (e.g. NVIDIA NIM)
    "ResourceExhausted",
];

#[expect(
    clippy::expect_used,
    reason = "the joined pattern is a compile-time constant; a failure is a programming error, not a runtime condition"
)]
static RETRYABLE_PROVIDER_ERROR_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        "(?i){}",
        RETRYABLE_PROVIDER_ERROR_ALTERNATIVES.join("|")
    ))
    .expect("the joined retryable pattern is a valid regex")
});

/// Retry policy: bounded attempts with exponential backoff.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetryPolicy {
    /// Whether retries are enabled.
    pub enabled: bool,
    /// Max retry attempts (0 = no retries). The initial call never counts as
    /// a retry.
    pub max_retries: u32,
    /// Base delay in ms. Per-attempt delay is `baseDelayMs * 2^(attempt-1)`
    /// before jitter.
    pub base_delay_ms: u64,
    /// Optional cap for agent-level retry delays in ms. Defaults to 60
    /// seconds.
    pub max_agent_delay_ms: Option<u64>,
}

/// The default cap for agent-level retry delays, upstream's
/// `DEFAULT_MAX_AGENT_RETRY_DELAY_MS`.
pub const DEFAULT_MAX_AGENT_RETRY_DELAY_MS: u64 = 60_000;

/// The policy a caller with no policy falls back to: no retries, no delay.
const DISABLED_RETRY_POLICY: RetryPolicy = RetryPolicy {
    enabled: false,
    max_retries: 0,
    base_delay_ms: 0,
    max_agent_delay_ms: None,
};

/// The delay before retry `attempt` (1-indexed).
///
/// The base delay doubles per prior attempt, saturated like upstream's
/// safe-integer clamp, capped by `maxAgentDelayMs` (default 60 seconds; a
/// zero cap holds the delay at zero).
#[must_use]
pub fn retry_delay_ms(policy: &RetryPolicy, attempt: u32) -> u64 {
    // `baseDelayMs * 2^(attempt-1)`, shifted not exponentiated: the doubling
    // saturates at the u64 ceiling, upstream's safe-integer clamp.
    let exponent = attempt.saturating_sub(1);
    let doubling = u64::try_from(1u128 << exponent).unwrap_or(u64::MAX);
    let delay = policy.base_delay_ms.saturating_mul(doubling);
    delay.min(
        policy
            .max_agent_delay_ms
            .unwrap_or(DEFAULT_MAX_AGENT_RETRY_DELAY_MS),
    )
}

impl std::fmt::Debug for RetryCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryCallbacks")
            .field("on_retry_scheduled", &self.on_retry_scheduled.is_some())
            .field(
                "on_retry_attempt_start",
                &self.on_retry_attempt_start.is_some(),
            )
            .field("on_retry_finished", &self.on_retry_finished.is_some())
            .finish()
    }
}

/// The retry-scheduled callback: attempt (1-indexed), max attempts, delay ms,
/// and the error text.
type RetryScheduledCallback = Arc<dyn Fn(u32, u32, u64, &str) + Send + Sync>;
/// The retry-finished callback: success, the last attempt number, and the
/// final error text when the loop ended in failure.
type RetryFinishedCallback = Arc<dyn Fn(bool, u32, Option<String>) + Send + Sync>;

/// The optional callbacks [`retry_assistant_call`] emits around each retry.
#[derive(Clone, Default)]
pub struct RetryCallbacks {
    /// Emitted before the backoff sleep of each retry attempt (1-indexed).
    pub on_retry_scheduled: Option<RetryScheduledCallback>,
    /// Emitted after the backoff sleep, immediately before the retried call
    /// starts.
    pub on_retry_attempt_start: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Emitted once when the loop ends: success if a later call completed
    /// normally.
    pub on_retry_finished: Option<RetryFinishedCallback>,
}

/// Run a single assistant-producing call with bounded retry on transient
/// errors.
///
/// Behavior:
/// - A successful response is returned immediately. Aborts are terminal and
///   never retried, but reported as unsuccessful if they happen after a
///   retry was scheduled. Aborts during the backoff sleep are normalized to
///   an aborted [`AssistantMessage`] too, so callers do not need to care
///   when cancellation happened.
/// - A non-retryable error (per [`is_retryable_assistant_error`], including
///   quota/billing exhaustion) is returned immediately so deterministic
///   errors fail fast.
/// - Otherwise retries up to `maxRetries` times with exponential backoff,
///   emitting `onRetryScheduled` before each sleep, `onRetryAttemptStart`
///   after each sleep before the retried call starts, and `onRetryFinished`
///   once at the end (whether the loop ends in success, exhausted retries,
///   or an aborted backoff).
///
/// When `policy` is [`None`] or disabled, the first response is returned
/// unchanged (equivalent to calling `produce` directly).
pub async fn retry_assistant_call<P, Fut>(
    mut produce: P,
    policy: Option<&RetryPolicy>,
    signal: Option<&CancellationToken>,
    callbacks: Option<&RetryCallbacks>,
) -> AssistantMessage
where
    P: FnMut() -> Fut,
    Fut: Future<Output = AssistantMessage>,
{
    let max_attempts = policy
        .filter(|policy| policy.enabled)
        .map_or(0, |policy| policy.max_retries);

    let mut attempt: u32 = 0;
    let mut last_retry: Option<(u32, String)> = None;
    loop {
        let response = produce().await;

        // Abort: terminal but not successful. Never retry an aborted message.
        if response.stop_reason == StopReason::Aborted {
            if let Some((attempt, _)) = last_retry
                && let Some(on_retry_finished) =
                    callbacks.and_then(|callbacks| callbacks.on_retry_finished.as_ref())
            {
                on_retry_finished(false, attempt, None);
            }
            return response;
        }

        // Success: non-error, non-abort responses return as-is.
        if response.stop_reason != StopReason::Error {
            if let Some(last_retry) = &last_retry
                && let Some(on_retry_finished) =
                    callbacks.and_then(|callbacks| callbacks.on_retry_finished.as_ref())
            {
                on_retry_finished(true, last_retry.0, None);
            }
            return response;
        }

        // Non-retryable, or budget exhausted: return the final error message.
        if attempt >= max_attempts || !is_retryable_assistant_error(&response) {
            if let Some(last_retry) = &last_retry
                && let Some(on_retry_finished) =
                    callbacks.and_then(|callbacks| callbacks.on_retry_finished.as_ref())
            {
                on_retry_finished(false, last_retry.0, Some(last_retry.1.clone()));
            }
            return response;
        }

        attempt += 1;
        let error_message = response
            .error_message
            .clone()
            .unwrap_or_else(|| String::from("Unknown error"));
        last_retry = Some((attempt, error_message.clone()));
        // A retryable error past the budget check implies an enabled policy;
        // the disabled stand-in is unreachable from the delay computation.
        let policy = policy.unwrap_or(&DISABLED_RETRY_POLICY);
        let delay_ms = retry_delay_ms(policy, attempt);
        if let Some(on_retry_scheduled) =
            callbacks.and_then(|callbacks| callbacks.on_retry_scheduled.as_ref())
        {
            on_retry_scheduled(attempt, max_attempts, delay_ms, &error_message);
        }

        // Normalize aborts during retry backoff to the same assistant-message
        // shape as provider stream aborts, so callers do not need to care
        // when cancellation happened.
        let slept = if let Some(signal) = signal {
            tokio::select! {
                () = signal.cancelled() => Err(()),
                () = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => Ok(()),
            }
        } else {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            Ok(())
        };
        if slept.is_err() {
            if let Some(on_retry_finished) =
                callbacks.and_then(|callbacks| callbacks.on_retry_finished.as_ref())
            {
                on_retry_finished(false, attempt, Some(error_message));
            }
            return AssistantMessage {
                stop_reason: StopReason::Aborted,
                error_message: None,
                ..response
            };
        }
        if let Some(on_retry_attempt_start) =
            callbacks.and_then(|callbacks| callbacks.on_retry_attempt_start.as_ref())
        {
            on_retry_attempt_start();
        }
    }
}

/// Classifies whether a failed assistant message looks like a transient
/// provider or transport error, so callers can decide if the last assistant
/// turn should be restarted.
///
/// This does not implement retry policy. Callers should first handle context
/// overflow separately, then apply their own retry budget, backoff, and
/// reporting before restarting the assistant turn.
#[must_use]
pub fn is_retryable_assistant_error(message: &AssistantMessage) -> bool {
    if message.stop_reason != StopReason::Error {
        return false;
    }
    let Some(error_message) = &message.error_message else {
        return false;
    };
    if NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN.is_match(error_message) {
        return false;
    }
    RETRYABLE_PROVIDER_ERROR_PATTERN.is_match(error_message)
}

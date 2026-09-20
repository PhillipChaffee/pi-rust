//! The OAuth device-code poll loop, ported from
//! `packages/ai/src/auth/oauth/device-code.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! RFC 8628 polling: an interval, a `slow_down` bump, an expiry deadline,
//! and cancellation. Sleeps and the deadline ride the paused tokio clock —
//! the injected clock, so the fake-timer suites drive real code.
//!
//! Porting restatements this module records:
//!
//! - The deadline is a duration from start on the tokio clock, not an epoch
//!   timestamp: `Date.now() + expiresInSeconds * 1000` versus
//!   `tokio::time::Instant::now() + ...` behave identically under the
//!   pause-and-advance tests.
//! - `abortableSleep` collapses into the select in
//!   [`poll_oauth_device_code_flow`]'s wait: cancellation surfaces as the
//!   `Login cancelled` rejection the TypeScript helper carries.

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::auth_error;
use crate::auth::types::AuthError;
use crate::types::BoxedFuture;

const CANCEL_MESSAGE: &str = "Login cancelled";
const TIMEOUT_MESSAGE: &str = "Device flow timed out";
const SLOW_DOWN_TIMEOUT_MESSAGE: &str = "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again.";
const MINIMUM_INTERVAL_MS: u64 = 1000;
// RFC 8628 section 3.2: if the authorization server omits `interval`, the
// client must use 5 seconds.
const DEFAULT_POLL_INTERVAL_SECONDS: u64 = 5;
// RFC 8628 section 3.5: `slow_down` means the polling interval must increase
// by 5 seconds.
const SLOW_DOWN_INTERVAL_INCREMENT_MS: u64 = 5000;

/// The result of one poll, upstream's `OAuthDeviceCodePollResult`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PollOutcome<T> {
    /// The server is still waiting for the user.
    Pending,
    /// The server asked the client to slow down, optionally reporting the
    /// new required interval in seconds.
    SlowDown {
        /// The server-provided new interval, when it reports one.
        interval_seconds: Option<u64>,
    },
    /// The flow failed terminally, with the user-facing reason.
    Failed(String),
    /// The flow completed with the carried value.
    Complete(T),
}

impl<T> PollOutcome<T> {
    /// The completed value, [`None`] for any other outcome.
    #[must_use]
    pub fn complete(self) -> Option<T> {
        match self {
            Self::Complete(value) => Some(value),
            Self::Pending | Self::SlowDown { .. } | Self::Failed(_) => None,
        }
    }
}

/// The options of [`poll_oauth_device_code_flow`], upstream's
/// `OAuthDeviceCodePollOptions`.
pub struct PollOptions<T> {
    /// The server's polling interval in seconds, when it reports one.
    pub interval_seconds: Option<u64>,
    /// The device code's lifetime in seconds, when it reports one.
    pub expires_in_seconds: Option<u64>,
    /// Wait one interval before the first poll, for servers that count an
    /// immediate poll against the user.
    pub wait_before_first_poll: bool,
    /// The poll step itself; its rejections propagate out of the flow.
    pub poll: Arc<dyn Fn() -> BoxedFuture<'static, Result<PollOutcome<T>, AuthError>> + Send + Sync>,
    /// Cancels the whole flow.
    pub signal: CancellationToken,
}

impl<T> std::fmt::Debug for PollOptions<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PollOptions")
            .field("interval_seconds", &self.interval_seconds)
            .field("expires_in_seconds", &self.expires_in_seconds)
            .field("wait_before_first_poll", &self.wait_before_first_poll)
            .finish_non_exhaustive()
    }
}

/// Sleep for `ms`, stopping with the cancel message when the token aborts.
async fn abortable_sleep(ms: u64, signal: &CancellationToken) -> Result<(), AuthError> {
    tokio::select! {
        () = signal.cancelled() => Err(auth_error(CANCEL_MESSAGE.to_owned())),
        () = tokio::time::sleep(Duration::from_millis(ms)) => Ok(()),
    }
}

/// Drive an RFC 8628 device-code flow to completion: poll, honour
/// `slow_down`, and time out at the expiry.
///
/// # Errors
/// Rejects with `Login cancelled` on cancellation, the poll step's own
/// error on transport failure, the poll's `Failed` message, and the
/// timeout message — the slow-down-flavoured one when any `slow_down` was
/// seen — when the deadline passes.
pub async fn poll_oauth_device_code_flow<T>(options: PollOptions<T>) -> Result<T, AuthError> {
    // An absent lifetime is upstream's `Number.POSITIVE_INFINITY`.
    let deadline: Option<tokio::time::Instant> = options.expires_in_seconds.map(|seconds| {
        tokio::time::Instant::now() + Duration::from_secs(saturating_seconds_ms(seconds))
    });
    let mut interval_ms = MINIMUM_INTERVAL_MS.max(
        options
            .interval_seconds
            .unwrap_or(DEFAULT_POLL_INTERVAL_SECONDS)
            * 1000,
    );

    let mut slow_down_responses = 0_u64;
    if options.signal.is_cancelled() {
        return Err(auth_error(CANCEL_MESSAGE.to_owned()));
    }
    // waitBeforeFirstPoll: some servers count an immediate poll as a strike.
    if options.wait_before_first_poll {
        let remaining_ms = remaining_ms(deadline);
        if remaining_ms > 0 {
            abortable_sleep(interval_ms.min(remaining_ms), &options.signal).await?;
        }
    }

    while deadline.is_none_or(|deadline| tokio::time::Instant::now() < deadline) {
        if options.signal.is_cancelled() {
            return Err(auth_error(CANCEL_MESSAGE.to_owned()));
        }

        let result = (options.poll)().await?;
        match result {
            PollOutcome::Complete(value) => return Ok(value),
            PollOutcome::Failed(message) => return Err(auth_error(message)),
            PollOutcome::SlowDown { interval_seconds } => {
                slow_down_responses += 1;
                // Use the server-provided interval when given (GitHub reports
                // the new required minimum in `interval`); trusting only a
                // client-tracked value risks polling early forever under
                // WSL/VM clock drift. Otherwise apply RFC 8628 section 3.5:
                // increase by 5 seconds.
                interval_ms = match interval_seconds {
                    Some(interval) if interval > 0 => {
                        MINIMUM_INTERVAL_MS.max(saturating_seconds_ms(interval))
                    }
                    _ => MINIMUM_INTERVAL_MS.max(interval_ms + SLOW_DOWN_INTERVAL_INCREMENT_MS),
                };
            }
            PollOutcome::Pending => {}
        }

        let remaining_ms = remaining_ms(deadline);
        if remaining_ms == 0 {
            break;
        }
        abortable_sleep(interval_ms.min(remaining_ms), &options.signal).await?;
    }

    Err(auth_error(
        if slow_down_responses > 0 {
            SLOW_DOWN_TIMEOUT_MESSAGE
        } else {
            TIMEOUT_MESSAGE
        }
        .to_owned(),
    ))
}

/// Seconds converted to the milliseconds they represent, saturating.
#[must_use]
const fn saturating_seconds_ms(seconds: u64) -> u64 {
    seconds.saturating_mul(1000)
}

/// Milliseconds until the deadline, `u64::MAX` when the deadline is open.
#[must_use]
fn remaining_ms(deadline: Option<tokio::time::Instant>) -> u64 {
    let Some(deadline) = deadline else {
        return u64::MAX;
    };
    deadline
        .saturating_duration_since(tokio::time::Instant::now())
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

/// The message an aborted login carries, shared with the flows' request
/// paths.
pub(crate) const LOGIN_CANCELLED_MESSAGE: &str = CANCEL_MESSAGE;

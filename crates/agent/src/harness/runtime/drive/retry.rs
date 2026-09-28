//! The retry-wait clock helpers the drive procedures share, ported from
//! upstream `src/harness/runtime/drive/retry.ts`.
//!
//! `notBefore` is a wire field ([`crate::harness::agent_harness::DriveWaitReason::Retry`]),
//! so the saturation ceiling is JS's `Number.MAX_SAFE_INTEGER` (2^53 − 1),
//! not `i64::MAX`: the value crosses JSON round-trips and callers compare it
//! against timestamps the session backends send.

use std::sync::Arc;
use std::time::Duration;

use pi_ai::utils::retry::RetryPolicy;
use pi_ai::utils::retry::retry_delay_ms;

use crate::harness::gate::AbortRequested;
use crate::harness::runtime::clock::now_ms;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::types::NormalizedRetryPolicy;

/// JavaScript's `Number.MAX_SAFE_INTEGER` (2^53 − 1), the ceiling
/// `notBefore` saturates at, upstream's `Number.MAX_SAFE_INTEGER` fallback.
const MAX_SAFE_INTEGER: i64 = 9_007_199_254_740_991;

/// The wall-clock instant the next retry attempt may start at, upstream's
/// `retryNotBefore`: `now + retryDelayMs(policy, attempt)` clamped to the
/// safe-integer ceiling.
///
/// The policy is the generation context's [`NormalizedRetryPolicy`]; the
/// delay re-derives through pi-ai's `retryDelayMs` over the equivalent
/// [`RetryPolicy`] (`maxAttempts − 1` retries, the normalized delay cap made
/// explicit), whose doubling and cap produce upstream's numbers.
#[must_use]
pub(crate) fn retry_not_before(policy: &NormalizedRetryPolicy, attempt: u32, now: i64) -> i64 {
    let policy = RetryPolicy {
        enabled: true,
        max_retries: policy.max_attempts.saturating_sub(1),
        base_delay_ms: policy.base_delay_ms,
        max_agent_delay_ms: Some(policy.max_agent_delay_ms),
    };
    let delay = retry_delay_ms(&policy, attempt);
    // The safe-integer ceiling binds on the SUM, upstream's
    // `Number.isSafeInteger(sum) ? sum : Number.MAX_SAFE_INTEGER` — an
    // oversized cap can push `now + delay` past 2^53 − 1 even when each
    // operand fits.
    let sum = i64::try_from(delay)
        .ok()
        .and_then(|delay| now.checked_add(delay));
    match sum {
        Some(sum) if (-MAX_SAFE_INTEGER..=MAX_SAFE_INTEGER).contains(&sum) => sum,
        _ => MAX_SAFE_INTEGER,
    }
}

/// Awaits the wall clock past `not_before`, racing the drive pass's gate
/// signal, upstream's `waitUntil(notBefore, signal)`.
///
/// Each wake re-derives the remaining wait from the wall clock (upstream's
/// re-arming `setTimeout(check, ...)`), so wall-clock jumps and waits longer
/// than any single sleep still resolve. The `2_147_483_647` ms sleep cap is
/// upstream's `setTimeout` ceiling; tokio carries no 32-bit cap and the
/// re-arm makes it unobservable, kept only to preserve the wake cadence.
///
/// Upstream registers the abort listener before the first clock check, so an
/// already-aborted signal rejects even when the deadline is due; the loop
/// keeps that order.
///
/// # Errors
/// A [`crate::harness::gate::AbortRequested`] carrying the drive's
/// cancellation when the gate signal aborts — upstream rejects with
/// `signal.reason`, an `AbortRequested` — which the spine's catch awaits
/// before continuing. Admission refusal stays the caller's `gate.admit`
/// rejection; this wait only carries the mid-wait abort.
pub(crate) async fn wait_until(not_before: i64, drive: &Arc<Drive>) -> Result<(), LaneError> {
    let signal = drive.gate.signal().clone();
    loop {
        if signal.aborted() {
            return Err(lane_error(AbortRequested {
                cancellation: drive.abort_cancellation(),
            }));
        }
        let remaining = not_before.saturating_sub(now_ms());
        if remaining <= 0 {
            return Ok(());
        }
        let Ok(remaining) = u64::try_from(remaining) else {
            unreachable!("the remaining wait is positive")
        };
        tokio::select! {
            _ = signal.wait() => {
                return Err(lane_error(AbortRequested {
                    cancellation: drive.abort_cancellation(),
                }));
            }
            () = tokio::time::sleep(Duration::from_millis(remaining.min(2_147_483_647))) => {}
        }
    }
}

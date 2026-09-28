//! The retry-delay suite, ported 1:1 from upstream
//! `test/harness/runtime/drive-retry.test.ts` ("runtime retry delay") at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The suite is a pure-function pin — no fixtures, no session, no clock —
//! and upstream's `it` awaits nothing, so the case is a plain `#[test]`.
//! Upstream's policy literal is a `Pick<RetryPolicy, "baseDelayMs" |
//! "maxAgentDelayMs">`; the normalized policy the Rust signature carries
//! also holds the attempt budget, which the delay arithmetic never reads,
//! so the test supplies one admitting attempt 5.
//!
//! The boundary cases past upstream's suite bind the arms upstream's
//! behavior pins but its tests never reach: the safe-integer saturation
//! `retryNotBefore`'s sum and delay both cross, and `waitUntil`'s
//! already-aborted rejection — the abort listener registering before the
//! first clock check, so even a due deadline rejects.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;

use crate::harness::agent_harness::DriveOptions;
use crate::harness::context::background_context;
use crate::harness::runtime::drive::retry::{retry_not_before, wait_until};
use crate::harness::runtime::types::Drive;
use crate::harness::session::types::NormalizedRetryPolicy;

/// Upstream `it`: uses capped delay when computing retry readiness.
#[test]
fn uses_capped_delay_when_computing_retry_readiness() {
    // The delay-cap regression, #8826.
    let not_before = retry_not_before(
        &NormalizedRetryPolicy {
            max_attempts: 5,
            base_delay_ms: 2_000,
            max_agent_delay_ms: 30_000,
        },
        5,
        100,
    );
    assert_eq!(not_before, 30_100, "the cap bounds the readiness");
}

/// The boundary arm of the safe-integer ceiling, upstream's
/// `Number.isSafeInteger(sum)` check: a delay whose sum with the read clock
/// passes 2^53 − 1 saturates at the ceiling, and a delay past the i64
/// ceiling saturates the same way through the failed conversion.
#[test]
fn saturates_the_readiness_at_the_safe_integer_ceiling() {
    let saturated = retry_not_before(
        &NormalizedRetryPolicy {
            max_attempts: 2,
            base_delay_ms: 9_200_000_000_000_000,
            max_agent_delay_ms: u64::MAX,
        },
        1,
        0,
    );
    assert_eq!(
        saturated, 9_007_199_254_740_991,
        "the sum saturates at the safe-integer ceiling",
    );
    let overflowing = retry_not_before(
        &NormalizedRetryPolicy {
            max_attempts: 2,
            base_delay_ms: u64::MAX,
            max_agent_delay_ms: u64::MAX,
        },
        1,
        0,
    );
    assert_eq!(
        overflowing, 9_007_199_254_740_991,
        "the delay overflow saturates too",
    );
}

/// The boundary arm of the wait's registered-listener-before-clock-check
/// order, upstream's `waitUntil` rejecting an already-aborted signal with
/// `signal.reason` even when the deadline is due: the wait rejects instead
/// of resolving through the clock check.
#[tokio::test]
async fn rejects_an_already_aborted_wait_even_when_the_deadline_is_due() {
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: "retry-wait".to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    // The settled cancellation, upstream's already-fired `signal.reason`:
    // dropping the sender resolves the receiver immediately.
    let (sender, cancellation) = tokio::sync::watch::channel(());
    drop(sender);
    drive.begin_abort(cancellation);
    drive.signal_abort();

    let error = wait_until(0, &drive)
        .await
        .expect_err("the aborted wait rejects");
    assert_eq!(
        error.to_string(),
        "Abort requested",
        "the already-aborted wait rejects",
    );
}

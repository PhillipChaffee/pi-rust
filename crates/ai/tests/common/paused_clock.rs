//! The paused-clock helper the fake-timer OAuth suites share: one [`advance`]
//! moves the tokio clock and the epoch [`FixedClock`] in lockstep, the pair
//! `vi.advanceTimersByTimeAsync` + `vi.setSystemTime` gave upstream's
//! `test/oauth-device-code.test.ts` and `test/xai-oauth.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::time::Duration;

use pi_ai::auth::clock::{AuthClock as _, FixedClock};

/// Advance the paused tokio clock by `ms`, asserting the epoch clock rides
/// it one-for-one — the lockstep `vi.advanceTimersByTimeAsync` plus
/// `vi.setSystemTime` gave upstream.
///
/// # Panics
/// Panics when `ms` does not fit an epoch offset, or when the epoch clock
/// drifts from the paused tokio clock — the lockstep the fake-timer suites
/// pin their poll times on.
pub async fn advance(clock: &FixedClock, ms: u64) {
    let before = clock.now_ms();
    tokio::time::advance(Duration::from_millis(ms)).await;
    // No test advances by more than an epoch offset; the fallback keeps the
    // arithmetic total and lets the lockstep assert below reject it.
    assert_eq!(
        clock.now_ms(),
        before + i64::try_from(ms).unwrap_or(i64::MAX),
        "the epoch clock must ride the paused tokio clock"
    );
}

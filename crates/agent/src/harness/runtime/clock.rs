//! The drive tree's epoch-millisecond clock, the seam that stands in for
//! upstream tests' mocked `Date.now()`.
//!
//! Upstream's fake-timer suites pin and move the clock with
//! `vi.setSystemTime`; [`pi_ai::auth::resolve::now_ms`] is a plain
//! wall-clock read with no injection point. Every drive-module clock read
//! goes through [`now_ms`] here: under `cfg(test)` a pin set with
//! [`set_test_now`] stands in for the frozen `Date.now()`, without one the
//! read falls through to pi-ai's wall clock. The override governs reads
//! only — sleeps stay on tokio time so `start_paused` keeps working — so a
//! test moves the clock where upstream moves `Date.now()`.
//!
//! The pin is scoped to the calling thread, upstream's per-file module
//! instance under vitest: libtest runs suites on parallel threads and
//! tokio's current-thread runtime polls the suite's future on that same
//! thread, so every drive read sees its own suite's pin and a moved clock
//! can never leak into a concurrent suite.
//!
//! The override value is a test-supplied clock, never a fabricated display
//! number: house rule says nothing may show a count no provider or session
//! backend sends, so the pin restates an epoch an upstream test fixed and
//! this module never invents one.

#[cfg(test)]
use std::cell::Cell;

use pi_ai::auth::resolve::now_ms as wall_now_ms;

#[cfg(test)]
thread_local! {
    /// The test-supplied clock pin; `None` falls through to pi-ai's wall clock.
    static TEST_OVERRIDE: Cell<Option<i64>> = const { Cell::new(None) };
}

/// Unix milliseconds the drive modules read, upstream's `Date.now()`: the
/// calling thread's pinned test clock when one is set, else pi-ai's wall
/// clock.
#[cfg(test)]
#[must_use]
pub(super) fn now_ms() -> i64 {
    TEST_OVERRIDE.with(Cell::get).unwrap_or_else(wall_now_ms)
}

/// Unix milliseconds the drive modules read, upstream's `Date.now()` with
/// no test mock installed — pi-ai's wall clock read.
#[cfg(not(test))]
#[must_use]
pub(super) fn now_ms() -> i64 {
    wall_now_ms()
}

/// Pins the drive clock to `ms` on the calling thread, the restatement of
/// upstream's `vi.setSystemTime(ms)`.
#[cfg(test)]
pub(super) fn set_test_now(ms: i64) {
    TEST_OVERRIDE.with(|pin| pin.set(Some(ms)));
}

/// Clears the calling thread's test clock pin, restoring pi-ai's wall clock.
#[cfg(test)]
pub(super) fn reset() {
    TEST_OVERRIDE.with(|pin| pin.set(None));
}

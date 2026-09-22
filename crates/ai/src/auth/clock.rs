//! The epoch clock the auth resolution and flows read epoch milliseconds
//! from, the seam fake timers freeze upstream.
//!
//! `Date.now()` does not pause with the tokio clock, and OAuth credentials
//! carry epoch-millisecond expiries, so every place the ported code reads
//! the clock takes an [`AuthClock`]. [`FixedClock`] rides the paused tokio
//! clock: its epoch advances exactly when `tokio::time::advance` does, the
//! single-knob behaviour `vi.advanceTimersByTimeAsync` gives upstream tests.

use std::sync::atomic::AtomicI64;

/// A wall-clock source the auth code reads epoch milliseconds from.
pub trait AuthClock: std::fmt::Debug + Send + Sync {
    /// Current epoch time in milliseconds.
    fn now_ms(&self) -> i64;
}

/// The wall clock, upstream's ambient `Date.now()`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl AuthClock for SystemClock {
    fn now_ms(&self) -> i64 {
        // A wall clock reading before 1970 does not exist; clamp to zero.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| {
                i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
            })
    }
}

/// A fixed clock whose epoch rides the paused tokio clock.
///
/// Constructed at `start_epoch_ms`, it advances one millisecond per
/// millisecond `tokio::time::advance` adds, so tests freeze "now" the way
/// `vi.setSystemTime` does and move it with the same knob as the sleeps.
pub struct FixedClock {
    start_epoch_ms: i64,
    anchor: tokio::time::Instant,
}

impl FixedClock {
    /// Fix "now" at `start_epoch_ms` as of this call's tokio instant.
    #[must_use]
    pub fn new(start_epoch_ms: i64) -> Self {
        Self {
            start_epoch_ms,
            anchor: tokio::time::Instant::now(),
        }
    }
}

impl std::fmt::Debug for FixedClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixedClock")
            .field("start_epoch_ms", &self.start_epoch_ms)
            .finish_non_exhaustive()
    }
}

impl AuthClock for FixedClock {
    fn now_ms(&self) -> i64 {
        self.start_epoch_ms
            + i64::try_from(
                (tokio::time::Instant::now() - self.anchor)
                    .as_millis()
                    .min(i64::MAX as u128),
            )
            .unwrap_or(i64::MAX)
    }
}

/// A clock the tests can step by hand, for assertions that need an exact
/// epoch value without a paused runtime.
#[derive(Debug)]
pub struct SteppedClock(AtomicI64);

impl SteppedClock {
    /// A clock fixed at `epoch_ms`.
    #[must_use]
    pub const fn new(epoch_ms: i64) -> Self {
        Self(AtomicI64::new(epoch_ms))
    }

    /// Move the clock forward by `ms` milliseconds.
    pub fn advance(&self, ms: i64) {
        self.0.fetch_add(ms, std::sync::atomic::Ordering::Relaxed);
    }
}

impl AuthClock for SteppedClock {
    fn now_ms(&self) -> i64 {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

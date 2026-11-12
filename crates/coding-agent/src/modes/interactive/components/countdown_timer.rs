//! The reusable countdown timer, upstream's
//! `src/modes/interactive/components/countdown-timer.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#213).
//!
//! Upstream constructs the timer and arms a `setInterval` that ticks once
//! a second; the port restates the interval as a manual-tick handle the
//! host drives — the frame's render loop owns the wall clock, so
//! `tick` is what the interval body ran, and tests fire it inline (the
//! #78 manual-hide-timer pattern).

use std::cell::Cell;
use std::rc::Rc;

/// A second-by-second countdown toward a callback, upstream's
/// `CountdownTimer`.
pub struct CountdownTimer {
    /// Seconds left before expiry, upstream's `remainingSeconds`.
    remaining_seconds: Cell<i64>,
    /// Whether the underlying interval is armed, upstream's `intervalId`.
    armed: Cell<bool>,
    /// The render request upstream drives over the `TUI` handle, captured
    /// because the frame drives ticks outside the render call.
    request_render: Option<Rc<dyn Fn()>>,
    on_tick: Rc<dyn Fn(i64)>,
    on_expire: Rc<dyn Fn()>,
}

impl std::fmt::Debug for CountdownTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountdownTimer")
            .field("remaining_seconds", &self.remaining_seconds.get())
            .field("armed", &self.armed.get())
            .finish_non_exhaustive()
    }
}

impl CountdownTimer {
    /// Start the countdown: the first tick fires immediately with the
    /// seconds remaining, upstream's constructor body.
    #[must_use]
    pub fn new(
        timeout_ms: u64,
        request_render: Option<Rc<dyn Fn()>>,
        on_tick: Rc<dyn Fn(i64)>,
        on_expire: Rc<dyn Fn()>,
    ) -> Self {
        let remaining = i64::try_from(timeout_ms.div_ceil(1000)).unwrap_or(i64::MAX);
        let timer = Self {
            remaining_seconds: Cell::new(remaining),
            armed: Cell::new(true),
            request_render,
            on_tick,
            on_expire,
        };
        (timer.on_tick)(remaining);
        timer
    }

    /// Elapse one second, upstream's `setInterval` body: decrement, report,
    /// request a render, and expire at zero.
    pub fn tick(&self) {
        if !self.armed.get() {
            return;
        }
        let next = self.remaining_seconds.get() - 1;
        self.remaining_seconds.set(next);
        (self.on_tick)(next);
        if let Some(request_render) = &self.request_render {
            request_render();
        }
        if next <= 0 {
            self.dispose();
            (self.on_expire)();
        }
    }

    /// Stop the countdown, upstream `dispose`; expired timers cannot retick.
    pub fn dispose(&self) {
        self.armed.set(false);
    }

    /// Whether the underlying interval is still armed.
    #[must_use]
    pub const fn is_armed(&self) -> bool {
        self.armed.get()
    }

    /// The seconds left before expiry.
    #[must_use]
    pub const fn remaining_seconds(&self) -> i64 {
        self.remaining_seconds.get()
    }
}

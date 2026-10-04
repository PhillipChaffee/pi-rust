//! The shared one-shot completion slot, the primitive behind upstream's
//! settled-promise reuse: a latch settles at most once and hands late waiters
//! the same value an already-resolved promise would resolve with.
//!
//! Upstream reaches for a promise whenever a caller needs "wait for this to
//! finish" from several places — `closePromise`, a release latch, the
//! handshake promise queued messages chain behind. Rust futures are
//! single-consumer, so the port stores one [`Latch`] and lets every waiter
//! register a oneshot against it.
//!
//! The module is crate-internal; the public surface is the testing
//! `Deferred`.
#![allow(
    clippy::redundant_pub_crate,
    reason = "the latch is crate-internal; the public surface is the testing Deferred"
)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::errors::Failure;
use pi_chord::future::{LocalBoxFuture, boxed};

use tokio::sync::oneshot;

/// The close-shaped latch: one shutdown or release outcome, shared across
/// every waiter that joined.
pub(crate) type CloseLatch = Rc<Latch<Result<(), Failure>>>;

/// The shared completion slot; cloning shares the settle.
pub(crate) struct Latch<T> {
    inner: Rc<LatchInner<T>>,
}

enum LatchState<T> {
    Pending(Vec<oneshot::Sender<T>>),
    Settled(T),
}

struct LatchInner<T> {
    state: RefCell<LatchState<T>>,
}

impl<T> std::fmt::Debug for Latch<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Latch").finish_non_exhaustive()
    }
}

impl<T> Clone for Latch<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<T: Clone + 'static> Latch<T> {
    pub(crate) fn new() -> Self {
        Self {
            inner: Rc::new(LatchInner {
                state: RefCell::new(LatchState::Pending(Vec::new())),
            }),
        }
    }

    /// Whether the slot has settled, the join decision the router's acquire
    /// makes for opens that finished while a request was being dispatched.
    pub(crate) fn is_settled(&self) -> bool {
        matches!(&*self.inner.state.borrow(), LatchState::Settled(_))
    }

    /// Settles the slot with `value`; later settles are dropped, upstream's
    /// resolve-after-resolve no-op.
    pub(crate) fn settle(&self, value: T) {
        let waiters = match &mut *self.inner.state.borrow_mut() {
            LatchState::Settled(_) => return,
            LatchState::Pending(waiters) => std::mem::take(waiters),
        };
        *self.inner.state.borrow_mut() = LatchState::Settled(value.clone());
        for waiter in waiters {
            let _ = waiter.send(value.clone());
        }
    }

    /// The settled value, or a future resolving with it once settled.
    pub(crate) fn wait(&self) -> LocalBoxFuture<T> {
        let inner = Rc::clone(&self.inner);
        boxed(async move {
            loop {
                let receiver = {
                    let mut state = inner.state.borrow_mut();
                    match &mut *state {
                        LatchState::Settled(value) => return value.clone(),
                        LatchState::Pending(waiters) => {
                            let (sender, receiver) = oneshot::channel();
                            waiters.push(sender);
                            receiver
                        }
                    }
                };
                // The settle drains every registered sender, so an error here
                // cannot happen while this future holds the latch; retrying
                // keeps the loop total regardless.
                let _ignored = receiver.await;
            }
        })
    }
}

/// The resolve-only promise upstream's testing `Deferred<T>` exports.
pub struct Deferred<T> {
    latch: Latch<T>,
}

impl<T: Clone + 'static> Default for Deferred<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::fmt::Debug for Deferred<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Deferred").finish_non_exhaustive()
    }
}

impl<T> Clone for Deferred<T> {
    fn clone(&self) -> Self {
        Self {
            latch: self.latch.clone(),
        }
    }
}

impl<T: Clone + 'static> Deferred<T> {
    /// The unresolved deferred, upstream's `new Deferred<T>()`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            latch: Latch::new(),
        }
    }

    /// Resolves the deferred value; later resolves are dropped.
    pub fn resolve(&self, value: T) {
        self.latch.settle(value);
    }

    /// The resolved value, or a future resolving with it.
    #[must_use]
    pub fn promise(&self) -> LocalBoxFuture<T> {
        self.latch.wait()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_latch_debug_renders() {
        let latch: Latch<u32> = Latch::new();
        assert!(format!("{latch:?}").contains("Latch"));
        latch.settle(1);
        assert!(latch.is_settled());
    }

    #[tokio::test]
    async fn the_latch_hands_late_waiters_the_settled_value() {
        let latch: Latch<u32> = Latch::new();
        latch.settle(7);
        assert_eq!(latch.wait().await, 7);
        // A second settle is the no-op, upstream's resolve-after-resolve.
        latch.settle(9);
        assert_eq!(latch.wait().await, 7);
    }
}

//! The crate's boxed-future vocabulary.
//!
//! Upstream chord runs on the JavaScript event loop: one logical thread,
//! cooperative promises, no ambient executor. The port keeps that shape —
//! every asynchronous surface is a [`LocalBoxFuture`], which the caller's
//! runtime drives; the crate itself never spawns and declares no async
//! runtime dependency. `Local` is the point: futures here are never sent
//! across threads, matching the single-threaded contract the tests pin.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

/// A boxed, single-threaded future.
pub type LocalBoxFuture<T> = Pin<Box<dyn Future<Output = T>>>;

/// Boxes a future, the form trait methods and stored continuations take.
#[must_use]
pub fn boxed<T, F: Future<Output = T> + 'static>(future: F) -> LocalBoxFuture<T> {
    Box::pin(future)
}

/// A future already resolved with `value`.
///
/// Upstream spells this `async () => value`; the helper keeps closures that
/// produce a value without an await point off async blocks.
pub fn ready_with<T>(value: T) -> std::future::Ready<T> {
    std::future::ready(value)
}

/// Settles every future and returns the results in input order.
///
/// Upstream's `Promise.all`: callers that must not short-circuit on a
/// failure pass `Result` futures, which makes this `Promise.allSettled`
/// instead. Each input future is polled at most once per wakeup round, so
/// the order in which the futures were started is the order they advance
/// in — the event-loop semantics the subscriber-buffering tests pin.
pub fn join_all<T>(futures: Vec<LocalBoxFuture<T>>) -> impl Future<Output = Vec<T>> {
    let mut results = Vec::with_capacity(futures.len());
    for _ in 0..futures.len() {
        results.push(None);
    }
    JoinAll { futures, results }
}

struct JoinAll<T> {
    futures: Vec<LocalBoxFuture<T>>,
    results: Vec<Option<T>>,
}

impl<T> Unpin for JoinAll<T> {}

impl<T> Future for JoinAll<T> {
    type Output = Vec<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        for (slot, future) in this.results.iter_mut().zip(this.futures.iter_mut()) {
            if slot.is_some() {
                continue;
            }
            if let Poll::Ready(value) = future.as_mut().poll(cx) {
                *slot = Some(value);
            }
        }
        if this.results.iter().all(Option::is_some) {
            let results: Vec<T> = this.results.drain(..).flatten().collect();
            Poll::Ready(results)
        } else {
            Poll::Pending
        }
    }
}
/// Drives one boxed future to completion synchronously when its first poll
/// suffices, the form fire-and-forget calls take where the caller cannot
/// await.
///
/// Upstream fires promises and lets the event loop run them; this crate has
/// no ambient executor, so callers poll once and drop anything still
/// pending, which is the documented restatement of the fire-and-forget
/// shape for futures that settle without yielding.
#[must_use]
pub fn settle_now<T>(mut future: LocalBoxFuture<T>) -> Option<T> {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(value) => Some(value),
        Poll::Pending => None,
    }
}

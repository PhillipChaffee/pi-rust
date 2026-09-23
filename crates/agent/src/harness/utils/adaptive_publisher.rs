//! Publishes the latest state without queuing intermediate mutations,
//! ported from upstream `src/harness/utils/adaptive-publisher.ts`.
//!
//! The first dirty state after idle is immediate. Each publication then
//! buys a delay proportional to its encoded size, with a minimum interval
//! that also bounds event count. A single trailing timer guarantees
//! eventual publication. The trailing timer is a tokio task over the
//! pausable clock (the stack decision's fake-timer substitute), so tests
//! can drive publications deterministically.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio::time::Instant;

/// The callbacks one publisher consults, upstream's
/// `AdaptivePublisherOptions<TValue, TUpdate>`.
pub struct AdaptivePublisherOptions<TValue, TUpdate> {
    /// The current complete state.
    pub snapshot: Arc<dyn Fn() -> TValue + Send + Sync>,
    /// The update from the previously published state to `current`;
    /// `None` when the two states are equivalent.
    pub update: Arc<dyn Fn(Option<&TValue>, &TValue) -> Option<TUpdate> + Send + Sync>,
    /// The encoded size the rate limit buys the delay from.
    pub measure: Arc<dyn Fn(&TUpdate) -> u64 + Send + Sync>,
    /// Delivers one update. Runs after the publisher has committed its
    /// baseline; a consumer may reenter the producer.
    pub publish: Arc<dyn Fn(TUpdate) + Send + Sync>,
    /// Receives failures from the timer-driven flush.
    pub on_error: Arc<dyn Fn(String) + Send + Sync>,
    /// The floor on publication spacing, in milliseconds. Defaults to 100.
    pub min_interval_ms: Option<u64>,
    /// The encoded-bytes budget per second the delay derives from.
    /// Defaults to 100 * 1024.
    pub target_bytes_per_second: Option<u64>,
}

impl<TValue, TUpdate> std::fmt::Debug for AdaptivePublisherOptions<TValue, TUpdate> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptivePublisherOptions").finish_non_exhaustive()
    }
}

struct PublisherState<TValue> {
    published: Option<TValue>,
    dirty: bool,
    next_emit_at: Instant,
    disposed: bool,
}

/// The shared publisher core; the [`AdaptivePublisher`] handle clones it
/// so the timer task and the producer drive the same state.
struct PublisherCore<TValue, TUpdate> {
    options: AdaptivePublisherOptions<TValue, TUpdate>,
    min_interval_ms: u64,
    target_bytes_per_second: u64,
    state: Arc<Mutex<PublisherState<TValue>>>,
    timer: Arc<Mutex<Option<JoinHandle<()>>>>,
}

/// Publishes the latest state without queuing intermediate mutations,
/// upstream's `AdaptivePublisher<TValue, TUpdate>`.
pub struct AdaptivePublisher<TValue, TUpdate> {
    core: Arc<PublisherCore<TValue, TUpdate>>,
}

impl<TValue, TUpdate> std::fmt::Debug for AdaptivePublisher<TValue, TUpdate> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptivePublisher").finish_non_exhaustive()
    }
}

impl<TValue, TUpdate> AdaptivePublisher<TValue, TUpdate> {
    /// Builds a publisher over the supplied callbacks.
    #[must_use]
    pub fn new(options: AdaptivePublisherOptions<TValue, TUpdate>) -> Self {
        Self {
            core: Arc::new(PublisherCore {
                min_interval_ms: options.min_interval_ms.unwrap_or(100),
                target_bytes_per_second: options.target_bytes_per_second.unwrap_or(100 * 1024),
                state: Arc::new(Mutex::new(PublisherState {
                    published: None,
                    dirty: false,
                    next_emit_at: Instant::now(),
                    disposed: false,
                })),
                timer: Arc::new(Mutex::new(None)),
                options,
            }),
        }
    }

    /// Flags the state changed and publishes immediately when the rate
    /// limit allows, arming the trailing timer otherwise.
    pub fn mark_dirty(&self) {
        let wait = {
            let mut state = self.core.lock_state();
            if state.disposed {
                return;
            }
            state.dirty = true;
            let now = Instant::now();
            if state.next_emit_at <= now {
                None
            } else {
                Some(state.next_emit_at.duration_since(now))
            }
        };
        match wait {
            None => self.core.flush(false),
            Some(wait) => self.core.arm_timer(wait),
        }
    }

    /// Publishes the dirty state when the rate limit allows; `force`
    /// publishes regardless. A non-forceful publication re-arms the
    /// trailing timer.
    pub fn flush(&self, force: bool) {
        self.core.flush(force);
    }

    /// Stops the publisher and cancels any armed timer.
    pub fn dispose(&self) {
        self.core.clear_timer();
        self.core.lock_state().disposed = true;
    }
}

impl<TValue, TUpdate> PublisherCore<TValue, TUpdate> {
    fn lock_state(&self) -> std::sync::MutexGuard<'_, PublisherState<TValue>> {
        self.state.lock().expect("publisher state lock")
    }

    fn clear_timer(&self) {
        if let Some(handle) = self.timer.lock().expect("publisher timer lock").take() {
            handle.abort();
        }
    }

    fn arm_timer(self: &Arc<Self>, wait: Duration) {
        let mut timer = self.timer.lock().expect("publisher timer lock");
        if timer.is_some() {
            return;
        }
        // A fired-but-stale timer is harmless: the flush re-checks the
        // rate limit before publishing.
        let core = Arc::clone(self);
        let handle = tokio::spawn(async move {
            tokio::time::sleep(wait).await;
            core.flush(false);
        });
        *timer = Some(handle);
    }

    fn flush(&self, force: bool) {
        let update = {
            let mut state = self.lock_state();
            if state.disposed || !state.dirty {
                return;
            }
            let now = Instant::now();
            if !force && state.next_emit_at > now {
                let wait = state.next_emit_at.duration_since(now);
                drop(state);
                self.arm_timer(wait);
                return;
            }
            self.clear_timer();
            let current = (self.options.snapshot)();
            let update = (self.options.update)(state.published.as_ref(), &current);
            if let Some(update) = update {
                let encoded_bytes = (self.options.measure)(&update);
                state.published = Some(current);
                state.dirty = false;
                state.next_emit_at = now
                    + Duration::from_millis(self.min_interval_ms.max(encoded_emit_delay_ms(
                        encoded_bytes,
                        self.target_bytes_per_second,
                    )));
                // Commit before delivery: a consumer may reenter the
                // producer, and retaining the old baseline would duplicate
                // that delta.
                Some(update)
            } else {
                state.published = Some(current);
                state.dirty = false;
                None
            }
        };
        if let Some(update) = update {
            (self.options.publish)(update);
        }
    }
}

/// The publication delay one encoded update buys, upstream's
/// `(encodedBytes * 1000) / targetBytesPerSecond`.
fn encoded_emit_delay_ms(encoded_bytes: u64, target_bytes_per_second: u64) -> u64 {
    encoded_bytes.saturating_mul(1000) / target_bytes_per_second.max(1)
}

#[cfg(test)]
mod tests;
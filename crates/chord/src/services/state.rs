//! Replicated state, ported from upstream `src/services/state.ts` and
//! `state-internals.ts`.
//!
//! A [`MutableReplicatedState`] owns a tracked value: producers mutate it
//! through the tracker ([`MutableReplicatedState::mutate`], the Proxy-write
//! restatement) and publish flushed operation batches; a
//! [`ReplicatedStateReplica`] is the cold read-only state consumers hold
//! until a complete snapshot arrives. Upstream registers state internals in
//! a `WeakMap` so the provider can discover them on a member object; the
//! handle type itself carries those roles here, so the registry disappears.

use std::panic::AssertUnwindSafe;
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard};

use crate::context::{Context, background_context};
use crate::delta::{Op, Tracker, apply_immutable, is_base};
use crate::errors::ChordError;
use crate::handle::{ErrorReporter, no_error_reporter};
use crate::types::{JsonValue, ReplicatedStateDelivery, ReplicatedStateDeliveryKind, Unsubscribe};

/// The context for synthetic service deliveries without a caller.
#[must_use]
pub fn service_delivery_context() -> Context {
    background_context()
}

/// A value-delivery listener, upstream's `(value, context, delivery) => void`.
pub type ValueListener = Rc<dyn Fn(&JsonValue, &Context, &ReplicatedStateDelivery)>;
/// An operation-batch listener; the provider's replication plumbing
/// publishes through it, and a failure propagates out of
/// [`MutableReplicatedState::publish`].
pub type SourceListener = Rc<dyn Fn(&[Op], u64, &Context) -> Result<(), ChordError>>;

/// Listeners keyed by a stable id so an unsubscribe handle removes exactly
/// its own registration, upstream's `Set` plus `() => set.delete(fn)`.
struct ListenerList<T> {
    next_id: u64,
    entries: Vec<(u64, T)>,
}

impl<T> ListenerList<T> {
    const fn new() -> Self {
        Self {
            next_id: 1,
            entries: Vec::new(),
        }
    }

    fn add(&mut self, listener: T) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push((id, listener));
        id
    }

    fn remove(&mut self, id: u64) {
        self.entries.retain(|(stored, _)| *stored != id);
    }
}

/// Locks one of the state mutexes, panicking with the lock's label when the
/// lock is poisoned.
#[allow(
    clippy::expect_used,
    reason = "a poisoned state lock panics with its label, the port's single-threaded contract; recovering differently would change the observable panic"
)]
fn lock_state<'a, T>(mutex: &'a Mutex<T>, label: &'static str) -> MutexGuard<'a, T> {
    mutex.lock().expect(label)
}

struct MutableStateInner {
    tracker: Tracker,
    published: JsonValue,
    sequence: u64,
    listeners: ListenerList<ValueListener>,
    source_listeners: ListenerList<SourceListener>,
}

/// Initialized mutable state a producer exposes through a service implementation.
///
/// Reads see the last published value, writes go through the tracked
/// surface, and [`publish`](Self::publish) emits the flushed operation
/// batch to every consumer.
#[derive(Clone)]
pub struct MutableReplicatedState(Rc<Mutex<MutableStateInner>>);

impl std::fmt::Debug for MutableReplicatedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = lock_state(&self.0, "state lock");
        f.debug_struct("MutableReplicatedState")
            .field("sequence", &inner.sequence)
            .field("published", &inner.published)
            .finish()
    }
}

impl MutableReplicatedState {
    /// Tracks `initial`; the first publication is a base batch.
    ///
    /// # Panics
    /// Only if the tracker's own base flush fails to apply, which its
    /// emitted vocabulary cannot make happen; a panic here is a port bug,
    /// not a runtime condition.
    #[must_use]
    pub fn new(initial: JsonValue) -> Self {
        let mut tracker = Tracker::new(initial);
        #[allow(
            clippy::expect_used,
            reason = "the tracker's own base flush applies to an empty value by construction; a failure is a port bug"
        )]
        let published = apply_immutable(None, &tracker.flush())
            .expect("the tracker's base flush applies to an empty value");
        Self(Rc::new(Mutex::new(MutableStateInner {
            tracker,
            published,
            sequence: 0,
            listeners: ListenerList::new(),
            source_listeners: ListenerList::new(),
        })))
    }

    /// The last published value. Later publications do not mutate a value a
    /// listener previously received, because each publication owns a fresh
    /// tree.
    ///
    /// # Panics
    /// Never in the single-threaded runtime chord runs on: the state lock
    /// is uncontended.
    #[must_use]
    pub fn value(&self) -> JsonValue {
        lock_state(&self.0, "state lock").published.clone()
    }

    /// The publication sequence, `0` before the first publication.
    #[must_use]
    pub fn sequence(&self) -> u64 {
        lock_state(&self.0, "state lock").sequence
    }

    /// Accesses the tracked surface, the `state.` proxy writes upstream
    /// producers make: every mutation records an intent, and the next
    /// [`publish`](Self::publish) flushes it.
    pub fn mutate<R>(&self, mutate: impl FnOnce(&mut Tracker) -> R) -> R {
        let mut inner = lock_state(&self.0, "state lock");
        mutate(&mut inner.tracker)
    }

    /// Emits the operation batch the mutations since the last publication
    /// produce, and delivers the new value. A flush with no operations is a
    /// no-op, upstream's `ops.length === 0` early return. The source
    /// listeners (the provider's replication plumbing) hear the batch
    /// before the value listeners, upstream's delivery order, and both run
    /// with the state lock released so listener bodies may read state.
    ///
    /// # Errors
    /// [`ChordError`] when a source listener rejects the flushed batch,
    /// propagated before any value listener runs.
    ///
    /// # Panics
    /// Only if the tracker's own flush fails to apply, which its emitted
    /// vocabulary cannot make happen; a panic here is a port bug, not a
    /// runtime condition.
    pub fn publish(&self, context: &Context) -> Result<(), ChordError> {
        let (ops, sequence, published, sources, listeners) = {
            let mut inner = lock_state(&self.0, "state lock");
            let ops = inner.tracker.flush();
            if ops.is_empty() {
                return Ok(());
            }
            inner.sequence += 1;
            #[allow(
                clippy::expect_used,
                reason = "tracker-emitted ops apply to their own baseline by construction; a failure is a port bug"
            )]
            let published = apply_immutable(Some(inner.published.clone()), &ops)
                .expect("tracker-emitted ops apply to their own baseline");
            inner.published = published.clone();
            (
                ops,
                inner.sequence,
                published,
                inner
                    .source_listeners
                    .entries
                    .iter()
                    .map(|(_, l)| l.clone())
                    .collect::<Vec<SourceListener>>(),
                inner
                    .listeners
                    .entries
                    .iter()
                    .map(|(_, l)| l.clone())
                    .collect::<Vec<ValueListener>>(),
            )
        };
        for source in &sources {
            source(&ops, sequence, context)?;
        }
        let delivery = ReplicatedStateDelivery {
            kind: ReplicatedStateDeliveryKind::Update,
            sequence,
        };
        for listener in &listeners {
            listener(&published, context, &delivery);
        }
        Ok(())
    }

    /// Subscribes to value deliveries. Pending mutations flush first —
    /// upstream publishes before registering — and the listener then
    /// receives one hydrate delivery with the current value.
    ///
    /// # Errors
    /// [`ChordError`] when the flush-before-hydrate publication fails.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the listener is registered and then invoked for the hydrate delivery; the by-value signature is the ported public surface"
    )]
    pub fn subscribe(&self, listener: ValueListener) -> Result<Unsubscribe, ChordError> {
        let context = service_delivery_context();
        self.publish(&context)?;
        let mut inner = lock_state(&self.0, "state lock");
        let id = inner.listeners.add(listener.clone());
        let delivery = ReplicatedStateDelivery {
            kind: ReplicatedStateDeliveryKind::Hydrate,
            sequence: inner.sequence,
        };
        listener(&inner.published, &context, &delivery);
        drop(inner);
        let state = self.clone();
        Ok(Box::new(move || {
            lock_state(&state.0, "state lock").listeners.remove(id);
        }))
    }

    /// Registers an operation-batch listener; the provider's replication
    /// plumbing publishes through it.
    #[must_use]
    pub(crate) fn source_subscribe(&self, listener: SourceListener) -> Unsubscribe {
        let id = lock_state(&self.0, "state lock")
            .source_listeners
            .add(listener);
        let state = self.clone();
        Box::new(move || {
            lock_state(&state.0, "state lock")
                .source_listeners
                .remove(id);
        })
    }
}

struct ReplicaInner {
    value: Option<JsonValue>,
    sequence: Option<u64>,
    listeners: ListenerList<ValueListener>,
    report_error: ErrorReporter,
}

/// A cold read-only state used by service consumers until a complete snapshot arrives, upstream's `ReplicatedStateReplica`.
///
/// Sequence fencing is strict: an update that skips a sequence clears the
/// replica before the error propagates. Listener failures are reported to
/// the constructor's reporter, the try/catch shape upstream wraps every
/// delivery in.
#[derive(Clone)]
pub struct ReplicatedStateReplica(Rc<Mutex<ReplicaInner>>);

impl std::fmt::Debug for ReplicatedStateReplica {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = lock_state(&self.0, "replica lock");
        f.debug_struct("ReplicatedStateReplica")
            .field("sequence", &inner.sequence)
            .field("value", &inner.value)
            .finish()
    }
}

impl ReplicatedStateReplica {
    /// A replica that reports listener failures to `report_error`.
    #[must_use]
    pub fn new(report_error: ErrorReporter) -> Self {
        Self(Rc::new(Mutex::new(ReplicaInner {
            value: None,
            sequence: None,
            listeners: ListenerList::new(),
            report_error,
        })))
    }

    /// The current value, or [`None`] until hydration; a cleared replica
    /// reports [`None`] again.
    #[must_use]
    pub fn value(&self) -> Option<JsonValue> {
        lock_state(&self.0, "replica lock").value.clone()
    }

    /// Subscribes; an already-hydrated replica delivers one hydrate
    /// immediately.
    #[must_use]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "the listener is registered and then invoked for a possible hydrate delivery; the by-value signature is the ported public surface"
    )]
    pub fn subscribe(&self, listener: ValueListener) -> Unsubscribe {
        let mut inner = lock_state(&self.0, "replica lock");
        let id = inner.listeners.add(listener.clone());
        if let Some(value) = inner.value.clone()
            && let Some(sequence) = inner.sequence
        {
            let delivery = ReplicatedStateDelivery {
                kind: ReplicatedStateDeliveryKind::Hydrate,
                sequence,
            };
            listener(&value, &service_delivery_context(), &delivery);
        }
        drop(inner);
        let replica = self.clone();
        Box::new(move || {
            lock_state(&replica.0, "replica lock").listeners.remove(id);
        })
    }

    /// Installs the complete initial value from a base batch.
    ///
    /// # Errors
    /// [`ChordError`] when the batch is not a base operation batch.
    pub fn hydrate(&self, sequence: u64, ops: &[Op], context: &Context) -> Result<(), ChordError> {
        if !is_base(ops) {
            return Err(ChordError::Message(
                "Replicated state snapshot is not a base operation batch".to_string(),
            ));
        }
        let value = apply_immutable(None, ops)?;
        let mut inner = lock_state(&self.0, "replica lock");
        inner.sequence = Some(sequence);
        inner.value = Some(value);
        let delivery = ReplicatedStateDelivery {
            kind: ReplicatedStateDeliveryKind::Hydrate,
            sequence,
        };
        deliver_all(&inner, context, &delivery);
        drop(inner);
        Ok(())
    }

    /// Applies one incremental revision.
    ///
    /// # Errors
    /// [`ChordError`] when an update arrives before hydration, or when the
    /// sequence is not exactly one past the applied one — the gap clears the
    /// replica before the error surfaces.
    pub fn update(&self, sequence: u64, ops: &[Op], context: &Context) -> Result<(), ChordError> {
        let mut inner = lock_state(&self.0, "replica lock");
        let (Some(applied_sequence), Some(applied_value)) = (inner.sequence, inner.value.clone())
        else {
            return Err(ChordError::Message(
                "Replicated state received an update before hydration".to_string(),
            ));
        };
        if sequence != applied_sequence + 1 {
            inner.value = None;
            inner.sequence = None;
            return Err(ChordError::Message(
                "Replicated state update sequence has a gap".to_string(),
            ));
        }
        let value = apply_immutable(Some(applied_value), ops)?;
        inner.sequence = Some(sequence);
        inner.value = Some(value);
        let delivery = ReplicatedStateDelivery {
            kind: ReplicatedStateDeliveryKind::Update,
            sequence,
        };
        deliver_all(&inner, context, &delivery);
        drop(inner);
        Ok(())
    }

    /// Drops the retained value and sequence, the cold state a consumer
    /// restarts from.
    pub fn clear(&self) {
        let mut inner = lock_state(&self.0, "replica lock");
        inner.value = None;
        inner.sequence = None;
    }
}

fn deliver_all(inner: &ReplicaInner, context: &Context, delivery: &ReplicatedStateDelivery) {
    let Some(value) = inner.value.clone() else {
        return;
    };
    let listeners: Vec<ValueListener> = inner
        .listeners
        .entries
        .iter()
        .map(|(_, l)| l.clone())
        .collect();
    for listener in listeners {
        let result =
            std::panic::catch_unwind(AssertUnwindSafe(|| listener(&value, context, delivery)));
        if let Err(panic) = result {
            (inner.report_error)(&ChordError::Message(panic_message(&*panic)));
        }
    }
}

/// The text a panic payload carries, upstream's `String(error)` fallback.
#[must_use]
pub fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = panic.downcast_ref::<String>() {
        return text.clone();
    }
    if let Some(text) = panic.downcast_ref::<&str>() {
        return (*text).to_string();
    }
    String::new()
}

/// A reporter that drops everything, the default the consumer plumbing
/// passes when the binding owns none.
#[must_use]
pub fn default_reporter() -> ErrorReporter {
    no_error_reporter()
}

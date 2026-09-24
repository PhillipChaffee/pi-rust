//! The passive harness event bus with isolated handler failures, ported
//! from upstream `src/harness/events.ts`.
//!
//! Delivery serializes through one ordered tail: `emit_batch` binds the
//! recipients at emit time, clones each event per listener, and appends
//! one contiguous batch to the tail; each emitted batch resolves when its
//! delivery completes. Handler failures isolate: the listener's failure
//! converts into a `handler_error` event delivered to the remaining
//! recipients without reporting further failures. Upstream's promise-chain
//! tail restates as a drain-now task over a job queue — one drainer at a
//! time, so batches and barriers keep their order.
//!
//! Event listeners report failures through their `Result` (upstream's
//! handler throw); bodies that cannot fail return `Ok(())`. The watcher
//! keeps its own delivery tail separate from the bus tail: a watcher
//! listener may await a bus-tail barrier (the resnapshot boundary) without
//! deadlocking, mirroring upstream's two independent promise chains.

use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use pi_ai::types::BoxedFuture;
use tokio::sync::oneshot;

use crate::harness::agent_harness::{
    EventListener, HandlerErrorKind, HarnessEvent, HarnessEventPayload, HarnessEventType,
    Subscription, WatchHandle,
};
use crate::harness::context::Context;

/// The listener failure type, upstream's handler throw.
pub type ListenerError = Box<dyn std::error::Error + Send + Sync>;

/// The event filter a watch subscribes with, upstream's
/// `filter: (event) => boolean`.
pub type WatchFilter = Arc<dyn Fn(&HarnessEvent) -> bool + Send + Sync>;

/// One watcher's resnapshot capture, upstream's `ResnapshotCapture<T>`:
/// computes the next snapshot, marking the boundary exactly once.
pub type ResnapshotCapture<T> = Arc<
    dyn for<'a> Fn(&'a Context, ResnapshotBoundary) -> BoxedFuture<'a, Result<T, ListenerError>>
        + Send
        + Sync,
>;

/// The boundary marker a resnapshot capture must call exactly once,
/// upstream's `markBoundary()`: enqueues the delivery-tail barrier that
/// flips the watcher from dropping to holding.
#[derive(Clone)]
pub struct ResnapshotBoundary {
    mark: Arc<dyn Fn() + Send + Sync>,
}

impl std::fmt::Debug for ResnapshotBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ResnapshotBoundary(..)")
    }
}

impl ResnapshotBoundary {
    /// Marks the delivery-tail barrier.
    pub fn mark(&self) {
        (self.mark)();
    }
}

type BusListener = Arc<
    dyn for<'a> Fn(&'a HarnessEvent, &'a Context) -> BoxedFuture<'a, Result<(), ListenerError>>
        + Send
        + Sync,
>;

/// The job types the ordered tail carries.
enum BusJob {
    /// One contiguous batch: bound events, bound recipients, and the
    /// completion signal.
    Deliver {
        batch: Vec<(HarnessEvent, Context)>,
        recipients: Vec<BusListener>,
        done: oneshot::Sender<()>,
    },
    /// A closure that awaits at its position in the tail — the resnapshot
    /// boundary's vehicle.
    Barrier(Box<dyn FnOnce() -> BoxedFuture<'static, ()> + Send>),
}

struct BusCore {
    listeners: HashMap<HarnessEventType, Vec<BusListener>>,
    watch_listeners: Vec<BusListener>,
    closed_error: Option<String>,
    queue: VecDeque<BusJob>,
    draining: bool,
}

impl BusCore {
    /// Pushes one job and spawns one drain cycle when none is running.
    fn push_job(core: &Arc<Mutex<BusCore>>, job: BusJob) {
        {
            let mut core = core.lock().expect("bus core lock");
            core.queue.push_back(job);
        }
        BusCore::ensure_draining(core);
    }

    fn ensure_draining(core: &Arc<Mutex<BusCore>>) {
        let spawn = {
            let mut core = core.lock().expect("bus core lock");
            if core.draining {
                false
            } else {
                core.draining = true;
                true
            }
        };
        if spawn {
            let core = Arc::clone(core);
            tokio::spawn(async move {
                loop {
                    let job = {
                        let mut core = core.lock().expect("bus core lock");
                        match core.queue.pop_front() {
                            Some(job) => job,
                            None => {
                                core.draining = false;
                                return;
                            }
                        }
                    };
                    match job {
                        BusJob::Deliver {
                            batch,
                            recipients,
                            done,
                        } => {
                            for (event, context) in &batch {
                                deliver(&core, event, &recipients, true, context).await;
                            }
                            let _ = done.send(());
                        }
                        BusJob::Barrier(barrier) => barrier().await,
                    }
                }
            });
        }
    }
}

/// Passive harness event bus, upstream's `HarnessEventBus`.
pub struct HarnessEventBus {
    core: Arc<Mutex<BusCore>>,
}

impl std::fmt::Debug for HarnessEventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HarnessEventBus").finish_non_exhaustive()
    }
}

impl Default for HarnessEventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl HarnessEventBus {
    /// Builds an empty bus.
    #[must_use]
    pub fn new() -> Self {
        Self {
            core: Arc::new(Mutex::new(BusCore {
                listeners: HashMap::new(),
                watch_listeners: Vec::new(),
                closed_error: None,
                queue: VecDeque::new(),
                draining: false,
            })),
        }
    }

    fn closed_error(&self) -> Option<String> {
        self.core.lock().expect("bus core lock").closed_error.clone()
    }

    /// Emits one event, upstream's `emit`.
    pub fn emit(&self, event: HarnessEvent, context: &Context) -> BoxedFuture<'_, ()> {
        self.emit_batch(vec![(event, context.clone())])
    }

    /// Binds current recipients and appends one contiguous batch to the
    /// global delivery tail, upstream's `emitBatch`.
    ///
    /// A closed bus or an empty batch resolves immediately.
    pub fn emit_batch(&self, events: Vec<(HarnessEvent, Context)>) -> BoxedFuture<'_, ()> {
        if self.closed_error().is_some() || events.is_empty() {
            return Box::pin(async {});
        }
        let (done, receiver) = oneshot::channel();
        let recipients = self.snapshot_recipients_union(&events);
        BusCore::push_job(
            &self.core,
            BusJob::Deliver {
                batch: events,
                recipients,
                done,
            },
        );
        Box::pin(async move {
            let _ = receiver.await;
        })
    }

    fn snapshot_recipients_union(&self, events: &[(HarnessEvent, Context)]) -> Vec<BusListener> {
        let core = self.core.lock().expect("bus core lock");
        let mut union: Vec<BusListener> = Vec::new();
        for (event, _) in events {
            if let Some(list) = core.listeners.get(&event.payload.event_type()) {
                for listener in list {
                    if !union.iter().any(|registered| Arc::ptr_eq(registered, listener)) {
                        union.push(Arc::clone(listener));
                    }
                }
            }
        }
        for listener in &core.watch_listeners {
            if !union.iter().any(|registered| Arc::ptr_eq(registered, listener)) {
                union.push(Arc::clone(listener));
            }
        }
        union
    }

    /// Subscribes one listener to an event type, upstream's `on`.
    ///
    /// # Errors
    /// The close error when the bus has closed.
    pub fn on(
        &self,
        event_type: HarnessEventType,
        listener: EventListener,
    ) -> Result<Subscription, String> {
        if let Some(closed_error) = self.closed_error() {
            return Err(closed_error);
        }
        let wrapped: BusListener = {
            let listener = Arc::clone(&listener);
            Arc::new(move |event, context| {
                let listener = Arc::clone(&listener);
                Box::pin(async move { listener(event, context).await })
            })
        };
        let mut core = self.core.lock().expect("bus core lock");
        core.listeners
            .entry(event_type)
            .or_default()
            .push(Arc::clone(&wrapped));
        let unsubscribe_core = Arc::clone(&self.core);
        Ok(Subscription::new(Arc::new(move || {
            let mut core = unsubscribe_core.lock().expect("bus core lock");
            if let Some(list) = core.listeners.get_mut(&event_type) {
                list.retain(|registered| !Arc::ptr_eq(registered, &wrapped));
            }
        })))
    }

    /// Watches with one initial snapshot, upstream's `watch`.
    ///
    /// # Errors
    /// The close error when the bus has closed.
    pub fn watch<T: Send + Sync + 'static>(
        &self,
        snapshot: T,
        filter: WatchFilter,
        resnapshot: Option<ResnapshotCapture<T>>,
    ) -> Result<Arc<BufferedEventWatcher<T>>, String> {
        if let Some(closed_error) = self.closed_error() {
            return Err(closed_error);
        }
        Ok(self.install_watcher(Some(snapshot), filter, resnapshot))
    }

    /// Watches with a snapshot captured ahead of the subscription, upstream's
    /// `watchFromSnapshot`.
    ///
    /// # Errors
    /// The close error when the bus has closed, or the capture's failure;
    /// a failed capture unsubscribes the installed watcher.
    pub async fn watch_from_snapshot<T, F, Fut, TError>(
        &self,
        capture: F,
        filter: WatchFilter,
        context: &Context,
    ) -> Result<Arc<BufferedEventWatcher<T>>, String>
    where
        T: Send + Sync + 'static + Clone,
        F: Fn(&Context) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T, TError>> + Send,
        TError: Into<ListenerError> + std::error::Error + Send + Sync + 'static,
    {
        if let Some(closed_error) = self.closed_error() {
            return Err(closed_error);
        }
        let capture = Arc::new(capture);
        let resnapshot: ResnapshotCapture<T> = {
            let capture = Arc::clone(&capture);
            Arc::new(move |capture_context, boundary| {
                let capture = Arc::clone(&capture);
                Box::pin(async move {
                    let snapshot = capture(capture_context).await?;
                    boundary.mark();
                    Ok(snapshot)
                })
            })
        };
        let watcher = self.install_watcher::<T>(None, filter, Some(resnapshot));
        match capture(context).await {
            Ok(snapshot) => {
                watcher.set_snapshot(snapshot);
                Ok(watcher)
            }
            Err(error) => {
                watcher.unsubscribe();
                Err(error.into().to_string())
            }
        }
    }

    /// Closes the bus; the first error wins and listeners clear after the
    /// delivery tail settles, upstream's `close`.
    pub fn close(&self, error: String) {
        let already_closed = {
            let mut core = self.core.lock().expect("bus core lock");
            let already_closed = core.closed_error.is_some();
            if !already_closed {
                core.closed_error = Some(error);
            }
            already_closed
        };
        if already_closed {
            return;
        }
        BusCore::push_job(
            &self.core,
            BusJob::Barrier(Box::new({
                let core = Arc::clone(&self.core);
                move || {
                    Box::pin(async move {
                        let mut core = core.lock().expect("bus core lock");
                        core.listeners.clear();
                        core.watch_listeners.clear();
                    })
                }
            })),
        );
    }

    fn install_watcher<T: Send + Sync + 'static>(
        &self,
        snapshot: Option<T>,
        filter: WatchFilter,
        resnapshot: Option<ResnapshotCapture<T>>,
    ) -> Arc<BufferedEventWatcher<T>> {
        let on_error: Arc<dyn Fn(String, HarnessEvent, Context) + Send + Sync> = {
            let bus = Arc::downgrade(&self.core);
            Arc::new(move |error: String, event: HarnessEvent, context: Context| {
                if event.payload.event_type() == HarnessEventType::HandlerError {
                    return;
                }
                let Some(core) = bus.upgrade() else {
                    return;
                };
                let payload = HarnessEventPayload::HandlerError {
                    error,
                    stack: None,
                    kind: HandlerErrorKind::Event {
                        event: event.payload.event_type().as_str().to_owned(),
                    },
                };
                let emitted = HarnessEvent {
                    lane: event.lane.clone(),
                    recovery: false,
                    payload,
                };
                // The bus holds its delivery tail synchronously; the
                // dropped future is only a completion signal.
                let _ = HarnessEventBus { core }.emit(emitted, &context);
            })
        };
        let watcher = Arc::new(BufferedEventWatcher {
            shared: Arc::new(WatcherShared {
                state: Mutex::new(WatcherState {
                    phase: WatcherPhase::Buffering,
                    snapshot,
                    buffer: VecDeque::new(),
                    epoch: 0,
                    listener: None,
                    resnapshot_state: None,
                    resnapshot_callback: resnapshot,
                    on_error,
                    unsubscribe: None,
                }),
                queue: Mutex::new(VecDeque::new()),
                draining: AtomicBool::new(false),
            }),
            filter,
            bus: Arc::downgrade(&self.core),
            self_weak: OnceLockWeak::new(),
            _snapshot: PhantomData,
        });
        let watch_listener: BusListener = {
            let watcher = Arc::clone(&watcher);
            Arc::new(move |event, context| {
                let watcher = Arc::clone(&watcher);
                Box::pin(async move {
                    if (watcher.filter)(event) {
                        watcher.push(event.clone(), context.clone());
                    }
                    Ok(())
                })
            })
        };
        let unsubscribe_core = Arc::clone(&self.core);
        let unsubscribe_token = Arc::clone(&watch_listener);
        watcher.set_unsubscribe(Arc::new(move || {
            let mut core = unsubscribe_core.lock().expect("bus core lock");
            core.watch_listeners
                .retain(|registered| !Arc::ptr_eq(registered, &unsubscribe_token));
        }));
        self.core
            .lock()
            .expect("bus core lock")
            .watch_listeners
            .push(watch_listener);
        watcher.set_self_weak(Arc::downgrade(&watcher));
        watcher
    }
}

/// Delivers one event to its recipients with isolated failures, upstream's
/// `deliver`.
async fn deliver(
    core: &Arc<Mutex<BusCore>>,
    event: &HarnessEvent,
    recipients: &[BusListener],
    report_errors: bool,
    context: &Context,
) {
    for listener in recipients {
        let event_clone = event.clone();
        match listener(&event_clone, context).await {
            Ok(()) => {}
            Err(error) => {
                if !report_errors || event.payload.event_type() == HarnessEventType::HandlerError {
                    continue;
                }
                let payload = HarnessEventPayload::HandlerError {
                    error: error.to_string(),
                    stack: None,
                    kind: HandlerErrorKind::Event {
                        event: event.payload.event_type().as_str().to_owned(),
                    },
                };
                let handler_error = HarnessEvent {
                    lane: event.lane.clone(),
                    recovery: false,
                    payload,
                };
                // Recipients bind at failure time to the handler_error
                // event's own audience; their failures are not reported
                // further, and the recursion boxes through one level so
                // the future stays finite.
                let handler_recipients = snapshot_recipients(core, &handler_error);
                Box::pin(deliver(core, &handler_error, &handler_recipients, false, context)).await;
            }
        }
    }
}

/// The listeners one delivered event binds, upstream's
/// `snapshotRecipients`.
fn snapshot_recipients(core: &Mutex<BusCore>, event: &HarnessEvent) -> Vec<BusListener> {
    let bound = core.lock().expect("bus core lock");
    let mut union: Vec<BusListener> = Vec::new();
    if let Some(list) = bound.listeners.get(&event.payload.event_type()) {
        for listener in list {
            union.push(Arc::clone(listener));
        }
    }
    for listener in &bound.watch_listeners {
        union.push(Arc::clone(listener));
    }
    union
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WatcherPhase {
    Buffering,
    Started,
    Unsubscribed,
}

struct ResnapshotHold {
    resolve: Option<oneshot::Sender<()>>,
    held: VecDeque<(HarnessEvent, Context)>,
    phase: ResnapshotPhase,
}

/// The resnapshot hold phases, upstream's `"dropping" | "holding"`: drops
/// arrive before the boundary, holds after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResnapshotPhase {
    Dropping,
    Holding,
}

struct WatcherState<T> {
    phase: WatcherPhase,
    snapshot: Option<T>,
    buffer: VecDeque<(HarnessEvent, Context, u64)>,
    epoch: u64,
    listener: Option<EventListener>,
    resnapshot_state: Option<ResnapshotHold>,
    resnapshot_callback: Option<ResnapshotCapture<T>>,
    on_error: Arc<dyn Fn(String, HarnessEvent, Context) + Send + Sync>,
    unsubscribe: Option<Arc<dyn Fn() + Send + Sync>>,
}

/// The watcher's shared delivery machinery: state under one lock, the
/// delivery queue under another, and the drain flag.
struct WatcherShared<T> {
    state: Mutex<WatcherState<T>>,
    queue: Mutex<VecDeque<(HarnessEvent, Context, u64)>>,
    draining: AtomicBool,
}

/// The buffered event watcher, upstream's `BufferedEventWatcher<T>`.
pub struct BufferedEventWatcher<T> {
    shared: Arc<WatcherShared<T>>,
    filter: WatchFilter,
    bus: Weak<Mutex<BusCore>>,
    self_weak: OnceLockWeak<BufferedEventWatcher<T>>,
    _snapshot: PhantomData<fn(T) -> T>,
}

impl<T> std::fmt::Debug for BufferedEventWatcher<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferedEventWatcher").finish_non_exhaustive()
    }
}

impl<T> Clone for BufferedEventWatcher<T> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            filter: Arc::clone(&self.filter),
            bus: self.bus.clone(),
            self_weak: self.self_weak.clone(),
            _snapshot: PhantomData,
        }
    }
}

/// A cloneable weak self reference; the watcher installs itself after the
/// bus holds its subscription, so boundary barriers can reach it.
struct OnceLockWeak<T> {
    cell: Mutex<Option<Weak<T>>>,
}

impl<T> Clone for OnceLockWeak<T> {
    fn clone(&self) -> Self {
        Self {
            cell: Mutex::new(self.cell.lock().expect("self weak lock").clone()),
        }
    }
}

impl<T> OnceLockWeak<T> {
    fn new() -> Self {
        Self {
            cell: Mutex::new(None),
        }
    }

    fn set(&self, weak: Weak<T>) {
        *self.cell.lock().expect("self weak lock") = Some(weak);
    }

    fn upgrade(&self) -> Option<Arc<T>> {
        self.cell
            .lock()
            .expect("self weak lock")
            .as_ref()
            .and_then(Weak::upgrade)
    }
}

impl<T: Send + Sync + 'static> BufferedEventWatcher<T> {
    /// Replaces the snapshot, upstream's `setSnapshot`.
    pub fn set_snapshot(&self, snapshot: T) {
        self.shared
            .state
            .lock()
            .expect("watcher state lock")
            .snapshot = Some(snapshot);
    }

    /// Starts delivering buffered events to the listener; callable once,
    /// upstream's `start`.
    ///
    /// # Panics
    /// When called twice, upstream's "may be called only once" guard.
    pub fn start(&self, listener: EventListener) {
        let buffered = {
            let mut state = self.shared.state.lock().expect("watcher state lock");
            assert!(
                matches!(state.phase, WatcherPhase::Buffering),
                "WatchHandle.start() may be called only once"
            );
            state.phase = WatcherPhase::Started;
            state.listener = Some(listener);
            std::mem::take(&mut state.buffer)
        };
        for (event, context, epoch) in buffered {
            self.enqueue(event, context, epoch);
        }
    }

    /// Unsubscribes the watcher, upstream's `unsubscribe`.
    pub fn unsubscribe(&self) {
        let unsubscribe = {
            let mut state = self.shared.state.lock().expect("watcher state lock");
            if state.phase == WatcherPhase::Unsubscribed {
                return;
            }
            state.phase = WatcherPhase::Unsubscribed;
            state.buffer.clear();
            state.listener = None;
            state.unsubscribe.take()
        };
        if let Some(unsubscribe) = unsubscribe {
            unsubscribe();
        }
    }

    /// Pushes one event into the watcher's pipeline, upstream's `push`.
    pub fn push(&self, event: HarnessEvent, context: Context) {
        let action = {
            let mut state = self.shared.state.lock().expect("watcher state lock");
            match state.phase {
                WatcherPhase::Unsubscribed => None,
                _ if state
                    .resnapshot_state
                    .as_ref()
                    .is_some_and(|hold| hold.phase == ResnapshotPhase::Dropping) => None,
                _ if state.resnapshot_state.is_some() => {
                    if let Some(hold) = &mut state.resnapshot_state {
                        hold.held.push_back((event, context));
                    }
                    None
                }
                WatcherPhase::Buffering => {
                    let epoch = state.epoch;
                    state.buffer.push_back((event, context, epoch));
                    None
                }
                WatcherPhase::Started => {
                    let epoch = state.epoch;
                    Some((event, context, epoch))
                }
            }
        };
        if let Some((event, context, epoch)) = action {
            self.enqueue(event, context, epoch);
        }
    }

    fn enqueue(&self, event: HarnessEvent, context: Context, epoch: u64) {
        {
            let mut queue = self.shared.queue.lock().expect("watcher queue lock");
            queue.push_back((event, context, epoch));
        }
        self.ensure_draining();
    }

fn ensure_draining(&self) {
        if self.shared.draining.swap(true, Ordering::AcqRel) {
            return;
        }
        let shared = Arc::clone(&self.shared);
        tokio::spawn(async move {
            loop {
                let job = {
                    let mut queue = shared.queue.lock().expect("watcher queue lock");
                    match queue.pop_front() {
                        Some(job) => job,
                        None => {
                            // Exit and clear the flag under the same lock:
                            // a pusher cannot interleave between the empty
                            // pop and the flag clear.
                            shared.draining.store(false, Ordering::Release);
                            return;
                        }
                    }
                };
                let (event, context, epoch) = job;
                let (listener, on_error) = {
                    let state = shared.state.lock().expect("watcher state lock");
                    if state.phase != WatcherPhase::Started || epoch != state.epoch {
                        continue;
                    }
                    (state.listener.clone(), Arc::clone(&state.on_error))
                };
                let Some(listener) = listener else {
                    continue;
                };
                if let Err(error) = listener(&event, &context).await {
                    on_error(error.to_string(), event, context);
                }
            }
        });
    }

    /// Flips the resnapshot hold from dropping to holding, upstream's
    /// `markResnapshotBoundary`.
    pub fn mark_resnapshot_boundary(&self) {
        let mut state = self.shared.state.lock().expect("watcher state lock");
        if let Some(hold) = state.resnapshot_state.as_mut()
            && hold.phase == ResnapshotPhase::Dropping
        {
            hold.phase = ResnapshotPhase::Holding;
            if let Some(resolve) = hold.resolve.take() {
                let _ = resolve.send(());
            }
        }
    }

fn boundary(&self) -> ResnapshotBoundary {
        let bus = self.bus.clone();
        let self_weak = self.self_weak.clone();
        ResnapshotBoundary {
            mark: Arc::new(move || {
                let Some(bus) = bus.upgrade() else {
                    return;
                };
                let self_weak = self_weak.clone();
                BusCore::push_job(
                    &bus,
                    BusJob::Barrier(Box::new(move || {
                        Box::pin(async move {
                            if let Some(watcher) = self_weak.upgrade() {
                                watcher.mark_resnapshot_boundary();
                            }
                        })
                    })),
                );
            }),
        }
    }

    fn set_self_weak(&self, weak: Weak<Self>) {
        self.self_weak.set(weak);
    }

    fn set_unsubscribe(&self, unsubscribe: Arc<dyn Fn() + Send + Sync>) {
        self.shared
            .state
            .lock()
            .expect("watcher state lock")
            .unsubscribe = Some(unsubscribe);
    }
}

impl<T: Send + Sync + 'static + Clone> BufferedEventWatcher<T> {
/// The current snapshot, upstream's `WatchHandle.snapshot`.
    ///
    /// # Panics
    /// When no snapshot was installed; upstream's typed handle always
    /// carries one.
    #[must_use]
    pub fn snapshot(&self) -> T {
        self.shared
            .state
            .lock()
            .expect("watcher state lock")
            .snapshot
            .clone()
            .expect("watcher snapshot")
    }

    /// Resnapshots the watch, upstream's `resnapshot`.
    ///
    /// # Errors
    /// An unsubscribed watcher, a non-resnapshotting watcher, an
    /// in-progress resnapshot, a capture that never marked its boundary,
    /// or the capture's own failure.
    ///
    /// The snapshot reads need [`T`]'s `Clone`; the mutation surface keeps
    /// the un-`Clone`d bound.
    pub fn resnapshot<'a>(
        &self,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<T, ListenerError>> {
        // The future owns a watcher clone so its only borrow is the
        // context reference the signature elides to.
        let this = self.clone();
        Box::pin(async move {
            let (callback, reached) = {
                let mut state = this.shared.state.lock().expect("watcher state lock");
                if state.phase == WatcherPhase::Unsubscribed {
                    return Err(ListenerError::from("WatchHandle is unsubscribed"));
                }
                let Some(callback) = state.resnapshot_callback.clone() else {
                    return Err(ListenerError::from("WatchHandle does not support resnapshot"));
                };
                if state.resnapshot_state.is_some() {
                    return Err(ListenerError::from(
                        "WatchHandle resnapshot is already in progress",
                    ));
                }
                let (resolve, reached) = oneshot::channel();
                state.epoch += 1;
                state.resnapshot_state = Some(ResnapshotHold {
                    resolve: Some(resolve),
                    held: VecDeque::new(),
                    phase: ResnapshotPhase::Dropping,
                });
                (callback, reached)
            };
            // The capture's mark call is synchronous; the boundary flip
            // rides the bus tail, so "did the capture mark" is its own
            // flag, upstream's local `marked`.
            let marked = Arc::new(AtomicBool::new(false));
            let boundary = {
                let inner = this.boundary();
                let flag = Arc::clone(&marked);
                ResnapshotBoundary {
                    mark: Arc::new(move || {
                        flag.store(true, Ordering::Release);
                        inner.mark();
                    }),
                }
            };
            let snapshot = callback(context, boundary).await;
            if !marked.load(Ordering::Acquire) {
                // The capture never marked its boundary; release the
                // hold and report, upstream's "did not mark" throw.
                let held = {
                    let mut state = this.shared.state.lock().expect("watcher state lock");
                    state
                        .resnapshot_state
                        .take()
                        .map_or_else(VecDeque::new, |hold| hold.held)
                };
                for (event, context) in held {
                    this.push(event, context);
                }
                return Err(ListenerError::from(
                    "Resnapshot capture did not mark its boundary",
                ));
            }
            let _ = reached.await;
            let (snapshot, held) = {
                let mut state = this.shared.state.lock().expect("watcher state lock");
                let held = state
                    .resnapshot_state
                    .take()
                    .map_or_else(VecDeque::new, |hold| hold.held);
                (snapshot, held)
            };
            match snapshot {
                Ok(snapshot) => {
                    this.set_snapshot(snapshot.clone());
                    for (event, context) in held {
                        this.push(event, context);
                    }
                    Ok(snapshot)
                }
                Err(error) => {
                    for (event, context) in held {
                        this.push(event, context);
                    }
                    Err(error)
                }
            }
        })
    }

}

impl<T: Send + Sync + 'static + Clone> WatchHandle<T> for BufferedEventWatcher<T> {
    fn snapshot(&self) -> T {
        BufferedEventWatcher::snapshot(self)
    }

    fn set_snapshot(&self, snapshot: T) {
        BufferedEventWatcher::set_snapshot(self, snapshot);
    }

    fn start(&self, listener: EventListener) {
        BufferedEventWatcher::start(self, listener);
    }

    fn resnapshot<'a>(
        &self,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<T, ListenerError>> {
        BufferedEventWatcher::resnapshot(self, context)
    }

    fn unsubscribe(&self) {
        BufferedEventWatcher::unsubscribe(self);
    }
}

#[cfg(test)]
mod tests;

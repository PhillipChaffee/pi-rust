//! The harness container, ported from upstream
//! `packages/agent/src/harness/runtime/harness.ts` (its one-line
//! `runtime/index.ts` restates as the `create_agent_harness` re-export in
//! `runtime/mod.rs`).
//!
//! [`Harness`] is the container: it owns the session, the models runtime,
//! the hook registry, the event bus, the config store, the fault/closed
//! latches, and the lane map, and it is not itself a lane. The four lane
//! seams the container builds ([`FaultHandler`], [`EmitBatch`],
//! [`WatchInstaller`], [`ConfigProvider`]) close over a strong
//! `Arc<HarnessShared>`: every lane holds its seams and the seams reach the
//! container, so the reference cycle is the port of upstream's closure
//! reachability — the TS closures keep the harness alive the same way — and
//! no `Weak` breaks it.
//!
//! The config store restates upstream's `{ value }` whole-object
//! replacement as a `RwLock<Config>`: each setter's read-modify-write runs
//! under the write lock, so a reader never observes a torn store and the
//! lanes' `read_config` closure stays live across later sets. `close`
//! memoizes its drain the way upstream's `closePromise` does: concurrent
//! closes share one completion channel, and the joined drain (the session
//! close with every lane's idle wait) runs in its own task like the event
//! bus's delivery tail, so no closer's future holds it hostage.
//!
//! Upstream's `seal` is a plain method: the sealed-error latch, the
//! drive-gate close, and the state-change signal all run before the caller
//! continues, and only the idle-owner await defers. The port polls each
//! seal future once (chord's drive-once restatement) so the synchronous
//! prefix runs before the bus closes; `fault` drops the still-pending
//! waits like upstream's discarded seal promises and `close` joins them
//! with the session close.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};

use pi_ai::types::BoxedFuture;
use pi_ai::utils::retry::RetryPolicy;

use crate::harness::agent_harness::{
    AcquireLaneOptions, AgentHarness, AgentHarnessOptions, AgentLane, ConfigUpdateKind, CreateAt,
    EventListener, Events, GlobalConfigUpdate, HandlerErrorKind, HarnessEvent, HarnessEventPayload,
    HarnessEventType, HookFailure, HookHandler, HookName, HookOptions, Hooks, LaneInfo,
    LaneOperationError, LaneSnapshot, OpenOperation, Resources, SessionSnapshot, Subscription,
    ToProviderMessages, ValueUpdateKind, WatchHandle,
};
use crate::harness::compaction::types::{CompactionSettings, DEFAULT_COMPACTION_SETTINGS};
use crate::harness::config::{
    default_retry_policy, validate_compaction_settings, validate_retry_policy, validate_tool_names,
};
use crate::harness::context::Context;
use crate::harness::events::{
    BufferedEventWatcher, HarnessEventBus, ListenerError, ResnapshotCapture, WatchFilter,
};
use crate::harness::hooks::{HookErrorReporter, HookRegistry};
use crate::harness::messages::convert_to_llm;
use crate::harness::result::{HarnessClosed, HarnessError, HarnessFault};
use crate::harness::runtime::lane::{
    ConfigProvider, EmitBatch, FaultHandler, Lane, WatchInstaller,
};
use crate::harness::runtime::restore::{
    ClassifiedLaneStorage, read_lane_storage, restore_lane_state, restore_session,
};
use crate::harness::runtime::types::{
    Config, LaneError, LaneState, SliceNotImplemented, any_payload, from_arc, lane_error,
};
use crate::harness::session::types::{
    Control, LaneConfiguration, ModelIdentity, OperationIntent, OperationKind, OperationMeta,
    Session, SessionError, SessionMutationCallback, SessionStats, operation_scope_of,
};
use crate::harness::session::values::{
    StoredValue, branch_tip, delete_value_write, entry_label, lane_config, lane_state,
    session_name, set_value_write,
};
use crate::harness::types::{AgentHarnessStreamOptions, AgentHarnessTool};
use crate::types::{AgentMessage, QueueMode, ThinkingLevel};

/// The memoized close completion's payload, upstream's `closePromise`
/// settlement: `None` until the drain settles, then the session close's
/// result.
type CloseCompletion = Option<Result<(), LaneError>>;

/// The fault/closed latches, upstream's `faultError`/`closedError`: one
/// lock guards the pair so the check-and-set in [`fault`] and the
/// `assertOpen` reads stay atomic, and each stored error rethrows by `Arc`
/// clone, upstream's error-object identity.
struct Latches {
    /// The latched fault, upstream's `faultError`.
    fault: Option<LaneError>,
    /// The latched close error, upstream's `closedError`.
    closed: Option<LaneError>,
}

/// The harness container's shared state, upstream's `Harness` instance
/// fields: everything the lane seams close over lives here, and the
/// closures hold strong `Arc`s (see the module docs for the cycle).
struct HarnessShared {
    /// The session the harness attaches to, upstream's `session`.
    session: Arc<dyn Session>,
    /// The models runtime, upstream's `models`.
    models: Arc<pi_ai::models::Models>,
    /// The hook registry, upstream's `hooks`.
    hooks: Arc<HookRegistry>,
    /// The event bus, upstream's `events`.
    events: Arc<HarnessEventBus>,
    /// The seed configuration new lanes attach with, upstream's `seed`.
    seed: LaneConfiguration,
    /// The lane map, upstream's `lanesByName`; each entry's `Arc` is the
    /// stable identity every acquisition of the name returns.
    lanes: Mutex<HashMap<String, Arc<Lane>>>,
    /// The config store, upstream's `configStore.value`.
    config_store: RwLock<Config>,
    /// The fault/closed latches, upstream's `faultError`/`closedError`.
    latches: Mutex<Latches>,
    /// The memoized close completion, upstream's `closePromise`.
    close_completion: Mutex<Option<tokio::sync::watch::Receiver<CloseCompletion>>>,
}

/// Runtime implementation of the harness capability, upstream's `Harness`:
/// the container manages lanes but is not itself a lane.
#[derive(Clone)]
pub struct Harness {
    /// The container's shared state; a clone shares it.
    shared: Arc<HarnessShared>,
}

impl std::fmt::Debug for Harness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Harness").finish_non_exhaustive()
    }
}

/// The payload one harness mutation carries back: the deferred delivery
/// batch, or the thrown error the catch block routes (the boxed-Any
/// contract has no error channel, so errors ride the payload like the
/// restore's).
enum MutationOutcome {
    /// The mutation committed or early-returned; the delivery batch awaits
    /// outside the mutation, upstream's `delivery` (`None` when the lane
    /// already existed or nothing published).
    Published {
        /// The deferred delivery batch.
        delivery: Option<BoxedFuture<'static, Result<(), LaneError>>>,
    },
    /// The callback threw, upstream's propagated mutation-callback error.
    Threw(LaneError),
}

/// The boxed-cause adapter the fault carries: the fault handler receives
/// the lane's `Arc`-erased error and [`HarnessFault`] stores `Box`, so the
/// `Arc` wraps through this delegating error and the display and source
/// chain pass through.
#[derive(Debug)]
struct FaultCause(LaneError);

impl std::fmt::Display for FaultCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for FaultCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

/// The erased watch handle the installer boxes, upstream's typed
/// `WatchHandle` the bus mints (`BufferedEventWatcher` implements the
/// contract on the concrete type, so the `Arc` wraps through this
/// delegating adapter).
struct BusWatchHandle {
    /// The bus-backed watcher.
    inner: Arc<BufferedEventWatcher<LaneSnapshot>>,
}

impl WatchHandle<LaneSnapshot> for BusWatchHandle {
    fn snapshot(&self) -> LaneSnapshot {
        WatchHandle::<LaneSnapshot>::snapshot(&*self.inner)
    }

    fn set_snapshot(&self, snapshot: LaneSnapshot) {
        WatchHandle::<LaneSnapshot>::set_snapshot(&*self.inner, snapshot);
    }

    fn start(&self, listener: EventListener) {
        WatchHandle::<LaneSnapshot>::start(&*self.inner, listener);
    }

    fn resnapshot<'a>(
        &self,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<LaneSnapshot, ListenerError>> {
        WatchHandle::<LaneSnapshot>::resnapshot(&*self.inner, context)
    }

    fn unsubscribe(&self) {
        WatchHandle::<LaneSnapshot>::unsubscribe(&*self.inner);
    }
}

/// The created harness with its open-operation view, upstream's
/// `createAgentHarness` return `{ harness, open }`.
#[derive(Debug)]
pub struct CreatedHarness {
    /// The attached runtime, upstream's `harness`.
    pub harness: Harness,
    /// The operations the restore found live, upstream's `open`.
    pub open: Vec<OpenOperation>,
}

/// The creation failure, upstream's `createAgentHarness` throws: the
/// options validation rejections and the restore wrap, a fresh fault per
/// attempt with no latch.
#[derive(Debug)]
pub enum HarnessCreationError {
    /// An options validation rejection, upstream's TypeError/RangeError
    /// throw; the message is the validator's.
    Validation(String),
    /// The storage or invariant fault, upstream's `HarnessFault` wrap of
    /// the restore failure.
    Fault(HarnessFault),
}

impl std::fmt::Display for HarnessCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(message) => f.write_str(message),
            Self::Fault(fault) => f.write_str(&fault.message),
        }
    }
}

impl std::error::Error for HarnessCreationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Validation(_) => None,
            Self::Fault(fault) => Some(fault),
        }
    }
}

fn lock_latches(shared: &HarnessShared) -> MutexGuard<'_, Latches> {
    shared
        .latches
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

fn lock_lanes(shared: &HarnessShared) -> MutexGuard<'_, HashMap<String, Arc<Lane>>> {
    shared.lanes.lock().unwrap_or_else(PoisonError::into_inner)
}

fn lock_close_completion(
    shared: &HarnessShared,
) -> MutexGuard<'_, Option<tokio::sync::watch::Receiver<CloseCompletion>>> {
    shared
        .close_completion
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// The latched fault or closed error, whichever the open check throws,
/// upstream's `assertOpen` reads: the fault wins, then the closed error.
fn latched_error(shared: &HarnessShared) -> Option<LaneError> {
    let latches = lock_latches(shared);
    latches.fault.clone().or_else(|| latches.closed.clone())
}

/// The open check, upstream's `assertOpen`.
fn assert_open(shared: &HarnessShared) -> Result<(), LaneError> {
    latched_error(shared).map_or(Ok(()), Err)
}

/// Collapses one harness error into the landed trait's closed surface, the
/// error-surface restatement the lane's trait layer records: the impl layer
/// rethrows the boxed error object, the trait layer reads only its message.
fn closed_lane_error(error: &LaneError) -> LaneOperationError {
    LaneOperationError::Closed(HarnessError::Closed {
        message: error.to_string(),
    })
}

/// Faults the harness, upstream's `fault`: the idempotent poison pill. The
/// latch, every lane's seal, the hook-registry close, the queued fault
/// event, and the bus close run in order; the fault event queues before the
/// bus closes so the delivery tail still drains it.
fn fault(shared: &HarnessShared, cause: LaneError, context: &Context) -> LaneError {
    let fault = {
        let mut latches = lock_latches(shared);
        if let Some(existing) = &latches.fault {
            return Arc::clone(existing);
        }
        if let Some(closed) = &latches.closed {
            return Arc::clone(closed);
        }
        let fault = Arc::new(HarnessFault::new(
            "AgentHarness storage or invariant fault",
            Box::new(FaultCause(cause)),
        ));
        latches.fault = Some(from_arc(Arc::clone(&fault)));
        fault
    };
    let fault_error = from_arc(Arc::clone(&fault));
    // The idle-drain waits drop here like upstream's discarded seal
    // promises; the seals' synchronous prefix ran in the kick.
    drop(seal_all_lanes(shared, &fault_error));
    shared.hooks.close(fault.message.clone());
    let event = HarnessEvent::global(HarnessEventPayload::Fault {
        code: "harness_fault".to_owned(),
        message: fault.message.clone(),
    })
    .unwrap_or_else(|construction_error| {
        unreachable!("fault is harness-global: {construction_error}")
    });
    // Fire-and-forget: the bus holds its delivery tail synchronously, so
    // the dropped future is only a completion signal.
    drop(shared.events.emit(event, context));
    shared.events.close(fault.message.clone());
    from_arc(fault)
}

/// Seals every lane with the one error object, upstream's
/// `for (const lane of this.lanesByName.values()) lane.seal(error)`, and
/// returns the still-pending idle-drain waits for the close's join.
fn seal_all_lanes(shared: &HarnessShared, error: &LaneError) -> Vec<BoxedFuture<'static, ()>> {
    let lanes = lock_lanes(shared).values().cloned().collect::<Vec<_>>();
    let mut pending = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let error = Arc::clone(error);
        let mut seal: BoxedFuture<'static, ()> = Box::pin(async move { lane.seal(error).await });
        // The synchronous prefix (the sealed-error latch, the drive-gate
        // close, the state-change signal) must run before the caller
        // continues; only the idle-owner await defers.
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        if seal.as_mut().poll(&mut cx).is_ready() {
            continue;
        }
        pending.push(seal);
    }
    pending
}

/// Builds the deferred delivery batch one mutation publishes, upstream's
/// `delivery = this.events.emitBatch(events, context)`: the batch's
/// recipients bind now and the future awaits outside the mutation.
fn deferred_delivery(
    shared: &Arc<HarnessShared>,
    events: Vec<HarnessEvent>,
    context: &Context,
) -> BoxedFuture<'static, Result<(), LaneError>> {
    let bus = Arc::clone(&shared.events);
    let context = context.clone();
    Box::pin(async move {
        bus.emit_batch(
            events
                .into_iter()
                .map(|event| (event, context.clone()))
                .collect(),
        )
        .await;
        Ok(())
    })
}

/// Awaits the memoized close completion, upstream's awaiting the shared
/// `closePromise`.
async fn await_close_completion(
    mut receiver: tokio::sync::watch::Receiver<CloseCompletion>,
) -> Result<(), LaneError> {
    loop {
        let settled = receiver.borrow_and_update().clone();
        if let Some(settled) = settled {
            return settled;
        }
        if receiver.changed().await.is_err() {
            // The drainer settles or the process is stopping; the sender
            // dropping early cannot carry a result.
            unreachable!("harness close drainer dropped without settling")
        }
    }
}

/// Renders a name the way `JSON.stringify` does, upstream's
/// `JSON.stringify(name)` in the invalid-lane and not-published messages
/// (serde escapes the NUL as `\u0000`, matching).
fn quoted(name: &str) -> String {
    serde_json::to_string(name).unwrap_or_else(|_| format!("{name:?}"))
}

/// Parses the stored branch tip, upstream's typed
/// `StoredValue<string | null>` read: the erased value parses here, and a
/// malformed tip carries the invariant the state restore reports.
fn parse_branch_tip(lane: &str, tip: &StoredValue) -> Result<Option<String>, LaneError> {
    serde_json::from_value::<Option<String>>(tip.value.clone()).map_err(|error| {
        lane_error(SessionError::Invariant(format!(
            "Lane {lane:?} tip is malformed: {error}"
        )))
    })
}

/// Resolves the acquisition's requested tip, upstream's
/// `options.createAt ?? null`: `Root` restates the `null` form, and both it
/// and an absent option tip at the branch's current state.
fn requested_tip(options: &AcquireLaneOptions) -> Option<String> {
    match &options.create_at {
        Some(CreateAt::Entry(entry_id)) => Some(entry_id.clone()),
        Some(CreateAt::Root) | None => None,
    }
}

/// The operation kind one intent serves, upstream's
/// `operation.meta.intent.kind` reads.
const fn intent_kind(meta: &OperationMeta) -> OperationKind {
    match &meta.intent {
        OperationIntent::Run { .. } => OperationKind::Run,
        OperationIntent::Compaction { .. } => OperationKind::Compaction,
        OperationIntent::Navigation { .. } => OperationKind::Navigation,
    }
}

/// The placeholder lane snapshot the watch installer passes, upstream's
/// `{} as LaneSnapshot`: the lane overwrites it with the real capture right
/// after the install, so the placeholder only satisfies the bus's typed
/// `watch` entry.
fn placeholder_lane_snapshot(lane: &str) -> LaneSnapshot {
    LaneSnapshot {
        lane: lane.to_owned(),
        transcript: Vec::new(),
        tip_id: None,
        last_result: None,
        configuration: LaneConfiguration {
            model: ModelIdentity {
                provider: String::new(),
                model_id: String::new(),
            },
            thinking_level: ThinkingLevel::Off,
            active_tool_names: Vec::new(),
        },
        stats: SessionStats::default(),
        operation: None,
        queues: Vec::new(),
        faulted: false,
    }
}

/// The default transcript conversion, upstream's
/// `toProviderMessages ?? ((messages) => convertToLlm(messages))`.
fn default_to_provider_messages() -> ToProviderMessages {
    Arc::new(|messages: &[AgentMessage], _context: &Context| {
        Box::pin(async move { convert_to_llm(messages) })
    })
}

/// Builds one lane over its durable state with the container's four seams,
/// upstream's `buildLane`.
fn build_lane(shared: &Arc<HarnessShared>, name: &str, state: LaneState) -> Arc<Lane> {
    let on_fault: FaultHandler = {
        let shared = Arc::clone(shared);
        Arc::new(move |cause: LaneError, context: &Context| fault(&shared, cause, context))
    };
    let emit_batch: EmitBatch = {
        let bus = Arc::clone(&shared.events);
        Arc::new(move |events: Vec<HarnessEvent>, context: Context| {
            let bus = Arc::clone(&bus);
            Box::pin(async move {
                bus.emit_batch(
                    events
                        .into_iter()
                        .map(|event| (event, context.clone()))
                        .collect(),
                )
                .await;
                Ok(())
            })
        })
    };
    let install_watch: WatchInstaller = {
        let bus = Arc::clone(&shared.events);
        let lane_name = name.to_owned();
        Arc::new(
            move |filter: WatchFilter,
                  _context: &Context,
                  resnapshot: ResnapshotCapture<LaneSnapshot>| {
                match bus.watch(
                    placeholder_lane_snapshot(&lane_name),
                    filter,
                    Some(resnapshot),
                ) {
                    Ok(watcher) => Ok(Box::new(BusWatchHandle { inner: watcher })),
                    // The closed error rides to the caller, upstream's
                    // `watch` throwing `closedError` — a watch racing the
                    // bus close rejects instead of raising an invariant.
                    Err(closed) => Err(lane_error(SessionError::Message(closed))),
                }
            },
        )
    };
    let read_config: ConfigProvider = {
        let shared = Arc::clone(shared);
        Arc::new(move || {
            shared
                .config_store
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        })
    };
    Arc::new(Lane::new(
        name,
        Arc::clone(&shared.session),
        Arc::clone(&shared.models),
        Arc::clone(&shared.hooks),
        state,
        on_fault,
        emit_batch,
        install_watch,
        read_config,
    ))
}

impl Harness {
    /// Builds the container over its options, seed configuration, and
    /// restored lanes, upstream's `constructor`: the hook registry's error
    /// reporter emits `handler_error` events through the bus, and the
    /// config store applies the options defaults
    /// (`"all"` steering/follow-up, `"parallel"` tool execution,
    /// [`default_retry_policy`], [`DEFAULT_COMPACTION_SETTINGS`],
    /// [`convert_to_llm`]).
    #[must_use]
    pub fn new(
        options: AgentHarnessOptions,
        seed: LaneConfiguration,
        restored: BTreeMap<String, LaneState>,
    ) -> Self {
        let events = Arc::new(HarnessEventBus::new());
        let report_error: HookErrorReporter = {
            let bus = Arc::clone(&events);
            Arc::new(
                move |error: HookFailure, hook: HookName, lane: &str, context: &Context| {
                    let bus = Arc::clone(&bus);
                    let payload = HarnessEventPayload::HandlerError {
                        error: error.to_string(),
                        stack: None,
                        kind: HandlerErrorKind::Hook {
                            hook: hook.as_str().to_owned(),
                        },
                    };
                    let event = HarnessEvent::lane_scoped(lane, false, payload).unwrap_or_else(
                        |construction_error| {
                            unreachable!("handler_error is lane-scoped: {construction_error}")
                        },
                    );
                    Box::pin(async move { bus.emit(event, context).await })
                },
            )
        };
        let hooks = Arc::new(HookRegistry::new(report_error));
        let config_store = RwLock::new(Config {
            tools: options.tools.unwrap_or_default(),
            resources: options.resources.unwrap_or_default(),
            stream_options: options.stream_options.unwrap_or_default(),
            retry_policy: options.retry.unwrap_or_else(default_retry_policy),
            compaction: options.compaction.unwrap_or(DEFAULT_COMPACTION_SETTINGS),
            steering_mode: options.steering_mode.unwrap_or_default(),
            follow_up_mode: options.follow_up_mode.unwrap_or_default(),
            tool_execution: options.tool_execution.unwrap_or_default(),
            tool_context: options.tool_context,
            system_prompt: options.system_prompt,
            to_provider_messages: options
                .to_provider_messages
                .unwrap_or_else(default_to_provider_messages),
            entry_projectors: options.entry_projectors.unwrap_or_default(),
        });
        let shared = Arc::new(HarnessShared {
            session: options.session,
            models: options.models,
            hooks,
            events,
            seed,
            lanes: Mutex::new(HashMap::new()),
            config_store,
            latches: Mutex::new(Latches {
                fault: None,
                closed: None,
            }),
            close_completion: Mutex::new(None),
        });
        {
            let mut lanes = lock_lanes(&shared);
            for (name, state) in restored {
                let lane = build_lane(&shared, &name, state);
                lanes.insert(name, lane);
            }
        }
        Self { shared }
    }

    /// The session the harness attaches to, upstream's `session`.
    #[must_use]
    pub fn session(&self) -> &Arc<dyn Session> {
        &self.shared.session
    }

    /// The models runtime, upstream's `models`.
    #[must_use]
    pub fn models(&self) -> &Arc<pi_ai::models::Models> {
        &self.shared.models
    }

    /// The hook registry, upstream's `hooks`.
    #[must_use]
    pub fn hooks(&self) -> &Arc<HookRegistry> {
        &self.shared.hooks
    }

    /// The event bus, upstream's `events`.
    #[must_use]
    pub fn events(&self) -> &Arc<HarnessEventBus> {
        &self.shared.events
    }

    /// Acquires one lane by name, upstream's `lane(name, context)`: the
    /// same instance returns for every acquisition of one name.
    ///
    /// # Errors
    /// The latched fault or closed error, the `InvalidLane`/`UnknownTarget`
    /// caller errors passed through without faulting, and the fault every
    /// other mutation failure becomes.
    pub async fn lane(
        &self,
        name: &str,
        context: &Context,
    ) -> Result<Arc<dyn AgentLane>, LaneError> {
        self.lane_impl(name, AcquireLaneOptions::default(), context)
            .await
    }

    /// Acquires one lane with options, upstream's
    /// `lane(name, options, context)`.
    ///
    /// # Errors
    /// The latched fault or closed error, the `InvalidLane`/`UnknownTarget`
    /// caller errors passed through without faulting, and the fault every
    /// other mutation failure becomes.
    pub async fn lane_with_options(
        &self,
        name: &str,
        options: AcquireLaneOptions,
        context: &Context,
    ) -> Result<Arc<dyn AgentLane>, LaneError> {
        self.lane_impl(name, options, context).await
    }

    /// Acquires or builds the lane, upstream's `lane` body: the serialized
    /// mutation line classifies the lane's storage, restores a configured
    /// lane, adopts a stored branch tip, or creates the lane at the
    /// requested tip; the `lane_created` delivery awaits after the mutation
    /// settles.
    ///
    /// # Errors
    /// The latched fault or closed error (checked again inside the mutation
    /// line), the `InvalidLane`/`UnknownTarget` caller errors passed
    /// through without faulting, and the fault every other mutation failure
    /// becomes.
    #[expect(
        clippy::too_many_lines,
        reason = "the port mirrors upstream's single lane method"
    )]
    async fn lane_impl(
        &self,
        name: &str,
        options: AcquireLaneOptions,
        context: &Context,
    ) -> Result<Arc<dyn AgentLane>, LaneError> {
        assert_open(&self.shared)?;
        if name.is_empty() || name.contains('\u{0}') {
            let reason = if name.is_empty() {
                "lane name must not be empty"
            } else {
                "lane name must not contain \\u0000"
            };
            return Err(lane_error(HarnessError::InvalidLane {
                lane: name.to_owned(),
                reason: reason.to_owned(),
                message: format!("Invalid lane {}: {reason}", quoted(name)),
            }));
        }
        let shared = Arc::clone(&self.shared);
        let callback_name = name.to_owned();
        let callback: SessionMutationCallback = Box::new(move |mutator, mutation_context| {
            let shared = Arc::clone(&shared);
            let name = callback_name.clone();
            let options = options.clone();
            Box::pin(async move {
                let outcome: Result<
                    Option<BoxedFuture<'static, Result<(), LaneError>>>,
                    LaneError,
                > = async {
                    assert_open(&shared)?;
                    if lock_lanes(&shared).contains_key(&name) {
                        return Ok(None);
                    }
                    let stored = read_lane_storage(mutator, &name, mutation_context)
                        .await
                        .map_err(lane_error)?;
                    if let ClassifiedLaneStorage::Lane {
                        tip,
                        configuration,
                        lane_state,
                    } = &stored
                    {
                        let restored = restore_lane_state(
                            mutator,
                            &name,
                            tip,
                            configuration,
                            lane_state,
                            mutation_context,
                        )
                        .await
                        .map_err(lane_error)?;
                        let lane = build_lane(&shared, &name, restored);
                        lock_lanes(&shared).insert(name.clone(), Arc::clone(&lane));
                        return Ok(None);
                    }
                    let tip_id = match &stored {
                        ClassifiedLaneStorage::Branch { tip } => parse_branch_tip(&name, tip)?,
                        ClassifiedLaneStorage::Absent => requested_tip(&options),
                        ClassifiedLaneStorage::Lane { .. } => {
                            unreachable!("the restore path returned above")
                        }
                    };
                    if let (ClassifiedLaneStorage::Absent, Some(target_id)) = (&stored, &tip_id) {
                        let entries = mutator
                            .get_entries(vec![target_id.clone()], mutation_context)
                            .await
                            .map_err(lane_error)?;
                        if !entries.contains_key(target_id) {
                            return Err(lane_error(HarnessError::UnknownTarget {
                                target_id: target_id.clone(),
                                message: format!("Unknown target: {target_id}"),
                            }));
                        }
                    }
                    let attached_configuration = LaneConfiguration {
                        model: shared.seed.model.clone(),
                        thinking_level: shared.seed.thinking_level,
                        active_tool_names: shared.seed.active_tool_names.clone(),
                    };
                    let state = LaneState {
                        tip_id: tip_id.clone(),
                        configuration: attached_configuration.clone(),
                        inbox: Vec::new(),
                        last_operation_id: None,
                        operation: None,
                    };
                    let mut writes = Vec::new();
                    if matches!(&stored, ClassifiedLaneStorage::Absent) {
                        writes.push(
                            set_value_write(&branch_tip(&name), tip_id.clone())
                                .map_err(lane_error)?,
                        );
                    }
                    writes.push(
                        set_value_write(&lane_config(&name), attached_configuration)
                            .map_err(lane_error)?,
                    );
                    writes.push(
                        set_value_write(
                            &lane_state(&name),
                            crate::harness::session::types::LaneState {
                                current_operation_id: None,
                                last_operation_id: None,
                                inbox: Vec::new(),
                            },
                        )
                        .map_err(lane_error)?,
                    );
                    mutator
                        .commit(writes, mutation_context)
                        .await
                        .map_err(lane_error)?;
                    let lane = build_lane(&shared, &name, state);
                    lock_lanes(&shared).insert(name.clone(), Arc::clone(&lane));
                    let event = HarnessEvent::lane_scoped(
                        &name,
                        false,
                        HarnessEventPayload::LaneCreated { at: tip_id },
                    )
                    .unwrap_or_else(|construction_error| {
                        unreachable!("lane_created is lane-scoped: {construction_error}")
                    });
                    Ok(Some(deferred_delivery(
                        &shared,
                        vec![event],
                        mutation_context,
                    )))
                }
                .await;
                match outcome {
                    Ok(delivery) => Ok(any_payload(MutationOutcome::Published { delivery })),
                    Err(error) => Ok(any_payload(MutationOutcome::Threw(error))),
                }
            })
        });
        let delivery = match self.run_mutation(callback, context).await {
            Ok(delivery) => delivery,
            Err(error) => return Err(self.catch_mutation(error, context)),
        };
        if let Some(delivery) = delivery {
            delivery.await?;
        }
        let published = lock_lanes(&self.shared).get(name).cloned();
        let Some(published) = published else {
            return Err(self.fault(
                lane_error(SessionError::Invariant(format!(
                    "Lane {} was not published",
                    quoted(name)
                ))),
                context,
            ));
        };
        Ok(published)
    }

    /// The lane identity views, upstream's `lanes`: the lanes' execution
    /// reads carry no effects and their results pin the lane order, so the
    /// `Promise.all` restates as ordered awaits.
    ///
    /// # Errors
    /// The latched fault or closed error, or a lane's sealed error.
    pub async fn lanes(&self, context: &Context) -> Result<Vec<LaneInfo>, LaneError> {
        assert_open(&self.shared)?;
        let lanes = lock_lanes(&self.shared)
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let mut infos = Vec::with_capacity(lanes.len());
        for lane in lanes {
            let execution = lane
                .inspect_execution_impl(context)
                .await
                .map_err(lane_error)?;
            infos.push(LaneInfo {
                name: execution.lane,
                tip_id: execution.tip_id,
                operation: execution.current,
            });
        }
        Ok(infos)
    }

    /// The session name, upstream's `getName`.
    ///
    /// # Errors
    /// The latched fault or closed error, or the session read's failure.
    pub async fn get_name(&self, context: &Context) -> Result<Option<String>, LaneError> {
        assert_open(&self.shared)?;
        self.shared
            .session
            .get_name(context)
            .await
            .map_err(lane_error)
    }

    /// Sets the session name, upstream's `setName`: the write and the
    /// `value_update` delivery cross the mutation line in order.
    ///
    /// # Errors
    /// The latched fault or closed error, or the fault any mutation failure
    /// becomes (this catch has no passthrough variants).
    pub async fn set_name(&self, name: Option<String>, context: &Context) -> Result<(), LaneError> {
        assert_open(&self.shared)?;
        let shared = Arc::clone(&self.shared);
        let callback: SessionMutationCallback = Box::new(move |mutator, mutation_context| {
            let shared = Arc::clone(&shared);
            let name = name.clone();
            Box::pin(async move {
                let outcome: Result<
                    Option<BoxedFuture<'static, Result<(), LaneError>>>,
                    LaneError,
                > = async {
                    assert_open(&shared)?;
                    let write = match &name {
                        Some(name) => {
                            set_value_write(&session_name(), name.clone()).map_err(lane_error)?
                        }
                        None => delete_value_write(&session_name()),
                    };
                    mutator
                        .commit(vec![write], mutation_context)
                        .await
                        .map_err(lane_error)?;
                    let event = HarnessEvent::global(HarnessEventPayload::ValueUpdate {
                        kind: ValueUpdateKind::SessionName { name },
                    })
                    .unwrap_or_else(|construction_error| {
                        unreachable!("value_update is harness-global: {construction_error}")
                    });
                    Ok(Some(deferred_delivery(
                        &shared,
                        vec![event],
                        mutation_context,
                    )))
                }
                .await;
                match outcome {
                    Ok(delivery) => Ok(any_payload(MutationOutcome::Published { delivery })),
                    Err(error) => Ok(any_payload(MutationOutcome::Threw(error))),
                }
            })
        });
        let delivery = match self.run_mutation(callback, context).await {
            Ok(delivery) => delivery,
            Err(error) => return Err(self.catch_closed_or_fault(error, context)),
        };
        if let Some(delivery) = delivery {
            delivery.await?;
        }
        Ok(())
    }

    /// One entry's label, upstream's `getLabel`.
    ///
    /// # Errors
    /// The latched fault or closed error, or the session read's failure.
    pub async fn get_label(
        &self,
        target_id: &str,
        context: &Context,
    ) -> Result<Option<String>, LaneError> {
        assert_open(&self.shared)?;
        self.shared
            .session
            .get_label(target_id, context)
            .await
            .map_err(lane_error)
    }

    /// Sets one entry's label, upstream's `setLabel`: the write and the
    /// `value_update` delivery cross the mutation line in order.
    ///
    /// # Errors
    /// The latched fault or closed error, or the fault any mutation failure
    /// becomes (this catch has no passthrough variants).
    pub async fn set_label(
        &self,
        target_id: &str,
        label: Option<String>,
        context: &Context,
    ) -> Result<(), LaneError> {
        assert_open(&self.shared)?;
        let shared = Arc::clone(&self.shared);
        let target_id = target_id.to_owned();
        let callback: SessionMutationCallback = Box::new(move |mutator, mutation_context| {
            let shared = Arc::clone(&shared);
            let target_id = target_id.clone();
            let label = label.clone();
            Box::pin(async move {
                let outcome: Result<
                    Option<BoxedFuture<'static, Result<(), LaneError>>>,
                    LaneError,
                > = async {
                    assert_open(&shared)?;
                    let address = entry_label(&target_id);
                    let write = match &label {
                        Some(label) => {
                            set_value_write(&address, label.clone()).map_err(lane_error)?
                        }
                        None => delete_value_write(&address),
                    };
                    mutator
                        .commit(vec![write], mutation_context)
                        .await
                        .map_err(lane_error)?;
                    let event = HarnessEvent::global(HarnessEventPayload::ValueUpdate {
                        kind: ValueUpdateKind::EntryLabel {
                            target_id,
                            label: label.clone(),
                        },
                    })
                    .unwrap_or_else(|construction_error| {
                        unreachable!("value_update is harness-global: {construction_error}")
                    });
                    Ok(Some(deferred_delivery(
                        &shared,
                        vec![event],
                        mutation_context,
                    )))
                }
                .await;
                match outcome {
                    Ok(delivery) => Ok(any_payload(MutationOutcome::Published { delivery })),
                    Err(error) => Ok(any_payload(MutationOutcome::Threw(error))),
                }
            })
        });
        let delivery = match self.run_mutation(callback, context).await {
            Ok(delivery) => delivery,
            Err(error) => return Err(self.catch_closed_or_fault(error, context)),
        };
        if let Some(delivery) = delivery {
            delivery.await?;
        }
        Ok(())
    }

    /// Watches the session, upstream's `watchSession`: the later harness
    /// slice owns the session watch, and the operation reports
    /// [`SliceNotImplemented`] to the caller without faulting the harness.
    ///
    /// # Errors
    /// The `watchSession is not implemented until its later AgentHarness
    /// slice` message, wrapped for the trait surface; the harness stays open.
    pub fn watch_session(
        &self,
        _context: &Context,
    ) -> Result<Box<dyn WatchHandle<SessionSnapshot>>, LaneError> {
        Err(lane_error(SliceNotImplemented::new("watchSession")))
    }

    /// Faults the harness, upstream's `fault`: the idempotent poison pill
    /// that seals every lane, closes the hooks and the bus, and returns the
    /// error the thrower rethrows.
    pub fn fault(&self, cause: LaneError, context: &Context) -> LaneError {
        fault(&self.shared, cause, context)
    }

    /// Closes the harness, upstream's `close`: the closed latch, every
    /// lane's seal prefix, and the hooks+bus closes run before the promise
    /// returns, and the memoized drain joins the session close with every
    /// lane's idle wait. Concurrent closes share one completion.
    ///
    /// # Errors
    /// The session close's failure.
    #[must_use]
    pub fn close(&self, context: &Context) -> BoxedFuture<'static, Result<(), LaneError>> {
        let (sender, receiver) = tokio::sync::watch::channel(CloseCompletion::None);
        let claimed = {
            let mut memo = lock_close_completion(&self.shared);
            if memo.is_some() {
                false
            } else {
                *memo = Some(receiver.clone());
                true
            }
        };
        if !claimed {
            let winner = lock_close_completion(&self.shared)
                .as_ref()
                .cloned()
                .unwrap_or_else(|| {
                    unreachable!("the losing close reads the winner's memoized receiver")
                });
            return Box::pin(await_close_completion(winner));
        }
        let closed: LaneError = from_arc(Arc::new(HarnessClosed));
        {
            let mut latches = lock_latches(&self.shared);
            if latches.closed.is_none() {
                latches.closed = Some(Arc::clone(&closed));
            }
        }
        let pending_seals = seal_all_lanes(&self.shared, &closed);
        self.shared.hooks.close(closed.to_string());
        self.shared.events.close(closed.to_string());
        let session = Arc::clone(&self.shared.session);
        let context = context.clone();
        tokio::spawn(async move {
            let session_close = session.close(&context);
            let (session_result, ()) = tokio::join!(session_close, async move {
                for seal in pending_seals {
                    seal.await;
                }
            });
            let _ = sender.send(Some(session_result.map_err(lane_error)));
        });
        Box::pin(await_close_completion(receiver))
    }

    /// Runs one mutation callback through the session line and unwraps its
    /// outcome payload into the thrown error or the deferred delivery,
    /// upstream's `await this.session.mutate(...)`: a session-level failure
    /// rethrows into the catch like a thrown callback error.
    async fn run_mutation(
        &self,
        callback: SessionMutationCallback,
        context: &Context,
    ) -> Result<Option<BoxedFuture<'static, Result<(), LaneError>>>, LaneError> {
        let outcome = match self.shared.session.mutate(callback, context).await {
            Ok(payload) => payload.downcast::<MutationOutcome>().map_or_else(
                |_| {
                    MutationOutcome::Threw(lane_error(SessionError::Message(
                        "the mutation callback returns a mutation outcome".to_owned(),
                    )))
                },
                |outcome| *outcome,
            ),
            Err(error) => MutationOutcome::Threw(lane_error(error)),
        };
        match outcome {
            MutationOutcome::Threw(error) => Err(error),
            MutationOutcome::Published { delivery } => Ok(delivery),
        }
    }

    /// Applies the lane-acquisition catch block, upstream's `lane`'s catch:
    /// the closed error wins, the `InvalidLane`/`UnknownTarget` caller
    /// errors pass through without faulting, and everything else faults.
    fn catch_mutation(&self, error: LaneError, context: &Context) -> LaneError {
        let closed = lock_latches(&self.shared).closed.clone();
        if let Some(closed) = closed {
            return closed;
        }
        if matches!(
            error.as_ref().downcast_ref::<HarnessError>(),
            Some(HarnessError::InvalidLane { .. } | HarnessError::UnknownTarget { .. })
        ) {
            return error;
        }
        self.fault(error, context)
    }

    /// Applies the name/label catch block, upstream's `setName`/`setLabel`'s
    /// catch: the closed error wins and everything else faults.
    fn catch_closed_or_fault(&self, error: LaneError, context: &Context) -> LaneError {
        let closed = lock_latches(&self.shared).closed.clone();
        if let Some(closed) = closed {
            return closed;
        }
        self.fault(error, context)
    }

    /// Reads one config field, upstream's `getConfig`: the open check gates
    /// the read.
    fn config_field<T>(&self, read: impl FnOnce(&Config) -> T) -> Result<T, LaneError> {
        assert_open(&self.shared)?;
        let store = self
            .shared
            .config_store
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        Ok(read(&store))
    }

    /// Replaces one config field and emits its update event, upstream's
    /// `setConfig`: the read-modify-write runs under the store's write lock
    /// so the captured `previous` and the replacement commit together, then
    /// the single emit carries the event.
    async fn set_config(
        &self,
        replace: impl FnOnce(&mut Config) -> HarnessEventPayload,
        context: &Context,
    ) -> Result<(), LaneError> {
        assert_open(&self.shared)?;
        let event = {
            let mut store = self
                .shared
                .config_store
                .write()
                .unwrap_or_else(PoisonError::into_inner);
            replace(&mut store)
        };
        let event = HarnessEvent::global(event).unwrap_or_else(|construction_error| {
            unreachable!("config_update is harness-global: {construction_error}")
        });
        self.shared.events.emit(event, context).await;
        Ok(())
    }
}

impl AgentHarness for Harness {
    fn lane(
        &self,
        name: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Arc<dyn AgentLane>, LaneOperationError>> {
        let harness = self.clone();
        let name = name.to_owned();
        let context = context.clone();
        Box::pin(async move {
            harness
                .lane(&name, &context)
                .await
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn lane_with_options(
        &self,
        name: &str,
        options: AcquireLaneOptions,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Arc<dyn AgentLane>, LaneOperationError>> {
        let harness = self.clone();
        let name = name.to_owned();
        let context = context.clone();
        Box::pin(async move {
            harness
                .lane_with_options(&name, options, &context)
                .await
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn lanes(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<LaneInfo>, LaneOperationError>> {
        let harness = self.clone();
        let context = context.clone();
        Box::pin(async move {
            harness
                .lanes(&context)
                .await
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn get_name(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, LaneOperationError>> {
        let harness = self.clone();
        let context = context.clone();
        Box::pin(async move {
            harness
                .get_name(&context)
                .await
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn set_name(
        &self,
        name: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>> {
        let harness = self.clone();
        let context = context.clone();
        Box::pin(async move {
            harness
                .set_name(name, &context)
                .await
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn get_label(
        &self,
        target_id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, LaneOperationError>> {
        let harness = self.clone();
        let target_id = target_id.to_owned();
        let context = context.clone();
        Box::pin(async move {
            harness
                .get_label(&target_id, &context)
                .await
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn set_label(
        &self,
        target_id: &str,
        label: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>> {
        let harness = self.clone();
        let target_id = target_id.to_owned();
        let context = context.clone();
        Box::pin(async move {
            harness
                .set_label(&target_id, label, &context)
                .await
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn get_tools(
        &self,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<AgentHarnessTool>, LaneOperationError>> {
        Box::pin(async move {
            Self::config_field(self, |config| config.tools.clone())
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn set_tools(
        &self,
        tools: Vec<AgentHarnessTool>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>> {
        let context = context.clone();
        Box::pin(async move {
            let names: Vec<&str> = tools.iter().map(|tool| tool.tool.name.as_str()).collect();
            if let Err(message) = validate_tool_names(&names) {
                return Err(closed_lane_error(&lane_error(SessionError::Message(
                    message,
                ))));
            }
            Self::set_config(
                self,
                move |config: &mut Config| {
                    config.tools = tools;
                    HarnessEventPayload::ConfigUpdate {
                        property: ConfigUpdateKind::Global(GlobalConfigUpdate::Tools),
                    }
                },
                &context,
            )
            .await
            .map_err(|error| closed_lane_error(&error))
        })
    }

    fn get_resources(
        &self,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Resources, LaneOperationError>> {
        Box::pin(async move {
            Self::config_field(self, |config| config.resources.clone())
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn set_resources(
        &self,
        resources: Resources,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>> {
        let context = context.clone();
        Box::pin(async move {
            Self::set_config(
                self,
                move |config: &mut Config| {
                    config.resources = resources;
                    HarnessEventPayload::ConfigUpdate {
                        property: ConfigUpdateKind::Global(GlobalConfigUpdate::Resources),
                    }
                },
                &context,
            )
            .await
            .map_err(|error| closed_lane_error(&error))
        })
    }

    fn get_stream_options(
        &self,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<AgentHarnessStreamOptions, LaneOperationError>> {
        Box::pin(async move {
            Self::config_field(self, |config| config.stream_options.clone())
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn set_stream_options(
        &self,
        options: AgentHarnessStreamOptions,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>> {
        let context = context.clone();
        Box::pin(async move {
            Self::set_config(
                self,
                move |config: &mut Config| {
                    let previous = std::mem::replace(&mut config.stream_options, options.clone());
                    HarnessEventPayload::ConfigUpdate {
                        property: ConfigUpdateKind::Global(GlobalConfigUpdate::StreamOptions {
                            value: options,
                            previous,
                        }),
                    }
                },
                &context,
            )
            .await
            .map_err(|error| closed_lane_error(&error))
        })
    }

    fn get_retry_policy(
        &self,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<RetryPolicy, LaneOperationError>> {
        Box::pin(async move {
            Self::config_field(self, |config| config.retry_policy)
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn set_retry_policy(
        &self,
        policy: RetryPolicy,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>> {
        let context = context.clone();
        Box::pin(async move {
            if let Err(message) = validate_retry_policy(&policy) {
                return Err(closed_lane_error(&lane_error(SessionError::Message(
                    message,
                ))));
            }
            Self::set_config(
                self,
                move |config: &mut Config| {
                    let previous = std::mem::replace(&mut config.retry_policy, policy);
                    HarnessEventPayload::ConfigUpdate {
                        property: ConfigUpdateKind::Global(GlobalConfigUpdate::RetryPolicy {
                            value: policy,
                            previous,
                        }),
                    }
                },
                &context,
            )
            .await
            .map_err(|error| closed_lane_error(&error))
        })
    }

    fn get_compaction_settings(
        &self,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<CompactionSettings, LaneOperationError>> {
        Box::pin(async move {
            Self::config_field(self, |config| config.compaction)
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn set_compaction_settings(
        &self,
        settings: CompactionSettings,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>> {
        let context = context.clone();
        Box::pin(async move {
            if let Err(message) = validate_compaction_settings(&settings) {
                return Err(closed_lane_error(&lane_error(SessionError::Message(
                    message,
                ))));
            }
            Self::set_config(
                self,
                move |config: &mut Config| {
                    let previous = std::mem::replace(&mut config.compaction, settings);
                    HarnessEventPayload::ConfigUpdate {
                        property: ConfigUpdateKind::Global(
                            GlobalConfigUpdate::CompactionSettings {
                                value: settings,
                                previous,
                            },
                        ),
                    }
                },
                &context,
            )
            .await
            .map_err(|error| closed_lane_error(&error))
        })
    }

    fn get_steering_mode(
        &self,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<QueueMode, LaneOperationError>> {
        Box::pin(async move {
            Self::config_field(self, |config| config.steering_mode)
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn set_steering_mode(
        &self,
        mode: QueueMode,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>> {
        let context = context.clone();
        Box::pin(async move {
            Self::set_config(
                self,
                move |config: &mut Config| {
                    let previous = std::mem::replace(&mut config.steering_mode, mode);
                    HarnessEventPayload::ConfigUpdate {
                        property: ConfigUpdateKind::Global(GlobalConfigUpdate::SteeringMode {
                            value: mode,
                            previous,
                        }),
                    }
                },
                &context,
            )
            .await
            .map_err(|error| closed_lane_error(&error))
        })
    }

    fn get_follow_up_mode(
        &self,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<QueueMode, LaneOperationError>> {
        Box::pin(async move {
            Self::config_field(self, |config| config.follow_up_mode)
                .map_err(|error| closed_lane_error(&error))
        })
    }

    fn set_follow_up_mode(
        &self,
        mode: QueueMode,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>> {
        let context = context.clone();
        Box::pin(async move {
            Self::set_config(
                self,
                move |config: &mut Config| {
                    let previous = std::mem::replace(&mut config.follow_up_mode, mode);
                    HarnessEventPayload::ConfigUpdate {
                        property: ConfigUpdateKind::Global(GlobalConfigUpdate::FollowUpMode {
                            value: mode,
                            previous,
                        }),
                    }
                },
                &context,
            )
            .await
            .map_err(|error| closed_lane_error(&error))
        })
    }

    fn watch_session(
        &self,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn WatchHandle<SessionSnapshot>>, LaneOperationError>> {
        Box::pin(async {
            Err(LaneOperationError::Closed(HarnessError::Closed {
                message: SliceNotImplemented::new("watchSession").to_string(),
            }))
        })
    }

    fn hooks(&self) -> &dyn Hooks {
        self.shared.hooks.as_ref()
    }

    fn events(&self) -> &dyn Events {
        self.shared.events.as_ref()
    }

    fn close(&self, context: &Context) -> BoxedFuture<'_, Result<(), LaneOperationError>> {
        let close = Self::close(self, context);
        Box::pin(async move { close.await.map_err(|error| closed_lane_error(&error)) })
    }
}

/// The `Events` contract over the concrete bus, upstream's
/// `HarnessEventBus implements Events`: the closed-bus rejection restates
/// as the throw it is, since the trait surface carries no error channel.
impl Events for HarnessEventBus {
    #[expect(
        clippy::panic,
        reason = "upstream's `on` throws the closed error; the trait surface carries no error channel, so the throw restates as the panic"
    )]
    fn on(&self, event_type: HarnessEventType, listener: EventListener) -> Subscription {
        match Self::on(self, event_type, listener) {
            Ok(subscription) => subscription,
            Err(closed_error) => panic!("{closed_error}"),
        }
    }
}

/// The `Hooks` contract over the concrete registry, upstream's
/// `HookRegistry implements Hooks`: the closed-registry rejection restates
/// as the throw it is, since the trait surface carries no error channel.
impl Hooks for HookRegistry {
    #[expect(
        clippy::panic,
        reason = "upstream's `on` throws the closed error; the trait surface carries no error channel, so the throw restates as the panic"
    )]
    fn on(&self, name: HookName, handler: HookHandler, options: HookOptions) -> Subscription {
        match Self::on(self, name, handler, options) {
            Ok(subscription) => subscription,
            Err(closed_error) => panic!("{closed_error}"),
        }
    }
}

/// Attaches the runtime without starting provider, tool, hook, or timer
/// effects, upstream's `createAgentHarness`.
///
/// The options validate eagerly in tools, retry, compaction order; the seed
/// restates the options' model and tool names (`model.id` becomes the
/// wire's `modelId`); the restore reads one coherent session mutation; and
/// the restore failures wrap in a fresh [`HarnessFault`] per attempt — not
/// the instance fault path, which has no instance yet.
///
/// # Errors
/// The options validation rejections, or the fresh
/// `AgentHarness storage or invariant fault` wrap of a restore failure.
pub async fn create_agent_harness(
    options: AgentHarnessOptions,
    context: &Context,
) -> Result<CreatedHarness, HarnessCreationError> {
    let tools = options.tools.clone().unwrap_or_default();
    let tool_names: Vec<&str> = tools.iter().map(|tool| tool.tool.name.as_str()).collect();
    validate_tool_names(&tool_names).map_err(HarnessCreationError::Validation)?;
    let retry_policy = options.retry.unwrap_or_else(default_retry_policy);
    validate_retry_policy(&retry_policy).map_err(HarnessCreationError::Validation)?;
    let compaction = options.compaction.unwrap_or(DEFAULT_COMPACTION_SETTINGS);
    validate_compaction_settings(&compaction).map_err(HarnessCreationError::Validation)?;
    let seed = LaneConfiguration {
        model: ModelIdentity {
            provider: options.model.provider.0.clone(),
            model_id: options.model.id.clone(),
        },
        thinking_level: options.thinking_level.unwrap_or_default(),
        active_tool_names: options
            .active_tool_names
            .clone()
            .unwrap_or_else(|| tools.iter().map(|tool| tool.tool.name.clone()).collect()),
    };
    let restored = match restore_session(Arc::clone(&options.session), context).await {
        Ok(restored) => restored,
        Err(error) => {
            return Err(HarnessCreationError::Fault(HarnessFault::new(
                "AgentHarness storage or invariant fault",
                Box::new(error),
            )));
        }
    };
    let open: Vec<OpenOperation> = restored
        .iter()
        .filter_map(|(lane, state)| {
            let operation = state.operation.as_ref()?;
            Some(OpenOperation {
                lane: lane.clone(),
                operation_id: operation.meta.operation_id.clone(),
                kind: intent_kind(&operation.meta),
                started_at: operation.meta.started_at,
                aborting: matches!(
                    operation_scope_of(&operation.state).control,
                    Control::CancelRequested { .. }
                )
                .then_some(true),
            })
        })
        .collect();
    let harness = Harness::new(options, seed, restored);
    Ok(CreatedHarness { harness, open })
}

#[cfg(test)]
mod tests;

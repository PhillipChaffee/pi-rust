//! The runtime suites' fixtures, ported from upstream
//! `test/harness/runtime/test-utils.ts` over the landed session layer.
//!
//! The controlled storage decorator restates upstream's
//! `ControlledMemoryStorage`, `FailingStorage`, and `FailingMemoryStorage`
//! subclasses of `MemoryStorage`, alongside the one-shot release gate, the
//! recording watch adapter the lane fixture installs, and the shared lane
//! configuration. The hand-rolled contract-implementing memory session this
//! module once staged is retired by the session-layer child's merge — the
//! suites construct the real [`StorageBackedSession`] over
//! [`MemoryStorage`].
//!
//! The unused watch installer raises when reached, upstream's `unusedWatch`
//! throw; the suite pins that raise with the module-level panic expectation.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the unused watch installer raises deliberately, upstream's `unusedWatch` throw"
)]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_ai::types::BoxedFuture;

use crate::harness::agent_harness::EventListener;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::LaneSnapshot;
use crate::harness::agent_harness::WatchHandle;
use crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS;
use crate::harness::context::Context;
use crate::harness::events::ListenerError;
use crate::harness::events::ResnapshotCapture;
use crate::harness::events::WatchFilter;
use crate::harness::messages::convert_to_llm;
use crate::harness::runtime::lane::EmitBatch;
use crate::harness::runtime::lane::FaultHandler;
use crate::harness::runtime::lane::WatchInstaller;
use crate::harness::runtime::types::Config;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::memory::MemoryStorage;
use crate::harness::session::session::StorageBackedSession;
use crate::harness::session::session::StorageBackedSessionOptions;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::Entry;
use crate::harness::session::types::EntryScan;
use crate::harness::session::types::EntryStructure;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::ModelIdentity;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionMetadata;
use crate::harness::session::types::SessionStats;
use crate::harness::session::types::Storage;
use crate::harness::session::types::StorageBranchScan;
use crate::harness::session::types::UsageRow;
use crate::harness::session::types::UsageScan;
use crate::harness::session::values::ListAddress;
use crate::harness::session::values::ListElement;
use crate::harness::session::values::ListReadOptions;
use crate::harness::session::values::StoredValue;
use crate::harness::session::values::ValueAddress;
use crate::harness::session::values::Write;
use crate::harness::session::values::set_value_write;
use crate::harness::types::AgentHarnessResources;
use crate::harness::types::AgentHarnessStreamOptions;
use crate::types::QueueMode;
use crate::types::ThinkingLevel;
use crate::types::ToolExecutionMode;

/// The lane configuration every fixture starts from, upstream's
/// `configuration` constant.
#[must_use]
pub(super) fn lane_configuration() -> LaneConfiguration {
    LaneConfiguration {
        model: ModelIdentity {
            provider: "test".to_owned(),
            model_id: "model".to_owned(),
        },
        thinking_level: ThinkingLevel::Off,
        active_tool_names: Vec::new(),
    }
}

/// The empty lane snapshot a fresh watch captures from, upstream's
/// `{} as LaneSnapshot` seed the fixture adapters receive.
#[must_use]
pub(super) fn empty_lane_snapshot(lane: &str, configuration: &LaneConfiguration) -> LaneSnapshot {
    LaneSnapshot {
        lane: lane.to_owned(),
        transcript: Vec::new(),
        tip_id: None,
        last_result: None,
        configuration: configuration.clone(),
        stats: SessionStats {
            message_count: 0,
            usage: pi_ai::types::Usage::default(),
        },
        operation: None,
        queues: Vec::new(),
        faulted: false,
    }
}

/// The runtime configuration the fixtures read, upstream's `runtimeConfig`
/// and the lane fixture's config closure.
#[must_use]
pub(super) fn runtime_config() -> Config {
    Config {
        tools: Vec::new(),
        resources: AgentHarnessResources::default(),
        stream_options: AgentHarnessStreamOptions::default(),
        retry_policy: pi_ai::utils::retry::RetryPolicy {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 1_000,
            max_agent_delay_ms: None,
        },
        compaction: DEFAULT_COMPACTION_SETTINGS,
        steering_mode: QueueMode::All,
        follow_up_mode: QueueMode::All,
        tool_execution: ToolExecutionMode::Parallel,
        tool_context: None,
        system_prompt: None,
        to_provider_messages: Arc::new(|messages, _context| {
            Box::pin(async move { convert_to_llm(messages) })
        }),
        entry_projectors: std::collections::BTreeMap::new(),
    }
}

/// The one-shot release gate upstream's `deferred()` restates.
#[must_use]
pub(super) fn deferred() -> (
    tokio::sync::oneshot::Sender<()>,
    tokio::sync::oneshot::Receiver<()>,
) {
    tokio::sync::oneshot::channel()
}

/// The settle upstream's `setTimeout(resolve, 0)` await restates: yields
/// until the bus delivery tail and the watchers' spawned drain tasks have
/// flushed their work onto the current-thread runtime.
pub(super) async fn settle_events() {
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
}

/// The body one controlled commit runs ahead of its forward, upstream's
/// `beforeNextCommit` — it may park (awaiting the test's release gate) or
/// throw, and the throw becomes the commit's error.
pub(super) type BeforeCommitFn =
    Box<dyn FnOnce() -> BoxedFuture<'static, Result<(), SessionError>> + Send + Sync>;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The controlled memory storage the runtime suites drive, upstream's
/// `ControlledMemoryStorage` / `FailingStorage` / `FailingMemoryStorage`
/// subclasses: one hook ahead of the next commit, one one-shot commit
/// failure, and the value-read counter the progress assertions read
/// (upstream's `vi.spyOn(storage, "getValue")`).
pub(super) struct ControlledStorage {
    memory: Arc<MemoryStorage>,
    before_next_commit: Mutex<Option<BeforeCommitFn>>,
    failure: Mutex<Option<SessionError>>,
    get_value_calls: AtomicUsize,
}

impl std::fmt::Debug for ControlledStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlledStorage")
            .field(
                "get_value_calls",
                &self.get_value_calls.load(Ordering::SeqCst),
            )
            .finish_non_exhaustive()
    }
}

impl ControlledStorage {
    /// A controlled storage over the memory backend.
    #[must_use]
    pub(super) fn new(delegate: Arc<MemoryStorage>) -> Self {
        Self {
            memory: delegate,
            before_next_commit: Mutex::new(None),
            failure: Mutex::new(None),
            get_value_calls: AtomicUsize::new(0),
        }
    }

    /// Arms the next commit's hook, upstream's `beforeNextCommit`
    /// assignment.
    pub(super) fn set_before_next_commit(&self, hook: Option<BeforeCommitFn>) {
        *lock(&self.before_next_commit) = hook;
    }

    /// Arms the next commit's failure, upstream's `failure` assignment.
    pub(super) fn set_failure(&self, failure: Option<SessionError>) {
        *lock(&self.failure) = failure;
    }

    /// The recorded value-read count, upstream's `getValue` spy.
    #[must_use]
    pub(super) fn get_value_calls(&self) -> usize {
        self.get_value_calls.load(Ordering::SeqCst)
    }
}

impl Storage for ControlledStorage {
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        let hook = lock(&self.before_next_commit).take();
        let failure = lock(&self.failure).take();
        let memory = Arc::clone(&self.memory);
        let context = context.clone();
        Box::pin(async move {
            if let Some(hook) = hook {
                hook().await?;
            }
            match failure {
                Some(failure) => Err(failure),
                None => memory.commit(writes, &context).await,
            }
        })
    }

    fn get_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        self.get_value_calls.fetch_add(1, Ordering::SeqCst);
        self.memory.get_value(address, context)
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        self.memory.scan_values(prefix, context)
    }

    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<std::collections::BTreeMap<String, Entry>, SessionError>> {
        self.memory.get_entries(ids, context)
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        self.memory.read_list(address, options, context)
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.memory.scan_branch(query, context)
    }

    fn scan_branch_structure(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<EntryStructure>, SessionError>> {
        self.memory.scan_branch_structure(query, context)
    }

    fn scan_entries(
        &self,
        query: &EntryScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.memory.scan_entries(query, context)
    }

    fn scan_usage(
        &self,
        query: &UsageScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<UsageRow>, SessionError>> {
        self.memory.scan_usage(query, context)
    }

    fn get_stats(&self, context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        self.memory.get_stats(context)
    }

    fn close(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.memory.close(context)
    }
}

/// The no-op watch handle the lane fixture installs, upstream's
/// `(snapshot) => ({ snapshot, start: () => {}, resnapshot: () =>
/// Promise.resolve(snapshot), unsubscribe: () => {} })`: the stored
/// snapshot never changes, the listener is dropped, and resnapshot
/// resolves the captured snapshot.
pub(super) struct RecordingWatch {
    snapshot: Mutex<LaneSnapshot>,
}

impl WatchHandle<LaneSnapshot> for RecordingWatch {
    fn snapshot(&self) -> LaneSnapshot {
        lock(&self.snapshot).clone()
    }

    fn set_snapshot(&self, snapshot: LaneSnapshot) {
        *lock(&self.snapshot) = snapshot;
    }

    fn start(&self, _listener: EventListener) {}

    fn resnapshot<'a>(
        &self,
        _context: &'a Context,
    ) -> BoxedFuture<'a, Result<LaneSnapshot, ListenerError>> {
        let snapshot = WatchHandle::<LaneSnapshot>::snapshot(self);
        Box::pin(std::future::ready(Ok(snapshot)))
    }

    fn unsubscribe(&self) {}
}

/// Builds the lane fixture's watch installer: one that hands back the
/// recording adapter, upstream's `(snapshot) => ({ ... })` literal.
#[must_use]
pub(super) fn recording_watch_installer(initial: LaneSnapshot) -> WatchInstaller {
    Arc::new(
        move |_filter: WatchFilter,
              _context: &Context,
              _resnapshot: ResnapshotCapture<LaneSnapshot>| {
            Box::new(RecordingWatch {
                snapshot: Mutex::new(initial.clone()),
            })
        },
    )
}

/// Builds the watch installer the progress fixture passes, upstream's
/// `unusedWatch`: a watch these suites never call, which panics if a code
/// path ever reaches it.
#[must_use]
pub(super) fn unused_watch_installer() -> WatchInstaller {
    Arc::new(
        |_filter: WatchFilter, _context: &Context, _resnapshot: ResnapshotCapture<LaneSnapshot>| {
            panic!("watch is not used by progress tests")
        },
    )
}

/// The metadata the runtime fixtures pin, upstream's
/// `{ id, createdAt: 1, storageVersion: 1 }` literals.
#[must_use]
pub(super) fn runtime_session_metadata(id: String) -> SessionMetadata {
    SessionMetadata {
        id,
        created_at: 1,
        storage_version: 1,
        cwd: None,
        parent_session_id: None,
        legacy_parent_session_path: None,
    }
}

/// The seed commit the lane-constructing fixtures make, upstream's
/// `createLane`'s `mutator.commit([...])`.
///
/// The branch tip sits at none with the lane configuration and the empty
/// lane state; `Some(operation_id)` additionally persists an admitted
/// starting operation under that id (the drive-fault fixture's seed).
///
/// # Errors
/// The seed commit's failure.
pub(super) async fn seed_main_lane_values(
    session: &Arc<StorageBackedSession>,
    operation_id: Option<&str>,
) -> Result<(), SessionError> {
    let writes = {
        let mut writes = vec![
            set_value_write(
                &crate::harness::session::values::branch_tip("main"),
                Option::<String>::None,
            )?,
            set_value_write(
                &crate::harness::session::values::lane_config("main"),
                lane_configuration(),
            )?,
            set_value_write(
                &crate::harness::session::values::lane_state("main"),
                crate::harness::session::types::LaneState {
                    current_operation_id: operation_id.map(str::to_owned),
                    last_operation_id: None,
                    inbox: Vec::new(),
                },
            )?,
        ];
        if let Some(operation_id) = operation_id {
            let meta = crate::harness::session::types::OperationMeta {
                operation_id: operation_id.to_owned(),
                lane: "main".to_owned(),
                source_tip_id: None,
                started_at: 1,
                intent: crate::harness::session::types::OperationIntent::Run {
                    prompt_entry_ids: Vec::new(),
                },
            };
            writes.push(set_value_write(
                &crate::harness::session::values::operation_meta(operation_id),
                meta,
            )?);
            writes.push(set_value_write(
                &crate::harness::session::values::operation_state(operation_id),
                crate::harness::session::types::OperationState::Starting(
                    crate::harness::session::types::StartingOperation {
                        scope: crate::harness::session::types::OperationScope {
                            control: crate::harness::session::types::Control::Running,
                            settings: crate::harness::session::types::RunSettings {
                                compaction: DEFAULT_COMPACTION_SETTINGS,
                                steering_mode: QueueMode::All,
                                follow_up_mode: QueueMode::All,
                                tool_execution: ToolExecutionMode::Parallel,
                            },
                            latest_assistant_entry_id: None,
                        },
                    },
                ),
            )?);
        }
        writes
    };
    session
        .mutate(
            Box::new(move |mutator, context| {
                let writes = writes.clone();
                Box::pin(async move {
                    mutator.commit(writes, context).await?;
                    let payload: Box<dyn std::any::Any + Send> = Box::new(());
                    Ok(payload)
                })
            }),
            &crate::harness::context::background_context(),
        )
        .await
        .map(|_| ())
}

/// The storage-backed session the runtime fixtures construct over the
/// controlled memory backend.
#[must_use]
pub(super) fn runtime_session(id: String, storage: Arc<ControlledStorage>) -> StorageBackedSession {
    StorageBackedSession::new(
        runtime_session_metadata(id),
        storage,
        StorageBackedSessionOptions::default(),
    )
}

/// The fault handler upstream's `(cause) => (cause instanceof Error ? cause
/// : new Error(String(cause)))` restates to: the boxed error passes
/// through unchanged.
#[must_use]
pub(super) fn passthrough_fault_handler() -> FaultHandler {
    Arc::new(|cause: LaneError, _context: &Context| cause)
}

/// The no-op emit batch upstream's `() => Promise.resolve()` restates.
#[must_use]
pub(super) fn noop_emit_batch() -> EmitBatch {
    Arc::new(|_events: Vec<HarnessEvent>, _context: Context| Box::pin(std::future::ready(Ok(()))))
}

/// The lane error a one-shot commit failure carries, upstream's thrown
/// `Error("commit failed")`.
#[must_use]
pub(super) fn commit_failure(message: &str) -> LaneError {
    lane_error(SessionError::Message(message.to_owned()))
}

/// The no-op hook error reporter the fixtures install, upstream's
/// `new HookRegistry(() => {})` argument.
#[must_use]
pub(super) fn noop_hook_reporter() -> crate::harness::hooks::HookErrorReporter {
    Arc::new(|_error, _name, _site, _context| Box::pin(std::future::ready(())))
}

/// The watch handle the bus-backed fixtures install, the erased restatement
/// of the buffered watcher the bus mints (`BufferedEventWatcher` implements
/// [`WatchHandle`] on the concrete type, so the `Arc` wraps through this
/// delegating adapter).
pub(super) struct BusWatchHandle {
    inner: Arc<crate::harness::events::BufferedEventWatcher<LaneSnapshot>>,
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

/// The emit surface the bus-backed fixtures install: every batch rides the
/// bus's ordered delivery tail, upstream's `HarnessEventBus` restated.
#[must_use]
pub(super) fn bus_emit_batch(bus: Arc<crate::harness::events::HarnessEventBus>) -> EmitBatch {
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
}

/// The watch installer the bus-backed fixtures hand the lane: the bus
/// watches with the fixture's seed snapshot, the resnapshot capture the
/// lane supplies, and the erased adapter above.
#[must_use]
pub(super) fn bus_watch_installer(
    bus: Arc<crate::harness::events::HarnessEventBus>,
    initial: LaneSnapshot,
) -> WatchInstaller {
    Arc::new(
        move |filter: WatchFilter,
              _context: &Context,
              resnapshot: ResnapshotCapture<LaneSnapshot>| {
            let watcher = bus
                .watch::<LaneSnapshot>(initial.clone(), filter, Some(resnapshot))
                .expect("watch installs");
            let boxed: Box<dyn WatchHandle<LaneSnapshot>> =
                Box::new(BusWatchHandle { inner: watcher });
            boxed
        },
    )
}

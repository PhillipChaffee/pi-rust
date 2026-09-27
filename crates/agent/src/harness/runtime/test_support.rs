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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pi_ai::models::create_models;
use pi_ai::types::{BoxedFuture, Message, TextContent, UserBlock, UserContent, UserMessage};

use crate::harness::agent_harness::AgentLane;
use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::EventListener;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::LaneSnapshot;
use crate::harness::agent_harness::OperationAdmission;
use crate::harness::agent_harness::OperationRequest;
use crate::harness::agent_harness::PromptMessagesPayload;
use crate::harness::agent_harness::WatchHandle;
use crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS;
use crate::harness::context::Context;
use crate::harness::context::background_context;
use crate::harness::events::ListenerError;
use crate::harness::events::ResnapshotCapture;
use crate::harness::events::WatchFilter;
use crate::harness::messages::convert_to_llm;
use crate::harness::runtime::lane::ConfigProvider;
use crate::harness::runtime::lane::EmitBatch;
use crate::harness::runtime::lane::FaultHandler;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::lane::WatchInstaller;
use crate::harness::runtime::restore::restore_lane;
use crate::harness::runtime::types::Config;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::memory::MemoryStorage;
use crate::harness::session::memory::MemoryStorageOptions;
use crate::harness::session::session::StorageBackedSession;
use crate::harness::session::session::StorageBackedSessionOptions;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::DeferredEffectPendingOperation;
use crate::harness::session::types::DeferredScope;
use crate::harness::session::types::Entry;
use crate::harness::session::types::EntryScan;
use crate::harness::session::types::EntryStructure;
use crate::harness::session::types::GenerationContext;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::ModelIdentity;
use crate::harness::session::types::OperationResultRecord;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionMetadata;
use crate::harness::session::types::SessionStats;
use crate::harness::session::types::Storage;
use crate::harness::session::types::StorageBranchScan;
use crate::harness::session::types::SummaryContext;
use crate::harness::session::types::SummaryGenerationScope;
use crate::harness::session::types::SummaryTask;
use crate::harness::session::types::ToolBatch;
use crate::harness::session::types::ToolCall;
use crate::harness::session::types::ToolsOperation;
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
use crate::types::AgentMessage;
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
    let (sender, receiver) = tokio::sync::oneshot::channel();
    (sender, receiver)
}

/// The session id counter every runtime suite shares: one id per fixture
/// session within the test process, the uniqueness the per-suite counters
/// gave.
#[must_use]
pub(super) fn next_session_id() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("runtime-fixture-{}", COUNTER.fetch_add(1, Ordering::SeqCst))
}

/// Locks one mutex, poisoned-lock recovery included, upstream's plain
/// field reads under the single event loop.
pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Builds one plain user text message, upstream's `{ role: "user",
/// content: text, timestamp: 1 }` literals.
#[must_use]
pub(super) fn user_text_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        timestamp: 1,
        content: UserContent::Text(text.to_owned()),
    }))
}

/// Builds one blocks-carried user message, upstream's
/// `{ type: "user", content: [...] }` construction.
#[must_use]
pub(super) fn user_blocks_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        timestamp: 1,
        content: UserContent::Blocks(vec![UserBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })]),
    }))
}

/// The running scope every leaf fixture carries, upstream's
/// `operationScope()` / `runScope()` fixtures.
#[must_use]
pub(super) fn operation_scope() -> OperationScope {
    OperationScope {
        control: crate::harness::session::types::Control::Running,
        settings: crate::harness::session::types::RunSettings {
            compaction: DEFAULT_COMPACTION_SETTINGS,
            steering_mode: QueueMode::All,
            follow_up_mode: QueueMode::All,
            tool_execution: ToolExecutionMode::Parallel,
        },
        latest_assistant_entry_id: None,
    }
}

/// The starting leaf the seeded operations carry, upstream's
/// `{ at: "starting", ... }` fixtures.
#[must_use]
pub(super) fn starting_run_state() -> OperationState {
    OperationState::Starting(crate::harness::session::types::StartingOperation {
        scope: operation_scope(),
    })
}

/// The generation inputs the model-carrying leaves carry, upstream's
/// `generationContext()` fixture.
#[must_use]
pub(super) fn generation_context() -> GenerationContext {
    GenerationContext {
        step_id: "step".to_owned(),
        trigger_entry_id: "trigger".to_owned(),
        configuration: lane_configuration(),
        stream_options: AgentHarnessStreamOptions::default(),
        retry_policy: crate::harness::session::types::NormalizedRetryPolicy {
            max_attempts: 2,
            base_delay_ms: 1,
            max_agent_delay_ms: 30_000,
        },
        overflow_recovery_used: false,
    }
}

/// The compaction entry body the fixtures build, upstream's inline
/// `{ summary, retainedTail: [], ... }` literals.
#[must_use]
pub(super) fn compaction_entry_body(
    summary: &str,
) -> crate::harness::session::types::CompactionEntryBody {
    crate::harness::session::types::CompactionEntryBody {
        summary: summary.to_owned(),
        retained_tail: Vec::new(),
        tokens_before: 0,
        details: None,
        usage: None,
        from_hook: false,
    }
}

/// The raw value-set write the malformed fixtures build: the serialized
/// payload bypasses the typed setter, mirroring a corrupted store.
#[must_use]
pub(super) fn raw_write(address: &ValueAddress, value: serde_json::Value) -> Write {
    Write::ValueSet(crate::harness::session::values::ValueSetWrite {
        value,
        key: address.key.clone(),
        namespace: address.namespace.clone(),
        op: "set".to_owned(),
        kind: "value".to_owned(),
    })
}

/// The zeroed usage block the test assistant wires carry, upstream's
/// inline `usage: { input: 0, ... }` literals.
#[must_use]
pub(super) fn zero_usage_wire() -> serde_json::Value {
    let zero = 0;
    serde_json::json!({
        "input": zero, "output": zero, "cacheRead": zero, "cacheWrite": zero, "totalTokens": zero,
        "cost": { "input": zero, "output": zero, "cacheRead": zero, "cacheWrite": zero, "total": zero },
    })
}

/// The assistant wire the fixtures parse, upstream's inline assistant
/// literals with the zero usage and the free stop reason.
#[must_use]
pub(super) fn assistant_wire_value(
    content: &serde_json::Value,
    stop_reason: &str,
) -> serde_json::Value {
    serde_json::json!({
        "role": "assistant",
        "content": content,
        "api": "anthropic-messages",
        "provider": "test",
        "model": "model",
        "usage": zero_usage_wire(),
        "stopReason": stop_reason,
        "timestamp": 1,
    })
}

/// The lane record write one commit publishes, upstream's
/// `storedValues.setValue(storedValues.laneState(...), ...)` calls.
///
/// # Errors
/// The write's serialization failure.
pub(super) fn lane_state_write(
    lane: &str,
    current_operation_id: Option<&str>,
    last_operation_id: Option<&str>,
    inbox: Vec<InboxItem>,
) -> Result<Write, SessionError> {
    set_value_write(
        &crate::harness::session::values::lane_state(lane),
        crate::harness::session::types::LaneState {
            current_operation_id: current_operation_id.map(str::to_owned),
            last_operation_id: last_operation_id.map(str::to_owned),
            inbox,
        },
    )
}

/// The main lane's seed writes, upstream's `createLane`'s commit list: the
/// branch tip at none, the lane configuration, and the idle lane state.
///
/// # Errors
/// A write's serialization failure.
pub(super) fn main_lane_seed_writes(
    configuration: &LaneConfiguration,
) -> Result<Vec<Write>, SessionError> {
    Ok(vec![
        set_value_write(
            &crate::harness::session::values::branch_tip("main"),
            Option::<String>::None,
        )?,
        set_value_write(
            &crate::harness::session::values::lane_config("main"),
            configuration.clone(),
        )?,
        lane_state_write("main", None, None, Vec::new())?,
    ])
}

/// Commits one write transaction through the session's mutation line,
/// upstream's fixtures' `mutator.commit(writes)` callbacks; an unexpected
/// result panics the test by design.
pub(super) async fn commit_writes<S>(session: &Arc<S>, writes: Vec<Write>)
where
    S: Session + ?Sized,
{
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
            &background_context(),
        )
        .await
        .expect("the commit settles");
}

/// The session over a plain memory backend, seeded with the idle (or the
/// seeded-started) `main` lane, upstream's `createSession` fixtures.
///
/// # Panics
/// The seed commit's failure.
pub(super) async fn memory_session_with_seed(
    operation_id: Option<&str>,
) -> Arc<StorageBackedSession> {
    let session = Arc::new(StorageBackedSession::new(
        runtime_session_metadata(next_session_id()),
        Arc::new(MemoryStorage::new(MemoryStorageOptions::default())),
        StorageBackedSessionOptions::default(),
    ));
    seed_main_lane_values(&session, operation_id)
        .await
        .expect("seed commit");
    session
}

/// The lane over one restored `main` state with the standard fixture
/// arguments, upstream's `createLane`'s `new Lane(...)` call.
///
/// # Panics
/// The restore's failure.
pub(super) async fn restored_lane(
    session: Arc<StorageBackedSession>,
    emit_batch: EmitBatch,
    install_watch: WatchInstaller,
) -> Lane {
    restored_lane_with_config(session, emit_batch, install_watch, Arc::new(runtime_config)).await
}

/// The lane variant the resource-configuring suites drive: one custom
/// config provider over the restored `main` state.
///
/// # Panics
/// The restore's failure.
pub(super) async fn restored_lane_with_config(
    session: Arc<StorageBackedSession>,
    emit_batch: EmitBatch,
    install_watch: WatchInstaller,
    read_config: ConfigProvider,
) -> Lane {
    let restored = restore_lane(session.clone(), "main", &background_context())
        .await
        .expect("restore");
    Lane::new(
        "main",
        session,
        Arc::new(create_models(None)),
        Arc::new(crate::harness::hooks::HookRegistry::new(
            noop_hook_reporter(),
        )),
        restored,
        passthrough_fault_handler(),
        emit_batch,
        install_watch,
        read_config,
    )
}

/// Gates one commit behind the test's release, upstream's
/// `beforeNextCommit` fixtures: the hook signals `commit_started` and parks
/// until released. Returns `(started, release)`.
#[must_use]
pub(super) fn gate_next_commit(
    storage: &ControlledStorage,
) -> (
    tokio::sync::oneshot::Receiver<()>,
    tokio::sync::oneshot::Sender<()>,
) {
    let (commit_started, started) = deferred();
    let (release, release_rx) = deferred();
    storage.set_before_next_commit(Some(Box::new(move || {
        Box::pin(async move {
            let _ = commit_started.send(());
            let _ = release_rx.await;
            Ok(())
        })
    })));
    (started, release)
}

/// Gates one read behind the test's release: the read signals `started` and
/// parks until released, the read-side rig the capture-mid-seal tests
/// sequence against. Returns `(started, release)`.
#[must_use]
pub(super) fn gate_next_read(
    storage: &ControlledStorage,
) -> (
    tokio::sync::oneshot::Receiver<()>,
    tokio::sync::oneshot::Sender<()>,
) {
    let (read_started, started) = deferred();
    let (release, release_rx) = deferred();
    storage.arm_read_gate(Box::new(move || {
        Box::pin(async move {
            let _ = read_started.send(());
            let _ = release_rx.await;
            Ok(())
        })
    }));
    (started, release)
}

/// One plain drive pass over the background context, the fixtures' `Drive`
/// constructions.
#[must_use]
pub(super) fn drive_pass(operation_id: &str) -> Arc<Drive> {
    Arc::new(Drive::new(
        &DriveOptions {
            operation_id: operation_id.to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ))
}

/// The settled run record the fixtures write, upstream's
/// `{ status: "completed", startedAt: 1, endedAt: 2 }` literals.
#[must_use]
pub(super) fn settled_run_record(operation_id: &str) -> OperationResultRecord {
    OperationResultRecord {
        operation_id: operation_id.to_owned(),
        kind: crate::harness::session::types::OperationKind::Run,
        status: crate::harness::session::types::TerminalStatus::Completed,
        error: None,
        from_tip_id: None,
        tip_id: None,
        started_at: 1,
        ended_at: 2,
    }
}

/// The admitted operation's id, upstream's
/// `lane.state().operation!.meta.operationId` reads; an unexpected absence
/// panics the test by design.
#[must_use]
pub(super) fn live_operation_id(lane: &Lane) -> String {
    lane.state()
        .operation
        .as_ref()
        .map(|operation| operation.meta.operation_id.clone())
        .expect("the live operation")
}

/// Swaps the live operation's durable state leaf, the capture fixtures'
/// patch, upstream's per-case operation-state commits.
///
/// # Panics
/// The live operation's absence or the patch's failure.
pub(super) async fn patch_live_state(lane: &Lane, next_state: OperationState) {
    let live = lane.state().operation.expect("the live operation");
    let operation_id = live.meta.operation_id.clone();
    let meta = live.meta;
    lane.command::<(), _>(
        move |state, _session, _context| {
            let operation_id = operation_id.clone();
            let meta = meta.clone();
            let next_state = next_state.clone();
            Box::pin(async move {
                let mut next = state.clone();
                next.operation = Some(crate::harness::runtime::types::LiveOperation {
                    meta,
                    state: next_state.clone(),
                });
                Ok(crate::harness::runtime::types::LaneCommand::Commit {
                    writes: vec![
                        set_value_write(
                            &crate::harness::session::values::operation_state(&operation_id),
                            next_state,
                        )
                        .expect("state write"),
                    ],
                    next,
                    materialize: Arc::new(|_commit| ()),
                    events: None,
                })
            })
        },
        &background_context(),
    )
    .await
    .expect("the patch commits");
}

/// Accepts one text prompt admission, the fixtures' `lane.accept` calls.
///
/// # Panics
/// The accept surface's failure or the admission's rejection.
pub(super) async fn accept_text(lane: &Lane, prompt: &str) -> OperationAdmission {
    lane.accept(
        OperationRequest::Prompt {
            operation_id: None,
            prompt: Box::new(PromptMessagesPayload::Text {
                prompt: prompt.to_owned(),
                images: None,
            }),
        },
        &background_context(),
    )
    .await
    .expect("accept serves")
    .expect("the admission")
}

/// Finishes the active operation through the lane's own finish path: the
/// result record writes and the operation clears, upstream's
/// `settleOperation` finish fixtures.
///
/// # Panics
/// The active operation's absence or the finish's failure.
pub(super) async fn finish_operation(lane: &Lane) {
    let operation_id = live_operation_id(lane);
    let record = settled_run_record(&operation_id);
    lane.settle_operation::<(), _>(
        move |_state, _operation_state, _meta, _session, _context| {
            let record = record.clone();
            Box::pin(async move {
                Ok(crate::harness::runtime::types::OperationCommand::Finish {
                    writes: Vec::new(),
                    record,
                    lane: None,
                    materialize: Arc::new(|_commit| ()),
                    events: None,
                })
            })
        },
        &background_context(),
    )
    .await
    .expect("the finish settles");
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

/// One armed read rejection, upstream's `FailingMemoryStorage`: the
/// namespace and key narrow the arm to one address family; `None` matches
/// every read of that axis.
#[derive(Clone)]
struct ReadFailureArm {
    namespace: Option<String>,
    key: Option<String>,
    error: SessionError,
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
    read_failure: Mutex<Option<ReadFailureArm>>,
    stats_failure: Mutex<Option<SessionError>>,
    read_gate: Mutex<Option<BeforeCommitFn>>,
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
            read_failure: Mutex::new(None),
            stats_failure: Mutex::new(None),
            read_gate: Mutex::new(None),
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

    /// Arms the read rejection, upstream's `FailingMemoryStorage`: the
    /// armed namespace and key narrow the arm (`None` matches the axis
    /// whole), and the arm stays until re-armed or cleared.
    pub(super) fn arm_read_failure(
        &self,
        namespace: Option<&str>,
        key: Option<&str>,
        failure: SessionError,
    ) {
        *lock(&self.read_failure) = Some(ReadFailureArm {
            namespace: namespace.map(str::to_owned),
            key: key.map(str::to_owned),
            error: failure,
        });
    }

    /// Clears the armed read rejection.
    pub(super) fn clear_read_failure(&self) {
        *lock(&self.read_failure) = None;
    }

    /// Arms one stats-read failure, the one-shot the snapshot capture's
    /// `get_stats` arm drives.
    pub(super) fn arm_stats_failure(&self, failure: SessionError) {
        *lock(&self.stats_failure) = Some(failure);
    }

    /// Arms the next read's gate: the read signals `started` and parks
    /// until released, the sequencing hook the capture-mid-seal tests
    /// drive.
    pub(super) fn arm_read_gate(&self, hook: BeforeCommitFn) {
        *lock(&self.read_gate) = Some(hook);
    }

    /// The recorded value-read count, upstream's `getValue` spy.
    #[must_use]
    pub(super) fn get_value_calls(&self) -> usize {
        self.get_value_calls.load(Ordering::SeqCst)
    }

    /// The armed rejection one address read carries, upstream's
    /// `FailingMemoryStorage` method guard.
    fn read_rejection(&self, namespace: &str, key: &str) -> Option<SessionError> {
        let arm = lock(&self.read_failure).clone()?;
        if arm
            .namespace
            .as_ref()
            .is_some_and(|armed| armed != namespace)
        {
            return None;
        }
        if arm.key.as_ref().is_some_and(|armed| armed != key) {
            return None;
        }
        Some(arm.error)
    }

    /// The armed rejection one key-scoped read (an entry-id read or a
    /// branch-scan start) carries.
    fn key_rejection(&self, key: &str) -> Option<SessionError> {
        let arm = lock(&self.read_failure).clone()?;
        if arm.namespace.is_some() {
            return None;
        }
        if arm.key.as_ref().is_some_and(|armed| armed != key) {
            return None;
        }
        Some(arm.error)
    }

    /// Runs the armed read gate, the parked read the sequencing tests
    /// release.
    fn run_read_gate(&self) -> Option<BeforeCommitFn> {
        lock(&self.read_gate).take()
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
        if let Some(hook) = self.run_read_gate() {
            let memory = Arc::clone(&self.memory);
            let context = context.clone();
            let address = address.clone();
            return Box::pin(async move {
                hook().await?;
                memory.get_value(&address, &context).await
            });
        }
        if let Some(error) = self.read_rejection(&address.namespace, &address.key) {
            return Box::pin(async move { Err(error) });
        }
        self.get_value_calls.fetch_add(1, Ordering::SeqCst);
        self.memory.get_value(address, context)
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        if let Some(error) = self.read_rejection(&prefix.namespace, &prefix.key) {
            return Box::pin(async move { Err(error) });
        }
        self.memory.scan_values(prefix, context)
    }

    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<std::collections::BTreeMap<String, Entry>, SessionError>> {
        if let Some(error) = ids.iter().find_map(|id| self.key_rejection(id)) {
            return Box::pin(async move { Err(error) });
        }
        self.memory.get_entries(ids, context)
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        if let Some(error) = self.read_rejection(&address.namespace, &address.key) {
            return Box::pin(async move { Err(error) });
        }
        self.memory.read_list(address, options, context)
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        if let Some(hook) = self.run_read_gate() {
            let memory = Arc::clone(&self.memory);
            let context = context.clone();
            let query = query.clone();
            return Box::pin(async move {
                hook().await?;
                memory.scan_branch(&query, &context).await
            });
        }
        if let Some(error) = self.key_rejection(&query.start) {
            return Box::pin(async move { Err(error) });
        }
        self.memory.scan_branch(query, context)
    }

    fn scan_branch_structure(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<EntryStructure>, SessionError>> {
        if let Some(error) = self.key_rejection(&query.start) {
            return Box::pin(async move { Err(error) });
        }
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
        let failure = lock(&self.stats_failure).take();
        if let Some(error) = failure {
            return Box::pin(async move { Err(error) });
        }
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
/// starting operation under that id (the drive-fault fixture's seed) and
/// points the lane record at it.
///
/// # Errors
/// The seed commit's failure.
pub(super) async fn seed_main_lane_values(
    session: &Arc<StorageBackedSession>,
    operation_id: Option<&str>,
) -> Result<(), SessionError> {
    let mut writes = if operation_id.is_some() {
        vec![
            set_value_write(
                &crate::harness::session::values::branch_tip("main"),
                Option::<String>::None,
            )?,
            set_value_write(
                &crate::harness::session::values::lane_config("main"),
                lane_configuration(),
            )?,
            lane_state_write("main", operation_id, None, Vec::new())?,
        ]
    } else {
        main_lane_seed_writes(&lane_configuration())?
    };
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
            starting_run_state(),
        )?);
    }
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
            &background_context(),
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

/// The structural task the summary leaves carry, upstream's `summaryTask`
/// fixture with the boundary's reason.
#[must_use]
pub(super) fn summary_task(boundary: ResultBoundary) -> SummaryTask {
    let reason = match &boundary {
        ResultBoundary::Finish => Some(CompactionReason::Manual),
        ResultBoundary::ResumeCheckpoint { .. } => Some(CompactionReason::Threshold),
        ResultBoundary::CommitNavigation { .. } => None,
    };
    SummaryTask {
        task_id: "task".to_owned(),
        reason,
        custom_instructions: None,
        boundary,
    }
}

/// The summary generation inputs the summary leaves carry, upstream's
/// `summaryGeneration` fixture.
#[must_use]
pub(super) fn summary_generation() -> SummaryGenerationScope {
    SummaryGenerationScope {
        task: summary_task(ResultBoundary::Finish),
        summary_context: SummaryContext {
            result_entry_id: "summary".to_owned(),
            configuration: lane_configuration(),
            stream_options: AgentHarnessStreamOptions::default(),
            retry_policy: generation_context().retry_policy,
        },
    }
}

/// The deferred scope the deferred leaves carry, upstream's `deferredScope`
/// fixture.
#[must_use]
pub(super) fn deferred_scope(
    scope: OperationScope,
    source_entry_id: &str,
    poll: u64,
) -> DeferredScope {
    DeferredScope {
        scope,
        step_id: "step".to_owned(),
        source_entry_id: source_entry_id.to_owned(),
        poll,
        configuration: lane_configuration(),
        stream_options: AgentHarnessStreamOptions::default(),
    }
}

/// The deferred-effect-pending leaf the deferred-phase fixtures drive.
#[must_use]
pub(super) fn deferred_effect_pending(
    source_entry_id: &str,
    response_entry_id: &str,
    poll: u64,
) -> OperationState {
    OperationState::DeferredEffectPending(DeferredEffectPendingOperation {
        scope: deferred_scope(operation_scope(), source_entry_id, poll),
        response_entry_id: response_entry_id.to_owned(),
        usage_id: "usage".to_owned(),
    })
}

/// The tools leaf the batch captures drive, upstream's
/// `{ ...runScope(), at: "tools", batch }` fixture.
#[must_use]
pub(super) fn tools_batch_state(assistant_entry_id: &str, calls: Vec<ToolCall>) -> OperationState {
    OperationState::Tools(ToolsOperation {
        scope: operation_scope(),
        batch: ToolBatch {
            assistant_entry_id: assistant_entry_id.to_owned(),
            configuration: lane_configuration(),
            turn_id: "turn".to_owned(),
            calls,
        },
    })
}

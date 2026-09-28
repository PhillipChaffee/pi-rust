//! The shared drive-suite fixtures, ported from the upstream drive test
//! files' per-file fixture blocks at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`:
//! `drive-structural.test.ts`, `drive-generation.test.ts`,
//! `drive-tools.test.ts`, `drive-reconcile.test.ts`, and
//! `drive-retry-deferred.test.ts`.
//!
//! The four hand-seeded suites (structural, generation, reconciliation,
//! retry/deferred) share one fixture skeleton — the instrumented storage
//! over the fixed `100` memory clock, the storage-backed session, the faux
//! provider, the collecting emit batch, the unused watch installer, and
//! the lane over the restored `main` state with the pass installed — and
//! the per-suite differences are the [`DriveFixtureSpec`] fields. The two
//! storage decorators, the leaf/task/preparation builders, and the seeded
//! installers restate the per-file fixture code; the public, terminal,
//! retry, and tools suites build their own fixtures on top of this module
//! (public via [`crate::harness::runtime::harness::create_agent_harness`],
//! terminal over a plain storage, tools over [`ObservedMemoryStorage`],
//! retry pure).
//!
//! The unused watch installer carries the suite's name in its raise —
//! upstream's per-file `unusedWatch` throw — and the wait helper carries
//! upstream's 200-poll budget with the `condition was not reached` raise.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the unused watch installer and the wait budget raise deliberately, upstream's `unusedWatch` throw and `waitFor` failure"
)]
#![expect(
    dead_code,
    reason = "the fixtures serve the per-suite drive test files; the test-port wave lands their callers and retires this expectation"
)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_ai::models::{Models, create_models};
use pi_ai::providers::faux::{FauxProviderHandle, RegisterFauxProviderOptions, faux_provider};
use pi_ai::types::{BoxedFuture, Message};
use pi_ai::utils::retry::RetryPolicy;

use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HookFailure;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::agent_harness::HookOptions;
use crate::harness::agent_harness::HookResult;
use crate::harness::agent_harness::{OperationRequest, PromptMessagesPayload, ToProviderMessages};
use crate::harness::compaction::types::CompactionSettings;
use crate::harness::compaction::types::{DEFAULT_COMPACTION_SETTINGS, create_file_ops};
use crate::harness::context::{Context, background_context};
use crate::harness::hooks::HookRegistry;
use crate::harness::runtime::drive::drive_operation;
use crate::harness::runtime::drive::structural::commit_navigation;
use crate::harness::runtime::drive::structural::recover_structural_generation;
use crate::harness::runtime::drive::structural::run_structural_decision;
use crate::harness::runtime::drive::structural::run_structural_generation;
use crate::harness::runtime::lane::EmitBatch;
use crate::harness::runtime::lane::{Lane, operation_state_with_scope};
use crate::harness::runtime::restore::restore_lane;
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::deferred;
use crate::harness::runtime::test_support::lane_state_write;
use crate::harness::runtime::test_support::lock;
use crate::harness::runtime::test_support::main_lane_seed_writes;
use crate::harness::runtime::test_support::memory_storage_tail;
pub(super) use crate::harness::runtime::test_support::next_session_id_in as next_suite_session_id;
use crate::harness::runtime::test_support::noop_hook_reporter;
pub(super) use crate::harness::runtime::test_support::unused_watch_installer as unused_watch;
pub(super) use crate::harness::runtime::test_support::user_message as user;
use crate::harness::runtime::test_support::{passthrough_fault_handler, runtime_session_metadata};
use crate::harness::runtime::types::Config;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneCommand;
use crate::harness::runtime::types::{LaneError, LaneState, LiveOperation, ProcedureResult};
use crate::harness::session::commit::insert_entry;
use crate::harness::session::memory::{MemoryStorage, MemoryStorageOptions, NowFn};
use crate::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use crate::harness::session::testing::InstrumentedStorage;
use crate::harness::session::types::BranchScan;
use crate::harness::session::types::BranchScanOrder;
use crate::harness::session::types::CheckpointData;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::Control;
use crate::harness::session::types::DurableStructuralPreparation;
use crate::harness::session::types::Entry;
use crate::harness::session::types::EntryScan;
use crate::harness::session::types::EntryStructure;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::ModelIdentity;
use crate::harness::session::types::NavigationReadyToCommitOperation;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::NormalizedRetryPolicy;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationMeta;
use crate::harness::session::types::OperationResultRecord;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::RunSettings;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionStats;
use crate::harness::session::types::Storage;
use crate::harness::session::types::StorageBranchScan;
use crate::harness::session::types::SummaryContext;
use crate::harness::session::types::SummaryDecidingOperation;
use crate::harness::session::types::SummaryEffectPendingOperation;
use crate::harness::session::types::SummaryGenerationScope;
use crate::harness::session::types::SummaryReadyOperation;
use crate::harness::session::types::{Session, SessionReader};
use crate::harness::session::types::{SummaryTask, UsageRow, UsageScan, operation_scope_of};
use crate::harness::session::values::ListAddress;
use crate::harness::session::values::ListElement;
use crate::harness::session::values::ListReadOptions;
use crate::harness::session::values::StoredValue;
use crate::harness::session::values::ValueAddress;
use crate::harness::session::values::Write;
use crate::harness::session::values::branch_tip;
use crate::harness::session::values::operation_meta;
use crate::harness::session::values::operation_preparation;
use crate::harness::session::values::{operation_state, pending_entry, set_value_write};
use crate::harness::types::{AgentHarnessResources, AgentHarnessStreamOptions};
use crate::types::{AgentMessage, QueueMode, ThinkingLevel, ToolExecutionMode};

/// The operation id the shared suites seed, upstream's per-file
/// `operationId` constant.
pub(super) const OPERATION_ID: &str = "01950000-0000-7000-8000-000000000001";

/// The read-side `Storage` forwards the memory-delegating fixtures share:
/// each fixture wrapper embeds a `memory: Arc<MemoryStorage>` delegate and
/// expands this inside its `impl Storage`, overriding only the methods it
/// instruments. Expanded at the use site so the delegate field's owner is
/// the wrapper, mirroring `session::testing`'s `storage_forwards!`.
macro_rules! memory_storage_forwards {
    () => {
        fn get_value(
            &self,
            address: &ValueAddress,
            context: &Context,
        ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
            self.memory.get_value(address, context)
        }

        fn scan_values(
            &self,
            prefix: &ValueAddress,
            context: &Context,
        ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
            self.memory.scan_values(prefix, context)
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

        memory_storage_tail!();

        fn get_stats(
            &self,
            context: &Context,
        ) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
            self.memory.get_stats(context)
        }

        fn close(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
            self.memory.close(context)
        }
    };
}

/// The fixed storage clock the drive fixtures inject, upstream's
/// `new MemoryStorage({ now: () => 100 })`: every entry timestamp renders
/// `100`, independent of the test clock.
fn fixed_clock() -> NowFn {
    Arc::new(|| 100)
}

/// The memory backend the drive fixtures default to, upstream's
/// `new MemoryStorage({ now: () => 100 })`; the tools suite wraps it in the
/// observing decorator.
pub(super) fn fixed_clock_memory_storage() -> Arc<MemoryStorage> {
    Arc::new(MemoryStorage::new(MemoryStorageOptions {
        now: Some(fixed_clock()),
    }))
}

/// The transcript conversion the drive fixtures install, upstream's
/// `toProviderMessages: (messages) => messages.filter(...)` filtering the
/// user, assistant, and toolResult roles through.
fn provider_message_filter() -> ToProviderMessages {
    Arc::new(|messages: &[AgentMessage], _context: &Context| {
        Box::pin(async move {
            messages
                .iter()
                .filter_map(|message| match message {
                    AgentMessage::Standard(
                        message @ (Message::User(_)
                        | Message::Assistant(_)
                        | Message::ToolResult(_)),
                    ) => Some(message.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
    })
}

/// The runtime configuration the drive fixtures read, upstream's per-file
/// `config` literal: the given retry policy and stream options over the
/// default compaction/queue settings and the filtered transcript
/// conversion.
#[must_use]
pub(super) fn drive_config(
    retry_policy: RetryPolicy,
    stream_options: AgentHarnessStreamOptions,
    system_prompt: Option<crate::harness::agent_harness::SystemPromptSource>,
) -> Config {
    Config {
        tools: Vec::new(),
        resources: AgentHarnessResources::default(),
        stream_options,
        retry_policy,
        compaction: DEFAULT_COMPACTION_SETTINGS,
        steering_mode: QueueMode::All,
        follow_up_mode: QueueMode::All,
        tool_execution: ToolExecutionMode::Parallel,
        tool_context: None,
        system_prompt,
        to_provider_messages: provider_message_filter(),
        entry_projectors: std::collections::BTreeMap::new(),
    }
}

/// The event collector the fixtures install, upstream's
/// `(batch) => { events.push(...structuredClone(batch)); }`: every batch's
/// events append to the shared list.
#[must_use]
pub(super) fn collecting_emit_batch(events: Arc<Mutex<Vec<HarnessEvent>>>) -> EmitBatch {
    Arc::new(move |batch: Vec<HarnessEvent>, _context: Context| {
        lock(&events).extend(batch);
        Box::pin(std::future::ready(Ok(())))
    })
}

/// The memory backend that parks one assistant-frame commit, upstream's
/// `FrameBlockingMemoryStorage`: the armed flag parks the first commit
/// carrying a `list`/`append` write on `pi.pending.assistant_frame` —
/// signaling the started gate first — and clears the flag, so the caller
/// observes the provider event loop making progress while frame storage
/// is parked, then releases.
pub(super) struct FrameBlockingMemoryStorage {
    memory: Arc<MemoryStorage>,
    block_next_frame: AtomicBool,
    frame_started: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    frame_signal: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    release_await: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    release_signal: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

impl FrameBlockingMemoryStorage {
    /// A frame-blocking backend over the memory storage.
    #[must_use]
    pub(super) fn new(delegate: Arc<MemoryStorage>) -> Self {
        let (frame_sender, frame_receiver) = deferred();
        let (release_sender, release_receiver) = deferred();
        Self {
            memory: delegate,
            block_next_frame: AtomicBool::new(false),
            frame_started: Mutex::new(Some(frame_receiver)),
            frame_signal: Mutex::new(Some(frame_sender)),
            release_await: Mutex::new(Some(release_receiver)),
            release_signal: Mutex::new(Some(release_sender)),
        }
    }

    /// Arms the next frame-list commit's park, upstream's
    /// `blockNextFrame = true` assignment.
    pub(super) fn arm_block_next_frame(&self) {
        self.block_next_frame.store(true, Ordering::SeqCst);
    }

    /// The parked frame commit's signal, upstream's
    /// `await backend.frameStarted.promise`; the backend parks once, so
    /// the receiver is taken once.
    pub(super) fn take_frame_started(&self) -> tokio::sync::oneshot::Receiver<()> {
        lock(&self.frame_started)
            .take()
            .expect("the frame park signals once")
    }

    /// Releases the parked frame commit, upstream's
    /// `backend.releaseFrame.resolve()`.
    pub(super) fn release_frame(&self) {
        let release = lock(&self.release_signal).take();
        if let Some(release) = release {
            let _ = release.send(());
        }
    }
}

impl Storage for FrameBlockingMemoryStorage {
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        let parks = self.block_next_frame.load(Ordering::SeqCst)
            && writes.iter().any(|write| {
                matches!(write, Write::ListAppend(list)
                    if list.kind == "list"
                        && list.op == "append"
                        && list.namespace == "pi.pending.assistant_frame")
            });
        if !parks {
            return self.memory.commit(writes, context);
        }
        self.block_next_frame.store(false, Ordering::SeqCst);
        let frame_signal = lock(&self.frame_signal).take();
        let release = lock(&self.release_await)
            .take()
            .expect("the frame release gate");
        let memory = Arc::clone(&self.memory);
        let context = context.clone();
        Box::pin(async move {
            if let Some(frame_signal) = frame_signal {
                let _ = frame_signal.send(());
            }
            let _ = release.await;
            memory.commit(writes, &context).await
        })
    }

    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<std::collections::BTreeMap<String, Entry>, SessionError>> {
        self.memory.get_entries(ids, context)
    }

    memory_storage_forwards!();
}

/// The memory backend that records commit families, upstream's
/// `ObservedMemoryStorage`: after every forwarded commit it appends
/// `intent_commit` when a `value`/`set` write targets `pi.op.tool_args`,
/// `outcome_commit` when a `value`/`set` targets `pi.pending.entry`, and
/// `replay_commit` when no outcome was staged but a `value`/`delete`
/// targets `pi.pending.tool_output`. The tools suite's emit hook also
/// records event types into the same list.
pub(super) struct ObservedMemoryStorage {
    memory: Arc<MemoryStorage>,
    observations: Arc<Mutex<Vec<String>>>,
}

impl ObservedMemoryStorage {
    /// An observing backend over the memory storage.
    #[must_use]
    pub(super) fn new(delegate: Arc<MemoryStorage>) -> Self {
        Self {
            memory: delegate,
            observations: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The recorded observations in commit order, upstream's
    /// `storage.observations` reads.
    #[must_use]
    pub(super) fn observations(&self) -> Vec<String> {
        lock(&self.observations).clone()
    }

    /// Records the emit hook's entries, upstream's
    /// `storage.observations.push(...batch.map((event) => event.type))`.
    pub(super) fn record_observations(&self, recorded: Vec<String>) {
        lock(&self.observations).extend(recorded);
    }
}

impl Storage for ObservedMemoryStorage {
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        let memory = Arc::clone(&self.memory);
        let observations = Arc::clone(&self.observations);
        let context = context.clone();
        Box::pin(async move {
            let result = memory.commit(writes.clone(), &context).await;
            let mut recorded: Vec<String> = Vec::new();
            if writes.iter().any(|write| {
                matches!(write, Write::ValueSet(value)
                    if value.kind == "value" && value.op == "set" && value.namespace == "pi.op.tool_args")
            }) {
                recorded.push("intent_commit".to_owned());
            }
            let stages_outcome = writes.iter().any(|write| {
                matches!(write, Write::ValueSet(value)
                    if value.kind == "value" && value.op == "set" && value.namespace == "pi.pending.entry")
            });
            if stages_outcome {
                recorded.push("outcome_commit".to_owned());
            }
            if !stages_outcome
                && writes.iter().any(|write| {
                    matches!(write, Write::ValueDelete(value)
                        if value.kind == "value" && value.op == "delete"
                            && value.namespace == "pi.pending.tool_output")
                })
            {
                recorded.push("replay_commit".to_owned());
            }
            lock(&observations).extend(recorded);
            result
        })
    }

    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<std::collections::BTreeMap<String, Entry>, SessionError>> {
        self.memory.get_entries(ids, context)
    }

    memory_storage_forwards!();
}

/// The fixture skeleton the hand-seeded drive suites share, upstream's
/// per-file `Fixture` interfaces.
pub(super) struct DriveFixture {
    /// The lane the procedures drive, upstream's `lane`.
    pub lane: Arc<Lane>,
    /// The installed pass, upstream's `drive`.
    pub drive: Arc<Drive>,
    /// The session the lane runs on, upstream's `session`.
    pub session: Arc<StorageBackedSession>,
    /// The instrumented storage recording commit attempts, upstream's
    /// `storage`.
    pub storage: Arc<InstrumentedStorage>,
    /// The models catalog, upstream's `models`.
    pub models: Arc<Models>,
    /// The faux provider handle, upstream's `faux`.
    pub faux: FauxProviderHandle,
    /// The hook registry, upstream's `hooks`.
    pub hooks: Arc<HookRegistry>,
    /// The collected events, upstream's `events`.
    pub events: Arc<Mutex<Vec<HarnessEvent>>>,
    /// The lane configuration the seed committed, upstream's
    /// `configuration`.
    pub configuration: LaneConfiguration,
    /// The runtime configuration the lane reads, upstream's `config`.
    pub config: Config,
    /// The operation id the suite seeds, upstream's `operationId`.
    pub operation_id: String,
}

/// The per-suite fixture differences, upstream's per-file `createFixture`
/// bodies.
pub(super) struct DriveFixtureSpec {
    /// The session-id prefix, upstream's `structural-`/`generation-`/
    /// `reconcile-`/`retry-deferred-`.
    pub suite: &'static str,
    /// The unused-watch raise's suite words, upstream's `unusedWatch`
    /// throw: "structural", "generation", "reconciliation", and
    /// "retry/deferred".
    pub watch_suite: &'static str,
    /// The faux provider's registration options, upstream's `fauxProvider`
    /// argument.
    pub faux: RegisterFauxProviderOptions,
    /// The config's retry policy, upstream's `retryPolicy` literal.
    pub retry_policy: RetryPolicy,
    /// The config's stream options, upstream's `streamOptions` literal.
    pub stream_options: AgentHarnessStreamOptions,
    /// The backend under the instrumented storage; `None` is the fixed
    /// `100` clock memory backend, upstream's `createFixture(backend)`
    /// default and the gated suites' pre-wrapped decorators.
    pub backend: Option<Arc<dyn Storage>>,
    /// Whether the fixture admits the `question` prompt through
    /// `lane.accept` (generation, retry/deferred) or leaves the lane idle
    /// for a hand-seeded install (structural, reconciliation).
    pub admit_prompt: bool,
    /// The config's system prompt, upstream's `systemPrompt` option; `None`
    /// is the empty prompt the plain fixtures carry.
    pub system_prompt: Option<crate::harness::agent_harness::SystemPromptSource>,
}

/// Builds one shared drive fixture, upstream's per-file `createFixture`:
/// the instrumented storage over the backend, the storage-backed session,
/// the faux provider registered into a fresh models catalog, the seeded
/// idle `main` lane, the lane over the restored state, the optional
/// prompt admission, and the pass installed with the attempts cleared.
///
/// # Panics
/// The seed commit's, restore's, or admission's failure.
pub(super) async fn create_drive_fixture(spec: DriveFixtureSpec) -> DriveFixture {
    let backend = spec.backend.unwrap_or_else(|| fixed_clock_memory_storage());
    let storage = Arc::new(InstrumentedStorage::new(backend));
    let session = Arc::new(StorageBackedSession::new(
        runtime_session_metadata(next_suite_session_id(spec.suite)),
        storage.clone(),
        StorageBackedSessionOptions::default(),
    ));
    let faux = faux_provider(spec.faux);
    let models = Arc::new(create_models(None));
    models.set_provider(Arc::new(faux.provider.clone()));
    let model = faux.first_model();
    let configuration = LaneConfiguration {
        model: ModelIdentity {
            provider: model.provider.0.clone(),
            model_id: model.id.clone(),
        },
        thinking_level: ThinkingLevel::Off,
        active_tool_names: Vec::new(),
    };
    commit_writes(
        &session,
        main_lane_seed_writes(&configuration).expect("seed writes"),
    )
    .await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let hooks = Arc::new(HookRegistry::new(noop_hook_reporter()));
    let config = drive_config(spec.retry_policy, spec.stream_options, spec.system_prompt);
    let restored = restore_lane(session.clone(), "main", &background_context())
        .await
        .expect("restore");
    let lane = Arc::new(Lane::new(
        "main",
        session.clone(),
        models.clone(),
        hooks.clone(),
        restored,
        passthrough_fault_handler(),
        collecting_emit_batch(Arc::clone(&events)),
        unused_watch(spec.watch_suite),
        Arc::new({
            let config = config.clone();
            move || config.clone()
        }),
    ));
    if spec.admit_prompt {
        lane.accept_impl(
            OperationRequest::Prompt {
                operation_id: Some(OPERATION_ID.to_owned()),
                prompt: Box::new(PromptMessagesPayload::Text {
                    prompt: "question".to_owned(),
                    images: None,
                }),
            },
            &background_context(),
        )
        .await
        .expect("accept serves")
        .expect("the admission");
    }
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: OPERATION_ID.to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    lane.set_active_drive(Some(Arc::clone(&drive)));
    storage.clear_commit_attempts();
    DriveFixture {
        lane,
        drive,
        session,
        storage,
        models,
        faux,
        hooks,
        events,
        configuration,
        config,
        operation_id: OPERATION_ID.to_owned(),
    }
}

/// The running scope the hand-seeded leaves carry, upstream's per-file
/// `runScope(compaction = DEFAULT_COMPACTION_SETTINGS)`.
#[must_use]
pub(super) fn run_scope(compaction: CompactionSettings) -> OperationScope {
    scope_with_control(Control::Running, compaction)
}

/// The scope variant the cancellation fixtures carry, upstream's
/// `scope(control = cancelledControl())`.
#[must_use]
pub(super) fn scope_with_control(
    control: Control,
    compaction: CompactionSettings,
) -> OperationScope {
    OperationScope {
        control,
        settings: RunSettings {
            compaction,
            steering_mode: QueueMode::All,
            follow_up_mode: QueueMode::All,
            tool_execution: ToolExecutionMode::Parallel,
        },
        latest_assistant_entry_id: None,
    }
}

/// The cancellation control the reconciliation fixtures seed, upstream's
/// `cancelledControl()`.
#[must_use]
pub(super) const fn cancelled_control() -> Control {
    Control::CancelRequested { requested_at: 10 }
}

/// The standalone compaction task, upstream's
/// `standaloneCompactionTask(customInstructions?)`.
#[must_use]
pub(super) fn standalone_compaction_task(custom_instructions: Option<&str>) -> SummaryTask {
    SummaryTask {
        task_id: "task".to_owned(),
        reason: Some(CompactionReason::Manual),
        custom_instructions: custom_instructions.map(str::to_owned),
        boundary: ResultBoundary::Finish,
    }
}

/// The run-compaction task, upstream's `runCompactionTask(reason,
/// resumeAfter)`.
#[must_use]
pub(super) fn run_compaction_task(
    reason: CompactionReason,
    resume_after: CheckpointData,
) -> SummaryTask {
    SummaryTask {
        task_id: "task".to_owned(),
        reason: Some(reason),
        custom_instructions: None,
        boundary: ResultBoundary::ResumeCheckpoint { resume_after },
    }
}

/// The navigation summary task, upstream's
/// `navigationSummaryTask(targetId, label?)`.
#[must_use]
pub(super) fn navigation_summary_task(target_id: &str, label: Option<&str>) -> SummaryTask {
    SummaryTask {
        task_id: "task".to_owned(),
        reason: None,
        custom_instructions: None,
        boundary: ResultBoundary::CommitNavigation {
            target_id: target_id.to_owned(),
            label: label.map(str::to_owned),
        },
    }
}

/// The summary generation inputs the summary leaves carry, upstream's
/// `summaryContext(configuration)` and `summaryReady`'s default
/// `summaryContext`: the `summary-entry` result id and the
/// `{ maxAttempts: 2, baseDelayMs: 10, maxAgentDelayMs: 30_000 }` policy.
#[must_use]
pub(super) fn summary_context(configuration: &LaneConfiguration) -> SummaryContext {
    SummaryContext {
        result_entry_id: "summary-entry".to_owned(),
        configuration: configuration.clone(),
        stream_options: AgentHarnessStreamOptions::default(),
        retry_policy: NormalizedRetryPolicy {
            max_attempts: 2,
            base_delay_ms: 10,
            max_agent_delay_ms: 30_000,
        },
    }
}

/// The summary-ready leaf, upstream's `summaryReady(scope, task,
/// configuration, retryPolicy = ...)` with the default policy in the
/// summary context; tests needing another policy overwrite the leaf's
/// `retry_policy` field.
#[must_use]
pub(super) fn summary_ready(
    scope: OperationScope,
    task: SummaryTask,
    configuration: &LaneConfiguration,
) -> SummaryReadyOperation {
    SummaryReadyOperation {
        scope,
        generation: SummaryGenerationScope {
            task,
            summary_context: summary_context(configuration),
        },
        next_attempt: 1,
    }
}

/// The durable compaction preparation, upstream's
/// `compactionPreparation(overrides?)`; tests needing overrides mutate the
/// returned variant's fields in place.
#[must_use]
pub(super) fn compaction_preparation() -> DurableStructuralPreparation {
    DurableStructuralPreparation::Compaction {
        messages_to_summarize: vec![user("history", 1)],
        turn_prefix_messages: Vec::new(),
        retained_tail: vec![user("tail", 1)],
        is_split_turn: false,
        tokens_before: 1_000,
        previous_summary: None,
        file_ops: create_file_ops(),
        settings: CompactionSettings {
            enabled: true,
            reserve_tokens: 1_000,
            keep_recent_tokens: 10,
        },
    }
}

/// The tip-history user entry, upstream's per-case `userEntry("tip")`
/// fixture row; the id/parent/content parameters cover the variants.
#[must_use]
pub(super) fn user_entry(id: &str, parent_id: Option<&str>, content: &str) -> NewEntry {
    NewEntry::Message {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        body: Box::new(MessageEntry {
            message: user(content, 1),
            terminate: None,
        }),
    }
}

/// The assistant history entry, upstream's `assistantEntry(id, parent,
/// message)` fixture row.
#[must_use]
pub(super) fn assistant_entry(
    id: &str,
    parent_id: Option<&str>,
    message: pi_ai::types::AssistantMessage,
) -> NewEntry {
    NewEntry::Message {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        body: Box::new(MessageEntry {
            message: AgentMessage::Standard(Message::Assistant(message)),
            terminate: None,
        }),
    }
}

/// The deciding leaf a threshold-driven run carries, upstream's per-case
/// `SummaryDecidingOperation` fixture: the run scope, the threshold task
/// over the `tip` trigger, and the case's continuation.
#[must_use]
pub(super) fn deciding_run(continuation: Continuation) -> SummaryDecidingOperation {
    SummaryDecidingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        task: run_compaction_task(
            CompactionReason::Threshold,
            CheckpointData {
                continuation,
                trigger_entry_id: "tip".to_owned(),
            },
        ),
    }
}

/// The navigation-tree install options, the navigation cases' fixture:
/// the `root`/`source`/`target` entries, the `source` tip, and the
/// `task` branch preparation.
#[must_use]
pub(super) fn navigation_install_options() -> InstallOptions {
    InstallOptions {
        entries: vec![
            user_entry("root", None, "root"),
            user_entry("source", Some("root"), "source"),
            user_entry("target", Some("root"), "target"),
        ],
        tip_id: Some(Some("source".to_owned())),
        preparation: Some(PreparationWrite {
            task_id: "task".to_owned(),
            value: branch_preparation(),
        }),
        ..InstallOptions::default()
    }
}

/// The tip-history install options with the `task` compaction preparation,
/// upstream's per-case `installOperation` options table.
#[must_use]
pub(super) fn tip_install_options() -> InstallOptions {
    InstallOptions {
        entries: vec![user_entry("tip", None, "history")],
        preparation: Some(PreparationWrite {
            task_id: "task".to_owned(),
            value: compaction_preparation(),
        }),
        ..InstallOptions::default()
    }
}

/// The deciding leaf a standalone compaction carries, upstream's
/// per-case `SummaryDecidingOperation` fixture over the manual task.
#[must_use]
pub(super) fn deciding_standalone() -> SummaryDecidingOperation {
    SummaryDecidingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        task: standalone_compaction_task(None),
    }
}

/// Installs one deciding run leaf over the `tip` history, the
/// threshold-decline cases' `installOperation` call.
pub(super) async fn install_deciding_run(
    fixture: &DriveFixture,
    deciding: &SummaryDecidingOperation,
) {
    install_operation(
        fixture,
        OperationState::SummaryDeciding(deciding.clone()),
        OperationIntent::Run {
            prompt_entry_ids: vec!["tip".to_owned()],
        },
        tip_install_options(),
    )
    .await;
}

/// Installs one deciding compaction leaf over the `tip` history, the
/// standalone-compaction cases' `installOperation` call.
pub(super) async fn install_deciding_compaction(
    fixture: &DriveFixture,
    deciding: &SummaryDecidingOperation,
) {
    install_operation(
        fixture,
        OperationState::SummaryDeciding(deciding.clone()),
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        tip_install_options(),
    )
    .await;
}

/// Installs the standalone `summary.ready` leaf over the `tip` history,
/// the per-case fixture of the standalone structural suite, and returns
/// the ready leaf the case drives.
pub(super) async fn install_standalone_summary_ready(
    fixture: &DriveFixture,
) -> SummaryReadyOperation {
    let ready = summary_ready(
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        standalone_compaction_task(None),
        &fixture.configuration,
    );
    install_operation(
        fixture,
        OperationState::SummaryReady(ready.clone()),
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        tip_install_options(),
    )
    .await;
    ready
}

/// Installs the standalone `summary.ready` leaf with the case's install
/// options, the preparation-boundary cases' fixture, and returns the ready
/// leaf the case drives.
pub(super) async fn install_standalone_summary_ready_with(
    fixture: &DriveFixture,
    options: InstallOptions,
) -> SummaryReadyOperation {
    let ready = summary_ready(
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        standalone_compaction_task(None),
        &fixture.configuration,
    );
    install_operation(
        fixture,
        OperationState::SummaryReady(ready.clone()),
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        options,
    )
    .await;
    ready
}

/// Installs the standalone compaction's deciding leaf with the case's
/// install options, and returns the deciding leaf the case drives.
pub(super) async fn install_deciding_standalone_with(
    fixture: &DriveFixture,
    options: InstallOptions,
) -> SummaryDecidingOperation {
    let deciding = deciding_standalone();
    install_operation(
        fixture,
        OperationState::SummaryDeciding(deciding.clone()),
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        options,
    )
    .await;
    deciding
}

/// Installs the navigation summary-deciding leaf over the target/source
/// branch, the navigation cases' `installOperation` prelude, and returns
/// the deciding leaf the case drives.
pub(super) async fn install_navigation_deciding(
    fixture: &DriveFixture,
) -> SummaryDecidingOperation {
    let deciding = SummaryDecidingOperation {
        scope: run_scope(DEFAULT_COMPACTION_SETTINGS),
        task: navigation_summary_task("target", None),
    };
    install_operation(
        fixture,
        OperationState::SummaryDeciding(deciding.clone()),
        OperationIntent::Navigation {
            target_id: Some("target".to_owned()),
            summarize: true,
            label: None,
            custom_instructions: None,
        },
        navigation_install_options(),
    )
    .await;
    deciding
}

/// Installs the navigation summary-ready leaf over the target/source
/// branch, the navigation generation cases' `installOperation` prelude,
/// and returns the ready leaf the case drives.
pub(super) async fn install_navigation_summary_ready(
    fixture: &DriveFixture,
) -> SummaryReadyOperation {
    let ready = summary_ready(
        run_scope(DEFAULT_COMPACTION_SETTINGS),
        navigation_summary_task("target", None),
        &fixture.configuration,
    );
    install_operation(
        fixture,
        OperationState::SummaryReady(ready.clone()),
        OperationIntent::Navigation {
            target_id: Some("target".to_owned()),
            summarize: true,
            label: None,
            custom_instructions: None,
        },
        navigation_install_options(),
    )
    .await;
    ready
}

/// Installs the effect-pending compaction leaf over the `tip` history, the
/// orphaned-attempt cases' `installOperation` call.
pub(super) async fn install_effect_pending_compaction(
    fixture: &DriveFixture,
    effect: &SummaryEffectPendingOperation,
) {
    install_operation(
        fixture,
        OperationState::SummaryEffectPending(effect.clone()),
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        tip_install_options(),
    )
    .await;
}

/// The durable branch-summary preparation, upstream's `branchPreparation()`.
#[must_use]
pub(super) fn branch_preparation() -> DurableStructuralPreparation {
    DurableStructuralPreparation::BranchSummary {
        messages: vec![user("abandoned", 1)],
        file_ops: create_file_ops(),
        total_tokens: 10,
    }
}

/// The optional preparation write the installs commit, upstream's
/// `options.preparation = { taskId, value }`.
#[derive(Clone)]
pub(super) struct PreparationWrite {
    /// The task id the preparation is keyed under.
    pub task_id: String,
    /// The durable preparation value.
    pub value: DurableStructuralPreparation,
}

/// The hand-seeded install's options, upstream's
/// `installOperation(fixture, state, intent, options)` options object.
#[derive(Default)]
pub(super) struct InstallOptions {
    /// The entries to insert ahead of the operation's writes; the last
    /// entry's id is the default tip.
    pub entries: Vec<NewEntry>,
    /// The seeded tip override; `None` derives from the entries (or null).
    #[expect(
        clippy::option_option,
        reason = "upstream's tipId is string | null | undefined; the double option restates the three states"
    )]
    pub tip_id: Option<Option<String>>,
    /// Extra writes committed between the tip and the operation's writes,
    /// upstream's `installed.writes`.
    pub extra_writes: Vec<Write>,
    /// The optional preparation write, upstream's `options.preparation`.
    pub preparation: Option<PreparationWrite>,
    /// Whether the lane record keeps the projection's
    /// `lastOperationId` (reconciliation) or clears it (structural).
    pub keep_last_operation_id: bool,
}

/// Hand-seeds one durable operation leaf, upstream's
/// `installOperation`: one `lane.command` commits the entries, the branch
/// tip, the operation meta and state, the lane record pointing the
/// operation at the projection's inbox, and the optional preparation;
/// then the commit attempts clear.
///
/// # Panics
/// The command's or a write's serialization failure.
pub(super) async fn install_operation(
    fixture: &DriveFixture,
    state: OperationState,
    intent: OperationIntent,
    options: InstallOptions,
) {
    let tip_id = options
        .tip_id
        .unwrap_or_else(|| options.entries.last().map(|entry| entry.id().to_owned()));
    let meta = OperationMeta {
        operation_id: fixture.operation_id.clone(),
        lane: "main".to_owned(),
        source_tip_id: tip_id.clone(),
        started_at: 1,
        intent,
    };
    let operation_id = fixture.operation_id.clone();
    let entries = options.entries;
    let extra_writes = options.extra_writes;
    let preparation = options.preparation;
    let keep_last_operation_id = options.keep_last_operation_id;
    fixture
        .lane
        .command::<(), _>(
            move |projection, _session, _context| {
                let entries = entries.clone();
                let extra_writes = extra_writes.clone();
                let tip_id = tip_id.clone();
                let meta = meta.clone();
                let state = state.clone();
                let operation_id = operation_id.clone();
                let preparation = preparation.clone();
                Box::pin(async move {
                    let mut writes: Vec<Write> = entries
                        .iter()
                        .map(|entry| Write::Entry(Box::new(insert_entry(entry.clone()))))
                        .collect();
                    writes.push(
                        set_value_write(&branch_tip("main"), tip_id.clone()).expect("tip write"),
                    );
                    writes.extend(extra_writes.iter().cloned());
                    writes.push(
                        set_value_write(&operation_meta(&operation_id), meta.clone())
                            .expect("meta write"),
                    );
                    writes.push(
                        set_value_write(&operation_state(&operation_id), state.clone())
                            .expect("state write"),
                    );
                    let last_operation_id = if keep_last_operation_id {
                        projection.last_operation_id.clone()
                    } else {
                        None
                    };
                    writes.push(
                        lane_state_write(
                            "main",
                            Some(&operation_id),
                            last_operation_id.as_deref(),
                            projection.inbox.clone(),
                        )
                        .expect("lane state write"),
                    );
                    if let Some(preparation) = &preparation {
                        writes.push(
                            set_value_write(
                                &operation_preparation(&operation_id, &preparation.task_id),
                                preparation.value.clone(),
                            )
                            .expect("preparation write"),
                        );
                    }
                    let next = LaneState {
                        tip_id: tip_id.clone(),
                        operation: Some(LiveOperation {
                            meta: meta.clone(),
                            state: state.clone(),
                        }),
                        ..projection.clone()
                    };
                    Ok(LaneCommand::Commit {
                        writes,
                        next,
                        materialize: Arc::new(|_commit| ()),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the install commits");
    fixture.storage.clear_commit_attempts();
}

/// Replaces the live operation's control with `cancel_requested`, upstream's
/// `cancelOperation`.
///
/// # Panics
/// The absence of a live operation or the commit's failure.
pub(super) async fn cancel_operation(fixture: &DriveFixture) {
    let operation = fixture
        .lane
        .state()
        .operation
        .expect("fixture has no operation");
    let state = operation_state_with_scope(
        &operation.state,
        OperationScope {
            control: Control::CancelRequested { requested_at: 2 },
            ..operation_scope_of(&operation.state)
        },
    );
    let operation_id = fixture.operation_id.clone();
    let meta = operation.meta.clone();
    fixture
        .lane
        .command::<(), _>(
            move |projection, _session, _context| {
                let meta = meta.clone();
                let state = state.clone();
                let operation_id = operation_id.clone();
                Box::pin(async move {
                    let next = LaneState {
                        operation: Some(LiveOperation {
                            meta: meta.clone(),
                            state: state.clone(),
                        }),
                        ..projection.clone()
                    };
                    Ok(LaneCommand::Commit {
                        writes: vec![
                            set_value_write(&operation_state(&operation_id), state.clone())
                                .expect("cancel write"),
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
        .expect("the cancel commits");
}

/// Drives the operation and unwraps the settled record, the pass-chasing
/// suites' settled read; a waiting or failing pass panics the test by
/// design.
pub(super) async fn settled_pass(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    serves_why: &str,
    settles_why: &str,
) -> OperationResultRecord {
    let outcome = drive_operation(lane, drive).await.expect(serves_why);
    let DriveOutcome::Settled { outcome } = outcome else {
        unreachable!("{settles_why}: {outcome:?}")
    };
    outcome
}

/// Registers the hook that aborts the pass through the drive's gate
/// signal, upstream's `gate.admit` throwing `AbortRequested` inside the
/// hook body; the hook returns the given no-op result.
pub(super) fn register_aborting_hook(fixture: &DriveFixture, name: HookName, result: HookResult) {
    let drive_for_hook = Arc::clone(&fixture.drive);
    fixture
        .hooks
        .on(
            name,
            Arc::new(move |_invocation: &HookInvocation, _context: &Context| {
                let drive = Arc::clone(&drive_for_hook);
                let result = result.clone();
                Box::pin(async move {
                    let (_, cancel_rx) = tokio::sync::watch::channel(());
                    drive.begin_abort(cancel_rx);
                    drive.signal_abort();
                    Ok::<HookResult, HookFailure>(result)
                })
            }),
            HookOptions::default(),
        )
        .expect("the aborting hook registers");
}

/// The stored entry, the entry reads the suites pin; a missing entry
/// panics the test by design.
pub(super) async fn entry_stored(
    session: &Arc<StorageBackedSession>,
    entry_id: &str,
    why: &str,
) -> Entry {
    session
        .get_entry(entry_id, &background_context())
        .await
        .expect("the entry reads")
        .expect(why)
}

/// The stored assistant-response message, the response-entry reads the
/// generation suites pin; a non-message or non-assistant entry panics the
/// test by design.
pub(super) async fn assistant_message_entry(
    session: &Arc<StorageBackedSession>,
    entry_id: &str,
    why: &str,
) -> pi_ai::types::AssistantMessage {
    let entry = entry_stored(session, entry_id, why).await;
    let Entry::Message { body, .. } = entry else {
        panic!("the response entry is a message entry");
    };
    let AgentMessage::Standard(Message::Assistant(message)) = body.message else {
        panic!("the response entry carries an assistant message");
    };
    message
}

/// The stored tool-result message, the result-entry reads the tools
/// suites pin; a non-message or non-result entry panics the test by
/// design.
pub(super) async fn tool_result_message(
    session: &Arc<StorageBackedSession>,
    entry_id: &str,
    why: &str,
) -> pi_ai::types::ToolResultMessage {
    let entry = entry_stored(session, entry_id, why).await;
    let Entry::Message { body, .. } = entry else {
        panic!("the entry is a message");
    };
    let AgentMessage::Standard(Message::ToolResult(result)) = body.message else {
        panic!("the entry carries the tool result");
    };
    result
}

/// Scans the lane's oldest-first transcript, upstream's `findEntries`
/// oldest-first reads.
pub(super) async fn transcript(lane: &Lane) -> Vec<Entry> {
    lane.find_entries_impl(
        Some(&BranchScan {
            order: Some(BranchScanOrder::OldestFirst),
            ..BranchScan::default()
        }),
        &background_context(),
    )
    .await
    .expect("the transcript scans")
}

/// The oldest-first transcript's entry ids, the placement reads the suites
/// pin.
pub(super) async fn transcript_ids(lane: &Lane) -> Vec<String> {
    transcript(lane)
        .await
        .iter()
        .map(|entry| entry.id().to_owned())
        .collect()
}

/// Closes one fixture's session, upstream's `afterEach` close loop.
///
/// # Panics
/// The close's failure.
pub(super) async fn close_session(fixture: &DriveFixture) {
    fixture
        .session
        .close(&background_context())
        .await
        .expect("the session closes");
}

/// The usage writes the storage recorded, the commit-attempts filter the
/// usage-settling suites count.
pub(super) fn usage_write_count(fixture: &DriveFixture) -> usize {
    fixture
        .storage
        .get_commit_attempts()
        .iter()
        .flatten()
        .filter(|write| matches!(write, Write::Usage(_)))
        .count()
}

/// The `Usage` events the fixture collected, the event-count reads the
/// usage suites pin.
pub(super) fn usage_event_count(fixture: &DriveFixture) -> usize {
    lock(&fixture.events)
        .iter()
        .filter(|event| {
            event.event_type() == crate::harness::agent_harness::HarnessEventType::Usage
        })
        .count()
}

/// Reads the stored structural preparation for the operation's task,
/// upstream's `session.getValue(storedValues.operationPreparation(...))`
/// reads; `None` when the task has none.
pub(super) async fn stored_preparation(
    fixture: &DriveFixture,
    task_id: &str,
) -> Option<StoredValue> {
    fixture
        .session
        .get_value(
            &operation_preparation(OPERATION_ID, task_id).address,
            &background_context(),
        )
        .await
        .expect("the preparation reads")
}

/// Reads the stored pending entry, upstream's
/// `session.getValue(storedValues.pendingEntry(...))` reads; `None` when
/// the entry is absent.
pub(super) async fn stored_pending_entry(
    fixture: &DriveFixture,
    entry_id: &str,
) -> Option<StoredValue> {
    fixture
        .session
        .get_value(&pending_entry(entry_id).address, &background_context())
        .await
        .expect("the pending entry reads")
}

/// The aborted assistant response the cancelled-request suites serve,
/// upstream's `stopReason: "aborted"` faux message over the empty content.
#[must_use]
pub(super) fn aborted_response() -> pi_ai::providers::faux::FauxResponseStep {
    pi_ai::providers::faux::FauxResponseStep::Message(
        pi_ai::providers::faux::faux_assistant_message(
            "",
            pi_ai::providers::faux::FauxAssistantMessageOptions {
                stop_reason: Some(pi_ai::types::StopReason::Aborted),
                error_message: Some("cancelled".to_owned()),
                ..pi_ai::providers::faux::FauxAssistantMessageOptions::default()
            },
        ),
    )
}

/// Installs one `SummaryReady` run leaf over the tip options, the ready
/// run install the structural suites share.
pub(super) async fn install_ready_run(fixture: &DriveFixture, ready: &SummaryReadyOperation) {
    install_operation(
        fixture,
        OperationState::SummaryReady(ready.clone()),
        OperationIntent::Run {
            prompt_entry_ids: vec!["tip".to_owned()],
        },
        tip_install_options(),
    )
    .await;
}

/// Installs one `SummaryReady` summarized-navigation leaf over the target
/// options, the navigation install the structural suites share.
pub(super) async fn install_navigation_ready(
    fixture: &DriveFixture,
    ready: &SummaryReadyOperation,
) {
    install_operation(
        fixture,
        OperationState::SummaryReady(ready.clone()),
        OperationIntent::Navigation {
            target_id: Some("target".to_owned()),
            summarize: true,
            label: None,
            custom_instructions: None,
        },
        navigation_install_options(),
    )
    .await;
}

/// Installs one `NavigationReadyToCommit` leaf over the source entries and
/// tip, the navigation-commit install the structural suites share.
pub(super) async fn install_navigation_commit_leaf(
    fixture: &DriveFixture,
    navigation: &NavigationReadyToCommitOperation,
) {
    install_operation(
        fixture,
        OperationState::NavigationReadyToCommit(navigation.clone()),
        OperationIntent::Navigation {
            target_id: Some("target".to_owned()),
            summarize: false,
            label: None,
            custom_instructions: None,
        },
        InstallOptions {
            entries: vec![user_entry("source", None, "source")],
            tip_id: Some(Some("source".to_owned())),
            ..InstallOptions::default()
        },
    )
    .await;
}

/// Runs the structural decision, the `runStructuralDecision` rows; an
/// unexpected outcome panics the test by design.
pub(super) async fn decision_serves(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    deciding: &SummaryDecidingOperation,
) -> ProcedureResult {
    run_structural_decision(lane, drive, deciding)
        .await
        .expect("the decision serves")
}

/// Runs the structural decision and pins the Continue routing; a wrong
/// routing panics the test by design.
pub(super) async fn decision_continues(
    fixture: &DriveFixture,
    deciding: &SummaryDecidingOperation,
    why: &str,
) {
    let result = decision_serves(&fixture.lane, &fixture.drive, deciding).await;
    assert!(
        matches!(result, ProcedureResult::Continue),
        "{why}: {result:?}"
    );
}

/// Runs the structural decision expecting its rejection, the
/// expect-throws rows; an unexpected serve panics the test by design.
pub(super) async fn decision_rejects(
    fixture: &DriveFixture,
    deciding: &SummaryDecidingOperation,
    why: &str,
) -> LaneError {
    run_structural_decision(&fixture.lane, &fixture.drive, deciding)
        .await
        .expect_err(why)
}

/// Runs the structural generation attempt, the `runStructuralGeneration`
/// rows; an unexpected failure panics the test by design.
pub(super) async fn generation_serves(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    ready: &SummaryReadyOperation,
) -> ProcedureResult {
    run_structural_generation(lane, drive, ready)
        .await
        .expect("the attempt serves")
}

/// Runs the structural generation attempt and pins the Continue routing;
/// a wrong routing panics the test by design.
pub(super) async fn generation_continues(
    fixture: &DriveFixture,
    ready: &SummaryReadyOperation,
    why: &str,
) {
    let result = generation_serves(&fixture.lane, &fixture.drive, ready).await;
    assert!(
        matches!(result, ProcedureResult::Continue),
        "{why}: {result:?}"
    );
}

/// Runs the structural generation attempt expecting its failure; an
/// unexpected serve panics the test by design.
pub(super) async fn generation_rejects(
    fixture: &DriveFixture,
    ready: &SummaryReadyOperation,
    why: &str,
) -> LaneError {
    run_structural_generation(&fixture.lane, &fixture.drive, ready)
        .await
        .expect_err(why)
}

/// Recovers the structural generation leaf, the
/// `recoverStructuralGeneration` rows; an unexpected failure panics the
/// test by design.
pub(super) async fn recovery_serves(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    effect: &SummaryEffectPendingOperation,
) -> ProcedureResult {
    recover_structural_generation(lane, drive, effect)
        .await
        .expect("the recovery serves")
}

/// Recovers the structural generation leaf and pins the Continue routing;
/// a wrong routing panics the test by design.
pub(super) async fn recovery_continues(
    fixture: &DriveFixture,
    effect: &SummaryEffectPendingOperation,
    why: &str,
) {
    let result = recovery_serves(&fixture.lane, &fixture.drive, effect).await;
    assert!(
        matches!(result, ProcedureResult::Continue),
        "{why}: {result:?}"
    );
}

/// Commits the navigation leaf, the `commitNavigation` rows; an
/// unexpected outcome panics the test by design.
pub(super) async fn navigation_serves(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    navigation: &NavigationReadyToCommitOperation,
) -> ProcedureResult {
    commit_navigation(lane, drive, navigation)
        .await
        .expect("the navigation serves")
}

/// Commits the navigation leaf and pins the Continue routing; a wrong
/// routing panics the test by design.
pub(super) async fn navigation_continues(
    fixture: &DriveFixture,
    navigation: &NavigationReadyToCommitOperation,
    why: &str,
) {
    let result = navigation_serves(&fixture.lane, &fixture.drive, navigation).await;
    assert!(
        matches!(result, ProcedureResult::Continue),
        "{why}: {result:?}"
    );
}

/// Commits the navigation leaf expecting its rejection; an unexpected
/// serve panics the test by design.
pub(super) async fn navigation_rejects(
    fixture: &DriveFixture,
    navigation: &NavigationReadyToCommitOperation,
    why: &str,
) -> LaneError {
    commit_navigation(&fixture.lane, &fixture.drive, navigation)
        .await
        .expect_err(why)
}

/// The live operation's durable state leaf, upstream's `currentState`:
/// throws `fixture has no operation` when the lane is idle.
///
/// # Panics
/// The absence of a live operation.
#[must_use]
pub(super) fn current_state(fixture: &DriveFixture) -> OperationState {
    fixture
        .lane
        .state()
        .operation
        .expect("fixture has no operation")
        .state
}

/// The projection-restore assertion, upstream's `expectProjectionRestores`:
/// the lane's owned state deep-equals a fresh `restoreLane` read.
///
/// # Panics
/// The restore's failure or the state mismatch.
pub(super) async fn expect_projection_restores(fixture: &DriveFixture) {
    let restored = restore_lane(fixture.session.clone(), "main", &background_context())
        .await
        .expect("restore");
    assert_eq!(
        fixture.lane.state(),
        restored,
        "the lane projection restores"
    );
}

/// Polls the condition to truth, upstream's `waitFor(predicate)`: 200
/// iterations of one yield-spaced check, then the
/// `condition was not reached` raise.
///
/// # Panics
/// The budget expiring.
pub(super) async fn wait_for(predicate: impl Fn() -> bool) {
    for _ in 0..200 {
        if predicate() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("condition was not reached");
}

/// The drive clock pin's restore guard: dropping it clears the pin so a
/// fake-timer suite cannot leak its frozen clock into a sibling test in an
/// in-process runner.
pub(super) struct ClockPin(());

impl Drop for ClockPin {
    fn drop(&mut self) {
        crate::harness::runtime::clock::reset();
    }
}

/// Pins the drive clock to `ms`, upstream's `vi.useFakeTimers()` +
/// `vi.setSystemTime(ms)`; the returned guard restores the wall clock.
/// Later pins restate upstream's later `vi.setSystemTime` calls and the
/// guard keeps restoring on drop.
pub(super) fn pin_clock(ms: i64) -> ClockPin {
    crate::harness::runtime::clock::set_test_now(ms);
    ClockPin(())
}

/// The memory backend that counts entry reads, the restatement of the
/// generation suite's `vi.spyOn(fixture.storage, "getEntries")`: the
/// instrumented storage forwards every read to its backend, so a counter
/// here observes every `get_entries` call the tested surface makes.
pub(super) struct EntryReadCountingStorage {
    memory: Arc<MemoryStorage>,
    entry_reads: AtomicUsize,
}

impl EntryReadCountingStorage {
    /// A counting backend over the memory storage.
    #[must_use]
    pub(super) fn new(delegate: Arc<MemoryStorage>) -> Self {
        Self {
            memory: delegate,
            entry_reads: AtomicUsize::new(0),
        }
    }

    /// The recorded entry-read count, upstream's spy-call count.
    #[must_use]
    pub(super) fn entry_reads(&self) -> usize {
        self.entry_reads.load(Ordering::SeqCst)
    }

    /// Resets the recorded count, upstream's spy resetting between phases.
    pub(super) fn reset_entry_reads(&self) {
        self.entry_reads.store(0, Ordering::SeqCst);
    }
}

impl Storage for EntryReadCountingStorage {
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        self.memory.commit(writes, context)
    }

    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<std::collections::BTreeMap<String, Entry>, SessionError>> {
        self.entry_reads.fetch_add(1, Ordering::SeqCst);
        self.memory.get_entries(ids, context)
    }

    memory_storage_forwards!();
}

/// Queues pending entries onto the lane's inbox, the hand-seeded suites'
/// per-case `lane.command` blocks queuing `pendingEntry` writes: one commit
/// carries each pending entry plus the lane record pointing the operation
/// at the extended inbox, and the projection's inbox grows in call order.
///
/// # Panics
/// A write's serialization failure or the command's failure.
pub(super) async fn queue_pending_entries(
    fixture: &DriveFixture,
    pending: Vec<(&str, PendingEntry, InboxItemKind)>,
) {
    let operation_id = fixture.operation_id.clone();
    let pending: Vec<(String, PendingEntry, InboxItemKind)> = pending
        .into_iter()
        .map(|(entry_id, entry, kind)| (entry_id.to_owned(), entry, kind))
        .collect();
    fixture
        .lane
        .command::<(), _>(
            move |projection, _session, _context| {
                let operation_id = operation_id.clone();
                let pending = pending.clone();
                Box::pin(async move {
                    let mut inbox = projection.inbox.clone();
                    let mut writes = Vec::new();
                    for (entry_id, entry, kind) in &pending {
                        writes.push(
                            set_value_write(&pending_entry(entry_id), entry.clone())
                                .expect("pending write"),
                        );
                        inbox.push(InboxItem {
                            entry_id: entry_id.clone(),
                            kind: *kind,
                        });
                    }
                    writes.push(
                        lane_state_write("main", Some(&operation_id), None, inbox.clone())
                            .expect("lane state write"),
                    );
                    let next = LaneState {
                        inbox,
                        ..projection.clone()
                    };
                    Ok(LaneCommand::Commit {
                        writes,
                        next,
                        materialize: Arc::new(|_commit| ()),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the queue commits");
}

/// The silent commit arm the hand-seeded commands share, upstream's
/// `Commit { writes, next, materialize: () }` literals: no events and no
/// materialization.
pub(super) fn silent_commit(writes: Vec<Write>, next: LaneState) -> LaneCommand<()> {
    LaneCommand::Commit {
        writes,
        next,
        materialize: Arc::new(|_: &CommitResult| ()),
        events: None,
    }
}

/// The run intent over the `tip` history, the run-leaved cases' intent.
pub(super) fn run_tip_intent() -> OperationIntent {
    OperationIntent::Run {
        prompt_entry_ids: vec!["tip".to_owned()],
    }
}

/// The tip-history install options, the run-leaved cases' options.
pub(super) fn run_tip_install_options() -> InstallOptions {
    InstallOptions {
        entries: vec![user_entry("tip", None, "history")],
        ..InstallOptions::default()
    }
}

/// Begins the pass's abort with the watch pre-closed, upstream's dropped
/// `sender` followed by `beginAbort(cancellation)` + `signalAbort()`: the
/// cancellation cases' begin-abort choreography.
pub(super) fn begin_closed_abort(drive: &Drive) {
    let (sender, cancellation) = tokio::sync::watch::channel(());
    drop(sender);
    drive.begin_abort(cancellation);
    drive.signal_abort();
}

/// Commits the running operation's next state through the lane's command
/// surface, upstream's hand-seeded `lane.command` state-write blocks: the
/// command carries the given leading writes plus the
/// `operationState(&operationId, next)` write, repoints the projection's
/// operation at the next state over the same meta, and commits with no
/// events and no materialization. The `next` closure reads the operation's
/// current state — cancellation scopes and the pending effects derive from the
/// live projection, upstream's in-command `operationStateWithScope` calls.
///
/// # Panics
/// The lane projection's missing operation, upstream's
/// `` panic!("missing operation") ``, or the command's failure per
/// `expect`.
pub(super) async fn commit_next_operation_state<F>(
    lane: &Lane,
    operation_id: &str,
    leading_writes: Vec<Write>,
    next: F,
    expect: &str,
) where
    F: Fn(&OperationState) -> OperationState + Send + Sync + 'static,
{
    let operation_id = operation_id.to_owned();
    let leading_writes = Arc::new(leading_writes);
    let next = Arc::new(next);
    lane.command::<(), _>(
        move |state, _session, _context| {
            let operation_id = operation_id.clone();
            let leading_writes = Arc::clone(&leading_writes);
            let next = Arc::clone(&next);
            Box::pin(async move {
                let operation = state.operation.as_ref().expect("missing operation");
                let next_state = next(&operation.state);
                let mut writes = (*leading_writes).clone();
                writes.push(
                    set_value_write(&operation_state(&operation_id), next_state.clone())
                        .expect("the state write"),
                );
                Ok(LaneCommand::Commit {
                    writes,
                    next: LaneState {
                        operation: Some(LiveOperation {
                            meta: operation.meta.clone(),
                            state: next_state,
                        }),
                        ..state
                    },
                    materialize: Arc::new(|_: &CommitResult| ()),
                    events: None,
                })
            })
        },
        &background_context(),
    )
    .await
    .expect(expect);
}

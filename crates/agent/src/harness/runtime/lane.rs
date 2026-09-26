//! The runtime lane implementation, ported from upstream
//! `src/harness/runtime/lane.ts`.
//!
//! Upstream runs one JavaScript event loop; the port serializes the same
//! surfaces with locks and watch channels, keeping every critical section
//! synchronous (the agent-loop and Agent children's precedent):
//!
//! - `stateChange`'s resolve-once promise restates as a version counter on a
//!   watch channel; every awaiter captures the version and waits for the
//!   next bump.
//! - `idleOwner`'s promise restates as a cancellation token whose release
//!   fires `cancelled()` — the one-shot settlement with multi-consumer
//!   awaiting that `Promise.race` gives the command queue.
//! - `Drive.completion` rides the drive pass's watch channel;
//!   `Promise.race` restates as `tokio::select!`.
//! - the mutation-line reader restates as reads through the session itself:
//!   the session barrier bars concurrent mutations, so reads via the outer
//!   session see the same committed state upstream's in-mutation reader
//!   does.
//! - the mutation callback's throw restates through the boxed-Any contract:
//!   the callback always returns `Ok`, carrying the thrown error inside the
//!   payload; `session.mutate`'s own errors rethrow with the sealed-error
//!   check, upstream's outer catch.
//! - `settleOperation`'s type-level capability parameter erases: the planner
//!   receives the full [`OperationState`] and narrows by variant match.
//! - the `DataCloneError` an uncloneable event's synchronous emit raises
//!   restates as the delivery future's error: memory stays published and the
//!   command rejects after the commit.
//!
//! Staged seams raise [`SliceNotImplemented`] until their owning children
//! land: compaction and summarized-navigation preparation ride the
//! compaction child, skill and prompt-template formatting ride the skills
//! child, and the drive-pass procedure loop (`driveOperation`) rides the
//! drive child — an installed pass faults deterministically until then.
//!
//! Error-surface restatement: upstream lane methods reject with the thrown
//! error object; the landed [`AgentLane`] trait collapses that surface —
//! graceful seals surface as the inner `HarnessError::Closed`, faults and
//! other throws as the outer [`LaneOperationError::Closed`] carrying the
//! sealed message.
//!
//! `requestOperationAbort`'s cancellation promise rejects on a failed abort
//! request; the gate's landed `Cancellation` type is a one-shot watch
//! receiver with no rejection channel, so the error path releases the waiter
//! instead and the request's own rejection carries the failure — the drive
//! child's abort suites own that path and may revisit.

use std::any::Any;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use pi_ai::types::BoxedFuture;
use pi_ai::types::ImageContent;
use pi_ai::types::Message;
use pi_ai::types::Model;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::types::Usage;
use pi_ai::types::UserBlock;
use pi_ai::types::UserContent;
use pi_ai::types::UserMessage;
use pi_ai::utils::assistant_message_frame::reduce_assistant_message_frames;
use pi_chord::context::AbortReason;
use pi_chord::context::await_with_context;
use pi_chord::context::with_cancel;
use tokio_util::sync::CancellationToken;

use crate::harness::agent_harness::AbortOutcome;
use crate::harness::agent_harness::AbortRequestOutcome;
use crate::harness::agent_harness::AgentLane;
use crate::harness::agent_harness::CurrentOperationInfo;
use crate::harness::agent_harness::DeferredView;
use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::IdleCallback;
use crate::harness::agent_harness::LaneConfigUpdate;
use crate::harness::agent_harness::LaneExecutionInfo;
use crate::harness::agent_harness::LaneOperationError;
use crate::harness::agent_harness::LaneQueuedItem;
use crate::harness::agent_harness::LaneSnapshot;
use crate::harness::agent_harness::LaneSnapshotTool;
use crate::harness::agent_harness::LiveOperationView;
use crate::harness::agent_harness::OperationAdmission;
use crate::harness::agent_harness::OperationKind;
use crate::harness::agent_harness::OperationRequest;
use crate::harness::agent_harness::OperationStatus;
use crate::harness::agent_harness::PromptMessagesPayload;
use crate::harness::agent_harness::QueueImages;
use crate::harness::agent_harness::QueueMessage;
use crate::harness::agent_harness::RecordUsageOptions;
use crate::harness::agent_harness::ResnapshotBoundary as HarnessResnapshotBoundary;
use crate::harness::agent_harness::RetryView;
use crate::harness::agent_harness::RunFollowUp;
use crate::harness::agent_harness::RunOutcome;
use crate::harness::agent_harness::SuspensionFollowUp;
use crate::harness::agent_harness::WatchFilter;
use crate::harness::agent_harness::WatchHandle;
use crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS;
use crate::harness::context::Context;
use crate::harness::events::ListenerError;
use crate::harness::events::ResnapshotCapture;
use crate::harness::events::WatchFilter as BusWatchFilter;
use crate::harness::result::HarnessClosed;
use crate::harness::result::HarnessError;
use crate::harness::runtime::progress::read_assistant_frames;
use crate::harness::runtime::transcript::chain_entries;
use crate::harness::runtime::transcript::committed_entry_events;
use crate::harness::runtime::transcript::read_lane_queues;
use crate::harness::runtime::types::Config;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::any_payload;
use crate::harness::runtime::types::LaneCommand;
use crate::harness::runtime::types::from_arc;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::LanePatch;
use crate::harness::runtime::types::LaneState;
use crate::harness::runtime::types::LiveOperation;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::runtime::types::SliceNotImplemented;
use crate::harness::session::types::BranchScan;
use crate::harness::session::types::BranchScanOrder;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::Control;
use crate::harness::session::types::Entry;
use crate::harness::session::types::EntryType;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::OperationMeta;
use crate::harness::session::types::OperationResultRecord;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::RunSettings;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionInvariantError;
use crate::harness::session::types::SessionMutationCallback;
use crate::harness::session::types::SessionMutator;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::StartingOperation;
use crate::harness::session::types::UsageWriteRow;
use crate::harness::session::types::UsageRow;
use crate::harness::session::values::EntryWrite;
use crate::harness::session::values::UsageWrite;
use crate::harness::session::values::Write;
use crate::harness::session::values::branch_tip;
use crate::harness::session::values::delete_value;
use crate::harness::session::values::lane_config;
use crate::harness::session::values::lane_state;
use crate::harness::session::values::operation_meta;
use crate::harness::session::values::operation_preparation;
use crate::harness::session::values::operation_result;
use crate::harness::session::values::operation_state;
use crate::harness::session::values::operation_tool_args;
use crate::harness::session::values::pending_entry;
use crate::harness::session::values::pending_tool_output;
use crate::harness::session::values::set_value;
use crate::types::AgentMessage;
use crate::types::AgentToolResult;
use crate::types::QueueMode;
use crate::types::ToolExecutionMode;

const ABORT_ERROR_MESSAGE: &str = "The operation was aborted";

/// The fault handler the harness installs, upstream's `FaultHandler`:
/// converts one internal cause into the error the harness faults with.
pub type FaultHandler = Arc<dyn Fn(LaneError, &Context) -> LaneError + Send + Sync>;

/// The event publisher the harness installs, upstream's `EmitBatch`.
pub type EmitBatch =
    Arc<dyn Fn(Vec<HarnessEvent>, Context) -> BoxedFuture<'static, Result<(), LaneError>> + Send + Sync>;

/// The watcher installer the harness installs, upstream's `WatchHandler`
/// narrowed to the lane-snapshot watch the lane constructs. The initial
/// snapshot restates as the absence of one: the lane captures and sets it
/// after the install, upstream's `{} as LaneSnapshot` + assignment.
pub type WatchInstaller = Arc<
    dyn Fn(
            BusWatchFilter,
            &Context,
            ResnapshotCapture<LaneSnapshot>,
        ) -> Box<dyn WatchHandle<LaneSnapshot>>
        + Send
        + Sync,
>;

/// The configuration reader the harness installs, upstream's
/// `readConfig: () => Config`.
pub type ConfigProvider = Arc<dyn Fn() -> Config + Send + Sync>;

/// The drive claim one drive invocation resolves, upstream's `DriveClaim`.
enum DriveClaim {
    /// The pass installed on this call, or an earlier pass observed.
    Observe { drive: Arc<Drive>, installed: bool },
    /// Another operation's pass holds the lane.
    Occupied { drive: Arc<Drive> },
    /// The operation already settled.
    Settled { outcome: OperationResultRecord },
    /// The operation does not match the lane.
    Mismatch { error: HarnessError },
}

/// The payload one command's mutation carries back, upstream's
/// `LaneCommandOutcome`.
enum CommandOutcome<T> {
    Returned {
        result: T,
        delivery: Option<BoxedFuture<'static, Result<(), LaneError>>>,
    },
    IdleBlocked {
        owner: CancellationToken,
        change: tokio::sync::watch::Receiver<u64>,
    },
    Threw(LaneError),
}

/// The payload one read's mutation carries back, upstream's `readLane`'s
/// direct return.
enum ReadOutcome<T> {
    Payload(T),
    Threw(LaneError),
}

fn lock_core(shared: &LaneShared) -> MutexGuard<'_, LaneCore> {
    shared.core.lock().unwrap_or_else(PoisonError::into_inner)
}

fn inbox_items(inbox: &[InboxItem], kind: InboxItemKind) -> Vec<InboxItem> {
    inbox
        .iter()
        .filter(|item| item.kind == kind)
        .cloned()
        .collect()
}

fn without_inbox_items(inbox: &[InboxItem], removed: &[InboxItem]) -> Vec<InboxItem> {
    let removed_ids: std::collections::BTreeSet<&str> =
        removed.iter().map(|item| item.entry_id.as_str()).collect();
    inbox
        .iter()
        .filter(|item| !removed_ids.contains(item.entry_id.as_str()))
        .cloned()
        .collect()
}

/// The inbox admission split, upstream's `selectAcceptedInbox`.
fn select_accepted_inbox(
    inbox: &[InboxItem],
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,
) -> (Vec<InboxItem>, Vec<InboxItem>) {
    let mut steer_taken = false;
    let mut follow_up_taken = false;
    let mut selected = Vec::new();
    let mut remainder = Vec::new();
    for item in inbox {
        let eligible = item.kind == InboxItemKind::Write
            || item.kind == InboxItemKind::NextRun
            || (item.kind == InboxItemKind::Steer && (steering_mode == QueueMode::All || !steer_taken))
            || (item.kind == InboxItemKind::FollowUp
                && (follow_up_mode == QueueMode::All || !follow_up_taken));
        if eligible {
            if item.kind == InboxItemKind::Steer {
                steer_taken = true;
            }
            if item.kind == InboxItemKind::FollowUp {
                follow_up_taken = true;
            }
            selected.push(item.clone());
        } else {
            remainder.push(item.clone());
        }
    }
    (selected, remainder)
}

fn captured_settings(config: &Config) -> RunSettings {
    RunSettings {
        compaction: config.compaction,
        steering_mode: config.steering_mode,
        follow_up_mode: config.follow_up_mode,
        tool_execution: config.tool_execution,
    }
}

fn durable_lane_state(
    state: &LaneState,
    current_operation_id: Option<&str>,
    inbox: &[InboxItem],
    last_operation_id: Option<&str>,
) -> crate::harness::session::types::LaneState {
    crate::harness::session::types::LaneState {
        current_operation_id: current_operation_id.map(str::to_owned),
        last_operation_id: last_operation_id.map(str::to_owned),
        inbox: inbox.to_vec(),
    }
}

fn pending_entry_write(entry_id: &str, pending: &PendingEntry) -> NewEntry {
    match pending {
        PendingEntry::Message { payload } => NewEntry::Message {
            id: entry_id.to_owned(),
            parent_id: None,
            body: lane_error(MessageEntry {
                message: (**payload).clone(),
                terminate: None,
            }),
        },
        PendingEntry::Custom { custom_type, payload } => NewEntry::Custom {
            id: entry_id.to_owned(),
            parent_id: None,
            body: crate::harness::session::types::CustomEntryBody {
                custom_type: custom_type.clone(),
                data: payload.clone(),
            },
        },
    }
}

/// The model identity the operation's live state captured, upstream's
/// `capturedModel`.
#[must_use]
fn captured_model(operation: &OperationState) -> Option<crate::harness::session::types::ModelIdentity> {
    match operation {
        OperationState::AssistantReady(leaf) => Some(leaf.generation_context.configuration.model.clone()),
        OperationState::AssistantEffectPending(leaf) => Some(leaf.generation_context.configuration.model.clone()),
        OperationState::AssistantRetryWait(leaf) => Some(leaf.generation_context.configuration.model.clone()),
        OperationState::Tools(leaf) => Some(leaf.batch.configuration.model.clone()),
        OperationState::DeferredSuspended(leaf) => Some(leaf.deferred.configuration.model.clone()),
        OperationState::DeferredEffectPending(leaf) => Some(leaf.scope.configuration.model.clone()),
        OperationState::SummaryReady(leaf) => Some(leaf.generation.summary_context.configuration.model.clone()),
        OperationState::SummaryEffectPending(leaf) => {
            Some(leaf.generation.summary_context.configuration.model.clone())
        }
        OperationState::SummaryRetryWait(leaf) => Some(leaf.generation.summary_context.configuration.model.clone()),
        _ => None,
    }
}

/// The error one aborted wait carries, upstream's `DOMException` and the
/// caller-supplied abort reason rethrow.
fn abort_reason_error(reason: Option<AbortReason>) -> LaneError {
    let message = match reason {
        Some(AbortReason::Caller(message)) => message,
        Some(AbortReason::Aborted) | None => ABORT_ERROR_MESSAGE.to_owned(),
    };
    from_arc(Arc::new(SessionError(message)))
}

fn intent_kind_of(meta: &OperationMeta) -> OperationKind {
    match meta.intent {
        crate::harness::session::types::OperationIntent::Run { .. } => OperationKind::Run,
        crate::harness::session::types::OperationIntent::Compaction { .. } => OperationKind::Compaction,
        crate::harness::session::types::OperationIntent::Navigation { .. } => OperationKind::Navigation,
    }
}

fn inbox_kind_name(kind: InboxItemKind) -> &'static str {
    match kind {
        InboxItemKind::Steer => "steer",
        InboxItemKind::FollowUp => "followUp",
        InboxItemKind::NextRun => "nextRun",
        InboxItemKind::Write => "write",
    }
}

fn quoted(name: &str) -> String {
    serde_json::to_string(name).unwrap_or_else(|_| format!("{name:?}"))
}

/// Rebuilds one durable leaf with a fresh uniform scope, upstream's
/// `{...operation.state, control, ...}` spreads in the drive procedures.
#[must_use]
pub fn operation_state_with_scope(state: &OperationState, scope: OperationScope) -> OperationState {
    match state {
        OperationState::Starting(leaf) => OperationState::Starting(StartingOperation {
            scope: scope.clone(),
            ..leaf.clone()
        }),
        OperationState::Checkpoint(leaf) => OperationState::Checkpoint(
            crate::harness::session::types::CheckpointOperation {
                scope: scope.clone(),
                ..leaf.clone()
            },
        ),
        OperationState::AssistantReady(leaf) => OperationState::AssistantReady(
            crate::harness::session::types::AssistantReadyOperation {
                scope: scope.clone(),
                ..leaf.clone()
            },
        ),
        OperationState::AssistantEffectPending(leaf) => {
            OperationState::AssistantEffectPending(crate::harness::session::types::AssistantEffectPendingOperation {
                scope: scope.clone(),
                ..leaf.clone()
            })
        }
        OperationState::AssistantRetryWait(leaf) => {
            OperationState::AssistantRetryWait(crate::harness::session::types::AssistantRetryWaitOperation {
                scope: scope.clone(),
                ..leaf.clone()
            })
        }
        OperationState::Tools(leaf) => OperationState::Tools(crate::harness::session::types::ToolsOperation {
            scope: scope.clone(),
            ..leaf.clone()
        }),
        OperationState::DeferredSuspended(leaf) => {
            OperationState::DeferredSuspended(crate::harness::session::types::DeferredSuspendedOperation {
                deferred: crate::harness::session::types::DeferredScope {
                    scope: scope.clone(),
                    ..leaf.deferred.clone()
                },
            })
        }
        OperationState::DeferredEffectPending(leaf) => {
            OperationState::DeferredEffectPending(crate::harness::session::types::DeferredEffectPendingOperation {
                scope: crate::harness::session::types::DeferredScope {
                    scope: scope.clone(),
                    ..leaf.scope.clone()
                },
                ..leaf.clone()
            })
        }
        OperationState::SummaryDeciding(leaf) => {
            OperationState::SummaryDeciding(crate::harness::session::types::SummaryDecidingOperation {
                scope: scope.clone(),
                ..leaf.clone()
            })
        }
        OperationState::SummaryReady(leaf) => {
            OperationState::SummaryReady(crate::harness::session::types::SummaryReadyOperation {
                scope: scope.clone(),
                ..leaf.clone()
            })
        }
        OperationState::SummaryEffectPending(leaf) => OperationState::SummaryEffectPending(
            crate::harness::session::types::SummaryEffectPendingOperation {
                scope: scope.clone(),
                ..leaf.clone()
            },
        ),
        OperationState::SummaryRetryWait(leaf) => {
            OperationState::SummaryRetryWait(crate::harness::session::types::SummaryRetryWaitOperation {
                scope: scope.clone(),
                ..leaf.clone()
            })
        }
        OperationState::NavigationReadyToCommit(leaf) => OperationState::NavigationReadyToCommit(
            crate::harness::session::types::NavigationReadyToCommitOperation {
                scope: scope.clone(),
                ..leaf.clone()
            },
        ),
    }
}

struct LaneCore {
    state: LaneState,
    closed_error: Option<LaneError>,
    active_drive: Option<Arc<Drive>>,
    idle_owner: Option<CancellationToken>,
    state_change: tokio::sync::watch::Sender<u64>,
}

struct LaneShared {
    name: String,
    session: Arc<dyn Session>,
    models: Arc<pi_ai::models::Models>,
    hooks: Arc<crate::harness::hooks::HookRegistry>,
    on_fault: FaultHandler,
    emit_batch: EmitBatch,
    install_watch: WatchInstaller,
    read_config: ConfigProvider,
    core: Mutex<LaneCore>,
}

impl LaneShared {
    fn sealed_error(&self) -> Option<LaneError> {
        lock_core(self).closed_error.clone()
    }

    fn is_harness_closed(&self) -> bool {
        lock_core(self)
            .closed_error
            .as_ref()
            .is_some_and(|error| error.downcast_ref::<HarnessClosed>().is_some())
    }

    fn assert_open(&self) -> Result<(), LaneError> {
        match self.sealed_error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn signal_state_change(&self) {
        let core = lock_core(self);
        let _ = core.state_change.send_modify(|version| *version += 1);
    }

    fn state_change_receiver(&self) -> tokio::sync::watch::Receiver<u64> {
        lock_core(self).state_change.subscribe()
    }
}

/// Runtime implementation of one configured lane, upstream's `Lane`.
#[derive(Clone)]
pub struct Lane {
    inner: Arc<LaneShared>,
}

impl std::fmt::Debug for Lane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lane")
            .field("name", &self.inner.name)
            .finish_non_exhaustive()
    }
}

/// Waits for one state-change bump past the captured version, upstream's
/// awaiting `stateChange`.
async fn wait_state_change(mut receiver: tokio::sync::watch::Receiver<u64>) {
    let current = *receiver.borrow();
    loop {
        if *receiver.borrow_and_update() != current {
            return;
        }
        if receiver.changed().await.is_err() {
            return;
        }
    }
}

/// Races the idle owner's release against the next state change, upstream's
/// `Promise.race([idleOwner, stateChange])`.
async fn wait_owner_or_change(owner: &CancellationToken, receiver: tokio::sync::watch::Receiver<u64>) {
    tokio::select! {
        () = owner.cancelled() => (),
        () = wait_state_change(receiver) => (),
    }
}

impl Lane {
    /// Builds one lane over its durable state, upstream's `constructor`.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the constructor carries upstream's parameter list; the harness runtime child bundles them when it lands"
    )]
    pub fn new(
        name: impl Into<String>,
        session: Arc<dyn Session>,
        models: Arc<pi_ai::models::Models>,
        hooks: Arc<crate::harness::hooks::HookRegistry>,
        state: LaneState,
        on_fault: FaultHandler,
        emit_batch: EmitBatch,
        install_watch: WatchInstaller,
        read_config: ConfigProvider,
    ) -> Self {
        let (state_change, _) = tokio::sync::watch::channel(0u64);
        Self {
            inner: Arc::new(LaneShared {
                name: name.into(),
                session,
                models,
                hooks,
                on_fault,
                emit_batch,
                install_watch,
                read_config,
                core: Mutex::new(LaneCore {
                    state,
                    closed_error: None,
                    active_drive: None,
                    idle_owner: None,
                    state_change,
                }),
            }),
        }
    }

    /// The lane name, upstream's `name`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// The session the lane runs on, upstream's `session`.
    #[must_use]
    pub fn session(&self) -> &Arc<dyn Session> {
        &self.inner.session
    }

    /// The hook registry, upstream's `hooks`.
    #[must_use]
    pub fn hooks(&self) -> &Arc<crate::harness::hooks::HookRegistry> {
        &self.inner.hooks
    }

    /// The owned state projection, upstream's `state` property.
    #[must_use]
    pub fn state(&self) -> LaneState {
        lock_core(&self.inner).state.clone()
    }

    /// The active drive pass, upstream's `activeDrive` — package-internal;
    /// the deterministic procedure tests install exact owners directly.
    #[must_use]
    pub fn active_drive(&self) -> Option<Arc<Drive>> {
        lock_core(&self.inner).active_drive.clone()
    }

    /// Installs or clears the active drive pass; package-internal, the
    /// procedure tests' direct-owner statement.
    pub(crate) fn set_active_drive(&self, drive: Option<Arc<Drive>>) {
        lock_core(&self.inner).active_drive = drive;
    }

    /// The current configuration, upstream's `readConfig`.
    #[must_use]
    pub fn read_config(&self) -> Config {
        (self.inner.read_config)()
    }

    /// The mismatch error one operation id produces, upstream's `mismatch`.
    #[must_use]
    pub fn mismatch(
        &self,
        expected: &str,
        current_operation_id: Option<&str>,
        last_operation_id: Option<&str>,
    ) -> HarnessError {
        HarnessError::OperationMismatch {
            lane: self.inner.name.clone(),
            expected_operation_id: expected.to_owned(),
            current_operation_id: current_operation_id.map(str::to_owned),
            last_operation_id: last_operation_id.map(str::to_owned),
            message: format!(
                "Operation {expected} does not own lane {}",
                quoted(&self.inner.name)
            ),
        }
    }

    fn fault(&self, cause: LaneError, context: &Context) -> LaneError {
        (self.inner.on_fault)(cause, context)
    }

    /// Runs one body inside the session's mutation line with the fault and
    /// sealed-error wrapping, upstream's `readLane`/`command` shared
    /// scaffolding.
    async fn run_mutating<T, F>(&self, body: F, context: &Context) -> Result<T, LaneError>
    where
        T: Send + 'static,
        F: Fn(&LaneShared, &dyn SessionMutator, &Context) -> BoxedFuture<'static, Result<T, LaneError>>
            + Send
            + Sync
            + 'static,
    {
        self.inner.assert_open()?;
        let inner = Arc::clone(&self.inner);
        let callback: SessionMutationCallback = Box::new(move |mutator, mutation_context| {
            let inner = Arc::clone(&inner);
            let mutation_context = mutation_context.clone();
            Box::pin(async move {
                if let Err(sealed) = inner.assert_open() {
                    return Ok(any_payload(ReadOutcome::<T>::Threw(sealed)));
                }
                match (body)(&inner, mutator, &mutation_context).await {
                    Ok(payload) => Ok(any_payload(ReadOutcome::Payload(payload))),
                    Err(error) => {
                        let thrown = match inner.sealed_error() {
                            Some(sealed) => sealed,
                            None => (inner.on_fault)(error, &mutation_context),
                        };
                        Ok(any_payload(ReadOutcome::Threw(thrown)))
                    }
                }
            })
        });
        let payload = self.inner.session.mutate(callback, context).await;
        let payload = match payload {
            Ok(payload) => payload,
            Err(error) => {
                return match self.inner.sealed_error() {
                    Some(sealed) => Err(sealed),
                    None => Err(lane_error(error)),
                };
            }
        };
        let outcome = payload.downcast::<ReadOutcome<T>>().map_err(|_| {
            from_arc(Arc::new(SessionError("lane mutation returned an unexpected payload".to_owned()))
               )
        })?;
        match *outcome {
            ReadOutcome::Threw(error) => Err(error),
            ReadOutcome::Payload(value) => Ok(value),
        }
    }

    /// Runs one effect-free read on the lane's serialized mutation line,
    /// upstream's `readLane`.
    async fn read_lane<T, F>(&self, read: F, context: &Context) -> Result<T, LaneError>
    where
        T: Send + 'static,
        F: Fn(LaneState, Arc<dyn Session>, Context) -> BoxedFuture<'static, Result<T, LaneError>>
            + Send
            + Sync
            + 'static,
    {
        self.run_mutating::<T, _>(
            move |inner, _mutator, mutation_context| {
                let read = Arc::new(read);
                Box::pin(async move {
                    let state = lock_core(inner).state.clone();
                    let session = Arc::clone(&inner.session);
                    (read)(state, session, mutation_context.clone()).await
                })
            },
            context,
        )
        .await
    }

    /// Runs one effect-free command on the lane's serialized mutation line,
    /// upstream's `command`.
    ///
    /// A planner commits once (publishing `next`, then materializing the
    /// result synchronously from storage-assigned metadata), returns without
    /// a commit, or rejects as an expected caller error. Planner, commit,
    /// and materialization errors fault the harness before the session line
    /// releases. Close/fault gates are checked before queueing and when the
    /// callback starts: a callback admitted before a close may finish its
    /// commit, publish memory, and resolve without another open check.
    /// Providers, tools, hooks, timers, event handlers, and effect waits
    /// never run inside a planner.
    pub async fn command<T, F>(&self, plan: F, context: &Context) -> Result<T, LaneError>
    where
        T: Send + 'static,
        F: Fn(LaneState, Arc<dyn Session>, Context) -> BoxedFuture<'static, Result<LaneCommand<T>, LaneError>>
            + Send
            + Sync
            + 'static,
    {
        self.inner.assert_open()?;
        loop {
            // Wait out any idle-callback owner before queueing, upstream's
            // `while (this.idleOwner !== undefined)` loop.
            loop {
                let owner = {
                    let core = lock_core(&self.inner);
                    core.idle_owner.clone()
                };
                let Some(owner) = owner else {
                    break;
                };
                let change = self.inner.state_change_receiver();
                if await_with_context(wait_owner_or_change(&owner, change), context)
                    .await
                    .is_err()
                {
                    return Err(abort_reason_error(None));
                }
                self.inner.assert_open()?;
            }

            let outcome = self
                .run_mutating::<CommandOutcome<T>, _>(
                    {
                        let plan = Arc::new(plan);
                        move |inner, mutator, mutation_context| {
                            let plan = Arc::clone(&plan);
                            let inner = Arc::clone(inner);
                            Box::pin(async move {
                                {
                                    let core = lock_core(&inner);
                                    if let Some(owner) = core.idle_owner.clone() {
                                        let change = core.state_change.subscribe();
                                        return Ok(CommandOutcome::<T>::IdleBlocked { owner, change });
                                    }
                                }
                                let state = lock_core(&inner).state.clone();
                                let session = Arc::clone(&inner.session);
                                let decision = (plan)(state.clone(), session, mutation_context.clone()).await?;
                                match decision {
                                    LaneCommand::Return { result } => Ok(CommandOutcome::Returned {
                                        result,
                                        delivery: None,
                                    }),
                                    LaneCommand::Reject { error } => Ok(CommandOutcome::Threw(error)),
                                    LaneCommand::Commit {
                                        writes,
                                        next,
                                        materialize,
                                        events,
                                    } => {
                                        let commit = mutator.commit(writes, mutation_context).await?;
                                        {
                                            let mut core = lock_core(&inner);
                                            core.state = next;
                                            let _ = core.state_change.send_modify(|version| *version += 1);
                                        }
                                        let result = (materialize)(&commit);
                                        let events = match &events {
                                            Some(events) => (events)(&commit),
                                            None => Vec::new(),
                                        };
                                        let delivery = if events.is_empty() {
                                            None
                                        } else {
                                            Some((inner.emit_batch)(events, mutation_context.clone()))
                                        };
                                        Ok(CommandOutcome::Returned { result, delivery })
                                    }
                                }
                            })
                        }
                    },
                    context,
                )
                .await;

            match outcome {
                Ok(CommandOutcome::Returned { result, delivery }) => {
                    if let Some(delivery) = delivery {
                        delivery.await?;
                    }
                    return Ok(result);
                }
                Ok(CommandOutcome::IdleBlocked { owner, change }) => {
                    if await_with_context(wait_owner_or_change(&owner, change), context)
                        .await
                        .is_err()
                    {
                        return Err(abort_reason_error(None));
                    }
                    self.inner.assert_open()?;
                    // Retry the plan against the latest committed memory.
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Runs a command against the current operation even after cancellation
    /// is requested, upstream's `settleOperation`. Use this to settle
    /// admitted effects, finish the operation, or update concurrent child
    /// state; the drive continuation remains the sole top-level state
    /// writer. Upstream's type-level capability parameter erases into the
    /// planner's variant match.
    pub async fn settle_operation<T, F>(&self, plan: F, context: &Context) -> Result<T, LaneError>
    where
        T: Send + 'static,
        F: Fn(LaneState, &OperationState, &OperationMeta, Arc<dyn Session>, Context)
                -> BoxedFuture<'static, Result<OperationCommand<T>, LaneError>>
            + Send
            + Sync
            + 'static,
    {
        let plan = Arc::new(plan);
        self.command::<T, _>(
            move |state, session, command_context| {
                let plan = Arc::clone(&plan);
                Box::pin(async move {
                    let operation = state.operation.clone().ok_or_else(|| {
                        let error: LaneError = Arc::new(SessionInvariantError(
                            "settleOperation requires an active operation".to_owned(),
                        ));
                        error
                    })?;
                    let decision =
                        (plan)(&state, &operation.state, &operation.meta, session, command_context.clone()).await?;
                    Ok(match decision {
                        OperationCommand::Commit {
                            writes,
                            operation_state,
                            lane,
                            materialize,
                            events,
                        } => {
                            let mut writes = writes;
                            writes.push(
                                set_value(
                                    &operation_state(&operation.meta.operation_id),
                                    operation_state.clone(),
                                )
                                .map_err(|error| lane_error(error))?,
                            );
                            if let Some(LanePatch { inbox: Some(inbox), .. }) = &lane {
                                writes.push(
                                    set_value(
                                        &lane_state(&operation.meta.lane),
                                        durable_lane_state(
                                            &state,
                                            Some(&operation.meta.operation_id),
                                            inbox,
                                            state.last_operation_id.as_deref(),
                                        ),
                                    )
                                    .map_err(|error| lane_error(error))?,
                                );
                            }
                            let mut next = apply_lane_patch(&state, lane.as_ref());
                            next.operation = Some(LiveOperation {
                                meta: operation.meta.clone(),
                                state: operation_state,
                            });
                            LaneCommand::Commit {
                                writes,
                                next,
                                materialize,
                                events,
                            }
                        }
                        OperationCommand::Finish {
                            writes,
                            record,
                            lane,
                            materialize,
                            events,
                        } => {
                            let mut writes = writes;
                            writes.push(
                                set_value(&operation_result(&operation.meta.operation_id), record.clone())
                                    .map_err(|error| lane_error(error))?,
                            );
                            let inbox = lane
                                .as_ref()
                                .and_then(|patch| patch.inbox.clone())
                                .unwrap_or_else(|| state.inbox.clone());
                            writes.push(
                                set_value(
                                    &lane_state(&operation.meta.lane),
                                    durable_lane_state(
                                        &state,
                                        None,
                                        &inbox,
                                        Some(&operation.meta.operation_id),
                                    ),
                                )
                                .map_err(|error| lane_error(error))?,
                            );
                            let mut next = apply_lane_patch(&state, lane.as_ref());
                            next.inbox = inbox;
                            next.last_operation_id = Some(operation.meta.operation_id.clone());
                            next.operation = None;
                            LaneCommand::Commit {
                                writes,
                                next,
                                materialize,
                                events,
                            }
                        }
                        OperationCommand::Return { result } => LaneCommand::Return { result },
                    })
                })
            },
            context,
        )
        .await
    }

    /// Runs an ordinary operation command only while durable control is
    /// running, upstream's `continueOperation`; use this before starting new
    /// hooks, effects, or forward progress. Returns
    /// [`ContinueOperationResult::CancelRequested`] without invoking the
    /// planner once cancellation is requested.
    pub async fn continue_operation<T, F>(
        &self,
        plan: F,
        context: &Context,
    ) -> Result<ContinueOperationResult<T>, LaneError>
    where
        T: Send + 'static,
        F: Fn(LaneState, Arc<dyn Session>, Context) -> BoxedFuture<'static, Result<OperationCommand<T>, LaneError>>
            + Send
            + Sync
            + 'static,
    {
        let plan = Arc::new(plan);
        self.settle_operation::<ContinueOperationResult<T>, _>(
            move |state, latest, _meta, session, operation_context| {
                let plan = Arc::clone(&plan);
                Box::pin(async move {
                    if matches!(
                        crate::harness::session::types::operation_scope_of(latest).control,
                        Control::CancelRequested
                    ) {
                        return Ok(OperationCommand::Return {
                            result: ContinueOperationResult::CancelRequested,
                        });
                    }
                    let decision = (plan)(state, session, operation_context).await?;
                    Ok(match decision {
                        OperationCommand::Return { result } => OperationCommand::Return {
                            result: ContinueOperationResult::Result { value: result },
                        },
                        OperationCommand::Commit {
                            writes,
                            operation_state,
                            lane,
                            materialize,
                            events,
                        } => OperationCommand::Commit {
                            writes,
                            operation_state,
                            lane,
                            materialize: Arc::new(move |commit: &CommitResult| {
                                ContinueOperationResult::Result {
                                    value: (materialize)(commit),
                                }
                            }),
                            events,
                        },
                        OperationCommand::Finish {
                            writes,
                            record,
                            lane,
                            materialize,
                            events,
                        } => OperationCommand::Finish {
                            writes,
                            record,
                            lane,
                            materialize: Arc::new(move |commit: &CommitResult| {
                                ContinueOperationResult::Result {
                                    value: (materialize)(commit),
                                }
                            }),
                            events,
                        },
                    })
                })
            },
            context,
        )
        .await
    }

    /// Admits one operation, upstream's `accept`.
    ///
    /// # Errors
    /// The sealed error when the lane sealed with a fault; admission errors
    /// ride the inner result.
    pub async fn accept_impl(
        &self,
        request: OperationRequest,
        context: &Context,
    ) -> Result<OperationAdmissionResult, LaneError> {
        if self.inner.is_harness_closed() {
            return Ok(Err(HarnessError::Closed {
                message: self.sealed_message(),
            }));
        }
        self.inner.assert_open()?;
        let started_at = crate::harness::env::nodejs::now_ms();
        let operation_id = match &request {
            OperationRequest::Prompt { operation_id, .. }
            | OperationRequest::Skill { operation_id, .. }
            | OperationRequest::PromptTemplate { operation_id, .. }
            | OperationRequest::Compaction { operation_id, .. }
            | OperationRequest::Navigation { operation_id, .. } => match operation_id {
                Some(operation_id) => operation_id.clone(),
                None => self.inner.session.id_generator().next(Some(started_at)),
            },
        };
        let acceptance_config = self.read_config();
        match &request {
            OperationRequest::Compaction { .. } => {
                self.accept_compaction(&request, operation_id, started_at, acceptance_config, context)
                    .await
            }
            OperationRequest::Navigation { .. } => {
                self.accept_navigation(&request, operation_id, started_at, acceptance_config, context)
                    .await
            }
            _ => {
                self.accept_run(&request, operation_id, started_at, acceptance_config, context)
                    .await
            }
        }
    }

    fn sealed_message(&self) -> String {
        self.inner
            .sealed_error()
            .map(|error| error.to_string())
            .unwrap_or_default()
    }

    /// Builds the prompt messages one run request replays, upstream's
    /// `acceptRun`'s message switch.
    fn run_request_messages(
        &self,
        request: &OperationRequest,
        started_at: i64,
        acceptance_config: &Config,
    ) -> Result<Vec<AgentMessage>, HarnessError> {
        match request {
            OperationRequest::Prompt { prompt, .. } => match prompt.as_ref() {
                PromptMessagesPayload::Text { prompt, images } => {
                    let images = images.clone().unwrap_or_default();
                    if prompt.is_empty() && images.is_empty() {
                        return Ok(Vec::new());
                    }
                    let mut content: Vec<UserBlock> = Vec::new();
                    if !prompt.is_empty() {
                        content.push(UserBlock::Text(TextContent {
                            text: prompt.clone(),
                            text_signature: None,
                        }));
                    }
                    content.extend(images.iter().cloned().map(UserBlock::Image));
                    Ok(vec![AgentMessage::Standard(Message::User(UserMessage {
                        content: UserContent::Blocks(content),
                        timestamp: started_at,
                    }))])
                }
                PromptMessagesPayload::Message(message) => Ok(vec[(**message).clone()]),
                PromptMessagesPayload::Messages(messages) => Ok(messages.as_ref().clone()),
            },
            OperationRequest::Skill {
                name,
                additional_instructions,
                ..
            } => {
                let skill = acceptance_config
                    .resources
                    .skills
                    .iter()
                    .find(|candidate| candidate.name == *name);
                let Some(skill) = skill else {
                    return Err(HarnessError::UnknownSkill {
                        name: name.clone(),
                        message: format!("Unknown skill: {name}"),
                    });
                };
                let _ = (skill, additional_instructions);
                Err(HarnessError::Closed {
                    message: SliceNotImplemented::new("skill invocation").to_string(),
                })
            }
            OperationRequest::PromptTemplate { name, args, .. } => {
                let template = acceptance_config
                    .resources
                    .prompt_templates
                    .iter()
                    .find(|candidate| candidate.name == *name);
                let Some(template) = template else {
                    return Err(HarnessError::UnknownTemplate {
                        name: name.clone(),
                        message: format!("Unknown prompt template: {name}"),
                    });
                };
                let _ = (template, args);
                Err(HarnessError::Closed {
                    message: SliceNotImplemented::new("prompt template invocation").to_string(),
                })
            }
            _ => unreachable!("accept_run takes run requests only"),
        }
    }

    /// Admits a run operation, upstream's `acceptRun`.
    async fn accept_run(
        &self,
        request: &OperationRequest,
        operation_id: String,
        started_at: i64,
        acceptance_config: Config,
        context: &Context,
    ) -> Result<OperationAdmissionResult, LaneError> {
        let messages = self.run_request_messages(request, started_at, &acceptance_config)?;
        let messages = match messages {
            Ok(messages) => messages,
            Err(error) => return Ok(Err(error)),
        };
        for message in &messages {
            if let AgentMessage::Standard(Message::Assistant(assistant)) = message {
                if assistant.stop_reason == StopReason::Pending {
                    return Ok(Err(HarnessError::InvalidMessage {
                        lane: self.inner.name.clone(),
                        reason: "pending_assistant".to_owned(),
                        message: "Cannot accept a pending assistant message".to_owned(),
                    }));
                }
            }
        }
        let prompt: Vec<(String, AgentMessage)> = messages
            .into_iter()
            .map(|message| (self.inner.session.id_generator().next(Some(started_at)), message))
            .collect();
        let lane_name = self.inner.name.clone();
        let steering_mode = acceptance_config.steering_mode;
        let follow_up_mode = acceptance_config.follow_up_mode;

        self.command::<OperationAdmissionResult, _>(
            move |state, session, command_context| {
                Box::pin(async move {
                    if let Some(operation) = &state.operation {
                        return Ok(LaneCommand::Return {
                            result: Err(HarnessError::LaneBusy {
                                lane: lane_name.clone(),
                                operation_id: operation.meta.operation_id.clone(),
                                operation_kind: intent_kind_of(&operation.meta),
                                message: format!(
                                    "Lane {} already has an active operation",
                                    quoted(&lane_name)
                                ),
                            }),
                        });
                    }
                    let (selected_items, inbox) =
                        select_accepted_inbox(&state.inbox, steering_mode, follow_up_mode);
                    let mut captured_entries: Vec<(InboxItem, PendingEntry)> = Vec::new();
                    for item in &selected_items {
                        let stored = session
                            .get_value(&pending_entry(&item.entry_id).address, &command_context)
                            .await
                            .map_err(|error| lane_error(error))?;
                        let Some(stored) = stored else {
                            return Err(Arc::new(SessionInvariantError(format!(
                                "Pending {} entry {} is missing its payload",
                                inbox_kind_name(item.kind),
                                item.entry_id
                            ))));
                        };
                        let pending: PendingEntry = serde_json::from_value(stored.value).map_err(|error| {
                            lane_error(SessionInvariantError(format!(
                                "Pending entry payload is malformed: {error}"
                            )))
                        })?;
                        if item.kind != InboxItemKind::Write {
                            let PendingEntry::Message { payload } = &pending else {
                                return Err(lane_error(SessionInvariantError(format!(
                                    "Pending {} entry {} is not a message",
                                    inbox_kind_name(item.kind),
                                    item.entry_id
                                ))));
                            };
                            if let AgentMessage::Standard(Message::Assistant(assistant)) = payload.as_ref() {
                                if assistant.stop_reason == StopReason::Pending {
                                    return Err(lane_error(SessionInvariantError(format!(
                                        "Pending {} entry {} contains a pending assistant",
                                        inbox_kind_name(item.kind),
                                        item.entry_id
                                    ))));
                                }
                            }
                        }
                        captured_entries.push((item.clone(), pending));
                    }
                    let has_captured_conversation =
                        selected_items.iter().any(|item| item.kind != InboxItemKind::Write);
                    if prompt.is_empty() && !has_captured_conversation {
                        return Ok(LaneCommand::Return {
                            result: Err(HarnessError::InvalidMessage {
                                lane: lane_name.clone(),
                                reason: "empty".to_owned(),
                                message: "Acceptance must append at least one message".to_owned(),
                            }),
                        });
                    }

                    let mut chained: Vec<NewEntry> = Vec::new();
                    for (item, pending) in &captured_entries {
                        chained.push(pending_entry_write(&item.entry_id, pending));
                    }
                    for (id, message) in &prompt {
                        chained.push(NewEntry::Message {
                            id: id.clone(),
                            parent_id: None,
                            body: lane_error(MessageEntry {
                                message: message.clone(),
                                terminate: None,
                            }),
                        });
                    }
                    let entries = chain_entries(state.tip_id.clone(), chained);
                    let Some(parent_id) = entries.last().map(|entry| entry.id().to_owned()) else {
                        return Err(Arc::new(SessionInvariantError(
                            "Acceptance built no entries".to_owned(),
                        )));
                    };
                    let meta = OperationMeta {
                        operation_id: operation_id.clone(),
                        lane: lane_name.clone(),
                        source_tip_id: state.tip_id.clone(),
                        started_at,
                        intent: crate::harness::session::types::OperationIntent::Run {
                            prompt_entry_ids: prompt.iter().map(|(id, _)| id.clone()).collect(),
                        },
                    };
                    let operation_state = OperationState::Starting(StartingOperation {
                        scope: OperationScope {
                            control: Control::Running,
                            settings: captured_settings(&acceptance_config),
                            latest_assistant_entry_id: None,
                        },
                    });
                    let remaining_queues = read_lane_queues(session.as_ref(), &inbox, &command_context)
                        .await
                        .map_err(|error| lane_error(error))?;
                    let mut next = state.clone();
                    next.tip_id = Some(parent_id.clone());
                    next.inbox = inbox.clone();
                    next.operation = Some(LiveOperation {
                        meta: meta.clone(),
                        state: operation_state.clone(),
                    });

                    let mut writes: Vec<Write> = Vec::new();
                    for entry in &entries {
                        writes.push(Write::Entry(lane_error(EntryWrite {
                            kind: "entry".to_owned(),
                            entry: entry.clone(),
                        })));
                    }
                    for item in &selected_items {
                        writes.push(delete_value(&pending_entry(&item.entry_id)));
                    }
                    writes.push(
                        set_value(&branch_tip(&lane_name), Some(parent_id.clone()))
                            .map_err(|error| lane_error(error))?,
                    );
                    writes.push(
                        set_value(&operation_meta(&operation_id), meta.clone())
                            .map_err(|error| lane_error(error))?,
                    );
                    writes.push(
                        set_value(&operation_state(&operation_id), operation_state.clone())
                            .map_err(|error| lane_error(error))?,
                    );
                    writes.push(
                        set_value(
                            &lane_state(&lane_name),
                            durable_lane_state(
                                &state,
                                Some(&operation_id),
                                &inbox,
                                state.last_operation_id.as_deref(),
                            ),
                        )
                        .map_err(|error| lane_error(error))?,
                    );

                    let events_lane = lane_name.clone();
                    let events_operation_id = operation_id.clone();
                    let events_entries = entries.clone();
                    let events_queues = remaining_queues.clone();
                    let selected_count = selected_items.len();
                    Ok(LaneCommand::Commit {
                        writes,
                        next,
                        materialize: Arc::new(move |_commit: &CommitResult| {
                            Ok(OperationAdmission {
                                operation_id: operation_id.clone(),
                                kind: OperationKind::Run,
                                started_at,
                            })
                        }),
                        events: Some(Arc::new(move |commit: &CommitResult| {
                            let mut events: Vec<HarnessEvent> = vec![HarnessEvent::lane_scoped(
                                &events_lane,
                                false,
                                HarnessEventPayload::RunStart {
                                    run_id: events_operation_id.clone(),
                                    started_at,
                                },
                            )
                            .unwrap_or_else(|error| unreachable!("run_start is lane-scoped: {error}"))];
                            events.extend(committed_entry_events(
                                &events_entries,
                                commit,
                                &events_lane,
                                Some(&events_operation_id),
                                0,
                            ));
                            if selected_count > 0 {
                                events.push(
                                    HarnessEvent::lane_scoped(
                                        &events_lane,
                                        false,
                                        HarnessEventPayload::QueueUpdate {
                                            queues: events_queues.clone(),
                                        },
                                    )
                                    .unwrap_or_else(|error| {
                                        unreachable!("queue_update is lane-scoped: {error}")
                                    }),
                                );
                            }
                            events
                        })),
                    })
                })
            },
            context,
        )
        .await
    }

    /// Admits a compaction operation, upstream's `acceptCompaction`. The
    /// preparation computation rides the compaction child; the staged seam
    /// raises until it lands.
    async fn accept_compaction(
        &self,
        request: &OperationRequest,
        operation_id: String,
        started_at: i64,
        acceptance_config: Config,
        context: &Context,
    ) -> Result<OperationAdmissionResult, LaneError> {
        let _ = (request, operation_id, started_at, acceptance_config, context);
        Err(Arc::new(SliceNotImplemented::new("compaction")))
    }

    /// Admits a navigation operation, upstream's `acceptNavigation`. The
    /// summarized path's branch preparation rides the compaction child; the
    /// staged seam raises before the retry loop until it lands.
    async fn accept_navigation(
        &self,
        request: &OperationRequest,
        operation_id: String,
        started_at: i64,
        acceptance_config: Config,
        context: &Context,
    ) -> Result<OperationAdmissionResult, LaneError> {
        let OperationRequest::Navigation {
            target_id,
            options,
            ..
        } = request
        else {
            return Err(lane_error(SessionInvariantError(
                "accept_navigation takes navigation requests".to_owned(),
            )));
        };
        let options = options.clone().unwrap_or_default();
        let summarize = options.summarize.unwrap_or(false);
        if summarize {
            return Err(lane_error(SliceNotImplemented::new("summarized navigation")));
        }
        let target_id = target_id.clone();
        let options_label = options.label.clone();
        let options_custom_instructions = options.custom_instructions.clone();
        let lane_name = self.inner.name.clone();

        self.command::<OperationAdmissionResult, _>(
            move |state, session, command_context| {
                Box::pin(async move {
                    if let Some(operation) = &state.operation {
                        return Ok(LaneCommand::Return {
                            result: Err(HarnessError::LaneBusy {
                                lane: lane_name.clone(),
                                operation_id: operation.meta.operation_id.clone(),
                                operation_kind: intent_kind_of(&operation.meta),
                                message: format!(
                                    "Lane {} already has an active operation",
                                    quoted(&lane_name)
                                ),
                            }),
                        });
                    }
                    if target_id == state.tip_id {
                        return Ok(LaneCommand::Return {
                            result: Err(HarnessError::InvalidNavigation {
                                lane: lane_name.clone(),
                                reason: "current_tip".to_owned(),
                                message: "Navigation target must differ from the current tip".to_owned(),
                            }),
                        });
                    }
                    if target_id.is_none() && options_label.is_some() {
                        return Ok(LaneCommand::Return {
                            result: Err(HarnessError::InvalidNavigation {
                                lane: lane_name.clone(),
                                reason: "root_label".to_owned(),
                                message: "Root navigation cannot set a label".to_owned(),
                            }),
                        });
                    }
                    if let Some(target_id) = &target_id {
                        let found = session
                            .get_entries(vec![target_id.clone()], &command_context)
                            .await
                            .map_err(|error| lane_error(error))?;
                        if !found.contains_key(target_id) {
                            return Ok(LaneCommand::Return {
                                result: Err(HarnessError::UnknownTarget {
                                    target_id: target_id.clone(),
                                    message: format!("Unknown target: {target_id}"),
                                }),
                            });
                        }
                    }

                    let meta = OperationMeta {
                        operation_id: operation_id.clone(),
                        lane: lane_name.clone(),
                        source_tip_id: state.tip_id.clone(),
                        started_at,
                        intent: crate::harness::session::types::OperationIntent::Navigation {
                            target_id: target_id.clone(),
                            summarize: false,
                            label: options_label.clone(),
                            custom_instructions: options_custom_instructions.clone(),
                        },
                    };
                    let operation_state = OperationState::NavigationReadyToCommit(NavigationReadyToCommitOperation {
                        scope: OperationScope {
                            control: Control::Running,
                            settings: captured_settings(&acceptance_config),
                            latest_assistant_entry_id: None,
                        },
                        target_id: target_id.clone(),
                        label: options_label.clone(),
                    });
                    let mut writes: Vec<Write> = Vec::new();
                    writes.push(
                        set_value(&operation_meta(&operation_id), meta.clone())
                            .map_err(|error| lane_error(error))?,
                    );
                    writes.push(
                        set_value(&operation_state(&operation_id), operation_state.clone())
                            .map_err(|error| lane_error(error))?,
                    );
                    writes.push(
                        set_value(
                            &lane_state(&lane_name),
                            durable_lane_state(&state, Some(&operation_id), &state.inbox, state.last_operation_id.as_deref()),
                        )
                        .map_err(|error| lane_error(error))?,
                    );
                    let mut next = state.clone();
                    next.operation = Some(LiveOperation {
                        meta: meta.clone(),
                        state: operation_state.clone(),
                    });
                    let events_lane = lane_name.clone();
                    let events_target = target_id.clone();
                    Ok(LaneCommand::Commit {
                        writes,
                        next,
                        materialize: Arc::new(move |_commit: &CommitResult| {
                            Ok(OperationAdmission {
                                operation_id: operation_id.clone(),
                                kind: OperationKind::Navigation,
                                started_at,
                            })
                        }),
                        events: Some(Arc::new(move |_commit: &CommitResult| {
                            vec![HarnessEvent::lane_scoped(
                                &events_lane,
                                false,
                                HarnessEventPayload::NavigationStart {
                                    run_id: operation_id.clone(),
                                    target_id: events_target.clone(),
                                    started_at,
                                },
                            )
                            .unwrap_or_else(|error| unreachable!("navigation_start is lane-scoped: {error}"))]
                        })),
                    })
                })
            },
            context,
        )
        .await
    }

    /// Drives the current operation, upstream's `drive`.
    ///
    /// The claim loop installs one pass or joins an existing one; the
    /// procedure loop (`driveOperation`) rides the drive child, so a freshly
    /// installed pass faults with the staged raise until that child lands.
    ///
    /// # Errors
    /// The sealed error when the lane sealed with a fault; the drive's
    /// settled outcome rides the inner result.
    pub async fn drive_impl(
        &self,
        options: DriveOptions,
        context: &Context,
    ) -> Result<DriveResult, LaneError> {
        if self.inner.is_harness_closed() {
            return Ok(Err(HarnessError::Closed {
                message: self.sealed_message(),
            }));
        }
        self.inner.assert_open()?;
        let lane_name = self.inner.name.clone();
        loop {
            let claim = self
                .command::<DriveClaim, _>(
                    {
                        let options = options.clone();
                        let lane_name = lane_name.clone();
                        move |state, session, command_context| {
                            Box::pin(async move {
                                if let Some(signal) = command_context.abort_signal()
                                    && signal.aborted()
                                {
                                    return Ok(LaneCommand::Reject {
                                        error: abort_reason_error(signal.reason()),
                                    });
                                }
                                if let Some(operation) = &state.operation {
                                    if operation.meta.operation_id == options.operation_id {
                                        let active = lock_core(_lane_shared_of(command_context)).active_drive.clone();
                                        // The active drive reads ride the lane
                                        // shared state captured below.
                                        let _ = active;
                                        return Ok(LaneCommand::Return {
                                            result: DriveClaim::Observe {
                                                drive: Arc::new(Drive::new(&options, &command_context)),
                                                installed: true,
                                            },
                                        });
                                    }
                                }
                                let stored = session
                                    .get_value(&operation_result(&options.operation_id).address, &command_context)
                                    .await
                                    .map_err(|error| lane_error(error))?;
                                match stored {
                                    Some(stored) => {
                                        let record: OperationResultRecord = serde_json::from_value(stored.value)
                                            .map_err(|error| {
                                                lane_error(SessionInvariantError(format!(
                                                    "Operation result is malformed: {error}"
                                                )))
                                            })?;
                                        Ok(LaneCommand::Return {
                                            result: DriveClaim::Settled { outcome: record },
                                        })
                                    }
                                    None => Ok(LaneCommand::Return {
                                        result: DriveClaim::Mismatch {
                                            error: HarnessError::OperationMismatch {
                                                lane: lane_name.clone(),
                                                expected_operation_id: options.operation_id.clone(),
                                                current_operation_id: state
                                                    .operation
                                                    .as_ref()
                                                    .map(|operation| operation.meta.operation_id.clone()),
                                                last_operation_id: state.last_operation_id.clone(),
                                                message: format!(
                                                    "Operation {} does not own lane {}",
                                                    options.operation_id,
                                                    quoted(&lane_name)
                                                ),
                                            },
                                        },
                                    }),
                                }
                            })
                        }
                    },
                    context,
                )
                .await?;

            match claim {
                DriveClaim::Settled { outcome } => {
                    return Ok(DriveOutcome::Settled { outcome });
                }
                DriveClaim::Mismatch { error } => return Err(Arc::from(lane_error(error))),
                DriveClaim::Occupied { drive } => {
                    if let Err(reason) = await_with_context(drive.completion(), context).await {
                        return Err(LaneOperationError::Closed(HarnessError::Closed {
                            message: abort_reason_error(Some(reason)).to_string(),
                        }));
                    }
                    continue;
                }
                DriveClaim::Observe { drive, installed } => {
                    if installed {
                        // The drive child owns the procedure loop; the staged
                        // seam faults the pass deterministically, mirroring
                        // the real spawn handler's failure path.
                        let fault = self.fault(
                            from_arc(Arc::new(SliceNotImplemented::new("drive operation"))
                               ),
                            &drive.context,
                        );
                        {
                            let mut core = lock_core(&self.inner);
                            if core
                                .active_drive
                                .as_ref()
                                .is_some_and(|active| Arc::ptr_eq(active, &drive))
                            {
                                core.active_drive = None;
                                let _ = core.state_change.send_modify(|version| *version += 1);
                            }
                        }
                        drive.fail(fault);
                    }
                    return match await_with_context(drive.completion(), context).await {
                        Ok(outcome) => Ok(outcome),
                        Err(reason) => Err(LaneOperationError::Closed(HarnessError::Closed {
                            message: abort_reason_error(Some(reason)).to_string(),
                        })),
                    };
                }
            }
        }
    }

    /// Requests durable cancellation of one operation, upstream's
    /// `requestOperationAbort` — the package-private primitive; public
    /// exposure remains guarded until the drive child lands.
    ///
    /// # Errors
    /// The sealed error when the lane sealed with a fault; the request's
    /// taxonomy errors ride the inner result.
    pub async fn request_operation_abort(
        &self,
        operation_id: &str,
        context: &Context,
    ) -> Result<AbortRequestResult, LaneError> {
        if self.inner.is_harness_closed() {
            return Ok(Err(HarnessError::Closed {
                message: self.sealed_message(),
            }));
        }
        self.inner.assert_open()?;

        let drive = {
            let core = lock_core(&self.inner);
            core.active_drive
                .clone()
                .filter(|drive| drive.operation_id == operation_id)
        };
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        if let Some(drive) = &drive {
            drive.begin_abort(cancel_rx.clone());
        }
        let gate_settled = Arc::new(AtomicBool::new(false));
        let settle_gate = {
            let gate_settled = Arc::clone(&gate_settled);
            let cancel_tx = cancel_tx.clone();
            let drive = drive.clone();
            Arc::new(move |signal: bool| {
                if !gate_settled.swap(true, Ordering::SeqCst) {
                    let _ = cancel_tx.send(true);
                    if signal {
                        if let Some(drive) = &drive {
                            drive.signal_abort();
                        }
                    }
                }
            })
        };

        let operation_id_owned = operation_id.to_owned();
        let lane_name = self.inner.name.clone();
        let materialize_settle = Arc::clone(&settle_gate);
        let result = self
            .command::<AbortRequestResult, _>(
                move |state, session, command_context| {
                    let operation_id = operation_id_owned.clone();
                    let lane_name = lane_name.clone();
                    let settle_gate = Arc::clone(&materialize_settle);
                    Box::pin(async move {
                        let operation = match &state.operation {
                            Some(operation) if operation.meta.operation_id == operation_id => operation.clone(),
                            other => {
                                return Ok(LaneCommand::Return {
                                    result: Err(mismatch_for(
                                        &lane_name,
                                        &operation_id,
                                        other.map(|operation| operation.meta.operation_id.clone()),
                                        state.last_operation_id.clone(),
                                    )),
                                });
                            }
                        };
                        if matches!(
                            crate::harness::session::types::operation_scope_of(&operation.state).control,
                            Control::CancelRequested
                        ) {
                            return Ok(LaneCommand::Return {
                                result: Ok(AbortRequestOutcome {
                                    operation_id,
                                    newly_requested: false,
                                    steer: Vec::new(),
                                    follow_up: Vec::new(),
                                }),
                            });
                        }

                        let removed: Vec<InboxItem> = state
                            .inbox
                            .iter()
                            .filter(|item| matches!(item.kind, InboxItemKind::Steer | InboxItemKind::FollowUp))
                            .cloned()
                            .collect();
                        let mut payloads: Vec<(InboxItemKind, AgentMessage)> = Vec::new();
                        for item in &removed {
                            let stored = session
                                .get_value(&pending_entry(&item.entry_id).address, &command_context)
                                .await
                                .map_err(|error| lane_error(error))?;
                            let pending = match stored {
                                Some(StoredValue { value, .. }) => {
                                    serde_json::from_value::<PendingEntry>(value).map_err(|_| {
                                        Arc::new(SessionInvariantError(format!(
                                            "Pending {} entry {} is missing its message",
                                            inbox_kind_name(item.kind),
                                            item.entry_id
                                        )))
                                    })?
                                }
                                None => {
                                    return Err(lane_error(SessionInvariantError(format!(
                                        "Pending {} entry {} is missing its message",
                                        inbox_kind_name(item.kind),
                                        item.entry_id
                                    ))));
                                }
                            };
                            let PendingEntry::Message { payload } = pending else {
                                return Err(lane_error(SessionInvariantError(format!(
                                    "Pending {} entry {} is missing its message",
                                    inbox_kind_name(item.kind),
                                    item.entry_id
                                ))));
                            };
                            payloads.push((item.kind, *payload));
                        }
                        let steer: Vec<AgentMessage> = payloads
                            .iter()
                            .filter(|(kind, _)| *kind == InboxItemKind::Steer)
                            .map(|(_, message)| message.clone())
                            .collect();
                        let follow_up: Vec<AgentMessage> = payloads
                            .iter()
                            .filter(|(kind, _)| *kind == InboxItemKind::FollowUp)
                            .map(|(_, message)| message.clone())
                            .collect();
                        let removed_ids: std::collections::BTreeSet<&str> =
                            removed.iter().map(|item| item.entry_id.as_str()).collect();
                        let inbox: Vec<InboxItem> = state
                            .inbox
                            .iter()
                            .filter(|item| !removed_ids.contains(item.entry_id.as_str()))
                            .cloned()
                            .collect();
                        let queues = read_lane_queues(session.as_ref(), &inbox, &command_context)
                            .await
                            .map_err(|error| lane_error(error))?;
                        let scope =
                            crate::harness::session::types::operation_scope_of(&operation.state);
                        let mut scope = scope;
                        scope.control = Control::CancelRequested {
                            requested_at: crate::harness::env::nodejs::now_ms(),
                        };
                        let operation_state = operation_state_with_scope(&operation.state, scope);

                        let mut writes: Vec<Write> = Vec::new();
                        for item in &removed {
                            writes.push(delete_value(&pending_entry(&item.entry_id)));
                        }
                        writes.push(
                            set_value(&operation_state(&operation_id), operation_state.clone())
                                .map_err(|error| lane_error(error))?,
                        );
                        writes.push(
                            set_value(
                                &lane_state(&lane_name),
                                durable_lane_state(&state, Some(&operation_id), &inbox, state.last_operation_id.as_deref()),
                            )
                            .map_err(|error| lane_error(error))?,
                        );
                        let mut next = state.clone();
                        next.inbox = inbox.clone();
                        next.operation = Some(LiveOperation {
                            meta: operation.meta.clone(),
                            state: operation_state.clone(),
                        });

                        let events_lane = lane_name.clone();
                        let events_operation_id = operation_id.clone();
                        let events_steer = steer.clone();
                        let events_follow_up = follow_up.clone();
                        let events_queues = queues.clone();
                        let removed_count = removed.len();
                        Ok(LaneCommand::Commit {
                            writes,
                            next,
                            materialize: Arc::new(move |_commit: &CommitResult| {
                                (settle_gate)(true);
                                Ok(AbortRequestOutcome {
                                    operation_id: operation_id.clone(),
                                    newly_requested: true,
                                    steer: steer.clone(),
                                    follow_up: follow_up.clone(),
                                })
                            }),
                            events: Some(Arc::new(move |_commit: &CommitResult| {
                                let mut events = vec![HarnessEvent::lane_scoped(
                                    &events_lane,
                                    false,
                                    HarnessEventPayload::OperationAbort {
                                        operation_id: events_operation_id.clone(),
                                        steer: events_steer.clone(),
                                        follow_up: events_follow_up.clone(),
                                    },
                                )
                                .unwrap_or_else(|error| {
                                    unreachable!("operation_abort is lane-scoped: {error}")
                                })];
                                if removed_count > 0 {
                                    events.push(
                                        HarnessEvent::lane_scoped(
                                            &events_lane,
                                            false,
                                            HarnessEventPayload::QueueUpdate {
                                                queues: events_queues.clone(),
                                            },
                                        )
                                        .unwrap_or_else(|error| {
                                            unreachable!("queue_update is lane-scoped: {error}")
                                        }),
                                    );
                                }
                                events
                            })),
                        })
                    })
                },
                context,
            )
            .await;

        match result {
            Ok(result) => {
                let settle_now = match &result {
                    Ok(outcome) => !outcome.newly_requested,
                    Err(_) => false,
                };
                (settle_gate)(settle_now);
                Ok(result)
            }
            Err(error) => {
                // Upstream rejects the cancellation promise so the drive's
                // abort path fails with it; the gate's landed Cancellation
                // type has no rejection channel, so the release lands here
                // and the request's own rejection carries the failure.
                if !gate_settled.load(Ordering::SeqCst) {
                    let _ = cancel_tx.send(true);
                }
                Err(error)
            }
        }
    }

    /// The execution view one lane reports, upstream's `inspectExecution`.
    pub async fn inspect_execution_impl(
        &self,
        context: &Context,
    ) -> Result<LaneExecutionInfo, LaneError> {
        self.read_lane::<LaneExecutionInfo, _>(
            {
                let lane_name = self.inner.name.clone();
                move |state, _session, _context| {
                    Box::pin(async move {
                        let operation = &state.operation;
                        let captured = operation
                            .as_ref()
                            .map(|operation| captured_model(&operation.state));
                        let current = operation.as_ref().map(|operation| {
                            let status = match crate::harness::session::types::operation_scope_of(&operation.state)
                                .control
                            {
                                Control::CancelRequested { .. } => OperationStatus::Aborting,
                                Control::Running => OperationStatus::Open,
                            };
                            CurrentOperationInfo {
                                id: operation.meta.operation_id.clone(),
                                kind: intent_kind_of(&operation.meta),
                                status,
                                started_at: operation.meta.started_at,
                                captured_model: match captured {
                                    Some(Some(model)) => Some(model),
                                    _ => None,
                                },
                            }
                        });
                        Ok(LaneExecutionInfo {
                            lane: lane_name.clone(),
                            tip_id: state.tip_id.clone(),
                            configured_model: state.configuration.model.clone(),
                            current,
                            last_operation_id: state.last_operation_id.clone(),
                        })
                    })
                }
            },
            context,
        )
        .await
    }

    /// Runs a text prompt to settlement, upstream's `prompt(text, images)`.
    pub async fn prompt_text_impl(
        &self,
        text: &str,
        images: Option<Vec<ImageContent>>,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::RunResult, LaneError> {
        self.drive_run_request(
            OperationRequest::Prompt {
                operation_id: None,
                prompt: lane_error(PromptMessagesPayload::Text {
                    prompt: text.to_owned(),
                    images,
                }),
            },
            context,
        )
        .await
    }

    /// Runs a prebuilt prompt to settlement, upstream's
    /// `prompt(message | messages)`.
    pub async fn prompt_messages_impl(
        &self,
        prompt: PromptMessagesPayload,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::RunResult, LaneError> {
        self.drive_run_request(
            OperationRequest::Prompt {
                operation_id: None,
                prompt: Box::new(prompt),
            },
            context,
        )
        .await
    }

    async fn drive_run_request(
        &self,
        request: OperationRequest,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::RunResult, LaneError> {
        let admission = self.accept_impl(request, context).await?;
        let admission = match admission {
            Ok(admission) => admission,
            Err(error) => {
                return match &error {
                    HarnessError::LaneBusy { .. }
                    | HarnessError::InvalidMessage { .. }
                    | HarnessError::UnknownSkill { .. }
                    | HarnessError::UnknownTemplate { .. }
                    | HarnessError::Closed { .. } => Ok(Err(error)),
                    _ => Err(self.fault(
                        from_arc(Arc::new(SessionInvariantError(format!(
                            "Run acceptance returned {}",
                            error.tag()
                        )))),
                        context,
                    )),
                };
            }
        };
        let driven = self
            .drive_impl(
                DriveOptions {
                    operation_id: admission.operation_id.clone(),
                    wait_for_retry: Some(true),
                    poll_deferred: None,
                },
                context,
            )
            .await?;
        let driven = match driven {
            Ok(driven) => driven,
            Err(error) => {
                return match &error {
                    HarnessError::Closed { .. } => Ok(Err(error)),
                    _ => Err(self.fault(
                        from_arc(Arc::new(SessionInvariantError(format!(
                            "Accepted run {} no longer matches its lane",
                            admission.operation_id
                        )))),
                        context,
                    )),
                };
            }
        };
        match driven {
            DriveOutcome::Settled { outcome } => Ok(Ok(RunOutcome::Settled(outcome))),
            DriveOutcome::Waiting {
                reason: crate::harness::agent_harness::DriveWaitReason::Deferred { deferred },
                ..
            } => Ok(Ok(RunOutcome::Suspended(SuspendedRun {
                operation_id: admission.operation_id,
                status: "suspended".to_owned(),
                deferred,
            }))),
            DriveOutcome::Waiting {
                reason: crate::harness::agent_harness::DriveWaitReason::Retry { .. },
                ..
            } => Err(self.fault(
                from_arc(Arc::new(SessionInvariantError(format!(
                    "Run {} returned an unwaited retry",
                    admission.operation_id
                )))),
                context,
            )),
        }
    }

    /// Invokes one skill, upstream's `skill`.
    pub async fn skill_impl(
        &self,
        name: &str,
        additional_instructions: Option<String>,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::RunResult, LaneError> {
        self.drive_run_request(
            OperationRequest::Skill {
                operation_id: None,
                name: name.to_owned(),
                additional_instructions,
            },
            context,
        )
        .await
    }

    /// Invokes one prompt template, upstream's `promptFromTemplate`.
    pub async fn prompt_from_template_impl(
        &self,
        name: &str,
        args: Option<Vec<String>>,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::RunResult, LaneError> {
        self.drive_run_request(
            OperationRequest::PromptTemplate {
                operation_id: None,
                name: name.to_owned(),
                args,
            },
            context,
        )
        .await
    }

    /// Compacts the history, upstream's `compact`. The compaction acceptance
    /// rides the compaction child; the staged seam raises until it lands.
    ///
    /// # Errors
    /// The sealed error when the lane sealed with a fault.
    pub async fn compact_impl(
        &self,
        options: Option<crate::harness::agent_harness::CompactRequestOptions>,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::CompactionResult, LaneError> {
        let request = OperationRequest::Compaction {
            operation_id: None,
            custom_instructions: options.and_then(|options| options.custom_instructions),
        };
        let admission = self.accept_impl(request, context).await?;
        let admission = match admission {
            Ok(admission) => admission,
            Err(error) => {
                return match &error {
                    HarnessError::LaneBusy { .. }
                    | HarnessError::NothingToCompact { .. }
                    | HarnessError::Closed { .. } => Ok(Err(error)),
                    _ => Err(self.fault(
                        from_arc(Arc::new(SessionInvariantError(format!(
                            "Compaction acceptance returned {}",
                            error.tag()
                        )))),
                        context,
                    )),
                };
            }
        };
        let compacted = self.drive_structural_admission(&admission, context).await?;
        match compacted {
            Err(error) => Ok(Err(error)),
            Ok(record) => {
                let continuation = self.continue_after_structural(&record, context).await?;
                match continuation {
                    Err(error) => Ok(Err(error)),
                    Ok(continuation) => Ok(Ok(crate::harness::agent_harness::CompactionOutcome {
                        compaction: record,
                        run: continuation.map(RunFollowUp::from_record_or_suspended),
                    })),
                }
            }
        }
    }

    /// Navigates the tree, upstream's `navigateTree`. The summarized path's
    /// preparation rides the compaction child; the staged seam raises until
    /// it lands.
    ///
    /// # Errors
    /// The sealed error when the lane sealed with a fault.
    pub async fn navigate_tree_impl(
        &self,
        target_id: Option<String>,
        options: Option<crate::harness::agent_harness::NavigateOptions>,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::NavigationResult, LaneError> {
        let request = OperationRequest::Navigation {
            operation_id: None,
            target_id,
            options,
        };
        let admission = self.accept_impl(request, context).await?;
        let admission = match admission {
            Ok(admission) => admission,
            Err(error) => {
                return match &error {
                    HarnessError::LaneBusy { .. }
                    | HarnessError::InvalidNavigation { .. }
                    | HarnessError::UnknownTarget { .. }
                    | HarnessError::Closed { .. } => Ok(Err(error)),
                    _ => Err(self.fault(
                        from_arc(Arc::new(SessionInvariantError(format!(
                            "Navigation acceptance returned {}",
                            error.tag()
                        )))),
                        context,
                    )),
                };
            }
        };
        let navigated = self.drive_structural_admission(&admission, context).await?;
        match navigated {
            Err(error) => Ok(Err(error)),
            Ok(record) => {
                let continuation = self.continue_after_structural(&record, context).await?;
                match continuation {
                    Err(error) => Ok(Err(error)),
                    Ok(continuation) => Ok(Ok(crate::harness::agent_harness::NavigationOutcome {
                        navigation: record,
                        run: continuation.map(RunFollowUp::from_record_or_suspended),
                    })),
                }
            }
        }
    }

    async fn drive_structural_admission(
        &self,
        admission: &OperationAdmission,
        context: &Context,
    ) -> Result<Result<OperationResultRecord, HarnessError>, LaneError> {
        let driven = self
            .drive_impl(
                DriveOptions {
                    operation_id: admission.operation_id.clone(),
                    wait_for_retry: Some(true),
                    poll_deferred: None,
                },
                context,
            )
            .await?;
        let driven = match driven {
            Ok(driven) => driven,
            Err(error) => {
                return Err(self.fault(
                    from_arc(Arc::new(SessionInvariantError(format!(
                        "Accepted {} {} no longer matches its lane",
                        admission_kind_name(admission.kind),
                        admission.operation_id
                    )))),
                    context,
                ));
            }
        };
        Ok(match driven {
            DriveOutcome::Settled { outcome } => Ok(outcome),
            DriveOutcome::Waiting { reason, .. } => Err(self.fault(
                from_arc(Arc::new(SessionInvariantError(format!(
                    "{} {} returned {}",
                    admission_kind_name(admission.kind),
                    admission.operation_id,
                    wait_reason_name(&reason)
                )))),
                context,
            )),
        })
    }

    async fn continue_after_structural(
        &self,
        record: &OperationResultRecord,
        context: &Context,
    ) -> Result<Result<Option<crate::harness::agent_harness::RunFollowUp>, HarnessError>, LaneError> {
        if record.status == crate::harness::session::types::TerminalStatus::Aborted {
            return Ok(Ok(None));
        }
        let admission = self
            .accept_impl(
                OperationRequest::Prompt {
                    operation_id: None,
                    prompt: Box::new(PromptMessagesPayload::Text {
                        prompt: String::new(),
                        images: None,
                    }),
                },
                context,
            )
            .await?;
        let admission = match admission {
            Ok(admission) => admission,
            Err(error) => {
                return match &error {
                    HarnessError::InvalidMessage { .. } | HarnessError::LaneBusy { .. } => Ok(Ok(None)),
                    HarnessError::Closed { .. } => Ok(Err(error)),
                    _ => Err(self.fault(
                        from_arc(Arc::new(SessionInvariantError(format!(
                            "Structural continuation acceptance returned {}",
                            error.tag()
                        )))),
                        context,
                    )),
                };
            }
        };
        let driven = self
            .drive_impl(
                DriveOptions {
                    operation_id: admission.operation_id.clone(),
                    wait_for_retry: Some(true),
                    poll_deferred: None,
                },
                context,
            )
            .await?;
        let driven = match driven {
            Ok(driven) => driven,
            Err(error) => {
                return Err(self.fault(
                    from_arc(Arc::new(SessionInvariantError(format!(
                        "Continuation run {} no longer matches its lane",
                        admission.operation_id
                    )))),
                    context,
                ));
            }
        };
        Ok(match driven {
            DriveOutcome::Settled { outcome } => Ok(Some(RunFollowUp::Settled(outcome))),
            DriveOutcome::Waiting {
                reason: crate::harness::agent_harness::DriveWaitReason::Deferred { deferred },
                ..
            } => Ok(Some(RunFollowUp::Suspended(SuspendedRun {
                operation_id: admission.operation_id,
                status: "suspended".to_owned(),
                deferred,
            }))),
            DriveOutcome::Waiting {
                reason: crate::harness::agent_harness::DriveWaitReason::Retry { .. },
                ..
            } => Err(self.fault(
                from_arc(Arc::new(SessionInvariantError(format!(
                    "Continuation run {} returned an unwaited retry",
                    admission.operation_id
                )))),
                context,
            )),
        })
    }

    /// Resumes a suspended run, upstream's `resume`.
    ///
    /// # Errors
    /// The sealed error when the lane sealed with a fault.
    pub async fn resume_impl(
        &self,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::ResumeResult, LaneError> {
        if self.inner.is_harness_closed() {
            return Ok(Err(HarnessError::Closed {
                message: self.sealed_message(),
            }));
        }
        self.inner.assert_open()?;
        let lane_name = self.inner.name.clone();
        let inspected = self
            .command::<Result<String, HarnessError>, _>(
                move |state, _session, _context| {
                    Box::pin(async move {
                        match &state.operation {
                            Some(operation) => Ok(LaneCommand::Return {
                                result: Ok(operation.meta.operation_id.clone()),
                            }),
                            None => Ok(LaneCommand::Return {
                                result: Err(HarnessError::NothingToResume {
                                    lane: lane_name.clone(),
                                    message: format!(
                                        "Lane {} has no active operation to resume",
                                        quoted(&lane_name)
                                    ),
                                }),
                            }),
                        }
                    })
                },
                context,
            )
            .await?;
        let operation_id = match inspected {
            Ok(operation_id) => operation_id,
            Err(error) => return Ok(Err(error)),
        };

        let driven = self
            .drive_impl(
                DriveOptions {
                    operation_id: operation_id.clone(),
                    poll_deferred: Some(true),
                    wait_for_retry: Some(true),
                },
                context,
            )
            .await?;
        let driven = match driven {
            Ok(driven) => driven,
            Err(error) => {
                return Err(self.fault(
                    from_arc(Arc::new(SessionInvariantError(format!(
                        "Operation {operation_id} no longer matches its lane"
                    )))),
                    context,
                ));
            }
        };
        match driven {
            DriveOutcome::Settled { outcome } => Ok(Ok(RunOutcome::Settled(outcome))),
            DriveOutcome::Waiting {
                reason: crate::harness::agent_harness::DriveWaitReason::Deferred { deferred },
                ..
            } => Ok(Ok(RunOutcome::Suspended(SuspendedRun {
                operation_id,
                status: "suspended".to_owned(),
                deferred,
            }))),
            DriveOutcome::Waiting {
                reason: crate::harness::agent_harness::DriveWaitReason::Retry { .. },
                ..
            } => Err(self.fault(
                from_arc(Arc::new(SessionInvariantError(format!(
                    "Operation {operation_id} returned an unwaited retry"
                )))),
                context,
            )),
        }
    }

    /// Aborts the active operation, upstream's `abort`.
    ///
    /// # Errors
    /// The sealed error when the lane sealed with a fault.
    pub async fn abort_impl(&self, context: &Context) -> Result<crate::harness::agent_harness::AbortResult, LaneError> {
        if self.inner.is_harness_closed() {
            return Ok(Err(HarnessError::Closed {
                message: self.sealed_message(),
            }));
        }
        self.inner.assert_open()?;
        let lane_name = self.inner.name.clone();
        let operation_id = self
            .command::<Option<String>, _>(
                move |state, _session, _context| {
                    Box::pin(async move {
                        Ok(LaneCommand::Return {
                            result: state
                                .operation
                                .as_ref()
                                .map(|operation| operation.meta.operation_id.clone()),
                        })
                    })
                },
                context,
            )
            .await?;
        let Some(operation_id) = operation_id else {
            return Ok(Err(HarnessError::NoActiveOperation {
                lane: lane_name.clone(),
                message: format!("Lane {} has no active operation", quoted(&lane_name)),
            }));
        };
        let requested = self.request_operation_abort(&operation_id, context).await?;
        let requested = match requested {
            Ok(requested) => requested,
            Err(error) => {
                return match &error {
                    HarnessError::Closed { .. } => Ok(Err(error.clone())),
                    _ => Ok(Err(HarnessError::NoActiveOperation {
                        lane: lane_name.clone(),
                        message: format!(
                            "Lane {} no longer has the inspected operation",
                            quoted(&lane_name)
                        ),
                    })),
                };
            }
        };
        let driven = self
            .drive_impl(
                DriveOptions {
                    operation_id: operation_id.clone(),
                    wait_for_retry: None,
                    poll_deferred: None,
                },
                context,
            )
            .await?;
        if let Err(error) = driven {
            return match &error {
                HarnessError::Closed { .. } => Ok(Err(error.clone())),
                _ => Err(self.fault(
                    from_arc(Arc::new(SessionInvariantError(format!(
                        "Cancelled operation {operation_id} no longer matches its lane"
                    )))),
                    context,
                )),
            };
        }
        Ok(Ok(AbortOutcome {
            operation_id,
            steer: requested.steer,
            follow_up: requested.follow_up,
        }))
    }

    /// Queues one steering message, upstream's `steer`.
    pub async fn steer_impl(
        &self,
        message: QueueMessage,
        images: QueueImages,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::QueueResult, LaneError> {
        self.enqueue(InboxItemKind::Steer, message, images, context)
            .await
    }

    /// Queues one follow-up message, upstream's `followUp`.
    pub async fn follow_up_impl(
        &self,
        message: QueueMessage,
        images: QueueImages,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::QueueResult, LaneError> {
        self.enqueue(InboxItemKind::FollowUp, message, images, context)
            .await
    }

    /// Queues one next-run message, upstream's `nextRun`.
    pub async fn next_run_impl(
        &self,
        message: QueueMessage,
        images: QueueImages,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::QueueResult, LaneError> {
        self.enqueue(InboxItemKind::NextRun, message, images, context)
            .await
    }

    async fn enqueue(
        &self,
        kind: InboxItemKind,
        input: QueueMessage,
        images: QueueImages,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::QueueResult, LaneError> {
        if self.inner.is_harness_closed() {
            return Ok(Err(HarnessError::Closed {
                message: self.sealed_message(),
            }));
        }
        self.inner.assert_open()?;
        let at = crate::harness::env::nodejs::now_ms();
        let lane_name = self.inner.name.clone();
        let message = match input {
            QueueMessage::Text(text) => {
                if text.is_empty() && images.is_empty() {
                    return Ok(Err(HarnessError::InvalidMessage {
                        lane: lane_name.clone(),
                        reason: "empty".to_owned(),
                        message: "Queued input must contain text or an image".to_owned(),
                    }));
                }
                let mut content: Vec<UserBlock> = Vec::new();
                if !text.is_empty() {
                    content.push(UserBlock::Text(TextContent {
                        text,
                        text_signature: None,
                    }));
                }
                content.extend(images.iter().cloned().map(UserBlock::Image));
                AgentMessage::Standard(Message::User(UserMessage {
                    content: UserContent::Blocks(content),
                    timestamp: at,
                }))
            }
            QueueMessage::Message(input) => {
                let input = *input;
                if let AgentMessage::Standard(Message::Assistant(assistant)) = &input {
                    if assistant.stop_reason == StopReason::Pending {
                        return Ok(Err(HarnessError::InvalidMessage {
                            lane: lane_name.clone(),
                            reason: "pending_assistant".to_owned(),
                            message: "Cannot queue a pending assistant message".to_owned(),
                        }));
                    }
                }
                let is_user = matches!(
                    &input,
                    AgentMessage::Standard(Message::User(_))
                );
                if !images.is_empty() && !is_user {
                    return Ok(Err(HarnessError::InvalidMessage {
                        lane: lane_name.clone(),
                        reason: "images_with_non_user".to_owned(),
                        message: "Images can be added only to queued user messages".to_owned(),
                    }));
                }
                if images.is_empty() || !is_user {
                    input
                } else {
                    let AgentMessage::Standard(Message::User(user)) = &input else {
                        unreachable!("is_user checked above")
                    };
                    let mut content: Vec<UserBlock> = match &user.content {
                        UserContent::Text(text) => {
                            if text.is_empty() {
                                Vec::new()
                            } else {
                                vec![UserBlock::Text(TextContent {
                                    text: text.clone(),
                                    text_signature: None,
                                })]
                            }
                        }
                        UserContent::Blocks(blocks) => blocks.clone(),
                    };
                    content.extend(images.iter().cloned().map(UserBlock::Image));
                    AgentMessage::Standard(Message::User(UserMessage {
                        content: UserContent::Blocks(content),
                        timestamp: user.timestamp,
                    }))
                }
            }
        };
        let entry_id = self.inner.session.id_generator().next(Some(at));

        self.command::<crate::harness::agent_harness::QueueResult, _>(
            move |state, session, command_context| {
                let kind = kind;
                let message = message.clone();
                let entry_id = entry_id.clone();
                let lane_name = lane_name.clone();
                Box::pin(async move {
                    let mut inbox = state.inbox.clone();
                    inbox.push(InboxItem {
                        entry_id: entry_id.clone(),
                        kind,
                    });
                    let mut queues =
                        read_lane_queues(session.as_ref(), &state.inbox, &command_context)
                            .await
                            .map_err(|error| lane_error(error))?;
                    queues.push(LaneQueuedItem::Message {
                        entry_id: entry_id.clone(),
                        kind,
                        message: Box::new(message.clone()),
                    });
                    let mut writes: Vec<Write> = Vec::new();
                    writes.push(
                        set_value(
                            &pending_entry(&entry_id),
                            PendingEntry::Message {
                                payload: Box::new(message.clone()),
                            },
                        )
                        .map_err(|error| lane_error(error))?,
                    );
                    writes.push(
                        set_value(
                            &lane_state(&lane_name),
                            durable_lane_state(
                                &state,
                                state.operation.as_ref().map(|operation| operation.meta.operation_id.as_str()),
                                &inbox,
                                state.last_operation_id.as_deref(),
                            ),
                        )
                        .map_err(|error| lane_error(error))?,
                    );
                    let mut next = state.clone();
                    next.inbox = inbox;
                    let events_lane = lane_name.clone();
                    let events_queues = queues.clone();
                    Ok(LaneCommand::Commit {
                        writes,
                        next,
                        materialize: Arc::new(move |_commit: &CommitResult| Ok(entry_id.clone())),
                        events: Some(Arc::new(move |_commit: &CommitResult| {
                            vec![HarnessEvent::lane_scoped(
                                &events_lane,
                                false,
                                HarnessEventPayload::QueueUpdate {
                                    queues: events_queues.clone(),
                                },
                            )
                            .unwrap_or_else(|error| unreachable!("queue_update is lane-scoped: {error}"))]
                        })),
                    })
                })
            },
            context,
        )
        .await
    }

    /// Cancels one queued item, upstream's `cancelQueued`.
    ///
    /// # Errors
    /// The taxonomy errors ride the result; the sealed error carries the
    /// closed/fault surface.
    pub async fn cancel_queued_impl(
        &self,
        entry_id: &str,
        context: &Context,
    ) -> Result<crate::harness::agent_harness::CancelQueuedKind, LaneError> {
        if self.inner.is_harness_closed() {
            return Err(HarnessError::Closed {
                message: self.sealed_message(),
            });
        }
        self.inner.assert_open()?;
        let lane_name = self.inner.name.clone();
        let entry_id_owned = entry_id.to_owned();
        self.command::<crate::harness::agent_harness::CancelQueuedKind, _>(
            move |state, session, command_context| {
                let lane_name = lane_name.clone();
                let entry_id = entry_id_owned.clone();
                Box::pin(async move {
                    let queued = state.inbox.iter().find(|item| item.entry_id == entry_id).cloned();
                    let Some(queued) = queued else {
                        let entries = session
                            .get_entries(vec![entry_id.clone()], &command_context)
                            .await
                            .map_err(|error| lane_error(error))?;
                        let consumed = entries.contains_key(&entry_id);
                        return Ok(LaneCommand::Return {
                            result: if consumed {
                                crate::harness::agent_harness::CancelQueuedKind::AlreadyConsumed
                            } else {
                                crate::harness::agent_harness::CancelQueuedKind::NotFound
                            },
                        });
                    };
                    let stored = session
                        .get_value(&pending_entry(&entry_id).address, &command_context)
                        .await
                        .map_err(|error| lane_error(error))?;
                    if stored.is_none() {
                        return Err(Arc::new(SessionInvariantError(format!(
                            "Queued {} entry {} is missing its payload",
                            inbox_kind_name(queued.kind),
                            entry_id
                        ))));
                    }
                    let inbox: Vec<InboxItem> = state
                        .inbox
                        .iter()
                        .filter(|item| item.entry_id != entry_id)
                        .cloned()
                        .collect();
                    let queues = read_lane_queues(session.as_ref(), &inbox, &command_context)
                        .await
                        .map_err(|error| lane_error(error))?;
                    let mut writes: Vec<Write> = Vec::new();
                    writes.push(delete_value(&pending_entry(&entry_id)));
                    writes.push(
                        set_value(
                            &lane_state(&lane_name),
                            durable_lane_state(
                                &state,
                                state.operation.as_ref().map(|operation| operation.meta.operation_id.as_str()),
                                &inbox,
                                state.last_operation_id.as_deref(),
                            ),
                        )
                        .map_err(|error| lane_error(error))?,
                    );
                    let mut next = state.clone();
                    next.inbox = inbox;
                    let events_lane = lane_name.clone();
                    let events_queues = queues.clone();
                    Ok(LaneCommand::Commit {
                        writes,
                        next,
                        materialize: Arc::new(|_commit: &CommitResult| {
                            crate::harness::agent_harness::CancelQueuedKind::Cancelled
                        }),
                        events: Some(Arc::new(move |_commit: &CommitResult| {
                            vec![HarnessEvent::lane_scoped(
                                &events_lane,
                                false,
                                HarnessEventPayload::QueueUpdate {
                                    queues: events_queues.clone(),
                                },
                            )
                            .unwrap_or_else(|error| unreachable!("queue_update is lane-scoped: {error}"))]
                        })),
                    })
                })
            },
            context,
        )
        .await
    }

    /// Records one usage row, upstream's `recordUsage`.
    ///
    /// # Errors
    /// The taxonomy errors ride the result; the sealed error carries the
    /// closed/fault surface.
    pub async fn record_usage_impl(
        &self,
        usage: Usage,
        options: Option<RecordUsageOptions>,
        context: &Context,
    ) -> Result<String, LaneError> {
        if self.inner.is_harness_closed() {
            return Err(HarnessError::Closed {
                message: self.sealed_message(),
            });
        }
        self.inner.assert_open()?;
        let lane_name = self.inner.name.clone();
        let options = options.unwrap_or_default();
        self.command::<String, _>(
            move |state, _session, _command_context| {
                let lane_name = lane_name.clone();
                let usage = usage.clone();
                let entry_id = options.entry_id.clone();
                let details = options.details.clone();
                Box::pin(async move {
                    let usage_id = _session_id_of(&_session).next(None);
                    let row = UsageWriteRow {
                        id: usage_id.clone(),
                        usage,
                        entry_id,
                        adjustment: true,
                        details,
                    };
                    let events_lane = lane_name.clone();
                    let events_row = row.clone();
                    Ok(LaneCommand::Commit {
                        writes: vec![Write::Usage(UsageWrite {
                            kind: "usage".to_owned(),
                            row: row.clone(),
                        })],
                        next: state,
                        materialize: Arc::new(move |_commit: &CommitResult| usage_id.clone()),
                        events: Some(Arc::new(move |commit: &CommitResult| {
                            let seq = commit.seqs.first().copied().unwrap_or_else(|| {
                                unreachable!("usage commit carries one sequence")
                            });
                            vec![HarnessEvent::global(HarnessEventPayload::Usage {
                                lane: events_row_id_lane(&events_lane),
                                row: UsageRow {
                                    seq,
                                    id: events_row.id.clone(),
                                    usage: events_row.usage.clone(),
                                    entry_id: events_row.entry_id.clone(),
                                    adjustment: events_row.adjustment,
                                    details: events_row.details.clone(),
                                },
                                totals: commit.stats.usage.clone(),
                            })
                            .unwrap_or_else(|error| unreachable!("usage is global: {error}"))]
                        })),
                    })
                })
            },
            context,
        )
        .await
    }

    /// Waits for the lane to idle, upstream's `waitForIdle`.
    pub async fn wait_for_idle_impl(&self, context: &Context) -> Result<(), LaneError> {
        loop {
            let observation = self
                .command::<IdleObservation, _>(
                    {
                        let inner = Arc::clone(&self.inner);
                        move |state, _session, _context| {
                            let inner = Arc::clone(&inner);
                            Box::pin(async move {
                                let drive = {
                                    let core = lock_core(&inner);
                                    core.active_drive.clone()
                                };
                                if state.operation.is_none() && drive.is_none() {
                                    Ok(LaneCommand::Return {
                                        result: IdleObservation::Idle,
                                    })
                                } else {
                                    let change = inner.state_change_receiver();
                                    Ok(LaneCommand::Return {
                                        result: IdleObservation::Wait { drive, change },
                                    })
                                }
                            })
                        }
                    },
                    context,
                )
                .await?;
            match observation {
                IdleObservation::Idle => return Ok(()),
                IdleObservation::Wait { drive, change } => {
                    let wait: BoxedFuture<'static, ()> = match drive {
                        Some(drive) => Box::pin(async move {
                            let _ = drive.completion().await;
                        }),
                        None => Box::pin(wait_state_change(change)),
                    };
                    if await_with_context(wait, context).await.is_err() {
                        return Err(abort_reason_error(None));
                    }
                }
            }
        }
    }

    /// Runs one callback when idle, upstream's `runWhenIdle`.
    pub async fn run_when_idle_impl(
        &self,
        callback: IdleCallback,
        context: &Context,
    ) -> Result<(), LaneError> {
        let owner = CancellationToken::new();
        loop {
            let observation = self
                .command::<IdleClaimObservation, _>(
                    {
                        let inner = Arc::clone(&self.inner);
                        let owner = owner.clone();
                        move |state, _session, _context| {
                            let inner = Arc::clone(&inner);
                            let owner = owner.clone();
                            Box::pin(async move {
                                let drive = {
                                    let core = lock_core(&inner);
                                    core.active_drive.clone()
                                };
                                if state.operation.is_some() || drive.is_some() {
                                    let change = inner.state_change_receiver();
                                    return Ok(LaneCommand::Return {
                                        result: IdleClaimObservation::Wait { drive, change },
                                    });
                                }
                                {
                                    let mut core = lock_core(&inner);
                                    core.idle_owner = Some(owner.clone());
                                    let _ = core.state_change.send_modify(|version| *version += 1);
                                }
                                Ok(LaneCommand::Return {
                                    result: IdleClaimObservation::Claimed,
                                })
                            })
                        }
                    },
                    context,
                )
                .await?;
            match observation {
                IdleClaimObservation::Claimed => break,
                IdleClaimObservation::Wait { drive, change } => {
                    let wait: BoxedFuture<'static, ()> = match drive {
                        Some(drive) => Box::pin(async move {
                            let _ = drive.completion().await;
                        }),
                        None => Box::pin(wait_state_change(change)),
                    };
                    if await_with_context(wait, context).await.is_err() {
                        return Err(abort_reason_error(None));
                    }
                }
            }
        }
        let result = async {
            self.inner.assert_open()?;
            (callback)(context).await;
            Ok(())
        }
        .await;
        // The claim is exclusive: no other claimant can take the owner slot
        // while it is held, so the release clears unconditionally.
        {
            let mut core = lock_core(&self.inner);
            core.idle_owner = None;
            let _ = core.state_change.send_modify(|version| *version += 1);
        }
        owner.cancel();
        result
    }

    /// The configured model, upstream's `getModel`.
    pub async fn get_model_impl(
        &self,
        _context: &Context,
    ) -> Result<Option<Model>, LaneError> {
        self.inner.assert_open()?;
        let configuration = lock_core(&self.inner).state.configuration.clone();
        Ok(self
            .inner
            .models
            .model(&configuration.model.provider, &configuration.model.model_id))
    }

    /// Sets the model, upstream's `setModel`.
    pub async fn set_model_impl(
        &self,
        model: crate::harness::session::types::ModelIdentity,
        context: &Context,
    ) -> Result<(), LaneError> {
        self.set_configuration(
            {
                let model = model.clone();
                move |configuration: &crate::harness::session::types::LaneConfiguration| {
                    let mut next = configuration.clone();
                    next.model = model.clone();
                    next
                }
            },
            |previous: &crate::harness::session::types::LaneConfiguration,
             value: &crate::harness::session::types::LaneConfiguration| {
                LaneConfigUpdate::Model {
                    previous: Some(previous.model.clone()),
                    value: value.model.clone(),
                }
            },
            context,
        )
        .await
    }

    /// The thinking level, upstream's `getThinkingLevel`.
    pub async fn get_thinking_level_impl(
        &self,
        _context: &Context,
    ) -> Result<crate::types::ThinkingLevel, LaneError> {
        self.inner.assert_open()?;
        Ok(lock_core(&self.inner).state.configuration.thinking_level)
    }

    /// Sets the thinking level, upstream's `setThinkingLevel`.
    pub async fn set_thinking_level_impl(
        &self,
        level: crate::types::ThinkingLevel,
        context: &Context,
    ) -> Result<(), LaneError> {
        self.set_configuration(
            move |configuration: &crate::harness::session::types::LaneConfiguration| {
                let mut next = configuration.clone();
                next.thinking_level = level;
                next
            },
            |previous: &crate::harness::session::types::LaneConfiguration,
             value: &crate::harness::session::types::LaneConfiguration| {
                LaneConfigUpdate::ThinkingLevel {
                    previous: previous.thinking_level,
                    value: value.thinking_level,
                }
            },
            context,
        )
        .await
    }

    /// The active tools, upstream's `getActiveTools`.
    pub async fn get_active_tools_impl(&self, _context: &Context) -> Result<Vec<String>, LaneError> {
        self.inner.assert_open()?;
        Ok(lock_core(&self.inner).state.configuration.active_tool_names.clone())
    }

    /// Sets the active tools, upstream's `setActiveTools`.
    pub async fn set_active_tools_impl(
        &self,
        names: Vec<String>,
        context: &Context,
    ) -> Result<(), LaneError> {
        self.set_configuration(
            move |configuration: &crate::harness::session::types::LaneConfiguration| {
                let mut next = configuration.clone();
                next.active_tool_names = names.clone();
                next
            },
            |previous: &crate::harness::session::types::LaneConfiguration,
             value: &crate::harness::session::types::LaneConfiguration| {
                LaneConfigUpdate::ActiveTools {
                    previous: previous.active_tool_names.clone(),
                    value: value.active_tool_names.clone(),
                }
            },
            context,
        )
        .await
    }

    /// Watches the lane, upstream's `watch`.
    ///
    /// # Errors
    /// The sealed error when the lane sealed with a fault; a failed initial
    /// capture unsubscribes the installed watcher.
    pub async fn watch_impl(
        &self,
        context: &Context,
    ) -> Result<Box<dyn WatchHandle<LaneSnapshot>>, LaneError> {
        let lane_for_capture = self.clone();
        let watcher = self
            .read_lane::<Box<dyn WatchHandle<LaneSnapshot>>, _>(
                move |state, reader, watch_context| {
                    let lane = lane_for_capture.clone();
                    Box::pin(async move {
                        let lane_name = lane.name().to_owned();
                        let filter: BusWatchFilter = Arc::new(move |event: &HarnessEvent| {
                            event.event_type() == crate::harness::agent_harness::HarnessEventType::Usage
                                || event.lane.is_none()
                                || event.lane.as_deref() == Some(lane_name.as_str())
                        });
                        let resnapshot: ResnapshotCapture<LaneSnapshot> = {
                            let lane = lane.clone();
                            Arc::new(move |resnapshot_context: &Context, boundary: &HarnessResnapshotBoundary| {
                                let lane = lane.clone();
                                let resnapshot_context = resnapshot_context.clone();
                                Box::pin(async move {
                                    let snapshot = lane
                                        .capture_lane_snapshot_in_read(&resnapshot_context)
                                        .await
                                        .map_err(|error| -> ListenerError { lane_error(SessionError(error.to_string())) })?;
                                    boundary.mark();
                                    Ok(snapshot)
                                })
                                    as BoxedFuture<'static, Result<LaneSnapshot, ListenerError>>
                            })
                        };
                        let watcher = (lane.inner.install_watch)(filter, &watch_context, resnapshot);
                        let snapshot = lane
                            .capture_lane_snapshot(&state, reader.as_ref(), &watch_context)
                            .await;
                        match snapshot {
                            Ok(snapshot) => {
                                watcher.set_snapshot(snapshot);
                                Ok(watcher)
                            }
                            Err(error) => {
                                watcher.unsubscribe();
                                Err(error)
                            }
                        }
                    })
                },
                context,
            )
            .await?;
        Ok(watcher)
    }

    /// Captures the lane snapshot inside one read, upstream's
    /// `captureLaneSnapshot` under `watch`'s resnapshot closure.
    async fn capture_lane_snapshot_in_read(&self, context: &Context) -> Result<LaneSnapshot, LaneError> {
        self.read_lane::<LaneSnapshot, _>(
            {
                let lane = self.clone();
                move |state, reader, read_context| {
                    let lane = Arc::new(lane);
                    Box::pin(async move {
                        lane.capture_lane_snapshot(&state, reader.as_ref(), &read_context)
                            .await
                    })
                }
            },
            context,
        )
        .await
    }

    /// Captures one lane snapshot, upstream's `captureLaneSnapshot`.
    async fn capture_lane_snapshot(
        &self,
        state: &LaneState,
        reader: &dyn SessionReader,
        context: &Context,
    ) -> Result<LaneSnapshot, LaneError> {
        let captured = state.clone();
        let transcript = match &captured.tip_id {
            None => Vec::new(),
            Some(tip_id) => {
                let entries = reader
                    .scan_branch(
                        &crate::harness::session::types::StorageBranchScan {
                            start: tip_id.clone(),
                            stop_at_type: Some(EntryType::Compaction),
                            order: Some(BranchScanOrder::NewestFirst),
                            ..Default::default()
                        },
                        context,
                    )
                    .await
                    .map_err(|error| lane_error(error))?;
                let mut entries = entries;
                entries.reverse();
                entries
            }
        };
        let queues = read_lane_queues(reader, &captured.inbox, context)
            .await
            .map_err(|error| lane_error(error))?;
        let last_result = match &captured.last_operation_id {
            None => None,
            Some(last_operation_id) => {
                let stored = reader
                    .get_value(&operation_result(last_operation_id).address, context)
                    .await
                    .map_err(|error| lane_error(error))?;
                let Some(stored) = stored else {
                    return Err(Arc::new(SessionInvariantError(format!(
                        "Lane {} is missing result {}",
                        quoted(&self.inner.name),
                        last_operation_id
                    ))));
                };
                Some(serde_json::from_value::<OperationResultRecord>(stored.value).map_err(|error| {
                    lane_error(SessionInvariantError(format!(
                        "Operation result is malformed: {error}"
                    )))
                })?)
            }
        };
        let stats = reader
            .get_stats(context)
            .await
            .map_err(|error| lane_error(error))?;
        let operation = &captured.operation;

        let mut operation_snapshot: Option<LiveOperationView> = None;
        if let Some(operation) = operation {
            let mut running_tools: Vec<LaneSnapshotTool> = Vec::new();
            let mut streaming_message: Option<pi_ai::types::AssistantMessage> = None;
            let mut retry: Option<RetryView> = None;
            let mut deferred: Option<DeferredView> = None;
            let operation_id = operation.meta.operation_id.clone();
            match &operation.state {
                OperationState::AssistantRetryWait(leaf) => {
                    retry = Some(RetryView {
                        attempt: leaf.retry_wait.next_attempt,
                        max_attempts: leaf.generation_context.retry_policy.max_attempts,
                        next_attempt_at: leaf.retry_wait.not_before,
                    });
                }
                OperationState::AssistantEffectPending(leaf) => {
                    streaming_message = self
                        .read_streaming_message(reader, &operation_id, &leaf.response_entry_id, context)
                        .await?;
                }
                OperationState::DeferredSuspended(leaf) | OperationState::DeferredEffectPending(leaf) => {
                    let source_entry_id = leaf.deferred.source_entry_id.clone();
                    let poll = leaf.deferred.poll;
                    let entries = reader
                        .get_entries(vec![source_entry_id.clone()], context)
                        .await
                        .map_err(|error| lane_error(error))?;
                    let source = entries.get(&source_entry_id);
                    let valid = match source {
                        Some(Entry::Message { body, .. }) => match &body.message {
                            AgentMessage::Standard(Message::Assistant(assistant)) => {
                                assistant.deferred.clone()
                            }
                            _ => None,
                        },
                        _ => None,
                    };
                    let Some(handle) = valid else {
                        return Err(lane_error(SessionInvariantError(
                            "Deferred source is missing its assistant handle".to_owned(),
                        )));
                    };
                    deferred = Some(DeferredView { handle, poll });
                    if let OperationState::DeferredEffectPending(effect_leaf) = &operation.state {
                        streaming_message = self
                            .read_streaming_message(reader, &operation_id, &effect_leaf.response_entry_id, context)
                            .await?;
                    }
                }
                OperationState::Tools(leaf) => {
                    let entries = reader
                        .get_entries(vec![leaf.batch.assistant_entry_id.clone()], context)
                        .await
                        .map_err(|error| lane_error(error))?;
                    let assistant = entries.get(&leaf.batch.assistant_entry_id);
                    let assistant_message = match assistant {
                        Some(Entry::Message { body, .. }) => match &body.message {
                            AgentMessage::Standard(Message::Assistant(assistant)) => Some(assistant),
                            _ => None,
                        },
                        _ => None,
                    };
                    let Some(assistant_message) = assistant_message else {
                        return Err(lane_error(SessionInvariantError(
                            "Tool batch assistant entry is invalid".to_owned(),
                        )));
                    };
                    for call in &leaf.batch.calls {
                        if matches!(
                            call.status(),
                            crate::harness::session::types::ToolCallStatus::Planned
                                | crate::harness::session::types::ToolCallStatus::Completed { .. }
                        ) {
                            continue;
                        }
                        let block = assistant_message
                            .content
                            .get(usize::try_from(call.source_index).unwrap_or(usize::MAX));
                        let block = match block {
                            Some(pi_ai::types::AssistantBlock::ToolCall(block)) => block,
                            _ => {
                                return Err(lane_error(SessionInvariantError(format!(
                                    "Tool call source index {} does not name a tool-call block",
                                    call.source_index
                                ))));
                            }
                        };
                        let args = reader
                            .get_value(
                                &operation_tool_args(&operation_id, &leaf.batch.turn_id, call.source_index).address,
                                context,
                            )
                            .await
                            .map_err(|error| lane_error(error))?;
                        match call.status() {
                            crate::harness::session::types::ToolCallStatus::EffectPending { .. } => {
                                let Some(args) = args else {
                                    return Err(lane_error(SessionInvariantError(format!(
                                        "Tool call {} is missing persisted arguments",
                                        block.id
                                    ))));
                                };
                                let args: serde_json::Map<String, serde_json::Value> =
                                    serde_json::from_value(args.value).map_err(|error| {
                                        lane_error(SessionInvariantError(format!(
                                            "Tool arguments are malformed: {error}"
                                        )))
                                    })?;
                                let checkpoint = reader
                                    .get_value(
                                        &pending_tool_output(&operation_id, &call.result_entry_id).address,
                                        context,
                                    )
                                    .await
                                    .map_err(|error| lane_error(error))?;
                                let checkpoint: Option<crate::harness::session::values::ToolOutputPayload> =
                                    match checkpoint {
                                        Some(StoredValue { value, .. }) => Some(serde_json::from_value(value).map_err(|error| {
                                            lane_error(SessionInvariantError(format!(
                                                "Pending tool output is malformed: {error}"
                                            )))
                                        })?),
                                        None => None,
                                    };
                                running_tools.push(LaneSnapshotTool::Running {
                                    tool_call_id: block.id.clone(),
                                    tool_name: block.name.clone(),
                                    args: serde_json::Value::Object(args),
                                    result: checkpoint.map(tool_output_payload_to_result),
                                });
                            }
                            _ => {
                                let staged = reader
                                    .get_value(&pending_entry(&call.result_entry_id).address, context)
                                    .await
                                    .map_err(|error| lane_error(error))?;
                                let staged = match staged {
                                    Some(StoredValue { value, .. }) => {
                                        serde_json::from_value::<PendingEntry>(value).map_err(|error| {
                                            lane_error(SessionInvariantError(format!(
                                                "Pending entry payload is malformed: {error}"
                                            )))
                                        })?
                                    }
                                    None => {
                                        return Err(lane_error(SessionInvariantError(format!(
                                            "Tool call {} is missing its staged result",
                                            call.result_entry_id
                                        ))));
                                    }
                                };
                                let PendingEntry::Message { payload } = &staged else {
                                    return Err(lane_error(SessionInvariantError(format!(
                                        "Tool call {} is missing its staged result",
                                        call.result_entry_id
                                    ))));
                                };
                                let AgentMessage::Standard(Message::ToolResult(tool_result)) = payload.as_ref()
                                else {
                                    return Err(lane_error(SessionInvariantError(format!(
                                        "Tool call {} is missing its staged result",
                                        call.result_entry_id
                                    ))));
                                };
                                if tool_result.tool_call_id != block.id || tool_result.tool_name != block.name {
                                    return Err(lane_error(SessionInvariantError(format!(
                                        "Tool call {} has a mismatched staged result",
                                        call.result_entry_id
                                    ))));
                                }
                                running_tools.push(LaneSnapshotTool::Settled {
                                    tool_call_id: block.id.clone(),
                                    tool_name: block.name.clone(),
                                    args: match &args {
                                        Some(StoredValue { value, .. }) => value.clone(),
                                        None => serde_json::Value::Object(block.arguments.clone()),
                                    },
                                    result: tool_result_from_message(
                                        tool_result,
                                        call.terminate().unwrap_or(false),
                                    ),
                                    is_error: tool_result.is_error,
                                });
                            }
                        }
                    }
                }
                OperationState::SummaryRetryWait(leaf) => {
                    retry = Some(RetryView {
                        attempt: leaf.retry_wait.next_attempt,
                        max_attempts: leaf.generation.retry_policy.max_attempts,
                        next_attempt_at: leaf.retry_wait.not_before,
                    });
                }
                _ => {}
            }

            operation_snapshot = Some(LiveOperationView {
                id: operation.meta.operation_id.clone(),
                kind: intent_kind_of(&operation.meta),
                started_at: operation.meta.started_at,
                from_tip_id: operation.meta.source_tip_id.clone(),
                status: match crate::harness::session::types::operation_scope_of(&operation.state).control {
                    Control::CancelRequested { .. } => OperationStatus::Aborting,
                    Control::Running => OperationStatus::Open,
                },
                retry,
                deferred,
                streaming_message,
                running_tools,
            });
        }

        let faulted = lock_core(&self.inner)
            .closed_error
            .as_ref()
            .is_some_and(|error| error.downcast_ref::<crate::harness::result::HarnessFault>().is_some());

        Ok(LaneSnapshot {
            lane: self.inner.name.clone(),
            transcript,
            tip_id: captured.tip_id.clone(),
            last_result,
            configuration: captured.configuration.clone(),
            stats,
            operation: operation_snapshot,
            queues,
            faulted,
        })
    }

    async fn read_streaming_message(
        &self,
        reader: &dyn SessionReader,
        operation_id: &str,
        response_entry_id: &str,
        context: &Context,
    ) -> Result<Option<pi_ai::types::AssistantMessage>, LaneError> {
        let frames = read_assistant_frames(reader, operation_id, response_entry_id, context)
            .await
            .map_err(|error| lane_error(error))?;
        let reduced = reduce_assistant_message_frames(frames)
            .map_err(|error| -> LaneError { Arc::from(lane_error(SessionError(error))) })?;
        Ok(reduced)
    }

    async fn set_configuration<F, E>(&self, update: F, event: E, context: &Context) -> Result<(), LaneError>
    where
        F: Fn(&crate::harness::session::types::LaneConfiguration) -> crate::harness::session::types::LaneConfiguration
            + Send
            + Sync
            + 'static,
        E: Fn(
                &crate::harness::session::types::LaneConfiguration,
                &crate::harness::session::types::LaneConfiguration,
            ) -> LaneConfigUpdate
            + Send
            + Sync
            + 'static,
    {
        let lane_name = self.inner.name.clone();
        self.command::<(), _>(
            move |state, _session, _command_context| {
                let lane_name = lane_name.clone();
                Box::pin(async move {
                    let configuration = (update)(&state.configuration);
                    let event_payload = (event)(&state.configuration, &configuration);
                    let writes = vec![set_value(&lane_config(&lane_name), configuration.clone())
                        .map_err(|error| lane_error(error))?];
                    let mut next = state.clone();
                    next.configuration = configuration.clone();
                    let events_lane = lane_name.clone();
                    Ok(LaneCommand::Commit {
                        writes,
                        next,
                        materialize: Arc::new(|_commit: &CommitResult| ()),
                        events: Some(Arc::new(move |_commit: &CommitResult| {
                            vec![HarnessEvent::lane_scoped(
                                &events_lane,
                                false,
                                HarnessEventPayload::ConfigUpdate {
                                    property: crate::harness::agent_harness::ConfigUpdateKind::Lane(
                                        event_payload.clone(),
                                    ),
                                },
                            )
                            .unwrap_or_else(|error| unreachable!("config_update is lane-scoped: {error}"))]
                        })),
                    })
                })
            },
            context,
        )
        .await
    }

    /// Scans the branch path, upstream's `findEntries`.
    pub async fn find_entries_impl(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> Result<Vec<Entry>, LaneError> {
        self.inner.assert_open()?;
        let query = query.cloned().unwrap_or_default();
        let start = query.start.clone().or_else(|| lock_core(&self.inner).state.tip_id.clone());
        let Some(start) = start else {
            return Ok(Vec::new());
        };
        self.inner
            .session
            .scan_branch(
                &crate::harness::session::types::StorageBranchScan {
                    start,
                    stop_at_type: query.stop_at_type,
                    stop_at_id: query.stop_at_id,
                    kind: query.kind,
                    custom_type: query.custom_type,
                    order: Some(query.order.unwrap_or(BranchScanOrder::NewestFirst)),
                    limit: query.limit,
                    cursor: query.cursor,
                },
                context,
            )
            .await
            .map_err(|error| lane_error(error))
    }

    /// First match on the branch path, upstream's `findEntry`.
    pub async fn find_entry_impl(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> Result<Option<Entry>, LaneError> {
        let mut query = query.cloned().unwrap_or_default();
        query.limit = Some(match query.limit {
            Some(limit) => limit.min(1),
            None => 1,
        });
        Ok(self.find_entries_impl(Some(&query), context).await?.into_iter().next())
    }

    /// Appends one message at the tip, upstream's `appendMessage`.
    pub async fn append_message_impl(
        &self,
        message: AgentMessage,
        context: &Context,
    ) -> Result<String, LaneError> {
        self.append(PendingEntry::Message {
            payload: Box::new(message),
        }, context)
        .await
    }

    /// Appends one custom entry at the tip, upstream's `appendCustomEntry`.
    pub async fn append_custom_entry_impl(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
        context: &Context,
    ) -> Result<String, LaneError> {
        self.append(
            PendingEntry::Custom {
                custom_type: custom_type.to_owned(),
                payload: data,
            },
            context,
        )
        .await
    }

    async fn append(&self, pending: PendingEntry, context: &Context) -> Result<String, LaneError> {
        self.inner.assert_open()?;
        if let PendingEntry::Message { payload } = &pending {
            if let AgentMessage::Standard(Message::Assistant(assistant)) = payload.as_ref() {
                if assistant.stop_reason == StopReason::Pending {
                    return Err(lane_error(
                        crate::harness::session::types::SessionPendingAssistantMessageError,
                    ));
                }
            }
        }
        let id = self.inner.session.id_generator().next(None);
        let lane_name = self.inner.name.clone();
        let id_for_plan = id.clone();
        self.command::<String, _>(
            move |state, session, command_context| {
                let id = id_for_plan.clone();
                let lane_name = lane_name.clone();
                let pending = pending.clone();
                Box::pin(async move {
                    let operation = state.operation.clone();
                    let Some(operation) = operation else {
                        let queued = inbox_items(&state.inbox, InboxItemKind::Write);
                        let mut captured: Vec<NewEntry> = Vec::new();
                        for item in &queued {
                            let stored = session
                                .get_value(&pending_entry(&item.entry_id).address, &command_context)
                                .await
                                .map_err(|error| lane_error(error))?;
                            let Some(stored) = stored else {
                                return Err(Arc::new(SessionInvariantError(format!(
                                    "Pending write {} is missing its payload",
                                    item.entry_id
                                ))));
                            };
                            let pending_stored: PendingEntry = serde_json::from_value(stored.value)
                                .map_err(|error| {
                                    lane_error(SessionInvariantError(format!(
                                        "Pending entry payload is malformed: {error}"
                                    )))
                                })?;
                            captured.push(pending_entry_write(&item.entry_id, &pending_stored));
                        }
                        let inbox = without_inbox_items(&state.inbox, &queued);
                        let queues = if queued.is_empty() {
                            None
                        } else {
                            Some(read_lane_queues(session.as_ref(), &inbox, &command_context).await)
                        };
                        let entries = chain_entries(
                            state.tip_id.clone(),
                            {
                                let mut chained = captured;
                                chained.push(pending_entry_write(&id, &pending));
                                chained
                            },
                        );
                        let mut writes: Vec<Write> = Vec::new();
                        for entry in &entries {
                            writes.push(Write::Entry(Box::new(EntryWrite {
                                kind: "entry".to_owned(),
                                entry: entry.clone(),
                            })));
                        }
                        for item in &queued {
                            writes.push(delete_value(&pending_entry(&item.entry_id)));
                        }
                        writes.push(
                            set_value(&branch_tip(&lane_name), Some(id.clone()))
                                .map_err(|error| lane_error(error))?,
                        );
                        writes.push(
                            set_value(
                                &lane_state(&lane_name),
                                durable_lane_state(&state, None, &inbox, state.last_operation_id.as_deref()),
                            )
                            .map_err(|error| lane_error(error))?,
                        );
                        let mut next = state.clone();
                        next.tip_id = Some(id.clone());
                        next.inbox = inbox;
                        let events_lane = lane_name.clone();
                        let events_entries = entries.clone();
                        let events_queues = queues.clone();
                        return Ok(LaneCommand::Commit {
                            writes,
                            next,
                            materialize: Arc::new(move |_commit: &CommitResult| id.clone()),
                            events: Some(Arc::new(move |commit: &CommitResult| {
                                let mut events =
                                    committed_entry_events(&events_entries, commit, &events_lane, None, 0);
                                if let Some(Ok(queues)) = &events_queues {
                                    events.push(
                                        HarnessEvent::lane_scoped(
                                            &events_lane,
                                            false,
                                            HarnessEventPayload::QueueUpdate {
                                                queues: queues.clone(),
                                            },
                                        )
                                        .unwrap_or_else(|error| {
                                            unreachable!("queue_update is lane-scoped: {error}")
                                        }),
                                    );
                                }
                                events
                            })),
                        });
                    }

                    let mut inbox = state.inbox.clone();
                    inbox.push(InboxItem {
                        entry_id: id.clone(),
                        kind: InboxItemKind::Write,
                    });
                    let mut queues = read_lane_queues(session.as_ref(), &state.inbox, &command_context)
                        .await
                        .map_err(|error| lane_error(error))?;
                    queues.push(match &pending {
                        PendingEntry::Message { payload } => LaneQueuedItem::Message {
                            entry_id: id.clone(),
                            kind: InboxItemKind::Write,
                            message: payload.clone(),
                        },
                        PendingEntry::Custom { custom_type, payload } => LaneQueuedItem::Custom {
                            entry_id: id.clone(),
                            kind: InboxItemKind::Write,
                            custom_type: custom_type.clone(),
                            data: payload.clone(),
                        },
                    });
                    let mut writes: Vec<Write> = Vec::new();
                    writes.push(
                        set_value(&pending_entry(&id), pending.clone())
                            .map_err(|error| lane_error(error))?,
                    );
                    writes.push(
                        set_value(
                            &lane_state(&lane_name),
                            durable_lane_state(
                                &state,
                                Some(&operation.meta.operation_id),
                                &inbox,
                                state.last_operation_id.as_deref(),
                            ),
                        )
                        .map_err(|error| lane_error(error))?,
                    );
                    let mut next = state.clone();
                    next.inbox = inbox;
                    let events_lane = lane_name.clone();
                    let events_queues = queues.clone();
                    Ok(LaneCommand::Commit {
                        writes,
                        next,
                        materialize: Arc::new(move |_commit: &CommitResult| id.clone()),
                        events: Some(Arc::new(move |_commit: &CommitResult| {
                            vec![HarnessEvent::lane_scoped(
                                &events_lane,
                                false,
                                HarnessEventPayload::QueueUpdate {
                                    queues: events_queues.clone(),
                                },
                            )
                            .unwrap_or_else(|error| unreachable!("queue_update is lane-scoped: {error}"))]
                        })),
                    })
                })
            },
            context,
        )
        .await
    }

    /// Seals the lane with its close or fault error, upstream's `seal`.
    /// Work admitted before the seal finishes; later work rejects with the
    /// sealed error. The returned future resolves when any in-flight idle
    /// callback settles.
    pub async fn seal(&self, error: LaneError) {
        let owner = {
            let mut core = lock_core(&self.inner);
            if core.closed_error.is_none() {
                core.closed_error = Some(Arc::clone(&error));
            }
            core.idle_owner.clone()
        };
        if let Some(drive) = self.active_drive() {
            drive.close_gate(Arc::clone(&error));
        }
        self.inner.signal_state_change();
        if let Some(owner) = owner {
            owner.cancelled().await;
        }
    }
}

enum IdleObservation {
    Idle,
    Wait {
        drive: Option<Arc<Drive>>,
        change: tokio::sync::watch::Receiver<u64>,
    },
}

enum IdleClaimObservation {
    Claimed,
    Wait {
        drive: Option<Arc<Drive>>,
        change: tokio::sync::watch::Receiver<u64>,
    },
}

fn mismatch_for(
    lane: &str,
    expected: &str,
    current_operation_id: Option<String>,
    last_operation_id: Option<String>,
) -> HarnessError {
    HarnessError::OperationMismatch {
        lane: lane.to_owned(),
        expected_operation_id: expected.to_owned(),
        current_operation_id,
        last_operation_id,
        message: format!(
            "Operation {expected} does not own lane {}",
            quoted(lane)
        ),
    }
}

fn admission_kind_name(kind: OperationKind) -> &'static str {
    match kind {
        OperationKind::Run => "run",
        OperationKind::Compaction => "compaction",
        OperationKind::Navigation => "navigation",
    }
}

fn wait_reason_name(reason: &crate::harness::agent_harness::DriveWaitReason) -> &'static str {
    match reason {
        crate::harness::agent_harness::DriveWaitReason::Retry { .. } => "retry",
        crate::harness::agent_harness::DriveWaitReason::Deferred { .. } => "deferred",
    }
}

fn tool_output_payload_to_result(payload: crate::harness::session::values::ToolOutputPayload) -> AgentToolResult {
    AgentToolResult {
        content: payload.content,
        details: payload.details,
        usage: payload.usage,
        added_tool_names: payload.added_tool_names,
        terminate: payload.terminate,
    }
}

fn tool_result_from_message(
    message: &pi_ai::types::ToolResultMessage,
    terminate: bool,
) -> AgentToolResult {
    AgentToolResult {
        content: message.content.clone(),
        details: message.details.clone().unwrap_or_default(),
        usage: message.usage.clone(),
        added_tool_names: message.added_tool_names.clone(),
        terminate: if terminate { Some(true) } else { None },
    }
}

fn events_row_id_lane(lane: &str) -> String {
    lane.to_owned()
}

fn _session_id_of(session: &Arc<dyn Session>) -> &dyn crate::harness::session::types::IdGenerator {
    session.id_generator()
}

/// The run request a stalled suspension rides — re-exported for the
/// drive child's use of the suspended-run surface.
pub type LaneSuspension = SuspensionFollowUp;

impl AgentLane for Lane {
    fn name(&self) -> &str {
        Lane::name(self)
    }

    fn get_tip_id(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, LaneOperationError>> {
        let result = self.read_lane::<Option<String>, _>(
            {
                let lane = self.clone();
                move |state, _session, _context| {
                    let _ = &lane;
                    Box::pin(async move { Ok(state.tip_id.clone()) })
                }
            },
            context,
        );
        Box::pin(async move {
            result
                .await
                .map_err(|error| LaneOperationError::Closed(HarnessError::Closed {
                    message: error.to_string(),
                }))
        })
    }
}
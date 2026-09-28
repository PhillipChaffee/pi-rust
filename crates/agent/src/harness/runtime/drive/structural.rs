//! The structural summary and navigation procedures, ported from upstream
//! `src/harness/runtime/drive/structural.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The five summary and navigation leaves advance one structural task's
//! state machine: `run_structural_decision` consumes one durable
//! preparation and decision hook, `run_structural_generation` executes
//! one ready attempt, `recover_structural_generation` converts an
//! orphaned attempt into a fresh numbered attempt or terminal failure,
//! `run_structural_retry_wait` consumes one retry wait without starting
//! a provider effect, and `commit_navigation` atomically moves an
//! unsummarized navigation and finishes its operation. The compaction
//! preparations (`prepare_compaction_threshold`,
//! `prepare_overflow_compaction`) create the summary tasks the run and
//! response procedures commit through, and the two serializers
//! (`durable_compaction_preparation`, `durable_branch_preparation`)
//! are the accept paths' write payloads.
//!
//! Porting restatements: upstream's `StructuralCancelled` throw restates
//! as the `RequestSignal` cell — the [`SummaryRequest`] boundary's
//! infallible signature cannot carry a thrown error, so the request
//! closure records the signal and returns a synthetic aborted settlement
//! the attempt's result mapping discards. Upstream's per-attempt
//! `requestIndex`/`lastResponse` locals restate as shared cells for the
//! same reason.
#![expect(
    dead_code,
    reason = "the two serializers serve the lane accept seams and the two preparation constructors serve the checkpoint and response children; those waves land later and retire this expectation"
)]

use super::await_abort_cancellation;
use super::gate_rejection_error;
use super::hook_error_to_lane_error;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use pi_ai::models::WithTransforms;
use pi_ai::types::AssistantMessage;
use pi_ai::types::CacheRetention;
use pi_ai::types::Context as AiContext;
use pi_ai::types::DeferredRequest;
use pi_ai::types::Model;
use pi_ai::types::OnPayload;
use pi_ai::types::SimpleStreamOptions;
use pi_ai::types::StopReason;
use pi_ai::types::TransportOptions;
use pi_ai::types::Usage;
use pi_ai::utils::retry::RetryPolicy;
use pi_ai::utils::retry::is_retryable_assistant_error;
use pi_ai::utils::retry::retry_delay_ms;
use serde_json::Value as JsonValue;

use pi_chord::context::with_abort_signal;

use crate::harness::agent_harness::CompactionEndStatus;
use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::DriveWaitReason;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HookEvent;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::agent_harness::HookResult;
use crate::harness::agent_harness::NavigationEndStatus;
use crate::harness::agent_harness::RunEndStatus;
use crate::harness::agent_harness::StepKind;
use crate::harness::agent_harness::global_event;
use crate::harness::agent_harness::lane_scoped_event;
use crate::harness::compaction::branch_summarization::PreparedBranchSummaryOptions;
use crate::harness::compaction::branch_summarization::generate_branch_summary_with_request;
use crate::harness::compaction::compaction::CompactGenerationOptions;
use crate::harness::compaction::compaction::SummaryRequest;
use crate::harness::compaction::compaction::compact_with_request;
use crate::harness::compaction::compaction::prepare_compaction;
use crate::harness::compaction::compaction::should_compact;
use crate::harness::compaction::types::BranchPreparation;
use crate::harness::compaction::types::BranchSummaryResult;
use crate::harness::compaction::types::CompactResult;
use crate::harness::compaction::types::CompactionDetails;
use crate::harness::compaction::types::CompactionPreparation;
use crate::harness::context::Context;
use crate::harness::context::get_telemetry_context;
use crate::harness::context::request_signal;
use crate::harness::gate::GateRejection;
use crate::harness::hooks::HookRunError;
use crate::harness::hooks::apply_stream_options_patch;
use crate::harness::runtime::clock::now_ms;
use crate::harness::runtime::drive::boundary::BoundaryFinishPending;
use crate::harness::runtime::drive::boundary::assistant_ready_at_boundary;
use crate::harness::runtime::drive::boundary::boundary_placement_events;
use crate::harness::runtime::drive::boundary::finish_run_boundary;
use crate::harness::runtime::drive::boundary::normalized_retry_policy;
use crate::harness::runtime::drive::boundary::plan_boundary_inbox;
use crate::harness::runtime::drive::retry::retry_not_before;
use crate::harness::runtime::drive::retry::wait_until;
use crate::harness::runtime::drive::terminal::operation_cleanup_writes;
use crate::harness::runtime::drive::terminal::operation_result_record;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::transcript::committed_entry_events;
use crate::harness::runtime::transcript::read_bounded_entries;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::EventsFn;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::LanePatch;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::runtime::types::ProcedureResult;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::commit::insert_entry;
use crate::harness::session::commit::insert_usage;
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::BranchSummaryEntryBody;
use crate::harness::session::types::CheckpointOperation;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::CompactionEntryBody;
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::Control;
use crate::harness::session::types::DurableStructuralPreparation;
use crate::harness::session::types::Entry;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::NavigationReadyToCommitOperation;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::NormalizedRetryPolicy;
use crate::harness::session::types::OperationError;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::RetryWait;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SummaryContext;
use crate::harness::session::types::SummaryDecidingOperation;
use crate::harness::session::types::SummaryEffectPendingOperation;
use crate::harness::session::types::SummaryEffectRequest;
use crate::harness::session::types::SummaryGenerationScope;
use crate::harness::session::types::SummaryReadyOperation;
use crate::harness::session::types::SummaryRetryWaitOperation;
use crate::harness::session::types::SummaryTask;
use crate::harness::session::types::TerminalStatus;
use crate::harness::session::types::UsageRow;
use crate::harness::session::types::UsageWriteRow;
use crate::harness::session::types::operation_scope_of;
use crate::harness::session::values::Write;
use crate::harness::session::values::branch_tip;
use crate::harness::session::values::entry_label;
use crate::harness::session::values::operation_preparation;
use crate::harness::session::values::set_value_write;
use crate::harness::types::AgentHarnessStreamOptions;
use crate::harness::types::BranchSummaryErrorCode;
use crate::harness::types::CompactionErrorCode;

/// The task id and durable preparation one compaction creates, upstream's
/// `prepareCompactionThreshold`'s value shape
/// (`{ taskId, preparation: DurableStructuralPreparation }`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ThresholdPreparation {
    /// The summary task id, from the session's id generator.
    pub task_id: String,
    /// The durable preparation the task commits under.
    pub preparation: DurableStructuralPreparation,
}

/// The out-of-band terminal signal one nested request records, upstream's
/// `StructuralCancelled` throw and the rethrow of any other error: the
/// [`SummaryRequest`] boundary's Rust signature returns the settlement
/// only, so the closure records the signal and returns a synthetic aborted
/// settlement ([`aborted_settlement`]) the attempt's result mapping
/// discards.
enum RequestSignal {
    /// Cancellation won, upstream's `throw new StructuralCancelled()` —
    /// the attempt reports `cancel_requested` and never runs a terminal.
    Cancelled,
    /// Any other failure, upstream's rethrow — the attempt propagates the
    /// error and the pass faults.
    Fault(LaneError),
}

/// The shared signal cell one attempt's request closure writes.
type RequestSignalCell = Arc<Mutex<Option<RequestSignal>>>;

/// The rehydrated preparation one structural attempt runs, upstream's
/// `CompactionPreparation | BranchPreparation` union — the kind check
/// against the task already narrowed the stored value before rehydration.
enum StructuralPreparation {
    /// A compaction preparation.
    Compaction(CompactionPreparation),
    /// A branch preparation.
    BranchSummary(BranchPreparation),
}

/// What one structural attempt's generation call produced, upstream's
/// `performStructuralAttempt` result union.
enum AttemptOutcome {
    /// A compaction result, upstream's `{ kind: "compaction", ... }`.
    Compaction {
        /// The generated compaction data.
        result: CompactResult,
        /// Whether the last nested response's stop reason is retryable.
        retryable: bool,
    },
    /// A branch summary result, upstream's `{ kind: "branch_summary", ... }`.
    BranchSummary {
        /// The generated summary data.
        result: BranchSummaryResult,
        /// Whether the last nested response's stop reason is retryable.
        retryable: bool,
    },
    /// A generation failure, upstream's `{ kind: "error", ... }`.
    Error {
        /// The durable error the attempt failed with.
        error: OperationError,
        /// Whether the last nested response's stop reason is retryable.
        retryable: bool,
    },
    /// Cancellation won, upstream's `{ kind: "cancel_requested" }`.
    CancelRequested,
}

/// The terminal decision one structural publication commits, upstream's
/// `StructuralOutcome`.
#[derive(Clone, Debug)]
enum StructuralOutcome {
    /// A generated compaction, upstream's `{ kind: "compaction", ... }`.
    Compaction {
        /// The reserved summary entry id.
        result_entry_id: String,
        /// The compaction result.
        result: CompactResult,
        /// Whether a hook produced the result.
        from_hook: bool,
    },
    /// A generated branch summary, upstream's `{ kind: "branch_summary", ... }`.
    BranchSummary {
        /// The reserved summary entry id.
        result_entry_id: String,
        /// The branch summary result.
        result: BranchSummaryResult,
        /// Whether a hook produced the result.
        from_hook: bool,
    },
    /// The decision hook declined, upstream's `{ kind: "declined" }`.
    Declined,
    /// The attempt failed, upstream's `{ kind: "failed", ... }`.
    Failed {
        /// The failure's durable error.
        error: OperationError,
    },
}

impl StructuralOutcome {
    /// The preparation kind a result outcome carries, upstream's
    /// `outcome.kind` on the result halves; `None` on declined/failed.
    const fn kind(&self) -> Option<&'static str> {
        match self {
            Self::Compaction { .. } => Some("compaction"),
            Self::BranchSummary { .. } => Some("branch_summary"),
            Self::Declined | Self::Failed { .. } => None,
        }
    }

    /// The result usage a result outcome carries, upstream's
    /// `outcome.result.usage` on the result halves; `None` on
    /// declined/failed.
    const fn usage(&self) -> Option<&Usage> {
        match self {
            Self::Compaction { result, .. } => result.usage.as_ref(),
            Self::BranchSummary { result, .. } => result.usage.as_ref(),
            Self::Declined | Self::Failed { .. } => None,
        }
    }
}

/// What one structural publication's transaction materialized, upstream's
/// `StructuralPublication = ProcedureResult | BoundaryFinishPending` — the
/// finish-pending half leaves the transaction without a commit so the
/// caller mediates the finish through [`finish_run_boundary`].
enum StructuralPublication {
    /// A plain procedure result.
    Procedure(ProcedureResult),
    /// The declined-threshold finish mediation's planned entries.
    FinishPending(BoundaryFinishPending),
}

/// The three summary leaves a publication may run under, upstream's
/// `SummaryDecidingOperation | SummaryReadyOperation | SummaryEffectPendingOperation`
/// capability union.
#[derive(Clone, Debug)]
enum SummaryCapability {
    /// The `summary.deciding` leaf.
    Deciding(SummaryDecidingOperation),
    /// The `summary.ready` leaf.
    Ready(SummaryReadyOperation),
    /// The `summary.effect_pending` leaf.
    EffectPending(SummaryEffectPendingOperation),
}

impl SummaryCapability {
    /// The leaf's task, upstream's `capability.task`.
    const fn task(&self) -> &SummaryTask {
        match self {
            Self::Deciding(operation) => &operation.task,
            Self::Ready(operation) => &operation.generation.task,
            Self::EffectPending(operation) => &operation.generation.task,
        }
    }

    /// The leaf as the operation state the finish mediation's capability
    /// parameter takes.
    fn state(&self) -> OperationState {
        match self {
            Self::Deciding(operation) => OperationState::SummaryDeciding(operation.clone()),
            Self::Ready(operation) => OperationState::SummaryReady(operation.clone()),
            Self::EffectPending(operation) => {
                OperationState::SummaryEffectPending(operation.clone())
            }
        }
    }
}

/// The summary leaf a transaction re-read, upstream's `current` narrowed
/// from the fresh operation state the planner receives.
enum SummaryLeaf<'a> {
    /// The `summary.deciding` leaf.
    Deciding(&'a SummaryDecidingOperation),
    /// The `summary.ready` leaf.
    Ready(&'a SummaryReadyOperation),
    /// The `summary.effect_pending` leaf.
    EffectPending(&'a SummaryEffectPendingOperation),
}

impl<'a> SummaryLeaf<'a> {
    /// The leaf's task, upstream's `current.task`.
    const fn task(&self) -> &'a SummaryTask {
        match self {
            Self::Deciding(leaf) => &leaf.task,
            Self::Ready(leaf) => &leaf.generation.task,
            Self::EffectPending(leaf) => &leaf.generation.task,
        }
    }

    /// The attempt bookkeeping the terminal events carry, upstream's
    /// `current.at === "summary.ready" ? current.nextAttempt :
    /// current.at === "summary.effect_pending" ? current.attempt : undefined`.
    const fn attempt(&self) -> Option<u32> {
        match self {
            Self::Deciding(_) => None,
            Self::Ready(leaf) => Some(leaf.next_attempt),
            Self::EffectPending(leaf) => Some(leaf.attempt),
        }
    }
}

/// The preparation kind one summary task expects, upstream's `summaryKind`:
/// navigation tasks summarize their branch, everything else compacts.
const fn summary_kind(task: &SummaryTask) -> &'static str {
    if matches!(task.boundary, ResultBoundary::CommitNavigation { .. }) {
        "branch_summary"
    } else {
        "compaction"
    }
}

/// The preparation kind one durable preparation carries, upstream's
/// `stored.value.kind` probe.
const fn durable_kind(preparation: &DurableStructuralPreparation) -> &'static str {
    match preparation {
        DurableStructuralPreparation::Compaction { .. } => "compaction",
        DurableStructuralPreparation::BranchSummary { .. } => "branch_summary",
    }
}

/// The compaction trigger one task's events carry, upstream's
/// `compactionReason`: the task's own reason, `"manual"` for the finish
/// boundary's standalone compactions.
///
/// # Errors
/// The [`SessionError::Invariant`] `` `In-run compaction task {taskId} is
/// missing its reason` `` when a compaction-kind task carries no reason —
/// the admission paths always set one.
fn compaction_reason(task: &SummaryTask) -> Result<CompactionReason, LaneError> {
    if let Some(reason) = task.reason {
        return Ok(reason);
    }
    if matches!(task.boundary, ResultBoundary::Finish) {
        return Ok(CompactionReason::Manual);
    }
    Err(lane_error(SessionError::Invariant(format!(
        "In-run compaction task {} is missing its reason",
        task.task_id
    ))))
}

/// The commit-navigation boundary one branch-summary task settles into,
/// upstream's `navigationBoundary`.
///
/// # Errors
/// The [`SessionError::Invariant`] `` `Summary task {taskId} is not a
/// navigation` `` when the task's boundary is not a commit navigation —
/// the outcome-kind check already bound branch-summary outcomes to one.
fn navigation_boundary(task: &SummaryTask) -> Result<(&String, &Option<String>), LaneError> {
    match &task.boundary {
        ResultBoundary::CommitNavigation { target_id, label } => Ok((target_id, label)),
        _ => Err(lane_error(SessionError::Invariant(format!(
            "Summary task {} is not a navigation",
            task.task_id
        )))),
    }
}

/// The `before_request` hook's step kind for one summary task, upstream's
/// `step: summaryKind(effect.task)` — the wire `"compaction"` and
/// `"branchSummary"` names.
const fn summary_step_kind(task: &SummaryTask) -> StepKind {
    if matches!(task.boundary, ResultBoundary::CommitNavigation { .. }) {
        StepKind::BranchSummary
    } else {
        StepKind::Compaction
    }
}

/// Builds one durable operation error, upstream's `operationError`: the
/// `details` field is absent when not supplied, upstream's conditional
/// spread.
fn operation_error(code: &str, message: &str, details: Option<JsonValue>) -> OperationError {
    OperationError {
        code: code.to_owned(),
        message: message.to_owned(),
        details,
    }
}

/// The wire error code one compaction failure carries, upstream's
/// `CompactionErrorCode` strings.
const fn compaction_error_code(code: CompactionErrorCode) -> &'static str {
    match code {
        CompactionErrorCode::Aborted => "aborted",
        CompactionErrorCode::SummarizationFailed => "summarization_failed",
    }
}

/// The wire error code one branch-summary failure carries, upstream's
/// `BranchSummaryErrorCode` strings.
const fn branch_summary_error_code(code: BranchSummaryErrorCode) -> &'static str {
    match code {
        BranchSummaryErrorCode::Aborted => "aborted",
        BranchSummaryErrorCode::SummarizationFailed => "summarization_failed",
    }
}

/// The delay one failed attempt waits, upstream's `retryDelayMs(policy,
/// attempt)` over the generation context's normalized policy: pi-ai's delay
/// re-derives through the equivalent [`RetryPolicy`] (`maxAttempts − 1`
/// retries, the normalized delay cap made explicit), the same seam
/// [`crate::harness::runtime::drive::retry::retry_not_before`] builds.
fn policy_delay_ms(policy: &NormalizedRetryPolicy, attempt: u32) -> u64 {
    let policy = RetryPolicy {
        enabled: true,
        max_retries: policy.max_attempts.saturating_sub(1),
        base_delay_ms: policy.base_delay_ms,
        max_agent_delay_ms: Some(policy.max_agent_delay_ms),
    };
    retry_delay_ms(&policy, attempt)
}

/// Records the cancelled signal and returns the synthetic aborted
/// settlement the [`SummaryRequest`] boundary's infallible signature forces,
/// upstream's `throw new StructuralCancelled()`.
fn cancelled_settlement(signal: &RequestSignalCell, model: &Model) -> AssistantMessage {
    *signal.lock().unwrap_or_else(PoisonError::into_inner) = Some(RequestSignal::Cancelled);
    aborted_settlement(model, "Structural generation was cancelled")
}

/// Records one fault and returns the synthetic aborted settlement, upstream's
/// rethrow of any other error out of the request closure.
fn faulted_settlement(
    signal: &RequestSignalCell,
    model: &Model,
    error: LaneError,
) -> AssistantMessage {
    let message = error.to_string();
    *signal.lock().unwrap_or_else(PoisonError::into_inner) = Some(RequestSignal::Fault(error));
    aborted_settlement(model, &message)
}

/// The aborted settlement the request closure returns when it cannot carry
/// its signal out-of-band; the generation layer reads only `stopReason` and
/// `errorMessage` before the attempt's result mapping discards it.
fn aborted_settlement(model: &Model, message: &str) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Aborted,
        deferred: None,
        error_message: Some(message.to_owned()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms(),
    }
}

/// Builds the `usage` event one usage-write commit publishes, upstream's
/// `usageEvent`: the row's sequence reads the commit's write-index entry and
/// the totals read the commit's post-apply usage.
fn usage_event(
    row: &UsageWriteRow,
    write_index: usize,
    commit: &CommitResult,
    lane: &str,
) -> HarnessEvent {
    let Some(seq) = commit.seqs.get(write_index) else {
        unreachable!("commit carries one sequence per write");
    };
    global_event(
        "usage",
        HarnessEventPayload::Usage {
            lane: lane.to_owned(),
            row: UsageRow {
                id: row.id.clone(),
                seq: *seq,
                usage: row.usage,
                entry_id: row.entry_id.clone(),
                adjustment: row.adjustment,
                details: row.details.clone(),
            },
            totals: commit.stats.usage,
        },
    )
}

/// Serializes one live compaction preparation into the durable wire form,
/// upstream's `durableCompactionPreparation` — the accept-compaction
/// admission's write payload. Upstream's `durableFileOperations` Set→array
/// adapter restates as the shared sorted-vector [`FileOperations`](crate::harness::compaction::types::FileOperations)
/// wire type the live form already carries.
#[must_use]
pub(crate) fn durable_compaction_preparation(
    preparation: CompactionPreparation,
) -> DurableStructuralPreparation {
    DurableStructuralPreparation::Compaction {
        messages_to_summarize: preparation.messages_to_summarize,
        turn_prefix_messages: preparation.turn_prefix_messages,
        retained_tail: preparation.retained_tail,
        is_split_turn: preparation.is_split_turn,
        tokens_before: preparation.tokens_before,
        previous_summary: preparation.previous_summary,
        file_ops: preparation.file_ops,
        settings: preparation.settings,
    }
}

/// Serializes one live branch preparation into the durable wire form,
/// upstream's `durableBranchPreparation` — the summarized navigation
/// admission's write payload.
#[must_use]
pub(crate) fn durable_branch_preparation(
    preparation: BranchPreparation,
) -> DurableStructuralPreparation {
    DurableStructuralPreparation::BranchSummary {
        messages: preparation.messages,
        file_ops: preparation.file_ops,
        total_tokens: preparation.total_tokens,
    }
}

/// Rehydrates one durable compaction preparation into the live form,
/// upstream's `compactionPreparation` (upstream's `fileOperations`
/// Set-construction adapter restates as the shared wire type).
///
/// # Panics
/// Never reaches the wrong-variant arm: the caller's kind check against the
/// task bound the stored value first.
#[must_use]
fn compaction_preparation(durable: DurableStructuralPreparation) -> CompactionPreparation {
    let DurableStructuralPreparation::Compaction {
        messages_to_summarize,
        turn_prefix_messages,
        retained_tail,
        is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
        settings,
    } = durable
    else {
        unreachable!("the stored preparation kind matched the task's")
    };
    CompactionPreparation {
        messages_to_summarize,
        turn_prefix_messages,
        retained_tail,
        is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
        settings,
    }
}

/// Rehydrates one durable branch preparation into the live form, upstream's
/// `branchPreparation`.
///
/// # Panics
/// Never reaches the wrong-variant arm: the caller's kind check against the
/// task bound the stored value first.
#[must_use]
fn branch_preparation(durable: DurableStructuralPreparation) -> BranchPreparation {
    let DurableStructuralPreparation::BranchSummary {
        messages,
        file_ops,
        total_tokens,
    } = durable
    else {
        unreachable!("the stored preparation kind matched the task's")
    };
    BranchPreparation {
        messages,
        file_ops,
        total_tokens,
    }
}

/// Reads and rehydrates one summary task's durable preparation, upstream's
/// `readStructuralPreparation`.
///
/// # Errors
/// The invariants `` `Structural task {taskId} is missing its {expected}
/// preparation` `` (a missing or kind-mismatched stored value) and
/// `` `Navigation target {targetId} is missing` ``, plus the value read's
/// storage error.
async fn read_structural_preparation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    _deciding: &SummaryDecidingOperation,
) -> Result<ContinueOperationResult<StructuralPreparation>, LaneError> {
    let operation_id = drive.operation_id.clone();
    lane.continue_operation::<StructuralPreparation, _>(
        move |state, session, context| {
            let operation_id = operation_id.clone();
            Box::pin(async move {
                let Some(operation) = state.operation.as_ref() else {
                    unreachable!("continueOperation planner runs under an active operation")
                };
                let OperationState::SummaryDeciding(current) = &operation.state else {
                    unreachable!(
                        "continueOperation planner runs under the dispatched summary.deciding leaf"
                    )
                };
                let task = &current.task;
                let expected = summary_kind(task);
                let stored = session
                    .get_value(
                        &operation_preparation(&operation_id, &task.task_id).address,
                        &context,
                    )
                    .await
                    .map_err(lane_error)?;
                let Some(stored) = stored else {
                    return Err(lane_error(SessionError::Invariant(format!(
                        "Structural task {} is missing its {expected} preparation",
                        task.task_id
                    ))));
                };
                let stored: DurableStructuralPreparation = serde_json::from_value(stored.value)
                    .map_err(|error| {
                        lane_error(SessionError::Invariant(format!(
                            "Structural preparation {}:{} is malformed: {error}",
                            operation_id, task.task_id
                        )))
                    })?;
                if durable_kind(&stored) != expected {
                    return Err(lane_error(SessionError::Invariant(format!(
                        "Structural task {} is missing its {expected} preparation",
                        task.task_id
                    ))));
                }
                if let ResultBoundary::CommitNavigation { target_id, .. } = &task.boundary {
                    let found = session
                        .get_entries(vec![target_id.clone()], &context)
                        .await
                        .map_err(lane_error)?;
                    if !found.contains_key(target_id) {
                        return Err(lane_error(SessionError::Invariant(format!(
                            "Navigation target {target_id} is missing"
                        ))));
                    }
                }
                Ok(OperationCommand::Return {
                    result: match stored {
                        DurableStructuralPreparation::Compaction { .. } => {
                            StructuralPreparation::Compaction(compaction_preparation(stored))
                        }
                        DurableStructuralPreparation::BranchSummary { .. } => {
                            StructuralPreparation::BranchSummary(branch_preparation(stored))
                        }
                    },
                })
            })
        },
        &drive.context,
    )
    .await
}

/// Reads and rehydrates one ready attempt's durable preparation, upstream's
/// `readAttemptPreparation`.
///
/// # Errors
/// The invariant `` `Structural task {taskId} has invalid durable
/// preparation` `` when the stored value is missing or kind-mismatched,
/// plus the value read's storage error.
async fn read_attempt_preparation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    _ready: &SummaryReadyOperation,
) -> Result<ContinueOperationResult<StructuralPreparation>, LaneError> {
    let operation_id = drive.operation_id.clone();
    lane.continue_operation::<StructuralPreparation, _>(
        move |state, session, context| {
            let operation_id = operation_id.clone();
            Box::pin(async move {
                let Some(operation) = state.operation.as_ref() else {
                    unreachable!("continueOperation planner runs under an active operation")
                };
                let OperationState::SummaryReady(current) = &operation.state else {
                    unreachable!(
                        "continueOperation planner runs under the dispatched summary.ready leaf"
                    )
                };
                let task = &current.generation.task;
                let expected = summary_kind(task);
                let stored = session
                    .get_value(
                        &operation_preparation(&operation_id, &task.task_id).address,
                        &context,
                    )
                    .await
                    .map_err(lane_error)?;
                let stored: DurableStructuralPreparation = match stored {
                    Some(stored) => serde_json::from_value(stored.value).map_err(|error| {
                        lane_error(SessionError::Invariant(format!(
                            "Structural preparation {}:{} is malformed: {error}",
                            operation_id, task.task_id
                        )))
                    })?,
                    None => {
                        return Err(lane_error(SessionError::Invariant(format!(
                            "Structural task {} has invalid durable preparation",
                            task.task_id
                        ))));
                    }
                };
                if durable_kind(&stored) != expected {
                    return Err(lane_error(SessionError::Invariant(format!(
                        "Structural task {} has invalid durable preparation",
                        task.task_id
                    ))));
                }
                Ok(OperationCommand::Return {
                    result: match stored {
                        DurableStructuralPreparation::Compaction { .. } => {
                            StructuralPreparation::Compaction(compaction_preparation(stored))
                        }
                        DurableStructuralPreparation::BranchSummary { .. } => {
                            StructuralPreparation::BranchSummary(branch_preparation(stored))
                        }
                    },
                })
            })
        },
        &drive.context,
    )
    .await
}

/// Builds the summary generation inputs one ready transition carries,
/// upstream's `summaryContext`: the reserved entry id, the lane state's
/// configuration snapshot, the lane's stream options with the deferred flag
/// forced off, and the normalized retry policy.
fn summary_context(
    lane: &Lane,
    result_entry_id: String,
    configuration: &LaneConfiguration,
) -> SummaryContext {
    SummaryContext {
        result_entry_id,
        configuration: configuration.clone(),
        stream_options: AgentHarnessStreamOptions {
            deferred: Some(DeferredRequest::Enabled(false)),
            ..lane.read_config().stream_options
        },
        retry_policy: normalized_retry_policy(lane),
    }
}

/// Builds the `summary.ready` leaf one deciding transition commits, upstream's
/// `publishStructuralReady`'s planner: the id reserves outside the
/// transaction, upstream's `idGenerator.next()` before `continueOperation`.
async fn publish_structural_ready(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    _deciding: &SummaryDecidingOperation,
) -> Result<ProcedureResult, LaneError> {
    let result_entry_id = lane.session().id_generator().next(None);
    let planner_lane = Arc::clone(lane);
    let planner_result_entry_id = result_entry_id.clone();
    let published = lane
        .continue_operation::<ProcedureResult, _>(
            move |state, _session, _context| {
                let lane = Arc::clone(&planner_lane);
                let result_entry_id = planner_result_entry_id.clone();
                Box::pin(async move {
                    let Some(operation) = state.operation.as_ref() else {
                        unreachable!("continueOperation planner runs under an active operation")
                    };
                    let OperationState::SummaryDeciding(current) = &operation.state else {
                        unreachable!(
                            "continueOperation planner runs under the dispatched summary.deciding leaf"
                        )
                    };
                    let ready = SummaryReadyOperation {
                        scope: current.scope.clone(),
                        generation: SummaryGenerationScope {
                            task: current.task.clone(),
                            summary_context: summary_context(
                                &lane,
                                result_entry_id,
                                &state.configuration,
                            ),
                        },
                        next_attempt: 1,
                    };
                    Ok(OperationCommand::Commit {
                        writes: Vec::new(),
                        operation_state: OperationState::SummaryReady(ready),
                        lane: None,
                        materialize: Arc::new(|_: &CommitResult| ProcedureResult::Continue),
                        events: None,
                    })
                })
            },
            &drive.context,
        )
        .await?;
    Ok(match published {
        ContinueOperationResult::CancelRequested => ProcedureResult::Continue,
        ContinueOperationResult::Result { value } => value,
    })
}

/// Builds the `summary.effect_pending` leaf one ready transition commits,
/// upstream's `effectPendingFromReady`.
fn effect_pending_from_ready(ready: &SummaryReadyOperation) -> SummaryEffectPendingOperation {
    SummaryEffectPendingOperation {
        scope: ready.scope.clone(),
        generation: ready.generation.clone(),
        attempt: ready.next_attempt,
        request: None,
        usage_ids: Vec::new(),
    }
}

/// Builds the `summary.retry_wait` leaf one failed attempt schedules,
/// upstream's `retryWaitFromEffect`: the not-before derives from the
/// generation context's retry policy for the just-failed attempt.
fn retry_wait_from_effect(
    effect: &SummaryEffectPendingOperation,
    error_message: String,
) -> SummaryRetryWaitOperation {
    SummaryRetryWaitOperation {
        scope: effect.scope.clone(),
        generation: effect.generation.clone(),
        retry_wait: RetryWait {
            next_attempt: effect.attempt.saturating_add(1),
            not_before: retry_not_before(
                &effect.generation.summary_context.retry_policy,
                effect.attempt,
                now_ms(),
            ),
            error_message,
        },
    }
}

/// Builds the `summary.ready` leaf one retry wait resumes, upstream's
/// `readyFromRetryWait` — the attempt number carries through.
fn ready_from_retry_wait(retry: &SummaryRetryWaitOperation) -> SummaryReadyOperation {
    SummaryReadyOperation {
        scope: retry.scope.clone(),
        generation: retry.generation.clone(),
        next_attempt: retry.retry_wait.next_attempt,
    }
}

/// Publishes the attempt intent one ready transition commits, upstream's
/// `publishAttemptIntent`: the effect leaf materializes itself so the
/// caller receives the live state.
async fn publish_attempt_intent(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    _ready: &SummaryReadyOperation,
) -> Result<ContinueOperationResult<SummaryEffectPendingOperation>, LaneError> {
    lane.continue_operation::<SummaryEffectPendingOperation, _>(
        move |state, _session, _context| {
            Box::pin(async move {
                let Some(operation) = state.operation.as_ref() else {
                    unreachable!("continueOperation planner runs under an active operation")
                };
                let OperationState::SummaryReady(current) = &operation.state else {
                    unreachable!(
                        "continueOperation planner runs under the dispatched summary.ready leaf"
                    )
                };
                let effect_pending = effect_pending_from_ready(current);
                Ok(OperationCommand::Commit {
                    writes: Vec::new(),
                    operation_state: OperationState::SummaryEffectPending(effect_pending.clone()),
                    lane: None,
                    materialize: Arc::new(move |_: &CommitResult| effect_pending.clone()),
                    events: None,
                })
            })
        },
        &drive.context,
    )
    .await
}

/// Publishes the nested-request intent one provider request records before
/// it runs, upstream's `publishNestedRequestIntent`: the request rides the
/// effect leaf so a crash leaves the in-flight request durable.
async fn publish_nested_request_intent(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    _effect: &SummaryEffectPendingOperation,
    index: u64,
    usage_id: &str,
) -> Result<ContinueOperationResult<SummaryEffectPendingOperation>, LaneError> {
    let usage_id = usage_id.to_owned();
    lane.continue_operation::<SummaryEffectPendingOperation, _>(
        move |state, _session, _context| {
            let usage_id = usage_id.clone();
            Box::pin(async move {
                let Some(operation) = state.operation.as_ref() else {
                    unreachable!("continueOperation planner runs under an active operation")
                };
                let OperationState::SummaryEffectPending(current) = &operation.state else {
                    unreachable!(
                        "continueOperation planner runs under the dispatched summary.effect_pending leaf"
                    )
                };
                let mut next = current.clone();
                next.request = Some(SummaryEffectRequest {
                    index,
                    usage_id: usage_id.clone(),
                });
                Ok(OperationCommand::Commit {
                    writes: Vec::new(),
                    operation_state: OperationState::SummaryEffectPending(next.clone()),
                    lane: None,
                    materialize: Arc::new(move |_: &CommitResult| next.clone()),
                    events: None,
                })
            })
        },
        &drive.context,
    )
    .await
}

/// Settles one nested request's usage row and clears the request intent,
/// upstream's `publishNestedRequestOutcome`. This runs on
/// [`Lane::settle_operation`] — no cancel check — so the usage row lands
/// even when durable cancellation aborts the request. Upstream's capability
/// parameter types the leaf the landed planner signature erases; the plan
/// reads the fresh leaf instead.
///
/// # Errors
/// The commit's storage and delivery errors.
async fn publish_nested_request_outcome(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    _effect: &SummaryEffectPendingOperation,
    usage_id: &str,
    response: &AssistantMessage,
) -> Result<(), LaneError> {
    let lane_name = lane.name().to_owned();
    let usage_id = usage_id.to_owned();
    let response_usage = response.usage;
    lane.settle_operation::<(), _>(
        move |_state, latest, _meta, _reader, _context| {
            let lane_name = lane_name.clone();
            let usage_id = usage_id.clone();
            let latest = latest.clone();
            Box::pin(async move {
                let OperationState::SummaryEffectPending(current) = &latest else {
                    unreachable!(
                        "settleOperation planner runs under the dispatched summary.effect_pending leaf"
                    )
                };
                let mut next = current.clone();
                next.usage_ids.push(usage_id.clone());
                next.request = None;
                let row = UsageWriteRow {
                    id: usage_id.clone(),
                    usage: response_usage,
                    entry_id: None,
                    adjustment: false,
                    details: None,
                };
                Ok(OperationCommand::Commit {
                    writes: vec![Write::Usage(insert_usage(row.clone()))],
                    operation_state: OperationState::SummaryEffectPending(next),
                    lane: None,
                    materialize: Arc::new(|_: &CommitResult| ()),
                    events: Some(Arc::new(move |commit: &CommitResult| {
                        vec![usage_event(&row, 0, commit, &lane_name)]
                    })),
                })
            })
        },
        &drive.context,
    )
    .await
}

/// Builds the simple-request options one summary request sends, upstream's
/// `requestStreamOptions`: the generation layer's options carry the prompt
/// sizing, the harness stream options carry the curated transport/timeout/
/// retry/headers/metadata fields, the cache hint drops, the deferred flag
/// forces off, and the signal and telemetry parent read the admitted
/// request context.
#[must_use]
fn request_stream_options(
    options: &SimpleStreamOptions,
    stream_options: &AgentHarnessStreamOptions,
    context: &Context,
    on_payload: OnPayload,
) -> SimpleStreamOptions {
    SimpleStreamOptions {
        transport: stream_options.transport,
        timeout_ms: stream_options.timeout_ms,
        max_retries: stream_options.max_retries,
        max_retry_delay_ms: stream_options.max_retry_delay_ms,
        headers: stream_options.headers.clone().map(|headers| {
            headers
                .into_iter()
                .map(|(key, value)| (key, Some(value)))
                .collect()
        }),
        metadata: stream_options.metadata.clone(),
        cache_retention: Some(CacheRetention::None),
        deferred: Some(DeferredRequest::Enabled(false)),
        telemetry_context: Some(get_telemetry_context(context)),
        transport_options: TransportOptions {
            http_client: options.transport_options.http_client.clone(),
            signal: request_signal(context),
            on_payload: Some(on_payload),
            on_response: options.transport_options.on_response.clone(),
        },
        ..options.clone()
    }
}

/// Performs one structural generation attempt, upstream's
/// `performStructuralAttempt`: each nested provider request publishes its
/// durable intent, runs the hook-gated completion inside the gate, and
/// settles its usage row; retryability reads the LAST nested response's
/// stop reason, not the transport failure.
///
/// # Errors
/// The preparation invariants — `` `Compaction summary has invalid durable
/// preparation` `` and `` `Branch summary has invalid durable preparation` ``
/// — and any fault the nested request recorded.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single performStructuralAttempt method and its per-request closure"
)]
async fn perform_structural_attempt(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    effect: &SummaryEffectPendingOperation,
    model: &Model,
    preparation: &StructuralPreparation,
) -> Result<AttemptOutcome, LaneError> {
    let request_index = Arc::new(AtomicU64::new(0));
    let last_response: Arc<Mutex<Option<AssistantMessage>>> = Arc::new(Mutex::new(None));
    let signal: RequestSignalCell = Arc::new(Mutex::new(None));
    let request: SummaryRequest = {
        let lane = Arc::clone(lane);
        let drive = Arc::clone(drive);
        let effect = effect.clone();
        let model = model.clone();
        let request_index = Arc::clone(&request_index);
        let last_response = Arc::clone(&last_response);
        let signal = Arc::clone(&signal);
        Arc::new(
            move |ai_context: &AiContext,
                  options: &SimpleStreamOptions,
                  request_context: &Context| {
                let ai_context = ai_context.clone();
                let options = options.clone();
                let request_context = request_context.clone();
                let lane = Arc::clone(&lane);
                let drive = Arc::clone(&drive);
                let effect = effect.clone();
                let model = model.clone();
                let request_index = Arc::clone(&request_index);
                let last_response = Arc::clone(&last_response);
                let signal = Arc::clone(&signal);
                Box::pin(async move {
                    let base_options = AgentHarnessStreamOptions {
                        deferred: Some(DeferredRequest::Enabled(false)),
                        ..effect.generation.summary_context.stream_options.clone()
                    };
                    let hook = match lane
                        .hooks()
                        .run_with_gate(
                            HookName::BeforeRequest,
                            HookInvocation {
                                lane: lane.name().to_owned(),
                                run_id: drive.operation_id.clone(),
                                event: HookEvent::BeforeRequest {
                                    model: model.clone(),
                                    step: summary_step_kind(&effect.generation.task),
                                    attempt: effect.attempt,
                                    stream_options: base_options.clone(),
                                },
                            },
                            &drive.gate,
                            &request_context,
                        )
                        .await
                    {
                        Ok(hook) => hook,
                        Err(HookRunError::GateAborted(abort)) => {
                            await_abort_cancellation(&abort.cancellation).await;
                            return cancelled_settlement(&signal, &model);
                        }
                        Err(HookRunError::Aborted(_)) => {
                            await_abort_cancellation(&drive.abort_cancellation()).await;
                            return cancelled_settlement(&signal, &model);
                        }
                        Err(error) => {
                            return faulted_settlement(&signal, &model, lane_error(error));
                        }
                    };
                    let HookResult::BeforeRequest(patch) = hook else {
                        unreachable!("before_request returns its own result variant")
                    };
                    let patched = patch.map_or_else(
                        || base_options.clone(),
                        |patch| apply_stream_options_patch(&base_options, &patch.stream_options),
                    );
                    let mut stream_options = patched;
                    stream_options.deferred = Some(DeferredRequest::Enabled(false));
                    let usage_id = lane.session().id_generator().next(None);
                    let index = request_index.load(Ordering::Relaxed);
                    let intent = match publish_nested_request_intent(
                        &lane, &drive, &effect, index, &usage_id,
                    )
                    .await
                    {
                        Ok(intent) => intent,
                        Err(error) => {
                            return faulted_settlement(&signal, &model, error);
                        }
                    };
                    request_index.fetch_add(1, Ordering::Relaxed);
                    let ContinueOperationResult::Result { value: effect } = intent else {
                        return cancelled_settlement(&signal, &model);
                    };
                    let admitted_context =
                        with_abort_signal(drive.gate.signal().clone(), &request_context);
                    let on_payload = {
                        let lane = Arc::clone(&lane);
                        let drive = Arc::clone(&drive);
                        let hook_context = admitted_context.clone();
                        OnPayload::new(move |payload: JsonValue, request_model: Model| {
                            let lane = Arc::clone(&lane);
                            let drive = Arc::clone(&drive);
                            let admitted_context = hook_context.clone();
                            Box::pin(async move {
                                match lane
                                    .hooks()
                                    .run_with_gate(
                                        HookName::BeforePayload,
                                        HookInvocation {
                                            lane: lane.name().to_owned(),
                                            run_id: drive.operation_id.clone(),
                                            event: HookEvent::BeforePayload {
                                                model: request_model,
                                                payload,
                                            },
                                        },
                                        &drive.gate,
                                        &admitted_context,
                                    )
                                    .await
                                {
                                    Ok(HookResult::BeforePayload(result)) => {
                                        result.map(|result| result.payload)
                                    }
                                    Ok(_) => {
                                        unreachable!(
                                            "before_payload returns its own result variant"
                                        )
                                    }
                                    // The OnPayload seam carries no error
                                    // channel; the request's linked
                                    // cancellation token aborts it when the
                                    // signal aborted, upstream's onPayload
                                    // throw reaching the same settlement.
                                    Err(_) => None,
                                }
                            })
                        })
                    };
                    let request_options = request_stream_options(
                        &options,
                        &stream_options,
                        &admitted_context,
                        on_payload,
                    );
                    let response = match drive.gate.admit(|| {
                        let models = Arc::clone(lane.models());
                        let model = model.clone();
                        let ai_context = ai_context.clone();
                        let request_options = request_options.clone();
                        Box::pin(async move {
                            let request_options = WithTransforms {
                                options: request_options,
                                transform_headers: None,
                            };
                            models
                                .complete_simple(&model, &ai_context, Some(&request_options))
                                .await
                        })
                    }) {
                        Ok(response) => response.await,
                        Err(GateRejection::AbortRequested(abort)) => {
                            await_abort_cancellation(&abort.cancellation).await;
                            return cancelled_settlement(&signal, &model);
                        }
                        Err(GateRejection::Closed(error)) => {
                            return faulted_settlement(&signal, &model, lane_error(error));
                        }
                    };
                    *last_response.lock().unwrap_or_else(PoisonError::into_inner) =
                        Some(response.clone());
                    if let Err(error) =
                        publish_nested_request_outcome(&lane, &drive, &effect, &usage_id, &response)
                            .await
                    {
                        return faulted_settlement(&signal, &model, error);
                    }
                    response
                })
            },
        )
    };

    let generated = if summary_kind(&effect.generation.task) == "compaction" {
        let StructuralPreparation::Compaction(preparation) = preparation else {
            return Err(lane_error(SessionError::Invariant(
                "Compaction summary has invalid durable preparation".to_owned(),
            )));
        };
        let options = CompactGenerationOptions {
            model: model.clone(),
            custom_instructions: effect.generation.task.custom_instructions.clone(),
            thinking_level: Some(
                effect
                    .generation
                    .summary_context
                    .configuration
                    .thinking_level,
            ),
        };
        match compact_with_request(preparation, &options, &request, &drive.context).await {
            Ok(result) => GeneratedOutcome::Compaction(result),
            Err(error) => GeneratedOutcome::Error {
                code: compaction_error_code(error.code),
                message: error.message,
            },
        }
    } else {
        let StructuralPreparation::BranchSummary(preparation) = preparation else {
            return Err(lane_error(SessionError::Invariant(
                "Branch summary has invalid durable preparation".to_owned(),
            )));
        };
        let options = PreparedBranchSummaryOptions {
            custom_instructions: effect.generation.task.custom_instructions.clone(),
            replace_instructions: None,
        };
        match generate_branch_summary_with_request(preparation, &options, &request, &drive.context)
            .await
        {
            Ok(result) => GeneratedOutcome::BranchSummary(result),
            Err(error) => GeneratedOutcome::Error {
                code: branch_summary_error_code(error.code),
                message: error.message,
            },
        }
    };

    let signal_taken = signal.lock().unwrap_or_else(PoisonError::into_inner).take();
    if let Some(RequestSignal::Fault(error)) = &signal_taken {
        return Err(Arc::clone(error));
    }
    if matches!(signal_taken, Some(RequestSignal::Cancelled)) {
        return Ok(AttemptOutcome::CancelRequested);
    }
    let retryable = {
        let guard = last_response.lock().unwrap_or_else(PoisonError::into_inner);
        guard.as_ref().is_some_and(is_retryable_assistant_error)
    };
    Ok(match generated {
        GeneratedOutcome::Compaction(result) => AttemptOutcome::Compaction { result, retryable },
        GeneratedOutcome::BranchSummary(result) => {
            AttemptOutcome::BranchSummary { result, retryable }
        }
        GeneratedOutcome::Error { code, message } => AttemptOutcome::Error {
            error: operation_error(code, &message, None),
            retryable,
        },
    })
}

/// What the generation call produced before the attempt's signal mapping,
/// the intermediate the two preparation branches unify into.
enum GeneratedOutcome {
    /// A compaction result.
    Compaction(CompactResult),
    /// A branch summary result.
    BranchSummary(BranchSummaryResult),
    /// A generation failure with its wire code.
    Error { code: &'static str, message: String },
}

/// Publishes one attempt's result, upstream's `publishAttemptResult`: a
/// result commits through the terminal transaction, a retryable failure
/// under the attempt budget schedules the wait, and anything else fails the
/// task.
///
/// # Errors
/// The aborted-while-running invariant `` `Structural provider response is
/// aborted while durable control is running` ``, the terminal transaction's
/// errors, and the commit's storage and delivery errors.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single publishAttemptResult method and its retry-scheduling commit"
)]
async fn publish_attempt_result(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    effect: &SummaryEffectPendingOperation,
    result: AttemptOutcome,
) -> Result<ProcedureResult, LaneError> {
    match result {
        AttemptOutcome::CancelRequested => Ok(ProcedureResult::Continue),
        AttemptOutcome::Compaction { result, .. } => {
            publish_structural_outcome(
                lane,
                drive,
                SummaryCapability::EffectPending(effect.clone()),
                StructuralOutcome::Compaction {
                    result_entry_id: effect.generation.summary_context.result_entry_id.clone(),
                    result,
                    from_hook: false,
                },
            )
            .await
        }
        AttemptOutcome::BranchSummary { result, .. } => {
            publish_structural_outcome(
                lane,
                drive,
                SummaryCapability::EffectPending(effect.clone()),
                StructuralOutcome::BranchSummary {
                    result_entry_id: effect.generation.summary_context.result_entry_id.clone(),
                    result,
                    from_hook: false,
                },
            )
            .await
        }
        AttemptOutcome::Error { error, retryable } => {
            if error.code == "aborted" {
                // Upstream's non-null assertion cannot tolerate an absent
                // operation; the invariant fires there too.
                let running = match lane.state().operation {
                    Some(operation) => matches!(
                        operation_scope_of(&operation.state).control,
                        Control::Running
                    ),
                    None => true,
                };
                if running {
                    return Err(lane_error(SessionError::Invariant(
                        "Structural provider response is aborted while durable control is running"
                            .to_owned(),
                    )));
                }
            }
            let max_attempts = effect.generation.summary_context.retry_policy.max_attempts;
            if retryable && effect.attempt < max_attempts {
                let retry_wait = retry_wait_from_effect(effect, error.message.clone());
                let lane_name = lane.name().to_owned();
                let run_id = drive.operation_id.clone();
                let step = effect.generation.task.task_id.clone();
                let attempt = retry_wait.retry_wait.next_attempt;
                let delay_ms = policy_delay_ms(
                    &effect.generation.summary_context.retry_policy,
                    effect.attempt,
                );
                let not_before = retry_wait.retry_wait.not_before;
                let error_message = error.message.clone();
                let planner_retry_wait = retry_wait.clone();
                let published = lane
                    .continue_operation::<ProcedureResult, _>(
                        move |_state, _session, _context| {
                            let retry_wait = planner_retry_wait.clone();
                            let lane_name = lane_name.clone();
                            let run_id = run_id.clone();
                            let step = step.clone();
                            let error_message = error_message.clone();
                            Box::pin(async move {
                                Ok(OperationCommand::Commit {
                                    writes: Vec::new(),
                                    operation_state: OperationState::SummaryRetryWait(retry_wait),
                                    lane: None,
                                    materialize: Arc::new(|_: &CommitResult| {
                                        ProcedureResult::Continue
                                    }),
                                    events: Some(Arc::new(move |_: &CommitResult| {
                                        vec![lane_scoped_event(
                                            &lane_name,
                                            false,
                                            "retry_scheduled",
                                            HarnessEventPayload::RetryScheduled {
                                                run_id: run_id.clone(),
                                                step: step.clone(),
                                                attempt,
                                                max_attempts,
                                                delay_ms,
                                                not_before,
                                                error_message: error_message.clone(),
                                            },
                                        )]
                                    })),
                                })
                            })
                        },
                        &drive.context,
                    )
                    .await?;
                return Ok(match published {
                    ContinueOperationResult::CancelRequested => ProcedureResult::Continue,
                    ContinueOperationResult::Result { value } => value,
                });
            }
            publish_structural_outcome(
                lane,
                drive,
                SummaryCapability::EffectPending(effect.clone()),
                StructuralOutcome::Failed { error },
            )
            .await
        }
    }
}

/// Consume one durable structural preparation and decision hook, upstream's
/// `runStructuralDecision`.
///
/// The preparation rehydrates inside a cancel-checked transaction, then the
/// task's boundary picks the hook: `before_navigation` for navigation
/// tasks, `before_compaction` for compaction ones. A hook decline or
/// replacement publishes through the terminal transaction directly; an
/// accepted decision reserves the summary entry id and commits the ready
/// leaf.
///
/// # Errors
/// The preparation invariants — `` `Structural task {taskId} is missing its
/// {expected} preparation` ``, `` `Navigation target {targetId} is
/// missing` ``, `` `Navigation task has invalid durable preparation` ``, and
/// `` `Compaction task has invalid durable preparation` `` — the hook
/// failures outside the abort path, and the publication's storage and
/// delivery errors.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single runStructuralDecision method and its two boundary branches"
)]
pub(crate) async fn run_structural_decision(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    deciding: &SummaryDecidingOperation,
) -> Result<ProcedureResult, LaneError> {
    let preparation = read_structural_preparation(lane, drive, deciding).await?;
    let ContinueOperationResult::Result { value: preparation } = preparation else {
        return Ok(ProcedureResult::Continue);
    };
    if let ResultBoundary::CommitNavigation { target_id, .. } = &deciding.task.boundary {
        let StructuralPreparation::BranchSummary(preparation) = preparation else {
            return Err(lane_error(SessionError::Invariant(
                "Navigation task has invalid durable preparation".to_owned(),
            )));
        };
        let hook = match lane
            .hooks()
            .run_with_gate(
                HookName::BeforeNavigation,
                HookInvocation {
                    lane: lane.name().to_owned(),
                    run_id: drive.operation_id.clone(),
                    event: HookEvent::BeforeNavigation {
                        target_id: target_id.clone(),
                        preparation: preparation.clone(),
                        custom_instructions: deciding.task.custom_instructions.clone(),
                    },
                },
                &drive.gate,
                &drive.context,
            )
            .await
        {
            Ok(hook) => hook,
            Err(error) => return Err(hook_error_to_lane_error(error, drive)),
        };
        let HookResult::BeforeNavigation(hook) = hook else {
            unreachable!("before_navigation returns its own result variant")
        };
        if hook.as_ref().is_some_and(|hook| hook.decline == Some(true)) {
            return publish_structural_outcome(
                lane,
                drive,
                SummaryCapability::Deciding(deciding.clone()),
                StructuralOutcome::Declined,
            )
            .await;
        }
        if let Some(result) = hook.and_then(|hook| hook.summary) {
            return publish_structural_outcome(
                lane,
                drive,
                SummaryCapability::Deciding(deciding.clone()),
                StructuralOutcome::BranchSummary {
                    result_entry_id: lane.session().id_generator().next(None),
                    result,
                    from_hook: true,
                },
            )
            .await;
        }
        return publish_structural_ready(lane, drive, deciding).await;
    }

    let StructuralPreparation::Compaction(preparation) = preparation else {
        return Err(lane_error(SessionError::Invariant(
            "Compaction task has invalid durable preparation".to_owned(),
        )));
    };
    let reason = compaction_reason(&deciding.task)?;
    let hook = match lane
        .hooks()
        .run_with_gate(
            HookName::BeforeCompaction,
            HookInvocation {
                lane: lane.name().to_owned(),
                run_id: drive.operation_id.clone(),
                event: HookEvent::BeforeCompaction {
                    reason,
                    preparation: preparation.clone(),
                    custom_instructions: deciding.task.custom_instructions.clone(),
                },
            },
            &drive.gate,
            &drive.context,
        )
        .await
    {
        Ok(hook) => hook,
        Err(error) => return Err(hook_error_to_lane_error(error, drive)),
    };
    let HookResult::BeforeCompaction(hook) = hook else {
        unreachable!("before_compaction returns its own result variant")
    };
    if hook.as_ref().is_some_and(|hook| hook.decline == Some(true)) {
        return publish_structural_outcome(
            lane,
            drive,
            SummaryCapability::Deciding(deciding.clone()),
            StructuralOutcome::Declined,
        )
        .await;
    }
    if let Some(result) = hook.and_then(|hook| hook.compaction) {
        return publish_structural_outcome(
            lane,
            drive,
            SummaryCapability::Deciding(deciding.clone()),
            StructuralOutcome::Compaction {
                result_entry_id: lane.session().id_generator().next(None),
                result,
                from_hook: true,
            },
        )
        .await;
    }
    publish_structural_ready(lane, drive, deciding).await
}

/// Execute one ready structural generation attempt, upstream's
/// `runStructuralGeneration`.
///
/// The preparation rehydrates, the CAPTURED configuration's model resolves
/// against the registry, the attempt intent commits, and the nested
/// requests run inside the gate. A result publishes through the terminal
/// transaction; a missing model fails the task with the configuration's
/// provenance.
///
/// # Errors
/// The preparation and configuration invariants — `` `Structural task
/// {taskId} has invalid durable preparation` `` and the terminal
/// publication's invariants — the hook failures outside the abort path, and
/// the publication's storage and delivery errors.
pub(crate) async fn run_structural_generation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    ready: &SummaryReadyOperation,
) -> Result<ProcedureResult, LaneError> {
    let preparation = read_attempt_preparation(lane, drive, ready).await?;
    let ContinueOperationResult::Result { value: preparation } = preparation else {
        return Ok(ProcedureResult::Continue);
    };
    let identity = ready.generation.summary_context.configuration.model.clone();
    let model = lane.models().model(&identity.provider, &identity.model_id);
    let Some(model) = model else {
        return publish_structural_outcome(
            lane,
            drive,
            SummaryCapability::Ready(ready.clone()),
            StructuralOutcome::Failed {
                error: operation_error(
                    "model_unavailable",
                    "The configured model is unavailable in this process",
                    Some(serde_json::to_value(&identity).unwrap_or_default()),
                ),
            },
        )
        .await;
    };
    let intent = publish_attempt_intent(lane, drive, ready).await?;
    let ContinueOperationResult::Result { value: effect } = intent else {
        return Ok(ProcedureResult::Continue);
    };
    let result = perform_structural_attempt(lane, drive, &effect, &model, &preparation).await?;
    publish_attempt_result(lane, drive, &effect, result).await
}

/// Convert an orphaned structural attempt into a fresh numbered attempt or
/// terminal failure, upstream's `recoverStructuralGeneration`. The
/// orphaned attempt's nested request is never resumed: only the durable
/// leaf advances.
///
/// # Errors
/// The terminal publication's invariants and the commit's storage and
/// delivery errors.
pub(crate) async fn recover_structural_generation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    effect: &SummaryEffectPendingOperation,
) -> Result<ProcedureResult, LaneError> {
    let error = operation_error(
        "structural_interrupted",
        "Structural summary attempt was interrupted and its external outcome is unknown",
        None,
    );
    if effect.attempt >= effect.generation.summary_context.retry_policy.max_attempts {
        return publish_structural_outcome(
            lane,
            drive,
            SummaryCapability::EffectPending(effect.clone()),
            StructuralOutcome::Failed { error },
        )
        .await;
    }
    let retry_wait = retry_wait_from_effect(effect, error.message.clone());
    let lane_name = lane.name().to_owned();
    let run_id = drive.operation_id.clone();
    let step = effect.generation.task.task_id.clone();
    let attempt = retry_wait.retry_wait.next_attempt;
    let max_attempts = effect.generation.summary_context.retry_policy.max_attempts;
    let delay_ms = policy_delay_ms(
        &effect.generation.summary_context.retry_policy,
        effect.attempt,
    );
    let not_before = retry_wait.retry_wait.not_before;
    let error_message = error.message.clone();
    let planner_retry_wait = retry_wait.clone();
    let published = lane
        .continue_operation::<ProcedureResult, _>(
            move |_state, _session, _context| {
                let retry_wait = planner_retry_wait.clone();
                let lane_name = lane_name.clone();
                let run_id = run_id.clone();
                let step = step.clone();
                let error_message = error_message.clone();
                Box::pin(async move {
                    Ok(OperationCommand::Commit {
                        writes: Vec::new(),
                        operation_state: OperationState::SummaryRetryWait(retry_wait),
                        lane: None,
                        materialize: Arc::new(|_: &CommitResult| ProcedureResult::Continue),
                        events: Some(Arc::new(move |_: &CommitResult| {
                            vec![lane_scoped_event(
                                &lane_name,
                                true,
                                "retry_scheduled",
                                HarnessEventPayload::RetryScheduled {
                                    run_id: run_id.clone(),
                                    step: step.clone(),
                                    attempt,
                                    max_attempts,
                                    delay_ms,
                                    not_before,
                                    error_message: error_message.clone(),
                                },
                            )]
                        })),
                    })
                })
            },
            &drive.context,
        )
        .await?;
    Ok(match published {
        ContinueOperationResult::CancelRequested => ProcedureResult::Continue,
        ContinueOperationResult::Result { value } => value,
    })
}

/// Consume one structural retry wait without starting a provider effect,
/// upstream's `runStructuralRetryWait`.
///
/// A not-yet-due wait either returns the `waiting` outcome for the caller
/// to re-drive later (`waitForRetry` off) or sleeps inside a gate permit
/// (`waitForRetry` on) — an abort rejects through the gate so the spine
/// awaits the cancellation. The `retry_start` event publishes even when no
/// wait elapsed.
///
/// # Errors
/// The abort rejection's cancellation and the commit's storage and delivery
/// errors.
pub(crate) async fn run_structural_retry_wait(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    retry: &SummaryRetryWaitOperation,
) -> Result<ProcedureResult, LaneError> {
    if now_ms() < retry.retry_wait.not_before {
        if !drive.wait_for_retry {
            return Ok(ProcedureResult::Waiting {
                outcome: DriveOutcome::Waiting {
                    operation_id: drive.operation_id.clone(),
                    reason: DriveWaitReason::Retry {
                        not_before: retry.retry_wait.not_before,
                    },
                },
            });
        }
        match drive
            .gate
            .admit(|| wait_until(retry.retry_wait.not_before, drive))
        {
            Ok(wait) => wait.await?,
            Err(rejection) => return Err(gate_rejection_error(rejection)),
        }
    }
    let planner_lane = Arc::clone(lane);
    let planner_drive = Arc::clone(drive);
    let published = lane
        .continue_operation::<ProcedureResult, _>(
            move |state, _session, _context| {
                let lane = Arc::clone(&planner_lane);
                let drive = Arc::clone(&planner_drive);
                Box::pin(async move {
                    let Some(operation) = state.operation.as_ref() else {
                        unreachable!("continueOperation planner runs under an active operation")
                    };
                    let OperationState::SummaryRetryWait(current) = &operation.state else {
                        unreachable!(
                            "continueOperation planner runs under the dispatched summary.retry_wait leaf"
                        )
                    };
                    let ready = ready_from_retry_wait(current);
                    let lane_name = lane.name().to_owned();
                    let run_id = drive.operation_id.clone();
                    let step = current.generation.task.task_id.clone();
                    let attempt = current.retry_wait.next_attempt;
                    Ok(OperationCommand::Commit {
                        writes: Vec::new(),
                        operation_state: OperationState::SummaryReady(ready),
                        lane: None,
                        materialize: Arc::new(|_: &CommitResult| ProcedureResult::Continue),
                        events: Some(Arc::new(move |_: &CommitResult| {
                            vec![lane_scoped_event(
                                &lane_name,
                                false,
                                "retry_start",
HarnessEventPayload::RetryStart {
                                    run_id: run_id.clone(),
                                    step: step.clone(),
                                    attempt,
                                },
                            )]
                        })),
                    })
                })
            },
            &drive.context,
        )
        .await?;
    Ok(match published {
        ContinueOperationResult::CancelRequested => ProcedureResult::Continue,
        ContinueOperationResult::Result { value } => value,
    })
}

/// The base-event pieces one terminal transaction precomputes, upstream's
/// `baseEvents` array's captured data.
struct BaseEventPlan {
    /// The hook usage row and its write index, when a hook summary carries
    /// usage.
    hook_usage_row: Option<(UsageWriteRow, usize)>,
    /// The summary entry and its write index, when the outcome commits one.
    entry_plan: Option<(NewEntry, usize)>,
    /// The retry-end bookkeeping — attempt, success, final error — when the
    /// leaf is past its first attempt.
    retry_end: Option<(u32, bool, Option<String>)>,
    /// The completed compaction end — reason and entry id — when the
    /// outcome commits a compaction.
    completed_end: Option<(CompactionReason, String)>,
    /// The task id the retry events carry, upstream's `step`.
    step: String,
}

/// Builds the transaction's base-event materializer from the precomputed
/// write plan, upstream's `baseEvents` array: the hook usage row first,
/// then the entry's lifecycle events, the retry end, and the completed
/// compaction end. The completed end's `endedAt` reads
/// `terminal_compaction_ended_at` when the finish boundary assigned it —
/// upstream's closure reads the mutated variable at materialize time —
/// and otherwise the commit's timestamp.
#[must_use]
fn base_events_fn(
    plan: BaseEventPlan,
    terminal_compaction_ended_at: Option<i64>,
    lane_name: String,
    run_id: String,
) -> EventsFn {
    let BaseEventPlan {
        hook_usage_row,
        entry_plan,
        retry_end,
        completed_end,
        step,
    } = plan;
    Arc::new(move |commit: &CommitResult| {
        let mut events = Vec::new();
        if let Some((row, write_index)) = &hook_usage_row {
            events.push(usage_event(row, *write_index, commit, &lane_name));
        }
        if let Some((entry, entry_write_index)) = &entry_plan {
            events.extend(committed_entry_events(
                std::slice::from_ref(entry),
                commit,
                &lane_name,
                Some(&run_id),
                *entry_write_index,
            ));
        }
        if let Some((attempt, success, final_error)) = &retry_end {
            events.push(lane_scoped_event(
                &lane_name,
                false,
                "retry_end",
                HarnessEventPayload::RetryEnd {
                    run_id: run_id.clone(),
                    step: step.clone(),
                    attempt: *attempt,
                    success: *success,
                    final_error: final_error.clone(),
                },
            ));
        }
        if let Some((reason, entry_id)) = &completed_end {
            events.push(lane_scoped_event(
                &lane_name,
                false,
                "compaction_end",
                HarnessEventPayload::CompactionEnd {
                    run_id: run_id.clone(),
                    reason: *reason,
                    ended_at: terminal_compaction_ended_at.unwrap_or(commit.timestamp),
                    status: CompactionEndStatus::Completed {
                        entry_id: entry_id.clone(),
                    },
                },
            ));
        }
        events
    })
}

/// Publishes one structural outcome through the terminal transaction,
/// upstream's `publishStructuralOutcome`: the hook usage row and the entry
/// insert ride one commit in write order, the attempt bookkeeping and
/// terminal events materialize from it, and the task's boundary picks the
/// settlement — the resume-checkpoint boundary replans the inbox, the
/// finish and commit-navigation boundaries record their result.
///
/// # Errors
/// The invariants `` `Structural {kind} result does not match {expected}
/// task {taskId}` ``, `` `Hook usage id exists without structural usage` ``,
/// the boundary-switch invariants (`` `Run compaction boundary received a
/// branch summary` ``, `` `Run compaction has no Branch tip` ``, `` `Failed
/// run has no Branch tip` ``, `` `Compaction finish boundary received a
/// branch summary` ``, `` `Standalone compaction has no Branch tip` ``,
/// `` `Navigation boundary received a compaction result` ``, and
/// `` `Structural finish mediation requires a resumable finish boundary` ``),
/// the placement's and cleanup's errors, and the commit's storage and
/// delivery errors.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single publishStructuralOutcome method and its boundary switch"
)]
async fn publish_structural_outcome(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    capability: SummaryCapability,
    outcome: StructuralOutcome,
) -> Result<ProcedureResult, LaneError> {
    // The hook usage row's id reserves outside the transaction, upstream's
    // idGenerator.next() before continueOperation.
    let hook_usage_id = match &outcome {
        StructuralOutcome::Compaction {
            result,
            from_hook: true,
            ..
        } if result.usage.is_some() => Some(lane.session().id_generator().next(None)),
        StructuralOutcome::BranchSummary {
            result,
            from_hook: true,
            ..
        } if result.usage.is_some() => Some(lane.session().id_generator().next(None)),
        _ => None,
    };
    let planner_lane = Arc::clone(lane);
    let planner_drive = Arc::clone(drive);
    let planner_outcome = outcome.clone();
    let published = lane
        .continue_operation::<StructuralPublication, _>(
            move |state, session, _context| {
                let lane = Arc::clone(&planner_lane);
                let drive = Arc::clone(&planner_drive);
                let outcome = planner_outcome.clone();
                let hook_usage_id = hook_usage_id.clone();
                Box::pin(async move {
                    let Some(operation) = state.operation.as_ref() else {
                        unreachable!("continueOperation planner runs under an active operation")
                    };
                    let leaf = match &operation.state {
                        OperationState::SummaryDeciding(leaf) => SummaryLeaf::Deciding(leaf),
                        OperationState::SummaryReady(leaf) => SummaryLeaf::Ready(leaf),
                        OperationState::SummaryEffectPending(leaf) => {
                            SummaryLeaf::EffectPending(leaf)
                        }
                        other => unreachable!(
                            "continueOperation planner runs under the {} leaf",
                            other.at()
                        ),
                    };
                    let task = leaf.task();
                    let expected = summary_kind(task);
                    if let Some(kind) = outcome.kind()
                        && kind != expected
                    {
                        return Err(lane_error(SessionError::Invariant(format!(
                            "Structural {kind} result does not match {expected} task {}",
                            task.task_id
                        ))));
                    }

                    let mut writes: Vec<Write> = Vec::new();
                    let hook_usage_row: Option<(UsageWriteRow, usize)> =
                        if let Some(id) = &hook_usage_id {
                            let Some(usage) = outcome.usage() else {
                                return Err(lane_error(SessionError::Invariant(
                                    "Hook usage id exists without structural usage".to_owned(),
                                )));
                            };
                            let row = UsageWriteRow {
                                id: id.clone(),
                                usage: *usage,
                                entry_id: None,
                                adjustment: false,
                                details: None,
                            };
                            let write_index = writes.len();
                            writes.push(Write::Usage(insert_usage(row.clone())));
                            Some((row, write_index))
                        } else {
                            None
                        };

                    let mut terminal_tip_id = state.tip_id.clone();
                    let entry_plan: Option<(NewEntry, usize)> = match &outcome {
                        StructuralOutcome::Compaction {
                            result_entry_id,
                            result,
                            from_hook,
                        } => {
                            let entry = NewEntry::Compaction {
                                id: result_entry_id.clone(),
                                parent_id: state.tip_id.clone(),
                                body: CompactionEntryBody {
                                    summary: result.summary.clone(),
                                    retained_tail: result.retained_tail.clone(),
                                    tokens_before: result.tokens_before,
                                    details: result.details.clone(),
                                    usage: result.usage,
                                    from_hook: *from_hook,
                                },
                            };
                            let entry_write_index = writes.len();
                            writes.push(Write::Entry(Box::new(insert_entry(entry.clone()))));
                            writes.push(
                                set_value_write(
                                    &branch_tip(lane.name()),
                                    Some(result_entry_id.clone()),
                                )
                                .map_err(lane_error)?,
                            );
                            terminal_tip_id = Some(result_entry_id.clone());
                            Some((entry, entry_write_index))
                        }
                        StructuralOutcome::BranchSummary {
                            result_entry_id,
                            result,
                            from_hook,
                        } => {
                            let (target_id, label) = navigation_boundary(task)?;
                            let entry = NewEntry::BranchSummary {
                                id: result_entry_id.clone(),
                                parent_id: Some(target_id.clone()),
                                body: BranchSummaryEntryBody {
                                    from_id: operation.meta.source_tip_id.clone(),
                                    summary: result.summary.clone(),
                                    details: Some(
                                        serde_json::to_value(&CompactionDetails {
                                            read_files: result.read_files.clone(),
                                            modified_files: result.modified_files.clone(),
                                        })
                                        .unwrap_or_default(),
                                    ),
                                    usage: result.usage,
                                    from_hook: *from_hook,
                                },
                            };
                            // The redundant intermediate tip set still
                            // consumes a sequence; the write order is the
                            // wire order.
                            writes.push(
                                set_value_write(&branch_tip(lane.name()), Some(target_id.clone()))
                                    .map_err(lane_error)?,
                            );
                            let entry_write_index = writes.len();
                            writes.push(Write::Entry(Box::new(insert_entry(entry.clone()))));
                            writes.push(
                                set_value_write(
                                    &branch_tip(lane.name()),
                                    Some(result_entry_id.clone()),
                                )
                                .map_err(lane_error)?,
                            );
                            if let Some(label) = label {
                                writes.push(
                                    set_value_write(&entry_label(target_id), label.clone())
                                        .map_err(lane_error)?,
                                );
                            }
                            terminal_tip_id = Some(result_entry_id.clone());
                            Some((entry, entry_write_index))
                        }
                        StructuralOutcome::Declined | StructuralOutcome::Failed { .. } => None,
                    };

                    let retry_end = leaf
                        .attempt()
                        .filter(|attempt| *attempt > 1)
                        .map(|attempt| {
                            let success =
                                matches!(outcome.kind(), Some("compaction" | "branch_summary"));
                            let final_error = match &outcome {
                                StructuralOutcome::Failed { error } => Some(error.message.clone()),
                                _ => None,
                            };
                            (attempt, success, final_error)
                        });
                    let completed_end = match &outcome {
                        StructuralOutcome::Compaction {
                            result_entry_id, ..
                        } => Some((compaction_reason(task)?, result_entry_id.clone())),
                        _ => None,
                    };
                    let step = task.task_id.clone();

                    match &task.boundary {
                        ResultBoundary::ResumeCheckpoint { resume_after } => {
                            if matches!(outcome, StructuralOutcome::BranchSummary { .. }) {
                                return Err(lane_error(SessionError::Invariant(
                                    "Run compaction boundary received a branch summary".to_owned(),
                                )));
                            }
                            let compaction_or_declined_threshold =
                                matches!(outcome, StructuralOutcome::Compaction { .. })
                                    || (matches!(outcome, StructuralOutcome::Declined)
                                        && matches!(
                                            task.reason,
                                            Some(CompactionReason::Threshold)
                                        ));
                            if compaction_or_declined_threshold {
                                let Some(tip_id) = &terminal_tip_id else {
                                    return Err(lane_error(SessionError::Invariant(
                                        "Run compaction has no Branch tip".to_owned(),
                                    )));
                                };
                                let continuation = resume_after.continuation;
                                let placement = plan_boundary_inbox(
                                    &lane,
                                    &drive,
                                    &state,
                                    operation_scope_of(&operation.state),
                                    session.as_ref(),
                                    Some(tip_id.clone()),
                                    matches!(outcome, StructuralOutcome::Declined)
                                        && matches!(continuation, Continuation::MayFinish { .. }),
                                )
                                .await?;
                                if matches!(outcome, StructuralOutcome::Declined)
                                    && placement.trigger_entry_id.is_none()
                                    && matches!(continuation, Continuation::MayFinish { .. })
                                {
                                    return Ok(OperationCommand::Return {
                                        result: StructuralPublication::FinishPending(
                                            BoundaryFinishPending {
                                                entry_ids: placement
                                                    .entries
                                                    .iter()
                                                    .map(|entry| entry.id().to_owned())
                                                    .collect(),
                                            },
                                        ),
                                    });
                                }
                                let placement_write_index = writes.len();
                                writes.extend(placement.writes.clone());
                                let operation_state = if placement.trigger_entry_id.is_some()
                                    || matches!(continuation, Continuation::NeedAssistant { .. })
                                {
                                    let overflow_recovery_used = if placement
                                        .trigger_entry_id
                                        .is_none()
                                        && matches!(
                                            continuation,
                                            Continuation::NeedAssistant { .. }
                                        ) {
                                        match continuation {
                                            Continuation::NeedAssistant {
                                                overflow_recovery_used,
                                            } => overflow_recovery_used,
                                            Continuation::MayFinish { .. } => false,
                                        }
                                    } else {
                                        false
                                    };
                                    OperationState::AssistantReady(assistant_ready_at_boundary(
                                        &lane,
                                        &state,
                                        operation_scope_of(&operation.state),
                                        placement.trigger_entry_id.clone().unwrap_or_else(|| {
                                            resume_after.trigger_entry_id.clone()
                                        }),
                                        overflow_recovery_used,
                                    ))
                                } else {
                                    OperationState::Checkpoint(CheckpointOperation {
                                        scope: operation_scope_of(&operation.state),
                                        checkpoint: resume_after.clone(),
                                    })
                                };
                                let compaction_committed =
                                    matches!(outcome, StructuralOutcome::Compaction { .. });
                                let lane_name = lane.name().to_owned();
                                let run_id = drive.operation_id.clone();
                                let base = base_events_fn(
                                    BaseEventPlan {
                                        hook_usage_row,
                                        entry_plan,
                                        retry_end,
                                        completed_end,
                                        step,
                                    },
                                    None,
                                    lane_name.clone(),
                                    run_id.clone(),
                                );
                                let placement_for_events = placement.clone();
                                return Ok(OperationCommand::Commit {
                                    writes,
                                    operation_state,
                                    lane: Some(LanePatch {
                                        tip_id: Some(placement.tip_id.clone()),
                                        inbox: Some(placement.inbox),
                                        ..LanePatch::default()
                                    }),
                                    materialize: Arc::new(|_: &CommitResult| {
                                        StructuralPublication::Procedure(ProcedureResult::Continue)
                                    }),
                                    events: Some(Arc::new(move |commit: &CommitResult| {
                                        let mut events = if compaction_committed {
                                            base(commit)
                                        } else {
                                            vec![lane_scoped_event(
                                                &lane_name,
                                                false,
                                                "compaction_end",
                                                HarnessEventPayload::CompactionEnd {
                                                    run_id: run_id.clone(),
                                                    reason: CompactionReason::Threshold,
                                                    ended_at: commit.timestamp,
                                                    status: CompactionEndStatus::Declined,
                                                },
                                            )]
                                        };
                                        events.extend(boundary_placement_events(
                                            &placement_for_events,
                                            commit,
                                            placement_write_index,
                                            &lane_name,
                                            &run_id,
                                        ));
                                        events
                                    })),
                                });
                            }
                            if state.tip_id.is_none() {
                                return Err(lane_error(SessionError::Invariant(
                                    "Failed run has no Branch tip".to_owned(),
                                )));
                            }
                            let error = match &outcome {
                                StructuralOutcome::Declined => operation_error(
                                    "compaction_declined",
                                    "Overflow compaction was declined",
                                    None,
                                ),
                                StructuralOutcome::Failed { error } => error.clone(),
                                _ => unreachable!(
                                    "the failed arm takes failed or non-threshold declined outcomes"
                                ),
                            };
                            let reason = compaction_reason(task)?;
                            let cleanup = operation_cleanup_writes(
                                session.as_ref(),
                                &drive.operation_id,
                                &operation.state,
                                &drive.context,
                            )
                            .await?;
                            let record = operation_result_record(
                                &operation.meta,
                                TerminalStatus::Failed,
                                state.tip_id.clone(),
                                Some(error.clone()),
                            )?;
                            let compaction_end_status = match &outcome {
                                StructuralOutcome::Declined => CompactionEndStatus::Declined,
                                StructuralOutcome::Failed { .. } => CompactionEndStatus::Failed {
                                    error: error.clone(),
                                },
                                _ => unreachable!(
                                    "the failed arm takes failed or non-threshold declined outcomes"
                                ),
                            };
                            let ended_at = record.ended_at;
                            let from_tip_id = operation.meta.source_tip_id.clone();
                            let tip_id = state.tip_id.clone();
                            let lane_name = lane.name().to_owned();
                            let run_id = drive.operation_id.clone();
                            let base = base_events_fn(
                                BaseEventPlan {
                                    hook_usage_row,
                                    entry_plan,
                                    retry_end,
                                    completed_end,
                                    step,
                                },
                                None,
                                lane_name.clone(),
                                run_id.clone(),
                            );
                            Ok(OperationCommand::Finish {
                                writes: [writes, cleanup].concat(),
                                record: record.clone(),
                                lane: None,
                                materialize: Arc::new(move |_: &CommitResult| {
                                    StructuralPublication::Procedure(ProcedureResult::Settled {
                                        outcome: record.clone(),
                                    })
                                }),
                                events: Some(Arc::new(move |commit: &CommitResult| {
                                    let mut events = base(commit);
                                    events.push(lane_scoped_event(
                                        &lane_name,
                                        false,
                                        "compaction_end",
                                        HarnessEventPayload::CompactionEnd {
                                            run_id: run_id.clone(),
                                            reason,
                                            ended_at,
                                            status: compaction_end_status.clone(),
                                        },
                                    ));
                                    events.push(lane_scoped_event(
                                        &lane_name,
                                        false,
                                        "run_end",
                                        HarnessEventPayload::RunEnd {
                                            run_id: run_id.clone(),
                                            status: RunEndStatus::Failed {
                                                error: error.clone(),
                                            },
                                            from_tip_id: from_tip_id.clone(),
                                            tip_id: tip_id.clone(),
                                            ended_at,
                                        },
                                    ));
                                    events
                                })),
                            })
                        }
                        ResultBoundary::Finish => {
                            if matches!(outcome, StructuralOutcome::BranchSummary { .. }) {
                                return Err(lane_error(SessionError::Invariant(
                                    "Compaction finish boundary received a branch summary"
                                        .to_owned(),
                                )));
                            }
                            if !matches!(outcome, StructuralOutcome::Compaction { .. })
                                && state.tip_id.is_none()
                            {
                                return Err(lane_error(SessionError::Invariant(
                                    "Standalone compaction has no Branch tip".to_owned(),
                                )));
                            }
                            let error = match &outcome {
                                StructuralOutcome::Failed { error } => Some(error.clone()),
                                _ => None,
                            };
                            let status = match &outcome {
                                StructuralOutcome::Declined => TerminalStatus::Declined,
                                StructuralOutcome::Failed { .. } => TerminalStatus::Failed,
                                _ => TerminalStatus::Completed,
                            };
                            let cleanup = operation_cleanup_writes(
                                session.as_ref(),
                                &drive.operation_id,
                                &operation.state,
                                &drive.context,
                            )
                            .await?;
                            let record = operation_result_record(
                                &operation.meta,
                                status,
                                terminal_tip_id.clone(),
                                error,
                            )?;
                            // The completed compaction_end reads the
                            // record's end instant, upstream's
                            // `terminalCompactionEndedAt` assignment before
                            // the events materialize.
                            let terminal_compaction_ended_at = Some(record.ended_at);
                            let compaction_end = match &outcome {
                                StructuralOutcome::Compaction { .. } => None,
                                StructuralOutcome::Declined => {
                                    Some((CompactionEndStatus::Declined, record.ended_at))
                                }
                                StructuralOutcome::Failed { error } => Some((
                                    CompactionEndStatus::Failed {
                                        error: error.clone(),
                                    },
                                    record.ended_at,
                                )),
                                StructuralOutcome::BranchSummary { .. } => {
                                    unreachable!("the finish arm rejects branch summaries above")
                                }
                            };
                            let patches_tip =
                                matches!(outcome, StructuralOutcome::Compaction { .. });
                            let lane_name = lane.name().to_owned();
                            let run_id = drive.operation_id.clone();
                            let base = base_events_fn(
                                BaseEventPlan {
                                    hook_usage_row,
                                    entry_plan,
                                    retry_end,
                                    completed_end,
                                    step,
                                },
                                terminal_compaction_ended_at,
                                lane_name.clone(),
                                run_id.clone(),
                            );
                            Ok(OperationCommand::Finish {
                                writes: [writes, cleanup].concat(),
                                record: record.clone(),
                                lane: patches_tip.then(|| LanePatch {
                                    tip_id: Some(terminal_tip_id.clone()),
                                    ..LanePatch::default()
                                }),
                                materialize: Arc::new(move |_: &CommitResult| {
                                    StructuralPublication::Procedure(ProcedureResult::Settled {
                                        outcome: record.clone(),
                                    })
                                }),
                                events: Some(Arc::new(move |commit: &CommitResult| {
                                    let mut events = base(commit);
                                    if let Some((status, ended_at)) = &compaction_end {
                                        events.push(lane_scoped_event(
                                            &lane_name,
                                            false,
                                            "compaction_end",
                                            HarnessEventPayload::CompactionEnd {
                                                run_id: run_id.clone(),
                                                reason: CompactionReason::Manual,
                                                ended_at: *ended_at,
                                                status: status.clone(),
                                            },
                                        ));
                                    }
                                    events
                                })),
                            })
                        }
                        ResultBoundary::CommitNavigation { .. } => {
                            if matches!(outcome, StructuralOutcome::Compaction { .. }) {
                                return Err(lane_error(SessionError::Invariant(
                                    "Navigation boundary received a compaction result".to_owned(),
                                )));
                            }
                            let error = match &outcome {
                                StructuralOutcome::Failed { error } => Some(error.clone()),
                                _ => None,
                            };
                            let status = match &outcome {
                                StructuralOutcome::Declined => TerminalStatus::Declined,
                                StructuralOutcome::Failed { .. } => TerminalStatus::Failed,
                                _ => TerminalStatus::Completed,
                            };
                            let cleanup = operation_cleanup_writes(
                                session.as_ref(),
                                &drive.operation_id,
                                &operation.state,
                                &drive.context,
                            )
                            .await?;
                            let record = operation_result_record(
                                &operation.meta,
                                status,
                                terminal_tip_id.clone(),
                                error,
                            )?;
                            let ended_at = record.ended_at;
                            let from_tip_id = operation.meta.source_tip_id.clone();
                            let event_tip_id = terminal_tip_id.clone();
                            let navigation_end_status = match &outcome {
                                StructuralOutcome::BranchSummary { .. } => {
                                    NavigationEndStatus::Completed
                                }
                                StructuralOutcome::Declined => NavigationEndStatus::Declined,
                                StructuralOutcome::Failed { error } => {
                                    NavigationEndStatus::Failed {
                                        error: error.clone(),
                                    }
                                }
                                StructuralOutcome::Compaction { .. } => unreachable!(
                                    "the navigation arm rejects compaction results above"
                                ),
                            };
                            let ends_at_tip =
                                matches!(outcome, StructuralOutcome::BranchSummary { .. });
                            let lane_name = lane.name().to_owned();
                            let run_id = drive.operation_id.clone();
                            let base = base_events_fn(
                                BaseEventPlan {
                                    hook_usage_row,
                                    entry_plan,
                                    retry_end,
                                    completed_end,
                                    step,
                                },
                                None,
                                lane_name.clone(),
                                run_id.clone(),
                            );
                            Ok(OperationCommand::Finish {
                                writes: [writes, cleanup].concat(),
                                record: record.clone(),
                                lane: ends_at_tip.then(|| LanePatch {
                                    tip_id: Some(terminal_tip_id.clone()),
                                    ..LanePatch::default()
                                }),
                                materialize: Arc::new(move |_: &CommitResult| {
                                    StructuralPublication::Procedure(ProcedureResult::Settled {
                                        outcome: record.clone(),
                                    })
                                }),
                                events: Some(Arc::new(move |commit: &CommitResult| {
                                    let mut events = base(commit);
                                    events.push(lane_scoped_event(
                                        &lane_name,
                                        false,
                                        "navigation_end",
                                        HarnessEventPayload::NavigationEnd {
                                            run_id: run_id.clone(),
                                            status: navigation_end_status.clone(),
                                            from_tip_id: from_tip_id.clone(),
                                            tip_id: event_tip_id.clone(),
                                            ended_at,
                                        },
                                    ));
                                    events
                                })),
                            })
                        }
                    }
                })
            },
            &drive.context,
        )
        .await?;
    let ContinueOperationResult::Result { value: publication } = published else {
        return Ok(ProcedureResult::Continue);
    };
    let StructuralPublication::Procedure(result) = publication else {
        let StructuralPublication::FinishPending(pending) = publication else {
            unreachable!("the publication is either a procedure result or finish-pending")
        };
        let boundary = &capability.task().boundary;
        let ResultBoundary::ResumeCheckpoint { resume_after } = boundary else {
            return Err(lane_error(SessionError::Invariant(
                "Structural finish mediation requires a resumable finish boundary".to_owned(),
            )));
        };
        let Continuation::MayFinish { .. } = resume_after.continuation else {
            return Err(lane_error(SessionError::Invariant(
                "Structural finish mediation requires a resumable finish boundary".to_owned(),
            )));
        };
        let pending_event = lane_scoped_event(
            lane.name(),
            false,
            "compaction_end",
            HarnessEventPayload::CompactionEnd {
                run_id: drive.operation_id.clone(),
                reason: CompactionReason::Threshold,
                // Process-local: the mediation runs outside any commit.
                ended_at: now_ms(),
                status: CompactionEndStatus::Declined,
            },
        );
        return finish_run_boundary(
            lane,
            drive,
            &capability.state(),
            resume_after.continuation,
            &pending.entry_ids,
            vec![pending_event],
        )
        .await;
    };
    Ok(result)
}

/// Prepare threshold compaction only when no newer compaction already
/// guards this trigger, upstream's `prepareCompactionThreshold`.
///
/// The model reads the LIVE lane configuration, not the checkpoint's. The
/// newer-compaction guard runs before the missing-trigger check: a missing
/// trigger with a newer compaction returns `None`, without one it is an
/// invariant. A preparation that exists but does not cross the threshold
/// also returns `None`.
///
/// # Errors
/// The invariant `` `Checkpoint trigger {triggerEntryId} is missing from
/// its Branch` ``, the compaction preparation's error, and the bounded
/// entries read's storage error.
pub(crate) async fn prepare_compaction_threshold(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    checkpoint: &CheckpointOperation,
) -> Result<ContinueOperationResult<Option<ThresholdPreparation>>, LaneError> {
    let settings = checkpoint.scope.settings.compaction;
    let identity = lane.state().configuration.model.clone();
    let Some(model) = settings
        .enabled
        .then(|| lane.models().model(&identity.provider, &identity.model_id))
        .flatten()
    else {
        return Ok(ContinueOperationResult::Result { value: None });
    };
    let path = read_bounded_entries(lane, drive).await?;
    let ContinueOperationResult::Result { value: path } = path else {
        return Ok(ContinueOperationResult::CancelRequested);
    };
    let trigger_index = path
        .iter()
        .position(|entry| entry.id() == checkpoint.checkpoint.trigger_entry_id);
    let newest_compaction_index = path
        .iter()
        .rposition(|entry| matches!(entry, Entry::Compaction { .. }));
    let guarded = match (newest_compaction_index, trigger_index) {
        (Some(newest), Some(trigger)) => newest >= trigger,
        // A newer compaction guards the trigger and suppresses the
        // missing-trigger check.
        (Some(_), None) => true,
        (None, _) => false,
    };
    if guarded {
        return Ok(ContinueOperationResult::Result { value: None });
    }
    if trigger_index.is_none() {
        return Err(lane_error(SessionError::Invariant(format!(
            "Checkpoint trigger {} is missing from its Branch",
            checkpoint.checkpoint.trigger_entry_id
        ))));
    }
    let prepared = prepare_compaction(&path, settings).map_err(lane_error)?;
    let Some(prepared) = prepared else {
        return Ok(ContinueOperationResult::Result { value: None });
    };
    #[expect(
        clippy::cast_sign_loss,
        reason = "the token estimate is non-negative; the cast preserves shouldCompact's comparison"
    )]
    let context_tokens = prepared.tokens_before as u64;
    if !should_compact(context_tokens, model.context_window, &settings) {
        return Ok(ContinueOperationResult::Result { value: None });
    }
    Ok(ContinueOperationResult::Result {
        value: Some(ThresholdPreparation {
            task_id: lane.session().id_generator().next(None),
            preparation: durable_compaction_preparation(prepared),
        }),
    })
}

/// Prepare one overflow compaction before the response settlement
/// transaction, upstream's `prepareOverflowCompaction`. The overflow
/// recovery bound suppresses preparation (it is what compaction resumes
/// generation through), a cancelled bounded read silently skips the
/// compaction, and there is no threshold gate — an existing preparation
/// always compacts.
///
/// # Errors
/// The compaction preparation's error and the bounded entries read's
/// storage error.
pub(crate) async fn prepare_overflow_compaction(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    generation: &AssistantEffectPendingOperation,
) -> Result<Option<ThresholdPreparation>, LaneError> {
    if generation.generation_context.overflow_recovery_used {
        return Ok(None);
    }
    let path = read_bounded_entries(lane, drive).await?;
    let ContinueOperationResult::Result { value: path } = path else {
        return Ok(None);
    };
    let prepared =
        prepare_compaction(&path, generation.scope.settings.compaction).map_err(lane_error)?;
    let Some(prepared) = prepared else {
        return Ok(None);
    };
    Ok(Some(ThresholdPreparation {
        task_id: lane.session().id_generator().next(None),
        preparation: durable_compaction_preparation(prepared),
    }))
}

/// Atomically move an unsummarized navigation and finish its operation,
/// upstream's `commitNavigation`.
///
/// The target must exist and differ from the source tip (a null target is
/// the branch root), a root navigation cannot set a label, and the tip
/// move, the label write, and the operation cleanup commit together. The
/// settled record and the `navigation_end` event carry the nullable tip.
///
/// # Errors
/// The invariants `` `Navigation target {targetId} is missing` ``,
/// `` `Navigation target must differ from its source tip` ``, and
/// `` `Root navigation cannot set a label` ``, the cleanup's storage
/// errors, and the commit's storage and delivery errors.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single commitNavigation method"
)]
pub(crate) async fn commit_navigation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    _navigation: &NavigationReadyToCommitOperation,
) -> Result<ProcedureResult, LaneError> {
    let planner_lane = Arc::clone(lane);
    let planner_drive = Arc::clone(drive);
    let published = lane
        .continue_operation::<ProcedureResult, _>(
            move |state, session, context| {
                let lane = Arc::clone(&planner_lane);
                let drive = Arc::clone(&planner_drive);
                Box::pin(async move {
                    let Some(operation) = state.operation.as_ref() else {
                        unreachable!("continueOperation planner runs under an active operation")
                    };
                    let OperationState::NavigationReadyToCommit(current) = &operation.state else {
                        unreachable!(
                            "continueOperation planner runs under the dispatched navigation.ready_to_commit leaf"
                        )
                    };
                    if let Some(target_id) = &current.target_id {
                        let found = session
                            .get_entries(vec![target_id.clone()], &context)
                            .await
                            .map_err(lane_error)?;
                        if !found.contains_key(target_id) {
                            return Err(lane_error(SessionError::Invariant(format!(
                                "Navigation target {target_id} is missing"
                            ))));
                        }
                    }
                    if current.target_id == operation.meta.source_tip_id {
                        return Err(lane_error(SessionError::Invariant(
                            "Navigation target must differ from its source tip".to_owned(),
                        )));
                    }
                    if current.target_id.is_none() && current.label.is_some() {
                        return Err(lane_error(SessionError::Invariant(
                            "Root navigation cannot set a label".to_owned(),
                        )));
                    }
                    let mut writes = vec![
                        set_value_write(&branch_tip(lane.name()), current.target_id.clone())
                            .map_err(lane_error)?,
                    ];
                    if let (Some(label), Some(target_id)) = (&current.label, &current.target_id) {
                        writes.push(
                            set_value_write(&entry_label(target_id), label.clone())
                                .map_err(lane_error)?,
                        );
                    }
                    let cleanup = operation_cleanup_writes(
                        session.as_ref(),
                        &drive.operation_id,
                        &operation.state,
                        &drive.context,
                    )
                    .await?;
                    let record = operation_result_record(
                        &operation.meta,
                        TerminalStatus::Completed,
                        current.target_id.clone(),
                        None,
                    )?;
                    let ended_at = record.ended_at;
                    let from_tip_id = operation.meta.source_tip_id.clone();
                    let tip_id = current.target_id.clone();
                    let lane_name = lane.name().to_owned();
                    let run_id = drive.operation_id.clone();
                    Ok(OperationCommand::Finish {
                        writes: [writes, cleanup].concat(),
                        record: record.clone(),
                        lane: Some(LanePatch {
                            tip_id: Some(current.target_id.clone()),
                            ..LanePatch::default()
                        }),
                        materialize: Arc::new(move |_: &CommitResult| {
                            ProcedureResult::Settled { outcome: record.clone() }
                        }),
                        events: Some(Arc::new(move |_: &CommitResult| {
                            vec![lane_scoped_event(
                                &lane_name,
                                false,
                                "navigation_end",
                                HarnessEventPayload::NavigationEnd {
                                    run_id: run_id.clone(),
                                    status: NavigationEndStatus::Completed,
                                    from_tip_id: from_tip_id.clone(),
                                    tip_id: tip_id.clone(),
                                    ended_at,
                                },
                            )]
                        })),
                    })
                })
            },
            &drive.context,
        )
        .await?;
    Ok(match published {
        ContinueOperationResult::CancelRequested => ProcedureResult::Continue,
        ContinueOperationResult::Result { value } => value,
    })
}

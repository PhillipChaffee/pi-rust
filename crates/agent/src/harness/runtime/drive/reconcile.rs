//! The cancelled-operation reconciliation, ported from upstream
//! `src/harness/runtime/drive/reconcile.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! `reconcile_operation` advances one cancelled durable leaf without
//! starting new ordinary work: it arms the pass's abort gate, then routes
//! by leaf — effect-pending leaves recover synthetically, tool batches run
//! their cancellation-only pass, deferred leaves cancel their provider
//! handle best-effort, and the remaining leaves publish the aborted
//! terminal record.
//!
//! Porting restatements: upstream's `beginAbort(Promise.resolve())`
//! restates as a pre-settled cancellation receiver (its sender dropped, so
//! every waiter resolves immediately), and upstream's
//! `signal: drive.closeSignal` rides a link-task cancellation token because
//! pi-ai's transport seam carries a token, not the chord signal.

use std::sync::Arc;

use pi_ai::models::ModelsDeferredCancelOptions;
use pi_ai::types::DeferredHandle;
use pi_ai::types::ProviderRequestOptions;
use pi_ai::types::TransportOptions;
use tokio_util::sync::CancellationToken;

use crate::harness::agent_harness::CompactionEndStatus;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::NavigationEndStatus;
use crate::harness::agent_harness::RunEndStatus;
use crate::harness::agent_harness::lane_scoped_event;
use crate::harness::context::AbortSignal;
use crate::harness::context::get_telemetry_context;
use crate::harness::runtime::drive::deferred::DeferredLeaf;
use crate::harness::runtime::drive::deferred::read_deferred_source_handle;
use crate::harness::runtime::drive::recovery::recover_cancelled_assistant_effect;
use crate::harness::runtime::drive::terminal::operation_cleanup_writes;
use crate::harness::runtime::drive::terminal::operation_result_record;
use crate::harness::runtime::drive::tools::run_tools;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::runtime::types::ProcedureResult;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::Control;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SummaryTask;
use crate::harness::session::types::TerminalStatus;
use crate::harness::session::types::operation_scope_of;

/// Links the pass's close signal to a request cancellation token — the seam
/// `request_signal` provides for contexts, restated here for the bare
/// controller signal. The linked token cancels when the signal aborts, and
/// the link task exits when either side settles.
fn close_request_signal(signal: &AbortSignal) -> CancellationToken {
    let token = CancellationToken::new();
    let linked = token.clone();
    let signal = signal.clone();
    tokio::spawn(async move {
        tokio::select! {
            _ = signal.wait() => linked.cancel(),
            () = linked.cancelled() => {},
        }
    });
    token
}

/// Cancels one deferred handle at its provider, best-effort, upstream's
/// `cancelDeferredBestEffort`: the model resolves from the leaf's captured
/// configuration — a missing model returns silently — the options carry the
/// close-linked signal, the telemetry parent, and the captured
/// timeout/retry/headers fields, and every failure is swallowed.
///
/// Remote cancellation is best-effort; durable local reconciliation must
/// continue.
async fn cancel_deferred_best_effort(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    deferred: &DeferredLeaf,
    handle: &DeferredHandle,
) {
    let scope = match deferred {
        DeferredLeaf::Suspended(leaf) => &leaf.deferred,
        DeferredLeaf::EffectPending(leaf) => &leaf.scope,
    };
    let identity = &scope.configuration.model;
    let Some(model) = lane.models().model(&identity.provider, &identity.model_id) else {
        return;
    };
    let options = ModelsDeferredCancelOptions {
        options: ProviderRequestOptions {
            transport_options: TransportOptions {
                signal: Some(close_request_signal(&drive.close_signal)),
                ..TransportOptions::default()
            },
            telemetry_context: Some(get_telemetry_context(&drive.context)),
            timeout_ms: scope.stream_options.timeout_ms,
            max_retries: scope.stream_options.max_retries,
            max_retry_delay_ms: scope.stream_options.max_retry_delay_ms,
            headers: scope.stream_options.headers.clone().map(|headers| {
                headers
                    .into_iter()
                    .map(|(key, value)| (key, Some(value)))
                    .collect()
            }),
            ..ProviderRequestOptions::default()
        },
        transform_headers: None,
    };
    let models = Arc::clone(lane.models());
    let _ = models.cancel_deferred(&model, handle, Some(&options)).await;
}

/// Reads one deferred leaf's source handle through the lane's mutation
/// line, upstream's `readDeferredHandle`: a `settleOperation` whose plan
/// returns the handle [`read_deferred_source_handle`] validates.
///
/// # Errors
/// [`read_deferred_source_handle`]'s invariants and storage errors, and the
/// settle's no-active-operation invariant.
async fn read_deferred_handle(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    deferred: &DeferredLeaf,
) -> Result<DeferredHandle, LaneError> {
    let planner_drive = Arc::clone(drive);
    let planner_deferred = deferred.clone();
    lane.settle_operation::<DeferredHandle, _>(
        move |_state, _current, _meta, reader, _context| {
            let drive = Arc::clone(&planner_drive);
            let deferred = planner_deferred.clone();
            Box::pin(async move {
                let handle =
                    read_deferred_source_handle(reader.as_ref(), &deferred, &drive.context).await?;
                Ok(OperationCommand::Return { result: handle })
            })
        },
        &drive.context,
    )
    .await
}

/// The summary task a run-intent summary leaf carries, upstream's
/// `current.task` over the four summary leaves; the other leaves carry
/// none.
const fn summary_task_of(current: &OperationState) -> Option<&SummaryTask> {
    match current {
        OperationState::SummaryDeciding(leaf) => Some(&leaf.task),
        OperationState::SummaryReady(leaf) => Some(&leaf.generation.task),
        OperationState::SummaryEffectPending(leaf) => Some(&leaf.generation.task),
        OperationState::SummaryRetryWait(leaf) => Some(&leaf.generation.task),
        _ => None,
    }
}

/// Publishes one cancelled leaf's terminal record, upstream's
/// `publishAbortedTerminal`: a `settleOperation` — no cancel check, the
/// leaf IS the cancelled one — that requires cancelled durable control,
/// records the aborted result at the lane tip, cleans the operation's
/// stored values, and publishes the intent's terminal events. The run
/// intent's four summary leaves additionally require a `resume_checkpoint`
/// boundary with a reason and publish their `compaction_end` before the
/// run's `run_end`; the compaction intent publishes `compaction_end` under
/// the manual reason and the navigation intent publishes `navigation_end`,
/// both aborted.
///
/// # Errors
/// The invariants `` `Cancellation reconciliation requires cancelled
/// durable control` `` and `` `Cancelled run summary has an invalid result
/// boundary` ``, the cleanup's storage errors, and the commit's storage and
/// delivery errors.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single publishAbortedTerminal method and its per-intent event matrix"
)]
async fn publish_aborted_terminal(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    _capability: &OperationState,
) -> Result<ProcedureResult, LaneError> {
    let planner_lane = Arc::clone(lane);
    let planner_drive = Arc::clone(drive);
    lane.settle_operation::<ProcedureResult, _>(
        move |state, _current, _meta, session, _context| {
            let lane = Arc::clone(&planner_lane);
            let drive = Arc::clone(&planner_drive);
            Box::pin(async move {
                let Some(operation) = state.operation.as_ref() else {
                    unreachable!("settleOperation planner runs under an active operation")
                };
                let current = &operation.state;
                let meta = &operation.meta;
                if !matches!(
                    operation_scope_of(current).control,
                    Control::CancelRequested { .. }
                ) {
                    return Err(lane_error(SessionError::Invariant(
                        "Cancellation reconciliation requires cancelled durable control".to_owned(),
                    )));
                }
                let record = operation_result_record(
                    meta,
                    TerminalStatus::Aborted,
                    state.tip_id.clone(),
                    None,
                )?;
                let cleanup = operation_cleanup_writes(
                    session.as_ref(),
                    &drive.operation_id,
                    current,
                    &drive.context,
                )
                .await?;

                let mut events: Vec<HarnessEvent> = Vec::new();
                let lane_name = lane.name().to_owned();
                let run_id = drive.operation_id.clone();
                let ended_at = record.ended_at;
                match &meta.intent {
                    OperationIntent::Run { .. } => {
                        if let Some(task) = summary_task_of(current) {
                            let (ResultBoundary::ResumeCheckpoint { .. }, Some(reason)) =
                                (&task.boundary, task.reason)
                            else {
                                return Err(lane_error(SessionError::Invariant(
                                    "Cancelled run summary has an invalid result boundary"
                                        .to_owned(),
                                )));
                            };
                            events.push(lane_scoped_event(
                                &lane_name,
                                false,
                                "compaction_end",
                                HarnessEventPayload::CompactionEnd {
                                    run_id: run_id.clone(),
                                    reason,
                                    ended_at,
                                    status: CompactionEndStatus::Aborted,
                                },
                            ));
                        }
                        events.push(lane_scoped_event(
                            &lane_name,
                            false,
                            "run_end",
                            HarnessEventPayload::RunEnd {
                                run_id,
                                status: RunEndStatus::Aborted,
                                from_tip_id: meta.source_tip_id.clone(),
                                tip_id: state.tip_id.clone(),
                                ended_at,
                            },
                        ));
                    }
                    OperationIntent::Compaction { .. } => {
                        events.push(lane_scoped_event(
                            &lane_name,
                            false,
                            "compaction_end",
                            HarnessEventPayload::CompactionEnd {
                                run_id,
                                reason: CompactionReason::Manual,
                                ended_at,
                                status: CompactionEndStatus::Aborted,
                            },
                        ));
                    }
                    OperationIntent::Navigation { .. } => {
                        events.push(lane_scoped_event(
                            &lane_name,
                            false,
                            "navigation_end",
                            HarnessEventPayload::NavigationEnd {
                                run_id,
                                status: NavigationEndStatus::Aborted,
                                from_tip_id: meta.source_tip_id.clone(),
                                tip_id: state.tip_id.clone(),
                                ended_at,
                            },
                        ));
                    }
                }
                Ok(OperationCommand::Finish {
                    writes: cleanup,
                    record: record.clone(),
                    lane: None,
                    materialize: Arc::new(move |_: &CommitResult| ProcedureResult::Settled {
                        outcome: record.clone(),
                    }),
                    events: Some(Arc::new(move |_: &CommitResult| events.clone())),
                })
            })
        },
        &drive.context,
    )
    .await
}

/// Advance one cancelled durable leaf without starting new ordinary work,
/// upstream's `reconcileOperation`.
///
/// The pass arms its abort gate before dispatch: [`Drive::begin_abort`]
/// records a pre-settled cancellation (upstream's `Promise.resolve()`) and
/// [`Drive::signal_abort`] fires the gate, so admissions racing the
/// settle refuse immediately. The leaf dispatch routes
/// `assistant.effect_pending` and `deferred.effect_pending` to the
/// synthetic recovery, `tools` to its cancellation-only pass, and the
/// deferred-suspended leaf through the best-effort remote cancellation
/// before its aborted terminal; every remaining leaf publishes the aborted
/// terminal directly.
///
/// # Errors
/// The invariants `` `Drive {operationId} has no matching operation to
/// reconcile` `` — no live operation or a foreign one — and
/// `` `Operation {operationId} is not cancelled` ``, the leaf procedures'
/// and [`publish_aborted_terminal`]'s invariants, and their storage and
/// delivery errors.
pub(crate) async fn reconcile_operation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
) -> Result<ProcedureResult, LaneError> {
    let operation = match lane.state().operation {
        Some(operation) if operation.meta.operation_id == drive.operation_id => operation,
        _ => {
            return Err(lane_error(SessionError::Invariant(format!(
                "Drive {} has no matching operation to reconcile",
                drive.operation_id
            ))));
        }
    };
    if !matches!(
        operation_scope_of(&operation.state).control,
        Control::CancelRequested { .. }
    ) {
        return Err(lane_error(SessionError::Invariant(format!(
            "Operation {} is not cancelled",
            drive.operation_id
        ))));
    }

    // The settled cancellation, upstream's `beginAbort(Promise.resolve())`:
    // dropping the sender resolves the receiver immediately.
    let (sender, cancellation) = tokio::sync::watch::channel(());
    drop(sender);
    drive.begin_abort(cancellation);
    drive.signal_abort();

    match &operation.state {
        OperationState::AssistantEffectPending(_) => {
            recover_cancelled_assistant_effect(lane, drive, &operation.state).await
        }
        OperationState::Tools(run) => run_tools(lane, drive, run).await,
        OperationState::DeferredSuspended(leaf) => {
            let deferred = DeferredLeaf::Suspended(leaf.clone());
            let handle = read_deferred_handle(lane, drive, &deferred).await?;
            cancel_deferred_best_effort(lane, drive, &deferred, &handle).await;
            publish_aborted_terminal(lane, drive, &operation.state).await
        }
        OperationState::DeferredEffectPending(leaf) => {
            let deferred = DeferredLeaf::EffectPending(leaf.clone());
            let handle = read_deferred_handle(lane, drive, &deferred).await?;
            cancel_deferred_best_effort(lane, drive, &deferred, &handle).await;
            recover_cancelled_assistant_effect(lane, drive, &operation.state).await
        }
        OperationState::Starting(_)
        | OperationState::Checkpoint(_)
        | OperationState::AssistantReady(_)
        | OperationState::AssistantRetryWait(_)
        | OperationState::SummaryDeciding(_)
        | OperationState::SummaryReady(_)
        | OperationState::SummaryEffectPending(_)
        | OperationState::SummaryRetryWait(_)
        | OperationState::NavigationReadyToCommit(_) => {
            publish_aborted_terminal(lane, drive, &operation.state).await
        }
    }
}

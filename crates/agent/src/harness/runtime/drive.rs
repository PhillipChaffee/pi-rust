//! The drive dispatch spine, ported from upstream
//! `src/harness/runtime/drive.ts`.
//!
//! One installed pass advances through direct durable procedures: the loop
//! re-reads the lane's live operation each iteration, routes
//! `cancel_requested` control to reconciliation before the ordinary
//! dispatch, and otherwise dispatches on the state leaf's `at` wire
//! discriminator — all thirteen leaves covered, with `assistant.ready` and
//! `assistant.retry_wait` sharing the generation procedure and the two
//! deferred leaves sharing theirs. A `continue` result must replace the
//! lane's operation projection, or land on `cancel_requested` control; an
//! unchanged re-read is the no-progress invariant. The procedure modules
//! below carry the leaves' bodies; the ones still pending their porting
//! waves raise the slice's `SliceNotImplemented` error, which faults the
//! pass through the
//! same error path upstream's staging device uses.

use std::sync::Arc;

use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::HookEvent;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::gate::AbortRequested;
use crate::harness::gate::Cancellation;
use crate::harness::gate::GateRejection;
use crate::harness::hooks::HookRunError;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::lane::intent_kind_of;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::LiveOperation;
use crate::harness::runtime::types::ProcedureResult;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::types::Control;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::operation_scope_of;

/// Converts one admission refusal to the lane error the spine's abort catch
/// recognizes, upstream's `gate.admit` throwing the carried
/// `AbortRequested` or the gate's closed error.
pub(crate) fn gate_rejection_error(rejection: GateRejection) -> LaneError {
    match rejection {
        GateRejection::AbortRequested(abort) => lane_error(abort),
        GateRejection::Closed(error) => lane_error(error),
    }
}

/// Converts one hook-run rejection to the error the spine's abort catch
/// recognizes, upstream's `runWithGate` throwing `AbortRequested` — the
/// admission refusal, or the already-aborted admitted context's reason,
/// whose cancellation the drive's begun abort carries (chord models abort
/// reasons as strings, so the carrier rides [`Drive::abort_cancellation`]).
pub(crate) fn hook_error_to_lane_error(error: HookRunError, drive: &Drive) -> LaneError {
    match error {
        HookRunError::GateAborted(abort) => lane_error(abort),
        HookRunError::Aborted(_) => lane_error(AbortRequested {
            cancellation: drive.abort_cancellation(),
        }),
        error => lane_error(error),
    }
}

/// Boxes one concrete error as the hook channels' erased error type,
/// upstream's thrown error object crossing the `Promise` rejection.
pub(crate) fn hook_error_box(
    error: impl std::error::Error + Send + Sync + 'static,
) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(error)
}

pub mod boundary;
pub mod checkpoint;
pub mod deferred;
pub mod generation;
pub mod reconcile;
pub mod recovery;
pub mod response;
pub mod retry;
pub mod structural;
pub mod terminal;
pub mod tool_placement;
pub mod tools;

/// Reads the lane's live operation for the pass, upstream's
/// `currentOperation`.
///
/// # Errors
/// The [`SessionError::Invariant`] `` `Drive {operationId} has no matching
/// current operation` `` when no operation is live or its id differs from
/// the pass's.
fn current_operation(lane: &Lane, drive: &Drive) -> Result<LiveOperation, LaneError> {
    match lane.state().operation {
        Some(operation) if operation.meta.operation_id == drive.operation_id => Ok(operation),
        _ => Err(lane_error(SessionError::Invariant(format!(
            "Drive {} has no matching current operation",
            drive.operation_id
        )))),
    }
}

/// Awaits one gate abort's cancellation to settle, upstream's
/// `await error.cancellation`.
pub(crate) async fn await_abort_cancellation(cancellation: &Cancellation) {
    let mut cancellation = cancellation.clone();
    let _ = cancellation.changed().await;
}

/// Drive one installed pass through direct durable procedures until
/// settlement or a durable wait, upstream's `driveOperation`.
///
/// The `before_drive` hook runs once, only while the live operation's
/// control is `running` and before the loop's `cancel_requested` check; a
/// gate abort during it awaits the abort's cancellation and falls through
/// to the loop, while any other hook failure faults the pass without
/// touching durable state. A `cancel_requested` control routes to
/// reconciliation ahead of the leaf switch, on every iteration.
///
/// # Errors
/// The mismatch invariant when no live operation matches the pass, any
/// procedure's propagated error (an abort-carried error instead awaits its
/// cancellation and continues the loop), and the no-progress invariant
/// `` `Drive procedure made no progress from {at}` `` when a `continue`
/// re-read the same operation under live control.
#[expect(
    clippy::too_many_lines,
    reason = "the leaf dispatch restates upstream's single switch arm by arm; compressing it would hide the leaf-to-procedure mapping"
)]
pub async fn drive_operation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
) -> Result<DriveOutcome, LaneError> {
    let operation = current_operation(lane, drive)?;
    if operation_scope_of(&operation.state).control == Control::Running {
        match lane
            .hooks()
            .run_with_gate(
                HookName::BeforeDrive,
                HookInvocation {
                    lane: lane.name().to_owned(),
                    run_id: drive.operation_id.clone(),
                    event: HookEvent::BeforeDrive {
                        operation: intent_kind_of(&operation.meta),
                    },
                },
                &drive.gate,
                &drive.context,
            )
            .await
        {
            Err(HookRunError::GateAborted(abort)) => {
                await_abort_cancellation(&abort.cancellation).await;
            }
            Err(error) => return Err(lane_error(error)),
            Ok(_) => {}
        }
    }

    loop {
        let operation = current_operation(lane, drive)?;
        let state = operation.state.clone();
        let dispatched: Result<ProcedureResult, LaneError> = if matches!(
            operation_scope_of(&state).control,
            Control::CancelRequested { .. }
        ) {
            // The cancel check precedes the leaf switch, upstream's
            // dispatch order; an aborted reconciliation is still the
            // abort-continue path.
            reconcile::reconcile_operation(lane, drive).await
        } else {
            match &state {
                OperationState::Starting(run) => checkpoint::start_run(lane, drive, run).await,
                OperationState::Checkpoint(run) => {
                    checkpoint::run_checkpoint(lane, drive, run).await
                }
                OperationState::AssistantReady(leaf) => {
                    generation::run_generation(
                        lane,
                        drive,
                        generation::GenerationLeaf::Ready(leaf.clone()),
                    )
                    .await
                }
                OperationState::AssistantRetryWait(leaf) => {
                    generation::run_generation(
                        lane,
                        drive,
                        generation::GenerationLeaf::RetryWait(leaf.clone()),
                    )
                    .await
                }
                OperationState::AssistantEffectPending(leaf) => {
                    recovery::recover_assistant_generation(lane, drive, leaf).await
                }
                OperationState::Tools(run) => tools::run_tools(lane, drive, run).await,
                OperationState::DeferredSuspended(leaf) => {
                    deferred::run_deferred(
                        lane,
                        drive,
                        deferred::DeferredLeaf::Suspended(leaf.clone()),
                    )
                    .await
                }
                OperationState::DeferredEffectPending(leaf) => {
                    deferred::run_deferred(
                        lane,
                        drive,
                        deferred::DeferredLeaf::EffectPending(leaf.clone()),
                    )
                    .await
                }
                OperationState::SummaryDeciding(leaf) => {
                    structural::run_structural_decision(lane, drive, leaf).await
                }
                OperationState::SummaryReady(leaf) => {
                    structural::run_structural_generation(lane, drive, leaf).await
                }
                OperationState::SummaryEffectPending(leaf) => {
                    structural::recover_structural_generation(lane, drive, leaf).await
                }
                OperationState::SummaryRetryWait(leaf) => {
                    structural::run_structural_retry_wait(lane, drive, leaf).await
                }
                OperationState::NavigationReadyToCommit(leaf) => {
                    structural::commit_navigation(lane, drive, leaf).await
                }
            }
        };
        let result: ProcedureResult = match dispatched {
            Ok(result) => result,
            Err(error) => {
                // Upstream catches the thrown `AbortRequested` only; every
                // other error propagates to the pass's failure.
                match error.downcast_ref::<AbortRequested>() {
                    Some(abort) => {
                        await_abort_cancellation(&abort.cancellation).await;
                        ProcedureResult::Continue
                    }
                    None => return Err(error),
                }
            }
        };

        match result {
            ProcedureResult::Settled { outcome } => {
                return Ok(DriveOutcome::Settled { outcome });
            }
            ProcedureResult::Waiting { outcome } => return Ok(outcome),
            ProcedureResult::Continue => {}
        }
        let next = current_operation(lane, drive)?;
        // The progress predicate is value-based where upstream compares the
        // state object's reference identity: a `continue` must replace the
        // lane's operation projection, or the re-read control must be
        // `cancel_requested`, which the next iteration routes to
        // reconciliation.
        if next == operation
            && !matches!(
                operation_scope_of(&next.state).control,
                Control::CancelRequested { .. }
            )
        {
            return Err(lane_error(SessionError::Invariant(format!(
                "Drive procedure made no progress from {}",
                state.at()
            ))));
        }
    }
}

#[cfg(test)]
mod tests;

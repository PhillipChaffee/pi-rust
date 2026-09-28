//! The run-start and checkpoint procedures, ported from upstream
//! `src/harness/runtime/drive/checkpoint.ts`.
//!
//! `start_run` consumes the `before_run` hook and commits the initial
//! checkpoint from the run's prompt entries and injected messages;
//! `run_checkpoint` advances one durable run boundary with at most one
//! commit, mediating renewed boundary work, threshold compaction, and the
//! finish decision through the boundary module.

use super::hook_error_to_lane_error;
use std::sync::Arc;

use pi_ai::types::Message;
use pi_ai::types::StopReason;

use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HookEvent;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::agent_harness::HookResult;
use crate::harness::agent_harness::lane_scoped_event;
use crate::harness::runtime::drive::boundary::BoundaryFinishPending;
use crate::harness::runtime::drive::boundary::assistant_ready_at_boundary;
use crate::harness::runtime::drive::boundary::boundary_placement_events;
use crate::harness::runtime::drive::boundary::finish_run_boundary;
use crate::harness::runtime::drive::boundary::plan_boundary_inbox;
use crate::harness::runtime::drive::structural::prepare_compaction_threshold;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::transcript::chain_entries;
use crate::harness::runtime::transcript::committed_entry_events;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::LanePatch;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::runtime::types::ProcedureResult;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::commit::insert_entry;
use crate::harness::session::types::CheckpointData;
use crate::harness::session::types::CheckpointOperation;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::Entry;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::StartingOperation;
use crate::harness::session::types::SummaryDecidingOperation;
use crate::harness::session::types::SummaryTask;
use crate::harness::session::types::operation_scope_of;
use crate::harness::session::values::Write;
use crate::harness::session::values::branch_tip;
use crate::harness::session::values::operation_preparation;
use crate::harness::session::values::set_value_write;
use crate::types::AgentMessage;

/// What one checkpoint boundary's planner produced, upstream's
/// `ProcedureResult | BoundaryFinishPending` union in `runCheckpoint` — the
/// finish-pending half leaves the transaction without a commit so the
/// caller mediates the finish through [`finish_run_boundary`].
enum CheckpointPublication {
    /// A plain procedure result.
    Procedure(ProcedureResult),
    /// The finish mediation's planned entries.
    FinishPending(BoundaryFinishPending),
}

/// Consume `before_run` and commit the initial checkpoint, upstream's
/// `startRun`.
///
/// The first planner replays the run intent's prompt entries; the hook may
/// inject messages ahead of them, each under a freshly reserved entry id;
/// the second planner chains prompt and injected messages after the branch
/// tip, records the checkpoint's `need_assistant` continuation, and moves
/// the lane tip to the last chained entry.
///
/// # Errors
/// The invariants `` `Run operation has non-run intent` ``, `` `Run prompt
/// entry {id} is missing its message` ``, `` `before_run returned a pending
/// assistant message` ``, and `` `Run start has no trigger entry` ``, plus
/// the prompt read's, hook run's, and commit's storage and delivery errors.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single startRun method"
)]
pub(crate) async fn start_run(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    _run: &StartingOperation,
) -> Result<ProcedureResult, LaneError> {
    let prompt = lane
        .continue_operation::<Vec<AgentMessage>, _>(
            |state, session, context| {
                Box::pin(async move {
                    let Some(current) = state.operation.as_ref() else {
                        unreachable!("continueOperation planner runs under an active operation")
                    };
                    let OperationIntent::Run { prompt_entry_ids } = &current.meta.intent else {
                        return Err(lane_error(SessionError::Invariant(
                            "Run operation has non-run intent".to_owned(),
                        )));
                    };
                    let entries = session
                        .get_entries(prompt_entry_ids.clone(), &context)
                        .await
                        .map_err(lane_error)?;
                    let mut messages = Vec::with_capacity(prompt_entry_ids.len());
                    for id in prompt_entry_ids {
                        let Some(Entry::Message { body, .. }) = entries.get(id) else {
                            return Err(lane_error(SessionError::Invariant(format!(
                                "Run prompt entry {id} is missing its message"
                            ))));
                        };
                        messages.push(body.message.clone());
                    }
                    Ok(OperationCommand::Return { result: messages })
                })
            },
            &drive.context,
        )
        .await?;
    let ContinueOperationResult::Result { value: prompt } = prompt else {
        return Ok(ProcedureResult::Continue);
    };

    let hook = lane
        .hooks()
        .run_with_gate(
            HookName::BeforeRun,
            HookInvocation {
                lane: lane.name().to_owned(),
                run_id: drive.operation_id.clone(),
                event: HookEvent::BeforeRun {
                    prompt,
                    resources: lane.read_config().resources,
                },
            },
            &drive.gate,
            &drive.context,
        )
        .await;
    let hook = match hook {
        Ok(hook) => hook,
        Err(error) => return Err(hook_error_to_lane_error(error, drive)),
    };
    let HookResult::BeforeRun(injected) = hook else {
        unreachable!("before_run returns its own result variant")
    };
    let injected = injected.map_or_else(Vec::new, |result| result.messages);
    for message in &injected {
        if let AgentMessage::Standard(Message::Assistant(assistant)) = message
            && assistant.stop_reason == StopReason::Pending
        {
            return Err(lane_error(SessionError::Invariant(
                "before_run returned a pending assistant message".to_owned(),
            )));
        }
    }
    let reserved: Vec<(String, AgentMessage)> = injected
        .into_iter()
        .map(|message| (lane.session().id_generator().next(None), message))
        .collect();

    let planner_lane = Arc::clone(lane);
    let planner_drive = Arc::clone(drive);
    let result = lane
        .continue_operation::<ProcedureResult, _>(
            move |state, _session, _context| {
                let lane = Arc::clone(&planner_lane);
                let drive = Arc::clone(&planner_drive);
                let reserved = reserved.clone();
                Box::pin(async move {
                    let Some(current) = state.operation.as_ref() else {
                        unreachable!("continueOperation planner runs under an active operation")
                    };
                    let entries = chain_entries(
                        state.tip_id.clone(),
                        reserved
                            .into_iter()
                            .map(|(id, message)| NewEntry::Message {
                                id,
                                parent_id: None,
                                body: Box::new(MessageEntry {
                                    message,
                                    terminate: None,
                                }),
                            })
                            .collect(),
                    );
                    // `entries.at(-1)?.id ?? state.tipId`: the None check
                    // precedes the collapse — no entries and no tip is the
                    // invariant, a tip alone is the trigger.
                    let trigger_entry_id = entries
                        .last()
                        .map_or_else(|| state.tip_id.clone(), |entry| Some(entry.id().to_owned()));
                    let Some(trigger_entry_id) = trigger_entry_id else {
                        return Err(lane_error(SessionError::Invariant(
                            "Run start has no trigger entry".to_owned(),
                        )));
                    };
                    let mut writes: Vec<Write> = entries
                        .iter()
                        .map(|entry| Write::Entry(Box::new(insert_entry(entry.clone()))))
                        .collect();
                    if !entries.is_empty() {
                        writes.push(
                            set_value_write(
                                &branch_tip(lane.name()),
                                Some(trigger_entry_id.clone()),
                            )
                            .map_err(lane_error)?,
                        );
                    }
                    Ok(OperationCommand::Commit {
                        writes,
                        operation_state: OperationState::Checkpoint(CheckpointOperation {
                            scope: operation_scope_of(&current.state),
                            checkpoint: CheckpointData {
                                continuation: Continuation::NeedAssistant {
                                    overflow_recovery_used: false,
                                },
                                trigger_entry_id: trigger_entry_id.clone(),
                            },
                        }),
                        lane: Some(LanePatch {
                            tip_id: Some(Some(trigger_entry_id)),
                            ..LanePatch::default()
                        }),
                        materialize: Arc::new(|_: &CommitResult| ProcedureResult::Continue),
                        events: Some(Arc::new({
                            let lane_name = lane.name().to_owned();
                            let run_id = drive.operation_id.clone();
                            move |commit: &CommitResult| {
                                committed_entry_events(
                                    &entries,
                                    commit,
                                    &lane_name,
                                    Some(&run_id),
                                    0,
                                )
                            }
                        })),
                    })
                })
            },
            &drive.context,
        )
        .await?;
    Ok(match result {
        ContinueOperationResult::CancelRequested => ProcedureResult::Continue,
        ContinueOperationResult::Result { value } => value,
    })
}

/// Advance one durable run boundary with at most one commit, upstream's
/// `runCheckpoint`.
///
/// The planner is four-way: a placed trigger commits the renewed
/// assistant-ready work; a threshold preparation commits the
/// `summary.deciding` task with its preparation write and the
/// `compaction_start` event; a `need_assistant` continuation re-commits the
/// assistant-ready work under the checkpoint's own trigger and overflow
/// flag; otherwise the placed entry ids hand the finish mediation to
/// [`finish_run_boundary`].
///
/// # Errors
/// The invariant `` `Checkpoint finish mediation requires a finish
/// continuation` ``, the boundary planning's invariants, and the commit's
/// storage and delivery errors.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single runCheckpoint method"
)]
pub(crate) async fn run_checkpoint(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    run: &CheckpointOperation,
) -> Result<ProcedureResult, LaneError> {
    let threshold = prepare_compaction_threshold(lane, drive, run).await?;
    let ContinueOperationResult::Result { value: threshold } = threshold else {
        return Ok(ProcedureResult::Continue);
    };

    let planner_lane = Arc::clone(lane);
    let planner_drive = Arc::clone(drive);
    let planned = lane
        .continue_operation::<CheckpointPublication, _>(
            move |state, session, _context| {
                let lane = Arc::clone(&planner_lane);
                let drive = Arc::clone(&planner_drive);
                let threshold = threshold.clone();
                Box::pin(async move {
                    let Some(current) = state.operation.as_ref() else {
                        unreachable!("continueOperation planner runs under an active operation")
                    };
                    let OperationState::Checkpoint(checkpoint) = &current.state else {
                        unreachable!(
                            "continueOperation planner runs under the {} leaf",
                            current.state.at()
                        )
                    };
                    let scope = operation_scope_of(&current.state);
                    // The follow-up fallback is consulted exactly when there
                    // is no compaction threshold and the checkpoint may
                    // finish.
                    let placement = plan_boundary_inbox(
                        &lane,
                        &drive,
                        &state,
                        scope.clone(),
                        session.as_ref(),
                        state.tip_id.clone(),
                        threshold.is_none()
                            && matches!(
                                checkpoint.checkpoint.continuation,
                                Continuation::MayFinish { .. }
                            ),
                    )
                    .await?;
                    if let Some(trigger_entry_id) = placement.trigger_entry_id.clone() {
                        return Ok(OperationCommand::Commit {
                            writes: placement.writes.clone(),
                            operation_state: OperationState::AssistantReady(
                                assistant_ready_at_boundary(
                                    &lane,
                                    &state,
                                    scope,
                                    trigger_entry_id,
                                    false,
                                ),
                            ),
                            lane: Some(LanePatch {
                                tip_id: Some(placement.tip_id.clone()),
                                inbox: Some(placement.inbox.clone()),
                                ..LanePatch::default()
                            }),
                            materialize: Arc::new(|_: &CommitResult| {
                                CheckpointPublication::Procedure(ProcedureResult::Continue)
                            }),
                            events: Some(Arc::new({
                                let lane_name = lane.name().to_owned();
                                let run_id = drive.operation_id.clone();
                                move |commit: &CommitResult| {
                                    boundary_placement_events(
                                        &placement, commit, 0, &lane_name, &run_id,
                                    )
                                }
                            })),
                        });
                    }
                    if let Some(preparation) = &threshold {
                        let structural = SummaryDecidingOperation {
                            scope,
                            task: SummaryTask {
                                task_id: preparation.task_id.clone(),
                                reason: Some(CompactionReason::Threshold),
                                custom_instructions: None,
                                boundary: ResultBoundary::ResumeCheckpoint {
                                    resume_after: CheckpointData {
                                        continuation: checkpoint.checkpoint.continuation,
                                        trigger_entry_id: checkpoint
                                            .checkpoint
                                            .trigger_entry_id
                                            .clone(),
                                    },
                                },
                            },
                        };
                        let mut writes = placement.writes.clone();
                        writes.push(
                            set_value_write(
                                &operation_preparation(&drive.operation_id, &preparation.task_id),
                                preparation.preparation.clone(),
                            )
                            .map_err(lane_error)?,
                        );
                        return Ok(OperationCommand::Commit {
                            writes,
                            operation_state: OperationState::SummaryDeciding(structural),
                            lane: Some(LanePatch {
                                tip_id: Some(placement.tip_id.clone()),
                                inbox: Some(placement.inbox.clone()),
                                ..LanePatch::default()
                            }),
                            materialize: Arc::new(|_: &CommitResult| {
                                CheckpointPublication::Procedure(ProcedureResult::Continue)
                            }),
                            events: Some(Arc::new({
                                let lane_name = lane.name().to_owned();
                                let run_id = drive.operation_id.clone();
                                move |commit: &CommitResult| {
                                    let mut events = boundary_placement_events(
                                        &placement, commit, 0, &lane_name, &run_id,
                                    );
                                    events.push(lane_scoped_event(
                                        &lane_name,
                                        false,
                                        "compaction_start",
                                        HarnessEventPayload::CompactionStart {
                                            run_id: run_id.clone(),
                                            reason: CompactionReason::Threshold,
                                            started_at: commit.timestamp,
                                        },
                                    ));
                                    events
                                }
                            })),
                        });
                    }
                    // The checkpoint's OWN trigger and overflow flag, not the
                    // placement's.
                    if let Continuation::NeedAssistant {
                        overflow_recovery_used,
                    } = checkpoint.checkpoint.continuation
                    {
                        return Ok(OperationCommand::Commit {
                            writes: placement.writes.clone(),
                            operation_state: OperationState::AssistantReady(
                                assistant_ready_at_boundary(
                                    &lane,
                                    &state,
                                    scope,
                                    checkpoint.checkpoint.trigger_entry_id.clone(),
                                    overflow_recovery_used,
                                ),
                            ),
                            lane: Some(LanePatch {
                                tip_id: Some(placement.tip_id.clone()),
                                inbox: Some(placement.inbox.clone()),
                                ..LanePatch::default()
                            }),
                            materialize: Arc::new(|_: &CommitResult| {
                                CheckpointPublication::Procedure(ProcedureResult::Continue)
                            }),
                            events: Some(Arc::new({
                                let lane_name = lane.name().to_owned();
                                let run_id = drive.operation_id.clone();
                                move |commit: &CommitResult| {
                                    boundary_placement_events(
                                        &placement, commit, 0, &lane_name, &run_id,
                                    )
                                }
                            })),
                        });
                    }
                    Ok(OperationCommand::Return {
                        result: CheckpointPublication::FinishPending(BoundaryFinishPending {
                            entry_ids: placement
                                .entries
                                .iter()
                                .map(|entry| entry.id().to_owned())
                                .collect(),
                        }),
                    })
                })
            },
            &drive.context,
        )
        .await?;
    let ContinueOperationResult::Result { value: publication } = planned else {
        return Ok(ProcedureResult::Continue);
    };
    let CheckpointPublication::Procedure(result) = publication else {
        let CheckpointPublication::FinishPending(pending) = publication else {
            unreachable!("the publication is either a procedure result or finish-pending")
        };
        let Continuation::MayFinish { .. } = run.checkpoint.continuation else {
            return Err(lane_error(SessionError::Invariant(
                "Checkpoint finish mediation requires a finish continuation".to_owned(),
            )));
        };
        return finish_run_boundary(
            lane,
            drive,
            &OperationState::Checkpoint(run.clone()),
            run.checkpoint.continuation,
            &pending.entry_ids,
            Vec::new(),
        )
        .await;
    };
    Ok(result)
}

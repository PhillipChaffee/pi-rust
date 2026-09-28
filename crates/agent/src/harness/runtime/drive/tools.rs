//! The durable tool-batch execution procedure, ported from upstream
//! `src/harness/runtime/drive/tools.ts`.
//!
//! `run_tools` executes, recovers, stages, and source-orders one complete
//! durable tool batch: it reads the batch's assistant source, materializes
//! any already-staged outcomes, and runs the batch sequentially or in
//! parallel per the captured settings, never starting effects under
//! `cancel_requested` control. The per-call pipeline reuses the
//! execution-layer stages ([`prepare_tool_call`],
//! [`apply_before_tool_decision`], [`execute_tool_call`],
//! `finalize_tool_call`) and the placement layer
//! (`materialize_ready`); a call's staged outcome rides the batch's
//! `outcome_ready` phase until `materialize_ready` places it.
//!
//! Upstream's thrown rejections restate as `Result` errors: the batch
//! invariants raise [`SessionError::Invariant`], the hook and gate aborts
//! carry their cancellation through [`crate::harness::gate::AbortRequested`],
//! and the invocation capability's rejections surface as the trait's
//! string errors.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pi_ai::types::BoxedFuture;
use pi_ai::types::Message;
use pi_ai::types::StopReason;
use pi_ai::types::ToolResultMessage;
use serde_json::Value as JsonValue;

use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HookEvent;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::agent_harness::HookResult;
use crate::harness::context::Context;
use crate::harness::execution::tools::ClearedToolCall;
use crate::harness::execution::tools::FinalizedToolCall;
use crate::harness::execution::tools::ToolExecutionRejection;
use crate::harness::execution::tools::apply_before_tool_decision;
use crate::harness::execution::tools::create_tool_result_message;
use crate::harness::execution::tools::execute_tool_call;
use crate::harness::execution::tools::finalize_tool_call;
use crate::harness::execution::tools::prepare_tool_call;
use crate::harness::execution::tools::tool_result_from_message;
use crate::harness::gate::GateRejection;
use crate::harness::runtime::clock::now_ms;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::progress::open_tool_progress;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneCommand;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::LaneState;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::runtime::types::ProcedureResult;
use crate::harness::runtime::types::lane_error;
use crate::harness::runtime::types::panicked_task_error;
use crate::harness::session::types::Control;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::ToolBatch;
use crate::harness::session::types::ToolCall;
use crate::harness::session::types::ToolCallStatus;
use crate::harness::session::types::ToolsOperation;
use crate::harness::session::types::operation_scope_of;
use crate::harness::session::values::ToolArgs;
use crate::harness::session::values::ToolOutputPayload;
use crate::harness::session::values::Write;
use crate::harness::session::values::delete_value_write;
use crate::harness::session::values::operation_tool_args;
use crate::harness::session::values::operation_tool_memo;
use crate::harness::session::values::operation_tool_memo_prefix;
use crate::harness::session::values::pending_entry;
use crate::harness::session::values::pending_tool_output;
use crate::harness::session::values::set_value_write;
use crate::harness::types::AgentHarnessTool;
use crate::harness::types::AgentHarnessToolContextSource;
use crate::harness::types::AgentHarnessToolInvocation;
use crate::types::AgentMessage;
use crate::types::AgentToolCall;
use crate::types::AgentToolResult;
use crate::types::ToolReplay;

use super::await_abort_cancellation;
use super::tool_placement::ToolBatchSource;
use super::tool_placement::delete_stored_value;
use super::tool_placement::materialize_ready;
use super::tool_placement::read_tool_batch_source;
use super::tool_placement::tool_call_for;
use super::tool_placement::tool_call_with_status;
use super::tool_placement::with_tool_batch;

/// The text one interrupted staged outcome closes with, upstream's
/// `INTERRUPTION_MARKER`.
const INTERRUPTION_MARKER: &str = "[Tool execution was interrupted. The preceding output is the latest durable progress snapshot; newer live output may be missing, and the external outcome is unknown.]";

/// The rejection one expired invocation capability carries, upstream's
/// `ToolInvocationEnded` error.
#[derive(Debug)]
struct ToolInvocationEnded;

impl std::fmt::Display for ToolInvocationEnded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Tool invocation no longer owns its durable effect")
    }
}

impl std::error::Error for ToolInvocationEnded {}

/// The tools leaf's live batch, upstream's `currentBatch`: `None` when no
/// operation is live or its leaf is not `tools`.
fn current_batch(lane: &Lane) -> Option<(ToolsOperation, ToolBatch)> {
    let operation = lane.state().operation?;
    let OperationState::Tools(run) = operation.state else {
        return None;
    };
    let batch = run.batch.clone();
    Some((run, batch))
}

/// The batch call one source index and result-entry id names, upstream's
/// `findCall`.
fn find_call<'a>(
    batch: &'a ToolBatch,
    source_index: u64,
    result_entry_id: &str,
) -> Option<&'a ToolCall> {
    batch
        .calls
        .iter()
        .find(|call| call.source_index == source_index && call.result_entry_id == result_entry_id)
}

/// Rebuilds one batch over a replaced call, upstream's `replaceCall`: the
/// call matching both `sourceIndex` and `resultEntryId` swaps, the rest
/// stand.
fn replace_call(batch: &ToolBatch, replacement: &ToolCall) -> ToolBatch {
    ToolBatch {
        calls: batch
            .calls
            .iter()
            .map(|call| {
                if call.source_index == replacement.source_index
                    && call.result_entry_id == replacement.result_entry_id
                {
                    replacement.clone()
                } else {
                    call.clone()
                }
            })
            .collect(),
        ..batch.clone()
    }
}

/// The memo-name validation, upstream's `validateMemoName` `TypeError`
/// throws.
///
/// # Errors
/// The empty-name and `:`-containing-name messages.
fn validate_memo_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Tool invocation memo name must not be empty".to_owned());
    }
    if name.contains(':') {
        return Err("Tool invocation memo name must not contain ':'".to_owned());
    }
    Ok(())
}

/// Whether the live operation still carries the call in its
/// `effect_pending` phase, upstream's `ownsEffect`.
fn owns_effect(state: &LaneState, source_index: u64, result_entry_id: &str) -> bool {
    let Some(operation) = &state.operation else {
        return false;
    };
    let OperationState::Tools(run) = &operation.state else {
        return false;
    };
    find_call(&run.batch, source_index, result_entry_id)
        .is_some_and(|call| matches!(call.status(), ToolCallStatus::EffectPending { .. }))
}

/// The stable invocation identity one in-flight call runs with, upstream's
/// `invocationCapability`: memo reads and writes route through the lane's
/// mutation line and reject once the capability expired or the live batch
/// no longer carries the call as `effect_pending`. The port restates
/// upstream's `{ invocation, expire }` pair as one struct — the trait
/// object is the invocation, [`ToolInvocationCapability::expire`] is the
/// expire closure.
struct ToolInvocationCapability {
    /// The invocation-local identity, upstream's `invocationId` (the call's
    /// reserved result-entry id).
    invocation_id: String,
    /// The durable operation id, upstream's `operationId`.
    operation_id: String,
    /// The invocation-local turn id, upstream's `turnId`.
    turn_id: String,
    /// The call's source index, the `ownsEffect` key.
    source_index: u64,
    /// The call's reserved result-entry id, the `ownsEffect` key.
    result_entry_id: String,
    /// The lane the memos read and write through.
    lane: Arc<Lane>,
    /// The context the lane commands run under, upstream's `drive.context`.
    context: Context,
    /// The expiry latch, upstream's `active` flag.
    active: Arc<AtomicBool>,
}

impl std::fmt::Debug for ToolInvocationCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolInvocationCapability")
            .field("invocation_id", &self.invocation_id)
            .field("operation_id", &self.operation_id)
            .field("turn_id", &self.turn_id)
            .field("source_index", &self.source_index)
            .field("result_entry_id", &self.result_entry_id)
            .finish_non_exhaustive()
    }
}

impl AgentHarnessToolInvocation for ToolInvocationCapability {
    fn invocation_id(&self) -> &str {
        &self.invocation_id
    }

    fn operation_id(&self) -> &str {
        &self.operation_id
    }

    fn turn_id(&self) -> &str {
        &self.turn_id
    }

    fn get_memo(
        &self,
        name: &str,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<JsonValue>, String>> {
        if let Err(message) = validate_memo_name(name) {
            return Box::pin(std::future::ready(Err(message)));
        }
        if !self.active.load(Ordering::SeqCst) {
            return Box::pin(std::future::ready(Err(ToolInvocationEnded.to_string())));
        }
        let lane = Arc::clone(&self.lane);
        let operation_id = self.operation_id.clone();
        let result_entry_id = self.result_entry_id.clone();
        let source_index = self.source_index;
        let name = name.to_owned();
        let drive_context = self.context.clone();
        Box::pin(async move {
            lane.command::<Option<JsonValue>, _>(
                move |state, session, command_context| {
                    let operation_id = operation_id.clone();
                    let result_entry_id = result_entry_id.clone();
                    let name = name.clone();
                    Box::pin(async move {
                        if !owns_effect(&state, source_index, &result_entry_id) {
                            return Ok(LaneCommand::Reject {
                                error: lane_error(ToolInvocationEnded),
                            });
                        }
                        let stored = session
                            .get_value(
                                &operation_tool_memo(&operation_id, &result_entry_id, &name)
                                    .address,
                                &command_context,
                            )
                            .await
                            .map_err(lane_error)?;
                        Ok(LaneCommand::Return {
                            result: stored.map(|stored| stored.value),
                        })
                    })
                },
                &drive_context,
            )
            .await
            .map_err(|error| error.to_string())
        })
    }

    fn set_memo(
        &self,
        name: &str,
        value: Option<JsonValue>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<(), String>> {
        if let Err(message) = validate_memo_name(name) {
            return Box::pin(std::future::ready(Err(message)));
        }
        if !self.active.load(Ordering::SeqCst) {
            return Box::pin(std::future::ready(Err(ToolInvocationEnded.to_string())));
        }
        let lane = Arc::clone(&self.lane);
        let operation_id = self.operation_id.clone();
        let result_entry_id = self.result_entry_id.clone();
        let source_index = self.source_index;
        let name = name.to_owned();
        let drive_context = self.context.clone();
        Box::pin(async move {
            lane.command::<(), _>(
                move |state, _session, _command_context| {
                    let operation_id = operation_id.clone();
                    let result_entry_id = result_entry_id.clone();
                    let name = name.clone();
                    let value = value.clone();
                    Box::pin(async move {
                        if !owns_effect(&state, source_index, &result_entry_id) {
                            return Ok(LaneCommand::Reject {
                                error: lane_error(ToolInvocationEnded),
                            });
                        }
                        let address = operation_tool_memo(&operation_id, &result_entry_id, &name);
                        let write = match value {
                            None => delete_value_write(&address),
                            Some(value) => set_value_write(&address, value).map_err(lane_error)?,
                        };
                        Ok(LaneCommand::Commit {
                            writes: vec![write],
                            next: state,
                            materialize: Arc::new(|_| ()),
                            events: None,
                        })
                    })
                },
                &drive_context,
            )
            .await
            .map_err(|error| error.to_string())
        })
    }
}

impl ToolInvocationCapability {
    /// Stops the capability's memo surface, upstream's `expire`.
    fn expire(&self) {
        self.active.store(false, Ordering::SeqCst);
    }
}

/// Builds the capability one in-flight call runs with, upstream's
/// `invocationCapability`.
fn invocation_capability(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    batch: &ToolBatch,
    call: &ToolCall,
) -> ToolInvocationCapability {
    ToolInvocationCapability {
        invocation_id: call.result_entry_id.clone(),
        operation_id: drive.operation_id.clone(),
        turn_id: batch.turn_id.clone(),
        source_index: call.source_index,
        result_entry_id: call.result_entry_id.clone(),
        lane: Arc::clone(lane),
        context: drive.context.clone(),
        active: Arc::new(AtomicBool::new(true)),
    }
}

/// The synthetic error result one refusal carries, upstream's
/// `syntheticMessage`: an error tool-result message stamped now, with the
/// details and usage only when the options carry them.
fn synthetic_message(
    tool_call: &AgentToolCall,
    content: Vec<crate::types::AgentToolContent>,
    options: Option<(&JsonValue, Option<pi_ai::types::Usage>)>,
) -> ToolResultMessage {
    let (details, usage) = match options {
        None => (None, None),
        Some((details, usage)) => ((!details.is_null()).then(|| details.clone()), usage),
    };
    ToolResultMessage {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        content,
        details,
        usage,
        added_tool_names: None,
        is_error: true,
        timestamp: now_ms(),
    }
}

/// One settled call's outcome ready to stage, upstream's `ToolOutcome`.
struct ToolOutcome {
    /// The call the outcome answers, upstream's `toolCall`.
    tool_call: AgentToolCall,
    /// The settled tool-result message, upstream's `message`.
    message: ToolResultMessage,
    /// Whether the call terminates the run, upstream's `terminate`.
    terminate: bool,
}

/// The outcome one cancelled-before-completion call stages, upstream's
/// `abortedOutcome`.
fn aborted_outcome(tool_call: &AgentToolCall) -> ToolOutcome {
    ToolOutcome {
        tool_call: tool_call.clone(),
        message: synthetic_message(
            tool_call,
            vec![crate::types::AgentToolContent::Text(
                pi_ai::types::TextContent {
                    text: "Tool execution was cancelled before completion.".to_owned(),
                    text_signature: None,
                },
            )],
            None,
        ),
        terminate: false,
    }
}

/// The outcome one interrupted effect stages: the durable checkpoint's
/// content (when any) closed with the interruption marker, upstream's
/// `interruptedOutcome`.
fn interrupted_outcome(
    tool_call: &AgentToolCall,
    checkpoint: Option<&AgentToolResult>,
) -> ToolOutcome {
    let mut content = checkpoint
        .map(|checkpoint| checkpoint.content.clone())
        .unwrap_or_default();
    content.push(crate::types::AgentToolContent::Text(
        pi_ai::types::TextContent {
            text: INTERRUPTION_MARKER.to_owned(),
            text_signature: None,
        },
    ));
    ToolOutcome {
        tool_call: tool_call.clone(),
        message: synthetic_message(
            tool_call,
            content,
            checkpoint.map(|checkpoint| (&checkpoint.details, checkpoint.usage)),
        ),
        terminate: false,
    }
}

/// The outcome one length-stopped call stages, upstream's `truncatedOutcome`.
fn truncated_outcome(tool_call: &AgentToolCall) -> ToolOutcome {
    ToolOutcome {
        tool_call: tool_call.clone(),
        message: synthetic_message(
            tool_call,
            vec![crate::types::AgentToolContent::Text(
                pi_ai::types::TextContent {
                    text: format!(
                        "Tool call {} was not executed because the assistant response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
                        serde_json::to_string(&tool_call.name).unwrap_or_else(
                            |error| unreachable!("the tool name serializes: {error}")
                        )
                    ),
                    text_signature: None,
                },
            )],
            None,
        ),
        terminate: false,
    }
}

/// The outcome one finalized call carries, upstream's `outcomeFromFinalizedCall`.
fn outcome_from_finalized(finalized: &FinalizedToolCall) -> ToolOutcome {
    ToolOutcome {
        tool_call: finalized.tool_call.clone(),
        message: create_tool_result_message(finalized),
        terminate: finalized.terminate,
    }
}

/// The `ToolOutcome` one immediate refusal carries: the immediate outcome's
/// result restates as the finalized shape's always-error statement.
fn outcome_from_immediate(
    immediate: crate::harness::execution::tools::ImmediateToolOutcome,
) -> ToolOutcome {
    let finalized = FinalizedToolCall {
        tool_call: immediate.tool_call,
        result: immediate.result,
        is_error: true,
        terminate: immediate.terminate,
    };
    outcome_from_finalized(&finalized)
}

/// The event-wrapped lane name one commit event carries, for readability at
/// the three publishers below.
fn lane_scoped_event(lane: &Lane, recovery: bool, payload: HarnessEventPayload) -> HarnessEvent {
    crate::harness::agent_harness::lane_scoped_event(
        lane.name(),
        recovery,
        "the tool events",
        payload,
    )
}

/// The map view of one arguments object, the hook events' `Record<string,
/// JsonValue>` parameter.
fn args_map(args: &JsonValue) -> BTreeMap<String, JsonValue> {
    args.as_object()
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// Publishes the durable intent one planned call runs from, upstream's
/// `publishToolIntent`: a `continueOperation` that persists the validated
/// arguments and moves the call to `effect_pending`, the commit publishing
/// the call's `tool_start`.
///
/// # Errors
/// The `continueOperation`'s planner, commit, storage, and delivery
/// failures.
async fn publish_tool_intent(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    planned: &ToolCall,
    tool_call: &AgentToolCall,
    args: &JsonValue,
    replay: ToolReplay,
    recovery: bool,
) -> Result<ContinueOperationResult<ToolCall>, LaneError> {
    let planner_lane = Arc::clone(lane);
    let planner_drive = Arc::clone(drive);
    let planned = planned.clone();
    let tool_call = tool_call.clone();
    let args = args.clone();
    lane.continue_operation::<ToolCall, _>(
        move |state, _session, _plan_context| {
            let lane = Arc::clone(&planner_lane);
            let drive = Arc::clone(&planner_drive);
            let planned = planned.clone();
            let tool_call = tool_call.clone();
            let args = args.clone();
            Box::pin(async move {
                let Some(operation) = state.operation.as_ref() else {
                    unreachable!("continueOperation planner runs under an active operation")
                };
                let OperationState::Tools(run) = &operation.state else {
                    unreachable!("continueOperation planner runs under the dispatched tools leaf")
                };
                let effect_pending =
                    tool_call_with_status(&planned, ToolCallStatus::EffectPending { replay });
                let turn_id = run.batch.turn_id.clone();
                let next_run =
                    with_tool_batch(run.clone(), replace_call(&run.batch, &effect_pending));
                let event = lane_scoped_event(
                    &lane,
                    recovery,
                    HarnessEventPayload::ToolStart {
                        run_id: drive.operation_id.clone(),
                        turn_id,
                        tool_call_id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        args: args.clone(),
                    },
                );
                let args_write = args.as_object().cloned().unwrap_or_default();
                Ok(OperationCommand::Commit {
                    writes: vec![
                        set_value_write(
                            &operation_tool_args(
                                &drive.operation_id,
                                &run.batch.turn_id,
                                planned.source_index,
                            ),
                            args_write,
                        )
                        .map_err(lane_error)?,
                    ],
                    operation_state: OperationState::Tools(next_run),
                    lane: None,
                    materialize: Arc::new(
                        move |_: &crate::harness::session::types::CommitResult| {
                            effect_pending.clone()
                        },
                    ),
                    events: Some(Arc::new(
                        move |_: &crate::harness::session::types::CommitResult| vec![event.clone()],
                    )),
                })
            })
        },
        &drive.context,
    )
    .await
}

/// Stages one settled call's outcome, upstream's `publishToolOutcome`: a
/// `settleOperation` that writes the staged pending entry, clears the
/// call's checkpoint and memo family, moves the call to `outcome_ready`
/// with the durability-gated terminate hint, and publishes the
/// `tool_start` (when the call never published an intent) and `tool_end`.
///
/// # Errors
/// The settle's planner, commit, storage, and delivery failures.
async fn publish_tool_outcome(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    call: &ToolCall,
    outcome: ToolOutcome,
    recovery: bool,
) -> Result<(), LaneError> {
    let planner_lane = Arc::clone(lane);
    let planner_drive = Arc::clone(drive);
    let call = Arc::new(call.clone());
    let outcome = Arc::new(outcome);
    lane.settle_operation::<(), _>(
        move |_state, current, _meta, session, plan_context| {
            let lane = Arc::clone(&planner_lane);
            let drive = Arc::clone(&planner_drive);
            let call = Arc::clone(&call);
            let outcome = Arc::clone(&outcome);
            let current = current.clone();
            Box::pin(async move {
                let OperationState::Tools(run) = current else {
                    return Err(lane_error(SessionError::Invariant(
                        "Settled operation is not a tool batch".to_owned(),
                    )));
                };
                let memos = session
                    .scan_values(
                        &operation_tool_memo_prefix(
                            &drive.operation_id,
                            Some(&call.result_entry_id),
                        )
                        .address,
                        &plan_context,
                    )
                    .await
                    .map_err(lane_error)?;
                let durable_terminate =
                    matches!(run.scope.control, Control::Running) && outcome.terminate;
                let outcome_ready = tool_call_with_status(
                    &call,
                    ToolCallStatus::OutcomeReady {
                        terminate: durable_terminate,
                    },
                );
                let mut writes: Vec<Write> = vec![
                    set_value_write(
                        &pending_entry(&call.result_entry_id),
                        PendingEntry::Message {
                            payload: Box::new(AgentMessage::Standard(Message::ToolResult(
                                outcome.message.clone(),
                            ))),
                        },
                    )
                    .map_err(lane_error)?,
                    delete_value_write(&pending_tool_output(
                        &drive.operation_id,
                        &call.result_entry_id,
                    )),
                ];
                writes.extend(memos.iter().map(delete_stored_value));
                let mut events: Vec<HarnessEvent> = Vec::new();
                if matches!(call.status(), ToolCallStatus::Planned) {
                    events.push(lane_scoped_event(
                        &lane,
                        recovery,
                        HarnessEventPayload::ToolStart {
                            run_id: drive.operation_id.clone(),
                            turn_id: run.batch.turn_id.clone(),
                            tool_call_id: outcome.tool_call.id.clone(),
                            tool_name: outcome.tool_call.name.clone(),
                            args: JsonValue::Object(outcome.tool_call.arguments.clone()),
                        },
                    ));
                }
                events.push(lane_scoped_event(
                    &lane,
                    recovery,
                    HarnessEventPayload::ToolEnd {
                        run_id: drive.operation_id.clone(),
                        turn_id: run.batch.turn_id.clone(),
                        tool_call_id: outcome.tool_call.id.clone(),
                        tool_name: outcome.tool_call.name.clone(),
                        result: tool_result_from_message(&outcome.message, durable_terminate),
                        is_error: outcome.message.is_error,
                        terminate: durable_terminate,
                    },
                ));
                Ok(OperationCommand::Commit {
                    writes,
                    operation_state: OperationState::Tools(with_tool_batch(
                        run.clone(),
                        replace_call(&run.batch, &outcome_ready),
                    )),
                    lane: None,
                    materialize: Arc::new(|_: &crate::harness::session::types::CommitResult| ()),
                    events: Some(Arc::new(
                        move |_: &crate::harness::session::types::CommitResult| events.clone(),
                    )),
                })
            })
        },
        &drive.context,
    )
    .await
}

/// Clears one safe replay's checkpoint and returns the persisted arguments,
/// upstream's `clearReplayCheckpoint`: a command that deletes the staged
/// tool output, publishes the replay's recovery-flagged `tool_start`, and
/// materializes the stored arguments.
///
/// # Errors
/// The [`SessionError::Invariant`] `` `Tool call {resultEntryId} is missing
/// persisted arguments` `` when the persisted arguments are absent, and the
/// command's storage and delivery failures.
async fn clear_replay_checkpoint(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    batch: &ToolBatch,
    call: &ToolCall,
    tool_call: &AgentToolCall,
) -> Result<ToolArgs, LaneError> {
    let planner_lane = Arc::clone(lane);
    let planner_drive = Arc::clone(drive);
    let call = call.clone();
    let tool_call = tool_call.clone();
    let turn_id = batch.turn_id.clone();
    lane.command::<ToolArgs, _>(
        move |state, session, command_context| {
            let lane = Arc::clone(&planner_lane);
            let drive = Arc::clone(&planner_drive);
            let call = call.clone();
            let tool_call = tool_call.clone();
            let turn_id = turn_id.clone();
            Box::pin(async move {
                let stored = session
                    .get_value(
                        &operation_tool_args(&drive.operation_id, &turn_id, call.source_index)
                            .address,
                        &command_context,
                    )
                    .await
                    .map_err(lane_error)?;
                let Some(stored) = stored else {
                    return Err(lane_error(SessionError::Invariant(format!(
                        "Tool call {} is missing persisted arguments",
                        call.result_entry_id
                    ))));
                };
                let args: ToolArgs = serde_json::from_value(stored.value).map_err(|error| {
                    lane_error(SessionError::Message(format!(
                        "Stored tool arguments are malformed: {error}"
                    )))
                })?;
                let event = lane_scoped_event(
                    &lane,
                    true,
                    HarnessEventPayload::ToolStart {
                        run_id: drive.operation_id.clone(),
                        turn_id,
                        tool_call_id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        args: JsonValue::Object(args.clone()),
                    },
                );
                Ok(LaneCommand::Commit {
                    writes: vec![delete_value_write(&pending_tool_output(
                        &drive.operation_id,
                        &call.result_entry_id,
                    ))],
                    next: state,
                    materialize: Arc::new(
                        move |_: &crate::harness::session::types::CommitResult| args.clone(),
                    ),
                    events: Some(Arc::new(
                        move |_: &crate::harness::session::types::CommitResult| vec![event.clone()],
                    )),
                })
            })
        },
        &drive.context,
    )
    .await
}

/// Reads one call's durable checkpoint, upstream's `readCheckpoint`.
///
/// # Errors
/// The command's storage failure.
async fn read_checkpoint(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    call: &ToolCall,
) -> Result<Option<AgentToolResult>, LaneError> {
    let operation_id = drive.operation_id.clone();
    let result_entry_id = call.result_entry_id.clone();
    lane.command::<Option<AgentToolResult>, _>(
        move |_state, session, command_context| {
            let operation_id = operation_id.clone();
            let result_entry_id = result_entry_id.clone();
            Box::pin(async move {
                let stored = session
                    .get_value(
                        &pending_tool_output(&operation_id, &result_entry_id).address,
                        &command_context,
                    )
                    .await
                    .map_err(lane_error)?;
                let checkpoint = stored.map(|stored| {
                    serde_json::from_value::<ToolOutputPayload>(stored.value)
                        .map(|payload| AgentToolResult {
                            content: payload.content,
                            details: payload.details,
                            usage: payload.usage,
                            added_tool_names: payload.added_tool_names,
                            terminate: payload.terminate,
                        })
                        .map_err(|error| {
                            lane_error(SessionError::Message(format!(
                                "Pending tool output is malformed: {error}"
                            )))
                        })
                });
                let checkpoint = match checkpoint {
                    Some(result) => Some(result?),
                    None => None,
                };
                Ok(LaneCommand::Return { result: checkpoint })
            })
        },
        &drive.context,
    )
    .await
}

/// Resolves the turn's tool context, upstream's `resolveToolContext`: the
/// static value stands, the provider resolves per turn.
async fn resolve_tool_context(
    lane: &Lane,
    drive: &Arc<Drive>,
) -> crate::harness::types::ToolContext {
    match &lane.read_config().tool_context {
        None => None,
        Some(AgentHarnessToolContextSource::Static(value)) => value.clone(),
        Some(AgentHarnessToolContextSource::Resolved(provider)) => provider(&drive.context).await,
    }
}

/// The execution surface one batch's calls run with, upstream's
/// `{ tools, toolsByName, toolContext }` execution object.
struct ToolExecutionContext {
    /// The active tools in the config's order, upstream's `tools`.
    tools: Vec<AgentHarnessTool>,
    /// The same tools keyed by name, upstream's `toolsByName`.
    tools_by_name: HashMap<String, AgentHarnessTool>,
    /// The resolved tool context, upstream's `toolContext`.
    tool_context: crate::harness::types::ToolContext,
}

/// The future one started call completes through, upstream's `ToolCallTask`.
struct ToolCallTask {
    /// The started call's completion, upstream's `completion`.
    completion: BoxedFuture<'static, Result<(), LaneError>>,
}

/// Prepares one planned call for execution, upstream's
/// `prepareToolInvocation`: resolve the source block, refuse
/// length-stopped calls, prepare and validate, run `before_tool`, and
/// apply its decision. Upstream's ready/outcome union restates as the
/// inner result (the refused preparation on the error side); the thrown
/// errors — the source-read invariant and non-abort hook failures — ride
/// the outer one.
///
/// # Errors
/// [`tool_call_for`]'s invariant, and any non-abort hook failure.
async fn prepare_tool_invocation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    sources: &ToolBatchSource,
    call: &ToolCall,
    tools: &[AgentHarnessTool],
) -> Result<Result<ClearedToolCall, ToolOutcome>, LaneError> {
    let tool_call = tool_call_for(sources, call)?;
    if matches!(sources.assistant.stop_reason, StopReason::Length) {
        return Ok(Err(truncated_outcome(&tool_call)));
    }
    let prepared = match prepare_tool_call(&tool_call, tools) {
        Ok(prepared) => prepared,
        Err(immediate) => return Ok(Err(outcome_from_immediate(immediate))),
    };

    let invocation = HookInvocation {
        lane: lane.name().to_owned(),
        run_id: drive.operation_id.clone(),
        event: HookEvent::BeforeTool {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: args_map(&prepared.args),
        },
    };
    let decision = match lane
        .hooks()
        .run_tool_with_gate(
            HookName::BeforeTool,
            invocation,
            &drive.gate,
            &drive.context,
        )
        .await
    {
        Ok(HookResult::BeforeTool(decision)) => decision,
        Ok(_) => unreachable!("before_tool returns its own result variant"),
        Err(crate::harness::hooks::HookRunError::GateAborted(abort)) => {
            await_abort_cancellation(&abort.cancellation).await;
            return Ok(Err(aborted_outcome(&tool_call)));
        }
        Err(crate::harness::hooks::HookRunError::Aborted(_)) => {
            await_abort_cancellation(&drive.abort_cancellation()).await;
            return Ok(Err(aborted_outcome(&tool_call)));
        }
        Err(error) => return Err(lane_error(error)),
    };
    let cleared = match apply_before_tool_decision(prepared, decision.as_ref()) {
        Ok(cleared) => cleared,
        Err(immediate) => return Ok(Err(outcome_from_immediate(immediate))),
    };
    Ok(Ok(cleared))
}

/// Runs one cleared call's external effect and settles its staged outcome,
/// upstream's `performToolInvocation`: the invocation capability and
/// checkpoint progress open per call, partial updates publish before the
/// execution settles, and the update delivery and checkpoint writes drain
/// ahead of `after_tool`.
///
/// # Errors
/// The gate's closed rejection, the update deliveries' and checkpoint
/// commits' failures, and any non-abort hook failure.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single performToolInvocation and its abort choreography"
)]
async fn perform_tool_invocation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    batch: &ToolBatch,
    call: &ToolCall,
    cleared: &ClearedToolCall,
    tool_context: crate::harness::types::ToolContext,
    recovery: bool,
) -> Result<ToolOutcome, LaneError> {
    let capability = invocation_capability(lane, drive, batch, call);
    let progress = Arc::new(open_tool_progress(
        lane,
        drive,
        &batch.turn_id,
        call.source_index,
        &call.result_entry_id,
    ));
    // The newest update delivery's handle, upstream's
    // `latestUpdateDelivery` promise; the spawned deliveries detach so a
    // replaced delivery still settles, the detached promise's reachability.
    let latest_delivery = Arc::new(std::sync::Mutex::new(None));
    let publish_update = {
        let latest = Arc::clone(&latest_delivery);
        let lane = Arc::clone(lane);
        let context = drive.context.clone();
        let run_id = drive.operation_id.clone();
        let turn_id = batch.turn_id.clone();
        let tool_call_id = cleared.tool_call.id.clone();
        let tool_name = cleared.tool_call.name.clone();
        Arc::new(move |partial: &AgentToolResult| {
            let event = lane_scoped_event(
                &lane,
                recovery,
                HarnessEventPayload::ToolUpdate {
                    run_id: run_id.clone(),
                    turn_id: turn_id.clone(),
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    partial_result: partial.clone(),
                },
            );
            let delivery = (lane.emit_batch())(vec![event], context.clone());
            let handle = tokio::spawn(delivery);
            *latest
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(handle);
        })
    };

    let on_update = {
        let publish_update = Arc::clone(&publish_update);
        let progress = Arc::clone(&progress);
        move |partial: &AgentToolResult,
              options: Option<crate::harness::types::AgentHarnessToolUpdateOptions>| {
            (publish_update)(partial);
            if options.is_some_and(|options| options.checkpoint) {
                progress.write(partial.clone());
            }
        }
    };
    let executed = execute_tool_call(
        cleared,
        &drive.gate,
        Some(&on_update),
        tool_context.clone(),
        &capability,
        &drive.context,
    )
    .await;
    capability.expire();
    progress.seal();
    let executed = match executed {
        Ok(executed) => executed,
        Err(rejection) => {
            progress.drain().await?;
            match rejection {
                ToolExecutionRejection::Gate(GateRejection::AbortRequested(abort)) => {
                    await_abort_cancellation(&abort.cancellation).await;
                }
                ToolExecutionRejection::Aborted(_) => {
                    await_abort_cancellation(&drive.abort_cancellation()).await;
                }
                ToolExecutionRejection::Gate(GateRejection::Closed(error)) => {
                    return Err(lane_error(error));
                }
            }
            return Ok(if recovery {
                interrupted_outcome(&cleared.tool_call, None)
            } else {
                aborted_outcome(&cleared.tool_call)
            });
        }
    };

    // The newest delivery settles ahead of the drain, upstream's
    // `await latestUpdateDelivery` then `await progress.drain()`.
    let delivery = latest_delivery
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    if let Some(delivery) = delivery {
        delivery.await.unwrap_or_else(|error| {
            unreachable!("the update delivery task does not panic: {error}")
        })?;
    }
    progress.drain().await?;

    let invocation = HookInvocation {
        lane: lane.name().to_owned(),
        run_id: drive.operation_id.clone(),
        event: HookEvent::AfterTool {
            tool_call_id: cleared.tool_call.id.clone(),
            tool_name: cleared.tool_call.name.clone(),
            args: args_map(&cleared.args),
            content: executed.result.content.clone(),
            details: (!executed.result.details.is_null()).then(|| executed.result.details.clone()),
            is_error: executed.is_error,
            usage: executed.result.usage,
        },
    };
    let patch = match lane
        .hooks()
        .run_tool_with_gate(HookName::AfterTool, invocation, &drive.gate, &drive.context)
        .await
    {
        Ok(HookResult::AfterTool(patch)) => patch,
        Ok(_) => unreachable!("after_tool returns its own result variant"),
        Err(
            crate::harness::hooks::HookRunError::GateAborted(_)
            | crate::harness::hooks::HookRunError::Aborted(_),
        ) => None,
        Err(error) => return Err(lane_error(error)),
    };
    let finalized = finalize_tool_call(cleared, executed, patch);
    let message = create_tool_result_message(&finalized);
    Ok(ToolOutcome {
        tool_call: finalized.tool_call,
        message,
        terminate: finalized.terminate,
    })
}

/// Starts one planned call: prepare, publish the durable intent, and run
/// the effect, upstream's `startToolInvocation`. A refused preparation or
/// a `cancel_requested` intent stages its synthetic outcome directly.
///
/// # Errors
/// [`prepare_tool_invocation`]'s and [`publish_tool_intent`]'s errors.
#[expect(
    clippy::too_many_arguments,
    reason = "the port mirrors upstream's startToolInvocation signature"
)]
async fn start_tool_invocation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    run: &ToolsOperation,
    sources: &ToolBatchSource,
    call: &ToolCall,
    tools: &[AgentHarnessTool],
    tool_context: crate::harness::types::ToolContext,
    recovery: bool,
) -> Result<ToolCallTask, LaneError> {
    let prepared = prepare_tool_invocation(lane, drive, sources, call, tools).await?;
    let cleared = match prepared {
        Err(outcome) => {
            let completion = {
                let lane = Arc::clone(lane);
                let drive = Arc::clone(drive);
                let call = call.clone();
                Box::pin(async move {
                    publish_tool_outcome(&lane, &drive, &call, outcome, recovery).await
                })
            };
            return Ok(ToolCallTask { completion });
        }
        Ok(cleared) => cleared,
    };
    let effect_pending = publish_tool_intent(
        lane,
        drive,
        call,
        &cleared.tool_call,
        &cleared.args,
        cleared.tool.replay.unwrap_or(ToolReplay::Never),
        recovery,
    )
    .await?;
    let effect_pending = match effect_pending {
        ContinueOperationResult::CancelRequested => {
            let completion = {
                let lane = Arc::clone(lane);
                let drive = Arc::clone(drive);
                let call = call.clone();
                let outcome = aborted_outcome(&cleared.tool_call);
                Box::pin(async move {
                    publish_tool_outcome(&lane, &drive, &call, outcome, recovery).await
                })
            };
            return Ok(ToolCallTask { completion });
        }
        ContinueOperationResult::Result { value } => value,
    };
    let completion = {
        let lane = Arc::clone(lane);
        let drive = Arc::clone(drive);
        let run = run.clone();
        let cleared_captured = cleared;
        Box::pin(async move {
            let outcome = perform_tool_invocation(
                &lane,
                &drive,
                &run.batch,
                &effect_pending,
                &cleared_captured,
                tool_context,
                recovery,
            )
            .await?;
            publish_tool_outcome(&lane, &drive, &effect_pending, outcome, recovery).await
        })
    };
    Ok(ToolCallTask { completion })
}

/// Recovers one effect-pending call, upstream's `recoverToolInvocation`:
/// safe-replay calls with a safe-declared tool clear their checkpoint and
/// re-run, everything else stages the interrupted outcome.
///
/// # Errors
/// [`clear_replay_checkpoint`]'s, [`read_checkpoint`]'s, and the recovery
/// path's errors.
#[expect(
    clippy::too_many_arguments,
    reason = "the port mirrors upstream's recoverToolInvocation signature"
)]
async fn recover_tool_invocation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    run: &ToolsOperation,
    sources: &ToolBatchSource,
    call: &ToolCall,
    tools_by_name: &HashMap<String, AgentHarnessTool>,
    tool_context: crate::harness::types::ToolContext,
    cancelled: bool,
) -> Result<ToolCallTask, LaneError> {
    let tool_call = tool_call_for(sources, call)?;
    let tool = tools_by_name.get(&tool_call.name);
    let replayable = !cancelled
        && matches!(
            call.status(),
            ToolCallStatus::EffectPending {
                replay: ToolReplay::Safe
            }
        )
        && tool.is_some_and(|tool| tool.replay == Some(ToolReplay::Safe));
    if replayable {
        let args = clear_replay_checkpoint(lane, drive, &run.batch, call, &tool_call).await?;
        let Some(tool) = tool.cloned() else {
            unreachable!("the replayable tool resolved by name");
        };
        let cleared = ClearedToolCall {
            tool_call: tool_call.clone(),
            tool,
            args: JsonValue::Object(args),
        };
        let completion = {
            let lane = Arc::clone(lane);
            let drive = Arc::clone(drive);
            let run = run.clone();
            let call = call.clone();
            Box::pin(async move {
                let outcome = perform_tool_invocation(
                    &lane,
                    &drive,
                    &run.batch,
                    &call,
                    &cleared,
                    tool_context,
                    true,
                )
                .await?;
                publish_tool_outcome(&lane, &drive, &call, outcome, true).await
            })
        };
        return Ok(ToolCallTask { completion });
    }
    let checkpoint = read_checkpoint(lane, drive, call).await?;
    let completion = {
        let lane = Arc::clone(lane);
        let drive = Arc::clone(drive);
        let call = call.clone();
        let outcome = interrupted_outcome(&tool_call, checkpoint.as_ref());
        Box::pin(async move { publish_tool_outcome(&lane, &drive, &call, outcome, true).await })
    };
    Ok(ToolCallTask { completion })
}

/// Chains one scheduled materialization onto the previous one, upstream's
/// `scheduleMaterialization` `then`-chain: every scheduled materialization
/// awaits the chain's tail before running, so placements never overlap.
fn schedule_materialization(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    run: &ToolsOperation,
    sources: &ToolBatchSource,
    recovery: bool,
    tail: &Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
) -> BoxedFuture<'static, Result<(), LaneError>> {
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    // The take and the set hold one lock: two concurrent completions
    // scheduling on a multi-thread runtime must chain onto each other,
    // upstream's synchronous `materialization = scheduled.catch(...)` swap.
    let previous = {
        let mut tail = tail
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = tail.take();
        *tail = Some(done_rx);
        previous
    };
    let lane = Arc::clone(lane);
    let drive = Arc::clone(drive);
    let run = run.clone();
    let sources = sources.clone();
    Box::pin(async move {
        // A dropped predecessor's channel closes; the chain proceeds past
        // failures, upstream's `materialization.then(...)` over the
        // swallowed chain promise.
        if let Some(previous) = previous {
            let _ = previous.await;
        }
        let result = materialize_ready(&lane, &drive, &run, &sources, recovery).await;
        let _ = done_tx.send(());
        result
    })
}

/// Runs one batch one call at a time, upstream's `runSequential`: each
/// transition materializes staged outcomes, then starts or recovers the
/// first non-completed call — or stages its synthetic outcome under
/// `cancel_requested` control — until the batch lands at its checkpoint.
///
/// # Errors
/// The invariants `` `Tool batch remained open after every call
/// completed` ``, `` `Ready tool outcome was not materialized` ``, and
/// `` `Sequential tool batch exceeded its bounded transition count` ``, the
/// `` `Running tool batch is missing execution context` `` invariant, and
/// every started call's completion failure.
async fn run_sequential(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    run: &ToolsOperation,
    sources: &ToolBatchSource,
    execution: Option<&ToolExecutionContext>,
    recovery: bool,
) -> Result<ProcedureResult, LaneError> {
    let batch = &run.batch;
    let mut transition: usize = 0;
    while transition <= batch.calls.len() * 2 + 1 {
        materialize_ready(lane, drive, run, sources, recovery).await?;
        let Some((current_run, _current_batch)) = current_batch(lane) else {
            return Ok(ProcedureResult::Continue);
        };
        let call = current_run
            .batch
            .calls
            .iter()
            .find(|call| !matches!(call.status(), ToolCallStatus::Completed { .. }))
            .cloned()
            .ok_or_else(|| {
                lane_error(SessionError::Invariant(
                    "Tool batch remained open after every call completed".to_owned(),
                ))
            })?;
        if matches!(call.status(), ToolCallStatus::OutcomeReady { .. }) {
            return Err(lane_error(SessionError::Invariant(
                "Ready tool outcome was not materialized".to_owned(),
            )));
        }

        if matches!(current_run.scope.control, Control::CancelRequested { .. }) {
            let tool_call = tool_call_for(sources, &call)?;
            if matches!(call.status(), ToolCallStatus::Planned) {
                let outcome = aborted_outcome(&tool_call);
                publish_tool_outcome(lane, drive, &call, outcome, recovery).await?;
            } else {
                let checkpoint = read_checkpoint(lane, drive, &call).await?;
                let outcome = interrupted_outcome(&tool_call, checkpoint.as_ref());
                publish_tool_outcome(lane, drive, &call, outcome, recovery).await?;
            }
            transition += 1;
            continue;
        }

        let Some(execution) = execution else {
            return Err(lane_error(SessionError::Invariant(
                "Running tool batch is missing execution context".to_owned(),
            )));
        };
        let started = if matches!(call.status(), ToolCallStatus::Planned) {
            start_tool_invocation(
                lane,
                drive,
                &current_run,
                sources,
                &call,
                &execution.tools,
                execution.tool_context.clone(),
                recovery,
            )
            .await?
        } else {
            recover_tool_invocation(
                lane,
                drive,
                &current_run,
                sources,
                &call,
                &execution.tools_by_name,
                execution.tool_context.clone(),
                false,
            )
            .await?
        };
        started.completion.await?;
        transition += 1;
    }
    Err(lane_error(SessionError::Invariant(
        "Sequential tool batch exceeded its bounded transition count".to_owned(),
    )))
}

/// Prepares one batch's calls in order and runs them concurrently, upstream's
/// `runParallel`: each started call's job detaches as a spawned task — the
/// promise's eager start — and completes, then schedules a chained
/// materialization; every job's failure propagates after all jobs settle,
/// upstream's `Promise.all` over the pushed job promises.
///
/// # Errors
/// Every job's completion failure and every scheduled materialization's
/// failure.
async fn run_parallel(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    run: &ToolsOperation,
    sources: &ToolBatchSource,
    execution: &ToolExecutionContext,
    recovery: bool,
) -> Result<ProcedureResult, LaneError> {
    let batch = &run.batch;
    let tail = Arc::new(std::sync::Mutex::new(None));
    let mut jobs: Vec<BoxedFuture<'static, Result<(), LaneError>>> = Vec::new();
    for call in &batch.calls {
        if matches!(
            call.status(),
            ToolCallStatus::Completed { .. } | ToolCallStatus::OutcomeReady { .. }
        ) {
            continue;
        }
        let started = if matches!(call.status(), ToolCallStatus::Planned) {
            start_tool_invocation(
                lane,
                drive,
                run,
                sources,
                call,
                &execution.tools,
                execution.tool_context.clone(),
                recovery,
            )
            .await?
        } else {
            let cancelled = matches!(
                lane.state()
                    .operation
                    .as_ref()
                    .map(|operation| operation_scope_of(&operation.state).control),
                Some(Control::CancelRequested { .. })
            );
            recover_tool_invocation(
                lane,
                drive,
                run,
                sources,
                call,
                &execution.tools_by_name,
                execution.tool_context.clone(),
                cancelled,
            )
            .await?
        };
        let job_lane = Arc::clone(lane);
        let job_drive = Arc::clone(drive);
        let job_run = run.clone();
        let job_sources = sources.clone();
        let job_tail = Arc::clone(&tail);
        let job = tokio::spawn(async move {
            started.completion.await?;
            // The materialization schedules at completion time, upstream's
            // `completion.then(() => scheduleMaterialization())` — the
            // chain's order follows the completions.
            let scheduled = schedule_materialization(
                &job_lane,
                &job_drive,
                &job_run,
                &job_sources,
                recovery,
                &job_tail,
            );
            scheduled.await
        });
        jobs.push(Box::pin(async move {
            match job.await {
                Ok(result) => result,
                // An unwound tool job fails the batch like any other error,
                // upstream's job promise rejection reaching `Promise.all`.
                Err(join) => Err(panicked_task_error(join)),
            }
        }));
    }
    let mut failure: Option<LaneError> = None;
    for job in jobs {
        if let Err(error) = job.await {
            failure.get_or_insert(error);
        }
    }
    if let Some(error) = failure {
        return Err(error);
    }
    schedule_materialization(lane, drive, run, sources, recovery, &tail).await?;
    Ok(ProcedureResult::Continue)
}

/// Execute, recover, stage, and source-order one complete durable tool
/// batch, upstream's `runTools`.
///
/// The batch's recovery flag opens the recovery `turn_start` before the
/// source read; staged outcomes materialize before the first call starts;
/// a `cancel_requested` batch runs its cancellation-only sequential pass,
/// and a running batch resolves the config's active tools and tool context
/// once, then runs sequential or parallel per the captured settings.
///
/// # Errors
/// The batch's invariants — `` `Tool batch assistant entry is invalid` ``,
/// `` `Sequential tool batch exceeded its bounded transition count` `` and
/// its siblings — and the intents', outcomes', and placements' storage and
/// delivery errors.
pub(crate) async fn run_tools(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    run: &ToolsOperation,
) -> Result<ProcedureResult, LaneError> {
    let batch = &run.batch;
    let recovery = batch.calls.iter().any(|call| {
        matches!(
            call.status(),
            ToolCallStatus::EffectPending { .. } | ToolCallStatus::OutcomeReady { .. }
        )
    });
    if recovery {
        let event = lane_scoped_event(
            lane,
            true,
            HarnessEventPayload::TurnStart {
                run_id: drive.operation_id.clone(),
                turn_id: batch.turn_id.clone(),
            },
        );
        (lane.emit_batch())(vec![event], drive.context.clone()).await?;
    }
    let sources = read_tool_batch_source(lane, drive, batch).await?;
    materialize_ready(lane, drive, run, &sources, recovery).await?;
    let Some((current_run, _current_batch)) = current_batch(lane) else {
        return Ok(ProcedureResult::Continue);
    };
    if matches!(current_run.scope.control, Control::CancelRequested { .. }) {
        return run_sequential(lane, drive, &current_run, &sources, None, recovery).await;
    }
    let config = lane.read_config();
    let active: std::collections::HashSet<&String> =
        batch.configuration.active_tool_names.iter().collect();
    let tools: Vec<AgentHarnessTool> = config
        .tools
        .iter()
        .filter(|tool| active.contains(&tool.name().to_owned()))
        .cloned()
        .collect();
    let tools_by_name: HashMap<String, AgentHarnessTool> = tools
        .iter()
        .map(|tool| (tool.name().to_owned(), tool.clone()))
        .collect();
    let tool_context = resolve_tool_context(lane, drive).await;
    let execution = ToolExecutionContext {
        tools,
        tools_by_name,
        tool_context,
    };
    match run.scope.settings.tool_execution {
        crate::types::ToolExecutionMode::Sequential => {
            run_sequential(
                lane,
                drive,
                &current_run,
                &sources,
                Some(&execution),
                recovery,
            )
            .await
        }
        crate::types::ToolExecutionMode::Parallel => {
            run_parallel(lane, drive, &current_run, &sources, &execution, recovery).await
        }
    }
}

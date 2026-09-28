//! Tool-batch source reads and staged-outcome placement, ported from
//! upstream `src/harness/runtime/drive/tool-placement.ts`.
//!
//! `read_tool_batch_source` fetches the batch's assistant entry and the
//! tool-call blocks its calls point at. `materialize_ready` places the
//! consecutive `outcome_ready` prefix from the batch's first non-completed
//! call: each staged result announces `message_start`/`message_end`, one
//! settling transaction commits the entries with their usage rows (usage ids
//! minted before the transaction), introduced tool names fold into the lane
//! configuration, and a completed batch lands its checkpoint continuation —
//! `may_finish` when every call terminated, `need_assistant` otherwise — and
//! the batch `turn_end` carrying the full source-order tool results.
//! Placement settles through `settleOperation`, so a batch places its staged
//! results even under requested cancellation.
//!
//! The batch call's phase field is private to the session types module, so
//! rebuilding a call in a new phase (`tool_call_with_status`) round-trips
//! the wire shape — the statement upstream's object spread makes.

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::Message;
use pi_ai::types::ToolResultMessage;

use crate::harness::agent_harness::ConfigUpdateKind;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::LaneConfigUpdate;
use crate::harness::agent_harness::global_event;
use crate::harness::agent_harness::lane_scoped_event;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::EventsFn;
use crate::harness::runtime::types::LaneCommand;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::LanePatch;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::commit::insert_entry;
use crate::harness::session::commit::insert_usage;
use crate::harness::session::types::CheckpointData;
use crate::harness::session::types::CheckpointOperation;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::Entry;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::ToolBatch;
use crate::harness::session::types::ToolCall;
use crate::harness::session::types::ToolCallStatus;
use crate::harness::session::types::ToolsOperation;
use crate::harness::session::types::UsageRow;
use crate::harness::session::types::UsageWriteRow;
use crate::harness::session::types::operation_scope_of;
use crate::harness::session::values::StoredValue;
use crate::harness::session::values::ValueDeleteWrite;
use crate::harness::session::values::Write;
use crate::harness::session::values::branch_tip;
use crate::harness::session::values::delete_value_write;
use crate::harness::session::values::lane_config;
use crate::harness::session::values::operation_tool_args_prefix;
use crate::harness::session::values::pending_entry;
use crate::harness::session::values::set_value_write;
use crate::types::AgentMessage;
use crate::types::AgentToolCall;

/// The assistant message a tool batch runs from and its tool-call blocks by
/// source index, upstream's `ToolBatchSource`. A call's `sourceIndex` is the
/// zero-based index in the assistant message's complete content array (text
/// and thinking blocks included), not a filtered tool-call ordinal.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ToolBatchSource {
    /// The batch's assistant entry message.
    pub assistant: AssistantMessage,
    /// The tool-call blocks, keyed by `sourceIndex`.
    pub calls: BTreeMap<u64, AgentToolCall>,
}

/// Reads one batch's durable source, upstream's `readToolBatchSource`: the
/// assistant entry the batch runs from plus each call's tool-call block.
///
/// # Errors
/// The [`SessionError::Invariant`] `Tool batch assistant entry is invalid`
/// when the batch's assistant entry is missing, not a message entry, or not
/// an assistant message; the invariant
/// `Tool call source index {sourceIndex} does not name a tool-call block`
/// when a call's source index misses its tool-call block; the entry read's
/// storage error.
pub(crate) async fn read_tool_batch_source(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    batch: &ToolBatch,
) -> Result<ToolBatchSource, LaneError> {
    let assistant_entry_id = batch.assistant_entry_id.clone();
    let calls = batch.calls.clone();
    lane.command::<ToolBatchSource, _>(
        move |_state, reader, context| {
            let assistant_entry_id = assistant_entry_id.clone();
            let calls = calls.clone();
            Box::pin(async move {
                let entries = reader
                    .get_entries(vec![assistant_entry_id.clone()], &context)
                    .await
                    .map_err(lane_error)?;
                let Some(Entry::Message { body, .. }) = entries.get(&assistant_entry_id) else {
                    return Err(lane_error(SessionError::Invariant(
                        "Tool batch assistant entry is invalid".to_owned(),
                    )));
                };
                let AgentMessage::Standard(Message::Assistant(assistant)) = &body.message else {
                    return Err(lane_error(SessionError::Invariant(
                        "Tool batch assistant entry is invalid".to_owned(),
                    )));
                };
                let mut sources: BTreeMap<u64, AgentToolCall> = BTreeMap::new();
                for call in &calls {
                    let block = usize::try_from(call.source_index)
                        .ok()
                        .and_then(|index| assistant.content.get(index));
                    let Some(AssistantBlock::ToolCall(block)) = block else {
                        return Err(lane_error(SessionError::Invariant(format!(
                            "Tool call source index {} does not name a tool-call block",
                            call.source_index
                        ))));
                    };
                    sources.insert(call.source_index, block.clone());
                }
                Ok(LaneCommand::Return {
                    result: ToolBatchSource {
                        assistant: assistant.clone(),
                        calls: sources,
                    },
                })
            })
        },
        &drive.context,
    )
    .await
}

/// One staged outcome ready to place, upstream's `PlacementItem`: the batch
/// call in its `outcome_ready` phase and the staged tool-result message.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PlacementItem {
    /// The placed call.
    pub call: ToolCall,
    /// The staged tool result.
    pub message: ToolResultMessage,
}

/// The placement read, upstream's `PlacementRead`: the consecutive
/// `outcome_ready` items from the batch's first non-completed call, plus the
/// batch's full tool-result list in source order (wire `turnResults`) when
/// the ready run reaches the batch end.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PlacementRead {
    /// The staged results to place.
    pub items: Vec<PlacementItem>,
    /// The batch's full tool results when the batch completes.
    pub turn_results: Option<Vec<ToolResultMessage>>,
}

/// The tool-call block one batch call runs, upstream's `toolCallFor`.
///
/// # Errors
/// The [`SessionError::Invariant`] `Tool call source index {sourceIndex} is
/// invalid` when the source carries no block at the call's source index.
pub(crate) fn tool_call_for(
    sources: &ToolBatchSource,
    call: &ToolCall,
) -> Result<AgentToolCall, LaneError> {
    sources
        .calls
        .get(&call.source_index)
        .cloned()
        .ok_or_else(|| {
            lane_error(SessionError::Invariant(format!(
                "Tool call source index {} is invalid",
                call.source_index
            )))
        })
}

/// Rebuilds one tools leaf over a new batch, upstream's `withToolBatch` —
/// the uniform scope carries over, the batch replaces.
#[must_use]
pub(crate) fn with_tool_batch(run: ToolsOperation, batch: ToolBatch) -> ToolsOperation {
    ToolsOperation {
        scope: run.scope,
        batch,
    }
}

/// Reads the staged outcomes the batch's first non-completed index admits,
/// upstream's `readPlacement`.
///
/// `None` when the operation is not a tool batch, every call completed, or
/// the first non-completed call is not `outcome_ready`. Otherwise the
/// consecutive `outcome_ready` run from that index, and — when the run
/// reaches the batch end — the batch's full tool-result list in source order,
/// staged results and completed entries combined.
///
/// # Errors
/// The [`SessionError::Invariant`] `Tool call {resultEntryId} is missing its
/// staged result` when a staged pending entry is absent, malformed, not a
/// message, or not a tool result; `Tool call {resultEntryId} has a mismatched
/// staged result` when the staged result answers a different tool call;
/// `Tool call source index {sourceIndex} is invalid` and
/// `Completed tool call {resultEntryId} is missing its result entry` when the
/// source or the completed entries cannot serve the read; the storage errors.
async fn read_placement(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    sources: &ToolBatchSource,
) -> Result<Option<PlacementRead>, LaneError> {
    let sources = sources.clone();
    lane.command::<Option<PlacementRead>, _>(
        move |state, reader, context| {
            let sources = sources.clone();
            Box::pin(async move {
                let Some(operation) = state.operation else {
                    return Ok(LaneCommand::Return { result: None });
                };
                let OperationState::Tools(current) = operation.state else {
                    return Ok(LaneCommand::Return { result: None });
                };
                let Some(mut first) =
                    current.batch.calls.iter().position(|call| {
                        !matches!(call.status(), ToolCallStatus::Completed { .. })
                    })
                else {
                    return Ok(LaneCommand::Return { result: None });
                };
                let mut ready: Vec<ToolCall> = Vec::new();
                while first < current.batch.calls.len() {
                    let call = &current.batch.calls[first];
                    if !matches!(call.status(), ToolCallStatus::OutcomeReady { .. }) {
                        break;
                    }
                    ready.push(call.clone());
                    first += 1;
                }
                if ready.is_empty() {
                    return Ok(LaneCommand::Return { result: None });
                }

                let mut items: Vec<PlacementItem> = Vec::with_capacity(ready.len());
                for call in &ready {
                    let stored = reader
                        .get_value(&pending_entry(&call.result_entry_id).address, &context)
                        .await
                        .map_err(lane_error)?;
                    let message = staged_tool_result(stored, &call.result_entry_id)?;
                    let source = tool_call_for(&sources, call)?;
                    if message.tool_call_id != source.id || message.tool_name != source.name {
                        return Err(lane_error(SessionError::Invariant(format!(
                            "Tool call {} has a mismatched staged result",
                            call.result_entry_id
                        ))));
                    }
                    items.push(PlacementItem {
                        call: call.clone(),
                        message,
                    });
                }

                let turn_results: Option<Vec<ToolResultMessage>> = if first
                    == current.batch.calls.len()
                {
                    let placed_ids: Vec<String> = current
                        .batch
                        .calls
                        .iter()
                        .filter(|call| matches!(call.status(), ToolCallStatus::Completed { .. }))
                        .map(|call| call.result_entry_id.clone())
                        .collect();
                    let placed = reader
                        .get_entries(placed_ids, &context)
                        .await
                        .map_err(lane_error)?;
                    let staged: BTreeMap<String, ToolResultMessage> = items
                        .iter()
                        .map(|item| (item.call.result_entry_id.clone(), item.message.clone()))
                        .collect();
                    let mut results: Vec<ToolResultMessage> =
                        Vec::with_capacity(current.batch.calls.len());
                    for call in &current.batch.calls {
                        let message = staged
                            .get(&call.result_entry_id)
                            .cloned()
                            .or_else(|| completed_tool_result(placed.get(&call.result_entry_id)));
                        let Some(message) = message else {
                            return Err(lane_error(SessionError::Invariant(format!(
                                "Completed tool call {} is missing its result entry",
                                call.result_entry_id
                            ))));
                        };
                        results.push(message);
                    }
                    Some(results)
                } else {
                    None
                };
                Ok(LaneCommand::Return {
                    result: Some(PlacementRead {
                        items,
                        turn_results,
                    }),
                })
            })
        },
        &drive.context,
    )
    .await
}

/// The staged tool result one pending entry carries, upstream's
/// `isToolResultMessage` type guard over the stored pending payload: a
/// message pending entry whose payload is a tool result.
///
/// # Errors
/// The [`SessionError::Invariant`] `Tool call {resultEntryId} is missing its
/// staged result` when the entry is absent, its payload malformed, not a
/// message, or not a tool result.
fn staged_tool_result(
    stored: Option<StoredValue>,
    result_entry_id: &str,
) -> Result<ToolResultMessage, LaneError> {
    let missing = || {
        lane_error(SessionError::Invariant(format!(
            "Tool call {result_entry_id} is missing its staged result"
        )))
    };
    let Some(stored) = stored else {
        return Err(missing());
    };
    let Ok(pending) = serde_json::from_value::<PendingEntry>(stored.value) else {
        return Err(missing());
    };
    let PendingEntry::Message { payload } = pending else {
        return Err(missing());
    };
    let AgentMessage::Standard(Message::ToolResult(message)) = *payload else {
        return Err(missing());
    };
    Ok(message)
}

/// The tool result one completed message entry carries, upstream's
/// `entry?.type === "message" && isToolResultMessage(entry.message)` lookup:
/// `None` for every other entry or message role.
fn completed_tool_result(entry: Option<&Entry>) -> Option<ToolResultMessage> {
    let Entry::Message { body, .. } = entry? else {
        return None;
    };
    let AgentMessage::Standard(Message::ToolResult(message)) = &body.message else {
        return None;
    };
    Some(message.clone())
}

/// The terminate hint a staged `outcome_ready` call carries, upstream's
/// `item.call.terminate` read on the placement item's phase. `readPlacement`
/// stages only `outcome_ready` calls, whose phase always carries the hint;
/// the `false` fallback keeps the total when the phase invariant breaches.
const fn staged_terminate(call: &ToolCall) -> bool {
    let ToolCallStatus::OutcomeReady { terminate } = call.status() else {
        return false;
    };
    *terminate
}

/// Rebuilds one batch call in a new phase, upstream's `{ ...call, status }`
/// object spread. The phase field is private to the session types module, so
/// the spread restates through the wire shape both sides already serialize;
/// an in-memory call round-trips by construction.
pub(crate) fn tool_call_with_status(call: &ToolCall, status: ToolCallStatus) -> ToolCall {
    let mut wire = match serde_json::to_value(call) {
        Ok(value) => value,
        Err(error) => unreachable!("the phased tool call serializes: {error}"),
    };
    {
        let Some(object) = wire.as_object_mut() else {
            unreachable!("the phased tool call serializes to an object")
        };
        // The flattened phase rides the object as `status` plus its variant
        // fields; drop the outgoing phase's fields before merging the next.
        object.remove("status");
        object.remove("replay");
        object.remove("terminate");
        let phase = match serde_json::to_value(status) {
            Ok(value) => value,
            Err(error) => unreachable!("the tool-call phase serializes: {error}"),
        };
        let Some(phase) = phase.as_object() else {
            unreachable!("the tool-call phase serializes to an object")
        };
        for (key, value) in phase {
            object.insert(key.clone(), value.clone());
        }
    }
    match serde_json::from_value(wire) {
        Ok(call) => call,
        Err(error) => unreachable!("the rebuilt tool call round-trips: {error}"),
    }
}

/// Commits one placement read, upstream's `commitPlacement`: the entries and
/// usage rows in source order (each usage row follows its entry's write so
/// the events interleave the same way), the introduced tool names folded into
/// the lane configuration, the branch tip on the last placed entry, and the
/// completed batch's checkpoint continuation plus its persisted
/// tool-argument cleanup. The usage ids mint before the transaction; ids
/// minted for items whose settle never commits leave an unused gap.
///
/// # Errors
/// The [`SessionError::Invariant`] when the durable operation is no longer a
/// tool batch or the completed checkpoint has no branch tip to trigger from;
/// the configuration and branch-tip write serialization failures; the
/// tool-argument scan's storage error.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single commitPlacement method"
)]
async fn commit_placement(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    capability: &ToolsOperation,
    read: &PlacementRead,
) -> Result<bool, LaneError> {
    let usage_ids: Vec<Option<String>> = read
        .items
        .iter()
        .map(|item| {
            item.message
                .usage
                .is_some()
                .then(|| lane.session().id_generator().next(None))
        })
        .collect();
    let read = read.clone();
    let operation_id = drive.operation_id.clone();
    let turn_id = capability.batch.turn_id.clone();
    let lane_name = lane.name().to_owned();
    lane.settle_operation::<bool, _>(
        move |state, current, _meta, reader, context| {
            let current = current.clone();
            let read = read.clone();
            let usage_ids = usage_ids.clone();
            let operation_id = operation_id.clone();
            let turn_id = turn_id.clone();
            let lane_name = lane_name.clone();
            Box::pin(async move {
                let OperationState::Tools(run) = &current else {
                    return Err(lane_error(SessionError::Invariant(
                        "Settled operation is not a tool batch".to_owned(),
                    )));
                };
                let mut writes: Vec<Write> = Vec::new();
                let mut event_entries: Vec<(NewEntry, usize)> = Vec::new();
                let mut event_usage: Vec<(UsageWriteRow, usize)> = Vec::new();
                let mut parent_id: Option<String> = state.tip_id.clone();
                for (index, item) in read.items.iter().enumerate() {
                    let terminate = staged_terminate(&item.call);
                    let entry = NewEntry::Message {
                        id: item.call.result_entry_id.clone(),
                        parent_id: parent_id.clone(),
                        body: Box::new(MessageEntry {
                            message: AgentMessage::Standard(Message::ToolResult(
                                item.message.clone(),
                            )),
                            terminate: terminate.then_some(true),
                        }),
                    };
                    event_entries.push((entry.clone(), writes.len()));
                    writes.push(Write::Entry(Box::new(insert_entry(entry))));
                    writes.push(delete_value_write(&pending_entry(
                        &item.call.result_entry_id,
                    )));
                    if let (Some(usage_id), Some(usage)) = (&usage_ids[index], &item.message.usage)
                    {
                        let row = UsageWriteRow {
                            id: usage_id.clone(),
                            usage: *usage,
                            entry_id: Some(item.call.result_entry_id.clone()),
                            adjustment: false,
                            details: None,
                        };
                        event_usage.push((row.clone(), writes.len()));
                        writes.push(Write::Usage(insert_usage(row)));
                    }
                    parent_id = Some(item.call.result_entry_id.clone());
                }

                let completed_calls: Vec<ToolCall> = run
                    .batch
                    .calls
                    .iter()
                    .map(|call| {
                        let matched = read.items.iter().find(|candidate| {
                            candidate.call.source_index == call.source_index
                                && candidate.call.result_entry_id == call.result_entry_id
                        });
                        matched.map_or_else(
                            || call.clone(),
                            |item| {
                                tool_call_with_status(
                                    call,
                                    ToolCallStatus::Completed {
                                        terminate: staged_terminate(&item.call),
                                    },
                                )
                            },
                        )
                    })
                    .collect();
                let complete = completed_calls
                    .iter()
                    .all(|call| matches!(call.status(), ToolCallStatus::Completed { .. }));
                let mut next_configuration = state.configuration.clone();
                let mut added_names: Vec<String> = Vec::new();
                for item in &read.items {
                    for name in item.message.added_tool_names.iter().flatten() {
                        if !next_configuration.active_tool_names.contains(name)
                            && !added_names.contains(name)
                        {
                            added_names.push(name.clone());
                        }
                    }
                }
                if !added_names.is_empty() {
                    next_configuration
                        .active_tool_names
                        .extend(added_names.iter().cloned());
                    writes.push(
                        set_value_write(&lane_config(&lane_name), next_configuration.clone())
                            .map_err(lane_error)?,
                    );
                }
                writes.push(
                    set_value_write(&branch_tip(&lane_name), parent_id.clone())
                        .map_err(lane_error)?,
                );

                let next_state = if complete {
                    let all_terminate = completed_calls.iter().all(|call| {
                        matches!(call.status(), ToolCallStatus::Completed { terminate: true })
                    });
                    let trigger_entry_id = parent_id.clone().ok_or_else(|| {
                        lane_error(SessionError::Invariant(
                            "Completed tool batch has no branch tip".to_owned(),
                        ))
                    })?;
                    let args = reader
                        .scan_values(
                            &operation_tool_args_prefix(&operation_id, Some(&turn_id)).address,
                            &context,
                        )
                        .await
                        .map_err(lane_error)?;
                    writes.extend(args.iter().map(delete_stored_value));
                    OperationState::Checkpoint(CheckpointOperation {
                        scope: operation_scope_of(&current),
                        checkpoint: CheckpointData {
                            continuation: if all_terminate {
                                Continuation::MayFinish {
                                    include_final_assistant: false,
                                }
                            } else {
                                Continuation::NeedAssistant {
                                    overflow_recovery_used: false,
                                }
                            },
                            trigger_entry_id,
                        },
                    })
                } else {
                    OperationState::Tools(with_tool_batch(
                        run.clone(),
                        ToolBatch {
                            calls: completed_calls,
                            ..run.batch.clone()
                        },
                    ))
                };

                let previous_active = state.configuration.active_tool_names.clone();
                let next_active = next_configuration.active_tool_names.clone();
                let event_lane = lane_name.clone();
                let events: EventsFn = Arc::new(move |commit: &CommitResult| {
                    let mut events: Vec<HarnessEvent> = Vec::new();
                    for (entry, seq_index) in &event_entries {
                        let seq = commit.seqs.get(*seq_index).map_or_else(
                            || unreachable!("commit carries one sequence per write"),
                            |seq| *seq,
                        );
                        let materialized = entry.clone().materialize(seq, commit.timestamp);
                        events.push(lane_scoped_event(
                            &event_lane,
                            false,
                            "entry_added",
                            HarnessEventPayload::EntryAdded {
                                entry: materialized,
                            },
                        ));
                        let usage = event_usage
                            .iter()
                            .find(|(row, _)| row.entry_id.as_deref() == Some(entry.id()));
                        if let Some((row, usage_seq_index)) = usage {
                            let seq = commit.seqs.get(*usage_seq_index).map_or_else(
                                || unreachable!("commit carries one sequence per write"),
                                |seq| *seq,
                            );
                            let materialized = UsageRow {
                                seq,
                                id: row.id.clone(),
                                usage: row.usage,
                                entry_id: row.entry_id.clone(),
                                adjustment: row.adjustment,
                                details: row.details.clone(),
                            };
                            events.push(global_event(
                                "usage",
                                HarnessEventPayload::Usage {
                                    lane: event_lane.clone(),
                                    row: materialized,
                                    totals: commit.stats.usage,
                                },
                            ));
                        }
                    }
                    if !added_names.is_empty() {
                        events.push(lane_scoped_event(
                            &event_lane,
                            false,
                            "config_update",
                            HarnessEventPayload::ConfigUpdate {
                                property: ConfigUpdateKind::Lane(LaneConfigUpdate::ActiveTools {
                                    value: next_active.clone(),
                                    previous: previous_active.clone(),
                                }),
                            },
                        ));
                    }
                    events
                });
                Ok(OperationCommand::Commit {
                    writes,
                    operation_state: next_state,
                    lane: Some(LanePatch {
                        tip_id: Some(parent_id.clone()),
                        configuration: Some(next_configuration),
                        inbox: None,
                    }),
                    materialize: Arc::new(move |_commit: &CommitResult| complete),
                    events: Some(events),
                })
            })
        },
        &drive.context,
    )
    .await
}

/// The value-delete write one scanned tool-argument value removes, upstream's
/// `({ address }) => deleteValue(address)` projection (the stored view
/// carries the address as its `namespace`/`key` fields).
pub(crate) fn delete_stored_value(value: &StoredValue) -> Write {
    Write::ValueDelete(ValueDeleteWrite {
        kind: "value".to_owned(),
        op: "delete".to_owned(),
        namespace: value.namespace.clone(),
        key: value.key.clone(),
    })
}

/// Places the staged outcomes the batch's first non-completed index admits,
/// upstream's `materializeReady`.
///
/// The staged results announce `message_start`/`message_end` pairs first —
/// the `recovery` flag rides both while recovering — then the placement
/// transaction commits, and a completed batch emits its `turn_end` carrying
/// the assistant message and the full tool-result list in source order.
///
/// # Errors
/// [`read_placement`]'s and [`commit_placement`]'s errors; the announcement
/// and `turn_end` deliveries' failures.
pub(crate) async fn materialize_ready(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    capability: &ToolsOperation,
    sources: &ToolBatchSource,
    recovery: bool,
) -> Result<(), LaneError> {
    let Some(read) = read_placement(lane, drive, sources).await? else {
        return Ok(());
    };
    let mut announcements: Vec<HarnessEvent> = Vec::with_capacity(read.items.len() * 2);
    for item in &read.items {
        let message = AgentMessage::Standard(Message::ToolResult(item.message.clone()));
        announcements.push(lane_scoped_event(
            lane.name(),
            recovery,
            "message_start",
            HarnessEventPayload::MessageStart {
                run_id: Some(drive.operation_id.clone()),
                message: message.clone(),
            },
        ));
        announcements.push(lane_scoped_event(
            lane.name(),
            recovery,
            "message_end",
            HarnessEventPayload::MessageEnd {
                run_id: Some(drive.operation_id.clone()),
                message,
                entry_id: Some(item.call.result_entry_id.clone()),
            },
        ));
    }
    (lane.emit_batch())(announcements, drive.context.clone()).await?;
    let complete = commit_placement(lane, drive, capability, &read).await?;
    if complete && let Some(turn_results) = read.turn_results.clone() {
        let turn_end = lane_scoped_event(
            lane.name(),
            recovery,
            "turn_end",
            HarnessEventPayload::TurnEnd {
                run_id: drive.operation_id.clone(),
                turn_id: capability.batch.turn_id.clone(),
                message: sources.assistant.clone(),
                tool_results: turn_results,
            },
        );
        (lane.emit_batch())(vec![turn_end], drive.context.clone()).await?;
    }
    Ok(())
}

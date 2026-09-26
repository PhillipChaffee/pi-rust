//! The lane snapshot reducer, ported from upstream
//! `src/harness/runtime/reducer.ts`.
//!
//! Upstream mutates the snapshot in place and reports whether navigation
//! completion demands a fresh snapshot; the port takes the snapshot mutable
//! and returns the same `rebase` signal. The event/lane filter, the tool
//! upserts, and the compaction-segment splice restate 1:1.

use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::LaneSnapshot;
use crate::harness::agent_harness::LaneSnapshotTool;
use crate::harness::agent_harness::LiveOperationView;
use crate::harness::agent_harness::OperationStatus;

/// The reduction signal a harness event carries, upstream's
/// `LaneSnapshotReduction = "rebase" | undefined`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LaneSnapshotReduction {
    /// The event applied in place.
    #[default]
    Applied,
    /// Navigation completion; the caller must resnapshot.
    Rebase,
}

fn matching_operation<'a>(snapshot: &'a mut LaneSnapshot, operation_id: &str) -> Option<&'a mut LiveOperationView> {
    let matches = snapshot
        .operation
        .as_ref()
        .is_some_and(|operation| operation.id == operation_id);
    if matches {
        return snapshot.operation.as_mut();
    }
    None
}

/// Applies one harness event to a lane snapshot, upstream's
/// `reduceLaneSnapshot`. Navigation completion requires a fresh snapshot.
pub fn reduce_lane_snapshot(snapshot: &mut LaneSnapshot, event: &HarnessEvent) -> LaneSnapshotReduction {
    let lane_mismatch = match &event.lane {
        Some(event_lane) => event_lane != &snapshot.lane && event.event_type() != crate::harness::agent_harness::HarnessEventType::Usage,
        None => false,
    };
    if lane_mismatch {
        return LaneSnapshotReduction::Applied;
    }
    match &event.payload {
        HarnessEventPayload::RunStart { run_id, started_at } => {
            snapshot.operation = Some(LiveOperationView {
                id: run_id.clone(),
                kind: crate::harness::session::types::OperationKind::Run,
                started_at: *started_at,
                from_tip_id: snapshot.tip_id.clone(),
                status: OperationStatus::Open,
                retry: None,
                deferred: None,
                streaming_message: None,
                running_tools: Vec::new(),
            });
        }
        HarnessEventPayload::CompactionStart { run_id, started_at, .. } => {
            if snapshot.operation.is_some() {
                return LaneSnapshotReduction::Applied;
            }
            snapshot.operation = Some(LiveOperationView {
                id: run_id.clone(),
                kind: crate::harness::session::types::OperationKind::Compaction,
                started_at: *started_at,
                from_tip_id: snapshot.tip_id.clone(),
                status: OperationStatus::Open,
                retry: None,
                deferred: None,
                streaming_message: None,
                running_tools: Vec::new(),
            });
        }
        HarnessEventPayload::NavigationStart { run_id, started_at, .. } => {
            snapshot.operation = Some(LiveOperationView {
                id: run_id.clone(),
                kind: crate::harness::session::types::OperationKind::Navigation,
                started_at: *started_at,
                from_tip_id: snapshot.tip_id.clone(),
                status: OperationStatus::Open,
                retry: None,
                deferred: None,
                streaming_message: None,
                running_tools: Vec::new(),
            });
        }
        HarnessEventPayload::OperationAbort { operation_id, .. } => {
            if let Some(operation) = matching_operation(snapshot, operation_id) {
                operation.status = OperationStatus::Aborting;
            }
        }
        HarnessEventPayload::RunResume { run_id } => {
            if let Some(operation) = matching_operation(snapshot, run_id) {
                operation.deferred = None;
            }
        }
        HarnessEventPayload::RunSuspend { run_id, deferred, poll } => {
            let Some(operation) = matching_operation(snapshot, run_id) else {
                return LaneSnapshotReduction::Applied;
            };
            operation.streaming_message = None;
            operation.deferred = Some(crate::harness::agent_harness::DeferredView {
                handle: deferred.clone(),
                poll: *poll,
            });
        }
        HarnessEventPayload::RetryScheduled { run_id, attempt, max_attempts, not_before, .. } => {
            if let Some(operation) = matching_operation(snapshot, run_id) {
                operation.retry = Some(crate::harness::agent_harness::RetryView {
                    attempt: *attempt,
                    max_attempts: *max_attempts,
                    next_attempt_at: *not_before,
                });
            }
        }
        HarnessEventPayload::RetryStart { run_id, .. } | HarnessEventPayload::RetryEnd { run_id, .. } => {
            if let Some(operation) = matching_operation(snapshot, run_id) {
                operation.retry = None;
            }
        }
        HarnessEventPayload::MessageStart { run_id: Some(run_id), message } => {
            let streaming = match message {
                crate::types::AgentMessage::Standard(pi_ai::types::Message::Assistant(assistant))
                    if assistant.stop_reason == pi_ai::types::StopReason::Pending =>
                {
                    assistant.clone()
                }
                _ => return LaneSnapshotReduction::Applied,
            };
            if let Some(operation) = matching_operation(snapshot, run_id) {
                operation.streaming_message = Some(streaming);
            }
        }
        HarnessEventPayload::MessageStart { run_id: None, .. } => {}
        HarnessEventPayload::MessageUpdate { run_id, message, .. } => {
            let streaming = match message.as_ref() {
                crate::types::AgentMessage::Standard(pi_ai::types::Message::Assistant(assistant)) => {
                    assistant.clone()
                }
                _ => return LaneSnapshotReduction::Applied,
            };
            if let Some(operation) = matching_operation(snapshot, run_id) {
                operation.streaming_message = Some(streaming);
            }
        }
        HarnessEventPayload::MessageEnd { run_id: Some(run_id), .. } => {
            if let Some(operation) = matching_operation(snapshot, run_id) {
                operation.streaming_message = None;
            }
        }
        HarnessEventPayload::MessageEnd { run_id: None, .. } => {}
        HarnessEventPayload::ToolStart { run_id, tool_call_id, tool_name, args, .. } => {
            let Some(operation) = matching_operation(snapshot, run_id) else {
                return LaneSnapshotReduction::Applied;
            };
            upsert_tool(
                operation,
                LaneSnapshotTool::Running {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    args: args.clone(),
                    result: None,
                },
            );
        }
        HarnessEventPayload::ToolUpdate { run_id, tool_call_id, partial_result, .. } => {
            let Some(operation) = matching_operation(snapshot, run_id) else {
                return LaneSnapshotReduction::Applied;
            };
            if let Some(tool) = operation
                .running_tools
                .iter_mut()
                .find(|candidate| tool_call_id_of(candidate) == Some(tool_call_id.as_str()))
            {
                if let LaneSnapshotTool::Running { result, .. } = tool {
                    *result = Some(partial_result.clone());
                }
            }
        }
        HarnessEventPayload::ToolEnd { run_id, tool_call_id, tool_name, result, is_error, .. } => {
            let Some(operation) = matching_operation(snapshot, run_id) else {
                return LaneSnapshotReduction::Applied;
            };
            let Some(index) = operation
                .running_tools
                .iter()
                .position(|candidate| tool_call_id_of(candidate) == Some(tool_call_id.as_str()))
            else {
                return LaneSnapshotReduction::Applied;
            };
            let current = &operation.running_tools[index];
            let args = args_of(current).clone();
            operation.running_tools[index] = LaneSnapshotTool::Settled {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                args,
                result: result.clone(),
                is_error: *is_error,
            };
        }
        HarnessEventPayload::EntryAdded { entry } => {
            if let crate::harness::session::types::Entry::Message { body, .. } = &entry {
                if let crate::types::AgentMessage::Standard(pi_ai::types::Message::ToolResult(tool_result)) =
                    &body.message
                {
                    if let Some(operation) = &mut snapshot.operation {
                        let tool_call_id = &tool_result.tool_call_id;
                        if let Some(index) = operation
                            .running_tools
                            .iter()
                            .position(|candidate| tool_call_id_of(candidate) == Some(tool_call_id.as_str()))
                        {
                            operation.running_tools.remove(index);
                        }
                    }
                }
            }
            if entry.entry_type() == crate::harness::session::types::EntryType::Compaction {
                snapshot.transcript.clear();
                snapshot.transcript.push(entry.clone());
            } else {
                snapshot.transcript.push(entry.clone());
            }
            snapshot.tip_id = Some(entry.id().to_owned());
            if entry.entry_type() == crate::harness::session::types::EntryType::Message {
                snapshot.stats.message_count += 1;
            }
        }
        HarnessEventPayload::QueueUpdate { queues } => {
            snapshot.queues = queues.clone();
        }
        HarnessEventPayload::Usage { totals, .. } => {
            snapshot.stats.usage = totals.clone();
        }
        HarnessEventPayload::ConfigUpdate { property } => {
            let event_lane = event.lane.as_deref();
            match property {
                crate::harness::agent_harness::ConfigUpdateKind::Lane(lane_update) => {
                    if event_lane != Some(snapshot.lane.as_str()) {
                        return LaneSnapshotReduction::Applied;
                    }
                    match lane_update {
                        crate::harness::agent_harness::LaneConfigUpdate::Model { value, .. } => {
                            snapshot.configuration.model = value.clone();
                        }
                        crate::harness::agent_harness::LaneConfigUpdate::ThinkingLevel { value, .. } => {
                            snapshot.configuration.thinking_level = *value;
                        }
                        crate::harness::agent_harness::LaneConfigUpdate::ActiveTools { value, .. } => {
                            snapshot.configuration.active_tool_names = value.clone();
                        }
                    }
                }
                crate::harness::agent_harness::ConfigUpdateKind::Global(_) => {}
            }
        }
        HarnessEventPayload::RunEnd { run_id, status, tip_id, ended_at, from_tip_id, .. } => {
            let Some(operation) = matching_operation(snapshot, run_id) else {
                return LaneSnapshotReduction::Applied;
            };
            if operation.kind != crate::harness::session::types::OperationKind::Run {
                return LaneSnapshotReduction::Applied;
            }
            let record = run_end_record(status, from_tip_id, tip_id, operation, *ended_at);
            snapshot.last_result = Some(record);
            snapshot.operation = None;
            snapshot.tip_id = tip_id.clone();
        }
        HarnessEventPayload::CompactionEnd { run_id, status, ended_at, .. } => {
            let tip_id = snapshot.tip_id.clone();
            let Some(operation) = matching_operation(snapshot, run_id) else {
                return LaneSnapshotReduction::Applied;
            };
            if operation.kind != crate::harness::session::types::OperationKind::Compaction {
                return LaneSnapshotReduction::Applied;
            }
            let record = compaction_end_record(operation, status, *ended_at, &tip_id);
            snapshot.last_result = Some(record);
            snapshot.operation = None;
        }
        HarnessEventPayload::NavigationEnd { .. } => return LaneSnapshotReduction::Rebase,
        HarnessEventPayload::Fault { .. } => {
            snapshot.faulted = true;
        }
        HarnessEventPayload::HandlerError { .. }
        | HarnessEventPayload::TurnStart { .. }
        | HarnessEventPayload::TurnEnd { .. }
        | HarnessEventPayload::ValueUpdate { .. }
        | HarnessEventPayload::LaneCreated { .. } => {}
    }
    LaneSnapshotReduction::Applied
}

fn upsert_tool(operation: &mut LiveOperationView, tool: LaneSnapshotTool) {
    let tool_call_id = tool_call_id_of(&tool).map(str::to_owned);
    match operation
        .running_tools
        .iter()
        .position(|candidate| tool_call_id_of(candidate) == tool_call_id.as_deref())
    {
        Some(index) => operation.running_tools[index] = tool,
        None => operation.running_tools.push(tool),
    }
}

fn tool_call_id_of(tool: &LaneSnapshotTool) -> Option<&str> {
    match tool {
        LaneSnapshotTool::Running { tool_call_id, .. } | LaneSnapshotTool::Settled { tool_call_id, .. } => {
            Some(tool_call_id)
        }
    }
}

fn args_of(tool: &LaneSnapshotTool) -> &serde_json::Value {
    match tool {
        LaneSnapshotTool::Running { args, .. } | LaneSnapshotTool::Settled { args, .. } => args,
    }
}

fn run_end_record(
    status: &crate::harness::agent_harness::RunEndStatus,
    from_tip_id: &Option<String>,
    tip_id: &Option<String>,
    operation: &LiveOperationView,
    ended_at: i64,
) -> crate::harness::session::types::OperationResultRecord {
    use crate::harness::agent_harness::RunEndStatus;
    let (terminal_status, error) = match status {
        RunEndStatus::Completed => (crate::harness::session::types::TerminalStatus::Completed, None),
        RunEndStatus::Aborted => (crate::harness::session::types::TerminalStatus::Aborted, None),
        RunEndStatus::Failed { error } => (
            crate::harness::session::types::TerminalStatus::Failed,
            Some(error.clone()),
        ),
    };
    crate::harness::session::types::OperationResultRecord {
        operation_id: operation.id.clone(),
        kind: crate::harness::session::types::OperationKind::Run,
        status: terminal_status,
        error,
        from_tip_id: from_tip_id.clone(),
        tip_id: tip_id.clone(),
        started_at: operation.started_at,
        ended_at,
    }
}

fn compaction_end_record(
    operation: &LiveOperationView,
    status: &crate::harness::agent_harness::CompactionEndStatus,
    ended_at: i64,
    tip_id: &Option<String>,
) -> crate::harness::session::types::OperationResultRecord {
    use crate::harness::agent_harness::CompactionEndStatus;
    let (terminal_status, error) = match status {
        CompactionEndStatus::Completed { .. } => (crate::harness::session::types::TerminalStatus::Completed, None),
        CompactionEndStatus::Declined => (crate::harness::session::types::TerminalStatus::Declined, None),
        CompactionEndStatus::Aborted => (crate::harness::session::types::TerminalStatus::Aborted, None),
        CompactionEndStatus::Failed { error } => (
            crate::harness::session::types::TerminalStatus::Failed,
            Some(error.clone()),
        ),
    };
    crate::harness::session::types::OperationResultRecord {
        operation_id: operation.id.clone(),
        kind: crate::harness::session::types::OperationKind::Compaction,
        status: terminal_status,
        error,
        from_tip_id: operation.from_tip_id.clone(),
        tip_id: tip_id.clone(),
        started_at: operation.started_at,
        ended_at,
    }
}
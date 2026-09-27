//! The systematic reducer pass: every [`HarnessEventPayload`] variant folds
//! its upstream-documented effect into a hand-built snapshot, the guard
//! arms included. Upstream's `reduceLaneSnapshot` switch
//! (`src/harness/runtime/reducer.ts`) is the spec each assertion restates.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; a violated expectation panics the test by design"
)]

use super::*;
use crate::harness::session::types::ModelIdentity;
use crate::types::ThinkingLevel;

/// The pending-assistant streaming handle the suspend events carry,
/// `agent_harness/tests.rs`'s deferred wire.
fn deferred_handle() -> pi_ai::types::DeferredHandle {
    serde_json::from_value(json!({
        "provider": "provider",
        "modelId": "model",
        "api": "api",
        "id": "deferred",
        "pollAfterMs": 1000,
    }))
    .expect("deferred handle")
}

/// The assistant wire the streaming fixtures build, `lane/tests.rs`'s
/// assistant literal with the stop reason free.
fn assistant_wire(stop_reason: &str, text: &str) -> AgentMessage {
    serde_json::from_value(json!({
        "role": "assistant",
        "content": [{ "type": "text", "text": text }],
        "api": "anthropic-messages",
        "provider": "test",
        "model": "model",
        "usage": {
            "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
        },
        "stopReason": stop_reason,
        "timestamp": 1,
    }))
    .expect("assistant wire")
}

/// The user wire the guard fixtures build.
fn user_wire(text: &str) -> AgentMessage {
    serde_json::from_value(json!({ "role": "user", "content": text, "timestamp": 1 }))
        .expect("user wire")
}

/// The assistant message under a standard agent message, the streaming
/// snapshot's payload shape.
fn assistant_of(message: &AgentMessage) -> &pi_ai::types::AssistantMessage {
    match message {
        AgentMessage::Standard(Message::Assistant(assistant)) => assistant,
        other => panic!("the streaming message is an assistant: {other:?}"),
    }
}

/// The nonzero usage totals the usage events carry.
fn usage_totals(input: u64, output: u64) -> pi_ai::types::Usage {
    serde_json::from_value(json!({
        "input": input, "output": output, "cacheRead": 0, "cacheWrite": 0,
        "totalTokens": input + output,
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
    }))
    .expect("usage wire")
}

/// The usage row the usage events carry.
fn usage_row(id: &str, totals: pi_ai::types::Usage) -> crate::harness::session::types::UsageRow {
    crate::harness::session::types::UsageRow {
        seq: 1,
        id: id.to_owned(),
        usage: totals,
        entry_id: None,
        adjustment: true,
        details: None,
    }
}

fn operation_id_of(snapshot: &LaneSnapshot) -> &str {
    &snapshot.operation.as_ref().expect("the live operation").id
}

#[test]
fn folds_the_run_lifecycle_over_the_matching_operation() {
    let mut snapshot = empty_snapshot();
    snapshot.tip_id = Some("tip".to_owned());

    let reduced = reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::RunStart {
            run_id: "run".to_owned(),
            started_at: 2,
        }),
    );
    assert_eq!(reduced, LaneSnapshotReduction::Applied);
    let operation = snapshot.operation.as_ref().expect("the open operation");
    assert_eq!(operation.id, "run");
    assert_eq!(operation.kind, OperationKind::Run);
    assert_eq!(operation.started_at, 2);
    assert_eq!(operation.from_tip_id.as_deref(), Some("tip"));
    assert_eq!(operation.status, OperationStatus::Open);
    assert!(operation.retry.is_none());
    assert!(operation.deferred.is_none());
    assert!(operation.streaming_message.is_none());
    assert!(operation.running_tools.is_empty());

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::RunSuspend {
            run_id: "run".to_owned(),
            deferred: deferred_handle(),
            poll: 3,
        }),
    );
    let operation = snapshot.operation.as_ref().expect("the parked operation");
    let deferred = operation.deferred.as_ref().expect("the deferred view");
    assert_eq!(deferred.poll, 3);
    assert_eq!(deferred.handle, deferred_handle());

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::RunResume {
            run_id: "run".to_owned(),
        }),
    );
    assert!(
        snapshot
            .operation
            .as_ref()
            .expect("the resumed operation")
            .deferred
            .is_none(),
        "the resume cleared the deferred view",
    );

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::OperationAbort {
            operation_id: "run".to_owned(),
            steer: vec![user_wire("out")],
            follow_up: Vec::new(),
        }),
    );
    assert_eq!(
        snapshot
            .operation
            .as_ref()
            .expect("the aborting operation")
            .status,
        OperationStatus::Aborting,
    );

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::RunSuspend {
            run_id: "foreign".to_owned(),
            deferred: deferred_handle(),
            poll: 9,
        }),
    );
    let operation = snapshot
        .operation
        .as_ref()
        .expect("the untouched operation");
    assert!(
        operation.deferred.is_none(),
        "a foreign run's suspend parks nothing",
    );

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::OperationAbort {
            operation_id: "foreign".to_owned(),
            steer: Vec::new(),
            follow_up: Vec::new(),
        }),
    );
    assert_eq!(
        snapshot
            .operation
            .as_ref()
            .expect("the still-aborting operation")
            .status,
        OperationStatus::Aborting,
        "a foreign abort flips nothing",
    );
}

#[test]
fn schedules_and_clears_retry_views_on_the_matching_run() {
    let mut snapshot = empty_snapshot();
    snapshot.operation = Some(run_operation("run", OperationKind::Run));

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::RetryScheduled {
            run_id: "run".to_owned(),
            step: "assistant".to_owned(),
            attempt: 2,
            max_attempts: 3,
            delay_ms: 1_000,
            not_before: 20,
            error_message: "boom".to_owned(),
        }),
    );
    let retry = snapshot
        .operation
        .as_ref()
        .expect("the retrying operation")
        .retry
        .as_ref()
        .expect("the retry view");
    assert_eq!(retry.attempt, 2);
    assert_eq!(retry.max_attempts, 3);
    assert_eq!(retry.next_attempt_at, 20);

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::RetryStart {
            run_id: "run".to_owned(),
            step: "assistant".to_owned(),
            attempt: 2,
        }),
    );
    assert!(
        snapshot
            .operation
            .as_ref()
            .expect("the retrying operation")
            .retry
            .is_none(),
        "the retry start cleared the view",
    );

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::RetryScheduled {
            run_id: "run".to_owned(),
            step: "assistant".to_owned(),
            attempt: 3,
            max_attempts: 3,
            delay_ms: 1_000,
            not_before: 30,
            error_message: "boom".to_owned(),
        }),
    );
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::RetryEnd {
            run_id: "run".to_owned(),
            step: "assistant".to_owned(),
            attempt: 3,
            success: false,
            final_error: Some("gave up".to_owned()),
        }),
    );
    assert!(
        snapshot
            .operation
            .as_ref()
            .expect("the settled operation")
            .retry
            .is_none(),
        "the retry end cleared the view",
    );

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::RetryScheduled {
            run_id: "foreign".to_owned(),
            step: "assistant".to_owned(),
            attempt: 2,
            max_attempts: 3,
            delay_ms: 1_000,
            not_before: 40,
            error_message: "boom".to_owned(),
        }),
    );
    assert!(
        snapshot
            .operation
            .as_ref()
            .expect("the untouched operation")
            .retry
            .is_none(),
        "a foreign run's retry schedules nothing",
    );
}

#[expect(
    clippy::too_many_lines,
    reason = "the streaming guards and the streaming writes are one continuous fold"
)]
#[test]
fn guards_message_streaming_then_streams_over_the_matching_run() {
    let mut snapshot = empty_snapshot();
    snapshot.operation = Some(run_operation("run", OperationKind::Run));
    let pending = assistant_wire("pending", "partial");

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::MessageStart {
            run_id: Some("run".to_owned()),
            message: user_wire("hi"),
        }),
    );
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::MessageStart {
            run_id: Some("run".to_owned()),
            message: assistant_wire("stop", "done"),
        }),
    );
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::MessageStart {
            run_id: None,
            message: pending.clone(),
        }),
    );
    assert!(
        snapshot
            .operation
            .as_ref()
            .expect("the untouched operation")
            .streaming_message
            .is_none(),
        "non-assistant, settled, and run-less starts stream nothing",
    );

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::MessageStart {
            run_id: Some("run".to_owned()),
            message: pending.clone(),
        }),
    );
    assert_eq!(
        snapshot
            .operation
            .as_ref()
            .expect("the streaming operation")
            .streaming_message
            .as_ref(),
        Some(assistant_of(&pending)),
    );

    let event = pi_ai::types::AssistantMessageEvent::Start {
        partial: assistant_of(&pending).clone(),
    };
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::MessageUpdate {
            run_id: "run".to_owned(),
            message: Box::new(assistant_wire("pending", "grown")),
            event: Box::new(event.clone()),
            frame: None,
        }),
    );
    let streaming = snapshot
        .operation
        .as_ref()
        .expect("the streaming operation")
        .streaming_message
        .as_ref()
        .expect("the grown stream");
    assert_eq!(streaming.stop_reason, pi_ai::types::StopReason::Pending);

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::MessageUpdate {
            run_id: "run".to_owned(),
            message: Box::new(user_wire("hi")),
            event: Box::new(event),
            frame: None,
        }),
    );
    assert_eq!(
        snapshot
            .operation
            .as_ref()
            .expect("the streaming operation")
            .streaming_message
            .as_ref()
            .expect("the stream survived the non-assistant update")
            .stop_reason,
        pi_ai::types::StopReason::Pending,
    );

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::MessageEnd {
            run_id: Some("run".to_owned()),
            message: assistant_wire("stop", "done"),
            entry_id: Some("entry".to_owned()),
        }),
    );
    assert!(
        snapshot
            .operation
            .as_ref()
            .expect("the settled operation")
            .streaming_message
            .is_none(),
        "the message end cleared the stream",
    );

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::MessageEnd {
            run_id: None,
            message: assistant_wire("stop", "done"),
            entry_id: None,
        }),
    );
    assert_eq!(
        operation_id_of(&snapshot),
        "run",
        "a run-less end touched nothing",
    );
}

#[test]
fn ignores_tool_events_from_a_foreign_run() {
    let mut snapshot = empty_snapshot();
    snapshot.operation = Some(run_operation("run", OperationKind::Run));

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::ToolStart {
            run_id: "foreign".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call".to_owned(),
            tool_name: "tool".to_owned(),
            args: json!({}),
        }),
    );
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::ToolUpdate {
            run_id: "foreign".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call".to_owned(),
            tool_name: "tool".to_owned(),
            partial_result: tool_result("stale"),
        }),
    );
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::ToolEnd {
            run_id: "foreign".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call".to_owned(),
            tool_name: "tool".to_owned(),
            result: tool_result("stale"),
            is_error: false,
            terminate: false,
        }),
    );
    assert!(
        running_tools(&snapshot).is_empty(),
        "a foreign run's tool events touch nothing",
    );
}

#[test]
fn updates_only_running_tool_checkpoints() {
    let mut snapshot = empty_snapshot();
    snapshot.operation = Some(run_operation("run", OperationKind::Run));
    reduce_lane_snapshot(&mut snapshot, &tool_start_event(0));
    reduce_lane_snapshot(&mut snapshot, &tool_start_event(1));

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::ToolUpdate {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call-0".to_owned(),
            tool_name: "tool-0".to_owned(),
            partial_result: tool_result("checkpoint"),
        }),
    );
    assert!(
        matches!(
            running_tools(&snapshot)[0],
            LaneSnapshotTool::Running {
                result: Some(_),
                ..
            }
        ),
        "the running tool's checkpoint landed",
    );

    reduce_lane_snapshot(&mut snapshot, &tool_end_event(0));
    assert!(
        matches!(
            running_tools(&snapshot)[0],
            LaneSnapshotTool::Settled { .. }
        ),
        "call-0 settled",
    );

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::ToolUpdate {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call-0".to_owned(),
            tool_name: "tool-0".to_owned(),
            partial_result: tool_result("late"),
        }),
    );
    let settled = running_tools(&snapshot)[0].clone();
    match settled {
        LaneSnapshotTool::Settled { result, .. } => {
            assert_eq!(
                result,
                tool_result("done-0"),
                "the settled checkpoint holds"
            );
        }
        other @ LaneSnapshotTool::Running { .. } => panic!("the running tool: {other:?}"),
    }

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::ToolUpdate {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call-9".to_owned(),
            tool_name: "tool-9".to_owned(),
            partial_result: tool_result("absent"),
        }),
    );
    assert_eq!(
        running_tools(&snapshot).len(),
        2,
        "an absent call updates nothing"
    );
}

#[test]
fn folds_entry_additions_tip_transcript_and_counts() {
    let mut snapshot = empty_snapshot();

    let user_entry = Entry::Message {
        id: "user-1".to_owned(),
        parent_id: None,
        seq: 1,
        timestamp: 1,
        body: Box::new(MessageEntry {
            message: user_wire("history"),
            terminate: None,
        }),
    };
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::EntryAdded {
            entry: user_entry.clone(),
        }),
    );
    assert_eq!(snapshot.transcript, vec![user_entry]);
    assert_eq!(snapshot.tip_id.as_deref(), Some("user-1"));
    assert_eq!(snapshot.stats.message_count, 1, "the message entry counted");

    let custom_entry = Entry::Custom {
        id: "custom-1".to_owned(),
        parent_id: Some("user-1".to_owned()),
        seq: 2,
        timestamp: 2,
        body: crate::harness::session::types::CustomEntryBody {
            custom_type: "note".to_owned(),
            data: None,
        },
    };
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::EntryAdded {
            entry: custom_entry,
        }),
    );
    assert_eq!(snapshot.tip_id.as_deref(), Some("custom-1"));
    assert_eq!(
        snapshot.stats.message_count, 1,
        "a non-message entry counts nothing",
    );

    let compaction_entry = Entry::Compaction {
        id: "compact-1".to_owned(),
        parent_id: None,
        seq: 3,
        timestamp: 3,
        body: crate::harness::session::types::CompactionEntryBody {
            summary: "summary".to_owned(),
            retained_tail: Vec::new(),
            tokens_before: 0,
            details: None,
            usage: None,
            from_hook: false,
        },
    };
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::EntryAdded {
            entry: compaction_entry.clone(),
        }),
    );
    assert_eq!(
        snapshot.transcript,
        vec![compaction_entry],
        "the compaction entry spliced the transcript",
    );
    assert_eq!(snapshot.tip_id.as_deref(), Some("compact-1"));
    assert_eq!(snapshot.stats.message_count, 1);
}

#[test]
fn replaces_queues_and_usage_totals() {
    let mut snapshot = empty_snapshot();

    let queued = vec![LaneQueuedItem::Message {
        entry_id: "queued-1".to_owned(),
        kind: InboxItemKind::Steer,
        message: Box::new(user_wire("steer")),
    }];
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::QueueUpdate {
            queues: queued.clone(),
        }),
    );
    assert_eq!(snapshot.queues, queued);

    let totals = usage_totals(3, 6);
    reduce_lane_snapshot(
        &mut snapshot,
        &HarnessEvent::global(HarnessEventPayload::Usage {
            lane: "main".to_owned(),
            row: usage_row("usage-1", totals),
            totals,
        })
        .expect("global event"),
    );
    assert_eq!(snapshot.stats.usage, totals);
}

#[test]
fn usage_bypasses_the_lane_filter_while_foreign_lanes_drop() {
    let mut snapshot = empty_snapshot();

    let foreign_start = HarnessEvent::lane_scoped(
        "other",
        false,
        HarnessEventPayload::RunStart {
            run_id: "run".to_owned(),
            started_at: 2,
        },
    )
    .expect("lane-scoped event");
    assert_eq!(
        reduce_lane_snapshot(&mut snapshot, &foreign_start),
        LaneSnapshotReduction::Applied,
    );
    assert!(
        snapshot.operation.is_none(),
        "a foreign lane's run start drops",
    );

    let totals = usage_totals(3, 6);
    let foreign_usage = HarnessEvent {
        lane: Some("other".to_owned()),
        recovery: false,
        payload: HarnessEventPayload::Usage {
            lane: "other".to_owned(),
            row: usage_row("usage-1", totals),
            totals,
        },
    };
    assert_eq!(
        reduce_lane_snapshot(&mut snapshot, &foreign_usage),
        LaneSnapshotReduction::Applied,
    );
    assert_eq!(
        snapshot.stats.usage, totals,
        "usage folds regardless of the carrying lane",
    );
}

#[test]
fn folds_lane_config_updates_and_ignores_foreign_and_global_ones() {
    let mut snapshot = empty_snapshot();

    let identity = ModelIdentity {
        provider: "other".to_owned(),
        model_id: "other-model".to_owned(),
    };
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::ConfigUpdate {
            property: crate::harness::agent_harness::ConfigUpdateKind::Lane(
                crate::harness::agent_harness::LaneConfigUpdate::Model {
                    value: identity.clone(),
                    previous: None,
                },
            ),
        }),
    );
    assert_eq!(snapshot.configuration.model, identity, "the model folded");

    reduce_lane_snapshot(
        &mut snapshot,
        &HarnessEvent::lane_scoped(
            "other",
            false,
            HarnessEventPayload::ConfigUpdate {
                property: crate::harness::agent_harness::ConfigUpdateKind::Lane(
                    crate::harness::agent_harness::LaneConfigUpdate::ThinkingLevel {
                        value: ThinkingLevel::High,
                        previous: ThinkingLevel::Off,
                    },
                ),
            },
        )
        .expect("lane-scoped event"),
    );
    assert_eq!(
        snapshot.configuration.thinking_level,
        ThinkingLevel::Off,
        "a foreign lane's config update drops",
    );

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::ConfigUpdate {
            property: crate::harness::agent_harness::ConfigUpdateKind::Lane(
                crate::harness::agent_harness::LaneConfigUpdate::ActiveTools {
                    value: vec!["read".to_owned()],
                    previous: Vec::new(),
                },
            ),
        }),
    );
    assert_eq!(
        snapshot.configuration.active_tool_names,
        vec!["read".to_owned()],
    );

    reduce_lane_snapshot(
        &mut snapshot,
        &HarnessEvent::global(HarnessEventPayload::ConfigUpdate {
            property: crate::harness::agent_harness::ConfigUpdateKind::Global(
                crate::harness::agent_harness::GlobalConfigUpdate::Tools,
            ),
        })
        .expect("global event"),
    );
    assert_eq!(
        snapshot.configuration.active_tool_names,
        vec!["read".to_owned()],
        "a global config update folds nothing",
    );
}

#[test]
fn opens_structural_operations_on_an_idle_snapshot() {
    let mut snapshot = empty_snapshot();
    snapshot.tip_id = Some("tip".to_owned());

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::CompactionStart {
            run_id: "compact".to_owned(),
            reason: CompactionReason::Manual,
            started_at: 2,
        }),
    );
    let operation = snapshot
        .operation
        .as_ref()
        .expect("the compaction operation");
    assert_eq!(operation.id, "compact");
    assert_eq!(operation.kind, OperationKind::Compaction);
    assert_eq!(operation.started_at, 2);
    assert_eq!(operation.from_tip_id.as_deref(), Some("tip"));
    assert_eq!(operation.status, OperationStatus::Open);

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::NavigationStart {
            run_id: "nav".to_owned(),
            target_id: Some("target".to_owned()),
            started_at: 3,
        }),
    );
    let operation = snapshot
        .operation
        .as_ref()
        .expect("the navigation operation");
    assert_eq!(operation.id, "nav");
    assert_eq!(operation.kind, OperationKind::Navigation);
    assert_eq!(operation.started_at, 3);
}

fn operation_error(code: &str) -> crate::harness::session::types::OperationError {
    crate::harness::session::types::OperationError {
        code: code.to_owned(),
        message: "boom".to_owned(),
        details: None,
    }
}

#[test]
fn settles_run_end_into_the_record_and_the_tip() {
    let mut snapshot = empty_snapshot();
    snapshot.operation = Some(run_operation("run", OperationKind::Run));
    snapshot.tip_id = Some("start-tip".to_owned());

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::RunEnd {
            run_id: "run".to_owned(),
            from_tip_id: Some("start-tip".to_owned()),
            tip_id: Some("end-tip".to_owned()),
            ended_at: 9,
            status: crate::harness::agent_harness::RunEndStatus::Completed,
        }),
    );
    assert!(snapshot.operation.is_none(), "the run cleared");
    assert_eq!(snapshot.tip_id.as_deref(), Some("end-tip"));
    let record = snapshot.last_result.as_ref().expect("the settled record");
    assert_eq!(record.operation_id, "run");
    assert_eq!(record.kind, OperationKind::Run);
    assert_eq!(
        record.status,
        crate::harness::session::types::TerminalStatus::Completed
    );
    assert_eq!(record.error, None);
    assert_eq!(record.from_tip_id.as_deref(), Some("start-tip"));
    assert_eq!(record.tip_id.as_deref(), Some("end-tip"));
    assert_eq!(record.started_at, 1);
    assert_eq!(record.ended_at, 9);

    let mut failed = empty_snapshot();
    failed.operation = Some(run_operation("run", OperationKind::Run));
    reduce_lane_snapshot(
        &mut failed,
        &lane_event(HarnessEventPayload::RunEnd {
            run_id: "run".to_owned(),
            from_tip_id: None,
            tip_id: None,
            ended_at: 9,
            status: crate::harness::agent_harness::RunEndStatus::Failed {
                error: operation_error("provider_error"),
            },
        }),
    );
    let record = failed.last_result.as_ref().expect("the failed record");
    assert_eq!(
        record.status,
        crate::harness::session::types::TerminalStatus::Failed
    );
    assert_eq!(
        record.error,
        Some(operation_error("provider_error")),
        "the failed run carries its error",
    );

    let mut foreign = empty_snapshot();
    foreign.operation = Some(run_operation("run", OperationKind::Run));
    reduce_lane_snapshot(
        &mut foreign,
        &lane_event(HarnessEventPayload::RunEnd {
            run_id: "foreign".to_owned(),
            from_tip_id: None,
            tip_id: Some("end-tip".to_owned()),
            ended_at: 9,
            status: crate::harness::agent_harness::RunEndStatus::Completed,
        }),
    );
    assert!(
        foreign.last_result.is_none() && foreign.operation.is_some(),
        "a foreign run end settles nothing",
    );

    let mut structural = empty_snapshot();
    structural.operation = Some(run_operation("compact", OperationKind::Compaction));
    reduce_lane_snapshot(
        &mut structural,
        &lane_event(HarnessEventPayload::RunEnd {
            run_id: "compact".to_owned(),
            from_tip_id: None,
            tip_id: Some("end-tip".to_owned()),
            ended_at: 9,
            status: crate::harness::agent_harness::RunEndStatus::Completed,
        }),
    );
    assert!(
        structural.last_result.is_none() && structural.operation.is_some(),
        "a run end never settles a compaction operation",
    );
}

#[test]
fn settles_compaction_end_into_the_record() {
    let mut snapshot = empty_snapshot();
    snapshot.tip_id = Some("tip".to_owned());
    snapshot.operation = Some(run_operation("compact", OperationKind::Compaction));

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::CompactionEnd {
            run_id: "compact".to_owned(),
            reason: CompactionReason::Manual,
            ended_at: 9,
            status: crate::harness::agent_harness::CompactionEndStatus::Completed {
                entry_id: "compact-entry".to_owned(),
            },
        }),
    );
    assert!(snapshot.operation.is_none(), "the compaction cleared");
    let record = snapshot.last_result.as_ref().expect("the settled record");
    assert_eq!(record.operation_id, "compact");
    assert_eq!(record.kind, OperationKind::Compaction);
    assert_eq!(
        record.status,
        crate::harness::session::types::TerminalStatus::Completed
    );
    assert_eq!(
        record.from_tip_id, None,
        "the record carries the operation's tip"
    );
    assert_eq!(record.tip_id.as_deref(), Some("tip"), "the snapshot's tip");
    assert_eq!(record.started_at, 1);
    assert_eq!(record.ended_at, 9);

    let mut foreign = empty_snapshot();
    foreign.operation = Some(run_operation("compact", OperationKind::Compaction));
    reduce_lane_snapshot(
        &mut foreign,
        &lane_event(HarnessEventPayload::CompactionEnd {
            run_id: "foreign".to_owned(),
            reason: CompactionReason::Manual,
            ended_at: 9,
            status: crate::harness::agent_harness::CompactionEndStatus::Declined,
        }),
    );
    assert!(
        foreign.last_result.is_none() && foreign.operation.is_some(),
        "a foreign compaction end settles nothing",
    );

    let mut run = empty_snapshot();
    run.operation = Some(run_operation("run", OperationKind::Run));
    reduce_lane_snapshot(
        &mut run,
        &lane_event(HarnessEventPayload::CompactionEnd {
            run_id: "run".to_owned(),
            reason: CompactionReason::Manual,
            ended_at: 9,
            status: crate::harness::agent_harness::CompactionEndStatus::Declined,
        }),
    );
    assert!(
        run.last_result.is_none() && run.operation.is_some(),
        "a compaction end never settles a run operation",
    );
}

#[test]
fn marks_faults_and_ignores_the_noop_family() {
    let mut snapshot = empty_snapshot();
    snapshot.operation = Some(run_operation("run", OperationKind::Run));
    let before = snapshot.clone();

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::HandlerError {
            error: "boom".to_owned(),
            stack: None,
            kind: crate::harness::agent_harness::HandlerErrorKind::Hook {
                hook: "hook".to_owned(),
            },
        }),
    );
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::TurnStart {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
        }),
    );
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::TurnEnd {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
            message: match assistant_wire("stop", "done") {
                AgentMessage::Standard(Message::Assistant(assistant)) => assistant,
                other => panic!("the assistant wire: {other:?}"),
            },
            tool_results: Vec::new(),
        }),
    );
    reduce_lane_snapshot(
        &mut snapshot,
        &HarnessEvent::global(HarnessEventPayload::ValueUpdate {
            kind: crate::harness::agent_harness::ValueUpdateKind::SessionName {
                name: Some("name".to_owned()),
            },
        })
        .expect("global event"),
    );
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::LaneCreated { at: None }),
    );
    assert_eq!(snapshot, before, "the no-op family changed nothing");

    reduce_lane_snapshot(
        &mut snapshot,
        &HarnessEvent::global(HarnessEventPayload::Fault {
            code: "provider_error".to_owned(),
            message: "boom".to_owned(),
        })
        .expect("global event"),
    );
    assert!(snapshot.faulted, "the fault marked the snapshot");
}
#[test]
fn ignores_the_tool_end_for_a_call_that_never_started() {
    let mut snapshot = empty_snapshot();
    snapshot.operation = Some(run_operation("run", OperationKind::Run));
    reduce_lane_snapshot(&mut snapshot, &tool_end_event(7));
    assert!(
        running_tools(&snapshot).is_empty(),
        "an unstarted call's end settles nothing",
    );
}

#[test]
fn carries_the_aborted_and_failed_compaction_end_statuses() {
    let mut aborted = empty_snapshot();
    aborted.operation = Some(run_operation("compact", OperationKind::Compaction));
    reduce_lane_snapshot(
        &mut aborted,
        &lane_event(HarnessEventPayload::CompactionEnd {
            run_id: "compact".to_owned(),
            reason: CompactionReason::Manual,
            ended_at: 9,
            status: crate::harness::agent_harness::CompactionEndStatus::Aborted,
        }),
    );
    let record = aborted.last_result.as_ref().expect("the aborted record");
    assert_eq!(
        record.status,
        crate::harness::session::types::TerminalStatus::Aborted,
    );

    let mut failed = empty_snapshot();
    failed.operation = Some(run_operation("compact", OperationKind::Compaction));
    reduce_lane_snapshot(
        &mut failed,
        &lane_event(HarnessEventPayload::CompactionEnd {
            run_id: "compact".to_owned(),
            reason: CompactionReason::Manual,
            ended_at: 9,
            status: crate::harness::agent_harness::CompactionEndStatus::Failed {
                error: crate::harness::session::types::OperationError {
                    code: "model_error".to_owned(),
                    message: "boom".to_owned(),
                    details: None,
                },
            },
        }),
    );
    let record = failed.last_result.as_ref().expect("the failed record");
    assert_eq!(
        record.status,
        crate::harness::session::types::TerminalStatus::Failed,
    );
    assert_eq!(
        record.error.as_ref().map(|error| error.code.as_str()),
        Some("model_error"),
        "the failed compaction carries its error",
    );
}

#[test]
fn carries_the_aborted_run_end_status() {
    let mut snapshot = empty_snapshot();
    snapshot.operation = Some(run_operation("run", OperationKind::Run));
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::RunEnd {
            run_id: "run".to_owned(),
            from_tip_id: None,
            tip_id: None,
            ended_at: 9,
            status: crate::harness::agent_harness::RunEndStatus::Aborted,
        }),
    );
    let record = snapshot.last_result.as_ref().expect("the aborted record");
    assert_eq!(
        record.status,
        crate::harness::session::types::TerminalStatus::Aborted,
        "the aborted run carries its status",
    );
}

#[test]
fn folds_the_lane_thinking_level_update() {
    let mut snapshot = empty_snapshot();
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::ConfigUpdate {
            property: crate::harness::agent_harness::ConfigUpdateKind::Lane(
                crate::harness::agent_harness::LaneConfigUpdate::ThinkingLevel {
                    value: ThinkingLevel::High,
                    previous: ThinkingLevel::Off,
                },
            ),
        }),
    );
    assert_eq!(
        snapshot.configuration.thinking_level,
        ThinkingLevel::High,
        "the thinking level folded",
    );
}

#[test]
fn carries_the_declined_compaction_end_status() {
    let mut snapshot = empty_snapshot();
    snapshot.operation = Some(run_operation("compact", OperationKind::Compaction));
    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::CompactionEnd {
            run_id: "compact".to_owned(),
            reason: CompactionReason::Manual,
            ended_at: 9,
            status: crate::harness::agent_harness::CompactionEndStatus::Declined,
        }),
    );
    let record = snapshot.last_result.as_ref().expect("the declined record");
    assert_eq!(
        record.status,
        crate::harness::session::types::TerminalStatus::Declined,
        "the declined compaction carries its status",
    );
}

//! The agent-harness type surface's unit tests: the event payload's
//! lane-scoped split over every discriminant and the type discriminators'
//! wire names. Upstream exercises them through the runtime suites.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use std::sync::Arc;

use crate::harness::agent_harness::{
    HarnessEvent, HarnessEventPayload, HarnessEventType, LaneSnapshotTool, OperationStatus,
    Subscription,
};

fn deferred_handle() -> pi_ai::types::DeferredHandle {
    serde_json::from_value(serde_json::json!({
        "provider": "provider",
        "modelId": "model",
        "api": "api",
        "id": "deferred",
        "pollAfterMs": 1000
    }))
    .expect("deferred handle")
}

fn assistant_message() -> pi_ai::types::AssistantMessage {
    serde_json::from_value(serde_json::json!({
        "content": [{"type": "text", "text": "done"}],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "test-model",
        "timestamp": 2,
        "usage": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2,
            "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 } },
        "stopReason": "stop",
    }))
    .expect("assistant message")
}

fn user_agent_message() -> crate::types::AgentMessage {
    serde_json::from_value(serde_json::json!({
        "role": "user",
        "content": "hello",
        "timestamp": 1
    }))
    .expect("user message")
}

fn assistant_message_event() -> pi_ai::types::AssistantMessageEvent {
    serde_json::from_value(serde_json::json!({
        "type": "text_delta",
        "contentIndex": 0,
        "delta": "x",
        "partial": {
            "content": [{"type": "text", "text": "x"}],
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": "test-model",
            "timestamp": 2,
            "usage": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2,
                "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 } },
            "stopReason": "pending",
        }
    }))
    .expect("assistant message event")
}

fn usage() -> pi_ai::types::Usage {
    serde_json::from_value(serde_json::json!({
        "input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4, "totalTokens": 10,
        "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 }
    }))
    .expect("usage")
}

/// Every payload discriminant round-trips its upstream wire name and its
/// type discriminator matches, upstream's `HarnessEvent["type"]` table.
#[test]
fn every_payload_discriminant_carries_its_wire_name() {
    let kinds = [
        HarnessEventType::RunStart,
        HarnessEventType::RunResume,
        HarnessEventType::RunSuspend,
        HarnessEventType::OperationAbort,
        HarnessEventType::RunEnd,
        HarnessEventType::Fault,
        HarnessEventType::HandlerError,
        HarnessEventType::TurnStart,
        HarnessEventType::TurnEnd,
        HarnessEventType::RetryScheduled,
        HarnessEventType::RetryStart,
        HarnessEventType::RetryEnd,
        HarnessEventType::MessageStart,
        HarnessEventType::MessageUpdate,
        HarnessEventType::MessageEnd,
        HarnessEventType::ToolStart,
        HarnessEventType::ToolUpdate,
        HarnessEventType::ToolEnd,
        HarnessEventType::EntryAdded,
        HarnessEventType::QueueUpdate,
        HarnessEventType::ValueUpdate,
        HarnessEventType::ConfigUpdate,
        HarnessEventType::CompactionStart,
        HarnessEventType::CompactionEnd,
        HarnessEventType::NavigationStart,
        HarnessEventType::NavigationEnd,
        HarnessEventType::LaneCreated,
        HarnessEventType::Usage,
    ];
    for kind in kinds {
        let payload = payload_for(kind);
        let serialized = serde_json::to_value(&payload).expect("payload serializes");
        let wire = kind.as_str();
        assert_eq!(serialized["type"], serde_json::json!(wire), "{wire}");
        assert_eq!(
            payload.event_type().as_str(),
            wire,
            "the discriminant matches: {wire}"
        );
        let round: HarnessEventPayload = serde_json::from_value(serialized)
            .unwrap_or_else(|error| panic!("{wire} round-trips: {error}"));
        assert_eq!(round, payload, "{wire}");
    }
}

/// The lane-scoped split matches upstream's `LaneEventPayload` membership
/// through the runtime constructors.
#[test]
fn the_lane_split_matches_upstream_membership() {
    for kind in [
        HarnessEventType::RunStart,
        HarnessEventType::RunResume,
        HarnessEventType::RunSuspend,
        HarnessEventType::OperationAbort,
        HarnessEventType::RunEnd,
        HarnessEventType::TurnStart,
        HarnessEventType::TurnEnd,
        HarnessEventType::RetryScheduled,
        HarnessEventType::RetryStart,
        HarnessEventType::RetryEnd,
        HarnessEventType::MessageStart,
        HarnessEventType::MessageUpdate,
        HarnessEventType::MessageEnd,
        HarnessEventType::ToolStart,
        HarnessEventType::ToolUpdate,
        HarnessEventType::ToolEnd,
        HarnessEventType::EntryAdded,
        HarnessEventType::QueueUpdate,
        HarnessEventType::ConfigUpdate,
        HarnessEventType::CompactionStart,
        HarnessEventType::CompactionEnd,
        HarnessEventType::NavigationStart,
        HarnessEventType::NavigationEnd,
        HarnessEventType::LaneCreated,
    ] {
        let payload = payload_for(kind);
        assert!(payload.is_lane_scoped(), "{} is lane-scoped", kind.as_str());
    }
    for kind in [
        HarnessEventType::Fault,
        HarnessEventType::ValueUpdate,
        HarnessEventType::Usage,
    ] {
        let payload = payload_for(kind);
        assert!(
            !payload.is_lane_scoped(),
            "{} is harness-global",
            kind.as_str()
        );
    }
    // A lane-scoped config_update stays lane-scoped; a session-level one
    // does not.
    let lane_update = HarnessEventPayload::ConfigUpdate {
        property: crate::harness::agent_harness::ConfigUpdateKind::Lane(
            crate::harness::agent_harness::LaneConfigUpdate::ThinkingLevel {
                value: crate::types::ThinkingLevel::Low,
                previous: crate::types::ThinkingLevel::Off,
            },
        ),
    };
    assert!(lane_update.is_lane_scoped());
    let event = HarnessEvent::lane_scoped("main", false, lane_update).expect("lane scoped");
    assert_eq!(event.event_type().as_str(), "config_update");
}

/// The status and tool-snapshot unions carry their wire spellings.
#[test]
fn the_status_and_tool_snapshot_unions_keep_the_wire() {
    assert_eq!(
        serde_json::to_value(OperationStatus::Aborting).expect("status"),
        serde_json::json!("aborting")
    );
    let running_tool = LaneSnapshotTool::Running {
        tool_call_id: "call".to_owned(),
        tool_name: "read".to_owned(),
        args: serde_json::json!({}),
        result: None,
    };
    let LaneSnapshotTool::Running {
        tool_call_id,
        tool_name,
        ..
    } = &running_tool
    else {
        panic!("the running shape");
    };
    assert_eq!(tool_call_id, "call");
    assert_eq!(tool_name, "read");
    let subscription = Subscription::new(Arc::new(|| {}));
    subscription.unsubscribe();
    subscription.unsubscribe();
}

/// One fixture payload per event discriminant, upstream's
/// `HarnessEventPayload` table.
#[expect(
    clippy::too_many_lines,
    reason = "the fixture enumerates every payload discriminant; splitting it would hide the table's completeness"
)]
fn payload_for(kind: HarnessEventType) -> HarnessEventPayload {
    use crate::harness::agent_harness::{
        CompactionEndStatus, ConfigUpdateKind, HandlerErrorKind, LaneConfigUpdate,
        NavigationEndStatus, RunEndStatus, ValueUpdateKind,
    };
    use crate::harness::session::types::{CompactionReason, MessageEntry, UsageRow};

    match kind {
        HarnessEventType::RunStart => HarnessEventPayload::RunStart {
            run_id: "run".to_owned(),
            started_at: 1,
        },
        HarnessEventType::RunResume => HarnessEventPayload::RunResume {
            run_id: "run".to_owned(),
        },
        HarnessEventType::RunSuspend => HarnessEventPayload::RunSuspend {
            run_id: "run".to_owned(),
            deferred: deferred_handle(),
            poll: 0,
        },
        HarnessEventType::OperationAbort => HarnessEventPayload::OperationAbort {
            operation_id: "run".to_owned(),
            steer: vec![],
            follow_up: vec![],
        },
        HarnessEventType::RunEnd => HarnessEventPayload::RunEnd {
            run_id: "run".to_owned(),
            from_tip_id: None,
            tip_id: None,
            ended_at: 2,
            status: RunEndStatus::Completed,
        },
        HarnessEventType::Fault => HarnessEventPayload::Fault {
            code: "code".to_owned(),
            message: "message".to_owned(),
        },
        HarnessEventType::HandlerError => HarnessEventPayload::HandlerError {
            error: "error".to_owned(),
            stack: None,
            kind: HandlerErrorKind::Event {
                event: "run_start".to_owned(),
            },
        },
        HarnessEventType::TurnStart => HarnessEventPayload::TurnStart {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
        },
        HarnessEventType::TurnEnd => HarnessEventPayload::TurnEnd {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
            message: assistant_message(),
            tool_results: vec![],
        },
        HarnessEventType::RetryScheduled => HarnessEventPayload::RetryScheduled {
            run_id: "run".to_owned(),
            step: "assistant".to_owned(),
            attempt: 1,
            max_attempts: 3,
            delay_ms: 100,
            not_before: 10,
            error_message: "error".to_owned(),
        },
        HarnessEventType::RetryStart => HarnessEventPayload::RetryStart {
            run_id: "run".to_owned(),
            step: "assistant".to_owned(),
            attempt: 1,
        },
        HarnessEventType::RetryEnd => HarnessEventPayload::RetryEnd {
            run_id: "run".to_owned(),
            step: "assistant".to_owned(),
            attempt: 1,
            success: true,
            final_error: None,
        },
        HarnessEventType::MessageStart => HarnessEventPayload::MessageStart {
            run_id: None,
            message: user_agent_message(),
        },
        HarnessEventType::MessageUpdate => HarnessEventPayload::MessageUpdate {
            run_id: "run".to_owned(),
            message: Box::new(user_agent_message()),
            event: Box::new(assistant_message_event()),
            frame: None,
        },
        HarnessEventType::MessageEnd => HarnessEventPayload::MessageEnd {
            run_id: None,
            message: user_agent_message(),
            entry_id: None,
        },
        HarnessEventType::ToolStart => HarnessEventPayload::ToolStart {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call".to_owned(),
            tool_name: "read".to_owned(),
            args: serde_json::json!({}),
        },
        HarnessEventType::ToolUpdate => HarnessEventPayload::ToolUpdate {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call".to_owned(),
            tool_name: "read".to_owned(),
            partial_result: crate::types::AgentToolResult {
                content: vec![],
                details: serde_json::json!({}),
                usage: None,
                added_tool_names: None,
                terminate: None,
            },
        },
        HarnessEventType::ToolEnd => HarnessEventPayload::ToolEnd {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call".to_owned(),
            tool_name: "read".to_owned(),
            result: crate::types::AgentToolResult {
                content: vec![],
                details: serde_json::json!({}),
                usage: None,
                added_tool_names: None,
                terminate: None,
            },
            is_error: false,
            terminate: false,
        },
        HarnessEventType::EntryAdded => HarnessEventPayload::EntryAdded {
            entry: crate::harness::session::types::Entry::Message {
                id: "entry".to_owned(),
                parent_id: None,
                seq: 1,
                timestamp: 1,
                body: Box::new(MessageEntry {
                    message: user_agent_message(),
                    terminate: None,
                }),
            },
        },
        HarnessEventType::QueueUpdate => HarnessEventPayload::QueueUpdate { queues: vec![] },
        HarnessEventType::ValueUpdate => HarnessEventPayload::ValueUpdate {
            kind: ValueUpdateKind::SessionName {
                name: Some("session".to_owned()),
            },
        },
        HarnessEventType::ConfigUpdate => HarnessEventPayload::ConfigUpdate {
            property: ConfigUpdateKind::Lane(LaneConfigUpdate::ThinkingLevel {
                value: crate::types::ThinkingLevel::Low,
                previous: crate::types::ThinkingLevel::Off,
            }),
        },
        HarnessEventType::CompactionStart => HarnessEventPayload::CompactionStart {
            run_id: "run".to_owned(),
            reason: CompactionReason::Manual,
            started_at: 1,
        },
        HarnessEventType::CompactionEnd => HarnessEventPayload::CompactionEnd {
            run_id: "run".to_owned(),
            reason: CompactionReason::Manual,
            ended_at: 2,
            status: CompactionEndStatus::Completed {
                entry_id: "entry".to_owned(),
            },
        },
        HarnessEventType::NavigationStart => HarnessEventPayload::NavigationStart {
            run_id: "run".to_owned(),
            target_id: None,
            started_at: 1,
        },
        HarnessEventType::NavigationEnd => HarnessEventPayload::NavigationEnd {
            run_id: "run".to_owned(),
            from_tip_id: None,
            tip_id: None,
            ended_at: 2,
            status: NavigationEndStatus::Completed,
        },
        HarnessEventType::LaneCreated => HarnessEventPayload::LaneCreated { at: None },
        HarnessEventType::Usage => HarnessEventPayload::Usage {
            lane: "main".to_owned(),
            row: UsageRow {
                id: "usage".to_owned(),
                seq: 2,
                usage: usage(),
                entry_id: None,
                adjustment: false,
                details: None,
            },
            totals: usage(),
        },
    }
}

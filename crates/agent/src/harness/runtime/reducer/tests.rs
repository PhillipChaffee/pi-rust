//! The lane snapshot reducer suite, ported 1:1 from upstream
//! `test/harness/runtime/reducer.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` — the portable half.
//!
//! The suite's first three cases ("folds an ordinary run…", "folds suspend
//! and resume…", "folds standalone compaction…") drive the real
//! `AgentHarness` — their `lane.prompt`/`resume`/`compact` settle through
//! the drive child's procedure loop — so they ride the drive child's
//! landing and are recorded there. The queue-replication case constructs
//! the lane directly over a real session with bus-backed emit and watch
//! surfaces, and the remaining cases fold hand-built snapshots over the
//! pure reducer.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#[cfg(test)]
mod events;
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;

use pi_ai::types::Message;
use pi_ai::types::ToolResultBlock;
use pi_ai::types::ToolResultMessage;

use crate::harness::agent_harness::AgentLane;
use crate::harness::agent_harness::EventListener;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::OperationStatus;
use crate::harness::agent_harness::QueueMessage;
use crate::harness::agent_harness::{
    LaneQueuedItem, LaneSnapshot, LaneSnapshotTool, LiveOperationView,
};
use crate::harness::context::background_context;
use crate::harness::events::HarnessEventBus;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::reducer::LaneSnapshotReduction;
use crate::harness::runtime::reducer::reduce_lane_snapshot;
use crate::harness::runtime::test_support::bus_emit_batch;
use crate::harness::runtime::test_support::bus_watch_installer;
use crate::harness::runtime::test_support::lane_configuration;
use crate::harness::runtime::test_support::lock;
use crate::harness::runtime::test_support::memory_session_with_seed;
use crate::harness::runtime::test_support::restored_lane;
use crate::harness::runtime::test_support::settle_events;
use crate::harness::session::session::StorageBackedSession;
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::Entry;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::OperationKind;
use crate::types::AgentMessage;
use crate::types::AgentToolContent;
use crate::types::AgentToolResult;
use pi_ai::types::TextContent;

/// The empty snapshot upstream's `(await lane.watch()).snapshot` seeds the
/// hand-built fixtures with.
fn empty_snapshot() -> LaneSnapshot {
    crate::harness::runtime::test_support::empty_lane_snapshot("main", &lane_configuration())
}

/// The live operation view the fixtures install, upstream's inline
/// `{ id, kind, startedAt: 1, fromTipId: null, status: "open",
/// runningTools: [] }` literals.
fn run_operation(id: &str, kind: OperationKind) -> LiveOperationView {
    LiveOperationView {
        id: id.to_owned(),
        kind,
        started_at: 1,
        from_tip_id: None,
        status: OperationStatus::Open,
        retry: None,
        deferred: None,
        streaming_message: None,
        running_tools: Vec::new(),
    }
}

fn lane_event(payload: HarnessEventPayload) -> HarnessEvent {
    HarnessEvent::lane_scoped("main", false, payload).expect("lane-scoped event")
}

/// The tool result upstream's `result(text)` builds.
fn tool_result(text: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![AgentToolContent::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        details: json!({ "text": text }),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

fn tool_start_event(index: usize) -> HarnessEvent {
    lane_event(HarnessEventPayload::ToolStart {
        run_id: "run".to_owned(),
        turn_id: "turn".to_owned(),
        tool_call_id: format!("call-{index}"),
        tool_name: format!("tool-{index}"),
        args: json!({ "index": index }),
    })
}

fn tool_end_event(index: usize) -> HarnessEvent {
    lane_event(HarnessEventPayload::ToolEnd {
        run_id: "run".to_owned(),
        turn_id: "turn".to_owned(),
        tool_call_id: format!("call-{index}"),
        tool_name: format!("tool-{index}"),
        result: tool_result(&format!("done-{index}")),
        is_error: false,
        terminate: false,
    })
}

/// The tool-result entry upstream's `entry(index)` builds.
fn tool_result_entry(index: usize) -> HarnessEvent {
    lane_event(HarnessEventPayload::EntryAdded {
        entry: Entry::Message {
            id: format!("result-{index}"),
            parent_id: if index == 0 {
                None
            } else {
                Some(format!("result-{}", index - 1))
            },
            seq: u64::try_from(index + 1).expect("seq"),
            timestamp: i64::try_from(index + 1).expect("timestamp"),
            body: Box::new(MessageEntry {
                message: AgentMessage::Standard(Message::ToolResult(ToolResultMessage {
                    tool_call_id: format!("call-{index}"),
                    tool_name: format!("tool-{index}"),
                    content: vec![ToolResultBlock::Text(TextContent {
                        text: format!("done-{index}"),
                        text_signature: None,
                    })],
                    details: None,
                    usage: None,
                    added_tool_names: None,
                    is_error: false,
                    timestamp: i64::try_from(index + 1).expect("timestamp"),
                })),
                terminate: None,
            }),
        },
    })
}

fn running_tools(snapshot: &LaneSnapshot) -> &Vec<LaneSnapshotTool> {
    &snapshot
        .operation
        .as_ref()
        .expect("the live operation")
        .running_tools
}

fn running_tool_ids(snapshot: &LaneSnapshot) -> Vec<String> {
    running_tools(snapshot)
        .iter()
        .map(|tool| match tool {
            LaneSnapshotTool::Running { tool_call_id, .. }
            | LaneSnapshotTool::Settled { tool_call_id, .. } => tool_call_id.clone(),
        })
        .collect()
}

fn running_tool_statuses(snapshot: &LaneSnapshot) -> Vec<&'static str> {
    running_tools(snapshot)
        .iter()
        .map(|tool| match tool {
            LaneSnapshotTool::Running { .. } => "running",
            LaneSnapshotTool::Settled { .. } => "settled",
        })
        .collect()
}

#[tokio::test]
async fn keeps_in_run_compaction_segments_inside_the_open_run() {
    let mut snapshot = empty_snapshot();
    snapshot.operation = Some(run_operation("run", OperationKind::Run));

    assert_eq!(
        reduce_lane_snapshot(
            &mut snapshot,
            &lane_event(HarnessEventPayload::CompactionStart {
                run_id: "run".to_owned(),
                reason: CompactionReason::Threshold,
                started_at: 2,
            }),
        ),
        LaneSnapshotReduction::Applied,
        "the in-run compaction start leaves the open run",
    );
    assert_eq!(
        reduce_lane_snapshot(
            &mut snapshot,
            &lane_event(HarnessEventPayload::CompactionEnd {
                run_id: "run".to_owned(),
                reason: CompactionReason::Threshold,
                ended_at: 3,
                status: crate::harness::agent_harness::CompactionEndStatus::Declined,
            }),
        ),
        LaneSnapshotReduction::Applied,
        "the in-run compaction end leaves the open run",
    );
    let operation = snapshot.operation.as_ref().expect("the open run survives");
    assert_eq!(operation.id, "run");
    assert_eq!(operation.kind, OperationKind::Run);
}

#[tokio::test]
async fn retains_settled_parallel_tools_until_each_source_ordered_result_is_placed() {
    let mut snapshot = empty_snapshot();
    snapshot.operation = Some(run_operation("run", OperationKind::Run));

    for index in 0..3 {
        reduce_lane_snapshot(&mut snapshot, &tool_start_event(index));
    }
    assert_eq!(running_tools(&snapshot).len(), 3);
    reduce_lane_snapshot(&mut snapshot, &tool_start_event(0));
    assert_eq!(
        running_tools(&snapshot).len(),
        3,
        "the upsert replaced in place"
    );

    reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::ToolUpdate {
            run_id: "stale".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call-1".to_owned(),
            tool_name: "tool-1".to_owned(),
            partial_result: tool_result("stale"),
        }),
    );
    assert!(
        matches!(
            running_tools(&snapshot)[1],
            LaneSnapshotTool::Running { result: None, .. }
        ),
        "the stale run's partial result never landed",
    );

    reduce_lane_snapshot(&mut snapshot, &tool_end_event(2));
    assert!(
        matches!(
            running_tools(&snapshot)
                .iter()
                .find(|tool| match tool {
                    LaneSnapshotTool::Running { tool_call_id, .. }
                    | LaneSnapshotTool::Settled { tool_call_id, .. } => tool_call_id == "call-2",
                })
                .expect("call-2 present"),
            LaneSnapshotTool::Settled { args, result, .. }
                if args == &json!({ "index": 2 }) && *result == tool_result("done-2"),
        ),
        "call-2 settled with its args and result",
    );

    reduce_lane_snapshot(&mut snapshot, &tool_end_event(0));
    assert_eq!(
        running_tool_statuses(&snapshot),
        vec!["settled", "running", "settled"]
    );
    reduce_lane_snapshot(&mut snapshot, &tool_result_entry(0));
    assert_eq!(running_tool_ids(&snapshot), vec!["call-1", "call-2"]);
    let transcript_ids: Vec<String> = snapshot
        .transcript
        .iter()
        .map(|entry| entry.id().to_owned())
        .collect();
    assert_eq!(transcript_ids, vec!["result-0"]);

    reduce_lane_snapshot(&mut snapshot, &tool_end_event(1));
    assert!(
        running_tools(&snapshot)
            .iter()
            .all(|tool| matches!(tool, LaneSnapshotTool::Settled { .. })),
        "every started call settled",
    );
    reduce_lane_snapshot(&mut snapshot, &tool_result_entry(1));
    assert_eq!(running_tool_ids(&snapshot), vec!["call-2"]);
    reduce_lane_snapshot(&mut snapshot, &tool_result_entry(2));
    assert!(running_tools(&snapshot).is_empty());
    let transcript_ids: Vec<String> = snapshot
        .transcript
        .iter()
        .map(|entry| entry.id().to_owned())
        .collect();
    assert_eq!(transcript_ids, vec!["result-0", "result-1", "result-2"]);
}

#[tokio::test]
async fn marks_navigation_completion_for_rebase() {
    let mut snapshot = empty_snapshot();
    snapshot.operation = Some(run_operation("navigation", OperationKind::Navigation));

    let reduced = reduce_lane_snapshot(
        &mut snapshot,
        &lane_event(HarnessEventPayload::NavigationEnd {
            run_id: "navigation".to_owned(),
            status: crate::harness::agent_harness::NavigationEndStatus::Completed,
            from_tip_id: None,
            tip_id: Some("target".to_owned()),
            ended_at: 2,
        }),
    );
    assert_eq!(reduced, LaneSnapshotReduction::Rebase);
}
/// The collector listener the fold's event recording installs.
fn collector(sink: Arc<Mutex<Vec<HarnessEvent>>>) -> EventListener {
    Arc::new(move |event: &HarnessEvent, _context| {
        let sink = Arc::clone(&sink);
        let event = event.clone();
        Box::pin(async move {
            lock(&sink).push(event);
            Ok(())
        })
    })
}

/// The lane the queue-replication case constructs directly, upstream's
/// `createFixture` + `lane.watch`: a real session, the bus-backed emit and
/// watch surfaces, no drive.
async fn bus_lane(bus: &Arc<HarnessEventBus>) -> (Lane, Arc<StorageBackedSession>) {
    let session = memory_session_with_seed(None).await;
    let lane = restored_lane(
        session.clone(),
        bus_emit_batch(Arc::clone(bus)),
        bus_watch_installer(Arc::clone(bus), empty_snapshot()),
    )
    .await;
    (lane, session)
}

/// The queue changes every lane method publishes fold into the replica's
/// globally ordered queues and match the authoritative resnapshot.
#[tokio::test]
async fn replicates_globally_ordered_queue_changes() {
    let bus = Arc::new(HarnessEventBus::new());
    let (lane, _session) = bus_lane(&bus).await;
    let watch = lane.watch(&background_context()).await.expect("watch");
    let delivered: Arc<Mutex<Vec<HarnessEvent>>> = Arc::new(Mutex::new(Vec::new()));
    watch.start(collector(Arc::clone(&delivered)));

    let _next = lane
        .next_run(
            QueueMessage::Text("next".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("nextRun serves")
        .expect("nextRun ok");
    let steer = lane
        .steer(
            QueueMessage::Text("steer".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("steer serves")
        .expect("steer ok");
    let _follow = lane
        .follow_up(
            QueueMessage::Text("follow".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("followUp serves")
        .expect("followUp ok");
    lane.cancel_queued(&steer, &background_context())
        .await
        .expect("cancelQueued serves");
    settle_events().await;

    let mut replica = watch.snapshot();
    for event in lock(&delivered).iter() {
        let reduction = reduce_lane_snapshot(&mut replica, event);
        assert_eq!(
            reduction,
            LaneSnapshotReduction::Applied,
            "unexpected rebase for {}",
            event.event_type().as_str()
        );
    }
    let kinds: Vec<InboxItemKind> = replica
        .queues
        .iter()
        .map(|item| match item {
            LaneQueuedItem::Message { kind, .. } | LaneQueuedItem::Custom { kind, .. } => *kind,
        })
        .collect();
    assert_eq!(kinds, vec![InboxItemKind::NextRun, InboxItemKind::FollowUp]);
    let resnapshotted = watch
        .resnapshot(&background_context())
        .await
        .expect("resnapshot");
    assert_eq!(replica, resnapshotted);
    watch.unsubscribe();
}

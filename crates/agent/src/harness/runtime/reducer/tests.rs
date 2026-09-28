//! The lane snapshot reducer suite, ported 1:1 from upstream
//! `test/harness/runtime/reducer.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The suite's first three cases drive the real `AgentHarness`: the faux
//! provider's scripted responses settle `lane.prompt`/`resume`/`compact`
//! through the drive procedures, and the captured events folded over the
//! pre-run snapshot must reach the watch's authoritative resnapshot
//! exactly — any rebase fails the case. The queue-replication case
//! constructs the lane directly over a real session with bus-backed emit
//! and watch surfaces, and the remaining cases fold hand-built snapshots
//! over the pure reducer.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; a violated expectation panics the test by design"
)]
#[cfg(test)]
mod events;
use serde_json::json;
use std::sync::{Arc, Mutex};

use pi_ai::models::create_models;
use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxProviderHandle;
use pi_ai::providers::faux::{RegisterFauxProviderOptions, faux_assistant_message, faux_provider};
use pi_ai::types::{DeferredRequest, Message, ToolResultBlock, ToolResultMessage};

use crate::harness::agent_harness::AgentHarnessOptions;
use crate::harness::agent_harness::AgentLane;
use crate::harness::agent_harness::EventListener;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::{
    HarnessEventPayload, OperationStatus, QueueMessage, RunOutcome,
};
use crate::harness::agent_harness::{
    LaneQueuedItem, LaneSnapshot, LaneSnapshotTool, LiveOperationView,
};
use crate::harness::context::background_context;
use crate::harness::events::HarnessEventBus;
use crate::harness::runtime::harness::{Harness, create_agent_harness};
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::reducer::{LaneSnapshotReduction, reduce_lane_snapshot};
use crate::harness::runtime::test_support::bus_emit_batch;
use crate::harness::runtime::test_support::bus_watch_installer;
use crate::harness::runtime::test_support::lane_configuration;
use crate::harness::runtime::test_support::lock;
use crate::harness::runtime::test_support::memory_session_with_seed;
use crate::harness::runtime::test_support::restored_lane;
use crate::harness::runtime::test_support::runtime_session_metadata;
use crate::harness::runtime::test_support::{settle_events, user_text_message};
use crate::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use crate::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::Entry;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::{MessageEntry, OperationKind, Session, TerminalStatus};
use crate::harness::types::AgentHarnessStreamOptions;
use crate::types::{AgentMessage, AgentToolContent, AgentToolResult};
use pi_ai::types::TextContent;

// The drive-anchored cases: the real `AgentHarness` container, the faux
// provider, the watch capture, and the fold helper — upstream's
// `createFixture`, `settleEvents`, and `fold`.

/// The drive-anchored cases' fixture, upstream's `createFixture`: the
/// storage-backed session over a plain memory backend, the faux provider
/// registered into a fresh models catalog, and the container attaching the
/// `main` lane. The ledger rides the fixture so the teardown closes the
/// session.
struct ReducerFixture {
    /// The attached container keeping the shared state.
    _harness: Harness,
    /// The attached `main` lane, upstream's `lane`.
    lane: Arc<dyn AgentLane>,
    /// The faux handle the cases script their responses through.
    faux: FauxProviderHandle,
}

/// Attaches the runtime over one ledgered session, upstream's
/// `createFixture({ deferred })`: the deferred stream option rides the
/// options when asked, upstream's `streamOptions: { deferred: true }`.
///
/// # Panics
/// The creation's or the attach's failure.
async fn create_fixture(
    ledger: &mut Vec<Arc<StorageBackedSession>>,
    deferred: bool,
) -> ReducerFixture {
    let session = Arc::new(StorageBackedSession::new(
        runtime_session_metadata(format!("reducer-{}", ledger.len())),
        Arc::new(MemoryStorage::new(MemoryStorageOptions::default())),
        StorageBackedSessionOptions::default(),
    ));
    ledger.push(Arc::clone(&session));
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    let models = Arc::new(create_models(None));
    models.set_provider(Arc::new(faux.provider.clone()));
    let options = AgentHarnessOptions {
        session: session.clone(),
        models,
        model: faux.first_model(),
        thinking_level: None,
        active_tool_names: None,
        tools: None,
        tool_context: None,
        system_prompt: None,
        resources: None,
        stream_options: deferred.then(|| AgentHarnessStreamOptions {
            deferred: Some(DeferredRequest::Enabled(true)),
            ..AgentHarnessStreamOptions::default()
        }),
        retry: None,
        compaction: None,
        steering_mode: None,
        follow_up_mode: None,
        tool_execution: None,
        to_provider_messages: None,
        entry_projectors: None,
    };
    let created = create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates");
    let lane = created
        .harness
        .lane("main", &background_context())
        .await
        .expect("the lane attaches");
    ReducerFixture {
        _harness: created.harness,
        lane,
        faux,
    }
}

/// The fold upstream's `fold(snapshot, events)` helper: reduces every
/// captured event into the replica; a rebase fails the case with the
/// event's type.
fn fold(mut replica: LaneSnapshot, events: &[HarnessEvent]) -> LaneSnapshot {
    for event in events {
        let reduction = reduce_lane_snapshot(&mut replica, event);
        assert_eq!(
            reduction,
            LaneSnapshotReduction::Applied,
            "unexpected rebase for {}",
            event.event_type().as_str(),
        );
    }
    replica
}

/// Closes the ledgered sessions and clears the ledger, upstream's
/// `afterEach` teardown.
///
/// # Panics
/// A session close's failure.
async fn close_sessions(ledger: &mut Vec<Arc<StorageBackedSession>>) {
    for session in ledger.drain(..) {
        session
            .close(&background_context())
            .await
            .expect("the session closes");
    }
}

/// Upstream `it` under "lane snapshot reducer": folds an ordinary run to
/// the authoritative resnapshot.
#[tokio::test]
async fn folds_an_ordinary_run_to_the_authoritative_resnapshot() {
    let mut ledger = Vec::new();
    let fixture = create_fixture(&mut ledger, false).await;
    let watch = fixture
        .lane
        .watch(&background_context())
        .await
        .expect("watch");
    let delivered: Arc<Mutex<Vec<HarnessEvent>>> = Arc::new(Mutex::new(Vec::new()));
    watch.start(collector(Arc::clone(&delivered)));
    fixture.faux.set_responses([faux_assistant_message(
        "answer",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);

    let outcome = fixture
        .lane
        .prompt_text("question", None, &background_context())
        .await
        .expect("the prompt serves")
        .expect("the run settles");
    match outcome {
        RunOutcome::Settled(record) => {
            assert_eq!(record.kind, OperationKind::Run);
            assert_eq!(record.status, TerminalStatus::Completed);
        }
        RunOutcome::Suspended(suspended) => panic!("the completed run: {suspended:?}"),
    }
    settle_events().await;

    let replica = fold(watch.snapshot(), &lock(&delivered).clone());
    let authoritative = watch
        .resnapshot(&background_context())
        .await
        .expect("resnapshot");
    assert_eq!(
        replica, authoritative,
        "the folded events reached the authoritative resnapshot",
    );
    watch.unsubscribe();
    close_sessions(&mut ledger).await;
}

/// Upstream `it` under "lane snapshot reducer": folds suspend and resume
/// without closing the operation early.
#[tokio::test]
async fn folds_suspend_and_resume_without_closing_the_operation_early() {
    let mut ledger = Vec::new();
    let fixture = create_fixture(&mut ledger, true).await;
    let watch = fixture
        .lane
        .watch(&background_context())
        .await
        .expect("watch");
    let delivered: Arc<Mutex<Vec<HarnessEvent>>> = Arc::new(Mutex::new(Vec::new()));
    watch.start(collector(Arc::clone(&delivered)));
    fixture.faux.set_responses([faux_assistant_message(
        "answer",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);

    let suspended = fixture
        .lane
        .prompt_text("question", None, &background_context())
        .await
        .expect("the prompt serves")
        .expect("the run suspends");
    match suspended {
        RunOutcome::Suspended(suspended) => assert_eq!(suspended.status, "suspended"),
        RunOutcome::Settled(record) => panic!("the suspended run: {record:?}"),
    }
    settle_events().await;
    let replica = fold(watch.snapshot(), &lock(&delivered).clone());
    let operation = replica.operation.as_ref().expect("the open operation");
    assert_eq!(operation.kind, OperationKind::Run);
    let deferred = operation.deferred.as_ref().expect("the deferred view");
    assert_eq!(deferred.poll, 0, "the suspension carried the first poll");
    let authoritative = watch
        .resnapshot(&background_context())
        .await
        .expect("resnapshot");
    assert_eq!(
        replica, authoritative,
        "the suspended fold reached the authoritative resnapshot",
    );

    lock(&delivered).clear();
    let outcome = fixture
        .lane
        .resume(&background_context())
        .await
        .expect("resume serves")
        .expect("the run settles");
    match outcome {
        RunOutcome::Settled(record) => {
            assert_eq!(record.kind, OperationKind::Run);
            assert_eq!(record.status, TerminalStatus::Completed);
        }
        RunOutcome::Suspended(suspended) => panic!("the completed resume: {suspended:?}"),
    }
    settle_events().await;
    let replica = fold(watch.snapshot(), &lock(&delivered).clone());
    let authoritative = watch
        .resnapshot(&background_context())
        .await
        .expect("resnapshot");
    assert_eq!(
        replica, authoritative,
        "the resumed fold reached the authoritative resnapshot",
    );
    watch.unsubscribe();
    close_sessions(&mut ledger).await;
}

/// Upstream `it` under "lane snapshot reducer": folds standalone compaction
/// and preserves segment semantics.
#[tokio::test]
async fn folds_standalone_compaction_and_preserves_segment_semantics() {
    let mut ledger = Vec::new();
    let fixture = create_fixture(&mut ledger, false).await;
    fixture
        .lane
        .append_message(user_text_message("history"), &background_context())
        .await
        .expect("append serves");
    let watch = fixture
        .lane
        .watch(&background_context())
        .await
        .expect("watch");
    let delivered: Arc<Mutex<Vec<HarnessEvent>>> = Arc::new(Mutex::new(Vec::new()));
    watch.start(collector(Arc::clone(&delivered)));
    fixture.faux.set_responses([faux_assistant_message(
        "summary",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);

    let outcome = fixture
        .lane
        .compact(None, &background_context())
        .await
        .expect("compact serves")
        .expect("the compaction settles");
    assert_eq!(outcome.compaction.kind, OperationKind::Compaction);
    assert_eq!(outcome.compaction.status, TerminalStatus::Completed);
    settle_events().await;

    let replica = fold(watch.snapshot(), &lock(&delivered).clone());
    assert!(
        replica.operation.is_none(),
        "the compaction cleared its operation",
    );
    let last = replica.last_result.as_ref().expect("the last result");
    assert_eq!(last.kind, OperationKind::Compaction);
    assert_eq!(last.status, TerminalStatus::Completed);
    let authoritative = watch
        .resnapshot(&background_context())
        .await
        .expect("resnapshot");
    assert_eq!(
        replica, authoritative,
        "the folded events reached the authoritative resnapshot",
    );
    watch.unsubscribe();
    close_sessions(&mut ledger).await;
}

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

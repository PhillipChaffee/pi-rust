//! The runtime boundary suite: behavior upstream exercises from other
//! layers (the harness lane surface, the drive procedures, the storage
//! contract) bound here where the runtime child owns the seams — the staged
//! seam raises, the drive spawn's failure path, the mismatch shape, the
//! durable lane record's wire round-trip, the inbox admission split, and
//! the captured-model projection over the state leaves.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; a violated expectation panics the test by design"
)]

use std::sync::Arc;

use serde_json::json;

use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::NavigateOptions;
use crate::harness::agent_harness::OperationRequest;
use crate::harness::context::background_context;
use crate::harness::result::HarnessError;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::lane::captured_model;
use crate::harness::runtime::lane::durable_lane_state;
use crate::harness::runtime::lane::select_accepted_inbox;
use crate::harness::runtime::test_support::ControlledStorage;
use crate::harness::runtime::test_support::assistant_wire_value;
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::empty_lane_snapshot;
use crate::harness::runtime::test_support::finish_operation;
use crate::harness::runtime::test_support::generation_context;
use crate::harness::runtime::test_support::lane_configuration;
use crate::harness::runtime::test_support::memory_session_with_seed;
use crate::harness::runtime::test_support::noop_emit_batch;
use crate::harness::runtime::test_support::operation_scope;
use crate::harness::runtime::test_support::patch_live_state;
use crate::harness::runtime::test_support::raw_write;
use crate::harness::runtime::test_support::restored_lane;
use crate::harness::runtime::test_support::runtime_config;
use crate::harness::runtime::test_support::unused_watch_installer;
use crate::harness::runtime::test_support::user_text_message;
use crate::harness::runtime::test_support::zero_usage_wire;
use crate::harness::runtime::types::SliceNotImplemented;
use crate::harness::session::memory::MemoryStorage;
use crate::harness::session::memory::MemoryStorageOptions;
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::AssistantReadyOperation;
use crate::harness::session::types::AssistantRetryWaitOperation;
use crate::harness::session::types::CheckpointData;
use crate::harness::session::types::CheckpointOperation;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::Control;
use crate::harness::session::types::DeferredEffectPendingOperation;
use crate::harness::session::types::DeferredScope;
use crate::harness::session::types::DeferredSuspendedOperation;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::NavigationReadyToCommitOperation;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::RetryWait;
use crate::harness::session::types::SummaryContext;
use crate::harness::session::types::SummaryDecidingOperation;
use crate::harness::session::types::SummaryEffectPendingOperation;
use crate::harness::session::types::SummaryGenerationScope;
use crate::harness::session::types::SummaryReadyOperation;
use crate::harness::session::types::SummaryRetryWaitOperation;
use crate::harness::session::types::SummaryTask;
use crate::harness::session::types::ToolBatch;
use crate::harness::session::types::ToolsOperation;
use crate::types::QueueMode;

/// The structural task the summary leaves carry.
fn summary_task() -> SummaryTask {
    SummaryTask {
        task_id: "task".to_owned(),
        reason: Some(crate::harness::session::types::CompactionReason::Manual),
        custom_instructions: None,
        boundary: ResultBoundary::Finish,
    }
}

/// The summary generation inputs the summary leaves carry.
fn summary_context() -> SummaryContext {
    SummaryContext {
        result_entry_id: "summary".to_owned(),
        configuration: lane_configuration(),
        stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
        retry_policy: generation_context().retry_policy,
    }
}

/// A configured `main` lane over a seeded session, the fixture the seam
/// tests admit through; `Some(operation_id)` seeds an admitted starting
/// operation for the drive-fault path.
async fn seam_lane(operation_id: Option<&str>) -> Lane {
    let session = memory_session_with_seed(operation_id).await;
    restored_lane(session, noop_emit_batch(), unused_watch_installer()).await
}

/// The staged seam error the result carries, unwrapped.
fn slice_error_of(error: &crate::harness::runtime::types::LaneError) -> SliceNotImplemented {
    error
        .downcast_ref::<SliceNotImplemented>()
        .expect("the staged seam raises SliceNotImplemented")
        .clone()
}

/// `accept` on a compaction request raises the staged seam; the raise
/// happens before any session work.
#[tokio::test]
async fn accept_compaction_raises_the_staged_seam() {
    let lane = seam_lane(None).await;
    let error = lane
        .accept_impl(
            OperationRequest::Compaction {
                operation_id: None,
                custom_instructions: None,
            },
            &background_context(),
        )
        .await
        .expect_err("the staged seam raises");
    assert_eq!(slice_error_of(&error).operation, "compaction");
}

/// A summarized navigation raises the staged seam; the plain path does not.
#[tokio::test]
async fn accept_navigation_summarize_raises_the_staged_seam() {
    let lane = seam_lane(None).await;
    let error = lane
        .accept_impl(
            OperationRequest::Navigation {
                operation_id: None,
                target_id: Some("target".to_owned()),
                options: Some(NavigateOptions {
                    summarize: Some(true),
                    label: None,
                    custom_instructions: None,
                }),
            },
            &background_context(),
        )
        .await
        .expect_err("the staged seam raises");
    assert_eq!(slice_error_of(&error).operation, "summarized navigation");
}

/// The compact surface collapses into the same staged seam raise.
#[tokio::test]
async fn compact_impl_carries_the_staged_seam_error() {
    let lane = seam_lane(None).await;
    let error = lane
        .compact_impl(None, &background_context())
        .await
        .expect_err("the staged seam raises");
    assert_eq!(slice_error_of(&error).operation, "compaction");
}

/// A freshly installed drive pass faults with the staged seam (the drive
/// child owns the procedure loop) and clears its owner, mirroring the real
/// spawn handler's failure path.
#[tokio::test]
async fn drive_impl_faults_a_fresh_install_with_the_staged_seam() {
    let lane = seam_lane(Some("operation")).await;
    let driven = lane
        .drive_impl(
            DriveOptions {
                operation_id: "operation".to_owned(),
                wait_for_retry: None,
                poll_deferred: None,
            },
            &background_context(),
        )
        .await;
    let error = driven.expect_err("the fresh install faults with the staged seam");
    assert_eq!(slice_error_of(&error).operation, "drive operation");
    assert!(
        lane.active_drive().is_none(),
        "the failed pass cleared its owner"
    );
}

/// The mismatch error names the lane, the expected operation, and the
/// current and last ids.
#[tokio::test]
async fn mismatch_names_the_lane_and_the_expected_operation() {
    let lane = seam_lane(None).await;
    let error = lane.mismatch("expected", Some("current"), Some("last"));
    assert_eq!(error.tag(), "OperationMismatch");
    match error {
        HarnessError::OperationMismatch {
            lane,
            expected_operation_id,
            current_operation_id,
            last_operation_id,
            message,
        } => {
            assert_eq!(lane, "main");
            assert_eq!(expected_operation_id, "expected");
            assert_eq!(current_operation_id, Some("current".to_owned()));
            assert_eq!(last_operation_id, Some("last".to_owned()));
            assert_eq!(message, "Operation expected does not own lane \"main\"");
        }
        other => panic!("the mismatch shape: {other:?}"),
    }
    let without = lane.mismatch("expected", None, None);
    match without {
        HarnessError::OperationMismatch {
            current_operation_id,
            last_operation_id,
            ..
        } => {
            assert_eq!(current_operation_id, None);
            assert_eq!(last_operation_id, None);
        }
        other => panic!("the mismatch shape without ids: {other:?}"),
    }
}

/// The durable lane record round-trips through serde, the wire shape the
/// storage contract persists.
#[test]
fn durable_lane_state_round_trips_through_serde() {
    let populated = durable_lane_state(
        Some("operation"),
        &[
            InboxItem {
                entry_id: "steer-1".to_owned(),
                kind: InboxItemKind::Steer,
            },
            InboxItem {
                entry_id: "write-1".to_owned(),
                kind: InboxItemKind::Write,
            },
        ],
        Some("last"),
    );
    let wire = serde_json::to_value(&populated).expect("the lane record serializes");
    let restored: crate::harness::session::types::LaneState =
        serde_json::from_value(wire).expect("the lane record deserializes");
    assert_eq!(restored, populated);
    assert_eq!(restored.current_operation_id, Some("operation".to_owned()));
    assert_eq!(restored.last_operation_id, Some("last".to_owned()));
    assert_eq!(restored.inbox.len(), 2);

    let empty = durable_lane_state(None, &[], None);
    let wire = serde_json::to_value(&empty).expect("the empty record serializes");
    let restored: crate::harness::session::types::LaneState =
        serde_json::from_value(wire).expect("the empty record deserializes");
    assert_eq!(restored, empty);
    assert_eq!(restored.current_operation_id, None);
    assert!(restored.inbox.is_empty());
}

fn inbox_item(entry_id: &str, kind: InboxItemKind) -> InboxItem {
    InboxItem {
        entry_id: entry_id.to_owned(),
        kind,
    }
}

fn inbox_ids(items: &[InboxItem]) -> Vec<String> {
    items.iter().map(|item| item.entry_id.clone()).collect()
}

/// The one-at-a-time modes admit the oldest steer and follow-up and hold
/// the rest; writes and next-run items are always eligible.
#[test]
fn select_accepted_inbox_takes_one_steer_and_one_follow_up_at_a_time() {
    let inbox = vec![
        inbox_item("steer-1", InboxItemKind::Steer),
        inbox_item("steer-2", InboxItemKind::Steer),
        inbox_item("follow-1", InboxItemKind::FollowUp),
        inbox_item("follow-2", InboxItemKind::FollowUp),
        inbox_item("write-1", InboxItemKind::Write),
        inbox_item("next-1", InboxItemKind::NextRun),
    ];
    let (selected, remainder) =
        select_accepted_inbox(&inbox, QueueMode::OneAtATime, QueueMode::OneAtATime);
    assert_eq!(
        inbox_ids(&selected),
        vec!["steer-1", "follow-1", "write-1", "next-1"],
        "the oldest of each one-at-a-time queue plus the always-eligible kinds"
    );
    assert_eq!(inbox_ids(&remainder), vec!["steer-2", "follow-2"]);
}

/// The all modes admit every queued item in order.
#[test]
fn select_accepted_inbox_all_modes_take_every_item() {
    let inbox = vec![
        inbox_item("steer-1", InboxItemKind::Steer),
        inbox_item("follow-1", InboxItemKind::FollowUp),
        inbox_item("write-1", InboxItemKind::Write),
        inbox_item("next-1", InboxItemKind::NextRun),
    ];
    let (selected, remainder) = select_accepted_inbox(&inbox, QueueMode::All, QueueMode::All);
    assert_eq!(
        inbox_ids(&selected),
        vec!["steer-1", "follow-1", "write-1", "next-1"]
    );
    assert!(remainder.is_empty());
}

/// A one-at-a-time steering mode leaves every follow-up untouched when the
/// follow-up mode is all, and the other way round.
#[test]
fn select_accepted_inbox_modes_independent_per_queue() {
    let inbox = vec![
        inbox_item("steer-1", InboxItemKind::Steer),
        inbox_item("steer-2", InboxItemKind::Steer),
        inbox_item("follow-1", InboxItemKind::FollowUp),
        inbox_item("follow-2", InboxItemKind::FollowUp),
    ];
    let (selected, remainder) =
        select_accepted_inbox(&inbox, QueueMode::OneAtATime, QueueMode::All);
    assert_eq!(
        inbox_ids(&selected),
        vec!["steer-1", "follow-1", "follow-2"]
    );
    assert_eq!(inbox_ids(&remainder), vec!["steer-2"]);
    let (selected, remainder) =
        select_accepted_inbox(&inbox, QueueMode::All, QueueMode::OneAtATime);
    assert_eq!(inbox_ids(&selected), vec!["steer-1", "steer-2", "follow-1"]);
    assert_eq!(inbox_ids(&remainder), vec!["follow-2"]);
}

/// The leaves whose generation or batch context captures a model.
fn model_carrying_leaves(scope: OperationScope) -> Vec<OperationState> {
    let configuration = lane_configuration();
    vec![
        OperationState::AssistantReady(AssistantReadyOperation {
            scope: scope.clone(),
            generation_context: generation_context(),
            next_attempt: 1,
        }),
        OperationState::AssistantEffectPending(AssistantEffectPendingOperation {
            scope: scope.clone(),
            generation_context: generation_context(),
            attempt: 1,
            response_entry_id: "response".to_owned(),
            usage_id: "usage".to_owned(),
            intended_output_limit: 100,
            context_window: 1_000,
        }),
        OperationState::AssistantRetryWait(AssistantRetryWaitOperation {
            scope: scope.clone(),
            generation_context: generation_context(),
            retry_wait: RetryWait {
                next_attempt: 2,
                not_before: 10,
                error_message: "boom".to_owned(),
            },
        }),
        OperationState::Tools(ToolsOperation {
            scope: scope.clone(),
            batch: ToolBatch {
                assistant_entry_id: "assistant".to_owned(),
                configuration,
                turn_id: "turn".to_owned(),
                calls: vec![
                    serde_json::from_value(json!({
                        "sourceIndex": 0,
                        "resultEntryId": "result",
                        "status": "planned",
                    }))
                    .expect("tool call wire"),
                ],
            },
        }),
        OperationState::DeferredSuspended(DeferredSuspendedOperation {
            deferred: deferred_scope(scope.clone()),
        }),
        OperationState::DeferredEffectPending(DeferredEffectPendingOperation {
            scope: deferred_scope(scope.clone()),
            response_entry_id: "response".to_owned(),
            usage_id: "usage".to_owned(),
        }),
        OperationState::SummaryReady(SummaryReadyOperation {
            scope: scope.clone(),
            generation: summary_generation(),
            next_attempt: 1,
        }),
        OperationState::SummaryEffectPending(SummaryEffectPendingOperation {
            scope: scope.clone(),
            generation: summary_generation(),
            attempt: 1,
            request: None,
            usage_ids: Vec::new(),
        }),
        OperationState::SummaryRetryWait(SummaryRetryWaitOperation {
            scope,
            generation: summary_generation(),
            retry_wait: RetryWait {
                next_attempt: 2,
                not_before: 10,
                error_message: "boom".to_owned(),
            },
        }),
    ]
}

/// The leaves without a generation or batch context capture nothing.
fn model_less_leaves(scope: OperationScope) -> Vec<OperationState> {
    vec![
        OperationState::Starting(crate::harness::session::types::StartingOperation {
            scope: scope.clone(),
        }),
        OperationState::Checkpoint(CheckpointOperation {
            scope: scope.clone(),
            checkpoint: CheckpointData {
                continuation: Continuation::MayFinish {
                    include_final_assistant: true,
                },
                trigger_entry_id: "trigger".to_owned(),
            },
        }),
        OperationState::SummaryDeciding(SummaryDecidingOperation {
            scope: scope.clone(),
            task: summary_task(),
        }),
        OperationState::NavigationReadyToCommit(NavigationReadyToCommitOperation {
            scope,
            target_id: Some("target".to_owned()),
            label: None,
        }),
    ]
}

fn deferred_scope(scope: OperationScope) -> DeferredScope {
    DeferredScope {
        scope,
        step_id: "step".to_owned(),
        source_entry_id: "source".to_owned(),
        poll: 0,
        configuration: lane_configuration(),
        stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
    }
}

fn summary_generation() -> SummaryGenerationScope {
    SummaryGenerationScope {
        task: summary_task(),
        summary_context: summary_context(),
    }
}

/// The captured model reads each state leaf's configuration; leaves without
/// a generation or batch context capture nothing.
#[test]
fn captured_model_reads_the_configuration_over_the_state_leaves() {
    let expected = lane_configuration().model;
    for state in model_carrying_leaves(operation_scope()) {
        assert_eq!(
            captured_model(&state),
            Some(expected.clone()),
            "the {} leaf captures its configuration's model",
            state.at()
        );
    }
    for state in model_less_leaves(operation_scope()) {
        assert_eq!(
            captured_model(&state),
            None,
            "the {} leaf captures nothing",
            state.at()
        );
    }
}

/// The staged-seam error reports the operation it names (the assertion the
/// types vocabulary's removed test module carried).
#[test]
fn slice_not_implemented_reports_the_operation_it_names() {
    let error = SliceNotImplemented::new("compaction");
    assert_eq!(
        error.to_string(),
        "compaction is not implemented until its later AgentHarness slice"
    );
    assert_eq!(error, SliceNotImplemented::new("compaction"));
    assert_ne!(error, SliceNotImplemented::new("navigation"));
}

// The snapshot-capture suite: the lane's watch surface over the leaf
// fixtures — the arms `captureLaneSnapshot` reads — and the scope
// rebuild's leaf coverage.

use crate::harness::agent_harness::AgentLane;
use crate::harness::agent_harness::WatchHandle;
use crate::harness::runtime::lane::operation_state_with_scope;
use crate::harness::runtime::test_support::recording_watch_installer;
use crate::harness::session::types::Entry;
use crate::harness::session::types::Storage;
use crate::types::AgentMessage;

/// The capturing fixture: the seam lane over a recording watch, the watch
/// surface the capture tests drive.
async fn capturing_lane(operation_id: Option<&str>) -> Lane {
    let session = memory_session_with_seed(operation_id).await;
    restored_lane(
        session,
        noop_emit_batch(),
        recording_watch_installer(empty_lane_snapshot("main", &lane_configuration())),
    )
    .await
}

/// The assistant wire the deferred and tool fixtures build, the deferred
/// handle and the tool-call blocks free.
fn assistant_wire(content: &serde_json::Value, stop_reason: &str, deferred: bool) -> AgentMessage {
    let mut wire = assistant_wire_value(content, stop_reason);
    if deferred {
        wire["deferred"] = json!({
            "provider": "provider",
            "modelId": "model",
            "api": "api",
            "id": "deferred",
            "pollAfterMs": 1000,
        });
    }
    serde_json::from_value(wire).expect("assistant wire")
}

#[tokio::test]
async fn watch_captures_the_lane_snapshot_over_the_durable_values() {
    let lane = capturing_lane(None).await;
    accept_prompt(&lane).await;
    let prompt_tip = lane.state().tip_id.expect("the accepted tip");
    let operation_id = lane
        .state()
        .operation
        .as_ref()
        .map(|operation| operation.meta.operation_id.clone())
        .expect("the live operation");
    finish_operation(&lane).await;
    let appended =
        AgentLane::append_message(&lane, user_text_message("history"), &background_context())
            .await
            .expect("append serves");

    let watch = AgentLane::watch(&lane, &background_context())
        .await
        .expect("watch serves");
    let snapshot = WatchHandle::<crate::harness::agent_harness::LaneSnapshot>::snapshot(&*watch);
    assert_eq!(snapshot.lane, "main");
    let ids: Vec<&str> = snapshot.transcript.iter().map(Entry::id).collect();
    assert_eq!(
        ids,
        vec![prompt_tip.as_str(), appended.as_str()],
        "the transcript is the branch path, oldest first",
    );
    assert_eq!(snapshot.tip_id.as_deref(), Some(appended.as_str()));
    assert_eq!(
        snapshot
            .last_result
            .as_ref()
            .map(|record| record.operation_id.as_str()),
        Some(operation_id.as_str()),
        "the finished operation's record rides the snapshot",
    );
    assert_eq!(
        snapshot.queues,
        Vec::new(),
        "the empty inbox reads no queues"
    );
    assert_eq!(snapshot.operation, None);
    assert_eq!(snapshot.stats.message_count, 2, "the storage totals read");
    assert!(!snapshot.faulted, "the open lane is not faulted");
    watch.unsubscribe();
}

#[tokio::test]
async fn watch_reports_the_streaming_message_from_the_stored_frames() {
    let lane = capturing_lane(None).await;
    AgentLane::accept(
        &lane,
        OperationRequest::Prompt {
            operation_id: None,
            prompt: Box::new(crate::harness::agent_harness::PromptMessagesPayload::Text {
                prompt: "hello".to_owned(),
                images: None,
            }),
        },
        &background_context(),
    )
    .await
    .expect("accept serves")
    .expect("the admission");
    patch_live_state(
        &lane,
        OperationState::AssistantEffectPending(AssistantEffectPendingOperation {
            scope: operation_scope(),
            generation_context: generation_context(),
            attempt: 1,
            response_entry_id: "response".to_owned(),
            usage_id: "usage".to_owned(),
            intended_output_limit: 100,
            context_window: 1_000,
        }),
    )
    .await;

    let operation_id = lane
        .state()
        .operation
        .as_ref()
        .map(|operation| operation.meta.operation_id.clone())
        .expect("the live operation");
    let partial: pi_ai::types::AssistantMessage = serde_json::from_value(json!({
        "content": [],
        "api": "anthropic-messages",
        "provider": "test",
        "model": "model",
        "usage": {
            "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
        },
        "stopReason": "pending",
        "timestamp": 1,
    }))
    .expect("the partial wire");
    let address =
        crate::harness::session::values::pending_assistant_frames(&operation_id, "response");
    commit_writes(
        lane.session(),
        vec![
            crate::harness::session::values::append_list_write(
                &address,
                pi_ai::utils::assistant_message_frame::AssistantMessageFrame::Start {
                    partial: partial.clone(),
                },
            )
            .expect("frame append"),
            crate::harness::session::values::append_list_write(
                &address,
                pi_ai::utils::assistant_message_frame::AssistantMessageFrame::TextStart {
                    content_index: 0,
                    content: pi_ai::types::TextContent {
                        text: String::new(),
                        text_signature: None,
                    },
                },
            )
            .expect("frame append"),
            crate::harness::session::values::append_list_write(
                &address,
                pi_ai::utils::assistant_message_frame::AssistantMessageFrame::TextDelta {
                    content_index: 0,
                    delta: "a".to_owned(),
                },
            )
            .expect("frame append"),
        ],
    )
    .await;

    let watch = AgentLane::watch(&lane, &background_context())
        .await
        .expect("watch serves");
    let snapshot = WatchHandle::<crate::harness::agent_harness::LaneSnapshot>::snapshot(&*watch);
    let streaming = snapshot
        .operation
        .as_ref()
        .expect("the operation view")
        .streaming_message
        .as_ref()
        .expect("the streaming message");
    assert_eq!(
        streaming.stop_reason,
        pi_ai::types::StopReason::Pending,
        "the reduced frames carry the pending stream",
    );
    watch.unsubscribe();
}

#[tokio::test]
async fn watch_reports_the_deferred_view_from_the_source_entry() {
    let lane = capturing_lane(None).await;
    // The source entry commits while the lane is idle; the run admits after.
    let source = AgentLane::append_message(
        &lane,
        assistant_wire(&json!([]), "stop", true),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        OperationState::DeferredSuspended(DeferredSuspendedOperation {
            deferred: DeferredScope {
                scope: operation_scope(),
                step_id: "step".to_owned(),
                source_entry_id: source.clone(),
                poll: 3,
                configuration: lane_configuration(),
                stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
            },
        }),
    )
    .await;

    let watch = AgentLane::watch(&lane, &background_context())
        .await
        .expect("watch serves");
    let snapshot = WatchHandle::<crate::harness::agent_harness::LaneSnapshot>::snapshot(&*watch);
    let deferred = snapshot
        .operation
        .as_ref()
        .expect("the operation view")
        .deferred
        .as_ref()
        .expect("the deferred view");
    assert_eq!(deferred.poll, 3);
    assert_eq!(
        deferred.handle.id, "deferred",
        "the source entry's handle rode the view",
    );
    watch.unsubscribe();
}

#[tokio::test]
async fn watch_faults_when_the_deferred_source_lacks_its_handle() {
    let lane = capturing_lane(None).await;
    let source = AgentLane::append_message(
        &lane,
        user_text_message("not deferred"),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        OperationState::DeferredSuspended(DeferredSuspendedOperation {
            deferred: DeferredScope {
                scope: operation_scope(),
                step_id: "step".to_owned(),
                source_entry_id: source,
                poll: 0,
                configuration: lane_configuration(),
                stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
            },
        }),
    )
    .await;

    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the handle-less source faults the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason
                    .to_string()
                    .contains("Deferred source is missing its assistant handle"),
                "the capture faults with the deferred invariant: {reason}",
            );
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "the batch's two phases and their durable reads are one continuous capture"
)]
#[tokio::test]
async fn watch_reports_the_tools_batch_running_and_settled_calls() {
    let lane = capturing_lane(None).await;
    // The batch's assistant entry commits while the lane is idle; the run
    // admits after.
    let assistant_entry = AgentLane::append_message(
        &lane,
        assistant_wire(
            &json!([
                { "type": "toolCall", "id": "call-1", "name": "tool-1", "arguments": {} },
                { "type": "toolCall", "id": "call-2", "name": "tool-2", "arguments": { "k": "block" } },
            ]),
            "toolUse",
            false,
        ),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        OperationState::Tools(ToolsOperation {
            scope: operation_scope(),
            batch: ToolBatch {
                assistant_entry_id: assistant_entry.clone(),
                configuration: lane_configuration(),
                turn_id: "turn".to_owned(),
                calls: vec![
                    serde_json::from_value(json!({
                        "sourceIndex": 2,
                        "resultEntryId": "result-3",
                        "status": "planned",
                    }))
                    .expect("the planned call wire"),
                    serde_json::from_value(json!({
                        "sourceIndex": 0,
                        "resultEntryId": "result-1",
                        "status": "effect_pending",
                        "replay": "safe",
                    }))
                    .expect("the effect-pending call wire"),
                    serde_json::from_value(json!({
                        "sourceIndex": 1,
                        "resultEntryId": "result-2",
                        "status": "outcome_ready",
                        "terminate": false,
                    }))
                    .expect("the outcome-ready call wire"),
                ],
            },
        }),
    )
    .await;

    let operation_id = lane
        .state()
        .operation
        .as_ref()
        .map(|operation| operation.meta.operation_id.clone())
        .expect("the live operation");
    commit_writes(
        lane.session(),
        vec![
            raw_write(
                &crate::harness::session::values::operation_tool_args(&operation_id, "turn", 0)
                    .address,
                json!({ "k": "persisted" }),
            ),
            crate::harness::session::values::set_value_write(
                &crate::harness::session::values::pending_tool_output(&operation_id, "result-1"),
                crate::harness::session::values::ToolOutputPayload {
                    content: vec![crate::types::AgentToolContent::Text(
                        pi_ai::types::TextContent {
                            text: "checkpoint".to_owned(),
                            text_signature: None,
                        },
                    )],
                    details: serde_json::Value::Null,
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                },
            )
            .expect("checkpoint write"),
            crate::harness::session::values::set_value_write(
                &crate::harness::session::values::pending_entry("result-2"),
                crate::harness::session::types::PendingEntry::Message {
                    payload: Box::new(
                        serde_json::from_value(json!({
                            "role": "toolResult",
                            "toolCallId": "call-2",
                            "toolName": "tool-2",
                            "content": [{ "type": "text", "text": "done-2" }],
                            "isError": false,
                            "timestamp": 2,
                        }))
                        .expect("the tool result wire"),
                    ),
                },
            )
            .expect("staged write"),
        ],
    )
    .await;

    let watch = AgentLane::watch(&lane, &background_context())
        .await
        .expect("watch serves");
    let snapshot = WatchHandle::<crate::harness::agent_harness::LaneSnapshot>::snapshot(&*watch);
    let running_tools = &snapshot
        .operation
        .as_ref()
        .expect("the operation view")
        .running_tools;
    assert_eq!(
        running_tools.len(),
        2,
        "the planned call skipped, both calls read"
    );
    match &running_tools[0] {
        crate::harness::agent_harness::LaneSnapshotTool::Running {
            tool_call_id,
            tool_name,
            args,
            result,
        } => {
            assert_eq!(tool_call_id, "call-1");
            assert_eq!(tool_name, "tool-1");
            assert_eq!(
                *args,
                json!({ "k": "persisted" }),
                "the persisted args ride"
            );
            assert!(
                result.is_some(),
                "the durable checkpoint rides the running call",
            );
        }
        other @ crate::harness::agent_harness::LaneSnapshotTool::Settled { .. } => {
            panic!("the settled call: {other:?}");
        }
    }
    match &running_tools[1] {
        crate::harness::agent_harness::LaneSnapshotTool::Settled {
            tool_call_id,
            tool_name,
            args,
            result,
            is_error,
        } => {
            assert_eq!(tool_call_id, "call-2");
            assert_eq!(tool_name, "tool-2");
            assert_eq!(*args, json!({ "k": "block" }), "the block's arguments ride");
            assert!(!*is_error);
            let crate::types::AgentToolResult { content, .. } = result;
            assert_eq!(content.len(), 1, "the staged result's content rides");
        }
        other @ crate::harness::agent_harness::LaneSnapshotTool::Running { .. } => {
            panic!("the running call: {other:?}");
        }
    }
    watch.unsubscribe();
}

#[expect(
    clippy::too_many_lines,
    reason = "the three invariants share the fixture; one body keeps the fixture shape visible"
)]
#[tokio::test]
async fn watch_faults_on_the_tools_batch_invariants() {
    // The invalid assistant entry.
    let lane = capturing_lane(None).await;
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        OperationState::Tools(ToolsOperation {
            scope: operation_scope(),
            batch: ToolBatch {
                assistant_entry_id: "absent".to_owned(),
                configuration: lane_configuration(),
                turn_id: "turn".to_owned(),
                calls: Vec::new(),
            },
        }),
    )
    .await;
    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the invalid batch faults the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason
                    .to_string()
                    .contains("Tool batch assistant entry is invalid"),
                "the invalid batch names its invariant: {reason}",
            );
        }
    }

    // The missing persisted arguments.
    let lane = capturing_lane(None).await;
    let assistant_entry = AgentLane::append_message(
        &lane,
        assistant_wire(
            &json!([{ "type": "toolCall", "id": "call-1", "name": "tool-1", "arguments": {} }]),
            "toolUse",
            false,
        ),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        OperationState::Tools(ToolsOperation {
            scope: operation_scope(),
            batch: ToolBatch {
                assistant_entry_id: assistant_entry,
                configuration: lane_configuration(),
                turn_id: "turn".to_owned(),
                calls: vec![
                    serde_json::from_value(json!({
                        "sourceIndex": 0,
                        "resultEntryId": "result-1",
                        "status": "effect_pending",
                        "replay": "safe",
                    }))
                    .expect("the effect-pending call wire"),
                ],
            },
        }),
    )
    .await;
    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the arg-less call faults the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason
                    .to_string()
                    .contains("Tool call call-1 is missing persisted arguments"),
                "the arg-less call names its invariant: {reason}",
            );
        }
    }

    // The missing staged result.
    let lane = capturing_lane(None).await;
    let assistant_entry = AgentLane::append_message(
        &lane,
        assistant_wire(
            &json!([{ "type": "toolCall", "id": "call-2", "name": "tool-2", "arguments": {} }]),
            "toolUse",
            false,
        ),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        OperationState::Tools(ToolsOperation {
            scope: operation_scope(),
            batch: ToolBatch {
                assistant_entry_id: assistant_entry,
                configuration: lane_configuration(),
                turn_id: "turn".to_owned(),
                calls: vec![
                    serde_json::from_value(json!({
                        "sourceIndex": 0,
                        "resultEntryId": "result-2",
                        "status": "outcome_ready",
                        "terminate": false,
                    }))
                    .expect("the outcome-ready call wire"),
                ],
            },
        }),
    )
    .await;
    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the staged-less call faults the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason
                    .to_string()
                    .contains("Tool call result-2 is missing its staged result"),
                "the staged-less call names its invariant: {reason}",
            );
        }
    }
}

/// The prompt admission the capture fixtures share.
async fn accept_prompt(lane: &Lane) {
    AgentLane::accept(
        lane,
        OperationRequest::Prompt {
            operation_id: None,
            prompt: Box::new(crate::harness::agent_harness::PromptMessagesPayload::Text {
                prompt: "hello".to_owned(),
                images: None,
            }),
        },
        &background_context(),
    )
    .await
    .expect("accept serves")
    .expect("the admission");
}

#[tokio::test]
async fn watch_reports_the_retry_views() {
    let lane = capturing_lane(None).await;
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        OperationState::AssistantRetryWait(AssistantRetryWaitOperation {
            scope: operation_scope(),
            generation_context: generation_context(),
            retry_wait: RetryWait {
                next_attempt: 2,
                not_before: 10,
                error_message: "boom".to_owned(),
            },
        }),
    )
    .await;
    let watch = AgentLane::watch(&lane, &background_context())
        .await
        .expect("watch serves");
    let snapshot = WatchHandle::<crate::harness::agent_harness::LaneSnapshot>::snapshot(&*watch);
    let retry = snapshot
        .operation
        .as_ref()
        .expect("the operation view")
        .retry
        .as_ref()
        .expect("the retry view");
    assert_eq!(retry.attempt, 2);
    assert_eq!(retry.max_attempts, 2, "the generation policy's budget");
    assert_eq!(retry.next_attempt_at, 10);
    watch.unsubscribe();

    patch_live_state(
        &lane,
        OperationState::SummaryRetryWait(SummaryRetryWaitOperation {
            scope: operation_scope(),
            generation: summary_generation(),
            retry_wait: RetryWait {
                next_attempt: 3,
                not_before: 20,
                error_message: "boom".to_owned(),
            },
        }),
    )
    .await;
    let watch = AgentLane::watch(&lane, &background_context())
        .await
        .expect("watch serves");
    let snapshot = WatchHandle::<crate::harness::agent_harness::LaneSnapshot>::snapshot(&*watch);
    let retry = snapshot
        .operation
        .as_ref()
        .expect("the operation view")
        .retry
        .as_ref()
        .expect("the summary retry view");
    assert_eq!(retry.attempt, 3);
    assert_eq!(retry.next_attempt_at, 20);
    watch.unsubscribe();
}

/// Every state leaf rebuilds under a fresh scope, the drive procedures'
/// `{ ...state, control, ... }` spreads.
#[test]
fn the_state_scope_rebuild_carries_every_leaf() {
    let scope = operation_scope();
    let states = model_carrying_leaves(scope.clone());
    let states = states.into_iter().chain(model_less_leaves(scope));
    for state in states {
        let fresh = OperationScope {
            control: Control::CancelRequested { requested_at: 5 },
            ..operation_scope()
        };
        let rebuilt = operation_state_with_scope(&state, fresh);
        assert_eq!(
            crate::harness::session::types::operation_scope_of(&rebuilt).control,
            Control::CancelRequested { requested_at: 5 },
            "the {} leaf rebuilt under the fresh scope",
            state.at(),
        );
        assert_eq!(
            rebuilt.at(),
            state.at(),
            "the {} leaf keeps its family",
            state.at(),
        );
    }
}

/// The deferred-effect-pending leaf the deferred+streaming capture drives.
fn deferred_effect_pending(source_entry_id: &str, response_entry_id: &str) -> OperationState {
    OperationState::DeferredEffectPending(DeferredEffectPendingOperation {
        scope: DeferredScope {
            scope: operation_scope(),
            step_id: "step".to_owned(),
            source_entry_id: source_entry_id.to_owned(),
            poll: 1,
            configuration: lane_configuration(),
            stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
        },
        response_entry_id: response_entry_id.to_owned(),
        usage_id: "usage".to_owned(),
    })
}

#[tokio::test]
async fn watch_reports_the_deferred_and_streaming_views_together() {
    let lane = capturing_lane(None).await;
    let source = AgentLane::append_message(
        &lane,
        assistant_wire(&json!([]), "stop", true),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    patch_live_state(&lane, deferred_effect_pending(&source, "response")).await;

    let partial: pi_ai::types::AssistantMessage = serde_json::from_value(json!({
        "content": [],
        "api": "anthropic-messages",
        "provider": "test",
        "model": "model",
        "usage": zero_usage_wire(),
        "stopReason": "pending",
        "timestamp": 1,
    }))
    .expect("the partial wire");
    let address = crate::harness::session::values::pending_assistant_frames(
        &lane
            .state()
            .operation
            .as_ref()
            .map(|operation| operation.meta.operation_id.clone())
            .expect("the operation"),
        "response",
    );
    commit_writes(
        lane.session(),
        vec![
            crate::harness::session::values::append_list_write(
                &address,
                pi_ai::utils::assistant_message_frame::AssistantMessageFrame::Start { partial },
            )
            .expect("frame append"),
            crate::harness::session::values::append_list_write(
                &address,
                pi_ai::utils::assistant_message_frame::AssistantMessageFrame::TextStart {
                    content_index: 0,
                    content: pi_ai::types::TextContent {
                        text: String::new(),
                        text_signature: None,
                    },
                },
            )
            .expect("frame append"),
            crate::harness::session::values::append_list_write(
                &address,
                pi_ai::utils::assistant_message_frame::AssistantMessageFrame::TextDelta {
                    content_index: 0,
                    delta: "a".to_owned(),
                },
            )
            .expect("frame append"),
        ],
    )
    .await;

    let watch = AgentLane::watch(&lane, &background_context())
        .await
        .expect("watch serves");
    let snapshot = WatchHandle::<crate::harness::agent_harness::LaneSnapshot>::snapshot(&*watch);
    let operation = snapshot.operation.as_ref().expect("the operation view");
    assert_eq!(
        operation.deferred.as_ref().expect("the deferred view").poll,
        1,
        "the deferred view rode the capture",
    );
    assert!(
        operation.streaming_message.is_some(),
        "the streaming view rode the capture",
    );
    watch.unsubscribe();
}

#[tokio::test]
async fn watch_reports_the_aborting_operation_status() {
    let lane = capturing_lane(None).await;
    accept_prompt(&lane).await;
    let accepted = lane
        .state()
        .operation
        .as_ref()
        .map(|operation| operation.state.clone())
        .expect("the accepted operation");
    patch_live_state(
        &lane,
        operation_state_with_scope(
            &accepted,
            OperationScope {
                control: Control::CancelRequested { requested_at: 5 },
                ..operation_scope()
            },
        ),
    )
    .await;

    let watch = AgentLane::watch(&lane, &background_context())
        .await
        .expect("watch serves");
    let snapshot = WatchHandle::<crate::harness::agent_harness::LaneSnapshot>::snapshot(&*watch);
    assert_eq!(
        snapshot
            .operation
            .as_ref()
            .expect("the operation view")
            .status,
        crate::harness::agent_harness::OperationStatus::Aborting,
        "the cancelled operation reports aborting",
    );
    watch.unsubscribe();
}

/// The last-operation variants the capture's result read rejects.
#[tokio::test]
async fn watch_faults_on_the_last_result_invariants() {
    // The missing record.
    let lane = capturing_lane(None).await;
    accept_prompt(&lane).await;
    let operation_id = lane
        .state()
        .operation
        .as_ref()
        .map(|operation| operation.meta.operation_id.clone())
        .expect("the live operation");
    finish_operation(&lane).await;
    commit_writes(
        lane.session(),
        vec![crate::harness::session::values::delete_value_write(
            &crate::harness::session::values::operation_result(&operation_id),
        )],
    )
    .await;
    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the missing record faults the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason
                    .to_string()
                    .contains(&format!("Lane \"main\" is missing result {operation_id}")),
                "the missing record names its invariant: {reason}",
            );
        }
    }

    // The malformed record.
    let lane = capturing_lane(None).await;
    accept_prompt(&lane).await;
    let operation_id = lane
        .state()
        .operation
        .as_ref()
        .map(|operation| operation.meta.operation_id.clone())
        .expect("the live operation");
    finish_operation(&lane).await;
    commit_writes(
        lane.session(),
        vec![raw_write(
            &crate::harness::session::values::operation_result(&operation_id).address,
            json!(42),
        )],
    )
    .await;
    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the malformed record faults the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason.to_string().contains("Operation result is malformed"),
                "the malformed record names its parse failure: {reason}",
            );
        }
    }
}

/// The remaining tools-batch parse invariants: the non-tool-call source
/// block, the malformed args, the malformed checkpoint, the malformed and
/// non-message and mismatched staged results.
#[expect(
    clippy::too_many_lines,
    reason = "the parse invariants share the fixture; one body keeps the fixture shape visible"
)]
#[tokio::test]
async fn watch_faults_on_the_tools_batch_parse_invariants() {
    let call = |source_index: u64, result_entry_id: &str, status: &str| {
        serde_json::from_value::<crate::harness::session::types::ToolCall>(json!({
            "sourceIndex": source_index,
            "resultEntryId": result_entry_id,
            "status": status,
            "replay": "safe",
            "terminate": false,
        }))
        .expect("the call wire")
    };
    let batch = |assistant_entry_id: &str, calls: Vec<crate::harness::session::types::ToolCall>| {
        OperationState::Tools(ToolsOperation {
            scope: operation_scope(),
            batch: ToolBatch {
                assistant_entry_id: assistant_entry_id.to_owned(),
                configuration: lane_configuration(),
                turn_id: "turn".to_owned(),
                calls,
            },
        })
    };

    // A text block where the call's source index names a tool call.
    let lane = capturing_lane(None).await;
    let assistant_entry = AgentLane::append_message(
        &lane,
        assistant_wire(
            &json!([{ "type": "text", "text": "not a call" }]),
            "stop",
            false,
        ),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        batch(
            &assistant_entry,
            vec![call(0, "result-1", "effect_pending")],
        ),
    )
    .await;
    commit_writes(
        lane.session(),
        vec![raw_write(
            &crate::harness::session::values::operation_tool_args(
                &lane
                    .state()
                    .operation
                    .as_ref()
                    .map(|operation| operation.meta.operation_id.clone())
                    .expect("the operation"),
                "turn",
                0,
            )
            .address,
            json!({}),
        )],
    )
    .await;
    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the text block faults the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason
                    .to_string()
                    .contains("does not name a tool-call block"),
                "the text block names its invariant: {reason}",
            );
        }
    }

    // The malformed persisted arguments.
    let lane = capturing_lane(None).await;
    let assistant_entry = AgentLane::append_message(
        &lane,
        assistant_wire(
            &json!([{ "type": "toolCall", "id": "call-1", "name": "tool-1", "arguments": {} }]),
            "toolUse",
            false,
        ),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    let operation_id = lane
        .state()
        .operation
        .as_ref()
        .map(|operation| operation.meta.operation_id.clone())
        .expect("the operation");
    patch_live_state(
        &lane,
        batch(
            &assistant_entry,
            vec![call(0, "result-1", "effect_pending")],
        ),
    )
    .await;
    commit_writes(
        lane.session(),
        vec![raw_write(
            &crate::harness::session::values::operation_tool_args(&operation_id, "turn", 0).address,
            json!(42),
        )],
    )
    .await;
    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the malformed args fault the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason.to_string().contains("Tool arguments are malformed"),
                "the malformed args name their parse failure: {reason}",
            );
        }
    }

    // The malformed checkpoint.
    let lane = capturing_lane(None).await;
    let assistant_entry = AgentLane::append_message(
        &lane,
        assistant_wire(
            &json!([{ "type": "toolCall", "id": "call-1", "name": "tool-1", "arguments": {} }]),
            "toolUse",
            false,
        ),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    let operation_id = lane
        .state()
        .operation
        .as_ref()
        .map(|operation| operation.meta.operation_id.clone())
        .expect("the operation");
    patch_live_state(
        &lane,
        batch(
            &assistant_entry,
            vec![call(0, "result-1", "effect_pending")],
        ),
    )
    .await;
    commit_writes(
        lane.session(),
        vec![
            raw_write(
                &crate::harness::session::values::operation_tool_args(&operation_id, "turn", 0)
                    .address,
                json!({}),
            ),
            raw_write(
                &crate::harness::session::values::pending_tool_output(&operation_id, "result-1")
                    .address,
                json!(42),
            ),
        ],
    )
    .await;
    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the malformed checkpoint faults the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason
                    .to_string()
                    .contains("Pending tool output is malformed"),
                "the malformed checkpoint names its parse failure: {reason}",
            );
        }
    }

    // The mismatched staged result.
    let lane = capturing_lane(None).await;
    let assistant_entry = AgentLane::append_message(
        &lane,
        assistant_wire(
            &json!([{ "type": "toolCall", "id": "call-2", "name": "tool-2", "arguments": {} }]),
            "toolUse",
            false,
        ),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        batch(&assistant_entry, vec![call(0, "result-2", "outcome_ready")]),
    )
    .await;
    commit_writes(
        lane.session(),
        vec![
            crate::harness::session::values::set_value_write(
                &crate::harness::session::values::pending_entry("result-2"),
                crate::harness::session::types::PendingEntry::Message {
                    payload: Box::new(
                        serde_json::from_value(json!({
                            "role": "toolResult",
                            "toolCallId": "other-call",
                            "toolName": "tool-2",
                            "content": [{ "type": "text", "text": "done" }],
                            "isError": false,
                            "timestamp": 2,
                        }))
                        .expect("the tool result wire"),
                    ),
                },
            )
            .expect("staged write"),
        ],
    )
    .await;
    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the mismatched staged result faults the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason
                    .to_string()
                    .contains("Tool call result-2 has a mismatched staged result"),
                "the mismatched staged result names its invariant: {reason}",
            );
        }
    }
}

/// The watch handle's start and resnapshot surfaces the capture fixture
/// installs, upstream's `(snapshot) => ({...})` literal.
#[tokio::test]
async fn the_recording_watch_replays_its_snapshot_and_drains_its_listener() {
    let lane = capturing_lane(None).await;
    let watch = AgentLane::watch(&lane, &background_context())
        .await
        .expect("watch serves");
    let listener: crate::harness::agent_harness::EventListener =
        Arc::new(|_event, _context| Box::pin(std::future::ready(Ok(()))));
    watch.start(listener);
    let resnapshotted = watch
        .resnapshot(&background_context())
        .await
        .expect("the resnapshot");
    assert_eq!(
        resnapshotted,
        WatchHandle::<crate::harness::agent_harness::LaneSnapshot>::snapshot(&*watch),
        "the resnapshot replays the captured snapshot",
    );
    watch.unsubscribe();
}

/// The controlled storage forwards its reads and lifecycle to the memory
/// backend, the decorator contract the suites' spies rest on: each read's
/// result matches the delegate's own.
#[tokio::test]
async fn the_controlled_storage_forwards_its_reads() {
    let plain = Arc::new(MemoryStorage::new(MemoryStorageOptions::default()));
    let storage = ControlledStorage::new(Arc::clone(&plain));
    let debug = format!("{storage:?}");
    assert!(
        debug.contains("ControlledStorage"),
        "the storage's debug header: {debug}",
    );
    assert!(debug.contains("get_value_calls: 0"));

    let address = crate::harness::session::values::branch_tip("main");
    let forward = storage
        .scan_values(&address.address, &background_context())
        .await;
    let delegate = plain
        .scan_values(&address.address, &background_context())
        .await;
    assert_eq!(
        format!("{forward:?}"),
        format!("{delegate:?}"),
        "the tip scan forwards"
    );

    let query = crate::harness::session::types::StorageBranchScan::default();
    let forward = storage.scan_branch(&query, &background_context()).await;
    let delegate = plain.scan_branch(&query, &background_context()).await;
    assert_eq!(
        format!("{forward:?}"),
        format!("{delegate:?}"),
        "the branch scan forwards"
    );

    let forward = storage
        .scan_branch_structure(&query, &background_context())
        .await;
    let delegate = plain
        .scan_branch_structure(&query, &background_context())
        .await;
    assert_eq!(
        format!("{forward:?}"),
        format!("{delegate:?}"),
        "the structural scan forwards",
    );

    let entries = crate::harness::session::types::EntryScan::default();
    let forward = storage.scan_entries(&entries, &background_context()).await;
    let delegate = plain.scan_entries(&entries, &background_context()).await;
    assert_eq!(
        format!("{forward:?}"),
        format!("{delegate:?}"),
        "the entry scan forwards"
    );

    let usage = crate::harness::session::types::UsageScan::default();
    let forward = storage.scan_usage(&usage, &background_context()).await;
    let delegate = plain.scan_usage(&usage, &background_context()).await;
    assert_eq!(
        format!("{forward:?}"),
        format!("{delegate:?}"),
        "the usage scan forwards"
    );

    let forward = storage.get_stats(&background_context()).await;
    let delegate = plain.get_stats(&background_context()).await;
    assert_eq!(
        format!("{forward:?}"),
        format!("{delegate:?}"),
        "the stats read forwards"
    );

    let forward = storage.close(&background_context()).await;
    assert_eq!(
        forward,
        plain.close(&background_context()).await,
        "the close forwards"
    );
}

/// The runtime config's message conversion, the closure body the lane's
/// planners invoke.
#[tokio::test]
async fn the_runtime_config_converts_messages() {
    let config = runtime_config();
    let messages: Vec<AgentMessage> = vec![
        serde_json::from_value(json!({
            "role": "user",
            "content": "hello",
            "timestamp": 1,
        }))
        .expect("the user wire"),
    ];
    let converted = (config.to_provider_messages)(&messages, &background_context()).await;
    assert_eq!(converted.len(), 1, "the conversion carried the message");
}

/// The staged result's parse invariants: the malformed payload, the custom
/// entry, and the non-toolResult message the staged arm rejects.
#[expect(
    clippy::too_many_lines,
    reason = "the three parse invariants share the fixture; one body keeps the fixture shape visible"
)]
#[tokio::test]
async fn watch_faults_on_the_staged_result_parse_invariants() {
    let call = || {
        serde_json::from_value::<crate::harness::session::types::ToolCall>(json!({
            "sourceIndex": 0,
            "resultEntryId": "result-1",
            "status": "outcome_ready",
            "terminate": false,
        }))
        .expect("the call wire")
    };
    let staged_for = |payload: serde_json::Value| {
        crate::harness::session::values::set_value_write(
            &crate::harness::session::values::pending_entry("result-1"),
            crate::harness::session::types::PendingEntry::Message {
                payload: Box::new(
                    serde_json::from_value(payload).expect("the staged message wire"),
                ),
            },
        )
        .expect("staged write")
    };

    // The malformed staged payload.
    let lane = capturing_lane(None).await;
    let assistant_entry = AgentLane::append_message(
        &lane,
        assistant_wire(
            &json!([{ "type": "toolCall", "id": "call-1", "name": "tool-1", "arguments": {} }]),
            "toolUse",
            false,
        ),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        OperationState::Tools(ToolsOperation {
            scope: operation_scope(),
            batch: ToolBatch {
                assistant_entry_id: assistant_entry,
                configuration: lane_configuration(),
                turn_id: "turn".to_owned(),
                calls: vec![call()],
            },
        }),
    )
    .await;
    commit_writes(
        lane.session(),
        vec![raw_write(
            &crate::harness::session::values::pending_entry("result-1").address,
            json!(42),
        )],
    )
    .await;
    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the malformed staged result faults the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason
                    .to_string()
                    .contains("Pending entry payload is malformed"),
                "the malformed staged result names its parse failure: {reason}",
            );
        }
    }

    // The custom staged payload.
    let lane = capturing_lane(None).await;
    let assistant_entry = AgentLane::append_message(
        &lane,
        assistant_wire(
            &json!([{ "type": "toolCall", "id": "call-1", "name": "tool-1", "arguments": {} }]),
            "toolUse",
            false,
        ),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        OperationState::Tools(ToolsOperation {
            scope: operation_scope(),
            batch: ToolBatch {
                assistant_entry_id: assistant_entry,
                configuration: lane_configuration(),
                turn_id: "turn".to_owned(),
                calls: vec![call()],
            },
        }),
    )
    .await;
    commit_writes(
        lane.session(),
        vec![
            crate::harness::session::values::set_value_write(
                &crate::harness::session::values::pending_entry("result-1"),
                crate::harness::session::types::PendingEntry::Custom {
                    custom_type: "note".to_owned(),
                    payload: None,
                },
            )
            .expect("staged write"),
        ],
    )
    .await;
    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the custom staged result faults the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason
                    .to_string()
                    .contains("Tool call result-1 is missing its staged result"),
                "the custom staged result names its invariant: {reason}",
            );
        }
    }

    // The non-toolResult staged message.
    let lane = capturing_lane(None).await;
    let assistant_entry = AgentLane::append_message(
        &lane,
        assistant_wire(
            &json!([{ "type": "toolCall", "id": "call-1", "name": "tool-1", "arguments": {} }]),
            "toolUse",
            false,
        ),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        OperationState::Tools(ToolsOperation {
            scope: operation_scope(),
            batch: ToolBatch {
                assistant_entry_id: assistant_entry,
                configuration: lane_configuration(),
                turn_id: "turn".to_owned(),
                calls: vec![call()],
            },
        }),
    )
    .await;
    commit_writes(
        lane.session(),
        vec![staged_for(json!({
            "role": "user",
            "content": "not a result",
            "timestamp": 1,
        }))],
    )
    .await;
    let Err(error) = AgentLane::watch(&lane, &background_context()).await else {
        panic!("the non-toolResult staged message faults the capture");
    };
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => {
            assert!(
                reason
                    .to_string()
                    .contains("Tool call result-1 is missing its staged result"),
                "the non-toolResult staged message names its invariant: {reason}",
            );
        }
    }
}

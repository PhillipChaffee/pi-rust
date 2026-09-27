//! The drive-ownership and progress-channel suite, ported 1:1 from upstream
//! `test/harness/runtime/progress.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements the port makes and the tests bind:
//! - `lane.activeDrive = drive` restates as the package-internal
//!   [`Lane::set_active_drive`] setter; the suite lives as a unit-test
//!   module to reach it.
//! - `vi.spyOn(storage, "getValue")` restates as the controlled storage's
//!   read counter.
//! - the unrepresentable API-surface assertion (`"beginAbort" in gate` is
//!   false) rides the type system: `Gate` carries no control methods —
//!   noted, not asserted.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the boundary tests pin outcomes; a violated expectation panics the test by design"
)]
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_ai::models::create_models;
use pi_ai::utils::assistant_message_frame::AssistantMessageFrame;
use serde_json::json;

use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::DriveWaitReason;
use crate::harness::context::background_context;
use crate::harness::context::with_abort_signal;
use crate::harness::hooks::HookRegistry;
use crate::harness::runtime::lane::EmitBatch;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::progress::ProgressChannel;
use crate::harness::runtime::progress::open_frame_progress;
use crate::harness::runtime::progress::open_tool_progress;
use crate::harness::runtime::test_support::ControlledStorage;
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::deferred_effect_pending;
use crate::harness::runtime::test_support::lane_configuration;
use crate::harness::runtime::test_support::operation_scope;
use crate::harness::runtime::test_support::tools_batch_state;
use crate::harness::runtime::test_support::{gate_next_commit, generation_context};
use crate::harness::runtime::test_support::{noop_emit_batch, noop_hook_reporter};
use crate::harness::runtime::test_support::{passthrough_fault_handler, patch_live_state};
use crate::harness::runtime::test_support::{runtime_config, unused_watch_installer};
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneState;
use crate::harness::runtime::types::LiveOperation;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::memory::MemoryStorage;
use crate::harness::session::memory::MemoryStorageOptions;
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::CheckpointData;
use crate::harness::session::types::CheckpointOperation;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationMeta;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::SessionError;
use crate::harness::session::values as stored_values;
use crate::harness::session::values::StoredValue;
use crate::harness::session::values::delete_value_write;
use crate::harness::session::values::set_value_write;
use crate::types::AgentToolContent;
use crate::types::AgentToolResult;
use pi_ai::types::TextContent;

fn next_session_id() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("progress-{}", COUNTER.fetch_add(1, Ordering::SeqCst))
}

/// The assistant-effect-pending leaf upstream's `assistantEffectPending`
/// builds.
fn assistant_effect_pending(response_entry_id: &str) -> OperationState {
    OperationState::AssistantEffectPending(AssistantEffectPendingOperation {
        scope: operation_scope(),
        generation_context: generation_context(),
        attempt: 1,
        response_entry_id: response_entry_id.to_owned(),
        usage_id: "usage".to_owned(),
        intended_output_limit: 100,
        context_window: 1_000,
    })
}

/// The tools leaf the checkpoint test builds, upstream's
/// `{ ...runScope(), at: "tools", batch: { ... } }`.
fn tools_state(invocation_id: &str) -> OperationState {
    tools_batch_state(
        "assistant",
        vec![
            serde_json::from_value(json!({
                "sourceIndex": 0,
                "resultEntryId": invocation_id,
                "status": "effect_pending",
                "replay": "safe",
            }))
            .expect("tool call wire"),
        ],
    )
}

/// The fixture upstream's `createFixture` builds: the storage-backed
/// session over the failing storage, the hand-built projection (no
/// restore), and the drive pass installed as the lane's active owner.
struct ProgressFixture {
    lane: Arc<Lane>,
    drive: Arc<Drive>,
    storage: Arc<ControlledStorage>,
}

async fn create_fixture(state: OperationState) -> ProgressFixture {
    let storage = Arc::new(ControlledStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = Arc::new(crate::harness::runtime::test_support::runtime_session(
        next_session_id(),
        storage.clone(),
    ));
    let meta = OperationMeta {
        operation_id: "operation".to_owned(),
        lane: "main".to_owned(),
        source_tip_id: None,
        started_at: 1,
        intent: OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
    };
    let configuration = lane_configuration();
    let projection = LaneState {
        tip_id: None,
        configuration: configuration.clone(),
        inbox: Vec::new(),
        last_operation_id: None,
        operation: Some(LiveOperation {
            meta: meta.clone(),
            state: state.clone(),
        }),
    };
    let emit_batch: EmitBatch = noop_emit_batch();
    let lane = Lane::new(
        "main",
        session.clone(),
        Arc::new(create_models(None)),
        Arc::new(HookRegistry::new(noop_hook_reporter())),
        projection,
        passthrough_fault_handler(),
        emit_batch,
        unused_watch_installer(),
        Arc::new(runtime_config),
    );
    let writes = vec![
        set_value_write(&stored_values::branch_tip("main"), Option::<String>::None)
            .expect("tip write"),
        set_value_write(&stored_values::lane_config("main"), configuration.clone())
            .expect("config write"),
        set_value_write(
            &stored_values::lane_state("main"),
            crate::harness::session::types::LaneState {
                current_operation_id: Some("operation".to_owned()),
                last_operation_id: None,
                inbox: Vec::new(),
            },
        )
        .expect("state write"),
        set_value_write(&stored_values::operation_meta("operation"), meta.clone())
            .expect("meta write"),
        set_value_write(&stored_values::operation_state("operation"), state).expect("state write"),
    ];
    commit_writes(&session, writes).await;
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: "operation".to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    lane.set_active_drive(Some(Arc::clone(&drive)));
    ProgressFixture {
        lane: Arc::new(lane),
        drive,
        storage,
    }
}

/// The pending-frame list the drain tests inspect, oldest first, each
/// element's wire value.
async fn read_pending_frames(fixture: &ProgressFixture) -> Vec<serde_json::Value> {
    let elements = fixture
        .lane
        .session()
        .read_list(
            &stored_values::pending_assistant_frames("operation", "response").address,
            None,
            &background_context(),
        )
        .await
        .expect("read frames");
    elements.into_iter().map(|element| element.value).collect()
}

/// The decline outcome the queued-frame tests pin: the frame list empty and
/// the decline decided without control reads.
async fn assert_queued_frame_declined(fixture: &ProgressFixture) {
    assert!(
        read_pending_frames(fixture).await.is_empty(),
        "the queued frame was declined"
    );
    assert_eq!(
        fixture.storage.get_value_calls(),
        0,
        "the decline decided without control reads"
    );
}

#[tokio::test]
async fn settles_or_rejects_the_shared_completion_one_shot() {
    // Upstream also asserts `"beginAbort" in failed.gate` is false; the Gate
    // type carries no control surface, the separation is structural.
    let settled = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: "operation".to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    let outcome = DriveOutcome::Waiting {
        operation_id: "operation".to_owned(),
        reason: DriveWaitReason::Retry { not_before: 10 },
    };
    settled.settle(outcome.clone());
    settled.fail(lane_error(SessionError::Message("late failure".to_owned())));
    assert_eq!(
        settled.completion().await.expect("the settlement wins"),
        outcome
    );

    let failed = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: "operation".to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    let error = lane_error(SessionError::Message("drive failed".to_owned()));
    failed.fail(Arc::clone(&error));
    let rejected = failed.completion().await.expect_err("the failure wins");
    assert!(Arc::ptr_eq(&rejected, &error));
}

#[tokio::test]
async fn strips_invocation_cancellation_from_pass_context_and_owns_policy_and_gate_control() {
    let (_, controller) = pi_chord::context::with_cancel(&background_context());
    let context = with_abort_signal(controller.signal().clone(), &background_context());
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: "operation".to_owned(),
            wait_for_retry: Some(true),
            poll_deferred: Some(true),
        },
        &context,
    ));

    assert_eq!(drive.operation_id, "operation");
    assert!(drive.context.abort_signal().is_none());
    assert!(drive.wait_for_retry);
    assert_eq!(drive.deferred_permits, 1);

    let closed = lane_error(SessionError::Message("closed".to_owned()));
    drive.close_gate(closed);
    assert!(drive.gate.signal().aborted());
    assert!(
        drive.gate.admit(|| ()).is_err(),
        "the closed gate rejects admission"
    );
}

#[tokio::test]
async fn enqueues_assistant_frames_in_order_seals_admission_and_drops_late_writes() {
    let response_entry_id = "response";
    let fixture = create_fixture(assistant_effect_pending(response_entry_id)).await;
    let progress = open_frame_progress(&fixture.lane, &fixture.drive, response_entry_id);
    let frames = vec![
        AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: "a".to_owned(),
        },
        AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: "b".to_owned(),
        },
    ];

    progress.write(frames[0].clone());
    progress.write(frames[1].clone());
    progress.seal();
    progress.write(AssistantMessageFrame::TextDelta {
        content_index: 0,
        delta: "late".to_owned(),
    });
    progress.drain().await.expect("drain");

    assert_eq!(
        serde_json::Value::Array(read_pending_frames(&fixture).await),
        serde_json::to_value(&frames).expect("frames wire"),
    );
}

/// The late-write rig the two queued-frame decline tests share: the fixture
/// over the assistant-effect-pending projection, the frame channel opened on
/// it, and the commit gate parked on the controlled storage with its release
/// handle.
async fn late_write_rig() -> (
    ProgressFixture,
    ProgressChannel<AssistantMessageFrame>,
    tokio::sync::oneshot::Receiver<()>,
    tokio::sync::oneshot::Sender<()>,
) {
    let response_entry_id = "response";
    let fixture = create_fixture(assistant_effect_pending(response_entry_id)).await;
    let progress = open_frame_progress(&fixture.lane, &fixture.drive, response_entry_id);
    let (started, release) = gate_next_commit(&fixture.storage);
    (fixture, progress, started, release)
}

#[tokio::test]
async fn declines_a_queued_frame_after_the_authoritative_projection_leaves_its_phase() {
    let (fixture, progress, started, release) = late_write_rig().await;
    let moving = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move {
            let live_state = lane
                .state()
                .operation
                .as_ref()
                .expect("the live operation")
                .state
                .clone();
            let next_state = OperationState::Checkpoint(CheckpointOperation {
                scope: crate::harness::session::types::operation_scope_of(&live_state),
                checkpoint: CheckpointData {
                    continuation: Continuation::MayFinish {
                        include_final_assistant: true,
                    },
                    trigger_entry_id: "response".to_owned(),
                },
            });
            patch_live_state(&lane, next_state).await;
        })
    };
    started.await.expect("commit started");
    progress.write(frame_delta(0, "late"));
    let _ = release.send(());
    moving.await.expect("move join");
    progress.drain().await.expect("drain");
    assert_queued_frame_declined(&fixture).await;
}

#[tokio::test]
async fn declines_a_queued_frame_after_terminal_projection_publication() {
    let (fixture, progress, started, release) = late_write_rig().await;
    let ending = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move {
            clear_operation(&lane).await;
        })
    };
    started.await.expect("commit started");
    progress.write(frame_delta(0, "late"));
    let _ = release.send(());
    ending.await.expect("ending join");
    progress.drain().await.expect("drain");
    assert_queued_frame_declined(&fixture).await;
}

#[tokio::test]
async fn replaces_tool_checkpoints_in_invocation_order() {
    let invocation_id = "result";
    let fixture = create_fixture(tools_state(invocation_id)).await;
    let progress = open_tool_progress(&fixture.lane, &fixture.drive, "turn", 0, invocation_id);
    progress.write(checkpoint_of("first"));
    progress.write(checkpoint_of("second"));
    progress.drain().await.expect("drain");
    let stored = fixture
        .lane
        .session()
        .get_value(
            &stored_values::pending_tool_output("operation", invocation_id).address,
            &background_context(),
        )
        .await
        .expect("read checkpoint")
        .expect("stored checkpoint");
    assert_eq!(
        stored.value.get("content"),
        Some(&json!([{ "type": "text", "text": "second" }])),
    );
}

#[tokio::test]
async fn retains_the_rejecting_write_promise_so_drain_propagates_commit_failure() {
    let response_entry_id = "response";
    let fixture = create_fixture(assistant_effect_pending(response_entry_id)).await;
    fixture.storage.set_failure(Some(SessionError::Message(
        "frame commit failed".to_owned(),
    )));
    let progress = open_frame_progress(&fixture.lane, &fixture.drive, response_entry_id);

    progress.write(AssistantMessageFrame::TextDelta {
        content_index: 0,
        delta: "lost".to_owned(),
    });

    let drained = progress.drain().await;
    let error = drained.expect_err("drain propagates the failure");
    assert_eq!(error.to_string(), "frame commit failed");
}

// The boundary additions: the reader, the channel Debug, the write-less
// drain, and the still-owns guards the 1:1 suite leaves.

fn frame_delta(index: u64, delta: &str) -> AssistantMessageFrame {
    AssistantMessageFrame::TextDelta {
        content_index: index,
        delta: delta.to_owned(),
    }
}

#[tokio::test]
async fn reads_the_pending_assistant_frames_oldest_first() {
    let response_entry_id = "response";
    let fixture = create_fixture(assistant_effect_pending(response_entry_id)).await;
    let address = stored_values::pending_assistant_frames("operation", response_entry_id);
    commit_writes(
        fixture.lane.session(),
        vec![
            crate::harness::session::values::append_list_write(&address, frame_delta(0, "a"))
                .expect("frame append"),
            crate::harness::session::values::append_list_write(&address, frame_delta(0, "b"))
                .expect("frame append"),
        ],
    )
    .await;

    let frames = crate::harness::runtime::progress::read_assistant_frames(
        fixture.lane.session().as_ref(),
        "operation",
        response_entry_id,
        &background_context(),
    )
    .await
    .expect("the frames read");
    assert_eq!(frames, vec![frame_delta(0, "a"), frame_delta(0, "b")]);

    // A malformed stored element reports its parse failure.
    commit_writes(
        fixture.lane.session(),
        vec![crate::harness::session::values::Write::ListAppend(
            crate::harness::session::values::ListAppendWrite {
                kind: "list".to_owned(),
                op: "append".to_owned(),
                namespace: address.address.namespace.clone(),
                key: address.address.key.clone(),
                value: json!(42),
            },
        )],
    )
    .await;
    let read = crate::harness::runtime::progress::read_assistant_frames(
        fixture.lane.session().as_ref(),
        "operation",
        response_entry_id,
        &background_context(),
    )
    .await;
    let error = read.expect_err("the malformed frame rejects");
    assert!(
        error
            .to_string()
            .contains("Pending assistant frame is malformed"),
        "the malformed frame names its parse failure: {error}",
    );
}

#[tokio::test]
async fn pages_long_frame_lists_through_the_cursor() {
    let response_entry_id = "response";
    let fixture = create_fixture(assistant_effect_pending(response_entry_id)).await;
    let address = stored_values::pending_assistant_frames("operation", response_entry_id);
    let count = 1_001;
    let mut writes = Vec::new();
    for index in 0..count {
        writes.push(
            crate::harness::session::values::append_list_write(
                &address,
                frame_delta(0, &format!("f-{index}")),
            )
            .expect("frame append"),
        );
    }
    commit_writes(fixture.lane.session(), writes).await;

    let frames = crate::harness::runtime::progress::read_assistant_frames(
        fixture.lane.session().as_ref(),
        "operation",
        response_entry_id,
        &background_context(),
    )
    .await
    .expect("the paged read");
    assert_eq!(
        frames.len(),
        count,
        "every element pages through the cursor"
    );
    let AssistantMessageFrame::TextDelta { delta, .. } = &frames[count - 1] else {
        panic!("the last frame");
    };
    assert_eq!(delta, "f-1000", "the paged read preserves the order");
}

#[tokio::test]
async fn drain_resolves_when_no_write_was_published() {
    let response_entry_id = "response";
    let fixture = create_fixture(assistant_effect_pending(response_entry_id)).await;
    let progress = open_frame_progress(&fixture.lane, &fixture.drive, response_entry_id);
    progress.drain().await.expect("the empty drain");
}

#[tokio::test]
async fn the_progress_channel_renders_a_summary_debug() {
    let response_entry_id = "response";
    let fixture = create_fixture(assistant_effect_pending(response_entry_id)).await;
    let progress = open_frame_progress(&fixture.lane, &fixture.drive, response_entry_id);
    assert_eq!(format!("{progress:?}"), "ProgressChannel(..)");
}

#[tokio::test]
async fn writes_frames_through_a_deferred_effect_pending_phase() {
    let response_entry_id = "response";
    let fixture = create_fixture(deferred_effect_pending("source", response_entry_id, 0)).await;
    let progress = open_frame_progress(&fixture.lane, &fixture.drive, response_entry_id);
    progress.write(frame_delta(0, "deferred"));
    progress.drain().await.expect("drain");

    assert_eq!(
        read_pending_frames(&fixture).await.len(),
        1,
        "the deferred phase still owns the write"
    );
}

/// The late tool-checkpoint the decline tests probe: the channel opened on
/// the fixture, the write drained, and the stored checkpoint read back.
async fn declined_tool_checkpoint(fixture: &ProgressFixture) -> Option<StoredValue> {
    let progress = open_tool_progress(&fixture.lane, &fixture.drive, "turn", 0, "result");
    progress.write(checkpoint_of("late"));
    progress.drain().await.expect("drain");
    fixture
        .lane
        .session()
        .get_value(
            &stored_values::pending_tool_output("operation", "result").address,
            &background_context(),
        )
        .await
        .expect("read checkpoint")
}

#[tokio::test]
async fn declines_tool_checkpoints_outside_the_tools_phase() {
    let response_entry_id = "response";
    let fixture = create_fixture(assistant_effect_pending(response_entry_id)).await;
    let stored = declined_tool_checkpoint(&fixture).await;
    assert!(stored.is_none(), "the non-tools phase declined the write");
}

#[tokio::test]
async fn declines_tool_checkpoints_after_the_operation_clears() {
    let fixture = create_fixture(tools_state("result")).await;
    clear_operation(&fixture.lane).await;
    let stored = declined_tool_checkpoint(&fixture).await;
    assert!(stored.is_none(), "the cleared operation declined the write");
}

fn checkpoint_of(text: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![AgentToolContent::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        details: json!({}),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

/// Commits the terminal projection the queued-frame decline tests drive,
/// upstream's cleared-operation write list restated: the operation's meta
/// and state deleted and the lane state cleared.
async fn clear_operation(lane: &Lane) {
    lane.command::<(), _>(
        move |state, _session, _context| {
            Box::pin(async move {
                Ok(crate::harness::runtime::types::LaneCommand::Commit {
                    writes: vec![
                        delete_value_write(&stored_values::operation_meta("operation")),
                        delete_value_write(&stored_values::operation_state("operation")),
                        set_value_write(
                            &stored_values::lane_state("main"),
                            crate::harness::session::types::LaneState {
                                current_operation_id: None,
                                last_operation_id: None,
                                inbox: state.inbox.clone(),
                            },
                        )
                        .expect("state write"),
                    ],
                    next: LaneState {
                        operation: None,
                        ..state
                    },
                    materialize: Arc::new(|_commit| ()),
                    events: None,
                })
            })
        },
        &background_context(),
    )
    .await
    .expect("the terminal projection commits");
}

// The storage-error arms and the channel guards: the failing frame read,
// the failing write plan, the vanished writer, and the completed call's
// late checkpoint.

use crate::harness::runtime::progress::open_progress;

/// The tools leaf the planned-call checkpoint test builds: the call never
/// reached its effect, the still-owns guard declines the write.
fn planned_tools_state(invocation_id: &str) -> OperationState {
    tools_batch_state(
        "assistant",
        vec![
            serde_json::from_value(json!({
                "sourceIndex": 0,
                "resultEntryId": invocation_id,
                "status": "planned",
            }))
            .expect("tool call wire"),
        ],
    )
}

#[tokio::test]
async fn a_failing_frame_read_reports_its_storage_error() {
    let response_entry_id = "response";
    let fixture = create_fixture(assistant_effect_pending(response_entry_id)).await;
    fixture.storage.arm_read_failure(
        None,
        Some(&format!("operation:{response_entry_id}")),
        SessionError::Message("read failed".to_owned()),
    );
    let read = crate::harness::runtime::progress::read_assistant_frames(
        fixture.lane.session().as_ref(),
        "operation",
        response_entry_id,
        &background_context(),
    )
    .await;
    let error = read.expect_err("the frame read rejects");
    assert_eq!(
        error.to_string(),
        "read failed",
        "the list read's storage error carried",
    );
}

/// The write plan's failure retains in the settlement and surfaces through
/// the drain, upstream's rejected `latest` promise.
#[tokio::test]
async fn a_failing_write_plan_retains_its_failure_in_the_drain() {
    let fixture = create_fixture(assistant_effect_pending("response")).await;
    let progress = open_progress::<AssistantMessageFrame>(
        &fixture.lane,
        &fixture.drive,
        Arc::new(|_frame: &AssistantMessageFrame| {
            Err(SessionError::Message("plan failed".to_owned()))
        }),
        Arc::new(|_state: &LaneState| true),
    );
    progress.write(frame_delta(0, "lost"));
    let error = progress
        .drain()
        .await
        .expect_err("drain propagates the plan failure");
    assert_eq!(error.to_string(), "plan failed");
}

/// The writer vanishing between the write and its settlement releases the
/// drain, upstream's `latest` rejection never surfacing.
#[tokio::test]
async fn a_vanished_writer_releases_the_drain() {
    let fixture = create_fixture(assistant_effect_pending("response")).await;
    let progress = open_progress::<AssistantMessageFrame>(
        &fixture.lane,
        &fixture.drive,
        Arc::new(|_frame: &AssistantMessageFrame| {
            Ok(stored_values::append_list_write(
                &stored_values::pending_assistant_frames("operation", "response"),
                frame_delta(0, "lost"),
            )
            .expect("frame write"))
        }),
        Arc::new(|_state: &LaneState| -> bool { panic!("the writer vanished") }),
    );
    progress.write(frame_delta(0, "lost"));
    progress
        .drain()
        .await
        .expect("the vanished writer's drain resolved");
}

/// The completed call declines its late checkpoint: the still-owns guard's
/// phase match leaves only the effect-pending calls.
#[tokio::test]
async fn a_completed_call_declines_its_late_checkpoint() {
    let invocation_id = "result";
    let fixture = create_fixture(planned_tools_state(invocation_id)).await;
    let progress = open_tool_progress(&fixture.lane, &fixture.drive, "turn", 0, invocation_id);
    progress.write(checkpoint_of("late"));
    progress.drain().await.expect("drain");
    let stored = declined_tool_checkpoint(&fixture).await;
    assert!(
        stored.is_none(),
        "the planned call's checkpoint never landed",
    );
}

/// The fixture commits' failures surface as the helpers' panics, upstream's
/// awaited `commit` throw.
#[tokio::test]
async fn the_fixture_commit_helpers_panic_on_their_failures() {
    // The commit-writes helper's seeded commit.
    let fixture = create_fixture(assistant_effect_pending("response")).await;
    fixture
        .storage
        .set_failure(Some(SessionError::Message("commit failed".to_owned())));
    let joined = {
        let session = fixture.lane.session().clone();
        tokio::spawn(async move { commit_writes(&session, Vec::new()).await })
    };
    assert!(
        joined.await.is_err(),
        "the failing commit panicked the helper",
    );

    // The seed helper's commit.
    let storage = Arc::new(ControlledStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = Arc::new(crate::harness::runtime::test_support::runtime_session(
        next_session_id(),
        storage.clone(),
    ));
    storage.set_failure(Some(SessionError::Message("seed failed".to_owned())));
    let joined = tokio::spawn(async move {
        crate::harness::runtime::test_support::seed_main_lane_values(&session, None).await
    });
    let outcome = joined.await.expect("seed join");
    assert!(outcome.is_err(), "the armed failure rejected the seed");
}

/// The unused watch installer raises when reached, upstream's `unusedWatch`
/// throw the module docs pin.
#[tokio::test]
async fn the_unused_watch_installer_raises_when_reached() {
    let fixture = create_fixture(assistant_effect_pending("response")).await;
    let joined = {
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move {
            crate::harness::agent_harness::AgentLane::watch(lane.as_ref(), &background_context())
                .await
        })
    };
    assert!(
        joined.await.is_err(),
        "reaching the unused watch raised, upstream's unusedWatch throw",
    );
}

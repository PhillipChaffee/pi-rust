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
use crate::harness::runtime::progress::open_frame_progress;
use crate::harness::runtime::progress::open_tool_progress;
use crate::harness::runtime::test_support::ControlledStorage;
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::gate_next_commit;
use crate::harness::runtime::test_support::generation_context;
use crate::harness::runtime::test_support::lane_configuration;
use crate::harness::runtime::test_support::noop_emit_batch;
use crate::harness::runtime::test_support::noop_hook_reporter;
use crate::harness::runtime::test_support::operation_scope;
use crate::harness::runtime::test_support::passthrough_fault_handler;
use crate::harness::runtime::test_support::patch_live_state;
use crate::harness::runtime::test_support::runtime_config;
use crate::harness::runtime::test_support::unused_watch_installer;
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
use crate::harness::session::types::DeferredEffectPendingOperation;
use crate::harness::session::types::DeferredScope;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationMeta;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::SessionError;
use crate::harness::session::values as stored_values;
use crate::harness::session::values::delete_value_write;
use crate::harness::session::values::set_value_write;
use crate::harness::types::AgentHarnessStreamOptions;
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
    OperationState::Tools(crate::harness::session::types::ToolsOperation {
        scope: operation_scope(),
        batch: crate::harness::session::types::ToolBatch {
            assistant_entry_id: "assistant".to_owned(),
            configuration: lane_configuration(),
            turn_id: "turn".to_owned(),
            calls: vec![
                serde_json::from_value(json!({
                    "sourceIndex": 0,
                    "resultEntryId": invocation_id,
                    "status": "effect_pending",
                    "replay": "safe",
                }))
                .expect("tool call wire"),
            ],
        },
    })
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

    let elements = fixture
        .lane
        .session()
        .read_list(
            &stored_values::pending_assistant_frames("operation", response_entry_id).address,
            None,
            &background_context(),
        )
        .await
        .expect("read frames");
    let values: Vec<serde_json::Value> =
        elements.into_iter().map(|element| element.value).collect();
    assert_eq!(
        serde_json::Value::Array(values),
        serde_json::to_value(&frames).expect("frames wire"),
    );
}

#[tokio::test]
async fn declines_a_queued_frame_after_the_authoritative_projection_leaves_its_phase() {
    let response_entry_id = "response";
    let fixture = create_fixture(assistant_effect_pending(response_entry_id)).await;
    let progress = open_frame_progress(&fixture.lane, &fixture.drive, response_entry_id);
    let (started, release) = gate_next_commit(&fixture.storage);
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
                    trigger_entry_id: response_entry_id.to_owned(),
                },
            });
            patch_live_state(&lane, next_state).await;
        })
    };
    started.await.expect("commit started");
    progress.write(AssistantMessageFrame::TextDelta {
        content_index: 0,
        delta: "late".to_owned(),
    });
    let _ = release.send(());
    moving.await.expect("move join");
    progress.drain().await.expect("drain");

    let elements = fixture
        .lane
        .session()
        .read_list(
            &stored_values::pending_assistant_frames("operation", response_entry_id).address,
            None,
            &background_context(),
        )
        .await
        .expect("read frames");
    assert!(elements.is_empty(), "the queued frame was declined");
    assert_eq!(
        fixture.storage.get_value_calls(),
        0,
        "the decline decided without control reads"
    );
}

#[tokio::test]
async fn declines_a_queued_frame_after_terminal_projection_publication() {
    let response_entry_id = "response";
    let fixture = create_fixture(assistant_effect_pending(response_entry_id)).await;
    let progress = open_frame_progress(&fixture.lane, &fixture.drive, response_entry_id);
    let (started, release) = gate_next_commit(&fixture.storage);
    let ending = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move {
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
        })
    };
    started.await.expect("commit started");
    progress.write(AssistantMessageFrame::TextDelta {
        content_index: 0,
        delta: "late".to_owned(),
    });
    let _ = release.send(());
    ending.await.expect("ending join");
    progress.drain().await.expect("drain");

    let elements = fixture
        .lane
        .session()
        .read_list(
            &stored_values::pending_assistant_frames("operation", response_entry_id).address,
            None,
            &background_context(),
        )
        .await
        .expect("read frames");
    assert!(elements.is_empty(), "the queued frame was declined");
    assert_eq!(
        fixture.storage.get_value_calls(),
        0,
        "the decline decided without control reads"
    );
}

#[tokio::test]
async fn replaces_tool_checkpoints_in_invocation_order() {
    let invocation_id = "result";
    let fixture = create_fixture(tools_state(invocation_id)).await;
    let progress = open_tool_progress(&fixture.lane, &fixture.drive, "turn", 0, invocation_id);
    let checkpoint = |text: &str| AgentToolResult {
        content: vec![AgentToolContent::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        details: json!({}),
        usage: None,
        added_tool_names: None,
        terminate: None,
    };
    progress.write(checkpoint("first"));
    progress.write(checkpoint("second"));
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

/// The deferred-effect-pending leaf the deferred-phase fixture drives.
fn deferred_effect_pending(response_entry_id: &str) -> OperationState {
    OperationState::DeferredEffectPending(DeferredEffectPendingOperation {
        scope: DeferredScope {
            scope: operation_scope(),
            step_id: "step".to_owned(),
            source_entry_id: "source".to_owned(),
            poll: 0,
            configuration: lane_configuration(),
            stream_options: AgentHarnessStreamOptions::default(),
        },
        response_entry_id: response_entry_id.to_owned(),
        usage_id: "usage".to_owned(),
    })
}

#[tokio::test]
async fn writes_frames_through_a_deferred_effect_pending_phase() {
    let response_entry_id = "response";
    let fixture = create_fixture(deferred_effect_pending(response_entry_id)).await;
    let progress = open_frame_progress(&fixture.lane, &fixture.drive, response_entry_id);
    progress.write(frame_delta(0, "deferred"));
    progress.drain().await.expect("drain");

    let elements = fixture
        .lane
        .session()
        .read_list(
            &stored_values::pending_assistant_frames("operation", response_entry_id).address,
            None,
            &background_context(),
        )
        .await
        .expect("read frames");
    assert_eq!(elements.len(), 1, "the deferred phase still owns the write");
}

#[tokio::test]
async fn declines_tool_checkpoints_outside_the_tools_phase() {
    let response_entry_id = "response";
    let fixture = create_fixture(assistant_effect_pending(response_entry_id)).await;
    let progress = open_tool_progress(&fixture.lane, &fixture.drive, "turn", 0, "result");
    progress.write(checkpoint_of("late"));
    progress.drain().await.expect("drain");

    let stored = fixture
        .lane
        .session()
        .get_value(
            &stored_values::pending_tool_output("operation", "result").address,
            &background_context(),
        )
        .await
        .expect("read checkpoint");
    assert!(stored.is_none(), "the non-tools phase declined the write");
}

#[tokio::test]
async fn declines_tool_checkpoints_after_the_operation_clears() {
    let fixture = create_fixture(tools_state("result")).await;
    clear_operation(&fixture).await;
    let progress = open_tool_progress(&fixture.lane, &fixture.drive, "turn", 0, "result");
    progress.write(checkpoint_of("late"));
    progress.drain().await.expect("drain");

    let stored = fixture
        .lane
        .session()
        .get_value(
            &stored_values::pending_tool_output("operation", "result").address,
            &background_context(),
        )
        .await
        .expect("read checkpoint");
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

/// Commits the terminal projection the cleared-operation fixture drives,
/// the terminal test's write list restated.
async fn clear_operation(fixture: &ProgressFixture) {
    let lane = Arc::clone(&fixture.lane);
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

//! The terminal cleanup mechanics and operation-result-record suite,
//! ported 1:1 from upstream `test/harness/runtime/drive-terminal.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements the port makes and the tests bind:
//! - upstream's `vi.spyOn(Date, "now").mockReturnValue(20)` reads the epoch
//!   clock through the drive tree's [`now_ms`] seam, so the record
//!   assertion bounds `ended_at` between the readings taken around the
//!   call and pins every other field exactly.
//! - the per-file `runScope`/`commit`/`createSession` fixtures ride the
//!   shared drive fixture module (`run_scope`, `commit_writes`); the
//!   memory backend stays anonymous — upstream's `createSession` returns
//!   it alongside the session, which no case reads.
//! - the write-address rendering restates upstream's local `address`
//!   helper over the erased write families: entries and usage rows by
//!   their id, the remaining families by their wire fields.
//! - upstream's `afterEach` close restates as a close at each test's end.
//! - the two record tests are synchronous (upstream's `it`s await
//!   nothing), so they restate as plain `#[test]`s.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;

use pi_ai::types::{Message, TextContent, ToolResultMessage};
use pi_ai::utils::assistant_message_frame::AssistantMessageFrame;

use super::common;
use crate::harness::compaction::types::{CompactionSettings, create_file_ops};
use crate::harness::context::background_context;
use crate::harness::runtime::clock::now_ms;
use crate::harness::runtime::drive::terminal::{operation_cleanup_writes, operation_result_record};
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::generation_context;
use crate::harness::runtime::test_support::lane_configuration;
use crate::harness::runtime::test_support::{
    lane_state_write, raw_write, runtime_session_metadata,
};
use crate::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use crate::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::Control;
use crate::harness::session::types::DurableStructuralPreparation;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::LaneState;
use crate::harness::session::types::NavigationReadyToCommitOperation;
use crate::harness::session::types::OperationError;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationKind;
use crate::harness::session::types::OperationMeta;
use crate::harness::session::types::OperationResultRecord;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::SummaryDecidingOperation;
use crate::harness::session::types::{TerminalStatus, ToolBatch, ToolCall, ToolsOperation};
use crate::harness::session::values as stored_values;
use crate::harness::session::values::{
    ToolOutputPayload, Write, append_list_write, set_value_write,
};
use crate::types::{AgentMessage, AgentToolContent};

/// The compaction settings the terminal leaves carry, upstream's
/// `runScope()` settings literal.
fn terminal_compaction() -> CompactionSettings {
    CompactionSettings {
        enabled: true,
        reserve_tokens: 1_000,
        keep_recent_tokens: 2_000,
    }
}

/// The scope fixture, upstream's `runScope()`.
#[must_use]
fn run_scope() -> OperationScope {
    common::run_scope(terminal_compaction())
}

/// The queued items the fixture seeds and the cleanup preserves, upstream's
/// `laneState("main")` inbox literal.
fn inbox_items() -> Vec<InboxItem> {
    vec![
        InboxItem {
            entry_id: "steer".to_owned(),
            kind: InboxItemKind::Steer,
        },
        InboxItem {
            entry_id: "follow".to_owned(),
            kind: InboxItemKind::FollowUp,
        },
        InboxItem {
            entry_id: "write".to_owned(),
            kind: InboxItemKind::Write,
        },
        InboxItem {
            entry_id: "next".to_owned(),
            kind: InboxItemKind::NextRun,
        },
    ]
}

/// The session fixture, upstream's `createSession`: the storage-backed
/// session over a fresh memory backend; upstream returns the backend
/// alongside, which no case reads, so the restatement returns the session
/// alone.
#[must_use]
fn create_session() -> Arc<StorageBackedSession> {
    Arc::new(StorageBackedSession::new(
        runtime_session_metadata(common::next_suite_session_id("terminal")),
        Arc::new(MemoryStorage::new(MemoryStorageOptions::default())),
        StorageBackedSessionOptions::default(),
    ))
}

/// The operation meta the seeds commit, upstream's `meta(operationId,
/// state)`: the intent derives from the leaf's `at` discriminator —
/// compaction for the summary-deciding leaf, an unsummarized navigation
/// for the navigation-ready leaf, a run otherwise.
#[must_use]
fn meta(operation_id: &str, state: &OperationState) -> OperationMeta {
    let intent = match state {
        OperationState::SummaryDeciding(_) => OperationIntent::Compaction {
            custom_instructions: None,
        },
        OperationState::NavigationReadyToCommit(leaf) => OperationIntent::Navigation {
            target_id: leaf.target_id.clone(),
            summarize: false,
            label: None,
            custom_instructions: None,
        },
        _ => OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
    };
    OperationMeta {
        operation_id: operation_id.to_owned(),
        lane: "main".to_owned(),
        source_tip_id: None,
        started_at: 1,
        intent,
    }
}

/// The write-address rendering, upstream's `address(write)`.
fn address(write: &Write) -> String {
    let kinded =
        |kind: &str, op: &str, namespace: &str, key: &str| format!("{kind}:{op}:{namespace}:{key}");
    match write {
        Write::Entry(entry) => format!("entry:{}", entry.entry.id()),
        Write::Usage(row) => format!("usage:{}", row.row.id),
        Write::ValueSet(value) => kinded(&value.kind, &value.op, &value.namespace, &value.key),
        Write::ValueDelete(value) => kinded(&value.kind, &value.op, &value.namespace, &value.key),
        Write::ListAppend(value) => kinded(&value.kind, &value.op, &value.namespace, &value.key),
        Write::ListDelete(value) => kinded(&value.kind, &value.op, &value.namespace, &value.key),
    }
}

/// Seeds every leftover family the cleanup deletes, upstream's
/// `seedLeftovers`: the operation meta and state, the tool arguments and
/// memo, the branch-summary preparation, and the staged tool output.
async fn seed_leftovers(
    session: &Arc<StorageBackedSession>,
    operation_id: &str,
    state: &OperationState,
) {
    commit_writes(
        session,
        vec![
            set_value_write(
                &stored_values::operation_meta(operation_id),
                meta(operation_id, state),
            )
            .expect("the meta write"),
            set_value_write(&stored_values::operation_state(operation_id), state.clone())
                .expect("the state write"),
            raw_write(
                &stored_values::operation_tool_args(operation_id, "step", 0).address,
                serde_json::json!({ "value": true }),
            ),
            set_value_write(
                &stored_values::operation_tool_memo(operation_id, "invocation", "memo"),
                serde_json::json!({ "value": true }),
            )
            .expect("the tool memo write"),
            set_value_write(
                &stored_values::operation_preparation(operation_id, "task"),
                DurableStructuralPreparation::BranchSummary {
                    messages: Vec::new(),
                    file_ops: create_file_ops(),
                    total_tokens: 0,
                },
            )
            .expect("the preparation write"),
            set_value_write(
                &stored_values::pending_tool_output(operation_id, "invocation"),
                ToolOutputPayload {
                    content: vec![AgentToolContent::Text(TextContent {
                        text: "partial".to_owned(),
                        text_signature: None,
                    })],
                    details: serde_json::json!({}),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                },
            )
            .expect("the tool output write"),
        ],
    )
    .await;
}

/// The batch-call literal, upstream's inline
/// `{ status, sourceIndex, resultEntryId, terminate }` objects; the phase
/// field is private to the session types module, so the literal restates
/// through the wire shape both sides already serialize.
fn wire_call(source_index: u64, result_entry_id: &str, status: &str, terminate: bool) -> ToolCall {
    let wire = serde_json::json!({
        "sourceIndex": source_index,
        "resultEntryId": result_entry_id,
        "status": status,
        "terminate": terminate,
    });
    serde_json::from_value(wire)
        .unwrap_or_else(|error| unreachable!("the tool call round-trips: {error}"))
}

/// Upstream `it`: deletes every operation-owned family and exact live
/// frame list but preserves the lane inbox.
#[tokio::test]
async fn deletes_every_operation_owned_family_and_exact_live_frame_list_but_preserves_the_lane_inbox()
 {
    let session = create_session();
    let operation_id = "run";
    let response_entry_id = "response";
    let state = OperationState::AssistantEffectPending(AssistantEffectPendingOperation {
        scope: common::scope_with_control(
            Control::CancelRequested { requested_at: 2 },
            terminal_compaction(),
        ),
        generation_context: generation_context(),
        attempt: 1,
        response_entry_id: response_entry_id.to_owned(),
        usage_id: "usage".to_owned(),
        intended_output_limit: 100,
        context_window: 1_000,
    });
    seed_leftovers(&session, operation_id, &state).await;
    let mut writes: Vec<Write> = ["steer", "follow", "write", "next"]
        .iter()
        .map(|id| {
            set_value_write(
                &stored_values::pending_entry(id),
                PendingEntry::Custom {
                    custom_type: "test".to_owned(),
                    payload: Some(serde_json::json!({ "id": id })),
                },
            )
            .expect("the pending entry write")
        })
        .collect();
    writes.push(
        append_list_write(
            &stored_values::pending_assistant_frames(operation_id, response_entry_id),
            AssistantMessageFrame::TextDelta {
                content_index: 0,
                delta: "partial".to_owned(),
            },
        )
        .expect("the frame write"),
    );
    writes.push(
        lane_state_write("main", Some(operation_id), None, inbox_items())
            .expect("the lane state write"),
    );
    commit_writes(&session, writes).await;

    let writes = operation_cleanup_writes(
        session.as_ref(),
        operation_id,
        &state,
        &background_context(),
    )
    .await
    .expect("the cleanup writes");

    assert_eq!(
        writes.iter().map(address).collect::<Vec<_>>(),
        vec![
            "value:delete:pi.op.meta:run".to_owned(),
            "value:delete:pi.op.state:run".to_owned(),
            "value:delete:pi.op.tool_args:run:step:0".to_owned(),
            "value:delete:pi.op.tool_memo:run:invocation:memo".to_owned(),
            "value:delete:pi.op.preparation:run:task".to_owned(),
            "value:delete:pi.pending.tool_output:run:invocation".to_owned(),
            "list:delete:pi.pending.assistant_frame:run:response".to_owned(),
        ],
        "the cleanup deletes the operation-owned families in order",
    );
    commit_writes(&session, writes).await;
    for id in ["steer", "follow", "write", "next"] {
        let entry = session
            .get_value(
                &stored_values::pending_entry(id).address,
                &background_context(),
            )
            .await
            .expect("the pending entry read");
        assert!(entry.is_some(), "the {id} pending entry survives");
    }
    let lane_record = session
        .get_value(
            &stored_values::lane_state("main").address,
            &background_context(),
        )
        .await
        .expect("the lane state read")
        .expect("the lane record");
    let lane_state: LaneState =
        serde_json::from_value(lane_record.value).expect("the lane record parses");
    assert_eq!(lane_state.inbox, inbox_items(), "the lane inbox survives");
    let frames = session
        .read_list(
            &stored_values::pending_assistant_frames(operation_id, response_entry_id).address,
            None,
            &background_context(),
        )
        .await
        .expect("the frame read");
    assert!(frames.is_empty(), "the frame list reads empty");

    Session::close(session.as_ref(), &background_context())
        .await
        .expect("close");
}

/// Upstream `it`: deletes staged tool outcomes and leaves completed
/// results alone.
#[tokio::test]
async fn deletes_staged_tool_outcomes_and_leaves_completed_results_alone() {
    let session = create_session();
    let operation_id = "tools";
    let state = OperationState::Tools(ToolsOperation {
        scope: run_scope(),
        batch: ToolBatch {
            assistant_entry_id: "assistant".to_owned(),
            configuration: lane_configuration(),
            turn_id: "step".to_owned(),
            calls: vec![
                wire_call(0, "staged", "outcome_ready", false),
                wire_call(1, "placed", "completed", false),
            ],
        },
    });
    seed_leftovers(&session, operation_id, &state).await;
    commit_writes(
        &session,
        vec![
            set_value_write(
                &stored_values::pending_entry("staged"),
                PendingEntry::Message {
                    payload: Box::new(AgentMessage::Standard(Message::ToolResult(
                        ToolResultMessage {
                            tool_call_id: "call".to_owned(),
                            tool_name: "tool".to_owned(),
                            content: Vec::new(),
                            details: None,
                            usage: None,
                            added_tool_names: None,
                            is_error: false,
                            timestamp: 1,
                        },
                    ))),
                },
            )
            .expect("the staged entry write"),
        ],
    )
    .await;

    let writes = operation_cleanup_writes(
        session.as_ref(),
        operation_id,
        &state,
        &background_context(),
    )
    .await
    .expect("the cleanup writes");
    let addresses: Vec<String> = writes.iter().map(address).collect();
    assert!(
        addresses.contains(&"value:delete:pi.pending.entry:staged".to_owned()),
        "the staged outcome's pending entry deletes: {addresses:?}"
    );
    assert!(
        !addresses.contains(&"value:delete:pi.pending.entry:placed".to_owned()),
        "the completed result's entry stays: {addresses:?}"
    );

    Session::close(session.as_ref(), &background_context())
        .await
        .expect("close");
}

/// Upstream `it.each` over the `compaction` (`summary.deciding` with a
/// manual finish task) and `navigation` (`navigation.ready_to_commit`,
/// `targetId: null`) rows: defensively deletes leftover %s operation
/// families.
#[tokio::test]
async fn defensively_deletes_leftover_operation_families() {
    for (operation_id, state) in [
        (
            "compaction",
            OperationState::SummaryDeciding(SummaryDecidingOperation {
                scope: run_scope(),
                task: common::standalone_compaction_task(None),
            }),
        ),
        (
            "navigation",
            OperationState::NavigationReadyToCommit(NavigationReadyToCommitOperation {
                scope: run_scope(),
                target_id: None,
                label: None,
            }),
        ),
    ] {
        let session = create_session();
        seed_leftovers(&session, operation_id, &state).await;

        let writes = operation_cleanup_writes(
            session.as_ref(),
            operation_id,
            &state,
            &background_context(),
        )
        .await
        .expect("the cleanup writes");

        assert_eq!(
            writes.iter().map(address).collect::<Vec<_>>(),
            vec![
                format!("value:delete:pi.op.meta:{operation_id}"),
                format!("value:delete:pi.op.state:{operation_id}"),
                format!("value:delete:pi.op.tool_args:{operation_id}:step:0"),
                format!("value:delete:pi.op.tool_memo:{operation_id}:invocation:memo"),
                format!("value:delete:pi.op.preparation:{operation_id}:task"),
                format!("value:delete:pi.pending.tool_output:{operation_id}:invocation"),
            ],
            "the {operation_id} leaf's defensive deletes cover its families"
        );
        Session::close(session.as_ref(), &background_context())
            .await
            .expect("close");
    }
}

/// Upstream `it`: constructs one flat immutable observation from terminal
/// metadata. Upstream pins `Date.now` at 20; the record builder reads the
/// epoch clock through [`now_ms`], so the assertion bounds `ended_at`
/// between the readings taken around the call.
#[test]
fn constructs_one_flat_immutable_observation_from_terminal_metadata() {
    let before = now_ms();
    let record = operation_result_record(
        &OperationMeta {
            operation_id: "run".to_owned(),
            lane: "main".to_owned(),
            source_tip_id: Some("source".to_owned()),
            started_at: 10,
            intent: OperationIntent::Run {
                prompt_entry_ids: vec!["prompt".to_owned()],
            },
        },
        TerminalStatus::Failed,
        Some("tip".to_owned()),
        Some(OperationError {
            code: "provider".to_owned(),
            message: "failed".to_owned(),
            details: None,
        }),
    )
    .expect("the record");
    let after = now_ms();

    assert!(
        record.ended_at >= before,
        "the record reads the clock at or after the call's start: {}",
        record.ended_at
    );
    assert!(
        record.ended_at <= after,
        "the record reads the clock at or before the call's end: {}",
        record.ended_at
    );
    assert_eq!(
        record,
        OperationResultRecord {
            operation_id: "run".to_owned(),
            kind: OperationKind::Run,
            status: TerminalStatus::Failed,
            error: Some(OperationError {
                code: "provider".to_owned(),
                message: "failed".to_owned(),
                details: None,
            }),
            from_tip_id: Some("source".to_owned()),
            tip_id: Some("tip".to_owned()),
            started_at: 10,
            ended_at: record.ended_at,
        },
        "the record carries the metadata verbatim"
    );
}

/// Upstream `it`: rejects errors on non-failed records and missing errors
/// on failed records.
#[test]
fn rejects_errors_on_non_failed_records_and_missing_errors_on_failed_records() {
    let metadata = OperationMeta {
        operation_id: "run".to_owned(),
        lane: "main".to_owned(),
        source_tip_id: None,
        started_at: 1,
        intent: OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
    };
    let with_error = operation_result_record(
        &metadata,
        TerminalStatus::Completed,
        Some("tip".to_owned()),
        Some(OperationError {
            code: "x".to_owned(),
            message: "x".to_owned(),
            details: None,
        }),
    )
    .expect_err("the non-failed record rejects an error");
    assert!(
        with_error
            .to_string()
            .contains("Only a failed operation result may carry an error"),
        "the non-failed record's rejection names the invariant: {with_error}"
    );
    let without_error = operation_result_record(
        &metadata,
        TerminalStatus::Failed,
        Some("tip".to_owned()),
        None,
    )
    .expect_err("the failed record rejects a missing error");
    assert!(
        without_error
            .to_string()
            .contains("Only a failed operation result may carry an error"),
        "the failed record's rejection names the invariant: {without_error}"
    );
}

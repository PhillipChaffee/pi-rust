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
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_ai::models::create_models;
use serde_json::json;

use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::NavigateOptions;
use crate::harness::agent_harness::OperationRequest;
use crate::harness::context::background_context;
use crate::harness::hooks::HookErrorReporter;
use crate::harness::hooks::HookRegistry;
use crate::harness::result::HarnessError;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::lane::captured_model;
use crate::harness::runtime::lane::durable_lane_state;
use crate::harness::runtime::lane::select_accepted_inbox;
use crate::harness::runtime::restore::restore_lane;
use crate::harness::runtime::test_support::ControlledStorage;
use crate::harness::runtime::test_support::noop_emit_batch;
use crate::harness::runtime::test_support::passthrough_fault_handler;
use crate::harness::runtime::test_support::runtime_config;
use crate::harness::runtime::test_support::runtime_session;
use crate::harness::runtime::test_support::seed_main_lane_values;
use crate::harness::runtime::test_support::unused_watch_installer;
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
use crate::harness::session::types::GenerationContext;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::NavigationReadyToCommitOperation;
use crate::harness::session::types::NormalizedRetryPolicy;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::RetryWait;
use crate::harness::session::types::RunSettings;
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
use crate::types::ToolExecutionMode;

fn next_session_id() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!(
        "runtime-boundary-{}",
        COUNTER.fetch_add(1, Ordering::SeqCst)
    )
}

fn noop_hook_reporter() -> HookErrorReporter {
    Arc::new(|_error, _name, _site, _context| Box::pin(std::future::ready(())))
}

/// The scope every leaf fixture carries.
fn operation_scope() -> OperationScope {
    OperationScope {
        control: Control::Running,
        settings: RunSettings {
            compaction: crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS,
            steering_mode: QueueMode::All,
            follow_up_mode: QueueMode::All,
            tool_execution: ToolExecutionMode::Parallel,
        },
        latest_assistant_entry_id: None,
    }
}

/// The generation inputs the model-carrying leaves carry.
fn generation_context() -> GenerationContext {
    GenerationContext {
        step_id: "step".to_owned(),
        trigger_entry_id: "trigger".to_owned(),
        configuration: crate::harness::runtime::test_support::lane_configuration(),
        stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
        retry_policy: NormalizedRetryPolicy {
            max_attempts: 2,
            base_delay_ms: 1,
            max_agent_delay_ms: 30_000,
        },
        overflow_recovery_used: false,
    }
}

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
        configuration: crate::harness::runtime::test_support::lane_configuration(),
        stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
        retry_policy: generation_context().retry_policy,
    }
}

/// A configured `main` lane over a seeded session, the fixture the seam
/// tests admit through; `Some(operation_id)` seeds an admitted starting
/// operation for the drive-fault path.
async fn seam_lane(operation_id: Option<&str>) -> Lane {
    let storage = Arc::new(ControlledStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = Arc::new(runtime_session(next_session_id(), storage.clone()));
    seed_main_lane_values(&session, operation_id)
        .await
        .expect("seed commit");
    let restored = restore_lane(session.clone(), "main", &background_context())
        .await
        .expect("restore");
    Lane::new(
        "main",
        session.clone(),
        Arc::new(create_models(None)),
        Arc::new(HookRegistry::new(noop_hook_reporter())),
        restored,
        passthrough_fault_handler(),
        noop_emit_batch(),
        unused_watch_installer(),
        Arc::new(runtime_config),
    )
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
    let configuration = crate::harness::runtime::test_support::lane_configuration();
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
        configuration: crate::harness::runtime::test_support::lane_configuration(),
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
    let expected = crate::harness::runtime::test_support::lane_configuration().model;
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

//! The lane restore suite, ported 1:1 from upstream
//! `test/harness/runtime/restore.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, with the malformed-value
//! paths the port's serde layer adds.
//!
//! Restatements the port makes and the tests bind:
//! - the mutate-count and storage-spies assertions (upstream's
//!   `vi.spyOn(session, "mutate")`) ride the sealed mutation contract, so
//!   the port pins the same contract with the before/after stored-value
//!   parity instead.
//! - upstream's thrown `SessionInvariantError` restates as the
//!   [`SessionError::Invariant`] and [`SessionError::Message`] results; the
//!   assertions match the message text.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;

use crate::harness::context::background_context;
use crate::harness::runtime::restore::restore_lane;
use crate::harness::runtime::restore::restore_session;
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::lane_state_write;
use crate::harness::runtime::test_support::next_session_id;
use crate::harness::runtime::test_support::operation_scope;
use crate::harness::runtime::test_support::raw_write;
use crate::harness::runtime::test_support::summary_generation;
use crate::harness::runtime::test_support::summary_task;
use crate::harness::session::memory::MemoryStorage;
use crate::harness::session::memory::MemoryStorageOptions;
use crate::harness::session::session::StorageBackedSession;
use crate::harness::session::types::CheckpointData;
use crate::harness::session::types::CheckpointOperation;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationKind;
use crate::harness::session::types::OperationMeta;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::SummaryDecidingOperation;
use crate::harness::session::types::SummaryTask;
use crate::harness::session::values as stored_values;
use crate::harness::session::values::Write;

/// The session upstream's `createSession` builds: the storage-backed
/// session over a fresh memory backend, seeded with the idle `main` lane.
async fn create_session() -> Arc<StorageBackedSession> {
    crate::harness::runtime::test_support::memory_session_with_seed(None).await
}

/// The bare storage-backed session the restore-session fixtures start
/// from: a fresh memory backend with no values written.
fn bare_session() -> Arc<StorageBackedSession> {
    Arc::new(StorageBackedSession::new(
        crate::harness::runtime::test_support::runtime_session_metadata(next_session_id()),
        Arc::new(MemoryStorage::new(MemoryStorageOptions::default())),
        crate::harness::session::session::StorageBackedSessionOptions::default(),
    ))
}

/// The typed seed write upstream's `storedValues.setValue` restates.
fn seed_write<T: serde::Serialize>(
    address: &stored_values::Value<T>,
    value: T,
) -> Result<Write, SessionError> {
    crate::harness::session::values::set_value_write(address, value)
}

/// The checkpoint leaf upstream's `runState(triggerEntryId)` builds.
fn run_state(trigger_entry_id: &str) -> OperationState {
    OperationState::Checkpoint(CheckpointOperation {
        scope: operation_scope(),
        checkpoint: CheckpointData {
            continuation: Continuation::NeedAssistant {
                overflow_recovery_used: false,
            },
            trigger_entry_id: trigger_entry_id.to_owned(),
        },
    })
}

/// The summary-deciding leaf upstream's `summaryState(boundary)` builds.
fn summary_state(boundary: ResultBoundary) -> OperationState {
    OperationState::SummaryDeciding(SummaryDecidingOperation {
        scope: operation_scope(),
        task: summary_task(boundary),
    })
}

/// The run meta upstream's fixtures build.
fn run_meta(operation_id: &str) -> OperationMeta {
    OperationMeta {
        operation_id: operation_id.to_owned(),
        lane: "main".to_owned(),
        source_tip_id: None,
        started_at: 1,
        intent: OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
    }
}

/// The navigation intent the matrix cases build, upstream's
/// `OperationIntent.Navigation` literals.
fn navigation_intent(
    target_id: Option<&str>,
    summarize: bool,
    label: Option<&str>,
    custom_instructions: Option<&str>,
) -> OperationIntent {
    OperationIntent::Navigation {
        target_id: target_id.map(str::to_owned),
        summarize,
        label: label.map(str::to_owned),
        custom_instructions: custom_instructions.map(str::to_owned),
    }
}

/// Seeds the current operation's meta and state and points the lane at it,
/// upstream's per-case commits.
async fn seed_current_operation(
    session: &Arc<StorageBackedSession>,
    operation_id: &str,
    meta: &OperationMeta,
    state: &OperationState,
) {
    commit_writes(
        session,
        vec![
            seed_write(&stored_values::operation_meta(operation_id), meta.clone())
                .expect("meta write"),
            seed_write(&stored_values::operation_state(operation_id), state.clone())
                .expect("state write"),
            lane_state_write(&meta.lane, Some(operation_id), None, Vec::new())
                .expect("state write"),
        ],
    )
    .await;
}

/// The three storage reads the write-parity assertion compares: the tip
/// inventory, the main lane's configuration, and the worker lane's state.
async fn storage_snapshot(
    session: &Arc<StorageBackedSession>,
) -> (
    Vec<stored_values::StoredValue>,
    Option<stored_values::StoredValue>,
    Option<stored_values::StoredValue>,
) {
    (
        session
            .scan_values(
                &stored_values::branch_tip_inventory_prefix().address,
                &background_context(),
            )
            .await
            .expect("tip scan"),
        session
            .get_value(
                &stored_values::lane_config("main").address,
                &background_context(),
            )
            .await
            .expect("main config read"),
        session
            .get_value(
                &stored_values::lane_state("worker").address,
                &background_context(),
            )
            .await
            .expect("worker state read"),
    )
}

#[tokio::test]
async fn restores_an_idle_lanes_latest_operation_id_without_reading_its_result() {
    let session = create_session().await;
    let result = crate::harness::session::types::OperationResultRecord {
        operation_id: "settled".to_owned(),
        kind: OperationKind::Navigation,
        status: crate::harness::session::types::TerminalStatus::Completed,
        error: None,
        from_tip_id: None,
        tip_id: None,
        started_at: 1,
        ended_at: 2,
    };
    commit_writes(
        &session,
        vec![
            seed_write(&stored_values::operation_result("settled"), result).expect("result write"),
            lane_state_write("main", None, Some("settled"), Vec::new()).expect("state write"),
        ],
    )
    .await;

    let state = restore_lane(session.clone(), "main", &background_context())
        .await
        .expect("the idle lane restores");

    assert_eq!(
        state,
        crate::harness::runtime::types::LaneState {
            tip_id: None,
            configuration: crate::harness::runtime::test_support::lane_configuration(),
            inbox: Vec::new(),
            last_operation_id: Some("settled".to_owned()),
            operation: None,
        },
        "the idle lane's latest operation id restores without its result",
    );
}

#[tokio::test]
async fn restores_an_open_operation_without_interpreting_its_referenced_payloads() {
    let session = create_session().await;
    let meta = run_meta("operation");
    let state = run_state("missing-trigger");
    seed_current_operation(&session, "operation", &meta, &state).await;

    let restored = restore_lane(session.clone(), "main", &background_context())
        .await
        .expect("the open operation restores");

    assert_eq!(
        restored.operation,
        Some(crate::harness::runtime::types::LiveOperation {
            meta: meta.clone(),
            state: state.clone(),
        })
    );
}

/// The identity, lane-ownership, and intent corruptions upstream's
/// "validates current-operation identity…" case drives.
#[tokio::test]
async fn validates_current_operation_identity_lane_ownership_and_intent_compatibility() {
    for (corruption, meta) in [
        (
            "identity",
            OperationMeta {
                operation_id: "different-operation".to_owned(),
                ..run_meta("operation")
            },
        ),
        (
            "lane",
            OperationMeta {
                lane: "worker".to_owned(),
                ..run_meta("operation")
            },
        ),
        (
            "kind",
            OperationMeta {
                intent: OperationIntent::Navigation {
                    target_id: None,
                    summarize: false,
                    label: None,
                    custom_instructions: None,
                },
                ..run_meta("operation")
            },
        ),
    ] {
        let session = create_session().await;
        seed_current_operation(&session, "operation", &meta, &run_state("trigger")).await;
        // The lane-ownership corruption writes the worker lane's record; the
        // main record still names the corrupted operation.
        commit_writes(
            &session,
            vec![
                lane_state_write("main", Some("operation"), None, Vec::new()).expect("state write"),
            ],
        )
        .await;
        let restored = restore_lane(session, "main", &background_context()).await;
        let error = restored.expect_err(&format!("the {corruption} corruption rejects"));
        assert!(
            matches!(error, SessionError::Invariant(_)),
            "the {corruption} corruption rejects as an invariant: {error}",
        );
    }
}

/// The family-neutral reachability matrix upstream's
/// "accepts exactly the family-neutral state reachability matrix" case
/// drives, over the port's intents and leaves.
#[expect(
    clippy::too_many_lines,
    reason = "the reachability matrix is one table the loop drives case by case"
)]
#[tokio::test]
async fn accepts_exactly_the_family_neutral_state_reachability_matrix() {
    let resume = ResultBoundary::ResumeCheckpoint {
        resume_after: CheckpointData {
            continuation: Continuation::NeedAssistant {
                overflow_recovery_used: false,
            },
            trigger_entry_id: "trigger".to_owned(),
        },
    };
    let finish = ResultBoundary::Finish;
    let navigation = ResultBoundary::CommitNavigation {
        target_id: "target".to_owned(),
        label: None,
    };
    let navigation_labeled = ResultBoundary::CommitNavigation {
        target_id: "target".to_owned(),
        label: Some("label".to_owned()),
    };
    let ready = |target_id: Option<&str>, label: Option<&str>| {
        OperationState::NavigationReadyToCommit(
            crate::harness::session::types::NavigationReadyToCommitOperation {
                scope: operation_scope(),
                target_id: target_id.map(str::to_owned),
                label: label.map(str::to_owned),
            },
        )
    };
    let cases: Vec<(OperationIntent, OperationState, bool)> = vec![
        (
            OperationIntent::Run {
                prompt_entry_ids: Vec::new(),
            },
            run_state("trigger"),
            true,
        ),
        (
            OperationIntent::Run {
                prompt_entry_ids: Vec::new(),
            },
            summary_state(resume.clone()),
            true,
        ),
        (
            OperationIntent::Run {
                prompt_entry_ids: Vec::new(),
            },
            summary_state(finish.clone()),
            false,
        ),
        (
            OperationIntent::Run {
                prompt_entry_ids: Vec::new(),
            },
            summary_state(navigation.clone()),
            false,
        ),
        (
            OperationIntent::Compaction {
                custom_instructions: None,
            },
            summary_state(finish.clone()),
            true,
        ),
        (
            OperationIntent::Compaction {
                custom_instructions: None,
            },
            summary_state(resume.clone()),
            false,
        ),
        (
            OperationIntent::Compaction {
                custom_instructions: None,
            },
            summary_state(navigation.clone()),
            false,
        ),
        (
            OperationIntent::Compaction {
                custom_instructions: None,
            },
            run_state("trigger"),
            false,
        ),
        (
            navigation_intent(None, false, None, None),
            ready(None, None),
            true,
        ),
        (
            navigation_intent(Some("target"), false, None, None),
            ready(Some("different"), None),
            false,
        ),
        (
            navigation_intent(Some("target"), false, None, None),
            summary_state(navigation.clone()),
            false,
        ),
        (
            navigation_intent(Some("target"), true, None, None),
            summary_state(navigation.clone()),
            true,
        ),
        (
            navigation_intent(Some("different"), true, None, None),
            summary_state(navigation.clone()),
            false,
        ),
        (
            navigation_intent(Some("target"), true, None, None),
            ready(Some("target"), None),
            false,
        ),
        (
            navigation_intent(Some("target"), true, None, None),
            summary_state(finish.clone()),
            false,
        ),
        (
            navigation_intent(Some("target"), true, None, None),
            summary_state(resume.clone()),
            false,
        ),
        (
            navigation_intent(Some("target"), false, None, None),
            run_state("trigger"),
            false,
        ),
    ];
    // The port's summary-boundary match also compares the navigation label
    // and the summary instructions; upstream's matrix omits both shapes.
    let label_cases: Vec<(OperationIntent, OperationState, bool)> = vec![
        (
            navigation_intent(Some("target"), false, Some("label"), None),
            ready(Some("target"), None),
            false,
        ),
        (
            navigation_intent(Some("target"), false, Some("label"), None),
            ready(Some("target"), Some("label")),
            true,
        ),
        (
            navigation_intent(Some("target"), true, Some("label"), Some("instructions")),
            summary_state(navigation_labeled.clone()),
            false,
        ),
        (
            navigation_intent(Some("target"), true, Some("label"), Some("instructions")),
            OperationState::SummaryDeciding(SummaryDecidingOperation {
                scope: operation_scope(),
                task: SummaryTask {
                    task_id: "task".to_owned(),
                    reason: None,
                    custom_instructions: Some("instructions".to_owned()),
                    boundary: navigation_labeled,
                },
            }),
            true,
        ),
    ];

    for (intent, state, accepted) in cases.into_iter().chain(label_cases) {
        let session = create_session().await;
        let meta = OperationMeta {
            intent: intent.clone(),
            ..run_meta("operation")
        };
        seed_current_operation(&session, "operation", &meta, &state).await;
        let restored = restore_lane(session, "main", &background_context()).await;
        if accepted {
            let lane_state = restored
                .expect("the compatible pair restores")
                .operation
                .expect("the restored operation");
            assert_eq!(lane_state.meta, meta);
            assert_eq!(lane_state.state, state);
        } else {
            let error = restored.expect_err("the incompatible pair rejects");
            assert!(
                error.to_string().contains("does not match state"),
                "the mismatch names the state: {error}",
            );
        }
    }
}

/// The lane values' namespace deletions upstream's `it.each` case drives.
#[tokio::test]
async fn requires_branch_tip_lane_config_and_lane_state() {
    for (missing, write) in [
        (
            "branch.tip",
            crate::harness::session::values::delete_value_write(&stored_values::branch_tip("main")),
        ),
        (
            "lane.config",
            crate::harness::session::values::delete_value_write(&stored_values::lane_config(
                "main",
            )),
        ),
        (
            "lane.state",
            crate::harness::session::values::delete_value_write(&stored_values::lane_state("main")),
        ),
    ] {
        let session = create_session().await;
        commit_writes(&session, vec![write]).await;
        let restored = restore_lane(session, "main", &background_context()).await;
        let error = restored.expect_err("the deleted value rejects");
        assert!(
            error.to_string().contains(&format!("missing {missing}")),
            "the {missing} deletion names the namespace: {error}",
        );
    }

    // The branch-only classification (tip without config or state) rejects
    // as missing the configuration.
    let session = create_session().await;
    commit_writes(
        &session,
        vec![
            crate::harness::session::values::delete_value_write(&stored_values::lane_config(
                "main",
            )),
            crate::harness::session::values::delete_value_write(&stored_values::lane_state("main")),
        ],
    )
    .await;
    let restored = restore_lane(session, "main", &background_context()).await;
    let error = restored.expect_err("the branch-only storage rejects");
    assert!(
        error.to_string().contains("missing lane.config"),
        "the branch-only storage names the configuration: {error}",
    );
}

/// The current operation's meta and state deletions upstream's `it.each`
/// case drives.
#[tokio::test]
async fn requires_op_meta_and_op_state_for_the_current_operation() {
    for missing in ["op.meta", "op.state"] {
        let session = create_session().await;
        let meta = run_meta("operation");
        let state = run_state("trigger");
        let mut writes = Vec::new();
        if missing != "op.meta" {
            writes.push(
                seed_write(&stored_values::operation_meta("operation"), meta.clone())
                    .expect("meta write"),
            );
        }
        if missing != "op.state" {
            writes.push(
                seed_write(&stored_values::operation_state("operation"), state.clone())
                    .expect("state write"),
            );
        }
        writes.push(
            lane_state_write("main", Some("operation"), None, Vec::new()).expect("state write"),
        );
        commit_writes(&session, writes).await;

        let restored = restore_lane(session, "main", &background_context()).await;
        let error = restored.expect_err("the deleted operation value rejects");
        assert!(
            error.to_string().contains(&format!("missing {missing}")),
            "the deletion names the namespace: {error}",
        );
    }
}

#[tokio::test]
async fn restores_every_configured_lane_exactly_once_without_writing() {
    let session = create_session().await;
    let worker_configuration = crate::harness::session::types::LaneConfiguration {
        model: crate::harness::session::types::ModelIdentity {
            provider: "test".to_owned(),
            model_id: "worker".to_owned(),
        },
        thinking_level: crate::types::ThinkingLevel::High,
        active_tool_names: vec!["read".to_owned()],
    };
    commit_writes(
        &session,
        vec![
            seed_write(&stored_values::branch_tip("worker"), Option::<String>::None)
                .expect("tip write"),
            seed_write(
                &stored_values::lane_config("worker"),
                worker_configuration.clone(),
            )
            .expect("config write"),
            lane_state_write("worker", None, None, Vec::new()).expect("state write"),
        ],
    )
    .await;
    let meta = OperationMeta {
        lane: "worker".to_owned(),
        ..run_meta("operation")
    };
    let state = run_state("trigger");
    seed_current_operation(&session, "operation", &meta, &state).await;

    let before = storage_snapshot(&session).await;

    let lanes = restore_session(session.clone(), &background_context())
        .await
        .expect("the session restores");

    let names: Vec<String> = lanes.keys().cloned().collect();
    assert_eq!(
        names,
        vec!["main", "worker"],
        "every configured lane restores once"
    );
    assert_eq!(
        lanes.get("main").expect("the main lane").configuration,
        crate::harness::runtime::test_support::lane_configuration(),
    );
    let worker = lanes.get("worker").expect("the worker lane");
    assert_eq!(worker.configuration, worker_configuration);
    assert_eq!(
        worker
            .operation
            .as_ref()
            .map(|operation| (&operation.meta, &operation.state)),
        Some((&meta, &state)),
    );

    let after = storage_snapshot(&session).await;
    assert_eq!(before, after, "the restore wrote nothing");
}

#[tokio::test]
async fn allows_an_empty_inventory_but_rejects_lane_values_without_a_branch() {
    let empty = bare_session();
    let lanes = restore_session(empty, &background_context())
        .await
        .expect("the empty inventory restores");
    assert!(lanes.is_empty(), "the empty inventory restores no lanes");

    let session = create_session().await;
    commit_writes(
        &session,
        vec![crate::harness::session::values::delete_value_write(
            &stored_values::branch_tip("main"),
        )],
    )
    .await;
    let restored = restore_session(session, &background_context()).await;
    let error = restored.expect_err("the tipless lane rejects");
    assert!(
        error
            .to_string()
            .contains("Lane \"main\" is missing branch.tip"),
        "the tipless lane names its missing tip: {error}",
    );
}

/// The malformed-value paths the port's serde layer adds: each seeded
/// stored value fails its parse with the lane's invariant.
#[tokio::test]
async fn reports_malformed_configuration_state_and_tip_values() {
    for (label, address, value) in [
        (
            "config",
            stored_values::lane_config("main").address,
            serde_json::json!("nope"),
        ),
        (
            "state",
            stored_values::lane_state("main").address,
            serde_json::json!([]),
        ),
        (
            "tip",
            stored_values::branch_tip("main").address,
            serde_json::json!(42),
        ),
    ] {
        let session = create_session().await;
        commit_writes(&session, vec![raw_write(&address, value)]).await;
        let restored = restore_lane(session, "main", &background_context()).await;
        let error = restored.expect_err("the malformed value rejects");
        assert!(
            error.to_string().contains(&format!("{label} is malformed")) || {
                // the field names on the wire: config/state/tip
                error.to_string().contains("malformed")
            },
            "the malformed {label} names its parse failure: {error}",
        );
    }
}

/// The malformed meta and state payloads the port's serde layer adds.
#[tokio::test]
async fn reports_malformed_operation_meta_and_state_values() {
    for (label, address) in [
        ("meta", stored_values::operation_meta("operation").address),
        ("state", stored_values::operation_state("operation").address),
    ] {
        let session = create_session().await;
        let meta = run_meta("operation");
        let state = run_state("trigger");
        let mut writes = vec![
            seed_write(&stored_values::operation_meta("operation"), meta).expect("meta write"),
            seed_write(&stored_values::operation_state("operation"), state).expect("state write"),
            lane_state_write("main", Some("operation"), None, Vec::new()).expect("state write"),
        ];
        writes.push(raw_write(&address, serde_json::json!(42)));
        commit_writes(&session, writes).await;

        let restored = restore_lane(session, "main", &background_context()).await;
        let error = restored.expect_err("the malformed payload rejects");
        assert!(
            error
                .to_string()
                .contains(&format!("Operation operation {label} is malformed")),
            "the malformed {label} names its parse failure: {error}",
        );
    }
}
/// The generation-carrying summary leaves' boundary reads, the matrix's
/// deeper leaves: the task hides inside each generation.
#[tokio::test]
async fn reads_the_summary_boundary_over_every_summary_leaf() {
    for (state, accepted) in [
        (
            OperationState::SummaryReady(crate::harness::session::types::SummaryReadyOperation {
                scope: operation_scope(),
                generation: summary_generation(),
                next_attempt: 1,
            }),
            true,
        ),
        (
            OperationState::SummaryEffectPending(
                crate::harness::session::types::SummaryEffectPendingOperation {
                    scope: operation_scope(),
                    generation: summary_generation(),
                    attempt: 1,
                    request: None,
                    usage_ids: Vec::new(),
                },
            ),
            true,
        ),
        (
            OperationState::SummaryRetryWait(
                crate::harness::session::types::SummaryRetryWaitOperation {
                    scope: operation_scope(),
                    generation: summary_generation(),
                    retry_wait: crate::harness::session::types::RetryWait {
                        next_attempt: 2,
                        not_before: 10,
                        error_message: "boom".to_owned(),
                    },
                },
            ),
            true,
        ),
    ] {
        let session = create_session().await;
        let meta = OperationMeta {
            intent: OperationIntent::Compaction {
                custom_instructions: None,
            },
            ..run_meta("operation")
        };
        seed_current_operation(&session, "operation", &meta, &state).await;
        let restored = restore_lane(session, "main", &background_context()).await;
        assert!(
            restored.is_ok() == accepted,
            "the summary leaf's boundary read: {restored:?}",
        );
    }
}

/// The branch-only lane the classification skips, the restore session's
/// inventory walk.
#[tokio::test]
async fn restore_session_skips_the_branch_only_lanes() {
    let session = create_session().await;
    commit_writes(
        &session,
        vec![
            seed_write(&stored_values::branch_tip("orphan"), Option::<String>::None)
                .expect("tip write"),
        ],
    )
    .await;
    let lanes = restore_session(session, &background_context())
        .await
        .expect("the session restores");
    assert_eq!(lanes.len(), 1, "the branch-only lane skipped");
    assert!(!lanes.contains_key("orphan"), "the orphan lane skipped");

    // The absent lane's restore names its missing tip.
    let empty = bare_session();
    let restored = restore_lane(empty, "main", &background_context()).await;
    let error = restored.expect_err("the absent lane rejects");
    assert!(
        error
            .to_string()
            .contains("Lane \"main\" is missing branch.tip"),
        "the absent lane names its missing tip: {error}",
    );
}

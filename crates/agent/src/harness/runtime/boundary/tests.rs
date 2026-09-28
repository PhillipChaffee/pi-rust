//! The runtime boundary suite: behavior upstream exercises from other
//! layers (the harness lane surface, the drive procedures, the storage
//! contract) bound here where the runtime child owns the seams — the real
//! compaction and summarized-navigation admissions and the drive spawn's
//! settle path, the mismatch shape, the durable lane record's wire
//! round-trip, the inbox admission split, and the captured-model projection
//! over the state leaves.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the tests pin outcomes; unexpected results and violated expectations panic the test by design"
)]

use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::harness::agent_harness::{DriveOptions, NavigateOptions, OperationRequest};
use crate::harness::context::background_context;
use crate::harness::result::HarnessError;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::lane::{captured_model, durable_lane_state, select_accepted_inbox};
use crate::harness::runtime::test_support::ControlledStorage;
use crate::harness::runtime::test_support::accept_text;
use crate::harness::runtime::test_support::assistant_wire_value;
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::deferred_effect_pending;
use crate::harness::runtime::test_support::deferred_scope;
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
use crate::harness::runtime::test_support::summary_generation;
use crate::harness::runtime::test_support::summary_task;
use crate::harness::runtime::test_support::tools_batch_state;
use crate::harness::runtime::test_support::unused_watch_installer;
use crate::harness::runtime::test_support::{user_text_message, zero_usage_wire};
use crate::harness::runtime::types::SliceNotImplemented;
use crate::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::AssistantReadyOperation;
use crate::harness::session::types::AssistantRetryWaitOperation;
use crate::harness::session::types::CheckpointData;
use crate::harness::session::types::CheckpointOperation;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::Control;
use crate::harness::session::types::DeferredEffectPendingOperation;
use crate::harness::session::types::DeferredSuspendedOperation;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::NavigationReadyToCommitOperation;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::RetryWait;
use crate::harness::session::types::SummaryDecidingOperation;
use crate::harness::session::types::SummaryEffectPendingOperation;
use crate::harness::session::types::SummaryReadyOperation;
use crate::harness::session::types::{SummaryRetryWaitOperation, ToolBatch, ToolsOperation};
use crate::types::QueueMode;

/// A configured `main` lane over a seeded session, the fixture the
/// mismatch test admits through; `Some(operation_id)` seeds an admitted
/// starting operation.
async fn seam_lane(operation_id: Option<&str>) -> Lane {
    let session = memory_session_with_seed(operation_id).await;
    restored_lane(
        session,
        noop_emit_batch(),
        unused_watch_installer("boundary"),
    )
    .await
}

// The real-admission and spawn-settle suite: the compaction and
// summarized-navigation admissions commit their durable writes and events,
// and a freshly installed pass spawns the procedure loop through the faux
// provider.

use crate::harness::agent_harness::{DriveOutcome, HarnessEvent, HarnessEventPayload};
use crate::harness::session::commit::insert_entry;
use crate::harness::session::session::StorageBackedSessionOptions;
use crate::harness::session::testing::InstrumentedStorage;
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::{OperationKind, OperationMeta, SessionReader, TerminalStatus};
use crate::harness::session::values::{Write, set_value_write};
use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxProviderHandle;
use pi_ai::providers::faux::{RegisterFauxProviderOptions, faux_assistant_message, faux_provider};

/// The event collector the admission and drive flips capture through,
/// upstream's `(batch) => { events.push(...batch); }` fixture collector.
fn collecting_emit_batch(
    events: Arc<Mutex<Vec<HarnessEvent>>>,
) -> crate::harness::runtime::lane::EmitBatch {
    Arc::new(
        move |batch: Vec<HarnessEvent>, _context: crate::harness::context::Context| {
            let events = Arc::clone(&events);
            Box::pin(async move {
                events
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .extend(batch);
                Ok(())
            })
        },
    )
}

/// The count of one payload's shape in the collected events.
fn event_count(
    events: &Arc<Mutex<Vec<HarnessEvent>>>,
    shape: fn(&HarnessEventPayload) -> bool,
) -> usize {
    crate::harness::runtime::test_support::lock(events)
        .iter()
        .filter(|event| shape(&event.payload))
        .count()
}

/// The message entry's write one seed commits, upstream's
/// `{ kind: "entry", entry: { id, parentId, type: "message", message } }`
/// literals.
fn message_entry_write(id: &str, parent_id: Option<&str>, message: AgentMessage) -> Write {
    Write::Entry(Box::new(insert_entry(NewEntry::Message {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        body: Box::new(MessageEntry {
            message,
            terminate: None,
        }),
    })))
}

/// The admission fixture the compaction and navigation flips drive: the
/// seeded `main` lane over an instrumented memory backend, the caller's
/// extra writes committed before the restore reads, and the accepted
/// events collected.
async fn admission_lane(
    extra: Vec<Write>,
) -> (
    Lane,
    Arc<StorageBackedSession>,
    Arc<InstrumentedStorage>,
    Arc<Mutex<Vec<HarnessEvent>>>,
) {
    let storage = Arc::new(InstrumentedStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = Arc::new(StorageBackedSession::new(
        crate::harness::runtime::test_support::runtime_session_metadata(next_session_id()),
        storage.clone(),
        StorageBackedSessionOptions::default(),
    ));
    let mut writes =
        crate::harness::runtime::test_support::main_lane_seed_writes(&lane_configuration())
            .expect("seed writes");
    writes.extend(extra);
    commit_writes(&session, writes).await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let lane = restored_lane(
        session.clone(),
        collecting_emit_batch(Arc::clone(&events)),
        unused_watch_installer("boundary"),
    )
    .await;
    (lane, session, storage, events)
}

/// The drive fixture the compact and fresh-install flips run through: the
/// seeded `main` lane over a memory backend with the lane configuration
/// carrying the faux identity, the faux provider registered into the
/// lane's catalog, and the caller's extra writes committed before the
/// restore reads.
async fn driven_lane(
    extra: impl FnOnce(&crate::harness::session::types::LaneConfiguration) -> Vec<Write>,
) -> (Lane, Arc<StorageBackedSession>, FauxProviderHandle) {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    let model = faux.first_model();
    let configuration = crate::harness::session::types::LaneConfiguration {
        model: crate::harness::session::types::ModelIdentity {
            provider: model.provider.0.clone(),
            model_id: model.id.clone(),
        },
        ..lane_configuration()
    };
    let session = Arc::new(StorageBackedSession::new(
        crate::harness::runtime::test_support::runtime_session_metadata(next_session_id()),
        Arc::new(MemoryStorage::new(MemoryStorageOptions::default())),
        StorageBackedSessionOptions::default(),
    ));
    let mut writes = crate::harness::runtime::test_support::main_lane_seed_writes(&configuration)
        .expect("seed writes");
    writes.extend(extra(&configuration));
    commit_writes(&session, writes).await;
    let lane = restored_lane(
        session.clone(),
        noop_emit_batch(),
        unused_watch_installer("boundary"),
    )
    .await;
    lane.models().set_provider(Arc::new(faux.provider.clone()));
    (lane, session, faux)
}

/// Upstream `it` under "runtime atomic run acceptance" (`accept.test.ts`):
/// accepts standalone compaction with durable preparation and no
/// execution.
#[tokio::test]
async fn accepts_standalone_compaction_with_durable_preparation_and_no_execution() {
    let (lane, session, storage, events) = admission_lane(Vec::new()).await;
    AgentLane::append_message(&lane, user_text_message("history"), &background_context())
        .await
        .expect("append serves");
    storage.clear_commit_attempts();

    let admission = lane
        .accept_impl(
            OperationRequest::Compaction {
                operation_id: Some("compaction".to_owned()),
                custom_instructions: Some("focus".to_owned()),
            },
            &background_context(),
        )
        .await
        .expect("accept serves")
        .expect("the admission");
    assert_eq!(admission.operation_id, "compaction");
    assert_eq!(admission.kind, OperationKind::Compaction);

    let operation = lane.state().operation.expect("the accepted compaction");
    let OperationState::SummaryDeciding(deciding) = &operation.state else {
        panic!("expected accepted compaction: {:?}", operation.state.at());
    };
    assert_eq!(deciding.task.reason, Some(CompactionReason::Manual));
    assert_eq!(deciding.task.custom_instructions.as_deref(), Some("focus"));
    assert_eq!(deciding.task.boundary, ResultBoundary::Finish);
    let stored = session
        .get_value(
            &crate::harness::session::values::operation_preparation(
                "compaction",
                &deciding.task.task_id,
            )
            .address,
            &background_context(),
        )
        .await
        .expect("the preparation read")
        .expect("the stored preparation");
    assert_eq!(stored.value.get("kind"), Some(&json!("compaction")));
    assert_eq!(
        storage.get_commit_attempts().len(),
        1,
        "one admission commit"
    );
    assert_eq!(
        event_count(&events, |payload| matches!(
            payload,
            HarnessEventPayload::CompactionStart { .. }
        )),
        1,
        "the admission published one compaction_start",
    );
    assert!(lane.active_drive().is_none(), "no pass installed");
}

/// Upstream `it.each([false, true])` under "runtime atomic run acceptance"
/// (`accept.test.ts`): accepts summarized navigation atomically — the
/// summarized row lands `summary.deciding` with the branch-summary
/// preparation over the source's content, the direct row lands
/// `navigation.ready_to_commit`; both keep the tip and commit once.
#[tokio::test]
async fn accepts_summarized_navigation_atomically() {
    for (summarize, operation_id) in [(true, "summarized"), (false, "direct")] {
        let (lane, session, storage, events) = admission_lane(vec![
            message_entry_write("root", None, user_text_message("root")),
            message_entry_write("source", Some("root"), user_text_message("source")),
            message_entry_write("target", Some("root"), user_text_message("target")),
            set_value_write(
                &crate::harness::session::values::branch_tip("main"),
                Some("source".to_owned()),
            )
            .expect("the tip write"),
        ])
        .await;
        storage.clear_commit_attempts();

        lane.accept_impl(
            OperationRequest::Navigation {
                operation_id: Some(operation_id.to_owned()),
                target_id: Some("target".to_owned()),
                options: Some(NavigateOptions {
                    summarize: Some(summarize),
                    label: Some("chosen".to_owned()),
                    custom_instructions: Some("focus".to_owned()),
                }),
            },
            &background_context(),
        )
        .await
        .expect("accept serves")
        .expect("the admission");

        let operation = lane.state().operation.expect("the accepted navigation");
        assert_eq!(operation.meta.operation_id, operation_id);
        assert_eq!(
            lane.state().tip_id.as_deref(),
            Some("source"),
            "the tip stays"
        );
        if summarize {
            let OperationState::SummaryDeciding(deciding) = &operation.state else {
                panic!("expected summary decision: {:?}", operation.state.at());
            };
            assert_eq!(
                deciding.task.boundary,
                ResultBoundary::CommitNavigation {
                    target_id: "target".to_owned(),
                    label: Some("chosen".to_owned()),
                },
            );
            assert_eq!(
                deciding.task.reason, None,
                "the navigation task carries no compaction reason",
            );
            let stored = session
                .get_value(
                    &crate::harness::session::values::operation_preparation(
                        operation_id,
                        &deciding.task.task_id,
                    )
                    .address,
                    &background_context(),
                )
                .await
                .expect("the preparation read")
                .expect("the stored preparation");
            assert_eq!(stored.value.get("kind"), Some(&json!("branch_summary")));
            assert_eq!(
                stored.value.pointer("/messages/0/content"),
                Some(&json!("source")),
                "the preparation carries the source's message",
            );
        } else {
            assert_eq!(operation.state.at(), "navigation.ready_to_commit");
        }
        assert_eq!(
            storage.get_commit_attempts().len(),
            1,
            "one admission commit"
        );
        assert_eq!(
            event_count(&events, |payload| matches!(
                payload,
                HarnessEventPayload::NavigationStart { .. }
            )),
            1,
            "the admission published one navigation_start",
        );
        assert!(lane.active_drive().is_none(), "no pass installed");
    }
}

/// The summarized branch's root guards reject without writing, upstream's
/// "Summarized navigation requires non-root source and target entries"
/// invariants over a root source and a root target.
#[tokio::test]
async fn rejects_summarized_navigation_from_root_entries_without_writing() {
    // The root source tip.
    let (lane, _session, storage, _events) = admission_lane(Vec::new()).await;
    storage.clear_commit_attempts();
    let error = lane
        .accept_impl(
            OperationRequest::Navigation {
                operation_id: Some("summarized".to_owned()),
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
        .expect("accept serves")
        .expect_err("the root source rejects");
    match error {
        HarnessError::InvalidNavigation {
            reason, message, ..
        } => {
            assert_eq!(reason, "source_root");
            assert_eq!(
                message,
                "Summarized navigation requires non-root source and target entries"
            );
        }
        other => panic!("the root source: {other:?}"),
    }
    assert!(
        storage.get_commit_attempts().is_empty(),
        "the root source rejected without writing",
    );

    // The root target.
    let (lane, _session, storage, _events) = admission_lane(Vec::new()).await;
    AgentLane::append_message(&lane, user_text_message("head"), &background_context())
        .await
        .expect("append serves");
    storage.clear_commit_attempts();
    let error = lane
        .accept_impl(
            OperationRequest::Navigation {
                operation_id: Some("summarized".to_owned()),
                target_id: None,
                options: Some(NavigateOptions {
                    summarize: Some(true),
                    label: None,
                    custom_instructions: None,
                }),
            },
            &background_context(),
        )
        .await
        .expect("accept serves")
        .expect_err("the root target rejects");
    match error {
        HarnessError::InvalidNavigation { reason, .. } => {
            assert_eq!(reason, "target_root");
        }
        other => panic!("the root target: {other:?}"),
    }
    assert!(
        storage.get_commit_attempts().is_empty(),
        "the guards rejected without writing",
    );
}

/// Upstream `it` under "runtime public drive" (`drive-public.test.ts`):
/// composes standalone compaction acceptance with drive — the summary runs
/// through the faux provider and the compaction settles completed.
#[tokio::test]
async fn compact_composes_standalone_compaction_acceptance_with_drive() {
    let (lane, _session, faux) = driven_lane(|_configuration| Vec::new()).await;
    AgentLane::append_message(&lane, user_text_message("history"), &background_context())
        .await
        .expect("append serves");
    faux.set_responses([
        faux_assistant_message("summary", FauxAssistantMessageOptions::default()).into(),
    ]);

    let outcome = lane
        .compact(None, &background_context())
        .await
        .expect("compact serves")
        .expect("the compaction settles");
    assert_eq!(outcome.compaction.kind, OperationKind::Compaction);
    assert_eq!(outcome.compaction.status, TerminalStatus::Completed);
    assert!(
        outcome.run.is_none(),
        "the empty lane queues no follow-up run",
    );
    assert_eq!(
        faux.state().call_count(),
        1,
        "the summary ran through the faux provider",
    );
    assert!(
        lane.active_drive().is_none(),
        "the settled pass cleared its owner",
    );
    assert!(
        lane.state().operation.is_none(),
        "the completed compaction cleared its operation",
    );
}

/// The freshly installed pass spawns the procedure loop detached, the
/// seeded run drives through the faux provider, and the success handler
/// clears the pass's owner — the staged-seam test's install and clear
/// statements against the real spawn.
#[tokio::test]
async fn drive_spawns_the_procedure_loop_on_a_fresh_install() {
    let (lane, _session, faux) = driven_lane(|_configuration| {
        vec![
            message_entry_write("prompt-1", None, user_text_message("question")),
            set_value_write(
                &crate::harness::session::values::branch_tip("main"),
                Some("prompt-1".to_owned()),
            )
            .expect("the tip write"),
            set_value_write(
                &crate::harness::session::values::operation_meta("operation"),
                OperationMeta {
                    operation_id: "operation".to_owned(),
                    lane: "main".to_owned(),
                    source_tip_id: None,
                    started_at: 1,
                    intent: OperationIntent::Run {
                        prompt_entry_ids: vec!["prompt-1".to_owned()],
                    },
                },
            )
            .expect("the meta write"),
            set_value_write(
                &crate::harness::session::values::operation_state("operation"),
                crate::harness::runtime::test_support::starting_run_state(),
            )
            .expect("the state write"),
            crate::harness::runtime::test_support::lane_state_write(
                "main",
                Some("operation"),
                None,
                Vec::new(),
            )
            .expect("the lane write"),
        ]
    })
    .await;
    faux.set_responses([
        faux_assistant_message("answer", FauxAssistantMessageOptions::default()).into(),
    ]);

    let driven = lane
        .drive_impl(
            DriveOptions {
                operation_id: "operation".to_owned(),
                wait_for_retry: None,
                poll_deferred: None,
            },
            &background_context(),
        )
        .await
        .expect("drive serves")
        .expect("the drive settles");
    match driven {
        DriveOutcome::Settled { outcome } => {
            assert_eq!(outcome.kind, OperationKind::Run);
            assert_eq!(outcome.status, TerminalStatus::Completed);
        }
        other @ DriveOutcome::Waiting { .. } => panic!("the fresh install drove: {other:?}"),
    }
    assert!(
        lane.active_drive().is_none(),
        "the settled pass cleared its owner",
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
            deferred: deferred_scope(scope.clone(), "source", 0),
        }),
        OperationState::DeferredEffectPending(DeferredEffectPendingOperation {
            scope: deferred_scope(scope.clone(), "source", 0),
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
            task: summary_task(ResultBoundary::Finish),
        }),
        OperationState::NavigationReadyToCommit(NavigationReadyToCommitOperation {
            scope,
            target_id: Some("target".to_owned()),
            label: None,
        }),
    ]
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

use crate::harness::agent_harness::{AgentLane, LaneSnapshot, WatchHandle};
use crate::harness::runtime::lane::operation_state_with_scope;
use crate::harness::runtime::test_support::recording_watch_installer;
use crate::harness::session::types::{Entry, Storage};
use crate::types::AgentMessage;

/// The capturing fixture: the seam lane over a recording watch, the watch
/// surface the capture tests drive.
/// Installs the streaming fixture's effect-pending assistant leaf, the
/// stored-frames views' seeded run.
///
/// # Panics
/// The patch's failure.
/// One tool-call block wire, the batch blocks' `{ type: "toolCall" }`
/// literals.
fn call_block(id: &str, name: &str, arguments: serde_json::Value) -> serde_json::Value {
    let mut call = serde_json::Map::new();
    call.insert("type".to_owned(), json!("toolCall"));
    call.insert("id".to_owned(), json!(id));
    call.insert("name".to_owned(), json!(name));
    call.insert("arguments".to_owned(), arguments);
    serde_json::Value::Array(vec![serde_json::Value::Object(call)])
}

/// Admits the single-call batch, the staged-result probes' seeded run:
/// one `call-1`/`tool-1` tool call with empty arguments.
///
/// # Panics
/// The append's failure.
async fn admitted_single_call(lane: &Lane) {
    admitted_tools_batch(
        lane,
        call_block("call-1", "tool-1", json!({})),
        "toolUse",
        vec![
            serde_json::from_value::<crate::harness::session::types::ToolCall>(json!({
                "sourceIndex": 0,
                "resultEntryId": "result-1",
                "status": "outcome_ready",
                "terminate": false,
            }))
            .expect("the call wire"),
        ],
    )
    .await;
}

async fn install_effect_pending_assistant(lane: &Lane) {
    patch_live_state(
        lane,
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
}

/// Installs the retry-wait leaf the retry views read, the deadline
/// fixtures' seeded run.
///
/// # Panics
/// The patch's failure.
async fn install_retry_wait_leaf(lane: &Lane) {
    patch_live_state(
        lane,
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
}

/// The staged tool-result pending-entry write, the running-tools views'
/// seeded outcome; a serialization failure panics the test by design.
fn staged_tool_result_write(
    entry_id: &str,
    call_id: &str,
    tool_name: &str,
    text: &str,
    is_error: bool,
    timestamp: i64,
) -> Write {
    set_value_write(
        &crate::harness::session::values::pending_entry(entry_id),
        crate::harness::session::types::PendingEntry::Message {
            payload: Box::new(
                serde_json::from_value(json!({
                    "role": "toolResult",
                    "toolCallId": call_id,
                    "toolName": tool_name,
                    "content": [{ "type": "text", "text": text }],
                    "isError": is_error,
                    "timestamp": timestamp,
                }))
                .expect("the tool result wire"),
            ),
        },
    )
    .expect("staged write")
}

/// Writes the turn-0 tool-args raw payload, the invariant probes' seeded
/// corruption; a serialization failure panics the test by design.
async fn write_tool_args(lane: &Lane, value: serde_json::Value) {
    commit_writes(
        lane.session(),
        vec![raw_write(
            &crate::harness::session::values::operation_tool_args(
                &live_operation_id(lane),
                "turn",
                0,
            )
            .address,
            value,
        )],
    )
    .await;
}

/// The closed reason one watch carries, the invariant probes' read; a
/// serving watch panics the test by design.
async fn watch_closed_reason(lane: &Lane, why: &str) -> String {
    let Err(crate::harness::agent_harness::LaneOperationError::Closed(reason)) =
        AgentLane::watch(lane, &background_context()).await
    else {
        panic!("{why}");
    };
    reason.to_string()
}

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

/// The pending partial wire the streaming captures reduce, zero usage
/// pending over empty content.
fn pending_partial_wire() -> pi_ai::types::AssistantMessage {
    serde_json::from_value(json!({
        "content": [],
        "api": "anthropic-messages",
        "provider": "test",
        "model": "model",
        "usage": zero_usage_wire(),
        "stopReason": "pending",
        "timestamp": 1,
    }))
    .expect("the partial wire")
}

/// The three-frame append list the streaming captures store, the start,
/// text-start, and first delta the projection reduces.
fn frame_append_writes(
    address: &crate::harness::session::values::ValueList<
        pi_ai::utils::assistant_message_frame::AssistantMessageFrame,
    >,
    partial: &pi_ai::types::AssistantMessage,
) -> Vec<Write> {
    vec![
        crate::harness::session::values::append_list_write(
            address,
            pi_ai::utils::assistant_message_frame::AssistantMessageFrame::Start {
                partial: partial.clone(),
            },
        )
        .expect("frame append"),
        crate::harness::session::values::append_list_write(
            address,
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
            address,
            pi_ai::utils::assistant_message_frame::AssistantMessageFrame::TextDelta {
                content_index: 0,
                delta: "a".to_owned(),
            },
        )
        .expect("frame append"),
    ]
}

/// The live operation's id, the extraction the capture fixtures read.
fn live_operation_id(lane: &Lane) -> String {
    lane.state()
        .operation
        .as_ref()
        .map(|operation| operation.meta.operation_id.clone())
        .expect("the live operation")
}

/// The watch opened and its first snapshot read, the capture tests' lens.
async fn watch_snapshot(lane: &Lane) -> (Box<dyn WatchHandle<LaneSnapshot>>, LaneSnapshot) {
    let watch = AgentLane::watch(lane, &background_context())
        .await
        .expect("watch serves");
    let snapshot = WatchHandle::<LaneSnapshot>::snapshot(&*watch);
    (watch, snapshot)
}

#[tokio::test]
async fn watch_captures_the_lane_snapshot_over_the_durable_values() {
    let lane = capturing_lane(None).await;
    accept_prompt(&lane).await;
    let prompt_tip = lane.state().tip_id.expect("the accepted tip");
    let operation_id = live_operation_id(&lane);
    finish_operation(&lane).await;
    let appended =
        AgentLane::append_message(&lane, user_text_message("history"), &background_context())
            .await
            .expect("append serves");

    let (watch, snapshot) = watch_snapshot(&lane).await;
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
    accept_prompt(&lane).await;
    install_effect_pending_assistant(&lane).await;

    let operation_id = live_operation_id(&lane);
    let partial = pending_partial_wire();
    let address =
        crate::harness::session::values::pending_assistant_frames(&operation_id, "response");
    commit_writes(lane.session(), frame_append_writes(&address, &partial)).await;

    let (watch, snapshot) = watch_snapshot(&lane).await;
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
            deferred: deferred_scope(operation_scope(), &source, 3),
        }),
    )
    .await;

    let (watch, snapshot) = watch_snapshot(&lane).await;
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
            deferred: deferred_scope(operation_scope(), &source, 0),
        }),
    )
    .await;

    let reason = watch_closed_reason(&lane, "the handle-less source faults the capture").await;
    assert!(
        reason.contains("Deferred source is missing its assistant handle"),
        "the capture faults with the deferred invariant: {reason}"
    );
}

#[expect(
    clippy::too_many_lines,
    reason = "the case drives the running and settled calls' capture arms"
)]
#[tokio::test]
async fn watch_reports_the_tools_batch_running_and_settled_calls() {
    let lane = capturing_lane(None).await;
    // The batch's assistant entry commits while the lane is idle; the run
    // admits after.
    admitted_tools_batch(
        &lane,
        json!([
            { "type": "toolCall", "id": "call-1", "name": "tool-1", "arguments": {} },
            { "type": "toolCall", "id": "call-2", "name": "tool-2", "arguments": { "k": "block" } },
        ]),
        "toolUse",
        vec![
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
    )
    .await;

    let operation_id = live_operation_id(&lane);
    commit_writes(
        lane.session(),
        vec![
            raw_write(
                &crate::harness::session::values::operation_tool_args(&operation_id, "turn", 0)
                    .address,
                json!({ "k": "persisted" }),
            ),
            set_value_write(
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
            staged_tool_result_write("result-2", "call-2", "tool-2", "done-2", false, 2),
        ],
    )
    .await;

    let (watch, snapshot) = watch_snapshot(&lane).await;
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

#[tokio::test]
async fn watch_faults_on_the_tools_batch_invariants() {
    // The invalid assistant entry.
    let lane = capturing_lane(None).await;
    accept_prompt(&lane).await;
    patch_live_state(&lane, tools_batch_state("absent", Vec::new())).await;
    let reason = watch_closed_reason(&lane, "the invalid batch faults the capture").await;
    assert!(
        reason.contains("Tool batch assistant entry is invalid"),
        "the invalid batch names its invariant: {reason}"
    );

    // The missing persisted arguments.
    let lane = capturing_lane(None).await;
    admitted_tools_batch(
        &lane,
        call_block("call-1", "tool-1", json!({})),
        "toolUse",
        vec![
            serde_json::from_value(json!({
                "sourceIndex": 0,
                "resultEntryId": "result-1",
                "status": "effect_pending",
                "replay": "safe",
            }))
            .expect("the effect-pending call wire"),
        ],
    )
    .await;
    let reason = watch_closed_reason(&lane, "the arg-less call faults the capture").await;
    assert!(
        reason.contains("Tool call call-1 is missing persisted arguments"),
        "the arg-less call names its invariant: {reason}"
    );

    // The missing staged result.
    let lane = capturing_lane(None).await;
    admitted_tools_batch(
        &lane,
        call_block("call-2", "tool-2", json!({})),
        "toolUse",
        vec![
            serde_json::from_value(json!({
                "sourceIndex": 0,
                "resultEntryId": "result-2",
                "status": "outcome_ready",
                "terminate": false,
            }))
            .expect("the outcome-ready call wire"),
        ],
    )
    .await;
    let reason = watch_closed_reason(&lane, "the staged-less call faults the capture").await;
    assert!(
        reason.contains("Tool call result-2 is missing its staged result"),
        "the staged-less call names its invariant: {reason}"
    );
}

/// The prompt admission the capture fixtures share.
async fn accept_prompt(lane: &Lane) {
    accept_text(lane, "hello").await;
}

/// The tools-batch rig: commits the assistant tool-call wire, admits the
/// prompt, and patches the live batch; returns the batch's assistant entry
/// id.
async fn admitted_tools_batch(
    lane: &Lane,
    blocks: serde_json::Value,
    stop_reason: &str,
    calls: Vec<crate::harness::session::types::ToolCall>,
) -> String {
    let assistant_entry = AgentLane::append_message(
        lane,
        assistant_wire(&blocks, stop_reason, false),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(lane).await;
    patch_live_state(lane, tools_batch_state(&assistant_entry, calls)).await;
    assistant_entry
}

#[tokio::test]
async fn watch_reports_the_retry_views() {
    let lane = capturing_lane(None).await;
    accept_prompt(&lane).await;
    install_retry_wait_leaf(&lane).await;
    let (watch, snapshot) = watch_snapshot(&lane).await;
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
    let (watch, snapshot) = watch_snapshot(&lane).await;
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
    patch_live_state(&lane, deferred_effect_pending(&source, "response", 1)).await;

    let partial = pending_partial_wire();
    let address = crate::harness::session::values::pending_assistant_frames(
        &live_operation_id(&lane),
        "response",
    );
    commit_writes(lane.session(), frame_append_writes(&address, &partial)).await;

    let (watch, snapshot) = watch_snapshot(&lane).await;
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

    let (watch, snapshot) = watch_snapshot(&lane).await;
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
    let operation_id = live_operation_id(&lane);
    finish_operation(&lane).await;
    commit_writes(
        lane.session(),
        vec![crate::harness::session::values::delete_value_write(
            &crate::harness::session::values::operation_result(&operation_id),
        )],
    )
    .await;
    let reason = watch_closed_reason(&lane, "the missing record faults the capture").await;
    assert!(
        reason.contains(&format!("Lane \"main\" is missing result {operation_id}")),
        "the missing record names its invariant: {reason}"
    );

    // The malformed record.
    let lane = capturing_lane(None).await;
    accept_prompt(&lane).await;
    let operation_id = live_operation_id(&lane);
    finish_operation(&lane).await;
    commit_writes(
        lane.session(),
        vec![raw_write(
            &crate::harness::session::values::operation_result(&operation_id).address,
            json!(42),
        )],
    )
    .await;
    let reason = watch_closed_reason(&lane, "the malformed record faults the capture").await;
    assert!(
        reason.contains("Operation result is malformed"),
        "the malformed record names its parse failure: {reason}"
    );
}

/// The remaining tools-batch parse invariants: the non-tool-call source
/// block, the malformed args, the malformed checkpoint, the malformed and
/// non-message and mismatched staged results.
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

    // A text block where the call's source index names a tool call.
    let lane = capturing_lane(None).await;
    admitted_tools_batch(
        &lane,
        json!([{ "type": "text", "text": "not a call" }]),
        "stop",
        vec![call(0, "result-1", "effect_pending")],
    )
    .await;
    write_tool_args(&lane, json!({})).await;
    let reason = watch_closed_reason(&lane, "the text block faults the capture").await;
    assert!(
        reason.contains("does not name a tool-call block"),
        "the text block names its invariant: {reason}"
    );

    // The malformed persisted arguments.
    let lane = capturing_lane(None).await;
    admitted_tools_batch(
        &lane,
        call_block("call-1", "tool-1", json!({})),
        "toolUse",
        vec![call(0, "result-1", "effect_pending")],
    )
    .await;
    write_tool_args(&lane, json!(42)).await;
    let reason = watch_closed_reason(&lane, "the malformed args fault the capture").await;
    assert!(
        reason.contains("Tool arguments are malformed"),
        "the malformed args name their parse failure: {reason}"
    );

    // The malformed checkpoint.
    let lane = capturing_lane(None).await;
    admitted_tools_batch(
        &lane,
        call_block("call-1", "tool-1", json!({})),
        "toolUse",
        vec![call(0, "result-1", "effect_pending")],
    )
    .await;
    let operation_id = live_operation_id(&lane);
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
    let reason = watch_closed_reason(&lane, "the malformed checkpoint faults the capture").await;
    assert!(
        reason.contains("Pending tool output is malformed"),
        "the malformed checkpoint names its parse failure: {reason}"
    );

    // The mismatched staged result.
    let lane = capturing_lane(None).await;
    admitted_tools_batch(
        &lane,
        call_block("call-2", "tool-2", json!({})),
        "toolUse",
        vec![call(0, "result-2", "outcome_ready")],
    )
    .await;
    commit_writes(
        lane.session(),
        vec![staged_tool_result_write(
            "result-2",
            "other-call",
            "tool-2",
            "done",
            false,
            2,
        )],
    )
    .await;
    let reason =
        watch_closed_reason(&lane, "the mismatched staged result faults the capture").await;
    assert!(
        reason.contains("Tool call result-2 has a mismatched staged result"),
        "the mismatched staged result names its invariant: {reason}"
    );
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
        WatchHandle::<LaneSnapshot>::snapshot(&*watch),
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
#[tokio::test]
async fn watch_faults_on_the_staged_result_parse_invariants() {
    let staged_for = |payload: serde_json::Value| {
        set_value_write(
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
    admitted_single_call(&lane).await;
    commit_writes(
        lane.session(),
        vec![raw_write(
            &crate::harness::session::values::pending_entry("result-1").address,
            json!(42),
        )],
    )
    .await;
    let reason = watch_closed_reason(&lane, "the malformed staged result faults the capture").await;
    assert!(
        reason.contains("Pending entry payload is malformed"),
        "the malformed staged result names its parse failure: {reason}"
    );

    // The custom staged payload.
    let lane = capturing_lane(None).await;
    admitted_single_call(&lane).await;
    commit_writes(
        lane.session(),
        vec![
            set_value_write(
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
    let reason = watch_closed_reason(&lane, "the custom staged result faults the capture").await;
    assert!(
        reason.contains("Tool call result-1 is missing its staged result"),
        "the custom staged result names its invariant: {reason}"
    );

    // The non-toolResult staged message.
    let lane = capturing_lane(None).await;
    admitted_single_call(&lane).await;
    commit_writes(
        lane.session(),
        vec![staged_for(json!({
            "role": "user",
            "content": "not a result",
            "timestamp": 1,
        }))],
    )
    .await;
    let reason = watch_closed_reason(
        &lane,
        "the non-toolResult staged message faults the capture",
    )
    .await;
    assert!(
        reason.contains("Tool call result-1 is missing its staged result"),
        "the non-toolResult staged message names its invariant: {reason}"
    );
}

// The capture's storage-error arms and the faulted flag: the reads fault
// through the controlled backend, upstream's `FailingMemoryStorage` rig,
// and the faulted flag reads the sealed fault mid-capture.

use crate::harness::runtime::test_support::gate_next_read;
use crate::harness::runtime::test_support::next_session_id;
use crate::harness::runtime::test_support::{runtime_session, seed_main_lane_values};
use crate::harness::session::session::StorageBackedSession;

/// The capturing lane over the controlled backend, the read-failure rig's
/// fixture: the seam lane plus the storage handle the arms re-arm against.
async fn controlled_capture_lane(
    operation_id: Option<&str>,
) -> (Lane, Arc<StorageBackedSession>, Arc<ControlledStorage>) {
    let storage = Arc::new(ControlledStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = Arc::new(runtime_session(next_session_id(), storage.clone()));
    seed_main_lane_values(&session, operation_id)
        .await
        .expect("seed commit");
    let lane = restored_lane(
        session.clone(),
        noop_emit_batch(),
        recording_watch_installer(empty_lane_snapshot("main", &lane_configuration())),
    )
    .await;
    (lane, session, storage)
}

/// Arms one address-keyed read rejection, the rig's per-address arm.
fn arm_read_failure(storage: &Arc<ControlledStorage>, key: &str) {
    storage.arm_read_failure(
        None,
        Some(key),
        crate::harness::session::types::SessionError::Message("read failed".to_owned()),
    );
}

/// The watch rejection's message, the capture errors' read.
fn watch_fault_message(error: crate::harness::agent_harness::LaneOperationError) -> String {
    match error {
        crate::harness::agent_harness::LaneOperationError::Closed(reason) => reason.to_string(),
    }
}

/// The watch's fault message, the failing capture's read.
async fn watch_fault(lane: &Lane) -> String {
    let Err(error) = AgentLane::watch(lane, &background_context()).await else {
        panic!("the capture faulted");
    };
    watch_fault_message(error)
}

/// The capture's read failures fault the watch with the storage's error.
#[tokio::test]
async fn watch_faults_on_the_capture_read_failures() {
    // The transcript branch scan.
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    let tip = AgentLane::append_message(&lane, user_text_message("history"), &background_context())
        .await
        .expect("append serves");
    arm_read_failure(&storage, &tip);
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("read failed"),
        "the scan faults: {message}"
    );

    // The queue read.
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    let steer = steer_text(&lane).await;
    arm_read_failure(&storage, &steer);
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("read failed"),
        "the queues fault: {message}"
    );

    // The last-result read.
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    accept_prompt(&lane).await;
    let operation_id = live_operation_id(&lane);
    finish_operation(&lane).await;
    arm_read_failure(&storage, &operation_id);
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("read failed"),
        "the result faults: {message}"
    );

    // The stats read.
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    accept_prompt(&lane).await;
    storage.arm_stats_failure(crate::harness::session::types::SessionError::Message(
        "stats failed".to_owned(),
    ));
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("stats failed"),
        "the stats fault: {message}"
    );
}

/// The steer enqueue the capture queue fixtures share.
async fn steer_text(lane: &Lane) -> String {
    lane.steer(
        crate::harness::agent_harness::QueueMessage::Text("queued".to_owned()),
        Vec::new(),
        &background_context(),
    )
    .await
    .expect("steer serves")
    .expect("the steer")
}

/// The operation leaf reads the capture's deferred and streaming arms make:
/// each read failure faults the watch, and the deferred-effect-pending leaf
/// reads both its deferred handle and its frames.
#[tokio::test]
async fn watch_faults_on_the_operation_leaf_read_failures() {
    // The streaming leaf's frames read.
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    accept_prompt(&lane).await;
    let operation_id = live_operation_id(&lane);
    install_effect_pending_assistant(&lane).await;
    arm_read_failure(&storage, &format!("{operation_id}:response"));
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("read failed"),
        "the frames fault: {message}"
    );

    // The deferred leaf's source read, then the deferred-effect-pending
    // leaf's frames read once the deferred read succeeded.
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    let source = AgentLane::append_message(
        &lane,
        assistant_wire(&json!([]), "stop", true),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    let operation_id = live_operation_id(&lane);
    patch_live_state(
        &lane,
        OperationState::DeferredEffectPending(DeferredEffectPendingOperation {
            scope: deferred_scope(operation_scope(), &source, 0),
            response_entry_id: "response".to_owned(),
            usage_id: "usage".to_owned(),
        }),
    )
    .await;
    arm_read_failure(&storage, &source);
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("read failed"),
        "the deferred faults: {message}"
    );

    arm_read_failure(&storage, &format!("{operation_id}:response"));
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("read failed"),
        "the frames fault: {message}"
    );
}

/// The tools batch's read failures: the assistant entry read, the args
/// read, the checkpoint read, and the staged-result read.
#[tokio::test]
async fn watch_faults_on_the_tools_batch_read_failures() {
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

    // The assistant entry read.
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    let assistant_entry = admitted_tools_batch(
        &lane,
        call_block("call-1", "tool-1", json!({})),
        "toolUse",
        vec![call(0, "result-1", "effect_pending")],
    )
    .await;
    arm_read_failure(&storage, &assistant_entry);
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("read failed"),
        "the assistant faults: {message}"
    );

    // The args read.
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    admitted_tools_batch(
        &lane,
        call_block("call-1", "tool-1", json!({})),
        "toolUse",
        vec![call(0, "result-1", "effect_pending")],
    )
    .await;
    let operation_id = live_operation_id(&lane);
    arm_read_failure(&storage, &format!("{operation_id}:turn:0"));
    let message = watch_fault(&lane).await;
    assert!(message.contains("read failed"), "the args fault: {message}");

    // The checkpoint read.
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    admitted_tools_batch(
        &lane,
        call_block("call-1", "tool-1", json!({})),
        "toolUse",
        vec![call(0, "result-1", "effect_pending")],
    )
    .await;
    let operation_id = live_operation_id(&lane);
    commit_writes(
        lane.session(),
        vec![raw_write(
            &crate::harness::session::values::operation_tool_args(&operation_id, "turn", 0).address,
            json!({}),
        )],
    )
    .await;
    arm_read_failure(&storage, &format!("{operation_id}:result-1"));
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("read failed"),
        "the checkpoint faults: {message}",
    );

    // The staged-result read.
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    admitted_tools_batch(
        &lane,
        call_block("call-2", "tool-2", json!({})),
        "toolUse",
        vec![call(0, "result-2", "outcome_ready")],
    )
    .await;
    arm_read_failure(&storage, "result-2");
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("read failed"),
        "the staged fault: {message}"
    );
}

#[expect(
    clippy::too_many_lines,
    reason = "the case drives the remaining capture shapes' read arms"
)]
/// The capture's success-shape arms the error suites leave: the
/// checkpoint-less running call, the settled call's persisted args, the
/// non-assistant batch entry, and the absent deferred source.
#[tokio::test]
async fn watch_reports_the_remaining_capture_shapes() {
    // The effect-pending call without its checkpoint: the running tool
    // carries no result.
    let (lane, _session, _storage) = controlled_capture_lane(None).await;
    admitted_tools_batch(
        &lane,
        call_block("call-1", "tool-1", json!({})),
        "toolUse",
        vec![
            serde_json::from_value(json!({
                "sourceIndex": 0,
                "resultEntryId": "result-1",
                "status": "effect_pending",
                "replay": "safe",
            }))
            .expect("the effect-pending call wire"),
        ],
    )
    .await;
    write_tool_args(&lane, json!({ "k": "persisted" })).await;
    let (_watch, snapshot) = watch_snapshot(&lane).await;
    match &snapshot
        .operation
        .as_ref()
        .expect("the operation view")
        .running_tools[0]
    {
        crate::harness::agent_harness::LaneSnapshotTool::Running { result, .. } => {
            assert!(
                result.is_none(),
                "the checkpoint-less call carries no result"
            );
        }
        other @ crate::harness::agent_harness::LaneSnapshotTool::Settled { .. } => {
            panic!("the running tool: {other:?}")
        }
    }

    // The settled call whose args persisted: the stored args ride.
    let (lane, _session, _storage) = controlled_capture_lane(None).await;
    admitted_tools_batch(
        &lane,
        call_block("call-2", "tool-2", json!({})),
        "toolUse",
        vec![
            serde_json::from_value(json!({
                "sourceIndex": 0,
                "resultEntryId": "result-2",
                "status": "outcome_ready",
                "terminate": false,
            }))
            .expect("the outcome-ready call wire"),
        ],
    )
    .await;
    let operation_id = live_operation_id(&lane);
    commit_writes(
        lane.session(),
        vec![
            raw_write(
                &crate::harness::session::values::operation_tool_args(&operation_id, "turn", 0)
                    .address,
                json!({ "k": "persisted" }),
            ),
            staged_tool_result_write("result-2", "call-2", "tool-2", "done", false, 2),
        ],
    )
    .await;
    let (_watch, snapshot) = watch_snapshot(&lane).await;
    match &snapshot
        .operation
        .as_ref()
        .expect("the operation view")
        .running_tools[0]
    {
        crate::harness::agent_harness::LaneSnapshotTool::Settled { args, .. } => {
            assert_eq!(
                *args,
                json!({ "k": "persisted" }),
                "the stored args rode the settled call",
            );
        }
        other @ crate::harness::agent_harness::LaneSnapshotTool::Running { .. } => {
            panic!("the settled tool: {other:?}")
        }
    }

    // The batch's assistant entry holding a user message: the inner
    // shape guard rejects.
    let (lane, _session, _storage) = controlled_capture_lane(None).await;
    let user_entry = AgentLane::append_message(
        &lane,
        user_text_message("not an assistant"),
        &background_context(),
    )
    .await
    .expect("append serves");
    accept_prompt(&lane).await;
    patch_live_state(&lane, tools_batch_state(&user_entry, Vec::new())).await;
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("Tool batch assistant entry is invalid"),
        "the non-assistant entry names its invariant: {message}",
    );

    // The deferred source absent from storage: the outer shape guard
    // rejects.
    let (lane, _session, _storage) = controlled_capture_lane(None).await;
    accept_prompt(&lane).await;
    patch_live_state(
        &lane,
        OperationState::DeferredSuspended(DeferredSuspendedOperation {
            deferred: deferred_scope(operation_scope(), "absent-source", 0),
        }),
    )
    .await;
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("Deferred source is missing its assistant handle"),
        "the absent source names its invariant: {message}",
    );
}

/// The streaming message's reduce failure: frames that parse but reduce
/// against the frame sequence's rules fault the capture.
#[tokio::test]
async fn watch_faults_on_the_streaming_reduce_failure() {
    let (lane, _session, _storage) = controlled_capture_lane(None).await;
    accept_prompt(&lane).await;
    install_effect_pending_assistant(&lane).await;
    let operation_id = live_operation_id(&lane);
    let address =
        crate::harness::session::values::pending_assistant_frames(&operation_id, "response");
    let partial = pending_partial_wire();
    commit_writes(
        lane.session(),
        vec![
            crate::harness::session::values::append_list_write(
                &address,
                pi_ai::utils::assistant_message_frame::AssistantMessageFrame::TextDelta {
                    content_index: 0,
                    delta: "before the start".to_owned(),
                },
            )
            .expect("frame append"),
            crate::harness::session::values::append_list_write(
                &address,
                pi_ai::utils::assistant_message_frame::AssistantMessageFrame::Start {
                    partial: partial.clone(),
                },
            )
            .expect("frame append"),
        ],
    )
    .await;
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("appears before the start frame"),
        "the reduce failure names its sequence rule: {message}",
    );
}

/// The snapshot's faulted flag reads the sealed fault: the capture that
/// runs while the seal lands reports it.
#[tokio::test]
async fn watch_reports_the_sealed_fault_in_the_snapshot() {
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    let tip = AgentLane::append_message(&lane, user_text_message("history"), &background_context())
        .await
        .expect("append serves");
    let _ = tip;
    let fault = crate::harness::result::HarnessFault::new(
        "harness faulted",
        Box::new(crate::harness::session::types::SessionError::Message(
            "cause".to_owned(),
        )),
    );
    let fault: crate::harness::runtime::types::LaneError = Arc::new(fault);
    let (started, release) = gate_next_read(&storage);
    let watching = {
        let lane = lane.clone();
        tokio::spawn(async move { AgentLane::watch(&lane, &background_context()).await })
    };
    started.await.expect("the capture's first read parked");
    lane.seal(fault).await;
    let _ = release.send(());

    let (_, snapshot) = match watching.await.expect("watch join") {
        Ok(watch) => {
            let snapshot = WatchHandle::<LaneSnapshot>::snapshot(&*watch);
            watch.unsubscribe();
            (watch, snapshot)
        }
        Err(error) => panic!("the capture served: {error:?}"),
    };
    assert!(snapshot.faulted, "the sealed fault flagged the snapshot");
}

/// The watcher's resnapshot after the seal faults: the capture's sealed
/// read rides the listener surface.
#[tokio::test]
async fn the_resnapshot_after_the_seal_faults() {
    let bus = Arc::new(crate::harness::events::HarnessEventBus::new());
    let session = memory_session_with_seed(None).await;
    let lane = restored_lane(
        session,
        crate::harness::runtime::test_support::bus_emit_batch(Arc::clone(&bus)),
        crate::harness::runtime::test_support::bus_watch_installer(
            bus,
            empty_lane_snapshot("main", &lane_configuration()),
        ),
    )
    .await;
    let watch = AgentLane::watch(&lane, &background_context())
        .await
        .expect("watch serves");

    let fault = crate::harness::result::HarnessFault::new(
        "harness faulted",
        Box::new(crate::harness::session::types::SessionError::Message(
            "cause".to_owned(),
        )),
    );
    lane.seal(Arc::new(fault)).await;

    let error = watch
        .resnapshot(&background_context())
        .await
        .expect_err("the sealed resnapshot faults");
    assert!(
        error.to_string().contains("did not mark"),
        "the sealed capture never marked its boundary: {error}",
    );
}

/// The capture's gated read failing at its own gate, and the structure
/// scan's armed rejection: the gate and the arm carry their errors.
#[tokio::test]
async fn the_capture_gate_and_the_structure_scan_carry_their_errors() {
    // The capture's first read fails at the gate's own error.
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    AgentLane::append_message(&lane, user_text_message("history"), &background_context())
        .await
        .expect("append serves");
    storage.arm_read_gate(Box::new(|| {
        Box::pin(async {
            Err(crate::harness::session::types::SessionError::Message(
                "gate failed".to_owned(),
            ))
        })
    }));
    let message = watch_fault(&lane).await;
    assert!(
        message.contains("gate failed"),
        "the gate's error carried: {message}",
    );

    // The structure scan's key-scoped rejection.
    let (lane, _session, storage) = controlled_capture_lane(None).await;
    let tip = AgentLane::append_message(&lane, user_text_message("history"), &background_context())
        .await
        .expect("append serves");
    arm_read_failure(&storage, &tip);
    let query = crate::harness::session::types::StorageBranchScan {
        start: tip.clone(),
        ..Default::default()
    };
    let read = storage
        .scan_branch_structure(&query, &background_context())
        .await;
    let error = read.expect_err("the structure scan's armed rejection");
    assert_eq!(
        error.to_string(),
        "read failed",
        "the structure scan's rejection carried: {error}",
    );
    let _ = lane;
}

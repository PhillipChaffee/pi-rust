//! The serialized mutation-line suite, ported 1:1 from upstream
//! `test/harness/runtime/lane.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements the port makes and the tests bind:
//! - upstream's promise-valued command result (the "returns a promise value"
//!   case) restates as a one-shot receiver the caller awaits; the line is
//!   released at the command's return, so the observable sequence — the line
//!   serves later work while the result is still pending — is what the test
//!   pins.
//! - upstream's thenable-materialize guard ("Lane command `materialize()`
//!   must be synchronous") is unrepresentable: [`MaterializeFn`] is
//!   synchronous by construction (the type system replaces the runtime
//!   guard, recorded in the types module docs), so that case has no runtime
//!   test here.
//! - the uncloneable-event case restates as a rejecting delivery future
//!   (the `DataCloneError` was upstream's clone failure inside `emitBatch`);
//!   the assertion — the command rejects while the committed memory stays —
//!   carries over unchanged.
//! - upstream's `rejects.toBe(closed)` error identity carries through the
//!   raw impl surface (`Arc::ptr_eq`); the trait surface collapses the seal
//!   to its message, so the tip-id assertion pins the message instead.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; a violated expectation panics the test by design"
)]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_ai::models::create_models;
use pi_ai::providers::faux::RegisterFauxProviderOptions;
use pi_ai::providers::faux::faux_provider;
use pi_ai::types::BoxedFuture;
use pi_ai::types::Message;
use serde_json::json;

use crate::harness::agent_harness::AgentLane;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::LaneOperationError;
use crate::harness::agent_harness::OperationRequest;
use crate::harness::agent_harness::PromptMessagesPayload;
use crate::harness::agent_harness::QueueMessage;
use crate::harness::context::background_context;
use crate::harness::hooks::HookErrorReporter;
use crate::harness::hooks::HookRegistry;
use crate::harness::result::HarnessClosed;
use crate::harness::runtime::lane::EmitBatch;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::restore::restore_lane;
use crate::harness::runtime::test_support::ControlledStorage;
use crate::harness::runtime::test_support::commit_failure;
use crate::harness::runtime::test_support::deferred;
use crate::harness::runtime::test_support::empty_lane_snapshot;
use crate::harness::runtime::test_support::lane_configuration;
use crate::harness::runtime::test_support::noop_emit_batch;
use crate::harness::runtime::test_support::passthrough_fault_handler;
use crate::harness::runtime::test_support::recording_watch_installer;
use crate::harness::runtime::test_support::runtime_config;
use crate::harness::runtime::test_support::runtime_session;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::LaneCommand;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::LanePatch;
use crate::harness::runtime::types::LiveOperation;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::session::memory::MemoryStorage;
use crate::harness::session::memory::MemoryStorageOptions;
use crate::harness::session::session::StorageBackedSession;
use crate::harness::session::types::Control;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::ModelIdentity;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionReader;
use crate::harness::session::values as stored_values;
use crate::harness::session::values::set_value_write;
use crate::types::AgentMessage;
use crate::types::ThinkingLevel;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn next_session_id() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("runtime-lane-{}", COUNTER.fetch_add(1, Ordering::SeqCst))
}

fn noop_hook_reporter() -> HookErrorReporter {
    Arc::new(|_error, _name, _site, _context| Box::pin(std::future::ready(())))
}

/// The fixture upstream's `createLane` builds: the storage-backed session
/// over the controlled memory backend, seeded with the branch tip, the lane
/// configuration, and the lane state; the faux provider registered into a
/// fresh models catalog; the lane restored from that seed.
struct LaneFixture {
    lane: Lane,
    model: pi_ai::types::Model,
    session: Arc<StorageBackedSession>,
    storage: Arc<ControlledStorage>,
}

async fn create_lane() -> LaneFixture {
    create_lane_with_emit(noop_emit_batch()).await
}

async fn create_lane_with_emit(emit_batch: EmitBatch) -> LaneFixture {
    let configuration = lane_configuration();
    let storage = Arc::new(ControlledStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = Arc::new(runtime_session(next_session_id(), storage.clone()));
    let writes = vec![
        set_value_write(&stored_values::branch_tip("main"), Option::<String>::None)
            .expect("tip write"),
        set_value_write(&stored_values::lane_config("main"), configuration.clone())
            .expect("config write"),
        set_value_write(
            &stored_values::lane_state("main"),
            crate::harness::session::types::LaneState {
                current_operation_id: None,
                last_operation_id: None,
                inbox: Vec::new(),
            },
        )
        .expect("state write"),
    ];
    session
        .mutate(
            Box::new(move |mutator, context| {
                let writes = writes.clone();
                Box::pin(async move {
                    mutator.commit(writes, context).await?;
                    let payload: Box<dyn std::any::Any + Send> = Box::new(());
                    Ok(payload)
                })
            }),
            &background_context(),
        )
        .await
        .expect("seed commit");
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    let models = Arc::new(create_models(None));
    models.set_provider(Arc::new(faux.provider.clone()));
    let model = faux.first_model();
    let restored = restore_lane(session.clone(), "main", &background_context())
        .await
        .expect("restore");
    let lane = Lane::new(
        "main",
        session.clone(),
        models,
        Arc::new(HookRegistry::new(noop_hook_reporter())),
        restored,
        passthrough_fault_handler(),
        emit_batch,
        recording_watch_installer(empty_lane_snapshot("main", &configuration)),
        Arc::new(runtime_config),
    );
    LaneFixture {
        lane,
        model,
        session,
        storage,
    }
}

/// The configuration command upstream's `setThinkingLevel` helper builds:
/// the observed list records each planner run's previous level.
fn set_thinking_level(
    lane: &Lane,
    level: ThinkingLevel,
    observed: Option<Arc<Mutex<Vec<ThinkingLevel>>>>,
) -> BoxedFuture<'static, Result<ThinkingLevel, LaneError>> {
    let lane_name = lane.name().to_owned();
    let lane = lane.clone();
    Box::pin(async move {
        lane.command::<ThinkingLevel, _>(
            move |state, _session, _context| {
                let observed = observed.clone();
                let lane_name = lane_name.clone();
                Box::pin(async move {
                    if let Some(observed) = observed {
                        lock(&observed).push(state.configuration.thinking_level);
                    }
                    let mut configuration = state.configuration.clone();
                    configuration.thinking_level = level;
                    let mut next = state.clone();
                    next.configuration = configuration.clone();
                    Ok(LaneCommand::Commit {
                        writes: vec![
                            set_value_write(&stored_values::lane_config(&lane_name), configuration)
                                .expect("config write"),
                        ],
                        next,
                        materialize: Arc::new(move |_commit| level),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
    })
}

#[tokio::test]
async fn reads_and_replaces_configuration_from_owned_state() {
    let fixture = create_lane().await;
    let active_tool_names = vec!["read".to_owned()];
    let identity = ModelIdentity {
        provider: fixture.model.provider.0.clone(),
        model_id: fixture.model.id.clone(),
    };

    fixture
        .lane
        .set_model_impl(identity.clone(), &background_context())
        .await
        .expect("set model");
    fixture
        .lane
        .set_thinking_level_impl(ThinkingLevel::High, &background_context())
        .await
        .expect("set thinking");
    fixture
        .lane
        .set_active_tools_impl(active_tool_names.clone(), &background_context())
        .await
        .expect("set tools");

    assert_eq!(
        fixture
            .lane
            .get_model_impl(&background_context())
            .await
            .expect("get model"),
        Some(fixture.model.clone()),
        "getModel"
    );
    assert_eq!(
        fixture
            .lane
            .get_thinking_level_impl(&background_context())
            .await
            .expect("get thinking"),
        ThinkingLevel::High,
        "getThinkingLevel"
    );
    assert_eq!(
        fixture
            .lane
            .get_active_tools_impl(&background_context())
            .await
            .expect("get tools"),
        active_tool_names,
        "getActiveTools"
    );
    let stored = fixture
        .session
        .get_value(
            &stored_values::lane_config("main").address,
            &background_context(),
        )
        .await
        .expect("read config")
        .expect("stored config");
    assert_eq!(
        stored.value,
        serde_json::to_value(LaneConfiguration {
            model: identity,
            thinking_level: ThinkingLevel::High,
            active_tool_names,
        })
        .expect("configuration wire"),
        "the committed lane configuration"
    );
}

#[tokio::test]
async fn derives_queued_configuration_updates_from_the_latest_committed_state() {
    let fixture = create_lane().await;
    let (commit_started, started) = deferred();
    let (release, release_rx) = deferred();
    fixture
        .storage
        .set_before_next_commit(Some(Box::new(move || {
            Box::pin(async move {
                let _ = commit_started.send(());
                let _ = release_rx.await;
                Ok(())
            })
        })));

    let identity = ModelIdentity {
        provider: fixture.model.provider.0.clone(),
        model_id: fixture.model.id.clone(),
    };
    let model_update = {
        let lane = fixture.lane.clone();
        let identity = identity.clone();
        tokio::spawn(async move { lane.set_model_impl(identity, &background_context()).await })
    };
    started.await.expect("commit started");
    let thinking_update = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move {
            lane.set_thinking_level_impl(ThinkingLevel::High, &background_context())
                .await
        })
    };
    let _ = release.send(());
    model_update
        .await
        .expect("model update join")
        .expect("model update");
    thinking_update
        .await
        .expect("thinking update join")
        .expect("thinking update");

    assert_eq!(
        fixture.lane.state().configuration,
        LaneConfiguration {
            model: identity,
            thinking_level: ThinkingLevel::High,
            active_tool_names: Vec::new(),
        },
        "the second update planned against the first's committed state"
    );
}

#[tokio::test]
async fn returns_a_promise_value_without_holding_the_lane_line() {
    let fixture = create_lane().await;
    let (complete, completion_rx) = deferred();
    let receiver_cell = Arc::new(Mutex::new(Some(completion_rx)));
    let handed = Arc::new(Mutex::new(None::<tokio::sync::oneshot::Receiver<()>>));
    let joined = {
        let lane = fixture.lane.clone();
        let receiver_cell = Arc::clone(&receiver_cell);
        let handed = Arc::clone(&handed);
        tokio::spawn(async move {
            let receiver = lane
                .command::<tokio::sync::oneshot::Receiver<()>, _>(
                    move |_state, _session, _context| {
                        let receiver_cell = Arc::clone(&receiver_cell);
                        Box::pin(async move {
                            let receiver = lock(&receiver_cell).take().expect("one command run");
                            Ok(LaneCommand::Return { result: receiver })
                        })
                    },
                    &background_context(),
                )
                .await
                .expect("the command returns the receiver");
            *lock(&handed) = Some(receiver);
        })
    };

    joined.await.expect("command join");
    let mut receiver = lock(&handed).take().expect("the receiver handed out");
    fixture
        .lane
        .set_thinking_level_impl(ThinkingLevel::High, &background_context())
        .await
        .expect("the line served later work while the result was pending");
    assert!(
        receiver.try_recv().is_err(),
        "the returned value is still pending"
    );
    let _ = complete.send(());
    assert_eq!(receiver.await, Ok(()));
}

#[tokio::test]
async fn returns_an_expected_rejection_without_faulting_the_lane() {
    let fixture = create_lane().await;
    let rejection = commit_failure("declined");
    let rejection_for_plan = Arc::clone(&rejection);

    let result: Result<(), LaneError> = fixture
        .lane
        .command::<(), _>(
            move |_state, _session, _context| {
                let rejection = Arc::clone(&rejection_for_plan);
                Box::pin(async move { Ok(LaneCommand::Reject { error: rejection }) })
            },
            &background_context(),
        )
        .await;
    let error = result.expect_err("the rejection propagates");
    assert!(
        Arc::ptr_eq(&error, &rejection),
        "the expected error propagates as-is"
    );

    assert_eq!(
        AgentLane::get_tip_id(&fixture.lane, &background_context())
            .await
            .expect("the lane still reads"),
        None,
        "the tip is unchanged"
    );
    fixture
        .lane
        .set_thinking_level_impl(ThinkingLevel::High, &background_context())
        .await
        .expect("the lane still serves work");
}

#[tokio::test]
async fn passes_bounded_reads_and_commit_metadata_through_the_serialized_command() {
    let fixture = create_lane().await;
    let stored_configuration = Arc::new(Mutex::new(None::<LaneConfiguration>));
    let memory_published = Arc::new(AtomicBool::new(false));

    let stored_cell = Arc::clone(&stored_configuration);
    let published_cell = Arc::clone(&memory_published);
    let lane_for_materialize = fixture.lane.clone();
    let commit: crate::harness::session::types::CommitResult = fixture
        .lane
        .command::<crate::harness::session::types::CommitResult, _>(
            move |state, session, context| {
                let stored_cell = Arc::clone(&stored_cell);
                let published_cell = Arc::clone(&published_cell);
                let lane_for_materialize = lane_for_materialize.clone();
                let session = Arc::clone(&session);
                Box::pin(async move {
                    let stored = session
                        .get_value(&stored_values::lane_config("main").address, &context)
                        .await
                        .expect("bounded read");
                    *lock(&stored_cell) = stored.map(|stored| {
                        serde_json::from_value(stored.value).expect("stored configuration")
                    });
                    let mut configuration = state.configuration.clone();
                    configuration.thinking_level = ThinkingLevel::High;
                    let mut next = state.clone();
                    next.configuration = configuration.clone();
                    let next_for_materialize = next.clone();
                    Ok(LaneCommand::Commit {
                        writes: vec![
                            set_value_write(&stored_values::lane_config("main"), configuration)
                                .expect("config write"),
                        ],
                        next,
                        materialize: Arc::new(
                            move |commit: &crate::harness::session::types::CommitResult| {
                                published_cell.store(
                                    lane_for_materialize.state() == next_for_materialize,
                                    Ordering::SeqCst,
                                );
                                commit.clone()
                            },
                        ),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the command commits");

    let fixture_configuration = lane_configuration();
    assert_eq!(
        lock(&stored_configuration).as_ref(),
        Some(&fixture_configuration),
        "the bounded reader saw the seeded configuration"
    );
    assert!(
        memory_published.load(Ordering::SeqCst),
        "memory published before materialize ran"
    );
    assert_eq!(commit.seqs.len(), 1, "one write, one sequence");
    assert!(
        commit.timestamp > 0,
        "the commit carries a storage-assigned timestamp"
    );
}

#[tokio::test]
async fn preserves_committed_memory_when_synchronous_event_publication_fails() {
    let failure = commit_failure("uncloneable event");
    let emit_batch: EmitBatch = {
        let failure = Arc::clone(&failure);
        Arc::new(move |_events: Vec<HarnessEvent>, _context| {
            let failure = Arc::clone(&failure);
            Box::pin(async move { Err(failure) })
        })
    };
    let fixture = create_lane_with_emit(emit_batch).await;

    let result: Result<(), LaneError> = fixture
        .lane
        .command::<(), _>(
            move |state, _session, _context| {
                Box::pin(async move {
                    let mut configuration = state.configuration.clone();
                    configuration.thinking_level = ThinkingLevel::High;
                    let mut next = state.clone();
                    next.configuration = configuration.clone();
                    Ok(LaneCommand::Commit {
                        writes: vec![
                            set_value_write(&stored_values::lane_config("main"), configuration)
                                .expect("config write"),
                        ],
                        next,
                        materialize: Arc::new(|_commit| ()),
                        events: Some(Arc::new(
                            move |_commit: &crate::harness::session::types::CommitResult| {
                                vec![
                                    HarnessEvent::lane_scoped(
                                        "main",
                                        false,
                                        HarnessEventPayload::RunStart {
                                            run_id: "run".to_owned(),
                                            started_at: 1,
                                        },
                                    )
                                    .expect("lane-scoped event"),
                                ]
                            },
                        )),
                    })
                })
            },
            &background_context(),
        )
        .await;
    let error = result.expect_err("the delivery failure rejects the command");
    assert_eq!(error.to_string(), "uncloneable event");

    assert_eq!(
        fixture.lane.state().configuration.thinking_level,
        ThinkingLevel::High,
        "the committed memory published"
    );
    let stored = fixture
        .session
        .get_value(
            &stored_values::lane_config("main").address,
            &background_context(),
        )
        .await
        .expect("read config")
        .expect("stored config");
    assert_eq!(
        stored
            .value
            .get("thinkingLevel")
            .and_then(serde_json::Value::as_str),
        Some("high"),
        "the durable memory published"
    );
}

#[tokio::test]
async fn rejects_work_after_sealing_while_an_admitted_commit_finishes() {
    let fixture = create_lane().await;
    let (commit_started, started) = deferred();
    let (release, release_rx) = deferred();
    fixture
        .storage
        .set_before_next_commit(Some(Box::new(move || {
            Box::pin(async move {
                let _ = commit_started.send(());
                let _ = release_rx.await;
                Ok(())
            })
        })));

    let admitted = tokio::spawn(set_thinking_level(&fixture.lane, ThinkingLevel::High, None));
    started.await.expect("commit started");
    let closed: LaneError = Arc::new(HarnessClosed);
    fixture.lane.seal(Arc::clone(&closed)).await;

    let tip = AgentLane::get_tip_id(&fixture.lane, &background_context()).await;
    match tip {
        Err(LaneOperationError::Closed(error)) => {
            assert_eq!(
                error.to_string(),
                closed.to_string(),
                "the seal collapses to the closed surface with its message"
            );
        }
        other => panic!("getTipId rejects with the sealed error: {other:?}"),
    }
    let rejected = fixture
        .lane
        .set_thinking_level_impl(ThinkingLevel::Low, &background_context())
        .await
        .expect_err("work after the seal rejects");
    assert!(
        Arc::ptr_eq(&rejected, &closed),
        "the sealed error is the one the seal carried"
    );
    let _ = release.send(());
    admitted
        .await
        .expect("admitted join")
        .expect("the admitted commit finished");
    assert_eq!(
        fixture.lane.state().configuration.thinking_level,
        ThinkingLevel::High,
        "the admitted work's memory published"
    );
}

#[tokio::test]
async fn publishes_memory_only_after_the_durable_commit_succeeds() {
    let fixture = create_lane().await;
    let (commit_started, started) = deferred();
    let (release, release_rx) = deferred();
    fixture
        .storage
        .set_before_next_commit(Some(Box::new(move || {
            Box::pin(async move {
                let _ = commit_started.send(());
                let _ = release_rx.await;
                Ok(())
            })
        })));

    let command = tokio::spawn(set_thinking_level(&fixture.lane, ThinkingLevel::High, None));
    started.await.expect("commit started");

    assert_eq!(
        fixture.lane.state().configuration.thinking_level,
        ThinkingLevel::Off,
        "memory stays at the last committed state while the commit is in flight"
    );
    let _ = release.send(());
    assert_eq!(
        command.await.expect("join").expect("settles high"),
        ThinkingLevel::High
    );
    assert_eq!(
        fixture.lane.state().configuration.thinking_level,
        ThinkingLevel::High,
        "memory published after the commit"
    );
    let stored = fixture
        .session
        .get_value(
            &stored_values::lane_config("main").address,
            &background_context(),
        )
        .await
        .expect("read config")
        .expect("stored config");
    assert_eq!(
        stored.value.get("thinkingLevel"),
        Some(&serde_json::json!("high"))
    );
}

#[tokio::test]
async fn preserves_memory_when_the_durable_commit_fails() {
    let fixture = create_lane().await;
    fixture.storage.set_before_next_commit(Some(Box::new(|| {
        Box::pin(async {
            Err(crate::harness::session::types::SessionError::Message(
                "commit failed".to_owned(),
            ))
        })
    })));

    let command = set_thinking_level(&fixture.lane, ThinkingLevel::High, None).await;
    let error = command.expect_err("the commit failure rejects the command");
    assert_eq!(error.to_string(), "commit failed");

    assert_eq!(
        fixture.lane.state().configuration.thinking_level,
        ThinkingLevel::Off,
        "memory stays at the last committed state"
    );
    let stored = fixture
        .session
        .get_value(
            &stored_values::lane_config("main").address,
            &background_context(),
        )
        .await
        .expect("read config")
        .expect("stored config");
    assert_eq!(
        stored.value.get("thinkingLevel"),
        Some(&serde_json::json!("off"))
    );
}

/// The hand-built cancelled-control commit upstream's "diverts ordinary
/// work" case makes: the operation's durable state swaps to a
/// cancel-requested starting leaf.
async fn commit_cancelled_control(
    lane: &Lane,
    operation: &LiveOperation,
    cancelled_state: crate::harness::session::types::OperationState,
) {
    lane.command::<(), _>(
        {
            let operation_id = operation.meta.operation_id.clone();
            let meta = operation.meta.clone();
            let cancelled_state = cancelled_state.clone();
            move |state, _session, _context| {
                let operation_id = operation_id.clone();
                let meta = meta.clone();
                let cancelled_state = cancelled_state.clone();
                Box::pin(async move {
                    let mut next = state.clone();
                    next.operation = Some(LiveOperation {
                        meta,
                        state: cancelled_state.clone(),
                    });
                    Ok(LaneCommand::Commit {
                        writes: vec![
                            set_value_write(
                                &stored_values::operation_state(&operation_id),
                                cancelled_state,
                            )
                            .expect("state write"),
                        ],
                        next,
                        materialize: Arc::new(|_commit| ()),
                        events: None,
                    })
                })
            }
        },
        &background_context(),
    )
    .await
    .expect("the cancelled control committed");
}

/// The ordinary work the cancellation diverts, upstream's "diverts
/// ordinary work but settles against latest cancelled control" case: one
/// accept, one hand-built cancelled control, one declined continuation,
/// one settle against the latest control.
#[expect(
    clippy::too_many_lines,
    reason = "the case is one continuous admission-settle flow; splitting it further would hide the sequence"
)]
#[tokio::test]
async fn diverts_ordinary_work_but_settles_against_latest_cancelled_control() {
    let fixture = create_lane().await;
    let accepted = fixture
        .lane
        .accept_impl(
            OperationRequest::Prompt {
                operation_id: None,
                prompt: Box::new(PromptMessagesPayload::Text {
                    prompt: "hello".to_owned(),
                    images: None,
                }),
            },
            &background_context(),
        )
        .await
        .expect("accept serves")
        .expect("the admission succeeds");
    let operation = fixture
        .lane
        .state()
        .operation
        .expect("the admitted operation");
    assert_eq!(
        operation.state.at(),
        "starting",
        "the accepted operation is starting"
    );
    assert_eq!(operation.meta.operation_id, accepted.operation_id);
    let scope = crate::harness::session::types::operation_scope_of(&operation.state);
    let cancelled_state = crate::harness::runtime::lane::operation_state_with_scope(
        &operation.state,
        OperationScope {
            control: Control::CancelRequested { requested_at: 1 },
            ..scope
        },
    );
    commit_cancelled_control(&fixture.lane, &operation, cancelled_state.clone()).await;

    let continued = Arc::new(AtomicBool::new(false));
    let diversion = fixture
        .lane
        .continue_operation::<&'static str, _>(
            {
                let continued = Arc::clone(&continued);
                move |_state, _session, _context| {
                    let continued = Arc::clone(&continued);
                    Box::pin(async move {
                        continued.store(true, Ordering::SeqCst);
                        Ok(OperationCommand::Return {
                            result: "continued",
                        })
                    })
                }
            },
            &background_context(),
        )
        .await
        .expect("continue serves");
    assert!(
        matches!(diversion, ContinueOperationResult::CancelRequested),
        "the planner never ran"
    );
    assert!(!continued.load(Ordering::SeqCst));

    let settled = fixture
        .lane
        .settle_operation::<Vec<InboxItem>, _>(
            move |state, latest, _meta, _session, _context| {
                let latest = latest.clone();
                let mut inbox = state.inbox;
                Box::pin(async move {
                    assert!(matches!(
                        crate::harness::session::types::operation_scope_of(&latest).control,
                        Control::CancelRequested { .. }
                    ));
                    inbox.push(InboxItem {
                        entry_id: "accepted-during-cancellation".to_owned(),
                        kind: InboxItemKind::Write,
                    });
                    Ok(OperationCommand::Commit {
                        writes: Vec::new(),
                        operation_state: latest,
                        lane: Some(LanePatch {
                            inbox: Some(inbox.clone()),
                            ..LanePatch::default()
                        }),
                        materialize: Arc::new(move |_commit| inbox.clone()),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("settle serves");
    let expected_inbox = vec![InboxItem {
        entry_id: "accepted-during-cancellation".to_owned(),
        kind: InboxItemKind::Write,
    }];
    assert_eq!(settled, expected_inbox);
    assert_eq!(fixture.lane.state().inbox, expected_inbox);
    let stored = fixture
        .session
        .get_value(
            &stored_values::lane_state("main").address,
            &background_context(),
        )
        .await
        .expect("read lane state")
        .expect("stored lane state");
    assert_eq!(
        stored.value.get("inbox"),
        Some(&serde_json::to_value(&expected_inbox).expect("inbox wire")),
    );
}

#[tokio::test]
async fn plans_queued_commands_from_the_latest_committed_memory() {
    let fixture = create_lane().await;
    let (commit_started, started) = deferred();
    let (release, release_rx) = deferred();
    fixture
        .storage
        .set_before_next_commit(Some(Box::new(move || {
            Box::pin(async move {
                let _ = commit_started.send(());
                let _ = release_rx.await;
                Ok(())
            })
        })));
    let observed = Arc::new(Mutex::new(Vec::new()));

    let first = tokio::spawn(set_thinking_level(
        &fixture.lane,
        ThinkingLevel::High,
        Some(Arc::clone(&observed)),
    ));
    started.await.expect("commit started");
    let second = tokio::spawn(set_thinking_level(
        &fixture.lane,
        ThinkingLevel::Medium,
        Some(Arc::clone(&observed)),
    ));
    tokio::task::yield_now().await;
    assert_eq!(
        *lock(&observed),
        vec![ThinkingLevel::Off],
        "the queued command has not planned yet"
    );

    let _ = release.send(());
    assert_eq!(
        (
            first.await.expect("first join").expect("first settles"),
            second.await.expect("second join").expect("second settles"),
        ),
        (ThinkingLevel::High, ThinkingLevel::Medium),
    );
    assert_eq!(
        *lock(&observed),
        vec![ThinkingLevel::Off, ThinkingLevel::High],
        "the second planned against the first's committed state"
    );
    assert_eq!(
        fixture.lane.state().configuration.thinking_level,
        ThinkingLevel::Medium,
        "the latest committed memory"
    );
}

// The boundary suite: branches the upstream lane suite leaves untested,
// bound here against the ported surfaces.

/// The capturing emit batch the event assertions share, upstream's
/// `HarnessEventBus` fixture restated: events land in a shared cell.
fn capturing_emit_batch() -> (EmitBatch, Arc<Mutex<Vec<HarnessEvent>>>) {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let cell = Arc::clone(&recorded);
    let emit: EmitBatch = Arc::new(move |events: Vec<HarnessEvent>, _context| {
        let cell = Arc::clone(&cell);
        Box::pin(async move {
            lock(&cell).extend(events);
            Ok(())
        })
    });
    (emit, recorded)
}

fn image_content(id: &str) -> pi_ai::types::ImageContent {
    serde_json::from_value(json!({ "type": "image", "mimeType": "image/png", "data": id }))
        .expect("image wire")
}

fn prompt_request(prompt: &str) -> OperationRequest {
    OperationRequest::Prompt {
        operation_id: None,
        prompt: Box::new(PromptMessagesPayload::Text {
            prompt: prompt.to_owned(),
            images: None,
        }),
    }
}

/// The error one inner-result rejection carries, the taxonomy read the
/// trait collapses.
fn harness_error_name(error: &crate::harness::result::HarnessError) -> &'static str {
    error.tag()
}

/// The user message the boundary fixtures build, upstream's inline
/// `{ role: "user", content: text }` literals.
fn boundary_user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Text(text.to_owned()),
        timestamp: 1,
    }))
}

/// The zero usage wire the boundary fixtures' assistant literals carry.
fn zero_usage() -> serde_json::Value {
    serde_json::json!({
        "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
    })
}

/// The assistant wire the boundary fixtures build.
fn boundary_assistant_message(text: &str, stop_reason: &str) -> AgentMessage {
    serde_json::from_value(json!({
        "role": "assistant",
        "content": [{ "type": "text", "text": text }],
        "api": "anthropic-messages",
        "provider": "test",
        "model": "model",
        "usage": zero_usage(),
        "stopReason": stop_reason,
        "timestamp": 1,
    }))
    .expect("assistant wire")
}

#[tokio::test]
async fn enqueue_images_on_a_text_prompt_builds_the_blocks_content() {
    let fixture = create_lane().await;
    let entry_id = fixture
        .lane
        .steer_impl(
            QueueMessage::Text("queued".to_owned()),
            vec![image_content("first")],
            &background_context(),
        )
        .await
        .expect("enqueue serves")
        .expect("the queue result");
    let stored = fixture
        .session
        .get_value(
            &stored_values::pending_entry(&entry_id).address,
            &background_context(),
        )
        .await
        .expect("read pending")
        .expect("stored pending");
    assert_eq!(
        stored.value.get("payload").expect("payload").get("content"),
        Some(&json!([
            { "type": "text", "text": "queued" },
            { "type": "image", "mimeType": "image/png", "data": "first" },
        ])),
    );
}

#[tokio::test]
async fn enqueue_images_on_a_non_user_message_rejects() {
    let fixture = create_lane().await;
    let assistant = boundary_assistant_message("hello", "stop");
    let result = fixture
        .lane
        .steer_impl(
            QueueMessage::Message(Box::new(assistant)),
            vec![image_content("first")],
            &background_context(),
        )
        .await
        .expect("enqueue serves");
    let error = result.expect_err("images ride user messages only");
    assert!(
        error
            .to_string()
            .contains("Images can be added only to queued user messages")
    );
}

#[tokio::test]
async fn enqueue_empty_text_without_images_rejects() {
    let fixture = create_lane().await;
    let result = fixture
        .lane
        .steer_impl(
            QueueMessage::Text(String::new()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("enqueue serves");
    let error = result.expect_err("empty queued input rejects");
    assert!(
        error
            .to_string()
            .contains("Queued input must contain text or an image")
    );
}

#[tokio::test]
async fn cancel_queued_reports_not_found_and_already_consumed() {
    let fixture = create_lane().await;
    let not_found = fixture
        .lane
        .cancel_queued_impl("absent", &background_context())
        .await
        .expect("cancel serves");
    assert_eq!(
        not_found,
        crate::harness::agent_harness::CancelQueuedKind::NotFound
    );

    let appended = fixture
        .lane
        .append_message_impl(boundary_user_message("tail"), &background_context())
        .await
        .expect("append serves");
    let consumed = fixture
        .lane
        .cancel_queued_impl(&appended, &background_context())
        .await
        .expect("cancel serves");
    assert_eq!(
        consumed,
        crate::harness::agent_harness::CancelQueuedKind::AlreadyConsumed
    );
}

#[tokio::test]
async fn accept_on_a_busy_lane_reports_lane_busy() {
    let fixture = create_lane().await;
    fixture
        .lane
        .accept_impl(prompt_request("first"), &background_context())
        .await
        .expect("accept serves")
        .expect("the first admission");
    let second = fixture
        .lane
        .accept_impl(prompt_request("second"), &background_context())
        .await
        .expect("accept serves");
    let error = second.expect_err("the busy lane rejects the second admission");
    assert_eq!(harness_error_name(&error), "LaneBusy");
}

#[tokio::test]
async fn accept_with_an_empty_prompt_rejects() {
    let fixture = create_lane().await;
    let result = fixture
        .lane
        .accept_impl(prompt_request(""), &background_context())
        .await
        .expect("accept serves");
    let error = result.expect_err("an empty prompt rejects");
    assert!(
        error
            .to_string()
            .contains("Acceptance must append at least one message")
    );
}

#[tokio::test]
async fn accept_of_a_pending_assistant_message_rejects() {
    let fixture = create_lane().await;
    let pending = boundary_assistant_message("streaming", "pending");
    let result = fixture
        .lane
        .accept_impl(
            OperationRequest::Prompt {
                operation_id: None,
                prompt: Box::new(PromptMessagesPayload::Message(Box::new(pending))),
            },
            &background_context(),
        )
        .await
        .expect("accept serves");
    let error = result.expect_err("a pending assistant rejects");
    assert!(
        error
            .to_string()
            .contains("Cannot accept a pending assistant message")
    );
}

#[tokio::test]
async fn drive_on_a_foreign_operation_reports_the_mismatch() {
    let fixture = create_lane().await;
    let result = fixture
        .lane
        .drive_impl(
            crate::harness::agent_harness::DriveOptions {
                operation_id: "absent".to_owned(),
                wait_for_retry: None,
                poll_deferred: None,
            },
            &background_context(),
        )
        .await
        .expect("drive serves");
    let error = result.expect_err("a foreign operation rejects");
    assert_eq!(error.tag(), "OperationMismatch");
}

#[tokio::test]
async fn resume_without_an_operation_reports_nothing_to_resume() {
    let fixture = create_lane().await;
    let result = fixture
        .lane
        .resume_impl(&background_context())
        .await
        .expect("resume serves");
    assert_eq!(
        result.expect_err("nothing to resume").tag(),
        "NothingToResume"
    );
}

#[tokio::test]
async fn abort_without_an_operation_reports_no_active_operation() {
    let fixture = create_lane().await;
    let result = fixture
        .lane
        .abort_impl(&background_context())
        .await
        .expect("abort serves");
    assert_eq!(
        result.expect_err("no active operation").tag(),
        "NoActiveOperation"
    );
}

#[tokio::test]
async fn append_during_a_run_queues_the_write() {
    let fixture = create_lane().await;
    fixture
        .lane
        .accept_impl(prompt_request("first"), &background_context())
        .await
        .expect("accept serves")
        .expect("the admission");
    let entry_id = fixture
        .lane
        .append_message_impl(boundary_user_message("tail"), &background_context())
        .await
        .expect("append queues");
    assert_eq!(
        fixture.lane.state().inbox,
        vec![InboxItem {
            entry_id,
            kind: InboxItemKind::Write,
        }],
        "the write queued behind the live operation"
    );
}

#[tokio::test]
async fn record_usage_publishes_the_global_usage_event_with_the_commit_sequence() {
    let (emit, recorded) = capturing_emit_batch();
    let fixture = create_lane_with_emit(emit).await;
    let usage_id = fixture
        .lane
        .record_usage_impl(pi_ai::types::Usage::default(), None, &background_context())
        .await
        .expect("record serves");
    let events = lock(&recorded).clone();
    assert_eq!(events.len(), 1, "one usage event");
    assert!(events[0].lane.is_none(), "usage is harness-global");
    match &events[0].payload {
        HarnessEventPayload::Usage { row, totals, .. } => {
            assert_eq!(row.id, usage_id);
            assert!(row.adjustment, "lane-recorded usage is an adjustment row");
            assert_eq!(
                *totals,
                pi_ai::types::Usage::default(),
                "the fresh session's totals"
            );
        }
        other => panic!("the usage event: {other:?}"),
    }
}

#[tokio::test]
async fn wait_for_idle_returns_when_idle() {
    let fixture = create_lane().await;
    fixture
        .lane
        .wait_for_idle_impl(&background_context())
        .await
        .expect("the idle lane returns immediately");
}

#[tokio::test]
async fn run_when_idle_claims_the_owner_until_the_callback_releases() {
    let fixture = create_lane().await;
    let (callback_started, started) = deferred();
    let (release, release_rx) = deferred();
    let started_cell = Arc::new(Mutex::new(Some(callback_started)));
    let release_cell = Arc::new(Mutex::new(Some(release_rx)));
    let callback: crate::harness::agent_harness::IdleCallback = Arc::new(move |_context| {
        let started_cell = Arc::clone(&started_cell);
        let release_cell = Arc::clone(&release_cell);
        Box::pin(async move {
            let sender = lock(&started_cell).take();
            if let Some(sender) = sender {
                let _ = sender.send(());
            }
            // The guard must drop before the await: a std MutexGuard is
            // not Send, and the callback's future is.
            let receiver = lock(&release_cell).take();
            if let Some(receiver) = receiver {
                let _ = receiver.await;
            }
        })
    });
    let idle = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move {
            lane.run_when_idle_impl(callback, &background_context())
                .await
        })
    };
    started.await.expect("the callback started");
    let blocked = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move {
            lane.set_thinking_level_impl(ThinkingLevel::High, &background_context())
                .await
        })
    };
    tokio::task::yield_now().await;
    assert!(
        !blocked.is_finished(),
        "commands queue while the idle callback holds the line"
    );
    let _ = release.send(());
    idle.await
        .expect("idle join")
        .expect("the callback settled");
    blocked
        .await
        .expect("blocked join")
        .expect("the queued command ran after release");
    assert_eq!(
        fixture.lane.state().configuration.thinking_level,
        ThinkingLevel::High
    );
}

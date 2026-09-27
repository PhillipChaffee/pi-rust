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
    lane_seeded(emit_batch, Vec::new()).await
}

/// The fixture variant the corrupt-inbox and queued-write tests drive: the
/// extra writes commit with the seed, before the lane's restore reads.
async fn lane_seeded(emit_batch: EmitBatch, extra: Vec<stored_values::Write>) -> LaneFixture {
    let configuration = lane_configuration();
    let storage = Arc::new(ControlledStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = Arc::new(runtime_session(next_session_id(), storage.clone()));
    let mut writes = vec![
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
    writes.extend(extra);
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
        .set_model(identity.clone(), &background_context())
        .await
        .expect("set model");
    fixture
        .lane
        .set_thinking_level(ThinkingLevel::High, &background_context())
        .await
        .expect("set thinking");
    fixture
        .lane
        .set_active_tools(active_tool_names.clone(), &background_context())
        .await
        .expect("set tools");

    assert_eq!(
        fixture
            .lane
            .get_model(&background_context())
            .await
            .expect("get model"),
        Some(fixture.model.clone()),
        "getModel"
    );
    assert_eq!(
        fixture
            .lane
            .get_thinking_level(&background_context())
            .await
            .expect("get thinking"),
        ThinkingLevel::High,
        "getThinkingLevel"
    );
    assert_eq!(
        fixture
            .lane
            .get_active_tools(&background_context())
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
        tokio::spawn(async move { lane.set_model(identity, &background_context()).await })
    };
    started.await.expect("commit started");
    let thinking_update = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move {
            lane.set_thinking_level(ThinkingLevel::High, &background_context())
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
        .set_thinking_level(ThinkingLevel::High, &background_context())
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
        .set_thinking_level(ThinkingLevel::High, &background_context())
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
        .accept(
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

/// The closed surface's message, the trait's error read.
fn closed_message(error: LaneOperationError) -> String {
    match error {
        LaneOperationError::Closed(reason) => reason.to_string(),
    }
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
/// The queued text message enqueue builds: the content is blocks with the
/// text first, upstream's `{ type: "user", content: [...] }` construction.
fn steer_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Blocks(vec![pi_ai::types::UserBlock::Text(
            pi_ai::types::TextContent {
                text: text.to_owned(),
                text_signature: None,
            },
        )]),
        timestamp: 1,
    }))
}

/// The message's content wire, the shape the queued/committed comparisons
/// pin (the timestamps come from the lane's clock).
fn content_wire(message: &AgentMessage) -> serde_json::Value {
    match message {
        AgentMessage::Standard(Message::User(user)) => {
            serde_json::to_value(&user.content).expect("content wire")
        }
        other => panic!("the user message: {other:?}"),
    }
}

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
        .steer(
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
        .steer(
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
        .steer(
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
        .cancel_queued("absent", &background_context())
        .await
        .expect("cancel serves");
    assert_eq!(
        not_found,
        crate::harness::agent_harness::CancelQueuedKind::NotFound
    );

    let appended = fixture
        .lane
        .append_message(boundary_user_message("tail"), &background_context())
        .await
        .expect("append serves");
    let consumed = fixture
        .lane
        .cancel_queued(&appended, &background_context())
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
        .accept(prompt_request("first"), &background_context())
        .await
        .expect("accept serves")
        .expect("the first admission");
    let second = fixture
        .lane
        .accept(prompt_request("second"), &background_context())
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
        .accept(prompt_request(""), &background_context())
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
        .accept(
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
        .drive(
            DriveOptions {
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
        .resume(&background_context())
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
        .abort(&background_context())
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
        .accept(prompt_request("first"), &background_context())
        .await
        .expect("accept serves")
        .expect("the admission");
    let entry_id = fixture
        .lane
        .append_message(boundary_user_message("tail"), &background_context())
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
        .record_usage(pi_ai::types::Usage::default(), None, &background_context())
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
        .wait_for_idle(&background_context())
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
        tokio::spawn(async move { lane.run_when_idle(callback, &background_context()).await })
    };
    started.await.expect("the callback started");
    let blocked = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move {
            lane.set_thinking_level(ThinkingLevel::High, &background_context())
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

// The runtime-surface additions: the flows the ported suite binds to the
// landed trait surface, minus the drive child's procedure loop.

use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::DriveOutcome;
use crate::harness::runtime::types::Drive;
use crate::harness::session::types::OperationKind;
use crate::harness::session::types::OperationResultRecord;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::TerminalStatus;

fn settled_record(operation_id: &str, tip_id: Option<&str>) -> OperationResultRecord {
    OperationResultRecord {
        operation_id: operation_id.to_owned(),
        kind: OperationKind::Run,
        status: TerminalStatus::Completed,
        error: None,
        from_tip_id: None,
        tip_id: tip_id.map(str::to_owned),
        started_at: 1,
        ended_at: 2,
    }
}

async fn commit_writes(fixture: &LaneFixture, writes: Vec<stored_values::Write>) {
    let session = Arc::clone(&fixture.session);
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
        .expect("the commit settles");
}

async fn accepted_operation(fixture: &LaneFixture) -> String {
    fixture
        .lane
        .accept(prompt_request("first"), &background_context())
        .await
        .expect("accept serves")
        .expect("the admission")
        .operation_id
}

fn event_types(events: &[HarnessEvent]) -> Vec<&'static str> {
    events
        .iter()
        .map(|event| event.payload.event_type().as_str())
        .collect()
}

#[tokio::test]
async fn accept_captures_a_queued_steer_into_the_run() {
    let (emit, recorded) = capturing_emit_batch();
    let fixture = create_lane_with_emit(emit).await;
    let steered = fixture
        .lane
        .steer(
            QueueMessage::Text("steer".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("steer serves")
        .expect("the steer");

    fixture
        .lane
        .accept(prompt_request("first"), &background_context())
        .await
        .expect("accept serves")
        .expect("the admission");

    assert!(
        fixture.lane.state().inbox.is_empty(),
        "the steer rode the admission",
    );
    let entries = AgentLane::find_entries(&fixture.lane, None, &background_context())
        .await
        .expect("the scan");
    assert_eq!(entries.len(), 2, "the steer entry and the prompt committed");
    assert_eq!(
        entries[1].id(),
        steered,
        "the steer entry kept its pending id"
    );
    match &entries[1] {
        crate::harness::session::types::Entry::Message { body, .. } => {
            assert_eq!(
                content_wire(&body.message),
                content_wire(&steer_message("steer"))
            );
        }
        other => panic!("the steer entry: {other:?}"),
    }
    assert_eq!(
        fixture.lane.state().tip_id.as_deref(),
        Some(entries[0].id()),
        "the tip settled on the prompt",
    );
    assert_eq!(
        event_types(&lock(&recorded)),
        vec![
            "queue_update",
            "run_start",
            "message_start",
            "message_end",
            "entry_added",
            "message_start",
            "message_end",
            "entry_added",
            "queue_update",
        ],
        "the steer published its queue, the admission published the run, the committed entries, and the emptied queues",
    );
}

#[tokio::test]
async fn drive_returns_the_settled_record_of_a_finished_operation() {
    let fixture = create_lane().await;
    commit_writes(
        &fixture,
        vec![
            set_value_write(
                &stored_values::operation_result("settled"),
                settled_record("settled", Some("tip")),
            )
            .expect("result write"),
            set_value_write(
                &stored_values::lane_state("main"),
                crate::harness::session::types::LaneState {
                    current_operation_id: None,
                    last_operation_id: Some("settled".to_owned()),
                    inbox: Vec::new(),
                },
            )
            .expect("state write"),
        ],
    )
    .await;

    let driven = fixture
        .lane
        .drive(
            DriveOptions {
                operation_id: "settled".to_owned(),
                wait_for_retry: None,
                poll_deferred: None,
            },
            &background_context(),
        )
        .await
        .expect("drive serves")
        .expect("the settled claim");
    assert_eq!(
        driven,
        DriveOutcome::Settled {
            outcome: settled_record("settled", Some("tip")),
        },
        "the durable record settles the drive",
    );
}

#[tokio::test]
async fn drive_joins_a_settled_active_pass() {
    let fixture = create_lane().await;
    let operation_id = accepted_operation(&fixture).await;
    let joined = operation_id.clone();
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: operation_id.clone(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    drive.settle(DriveOutcome::Settled {
        outcome: settled_record(&operation_id, None),
    });
    fixture.lane.set_active_drive(Some(Arc::clone(&drive)));

    let driven = fixture
        .lane
        .drive(
            DriveOptions {
                operation_id,
                wait_for_retry: None,
                poll_deferred: None,
            },
            &background_context(),
        )
        .await
        .expect("drive serves")
        .expect("the joined pass");
    assert_eq!(
        driven,
        DriveOutcome::Settled {
            outcome: settled_record(&joined, None),
        },
        "the joined pass settled into its outcome",
    );
}

#[tokio::test]
async fn drive_joins_a_failed_active_pass() {
    let fixture = create_lane().await;
    let operation_id = accepted_operation(&fixture).await;
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: operation_id.clone(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    let failure = commit_failure("boom");
    drive.fail(Arc::clone(&failure));
    fixture.lane.set_active_drive(Some(drive));

    let error = fixture
        .lane
        .drive(
            DriveOptions {
                operation_id,
                wait_for_retry: None,
                poll_deferred: None,
            },
            &background_context(),
        )
        .await
        .expect_err("the failed pass rejects the join");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "boom",
                "the pass's failure carries through"
            );
        }
    }
}

#[tokio::test]
async fn drive_reports_an_occupied_line() {
    let fixture = create_lane().await;
    let operation_id = accepted_operation(&fixture).await;
    let rival = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: "rival".to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    let failure = commit_failure("rival failed");
    rival.fail(Arc::clone(&failure));
    fixture.lane.set_active_drive(Some(rival));

    let error = fixture
        .lane
        .drive(
            DriveOptions {
                operation_id,
                wait_for_retry: None,
                poll_deferred: None,
            },
            &background_context(),
        )
        .await
        .expect_err("the occupied line rejects");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "rival failed",
                "the rival's failure carries"
            );
        }
    }
}

#[tokio::test]
async fn drive_reports_the_context_abort_during_the_pass() {
    let fixture = create_lane().await;
    let operation_id = accepted_operation(&fixture).await;
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: operation_id.clone(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    fixture.lane.set_active_drive(Some(Arc::clone(&drive)));

    let (_, controller) = pi_chord::context::with_cancel(&background_context());
    let context = crate::harness::context::with_abort_signal(
        controller.signal().clone(),
        &background_context(),
    );
    let waiting = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move {
            lane.drive(
                DriveOptions {
                    operation_id,
                    wait_for_retry: None,
                    poll_deferred: None,
                },
                &context,
            )
            .await
        })
    };
    tokio::task::yield_now().await;
    controller.abort_without_reason();
    let error = waiting
        .await
        .expect("drive join")
        .expect_err("the aborted drive rejects");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "The operation was aborted",
                "the context abort carries the standard reason"
            );
        }
    }
}

#[tokio::test]
async fn request_abort_marks_the_operation_cancel_requested() {
    let (emit, recorded) = capturing_emit_batch();
    let fixture = create_lane_with_emit(emit).await;
    let operation_id = accepted_operation(&fixture).await;

    let outcome = fixture
        .lane
        .request_abort(&operation_id, &background_context())
        .await
        .expect("abort serves")
        .expect("the request");
    assert_eq!(outcome.operation_id, operation_id);
    assert!(outcome.newly_requested, "the first request is new");
    assert!(outcome.steer.is_empty() && outcome.follow_up.is_empty());

    let state = fixture.lane.state().operation.expect("the operation").state;
    assert!(
        matches!(
            &state,
            crate::harness::session::types::OperationState::Starting(leaf)
                if matches!(leaf.scope.control, Control::CancelRequested { .. })
        ),
        "the durable state flipped to cancel_requested"
    );
    let stored = fixture
        .session
        .get_value(
            &stored_values::operation_state(&operation_id).address,
            &background_context(),
        )
        .await
        .expect("read state")
        .expect("stored state");
    assert!(
        stored.value.to_string().contains("cancel_requested"),
        "the durable state flipped to cancel_requested",
    );
    assert_eq!(
        event_types(&lock(&recorded)),
        vec![
            "run_start",
            "message_start",
            "message_end",
            "entry_added",
            "operation_abort",
        ],
        "the abort published its event after the admission's",
    );
}

#[tokio::test]
async fn request_abort_returns_the_queued_steer_and_follow_up() {
    let (emit, recorded) = capturing_emit_batch();
    let fixture = create_lane_with_emit(emit).await;
    let operation_id = accepted_operation(&fixture).await;
    let steer = fixture
        .lane
        .steer(
            QueueMessage::Text("steer".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("steer serves")
        .expect("the steer");
    let follow_up = fixture
        .lane
        .follow_up(
            QueueMessage::Text("follow".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("followUp serves")
        .expect("the follow-up");

    let outcome = fixture
        .lane
        .request_abort(&operation_id, &background_context())
        .await
        .expect("abort serves")
        .expect("the request");
    assert!(outcome.newly_requested);
    assert_eq!(
        outcome.steer.iter().map(content_wire).collect::<Vec<_>>(),
        vec![content_wire(&steer_message("steer"))],
    );
    assert_eq!(
        outcome
            .follow_up
            .iter()
            .map(content_wire)
            .collect::<Vec<_>>(),
        vec![content_wire(&steer_message("follow"))],
    );
    assert!(
        fixture.lane.state().inbox.is_empty(),
        "the queued items left with the abort",
    );
    assert_eq!(
        event_types(&lock(&recorded)),
        vec![
            "run_start",
            "message_start",
            "message_end",
            "entry_added",
            "queue_update",
            "queue_update",
            "operation_abort",
            "queue_update",
        ],
        "the abort published the emptied queues",
    );
    let _ = (steer, follow_up);
}

#[tokio::test]
async fn request_abort_is_idempotent_once_requested() {
    let fixture = create_lane().await;
    let operation_id = accepted_operation(&fixture).await;
    fixture
        .lane
        .request_abort(&operation_id, &background_context())
        .await
        .expect("abort serves")
        .expect("the first request");
    let outcome = fixture
        .lane
        .request_abort(&operation_id, &background_context())
        .await
        .expect("abort serves")
        .expect("the second request");
    assert!(
        !outcome.newly_requested,
        "the second request settles the already-cancelled operation",
    );
    assert!(outcome.steer.is_empty() && outcome.follow_up.is_empty());
}

#[tokio::test]
async fn request_abort_reports_the_mismatch() {
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;
    let error = fixture
        .lane
        .request_abort("foreign", &background_context())
        .await
        .expect("abort serves")
        .expect_err("the foreign id rejects");
    assert_eq!(error.tag(), "OperationMismatch");
}

#[tokio::test]
async fn request_abort_signals_the_installed_gate() {
    let fixture = create_lane().await;
    let operation_id = accepted_operation(&fixture).await;
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: operation_id.clone(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    fixture.lane.set_active_drive(Some(Arc::clone(&drive)));

    fixture
        .lane
        .request_abort(&operation_id, &background_context())
        .await
        .expect("abort serves")
        .expect("the request");
    assert!(
        drive.gate.signal().aborted(),
        "the request signalled the installed pass's gate",
    );
    assert!(
        drive.gate.admit(|| ()).is_err(),
        "the aborting gate refuses further admission"
    );
}

#[tokio::test]
async fn inspect_execution_reports_the_lane_view() {
    let fixture = create_lane().await;
    let identity = ModelIdentity {
        provider: fixture.model.provider.0.clone(),
        model_id: fixture.model.id.clone(),
    };
    fixture
        .lane
        .set_model(identity.clone(), &background_context())
        .await
        .expect("set model");

    let idle = fixture
        .lane
        .inspect_execution(&background_context())
        .await
        .expect("inspect serves");
    assert_eq!(idle.lane, "main");
    assert_eq!(idle.tip_id, None);
    assert_eq!(idle.configured_model, identity);
    assert_eq!(idle.current, None);
    assert_eq!(idle.last_operation_id, None);

    let operation_id = accepted_operation(&fixture).await;
    let running = fixture
        .lane
        .inspect_execution(&background_context())
        .await
        .expect("inspect serves");
    let current = running.current.as_ref().expect("the current operation");
    assert_eq!(current.id, operation_id);
    assert_eq!(current.kind, OperationKind::Run);
    assert_eq!(
        current.status,
        crate::harness::agent_harness::OperationStatus::Open
    );
    assert_eq!(
        current.captured_model, None,
        "the starting leaf captures no model"
    );

    fixture
        .lane
        .request_abort(&operation_id, &background_context())
        .await
        .expect("abort serves")
        .expect("the request");
    let aborting = fixture
        .lane
        .inspect_execution(&background_context())
        .await
        .expect("inspect serves");
    assert_eq!(
        aborting
            .current
            .as_ref()
            .expect("the current operation")
            .status,
        crate::harness::agent_harness::OperationStatus::Aborting,
        "the cancelled operation reports aborting",
    );
}

#[tokio::test]
async fn prompt_text_faults_at_the_staged_drive_seam() {
    let fixture = create_lane().await;
    let error = fixture
        .lane
        .prompt_text("hello", None, &background_context())
        .await
        .expect_err("the staged drive seam faults the run");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "drive operation is not implemented until its later AgentHarness slice",
                "the staged drive seam faults the accepted run",
            );
        }
    }
}

#[tokio::test]
async fn prompt_messages_accepts_prebuilt_payloads() {
    let fixture = create_lane().await;
    let single = fixture
        .lane
        .prompt_messages(
            PromptMessagesPayload::Message(Box::new(boundary_user_message("hello"))),
            &background_context(),
        )
        .await;
    assert!(
        single.is_err(),
        "the prebuilt message runs to the staged drive seam",
    );
    // The faulted run's operation stays admitted; the next prompt is busy.
    let busy = fixture
        .lane
        .prompt_messages(
            PromptMessagesPayload::Messages(vec![
                boundary_user_message("first"),
                boundary_user_message("second"),
            ]),
            &background_context(),
        )
        .await
        .expect("prompt serves");
    assert_eq!(
        busy.expect_err("the busy lane rejects").tag(),
        "LaneBusy",
        "the message list runs against the still-admitted operation",
    );
}

#[tokio::test]
async fn prompt_text_builds_blocks_from_images() {
    let fixture = create_lane().await;
    let error = fixture
        .lane
        .prompt_text(
            "hello",
            Some(vec![image_content("first")]),
            &background_context(),
        )
        .await;
    assert!(
        error.is_err(),
        "the image prompt runs to the staged drive seam"
    );
    let entries = AgentLane::find_entries(&fixture.lane, None, &background_context())
        .await
        .expect("the scan");
    match &entries[0] {
        crate::harness::session::types::Entry::Message { body, .. } => match &body.message {
            AgentMessage::Standard(Message::User(user)) => match &user.content {
                pi_ai::types::UserContent::Blocks(blocks) => {
                    assert_eq!(blocks.len(), 2, "the text and the image ride one message");
                    assert!(matches!(blocks[0], pi_ai::types::UserBlock::Text(_)));
                    assert!(matches!(blocks[1], pi_ai::types::UserBlock::Image(_)));
                }
                other @ pi_ai::types::UserContent::Text(_) => panic!("the text content: {other:?}"),
            },
            other => panic!("the user message: {other:?}"),
        },
        other => panic!("the entry: {other:?}"),
    }
}

#[tokio::test]
async fn skill_reports_unknown_skills() {
    let fixture = create_lane().await;
    let result = fixture
        .lane
        .skill("nope", None, &background_context())
        .await
        .expect("skill serves");
    let error = result.expect_err("the unknown skill rejects");
    assert_eq!(error.tag(), "UnknownSkill");
}

#[tokio::test]
async fn skill_formats_a_known_invocation() {
    let mut config = runtime_config();
    config.resources.skills = vec![crate::harness::types::Skill {
        name: "greet".to_owned(),
        description: "says hello".to_owned(),
        content: "Hello!".to_owned(),
        file_path: "/skills/greet/SKILL.md".to_owned(),
        disable_model_invocation: None,
    }];
    let configuration = lane_configuration();
    let storage = Arc::new(ControlledStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = Arc::new(runtime_session(next_session_id(), storage));
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
    let restored = restore_lane(session.clone(), "main", &background_context())
        .await
        .expect("restore");
    let lane = Lane::new(
        "main",
        session,
        Arc::new(create_models(None)),
        Arc::new(HookRegistry::new(noop_hook_reporter())),
        restored,
        passthrough_fault_handler(),
        noop_emit_batch(),
        recording_watch_installer(empty_lane_snapshot("main", &configuration)),
        Arc::new(move || config.clone()),
    );
    let result = lane
        .skill("greet", Some("be kind".to_owned()), &background_context())
        .await;
    assert!(
        result.is_err(),
        "the known skill runs to the staged drive seam"
    );
    let entries = AgentLane::find_entries(&lane, None, &background_context())
        .await
        .expect("the scan");
    match &entries[0] {
        crate::harness::session::types::Entry::Message { body, .. } => match &body.message {
            AgentMessage::Standard(Message::User(user)) => match &user.content {
                pi_ai::types::UserContent::Text(text) => {
                    assert!(
                        text.contains("Hello!") && text.contains("be kind"),
                        "the invocation carries the skill and the instructions: {text}",
                    );
                }
                other @ pi_ai::types::UserContent::Blocks(_) => {
                    panic!("the blocks content: {other:?}")
                }
            },
            other => panic!("the user message: {other:?}"),
        },
        other => panic!("the entry: {other:?}"),
    }
}

#[tokio::test]
async fn prompt_from_template_reports_unknown_templates() {
    let fixture = create_lane().await;
    let result = fixture
        .lane
        .prompt_from_template("nope", None, &background_context())
        .await
        .expect("template serves");
    let error = result.expect_err("the unknown template rejects");
    assert_eq!(error.tag(), "UnknownTemplate");
}

#[tokio::test]
async fn prompt_from_template_formats_known_templates() {
    let mut config = runtime_config();
    config.resources.prompt_templates = vec![crate::harness::types::PromptTemplate {
        name: "greet".to_owned(),
        description: None,
        content: "Args: $1?".to_owned(),
    }];
    let configuration = lane_configuration();
    let storage = Arc::new(ControlledStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = Arc::new(runtime_session(next_session_id(), storage));
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
    let restored = restore_lane(session.clone(), "main", &background_context())
        .await
        .expect("restore");
    let lane = Lane::new(
        "main",
        session,
        Arc::new(create_models(None)),
        Arc::new(HookRegistry::new(noop_hook_reporter())),
        restored,
        passthrough_fault_handler(),
        noop_emit_batch(),
        recording_watch_installer(empty_lane_snapshot("main", &configuration)),
        Arc::new(move || config.clone()),
    );
    let result = lane
        .prompt_from_template(
            "greet",
            Some(vec!["world".to_owned()]),
            &background_context(),
        )
        .await;
    assert!(
        result.is_err(),
        "the known template runs to the staged drive seam"
    );
    let entries = AgentLane::find_entries(&lane, None, &background_context())
        .await
        .expect("the scan");
    match &entries[0] {
        crate::harness::session::types::Entry::Message { body, .. } => match &body.message {
            AgentMessage::Standard(Message::User(user)) => match &user.content {
                pi_ai::types::UserContent::Text(text) => {
                    assert_eq!(
                        text, "Args: world?",
                        "the invocation carries the formatted args: {text}",
                    );
                }
                other @ pi_ai::types::UserContent::Blocks(_) => {
                    panic!("the blocks content: {other:?}")
                }
            },
            other => panic!("the user message: {other:?}"),
        },
        other => panic!("the entry: {other:?}"),
    }
}

#[tokio::test]
async fn resume_drives_the_active_operation_to_the_staged_seam() {
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;
    let error = fixture
        .lane
        .resume(&background_context())
        .await
        .expect_err("the staged drive seam faults the resume");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "drive operation is not implemented until its later AgentHarness slice",
                "the staged drive seam faults the resume",
            );
        }
    }
}

#[tokio::test]
async fn abort_requests_cancellation_then_faults_at_the_staged_seam() {
    let fixture = create_lane().await;
    let _operation_id = accepted_operation(&fixture).await;
    let error = fixture
        .lane
        .abort(&background_context())
        .await
        .expect_err("the staged drive seam faults the abort");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "drive operation is not implemented until its later AgentHarness slice",
                "the staged drive seam faults the cancelled drive",
            );
        }
    }
    let state = fixture.lane.state().operation.expect("the operation").state;
    assert!(
        matches!(
            &state,
            crate::harness::session::types::OperationState::Starting(leaf)
                if matches!(leaf.scope.control, Control::CancelRequested { .. })
        ),
        "the durable state flipped to cancel_requested"
    );
}

#[tokio::test]
async fn steer_queues_a_stopped_assistant_message() {
    let fixture = create_lane().await;
    let assistant = boundary_assistant_message("hello", "stop");
    let entry_id = fixture
        .lane
        .steer(
            QueueMessage::Message(Box::new(assistant.clone())),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("steer serves")
        .expect("the steer");
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
        stored.value.get("payload").expect("payload"),
        &serde_json::to_value(&assistant).expect("assistant wire"),
        "the queued assistant message stored whole",
    );
}

#[tokio::test]
async fn steer_rejects_a_pending_assistant_message() {
    let fixture = create_lane().await;
    let pending = boundary_assistant_message("streaming", "pending");
    let result = fixture
        .lane
        .steer(
            QueueMessage::Message(Box::new(pending)),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("steer serves");
    let error = result.expect_err("the pending assistant rejects");
    assert!(
        error
            .to_string()
            .contains("Cannot queue a pending assistant message"),
    );
}

#[tokio::test]
async fn steer_merges_images_into_a_prebuilt_user_message() {
    let fixture = create_lane().await;
    let user = boundary_user_message("hello");
    let entry_id = fixture
        .lane
        .steer(
            QueueMessage::Message(Box::new(user)),
            vec![image_content("first")],
            &background_context(),
        )
        .await
        .expect("steer serves")
        .expect("the steer");
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
            { "type": "text", "text": "hello" },
            { "type": "image", "mimeType": "image/png", "data": "first" },
        ])),
        "the images merged into the user message's blocks",
    );
}

#[tokio::test]
async fn steer_after_seal_reports_closed() {
    let fixture = create_lane().await;
    fixture.lane.seal(Arc::new(HarnessClosed)).await;
    let error = fixture
        .lane
        .steer(
            QueueMessage::Text("late".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("steer serves")
        .expect_err("the sealed lane rejects the steer");
    assert_eq!(
        error.tag(),
        "Closed",
        "the sealed taxonomy carries the closed error"
    );
}

#[tokio::test]
async fn cancel_queued_after_seal_reports_closed() {
    let fixture = create_lane().await;
    fixture.lane.seal(Arc::new(HarnessClosed)).await;
    let error = fixture
        .lane
        .cancel_queued("absent", &background_context())
        .await
        .expect_err("the sealed lane rejects the cancel");
    assert_eq!(error.tag(), "Closed");
}

#[tokio::test]
async fn record_usage_after_seal_reports_closed() {
    let fixture = create_lane().await;
    fixture.lane.seal(Arc::new(HarnessClosed)).await;
    let error = fixture
        .lane
        .record_usage(pi_ai::types::Usage::default(), None, &background_context())
        .await
        .expect_err("the sealed lane rejects the usage");
    assert_eq!(error.tag(), "Closed");
}

#[tokio::test]
async fn cancel_queued_reports_the_missing_payload_invariant() {
    let fixture = create_lane().await;
    let entry_id = fixture
        .lane
        .steer(
            QueueMessage::Text("queued".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("steer serves")
        .expect("the steer");
    let deleted = vec![stored_values::delete_value_write(
        &stored_values::pending_entry(&entry_id),
    )];
    commit_writes(&fixture, deleted).await;

    let error = fixture
        .lane
        .cancel_queued(&entry_id, &background_context())
        .await
        .expect_err("the missing payload faults");
    assert!(
        error.to_string().contains(&format!(
            "Queued steer entry {entry_id} is missing its payload"
        )),
        "the missing payload names the queue and the entry: {error}",
    );
}

#[tokio::test]
async fn append_custom_entry_commits_the_custom_shape() {
    let fixture = create_lane().await;
    let entry_id = fixture
        .lane
        .append_custom_entry("note", Some(json!({ "k": "v" })), &background_context())
        .await
        .expect("append serves");
    let entries = AgentLane::find_entries(&fixture.lane, None, &background_context())
        .await
        .expect("the scan");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].id(), entry_id);
    match &entries[0] {
        crate::harness::session::types::Entry::Custom { body, .. } => {
            assert_eq!(body.custom_type, "note");
            assert_eq!(body.data.as_ref(), Some(&json!({ "k": "v" })));
        }
        other => panic!("the custom entry: {other:?}"),
    }
    assert_eq!(
        fixture.lane.state().tip_id.as_deref(),
        Some(entry_id.as_str()),
        "the custom entry moved the tip",
    );
}

#[tokio::test]
async fn find_entries_scan_from_the_tip_and_get_result_reads_records() {
    let fixture = create_lane().await;
    let empty = AgentLane::find_entries(&fixture.lane, None, &background_context())
        .await
        .expect("the scan");
    assert!(
        empty.is_empty(),
        "the tip-less lane scans to an empty branch path",
    );
    assert_eq!(
        AgentLane::find_entry(&fixture.lane, None, &background_context())
            .await
            .expect("the read"),
        None,
        "the tip-less lane reads no entry",
    );

    let appended = fixture
        .lane
        .append_message(boundary_user_message("tail"), &background_context())
        .await
        .expect("append serves");
    let entries = AgentLane::find_entries(&fixture.lane, None, &background_context())
        .await
        .expect("the scan");
    assert_eq!(entries.len(), 1, "the appended entry is the branch path");
    let first = AgentLane::find_entry(&fixture.lane, None, &background_context())
        .await
        .expect("the read");
    assert_eq!(
        first
            .as_ref()
            .map(crate::harness::session::types::Entry::id),
        Some(appended.as_str())
    );

    assert_eq!(
        AgentLane::get_result(&fixture.lane, "absent", &background_context())
            .await
            .expect("the read"),
        None,
        "the missing record reads none",
    );
}

#[tokio::test]
async fn wait_for_idle_wakes_on_state_changes_until_the_lane_idles() {
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;
    let waiting = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move { lane.wait_for_idle(&background_context()).await })
    };
    tokio::task::yield_now().await;
    assert!(
        !waiting.is_finished(),
        "the active operation blocks the wait"
    );

    // A state change wakes the waiter, which re-observes and waits again.
    fixture
        .lane
        .set_thinking_level(ThinkingLevel::High, &background_context())
        .await
        .expect("set thinking");
    tokio::task::yield_now().await;
    assert!(
        !waiting.is_finished(),
        "the operation still blocks the wait"
    );

    // Finishing the operation idles the lane; the wait returns.
    finish_operation(&fixture).await;
    waiting
        .await
        .expect("wait join")
        .expect("the finished operation idles the lane");
    assert!(
        fixture.lane.state().last_operation_id.is_some(),
        "the finished operation recorded",
    );
}

/// Finishes the admitted operation through `settleOperation`, the lane's
/// own finish path: the result record writes and the operation clears.
async fn finish_operation(fixture: &LaneFixture) {
    let operation_id = fixture
        .lane
        .state()
        .operation
        .as_ref()
        .map(|operation| operation.meta.operation_id.clone())
        .expect("the active operation");
    let record = settled_record(&operation_id, None);
    fixture
        .lane
        .settle_operation::<(), _>(
            move |_state, _operation_state, _meta, _session, _context| {
                let record = record.clone();
                Box::pin(async move {
                    Ok(OperationCommand::Finish {
                        writes: Vec::new(),
                        record,
                        lane: None,
                        materialize: Arc::new(|_commit| ()),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the finish settles");
}

#[tokio::test]
async fn wait_for_idle_aborts_with_the_context_reason() {
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;
    let (_, controller) = pi_chord::context::with_cancel(&background_context());
    let context = crate::harness::context::with_abort_signal(
        controller.signal().clone(),
        &background_context(),
    );
    let waiting = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move { lane.wait_for_idle(&context).await })
    };
    tokio::task::yield_now().await;
    controller.abort_without_reason();
    let error = waiting
        .await
        .expect("wait join")
        .expect_err("the aborted wait rejects");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "The operation was aborted",
                "the reason-less abort carries the standard abort error",
            );
        }
    }

    // A caller-supplied abort reason rides the wait's rejection.
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;
    let (_, controller) = pi_chord::context::with_cancel(&background_context());
    let context = crate::harness::context::with_abort_signal(
        controller.signal().clone(),
        &background_context(),
    );
    let waiting = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move { lane.wait_for_idle(&context).await })
    };
    tokio::task::yield_now().await;
    controller.abort("stop");
    let error = waiting
        .await
        .expect("wait join")
        .expect_err("the aborted wait rejects");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(reason.to_string(), "stop", "the caller's reason carries");
        }
    }
}

#[tokio::test]
async fn run_when_idle_waits_for_the_operation_to_finish() {
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;
    let callback: crate::harness::agent_harness::IdleCallback =
        Arc::new(|_context| Box::pin(std::future::ready(())));
    let waiting = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move { lane.run_when_idle(callback, &background_context()).await })
    };
    tokio::task::yield_now().await;
    assert!(
        !waiting.is_finished(),
        "the active operation blocks the claim"
    );

    finish_operation(&fixture).await;
    waiting
        .await
        .expect("claim join")
        .expect("the finished operation frees the claim");
    assert_eq!(
        fixture.lane.state().configuration.thinking_level,
        ThinkingLevel::Off,
        "the callback ran after the claim"
    );
}

#[tokio::test]
async fn run_when_idle_aborts_while_waiting() {
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;
    let (_, controller) = pi_chord::context::with_cancel(&background_context());
    let context = crate::harness::context::with_abort_signal(
        controller.signal().clone(),
        &background_context(),
    );
    let callback: crate::harness::agent_harness::IdleCallback =
        Arc::new(|_context| Box::pin(std::future::ready(())));
    let waiting = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move { lane.run_when_idle(callback, &context).await })
    };
    tokio::task::yield_now().await;
    controller.abort("stop");
    let error = waiting
        .await
        .expect("claim join")
        .expect_err("the aborted claim rejects");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(reason.to_string(), "stop", "the caller's reason carries");
        }
    }
}

#[tokio::test]
async fn commands_abort_while_waiting_for_the_idle_owner() {
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
        tokio::spawn(async move { lane.run_when_idle(callback, &background_context()).await })
    };
    started.await.expect("the callback claimed the idle slot");

    let (_, controller) = pi_chord::context::with_cancel(&background_context());
    let context = crate::harness::context::with_abort_signal(
        controller.signal().clone(),
        &background_context(),
    );
    let blocked = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move { lane.set_thinking_level(ThinkingLevel::High, &context).await })
    };
    tokio::task::yield_now().await;
    assert!(!blocked.is_finished(), "the idle owner blocks the command");
    controller.abort_without_reason();
    let error = blocked
        .await
        .expect("command join")
        .expect_err("the aborted command rejects");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "The operation was aborted",
                "the aborted wait carries the standard abort error",
            );
        }
    }
    let _ = release.send(());
    idle.await
        .expect("idle join")
        .expect("the callback settles");
}

#[tokio::test]
async fn compact_and_navigate_raise_the_staged_seams_through_the_trait() {
    let fixture = create_lane().await;
    let error = fixture
        .lane
        .compact(None, &background_context())
        .await
        .expect_err("the staged compaction seam faults the compact");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "compaction is not implemented until its later AgentHarness slice",
            );
        }
    }

    let error = fixture
        .lane
        .navigate_tree(Some("target".to_owned()), None, &background_context())
        .await
        .expect("navigate serves")
        .expect_err("the unknown target rejects");
    assert_eq!(error.tag(), "UnknownTarget");
}

#[tokio::test]
async fn navigation_admits_against_a_known_target_and_validates_its_shapes() {
    let fixture = create_lane().await;
    let first = fixture
        .lane
        .append_message(boundary_user_message("root"), &background_context())
        .await
        .expect("append serves");
    let second = fixture
        .lane
        .append_message(boundary_user_message("head"), &background_context())
        .await
        .expect("append serves");

    // The current tip rejects.
    let error = fixture
        .lane
        .navigate_tree(Some(second.clone()), None, &background_context())
        .await
        .expect("navigate serves");
    assert!(
        error
            .expect_err("the current tip rejects")
            .to_string()
            .contains("Navigation target must differ from the current tip"),
    );

    // A root navigation cannot set a label.
    let error = fixture
        .lane
        .navigate_tree(
            None,
            Some(crate::harness::agent_harness::NavigateOptions {
                summarize: None,
                label: Some("label".to_owned()),
                custom_instructions: None,
            }),
            &background_context(),
        )
        .await
        .expect("navigate serves");
    assert!(
        error
            .expect_err("the labeled root rejects")
            .to_string()
            .contains("Root navigation cannot set a label"),
    );

    // An unknown target rejects.
    let error = fixture
        .lane
        .navigate_tree(Some("absent".to_owned()), None, &background_context())
        .await
        .expect("navigate serves");
    assert_eq!(
        error.expect_err("the unknown target rejects").tag(),
        "UnknownTarget",
    );

    // A known non-tip target admits; the operation starts and the staged
    // drive seam faults the navigation.
    let error = fixture
        .lane
        .navigate_tree(Some(first.clone()), None, &background_context())
        .await
        .expect_err("the staged drive seam faults the navigation");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "drive operation is not implemented until its later AgentHarness slice",
            );
        }
    }
    let operation = fixture.lane.state().operation.expect("the navigation");
    match &operation.state {
        crate::harness::session::types::OperationState::NavigationReadyToCommit(leaf) => {
            assert_eq!(leaf.target_id.as_deref(), Some(first.as_str()));
            assert_eq!(leaf.label, None);
        }
        other => panic!("the navigation state: {other:?}"),
    }
}

#[tokio::test]
async fn sealed_lanes_reject_their_surfaces_with_the_collapsed_closed_error() {
    let fixture = create_lane().await;
    let closed: LaneError = Arc::new(HarnessClosed);
    fixture.lane.seal(Arc::clone(&closed)).await;

    // accept reports the sealed message inside the admission result.
    let accepted = fixture
        .lane
        .accept(prompt_request("late"), &background_context())
        .await
        .expect("accept serves");
    assert_eq!(
        accepted
            .expect_err("the sealed lane rejects the admission")
            .tag(),
        "Closed",
    );

    // drive, resume, abort, and request_abort report the sealed error
    // inside their result, the closed checks the sealed-lane returns early.
    for rejection in [
        fixture
            .lane
            .drive(
                DriveOptions {
                    operation_id: "any".to_owned(),
                    wait_for_retry: None,
                    poll_deferred: None,
                },
                &background_context(),
            )
            .await
            .expect("drive serves")
            .expect_err("the sealed drive rejects"),
        fixture
            .lane
            .resume(&background_context())
            .await
            .expect("resume serves")
            .expect_err("the sealed resume rejects"),
        fixture
            .lane
            .abort(&background_context())
            .await
            .expect("abort serves")
            .expect_err("the sealed abort rejects"),
        fixture
            .lane
            .request_abort("any", &background_context())
            .await
            .expect("abort serves")
            .expect_err("the sealed request rejects"),
    ] {
        assert_eq!(
            rejection.tag(),
            "Closed",
            "the sealed surface collapses to Closed"
        );
    }
}

#[tokio::test]
async fn settle_operation_requires_an_active_operation() {
    let fixture = create_lane().await;
    let error = fixture
        .lane
        .settle_operation::<(), _>(
            |_state, _operation_state, _meta, _session, _context| {
                Box::pin(async move { Ok(OperationCommand::Return { result: () }) })
            },
            &background_context(),
        )
        .await
        .expect_err("the idle lane rejects the settle");
    assert!(
        error
            .to_string()
            .contains("settleOperation requires an active operation"),
    );
}

#[tokio::test]
async fn continue_operation_commits_and_finishes_through_the_wrappers() {
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;

    let committed = fixture
        .lane
        .continue_operation::<String, _>(
            |state, _session, _context| {
                Box::pin(async move {
                    let configuration = state.configuration.clone();
                    Ok(OperationCommand::Commit {
                        writes: Vec::new(),
                        operation_state: crate::harness::session::types::OperationState::Checkpoint(
                            crate::harness::session::types::CheckpointOperation {
                                scope: OperationScope {
                                    control: Control::Running,
                                    settings: crate::harness::session::types::RunSettings {
                                        compaction:
                                            crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS,
                                        steering_mode: crate::types::QueueMode::All,
                                        follow_up_mode: crate::types::QueueMode::All,
                                        tool_execution: crate::types::ToolExecutionMode::Parallel,
                                    },
                                    latest_assistant_entry_id: None,
                                },
                                checkpoint: crate::harness::session::types::CheckpointData {
                                    continuation:
                                        crate::harness::session::types::Continuation::NeedAssistant {
                                            overflow_recovery_used: false,
                                        },
                                    trigger_entry_id: "trigger".to_owned(),
                                },
                            },
                        ),
                        lane: Some(LanePatch {
                            tip_id: Some(None),
                            configuration: Some(configuration),
                            inbox: None,
                        }),
                        materialize: Arc::new(|_commit| "committed".to_owned()),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("continue serves");
    match committed {
        ContinueOperationResult::Result { value } => assert_eq!(value, "committed"),
        other @ ContinueOperationResult::CancelRequested => {
            panic!("the continuation: {other:?}")
        }
    }

    let finished = fixture
        .lane
        .continue_operation::<&'static str, _>(
            |_state, _session, _context| {
                Box::pin(async move {
                    Ok(OperationCommand::Finish {
                        writes: Vec::new(),
                        record: settled_record("the-active", None),
                        lane: None,
                        materialize: Arc::new(|_commit| "finished"),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("continue serves");
    match finished {
        ContinueOperationResult::Result { value } => assert_eq!(value, "finished"),
        other @ ContinueOperationResult::CancelRequested => {
            panic!("the continuation: {other:?}")
        }
    }
    assert_eq!(
        fixture.lane.state().operation,
        None,
        "the finish cleared the operation"
    );
    assert!(
        fixture.lane.state().last_operation_id.is_some(),
        "the finish recorded the operation",
    );
}

#[tokio::test]
async fn wait_for_idle_joins_the_installed_drive_pass() {
    // The pass settles while the wait holds it; the owner then clears the
    // pass, the real procedure's post-settlement cleanup, and the wait idles.
    let fixture = create_lane().await;
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: "drive".to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    fixture.lane.set_active_drive(Some(Arc::clone(&drive)));
    let waiting = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move { lane.wait_for_idle(&background_context()).await })
    };
    tokio::task::yield_now().await;
    drive.settle(DriveOutcome::Settled {
        outcome: settled_record("drive", None),
    });
    fixture.lane.set_active_drive(None);
    waiting
        .await
        .expect("wait join")
        .expect("the settled pass freed the wait");

    // A failed pass frees the wait the same way.
    let failed = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: "drive".to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    fixture.lane.set_active_drive(Some(Arc::clone(&failed)));
    let waiting = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move { lane.wait_for_idle(&background_context()).await })
    };
    tokio::task::yield_now().await;
    failed.fail(commit_failure("boom"));
    fixture.lane.set_active_drive(None);
    waiting
        .await
        .expect("wait join")
        .expect("the failed pass freed the wait");
}

#[tokio::test]
async fn run_when_idle_joins_the_installed_drive_pass() {
    let fixture = create_lane().await;
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: "drive".to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    fixture.lane.set_active_drive(Some(Arc::clone(&drive)));
    let callback: crate::harness::agent_harness::IdleCallback =
        Arc::new(|_context| Box::pin(std::future::ready(())));
    let waiting = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move { lane.run_when_idle(callback, &background_context()).await })
    };
    tokio::task::yield_now().await;
    drive.settle(DriveOutcome::Settled {
        outcome: settled_record("drive", None),
    });
    fixture.lane.set_active_drive(None);
    waiting
        .await
        .expect("claim join")
        .expect("the settled pass freed the claim");
}

#[tokio::test]
async fn get_result_reports_the_malformed_record() {
    let fixture = create_lane().await;
    commit_writes(
        &fixture,
        vec![raw_set(
            &stored_values::operation_result("broken").address,
            json!(42),
        )],
    )
    .await;
    let error = AgentLane::get_result(&fixture.lane, "broken", &background_context())
        .await
        .expect_err("the malformed record faults");
    match error {
        LaneOperationError::Closed(reason) => {
            assert!(
                reason.to_string().contains("Operation result is malformed"),
                "the malformed record names its parse failure: {reason}",
            );
        }
    }
}

fn raw_set(
    address: &stored_values::ValueAddress,
    value: serde_json::Value,
) -> stored_values::Write {
    stored_values::Write::ValueSet(stored_values::ValueSetWrite {
        kind: "value".to_owned(),
        op: "set".to_owned(),
        namespace: address.namespace.clone(),
        key: address.key.clone(),
        value,
    })
}

/// The corrupt-inbox variants the admission's capture loop rejects, the
/// pending payloads seeded raw under the item ids.
async fn corrupt_inbox_lane(corruption: Corruption) -> LaneFixture {
    let write = match corruption {
        Corruption::Missing => {
            stored_values::delete_value_write(&stored_values::pending_entry("corrupt-steer"))
        }
        Corruption::Malformed => raw_set(
            &stored_values::pending_entry("corrupt-steer").address,
            json!(42),
        ),
        Corruption::NotAMessage => set_value_write(
            &stored_values::pending_entry("corrupt-steer"),
            PendingEntry::Custom {
                custom_type: "note".to_owned(),
                payload: None,
            },
        )
        .expect("custom write"),
        Corruption::PendingAssistant => set_value_write(
            &stored_values::pending_entry("corrupt-steer"),
            PendingEntry::Message {
                payload: Box::new(boundary_assistant_message("streaming", "pending")),
            },
        )
        .expect("pending write"),
    };
    lane_seeded(
        noop_emit_batch(),
        vec![
            write,
            set_value_write(
                &stored_values::lane_state("main"),
                crate::harness::session::types::LaneState {
                    current_operation_id: None,
                    last_operation_id: None,
                    inbox: vec![InboxItem {
                        entry_id: "corrupt-steer".to_owned(),
                        kind: InboxItemKind::Steer,
                    }],
                },
            )
            .expect("state write"),
        ],
    )
    .await
}

enum Corruption {
    Missing,
    Malformed,
    NotAMessage,
    PendingAssistant,
}

#[tokio::test]
async fn accept_faults_on_the_corrupt_inbox_invariants() {
    for (corruption, expected) in [
        (
            Corruption::Missing,
            "Pending steer entry corrupt-steer is missing its payload",
        ),
        (Corruption::Malformed, "Pending entry payload is malformed"),
        (
            Corruption::NotAMessage,
            "Pending steer entry corrupt-steer is not a message",
        ),
        (
            Corruption::PendingAssistant,
            "Pending steer entry corrupt-steer contains a pending assistant",
        ),
    ] {
        let fixture = corrupt_inbox_lane(corruption).await;
        let message = closed_message(
            fixture
                .lane
                .accept(prompt_request("first"), &background_context())
                .await
                .expect_err("the corrupt inbox faults the admission"),
        );
        assert!(
            message.contains(expected),
            "the corrupt inbox names its invariant: {message}",
        );
    }
}

/// Pushes the corrupt steer item into the live inbox after the operation
/// admitted, the abort fixtures' corruption seed.
async fn seed_corrupt_inbox(fixture: &LaneFixture, corruption: Corruption) {
    let write = match corruption {
        Corruption::Missing => None,
        Corruption::Malformed => Some(raw_set(
            &stored_values::pending_entry("corrupt-steer").address,
            json!(42),
        )),
        Corruption::NotAMessage => Some(
            set_value_write(
                &stored_values::pending_entry("corrupt-steer"),
                PendingEntry::Custom {
                    custom_type: "note".to_owned(),
                    payload: None,
                },
            )
            .expect("custom write"),
        ),
        Corruption::PendingAssistant => Some(
            set_value_write(
                &stored_values::pending_entry("corrupt-steer"),
                PendingEntry::Message {
                    payload: Box::new(boundary_assistant_message("streaming", "pending")),
                },
            )
            .expect("pending write"),
        ),
    };
    let mut writes = write.map_or_else(Vec::new, |write| vec![write]);
    writes.push(
        set_value_write(
            &stored_values::lane_state("main"),
            crate::harness::session::types::LaneState {
                current_operation_id: Some("live".to_owned()),
                last_operation_id: None,
                inbox: vec![InboxItem {
                    entry_id: "corrupt-steer".to_owned(),
                    kind: InboxItemKind::Steer,
                }],
            },
        )
        .expect("state write"),
    );
    fixture
        .lane
        .command::<(), _>(
            move |state, _session, _context| {
                let writes = writes.clone();
                Box::pin(async move {
                    let mut next = state.clone();
                    next.inbox = vec![InboxItem {
                        entry_id: "corrupt-steer".to_owned(),
                        kind: InboxItemKind::Steer,
                    }];
                    Ok(LaneCommand::Commit {
                        writes,
                        next,
                        materialize: Arc::new(|_commit| ()),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the corruption commits");
}

#[tokio::test]
async fn abort_faults_on_the_corrupt_inbox_invariants() {
    for corruption in [
        Corruption::Malformed,
        Corruption::NotAMessage,
        Corruption::Missing,
    ] {
        let fixture = create_lane().await;
        accepted_operation(&fixture).await;
        seed_corrupt_inbox(&fixture, corruption).await;
        let operation_id = fixture
            .lane
            .state()
            .operation
            .as_ref()
            .map(|operation| operation.meta.operation_id.clone())
            .expect("the operation");
        let message = closed_message(
            fixture
                .lane
                .request_abort(&operation_id, &background_context())
                .await
                .expect_err("the corrupt inbox faults the abort"),
        );
        assert!(
            message.contains("missing its message") || message.contains("malformed"),
            "the corrupt inbox faults the abort: {message}",
        );
    }
}

#[tokio::test]
async fn append_rejects_a_pending_assistant_message() {
    let fixture = create_lane().await;
    let message = closed_message(
        fixture
            .lane
            .append_message(
                boundary_assistant_message("streaming", "pending"),
                &background_context(),
            )
            .await
            .expect_err("the pending assistant rejects"),
    );
    assert_eq!(
        message, "Cannot persist a pending assistant message",
        "the append rejects the pending assistant",
    );
}

/// The idle-lane fixture with one queued write seeded, the append-capture
/// fixture's seed.
async fn seeded_write_lane() -> LaneFixture {
    lane_seeded(
        noop_emit_batch(),
        vec![
            set_value_write(
                &stored_values::pending_entry("queued-write"),
                PendingEntry::Custom {
                    custom_type: "note".to_owned(),
                    payload: Some(json!({ "k": "v" })),
                },
            )
            .expect("pending write"),
            set_value_write(
                &stored_values::lane_state("main"),
                crate::harness::session::types::LaneState {
                    current_operation_id: None,
                    last_operation_id: None,
                    inbox: vec![InboxItem {
                        entry_id: "queued-write".to_owned(),
                        kind: InboxItemKind::Write,
                    }],
                },
            )
            .expect("state write"),
        ],
    )
    .await
}

#[tokio::test]
async fn append_captures_the_queued_writes_when_idle() {
    let fixture = seeded_write_lane().await;
    let appended = fixture
        .lane
        .append_message(boundary_user_message("tail"), &background_context())
        .await
        .expect("append serves");

    assert!(
        fixture.lane.state().inbox.is_empty(),
        "the queued write rode the append"
    );
    let entries = AgentLane::find_entries(&fixture.lane, None, &background_context())
        .await
        .expect("the scan");
    assert_eq!(
        entries.len(),
        2,
        "the queued write and the appended message committed"
    );
    match &entries[1] {
        crate::harness::session::types::Entry::Custom { body, id, .. } => {
            assert_eq!(id, "queued-write");
            assert_eq!(body.custom_type, "note");
            assert_eq!(body.data.as_ref(), Some(&json!({ "k": "v" })));
        }
        other => panic!("the queued write entry: {other:?}"),
    }
    assert_eq!(
        fixture.lane.state().tip_id.as_deref(),
        Some(appended.as_str()),
        "the append moved the tip",
    );
    let stored = fixture
        .session
        .get_value(
            &stored_values::pending_entry("queued-write").address,
            &background_context(),
        )
        .await
        .expect("read pending");
    assert!(stored.is_none(), "the queued write's payload deleted");
}

#[tokio::test]
async fn append_faults_on_the_corrupt_write_inbox() {
    for (name, write) in [
        (
            "missing",
            stored_values::delete_value_write(&stored_values::pending_entry("corrupt-write")),
        ),
        (
            "malformed",
            raw_set(
                &stored_values::pending_entry("corrupt-write").address,
                json!(42),
            ),
        ),
    ] {
        let fixture = lane_seeded(
            noop_emit_batch(),
            vec![
                write,
                set_value_write(
                    &stored_values::lane_state("main"),
                    crate::harness::session::types::LaneState {
                        current_operation_id: None,
                        last_operation_id: None,
                        inbox: vec![InboxItem {
                            entry_id: "corrupt-write".to_owned(),
                            kind: InboxItemKind::Write,
                        }],
                    },
                )
                .expect("state write"),
            ],
        )
        .await;
        let message = closed_message(
            fixture
                .lane
                .append_message(boundary_user_message("tail"), &background_context())
                .await
                .expect_err("the corrupt write inbox faults the append"),
        );
        assert!(
            message.contains("missing its payload") || message.contains("malformed"),
            "the corrupt {name} write inbox names its invariant: {message}",
        );
    }
}

#[tokio::test]
async fn append_during_a_run_queues_the_custom_shape() {
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;
    let entry_id = fixture
        .lane
        .append_custom_entry("note", Some(json!({ "k": "v" })), &background_context())
        .await
        .expect("append queues");
    match &fixture.lane.state().inbox[..] {
        [item] => {
            assert_eq!(item.entry_id, entry_id);
            assert_eq!(item.kind, InboxItemKind::Write);
        }
        other => panic!("the inbox: {other:?}"),
    }
}

#[tokio::test]
async fn follow_up_names_its_queue_in_the_rejections() {
    let fixture = create_lane().await;
    let error = fixture
        .lane
        .follow_up(
            QueueMessage::Message(Box::new(boundary_assistant_message("hello", "stop"))),
            vec![image_content("first")],
            &background_context(),
        )
        .await
        .expect("followUp serves")
        .expect_err("the images ride user messages only");
    assert!(
        error
            .to_string()
            .contains("Images can be added only to queued user messages"),
        "the follow-up rejection rides the shared guard",
    );

    let empty_text_user = AgentMessage::Standard(Message::User(pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Text(String::new()),
        timestamp: 1,
    }));
    let entry_id = fixture
        .lane
        .follow_up(
            QueueMessage::Message(Box::new(empty_text_user)),
            vec![image_content("first")],
            &background_context(),
        )
        .await
        .expect("followUp serves")
        .expect("the follow-up");
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
        Some(&json!([{ "type": "image", "mimeType": "image/png", "data": "first" }])),
        "the empty-text user message kept only the image",
    );
}

#[tokio::test]
async fn the_lane_and_its_surfaces_render_summary_debugs() {
    let fixture = create_lane().await;
    let debug = format!("{:?}", fixture.lane);
    assert!(debug.contains("Lane {"), "the lane's debug header: {debug}");
    assert!(debug.contains("name:"), "the lane's name renders");
    assert!(
        !debug.contains("session:"),
        "the heavy handles stay behind the summary"
    );

    let hooks = fixture.lane.hooks();
    let _ = hooks;
    let _ = fixture.lane.read_config();
}

#[tokio::test]
async fn commands_fault_when_the_session_closes_under_them() {
    let fixture = create_lane().await;
    fixture
        .session
        .close(&background_context())
        .await
        .expect("the session closes");
    let error = fixture
        .lane
        .set_thinking_level(ThinkingLevel::High, &background_context())
        .await
        .expect_err("the closed session faults the command");
    match error {
        LaneOperationError::Closed(reason) => {
            assert!(
                !reason.to_string().is_empty(),
                "the closed session's error carries through the lane: {reason}",
            );
        }
    }
}

#[tokio::test]
async fn drive_rejects_an_already_aborted_context() {
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;
    let (_, controller) = pi_chord::context::with_cancel(&background_context());
    controller.abort("stop");
    let context = crate::harness::context::with_abort_signal(
        controller.signal().clone(),
        &background_context(),
    );
    let error = fixture
        .lane
        .drive(
            DriveOptions {
                operation_id: fixture
                    .lane
                    .state()
                    .operation
                    .as_ref()
                    .map(|operation| operation.meta.operation_id.clone())
                    .expect("the operation"),
                wait_for_retry: None,
                poll_deferred: None,
            },
            &context,
        )
        .await
        .expect_err("the aborted context rejects the drive");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(reason.to_string(), "stop", "the caller's reason carries");
        }
    }
}

#[tokio::test]
async fn drive_faults_on_a_malformed_settled_record() {
    let fixture = create_lane().await;
    commit_writes(
        &fixture,
        vec![raw_set(
            &stored_values::operation_result("broken").address,
            json!(42),
        )],
    )
    .await;
    let message = closed_message(
        fixture
            .lane
            .drive(
                DriveOptions {
                    operation_id: "broken".to_owned(),
                    wait_for_retry: None,
                    poll_deferred: None,
                },
                &background_context(),
            )
            .await
            .expect_err("the malformed record faults the drive"),
    );
    assert!(
        message.contains("Operation result is malformed"),
        "the malformed record names its parse failure: {message}",
    );
}

#[tokio::test]
async fn drive_reports_the_mismatch_against_the_live_operation() {
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;
    let error = fixture
        .lane
        .drive(
            DriveOptions {
                operation_id: "foreign".to_owned(),
                wait_for_retry: None,
                poll_deferred: None,
            },
            &background_context(),
        )
        .await
        .expect("drive serves")
        .expect_err("the foreign id rejects");
    match error {
        crate::harness::result::HarnessError::OperationMismatch {
            current_operation_id,
            ..
        } => {
            assert!(
                current_operation_id.is_some(),
                "the mismatch names the live operation",
            );
        }
        other => panic!("the mismatch shape: {other:?}"),
    }
}

#[tokio::test]
async fn navigation_on_a_busy_lane_reports_lane_busy() {
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;
    let error = fixture
        .lane
        .navigate_tree(Some("target".to_owned()), None, &background_context())
        .await
        .expect("navigate serves")
        .expect_err("the busy lane rejects the navigation");
    assert_eq!(error.tag(), "LaneBusy");
}

#[tokio::test]
async fn inspect_execution_carries_the_captured_model() {
    let fixture = create_lane().await;
    accepted_operation(&fixture).await;
    patch_live_operation(
        &fixture,
        crate::harness::session::types::OperationState::AssistantEffectPending(
            crate::harness::session::types::AssistantEffectPendingOperation {
                scope: OperationScope {
                    control: Control::Running,
                    settings: crate::harness::session::types::RunSettings {
                        compaction: crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS,
                        steering_mode: crate::types::QueueMode::All,
                        follow_up_mode: crate::types::QueueMode::All,
                        tool_execution: crate::types::ToolExecutionMode::Parallel,
                    },
                    latest_assistant_entry_id: None,
                },
                generation_context: crate::harness::session::types::GenerationContext {
                    step_id: "step".to_owned(),
                    trigger_entry_id: "trigger".to_owned(),
                    configuration: lane_configuration(),
                    stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
                    retry_policy: crate::harness::session::types::NormalizedRetryPolicy {
                        max_attempts: 2,
                        base_delay_ms: 1,
                        max_agent_delay_ms: 30_000,
                    },
                    overflow_recovery_used: false,
                },
                attempt: 1,
                response_entry_id: "response".to_owned(),
                usage_id: "usage".to_owned(),
                intended_output_limit: 100,
                context_window: 1_000,
            },
        ),
    )
    .await;

    let info = fixture
        .lane
        .inspect_execution(&background_context())
        .await
        .expect("inspect serves");
    let current = info.current.as_ref().expect("the current operation");
    assert_eq!(
        current
            .captured_model
            .as_ref()
            .map(|model| model.model_id.as_str()),
        Some("model"),
        "the leaf's configuration captures the model",
    );
}

/// Swaps the live operation's durable leaf, the capture fixtures' patch.
async fn patch_live_operation(
    fixture: &LaneFixture,
    next_state: crate::harness::session::types::OperationState,
) {
    let live = fixture.lane.state().operation.expect("the live operation");
    let operation_id = live.meta.operation_id.clone();
    let meta = live.meta;
    fixture
        .lane
        .command::<(), _>(
            move |state, _session, _context| {
                let operation_id = operation_id.clone();
                let meta = meta.clone();
                let next_state = next_state.clone();
                Box::pin(async move {
                    let mut next = state.clone();
                    next.operation = Some(LiveOperation {
                        meta,
                        state: next_state.clone(),
                    });
                    Ok(LaneCommand::Commit {
                        writes: vec![
                            set_value_write(
                                &stored_values::operation_state(&operation_id),
                                next_state,
                            )
                            .expect("state write"),
                        ],
                        next,
                        materialize: Arc::new(|_commit| ()),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the patch commits");
}

#[test]
fn the_trait_name_reads_the_lane_name() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("the runtime");
    let fixture = runtime.block_on(create_lane());
    assert_eq!(AgentLane::name(&fixture.lane), "main");
}

#[tokio::test]
async fn cancel_queued_names_the_next_run_queue_in_the_invariant() {
    let fixture = create_lane().await;
    let entry_id = fixture
        .lane
        .next_run(
            QueueMessage::Text("next".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("nextRun serves")
        .expect("the next-run");
    commit_writes(
        &fixture,
        vec![stored_values::delete_value_write(
            &stored_values::pending_entry(&entry_id),
        )],
    )
    .await;
    let error = fixture
        .lane
        .cancel_queued(&entry_id, &background_context())
        .await
        .expect_err("the missing payload faults");
    assert!(
        error.to_string().contains(&format!(
            "Queued nextRun entry {entry_id} is missing its payload"
        )),
        "the missing payload names the next-run queue: {error}",
    );
}

#[tokio::test]
async fn navigation_reports_its_kind_through_the_inspection() {
    let fixture = create_lane().await;
    let first = fixture
        .lane
        .append_message(boundary_user_message("root"), &background_context())
        .await
        .expect("append serves");
    fixture
        .lane
        .append_message(boundary_user_message("head"), &background_context())
        .await
        .expect("append serves");
    let error = fixture
        .lane
        .navigate_tree(Some(first), None, &background_context())
        .await
        .expect_err("the staged drive seam faults the navigation");
    assert_eq!(
        closed_message(error),
        "drive operation is not implemented until its later AgentHarness slice",
    );
    let info = fixture
        .lane
        .inspect_execution(&background_context())
        .await
        .expect("inspect serves");
    assert_eq!(
        info.current.as_ref().expect("the current operation").kind,
        OperationKind::Navigation,
        "the navigation's kind rides the inspection",
    );
}

#[tokio::test]
async fn cancel_queued_names_the_follow_up_queue_in_the_invariant() {
    let fixture = create_lane().await;
    let entry_id = fixture
        .lane
        .follow_up(
            QueueMessage::Text("follow".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("followUp serves")
        .expect("the follow-up");
    commit_writes(
        &fixture,
        vec![stored_values::delete_value_write(
            &stored_values::pending_entry(&entry_id),
        )],
    )
    .await;
    let error = fixture
        .lane
        .cancel_queued(&entry_id, &background_context())
        .await
        .expect_err("the missing payload faults");
    assert!(
        error.to_string().contains(&format!(
            "Queued followUp entry {entry_id} is missing its payload"
        )),
        "the missing payload names the follow-up queue: {error}",
    );
}

#[tokio::test]
async fn seal_closes_the_installed_drive_and_waits_out_the_idle_owner() {
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
            let receiver = lock(&release_cell).take();
            if let Some(receiver) = receiver {
                let _ = receiver.await;
            }
        })
    });
    let idle = {
        let lane = fixture.lane.clone();
        tokio::spawn(async move { lane.run_when_idle(callback, &background_context()).await })
    };
    started.await.expect("the callback claimed the idle slot");
    // The drive installs while the idle callback holds the lane; the seal
    // closes both.
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: "drive".to_owned(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    fixture.lane.set_active_drive(Some(Arc::clone(&drive)));

    let sealed = {
        let lane = fixture.lane.clone();
        let closed: LaneError = Arc::new(HarnessClosed);
        tokio::spawn(async move { lane.seal(closed).await })
    };
    tokio::task::yield_now().await;
    assert!(!sealed.is_finished(), "the seal waits out the idle owner",);
    let completion = drive.completion().await;
    assert!(completion.is_err(), "the seal closed the installed pass",);
    let _ = release.send(());
    sealed.await.expect("seal join");
    idle.await
        .expect("idle join")
        .expect("the callback settled");
}

#[tokio::test]
async fn cancel_queued_names_the_write_queue_in_the_invariant() {
    let fixture = lane_seeded(
        noop_emit_batch(),
        vec![
            set_value_write(
                &stored_values::lane_state("main"),
                crate::harness::session::types::LaneState {
                    current_operation_id: None,
                    last_operation_id: None,
                    inbox: vec![InboxItem {
                        entry_id: "orphan-write".to_owned(),
                        kind: InboxItemKind::Write,
                    }],
                },
            )
            .expect("state write"),
        ],
    )
    .await;
    let error = fixture
        .lane
        .cancel_queued("orphan-write", &background_context())
        .await
        .expect_err("the missing payload faults");
    assert!(
        error
            .to_string()
            .contains("Queued write entry orphan-write is missing its payload"),
        "the missing payload names the write queue: {error}",
    );
}

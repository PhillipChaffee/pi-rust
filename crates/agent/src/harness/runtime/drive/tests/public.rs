//! The public drive surface suite, ported 1:1 from upstream
//! `test/harness/runtime/drive-public.test.ts` ("runtime public drive") at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The fixture restates upstream's local `createFixture`/`acceptRun`: the
//! harness builds through [`create_agent_harness`] over a plain memory
//! backend, the faux provider registers into a fresh models catalog, and
//! the `main` lane attaches through the public surface; the parked-response
//! factory restates upstream's async provider closures (the started gate
//! resolves, the release gate parks the provider call), and every test
//! closes its session where upstream's `afterEach` did.
//!
//! Restatements the port makes and the tests bind:
//! - upstream's `lane instanceof Lane` check (`Expected runtime Lane`)
//!   restates as the static guarantee that [`Harness::lane`] publishes
//!   [`crate::harness::runtime::lane::Lane`] instances — the [`AgentLane`]
//!   trait carries no downcast seam and the crate forbids unsafe — so the
//!   `runtimeLane` projection reads restate through the durable
//!   `pi.op.state`/`pi.lane.state` values and the
//!   [`AgentLane::inspect_execution`] view, and the `activeDrive` reads
//!   pin behaviorally: the same-pass join's equal results and single
//!   provider call, the still-serving lane after a rejected install, and
//!   the owner-clear the boundary suite's spawn-path test pins directly.
//! - upstream's `rejects.toBe(error)` caller-cancellation identity carries
//!   the abort reason's message; the trait surface collapses the error
//!   object to its message, the lane suite's standing note.
//! - `it.each` tables restate as case-table loops inside one test.
//! - upstream's "releases idle ownership when the callback fails" case is
//!   unrepresentable: the [`IdleCallback`] contract returns `()`, so the
//!   callback has no failure channel, and a panicking callback would strand
//!   the idle owner the release block clears. The case has no runtime test
//!   here.
//! - the competing acceptance's listener holds its delivery slot until the
//!   spawned admission installs — upstream's microtask ordering admits the
//!   competitor before the delivery resolves for free, and that install is
//!   what closes the continuation window; tokio's scheduler gives no such
//!   ordering, so the listener observes the install explicitly.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the tests pin outcomes; unexpected results and violated expectations panic the test by design"
)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pi_ai::models::create_models;
use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxProviderHandle;
use pi_ai::providers::faux::FauxResponseStep;
use pi_ai::providers::faux::{RegisterFauxProviderOptions, faux_assistant_message, faux_provider};
use pi_ai::types::{DeferredRequest, Message, UserContent, UserMessage};
use serde_json::json;

use crate::harness::agent_harness::AgentHarnessOptions;
use crate::harness::agent_harness::AgentLane;
use crate::harness::agent_harness::CancelQueuedKind;
use crate::harness::agent_harness::CompactionEndStatus;
use crate::harness::agent_harness::CompactionOutcome;
use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HarnessEventType;
use crate::harness::agent_harness::HookFailure;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::agent_harness::HookOptions;
use crate::harness::agent_harness::LaneOperationError;
use crate::harness::agent_harness::NavigateOptions;
use crate::harness::agent_harness::OperationAdmission;
use crate::harness::agent_harness::OperationRequest;
use crate::harness::agent_harness::PromptMessagesPayload;
use crate::harness::agent_harness::{QueueMessage, RecordUsageOptions, RunFollowUp, RunOutcome};
use crate::harness::context::{Context, background_context};
use crate::harness::result::HarnessError;
use crate::harness::runtime::harness::{Harness, create_agent_harness};
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::{deferred, lock, runtime_session_metadata};
use crate::harness::session::commit::insert_entry;
use crate::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use crate::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use crate::harness::session::types::LaneState as DurableLaneState;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::ModelIdentity;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::OperationKind;
use crate::harness::session::types::{OperationState, Session, SessionReader, TerminalStatus};
use crate::harness::session::values::{Write, lane_state, operation_meta, operation_state};
use crate::harness::types::AgentHarnessResources;
use crate::harness::types::{AgentHarnessStreamOptions, PromptTemplate, Skill};
use crate::types::AgentMessage;

/// The process-unique session id one fixture carries, upstream's
/// `` `public-drive-${sessions.length}` ``.
fn next_session_id() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("public-drive-{}", COUNTER.fetch_add(1, Ordering::SeqCst))
}

/// The fixture's per-suite options, upstream's `createFixture(options)`
/// parameter.
#[derive(Default)]
struct PublicFixtureOptions {
    /// Whether the harness streams with deferred execution, upstream's
    /// `{ streamOptions: { deferred: true } }`.
    deferred: bool,
    /// The resources the harness serves, upstream's `resources`.
    resources: Option<AgentHarnessResources>,
}

/// The fixture upstream's `createFixture` builds, minus the runtime-Lane
/// downcast (see the module docs): the public lane surface, the concrete
/// container, the faux handle, and the session the durable reads and the
/// teardown close use.
struct PublicFixture {
    /// The lane's public surface, upstream's `lane`/`runtimeLane` pair.
    lane: Arc<dyn AgentLane>,
    /// The concrete container, upstream's `harness`.
    harness: Harness,
    /// The faux provider handle, upstream's `faux`.
    faux: FauxProviderHandle,
    /// The session, upstream's `session`.
    session: Arc<StorageBackedSession>,
}

/// Builds one fixture, upstream's `createFixture({ deferred?, resources? })`.
///
/// # Panics
/// The creation's or the lane attach's failure.
async fn create_fixture_with(options: PublicFixtureOptions) -> PublicFixture {
    let session = Arc::new(StorageBackedSession::new(
        runtime_session_metadata(next_session_id()),
        Arc::new(MemoryStorage::new(MemoryStorageOptions::default())),
        StorageBackedSessionOptions::default(),
    ));
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    let models = Arc::new(create_models(None));
    models.set_provider(Arc::new(faux.provider.clone()));
    let stream_options = options.deferred.then(|| AgentHarnessStreamOptions {
        deferred: Some(DeferredRequest::Enabled(true)),
        ..AgentHarnessStreamOptions::default()
    });
    let created = create_agent_harness(
        AgentHarnessOptions {
            session: session.clone(),
            models,
            model: faux.first_model(),
            thinking_level: None,
            active_tool_names: None,
            tools: None,
            tool_context: None,
            system_prompt: None,
            resources: options.resources,
            stream_options,
            retry: None,
            compaction: None,
            steering_mode: None,
            follow_up_mode: None,
            tool_execution: None,
            to_provider_messages: None,
            entry_projectors: None,
        },
        &background_context(),
    )
    .await
    .expect("the harness creates");
    let lane = created
        .harness
        .lane("main", &background_context())
        .await
        .expect("the lane attaches");
    PublicFixture {
        lane,
        harness: created.harness,
        faux,
        session,
    }
}

/// Builds one default fixture, upstream's `createFixture()`.
///
/// # Panics
/// The creation's or the lane attach's failure.
async fn create_fixture() -> PublicFixture {
    create_fixture_with(PublicFixtureOptions::default()).await
}

/// Admits one named prompt run, upstream's `acceptRun(lane, operationId)`:
/// the operation id doubles as the prompt text.
///
/// # Panics
/// The admission's failure.
async fn accept_run(lane: &Arc<dyn AgentLane>, operation_id: &str) -> OperationAdmission {
    lane.accept(
        OperationRequest::Prompt {
            operation_id: Some(operation_id.to_owned()),
            prompt: Box::new(PromptMessagesPayload::Text {
                prompt: operation_id.to_owned(),
                images: None,
            }),
        },
        &background_context(),
    )
    .await
    .expect("accept serves")
    .expect("the admission")
}

/// The parked summary response, upstream's
/// `async () => { summaryStarted.resolve(); await releaseSummary.promise;
/// return fauxAssistantMessage(text) }` factories: the factory resolves the
/// started gate, parks on the release gate, then returns the message.
fn parked_response(
    text: &'static str,
    started: tokio::sync::oneshot::Sender<()>,
    release: tokio::sync::oneshot::Receiver<()>,
) -> FauxResponseStep {
    let started_cell = Arc::new(Mutex::new(Some(started)));
    let release_cell = Arc::new(Mutex::new(Some(release)));
    FauxResponseStep::Factory(Arc::new(move |_context, _options, _state, _model| {
        let started_cell = Arc::clone(&started_cell);
        let release_cell = Arc::clone(&release_cell);
        Box::pin(async move {
            let started = lock(&started_cell).take();
            if let Some(started) = started {
                let _ = started.send(());
            }
            // The guard drops before the await: a std MutexGuard is not
            // Send, and the factory's future is.
            let release = lock(&release_cell).take();
            if let Some(release) = release {
                let _ = release.await;
            }
            Ok(faux_assistant_message(
                text,
                FauxAssistantMessageOptions::default(),
            ))
        })
    }))
}

/// Builds one user text message with the fixture's timestamp, upstream's
/// `{ role: "user", content, timestamp }` literals.
fn user_message(content: &str, timestamp: i64) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        timestamp,
        content: UserContent::Text(content.to_owned()),
    }))
}

/// Closes the fixture's session, upstream's `afterEach` teardown; closing
/// twice is idempotent where the harness close already drained it.
///
/// # Panics
/// The session close's failure.
/// Drives one operation id and unwraps the settled outcome, the
/// convenience-surface tests' pass; an unexpected failure panics the test
/// by design.
async fn drive_serves(fixture: &PublicFixture, operation_id: &str) -> DriveOutcome {
    fixture
        .lane
        .drive(
            DriveOptions {
                operation_id: operation_id.to_owned(),
                wait_for_retry: None,
                poll_deferred: None,
            },
            &background_context(),
        )
        .await
        .expect("the drive serves")
        .expect("the drive settles")
}

async fn close_session(fixture: &PublicFixture) {
    fixture
        .session
        .close(&background_context())
        .await
        .expect("the session closes");
}

/// The settled record the drive results carry, upstream's
/// `value: { kind: "settled", outcome }` reads.
#[must_use]
fn settled_record(outcome: DriveOutcome) -> crate::harness::session::types::OperationResultRecord {
    let DriveOutcome::Settled { outcome } = outcome else {
        panic!("expected a settled outcome")
    };
    outcome
}

/// The settled record the run results carry, upstream's
/// `value: { kind: "run", status, error? }` reads.
#[must_use]
fn run_record(run: RunOutcome) -> crate::harness::session::types::OperationResultRecord {
    let RunOutcome::Settled(record) = run else {
        panic!("expected a settled run")
    };
    record
}

/// Upstream's "persists model identity without requiring a local
/// registration" case.
#[tokio::test]
async fn persists_model_identity_without_requiring_a_local_registration() {
    let fixture = create_fixture().await;
    fixture
        .lane
        .set_model(
            ModelIdentity {
                provider: "missing".to_owned(),
                model_id: "missing-model".to_owned(),
            },
            &background_context(),
        )
        .await
        .expect("the model sets");
    assert!(
        fixture
            .lane
            .get_model(&background_context())
            .await
            .expect("the model reads")
            .is_none(),
        "the unregistered model reads undefined",
    );

    let run = fixture
        .lane
        .prompt_text("prompt", None, &background_context())
        .await
        .expect("prompt serves")
        .expect("the run");
    let record = run_record(run);
    assert_eq!(record.kind, OperationKind::Run);
    assert_eq!(record.status, TerminalStatus::Failed);
    assert_eq!(
        record.error.as_ref().map(|error| error.code.as_str()),
        Some("model_unavailable"),
    );
    assert_eq!(fixture.faux.state().call_count(), 0);
    close_session(&fixture).await;
}

/// Upstream's "composes prompt, skill, and template acceptance with drive"
/// case.
#[tokio::test]
async fn composes_prompt_skill_and_template_acceptance_with_drive() {
    let resources = AgentHarnessResources {
        prompt_templates: vec![PromptTemplate {
            name: "fix".to_owned(),
            description: None,
            content: "Fix $1".to_owned(),
        }],
        skills: vec![Skill {
            name: "review".to_owned(),
            description: "Review".to_owned(),
            content: "Inspect it".to_owned(),
            file_path: "/skills/review/SKILL.md".to_owned(),
            disable_model_invocation: None,
        }],
    };
    let fixture = create_fixture_with(PublicFixtureOptions {
        deferred: false,
        resources: Some(resources),
    })
    .await;
    fixture.faux.set_responses([
        faux_assistant_message("prompt answer", FauxAssistantMessageOptions::default()).into(),
        faux_assistant_message("skill answer", FauxAssistantMessageOptions::default()).into(),
        faux_assistant_message("template answer", FauxAssistantMessageOptions::default()).into(),
    ]);

    for (run, what) in [
        (
            fixture
                .lane
                .prompt_text("prompt", None, &background_context())
                .await,
            "the prompt",
        ),
        (
            fixture
                .lane
                .skill("review", Some("strict".to_owned()), &background_context())
                .await,
            "the skill",
        ),
        (
            fixture
                .lane
                .prompt_from_template("fix", Some(vec!["it".to_owned()]), &background_context())
                .await,
            "the template",
        ),
    ] {
        let run = run.expect("the invocation serves").expect(what);
        let record = run_record(run);
        assert_eq!(record.status, TerminalStatus::Completed, "{what} completes");
    }
    assert_eq!(fixture.faux.state().call_count(), 3);
    close_session(&fixture).await;
}

/// Upstream's "returns a convenience-only suspension observation" case.
#[tokio::test]
async fn returns_a_convenience_only_suspension_observation() {
    let fixture = create_fixture_with(PublicFixtureOptions {
        deferred: true,
        resources: None,
    })
    .await;
    fixture.faux.set_responses([faux_assistant_message(
        "eventual answer",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);

    let run = fixture
        .lane
        .prompt_text("defer", None, &background_context())
        .await
        .expect("prompt serves")
        .expect("the run");
    let RunOutcome::Suspended(suspended) = run else {
        panic!("the deferred run suspends")
    };
    assert_eq!(suspended.status, "suspended");
    assert_eq!(suspended.deferred.provider, "faux");
    assert_eq!(suspended.deferred.model_id, "faux-1");
    let stored = crate::harness::runtime::test_support::stored_value(
        &fixture.session,
        &operation_state(&suspended.operation_id).address,
        &background_context(),
    )
    .await;
    let state: OperationState = serde_json::from_value(stored.value).expect("the leaf parses");
    assert_eq!(state.at(), "deferred.suspended");
    close_session(&fixture).await;
}

/// Upstream's "records caller usage as an adjustment and publishes
/// committed totals" case.
#[tokio::test]
async fn records_caller_usage_as_an_adjustment_and_publishes_committed_totals() {
    let fixture = create_fixture().await;
    let usage = faux_assistant_message("usage", FauxAssistantMessageOptions::default()).usage;
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&recorded);
    fixture
        .harness
        .events()
        .on(
            HarnessEventType::Usage,
            crate::harness::runtime::test_support::recording_listener(Arc::clone(&sink)),
        )
        .expect("the usage listener subscribes");

    let usage_id = fixture
        .lane
        .record_usage(
            usage,
            Some(RecordUsageOptions {
                entry_id: Some("external".to_owned()),
                details: Some(json!({ "source": "test" })),
            }),
            &background_context(),
        )
        .await
        .expect("the record serves");

    let events = lock(&recorded).clone();
    assert_eq!(events.len(), 1, "one usage event publishes");
    let HarnessEventPayload::Usage { lane, row, totals } = &events[0].payload else {
        panic!("the event is a usage event")
    };
    assert_eq!(lane, "main");
    assert_eq!(row.id, usage_id, "the row carries the caller's id");
    assert!(row.adjustment, "the row is an adjustment");
    assert_eq!(
        totals,
        &fixture
            .session
            .get_stats(&background_context())
            .await
            .expect("the stats read")
            .usage,
        "the totals are the committed session usage",
    );
    close_session(&fixture).await;
}

/// Upstream's `it.each(["steer", "followUp", "nextRun"])` cases:
/// "starts an ordinary continuation run from queued steer input",
/// "starts an ordinary continuation run from queued followUp input", and
/// "starts an ordinary continuation run from queued nextRun input".
#[tokio::test]
async fn starts_an_ordinary_continuation_run_from_queued_input() {
    /// The queue kinds the upstream `it.each` table drives.
    #[derive(Clone, Copy)]
    enum QueuedKind {
        Steer,
        FollowUp,
        NextRun,
    }
    for kind in [QueuedKind::Steer, QueuedKind::FollowUp, QueuedKind::NextRun] {
        let fixture = create_fixture().await;
        fixture
            .lane
            .append_message(user_message("history", 1), &background_context())
            .await
            .expect("the append commits");
        let (started_tx, started) = deferred();
        let (release_tx, release_rx) = deferred();
        fixture.faux.set_responses([
            parked_response("summary", started_tx, release_rx),
            faux_assistant_message(
                "continuation answer",
                FauxAssistantMessageOptions::default(),
            )
            .into(),
        ]);
        let compacting = {
            let lane = Arc::clone(&fixture.lane);
            tokio::spawn(async move { lane.compact(None, &background_context()).await })
        };
        started.await.expect("the summary started");
        let queued = match kind {
            QueuedKind::Steer => {
                fixture
                    .lane
                    .steer(
                        QueueMessage::Text("continue".to_owned()),
                        Vec::new(),
                        &background_context(),
                    )
                    .await
            }
            QueuedKind::FollowUp => {
                fixture
                    .lane
                    .follow_up(
                        QueueMessage::Text("continue".to_owned()),
                        Vec::new(),
                        &background_context(),
                    )
                    .await
            }
            QueuedKind::NextRun => {
                fixture
                    .lane
                    .next_run(
                        QueueMessage::Text("continue".to_owned()),
                        Vec::new(),
                        &background_context(),
                    )
                    .await
            }
        };
        queued.expect("the queue serves").expect("the queue");
        let _ = release_tx.send(());

        let compacted = compacting
            .await
            .expect("the compaction join")
            .expect("the compaction serves")
            .expect("the compaction");
        let CompactionOutcome { compaction, run } = compacted;
        assert_eq!(compaction.kind, OperationKind::Compaction);
        assert_eq!(compaction.status, TerminalStatus::Completed);
        let RunFollowUp::Settled(run_record) = run.expect("the queued input ran") else {
            panic!("the continuation run settles")
        };
        assert_eq!(run_record.kind, OperationKind::Run);
        assert_eq!(run_record.status, TerminalStatus::Completed);
        assert_eq!(fixture.faux.state().call_count(), 2);
        close_session(&fixture).await;
    }
}

/// The competing admission's task handle, the deferred `accept` promise
/// the hook listener parks and the test later joins.
type CompetingAdmission = tokio::task::JoinHandle<
    Result<crate::harness::agent_harness::OperationAdmissionResult, LaneOperationError>,
>;

/// Upstream's "lets a competing acceptance win the structural continuation
/// window" case.
#[expect(
    clippy::too_many_lines,
    reason = "the competing-acceptance choreography is one flow the window's sides sequence"
)]
#[tokio::test]
async fn lets_a_competing_acceptance_win_the_structural_continuation_window() {
    let fixture = create_fixture().await;
    fixture
        .lane
        .append_message(user_message("history", 1), &background_context())
        .await
        .expect("the append commits");
    let (started_tx, started) = deferred();
    let (release_tx, release_rx) = deferred();
    fixture.faux.set_responses([
        parked_response("summary", started_tx, release_rx),
        faux_assistant_message("competitor answer", FauxAssistantMessageOptions::default()).into(),
    ]);
    let competing: Arc<Mutex<Option<CompetingAdmission>>> = Arc::new(Mutex::new(None));
    {
        let competing = Arc::clone(&competing);
        let listener_lane = Arc::clone(&fixture.lane);
        let listener_session = Arc::clone(&fixture.session);
        fixture
            .harness
            .events()
            .on(
                HarnessEventType::CompactionEnd,
                Arc::new(move |event: &HarnessEvent, _context| {
                    let competing = Arc::clone(&competing);
                    let listener_lane = Arc::clone(&listener_lane);
                    let listener_session = Arc::clone(&listener_session);
                    Box::pin(async move {
                        let HarnessEventPayload::CompactionEnd { status, .. } = &event.payload
                        else {
                            return Ok(());
                        };
                        if !matches!(status, CompactionEndStatus::Completed { .. }) {
                            return Ok(());
                        }
                        // The listener holds its delivery slot until the
                        // spawned admission installs — upstream's microtask
                        // ordering admits the competitor before the delivery
                        // resolves for free, and that install is what closes
                        // the continuation window. The install reads through
                        // the durable lane record: the spawned accept parks
                        // on its own event delivery, which queues behind
                        // this one, so its resolution cannot observe here.
                        let lane = Arc::clone(&listener_lane);
                        let handle = tokio::spawn(async move {
                            lane.accept(
                                OperationRequest::Prompt {
                                    operation_id: None,
                                    prompt: Box::new(PromptMessagesPayload::Text {
                                        prompt: "competitor".to_owned(),
                                        images: None,
                                    }),
                                },
                                &background_context(),
                            )
                            .await
                        });
                        *lock(&competing) = Some(handle);
                        for _ in 0..200 {
                            let installed = listener_session
                                .get_value(&lane_state("main").address, &background_context())
                                .await
                                .expect("the lane state reads")
                                .is_some_and(|stored| {
                                    serde_json::from_value::<DurableLaneState>(stored.value)
                                        .is_ok_and(|state| state.current_operation_id.is_some())
                                });
                            if installed {
                                return Ok(());
                            }
                            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                        }
                        panic!("the competing admission did not install");
                    })
                }),
            )
            .expect("the listener registers");
    }
    let compacting = {
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move { lane.compact(None, &background_context()).await })
    };
    started.await.expect("the summary started");
    fixture
        .lane
        .next_run(
            QueueMessage::Text("queued".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("the queue serves")
        .expect("the queue");
    let _ = release_tx.send(());

    let compacted = compacting
        .await
        .expect("the compaction join")
        .expect("the compaction serves")
        .expect("the compaction");
    assert_eq!(compacted.compaction.status, TerminalStatus::Completed);
    assert!(
        compacted.run.is_none(),
        "the competing acceptance won the continuation window"
    );
    let competing = lock(&competing)
        .take()
        .expect("Competing acceptance did not start");
    let admission = competing
        .await
        .expect("the competing join")
        .expect("the admission serves")
        .expect("the admission");
    let driven = drive_serves(&fixture, &admission.operation_id).await;
    assert_eq!(
        settled_record(driven).status,
        TerminalStatus::Completed,
        "the competitor's run drove to completion",
    );
    close_session(&fixture).await;
}

/// Upstream's "cancels queued input and reports consumed or missing ids"
/// case.
#[tokio::test]
async fn cancels_queued_input_and_reports_consumed_or_missing_ids() {
    let fixture = create_fixture().await;
    let cancelled = fixture
        .lane
        .next_run(
            QueueMessage::Text("cancel".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("the queue serves")
        .expect("the queue");
    assert_eq!(
        fixture
            .lane
            .cancel_queued(&cancelled, &background_context())
            .await
            .expect("the cancel serves"),
        CancelQueuedKind::Cancelled,
    );
    assert_eq!(
        fixture
            .lane
            .cancel_queued(&cancelled, &background_context())
            .await
            .expect("the cancel serves"),
        CancelQueuedKind::NotFound,
    );

    let consumed = fixture
        .lane
        .next_run(
            QueueMessage::Text("consume".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("the queue serves")
        .expect("the queue");
    let admission = {
        let lane = Arc::clone(&fixture.lane);
        lane.accept(
            OperationRequest::Prompt {
                operation_id: None,
                prompt: Box::new(PromptMessagesPayload::Text {
                    prompt: String::new(),
                    images: None,
                }),
            },
            &background_context(),
        )
        .await
        .expect("accept serves")
        .expect("the admission")
    };
    assert_eq!(
        fixture
            .lane
            .cancel_queued(&consumed, &background_context())
            .await
            .expect("the cancel serves"),
        CancelQueuedKind::AlreadyConsumed,
    );
    fixture.faux.set_responses([faux_assistant_message(
        "answer",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);
    let _ = drive_serves(&fixture, &admission.operation_id).await;
    close_session(&fixture).await;
}

/// Upstream's "admits input after cancellation and preserves it through
/// reconciliation" case.
#[tokio::test]
async fn admits_input_after_cancellation_and_preserves_it_through_reconciliation() {
    let fixture = create_fixture().await;
    accept_run(&fixture.lane, "cancelled").await;
    fixture
        .lane
        .request_abort("cancelled", &background_context())
        .await
        .expect("the abort request serves")
        .expect("the abort request");
    fixture
        .lane
        .steer(
            QueueMessage::Text("late".to_owned()),
            Vec::new(),
            &background_context(),
        )
        .await
        .expect("the steer serves")
        .expect("the steer");
    let driven = drive_serves(&fixture, "cancelled").await;
    assert_eq!(settled_record(driven).status, TerminalStatus::Aborted);
    let stored = fixture
        .session
        .get_value(&lane_state("main").address, &background_context())
        .await
        .expect("the lane state reads")
        .expect("the lane state is durable");
    let durable: DurableLaneState =
        serde_json::from_value(stored.value).expect("the lane state parses");
    assert_eq!(durable.inbox.len(), 1, "the late steer survived");

    fixture.faux.set_responses([faux_assistant_message(
        "late answer",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);
    let run = fixture
        .lane
        .prompt_text("", None, &background_context())
        .await
        .expect("prompt serves")
        .expect("the run");
    assert_eq!(run_record(run).status, TerminalStatus::Completed);
    close_session(&fixture).await;
}

/// Upstream's "composes standalone compaction acceptance with drive" case.
#[tokio::test]
async fn composes_standalone_compaction_acceptance_with_drive() {
    let fixture = create_fixture().await;
    fixture
        .lane
        .append_message(user_message("history", 1), &background_context())
        .await
        .expect("the append commits");
    fixture.faux.set_responses([faux_assistant_message(
        "summary",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);

    let compacted = fixture
        .lane
        .compact(None, &background_context())
        .await
        .expect("the compaction serves")
        .expect("the compaction");
    assert_eq!(compacted.compaction.kind, OperationKind::Compaction);
    assert_eq!(compacted.compaction.status, TerminalStatus::Completed);
    assert_eq!(fixture.faux.state().call_count(), 1);
    close_session(&fixture).await;
}

/// Upstream's `it.each([false, true])` cases: "composes false/true
/// summarized navigation acceptance with drive".
#[tokio::test]
async fn composes_summarized_navigation_acceptance_with_drive() {
    for summarize in [false, true] {
        let fixture = create_fixture().await;
        let root_id = fixture
            .lane
            .append_message(user_message("root", 1), &background_context())
            .await
            .expect("the append commits");
        fixture
            .lane
            .append_message(user_message("source", 2), &background_context())
            .await
            .expect("the append commits");
        commit_writes(
            &fixture.session,
            vec![Write::Entry(Box::new(insert_entry(NewEntry::Message {
                id: "target".to_owned(),
                parent_id: Some(root_id),
                body: Box::new(MessageEntry {
                    message: AgentMessage::Standard(Message::User(UserMessage {
                        timestamp: 3,
                        content: UserContent::Text("target".to_owned()),
                    })),
                    terminate: None,
                }),
            })))],
        )
        .await;
        if summarize {
            fixture.faux.set_responses([faux_assistant_message(
                "branch summary",
                FauxAssistantMessageOptions::default(),
            )
            .into()]);
        }

        let navigated = fixture
            .lane
            .navigate_tree(
                Some("target".to_owned()),
                Some(NavigateOptions {
                    summarize: Some(summarize),
                    label: Some("chosen".to_owned()),
                    custom_instructions: None,
                }),
                &background_context(),
            )
            .await
            .expect("the navigation serves")
            .expect("the navigation");
        assert_eq!(navigated.navigation.kind, OperationKind::Navigation);
        assert_eq!(navigated.navigation.status, TerminalStatus::Completed);
        assert_eq!(
            fixture.faux.state().call_count(),
            u64::from(summarize),
            "the summarized navigation generates; the plain one does not",
        );
        assert!(
            fixture
                .lane
                .get_tip_id(&background_context())
                .await
                .expect("the tip read")
                .is_some(),
            "the navigation left a tip",
        );
        close_session(&fixture).await;
    }
}

/// Upstream's "resumes any current operation after acceptance or reopen"
/// case.
#[tokio::test]
async fn resumes_any_current_operation_after_acceptance_or_reopen() {
    let fixture = create_fixture().await;
    accept_run(&fixture.lane, "run").await;
    fixture.faux.set_responses([faux_assistant_message(
        "answer",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);

    let resumed = fixture
        .lane
        .resume(&background_context())
        .await
        .expect("resume serves")
        .expect("the run");
    let record = run_record(resumed);
    assert_eq!(record.operation_id, "run");
    assert_eq!(record.status, TerminalStatus::Completed);
    close_session(&fixture).await;
}

/// Upstream's "polls one deferred permit through resume" case.
#[tokio::test]
async fn polls_one_deferred_permit_through_resume() {
    let fixture = create_fixture_with(PublicFixtureOptions {
        deferred: true,
        resources: None,
    })
    .await;
    fixture.faux.set_responses([faux_assistant_message(
        "eventual answer",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);
    let suspended = fixture
        .lane
        .prompt_text("defer", None, &background_context())
        .await
        .expect("prompt serves")
        .expect("the run");
    let RunOutcome::Suspended(suspended) = suspended else {
        panic!("the deferred run suspends")
    };

    let resumed = fixture
        .lane
        .resume(&background_context())
        .await
        .expect("resume serves")
        .expect("the run");
    let record = run_record(resumed);
    assert_eq!(record.operation_id, suspended.operation_id);
    assert_eq!(record.status, TerminalStatus::Completed);
    assert!(
        fixture
            .session
            .get_value(
                &operation_state(&suspended.operation_id).address,
                &background_context()
            )
            .await
            .expect("the operation state reads")
            .is_none(),
        "the resumed operation cleared",
    );
    assert_eq!(fixture.faux.state().deferred_fetch_count(), 1);
    let again = fixture
        .lane
        .resume(&background_context())
        .await
        .expect("resume serves");
    assert!(
        matches!(again, Err(HarnessError::NothingToResume { .. })),
        "the empty resume rejects: {again:?}"
    );
    close_session(&fixture).await;
}

/// Upstream's "aborts and reconciles the current operation" case.
#[tokio::test]
async fn aborts_and_reconciles_the_current_operation() {
    let fixture = create_fixture().await;
    accept_run(&fixture.lane, "run").await;

    let aborted = fixture
        .lane
        .abort(&background_context())
        .await
        .expect("the abort serves")
        .expect("the abort");
    assert_eq!(aborted.operation_id, "run");
    assert!(aborted.steer.is_empty());
    assert!(aborted.follow_up.is_empty());
    assert!(
        fixture
            .session
            .get_value(&operation_state("run").address, &background_context())
            .await
            .expect("the operation state reads")
            .is_none(),
        "the aborted operation cleared",
    );
    assert_eq!(fixture.faux.state().call_count(), 0);
    let again = fixture
        .lane
        .abort(&background_context())
        .await
        .expect("the abort serves");
    assert!(
        matches!(again, Err(HarnessError::NoActiveOperation { .. })),
        "the empty abort rejects: {again:?}"
    );
    close_session(&fixture).await;
}

/// Upstream's "waits for an operation that has no installed drive" case.
#[tokio::test]
async fn waits_for_an_operation_that_has_no_installed_drive() {
    let fixture = create_fixture().await;
    accept_run(&fixture.lane, "run").await;
    let idle = Arc::new(AtomicBool::new(false));
    let waiting = {
        let idle = Arc::clone(&idle);
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move {
            lane.wait_for_idle(&background_context())
                .await
                .expect("the wait serves");
            idle.store(true, Ordering::SeqCst);
        })
    };
    tokio::task::yield_now().await;
    assert!(
        !idle.load(Ordering::SeqCst),
        "the wait holds while the operation runs"
    );

    fixture.faux.set_responses([faux_assistant_message(
        "answer",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);
    let _ = drive_serves(&fixture, "run").await;
    waiting.await.expect("the wait joins");
    assert!(idle.load(Ordering::SeqCst), "the wait released");
    close_session(&fixture).await;
}

/// Upstream's "serializes concurrent runWhenIdle callbacks" case.
#[tokio::test]
async fn serializes_concurrent_run_when_idle_callbacks() {
    let fixture = create_fixture().await;
    let (started_tx, started) = deferred();
    let (release_tx, release_rx) = deferred();
    let started_cell = Arc::new(Mutex::new(Some(started_tx)));
    let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let release_cell = Arc::new(Mutex::new(Some(release_rx)));
    let first_order = Arc::clone(&order);
    let first = {
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move {
            lane.run_when_idle(
                Arc::new(move |_context| {
                    let first_order = Arc::clone(&first_order);
                    let started_cell = Arc::clone(&started_cell);
                    let release_cell = Arc::clone(&release_cell);
                    Box::pin(async move {
                        lock(&first_order).push("first:start");
                        let started = lock(&started_cell).take();
                        if let Some(started) = started {
                            let _ = started.send(());
                        }
                        let release = lock(&release_cell).take();
                        if let Some(release) = release {
                            let _ = release.await;
                        }
                        lock(&first_order).push("first:end");
                    })
                }),
                &background_context(),
            )
            .await
            .expect("the first callback serves");
        })
    };
    started.await.expect("the first callback started");
    let second_order = Arc::clone(&order);
    let second = {
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move {
            lane.run_when_idle(
                Arc::new(move |_context| {
                    let second_order = Arc::clone(&second_order);
                    Box::pin(async move {
                        lock(&second_order).push("second");
                    })
                }),
                &background_context(),
            )
            .await
            .expect("the second callback serves");
        })
    };
    tokio::task::yield_now().await;
    assert_eq!(
        lock(&order).clone(),
        ["first:start"],
        "the second callback waits for the first",
    );

    let _ = release_tx.send(());
    first.await.expect("the first joins");
    second.await.expect("the second joins");
    assert_eq!(lock(&order).clone(), ["first:start", "first:end", "second"]);
    close_session(&fixture).await;
}

/// Upstream's "owns the idle window while runWhenIdle executes" case.
#[tokio::test]
async fn owns_the_idle_window_while_run_when_idle_executes() {
    let fixture = create_fixture().await;
    let (started_tx, started) = deferred();
    let (release_tx, release_rx) = deferred();
    let callback =
        crate::harness::runtime::test_support::parking_idle_callback(started_tx, release_rx);
    let callback = {
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move { lane.run_when_idle(callback, &background_context()).await })
    };
    started.await.expect("the callback started");
    let acceptance = {
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move {
            lane.accept(
                OperationRequest::Prompt {
                    operation_id: None,
                    prompt: Box::new(PromptMessagesPayload::Text {
                        prompt: "after".to_owned(),
                        images: None,
                    }),
                },
                &background_context(),
            )
            .await
        })
    };
    tokio::task::yield_now().await;
    assert!(
        !acceptance.is_finished(),
        "the acceptance waits for the idle window",
    );

    let _ = release_tx.send(());
    callback
        .await
        .expect("the callback joins")
        .expect("the callback serves");
    let admission = acceptance
        .await
        .expect("the acceptance join")
        .expect("the admission serves")
        .expect("the admission");
    fixture.faux.set_responses([faux_assistant_message(
        "answer",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);
    let _ = drive_serves(&fixture, &admission.operation_id).await;
    close_session(&fixture).await;
}

/// Upstream's "allows coherent lane reads from an idle callback" case.
#[tokio::test]
async fn allows_coherent_lane_reads_from_an_idle_callback() {
    let fixture = create_fixture().await;
    let lane = Arc::clone(&fixture.lane);
    fixture
        .lane
        .run_when_idle(
            Arc::new(move |context: &Context| {
                let lane = Arc::clone(&lane);
                Box::pin(async move {
                    let execution = lane
                        .inspect_execution(context)
                        .await
                        .expect("the execution reads");
                    assert!(
                        execution.current.is_none(),
                        "the execution view reads empty"
                    );
                    let watch = lane.watch(context).await.expect("the watch serves");
                    let snapshot = watch.snapshot();
                    assert!(snapshot.operation.is_none(), "the snapshot reads empty");
                    watch.unsubscribe();
                })
            }),
            &background_context(),
        )
        .await
        .expect("the callback serves");
    close_session(&fixture).await;
}

/// Upstream's "close waits for an already-running idle callback" case.
#[tokio::test]
async fn close_waits_for_an_already_running_idle_callback() {
    let fixture = create_fixture().await;
    let (started_tx, started) = deferred();
    let (release_tx, release_rx) = deferred();
    let callback =
        crate::harness::runtime::test_support::parking_idle_callback(started_tx, release_rx);
    let callback = {
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move { lane.run_when_idle(callback, &background_context()).await })
    };
    started.await.expect("the callback started");
    let harness = fixture.harness.clone();
    let closing = tokio::spawn(async move { harness.close(&background_context()).await });
    tokio::task::yield_now().await;
    assert!(
        !closing.is_finished(),
        "the close waits for the idle callback",
    );

    let _ = release_tx.send(());
    callback
        .await
        .expect("the callback joins")
        .expect("the callback serves");
    closing
        .await
        .expect("the close joins")
        .expect("the close settles");
    close_session(&fixture).await;
}

/// Upstream's "installs one pass and joins same-operation callers" case.
#[tokio::test]
async fn installs_one_pass_and_joins_same_operation_callers() {
    let fixture = create_fixture().await;
    let (started_tx, started) = deferred();
    let (release_tx, release_rx) = deferred();
    fixture
        .faux
        .set_responses([parked_response("answer", started_tx, release_rx)]);
    let admission = accept_run(&fixture.lane, "run").await;
    let operation_id = admission.operation_id.clone();

    let first = {
        let lane = Arc::clone(&fixture.lane);
        let operation_id = operation_id.clone();
        tokio::spawn(async move {
            lane.drive(
                DriveOptions {
                    operation_id,
                    wait_for_retry: None,
                    poll_deferred: None,
                },
                &background_context(),
            )
            .await
        })
    };
    started.await.expect("the pass started");
    let second = {
        let lane = Arc::clone(&fixture.lane);
        let operation_id = operation_id.clone();
        tokio::spawn(async move {
            lane.drive(
                DriveOptions {
                    operation_id,
                    wait_for_retry: None,
                    poll_deferred: None,
                },
                &background_context(),
            )
            .await
        })
    };
    let _ = release_tx.send(());

    let first_result = first.await.expect("the first joins");
    let second_result = second.await.expect("the second joins");
    let first_outcome = first_result
        .as_ref()
        .expect("the first drive serves")
        .as_ref()
        .expect("the first drive settles");
    let second_outcome = second_result
        .as_ref()
        .expect("the second drive serves")
        .as_ref()
        .expect("the second drive settles");
    assert_eq!(
        first_outcome, second_outcome,
        "the same-operation join returns the settled result to both callers",
    );
    let record = settled_record(first_outcome.clone());
    assert_eq!(record.operation_id, "run");
    assert_eq!(record.status, TerminalStatus::Completed);
    assert_eq!(fixture.faux.state().call_count(), 1);
    close_session(&fixture).await;
}

/// Upstream's "returns old result records without disturbing the current
/// operation" case.
#[tokio::test]
async fn returns_old_result_records_without_disturbing_the_current_operation() {
    let fixture = create_fixture().await;
    fixture.faux.set_responses([faux_assistant_message(
        "first",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);
    accept_run(&fixture.lane, "first").await;
    let first_driven = drive_serves(&fixture, "first").await;
    assert_eq!(settled_record(first_driven).operation_id, "first");
    accept_run(&fixture.lane, "second").await;

    let old_driven = drive_serves(&fixture, "first").await;
    assert_eq!(
        settled_record(old_driven).operation_id,
        "first",
        "the old record returns",
    );
    let stored = fixture
        .session
        .get_value(&operation_meta("second").address, &background_context())
        .await
        .expect("the operation meta reads")
        .expect("the current operation is durable");
    let meta: crate::harness::session::types::OperationMeta =
        serde_json::from_value(stored.value).expect("the meta parses");
    assert_eq!(meta.operation_id, "second", "the current operation stands");
    assert!(
        fixture
            .session
            .get_value(&operation_meta("first").address, &background_context())
            .await
            .expect("the operation meta reads")
            .is_none(),
        "the settled operation's meta cleared",
    );

    fixture.faux.set_responses([faux_assistant_message(
        "second",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);
    let second_driven = drive_serves(&fixture, "second").await;
    assert_eq!(settled_record(second_driven).operation_id, "second");
    close_session(&fixture).await;
}

/// Upstream's "isolates stale operation ids" case.
#[tokio::test]
async fn isolates_stale_operation_ids() {
    let fixture = create_fixture().await;
    accept_run(&fixture.lane, "current").await;

    let error = fixture
        .lane
        .drive(
            DriveOptions {
                operation_id: "stale".to_owned(),
                wait_for_retry: None,
                poll_deferred: None,
            },
            &background_context(),
        )
        .await
        .expect("the drive serves")
        .expect_err("the stale id mismatches");
    match error {
        HarnessError::OperationMismatch {
            expected_operation_id,
            current_operation_id,
            ..
        } => {
            assert_eq!(expected_operation_id, "stale");
            assert_eq!(current_operation_id.as_deref(), Some("current"));
        }
        other => panic!("the stale drive rejects with the mismatch: {other:?}"),
    }
    assert_eq!(fixture.faux.state().call_count(), 0);
    close_session(&fixture).await;
}

/// Upstream's "does not install for a caller already cancelled" case.
#[tokio::test]
async fn does_not_install_for_a_caller_already_cancelled() {
    let fixture = create_fixture().await;
    accept_run(&fixture.lane, "run").await;
    let (_, controller) = pi_chord::context::with_cancel(&background_context());
    let caller = crate::harness::context::with_abort_signal(
        controller.signal().clone(),
        &background_context(),
    );
    controller.abort("caller cancelled");

    let error = fixture
        .lane
        .drive(
            DriveOptions {
                operation_id: "run".to_owned(),
                wait_for_retry: None,
                poll_deferred: None,
            },
            &caller,
        )
        .await
        .expect_err("the cancelled caller rejects");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "caller cancelled",
                "the caller's reason carries"
            );
        }
    }
    assert_eq!(fixture.faux.state().call_count(), 0);
    close_session(&fixture).await;
}

/// Upstream's "caller cancellation stops only that caller's observation"
/// case.
#[tokio::test]
async fn caller_cancellation_stops_only_that_callers_observation() {
    let fixture = create_fixture().await;
    let (started_tx, started) = deferred();
    let (release_tx, release_rx) = deferred();
    fixture
        .faux
        .set_responses([parked_response("answer", started_tx, release_rx)]);
    accept_run(&fixture.lane, "run").await;
    let owner = {
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move {
            lane.drive(
                DriveOptions {
                    operation_id: "run".to_owned(),
                    wait_for_retry: None,
                    poll_deferred: None,
                },
                &background_context(),
            )
            .await
        })
    };
    started.await.expect("the pass started");
    let (_, controller) = pi_chord::context::with_cancel(&background_context());
    let caller = crate::harness::context::with_abort_signal(
        controller.signal().clone(),
        &background_context(),
    );
    let observer = {
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move {
            lane.drive(
                DriveOptions {
                    operation_id: "run".to_owned(),
                    wait_for_retry: None,
                    poll_deferred: None,
                },
                &caller,
            )
            .await
        })
    };
    controller.abort("observer cancelled");

    let error = observer
        .await
        .expect("the observer joins")
        .expect_err("the cancelled observer rejects");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "observer cancelled",
                "the observer's reason carries"
            );
        }
    }
    let _ = release_tx.send(());
    let owner_driven = owner
        .await
        .expect("the owner joins")
        .expect("the owner drive serves")
        .expect("the owner drive settles");
    assert_eq!(
        settled_record(owner_driven).status,
        TerminalStatus::Completed,
        "the owner observation settles",
    );
    assert_eq!(fixture.faux.state().call_count(), 1);
    close_session(&fixture).await;
}

/// Upstream's "close rejects observation without waiting for a
/// non-cooperative effect" case.
#[tokio::test]
async fn close_rejects_observation_without_waiting_for_a_non_cooperative_effect() {
    let fixture = create_fixture().await;
    let (started_tx, started) = deferred();
    let (release_tx, release_rx) = deferred();
    fixture
        .faux
        .set_responses([parked_response("late", started_tx, release_rx)]);
    accept_run(&fixture.lane, "run").await;
    let observation = {
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move {
            lane.drive(
                DriveOptions {
                    operation_id: "run".to_owned(),
                    wait_for_retry: None,
                    poll_deferred: None,
                },
                &background_context(),
            )
            .await
        })
    };
    started.await.expect("the pass started");

    fixture
        .harness
        .close(&background_context())
        .await
        .expect("the close settles");
    let error = observation
        .await
        .expect("the observation joins")
        .expect_err("the closed observation rejects");
    match error {
        LaneOperationError::Closed(reason) => {
            assert!(
                matches!(reason, HarnessError::Closed { .. }),
                "the closed observation rejects with the closed error: {reason:?}"
            );
        }
    }
    let _ = release_tx.send(());
    close_session(&fixture).await;
}

/// Upstream's "exposes durable abort and reconciles through public drive"
/// case.
#[tokio::test]
async fn exposes_durable_abort_and_reconciles_through_public_drive() {
    let fixture = create_fixture().await;
    accept_run(&fixture.lane, "run").await;

    let requested = fixture
        .lane
        .request_abort("run", &background_context())
        .await
        .expect("the abort request serves")
        .expect("the abort request");
    assert_eq!(requested.operation_id, "run");
    assert!(requested.newly_requested, "the abort newly requested");
    assert!(requested.steer.is_empty());
    assert!(requested.follow_up.is_empty());
    let driven = drive_serves(&fixture, "run").await;
    let record = settled_record(driven);
    assert_eq!(record.operation_id, "run");
    assert_eq!(record.status, TerminalStatus::Aborted);
    assert_eq!(fixture.faux.state().call_count(), 0);
    let again = fixture
        .lane
        .request_abort("run", &background_context())
        .await
        .expect("the abort request serves");
    match again {
        Err(HarnessError::OperationMismatch {
            expected_operation_id,
            last_operation_id,
            ..
        }) => {
            assert_eq!(expected_operation_id, "run");
            assert_eq!(last_operation_id.as_deref(), Some("run"));
        }
        other => panic!("the stale abort request rejects: {other:?}"),
    }
    close_session(&fixture).await;
}

/// Upstream's "faults the harness when a detached pass fails" case.
#[tokio::test]
async fn faults_the_harness_when_a_detached_pass_fails() {
    let fixture = create_fixture().await;
    fixture
        .harness
        .hooks()
        .on(
            HookName::BeforeDrive,
            Arc::new(|_event: &HookInvocation, _context: &Context| {
                let failure: HookFailure =
                    Box::new(crate::harness::session::types::SessionError::Message(
                        "drive failed".to_owned(),
                    ));
                Box::pin(async move { Err(failure) })
            }),
            HookOptions::default(),
        )
        .expect("the hook registers");
    accept_run(&fixture.lane, "run").await;

    let error = fixture
        .lane
        .drive(
            DriveOptions {
                operation_id: "run".to_owned(),
                wait_for_retry: None,
                poll_deferred: None,
            },
            &background_context(),
        )
        .await
        .expect_err("the failed pass faults");
    match error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "AgentHarness storage or invariant fault",
                "the drive rejects with the fault surface: {reason}"
            );
        }
    }
    let tip_error = fixture
        .lane
        .get_tip_id(&background_context())
        .await
        .expect_err("the faulted lane refuses reads");
    match tip_error {
        LaneOperationError::Closed(reason) => {
            assert_eq!(
                reason.to_string(),
                "AgentHarness storage or invariant fault",
                "the fault poisons later reads: {reason}"
            );
        }
    }
    close_session(&fixture).await;
}

//! The runtime Harness lane management and global metadata cases, ported
//! 1:1 from upstream `test/harness/runtime/harness.test.ts` ("runtime
//! Harness lane management" / "runtime Harness global metadata") at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements the port makes and the tests bind:
//! - upstream's `"accept" in harness` / `"getTipId" in harness` /
//!   `"appendMessage" in harness` checks are structural: the
//!   [`AgentHarness`] trait carries no lane operations, so the
//!   fresh-attach case pins the branchless surface those checks guard
//!   (`lanes()` empty, no branch) and the trait-level fact holds by
//!   construction.
//! - upstream's `expect(second).toBe(lane)` identity restates as
//!   `Arc::ptr_eq` on the acquired handles, and the `instanceof Lane`
//!   narrowings restate through the harness's lane map, whose entry is the
//!   very instance the trait handle wraps.
//! - upstream's `afterEach` close restates as a `close_sessions` call at
//!   each test's end.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::{Arc, Mutex};

use pi_ai::auth::types::ProviderAuth;
use pi_ai::models::{Provider, ProviderImpl, ProviderModelError, create_models};
use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::{RegisterFauxProviderOptions, faux_assistant_message, faux_provider};
use pi_ai::types::BoxedFuture;
use pi_ai::types::Context as AiContext;
use pi_ai::types::{Model, SimpleStreamOptions, StreamOptions};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde_json::json;
use tokio::sync::oneshot;

use super::common;
use crate::harness::agent_harness::AcquireLaneOptions;
use crate::harness::agent_harness::AgentHarness;
use crate::harness::agent_harness::AgentHarnessOptions;
use crate::harness::agent_harness::AgentLane;
use crate::harness::agent_harness::ConfigUpdateKind;
use crate::harness::agent_harness::CreateAt;
use crate::harness::agent_harness::GlobalConfigUpdate;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HarnessEventType;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::agent_harness::HookOptions;
use crate::harness::agent_harness::HookResult;
use crate::harness::agent_harness::LaneOperationError;
use crate::harness::agent_harness::{OperationRequest, PromptMessagesPayload, ValueUpdateKind};
use crate::harness::context::{Context, background_context};
use crate::harness::result::HarnessError;
use crate::harness::runtime::harness::Harness;
use crate::harness::runtime::harness::{HarnessCreationError, create_agent_harness, lock_lanes};
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::deferred;
use crate::harness::runtime::test_support::provider_pass_through;
use crate::harness::runtime::test_support::{lane_state_write, lock, settle_events};
use crate::harness::runtime::types::LaneCommand;
use crate::harness::session::commit::insert_entry;
use crate::harness::session::memory::{MemorySessionRepo, MemorySessionRepoOptions};
use crate::harness::session::session::StorageBackedSession;
use crate::harness::session::types::BranchScan;
use crate::harness::session::types::BranchScanOrder;
use crate::harness::session::types::CustomEntryBody;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::LaneState;
use crate::harness::session::types::ModelIdentity;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::{Session, SessionCreateOptions, SessionReader, SessionRepo};
use crate::harness::session::values as stored_values;
use crate::harness::session::values::{Write, set_value_write};
use crate::harness::types::AgentHarnessStreamOptions;
use crate::types::{QueueMode, ThinkingLevel};

/// The session-id recording provider, upstream's `provider: Provider = {
/// ..., streamSimple: ... }` wrapper over the faux `base`: every simple
/// stream records the request's session id and delegates to the base
/// provider, and the members upstream's object literal omits stay at the
/// trait defaults.
struct RecordingProvider {
    /// The faux provider the streams delegate to.
    base: ProviderImpl,
    /// The recorded session ids, one per simple stream.
    session_ids: Arc<Mutex<Vec<Option<String>>>>,
}

impl Provider for RecordingProvider {
    provider_pass_through!(base);

    fn stream_simple(
        &self,
        model: &Model,
        context: &AiContext,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        lock(&self.session_ids).push(options.and_then(|options| options.session_id.clone()));
        self.base.stream_simple(model, context, options)
    }
}

/// Attaches the runtime over one ledgered session, upstream's
/// `createHarness(session)` overload; the session-less cases ride
/// [`common::create_harness`].
///
/// # Panics
/// The creation's failure.
async fn create_harness_with_session(session: &Arc<StorageBackedSession>) -> Harness {
    let (options, _faux) = common::harness_options(Arc::clone(session));
    create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates")
        .harness
}

/// The concrete runtime lane the acquisition returns, upstream's
/// `instanceof Lane` narrowing: the harness's lane map holds the very
/// instance the trait handle wraps (one `Arc` allocation), so the shared
/// handle restates the downcast the trait surface cannot express.
///
/// # Panics
/// The `expected` message when the handle and the map entry diverge —
/// upstream's `"Expected runtime Lane"` / `"Expected runtime Lanes"`
/// throws — or the lane's absence.
fn runtime_lane(
    harness: &Harness,
    acquired: &Arc<dyn AgentLane>,
    name: &str,
    expected: &str,
) -> Arc<Lane> {
    let lane = lock_lanes(&harness.shared)
        .get(name)
        .cloned()
        .expect("the acquired lane");
    assert_eq!(
        Arc::as_ptr(acquired).cast::<u8>(),
        Arc::as_ptr(&lane).cast::<u8>(),
        "{expected}",
    );
    lane
}

/// The custom entry write upstream's inline
/// `{ kind: "entry", entry: { id, parentId: null, type: "custom",
/// customType } }` commit builds.
fn custom_entry_write(id: &str, custom_type: &str) -> Write {
    Write::Entry(Box::new(insert_entry(NewEntry::Custom {
        id: id.to_owned(),
        parent_id: None,
        body: CustomEntryBody {
            custom_type: custom_type.to_owned(),
            data: None,
        },
    })))
}

/// The stored lane record a read parses, upstream's
/// `(await session.getValue(laneState(lane), ...))?.value` reads.
async fn stored_lane_record(session: &Arc<StorageBackedSession>, lane: &str) -> LaneState {
    let stored = crate::harness::runtime::test_support::stored_value(
        session,
        &stored_values::lane_state(lane).address,
        &background_context(),
    )
    .await;
    serde_json::from_value(stored.value).expect("the lane record parses")
}

/// Upstream `it` under "runtime Harness lane management": attaches to a
/// fresh Session without creating an implicit main lane.
#[tokio::test]
async fn attaches_to_a_fresh_session_without_creating_an_implicit_main_lane() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    let harness = create_harness_with_session(&session).await;

    let lanes = harness
        .lanes(&background_context())
        .await
        .expect("the lanes read");
    assert!(lanes.is_empty(), "the fresh harness carries no lanes");
    let branch = session
        .branch("main", &background_context())
        .await
        .expect("the branch read");
    assert!(branch.is_none(), "the session stays branchless");
    // The trait surface carries no lane operations — upstream's
    // `"accept" in harness` / `"getTipId" in harness` /
    // `"appendMessage" in harness` checks hold by construction.

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness lane management": atomically gets
/// or creates a complete `AgentLane`.
#[tokio::test]
async fn atomically_gets_or_creates_a_complete_agent_lane() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    let harness = create_harness_with_session(&session).await;
    let created: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&created);
    harness
        .events()
        .on(
            HarnessEventType::LaneCreated,
            Arc::new(move |event: &HarnessEvent, _context| {
                let sink = Arc::clone(&sink);
                Box::pin(async move {
                    lock(&sink).push(
                        event
                            .lane
                            .clone()
                            .expect("the lane-created event carries its lane"),
                    );
                    Ok(())
                })
            }),
        )
        .expect("the listener subscribes");

    let context = background_context();
    let lane = harness
        .lane("main", &context)
        .await
        .expect("the lane attaches");
    let same = harness
        .lane_with_options(
            "main",
            AcquireLaneOptions {
                create_at: Some(CreateAt::Entry("ignored".to_owned())),
            },
            &context,
        )
        .await
        .expect("the existing lane returns");

    assert!(
        Arc::ptr_eq(&same, &lane),
        "the acquisition returns one instance"
    );
    assert_eq!(
        lane.get_tip_id(&context).await.expect("the tip read"),
        None,
        "the created lane tips at none",
    );
    assert_eq!(
        lane.get_thinking_level(&context)
            .await
            .expect("the level read"),
        ThinkingLevel::Medium,
        "the lane carries the options' thinking level",
    );
    assert_eq!(
        lane.get_active_tools(&context)
            .await
            .expect("the tools read"),
        vec!["read".to_owned(), "bash".to_owned()],
        "the lane carries the options' active tools",
    );
    assert_eq!(
        stored_lane_record(&session, "main").await,
        LaneState {
            current_operation_id: None,
            last_operation_id: None,
            inbox: Vec::new(),
        },
        "the durable lane state is the complete idle record",
    );
    assert_eq!(
        *lock(&created),
        vec!["main".to_owned()],
        "one lane_created publishes"
    );
    let infos = harness.lanes(&context).await.expect("the lanes read");
    assert_eq!(infos.len(), 1, "one lane carries");
    assert_eq!(infos[0].name, "main", "the lane's name carries");
    assert_eq!(infos[0].tip_id, None, "the lane tips at none");

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness lane management": returns one
/// published `AgentLane` under concurrent acquisition.
#[tokio::test]
async fn returns_one_published_agent_lane_under_concurrent_acquisition() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let created: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&created);
    harness
        .events()
        .on(
            HarnessEventType::LaneCreated,
            Arc::new(move |event: &HarnessEvent, _context| {
                let sink = Arc::clone(&sink);
                Box::pin(async move {
                    lock(&sink).push(
                        event
                            .lane
                            .clone()
                            .expect("the lane-created event carries its lane"),
                    );
                    Ok(())
                })
            }),
        )
        .expect("the listener subscribes");

    let acquire = |harness: Harness| async move {
        harness
            .lane("main", &background_context())
            .await
            .expect("the lane attaches")
    };
    let (first, second, third) = tokio::join!(
        tokio::spawn(acquire(harness.clone())),
        tokio::spawn(acquire(harness.clone())),
        tokio::spawn(acquire(harness.clone())),
    );
    let first = first.expect("the acquisition joins");
    let second = second.expect("the acquisition joins");
    let third = third.expect("the acquisition joins");

    assert!(
        Arc::ptr_eq(&second, &first),
        "the acquisitions share one instance"
    );
    assert!(
        Arc::ptr_eq(&third, &first),
        "the acquisitions share one instance"
    );
    assert_eq!(
        *lock(&created),
        vec!["main".to_owned()],
        "exactly one lane_created publishes",
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness lane management": uses one stable
/// provider session id per lane.
#[tokio::test]
async fn uses_one_stable_provider_session_id_per_lane() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session_with_id(&mut sessions, "shared-session");
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    faux.set_responses([
        faux_assistant_message("main one", FauxAssistantMessageOptions::default()).into(),
        faux_assistant_message("main two", FauxAssistantMessageOptions::default()).into(),
        faux_assistant_message("review", FauxAssistantMessageOptions::default()).into(),
    ]);
    let session_ids: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let models = Arc::new(create_models(None));
    models.set_provider(Arc::new(RecordingProvider {
        base: faux.provider.clone(),
        session_ids: Arc::clone(&session_ids),
    }));
    let options = AgentHarnessOptions {
        session,
        models,
        model: faux.first_model(),
        thinking_level: None,
        active_tool_names: Some(Vec::new()),
        tools: None,
        tool_context: None,
        system_prompt: None,
        resources: None,
        stream_options: None,
        retry: None,
        compaction: None,
        steering_mode: None,
        follow_up_mode: None,
        tool_execution: None,
        to_provider_messages: None,
        entry_projectors: None,
    };
    let created = create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates");
    let context = background_context();
    let main = created
        .harness
        .lane("main", &context)
        .await
        .expect("the main lane attaches");
    let review = created
        .harness
        .lane("review", &context)
        .await
        .expect("the review lane attaches");

    let _ = main
        .prompt_text("one", None, &background_context())
        .await
        .expect("the prompt serves")
        .expect("the run settles");
    let _ = main
        .prompt_text("two", None, &background_context())
        .await
        .expect("the prompt serves")
        .expect("the run settles");
    let _ = review
        .prompt_text("review", None, &background_context())
        .await
        .expect("the prompt serves")
        .expect("the run settles");

    assert_eq!(
        *lock(&session_ids),
        vec![
            Some("shared-session:main".to_owned()),
            Some("shared-session:main".to_owned()),
            Some("shared-session:review".to_owned()),
        ],
        "the provider session ids stay stable per lane",
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness lane management": serializes
/// commands from different `AgentLanes` on the one Session line.
#[tokio::test]
async fn serializes_commands_from_different_agent_lanes_on_the_one_session_line() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();
    let main_acquired = harness
        .lane("main", &context)
        .await
        .expect("the main lane attaches");
    let review_acquired = harness
        .lane("review", &context)
        .await
        .expect("the review lane attaches");
    let main = runtime_lane(&harness, &main_acquired, "main", "Expected runtime Lanes");
    let review = runtime_lane(
        &harness,
        &review_acquired,
        "review",
        "Expected runtime Lanes",
    );

    let order: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let (started_tx, started_rx) = deferred();
    let (gate_tx, gate_rx) = deferred();
    let started: Arc<Mutex<Option<oneshot::Sender<()>>>> = Arc::new(Mutex::new(Some(started_tx)));
    let gate: Arc<Mutex<Option<oneshot::Receiver<()>>>> = Arc::new(Mutex::new(Some(gate_rx)));

    let first = {
        let lane = Arc::clone(&main);
        let order = Arc::clone(&order);
        let started = Arc::clone(&started);
        let gate = Arc::clone(&gate);
        tokio::spawn(async move {
            lane.command::<(), _>(
                move |_state, _session, _context| {
                    let order = Arc::clone(&order);
                    let started = Arc::clone(&started);
                    let gate = Arc::clone(&gate);
                    Box::pin(async move {
                        lock(&order).push("main:start".to_owned());
                        let started = lock(&started).take();
                        if let Some(started) = started {
                            let _ = started.send(());
                        }
                        let gate = lock(&gate).take();
                        if let Some(gate) = gate {
                            let _ = gate.await;
                        }
                        lock(&order).push("main:end".to_owned());
                        Ok(LaneCommand::Return { result: () })
                    })
                },
                &background_context(),
            )
            .await
            .expect("the command serves");
        })
    };
    started_rx.await.expect("the first command starts");
    let second = {
        let lane = Arc::clone(&review);
        let order = Arc::clone(&order);
        tokio::spawn(async move {
            lane.command::<(), _>(
                move |_state, _session, _context| {
                    let order = Arc::clone(&order);
                    Box::pin(async move {
                        lock(&order).push("review".to_owned());
                        Ok(LaneCommand::Return { result: () })
                    })
                },
                &background_context(),
            )
            .await
            .expect("the command serves");
        })
    };
    settle_events().await;
    assert_eq!(
        *lock(&order),
        vec!["main:start".to_owned()],
        "the second command waits on the line",
    );
    let _ = gate_tx.send(());
    let (first_outcome, second_outcome) = tokio::join!(first, second);
    first_outcome.expect("the first command joins");
    second_outcome.expect("the second command joins");
    assert_eq!(
        *lock(&order),
        vec![
            "main:start".to_owned(),
            "main:end".to_owned(),
            "review".to_owned()
        ],
        "the commands serialize in arrival order",
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness lane management": uses createAt
/// only for a missing lane and validates the target.
#[tokio::test]
async fn uses_create_at_only_for_a_missing_lane_and_validates_the_target() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    commit_writes(&session, vec![custom_entry_write("target", "target")]).await;
    let harness = create_harness_with_session(&session).await;
    let context = background_context();
    let lane = harness
        .lane_with_options(
            "review",
            AcquireLaneOptions {
                create_at: Some(CreateAt::Entry("target".to_owned())),
            },
            &context,
        )
        .await
        .expect("the lane creates at the target");

    assert_eq!(
        lane.get_tip_id(&context).await.expect("the tip read"),
        Some("target".to_owned()),
        "the lane tips at the createAt entry",
    );
    let same = harness
        .lane_with_options(
            "review",
            AcquireLaneOptions {
                create_at: Some(CreateAt::Entry("missing".to_owned())),
            },
            &context,
        )
        .await
        .expect("the existing lane returns");
    assert!(
        Arc::ptr_eq(&same, &lane),
        "createAt is ignored for existing lanes"
    );

    let Err(unknown) = harness
        .lane_with_options(
            "missing",
            AcquireLaneOptions {
                create_at: Some(CreateAt::Entry("unknown".to_owned())),
            },
            &context,
        )
        .await
    else {
        panic!("the unknown target rejects");
    };
    let error = unknown
        .as_ref()
        .downcast_ref::<HarnessError>()
        .expect("the caller error");
    assert!(
        matches!(error, HarnessError::UnknownTarget { .. }),
        "the unknown target rejects: {error}",
    );

    let Err(empty) = harness.lane("", &context).await else {
        panic!("the empty name rejects");
    };
    let error = empty
        .as_ref()
        .downcast_ref::<HarnessError>()
        .expect("the caller error");
    assert!(
        matches!(error, HarnessError::InvalidLane { .. }),
        "the empty name rejects: {error}",
    );

    let Err(nul) = harness.lane("bad\u{0}name", &context).await else {
        panic!("the NUL name rejects");
    };
    let error = nul
        .as_ref()
        .downcast_ref::<HarnessError>()
        .expect("the caller error");
    assert!(
        matches!(error, HarnessError::InvalidLane { .. }),
        "the NUL name rejects: {error}",
    );
    let HarnessError::InvalidLane { message, .. } = error else {
        panic!("the NUL name rejects InvalidLane");
    };
    assert_eq!(
        message, "Invalid lane \"bad\\u0000name\": lane name must not contain \\u0000",
        "the NUL name's message renders the escaped form",
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness lane management": attaches agent
/// state to a data-only Branch without moving its tip.
#[tokio::test]
async fn attaches_agent_state_to_a_data_only_branch_without_moving_its_tip() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    commit_writes(
        &session,
        vec![
            custom_entry_write("target", "target"),
            set_value_write(
                &stored_values::branch_tip("main"),
                Some("target".to_owned()),
            )
            .expect("the branch tip write"),
        ],
    )
    .await;
    let harness = create_harness_with_session(&session).await;
    let context = background_context();

    let lanes = harness.lanes(&context).await.expect("the lanes read");
    assert!(lanes.is_empty(), "the data-only branch carries no lanes");
    let lane = harness
        .lane("main", &context)
        .await
        .expect("the lane attaches");
    assert_eq!(
        lane.get_tip_id(&context).await.expect("the tip read"),
        Some("target".to_owned()),
        "the lane adopts the stored tip",
    );
    let stored = crate::harness::runtime::test_support::stored_value(
        &session,
        &stored_values::lane_config("main").address,
        &background_context(),
    )
    .await;
    assert_eq!(
        serde_json::from_value::<LaneConfiguration>(stored.value).expect("the config parses"),
        LaneConfiguration {
            model: ModelIdentity {
                provider: "faux".to_owned(),
                model_id: "faux-1".to_owned(),
            },
            thinking_level: ThinkingLevel::Medium,
            active_tool_names: vec!["read".to_owned(), "bash".to_owned()],
        },
        "the lane seeds the options' configuration",
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness lane management": keeps `AgentLane`
/// appends operation-aware while exposing the Branch surface directly.
#[tokio::test]
async fn keeps_agent_lane_appends_operation_aware_while_exposing_the_branch_surface_directly() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    let harness = create_harness_with_session(&session).await;
    let context = background_context();
    let lane = harness
        .lane("main", &context)
        .await
        .expect("the lane attaches");
    let idle_id = lane
        .append_custom_entry("idle", None, &context)
        .await
        .expect("the idle append");
    assert_eq!(
        lane.get_tip_id(&context).await.expect("the tip read"),
        Some(idle_id.clone()),
        "the idle append moves the tip",
    );

    lane.accept(
        OperationRequest::Prompt {
            operation_id: None,
            prompt: Box::new(PromptMessagesPayload::Text {
                prompt: "run".to_owned(),
                images: None,
            }),
        },
        &context,
    )
    .await
    .expect("accept serves")
    .expect("the admission");
    let accepted_tip = lane
        .get_tip_id(&context)
        .await
        .expect("the tip read")
        .expect("the accepted tip");
    let pending_id = lane
        .append_custom_entry("pending", Some(json!({ "queued": true })), &context)
        .await
        .expect("the pending append");

    assert_eq!(
        lane.get_tip_id(&context).await.expect("the tip read"),
        Some(accepted_tip.clone()),
        "the append during the operation queues instead of moving the tip",
    );
    let stored_tip = crate::harness::runtime::test_support::stored_value(
        &session,
        &stored_values::branch_tip("main").address,
        &background_context(),
    )
    .await;
    assert_eq!(
        serde_json::from_value::<Option<String>>(stored_tip.value).expect("the tip parses"),
        Some(accepted_tip.clone()),
        "the durable branch tip stays at the accepted operation's tip",
    );
    let pending = crate::harness::runtime::test_support::stored_value(
        &session,
        &stored_values::pending_entry(&pending_id).address,
        &background_context(),
    )
    .await;
    assert_eq!(
        serde_json::from_value::<PendingEntry>(pending.value).expect("the pending entry parses"),
        PendingEntry::Custom {
            custom_type: "pending".to_owned(),
            payload: Some(json!({ "queued": true })),
        },
        "the queued write stores its pending entry",
    );
    let record = stored_lane_record(&session, "main").await;
    assert_eq!(
        record.inbox,
        vec![InboxItem {
            entry_id: pending_id.clone(),
            kind: InboxItemKind::Write,
        }],
        "the queued write rides the lane inbox",
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness lane management": flushes queued
/// writes before a new idle append in one commit.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single case and its one-commit choreography"
)]
#[tokio::test]
async fn flushes_queued_writes_before_a_new_idle_append_in_one_commit() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    let harness = create_harness_with_session(&session).await;
    let context = background_context();
    let acquired = harness
        .lane("main", &context)
        .await
        .expect("the lane attaches");
    let lane = runtime_lane(&harness, &acquired, "main", "Expected runtime Lane");
    let first = session.id_generator().next(None);
    let second = session.id_generator().next(None);
    let command_first = first.clone();
    let command_second = second.clone();
    lane.command::<(), _>(
        move |state, _session, _context| {
            let first = command_first.clone();
            let second = command_second.clone();
            Box::pin(async move {
                let inbox = vec![
                    InboxItem {
                        entry_id: first.clone(),
                        kind: InboxItemKind::Write,
                    },
                    InboxItem {
                        entry_id: second.clone(),
                        kind: InboxItemKind::Write,
                    },
                ];
                let mut next = state.clone();
                next.inbox = inbox.clone();
                Ok(LaneCommand::Commit {
                    writes: vec![
                        set_value_write(
                            &stored_values::pending_entry(&first),
                            PendingEntry::Custom {
                                custom_type: "queued-first".to_owned(),
                                payload: None,
                            },
                        )
                        .expect("the first pending write"),
                        set_value_write(
                            &stored_values::pending_entry(&second),
                            PendingEntry::Custom {
                                custom_type: "queued-second".to_owned(),
                                payload: None,
                            },
                        )
                        .expect("the second pending write"),
                        lane_state_write("main", None, None, inbox).expect("the lane state write"),
                    ],
                    next,
                    materialize: Arc::new(|_commit| ()),
                    events: None,
                })
            })
        },
        &context,
    )
    .await
    .expect("the queued writes commit");

    let appended = lane
        .append_custom_entry("new", None, &context)
        .await
        .expect("the idle append");

    let entries = lane
        .find_entries(
            Some(&BranchScan {
                order: Some(BranchScanOrder::OldestFirst),
                ..BranchScan::default()
            }),
            &context,
        )
        .await
        .expect("the entries read");
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.id().to_owned())
            .collect::<Vec<_>>(),
        vec![first.clone(), second.clone(), appended.clone()],
        "the queued writes land before the idle append",
    );
    assert!(lane.state().inbox.is_empty(), "the in-memory inbox clears");
    assert!(
        stored_lane_record(&session, "main").await.inbox.is_empty(),
        "the durable inbox clears",
    );
    assert!(
        session
            .get_value(
                &stored_values::pending_entry(&first).address,
                &background_context(),
            )
            .await
            .expect("the first pending entry read")
            .is_none(),
        "the first pending entry value deletes",
    );
    assert!(
        session
            .get_value(
                &stored_values::pending_entry(&second).address,
                &background_context(),
            )
            .await
            .expect("the second pending entry read")
            .is_none(),
        "the second pending entry value deletes",
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness lane management": restores
/// complete lanes without requiring main.
#[tokio::test]
async fn restores_complete_lanes_without_requiring_main() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    commit_writes(
        &session,
        vec![
            set_value_write(&stored_values::branch_tip("review"), Option::<String>::None)
                .expect("the branch tip write"),
            set_value_write(&stored_values::lane_config("review"), common::configured())
                .expect("the lane config write"),
            set_value_write(
                &stored_values::lane_state("review"),
                LaneState {
                    current_operation_id: None,
                    last_operation_id: None,
                    inbox: Vec::new(),
                },
            )
            .expect("the lane state write"),
        ],
    )
    .await;

    let (options, _faux) = common::harness_options(Arc::clone(&session));
    let created = create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates");
    assert!(created.open.is_empty(), "no live operations restore");
    let infos = created
        .harness
        .lanes(&background_context())
        .await
        .expect("the lanes read");
    assert_eq!(
        infos
            .iter()
            .map(|info| info.name.clone())
            .collect::<Vec<_>>(),
        vec!["review".to_owned()],
        "the complete lane restores without main",
    );
    let _review = created
        .harness
        .lane("review", &background_context())
        .await
        .expect("the restored lane attaches");
    let branch = session
        .branch("main", &background_context())
        .await
        .expect("the branch read");
    assert!(branch.is_none(), "the session stays branchless");

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness lane management": rejects partial
/// durable lane state as a Harness fault.
#[tokio::test]
async fn rejects_partial_durable_lane_state_as_a_harness_fault() {
    let mut sessions = common::SessionLedger::new();
    let session = common::create_session(&mut sessions);
    commit_writes(
        &session,
        vec![
            set_value_write(&stored_values::branch_tip("main"), Option::<String>::None)
                .expect("the branch tip write"),
            set_value_write(&stored_values::lane_config("main"), common::configured())
                .expect("the lane config write"),
        ],
    )
    .await;

    let (options, _faux) = common::harness_options(Arc::clone(&session));
    let created = create_agent_harness(options, &background_context()).await;
    let error = created.expect_err("the partial durable state faults creation");
    assert!(
        matches!(error, HarnessCreationError::Fault(_)),
        "the partial durable state faults creation: {error}",
    );
    assert_eq!(
        error.to_string(),
        "AgentHarness storage or invariant fault",
        "the creation fault carries the harness fault message",
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness global metadata": preserves
/// `value_update` publication and delivery.
#[tokio::test]
async fn preserves_value_update_publication_and_delivery() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let listener_harness = harness.clone();
    harness
        .events()
        .on(
            HarnessEventType::ValueUpdate,
            Arc::new(move |event: &HarnessEvent, _context| {
                let sink = Arc::clone(&sink);
                let harness = listener_harness.clone();
                Box::pin(async move {
                    match &event.payload {
                        HarnessEventPayload::ValueUpdate {
                            kind: ValueUpdateKind::SessionName { .. },
                        } => {
                            let name = harness
                                .get_name(&background_context())
                                .await
                                .expect("the name read");
                            lock(&sink).push(format!(
                                "name:{}",
                                name.unwrap_or_else(|| "undefined".to_owned())
                            ));
                        }
                        HarnessEventPayload::ValueUpdate {
                            kind: ValueUpdateKind::EntryLabel { .. },
                        } => {
                            lock(&sink).push("entry_label".to_owned());
                        }
                        _ => {}
                    }
                    Ok(())
                })
            }),
        )
        .expect("the listener subscribes");
    let context = background_context();

    harness
        .set_name(Some("named".to_owned()), &context)
        .await
        .expect("the name sets");
    harness
        .set_label("entry", Some("label".to_owned()), &context)
        .await
        .expect("the label sets");

    assert_eq!(
        harness.get_name(&context).await.expect("the name read"),
        Some("named".to_owned()),
        "the session name reads back",
    );
    assert_eq!(
        harness
            .get_label("entry", &context)
            .await
            .expect("the label read"),
        Some("label".to_owned()),
        "the entry label reads back",
    );
    assert_eq!(
        *lock(&seen),
        vec!["name:named".to_owned(), "entry_label".to_owned()],
        "the value updates publish and deliver",
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness global metadata": publishes
/// previous and current data-bearing global configuration.
#[tokio::test]
async fn publishes_previous_and_current_data_bearing_global_configuration() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let updates: Arc<Mutex<Vec<HarnessEventPayload>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&updates);
    harness
        .events()
        .on(
            HarnessEventType::ConfigUpdate,
            Arc::new(move |event: &HarnessEvent, _context| {
                let sink = Arc::clone(&sink);
                Box::pin(async move {
                    lock(&sink).push(event.payload.clone());
                    Ok(())
                })
            }),
        )
        .expect("the listener subscribes");
    let context = background_context();

    harness
        .set_stream_options(
            AgentHarnessStreamOptions {
                timeout_ms: Some(123),
                ..AgentHarnessStreamOptions::default()
            },
            &context,
        )
        .await
        .expect("the stream options set");
    harness
        .set_steering_mode(QueueMode::OneAtATime, &context)
        .await
        .expect("the steering mode sets");

    assert_eq!(lock(&updates).len(), 2, "the two setters publish");
    let updates = lock(&updates).clone();
    let (
        Some(HarnessEventPayload::ConfigUpdate {
            property:
                ConfigUpdateKind::Global(GlobalConfigUpdate::StreamOptions { value, previous }),
        }),
        Some(HarnessEventPayload::ConfigUpdate {
            property:
                ConfigUpdateKind::Global(GlobalConfigUpdate::SteeringMode {
                    value: steering_value,
                    previous: steering_previous,
                }),
        }),
    ) = (updates.first(), updates.get(1))
    else {
        unreachable!("the config updates carry their properties: {updates:?}")
    };
    assert_eq!(
        value.timeout_ms,
        Some(123),
        "the stream options' current value carries",
    );
    assert_eq!(
        previous,
        &AgentHarnessStreamOptions::default(),
        "the stream options' previous value carries",
    );
    assert_eq!(
        steering_value,
        &QueueMode::OneAtATime,
        "the steering mode's current value carries",
    );
    assert_eq!(
        steering_previous,
        &QueueMode::All,
        "the steering mode's previous value carries",
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness global metadata": closes every
/// lane and rejects later acquisition.
#[tokio::test]
async fn closes_every_lane_and_rejects_later_acquisition() {
    let mut sessions = common::SessionLedger::new();
    let harness = common::create_harness(&mut sessions).await;
    let context = background_context();
    let lane = harness
        .lane("main", &context)
        .await
        .expect("the lane attaches");
    harness.close(&context).await.expect("the close settles");

    let Err(error) = harness.lane("other", &context).await else {
        panic!("the later acquisition rejects");
    };
    assert!(
        error.to_string().contains("closed"),
        "the later acquisition rejects closed: {error}",
    );
    let error = lane
        .get_tip_id(&context)
        .await
        .expect_err("the sealed lane rejects");
    let LaneOperationError::Closed(closed) = &error;
    assert!(
        closed.message().contains("closed"),
        "the sealed lane rejects closed: {}",
        closed.message(),
    );

    common::close_sessions(&mut sessions).await;
}

/// Upstream `it` under "runtime Harness global metadata":
/// `MemorySessionRepo` creation also remains branchless.
#[tokio::test]
async fn memory_session_repo_creation_also_remains_branchless() {
    let repo = MemorySessionRepo::new(MemorySessionRepoOptions::default());
    let session = SessionRepo::create(
        &repo,
        SessionCreateOptions {
            id: Some("repo-session".to_owned()),
            parent_session_id: None,
        },
        &background_context(),
    )
    .await
    .expect("the repo creates the session");

    let branch = session
        .branch("main", &background_context())
        .await
        .expect("the branch read");
    assert!(branch.is_none(), "the repo session stays branchless");
    Session::close(session.as_ref(), &background_context())
        .await
        .expect("the session closes");
    repo.close(&background_context())
        .await
        .expect("the repo closes");
}

/// The port's trait-surface sweep: upstream's class carries one
/// implementation, the port adds the [`AgentHarness`] capability impl over
/// the concrete container, so the sweep drives every method through the
/// `dyn` surface — the lane map's identity, the session metadata and
/// label reads, the config round-trips, the slice stub, the registry
/// subscriptions, and the close.
#[expect(
    clippy::too_many_lines,
    reason = "the sweep drives every capability method in one surface pass"
)]
#[tokio::test]
async fn the_capability_trait_surface_drives_every_container_method() {
    let mut sessions = common::SessionLedger::new();
    let concrete = common::create_harness(&mut sessions).await;
    let harness: Arc<dyn AgentHarness> = Arc::new(concrete.clone());
    let context = background_context();

    let lane = harness
        .lane("main", &context)
        .await
        .expect("the lane attaches");
    let again = harness
        .lane("main", &context)
        .await
        .expect("the lane reattaches");
    assert!(Arc::ptr_eq(&lane, &again), "the lane map's identity holds");
    let with_options = harness
        .lane_with_options("main", AcquireLaneOptions::default(), &context)
        .await
        .expect("the options lane attaches");
    assert!(Arc::ptr_eq(&lane, &with_options));
    let lanes = harness.lanes(&context).await.expect("the lanes read");
    assert_eq!(lanes.len(), 1, "the attached lane lists");
    assert_eq!(lanes[0].name, "main");

    harness
        .set_name(Some("session-name".to_owned()), &context)
        .await
        .expect("the name sets");
    assert_eq!(
        harness.get_name(&context).await.expect("the name reads"),
        Some("session-name".to_owned()),
        "the session name round-trips",
    );
    assert_eq!(
        harness
            .get_label("missing", &context)
            .await
            .expect("the label reads"),
        None,
        "the unlabelled entry reads None",
    );

    let tools = harness.get_tools(&context).await.expect("the tools read");
    assert!(tools.is_empty(), "the seed carries no tools");
    harness
        .set_tools(vec![probe_tool()], &context)
        .await
        .expect("the tools set");
    assert_eq!(
        harness
            .get_tools(&context)
            .await
            .expect("the tools read")
            .len(),
        1,
        "the tool round-trips",
    );

    let resources = harness
        .get_resources(&context)
        .await
        .expect("the resources read");
    assert_eq!(
        resources,
        crate::harness::agent_harness::Resources::default(),
        "the seed carries no resources",
    );

    harness
        .set_stream_options(
            AgentHarnessStreamOptions {
                timeout_ms: Some(123),
                ..AgentHarnessStreamOptions::default()
            },
            &context,
        )
        .await
        .expect("the stream options set");
    assert_eq!(
        harness
            .get_stream_options(&context)
            .await
            .expect("the stream options read")
            .timeout_ms,
        Some(123),
        "the stream options round-trip",
    );

    let policy = harness
        .get_retry_policy(&context)
        .await
        .expect("the policy read");
    harness
        .set_retry_policy(policy, &context)
        .await
        .expect("the policy sets");
    let compaction = harness
        .get_compaction_settings(&context)
        .await
        .expect("the compaction read");
    harness
        .set_compaction_settings(compaction, &context)
        .await
        .expect("the compaction sets");

    assert_eq!(
        harness
            .get_steering_mode(&context)
            .await
            .expect("the steering read"),
        QueueMode::All,
        "the seed steering mode",
    );
    harness
        .set_steering_mode(QueueMode::OneAtATime, &context)
        .await
        .expect("the steering mode sets");
    harness
        .set_follow_up_mode(QueueMode::OneAtATime, &context)
        .await
        .expect("the follow-up mode sets");
    assert_eq!(
        harness
            .get_follow_up_mode(&context)
            .await
            .expect("the follow-up read"),
        QueueMode::OneAtATime,
        "the follow-up mode round-trips",
    );

    let watched = harness.watch_session(&context).await;
    assert!(
        matches!(watched, Err(LaneOperationError::Closed(_))),
        "the watch-session slice raises its stub",
    );

    let subscription = harness.events().on(
        HarnessEventType::RunStart,
        Arc::new(|_event: &HarnessEvent, _context| Box::pin(async { Ok(()) })),
    );
    subscription.unsubscribe();
    let hook_subscription = harness.hooks().on(
        HookName::BeforeRun,
        Arc::new(|_invocation: &HookInvocation, _context: &Context| {
            Box::pin(async move { Ok(HookResult::BeforeRun(None)) })
        }),
        HookOptions::default(),
    );
    hook_subscription.unsubscribe();

    let rendered = format!("{concrete:?}");
    assert!(rendered.contains("Harness"), "the debug render: {rendered}");

    harness.close(&context).await.expect("the close settles");
    common::close_sessions(&mut sessions).await;
}

/// The probe tool the trait surface's tools round-trip carries, upstream's
/// `tools: [...]` config row: a no-executor declarative tool.
fn probe_tool() -> crate::harness::types::AgentHarnessTool {
    crate::harness::types::AgentHarnessTool {
        tool: pi_ai::types::Tool {
            name: "probe".to_owned(),
            description: "the probe".to_owned(),
            parameters: json!({ "type": "object" }),
            constrained_sampling: None,
        },
        label: "probe".to_owned(),
        prepare_arguments: None,
        execute: Arc::new(
            |_tool_call_id: &str,
             _args: &serde_json::Value,
             _update: Option<crate::harness::types::AgentHarnessToolUpdateCallback<'_>>,
             _tool_context: crate::harness::types::ToolContext,
             _invocation: &dyn crate::harness::types::AgentHarnessToolInvocation,
             _context: &Context|
             -> BoxedFuture<
                '_,
                Result<crate::types::AgentToolResult, crate::types::AgentToolError>,
            > { unreachable!("the probe never executes") },
        ),
        replay: None,
        execution_mode: None,
    }
}

//! The atomic run acceptance suite, ported 1:1 from upstream
//! `test/harness/runtime/accept.test.ts` ("runtime atomic run acceptance")
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements the port makes and the tests bind:
//! - upstream's `vi.spyOn(models, "getModel")` has no spy seam on the Rust
//!   models catalog; the assertion pins the observable the spy guards — the
//!   faux provider's `call_count` stays at zero across admission, so
//!   acceptance never resolved into a provider call.
//! - upstream's `sameContext` object-identity check restates as the
//!   context-key round-trip (the bus delivers each event with a clone of
//!   the caller's context, so the key's value is the identity witness).
//! - the `setTimeout(0)` tick restates as `test_support::settle_events()`.
//! - `FailingMemoryStorage` and `ControlledMemoryStorage` ride the shared
//!   [`crate::harness::runtime::test_support::ControlledStorage`].
//! - the concrete runtime `Lane` the tests assert state on is read out of
//!   the container's lane map (the child-module privacy seam); the
//!   `Harness::lane` acquisition handle stays on the trait surface.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the tests pin outcomes; unexpected results and violated expectations panic the test by design"
)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pi_ai::models::create_models;
use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxContentBlock;
use pi_ai::providers::faux::FauxProviderHandle;
use pi_ai::providers::faux::{RegisterFauxProviderOptions, faux_assistant_message, faux_provider};
use pi_ai::types::BoxedFuture;
use pi_ai::types::ImageContent;
use pi_ai::types::{Message, StopReason, TextContent, UserBlock, UserContent, UserMessage};

use super::common::SessionLedger;
use super::common::close_sessions;
use super::common::default_agent_options;
use crate::harness::agent_harness::AgentHarnessOptions;
use crate::harness::agent_harness::AgentLane;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventType;
use crate::harness::agent_harness::NavigateOptions;
use crate::harness::agent_harness::OperationAdmission;
use crate::harness::agent_harness::OperationAdmissionResult;
use crate::harness::agent_harness::OperationRequest;
use crate::harness::agent_harness::{OperationStatus, PromptMessagesPayload, Resources};
use crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS;
use crate::harness::context::{
    Context, background_context, create_context_key, with_context_value,
};
use crate::harness::result::{HarnessError, HarnessFault};
use crate::harness::runtime::harness::{Harness, create_agent_harness};
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::test_support::ControlledStorage;
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::deferred;
use crate::harness::runtime::test_support::gate_next_commit;
use crate::harness::runtime::test_support::lane_state_write;
use crate::harness::runtime::test_support::runtime_session;
use crate::harness::runtime::test_support::{runtime_session_metadata, settle_events};
use crate::harness::runtime::types::LaneError;
use crate::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use crate::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use crate::harness::session::testing::InstrumentedStorage;
use crate::harness::session::types::BranchScanOrder;
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::DurableStructuralPreparation;
use crate::harness::session::types::Entry;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::LaneState;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::ModelIdentity;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationKind;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::RunSettings;
use crate::harness::session::types::{Session, SessionError, SessionReader, StorageBranchScan};
use crate::harness::session::values as stored_values;
use crate::harness::session::values::{EntryWrite, Write, set_value_write};
use crate::harness::types::{PromptTemplate, Skill};
use crate::types::{AgentMessage, QueueMode, ThinkingLevel, ToolExecutionMode};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The lane configuration every fixture seeds, upstream's `configuration`
/// constant: `faux`/`faux-1`, `off`, and no active tools.
fn configuration() -> LaneConfiguration {
    LaneConfiguration {
        model: ModelIdentity {
            provider: "faux".to_owned(),
            model_id: "faux-1".to_owned(),
        },
        thinking_level: ThinkingLevel::Off,
        active_tool_names: Vec::new(),
    }
}

/// One plain user text message, upstream's `{ role: "user", content,
/// timestamp }` literals.
fn user_message(content: &str, timestamp: i64) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Text(content.to_owned()),
        timestamp,
    }))
}

/// The idle `main` lane's seed writes, upstream's `createSession` commit:
/// the branch tip at none, the lane configuration, and the empty lane
/// state.
fn accept_seed_writes() -> Result<Vec<Write>, SessionError> {
    Ok(vec![
        set_value_write(&stored_values::branch_tip("main"), Option::<String>::None)?,
        set_value_write(&stored_values::lane_config("main"), configuration())?,
        lane_state_write("main", None, None, Vec::new())?,
    ])
}

/// The pending message write the queue fixtures seed, upstream's
/// `setValue(pendingEntry(id), { type: "message", payload })`.
fn pending_message_write(
    entry_id: &str,
    content: &str,
    timestamp: i64,
) -> Result<Write, SessionError> {
    set_value_write(
        &stored_values::pending_entry(entry_id),
        PendingEntry::Message {
            payload: Box::new(user_message(content, timestamp)),
        },
    )
}

/// The pending custom write the queued-write fixtures seed, upstream's
/// `{ type: "custom", customType: "queued-write" }`.
fn pending_custom_write(entry_id: &str) -> Result<Write, SessionError> {
    set_value_write(
        &stored_values::pending_entry(entry_id),
        PendingEntry::Custom {
            custom_type: "queued-write".to_owned(),
            payload: None,
        },
    )
}

/// The transcript entry write the navigation fixtures seed, upstream's
/// `{ kind: "entry", entry }` commit members.
fn message_entry_write(id: &str, parent_id: Option<&str>, content: &str, timestamp: i64) -> Write {
    Write::Entry(Box::new(EntryWrite {
        kind: "entry".to_owned(),
        entry: NewEntry::Message {
            id: id.to_owned(),
            parent_id: parent_id.map(str::to_owned),
            body: Box::new(MessageEntry {
                message: user_message(content, timestamp),
                terminate: None,
            }),
        },
    }))
}

/// The session upstream's `createSession` builds over the instrumented
/// memory backend: id `accept-${sessions.length}`, ledgered, seeded with
/// the idle `main` lane. Returns the storage handle the write-family
/// assertions read.
async fn create_session(
    sessions: &mut SessionLedger,
) -> (Arc<StorageBackedSession>, Arc<InstrumentedStorage>) {
    let storage = Arc::new(InstrumentedStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = Arc::new(StorageBackedSession::new(
        runtime_session_metadata(format!("accept-{}", sessions.len())),
        storage.clone(),
        StorageBackedSessionOptions::default(),
    ));
    sessions.push(Arc::clone(&session));
    commit_writes(&session, accept_seed_writes().expect("the seed writes")).await;
    (session, storage)
}

/// The session variant the storage-fault and close-race tests drive, over
/// the controlled backend the test arms after the seed, upstream's
/// `FailingMemoryStorage` / `ControlledMemoryStorage` sessions.
async fn create_session_over(
    sessions: &mut SessionLedger,
    storage: Arc<ControlledStorage>,
    id: &str,
) -> Arc<StorageBackedSession> {
    let session = Arc::new(runtime_session(id.to_owned(), Arc::clone(&storage)));
    sessions.push(Arc::clone(&session));
    commit_writes(&session, accept_seed_writes().expect("the seed writes")).await;
    session
}

/// The harness options upstream's `options(session)` builds: a fresh faux
/// provider registered into a fresh models catalog, the first faux model,
/// and no further overrides (the seed configuration defaults to `off` and
/// no active tools).
fn accept_options(session: Arc<StorageBackedSession>) -> (AgentHarnessOptions, FauxProviderHandle) {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    let models = Arc::new(create_models(None));
    models.set_provider(Arc::new(faux.provider.clone()));
    let options =
        default_agent_options(session, Arc::clone(&models), faux.first_model(), None, None);
    (options, faux)
}

/// The pre-creation hook, upstream's `beforeCreate?: (session) => Promise<void>`:
/// the closure captures its seed writes and commits them against the
/// created session before the harness attaches.
type BeforeCreate = Box<dyn FnOnce(Arc<StorageBackedSession>) -> BoxedFuture<'static, ()>>;

/// The fixture upstream's `createHarness` returns: the harness, the
/// concrete runtime lane (read out of the container's lane map), the
/// session, the instrumented storage, and the faux provider handle (the
/// models catalog rides the options; the `getModel` spy restates on the
/// faux state).
struct AcceptFixture {
    harness: Harness,
    lane: Arc<Lane>,
    session: Arc<StorageBackedSession>,
    storage: Arc<InstrumentedStorage>,
    faux: FauxProviderHandle,
}

/// The concrete lane the container published for `name`, upstream's
/// `lane instanceof Lane` identity: the `Harness::lane` acquisition
/// returns the trait handle, and the suite reads the concrete lane out of
/// the container's lane map (the child-module privacy seam) for the state
/// and drive-pass assertions.
fn runtime_lane(harness: &Harness, name: &str) -> Arc<Lane> {
    let lanes = crate::harness::runtime::harness::lock_lanes(&harness.shared);
    Arc::clone(lanes.get(name).expect("the runtime lane publishes"))
}

/// Attaches the runtime over a fresh session, upstream's `createHarness`:
/// the session defaults to a ledgered instrumented one, `beforeCreate`
/// seeds before the attach, `resources` and the queue modes ride the
/// options, and the commit attempts clear after the lane publishes.
async fn create_harness(
    sessions: &mut SessionLedger,
    before_create: Option<BeforeCreate>,
    resources: Resources,
    steering_mode: Option<QueueMode>,
    follow_up_mode: Option<QueueMode>,
) -> AcceptFixture {
    let (session, storage) = create_session(sessions).await;
    if let Some(before_create) = before_create {
        before_create(Arc::clone(&session)).await;
    }
    let (options, faux) = accept_options(Arc::clone(&session));
    let options = AgentHarnessOptions {
        resources: Some(resources),
        steering_mode,
        follow_up_mode,
        ..options
    };
    let created = create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates");
    created
        .harness
        .lane("main", &background_context())
        .await
        .expect("the lane attaches");
    let lane = runtime_lane(&created.harness, "main");
    storage.clear_commit_attempts();
    AcceptFixture {
        harness: created.harness,
        lane,
        session,
        storage,
        faux,
    }
}

/// Upstream's `unwrap(result)` helper: the tagged admission error throws.
/// The outer error only carries a sealed lane, which the test panics on.
fn unwrap(admission: Result<OperationAdmissionResult, LaneError>) -> OperationAdmission {
    admission.expect("accept serves").expect("the admission")
}

/// The text-prompt request upstream's `{ kind: "prompt", prompt }` builds.
fn prompt_request(prompt: &str) -> OperationRequest {
    OperationRequest::Prompt {
        operation_id: None,
        prompt: Box::new(PromptMessagesPayload::Text {
            prompt: prompt.to_owned(),
            images: None,
        }),
    }
}

/// The prebuilt-messages request upstream's `{ kind: "prompt",
/// operationId, prompt: AgentMessage[] }` builds.
fn prompt_messages_request(operation_id: &str, messages: Vec<AgentMessage>) -> OperationRequest {
    OperationRequest::Prompt {
        operation_id: Some(operation_id.to_owned()),
        prompt: Box::new(PromptMessagesPayload::Messages(messages)),
    }
}

/// The compaction request upstream's `{ kind: "compaction", ... }` builds.
fn compaction_request(
    operation_id: Option<&str>,
    custom_instructions: Option<&str>,
) -> OperationRequest {
    OperationRequest::Compaction {
        operation_id: operation_id.map(str::to_owned),
        custom_instructions: custom_instructions.map(str::to_owned),
    }
}

/// The navigation request upstream's `{ kind: "navigation", ... }` builds.
fn navigation_request(
    operation_id: Option<&str>,
    target_id: Option<&str>,
    options: Option<NavigateOptions>,
) -> OperationRequest {
    OperationRequest::Navigation {
        operation_id: operation_id.map(str::to_owned),
        target_id: target_id.map(str::to_owned),
        options,
    }
}

/// The skill request upstream's `{ kind: "skill", ... }` builds.
fn skill_request(name: &str, additional_instructions: Option<&str>) -> OperationRequest {
    OperationRequest::Skill {
        operation_id: None,
        name: name.to_owned(),
        additional_instructions: additional_instructions.map(str::to_owned),
    }
}

/// The template request upstream's `{ kind: "prompt_template", ... }`
/// builds.
fn template_request(name: &str, args: &[&str]) -> OperationRequest {
    OperationRequest::PromptTemplate {
        operation_id: None,
        name: name.to_owned(),
        args: Some(args.iter().map(|arg| (*arg).to_owned()).collect()),
    }
}

/// The write-family label upstream's
/// `write.kind === "value" ? `${write.kind}:${write.op}` : write.kind`
/// renders.
fn write_family(write: &Write) -> String {
    match write {
        Write::Entry(_) => "entry".to_owned(),
        Write::Usage(_) => "usage".to_owned(),
        Write::ValueSet(write) => format!("{}:{}", write.kind, write.op),
        Write::ValueDelete(write) => format!("{}:{}", write.kind, write.op),
        Write::ListAppend(write) => format!("{}:{}", write.kind, write.op),
        Write::ListDelete(write) => format!("{}:{}", write.kind, write.op),
    }
}

/// The write families of the one commit attempt upstream's
/// `storage.getCommitAttempts()[0]` reads.
fn commit_families(storage: &InstrumentedStorage) -> Vec<String> {
    let attempts = storage.get_commit_attempts();
    assert_eq!(attempts.len(), 1, "exactly one commit attempt");
    attempts[0].iter().map(write_family).collect()
}

/// Upstream's `it.each` table over the normalized prompt shapes, wire
/// names `text`, `images`, and `text and images`.
#[tokio::test]
async fn accepts_normalized_text_images_and_mixed_prompts_into_starting() {
    let image = ImageContent {
        data: "aW1hZ2U=".to_owned(),
        mime_type: "image/png".to_owned(),
    };
    for (_name, request, expected_content) in [
        (
            "text",
            prompt_request("hello"),
            vec![UserBlock::Text(TextContent {
                text: "hello".to_owned(),
                text_signature: None,
            })],
        ),
        (
            "images",
            OperationRequest::Prompt {
                operation_id: None,
                prompt: Box::new(PromptMessagesPayload::Text {
                    prompt: String::new(),
                    images: Some(vec![image.clone()]),
                }),
            },
            vec![UserBlock::Image(image.clone())],
        ),
        (
            "text and images",
            OperationRequest::Prompt {
                operation_id: None,
                prompt: Box::new(PromptMessagesPayload::Text {
                    prompt: "hello".to_owned(),
                    images: Some(vec![image.clone()]),
                }),
            },
            vec![
                UserBlock::Text(TextContent {
                    text: "hello".to_owned(),
                    text_signature: None,
                }),
                UserBlock::Image(image),
            ],
        ),
    ] {
        let mut sessions = SessionLedger::default();
        let fixture = create_harness(&mut sessions, None, Resources::default(), None, None).await;

        let admission = unwrap(
            fixture
                .lane
                .accept_impl(request, &background_context())
                .await,
        );
        let operation = fixture
            .lane
            .state()
            .operation
            .expect("the admitted operation");
        let OperationState::Starting(starting) = &operation.state else {
            panic!("Expected accepted run: {:?}", operation.state)
        };
        let OperationIntent::Run { prompt_entry_ids } = &operation.meta.intent else {
            panic!("Expected prompt entry: {:?}", operation.meta.intent)
        };
        let entry_id = prompt_entry_ids.first().expect("Expected prompt entry");
        let entry = fixture
            .session
            .get_entry(entry_id, &background_context())
            .await
            .expect("the entry read")
            .expect("the prompt entry");

        assert_eq!(
            admission.operation_id, operation.meta.operation_id,
            "the admitted operation id"
        );
        assert_eq!(admission.kind, OperationKind::Run, "the admitted kind");
        let Entry::Message { body, .. } = &entry else {
            panic!("the message entry: {entry:?}")
        };
        let AgentMessage::Standard(Message::User(user)) = &body.message else {
            panic!("the user prompt: {:?}", body.message)
        };
        assert_eq!(
            user.content,
            UserContent::Blocks(expected_content),
            "the normalized prompt content"
        );
        assert_eq!(
            starting.scope.settings,
            RunSettings {
                compaction: DEFAULT_COMPACTION_SETTINGS,
                steering_mode: QueueMode::All,
                follow_up_mode: QueueMode::All,
                tool_execution: ToolExecutionMode::Parallel,
            },
            "the accepted settings"
        );
        close_sessions(&mut sessions).await;
    }
}

/// The supplied message array replays verbatim: one commit, the exact
/// write families, the caller's context on every event, and no model
/// resolution.
#[expect(
    clippy::too_many_lines,
    reason = "the case pins the write families, the event list, and the meta in one flow"
)]
#[tokio::test]
async fn preserves_supplied_message_arrays_and_commits_the_exact_acceptance_write_families_once() {
    let mut sessions = SessionLedger::default();
    let fixture = create_harness(&mut sessions, None, Resources::default(), None, None).await;
    let messages = vec![user_message("one", 10), user_message("two", 11)];
    let context_key = create_context_key::<String>("accept.context");
    let context = with_context_value(&context_key, "source".to_owned(), &background_context());
    let seen = Arc::new(Mutex::new(Vec::<(String, bool)>::new()));
    for event_type in [
        HarnessEventType::RunStart,
        HarnessEventType::MessageStart,
        HarnessEventType::MessageEnd,
        HarnessEventType::EntryAdded,
    ] {
        let seen = Arc::clone(&seen);
        let context_key = context_key.clone();
        fixture
            .harness
            .events()
            .on(
                event_type,
                Arc::new(move |event: &HarnessEvent, context: &Context| {
                    let seen = Arc::clone(&seen);
                    let context_key = context_key.clone();
                    let context = context.clone();
                    Box::pin(async move {
                        let same_context =
                            context.value(&context_key).map(String::as_str) == Some("source");
                        lock(&seen).push((event.event_type().as_str().to_owned(), same_context));
                        Ok(())
                    })
                }),
            )
            .expect("the listener registers");
    }

    let admission = unwrap(
        fixture
            .lane
            .accept_impl(prompt_messages_request("operation", messages), &context)
            .await,
    );

    assert_eq!(
        admission.operation_id, "operation",
        "the pinned operation id"
    );
    assert_eq!(admission.kind, OperationKind::Run, "the admitted kind");
    assert!(
        admission.started_at > 0,
        "startedAt is a provider-clock number"
    );
    assert_eq!(
        commit_families(&fixture.storage),
        [
            "entry",
            "entry",
            "value:set",
            "value:set",
            "value:set",
            "value:set"
        ],
        "the acceptance write families"
    );
    let operation = fixture
        .lane
        .state()
        .operation
        .expect("the admitted operation");
    assert_eq!(
        operation.meta.operation_id, "operation",
        "the meta operation id"
    );
    assert_eq!(operation.meta.lane, "main", "the meta lane");
    assert!(operation.meta.source_tip_id.is_none(), "the source tip id");
    let OperationIntent::Run { prompt_entry_ids } = &operation.meta.intent else {
        panic!("Expected run metadata: {:?}", operation.meta.intent)
    };
    assert_eq!(prompt_entry_ids.len(), 2, "the two prompt entries");
    let tip = fixture
        .lane
        .get_tip_id(&background_context())
        .await
        .expect("the tip read");
    assert_eq!(
        tip.as_deref(),
        prompt_entry_ids.get(1).map(String::as_str),
        "the tip is the last prompt entry"
    );
    let seen = lock(&seen).clone();
    let types: Vec<&str> = seen
        .iter()
        .map(|(event_type, _)| event_type.as_str())
        .collect();
    assert_eq!(
        types,
        [
            "run_start",
            "message_start",
            "message_end",
            "entry_added",
            "message_start",
            "message_end",
            "entry_added",
        ],
        "the delivered events"
    );
    assert!(
        seen.iter().all(|(_, same_context)| *same_context),
        "every event carries the caller's context"
    );
    // The `getModel` spy restates as the provider's dispatch count: the
    // admission must not have resolved the model into a provider call.
    assert_eq!(
        fixture.faux.state().call_count(),
        0,
        "getModel never resolved into a provider call"
    );
    close_sessions(&mut sessions).await;
}

/// The queued next-run messages replay before the request messages, and an
/// otherwise empty request admits on their strength alone.
#[expect(
    clippy::too_many_lines,
    reason = "the case pins the write families, the drained branch, and the pending deletes in one flow"
)]
#[tokio::test]
async fn captures_next_run_messages_before_request_messages_and_accepts_an_otherwise_empty_request()
{
    let first = "pending-first";
    let second = "pending-second";
    let mut sessions = SessionLedger::default();
    let fixture = create_harness(
        &mut sessions,
        Some(Box::new(move |session: Arc<StorageBackedSession>| {
            Box::pin(async move {
                commit_writes(
                    &session,
                    vec![
                        pending_message_write(first, "first", 1).expect("the pending write"),
                        pending_message_write(second, "second", 2).expect("the pending write"),
                        lane_state_write(
                            "main",
                            None,
                            None,
                            vec![
                                InboxItem {
                                    entry_id: first.to_owned(),
                                    kind: InboxItemKind::NextRun,
                                },
                                InboxItem {
                                    entry_id: second.to_owned(),
                                    kind: InboxItemKind::NextRun,
                                },
                            ],
                        )
                        .expect("the lane state write"),
                    ],
                )
                .await;
            })
        })),
        Resources::default(),
        None,
        None,
    )
    .await;
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = Arc::clone(&events);
    fixture
        .harness
        .events()
        .on(
            HarnessEventType::QueueUpdate,
            Arc::new(move |event: &HarnessEvent, _context| {
                let sink = Arc::clone(&sink);
                let owned = event.event_type().as_str().to_owned();
                Box::pin(async move {
                    lock(&sink).push(owned);
                    Ok(())
                })
            }),
        )
        .expect("the listener registers");

    unwrap(
        fixture
            .lane
            .accept_impl(prompt_request(""), &background_context())
            .await,
    );

    let operation = fixture
        .lane
        .state()
        .operation
        .expect("the admitted operation");
    let OperationIntent::Run { prompt_entry_ids } = &operation.meta.intent else {
        panic!("Expected run metadata: {:?}", operation.meta.intent)
    };
    assert!(
        matches!(&operation.state, OperationState::Starting(_)),
        "Expected accepted run"
    );
    assert!(
        prompt_entry_ids.is_empty(),
        "the request messages stayed empty"
    );
    assert!(operation.meta.source_tip_id.is_none(), "the source tip id");
    assert!(fixture.lane.state().inbox.is_empty(), "the inbox drained");
    let entries = fixture
        .session
        .scan_branch(
            &StorageBranchScan {
                start: second.to_owned(),
                order: Some(BranchScanOrder::OldestFirst),
                ..StorageBranchScan::default()
            },
            &background_context(),
        )
        .await
        .expect("the branch scan");
    let ids: Vec<&str> = entries.iter().map(Entry::id).collect();
    assert_eq!(ids, [first, second], "the replayed branch order");
    for entry_id in [first, second] {
        assert!(
            fixture
                .session
                .get_value(
                    &stored_values::pending_entry(entry_id).address,
                    &background_context()
                )
                .await
                .expect("the pending read")
                .is_none(),
            "the consumed pending entry {entry_id}"
        );
    }
    assert_eq!(
        commit_families(&fixture.storage),
        [
            "entry",
            "entry",
            "value:delete",
            "value:delete",
            "value:set",
            "value:set",
            "value:set",
            "value:set",
        ],
        "the acceptance write families"
    );
    assert_eq!(
        lock(&events).clone(),
        ["queue_update"],
        "the drained queues published once"
    );
    close_sessions(&mut sessions).await;
}

/// One-at-a-time modes admit one steer and one follow-up beside the writes
/// and next-runs, in admission order ahead of the request messages, and
/// leave the rest queued.
#[expect(
    clippy::too_many_lines,
    reason = "the case pins the admission order, the residual inbox, and the pending reads in one flow"
)]
#[tokio::test]
async fn captures_every_eligible_tag_by_mode_and_preserves_admission_order_before_the_request() {
    struct QueueIds {
        write: &'static str,
        next: &'static str,
        steer1: &'static str,
        steer2: &'static str,
        follow1: &'static str,
        follow2: &'static str,
    }
    let ids = QueueIds {
        write: "queued-write",
        next: "queued-next",
        steer1: "queued-steer-1",
        steer2: "queued-steer-2",
        follow1: "queued-follow-1",
        follow2: "queued-follow-2",
    };
    let mut sessions = SessionLedger::default();
    let fixture = create_harness(
        &mut sessions,
        Some(Box::new(move |session: Arc<StorageBackedSession>| {
            Box::pin(async move {
                let mut writes = vec![pending_custom_write(ids.write).expect("the pending write")];
                for entry_id in [ids.next, ids.steer1, ids.steer2, ids.follow1, ids.follow2] {
                    writes.push(
                        pending_message_write(entry_id, entry_id, 1).expect("the pending write"),
                    );
                }
                writes.push(
                    lane_state_write(
                        "main",
                        None,
                        None,
                        vec![
                            InboxItem {
                                entry_id: ids.steer1.to_owned(),
                                kind: InboxItemKind::Steer,
                            },
                            InboxItem {
                                entry_id: ids.write.to_owned(),
                                kind: InboxItemKind::Write,
                            },
                            InboxItem {
                                entry_id: ids.next.to_owned(),
                                kind: InboxItemKind::NextRun,
                            },
                            InboxItem {
                                entry_id: ids.follow1.to_owned(),
                                kind: InboxItemKind::FollowUp,
                            },
                            InboxItem {
                                entry_id: ids.steer2.to_owned(),
                                kind: InboxItemKind::Steer,
                            },
                            InboxItem {
                                entry_id: ids.follow2.to_owned(),
                                kind: InboxItemKind::FollowUp,
                            },
                        ],
                    )
                    .expect("the lane state write"),
                );
                commit_writes(&session, writes).await;
            })
        })),
        Resources::default(),
        Some(QueueMode::OneAtATime),
        Some(QueueMode::OneAtATime),
    )
    .await;

    unwrap(
        fixture
            .lane
            .accept_impl(prompt_request("request"), &background_context())
            .await,
    );

    let tip_id = fixture.lane.state().tip_id.expect("Expected accepted tip");
    let entries = fixture
        .session
        .scan_branch(
            &StorageBranchScan {
                start: tip_id.clone(),
                order: Some(BranchScanOrder::OldestFirst),
                ..StorageBranchScan::default()
            },
            &background_context(),
        )
        .await
        .expect("the branch scan");
    let branch: Vec<&str> = entries.iter().map(Entry::id).collect();
    assert_eq!(
        branch,
        [
            ids.steer1,
            ids.write,
            ids.next,
            ids.follow1,
            tip_id.as_str()
        ],
        "the admission order before the request"
    );
    assert_eq!(
        fixture.lane.state().inbox,
        vec![
            InboxItem {
                entry_id: ids.steer2.to_owned(),
                kind: InboxItemKind::Steer,
            },
            InboxItem {
                entry_id: ids.follow2.to_owned(),
                kind: InboxItemKind::FollowUp,
            },
        ],
        "the residual inbox"
    );
    let durable = crate::harness::runtime::test_support::stored_value(
        &fixture.session,
        &stored_values::lane_state("main").address,
        &background_context(),
    )
    .await;
    let durable: LaneState = serde_json::from_value(durable.value).expect("the lane state wire");
    assert_eq!(
        durable.inbox,
        fixture.lane.state().inbox,
        "the durable inbox"
    );
    for entry_id in [ids.write, ids.next, ids.steer1, ids.follow1] {
        assert!(
            fixture
                .session
                .get_value(
                    &stored_values::pending_entry(entry_id).address,
                    &background_context()
                )
                .await
                .expect("the pending read")
                .is_none(),
            "the consumed pending entry {entry_id}"
        );
    }
    for entry_id in [ids.steer2, ids.follow2] {
        assert!(
            fixture
                .session
                .get_value(
                    &stored_values::pending_entry(entry_id).address,
                    &background_context()
                )
                .await
                .expect("the pending read")
                .is_some(),
            "the residual pending entry {entry_id}"
        );
    }
    close_sessions(&mut sessions).await;
}

/// A queued write alone does not validate an empty acceptance; the
/// rejection leaves the inbox and the pending entry intact.
#[tokio::test]
async fn does_not_let_a_lone_queued_write_validate_empty_acceptance() {
    let entry_id = "queued-write";
    let mut sessions = SessionLedger::default();
    let fixture = create_harness(
        &mut sessions,
        Some(Box::new(move |session: Arc<StorageBackedSession>| {
            Box::pin(async move {
                commit_writes(
                    &session,
                    vec![
                        pending_custom_write(entry_id).expect("the pending write"),
                        lane_state_write(
                            "main",
                            None,
                            None,
                            vec![InboxItem {
                                entry_id: entry_id.to_owned(),
                                kind: InboxItemKind::Write,
                            }],
                        )
                        .expect("the lane state write"),
                    ],
                )
                .await;
            })
        })),
        Resources::default(),
        None,
        None,
    )
    .await;

    let result = fixture
        .lane
        .accept_impl(prompt_request(""), &background_context())
        .await
        .expect("accept serves");

    let error = result.expect_err("the lone write does not validate");
    assert!(
        matches!(&error, HarnessError::InvalidMessage { reason, .. } if reason == "empty"),
        "{error:?}"
    );
    assert_eq!(
        fixture.lane.state().inbox,
        vec![InboxItem {
            entry_id: entry_id.to_owned(),
            kind: InboxItemKind::Write,
        }],
        "the inbox survived"
    );
    assert!(
        fixture
            .session
            .get_value(
                &stored_values::pending_entry(entry_id).address,
                &background_context()
            )
            .await
            .expect("the pending read")
            .is_some(),
        "the pending entry survived"
    );
    close_sessions(&mut sessions).await;
}

/// Skills and templates format into the accepted prompt entries.
#[tokio::test]
async fn formats_skills_and_templates_before_acceptance() {
    let resources = Resources {
        skills: vec![Skill {
            name: "review".to_owned(),
            description: "Review".to_owned(),
            content: "Inspect it".to_owned(),
            file_path: "/skills/review/SKILL.md".to_owned(),
            disable_model_invocation: None,
        }],
        prompt_templates: vec![PromptTemplate {
            name: "fix".to_owned(),
            description: None,
            content: "Fix $1 then $@".to_owned(),
        }],
    };
    let mut sessions = SessionLedger::default();
    let skill = create_harness(&mut sessions, None, resources.clone(), None, None).await;
    unwrap(
        skill
            .lane
            .accept_impl(
                skill_request("review", Some("Be strict")),
                &background_context(),
            )
            .await,
    );
    let skill_tip = skill.lane.state().tip_id.expect("the skill tip");
    let skill_entry = skill
        .session
        .get_entry(&skill_tip, &background_context())
        .await
        .expect("the entry read")
        .expect("the skill entry");
    let Entry::Message { body, .. } = &skill_entry else {
        panic!("the message entry: {skill_entry:?}")
    };
    let AgentMessage::Standard(Message::User(user)) = &body.message else {
        panic!("the user prompt: {:?}", body.message)
    };
    let UserContent::Blocks(blocks) = &user.content else {
        panic!("the blocks content: {:?}", user.content)
    };
    let Some(UserBlock::Text(text)) = blocks.first() else {
        panic!("the text block: {blocks:?}")
    };
    assert!(
        text.text.contains("<skill name=\"review\""),
        "the formatted skill invocation: {}",
        text.text
    );

    let template = create_harness(&mut sessions, None, resources, None, None).await;
    unwrap(
        template
            .lane
            .accept_impl(template_request("fix", &["A", "B"]), &background_context())
            .await,
    );
    let template_tip = template.lane.state().tip_id.expect("the template tip");
    let template_entry = template
        .session
        .get_entry(&template_tip, &background_context())
        .await
        .expect("the entry read")
        .expect("the template entry");
    let Entry::Message { body, .. } = &template_entry else {
        panic!("the message entry: {template_entry:?}")
    };
    let AgentMessage::Standard(Message::User(user)) = &body.message else {
        panic!("the user prompt: {:?}", body.message)
    };
    let UserContent::Blocks(blocks) = &user.content else {
        panic!("the blocks content: {:?}", user.content)
    };
    let Some(UserBlock::Text(text)) = blocks.first() else {
        panic!("the text block: {blocks:?}")
    };
    assert_eq!(text.text, "Fix A then A B", "the formatted template");
    close_sessions(&mut sessions).await;
}

/// The standalone compaction admission lands at `summary.deciding` with
/// the durable preparation, one commit, one `compaction_start`, and no
/// installed drive.
#[tokio::test]
async fn accepts_standalone_compaction_with_durable_preparation_and_no_execution() {
    let mut sessions = SessionLedger::default();
    let fixture = create_harness(&mut sessions, None, Resources::default(), None, None).await;
    fixture
        .lane
        .append_message(user_message("history", 1), &background_context())
        .await
        .expect("the append serves");
    fixture.storage.clear_commit_attempts();
    let starts = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&starts);
    fixture
        .harness
        .events()
        .on(
            HarnessEventType::CompactionStart,
            Arc::new(move |_event: &HarnessEvent, _context| {
                counter.fetch_add(1, Ordering::SeqCst);
                Box::pin(std::future::ready(Ok(())))
            }),
        )
        .expect("the listener registers");

    let admission = unwrap(
        fixture
            .lane
            .accept_impl(
                compaction_request(Some("compaction"), Some("focus")),
                &background_context(),
            )
            .await,
    );

    assert_eq!(
        admission.operation_id, "compaction",
        "the pinned operation id"
    );
    assert_eq!(
        admission.kind,
        OperationKind::Compaction,
        "the admitted kind"
    );
    assert!(
        admission.started_at > 0,
        "startedAt is a provider-clock number"
    );
    let operation = fixture
        .lane
        .state()
        .operation
        .expect("the admitted operation");
    let OperationState::SummaryDeciding(deciding) = &operation.state else {
        panic!("Expected accepted compaction: {:?}", operation.state)
    };
    assert_eq!(
        deciding.task.reason,
        Some(CompactionReason::Manual),
        "the task reason"
    );
    assert_eq!(
        deciding.task.custom_instructions.as_deref(),
        Some("focus"),
        "the task instructions"
    );
    assert_eq!(
        deciding.task.boundary,
        ResultBoundary::Finish,
        "the task boundary"
    );
    let preparation = fixture
        .session
        .get_value(
            &stored_values::operation_preparation("compaction", &deciding.task.task_id).address,
            &background_context(),
        )
        .await
        .expect("the preparation read")
        .expect("the stored preparation");
    let preparation: DurableStructuralPreparation =
        serde_json::from_value(preparation.value).expect("the preparation wire");
    assert!(
        matches!(preparation, DurableStructuralPreparation::Compaction { .. }),
        "the durable preparation: {preparation:?}"
    );
    assert_eq!(
        fixture.storage.get_commit_attempts().len(),
        1,
        "one commit attempt"
    );
    assert_eq!(starts.load(Ordering::SeqCst), 1, "one compaction_start");
    assert!(fixture.lane.active_drive().is_none(), "no installed drive");
    close_sessions(&mut sessions).await;
}

/// The empty standalone compaction rejects before any write.
#[tokio::test]
async fn rejects_empty_standalone_compaction_without_writing() {
    let mut sessions = SessionLedger::default();
    let fixture = create_harness(&mut sessions, None, Resources::default(), None, None).await;

    let result = fixture
        .lane
        .accept_impl(compaction_request(None, None), &background_context())
        .await
        .expect("accept serves");

    let error = result.expect_err("the empty history rejects");
    assert_eq!(error.tag(), "NothingToCompact", "{error:?}");
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "no writes"
    );
    close_sessions(&mut sessions).await;
}

/// Upstream's `it.each([false, true])` table: the direct navigation lands
/// at `navigation.ready_to_commit` and the summarized one at
/// `summary.deciding` with the branch-summary preparation.
#[expect(
    clippy::too_many_lines,
    reason = "the reachability table is one loop the two summarize rows drive case by case"
)]
#[tokio::test]
async fn accepts_false_and_true_summarized_navigation_atomically() {
    for summarize in [false, true] {
        let mut sessions = SessionLedger::default();
        let fixture = create_harness(
            &mut sessions,
            Some(Box::new(|session: Arc<StorageBackedSession>| {
                Box::pin(async move {
                    commit_writes(
                        &session,
                        vec![
                            message_entry_write("root", None, "root", 1),
                            message_entry_write("source", Some("root"), "source", 2),
                            message_entry_write("target", Some("root"), "target", 3),
                            set_value_write(
                                &stored_values::branch_tip("main"),
                                Some("source".to_owned()),
                            )
                            .expect("the tip write"),
                        ],
                    )
                    .await;
                })
            })),
            Resources::default(),
            None,
            None,
        )
        .await;
        fixture.storage.clear_commit_attempts();
        let starts = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&starts);
        fixture
            .harness
            .events()
            .on(
                HarnessEventType::NavigationStart,
                Arc::new(move |_event: &HarnessEvent, _context| {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Box::pin(std::future::ready(Ok(())))
                }),
            )
            .expect("the listener registers");

        unwrap(
            fixture
                .lane
                .accept_impl(
                    navigation_request(
                        Some(if summarize { "summarized" } else { "direct" }),
                        Some("target"),
                        Some(NavigateOptions {
                            summarize: Some(summarize),
                            label: Some("chosen".to_owned()),
                            custom_instructions: Some("focus".to_owned()),
                        }),
                    ),
                    &background_context(),
                )
                .await,
        );

        let operation = fixture
            .lane
            .state()
            .operation
            .expect("Expected accepted navigation");
        if summarize {
            let OperationState::SummaryDeciding(deciding) = &operation.state else {
                panic!("Expected summary decision: {:?}", operation.state)
            };
            let preparation = fixture
                .session
                .get_value(
                    &stored_values::operation_preparation(
                        &operation.meta.operation_id,
                        &deciding.task.task_id,
                    )
                    .address,
                    &background_context(),
                )
                .await
                .expect("the preparation read")
                .expect("the stored preparation");
            let preparation: DurableStructuralPreparation =
                serde_json::from_value(preparation.value).expect("the preparation wire");
            let DurableStructuralPreparation::BranchSummary { messages, .. } = preparation else {
                panic!("the branch summary preparation")
            };
            let AgentMessage::Standard(Message::User(source)) = &messages[0] else {
                panic!("the summarized source message: {:?}", messages[0])
            };
            assert_eq!(
                source.content,
                UserContent::Text("source".to_owned()),
                "the summarized source messages"
            );
        } else {
            assert!(
                matches!(&operation.state, OperationState::NavigationReadyToCommit(_)),
                "the direct navigation leaf: {:?}",
                operation.state
            );
        }
        assert_eq!(
            fixture.lane.state().tip_id.as_deref(),
            Some("source"),
            "the tip stays"
        );
        assert_eq!(
            fixture.storage.get_commit_attempts().len(),
            1,
            "one commit attempt"
        );
        assert_eq!(starts.load(Ordering::SeqCst), 1, "one navigation_start");
        assert!(fixture.lane.active_drive().is_none(), "no installed drive");
        close_sessions(&mut sessions).await;
    }
}

/// The navigation target and options validate before any write.
#[tokio::test]
async fn validates_navigation_before_acceptance() {
    let mut sessions = SessionLedger::default();
    let fixture = create_harness(
        &mut sessions,
        Some(Box::new(|session: Arc<StorageBackedSession>| {
            Box::pin(async move {
                commit_writes(
                    &session,
                    vec![
                        message_entry_write("source", None, "source", 1),
                        set_value_write(
                            &stored_values::branch_tip("main"),
                            Some("source".to_owned()),
                        )
                        .expect("the tip write"),
                    ],
                )
                .await;
            })
        })),
        Resources::default(),
        None,
        None,
    )
    .await;

    let current_tip = fixture
        .lane
        .accept_impl(
            navigation_request(None, Some("source"), None),
            &background_context(),
        )
        .await
        .expect("accept serves");
    let error = current_tip.expect_err("the current tip rejects");
    assert!(
        matches!(&error, HarnessError::InvalidNavigation { reason, .. } if reason == "current_tip"),
        "{error:?}"
    );
    let root_label = fixture
        .lane
        .accept_impl(
            navigation_request(
                None,
                None,
                Some(NavigateOptions {
                    summarize: None,
                    label: Some("bad".to_owned()),
                    custom_instructions: None,
                }),
            ),
            &background_context(),
        )
        .await
        .expect("accept serves");
    let error = root_label.expect_err("the root label rejects");
    assert!(
        matches!(&error, HarnessError::InvalidNavigation { reason, .. } if reason == "root_label"),
        "{error:?}"
    );
    let missing = fixture
        .lane
        .accept_impl(
            navigation_request(None, Some("missing"), None),
            &background_context(),
        )
        .await
        .expect("accept serves");
    let error = missing.expect_err("the missing target rejects");
    assert_eq!(error.tag(), "UnknownTarget", "{error:?}");
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "no writes"
    );
    close_sessions(&mut sessions).await;
}

/// The expected pre-acceptance rejections return without writing.
#[tokio::test]
async fn returns_expected_pre_acceptance_errors_without_writing() {
    let mut sessions = SessionLedger::default();
    let fixture = create_harness(&mut sessions, None, Resources::default(), None, None).await;
    let pending = AgentMessage::Standard(Message::Assistant(faux_assistant_message(
        Vec::<FauxContentBlock>::new(),
        FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::Pending),
            ..FauxAssistantMessageOptions::default()
        },
    )));

    let empty = fixture
        .lane
        .accept_impl(prompt_request(""), &background_context())
        .await
        .expect("accept serves");
    let error = empty.expect_err("the empty prompt rejects");
    assert!(
        matches!(&error, HarnessError::InvalidMessage { reason, .. } if reason == "empty"),
        "{error:?}"
    );
    let pending_result = fixture
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
    let error = pending_result.expect_err("the pending assistant rejects");
    assert!(
        matches!(&error, HarnessError::InvalidMessage { reason, .. } if reason == "pending_assistant"),
        "{error:?}"
    );
    let skill = fixture
        .lane
        .accept_impl(skill_request("missing", None), &background_context())
        .await
        .expect("accept serves");
    let error = skill.expect_err("the missing skill rejects");
    assert_eq!(error.tag(), "UnknownSkill", "{error:?}");
    let template = fixture
        .lane
        .accept_impl(template_request("missing", &[]), &background_context())
        .await
        .expect("accept serves");
    let error = template.expect_err("the missing template rejects");
    assert_eq!(error.tag(), "UnknownTemplate", "{error:?}");
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "no writes"
    );
    close_sessions(&mut sessions).await;
}

/// Two concurrent accepts serialize on the lane line so exactly one wins.
#[tokio::test]
async fn serializes_concurrent_accepts_so_exactly_one_wins() {
    let mut sessions = SessionLedger::default();
    let fixture = create_harness(&mut sessions, None, Resources::default(), None, None).await;

    let first = {
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move {
            lane.accept_impl(prompt_request("first"), &background_context())
                .await
        })
    };
    let second = {
        let lane = Arc::clone(&fixture.lane);
        tokio::spawn(async move {
            lane.accept_impl(prompt_request("second"), &background_context())
                .await
        })
    };
    let first = first
        .await
        .expect("the first accept joins")
        .expect("accept serves");
    let second = second
        .await
        .expect("the second accept joins")
        .expect("accept serves");

    let winners = [&first, &second]
        .iter()
        .filter(|result| result.is_ok())
        .count();
    assert_eq!(winners, 1, "exactly one admission wins");
    let loser = if first.is_err() { &first } else { &second };
    let error = loser.as_ref().expect_err("the busy lane rejects");
    assert_eq!(error.tag(), "LaneBusy", "{error:?}");
    assert_eq!(
        fixture.storage.get_commit_attempts().len(),
        1,
        "one commit attempt"
    );
    close_sessions(&mut sessions).await;
}

/// A commit failure faults the acceptance without publishing state.
#[tokio::test]
async fn faults_on_commit_failure_without_publishing_acceptance() {
    let mut sessions = SessionLedger::default();
    let storage = Arc::new(ControlledStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = create_session_over(&mut sessions, Arc::clone(&storage), "failing").await;
    let (options, _faux) = accept_options(Arc::clone(&session));
    let created = create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates");
    created
        .harness
        .lane("main", &background_context())
        .await
        .expect("the lane attaches");
    let lane = runtime_lane(&created.harness, "main");
    storage.set_failure(Some(SessionError::Message("accept failed".to_owned())));

    let result = lane
        .accept_impl(prompt_request("hello"), &background_context())
        .await;

    let fault = result.expect_err("the failed commit faults the acceptance");
    let fault = fault
        .as_ref()
        .downcast_ref::<HarnessFault>()
        .expect("the rejection is a HarnessFault");
    assert_eq!(
        fault.message, "AgentHarness storage or invariant fault",
        "the fault message"
    );
    assert!(lane.state().operation.is_none(), "nothing published");
    close_sessions(&mut sessions).await;
}

/// The acceptance listeners run after the state publishes and their reads
/// serialize against the lane line.
#[tokio::test]
async fn delivers_acceptance_listeners_after_publishing_state_and_permits_serialized_reads() {
    let mut sessions = SessionLedger::default();
    let fixture = create_harness(&mut sessions, None, Resources::default(), None, None).await;
    let inspected = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&inspected);
    let listener_lane = Arc::clone(&fixture.lane);
    fixture
        .harness
        .events()
        .on(
            HarnessEventType::RunStart,
            Arc::new(move |_event: &HarnessEvent, context: &Context| {
                let lane = Arc::clone(&listener_lane);
                let flag = Arc::clone(&flag);
                let context = context.clone();
                Box::pin(async move {
                    let execution = lane
                        .inspect_execution_impl(&context)
                        .await
                        .expect("the inspection serves");
                    flag.store(
                        execution
                            .current
                            .is_some_and(|current| current.status == OperationStatus::Open),
                        Ordering::SeqCst,
                    );
                    Ok(())
                })
            }),
        )
        .expect("the listener registers");

    unwrap(
        fixture
            .lane
            .accept_impl(prompt_request("hello"), &background_context())
            .await,
    );

    assert!(
        inspected.load(Ordering::SeqCst),
        "the listener read the open execution"
    );
    close_sessions(&mut sessions).await;
}

/// The acceptance does not resolve until its direct listeners settle.
#[tokio::test]
async fn does_not_resolve_acceptance_before_its_direct_listeners_settle() {
    let mut sessions = SessionLedger::default();
    let fixture = create_harness(&mut sessions, None, Resources::default(), None, None).await;
    let (started_tx, started_rx) = deferred();
    let (release_tx, release_rx) = deferred();
    let gate = Arc::new(Mutex::new(Some((started_tx, release_rx))));
    fixture
        .harness
        .events()
        .on(
            HarnessEventType::RunStart,
            Arc::new(move |_event: &HarnessEvent, _context| {
                let gate = Arc::clone(&gate);
                Box::pin(async move {
                    let gate = lock(&gate).take();
                    if let Some((started, release)) = gate {
                        let _ = started.send(());
                        let _ = release.await;
                    }
                    Ok(())
                })
            }),
        )
        .expect("the listener registers");

    let resolved = Arc::new(AtomicBool::new(false));
    let acceptance = {
        let lane = Arc::clone(&fixture.lane);
        let resolved = Arc::clone(&resolved);
        tokio::spawn(async move {
            let admission = lane
                .accept_impl(prompt_request("hello"), &background_context())
                .await;
            resolved.store(true, Ordering::SeqCst);
            admission
        })
    };
    started_rx.await.expect("the listener started");
    settle_events().await;
    assert!(
        !resolved.load(Ordering::SeqCst),
        "the acceptance waits for its listeners"
    );
    let _ = release_tx.send(());
    unwrap(acceptance.await.expect("the acceptance joins"));
    assert!(resolved.load(Ordering::SeqCst), "the acceptance resolved");
    close_sessions(&mut sessions).await;
}

/// An acceptance starting after close returns the closed error without
/// writing.
#[tokio::test]
async fn returns_closed_when_acceptance_starts_after_close() {
    let mut sessions = SessionLedger::default();
    let fixture = create_harness(&mut sessions, None, Resources::default(), None, None).await;
    fixture
        .harness
        .close(&background_context())
        .await
        .expect("the close settles");

    let result = fixture
        .lane
        .accept_impl(prompt_request("late"), &background_context())
        .await
        .expect("accept serves");

    let error = result.expect_err("the closed lane rejects");
    assert_eq!(error.tag(), "Closed", "{error:?}");
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "no writes"
    );
    close_sessions(&mut sessions).await;
}

/// An acceptance admitted before close still publishes when its commit
/// races the close.
#[tokio::test]
async fn publishes_an_acceptance_admitted_before_close() {
    let mut sessions = SessionLedger::default();
    let storage = Arc::new(ControlledStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let session = create_session_over(&mut sessions, Arc::clone(&storage), "closing").await;
    let (options, _faux) = accept_options(Arc::clone(&session));
    let created = create_agent_harness(options, &background_context())
        .await
        .expect("the harness creates");
    created
        .harness
        .lane("main", &background_context())
        .await
        .expect("the lane attaches");
    let lane = runtime_lane(&created.harness, "main");
    let (started, release) = gate_next_commit(&storage);
    let acceptance = {
        let lane = Arc::clone(&lane);
        tokio::spawn(async move {
            lane.accept_impl(prompt_request("hello"), &background_context())
                .await
        })
    };
    started.await.expect("the commit started");
    let closing = {
        let harness = created.harness.clone();
        tokio::spawn(async move { harness.close(&background_context()).await })
    };
    let _ = release.send(());

    let acceptance = acceptance.await.expect("the acceptance joins");
    assert!(
        acceptance.is_ok(),
        "the admitted acceptance publishes: {acceptance:?}"
    );
    closing
        .await
        .expect("the close joins")
        .expect("the close settles");
    close_sessions(&mut sessions).await;
}

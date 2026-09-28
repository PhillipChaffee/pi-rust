//! The lane watch cases, ported 1:1 from upstream
//! `test/harness/runtime/watch.test.ts` ("runtime lane watch") at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The fixtures restate upstream's per-file block: the `configured`/`model`
//! configuration, the `watch-N` storage-backed sessions with the seeded
//! idle `main` lane (`createSession`), the direct commits (`commit`), the
//! running or cancel-requested scope (`runScope`), and the harness attach
//! (`attach`) building the faux provider, the fresh models catalog, and the
//! runtime `main` lane. Upstream's `Object.assign(lane, { events:
//! harness.events })` attach restates as the lane alone — no case outside
//! the recording variant reads the bus, and the recording variant mints its
//! own.
//!
//! Restatements the port makes and the tests bind:
//! - upstream's `BlockingScanStorage` (the `scanBranch` park) restates as
//!   [`ControlledStorage`]'s read gate: the capture's first storage read is
//!   the branch scan, so the gate parks exactly that read.
//! - upstream's `harness.events.watch` spy ("faults required payload
//!   corruption and unsubscribes the incomplete watcher") restates as a
//!   wrapping watch installer over the same concrete bus — the bus carries
//!   no spy seam — with the lane built directly over the fixture session,
//!   the landed `bus_watch_installer` seam's shape; the fault and the
//!   unsubscribe observation stay one watch call, and the neighboring
//!   harness-built cases pin the `HarnessFault` typing through the
//!   container's fault handler.
//! - upstream's `eventContext === sourceContext` identity restates as the
//!   context key's value round-trip
//!   ([`create_context_key`]/[`with_context_value`]); the chord context
//!   carries no identity comparison.
//! - upstream's `expect(...).not.toHaveProperty` reads restate as
//!   `Option`/enum-field `None` checks.
//! - upstream's `afterEach` closes the ledgered sessions; every test closes
//!   its sessions explicitly at the end.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the tests pin outcomes; unexpected results and violated expectations panic the test by design"
)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pi_ai::models::create_models;
use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxContentBlock;
use pi_ai::providers::faux::RegisterFauxProviderOptions;
use pi_ai::providers::faux::{faux_assistant_message, faux_provider, faux_text, faux_tool_call};
use pi_ai::types::AssistantBlock;
use pi_ai::types::BoxedFuture;
use pi_ai::types::DeferredHandle;
use pi_ai::types::{Message, StopReason, TextContent, Usage, UserContent, UserMessage};
use pi_ai::utils::assistant_message_frame::AssistantMessageFrame;
use serde_json::Value as JsonValue;
use serde_json::json;

use crate::harness::agent_harness::AgentHarnessOptions;
use crate::harness::agent_harness::AgentLane;
use crate::harness::agent_harness::DeferredView;
use crate::harness::agent_harness::EventListener;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::LaneOperationError;
use crate::harness::agent_harness::LaneQueuedItem;
use crate::harness::agent_harness::LaneSnapshot;
use crate::harness::agent_harness::LaneSnapshotTool;
use crate::harness::agent_harness::{LiveOperationView, OperationStatus, WatchHandle};
use crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS;
use crate::harness::context::{
    Context, background_context, create_context_key, with_context_value,
};
use crate::harness::events::BufferedEventWatcher;
use crate::harness::events::{HarnessEventBus, ListenerError, ResnapshotCapture, WatchFilter};
use crate::harness::runtime::harness::create_agent_harness;
use crate::harness::runtime::lane::WatchInstaller;
use crate::harness::runtime::test_support::ControlledStorage;
use crate::harness::runtime::test_support::bus_emit_batch;
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::empty_lane_snapshot;
use crate::harness::runtime::test_support::gate_next_read;
use crate::harness::runtime::test_support::lane_state_write;
use crate::harness::runtime::test_support::lock;
use crate::harness::runtime::test_support::main_lane_seed_writes;
use crate::harness::runtime::test_support::{
    restored_lane, runtime_session_metadata, settle_events,
};
use crate::harness::session::commit::insert_entry;
use crate::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use crate::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::CompactionEntryBody;
use crate::harness::session::types::Control;
use crate::harness::session::types::CustomEntryBody;
use crate::harness::session::types::DeferredScope;
use crate::harness::session::types::DeferredSuspendedOperation;
use crate::harness::session::types::Entry;
use crate::harness::session::types::GenerationContext;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::ModelIdentity;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::NormalizedRetryPolicy;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationMeta;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::RunSettings;
use crate::harness::session::types::{Session, Storage, ToolBatch, ToolCall, ToolsOperation};
use crate::harness::session::values::ToolOutputPayload;
use crate::harness::session::values::Write;
use crate::harness::session::values::append_list_write;
use crate::harness::session::values::branch_tip;
use crate::harness::session::values::operation_meta;
use crate::harness::session::values::operation_state;
use crate::harness::session::values::operation_tool_args;
use crate::harness::session::values::pending_assistant_frames;
use crate::harness::session::values::{pending_entry, pending_tool_output, set_value_write};
use crate::harness::types::AgentHarnessStreamOptions;
use crate::types::AgentMessage;
use crate::types::{
    AgentToolContent, AgentToolResult, QueueMode, ThinkingLevel, ToolExecutionMode,
};

/// The lane configuration every fixture seeds, upstream's `configuration`
/// constant: `configured`/`model`, `off`, and no active tools.
#[must_use]
fn configuration() -> LaneConfiguration {
    LaneConfiguration {
        model: ModelIdentity {
            provider: "configured".to_owned(),
            model_id: "model".to_owned(),
        },
        thinking_level: ThinkingLevel::Off,
        active_tool_names: Vec::new(),
    }
}

/// The process-unique session id one fixture carries, upstream's
/// `` `watch-${sessions.length}` ``.
fn next_session_id(ledger: &[Arc<StorageBackedSession>]) -> String {
    format!("watch-{}", ledger.len())
}

/// Opens one session over the plain (or the passed) backend, records it in
/// the ledger, and seeds the idle `main` lane, upstream's `createSession`.
///
/// # Panics
/// The seed commit's failure.
async fn create_session(
    ledger: &mut Vec<Arc<StorageBackedSession>>,
    storage: Option<Arc<dyn Storage>>,
) -> Arc<StorageBackedSession> {
    let session = Arc::new(StorageBackedSession::new(
        runtime_session_metadata(next_session_id(ledger)),
        storage.unwrap_or_else(|| Arc::new(MemoryStorage::new(MemoryStorageOptions::default()))),
        StorageBackedSessionOptions::default(),
    ));
    ledger.push(Arc::clone(&session));
    commit_writes(
        &session,
        main_lane_seed_writes(&configuration()).expect("seed writes"),
    )
    .await;
    session
}

/// The scope one seeded operation carries, upstream's `runScope`: the
/// default settings over the given control and no latest assistant entry.
#[must_use]
fn run_scope(control: Control) -> OperationScope {
    OperationScope {
        control,
        settings: RunSettings {
            compaction: DEFAULT_COMPACTION_SETTINGS,
            steering_mode: QueueMode::All,
            follow_up_mode: QueueMode::All,
            tool_execution: ToolExecutionMode::Parallel,
        },
        latest_assistant_entry_id: None,
    }
}

/// Attaches the runtime over one session, upstream's `attach`: the faux
/// provider registers into a fresh models catalog, the harness creates over
/// the session, and the `main` lane attaches through the container. The
/// `instanceof Harness`/`instanceof Lane` checks restate as the static
/// guarantees [`create_agent_harness`] and [`Harness::lane`] give.
///
/// # Panics
/// The creation's or the attach's failure.
async fn attach(session: Arc<StorageBackedSession>) -> Arc<dyn AgentLane> {
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    let models = Arc::new(create_models(None));
    models.set_provider(Arc::new(faux.provider.clone()));
    let created = create_agent_harness(
        AgentHarnessOptions {
            session,
            models,
            model: faux.first_model(),
            thinking_level: None,
            active_tool_names: None,
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
        },
        &background_context(),
    )
    .await
    .expect("the harness creates");
    created
        .harness
        .lane("main", &background_context())
        .await
        .expect("the lane attaches")
}

/// Closes the ledgered sessions and clears the ledger, upstream's
/// `afterEach` teardown.
///
/// # Panics
/// A session close's failure.
async fn close_all(ledger: &mut Vec<Arc<StorageBackedSession>>) {
    for session in ledger.drain(..) {
        session
            .close(&background_context())
            .await
            .expect("the session closes");
    }
}

/// The entry-insert write one seed commits, upstream's
/// `sessionWrites.insertEntry(...)` calls.
#[must_use]
fn entry_write(entry: NewEntry) -> Write {
    Write::Entry(Box::new(insert_entry(entry)))
}

/// The message entry one seed commits, upstream's
/// `{ id, parentId, type: "message", message }` literals.
#[must_use]
fn message_entry(id: &str, parent_id: Option<&str>, message: AgentMessage) -> NewEntry {
    NewEntry::Message {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        body: Box::new(MessageEntry {
            message,
            terminate: None,
        }),
    }
}

/// Builds one user text message, upstream's
/// `{ role: "user", content, timestamp }` literals.
#[must_use]
fn user_message(content: &str, timestamp: i64) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        timestamp,
        content: UserContent::Text(content.to_owned()),
    }))
}

/// Builds one tool-result message from its wire literal, upstream's
/// `role: "toolResult"` objects.
///
/// # Panics
/// The wire's parse failure.
#[must_use]
fn tool_result_message(wire: JsonValue) -> AgentMessage {
    serde_json::from_value(wire).expect("the tool result wire")
}

/// The operation metadata the seeded operations carry, upstream's
/// `{ operationId, lane: "main", sourceTipId: null, startedAt: 1, intent:
/// { kind: "run", promptEntryIds: [] } }` literals.
#[must_use]
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

/// The generation inputs the effect-pending leaves carry, upstream's
/// `generationContext` literals: the seeded configuration, the empty stream
/// options, and the `{ maxAttempts: 2, baseDelayMs: 0, maxAgentDelayMs:
/// 30_000 }` policy.
#[must_use]
fn generation_context_fixture() -> GenerationContext {
    GenerationContext {
        step_id: "step".to_owned(),
        trigger_entry_id: "trigger".to_owned(),
        configuration: configuration(),
        stream_options: AgentHarnessStreamOptions::default(),
        retry_policy: NormalizedRetryPolicy {
            max_attempts: 2,
            base_delay_ms: 0,
            max_agent_delay_ms: 30_000,
        },
        overflow_recovery_used: false,
    }
}

/// The tools leaf one seeded batch carries, upstream's
/// `{ ...runScope(), at: "tools", batch }` literals: the batch names the
/// given assistant entry, the `turn` id, and the seeded configuration.
#[must_use]
fn tools_state(assistant_entry_id: &str, calls: Vec<ToolCall>) -> OperationState {
    OperationState::Tools(ToolsOperation {
        scope: run_scope(Control::Running),
        batch: ToolBatch {
            assistant_entry_id: assistant_entry_id.to_owned(),
            configuration: configuration(),
            turn_id: "turn".to_owned(),
            calls,
        },
    })
}

/// The tool call one batch wires, upstream's call literals parsed through
/// the wire shape.
///
/// # Panics
/// The wire's parse failure.
#[must_use]
fn tool_call(wire: JsonValue) -> ToolCall {
    serde_json::from_value(wire).expect("the call wire")
}

/// The arguments map the tool calls and staged args carry, upstream's
/// inline object literals.
///
/// # Panics
/// The wire's parse failure.
#[must_use]
fn args(value: JsonValue) -> serde_json::Map<String, JsonValue> {
    serde_json::from_value(value).expect("the arguments map")
}

/// The staged tool output one running call carries, upstream's
/// `storedValues.pendingToolOutput(...)` literals.
///
/// # Panics
/// The write's serialization failure.
#[must_use]
fn tool_output_write(
    operation_id: &str,
    invocation_id: &str,
    content: Vec<AgentToolContent>,
    details: JsonValue,
) -> Write {
    set_value_write(
        &pending_tool_output(operation_id, invocation_id),
        ToolOutputPayload {
            content,
            details,
            usage: None,
            added_tool_names: None,
            terminate: None,
        },
    )
    .expect("tool output write")
}

/// The pending message one queue item carries, upstream's
/// `{ type: "message", payload: { role: "user", content, timestamp: 1 } }`.
#[must_use]
fn queued_message(content: &str) -> PendingEntry {
    PendingEntry::Message {
        payload: Box::new(user_message(content, 1)),
    }
}

/// The inbox item one lane record queues, upstream's `{ entryId, kind }`
/// literals.
#[must_use]
fn inbox_item(entry_id: &str, kind: InboxItemKind) -> InboxItem {
    InboxItem {
        entry_id: entry_id.to_owned(),
        kind,
    }
}

/// The watch rejection's message, the fault's read through the collapsed
/// trait surface.
#[must_use]
fn watch_fault_message(error: LaneOperationError) -> String {
    match error {
        LaneOperationError::Closed(reason) => reason.to_string(),
    }
}

/// The watch handle that records its unsubscribe, upstream's spy wrapper's
/// `{ ...handle, unsubscribe }` restatement: every read and the start
/// delegate to the minted watcher, and the unsubscribe records first.
struct UnsubscribeRecordingWatch {
    /// The bus-minted watcher.
    inner: Arc<BufferedEventWatcher<LaneSnapshot>>,
    /// The recorded unsubscribe flag.
    unsubscribed: Arc<AtomicBool>,
}

impl WatchHandle<LaneSnapshot> for UnsubscribeRecordingWatch {
    fn snapshot(&self) -> LaneSnapshot {
        WatchHandle::<LaneSnapshot>::snapshot(&*self.inner)
    }

    fn set_snapshot(&self, snapshot: LaneSnapshot) {
        WatchHandle::<LaneSnapshot>::set_snapshot(&*self.inner, snapshot);
    }

    fn start(&self, listener: EventListener) {
        WatchHandle::<LaneSnapshot>::start(&*self.inner, listener);
    }

    fn resnapshot<'a>(
        &self,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<LaneSnapshot, ListenerError>> {
        WatchHandle::<LaneSnapshot>::resnapshot(&*self.inner, context)
    }

    fn unsubscribe(&self) {
        self.unsubscribed.store(true, Ordering::SeqCst);
        WatchHandle::<LaneSnapshot>::unsubscribe(&*self.inner);
    }
}

/// The wrapping installer, upstream's spied `harness.events.watch`: the bus
/// mints the watcher and the recording adapter wraps it, the landed
/// `bus_watch_installer` seam's shape with the unsubscribe record.
///
/// # Panics
/// The bus being closed.
fn recording_unsubscribe_installer(
    bus: Arc<HarnessEventBus>,
    unsubscribed: Arc<AtomicBool>,
) -> WatchInstaller {
    Arc::new(
        move |filter: WatchFilter,
              _context: &Context,
              resnapshot: ResnapshotCapture<LaneSnapshot>| {
            let watcher = bus
                .watch(
                    empty_lane_snapshot("main", &configuration()),
                    filter,
                    Some(resnapshot),
                )
                .expect("watch installs");
            Ok(Box::new(UnsubscribeRecordingWatch {
                inner: watcher,
                unsubscribed: Arc::clone(&unsubscribed),
            }))
        },
    )
}

/// Upstream's "captures a compaction-bounded transcript and isolates the
/// returned snapshot" case.
#[tokio::test]
async fn captures_a_compaction_bounded_transcript_and_isolates_the_returned_snapshot() {
    let mut ledger = Vec::new();
    let session = create_session(&mut ledger, None).await;
    commit_writes(
        &session,
        vec![
            entry_write(NewEntry::Custom {
                id: "root".to_owned(),
                parent_id: None,
                body: CustomEntryBody {
                    custom_type: "root".to_owned(),
                    data: None,
                },
            }),
            entry_write(NewEntry::Compaction {
                id: "compact".to_owned(),
                parent_id: Some("root".to_owned()),
                body: CompactionEntryBody {
                    summary: "summary".to_owned(),
                    retained_tail: Vec::new(),
                    tokens_before: 10,
                    details: None,
                    usage: None,
                    from_hook: false,
                },
            }),
            entry_write(message_entry(
                "after",
                Some("compact"),
                user_message("after", 2),
            )),
            set_value_write(&branch_tip("main"), Some("after".to_owned())).expect("tip write"),
        ],
    )
    .await;
    let lane = attach(session).await;

    let first = lane
        .watch(&background_context())
        .await
        .expect("watch serves");
    let snapshot = first.snapshot();
    let ids: Vec<&str> = snapshot.transcript.iter().map(Entry::id).collect();
    assert_eq!(
        ids,
        vec!["compact", "after"],
        "the transcript stops at the compaction boundary"
    );
    assert_eq!(snapshot.lane, "main");
    assert_eq!(snapshot.tip_id.as_deref(), Some("after"));
    assert_eq!(
        snapshot.configuration,
        configuration(),
        "the seeded configuration rides"
    );
    assert_eq!(snapshot.stats.message_count, 1, "the message total reads");
    assert_eq!(snapshot.operation, None);
    assert!(snapshot.queues.is_empty(), "no queues read");
    assert!(!snapshot.faulted, "the open lane is not faulted");

    // Mutating the returned snapshot must not corrupt the next watch.
    let mut mutated = first.snapshot();
    mutated.transcript.clear();
    mutated.tip_id = None;

    let second = lane
        .watch(&background_context())
        .await
        .expect("watch serves");
    let snapshot = second.snapshot();
    let ids: Vec<&str> = snapshot.transcript.iter().map(Entry::id).collect();
    assert_eq!(
        ids,
        vec!["compact", "after"],
        "the second watch re-captures"
    );
    assert_eq!(snapshot.tip_id.as_deref(), Some("after"));
    second.unsubscribe();
    first.unsubscribe();
    close_all(&mut ledger).await;
}

/// Upstream's "dereferences queues, pending writes, and deferred handles"
/// case.
#[expect(
    clippy::too_many_lines,
    reason = "the seeded queues, the deferred leaf, and their dereferenced views are one capture"
)]
#[tokio::test]
async fn dereferences_queues_pending_writes_and_deferred_handles() {
    let mut ledger = Vec::new();
    let session = create_session(&mut ledger, None).await;
    let handle = DeferredHandle {
        provider: "provider".to_owned(),
        model_id: "model".to_owned(),
        api: "test".to_owned(),
        id: "deferred".to_owned(),
        expires_at: None,
        poll_after_ms: None,
        data: None,
    };
    let source = faux_assistant_message(
        Vec::<FauxContentBlock>::new(),
        FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::Deferred),
            deferred: Some(handle.clone()),
            ..FauxAssistantMessageOptions::default()
        },
    );
    let ids = ["next", "steer", "follow", "write"];
    let operation_id = session.id_generator().next(None);
    let state = OperationState::DeferredSuspended(DeferredSuspendedOperation {
        deferred: DeferredScope {
            scope: run_scope(Control::CancelRequested { requested_at: 2 }),
            step_id: "step".to_owned(),
            source_entry_id: "source".to_owned(),
            poll: 3,
            configuration: configuration(),
            stream_options: AgentHarnessStreamOptions::default(),
        },
    });
    let mut writes = vec![
        entry_write(message_entry(
            "source",
            None,
            AgentMessage::Standard(Message::Assistant(source)),
        )),
        set_value_write(&operation_meta(&operation_id), run_meta(&operation_id))
            .expect("meta write"),
        set_value_write(&operation_state(&operation_id), state).expect("state write"),
        set_value_write(&branch_tip("main"), Some("source".to_owned())).expect("tip write"),
    ];
    for id in ids {
        let payload = if id == "write" {
            PendingEntry::Custom {
                custom_type: "note".to_owned(),
                payload: Some(json!({ "id": id })),
            }
        } else {
            queued_message(id)
        };
        writes.push(set_value_write(&pending_entry(id), payload).expect("pending write"));
    }
    writes.push(
        lane_state_write(
            "main",
            Some(&operation_id),
            None,
            vec![
                inbox_item("next", InboxItemKind::NextRun),
                inbox_item("steer", InboxItemKind::Steer),
                inbox_item("follow", InboxItemKind::FollowUp),
                inbox_item("write", InboxItemKind::Write),
            ],
        )
        .expect("lane state write"),
    );
    commit_writes(&session, writes).await;
    let lane = attach(session).await;

    let watch = lane
        .watch(&background_context())
        .await
        .expect("watch serves");
    let snapshot = watch.snapshot();
    assert_eq!(
        snapshot.queues,
        vec![
            LaneQueuedItem::Message {
                entry_id: "next".to_owned(),
                kind: InboxItemKind::NextRun,
                message: Box::new(user_message("next", 1)),
            },
            LaneQueuedItem::Message {
                entry_id: "steer".to_owned(),
                kind: InboxItemKind::Steer,
                message: Box::new(user_message("steer", 1)),
            },
            LaneQueuedItem::Message {
                entry_id: "follow".to_owned(),
                kind: InboxItemKind::FollowUp,
                message: Box::new(user_message("follow", 1)),
            },
            LaneQueuedItem::Custom {
                entry_id: "write".to_owned(),
                kind: InboxItemKind::Write,
                custom_type: "note".to_owned(),
                data: Some(json!({ "id": "write" })),
            },
        ],
        "the queues dereference their pending payloads"
    );
    let LiveOperationView {
        id,
        status,
        deferred,
        running_tools,
        ..
    } = snapshot.operation.expect("the operation view");
    assert_eq!(id, operation_id);
    assert_eq!(
        status,
        OperationStatus::Aborting,
        "the cancel request aborts the operation"
    );
    assert_eq!(
        deferred.as_ref(),
        Some(&DeferredView { handle, poll: 3 }),
        "the source entry's handle and the poll ride the view"
    );
    assert!(running_tools.is_empty(), "no tools run");
    watch.unsubscribe();
    close_all(&mut ledger).await;
}

/// Upstream's "omits streaming presentation when an effect-pending response
/// has no frames" case.
#[tokio::test]
async fn omits_streaming_presentation_when_an_effect_pending_response_has_no_frames() {
    let mut ledger = Vec::new();
    let session = create_session(&mut ledger, None).await;
    let operation_id = session.id_generator().next(None);
    commit_writes(
        &session,
        vec![
            set_value_write(&operation_meta(&operation_id), run_meta(&operation_id))
                .expect("meta write"),
            set_value_write(
                &operation_state(&operation_id),
                OperationState::AssistantEffectPending(AssistantEffectPendingOperation {
                    scope: run_scope(Control::Running),
                    generation_context: generation_context_fixture(),
                    attempt: 1,
                    response_entry_id: "response-without-frames".to_owned(),
                    usage_id: "usage".to_owned(),
                    intended_output_limit: 100,
                    context_window: 1_000,
                }),
            )
            .expect("state write"),
            lane_state_write("main", Some(&operation_id), None, Vec::new())
                .expect("lane state write"),
        ],
    )
    .await;
    let lane = attach(session).await;

    let watch = lane
        .watch(&background_context())
        .await
        .expect("watch serves");
    let snapshot = watch.snapshot();
    let operation = snapshot.operation.expect("the operation view");
    assert!(
        operation.streaming_message.is_none(),
        "the frame-less response streams nothing"
    );
    watch.unsubscribe();
    close_all(&mut ledger).await;
}

/// Upstream's "reduces assistant frames and projects running and settled
/// tools with full-content indexes" case: the streaming half reduces the
/// stored frames, and the tools half projects the batch's running and
/// settled calls over their source indexes.
#[expect(
    clippy::too_many_lines,
    reason = "the streaming and tools halves are one case upstream; both fixtures bind the capture"
)]
#[tokio::test]
async fn reduces_assistant_frames_and_projects_running_and_settled_tools_with_full_content_indexes()
{
    // The streaming half.
    let mut ledger = Vec::new();
    let frame_session = create_session(&mut ledger, None).await;
    let partial = faux_assistant_message(
        Vec::<FauxContentBlock>::new(),
        FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::Pending),
            ..FauxAssistantMessageOptions::default()
        },
    );
    let frames = vec![
        AssistantMessageFrame::Start {
            partial: partial.clone(),
        },
        AssistantMessageFrame::TextStart {
            content_index: 0,
            content: TextContent {
                text: String::new(),
                text_signature: None,
            },
        },
        AssistantMessageFrame::TextDelta {
            content_index: 0,
            delta: "partial".to_owned(),
        },
    ];
    let frame_operation_id = frame_session.id_generator().next(None);
    let mut frame_writes = vec![
        set_value_write(
            &operation_meta(&frame_operation_id),
            run_meta(&frame_operation_id),
        )
        .expect("meta write"),
        set_value_write(
            &operation_state(&frame_operation_id),
            OperationState::AssistantEffectPending(AssistantEffectPendingOperation {
                scope: run_scope(Control::Running),
                generation_context: generation_context_fixture(),
                attempt: 1,
                response_entry_id: "response".to_owned(),
                usage_id: "usage".to_owned(),
                intended_output_limit: 100,
                context_window: 1_000,
            }),
        )
        .expect("state write"),
    ];
    for frame in frames {
        frame_writes.push(
            append_list_write(
                &pending_assistant_frames(&frame_operation_id, "response"),
                frame,
            )
            .expect("frame append"),
        );
    }
    frame_writes.push(
        lane_state_write("main", Some(&frame_operation_id), None, Vec::new())
            .expect("lane state write"),
    );
    commit_writes(&frame_session, frame_writes).await;
    let frame_lane = attach(frame_session).await;
    let frame_watch = frame_lane
        .watch(&background_context())
        .await
        .expect("watch serves");
    let frame_snapshot = frame_watch.snapshot();
    let streaming = frame_snapshot
        .operation
        .expect("the operation view")
        .streaming_message
        .expect("the streaming message");
    assert_eq!(
        streaming.content,
        vec![AssistantBlock::Text(TextContent {
            text: "partial".to_owned(),
            text_signature: None,
        })],
        "the frames reduce to the partial text"
    );
    frame_watch.unsubscribe();

    // The tools half.
    let tool_session = create_session(&mut ledger, None).await;
    let tool_usage: Usage = serde_json::from_value(json!({
        "input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4, "totalTokens": 10,
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
    }))
    .expect("the usage wire");
    let assistant = faux_assistant_message(
        vec![
            faux_text("before"),
            faux_tool_call(
                "completed",
                args(json!({ "source": "completed" })),
                Some("call-completed".to_owned()),
            ),
            faux_tool_call(
                "read",
                args(json!({ "source": "running" })),
                Some("call-running".to_owned()),
            ),
            faux_tool_call(
                "write",
                args(json!({ "source": "without-checkpoint" })),
                Some("call-without-checkpoint".to_owned()),
            ),
            faux_tool_call(
                "real",
                args(json!({ "source": "real" })),
                Some("call-ready".to_owned()),
            ),
            faux_tool_call(
                "missing",
                args(json!({ "source": "synthetic" })),
                Some("call-synthetic".to_owned()),
            ),
            faux_tool_call(
                "planned",
                args(json!({ "source": "planned" })),
                Some("call-planned".to_owned()),
            ),
        ],
        FauxAssistantMessageOptions::default(),
    );
    let tool_operation_id = tool_session.id_generator().next(None);
    let completed_result = tool_result_message(json!({
        "role": "toolResult",
        "toolCallId": "call-completed",
        "toolName": "completed",
        "content": [{ "type": "text", "text": "completed" }],
        "isError": false,
        "timestamp": 2,
    }));
    let ready_result = tool_result_message(json!({
        "role": "toolResult",
        "toolCallId": "call-ready",
        "toolName": "real",
        "content": [{ "type": "text", "text": "settled" }],
        "details": { "kind": "real" },
        "usage": {
            "input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4, "totalTokens": 10,
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
        },
        "addedToolNames": ["later"],
        "isError": false,
        "timestamp": 3,
    }));
    let synthetic_result = tool_result_message(json!({
        "role": "toolResult",
        "toolCallId": "call-synthetic",
        "toolName": "missing",
        "content": [{ "type": "text", "text": "unavailable" }],
        "isError": true,
        "timestamp": 4,
    }));
    let tool_writes = vec![
        entry_write(message_entry(
            "assistant",
            None,
            AgentMessage::Standard(Message::Assistant(assistant)),
        )),
        set_value_write(
            &operation_meta(&tool_operation_id),
            run_meta(&tool_operation_id),
        )
        .expect("meta write"),
        set_value_write(
            &operation_state(&tool_operation_id),
            tools_state(
                "assistant",
                vec![
                    tool_call(json!({
                        "sourceIndex": 1, "resultEntryId": "completed",
                        "status": "completed", "terminate": false,
                    })),
                    tool_call(json!({
                        "sourceIndex": 2, "resultEntryId": "result",
                        "status": "effect_pending", "replay": "safe",
                    })),
                    tool_call(json!({
                        "sourceIndex": 3, "resultEntryId": "without-checkpoint",
                        "status": "effect_pending", "replay": "never",
                    })),
                    tool_call(json!({
                        "sourceIndex": 4, "resultEntryId": "ready",
                        "status": "outcome_ready", "terminate": true,
                    })),
                    tool_call(json!({
                        "sourceIndex": 5, "resultEntryId": "synthetic",
                        "status": "outcome_ready", "terminate": false,
                    })),
                    tool_call(json!({
                        "sourceIndex": 6, "resultEntryId": "planned",
                        "status": "planned",
                    })),
                ],
            ),
        )
        .expect("state write"),
        entry_write(message_entry(
            "completed",
            Some("assistant"),
            completed_result,
        )),
        set_value_write(
            &operation_tool_args(&tool_operation_id, "turn", 2),
            args(json!({ "path": "file" })),
        )
        .expect("args write"),
        set_value_write(
            &operation_tool_args(&tool_operation_id, "turn", 3),
            args(json!({ "path": "output" })),
        )
        .expect("args write"),
        set_value_write(
            &operation_tool_args(&tool_operation_id, "turn", 4),
            args(json!({ "path": "settled" })),
        )
        .expect("args write"),
        tool_output_write(
            &tool_operation_id,
            "result",
            vec![AgentToolContent::Text(TextContent {
                text: "partial".to_owned(),
                text_signature: None,
            })],
            json!({ "bytes": 1 }),
        ),
        set_value_write(
            &pending_entry("ready"),
            PendingEntry::Message {
                payload: Box::new(ready_result),
            },
        )
        .expect("staged write"),
        set_value_write(
            &pending_entry("synthetic"),
            PendingEntry::Message {
                payload: Box::new(synthetic_result),
            },
        )
        .expect("staged write"),
        set_value_write(&branch_tip("main"), Some("completed".to_owned())).expect("tip write"),
        lane_state_write("main", Some(&tool_operation_id), None, Vec::new())
            .expect("lane state write"),
    ];
    commit_writes(&tool_session, tool_writes).await;
    let tool_lane = attach(tool_session).await;
    let tool_watch = tool_lane
        .watch(&background_context())
        .await
        .expect("watch serves");
    let tool_snapshot = tool_watch.snapshot();
    let ids: Vec<&str> = tool_snapshot.transcript.iter().map(Entry::id).collect();
    assert_eq!(ids, vec!["assistant", "completed"], "the transcript reads");
    let running_tools = &tool_snapshot
        .operation
        .expect("the operation view")
        .running_tools;
    assert_eq!(
        running_tools.len(),
        4,
        "the completed and planned calls never project"
    );
    let LaneSnapshotTool::Running {
        tool_call_id,
        tool_name,
        args: call_args,
        result,
    } = &running_tools[0]
    else {
        panic!("the first running call: {:?}", running_tools[0]);
    };
    assert_eq!(tool_call_id, "call-running");
    assert_eq!(tool_name, "read");
    assert_eq!(call_args, &json!({ "path": "file" }));
    assert_eq!(
        result,
        &Some(AgentToolResult {
            content: vec![AgentToolContent::Text(TextContent {
                text: "partial".to_owned(),
                text_signature: None,
            })],
            details: json!({ "bytes": 1 }),
            usage: None,
            added_tool_names: None,
            terminate: None,
        }),
        "the staged checkpoint rides the running call"
    );
    let LaneSnapshotTool::Running {
        tool_call_id,
        tool_name,
        args: call_args,
        result,
    } = &running_tools[1]
    else {
        panic!("the second running call: {:?}", running_tools[1]);
    };
    assert_eq!(tool_call_id, "call-without-checkpoint");
    assert_eq!(tool_name, "write");
    assert_eq!(call_args, &json!({ "path": "output" }));
    assert!(
        result.is_none(),
        "the checkpoint-less call carries no result"
    );
    let LaneSnapshotTool::Settled {
        tool_call_id,
        tool_name,
        args: call_args,
        result,
        is_error,
    } = &running_tools[2]
    else {
        panic!("the first settled call: {:?}", running_tools[2]);
    };
    assert_eq!(tool_call_id, "call-ready");
    assert_eq!(tool_name, "real");
    assert_eq!(call_args, &json!({ "path": "settled" }));
    assert_eq!(
        result,
        &AgentToolResult {
            content: vec![AgentToolContent::Text(TextContent {
                text: "settled".to_owned(),
                text_signature: None,
            })],
            details: json!({ "kind": "real" }),
            usage: Some(tool_usage),
            added_tool_names: Some(vec!["later".to_owned()]),
            terminate: Some(true),
        },
        "the staged result rides with its usage and tool names"
    );
    assert!(!is_error);
    let LaneSnapshotTool::Settled {
        tool_call_id,
        tool_name,
        args: call_args,
        result,
        is_error,
    } = &running_tools[3]
    else {
        panic!("the second settled call: {:?}", running_tools[3]);
    };
    assert_eq!(tool_call_id, "call-synthetic");
    assert_eq!(tool_name, "missing");
    assert_eq!(
        call_args,
        &json!({ "source": "synthetic" }),
        "the block's arguments ride the persisted-less call"
    );
    assert_eq!(
        result,
        &AgentToolResult {
            content: vec![AgentToolContent::Text(TextContent {
                text: "unavailable".to_owned(),
                text_signature: None,
            })],
            details: JsonValue::Null,
            usage: None,
            added_tool_names: None,
            terminate: None,
        },
        "the synthetic result carries no details"
    );
    assert!(is_error);
    tool_watch.unsubscribe();
    close_all(&mut ledger).await;
}

/// Upstream's "faults missing or mismatched staged outcome-ready results"
/// case: the corruption table loops inside one test, upstream's
/// `for (const corruption of ...)`.
#[tokio::test]
async fn faults_missing_or_mismatched_staged_outcome_ready_results() {
    for corruption in ["missing", "mismatched"] {
        let mut ledger = Vec::new();
        let session = create_session(&mut ledger, None).await;
        let operation_id = session.id_generator().next(None);
        let assistant = faux_assistant_message(
            vec![faux_tool_call(
                "read",
                args(json!({ "path": "file" })),
                Some("call".to_owned()),
            )],
            FauxAssistantMessageOptions::default(),
        );
        let mut writes = vec![
            entry_write(message_entry(
                "assistant",
                None,
                AgentMessage::Standard(Message::Assistant(assistant)),
            )),
            set_value_write(&operation_meta(&operation_id), run_meta(&operation_id))
                .expect("meta write"),
            set_value_write(
                &operation_state(&operation_id),
                tools_state(
                    "assistant",
                    vec![tool_call(json!({
                        "sourceIndex": 0, "resultEntryId": "result",
                        "status": "outcome_ready", "terminate": false,
                    }))],
                ),
            )
            .expect("state write"),
            set_value_write(&branch_tip("main"), Some("assistant".to_owned())).expect("tip write"),
            lane_state_write("main", Some(&operation_id), None, Vec::new())
                .expect("lane state write"),
        ];
        if corruption == "mismatched" {
            writes.push(
                set_value_write(
                    &pending_entry("result"),
                    PendingEntry::Message {
                        payload: Box::new(tool_result_message(json!({
                            "role": "toolResult",
                            "toolCallId": "other-call",
                            "toolName": "read",
                            "content": [],
                            "isError": false,
                            "timestamp": 2,
                        }))),
                    },
                )
                .expect("staged write"),
            );
        }
        commit_writes(&session, writes).await;
        let lane = attach(session).await;

        let Err(error) = lane.watch(&background_context()).await else {
            panic!("the corrupted staged result faults the watch");
        };
        assert!(
            watch_fault_message(error).contains("AgentHarness storage or invariant fault"),
            "the {corruption} staged result faults the watch"
        );
        close_all(&mut ledger).await;
    }
}

/// Upstream's "faults required payload corruption and unsubscribes the
/// incomplete watcher" case: upstream spies `harness.events.watch` to
/// observe the unsubscribe; the concrete bus carries no spy seam, so the
/// wrapping installer mints the watcher over the same bus and records.
#[tokio::test]
async fn faults_required_payload_corruption_and_unsubscribes_the_incomplete_watcher() {
    let mut ledger = Vec::new();
    let session = create_session(&mut ledger, None).await;
    commit_writes(
        &session,
        vec![
            lane_state_write(
                "main",
                None,
                None,
                vec![inbox_item("missing", InboxItemKind::NextRun)],
            )
            .expect("lane state write"),
        ],
    )
    .await;
    let unsubscribed = Arc::new(AtomicBool::new(false));
    let bus = Arc::new(HarnessEventBus::new());
    let lane = restored_lane(
        session,
        bus_emit_batch(Arc::clone(&bus)),
        recording_unsubscribe_installer(Arc::clone(&bus), Arc::clone(&unsubscribed)),
    )
    .await;

    let Err(error) = lane.watch(&background_context()).await else {
        panic!("the corrupted inbox faults the watch");
    };
    assert!(
        watch_fault_message(error).contains("missing its payload"),
        "the queue's missing payload names its invariant"
    );
    assert!(
        unsubscribed.load(Ordering::SeqCst),
        "the incomplete watcher unsubscribed"
    );
    close_all(&mut ledger).await;
}

/// Upstream's "returns snapshot-before plus buffered events when watch wins
/// the lane line" case: the read gate parks the capture's branch scan while
/// the append queues on the lane line behind it.
#[tokio::test]
async fn returns_snapshot_before_plus_buffered_events_when_watch_wins_the_lane_line() {
    let mut ledger = Vec::new();
    let storage = Arc::new(ControlledStorage::new(Arc::new(MemoryStorage::new(
        MemoryStorageOptions::default(),
    ))));
    let backend_arc = Arc::clone(&storage);
    let backend: Arc<dyn Storage> = backend_arc;
    let session = create_session(&mut ledger, Some(backend)).await;
    commit_writes(
        &session,
        vec![
            entry_write(NewEntry::Custom {
                id: "root".to_owned(),
                parent_id: None,
                body: CustomEntryBody {
                    custom_type: "root".to_owned(),
                    data: None,
                },
            }),
            set_value_write(&branch_tip("main"), Some("root".to_owned())).expect("tip write"),
        ],
    )
    .await;
    let lane = attach(session).await;

    let (started, release) = gate_next_read(&storage);
    let watching = {
        let lane = Arc::clone(&lane);
        tokio::spawn(async move { lane.watch(&background_context()).await })
    };
    started.await.expect("the branch scan parked");
    let source_key = create_context_key::<String>("watch.event.source");
    let source_context =
        with_context_value(&source_key, "append".to_owned(), &background_context());
    let appending = {
        let lane = Arc::clone(&lane);
        let source_context = source_context.clone();
        tokio::spawn(async move {
            lane.append_message(user_message("later", 2), &source_context)
                .await
        })
    };
    let _ = release.send(());

    let watch = watching.await.expect("watch join").expect("watch serves");
    let snapshot = watch.snapshot();
    let ids: Vec<&str> = snapshot.transcript.iter().map(Entry::id).collect();
    assert_eq!(ids, vec!["root"], "the snapshot captured before the append");

    let seen: Arc<Mutex<Vec<(String, bool)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let listener_key = source_key;
    watch.start(Arc::new(move |event: &HarnessEvent, context: &Context| {
        let same_context = context.value(&listener_key) == Some(&"append".to_owned());
        let sink = Arc::clone(&sink);
        let event_type = event.event_type().as_str().to_owned();
        Box::pin(async move {
            lock(&sink).push((event_type, same_context));
            Ok(())
        })
    }));

    appending
        .await
        .expect("append join")
        .expect("the append commits");
    settle_events().await;

    let seen = lock(&seen).clone();
    let types: Vec<&str> = seen
        .iter()
        .map(|(event_type, _)| event_type.as_str())
        .collect();
    assert_eq!(
        types,
        vec!["message_start", "message_end", "entry_added"],
        "the append's events deliver in order"
    );
    assert!(
        seen.iter().all(|(_, same_context)| *same_context),
        "every event carries the append's context"
    );
    watch.unsubscribe();
    close_all(&mut ledger).await;
}

/// Upstream's "returns snapshot-after without replay when publication wins"
/// case.
#[tokio::test]
async fn returns_snapshot_after_without_replay_when_publication_wins() {
    let mut ledger = Vec::new();
    let session = create_session(&mut ledger, None).await;
    let lane = attach(session).await;
    let entry_id = lane
        .append_message(user_message("existing", 1), &background_context())
        .await
        .expect("append serves");

    let watch = lane
        .watch(&background_context())
        .await
        .expect("watch serves");
    let seen: Arc<Mutex<Vec<HarnessEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    watch.start(crate::harness::runtime::test_support::recording_listener(
        Arc::clone(&sink),
    ));
    settle_events().await;

    let snapshot = watch.snapshot();
    let ids: Vec<&str> = snapshot.transcript.iter().map(Entry::id).collect();
    assert_eq!(
        ids,
        vec![entry_id.as_str()],
        "the snapshot captured after the append"
    );
    assert!(lock(&seen).is_empty(), "the start listener replays nothing");
    watch.unsubscribe();
    close_all(&mut ledger).await;
}

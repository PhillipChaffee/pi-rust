//! The generation checkpoint and assistant-generation suite, ported 1:1
//! from upstream `test/harness/runtime/drive-generation.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements the port makes and the tests bind:
//! - upstream's `vi.spyOn(fixture.storage, "getEntries")` (the result
//!   record's no-scan proof) rides the [`common::EntryReadCountingStorage`]
//!   backend counter, reset right before the two `get_result` reads.
//! - upstream's shared-object frame mutation (`text.text = "ab"` mutating
//!   the pushed events' accumulator before the encoder reads it) restates
//!   as the pushed events carrying the accumulator as it stands at encode
//!   time; the recorded frame shapes are the asserted observable.
//! - upstream's `afterEach` (restoreAllMocks + session closes) restates as
//!   an explicit session close at each test's end; no fixture carries a
//!   mock to restore.
//!
//! The boundary cases past upstream's suites bind the arms upstream's
//! behavior pins but its tests never reach: the tool-context sources and
//! the active-tool surface's resolution, the `before_request` patch, the
//! transform hook's and the payload hook's gate refusals, the closed
//! registry, the provider stream's protocol violation, the request hook's
//! recorded refusal, and the preparation's and the retry wait's cancel
//! downgrades.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the fixture helpers raise deliberately, upstream's thrown fixture errors"
)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pi_ai::auth::types::ProviderAuth;
use pi_ai::models::{Provider, ProviderImpl, ProviderModelError};
use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxResponseStep;
use pi_ai::providers::faux::{FauxTokenSize, RegisterFauxProviderOptions, faux_assistant_message};
use pi_ai::types::Context as AiContext;
use pi_ai::types::Message;
use pi_ai::types::Model;
use pi_ai::types::ProviderId;
use pi_ai::types::{Api, AssistantBlock, AssistantMessage, AssistantMessageEvent};
use pi_ai::types::{
    SimpleStreamOptions, StopReason, StreamOptions, TextContent, Tool, UserContent,
};
use pi_ai::utils::assistant_message_frame::AssistantMessageFrame;
use pi_ai::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use pi_ai::utils::retry::RetryPolicy;
use serde_json::Value as JsonValue;
use serde_json::json;

use crate::harness::agent_harness::BeforeRequestResult;
use crate::harness::agent_harness::BeforeRunResult;
use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::FollowUpResult;
use crate::harness::agent_harness::HarnessEventType;
use crate::harness::agent_harness::{HookName, HookOptions, HookResult, PayloadResult};
use crate::harness::context::background_context;
use crate::harness::runtime::drive::checkpoint::{run_checkpoint, start_run};
use crate::harness::runtime::drive::generation::{GenerationLeaf, run_generation};
use crate::harness::runtime::drive::recovery::recover_assistant_generation;
use crate::harness::runtime::drive::retry::retry_not_before;
use crate::harness::runtime::lane::{Lane, operation_state_with_scope};
use crate::harness::runtime::restore::restore_lane;
use crate::harness::runtime::test_support::deferred;
use crate::harness::runtime::test_support::provider_pass_through;
use crate::harness::runtime::test_support::{lock, passthrough_fault_handler, patch_live_state};
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneCommand;
use crate::harness::runtime::types::{LaneError, LaneState, LiveOperation, ProcedureResult};
use crate::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::AssistantReadyOperation;
use crate::harness::session::types::AssistantRetryWaitOperation;
use crate::harness::session::types::CheckpointData;
use crate::harness::session::types::CheckpointOperation;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::Control;
use crate::harness::session::types::Entry;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::NormalizedRetryPolicy;
use crate::harness::session::types::OperationKind;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::RetryWait;
use crate::harness::session::types::Session;
use crate::harness::session::types::{SessionReader, Storage, TerminalStatus, operation_scope_of};
use crate::harness::session::values::Write;
use crate::harness::session::values::append_list_write;
use crate::harness::session::values::operation_meta;
use crate::harness::session::values::operation_state;
use crate::harness::session::values::{pending_assistant_frames, pending_entry, set_value_write};
use crate::harness::types::AgentHarnessStreamOptions;
use crate::harness::types::AgentHarnessStreamOptionsPatch;
use crate::harness::types::AgentHarnessTool;
use crate::harness::types::AgentHarnessToolContextSource;
use crate::harness::types::{AgentHarnessToolExecuteFn, ToolContext, ToolContextProvider};
use crate::types::{AgentMessage, AgentToolResult};

use super::common;
use crate::harness::agent_harness::SystemPromptSource;
use crate::harness::context::Context;

/// The generation suite's fixture spec, upstream's `createFixture`
/// defaults: the `{ min: 1, max: 1 }` faux token size, the
/// `{ enabled: true, maxRetries: 3, baseDelayMs: 1 }` retry policy, the
/// admitted `question` prompt, and the optional backend override.
#[must_use]
fn generation_spec(backend: Option<Arc<dyn Storage>>) -> common::DriveFixtureSpec {
    common::DriveFixtureSpec {
        suite: "generation",
        watch_suite: "generation",
        faux: RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(1),
                max: Some(1),
            }),
            ..RegisterFauxProviderOptions::default()
        },
        retry_policy: RetryPolicy {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 1,
            max_agent_delay_ms: None,
        },
        stream_options: AgentHarnessStreamOptions::default(),
        backend,
        admit_prompt: true,
        system_prompt: None,
    }
}

/// The ready generation the fixture's operation holds, upstream's
/// `readyGeneration`: raises `fixture has no ready generation` at any other
/// leaf.
///
/// # Panics
/// The leaf not being `assistant.ready`.
#[must_use]
fn ready_generation(fixture: &common::DriveFixture) -> AssistantReadyOperation {
    match common::current_state(fixture) {
        OperationState::AssistantReady(ready) => ready,
        _ => panic!("fixture has no ready generation"),
    }
}

/// Starts the fixture's run at its initial boundary, upstream's
/// `startFixtureRun`: raises `run did not start at its initial boundary`
/// when the leaf is not `starting`.
///
/// # Panics
/// The leaf not being `starting`, or the procedure failing.
async fn start_fixture_run(fixture: &common::DriveFixture) -> ProcedureResult {
    let OperationState::Starting(run) = common::current_state(fixture) else {
        panic!("run did not start at its initial boundary");
    };
    start_run(&fixture.lane, &fixture.drive, &run)
        .await
        .expect("the run starts")
}

/// Advances the fixture to its ready generation, upstream's
/// `advanceToReady`: starts the run, walks its checkpoint, and reads the
/// `assistant.ready` leaf.
///
/// # Panics
/// The leaf not reaching `checkpoint`, or either procedure failing.
async fn advance_to_ready(fixture: &common::DriveFixture) -> AssistantReadyOperation {
    start_fixture_run(fixture).await;
    let OperationState::Checkpoint(checkpoint) = common::current_state(fixture) else {
        panic!("run did not reach checkpoint");
    };
    run_checkpoint(&fixture.lane, &fixture.drive, &checkpoint)
        .await
        .expect("the checkpoint advances");
    ready_generation(fixture)
}

/// Whether one commit attempt carries an operation-state write whose leaf
/// is `assistant.{status}`, upstream's `writesOperationState`.
fn writes_operation_state(writes: &[Write], status: &str) -> bool {
    writes.iter().any(|write| {
        matches!(write, Write::ValueSet(value)
            if value.kind == "value"
                && value.op == "set"
                && value.namespace == "pi.op.state"
                && value.value.get("at").and_then(JsonValue::as_str)
                    == Some(format!("assistant.{status}").as_str()))
    })
}

/// The write-family string one commit write renders, upstream's
/// `` `${write.kind}:${write.op}:${write.namespace}` `` mapping with the
/// entry and usage families keeping their bare kind.
fn write_family(write: &Write) -> String {
    fn family(kind: &str, op: &str, namespace: &str) -> String {
        format!("{kind}:{op}:{namespace}")
    }
    match write {
        Write::Entry(_) => "entry".to_owned(),
        Write::Usage(_) => "usage".to_owned(),
        Write::ValueSet(value) => family(&value.kind, &value.op, &value.namespace),
        Write::ValueDelete(value) => family(&value.kind, &value.op, &value.namespace),
        Write::ListAppend(value) => family(&value.kind, &value.op, &value.namespace),
        Write::ListDelete(value) => family(&value.kind, &value.op, &value.namespace),
    }
}

/// Closes the fixture's session, upstream's `afterEach` close.
///
/// # Panics
/// The close's failure.
async fn close_session(fixture: &common::DriveFixture) {
    fixture
        .session
        .close(&background_context())
        .await
        .expect("the session closes");
}

/// Spawns the generation pass, waits for the parked hook's started gate,
/// runs the case's interlude between the gate and the release, releases
/// the hook, and joins the pass, the hook-boundary cases' choreography.
///
/// # Panics
/// The hook never starting or the joined pass failing, per `expect`.
/// Spawns the ready generation attempt, the orchestration suites'
/// concurrent pass; the handle joins the spawned attempt.
fn spawn_generation_run(
    fixture: &common::DriveFixture,
    ready: AssistantReadyOperation,
) -> tokio::task::JoinHandle<Result<ProcedureResult, LaneError>> {
    let lane = Arc::clone(&fixture.lane);
    let drive = Arc::clone(&fixture.drive);
    tokio::spawn(async move { run_generation(&lane, &drive, GenerationLeaf::Ready(ready)).await })
}

/// Spawns the checkpoint run, the drain suites' concurrent checkpoint.
fn spawn_checkpoint_run(
    fixture: &common::DriveFixture,
    checkpoint: CheckpointOperation,
) -> tokio::task::JoinHandle<Result<ProcedureResult, LaneError>> {
    let lane = Arc::clone(&fixture.lane);
    let drive = Arc::clone(&fixture.drive);
    tokio::spawn(async move { run_checkpoint(&lane, &drive, &checkpoint).await })
}

async fn parked_generation(
    fixture: &common::DriveFixture,
    ready: AssistantReadyOperation,
    hook_started_rx: tokio::sync::oneshot::Receiver<()>,
    release_hook: tokio::sync::oneshot::Sender<()>,
    interlude: impl FnOnce(&common::DriveFixture),
) -> Result<ProcedureResult, LaneError> {
    let generating = spawn_generation_run(fixture, ready);
    hook_started_rx.await.expect("the hook started");
    interlude(fixture);
    let _ = release_hook.send(());
    generating.await.expect("the generation runs")
}

/// Parks one hook between its started and release gates, upstream's
/// per-test parked handlers: the handler signals its start, waits out the
/// release, and returns the quiet result the caller passes — the
/// `before_request` handlers' `undefined` and the `transform_context`
/// handlers' absent replacement.
///
/// # Panics
/// The registration's failure.
fn park_hook(
    fixture: &common::DriveFixture,
    name: HookName,
    quiet_result: HookResult,
) -> (
    tokio::sync::oneshot::Receiver<()>,
    tokio::sync::oneshot::Sender<()>,
) {
    let (hook_started, hook_started_rx) = deferred();
    let (release_hook, release_hook_rx) = deferred();
    let started_cell = Arc::new(Mutex::new(Some(hook_started)));
    let release_cell = Arc::new(Mutex::new(Some(release_hook_rx)));
    fixture
        .hooks
        .on(
            name,
            Arc::new(move |_invocation, _context| {
                let started = lock(&started_cell).take();
                let release = lock(&release_cell).take();
                let quiet_result = quiet_result.clone();
                Box::pin(async move {
                    if let Some(started) = started {
                        let _ = started.send(());
                    }
                    if let Some(release) = release {
                        let _ = release.await;
                    }
                    Ok(quiet_result.clone())
                })
            }),
            HookOptions::default(),
        )
        .expect("register the parked hook");
    (hook_started_rx, release_hook)
}

/// The pending-assistant partial the frame fixtures build, upstream's
/// `{ ...fauxAssistantMessage([], { timestamp }), stopReason: "pending" }`.
#[must_use]
fn pending_partial(timestamp: i64) -> AssistantMessage {
    let mut message = faux_assistant_message(
        Vec::new(),
        FauxAssistantMessageOptions {
            timestamp: Some(timestamp),
            ..FauxAssistantMessageOptions::default()
        },
    );
    message.stop_reason = StopReason::Pending;
    message
}

/// The assistant message with one text block carrying `text`, the frame
/// fixtures' accumulator shape.
#[must_use]
fn partial_with_text(base: &AssistantMessage, text: &str) -> AssistantMessage {
    let mut message = base.clone();
    message.content = vec![AssistantBlock::Text(TextContent {
        text: text.to_owned(),
        text_signature: None,
    })];
    message
}

/// Upstream `describe "runtime generation checkpoint"` case "consumes
/// `before_run` and snapshots a ready generation": the injected message
/// entries into the initial checkpoint and the ready generation carries the
/// normalized retry policy.
#[tokio::test]
async fn consumes_before_run_and_snapshots_a_ready_generation() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    fixture
        .hooks
        .on(
            HookName::BeforeRun,
            Arc::new(|_invocation, _context| {
                Box::pin(async {
                    Ok(HookResult::BeforeRun(Some(BeforeRunResult {
                        messages: vec![common::user("injected", 2)],
                    })))
                })
            }),
            HookOptions::default(),
        )
        .expect("register before_run");

    let started = start_fixture_run(&fixture).await;
    assert!(
        matches!(started, ProcedureResult::Continue),
        "the run starts as a continue"
    );
    let OperationState::Checkpoint(run) = common::current_state(&fixture) else {
        panic!("missing checkpoint");
    };
    assert_eq!(
        Some(run.checkpoint.trigger_entry_id.clone()),
        fixture.lane.state().tip_id,
        "the checkpoint triggers from the lane tip"
    );
    assert!(
        lock(&fixture.events)
            .iter()
            .any(|event| event.payload.event_type() == HarnessEventType::EntryAdded),
        "the injected entry publishes"
    );
    common::expect_projection_restores(&fixture).await;

    let advanced = run_checkpoint(&fixture.lane, &fixture.drive, &run)
        .await
        .expect("the checkpoint advances");
    assert!(
        matches!(advanced, ProcedureResult::Continue),
        "the checkpoint continues"
    );
    let OperationState::AssistantReady(ready) = common::current_state(&fixture) else {
        panic!("missing ready generation")
    };
    assert_eq!(
        ready.generation_context.configuration,
        fixture.lane.state().configuration,
        "the generation carries the lane configuration"
    );
    assert_eq!(
        ready.generation_context.stream_options,
        AgentHarnessStreamOptions::default(),
        "the generation carries the snapshotted stream options"
    );
    assert_eq!(
        ready.generation_context.retry_policy,
        NormalizedRetryPolicy {
            max_attempts: 4,
            base_delay_ms: 1,
            max_agent_delay_ms: 60_000,
        },
        "the retry policy normalizes maxRetries + 1 with the default agent delay cap"
    );
    assert!(!ready.generation_context.overflow_recovery_used);
    common::expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

/// Upstream `describe "runtime generation checkpoint"` case "drains mixed
/// pending writes while separating leaf and generation trigger": one commit
/// drains both pending entries and the message entry triggers the
/// generation while the custom entry takes the tip.
#[tokio::test]
async fn drains_mixed_pending_writes_while_separating_leaf_and_generation_trigger() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    start_fixture_run(&fixture).await;
    let message_id = "01950000-0000-7000-8000-000000000010".to_owned();
    let custom_id = "01950000-0000-7000-8000-000000000011".to_owned();
    let items = vec![
        InboxItem {
            entry_id: message_id.clone(),
            kind: InboxItemKind::Write,
        },
        InboxItem {
            entry_id: custom_id.clone(),
            kind: InboxItemKind::Write,
        },
    ];
    let message_pending = PendingEntry::Message {
        payload: Box::new(common::user("projecting", 3)),
    };
    let custom_pending = PendingEntry::Custom {
        custom_type: "display-only".to_owned(),
        payload: Some(json!({ "value": true })),
    };
    let operation_id = fixture.operation_id.clone();
    let command_message_id = message_id.clone();
    let command_custom_id = custom_id.clone();
    fixture
        .lane
        .command::<(), _>(
            move |state, _session, _context| {
                let operation_id = operation_id.clone();
                let items = items.clone();
                let message_id = command_message_id.clone();
                let custom_id = command_custom_id.clone();
                let message_pending = message_pending.clone();
                let custom_pending = custom_pending.clone();
                Box::pin(async move {
                    assert!(state.operation.is_some(), "missing operation");
                    Ok(LaneCommand::Commit {
                        writes: vec![
                            set_value_write(&pending_entry(&message_id), message_pending)
                                .expect("the pending write"),
                            set_value_write(&pending_entry(&custom_id), custom_pending)
                                .expect("the pending write"),
                            crate::harness::runtime::test_support::lane_state_write(
                                "main",
                                Some(&operation_id),
                                None,
                                items.clone(),
                            )
                            .expect("the lane state write"),
                        ],
                        next: LaneState {
                            inbox: items,
                            ..state
                        },
                        materialize: Arc::new(
                            |_: &crate::harness::session::types::CommitResult| (),
                        ),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the pending writes commit");
    let OperationState::Checkpoint(run) = common::current_state(&fixture) else {
        panic!("missing checkpoint");
    };
    fixture.storage.clear_commit_attempts();

    run_checkpoint(&fixture.lane, &fixture.drive, &run)
        .await
        .expect("the checkpoint advances");

    assert_eq!(
        fixture.storage.get_commit_attempts().len(),
        1,
        "one commit carries the whole drain"
    );
    let OperationState::AssistantReady(routed) = common::current_state(&fixture) else {
        panic!("pending writes did not route to generation");
    };
    assert_eq!(
        fixture.lane.state().tip_id.as_deref(),
        Some(custom_id.as_str()),
        "the custom entry takes the tip"
    );
    assert_eq!(
        routed.generation_context.trigger_entry_id, message_id,
        "the message entry triggers the generation"
    );
    assert!(!routed.generation_context.overflow_recovery_used);
    assert!(fixture.lane.state().inbox.is_empty(), "the inbox drains");
    common::expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins the custom-only write's projection and the follow-up's re-entry parentage"
)]
/// Upstream `describe "runtime generation checkpoint"` case "preserves
/// checkpoint routing when a custom-only pending write does not project":
/// the finish hook's follow-up re-enters generation from a new entry
/// parented on the custom entry.
#[tokio::test]
async fn preserves_checkpoint_routing_when_a_custom_only_pending_write_does_not_project() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    start_fixture_run(&fixture).await;
    let OperationState::Checkpoint(current) = common::current_state(&fixture) else {
        panic!("missing checkpoint");
    };
    let custom_id = "01950000-0000-7000-8000-000000000012".to_owned();
    let checkpoint = CheckpointOperation {
        checkpoint: CheckpointData {
            continuation: Continuation::MayFinish {
                include_final_assistant: false,
            },
            ..current.checkpoint.clone()
        },
        ..current.clone()
    };
    let operation_id = fixture.operation_id.clone();
    let command_custom_id = custom_id.clone();
    let command_checkpoint = checkpoint.clone();
    fixture
        .lane
        .command::<(), _>(
            move |state, _session, _context| {
                let operation_id = operation_id.clone();
                let custom_id = command_custom_id.clone();
                let checkpoint = command_checkpoint.clone();
                Box::pin(async move {
                    let Some(operation) = state.operation.as_ref() else {
                        panic!("missing operation");
                    };
                    let inbox = [
                        state.inbox.clone(),
                        vec![InboxItem {
                            entry_id: custom_id.clone(),
                            kind: InboxItemKind::Write,
                        }],
                    ]
                    .concat();
                    Ok(LaneCommand::Commit {
                        writes: vec![
                            set_value_write(
                                &pending_entry(&custom_id),
                                PendingEntry::Custom {
                                    custom_type: "display-only".to_owned(),
                                    payload: Some(json!({ "value": true })),
                                },
                            )
                            .expect("the pending write"),
                            set_value_write(
                                &operation_state(&operation_id),
                                OperationState::Checkpoint(checkpoint.clone()),
                            )
                            .expect("the state write"),
                            crate::harness::runtime::test_support::lane_state_write(
                                "main",
                                Some(&operation_id),
                                None,
                                inbox.clone(),
                            )
                            .expect("the lane state write"),
                        ],
                        next: LaneState {
                            inbox,
                            operation: Some(LiveOperation {
                                meta: operation.meta.clone(),
                                state: OperationState::Checkpoint(checkpoint.clone()),
                            }),
                            ..state
                        },
                        materialize: Arc::new(
                            |_: &crate::harness::session::types::CommitResult| (),
                        ),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the checkpoint commit serves");
    fixture
        .hooks
        .on(
            HookName::BeforeRunEnd,
            Arc::new(|_invocation, _context| {
                Box::pin(async {
                    Ok(HookResult::BeforeRunEnd(Some(FollowUpResult {
                        follow_up: "continue after custom write".to_owned(),
                    })))
                })
            }),
            HookOptions::default(),
        )
        .expect("register before_run_end");

    let advanced = run_checkpoint(&fixture.lane, &fixture.drive, &checkpoint)
        .await
        .expect("the checkpoint advances");
    assert!(
        matches!(advanced, ProcedureResult::Continue),
        "the checkpoint continues"
    );

    let OperationState::AssistantReady(ready) = common::current_state(&fixture) else {
        panic!("hook follow-up did not route to generation");
    };
    let follow_up_id = ready.generation_context.trigger_entry_id.clone();
    assert_eq!(
        fixture.lane.state().tip_id.as_deref(),
        Some(follow_up_id.as_str()),
        "the follow-up entry takes the tip"
    );
    let custom = fixture
        .session
        .get_entry(&custom_id, &background_context())
        .await
        .expect("the entry read")
        .expect("the custom entry");
    assert!(
        matches!(custom, Entry::Custom { .. }),
        "the custom entry stays custom"
    );
    let follow_up = fixture
        .session
        .get_entry(&follow_up_id, &background_context())
        .await
        .expect("the entry read")
        .expect("the follow-up entry");
    assert_eq!(
        follow_up.parent_id(),
        Some(custom_id.as_str()),
        "the follow-up parents on the custom entry"
    );
    let Entry::Message { body, .. } = follow_up else {
        panic!("the follow-up is a message entry");
    };
    let AgentMessage::Standard(Message::User(user_message)) = &body.message else {
        panic!("the follow-up is a user message");
    };
    assert_eq!(
        user_message.content,
        UserContent::Text("continue after custom write".to_owned()),
        "the follow-up carries the hook's prompt"
    );
    assert!(fixture.lane.state().inbox.is_empty());
    let staged = fixture
        .session
        .get_value(&pending_entry(&custom_id).address, &background_context())
        .await
        .expect("the pending read");
    assert!(staged.is_none(), "the pending entry deletes");
    common::expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins the stale hook follow-up's replacement by the input arriving during the hook"
)]
/// Upstream `describe "runtime generation checkpoint"` case "drops a stale
/// finish-hook follow-up when input arrives during the hook": the steer
/// committed while the hook parks replaces the hook's follow-up as the
/// generation trigger and tip.
#[tokio::test]
async fn drops_a_stale_finish_hook_follow_up_when_input_arrives_during_the_hook() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    start_fixture_run(&fixture).await;
    let OperationState::Checkpoint(current) = common::current_state(&fixture) else {
        panic!("missing checkpoint");
    };
    let checkpoint = CheckpointOperation {
        checkpoint: CheckpointData {
            continuation: Continuation::MayFinish {
                include_final_assistant: false,
            },
            ..current.checkpoint.clone()
        },
        ..current.clone()
    };
    let operation_id = fixture.operation_id.clone();
    let command_checkpoint = checkpoint.clone();
    fixture
        .lane
        .command::<(), _>(
            move |state, _session, _context| {
                let operation_id = operation_id.clone();
                let checkpoint = command_checkpoint.clone();
                Box::pin(async move {
                    let Some(operation) = state.operation.as_ref() else {
                        panic!("missing operation");
                    };
                    Ok(LaneCommand::Commit {
                        writes: vec![
                            set_value_write(
                                &operation_state(&operation_id),
                                OperationState::Checkpoint(checkpoint.clone()),
                            )
                            .expect("the state write"),
                        ],
                        next: LaneState {
                            operation: Some(LiveOperation {
                                meta: operation.meta.clone(),
                                state: OperationState::Checkpoint(checkpoint.clone()),
                            }),
                            ..state
                        },
                        materialize: Arc::new(
                            |_: &crate::harness::session::types::CommitResult| (),
                        ),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the checkpoint commit serves");
    let (hook_started, hook_started_rx) = deferred();
    let (release_hook, release_hook_rx) = deferred();
    let started_cell = Arc::new(Mutex::new(Some(hook_started)));
    let release_cell = Arc::new(Mutex::new(Some(release_hook_rx)));
    fixture
        .hooks
        .on(
            HookName::BeforeRunEnd,
            Arc::new(move |_invocation, _context| {
                let started = lock(&started_cell).take();
                let release = lock(&release_cell).take();
                Box::pin(async move {
                    if let Some(started) = started {
                        let _ = started.send(());
                    }
                    if let Some(release) = release {
                        let _ = release.await;
                    }
                    Ok(HookResult::BeforeRunEnd(Some(FollowUpResult {
                        follow_up: "stale hook follow-up".to_owned(),
                    })))
                })
            }),
            HookOptions::default(),
        )
        .expect("register before_run_end");

    let running = spawn_checkpoint_run(&fixture, checkpoint);
    hook_started_rx.await.expect("the hook started");
    common::queue_pending_entries(
        &fixture,
        vec![(
            "steer-during-finish",
            PendingEntry::Message {
                payload: Box::new(common::user("new input", 4)),
            },
            InboxItemKind::Steer,
        )],
    )
    .await;
    let _ = release_hook.send(());

    let advanced = running
        .await
        .expect("the checkpoint advances")
        .expect("the checkpoint continues");
    assert!(
        matches!(advanced, ProcedureResult::Continue),
        "the checkpoint continues"
    );
    let OperationState::AssistantReady(ready) = common::current_state(&fixture) else {
        panic!("new input did not replace stale finish decision");
    };
    assert_eq!(
        ready.generation_context.trigger_entry_id, "steer-during-finish",
        "the steer entry triggers the generation"
    );
    assert_eq!(
        fixture.lane.state().tip_id.as_deref(),
        Some("steer-during-finish"),
        "the steer entry takes the tip"
    );
    let steered = fixture
        .session
        .get_entry("steer-during-finish", &background_context())
        .await
        .expect("the entry read")
        .expect("the steer entry");
    assert!(
        matches!(steered, Entry::Message { .. }),
        "the steer entry is a message entry"
    );
    close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins the admission-ordered commit families and the reserved ids' settlement"
)]
/// Upstream `describe "runtime assistant generation"` case "commits intent
/// before provider admission, preserves queued inbox state, and settles
/// reserved ids": the provider observes the durable intent already at
/// `assistant.effect_pending`, the late steer survives in the inbox, and
/// the settlement commit's write families are exact.
#[tokio::test]
async fn commits_intent_before_provider_admission_preserves_queued_inbox_state_and_settles_reserved_ids()
 {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    let (hook_started_rx, release_hook) = park_hook(
        &fixture,
        HookName::BeforeRequest,
        HookResult::BeforeRequest(None),
    );
    let effect_pending_at_provider = Arc::new(AtomicBool::new(false));
    let session = Arc::clone(&fixture.session);
    let recorded = Arc::clone(&effect_pending_at_provider);
    fixture
        .faux
        .set_responses([FauxResponseStep::Factory(Arc::new(
            move |_context: &AiContext,
                  _options: Option<&SimpleStreamOptions>,
                  _state: &pi_ai::providers::faux::FauxProviderState,
                  _model: &Model| {
                let session = Arc::clone(&session);
                let recorded = Arc::clone(&recorded);
                Box::pin(async move {
                    let stored = session
                        .get_value(
                            &operation_state(common::OPERATION_ID).address,
                            &background_context(),
                        )
                        .await
                        .map_err(|error| {
                            let error: pi_ai::providers::faux::FauxFactoryError = Box::new(error);
                            error
                        })?;
                    let effect_pending = stored.is_some_and(|stored| {
                        stored.value.get("at").and_then(JsonValue::as_str)
                            == Some("assistant.effect_pending")
                    });
                    recorded.store(effect_pending, Ordering::SeqCst);
                    Ok(faux_assistant_message(
                        "answer",
                        FauxAssistantMessageOptions {
                            timestamp: Some(5),
                            ..FauxAssistantMessageOptions::default()
                        },
                    ))
                })
            },
        ))]);
    fixture.storage.clear_commit_attempts();
    let generating = spawn_generation_run(&fixture, ready);
    hook_started_rx.await.expect("the hook started");
    let steer_id = "01950000-0000-7000-8000-000000000020".to_owned();
    let operation_id = fixture.operation_id.clone();
    let command_steer_id = steer_id.clone();
    fixture
        .lane
        .command::<(), _>(
            move |state, _session, _context| {
                let steer_id = command_steer_id.clone();
                let operation_id = operation_id.clone();
                Box::pin(async move {
                    assert!(state.operation.is_some(), "missing operation");
                    let inbox = [
                        state.inbox.clone(),
                        vec![InboxItem {
                            entry_id: steer_id.clone(),
                            kind: InboxItemKind::Steer,
                        }],
                    ]
                    .concat();
                    Ok(LaneCommand::Commit {
                        writes: vec![
                            set_value_write(
                                &pending_entry(&steer_id),
                                PendingEntry::Message {
                                    payload: Box::new(common::user("late steer", 4)),
                                },
                            )
                            .expect("the pending write"),
                            crate::harness::runtime::test_support::lane_state_write(
                                "main",
                                Some(&operation_id),
                                None,
                                inbox.clone(),
                            )
                            .expect("the lane state write"),
                        ],
                        next: LaneState { inbox, ..state },
                        materialize: Arc::new(
                            |_: &crate::harness::session::types::CommitResult| (),
                        ),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the steer commits");
    let _ = release_hook.send(());

    let generating = generating
        .await
        .expect("the generation runs")
        .expect("the generation continues");
    assert!(
        matches!(generating, ProcedureResult::Continue),
        "the generation continues"
    );
    assert!(
        effect_pending_at_provider.load(Ordering::SeqCst),
        "the intent precedes admission"
    );
    let OperationState::Checkpoint(settled) = common::current_state(&fixture) else {
        panic!("response did not reach checkpoint");
    };
    assert_eq!(
        fixture.lane.state().inbox,
        vec![InboxItem {
            entry_id: steer_id.clone(),
            kind: InboxItemKind::Steer,
        }],
        "the late steer survives in the inbox"
    );
    let response_id = settled
        .scope
        .latest_assistant_entry_id
        .clone()
        .expect("missing response id");
    let entry = fixture
        .session
        .get_entry(&response_id, &background_context())
        .await
        .expect("the entry read")
        .expect("the response entry");
    let Entry::Message { body, .. } = &entry else {
        panic!("the response entry is a message entry");
    };
    let AgentMessage::Standard(Message::Assistant(message)) = &body.message else {
        panic!("the response entry carries an assistant message");
    };
    assert_eq!(
        message.content,
        vec![AssistantBlock::Text(TextContent {
            text: "answer".to_owned(),
            text_signature: None,
        })],
        "the response entry carries the settled text"
    );
    let frames = fixture
        .session
        .read_list(
            &pending_assistant_frames(common::OPERATION_ID, &response_id).address,
            None,
            &background_context(),
        )
        .await
        .expect("the frames read");
    assert!(frames.is_empty(), "the frame list settles empty");
    let attempts = fixture.storage.get_commit_attempts();
    assert!(
        attempts
            .iter()
            .any(|writes| writes_operation_state(writes, "effect_pending")),
        "the intent commits effect_pending"
    );
    let settlement = attempts
        .iter()
        .find(|writes| writes.iter().any(|write| matches!(write, Write::Entry(_))))
        .expect("the settlement commits");
    let families: Vec<String> = settlement.iter().map(write_family).collect();
    assert_eq!(
        families,
        [
            "entry",
            "usage",
            "value:set:pi.branch.tip",
            "list:delete:pi.pending.assistant_frame",
            "value:set:pi.op.state",
        ],
        "the settlement commit's write families are exact"
    );
    let types: Vec<&str> = lock(&fixture.events)
        .iter()
        .map(|event| event.payload.event_type().as_str())
        .collect();
    for expected in [
        "turn_start",
        "message_start",
        "message_update",
        "message_end",
        "entry_added",
        "usage",
        "turn_end",
    ] {
        assert!(types.contains(&expected), "the events include {expected}");
    }
    common::expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

/// The scripted `stream_simple` body one boundary case installs, upstream's
/// per-case `streamSimple` overrides: the closure builds the event stream
/// the request consumes.
type SimpleStreamScript =
    Arc<dyn Fn(&Model, Option<&SimpleStreamOptions>) -> AssistantMessageEventStream + Send + Sync>;

/// The scripted simple-stream provider over the fixture's faux core,
/// upstream's wrapper-provider literals.
/// Runs the ready generation attempt, the orchestration suites' pass; an
/// unexpected failure panics the test by design.
async fn generation_attempt(
    fixture: &common::DriveFixture,
    ready: AssistantReadyOperation,
    why: &str,
) -> ProcedureResult {
    run_generation(&fixture.lane, &fixture.drive, GenerationLeaf::Ready(ready))
        .await
        .expect(why)
}

fn scripted_provider(
    fixture: &common::DriveFixture,
    script: SimpleStreamScript,
) -> Arc<ScriptedSimpleProvider> {
    Arc::new(ScriptedSimpleProvider {
        base: fixture.faux.provider.clone(),
        script,
    })
}

/// The provider whose `stream_simple` runs a case's script; the catalog and
/// plain `stream` delegate to the faux provider, upstream's wrapper
/// providers.
struct ScriptedSimpleProvider {
    base: ProviderImpl,
    script: SimpleStreamScript,
}

impl Provider for ScriptedSimpleProvider {
    provider_pass_through!(base);

    fn stream_simple(
        &self,
        model: &Model,
        _context: &AiContext,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        (self.script)(model, options)
    }
}

/// The `stream_simple` script the payload-hook boundary cases build: the
/// stream hands the request's `on_payload` seam the fixed payload — parked
/// between the reached and release gates when armed — records what the seam
/// returned, then starts and stops the message, upstream's providers
/// invoking `onPayload`.
/// The `before_payload` pass's fixture: the drive fixture over the
/// scripted provider capturing the payload, the hook passes' prelude;
/// returns (fixture, ready).
async fn payload_script_fixture(
    returned: Arc<Mutex<Vec<Option<JsonValue>>>>,
) -> (common::DriveFixture, AssistantReadyOperation) {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    fixture.models.set_provider(scripted_provider(
        &fixture,
        on_payload_script(
            returned,
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(None)),
        ),
    ));
    (fixture, ready)
}

fn on_payload_script(
    returned: Arc<Mutex<Vec<Option<JsonValue>>>>,
    reached: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    release: Arc<Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
) -> SimpleStreamScript {
    Arc::new(
        move |model: &Model, options: Option<&SimpleStreamOptions>| {
            let on_payload =
                options.and_then(|options| options.transport_options.on_payload.clone());
            let reached = lock(&reached).take();
            let release = lock(&release).take();
            let returned = Arc::clone(&returned);
            let model = model.clone();
            let stream = assistant_message_event_stream();
            let pusher = stream.clone();
            tokio::spawn(async move {
                if let Some(reached) = reached {
                    let _ = reached.send(());
                }
                if let Some(release) = release {
                    let _ = release.await;
                }
                let payload = json!({ "prompt": "generation payload" });
                let outcome = match on_payload {
                    Some(on_payload) => on_payload.call(payload, model).await,
                    None => Some(payload),
                };
                lock(&returned).push(outcome);
                pusher.push(AssistantMessageEvent::Start {
                    partial: pending_partial(5),
                });
                pusher.push(AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message: faux_assistant_message(
                        "answer",
                        FauxAssistantMessageOptions {
                            timestamp: Some(5),
                            ..FauxAssistantMessageOptions::default()
                        },
                    ),
                });
            });
            stream
        },
    )
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins the event-loop timing and the frames' persisted order"
)]
/// Upstream `describe "runtime assistant generation"` case "does not await
/// frame storage inside the provider event loop and persists frames in
/// order": the provider's stream keeps emitting while the frame commit
/// parks, and the recorded frames are the start, start-of-text, and
/// end-of-text frames in order.
#[tokio::test]
async fn does_not_await_frame_storage_inside_the_provider_event_loop_and_persists_frames_in_order()
{
    let memory = Arc::new(MemoryStorage::new(MemoryStorageOptions {
        now: Some(Arc::new(|| 100)),
    }));
    let backend = Arc::new(common::FrameBlockingMemoryStorage::new(memory));
    let storage: Arc<dyn Storage> = backend.clone();
    let fixture = common::create_drive_fixture(generation_spec(Some(storage))).await;
    let ready = advance_to_ready(&fixture).await;
    // The frame-streaming script upstream's `streamSimple` override builds:
    // `start`/`text_start`/`text_delta`/`text_end`/`done` pushed off the
    // call, the accumulator already carrying `"ab"` where upstream's shared
    // `text` mutation made the encoder observe it.
    let script: SimpleStreamScript = Arc::new(
        move |_model: &Model, _options: Option<&SimpleStreamOptions>| {
            let stream = assistant_message_event_stream();
            let pusher = stream.clone();
            tokio::spawn(async move {
                let empty = pending_partial(5);
                pusher.push(AssistantMessageEvent::Start {
                    partial: empty.clone(),
                });
                let started = partial_with_text(&empty, "");
                pusher.push(AssistantMessageEvent::TextStart {
                    content_index: 0,
                    partial: partial_with_text(&started, "ab"),
                });
                let grown = partial_with_text(&started, "ab");
                pusher.push(AssistantMessageEvent::TextDelta {
                    content_index: 0,
                    delta: "ab".to_owned(),
                    partial: grown.clone(),
                });
                pusher.push(AssistantMessageEvent::TextEnd {
                    content_index: 0,
                    content: "ab".to_owned(),
                    partial: grown,
                });
                pusher.push(AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message: faux_assistant_message(
                        "ab",
                        FauxAssistantMessageOptions {
                            timestamp: Some(5),
                            ..FauxAssistantMessageOptions::default()
                        },
                    ),
                });
            });
            stream
        },
    );
    fixture
        .models
        .set_provider(scripted_provider(&fixture, script));
    backend.arm_block_next_frame();
    fixture.storage.clear_commit_attempts();

    let generating = spawn_generation_run(&fixture, ready);
    backend
        .take_frame_started()
        .await
        .expect("the frame park signals");
    common::wait_for(|| {
        lock(&fixture.events)
            .iter()
            .filter(|event| event.payload.event_type() == HarnessEventType::MessageUpdate)
            .count()
            > 1
    })
    .await;
    backend.release_frame();
    let _ = generating
        .await
        .expect("the generation runs")
        .expect("the generation continues");

    let frames: Vec<JsonValue> = fixture
        .storage
        .get_commit_attempts()
        .into_iter()
        .flat_map(|writes| {
            writes.into_iter().filter_map(|write| match write {
                Write::ListAppend(list)
                    if list.kind == "list"
                        && list.op == "append"
                        && list.namespace == "pi.pending.assistant_frame" =>
                {
                    Some(list.value)
                }
                _ => None,
            })
        })
        .collect();
    let frame_types: Vec<Option<&str>> = frames
        .iter()
        .map(|frame| frame.get("type").and_then(JsonValue::as_str))
        .collect();
    assert_eq!(
        frame_types,
        [Some("start"), Some("text_start"), Some("text_end")],
        "the frames persist in order"
    );
    let start_content = frames[0]
        .get("partial")
        .and_then(|partial| partial.get("content"))
        .and_then(JsonValue::as_array)
        .expect("the start frame carries its partial");
    assert!(
        start_content.is_empty(),
        "the start frame's partial content is empty"
    );
    let started_text = frames[1]
        .get("content")
        .and_then(|content| content.get("text"))
        .and_then(JsonValue::as_str)
        .expect("the text_start frame carries its content");
    assert_eq!(
        started_text, "ab",
        "the text_start frame carries the grown text"
    );
    close_session(&fixture).await;
}

/// Upstream `describe "runtime assistant generation"` case "enters
/// configuration failure without reserving response ids or calling the
/// provider": the unavailable active tools fail the run with
/// `configured_tools_unavailable` and the queued write survives.
#[tokio::test]
async fn enters_configuration_failure_without_reserving_response_ids_or_calling_the_provider() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    fixture
        .lane
        .set_active_tools_impl(vec!["missing".to_owned()], &background_context())
        .await
        .expect("the active tools set");
    let ready = advance_to_ready(&fixture).await;
    let queued_id = fixture
        .lane
        .append_custom_entry_impl(
            "after-failure",
            Some(json!({ "retained": true })),
            &background_context(),
        )
        .await
        .expect("the entry appends");
    fixture.storage.clear_commit_attempts();

    let settled = generation_attempt(&fixture, ready, "the generation runs").await;
    let ProcedureResult::Settled { outcome } = settled else {
        panic!("the configuration failure settles");
    };
    assert_eq!(outcome.operation_id, common::OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Run);
    assert_eq!(outcome.status, TerminalStatus::Failed);
    let error = outcome
        .error
        .as_ref()
        .expect("the failure carries its error");
    assert_eq!(error.code, "configured_tools_unavailable");
    assert_eq!(
        error.details.as_ref(),
        Some(&json!({ "tools": ["missing"] })),
        "the failure carries the missing tools"
    );
    assert!(
        fixture.lane.state().operation.is_none(),
        "the operation clears"
    );
    assert_eq!(
        fixture.lane.state().inbox,
        vec![InboxItem {
            entry_id: queued_id.clone(),
            kind: InboxItemKind::Write,
        }],
        "the queued write survives"
    );
    let staged = fixture
        .session
        .get_value(&pending_entry(&queued_id).address, &background_context())
        .await
        .expect("the pending read");
    assert!(staged.is_some(), "the pending entry stays");
    assert_eq!(
        fixture.faux.state().call_count(),
        0,
        "the provider never runs"
    );
    let wrote_entries_or_usage = fixture
        .storage
        .get_commit_attempts()
        .iter()
        .flatten()
        .any(|write| matches!(write, Write::Entry(_) | Write::Usage(_)));
    assert!(!wrote_entries_or_usage, "no entry or usage writes");
    common::expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

/// Upstream `describe "runtime assistant generation"` case "fails a missing
/// captured model before reserving ids": the deleted provider fails the run
/// with `model_unavailable`.
#[tokio::test]
async fn fails_a_missing_captured_model_before_reserving_ids() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    fixture.models.delete_provider(fixture.faux.provider.id());
    fixture.storage.clear_commit_attempts();

    let settled = generation_attempt(&fixture, ready, "the generation runs").await;
    let ProcedureResult::Settled { outcome } = settled else {
        panic!("the failure settles");
    };
    assert_eq!(outcome.operation_id, common::OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Run);
    assert_eq!(outcome.status, TerminalStatus::Failed);
    assert_eq!(
        outcome.error.as_ref().map(|error| error.code.as_str()),
        Some("model_unavailable"),
        "the failure is model_unavailable"
    );
    assert!(
        fixture.lane.state().operation.is_none(),
        "the operation clears"
    );
    assert_eq!(
        fixture.faux.state().call_count(),
        0,
        "the provider never runs"
    );
    let wrote_entries_or_usage = fixture
        .storage
        .get_commit_attempts()
        .iter()
        .flatten()
        .any(|write| matches!(write, Write::Entry(_) | Write::Usage(_)));
    assert!(!wrote_entries_or_usage, "no entry or usage writes");
    common::expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

/// Upstream `describe "runtime assistant generation"` case "declines intent
/// when cancellation wins preparation": the cancel committed during the
/// parked hook downgrades the generation to a plain continue without an
/// effect-pending intent or a provider call.
#[tokio::test]
async fn declines_intent_when_cancellation_wins_preparation() {
    let cancelled = common::create_drive_fixture(generation_spec(None)).await;
    let cancelled_ready = advance_to_ready(&cancelled).await;
    let (hook_started_rx, release_hook) = park_hook(
        &cancelled,
        HookName::BeforeRequest,
        HookResult::BeforeRequest(None),
    );
    cancelled.storage.clear_commit_attempts();
    let lane = Arc::clone(&cancelled.lane);
    let drive = Arc::clone(&cancelled.drive);
    let cancelling = tokio::spawn(async move {
        run_generation(&lane, &drive, GenerationLeaf::Ready(cancelled_ready)).await
    });
    hook_started_rx.await.expect("the hook started");
    let operation = cancelled
        .lane
        .state()
        .operation
        .expect("fixture has no operation");
    let next_state = operation_state_with_scope(
        &operation.state,
        OperationScope {
            control: Control::CancelRequested { requested_at: 10 },
            ..operation_scope_of(&operation.state)
        },
    );
    let operation_id = cancelled.operation_id.clone();
    let meta = operation.meta.clone();
    cancelled
        .lane
        .command::<(), _>(
            move |state, _session, _context| {
                let operation_id = operation_id.clone();
                let meta = meta.clone();
                let next_state = next_state.clone();
                Box::pin(async move {
                    Ok(LaneCommand::Commit {
                        writes: vec![
                            set_value_write(&operation_state(&operation_id), next_state.clone())
                                .expect("the cancel write"),
                        ],
                        next: LaneState {
                            operation: Some(LiveOperation {
                                meta,
                                state: next_state,
                            }),
                            ..state
                        },
                        materialize: Arc::new(
                            |_: &crate::harness::session::types::CommitResult| (),
                        ),
                        events: None,
                    })
                })
            },
            &background_context(),
        )
        .await
        .expect("the cancel commits");
    let _ = release_hook.send(());

    let declined = cancelling
        .await
        .expect("the generation runs")
        .expect("the generation continues");
    assert!(
        matches!(declined, ProcedureResult::Continue),
        "the cancellation declines to a continue"
    );
    assert_eq!(
        cancelled.faux.state().call_count(),
        0,
        "the provider never runs"
    );
    assert!(
        !cancelled
            .storage
            .get_commit_attempts()
            .iter()
            .any(|writes| writes_operation_state(writes, "effect_pending")),
        "no effect_pending intent commits"
    );
    close_session(&cancelled).await;
}

/// Upstream `describe "runtime assistant generation"` case "recovers an
/// orphan with no frames into a zero-usage retry wait": the seeded
/// effect-pending leaf recovers into the fresh `unknown`-api error
/// settlement and waits for attempt 2 without a provider call.
#[tokio::test]
async fn recovers_an_orphan_with_no_frames_into_a_zero_usage_retry_wait() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    let response_entry_id = "01950000-0000-7000-8000-000000000028".to_owned();
    let usage_id = "01950000-0000-7000-8000-000000000029".to_owned();
    let mut generation_context = ready.generation_context.clone();
    generation_context.retry_policy = NormalizedRetryPolicy {
        max_attempts: 2,
        base_delay_ms: 1,
        max_agent_delay_ms: 30_000,
    };
    let pending = AssistantEffectPendingOperation {
        scope: operation_scope_of(&OperationState::AssistantReady(ready.clone())),
        generation_context,
        attempt: 1,
        response_entry_id: response_entry_id.clone(),
        usage_id,
        intended_output_limit: 100,
        context_window: 1_000,
    };
    common::commit_next_operation_state(
        &fixture.lane,
        &fixture.operation_id,
        Vec::new(),
        {
            let pending = pending.clone();
            move |_| OperationState::AssistantEffectPending(pending.clone())
        },
        "the pending state commits",
    )
    .await;

    recover_assistant_generation(&fixture.lane, &fixture.drive, &pending)
        .await
        .expect("the recovery settles");

    let message =
        common::assistant_message_entry(&fixture.session, &response_entry_id, "the response entry")
            .await;
    assert_eq!(
        message.api,
        Api("unknown".to_owned()),
        "the fresh settlement carries the unknown api"
    );
    assert_eq!(
        message.provider,
        ProviderId(
            ready
                .generation_context
                .configuration
                .model
                .provider
                .clone()
        ),
        "the fresh settlement carries the captured provider"
    );
    assert_eq!(
        message.model, ready.generation_context.configuration.model.model_id,
        "the fresh settlement carries the captured model"
    );
    assert!(
        message.content.is_empty(),
        "the fresh settlement carries no content"
    );
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.usage.total_tokens, 0,
        "the settlement carries zero usage"
    );
    let OperationState::AssistantRetryWait(wait) = common::current_state(&fixture) else {
        panic!("fixture has no ready generation");
    };
    assert_eq!(
        wait.retry_wait.next_attempt, 2,
        "the retry wait plans attempt 2"
    );
    assert_eq!(
        fixture.faux.state().call_count(),
        0,
        "the provider never runs"
    );
    common::expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins the committed frames' settlement without a provider or response hook"
)]
/// Upstream `describe "runtime assistant generation"` case "recovers an
/// orphan from committed frames without a provider or response hook": the
/// committed frame prefix settles as the interrupted error message with
/// zero usage and the recovery-flagged event set is exact.
#[tokio::test]
async fn recovers_an_orphan_from_committed_frames_without_a_provider_or_response_hook() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    let response_entry_id = "01950000-0000-7000-8000-000000000030".to_owned();
    let usage_id = "01950000-0000-7000-8000-000000000031".to_owned();
    let mut generation_context = ready.generation_context.clone();
    generation_context.retry_policy = NormalizedRetryPolicy {
        max_attempts: 1,
        base_delay_ms: 1,
        max_agent_delay_ms: 30_000,
    };
    let pending = AssistantEffectPendingOperation {
        scope: operation_scope_of(&OperationState::AssistantReady(ready.clone())),
        generation_context,
        attempt: 1,
        response_entry_id: response_entry_id.clone(),
        usage_id,
        intended_output_limit: 100,
        context_window: 1_000,
    };
    let frames_address = pending_assistant_frames(common::OPERATION_ID, &response_entry_id);
    let leading_writes = vec![
        append_list_write(
            &frames_address,
            AssistantMessageFrame::Start {
                partial: pending_partial(6),
            },
        )
        .expect("the frame write"),
        append_list_write(
            &frames_address,
            AssistantMessageFrame::TextStart {
                content_index: 0,
                content: TextContent {
                    text: String::new(),
                    text_signature: None,
                },
            },
        )
        .expect("the frame write"),
        append_list_write(
            &frames_address,
            AssistantMessageFrame::TextDelta {
                content_index: 0,
                delta: "draft".to_owned(),
            },
        )
        .expect("the frame write"),
        append_list_write(
            &frames_address,
            AssistantMessageFrame::TextEnd {
                content_index: 0,
                content: "corrected".to_owned(),
                text_signature: None,
            },
        )
        .expect("the frame write"),
    ];
    common::commit_next_operation_state(
        &fixture.lane,
        &fixture.operation_id,
        leading_writes,
        {
            let pending = pending.clone();
            move |_| OperationState::AssistantEffectPending(pending.clone())
        },
        "the pending frames commit",
    )
    .await;
    let after_response_calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&after_response_calls);
    fixture
        .hooks
        .on(
            HookName::AfterResponse,
            Arc::new(move |_invocation, _context| {
                let counted = Arc::clone(&counted);
                Box::pin(async move {
                    counted.fetch_add(1, Ordering::SeqCst);
                    Ok(HookResult::AfterResponse(None))
                })
            }),
            HookOptions::default(),
        )
        .expect("register after_response");

    let settled = recover_assistant_generation(&fixture.lane, &fixture.drive, &pending)
        .await
        .expect("the recovery settles");
    let ProcedureResult::Settled { outcome } = settled else {
        panic!("the recovery settles");
    };
    assert_eq!(outcome.operation_id, common::OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Run);
    assert_eq!(outcome.status, TerminalStatus::Failed);
    assert_eq!(
        outcome.tip_id.as_deref(),
        Some(response_entry_id.as_str()),
        "the failed tip is the response entry"
    );
    assert_eq!(
        fixture.faux.state().call_count(),
        0,
        "the provider never runs"
    );
    assert_eq!(
        after_response_calls.load(Ordering::SeqCst),
        0,
        "the response hook never runs"
    );
    let message =
        common::assistant_message_entry(&fixture.session, &response_entry_id, "the response entry")
            .await;
    assert_eq!(
        message.content,
        vec![AssistantBlock::Text(TextContent {
            text: "corrected".to_owned(),
            text_signature: None,
        })],
        "the settlement reduces the committed frames"
    );
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        (message.usage.input, message.usage.output),
        (0, 0),
        "the settlement carries zero usage"
    );
    assert!(
        fixture.lane.state().operation.is_none(),
        "the operation clears"
    );
    let frames = fixture
        .session
        .read_list(
            &pending_assistant_frames(common::OPERATION_ID, &response_entry_id).address,
            None,
            &background_context(),
        )
        .await
        .expect("the frames read");
    assert!(frames.is_empty(), "the frame list deletes");
    let recovery_events: Vec<&str> = lock(&fixture.events)
        .iter()
        .filter(|event| event.recovery)
        .map(|event| event.payload.event_type().as_str())
        .collect();
    assert_eq!(
        recovery_events,
        ["message_start", "message_end", "entry_added"],
        "the recovery-flagged events are exact"
    );
    common::expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

/// Upstream `describe "runtime assistant generation"` case "finishes the
/// no-tool run with terminal cleanup and an immutable result record": the
/// settled run reads its result record without an entry scan and the
/// operation families delete.
#[tokio::test]
async fn finishes_the_no_tool_run_with_terminal_cleanup_and_an_immutable_result_record() {
    let counting = Arc::new(common::EntryReadCountingStorage::new(Arc::new(
        MemoryStorage::new(MemoryStorageOptions {
            now: Some(Arc::new(|| 100)),
        }),
    )));
    let storage: Arc<dyn Storage> = counting.clone();
    let fixture = common::create_drive_fixture(generation_spec(Some(storage))).await;
    let ready = advance_to_ready(&fixture).await;
    fixture
        .faux
        .set_responses([FauxResponseStep::Message(faux_assistant_message(
            "done",
            FauxAssistantMessageOptions {
                timestamp: Some(7),
                ..FauxAssistantMessageOptions::default()
            },
        ))]);
    generation_attempt(&fixture, ready, "the generation runs").await;
    let OperationState::Checkpoint(run) = common::current_state(&fixture) else {
        panic!("missing terminal checkpoint");
    };

    let settled = run_checkpoint(&fixture.lane, &fixture.drive, &run)
        .await
        .expect("the checkpoint settles");
    let ProcedureResult::Settled { outcome } = settled else {
        panic!("the run settles");
    };
    let tip_id = fixture.lane.state().tip_id.expect("the settled tip");
    assert_eq!(outcome.operation_id, common::OPERATION_ID);
    assert_eq!(outcome.kind, OperationKind::Run);
    assert_eq!(outcome.status, TerminalStatus::Completed);
    assert_eq!(
        outcome.tip_id.as_deref(),
        Some(tip_id.as_str()),
        "the outcome carries the settled tip"
    );
    assert!(
        fixture.lane.state().operation.is_none(),
        "the operation clears"
    );
    assert_eq!(
        fixture.lane.state().last_operation_id.as_deref(),
        Some(common::OPERATION_ID),
        "the last operation id sets"
    );
    counting.reset_entry_reads();
    let record = fixture
        .lane
        .get_result_impl(common::OPERATION_ID, &background_context())
        .await
        .expect("the result reads")
        .expect("the result record");
    assert_eq!(record.operation_id, common::OPERATION_ID);
    assert_eq!(record.kind, OperationKind::Run);
    assert_eq!(record.status, TerminalStatus::Completed);
    assert_eq!(
        record.tip_id.as_deref(),
        Some(tip_id.as_str()),
        "the record carries the settled tip"
    );
    let unknown = fixture
        .lane
        .get_result_impl("unknown", &background_context())
        .await
        .expect("the unknown result reads");
    assert!(unknown.is_none(), "the unknown operation has no record");
    assert_eq!(
        counting.entry_reads(),
        0,
        "the result record reads without scanning entries"
    );
    let meta = fixture
        .session
        .get_value(
            &operation_meta(common::OPERATION_ID).address,
            &background_context(),
        )
        .await
        .expect("the meta read");
    assert!(meta.is_none(), "the meta deletes");
    let state = fixture
        .session
        .get_value(
            &operation_state(common::OPERATION_ID).address,
            &background_context(),
        )
        .await
        .expect("the state read");
    assert!(state.is_none(), "the state deletes");
    assert_eq!(
        lock(&fixture.events)
            .iter()
            .filter(|event| event.payload.event_type() == HarnessEventType::RunEnd)
            .count(),
        1,
        "exactly one run_end publishes"
    );
    common::expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

/// The provided system prompt the per-turn-prompt cases build: the
/// provider renders the resolved tool context's string value into the
/// per-turn prompt, upstream's `(toolContext, context) => ...` literal.
fn provided_prompt() -> SystemPromptSource {
    SystemPromptSource::Provided(Arc::new(|tool_context: ToolContext, _context: &Context| {
        let mode = tool_context.map_or_else(
            || "none".to_owned(),
            |value| {
                format!(
                    "context:{}",
                    value
                        .downcast_ref::<String>()
                        .map_or("other", String::as_str)
                )
            },
        );
        Box::pin(async move { format!("per-turn:{mode}") })
    }))
}

/// The faux response factory recording each request's system prompt,
/// upstream's prompt-recording `streamSimple` overrides.
fn prompt_recording_factory(seen: Arc<Mutex<Vec<String>>>) -> FauxResponseStep {
    FauxResponseStep::Factory(Arc::new(
        move |context: &AiContext,
              _options: Option<&SimpleStreamOptions>,
              _state: &pi_ai::providers::faux::FauxProviderState,
              _model: &Model| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                lock(&seen).push(context.system_prompt.clone().unwrap_or_default());
                Ok(faux_assistant_message(
                    "answer",
                    FauxAssistantMessageOptions::default(),
                ))
            })
        },
    ))
}

/// The system-prompt resolution arms, upstream's `resolveSystemPrompt`:
/// the static prompt stands, the per-turn provider reads the resolved tool
/// context (the static tool context clones, the resolved one awaits), and
/// the resolved prompt rides the request context the provider sees.
#[tokio::test]
async fn resolves_the_static_and_provided_system_prompts_into_the_request() {
    let prompts_seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    // The static prompt: the request context carries it verbatim.
    let mut spec = generation_spec(None);
    spec.system_prompt = Some(SystemPromptSource::Static("be terse".to_owned()));
    let fixture = common::create_drive_fixture(spec).await;
    let ready = advance_to_ready(&fixture).await;
    fixture
        .faux
        .set_responses([prompt_recording_factory(Arc::clone(&prompts_seen))]);
    generation_attempt(&fixture, ready, "the attempt serves").await;
    assert_eq!(
        lock(&prompts_seen).as_slice(),
        ["be terse"],
        "the static prompt rides the request",
    );
    close_session(&fixture).await;

    // The per-turn provider: the resolved prompt rides without a tool
    // context.
    let mut spec = generation_spec(None);
    spec.system_prompt = Some(provided_prompt());
    let fixture = common::create_drive_fixture(spec).await;
    let ready = advance_to_ready(&fixture).await;
    fixture
        .faux
        .set_responses([prompt_recording_factory(Arc::clone(&prompts_seen))]);
    generation_attempt(&fixture, ready, "the attempt serves").await;
    assert_eq!(
        lock(&prompts_seen).as_slice(),
        ["be terse", "per-turn:none"],
        "the provided prompt resolves without a tool context",
    );
    close_session(&fixture).await;
}

/// Builds the drive fixture whose runtime `Config` carries the tools and
/// tool context the shared spec cannot express: the shared fixture's seed
/// runs first, then a second lane over the same session installs the edited
/// config — the boundary arms `prepareGeneration`'s config reads serve.
///
/// # Panics
/// The restore's failure.
async fn create_fixture_with_config(
    spec: common::DriveFixtureSpec,
    tools: Vec<AgentHarnessTool>,
    tool_context: Option<AgentHarnessToolContextSource>,
) -> common::DriveFixture {
    let fixture = common::create_drive_fixture(spec).await;
    let mut config = fixture.config.clone();
    config.tools = tools;
    config.tool_context = tool_context;
    let restored = restore_lane(fixture.session.clone(), "main", &background_context())
        .await
        .expect("restore");
    let lane = Arc::new(Lane::new(
        "main",
        fixture.session.clone(),
        fixture.models.clone(),
        fixture.hooks.clone(),
        restored,
        passthrough_fault_handler(),
        common::collecting_emit_batch(Arc::clone(&fixture.events)),
        common::unused_watch("generation"),
        Arc::new(move || config.clone()),
    ));
    lane.set_active_drive(Some(Arc::clone(&fixture.drive)));
    common::DriveFixture { lane, ..fixture }
}

/// The one configured tool the resolution boundary cases carry; the tool
/// never executes (the faux responses carry no tool calls).
fn generation_tool(name: &str) -> AgentHarnessTool {
    let execute: Arc<AgentHarnessToolExecuteFn> = Arc::new(
        |_tool_call_id: &str,
         _args: &JsonValue,
         _on_update,
         _tool_context: ToolContext,
         _invocation,
         _context: &Context| {
            Box::pin(async {
                Ok(AgentToolResult {
                    content: Vec::new(),
                    details: JsonValue::Null,
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        },
    );
    AgentHarnessTool {
        tool: Tool {
            name: name.to_owned(),
            description: name.to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"],
            }),
            constrained_sampling: None,
        },
        label: name.to_owned(),
        prepare_arguments: None,
        execute,
        replay: None,
        execution_mode: None,
    }
}

/// The boundary arms of the provided system prompt's tool context, upstream's
/// `resolveSystemPrompt` `toolContext` reads: the static source clones
/// through and the resolved source awaits its resolver, and either rides the
/// per-turn provider's prompt.
#[tokio::test]
async fn resolves_the_static_and_resolved_tool_contexts_into_the_provided_prompt() {
    let prompts_seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    // The static tool context: the provided prompt reads the cloned value.
    let mut spec = generation_spec(None);
    spec.system_prompt = Some(provided_prompt());
    let static_context: ToolContext = Some(Arc::new("static-value".to_owned()));
    let fixture = create_fixture_with_config(
        spec,
        Vec::new(),
        Some(AgentHarnessToolContextSource::Static(static_context)),
    )
    .await;
    let ready = advance_to_ready(&fixture).await;
    fixture
        .faux
        .set_responses([prompt_recording_factory(Arc::clone(&prompts_seen))]);
    generation_attempt(&fixture, ready, "the attempt serves").await;
    close_session(&fixture).await;

    // The resolved tool context: the resolver awaits per turn snapshot.
    let mut spec = generation_spec(None);
    spec.system_prompt = Some(provided_prompt());
    let resolver: ToolContextProvider = Arc::new(|_context: &Context| {
        Box::pin(async {
            let value: Arc<dyn std::any::Any + Send + Sync> = Arc::new("resolved-value".to_owned());
            Some(value)
        })
    });
    let fixture = create_fixture_with_config(
        spec,
        Vec::new(),
        Some(AgentHarnessToolContextSource::Resolved(resolver)),
    )
    .await;
    let ready = advance_to_ready(&fixture).await;
    fixture
        .faux
        .set_responses([prompt_recording_factory(Arc::clone(&prompts_seen))]);
    generation_attempt(&fixture, ready, "the attempt serves").await;
    assert_eq!(
        lock(&prompts_seen).as_slice(),
        [
            "per-turn:context:static-value".to_owned(),
            "per-turn:context:resolved-value".to_owned(),
        ],
        "the tool context sources resolve into the provided prompt",
    );
    close_session(&fixture).await;
}

/// The boundary arm of the tool surface's resolution: the ready leaf's
/// active tool names build the provider tool surface the request context
/// carries, upstream's `tools` mapping.
#[tokio::test]
async fn resolves_the_active_tools_into_the_provider_request() {
    let tools_seen: Arc<Mutex<Vec<Option<Vec<String>>>>> = Arc::new(Mutex::new(Vec::new()));
    let fixture = create_fixture_with_config(
        generation_spec(None),
        vec![generation_tool("my-tool")],
        None,
    )
    .await;
    fixture
        .lane
        .set_active_tools_impl(vec!["my-tool".to_owned()], &background_context())
        .await
        .expect("the active tools set");
    let ready = advance_to_ready(&fixture).await;
    assert_eq!(
        ready.generation_context.configuration.active_tool_names,
        ["my-tool".to_owned()],
        "the ready leaf captures the active tool",
    );
    let seen = Arc::clone(&tools_seen);
    fixture
        .faux
        .set_responses([FauxResponseStep::Factory(Arc::new(
            move |context: &AiContext, _options, _state, _model| {
                let seen = Arc::clone(&seen);
                Box::pin(async move {
                    lock(&seen).push(context.tools.as_ref().map(|tools| {
                        tools
                            .iter()
                            .map(|tool| tool.name.clone())
                            .collect::<Vec<_>>()
                    }));
                    Ok(faux_assistant_message(
                        "answer",
                        FauxAssistantMessageOptions::default(),
                    ))
                })
            },
        ))]);
    generation_attempt(&fixture, ready, "the attempt serves").await;
    assert_eq!(
        lock(&tools_seen).as_slice(),
        [Some(vec!["my-tool".to_owned()])],
        "the resolved tool surface rides the provider request",
    );
    close_session(&fixture).await;
}

/// The curated options pair the patch case records, upstream's observed
/// `streamOptions` timeout and retry-attempt fields.
type OptionsSeen = Arc<Mutex<Vec<(Option<u64>, Option<u32>)>>>;

/// The boundary arm of the `before_request` patch, upstream's
/// `applyStreamOptionsPatch` application: the hook's stream-options patch
/// replaces the snapshotted options the provider request receives.
#[tokio::test]
async fn applies_the_before_request_patch_to_the_provider_options() {
    let options_seen: OptionsSeen = Arc::new(Mutex::new(Vec::new()));
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    fixture
        .hooks
        .on(
            HookName::BeforeRequest,
            Arc::new(|_invocation, _context| {
                Box::pin(async {
                    Ok(HookResult::BeforeRequest(Some(BeforeRequestResult {
                        stream_options: AgentHarnessStreamOptionsPatch {
                            timeout_ms: Some(Some(4_321)),
                            max_retries: Some(Some(2)),
                            ..AgentHarnessStreamOptionsPatch::default()
                        },
                    })))
                })
            }),
            HookOptions::default(),
        )
        .expect("register before_request");
    let seen = Arc::clone(&options_seen);
    fixture
        .faux
        .set_responses([FauxResponseStep::Factory(Arc::new(
            move |_context: &AiContext,
                  options: Option<&SimpleStreamOptions>,
                  _state: &pi_ai::providers::faux::FauxProviderState,
                  _model: &Model| {
                let seen = Arc::clone(&seen);
                Box::pin(async move {
                    lock(&seen).push(options.map_or((None, None), |options| {
                        (options.timeout_ms, options.max_retries)
                    }));
                    Ok(faux_assistant_message(
                        "answer",
                        FauxAssistantMessageOptions::default(),
                    ))
                })
            },
        ))]);
    generation_attempt(&fixture, ready, "the attempt serves").await;
    assert_eq!(
        lock(&options_seen).as_slice(),
        [(Some(4_321), Some(2))],
        "the patch rides the provider options",
    );
    close_session(&fixture).await;
}

/// The boundary arm of the context-transform hook's gate refusal, upstream's
/// `runWithGate` throwing `AbortRequested` into `transformContext`: the
/// cancellation landing between the request's own admission and the
/// transform's refuses the transform's admission, and the faulted pass
/// keeps the committed intent.
#[tokio::test]
async fn faults_the_pass_when_the_gate_refuses_the_transform_admission() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    let (hook_started_rx, release_hook) = park_hook(
        &fixture,
        HookName::BeforeRequest,
        HookResult::BeforeRequest(None),
    );
    fixture.storage.clear_commit_attempts();
    let error = parked_generation(&fixture, ready, hook_started_rx, release_hook, |fixture| {
        common::begin_closed_abort(&fixture.drive);
    })
    .await
    .expect_err("the refusal faults the pass");
    assert_eq!(
        error.to_string(),
        "Abort requested",
        "the gate refusal faults the pass",
    );
    assert!(
        fixture
            .storage
            .get_commit_attempts()
            .iter()
            .any(|writes| writes_operation_state(writes, "effect_pending")),
        "the intent still commits before the stream",
    );
    assert_eq!(
        fixture.faux.state().call_count(),
        0,
        "the provider never runs",
    );
    assert!(
        matches!(
            common::current_state(&fixture),
            OperationState::AssistantEffectPending(_)
        ),
        "the faulted leaf stays effect_pending",
    );
    common::expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

/// The boundary arm of the context-transform hook's closed registry,
/// upstream's `runWithGate` throwing the closed error: a registry closed
/// between the request hook and the transform faults the pass with the
/// closed error.
#[tokio::test]
async fn faults_the_pass_when_the_hook_registry_closed_mid_generation() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    let (hook_started_rx, release_hook) = park_hook(
        &fixture,
        HookName::BeforeRequest,
        HookResult::BeforeRequest(None),
    );
    let error = parked_generation(&fixture, ready, hook_started_rx, release_hook, |fixture| {
        fixture
            .hooks
            .close("generation hooks closed mid-generation".to_owned());
    })
    .await
    .expect_err("the closed registry faults the pass");
    assert_eq!(
        error.to_string(),
        "generation hooks closed mid-generation",
        "the closed registry faults the pass",
    );
    assert_eq!(
        fixture.faux.state().call_count(),
        0,
        "the provider never runs",
    );
    close_session(&fixture).await;
}

/// The ported request hook's refusal arm, upstream's `gate.admit` throw
/// crossing the stream-less request seam: the cancellation landing between
/// the transform hook's admission and the request's records the refusal,
/// ends the stream aborted, and the generation raises it after the
/// lifecycle closes — the recovery reconciliation reads the untouched
/// effect-pending leaf.
#[tokio::test]
async fn records_the_request_refusal_and_raises_it_after_the_stream_settles() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    let (hook_started_rx, release_hook) = park_hook(
        &fixture,
        HookName::TransformContext,
        HookResult::TransformContext(None),
    );
    fixture.storage.clear_commit_attempts();
    let error = parked_generation(&fixture, ready, hook_started_rx, release_hook, |fixture| {
        common::begin_closed_abort(&fixture.drive);
    })
    .await
    .expect_err("the refusal raises");
    assert_eq!(
        error.to_string(),
        "Abort requested",
        "the recorded refusal raises",
    );
    assert_eq!(
        fixture.faux.state().call_count(),
        0,
        "the refused admission never builds the stream",
    );
    let OperationState::AssistantEffectPending(pending) = common::current_state(&fixture) else {
        panic!("the refused generation keeps its intent");
    };
    let entry = fixture
        .session
        .get_entry(&pending.response_entry_id, &background_context())
        .await
        .expect("the entry read");
    assert!(entry.is_none(), "no response entry commits");
    common::expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

/// The boundary arm of the payload hook's pass-through, upstream's
/// `beforePayload` invocation: the provider's `on_payload` seam returns the
/// hook's replacement payload, or the unchanged payload when no handler
/// replaced it.
#[tokio::test]
async fn passes_the_provider_payload_through_the_before_payload_hook() {
    // No before_payload handler: the payload passes through unchanged.
    let returned: Arc<Mutex<Vec<Option<JsonValue>>>> = Arc::new(Mutex::new(Vec::new()));
    let (fixture, ready) = payload_script_fixture(Arc::clone(&returned)).await;
    generation_attempt(&fixture, ready, "the attempt serves").await;
    assert_eq!(
        lock(&returned).as_slice(),
        [Some(json!({ "prompt": "generation payload" }))],
        "the payload passes through unchanged",
    );
    close_session(&fixture).await;

    // The before_payload handler: its replacement rides to the provider.
    let returned: Arc<Mutex<Vec<Option<JsonValue>>>> = Arc::new(Mutex::new(Vec::new()));
    let (fixture, ready) = payload_script_fixture(Arc::clone(&returned)).await;
    fixture
        .hooks
        .on(
            HookName::BeforePayload,
            Arc::new(|_invocation, _context| {
                Box::pin(async {
                    Ok(HookResult::BeforePayload(Some(PayloadResult {
                        payload: json!({ "prompt": "hooked payload" }),
                    })))
                })
            }),
            HookOptions::default(),
        )
        .expect("register before_payload");
    generation_attempt(&fixture, ready, "the attempt serves").await;
    assert_eq!(
        lock(&returned).as_slice(),
        [Some(json!({ "prompt": "hooked payload" }))],
        "the hook's replacement rides",
    );
    close_session(&fixture).await;
}

/// The boundary arm of the payload hook's refused admission, upstream's
/// `onPayload` throw: the cancellation landing while the provider holds the
/// payload refuses the hook's admission, the seam drops the payload, and
/// the handler is never consulted — the stream still settles.
#[tokio::test]
async fn drops_the_payload_when_the_gate_refuses_the_before_payload_admission() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    let (reached_tx, reached_rx) = deferred();
    let (release_tx, release_rx) = deferred();
    let returned = Arc::new(Mutex::new(Vec::new()));
    let hook_calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&hook_calls);
    fixture
        .hooks
        .on(
            HookName::BeforePayload,
            Arc::new(move |_invocation, _context| {
                let counted = Arc::clone(&counted);
                Box::pin(async move {
                    counted.fetch_add(1, Ordering::SeqCst);
                    Ok(HookResult::BeforePayload(Some(PayloadResult {
                        payload: json!({ "prompt": "hooked payload" }),
                    })))
                })
            }),
            HookOptions::default(),
        )
        .expect("register before_payload");
    fixture.models.set_provider(scripted_provider(
        &fixture,
        on_payload_script(
            Arc::clone(&returned),
            Arc::new(Mutex::new(Some(reached_tx))),
            Arc::new(Mutex::new(Some(release_rx))),
        ),
    ));

    let generating = spawn_generation_run(&fixture, ready);
    reached_rx
        .await
        .expect("the provider reached the payload seam");
    let (sender, cancellation) = tokio::sync::watch::channel(());
    drop(sender);
    fixture.drive.begin_abort(cancellation);
    fixture.drive.signal_abort();
    let _ = release_tx.send(());

    let settled = generating
        .await
        .expect("the generation runs")
        .expect("the attempt serves");
    assert!(
        matches!(settled, ProcedureResult::Continue),
        "the dropped payload still settles",
    );
    assert_eq!(
        lock(&returned).as_slice(),
        [None],
        "the refused seam drops the payload"
    );
    assert_eq!(
        hook_calls.load(Ordering::SeqCst),
        0,
        "the refusal precedes the handler",
    );
    let OperationState::Checkpoint(_) = common::current_state(&fixture) else {
        panic!("the settled response checkpoints");
    };
    common::expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

/// The boundary arm of the stream consumption's protocol violation,
/// upstream's `consumeAssistantStream` throw: a provider stream emitting
/// `done` before `start` faults the pass after the lifecycle closes.
#[tokio::test]
async fn faults_the_pass_when_the_provider_stream_breaks_the_protocol() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    let script: SimpleStreamScript = Arc::new(
        move |_model: &Model, _options: Option<&SimpleStreamOptions>| {
            let stream = assistant_message_event_stream();
            let pusher = stream.clone();
            tokio::spawn(async move {
                pusher.push(AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message: faux_assistant_message(
                        "never",
                        FauxAssistantMessageOptions {
                            timestamp: Some(5),
                            ..FauxAssistantMessageOptions::default()
                        },
                    ),
                });
            });
            stream
        },
    );
    fixture
        .models
        .set_provider(scripted_provider(&fixture, script));

    let error = run_generation(&fixture.lane, &fixture.drive, GenerationLeaf::Ready(ready))
        .await
        .expect_err("the broken stream faults the pass");
    assert_eq!(
        error.to_string(),
        "Assistant message stream emitted done before start",
        "the protocol violation faults the pass",
    );
    assert!(
        matches!(
            common::current_state(&fixture),
            OperationState::AssistantEffectPending(_)
        ),
        "the faulted leaf stays effect_pending",
    );
    close_session(&fixture).await;
}

/// The boundary arm of preparation's cancel downgrade, upstream's
/// `cancel_requested` preparation: a cancellation committed before the call
/// wins the bounded-context read, and the generation continues without an
/// intent, a provider call, or a commit.
#[tokio::test]
async fn declines_preparation_when_cancellation_won_the_bounded_context_read() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    common::cancel_operation(&fixture).await;
    fixture.storage.clear_commit_attempts();

    let declined = generation_attempt(&fixture, ready, "the generation runs").await;
    assert!(
        matches!(declined, ProcedureResult::Continue),
        "the cancellation declines to a continue",
    );
    assert_eq!(
        fixture.faux.state().call_count(),
        0,
        "the provider never runs",
    );
    assert!(
        !fixture
            .storage
            .get_commit_attempts()
            .iter()
            .any(|writes| writes_operation_state(writes, "effect_pending")),
        "no effect_pending intent commits",
    );
    close_session(&fixture).await;
}

/// The boundary arm of the retry wait's admission, upstream's
/// `gate.admit` throw around `waitUntil`: an aborted gate refuses the local
/// wait's admission and the pass faults with the refusal.
#[tokio::test]
async fn faults_the_retry_wait_when_the_gate_refuses_the_wait_admission() {
    let _clock = common::pin_clock(1_000);
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    let not_before = retry_not_before(&ready.generation_context.retry_policy, 1, 1_000);
    let retry_wait = AssistantRetryWaitOperation {
        scope: ready.scope.clone(),
        generation_context: ready.generation_context.clone(),
        retry_wait: RetryWait {
            next_attempt: 2,
            not_before,
            error_message: "retry".to_owned(),
        },
    };
    patch_live_state(
        &fixture.lane,
        OperationState::AssistantRetryWait(retry_wait.clone()),
    )
    .await;
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: common::OPERATION_ID.to_owned(),
            wait_for_retry: Some(true),
            poll_deferred: None,
        },
        &background_context(),
    ));
    let (sender, cancellation) = tokio::sync::watch::channel(());
    drop(sender);
    drive.begin_abort(cancellation);
    drive.signal_abort();

    let error = run_generation(&fixture.lane, &drive, GenerationLeaf::RetryWait(retry_wait))
        .await
        .expect_err("the refused wait faults the pass");
    assert_eq!(
        error.to_string(),
        "Abort requested",
        "the refusal faults the pass",
    );
    close_session(&fixture).await;
}

/// The boundary arm of the retry wait's commit, upstream's
/// `continueOperation` cancel-refused contract: a cancellation committed
/// before the call downgrades the due wait to a plain continue, and the
/// cancelled leaf stays for the reconciliation.
#[tokio::test]
async fn downgrades_a_cancelled_retry_wait_to_a_plain_continue() {
    let fixture = common::create_drive_fixture(generation_spec(None)).await;
    let ready = advance_to_ready(&fixture).await;
    let retry_wait = AssistantRetryWaitOperation {
        scope: ready.scope.clone(),
        generation_context: ready.generation_context.clone(),
        retry_wait: RetryWait {
            next_attempt: 2,
            not_before: retry_not_before(&ready.generation_context.retry_policy, 1, 0),
            error_message: "retry".to_owned(),
        },
    };
    patch_live_state(
        &fixture.lane,
        OperationState::AssistantRetryWait(retry_wait.clone()),
    )
    .await;
    common::cancel_operation(&fixture).await;
    fixture.storage.clear_commit_attempts();

    let declined = run_generation(
        &fixture.lane,
        &fixture.drive,
        GenerationLeaf::RetryWait(retry_wait),
    )
    .await
    .expect("the wait settles");
    assert!(
        matches!(declined, ProcedureResult::Continue),
        "the cancellation continues",
    );
    let OperationState::AssistantRetryWait(wait) = common::current_state(&fixture) else {
        panic!("the cancelled wait stays");
    };
    assert!(
        matches!(wait.scope.control, Control::CancelRequested { .. }),
        "the leaf keeps the cancellation",
    );
    assert!(
        fixture.storage.get_commit_attempts().is_empty(),
        "the downgrade commits nothing",
    );
    close_session(&fixture).await;
}

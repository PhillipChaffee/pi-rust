//! The drive spine and its procedure modules' boundary cases, the
//! uncovered-arm sweep the coverage gate drove: the spine's no-progress
//! invariant and its error/abort-catch routes, the before-drive gate
//! abort, the effect-pending and retry-wait dispatch arms, the
//! current-operation invariant, the boundary planners' payload invariants
//! and follow-up merge, the finish mediation's cancel routes and invariants,
//! the tool placement's source and staged-result invariants, the
//! checkpoint planner's intent/hook/trigger arms, and the recovery's
//! cancel route.
//!
//! Upstream at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` pins the
//! happy choreography per procedure; these cases bind the arms its suites
//! leave uncovered, through the same public procedures, with the
//! assertion on the observable error or outcome only.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxContentBlock;
use pi_ai::providers::faux::{RegisterFauxProviderOptions, faux_assistant_message, faux_tool_call};
use pi_ai::types::{Message, StopReason, TextContent, ToolResultBlock, ToolResultMessage};
use pi_ai::utils::assistant_message_frame::AssistantMessageFrame;
use pi_ai::utils::retry::RetryPolicy;
use serde_json::json;

use super::common;
use crate::harness::agent_harness::BeforeRunResult;
use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::DriveWaitReason;
use crate::harness::agent_harness::{HookInvocation, HookName, HookOptions, HookResult};
use crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS;
use crate::harness::context::background_context;
use crate::harness::runtime::drive::boundary::BoundaryPlacement;
use crate::harness::runtime::drive::boundary::finish_run_boundary;
use crate::harness::runtime::drive::boundary::{normalized_retry_policy, plan_boundary_inbox};
use crate::harness::runtime::drive::checkpoint::{run_checkpoint, start_run};
use crate::harness::runtime::drive::drive_operation;
use crate::harness::runtime::drive::recovery::recover_assistant_generation;
use crate::harness::runtime::drive::tool_placement::ToolBatchSource;
use crate::harness::runtime::drive::tool_placement::materialize_ready;
use crate::harness::runtime::drive::tool_placement::{read_tool_batch_source, tool_call_for};
use crate::harness::runtime::lane::{Lane, operation_state_with_scope};
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::deferred_effect_pending;
use crate::harness::runtime::test_support::generation_context;
use crate::harness::runtime::test_support::{patch_live_state, raw_write, starting_run_state};
use crate::harness::runtime::types::{Drive, LaneError, LaneState, ProcedureResult};
use crate::harness::session::session::StorageBackedSession;
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::CheckpointData;
use crate::harness::session::types::CheckpointOperation;
use crate::harness::session::types::CompactionEntryBody;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::Control;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::StartingOperation;
use crate::harness::session::types::SummaryEffectPendingOperation;
use crate::harness::session::types::SummaryGenerationScope;
use crate::harness::session::types::{ToolBatch, ToolCall, ToolsOperation, operation_scope_of};
use crate::harness::session::values as stored_values;
use crate::harness::session::values::{Write, append_list_write, pending_entry, set_value_write};
use crate::harness::types::AgentHarnessStreamOptions;
use crate::types::{AgentMessage, QueueMode};

/// The drive-pass id the boundary fixtures' mismatch variant uses.
const OTHER_OPERATION_ID: &str = "01950000-0000-7000-8000-0000000000ff";

/// The fixed response entry id the recovery fixtures reserve.
const RESPONSE_ENTRY_ID: &str = "01950000-0000-7000-8000-0000000000aa";

/// The boundary suite's fixture spec, upstream's per-file `createFixture`
/// spec shape: the fixed clock memory backend (or the caller's), no
/// prompt admission, and a disabled retry budget.
fn boundary_spec(
    backend: Option<Arc<dyn crate::harness::session::types::Storage>>,
) -> common::DriveFixtureSpec {
    common::DriveFixtureSpec {
        suite: "drive-boundary",
        watch_suite: "boundary",
        faux: RegisterFauxProviderOptions::default(),
        retry_policy: RetryPolicy {
            enabled: false,
            max_retries: 0,
            base_delay_ms: 0,
            max_agent_delay_ms: None,
        },
        stream_options: AgentHarnessStreamOptions::default(),
        backend,
        admit_prompt: false,
        system_prompt: None,
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

/// The batch-call literal, upstream's inline `{ status, sourceIndex,
/// resultEntryId, ... }` objects; the phase field is private to the
/// session types module, so the literal restates through the wire shape.
fn wire_call(wire: serde_json::Value) -> ToolCall {
    serde_json::from_value(wire).expect("the tool call round-trips")
}

/// The planned phase, upstream's `{ status: "planned", ... }`.
fn wire_planned_call(source_index: u64, result_entry_id: &str) -> ToolCall {
    wire_call(json!({
        "sourceIndex": source_index,
        "resultEntryId": result_entry_id,
        "status": "planned",
    }))
}

/// The outcome-ready phase, upstream's
/// `{ status: "outcome_ready", terminate, ... }`.
fn wire_outcome_ready_call(source_index: u64, result_entry_id: &str, terminate: bool) -> ToolCall {
    wire_call(json!({
        "sourceIndex": source_index,
        "resultEntryId": result_entry_id,
        "status": "outcome_ready",
        "terminate": terminate,
    }))
}

/// The completed phase, upstream's
/// `{ status: "completed", terminate, ... }`.
fn wire_completed_call(source_index: u64, result_entry_id: &str, terminate: bool) -> ToolCall {
    wire_call(json!({
        "sourceIndex": source_index,
        "resultEntryId": result_entry_id,
        "status": "completed",
        "terminate": terminate,
    }))
}

/// The assistant message one batch runs from: one tool-call block per
/// call, `call-{index}` ids, upstream's fixture shape.
fn batch_assistant(calls: usize) -> pi_ai::types::AssistantMessage {
    let blocks: Vec<FauxContentBlock> = (0..calls)
        .map(|index| {
            let mut arguments = serde_json::Map::new();
            arguments.insert("value".to_owned(), json!("called"));
            faux_tool_call("read", arguments, Some(format!("call-{index}")))
        })
        .collect();
    faux_assistant_message(
        blocks,
        FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::ToolUse),
            timestamp: Some(20),
            ..FauxAssistantMessageOptions::default()
        },
    )
}

/// The staged tool result one pending entry carries, upstream's
/// `{ toolCallId, toolName, content: [text], isError: false }` literals.
fn staged_tool_result(call_id: &str, tool_name: &str, text: &str) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: call_id.to_owned(),
        tool_name: tool_name.to_owned(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 1,
    }
}

/// Writes one staged pending payload, upstream's per-case
/// `pendingEntry` value commits.
async fn stage_pending(session: &Arc<StorageBackedSession>, entry_id: &str, pending: PendingEntry) {
    commit_writes(
        session,
        vec![set_value_write(&pending_entry(entry_id), pending).expect("the pending write")],
    )
    .await;
}

/// Installs the fixture's tools leaf over the batch's assistant entry, the
/// hand-seeded `installOperation` call the tool-boundary cases share, and
/// returns the installed leaf; the tip derives from the assistant entry.
async fn install_tools_leaf(
    fixture: &common::DriveFixture,
    assistant: &pi_ai::types::AssistantMessage,
    calls: Vec<ToolCall>,
) -> ToolsOperation {
    let assistant_entry_id = "01950000-0000-7000-8000-0000000000aa";
    let capability = ToolsOperation {
        scope: common::run_scope(DEFAULT_COMPACTION_SETTINGS),
        batch: ToolBatch {
            assistant_entry_id: assistant_entry_id.to_owned(),
            configuration: fixture.configuration.clone(),
            turn_id: "turn-1".to_owned(),
            calls,
        },
    };
    let options = common::InstallOptions {
        entries: vec![common::assistant_entry(
            assistant_entry_id,
            None,
            assistant.clone(),
        )],
        ..common::InstallOptions::default()
    };
    common::install_operation(
        fixture,
        OperationState::Tools(capability.clone()),
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        options,
    )
    .await;
    capability
}

/// Flips the live operation's control to `cancel_requested` through the
/// lane's own command line, the hook-side equivalent of
/// [`common::cancel_operation`]'s patch for the mid-procedure cases.
///
/// # Panics
/// The absence of a live operation or the commit's failure.
async fn request_cancel(lane: &Arc<Lane>) {
    let live = lane.state().operation.expect("the live operation");
    patch_live_state(
        lane,
        operation_state_with_scope(
            &live.state,
            OperationScope {
                control: Control::CancelRequested { requested_at: 2 },
                ..operation_scope_of(&live.state)
            },
        ),
    )
    .await;
}

/// One inbox item, the plan cases' per-entry `InboxItem` literal.
fn inbox_item(entry_id: &str, kind: InboxItemKind) -> InboxItem {
    InboxItem {
        entry_id: entry_id.to_owned(),
        kind,
    }
}

/// The tipful idle planning state over an inbox, the plan cases' shared
/// `LaneState` literal.
fn plan_state(configuration: &LaneConfiguration, inbox: Vec<InboxItem>) -> LaneState {
    LaneState {
        tip_id: Some("tip".to_owned()),
        configuration: configuration.clone(),
        inbox,
        last_operation_id: None,
        operation: None,
    }
}

/// One plan read over the idle planning state, the plan cases' shared
/// `plan_boundary_inbox` call.
async fn plan_over(
    fixture: &common::DriveFixture,
    state: &LaneState,
    scope: OperationScope,
    follow_up_when_no_trigger: bool,
) -> Result<BoundaryPlacement, LaneError> {
    plan_boundary_inbox(
        &fixture.lane,
        &fixture.drive,
        state,
        scope,
        fixture.session.as_ref(),
        state.tip_id.clone(),
        follow_up_when_no_trigger,
    )
    .await
}

/// One installed tool batch: the fixture over the hand-seeded leaf, the
/// placement cases' shared choreography prefix; the tip derives from the
/// assistant entry.
async fn installed_batch(calls: Vec<ToolCall>) -> (common::DriveFixture, ToolsOperation) {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let assistant = batch_assistant(calls.len());
    let capability = install_tools_leaf(&fixture, &assistant, calls).await;
    (fixture, capability)
}

/// The installed batch's source read, the placement cases' shared
/// `readToolBatchSource` call.
async fn batch_sources(
    fixture: &common::DriveFixture,
    capability: &ToolsOperation,
) -> ToolBatchSource {
    read_tool_batch_source(&fixture.lane, &fixture.drive, &capability.batch)
        .await
        .expect("the source reads")
}

/// The placement read's rejection, the staged-placement cases' shared
/// `materializeReady` call.
async fn placement_rejection(
    fixture: &common::DriveFixture,
    capability: &ToolsOperation,
    sources: &ToolBatchSource,
) -> LaneError {
    materialize_ready(&fixture.lane, &fixture.drive, capability, sources, false)
        .await
        .expect_err("the placement read rejects")
}

/// The effect-pending leaf the recovery-boundary cases carry: attempt 1
/// over the reserved response entry, the seeded generation inputs.
/// Drives the operation and pins the retry-waiting outcome, the recovery
/// suites' wait; a settled pass panics the test by design.
async fn retry_waiting_pass(fixture: &common::DriveFixture, why: &str, settles_why: &str) {
    let outcome = drive_operation(&fixture.lane, &fixture.drive)
        .await
        .expect(why);
    let DriveOutcome::Waiting {
        reason: DriveWaitReason::Retry { .. },
        ..
    } = outcome
    else {
        unreachable!("{settles_why}: {outcome:?}")
    };
}

fn assistant_effect_pending_leaf(scope: OperationScope) -> AssistantEffectPendingOperation {
    AssistantEffectPendingOperation {
        scope,
        generation_context: generation_context(),
        attempt: 1,
        response_entry_id: RESPONSE_ENTRY_ID.to_owned(),
        usage_id: "usage-1".to_owned(),
        intended_output_limit: 100,
        context_window: 1_000,
    }
}

/// The tip-triggered checkpoint leaf the checkpoint-boundary cases carry.
fn checkpoint_leaf(scope: OperationScope, continuation: Continuation) -> CheckpointOperation {
    CheckpointOperation {
        scope,
        checkpoint: CheckpointData {
            continuation,
            trigger_entry_id: "tip".to_owned(),
        },
    }
}

/// The tip-history install options, the leaf installs' shared options.
fn tip_history_options() -> common::InstallOptions {
    common::InstallOptions {
        entries: vec![common::user_entry("tip", None, "history")],
        ..common::InstallOptions::default()
    }
}

/// Installs one checkpoint leaf over the `tip` history, the
/// checkpoint-boundary cases' shared `installOperation` call.
async fn install_checkpoint_leaf(fixture: &common::DriveFixture, leaf: CheckpointOperation) {
    common::install_operation(
        fixture,
        OperationState::Checkpoint(leaf),
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        tip_history_options(),
    )
    .await;
}

/// Installs a starting run leaf over the `tip` history with the `tip`
/// prompt entry, the start-boundary cases' shared `installOperation` call.
async fn install_starting_run_with_tip(fixture: &common::DriveFixture) {
    common::install_operation(
        fixture,
        starting_run_state(),
        OperationIntent::Run {
            prompt_entry_ids: vec!["tip".to_owned()],
        },
        tip_history_options(),
    )
    .await;
}

/// The installed starting leaf, the start-boundary cases' unpack.
fn starting_leaf(fixture: &common::DriveFixture) -> StartingOperation {
    match common::current_state(fixture) {
        OperationState::Starting(run) => run,
        _ => unreachable!("the starting leaf installs"),
    }
}

/// One drive pass's rejection, the spine cases' shared `drive_operation`
/// call.
async fn drive_pass_error(lane: &Arc<Lane>, drive: &Arc<Drive>) -> LaneError {
    drive_operation(lane, drive)
        .await
        .expect_err("the pass faults")
}

/// One finish read over the installed checkpoint leaf, the finish-boundary
/// cases' shared `finishRunBoundary` call.
async fn finish_over(
    fixture: &common::DriveFixture,
    continuation: Continuation,
) -> Result<ProcedureResult, LaneError> {
    finish_run_boundary(
        &fixture.lane,
        &fixture.drive,
        &common::current_state(fixture),
        continuation,
        &[],
        Vec::new(),
    )
    .await
}

/// Upstream pin: none (upstream's guard is defensive; its suites never
/// re-drive a fully completed batch). A tools leaf whose every call is
/// completed re-runs without materialization and returns `continue` with
/// an unchanged projection: the no-progress invariant faults the pass.
#[tokio::test]
async fn the_drive_fails_the_no_progress_invariant_when_a_completed_batch_stays_put() {
    let (fixture, capability) = installed_batch(vec![wire_completed_call(0, "r0", false)]).await;

    let error = drive_pass_error(&fixture.lane, &fixture.drive).await;
    assert!(
        error.to_string().contains("made no progress from tools"),
        "the no-progress invariant names the leaf: {error}"
    );
    assert_eq!(
        fixture
            .lane
            .state()
            .operation
            .expect("the leaf stays live")
            .state,
        OperationState::Tools(capability),
        "the batch stays put"
    );
    close_session(&fixture).await;
}

/// Upstream pin: `currentOperation`'s invariant. An idle lane and a
/// mismatched pass id both reject with the no-matching-operation
/// invariant before any dispatch.
#[tokio::test]
async fn the_drive_rejects_an_unmatched_or_absent_operation() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let context = background_context();
    let _ = context;

    let error = drive_pass_error(&fixture.lane, &fixture.drive).await;
    assert!(
        error.to_string().contains(&format!(
            "Drive {} has no matching current operation",
            fixture.operation_id
        )),
        "the idle invariant names the pass: {error}"
    );

    common::install_operation(
        &fixture,
        starting_run_state(),
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        common::InstallOptions::default(),
    )
    .await;
    let mismatched = crate::harness::runtime::test_support::drive_pass(OTHER_OPERATION_ID);
    let error = drive_pass_error(&fixture.lane, &mismatched).await;
    assert!(
        error
            .to_string()
            .contains("has no matching current operation"),
        "the mismatch invariant names the pass: {error}"
    );
    close_session(&fixture).await;
}

/// Upstream pin: the catch's non-abort rethrow. A procedure's plain error
/// propagates to the pass's failure instead of the abort-continue route.
#[tokio::test]
async fn the_drive_propagates_a_non_abort_procedure_error() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    common::install_operation(
        &fixture,
        starting_run_state(),
        OperationIntent::Run {
            prompt_entry_ids: vec!["ghost".to_owned()],
        },
        common::InstallOptions::default(),
    )
    .await;

    let error = drive_pass_error(&fixture.lane, &fixture.drive).await;
    assert!(
        error
            .to_string()
            .contains("Run prompt entry ghost is missing its message"),
        "the start invariant propagates: {error}"
    );
    close_session(&fixture).await;
}

/// Upstream pin: the before-drive catch's abort fall-through. A gate
/// abort at `before_drive` awaits the abort's cancellation, falls through
/// to the loop, and the re-aborted `before_run` inside `start_run` rides
/// the abort-continue path into the no-progress invariant.
#[tokio::test]
async fn the_drive_awaits_a_before_drive_gate_abort_and_falls_through() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    install_starting_run_with_tip(&fixture).await;
    let (sender, receiver) = tokio::sync::watch::channel(());
    drop(sender);
    fixture.drive.begin_abort(receiver);

    let error = drive_pass_error(&fixture.lane, &fixture.drive).await;
    assert!(
        error.to_string().contains("made no progress from starting"),
        "the fall-through lands on the no-progress invariant: {error}"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (upstream's dispatch arm is exercised through the
/// same procedure; the port binds the spine arm). An
/// `assistant.effect_pending` leaf with no frames recovers into the retry
/// wait and reports the durable wait.
#[tokio::test]
async fn the_drive_routes_an_assistant_effect_pending_leaf_to_recovery() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    // A long base delay keeps the recovered retry wait not-yet-due, so the
    // drive's re-dispatch reports the durable wait instead of running the
    // next attempt.
    let mut leaf = assistant_effect_pending_leaf(common::run_scope(DEFAULT_COMPACTION_SETTINGS));
    leaf.generation_context.retry_policy.base_delay_ms = 60_000;
    common::install_operation(
        &fixture,
        OperationState::AssistantEffectPending(leaf.clone()),
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        common::InstallOptions::default(),
    )
    .await;

    retry_waiting_pass(
        &fixture,
        "the recovery drives",
        "the recovery waits for the retry",
    )
    .await;
    assert!(
        matches!(
            common::current_state(&fixture),
            OperationState::AssistantRetryWait(_)
        ),
        "the recovery lands at the retry wait"
    );
    assert_eq!(
        fixture.faux.state().call_count(),
        0,
        "the recovery never calls the provider"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (same spine-arm binding as the assistant case). A
/// `summary.effect_pending` leaf recovers into the structural retry wait
/// and the next iteration reports the durable wait.
#[tokio::test]
async fn the_drive_routes_a_summary_effect_pending_leaf_and_its_retry_wait() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let leaf = SummaryEffectPendingOperation {
        scope: common::run_scope(DEFAULT_COMPACTION_SETTINGS),
        generation: SummaryGenerationScope {
            task: common::standalone_compaction_task(None),
            summary_context: common::summary_context(&fixture.configuration),
        },
        attempt: 1,
        request: None,
        usage_ids: vec!["usage-1".to_owned()],
    };
    common::install_operation(
        &fixture,
        OperationState::SummaryEffectPending(leaf),
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        common::InstallOptions::default(),
    )
    .await;

    retry_waiting_pass(
        &fixture,
        "the recovery drives",
        "the recovery waits for the retry",
    )
    .await;
    assert!(
        matches!(
            common::current_state(&fixture),
            OperationState::SummaryRetryWait(_)
        ),
        "the recovery lands at the structural retry wait"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (the deferred suites drive the procedure directly;
/// the spine arm binds through it). A `deferred.effect_pending` leaf whose
/// source entry is missing faults the pass through the source-handle
/// invariant.
#[tokio::test]
async fn the_drive_routes_a_deferred_effect_pending_leaf_to_the_poll() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    common::install_operation(
        &fixture,
        deferred_effect_pending("ghost-source", RESPONSE_ENTRY_ID, 1),
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        common::InstallOptions::default(),
    )
    .await;

    let error = drive_operation(&fixture.lane, &fixture.drive)
        .await
        .expect_err("the missing source faults the pass");
    assert!(
        error
            .to_string()
            .contains("Deferred source ghost-source is missing its assistant handle"),
        "the source-handle invariant propagates: {error}"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (upstream's recovery awaits the same cancelled
/// continue; its suites drive the happy recovery). A cancelled
/// `assistant.effect_pending` leaf reports the cancel route without any
/// settlement.
#[tokio::test]
async fn the_assistant_recovery_reports_the_cancel_requested_route() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let leaf = assistant_effect_pending_leaf(common::scope_with_control(
        common::cancelled_control(),
        DEFAULT_COMPACTION_SETTINGS,
    ));
    common::install_operation(
        &fixture,
        OperationState::AssistantEffectPending(leaf.clone()),
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        common::InstallOptions::default(),
    )
    .await;

    let result = recover_assistant_generation(&fixture.lane, &fixture.drive, &leaf)
        .await
        .expect("the cancelled recovery continues");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the cancelled recovery continues the pass"
    );
    let entry = fixture
        .session
        .get_entry(RESPONSE_ENTRY_ID, &background_context())
        .await
        .expect("the entry read");
    assert!(entry.is_none(), "the cancelled recovery settles nothing");
    close_session(&fixture).await;
}

/// Upstream pin: `readToolBatchSource`'s invariants. A missing, non-leaf,
/// or non-assistant assistant entry and a source index that misses its
/// tool-call block each reject with the source invariants.
#[tokio::test]
async fn the_batch_source_read_rejects_invalid_assistant_entries_and_indices() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let configuration = fixture.configuration.clone();
    let missing = ToolBatch {
        assistant_entry_id: "ghost".to_owned(),
        configuration,
        turn_id: "turn-1".to_owned(),
        calls: Vec::new(),
    };
    let error = read_tool_batch_source(&fixture.lane, &fixture.drive, &missing)
        .await
        .expect_err("the missing entry rejects");
    assert_eq!(
        error.to_string(),
        "Tool batch assistant entry is invalid",
        "the missing-entry invariant surfaces: {error}"
    );

    commit_writes(
        &fixture.session,
        vec![Write::Entry(Box::new(
            crate::harness::session::commit::insert_entry(common::user_entry("u1", None, "user")),
        ))],
    )
    .await;
    let configuration = fixture.configuration.clone();
    let user_entry_batch = ToolBatch {
        assistant_entry_id: "u1".to_owned(),
        configuration,
        turn_id: "turn-1".to_owned(),
        calls: Vec::new(),
    };
    let error = read_tool_batch_source(&fixture.lane, &fixture.drive, &user_entry_batch)
        .await
        .expect_err("the non-assistant entry rejects");
    assert_eq!(
        error.to_string(),
        "Tool batch assistant entry is invalid",
        "the non-assistant invariant surfaces: {error}"
    );

    let assistant = batch_assistant(1);
    commit_writes(
        &fixture.session,
        vec![Write::Entry(Box::new(
            crate::harness::session::commit::insert_entry(common::assistant_entry(
                "a1", None, assistant,
            )),
        ))],
    )
    .await;
    let configuration = fixture.configuration.clone();
    let far_index_batch = ToolBatch {
        assistant_entry_id: "a1".to_owned(),
        configuration,
        turn_id: "turn-1".to_owned(),
        calls: vec![wire_planned_call(2, "r0")],
    };
    let error = read_tool_batch_source(&fixture.lane, &fixture.drive, &far_index_batch)
        .await
        .expect_err("the far index rejects");
    assert!(
        error
            .to_string()
            .contains("Tool call source index 2 does not name a tool-call block"),
        "the source-index invariant surfaces: {error}"
    );
    close_session(&fixture).await;
}

/// Upstream pin: `toolCallFor`'s invariant. A source read that carries no
/// block at the call's index rejects with the invalid-index invariant.
#[tokio::test]
async fn tool_call_for_rejects_a_missing_source_index() {
    let sources = ToolBatchSource {
        assistant: faux_assistant_message(Vec::new(), FauxAssistantMessageOptions::default()),
        calls: BTreeMap::new(),
    };
    let call = wire_planned_call(0, "r0");
    let error = tool_call_for(&sources, &call).expect_err("the missing block rejects");
    assert_eq!(
        error.to_string(),
        "Tool call source index 0 is invalid",
        "the invalid-index invariant surfaces: {error}"
    );
}

/// Upstream pin: `readPlacement`'s staged-result invariants. An absent,
/// malformed, non-message, and non-tool-result staged payload each reject
/// with the missing-staged-result invariant.
#[tokio::test]
async fn the_placement_rejects_unreadable_staged_results() {
    let (absent, capability) = installed_batch(vec![wire_outcome_ready_call(0, "r0", false)]).await;
    let sources = batch_sources(&absent, &capability).await;
    let error = placement_rejection(&absent, &capability, &sources).await;
    assert_eq!(
        error.to_string(),
        "Tool call r0 is missing its staged result",
        "the absent invariant surfaces: {error}"
    );
    close_session(&absent).await;

    let (malformed, capability) =
        installed_batch(vec![wire_outcome_ready_call(0, "r0", false)]).await;
    let sources = batch_sources(&malformed, &capability).await;
    stage_pending(
        &malformed.session,
        "r0",
        PendingEntry::Custom {
            custom_type: "read".to_owned(),
            payload: Some(json!({})),
        },
    )
    .await;
    let error = placement_rejection(&malformed, &capability, &sources).await;
    assert_eq!(
        error.to_string(),
        "Tool call r0 is missing its staged result",
        "the non-message invariant surfaces: {error}"
    );
    close_session(&malformed).await;

    let (not_result, capability) =
        installed_batch(vec![wire_outcome_ready_call(0, "r0", false)]).await;
    let sources = batch_sources(&not_result, &capability).await;
    stage_pending(
        &not_result.session,
        "r0",
        PendingEntry::Message {
            payload: Box::new(common::user("queued", 1)),
        },
    )
    .await;
    let error = placement_rejection(&not_result, &capability, &sources).await;
    assert_eq!(
        error.to_string(),
        "Tool call r0 is missing its staged result",
        "the non-tool-result invariant surfaces: {error}"
    );
    close_session(&not_result).await;

    let (malformed_payload, capability) =
        installed_batch(vec![wire_outcome_ready_call(0, "r0", false)]).await;
    let sources = batch_sources(&malformed_payload, &capability).await;
    commit_writes(
        &malformed_payload.session,
        vec![raw_write(
            &pending_entry("r0").address,
            json!({ "type": "bogus" }),
        )],
    )
    .await;
    let error = placement_rejection(&malformed_payload, &capability, &sources).await;
    assert_eq!(
        error.to_string(),
        "Tool call r0 is missing its staged result",
        "the malformed-payload invariant surfaces: {error}"
    );
    close_session(&malformed_payload).await;
}

/// Upstream pin: `readPlacement`'s mismatch invariant. A staged result
/// that answers a different tool call rejects with the mismatch
/// invariant.
#[tokio::test]
async fn the_placement_rejects_a_mismatched_staged_result() {
    let (fixture, capability) =
        installed_batch(vec![wire_outcome_ready_call(0, "r0", false)]).await;
    let sources = batch_sources(&fixture, &capability).await;
    stage_pending(
        &fixture.session,
        "r0",
        PendingEntry::Message {
            payload: Box::new(AgentMessage::Standard(Message::ToolResult(
                staged_tool_result("other", "other", "staged"),
            ))),
        },
    )
    .await;

    let error = placement_rejection(&fixture, &capability, &sources).await;
    assert!(
        error
            .to_string()
            .contains("Tool call r0 has a mismatched staged result"),
        "the mismatch invariant surfaces: {error}"
    );
    close_session(&fixture).await;
}

/// Upstream pin: `readPlacement`'s completed-entry invariants. A
/// completed call whose result entry carries no tool result (a compaction
/// entry, then a user message) rejects with the missing-result-entry
/// invariant.
#[tokio::test]
async fn the_placement_rejects_a_completed_call_missing_its_result_entry() {
    let (compaction_case, capability) = installed_batch(vec![
        wire_completed_call(0, "r0", false),
        wire_outcome_ready_call(1, "r1", false),
    ])
    .await;
    let sources = batch_sources(&compaction_case, &capability).await;
    stage_pending(
        &compaction_case.session,
        "r1",
        PendingEntry::Message {
            payload: Box::new(AgentMessage::Standard(Message::ToolResult(
                staged_tool_result("call-1", "read", "staged"),
            ))),
        },
    )
    .await;
    commit_writes(
        &compaction_case.session,
        vec![Write::Entry(Box::new(
            crate::harness::session::commit::insert_entry(NewEntry::Compaction {
                id: "r0".to_owned(),
                parent_id: None,
                body: CompactionEntryBody {
                    summary: "summary".to_owned(),
                    retained_tail: Vec::new(),
                    tokens_before: 0,
                    details: None,
                    usage: None,
                    from_hook: false,
                },
            }),
        ))],
    )
    .await;

    let error = placement_rejection(&compaction_case, &capability, &sources).await;
    assert!(
        error
            .to_string()
            .contains("Completed tool call r0 is missing its result entry"),
        "the completed-entry invariant surfaces: {error}"
    );
    close_session(&compaction_case).await;

    let (user_case, capability) = installed_batch(vec![
        wire_completed_call(0, "r0", false),
        wire_outcome_ready_call(1, "r1", false),
    ])
    .await;
    let sources = batch_sources(&user_case, &capability).await;
    stage_pending(
        &user_case.session,
        "r1",
        PendingEntry::Message {
            payload: Box::new(AgentMessage::Standard(Message::ToolResult(
                staged_tool_result("call-1", "read", "staged"),
            ))),
        },
    )
    .await;
    commit_writes(
        &user_case.session,
        vec![Write::Entry(Box::new(
            crate::harness::session::commit::insert_entry(common::user_entry(
                "r0",
                None,
                "not a result",
            )),
        ))],
    )
    .await;

    let error = placement_rejection(&user_case, &capability, &sources).await;
    assert!(
        error
            .to_string()
            .contains("Completed tool call r0 is missing its result entry"),
        "the completed-entry invariant surfaces: {error}"
    );
    close_session(&user_case).await;
}

// Upstream pin: none. `commitPlacement`'s `` `Completed tool batch has no
// branch tip` `` invariant stays uncovered: every placement commit carries
// at least one item whose result-entry id rebinds `parentId`, so the guard
// is as unreachable through `materializeReady` upstream as it is here.

/// Upstream pin: none (upstream's `readPlacement` returns `None` on the
/// same shape). A placement read on an idle lane returns no items and the
/// materialization settles without a commit.
#[tokio::test]
async fn the_placement_returns_without_a_live_operation() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let assistant = batch_assistant(1);
    commit_writes(
        &fixture.session,
        vec![
            Write::Entry(Box::new(crate::harness::session::commit::insert_entry(
                common::assistant_entry("a1", None, assistant.clone()),
            ))),
            set_value_write(&stored_values::branch_tip("main"), Some("a1".to_owned()))
                .expect("the tip write"),
        ],
    )
    .await;
    let batch = ToolBatch {
        assistant_entry_id: "a1".to_owned(),
        configuration: fixture.configuration.clone(),
        turn_id: "turn-1".to_owned(),
        calls: vec![wire_outcome_ready_call(0, "r0", false)],
    };
    let capability = ToolsOperation {
        scope: common::run_scope(DEFAULT_COMPACTION_SETTINGS),
        batch,
    };
    let sources = read_tool_batch_source(&fixture.lane, &fixture.drive, &capability.batch)
        .await
        .expect("the source reads");

    materialize_ready(&fixture.lane, &fixture.drive, &capability, &sources, false)
        .await
        .expect("the idle read returns no items");
    assert!(
        fixture.lane.state().operation.is_none(),
        "the lane stays idle"
    );
    close_session(&fixture).await;
}

/// Upstream pin: `planBoundaryInbox`'s invariants. A selected steer and
/// write item whose pending payloads are missing each reject with the
/// missing-payload invariant, naming the item's kind.
#[tokio::test]
async fn the_plan_rejects_a_selected_item_missing_its_payload() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let state = plan_state(
        &fixture.configuration,
        vec![
            inbox_item("p1", InboxItemKind::Steer),
            inbox_item("w1", InboxItemKind::Write),
        ],
    );
    let scope = common::run_scope(DEFAULT_COMPACTION_SETTINGS);

    let error = plan_over(&fixture, &state, scope.clone(), false)
        .await
        .expect_err("the missing steer payload rejects");
    assert_eq!(
        error.to_string(),
        "Pending steer entry p1 is missing its payload",
        "the steer invariant surfaces: {error}"
    );

    let write_only = plan_state(
        &fixture.configuration,
        vec![inbox_item("w1", InboxItemKind::Write)],
    );
    let error = plan_over(&fixture, &write_only, scope, true)
        .await
        .expect_err("the fallback reload hits the missing write payload");
    assert_eq!(
        error.to_string(),
        "Pending write entry w1 is missing its payload",
        "the write invariant surfaces: {error}"
    );
    close_session(&fixture).await;
}

/// Upstream pin: `load`'s message probe. A steer item whose stored payload
/// is not a message rejects with the not-a-message invariant.
#[tokio::test]
async fn the_plan_rejects_a_non_message_staged_payload() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let state = plan_state(
        &fixture.configuration,
        vec![inbox_item("p1", InboxItemKind::Steer)],
    );
    commit_writes(
        &fixture.session,
        vec![raw_write(
            &pending_entry("p1").address,
            json!({ "type": "custom" }),
        )],
    )
    .await;

    let error = plan_over(
        &fixture,
        &state,
        common::run_scope(DEFAULT_COMPACTION_SETTINGS),
        false,
    )
    .await
    .expect_err("the non-message payload rejects");
    assert_eq!(
        error.to_string(),
        "Queued steer entry p1 is not a message",
        "the not-a-message invariant surfaces: {error}"
    );
    close_session(&fixture).await;
}

/// Upstream pin: `load`'s typed parse. A message-tagged payload without
/// its message body rejects with the malformed-payload invariant.
#[tokio::test]
async fn the_plan_rejects_a_malformed_staged_payload() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let state = plan_state(
        &fixture.configuration,
        vec![inbox_item("p1", InboxItemKind::Steer)],
    );
    commit_writes(
        &fixture.session,
        vec![raw_write(
            &pending_entry("p1").address,
            json!({ "type": "message" }),
        )],
    )
    .await;

    let error = plan_over(
        &fixture,
        &state,
        common::run_scope(DEFAULT_COMPACTION_SETTINGS),
        false,
    )
    .await
    .expect_err("the malformed payload rejects");
    assert!(
        error
            .to_string()
            .contains("Pending entry payload is malformed"),
        "the malformed invariant surfaces: {error}"
    );
    close_session(&fixture).await;
}

/// Stages the two queued follow-up messages, the pending-routing cases'
/// seeds.
async fn stage_two_pending(fixture: &common::DriveFixture) {
    stage_pending(
        &fixture.session,
        "f1",
        PendingEntry::Message {
            payload: Box::new(common::user("first", 1)),
        },
    )
    .await;
    stage_pending(
        &fixture.session,
        "f2",
        PendingEntry::Message {
            payload: Box::new(common::user("second", 1)),
        },
    )
    .await;
}

/// Upstream pin: `planBoundaryInbox`'s follow-up fallback. With no
/// projecting selection, the follow-up items re-merge into inbox order
/// with a full payload reload and both join the placement.
#[tokio::test]
async fn the_plan_merges_follow_ups_back_into_inbox_order() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let state = plan_state(
        &fixture.configuration,
        vec![
            inbox_item("f2", InboxItemKind::FollowUp),
            inbox_item("f1", InboxItemKind::FollowUp),
        ],
    );
    stage_two_pending(&fixture).await;

    let placement = plan_over(
        &fixture,
        &state,
        common::run_scope(DEFAULT_COMPACTION_SETTINGS),
        true,
    )
    .await
    .expect("the follow-up fallback plans");
    assert_eq!(
        placement
            .entries
            .iter()
            .map(NewEntry::id)
            .collect::<Vec<_>>(),
        vec!["f2".to_owned(), "f1".to_owned()],
        "the follow-ups re-merge into inbox order"
    );
    assert_eq!(
        placement.trigger_entry_id.as_deref(),
        Some("f1"),
        "the last projecting entry triggers"
    );
    assert_eq!(placement.tip_id.as_deref(), Some("f1"), "the tip moves");
    assert!(
        placement.queues.is_some(),
        "the remaining inbox reads its queues"
    );
    close_session(&fixture).await;

    let replan = common::create_drive_fixture(boundary_spec(None)).await;
    let state = plan_state(
        &replan.configuration,
        vec![inbox_item("f1", InboxItemKind::FollowUp)],
    );
    let error = plan_over(
        &replan,
        &state,
        common::run_scope(DEFAULT_COMPACTION_SETTINGS),
        true,
    )
    .await
    .expect_err("the reloaded follow-up payload rejects");
    assert_eq!(
        error.to_string(),
        "Pending followUp entry f1 is missing its payload",
        "the reloaded payload's invariant surfaces: {error}"
    );
    close_session(&replan).await;
}

/// Upstream pin: `planBoundaryInbox`'s follow-up mode read. A
/// `OneAtATime` follow-up mode takes only the oldest follow-up.
#[tokio::test]
async fn the_plan_takes_one_follow_up_at_a_time() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let state = plan_state(
        &fixture.configuration,
        vec![
            inbox_item("f1", InboxItemKind::FollowUp),
            inbox_item("f2", InboxItemKind::FollowUp),
        ],
    );
    stage_two_pending(&fixture).await;
    let mut scope = common::run_scope(DEFAULT_COMPACTION_SETTINGS);
    scope.settings.follow_up_mode = QueueMode::OneAtATime;

    let placement = plan_over(&fixture, &state, scope, true)
        .await
        .expect("the one-at-a-time fallback plans");
    assert_eq!(
        placement
            .entries
            .iter()
            .map(NewEntry::id)
            .collect::<Vec<_>>(),
        vec!["f1".to_owned()],
        "only the oldest follow-up selects"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (upstream's `normalizedRetryPolicy` is exercised
/// with retry enabled; the disabled budget is the same fn's other arm). A
/// disabled retry policy normalizes to a one-attempt budget with the
/// agent-delay fallback.
#[tokio::test]
async fn the_normalized_retry_policy_disables_the_attempt_budget() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let normalized = normalized_retry_policy(&fixture.lane);
    assert_eq!(
        normalized.max_attempts, 1,
        "the disabled budget normalizes to one attempt"
    );
    assert_eq!(normalized.base_delay_ms, 0, "the base delay copies through");
    assert_eq!(
        normalized.max_agent_delay_ms,
        pi_ai::utils::retry::DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
        "the unset agent delay falls back to the default"
    );
    close_session(&fixture).await;
}

/// Upstream pin: `runWithGate` throwing the gate's abort refusal through
/// the hookless `finishRunBoundary` into `driveOperation`'s catch, whose
/// `instanceof AbortRequested` continues the pass. A gate abort begun
/// while the finish mediation runs surfaces as the bare `AbortRequested`
/// the spine's abort catch downcasts — the boxed hook error would fault
/// the harness through the spawn's fail handler instead.
#[tokio::test]
async fn the_finish_boundary_routes_a_mediation_abort_to_the_abort_carrier() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    install_checkpoint_leaf(
        &fixture,
        checkpoint_leaf(
            common::run_scope(DEFAULT_COMPACTION_SETTINGS),
            Continuation::MayFinish {
                include_final_assistant: false,
            },
        ),
    )
    .await;
    common::begin_closed_abort(&fixture.drive);

    let error = finish_over(
        &fixture,
        Continuation::MayFinish {
            include_final_assistant: false,
        },
    )
    .await
    .expect_err("the aborting mediation rejects with the carrier");
    assert!(
        error
            .downcast_ref::<crate::harness::gate::AbortRequested>()
            .is_some(),
        "the hook's gate abort rides the bare carrier the spine downcasts: {error}"
    );
    assert_eq!(error.to_string(), "Abort requested");
    assert!(
        matches!(
            common::current_state(&fixture),
            OperationState::Checkpoint(_)
        ),
        "the operation stays live for reconciliation"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (upstream's finish mediation awaits the same
/// cancelled bounded read). A `cancel_requested` leaf reports the cancel
/// route from the bounded read without planning.
#[tokio::test]
async fn the_finish_boundary_reports_the_cancel_requested_route() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    install_checkpoint_leaf(
        &fixture,
        checkpoint_leaf(
            common::scope_with_control(common::cancelled_control(), DEFAULT_COMPACTION_SETTINGS),
            Continuation::MayFinish {
                include_final_assistant: false,
            },
        ),
    )
    .await;

    let result = finish_over(
        &fixture,
        Continuation::MayFinish {
            include_final_assistant: false,
        },
    )
    .await
    .expect("the cancelled finish continues");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the cancelled finish continues the pass"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (upstream's planner awaits the hook outside the
/// mutation line; the port's hook is the same gap). A cancellation that
/// lands during `before_run_end` routes the planner's continue to the
/// cancel result.
#[tokio::test]
async fn the_finish_boundary_routes_a_mid_hook_cancellation_to_continue() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    install_checkpoint_leaf(
        &fixture,
        checkpoint_leaf(
            common::run_scope(DEFAULT_COMPACTION_SETTINGS),
            Continuation::MayFinish {
                include_final_assistant: false,
            },
        ),
    )
    .await;
    {
        let lane = Arc::clone(&fixture.lane);
        fixture
            .hooks
            .on(
                HookName::BeforeRunEnd,
                Arc::new(move |_invocation: &HookInvocation, _context| {
                    let lane = Arc::clone(&lane);
                    Box::pin(async move {
                        request_cancel(&lane).await;
                        Ok(HookResult::BeforeRunEnd(None))
                    })
                }),
                HookOptions::default(),
            )
            .expect("register before_run_end");
    }

    let result = finish_over(
        &fixture,
        Continuation::MayFinish {
            include_final_assistant: false,
        },
    )
    .await
    .expect("the cancelled planner continues");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the mid-hook cancellation continues the pass"
    );
    assert!(
        matches!(
            common::current_state(&fixture),
            OperationState::Checkpoint(_)
        ),
        "the operation stays live for reconciliation"
    );
    close_session(&fixture).await;
}

/// Upstream pin: `startRun`'s intent invariant. A starting leaf over a
/// compaction intent rejects with the non-run-intent invariant.
#[tokio::test]
async fn the_start_run_rejects_a_non_run_intent() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    common::install_operation(
        &fixture,
        starting_run_state(),
        OperationIntent::Compaction {
            custom_instructions: None,
        },
        common::InstallOptions::default(),
    )
    .await;
    let run = starting_leaf(&fixture);

    let error = start_run(&fixture.lane, &fixture.drive, &run)
        .await
        .expect_err("the non-run intent rejects");
    assert_eq!(
        error.to_string(),
        "Run operation has non-run intent",
        "the intent invariant surfaces: {error}"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (upstream's `startRun` awaits the same cancelled
/// continue). A cancelled starting leaf reports the cancel route without
/// planning.
#[tokio::test]
async fn the_start_run_reports_the_cancel_requested_route() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let cancelled = starting_run_state_with_control(common::cancelled_control());
    common::install_operation(
        &fixture,
        cancelled,
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        common::InstallOptions::default(),
    )
    .await;
    let run = starting_leaf(&fixture);

    let result = start_run(&fixture.lane, &fixture.drive, &run)
        .await
        .expect("the cancelled start continues");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the cancelled start continues the pass"
    );
    close_session(&fixture).await;
}

/// The starting leaf under one control, upstream's `startingRun(control)`
/// fixture variant.
fn starting_run_state_with_control(control: Control) -> OperationState {
    OperationState::Starting(StartingOperation {
        scope: common::scope_with_control(control, DEFAULT_COMPACTION_SETTINGS),
    })
}

/// Upstream pin: none (upstream's `before_run` aggregates fail open — a
/// throwing handler reports and the start proceeds — so the spine's
/// conversion fn binds through the gate and registry rejections instead).
/// The spine's three error conversions map each rejection to the lane
/// error the abort catch recognizes.
#[tokio::test]
async fn the_spine_error_conversions_map_every_rejection() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let (sender, receiver) = tokio::sync::watch::channel(());
    drop(sender);
    let abort = crate::harness::gate::AbortRequested {
        cancellation: receiver.clone(),
    };

    let error = crate::harness::runtime::drive::gate_rejection_error(
        crate::harness::gate::GateRejection::AbortRequested(crate::harness::gate::AbortRequested {
            cancellation: receiver,
        }),
    );
    assert!(
        error
            .as_ref()
            .downcast_ref::<crate::harness::gate::AbortRequested>()
            .is_some(),
        "the admission abort converts to its carried error"
    );

    let error = crate::harness::runtime::drive::gate_rejection_error(
        crate::harness::gate::GateRejection::Closed(crate::harness::gate::GateClosedError(
            "gate closed".to_owned(),
        )),
    );
    assert_eq!(
        error.to_string(),
        "gate closed",
        "the closed gate converts to its error"
    );

    let error = crate::harness::runtime::drive::hook_error_to_lane_error(
        crate::harness::hooks::HookRunError::GateAborted(abort),
        &fixture.drive,
    );
    assert!(
        error
            .as_ref()
            .downcast_ref::<crate::harness::gate::AbortRequested>()
            .is_some(),
        "the gate abort converts to its carried error"
    );
    let error = crate::harness::runtime::drive::hook_error_to_lane_error(
        crate::harness::hooks::HookRunError::Aborted(pi_chord::context::AbortReason::Aborted),
        &fixture.drive,
    );
    assert!(
        error
            .as_ref()
            .downcast_ref::<crate::harness::gate::AbortRequested>()
            .is_some(),
        "the aborted context converts to the pass's abort cancellation"
    );
    let error = crate::harness::runtime::drive::hook_error_to_lane_error(
        crate::harness::hooks::HookRunError::Handler(
            Box::<dyn std::error::Error + Send + Sync>::from("hook exploded"),
        ),
        &fixture.drive,
    );
    assert_eq!(
        error.to_string(),
        "hook exploded",
        "the plain error converts through"
    );

    let boxed =
        crate::harness::runtime::drive::hook_error_box(SessionError::Message("boxed".to_owned()));
    assert_eq!(boxed.to_string(), "boxed", "the box wraps the error");
    close_session(&fixture).await;
}

/// Upstream pin: `startRun`'s injected-message invariant. A `before_run`
/// result carrying a pending assistant message rejects with the
/// pending-assistant invariant.
#[tokio::test]
async fn the_start_run_rejects_a_pending_assistant_injection() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    install_starting_run_with_tip(&fixture).await;
    fixture
        .hooks
        .on(
            HookName::BeforeRun,
            Arc::new(|_invocation: &HookInvocation, _context| {
                let pending = faux_assistant_message(
                    Vec::new(),
                    FauxAssistantMessageOptions {
                        stop_reason: Some(StopReason::Pending),
                        ..FauxAssistantMessageOptions::default()
                    },
                );
                Box::pin(std::future::ready(Ok(HookResult::BeforeRun(Some(
                    BeforeRunResult {
                        messages: vec![AgentMessage::Standard(Message::Assistant(pending))],
                    },
                )))))
            }),
            HookOptions::default(),
        )
        .expect("register before_run");
    let run = starting_leaf(&fixture);

    let error = start_run(&fixture.lane, &fixture.drive, &run)
        .await
        .expect_err("the pending injection rejects");
    assert_eq!(
        error.to_string(),
        "before_run returned a pending assistant message",
        "the pending-assistant invariant surfaces: {error}"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (upstream's start planner throws the same
/// no-trigger invariant). A run with no prompt entries and no branch tip
/// rejects with the no-trigger invariant.
#[tokio::test]
async fn the_start_run_rejects_a_run_without_a_trigger_entry() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    common::install_operation(
        &fixture,
        starting_run_state(),
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        common::InstallOptions {
            tip_id: Some(None),
            ..common::InstallOptions::default()
        },
    )
    .await;
    let run = starting_leaf(&fixture);

    let error = start_run(&fixture.lane, &fixture.drive, &run)
        .await
        .expect_err("the tipless start rejects");
    assert_eq!(
        error.to_string(),
        "Run start has no trigger entry",
        "the no-trigger invariant surfaces: {error}"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (upstream's second planner awaits across the hook;
/// the port's hook is the same gap). A cancellation that lands during
/// `before_run` routes the second planner's continue to the cancel result.
#[tokio::test]
async fn the_start_run_routes_a_mid_hook_cancellation_to_continue() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    install_starting_run_with_tip(&fixture).await;
    {
        let lane = Arc::clone(&fixture.lane);
        fixture
            .hooks
            .on(
                HookName::BeforeRun,
                Arc::new(move |_invocation: &HookInvocation, _context| {
                    let lane = Arc::clone(&lane);
                    Box::pin(async move {
                        request_cancel(&lane).await;
                        Ok(HookResult::BeforeRun(None))
                    })
                }),
                HookOptions::default(),
            )
            .expect("register before_run");
    }
    let run = starting_leaf(&fixture);

    let result = start_run(&fixture.lane, &fixture.drive, &run)
        .await
        .expect("the cancelled start continues");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the mid-hook cancellation continues the pass"
    );
    assert!(
        matches!(
            operation_scope_of(&common::current_state(&fixture)).control,
            Control::CancelRequested { .. }
        ),
        "the operation stays cancelled for reconciliation"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (upstream's `runCheckpoint` awaits the same
/// cancelled threshold read). A cancelled checkpoint leaf reports the
/// threshold read's cancel route.
#[tokio::test]
async fn the_checkpoint_reports_the_cancel_requested_threshold_route() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let leaf = checkpoint_leaf(
        common::scope_with_control(common::cancelled_control(), DEFAULT_COMPACTION_SETTINGS),
        Continuation::NeedAssistant {
            overflow_recovery_used: false,
        },
    );
    install_checkpoint_leaf(&fixture, leaf.clone()).await;

    let result = run_checkpoint(&fixture.lane, &fixture.drive, &leaf)
        .await
        .expect("the cancelled threshold continues");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the cancelled threshold continues the pass"
    );
    close_session(&fixture).await;
}

/// Upstream pin: `finishRunBoundary`'s final-assistant invariant. A
/// `may_finish` continuation that includes the final assistant rejects
/// when the run's scope carries no assistant entry id.
#[tokio::test]
async fn the_checkpoint_rejects_a_completed_run_missing_its_final_assistant() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let leaf = checkpoint_leaf(
        common::run_scope(DEFAULT_COMPACTION_SETTINGS),
        Continuation::MayFinish {
            include_final_assistant: true,
        },
    );
    install_checkpoint_leaf(&fixture, leaf.clone()).await;

    let error = run_checkpoint(&fixture.lane, &fixture.drive, &leaf)
        .await
        .expect_err("the assistant-less finish rejects");
    assert_eq!(
        error.to_string(),
        "Completed run is missing its final assistant",
        "the final-assistant invariant surfaces: {error}"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (upstream's `runCheckpoint` awaits the same cancelled
/// planned read; its suites reach only the threshold read's route). A
/// cancelled checkpoint leaf with compaction disabled skips the threshold
/// read entirely — the plannerless threshold returns before any control
/// check — so the planned read's entry check carries the cancel route.
#[tokio::test]
async fn the_checkpoint_routes_a_cancelled_planned_read_to_continue() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let mut settings = DEFAULT_COMPACTION_SETTINGS;
    settings.enabled = false;
    let leaf = checkpoint_leaf(
        common::scope_with_control(common::cancelled_control(), settings),
        Continuation::NeedAssistant {
            overflow_recovery_used: false,
        },
    );
    install_checkpoint_leaf(&fixture, leaf.clone()).await;
    let OperationState::Checkpoint(run) = common::current_state(&fixture) else {
        unreachable!("the checkpoint leaf installs")
    };

    let result = run_checkpoint(&fixture.lane, &fixture.drive, &run)
        .await
        .expect("the cancelled planned read continues");
    assert!(
        matches!(result, ProcedureResult::Continue),
        "the planned read's cancel route continues the pass"
    );
    close_session(&fixture).await;
}

/// Upstream pin: none (upstream's guard binds the same caller/lane
/// mismatch). A finish mediation whose caller's continuation is not
/// `may_finish` rejects with the finish-continuation invariant.
#[tokio::test]
async fn the_checkpoint_rejects_a_non_finish_mediation() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let leaf = checkpoint_leaf(
        common::run_scope(DEFAULT_COMPACTION_SETTINGS),
        Continuation::MayFinish {
            include_final_assistant: false,
        },
    );
    install_checkpoint_leaf(&fixture, leaf.clone()).await;
    let caller_leaf = checkpoint_leaf(
        leaf.scope.clone(),
        Continuation::NeedAssistant {
            overflow_recovery_used: false,
        },
    );

    let error = run_checkpoint(&fixture.lane, &fixture.drive, &caller_leaf)
        .await
        .expect_err("the non-finish mediation rejects");
    assert_eq!(
        error.to_string(),
        "Checkpoint finish mediation requires a finish continuation",
        "the mediation invariant surfaces: {error}"
    );
    close_session(&fixture).await;
}

/// Appends one assistant frame to the pass's committed prefix, the
/// recovery fixtures' frame seeding.
fn start_frame() -> AssistantMessageFrame {
    AssistantMessageFrame::Start {
        partial: faux_assistant_message(Vec::new(), FauxAssistantMessageOptions::default()),
    }
}

async fn seed_frames(fixture: &common::DriveFixture, frames: Vec<AssistantMessageFrame>) {
    let operation_id = fixture.operation_id.clone();
    commit_writes(
        &fixture.session,
        frames
            .into_iter()
            .map(|frame| {
                append_list_write(
                    &stored_values::pending_assistant_frames(&operation_id, RESPONSE_ENTRY_ID),
                    frame,
                )
                .expect("the frame write")
            })
            .collect(),
    )
    .await;
}

/// Upstream pin: `recoverAssistantGeneration`'s reduce throw (upstream's
/// `reduceAssistantMessageFrames` throws for a malformed frame sequence
/// and the error propagates to the pass). Two start frames read well but
/// do not reduce: the recovery rejects with the reducer's message.
#[tokio::test]
async fn the_assistant_recovery_rejects_a_frame_prefix_that_does_not_reduce() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let leaf = assistant_effect_pending_leaf(common::run_scope(DEFAULT_COMPACTION_SETTINGS));
    common::install_operation(
        &fixture,
        OperationState::AssistantEffectPending(leaf.clone()),
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        common::InstallOptions::default(),
    )
    .await;
    seed_frames(&fixture, vec![start_frame(), start_frame()]).await;

    let error = recover_assistant_generation(&fixture.lane, &fixture.drive, &leaf)
        .await
        .expect_err("the unreducible prefix rejects");
    assert!(
        error.to_string().contains("more than one start frame"),
        "the reducer's message propagates: {error}"
    );
    let entry = fixture
        .session
        .get_entry(RESPONSE_ENTRY_ID, &background_context())
        .await
        .expect("the entry read");
    assert!(entry.is_none(), "the failed recovery settles nothing");
    close_session(&fixture).await;
}

/// Upstream pin: none (upstream's `recoverCancelledAssistantEffect` runs
/// the same reduce throw; the port binds it through the cancelled settle).
/// Two start frames under a cancelled effect-pending leaf reject with the
/// reducer's message instead of settling.
#[tokio::test]
async fn the_cancelled_effect_recovery_rejects_a_frame_prefix_that_does_not_reduce() {
    let fixture = common::create_drive_fixture(boundary_spec(None)).await;
    let leaf = assistant_effect_pending_leaf(common::scope_with_control(
        common::cancelled_control(),
        DEFAULT_COMPACTION_SETTINGS,
    ));
    common::install_operation(
        &fixture,
        OperationState::AssistantEffectPending(leaf.clone()),
        OperationIntent::Run {
            prompt_entry_ids: Vec::new(),
        },
        common::InstallOptions::default(),
    )
    .await;
    seed_frames(&fixture, vec![start_frame(), start_frame()]).await;

    let error = crate::harness::runtime::drive::recovery::recover_cancelled_assistant_effect(
        &fixture.lane,
        &fixture.drive,
        &OperationState::AssistantEffectPending(leaf),
    )
    .await
    .expect_err("the unreducible prefix rejects");
    assert!(
        error.to_string().contains("more than one start frame"),
        "the reducer's message propagates: {error}"
    );
    let entry = fixture
        .session
        .get_entry(RESPONSE_ENTRY_ID, &background_context())
        .await
        .expect("the entry read");
    assert!(entry.is_none(), "the failed recovery settles nothing");
    close_session(&fixture).await;
}

// The sweep's remaining arms stay uncovered, verified against upstream at
// pin `60e7e76`: `planBoundaryInbox`'s `nextRun` kind name (next-run items
// never select; upstream interpolates `${item.kind}` into one template),
// `finishRunBoundary`'s `Completed run has no tip` (upstream rejects a
// null tip in `readBoundedEntries` before the planner reads it, and a
// placement never unsets the tip), `operation_result_record`'s `?` error
// region (a `completed` status never fails the record builder), and the
// `unreachable!` arms (the hook-result variant match, the planner's
// active-operation precondition, the commit's per-write sequence contract,
// the lane-scoped event constructions). `recovery.rs`'s
// `recoverCancelledAssistantEffect` non-effect-pending panic is port-only
// residue — upstream types the parameter as the two-leaf union with no
// runtime default — and waits on a source minimization, not a test.

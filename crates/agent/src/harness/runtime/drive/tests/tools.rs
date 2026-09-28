//! The durable tool batch suite, ported 1:1 from upstream
//! `test/harness/runtime/drive-tools.test.ts` ("durable tool batch") at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The suite drives [`run_tools`] directly over the fixture lane and pins
//! the durable tool batch's phases: sequential execution with memos,
//! checkpoints, hooks, usage, and source-order placement; parallel
//! completion-order staging with source-order materialization; safe replay
//! of persisted arguments and memos against interrupted unsafe effects;
//! restored cancelled batches running nothing; outcome-ready
//! materialization without tool resolution; never-executed length and
//! missing calls; the update-delivery and checkpoint ordering before
//! `after_tool`; and the cancel-before-admission and mid-effect
//! cancellation choreography.
//!
//! Restatements the port makes and the tests bind:
//! - upstream's per-file fixture block restates here — the observing
//!   backend rides [`super::common::ObservedMemoryStorage`], the fixed
//!   `100` storage clock rides [`super::common::fixed_clock_memory_storage`],
//!   and the session-id counter rides [`super::common::next_suite_session_id`].
//! - upstream's local `deferred<T>()` gates ride
//!   [`crate::harness::runtime::test_support::deferred`] pairs (every gate
//!   in this suite is void; the abort carrier is the pass's watch-based
//!   cancellation) and `waitFor(predicate)` rides [`super::common::wait_for`]
//!   with the same 200-poll budget and `condition was not reached` raise;
//!   the one-macrotask rests restate as [`settle_events`].
//! - upstream's `vi.fn()` observers restate as shared atomic counters and
//!   recorded-argument lists; `vi.spyOn` has no counterpart here.
//! - upstream's `lateInvocation` capture (case 1's expired-capability
//!   `setMemo` rejection) cannot restate: the landed execute signature
//!   hands the invocation as a `&dyn` reference that cannot outlive its
//!   future, so the case keeps the same settlement's durable memo-family
//!   scan, which pins the cleanup the rejection guards.
//! - upstream's `afterEach` (concurrent session closes) restates as an
//!   explicit session close at each test's end.

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

use pi_ai::models::create_models;
use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxContentBlock;
use pi_ai::providers::faux::RegisterFauxProviderOptions;
use pi_ai::providers::faux::{faux_assistant_message, faux_provider, faux_tool_call};
use pi_ai::types::Message;
use pi_ai::types::{StopReason, TextContent, ToolResultBlock, ToolResultMessage, Usage, UsageCost};
use pi_ai::utils::retry::RetryPolicy;
use serde_json::Value as JsonValue;
use serde_json::json;

use super::common;
use crate::harness::agent_harness::BeforeToolResult;
use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HarnessEventType;
use crate::harness::agent_harness::HookEvent;
use crate::harness::agent_harness::{HookInvocation, HookName, HookOptions, HookResult, ToolBlock};
use crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS;
use crate::harness::context::{Context, background_context};
use crate::harness::runtime::drive::tools::run_tools;
use crate::harness::runtime::lane::{EmitBatch, Lane, operation_state_with_scope};
use crate::harness::runtime::restore::restore_lane;
use crate::harness::runtime::test_support::commit_writes;
use crate::harness::runtime::test_support::deferred;
use crate::harness::runtime::test_support::lane_state_write;
use crate::harness::runtime::test_support::lock;
use crate::harness::runtime::test_support::noop_hook_reporter;
use crate::harness::runtime::test_support::passthrough_fault_handler;
use crate::harness::runtime::test_support::{runtime_session_metadata, settle_events};
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneCommand;
use crate::harness::runtime::types::{LaneError, LaneState, LiveOperation, ProcedureResult};
use crate::harness::session::commit::insert_entry;
use crate::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use crate::harness::session::types::Continuation;
use crate::harness::session::types::Control;
use crate::harness::session::types::Entry;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::ModelIdentity;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationMeta;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::RunSettings;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::Storage;
use crate::harness::session::types::ToolBatch;
use crate::harness::session::types::{
    ToolCall, ToolCallStatus, ToolsOperation, operation_scope_of,
};
use crate::harness::session::values as stored_values;
use crate::harness::session::values::{ToolOutputPayload, Write, set_value_write};
use crate::harness::types::AgentHarnessStreamOptions;
use crate::harness::types::AgentHarnessTool;
use crate::harness::types::AgentHarnessToolContextSource;
use crate::harness::types::AgentHarnessToolExecuteFn;
use crate::harness::types::AgentHarnessToolInvocation;
use crate::harness::types::AgentHarnessToolUpdateCallback;
use crate::harness::types::{AgentHarnessToolUpdateOptions, ToolContext};
use crate::types::AgentMessage;
use crate::types::AgentToolContent;
use crate::types::AgentToolError;
use crate::types::{AgentToolResult, QueueMode, ThinkingLevel, ToolExecutionMode, ToolReplay};

/// The per-call phase builder, upstream's `FixtureOptions.callStates`
/// callback shape.
type CallStatesBuilder = Box<dyn FnOnce(&[String]) -> Vec<ToolCall>>;

/// One seeded batch call, upstream's `FixtureOptions.calls` row
/// `{ name, value? }`; the absent value defaults to the name.
struct CallSpec {
    /// The tool name the assistant's tool-call block carries.
    name: &'static str,
    /// The `value` argument override; `None` falls back to the name.
    value: Option<&'static str>,
}

/// The extra-write parts the fixture hands the seeding closure, upstream's
/// `extraWrites` argument `{ operationId, assistantEntryId, resultEntryIds,
/// run }`.
struct ExtraWriteParts {
    /// The seeded operation id.
    operation_id: String,
    /// The seeded assistant entry id.
    assistant_entry_id: String,
    /// The reserved result-entry ids, one per call.
    result_entry_ids: Vec<String>,
    /// The seeded tools leaf.
    #[expect(
        dead_code,
        reason = "upstream's extraWrites parts carry the whole fixture shape; no case reads the run"
    )]
    run: ToolsOperation,
}

/// The tools suite's fixture options, upstream's `FixtureOptions`.
#[derive(Default)]
struct FixtureOptions {
    /// The seeded batch calls, upstream's `calls`.
    calls: Vec<CallSpec>,
    /// The harness tools the config carries, upstream's `tools`.
    tools: Option<Vec<AgentHarnessTool>>,
    /// The execution mode, upstream's `mode`.
    mode: Option<ToolExecutionMode>,
    /// The assistant stop reason, upstream's `stopReason`.
    stop_reason: Option<StopReason>,
    /// The batch's call phases, upstream's `callStates`.
    call_states: Option<CallStatesBuilder>,
    /// The extra seed writes, upstream's `extraWrites`.
    extra_writes: Option<Box<dyn FnOnce(ExtraWriteParts) -> Vec<Write>>>,
    /// The tool-context source, upstream's `toolContext`.
    tool_context: Option<AgentHarnessToolContextSource>,
    /// Whether the seeded control is `cancel_requested`, upstream's
    /// `cancelled`.
    cancelled: bool,
    /// The parked emit hook, upstream's `onEmit`.
    on_emit: Option<EmitBatch>,
}

/// The per-test fixture, upstream's `Fixture` interface.
struct Fixture {
    /// The lane the procedure drives, upstream's `lane`.
    lane: Arc<Lane>,
    /// The installed pass, upstream's `drive`.
    drive: Arc<Drive>,
    /// The session the lane runs on, upstream's `session`.
    session: Arc<StorageBackedSession>,
    /// The hook registry, upstream's `hooks`.
    hooks: Arc<crate::harness::hooks::HookRegistry>,
    /// The collected events, upstream's `events`.
    events: Arc<Mutex<Vec<HarnessEvent>>>,
    /// The seeded assistant entry id, upstream's `assistantEntryId`.
    assistant_entry_id: String,
    /// The reserved result-entry ids, upstream's `resultEntryIds`.
    result_entry_ids: Vec<String>,
    /// The seeded operation id, upstream's `operationId`.
    operation_id: String,
    /// The observing backend, upstream's `storage.observations` reads.
    observed: Arc<common::ObservedMemoryStorage>,
}

/// The tool-argument schema, upstream's
/// `const schema = Type.Object({ value: Type.String() })` wire document.
fn schema() -> JsonValue {
    json!({
        "type": "object",
        "properties": { "value": { "type": "string" } },
        "required": ["value"],
    })
}

/// The plain done tool result the executing fixtures return, upstream's
/// `AgentToolResult { content: [text("done")] }` literals.
fn done_tool_result() -> AgentToolResult {
    AgentToolResult {
        content: vec![text_block("done")],
        details: json!({}),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

/// One text content block, the suite's `{ type: "text", text }` literals.
fn text_block(text: &str) -> AgentToolContent {
    AgentToolContent::Text(TextContent {
        text: text.to_owned(),
        text_signature: None,
    })
}

/// The typed tool-argument payload from its wire object, upstream's
/// `Record<string, JsonValue>` literals against the
/// `Value<ToolArgs>`-typed address.
fn tool_args(wire: &JsonValue) -> stored_values::ToolArgs {
    wire.as_object()
        .expect("the tool arguments are an object")
        .clone()
}

/// Builds one harness tool, upstream's `tool(name, execute, replay =
/// "never")`: the tool's name rides the declarative surface and the
/// `value` argument reaches the executor; the replay policy is `None` when
/// upstream's default `"never"` applies (the intent publisher's
/// `?? "never"` reads the same).
fn tool(
    name: &str,
    execute: Arc<AgentHarnessToolExecuteFn>,
    replay: Option<ToolReplay>,
) -> AgentHarnessTool {
    AgentHarnessTool {
        tool: pi_ai::types::Tool {
            name: name.to_owned(),
            description: name.to_owned(),
            parameters: schema(),
            constrained_sampling: None,
        },
        label: name.to_owned(),
        prepare_arguments: None,
        execute,
        replay,
        execution_mode: None,
    }
}

/// The batch-call literal, upstream's inline
/// `{ status, sourceIndex, resultEntryId, ... }` objects; the phase field
/// is private to the session types module, so the literal restates through
/// the wire shape both sides already serialize.
fn wire_call(wire: JsonValue) -> ToolCall {
    serde_json::from_value(wire)
        .unwrap_or_else(|error| unreachable!("the tool call round-trips: {error}"))
}

/// The planned phase, upstream's `{ status: "planned", ... }`.
fn wire_planned_call(source_index: u64, result_entry_id: &str) -> ToolCall {
    wire_call(json!({
        "sourceIndex": source_index,
        "resultEntryId": result_entry_id,
        "status": "planned",
    }))
}

/// The effect-pending phase, upstream's
/// `{ status: "effect_pending", replay, ... }`.
fn wire_effect_pending_call(
    source_index: u64,
    result_entry_id: &str,
    replay: ToolReplay,
) -> ToolCall {
    wire_call(json!({
        "sourceIndex": source_index,
        "resultEntryId": result_entry_id,
        "status": "effect_pending",
        "replay": replay,
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

#[expect(
    clippy::too_many_lines,
    reason = "the fixture assembles the tools suite's faux, storage, hooks, and lane wiring in one place"
)]
/// Builds one fixture, upstream's `createFixture`: the observing backend
/// over the fixed `100` clock, the storage-backed session, the seeded ids,
/// the faux provider, the assistant entry with one tool-call block per
/// call (`call-{index}` ids), the tools leaf, the seed commit, the fresh
/// models catalog, the event-recording emit hook, and the pass installed
/// on the restored `main` lane.
///
/// # Panics
/// A write's serialization failure or the seed commit's failure.
async fn create_fixture(options: FixtureOptions) -> Fixture {
    let observed = Arc::new(common::ObservedMemoryStorage::new(
        common::fixed_clock_memory_storage(),
    ));
    let storage: Arc<dyn Storage> = observed.clone();
    let session = Arc::new(StorageBackedSession::new(
        runtime_session_metadata(common::next_suite_session_id("drive-tools")),
        storage,
        StorageBackedSessionOptions::default(),
    ));
    let operation_id = session.id_generator().next(Some(10));
    let assistant_entry_id = session.id_generator().next(Some(20));
    let result_entry_ids: Vec<String> = (0..options.calls.len())
        .map(|_| session.id_generator().next(Some(20)))
        .collect();
    let faux = faux_provider(RegisterFauxProviderOptions::default());
    let models = Arc::new(create_models(None));
    models.set_provider(Arc::new(faux.provider.clone()));
    let model = faux.first_model();
    let configuration = LaneConfiguration {
        model: ModelIdentity {
            provider: model.provider.0.clone(),
            model_id: model.id.clone(),
        },
        thinking_level: ThinkingLevel::Off,
        active_tool_names: options
            .calls
            .iter()
            .map(|call| call.name.to_owned())
            .collect(),
    };
    let blocks: Vec<FauxContentBlock> = options
        .calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let mut arguments = serde_json::Map::new();
            arguments.insert(
                "value".to_owned(),
                JsonValue::String(call.value.unwrap_or(call.name).to_owned()),
            );
            faux_tool_call(call.name, arguments, Some(format!("call-{index}")))
        })
        .collect();
    let assistant = faux_assistant_message(
        blocks,
        FauxAssistantMessageOptions {
            stop_reason: Some(options.stop_reason.unwrap_or(StopReason::ToolUse)),
            timestamp: Some(20),
            ..FauxAssistantMessageOptions::default()
        },
    );
    let calls = options.call_states.map_or_else(
        || {
            result_entry_ids
                .iter()
                .enumerate()
                .map(|(index, result_entry_id)| wire_planned_call(index as u64, result_entry_id))
                .collect()
        },
        |build| build(&result_entry_ids),
    );
    let run = ToolsOperation {
        scope: OperationScope {
            control: if options.cancelled {
                Control::CancelRequested { requested_at: 30 }
            } else {
                Control::Running
            },
            settings: RunSettings {
                compaction: DEFAULT_COMPACTION_SETTINGS,
                steering_mode: QueueMode::All,
                follow_up_mode: QueueMode::All,
                tool_execution: options.mode.unwrap_or(ToolExecutionMode::Parallel),
            },
            latest_assistant_entry_id: Some(assistant_entry_id.clone()),
        },
        batch: ToolBatch {
            assistant_entry_id: assistant_entry_id.clone(),
            configuration: configuration.clone(),
            turn_id: "turn-1".to_owned(),
            calls,
        },
    };
    let mut writes: Vec<Write> = vec![
        Write::Entry(Box::new(insert_entry(NewEntry::Message {
            id: assistant_entry_id.clone(),
            parent_id: None,
            body: Box::new(MessageEntry {
                message: AgentMessage::Standard(Message::Assistant(assistant)),
                terminate: None,
            }),
        }))),
        set_value_write(
            &stored_values::branch_tip("main"),
            Some(assistant_entry_id.clone()),
        )
        .expect("the tip write"),
        set_value_write(&stored_values::lane_config("main"), configuration)
            .expect("the config write"),
        lane_state_write("main", Some(&operation_id), None, Vec::new())
            .expect("the lane state write"),
        set_value_write(
            &stored_values::operation_meta(&operation_id),
            OperationMeta {
                operation_id: operation_id.clone(),
                lane: "main".to_owned(),
                source_tip_id: None,
                started_at: 10,
                intent: OperationIntent::Run {
                    prompt_entry_ids: Vec::new(),
                },
            },
        )
        .expect("the meta write"),
        set_value_write(
            &stored_values::operation_state(&operation_id),
            OperationState::Tools(run.clone()),
        )
        .expect("the state write"),
    ];
    if let Some(extra_writes) = options.extra_writes {
        writes.extend(extra_writes(ExtraWriteParts {
            operation_id: operation_id.clone(),
            assistant_entry_id: assistant_entry_id.clone(),
            result_entry_ids: result_entry_ids.clone(),
            run,
        }));
    }
    commit_writes(&session, writes).await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let emit: EmitBatch = {
        let events = Arc::clone(&events);
        let observed = Arc::clone(&observed);
        let on_emit = options.on_emit;
        Arc::new(move |batch: Vec<HarnessEvent>, context: Context| {
            let events = Arc::clone(&events);
            let observed = Arc::clone(&observed);
            let on_emit = on_emit.clone();
            Box::pin(async move {
                lock(&events).extend(batch.iter().cloned());
                observed.record_observations(
                    batch
                        .iter()
                        .map(|event| event.event_type().as_str().to_owned())
                        .collect(),
                );
                if let Some(on_emit) = on_emit {
                    on_emit(batch, context).await?;
                }
                Ok(())
            })
        })
    };
    let hooks = Arc::new(crate::harness::hooks::HookRegistry::new(
        noop_hook_reporter(),
    ));
    let mut config = common::drive_config(
        RetryPolicy {
            enabled: false,
            max_retries: 0,
            base_delay_ms: 0,
            max_agent_delay_ms: None,
        },
        AgentHarnessStreamOptions::default(),
        None,
    );
    config.tools = options.tools.unwrap_or_default();
    config.tool_execution = options.mode.unwrap_or(ToolExecutionMode::Parallel);
    config.tool_context = options.tool_context;
    let restored = restore_lane(session.clone(), "main", &background_context())
        .await
        .expect("restore");
    let lane = Arc::new(Lane::new(
        "main",
        session.clone(),
        models,
        hooks.clone(),
        restored,
        passthrough_fault_handler(),
        emit,
        common::unused_watch("tool procedure"),
        Arc::new({
            let config = config.clone();
            move || config.clone()
        }),
    ));
    let drive = Arc::new(Drive::new(
        &DriveOptions {
            operation_id: operation_id.clone(),
            wait_for_retry: None,
            poll_deferred: None,
        },
        &background_context(),
    ));
    lane.set_active_drive(Some(Arc::clone(&drive)));
    Fixture {
        lane,
        drive,
        session,
        hooks,
        events,
        assistant_entry_id,
        result_entry_ids,
        operation_id,
        observed,
    }
}

/// The live operation's durable state leaf, upstream's `currentRun`:
/// raises `fixture has no operation` when the lane is idle.
///
/// # Panics
/// The absence of a live operation.
fn current_run(fixture: &Fixture) -> OperationState {
    fixture
        .lane
        .state()
        .operation
        .expect("fixture has no operation")
        .state
}

/// The batch's calls, upstream's `currentCalls`: empty away from the
/// tools leaf.
fn current_calls(fixture: &Fixture) -> Vec<ToolCall> {
    match current_run(fixture) {
        OperationState::Tools(run) => run.batch.calls,
        _ => Vec::new(),
    }
}

/// The seeded tools leaf, for the spawned drives; raises
/// `fixture has no tool batch` at any other leaf, upstream's
/// `driveTools` throw.
///
/// # Panics
/// The leaf not being `tools`.
fn tools_run(fixture: &Fixture) -> ToolsOperation {
    match current_run(fixture) {
        OperationState::Tools(run) => run,
        _ => panic!("fixture has no tool batch"),
    }
}

/// Drives the fixture's tool batch, upstream's `driveTools`.
///
/// # Panics
/// The leaf not being `tools`.
async fn drive_tools(fixture: &Fixture) -> Result<ProcedureResult, LaneError> {
    let run = tools_run(fixture);
    run_tools(&fixture.lane, &fixture.drive, &run).await
}

/// The projection-restore assertion, upstream's `expectProjectionRestores`:
/// the lane's owned state deep-equals a fresh `restoreLane` read.
///
/// # Panics
/// The restore's failure or the state mismatch.
async fn expect_projection_restores(fixture: &Fixture) {
    let restored = restore_lane(fixture.session.clone(), "main", &background_context())
        .await
        .expect("restore");
    assert_eq!(
        fixture.lane.state(),
        restored,
        "the lane projection restores"
    );
}

/// Closes the fixture's session, upstream's `afterEach` close.
///
/// # Panics
/// The close's failure.
async fn close_session(fixture: &Fixture) {
    fixture
        .session
        .close(&background_context())
        .await
        .expect("the session closes");
}

/// The observation's first position, upstream's `observations.indexOf`;
/// raises when the marker never landed.
///
/// # Panics
/// The marker missing.
fn observation_index(observations: &[String], marker: &str) -> usize {
    observations
        .iter()
        .position(|observed| observed == marker)
        .unwrap_or_else(|| panic!("the {marker} observation is recorded"))
}

/// The event's first position by type, upstream's
/// `eventTypes.indexOf(event.type)`; raises when the event never landed.
///
/// # Panics
/// The event missing.
fn event_index(events: &[HarnessEvent], event_type: HarnessEventType) -> usize {
    events
        .iter()
        .position(|event| event.event_type() == event_type)
        .unwrap_or_else(|| panic!("the {} event is recorded", event_type.as_str()))
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins the sequential batch's memos, checkpoints, hooks, usage, and placement end to end"
)]
/// Upstream `it`: executes a sequential batch with memos, checkpoints,
/// hooks, usage, and source-order placement.
#[tokio::test]
async fn executes_a_sequential_batch_with_memos_checkpoints_hooks_usage_and_source_order_placement()
{
    let context_resolutions = Arc::new(AtomicUsize::new(0));
    // The tool-side usage payload the placement writes, upstream's `usage`
    // literal served by the fixture.
    let usage = Usage {
        input: 1,
        output: 2,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 3,
        cost: UsageCost::default(),
    };
    let first = tool(
        "first",
        Arc::new(
            move |_tool_call_id: &str,
                  args: &JsonValue,
                  on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
                  _tool_context: ToolContext,
                  invocation: &dyn AgentHarnessToolInvocation,
                  context: &Context| {
                let value = args["value"].as_str().unwrap_or_default().to_owned();
                let usage = usage;
                Box::pin(async move {
                    invocation
                        .set_memo("step/a", Some(json!({ "value": "memo" })), context)
                        .await
                        .expect("the memo writes");
                    let memo = invocation
                        .get_memo("step/a", context)
                        .await
                        .expect("the memo reads");
                    assert_eq!(
                        memo,
                        Some(json!({ "value": "memo" })),
                        "the memo round-trips"
                    );
                    if let Some(on_update) = on_update {
                        on_update(
                            &AgentToolResult {
                                content: vec![text_block("partial")],
                                details: json!({ "progress": "partial" }),
                                usage: None,
                                added_tool_names: None,
                                terminate: None,
                            },
                            Some(AgentHarnessToolUpdateOptions { checkpoint: true }),
                        );
                    }
                    Ok(AgentToolResult {
                        content: vec![text_block(&value)],
                        details: json!({ "value": value }),
                        usage: Some(usage),
                        added_tool_names: None,
                        terminate: None,
                    })
                })
            },
        ),
        None,
    );
    let second = tool(
        "second",
        Arc::new(
            move |_tool_call_id: &str,
                  args: &JsonValue,
                  _on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
                  _tool_context: ToolContext,
                  _invocation: &dyn AgentHarnessToolInvocation,
                  _context: &Context| {
                let value = args["value"].as_str().unwrap_or_default().to_owned();
                Box::pin(async move {
                    Ok(AgentToolResult {
                        content: vec![text_block(&value)],
                        details: json!({ "value": value }),
                        usage: None,
                        added_tool_names: Some(vec!["introduced".to_owned()]),
                        terminate: None,
                    })
                })
            },
        ),
        None,
    );
    let fixture = create_fixture(FixtureOptions {
        calls: vec![
            CallSpec {
                name: "first",
                value: None,
            },
            CallSpec {
                name: "second",
                value: None,
            },
        ],
        tools: Some(vec![first, second]),
        mode: Some(ToolExecutionMode::Sequential),
        tool_context: Some(AgentHarnessToolContextSource::Resolved({
            let resolutions = Arc::clone(&context_resolutions);
            Arc::new(move |_context: &Context| {
                resolutions.fetch_add(1, Ordering::SeqCst);
                Box::pin(std::future::ready(None))
            })
        })),
        ..FixtureOptions::default()
    })
    .await;
    let checkpoint_seen_by_after_hook = Arc::new(AtomicBool::new(false));
    let session = Arc::clone(&fixture.session);
    let operation_id = fixture.operation_id.clone();
    let result_entry_id = fixture.result_entry_ids[0].clone();
    let seen = Arc::clone(&checkpoint_seen_by_after_hook);
    fixture
        .hooks
        .on(
            HookName::BeforeTool,
            Arc::new(|invocation: &HookInvocation, _context: &Context| {
                Box::pin(async move {
                    let HookEvent::BeforeTool {
                        tool_name, args, ..
                    } = &invocation.event
                    else {
                        panic!("the before_tool invocation carries its event");
                    };
                    if tool_name == "first" {
                        let mut args = args.clone();
                        args.insert("value".to_owned(), json!("prepared"));
                        Ok(HookResult::BeforeTool(Some(BeforeToolResult {
                            args: Some(args),
                            block: None,
                        })))
                    } else {
                        Ok(HookResult::BeforeTool(None))
                    }
                })
            }),
            HookOptions::default(),
        )
        .expect("register before_tool");
    fixture
        .hooks
        .on(
            HookName::AfterTool,
            Arc::new(move |invocation: &HookInvocation, context: &Context| {
                let session = Arc::clone(&session);
                let seen = Arc::clone(&seen);
                let operation_id = operation_id.clone();
                let result_entry_id = result_entry_id.clone();
                Box::pin(async move {
                    let HookEvent::AfterTool { tool_name, .. } = &invocation.event else {
                        panic!("the after_tool invocation carries its event");
                    };
                    if tool_name == "first" {
                        let staged = session
                            .get_value(
                                &stored_values::pending_tool_output(
                                    &operation_id,
                                    &result_entry_id,
                                )
                                .address,
                                context,
                            )
                            .await
                            .expect("the checkpoint reads");
                        seen.store(staged.is_some(), Ordering::SeqCst);
                    }
                    Ok(HookResult::AfterTool(None))
                })
            }),
            HookOptions::default(),
        )
        .expect("register after_tool");

    let advanced = drive_tools(&fixture).await.expect("the batch drives");
    assert!(
        matches!(advanced, ProcedureResult::Continue),
        "the batch continues"
    );
    let transcript = common::transcript(&fixture.lane).await;
    assert_eq!(
        transcript.iter().map(Entry::id).collect::<Vec<_>>(),
        vec![
            fixture.assistant_entry_id.clone(),
            fixture.result_entry_ids[0].clone(),
            fixture.result_entry_ids[1].clone(),
        ],
        "the results place in source order"
    );
    let Entry::Message { body, .. } = &transcript[1] else {
        panic!("the first result is a message entry");
    };
    let AgentMessage::Standard(Message::ToolResult(result)) = &body.message else {
        panic!("the entry carries the tool result");
    };
    assert_eq!(
        result.content.first(),
        Some(&text_block("prepared")),
        "the prepared arguments reach the result text"
    );
    assert_eq!(
        context_resolutions.load(Ordering::SeqCst),
        1,
        "the tool context resolves once"
    );
    assert!(
        checkpoint_seen_by_after_hook.load(Ordering::SeqCst),
        "the after_tool hook sees the staged checkpoint"
    );
    let OperationState::Checkpoint(leaf) = current_run(&fixture) else {
        panic!("the batch lands at checkpoint");
    };
    assert!(
        matches!(
            leaf.checkpoint.continuation,
            Continuation::NeedAssistant {
                overflow_recovery_used: false
            }
        ),
        "the batch continues to assistant generation"
    );
    assert_eq!(
        fixture.lane.state().configuration.active_tool_names,
        vec![
            "first".to_owned(),
            "second".to_owned(),
            "introduced".to_owned(),
        ],
        "the introduced tool joins the lane configuration"
    );
    let tool_args = fixture
        .session
        .scan_values(
            &stored_values::operation_tool_args_prefix(&fixture.operation_id, None).address,
            &background_context(),
        )
        .await
        .expect("the tool args scan");
    assert!(tool_args.is_empty(), "the tool args family scans empty");
    let tool_memos = fixture
        .session
        .scan_values(
            &stored_values::operation_tool_memo_prefix(&fixture.operation_id, None).address,
            &background_context(),
        )
        .await
        .expect("the tool memo scan");
    assert!(tool_memos.is_empty(), "the tool memo family scans empty");
    let staged = fixture
        .session
        .get_value(
            &stored_values::pending_tool_output(
                &fixture.operation_id,
                &fixture.result_entry_ids[0],
            )
            .address,
            &background_context(),
        )
        .await
        .expect("the checkpoint reads");
    assert!(
        staged.is_none(),
        "the staged checkpoint deletes with the outcome"
    );
    // Upstream's `lateInvocation.setMemo("late", true)` rejection ("no
    // longer owns") cannot restate: the invocation reference cannot outlive
    // its execute future; the memo-family scan above pins the same cleanup
    // contract.

    let events = lock(&fixture.events).clone();
    assert!(
        event_index(&events, HarnessEventType::ToolUpdate)
            < event_index(&events, HarnessEventType::ToolEnd),
        "the tool update publishes before the tool end"
    );
    drop(events);
    let observations = fixture.observed.observations();
    let first_intent = observation_index(&observations, "intent_commit");
    let first_start = observation_index(&observations, "tool_start");
    let first_update = observation_index(&observations, "tool_update");
    let first_commit = observation_index(&observations, "outcome_commit");
    let first_end = observation_index(&observations, "tool_end");
    let first_placement = observation_index(&observations, "entry_added");
    assert!(
        first_intent < first_start,
        "the intent commits before the tool starts: {observations:?}"
    );
    assert!(
        first_start < first_update,
        "the tool starts before its update: {observations:?}"
    );
    assert!(
        first_update < first_commit,
        "the update publishes before the outcome stages: {observations:?}"
    );
    assert!(
        first_commit < first_end,
        "the outcome commits before the tool ends: {observations:?}"
    );
    assert!(
        first_end < first_placement,
        "the tool ends before the entry places: {observations:?}"
    );
    let events = lock(&fixture.events).clone();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type() == HarnessEventType::EntryAdded)
            .count(),
        2,
        "the two results publish their entries"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type() == HarnessEventType::Usage)
            .count(),
        1,
        "the usage-bearing result publishes its row"
    );
    assert_eq!(
        events.last().map(HarnessEvent::event_type),
        Some(HarnessEventType::TurnEnd),
        "the batch closes its turn"
    );
    drop(events);
    expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins the staged completion order against the source-order materialization"
)]
/// Upstream `it`: stages parallel completion order but materializes source
/// order.
#[tokio::test]
async fn stages_parallel_completion_order_but_materializes_source_order() {
    let (finish_a_tx, finish_a_rx) = deferred();
    let (finish_b_tx, finish_b_rx) = deferred();
    let started: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let make =
        |name: &'static str,
         finish_cell: Arc<Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>| {
            let started = Arc::clone(&started);
            tool(
                name,
                Arc::new(
                    move |_tool_call_id: &str,
                          _args: &JsonValue,
                          _on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
                          _tool_context: ToolContext,
                          _invocation: &dyn AgentHarnessToolInvocation,
                          _context: &Context| {
                        let started = Arc::clone(&started);
                        let finish_cell = Arc::clone(&finish_cell);
                        Box::pin(async move {
                            lock(&started).push(name);
                            let finish = lock(&finish_cell).take();
                            if let Some(finish) = finish {
                                let _ = finish.await;
                            }
                            Ok(AgentToolResult {
                                content: vec![text_block(name)],
                                details: json!({ "value": name }),
                                usage: None,
                                added_tool_names: None,
                                terminate: None,
                            })
                        })
                    },
                ),
                None,
            )
        };
    let fixture = create_fixture(FixtureOptions {
        calls: vec![
            CallSpec {
                name: "a",
                value: None,
            },
            CallSpec {
                name: "b",
                value: None,
            },
        ],
        tools: Some(vec![
            make("a", Arc::new(Mutex::new(Some(finish_a_rx)))),
            make("b", Arc::new(Mutex::new(Some(finish_b_rx)))),
        ]),
        mode: Some(ToolExecutionMode::Parallel),
        ..FixtureOptions::default()
    })
    .await;
    let lane = Arc::clone(&fixture.lane);
    let drive = Arc::clone(&fixture.drive);
    let run = tools_run(&fixture);
    let running = tokio::spawn(async move { run_tools(&lane, &drive, &run).await });
    common::wait_for(|| lock(&started).len() == 2).await;
    let _ = finish_b_tx.send(());
    common::wait_for(|| {
        matches!(
            current_calls(&fixture).get(1).map(ToolCall::status),
            Some(ToolCallStatus::OutcomeReady { .. })
        )
    })
    .await;
    let calls = current_calls(&fixture);
    assert!(
        matches!(calls[0].status(), ToolCallStatus::EffectPending { .. }),
        "a stays effect-pending"
    );
    assert!(
        matches!(calls[1].status(), ToolCallStatus::OutcomeReady { .. }),
        "b is outcome-ready"
    );
    let b_entry = fixture
        .session
        .get_entry(&fixture.result_entry_ids[1], &background_context())
        .await
        .expect("the entry reads");
    assert!(b_entry.is_none(), "b's entry is not yet on the branch");
    let events = lock(&fixture.events).clone();
    let b_start = events
        .iter()
        .find(|event| {
            matches!(
                &event.payload,
                HarnessEventPayload::ToolStart { tool_name, .. } if tool_name == "b"
            )
        })
        .expect("b's tool start");
    match &b_start.payload {
        HarnessEventPayload::ToolStart { args, .. } => {
            assert_eq!(args["value"], json!("b"), "b starts with its arguments");
        }
        _ => unreachable!("the start payload"),
    }
    let b_end = events
        .iter()
        .find(|event| {
            matches!(
                &event.payload,
                HarnessEventPayload::ToolEnd { tool_name, .. } if tool_name == "b"
            )
        })
        .expect("b's tool end");
    match &b_end.payload {
        HarnessEventPayload::ToolEnd { result, .. } => {
            assert_eq!(
                result.content.first(),
                Some(&text_block("b")),
                "b ends with its result"
            );
        }
        _ => unreachable!("the end payload"),
    }
    drop(events);
    let _ = finish_a_tx.send(());
    running
        .await
        .expect("the pass joins")
        .expect("the batch drives");

    let transcript = common::transcript(&fixture.lane).await;
    assert_eq!(
        transcript.iter().map(Entry::id).collect::<Vec<_>>(),
        vec![
            fixture.assistant_entry_id.clone(),
            fixture.result_entry_ids[0].clone(),
            fixture.result_entry_ids[1].clone(),
        ],
        "the results materialize in source order"
    );
    expect_projection_restores(&fixture).await;
    close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins the safe replays while the unsafe effects interrupt"
)]
/// Upstream `it`: safe-replays persisted arguments and memos while
/// interrupting unsafe effects.
#[tokio::test]
async fn safe_replays_persisted_arguments_and_memos_while_interrupting_unsafe_effects() {
    let safe_args = Arc::new(Mutex::new(Vec::<JsonValue>::new()));
    let safe_recorded = Arc::clone(&safe_args);
    let safe_execute = tool(
        "safe",
        Arc::new(
            move |_tool_call_id: &str,
                  args: &JsonValue,
                  _on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
                  _tool_context: ToolContext,
                  invocation: &dyn AgentHarnessToolInvocation,
                  context: &Context| {
                let recorded = Arc::clone(&safe_recorded);
                Box::pin(async move {
                    lock(&recorded).push(args.clone());
                    assert!(
                        !invocation.invocation_id().is_empty(),
                        "the replay carries the invocation identity"
                    );
                    let memo = invocation
                        .get_memo("step/a", context)
                        .await
                        .expect("the memo reads");
                    assert_eq!(
                        memo,
                        Some(json!({ "complete": true })),
                        "the memo survives the crash"
                    );
                    Ok(AgentToolResult {
                        content: vec![text_block("safe replay")],
                        details: json!({ "value": "safe" }),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    })
                })
            },
        ),
        Some(ToolReplay::Safe),
    );
    let unsafe_count = Arc::new(AtomicUsize::new(0));
    let unsafe_counted = Arc::clone(&unsafe_count);
    let unsafe_execute = tool(
        "unsafe",
        Arc::new(
            move |_tool_call_id: &str,
                  _args: &JsonValue,
                  _on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
                  _tool_context: ToolContext,
                  _invocation: &dyn AgentHarnessToolInvocation,
                  _context: &Context| {
                let count = Arc::clone(&unsafe_counted);
                Box::pin(async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    Ok(AgentToolResult {
                        content: vec![text_block("must not run")],
                        details: json!({}),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    })
                })
            },
        ),
        Some(ToolReplay::Never),
    );
    let fixture = create_fixture(FixtureOptions {
        calls: vec![
            CallSpec {
                name: "safe",
                value: None,
            },
            CallSpec {
                name: "unsafe",
                value: None,
            },
        ],
        tools: Some(vec![safe_execute, unsafe_execute]),
        call_states: Some(Box::new(|result_entry_ids: &[String]| {
            vec![
                wire_effect_pending_call(0, &result_entry_ids[0], ToolReplay::Safe),
                wire_effect_pending_call(1, &result_entry_ids[1], ToolReplay::Never),
            ]
        })),
        extra_writes: Some(Box::new(|parts: ExtraWriteParts| {
            let ExtraWriteParts {
                operation_id,
                result_entry_ids,
                ..
            } = parts;
            vec![
                set_value_write(
                    &stored_values::operation_tool_args(&operation_id, "turn-1", 0),
                    tool_args(&json!({ "value": "persisted" })),
                )
                .expect("the persisted args write"),
                set_value_write(
                    &stored_values::operation_tool_args(&operation_id, "turn-1", 1),
                    tool_args(&json!({ "value": "unsafe" })),
                )
                .expect("the unsafe args write"),
                set_value_write(
                    &stored_values::operation_tool_memo(
                        &operation_id,
                        &result_entry_ids[0],
                        "step/a",
                    ),
                    json!({ "complete": true }),
                )
                .expect("the memo write"),
                set_value_write(
                    &stored_values::pending_tool_output(&operation_id, &result_entry_ids[0]),
                    ToolOutputPayload {
                        content: vec![text_block("old progress")],
                        details: json!({}),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    },
                )
                .expect("the safe checkpoint write"),
                set_value_write(
                    &stored_values::pending_tool_output(&operation_id, &result_entry_ids[1]),
                    ToolOutputPayload {
                        content: vec![text_block("durable partial")],
                        details: json!({ "progress": "kept" }),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    },
                )
                .expect("the unsafe checkpoint write"),
            ]
        })),
        ..FixtureOptions::default()
    })
    .await;

    drive_tools(&fixture).await.expect("the batch recovers");
    assert_eq!(lock(&safe_args).len(), 1, "the safe tool replays once");
    assert_eq!(
        lock(&safe_args)[0]["value"],
        json!("persisted"),
        "the replay uses the persisted arguments"
    );
    assert_eq!(
        unsafe_count.load(Ordering::SeqCst),
        0,
        "the unsafe tool never executes"
    );
    let events = lock(&fixture.events).clone();
    let safe_start = events
        .iter()
        .find(|event| {
            matches!(
                &event.payload,
                HarnessEventPayload::ToolStart { tool_call_id, .. } if tool_call_id == "call-0"
            )
        })
        .expect("the replayed call starts");
    assert!(
        safe_start.recovery,
        "the replayed start is recovery-flagged"
    );
    match &safe_start.payload {
        HarnessEventPayload::ToolStart { args, .. } => {
            assert_eq!(
                args["value"],
                json!("persisted"),
                "the start carries the persisted args"
            );
        }
        _ => unreachable!("the start payload"),
    }
    drop(events);
    let observations = fixture.observed.observations();
    assert!(
        observation_index(&observations, "replay_commit")
            < observation_index(&observations, "tool_start"),
        "the checkpoint clears before the replay starts: {observations:?}"
    );
    let events = lock(&fixture.events).clone();
    let safe_end = events
        .iter()
        .find(|event| {
            matches!(
                &event.payload,
                HarnessEventPayload::ToolEnd { tool_call_id, .. } if tool_call_id == "call-0"
            )
        })
        .expect("the replayed call ends");
    assert!(safe_end.recovery, "the replayed end is recovery-flagged");
    match &safe_end.payload {
        HarnessEventPayload::ToolEnd { is_error, .. } => {
            assert!(!is_error, "the replayed call succeeds");
        }
        _ => unreachable!("the end payload"),
    }
    drop(events);
    let result = common::tool_result_message(
        &fixture.session,
        &fixture.result_entry_ids[1],
        "the interrupted entry",
    )
    .await;
    assert!(result.is_error, "the interrupted result is an error");
    assert_eq!(
        result.details.as_ref(),
        Some(&json!({ "progress": "kept" })),
        "the staged checkpoint details survive"
    );
    match result.content.last() {
        Some(ToolResultBlock::Text(text)) => assert!(
            text.text.contains("external outcome is unknown"),
            "the interruption marker closes the result: {}",
            text.text
        ),
        _ => panic!("the marker is text"),
    }
    let events = lock(&fixture.events).clone();
    assert!(
        !events.iter().any(|event| matches!(
            &event.payload,
            HarnessEventPayload::ToolStart { tool_call_id, .. } if tool_call_id == "call-1"
        )),
        "the unsafe call never starts"
    );
    let unsafe_end = events
        .iter()
        .find(|event| {
            matches!(
                &event.payload,
                HarnessEventPayload::ToolEnd { tool_call_id, .. } if tool_call_id == "call-1"
            )
        })
        .expect("the unsafe call ends");
    assert!(
        unsafe_end.recovery,
        "the interrupted end is recovery-flagged"
    );
    match &unsafe_end.payload {
        HarnessEventPayload::ToolEnd { is_error, .. } => {
            assert!(*is_error, "the interrupted call ends with an error");
        }
        _ => unreachable!("the end payload"),
    }
    drop(events);
    close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case drives the restored cancelled batch's hookless reconciliation"
)]
/// Upstream `it`: reconciles a restored cancelled batch without hooks,
/// context, or effects.
#[tokio::test]
async fn reconciles_a_restored_cancelled_batch_without_hooks_context_or_effects() {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let context_count = Arc::new(AtomicUsize::new(0));
    let execute = counting_execute(Arc::clone(&execute_count), Vec::new());
    let fixture = create_fixture(FixtureOptions {
        calls: vec![
            CallSpec {
                name: "planned",
                value: None,
            },
            CallSpec {
                name: "pending",
                value: None,
            },
        ],
        tools: Some(vec![
            tool("planned", Arc::clone(&execute), None),
            tool("pending", execute, Some(ToolReplay::Safe)),
        ]),
        tool_context: Some(AgentHarnessToolContextSource::Resolved({
            let resolutions = Arc::clone(&context_count);
            Arc::new(move |_context: &Context| {
                resolutions.fetch_add(1, Ordering::SeqCst);
                Box::pin(std::future::ready(None))
            })
        })),
        cancelled: true,
        call_states: Some(Box::new(|result_entry_ids: &[String]| {
            vec![
                wire_planned_call(0, &result_entry_ids[0]),
                wire_effect_pending_call(1, &result_entry_ids[1], ToolReplay::Safe),
            ]
        })),
        extra_writes: Some(Box::new(|parts: ExtraWriteParts| {
            let ExtraWriteParts {
                operation_id,
                result_entry_ids,
                ..
            } = parts;
            vec![
                set_value_write(
                    &stored_values::operation_tool_args(&operation_id, "turn-1", 1),
                    tool_args(&json!({ "value": "pending" })),
                )
                .expect("the pending args write"),
                set_value_write(
                    &stored_values::pending_tool_output(&operation_id, &result_entry_ids[1]),
                    ToolOutputPayload {
                        content: vec![text_block("checkpoint")],
                        details: json!({}),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    },
                )
                .expect("the staged checkpoint write"),
            ]
        })),
        ..FixtureOptions::default()
    })
    .await;
    let before_tool_ran = Arc::new(AtomicUsize::new(0));
    let after_tool_ran = Arc::new(AtomicUsize::new(0));
    for (name, ran) in [
        (HookName::BeforeTool, Arc::clone(&before_tool_ran)),
        (HookName::AfterTool, Arc::clone(&after_tool_ran)),
    ] {
        fixture
            .hooks
            .on(
                name,
                Arc::new(move |_invocation: &HookInvocation, _context: &Context| {
                    let ran = Arc::clone(&ran);
                    Box::pin(async move {
                        ran.fetch_add(1, Ordering::SeqCst);
                        Ok(if name == HookName::BeforeTool {
                            HookResult::BeforeTool(None)
                        } else {
                            HookResult::AfterTool(None)
                        })
                    })
                }),
                HookOptions::default(),
            )
            .expect("register the tool hook");
    }

    drive_tools(&fixture)
        .await
        .expect("the cancelled batch settles");
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "no effect runs");
    assert_eq!(
        context_count.load(Ordering::SeqCst),
        0,
        "no context resolves"
    );
    assert_eq!(
        before_tool_ran.load(Ordering::SeqCst),
        0,
        "no before_tool runs"
    );
    assert_eq!(
        after_tool_ran.load(Ordering::SeqCst),
        0,
        "no after_tool runs"
    );
    let OperationState::Checkpoint(leaf) = current_run(&fixture) else {
        panic!("the batch lands at checkpoint");
    };
    assert!(
        matches!(leaf.scope.control, Control::CancelRequested { .. }),
        "the control stays cancel_requested"
    );
    assert!(
        matches!(
            leaf.checkpoint.continuation,
            Continuation::NeedAssistant { .. }
        ),
        "the cancelled batch continues to assistant generation"
    );
    let result = common::tool_result_message(
        &fixture.session,
        &fixture.result_entry_ids[1],
        "the interrupted entry",
    )
    .await;
    match result.content.last() {
        Some(ToolResultBlock::Text(text)) => assert!(
            text.text.contains("external outcome is unknown"),
            "the interrupted result carries the unknown-outcome marker: {}",
            text.text
        ),
        _ => panic!("the marker is text"),
    }
    let events = lock(&fixture.events).clone();
    let starts: Vec<&HarnessEvent> = events
        .iter()
        .filter(|event| event.event_type() == HarnessEventType::ToolStart)
        .collect();
    assert_eq!(starts.len(), 1, "only the planned call starts");
    let started_call_id = match &starts[0].payload {
        HarnessEventPayload::ToolStart {
            tool_name,
            tool_call_id,
            args,
            ..
        } => {
            assert_eq!(tool_name, "planned", "the planned call starts");
            assert_eq!(
                args["value"],
                json!("planned"),
                "the start carries the raw arguments"
            );
            tool_call_id.clone()
        }
        _ => unreachable!("the start payload"),
    };
    let ends: Vec<&HarnessEvent> = events
        .iter()
        .filter(|event| event.event_type() == HarnessEventType::ToolEnd)
        .collect();
    assert_eq!(ends.len(), 2, "both calls end");
    let started_end = ends
        .iter()
        .find(|event| {
            matches!(
                &event.payload,
                HarnessEventPayload::ToolEnd { tool_call_id, .. } if *tool_call_id == started_call_id
            )
        })
        .expect("the started call ends");
    match &started_end.payload {
        HarnessEventPayload::ToolEnd { is_error, .. } => {
            assert!(*is_error, "the aborted planned call ends with an error");
        }
        _ => unreachable!("the end payload"),
    }
    drop(events);
    close_session(&fixture).await;
}

/// Upstream `it`: materializes outcome-ready state without resolving tools
/// or tool context.
#[tokio::test]
async fn materializes_outcome_ready_state_without_resolving_tools_or_tool_context() {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let context_count = Arc::new(AtomicUsize::new(0));
    let executing = counting_tool("ready", Arc::clone(&execute_count), Vec::new(), None);
    let fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "ready",
            value: None,
        }],
        tools: Some(vec![executing]),
        tool_context: Some(AgentHarnessToolContextSource::Resolved({
            let resolutions = Arc::clone(&context_count);
            Arc::new(move |_context: &Context| {
                resolutions.fetch_add(1, Ordering::SeqCst);
                Box::pin(std::future::ready(None))
            })
        })),
        call_states: Some(Box::new(|result_entry_ids: &[String]| {
            vec![wire_outcome_ready_call(0, &result_entry_ids[0], true)]
        })),
        extra_writes: Some(Box::new(|parts: ExtraWriteParts| {
            vec![
                set_value_write(
                    &stored_values::pending_entry(&parts.result_entry_ids[0]),
                    PendingEntry::Message {
                        payload: Box::new(AgentMessage::Standard(Message::ToolResult(
                            ToolResultMessage {
                                tool_call_id: "call-0".to_owned(),
                                tool_name: "ready".to_owned(),
                                content: vec![text_block("already done")],
                                details: None,
                                usage: None,
                                added_tool_names: Some(vec!["later".to_owned()]),
                                is_error: false,
                                timestamp: 30,
                            },
                        ))),
                    },
                )
                .expect("the staged entry write"),
            ]
        })),
        ..FixtureOptions::default()
    })
    .await;

    drive_tools(&fixture)
        .await
        .expect("the staged batch materializes");
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "no effect runs");
    assert_eq!(
        context_count.load(Ordering::SeqCst),
        0,
        "no context resolves"
    );
    let OperationState::Checkpoint(leaf) = current_run(&fixture) else {
        panic!("the staged batch lands at checkpoint");
    };
    assert!(
        matches!(
            leaf.checkpoint.continuation,
            Continuation::MayFinish {
                include_final_assistant: false
            }
        ),
        "the terminating batch may finish without the final assistant"
    );
    assert_eq!(
        fixture.lane.state().configuration.active_tool_names,
        vec!["ready".to_owned(), "later".to_owned()],
        "the staged result introduces the tool"
    );
    let events = lock(&fixture.events).clone();
    let first = events.first().expect("the first event");
    assert_eq!(first.event_type(), HarnessEventType::TurnStart);
    assert!(first.recovery, "the recovery turn opens");
    match &first.payload {
        HarnessEventPayload::TurnStart { turn_id, .. } => {
            assert_eq!(turn_id, "turn-1", "the turn identity");
        }
        _ => unreachable!("the turn start payload"),
    }
    let last = events.last().expect("the last event");
    assert_eq!(last.event_type(), HarnessEventType::TurnEnd);
    assert!(last.recovery, "the recovery turn closes");
    match &last.payload {
        HarnessEventPayload::TurnEnd { turn_id, .. } => {
            assert_eq!(turn_id, "turn-1", "the turn identity");
        }
        _ => unreachable!("the turn end payload"),
    }
    drop(events);
    close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case drives the genuine-length and missing-call rejection arms"
)]
/// Upstream `it`: never executes genuine-length or missing tool calls.
#[tokio::test]
async fn never_executes_genuine_length_or_missing_tool_calls() {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let executing = counting_tool("present", Arc::clone(&execute_count), Vec::new(), None);
    let truncated = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "present",
            value: None,
        }],
        tools: Some(vec![executing]),
        stop_reason: Some(StopReason::Length),
        ..FixtureOptions::default()
    })
    .await;
    let truncated_after_tool = Arc::new(AtomicUsize::new(0));
    let ran = Arc::clone(&truncated_after_tool);
    truncated
        .hooks
        .on(
            HookName::AfterTool,
            Arc::new(move |_invocation: &HookInvocation, _context: &Context| {
                let ran = Arc::clone(&ran);
                Box::pin(async move {
                    ran.fetch_add(1, Ordering::SeqCst);
                    Ok(HookResult::AfterTool(None))
                })
            }),
            HookOptions::default(),
        )
        .expect("register after_tool");
    drive_tools(&truncated)
        .await
        .expect("the truncated batch settles");
    assert_eq!(
        execute_count.load(Ordering::SeqCst),
        0,
        "the length-stopped call never executes"
    );
    let result = common::tool_result_message(
        &truncated.session,
        &truncated.result_entry_ids[0],
        "the result entry",
    )
    .await;
    match result.content.first() {
        Some(ToolResultBlock::Text(text)) => assert!(
            text.text.contains("arguments may be truncated"),
            "the truncation text: {}",
            text.text
        ),
        _ => panic!("the truncation result is text"),
    }
    assert_eq!(
        truncated_after_tool.load(Ordering::SeqCst),
        0,
        "the after_tool hook never runs"
    );
    let events = lock(&truncated.events).clone();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type() == HarnessEventType::ToolStart)
            .count(),
        1,
        "the truncated call starts once"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type() == HarnessEventType::ToolEnd)
            .count(),
        1,
        "the truncated call ends once"
    );
    drop(events);
    let observations = truncated.observed.observations();
    let outcome = observation_index(&observations, "outcome_commit");
    let start = observation_index(&observations, "tool_start");
    let end = observation_index(&observations, "tool_end");
    let placement = observation_index(&observations, "entry_added");
    assert!(
        outcome < start,
        "the immediate outcome commits before its start event: {observations:?}"
    );
    assert!(start < end, "the start precedes the end: {observations:?}");
    assert!(
        end < placement,
        "the end precedes the placement: {observations:?}"
    );
    close_session(&truncated).await;

    let missing = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "missing",
            value: None,
        }],
        tools: Some(Vec::new()),
        ..FixtureOptions::default()
    })
    .await;
    let missing_after_tool = Arc::new(AtomicUsize::new(0));
    let ran = Arc::clone(&missing_after_tool);
    missing
        .hooks
        .on(
            HookName::AfterTool,
            Arc::new(move |_invocation: &HookInvocation, _context: &Context| {
                let ran = Arc::clone(&ran);
                Box::pin(async move {
                    ran.fetch_add(1, Ordering::SeqCst);
                    Ok(HookResult::AfterTool(None))
                })
            }),
            HookOptions::default(),
        )
        .expect("register after_tool");
    drive_tools(&missing)
        .await
        .expect("the missing batch settles");
    let result = common::tool_result_message(
        &missing.session,
        &missing.result_entry_ids[0],
        "the result entry",
    )
    .await;
    assert!(result.is_error, "the missing tool result is an error");
    assert_eq!(
        result.content.first(),
        Some(&text_block("Tool \"missing\" is unavailable")),
        "the unavailable text"
    );
    let wire = serde_json::to_value(result.clone()).expect("the message serializes");
    assert!(
        wire.get("details").is_none(),
        "the missing-tool result omits the details key"
    );
    assert_eq!(
        missing_after_tool.load(Ordering::SeqCst),
        0,
        "the after_tool hook never runs"
    );
    let events = lock(&missing.events).clone();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type() == HarnessEventType::ToolStart)
            .count(),
        1,
        "the missing call starts once"
    );
    let missing_start = events
        .iter()
        .find(|event| event.event_type() == HarnessEventType::ToolStart)
        .expect("the missing call starts");
    match &missing_start.payload {
        HarnessEventPayload::ToolStart { args, .. } => {
            assert_eq!(
                args["value"],
                json!("missing"),
                "the start carries the raw arguments"
            );
        }
        _ => unreachable!("the start payload"),
    }
    let missing_ends: Vec<&HarnessEvent> = events
        .iter()
        .filter(|event| event.event_type() == HarnessEventType::ToolEnd)
        .collect();
    assert_eq!(missing_ends.len(), 1, "the missing call ends once");
    match &missing_ends[0].payload {
        HarnessEventPayload::ToolEnd {
            result, is_error, ..
        } => {
            assert_eq!(
                result.content.first(),
                Some(&text_block("Tool \"missing\" is unavailable")),
                "the end carries the unavailable text"
            );
            assert!(
                result.details.is_null(),
                "the end result carries no details"
            );
            assert!(*is_error, "the missing call ends with an error");
        }
        _ => unreachable!("the end payload"),
    }
    drop(events);
    close_session(&missing).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins the update delivery and checkpoint persistence ordering before after_tool"
)]
/// Upstream `it`: awaits update delivery and checkpoint persistence before
/// `after_tool`.
#[tokio::test]
async fn awaits_update_delivery_and_checkpoint_persistence_before_after_tool() {
    let (release_update_tx, release_update_rx) = deferred();
    let (update_queued_tx, update_queued_rx) = deferred();
    let after_started = Arc::new(AtomicBool::new(false));
    let executing = tool(
        "updating",
        Arc::new(
            move |_tool_call_id: &str,
                  _args: &JsonValue,
                  on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
                  _tool_context: ToolContext,
                  _invocation: &dyn AgentHarnessToolInvocation,
                  _context: &Context| {
                Box::pin(async move {
                    if let Some(on_update) = on_update {
                        on_update(
                            &AgentToolResult {
                                content: vec![text_block("partial")],
                                details: json!({ "progress": "partial" }),
                                usage: None,
                                added_tool_names: None,
                                terminate: None,
                            },
                            Some(AgentHarnessToolUpdateOptions { checkpoint: true }),
                        );
                    }
                    Ok(done_tool_result())
                })
            },
        ),
        None,
    );
    let queued_cell = Arc::new(Mutex::new(Some(update_queued_tx)));
    let release_cell = Arc::new(Mutex::new(Some(release_update_rx)));
    let on_emit: EmitBatch = Arc::new(move |batch: Vec<HarnessEvent>, _context: Context| {
        let queued_cell = Arc::clone(&queued_cell);
        let release_cell = Arc::clone(&release_cell);
        Box::pin(async move {
            if batch
                .iter()
                .any(|event| event.event_type() == HarnessEventType::ToolUpdate)
            {
                let queued = lock(&queued_cell).take();
                let release = lock(&release_cell).take();
                if let Some(queued) = queued {
                    let _ = queued.send(());
                }
                if let Some(release) = release {
                    let _ = release.await;
                }
            }
            Ok(())
        })
    });
    let fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "updating",
            value: None,
        }],
        tools: Some(vec![executing]),
        on_emit: Some(on_emit),
        ..FixtureOptions::default()
    })
    .await;
    let session = Arc::clone(&fixture.session);
    let operation_id = fixture.operation_id.clone();
    let result_entry_id = fixture.result_entry_ids[0].clone();
    let started = Arc::clone(&after_started);
    fixture
        .hooks
        .on(
            HookName::AfterTool,
            Arc::new(move |_invocation: &HookInvocation, context: &Context| {
                let session = Arc::clone(&session);
                let started = Arc::clone(&started);
                let operation_id = operation_id.clone();
                let result_entry_id = result_entry_id.clone();
                Box::pin(async move {
                    started.store(true, Ordering::SeqCst);
                    let staged = session
                        .get_value(
                            &stored_values::pending_tool_output(&operation_id, &result_entry_id)
                                .address,
                            context,
                        )
                        .await
                        .expect("the checkpoint reads");
                    assert!(
                        staged.is_some(),
                        "the after_tool hook sees the staged checkpoint"
                    );
                    Ok(HookResult::AfterTool(None))
                })
            }),
            HookOptions::default(),
        )
        .expect("register after_tool");

    let lane = Arc::clone(&fixture.lane);
    let drive = Arc::clone(&fixture.drive);
    let run = tools_run(&fixture);
    let running = tokio::spawn(async move { run_tools(&lane, &drive, &run).await });
    update_queued_rx.await.expect("the update queues");
    settle_events().await;
    assert!(
        !after_started.load(Ordering::SeqCst),
        "the parked update delivery blocks after_tool"
    );
    let _ = release_update_tx.send(());
    running
        .await
        .expect("the pass joins")
        .expect("the batch drives");
    assert!(
        after_started.load(Ordering::SeqCst),
        "after_tool starts after the delivery settles"
    );
    close_session(&fixture).await;
}

/// Upstream `it`: does not stage a cancelled outcome before cancellation is
/// durable.
#[tokio::test]
async fn does_not_stage_a_cancelled_outcome_before_cancellation_is_durable() {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let executing = counting_tool(
        "cancel-before-admission",
        Arc::clone(&execute_count),
        Vec::new(),
        None,
    );
    let fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "cancel-before-admission",
            value: None,
        }],
        tools: Some(vec![executing]),
        mode: Some(ToolExecutionMode::Sequential),
        ..FixtureOptions::default()
    })
    .await;
    let (cancellation_tx, cancellation_rx) = tokio::sync::watch::channel(());
    fixture.drive.begin_abort(cancellation_rx);
    let lane = Arc::clone(&fixture.lane);
    let drive = Arc::clone(&fixture.drive);
    let run = tools_run(&fixture);
    let running = tokio::spawn(async move { run_tools(&lane, &drive, &run).await });
    settle_events().await;

    assert!(
        matches!(current_calls(&fixture)[0].status(), ToolCallStatus::Planned),
        "the planned call stays planned"
    );
    let staged = fixture
        .session
        .get_value(
            &stored_values::pending_entry(&fixture.result_entry_ids[0]).address,
            &background_context(),
        )
        .await
        .expect("the staged entry reads");
    assert!(
        staged.is_none(),
        "the cancelled outcome is not staged before cancellation is durable"
    );
    common::commit_next_operation_state(
        &fixture.lane,
        &fixture.operation_id,
        Vec::new(),
        |state| {
            operation_state_with_scope(
                state,
                OperationScope {
                    control: Control::CancelRequested { requested_at: 40 },
                    ..operation_scope_of(state)
                },
            )
        },
        "the cancel commits",
    )
    .await;
    let _ = cancellation_tx.send(());
    fixture.drive.signal_abort();
    running
        .await
        .expect("the pass joins")
        .expect("the batch drives");

    assert_eq!(
        execute_count.load(Ordering::SeqCst),
        0,
        "the tool never executes"
    );
    let placed = fixture
        .session
        .get_entry(&fixture.result_entry_ids[0], &background_context())
        .await
        .expect("the entry reads");
    assert!(
        placed.is_some(),
        "the aborted outcome stages after cancellation"
    );
    close_session(&fixture).await;
}

#[expect(
    clippy::too_many_lines,
    reason = "the case pins the live-update drain and the cancelled non-terminating staging"
)]
/// Upstream `it`: drains live updates before `after_tool` and stages a
/// non-terminating result after cancellation.
#[tokio::test]
async fn drains_live_updates_before_after_tool_and_stages_a_non_terminating_result_after_cancellation()
 {
    let (update_delivery_tx, update_delivery_rx) = deferred();
    let (started_tx, started_rx) = deferred();
    let update_delivered = Arc::new(AtomicBool::new(false));
    let after_tool_ran = Arc::new(AtomicUsize::new(0));
    let ran_for_hook = Arc::clone(&after_tool_ran);
    let delivered_for_hook = Arc::clone(&update_delivered);
    let after_tool_hook: crate::harness::agent_harness::HookHandler =
        Arc::new(move |_invocation: &HookInvocation, _context: &Context| {
            let ran = Arc::clone(&ran_for_hook);
            let delivered = Arc::clone(&delivered_for_hook);
            Box::pin(async move {
                ran.fetch_add(1, Ordering::SeqCst);
                assert!(
                    delivered.load(Ordering::SeqCst),
                    "the update delivery settles before after_tool"
                );
                Ok(HookResult::AfterTool(None))
            })
        });
    let started_cell = Arc::new(Mutex::new(Some(started_tx)));
    let executing = tool(
        "slow",
        Arc::new(
            move |_tool_call_id: &str,
                  _args: &JsonValue,
                  on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
                  _tool_context: ToolContext,
                  _invocation: &dyn AgentHarnessToolInvocation,
                  context: &Context| {
                let started_cell = Arc::clone(&started_cell);
                Box::pin(async move {
                    if let Some(on_update) = on_update {
                        on_update(
                            &AgentToolResult {
                                content: vec![text_block("partial")],
                                details: json!({ "progress": "partial" }),
                                usage: None,
                                added_tool_names: None,
                                terminate: None,
                            },
                            None,
                        );
                    }
                    let started = lock(&started_cell).take();
                    if let Some(started) = started {
                        let _ = started.send(());
                    }
                    let signal = context
                        .abort_signal()
                        .expect("the admitted context carries the abort signal");
                    let _ = signal.wait().await;
                    Err::<AgentToolResult, AgentToolError>("cancelled effect".into())
                })
            },
        ),
        None,
    );
    let delivery_cell = Arc::new(Mutex::new(Some(update_delivery_rx)));
    let delivered = Arc::clone(&update_delivered);
    let on_emit: EmitBatch = Arc::new(move |batch: Vec<HarnessEvent>, _context: Context| {
        let delivery_cell = Arc::clone(&delivery_cell);
        let delivered = Arc::clone(&delivered);
        Box::pin(async move {
            if batch
                .iter()
                .any(|event| event.event_type() == HarnessEventType::ToolUpdate)
            {
                let delivery = lock(&delivery_cell).take();
                if let Some(delivery) = delivery {
                    let _ = delivery.await;
                }
                delivered.store(true, Ordering::SeqCst);
            }
            Ok(())
        })
    });
    let fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "slow",
            value: None,
        }],
        tools: Some(vec![executing]),
        mode: Some(ToolExecutionMode::Sequential),
        on_emit: Some(on_emit),
        ..FixtureOptions::default()
    })
    .await;
    fixture
        .hooks
        .on(HookName::AfterTool, after_tool_hook, HookOptions::default())
        .expect("register after_tool");
    let lane = Arc::clone(&fixture.lane);
    let drive = Arc::clone(&fixture.drive);
    let run = tools_run(&fixture);
    let running = tokio::spawn(async move { run_tools(&lane, &drive, &run).await });
    started_rx.await.expect("the tool starts");
    let (cancellation_tx, cancellation_rx) = tokio::sync::watch::channel(());
    fixture.drive.begin_abort(cancellation_rx);
    common::commit_next_operation_state(
        &fixture.lane,
        &fixture.operation_id,
        Vec::new(),
        |state| {
            operation_state_with_scope(
                state,
                OperationScope {
                    control: Control::CancelRequested { requested_at: 40 },
                    ..operation_scope_of(state)
                },
            )
        },
        "the cancel commits",
    )
    .await;
    let _ = cancellation_tx.send(());
    fixture.drive.signal_abort();
    let _ = update_delivery_tx.send(());
    running
        .await
        .expect("the pass joins")
        .expect("the batch drives");
    assert_eq!(
        after_tool_ran.load(Ordering::SeqCst),
        0,
        "the cancelled run never reaches after_tool"
    );
    let entry = fixture
        .session
        .get_entry(&fixture.result_entry_ids[0], &background_context())
        .await
        .expect("the entry reads")
        .expect("the placed entry");
    let Entry::Message { body, .. } = entry else {
        panic!("the entry is a message");
    };
    assert!(
        body.terminate.is_none(),
        "the terminate hint is stripped under cancellation"
    );
    let AgentMessage::Standard(Message::ToolResult(result)) = &body.message else {
        panic!("the entry carries the tool result");
    };
    assert!(result.is_error, "the cancelled effect is an error");
    close_session(&fixture).await;
}

// The remaining cases are port-local boundary tests: they bind uncovered
// arms the upstream suite does not reach. The upstream source at pin
// `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` stays the behavior oracle —
// every bound arm exists there, and the memo/tool-result strings carry
// verbatim.

/// The completed call phase, upstream's `{ status: "completed", terminate }`.
fn wire_completed_call(source_index: u64, result_entry_id: &str, terminate: bool) -> ToolCall {
    wire_call(json!({
        "sourceIndex": source_index,
        "resultEntryId": result_entry_id,
        "status": "completed",
        "terminate": terminate,
    }))
}

/// The raw malformed value write the malformed-payload cases seed, bypassing
/// the typed serializer with a JSON scalar.
fn malformed_value_write(address: &stored_values::ValueAddress, payload: JsonValue) -> Write {
    Write::ValueSet(stored_values::ValueSetWrite {
        kind: "value".to_owned(),
        op: "set".to_owned(),
        namespace: address.namespace.clone(),
        key: address.key.clone(),
        value: payload,
    })
}

/// The counting execute the counting cases build, upstream's
/// `executeCount`-guarded `tool(...)` literals: every invocation bumps the
/// counter and resolves the given content blocks.
fn counting_execute(
    count: Arc<AtomicUsize>,
    content: Vec<AgentToolContent>,
) -> Arc<AgentHarnessToolExecuteFn> {
    Arc::new(
        move |_tool_call_id: &str,
              _args: &JsonValue,
              _on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
              _tool_context: ToolContext,
              _invocation: &dyn AgentHarnessToolInvocation,
              _context: &Context| {
            let count = Arc::clone(&count);
            let content = content.clone();
            Box::pin(async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(AgentToolResult {
                    content,
                    details: json!({}),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        },
    )
}

fn counting_tool(
    name: &'static str,
    count: Arc<AtomicUsize>,
    content: Vec<AgentToolContent>,
    replay: Option<ToolReplay>,
) -> AgentHarnessTool {
    tool(name, counting_execute(count, content), replay)
}

/// The counting execute that must never run, upstream's
/// `executeCount`-guarded tools: every use asserts the counter stays at
/// zero after the call; the replay policy restates the tool literal's.
fn never_executing_tool(
    name: &'static str,
    count: Arc<AtomicUsize>,
    replay: Option<ToolReplay>,
) -> AgentHarnessTool {
    counting_tool(name, count, vec![text_block("must not run")], replay)
}

/// Port-local boundary: an open batch whose every call completed, upstream's
/// `` `Tool batch remained open after every call completed` `` invariant —
/// a corrupted durable leaf the placement never checkpointed.
#[tokio::test]
async fn binds_the_all_completed_open_batch_invariant() {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let staged_result = |tool_name: &str, text: &str| ToolResultMessage {
        tool_call_id: String::new(),
        tool_name: tool_name.to_owned(),
        content: vec![text_block(text)],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 30,
    };
    let first = staged_result("first", "first result");
    let second = staged_result("second", "second result");
    let fixture = create_fixture(FixtureOptions {
        calls: vec![
            CallSpec {
                name: "first",
                value: None,
            },
            CallSpec {
                name: "second",
                value: None,
            },
        ],
        tools: Some(vec![never_executing_tool(
            "first",
            Arc::clone(&execute_count),
            None,
        )]),
        mode: Some(ToolExecutionMode::Sequential),
        call_states: Some(Box::new(|result_entry_ids: &[String]| {
            vec![
                wire_completed_call(0, &result_entry_ids[0], false),
                wire_completed_call(1, &result_entry_ids[1], false),
            ]
        })),
        extra_writes: Some(Box::new(move |parts: ExtraWriteParts| {
            let entry = |id: &str, message: ToolResultMessage| {
                Write::Entry(Box::new(insert_entry(NewEntry::Message {
                    id: id.to_owned(),
                    parent_id: Some(parts.assistant_entry_id.clone()),
                    body: Box::new(MessageEntry {
                        message: AgentMessage::Standard(Message::ToolResult(message)),
                        terminate: None,
                    }),
                })))
            };
            vec![
                entry(&parts.result_entry_ids[0], first),
                entry(&parts.result_entry_ids[1], second),
                set_value_write(
                    &stored_values::branch_tip("main"),
                    Some(parts.result_entry_ids[1].clone()),
                )
                .expect("the completed tip write"),
            ]
        })),
        ..FixtureOptions::default()
    })
    .await;

    let error = drive_tools(&fixture)
        .await
        .expect_err("the open completed batch rejects");
    assert!(
        error
            .to_string()
            .contains("Tool batch remained open after every call completed"),
        "the open batch carries its invariant: {error}"
    );
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "no effect runs");
    close_session(&fixture).await;
}

/// Port-local boundary: the memo name validation, the `None` memo delete,
/// and the invocation identity's accessors, upstream's `validateMemoName`
/// `TypeError` throws and `invocationCapability`'s identity fields.
#[tokio::test]
async fn binds_the_memo_name_validation_delete_and_capability_identity() {
    let identities: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&identities);
    let memo_tool = tool(
        "memo-tool",
        Arc::new(
            move |_tool_call_id: &str,
                  _args: &JsonValue,
                  _on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
                  _tool_context: ToolContext,
                  invocation: &dyn AgentHarnessToolInvocation,
                  context: &Context| {
                let recorded = Arc::clone(&recorded);
                Box::pin(async move {
                    lock(&recorded).push((
                        invocation.operation_id().to_owned(),
                        invocation.turn_id().to_owned(),
                    ));
                    invocation
                        .set_memo("step/a", Some(json!({ "value": "memo" })), context)
                        .await
                        .expect("the memo writes");
                    let memo = invocation
                        .get_memo("step/a", context)
                        .await
                        .expect("the memo reads");
                    assert_eq!(
                        memo,
                        Some(json!({ "value": "memo" })),
                        "the memo round-trips"
                    );
                    invocation
                        .set_memo("step/a", None, context)
                        .await
                        .expect("the memo deletes");
                    let gone = invocation
                        .get_memo("step/a", context)
                        .await
                        .expect("the memo reads");
                    assert_eq!(gone, None, "the delete clears the memo");
                    let empty = invocation
                        .set_memo("", Some(json!({})), context)
                        .await
                        .expect_err("the empty name rejects");
                    assert_eq!(
                        empty, "Tool invocation memo name must not be empty",
                        "the empty name carries its message"
                    );
                    let colon = invocation
                        .get_memo("bad:name", context)
                        .await
                        .expect_err("the colon name rejects");
                    assert_eq!(
                        colon, "Tool invocation memo name must not contain ':'",
                        "the colon name carries its message"
                    );
                    let colon_set = invocation
                        .set_memo("bad:name", None, context)
                        .await
                        .expect_err("the colon name rejects");
                    assert_eq!(
                        colon_set, "Tool invocation memo name must not contain ':'",
                        "the colon name carries its message"
                    );
                    Ok(done_tool_result())
                })
            },
        ),
        None,
    );
    let fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "memo-tool",
            value: None,
        }],
        tools: Some(vec![memo_tool]),
        mode: Some(ToolExecutionMode::Sequential),
        ..FixtureOptions::default()
    })
    .await;

    drive_tools(&fixture).await.expect("the batch drives");
    assert_eq!(
        lock(&identities).as_slice(),
        &[(fixture.operation_id.clone(), "turn-1".to_owned())],
        "the invocation carries the durable identity"
    );
    let memos = fixture
        .session
        .scan_values(
            &stored_values::operation_tool_memo_prefix(&fixture.operation_id, None).address,
            &background_context(),
        )
        .await
        .expect("the tool memo scan");
    assert!(memos.is_empty(), "the deleted memo leaves no residue");

    close_session(&fixture).await;
}

/// The safe-replay sequential fixture's options, the replay-invariant
/// probes' scaffold: one never-executing `safe` tool wired for the safe
/// replay; returns the execute counter and the options.
fn safe_replay_options() -> (Arc<AtomicUsize>, FixtureOptions) {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let options = FixtureOptions {
        calls: vec![CallSpec {
            name: "safe",
            value: None,
        }],
        tools: Some(vec![never_executing_tool(
            "safe",
            Arc::clone(&execute_count),
            Some(ToolReplay::Safe),
        )]),
        mode: Some(ToolExecutionMode::Sequential),
        call_states: Some(Box::new(|result_entry_ids: &[String]| {
            vec![wire_effect_pending_call(
                0,
                &result_entry_ids[0],
                ToolReplay::Safe,
            )]
        })),
        ..FixtureOptions::default()
    };
    (execute_count, options)
}

/// Port-local boundary: the static tool-context source reaches the tool,
/// upstream's `resolveToolContext` value arm.
#[tokio::test]
async fn binds_the_static_tool_context_source() {
    let seen: Arc<Mutex<Option<JsonValue>>> = Arc::new(Mutex::new(None));
    let recorded = Arc::clone(&seen);
    let context_tool = tool(
        "context-tool",
        Arc::new(
            move |_tool_call_id: &str,
                  _args: &JsonValue,
                  _on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
                  tool_context: ToolContext,
                  _invocation: &dyn AgentHarnessToolInvocation,
                  _context: &Context| {
                let recorded = Arc::clone(&recorded);
                Box::pin(async move {
                    let value = tool_context
                        .as_ref()
                        .and_then(|value| value.downcast_ref::<serde_json::Value>())
                        .cloned();
                    *lock(&recorded) = value;
                    Ok(done_tool_result())
                })
            },
        ),
        None,
    );
    let fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "context-tool",
            value: None,
        }],
        tools: Some(vec![context_tool]),
        mode: Some(ToolExecutionMode::Sequential),
        tool_context: Some(AgentHarnessToolContextSource::Static(Some(Arc::new(
            json!({ "team": "drive" }),
        )))),
        ..FixtureOptions::default()
    })
    .await;

    drive_tools(&fixture).await.expect("the batch drives");
    assert_eq!(
        lock(&seen).as_ref(),
        Some(&json!({ "team": "drive" })),
        "the static context reaches the tool"
    );

    close_session(&fixture).await;
}

/// Port-local boundary: the safe replay's missing-persisted-arguments
/// invariant, upstream's `` `Tool call {resultEntryId} is missing persisted
/// arguments` ``.
#[tokio::test]
async fn binds_the_safe_replays_missing_persisted_arguments_invariant() {
    let (execute_count, options) = safe_replay_options();
    let fixture = create_fixture(options).await;

    let error = drive_tools(&fixture)
        .await
        .expect_err("the missing arguments reject");
    assert!(
        error.to_string().contains(&format!(
            "Tool call {} is missing persisted arguments",
            fixture.result_entry_ids[0]
        )),
        "the missing arguments carry their invariant: {error}"
    );
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "no effect runs");
    close_session(&fixture).await;
}

/// Port-local boundary: the malformed durable tool payloads, upstream's
/// `Stored tool arguments are malformed` and `Pending tool output is
/// malformed` raises.
#[tokio::test]
async fn binds_malformed_durable_tool_payloads() {
    let (execute_count, mut options) = safe_replay_options();
    options = FixtureOptions {
        extra_writes: Some(Box::new(|parts: ExtraWriteParts| {
            vec![malformed_value_write(
                &stored_values::operation_tool_args(&parts.operation_id, "turn-1", 0).address,
                json!("garbage"),
            )]
        })),
        ..options
    };
    let fixture = create_fixture(options).await;
    let error = drive_tools(&fixture)
        .await
        .expect_err("the malformed arguments reject");
    assert!(
        error
            .to_string()
            .contains("Stored tool arguments are malformed"),
        "the malformed arguments carry their error: {error}"
    );
    close_session(&fixture).await;

    let checkpoint_fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "unsafe",
            value: None,
        }],
        tools: Some(vec![never_executing_tool(
            "unsafe",
            Arc::clone(&execute_count),
            None,
        )]),
        mode: Some(ToolExecutionMode::Parallel),
        call_states: Some(Box::new(|result_entry_ids: &[String]| {
            vec![wire_effect_pending_call(
                0,
                &result_entry_ids[0],
                ToolReplay::Never,
            )]
        })),
        extra_writes: Some(Box::new(|parts: ExtraWriteParts| {
            vec![malformed_value_write(
                &stored_values::pending_tool_output(
                    &parts.operation_id,
                    &parts.result_entry_ids[0],
                )
                .address,
                json!("garbage"),
            )]
        })),
        ..FixtureOptions::default()
    })
    .await;
    let error = drive_tools(&checkpoint_fixture)
        .await
        .expect_err("the malformed checkpoint rejects");
    assert!(
        error
            .to_string()
            .contains("Pending tool output is malformed"),
        "the malformed checkpoint carries its error: {error}"
    );
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "no effect runs");
    close_session(&checkpoint_fixture).await;
}

/// Port-local boundary: the interrupted outcome without a staged
/// checkpoint, upstream's `interruptedOutcome` undefined-checkpoint arm —
/// the marker is the result's only content.
#[tokio::test]
async fn binds_the_interrupted_outcome_without_a_checkpoint() {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "unsafe",
            value: None,
        }],
        tools: Some(vec![never_executing_tool(
            "unsafe",
            Arc::clone(&execute_count),
            None,
        )]),
        call_states: Some(Box::new(|result_entry_ids: &[String]| {
            vec![wire_effect_pending_call(
                0,
                &result_entry_ids[0],
                ToolReplay::Never,
            )]
        })),
        ..FixtureOptions::default()
    })
    .await;

    drive_tools(&fixture)
        .await
        .expect("the interrupted batch settles");
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "no effect runs");
    let result = common::tool_result_message(
        &fixture.session,
        &fixture.result_entry_ids[0],
        "the interrupted entry",
    )
    .await;
    assert!(result.is_error, "the interrupted result is an error");
    let wire = serde_json::to_value(result.clone()).expect("the message serializes");
    assert!(
        wire.get("details").is_none(),
        "the checkpointless result omits the details key"
    );
    assert_eq!(
        result.content.len(),
        1,
        "the marker is the only content block"
    );
    match result.content.last() {
        Some(ToolResultBlock::Text(text)) => assert!(
            text.text.contains("external outcome is unknown"),
            "the interruption marker closes the result: {}",
            text.text
        ),
        _ => panic!("the marker is text"),
    }
    close_session(&fixture).await;
}

/// Port-local boundary: a sequential batch recovers its interrupted call,
/// upstream's `runSequential` recovery arm driving
/// `recoverToolInvocation`.
#[tokio::test]
async fn binds_the_sequential_recovery_path() {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "pending",
            value: None,
        }],
        tools: Some(vec![never_executing_tool(
            "pending",
            Arc::clone(&execute_count),
            None,
        )]),
        mode: Some(ToolExecutionMode::Sequential),
        call_states: Some(Box::new(|result_entry_ids: &[String]| {
            vec![wire_effect_pending_call(
                0,
                &result_entry_ids[0],
                ToolReplay::Never,
            )]
        })),
        extra_writes: Some(Box::new(|parts: ExtraWriteParts| {
            vec![
                set_value_write(
                    &stored_values::operation_tool_args(&parts.operation_id, "turn-1", 0),
                    tool_args(&json!({ "value": "pending" })),
                )
                .expect("the persisted args write"),
                set_value_write(
                    &stored_values::pending_tool_output(
                        &parts.operation_id,
                        &parts.result_entry_ids[0],
                    ),
                    ToolOutputPayload {
                        content: vec![text_block("durable partial")],
                        details: json!({ "progress": "kept" }),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    },
                )
                .expect("the staged checkpoint write"),
            ]
        })),
        ..FixtureOptions::default()
    })
    .await;

    drive_tools(&fixture)
        .await
        .expect("the sequential batch recovers");
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "no effect runs");
    let result = common::tool_result_message(
        &fixture.session,
        &fixture.result_entry_ids[0],
        "the interrupted entry",
    )
    .await;
    assert_eq!(
        result.details.as_ref(),
        Some(&json!({ "progress": "kept" })),
        "the staged checkpoint details survive"
    );
    let events = lock(&fixture.events).clone();
    let first = events.first().expect("the first event");
    assert_eq!(first.event_type(), HarnessEventType::TurnStart);
    assert!(first.recovery, "the recovery turn opens");
    assert!(
        !events.iter().any(|event| matches!(
            &event.payload,
            HarnessEventPayload::ToolStart { tool_call_id, .. } if tool_call_id == "call-0"
        )),
        "the recovered call never re-starts"
    );
    let end = events
        .iter()
        .find(|event| {
            matches!(
                &event.payload,
                HarnessEventPayload::ToolEnd { tool_call_id, .. } if tool_call_id == "call-0"
            )
        })
        .expect("the recovered call ends");
    assert!(end.recovery, "the interrupted end is recovery-flagged");

    close_session(&fixture).await;
}

/// Port-local boundary: a parallel batch skips its completed and
/// outcome-ready calls, upstream's `runParallel` skip arm.
#[tokio::test]
async fn binds_the_parallel_skips_of_completed_and_outcome_ready_calls() {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let fixture = create_fixture(FixtureOptions {
        calls: vec![
            CallSpec {
                name: "first",
                value: None,
            },
            CallSpec {
                name: "second",
                value: None,
            },
        ],
        tools: Some(vec![never_executing_tool(
            "second",
            Arc::clone(&execute_count),
            None,
        )]),
        call_states: Some(Box::new(|result_entry_ids: &[String]| {
            vec![
                wire_completed_call(0, &result_entry_ids[0], false),
                wire_planned_call(1, &result_entry_ids[1]),
            ]
        })),
        extra_writes: Some(Box::new(|parts: ExtraWriteParts| {
            let first_result = ToolResultMessage {
                tool_call_id: "call-0".to_owned(),
                tool_name: "first".to_owned(),
                content: vec![text_block("already done")],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: false,
                timestamp: 30,
            };
            vec![
                Write::Entry(Box::new(insert_entry(NewEntry::Message {
                    id: parts.result_entry_ids[0].clone(),
                    parent_id: Some(parts.assistant_entry_id.clone()),
                    body: Box::new(MessageEntry {
                        message: AgentMessage::Standard(Message::ToolResult(first_result)),
                        terminate: None,
                    }),
                }))),
                set_value_write(
                    &stored_values::branch_tip("main"),
                    Some(parts.result_entry_ids[0].clone()),
                )
                .expect("the completed tip write"),
            ]
        })),
        ..FixtureOptions::default()
    })
    .await;

    drive_tools(&fixture).await.expect("the batch drives");
    assert_eq!(
        execute_count.load(Ordering::SeqCst),
        1,
        "only the second runs"
    );
    let transcript = common::transcript(&fixture.lane).await;
    assert_eq!(
        transcript.iter().map(Entry::id).collect::<Vec<_>>(),
        vec![
            fixture.assistant_entry_id.clone(),
            fixture.result_entry_ids[0].clone(),
            fixture.result_entry_ids[1].clone(),
        ],
        "the results materialize in source order"
    );
    let events = lock(&fixture.events).clone();
    assert!(
        !events.iter().any(|event| matches!(
            &event.payload,
            HarnessEventPayload::ToolStart { tool_call_id, .. } if tool_call_id == "call-0"
        )),
        "the completed call never re-starts"
    );
    close_session(&fixture).await;
}

/// Port-local boundary: the `before_tool` block decision, upstream's
/// `applyBeforeToolDecision` immediate outcome carrying the block's reason.
#[tokio::test]
async fn binds_the_before_tool_block_decision() {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let after_tool_ran = Arc::new(AtomicUsize::new(0));
    let fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "guarded",
            value: None,
        }],
        tools: Some(vec![never_executing_tool(
            "guarded",
            Arc::clone(&execute_count),
            None,
        )]),
        mode: Some(ToolExecutionMode::Sequential),
        ..FixtureOptions::default()
    })
    .await;
    let ran = Arc::clone(&after_tool_ran);
    fixture
        .hooks
        .on(
            HookName::AfterTool,
            Arc::new(move |_invocation: &HookInvocation, _context: &Context| {
                let ran = Arc::clone(&ran);
                Box::pin(async move {
                    ran.fetch_add(1, Ordering::SeqCst);
                    Ok(HookResult::AfterTool(None))
                })
            }),
            HookOptions::default(),
        )
        .expect("register after_tool");
    fixture
        .hooks
        .on(
            HookName::BeforeTool,
            Arc::new(|_invocation: &HookInvocation, _context: &Context| {
                Box::pin(async move {
                    Ok(HookResult::BeforeTool(Some(BeforeToolResult {
                        args: None,
                        block: Some(ToolBlock {
                            reason: "policy says no".to_owned(),
                            terminate: None,
                        }),
                    })))
                })
            }),
            HookOptions::default(),
        )
        .expect("register the blocking hook");

    drive_tools(&fixture)
        .await
        .expect("the blocked batch settles");
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "no effect runs");
    assert_eq!(
        after_tool_ran.load(Ordering::SeqCst),
        0,
        "the blocked call never reaches after_tool"
    );
    let result = common::tool_result_message(
        &fixture.session,
        &fixture.result_entry_ids[0],
        "the blocked entry",
    )
    .await;
    assert!(result.is_error, "the blocked result is an error");
    assert_eq!(
        result.content.first(),
        Some(&text_block("policy says no")),
        "the block reason reaches the result"
    );

    close_session(&fixture).await;
}

/// Port-local boundary: a closed hook registry faults the batch through
/// the `before_tool` and `after_tool` hook runs, upstream's closed-registry
/// error propagating through `prepareToolInvocation` and
/// `performToolInvocation` (handler failures restate as tool blocks, so
/// only the closed registry reaches the fault arms).
#[tokio::test]
async fn binds_hook_faults_in_the_tool_pipeline() {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "faulted",
            value: None,
        }],
        tools: Some(vec![never_executing_tool(
            "faulted",
            Arc::clone(&execute_count),
            None,
        )]),
        mode: Some(ToolExecutionMode::Sequential),
        ..FixtureOptions::default()
    })
    .await;
    fixture.hooks.close("closed".to_owned());
    let error = drive_tools(&fixture)
        .await
        .expect_err("the closed registry faults the before_tool run");
    assert!(
        error.to_string().contains("closed"),
        "the before_tool fault propagates: {error}"
    );
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "no effect runs");
    assert!(
        fixture
            .session
            .get_entry(&fixture.result_entry_ids[0], &background_context())
            .await
            .expect("the entry reads")
            .is_none(),
        "the faulted call never stages"
    );
    close_session(&fixture).await;

    let after_fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "faulted",
            value: None,
        }],
        tools: Some(vec![tool(
            "faulted",
            Arc::new(
                |_tool_call_id: &str,
                 _args: &JsonValue,
                 _on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
                 _tool_context: ToolContext,
                 _invocation: &dyn AgentHarnessToolInvocation,
                 _context: &Context| {
                    Box::pin(async move { Ok(done_tool_result()) })
                },
            ),
            None,
        )]),
        mode: Some(ToolExecutionMode::Sequential),
        ..FixtureOptions::default()
    })
    .await;
    after_fixture.hooks.close("closed".to_owned());
    let error = drive_tools(&after_fixture)
        .await
        .expect_err("the closed registry faults the after_tool run");
    assert!(
        error.to_string().contains("closed"),
        "the after_tool fault propagates: {error}"
    );
    assert!(
        after_fixture
            .session
            .get_entry(&after_fixture.result_entry_ids[0], &background_context())
            .await
            .expect("the entry reads")
            .is_none(),
        "the faulted outcome never stages"
    );
    close_session(&after_fixture).await;
}

/// Port-local boundary: durable cancellation arriving between the
/// `before_tool` hook and the tool intent stages the aborted outcome,
/// upstream's `publishToolIntent` returning `cancel_requested`.
#[tokio::test]
async fn binds_the_intent_cancelled_downgrade() {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "cancel-before-intent",
            value: None,
        }],
        tools: Some(vec![never_executing_tool(
            "cancel-before-intent",
            Arc::clone(&execute_count),
            None,
        )]),
        mode: Some(ToolExecutionMode::Sequential),
        ..FixtureOptions::default()
    })
    .await;
    let lane_for_hook = Arc::clone(&fixture.lane);
    let operation_id = fixture.operation_id.clone();
    fixture
        .hooks
        .on(
            HookName::BeforeTool,
            Arc::new(move |_invocation: &HookInvocation, _context: &Context| {
                let lane = Arc::clone(&lane_for_hook);
                let operation_id = operation_id.clone();
                Box::pin(async move {
                    let operation = lane.state().operation.expect("fixture has no operation");
                    let next_run = operation_state_with_scope(
                        &operation.state,
                        OperationScope {
                            control: Control::CancelRequested { requested_at: 40 },
                            ..operation_scope_of(&operation.state)
                        },
                    );
                    lane.command::<(), _>(
                        move |projection, _session, _context| {
                            let next_run = next_run.clone();
                            let operation_id = operation_id.clone();
                            let meta = operation.meta.clone();
                            Box::pin(async move {
                                let next = LaneState {
                                    operation: Some(LiveOperation {
                                        meta: meta.clone(),
                                        state: next_run.clone(),
                                    }),
                                    ..projection.clone()
                                };
                                Ok(LaneCommand::Commit {
                                    writes: vec![
                                        set_value_write(
                                            &stored_values::operation_state(&operation_id),
                                            next_run.clone(),
                                        )
                                        .expect("cancel write"),
                                    ],
                                    next,
                                    materialize: Arc::new(|_| ()),
                                    events: None,
                                })
                            })
                        },
                        &background_context(),
                    )
                    .await
                    .expect("the cancel commits");
                    Ok(HookResult::BeforeTool(None))
                })
            }),
            HookOptions::default(),
        )
        .expect("the cancelling hook registers");

    drive_tools(&fixture)
        .await
        .expect("the cancelled batch settles");
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "no effect runs");
    let result = common::tool_result_message(
        &fixture.session,
        &fixture.result_entry_ids[0],
        "the aborted entry",
    )
    .await;
    assert!(result.is_error, "the aborted result is an error");
    assert_eq!(
        result.content.first(),
        Some(&text_block(
            "Tool execution was cancelled before completion."
        )),
        "the aborted text"
    );
    close_session(&fixture).await;
}

/// Port-local boundary: the gate aborting between the tool intent and the
/// execution turns the effect cancelled, upstream's `gate.admit` throwing
/// `AbortRequested` inside `performToolInvocation`.
#[tokio::test]
async fn binds_the_execute_abort_rejection_paths() {
    let execute_count = Arc::new(AtomicUsize::new(0));
    let fixture = create_fixture(FixtureOptions {
        calls: vec![CallSpec {
            name: "cancel-before-admission",
            value: None,
        }],
        tools: Some(vec![never_executing_tool(
            "cancel-before-admission",
            Arc::clone(&execute_count),
            None,
        )]),
        mode: Some(ToolExecutionMode::Sequential),
        ..FixtureOptions::default()
    })
    .await;
    let drive_for_hook = Arc::clone(&fixture.drive);
    fixture
        .hooks
        .on(
            HookName::BeforeTool,
            Arc::new(move |_invocation: &HookInvocation, _context: &Context| {
                let drive = Arc::clone(&drive_for_hook);
                Box::pin(async move {
                    let (_, cancel_rx) = tokio::sync::watch::channel(());
                    drive.begin_abort(cancel_rx);
                    drive.signal_abort();
                    Ok(HookResult::BeforeTool(None))
                })
            }),
            HookOptions::default(),
        )
        .expect("the aborting hook registers");

    drive_tools(&fixture)
        .await
        .expect("the aborted batch settles");
    assert_eq!(execute_count.load(Ordering::SeqCst), 0, "no effect runs");
    let result = common::tool_result_message(
        &fixture.session,
        &fixture.result_entry_ids[0],
        "the aborted entry",
    )
    .await;
    assert!(result.is_error, "the aborted result is an error");
    assert_eq!(
        result.content.first(),
        Some(&text_block(
            "Tool execution was cancelled before completion."
        )),
        "the aborted text"
    );
    close_session(&fixture).await;
}

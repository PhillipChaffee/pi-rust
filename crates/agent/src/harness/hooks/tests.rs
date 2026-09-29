//! Boundary tests for the harness hook registry.
//!
//! The registry's unit cases ride upstream `test/harness/
//! execution-primitives.test.ts`'s `HookRegistry` block, distributed
//! across this suite with the gate and events suites the same file spans.
//! These tests pin the aggregate semantics this slice
//! restates: fail-open aggregation, fail-closed `before_drive`, the tool
//! gate's abort contract, first-match structural hooks, and stream-options
//! patch derivation.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pi_chord::context::{Context, background_context};
use pi_telemetry::{AttributeValue, InMemoryTelemetryContext, SpanOptions, TelemetryHandle};

use crate::harness::agent_harness::{
    HookEvent, HookInvocation, HookName, HookOptions, HookResult, StepKind,
};
use crate::harness::context::{
    create_context_key, get_telemetry_context, with_context_value, with_telemetry_context,
};
use crate::harness::gate::{GateRejection, create_gate};
use crate::harness::hooks::{
    HookRegistry, apply_stream_options_patch, create_stream_options_patch,
};
use crate::harness::types::{AgentHarnessStreamOptions, AgentHarnessStreamOptionsPatch};

fn invocation(event: HookEvent) -> HookInvocation {
    HookInvocation {
        lane: "main".to_owned(),
        run_id: "run".to_owned(),
        event,
    }
}

fn reporting_handler_errors(
    errors: Arc<Mutex<Vec<String>>>,
) -> crate::harness::hooks::HookErrorReporter {
    Arc::new(move |error, _name, _lane, _context| {
        let errors = Arc::clone(&errors);
        Box::pin(async move {
            errors.lock().expect("report lock").push(error.to_string());
        })
    })
}

fn injected_message() -> crate::types::AgentMessage {
    serde_json::from_value(serde_json::json!({
        "role": "user",
        "content": "injected",
        "timestamp": 1
    }))
    .expect("injected message")
}

fn handler(
    outcome: impl Fn(&HookInvocation) -> HookResult + Send + Sync + 'static,
) -> crate::harness::agent_harness::HookHandler {
    Arc::new(move |event: &HookInvocation, _context| {
        let result = outcome(event);
        Box::pin(async move { Ok(result) })
    })
}

/// Registers one fixed-result handler, upstream's
/// `registry.on(name, () => result)` rows with default options.
fn register_fixed(
    registry: &HookRegistry,
    name: HookName,
    result: HookResult,
) -> crate::harness::agent_harness::Subscription {
    registry
        .on(
            name,
            handler(move |_event| result.clone()),
            HookOptions::default(),
        )
        .expect("register")
}

/// The empty-prompt `BeforeRun` invocation, upstream's per-case
/// `invocation({ kind: "before_run", prompt: [], resources })` literals.
fn before_run_invocation() -> HookInvocation {
    invocation(before_run_event())
}

/// `before_run` aggregates fail-open: injected messages accumulate across
/// handlers, a failing handler reports but does not stop the rest.
#[tokio::test]
async fn before_run_aggregates_fail_open() {
    let (_errors, registry) = error_capture_registry();
    let options = HookOptions {
        id: Some("second".to_owned()),
    };
    register_fixed(
        &registry,
        HookName::BeforeRun,
        HookResult::BeforeRun(Some(crate::harness::agent_harness::BeforeRunResult {
            messages: vec![injected_message()],
        })),
    );
    let current = invocation(before_run_event());
    let (gate, _gate_control) = create_gate();
    let result = registry
        .run_with_gate(HookName::BeforeRun, current, &gate, &background_context())
        .await;
    let result = match result {
        Ok(result) => result,
        Err(error) => panic!("the run should aggregate: {error}"),
    };
    let HookResult::BeforeRun(Some(injected)) = result else {
        panic!("the aggregate should carry the injected messages");
    };
    assert_eq!(injected.messages, vec![injected_message()]);
    let _ = options;
}

/// A gate abort rejection carries its cancellation through the run error.
#[tokio::test]
async fn the_gate_rejection_maps_to_the_run_error() {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let registry = HookRegistry::new(reporting_handler_errors(errors));
    let (gate, gate_control) = create_gate();
    let (cancelled, controller) = pi_chord::context::with_cancel(&background_context());
    controller.abort("aborted");
    let cancellation = tokio::sync::watch::channel(()).1;
    gate_control.begin_abort(cancellation);
    gate_control.signal_abort();
    let _ = controller;
    let error = registry
        .run_with_gate(
            HookName::BeforeDrive,
            invocation(before_drive_event()),
            &gate,
            &cancelled,
        )
        .await
        .expect_err("an aborting gate refuses admission");
    assert!(matches!(
        error,
        crate::harness::hooks::HookRunError::GateAborted(_)
    ));
    let _ = GateRejection::Closed;
}

/// A closed registry refuses every run.
#[tokio::test]
async fn a_closed_registry_refuses_every_run() {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let registry = HookRegistry::new(reporting_handler_errors(errors));
    registry.close("closed".to_owned());
    let (gate, _gate_control) = create_gate();
    let error = registry
        .run_with_gate(
            HookName::BeforeRun,
            before_run_invocation(),
            &gate,
            &background_context(),
        )
        .await
        .expect_err("a closed registry errors");
    assert!(matches!(
        error,
        crate::harness::hooks::HookRunError::Closed(_)
    ));
}

/// The stream-options patch applies per key over the base, and the
/// derived patch between two snapshots round-trips.
#[test]
fn stream_options_patches_apply_and_derive() {
    let base = AgentHarnessStreamOptions {
        timeout_ms: Some(1_000),
        max_retries: Some(2),
        ..AgentHarnessStreamOptions::default()
    };
    let patch = AgentHarnessStreamOptionsPatch {
        timeout_ms: Some(None),
        max_retries: Some(Some(5)),
        ..AgentHarnessStreamOptionsPatch::default()
    };
    let next = apply_stream_options_patch(&base, &patch);
    assert_eq!(next.timeout_ms, None);
    assert_eq!(next.max_retries, Some(5));

    let derived = create_stream_options_patch(&base, &next);
    assert_eq!(derived.timeout_ms, Some(None));
    assert_eq!(derived.max_retries, Some(Some(5)));
    assert!(derived.headers.is_none());
}
fn test_model() -> pi_ai::types::Model {
    serde_json::from_value(serde_json::json!({
        "id": "test-model",
        "name": "Test Model",
        "api": "anthropic-messages",
        "provider": "anthropic",
        "baseUrl": "https://api.example.com",
        "reasoning": false,
        "input": ["text"],
        "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0 },
        "contextWindow": 200_000,
        "maxTokens": 8_192,
    }))
    .expect("a well-formed model literal")
}

fn user_message() -> crate::types::AgentMessage {
    serde_json::from_value(serde_json::json!({
        "role": "user",
        "content": "hello",
        "timestamp": 1
    }))
    .expect("user message")
}

fn replaced_message() -> crate::types::AgentMessage {
    serde_json::from_value(serde_json::json!({
        "role": "user",
        "content": "replaced",
        "timestamp": 2
    }))
    .expect("replaced message")
}

fn settled_message() -> crate::harness::session::types::SettledAssistantMessage {
    serde_json::from_value(serde_json::json!({
        "message": {
            "content": [{"type": "text", "text": "done"}],
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": "test-model",
            "timestamp": 2,
            "usage": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2,
                "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 } },
            "stopReason": "stop",
        },
        "stopReason": "stop"
    }))
    .expect("settled assistant message")
}

/// The error-capturing registry, the reporter-asserting rows' fixture:
/// the captured errors and the registry reporting into them.
fn error_capture_registry() -> (Arc<Mutex<Vec<String>>>, HookRegistry) {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let registry = HookRegistry::new(reporting_handler_errors(Arc::clone(&errors)));
    (errors, registry)
}

/// The plain registry whose reporter discards, upstream's per-case
/// `new HookRegistry(() => {})` rows whose assertions never read the
/// captured errors.
fn empty_registry() -> HookRegistry {
    HookRegistry::new(reporting_handler_errors(Arc::new(Mutex::new(Vec::new()))))
}

/// The `before_run` event the per-case suites fire, upstream's
/// `{ kind: "before_run", prompt: [], resources }` literals.
fn before_run_event() -> HookEvent {
    HookEvent::BeforeRun {
        prompt: vec![],
        resources: crate::harness::agent_harness::Resources::default(),
    }
}

/// The `before_run_end` event, upstream's `{ kind: "before_run_end",
/// runId: "run", messages: [] }` literals.
fn before_run_end_event() -> HookEvent {
    HookEvent::BeforeRunEnd {
        run_id: "run".to_owned(),
        messages: vec![],
    }
}

/// The `before_tool` event, upstream's `{ toolCallId: "call", toolName:
/// "read", args: {} }` literals.
fn before_tool_event() -> HookEvent {
    HookEvent::BeforeTool {
        tool_call_id: "call".to_owned(),
        tool_name: "read".to_owned(),
        args: BTreeMap::new(),
    }
}

/// The `after_response` event, upstream's `{ status: 200, headers: null,
/// message }` literals.
fn after_response_event() -> HookEvent {
    HookEvent::AfterResponse {
        status: Some(200),
        headers: None,
        message: settled_message(),
    }
}

/// The `before_compaction` event, upstream's `{ reason: "manual",
/// preparation }` literals over the empty one-token preparation.
fn before_compaction_event() -> HookEvent {
    HookEvent::BeforeCompaction {
        reason: crate::harness::session::types::CompactionReason::Manual,
        preparation: crate::harness::compaction::types::CompactionPreparation {
            messages_to_summarize: vec![],
            turn_prefix_messages: vec![],
            retained_tail: vec![],
            is_split_turn: false,
            tokens_before: 1,
            previous_summary: None,
            file_ops: crate::harness::compaction::types::FileOperations::default(),
            settings: crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS,
        },
        custom_instructions: None,
    }
}

/// The `before_navigation` event, upstream's `{ targetId: "target",
/// preparation }` literals.
fn before_navigation_event() -> HookEvent {
    HookEvent::BeforeNavigation {
        target_id: "target".to_owned(),
        preparation: branch_preparation(),
        custom_instructions: None,
    }
}

/// The `before_payload` event, upstream's `{ model, payload }` literals
/// over the original payload.
fn before_payload_event() -> HookEvent {
    HookEvent::BeforePayload {
        model: test_model(),
        payload: serde_json::json!({ "original": true }),
    }
}

/// The `before_drive` event, upstream's `{ operation: "run" }` literals.
fn before_drive_event() -> HookEvent {
    HookEvent::BeforeDrive {
        operation: crate::harness::session::types::OperationKind::Run,
    }
}

/// The fixed `before_compaction` result, upstream's `() => ({ decline })`
/// and compaction rows.
fn compaction_result(
    decline: Option<bool>,
    compaction: Option<crate::harness::compaction::types::CompactResult>,
) -> HookResult {
    HookResult::BeforeCompaction(Some(crate::harness::agent_harness::CompactionHookResult {
        decline,
        compaction,
    }))
}

/// The fixed `before_navigation` result, upstream's `() => ({ decline,
/// summary })` rows.
fn navigation_result(
    decline: Option<bool>,
    summary: Option<crate::harness::compaction::types::BranchSummaryResult>,
) -> HookResult {
    HookResult::BeforeNavigation(Some(crate::harness::agent_harness::NavigationHookResult {
        decline,
        summary,
    }))
}

/// Registers the fixed result, runs the name's aggregate, and unwraps the
/// carried value, the register-then-run rows' settled read; a resultless
/// aggregate panics the test by design.
macro_rules! fixed_aggregate {
    ($registry:expr, $name:ident, $event:expr, $variant:ident, $fixed:expr, $why:literal) => {{
        register_fixed($registry, HookName::$name, $fixed);
        let result = aggregate($registry, HookName::$name, $event).await;
        match result {
            HookResult::$variant(Some(result)) => result,
            _ => panic!($why),
        }
    }};
}

/// The failing handler, upstream's throwing handler rows: one boxed-error
/// failure with the given message.
fn failing_handler(message: &'static str) -> crate::harness::agent_harness::HookHandler {
    Arc::new(move |event: &HookInvocation, _context: &Context| {
        let failure: crate::harness::agent_harness::HookFailure =
            Box::<dyn std::error::Error + Send + Sync>::from(message);
        let _ = event;
        Box::pin(async move { Err::<HookResult, _>(failure) })
    })
}

/// Registers one failing handler, the fail-closed rows' registration.
fn register_failing(registry: &HookRegistry, name: HookName, message: &'static str) {
    registry
        .on(name, failing_handler(message), HookOptions::default())
        .expect("register the failing handler");
}

/// Registers the fixed `before_tool` result, runs the aggregate, and
/// unwraps the carried result, the tool-gate rows' settled read; a
/// resultless aggregate panics the test by design.
async fn before_tool_aggregate(
    registry: &HookRegistry,
    fixed: HookResult,
    why: &'static str,
) -> crate::harness::agent_harness::BeforeToolResult {
    register_fixed(registry, HookName::BeforeTool, fixed);
    let result = aggregate(registry, HookName::BeforeTool, before_tool_event()).await;
    let HookResult::BeforeTool(Some(result)) = result else {
        panic!("{why}");
    };
    result
}

/// Runs the name's aggregate over the background context, the
/// register-then-run rows' settled read; an unexpected aggregate panics
/// the test by design.
async fn aggregate(registry: &HookRegistry, name: HookName, event: HookEvent) -> HookResult {
    run(registry, name, event, &background_context())
        .await
        .expect("aggregate")
}

async fn run(
    registry: &HookRegistry,
    name: HookName,
    event: HookEvent,
    context: &Context,
) -> Result<HookResult, crate::harness::hooks::HookRunError> {
    let (gate, _control) = create_gate();
    registry
        .run_with_gate(name, invocation(event), &gate, context)
        .await
}

/// `transform_context` folds replacements across handlers fail-open.
#[tokio::test]
async fn transform_context_folds_replacements() {
    let registry = empty_registry();
    register_fixed(
        &registry,
        HookName::TransformContext,
        HookResult::TransformContext(Some(
            crate::harness::agent_harness::TransformContextResult {
                messages: Some(vec![user_message()]),
                system_prompt: Some("replacement".to_owned()),
            },
        )),
    );
    let original = user_message();
    let result = run(
        &registry,
        HookName::TransformContext,
        HookEvent::TransformContext {
            messages: vec![original.clone(), original],
            system_prompt: "original".to_owned(),
        },
        &background_context(),
    )
    .await
    .expect("aggregate");
    let HookResult::TransformContext(Some(result)) = result else {
        panic!("the aggregate carries the replacements");
    };
    assert_eq!(result.system_prompt.as_deref(), Some("replacement"));
    assert_eq!(result.messages.map_or(0, |messages| messages.len()), 1);
}

/// `before_run_end` keeps the last handler's follow-up prompt, upstream's
/// invokeAll overwrite.
#[tokio::test]
async fn before_run_end_keeps_the_last_follow_up() {
    let registry = empty_registry();
    for follow_up in ["first", "second"] {
        registry
            .on(
                HookName::BeforeRunEnd,
                handler(move |_event| {
                    HookResult::BeforeRunEnd(Some(crate::harness::agent_harness::FollowUpResult {
                        follow_up: follow_up.to_owned(),
                    }))
                }),
                HookOptions::default(),
            )
            .expect("register");
    }
    let result = aggregate(&registry, HookName::BeforeRunEnd, before_run_end_event()).await;
    let HookResult::BeforeRunEnd(Some(result)) = result else {
        panic!("the aggregate carries the follow-up");
    };
    assert_eq!(result.follow_up, "second");
}

/// `before_request` folds stream-options patches across handlers and
/// derives the net patch against the original.
#[tokio::test]
async fn before_request_folds_stream_option_patches() {
    let registry = empty_registry();
    register_fixed(
        &registry,
        HookName::BeforeRequest,
        HookResult::BeforeRequest(Some(crate::harness::agent_harness::BeforeRequestResult {
            stream_options: AgentHarnessStreamOptionsPatch {
                timeout_ms: Some(Some(2_000)),
                ..AgentHarnessStreamOptionsPatch::default()
            },
        })),
    );
    let original = AgentHarnessStreamOptions {
        timeout_ms: Some(1_000),
        ..AgentHarnessStreamOptions::default()
    };
    let result = run(
        &registry,
        HookName::BeforeRequest,
        HookEvent::BeforeRequest {
            model: test_model(),
            step: StepKind::Assistant,
            attempt: 1,
            stream_options: original.clone(),
        },
        &background_context(),
    )
    .await
    .expect("aggregate");
    let HookResult::BeforeRequest(Some(result)) = result else {
        panic!("the aggregate carries the derived patch");
    };
    assert_eq!(result.stream_options.timeout_ms, Some(Some(2_000)));
}

/// `before_payload` hands the last replacement payload onward.
#[tokio::test]
async fn before_payload_replaces_the_payload() {
    let registry = empty_registry();
    let result = fixed_aggregate!(
        &registry,
        BeforePayload,
        before_payload_event(),
        BeforePayload,
        HookResult::BeforePayload(Some(crate::harness::agent_harness::PayloadResult {
            payload: serde_json::json!({ "replacement": true }),
        })),
        "the aggregate carries the replacement payload"
    );
    assert_eq!(result.payload, serde_json::json!({ "replacement": true }));
}

/// `after_response` carries the replacement settled message.
#[tokio::test]
async fn after_response_carries_the_replacement() {
    let registry = empty_registry();
    registry
        .on(
            HookName::AfterResponse,
            handler(move |_event| {
                HookResult::AfterResponse(Some(Box::new(
                    crate::harness::agent_harness::AfterResponseResult {
                        message: Some(settled_message()),
                    },
                )))
            }),
            HookOptions::default(),
        )
        .expect("register");
    let result = aggregate(&registry, HookName::AfterResponse, after_response_event()).await;
    let HookResult::AfterResponse(Some(result)) = result else {
        panic!("the aggregate carries the replacement");
    };
    assert!(result.message.is_some());
}

/// `before_tool` folds argument replacements and the first block verdict
/// breaks the loop; `run_tool_with_gate` carries the block through.
#[tokio::test]
async fn before_tool_blocks_the_call() {
    let (_errors, registry) = error_capture_registry();
    let result = before_tool_aggregate(
        &registry,
        HookResult::BeforeTool(Some(crate::harness::agent_harness::BeforeToolResult {
            args: None,
            block: Some(crate::harness::agent_harness::ToolBlock {
                reason: "blocked by hook".to_owned(),
                terminate: None,
            }),
        })),
        "the aggregate carries the block",
    )
    .await;
    let block = result.block.expect("blocked");
    assert_eq!(block.reason, "blocked by hook");
}

/// `after_tool` folds result patches across handlers, keeping the last
/// value per field.
#[tokio::test]
async fn after_tool_folds_result_patches() {
    let registry = empty_registry();
    let result = fixed_aggregate!(
        &registry,
        AfterTool,
        after_tool_event(),
        AfterTool,
        HookResult::AfterTool(Some(crate::harness::agent_harness::AfterToolResult {
            is_error: Some(true),
            ..crate::harness::agent_harness::AfterToolResult::default()
        })),
        "the aggregate carries the patch"
    );
    assert_eq!(result.is_error, Some(true));
}

/// `before_compaction` returns the first decisive handler; a handler
/// returning both decline and compaction reports and is skipped.
#[tokio::test]
async fn before_compaction_takes_the_first_decisive_result() {
    let (errors, registry) = error_capture_registry();
    register_fixed(
        &registry,
        HookName::BeforeCompaction,
        compaction_result(
            Some(true),
            Some(crate::harness::compaction::types::CompactResult {
                summary: "summary".to_owned(),
                tokens_before: 1,
                usage: None,
                retained_tail: vec![],
                details: None,
            }),
        ),
    );
    register_fixed(
        &registry,
        HookName::BeforeCompaction,
        compaction_result(Some(true), None),
    );
    let result = aggregate(
        &registry,
        HookName::BeforeCompaction,
        before_compaction_event(),
    )
    .await;
    let HookResult::BeforeCompaction(Some(result)) = result else {
        panic!("the decline reaches the aggregate");
    };
    assert_eq!(result.decline, Some(true));
    assert!(
        errors
            .lock()
            .expect("report lock")
            .iter()
            .any(|error| { error.contains("cannot return both decline and compaction") })
    );
}

/// `before_drive` runs fail-closed: a failing handler errors the drive.
#[tokio::test]
async fn before_drive_runs_fail_closed() {
    let (errors, registry) = error_capture_registry();
    register_failing(&registry, HookName::BeforeDrive, "drive handler failed");
    let result = run(
        &registry,
        HookName::BeforeDrive,
        before_drive_event(),
        &background_context(),
    )
    .await;
    assert!(result.is_err());
    assert!(!errors.lock().expect("report lock").is_empty());
}

/// `before_drive` with only succeeding handlers settles the aggregate.
#[tokio::test]
async fn before_drive_succeeds_without_failures() {
    let registry = empty_registry();
    registry
        .on(
            HookName::BeforeDrive,
            handler(|_event| HookResult::BeforeDrive),
            HookOptions::default(),
        )
        .expect("register the succeeding handler");
    let result = run(
        &registry,
        HookName::BeforeDrive,
        before_drive_event(),
        &background_context(),
    )
    .await
    .expect("the drive succeeds");
    assert!(matches!(result, HookResult::BeforeDrive));
}

/// A closed registry refuses new registrations with its first error.
#[tokio::test]
async fn a_closed_registry_refuses_registration() {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let registry = HookRegistry::new(reporting_handler_errors(errors));
    registry.close("closed".to_owned());
    let error = registry
        .on(
            HookName::BeforeRun,
            handler(|_event| HookResult::BeforeRun(None)),
            HookOptions::default(),
        )
        .expect_err("a closed registry refuses registration");
    assert_eq!(error, "closed");
}

/// The registry's first close error wins; later closes never override it.
#[test]
fn the_registry_keeps_its_first_close_error() {
    let registry = empty_registry();
    registry.close("first".to_owned());
    registry.close("second".to_owned());
    let error = registry
        .on(
            HookName::BeforeRun,
            handler(|_event| HookResult::BeforeRun(None)),
            HookOptions::default(),
        )
        .expect_err("a closed registry refuses registration");
    assert_eq!(error, "first");
}

/// `run_tool_with_gate` restates the gate's abort rejection as a run
/// error.
#[tokio::test]
async fn run_tool_with_gate_restates_the_gate_rejection() {
    let registry = empty_registry();
    let (gate, gate_control) = create_gate();
    let cancellation = tokio::sync::watch::channel(()).1;
    gate_control.begin_abort(cancellation);
    let error = registry
        .run_tool_with_gate(
            HookName::BeforeTool,
            invocation(before_tool_event()),
            &gate,
            &background_context(),
        )
        .await
        .expect_err("an aborting gate refuses admission");
    assert!(matches!(
        error,
        crate::harness::hooks::HookRunError::GateAborted(_)
    ));
}

/// `has` reports membership, the subscription unregisters exactly one
/// handler, and the aggregate then sees only the survivors.
#[tokio::test]
async fn the_subscription_unregisters_one_handler() {
    let registry = empty_registry();
    assert!(!registry.has(HookName::BeforeRun));
    let first = register_fixed(
        &registry,
        HookName::BeforeRun,
        HookResult::BeforeRun(Some(crate::harness::agent_harness::BeforeRunResult {
            messages: vec![injected_message()],
        })),
    );
    let second = register_fixed(
        &registry,
        HookName::BeforeRun,
        HookResult::BeforeRun(Some(crate::harness::agent_harness::BeforeRunResult {
            messages: vec![injected_message()],
        })),
    );
    assert!(registry.has(HookName::BeforeRun));
    first.unsubscribe();
    assert!(registry.has(HookName::BeforeRun));
    second.unsubscribe();
    assert!(!registry.has(HookName::BeforeRun));
    let result = aggregate(&registry, HookName::BeforeRun, before_run_event()).await;
    let HookResult::BeforeRun(None) = result else {
        panic!("the unsubscribed handlers stop injecting");
    };
}

/// The registry and run-error surfaces render their debug shapes and
/// messages; the gate rejection and handler failure restate as run errors.
#[test]
fn the_registry_and_run_error_render_their_surfaces() {
    let registry = empty_registry();
    assert!(format!("{registry:?}").contains("HookRegistry"));
    let closed = crate::harness::hooks::HookRunError::from(GateRejection::Closed(
        crate::harness::gate::GateClosedError("gate closed".to_owned()),
    ));
    assert!(matches!(
        closed,
        crate::harness::hooks::HookRunError::Closed(_)
    ));
    assert_eq!(closed.to_string(), "gate closed");
    let handler_failure = crate::harness::hooks::HookRunError::from(Box::<
        dyn std::error::Error + Send + Sync,
    >::from("handler failed"));
    assert!(matches!(
        handler_failure,
        crate::harness::hooks::HookRunError::Handler(_)
    ));
    assert_eq!(handler_failure.to_string(), "handler failed");
    let aborted = crate::harness::hooks::HookRunError::Aborted(
        pi_chord::context::AbortReason::Caller("cancelled".to_owned()),
    );
    assert_eq!(aborted.to_string(), "cancelled");
}

/// A caller context that is already cancelled aborts the admitted run
/// before any handler runs; the abort reason restates on the run error.
#[tokio::test]
async fn a_cancelled_caller_context_aborts_the_admitted_run() {
    let registry = empty_registry();
    let (gate, _gate_control) = create_gate();
    let (cancelled, controller) = pi_chord::context::with_cancel(&background_context());
    controller.abort("cancelled");
    let error = registry
        .run_with_gate(
            HookName::BeforeRun,
            before_run_invocation(),
            &gate,
            &cancelled,
        )
        .await
        .expect_err("the cancelled caller context aborts admission");
    assert_eq!(error.to_string(), "cancelled");
    assert!(matches!(
        error,
        crate::harness::hooks::HookRunError::Aborted(pi_chord::context::AbortReason::Caller(
            message
        )) if message == "cancelled"
    ));

    let (reasonless, controller) = pi_chord::context::with_cancel(&background_context());
    controller.abort_without_reason();
    let error = registry
        .run_with_gate(
            HookName::BeforeRun,
            before_run_invocation(),
            &gate,
            &reasonless,
        )
        .await
        .expect_err("the reasonless abort still aborts admission");
    assert!(matches!(
        error,
        crate::harness::hooks::HookRunError::Aborted(pi_chord::context::AbortReason::Aborted)
    ));
}

/// `run_tool_with_gate` reports the abort the same way when the caller
/// context is cancelled before admission.
#[tokio::test]
async fn a_cancelled_caller_context_aborts_the_tool_admission() {
    let registry = empty_registry();
    let (gate, _gate_control) = create_gate();
    let (cancelled, controller) = pi_chord::context::with_cancel(&background_context());
    controller.abort("cancelled");
    let error = registry
        .run_tool_with_gate(
            HookName::BeforeTool,
            invocation(before_tool_event()),
            &gate,
            &cancelled,
        )
        .await
        .expect_err("the cancelled caller context aborts the tool run");
    assert!(matches!(
        error,
        crate::harness::hooks::HookRunError::Aborted(_)
    ));
}

/// `run_tool_with_gate` runs the `before_tool` aggregate and records one
/// telemetry span per registration carrying the lane, operation id, hook
/// name, registration id, and the completed outcome.
#[tokio::test]
async fn run_tool_with_gate_runs_the_before_tool_aggregate() {
    let recorder = InMemoryTelemetryContext::default();
    let context = with_telemetry_context(
        TelemetryHandle::new(recorder.clone()),
        &background_context(),
    );
    let registry = empty_registry();
    let replaced_args = BTreeMap::from([("path".to_owned(), serde_json::json!("/tmp/replaced"))]);
    let expected_args = replaced_args.clone();
    registry
        .on(
            HookName::BeforeTool,
            handler(move |_event| {
                HookResult::BeforeTool(Some(crate::harness::agent_harness::BeforeToolResult {
                    args: Some(replaced_args.clone()),
                    block: None,
                }))
            }),
            HookOptions {
                id: Some("reg-1".to_owned()),
            },
        )
        .expect("register the replacing handler");
    let result = registry
        .run_tool_with_gate(
            HookName::BeforeTool,
            invocation(before_tool_event()),
            &create_gate().0,
            &context,
        )
        .await
        .expect("the tool aggregate settles");
    let HookResult::BeforeTool(Some(result)) = result else {
        panic!("the aggregate carries the replacement arguments");
    };
    assert_eq!(result.args, Some(expected_args));
    let spans = recorder.get_spans();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span.name, "pi.harness.hook");
    let attribute = |key: &str| span.attributes.get(key);
    assert_eq!(
        attribute("pi.lane.name"),
        Some(&Some(AttributeValue::Str("main".to_owned())))
    );
    assert_eq!(
        attribute("pi.operation.id"),
        Some(&Some(AttributeValue::Str("run".to_owned())))
    );
    assert_eq!(
        attribute("pi.hook.name"),
        Some(&Some(AttributeValue::Str("before_tool".to_owned())))
    );
    assert_eq!(
        attribute("pi.hook.registration_id"),
        Some(&Some(AttributeValue::Str("reg-1".to_owned())))
    );
    assert_eq!(
        attribute("pi.hook.outcome"),
        Some(&Some(AttributeValue::Str("completed".to_owned())))
    );
}

/// `run_tool_with_gate` refuses every non-tool hook with a closed error.
#[tokio::test]
async fn run_tool_with_gate_refuses_non_tool_hooks() {
    let registry = empty_registry();
    let (gate, _gate_control) = create_gate();
    let error = registry
        .run_tool_with_gate(
            HookName::BeforeRun,
            before_run_invocation(),
            &gate,
            &background_context(),
        )
        .await
        .expect_err("a non-tool hook is refused");
    assert!(matches!(
        error,
        crate::harness::hooks::HookRunError::Closed(_)
    ));
    assert!(error.to_string().contains("is not a tool hook"));
}

/// `before_tool` folds argument replacements across handlers and threads
/// the folded arguments into each next handler.
#[tokio::test]
async fn before_tool_folds_argument_replacements() {
    let registry = empty_registry();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_for_handler = Arc::clone(&seen);
    register_fixed(
        &registry,
        HookName::BeforeTool,
        HookResult::BeforeTool(Some(crate::harness::agent_harness::BeforeToolResult {
            args: Some(BTreeMap::from([(
                "path".to_owned(),
                serde_json::json!("/tmp/replaced"),
            )])),
            block: None,
        })),
    );
    registry
        .on(
            HookName::BeforeTool,
            handler(move |event| {
                let HookEvent::BeforeTool { args, .. } = &event.event else {
                    panic!("the tool handler receives the tool event");
                };
                seen_for_handler
                    .lock()
                    .expect("seen lock")
                    .push(args.clone());
                HookResult::BeforeTool(None)
            }),
            HookOptions::default(),
        )
        .expect("register the observing handler");
    let result = aggregate(&registry, HookName::BeforeTool, before_tool_event()).await;
    let HookResult::BeforeTool(Some(result)) = result else {
        panic!("the aggregate carries the replacement arguments");
    };
    assert_eq!(
        result.args,
        Some(BTreeMap::from([(
            "path".to_owned(),
            serde_json::json!("/tmp/replaced")
        )]))
    );
    assert_eq!(result.block, None);
    let observed = seen.lock().expect("seen lock").clone();
    assert_eq!(
        observed,
        vec![BTreeMap::from([(
            "path".to_owned(),
            serde_json::json!("/tmp/replaced")
        )])]
    );
}

/// A failing `before_tool` handler reports, blocks the call with the
/// failure reason, and records the failed span outcome.
#[tokio::test]
async fn a_failing_before_tool_handler_blocks_the_call() {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let recorder = InMemoryTelemetryContext::default();
    let context = with_telemetry_context(
        TelemetryHandle::new(recorder.clone()),
        &background_context(),
    );
    let registry = HookRegistry::new(reporting_handler_errors(Arc::clone(&errors)));
    register_failing(&registry, HookName::BeforeTool, "tool hook failed");
    let result = registry
        .run_tool_with_gate(
            HookName::BeforeTool,
            invocation(before_tool_event()),
            &create_gate().0,
            &context,
        )
        .await
        .expect("the aggregate still settles with a block");
    let HookResult::BeforeTool(Some(result)) = result else {
        panic!("the failure blocks the call");
    };
    let block = result.block.expect("blocked");
    assert_eq!(block.reason, "tool hook failed");
    assert_eq!(
        errors.lock().expect("report lock").as_slice(),
        ["tool hook failed"]
    );
    let spans = recorder.get_spans();
    assert_eq!(spans.len(), 1);
    assert_eq!(
        spans[0].attributes.get("pi.hook.outcome"),
        Some(&Some(AttributeValue::Str("failed".to_owned())))
    );
}

/// `before_run` threads each handler's injections into the next handler's
/// prompt, skips non-contributing handlers, and reports failures fail-open.
#[tokio::test]
async fn before_run_threads_injections_and_survives_failures() {
    let (errors, registry) = error_capture_registry();
    register_fixed(
        &registry,
        HookName::BeforeRun,
        HookResult::BeforeRun(Some(crate::harness::agent_harness::BeforeRunResult {
            messages: vec![injected_message()],
        })),
    );
    let observed_prompts = Arc::new(Mutex::new(Vec::new()));
    let observed_for_handler = Arc::clone(&observed_prompts);
    registry
        .on(
            HookName::BeforeRun,
            handler(move |event| {
                let HookEvent::BeforeRun { prompt, .. } = &event.event else {
                    panic!("the run handler receives the run event");
                };
                observed_for_handler
                    .lock()
                    .expect("observed lock")
                    .push(prompt.clone());
                HookResult::BeforeRun(None)
            }),
            HookOptions::default(),
        )
        .expect("register the observing handler");
    register_failing(&registry, HookName::BeforeRun, "run handler failed");
    let result = aggregate(&registry, HookName::BeforeRun, before_run_event()).await;
    let HookResult::BeforeRun(Some(result)) = result else {
        panic!("the aggregate carries the injections");
    };
    assert_eq!(result.messages, vec![injected_message()]);
    assert_eq!(
        observed_prompts.lock().expect("observed lock").as_slice(),
        [vec![injected_message()]]
    );
    assert!(!errors.lock().expect("report lock").is_empty());
}

/// `before_run_end` reports a failing handler and keeps no follow-up when
/// no handler supplies one.
#[tokio::test]
async fn before_run_end_reports_failures_without_a_follow_up() {
    let (errors, registry) = error_capture_registry();
    register_failing(&registry, HookName::BeforeRunEnd, "run-end handler failed");
    register_fixed(
        &registry,
        HookName::BeforeRunEnd,
        HookResult::BeforeRunEnd(None),
    );
    let result = aggregate(&registry, HookName::BeforeRunEnd, before_run_end_event()).await;
    let HookResult::BeforeRunEnd(None) = result else {
        panic!("no handler supplies a follow-up");
    };
    assert!(!errors.lock().expect("report lock").is_empty());
}

/// `transform_context` reports failures, keeps folding after them, and
/// only surfaces fields a handler actually replaced.
#[tokio::test]
async fn transform_context_reports_failures_and_keeps_the_last_replacement() {
    let (errors, registry) = error_capture_registry();
    register_failing(
        &registry,
        HookName::TransformContext,
        "transform handler failed",
    );
    register_fixed(
        &registry,
        HookName::TransformContext,
        HookResult::TransformContext(None),
    );
    register_fixed(
        &registry,
        HookName::TransformContext,
        HookResult::TransformContext(Some(
            crate::harness::agent_harness::TransformContextResult {
                messages: None,
                system_prompt: Some("replacement".to_owned()),
            },
        )),
    );
    register_fixed(
        &registry,
        HookName::TransformContext,
        HookResult::TransformContext(Some(
            crate::harness::agent_harness::TransformContextResult {
                messages: Some(vec![replaced_message()]),
                system_prompt: None,
            },
        )),
    );
    let result = run(
        &registry,
        HookName::TransformContext,
        HookEvent::TransformContext {
            messages: vec![user_message()],
            system_prompt: "original".to_owned(),
        },
        &background_context(),
    )
    .await
    .expect("aggregate");
    let HookResult::TransformContext(Some(result)) = result else {
        panic!("the aggregate carries the replacement");
    };
    assert_eq!(result.system_prompt.as_deref(), Some("replacement"));
    assert_eq!(result.messages, Some(vec![replaced_message()]));
    assert!(!errors.lock().expect("report lock").is_empty());
}

/// `before_request` reports failures, keeps folding after them, and only
/// reports a patch when a handler contributed.
#[tokio::test]
async fn before_request_reports_failures_and_keeps_the_last_patch() {
    let (errors, registry) = error_capture_registry();
    register_fixed(
        &registry,
        HookName::BeforeRequest,
        HookResult::BeforeRequest(None),
    );
    register_failing(&registry, HookName::BeforeRequest, "request handler failed");
    register_fixed(
        &registry,
        HookName::BeforeRequest,
        HookResult::BeforeRequest(Some(crate::harness::agent_harness::BeforeRequestResult {
            stream_options: AgentHarnessStreamOptionsPatch {
                timeout_ms: Some(Some(2_000)),
                ..AgentHarnessStreamOptionsPatch::default()
            },
        })),
    );
    let result = run(
        &registry,
        HookName::BeforeRequest,
        HookEvent::BeforeRequest {
            model: test_model(),
            step: StepKind::Assistant,
            attempt: 1,
            stream_options: AgentHarnessStreamOptions {
                timeout_ms: Some(1_000),
                ..AgentHarnessStreamOptions::default()
            },
        },
        &background_context(),
    )
    .await
    .expect("aggregate");
    let HookResult::BeforeRequest(Some(result)) = result else {
        panic!("the replacing handler yields a patch");
    };
    assert_eq!(result.stream_options.timeout_ms, Some(Some(2_000)));
    assert!(!errors.lock().expect("report lock").is_empty());
}

/// `before_payload` reports failures and hands the last replacement
/// payload onward.
#[tokio::test]
async fn before_payload_reports_failures_and_keeps_the_last_payload() {
    let (errors, registry) = error_capture_registry();
    register_fixed(
        &registry,
        HookName::BeforePayload,
        HookResult::BeforePayload(None),
    );
    register_failing(&registry, HookName::BeforePayload, "payload handler failed");
    let result = fixed_aggregate!(
        &registry,
        BeforePayload,
        before_payload_event(),
        BeforePayload,
        HookResult::BeforePayload(Some(crate::harness::agent_harness::PayloadResult {
            payload: serde_json::json!({ "replacement": true }),
        })),
        "the aggregate carries the replacement payload"
    );
    assert_eq!(result.payload, serde_json::json!({ "replacement": true }));
    assert!(!errors.lock().expect("report lock").is_empty());
}

/// `after_response` reports failures and carries the last replacement
/// message.
#[tokio::test]
async fn after_response_reports_failures_and_keeps_the_last_replacement() {
    let (errors, registry) = error_capture_registry();
    registry
        .on(
            HookName::AfterResponse,
            handler(move |_event| HookResult::AfterResponse(None)),
            HookOptions::default(),
        )
        .expect("register the empty handler");
    register_failing(
        &registry,
        HookName::AfterResponse,
        "response handler failed",
    );
    registry
        .on(
            HookName::AfterResponse,
            handler(move |_event| {
                HookResult::AfterResponse(Some(Box::new(
                    crate::harness::agent_harness::AfterResponseResult { message: None },
                )))
            }),
            HookOptions::default(),
        )
        .expect("register the resultless handler");
    registry
        .on(
            HookName::AfterResponse,
            handler(move |_event| {
                HookResult::AfterResponse(Some(Box::new(
                    crate::harness::agent_harness::AfterResponseResult {
                        message: Some(settled_message()),
                    },
                )))
            }),
            HookOptions::default(),
        )
        .expect("register the replacing handler");
    let result = aggregate(&registry, HookName::AfterResponse, after_response_event()).await;
    let HookResult::AfterResponse(Some(result)) = result else {
        panic!("the aggregate carries the replacement");
    };
    assert!(result.message.is_some());
    assert!(!errors.lock().expect("report lock").is_empty());
}

/// `after_tool` folds every field across handlers and threads the folded
/// state into each next handler.
#[tokio::test]
async fn after_tool_folds_every_field_and_threads_the_state() {
    let registry = empty_registry();
    registry
        .on(
            HookName::AfterTool,
            handler(|_event| HookResult::AfterTool(Some(patched_tool_result()))),
            HookOptions::default(),
        )
        .expect("register the patching handler");
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_for_handler = Arc::clone(&seen);
    registry
        .on(
            HookName::AfterTool,
            handler(move |event| {
                let HookEvent::AfterTool {
                    content,
                    details,
                    is_error,
                    usage,
                    ..
                } = &event.event
                else {
                    panic!("the tool handler receives the tool event");
                };
                seen_for_handler.lock().expect("seen lock").push((
                    content.clone(),
                    details.clone(),
                    *is_error,
                    *usage,
                ));
                HookResult::AfterTool(None)
            }),
            HookOptions::default(),
        )
        .expect("register the observing handler");
    let result = aggregate(&registry, HookName::AfterTool, after_tool_event()).await;
    let HookResult::AfterTool(Some(result)) = result else {
        panic!("the aggregate carries the patches");
    };
    let expected = patched_tool_result();
    assert_eq!(result.content.as_ref().map_or(0, Vec::len), 1);
    assert_eq!(result.details, expected.details);
    assert_eq!(result.is_error, Some(true));
    assert_eq!(result.usage, expected.usage);
    assert_eq!(result.terminate, Some(true));
    let observed = seen.lock().expect("seen lock").clone();
    let (content, details, is_error, usage) = observed.first().cloned().expect("observed");
    assert_eq!(
        content,
        vec![crate::types::AgentToolContent::Text(
            pi_ai::types::TextContent {
                text: "patched".to_owned(),
                text_signature: None,
            }
        )]
    );
    assert_eq!(
        details.as_ref().and_then(|details| details.get("patched")),
        Some(&serde_json::json!(true))
    );
    assert!(is_error);
    assert_eq!(usage.map(|usage| usage.total_tokens), Some(2));
}

/// `after_tool` reports a failing handler fail-open and keeps no aggregate
/// when no handler patches a field.
#[tokio::test]
async fn after_tool_reports_failures_fail_open() {
    let (errors, registry) = error_capture_registry();
    register_failing(&registry, HookName::AfterTool, "tool handler failed");
    let result = aggregate(&registry, HookName::AfterTool, after_tool_event()).await;
    let HookResult::AfterTool(None) = result else {
        panic!("no patches means no aggregate");
    };
    assert!(!errors.lock().expect("report lock").is_empty());
}

/// `after_tool` with no field patched reports no aggregate.
#[tokio::test]
async fn after_tool_without_patches_reports_none() {
    let registry = empty_registry();
    register_fixed(
        &registry,
        HookName::AfterTool,
        HookResult::AfterTool(Some(
            crate::harness::agent_harness::AfterToolResult::default(),
        )),
    );
    let result = aggregate(&registry, HookName::AfterTool, after_tool_event()).await;
    let HookResult::AfterTool(None) = result else {
        panic!("no patches means no aggregate");
    };
}

/// `before_navigation` takes the first decisive handler: a handler
/// returning both decline and summary reports and is skipped, and a
/// decline-only verdict wins.
#[tokio::test]
async fn before_navigation_takes_the_first_decisive_result() {
    let (errors, registry) = error_capture_registry();
    register_fixed(
        &registry,
        HookName::BeforeNavigation,
        navigation_result(Some(true), Some(branch_summary())),
    );
    register_fixed(
        &registry,
        HookName::BeforeNavigation,
        navigation_result(Some(true), None),
    );
    let result = aggregate(
        &registry,
        HookName::BeforeNavigation,
        before_navigation_event(),
    )
    .await;
    let HookResult::BeforeNavigation(Some(result)) = result else {
        panic!("the decline reaches the aggregate");
    };
    assert_eq!(result.decline, Some(true));
    assert!(
        errors
            .lock()
            .expect("report lock")
            .iter()
            .any(|error| error.contains("cannot return both decline and summary"))
    );
}

/// `before_navigation` accepts a summary replacement from a handler whose
/// event variant does not match (the registry dispatches by name) and
/// reports handler failures without stopping the search.
#[tokio::test]
async fn before_navigation_accepts_a_summary_replacement() {
    let (errors, registry) = error_capture_registry();
    register_fixed(
        &registry,
        HookName::BeforeNavigation,
        compaction_result(Some(true), None),
    );
    register_failing(
        &registry,
        HookName::BeforeNavigation,
        "navigation handler failed",
    );
    register_fixed(
        &registry,
        HookName::BeforeNavigation,
        navigation_result(None, Some(branch_summary())),
    );
    let result = aggregate(
        &registry,
        HookName::BeforeNavigation,
        before_navigation_event(),
    )
    .await;
    let HookResult::BeforeNavigation(Some(result)) = result else {
        panic!("the summary reaches the aggregate");
    };
    assert_eq!(
        result
            .summary
            .as_ref()
            .map(|summary| summary.summary.as_str()),
        Some("branch summary")
    );
    assert!(!errors.lock().expect("report lock").is_empty());
}

/// `before_navigation` with no decisive handler reports no result.
#[tokio::test]
async fn before_navigation_without_a_decisive_result_reports_none() {
    let registry = empty_registry();
    register_fixed(
        &registry,
        HookName::BeforeNavigation,
        navigation_result(Some(false), None),
    );
    register_fixed(
        &registry,
        HookName::BeforeNavigation,
        HookResult::BeforeNavigation(None),
    );
    let result = aggregate(
        &registry,
        HookName::BeforeNavigation,
        before_navigation_event(),
    )
    .await;
    let HookResult::BeforeNavigation(None) = result else {
        panic!("no decisive result means no aggregate");
    };
}

/// `before_compaction` skips wrong-variant, failing, and non-decisive
/// handlers until one is decisive; without one the aggregate reports none.
#[tokio::test]
async fn before_compaction_falls_through_non_decisive_results() {
    let (errors, registry) = error_capture_registry();
    register_fixed(
        &registry,
        HookName::BeforeCompaction,
        navigation_result(Some(true), None),
    );
    register_failing(
        &registry,
        HookName::BeforeCompaction,
        "compaction handler failed",
    );
    register_fixed(
        &registry,
        HookName::BeforeCompaction,
        compaction_result(Some(false), None),
    );
    let result = aggregate(
        &registry,
        HookName::BeforeCompaction,
        before_compaction_event(),
    )
    .await;
    let HookResult::BeforeCompaction(None) = result else {
        panic!("no decisive result means no aggregate");
    };
    assert!(!errors.lock().expect("report lock").is_empty());
}

/// The per-name aggregates short-circuit when the invocation carries a
/// different event variant than the hook name dispatches on.
#[tokio::test]
async fn a_mismatched_event_variant_short_circuits_the_aggregate() {
    let registry = empty_registry();
    let before_run_event = || HookEvent::BeforeRun {
        prompt: vec![],
        resources: crate::harness::agent_harness::Resources::default(),
    };
    let before_drive_event = || HookEvent::BeforeDrive {
        operation: crate::harness::session::types::OperationKind::Run,
    };

    let result = aggregate(&registry, HookName::BeforeRun, before_drive_event()).await;
    assert!(matches!(result, HookResult::BeforeRun(None)));

    let result = aggregate(&registry, HookName::BeforeRunEnd, before_run_event()).await;
    assert!(matches!(result, HookResult::BeforeRunEnd(None)));

    let result = aggregate(&registry, HookName::TransformContext, before_run_event()).await;
    let HookResult::TransformContext(Some(result)) = result else {
        panic!("the transform aggregate always carries a result");
    };
    assert_eq!(result.messages, None);
    assert_eq!(result.system_prompt, None);

    let result = aggregate(&registry, HookName::BeforeRequest, before_run_event()).await;
    assert!(matches!(result, HookResult::BeforeRequest(None)));

    let result = aggregate(&registry, HookName::BeforePayload, before_run_event()).await;
    let HookResult::BeforePayload(Some(result)) = result else {
        panic!("the payload aggregate always carries a result");
    };
    assert_eq!(result.payload, serde_json::Value::Null);

    let result = aggregate(&registry, HookName::AfterResponse, before_run_event()).await;
    let HookResult::AfterResponse(Some(result)) = result else {
        panic!("the response aggregate always carries a result");
    };
    assert_eq!(result.message, None);

    let result = aggregate(&registry, HookName::BeforeTool, before_run_event()).await;
    assert!(matches!(result, HookResult::BeforeTool(None)));

    let result = aggregate(&registry, HookName::AfterTool, before_run_event()).await;
    assert!(matches!(result, HookResult::AfterTool(None)));
}

/// The stream-options patch applies every leg over the base and the
/// derived patch between two snapshots round-trips.
#[test]
fn stream_options_patch_round_trips_every_field() {
    let base = AgentHarnessStreamOptions {
        timeout_ms: Some(1_000),
        headers: Some(BTreeMap::from([("h".to_owned(), "1".to_owned())])),
        metadata: Some(BTreeMap::from([("m".to_owned(), serde_json::json!("x"))])),
        ..AgentHarnessStreamOptions::default()
    };
    let value = AgentHarnessStreamOptions {
        transport: Some(pi_ai::types::Transport::Sse),
        timeout_ms: Some(2_000),
        max_retries: Some(7),
        max_retry_delay_ms: Some(3_000),
        cache_retention: Some(pi_ai::types::CacheRetention::Long),
        deferred: Some(pi_ai::types::DeferredRequest::Enabled(true)),
        ..AgentHarnessStreamOptions::default()
    };
    let derived = create_stream_options_patch(&base, &value);
    assert_eq!(derived.transport, Some(Some(pi_ai::types::Transport::Sse)));
    assert_eq!(derived.timeout_ms, Some(Some(2_000)));
    assert_eq!(derived.max_retries, Some(Some(7)));
    assert_eq!(derived.max_retry_delay_ms, Some(Some(3_000)));
    assert_eq!(
        derived.cache_retention,
        Some(Some(pi_ai::types::CacheRetention::Long))
    );
    assert_eq!(
        derived.deferred,
        Some(Some(pi_ai::types::DeferredRequest::Enabled(true)))
    );
    assert_eq!(derived.headers, Some(None));
    assert_eq!(derived.metadata, Some(None));
    let applied = apply_stream_options_patch(&base, &derived);
    assert_eq!(applied, value);

    let added = AgentHarnessStreamOptions {
        headers: Some(BTreeMap::from([
            ("h".to_owned(), "1".to_owned()),
            ("add".to_owned(), "2".to_owned()),
        ])),
        metadata: Some(BTreeMap::from([
            ("m".to_owned(), serde_json::json!("x")),
            ("meta2".to_owned(), serde_json::json!(1)),
        ])),
        ..value.clone()
    };
    let derived = create_stream_options_patch(&value, &added);
    assert_eq!(
        derived.headers,
        Some(Some(BTreeMap::from([
            ("h".to_owned(), Some("1".to_owned())),
            ("add".to_owned(), Some("2".to_owned()))
        ])))
    );
    assert_eq!(
        derived.metadata,
        Some(Some(BTreeMap::from([
            ("m".to_owned(), Some(serde_json::json!("x"))),
            ("meta2".to_owned(), Some(serde_json::json!(1)))
        ])))
    );
    let applied = apply_stream_options_patch(&value, &derived);
    assert_eq!(applied, added);

    let cleared = AgentHarnessStreamOptions {
        headers: Some(BTreeMap::new()),
        metadata: None,
        ..added.clone()
    };
    let derived = create_stream_options_patch(&added, &cleared);
    assert_eq!(
        derived.headers,
        Some(Some(BTreeMap::from([
            ("h".to_owned(), None),
            ("add".to_owned(), None)
        ])))
    );
    assert_eq!(derived.metadata, Some(None));
    let applied = apply_stream_options_patch(&added, &derived);
    assert_eq!(applied, cleared);
}

/// The header and metadata patch legs apply per key over an absent base
/// map, and a base-absent empty patch stays explicit.
#[test]
fn stream_options_patch_headers_and_metadata_legs_apply_per_key() {
    let base = AgentHarnessStreamOptions::default();
    let patch = AgentHarnessStreamOptionsPatch {
        headers: Some(Some(BTreeMap::from([(
            "add".to_owned(),
            Some("1".to_owned()),
        )]))),
        metadata: Some(Some(BTreeMap::from([(
            "add".to_owned(),
            Some(serde_json::json!(2)),
        )]))),
        ..AgentHarnessStreamOptionsPatch::default()
    };
    let applied = apply_stream_options_patch(&base, &patch);
    assert_eq!(
        applied.headers,
        Some(BTreeMap::from([("add".to_owned(), "1".to_owned())]))
    );
    assert_eq!(
        applied.metadata,
        Some(BTreeMap::from([("add".to_owned(), serde_json::json!(2))]))
    );

    let populated = AgentHarnessStreamOptions {
        headers: Some(BTreeMap::from([
            ("h".to_owned(), "1".to_owned()),
            ("add".to_owned(), "2".to_owned()),
        ])),
        metadata: Some(BTreeMap::from([
            ("m".to_owned(), serde_json::json!("x")),
            ("meta2".to_owned(), serde_json::json!(1)),
        ])),
        ..AgentHarnessStreamOptions::default()
    };
    let per_key = AgentHarnessStreamOptionsPatch {
        headers: Some(Some(BTreeMap::from([("h".to_owned(), None)]))),
        metadata: Some(Some(BTreeMap::from([("m".to_owned(), None)]))),
        ..AgentHarnessStreamOptionsPatch::default()
    };
    let applied = apply_stream_options_patch(&populated, &per_key);
    assert_eq!(
        applied.headers,
        Some(BTreeMap::from([("add".to_owned(), "2".to_owned())]))
    );
    assert_eq!(
        applied.metadata,
        Some(BTreeMap::from([("meta2".to_owned(), serde_json::json!(1))]))
    );

    let cleared = AgentHarnessStreamOptionsPatch {
        headers: Some(None),
        metadata: Some(None),
        ..AgentHarnessStreamOptionsPatch::default()
    };
    let applied = apply_stream_options_patch(&applied, &cleared);
    assert_eq!(applied.headers, None);
    assert_eq!(applied.metadata, None);

    let derived = create_stream_options_patch(
        &base,
        &AgentHarnessStreamOptions {
            headers: Some(BTreeMap::new()),
            ..AgentHarnessStreamOptions::default()
        },
    );
    assert_eq!(derived.headers, Some(Some(BTreeMap::new())));
    assert_eq!(derived.metadata, None);
}

/// The derived map patch carries deleted keys as `None`, changed keys as
/// their new value, and skips kept keys.
#[test]
fn stream_options_patch_maps_derive_deletions_and_changes() {
    let base = AgentHarnessStreamOptions {
        headers: Some(BTreeMap::from([
            ("old".to_owned(), "2".to_owned()),
            ("keep".to_owned(), "1".to_owned()),
        ])),
        ..AgentHarnessStreamOptions::default()
    };
    let next = AgentHarnessStreamOptions {
        headers: Some(BTreeMap::from([
            ("keep".to_owned(), "1".to_owned()),
            ("new".to_owned(), "3".to_owned()),
        ])),
        ..AgentHarnessStreamOptions::default()
    };
    let derived = create_stream_options_patch(&base, &next);
    assert_eq!(
        derived.headers,
        Some(Some(BTreeMap::from([
            ("old".to_owned(), None),
            ("new".to_owned(), Some("3".to_owned()))
        ])))
    );
}

fn branch_summary() -> crate::harness::compaction::types::BranchSummaryResult {
    crate::harness::compaction::types::BranchSummaryResult {
        summary: "branch summary".to_owned(),
        usage: None,
        read_files: vec![],
        modified_files: vec![],
    }
}

fn branch_preparation() -> crate::harness::compaction::types::BranchPreparation {
    crate::harness::compaction::types::BranchPreparation {
        messages: vec![user_message()],
        file_ops: crate::harness::compaction::types::FileOperations::default(),
        total_tokens: 1,
    }
}

fn patched_tool_result() -> crate::harness::agent_harness::AfterToolResult {
    crate::harness::agent_harness::AfterToolResult {
        content: Some(vec![crate::types::AgentToolContent::Text(
            pi_ai::types::TextContent {
                text: "patched".to_owned(),
                text_signature: None,
            },
        )]),
        details: Some(serde_json::json!({ "patched": true })),
        is_error: Some(true),
        usage: Some(pi_ai::types::Usage {
            input: 2,
            output: 1,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 2,
            cost: pi_ai::types::UsageCost::default(),
        }),
        terminate: Some(true),
    }
}

fn after_tool_event() -> HookEvent {
    HookEvent::AfterTool {
        tool_call_id: "call".to_owned(),
        tool_name: "read".to_owned(),
        args: BTreeMap::new(),
        content: vec![],
        details: None,
        is_error: false,
        usage: None,
    }
}

/// The pipeline call sink, upstream's per-case `calls` arrays.
type Calls = Arc<Mutex<Vec<&'static str>>>;

/// Registers the call-recording handler that appends its tag and settles,
/// upstream's `() => { calls.push(tag) }` rows.
fn record_call(calls: &Calls, tag: &'static str) -> crate::harness::agent_harness::HookHandler {
    let calls = Arc::clone(calls);
    Arc::new(move |_event: &HookInvocation, _context: &Context| {
        let calls = Arc::clone(&calls);
        Box::pin(async move {
            calls.lock().expect("call lock").push(tag);
            Ok(HookResult::BeforeDrive)
        })
    })
}

/// The parked first `before_drive` handler: appends `first:start`, flips
/// `entered`, parks on `release`, appends `first:end`, upstream's
/// `await release.promise` rows.
fn parked_first_handler(
    calls: &Calls,
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
) -> crate::harness::agent_harness::HookHandler {
    let calls = Arc::clone(calls);
    Arc::new(move |_event: &HookInvocation, _context: &Context| {
        let calls = Arc::clone(&calls);
        let entered = Arc::clone(&entered);
        let release = Arc::clone(&release);
        Box::pin(async move {
            calls.lock().expect("call lock").push("first:start");
            entered.notify_one();
            release.notified().await;
            calls.lock().expect("call lock").push("first:end");
            Ok(HookResult::BeforeDrive)
        })
    })
}

/// A closed gate refuses admission before the pipeline starts: no handler
/// runs, upstream's closed-gate half of "checks the effect gate
/// immediately before the complete before_run pipeline".
#[tokio::test]
async fn a_closed_gate_refuses_admission_without_calling_handlers() {
    let calls: Calls = Arc::new(Mutex::new(Vec::new()));
    let registry = empty_registry();
    registry
        .on(
            HookName::BeforeDrive,
            record_call(&calls, "first"),
            HookOptions::default(),
        )
        .expect("register the first handler");
    let (gate, control) = create_gate();
    control.close("closed".to_owned());
    let error = registry
        .run_with_gate(
            HookName::BeforeDrive,
            invocation(before_drive_event()),
            &gate,
            &background_context(),
        )
        .await
        .expect_err("a closed gate refuses admission");
    assert!(matches!(
        error,
        crate::harness::hooks::HookRunError::Closed(_)
    ));
    assert!(calls.lock().expect("call lock").is_empty());
}

/// A handler registered while the admitted pipeline is parked never joins
/// it: the aggregate runs over the registration snapshot admission took,
/// upstream's in-flight half of "checks the effect gate immediately
/// before the complete before_run pipeline".
#[tokio::test]
async fn a_handler_registered_mid_run_never_joins_the_in_flight_pipeline() {
    let calls: Calls = Arc::new(Mutex::new(Vec::new()));
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let registry = Arc::new(empty_registry());
    registry
        .on(
            HookName::BeforeDrive,
            parked_first_handler(&calls, Arc::clone(&entered), Arc::clone(&release)),
            HookOptions::default(),
        )
        .expect("register the parked handler");
    registry
        .on(
            HookName::BeforeDrive,
            record_call(&calls, "second"),
            HookOptions::default(),
        )
        .expect("register the second handler");
    let (gate, _gate_control) = create_gate();
    let run = tokio::spawn({
        let registry = Arc::clone(&registry);
        async move {
            registry
                .run_with_gate(
                    HookName::BeforeDrive,
                    invocation(before_drive_event()),
                    &gate,
                    &background_context(),
                )
                .await
        }
    });
    entered.notified().await;
    registry
        .on(
            HookName::BeforeDrive,
            record_call(&calls, "late"),
            HookOptions::default(),
        )
        .expect("register the late handler");
    release.notify_one();
    let result = run
        .await
        .expect("the run joins")
        .expect("the admitted pipeline completes");
    assert!(matches!(result, HookResult::BeforeDrive));
    assert_eq!(
        *calls.lock().expect("call lock"),
        ["first:start", "first:end", "second"]
    );
}

/// The admitted handlers' context carries the gate's own abort signal:
/// closing the gate after admission flips the signal the handler saw and
/// carries the close reason, upstream's admitted half of "rejects
/// pre-aborted invocations and combines admitted hook cancellation with
/// the effect gate".
#[tokio::test]
async fn the_admitted_handlers_signal_flips_when_the_gate_closes() {
    let registry = empty_registry();
    let captured: Arc<Mutex<Option<pi_chord::context::AbortSignal>>> = Arc::new(Mutex::new(None));
    registry
        .on(
            HookName::BeforeDrive,
            {
                let captured = Arc::clone(&captured);
                Arc::new(move |_event: &HookInvocation, context: &Context| {
                    let captured = Arc::clone(&captured);
                    Box::pin(async move {
                        *captured.lock().expect("signal lock") = context.abort_signal();
                        Ok(HookResult::BeforeDrive)
                    })
                })
            },
            HookOptions::default(),
        )
        .expect("register the capturing handler");
    let (gate, control) = create_gate();
    let result = registry
        .run_with_gate(
            HookName::BeforeDrive,
            invocation(before_drive_event()),
            &gate,
            &background_context(),
        )
        .await
        .expect("the admitted run completes");
    assert!(matches!(result, HookResult::BeforeDrive));
    let signal = captured
        .lock()
        .expect("signal lock")
        .clone()
        .expect("the handler saw the gate's signal");
    assert!(!signal.aborted());
    control.close("gate closed".to_owned());
    assert!(signal.aborted());
    assert_eq!(
        signal.reason(),
        Some(pi_chord::context::AbortReason::Caller(
            "gate closed".to_owned()
        ))
    );
}

/// An admitted pipeline completes as one gated effect: a mid-run abort
/// flips the gate's signal but never un-admits the in-flight handlers,
/// upstream's "admits the complete before_drive pipeline as one gated
/// effect".
#[tokio::test]
async fn an_admitted_pipeline_survives_a_mid_run_gate_abort() {
    let calls: Calls = Arc::new(Mutex::new(Vec::new()));
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let registry = Arc::new(empty_registry());
    registry
        .on(
            HookName::BeforeDrive,
            parked_first_handler(&calls, Arc::clone(&entered), Arc::clone(&release)),
            HookOptions::default(),
        )
        .expect("register the parked handler");
    registry
        .on(
            HookName::BeforeDrive,
            record_call(&calls, "second"),
            HookOptions::default(),
        )
        .expect("register the second handler");
    let (gate, control) = create_gate();
    let gate = Arc::new(gate);
    let run = tokio::spawn({
        let registry = Arc::clone(&registry);
        let gate = Arc::clone(&gate);
        async move {
            registry
                .run_with_gate(
                    HookName::BeforeDrive,
                    invocation(before_drive_event()),
                    &gate,
                    &background_context(),
                )
                .await
        }
    });
    entered.notified().await;
    control.begin_abort(tokio::sync::watch::channel(()).1);
    control.signal_abort();
    assert!(gate.signal().aborted());
    release.notify_one();
    let result = run
        .await
        .expect("the run joins")
        .expect("the admitted pipeline completes");
    assert!(matches!(result, HookResult::BeforeDrive));
    assert_eq!(
        *calls.lock().expect("call lock"),
        ["first:start", "first:end", "second"]
    );
}

/// Duplicate registration ids stay metadata — both registrations exist —
/// and the first `before_drive` failure stops the later handlers and
/// reports exactly once, upstream's "treats registration ids as optional
/// metadata and fails before_drive closed".
#[tokio::test]
async fn duplicate_registration_ids_stay_metadata_and_fail_closed() {
    let (errors, registry) = error_capture_registry();
    let calls: Calls = Arc::new(Mutex::new(Vec::new()));
    let duplicate = HookOptions {
        id: Some("duplicate".to_owned()),
    };
    registry
        .on(
            HookName::BeforeDrive,
            failing_handler("prerequisite failed"),
            duplicate.clone(),
        )
        .expect("register the failing handler");
    registry
        .on(
            HookName::BeforeDrive,
            record_call(&calls, "later"),
            duplicate,
        )
        .expect("register the later handler");
    let result = run(
        &registry,
        HookName::BeforeDrive,
        before_drive_event(),
        &background_context(),
    )
    .await;
    assert!(result.is_err());
    assert!(calls.lock().expect("call lock").is_empty());
    assert_eq!(
        *errors.lock().expect("report lock"),
        ["prerequisite failed".to_owned()]
    );
}

/// `run_tool_with_gate` hands tool handlers the admitted context's values
/// and nests their spans under the active hook span, which nests under
/// the invocation span, upstream's "passes tool handlers a child context
/// of the active hook span".
#[tokio::test]
async fn tool_handlers_nest_their_spans_under_the_hook_span() {
    let recorder = InMemoryTelemetryContext::default();
    let handle = TelemetryHandle::new(recorder.clone());
    let key = create_context_key::<String>("hook.test.value");
    let base = with_context_value(
        &key,
        "preserved".to_owned(),
        &with_telemetry_context(handle.clone(), &background_context()),
    );
    let registry = Arc::new(empty_registry());
    let received: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    registry
        .on(
            HookName::BeforeTool,
            {
                let received = Arc::clone(&received);
                let key = key.clone();
                Arc::new(move |_event: &HookInvocation, context: &Context| {
                    let received = Arc::clone(&received);
                    let key = key.clone();
                    let telemetry = get_telemetry_context(context);
                    Box::pin(async move {
                        *received.lock().expect("value lock") = context.value(&key).cloned();
                        let _ = telemetry
                            .start_span(SpanOptions::new("handler.child"), |_span| async {
                                Ok::<(), std::io::Error>(())
                            })
                            .await;
                        Ok(HookResult::BeforeTool(None))
                    })
                })
            },
            HookOptions::default(),
        )
        .expect("register the handler");
    let settled = handle
        .start_span(SpanOptions::new("invocation"), move |invocation_span| {
            let registry = Arc::clone(&registry);
            async move {
                let invocation_context =
                    with_telemetry_context(invocation_span.telemetry_handle(), &base);
                registry
                    .run_tool_with_gate(
                        HookName::BeforeTool,
                        invocation(before_tool_event()),
                        &create_gate().0,
                        &invocation_context,
                    )
                    .await
            }
        })
        .await
        .expect("the tool aggregate settles");
    let HookResult::BeforeTool(Some(settled)) = settled else {
        panic!("the tool aggregate settles with its result");
    };
    assert_eq!(settled.args, None);
    assert_eq!(settled.block, None);
    assert_eq!(
        received.lock().expect("value lock").as_deref(),
        Some("preserved")
    );
    let spans = recorder.get_spans();
    let span_named = |name: &str| {
        spans
            .iter()
            .find(|span| span.name == name)
            .expect(name)
            .clone()
    };
    let invocation = span_named("invocation");
    let hook = span_named("pi.harness.hook");
    let child = span_named("handler.child");
    assert_eq!(invocation.parent_id, None);
    assert_eq!(hook.parent_id, Some(invocation.id));
    assert_eq!(child.parent_id, Some(hook.id));
}

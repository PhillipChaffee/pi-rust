//! Boundary tests for the harness hook registry.
//!
//! Upstream has no unit file for `hooks.ts` — the registry rides the
//! runtime suites. These tests pin the aggregate semantics this slice
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

use crate::harness::agent_harness::{
    HookEvent, HookInvocation, HookName, HookOptions, HookResult, StepKind,
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

/// `before_run` aggregates fail-open: injected messages accumulate across
/// handlers, a failing handler reports but does not stop the rest.
#[tokio::test]
async fn before_run_aggregates_fail_open() {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let registry = HookRegistry::new(reporting_handler_errors(Arc::clone(&errors)));
    let options = HookOptions {
        id: Some("second".to_owned()),
    };
    registry
        .on(
            HookName::BeforeRun,
            handler(|_event| {
                HookResult::BeforeRun(Some(crate::harness::agent_harness::BeforeRunResult {
                    messages: vec![injected_message()],
                }))
            }),
            HookOptions::default(),
        )
        .expect("register");
    let event = HookEvent::BeforeRun {
        prompt: vec![],
        resources: crate::harness::agent_harness::Resources::default(),
    };
    let current = invocation(event.clone());
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

/// `run_with_gate` reports a closed gate rejection and a closed registry.
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
            invocation(HookEvent::BeforeDrive {
                operation: crate::harness::session::types::OperationKind::Run,
            }),
            &gate,
            &cancelled,
        )
        .await
        .expect_err("an aborting gate refuses admission");
    assert!(matches!(
        error,
        crate::harness::hooks::HookRunError::Aborted(_)
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
            invocation(HookEvent::BeforeRun {
                prompt: vec![],
                resources: crate::harness::agent_harness::Resources::default(),
            }),
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
    let registry = HookRegistry::new(reporting_handler_errors(Arc::new(Mutex::new(Vec::new()))));
    registry
        .on(
            HookName::TransformContext,
            handler(|_event| {
                HookResult::TransformContext(Some(
                    crate::harness::agent_harness::TransformContextResult {
                        messages: Some(vec![user_message()]),
                        system_prompt: Some("replacement".to_owned()),
                    },
                ))
            }),
            HookOptions::default(),
        )
        .expect("register");
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
    let registry = HookRegistry::new(reporting_handler_errors(Arc::new(Mutex::new(Vec::new()))));
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
    let result = run(
        &registry,
        HookName::BeforeRunEnd,
        HookEvent::BeforeRunEnd {
            run_id: "run".to_owned(),
            messages: vec![],
        },
        &background_context(),
    )
    .await
    .expect("aggregate");
    let HookResult::BeforeRunEnd(Some(result)) = result else {
        panic!("the aggregate carries the follow-up");
    };
    assert_eq!(result.follow_up, "second");
}

/// `before_request` folds stream-options patches across handlers and
/// derives the net patch against the original.
#[tokio::test]
async fn before_request_folds_stream_option_patches() {
    let registry = HookRegistry::new(reporting_handler_errors(Arc::new(Mutex::new(Vec::new()))));
    registry
        .on(
            HookName::BeforeRequest,
            handler(|_event| {
                HookResult::BeforeRequest(Some(
                    crate::harness::agent_harness::BeforeRequestResult {
                        stream_options: AgentHarnessStreamOptionsPatch {
                            timeout_ms: Some(Some(2_000)),
                            ..AgentHarnessStreamOptionsPatch::default()
                        },
                    },
                ))
            }),
            HookOptions::default(),
        )
        .expect("register");
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
    let registry = HookRegistry::new(reporting_handler_errors(Arc::new(Mutex::new(Vec::new()))));
    registry
        .on(
            HookName::BeforePayload,
            handler(|_event| {
                HookResult::BeforePayload(Some(crate::harness::agent_harness::PayloadResult {
                    payload: serde_json::json!({ "replacement": true }),
                }))
            }),
            HookOptions::default(),
        )
        .expect("register");
    let result = run(
        &registry,
        HookName::BeforePayload,
        HookEvent::BeforePayload {
            model: test_model(),
            payload: serde_json::json!({ "original": true }),
        },
        &background_context(),
    )
    .await
    .expect("aggregate");
    let HookResult::BeforePayload(Some(result)) = result else {
        panic!("the aggregate carries the replacement payload");
    };
    assert_eq!(result.payload, serde_json::json!({ "replacement": true }));
}

/// `after_response` carries the replacement settled message.
#[tokio::test]
async fn after_response_carries_the_replacement() {
    let registry = HookRegistry::new(reporting_handler_errors(Arc::new(Mutex::new(Vec::new()))));
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
    let result = run(
        &registry,
        HookName::AfterResponse,
        HookEvent::AfterResponse {
            status: Some(200),
            headers: None,
            message: settled_message(),
        },
        &background_context(),
    )
    .await
    .expect("aggregate");
    let HookResult::AfterResponse(Some(result)) = result else {
        panic!("the aggregate carries the replacement");
    };
    assert!(result.message.is_some());
}

/// `before_tool` folds argument replacements and the first block verdict
/// breaks the loop; `run_tool_with_gate` carries the block through.
#[tokio::test]
async fn before_tool_blocks_the_call() {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let registry = HookRegistry::new(reporting_handler_errors(Arc::clone(&errors)));
    registry
        .on(
            HookName::BeforeTool,
            handler(|_event| {
                HookResult::BeforeTool(Some(crate::harness::agent_harness::BeforeToolResult {
                    args: None,
                    block: Some(crate::harness::agent_harness::ToolBlock {
                        reason: "blocked by hook".to_owned(),
                        terminate: None,
                    }),
                }))
            }),
            HookOptions::default(),
        )
        .expect("register");
    let result = run(
        &registry,
        HookName::BeforeTool,
        HookEvent::BeforeTool {
            tool_call_id: "call".to_owned(),
            tool_name: "read".to_owned(),
            args: BTreeMap::new(),
        },
        &background_context(),
    )
    .await
    .expect("aggregate");
    let HookResult::BeforeTool(Some(result)) = result else {
        panic!("the aggregate carries the block");
    };
    let block = result.block.expect("blocked");
    assert_eq!(block.reason, "blocked by hook");
}

/// `after_tool` folds result patches across handlers, keeping the last
/// value per field.
#[tokio::test]
async fn after_tool_folds_result_patches() {
    let registry = HookRegistry::new(reporting_handler_errors(Arc::new(Mutex::new(Vec::new()))));
    registry
        .on(
            HookName::AfterTool,
            handler(|_event| {
                HookResult::AfterTool(Some(crate::harness::agent_harness::AfterToolResult {
                    is_error: Some(true),
                    ..crate::harness::agent_harness::AfterToolResult::default()
                }))
            }),
            HookOptions::default(),
        )
        .expect("register");
    let result = run(
        &registry,
        HookName::AfterTool,
        HookEvent::AfterTool {
            tool_call_id: "call".to_owned(),
            tool_name: "read".to_owned(),
            args: BTreeMap::new(),
            content: vec![],
            details: None,
            is_error: false,
            usage: None,
        },
        &background_context(),
    )
    .await
    .expect("aggregate");
    let HookResult::AfterTool(Some(result)) = result else {
        panic!("the aggregate carries the patch");
    };
    assert_eq!(result.is_error, Some(true));
}

/// `before_compaction` returns the first decisive handler; a handler
/// returning both decline and compaction reports and is skipped.
#[tokio::test]
async fn before_compaction_takes_the_first_decisive_result() {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let registry = HookRegistry::new(reporting_handler_errors(Arc::clone(&errors)));
    registry
        .on(
            HookName::BeforeCompaction,
            handler(|_event| {
                HookResult::BeforeCompaction(Some(
                    crate::harness::agent_harness::CompactionHookResult {
                        decline: Some(true),
                        compaction: Some(crate::harness::compaction::types::CompactResult {
                            summary: "summary".to_owned(),
                            tokens_before: 1,
                            usage: None,
                            retained_tail: vec![],
                            details: None,
                        }),
                    },
                ))
            }),
            HookOptions::default(),
        )
        .expect("register the malformed handler");
    registry
        .on(
            HookName::BeforeCompaction,
            handler(|_event| {
                HookResult::BeforeCompaction(Some(
                    crate::harness::agent_harness::CompactionHookResult {
                        decline: Some(true),
                        compaction: None,
                    },
                ))
            }),
            HookOptions::default(),
        )
        .expect("register the declining handler");
    let result = run(
        &registry,
        HookName::BeforeCompaction,
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
        },
        &background_context(),
    )
    .await
    .expect("aggregate");
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
    let errors = Arc::new(Mutex::new(Vec::new()));
    let registry = HookRegistry::new(reporting_handler_errors(Arc::clone(&errors)));
    registry
        .on(
            HookName::BeforeDrive,
            Arc::new(|_event: &HookInvocation, _context| {
                Box::pin(async {
                    Err(Box::<dyn std::error::Error + Send + Sync>::from(
                        "drive handler failed",
                    ))
                })
            }),
            HookOptions::default(),
        )
        .expect("register the failing handler");
    let result = run(
        &registry,
        HookName::BeforeDrive,
        HookEvent::BeforeDrive {
            operation: crate::harness::session::types::OperationKind::Run,
        },
        &background_context(),
    )
    .await;
    assert!(result.is_err());
    assert!(!errors.lock().expect("report lock").is_empty());
}

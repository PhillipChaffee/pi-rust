//! The tool-execution pipeline's suite, ported 1:1 from upstream
//! `test/harness/execution-tools.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, with boundary tests binding
//! the branches the upstream suite leaves untested.
//!
//! Restatements the port carries: upstream's `isImmediate` narrowing
//! restates as `Result` matching; the "late updates" arm of the executes-
//! with-updates case is unobservable in Rust — the callback reference
//! cannot outlive the execution future, which is the latch's guarantee —
//! so the case pins the forwarded update count and the latch guards only
//! same-future racing pushes; and the untyped-tools case's missing
//! `content` (undefined upstream) restates as an empty content vec.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use pi_ai::auth::resolve::now_ms;
use pi_ai::types::{BoxedFuture, TextContent, ToolResultMessage, Usage, UsageCost};
use serde_json::Value as JsonValue;
use serde_json::json;

use crate::harness::agent_harness::{AfterToolResult, BeforeToolResult, ToolBlock};
use crate::harness::context::{AbortReason, Context, background_context, with_cancel};
use crate::harness::execution::tools::{
    ClearedToolCall, ExecutedToolCall, FinalizedToolCall, ImmediateToolOutcome, PreparedToolCall,
    ToolExecutionRejection, apply_before_tool_decision, create_tool_result_message,
    execute_tool_call, finalize_tool_call, prepare_tool_call, tool_result_from_message,
};
use crate::harness::gate::{Gate, GateRejection, create_gate};
use crate::harness::types::{
    AgentHarnessTool, AgentHarnessToolExecuteFn, AgentHarnessToolInvocation,
    AgentHarnessToolUpdateCallback, AgentHarnessToolUpdateOptions, ToolContext,
};
use crate::types::{AgentToolCall, AgentToolContent, AgentToolError, AgentToolResult};

fn parameters() -> JsonValue {
    json!({
        "type": "object",
        "properties": { "value": { "type": "string" } },
        "required": ["value"]
    })
}

fn call(arguments: &JsonValue) -> AgentToolCall {
    AgentToolCall {
        id: "call-1".to_owned(),
        name: "echo".to_owned(),
        arguments: arguments.as_object().expect("an arguments object").clone(),
        thought_signature: None,
        namespace: None,
    }
}

fn text_content(text: &str) -> AgentToolContent {
    AgentToolContent::Text(TextContent {
        text: text.to_owned(),
        text_signature: None,
    })
}

fn empty_result() -> AgentToolResult {
    AgentToolResult {
        content: Vec::new(),
        details: JsonValue::Null,
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

/// An execute fn counting its invocations and settling an empty result —
/// the probe for the rejection paths that must not run the tool.
fn counting_execute(counter: Arc<AtomicU64>) -> Arc<AgentHarnessToolExecuteFn> {
    Arc::new(
        move |_tool_call_id: &str,
              _args: &JsonValue,
              _on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
              _tool_context: ToolContext,
              _invocation: &dyn AgentHarnessToolInvocation,
              _context: &Context| {
            counter.fetch_add(1, Ordering::Relaxed);
            Box::pin(async { Ok(empty_result()) })
        },
    )
}

fn echo_execute() -> Arc<AgentHarnessToolExecuteFn> {
    Arc::new(
        |_tool_call_id: &str,
         args: &JsonValue,
         _on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
         _tool_context: ToolContext,
         _invocation: &dyn AgentHarnessToolInvocation,
         _context: &Context| {
            let value = args["value"].as_str().unwrap_or_default().to_owned();
            Box::pin(async move {
                Ok(AgentToolResult {
                    content: vec![AgentToolContent::Text(TextContent {
                        text: value.clone(),
                        text_signature: None,
                    })],
                    details: json!({ "value": value }),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        },
    )
}

fn tool(execute: Option<Arc<AgentHarnessToolExecuteFn>>) -> AgentHarnessTool {
    AgentHarnessTool {
        tool: pi_ai::types::Tool {
            name: "echo".to_owned(),
            description: "Echo input".to_owned(),
            parameters: parameters(),
            constrained_sampling: None,
        },
        label: "Echo".to_owned(),
        prepare_arguments: None,
        execute: execute.unwrap_or_else(echo_execute),
        replay: None,
        execution_mode: None,
    }
}

fn text(result: &AgentToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|content| match content {
            AgentToolContent::Text(text) => Some(text.text.clone()),
            AgentToolContent::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn clear_prepared(prepared: Result<PreparedToolCall, ImmediateToolOutcome>) -> ClearedToolCall {
    let Ok(prepared) = prepared else {
        panic!("expected prepared call")
    };
    let Ok(cleared) = apply_before_tool_decision(prepared, None) else {
        panic!("expected cleared call")
    };
    cleared
}

fn effect_gate() -> Gate {
    create_gate().0
}

#[derive(Debug)]
struct StubInvocation;

impl AgentHarnessToolInvocation for StubInvocation {
    fn invocation_id(&self) -> &'static str {
        "result-1"
    }

    fn operation_id(&self) -> &'static str {
        "operation-1"
    }

    fn turn_id(&self) -> &'static str {
        "turn-1"
    }

    fn get_memo(
        &self,
        _name: &str,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<JsonValue>, String>> {
        Box::pin(async { Ok(None) })
    }

    fn set_memo(
        &self,
        _name: &str,
        _value: Option<JsonValue>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
}

fn usage(input: u64, output: u64, total_tokens: u64) -> Usage {
    Usage {
        input,
        output,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens,
        cost: UsageCost::default(),
    }
}

fn immediate_text(outcome: &Result<PreparedToolCall, ImmediateToolOutcome>) -> String {
    match outcome {
        Err(immediate) => text(&immediate.result),
        Ok(_) => String::new(),
    }
}

/// Prepares arguments before validation and preserves the provider call.
#[test]
fn prepares_arguments_before_validation_and_preserves_the_provider_call() {
    let provider_call = call(&json!({ "legacy": "prepared" }));
    let mut echoing = tool(None);
    echoing.prepare_arguments = Some(Arc::new(|args: &JsonValue| {
        Ok(json!({ "value": args["legacy"] }))
    }));
    let prepared =
        prepare_tool_call(&provider_call, &[echoing]).expect("the prepared call settles");

    assert_eq!(prepared.tool_call, provider_call);
    assert_eq!(prepared.args, json!({ "value": "prepared" }));
    assert_eq!(
        provider_call.arguments,
        json!({ "legacy": "prepared" })
            .as_object()
            .expect("an arguments object")
            .clone()
    );
}

/// Returns immediate errors for unknown tools, preparation throws, and
/// invalid arguments.
#[test]
fn returns_immediate_errors_for_unknown_tools_preparation_throws_and_invalid_arguments() {
    let unknown = prepare_tool_call(&call(&json!({ "value": "input" })), &[]);
    let preparation_failure = {
        let mut failing = tool(None);
        failing.prepare_arguments = Some(Arc::new(
            |_args: &JsonValue| -> Result<JsonValue, AgentToolError> {
                Err("cannot prepare".into())
            },
        ));
        prepare_tool_call(&call(&json!({ "value": "input" })), &[failing])
    };
    let invalid = prepare_tool_call(&call(&json!({})), &[tool(None)]);

    assert_eq!(immediate_text(&unknown), "Tool \"echo\" is unavailable");
    assert!(
        unknown
            .as_ref()
            .map_or_else(|immediate| immediate.result.details.is_null(), |_| false)
    );
    assert_eq!(immediate_text(&preparation_failure), "cannot prepare");
    assert!(immediate_text(&invalid).contains("Validation failed for tool \"echo\""));
}

/// Blocks calls and revalidates replacement arguments; each pass rebuilds
/// the prepared value, `apply_before_tool_decision` consuming it.
#[test]
fn blocks_calls_and_revalidates_replacement_arguments() {
    let prepared = || {
        prepare_tool_call(&call(&json!({ "value": "input" })), &[tool(None)])
            .expect("the prepared call settles")
    };

    let blocked = apply_before_tool_decision(
        prepared(),
        Some(&BeforeToolResult {
            args: None,
            block: Some(ToolBlock {
                reason: "denied".to_owned(),
                terminate: Some(true),
            }),
        }),
    );
    let replaced = apply_before_tool_decision(
        prepared(),
        Some(&BeforeToolResult {
            args: Some(std::collections::BTreeMap::from([(
                "value".to_owned(),
                json!("replacement"),
            )])),
            block: None,
        }),
    );
    let invalid = apply_before_tool_decision(
        prepared(),
        Some(&BeforeToolResult {
            args: Some(std::collections::BTreeMap::new()),
            block: None,
        }),
    );

    let ImmediateToolOutcome {
        result, terminate, ..
    } = blocked.expect_err("a blocked call is immediate");
    assert!(terminate);
    assert_eq!(text(&result), "denied");
    assert_eq!(
        replaced.ok().map(|cleared| cleared.args),
        Some(json!({ "value": "replacement" }))
    );
    assert!(invalid.is_err());
}

/// Executes with updates and passes the gate's signal to the tool; calls
/// after the tool settles cannot exist in Rust — the callback reference
/// dies with the execution future — so the forwarded count pins the latch.
#[tokio::test]
async fn executes_with_updates_passes_the_signal_and_ignores_late_updates() {
    let gate = effect_gate();
    let received_signal = Arc::new(Mutex::new(None::<crate::harness::context::AbortSignal>));
    let signal_handle = Arc::clone(&received_signal);
    let updates: Arc<Mutex<Vec<AgentToolResult>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_handle = Arc::clone(&updates);
    let execute: Arc<AgentHarnessToolExecuteFn> = Arc::new(
        move |_tool_call_id: &str,
              args: &JsonValue,
              on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
              _tool_context: ToolContext,
              _invocation: &dyn AgentHarnessToolInvocation,
              context: &Context| {
            *signal_handle.lock().expect("signal lock") = context.abort_signal();
            if let Some(on_update) = on_update {
                on_update(
                    &AgentToolResult {
                        content: vec![AgentToolContent::Text(TextContent {
                            text: "partial".to_owned(),
                            text_signature: None,
                        })],
                        details: json!({ "value": args["value"] }),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    },
                    None,
                );
            }
            let value = args["value"].as_str().unwrap_or_default().to_owned();
            Box::pin(async move {
                Ok(AgentToolResult {
                    content: vec![AgentToolContent::Text(TextContent {
                        text: "done".to_owned(),
                        text_signature: None,
                    })],
                    details: json!({ "value": value }),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        },
    );
    let cleared = clear_prepared(prepare_tool_call(
        &call(&json!({ "value": "input" })),
        &[tool(Some(execute))],
    ));
    let outer = move |partial: &AgentToolResult,
                      _options: Option<AgentHarnessToolUpdateOptions>| {
        updates_handle
            .lock()
            .expect("updates lock")
            .push(partial.clone());
    };

    let result = execute_tool_call(
        &cleared,
        &gate,
        Some(&outer),
        None,
        &StubInvocation,
        &background_context(),
    )
    .await
    .expect("the execution settles");

    assert!(!result.is_error);
    assert_eq!(text(&result.result), "done");
    let updates = updates.lock().expect("updates lock");
    assert_eq!(updates.len(), 1);
    assert_eq!(text(&updates[0]), "partial");
    let received = received_signal
        .lock()
        .expect("signal lock")
        .clone()
        .expect("the tool received a signal");
    assert!(!received.aborted());
}

/// Converts a tool's failure to error output; upstream's synchronous and
/// asynchronous throw rows share the `Err` channel here.
#[tokio::test]
async fn converts_tool_throws_to_error_output() {
    let cleared = clear_prepared(prepare_tool_call(
        &call(&json!({ "value": "input" })),
        &[tool(Some(Arc::new(
            |_tool_call_id: &str,
             _args: &JsonValue,
             _on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
             _tool_context: ToolContext,
             _invocation: &dyn AgentHarnessToolInvocation,
             _context: &Context| {
                let error: AgentToolError = "tool failed".into();
                Box::pin(async move { Err(error) })
            },
        )))],
    ));

    let result = execute_tool_call(
        &cleared,
        &effect_gate(),
        None,
        None,
        &StubInvocation,
        &background_context(),
    )
    .await
    .expect("the execution settles");

    assert!(result.is_error);
    assert_eq!(text(&result.result), "tool failed");
}

/// Lets an abort-first gate refusal escape without invoking the tool.
#[tokio::test]
async fn lets_abort_first_gate_refusal_escape_without_invoking_the_tool() {
    let calls = Arc::new(AtomicU64::new(0));
    let cleared = clear_prepared(prepare_tool_call(
        &call(&json!({ "value": "input" })),
        &[tool(Some(counting_execute(Arc::clone(&calls))))],
    ));
    let (gate, control) = create_gate();
    let (sender, receiver) = tokio::sync::watch::channel(());
    drop(sender);
    control.begin_abort(receiver);

    let rejection = execute_tool_call(
        &cleared,
        &gate,
        None,
        None,
        &StubInvocation,
        &background_context(),
    )
    .await
    .expect_err("an abort-first gate refuses admission");

    assert!(matches!(
        rejection,
        ToolExecutionRejection::Gate(GateRejection::AbortRequested(_))
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

/// Applies patches field by field and constructs the tool-result message.
#[test]
fn applies_patches_field_by_field_and_constructs_the_tool_result_message() {
    let cleared = clear_prepared(prepare_tool_call(
        &call(&json!({ "value": "input" })),
        &[tool(None)],
    ));
    let original_usage = usage(1, 2, 3);
    let mut replacement_usage = original_usage;
    replacement_usage.input = 5;
    replacement_usage.total_tokens = 7;
    let executed = ExecutedToolCall {
        result: AgentToolResult {
            content: vec![text_content("original")],
            details: json!({ "original": true }),
            usage: Some(original_usage),
            added_tool_names: Some(vec!["new-tool".to_owned()]),
            terminate: None,
        },
        is_error: true,
    };

    let finalized = finalize_tool_call(
        &cleared,
        executed,
        Some(AfterToolResult {
            content: Some(vec![text_content("patched")]),
            details: Some(json!({ "patched": true })),
            usage: Some(replacement_usage),
            is_error: Some(false),
            terminate: Some(true),
        }),
    );
    let before = now_ms();
    let message = create_tool_result_message(&finalized);

    assert!(!finalized.is_error);
    assert!(finalized.terminate);
    assert_eq!(finalized.result.content, vec![text_content("patched")]);
    assert_eq!(finalized.result.details, json!({ "patched": true }));
    assert_eq!(finalized.result.usage, Some(replacement_usage));
    assert_eq!(
        finalized.result.added_tool_names,
        Some(vec!["new-tool".to_owned()])
    );
    assert_eq!(finalized.result.terminate, Some(true));
    assert_eq!(message.tool_call_id, "call-1");
    assert_eq!(message.tool_name, "echo");
    assert_eq!(message.content, vec![text_content("patched")]);
    assert_eq!(message.details, Some(json!({ "patched": true })));
    assert_eq!(message.usage, Some(replacement_usage));
    assert_eq!(message.added_tool_names, Some(vec!["new-tool".to_owned()]));
    assert!(!message.is_error);
    assert!(message.timestamp >= before);
}

/// Normalizes an untyped tool's empty content to the message's empty list;
/// upstream's undefined `content` restates as an empty vec.
#[test]
fn normalizes_missing_content_from_untyped_tools() {
    let cleared = clear_prepared(prepare_tool_call(
        &call(&json!({ "value": "input" })),
        &[tool(None)],
    ));
    let finalized = finalize_tool_call(
        &cleared,
        ExecutedToolCall {
            result: AgentToolResult {
                content: Vec::new(),
                details: json!({}),
                usage: None,
                added_tool_names: None,
                terminate: None,
            },
            is_error: false,
        },
        None,
    );

    assert!(create_tool_result_message(&finalized).content.is_empty());
}

// --- boundary tests: the branches the upstream suite leaves untested ---

/// A closed gate refuses execution with its error.
#[tokio::test]
async fn a_closed_gate_refuses_execution_with_its_error() {
    let cleared = clear_prepared(prepare_tool_call(
        &call(&json!({ "value": "input" })),
        &[tool(None)],
    ));
    let (gate, control) = create_gate();
    control.close("closed".to_owned());

    let rejection = execute_tool_call(
        &cleared,
        &gate,
        None,
        None,
        &StubInvocation,
        &background_context(),
    )
    .await
    .expect_err("a closed gate refuses admission");

    assert!(matches!(
        rejection,
        ToolExecutionRejection::Gate(GateRejection::Closed(_))
    ));
}

/// A caller context that arrived aborted rejects the execution before the
/// tool runs, with the caller's abort reason.
#[tokio::test]
async fn a_pre_aborted_caller_context_rejects_the_execution_before_the_tool_runs() {
    let calls = Arc::new(AtomicU64::new(0));
    let cleared = clear_prepared(prepare_tool_call(
        &call(&json!({ "value": "input" })),
        &[tool(Some(counting_execute(Arc::clone(&calls))))],
    ));
    let (gate, _control) = create_gate();
    let (aborted_context, controller) = with_cancel(&background_context());
    controller.abort("invocation aborted");

    let rejection = execute_tool_call(
        &cleared,
        &gate,
        None,
        None,
        &StubInvocation,
        &aborted_context,
    )
    .await
    .expect_err("a pre-aborted context refuses execution");

    assert!(matches!(
        rejection,
        ToolExecutionRejection::Aborted(AbortReason::Caller(reason)) if reason == "invocation aborted"
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

/// A preparation shim returning a non-object fails as the shim's own error.
#[test]
fn a_non_object_preparation_result_fails_as_the_shims_error() {
    let mut malformed = tool(None);
    malformed.prepare_arguments = Some(Arc::new(
        |_args: &JsonValue| -> Result<JsonValue, AgentToolError> { Ok(json!("not-an-object")) },
    ));

    let outcome = prepare_tool_call(&call(&json!({ "legacy": "prepared" })), &[malformed]);
    let immediate = outcome.expect_err("a non-object shim result is immediate");
    assert_eq!(
        text(&immediate.result),
        "prepareArguments must return an object"
    );
}

/// A patch carrying only `isError` keeps every executed field and flips
/// the error flag.
#[test]
fn an_is_error_only_patch_keeps_every_executed_field() {
    let cleared = clear_prepared(prepare_tool_call(
        &call(&json!({ "value": "input" })),
        &[tool(None)],
    ));
    let finalized = finalize_tool_call(
        &cleared,
        ExecutedToolCall {
            result: AgentToolResult {
                content: vec![text_content("original")],
                details: json!({ "original": true }),
                usage: Some(usage(1, 2, 3)),
                added_tool_names: None,
                terminate: None,
            },
            is_error: false,
        },
        Some(AfterToolResult {
            is_error: Some(true),
            ..AfterToolResult::default()
        }),
    );

    assert!(finalized.is_error);
    assert_eq!(text(&finalized.result), "original");
    assert_eq!(finalized.result.details, json!({ "original": true }));
    assert_eq!(finalized.result.usage, Some(usage(1, 2, 3)));
    assert!(!finalized.terminate);
}

/// A missing patch passes the executed output through untouched.
#[test]
fn a_missing_patch_passes_the_executed_output_through() {
    let cleared = clear_prepared(prepare_tool_call(
        &call(&json!({ "value": "input" })),
        &[tool(None)],
    ));
    let finalized = finalize_tool_call(
        &cleared,
        ExecutedToolCall {
            result: AgentToolResult {
                content: vec![text_content("original")],
                details: JsonValue::Null,
                usage: Some(usage(1, 2, 3)),
                added_tool_names: Some(vec!["added".to_owned()]),
                terminate: Some(true),
            },
            is_error: true,
        },
        None,
    );

    assert!(finalized.is_error);
    assert!(finalized.terminate);
    assert_eq!(text(&finalized.result), "original");
    assert_eq!(finalized.result.details, JsonValue::Null);
}

/// Reconstructing a tool result from a message spreads the optional fields
/// only when the message carries them, and `terminate` only when asked.
#[test]
fn tool_result_from_message_spreads_the_optional_fields_conditionally() {
    let bare = ToolResultMessage {
        tool_call_id: "call-1".to_owned(),
        tool_name: "echo".to_owned(),
        content: vec![text_content("raw")],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: true,
        timestamp: 5,
    };

    let without = tool_result_from_message(&bare, false);
    assert_eq!(text(&without), "raw");
    assert!(without.details.is_null());
    assert_eq!(without.usage, None);
    assert_eq!(without.added_tool_names, None);
    assert_eq!(without.terminate, None);

    let mut staged = bare;
    staged.details = Some(json!({ "staged": true }));
    staged.usage = Some(usage(1, 2, 3));
    staged.added_tool_names = Some(vec!["added".to_owned()]);
    let with = tool_result_from_message(&staged, true);
    assert_eq!(with.details, json!({ "staged": true }));
    assert_eq!(with.usage, Some(usage(1, 2, 3)));
    assert_eq!(with.added_tool_names, Some(vec!["added".to_owned()]));
    assert_eq!(with.terminate, Some(true));
}

/// The message omits null details and empty `addedToolNames`, and carries
/// the finalized error flag.
#[test]
fn the_message_omits_null_details_and_empty_added_tool_names() {
    let finalized = FinalizedToolCall {
        tool_call: call(&json!({ "value": "input" })),
        result: AgentToolResult {
            content: Vec::new(),
            details: JsonValue::Null,
            usage: None,
            added_tool_names: Some(Vec::new()),
            terminate: None,
        },
        is_error: true,
        terminate: false,
    };

    let message = create_tool_result_message(&finalized);
    assert!(message.details.is_none());
    assert!(message.added_tool_names.is_none());
    assert!(message.usage.is_none());
    assert!(message.is_error);
}

/// The update wrapper forwards the checkpoint option untouched while
/// accepting.
#[tokio::test]
async fn the_update_wrapper_forwards_the_checkpoint_option_while_accepting() {
    let options_seen = Arc::new(Mutex::new(
        Vec::<Option<AgentHarnessToolUpdateOptions>>::new(),
    ));
    let options_handle = Arc::clone(&options_seen);
    let cleared = clear_prepared(prepare_tool_call(
        &call(&json!({ "value": "input" })),
        &[tool(Some(Arc::new(
            move |_tool_call_id: &str,
                  _args: &JsonValue,
                  on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
                  _tool_context: ToolContext,
                  _invocation: &dyn AgentHarnessToolInvocation,
                  _context: &Context| {
                if let Some(on_update) = on_update {
                    on_update(
                        &empty_result(),
                        Some(AgentHarnessToolUpdateOptions { checkpoint: true }),
                    );
                }
                Box::pin(async move { Ok(empty_result()) })
            },
        )))],
    ));
    let outer = move |partial: &AgentToolResult, options: Option<AgentHarnessToolUpdateOptions>| {
        let _ = partial;
        options_handle.lock().expect("options lock").push(options);
    };

    let _ = execute_tool_call(
        &cleared,
        &effect_gate(),
        Some(&outer),
        None,
        &StubInvocation,
        &background_context(),
    )
    .await
    .expect("the execution settles");

    let options = options_seen.lock().expect("options lock");
    assert_eq!(options.len(), 1);
    assert_eq!(options[0].map(|options| options.checkpoint), Some(true));
}

/// The rejection surfaces render their shapes: the gate refusal's message
/// matches the gate's own rendering, and the abort reason displays.
#[test]
fn the_rejection_surfaces_render_their_shapes() {
    let (sender, receiver) = tokio::sync::watch::channel(());
    drop(sender);
    let abort_first = ToolExecutionRejection::Gate(GateRejection::AbortRequested(
        crate::harness::gate::AbortRequested {
            cancellation: receiver,
        },
    ));
    assert_eq!(abort_first.to_string(), "Abort requested");

    let closed = ToolExecutionRejection::Gate(GateRejection::Closed(
        crate::harness::gate::GateClosedError("the drive pass closed".to_owned()),
    ));
    assert_eq!(closed.to_string(), "the drive pass closed");

    let aborted =
        ToolExecutionRejection::Aborted(AbortReason::Caller("invocation aborted".to_owned()));
    assert_eq!(aborted.to_string(), "invocation aborted");
    let plain = ToolExecutionRejection::Aborted(AbortReason::Aborted);
    assert_eq!(plain.to_string(), "The operation was aborted");
}

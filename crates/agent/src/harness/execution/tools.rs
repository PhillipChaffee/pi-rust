//! The harness tool-execution pipeline, ported from upstream
//! `src/harness/execution/tools.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The stages restate as functions over owned values: upstream's union
//! returns (`PreparedToolCall | ImmediateToolOutcome`) restate as
//! [`std::result::Result`] with the immediate outcome on the error side.
//! The before- and after-tool aggregates restate as the hook layer's
//! [`BeforeToolResult`] / [`AfterToolResult`] — upstream defines the same
//! shapes twice (once in `hooks.ts`, once here); the port keeps one type.
//!
//! The late-update latch keeps its runtime form: the forwarding wrapper
//! stops passing updates once the tool's future settles. Rust's borrow
//! lattice already forbids a tool from retaining the callback past
//! settlement, so the latch guards only same-future racing pushes, exactly
//! like the agent-loop port's accepting latch.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pi_ai::auth::resolve::now_ms;
use pi_ai::types::{TextContent, ToolCall, ToolResultMessage};
use pi_ai::utils::validation::validate_tool_arguments;
use serde_json::Value as JsonValue;

use crate::harness::agent_harness::{AfterToolResult, BeforeToolResult};
use crate::harness::context::{Context, with_abort_signal};
use crate::harness::gate::{Gate, GateRejection};
use crate::harness::types::{
    AgentHarnessTool, AgentHarnessToolInvocation, AgentHarnessToolUpdateCallback,
    AgentHarnessToolUpdateOptions, ToolContext,
};
use crate::types::{AgentToolCall, AgentToolContent, AgentToolResult};

/// A tool call whose tool exists and whose prepared arguments passed
/// validation, upstream's `PreparedToolCall`.
#[derive(Debug)]
pub struct PreparedToolCall {
    /// The provider's call, untouched.
    pub tool_call: AgentToolCall,
    /// The resolved harness tool.
    pub tool: AgentHarnessTool,
    /// The validated (and coerced) arguments.
    pub args: JsonValue,
}

/// Synthetic result produced without crossing the external tool-effect
/// boundary, upstream's `ImmediateToolOutcome`.
///
/// `isError` is always true — the variant is that statement — so the port
/// carries only the terminate hint beside the result.
#[derive(Debug)]
pub struct ImmediateToolOutcome {
    /// The call the outcome answers.
    pub tool_call: AgentToolCall,
    /// The synthetic error result.
    pub result: AgentToolResult,
    /// Whether the immediate error terminates the run.
    pub terminate: bool,
}

/// A prepared call cleared for durable intent publication and execution,
/// upstream's `ClearedToolCall`.
#[derive(Debug)]
pub struct ClearedToolCall {
    /// The provider's call, untouched.
    pub tool_call: AgentToolCall,
    /// The resolved harness tool.
    pub tool: AgentHarnessTool,
    /// The validated (and coerced) arguments.
    pub args: JsonValue,
}

/// Raw phase-two tool output before after-tool patching, upstream's
/// `ExecutedToolCall`.
#[derive(Debug)]
pub struct ExecutedToolCall {
    /// The tool's result (the error result when execution failed).
    pub result: AgentToolResult,
    /// Whether the execution failed.
    pub is_error: bool,
}

/// Final tool output ready to become a durable tool-result message,
/// upstream's `FinalizedToolCall`.
#[derive(Debug)]
pub struct FinalizedToolCall {
    /// The provider's call, untouched.
    pub tool_call: AgentToolCall,
    /// The patched result.
    pub result: AgentToolResult,
    /// Whether the execution failed, after the patch.
    pub is_error: bool,
    /// Whether the run terminates after this call.
    pub terminate: bool,
}

/// Why [`execute_tool_call`] refused to run a tool, the restatement of
/// upstream's thrown rejections.
#[derive(Debug)]
pub enum ToolExecutionRejection {
    /// The effect gate refused admission.
    Gate(GateRejection),
    /// The admitted context arrived aborted, upstream's `throwIfAborted`
    /// on the derived context.
    Aborted(crate::harness::context::AbortReason),
}

impl std::fmt::Display for ToolExecutionRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gate(rejection) => std::fmt::Display::fmt(rejection_display(rejection), f),
            Self::Aborted(reason) => std::fmt::Display::fmt(&reason, f),
        }
    }
}

/// The rejection's message, upstream's thrown error text.
fn rejection_display(rejection: &GateRejection) -> &str {
    match rejection {
        GateRejection::AbortRequested(_) => "Abort requested",
        GateRejection::Closed(closed) => &closed.0,
    }
}

/// The synthetic error result, upstream's `createErrorToolResult`.
///
/// Upstream leaves `details` undefined; the port restates absent JSON as
/// null, which the message builder omits.
fn create_error_tool_result(message: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![AgentToolContent::Text(TextContent {
            text: message.to_owned(),
            text_signature: None,
        })],
        details: JsonValue::Null,
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

fn immediate_error(call: &AgentToolCall, message: &str, terminate: bool) -> ImmediateToolOutcome {
    ImmediateToolOutcome {
        tool_call: call.clone(),
        result: create_error_tool_result(message),
        terminate,
    }
}

/// Resolve a tool, apply its deterministic argument preparation, and
/// validate the result, upstream's `prepareToolCall`.
///
/// The returned `tool_call` is always the provider's original call; only
/// the validated arguments carry the preparation's effect.
///
/// # Errors
/// The immediate outcome when the tool is unknown, preparation throws, or
/// validation fails.
#[expect(
    clippy::result_large_err,
    reason = "the immediate outcome carries its full tool result by value, mirroring the upstream shape; the agent-loop port's Preparation enum states the same"
)]
pub fn prepare_tool_call(
    call: &AgentToolCall,
    tools: &[AgentHarnessTool],
) -> Result<PreparedToolCall, ImmediateToolOutcome> {
    let Some(tool) = tools.iter().find(|candidate| candidate.name() == call.name) else {
        let name = serde_json::to_string(&call.name).unwrap_or_default();
        return Err(immediate_error(
            call,
            &format!("Tool {name} is unavailable"),
            false,
        ));
    };

    let prepared_arguments = match &tool.prepare_arguments {
        Some(prepare) => {
            let shimmed = prepare(&JsonValue::Object(call.arguments.clone()))
                .map_err(|error| immediate_error(call, &error.to_string(), false))?;
            // The shim contract requires an object matching the tool's
            // schema; anything else is the shim's own error.
            let Some(shimmed) = shimmed.as_object() else {
                return Err(immediate_error(
                    call,
                    "prepareArguments must return an object",
                    false,
                ));
            };
            shimmed.clone()
        }
        None => call.arguments.clone(),
    };
    let prepared_call = if prepared_arguments == call.arguments {
        call.clone()
    } else {
        ToolCall {
            arguments: prepared_arguments,
            ..call.clone()
        }
    };
    let args = validate_tool_arguments(&tool.tool, &prepared_call)
        .map_err(|error| immediate_error(call, &error.to_string(), false))?;
    Ok(PreparedToolCall {
        tool_call: call.clone(),
        tool: tool.clone(),
        args,
    })
}

/// Apply an explicit hook decision and revalidate replacement arguments,
/// upstream's `applyBeforeToolDecision`.
///
/// # Errors
/// The immediate outcome when the decision blocks the call or the
/// replacement arguments fail validation.
#[expect(
    clippy::result_large_err,
    reason = "the immediate outcome carries its full tool result by value, mirroring the upstream shape"
)]
pub fn apply_before_tool_decision(
    prepared: PreparedToolCall,
    decision: Option<&BeforeToolResult>,
) -> Result<ClearedToolCall, ImmediateToolOutcome> {
    let call = prepared.tool_call.clone();
    if let Some(block) = decision.and_then(|decision| decision.block.as_ref()) {
        return Err(immediate_error(
            &call,
            &block.reason,
            block.terminate == Some(true),
        ));
    }
    let Some(replacement) = decision.and_then(|decision| decision.args.as_ref()) else {
        return Ok(ClearedToolCall {
            tool_call: call,
            tool: prepared.tool,
            args: prepared.args,
        });
    };
    let replacement_call = ToolCall {
        arguments: replacement
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        ..call.clone()
    };
    let args = validate_tool_arguments(&prepared.tool.tool, &replacement_call)
        .map_err(|error| immediate_error(&call, &error.to_string(), false))?;
    Ok(ClearedToolCall {
        tool_call: call,
        tool: prepared.tool,
        args,
    })
}

/// Execute one cleared external tool effect, converting expected tool
/// throws to error output, upstream's `executeToolCall`.
///
/// The tool receives a context derived with the gate's signal, and its
/// partial-result updates forward only while the execution runs.
///
/// # Errors
/// [`ToolExecutionRejection::Gate`] when the gate refuses admission and
/// [`ToolExecutionRejection::Aborted`] when the derived context arrived
/// aborted; the tool's own failures are error output, not rejections.
pub async fn execute_tool_call(
    call: &ClearedToolCall,
    gate: &Gate,
    on_update: Option<AgentHarnessToolUpdateCallback<'_>>,
    tool_context: ToolContext,
    invocation: &dyn AgentHarnessToolInvocation,
    context: &Context,
) -> Result<ExecutedToolCall, ToolExecutionRejection> {
    let admitted_context = gate
        .admit(|| with_abort_signal(gate.signal().clone(), context))
        .map_err(ToolExecutionRejection::Gate)?;
    if let Some(signal) = admitted_context.abort_signal()
        && let Some(reason) = signal.reason()
    {
        return Err(ToolExecutionRejection::Aborted(reason));
    }

    let accepting = Arc::new(AtomicBool::new(true));
    let forward = on_update.map(|outer| {
        let accepting = Arc::clone(&accepting);
        move |partial: &AgentToolResult, options: Option<AgentHarnessToolUpdateOptions>| {
            if accepting.load(Ordering::Relaxed) {
                outer(partial, options);
            }
        }
    });
    let update_callback: Option<AgentHarnessToolUpdateCallback<'_>> =
        forward.as_ref().map(|forward| {
            let callback: AgentHarnessToolUpdateCallback<'_> = forward;
            callback
        });

    let result = (call.tool.execute)(
        &call.tool_call.id,
        &call.args,
        update_callback,
        tool_context,
        invocation,
        &admitted_context,
    )
    .await;
    accepting.store(false, Ordering::Relaxed);
    match result {
        Ok(result) => Ok(ExecutedToolCall {
            result,
            is_error: false,
        }),
        Err(error) => Ok(ExecutedToolCall {
            result: create_error_tool_result(&error.to_string()),
            is_error: true,
        }),
    }
}

/// Apply an after-tool patch field by field, upstream's `finalizeToolCall`.
///
/// Every patch field present replaces the executed field; absent fields
/// keep the executed value. `is_error` reads the patch first and falls
/// back to the executed flag.
#[must_use]
pub fn finalize_tool_call(
    call: &ClearedToolCall,
    executed: ExecutedToolCall,
    patch: Option<AfterToolResult>,
) -> FinalizedToolCall {
    let AfterToolResult {
        content,
        details,
        is_error,
        usage,
        terminate,
    } = patch.unwrap_or_default();
    let mut result = executed.result;
    if let Some(content) = content {
        result.content = content;
    }
    if let Some(details) = details {
        result.details = details;
    }
    if let Some(usage) = usage {
        result.usage = Some(usage);
    }
    if let Some(terminate) = terminate {
        result.terminate = Some(terminate);
    }
    FinalizedToolCall {
        tool_call: call.tool_call.clone(),
        is_error: is_error.unwrap_or(executed.is_error),
        terminate: result.terminate == Some(true),
        result,
    }
}

/// Reconstruct the canonical tool result represented by a staged
/// transcript message, upstream's `toolResultFromMessage`.
#[must_use]
pub fn tool_result_from_message(message: &ToolResultMessage, terminate: bool) -> AgentToolResult {
    AgentToolResult {
        content: message.content.clone(),
        details: message.details.clone().unwrap_or(JsonValue::Null),
        usage: message.usage,
        added_tool_names: message.added_tool_names.clone(),
        terminate: terminate.then_some(true),
    }
}

/// Convert finalized tool output to the provider-facing transcript
/// message, upstream's `createToolResultMessage`.
///
/// Absent details (null), empty `addedToolNames`, and the timestamp are
/// normalized per upstream's conditional spreads: the message omits them
/// when the tool result carries nothing.
#[must_use]
pub fn create_tool_result_message(call: &FinalizedToolCall) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: call.tool_call.id.clone(),
        tool_name: call.tool_call.name.clone(),
        content: call.result.content.clone(),
        details: (!call.result.details.is_null()).then(|| call.result.details.clone()),
        usage: call.result.usage,
        added_tool_names: call
            .result
            .added_tool_names
            .clone()
            .filter(|names| !names.is_empty()),
        is_error: call.is_error,
        timestamp: now_ms(),
    }
}

#[cfg(test)]
mod tests;

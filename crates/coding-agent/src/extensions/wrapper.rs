//! Tool wrappers for extension-registered tools, upstream's
//! `src/core/extensions/wrapper.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The wrapper only adapts execution: extension tools run through the
//! definition wrapper and restate upstream's added-tool-names tracking —
//! the runner's active-tools read before and after the execute decides
//! which tool names appeared, and `addedToolNames` carries the union.
//! Tool-call and tool-result interception rides AgentSession's core hooks;
//! the extension-context threading into execution rides the session's
//! tool-execution slice.

use std::collections::HashSet;
use std::sync::Arc;

use pi_agent_core::harness::types::{AgentHarnessTool, AgentHarnessToolExecuteFn};
use pi_agent_core::types::AgentToolResult;

use crate::extensions::runner::ExtensionRunner;
use crate::extensions::runtime::RegisteredTool;
use crate::tools::tool_definition_wrapper::wrap_cwd_tool;

/// Wrap one registered tool into the harness tool, upstream's
/// `wrapRegisteredTool`: the runner's active-tool list is read around the
/// execute, and a stale runtime fails the call as upstream's throw would.
#[must_use]
pub fn wrap_registered_tool(
    registered_tool: RegisteredTool,
    runner: Arc<ExtensionRunner>,
) -> AgentHarnessTool {
    let tool = wrap_cwd_tool(registered_tool.definition);
    let execute: Arc<AgentHarnessToolExecuteFn> = Arc::clone(&tool.execute);
    AgentHarnessTool {
        tool: tool.tool,
        label: tool.label,
        prepare_arguments: tool.prepare_arguments,
        execution_mode: tool.execution_mode,
        replay: None,
        execute: Arc::new(
            move |tool_call_id: &str,
                  params: &serde_json::Value,
                  on_update,
                  tool_context,
                  invocation,
                  context| {
                let runner = Arc::clone(&runner);
                let execute = Arc::clone(&execute);
                Box::pin(async move {
                    let active_before = runner.active_tools().map_err(|stale| {
                        pi_agent_core::types::AgentToolError::from(std::io::Error::other(stale))
                    })?;
                    let result: AgentToolResult = (execute)(
                        tool_call_id,
                        params,
                        on_update,
                        tool_context,
                        invocation,
                        context,
                    )
                    .await?;
                    let active_after = runner.active_tools().map_err(|stale| {
                        pi_agent_core::types::AgentToolError::from(std::io::Error::other(stale))
                    })?;
                    Ok(fold_added_tools(result, &active_before, &active_after))
                })
            },
        ),
    }
}

/// Restate upstream's added-tool-names fold: the pre-execute snapshot and
/// the post-execute read decide the fresh names; a before-tool
/// disappearing keeps the result unchanged, and the union of the fresh
/// names dedupes into `addedToolNames`.
fn fold_added_tools(
    mut result: AgentToolResult,
    active_before: &[String],
    active_after: &[String],
) -> AgentToolResult {
    if !active_before.iter().all(|name| active_after.contains(name)) {
        return result;
    }
    let before: HashSet<&str> = active_before.iter().map(String::as_str).collect();
    let added: Vec<String> = active_after
        .iter()
        .filter(|name| !before.contains(name.as_str()))
        .cloned()
        .collect();
    if added.is_empty() {
        return result;
    }
    let mut merged: Vec<String> = Vec::new();
    for name in result
        .added_tool_names
        .clone()
        .unwrap_or_default()
        .into_iter()
        .chain(added)
    {
        if !merged.contains(&name) {
            merged.push(name);
        }
    }
    result.added_tool_names = Some(merged);
    result
}

/// Wrap every registered tool into the harness tools, upstream's
/// `wrapRegisteredTools`.
#[must_use]
pub fn wrap_registered_tools(
    registered_tools: &[RegisteredTool],
    runner: &Arc<ExtensionRunner>,
) -> Vec<AgentHarnessTool> {
    registered_tools
        .iter()
        .map(|tool| wrap_registered_tool(tool.clone(), Arc::clone(runner)))
        .collect()
}

#[cfg(test)]
mod tests;

//! Wraps a tool definition into a harness tool for the core runtime,
//! ported from upstream `src/core/tools/tool-definition-wrapper.ts`.
//!
//! The erased bridge restates upstream's `ctx` threading: the harness tool
//! receives the erased context slot; the wrapper downcasts it to the
//! definition's context type and falls back to the optional factory's value
//! (upstream's `ctxFactory`) when the slot is empty. The abort signal rides
//! the chord context, upstream's `signal` parameter.

use std::sync::Arc;

use pi_agent_core::harness::types::{AgentHarnessTool, AgentHarnessToolExecuteFn};
use pi_ai::types::Tool;

use crate::extensions::types::{CwdContext, ExtensionContext, ToolDefinition};

/// Erase the concrete context behind the trait object the tool execute
/// closure passes.
fn erase_context<C: ExtensionContext>(context: &C) -> &dyn ExtensionContext {
    context
}

/// Wrap a definition into the harness tool over the standalone `CwdContext`
/// default, the shape upstream's per-tool `createXTool` one-liners share.
#[must_use]
pub fn wrap_cwd_tool(definition: ToolDefinition) -> AgentHarnessTool {
    wrap_tool_definition::<CwdContext>(definition, None)
}

/// Wrap a [`ToolDefinition`] into a harness tool for the core runtime,
/// upstream's `wrapToolDefinition`.
///
/// `ctx_factory` supplies the extension context when the erased slot is
/// empty, upstream's `ctxFactory?.()`.
#[must_use]
pub fn wrap_tool_definition<C: ExtensionContext>(
    definition: ToolDefinition,
    ctx_factory: Option<Arc<C>>,
) -> AgentHarnessTool {
    let definition = Arc::new(definition);
    let factory = Arc::new(ctx_factory);
    let definition_for_surface = Arc::clone(&definition);
    let execute: Arc<AgentHarnessToolExecuteFn> = Arc::new(
        move |tool_call_id: &str,
              params: &serde_json::Value,
              on_update,
              tool_context,
              _invocation,
              context| {
            let definition = Arc::clone(&definition);
            let factory = Arc::clone(&factory);
            Box::pin(async move {
                let signal = context.abort_signal();
                let typed: Option<Arc<C>> = tool_context
                    .as_ref()
                    .and_then(|ctx| Arc::clone(ctx).downcast::<C>().ok());
                let ctx: Option<&C> = typed.as_deref().or_else(|| factory.as_ref().as_deref());
                let extension_ctx: Option<&dyn ExtensionContext> = ctx.map(erase_context::<C>);
                (definition.execute)(
                    tool_call_id,
                    params,
                    signal.as_ref(),
                    on_update,
                    extension_ctx,
                )
                .await
            })
        },
    );
    AgentHarnessTool {
        tool: Tool {
            name: definition_for_surface.name.clone(),
            description: definition_for_surface.description.clone(),
            parameters: definition_for_surface.parameters.clone(),
            constrained_sampling: definition_for_surface.constrained_sampling.clone(),
        },
        label: definition_for_surface.label.clone(),
        prepare_arguments: definition_for_surface.prepare_arguments.clone(),
        execute,
        replay: None,
        execution_mode: definition_for_surface.execution_mode,
    }
}

//! The extension-tool type surface the built-in tools execute over, upstream
//! `src/core/extensions/types.ts`.
//!
//! Map child "pi-coding-agent: built-in coding tools"; the extension loading,
//! event, and registration machinery ride the extension-system ticket.
//!
//! [`ToolDefinition`] is `registerTool()`'s shape: the declarative fields the
//! harness tool carries plus the system-prompt contributions and (with the
//! theme ticket) the render hooks. [`ExtensionContext`] is the context tools
//! receive; the trait carries the surface the built-in tools read — the cwd,
//! the model, the thinking level, and the session identifiers the bash spawn
//! environment exposes — with the full interactive surface landing with its
//! consumers. [`CwdContext`] is the minimal `{ cwd }` shape upstream's
//! `fakeCtx` tests pass and standalone tool creation defaults to.
//!
//! TypeScript parameterizes definitions over the typebox schema; the erased
//! runtime path carries validated `serde_json::Value` arguments the same way
//! the harness tools do.

use std::sync::Arc;

use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::harness::types::AgentHarnessToolUpdateCallback;
use pi_agent_core::types::ToolExecutionMode;
use pi_ai::types::{BoxedFuture, ConstrainedSamplingSetting, Model};
use serde_json::Value;

/// The result the tool executes return, upstream's `AgentToolResult`.
pub use pi_agent_core::types::AgentToolResult;

/// Whether a tool renders its own framing, upstream's
/// `renderShell?: "default" | "self"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderShell {
    /// The standard colored shell renders the tool row.
    Default,
    /// The tool renders its own framing.
    SelfRender,
}

/// The extension-facing tool executor, upstream's `ToolDefinition.execute`:
/// the call id, the validated arguments, the run's abort signal, the
/// full-snapshot update callback, and the extension context.
pub type ToolExecuteFn = dyn for<'a> Fn(
        &'a str,
        &'a Value,
        Option<&'a AbortSignal>,
        Option<AgentHarnessToolUpdateCallback<'a>>,
        Option<&'a dyn ExtensionContext>,
    )
        -> BoxedFuture<'a, Result<AgentToolResult, pi_agent_core::types::AgentToolError>>
    + Send
    + Sync;

/// Tool definition for `registerTool()`, upstream's `ToolDefinition`.
#[derive(Clone)]
pub struct ToolDefinition {
    /// Tool name (used in LLM tool calls).
    pub name: String,
    /// Human-readable label for UI.
    pub label: String,
    /// Description for LLM.
    pub description: String,
    /// Optional one-line snippet for the Available tools section in the
    /// default system prompt; custom tools are omitted from that section
    /// when absent.
    pub prompt_snippet: Option<String>,
    /// Optional guideline bullets appended to the default system prompt
    /// Guidelines section when this tool is active.
    pub prompt_guidelines: Option<Vec<String>>,
    /// Parameter schema (typebox's output on the wire).
    pub parameters: Value,
    /// Optional provider-side constrained sampling request for this tool.
    pub constrained_sampling: Option<ConstrainedSamplingSetting>,
    /// Whether `ToolExecutionComponent` renders the standard colored shell
    /// or the tool renders its own framing.
    pub render_shell: Option<RenderShell>,
    /// Optional compatibility shim preparing raw tool-call arguments before
    /// schema validation; must return an object conforming to the schema.
    pub prepare_arguments: Option<pi_agent_core::types::AgentToolPrepareArguments>,
    /// Per-tool execution mode override; absent, the default applies.
    pub execution_mode: Option<ToolExecutionMode>,
    /// Execute the tool.
    pub execute: Arc<ToolExecuteFn>,
}

impl std::fmt::Debug for ToolDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The executor does not debug; the declarative surface does, and the
        // non-exhaustive finish names the executor's presence.
        f.debug_struct("ToolDefinition")
            .field("name", &self.name)
            .field("label", &self.label)
            .field("description", &self.description)
            .field("prompt_snippet", &self.prompt_snippet.as_deref())
            .field("prompt_guidelines", &self.prompt_guidelines.as_deref())
            .field("parameters", &self.parameters)
            .field("constrained_sampling", &self.constrained_sampling)
            .field("render_shell", &self.render_shell)
            .field("prepare_arguments", &self.prepare_arguments.is_some())
            .field("execution_mode", &self.execution_mode)
            .field("execute", &())
            .finish_non_exhaustive()
    }
}

/// The context passed to tool execution, upstream's `ExtensionContext`
/// reduced to the surface the built-in tools read.
///
/// The carried surface: the cwd override, the current model, the thinking
/// level, and the session identifiers the bash spawn environment exposes.
/// The interactive surface (UI methods, run mode, abort, compaction) lands
/// with its consumers.
pub trait ExtensionContext: std::any::Any + Send + Sync {
    /// Current working directory, upstream's `ctx.cwd` — the cwd override
    /// every tool's path resolution falls back from.
    fn cwd(&self) -> &str;

    /// Current model, upstream's `ctx.model`; absent when no model is set.
    fn model(&self) -> Option<&Model>;

    /// Current thinking level, upstream's `ctx.thinkingLevel`.
    fn thinking_level(&self) -> Option<pi_agent_core::types::ThinkingLevel>;

    /// The session id the spawn environment exposes as `PI_SESSION_ID`,
    /// upstream's `ctx.sessionManager.getSessionId()`.
    fn session_id(&self) -> Option<String>;

    /// The session file the spawn environment exposes as
    /// `PI_SESSION_FILE`, upstream's `ctx.sessionManager.getSessionFile()`.
    fn session_file(&self) -> Option<String>;
}

/// The minimal `{ cwd }` context: the shape upstream's test stubs pass and
/// the plain default when no richer context is bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CwdContext {
    /// Current working directory.
    pub cwd: String,
}

impl ExtensionContext for CwdContext {
    fn cwd(&self) -> &str {
        &self.cwd
    }

    fn model(&self) -> Option<&Model> {
        None
    }

    fn thinking_level(&self) -> Option<pi_agent_core::types::ThinkingLevel> {
        None
    }

    fn session_id(&self) -> Option<String> {
        None
    }

    fn session_file(&self) -> Option<String> {
        None
    }
}

/// The erased-to-typed context downcast the wrapped tools run, upstream's
/// `ctx?: ExtensionContext` parameter threading.
///
/// # Errors
/// When the context is absent or not the wrapper's context type — an
/// internal contract violation the session cannot produce; the tool
/// surfaces it as a failed execution rather than panic.
pub fn extension_context<C: ExtensionContext>(
    context: &pi_agent_core::harness::types::ToolContext,
) -> Result<Arc<C>, pi_agent_core::types::AgentToolError> {
    let context = context.as_ref().ok_or_else(|| {
        pi_agent_core::types::AgentToolError::from(std::io::Error::other(
            "the tool ran without its extension context",
        ))
    })?;
    Arc::clone(context).downcast::<C>().map_err(|_| {
        pi_agent_core::types::AgentToolError::from(std::io::Error::other(
            "the tool context does not carry the expected extension context type",
        ))
    })
}

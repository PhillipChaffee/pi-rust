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

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::harness::types::AgentHarnessToolUpdateCallback;
use pi_agent_core::types::{CustomAgentMessage, ToolExecutionMode};
use pi_ai::types::{BoxedFuture, ConstrainedSamplingSetting, Model};
use pi_tui::tui::Component;
use serde_json::Value;

use crate::modes::interactive::theme::theme::Theme;

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

/// Rendering options for tool results, upstream's `ToolRenderResultOptions`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToolRenderResultOptions {
    /// Whether the result view is expanded.
    pub expanded: bool,
    /// Whether this is a partial/streaming result.
    pub is_partial: bool,
}

/// One tool row's shared renderer state, upstream's `ToolRenderContext.state`.
///
/// Renderers keep typed state under their own type id, so the call and result
/// slots of one row read and write the same value without a global map.
#[derive(Default)]
pub struct ToolRenderState {
    values: RefCell<HashMap<TypeId, Box<dyn Any>>>,
}

impl std::fmt::Debug for ToolRenderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRenderState")
            .field("len", &self.values.borrow().len())
            .finish()
    }
}

impl ToolRenderState {
    /// An empty state, upstream's `{}` initialized per tool row.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Run `f` over the state value for `T`, inserting `init()` first when
    /// the type has not been touched yet.
    ///
    /// Returns `None` only when the stored value's type and its key disagree,
    /// which no typed caller can produce.
    pub fn with_state_mut<T: 'static, R>(
        &self,
        init: impl FnOnce() -> T,
        f: impl FnOnce(&mut T) -> R,
    ) -> Option<R> {
        let mut values = self.values.borrow_mut();
        let value = values
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(init()));
        value.downcast_mut::<T>().map(f)
    }
}

/// Context passed to tool renderers, upstream's `ToolRenderContext`.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the context restates upstream's flag-for-flag `ToolRenderContext`; each bool is one upstream member"
)]
pub struct ToolRenderContext<'a> {
    /// Current tool call arguments, shared across the call/result slots.
    pub args: &'a Value,
    /// The tool call id, stable across the call/result slots.
    pub tool_call_id: &'a str,
    /// Redraw just this tool row.
    pub invalidate: &'a dyn Fn(),
    /// The component the same slot returned last time, if any.
    pub last_component: Option<Rc<dyn Component>>,
    /// Shared renderer state for this row.
    pub state: &'a ToolRenderState,
    /// Working directory for this tool execution.
    pub cwd: &'a str,
    /// Whether the tool execution has started.
    pub execution_started: bool,
    /// Whether the tool call arguments are complete.
    pub args_complete: bool,
    /// Whether the tool result is partial/streaming.
    pub is_partial: bool,
    /// Whether the result view is expanded.
    pub expanded: bool,
    /// Whether inline images are currently shown in the TUI.
    pub show_images: bool,
    /// Whether the current result is an error.
    pub is_error: bool,
}

impl std::fmt::Debug for ToolRenderContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRenderContext")
            .field("args", &self.args)
            .field("tool_call_id", &self.tool_call_id)
            .field("last_component", &self.last_component.is_some())
            .field("cwd", &self.cwd)
            .field("execution_started", &self.execution_started)
            .field("args_complete", &self.args_complete)
            .field("is_partial", &self.is_partial)
            .field("expanded", &self.expanded)
            .field("show_images", &self.show_images)
            .field("is_error", &self.is_error)
            .finish_non_exhaustive()
    }
}

/// The call-side render hook, upstream `ToolDefinition.renderCall`.
pub type ToolRenderCallFn =
    dyn for<'a> Fn(&'a Value, &'a Theme, &ToolRenderContext<'a>) -> Rc<dyn Component> + Send + Sync;

/// The result-side render hook, upstream `ToolDefinition.renderResult`.
pub type ToolRenderResultFn = dyn for<'a> Fn(
        &'a AgentToolResult,
        &ToolRenderResultOptions,
        &'a Theme,
        &ToolRenderContext<'a>,
    ) -> Rc<dyn Component>
    + Send
    + Sync;

/// Options passed to message renderers, upstream's `MessageRenderOptions`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MessageRenderOptions {
    /// Whether the message view is expanded.
    pub expanded: bool,
    /// Horizontal padding configured by the `outputPad` setting.
    pub output_pad: usize,
}

/// The message kind a Markdown transformer runs for, upstream's
/// `MarkdownTransformContext.messageType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkdownMessageType {
    /// A user message.
    User,
    /// An assistant text block.
    Assistant,
    /// An assistant thinking block.
    AssistantThinking,
}

impl MarkdownMessageType {
    /// The upstream wire spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::AssistantThinking => "assistant-thinking",
        }
    }
}

/// The context a Markdown transformer reads, upstream's
/// `MarkdownTransformContext`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MarkdownTransformContext {
    /// The block being rendered.
    pub message_type: MarkdownMessageType,
    /// Whether the message is still streaming.
    pub is_streaming: bool,
    /// The width available for content.
    pub available_width: usize,
}

/// A Markdown transformer, upstream's `MarkdownTransformer`.
pub type MarkdownTransformer = Arc<dyn Fn(&str, &MarkdownTransformContext) -> String + Send + Sync>;

/// A custom-message renderer, upstream's `MessageRenderer`.
pub type MessageRenderer = Arc<
    dyn Fn(&CustomAgentMessage, &MessageRenderOptions, &Theme) -> Option<Rc<dyn Component>>
        + Send
        + Sync,
>;

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
    /// Custom rendering for the tool call display.
    pub render_call: Option<Arc<ToolRenderCallFn>>,
    /// Custom rendering for the tool result display.
    pub render_result: Option<Arc<ToolRenderResultFn>>,
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
            .field("render_call", &self.render_call.is_some())
            .field("render_result", &self.render_result.is_some())
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
pub trait ExtensionContext: Any + Send + Sync {
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

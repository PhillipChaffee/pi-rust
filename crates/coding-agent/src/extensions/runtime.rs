//! The loaded-extension records and the handler context, upstream's
//! `Extension` and `ExtensionContext` in
//! `src/core/extensions/types.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The registrations carry what inline factories (#148's loading path) and
//! the out-of-process loader (#128) hand the runner for dispatch.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pi_ai::types::BoxedFuture;

use crate::extensions::ToolDefinition;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::results::HandlerResult;
use crate::source_info::SourceInfo;

/// One event handler, upstream's `ExtensionHandler`.
pub type ExtensionHandler = Arc<
    dyn Fn(ExtensionEvent, HandlerContext) -> BoxedFuture<'static, HandlerResult> + Send + Sync,
>;

/// A command handler, upstream's `(args, ctx) => Promise<void>`.
pub type CommandHandler = Arc<
    dyn Fn(String, crate::extensions::runner::ExtensionCommandContext) -> BoxedFuture<'static, ()>
        + Send
        + Sync,
>;

/// An extension-load or handler failure, upstream's `ExtensionError`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionError {
    /// The extension path the failure came from.
    pub extension_path: String,
    /// The event channel or API surface the failure came from.
    pub event: String,
    /// The failure message.
    pub error: String,
    /// The failure stack, when the source carried one.
    pub stack: Option<String>,
}

/// One registered CLI flag, upstream's `ExtensionFlag`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionFlag {
    /// The flag name, without the leading dashes.
    pub name: String,
    /// The flag description.
    pub description: Option<String>,
    /// Whether the flag takes a boolean or a string value.
    pub flag_type: FlagType,
    /// The default value, upstream's `default`.
    pub default: Option<FlagValue>,
    /// The extension that registered the flag.
    pub extension_path: String,
}

/// The flag value type, upstream's `"boolean" | "string"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlagType {
    /// A boolean flag.
    Boolean,
    /// A string-valued flag.
    String,
}

/// A flag value, upstream's `boolean | string`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlagValue {
    /// A boolean value.
    Bool(bool),
    /// A string value.
    Text(String),
}

/// One registered tool, upstream's `RegisteredTool`.
#[derive(Clone)]
pub struct RegisteredTool {
    /// The definition.
    pub definition: ToolDefinition,
    /// The provenance the loader stamped.
    pub source_info: SourceInfo,
}

/// One registered command, upstream's `ResolvedCommand`.
#[derive(Clone)]
pub struct ResolvedCommand {
    /// The registered name.
    pub name: String,
    /// The invocation name (`name`, `name:<sub>` when namespaced).
    pub invocation_name: String,
    /// The description.
    pub description: Option<String>,
    /// The provenance the loader stamped.
    pub source_info: SourceInfo,
    /// The handler.
    pub handler: CommandHandler,
}

/// One loaded extension with its registrations, upstream's `Extension`.
pub struct Extension {
    /// The path the extension was requested by.
    pub path: String,
    /// The resolved load path.
    pub resolved_path: String,
    /// Whether the extension stays out of user-facing lists; the inline
    /// loader marks named factories after their factory runs, so the flag
    /// is an atomic the loader writes post-construction.
    hidden: AtomicBool,
    /// The provenance the loader stamped.
    pub source_info: SourceInfo,
    /// The event handlers, upstream's `handlers: Map<string, HandlerFn[]>`.
    pub handlers: Mutex<HashMap<String, Vec<ExtensionHandler>>>,
    /// The registered tools, insertion-ordered like upstream's `Map`.
    pub tools: Mutex<Vec<RegisteredTool>>,
    /// The registered commands, upstream's `commands: Map`.
    pub commands: Mutex<Vec<ResolvedCommand>>,
    /// The registered flags, upstream's `flags: Map` in registration order.
    flags: Mutex<Vec<ExtensionFlag>>,
}

impl Extension {
    /// Create an empty extension record with the loader's fields.
    #[must_use]
    pub fn new(path: String, resolved_path: String, hidden: bool, source_info: SourceInfo) -> Self {
        Self {
            path,
            resolved_path,
            hidden: AtomicBool::new(hidden),
            source_info,
            handlers: Mutex::new(HashMap::new()),
            tools: Mutex::new(Vec::new()),
            commands: Mutex::new(Vec::new()),
            flags: Mutex::new(Vec::new()),
        }
    }

    /// Whether the extension stays out of user-facing lists.
    #[must_use]
    pub fn hidden(&self) -> bool {
        self.hidden.load(Ordering::Relaxed)
    }

    /// Mark the extension hidden, upstream's inline-factory loader
    /// `extension.hidden = ...` write after the factory ran.
    pub fn set_hidden(&self, hidden: bool) {
        self.hidden.store(hidden, Ordering::Relaxed);
    }

    /// Register one flag, upstream's `pi.registerFlag`; the first
    /// registration of a name wins.
    pub fn register_flag(&self, flag: ExtensionFlag) {
        let mut flags = self
            .flags
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if flags.iter().any(|listed| listed.name == flag.name) {
            return;
        }
        flags.push(flag);
    }

    /// Whether the extension registered one flag name, upstream's
    /// `extension.flags.has(name)`.
    #[must_use]
    pub fn has_flag(&self, name: &str) -> bool {
        self.flags
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .any(|flag| flag.name == name)
    }

    /// Register one event handler, upstream's `pi.on`.
    pub fn on(&self, channel: &str, handler: ExtensionHandler) {
        self.handlers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(channel.to_string())
            .or_default()
            .push(handler);
    }

    /// The registered flags snapshot in registration order, the read the
    /// runner's `getFlags` walks.
    #[must_use]
    pub fn flags(&self) -> Vec<ExtensionFlag> {
        self.flags
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Register one tool, upstream's `pi.registerTool`.
    pub fn register_tool(&self, tool: RegisteredTool) {
        self.tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(tool);
    }

    /// Register one command, upstream's `pi.registerCommand`.
    pub fn register_command(&self, command: ResolvedCommand) {
        self.commands
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(command);
    }
}

/// The abort-signal handle the context exposes, upstream's `AbortSignal`.
pub use pi_agent_core::harness::context::AbortSignal as AbortSignalHandle;

/// The context-usage reading, upstream's `ContextUsage`.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextUsage {
    /// The estimated tokens, `None` when unknown until the next response.
    pub tokens: Option<u64>,
    /// The model's context window.
    pub context_window: u64,
    /// The usage percent, `None` when tokens are unknown.
    pub percent: Option<f64>,
}

/// The staleness assert, upstream's `assertActive`: errors when the runtime
/// was invalidated by session replacement or reload.
pub type AssertActiveFn = Arc<dyn Fn() -> Result<(), String> + Send + Sync>;

/// The context action closures, upstream's `HandlerContextActions`.
#[derive(Clone, Default)]
pub struct ContextActions {
    /// The current model.
    pub get_model: Option<Arc<dyn Fn() -> Option<pi_ai::types::Model> + Send + Sync>>,
    /// The scoped models.
    pub get_scoped_models:
        Option<Arc<dyn Fn() -> Vec<crate::extensions::reasons::ScopedModel> + Send + Sync>>,
    /// The current thinking level.
    pub get_thinking_level:
        Option<Arc<dyn Fn() -> pi_agent_core::types::ThinkingLevel + Send + Sync>>,
    /// Whether the session is idle.
    pub is_idle: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    /// Whether the project is trusted.
    pub is_project_trusted: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    /// The active run's signal.
    pub get_signal: Option<Arc<dyn Fn() -> Option<AbortSignalHandle> + Send + Sync>>,
    /// Abort the current run.
    pub abort: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Whether messages are queued.
    pub has_pending_messages: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    /// Shut the app down.
    pub shutdown: Option<Arc<dyn Fn() + Send + Sync>>,
    /// The context usage.
    pub get_context_usage: Option<Arc<dyn Fn() -> Option<ContextUsage> + Send + Sync>>,
    /// Compact the session, upstream's `ctx.compact(options?)`.
    pub compact: Option<Arc<dyn Fn(Option<String>) + Send + Sync>>,
    /// The effective system prompt.
    pub get_system_prompt: Option<Arc<dyn Fn() -> String + Send + Sync>>,
    /// The options the system prompt was built with.
    pub get_system_prompt_options:
        Option<Arc<dyn Fn() -> crate::system_prompt::BuildSystemPromptOptions + Send + Sync>>,
}

/// The context the handlers and commands receive, upstream's
/// `HandlerContext`. Values resolve at call time through the bound
/// actions, and every accessor asserts the runtime is current first.
#[derive(Clone)]
pub struct HandlerContext {
    inner: Arc<ContextInner>,
}

/// The shared context state.
struct ContextInner {
    cwd: String,
    mode: crate::extensions::reasons::ExtensionMode,
    ui: Option<Arc<dyn std::any::Any + Send + Sync>>,
    session_manager: crate::session_manager::SessionManager,
    model_registry: crate::model_registry::ModelRegistry,
    assert_active: AssertActiveFn,
    actions: Arc<ContextActions>,
}

impl HandlerContext {
    /// Build the context over the shared state.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "the context shape mirrors upstream's"
    )]
    pub fn new(
        cwd: String,
        mode: crate::extensions::reasons::ExtensionMode,
        ui: Option<Arc<dyn std::any::Any + Send + Sync>>,
        session_manager: crate::session_manager::SessionManager,
        model_registry: crate::model_registry::ModelRegistry,
        assert_active: AssertActiveFn,
        actions: Arc<ContextActions>,
    ) -> Self {
        Self {
            inner: Arc::new(ContextInner {
                cwd,
                mode,
                ui,
                session_manager,
                model_registry,
                assert_active,
                actions,
            }),
        }
    }

    /// Assert the runtime is current, upstream's `assertActive`.
    ///
    /// # Errors
    /// When the runtime was invalidated by session replacement or reload.
    pub fn assert_active(&self) -> Result<(), String> {
        (self.inner.assert_active)()
    }

    /// The cwd, upstream's `ctx.cwd`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn cwd(&self) -> Result<String, String> {
        self.assert_active()?;
        Ok(self.inner.cwd.clone())
    }

    /// The bound mode, upstream's `ctx.mode`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn mode(&self) -> Result<crate::extensions::reasons::ExtensionMode, String> {
        self.assert_active()?;
        Ok(self.inner.mode)
    }

    /// Whether a UI is bound, upstream's `ctx.hasUI`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn has_ui(&self) -> Result<bool, String> {
        self.assert_active()?;
        Ok(self.inner.ui.is_some())
    }

    /// The bound UI handle, upstream's `ctx.ui`; the interactive surface
    /// downcasts (#132).
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn ui(&self) -> Result<Option<Arc<dyn std::any::Any + Send + Sync>>, String> {
        self.assert_active()?;
        Ok(self.inner.ui.clone())
    }

    /// The session manager handle, upstream's `ctx.sessionManager`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn session_manager(&self) -> Result<crate::session_manager::SessionManager, String> {
        self.assert_active()?;
        Ok(self.inner.session_manager.clone())
    }

    /// The model registry handle, upstream's `ctx.modelRegistry`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn model_registry(&self) -> Result<crate::model_registry::ModelRegistry, String> {
        self.assert_active()?;
        Ok(self.inner.model_registry.clone())
    }

    /// Run one `Fn() -> T` bound action, or upstream's no-op default.
    fn act<T>(
        &self,
        action: Option<&Arc<dyn Fn() -> T + Send + Sync>>,
        default: impl FnOnce() -> T,
    ) -> Result<T, String> {
        self.assert_active()?;
        Ok(action.map_or_else(default, |action| action()))
    }

    /// The current model, upstream's `ctx.model`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn get_model(&self) -> Result<Option<pi_ai::types::Model>, String> {
        self.act(self.inner.actions.get_model.as_ref(), || None)
    }

    /// The scoped models, upstream's `ctx.scopedModels`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn get_scoped_models(
        &self,
    ) -> Result<Vec<crate::extensions::reasons::ScopedModel>, String> {
        self.act(self.inner.actions.get_scoped_models.as_ref(), Vec::new)
    }

    /// The thinking level, upstream's `ctx.thinkingLevel`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn get_thinking_level(&self) -> Result<pi_agent_core::types::ThinkingLevel, String> {
        self.act(self.inner.actions.get_thinking_level.as_ref(), || {
            pi_agent_core::types::ThinkingLevel::Off
        })
    }

    /// Whether the session is idle, upstream's `ctx.isIdle()`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn is_idle(&self) -> Result<bool, String> {
        self.act(self.inner.actions.is_idle.as_ref(), || true)
    }

    /// Whether the project is trusted, upstream's `ctx.isProjectTrusted()`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn is_project_trusted(&self) -> Result<bool, String> {
        self.act(self.inner.actions.is_project_trusted.as_ref(), || true)
    }

    /// The active run's signal, upstream's `ctx.signal`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn get_signal(&self) -> Result<Option<AbortSignalHandle>, String> {
        self.act(self.inner.actions.get_signal.as_ref(), || None)
    }

    /// Abort the current run, upstream's `ctx.abort()`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn abort(&self) -> Result<(), String> {
        self.act(self.inner.actions.abort.as_ref(), || {})
    }

    /// Whether messages are queued, upstream's `ctx.hasPendingMessages()`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn has_pending_messages(&self) -> Result<bool, String> {
        self.act(self.inner.actions.has_pending_messages.as_ref(), || false)
    }

    /// Shut the app down, upstream's `ctx.shutdown()`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn shutdown(&self) -> Result<(), String> {
        self.act(self.inner.actions.shutdown.as_ref(), || {})
    }

    /// The context usage, upstream's `ctx.getContextUsage()`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn get_context_usage(&self) -> Result<Option<ContextUsage>, String> {
        self.act(self.inner.actions.get_context_usage.as_ref(), || None)
    }

    /// Compact the session, upstream's `ctx.compact(options?)`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn compact(&self, custom_instructions: Option<String>) -> Result<(), String> {
        self.assert_active()?;
        if let Some(action) = self.inner.actions.compact.as_ref() {
            action(custom_instructions);
        }
        Ok(())
    }

    /// The effective system prompt, upstream's `ctx.getSystemPrompt()`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn get_system_prompt(&self) -> Result<String, String> {
        self.act(self.inner.actions.get_system_prompt.as_ref(), String::new)
    }

    /// The system-prompt options, upstream's `ctx.getSystemPromptOptions()`.
    ///
    /// # Errors
    /// When the runtime is stale.
    pub fn get_system_prompt_options(
        &self,
    ) -> Result<crate::system_prompt::BuildSystemPromptOptions, String> {
        self.assert_active()?;
        let cwd = self.inner.cwd.clone();
        self.inner
            .actions
            .get_system_prompt_options
            .as_ref()
            .map_or_else(
                || {
                    Ok(crate::system_prompt::BuildSystemPromptOptions {
                        cwd,
                        ..Default::default()
                    })
                },
                |action| Ok(action()),
            )
    }
}
impl std::fmt::Debug for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let handlers = self
            .handlers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tools = self
            .tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let commands = self
            .commands
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let flags = self
            .flags
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f.debug_struct("Extension")
            .field("path", &self.path)
            .field("resolved_path", &self.resolved_path)
            .field("hidden", &self.hidden())
            .field("source_info", &self.source_info)
            .field("handlers", &handlers.keys().collect::<Vec<_>>())
            .field("tools", &tools.len())
            .field("commands", &commands.len())
            .field(
                "flags",
                &flags.iter().map(|flag| &flag.name).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl std::fmt::Debug for RegisteredTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredTool")
            .field("definition", &self.definition)
            .field("source_info", &self.source_info)
            .finish()
    }
}

impl std::fmt::Debug for ResolvedCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedCommand")
            .field("name", &self.name)
            .field("invocation_name", &self.invocation_name)
            .field("description", &self.description)
            .field("source_info", &self.source_info)
            .field("handler", &())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ContextActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bound = [
            self.get_model.is_some(),
            self.get_scoped_models.is_some(),
            self.get_thinking_level.is_some(),
            self.is_idle.is_some(),
            self.is_project_trusted.is_some(),
            self.get_signal.is_some(),
            self.abort.is_some(),
            self.has_pending_messages.is_some(),
            self.shutdown.is_some(),
            self.get_context_usage.is_some(),
            self.compact.is_some(),
            self.get_system_prompt.is_some(),
            self.get_system_prompt_options.is_some(),
        ]
        .iter()
        .filter(|bound| **bound)
        .count();
        f.debug_struct("ContextActions")
            .field("bound_actions", &bound)
            .finish()
    }
}

impl std::fmt::Debug for HandlerContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandlerContext")
            .field("cwd", &self.inner.cwd)
            .field("mode", &self.inner.mode)
            .field("has_ui", &self.inner.ui.is_some())
            .field("session_manager", &self.inner.session_manager)
            .field("model_registry", &self.inner.model_registry)
            .field("actions", &self.inner.actions)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;

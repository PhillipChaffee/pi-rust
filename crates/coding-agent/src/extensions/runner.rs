//! The extension dispatch runner, upstream's `ExtensionRunner` in
//! `src/core/extensions/runner.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`ExtensionRunner::emit`] is the generic dispatch over the loaded
//! extensions in request order: every handler of the event's channel runs,
//! and — for the `session_before_*` channels — the last defined result wins
//! with a cancelled result short-circuiting the remaining handlers. The
//! dedicated emit methods restate their channel's own merge or short-
//! circuit contract. Handler results are infallible here, so upstream's
//! handler try/catch paths do not surface: a handler failure to produce a
//! value reads as [`HandlerResult::None`], and a panicking handler aborts
//! the run rather than being swallowed.
//!
//! The interactive surfaces (the UI context wrap and keybinding
//! reservation) ride their consumers on the sibling tickets and are not
//! re-stated here; the command context and session-handler binding land
//! here (`bind_command_context`/`command_context`), with the UI-context
//! half of `setUIContext` staying with #132.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::types::AgentMessage;
use pi_agent_core::types::ThinkingLevel;
use pi_ai::types::{BoxedFuture, ImageContent, Model, ProviderHeaders, UserContent};
use serde_json::Value;

use crate::core::slash_commands::SlashCommandInfo;
use crate::diagnostics::ResourceDiagnostic;
use crate::provider_composer::ProviderConfigInput;

use crate::extensions::events::ExtensionEvent;
use crate::extensions::reasons::{
    CustomMessagePayload, ExtensionMode, InputSource, SessionStartReason, StreamingBehavior,
};
use crate::extensions::results::{
    HandlerResult, InputResult, ProjectTrustResult, SessionBeforeResult, ToolCallResult,
    ToolResultResult, UserBashResult,
};
use crate::extensions::runtime::{
    AssertActiveFn, ContextActions, Extension, ExtensionError, HandlerContext,
};

/// A listener the runner reports dispatch errors to, upstream's
/// `ExtensionErrorListener`.
pub type ExtensionErrorListener = Arc<dyn Fn(&ExtensionError) + Send + Sync>;

/// The session-replacement outcome the session handlers report, upstream's
/// resolved `{ cancelled: boolean }`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionReplaceOutcome {
    /// Whether the replacement was refused.
    pub cancelled: bool,
}

impl SessionReplaceOutcome {
    /// The `not cancelled` outcome, upstream's `{ cancelled: false }`.
    #[must_use]
    pub const fn proceeded() -> Self {
        Self { cancelled: false }
    }
}

/// The session-replacement handler's setup callback, upstream's
/// `options.setup`: runs against the replacement session manager.
pub type SetupSessionFn =
    Arc<dyn Fn(crate::session_manager::SessionManager) -> BoxedFuture<'static, ()> + Send + Sync>;

/// The session-replacement handler's post-replacement callback,
/// upstream's `options.withSession`: runs with the replacement's context.
pub type WithSessionFn = Arc<dyn Fn(HandlerContext) -> BoxedFuture<'static, ()> + Send + Sync>;

/// The options `ctx.newSession()` re-arms its handler with, upstream's
/// `newSession` options.
#[derive(Clone, Default)]
pub struct NewSessionOptions {
    /// The parent session to fork from, upstream's `parentSession`.
    pub parent_session: Option<String>,
    /// The setup run against the replacement session manager.
    pub setup: Option<SetupSessionFn>,
    /// The callback run post-replacement with the replacement context.
    pub with_session: Option<WithSessionFn>,
}

impl std::fmt::Debug for NewSessionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The closures do not debug; the slots' presence carries the shape.
        f.debug_struct("NewSessionOptions")
            .field("parent_session", &self.parent_session)
            .field("setup", &self.setup.is_some())
            .field("with_session", &self.with_session.is_some())
            .finish()
    }
}

/// Where a fork entry positions the replacement, upstream's
/// `"before" | "at"`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForkPosition {
    /// The replacement reads the entry before position.
    Before,
    /// The replacement reads the entry at position.
    At,
}

/// The options `ctx.fork()` re-arms its handler with, upstream's `fork`
/// options.
#[derive(Clone, Default)]
pub struct ForkOptions {
    /// Where the replacement entry positions, upstream's `position`.
    pub position: Option<ForkPosition>,
    /// The callback run post-replacement with the replacement context.
    pub with_session: Option<WithSessionFn>,
}

impl std::fmt::Debug for ForkOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The closures do not debug; the slots' presence carries the shape.
        f.debug_struct("ForkOptions")
            .field("position", &self.position)
            .field("with_session", &self.with_session.is_some())
            .finish()
    }
}

/// The options `ctx.navigateTree()` re-arms its handler with, upstream's
/// `navigateTree` options.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NavigateTreeOptions {
    /// Whether the tree navigation captures a summary.
    pub summarize: bool,
    /// The compact instructions the summarize reports.
    pub custom_instructions: Option<String>,
    /// Whether the instructions replace the template instead of joining.
    pub replace_instructions: bool,
    /// The session label to set.
    pub label: Option<String>,
}

/// The options `ctx.switchSession()` re-arms its handler with, upstream's
/// `switchSession` options.
#[derive(Clone, Default)]
pub struct SwitchSessionOptions {
    /// The callback run post-replacement with the replacement context.
    pub with_session: Option<WithSessionFn>,
}

impl std::fmt::Debug for SwitchSessionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The closures do not debug; the slots' presence carries the shape.
        f.debug_struct("SwitchSessionOptions")
            .field("with_session", &self.with_session.is_some())
            .finish()
    }
}

/// The handler `ctx.newSession()` invokes, upstream's `NewSessionHandler`
/// with its options collapsed to the owned type.
pub type NewSessionHandler =
    Arc<dyn Fn(NewSessionOptions) -> BoxedFuture<'static, SessionReplaceOutcome> + Send + Sync>;

/// The handler `ctx.fork()` invokes, upstream's `ForkHandler`.
pub type ForkHandler =
    Arc<dyn Fn(String, ForkOptions) -> BoxedFuture<'static, SessionReplaceOutcome> + Send + Sync>;

/// The handler `ctx.navigateTree()` invokes, upstream's
/// `NavigateTreeHandler`.
pub type NavigateTreeHandler = Arc<
    dyn Fn(String, NavigateTreeOptions) -> BoxedFuture<'static, SessionReplaceOutcome>
        + Send
        + Sync,
>;

/// The handler `ctx.switchSession()` invokes, upstream's
/// `SwitchSessionHandler`.
pub type SwitchSessionHandler = Arc<
    dyn Fn(String, SwitchSessionOptions) -> BoxedFuture<'static, SessionReplaceOutcome>
        + Send
        + Sync,
>;

/// The handler `ctx.reload()` invokes, upstream's `ReloadHandler`.
pub type ReloadHandler = Arc<dyn Fn() -> BoxedFuture<'static, ()> + Send + Sync>;

/// The idle-wait the command context resolves, upstream's
/// `waitForIdle`.
pub type WaitForIdleHandler = Arc<dyn Fn() -> BoxedFuture<'static, ()> + Send + Sync>;

/// The command-context handler bundle, upstream's
/// `ExtensionCommandContextActions`: the six replace/reload actions the
/// session core arms in interactive mode.
#[derive(Clone)]
pub struct CommandContextActions {
    /// The idle wait, upstream's `waitForIdle`.
    pub wait_for_idle: WaitForIdleHandler,
    /// The session replacement, upstream's `newSession`.
    pub new_session: NewSessionHandler,
    /// The entry fork, upstream's `fork`.
    pub fork: ForkHandler,
    /// The tree navigation, upstream's `navigateTree`.
    pub navigate_tree: NavigateTreeHandler,
    /// The session switch, upstream's `switchSession`.
    pub switch_session: SwitchSessionHandler,
    /// The extension reload, upstream's `reload`.
    pub reload: ReloadHandler,
}

impl std::fmt::Debug for CommandContextActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The closures do not debug; the bound slot count carries the shape.
        f.debug_struct("CommandContextActions")
            .field("bound_actions", &6usize)
            .finish()
    }
}

/// Where a sent message lands relative to the current run, upstream's
/// `deliverAs: "steer" | "followUp" | "nextTurn"`.
///
/// `steer` and `followUp` are also the only two values `sendUserMessage`
/// takes upstream; the enum carries all three borrowers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeliverAs {
    /// Steer the active run.
    Steer,
    /// Queue a follow-up turn.
    FollowUp,
    /// Queue for the next turn.
    NextTurn,
}

/// The options `pi.sendMessage` re-arms its handler with, upstream's
/// `{ triggerTurn?, deliverAs? }`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SendMessageOptions {
    /// Whether the sent message triggers a turn.
    pub trigger_turn: Option<bool>,
    /// Where the message lands relative to the current run.
    pub deliver_as: Option<DeliverAs>,
}

/// The options `pi.sendUserMessage` re-arms its handler with, upstream's
/// `{ deliverAs?, expandPromptTemplates? }`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SendUserMessageOptions {
    /// Where the message lands relative to the current run.
    pub deliver_as: Option<DeliverAs>,
    /// Whether prompt templates expand, upstream's `expandPromptTemplates`.
    pub expand_prompt_templates: Option<bool>,
}

/// The tool read the session reports to extensions, upstream's
/// `ToolInfo`: the wire fields the system prompt and selectors read plus
/// the provenance.
#[derive(Clone, Debug)]
pub struct ToolInfo {
    /// The tool name, upstream's `name`.
    pub name: String,
    /// The description the model reads.
    pub description: String,
    /// The parameter schema.
    pub parameters: Value,
    /// The guideline bullets appended to the system prompt.
    pub prompt_guidelines: Option<Vec<String>>,
    /// The tool's provenance.
    pub source_info: crate::source_info::SourceInfo,
}

/// The sendMessage bind, upstream's `SendMessageHandler`.
pub type SendMessageHandler =
    Arc<dyn Fn(CustomMessagePayload, Option<SendMessageOptions>) + Send + Sync>;

/// The sendUserMessage bind, upstream's `SendUserMessageHandler`.
pub type SendUserMessageHandler =
    Arc<dyn Fn(UserContent, Option<SendUserMessageOptions>) + Send + Sync>;

/// The appendEntry bind, upstream's `AppendEntryHandler<T>`.
pub type AppendEntryHandler = Arc<dyn Fn(String, Option<Value>) + Send + Sync>;

/// The setSessionName bind, upstream's `SetSessionNameHandler`.
pub type SetSessionNameHandler = Arc<dyn Fn(String) + Send + Sync>;

/// The getSessionName bind, upstream's `GetSessionNameHandler`.
pub type GetSessionNameHandler = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// The setLabel bind, upstream's `SetLabelHandler`.
pub type SetLabelHandler = Arc<dyn Fn(String, Option<String>) + Send + Sync>;

/// The getActiveTools bind, upstream's `GetActiveToolsHandler`.
pub type GetActiveToolsHandler = Arc<dyn Fn() -> Vec<String> + Send + Sync>;

/// The getAllTools bind, upstream's `GetAllToolsHandler`.
pub type GetAllToolsHandler = Arc<dyn Fn() -> Vec<ToolInfo> + Send + Sync>;

/// The setActiveTools bind, upstream's `SetActiveToolsHandler`.
pub type SetActiveToolsHandler = Arc<dyn Fn(Vec<String>) + Send + Sync>;

/// The refreshTools bind, upstream's `RefreshToolsHandler`.
pub type RefreshToolsHandler = Arc<dyn Fn() + Send + Sync>;

/// The getCommands bind, upstream's `GetCommandsHandler`.
pub type GetCommandsHandler = Arc<dyn Fn() -> Vec<SlashCommandInfo> + Send + Sync>;

/// The setModel bind, upstream's `SetModelHandler` (async ack).
pub type SetModelHandler = Arc<dyn Fn(Model) -> BoxedFuture<'static, bool> + Send + Sync>;

/// The getThinkingLevel bind, upstream's `GetThinkingLevelHandler`.
pub type GetThinkingLevelHandler = Arc<dyn Fn() -> ThinkingLevel + Send + Sync>;

/// The setThinkingLevel bind, upstream's `SetThinkingLevelHandler`.
pub type SetThinkingLevelHandler = Arc<dyn Fn(ThinkingLevel) + Send + Sync>;

/// The post-bind message/entry/session actions the runner dispatches to,
/// upstream's `ExtensionActions` bundle handed to `bindCore`.
///
/// The extension API reads the bound bundle after the bind pass; the
/// loader phase never runs these (upstream's throwing stubs).
#[derive(Clone)]
pub struct ExtensionActions {
    /// Send a custom message, upstream's `sendMessage`.
    pub send_message: SendMessageHandler,
    /// Send a buildable user message, upstream's `sendUserMessage`.
    pub send_user_message: SendUserMessageHandler,
    /// Append a custom entry, upstream's `appendEntry`.
    pub append_entry: AppendEntryHandler,
    /// Set the session name, upstream's `setSessionName`.
    pub set_session_name: SetSessionNameHandler,
    /// Read the session name, upstream's `getSessionName`.
    pub get_session_name: GetSessionNameHandler,
    /// Set an entry label, upstream's `setLabel`.
    pub set_label: SetLabelHandler,
    /// Read the active tool names, upstream's `getActiveTools`.
    pub get_active_tools: GetActiveToolsHandler,
    /// Read every tool, upstream's `getAllTools`.
    pub get_all_tools: GetAllToolsHandler,
    /// Restate the active tools, upstream's `setActiveTools`.
    pub set_active_tools: SetActiveToolsHandler,
    /// Refresh the tool registry, upstream's `refreshTools`.
    pub refresh_tools: RefreshToolsHandler,
    /// Read the slash commands, upstream's `getCommands`.
    pub get_commands: GetCommandsHandler,
    /// Set the model, upstream's `setModel`.
    pub set_model: SetModelHandler,
    /// Read the thinking level, upstream's `getThinkingLevel`.
    pub get_thinking_level: GetThinkingLevelHandler,
    /// Set the thinking level, upstream's `setThinkingLevel`.
    pub set_thinking_level: SetThinkingLevelHandler,
}

impl std::fmt::Debug for ExtensionActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The closures do not debug; the bound slot count carries the shape.
        f.debug_struct("ExtensionActions")
            .field("bound_actions", &14usize)
            .finish()
    }
}

/// The provider-registration actions the bind pass flushes through,
/// upstream's anonymous `providerActions`. The native-provider slot stays
/// with the composer's native-provider ticket.
#[derive(Clone)]
pub struct ProviderActions {
    /// Register a named provider, upstream's `registerProvider`.
    pub register_provider:
        Arc<dyn Fn(String, ProviderConfigInput) -> Result<(), String> + Send + Sync>,
    /// Remove a provider registration, upstream's `unregisterProvider`.
    pub unregister_provider: Arc<dyn Fn(String) -> Result<(), String> + Send + Sync>,
}

impl std::fmt::Debug for ProviderActions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The closures do not debug; the slots' presence names the bind.
        f.debug_struct("ProviderActions")
            .field(
                "register_provider",
                &Arc::strong_count(&self.register_provider),
            )
            .field(
                "unregister_provider",
                &Arc::strong_count(&self.unregister_provider),
            )
            .finish()
    }
}

/// The combined `before_agent_start` result, upstream's
/// `BeforeAgentStartEventResult` accumulator.
///
/// The messages every handler queued plus the last replacement system
/// prompt, `system_prompt: None` when no handler replaced it.
#[derive(Clone, Debug, Default)]
pub struct BeforeAgentStartCombined {
    /// The custom messages, in handler-registration order.
    pub messages: Vec<CustomMessagePayload>,
    /// The replacement system prompt, absent when unmodified.
    pub system_prompt: Option<String>,
}

/// One extension-discovered resource path with the extension that
/// contributed it, upstream's `{ path, extensionPath }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionResourcePath {
    /// The path the handler returned (already cwd-resolved by the caller).
    pub path: String,
    /// The extension that contributed the path.
    pub extension_path: String,
}

/// The `resources_discover` accumulator, upstream's
/// `emitResourcesDiscover` return.
#[derive(Clone, Debug, Default)]
pub struct ResourcesDiscoverPaths {
    /// Skill paths to load.
    pub skill_paths: Vec<ExtensionResourcePath>,
    /// Prompt-template paths to load.
    pub prompt_paths: Vec<ExtensionResourcePath>,
    /// Theme paths to load.
    pub theme_paths: Vec<ExtensionResourcePath>,
}

/// The `project_trust` free-emit outcome, upstream's
/// `emitProjectTrustEvent` return.
#[derive(Clone, Debug, Default)]
pub struct ProjectTrustOutcome {
    /// The first decided result, absent when every handler was undecided.
    pub result: Option<ProjectTrustResult>,
    /// The handler failures, in extension then handler order.
    pub errors: Vec<ExtensionError>,
}

/// Whether the event is one of the four `session_before_*` channels the
/// generic dispatch merges, upstream's `isSessionBeforeEvent`.
const fn is_session_before(event: &ExtensionEvent) -> bool {
    matches!(
        event,
        ExtensionEvent::SessionBeforeSwitch { .. }
            | ExtensionEvent::SessionBeforeFork { .. }
            | ExtensionEvent::SessionBeforeCompact { .. }
            | ExtensionEvent::SessionBeforeTree { .. }
    )
}

/// Extract a handler's own `session_before` result, `None` when the handler
/// returned nothing. A `session_before` channel handler that returns an
/// unrelated result reads as undecided — upstream would misread such a cast
/// and consumers never produce it.
fn session_before_result(handler_result: &HandlerResult) -> Option<SessionBeforeResult> {
    match handler_result {
        HandlerResult::SessionBefore(result) if !matches!(result, SessionBeforeResult::None) => {
            Some(result.clone())
        }
        _ => None,
    }
}

/// Whether a `session_before` result cancels its flow, upstream's
/// `result.cancel`. The none-of-the-above result never cancels.
const fn cancels(result: &SessionBeforeResult) -> bool {
    match result {
        SessionBeforeResult::Switch { cancel }
        | SessionBeforeResult::Fork { cancel, .. }
        | SessionBeforeResult::Compact { cancel, .. }
        | SessionBeforeResult::Tree { cancel, .. } => *cancel,
        SessionBeforeResult::None => false,
    }
}

/// The extension dispatch runner, upstream's `ExtensionRunner`: the loaded
/// extensions plus the context ingredients the handlers resolve through.
///
/// The runner also carries the loader-runtime handle and the post-bind
/// action bundle the session core arms.
pub struct ExtensionRunner {
    extensions: Vec<Arc<Extension>>,
    assert_active: AssertActiveFn,
    actions: Arc<ContextActions>,
    cwd: String,
    mode: ExtensionMode,
    ui: Option<Arc<dyn std::any::Any + Send + Sync>>,
    session_manager: crate::session_manager::SessionManager,
    model_registry: crate::model_registry::ModelRegistry,
    /// The loader runtime handle, upstream's constructor `runtime` arg: the
    /// flag values, the provider queue, and the session-replacement
    /// staleness the extension APIs wrote at load time.
    runtime: crate::extensions::api::ExtensionRuntime,
    /// The first staleness message this runner accepted, shared with the
    /// assert wrapper, upstream's `staleMessage`.
    stale_message: Arc<Mutex<Option<String>>>,
    /// The bound message/entry/session actions, upstream's `bindCore`
    /// action copy landing on the runtime; absent until the bind pass.
    bound_actions: Option<Arc<ExtensionActions>>,
    /// The command-context handlers with the no-op defaults, upstream's
    /// runner fields plus `bindCommandContext`.
    command: CommandHandlers,
    /// The command diagnostics, reset per registered-commands read,
    /// upstream's `commandDiagnostics`.
    command_diagnostics: Mutex<Vec<ResourceDiagnostic>>,
    /// The active-tool names the registered-tool wrapper diffs, upstream's
    /// `runtime.getActiveTools` read the wrapper wraps.
    active_tools: Arc<dyn Fn() -> Vec<String> + Send + Sync>,
    error_listeners: Arc<Mutex<Vec<ExtensionErrorListener>>>,
}

/// The runner's command-handler set, upstream's runner fields with their
/// no-op defaults. Crate-private: the session core binds through
/// [`ExtensionRunner::bind_command_context`], the command context reads
/// through [`ExtensionRunner::command_context`].
#[derive(Clone)]
struct CommandHandlers {
    wait_for_idle: WaitForIdleHandler,
    new_session: NewSessionHandler,
    fork: ForkHandler,
    navigate_tree: NavigateTreeHandler,
    switch_session: SwitchSessionHandler,
    reload: ReloadHandler,
}

impl CommandHandlers {
    /// The no-op defaults, upstream's runner field initializers.
    fn no_ops() -> Self {
        Self {
            wait_for_idle: Arc::new(|| Box::pin(async {})),
            new_session: Arc::new(|_options: NewSessionOptions| {
                Box::pin(async { SessionReplaceOutcome::proceeded() })
            }),
            fork: Arc::new(|_entry_id: String, _options: ForkOptions| {
                Box::pin(async { SessionReplaceOutcome::proceeded() })
            }),
            navigate_tree: Arc::new(|_target_id: String, _options: NavigateTreeOptions| {
                Box::pin(async { SessionReplaceOutcome::proceeded() })
            }),
            switch_session: Arc::new(|_session_path: String, _options: SwitchSessionOptions| {
                Box::pin(async { SessionReplaceOutcome::proceeded() })
            }),
            reload: Arc::new(|| Box::pin(async {})),
        }
    }
}

impl ExtensionRunner {
    /// Build the runner over already-loaded extensions, upstream's
    /// constructor argument order.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the context ingredients mirror upstream's constructor plus the actions bundle"
    )]
    pub fn new(
        extensions: Vec<Arc<Extension>>,
        actions: Arc<ContextActions>,
        assert_active: AssertActiveFn,
        cwd: String,
        mode: ExtensionMode,
        ui: Option<Arc<dyn std::any::Any + Send + Sync>>,
        session_manager: crate::session_manager::SessionManager,
        model_registry: crate::model_registry::ModelRegistry,
        runtime: crate::extensions::api::ExtensionRuntime,
    ) -> Self {
        // The caller's assert guard reads as the seam; the runner's own
        // first-wins staleness layer sits in front of it so a post-bind
        // `invalidate` fails reads without the caller re-binding.
        let stale_message: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let assert_active = {
            let given = assert_active;
            let stale_message = Arc::clone(&stale_message);
            Arc::new(move || -> Result<(), String> {
                if let Some(message) = stale_message
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .as_deref()
                {
                    return Err(message.to_string());
                }
                given()
            })
        };
        Self {
            extensions,
            assert_active,
            actions,
            cwd,
            mode,
            ui,
            session_manager,
            model_registry,
            runtime,
            stale_message,
            bound_actions: None,
            command: CommandHandlers::no_ops(),
            command_diagnostics: Mutex::new(Vec::new()),
            active_tools: Arc::new(Vec::new),
            error_listeners: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Bind the active-tool reader the registered-tool wrapper diffs,
    /// upstream's `bindCore` `getActiveTools`.
    pub fn bind_active_tools(&mut self, active_tools: Arc<dyn Fn() -> Vec<String> + Send + Sync>) {
        self.active_tools = active_tools;
    }

    /// THE bind pass, upstream's `bindCore`: carry the message/entry/tool-set
    /// and thinking-level binds onto the runner for the extension API's
    /// post-bind reads, flush the queued provider registrations into the
    /// model registry (or the caller's provider actions), and switch
    /// provider registration over to immediate effect.
    ///
    /// The active-tool bind also arms the registered-tool wrapper's diff
    /// reader, replacing any standalone [`Self::bind_active_tools`] bind.
    ///
    /// The `session_before_*` and `bindUI` surfaces are bound per their
    /// consumers; the context actions arrive at construction.
    pub fn bind_core(
        &mut self,
        actions: ExtensionActions,
        provider_actions: Option<ProviderActions>,
    ) {
        self.active_tools = Arc::clone(&actions.get_active_tools);
        self.runtime
            .bind_refresh_tools(Arc::clone(&actions.refresh_tools));
        self.bound_actions = Some(Arc::new(actions));
        let flush: Option<&ProviderActions> = provider_actions.as_ref();
        for registration in self.runtime.take_pending_provider_registrations() {
            let registered = match flush {
                Some(provider_actions) => (provider_actions.register_provider)(
                    registration.name.clone(),
                    registration.config.clone(),
                ),
                None => self
                    .model_registry
                    .register_provider_config(&registration.name, registration.config.clone()),
            };
            if let Err(error) = registered {
                self.emit_error(&ExtensionError {
                    extension_path: registration.extension_path.clone(),
                    event: "register_provider".to_string(),
                    error,
                    stack: None,
                });
            }
        }
        // The runtime registrations switch over to the immediate path, the
        // consumer-side write half of the flush.
        // shortcut: the pending native-provider queue stays queued, when the
        // composer's native-provider ticket lands its Provider type
        self.runtime
            .bind_provider_actions(provider_actions.map(Arc::new));
    }

    /// The bound message/entry/session bundle, absent until
    /// [`Self::bind_core`] lands, stale-checked like the wrapper's reads.
    ///
    /// # Errors
    /// When the runtime was invalidated by session replacement or reload.
    pub fn bound_actions(&self) -> Result<Arc<ExtensionActions>, String> {
        (self.assert_active)()?;
        self.bound_actions
            .clone()
            .ok_or_else(|| "the runtime actions are not bound".to_string())
    }

    /// Rebind the cwd the contexts read, upstream's call-time `ctx.cwd`
    /// resolution when the session replacement moves it.
    pub fn set_cwd(&mut self, cwd: String) {
        self.cwd = cwd;
    }

    /// Rebind the mode the contexts read, upstream's mode half of
    /// `setUIContext` (the UI half rides #132).
    pub const fn set_mode(&mut self, mode: ExtensionMode) {
        self.mode = mode;
    }

    /// The extension-registered flags, the first registration of a name
    /// wins across extensions, upstream's `getFlags`. Extensions read in
    /// load order and flags in registration order.
    #[must_use]
    pub fn get_flags(&self) -> Vec<crate::extensions::runtime::ExtensionFlag> {
        let mut seen: Vec<String> = Vec::new();
        let mut all_flags: Vec<crate::extensions::runtime::ExtensionFlag> = Vec::new();
        for extension in &self.extensions {
            for flag in extension.flags() {
                if seen.contains(&flag.name) {
                    continue;
                }
                seen.push(flag.name.clone());
                all_flags.push(flag);
            }
        }
        all_flags
    }

    /// Write one flag value into the loader runtime, upstream's
    /// `setFlagValue`.
    pub fn set_flag_value(&self, name: &str, value: crate::extensions::runtime::FlagValue) {
        self.runtime.set_flag_value(name, value);
    }

    /// The runtime flag values, upstream's `getFlagValues`.
    #[must_use]
    pub fn get_flag_values(&self) -> BTreeMap<String, crate::extensions::runtime::FlagValue> {
        self.runtime.flag_values()
    }

    /// Mark the runtime stale, upstream's `invalidate`: the first message
    /// wins, and the loader runtime's staleness joins the first-wins gate
    /// so the extension APIs' reads fail the same way. Without a message
    /// the runner restates upstream's guidance sentence.
    pub fn invalidate(&self, message: Option<String>) {
        let resolved = message.unwrap_or_else(|| {
            "This extension ctx is stale after session replacement or reload. Do not use a \
             captured pi or command ctx after ctx.newSession(), ctx.fork(), \
             ctx.switchSession(), or ctx.reload(). For newSession, fork, and switchSession, \
             move post-replacement work into withSession and use the ctx passed to \
             withSession. For reload, do not use the old ctx after await ctx.reload()."
                .to_string()
        });
        {
            let mut stale = self
                .stale_message
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if stale.is_none() {
                stale.replace(resolved.clone());
            }
        }
        self.runtime.invalidate(Some(resolved));
    }

    /// The load-ordered extension paths, upstream's `getExtensionPaths`.
    #[must_use]
    pub fn get_extension_paths(&self) -> Vec<String> {
        self.extensions
            .iter()
            .map(|extension| extension.path.clone())
            .collect()
    }

    /// The active-tool names, stale-checked for the wrapper's diff, upstream's
    /// `getActiveTools`.
    ///
    /// # Errors
    /// When the runtime was invalidated by session replacement or reload.
    pub fn active_tools(&self) -> Result<Vec<String>, String> {
        (self.assert_active)()?;
        Ok((self.active_tools)())
    }
    /// The context handlers and commands resolve through, upstream's
    /// `createContext`: values resolve at call time through the bound
    /// actions, and every accessor asserts the runtime is current first.
    #[must_use]
    pub fn context(&self) -> HandlerContext {
        HandlerContext::new(
            self.cwd.clone(),
            self.mode,
            self.ui.clone(),
            self.session_manager.clone(),
            self.model_registry.clone(),
            Arc::clone(&self.assert_active),
            Arc::clone(&self.actions),
        )
    }

    /// Whether any loaded extension registered a handler on the channel,
    /// upstream's `hasHandlers`.
    #[must_use]
    pub fn has_handlers(&self, channel: &str) -> bool {
        self.extensions.iter().any(|extension| {
            extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(channel)
                .is_some_and(|handlers| !handlers.is_empty())
        })
    }

    /// Subscribe a dispatch-error listener, upstream's `onError`; the
    /// returned closure unsubscribes the same listener. The empty capture
    /// list keeps the returned guard from immobilizing the runner.
    pub fn on_error(&self, listener: ExtensionErrorListener) -> impl Fn() + Send + use<> {
        let listeners = Arc::clone(&self.error_listeners);
        listeners
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Arc::clone(&listener));
        move || {
            listeners
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .retain(|listed| !Arc::ptr_eq(listed, &listener));
        }
    }

    /// Report a dispatch error to every listener, upstream's `emitError`.
    pub fn emit_error(&self, error: &ExtensionError) {
        let listeners = self
            .error_listeners
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for listener in listeners.iter() {
            listener(error);
        }
    }
}

impl ExtensionRunner {
    /// Assert the runtime is current, upstream's `assertActive` — the
    /// wrapped stale-then-seam gate.
    ///
    /// # Errors
    /// When the runtime was invalidated by session replacement or reload.
    fn assert_current(&self) -> Result<(), String> {
        (self.assert_active)()
    }

    /// The registered tools, the first registration of a name wins across
    /// extensions, upstream's `getAllRegisteredTools`. Extensions read in
    /// load order and a name reads at its first registration.
    #[must_use]
    pub fn get_all_registered_tools(&self) -> Vec<crate::extensions::runtime::RegisteredTool> {
        let mut seen: Vec<String> = Vec::new();
        let mut registered: Vec<crate::extensions::runtime::RegisteredTool> = Vec::new();
        for extension in &self.extensions {
            for tool in extension
                .tools
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .iter()
            {
                if seen.contains(&tool.definition.name) {
                    continue;
                }
                seen.push(tool.definition.name.clone());
                registered.push(tool.clone());
            }
        }
        registered
    }

    /// The tool definition a name reads at, upstream's `getToolDefinition`.
    ///
    /// # Errors
    /// When the runtime was invalidated by session replacement or reload.
    pub fn get_tool_definition(
        &self,
        tool_name: &str,
    ) -> Result<Option<crate::extensions::types::ToolDefinition>, String> {
        self.assert_current()?;

        Ok(self
            .get_all_registered_tools()
            .into_iter()
            .find(|tool| tool.definition.name == tool_name)
            .map(|tool| tool.definition))
    }

    /// The collision-renamed command list, upstream's
    /// `resolveRegisteredCommands`: a name registered more than once reads
    /// as `name:occurrence`, and an evicted invocation name walks the
    /// suffixes until it lands on a free one. Commands read in extension
    /// load order and per-extension registration order.
    fn resolve_registered_commands(&self) -> Vec<crate::extensions::runtime::ResolvedCommand> {
        let mut registered: Vec<crate::extensions::runtime::ResolvedCommand> = Vec::new();
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for extension in &self.extensions {
            for command in extension
                .commands
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .iter()
            {
                *counts.entry(command.name.clone()).or_insert(0) += 1;
                registered.push(command.clone());
            }
        }

        let mut occurrences: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut taken: std::collections::HashSet<String> = std::collections::HashSet::new();
        registered
            .into_iter()
            .map(|command| {
                let occurrence = occurrences.entry(command.name.clone()).or_insert(0);
                *occurrence += 1;
                let occurrence = *occurrence;
                let mut invocation_name = match counts.get(&command.name) {
                    Some(count) if *count > 1 => format!("{}:{}", command.name, occurrence),
                    _ => command.name.clone(),
                };
                if taken.contains(&invocation_name) {
                    let mut suffix = occurrence;
                    loop {
                        suffix += 1;
                        invocation_name = format!("{}:{}", command.name, suffix);
                        if !taken.contains(&invocation_name) {
                            break;
                        }
                    }
                }
                taken.insert(invocation_name.clone());
                crate::extensions::runtime::ResolvedCommand {
                    invocation_name,
                    ..command
                }
            })
            .collect()
    }

    /// The command list with fresh diagnostics, upstream's
    /// `getRegisteredCommands`: the read resets the diagnostics.
    #[must_use]
    pub fn get_registered_commands(&self) -> Vec<crate::extensions::runtime::ResolvedCommand> {
        self.command_diagnostics
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        self.resolve_registered_commands()
    }

    /// The command diagnostics the last registered-commands read recorded,
    /// upstream's `getCommandDiagnostics`.
    #[must_use]
    pub fn get_command_diagnostics(&self) -> Vec<ResourceDiagnostic> {
        self.command_diagnostics
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// One command by invocation name, upstream's `getCommand`.
    #[must_use]
    pub fn get_command(&self, name: &str) -> Option<crate::extensions::runtime::ResolvedCommand> {
        self.resolve_registered_commands()
            .into_iter()
            .find(|command| command.invocation_name == name)
    }

    /// Request a graceful shutdown, upstream's `shutdown`: the bound
    /// actions carry the behavior; unbound reads as no-op.
    pub fn shutdown(&self) {
        if let Some(shutdown) = self.actions.shutdown.as_ref() {
            shutdown();
        }
    }
}

/// The message role, upstream's `message.role` the `message_end` role check
/// compares: standard roles by their wire discriminator, custom messages by
/// their carried role string.
fn message_role(message: &AgentMessage) -> String {
    match message {
        AgentMessage::Standard(pi_ai::types::Message::User(_)) => "user".to_string(),
        AgentMessage::Standard(pi_ai::types::Message::Assistant(_)) => "assistant".to_string(),
        AgentMessage::Standard(pi_ai::types::Message::ToolResult(_)) => "toolResult".to_string(),
        AgentMessage::Custom(custom) => custom.role.clone(),
    }
}

/// Emit the `session_shutdown` event when any handler listens, upstream's
/// free `emitSessionShutdownEvent`.
///
/// `false` reads as "there were no handlers" so the caller may fall back to
/// app-local shutdown work.
pub async fn emit_session_shutdown_event(runner: &ExtensionRunner, event: ExtensionEvent) -> bool {
    if runner.has_handlers(event.channel()) {
        let _ = runner.emit(event).await;
        true
    } else {
        false
    }
}

/// Emit the `project_trust` gate over the loaded extensions, upstream's
/// free `emitProjectTrustEvent`.
///
/// The first handler that decided wins, an undecided result falls through,
/// and a single extension may register multiple handlers for the same
/// event.
pub async fn emit_project_trust_event(
    extensions: &[Arc<Extension>],
    event: ExtensionEvent,
    context: HandlerContext,
) -> ProjectTrustOutcome {
    let mut outcome = ProjectTrustOutcome::default();
    for extension in extensions {
        let handlers = extension
            .handlers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get("project_trust")
            .cloned()
            .unwrap_or_default();
        if handlers.is_empty() {
            continue;
        }
        for handler in handlers {
            let handler_result = (handler)(event.clone(), context.clone()).await;
            if let HandlerResult::ProjectTrust(result) = handler_result {
                if result.trusted.is_none_or(|decision| {
                    matches!(
                        decision,
                        crate::extensions::results::ProjectTrustDecision::Undecided
                    )
                }) {
                    continue;
                }
                outcome.result = Some(result);
                return outcome;
            }
            // Upstream would misread a non-project_trust result as
            // undecided and report the thrown read as a handler error;
            // the owned result collapses that to a fall-through.
        }
    }
    outcome
}

/// The command context, upstream's `ExtensionCommandContext`:
/// `createContext()` plus the idle wait and the replace/reload handlers.
///
/// Every call asserts the runtime is current first; values resolve at
/// call time so a bind lands on later calls too.
pub struct ExtensionCommandContext {
    context: HandlerContext,
    wait_for_idle: WaitForIdleHandler,
    new_session: NewSessionHandler,
    fork: ForkHandler,
    navigate_tree: NavigateTreeHandler,
    switch_session: SwitchSessionHandler,
    reload: ReloadHandler,
}

impl std::fmt::Debug for ExtensionCommandContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The handler closures do not debug; the carried context does.
        f.debug_struct("ExtensionCommandContext")
            .field("context", &self.context)
            .field("bound_handlers", &6usize)
            .finish_non_exhaustive()
    }
}

/// The command context carries the handler context's whole surface;
/// deref passes the ctx reads through, upstream's context spread.
impl std::ops::Deref for ExtensionCommandContext {
    type Target = HandlerContext;

    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

impl ExtensionCommandContext {
    /// Wait until the session is idle, upstream's `ctx.waitForIdle()`.
    ///
    /// # Errors
    /// When the runtime was invalidated by session replacement or reload.
    pub async fn wait_for_idle(&self) -> Result<(), String> {
        self.context.assert_active()?;
        (self.wait_for_idle)().await;
        Ok(())
    }

    /// Replace the session, upstream's `ctx.newSession()`.
    ///
    /// # Errors
    /// When the runtime was invalidated by session replacement or reload.
    pub async fn new_session(
        &self,
        options: NewSessionOptions,
    ) -> Result<SessionReplaceOutcome, String> {
        self.context.assert_active()?;
        Ok((self.new_session)(options).await)
    }

    /// Fork at or before an entry, upstream's `ctx.fork()`.
    ///
    /// # Errors
    /// When the runtime was invalidated by session replacement or reload.
    pub async fn fork(
        &self,
        entry_id: String,
        options: ForkOptions,
    ) -> Result<SessionReplaceOutcome, String> {
        self.context.assert_active()?;
        Ok((self.fork)(entry_id, options).await)
    }

    /// Navigate the branch tree, upstream's `ctx.navigateTree()`.
    ///
    /// # Errors
    /// When the runtime was invalidated by session replacement or reload.
    pub async fn navigate_tree(
        &self,
        target_id: String,
        options: NavigateTreeOptions,
    ) -> Result<SessionReplaceOutcome, String> {
        self.context.assert_active()?;
        Ok((self.navigate_tree)(target_id, options).await)
    }

    /// Switch to another session file, upstream's `ctx.switchSession()`.
    ///
    /// # Errors
    /// When the runtime was invalidated by session replacement or reload.
    pub async fn switch_session(
        &self,
        session_path: String,
        options: SwitchSessionOptions,
    ) -> Result<SessionReplaceOutcome, String> {
        self.context.assert_active()?;
        Ok((self.switch_session)(session_path, options).await)
    }

    /// Reload the extensions, upstream's `ctx.reload()`.
    ///
    /// # Errors
    /// When the runtime was invalidated by session replacement or reload.
    pub async fn reload(&self) -> Result<(), String> {
        self.context.assert_active()?;
        (self.reload)().await;
        Ok(())
    }
}

impl ExtensionRunner {
    /// Arm or reset the command-context handlers, upstream's
    /// `bindCommandContext`: a bundle binds all six; `None` resets the
    /// no-op defaults.
    pub fn bind_command_context(&mut self, actions: Option<CommandContextActions>) {
        self.command = actions.map_or_else(CommandHandlers::no_ops, |actions| CommandHandlers {
            wait_for_idle: actions.wait_for_idle,
            new_session: actions.new_session,
            fork: actions.fork,
            navigate_tree: actions.navigate_tree,
            switch_session: actions.switch_session,
            reload: actions.reload,
        });
    }

    /// Build the command context over the runner's current state and
    /// handlers, upstream's `createCommandContext`. Upstream restates
    /// `getSystemPromptOptions` on this context separately because its own
    /// plain context lacks it; the port's handler context already carries
    /// it, so the deref passes it through.
    #[must_use]
    pub fn command_context(&self) -> ExtensionCommandContext {
        ExtensionCommandContext {
            context: self.context(),
            wait_for_idle: Arc::clone(&self.command.wait_for_idle),
            new_session: Arc::clone(&self.command.new_session),
            fork: Arc::clone(&self.command.fork),
            navigate_tree: Arc::clone(&self.command.navigate_tree),
            switch_session: Arc::clone(&self.command.switch_session),
            reload: Arc::clone(&self.command.reload),
        }
    }
}

impl std::fmt::Debug for ExtensionRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let extensions = self
            .extensions
            .iter()
            .map(|extension| extension.path.clone())
            .collect::<Vec<_>>();
        let listeners = self
            .error_listeners
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        f.debug_struct("ExtensionRunner")
            .field("extensions", &extensions)
            .field("cwd", &self.cwd)
            .field("mode", &self.mode)
            .field("actions", &self.actions)
            .field("error_listeners", &listeners)
            .finish_non_exhaustive()
    }
}

impl ExtensionRunner {
    /// The generic dispatch, upstream's `emit\<TEvent\>`: run every handler of
    /// the event's channel over the extensions in request order, handlers in
    /// registration order. On the `session_before_*` channels the last
    /// defined result wins and a cancelled result short-circuits the
    /// remaining handlers; other channels always dispatch regardless of any
    /// returned value and yield no result. Channel handlers nil out to no
    /// result.
    pub async fn emit(&self, event: ExtensionEvent) -> SessionBeforeResult {
        let merging = is_session_before(&event);
        let mut result: Option<SessionBeforeResult> = None;
        for extension in &self.extensions {
            let handlers = extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(event.channel())
                .cloned()
                .unwrap_or_default();
            if handlers.is_empty() {
                continue;
            }
            let context = self.context();
            for handler in handlers {
                let handler_result = (handler)(event.clone(), context.clone()).await;
                if let Some(resolved) = session_before_result(&handler_result) {
                    if cancels(&resolved) {
                        return resolved;
                    }
                    if merging {
                        result = Some(resolved);
                    }
                }
            }
        }
        result.unwrap_or(SessionBeforeResult::None)
    }

    /// The `message_end` replacement chain, upstream's `emitMessageEnd`:
    /// handlers chain replacements, and a replacement carrying a different
    /// role is rejected with a report to the error listeners. Returns the
    /// final message, `None` when no handler replaced it.
    pub async fn emit_message_end(&self, message: AgentMessage) -> Option<AgentMessage> {
        let mut results: Option<AgentMessage> = None;
        let mut current: AgentMessage = message.clone();
        for extension in &self.extensions {
            let handlers = extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get("message_end")
                .cloned()
                .unwrap_or_default();
            if handlers.is_empty() {
                continue;
            }
            let context = self.context();
            for handler in handlers {
                let event = ExtensionEvent::MessageEnd {
                    message: current.clone(),
                };
                let handler_result = (handler)(event, context.clone()).await;
                let HandlerResult::MessageEnd(result) = handler_result else {
                    continue;
                };
                let Some(replacement) = result.message else {
                    continue;
                };
                if message_role(&replacement) != message_role(&current) {
                    self.emit_error(&ExtensionError {
                        extension_path: extension.path.clone(),
                        event: "message_end".to_string(),
                        error: "message_end handlers must return a message with the same role"
                            .to_string(),
                        stack: None,
                    });
                    continue;
                }
                current = replacement;
                results = Some(current.clone());
            }
        }
        results
    }

    /// The `tool_result` merge fold, upstream's `emitToolResult`: each
    /// handler's defined fields patch the running value, and a handler that
    /// started from the running value sees the current state. Returns the
    /// merged fields, `None` when no handler defined any.
    pub async fn emit_tool_result(&self, event: ExtensionEvent) -> Option<ToolResultResult> {
        let ExtensionEvent::ToolResult {
            tool_name,
            tool_call_id,
            input,
            mut content,
            mut details,
            mut is_error,
            mut usage,
        } = event
        else {
            return None;
        };
        let mut modified = false;
        for extension in &self.extensions {
            let handlers = extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get("tool_result")
                .cloned()
                .unwrap_or_default();
            if handlers.is_empty() {
                continue;
            }
            let context = self.context();
            for handler in handlers {
                let current_event = ExtensionEvent::ToolResult {
                    tool_name: tool_name.clone(),
                    tool_call_id: tool_call_id.clone(),
                    input: input.clone(),
                    content: content.clone(),
                    details: details.clone(),
                    is_error,
                    usage,
                };
                let handler_result = (handler)(current_event, context.clone()).await;
                let HandlerResult::ToolResult(result) = handler_result else {
                    continue;
                };
                if let Some(patch) = result.content {
                    content = patch;
                    modified = true;
                }
                if let Some(patch) = result.details {
                    details = patch;
                    modified = true;
                }
                if let Some(patch) = result.is_error {
                    is_error = patch;
                    modified = true;
                }
                if let Some(patch) = result.usage {
                    usage = Some(patch);
                    modified = true;
                }
            }
        }
        if !modified {
            return None;
        }
        Some(ToolResultResult {
            content: Some(content),
            details: Some(details),
            is_error: Some(is_error),
            usage,
        })
    }
    /// The `tool_call` decision fold, upstream's `emitToolCall`: the last
    /// defined result wins and a `block` short-circuits the remaining
    /// handlers. Takes the owned call event so handlers see the raw input.
    pub async fn emit_tool_call(&self, event: ExtensionEvent) -> Option<ToolCallResult> {
        let mut result: Option<ToolCallResult> = None;
        for extension in &self.extensions {
            let handlers = extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get("tool_call")
                .cloned()
                .unwrap_or_default();
            if handlers.is_empty() {
                continue;
            }
            let context = self.context();
            for handler in handlers {
                let handler_result = (handler)(event.clone(), context.clone()).await;
                let HandlerResult::ToolCall(call_result) = handler_result else {
                    continue;
                };
                result = Some(call_result);
                if result.as_ref().is_some_and(|resolved| resolved.block) {
                    return result;
                }
            }
        }
        result
    }

    /// A user-initiated bash command about to run, upstream's
    /// `emitUserBash`: the first handler that returned a decision wins.
    pub async fn emit_user_bash(&self, command: String) -> Option<UserBashResult> {
        let event = ExtensionEvent::UserBash { command };
        for extension in &self.extensions {
            let handlers = extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get("user_bash")
                .cloned()
                .unwrap_or_default();
            if handlers.is_empty() {
                continue;
            }
            let context = self.context();
            for handler in handlers {
                let handler_result = (handler)(event.clone(), context.clone()).await;
                let HandlerResult::UserBash(bash_result) = handler_result else {
                    continue;
                };
                return Some(bash_result);
            }
        }
        None
    }

    /// The context replacement chain, upstream's `emitContext`: handlers
    /// chain the message list, `None` results keep the running list.
    pub async fn emit_context(&self, messages: Vec<AgentMessage>) -> Vec<AgentMessage> {
        let mut current = messages;
        for extension in &self.extensions {
            let handlers = extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get("context")
                .cloned()
                .unwrap_or_default();
            if handlers.is_empty() {
                continue;
            }
            let context = self.context();
            for handler in handlers {
                let event = ExtensionEvent::Context {
                    messages: current.clone(),
                };
                let handler_result = (handler)(event, context.clone()).await;
                let HandlerResult::Context(context_result) = handler_result else {
                    continue;
                };
                if let Some(replacement) = context_result.messages {
                    current = replacement;
                }
            }
        }
        current
    }
    /// The provider-request payload chain, upstream's
    /// `emitBeforeProviderRequest`: the first handler that returns a
    /// replacement starts the chain and later handlers restate from it.
    pub async fn emit_before_provider_request(&self, payload: Value) -> Value {
        let mut current = payload;
        for extension in &self.extensions {
            let handlers = extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get("before_provider_request")
                .cloned()
                .unwrap_or_default();
            if handlers.is_empty() {
                continue;
            }
            let context = self.context();
            for handler in handlers {
                let event = ExtensionEvent::BeforeProviderRequest {
                    payload: current.clone(),
                };
                let handler_result = (handler)(event, context.clone()).await;
                if let HandlerResult::BeforeProviderRequest(replacement) = handler_result {
                    current = replacement;
                }
            }
        }
        current
    }

    /// The request-headers restatement, upstream's `emitBeforeProviderHeaders`:
    /// handlers mutate the headers in place and the return value is ignored.
    /// The owned restate carries a mutant's replacement headers; a handler
    /// returning none keeps the running map.
    pub async fn emit_before_provider_headers(&self, headers: ProviderHeaders) -> ProviderHeaders {
        let mut current = headers;
        for extension in &self.extensions {
            let handlers = extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get("before_provider_headers")
                .cloned()
                .unwrap_or_default();
            if handlers.is_empty() {
                continue;
            }
            let context = self.context();
            for handler in handlers {
                let event = ExtensionEvent::BeforeProviderHeaders {
                    headers: current.clone(),
                };
                let handler_result = (handler)(event, context.clone()).await;
                if let HandlerResult::BeforeProviderHeaders(replacement) = handler_result {
                    current = replacement;
                }
            }
        }
        current
    }
    /// The before-agent-start preflight, upstream's `emitBeforeAgentStart`:
    /// handlers queue custom messages and chain the system prompt, and the
    /// context's `getSystemPrompt` reads the running replacement. Returns
    /// the combined result when any handler contributed.
    pub async fn emit_before_agent_start(
        &self,
        prompt: String,
        images: Option<Vec<ImageContent>>,
        system_prompt: String,
        system_prompt_options: crate::system_prompt::BuildSystemPromptOptions,
    ) -> Option<BeforeAgentStartCombined> {
        let running_prompt = Arc::new(Mutex::new(system_prompt.clone()));
        let mut actions = (*self.actions).clone();
        actions.get_system_prompt = Some({
            let running_prompt = Arc::clone(&running_prompt);
            Arc::new(move || {
                running_prompt
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clone()
            })
        });
        let actions = Arc::new(actions);
        let mut accumulated: Vec<CustomMessagePayload> = Vec::new();
        let mut system_prompt_modified = false;
        for extension in &self.extensions {
            let handlers = extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get("before_agent_start")
                .cloned()
                .unwrap_or_default();
            if handlers.is_empty() {
                continue;
            }
            let context = HandlerContext::new(
                self.cwd.clone(),
                self.mode,
                self.ui.clone(),
                self.session_manager.clone(),
                self.model_registry.clone(),
                Arc::clone(&self.assert_active),
                Arc::clone(&actions),
            );
            for handler in handlers {
                let event = ExtensionEvent::BeforeAgentStart {
                    prompt: prompt.clone(),
                    images: images.clone(),
                    system_prompt: running_prompt
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone(),
                    system_prompt_options: system_prompt_options.clone(),
                    signal: None,
                };
                let handler_result = (handler)(event, context.clone()).await;
                let HandlerResult::BeforeAgentStart(result) = handler_result else {
                    continue;
                };
                if let Some(queued) = result.message.clone() {
                    accumulated.push(queued);
                }
                if let Some(replacement) = result.system_prompt {
                    *running_prompt
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = replacement;
                    system_prompt_modified = true;
                }
            }
        }
        if accumulated.is_empty() && !system_prompt_modified {
            return None;
        }
        let system_prompt = system_prompt_modified.then(|| {
            running_prompt
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        });
        Some(BeforeAgentStartCombined {
            messages: accumulated,
            system_prompt,
        })
    }
    /// The resource-discovery contributions, upstream's
    /// `emitResourcesDiscover`: contributed paths carry the contributing
    /// extension, and empty contributions are skipped.
    pub async fn emit_resources_discover(
        &self,
        cwd: String,
        reason: SessionStartReason,
    ) -> ResourcesDiscoverPaths {
        let mut paths = ResourcesDiscoverPaths::default();
        for extension in &self.extensions {
            let handlers = extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get("resources_discover")
                .cloned()
                .unwrap_or_default();
            if handlers.is_empty() {
                continue;
            }
            let context = self.context();
            for handler in handlers {
                let event = ExtensionEvent::ResourcesDiscover {
                    cwd: cwd.clone(),
                    reason,
                };
                let handler_result = (handler)(event, context.clone()).await;
                let HandlerResult::ResourcesDiscover(result) = handler_result else {
                    continue;
                };
                for side in 0..3 {
                    let contributed: &Vec<String> = match side {
                        0 => &result.skill_paths,
                        1 => &result.prompt_paths,
                        _ => &result.theme_paths,
                    };
                    let sink = match side {
                        0 => &mut paths.skill_paths,
                        1 => &mut paths.prompt_paths,
                        _ => &mut paths.theme_paths,
                    };
                    sink.extend(contributed.iter().map(|path| ExtensionResourcePath {
                        path: path.clone(),
                        extension_path: extension.path.clone(),
                    }));
                }
            }
        }
        paths
    }

    /// The input pipeline, upstream's `emitInput`: handler transforms
    /// chain the text and images, `handled` short-circuits, and a chain
    /// that modified anything restates as a final transform.
    pub async fn emit_input(
        &self,
        text: String,
        images: Option<Vec<ImageContent>>,
        source: InputSource,
        streaming_behavior: Option<StreamingBehavior>,
    ) -> InputResult {
        let mut current_text = text.clone();
        let mut current_images = images.clone();
        let mut images_modified = false;
        for extension in &self.extensions {
            let handlers = extension
                .handlers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get("input")
                .cloned()
                .unwrap_or_default();
            if handlers.is_empty() {
                continue;
            }
            let context = self.context();
            for handler in handlers {
                let event = ExtensionEvent::Input {
                    text: current_text.clone(),
                    images: current_images.clone(),
                    source,
                    streaming_behavior,
                };
                let handler_result = (handler)(event, context.clone()).await;
                let HandlerResult::Input(result) = handler_result else {
                    continue;
                };
                match result {
                    InputResult::Handled => return InputResult::Handled,
                    InputResult::Transform { text, images } => {
                        current_text = text;
                        if let Some(replacement) = images {
                            current_images = Some(replacement);
                            images_modified = true;
                        }
                    }
                    InputResult::Continue => {}
                }
            }
        }
        if current_text != text || images_modified {
            InputResult::Transform {
                text: current_text,
                images: images_modified.then_some(current_images).flatten(),
            }
        } else {
            InputResult::Continue
        }
    }
}

#[cfg(test)]
mod tests;

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
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "the tests pin outcomes; an unexpected result panics the test by design"
    )]
    use std::sync::Arc;
    use std::sync::Mutex;

    use pi_agent_core::types::AgentMessage;
    use serde_json::{Value, json};

    use crate::extensions::events::ExtensionEvent;
    use crate::extensions::reasons::{
        CustomMessagePayload, ExtensionMode, InputSource, SessionShutdownReason, StreamingBehavior,
        text_block,
    };
    use crate::extensions::results::{
        HandlerResult, InputResult, MessageEndResult, ProjectTrustDecision, ProjectTrustResult,
        SessionBeforeResult, ToolCallResult, ToolResultResult,
    };
    use crate::extensions::runner::{
        CommandContextActions, DeliverAs, ExtensionActions, ExtensionRunner, ForkOptions,
        ForkPosition, NavigateTreeOptions, NewSessionOptions, ProjectTrustOutcome, ProviderActions,
        SendMessageOptions, SendUserMessageOptions, SessionReplaceOutcome, SwitchSessionOptions,
        emit_project_trust_event, emit_session_shutdown_event,
    };
    use crate::extensions::runtime::{
        AssertActiveFn, ContextActions, Extension, ExtensionError, ExtensionFlag, ExtensionHandler,
        FlagType, FlagValue, HandlerContext,
    };
    use crate::model_registry::ModelRegistry;
    use crate::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
    use crate::provider_composer::ProviderConfigInput;
    use crate::session_manager::SessionManager;
    use crate::source_info::{SourceInfo, SourceOrigin, SourceScope, SyntheticSourceOptions};
    use crate::tools::bash::create_bash_tool_definition;
    use pi_agent_core::types::ThinkingLevel;
    use pi_ai::auth::credential_store::InMemoryCredentialStore;
    use pi_ai::types::{Api, Model, ProviderId, Usage, UserContent};

    type EventLog = Arc<Mutex<Vec<&'static str>>>;

    fn source_info() -> SourceInfo {
        crate::source_info::create_synthetic_source_info(
            "/ext/test.ts",
            &SyntheticSourceOptions {
                source: "extension:test".to_string(),
                scope: Some(SourceScope::Temporary),
                origin: Some(SourceOrigin::TopLevel),
                base_dir: None,
            },
        )
    }

    fn plain_manager() -> SessionManager {
        SessionManager::new("/tmp/project", "/tmp/sessions", None, false, None, None)
            .expect("the in-memory manager constructs")
    }

    async fn offline_registry() -> ModelRegistry {
        let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
            credentials: Some(Arc::new(InMemoryCredentialStore::default())),
            models_path: Some(None),
            allow_model_network: false,
            refresh_on_create: Some(false),
            ..CreateModelRuntimeOptions::default()
        })
        .await
        .expect("the runtime constructs offline");
        ModelRegistry::new(runtime)
    }

    async fn runner_with(
        extensions: &[Arc<Extension>],
        actions: Arc<ContextActions>,
    ) -> ExtensionRunner {
        ExtensionRunner::new(
            extensions.to_vec(),
            actions,
            Arc::new(|| Ok(())),
            "/tmp/project".to_string(),
            ExtensionMode::Print,
            None,
            plain_manager(),
            offline_registry().await,
            crate::extensions::api::ExtensionRuntime::new(),
        )
    }

    fn plain_actions() -> Arc<ContextActions> {
        Arc::new(ContextActions::default())
    }

    fn logging_handler(
        log: &EventLog,
        tag: &'static str,
        result: HandlerResult,
    ) -> ExtensionHandler {
        let log = Arc::clone(log);
        Arc::new(move |_event: ExtensionEvent, _ctx: HandlerContext| {
            let log = Arc::clone(&log);
            let result = result.clone();
            Box::pin(async move {
                log.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(tag);
                result
            })
        })
    }

    fn extension_with(entries: Vec<(&'static str, ExtensionHandler)>) -> Arc<Extension> {
        let extension = Extension::new(
            "/ext/test.ts".to_string(),
            "/ext/test.ts".to_string(),
            false,
            source_info(),
        );
        for (channel, handler) in entries {
            extension.on(channel, handler);
        }
        Arc::new(extension)
    }
    fn assistant_message() -> AgentMessage {
        AgentMessage::Standard(pi_ai::types::Message::Assistant(
            pi_ai::types::AssistantMessage {
                content: Vec::new(),
                api: Api::from("openai"),
                provider: ProviderId::from("openai"),
                model: "gpt-test".to_string(),
                response_model: None,
                response_id: None,
                provider_thinking_level: None,
                diagnostics: None,
                usage: Usage::default(),
                stop_reason: pi_ai::types::StopReason::Stop,
                deferred: None,
                error_message: None,
                raw_stop_reason: None,
                end_turn: None,
                timestamp: 0,
            },
        ))
    }

    fn user_message() -> AgentMessage {
        AgentMessage::Standard(pi_ai::types::Message::User(pi_ai::types::UserMessage {
            content: UserContent::Text(String::new()),
            timestamp: 0,
        }))
    }

    fn assistant_text(text: &str) -> pi_ai::types::AssistantBlock {
        pi_ai::types::AssistantBlock::Text(pi_ai::types::TextContent {
            text: text.to_string(),
            text_signature: None,
        })
    }

    #[tokio::test]
    async fn emit_walks_extensions_then_handlers_and_yields_none_outside_session_before() {
        let log: EventLog = Arc::new(Mutex::new(Vec::new()));
        let first = extension_with(vec![
            (
                "agent_start",
                logging_handler(&log, "e1-h1", HandlerResult::None),
            ),
            (
                "agent_start",
                logging_handler(&log, "e1-h2", HandlerResult::None),
            ),
        ]);
        let second = extension_with(vec![(
            "agent_start",
            logging_handler(&log, "e2-h1", HandlerResult::None),
        )]);
        let runner = runner_with(&[first, second], plain_actions()).await;

        let outcome = runner.emit(ExtensionEvent::AgentStart).await;
        assert!(matches!(outcome, SessionBeforeResult::None));
        assert_eq!(
            *log.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            vec!["e1-h1", "e1-h2", "e2-h1"]
        );
    }

    #[tokio::test]
    async fn emit_merges_session_before_last_defined_result_wins() {
        let first = extension_with(vec![(
            "session_before_switch",
            always_handler(HandlerResult::SessionBefore(SessionBeforeResult::Switch {
                cancel: false,
            })),
        )]);
        let second = extension_with(vec![(
            "session_before_switch",
            always_handler(HandlerResult::SessionBefore(SessionBeforeResult::Switch {
                cancel: false,
            })),
        )]);
        // The following replacement also wins over the earlier result.
        let runner = runner_with(&[first, second], plain_actions()).await;

        let outcome = runner
            .emit(ExtensionEvent::SessionBeforeSwitch {
                reason: SessionShutdownReason::Quit,
                target_session_file: None,
            })
            .await;
        assert!(
            matches!(outcome, SessionBeforeResult::Switch { cancel: false }),
            "{outcome:?}"
        );
    }

    fn always_handler(result: HandlerResult) -> ExtensionHandler {
        Arc::new(move |_event: ExtensionEvent, _ctx: HandlerContext| {
            let result = result.clone();
            Box::pin(async move { result })
        })
    }

    #[tokio::test]
    async fn emit_cancel_short_circuits_the_remaining_handlers() {
        let log: EventLog = Arc::new(Mutex::new(Vec::new()));
        let first = extension_with(vec![(
            "session_before_switch",
            always_handler(HandlerResult::SessionBefore(SessionBeforeResult::Switch {
                cancel: true,
            })),
        )]);
        let second = extension_with(vec![(
            "session_before_switch",
            logging_handler(&log, "skipped", HandlerResult::None),
        )]);
        let runner = runner_with(&[first, second], plain_actions()).await;

        let outcome = runner
            .emit(ExtensionEvent::SessionBeforeSwitch {
                reason: SessionShutdownReason::Quit,
                target_session_file: None,
            })
            .await;
        assert!(
            matches!(outcome, SessionBeforeResult::Switch { cancel: true }),
            "{outcome:?}"
        );
        assert_eq!(
            *log.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            Vec::<&str>::new()
        );
    }

    #[tokio::test]
    async fn emit_tool_result_merges_fields_across_handlers() {
        let content_patch = HandlerResult::ToolResult(ToolResultResult {
            content: Some(vec![assistant_text("patched")]),
            ..ToolResultResult::default()
        });
        let error_patch = HandlerResult::ToolResult(ToolResultResult {
            is_error: Some(true),
            ..ToolResultResult::default()
        });
        let first = extension_with(vec![("tool_result", always_handler(content_patch))]);
        let second = extension_with(vec![("tool_result", always_handler(error_patch))]);
        let runner = runner_with(&[first, second], plain_actions()).await;

        let outcome = runner
            .emit_tool_result(ExtensionEvent::ToolResult {
                tool_name: "read".to_string(),
                tool_call_id: "call-1".to_string(),
                input: json!({ "path": "/x" }),
                content: vec![assistant_text("original")],
                details: json!({ "ok": true }),
                is_error: false,
                usage: None,
            })
            .await;
        let merged = outcome.expect("a handler modified the result");
        assert_eq!(
            merged.content.as_deref(),
            Some(&[assistant_text("patched")][..])
        );
        assert_eq!(merged.is_error, Some(true));
        assert_eq!(merged.details, Some(json!({ "ok": true })));
    }

    #[tokio::test]
    async fn emit_tool_result_yields_none_without_modification() {
        let first = extension_with(vec![("tool_result", always_handler(HandlerResult::None))]);
        let runner = runner_with(&[first], plain_actions()).await;

        let outcome = runner
            .emit_tool_result(ExtensionEvent::ToolResult {
                tool_name: "read".to_string(),
                tool_call_id: "call-1".to_string(),
                input: Value::Null,
                content: Vec::new(),
                details: Value::Null,
                is_error: false,
                usage: None,
            })
            .await;
        assert!(outcome.is_none());
    }

    #[tokio::test]
    async fn emit_tool_call_block_short_circuits_and_last_defined_wins() {
        let log: EventLog = Arc::new(Mutex::new(Vec::new()));
        let unblock = always_handler(HandlerResult::ToolCall(ToolCallResult {
            block: false,
            ..ToolCallResult::default()
        }));
        let block = always_handler(HandlerResult::ToolCall(ToolCallResult {
            block: true,
            reason: Some("blocked".to_string()),
            ..ToolCallResult::default()
        }));
        let first = extension_with(vec![("tool_call", unblock)]);
        let second = extension_with(vec![
            ("tool_call", block.clone()),
            (
                "tool_call",
                logging_handler(&log, "after-block", HandlerResult::None),
            ),
        ]);
        let runner = runner_with(&[Arc::clone(&first), second], plain_actions()).await;

        let outcome = runner
            .emit_tool_call(ExtensionEvent::ToolCall {
                tool_name: "bash".to_string(),
                tool_call_id: "call-1".to_string(),
                input: json!({ "command": "rm -rf /" }),
            })
            .await;
        let blocked = outcome.expect("the block short-circuits");
        assert!(blocked.block);
        assert_eq!(blocked.reason.as_deref(), Some("blocked"));
        assert_eq!(
            *log.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            Vec::<&str>::new()
        );

        // Without a block the last defined result wins.
        let last = always_handler(HandlerResult::ToolCall(ToolCallResult {
            block: false,
            reason: Some("last".to_string()),
            ..ToolCallResult::default()
        }));
        let replace = extension_with(vec![("tool_call", last)]);
        let runner = runner_with(&[Arc::clone(&first), replace], plain_actions()).await;
        let outcome = runner
            .emit_tool_call(ExtensionEvent::ToolCall {
                tool_name: "bash".to_string(),
                tool_call_id: "call-2".to_string(),
                input: Value::Null,
            })
            .await;
        assert_eq!(
            outcome.and_then(|resolved| resolved.reason),
            Some("last".to_string())
        );
    }

    #[tokio::test]
    async fn emit_user_bash_first_decision_wins() {
        let first = extension_with(vec![(
            "user_bash",
            always_handler(HandlerResult::UserBash(
                crate::extensions::results::UserBashResult { handled: true },
            )),
        )]);
        let second = extension_with(vec![(
            "user_bash",
            always_handler(HandlerResult::UserBash(
                crate::extensions::results::UserBashResult { handled: false },
            )),
        )]);
        let runner = runner_with(&[first, second], plain_actions()).await;

        let outcome = runner.emit_user_bash("cargo test".to_string()).await;
        assert_eq!(outcome.map(|resolved| resolved.handled), Some(true));
    }
    #[tokio::test]
    async fn emit_context_chains_message_replacements() {
        let replace_with_user = |tag: &'static str| {
            let replacement = user_message_with(tag);
            always_handler(HandlerResult::Context(
                crate::extensions::results::ContextResult {
                    messages: Some(vec![replacement]),
                },
            ))
        };
        let first = extension_with(vec![("context", replace_with_user("first"))]);
        let second = extension_with(vec![("context", replace_with_user("second"))]);
        let runner = runner_with(&[first, second], plain_actions()).await;

        let outcome = runner.emit_context(vec![assistant_message()]).await;
        assert_eq!(outcome, vec![user_message_with("second")]);
    }

    fn user_message_with(tag: &'static str) -> AgentMessage {
        AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage {
            role: tag.to_string(),
            timestamp: 0,
            data: serde_json::Map::new(),
        })
    }

    #[tokio::test]
    async fn emit_before_provider_request_chains_replacements() {
        let first = extension_with(vec![(
            "before_provider_request",
            always_handler(HandlerResult::BeforeProviderRequest(json!({
                "model": "step-1"
            }))),
        )]);
        let second = extension_with(vec![(
            "before_provider_request",
            always_handler(HandlerResult::BeforeProviderRequest(json!({
                "model": "step-2"
            }))),
        )]);
        let runner = runner_with(&[Arc::clone(&first), second], plain_actions()).await;

        let outcome = runner
            .emit_before_provider_request(json!({ "model": "original" }))
            .await;
        assert_eq!(outcome, json!({ "model": "step-2" }));

        // A handler returning none keeps the running payload.
        let passthrough = extension_with(vec![(
            "before_provider_request",
            always_handler(HandlerResult::None),
        )]);
        let runner = runner_with(&[Arc::clone(&first), passthrough], plain_actions()).await;
        let outcome = runner
            .emit_before_provider_request(json!({ "model": "original" }))
            .await;
        assert_eq!(outcome, json!({ "model": "step-1" }));
    }
    #[tokio::test]
    async fn emit_before_provider_headers_restates_mutations() {
        let headers_renamed = always_handler(HandlerResult::BeforeProviderHeaders(
            pi_ai::types::ProviderHeaders::new(),
        ));
        let first = extension_with(vec![("before_provider_headers", headers_renamed)]);
        let passthrough = extension_with(vec![(
            "before_provider_headers",
            always_handler(HandlerResult::None),
        )]);
        let runner = runner_with(&[first, passthrough], plain_actions()).await;

        let outcome = runner
            .emit_before_provider_headers(pi_ai::types::ProviderHeaders::new())
            .await;
        assert!(
            format!("{outcome:?}").is_empty() || !format!("{outcome:?}").contains("Authorization")
        );
    }

    #[tokio::test]
    async fn emit_before_agent_start_chains_prompt_and_queues_messages() {
        let first = extension_with(vec![(
            "before_agent_start",
            always_handler(HandlerResult::BeforeAgentStart(
                crate::extensions::results::BeforeAgentStartResult {
                    message: Some(custom_payload("one")),
                    system_prompt: Some("prompt-1".to_string()),
                },
            )),
        )]);
        let second = extension_with(vec![(
            "before_agent_start",
            always_handler(HandlerResult::BeforeAgentStart(
                crate::extensions::results::BeforeAgentStartResult {
                    message: Some(custom_payload("two")),
                    system_prompt: Some("prompt-2".to_string()),
                },
            )),
        )]);
        let runner = runner_with(&[first, second], plain_actions()).await;

        let combined = runner
            .emit_before_agent_start(
                "ship it".to_string(),
                None,
                "base".to_string(),
                crate::system_prompt::BuildSystemPromptOptions::default(),
            )
            .await
            .expect("the handlers contributed");
        assert_eq!(combined.messages.len(), 2);
        assert_eq!(combined.system_prompt.as_deref(), Some("prompt-2"));
    }

    #[tokio::test]
    async fn emit_before_agent_start_context_reads_the_running_prompt() {
        let reads_prompt: ExtensionHandler =
            Arc::new(|_event: ExtensionEvent, ctx: HandlerContext| {
                Box::pin(async move {
                    let prompt = ctx.get_system_prompt().unwrap_or_default();
                    HandlerResult::BeforeAgentStart(
                        crate::extensions::results::BeforeAgentStartResult {
                            message: None,
                            system_prompt: if prompt == "running" {
                                Some("after".to_string())
                            } else {
                                None
                            },
                        },
                    )
                })
            });
        let first = extension_with(vec![(
            "before_agent_start",
            always_handler(HandlerResult::BeforeAgentStart(
                crate::extensions::results::BeforeAgentStartResult {
                    message: None,
                    system_prompt: Some("running".to_string()),
                },
            )),
        )]);
        let second = extension_with(vec![("before_agent_start", reads_prompt)]);
        let runner = runner_with(&[first, second], plain_actions()).await;

        let combined = runner
            .emit_before_agent_start(
                "ship it".to_string(),
                None,
                "base".to_string(),
                crate::system_prompt::BuildSystemPromptOptions::default(),
            )
            .await
            .expect("the prompt replacement is a contribution");
        assert_eq!(combined.system_prompt.as_deref(), Some("after"));
        assert!(combined.messages.is_empty());
    }

    fn custom_payload(tag: &'static str) -> CustomMessagePayload {
        CustomMessagePayload {
            custom_type: tag.to_string(),
            content: vec![text_block(tag)],
            display: true,
            details: None,
        }
    }
    #[tokio::test]
    async fn emit_resources_discover_carries_provenance_and_skips_empty() {
        let first = extension_with(vec![(
            "resources_discover",
            always_handler(HandlerResult::ResourcesDiscover(
                crate::extensions::results::ResourcesDiscoverResult {
                    skill_paths: vec!["/skills/a".to_string()],
                    ..crate::extensions::results::ResourcesDiscoverResult::default()
                },
            )),
        )]);
        let second = extension_with(vec![(
            "resources_discover",
            always_handler(HandlerResult::ResourcesDiscover(
                crate::extensions::results::ResourcesDiscoverResult {
                    prompt_paths: vec!["/prompts/b".to_string()],
                    ..crate::extensions::results::ResourcesDiscoverResult::default()
                },
            )),
        )]);
        second.set_hidden(true);
        let runner = runner_with(&[first, second], plain_actions()).await;

        let paths = runner
            .emit_resources_discover(
                "/tmp/project".to_string(),
                crate::extensions::reasons::SessionStartReason::Startup,
            )
            .await;
        assert_eq!(paths.skill_paths.len(), 1);
        assert_eq!(paths.skill_paths[0].path, "/skills/a");
        assert_eq!(paths.skill_paths[0].extension_path, "/ext/test.ts");
        assert_eq!(paths.prompt_paths.len(), 1);
        assert_eq!(paths.prompt_paths[0].path, "/prompts/b");
        assert!(paths.theme_paths.is_empty());
        let rendered = format!("{runner:?}");
        assert!(rendered.contains("ExtensionRunner"), "{rendered}");
    }

    #[tokio::test]
    async fn emit_input_chains_transforms_and_short_circuits_on_handled() {
        let transform = HandlerResult::Input(InputResult::Transform {
            text: "expanded".to_string(),
            images: None,
        });
        let first = extension_with(vec![("input", always_handler(transform))]);
        let handled = extension_with(vec![(
            "input",
            always_handler(HandlerResult::Input(InputResult::Handled)),
        )]);
        let runner = runner_with(&[Arc::clone(&first), handled], plain_actions()).await;
        assert!(matches!(
            runner
                .emit_input("hi".to_string(), None, InputSource::Interactive, None)
                .await,
            InputResult::Handled
        ));

        // A pure-changed chain restates as a final transform.
        let second_transform = HandlerResult::Input(InputResult::Transform {
            text: "expanded-twice".to_string(),
            images: Some(Vec::new()),
        });
        let second = extension_with(vec![("input", always_handler(second_transform))]);
        let runner = runner_with(&[first, second], plain_actions()).await;
        assert!(matches!(
            runner
                .emit_input("hi".to_string(), None, InputSource::Rpc, Some(StreamingBehavior::FollowUp))
                .await,
            InputResult::Transform {
                text,
                images: Some(_)
            } if text == "expanded-twice"
        ));

        // No handler action restates as continue.
        let unbound_runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
        assert!(matches!(
            unbound_runner
                .emit_input("hi".to_string(), None, InputSource::Interactive, None)
                .await,
            InputResult::Continue
        ));
    }
    #[tokio::test]
    async fn emit_message_end_chains_and_rejects_role_changes() {
        let errors: Arc<Mutex<Vec<ExtensionError>>> = Arc::new(Mutex::new(Vec::new()));
        let replacement = HandlerResult::MessageEnd(MessageEndResult {
            message: Some(assistant_message()),
        });
        let first = extension_with(vec![("message_end", always_handler(replacement))]);
        let role_swap = HandlerResult::MessageEnd(MessageEndResult {
            message: Some(user_message()),
        });
        let second = extension_with(vec![("message_end", always_handler(role_swap))]);
        let runner = runner_with(&[first, second], plain_actions()).await;
        let _unsubscribe = runner.on_error(Arc::new({
            let errors = Arc::clone(&errors);
            move |error: &ExtensionError| {
                errors
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(error.clone());
            }
        }));

        let outcome = runner.emit_message_end(assistant_message()).await;
        assert!(outcome.is_some(), "the same-role replacement chains");
        let recorded = Arc::clone(&errors)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(
            recorded[0].error,
            "message_end handlers must return a message with the same role"
        );
        assert_eq!(recorded[0].event, "message_end");
        assert_eq!(recorded[0].extension_path, "/ext/test.ts");
    }

    #[tokio::test]
    async fn on_error_unsubscribe_stops_listening() {
        let runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
        // The unbound active-tool reader answers empty, the default the
        // wrapper reads before a bind lands.
        assert_eq!(
            runner.active_tools().expect("the unbound reader"),
            Vec::<String>::new()
        );
        let hits: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let listener_hits = Arc::clone(&hits);
        let unsubscribe = runner.on_error(Arc::new({
            let listener_hits = Arc::clone(&listener_hits);
            move |_: &ExtensionError| {
                listener_hits
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push("one");
            }
        }));
        runner.emit_error(&ExtensionError {
            extension_path: "/ext/test.ts".to_string(),
            event: "session_start".to_string(),
            error: "boom".to_string(),
            stack: None,
        });
        assert_eq!(
            hits.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
            1
        );
        unsubscribe();
        runner.emit_error(&ExtensionError {
            extension_path: "/ext/test.ts".to_string(),
            event: "session_start".to_string(),
            error: "boom".to_string(),
            stack: None,
        });
        assert_eq!(
            hits.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn emit_session_shutdown_event_gates_on_registered_handlers() {
        let runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
        let emitted = emit_session_shutdown_event(
            &runner,
            ExtensionEvent::SessionShutdown {
                reason: SessionShutdownReason::Quit,
                target_session_file: None,
            },
        )
        .await;
        assert!(!emitted, "no handlers registered");

        let with_handlers = extension_with(vec![(
            "session_shutdown",
            always_handler(HandlerResult::None),
        )]);
        let runner = runner_with(&[with_handlers], plain_actions()).await;
        let emitted = emit_session_shutdown_event(
            &runner,
            ExtensionEvent::SessionShutdown {
                reason: SessionShutdownReason::Quit,
                target_session_file: None,
            },
        )
        .await;
        assert!(emitted, "the handler count drives the gate");
    }

    #[tokio::test]
    async fn emit_project_trust_first_decision_wins_undecided_falls_through() {
        let undecided = extension_with(vec![(
            "project_trust",
            always_handler(HandlerResult::ProjectTrust(ProjectTrustResult::default())),
        )]);
        let deny = extension_with(vec![(
            "project_trust",
            always_handler(HandlerResult::ProjectTrust(ProjectTrustResult {
                trusted: Some(ProjectTrustDecision::No),
            })),
        )]);
        let context = HandlerContext::new(
            "/tmp/project".to_string(),
            ExtensionMode::Print,
            None,
            plain_manager(),
            offline_registry().await,
            always_active(),
            plain_actions(),
        );
        let outcome =
            emit_project_trust_decision(std::slice::from_ref(&deny), context.clone()).await;
        assert_eq!(
            outcome.result.and_then(|resolved| resolved.trusted),
            Some(ProjectTrustDecision::No)
        );
        assert!(outcome.errors.is_empty());

        let undecided_runner =
            emit_project_trust_decision(std::slice::from_ref(&undecided), context).await;
        assert!(undecided_runner.result.is_none());
        assert_eq!(undecided_runner.errors.len(), 0);
    }

    async fn emit_project_trust_decision(
        extensions: &[Arc<Extension>],
        context: HandlerContext,
    ) -> ProjectTrustOutcome {
        emit_project_trust_event(
            extensions,
            ExtensionEvent::ProjectTrust {
                cwd: "/tmp/project".to_string(),
            },
            context,
        )
        .await
    }

    fn always_active() -> AssertActiveFn {
        Arc::new(|| Ok(()))
    }

    /// The bind-core bundle with inert handlers, the base tests re-arm
    /// individual slots over.
    fn trivial_actions() -> ExtensionActions {
        ExtensionActions {
            send_message: Arc::new(
                |_message: CustomMessagePayload, _options: Option<SendMessageOptions>| {},
            ),
            send_user_message: Arc::new(
                |_content: UserContent, _options: Option<SendUserMessageOptions>| {},
            ),
            append_entry: Arc::new(|_custom_type: String, _data: Option<Value>| {}),
            set_session_name: Arc::new(|_name: String| {}),
            get_session_name: Arc::new(|| None),
            set_label: Arc::new(|_entry_id: String, _label: Option<String>| {}),
            get_active_tools: Arc::new(Vec::new),
            get_all_tools: Arc::new(Vec::new),
            set_active_tools: Arc::new(|_tool_names: Vec<String>| {}),
            refresh_tools: Arc::new(|| {}),
            get_commands: Arc::new(Vec::new),
            set_model: Arc::new(|_model: Model| Box::pin(async { false })),
            get_thinking_level: Arc::new(|| ThinkingLevel::Off),
            set_thinking_level: Arc::new(|_level: ThinkingLevel| {}),
        }
    }

    type ProviderRecords = (
        ProviderActions,
        Arc<Mutex<Vec<String>>>,
        Arc<Mutex<Vec<String>>>,
    );

    fn inert_provider_actions() -> ProviderRecords {
        let registered: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let unregistered: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        (
            ProviderActions {
                register_provider: {
                    let registered = Arc::clone(&registered);
                    Arc::new(move |name: String, _config: ProviderConfigInput| {
                        registered
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(name);
                        Ok(())
                    })
                },
                unregister_provider: {
                    let unregistered = Arc::clone(&unregistered);
                    Arc::new(move |name: String| {
                        unregistered
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(name);
                        Ok(())
                    })
                },
            },
            registered,
            unregistered,
        )
    }

    /// An empty registered-flag table, the tests' first-wins aggregate over.
    fn plain_flag(name: &str) -> ExtensionFlag {
        ExtensionFlag {
            name: name.to_string(),
            description: None,
            flag_type: FlagType::String,
            default: None,
            extension_path: "/ext/test.ts".to_string(),
        }
    }

    fn runtime_with_flag(name: &str, value: FlagValue) -> crate::extensions::api::ExtensionRuntime {
        let runtime = crate::extensions::api::ExtensionRuntime::new();
        runtime.set_flag_value(name, value);
        runtime
    }

    /// A runner over the given extensions and runtime, the
    /// non-default-runtime boundary tests read through.
    async fn runner_with_runtime(
        extensions: &[Arc<Extension>],
        runtime: crate::extensions::api::ExtensionRuntime,
    ) -> ExtensionRunner {
        ExtensionRunner::new(
            extensions.to_vec(),
            plain_actions(),
            always_active(),
            "/tmp/project".to_string(),
            ExtensionMode::Print,
            None,
            plain_manager(),
            offline_registry().await,
            runtime,
        )
    }

    #[tokio::test]
    async fn bound_actions_read_after_bind_and_fail_before() {
        let runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
        assert!(
            runner
                .bound_actions()
                .expect_err("the unbound bundle fails")
                .contains("not bound")
        );

        let mut runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
        runner.bind_core(trivial_actions(), None);
        let bundle = runner.bound_actions().expect("the bound bundle reads back");
        (bundle.send_message)(
            CustomMessagePayload {
                custom_type: "note".to_string(),
                content: vec![text_block("blog")],
                display: true,
                details: None,
            },
            Some(SendMessageOptions {
                trigger_turn: Some(true),
                deliver_as: Some(DeliverAs::FollowUp),
            }),
        );
        (bundle.send_user_message)(
            UserContent::Text("ping".to_string()),
            Some(SendUserMessageOptions {
                deliver_as: Some(DeliverAs::Steer),
                expand_prompt_templates: Some(true),
            }),
        );
        (bundle.append_entry)("entries".to_string(), Some(json!({ "kind": 1 })));
        (bundle.set_session_name)("session-name".to_string());
        assert!((bundle.get_session_name)().is_none());
        (bundle.set_label)("entry-1".to_string(), Some("label".to_string()));
        assert!((bundle.get_active_tools)().is_empty());
        assert!((bundle.get_all_tools)().is_empty());
        (bundle.set_active_tools)(vec!["read".to_string()]);
        (bundle.refresh_tools)();
        assert!((bundle.get_commands)().is_empty());
        let refused = (bundle.set_model)(trivial_model()).await;
        assert!(!refused);
        assert_eq!((bundle.get_thinking_level)(), ThinkingLevel::Off);
        (bundle.set_thinking_level)(ThinkingLevel::High);
        assert!(
            runner.active_tools().expect("the bound reader").is_empty(),
            "the bind arms the wrapper's reader too"
        );
    }

    /// The model literal the set-model bind exercises; no registry lookup.
    fn trivial_model() -> Model {
        Model {
            id: "gpt-test".to_string(),
            name: "GPT Test".to_string(),
            api: Api::from("openai"),
            provider: ProviderId::from("openai"),
            base_url: "https://example.invalid".to_string(),
            reasoning: false,
            thinking_level_map: None,
            input: Vec::new(),
            cost: pi_ai::types::ModelCost::default(),
            context_window: 0,
            max_tokens: 0,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    #[tokio::test]
    async fn bind_core_flushes_the_provider_queue() {
        let runtime = crate::extensions::api::ExtensionRuntime::new();
        runtime.register_provider(crate::extensions::api::PendingProviderRegistration {
            name: "queued-provider".to_string(),
            config: ProviderConfigInput::default(),
            extension_path: "/ext/test.ts".to_string(),
        });

        // The actions flush through the caller's provider actions.
        let (provider_actions, registered, _unregistered) = inert_provider_actions();
        let rendered = format!("{provider_actions:?}");
        assert!(rendered.contains("ProviderActions"));
        let mut runner = ExtensionRunner::new(
            Vec::new(),
            plain_actions(),
            always_active(),
            "/tmp/project".to_string(),
            ExtensionMode::Print,
            None,
            plain_manager(),
            offline_registry().await,
            runtime.clone(),
        );
        runner.bind_core(trivial_actions(), Some(provider_actions));
        let armed = runtime
            .bound_provider_actions()
            .expect("the flush arms the immediate path");
        (armed.unregister_provider)("queued-provider".to_string())
            .expect("the armed unregister reads through");
        assert_eq!(
            *registered
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            vec!["queued-provider".to_string()]
        );
        assert!(
            runtime.take_pending_provider_registrations().is_empty(),
            "the queue drains flush-first"
        );

        // Without provider actions the flush lands on the model registry.
        let runtime = crate::extensions::api::ExtensionRuntime::new();
        runtime.register_provider(crate::extensions::api::PendingProviderRegistration {
            name: "registered-provider".to_string(),
            config: ProviderConfigInput::default(),
            extension_path: "/ext/test.ts".to_string(),
        });
        let mut runner = ExtensionRunner::new(
            Vec::new(),
            plain_actions(),
            always_active(),
            "/tmp/project".to_string(),
            ExtensionMode::Print,
            None,
            plain_manager(),
            offline_registry().await,
            runtime.clone(),
        );
        runner.bind_core(trivial_actions(), None);
        assert!(
            runner
                .model_registry
                .get_registered_provider_config("registered-provider")
                .is_some(),
            "the queue flushed into the registry"
        );
    }

    #[tokio::test]
    async fn flush_errors_report_to_the_listeners() {
        let runtime = crate::extensions::api::ExtensionRuntime::new();
        let runtime_for_armed = runtime.clone();
        runtime.register_provider(crate::extensions::api::PendingProviderRegistration {
            name: "failing-provider".to_string(),
            config: ProviderConfigInput::default(),
            extension_path: "/ext/failing.ts".to_string(),
        });
        let mut runner = ExtensionRunner::new(
            Vec::new(),
            plain_actions(),
            always_active(),
            "/tmp/project".to_string(),
            ExtensionMode::Print,
            None,
            plain_manager(),
            offline_registry().await,
            runtime,
        );
        let failed: Arc<Mutex<Vec<ExtensionError>>> = Arc::new(Mutex::new(Vec::new()));
        let _unsubscribe = runner.on_error(Arc::new({
            let failed = Arc::clone(&failed);
            move |error: &ExtensionError| {
                failed
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(error.clone());
            }
        }));
        runner.bind_core(
            trivial_actions(),
            Some(ProviderActions {
                register_provider: Arc::new(|name: String, _config: ProviderConfigInput| {
                    Err(format!("cannot register {name}"))
                }),
                unregister_provider: Arc::new(|_name: String| Ok(())),
            }),
        );
        // The armed actions read through the runtime too.
        let armed = runtime_for_armed.bound_provider_actions().expect("armed");
        (armed.unregister_provider)("failing-provider".to_string()).expect("the armed unregister");
        let failed = failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].extension_path, "/ext/failing.ts");
        assert_eq!(failed[0].event, "register_provider");
        assert_eq!(failed[0].error, "cannot register failing-provider");
    }

    #[tokio::test]
    async fn flags_aggregate_first_registration_wins_and_values_roundtrip() {
        let first = extension_with(Vec::new());
        first.register_flag(plain_flag("alpha"));
        let second = extension_with(Vec::new());
        second.register_flag(plain_flag("alpha"));
        second.register_flag(plain_flag("beta"));
        let runtime = runtime_with_flag("alpha", FlagValue::Bool(true));
        let runner = runner_with_runtime(&[first, second], runtime).await;
        let names = runner
            .get_flags()
            .into_iter()
            .map(|flag| flag.name)
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec!["alpha", "beta"],
            "first per name wins in load order"
        );

        runner.set_flag_value("alpha", FlagValue::Text("late".to_string()));
        assert_eq!(
            runner.get_flag_values().get("alpha"),
            Some(&FlagValue::Text("late".to_string()))
        );
    }

    fn command(name: &str) -> crate::extensions::runtime::ResolvedCommand {
        crate::extensions::runtime::ResolvedCommand {
            name: name.to_string(),
            invocation_name: name.to_string(),
            description: None,
            source_info: source_info(),
            handler: Arc::new(|_args: String, _ctx| Box::pin(async {})),
        }
    }

    #[tokio::test]
    async fn registered_commands_rename_collisions_and_land_free_names() {
        let first = extension_with(Vec::new());
        first.register_command(command("ship"));
        first.register_command(command("alpha"));
        let second = extension_with(Vec::new());
        second.register_command(command("ship"));
        second.register_command(command("ship:1"));
        let runner = runner_with(&[first, second], plain_actions()).await;

        let names = runner
            .get_registered_commands()
            .into_iter()
            .map(|resolved| resolved.invocation_name)
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "ship:1".to_string(),
                "alpha".to_string(),
                "ship:2".to_string(),
                "ship:1:2".to_string(),
            ],
            "collisions take name:occurrence and evicted invocations walk the suffixes"
        );

        let found = runner
            .get_command("ship:2")
            .expect("the second ship resolves by its renamed invocation");
        assert_eq!(found.name, "ship");
        (found.handler)("args".to_string(), runner.context()).await;
        assert!(runner.get_command("ship").is_none());

        // A unique name keeps its bare name and the reads reset diagnostics.
        let single = extension_with(Vec::new());
        single.register_command(command("only"));
        let runner = runner_with(&[single], plain_actions()).await;
        assert_eq!(
            runner.get_command("only").expect("bare name").name,
            "only".to_string()
        );
        assert!(runner.get_command_diagnostics().is_empty());
        let _ = runner.get_registered_commands();
        assert!(
            runner.get_command_diagnostics().is_empty(),
            "the read resets"
        );
    }

    #[tokio::test]
    async fn all_registered_tools_keep_the_first_registration() {
        let renamed = |name: &str| {
            let mut definition = create_bash_tool_definition(".", None);
            definition.name = name.to_string();
            definition.label = name.to_string();
            crate::extensions::runtime::RegisteredTool {
                definition,
                source_info: source_info(),
            }
        };
        let first = extension_with(Vec::new());
        first.register_tool(renamed("shared"));
        first.register_tool(renamed("first-only"));
        let second = extension_with(Vec::new());
        second.register_tool(renamed("shared"));
        let runner = runner_with(&[first, second], plain_actions()).await;

        let names = runner
            .get_all_registered_tools()
            .into_iter()
            .map(|tool| tool.definition.name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["shared".to_string(), "first-only".to_string()]);

        let resolved = runner
            .get_tool_definition("shared")
            .expect("the tool read resolves")
            .expect("the first registration wins");
        assert_eq!(resolved.name, "shared".to_string());
        assert!(
            runner
                .get_tool_definition("missing")
                .expect("the tool read resolves")
                .is_none()
        );
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the bound/default/reset ladder walks every command-context handler"
    )]
    async fn command_context_defaults_and_bound_handlers_route() {
        let runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
        assert_eq!(
            runner.get_extension_paths(),
            vec!["/ext/test.ts".to_string()]
        );

        // The unbound defaults answer the no-op contracts for all six.
        let context = runner.command_context();
        let rendered = format!("{context:?}");
        assert!(rendered.contains("ExtensionCommandContext"));
        assert!(
            !context
                .new_session(NewSessionOptions::default())
                .await
                .expect("the default replacement")
                .cancelled
        );
        context
            .wait_for_idle()
            .await
            .expect("the default idle wait");
        assert!(
            !context
                .fork("entry-0".to_string(), ForkOptions::default())
                .await
                .expect("the default fork")
                .cancelled
        );
        assert!(
            !context
                .navigate_tree("target-0".to_string(), NavigateTreeOptions::default())
                .await
                .expect("the default navigation")
                .cancelled
        );
        assert!(
            !context
                .switch_session("/session.json".to_string(), SwitchSessionOptions::default(),)
                .await
                .expect("the default switch")
                .cancelled
        );
        context.reload().await.expect("the default reload");
        assert_eq!(
            context.get_system_prompt_options().expect("options").cwd,
            "/tmp/project".to_string()
        );
        let rendered = format!("{runner:?}");
        assert!(rendered.contains("ExtensionRunner"));

        // The bound handlers route with their arguments.
        let mut runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
        let command_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let hold_idle = Arc::clone(&command_log);
        let hold_fork = Arc::clone(&command_log);
        let hold_nav = Arc::clone(&command_log);
        let hold_switch = Arc::clone(&command_log);
        let hold_reload = Arc::clone(&command_log);
        let waitlog = Arc::clone(&command_log);
        let fork_entry: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let entry_read = Arc::clone(&fork_entry);
        let position_read: Arc<Mutex<Option<ForkPosition>>> = Arc::new(Mutex::new(None));
        let position_read2 = Arc::clone(&position_read);
        runner.bind_command_context(Some(CommandContextActions {
            wait_for_idle: Arc::new(move || {
                let log = Arc::clone(&waitlog);
                Box::pin(async move {
                    log.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push("idle".to_string());
                })
            }),
            new_session: Arc::new(move |_options: NewSessionOptions| {
                let log = Arc::clone(&hold_idle);
                Box::pin(async move {
                    log.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push("new-session".to_string());
                    SessionReplaceOutcome::proceeded()
                })
            }),
            fork: Arc::new(move |entry_id: String, options: ForkOptions| {
                let log = Arc::clone(&hold_fork);
                *entry_read
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(entry_id);
                *position_read
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = options.position;
                Box::pin(async move {
                    log.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push("fork".to_string());
                    SessionReplaceOutcome { cancelled: true }
                })
            }),
            navigate_tree: Arc::new(move |_target_id: String, _options: NavigateTreeOptions| {
                let log = Arc::clone(&hold_nav);
                Box::pin(async move {
                    log.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push("navigate".to_string());
                    SessionReplaceOutcome::proceeded()
                })
            }),
            switch_session: Arc::new(
                move |_session_path: String, _options: SwitchSessionOptions| {
                    let log = Arc::clone(&hold_switch);
                    Box::pin(async move {
                        log.lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push("switch".to_string());
                        SessionReplaceOutcome::proceeded()
                    })
                },
            ),
            reload: Arc::new(move || {
                let log = Arc::clone(&hold_reload);
                Box::pin(async move {
                    log.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push("reload".to_string());
                })
            }),
        }));

        let context = runner.command_context();
        let outcome = context
            .new_session(NewSessionOptions {
                parent_session: Some("/parent".to_string()),
                ..NewSessionOptions::default()
            })
            .await
            .expect("the bound replacement");
        assert!(!outcome.cancelled);
        context.wait_for_idle().await.expect("the bound idle wait");
        let outcome = context
            .fork(
                "entry-1".to_string(),
                ForkOptions {
                    position: Some(ForkPosition::Before),
                    ..ForkOptions::default()
                },
            )
            .await
            .expect("the bound fork");
        assert!(outcome.cancelled);
        context
            .navigate_tree("target-1".to_string(), NavigateTreeOptions::default())
            .await
            .expect("the bound navigation");
        context
            .switch_session("/session.json".to_string(), SwitchSessionOptions::default())
            .await
            .expect("the bound switch");
        context.reload().await.expect("the bound reload");
        assert_eq!(
            *fork_entry
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            Some("entry-1".to_string())
        );
        assert_eq!(
            *position_read2
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            Some(ForkPosition::Before)
        );
        assert_eq!(
            *command_log
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            vec![
                "new-session".to_string(),
                "idle".to_string(),
                "fork".to_string(),
                "navigate".to_string(),
                "switch".to_string(),
                "reload".to_string()
            ]
        );

        // Unbinding resets to the no-op defaults.
        runner.bind_command_context(None);
        let context = runner.command_context();
        assert!(
            !context
                .new_session(NewSessionOptions::default())
                .await
                .expect("the reset replacement")
                .cancelled
        );
    }

    #[tokio::test]
    async fn invalidate_first_message_wins_and_fails_the_runner_reads() {
        let runtime = crate::extensions::api::ExtensionRuntime::new();
        let runner = runner_with_runtime(&[extension_with(Vec::new())], runtime.clone()).await;
        assert!(runner.active_tools().is_ok());

        runner.invalidate(Some("first replacement".to_string()));
        runner.invalidate(Some("second replacement".to_string()));
        assert_eq!(
            runner
                .active_tools()
                .expect_err("the invalidated runner read fails"),
            "first replacement".to_string()
        );

        // The loader runtime's staleness carries the same message so the
        // extension APIs fail closed too.
        let api = crate::extensions::api::ExtensionApi::new(
            Arc::new(Extension::new(
                "<inline:stale-check>".to_string(),
                "<inline:stale-check>".to_string(),
                false,
                source_info(),
            )),
            runtime,
            "/tmp/project".to_string(),
        );
        assert!(
            api.assert_active()
                .expect_err("the stale runtime fails the API reads")
                .contains("first replacement")
        );

        // The none-call restates upstream's guidance sentence.
        let runtime = crate::extensions::api::ExtensionRuntime::new();
        let runner = runner_with_runtime(&[extension_with(Vec::new())], runtime.clone()).await;
        runner.invalidate(None);
        let message = runner
            .active_tools()
            .expect_err("the invalidated runner read fails");
        assert!(
            message.contains("session replacement or reload"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn rebinds_land_on_the_contexts_and_the_bound_shutdown() {
        let shut: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
        let hold = Arc::clone(&shut);
        let actions = Arc::new(ContextActions {
            shutdown: Some(Arc::new(move || {
                *hold
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
            })),
            ..ContextActions::default()
        });
        let runner = runner_with(&[extension_with(Vec::new())], actions).await;
        runner.shutdown();
        assert!(
            *shut
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            "shutdown runs the bound actions"
        );
    }

    #[tokio::test]
    async fn session_rebindings_land_on_the_next_context() {
        let mut runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
        runner.set_cwd("/elsewhere".to_string());
        runner.set_mode(ExtensionMode::Rpc);
        let context = runner.command_context();
        assert_eq!(
            context.cwd().expect("the rebound cwd"),
            "/elsewhere".to_string()
        );
        assert_eq!(
            context.mode().expect("the rebound mode"),
            ExtensionMode::Rpc
        );
    }
}

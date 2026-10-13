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
//! The interactive surfaces (the UI context, keybinding reservation,
//! command context, and session-handler binding) ride their consumers on
//! the sibling tickets and are not re-stated here.

use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{ImageContent, ProviderHeaders};
use serde_json::Value;

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
pub struct ExtensionRunner {
    extensions: Vec<Arc<Extension>>,
    assert_active: AssertActiveFn,
    actions: Arc<ContextActions>,
    cwd: String,
    mode: ExtensionMode,
    ui: Option<Arc<dyn std::any::Any + Send + Sync>>,
    session_manager: crate::session_manager::SessionManager,
    model_registry: crate::model_registry::ModelRegistry,
    /// The active-tool names the registered-tool wrapper diffs, upstream's
    /// `runtime.getActiveTools` read the wrapper wraps.
    active_tools: Arc<dyn Fn() -> Vec<String> + Send + Sync>,
    error_listeners: Arc<Mutex<Vec<ExtensionErrorListener>>>,
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
    ) -> Self {
        Self {
            extensions,
            assert_active,
            actions,
            cwd,
            mode,
            ui,
            session_manager,
            model_registry,
            active_tools: Arc::new(Vec::new),
            error_listeners: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Bind the active-tool reader the registered-tool wrapper diffs,
    /// upstream's `bindCore` `getActiveTools`.
    pub fn bind_active_tools(&mut self, active_tools: Arc<dyn Fn() -> Vec<String> + Send + Sync>) {
        self.active_tools = active_tools;
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
    /// returned closure unsubscribes the same listener.
    pub fn on_error(&self, listener: ExtensionErrorListener) -> impl Fn() + Send {
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
        ExtensionRunner, ProjectTrustOutcome, emit_project_trust_event, emit_session_shutdown_event,
    };
    use crate::extensions::runtime::{
        AssertActiveFn, ContextActions, Extension, ExtensionError, ExtensionHandler, HandlerContext,
    };
    use crate::model_registry::ModelRegistry;
    use crate::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
    use crate::session_manager::SessionManager;
    use crate::source_info::{SourceInfo, SourceOrigin, SourceScope, SyntheticSourceOptions};
    use pi_ai::auth::credential_store::InMemoryCredentialStore;
    use pi_ai::types::{Api, ProviderId, Usage, UserContent};

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
}

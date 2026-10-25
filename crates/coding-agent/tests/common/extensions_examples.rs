//! Shared doubles for the example-extension suites (#179), the restatement
//! of upstream's `vi.fn` API/context mocks at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Each suite loads its example as an inline factory through the
//! [`super::extensions`] stand-in, then binds an [`ActionLog`] action bundle
//! onto the runtime so the example's `pi.sendMessage` family reaches captured
//! vectors, and passes a [`TestUi`] the body downcasts from `ctx.ui()`. The
//! production UI surface rides the interactive ticket, so the downcast is
//! the suite's stand-in for upstream's mock `ctx.ui`.
//!
//! The untested parts of upstream's mock bundle (session replacement, model
//! selection, thinking level) answer their no-op defaults: no example these
//! suites restate reaches them.

use std::any::Any;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use pi_ai::types::{BoxedFuture, UserContent};
use pi_coding_agent::extensions::api::{ExtensionFactory, ExtensionRuntime};
use pi_coding_agent::extensions::reasons::{CustomMessagePayload, ExtensionMode};
use pi_coding_agent::extensions::runner::{
    ExtensionActions, ExtensionRunner, SendMessageOptions, SendUserMessageOptions,
};
use pi_coding_agent::extensions::runtime::{ContextActions, ResolvedCommand};
use pi_coding_agent::model_registry::ModelRegistry;
use pi_coding_agent::session_manager::SessionManager;
use serde_json::Value;

use super::extensions::{TestExtensionInput, create_test_extensions_result};

/// One captured `pi.sendMessage` call.
pub struct CapturedMessage {
    /// The custom-message payload.
    pub message: CustomMessagePayload,
    /// The delivery options.
    pub options: Option<SendMessageOptions>,
}

/// One captured `pi.sendUserMessage` call.
pub struct CapturedUserMessage {
    /// The message content.
    pub content: UserContent,
    /// The delivery options.
    pub options: Option<SendUserMessageOptions>,
}

/// The action bundle's captured writes and the active-tool state.
#[derive(Default)]
pub struct ActionLog {
    /// The `pi.sendMessage` calls in order.
    pub messages: Mutex<Vec<CapturedMessage>>,
    /// The `pi.sendUserMessage` calls in order.
    pub user_messages: Mutex<Vec<CapturedUserMessage>>,
    /// The `pi.appendEntry` calls in order.
    pub entries: Mutex<Vec<(String, Option<Value>)>>,
    /// The active tool names, mutated by `pi.setActiveTools`.
    pub active_tools: Mutex<Vec<String>>,
}

impl ActionLog {
    /// Seed the active tool names the example's `pi.getActiveTools` reads.
    pub fn set_active_tools(&self, names: &[&str]) {
        *self
            .active_tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            names.iter().map(|name| (*name).to_string()).collect();
    }

    /// The current active tool names.
    pub fn active_tools(&self) -> Vec<String> {
        self.active_tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// The bundle the runtime binds.
    pub fn actions(self: &Arc<Self>) -> ExtensionActions {
        let log = Arc::clone(self);
        ExtensionActions {
            send_message: Arc::new(move |message, options| {
                log.messages
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(CapturedMessage { message, options });
            }),
            send_user_message: Arc::new({
                let log = Arc::clone(self);
                move |content, options| {
                    log.user_messages
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(CapturedUserMessage { content, options });
                }
            }),
            append_entry: Arc::new({
                let log = Arc::clone(self);
                move |custom_type, data| {
                    log.entries
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push((custom_type, data));
                }
            }),
            set_session_name: Arc::new(|_| {}),
            get_session_name: Arc::new(|| None),
            set_label: Arc::new(|_, _| {}),
            get_active_tools: Arc::new({
                let log = Arc::clone(self);
                move || {
                    log.active_tools
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone()
                }
            }),
            get_all_tools: Arc::new(Vec::new),
            set_active_tools: Arc::new({
                let log = Arc::clone(self);
                move |names| {
                    *log.active_tools
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = names;
                }
            }),
            refresh_tools: Arc::new(|| {}),
            get_commands: Arc::new(Vec::new),
            set_model: Arc::new(|_model| {
                let future: BoxedFuture<'static, bool> = Box::pin(async { true });
                future
            }),
            get_thinking_level: Arc::new(|| pi_agent_core::types::ThinkingLevel::Off),
            set_thinking_level: Arc::new(|_| {}),
        }
    }
}

/// The interactive surface double the example bodies downcast from
/// `ctx.ui()`. The production type rides the interactive ticket.
#[derive(Default)]
pub struct TestUi {
    /// The scripted `ui.select` answers, consumed front to back; an empty
    /// queue answers `None` (upstream's dismissed picker).
    pub selections: Mutex<VecDeque<Option<String>>>,
    /// How many `ui.select` prompts ran.
    pub select_calls: Mutex<usize>,
    /// The `ui.editor` answer; `None` reads as dismissed.
    pub editor_answer: Mutex<Option<String>>,
    /// The `ui.notify` calls as `(message, level)`.
    pub notifications: Mutex<Vec<(String, Option<String>)>>,
    /// The `ui.setStatus` calls as `(key, text)`.
    pub statuses: Mutex<Vec<(String, Option<String>)>>,
    /// The `ui.setWidget` calls as `(key, lines)`.
    pub widgets: Mutex<Vec<(String, Option<Vec<String>>)>>,
}

impl TestUi {
    /// Script one `ui.select` answer.
    pub fn select_with(&self, choice: Option<&str>) {
        self.selections
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(choice.map(str::to_string));
    }

    /// Script the `ui.editor` answer.
    pub fn edit_with(&self, text: &str) {
        *self
            .editor_answer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(text.to_string());
    }

    /// Upstream's `ctx.ui.notify(message, level?)`.
    pub fn notify(&self, message: &str, level: Option<&str>) {
        self.notifications
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((message.to_string(), level.map(str::to_string)));
    }

    /// Upstream's `ctx.ui.select(title, choices)`.
    pub fn select(&self, _title: &str, _choices: &[&str]) -> Option<String> {
        *self
            .select_calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
        self.selections
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .flatten()
    }

    /// How many `ui.select` prompts ran.
    pub fn select_calls(&self) -> usize {
        *self
            .select_calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Upstream's `ctx.ui.editor(title, prefill)`.
    pub fn editor(&self, _title: &str, _prefill: &str) -> Option<String> {
        self.editor_answer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Upstream's `ctx.ui.setStatus(key, text)`.
    pub fn set_status(&self, key: &str, text: Option<&str>) {
        self.statuses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((key.to_string(), text.map(str::to_string)));
    }

    /// Upstream's `ctx.ui.setWidget(key, lines)`.
    pub fn set_widget(&self, key: &str, lines: Option<Vec<String>>) {
        self.widgets
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((key.to_string(), lines));
    }

    /// Upstream's `ctx.ui.theme.fg(color, text)`: the tests carry no color
    /// escape codes, so the text passes through.
    #[expect(
        clippy::unused_self,
        reason = "the theme hooks mirror upstream's UI shape and carry no state"
    )]
    pub fn fg(&self, _color: &str, text: &str) -> String {
        text.to_string()
    }

    /// Upstream's `ctx.ui.theme.strikethrough(text)`.
    #[expect(
        clippy::unused_self,
        reason = "the theme hooks mirror upstream's UI shape and carry no state"
    )]
    pub fn strikethrough(&self, text: &str) -> String {
        text.to_string()
    }
}

/// The downcast the example bodies run: the UI the runner carries.
pub fn test_ui(ui: &Arc<dyn Any + Send + Sync>) -> Arc<TestUi> {
    Arc::clone(ui)
        .downcast::<TestUi>()
        .expect("the suite binds its TestUi double")
}

/// One loaded example extension over a runner with the action bundle bound.
pub struct ExampleHarness {
    /// The runner the suites emit through.
    pub runner: ExtensionRunner,
    /// The UI double.
    pub ui: Arc<TestUi>,
    /// The action log.
    pub log: Arc<ActionLog>,
}

/// Load `factory` as a named inline extension and build its runner.
///
/// `context_actions` seeds the `ctx` accessors; `model_registry` backs
/// `ctx.modelRegistry`.
pub async fn harness(
    name: &str,
    cwd: &str,
    factory: ExtensionFactory,
    ui: Arc<TestUi>,
    log: Arc<ActionLog>,
    context_actions: ContextActions,
    model_registry: ModelRegistry,
) -> ExampleHarness {
    let result = create_test_extensions_result(
        vec![TestExtensionInput::Named {
            name: name.to_string(),
            factory,
        }],
        cwd,
    )
    .await;
    let runtime: ExtensionRuntime = result.runtime;
    let actions = log.actions();
    runtime.bind_actions(Arc::new(actions.clone()));
    let extensions = result
        .extensions
        .into_iter()
        .filter_map(|entry| entry.record)
        .collect();
    let manager = SessionManager::in_memory(Some(cwd), None, None).expect("the session manager");
    let ui_arc: Arc<TestUi> = Arc::clone(&ui);
    let ui_double: Arc<dyn Any + Send + Sync> = ui_arc;
    let mut runner = ExtensionRunner::new(
        extensions,
        Arc::new(context_actions),
        Arc::new(|| Ok(())),
        cwd.to_string(),
        ExtensionMode::Print,
        Some(ui_double),
        manager,
        model_registry,
        runtime,
    );
    runner.bind_core(actions, None);
    ExampleHarness { runner, ui, log }
}

/// One registered command by its registered name.
pub fn command(runner: &ExtensionRunner, name: &str) -> ResolvedCommand {
    runner
        .get_registered_commands()
        .into_iter()
        .find(|command| command.name == name)
        .expect("the example registers the command")
}

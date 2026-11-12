//! The extension UI context and its dialogs, upstream's
//! `createExtensionUIContext`, `showExtensionSelector`, `showExtensionInput`,
//! `showExtensionEditor`, `showExtensionCustom`, and the project-trust
//! context.
//!
//! Restatements:
//!
//! - The out-of-process [`UiCallHandler`] runs on the extension host's task
//!   while every dialog is a thread-bound pi-tui component. Each call becomes
//!   a [`UiRequest`] on an unbounded channel; the frame presents the dialog
//!   and the call's future resolves through a oneshot. Upstream's context
//!   called the TUI directly because extensions shared its event loop.
//! - The factory-valued methods (`custom`, a component-valued `setWidget`,
//!   `setFooter`, `setHeader`, `setEditorComponent`,
//!   `addAutocompleteProvider`, `onTerminalInput`) cannot cross the wire, so
//!   the handler answers `unsupported`. The in-process frame path still
//!   implements them for the runner's own use.
//! - `getTheme` returns the theme's name because a theme instance has no wire
//!   shape; the in-process setter keeps the full instance.
//! - The context holds only the sender, so it is `Send + Sync` and bindable
//!   as `ctx.ui`; it is also the concrete type the extension runtime
//!   downcasts.

use std::rc::Rc;
use std::sync::Arc;

use pi_ai::types::BoxedFuture;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::extensions::host::{HostCallError, UiCallHandler};

use super::UiRequest;
use super::UiWidgetPlacement;
use super::{EditorFactory, InteractiveMode, ThemeUpdateResult, UiNotifyKind};
use crate::modes::interactive::components::extension_selector::ExtensionSelectorComponent;
use crate::modes::interactive::components::login_dialog::LoginDialogComponent;

/// The interactive extension UI context, upstream's `createExtensionUIContext`
/// object.
pub(crate) struct InteractiveUiContext {
    sender: mpsc::UnboundedSender<UiRequest>,
}

impl std::fmt::Debug for InteractiveUiContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InteractiveUiContext")
            .finish_non_exhaustive()
    }
}

impl InteractiveUiContext {
    /// Build the context over the frame's request channel.
    pub(crate) fn new(sender: mpsc::UnboundedSender<UiRequest>) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self { sender })
    }

    fn send(&self, request: UiRequest) {
        let _ = self.sender.send(request);
    }

    /// Show a selector and wait for its reply.
    pub(crate) fn select(
        &self,
        title: String,
        options: Vec<String>,
        signal: Option<pi_agent_core::harness::context::AbortSignal>,
        timeout: Option<u64>,
    ) -> BoxedFuture<'static, Option<String>> {
        let (reply, receiver) = tokio::sync::oneshot::channel();
        self.send(UiRequest::Select {
            title,
            options,
            signal,
            timeout,
            reply,
        });
        Box::pin(async move { receiver.await.ok().flatten() })
    }

    /// Show a confirmation and wait for its reply.
    pub(crate) fn confirm(
        &self,
        title: String,
        message: String,
        signal: Option<pi_agent_core::harness::context::AbortSignal>,
        timeout: Option<u64>,
    ) -> BoxedFuture<'static, bool> {
        let (reply, receiver) = tokio::sync::oneshot::channel();
        self.send(UiRequest::Confirm {
            title,
            message,
            signal,
            timeout,
            reply,
        });
        Box::pin(async move { receiver.await.unwrap_or(false) })
    }

    /// Show a text input and wait for its reply.
    pub(crate) fn input(
        &self,
        title: String,
        placeholder: Option<String>,
        signal: Option<pi_agent_core::harness::context::AbortSignal>,
        timeout: Option<u64>,
    ) -> BoxedFuture<'static, Option<String>> {
        let (reply, receiver) = tokio::sync::oneshot::channel();
        self.send(UiRequest::Input {
            title,
            placeholder,
            signal,
            timeout,
            reply,
        });
        Box::pin(async move { receiver.await.ok().flatten() })
    }

    /// Show the multi-line editor and wait for its reply.
    pub(crate) fn editor(
        &self,
        title: String,
        prefill: Option<String>,
    ) -> BoxedFuture<'static, Option<String>> {
        let (reply, receiver) = tokio::sync::oneshot::channel();
        self.send(UiRequest::Editor {
            title,
            prefill,
            reply,
        });
        Box::pin(async move { receiver.await.ok().flatten() })
    }

    /// Emit a fire-and-forget notice.
    pub(crate) fn notify(&self, message: &str, kind: Option<UiNotifyKind>) {
        self.send(UiRequest::Notify {
            message: message.to_owned(),
            kind,
        });
    }

    /// Set or clear a footer status entry.
    pub(crate) fn set_status(&self, key: &str, text: Option<String>) {
        self.send(UiRequest::SetStatus {
            key: key.to_owned(),
            text,
        });
    }

    /// Set the working message.
    pub(crate) fn set_working_message(&self, message: Option<String>) {
        self.send(UiRequest::SetWorkingMessage(message));
    }

    /// Show or hide the working indicator.
    pub(crate) fn set_working_visible(&self, visible: bool) {
        self.send(UiRequest::SetWorkingVisible(visible));
    }

    /// Set the hidden-thinking label.
    pub(crate) fn set_hidden_thinking_label(&self, label: Option<String>) {
        self.send(UiRequest::SetHiddenThinkingLabel(label));
    }

    /// Set or clear an extension widget.
    pub(crate) fn set_widget(
        &self,
        key: &str,
        lines: Option<Vec<String>>,
        placement: UiWidgetPlacement,
    ) {
        self.send(UiRequest::SetWidget {
            key: key.to_owned(),
            lines,
            placement,
        });
    }

    /// Set the terminal title.
    pub(crate) fn set_title(&self, title: &str) {
        self.send(UiRequest::SetTitle(title.to_owned()));
    }

    /// Paste text into the editor.
    pub(crate) fn paste_to_editor(&self, text: &str) {
        self.send(UiRequest::PasteToEditor(text.to_owned()));
    }

    /// Replace the editor's text.
    pub(crate) fn set_editor_text(&self, text: &str) {
        self.send(UiRequest::SetEditorText(text.to_owned()));
    }

    /// Read the editor's text.
    pub(crate) fn get_editor_text(&self) -> BoxedFuture<'static, String> {
        let (reply, receiver) = tokio::sync::oneshot::channel();
        self.send(UiRequest::GetEditorText(reply));
        Box::pin(async move { receiver.await.unwrap_or_default() })
    }

    /// Whether tool output is expanded.
    pub(crate) fn get_tools_expanded(&self) -> BoxedFuture<'static, bool> {
        let (reply, receiver) = tokio::sync::oneshot::channel();
        self.send(UiRequest::GetToolsExpanded(reply));
        Box::pin(async move { receiver.await.unwrap_or(false) })
    }

    /// Set tool output expansion.
    pub(crate) fn set_tools_expanded(&self, expanded: bool) {
        self.send(UiRequest::SetToolsExpanded(expanded));
    }

    /// Marshal one login prompt onto the frame.
    pub(crate) fn send_auth_prompt(
        &self,
        prompt: pi_ai::auth::types::AuthPrompt,
        reply: tokio::sync::oneshot::Sender<Result<String, String>>,
    ) {
        self.send(UiRequest::AuthPrompt { prompt, reply });
    }

    /// Marshal one login event onto the frame.
    pub(crate) fn send_auth_notify(&self, event: pi_ai::auth::types::AuthEvent) {
        self.send(UiRequest::AuthNotify(event));
    }
}

/// Read a required string parameter by name, falling back to position.
fn param_string(
    params: Option<&Value>,
    field: &str,
    index: usize,
) -> Result<String, HostCallError> {
    params
        .and_then(|params| {
            params
                .get(field)
                .cloned()
                .or_else(|| params.get(index).cloned())
        })
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| HostCallError::new("invalid_params", format!("missing `{field}`")))
}

fn param_timeout(params: Option<&Value>) -> Option<u64> {
    params
        .and_then(|params| params.get("timeout"))
        .and_then(Value::as_u64)
}

impl UiCallHandler for InteractiveUiContext {
    fn call(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> BoxedFuture<'static, Result<Value, HostCallError>> {
        match method {
            "select" => {
                let title = param_string(params.as_ref(), "title", 0);
                let options: Vec<String> = params
                    .as_ref()
                    .and_then(|params| params.get("options"))
                    .and_then(Value::as_array)
                    .map(|options| {
                        options
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect()
                    })
                    .unwrap_or_default();
                let timeout = param_timeout(params.as_ref());
                match title {
                    Ok(title) => {
                        let future = self.select(title, options, None, timeout);
                        Box::pin(async move { Ok(future.await.map_or(Value::Null, Value::from)) })
                    }
                    Err(error) => Box::pin(async move { Err(error) }),
                }
            }
            "confirm" => {
                let title = param_string(params.as_ref(), "title", 0);
                let message = param_string(params.as_ref(), "message", 1);
                let timeout = param_timeout(params.as_ref());
                match (title, message) {
                    (Ok(title), Ok(message)) => {
                        let future = self.confirm(title, message, None, timeout);
                        Box::pin(async move { Ok(Value::from(future.await)) })
                    }
                    (Err(error), _) | (_, Err(error)) => Box::pin(async move { Err(error) }),
                }
            }
            "input" => {
                let title = param_string(params.as_ref(), "title", 0);
                let placeholder = params
                    .as_ref()
                    .and_then(|params| params.get("placeholder"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let timeout = param_timeout(params.as_ref());
                match title {
                    Ok(title) => {
                        let future = self.input(title, placeholder, None, timeout);
                        Box::pin(async move { Ok(future.await.map_or(Value::Null, Value::from)) })
                    }
                    Err(error) => Box::pin(async move { Err(error) }),
                }
            }
            "editor" => {
                let title = param_string(params.as_ref(), "title", 0);
                let prefill = params
                    .as_ref()
                    .and_then(|params| params.get("prefill"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                match title {
                    Ok(title) => {
                        let future = self.editor(title, prefill);
                        Box::pin(async move { Ok(future.await.map_or(Value::Null, Value::from)) })
                    }
                    Err(error) => Box::pin(async move { Err(error) }),
                }
            }
            other => self.call_value(other, params),
        }
    }
}

impl InteractiveUiContext {
    /// The non-dialog wire methods, and the factory refusals.
    fn call_value(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> BoxedFuture<'static, Result<Value, HostCallError>> {
        match method {
            "notify" => {
                let message = param_string(params.as_ref(), "message", 0);
                let kind = params
                    .as_ref()
                    .and_then(|params| params.get("type"))
                    .and_then(Value::as_str)
                    .map(|kind| match kind {
                        "warning" => UiNotifyKind::Warning,
                        "error" => UiNotifyKind::Error,
                        _ => UiNotifyKind::Info,
                    });
                match message {
                    Ok(message) => {
                        self.notify(&message, kind);
                        Box::pin(async { Ok(Value::Null) })
                    }
                    Err(error) => Box::pin(async move { Err(error) }),
                }
            }
            "setStatus" => {
                let key = param_string(params.as_ref(), "key", 0);
                let text = params
                    .as_ref()
                    .and_then(|params| params.get("text"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                match key {
                    Ok(key) => {
                        self.set_status(&key, text);
                        Box::pin(async { Ok(Value::Null) })
                    }
                    Err(error) => Box::pin(async move { Err(error) }),
                }
            }
            "setTitle" => {
                let title = param_string(params.as_ref(), "title", 0);
                match title {
                    Ok(title) => {
                        self.set_title(&title);
                        Box::pin(async { Ok(Value::Null) })
                    }
                    Err(error) => Box::pin(async move { Err(error) }),
                }
            }
            "pasteToEditor" => {
                let text = param_string(params.as_ref(), "text", 0);
                match text {
                    Ok(text) => {
                        self.paste_to_editor(&text);
                        Box::pin(async { Ok(Value::Null) })
                    }
                    Err(error) => Box::pin(async move { Err(error) }),
                }
            }
            "setEditorText" => {
                let text = param_string(params.as_ref(), "text", 0);
                match text {
                    Ok(text) => {
                        self.set_editor_text(&text);
                        Box::pin(async { Ok(Value::Null) })
                    }
                    Err(error) => Box::pin(async move { Err(error) }),
                }
            }
            "getEditorText" => {
                let future = self.get_editor_text();
                Box::pin(async move { Ok(Value::from(future.await)) })
            }
            "getToolsExpanded" => {
                let future = self.get_tools_expanded();
                Box::pin(async move { Ok(Value::from(future.await)) })
            }
            "setToolsExpanded" => {
                let expanded = params
                    .as_ref()
                    .and_then(|params| params.get("expanded"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                self.set_tools_expanded(expanded);
                Box::pin(async { Ok(Value::Null) })
            }
            "setWorkingMessage" => {
                let message = params
                    .as_ref()
                    .and_then(|params| params.get("message"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                self.set_working_message(message);
                Box::pin(async { Ok(Value::Null) })
            }
            "setWorkingVisible" => {
                let visible = params
                    .as_ref()
                    .and_then(|params| params.get("visible"))
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                self.set_working_visible(visible);
                Box::pin(async { Ok(Value::Null) })
            }
            "setHiddenThinkingLabel" => {
                let label = params
                    .as_ref()
                    .and_then(|params| params.get("label"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                self.set_hidden_thinking_label(label);
                Box::pin(async { Ok(Value::Null) })
            }
            "setWidget" => {
                let key = param_string(params.as_ref(), "key", 0);
                let lines = params
                    .as_ref()
                    .and_then(|params| params.get("content"))
                    .and_then(Value::as_array)
                    .map(|lines| {
                        lines
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    });
                let placement = match params
                    .as_ref()
                    .and_then(|params| params.get("placement"))
                    .and_then(Value::as_str)
                {
                    Some("belowEditor") => UiWidgetPlacement::BelowEditor,
                    _ => UiWidgetPlacement::AboveEditor,
                };
                match key {
                    Ok(key) => {
                        self.set_widget(&key, lines, placement);
                        Box::pin(async { Ok(Value::Null) })
                    }
                    Err(error) => Box::pin(async move { Err(error) }),
                }
            }
            "onTerminalInput"
            | "custom"
            | "setFooter"
            | "setHeader"
            | "setEditorComponent"
            | "getEditorComponent"
            | "addAutocompleteProvider" => Box::pin(async {
                Err(HostCallError::unsupported(
                    "the interactive wire surface cannot carry a component factory",
                ))
            }),
            other => {
                let error = HostCallError::unknown_method(other);
                Box::pin(async move { Err(error) })
            }
        }
    }
}

impl<S: crate::settings_manager::SettingsStorageKind> InteractiveMode<S> {
    fn show_extension_selector(
        &self,
        title: &str,
        options: Vec<String>,
        timeout: Option<u64>,
        complete: Rc<dyn Fn(Option<String>)>,
    ) {
        let request_render: Rc<dyn Fn()> = {
            let tui = Rc::clone(&self.tui);
            Rc::new(move || tui.request_render(false))
        };
        let done = Rc::new(std::cell::Cell::new(false));
        let select_complete = Rc::clone(&complete);
        let cancel_complete = Rc::clone(&complete);
        let selector = Rc::new(
            crate::modes::interactive::components::extension_selector::ExtensionSelectorComponent::new(
                title,
                options,
                Rc::new({
                    let done = Rc::clone(&done);
                    move |value: &str| {
                        if done.replace(true) {
                            return;
                        }
                        select_complete(Some(value.to_owned()));
                    }
                }),
                Rc::new({
                    let done = Rc::clone(&done);
                    move || {
                        if done.replace(true) {
                            return;
                        }
                        cancel_complete(None);
                    }
                }),
                crate::modes::interactive::components::extension_selector::ExtensionSelectorOptions {
                    request_render: Some(request_render),
                    timeout_ms: timeout,
                    on_toggle_tools_expanded: None,
                },
            ),
        );
        *self.extension_selector.borrow_mut() = Some(Rc::clone(&selector));
        self.show_dialog_component(selector);
    }

    fn show_extension_input(
        &self,
        title: &str,
        placeholder: Option<&str>,
        timeout: Option<u64>,
        complete: Rc<dyn Fn(Option<String>)>,
    ) {
        let request_render: Rc<dyn Fn()> = {
            let tui = Rc::clone(&self.tui);
            Rc::new(move || tui.request_render(false))
        };
        let done = Rc::new(std::cell::Cell::new(false));
        let submit_complete = Rc::clone(&complete);
        let cancel_complete = Rc::clone(&complete);
        let input = Rc::new(
            crate::modes::interactive::components::extension_input::ExtensionInputComponent::new(
                title,
                placeholder,
                Rc::new({
                    let done = Rc::clone(&done);
                    move |value: String| {
                        if done.replace(true) {
                            return;
                        }
                        submit_complete(Some(value));
                    }
                }),
                Rc::new({
                    let done = Rc::clone(&done);
                    move || {
                        if done.replace(true) {
                            return;
                        }
                        cancel_complete(None);
                    }
                }),
                crate::modes::interactive::components::extension_input::ExtensionInputOptions {
                    request_render: Some(request_render),
                    timeout_ms: timeout,
                },
            ),
        );
        *self.extension_input.borrow_mut() = Some(Rc::clone(&input));
        self.show_dialog_component(input);
    }

    fn show_extension_editor(
        &self,
        title: &str,
        prefill: Option<&str>,
        complete: Rc<dyn Fn(Option<String>)>,
    ) {
        let external = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_external_editor_command();
        let submit_complete = Rc::clone(&complete);
        let cancel_complete = Rc::clone(&complete);
        let editor = Rc::new(
            crate::modes::interactive::components::extension_editor::ExtensionEditorComponent::new(
                Rc::clone(&self.tui),
                Rc::clone(&self.keybindings),
                title,
                prefill,
                Rc::new(move |value: &str| submit_complete(Some(value.to_owned()))),
                Rc::new(move || cancel_complete(None)),
                pi_tui::components::editor::EditorOptions::default(),
                Some(&external),
            ),
        );
        *self.extension_editor.borrow_mut() = Some(Rc::clone(&editor));
        self.show_dialog_component(editor);
    }

    fn show_dialog_component(&self, component: Rc<dyn pi_tui::tui::Component>) {
        self.dispose_active_selector();
        self.editor_container.clear();
        self.editor_container.add_child(Rc::clone(&component));
        self.tui.set_focus(Some(component));
        self.tui.request_render(false);
    }

    /// Hide the extension selector.
    pub(crate) fn hide_extension_selector(&self) {
        if let Some(selector) = self.extension_selector.borrow_mut().take() {
            self.editor_container.clear();
            selector.dispose();
        }
        self.restore_editor();
    }

    /// Hide the extension input.
    pub(crate) fn hide_extension_input(&self) {
        if let Some(input) = self.extension_input.borrow_mut().take() {
            self.editor_container.clear();
            input.dispose();
        }
        self.restore_editor();
    }

    /// Hide the extension editor.
    pub(crate) fn hide_extension_editor(&self) {
        if self.extension_editor.borrow_mut().take().is_some() {
            self.editor_container.clear();
        }
        self.restore_editor();
    }

    pub(crate) fn restore_editor(&self) {
        if self.extension_selector.borrow().is_some()
            || self.extension_input.borrow().is_some()
            || self.extension_editor.borrow().is_some()
        {
            return;
        }
        self.editor_container.clear();
        let editor: Rc<dyn pi_tui::tui::Component> = self.editor();
        self.editor_container.add_child(Rc::clone(&editor));
        self.tui.set_focus(Some(editor));
        self.tui.request_render(false);
    }
}

impl<S: crate::settings_manager::SettingsStorageKind> InteractiveMode<S> {
    pub(crate) fn present_ui_request(&self, request: UiRequest) {
        match request {
            UiRequest::Select {
                title,
                options,
                signal,
                timeout,
                reply,
            } => {
                if signal
                    .as_ref()
                    .is_some_and(pi_agent_core::harness::context::AbortSignal::aborted)
                {
                    let _ = reply.send(None);
                    return;
                }
                let reply = Rc::new(std::cell::RefCell::new(Some(reply)));
                self.show_extension_selector(
                    &title,
                    options,
                    timeout,
                    Rc::new(move |value| {
                        if let Some(reply) = reply.borrow_mut().take() {
                            let _ = reply.send(value);
                        }
                    }),
                );
            }
            UiRequest::Confirm {
                title,
                message,
                signal,
                timeout,
                reply,
            } => {
                if signal
                    .as_ref()
                    .is_some_and(pi_agent_core::harness::context::AbortSignal::aborted)
                {
                    let _ = reply.send(false);
                    return;
                }
                let reply = std::cell::Cell::new(Some(reply));
                let title = format!("{title}\n{message}");
                self.show_extension_selector(
                    &title,
                    vec!["Yes".to_owned(), "No".to_owned()],
                    timeout,
                    Rc::new(move |value| {
                        if let Some(reply) = reply.take() {
                            let _ = reply.send(value.as_deref() == Some("Yes"));
                        }
                    }),
                );
            }
            UiRequest::Input {
                title,
                placeholder,
                signal,
                timeout,
                reply,
            } => {
                if signal
                    .as_ref()
                    .is_some_and(pi_agent_core::harness::context::AbortSignal::aborted)
                {
                    let _ = reply.send(None);
                    return;
                }
                let reply = Rc::new(std::cell::RefCell::new(Some(reply)));
                self.show_extension_input(
                    &title,
                    placeholder.as_deref(),
                    timeout,
                    Rc::new(move |value| {
                        if let Some(reply) = reply.borrow_mut().take() {
                            let _ = reply.send(value);
                        }
                    }),
                );
            }
            UiRequest::Editor {
                title,
                prefill,
                reply,
            } => {
                let reply = Rc::new(std::cell::RefCell::new(Some(reply)));
                self.show_extension_editor(
                    &title,
                    prefill.as_deref(),
                    Rc::new(move |value| {
                        if let Some(reply) = reply.borrow_mut().take() {
                            let _ = reply.send(value);
                        }
                    }),
                );
            }
            UiRequest::Notify { message, kind } => match kind {
                Some(super::UiNotifyKind::Error) => self.show_error(&message),
                Some(super::UiNotifyKind::Warning) => self.show_warning(&message),
                _ => self.show_status(&message),
            },
            UiRequest::SetStatus { key, text } => {
                self.footer_data_provider
                    .set_extension_status(&key, text.as_deref());
                self.tui.request_render(false);
            }
            UiRequest::SetWorkingMessage(message) => {
                *self.working_message.borrow_mut() = message.clone();
                if let Some(indicator) = self.active_status_indicator.borrow().as_ref()
                    && indicator.kind() == crate::modes::interactive::components::status_indicator::StatusIndicatorKind::Working
                {
                    indicator.set_message(
                        message.unwrap_or_else(|| Self::DEFAULT_WORKING_MESSAGE.to_owned()),
                    );
                }
            }
            UiRequest::SetWorkingVisible(visible) => self.set_working_visible(visible),
            UiRequest::SetWorkingIndicator(options) => self.set_working_indicator(options),
            UiRequest::SetHiddenThinkingLabel(label) => {
                self.set_hidden_thinking_label(label.as_deref());
            }
            UiRequest::SetWidget {
                key,
                lines,
                placement,
            } => self.set_extension_widget(&key, lines, placement),
            UiRequest::SetTitle(title) => {
                self.tui.terminal_write(&format!("\x1b]0;{title}\x07"));
            }
            UiRequest::PasteToEditor(text) => {
                self.editor().insert_text_at_cursor(&text);
                self.tui.request_render(false);
            }
            UiRequest::SetEditorText(text) => {
                self.editor().set_text(&text);
                self.tui.request_render(false);
            }
            UiRequest::GetEditorText(reply) => {
                let text = self
                    .editor()
                    .get_expanded_text()
                    .unwrap_or_else(|| self.editor().get_text());
                let _ = reply.send(text);
            }
            UiRequest::GetToolsExpanded(reply) => {
                let _ = reply.send(self.tool_output_expanded.get());
            }
            UiRequest::SetToolsExpanded(expanded) => self.set_tools_expanded(expanded),
            UiRequest::SetTheme(name, reply) => {
                let result = self.theme_controller.set_theme_name(&name, true);
                if result.success {
                    let settings = self.settings();
                    let mut settings = settings
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if settings.get_theme().as_deref() != Some(name.as_str()) {
                        settings.set_theme(&name);
                    }
                }
                let _ = reply.send(ThemeUpdateResult {
                    success: result.success,
                    error: result.error,
                });
            }
            UiRequest::GetAllThemes(reply) => {
                let themes =
                    crate::modes::interactive::theme::theme::get_available_themes_with_paths()
                        .into_iter()
                        .map(|info| (info.name, info.path.unwrap_or_default()))
                        .collect();
                let _ = reply.send(themes);
            }
            UiRequest::GetTheme(name, reply) => {
                let _ = reply.send(
                    crate::modes::interactive::theme::theme::get_theme_by_name(&name).map(Arc::new),
                );
            }
            UiRequest::AuthPrompt { prompt, reply } => {
                self.present_auth_prompt(prompt, reply);
            }
            UiRequest::AuthNotify(event) => {
                if let Some(dialog) = self.login_dialog.borrow().clone() {
                    match event {
                        pi_ai::auth::types::AuthEvent::AuthUrl { url, instructions } => {
                            dialog.show_auth(&url, instructions.as_deref());
                        }
                        pi_ai::auth::types::AuthEvent::DeviceCode {
                            user_code,
                            verification_uri,
                            interval_seconds,
                            expires_in_seconds,
                        } => {
                            dialog.show_device_code(
                                &crate::provider_composer::OAuthDeviceCodeInfo {
                                    user_code,
                                    verification_uri,
                                    interval_seconds,
                                    expires_in_seconds,
                                },
                            );
                            dialog.show_waiting("Waiting for authentication...");
                        }
                        pi_ai::auth::types::AuthEvent::Info { message, links } => {
                            dialog.show_info(&message, &links.unwrap_or_default(), false);
                        }
                        pi_ai::auth::types::AuthEvent::Progress { message } => {
                            dialog.show_progress(&message);
                        }
                    }
                }
            }
        }
    }

    fn present_auth_prompt(
        &self,
        prompt: pi_ai::auth::types::AuthPrompt,
        reply: tokio::sync::oneshot::Sender<Result<String, String>>,
    ) {
        let Some(dialog) = self.login_dialog.borrow().clone() else {
            let _ = reply.send(Err("Login cancelled".to_owned()));
            return;
        };
        match prompt.kind {
            pi_ai::auth::types::AuthPromptKind::Select { message, options } => {
                let labels: Vec<String> =
                    options.iter().map(|option| option.label.clone()).collect();
                let reply = Rc::new(std::cell::RefCell::new(Some(reply)));
                let dialog_restore: Rc<std::cell::RefCell<Option<Rc<LoginDialogComponent>>>> =
                    Rc::new(std::cell::RefCell::new(Some(dialog)));
                let editor = Rc::clone(&self.editor_container);
                let tui = Rc::clone(&self.tui);
                let selector = Rc::new(ExtensionSelectorComponent::new(
                    &message,
                    labels,
                    Rc::new({
                        let reply = Rc::clone(&reply);
                        let editor = Rc::clone(&editor);
                        let tui = Rc::clone(&tui);
                        let dialog_restore = Rc::clone(&dialog_restore);
                        move |label: &str| {
                            let id = options
                                .iter()
                                .find(|option| option.label == label)
                                .map(|option| option.id.clone());
                            if let Some(reply) = reply.borrow_mut().take() {
                                let _ = reply.send(id.ok_or_else(|| "Login cancelled".to_owned()));
                            }
                            editor.clear();
                            if let Some(dialog) = dialog_restore.borrow().clone() {
                                let dialog: Rc<dyn pi_tui::tui::Component> = dialog;
                                editor.add_child(Rc::clone(&dialog));
                                tui.set_focus(Some(dialog));
                            }
                            tui.request_render(false);
                        }
                    }),
                    Rc::new({
                        let reply = Rc::clone(&reply);
                        let editor = Rc::clone(&editor);
                        let tui = Rc::clone(&tui);
                        let dialog_restore = Rc::clone(&dialog_restore);
                        move || {
                            if let Some(reply) = reply.borrow_mut().take() {
                                let _ = reply.send(Err("Login cancelled".to_owned()));
                            }
                            editor.clear();
                            if let Some(dialog) = dialog_restore.borrow().clone() {
                                let dialog: Rc<dyn pi_tui::tui::Component> = dialog;
                                editor.add_child(Rc::clone(&dialog));
                                tui.set_focus(Some(dialog));
                            }
                            tui.request_render(false);
                        }
                    }),
                    crate::modes::interactive::components::extension_selector::ExtensionSelectorOptions::default(),
                ));
                self.editor_container.clear();
                self.editor_container
                    .add_child(Rc::clone(&selector) as Rc<dyn pi_tui::tui::Component>);
                self.tui
                    .set_focus(Some(selector as Rc<dyn pi_tui::tui::Component>));
                self.tui.request_render(false);
            }
            pi_ai::auth::types::AuthPromptKind::ManualCode { message, .. } => {
                let receiver = dialog.show_manual_input(&message);
                watcher_send(reply, receiver, prompt.signal);
            }
            pi_ai::auth::types::AuthPromptKind::Text {
                message,
                placeholder,
            }
            | pi_ai::auth::types::AuthPromptKind::Secret {
                message,
                placeholder,
            } => {
                let receiver = dialog.show_prompt(&message, placeholder.as_deref());
                watcher_send(reply, receiver, prompt.signal);
            }
        }
    }
}

/// Forward one login-dialog reply, racing an abort signal.
fn watcher_send(
    reply: tokio::sync::oneshot::Sender<Result<String, String>>,
    receiver: crate::modes::interactive::components::login_dialog::LoginPromptReceiver,
    signal: Option<tokio_util::sync::CancellationToken>,
) {
    tokio::spawn(async move {
        let value = match signal {
            None => receiver.await.ok().and_then(Result::ok),
            Some(signal) => tokio::select! {
                result = receiver => result.ok().and_then(Result::ok),
                () = signal.cancelled() => None,
            },
        };
        let value = value.map_or_else(|| Err("Login cancelled".to_owned()), Ok);
        let _ = reply.send(value);
    });
}

impl<S: crate::settings_manager::SettingsStorageKind> InteractiveMode<S> {
    /// Install the extension-registered keyboard shortcuts, upstream's
    /// `setupExtensionShortcuts`.
    pub(crate) fn setup_extension_shortcuts(&self) {
        let shortcuts = self.session().extension_runner().get_shortcuts();
        if shortcuts.is_empty() {
            self.default_editor.set_on_extension_shortcut(None);
            self.editor().set_on_extension_shortcut(None);
            return;
        }
        let runner = self.session().extension_runner();
        let commands = self.commands();
        let parser = pi_tui::keys::KeyParser::new();
        let handler: crate::modes::interactive::components::custom_editor::ExtensionShortcutHandler =
            Rc::new(move |data: &str| {
                for (key, shortcut) in &shortcuts {
                    if !parser.matches_key(data, key) {
                        continue;
                    }
                    let context = runner.context();
                    let callback = Arc::clone(&shortcut.handler);
                    let commands = commands.clone();
                    let extension_path = shortcut.extension_path.clone();
                    tokio::spawn(async move {
                        callback(context).await;
                        let _ = (commands, extension_path);
                    });
                    return true;
                }
                false
            });
        self.default_editor
            .set_on_extension_shortcut(Some(Rc::clone(&handler)));
        self.editor().set_on_extension_shortcut(Some(handler));
    }

    /// Swap the active editor component, upstream's
    /// `setCustomEditorComponent`.
    ///
    /// Restatement: upstream copies the default editor's action handlers by
    /// iterating its `Map`; the port's `AppEditor` does not enumerate them, so
    /// a custom editor receives submit/change wiring and the appearance
    /// settings, and app actions fall back to the extension's own bindings.
    pub(crate) fn set_custom_editor_component(&self, factory: Option<EditorFactory>) {
        *self.editor_component_factory.borrow_mut() = factory.clone();
        let current_text = self.editor().get_text();
        self.dispose_active_selector();
        self.editor_container.clear();
        match factory {
            Some(factory) => {
                let editor = factory(
                    &self.tui,
                    crate::modes::interactive::theme::theme::get_editor_theme(),
                    Rc::clone(&self.keybindings),
                );
                let commands = self.commands();
                editor.set_on_submit(Some(Rc::new(move |text: &str| {
                    let _ = commands.send(super::FrameCommand::Submit(text.to_owned()));
                })));
                editor.set_text(&current_text);
                editor.set_border_color(self.default_editor.border_color());
                let (padding, visible) = {
                    let settings = self.settings();
                    let settings = settings
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    (
                        usize::try_from(settings.get_editor_padding_x()).unwrap_or_default(),
                        usize::try_from(settings.get_autocomplete_max_visible()).unwrap_or(5),
                    )
                };
                editor.set_padding_x(padding);
                editor.set_autocomplete_max_visible(visible);
                if let Some(provider) = self.autocomplete_provider.borrow().clone() {
                    editor.set_autocomplete_provider(provider);
                }
                *self.editor.borrow_mut() = editor;
            }
            None => {
                self.default_editor.set_text(&current_text);
                *self.editor.borrow_mut() =
                    Rc::new(super::DefaultEditor::new(Rc::clone(&self.default_editor)));
            }
        }
        let editor = self.editor();
        let child: Rc<dyn pi_tui::tui::Component> = editor;
        self.editor_container.add_child(Rc::clone(&child));
        if let Some(indicator) = self.active_status_indicator.borrow().as_ref()
            && self.editor().embed_working_status()
        {
            self.editor()
                .set_working_status_indicator(Some(Rc::clone(indicator) as Rc<dyn crate::modes::interactive::components::custom_editor::EmbeddedStatusIndicator>));
        }
        self.tui.set_focus(Some(child));
        self.tui.request_render(false);
    }
}

//! The OAuth login dialog, upstream's
//! `src/modes/interactive/components/login-dialog.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! It replaces the editor for the duration of a login flow: a title, a
//! dynamic content area that shows auth URLs, device codes, prompts, and
//! progress, and a shared [`Input`] for manual entry.
//!
//! Restatements against upstream:
//!
//! - Upstream's `Promise<string>` returns become [`LoginPromptReceiver`], a
//!   `tokio::sync::oneshot` receiver. Cancellation sends
//!   [`LoginCancelled`] (upstream rejects with `Error("Login cancelled")`);
//!   a prompt replaced before it settles leaves its receiver closed, where
//!   upstream's overwritten resolver left the old promise pending forever.
//! - `AbortController` is a [`CancellationToken`] shared with the auth flow
//!   through [`LoginDialogComponent::signal`].
//! - `Container.children` replacement checks pointer identity on the `Rc`
//!   input instead of JS object identity.
//! - [`LoginDialogComponent::new_with_open_browser`] injects the browser
//!   launcher so the auth screen is testable without opening a real
//!   browser; [`LoginDialogComponent::new`] uses
//!   [`open_browser`], as upstream
//!   does.
//! - The dialog owns its `Input` directly rather than inheriting from
//!   `Container`; [`Component::render`] delegates to the inner container.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_ai::auth::types::AuthInfoLink;
use pi_tui::components::{Input, Spacer, Text};
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container, Focusable, Tui, TuiMouseEvent, TuiMouseEventResult};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::key_hint;
use crate::modes::interactive::theme::theme::{bold, theme};
use crate::provider_composer::OAuthDeviceCodeInfo;
use crate::utils::open_browser::open_browser;

/// The dismissal of an auth step, upstream's rejected
/// `Error("Login cancelled")`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoginCancelled;

impl std::fmt::Display for LoginCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Login cancelled")
    }
}

impl std::error::Error for LoginCancelled {}

/// The outstanding prompt's outcome, upstream's `Promise<string>`.
pub type LoginPromptReceiver = oneshot::Receiver<Result<String, LoginCancelled>>;

/// The completion callback, upstream's
/// `(success: boolean, message?: string) => void`.
pub type LoginCompleteCallback = Rc<dyn Fn(bool, Option<&str>)>;

/// The state shared by the component and its `Input` callbacks, upstream's
/// resolver/rejecter pair plus the abort controller.
struct LoginDialogShared {
    abort: CancellationToken,
    pending: RefCell<Option<oneshot::Sender<Result<String, LoginCancelled>>>>,
    on_complete: LoginCompleteCallback,
}

/// The login dialog, upstream's `LoginDialogComponent`.
pub struct LoginDialogComponent {
    content: Rc<Container>,
    input: Rc<Input>,
    tui: Rc<Tui>,
    shared: Rc<LoginDialogShared>,
    open_browser: Rc<dyn Fn(&str)>,
    focused: Cell<bool>,
    parser: KeyParser,
    container: Container,
}

impl std::fmt::Debug for LoginDialogComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoginDialogComponent")
            .field("focused", &self.focused.get())
            .finish_non_exhaustive()
    }
}

impl LoginDialogComponent {
    /// Build the dialog, upstream's constructor.
    #[must_use]
    pub fn new(
        tui: Rc<Tui>,
        provider_id: &str,
        on_complete: LoginCompleteCallback,
        provider_name_override: Option<&str>,
        title_override: Option<&str>,
    ) -> Self {
        Self::new_with_open_browser(
            tui,
            provider_id,
            on_complete,
            provider_name_override,
            title_override,
            Rc::new(open_browser),
        )
    }

    /// Build the dialog over an injected browser launcher.
    #[must_use]
    pub fn new_with_open_browser(
        tui: Rc<Tui>,
        provider_id: &str,
        on_complete: LoginCompleteCallback,
        provider_name_override: Option<&str>,
        title_override: Option<&str>,
        open_browser: Rc<dyn Fn(&str)>,
    ) -> Self {
        let provider_name = provider_name_override.unwrap_or(provider_id);
        let title =
            title_override.map_or_else(|| format!("Login to {provider_name}"), ToString::to_string);

        let container = Container::default();
        container.add_child(Rc::new(DynamicBorder::themed()));
        container.add_child(Rc::new(Text::with_padding(
            theme().fg("accent", &bold(&title)),
            1,
            0,
        )));

        let content = Rc::new(Container::default());
        let content_child = Rc::clone(&content);
        container.add_child(content_child);
        container.add_child(Rc::new(DynamicBorder::themed()));

        let input = Rc::new(Input::new());
        let shared = Rc::new(LoginDialogShared {
            abort: CancellationToken::new(),
            pending: RefCell::new(None),
            on_complete,
        });

        {
            let shared = Rc::clone(&shared);
            let content = Rc::clone(&content);
            let input_handle = Rc::clone(&input);
            *input.on_submit.borrow_mut() = Some(Rc::new(move |_value: &str| {
                let Some(sender) = shared.pending.borrow_mut().take() else {
                    return;
                };
                let value = input_handle.get_value();
                replace_input_with_submitted_text(&content, &input_handle, &value);
                let _ = sender.send(Ok(value));
            }));
        }
        {
            let shared = Rc::clone(&shared);
            *input.on_escape.borrow_mut() = Some(Rc::new(move || cancel_dialog(&shared)));
        }

        Self {
            content,
            input,
            tui,
            shared,
            open_browser,
            focused: Cell::new(false),
            parser: KeyParser::new(),
            container,
        }
    }

    /// The abort handle the auth flow observes, upstream's `signal`.
    #[must_use]
    pub fn signal(&self) -> CancellationToken {
        self.shared.abort.clone()
    }

    /// Show the auth URL and optional instructions, upstream `showAuth`.
    pub fn show_auth(&self, url: &str, instructions: Option<&str>) {
        self.content.clear();
        self.content.add_child(Rc::new(Spacer::new()));
        self.content.add_child(Rc::new(Text::with_padding(
            theme().fg("accent", &hyperlink(url, url)),
            1,
            0,
        )));
        self.content.add_child(Rc::new(Text::with_padding(
            theme().fg("dim", &hyperlink(url, &click_hint())),
            1,
            0,
        )));
        if let Some(instructions) = instructions.filter(|text| !text.is_empty()) {
            self.content.add_child(Rc::new(Spacer::new()));
            self.content.add_child(Rc::new(Text::with_padding(
                theme().fg("warning", instructions),
                1,
                0,
            )));
        }
        (self.open_browser)(url);
        self.tui.request_render(false);
    }

    /// Show the device-code URL and code, upstream `showDeviceCode`.
    pub fn show_device_code(&self, info: &OAuthDeviceCodeInfo) {
        self.content.clear();
        self.content.add_child(Rc::new(Spacer::new()));
        let url = &info.verification_uri;
        self.content.add_child(Rc::new(Text::with_padding(
            theme().fg("accent", &hyperlink(url, url)),
            1,
            0,
        )));
        self.content.add_child(Rc::new(Text::with_padding(
            theme().fg("dim", &hyperlink(url, &click_hint())),
            1,
            0,
        )));
        self.content.add_child(Rc::new(Spacer::new()));
        self.content.add_child(Rc::new(Text::with_padding(
            theme().fg("warning", &format!("Enter code: {}", info.user_code)),
            1,
            0,
        )));
        self.tui.request_render(false);
    }

    /// Show the input for manual code/URL entry, upstream
    /// `showManualInput`.
    pub fn show_manual_input(&self, prompt: &str) -> LoginPromptReceiver {
        self.input.set_value(String::new());
        self.content.add_child(Rc::new(Spacer::new()));
        self.content
            .add_child(Rc::new(Text::with_padding(theme().fg("dim", prompt), 1, 0)));
        let input_child = Rc::clone(&self.input);
        self.content.add_child(input_child);
        self.content.add_child(Rc::new(Text::with_padding(
            format!("({})", key_hint("tui.select.cancel", "to cancel")),
            1,
            0,
        )));
        self.tui.request_render(false);
        self.start_prompt()
    }

    /// Show a prompt and wait for input, upstream `showPrompt`. It appends
    /// to the existing content so an auth URL stays visible.
    pub fn show_prompt(&self, message: &str, placeholder: Option<&str>) -> LoginPromptReceiver {
        self.content.add_child(Rc::new(Spacer::new()));
        self.content.add_child(Rc::new(Text::with_padding(
            theme().fg("text", message),
            1,
            0,
        )));
        if let Some(placeholder) = placeholder.filter(|text| !text.is_empty()) {
            self.content.add_child(Rc::new(Text::with_padding(
                theme().fg("dim", &format!("e.g., {placeholder}")),
                1,
                0,
            )));
        }
        let input_child = Rc::clone(&self.input);
        self.content.add_child(input_child);
        self.content.add_child(Rc::new(Text::with_padding(
            format!(
                "({} {})",
                key_hint("tui.select.cancel", "to cancel,"),
                key_hint("tui.select.confirm", "to submit")
            ),
            1,
            0,
        )));
        self.input.set_value(String::new());
        self.tui.request_render(false);
        self.start_prompt()
    }

    /// Arm the input's receiver, upstream's resolver/rejecter assignment.
    fn start_prompt(&self) -> LoginPromptReceiver {
        let (sender, receiver) = oneshot::channel();
        *self.shared.pending.borrow_mut() = Some(sender);
        receiver
    }

    /// Show informational lines before another login step, upstream
    /// `showDetails`.
    pub fn show_details(&self, lines: &[&str]) {
        self.content.clear();
        self.content.add_child(Rc::new(Spacer::new()));
        for line in lines {
            self.content
                .add_child(Rc::new(Text::with_padding(*line, 1, 0)));
        }
        self.tui.request_render(false);
    }

    /// Show provider-owned information and links, upstream `showInfo`.
    pub fn show_info(&self, message: &str, links: &[AuthInfoLink], show_close_hint: bool) {
        self.content.add_child(Rc::new(Spacer::new()));
        self.content.add_child(Rc::new(Text::with_padding(
            theme().fg("text", message),
            1,
            0,
        )));
        for link in links {
            let text = link.label.as_ref().map_or_else(
                || link.url.clone(),
                |label| format!("{label}: {}", link.url),
            );
            self.content.add_child(Rc::new(Text::with_padding(
                theme().fg("accent", &hyperlink(&link.url, &text)),
                1,
                0,
            )));
        }
        if show_close_hint {
            self.content.add_child(Rc::new(Spacer::new()));
            self.content.add_child(Rc::new(Text::with_padding(
                format!("({})", key_hint("tui.select.cancel", "to close")),
                1,
                0,
            )));
        }
        self.tui.request_render(false);
    }

    /// Show a polling wait message, upstream `showWaiting`.
    pub fn show_waiting(&self, message: &str) {
        self.content.add_child(Rc::new(Spacer::new()));
        self.content.add_child(Rc::new(Text::with_padding(
            theme().fg("dim", message),
            1,
            0,
        )));
        self.content.add_child(Rc::new(Text::with_padding(
            format!("({})", key_hint("tui.select.cancel", "to cancel")),
            1,
            0,
        )));
        self.tui.request_render(false);
    }

    /// Show one progress line, upstream `showProgress`.
    pub fn show_progress(&self, message: &str) {
        self.content.add_child(Rc::new(Text::with_padding(
            theme().fg("dim", message),
            1,
            0,
        )));
        self.tui.request_render(false);
    }

    /// The raw input dispatch, upstream `handleInput`.
    pub fn handle_input(&self, data: &str) {
        if get_keybindings().matches(&self.parser, data, "tui.select.cancel") {
            cancel_dialog(&self.shared);
            return;
        }
        self.input.handle_input(data);
    }
}

impl Component for LoginDialogComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_input(&self, data: &str) {
        self.handle_input(data);
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }

    fn is_stock_mouse_container(&self) -> bool {
        true
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.container.handle_mouse(event)
    }

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }
}

impl Focusable for LoginDialogComponent {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
        self.input.set_focused(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

/// Cancel the flow, upstream's `cancel`: abort the token, settle a pending
/// prompt as cancelled, and report through `onComplete`.
fn cancel_dialog(shared: &LoginDialogShared) {
    shared.abort.cancel();
    if let Some(sender) = shared.pending.borrow_mut().take() {
        let _ = sender.send(Err(LoginCancelled));
    }
    (shared.on_complete)(false, Some("Login cancelled"));
}

/// Replace the mounted input with the submitted value, upstream
/// `replaceInputWithSubmittedText`.
fn replace_input_with_submitted_text(content: &Container, input: &Rc<Input>, value: &str) {
    let input_pointer = Rc::as_ptr(input).cast::<()>();
    let mut children = content.children.borrow_mut();
    for child in children.iter_mut() {
        if Rc::as_ptr(child).cast::<()>() == input_pointer {
            *child = Rc::new(Text::with_padding(format!("> {value}"), 0, 0));
        }
    }
}

/// An OSC 8 hyperlink, upstream's inline escape template.
fn hyperlink(url: &str, text: &str) -> String {
    format!("\x1b]8;;{url}\x07{text}\x1b]8;;\x07")
}

/// The platform click hint upstream selects on `process.platform`.
fn click_hint() -> String {
    if cfg!(target_os = "macos") {
        "Cmd+click to open"
    } else {
        "Ctrl+click to open"
    }
    .to_string()
}

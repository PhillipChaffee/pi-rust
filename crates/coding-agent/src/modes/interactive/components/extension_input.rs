//! The extension text prompt, upstream's
//! `src/modes/interactive/components/extension-input.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! One line of input over the standard bordered frame, with the optional
//! countdown that appends the remaining seconds to the title and cancels at
//! zero. Upstream ignores its `placeholder` argument; the port keeps the
//! parameter so call sites read the same.
//!
//! Restatements: the callbacks are `Rc<dyn Fn>` closures, the synchronous
//! port of upstream's function properties. The countdown's `tui` handle
//! restates as its render request ([`ExtensionInputOptions::request_render`]),
//! because the frame's render loop owns the wall clock and the tick.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_tui::components::input::Input;
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container, Focusable};

use super::countdown_timer::CountdownTimer;
use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::key_hint;
use crate::modes::interactive::theme::theme::theme;

/// The submit callback, upstream's `onSubmit`.
pub type ExtensionInputSubmit = Rc<dyn Fn(String)>;

/// The dismissal callback, upstream's `onCancel`.
pub type ExtensionInputCancel = Rc<dyn Fn()>;

/// The optional extra knobs, upstream's `ExtensionInputOptions`.
#[derive(Default)]
pub struct ExtensionInputOptions {
    /// The render request the countdown drives, standing in for upstream's
    /// `tui` handle. A timeout without one leaves the countdown off,
    /// matching upstream's `timeout && tui` conjunct.
    pub request_render: Option<Rc<dyn Fn()>>,
    /// The countdown duration in milliseconds, upstream `timeout`.
    pub timeout_ms: Option<u64>,
}

impl std::fmt::Debug for ExtensionInputOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionInputOptions")
            .field("request_render", &self.request_render.is_some())
            .field("timeout_ms", &self.timeout_ms)
            .finish()
    }
}

/// The extension text prompt, upstream's `ExtensionInputComponent`.
pub struct ExtensionInputComponent {
    container: Container,
    input: Rc<Input>,
    title_text: Rc<Text>,
    base_title: String,
    countdown: RefCell<Option<Rc<CountdownTimer>>>,
    on_submit: ExtensionInputSubmit,
    on_cancel: ExtensionInputCancel,
    parser: KeyParser,
    focused: Cell<bool>,
}

impl std::fmt::Debug for ExtensionInputComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionInputComponent")
            .field("base_title", &self.base_title)
            .field("focused", &self.focused.get())
            .finish_non_exhaustive()
    }
}

impl ExtensionInputComponent {
    /// Build the prompt, upstream's constructor body.
    #[must_use]
    pub fn new(
        title: &str,
        _placeholder: Option<&str>,
        on_submit: ExtensionInputSubmit,
        on_cancel: ExtensionInputCancel,
        opts: ExtensionInputOptions,
    ) -> Self {
        let container = Container::default();
        container.add_child(Rc::new(DynamicBorder::themed()));
        container.add_child(Rc::new(Spacer::new()));

        let title_text = Rc::new(Text::with_padding(theme().fg("accent", title), 1, 0));
        container.add_child({
            let child: Rc<dyn Component> = title_text.clone();
            child
        });
        container.add_child(Rc::new(Spacer::new()));

        let countdown = opts
            .timeout_ms
            .filter(|timeout| *timeout > 0)
            .and_then(|timeout| {
                let request_render = opts.request_render?;
                let cancel = Rc::clone(&on_cancel);
                let title_text = title_text.clone();
                let base_title = title.to_string();
                Some(Rc::new(CountdownTimer::new(
                    timeout,
                    Some(request_render),
                    Rc::new(move |seconds| {
                        title_text
                            .set_text(&theme().fg("accent", &format!("{base_title} ({seconds}s)")));
                    }),
                    cancel,
                )))
            });

        let input = Rc::new(Input::new());
        container.add_child({
            let child: Rc<dyn Component> = input.clone();
            child
        });
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(Text::with_padding(
            format!(
                "{}  {}",
                key_hint("tui.select.confirm", "submit"),
                key_hint("tui.select.cancel", "cancel")
            ),
            1,
            0,
        )));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(DynamicBorder::themed()));

        Self {
            container,
            input,
            title_text,
            base_title: title.to_string(),
            countdown: RefCell::new(countdown),
            on_submit,
            on_cancel,
            parser: KeyParser::new(),
            focused: Cell::new(false),
        }
    }

    /// The armed countdown, when the prompt was built with a timeout.
    #[must_use]
    pub fn countdown(&self) -> Option<Rc<CountdownTimer>> {
        self.countdown.borrow().as_ref().map(Rc::clone)
    }

    /// The title row, for callers that assert on the countdown suffix.
    #[must_use]
    pub fn title_text(&self) -> Rc<Text> {
        Rc::clone(&self.title_text)
    }

    /// The current input value, upstream's `input.getValue`.
    #[must_use]
    pub fn value(&self) -> String {
        self.input.get_value()
    }

    /// The keyboard dispatch, upstream `handleInput`.
    pub fn handle_input(&self, data: &str) {
        let matches = |action| get_keybindings().matches(&self.parser, data, action);
        if matches("tui.select.confirm") || data == "\n" {
            (self.on_submit)(self.input.get_value());
        } else if matches("tui.select.cancel") {
            (self.on_cancel)();
        } else {
            self.input.handle_input(data);
        }
    }

    /// Stop the countdown, upstream `dispose`.
    pub fn dispose(&self) {
        if let Some(countdown) = self.countdown.borrow().as_ref() {
            countdown.dispose();
        }
    }
}

impl Focusable for ExtensionInputComponent {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
        self.input.set_focused(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

impl Component for ExtensionInputComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn handle_input(&self, data: &str) {
        Self::handle_input(self, data);
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }
}

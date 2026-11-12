//! The generic extension selector, upstream's
//! `src/modes/interactive/components/extension-selector.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#213).
//!
//! A string-option dialog with `j`/`k` and keybinding navigation, an
//! `app.tools.expand` toggle, confirm/cancel callbacks, and an optional
//! countdown that appends the remaining seconds to the title. The
//! extension dialogs mount it through the interactive frame; the list
//! itself owns nothing session-shaped.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container};

use super::countdown_timer::CountdownTimer;
use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::{key_hint, raw_key_hint};
use crate::modes::interactive::theme::theme::{bold, theme};

/// The selected option, upstream's `onSelect`.
pub type ExtensionSelectCallback = Rc<dyn Fn(&str)>;

/// The dismissal, upstream's `onCancel`.
pub type ExtensionCancelCallback = Rc<dyn Fn()>;

/// The optional extra knobs, upstream's `ExtensionSelectorOptions`.
#[derive(Default)]
pub struct ExtensionSelectorOptions {
    /// The render request the countdown drives, standing in for upstream's
    /// `tui` handle. A timeout without one leaves the countdown off,
    /// matching upstream's `timeout && tui` conjunct.
    pub request_render: Option<Rc<dyn Fn()>>,
    /// The countdown duration in milliseconds.
    pub timeout_ms: Option<u64>,
    /// The `app.tools.expand` binding, upstream's `onToggleToolsExpanded`.
    pub on_toggle_tools_expanded: Option<Rc<dyn Fn()>>,
}

impl std::fmt::Debug for ExtensionSelectorOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionSelectorOptions")
            .field("request_render", &self.request_render.is_some())
            .field("timeout_ms", &self.timeout_ms)
            .field(
                "on_toggle_tools_expanded",
                &self.on_toggle_tools_expanded.is_some(),
            )
            .finish()
    }
}

/// The generic option-list dialog, upstream's `ExtensionSelectorComponent`.
pub struct ExtensionSelectorComponent {
    options: Vec<String>,
    selected_index: Cell<usize>,
    list_container: Rc<Container>,
    base_title: String,
    countdown: RefCell<Option<Rc<CountdownTimer>>>,
    on_select: ExtensionSelectCallback,
    on_cancel: ExtensionCancelCallback,
    on_toggle_tools_expanded: Option<Rc<dyn Fn()>>,
    parser: KeyParser,
    container: Container,
}

impl std::fmt::Debug for ExtensionSelectorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionSelectorComponent")
            .field("options", &self.options.len())
            .field("selected_index", &self.selected_index.get())
            .finish_non_exhaustive()
    }
}

impl ExtensionSelectorComponent {
    /// Build the dialog over the title and option strings.
    #[must_use]
    pub fn new(
        title: &str,
        options: Vec<String>,
        on_select: ExtensionSelectCallback,
        on_cancel: ExtensionCancelCallback,
        opts: ExtensionSelectorOptions,
    ) -> Self {
        let list_container = Rc::new(Container::default());
        let title_text = Rc::new(Text::with_padding(theme().fg("accent", &bold(title)), 1, 0));
        let base_title = title.to_string();
        let component = Self {
            options,
            selected_index: Cell::new(0),
            list_container: Rc::clone(&list_container),
            base_title,
            countdown: RefCell::new(None),
            on_select,
            on_cancel,
            on_toggle_tools_expanded: opts.on_toggle_tools_expanded,
            parser: KeyParser::new(),
            container: Container::default(),
        };

        component
            .container
            .add_child(Rc::new(DynamicBorder::themed()));
        component.container.add_child(Rc::new(Spacer::new()));
        let title_child: Rc<dyn Component> = title_text.clone();
        component.container.add_child(title_child);
        component.container.add_child(Rc::new(Spacer::new()));

        let timeout_ms = opts.timeout_ms.unwrap_or(0);
        if timeout_ms > 0 && opts.request_render.is_some() {
            let cancel = Rc::clone(&component.on_cancel);
            let title_text = Rc::clone(&title_text);
            let base_title = component.base_title.clone();
            let countdown = Rc::new(CountdownTimer::new(
                timeout_ms,
                opts.request_render,
                Rc::new(move |seconds| {
                    title_text.set_text(
                        &theme().fg("accent", &bold(&format!("{base_title} ({seconds}s)"))),
                    );
                }),
                Rc::new(move || cancel()),
            ));
            *component.countdown.borrow_mut() = Some(Rc::clone(&countdown));
        }

        let list_child: Rc<dyn Component> = list_container;
        component.container.add_child(list_child);
        component.container.add_child(Rc::new(Spacer::new()));
        component.container.add_child(Rc::new(Text::with_padding(
            raw_key_hint("↑↓", "navigate")
                + "  "
                + &key_hint("tui.select.confirm", "select")
                + "  "
                + &key_hint("tui.select.cancel", "cancel"),
            1,
            0,
        )));
        component.container.add_child(Rc::new(Spacer::new()));
        component
            .container
            .add_child(Rc::new(DynamicBorder::themed()));

        component.update_list();
        component
    }

    /// The armed countdown, when the dialog was built with a timeout.
    #[must_use]
    pub fn countdown(&self) -> Option<Rc<CountdownTimer>> {
        self.countdown.borrow().as_ref().map(Rc::clone)
    }

    /// Redraw the option rows, upstream's `updateList`.
    fn update_list(&self) {
        self.list_container.clear();
        for (index, option) in self.options.iter().enumerate() {
            let is_selected = index == self.selected_index.get();
            let text = if is_selected {
                theme().fg("accent", "→ ") + &theme().fg("accent", option)
            } else {
                format!("  {}", theme().fg("text", option))
            };
            self.list_container
                .add_child(Rc::new(Text::with_padding(text, 1, 0)));
        }
    }

    /// The keyboard dispatch, upstream `handleInput`.
    pub fn handle_input(&self, data: &str) {
        let matches = |action: &str| get_keybindings().matches(&self.parser, data, action);
        if matches("app.tools.expand") {
            if let Some(toggle) = &self.on_toggle_tools_expanded {
                toggle();
            }
        } else if matches("tui.select.up") || data == "k" {
            self.selected_index
                .set(self.selected_index.get().saturating_sub(1));
            self.update_list();
        } else if matches("tui.select.down") || data == "j" {
            let last = self.options.len().saturating_sub(1);
            self.selected_index
                .set((self.selected_index.get() + 1).min(last));
            self.update_list();
        } else if matches("tui.select.confirm") || data == "\n" {
            if let Some(option) = self.options.get(self.selected_index.get()) {
                (self.on_select)(option);
            }
        } else if matches("tui.select.cancel") {
            (self.on_cancel)();
        }
    }

    /// Stop the countdown, upstream `dispose`.
    pub fn dispose(&self) {
        if let Some(countdown) = self.countdown.borrow().as_ref() {
            countdown.dispose();
        }
    }
}

impl Component for ExtensionSelectorComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn handle_input(&self, data: &str) {
        Self::handle_input(self, data);
    }

    fn invalidate(&self) {}

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }
}

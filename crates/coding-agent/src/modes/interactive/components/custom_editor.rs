//! The app-facing editor, upstream's
//! `src/modes/interactive/components/custom-editor.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! It fronts [`Editor`] with the app keybinding dispatch — extension
//! shortcuts, image paste, interrupt/exit hooks, and registered app actions,
//! with the editor's explicit history bindings taking precedence — and,
//! when [`CustomEditorOptions::embed_working_status`] is set, draws a
//! `StatusIndicator` inside the top border.
//!
//! Restatements against upstream:
//!
//! - JS subclassing becomes composition. [`CustomEditor`] derefs to
//!   [`Editor`] for the inherited surface, and [`CustomEditor::render`]
//!   replaces both border lines with borders drawn in the wrapper's own
//!   color function. Upstream mutated the inherited `borderColor` field;
//!   the wrapper owns that state instead, because a sourced border color
//!   stays configurable through [`CustomEditor::set_border_color`] even
//!   behind an `Rc`. The scroll-indicator wording is rebuilt from the
//!   editor's rendered borders (`create_scroll_border` mirrors the
//!   editor's private helper) and the hidden-line counts are read back from
//!   them, since the editor keeps its scroll offset private.
//! - Upstream's concrete `StatusIndicator` type becomes
//!   [`EmbeddedStatusIndicator`], the two-method surface the border reads.
//!   The sibling `status-indicator` unit (#213) owns the concrete type and
//!   supplies the impl once its module is registered; until then the frame
//!   can mount any adapter.
//! - [`CustomEditorOptions`] carries [`EditorOptions`] as a field rather
//!   than the TS intersection type.
//! - `Map<AppKeybinding, () => void>` becomes an [`IndexMap`] keyed by the
//!   action id string; insertion order is preserved the way JS `Map`
//!   iteration does, so two actions bound to one key dispatch in
//!   registration order.
//! - The upstream constructor takes the app keybinding manager by value;
//!   the port shares it as an `Rc`, matching how the interactive frame's
//!   other components reach the one manager.

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

use indexmap::IndexMap;
use pi_tui::autocomplete::AutocompleteProvider;
use pi_tui::components::EditorComponent;
use pi_tui::components::editor::{
    Editor, EditorCallback, EditorColorFn, EditorOptions, EditorTheme,
};
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Focusable, Tui, TuiMouseEvent, TuiMouseEventResult};
use pi_tui::utils::{slice_by_column, visible_width};

use crate::core::keybindings::AppKeybindingsManager;
use crate::utils::ansi::strip_ansi;

/// Construction options, upstream's `CustomEditorOptions`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CustomEditorOptions {
    /// The editor's own construction options, upstream's `EditorOptions`
    /// intersection member.
    pub editor: EditorOptions,
    /// Render working, compaction, summarization, and retry status in the
    /// editor's top border. Default: false.
    pub embed_working_status: bool,
}

/// An app-level action handler, upstream's `() => void`.
pub type AppActionHandler = Rc<dyn Fn()>;

/// A data-driven shortcut handler returning whether it consumed the input,
/// upstream's `onExtensionShortcut`.
pub type ExtensionShortcutHandler = Rc<dyn Fn(&str) -> bool>;

/// The status surface the editor's top border reads, upstream's
/// `StatusIndicator` methods `renderInBorder` and `renderSpinnerInBorder`.
pub trait EmbeddedStatusIndicator {
    /// The full status line clipped to `width`, without border padding.
    fn render_in_border(&self, width: usize) -> String;
    /// Only the animated spinner clipped to `width`.
    fn render_spinner_in_border(&self, width: usize) -> String;
}

/// The editor the interactive app drives, upstream's `CustomEditor`.
pub struct CustomEditor {
    editor: Editor,
    keybindings: Rc<AppKeybindingsManager>,
    working_status_indicator: RefCell<Option<Rc<dyn EmbeddedStatusIndicator>>>,
    /// Whether a status indicator may render in the top border, upstream's
    /// readonly `embedWorkingStatus`.
    pub embed_working_status: bool,
    action_handlers: RefCell<IndexMap<String, AppActionHandler>>,
    on_escape: RefCell<Option<AppActionHandler>>,
    on_ctrl_d: RefCell<Option<AppActionHandler>>,
    on_paste_image: RefCell<Option<AppActionHandler>>,
    on_extension_shortcut: RefCell<Option<ExtensionShortcutHandler>>,
    border_color: RefCell<EditorColorFn>,
    parser: KeyParser,
}

impl std::fmt::Debug for CustomEditor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomEditor")
            .field("embed_working_status", &self.embed_working_status)
            .field(
                "working_status_indicator",
                &self.working_status_indicator.borrow().is_some(),
            )
            .field("action_handlers", &self.action_handlers.borrow().len())
            .finish_non_exhaustive()
    }
}

impl CustomEditor {
    /// Build the editor with upstream's default options.
    #[must_use]
    pub fn new(tui: &Rc<Tui>, theme: EditorTheme, keybindings: Rc<AppKeybindingsManager>) -> Self {
        Self::with_options(tui, theme, keybindings, CustomEditorOptions::default())
    }

    /// Build the editor over explicit options.
    #[must_use]
    pub fn with_options(
        tui: &Rc<Tui>,
        theme: EditorTheme,
        keybindings: Rc<AppKeybindingsManager>,
        options: CustomEditorOptions,
    ) -> Self {
        let border_color = theme.border_color.clone();
        Self {
            editor: Editor::with_options(tui, theme, options.editor),
            keybindings,
            working_status_indicator: RefCell::new(None),
            embed_working_status: options.embed_working_status,
            action_handlers: RefCell::new(IndexMap::new()),
            on_escape: RefCell::new(None),
            on_ctrl_d: RefCell::new(None),
            on_paste_image: RefCell::new(None),
            on_extension_shortcut: RefCell::new(None),
            border_color: RefCell::new(border_color),
            parser: KeyParser::new(),
        }
    }

    /// The wrapped editor, for readers that need the concrete type.
    #[must_use]
    pub const fn editor(&self) -> &Editor {
        &self.editor
    }

    /// Install the indicator drawn inside the top border, upstream's
    /// `setWorkingStatusIndicator`.
    pub fn set_working_status_indicator(&self, indicator: Option<Rc<dyn EmbeddedStatusIndicator>>) {
        *self.working_status_indicator.borrow_mut() = indicator;
    }

    /// Replace the border color function, upstream's assigned
    /// `borderColor`; the next render picks it up.
    pub fn set_border_color(&self, color: EditorColorFn) {
        *self.border_color.borrow_mut() = color;
    }

    /// The current border color function.
    #[must_use]
    pub fn border_color(&self) -> EditorColorFn {
        Rc::clone(&self.border_color.borrow())
    }

    /// Register a handler for an app action, upstream's `onAction`.
    pub fn on_action(&self, action: &str, handler: AppActionHandler) {
        self.action_handlers
            .borrow_mut()
            .insert(action.to_string(), handler);
    }

    /// Install the Escape handler that overrides the registered
    /// `app.interrupt` action, upstream's assigned `onEscape`.
    pub fn set_on_escape(&self, handler: Option<AppActionHandler>) {
        *self.on_escape.borrow_mut() = handler;
    }

    /// Install the Ctrl+D handler that overrides the registered `app.exit`
    /// action, upstream's assigned `onCtrlD`.
    pub fn set_on_ctrl_d(&self, handler: Option<AppActionHandler>) {
        *self.on_ctrl_d.borrow_mut() = handler;
    }

    /// Install the clipboard-paste handler, upstream's assigned
    /// `onPasteImage`.
    pub fn set_on_paste_image(&self, handler: Option<AppActionHandler>) {
        *self.on_paste_image.borrow_mut() = handler;
    }

    /// Install the extension shortcut handler, upstream's assigned
    /// `onExtensionShortcut`.
    pub fn set_on_extension_shortcut(&self, handler: Option<ExtensionShortcutHandler>) {
        *self.on_extension_shortcut.borrow_mut() = handler;
    }

    /// Install the submit callback on the wrapped editor, upstream's
    /// assigned `onSubmit` field.
    pub fn set_on_submit(&self, callback: Option<EditorCallback>) {
        *self.editor.on_submit.borrow_mut() = callback;
    }

    /// Install the change callback on the wrapped editor, upstream's
    /// assigned `onChange` field.
    pub fn set_on_change(&self, callback: Option<EditorCallback>) {
        *self.editor.on_change.borrow_mut() = callback;
    }

    /// Toggle the submit path, upstream's assigned `disableSubmit` field.
    pub fn set_disable_submit(&self, disabled: bool) {
        self.editor.disable_submit.set(disabled);
    }

    /// Whether the input matches an app action.
    fn matches(&self, data: &str, action: &str) -> bool {
        self.keybindings
            .manager()
            .matches(&self.parser, data, action)
    }

    /// The input dispatch, upstream `CustomEditor.handleInput`.
    pub fn handle_input(&self, data: &str) {
        // Extension-registered shortcuts win over every app binding.
        let shortcut = self.on_extension_shortcut.borrow().as_ref().map(Rc::clone);
        if let Some(shortcut) = shortcut
            && shortcut(data)
        {
            return;
        }

        // Clipboard paste keybinding.
        if self.matches(data, "app.clipboard.pasteImage") {
            if let Some(handler) = self.on_paste_image.borrow().as_ref() {
                handler();
            }
            return;
        }

        // Escape/interrupt, but only while no autocomplete picker is open:
        // an open picker needs the editor's own cancellation path.
        if self.matches(data, "app.interrupt") {
            if !self.editor.is_showing_autocomplete() {
                let handler = self
                    .on_escape
                    .borrow()
                    .clone()
                    .or_else(|| self.action_handlers.borrow().get("app.interrupt").cloned());
                if let Some(handler) = handler {
                    handler();
                    return;
                }
            }
            self.editor.handle_input(data);
            return;
        }

        // Exit (Ctrl+D), only while the editor is empty; otherwise the
        // editor's delete-character-forward handling applies.
        if self.matches(data, "app.exit") && self.editor.get_text().is_empty() {
            let handler = self
                .on_ctrl_d
                .borrow()
                .clone()
                .or_else(|| self.action_handlers.borrow().get("app.exit").cloned());
            if let Some(handler) = handler {
                handler();
            }
            return;
        }

        // Explicit history bindings take precedence over app actions while
        // the editor is focused, so a user can bind Ctrl+P even though it
        // cycles models by default.
        if self.matches(data, "tui.editor.historyPrevious")
            || self.matches(data, "tui.editor.historyNext")
        {
            self.editor.handle_input(data);
            return;
        }

        // Every other registered app action.
        let matched = self
            .action_handlers
            .borrow()
            .iter()
            .find(|(action, _)| {
                action.as_str() != "app.interrupt"
                    && action.as_str() != "app.exit"
                    && self.matches(data, action)
            })
            .map(|(_, handler)| Rc::clone(handler));
        if let Some(handler) = matched {
            handler();
            return;
        }

        self.editor.handle_input(data);
    }

    /// The top border: the embedded status when enabled, else the editor's
    /// scroll-indicator rule, upstream `renderTopBorder`.
    fn render_top_border(&self, width: usize, hidden_line_count: usize) -> String {
        let indicator = self.working_status_indicator.borrow().clone();
        let (Some(indicator), true) = (indicator, self.embed_working_status) else {
            return self.render_border(width, hidden_line_count, '↑');
        };
        if width == 0 {
            return String::new();
        }

        let mut status = indicator.render_in_border(width.saturating_sub(5).max(1));
        let mut status_width = visible_width(&status);
        if status_width == 0 {
            return self.render_border(width, hidden_line_count, '↑');
        }

        let overflow_label =
            (hidden_line_count > 0).then(|| format!(" ↑ {hidden_line_count} more "));
        let overflow_label_width = overflow_label
            .as_ref()
            .map_or(0, |label| visible_width(label));
        let overflow_start = width.saturating_sub(overflow_label_width) / 2;
        let mut can_fit_overflow = overflow_label.is_some()
            && overflow_label_width + 2 <= width
            && overflow_start >= status_width.saturating_add(5);

        if overflow_label.is_some() && !can_fit_overflow {
            status = indicator.render_spinner_in_border(width);
            status_width = visible_width(&status);
            // Upstream re-evaluates the closure after the swap: a narrower
            // spinner can make the overflow label fit again.
            can_fit_overflow = overflow_label_width + 2 <= width
                && overflow_start >= status_width.saturating_add(5);
        }

        if can_fit_overflow {
            let left_block_width = 3 + status_width + 1;
            let label = overflow_label.unwrap_or_default();
            return self.render_border_text("── ")
                + &status
                + &self.render_border_text(&format!(
                    " {}{}{}",
                    "─".repeat(overflow_start - left_block_width),
                    label,
                    "─".repeat(width - overflow_start - overflow_label_width),
                ));
        }

        if width >= status_width + 5 {
            return self.render_border_text("── ")
                + &status
                + &self.render_border_text(&format!(" {}", "─".repeat(width - status_width - 4)));
        }

        status = indicator.render_spinner_in_border(width);
        status_width = visible_width(&status);
        let prefix_width = 3.min(width.saturating_sub(status_width));
        self.render_border_text(&"─".repeat(prefix_width))
            + &status
            + &self
                .render_border_text(&"─".repeat(width.saturating_sub(prefix_width + status_width)))
    }

    /// One plain or scroll-indicator border line, upstream
    /// `renderBottomBorder` and the non-status `renderTopBorder` arm.
    fn render_border(&self, width: usize, hidden_line_count: usize, direction: char) -> String {
        let border = if hidden_line_count > 0 {
            create_scroll_border(&direction.to_string(), hidden_line_count, width)
        } else {
            "─".repeat(width)
        };
        self.render_border_text(&border)
    }

    /// Color one border run with the current border color.
    fn render_border_text(&self, text: &str) -> String {
        let color = self.border_color.borrow();
        color(text)
    }
}

impl Deref for CustomEditor {
    type Target = Editor;

    fn deref(&self) -> &Self::Target {
        &self.editor
    }
}

impl Component for CustomEditor {
    fn render(&self, width: usize) -> Vec<String> {
        let mut lines = self.editor.render(width);
        if width == 0 {
            return lines;
        }
        let Some(top) = lines.first() else {
            return lines;
        };
        let hidden_top = hidden_line_count(top, '↑');
        lines[0] = self.render_top_border(width, hidden_top);

        if let Some(bottom) = lines.last() {
            let hidden_bottom = hidden_line_count(bottom, '↓');
            let rendered = self.render_border(width, hidden_bottom, '↓');
            if let Some(last) = lines.last_mut() {
                *last = rendered;
            }
        }
        lines
    }

    fn handle_input(&self, data: &str) {
        self.handle_input(data);
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        Component::handle_mouse(&self.editor, event)
    }

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }
}

impl Focusable for CustomEditor {
    fn set_focused(&self, focused: bool) {
        self.editor.set_focused(focused);
    }

    fn is_focused(&self) -> bool {
        self.editor.is_focused()
    }
}

impl EditorComponent for CustomEditor {
    fn get_text(&self) -> String {
        self.editor.get_text()
    }

    fn set_text(&self, text: &str) {
        self.editor.set_text(text);
    }

    fn add_to_history(&self, text: &str) {
        self.editor.add_to_history(text);
    }

    fn insert_text_at_cursor(&self, text: &str) {
        self.editor.insert_text_at_cursor(text);
    }

    fn get_expanded_text(&self) -> Option<String> {
        Some(self.editor.get_expanded_text())
    }

    fn set_padding_x(&self, padding: usize) {
        self.editor.set_padding_x(padding);
    }

    fn set_autocomplete_provider(&self, provider: std::sync::Arc<dyn AutocompleteProvider>) {
        self.editor.set_autocomplete_provider(provider);
    }

    fn set_autocomplete_max_visible(&self, max_visible: usize) {
        self.editor.set_autocomplete_max_visible(max_visible);
    }
}

/// Read the hidden-line count back out of a rendered scroll border, the
/// editor's private `hiddenLineCount` argument.
fn hidden_line_count(border: &str, direction: char) -> usize {
    let visible = strip_ansi(border);
    let Some((_, rest)) = visible.split_once(&format!("{direction} ")) else {
        return 0;
    };
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().unwrap_or(0)
}

/// The centered scroll-indicator border, the editor's private
/// `createScrollBorder`.
fn create_scroll_border(direction: &str, hidden_line_count: usize, width: usize) -> String {
    let label = format!(" {direction} {hidden_line_count} more ");
    let label_width = visible_width(&label);
    if label_width + 2 <= width {
        let left_width = (width - label_width) / 2;
        return format!(
            "{}{label}{}",
            "─".repeat(left_width),
            "─".repeat(width - left_width - label_width)
        );
    }

    let indicator = format!("─── {direction} {hidden_line_count} more ");
    let indicator_width = visible_width(&indicator);
    if width >= indicator_width {
        return format!("{indicator}{}", "─".repeat(width - indicator_width));
    }

    let ellipsis = "...".get(..width.min(3)).unwrap_or("...");
    let indicator_width = width.saturating_sub(visible_width(ellipsis));
    format!(
        "{}{ellipsis}",
        slice_by_column(&indicator, 0, indicator_width, true)
    )
}

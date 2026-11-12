//! The extension multi-line prompt, upstream's
//! `src/modes/interactive/components/extension-editor.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! A bordered, titled [`Editor`] with submit/cancel callbacks and Ctrl+G for
//! the external editor.
//!
//! Restatements against upstream:
//!
//! - `editInExternalEditor` is the shared
//!   [`crate::modes::interactive::external_editor`] module; this dialog calls
//!   its blocking form because `Component::handle_input` cannot await and the
//!   TUI stops for the editor's whole lifetime. A failed spawn, write, or
//!   read reports [`ExternalEditorResult::Failed`] where upstream rejects
//!   the promise.
//! - The app keybinding manager is shared as an `Rc`, like
//!   [`super::custom_editor::CustomEditor`].
//! - The command fallback's `explicit || VISUAL || EDITOR || platform
//!   default` order lives in the shared module's
//!   [`crate::modes::interactive::external_editor::resolve_external_editor_command_with`].

use std::cell::Cell;
use std::rc::Rc;

use pi_tui::components::editor::{Editor, EditorOptions};
use pi_tui::components::{Spacer, Text};
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{
    Component, Container, Focusable, Tui, TuiMouseEvent, TuiMouseEventResult, TuiStopOptions,
};

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::key_hint;
use crate::core::keybindings::AppKeybindingsManager;
use crate::modes::interactive::external_editor::{
    ExternalEditorResult, edit_in_external_editor_blocking, resolve_external_editor_command,
};
use crate::modes::interactive::theme::theme::{get_editor_theme, theme};

/// The extension prompt dialog, upstream's `ExtensionEditorComponent`.
pub struct ExtensionEditorComponent {
    editor: Rc<Editor>,
    tui: Rc<Tui>,
    keybindings: Rc<AppKeybindingsManager>,
    external_editor_command: String,
    on_cancel: Rc<dyn Fn()>,
    focused: Cell<bool>,
    parser: KeyParser,
    container: Container,
}

impl std::fmt::Debug for ExtensionEditorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionEditorComponent")
            .field("focused", &self.focused.get())
            .finish_non_exhaustive()
    }
}

impl ExtensionEditorComponent {
    /// Build the dialog: top border, title, editor, key hint, bottom border.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors upstream's constructor parameter for parameter"
    )]
    #[must_use]
    pub fn new(
        tui: Rc<Tui>,
        keybindings: Rc<AppKeybindingsManager>,
        title: &str,
        prefill: Option<&str>,
        on_submit: Rc<dyn Fn(&str)>,
        on_cancel: Rc<dyn Fn()>,
        options: EditorOptions,
        external_editor_command: Option<&str>,
    ) -> Self {
        let container = Container::default();
        container.add_child(Rc::new(DynamicBorder::themed()));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(Text::with_padding(
            theme().fg("accent", title),
            1,
            0,
        )));
        container.add_child(Rc::new(Spacer::new()));

        let editor = Rc::new(Editor::with_options(&tui, get_editor_theme(), options));
        if let Some(prefill) = prefill.filter(|text| !text.is_empty()) {
            editor.set_text(prefill);
        }
        *editor.on_submit.borrow_mut() = Some(Rc::new(move |text: &str| on_submit(text)));
        let editor_child = Rc::clone(&editor);
        container.add_child(editor_child);
        container.add_child(Rc::new(Spacer::new()));

        let hint = key_hint("tui.select.confirm", "submit")
            + "  "
            + &key_hint("tui.input.newLine", "newline")
            + "  "
            + &key_hint("tui.select.cancel", "cancel")
            + "  "
            + &key_hint("app.editor.external", "external editor");
        container.add_child(Rc::new(Text::with_padding(hint, 1, 0)));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(DynamicBorder::themed()));

        Self {
            editor,
            tui,
            keybindings,
            external_editor_command: resolve_external_editor_command(external_editor_command),
            on_cancel,
            focused: Cell::new(false),
            parser: KeyParser::new(),
            container,
        }
    }

    /// The wrapped editor, for callers that configure it after build.
    #[must_use]
    pub fn editor(&self) -> &Editor {
        &self.editor
    }

    /// Whether the dialog holds focus, upstream's `focused`.
    #[must_use]
    pub const fn is_focused(&self) -> bool {
        self.focused.get()
    }

    /// The raw input dispatch, upstream `handleInput`.
    pub fn handle_input(&self, data: &str) {
        if get_keybindings().matches(&self.parser, data, "tui.select.cancel") {
            (self.on_cancel)();
            return;
        }
        if self
            .keybindings
            .manager()
            .matches(&self.parser, data, "app.editor.external")
        {
            self.open_external_editor();
            return;
        }
        self.editor.handle_input(data);
    }

    /// Hand the prompt to the external editor and reload the result,
    /// upstream `handleOpenExternalEditor`.
    pub fn open_external_editor(&self) {
        let content = self.editor.get_text();
        self.tui.stop(TuiStopOptions::default());
        let result = edit_in_external_editor_blocking(&self.external_editor_command, &content);
        if let ExternalEditorResult::Complete { content } = result {
            self.editor.set_text(&content);
        }
        self.tui.start();
        self.tui.request_render(true);
    }
}

impl Component for ExtensionEditorComponent {
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

impl Focusable for ExtensionEditorComponent {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
        self.editor.set_focused(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

//! Input component, ported from
//! `packages/tui/src/components/input.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#47).
//!
//! A single-line text input with horizontal scrolling.
//!
//! Restatements against upstream:
//!
//! - Text offsets are byte offsets into the Rust `String` rather than UTF-16
//!   code units; cursors, paste-content lengths, and rendered start columns
//!   are all byte offsets.
//! - Upstream's module-global key parser behind `matchesKey` becomes a local
//!   [`crate::keys::KeyParser`]: the input's binding checks are stateless
//!   key-shape matches, and key releases never reach it (dispatch filters
//!   releases for components that do not opt in).
//! - `String.replace` with a string argument replaces only the first
//!   occurrence; the port uses the same single replacement.

use std::cell::RefCell;
use std::rc::Rc;

use crate::keybindings::get_keybindings;
use crate::keys::{KeyParser, decode_kitty_printable};
use crate::kill_ring::{KillRing, KillRingPushOptions};
use crate::tui::{
    CURSOR_MARKER, Component, Focusable, TuiMouseButton, TuiMouseEvent, TuiMouseEventResult,
    TuiMouseEventType,
};
use crate::undo_stack::UndoStack;
use crate::utils::{
    grapheme_segments, is_whitespace_char, slice_by_column, truncate_to_width, visible_width,
};
use crate::word_navigation::{find_word_backward, find_word_forward};

/// A text-decoration callback, upstream `(text: string) => string`.
pub type InputStyleFn = Rc<dyn Fn(&str) -> String>;

/// The submit/escape callback, upstream `onSubmit?` / `onEscape?`.
pub type InputCallback = Rc<dyn Fn(&str)>;

/// Undo snapshot: value plus cursor, upstream `InputState`.
#[derive(Clone)]
struct InputState {
    value: String,
    cursor: usize,
}

/// Construction options, upstream `InputOptions`.
#[derive(Default)]
pub struct InputOptions {
    /// Left-of-value prompt, upstream `prompt`; defaults to `"> "`.
    pub prompt: Option<String>,
    /// Shown when the value is empty, upstream `placeholder`.
    pub placeholder: Option<String>,
    /// Style applied to the placeholder text, upstream `placeholderStyle`;
    /// defaults to identity.
    pub placeholder_style: Option<InputStyleFn>,
}

/// The last mutation the input performed, upstream `lastAction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LastAction {
    #[default]
    None,
    Kill,
    Yank,
    TypeWord,
}

/// Single-line text input with horizontal scrolling, upstream `Input`.
pub struct Input {
    value: RefCell<String>,
    /// Cursor position in the value, in bytes.
    cursor: RefCell<usize>,
    prompt: String,
    placeholder: String,
    placeholder_style: InputStyleFn,
    rendered_start_column: std::cell::Cell<usize>,
    /// Focusable interface — set by the TUI when focus changes.
    focused: std::cell::Cell<bool>,
    /// Bracketed paste mode buffering.
    paste_buffer: RefCell<String>,
    is_in_paste: std::cell::Cell<bool>,
    /// Kill ring for Emacs-style kill/yank operations.
    kill_ring: RefCell<KillRing>,
    last_action: std::cell::Cell<LastAction>,
    /// Undo support.
    undo_stack: RefCell<UndoStack<InputState>>,
    /// Called on submit (bound key or `\n`), upstream `onSubmit`.
    pub on_submit: RefCell<Option<InputCallback>>,
    /// Called on cancel (bound key), upstream `onEscape`.
    pub on_escape: RefCell<Option<Rc<dyn Fn()>>>,
    parser: KeyParser,
}

impl Input {
    /// Upstream's default-argument constructor.
    #[must_use]
    pub fn new() -> Self {
        Self::with_options(InputOptions::default())
    }

    /// Upstream's `new Input(options)`.
    #[must_use]
    pub fn with_options(options: InputOptions) -> Self {
        Self {
            value: RefCell::new(String::new()),
            cursor: RefCell::new(0),
            prompt: options.prompt.unwrap_or_else(|| "> ".to_string()),
            placeholder: options.placeholder.unwrap_or_default(),
            placeholder_style: options
                .placeholder_style
                .unwrap_or_else(|| Rc::new(ToString::to_string)),
            rendered_start_column: std::cell::Cell::new(0),
            focused: std::cell::Cell::new(false),
            paste_buffer: RefCell::new(String::new()),
            is_in_paste: std::cell::Cell::new(false),
            kill_ring: RefCell::new(KillRing::default()),
            last_action: std::cell::Cell::new(LastAction::None),
            undo_stack: RefCell::new(UndoStack::new()),
            on_submit: RefCell::new(None),
            on_escape: RefCell::new(None),
            parser: KeyParser::new(),
        }
    }

    /// The current value, upstream `getValue`.
    #[must_use]
    pub fn get_value(&self) -> String {
        self.value.borrow().clone()
    }

    /// Replace the value and clamp the cursor to it, upstream `setValue`.
    pub fn set_value(&self, value: String) {
        let mut cursor = self.cursor.borrow_mut();
        *cursor = (*cursor).min(value.len());
        *self.value.borrow_mut() = value;
    }

    /// Feed raw terminal input to the input, upstream `handleInput`.
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's one dispatch chain branch for branch; splitting it would detach the port from the upstream method it mirrors"
    )]
    pub fn handle_input(&self, data: &str) {
        // Bracketed paste mode: start marker `\x1b[200~`, end marker
        // `\x1b[201~`.
        let mut data = data.to_string();
        if data.contains("\x1b[200~") {
            self.is_in_paste.set(true);
            *self.paste_buffer.borrow_mut() = String::new();
            data = data.replacen("\x1b[200~", "", 1);
        }

        if self.is_in_paste.get() {
            // Buffer the chunk, flushing once it carries the end marker.
            self.paste_buffer.borrow_mut().push_str(&data);

            let buffer = self.paste_buffer.borrow();
            if let Some(end_index) = buffer.find("\x1b[201~") {
                let paste_content = buffer[..end_index].to_string();
                drop(buffer);

                self.handle_paste(&paste_content);

                self.is_in_paste.set(false);

                // Handle any remaining input after the paste marker.
                let remaining = self.paste_buffer.borrow()[end_index + 6..].to_string();
                *self.paste_buffer.borrow_mut() = String::new();
                if !remaining.is_empty() {
                    self.handle_input(&remaining);
                }
            }
            return;
        }

        let matches = |action: &str| get_keybindings().matches(&self.parser, &data, action);

        // Escape/Cancel
        if matches("tui.select.cancel") {
            if let Some(on_escape) = self.on_escape.borrow().as_ref() {
                on_escape();
            }
            return;
        }

        // Undo
        if matches("tui.editor.undo") {
            self.undo();
            return;
        }

        // Submit
        if matches("tui.input.submit") || data == "\n" {
            if let Some(on_submit) = self.on_submit.borrow().as_ref() {
                let value = self.value.borrow().clone();
                on_submit(&value);
            }
            return;
        }

        // Deletion
        if matches("tui.editor.deleteCharBackward") {
            self.handle_backspace();
            return;
        }

        if matches("tui.editor.deleteCharForward") {
            self.handle_forward_delete();
            return;
        }

        if matches("tui.editor.deleteWordBackward") {
            self.delete_word_backwards();
            return;
        }

        if matches("tui.editor.deleteWordForward") {
            self.delete_word_forward();
            return;
        }

        if matches("tui.editor.deleteToLineStart") {
            self.delete_to_line_start();
            return;
        }

        if matches("tui.editor.deleteToLineEnd") {
            self.delete_to_line_end();
            return;
        }

        // Kill ring actions
        if matches("tui.editor.yank") {
            self.yank();
            return;
        }
        if matches("tui.editor.yankPop") {
            self.yank_pop();
            return;
        }

        // Cursor movement
        if matches("tui.editor.cursorLeft") {
            self.last_action.set(LastAction::None);
            if *self.cursor.borrow() > 0 {
                let before_cursor = self.value.borrow()[..*self.cursor.borrow()].to_string();
                let last_grapheme = grapheme_segments(&before_cursor).next_back();
                let step = last_grapheme.map_or(1, str::len);
                *self.cursor.borrow_mut() -= step;
            }
            return;
        }

        if matches("tui.editor.cursorRight") {
            self.last_action.set(LastAction::None);
            let value_len = self.value.borrow().len();
            if *self.cursor.borrow() < value_len {
                let after_cursor = self.value.borrow()[*self.cursor.borrow()..].to_string();
                let first_grapheme = grapheme_segments(&after_cursor).next();
                let step = first_grapheme.map_or(1, str::len);
                *self.cursor.borrow_mut() += step;
            }
            return;
        }

        if matches("tui.editor.cursorLineStart") {
            self.last_action.set(LastAction::None);
            *self.cursor.borrow_mut() = 0;
            return;
        }

        if matches("tui.editor.cursorLineEnd") {
            self.last_action.set(LastAction::None);
            *self.cursor.borrow_mut() = self.value.borrow().len();
            return;
        }

        if matches("tui.editor.cursorWordLeft") {
            self.move_word_backwards();
            return;
        }

        if matches("tui.editor.cursorWordRight") {
            self.move_word_forwards();
            return;
        }

        // Kitty CSI-u printable character (e.g. \x1b[97u for 'a').
        // Terminals with Kitty protocol flag 1 (disambiguate) send CSI-u for
        // all keys, including plain printable characters. Decode before the
        // control-char check since CSI-u sequences contain \x1b which would
        // be rejected.
        if let Some(kitty_printable) = decode_kitty_printable(&data) {
            self.insert_character(&kitty_printable);
            return;
        }

        // Regular character input - accept printable characters including
        // Unicode, but reject control characters (C0: 0x00-0x1F, DEL: 0x7F,
        // C1: 0x80-0x9F)
        let has_control_chars = data.chars().any(|ch| {
            let code = u32::from(ch);
            code < 32 || code == 0x7f || (0x80..=0x9f).contains(&code)
        });
        if !has_control_chars {
            self.insert_character(&data);
        }
    }

    /// Mouse cursor placement on the input row, upstream `handleMouse`.
    fn handle_mouse_impl(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        if event.event_type != TuiMouseEventType::Press
            || event.button != TuiMouseButton::Left
            || event.y != 0
        {
            return None;
        }
        let visible_column = usize::from(event.x).saturating_sub(2);
        let target_column = self.rendered_start_column.get() + visible_column;
        let value = self.value.borrow().clone();
        let mut current_column = 0;
        let mut offset = 0;
        let mut new_cursor = value.len();
        for grapheme in grapheme_segments(&value) {
            let next_column = current_column + visible_width(grapheme);
            if target_column < next_column {
                new_cursor = offset;
                break;
            }
            current_column = next_column;
            offset += grapheme.len();
        }
        *self.cursor.borrow_mut() = new_cursor;
        self.last_action.set(LastAction::None);
        Some(TuiMouseEventResult {
            handled: true,
            focus: true,
            ..TuiMouseEventResult::default()
        })
    }

    fn insert_character(&self, text: &str) {
        // Undo coalescing: consecutive word chars coalesce into one undo unit
        if text.chars().any(is_whitespace_char) || self.last_action.get() != LastAction::TypeWord {
            self.push_undo();
        }
        self.last_action.set(LastAction::TypeWord);

        let mut value = self.value.borrow_mut();
        let mut cursor = self.cursor.borrow_mut();
        value.insert_str(*cursor, text);
        *cursor += text.len();
    }

    fn handle_backspace(&self) {
        self.last_action.set(LastAction::None);
        if *self.cursor.borrow() == 0 {
            return;
        }
        self.push_undo();
        let mut value = self.value.borrow_mut();
        let cursor = *self.cursor.borrow();
        let before_cursor = value[..cursor].to_string();
        let last_grapheme = grapheme_segments(&before_cursor).next_back();
        let grapheme_length = last_grapheme.map_or(1, str::len);
        value.replace_range(cursor - grapheme_length..cursor, "");
        *self.cursor.borrow_mut() = cursor - grapheme_length;
    }

    fn handle_forward_delete(&self) {
        self.last_action.set(LastAction::None);
        if *self.cursor.borrow() >= self.value.borrow().len() {
            return;
        }
        self.push_undo();
        let mut value = self.value.borrow_mut();
        let cursor = *self.cursor.borrow();
        let value_len = value.len();
        let after_cursor = value[cursor..].to_string();
        let first_grapheme = grapheme_segments(&after_cursor).next();
        let grapheme_length = first_grapheme.map_or(1, str::len);
        value.replace_range(cursor..(cursor + grapheme_length).min(value_len), "");
    }

    fn delete_to_line_start(&self) {
        let cursor = *self.cursor.borrow();
        if cursor == 0 {
            return;
        }
        self.push_undo();
        let mut value = self.value.borrow_mut();
        let deleted_text = value[..cursor].to_string();
        self.kill_ring.borrow_mut().push(
            &deleted_text,
            KillRingPushOptions {
                prepend: true,
                accumulate: self.last_action.get() == LastAction::Kill,
            },
        );
        self.last_action.set(LastAction::Kill);
        *value = value[cursor..].to_string();
        *self.cursor.borrow_mut() = 0;
    }

    fn delete_to_line_end(&self) {
        let value_len = self.value.borrow().len();
        let cursor = *self.cursor.borrow();
        if cursor >= value_len {
            return;
        }
        self.push_undo();
        let mut value = self.value.borrow_mut();
        let deleted_text = value[cursor..].to_string();
        self.kill_ring.borrow_mut().push(
            &deleted_text,
            KillRingPushOptions {
                prepend: false,
                accumulate: self.last_action.get() == LastAction::Kill,
            },
        );
        self.last_action.set(LastAction::Kill);
        *value = value[..cursor].to_string();
    }

    fn delete_word_backwards(&self) {
        if *self.cursor.borrow() == 0 {
            return;
        }

        // Save lastAction before cursor movement (move_word_backwards resets it)
        let was_kill = self.last_action.get() == LastAction::Kill;

        self.push_undo();

        let old_cursor = *self.cursor.borrow();
        self.move_word_backwards();
        let delete_from = *self.cursor.borrow();
        *self.cursor.borrow_mut() = old_cursor;

        let mut value = self.value.borrow_mut();
        let deleted_text = value[delete_from..*self.cursor.borrow()].to_string();
        self.kill_ring.borrow_mut().push(
            &deleted_text,
            KillRingPushOptions {
                prepend: true,
                accumulate: was_kill,
            },
        );
        self.last_action.set(LastAction::Kill);

        let tail = value[*self.cursor.borrow()..].to_string();
        value.truncate(delete_from);
        value.push_str(&tail);
        *self.cursor.borrow_mut() = delete_from;
    }

    fn delete_word_forward(&self) {
        let value_len = self.value.borrow().len();
        if *self.cursor.borrow() >= value_len {
            return;
        }

        // Save lastAction before cursor movement (move_word_forwards resets it)
        let was_kill = self.last_action.get() == LastAction::Kill;

        self.push_undo();

        let old_cursor = *self.cursor.borrow();
        self.move_word_forwards();
        let delete_to = *self.cursor.borrow();
        *self.cursor.borrow_mut() = old_cursor;

        let mut value = self.value.borrow_mut();
        let deleted_text = value[*self.cursor.borrow()..delete_to].to_string();
        self.kill_ring.borrow_mut().push(
            &deleted_text,
            KillRingPushOptions {
                prepend: false,
                accumulate: was_kill,
            },
        );
        self.last_action.set(LastAction::Kill);

        let tail = value[delete_to..].to_string();
        value.truncate(*self.cursor.borrow());
        value.push_str(&tail);
    }

    fn yank(&self) {
        let Some(text) = self.kill_ring.borrow().peek().map(str::to_string) else {
            return;
        };

        self.push_undo();

        let mut value = self.value.borrow_mut();
        let mut cursor = self.cursor.borrow_mut();
        value.insert_str(*cursor, &text);
        *cursor += text.len();
        self.last_action.set(LastAction::Yank);
    }

    fn yank_pop(&self) {
        if self.last_action.get() != LastAction::Yank || self.kill_ring.borrow().len() <= 1 {
            return;
        }

        self.push_undo();

        // Delete the previously yanked text (still at end of ring before
        // rotation)
        let prev_text = self
            .kill_ring
            .borrow()
            .peek()
            .map_or_else(String::new, str::to_string);
        {
            let mut value = self.value.borrow_mut();
            let mut cursor = self.cursor.borrow_mut();
            value.replace_range(*cursor - prev_text.len()..*cursor, "");
            *cursor -= prev_text.len();
        }

        // Rotate and insert new entry
        self.kill_ring.borrow_mut().rotate();
        let text = self
            .kill_ring
            .borrow()
            .peek()
            .map_or_else(String::new, str::to_string);
        let mut value = self.value.borrow_mut();
        let mut cursor = self.cursor.borrow_mut();
        value.insert_str(*cursor, &text);
        *cursor += text.len();
        self.last_action.set(LastAction::Yank);
    }

    fn push_undo(&self) {
        self.undo_stack.borrow_mut().push(&InputState {
            value: self.value.borrow().clone(),
            cursor: *self.cursor.borrow(),
        });
    }

    fn undo(&self) {
        let Some(snapshot) = self.undo_stack.borrow_mut().pop() else {
            return;
        };
        *self.value.borrow_mut() = snapshot.value;
        *self.cursor.borrow_mut() = snapshot.cursor;
        self.last_action.set(LastAction::None);
    }

    fn move_word_backwards(&self) {
        if *self.cursor.borrow() == 0 {
            return;
        }
        self.last_action.set(LastAction::None);
        let new_cursor = find_word_backward(&self.value.borrow(), *self.cursor.borrow(), None);
        *self.cursor.borrow_mut() = new_cursor;
    }

    fn move_word_forwards(&self) {
        let value_len = self.value.borrow().len();
        if *self.cursor.borrow() >= value_len {
            return;
        }
        self.last_action.set(LastAction::None);
        let new_cursor = find_word_forward(&self.value.borrow(), *self.cursor.borrow(), None);
        *self.cursor.borrow_mut() = new_cursor;
    }

    fn handle_paste(&self, pasted_text: &str) {
        self.last_action.set(LastAction::None);
        self.push_undo();

        // Clean the pasted text - remove newlines and carriage returns
        let clean_text = pasted_text
            .replace("\r\n", "")
            .replace(['\r', '\n'], "")
            .replace('\t', "    ");

        // Insert at cursor position
        let mut value = self.value.borrow_mut();
        let mut cursor = self.cursor.borrow_mut();
        value.insert_str(*cursor, &clean_text);
        *cursor += clean_text.len();
    }

    /// The prompt-and-value line, upstream `Input.render`: the inverse-video
    /// fake cursor at the grapheme the cursor sits on, the hardware-cursor
    /// marker ahead of it when focused, and the horizontal-scroll window
    /// when the value overflows the available width.
    fn render_impl(&self, width: usize) -> Vec<String> {
        let value = self.value.borrow().clone();
        let cursor = (*self.cursor.borrow()).min(value.len());
        let available_width = width.saturating_sub(visible_width(&self.prompt));

        if available_width == 0 {
            return vec![truncate_to_width(&self.prompt, width, "", false)];
        }

        if value.is_empty() && !self.placeholder.is_empty() {
            let placeholder = truncate_to_width(&self.placeholder, available_width, "", false);
            let at_cursor = grapheme_segments(&placeholder).next().unwrap_or(" ");
            let after_cursor = &placeholder[at_cursor.len()..];
            let marker = if self.focused.get() {
                CURSOR_MARKER
            } else {
                ""
            };
            let cursor_char = format!(
                "\x1b[7m{}{}\x1b[27m",
                (self.placeholder_style)(at_cursor),
                (self.placeholder_style)(after_cursor)
            );
            let text_with_cursor = format!("{marker}{cursor_char}");
            let padding =
                " ".repeat(available_width.saturating_sub(visible_width(&text_with_cursor)));
            return vec![format!("{}{text_with_cursor}{padding}", self.prompt)];
        }

        // The horizontal-scroll window, upstream's over-long branch:
        // reserve one column for the cursor when it sits at the end, then
        // center the cursor column in the visible window.
        let total_width = visible_width(&value);
        let visible_text;
        let mut cursor_display = cursor;
        self.rendered_start_column.set(0);
        if total_width < available_width {
            // Everything fits (leave room for cursor at end)
            visible_text = value;
        } else {
            // Reserve one column for the cursor if it's at the end
            let scroll_width = if cursor == value.len() {
                available_width - 1
            } else {
                available_width
            };
            let cursor_col = visible_width(&value[..cursor]);

            if scroll_width > 0 {
                let half_width = scroll_width / 2;
                let start_col = if cursor_col < half_width {
                    // Cursor near start
                    0
                } else if cursor_col > total_width.saturating_sub(half_width) {
                    // Cursor near end
                    total_width.saturating_sub(scroll_width)
                } else {
                    // Cursor in middle
                    cursor_col.saturating_sub(half_width)
                };

                self.rendered_start_column.set(start_col);
                visible_text = slice_by_column(&value, start_col, scroll_width, true);
                let before_cursor = slice_by_column(
                    &value,
                    start_col,
                    cursor_col.saturating_sub(start_col),
                    true,
                );
                cursor_display = before_cursor.len();
            } else {
                cursor_display = 0;
                visible_text = String::new();
            }
        }

        // Build line with fake cursor: insert the inverse-video cursor at the
        // cursor position.
        let after = &visible_text[cursor_display.min(visible_text.len())..];
        let at_cursor = grapheme_segments(after).next().unwrap_or(" ");
        let before_cursor = &visible_text[..cursor_display.min(visible_text.len())];
        let after_cursor = after.get(at_cursor.len()..).unwrap_or_default();

        // Hardware cursor marker (zero-width, emitted before fake cursor for
        // IME positioning)
        let marker = if self.focused.get() {
            CURSOR_MARKER
        } else {
            ""
        };

        // Inverse video shows the cursor: ESC[7m = reverse, ESC[27m = normal
        let cursor_char = format!("\x1b[7m{at_cursor}\x1b[27m");
        let text_with_cursor = format!("{before_cursor}{marker}{cursor_char}{after_cursor}");

        let visual_length = visible_width(&text_with_cursor);
        let padding = " ".repeat(available_width.saturating_sub(visual_length));
        vec![format!("{}{text_with_cursor}{padding}", self.prompt)]
    }
}

impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for InputOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InputOptions")
            .field("prompt", &self.prompt)
            .field("placeholder", &self.placeholder)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Input")
            .field("value", &self.value.borrow())
            .field("cursor", &self.cursor.borrow())
            .field("prompt", &self.prompt)
            .finish_non_exhaustive()
    }
}

impl Component for Input {
    fn render(&self, width: usize) -> Vec<String> {
        self.render_impl(width)
    }

    fn handle_input(&self, data: &str) {
        self.handle_input(data);
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.handle_mouse_impl(event)
    }

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }
}

impl Focusable for Input {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

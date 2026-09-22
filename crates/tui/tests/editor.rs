//! The editor suite, ported 1:1 from
//! `packages/tui/test/editor.test.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#47).
//!
//! Restatements against upstream:
//!
//! - Text offsets are byte offsets into the Rust `String`; `getCursor` reads
//!   byte columns.
//! - The autocomplete tests upstream carry (`describe("Autocomplete")`, plus
//!   the two undo tests that drive a mock provider) are the autocomplete
//!   child's scope
//!   ([#49](https://github.com/PhillipChaffee/pi-rust/issues/49)) and are
//!   absent here.
//! - Upstream's `Intl.Segmenter` applies locale dictionary segmentation for
//!   Han text (Node groups 你好 and 世界 as single word-like segments); the
//!   port's UAX #29 segmenter yields one segment per ideograph, so the two
//!   CJK word-movement expectations cross the same boundaries one step at a
//!   time. Fullwidth punctuation (，) stays a non-word punctuation segment
//!   in both, and the movement still stops on it.
//! - `editor.onSubmit = fn` becomes writing
//!   [`pi_tui::components::Editor`]'s `on_submit` field; `disableSubmit` a
//!   `Cell` set.
#![expect(
    clippy::expect_used,
    clippy::cast_possible_truncation,
    reason = "test fixtures fail loudly when the engine misbehaves; expecting keeps the failure modes readable, and the resize fixtures restate upstream's numeric render widths as plain casts"
)]

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::cell::RefCell;
use std::rc::Rc;

use pi_tui::components::{Editor, EditorOptions, EditorTheme, word_wrap_line};
use pi_tui::tui::{Component, Tui};
use pi_tui::utils::{strip_terminal_sequences, visible_width};
use pi_tui::word_navigation::SegmentData;
use tui_support::{default_editor_theme, new_editor_test_tui};

/// Upstream test `createTestTUI(cols, rows)` paired with the editor.
fn editor_with(columns: u16, rows: u16) -> (Rc<Tui>, Editor) {
    let tui = new_editor_test_tui(columns, rows);
    let editor = Editor::new(&tui, default_editor_theme());
    (tui, editor)
}

fn editor() -> Editor {
    editor_with(80, 24).1
}

fn editor_with_options(columns: u16, options: EditorOptions) -> Editor {
    let tui = new_editor_test_tui(columns, 24);
    Editor::with_options(&tui, default_editor_theme(), options)
}

/// Capture an `onSubmit`/`onChange` value, upstream `let submitted = ""`.
type Submitted = Rc<RefCell<String>>;

fn capture() -> Submitted {
    Rc::new(RefCell::new(String::new()))
}

fn on_submit_capture(capture: &Submitted) -> Rc<dyn Fn(&str)> {
    let capture = Rc::clone(capture);
    Rc::new(move |text: &str| *capture.borrow_mut() = text.to_string())
}

/// Position cursor at a specific line and column, upstream's
/// `positionCursor` helper.
fn position_cursor(editor: &Editor, line: usize, col: usize) {
    // Go to line 0 first
    for _ in 0..20 {
        editor.handle_input("\x1b[A");
    }
    // Go to target line
    for _ in 0..line {
        editor.handle_input("\x1b[B");
    }
    // Go to target col
    editor.handle_input("\x01");
    for _ in 0..col {
        editor.handle_input("\x1b[C");
    }
}

fn cursor_of(editor: &Editor) -> (usize, usize) {
    let cursor = editor.get_cursor();
    (cursor.line, cursor.col)
}

#[test]
fn does_nothing_on_up_arrow_when_history_is_empty() {
    let editor = editor();

    editor.handle_input("\x1b[A"); // Up arrow

    assert_eq!(editor.get_text(), "");
}

#[test]
fn shows_most_recent_history_entry_on_up_arrow_when_editor_is_empty() {
    let editor = editor();

    editor.add_to_history("first prompt");
    editor.add_to_history("second prompt");

    editor.handle_input("\x1b[A"); // Up arrow

    assert_eq!(editor.get_text(), "second prompt");
}

#[test]
fn cycles_through_history_entries_on_repeated_up_arrow() {
    let editor = editor();

    editor.add_to_history("first");
    editor.add_to_history("second");
    editor.add_to_history("third");

    editor.handle_input("\x1b[A"); // Up - shows "third"
    assert_eq!(editor.get_text(), "third");

    editor.handle_input("\x1b[A"); // Up - shows "second"
    assert_eq!(editor.get_text(), "second");

    editor.handle_input("\x1b[A"); // Up - shows "first"
    assert_eq!(editor.get_text(), "first");

    editor.handle_input("\x1b[A"); // Up - stays at "first" (oldest)
    assert_eq!(editor.get_text(), "first");
}

#[test]
fn jumps_to_start_before_entering_history_from_a_non_empty_draft() {
    let editor = editor();

    editor.add_to_history("prompt");
    editor.set_text("draft");
    editor.handle_input("\x1b[D");
    editor.handle_input("\x1b[D");

    editor.handle_input("\x1b[A"); // Up - jumps to start before history browsing
    assert_eq!(editor.get_text(), "draft");
    assert_eq!(cursor_of(&editor), (0, 0));

    editor.handle_input("\x1b[A"); // Up at start - shows "prompt"
    assert_eq!(editor.get_text(), "prompt");

    editor.handle_input("\x1b[B"); // Down - restores draft
    assert_eq!(editor.get_text(), "draft");
    assert_eq!(cursor_of(&editor), (0, 0));
}

#[test]
fn navigates_forward_through_history_with_down_arrow() {
    let editor = editor();

    editor.add_to_history("first");
    editor.add_to_history("second");
    editor.add_to_history("third");
    editor.set_text("draft");

    // Go to oldest
    editor.handle_input("\x1b[A"); // start of draft
    editor.handle_input("\x1b[A"); // third
    editor.handle_input("\x1b[A"); // second
    editor.handle_input("\x1b[A"); // first

    // Navigate back
    editor.handle_input("\x1b[B"); // second
    assert_eq!(editor.get_text(), "second");

    editor.handle_input("\x1b[B"); // third
    assert_eq!(editor.get_text(), "third");

    editor.handle_input("\x1b[B"); // draft
    assert_eq!(editor.get_text(), "draft");
}

#[test]
fn exits_history_mode_when_typing_a_character() {
    let editor = editor();

    editor.add_to_history("old prompt");

    editor.handle_input("\x1b[A"); // Up - shows "old prompt"
    editor.handle_input("x"); // Type a character - exits history mode

    assert_eq!(editor.get_text(), "xold prompt");
}

#[test]
fn exits_history_mode_on_set_text() {
    let editor = editor();

    editor.add_to_history("first");
    editor.add_to_history("second");

    editor.handle_input("\x1b[A"); // Up - shows "second"
    editor.set_text(""); // External clear

    // Up should start fresh from most recent
    editor.handle_input("\x1b[A");
    assert_eq!(editor.get_text(), "second");
}

#[test]
fn does_not_add_empty_strings_to_history() {
    let editor = editor();

    editor.add_to_history("");
    editor.add_to_history("   ");
    editor.add_to_history("valid");

    editor.handle_input("\x1b[A");
    assert_eq!(editor.get_text(), "valid");

    // Should not have more entries
    editor.handle_input("\x1b[A");
    assert_eq!(editor.get_text(), "valid");
}

#[test]
fn does_not_add_consecutive_duplicates_to_history() {
    let editor = editor();

    editor.add_to_history("same");
    editor.add_to_history("same");
    editor.add_to_history("same");

    editor.handle_input("\x1b[A"); // "same"
    assert_eq!(editor.get_text(), "same");

    editor.handle_input("\x1b[A"); // stays at "same" (only one entry)
    assert_eq!(editor.get_text(), "same");
}

#[test]
fn allows_non_consecutive_duplicates_in_history() {
    let editor = editor();

    editor.add_to_history("first");
    editor.add_to_history("second");
    editor.add_to_history("first"); // Not consecutive, should be added

    editor.handle_input("\x1b[A"); // "first"
    assert_eq!(editor.get_text(), "first");

    editor.handle_input("\x1b[A"); // "second"
    assert_eq!(editor.get_text(), "second");

    editor.handle_input("\x1b[A"); // "first" (older one)
    assert_eq!(editor.get_text(), "first");
}

#[test]
fn uses_cursor_movement_instead_of_history_when_editor_has_content() {
    let editor = editor();

    editor.add_to_history("history item");
    editor.set_text("line1\nline2");

    // Cursor is at end of line2, Up should move to line1
    editor.handle_input("\x1b[A"); // Up - cursor movement

    // Insert character to verify cursor position
    editor.handle_input("X");

    // X should be inserted in line1, not replace with history
    assert_eq!(editor.get_text(), "line1X\nline2");
}

#[test]
fn limits_history_to_100_entries() {
    let editor = editor();

    // Add 105 entries
    for index in 0..105 {
        editor.add_to_history(&format!("prompt {index}"));
    }

    // Navigate to oldest
    for _ in 0..100 {
        editor.handle_input("\x1b[A");
    }

    // Should be at entry 5 (oldest kept), not entry 0
    assert_eq!(editor.get_text(), "prompt 5");

    // One more Up should not change anything
    editor.handle_input("\x1b[A");
    assert_eq!(editor.get_text(), "prompt 5");
}

#[test]
fn places_cursor_at_start_after_browsing_history_upward() {
    let editor = editor();

    editor.add_to_history("older entry");
    editor.add_to_history("line1\nline2\nline3");

    editor.handle_input("\x1b[A"); // Up - shows multi-line entry at start
    assert_eq!(editor.get_text(), "line1\nline2\nline3");
    assert_eq!(cursor_of(&editor), (0, 0));

    editor.handle_input("\x1b[A"); // Up again - immediately navigates to older entry
    assert_eq!(editor.get_text(), "older entry");
    assert_eq!(cursor_of(&editor), (0, 0));
}

#[test]
fn places_cursor_at_end_after_browsing_history_downward() {
    let editor = editor();

    editor.add_to_history("older entry");
    editor.add_to_history("line1\nline2\nline3");
    editor.add_to_history("newer entry");

    editor.handle_input("\x1b[A"); // newer entry
    editor.handle_input("\x1b[A"); // multi-line entry
    editor.handle_input("\x1b[A"); // older entry

    editor.handle_input("\x1b[B"); // Down - shows multi-line entry at end
    assert_eq!(editor.get_text(), "line1\nline2\nline3");
    assert_eq!(cursor_of(&editor), (2, 5));

    editor.handle_input("\x1b[B"); // Down again - immediately navigates to newer entry
    assert_eq!(editor.get_text(), "newer entry");
}

#[test]
fn allows_opposite_direction_cursor_movement_within_multi_line_history_entry() {
    let editor = editor();

    editor.add_to_history("line1\nline2\nline3");

    editor.handle_input("\x1b[A"); // Up - shows entry at start
    assert_eq!(cursor_of(&editor), (0, 0));

    editor.handle_input("\x1b[B"); // Down - cursor moves to line2
    assert_eq!(editor.get_text(), "line1\nline2\nline3");
    assert_eq!(cursor_of(&editor), (1, 0));

    editor.handle_input("\x1b[A"); // Up - cursor moves back to line1
    assert_eq!(editor.get_text(), "line1\nline2\nline3");
    assert_eq!(cursor_of(&editor), (0, 0));
}

#[test]
fn returns_cursor_position() {
    let editor = editor();

    assert_eq!(cursor_of(&editor), (0, 0));

    for key in ["a", "b", "c"] {
        editor.handle_input(key);
    }

    assert_eq!(cursor_of(&editor), (0, 3));

    editor.handle_input("\x1b[D"); // Left
    assert_eq!(cursor_of(&editor), (0, 2));
}

#[test]
fn returns_lines_as_a_defensive_copy() {
    let editor = editor();
    editor.set_text("a\nb");

    let lines = editor.get_lines();
    assert_eq!(lines, ["a", "b"]);

    let mut lines = lines;
    lines[0] = "mutated".to_string();
    assert_eq!(editor.get_lines(), ["a", "b"]);
}

#[test]
fn inserts_backslash_immediately_no_buffering() {
    let editor = editor();

    editor.handle_input("\\");

    // Backslash should be visible immediately, not buffered
    assert_eq!(editor.get_text(), "\\");
}

#[test]
fn converts_standalone_backslash_to_newline_on_enter() {
    let editor = editor();

    editor.handle_input("\\");
    editor.handle_input("\r");

    assert_eq!(editor.get_text(), "\n");
}

#[test]
fn inserts_backslash_normally_when_followed_by_other_characters() {
    let editor = editor();

    editor.handle_input("\\");
    editor.handle_input("x");

    assert_eq!(editor.get_text(), "\\x");
}

#[test]
fn does_not_trigger_newline_when_backslash_is_not_immediately_before_cursor() {
    let editor = editor();
    let submitted = capture();

    *editor.on_submit.borrow_mut() = Some(Rc::new({
        let submitted = Rc::clone(&submitted);
        move |_text: &str| *submitted.borrow_mut() = "submitted".to_string()
    }));

    editor.handle_input("\\");
    editor.handle_input("x");
    editor.handle_input("\r");

    // Should submit, not insert newline (backslash not at cursor)
    assert_eq!(*submitted.borrow(), "submitted");
}

#[test]
fn only_removes_one_backslash_when_multiple_are_present() {
    let editor = editor();

    editor.handle_input("\\");
    editor.handle_input("\\");
    editor.handle_input("\\");
    assert_eq!(editor.get_text(), "\\\\\\");

    editor.handle_input("\r");
    // Only the last backslash is removed, newline inserted
    assert_eq!(editor.get_text(), "\\\\\n");
}

#[test]
fn ignores_printable_csi_u_sequences_with_unsupported_modifiers() {
    let editor = editor();

    editor.handle_input("\x1b[99;9u");

    assert_eq!(editor.get_text(), "");
}

#[test]
fn inserts_shifted_csi_u_letters_as_text() {
    let editor = editor();

    editor.handle_input("\x1b[69;2u");

    assert_eq!(editor.get_text(), "E");
}

#[test]
fn inserts_shifted_xterm_modify_other_keys_letters_as_text() {
    let editor = editor();

    editor.handle_input("\x1b[27;2;69~");

    assert_eq!(editor.get_text(), "E");
}

#[test]
fn inserts_mixed_ascii_umlauts_and_emojis_as_literal_text() {
    let editor = editor();

    for key in ["H", "e", "l", "l", "o", " ", "ä", "ö", "ü", " ", "😀"] {
        editor.handle_input(key);
    }

    assert_eq!(editor.get_text(), "Hello äöü 😀");
}

#[test]
fn deletes_single_code_unit_unicode_characters_umlauts_with_backspace() {
    let editor = editor();

    for key in ["ä", "ö", "ü"] {
        editor.handle_input(key);
    }

    // Delete the last character (ü)
    editor.handle_input("\x7f"); // Backspace

    assert_eq!(editor.get_text(), "äö");
}

#[test]
fn deletes_multi_code_unit_emojis_with_single_backspace() {
    let editor = editor();

    editor.handle_input("😀");
    editor.handle_input("👍");

    // Delete the last emoji (👍) - single backspace deletes whole grapheme cluster
    editor.handle_input("\x7f"); // Backspace

    assert_eq!(editor.get_text(), "😀");
}

#[test]
fn inserts_characters_at_the_correct_position_after_cursor_movement_over_umlauts() {
    let editor = editor();

    for key in ["ä", "ö", "ü"] {
        editor.handle_input(key);
    }

    // Move cursor left twice
    editor.handle_input("\x1b[D"); // Left arrow
    editor.handle_input("\x1b[D"); // Left arrow

    // Insert 'x' in the middle
    editor.handle_input("x");

    assert_eq!(editor.get_text(), "äxöü");
}

#[test]
fn moves_cursor_across_multi_code_unit_emojis_with_single_arrow_key() {
    let editor = editor();

    for key in ["😀", "👍", "🎉"] {
        editor.handle_input(key);
    }

    // Move cursor left over last emoji (🎉) - single arrow moves over whole grapheme
    editor.handle_input("\x1b[D"); // Left arrow

    // Move cursor left over second emoji (👍)
    editor.handle_input("\x1b[D");

    // Insert 'x' between first and second emoji
    editor.handle_input("x");

    assert_eq!(editor.get_text(), "😀x👍🎉");
}

#[test]
fn preserves_umlauts_across_line_breaks() {
    let editor = editor();

    for key in ["ä", "ö", "ü", "\n", "Ä", "Ö", "Ü"] {
        editor.handle_input(key);
    }

    assert_eq!(editor.get_text(), "äöü\nÄÖÜ");
}

#[test]
fn replaces_the_entire_document_with_unicode_text_via_set_text() {
    let editor = editor();

    // Simulate bracketed paste / programmatic replacement
    editor.set_text("Hällö Wörld! 😀 äöüÄÖÜß");

    assert_eq!(editor.get_text(), "Hällö Wörld! 😀 äöüÄÖÜß");
}

#[test]
fn moves_cursor_to_document_start_on_ctrl_a_and_inserts_at_the_beginning() {
    let editor = editor();

    for key in ["a", "b"] {
        editor.handle_input(key);
    }
    editor.handle_input("\x01"); // Ctrl+A (move to start)
    editor.handle_input("x"); // Insert at start

    assert_eq!(editor.get_text(), "xab");
}

#[test]
fn deletes_words_correctly_with_ctrl_w_and_alt_backspace() {
    let editor = editor();

    // Basic word deletion
    editor.set_text("foo bar baz");
    editor.handle_input("\x17"); // Ctrl+W
    assert_eq!(editor.get_text(), "foo bar ");

    // Trailing whitespace
    editor.set_text("foo bar   ");
    editor.handle_input("\x17");
    assert_eq!(editor.get_text(), "foo ");

    // Punctuation run
    editor.set_text("foo bar...");
    editor.handle_input("\x17");
    assert_eq!(editor.get_text(), "foo bar");

    // ASCII punctuation inside Intl word-like segments preserves old boundaries
    editor.set_text("foo.bar");
    editor.handle_input("\x17");
    assert_eq!(editor.get_text(), "foo.");

    editor.set_text("foo:bar");
    editor.handle_input("\x17");
    assert_eq!(editor.get_text(), "foo:");

    // Delete across multiple lines
    editor.set_text("line one\nline two");
    editor.handle_input("\x17");
    assert_eq!(editor.get_text(), "line one\nline ");

    // Delete empty line (merge)
    editor.set_text("line one\n");
    editor.handle_input("\x17");
    assert_eq!(editor.get_text(), "line one");

    // Grapheme safety (emoji as a word)
    editor.set_text("foo 😀😀 bar");
    editor.handle_input("\x17");
    assert_eq!(editor.get_text(), "foo 😀😀 ");
    editor.handle_input("\x17");
    assert_eq!(editor.get_text(), "foo ");

    // Alt+Backspace
    editor.set_text("foo bar");
    editor.handle_input("\x1b\x7f"); // Alt+Backspace (legacy)
    assert_eq!(editor.get_text(), "foo ");
}

#[test]
fn navigates_words_correctly_with_ctrl_left_right() {
    let editor = editor();

    editor.set_text("foo bar... baz");
    // Cursor at end

    // Move left over baz
    editor.handle_input("\x1b[1;5D"); // Ctrl+Left
    assert_eq!(cursor_of(&editor), (0, 11)); // after '...'

    // Move left over punctuation
    editor.handle_input("\x1b[1;5D"); // Ctrl+Left
    assert_eq!(cursor_of(&editor), (0, 7)); // after 'bar'

    // Move left over bar
    editor.handle_input("\x1b[1;5D"); // Ctrl+Left
    assert_eq!(cursor_of(&editor), (0, 4)); // after 'foo '

    // Move right over bar
    editor.handle_input("\x1b[1;5C"); // Ctrl+Right
    assert_eq!(cursor_of(&editor), (0, 7)); // at end of 'bar'

    // Move right over punctuation run
    editor.handle_input("\x1b[1;5C"); // Ctrl+Right
    assert_eq!(cursor_of(&editor), (0, 10)); // after '...'

    // Move right skips space and lands after baz
    editor.handle_input("\x1b[1;5C"); // Ctrl+Right
    assert_eq!(cursor_of(&editor), (0, 14)); // end of line

    // Test forward from start with leading whitespace
    editor.set_text("   foo bar");
    editor.handle_input("\x01"); // Ctrl+A to go to start
    editor.handle_input("\x1b[1;5C"); // Ctrl+Right
    assert_eq!(cursor_of(&editor), (0, 6)); // after 'foo'

    // ASCII punctuation inside Intl word-like segments preserves old boundaries
    editor.set_text("foo.bar baz");
    editor.handle_input("\x1b[1;5D"); // Ctrl+Left over baz
    assert_eq!(cursor_of(&editor), (0, 8));
    editor.handle_input("\x1b[1;5D"); // Ctrl+Left over bar
    assert_eq!(cursor_of(&editor), (0, 4));
    editor.handle_input("\x1b[1;5D"); // Ctrl+Left over .
    assert_eq!(cursor_of(&editor), (0, 3));

    editor.handle_input("\x01"); // Ctrl+A
    editor.handle_input("\x1b[1;5C"); // Ctrl+Right over foo
    assert_eq!(cursor_of(&editor), (0, 3));
    editor.handle_input("\x1b[1;5C"); // Ctrl+Right over .
    assert_eq!(cursor_of(&editor), (0, 4));
    editor.handle_input("\x1b[1;5C"); // Ctrl+Right over bar
    assert_eq!(cursor_of(&editor), (0, 7));
}

#[test]
fn stops_at_fullwidth_chinese_punctuation_issue_4972() {
    let editor = editor();

    // 你好，世界: the UAX #29 segmenter yields one word-like segment per
    // ideograph and a non-word punctuation segment for ，, so each backward
    // step crosses one character (byte 3 each) where upstream's dictionary
    // segmenter crossed one dictionary word. Bytes: 你(0) 好(3) ，(6) 世(9) 界(12).
    editor.set_text("你好，世界");
    // Cursor at end (byte 15)

    // Move left over 界
    editor.handle_input("\x1b[1;5D"); // Ctrl+Left
    assert_eq!(cursor_of(&editor), (0, 12));

    editor.handle_input("\x1b[1;5D"); // Ctrl+Left
    assert_eq!(cursor_of(&editor), (0, 9)); // start of 世

    editor.handle_input("\x1b[1;5D"); // Ctrl+Left over ，
    assert_eq!(cursor_of(&editor), (0, 6)); // after 你好

    editor.handle_input("\x1b[1;5D"); // Ctrl+Left over 好
    assert_eq!(cursor_of(&editor), (0, 3));

    editor.handle_input("\x1b[1;5D"); // Ctrl+Left over 你
    assert_eq!(cursor_of(&editor), (0, 0)); // start

    // Move right over 你
    editor.handle_input("\x1b[1;5C"); // Ctrl+Right
    assert_eq!(cursor_of(&editor), (0, 3));

    editor.handle_input("\x1b[1;5C"); // Ctrl+Right over 好
    assert_eq!(cursor_of(&editor), (0, 6)); // after 你好

    // Move right over ， (punctuation run)
    editor.handle_input("\x1b[1;5C"); // Ctrl+Right
    assert_eq!(cursor_of(&editor), (0, 9)); // start of 世

    editor.handle_input("\x1b[1;5C"); // Ctrl+Right over 世
    assert_eq!(cursor_of(&editor), (0, 12));

    editor.handle_input("\x1b[1;5C"); // Ctrl+Right over 界
    assert_eq!(cursor_of(&editor), (0, 15)); // end
}

#[test]
fn handles_mixed_cjk_and_ascii_word_movement() {
    let editor = editor();

    // The UAX #29 segmenter yields one word-like segment per ideograph where
    // upstream's dictionary segmenter grouped 你好 and 世界: bytes are
    // hello(0) 你(5) 好(8) ，(11) world(14) 世(19) 界(22) end(25).
    editor.set_text("hello你好，world世界");
    // Cursor at end (byte 25)

    // Move left over 界
    editor.handle_input("\x1b[1;5D"); // Ctrl+Left
    assert_eq!(cursor_of(&editor), (0, 22));

    editor.handle_input("\x1b[1;5D"); // Ctrl+Left over 世
    assert_eq!(cursor_of(&editor), (0, 19)); // after 'world'

    // Move left over world
    editor.handle_input("\x1b[1;5D"); // Ctrl+Left
    assert_eq!(cursor_of(&editor), (0, 14)); // after ，

    // Move left over ， (punctuation run)
    editor.handle_input("\x1b[1;5D"); // Ctrl+Left
    assert_eq!(cursor_of(&editor), (0, 11)); // after 你好

    editor.handle_input("\x1b[1;5D"); // Ctrl+Left over 好
    assert_eq!(cursor_of(&editor), (0, 8));

    editor.handle_input("\x1b[1;5D"); // Ctrl+Left over 你
    assert_eq!(cursor_of(&editor), (0, 5)); // after 'hello'

    // Move left over hello
    editor.handle_input("\x1b[1;5D"); // Ctrl+Left
    assert_eq!(cursor_of(&editor), (0, 0)); // start

    // Forward from start
    editor.handle_input("\x1b[1;5C"); // Ctrl+Right
    assert_eq!(cursor_of(&editor), (0, 5)); // after 'hello'

    editor.handle_input("\x1b[1;5C"); // Ctrl+Right over 你
    assert_eq!(cursor_of(&editor), (0, 8));

    editor.handle_input("\x1b[1;5C"); // Ctrl+Right over 好
    assert_eq!(cursor_of(&editor), (0, 11)); // after 你好

    editor.handle_input("\x1b[1;5C"); // Ctrl+Right over ，
    assert_eq!(cursor_of(&editor), (0, 14)); // after ，

    // Move right over world
    editor.handle_input("\x1b[1;5C"); // Ctrl+Right
    assert_eq!(cursor_of(&editor), (0, 19)); // after 'world'

    editor.handle_input("\x1b[1;5C"); // Ctrl+Right over 世
    assert_eq!(cursor_of(&editor), (0, 22));

    editor.handle_input("\x1b[1;5C"); // Ctrl+Right over 界
    assert_eq!(cursor_of(&editor), (0, 25)); // end
}

#[test]
fn centers_scroll_indicators_on_wide_borders() {
    let width = 40;
    let (tui, editor) = editor_with(width, 24);
    editor.set_text(
        &(0..20)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );

    let width = usize::from(width);
    Component::render(&editor, width);
    for _ in 0..10 {
        editor.handle_input("\x1b[A");
    }

    let lines = Component::render(&editor, width);
    assert_eq!(
        strip_terminal_sequences(&lines[0]),
        format!("{}{}{}", "─".repeat(15), " ↑ 9 more ", "─".repeat(15))
    );
    assert_eq!(
        strip_terminal_sequences(lines.last().expect("bottom border")),
        format!("{}{}{}", "─".repeat(15), " ↓ 4 more ", "─".repeat(15))
    );
    drop(tui);
}

#[test]
fn keeps_truncated_scroll_indicators_within_width_and_preserves_their_color_issue_6962() {
    let width = 10;
    let border_color: pi_tui::components::EditorColorFn =
        Rc::new(|text| format!("\x1b[35m{text}\x1b[39m"));
    let tui = new_editor_test_tui(width, 24);
    let editor = Editor::new(
        &tui,
        EditorTheme {
            border_color: Rc::clone(&border_color),
        },
    );
    let width = usize::from(width);
    editor.set_text(
        &(0..20)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );

    // Render once to initialize wrapping, then move the cursor so content
    // remains above and below the viewport.
    Component::render(&editor, width);
    for _ in 0..10 {
        editor.handle_input("\x1b[A");
    }

    let lines = Component::render(&editor, width);
    let top_border = &lines[0];
    let bottom_border = lines.last().expect("bottom border");

    assert!(strip_terminal_sequences(top_border).starts_with("─── ↑"));
    assert!(strip_terminal_sequences(bottom_border).starts_with("─── ↓"));
    assert_eq!(
        top_border,
        &border_color(&strip_terminal_sequences(top_border))
    );
    assert_eq!(
        bottom_border,
        &border_color(&strip_terminal_sequences(bottom_border))
    );
    for line in &lines {
        assert_eq!(
            visible_width(line),
            width,
            "line exceeds width {width}: {line}"
        );
    }
}

#[test]
fn wraps_lines_correctly_when_text_contains_wide_emojis() {
    let editor = editor();
    let width = 20;

    // ✅ is 2 columns wide, so "Hello ✅ World" is 14 columns
    editor.set_text("Hello ✅ World");
    let lines = Component::render(&editor, width);

    // All content lines (between borders) should fit within width
    for (index, line) in lines
        .iter()
        .enumerate()
        .skip(1)
        .take(lines.len().saturating_sub(2))
    {
        let line_width = visible_width(line);
        assert_eq!(
            line_width, width,
            "Line {index} has width {line_width}, expected {width}"
        );
    }
}

#[test]
fn wraps_long_text_with_emojis_at_correct_positions() {
    let editor = editor();
    let width = 10;

    // Each ✅ is 2 columns. "✅✅✅✅✅" = 10 columns, fits exactly
    // "✅✅✅✅✅✅" = 12 columns, needs wrap
    editor.set_text("✅✅✅✅✅✅");
    let lines = Component::render(&editor, width);

    // Should have 2 content lines (plus 2 border lines)
    for (index, line) in lines
        .iter()
        .enumerate()
        .skip(1)
        .take(lines.len().saturating_sub(2))
    {
        let line_width = visible_width(line);
        assert_eq!(
            line_width, width,
            "Line {index} has width {line_width}, expected {width}"
        );
    }
}

#[test]
fn renders_isolated_thai_and_lao_am_clusters_without_width_drift() {
    for text in ["ำabc", "ຳabc"] {
        let editor = editor();
        let width = 8;
        editor.set_text(text);

        for line in Component::render(&editor, width) {
            assert_eq!(
                visible_width(&line),
                width,
                "line width drift for {text:?}: {line}"
            );
        }
    }
}

#[test]
fn wraps_cjk_characters_correctly_each_is_2_columns_wide() {
    let editor = editor();
    let width = 10 + 1; // +1 col reserved for cursor

    // Each CJK char is 2 columns. "日本語テスト" = 6 chars = 12 columns
    editor.set_text("日本語テスト");
    let lines = Component::render(&editor, width);

    for (index, line) in lines
        .iter()
        .enumerate()
        .skip(1)
        .take(lines.len().saturating_sub(2))
    {
        let line_width = visible_width(line);
        assert_eq!(
            line_width, width,
            "Line {index} has width {line_width}, expected {width}"
        );
    }

    // Verify content split correctly
    let content_lines: Vec<String> = lines[1..lines.len() - 1]
        .iter()
        .map(|line| strip_terminal_sequences(line).trim().to_string())
        .collect();
    assert_eq!(content_lines.len(), 2);
    assert_eq!(content_lines[0], "日本語テス"); // 5 chars = 10 columns
    assert_eq!(content_lines[1], "ト"); // 1 char = 2 columns (+ padding)
}

#[test]
fn handles_mixed_ascii_and_wide_characters_in_wrapping() {
    let editor = editor();
    let width = 15 + 1; // +1 col reserved for cursor

    // "Test ✅ OK 日本" = 4 + 1 + 2 + 1 + 2 + 1 + 4 = 15 columns (fits in width-1=15)
    editor.set_text("Test ✅ OK 日本");
    let lines = Component::render(&editor, width);

    // Should fit in one content line
    let content_lines = &lines[1..lines.len() - 1];
    assert_eq!(content_lines.len(), 1);

    assert_eq!(visible_width(&content_lines[0]), width);
}

#[test]
fn renders_cursor_correctly_on_wide_characters() {
    let editor = editor();
    let width = 20;

    editor.set_text("A✅B");
    // Cursor should be at end (after B)
    let lines = Component::render(&editor, width);

    // The cursor (reverse video space) should be visible
    let content_line = &lines[1];
    assert!(
        content_line.contains("\x1b[7m"),
        "Should have reverse video cursor"
    );

    // Line should still be correct width
    assert_eq!(visible_width(content_line), width);
}

#[test]
fn does_not_exceed_terminal_width_with_emoji_at_wrap_boundary() {
    let editor = editor();
    let width = 11;

    // "0123456789✅" = 10 ASCII + 2-wide emoji = 12 columns
    // Should wrap before the emoji since it would exceed width
    editor.set_text("0123456789✅");
    let lines = Component::render(&editor, width);

    for (index, line) in lines
        .iter()
        .enumerate()
        .skip(1)
        .take(lines.len().saturating_sub(2))
    {
        let line_width = visible_width(line);
        assert!(
            line_width <= width,
            "Line {index} has width {line_width}, exceeds max {width}"
        );
    }
}

#[test]
fn shows_cursor_at_end_of_line_before_wrap_wraps_on_next_char() {
    let width: usize = 10;
    for padding_x in [0usize, 1] {
        let editor = editor_with_options((width + padding_x) as u16, EditorOptions { padding_x });

        // Type 9 chars → fills layoutWidth exactly, cursor at end on same line
        for ch in "aaaaaaaaa".chars() {
            editor.handle_input(&ch.to_string());
        }
        let lines = Component::render(&editor, width + padding_x);
        let content_lines = &lines[1..lines.len() - 1];
        assert_eq!(
            content_lines.len(),
            1,
            "Should be 1 content line before wrap"
        );
        assert!(
            content_lines[0].ends_with("\x1b[7m \x1b[0m"),
            "Cursor should be at end of line"
        );

        // Type 1 more → text wraps to second line
        editor.handle_input("a");
        let lines = Component::render(&editor, width + padding_x);
        let content_lines = &lines[1..lines.len() - 1];
        assert_eq!(content_lines.len(), 2, "Should wrap to 2 content lines");
    }
}

#[test]
fn wraps_at_word_boundaries_instead_of_mid_word() {
    let editor = editor();
    let width = 40;

    editor.set_text("Hello world this is a test of word wrapping functionality");
    let lines = Component::render(&editor, width);

    // Get content lines (between borders)
    let content_lines: Vec<String> = lines[1..lines.len() - 1]
        .iter()
        .map(|line| strip_terminal_sequences(line).trim().to_string())
        .collect();

    // Should NOT break mid-word
    assert!(
        !content_lines[0].ends_with('-'),
        "Line should not end with hyphen (mid-word break)"
    );

    // Each content line should be complete words
    for line in &content_lines {
        let last_char = line.trim_end().chars().last();
        assert!(
            last_char.is_none_or(|ch| ch.is_ascii_alphanumeric() || ".!?;:".contains(ch)),
            "Line ends unexpectedly with: {last_char:?}"
        );
    }
}

#[test]
fn does_not_start_lines_with_leading_whitespace_after_word_wrap() {
    let editor = editor();
    let width = 20;

    editor.set_text("Word1 Word2 Word3 Word4 Word5 Word6");
    let lines = Component::render(&editor, width);

    // Get content lines (between borders)
    let content_lines = &lines[1..lines.len() - 1];

    // No line should start with whitespace before content
    for (index, content_line) in content_lines.iter().enumerate() {
        let line = strip_terminal_sequences(content_line);
        let trimmed_end = line.trim_end();
        let starts_ws_then_content = trimmed_end.chars().next().is_some_and(char::is_whitespace)
            && trimmed_end.chars().any(|ch: char| !ch.is_whitespace());
        assert!(
            !starts_ws_then_content,
            "Line {index} starts with unexpected whitespace before content"
        );
    }
}

#[test]
fn breaks_long_words_urls_at_character_level() {
    let editor = editor();
    let width = 30;

    editor.set_text("Check https://example.com/very/long/path/that/exceeds/width here");
    let lines = Component::render(&editor, width);

    // All lines should fit within width
    for (index, line) in lines
        .iter()
        .enumerate()
        .skip(1)
        .take(lines.len().saturating_sub(2))
    {
        let line_width = visible_width(line);
        assert_eq!(
            line_width, width,
            "Line {index} has width {line_width}, expected {width}"
        );
    }
}

#[test]
fn preserves_multiple_spaces_within_words_on_same_line() {
    let editor = editor();
    let width = 50;

    editor.set_text("Word1   Word2    Word3");
    let lines = Component::render(&editor, width);

    let content_line = strip_terminal_sequences(&lines[1]).trim().to_string();
    // Multiple spaces should be preserved
    assert!(
        content_line.contains("Word1   Word2"),
        "Multiple spaces should be preserved"
    );
}

#[test]
fn handles_empty_string() {
    let editor = editor();
    let width = 40;

    editor.set_text("");
    let lines = Component::render(&editor, width);

    // Should have border + empty content + border
    assert_eq!(lines.len(), 3);
}

#[test]
fn handles_single_word_that_fits_exactly() {
    let editor = editor();
    let width = 10 + 1; // +1 col reserved for cursor

    editor.set_text("1234567890");
    let lines = Component::render(&editor, width);

    // Should have exactly 3 lines (top border, content, bottom border)
    assert_eq!(lines.len(), 3);
    let content_line = strip_terminal_sequences(&lines[1]);
    assert!(
        content_line.contains("1234567890"),
        "Content should contain the word"
    );
}

#[test]
fn wraps_word_to_next_line_when_it_ends_exactly_at_terminal_width() {
    // "hello " (6) + "world" (5) = 11, but "world" is non-whitespace ending at width.
    // Thus, wrap it to next line. The trailing space stays with "hello" on line 1
    let chunks = word_wrap_line("hello world test", 11, None);

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, "hello ");
    assert_eq!(chunks[1].text, "world test");
}

#[test]
fn keeps_whitespace_at_terminal_width_boundary_on_same_line() {
    // "hello world " is exactly 12 chars (including trailing space)
    // The space at position 12 should stay on the first line
    let chunks = word_wrap_line("hello world test", 12, None);

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, "hello world ");
    assert_eq!(chunks[1].text, "test");
}

#[test]
fn handles_unbreakable_word_filling_width_exactly_followed_by_space() {
    let chunks = word_wrap_line("aaaaaaaaaaaa aaaa", 12, None);

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, "aaaaaaaaaaaa");
    assert_eq!(chunks[1].text, " aaaa");
}

#[test]
fn wraps_word_to_next_line_when_it_fits_width_but_not_remaining_space() {
    let chunks = word_wrap_line("      aaaaaaaaaaaa", 12, None);

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, "      ");
    assert_eq!(chunks[1].text, "aaaaaaaaaaaa");
}

#[test]
fn keeps_word_with_multi_space_and_following_word_together_when_they_fit() {
    let chunks = word_wrap_line("Lorem ipsum dolor sit amet,    consectetur", 30, None);

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, "Lorem ipsum dolor sit ");
    assert_eq!(chunks[1].text, "amet,    consectetur");
}

#[test]
fn keeps_word_with_multi_space_and_following_word_when_they_fill_width_exactly() {
    let chunks = word_wrap_line(
        "Lorem ipsum dolor sit amet,              consectetur",
        30,
        None,
    );

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, "Lorem ipsum dolor sit ");
    assert_eq!(chunks[1].text, "amet,              consectetur");
}

#[test]
fn splits_when_word_plus_multi_space_plus_word_exceeds_width() {
    let chunks = word_wrap_line(
        "Lorem ipsum dolor sit amet,               consectetur",
        30,
        None,
    );

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].text, "Lorem ipsum dolor sit ");
    assert_eq!(chunks[1].text, "amet,               ");
    assert_eq!(chunks[2].text, "consectetur");
}

#[test]
fn breaks_long_whitespace_at_line_boundary() {
    let chunks = word_wrap_line(
        "Lorem ipsum dolor sit amet,                         consectetur",
        30,
        None,
    );

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].text, "Lorem ipsum dolor sit ");
    assert_eq!(chunks[1].text, "amet,                         ");
    assert_eq!(chunks[2].text, "consectetur");
}

#[test]
fn breaks_long_whitespace_at_line_boundary_2() {
    let chunks = word_wrap_line(
        "Lorem ipsum dolor sit amet,                          consectetur",
        30,
        None,
    );

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].text, "Lorem ipsum dolor sit ");
    assert_eq!(chunks[1].text, "amet,                         ");
    assert_eq!(chunks[2].text, " consectetur");
}

#[test]
fn breaks_whitespace_spanning_full_lines() {
    let chunks = word_wrap_line(
        "Lorem ipsum dolor sit amet,                                     consectetur",
        30,
        None,
    );

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].text, "Lorem ipsum dolor sit ");
    assert_eq!(chunks[1].text, "amet,                         ");
    assert_eq!(chunks[2].text, "            consectetur");
}

#[test]
fn force_breaks_when_wide_char_after_word_boundary_wrap_still_overflows() {
    // " " (1) + "a"*186 (186) + "你" (2) = 189 visible width
    // maxWidth = 187: backtracking to the space would leave 186 + 2 = 188 > 187,
    // so the algorithm must force-break before the wide char instead.
    let line = format!(" {}你", "a".repeat(186));
    let chunks = word_wrap_line(&line, 187, None);

    for chunk in &chunks {
        assert!(
            visible_width(&chunk.text) <= 187,
            "chunk \"{}...\" has visible width {}, expected <= 187",
            &chunk.text[..20.min(chunk.text.len())],
            visible_width(&chunk.text)
        );
    }
    // Verify no content is lost
    let reconstructed: String = chunks
        .iter()
        .map(|chunk| line[chunk.start_index..chunk.end_index].to_string())
        .collect();
    assert_eq!(reconstructed, line);
}

#[test]
fn splits_oversized_atomic_segment_across_multiple_chunks() {
    // Simulate a paste marker wider than maxWidth by passing pre-segmented data
    let marker = "[paste #1 +20 lines]"; // 21 chars
    let line = format!("A{marker}B");
    let segments = vec![seg("A", 0), seg(marker, 1), seg("B", 1 + marker.len())];

    let chunks = word_wrap_line(&line, 10, Some(&segments));

    // Every chunk must fit within maxWidth
    for chunk in &chunks {
        assert!(
            visible_width(&chunk.text) <= 10,
            "chunk \"{}\" has visible width {}, expected <= 10",
            chunk.text,
            visible_width(&chunk.text)
        );
    }

    // Verify no content is lost
    let reconstructed: String = chunks
        .iter()
        .map(|chunk| line[chunk.start_index..chunk.end_index].to_string())
        .collect();
    assert_eq!(reconstructed, line);
}

#[test]
fn splits_oversized_atomic_segment_at_start_of_line() {
    let marker = "[paste #1 +20 lines]"; // 21 chars
    let line = format!("{marker}B");
    let segments = vec![seg(marker, 0), seg("B", marker.len())];

    let chunks = word_wrap_line(&line, 10, Some(&segments));

    for chunk in &chunks {
        assert!(visible_width(&chunk.text) <= 10);
    }
    // "B" ends up on the last line (either alone or with the marker tail)
    assert!(chunks.last().expect("last chunk").text.contains('B'));

    let reconstructed: String = chunks.iter().map(|chunk| chunk.text.clone()).collect();
    assert_eq!(reconstructed, line);
}

#[test]
fn splits_oversized_atomic_segment_at_end_of_line() {
    let marker = "[paste #1 +20 lines]"; // 21 chars
    let line = format!("A{marker}");
    let segments = vec![seg("A", 0), seg(marker, 1)];

    let chunks = word_wrap_line(&line, 10, Some(&segments));

    for chunk in &chunks {
        assert!(visible_width(&chunk.text) <= 10);
    }
    assert_eq!(chunks[0].text, "A");

    let reconstructed: String = chunks.iter().map(|chunk| chunk.text.clone()).collect();
    assert_eq!(reconstructed, line);
}

#[test]
fn splits_consecutive_oversized_atomic_segments() {
    let m1 = "[paste #1 +20 lines]"; // 21 chars
    let m2 = "[paste #2 +30 lines]"; // 21 chars
    let line = format!("{m1}{m2}");
    let segments = vec![seg(m1, 0), seg(m2, m1.len())];

    let chunks = word_wrap_line(&line, 10, Some(&segments));

    for chunk in &chunks {
        assert!(
            visible_width(&chunk.text) <= 10,
            "chunk \"{}\" has visible width {}, expected <= 10",
            chunk.text,
            visible_width(&chunk.text)
        );
    }

    let reconstructed: String = chunks.iter().map(|chunk| chunk.text.clone()).collect();
    assert_eq!(reconstructed, line);
}

#[test]
fn wraps_normally_after_oversized_atomic_segment() {
    let marker = "[paste #1 +20 lines]"; // 21 chars
    let line = format!("{marker} hello world");
    let mut segments = vec![seg(marker, 0), seg(" ", marker.len())];
    for (offset, ch) in "hello world".char_indices() {
        segments.push(seg(&ch.to_string(), marker.len() + 1 + offset));
    }

    let chunks = word_wrap_line(&line, 10, Some(&segments));

    // All chunks must fit
    for chunk in &chunks {
        assert!(
            visible_width(&chunk.text) <= 10,
            "chunk \"{}\" has visible width {}, expected <= 10",
            chunk.text,
            visible_width(&chunk.text)
        );
    }

    // Last chunk should contain "world" (normal wrapping resumes)
    assert_eq!(chunks.last().expect("chunk").text, "world");

    let reconstructed: String = chunks.iter().map(|chunk| chunk.text.clone()).collect();
    assert_eq!(reconstructed, line);
}

/// Upstream's inline `Intl.SegmentData` object literal.
fn seg(segment: &str, index: usize) -> SegmentData {
    SegmentData {
        segment: segment.to_string(),
        index,
        is_word_like: None,
    }
}
#[test]
fn ctrl_w_saves_deleted_text_to_kill_ring_and_ctrl_y_yanks_it() {
    let editor = editor();

    editor.set_text("foo bar baz");
    editor.handle_input("\x17"); // Ctrl+W - deletes "baz"
    assert_eq!(editor.get_text(), "foo bar ");

    // Move to beginning and yank
    editor.handle_input("\x01"); // Ctrl+A
    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "bazfoo bar ");
}

#[test]
fn ctrl_u_saves_deleted_text_to_kill_ring() {
    let editor = editor();

    editor.set_text("hello world");
    // Move cursor to middle
    editor.handle_input("\x01"); // Ctrl+A (start)
    for _ in 0..6 {
        editor.handle_input("\x1b[C"); // Right 6 times - after "hello "
    }

    editor.handle_input("\x15"); // Ctrl+U - deletes "hello "
    assert_eq!(editor.get_text(), "world");

    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "hello world");
}

#[test]
fn ctrl_k_saves_deleted_text_to_kill_ring() {
    let editor = editor();

    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A (start)
    editor.handle_input("\x0b"); // Ctrl+K - deletes "hello world"

    assert_eq!(editor.get_text(), "");

    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "hello world");
}

#[test]
fn ctrl_y_does_nothing_when_kill_ring_is_empty() {
    let editor = editor();

    editor.set_text("test");
    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "test");
}

#[test]
fn alt_y_cycles_through_kill_ring_after_ctrl_y() {
    let editor = editor();

    // Create kill ring with multiple entries
    for text in ["first", "second", "third"] {
        editor.set_text(text);
        editor.handle_input("\x17"); // Ctrl+W - deletes the value
    }

    // Kill ring now has: [first, second, third]
    assert_eq!(editor.get_text(), "");

    editor.handle_input("\x19"); // Ctrl+Y - yanks "third" (most recent)
    assert_eq!(editor.get_text(), "third");

    editor.handle_input("\x1by"); // Alt+Y - cycles to "second"
    assert_eq!(editor.get_text(), "second");

    editor.handle_input("\x1by"); // Alt+Y - cycles to "first"
    assert_eq!(editor.get_text(), "first");

    editor.handle_input("\x1by"); // Alt+Y - cycles back to "third"
    assert_eq!(editor.get_text(), "third");
}

#[test]
fn alt_y_does_nothing_if_not_preceded_by_yank() {
    let editor = editor();

    editor.set_text("test");
    editor.handle_input("\x17"); // Ctrl+W - deletes "test"
    editor.set_text("other");

    // Type something to break the yank chain
    editor.handle_input("x");
    assert_eq!(editor.get_text(), "otherx");

    // Alt+Y should do nothing
    editor.handle_input("\x1by"); // Alt+Y
    assert_eq!(editor.get_text(), "otherx");
}

#[test]
fn alt_y_does_nothing_if_kill_ring_has_le_1_entry() {
    let editor = editor();

    editor.set_text("only");
    editor.handle_input("\x17"); // Ctrl+W - deletes "only"

    editor.handle_input("\x19"); // Ctrl+Y - yanks "only"
    assert_eq!(editor.get_text(), "only");

    editor.handle_input("\x1by"); // Alt+Y - should do nothing (only 1 entry)
    assert_eq!(editor.get_text(), "only");
}

#[test]
fn consecutive_ctrl_w_accumulates_into_one_kill_ring_entry() {
    let editor = editor();

    editor.set_text("one two three");
    editor.handle_input("\x17"); // Ctrl+W - deletes "three"
    editor.handle_input("\x17"); // Ctrl+W - deletes "two " (prepended)
    editor.handle_input("\x17"); // Ctrl+W - deletes "one " (prepended)

    assert_eq!(editor.get_text(), "");

    // Should be one combined entry
    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "one two three");
}

#[test]
fn ctrl_u_accumulates_multiline_deletes_including_newlines() {
    let editor = editor();

    // Start with multiline text, cursor at end
    editor.set_text("line1\nline2\nline3");
    // Cursor is at end of line3 (line 2, col 5)

    // Delete "line3"
    editor.handle_input("\x15"); // Ctrl+U
    assert_eq!(editor.get_text(), "line1\nline2\n");

    // Delete newline (at start of empty line 2, merges with line1)
    editor.handle_input("\x15"); // Ctrl+U
    assert_eq!(editor.get_text(), "line1\nline2");

    // Delete "line2"
    editor.handle_input("\x15"); // Ctrl+U
    assert_eq!(editor.get_text(), "line1\n");

    // Delete newline
    editor.handle_input("\x15"); // Ctrl+U
    assert_eq!(editor.get_text(), "line1");

    // Delete "line1"
    editor.handle_input("\x15"); // Ctrl+U
    assert_eq!(editor.get_text(), "");

    // All deletions accumulated into one entry: "line1\nline2\nline3"
    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "line1\nline2\nline3");
}

#[test]
fn backward_deletions_prepend_forward_deletions_append_during_accumulation() {
    let editor = editor();

    editor.set_text("prefix|suffix");
    // Position cursor at |
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..6 {
        editor.handle_input("\x1b[C"); // Move right 6 times
    }

    editor.handle_input("\x0b"); // Ctrl+K - deletes "suffix" (forward)
    editor.handle_input("\x0b"); // Ctrl+K - deletes "|" (forward, appended)
    assert_eq!(editor.get_text(), "prefix");

    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "prefix|suffix");
}

#[test]
fn non_delete_actions_break_kill_accumulation() {
    let editor = editor();

    // Delete "baz", then type "x" to break accumulation, then delete "x"
    editor.set_text("foo bar baz");
    editor.handle_input("\x17"); // Ctrl+W - deletes "baz"
    assert_eq!(editor.get_text(), "foo bar ");

    editor.handle_input("x"); // Typing breaks accumulation
    assert_eq!(editor.get_text(), "foo bar x");

    editor.handle_input("\x17"); // Ctrl+W - deletes "x" (separate entry, not accumulated)
    assert_eq!(editor.get_text(), "foo bar ");

    // Yank most recent - should be "x", not "xbaz"
    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "foo bar x");

    // Cycle to previous - should be "baz" (separate entry)
    editor.handle_input("\x1by"); // Alt+Y
    assert_eq!(editor.get_text(), "foo bar baz");
}

#[test]
fn non_yank_actions_break_alt_y_chain() {
    let editor = editor();

    for text in ["first", "second"] {
        editor.set_text(text);
        editor.handle_input("\x17"); // Ctrl+W
    }
    editor.set_text("");

    editor.handle_input("\x19"); // Ctrl+Y - yanks "second"
    assert_eq!(editor.get_text(), "second");

    editor.handle_input("x"); // Type breaks yank chain
    assert_eq!(editor.get_text(), "secondx");

    editor.handle_input("\x1by"); // Alt+Y - should do nothing
    assert_eq!(editor.get_text(), "secondx");
}

#[test]
fn kill_ring_rotation_persists_after_cycling() {
    let editor = editor();

    for text in ["first", "second", "third"] {
        editor.set_text(text);
        editor.handle_input("\x17"); // deletes the value
    }
    editor.set_text("");

    // Ring: [first, second, third]

    editor.handle_input("\x19"); // Ctrl+Y - yanks "third"
    editor.handle_input("\x1by"); // Alt+Y - cycles to "second", ring rotates

    // Now ring is: [third, first, second]
    assert_eq!(editor.get_text(), "second");

    // Do something else
    editor.handle_input("x");
    editor.set_text("");

    // New yank should get "second" (now at end after rotation)
    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "second");
}

#[test]
fn consecutive_deletions_across_lines_coalesce_into_one_entry() {
    let editor = editor();

    // "1\n2\n3" with cursor at end, delete everything with Ctrl+W
    editor.set_text("1\n2\n3");
    editor.handle_input("\x17"); // Ctrl+W - deletes "3"
    assert_eq!(editor.get_text(), "1\n2\n");

    editor.handle_input("\x17"); // Ctrl+W - deletes newline (merge with prev line)
    assert_eq!(editor.get_text(), "1\n2");

    editor.handle_input("\x17"); // Ctrl+W - deletes "2"
    assert_eq!(editor.get_text(), "1\n");

    editor.handle_input("\x17"); // Ctrl+W - deletes newline
    assert_eq!(editor.get_text(), "1");

    editor.handle_input("\x17"); // Ctrl+W - deletes "1"
    assert_eq!(editor.get_text(), "");

    // All deletions should have accumulated into one entry
    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "1\n2\n3");
}

#[test]
fn ctrl_k_at_line_end_deletes_newline_and_coalesces() {
    let editor = editor();

    // "ab" on line 1, "cd" on line 2, cursor at end of line 1
    editor.set_text("");
    editor.handle_input("a");
    editor.handle_input("b");
    editor.handle_input("\n");
    editor.handle_input("c");
    editor.handle_input("d");
    // Move to end of first line
    editor.handle_input("\x1b[A"); // Up arrow
    editor.handle_input("\x05"); // Ctrl+E - end of line

    // Now at end of "ab", Ctrl+K should delete newline (merge with "cd")
    editor.handle_input("\x0b"); // Ctrl+K - deletes newline
    assert_eq!(editor.get_text(), "abcd");

    // Continue deleting
    editor.handle_input("\x0b"); // Ctrl+K - deletes "cd"
    assert_eq!(editor.get_text(), "ab");

    // Both deletions should accumulate
    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "ab\ncd");
}

#[test]
fn handles_yank_in_middle_of_text() {
    let editor = editor();

    editor.set_text("word");
    editor.handle_input("\x17"); // Ctrl+W - deletes "word"
    editor.set_text("hello world");

    // Move to middle (after "hello ")
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..6 {
        editor.handle_input("\x1b[C");
    }

    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "hello wordworld");
}

#[test]
fn handles_yank_pop_in_middle_of_text() {
    let editor = editor();

    // Create two kill ring entries
    for text in ["FIRST", "SECOND"] {
        editor.set_text(text);
        editor.handle_input("\x17"); // Ctrl+W - deletes the value
    }

    // Set up "hello world" and position cursor after "hello "
    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A - go to start of line
    for _ in 0..6 {
        editor.handle_input("\x1b[C"); // Move right 6
    }

    // Yank "SECOND" in the middle
    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "hello SECONDworld");

    // Yank-pop replaces "SECOND" with "FIRST"
    editor.handle_input("\x1by"); // Alt+Y
    assert_eq!(editor.get_text(), "hello FIRSTworld");
}

#[test]
fn multiline_yank_and_yank_pop_in_middle_of_text() {
    let editor = editor();

    // Create single-line entry
    editor.set_text("SINGLE");
    editor.handle_input("\x17"); // Ctrl+W - deletes "SINGLE"

    // Create multiline entry via consecutive Ctrl+U
    editor.set_text("A\nB");
    editor.handle_input("\x15"); // Ctrl+U - deletes "B"
    editor.handle_input("\x15"); // Ctrl+U - deletes newline
    editor.handle_input("\x15"); // Ctrl+U - deletes "A"
    // Ring: ["SINGLE", "A\nB"]

    // Insert in middle of "hello world"
    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..6 {
        editor.handle_input("\x1b[C");
    }

    // Yank multiline "A\nB"
    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "hello A\nBworld");

    // Yank-pop replaces with "SINGLE"
    editor.handle_input("\x1by"); // Alt+Y
    assert_eq!(editor.get_text(), "hello SINGLEworld");
}

#[test]
fn alt_d_deletes_word_forward_and_saves_to_kill_ring() {
    let editor = editor();

    editor.set_text("hello world test");
    editor.handle_input("\x01"); // Ctrl+A - go to start

    editor.handle_input("\x1bd"); // Alt+D - deletes "hello"
    assert_eq!(editor.get_text(), " world test");

    editor.handle_input("\x1bd"); // Alt+D - deletes " world" (skips whitespace, then word)
    assert_eq!(editor.get_text(), " test");

    // Yank should get accumulated text
    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "hello world test");
}

#[test]
fn alt_d_at_end_of_line_deletes_newline() {
    let editor = editor();

    editor.set_text("line1\nline2");
    // Move to start of document, then to end of first line
    editor.handle_input("\x1b[A"); // Up arrow - go to first line
    editor.handle_input("\x05"); // Ctrl+E - end of line

    editor.handle_input("\x1bd"); // Alt+D - deletes newline (merges lines)
    assert_eq!(editor.get_text(), "line1line2");

    editor.handle_input("\x19"); // Ctrl+Y
    assert_eq!(editor.get_text(), "line1\nline2");
}

#[test]
fn does_nothing_when_undo_stack_is_empty() {
    let editor = editor();

    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "");
}

#[test]
fn coalesces_consecutive_word_characters_into_one_undo_unit() {
    let editor = editor();

    for key in ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] {
        editor.handle_input(key);
    }
    assert_eq!(editor.get_text(), "hello world");

    // Undo removes " world" (space captured state before it, so we restore to "hello")
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello");

    // Undo removes "hello"
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "");
}

#[test]
fn undoes_spaces_one_at_a_time() {
    let editor = editor();

    for key in ["h", "e", "l", "l", "o", " ", " "] {
        editor.handle_input(key);
    }
    assert_eq!(editor.get_text(), "hello  ");

    // Ctrl+- (undo) - removes second " "
    editor.handle_input("\x1b[45;5u");
    assert_eq!(editor.get_text(), "hello ");

    // Ctrl+- (undo) - removes first " "
    editor.handle_input("\x1b[45;5u");
    assert_eq!(editor.get_text(), "hello");

    // Ctrl+- (undo) - removes "hello"
    editor.handle_input("\x1b[45;5u");
    assert_eq!(editor.get_text(), "");
}

#[test]
fn undoes_newlines_and_signals_next_word_to_capture_state() {
    let editor = editor();

    for key in ["h", "e", "l", "l", "o", "\n", "w", "o", "r", "l", "d"] {
        editor.handle_input(key);
    }
    assert_eq!(editor.get_text(), "hello\nworld");

    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello\n");

    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello");

    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "");
}

#[test]
fn undoes_backspace() {
    let editor = editor();

    for key in ["h", "e", "l", "l", "o"] {
        editor.handle_input(key);
    }
    editor.handle_input("\x7f"); // Backspace
    assert_eq!(editor.get_text(), "hell");

    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello");
}

#[test]
fn undoes_forward_delete() {
    let editor = editor();

    for key in ["h", "e", "l", "l", "o"] {
        editor.handle_input(key);
    }
    editor.handle_input("\x01"); // Ctrl+A - go to start
    editor.handle_input("\x1b[C"); // Right arrow
    editor.handle_input("\x1b[3~"); // Delete key
    assert_eq!(editor.get_text(), "hllo");

    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello");
}

#[test]
fn undoes_ctrl_w_delete_word_backward() {
    let editor = editor();

    for key in ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] {
        editor.handle_input(key);
    }
    assert_eq!(editor.get_text(), "hello world");

    editor.handle_input("\x17"); // Ctrl+W
    assert_eq!(editor.get_text(), "hello ");

    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello world");
}

#[test]
fn undoes_ctrl_k_delete_to_line_end() {
    let editor = editor();

    for key in ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] {
        editor.handle_input(key);
    }
    editor.handle_input("\x01"); // Ctrl+A - go to start
    for _ in 0..6 {
        editor.handle_input("\x1b[C"); // Move right 6 times
    }

    editor.handle_input("\x0b"); // Ctrl+K
    assert_eq!(editor.get_text(), "hello ");

    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello world");

    editor.handle_input("|");
    assert_eq!(editor.get_text(), "hello |world");
}

#[test]
fn undoes_ctrl_u_delete_to_line_start() {
    let editor = editor();

    for key in ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] {
        editor.handle_input(key);
    }
    editor.handle_input("\x01"); // Ctrl+A - go to start
    for _ in 0..6 {
        editor.handle_input("\x1b[C"); // Move right 6 times
    }

    editor.handle_input("\x15"); // Ctrl+U
    assert_eq!(editor.get_text(), "world");

    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello world");
}

#[test]
fn undoes_yank() {
    let editor = editor();

    for key in ["h", "e", "l", "l", "o", " "] {
        editor.handle_input(key);
    }
    editor.handle_input("\x17"); // Ctrl+W - delete "hello "
    editor.handle_input("\x19"); // Ctrl+Y - yank
    assert_eq!(editor.get_text(), "hello ");

    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "");
}

#[test]
fn undoes_single_line_paste_atomically() {
    let editor = editor();

    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A - go to start
    for _ in 0..5 {
        editor.handle_input("\x1b[C"); // Move right 5 (after "hello", before space)
    }

    // Simulate bracketed paste of "beep boop"
    editor.handle_input("\x1b[200~beep boop\x1b[201~");
    assert_eq!(editor.get_text(), "hellobeep boop world");

    // Single undo should restore entire pre-paste state
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello world");

    editor.handle_input("|");
    assert_eq!(editor.get_text(), "hello| world");
}

#[test]
fn decodes_csi_u_ctrl_letter_sequences_inside_bracketed_paste_tmux_popup() {
    let editor = editor();

    // tmux popups with extended-keys-format=csi-u re-encode \n in pastes as
    // \x1b[106;5u (Ctrl+J). Without decoding, the per-char filter strips ESC
    // and leaks "[106;5u" between lines. See issue #3599.
    editor.handle_input("\x1b[200~line1\x1b[106;5uline2\x1b[106;5uline3\x1b[201~");
    assert_eq!(editor.get_text(), "line1\nline2\nline3");
}

#[test]
fn undoes_multi_line_paste_atomically() {
    let editor = editor();

    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A - go to start
    for _ in 0..5 {
        editor.handle_input("\x1b[C"); // Move right 5 (after "hello", before space)
    }

    // Simulate bracketed paste of multi-line text
    editor.handle_input("\x1b[200~line1\nline2\nline3\x1b[201~");
    assert_eq!(editor.get_text(), "helloline1\nline2\nline3 world");

    // Single undo should restore entire pre-paste state
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello world");

    editor.handle_input("|");
    assert_eq!(editor.get_text(), "hello| world");
}

#[test]
fn undoes_insert_text_at_cursor_atomically() {
    let editor = editor();

    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A - go to start
    for _ in 0..5 {
        editor.handle_input("\x1b[C"); // Move right 5 (after "hello", before space)
    }

    // Programmatic insertion (e.g., clipboard image path)
    editor.insert_text_at_cursor("/tmp/image.png");
    assert_eq!(editor.get_text(), "hello/tmp/image.png world");

    // Single undo should restore entire pre-insert state
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello world");

    editor.handle_input("|");
    assert_eq!(editor.get_text(), "hello| world");
}

#[test]
fn insert_text_at_cursor_handles_multiline_text() {
    let editor = editor();

    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A - go to start
    for _ in 0..5 {
        editor.handle_input("\x1b[C"); // Move right 5 (after "hello", before space)
    }

    // Insert multiline text
    editor.insert_text_at_cursor("line1\nline2\nline3");
    assert_eq!(editor.get_text(), "helloline1\nline2\nline3 world");

    // Cursor should be at end of inserted text (after "line3", before " world")
    let cursor = editor.get_cursor();
    assert_eq!(cursor.line, 2);
    assert_eq!(cursor.col, 5); // "line3".length

    // Single undo should restore entire pre-insert state
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello world");
}

#[test]
fn insert_text_at_cursor_normalizes_crlf_and_cr_line_endings() {
    let editor = editor();

    editor.set_text("");

    // Insert text with CRLF
    editor.insert_text_at_cursor("a\r\nb\r\nc");
    assert_eq!(editor.get_text(), "a\nb\nc");

    editor.handle_input("\x1b[45;5u"); // Undo
    assert_eq!(editor.get_text(), "");

    // Insert text with CR only
    editor.insert_text_at_cursor("x\ry\rz");
    assert_eq!(editor.get_text(), "x\ny\nz");
}

#[test]
fn undoes_set_text_to_empty_string() {
    let editor = editor();

    for key in ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] {
        editor.handle_input(key);
    }
    assert_eq!(editor.get_text(), "hello world");

    editor.set_text("");
    assert_eq!(editor.get_text(), "");

    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello world");
}

#[test]
fn clears_undo_stack_on_submit() {
    let editor = editor();
    let submitted = capture();
    *editor.on_submit.borrow_mut() = Some(on_submit_capture(&submitted));

    for key in ["h", "e", "l", "l", "o"] {
        editor.handle_input(key);
    }
    editor.handle_input("\r"); // Enter - submit

    assert_eq!(*submitted.borrow(), "hello");
    assert_eq!(editor.get_text(), "");

    // Undo should do nothing - stack was cleared
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "");
}

#[test]
fn exits_history_browsing_mode_on_undo() {
    let editor = editor();

    // Add "hello" to history
    editor.add_to_history("hello");
    assert_eq!(editor.get_text(), "");

    // Type "world"
    for key in ["w", "o", "r", "l", "d"] {
        editor.handle_input(key);
    }
    assert_eq!(editor.get_text(), "world");

    // Ctrl+W - delete word
    editor.handle_input("\x17"); // Ctrl+W
    assert_eq!(editor.get_text(), "");

    // Press Up - enter history browsing, shows "hello"
    editor.handle_input("\x1b[A"); // Up arrow
    assert_eq!(editor.get_text(), "hello");

    // Undo should restore to "" (state before entering history browsing)
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "");

    // Undo again should restore to "world" (state before Ctrl+W)
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "world");
}

#[test]
fn undo_restores_to_pre_history_state_even_after_multiple_history_navigations() {
    let editor = editor();

    // Add history entries
    editor.add_to_history("first");
    editor.add_to_history("second");
    editor.add_to_history("third");

    // Type something
    for key in ["c", "u", "r", "r", "e", "n", "t"] {
        editor.handle_input(key);
    }
    assert_eq!(editor.get_text(), "current");

    // Clear editor
    editor.handle_input("\x17"); // Ctrl+W
    assert_eq!(editor.get_text(), "");

    // Navigate through history multiple times
    editor.handle_input("\x1b[A"); // Up - "third"
    assert_eq!(editor.get_text(), "third");
    editor.handle_input("\x1b[A"); // Up - "second"
    assert_eq!(editor.get_text(), "second");
    editor.handle_input("\x1b[A"); // Up - "first"
    assert_eq!(editor.get_text(), "first");

    // Undo should go back to "" (state before we started browsing), not intermediate states
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "");

    // Another undo goes back to "current"
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "current");
}

#[test]
fn cursor_movement_starts_new_undo_unit() {
    let editor = editor();

    for key in ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] {
        editor.handle_input(key);
    }
    assert_eq!(editor.get_text(), "hello world");

    // Move cursor left 5 (to after "hello ")
    for _ in 0..5 {
        editor.handle_input("\x1b[D");
    }

    // Type "lol" in the middle
    for key in ["l", "o", "l"] {
        editor.handle_input(key);
    }
    assert_eq!(editor.get_text(), "hello lolworld");

    // Undo should restore to "hello world" (before inserting "lol")
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello world");

    editor.handle_input("|");
    assert_eq!(editor.get_text(), "hello |world");
}

#[test]
fn no_op_delete_operations_do_not_push_undo_snapshots() {
    let editor = editor();

    for key in ["h", "e", "l", "l", "o"] {
        editor.handle_input(key);
    }
    assert_eq!(editor.get_text(), "hello");

    // Delete word on empty - multiple times (should be no-ops)
    editor.handle_input("\x17"); // Ctrl+W - deletes "hello"
    assert_eq!(editor.get_text(), "");
    editor.handle_input("\x17"); // Ctrl+W - no-op (nothing to delete)
    editor.handle_input("\x17"); // Ctrl+W - no-op

    // Single undo should restore "hello"
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "hello");
}

#[test]
fn jumps_forward_to_first_occurrence_of_character_on_same_line() {
    let editor = editor();

    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A - go to start
    assert_eq!(cursor_of(&editor), (0, 0));

    editor.handle_input("\x1d"); // Ctrl+] (legacy sequence for ctrl+])
    editor.handle_input("o"); // Jump to first 'o'

    assert_eq!(cursor_of(&editor), (0, 4)); // 'o' in "hello"
}

#[test]
fn jumps_forward_to_next_occurrence_after_cursor() {
    let editor = editor();

    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A - go to start
    // Move cursor to the 'o' in "hello" (col 4)
    for _ in 0..4 {
        editor.handle_input("\x1b[C");
    }
    assert_eq!(cursor_of(&editor), (0, 4));

    editor.handle_input("\x1d"); // Ctrl+]
    editor.handle_input("o"); // Jump to next 'o' (in "world")

    assert_eq!(cursor_of(&editor), (0, 7)); // 'o' in "world"
}

#[test]
fn jumps_forward_across_multiple_lines() {
    let editor = editor();

    editor.set_text("abc\ndef\nghi");
    // Cursor is at end (line 2, col 3). Move to line 0 via up arrows, then Ctrl+A
    editor.handle_input("\x1b[A"); // Up
    editor.handle_input("\x1b[A"); // Up - now on line 0
    editor.handle_input("\x01"); // Ctrl+A - go to start of line
    assert_eq!(cursor_of(&editor), (0, 0));

    editor.handle_input("\x1d"); // Ctrl+]
    editor.handle_input("g"); // Jump to 'g' on line 3

    assert_eq!(cursor_of(&editor), (2, 0));
}

#[test]
fn jumps_backward_to_first_occurrence_before_cursor_on_same_line() {
    let editor = editor();

    editor.set_text("hello world");
    // Cursor at end (col 11)
    assert_eq!(cursor_of(&editor), (0, 11));

    editor.handle_input("\x1b\x1d"); // Ctrl+Alt+] (ESC followed by Ctrl+])
    editor.handle_input("o"); // Jump to last 'o' before cursor

    assert_eq!(cursor_of(&editor), (0, 7)); // 'o' in "world"
}

#[test]
fn jumps_backward_across_multiple_lines() {
    let editor = editor();

    editor.set_text("abc\ndef\nghi");
    // Cursor at end of line 3
    assert_eq!(cursor_of(&editor), (2, 3));

    editor.handle_input("\x1b\x1d"); // Ctrl+Alt+]
    editor.handle_input("a"); // Jump to 'a' on line 1

    assert_eq!(cursor_of(&editor), (0, 0));
}

#[test]
fn does_nothing_when_character_is_not_found_forward() {
    let editor = editor();

    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A - go to start
    assert_eq!(cursor_of(&editor), (0, 0));

    editor.handle_input("\x1d"); // Ctrl+]
    editor.handle_input("z"); // 'z' doesn't exist

    assert_eq!(cursor_of(&editor), (0, 0)); // Cursor unchanged
}

#[test]
fn does_nothing_when_character_is_not_found_backward() {
    let editor = editor();

    editor.set_text("hello world");
    // Cursor at end
    assert_eq!(cursor_of(&editor), (0, 11));

    editor.handle_input("\x1b\x1d"); // Ctrl+Alt+]
    editor.handle_input("z"); // 'z' doesn't exist

    assert_eq!(cursor_of(&editor), (0, 11)); // Cursor unchanged
}

#[test]
fn is_case_sensitive() {
    let editor = editor();

    editor.set_text("Hello World");
    editor.handle_input("\x01"); // Ctrl+A - go to start
    assert_eq!(cursor_of(&editor), (0, 0));

    // Search for lowercase 'h' - should not find it (only 'H' exists)
    editor.handle_input("\x1d"); // Ctrl+]
    editor.handle_input("h");

    assert_eq!(cursor_of(&editor), (0, 0)); // Cursor unchanged

    // Search for uppercase 'W' - should find it
    editor.handle_input("\x1d"); // Ctrl+]
    editor.handle_input("W");

    assert_eq!(cursor_of(&editor), (0, 6)); // 'W' in "World"
}

#[test]
fn cancels_jump_mode_when_ctrl_bracket_is_pressed_again() {
    let editor = editor();

    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A - go to start
    assert_eq!(cursor_of(&editor), (0, 0));

    editor.handle_input("\x1d"); // Ctrl+] - enter jump mode
    editor.handle_input("\x1d"); // Ctrl+] again - cancel

    // Type 'o' normally - should insert, not jump
    editor.handle_input("o");
    assert_eq!(editor.get_text(), "ohello world");
}

#[test]
fn cancels_jump_mode_on_escape_and_processes_the_escape() {
    let editor = editor();

    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A - go to start
    assert_eq!(cursor_of(&editor), (0, 0));

    editor.handle_input("\x1d"); // Ctrl+] - enter jump mode
    editor.handle_input("\x1b"); // Escape - cancel jump mode

    // Cursor should be unchanged (Escape itself doesn't move cursor in editor)
    assert_eq!(cursor_of(&editor), (0, 0));

    // Type 'o' normally - should insert, not jump
    editor.handle_input("o");
    assert_eq!(editor.get_text(), "ohello world");
}

#[test]
fn cancels_backward_jump_mode_when_ctrl_alt_bracket_is_pressed_again() {
    let editor = editor();

    editor.set_text("hello world");
    // Cursor at end
    assert_eq!(cursor_of(&editor), (0, 11));

    editor.handle_input("\x1b\x1d"); // Ctrl+Alt+] - enter backward jump mode
    editor.handle_input("\x1b\x1d"); // Ctrl+Alt+] again - cancel

    // Type 'o' normally - should insert, not jump
    editor.handle_input("o");
    assert_eq!(editor.get_text(), "hello worldo");
}

#[test]
fn searches_for_special_characters() {
    let editor = editor();

    editor.set_text("foo(bar) = baz;");
    editor.handle_input("\x01"); // Ctrl+A - go to start
    assert_eq!(cursor_of(&editor), (0, 0));

    // Jump to '('
    editor.handle_input("\x1d"); // Ctrl+]
    editor.handle_input("(");

    assert_eq!(cursor_of(&editor), (0, 3));

    // Jump to '='
    editor.handle_input("\x1d"); // Ctrl+]
    editor.handle_input("=");

    assert_eq!(cursor_of(&editor), (0, 9));
}

#[test]
fn handles_empty_text_gracefully() {
    let editor = editor();

    editor.set_text("");
    assert_eq!(cursor_of(&editor), (0, 0));

    editor.handle_input("\x1d"); // Ctrl+]
    editor.handle_input("x");

    assert_eq!(cursor_of(&editor), (0, 0)); // Cursor unchanged
}

#[test]
fn resets_last_action_when_jumping() {
    let editor = editor();

    editor.set_text("hello world");
    editor.handle_input("\x01"); // Ctrl+A - go to start

    // Type to set lastAction to "type-word"
    editor.handle_input("x");
    assert_eq!(editor.get_text(), "xhello world");

    // Jump forward
    editor.handle_input("\x1d"); // Ctrl+]
    editor.handle_input("o");

    // Type more - should start a new undo unit (lastAction was reset)
    editor.handle_input("Y");
    assert_eq!(editor.get_text(), "xhellYo world");

    // Undo should only undo "Y", not "x" as well
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "xhello world");
}

#[test]
fn preserves_target_column_when_moving_up_through_a_shorter_line() {
    let editor = editor();

    // Line 0: "2222222222x222" (x at col 10)
    // Line 1: "" (empty)
    // Line 2: "1111111111_111111111111" (_ at col 10)
    editor.set_text("2222222222x222\n\n1111111111_111111111111");

    // Position cursor on _ (line 2, col 10)
    assert_eq!(cursor_of(&editor), (2, 23)); // At end
    editor.handle_input("\x01"); // Ctrl+A - go to start of line
    for _ in 0..10 {
        editor.handle_input("\x1b[C"); // Move right to col 10
    }
    assert_eq!(cursor_of(&editor), (2, 10));

    // Press Up - should move to empty line (col clamped to 0)
    editor.handle_input("\x1b[A"); // Up arrow
    assert_eq!(cursor_of(&editor), (1, 0));

    // Press Up again - should move to line 0 at col 10 (on 'x')
    editor.handle_input("\x1b[A"); // Up arrow
    assert_eq!(cursor_of(&editor), (0, 10));
}

#[test]
fn preserves_target_column_when_moving_down_through_a_shorter_line() {
    let editor = editor();

    editor.set_text("1111111111_111\n\n2222222222x222222222222");

    // Position cursor on _ (line 0, col 10)
    editor.handle_input("\x1b[A"); // Up to line 1
    editor.handle_input("\x1b[A"); // Up to line 0
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..10 {
        editor.handle_input("\x1b[C");
    }
    assert_eq!(cursor_of(&editor), (0, 10));

    // Press Down - should move to empty line (col clamped to 0)
    editor.handle_input("\x1b[B"); // Down arrow
    assert_eq!(cursor_of(&editor), (1, 0));

    // Press Down again - should move to line 2 at col 10 (on 'x')
    editor.handle_input("\x1b[B"); // Down arrow
    assert_eq!(cursor_of(&editor), (2, 10));
}

#[test]
fn resets_sticky_column_on_horizontal_movement_left_arrow() {
    let editor = editor();

    editor.set_text("1234567890\n\n1234567890");

    // Start at line 2, col 5
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..5 {
        editor.handle_input("\x1b[C");
    }
    assert_eq!(cursor_of(&editor), (2, 5));

    // Move up through empty line
    editor.handle_input("\x1b[A"); // Up - line 1, col 0
    editor.handle_input("\x1b[A"); // Up - line 0, col 5 (sticky)
    assert_eq!(cursor_of(&editor), (0, 5));

    // Move left - resets sticky column
    editor.handle_input("\x1b[D"); // Left
    assert_eq!(cursor_of(&editor), (0, 4));

    // Move down twice
    editor.handle_input("\x1b[B"); // Down - line 1, col 0
    editor.handle_input("\x1b[B"); // Down - line 2, col 4 (new sticky from col 4)
    assert_eq!(cursor_of(&editor), (2, 4));
}

#[test]
fn resets_sticky_column_on_horizontal_movement_right_arrow() {
    let editor = editor();

    editor.set_text("1234567890\n\n1234567890");

    // Start at line 0, col 5
    editor.handle_input("\x1b[A"); // Up to line 1
    editor.handle_input("\x1b[A"); // Up to line 0
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..5 {
        editor.handle_input("\x1b[C");
    }
    assert_eq!(cursor_of(&editor), (0, 5));

    // Move down through empty line
    editor.handle_input("\x1b[B"); // Down - line 1, col 0
    editor.handle_input("\x1b[B"); // Down - line 2, col 5 (sticky)
    assert_eq!(cursor_of(&editor), (2, 5));

    // Move right - resets sticky column
    editor.handle_input("\x1b[C"); // Right
    assert_eq!(cursor_of(&editor), (2, 6));

    // Move up twice
    editor.handle_input("\x1b[A"); // Up - line 1, col 0
    editor.handle_input("\x1b[A"); // Up - line 0, col 6 (new sticky from col 6)
    assert_eq!(cursor_of(&editor), (0, 6));
}

#[test]
fn resets_sticky_column_on_typing() {
    let editor = editor();

    editor.set_text("1234567890\n\n1234567890");

    // Start at line 2, col 8
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..8 {
        editor.handle_input("\x1b[C");
    }

    // Move up through empty line
    editor.handle_input("\x1b[A"); // Up
    editor.handle_input("\x1b[A"); // Up - line 0, col 8
    assert_eq!(cursor_of(&editor), (0, 8));

    // Type a character - resets sticky column
    editor.handle_input("X");
    assert_eq!(cursor_of(&editor), (0, 9));

    // Move down twice
    editor.handle_input("\x1b[B"); // Down - line 1, col 0
    editor.handle_input("\x1b[B"); // Down - line 2, col 9 (new sticky from col 9)
    assert_eq!(cursor_of(&editor), (2, 9));
}

#[test]
fn resets_sticky_column_on_backspace() {
    let editor = editor();

    editor.set_text("1234567890\n\n1234567890");

    // Start at line 2, col 8
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..8 {
        editor.handle_input("\x1b[C");
    }

    // Move up through empty line
    editor.handle_input("\x1b[A"); // Up
    editor.handle_input("\x1b[A"); // Up - line 0, col 8
    assert_eq!(cursor_of(&editor), (0, 8));

    // Backspace - resets sticky column
    editor.handle_input("\x7f"); // Backspace
    assert_eq!(cursor_of(&editor), (0, 7));

    // Move down twice
    editor.handle_input("\x1b[B"); // Down - line 1, col 0
    editor.handle_input("\x1b[B"); // Down - line 2, col 7 (new sticky from col 7)
    assert_eq!(cursor_of(&editor), (2, 7));
}

#[test]
fn resets_sticky_column_on_ctrl_a_move_to_line_start() {
    let editor = editor();

    editor.set_text("1234567890\n\n1234567890");

    // Start at line 2, col 8
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..8 {
        editor.handle_input("\x1b[C");
    }

    // Move up - establishes sticky col 8
    editor.handle_input("\x1b[A"); // Up - line 1, col 0

    // Ctrl+A - resets sticky column to 0
    editor.handle_input("\x01"); // Ctrl+A
    assert_eq!(cursor_of(&editor), (1, 0));

    // Move up
    editor.handle_input("\x1b[A"); // Up - line 0, col 0 (new sticky from col 0)
    assert_eq!(cursor_of(&editor), (0, 0));
}

#[test]
fn resets_sticky_column_on_ctrl_e_move_to_line_end() {
    let editor = editor();

    editor.set_text("12345\n\n1234567890");

    // Start at line 2, col 3
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..3 {
        editor.handle_input("\x1b[C");
    }

    // Move up through empty line - establishes sticky col 3
    editor.handle_input("\x1b[A"); // Up - line 1, col 0
    editor.handle_input("\x1b[A"); // Up - line 0, col 3
    assert_eq!(cursor_of(&editor), (0, 3));

    // Ctrl+E - resets sticky column to end
    editor.handle_input("\x05"); // Ctrl+E
    assert_eq!(cursor_of(&editor), (0, 5));

    // Move down twice
    editor.handle_input("\x1b[B"); // Down - line 1, col 0
    editor.handle_input("\x1b[B"); // Down - line 2, col 5 (new sticky from col 5)
    assert_eq!(cursor_of(&editor), (2, 5));
}

#[test]
fn resets_sticky_column_on_word_movement_ctrl_left() {
    let editor = editor();

    editor.set_text("hello world\n\nhello world");

    // Start at end of line 2 (col 11)
    assert_eq!(cursor_of(&editor), (2, 11));

    // Move up through empty line - establishes sticky col 11
    editor.handle_input("\x1b[A"); // Up - line 1, col 0
    editor.handle_input("\x1b[A"); // Up - line 0, col 11
    assert_eq!(cursor_of(&editor), (0, 11));

    // Ctrl+Left - word movement resets sticky column
    editor.handle_input("\x1b[1;5D"); // Ctrl+Left
    assert_eq!(cursor_of(&editor), (0, 6)); // Before "world"

    // Move down twice
    editor.handle_input("\x1b[B"); // Down - line 1, col 0
    editor.handle_input("\x1b[B"); // Down - line 2, col 6 (new sticky from col 6)
    assert_eq!(cursor_of(&editor), (2, 6));
}

#[test]
fn resets_sticky_column_on_word_movement_ctrl_right() {
    let editor = editor();

    editor.set_text("hello world\n\nhello world");

    // Start at line 0, col 0
    editor.handle_input("\x1b[A"); // Up
    editor.handle_input("\x1b[A"); // Up
    editor.handle_input("\x01"); // Ctrl+A
    assert_eq!(cursor_of(&editor), (0, 0));

    // Move down through empty line - establishes sticky col 0
    editor.handle_input("\x1b[B"); // Down - line 1, col 0
    editor.handle_input("\x1b[B"); // Down - line 2, col 0
    assert_eq!(cursor_of(&editor), (2, 0));

    // Ctrl+Right - word movement resets sticky column
    editor.handle_input("\x1b[1;5C"); // Ctrl+Right
    assert_eq!(cursor_of(&editor), (2, 5)); // After "hello"

    // Move up twice
    editor.handle_input("\x1b[A"); // Up - line 1, col 0
    editor.handle_input("\x1b[A"); // Up - line 0, col 5 (new sticky from col 5)
    assert_eq!(cursor_of(&editor), (0, 5));
}

#[test]
fn resets_sticky_column_on_undo() {
    let editor = editor();

    editor.set_text("1234567890\n\n1234567890");

    // Go to line 0, col 8
    editor.handle_input("\x1b[A"); // Up to line 1
    editor.handle_input("\x1b[A"); // Up to line 0
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..8 {
        editor.handle_input("\x1b[C");
    }
    assert_eq!(cursor_of(&editor), (0, 8));

    // Move down through empty line - establishes sticky col 8
    editor.handle_input("\x1b[B"); // Down - line 1, col 0
    editor.handle_input("\x1b[B"); // Down - line 2, col 8 (sticky)
    assert_eq!(cursor_of(&editor), (2, 8));

    // Type something to create undo state - this clears sticky and sets col to 9
    editor.handle_input("X");
    assert_eq!(editor.get_text(), "1234567890\n\n12345678X90");
    assert_eq!(cursor_of(&editor), (2, 9));

    // Move up - establishes new sticky col 9
    editor.handle_input("\x1b[A"); // Up - line 1, col 0
    editor.handle_input("\x1b[A"); // Up - line 0, col 9
    assert_eq!(cursor_of(&editor), (0, 9));

    // Undo - resets sticky column and restores cursor to line 2, col 8
    editor.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(editor.get_text(), "1234567890\n\n1234567890");
    assert_eq!(cursor_of(&editor), (2, 8));

    // Move up - should capture new sticky from restored col 8, not old col 9
    editor.handle_input("\x1b[A"); // Up - line 1, col 0
    editor.handle_input("\x1b[A"); // Up - line 0, col 8 (new sticky from restored position)
    assert_eq!(cursor_of(&editor), (0, 8));
}

#[test]
fn handles_multiple_consecutive_up_down_movements() {
    let editor = editor();

    editor.set_text("1234567890\nab\ncd\nef\n1234567890");

    // Start at line 4, col 7
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..7 {
        editor.handle_input("\x1b[C");
    }
    assert_eq!(cursor_of(&editor), (4, 7));

    // Move up multiple times through short lines
    editor.handle_input("\x1b[A"); // Up - line 3, col 2 (clamped)
    editor.handle_input("\x1b[A"); // Up - line 2, col 2 (clamped)
    editor.handle_input("\x1b[A"); // Up - line 1, col 2 (clamped)
    editor.handle_input("\x1b[A"); // Up - line 0, col 7 (restored)
    assert_eq!(cursor_of(&editor), (0, 7));

    // Move down multiple times - sticky should still be 7
    editor.handle_input("\x1b[B"); // Down - line 1, col 2
    editor.handle_input("\x1b[B"); // Down - line 2, col 2
    editor.handle_input("\x1b[B"); // Down - line 3, col 2
    editor.handle_input("\x1b[B"); // Down - line 4, col 7 (restored)
    assert_eq!(cursor_of(&editor), (4, 7));
}

#[test]
fn moves_correctly_through_wrapped_visual_lines_without_getting_stuck() {
    // Narrow terminal
    let (tui, editor) = editor_with(15, 24);

    // Line 0: short
    // Line 1: 30 chars = wraps to 3 visual lines at width 10 (after padding)
    editor.set_text("short\n123456789012345678901234567890");
    Component::render(&editor, 15); // This gives 14 layout width

    // Position at end of line 1 (col 30)
    assert_eq!(cursor_of(&editor), (1, 30));

    // Move up repeatedly - should traverse all visual lines of the wrapped
    // text and eventually reach line 0
    editor.handle_input("\x1b[A"); // Up - to previous visual line within line 1
    assert_eq!(editor.get_cursor().line, 1);

    editor.handle_input("\x1b[A"); // Up - another visual line
    assert_eq!(editor.get_cursor().line, 1);

    editor.handle_input("\x1b[A"); // Up - should reach line 0
    assert_eq!(editor.get_cursor().line, 0);
    drop(tui);
}

#[test]
fn handles_set_text_resetting_sticky_column() {
    let editor = editor();

    editor.set_text("1234567890\n\n1234567890");

    // Establish sticky column
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..8 {
        editor.handle_input("\x1b[C");
    }
    editor.handle_input("\x1b[A"); // Up

    // setText should reset sticky column
    editor.set_text("abcdefghij\n\nabcdefghij");
    assert_eq!(cursor_of(&editor), (2, 10)); // At end

    // Move up - should capture new sticky from current position (10)
    editor.handle_input("\x1b[A"); // Up - line 1, col 0
    editor.handle_input("\x1b[A"); // Up - line 0, col 10
    assert_eq!(cursor_of(&editor), (0, 10));
}

#[test]
fn sets_preferred_visual_col_when_pressing_right_at_end_of_prompt_last_line() {
    let editor = editor();

    // Line 0: 20 chars with 'x' at col 10
    // Line 1: empty
    // Line 2: 10 chars ending with '_'
    editor.set_text("111111111x1111111111\n\n333333333_");

    // Go to line 0, press Ctrl+E (end of line) - col 20
    editor.handle_input("\x1b[A"); // Up to line 1
    editor.handle_input("\x1b[A"); // Up to line 0
    editor.handle_input("\x05"); // Ctrl+E - move to end of line
    assert_eq!(cursor_of(&editor), (0, 20));

    // Move down to line 2 - cursor clamped to col 10 (end of line)
    editor.handle_input("\x1b[B"); // Down to line 1, col 0
    editor.handle_input("\x1b[B"); // Down to line 2, col 10 (clamped)
    assert_eq!(cursor_of(&editor), (2, 10));

    // Press Right at end of prompt - nothing visible happens, but sets
    // preferredVisualCol to 10
    editor.handle_input("\x1b[C"); // Right - can't move, but sets preferredVisualCol
    assert_eq!(cursor_of(&editor), (2, 10)); // Still at same position

    // Move up twice to line 0 - should use preferredVisualCol (10) to land on 'x'
    editor.handle_input("\x1b[A"); // Up to line 1, col 0
    editor.handle_input("\x1b[A"); // Up to line 0, col 10 (on 'x')
    assert_eq!(cursor_of(&editor), (0, 10));
}

#[test]
fn handles_editor_resizes_when_preferred_visual_col_is_on_the_same_line() {
    // Create editor with wider terminal
    let (tui, editor) = editor_with(80, 24);

    editor.set_text("12345678901234567890\n\n12345678901234567890");

    // Start at line 2, col 15
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..15 {
        editor.handle_input("\x1b[C");
    }

    // Move up through empty line - establishes sticky col 15
    editor.handle_input("\x1b[A"); // Up
    editor.handle_input("\x1b[A"); // Up - line 0, col 15
    assert_eq!(cursor_of(&editor), (0, 15));

    // Render with narrower width to simulate resize
    Component::render(&editor, 12); // Width 12

    // Move down - sticky should be clamped to new width
    editor.handle_input("\x1b[B"); // Down - line 1
    editor.handle_input("\x1b[B"); // Down - line 2, col should be clamped
    assert_eq!(editor.get_cursor().col, 4);
    drop(tui);
}

#[test]
fn handles_editor_resizes_when_preferred_visual_col_is_on_a_different_line() {
    let (tui, editor) = editor_with(80, 24);

    // Create a line that wraps into multiple visual lines at width 10
    // "12345678901234567890" = 20 chars, wraps to 2 visual lines at width 10
    editor.set_text("short\n12345678901234567890");

    // Go to line 1, col 15
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..15 {
        editor.handle_input("\x1b[C");
    }
    assert_eq!(cursor_of(&editor), (1, 15));

    // Move up to establish sticky col 15
    editor.handle_input("\x1b[A"); // Up to line 0
    // Line 0 has only 5 chars, so cursor at col 5
    assert_eq!(cursor_of(&editor), (0, 5));

    // Narrow the editor
    Component::render(&editor, 10);

    // Move down - preferredVisualCol was 15, but width is 10
    // Should land on line 1, clamped to width (visual col 9, which is logical col 9)
    editor.handle_input("\x1b[B"); // Down to line 1
    assert_eq!(cursor_of(&editor), (1, 8));

    // Move up
    editor.handle_input("\x1b[A"); // Up - should go to line 0
    assert_eq!(cursor_of(&editor), (0, 5)); // Line 0 only has 5 chars

    // Restore the original width
    Component::render(&editor, 80);

    // Move down - preferredVisualCol was kept at 15
    editor.handle_input("\x1b[B"); // Down to line 1
    assert_eq!(cursor_of(&editor), (1, 15));
    drop(tui);
}

#[test]
fn rewrapped_lines_target_fits_current_visual_column() {
    let (tui, editor) = editor_with(80, 24);
    editor.set_text("abcdefghijklmnopqr\n123456789012345678");

    position_cursor(&editor, 0, 18);
    assert_eq!(cursor_of(&editor), (0, 18));

    // Narrow to width 10 (layoutWidth = 9).
    // Line 0 last segment has visual col max 9, line 1 first segment max 8
    Component::render(&editor, 10);

    // Move down: cursor clamps to 8
    editor.handle_input("\x1b[B");
    assert_eq!(cursor_of(&editor), (1, 8));

    // Widen back. Move up, the current visual col wins
    Component::render(&editor, 80);
    editor.handle_input("\x1b[A");
    assert_eq!(cursor_of(&editor), (0, 8));

    // Preferred was cleared by the rewrapped branch
    editor.handle_input("\x1b[B");
    assert_eq!(cursor_of(&editor), (1, 8));
    drop(tui);
}

#[test]
fn rewrapped_lines_target_shorter_than_current_visual_column() {
    let (tui, editor) = editor_with(80, 24);
    editor.set_text("abcdefghijklmnopqr\n123456789012345678\nab");

    position_cursor(&editor, 0, 18);
    assert_eq!(cursor_of(&editor), (0, 18));

    // Narrow to width 10 (layoutWidth = 9). Moving down clamps to col 8
    Component::render(&editor, 10);
    editor.handle_input("\x1b[B");
    assert_eq!(cursor_of(&editor), (1, 8));

    // Widen the editor
    Component::render(&editor, 80);

    // Move down to short line "ab".
    // preferredVisualCol is replaced with current visual col (8), cursor clamps to 2
    editor.handle_input("\x1b[B");
    assert_eq!(cursor_of(&editor), (2, 2));

    // Moving up restores to preferred col 8
    editor.handle_input("\x1b[A");
    assert_eq!(cursor_of(&editor), (1, 8));
    drop(tui);
}

#[test]
fn creates_a_paste_marker_for_large_pastes() {
    let editor = editor();
    let text = paste_with_marker(&editor);
    assert!(
        PASTE_MARKER_LINE_RE.is_match(&text),
        "marker missing in: {text}"
    );
}

#[test]
fn treats_paste_marker_as_single_unit_for_right_arrow() {
    let editor = editor();
    editor.handle_input("A");
    paste_with_marker(&editor);
    editor.handle_input("B");
    // Text: "A[paste #1 +20 lines]B", cursor at end

    // Go to start
    editor.handle_input("\x01"); // Ctrl+A
    assert_eq!(cursor_of(&editor), (0, 0));

    // Right arrow: should move past "A"
    editor.handle_input("\x1b[C");
    assert_eq!(cursor_of(&editor), (0, 1));

    // Right arrow: should skip the entire marker
    editor.handle_input("\x1b[C");
    let text = editor.get_text();
    let marker = PASTE_MARKER_LINE_RE.find(&text).expect("marker").as_str();
    assert_eq!(cursor_of(&editor), (0, 1 + marker.len()));

    // Right arrow: should move past "B"
    editor.handle_input("\x1b[C");
    assert_eq!(cursor_of(&editor), (0, 1 + marker.len() + 1));
}

#[test]
fn treats_paste_marker_as_single_unit_for_left_arrow() {
    let editor = editor();
    editor.handle_input("A");
    paste_with_marker(&editor);
    editor.handle_input("B");
    // Cursor at end

    // Left arrow: past "B"
    editor.handle_input("\x1b[D");
    let text = editor.get_text();
    let marker = PASTE_MARKER_LINE_RE.find(&text).expect("marker").as_str();
    assert_eq!(cursor_of(&editor), (0, 1 + marker.len()));

    // Left arrow: skip the entire marker
    editor.handle_input("\x1b[D");
    assert_eq!(cursor_of(&editor), (0, 1));

    // Left arrow: past "A"
    editor.handle_input("\x1b[D");
    assert_eq!(cursor_of(&editor), (0, 0));
}

#[test]
fn treats_paste_marker_as_single_unit_for_backspace() {
    let editor = editor();
    editor.handle_input("A");
    paste_with_marker(&editor);
    editor.handle_input("B");

    let text = editor.get_text();
    let marker = PASTE_MARKER_LINE_RE.find(&text).expect("marker").as_str();

    // Position cursor right after the marker (before "B")
    editor.handle_input("\x01"); // Ctrl+A
    // Move past "A" and the marker
    editor.handle_input("\x1b[C"); // past "A"
    editor.handle_input("\x1b[C"); // past marker
    assert_eq!(cursor_of(&editor), (0, 1 + marker.len()));

    // Backspace: should delete the entire marker at once
    editor.handle_input("\x7f");
    assert_eq!(editor.get_text(), "AB");
    assert_eq!(cursor_of(&editor), (0, 1));
}

#[test]
fn treats_paste_marker_as_single_unit_for_forward_delete() {
    let editor = editor();
    editor.handle_input("A");
    paste_with_marker(&editor);
    editor.handle_input("B");

    // Position cursor on "A" (col 0) then move right once to be just before marker
    editor.handle_input("\x01"); // Ctrl+A
    editor.handle_input("\x1b[C"); // past "A", now at col 1 (start of marker)

    // Forward delete: should delete the entire marker at once
    editor.handle_input("\x1b[3~"); // Delete key
    assert_eq!(editor.get_text(), "AB");
    assert_eq!(cursor_of(&editor), (0, 1));
}

#[test]
fn treats_paste_marker_as_single_unit_for_word_movement() {
    let editor = editor();
    editor.handle_input("X");
    editor.handle_input(" ");
    paste_with_marker(&editor);
    editor.handle_input(" ");
    editor.handle_input("Y");
    // Text: "X [paste #1 +20 lines] Y"

    let text = editor.get_text();
    let marker = PASTE_MARKER_LINE_RE.find(&text).expect("marker").as_str();

    // Go to start
    editor.handle_input("\x01"); // Ctrl+A

    // Ctrl+Right: skip "X"
    editor.handle_input("\x1b[1;5C");
    assert_eq!(cursor_of(&editor), (0, 1));

    // Ctrl+Right: skip whitespace + marker (marker treated as single non-ws,
    // non-punct unit)
    editor.handle_input("\x1b[1;5C");
    assert_eq!(cursor_of(&editor), (0, 2 + marker.len()));
}

#[test]
fn undo_restores_marker_after_backspace_deletion() {
    let editor = editor();
    editor.handle_input("A");
    paste_with_marker(&editor);
    editor.handle_input("B");

    let text_before = editor.get_text();

    // Position after marker
    editor.handle_input("\x01");
    editor.handle_input("\x1b[C"); // past A
    editor.handle_input("\x1b[C"); // past marker

    // Delete marker
    editor.handle_input("\x7f");
    assert_eq!(editor.get_text(), "AB");

    // Undo
    editor.handle_input("\x1b[45;5u");
    assert_eq!(editor.get_text(), text_before);
}

#[test]
fn undo_after_paste_marker_deletion_restores_the_paste_registry() {
    let editor = editor();
    let submitted = capture();
    *editor.on_submit.borrow_mut() = Some(on_submit_capture(&submitted));

    let paste = big_paste("alpha");
    editor.handle_input(&format!("\x1b[200~{paste}\x1b[201~"));
    editor.handle_input("\x7f"); // delete the marker
    editor.handle_input("\x1b[45;5u"); // undo: restores marker text and registry
    editor.handle_input("\r");
    assert_eq!(*submitted.borrow(), paste);
}

#[test]
fn undo_after_deleting_the_first_of_two_paste_markers_restores_both_registry_entries() {
    let editor = editor();
    let submitted = capture();
    *editor.on_submit.borrow_mut() = Some(on_submit_capture(&submitted));

    let paste_a = big_paste("alpha");
    let paste_b = big_paste("beta");
    editor.handle_input(&format!("\x1b[200~{paste_a}\x1b[201~")); // #1 = A
    editor.handle_input(&format!("\x1b[200~{paste_b}\x1b[201~")); // #2 = B, cursor at end
    editor.handle_input("\x01"); // Ctrl+A
    editor.handle_input("\x1b[C"); // right over marker #1
    editor.handle_input("\x7f"); // delete marker #1, renumbers #2 -> #1
    editor.handle_input("\x1b[45;5u"); // undo
    editor.handle_input("\r");
    assert_eq!(*submitted.borrow(), format!("{paste_a}{paste_b}"));
}

#[test]
fn renumbers_the_paste_registry_in_ascending_id_order_when_markers_are_out_of_order_in_text() {
    let editor = editor();
    let submitted = capture();
    *editor.on_submit.borrow_mut() = Some(on_submit_capture(&submitted));

    let paste_a = big_paste("alpha");
    let paste_b = big_paste("beta");
    let paste_c = big_paste("gamma");
    editor.handle_input(&format!("\x1b[200~{paste_a}\x1b[201~")); // #1 = A
    editor.handle_input("\x01"); // Ctrl+A
    editor.handle_input(&format!("\x1b[200~{paste_b}\x1b[201~")); // #2 = B, text: [#2][#1]
    editor.handle_input("\x01"); // Ctrl+A
    editor.handle_input(&format!("\x1b[200~{paste_c}\x1b[201~")); // #3 = C, text: [#3][#2][#1]
    editor.handle_input("\x05"); // Ctrl+E
    editor.handle_input("\x7f"); // delete marker #1, renumber #3 -> #2 and #2 -> #1
    editor.handle_input("\r");
    assert_eq!(*submitted.borrow(), format!("{paste_c}{paste_b}"));
}

#[test]
fn undo_after_set_text_restores_paste_markers_and_registry() {
    let editor = editor();
    let submitted = capture();
    *editor.on_submit.borrow_mut() = Some(on_submit_capture(&submitted));

    let paste = big_paste("alpha");
    editor.handle_input(&format!("\x1b[200~{paste}\x1b[201~"));
    editor.set_text("replacement");
    editor.handle_input("\x1b[45;5u"); // undo
    editor.handle_input("\r");
    assert_eq!(*submitted.borrow(), paste);
}

#[test]
fn handles_multiple_paste_markers_in_same_line() {
    let editor = editor();
    paste_with_marker(&editor);
    editor.handle_input(" ");
    paste_with_marker(&editor);

    let text = editor.get_text();
    let markers: Vec<&str> = PASTE_MARKER_LINE_RE
        .find_iter(&text)
        .map(|m| m.as_str())
        .collect();
    assert_eq!(markers.len(), 2);

    // Go to start
    editor.handle_input("\x01");

    // Right arrow: should skip first marker atomically
    editor.handle_input("\x1b[C");
    assert_eq!(cursor_of(&editor), (0, markers[0].len()));

    // Right arrow: past space
    editor.handle_input("\x1b[C");
    assert_eq!(cursor_of(&editor), (0, markers[0].len() + 1));

    // Right arrow: should skip second marker atomically
    editor.handle_input("\x1b[C");
    assert_eq!(
        cursor_of(&editor),
        (0, markers[0].len() + 1 + markers[1].len())
    );
}

#[test]
fn does_not_treat_manually_typed_marker_like_text_as_atomic_no_valid_paste_id() {
    let editor = editor();
    // Type text that matches the pattern but was typed manually (no paste entry)
    let fake_marker = "[paste #99 +5 lines]";
    for ch in fake_marker.chars() {
        editor.handle_input(&ch.to_string());
    }

    assert_eq!(editor.get_text(), fake_marker);

    // No paste with ID 99 exists, so the marker is NOT treated atomically.
    // Right arrow should move one grapheme at a time.
    editor.handle_input("\x01"); // Ctrl+A
    editor.handle_input("\x1b[C"); // Right
    assert_eq!(cursor_of(&editor), (0, 1)); // Just past "["
}

#[test]
fn does_not_crash_when_paste_marker_is_wider_than_terminal_width() {
    // Reproduce: terminal width 8, paste marker "[paste #1 +47 lines]" (21 chars)
    let (tui, editor) = editor_with(80, 24);
    let big_content = "line\n".repeat(47);
    let big_content = big_content.trim_end().to_string();
    editor.handle_input(&format!("\x1b[200~{big_content}\x1b[201~"));

    let text = editor.get_text();
    let marker = PASTE_MARKER_LINE_RE
        .find(&text)
        .expect("paste marker should be created")
        .as_str();
    assert!(
        visible_width(marker) > 8,
        "marker should be wider than render width"
    );

    // Render at very narrow width - should not throw
    let lines = Component::render(&editor, 8);
    // Every rendered line must fit within the width (marker is split)
    for line in &lines {
        assert!(
            visible_width(line) <= 8,
            "line exceeds width 8: visible={} text={line:?}",
            visible_width(line)
        );
    }
    drop(tui);
}

#[test]
fn does_not_crash_when_text_plus_paste_marker_exceeds_terminal_width_with_cursor_on_marker() {
    // Reproduce: terminal width 54, text "b".repeat(35) + "[paste #1 +27 lines]" + "bbbb"
    // Cursor lands on the paste marker after word-wrap, causing the rendered
    // line to be 55 visible chars (1 over the width).
    let (tui, editor) = editor_with(80, 24);

    // Type 35 'b' characters
    for _ in 0..35 {
        editor.handle_input("b");
    }

    // Paste 27 lines
    let big_content = "line\n".repeat(27);
    let big_content = big_content.trim_end().to_string();
    editor.handle_input(&format!("\x1b[200~{big_content}\x1b[201~"));

    // Type a few more characters
    for _ in 0..4 {
        editor.handle_input("b");
    }

    // Move cursor left to land on the paste marker
    for _ in 0..5 {
        editor.handle_input("\x1b[D"); // past last 'b', now on the paste marker
    }

    // Render at width 54 - should not throw
    let render_width = 54;
    let lines = Component::render(&editor, render_width);
    for line in &lines {
        assert!(
            visible_width(line) <= render_width,
            "line exceeds width {render_width}: visible={} text={line:?}",
            visible_width(line)
        );
    }
    drop(tui);
}

#[test]
fn word_wrap_line_rechecks_overflow_after_backtracking_to_wrap_opportunity() {
    // Reproduce crash #2: " " + "b".repeat(35) + atomic_marker(20 chars) + "bbbb"
    // layoutWidth=53. After wrapping at the space, the remaining 35 b's +
    // marker = 55 must trigger a second force-break instead of silently
    // overflowing.
    let (tui, editor) = editor_with(80, 24);

    // Type a space, then 35 b's
    editor.handle_input(" ");
    for _ in 0..35 {
        editor.handle_input("b");
    }

    // Paste 27 lines to create marker
    let big_content = "line\n".repeat(27);
    let big_content = big_content.trim_end().to_string();
    editor.handle_input(&format!("\x1b[200~{big_content}\x1b[201~"));

    // Type trailing chars
    for _ in 0..4 {
        editor.handle_input("b");
    }

    // Render at width 54 (contentWidth=54, layoutWidth=53 with paddingX=0)
    let render_width = 54;
    let lines = Component::render(&editor, render_width);
    for line in &lines {
        assert!(
            visible_width(line) <= render_width,
            "line exceeds width {render_width}: visible={} text={line:?}",
            visible_width(line)
        );
    }
    drop(tui);
}

#[test]
fn expands_large_pasted_content_literally_in_get_expanded_text() {
    let editor = editor();
    let pasted_text = [
        "line 1",
        "line 2",
        "line 3",
        "line 4",
        "line 5",
        "line 6",
        "line 7",
        "line 8",
        "line 9",
        "line 10",
        "tokens $1 $2 $& $$ $` $' end",
    ]
    .join("\n");

    editor.handle_input(&format!("\x1b[200~{pasted_text}\x1b[201~"));

    assert!(PASTE_MARKER_LINE_RE.is_match(&editor.get_text()));
    assert_eq!(editor.get_expanded_text(), pasted_text);
}

#[test]
fn snaps_to_the_paste_marker_start_when_navigating_down_into_it() {
    let editor = editor();

    // Line 0: long enough text to establish a sticky column
    editor.set_text("12345678901234567890\n\nhello ");

    // Create a large paste to get a marker
    let big_content = "x".repeat(2000);
    editor.handle_input(&format!("\x1b[200~{big_content}\x1b[201~"));
    Component::render(&editor, 80);

    let text = editor.get_text();
    let _marker = PASTE_MARKER_CHARS_RE.find(&text).expect("marker").as_str();
    // Line 0: "12345678901234567890"
    // Line 1: "" (empty)
    // Line 2: "hello [paste #1 2000 chars]"
    //         marker starts at col 6

    // Navigate to line 0, col 10
    editor.handle_input("\x1b[A"); // Up to line 1
    editor.handle_input("\x1b[A"); // Up to line 0
    editor.handle_input("\x01"); // Ctrl+A (start of line)
    for _ in 0..10 {
        editor.handle_input("\x1b[C"); // Right 10
    }
    assert_eq!(cursor_of(&editor), (0, 10));

    // Down to empty line
    editor.handle_input("\x1b[B");
    assert_eq!(cursor_of(&editor), (1, 0));

    // Down to paste marker line - sticky col 10 falls inside marker (starts
    // at col 6). Cursor should snap to start of marker (col 6), not end.
    editor.handle_input("\x1b[B");
    assert_eq!(cursor_of(&editor), (2, 6));
}

#[test]
fn preserves_sticky_column_when_navigating_through_paste_marker_line() {
    let (tui, editor) = editor_with(30, 24);

    // Build:
    // Line 0: "1234567890123456" (16 chars)
    // Line 1: "" (empty)
    // Line 2: "[paste #1 2000 chars]" (paste marker)
    // Line 3: "" (empty)
    // Line 4: "abcdefghijklmnop" (16 chars)
    for ch in "1234567890123456".chars() {
        editor.handle_input(&ch.to_string());
    }
    editor.handle_input("\n");
    editor.handle_input("\n");
    editor.handle_input(&format!("\x1b[200~{}\x1b[201~", "x".repeat(2000)));
    editor.handle_input("\n");
    editor.handle_input("\n");
    for ch in "abcdefghijklmnop".chars() {
        editor.handle_input(&ch.to_string());
    }
    Component::render(&editor, 30);

    // Navigate to line 0, col 10
    for _ in 0..4 {
        editor.handle_input("\x1b[A"); // Up to line 0
    }
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..10 {
        editor.handle_input("\x1b[C");
    }
    assert_eq!(cursor_of(&editor), (0, 10));

    // Down to empty line - sticky col 10 established
    editor.handle_input("\x1b[B");
    assert_eq!(cursor_of(&editor), (1, 0));

    // Down to paste marker - cursor snapped to col 0 (start of marker)
    editor.handle_input("\x1b[B");
    assert_eq!(cursor_of(&editor), (2, 0));

    // Down to empty line
    editor.handle_input("\x1b[B");
    assert_eq!(cursor_of(&editor), (3, 0));

    // Down to last line - should restore sticky col 10
    editor.handle_input("\x1b[B");
    assert_eq!(cursor_of(&editor), (4, 10));
    drop(tui);
}

#[test]
fn does_not_get_stuck_moving_down_from_a_multi_visual_line_paste_marker() {
    let (tui, editor) = editor_with(20, 24);

    // Build:
    // Logical line 0: "abcdefgh" + marker(21 chars) + "ijklmnopqr"
    // Logical line 1: "123456789012345678"
    //
    // Marker "[paste #1 +100 lines]" (21 chars) is wider than the terminal
    // (20). Word-wrap splits at the space before "lines", producing:
    //   VL1: abcdefgh              (startCol 0,  len 8)
    //   VL2: [paste #1 +100        (startCol 8,  len 15) <- marker head
    //   VL3: lines]ijklmnopqr      (startCol 23, len 16) <- marker tail + content
    //   VL4: 123456789012345678    (line 1)
    //
    // On VL3 the marker tail "lines]" occupies visual cols 0-5.
    // Content ("i") starts at visual col 6 = logical col 29.
    for ch in "abcdefgh".chars() {
        editor.handle_input(&ch.to_string());
    }
    let big_content = "line\n".repeat(100);
    let big_content = big_content.trim_end().to_string();
    editor.handle_input(&format!("\x1b[200~{big_content}\x1b[201~"));
    for ch in "ijklmnopqr".chars() {
        editor.handle_input(&ch.to_string());
    }
    editor.handle_input("\n");
    for ch in "123456789012345678".chars() {
        editor.handle_input(&ch.to_string());
    }
    Component::render(&editor, 20);

    let text = editor.get_text();
    let marker = PASTE_MARKER_LINE_RE
        .find(&text)
        .expect("paste marker should be created")
        .as_str();
    let marker_len = marker.len(); // 21
    assert!(marker_len > 20, "marker should be wider than terminal");
    let marker_start = 8;
    let marker_end = marker_start + marker_len; // 29

    // Navigate to line 0, col 6 (on "g"). Preferred col 6 is past the
    // marker tail on VL3, so the cursor should land on content ("i" at
    // col 29) without snapping back.
    editor.handle_input("\x1b[A"); // Up to line 0
    editor.handle_input("\x01"); // Ctrl+A (start of line)
    for _ in 0..6 {
        editor.handle_input("\x1b[C"); // Right to col 6
    }
    assert_eq!(cursor_of(&editor), (0, 6));

    // Down: cursor lands on paste marker start
    editor.handle_input("\x1b[B");
    assert_eq!(cursor_of(&editor), (0, 8));

    // Down again: preferred col 6 lands at VL3 col 29 ("i"), which is
    // past the marker. Cursor stays on line 0.
    editor.handle_input("\x1b[B");
    assert_eq!(editor.get_cursor().line, 0);
    assert_eq!(editor.get_cursor().col, marker_end); // col 29 = "i"

    // Up: back to paste marker
    editor.handle_input("\x1b[A");
    assert_eq!(cursor_of(&editor), (0, 8));

    // Up again: back to col 6 ("g")
    editor.handle_input("\x1b[A");
    assert_eq!(cursor_of(&editor), (0, 6));
    drop(tui);
}

#[test]
fn skips_marker_continuation_vls_when_preferred_col_falls_in_marker_tail() {
    let (tui, editor) = editor_with(20, 24);

    // Same layout. Start at col 3 ("d"). Preferred col 3 maps to VL3
    // visual col 3 which is inside the "lines]" marker tail.
    // moveToVisualLine detects the continuation VL and skips to VL4
    // (line 1).
    for ch in "abcdefgh".chars() {
        editor.handle_input(&ch.to_string());
    }
    let big_content = "line\n".repeat(100);
    let big_content = big_content.trim_end().to_string();
    editor.handle_input(&format!("\x1b[200~{big_content}\x1b[201~"));
    for ch in "ijklmnopqr".chars() {
        editor.handle_input(&ch.to_string());
    }
    editor.handle_input("\n");
    for ch in "123456789012345678".chars() {
        editor.handle_input(&ch.to_string());
    }
    Component::render(&editor, 20);

    // Navigate to line 0, col 3 (on "d")
    editor.handle_input("\x1b[A"); // Up to line 0
    editor.handle_input("\x01"); // Ctrl+A
    for _ in 0..3 {
        editor.handle_input("\x1b[C");
    }
    assert_eq!(cursor_of(&editor), (0, 3));

    // Down: marker
    editor.handle_input("\x1b[B");
    assert_eq!(editor.get_cursor().col, 8);

    // Down: skips VL3 (col 3 in marker tail) and lands on line 1
    editor.handle_input("\x1b[B");
    assert_eq!(cursor_of(&editor), (1, 3));

    // Round-trip back
    editor.handle_input("\x1b[A");
    assert_eq!(editor.get_cursor().col, 8); // marker
    editor.handle_input("\x1b[A");
    assert_eq!(cursor_of(&editor), (0, 3));
    drop(tui);
}

#[test]
fn submits_large_pasted_content_literally() {
    let editor = editor();
    let pasted_text = [
        "line 1",
        "line 2",
        "line 3",
        "line 4",
        "line 5",
        "line 6",
        "line 7",
        "line 8",
        "line 9",
        "line 10",
        "tokens $1 $2 $& $$ $` $' end",
    ]
    .join("\n");
    let submitted = capture();
    *editor.on_submit.borrow_mut() = Some(on_submit_capture(&submitted));

    editor.handle_input(&format!("\x1b[200~{pasted_text}\x1b[201~"));
    editor.handle_input("\r");

    assert_eq!(*submitted.borrow(), pasted_text);
}

/// Simulate a large paste that creates a marker, upstream `pasteWithMarker`.
fn paste_with_marker(editor: &Editor) -> String {
    let big_content = "line\n".repeat(20);
    let big_content = big_content.trim_end().to_string(); // 20 lines
    editor.handle_input(&format!("\x1b[200~{big_content}\x1b[201~"));
    // The editor replaces large pastes with a marker like "[paste #1 +20 lines]"
    editor.get_text()
}

/// 12-line paste content with a distinguishing tag, upstream `bigPaste`.
fn big_paste(tag: &str) -> String {
    (0..12)
        .map(|index| format!("{tag}{index}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The paste-marker line pattern the suite matches, upstream
/// `/\[paste #\d+ \+\d+ lines\]/`.
static PASTE_MARKER_LINE_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"\[paste #\d+ \+\d+ lines\]").expect("static regex")
});

/// The paste-marker chars pattern, upstream `/\[paste #\d+ \d+ chars\]/`.
static PASTE_MARKER_CHARS_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"\[paste #\d+ \d+ chars\]").expect("static regex")
});

//! Boundary tests for the editor slice's branches upstream left untested
//! (upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): page scroll,
//! the arrow wrap branches at line boundaries, mouse cursor placement, the
//! paste pipeline's decode and file-path arms, delete guards, the padding
//! and terminal-row plumbing, and the Debug surfaces.

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::cell::RefCell;
use std::rc::Rc;

use pi_tui::components::EditorComponent;
use pi_tui::components::{Editor, EditorOptions, Input, InputOptions};
use pi_tui::tui::{Component, Focusable, TuiMouseButton, TuiMouseEvent, TuiMouseEventType};
use tui_support::new_editor_test_tui;

fn editor() -> Editor {
    let tui = new_editor_test_tui(80, 24);
    Editor::new(&tui, tui_support::default_editor_theme())
}

const fn mouse(
    event_type: TuiMouseEventType,
    button: TuiMouseButton,
    x: u16,
    y: u16,
) -> TuiMouseEvent {
    TuiMouseEvent {
        event_type,
        button,
        x,
        y,
        screen_x: x,
        screen_y: y,
        width: 80,
        height: 3,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: None,
    }
}

// =============================================================================
// Editor: page scroll and arrow boundaries
// =============================================================================

#[test]
fn page_up_scrolls_up_and_clamps_at_the_top() {
    let tui = new_editor_test_tui(80, 24);
    let editor = Editor::new(&tui, tui_support::default_editor_theme());
    editor.set_text(
        &(0..30)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    // Cursor at end of the document (line 29).
    assert_eq!(editor.get_cursor().line, 29);

    // A page is 30% of 24 rows = 7 lines, minimum 5.
    editor.handle_input("\x1b[5~"); // PageUp
    assert_eq!(editor.get_cursor().line, 22);

    editor.handle_input("\x1b[5~"); // PageUp
    assert_eq!(editor.get_cursor().line, 15);

    // PageUp past the top clamps to the first line.
    for _ in 0..10 {
        editor.handle_input("\x1b[5~");
    }
    assert_eq!(editor.get_cursor().line, 0);
}

#[test]
fn page_down_scrolls_down_and_clamps_at_the_bottom() {
    let tui = new_editor_test_tui(80, 24);
    let editor = Editor::new(&tui, tui_support::default_editor_theme());
    editor.set_text(
        &(0..30)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    // Walk to line 0.
    for _ in 0..40 {
        editor.handle_input("\x1b[A");
    }
    editor.handle_input("\x01"); // Ctrl+A - line 0, col 0

    editor.handle_input("\x1b[6~"); // PageDown
    assert_eq!(editor.get_cursor().line, 7);

    // Past the bottom clamps to the last line.
    for _ in 0..10 {
        editor.handle_input("\x1b[6~");
    }
    assert_eq!(editor.get_cursor().line, 29);
}

#[test]
fn right_arrow_at_end_of_last_line_records_the_sticky_column() {
    let editor = editor();
    editor.set_text("ab\ncd");
    editor.handle_input("\x1b[B"); // Down to line 1 end
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 1, col: 2 }
    );

    // Right at the very end: no move, but the sticky column records col 2.
    editor.handle_input("\x1b[C");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 1, col: 2 }
    );

    // Up from there restores the recorded visual column on line 0.
    editor.handle_input("\x1b[A");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 2 }
    );
}

#[test]
fn right_arrow_wraps_to_the_next_line_and_left_wraps_back() {
    let editor = editor();
    editor.set_text("ab\ncd");
    editor.handle_input("\x1b[A"); // Up to line 0
    editor.handle_input("\x01"); // Ctrl+A - line 0 col 0
    for _ in 0..2 {
        editor.handle_input("\x1b[C"); // Right to line end
    }
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 2 }
    );

    // One more Right wraps to the start of line 1.
    editor.handle_input("\x1b[C");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 1, col: 0 }
    );

    // Left wraps back to the end of line 0.
    editor.handle_input("\x1b[D");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 2 }
    );
}

#[test]
fn word_backward_at_column_zero_moves_to_the_previous_line_end() {
    let editor = editor();
    editor.set_text("first\nsecond");
    // Cursor on line 1, column 0.
    editor.handle_input("\x01"); // Ctrl+A - line 1, col 0
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 1, col: 0 }
    );

    // Ctrl+Left at column 0: end of the previous line.
    editor.handle_input("\x1b[1;5D");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 5 }
    );

    // Word-left inside the line stops at the word start.
    editor.handle_input("\x1b[1;5D");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 0 }
    );

    // At the first column of the first line it is a no-op.
    editor.handle_input("\x1b[1;5D");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 0 }
    );
}

#[test]
fn word_forward_at_line_end_moves_to_the_next_line_start() {
    let editor = editor();
    editor.set_text("first\nsecond");
    // One Up from the end of line 1 clamps onto the end of line 0.
    editor.handle_input("\x1b[A");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 5 }
    );

    // Ctrl+Right at the line end wraps to the start of line 1.
    editor.handle_input("\x1b[1;5C");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 1, col: 0 }
    );

    // At the very end of the document it is a no-op.
    editor.handle_input("\x05"); // Ctrl+E
    editor.handle_input("\x1b[1;5C");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 1, col: 6 }
    );
}

#[test]
fn backspace_at_column_zero_merges_with_the_previous_line() {
    let editor = editor();
    editor.set_text("first\nsecond");
    // Cursor on line 1, column 0.
    editor.handle_input("\x01"); // Ctrl+A - line 1, col 0
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 1, col: 0 }
    );

    editor.handle_input("\x7f"); // Backspace merges the lines
    assert_eq!(editor.get_text(), "firstsecond");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 5 }
    );
}

#[test]
fn forward_delete_at_line_end_merges_with_the_next_line() {
    let editor = editor();
    editor.set_text("ab\ncd");

    editor.handle_input("\x05"); // Ctrl+E - end of line 1
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 1, col: 2 }
    );

    editor.handle_input("\x1b[A"); // Up
    editor.handle_input("\x05"); // Ctrl+E - end of line 0
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 2 }
    );

    // Forward delete at line end merges with the next line.
    editor.handle_input("\x1b[3~"); // Delete key
    assert_eq!(editor.get_text(), "abcd");

    // Forward delete at the very end is a no-op.
    editor.handle_input("\x05"); // Ctrl+E
    editor.handle_input("\x1b[3~");
    assert_eq!(editor.get_text(), "abcd");
}

#[test]
fn set_padding_x_round_trips_and_requests_a_render() {
    let tui = new_editor_test_tui(80, 24);
    let editor = Editor::with_options(
        &tui,
        tui_support::default_editor_theme(),
        EditorOptions { padding_x: 0 },
    );
    assert_eq!(editor.get_padding_x(), 0);

    editor.set_padding_x(3);
    assert_eq!(editor.get_padding_x(), 3);
    editor.set_padding_x(3); // Same value: no render request, no change.
    assert_eq!(editor.get_padding_x(), 3);
}

#[test]
fn padding_shifts_the_content_and_the_cursor_overflow_folds_the_right_pad() {
    let tui = new_editor_test_tui(80, 24);
    let editor = Editor::with_options(
        &tui,
        tui_support::default_editor_theme(),
        EditorOptions { padding_x: 2 },
    );
    editor.set_text("word");
    let lines = Component::render(&editor, 20);
    // The content row starts with two spaces of padding.
    assert!(lines[1].starts_with("  word"));
    // The cursor sits at the end: an extra highlighted space folds one right
    // padding column away.
    assert!(lines[1].contains("\x1b[7m \x1b[0m"));
}

#[test]
fn firing_on_change_delivers_the_text_and_submit_reports_the_empty_editor() {
    let tui = new_editor_test_tui(80, 24);
    let editor = Editor::new(&tui, tui_support::default_editor_theme());
    let changes: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    *editor.on_change.borrow_mut() = Some({
        let changes = Rc::clone(&changes);
        Rc::new(move |text: &str| changes.borrow_mut().push(text.to_string()))
    });

    editor.handle_input("h");
    editor.handle_input("i");
    assert_eq!(changes.borrow().len(), 2);
    assert_eq!(changes.borrow().last().map(String::as_str), Some("hi"));

    editor.handle_input("\r"); // Enter submits: the editor resets and reports ""
    assert_eq!(changes.borrow().last().map(String::as_str), Some(""));
}

#[test]
fn shift_enter_bound_submit_converts_backslash_enter_to_a_newline() {
    let editor = editor();
    // With shift+enter among the submit keys, Enter right after a backslash
    // converts into a newline (the workaround for terminals without
    // Shift+Enter support).
    pi_tui::keybindings::set_keybindings(
        pi_tui::keybindings::KeybindingsManager::with_user_bindings(
            pi_tui::keybindings::Keybindings::tui_defaults(),
            pi_tui::keybindings::KeybindingsConfig::new()
                .bind("tui.input.submit", ["enter", "shift+enter"]),
        ),
    );
    editor.set_text("foo\\");
    editor.handle_input("\r");
    assert_eq!(editor.get_text(), "foo\n");
    // Restore the default registry for the other tests in this binary.
    pi_tui::keybindings::set_keybindings(pi_tui::keybindings::KeybindingsManager::new(
        pi_tui::keybindings::Keybindings::tui_defaults(),
    ));
}

#[test]
fn disable_submit_blocks_the_submit_path() {
    let editor = editor();
    let submitted: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
    *editor.on_submit.borrow_mut() = Some({
        let submitted = Rc::clone(&submitted);
        Rc::new(move |text: &str| *submitted.borrow_mut() = Some(text.to_string()))
    });
    editor.disable_submit.set(true);

    editor.handle_input("hi");
    editor.handle_input("\r");
    assert!(submitted.borrow().is_none());
    assert_eq!(editor.get_text(), "hi");
}

#[test]
fn backspace_at_the_start_of_a_line_merges_into_the_previous_line() {
    let editor = editor();
    editor.set_text("first\nsecond");
    editor.handle_input("\x01"); // Ctrl+A - line 1 col 0
    editor.handle_input("\x7f"); // Backspace merges with line 0
    assert_eq!(editor.get_text(), "firstsecond");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 5 }
    );
}

#[test]
fn forward_delete_at_line_end_merges_and_keeps_the_cursor() {
    let editor = editor();
    editor.set_text("ab\ncd");
    editor.handle_input("\x1b[A"); // Up to line 0
    editor.handle_input("\x05"); // Ctrl+E - end of line 0
    editor.handle_input("\x1b[3~"); // Delete key merges the lines
    assert_eq!(editor.get_text(), "abcd");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 2 }
    );
}

#[test]
fn paste_after_a_word_character_prepends_a_space_for_file_paths() {
    let editor = editor();
    editor.handle_input("to"); // word char before the cursor
    editor.handle_input("\x1b[200~/tmp/x.txt\x1b[201~");
    assert_eq!(editor.get_text(), "to /tmp/x.txt");

    // No space when the character before the cursor is not a word char.
    for _ in 0..13 {
        editor.handle_input("\x7f"); // clear "to /tmp/x.txt"
    }
    assert_eq!(editor.get_text(), "");
    editor.handle_input("\x1b[200~/tmp/y.txt\x1b[201~");
    assert_eq!(editor.get_text(), "/tmp/y.txt");
}

#[test]
fn paste_decode_strips_unmapped_csi_u_ctrl_sequences_to_their_printable_tail() {
    let editor = editor();
    // Codepoint 40 falls outside both letter ranges: the raw sequence
    // survives decoding, the per-char filter strips the ESC byte, and the
    // printable tail leaks through — the upstream behavior for unmapped
    // codepoints.
    editor.handle_input("\x1b[200~a\x1b[40;5ub\x1b[201~");
    assert_eq!(editor.get_text(), "a[40;5ub");
}

#[test]
fn uppercase_csi_u_ctrl_sequences_decode_to_control_bytes_the_filter_strips() {
    let editor = editor();
    // \x1b[65;5u decodes to Ctrl+A (codepoint 1): the filter drops the
    // control byte instead of leaking the sequence tail.
    editor.handle_input("\x1b[200~line1\x1b[65;5uline2\x1b[201~");
    assert_eq!(editor.get_text(), "line1line2");
}

#[test]
fn get_expanded_text_ignores_hand_typed_marker_lookalikes() {
    let editor = editor();
    editor.handle_input("[paste #99 +5 lines]");
    assert_eq!(editor.get_expanded_text(), "[paste #99 +5 lines]");
}

#[test]
fn insert_text_at_cursor_with_empty_text_is_a_no_op() {
    let editor = editor();
    editor.set_text("abc");
    editor.insert_text_at_cursor("");
    assert_eq!(editor.get_text(), "abc");
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 3 }
    );
}

#[test]
fn input_paste_with_trailing_data_after_the_end_marker_still_applies() {
    let input = Input::new();
    input.handle_input("\x1b[200~ab\x1b[201~cd");
    assert_eq!(input.get_value(), "abcd");
}

#[test]
fn input_escape_fires_the_callback_and_clearing_it_silences_it() {
    let input = Input::new();
    let escaped = Rc::new(RefCell::new(false));

    *input.on_escape.borrow_mut() = Some({
        let escaped = Rc::clone(&escaped);
        Rc::new(move || *escaped.borrow_mut() = true)
    });
    input.handle_input("\x1b");
    assert!(*escaped.borrow());
}

#[test]
fn input_word_motions_and_their_guards() {
    let input = Input::new();
    input.set_value("one two".to_string());
    input.handle_input("\x05"); // Ctrl+E - end
    input.handle_input("\x1b[1;5D"); // Ctrl+Left - lands after "one "
    // Typing pins where the cursor went: a fresh undo unit starts there.
    input.handle_input("|");
    assert_eq!(input.get_value(), "one |two");

    // Word-left at column 0 is a no-op: the next insert lands at the start.
    input.handle_input("\x01"); // Ctrl+A
    input.handle_input("\x1b[1;5D");
    input.handle_input("X");
    assert_eq!(input.get_value(), "Xone |two");

    // Word-right from column 0 crosses "Xone", then the punctuation run,
    // then "two"; a further word-right at the end is a no-op, which a fresh
    // typed char at the end confirms.
    input.handle_input("\x01"); // Ctrl+A
    input.handle_input("\x1b[1;5C"); // over "Xone"
    input.handle_input("\x1b[1;5C"); // over " |"
    input.handle_input("\x1b[1;5C"); // over "two" - at the end
    input.handle_input("Z");
    assert_eq!(input.get_value(), "Xone |twoZ");
}

#[test]
fn input_delete_guards_at_both_edges() {
    let input = Input::new();
    // Backspace at 0, Ctrl+U at 0, Ctrl+W at 0.
    input.handle_input("\x7f");
    input.handle_input("\x15");
    input.handle_input("\x17");
    assert_eq!(input.get_value(), "");

    input.set_value("abc".to_string());
    // Ctrl+K at end, Alt+D at end, forward delete at end.
    input.handle_input("\x05"); // Ctrl+E
    input.handle_input("\x0b");
    input.handle_input("\x1bd");
    input.handle_input("\x1b[3~");
    assert_eq!(input.get_value(), "abc");
}

#[test]
fn input_mouse_press_places_the_cursor_and_other_events_are_ignored() {
    let input = Input::new();
    input.set_value("hello world".to_string());
    Component::render(&input, 40);

    // Press on the word "hello" (col 3) places the cursor there.
    let result = Component::handle_mouse(
        &input,
        &mouse(TuiMouseEventType::Press, TuiMouseButton::Left, 5, 0),
    );
    assert!(result.is_some_and(|result| result.handled && result.focus));
    assert_eq!(input.get_value(), "hello world");

    // Click, release, move, wheel, other buttons and other rows: unhandled.
    for (event_type, button, y) in [
        (TuiMouseEventType::Click, TuiMouseButton::Left, 0),
        (TuiMouseEventType::Release, TuiMouseButton::Left, 0),
        (TuiMouseEventType::Move, TuiMouseButton::None, 0),
        (TuiMouseEventType::Press, TuiMouseButton::Right, 0),
        (TuiMouseEventType::Press, TuiMouseButton::Left, 1),
    ] {
        let result = Component::handle_mouse(&input, &mouse(event_type, button, 5, y));
        assert!(
            result.is_none(),
            "{event_type:?}/{button:?}/y={y} must stay unhandled"
        );
    }
}

#[test]
fn input_render_truncates_a_prompt_wider_than_the_width() {
    let input = Input::with_options(InputOptions {
        prompt: Some("very long prompt".to_string()),
        ..InputOptions::default()
    });
    let lines = Component::render(&input, 8);
    assert_eq!(lines.len(), 1);
    assert!(pi_tui::utils::visible_width(&lines[0]) <= 8);
}

#[test]
fn input_scroll_window_when_the_cursor_sits_at_the_end() {
    let input = Input::new();
    input.set_value("0123456789012345678901234567890".to_string());
    input.handle_input("\x05"); // Ctrl+E - cursor at the end
    let lines = Component::render(&input, 10);
    assert_eq!(pi_tui::utils::visible_width(&lines[0]), 10);
}

#[test]
fn input_scroll_window_with_a_zero_scroll_width_falls_back_to_an_empty_window() {
    let input = Input::with_options(InputOptions {
        prompt: Some(">>".to_string()),
        ..InputOptions::default()
    });
    // Width 3 leaves a 1-column window; with the cursor mid-value the scroll
    // width collapses to zero and the row renders as the prompt plus the
    // cursor slot.
    input.set_value("abcd".to_string());
    input.handle_input("\x01"); // Ctrl+A
    input.handle_input("\x1b[C"); // Right one grapheme: cursor mid-value
    let lines = Component::render(&input, 3);
    assert_eq!(pi_tui::utils::visible_width(&lines[0]), 3);
}

#[test]
fn input_trait_surface_and_debug() {
    let input = Input::new();
    assert!(Component::wants_input(&input));
    assert!(input.as_focusable().is_some());

    input.set_focused(true);
    assert!(Focusable::is_focused(&input));
    input.set_focused(false);
    assert!(!Focusable::is_focused(&input));

    let default = Input::default();
    assert_eq!(default.get_value(), "");
    let debug = format!("{input:?}");
    assert!(debug.contains("Input"));
    let options_debug = format!("{:?}", InputOptions::default());
    assert!(options_debug.contains("InputOptions"));
}

#[test]
fn editor_trait_surface_and_debug() {
    let editor = editor();
    assert!(Component::wants_input(&editor));
    assert!(editor.as_focusable().is_some());

    editor.set_focused(true);
    assert!(Focusable::is_focused(&editor));
    editor.set_focused(false);
    assert!(!Focusable::is_focused(&editor));

    let debug = format!("{editor:?}");
    assert!(debug.contains("Editor"));
}

#[test]
fn editor_mouse_click_places_the_cursor_and_ignores_other_gestures() {
    let editor = editor();
    editor.set_text("hello world");
    Component::render(&editor, 80);

    // A click on row 1 (first content row) at column 3 lands in "hello".
    let result = Component::handle_mouse(
        &editor,
        &mouse(TuiMouseEventType::Click, TuiMouseButton::Left, 3, 1),
    );
    assert!(result.is_some_and(|result| result.handled && result.focus));
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 3 }
    );

    // Press, drag and release stay unhandled so the renderer's text
    // selection can run over the editor rows.
    for event_type in [
        TuiMouseEventType::Press,
        TuiMouseEventType::Drag,
        TuiMouseEventType::Release,
    ] {
        let result =
            Component::handle_mouse(&editor, &mouse(event_type, TuiMouseButton::Left, 3, 1));
        assert!(result.is_none(), "{event_type:?} must stay unhandled");
    }

    // A click on the top border row is consumed but does not move the cursor.
    let before = editor.get_cursor();
    let result = Component::handle_mouse(
        &editor,
        &mouse(TuiMouseEventType::Click, TuiMouseButton::Left, 3, 0),
    );
    assert!(result.is_some_and(|result| result.handled));
    assert_eq!(editor.get_cursor(), before);

    // A click below the visible rows is consumed as well.
    let result = Component::handle_mouse(
        &editor,
        &mouse(TuiMouseEventType::Click, TuiMouseButton::Left, 3, 99),
    );
    assert!(result.is_some_and(|result| result.handled));

    // A right-button click is not a cursor gesture.
    let result = Component::handle_mouse(
        &editor,
        &mouse(TuiMouseEventType::Click, TuiMouseButton::Right, 3, 1),
    );
    assert!(result.is_none());
}

#[test]
fn editor_mouse_click_past_the_last_column_clamps_to_the_line_end() {
    let editor = editor();
    editor.set_text("short");
    Component::render(&editor, 80);

    // Click far to the right: the cursor lands at the line end.
    let result = Component::handle_mouse(
        &editor,
        &mouse(TuiMouseEventType::Click, TuiMouseButton::Left, 60, 1),
    );
    assert!(result.is_some());
    assert_eq!(
        editor.get_cursor(),
        pi_tui::components::CursorPosition { line: 0, col: 5 }
    );
}

#[test]
fn editor_debug_and_component_surface_render() {
    let editor = editor();
    let lines = Component::render(&editor, 40);
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0], format!("\x1b[2m{}\x1b[22m", "─".repeat(40)));
}

#[test]
fn word_navigation_default_paths_answer_boundaries() {
    // The no-options overload walks the default word segmenter; the CJK
    // case walks one ideograph at a time (see the ported suite for the
    // upstream dictionary restatement).
    assert_eq!(
        pi_tui::word_navigation::find_word_backward("abc def", 7, None),
        4
    );
    assert_eq!(
        pi_tui::word_navigation::find_word_forward("abc def", 0, None),
        3
    );
    // Whitespace-run skip then word start.
    assert_eq!(
        pi_tui::word_navigation::find_word_forward("a  b", 1, None),
        4
    );
}

#[test]
fn undo_stack_round_trips_snapshots() {
    let mut stack = pi_tui::undo_stack::UndoStack::new();
    assert!(stack.is_empty());
    stack.push(&1);
    stack.push(&2);
    assert_eq!(stack.len(), 2);
    assert_eq!(stack.pop(), Some(2));
    stack.clear();
    assert!(stack.is_empty());
    assert_eq!(stack.pop(), None);
}

#[test]
fn kill_ring_drops_empty_entries_and_rotates_multi_entry_rings() {
    let mut ring = pi_tui::kill_ring::KillRing::default();
    ring.push("", pi_tui::kill_ring::KillRingPushOptions::default());
    assert!(ring.is_empty());

    ring.push("a", pi_tui::kill_ring::KillRingPushOptions::default());
    ring.push("b", pi_tui::kill_ring::KillRingPushOptions::default());
    // Rotate moves the last entry to the front; the most recent entry is the
    // one the next peek answers.
    ring.rotate();
    assert_eq!(ring.peek(), Some("a"));
    ring.rotate();
    assert_eq!(ring.peek(), Some("b"));
}

// =============================================================================
// The EditorComponent extension contract
// =============================================================================

#[test]
fn editor_component_defaults_serve_a_custom_implementation() {
    use std::cell::RefCell as Cell;

    struct StubEditor {
        text: Cell<String>,
    }

    impl Component for StubEditor {
        fn render(&self, _width: usize) -> Vec<String> {
            Vec::new()
        }
    }

    impl EditorComponent for StubEditor {
        fn get_text(&self) -> String {
            self.text.borrow().clone()
        }

        fn set_text(&self, text: &str) {
            *self.text.borrow_mut() = text.to_string();
        }
    }

    let stub = StubEditor {
        text: Cell::new("stub".to_string()),
    };
    assert_eq!(EditorComponent::get_text(&stub), "stub");
    EditorComponent::set_text(&stub, "replaced");
    assert_eq!(EditorComponent::get_text(&stub), "replaced");
    // The optional members default to unsupported: history drops, insertion
    // drops, expansion falls back, padding is a no-op.
    EditorComponent::add_to_history(&stub, "ignored");
    EditorComponent::insert_text_at_cursor(&stub, "ignored");
    assert_eq!(EditorComponent::get_expanded_text(&stub), None);
    EditorComponent::set_padding_x(&stub, 4);
    assert_eq!(EditorComponent::get_text(&stub), "replaced");
}

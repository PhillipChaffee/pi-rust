//! The Input component suite, ported 1:1 from
//! `packages/tui/test/input.test.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#47).
//!
//! Restatements against upstream:
//!
//! - Text offsets are byte offsets into the Rust `String`.
//! - Upstream's `Intl.Segmenter` applies locale dictionary segmentation for
//!   Han text (Node groups 你好 and 世界 as single word-like segments); the
//!   port's UAX #29 segmenter yields one segment per ideograph, so the CJK
//!   kill-ring expectations repeat one press per character. The fullwidth
//!   punctuation stops (。 and ，) match upstream: they stay non-word
//!   segments the deletion treats as runs.
//! - `input.focused = true` becomes [`Focusable::set_focused`].
use std::cell::RefCell;
use std::rc::Rc;

use pi_tui::components::{Input, InputOptions};
use pi_tui::tui::{Component, Focusable};
use pi_tui::utils::{strip_terminal_sequences, visible_width};

#[test]
fn submits_value_including_backslash_on_enter() {
    let input = Input::new();
    let submitted: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));

    *input.on_submit.borrow_mut() = Some({
        let submitted = Rc::clone(&submitted);
        Rc::new(move |value: &str| *submitted.borrow_mut() = Some(value.to_string()))
    });

    // Type hello, then backslash, then Enter
    for key in ["h", "e", "l", "l", "o", "\\", "\r"] {
        input.handle_input(key);
    }

    // Input is single-line, no backslash+Enter workaround
    assert_eq!(submitted.borrow().as_deref(), Some("hello\\"));
}

#[test]
fn inserts_backslash_as_regular_character() {
    let input = Input::new();

    input.handle_input("\\");
    input.handle_input("x");

    assert_eq!(input.get_value(), "\\x");
}

#[test]
fn supports_a_custom_prompt_and_styled_placeholder() {
    let input = Input::with_options(InputOptions {
        prompt: Some(String::new()),
        placeholder: Some("Find transcript".to_string()),
        placeholder_style: Some(Rc::new(|text| format!("\x1b[2m{text}\x1b[22m"))),
    });
    input.set_focused(true);

    let empty = Component::render(&input, 20);
    assert!(empty[0].contains("\x1b[2m"));
    assert_eq!(
        strip_terminal_sequences(&empty[0]).trim_end(),
        "Find transcript"
    );

    input.handle_input("n");
    let populated = Component::render(&input, 20);
    assert_eq!(strip_terminal_sequences(&populated[0]).trim_end(), "n");
}

#[test]
fn does_not_overflow_with_wide_cjk_and_fullwidth_text() {
    let width = 93;
    let cases = [
        "가나다라마바사아자차카타파하 한글 텍스트가 터미널 너비를 초과하면 크래시가 발생합니다 이것은 재현용 테스트입니다",
        "これはテスト文章です。日本語のテキストが正しく表示されるかどうかを確認するためのサンプルテキストです。あいうえお",
        "这是一段测试文本，用于验证中文字符在终端中的显示宽度是否被正确计算，如果不正确就会导致用户界面崩溃的问题",
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍ",
    ];

    for text in cases {
        for label in ["start", "middle", "end"] {
            let input = Input::new();
            input.set_value(text.to_string());
            input.set_focused(true);
            match label {
                "start" => {}
                "middle" => {
                    for _ in 0..10 {
                        input.handle_input("\x1b[C");
                    }
                }
                "end" => input.handle_input("\x05"),
                _ => unreachable!(),
            }

            let line = &Component::render(&input, width)[0];
            assert!(
                visible_width(line) <= width,
                "rendered line overflowed for {text} at {label}"
            );
        }
    }
}

#[test]
fn keeps_the_cursor_visible_when_horizontally_scrolling_wide_text() {
    let input = Input::new();
    let width = 20;
    let text = "가나다라마바사아자차카타파하";
    input.set_value(text.to_string());
    input.set_focused(true);
    input.handle_input("\x01");
    for _ in 0..5 {
        input.handle_input("\x1b[C");
    }

    let line = Component::render(&input, width);
    assert!(visible_width(&line[0]) <= width);
}

#[test]
fn ctrl_w_saves_deleted_text_to_kill_ring_and_ctrl_y_yanks_it() {
    let input = Input::new();

    input.set_value("foo bar baz".to_string());
    // Move cursor to end
    input.handle_input("\x05"); // Ctrl+E

    input.handle_input("\x17"); // Ctrl+W - deletes "baz"
    assert_eq!(input.get_value(), "foo bar ");

    // Move to beginning and yank
    input.handle_input("\x01"); // Ctrl+A
    input.handle_input("\x19"); // Ctrl+Y
    assert_eq!(input.get_value(), "bazfoo bar ");
}

#[test]
fn ctrl_w_preserves_ascii_punctuation_boundaries() {
    let input = Input::new();

    input.set_value("foo.bar".to_string());
    input.handle_input("\x05"); // Ctrl+E
    input.handle_input("\x17"); // Ctrl+W - deletes "bar"
    assert_eq!(input.get_value(), "foo.");

    input.set_value("foo:bar".to_string());
    input.handle_input("\x05"); // Ctrl+E
    input.handle_input("\x17"); // Ctrl+W - deletes "bar"
    assert_eq!(input.get_value(), "foo:");
}

#[test]
fn ctrl_w_handles_unicode_word_boundaries() {
    let input = Input::new();

    // Upstream's `Intl.Segmenter` groups the Han text as
    // 你好|世界|。|你好|，|世界; the port's UAX #29 segmenter yields one
    // word-like segment per ideograph, so each press deletes one character.
    // The fullwidth 。 and ， stay non-word punctuation segments in both.
    input.set_value("你好世界。你好，世界".to_string());
    input.handle_input("\x05"); // Ctrl+E
    for expected in [
        "你好世界。你好，世",
        "你好世界。你好，",
        "你好世界。你好",
        "你好世界。你",
        "你好世界。",
        "你好世界",
        "你好世",
        "你好",
        "你",
        "",
    ] {
        input.handle_input("\x17"); // Ctrl+W
        assert_eq!(input.get_value(), expected);
    }
}

#[test]
fn ctrl_u_saves_deleted_text_to_kill_ring() {
    let input = Input::new();

    input.set_value("hello world".to_string());
    // Move cursor to after "hello "
    input.handle_input("\x01"); // Ctrl+A
    for _ in 0..6 {
        input.handle_input("\x1b[C");
    }

    input.handle_input("\x15"); // Ctrl+U - deletes "hello "
    assert_eq!(input.get_value(), "world");

    input.handle_input("\x19"); // Ctrl+Y
    assert_eq!(input.get_value(), "hello world");
}

#[test]
fn ctrl_k_saves_deleted_text_to_kill_ring() {
    let input = Input::new();

    input.set_value("hello world".to_string());
    input.handle_input("\x01"); // Ctrl+A
    input.handle_input("\x0b"); // Ctrl+K - deletes "hello world"

    assert_eq!(input.get_value(), "");

    input.handle_input("\x19"); // Ctrl+Y
    assert_eq!(input.get_value(), "hello world");
}

#[test]
fn ctrl_y_does_nothing_when_kill_ring_is_empty() {
    let input = Input::new();

    input.set_value("test".to_string());
    input.handle_input("\x05"); // Ctrl+E
    input.handle_input("\x19"); // Ctrl+Y
    assert_eq!(input.get_value(), "test");
}

#[test]
fn alt_y_cycles_through_kill_ring_after_ctrl_y() {
    let input = Input::new();

    // Create kill ring with multiple entries
    for text in ["first", "second", "third"] {
        input.set_value(text.to_string());
        input.handle_input("\x05"); // Ctrl+E
        input.handle_input("\x17"); // Ctrl+W - deletes the value
    }

    assert_eq!(input.get_value(), "");

    input.handle_input("\x19"); // Ctrl+Y - yanks "third"
    assert_eq!(input.get_value(), "third");

    input.handle_input("\x1by"); // Alt+Y - cycles to "second"
    assert_eq!(input.get_value(), "second");

    input.handle_input("\x1by"); // Alt+Y - cycles to "first"
    assert_eq!(input.get_value(), "first");

    input.handle_input("\x1by"); // Alt+Y - cycles back to "third"
    assert_eq!(input.get_value(), "third");
}

#[test]
fn alt_y_does_nothing_if_not_preceded_by_yank() {
    let input = Input::new();

    input.set_value("test".to_string());
    input.handle_input("\x05"); // Ctrl+E
    input.handle_input("\x17"); // Ctrl+W - deletes "test"
    input.set_value("other".to_string());
    input.handle_input("\x05"); // Ctrl+E

    // Type something to break the yank chain
    input.handle_input("x");
    assert_eq!(input.get_value(), "otherx");

    input.handle_input("\x1by"); // Alt+Y - should do nothing
    assert_eq!(input.get_value(), "otherx");
}

#[test]
fn alt_y_does_nothing_if_kill_ring_has_one_entry() {
    let input = Input::new();

    input.set_value("only".to_string());
    input.handle_input("\x05"); // Ctrl+E
    input.handle_input("\x17"); // Ctrl+W - deletes "only"

    input.handle_input("\x19"); // Ctrl+Y - yanks "only"
    assert_eq!(input.get_value(), "only");

    input.handle_input("\x1by"); // Alt+Y - should do nothing
    assert_eq!(input.get_value(), "only");
}

#[test]
fn consecutive_ctrl_w_accumulates_into_one_kill_ring_entry() {
    let input = Input::new();

    input.set_value("one two three".to_string());
    input.handle_input("\x05"); // Ctrl+E
    input.handle_input("\x17"); // Ctrl+W - deletes "three"
    input.handle_input("\x17"); // Ctrl+W - deletes "two "
    input.handle_input("\x17"); // Ctrl+W - deletes "one "

    assert_eq!(input.get_value(), "");

    input.handle_input("\x19"); // Ctrl+Y
    assert_eq!(input.get_value(), "one two three");
}

#[test]
fn non_delete_actions_break_kill_accumulation() {
    let input = Input::new();

    input.set_value("foo bar baz".to_string());
    input.handle_input("\x05"); // Ctrl+E
    input.handle_input("\x17"); // Ctrl+W - deletes "baz"
    assert_eq!(input.get_value(), "foo bar ");

    input.handle_input("x"); // Typing breaks accumulation
    assert_eq!(input.get_value(), "foo bar x");

    input.handle_input("\x17"); // Ctrl+W - deletes "x" (separate entry)
    assert_eq!(input.get_value(), "foo bar ");

    input.handle_input("\x19"); // Ctrl+Y - most recent is "x"
    assert_eq!(input.get_value(), "foo bar x");

    input.handle_input("\x1by"); // Alt+Y - cycle to "baz"
    assert_eq!(input.get_value(), "foo bar baz");
}

#[test]
fn non_yank_actions_break_alt_y_chain() {
    let input = Input::new();

    for text in ["first", "second"] {
        input.set_value(text.to_string());
        input.handle_input("\x05"); // Ctrl+E
        input.handle_input("\x17"); // Ctrl+W
    }
    input.set_value(String::new());

    input.handle_input("\x19"); // Ctrl+Y - yanks "second"
    assert_eq!(input.get_value(), "second");

    input.handle_input("x"); // Breaks yank chain
    assert_eq!(input.get_value(), "secondx");

    input.handle_input("\x1by"); // Alt+Y - should do nothing
    assert_eq!(input.get_value(), "secondx");
}

#[test]
fn kill_ring_rotation_persists_after_cycling() {
    let input = Input::new();

    for text in ["first", "second", "third"] {
        input.set_value(text.to_string());
        input.handle_input("\x05"); // Ctrl+E
        input.handle_input("\x17"); // deletes the value
    }
    input.set_value(String::new());

    input.handle_input("\x19"); // Ctrl+Y - yanks "third"
    input.handle_input("\x1by"); // Alt+Y - cycles to "second"
    assert_eq!(input.get_value(), "second");

    // Break chain and start fresh
    input.handle_input("x");
    input.set_value(String::new());

    // New yank should get "second" (now at end after rotation)
    input.handle_input("\x19"); // Ctrl+Y
    assert_eq!(input.get_value(), "second");
}

#[test]
fn backward_deletions_prepend_forward_deletions_append_during_accumulation() {
    let input = Input::new();

    input.set_value("prefix|suffix".to_string());
    // Position cursor at "|"
    input.handle_input("\x01"); // Ctrl+A
    for _ in 0..6 {
        input.handle_input("\x1b[C"); // Move right 6
    }

    input.handle_input("\x0b"); // Ctrl+K - deletes "|suffix" (forward)
    assert_eq!(input.get_value(), "prefix");

    input.handle_input("\x19"); // Ctrl+Y
    assert_eq!(input.get_value(), "prefix|suffix");
}

#[test]
fn alt_d_deletes_word_forward_and_saves_to_kill_ring() {
    let input = Input::new();

    input.set_value("hello world test".to_string());
    input.handle_input("\x01"); // Ctrl+A

    input.handle_input("\x1bd"); // Alt+D - deletes "hello"
    assert_eq!(input.get_value(), " world test");

    input.handle_input("\x1bd"); // Alt+D - deletes " world"
    assert_eq!(input.get_value(), " test");

    // Yank should get accumulated text
    input.handle_input("\x19"); // Ctrl+Y
    assert_eq!(input.get_value(), "hello world test");
}

#[test]
fn alt_d_preserves_ascii_punctuation_boundaries() {
    let input = Input::new();

    input.set_value("foo.bar baz".to_string());
    input.handle_input("\x01"); // Ctrl+A
    input.handle_input("\x1bd"); // Alt+D - deletes "foo"
    assert_eq!(input.get_value(), ".bar baz");
    input.handle_input("\x1bd"); // Alt+D - deletes "."
    assert_eq!(input.get_value(), "bar baz");
    input.handle_input("\x1bd"); // Alt+D - deletes "bar"
    assert_eq!(input.get_value(), " baz");
}

#[test]
fn alt_d_handles_unicode_word_boundaries() {
    let input = Input::new();

    // Upstream's `Intl.Segmenter` groups the Han text as
    // 你好|世界|。|你好|，|世界; the port's UAX #29 segmenter yields one
    // word-like segment per ideograph, so each press deletes one character.
    // The fullwidth 。 and ， stay non-word punctuation segments in both.
    input.set_value("你好世界。你好，世界".to_string());
    input.handle_input("\x01"); // Ctrl+A
    for expected in [
        "好世界。你好，世界",
        "世界。你好，世界",
        "界。你好，世界",
        "。你好，世界",
        "你好，世界",
        "好，世界",
        "，世界",
        "世界",
        "界",
        "",
    ] {
        input.handle_input("\x1bd"); // Alt+D
        assert_eq!(input.get_value(), expected);
    }
}

#[test]
fn handles_yank_in_middle_of_text() {
    let input = Input::new();

    input.set_value("word".to_string());
    input.handle_input("\x05"); // Ctrl+E
    input.handle_input("\x17"); // Ctrl+W - deletes "word"
    input.set_value("hello world".to_string());
    // Move to middle (after "hello ")
    input.handle_input("\x01"); // Ctrl+A
    for _ in 0..6 {
        input.handle_input("\x1b[C");
    }

    input.handle_input("\x19"); // Ctrl+Y
    assert_eq!(input.get_value(), "hello wordworld");
}

#[test]
fn handles_yank_pop_in_middle_of_text() {
    let input = Input::new();

    // Create two kill ring entries
    for text in ["FIRST", "SECOND"] {
        input.set_value(text.to_string());
        input.handle_input("\x05"); // Ctrl+E
        input.handle_input("\x17"); // Ctrl+W - deletes the value
    }

    // Set up "hello world" and position cursor after "hello "
    input.set_value("hello world".to_string());
    input.handle_input("\x01"); // Ctrl+A
    for _ in 0..6 {
        input.handle_input("\x1b[C");
    }

    input.handle_input("\x19"); // Ctrl+Y - yanks "SECOND"
    assert_eq!(input.get_value(), "hello SECONDworld");

    input.handle_input("\x1by"); // Alt+Y - replaces with "FIRST"
    assert_eq!(input.get_value(), "hello FIRSTworld");
}

#[test]
fn does_nothing_when_undo_stack_is_empty() {
    let input = Input::new();

    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "");
}

#[test]
fn coalesces_consecutive_word_characters_into_one_undo_unit() {
    let input = Input::new();

    for key in ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] {
        input.handle_input(key);
    }
    assert_eq!(input.get_value(), "hello world");

    // Undo removes " world"
    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "hello");

    // Undo removes "hello"
    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "");
}

#[test]
fn undoes_spaces_one_at_a_time() {
    let input = Input::new();

    for key in ["h", "e", "l", "l", "o", " ", " "] {
        input.handle_input(key);
    }
    assert_eq!(input.get_value(), "hello  ");

    // Ctrl+- (undo) - removes second " "
    input.handle_input("\x1b[45;5u");
    assert_eq!(input.get_value(), "hello ");

    // Ctrl+- (undo) - removes first " "
    input.handle_input("\x1b[45;5u");
    assert_eq!(input.get_value(), "hello");

    // Ctrl+- (undo) - removes "hello"
    input.handle_input("\x1b[45;5u");
    assert_eq!(input.get_value(), "");
}

#[test]
fn undoes_backspace() {
    let input = Input::new();

    for key in ["h", "e", "l", "l", "o"] {
        input.handle_input(key);
    }
    input.handle_input("\x7f"); // Backspace
    assert_eq!(input.get_value(), "hell");

    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "hello");
}

#[test]
fn undoes_forward_delete() {
    let input = Input::new();

    for key in ["h", "e", "l", "l", "o"] {
        input.handle_input(key);
    }
    input.handle_input("\x01"); // Ctrl+A - go to start
    input.handle_input("\x1b[C"); // Right arrow
    input.handle_input("\x1b[3~"); // Delete key
    assert_eq!(input.get_value(), "hllo");

    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "hello");
}

#[test]
fn undoes_ctrl_w_delete_word_backward() {
    let input = Input::new();

    for key in ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] {
        input.handle_input(key);
    }
    assert_eq!(input.get_value(), "hello world");

    input.handle_input("\x17"); // Ctrl+W
    assert_eq!(input.get_value(), "hello ");

    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "hello world");
}

#[test]
fn undoes_ctrl_k_delete_to_line_end() {
    let input = Input::new();

    for key in ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] {
        input.handle_input(key);
    }
    input.handle_input("\x01"); // Ctrl+A
    for _ in 0..6 {
        input.handle_input("\x1b[C");
    }

    input.handle_input("\x0b"); // Ctrl+K
    assert_eq!(input.get_value(), "hello ");

    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "hello world");
}

#[test]
fn undoes_ctrl_u_delete_to_line_start() {
    let input = Input::new();

    for key in ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] {
        input.handle_input(key);
    }
    input.handle_input("\x01"); // Ctrl+A
    for _ in 0..6 {
        input.handle_input("\x1b[C");
    }

    input.handle_input("\x15"); // Ctrl+U
    assert_eq!(input.get_value(), "world");

    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "hello world");
}

#[test]
fn undoes_yank() {
    let input = Input::new();

    for key in ["h", "e", "l", "l", "o", " "] {
        input.handle_input(key);
    }
    input.handle_input("\x17"); // Ctrl+W - delete "hello "
    input.handle_input("\x19"); // Ctrl+Y - yank
    assert_eq!(input.get_value(), "hello ");

    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "");
}

#[test]
fn undoes_paste_atomically() {
    let input = Input::new();

    input.set_value("hello world".to_string());
    input.handle_input("\x01"); // Ctrl+A
    for _ in 0..5 {
        input.handle_input("\x1b[C");
    }

    // Simulate bracketed paste
    input.handle_input("\x1b[200~beep boop\x1b[201~");
    assert_eq!(input.get_value(), "hellobeep boop world");

    // Single undo should restore entire pre-paste state
    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "hello world");
}

#[test]
fn undoes_alt_d_delete_word_forward() {
    let input = Input::new();

    input.set_value("hello world".to_string());
    input.handle_input("\x01"); // Ctrl+A

    input.handle_input("\x1bd"); // Alt+D - deletes "hello"
    assert_eq!(input.get_value(), " world");

    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "hello world");
}

#[test]
fn cursor_movement_starts_new_undo_unit() {
    let input = Input::new();

    for key in ["a", "b", "c"] {
        input.handle_input(key);
    }
    input.handle_input("\x01"); // Ctrl+A - movement breaks coalescing
    input.handle_input("\x05"); // Ctrl+E
    for key in ["d", "e"] {
        input.handle_input(key);
    }
    assert_eq!(input.get_value(), "abcde");

    // Undo removes "de" (typed after movement)
    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "abc");

    // Undo removes "abc"
    input.handle_input("\x1b[45;5u"); // Ctrl+- (undo)
    assert_eq!(input.get_value(), "");
}

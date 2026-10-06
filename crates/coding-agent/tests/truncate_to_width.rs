//! The truncate-to-width suite, upstream's
//! `test/truncate-to-width.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The suite lives in the coding-agent's test directory upstream and drives
//! pi-tui's renderer helpers, so the port drives the pi-tui crate over the
//! same cases.

use pi_tui::utils::{truncate_to_width, visible_width};

#[test]
fn truncates_messages_with_unicode_characters_correctly() {
    // This message contains a checkmark (✔) which may have display width > 1 byte
    let message = "✔ script to run › dev $ concurrently \"vite\" \"node --import tsx ./";
    let width = 67;
    let max_msg_width = width - 2; // Account for cursor

    let truncated = truncate_to_width(message, max_msg_width, "", false);
    let truncated_width = visible_width(&truncated);

    assert!(truncated_width <= max_msg_width);
}

#[test]
fn handles_emoji_characters() {
    let message = "🎉 Celebration! 🚀 Launch 📦 Package ready for deployment now";
    let width = 40;
    let max_msg_width = width - 2;

    let truncated = truncate_to_width(message, max_msg_width, "", false);
    let truncated_width = visible_width(&truncated);

    assert!(truncated_width <= max_msg_width);
}

#[test]
fn handles_wide_cjk_characters() {
    let message = "日本語のテキストはディスプレイ幅が2バイト分です";
    let width = 20;
    let max_msg_width = width - 2;

    let truncated = truncate_to_width(message, max_msg_width, "", false);
    let truncated_width = visible_width(&truncated);

    assert!(truncated_width <= max_msg_width);
}

#[test]
fn handles_combining_characters() {
    // é as e + combining acute accent
    let message = "cafe\u{0301} with combining characters that should not break truncation";
    let width = 30;
    let max_msg_width = width - 2;

    let truncated = truncate_to_width(message, max_msg_width, "", false);
    let truncated_width = visible_width(&truncated);

    assert!(truncated_width <= max_msg_width);
}

#[test]
fn handles_box_drawing_characters() {
    let message = "┌──────────────────────────────────────┐";
    let width = 25;
    let max_msg_width = width - 2;

    let truncated = truncate_to_width(message, max_msg_width, "", false);
    let truncated_width = visible_width(&truncated);

    assert!(truncated_width <= max_msg_width);
}

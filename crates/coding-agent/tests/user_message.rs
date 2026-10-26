//! The `user-message.test.ts` suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported 1:1.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;

use pi_coding_agent::extensions::types::MarkdownTransformContext;
use pi_coding_agent::modes::interactive::components::user_message::UserMessageComponent;
use pi_coding_agent::modes::interactive::theme::theme::init_theme;
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::tui::Component;

const OSC133_ZONE_START: &str = "\u{1b}]133;A\u{7}";
const OSC133_ZONE_END: &str = "\u{1b}]133;B\u{7}";
const OSC133_ZONE_FINAL: &str = "\u{1b}]133;C\u{7}";
const BG_RESET: &str = "\u{1b}[49m";

#[test]
fn keeps_user_message_height_stable_while_moving_closing_osc_markers_off_line_end() {
    init_theme(Some("dark"), false);
    let component = UserMessageComponent::new("hello", None, 1, Arc::new(Vec::new()));
    let lines = component.render(20);

    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains(OSC133_ZONE_START));
    assert!(lines[0].ends_with(BG_RESET));
    assert!(!lines[0].contains(OSC133_ZONE_END));
    assert!(lines[1].contains("hello"));
    assert!(lines[2].starts_with(&format!("{OSC133_ZONE_END}{OSC133_ZONE_FINAL}")));
    assert!(lines[2].ends_with(BG_RESET));
}

#[test]
fn chains_markdown_transformers_with_user_message_context() {
    init_theme(Some("dark"), false);
    let calls = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let formula_calls = Arc::clone(&calls);
    let component = UserMessageComponent::new(
        "The input is $x^2$.",
        None,
        1,
        Arc::new(vec![
            Arc::new(move |markdown: &str, context: &MarkdownTransformContext| {
                formula_calls
                    .lock()
                    .expect("lock")
                    .push("formula".to_string());
                assert_eq!(context.message_type.as_str(), "user");
                assert!(!context.is_streaming);
                assert_eq!(context.available_width, 78);
                markdown.replace("$x^2$", "x²")
            }),
            Arc::new(|markdown: &str, _context: &MarkdownTransformContext| {
                format!("{markdown} Done.")
            }),
        ]),
    );

    assert!(strip_ansi(&component.render(80).join("\n")).contains("The input is x². Done."));
    assert_eq!(*calls.lock().expect("lock"), vec!["formula".to_string()]);
}

#[test]
fn reapplies_markdown_transformers_when_invalidated() {
    init_theme(Some("dark"), false);
    let suffix = Arc::new(std::sync::Mutex::new("before".to_string()));
    let seen = Arc::clone(&suffix);
    let component = UserMessageComponent::new(
        "Message",
        None,
        1,
        Arc::new(vec![Arc::new(
            move |markdown: &str, _context: &MarkdownTransformContext| {
                format!("{markdown} {}", seen.lock().expect("lock"))
            },
        )]),
    );

    assert!(strip_ansi(&component.render(80).join("\n")).contains("Message before"));
    *suffix.lock().expect("lock") = "after".to_string();
    component.invalidate();
    assert!(strip_ansi(&component.render(80).join("\n")).contains("Message after"));
}

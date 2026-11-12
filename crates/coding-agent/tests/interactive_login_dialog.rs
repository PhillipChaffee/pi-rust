//! The login-dialog suite, upstream's `login-dialog.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (no upstream test file), plus
//! boundary cases.
//!
//! The browser launcher is injected so the auth screen renders without
//! opening a browser; prompt outcomes are the port's oneshot receivers.

#![expect(
    clippy::expect_used,
    reason = "tests pin outcomes; an unexpected result panics by design"
)]

use std::cell::RefCell;
use std::rc::Rc;

use pi_ai::auth::types::AuthInfoLink;
use pi_coding_agent::modes::interactive::components::login_dialog::{
    LoginCancelled, LoginDialogComponent,
};
use pi_coding_agent::modes::interactive::theme::theme::init_theme;
use pi_coding_agent::provider_composer::OAuthDeviceCodeInfo;
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::tui::{Component, Tui, TuiConfig};
use pi_tui::utils::visible_width;

type Completion = (bool, Option<String>);

type Harness = (
    Rc<LoginDialogComponent>,
    Rc<RefCell<Vec<Completion>>>,
    Rc<RefCell<Vec<String>>>,
);

fn new_dialog() -> Harness {
    init_theme(None, false);
    let completions: Rc<RefCell<Vec<Completion>>> = Rc::new(RefCell::new(Vec::new()));
    let opened: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let tui: Rc<Tui> = Tui::new(TuiConfig::default());

    let on_complete = {
        let completions = Rc::clone(&completions);
        Rc::new(move |success: bool, message: Option<&str>| {
            completions
                .borrow_mut()
                .push((success, message.map(str::to_string)));
        })
    };
    let open = {
        let opened = Rc::clone(&opened);
        Rc::new(move |url: &str| opened.borrow_mut().push(url.to_string()))
    };
    let dialog = LoginDialogComponent::new_with_open_browser(
        tui,
        "test-provider",
        on_complete,
        Some("Test Provider"),
        None,
        open,
    );
    (Rc::new(dialog), completions, opened)
}

fn rendered(dialog: &LoginDialogComponent, width: usize) -> String {
    dialog
        .render(width)
        .iter()
        .map(|line| strip_ansi(line))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn title_and_borders_render_the_provider_name() {
    let (dialog, _, _) = new_dialog();
    let lines = dialog.render(40);
    assert_eq!(strip_ansi(&lines[0]), "─".repeat(40));
    assert!(strip_ansi(&lines[1]).contains("Login to Test Provider"));
    assert_eq!(
        strip_ansi(lines.last().expect("bottom border")),
        "─".repeat(40)
    );
}

#[test]
fn title_override_replaces_the_generated_title() {
    init_theme(None, false);
    let tui: Rc<Tui> = Tui::new(TuiConfig::default());
    let dialog =
        LoginDialogComponent::new(tui, "provider", Rc::new(|_, _| {}), None, Some("Sign in"));
    assert!(rendered(&dialog, 40).contains("Sign in"));
}

#[test]
fn auth_url_instructions_and_button_hint_render() {
    let (dialog, _, opened) = new_dialog();
    dialog.show_auth(
        "https://example.com/auth",
        Some("Paste the verification code"),
    );

    let text = rendered(&dialog, 60);
    assert!(text.contains("https://example.com/auth"));
    assert!(text.contains("Paste the verification code"));
    assert!(text.contains("click to open"));
    assert_eq!(opened.borrow().as_slice(), ["https://example.com/auth"]);
}

#[test]
fn device_code_renders_url_and_code_without_opening_a_browser() {
    let (dialog, _, opened) = new_dialog();
    dialog.show_device_code(&OAuthDeviceCodeInfo {
        user_code: "ABCD-1234".to_string(),
        verification_uri: "https://example.com/device".to_string(),
        interval_seconds: Some(5),
        expires_in_seconds: Some(900),
    });

    let text = rendered(&dialog, 60);
    assert!(text.contains("https://example.com/device"));
    assert!(text.contains("Enter code: ABCD-1234"));
    assert!(opened.borrow().is_empty());
}

#[test]
fn manual_input_settles_with_the_typed_value() {
    let (dialog, _, _) = new_dialog();
    let mut receiver = dialog.show_manual_input("Paste the code");
    assert!(rendered(&dialog, 60).contains("Paste the code"));

    for character in "abc".chars() {
        dialog.handle_input(&character.to_string());
    }
    dialog.handle_input("\r");

    assert!(matches!(
        receiver.try_recv(),
        Ok(Ok(value)) if value == "abc"
    ));
    assert!(rendered(&dialog, 60).contains("> abc"));
}

#[test]
fn prompt_keeps_existing_content_and_settles() {
    let (dialog, _, _) = new_dialog();
    dialog.show_auth("https://example.com/auth", None);
    let mut receiver = dialog.show_prompt("Enter the code", Some("123456"));

    let text = rendered(&dialog, 60);
    assert!(text.contains("https://example.com/auth"));
    assert!(text.contains("Enter the code"));
    assert!(text.contains("e.g., 123456"));

    for character in "xyz".chars() {
        dialog.handle_input(&character.to_string());
    }
    dialog.handle_input("\r");
    assert!(matches!(
        receiver.try_recv(),
        Ok(Ok(value)) if value == "xyz"
    ));
}

#[test]
fn escape_cancels_the_pending_prompt_and_the_signal() {
    let (dialog, completions, _) = new_dialog();
    let mut receiver = dialog.show_prompt("Enter the code", None);
    let signal = dialog.signal();
    assert!(!signal.is_cancelled());

    dialog.handle_input("\x1b");

    assert!(matches!(receiver.try_recv(), Ok(Err(LoginCancelled))));
    assert!(signal.is_cancelled());
    assert_eq!(
        completions.borrow().as_slice(),
        [(false, Some("Login cancelled".to_string()))]
    );
}

#[test]
fn info_links_and_close_hint_render() {
    let (dialog, _, _) = new_dialog();
    dialog.show_info(
        "Provider is configured elsewhere.",
        &[AuthInfoLink {
            url: "https://example.com/docs".to_string(),
            label: Some("Docs".to_string()),
        }],
        true,
    );

    let text = rendered(&dialog, 60);
    assert!(text.contains("Provider is configured elsewhere."));
    assert!(text.contains("Docs: https://example.com/docs"));
    assert!(text.contains("to close"));
}

#[test]
fn waiting_and_progress_lines_append() {
    let (dialog, _, _) = new_dialog();
    dialog.show_waiting("Waiting for authentication...");
    dialog.show_progress("Still waiting...");

    let text = rendered(&dialog, 60);
    assert!(text.contains("Waiting for authentication..."));
    assert!(text.contains("Still waiting..."));
}

#[test]
fn long_content_keeps_every_line_within_width() {
    let (dialog, _, _) = new_dialog();
    dialog.show_info(&"wide ".repeat(40), &[], false);

    for width in [1, 10, 20, 40] {
        for line in dialog.render(width) {
            assert!(
                visible_width(&line) <= width,
                "width {width}: {:?}",
                strip_ansi(&line)
            );
        }
    }
}

#[test]
fn submitted_text_replaces_the_input_child() {
    let (dialog, _, _) = new_dialog();
    let _receiver = dialog.show_prompt("Enter the code", None);
    for character in "replaced".chars() {
        dialog.handle_input(&character.to_string());
    }
    dialog.handle_input("\r");

    let text = rendered(&dialog, 60);
    assert!(text.contains("> replaced"));
}

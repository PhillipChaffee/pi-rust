//! Boundary coverage for the interactive components whose public helper
//! surface the frame suites do not reach directly: the custom editor's
//! submit/change knobs, the login dialog's display and prompt paths, the
//! status indicator mutators, and the reusable settings submenu.

#![expect(
    clippy::expect_used,
    reason = "the component fixtures pin callbacks and values; a failure is a test bug"
)]

mod common;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use common::interactive::Harness;
use common::interactive::HarnessOptions;
use pi_ai::auth::types::AuthInfoLink;
use pi_coding_agent::core::keybindings::AppKeybindingsManager;
use pi_coding_agent::modes::interactive::components::custom_editor::{
    CustomEditor, CustomEditorOptions,
};
use pi_coding_agent::modes::interactive::components::login_dialog::LoginDialogComponent;
use pi_coding_agent::modes::interactive::components::settings_submenu::{
    SelectSubmenu, SelectSubmenuOptions,
};
use pi_coding_agent::modes::interactive::components::status_indicator::{
    StatusIndicator, StatusIndicatorKind,
};
use pi_coding_agent::modes::interactive::theme::theme::{get_editor_theme, init_theme};
use pi_coding_agent::provider_composer::OAuthDeviceCodeInfo;
use pi_tui::components::ColorFn;
use pi_tui::components::editor::EditorOptions;
use pi_tui::components::select_list::SelectItem;
use pi_tui::tui::{Component, Focusable, TuiStopOptions};

#[tokio::test]
async fn custom_editor_submit_change_disable_and_shortcut_paths() {
    init_theme(Some("dark"), false);
    let harness = Harness::new(HarnessOptions::default()).await;
    let tui = harness.frame.tui();
    let keybindings = Rc::new(AppKeybindingsManager::create(std::path::Path::new(
        &harness.agent_dir,
    )));
    let editor = CustomEditor::with_options(
        &tui,
        get_editor_theme(),
        keybindings,
        CustomEditorOptions {
            editor: EditorOptions::default(),
            embed_working_status: false,
        },
    );
    let submitted = Rc::new(RefCell::new(Vec::new()));
    let changed = Rc::new(RefCell::new(Vec::new()));
    {
        let submitted = Rc::clone(&submitted);
        editor.set_on_submit(Some(Rc::new(move |text: &str| {
            submitted.borrow_mut().push(text.to_owned());
        })));
    }
    {
        let changed = Rc::clone(&changed);
        editor.set_on_change(Some(Rc::new(move |text: &str| {
            changed.borrow_mut().push(text.to_owned());
        })));
    }
    editor.set_disable_submit(true);
    editor.handle_input("hello");
    editor.handle_input("\r");
    assert!(submitted.borrow().is_empty(), "submit is disabled");
    assert!(!changed.borrow().is_empty(), "change still fires");
    editor.set_disable_submit(false);
    editor.handle_input("\r");
    assert_eq!(submitted.borrow().as_slice(), ["hello".to_owned()]);

    let shortcut = Rc::new(RefCell::new(0));
    let shortcut_for_handler = Rc::clone(&shortcut);
    editor.set_on_extension_shortcut(Some(Rc::new(move |_data: &str| {
        *shortcut_for_handler.borrow_mut() += 1;
        true
    })));
    editor.handle_input("x");
    assert_eq!(*shortcut.borrow(), 1);
    editor.set_on_extension_shortcut(None);

    let action = Rc::new(RefCell::new(0));
    let action_for_handler = Rc::clone(&action);
    editor.on_action(
        "app.clear",
        Rc::new(move || *action_for_handler.borrow_mut() += 1),
    );
    editor.handle_input("\x03");
    assert_eq!(*action.borrow(), 1);
    let _ = editor.border_color();
    editor.set_border_color(Rc::new(|text: &str| text.to_owned()));
}

#[tokio::test]
async fn login_dialog_display_prompts_cancel_and_signal() {
    init_theme(Some("dark"), false);
    let harness = Harness::new(HarnessOptions::default()).await;
    let opened = Rc::new(RefCell::new(0));
    let opened_for_browser = Rc::clone(&opened);
    let dialog = LoginDialogComponent::new_with_open_browser(
        harness.frame.tui(),
        "provider",
        Rc::new(|_success, _message| {}),
        Some("Provider Name"),
        Some("Custom Title"),
        Rc::new(move |_url| *opened_for_browser.borrow_mut() += 1),
    );
    assert!(format!("{dialog:?}").contains("LoginDialogComponent"));
    dialog.show_auth("https://example.test/auth", Some("instructions"));
    assert_eq!(*opened.borrow(), 1);
    dialog.show_auth("https://example.test/auth", None);
    dialog.show_device_code(&OAuthDeviceCodeInfo {
        user_code: "ABCD".to_owned(),
        verification_uri: "https://example.test/device".to_owned(),
        interval_seconds: Some(5),
        expires_in_seconds: Some(60),
    });
    dialog.show_info(
        "info",
        &[AuthInfoLink {
            url: "https://example.test/info".to_owned(),
            label: Some("label".to_owned()),
        }],
        true,
    );
    dialog.show_details(&["one", "two"]);
    dialog.show_waiting("waiting");
    dialog.show_progress("progress");
    let _ = dialog.render(80);

    let receiver = dialog.show_prompt("Prompt", Some("placeholder"));
    dialog.handle_input("typed");
    dialog.handle_input("\r");
    assert_eq!(receiver.await.expect("prompt"), Ok("typed".to_owned()));

    let receiver = dialog.show_manual_input("Code");
    dialog.handle_input("123456");
    dialog.handle_input("\r");
    assert_eq!(receiver.await.expect("manual"), Ok("123456".to_owned()));

    let receiver = dialog.show_prompt("Cancel", None);
    dialog.handle_input("\x1b");
    assert!(receiver.await.expect("cancel").is_err());
    assert!(dialog.signal().is_cancelled());
    assert!(!dialog.is_focused());
    dialog.set_focused(true);
    assert!(dialog.is_focused());
}

#[tokio::test]
async fn status_indicator_message_and_indicator_mutators() {
    init_theme(Some("dark"), false);
    let harness = Harness::new(HarnessOptions::default()).await;
    let color: ColorFn = Arc::new(|text: &str| text.to_owned());
    let indicator = StatusIndicator::new(
        StatusIndicatorKind::Working,
        harness.frame.tui().render_request(),
        Arc::clone(&color),
        color,
        "Working".to_owned(),
        None,
    );
    assert_eq!(indicator.kind(), StatusIndicatorKind::Working);
    indicator.set_message("Still working".to_owned());
    indicator.set_indicator(None);
    let rendered = indicator.render(80);
    assert!(!rendered.is_empty());
    indicator.dispose();
}

#[tokio::test]
async fn select_submenu_selects_filters_and_cancels() {
    init_theme(Some("dark"), false);
    let selected = Rc::new(RefCell::new(Vec::new()));
    let cancelled = Rc::new(RefCell::new(0));
    let selected_for_cb = Rc::clone(&selected);
    let cancelled_for_cb = Rc::clone(&cancelled);
    let submenu = SelectSubmenu::new(
        "Theme",
        "Pick a theme",
        vec![
            SelectItem {
                value: "dark".to_owned(),
                label: "Dark".to_owned(),
                description: None,
            },
            SelectItem {
                value: "light".to_owned(),
                label: "Light".to_owned(),
                description: None,
            },
        ],
        "dark",
        Rc::new(move |value: &str| selected_for_cb.borrow_mut().push(value.to_owned())),
        Rc::new(move || *cancelled_for_cb.borrow_mut() += 1),
        None,
        Some(&SelectSubmenuOptions {
            searchable: true,
            layout: None,
        }),
    );
    submenu.handle_input("l");
    submenu.handle_input("\r");
    assert_eq!(selected.borrow().as_slice(), ["light".to_owned()]);
    submenu.handle_input("\x1b");
    assert_eq!(*cancelled.borrow(), 1);
    let _ = submenu.render(80);
    let _ = submenu.children();
    submenu.invalidate();

    let plain = SelectSubmenu::new(
        "Plain",
        "",
        Vec::new(),
        "",
        Rc::new(|_value| {}),
        Rc::new(|| {}),
        None,
        None,
    );
    plain.handle_input("\x1b");
}

#[tokio::test]
async fn tui_stop_is_idempotent_for_component_teardown() {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.tui().start();
    harness.frame.tui().stop(TuiStopOptions::default());
    harness.frame.tui().stop(TuiStopOptions::default());
}

//! Coverage of the interactive extension surface: every `UiCallHandler`
//! method, the dialog round trips, the pre-aborted arms, the shortcut path,
//! and the custom-editor swap.
//!
//! The wire methods settle through `drain_ui_requests`, the run loop's UI
//! half; the dialogs answer through the recording terminal the harness
//! starts. The `gh`/Radius child-process legs live in
//! `interactive_coverage_share.rs`.

#![expect(
    clippy::expect_used,
    reason = "the fixtures pin the provider, model, and extension shapes"
)]
#![allow(
    clippy::large_futures,
    reason = "the acceptance suites await one frame operation at a time"
)]

mod common;

use std::sync::Arc;

use common::interactive::{Harness, HarnessOptions, ResourceFixtures};
use pi_coding_agent::extensions::host::UiCallHandler;
use serde_json::{Value, json};

async fn harness() -> Harness {
    let harness = Harness::new(HarnessOptions::default()).await;
    harness.start_tui();
    harness
}

/// Run one wire call that settles without a dialog.
async fn call(
    harness: &Harness,
    method: &str,
    params: Option<Value>,
) -> Result<Value, pi_coding_agent::extensions::host::HostCallError> {
    let future = harness.frame.ui_context().call(method, params);
    harness.frame.drain_ui_requests();
    future.await
}

#[tokio::test]
async fn selects_a_value_and_reports_a_cancel_as_null() {
    let harness = harness().await;
    let ui = harness.frame.ui_context();

    let selected = ui.call(
        "select",
        Some(json!({ "title": "Pick one", "options": ["Alpha", "Beta"] })),
    );
    harness.frame.drain_ui_requests();
    harness.send_input("\r");
    assert_eq!(selected.await.expect("select"), json!("Alpha"));

    let cancelled = ui.call(
        "select",
        Some(json!({ "title": "Pick one", "options": ["Alpha"] })),
    );
    harness.frame.drain_ui_requests();
    harness.send_input("\x1b");
    assert_eq!(cancelled.await.expect("cancel"), Value::Null);
}

#[tokio::test]
async fn positions_the_selector_before_confirming() {
    let harness = harness().await;
    let ui = harness.frame.ui_context();

    let yes = ui.call(
        "confirm",
        Some(json!({ "title": "Proceed?", "message": "All good" })),
    );
    harness.frame.drain_ui_requests();
    harness.send_input("\r");
    assert_eq!(yes.await.expect("yes"), json!(true));

    let no = ui.call(
        "confirm",
        Some(json!({ "title": "Proceed?", "message": "All good" })),
    );
    harness.frame.drain_ui_requests();
    harness.send_input("\x1b[B");
    harness.send_input("\r");
    assert_eq!(no.await.expect("no"), json!(false));

    let cancelled = ui.call(
        "confirm",
        Some(json!({ "title": "Proceed?", "message": "All good" })),
    );
    harness.frame.drain_ui_requests();
    harness.send_input("\x1b");
    assert_eq!(cancelled.await.expect("cancel"), json!(false));
}

#[tokio::test]
async fn reads_text_from_the_input_and_editor_dialogs() {
    let harness = harness().await;
    let ui = harness.frame.ui_context();

    let input = ui.call(
        "input",
        Some(json!({ "title": "Name", "placeholder": "a name" })),
    );
    harness.frame.drain_ui_requests();
    harness.send_input("hello");
    harness.send_input("\r");
    assert_eq!(input.await.expect("input"), json!("hello"));

    let cancelled = ui.call("input", Some(json!({ "title": "Name" })));
    harness.frame.drain_ui_requests();
    harness.send_input("\x1b");
    assert_eq!(cancelled.await.expect("cancel"), Value::Null);

    let editor = ui.call("editor", Some(json!({ "title": "Body", "prefill": "old" })));
    harness.frame.drain_ui_requests();
    harness.send_input("!");
    harness.send_input("\r");
    assert_eq!(editor.await.expect("editor"), json!("old!"));

    let cancelled = ui.call("editor", Some(json!({ "title": "Body" })));
    harness.frame.drain_ui_requests();
    harness.send_input("\x1b");
    assert_eq!(cancelled.await.expect("cancel"), Value::Null);
}

#[tokio::test]
async fn dialog_timeouts_report_the_remaining_seconds_and_cancel_at_zero() {
    let harness = harness().await;

    let select = harness.frame.ui_context().call(
        "select",
        Some(json!({ "title": "Pick", "options": ["A", "B"], "timeout": 1000 })),
    );
    harness.frame.drain_ui_requests();
    let selector = harness.frame.extension_selector().expect("selector");
    let countdown = selector.countdown().expect("countdown armed");
    countdown.tick();
    assert!(select.await.expect("expired select").is_null());

    let input = harness
        .frame
        .ui_context()
        .call("input", Some(json!({ "title": "Type", "timeout": 1000 })));
    harness.frame.drain_ui_requests();
    let prompt = harness.frame.extension_input().expect("input prompt");
    let countdown = prompt.countdown().expect("countdown armed");
    prompt.handle_input("typed");
    assert_eq!(prompt.value(), "typed");
    countdown.tick();
    assert_eq!(input.await.expect("expired input"), Value::Null);
}

#[tokio::test]
async fn aborted_dialogs_settle_without_mounting() {
    let harness = harness().await;
    let context = pi_agent_core::harness::context::placeholder_context();
    let (_, controller) = pi_agent_core::harness::context::with_cancel(&context);
    controller.abort_without_reason();
    let signal = controller.signal().clone();
    let ui = harness.frame.ui_context();

    let select = ui.select_with_signal("Pick".to_owned(), vec!["A".to_owned()], signal.clone());
    harness.frame.drain_ui_requests();
    assert_eq!(select.await, None);
    assert!(harness.frame.extension_selector().is_none());

    let confirm =
        ui.confirm_with_signal("Confirm".to_owned(), "Message".to_owned(), signal.clone());
    harness.frame.drain_ui_requests();
    assert!(!confirm.await);

    let input = ui.input_with_signal("Name".to_owned(), None, signal);
    harness.frame.drain_ui_requests();
    assert_eq!(input.await, None);
}

#[tokio::test]
async fn notifies_with_every_severity_and_rejects_missing_messages() {
    let harness = harness().await;

    assert_eq!(
        call(&harness, "notify", Some(json!({ "message": "heads up" })))
            .await
            .expect("info"),
        Value::Null
    );
    assert!(harness.chat_text().contains("heads up"));

    call(
        &harness,
        "notify",
        Some(json!({ "message": "careful", "type": "warning" })),
    )
    .await
    .expect("warning");
    assert!(harness.chat_text().contains("Warning: careful"));

    call(
        &harness,
        "notify",
        Some(json!({ "message": "broke", "type": "error" })),
    )
    .await
    .expect("error");
    assert!(harness.chat_text().contains("Error: broke"));

    let error = call(&harness, "notify", Some(json!({ "type": "info" })))
        .await
        .expect_err("missing message");
    assert_eq!(error.code(), "invalid_params");
}

#[tokio::test]
async fn sets_and_clears_footer_status_entries() {
    let harness = harness().await;

    assert_eq!(
        call(
            &harness,
            "setStatus",
            Some(json!({ "key": "build", "text": "ok" }))
        )
        .await
        .expect("set"),
        Value::Null
    );
    call(&harness, "setStatus", Some(json!({ "key": "build" })))
        .await
        .expect("clear");
    let error = call(&harness, "setStatus", Some(json!({ "text": "no key" })))
        .await
        .expect_err("missing key");
    assert_eq!(error.code(), "invalid_params");
}

#[tokio::test]
async fn writes_the_terminal_title() {
    let harness = harness().await;
    call(&harness, "setTitle", Some(json!({ "title": "my session" })))
        .await
        .expect("title");
    assert!(harness.terminal.writes().contains("my session"));
    let error = call(&harness, "setTitle", Some(json!({})))
        .await
        .expect_err("missing title");
    assert_eq!(error.code(), "invalid_params");
}

#[tokio::test]
async fn pastes_and_replaces_the_editor_text() {
    let harness = harness().await;

    call(&harness, "pasteToEditor", Some(json!({ "text": "pasted" })))
        .await
        .expect("paste");
    assert_eq!(harness.frame.editor().get_text(), "pasted");
    call(
        &harness,
        "setEditorText",
        Some(json!({ "text": "replaced" })),
    )
    .await
    .expect("replace");
    assert_eq!(harness.frame.editor().get_text(), "replaced");

    let error = call(&harness, "pasteToEditor", Some(json!({})))
        .await
        .expect_err("missing paste text");
    assert_eq!(error.code(), "invalid_params");
    let error = call(&harness, "setEditorText", Some(json!({})))
        .await
        .expect_err("missing editor text");
    assert_eq!(error.code(), "invalid_params");
}

#[tokio::test]
async fn reads_the_editor_text_and_tool_expansion() {
    let harness = harness().await;
    harness.frame.editor().set_text("current");

    assert_eq!(
        call(&harness, "getEditorText", Some(json!({})))
            .await
            .expect("editor text"),
        json!("current")
    );
    assert_eq!(
        call(&harness, "getToolsExpanded", Some(json!({})))
            .await
            .expect("collapsed"),
        json!(false)
    );
    call(
        &harness,
        "setToolsExpanded",
        Some(json!({ "expanded": true })),
    )
    .await
    .expect("expand");
    assert_eq!(
        call(&harness, "getToolsExpanded", Some(json!({})))
            .await
            .expect("expanded"),
        json!(true)
    );
    assert!(harness.chat_text().contains("Tool output: expanded"));
}

#[tokio::test]
async fn applies_theme_settings_and_lists_the_registry() {
    let harness = harness().await;

    let result = call(&harness, "setTheme", Some(json!({ "name": "light" })))
        .await
        .expect("theme");
    assert!(result["success"].as_bool().expect("theme result"));

    let result = call(&harness, "setTheme", Some(json!({ "theme": "dark" })))
        .await
        .expect("alternate theme key");
    assert!(result["success"].as_bool().expect("theme result"));

    assert!(
        call(&harness, "setTheme", Some(json!({ "name": "__missing__" })))
            .await
            .expect("invalid theme")["success"]
            .as_bool()
            .is_some_and(|success| !success)
    );
    let error = call(&harness, "setTheme", Some(json!({})))
        .await
        .expect_err("missing name");
    assert_eq!(error.code(), "invalid_params");

    let themes = call(&harness, "getAllThemes", Some(json!({})))
        .await
        .expect("themes");
    assert!(themes.as_array().is_some_and(|themes| !themes.is_empty()));

    assert_eq!(
        call(&harness, "getTheme", Some(json!({ "name": "dark" })))
            .await
            .expect("theme"),
        json!(true)
    );
    assert_eq!(
        call(&harness, "getTheme", Some(json!({ "name": "__missing__" })))
            .await
            .expect("missing theme"),
        json!(false)
    );
    let error = call(&harness, "getTheme", Some(json!({}))).await;
    assert_eq!(error.expect_err("missing theme").code(), "invalid_params");
}

#[tokio::test]
async fn sets_working_visibility_and_the_hidden_thinking_label() {
    let harness = harness().await;

    call(
        &harness,
        "setWorkingMessage",
        Some(json!({ "message": "Working hard" })),
    )
    .await
    .expect("message");
    call(
        &harness,
        "setWorkingVisible",
        Some(json!({ "visible": false })),
    )
    .await
    .expect("hide");
    harness.frame.set_working_visible(true);
    assert_eq!(harness.frame.active_status_kind(), None);
    let _ = call(&harness, "setHiddenThinkingLabel", Some(json!({}))).await;
}

#[tokio::test]
async fn sets_widgets_on_both_placements_and_clears_them() {
    let harness = harness().await;
    for (key, placement) in [("above", "aboveEditor"), ("below", "belowEditor")] {
        call(
            &harness,
            "setWidget",
            Some(json!({
                "key": key,
                "content": ["one", "two"],
                "placement": placement
            })),
        )
        .await
        .expect("widget");
    }
    let long: Vec<String> = (0..12).map(|index| format!("line {index}")).collect();
    call(
        &harness,
        "setWidget",
        Some(json!({ "key": "above", "content": long })),
    )
    .await
    .expect("long widget");
    call(
        &harness,
        "setWidget",
        Some(json!({ "key": "above", "content": Value::Null })),
    )
    .await
    .expect("clear widget");
    let error = call(&harness, "setWidget", Some(json!({ "content": ["x"] })))
        .await
        .expect_err("missing key");
    assert_eq!(error.code(), "invalid_params");
}

#[tokio::test]
async fn refuses_factory_methods_and_unknown_methods() {
    let harness = harness().await;
    for method in [
        "onTerminalInput",
        "custom",
        "setFooter",
        "setHeader",
        "setEditorComponent",
        "getEditorComponent",
        "addAutocompleteProvider",
    ] {
        let error = call(&harness, method, Some(json!({})))
            .await
            .expect_err(method);
        assert_eq!(error.code(), "unsupported", "{method}");
    }
    let error = call(&harness, "frobnicate", None)
        .await
        .expect_err("unknown");
    assert_eq!(error.code(), "unknown_method");
}

#[tokio::test]
async fn extension_shortcuts_fire_and_reset_with_no_registrations() {
    let fired = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let fired_for_factory = Arc::clone(&fired);
    let fixtures = ResourceFixtures {
        extension_factories: vec![(
            Some("shortcut".to_owned()),
            false,
            Arc::new(move |api| {
                let fired = Arc::clone(&fired_for_factory);
                Box::pin(async move {
                    api.register_shortcut(
                        "ctrl+x".to_owned(),
                        Some("fire".to_owned()),
                        Arc::new(move |_context| {
                            let fired = Arc::clone(&fired);
                            Box::pin(async move {
                                fired.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            })
                        }),
                    )?;
                    Ok(())
                })
            }),
        )],
        ..ResourceFixtures::default()
    };
    let harness = Harness::new(HarnessOptions {
        fixtures: Some(fixtures),
        ..HarnessOptions::default()
    })
    .await;
    assert!(
        !harness
            .frame
            .session()
            .extension_runner()
            .get_shortcuts()
            .is_empty(),
        "the inline factory registered its shortcut"
    );
    harness.frame.setup_extension_shortcuts();
    let editor: std::rc::Rc<dyn pi_tui::tui::Component> = harness.frame.editor();
    harness.frame.tui().set_focus(Some(editor));
    harness.start_tui();
    harness.send_input("\x18");
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert_eq!(fired.load(std::sync::atomic::Ordering::SeqCst), 1);
    harness.frame.setup_extension_shortcuts();
}

#[tokio::test]
async fn reload_hides_every_extension_dialog() {
    let harness = harness().await;
    let ui = harness.frame.ui_context();

    let selector = ui.call("select", Some(json!({ "title": "s", "options": ["a"] })));
    harness.frame.drain_ui_requests();
    drop(selector);
    let input = ui.call("input", Some(json!({ "title": "i" })));
    harness.frame.drain_ui_requests();
    drop(input);
    let editor = ui.call("editor", Some(json!({ "title": "e" })));
    harness.frame.drain_ui_requests();
    drop(editor);
    assert!(harness.frame.extension_editor().is_some());

    harness.frame.submit_text("/reload").await;
    harness.frame.drain_commands().await;
    assert!(harness.frame.extension_selector().is_none());
    assert!(harness.frame.extension_input().is_none());
    assert!(harness.frame.extension_editor().is_none());
}

#[tokio::test]
async fn custom_editor_swap_copies_text_settings_and_focus() {
    use std::rc::Rc;

    use pi_coding_agent::modes::interactive::interactive_mode::{AppEditor, EditorFactory};

    let harness = Harness::new(HarnessOptions::default()).await;
    harness.frame.editor().set_text("carried over");
    let editor = Rc::new(common::interactive::TestEditor::new("CUSTOM", true));
    let editor_for_factory = Rc::clone(&editor);
    let factory: EditorFactory = Rc::new(move |_tui, _theme, _keybindings| {
        let editor: Rc<dyn AppEditor> = editor_for_factory.clone();
        editor
    });
    harness.frame.set_custom_editor_component(Some(factory));
    assert_eq!(harness.frame.editor().get_text(), "carried over");
    assert!(harness.frame.editor().embed_working_status());
    harness.frame.set_custom_editor_component(None);
    assert_eq!(harness.frame.editor().get_text(), "carried over");
}

#[tokio::test]
async fn ui_context_debug_surface() {
    let harness = Harness::new(HarnessOptions::default()).await;
    let ui = harness.frame.ui_context();
    assert!(format!("{ui:?}").contains("InteractiveUiContext"));
}

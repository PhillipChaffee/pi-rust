//! Boundary tests binding the keybindings branches the 1:1 suites leave
//! untested, at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::fs;

use pi_coding_agent::core::keybindings::{
    AppKeybindingsManager, keybindings_config_from_value, keybindings_with,
    migrate_keybindings_config, use_windows_keybindings,
};
use pi_tui::keybindings::TUI_KEYBINDINGS;
use serde_json::json;

use common::env_with;

/// The default keys bound to `action`, joined for compact assertions.
fn keys(registry: &pi_tui::keybindings::Keybindings, action: &str) -> String {
    registry
        .definition(action)
        .expect("the action is registered")
        .default_keys
        .join(",")
}

#[test]
fn wsl_detection_treats_empty_values_as_absent() {
    assert!(!use_windows_keybindings(
        "linux",
        &env_with(&[("WSL_DISTRO_NAME", ""), ("WSL_INTEROP", "")])
    ));
    assert!(!use_windows_keybindings("freebsd", &env_with(&[])));
}

#[test]
fn registry_order_is_the_tui_table_then_the_app_table() {
    let registry = keybindings_with("linux", &env_with(&[]));
    let actual: Vec<&str> = registry.action_ids().collect();
    let tui: Vec<&str> = TUI_KEYBINDINGS.iter().map(|entry| entry.id).collect();

    assert_eq!(
        &actual[..tui.len()],
        tui.as_slice(),
        "override rows keep their tui position"
    );
    let app = &actual[tui.len()..];
    assert_eq!(app.len(), 43);
    assert_eq!(app.first(), Some(&"app.interrupt"));
    assert_eq!(app.last(), Some(&"app.tree.filter.cycleBackward"));
    assert_eq!(app[16], "app.clipboard.pasteImage");
    assert_eq!(app[17], "app.session.new");
    assert_eq!(app[36], "app.tree.filter.default");
}

#[test]
fn platform_specific_defaults_cover_every_arm() {
    let linux = keybindings_with("linux", &env_with(&[]));
    assert_eq!(keys(&linux, "tui.editor.undo"), "ctrl+-");
    assert_eq!(keys(&linux, "app.suspend"), "ctrl+z");
    assert_eq!(keys(&linux, "app.tree.foldOrUp"), "ctrl+left,alt+left");
    assert_eq!(
        keys(&linux, "app.tree.unfoldOrDown"),
        "ctrl+right,alt+right"
    );
    assert_eq!(keys(&linux, "app.model.cycleBackward"), "shift+ctrl+p");
    assert_eq!(keys(&linux, "app.message.followUp"), "alt+enter");
    assert_eq!(keys(&linux, "app.message.dequeue"), "alt+up");
    assert_eq!(keys(&linux, "app.clipboard.pasteImage"), "ctrl+v");
    assert_eq!(
        keys(&linux, "tui.altScreen.previousPrompt"),
        "ctrl+shift+up,ctrl+up"
    );
    assert_eq!(
        keys(&linux, "tui.altScreen.nextPrompt"),
        "ctrl+shift+down,ctrl+down"
    );
    assert_eq!(keys(&linux, "tui.altScreen.search"), "ctrl+shift+f");

    let windows = keybindings_with("win32", &env_with(&[]));
    assert_eq!(keys(&windows, "tui.editor.undo"), "ctrl+z");
    assert_eq!(keys(&windows, "app.suspend"), "");
    assert_eq!(keys(&windows, "app.model.cycleBackward"), "alt+p");
    assert_eq!(keys(&windows, "app.message.followUp"), "ctrl+q");
    assert_eq!(keys(&windows, "app.message.dequeue"), "alt+q");
    assert_eq!(keys(&windows, "app.clipboard.pasteImage"), "alt+v");
    assert_eq!(keys(&windows, "tui.altScreen.previousPrompt"), "ctrl+up");
    assert_eq!(keys(&windows, "tui.altScreen.nextPrompt"), "ctrl+down");
    assert_eq!(keys(&windows, "tui.altScreen.search"), "ctrl+f");

    let wsl = keybindings_with("linux", &env_with(&[("WSL_INTEROP", "/run/WSL/1")]));
    assert_eq!(keys(&wsl, "tui.editor.undo"), "alt+z");
    assert_eq!(keys(&wsl, "app.suspend"), "ctrl+z");

    let darwin = keybindings_with("darwin", &env_with(&[]));
    assert_eq!(keys(&darwin, "app.tree.foldOrUp"), "alt+left,ctrl+left");
    assert_eq!(
        keys(&darwin, "app.tree.unfoldOrDown"),
        "alt+right,ctrl+right"
    );
}

#[test]
fn config_projection_keeps_strings_and_string_arrays_only() {
    let raw = json!({
        "app.clear": "ctrl+x",
        "app.exit": ["ctrl+d", "ctrl+e"],
        "app.suspend": ["ctrl+z", 7],
        "app.interrupt": null,
        "app.tools.expand": {"key": "ctrl+o"},
    });
    let config = keybindings_config_from_value(raw.as_object().expect("object"));

    let entries: Vec<(&str, Vec<String>)> = config
        .iter()
        .map(|(action, keys)| (action, keys.to_vec()))
        .collect();
    assert_eq!(
        entries,
        vec![
            ("app.clear", vec!["ctrl+x".to_string()]),
            ("app.exit", vec!["ctrl+d".to_string(), "ctrl+e".to_string()]),
        ]
    );
}

#[test]
fn migration_orders_known_actions_and_sorts_extras() {
    let raw = json!({
        "zeta": 1,
        "app.tools.expand": "ctrl+x",
        "alpha": 2,
        "tui.editor.cursorUp": "up",
    });

    let (migrated, names_rewritten) = migrate_keybindings_config(raw.as_object().expect("object"));

    assert!(!names_rewritten, "no legacy names are present");
    let keys: Vec<&str> = migrated.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        vec!["tui.editor.cursorUp", "app.tools.expand", "alpha", "zeta"]
    );
}

#[test]
fn migration_renames_legacy_keys_and_keeps_any_value() {
    let raw = json!({"expandTools": 7, "cursorUp": ["up"]});

    let (migrated, names_rewritten) = migrate_keybindings_config(raw.as_object().expect("object"));

    assert!(names_rewritten);
    let keys: Vec<&str> = migrated.keys().map(String::as_str).collect();
    assert_eq!(keys, vec!["tui.editor.cursorUp", "app.tools.expand"]);
    assert_eq!(migrated["app.tools.expand"], json!(7));
}

#[test]
fn manager_reads_missing_malformed_non_object_and_bom_files() {
    let missing = tempfile::tempdir().expect("scratch dir");
    let manager = AppKeybindingsManager::create(missing.path());
    assert_eq!(manager.get_user_bindings().len(), 0);

    let malformed = tempfile::tempdir().expect("scratch dir");
    fs::write(malformed.path().join("keybindings.json"), "{invalid").expect("write");
    assert_eq!(
        AppKeybindingsManager::create(malformed.path())
            .get_user_bindings()
            .len(),
        0
    );

    let array = tempfile::tempdir().expect("scratch dir");
    fs::write(array.path().join("keybindings.json"), "[1,2]").expect("write");
    assert_eq!(
        AppKeybindingsManager::create(array.path())
            .get_user_bindings()
            .len(),
        0,
        "a non-object root reads as no config"
    );

    let bom = tempfile::tempdir().expect("scratch dir");
    fs::write(
        bom.path().join("keybindings.json"),
        "\u{FEFF}{\"app.clear\": \"ctrl+x\"}",
    )
    .expect("write");
    let config = AppKeybindingsManager::create(bom.path()).get_user_bindings();
    assert_eq!(config.get("app.clear"), Some(&["ctrl+x".to_string()][..]));
}

#[test]
fn manager_reload_rereads_the_file_and_falls_back_to_defaults() {
    let agent_dir = tempfile::tempdir().expect("scratch dir");
    let path = agent_dir.path().join("keybindings.json");
    fs::write(&path, json!({"app.tools.expand": "ctrl+x"}).to_string()).expect("write");
    let mut manager = AppKeybindingsManager::create(agent_dir.path());
    let effective = |manager: &AppKeybindingsManager| {
        manager
            .get_effective_config()
            .get("app.tools.expand")
            .expect("resolved")
            .join(",")
    };
    assert_eq!(effective(&manager), "ctrl+x");

    fs::write(&path, json!({"app.tools.expand": "ctrl+y"}).to_string()).expect("write");
    manager.reload();
    assert_eq!(effective(&manager), "ctrl+y");

    fs::remove_file(&path).expect("remove");
    manager.reload();
    assert_eq!(effective(&manager), "ctrl+o", "the default returns");
    assert_eq!(
        manager.manager().get_keys("app.tools.expand"),
        vec!["ctrl+o".to_string()]
    );
}

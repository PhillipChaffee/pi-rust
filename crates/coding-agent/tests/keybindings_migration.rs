//! The `keybindings-migration.test.ts` suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported 1:1.
//!
//! Upstream threaded `PI_CODING_AGENT_DIR` through `process.env` and passed
//! the agent dir to `runMigrations`; the port's `_in` seams take the
//! directory directly.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::cell::RefCell;
use std::fs;
use std::path::Path;

use pi_coding_agent::core::keybindings::{AppKeybindingsManager, migrate_keybindings_config_value};
use pi_coding_agent::migrations::run_migrations_with_in;
use serde_json::json;
use tempfile::TempDir;

/// A scratch agent dir with `keybindings.json` written the way upstream's
/// fixture writes it: pretty JSON plus a trailing newline.
fn create_agent_dir(config: &serde_json::Value) -> TempDir {
    let agent_dir = tempfile::Builder::new()
        .prefix("pi-keybindings-test-")
        .tempdir()
        .expect("scratch dir");
    fs::write(
        agent_dir.path().join("keybindings.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(config).expect("serialize the fixture")
        ),
    )
    .expect("keybindings write");
    agent_dir
}

/// Run the migrations sweep over `agent_dir` with the real migrator and no
/// printed lines.
fn run_migrations(agent_dir: &Path) {
    let lines = RefCell::new(Vec::new());
    let mut print_line = |line: &str| lines.borrow_mut().push(line.to_string());
    let _report = run_migrations_with_in(
        "",
        agent_dir,
        &mut print_line,
        &mut migrate_keybindings_config_value,
    );
    assert!(lines.borrow().is_empty(), "the sweep prints nothing here");
}

#[test]
fn rewrites_old_key_names_to_namespaced_ids() {
    let agent_dir = create_agent_dir(&json!({
        "cursorUp": ["up", "ctrl+p"],
        "expandTools": "ctrl+x",
    }));

    run_migrations(agent_dir.path());

    let migrated: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(agent_dir.path().join("keybindings.json")).expect("read"),
    )
    .expect("parse");
    assert_eq!(
        migrated,
        json!({
            "tui.editor.cursorUp": ["up", "ctrl+p"],
            "app.tools.expand": "ctrl+x",
        })
    );
}

#[test]
fn keeps_the_namespaced_value_when_old_and_new_names_both_exist() {
    let agent_dir = create_agent_dir(&json!({
        "expandTools": "ctrl+x",
        "app.tools.expand": "ctrl+y",
    }));

    run_migrations(agent_dir.path());

    let migrated: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(agent_dir.path().join("keybindings.json")).expect("read"),
    )
    .expect("parse");
    assert_eq!(migrated, json!({"app.tools.expand": "ctrl+y"}));
}

#[test]
fn loads_old_key_names_in_memory_before_the_file_is_rewritten() {
    let agent_dir = create_agent_dir(&json!({
        "selectConfirm": "enter",
        "interrupt": "ctrl+x",
    }));

    let keybindings = AppKeybindingsManager::create(agent_dir.path());

    let user = keybindings
        .get_user_bindings()
        .iter()
        .map(|(action, keys)| (action.to_string(), keys.to_vec()))
        .collect::<Vec<_>>();
    assert_eq!(
        user,
        vec![
            ("tui.select.confirm".to_string(), vec!["enter".to_string()]),
            ("app.interrupt".to_string(), vec!["ctrl+x".to_string()]),
        ]
    );
    let effective = keybindings.get_effective_config();
    assert_eq!(
        effective.get("tui.select.confirm").map(<[String]>::to_vec),
        Some(vec!["enter".to_string()])
    );
    assert_eq!(
        effective.get("app.interrupt").map(<[String]>::to_vec),
        Some(vec!["ctrl+x".to_string()])
    );
}

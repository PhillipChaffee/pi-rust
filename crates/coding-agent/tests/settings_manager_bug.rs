//! The settings-manager external-edit-preservation suite, upstream's
//! `packages/coding-agent/test/settings-manager-bug.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated 1:1.
//!
//! The bug the suite pins: a save that replayed stale in-memory state
//! clobbered external file edits to arrays; only the fields a session
//! explicitly modified override the file on save.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::path::PathBuf;

use pi_coding_agent::settings_manager::{
    FileSettingsStorage, SettingsManager, SettingsManagerCreateOptions,
};
use serde_json::{Value, json};

/// The per-test agent/project tree, upstream's `beforeEach` block.
struct Fixture {
    _root: tempfile::TempDir,
    agent_dir: PathBuf,
    project_dir: PathBuf,
}

fn fixture() -> Fixture {
    let root = tempfile::tempdir().expect("scratch root");
    let agent_dir = root.path().join("agent");
    let project_dir = root.path().join("project");
    std::fs::create_dir_all(&agent_dir).expect("agent dir");
    std::fs::create_dir_all(project_dir.join(".pi")).expect("project .pi dir");
    Fixture {
        _root: root,
        agent_dir,
        project_dir,
    }
}

impl Fixture {
    fn manager(&self) -> SettingsManager<FileSettingsStorage> {
        SettingsManager::create(
            &self.project_dir.to_string_lossy(),
            &self.agent_dir.to_string_lossy(),
            SettingsManagerCreateOptions::default(),
        )
    }

    fn global_settings_path(&self) -> String {
        self.agent_dir
            .join("settings.json")
            .to_string_lossy()
            .into_owned()
    }

    fn project_settings_path(&self) -> String {
        self.project_dir
            .join(".pi")
            .join("settings.json")
            .to_string_lossy()
            .into_owned()
    }

    fn read_global(&self) -> Value {
        let content =
            std::fs::read_to_string(self.global_settings_path()).expect("global settings read");
        serde_json::from_str(&content).expect("global settings parse")
    }

    fn read_project(&self) -> Value {
        let content =
            std::fs::read_to_string(self.project_settings_path()).expect("project settings read");
        serde_json::from_str(&content).expect("project settings parse")
    }
}

#[tokio::test]
async fn should_preserve_file_changes_to_packages_array_when_changing_unrelated_setting() {
    let fixture = fixture();

    // Initial state: packages has one item
    std::fs::write(
        fixture.global_settings_path(),
        json!({"theme": "dark", "packages": ["npm:pi-mcp-adapter"]}).to_string(),
    )
    .expect("initial global settings");

    // Pi starts up, loads settings into memory
    let mut manager = fixture.manager();

    // At this point, globalSettings.packages = ["npm:pi-mcp-adapter"]
    assert_eq!(manager.get_packages(), vec![json!("npm:pi-mcp-adapter")]);

    // User externally edits settings.json to remove the package
    let mut current = fixture.read_global();
    current["packages"] = json!([]);
    std::fs::write(
        fixture.global_settings_path(),
        serde_json::to_string_pretty(&current).expect("external edit serialize"),
    )
    .expect("external edit write");

    // Verify file was changed
    assert_eq!(fixture.read_global()["packages"], json!([]));

    // User changes an UNRELATED setting via UI (this triggers save)
    manager.set_theme("light");
    manager.flush().await;

    // With the fix, packages should be preserved as [] (not reverted to
    // startup value)
    let saved = fixture.read_global();

    assert_eq!(saved["packages"], json!([]));
    assert_eq!(saved["theme"], json!("light"));
}

#[tokio::test]
async fn should_preserve_file_changes_to_extensions_array_when_changing_unrelated_setting() {
    let fixture = fixture();

    std::fs::write(
        fixture.global_settings_path(),
        json!({"theme": "dark", "extensions": ["/old/extension.ts"]}).to_string(),
    )
    .expect("initial global settings");

    let mut manager = fixture.manager();

    // User externally updates extensions
    let mut current = fixture.read_global();
    current["extensions"] = json!(["/new/extension.ts"]);
    std::fs::write(
        fixture.global_settings_path(),
        serde_json::to_string_pretty(&current).expect("external edit serialize"),
    )
    .expect("external edit write");

    // Change unrelated setting
    manager.set_default_thinking_level(pi_agent_core::types::ThinkingLevel::High);
    manager.flush().await;

    let saved = fixture.read_global();

    // With the fix, extensions should be preserved (not reverted to startup
    // value)
    assert_eq!(saved["extensions"], json!(["/new/extension.ts"]));
}

#[tokio::test]
async fn should_preserve_external_project_settings_changes_when_updating_unrelated_project_field() {
    let fixture = fixture();
    std::fs::write(
        fixture.project_settings_path(),
        json!({
            "extensions": ["./old-extension.ts"],
            "prompts": ["./old-prompt.md"],
        })
        .to_string(),
    )
    .expect("initial project settings");

    let mut manager = fixture.manager();

    let mut current = fixture.read_project();
    current["prompts"] = json!(["./new-prompt.md"]);
    std::fs::write(
        fixture.project_settings_path(),
        serde_json::to_string_pretty(&current).expect("external edit serialize"),
    )
    .expect("external edit write");

    manager
        .set_project_extension_paths(&["./updated-extension.ts".to_string()])
        .expect("trusted project write");
    manager.flush().await;

    let saved = fixture.read_project();
    assert_eq!(saved["prompts"], json!(["./new-prompt.md"]));
    assert_eq!(saved["extensions"], json!(["./updated-extension.ts"]));
}

#[tokio::test]
async fn should_let_in_memory_project_changes_override_external_changes_for_the_same_project_field()
{
    let fixture = fixture();
    std::fs::write(
        fixture.project_settings_path(),
        json!({"extensions": ["./initial-extension.ts"]}).to_string(),
    )
    .expect("initial project settings");

    let mut manager = fixture.manager();

    let mut current = fixture.read_project();
    current["extensions"] = json!(["./external-extension.ts"]);
    std::fs::write(
        fixture.project_settings_path(),
        serde_json::to_string_pretty(&current).expect("external edit serialize"),
    )
    .expect("external edit write");

    manager
        .set_project_extension_paths(&["./in-memory-extension.ts".to_string()])
        .expect("trusted project write");
    manager.flush().await;

    let saved = fixture.read_project();
    assert_eq!(saved["extensions"], json!(["./in-memory-extension.ts"]));
}

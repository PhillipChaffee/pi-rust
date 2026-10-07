//! The settings-manager suite, upstream's
//! `packages/coding-agent/test/settings-manager.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated 1:1.
//!
//! The regression suite `test/suite/regressions/8337-utf8-bom-parsing.test.ts`
//! folds in as the trailing `issue_8337_utf8_bom_parsing` module — it drives
//! [`SettingsManager`] beside the utils belt's BOM/frontmatter pair.
//!
//! Porting restatements this suite records:
//!
//! - The env-dependent editor cases ride the `_with` getters over a
//!   map-backed [`EnvLookup`] — the workspace forbids mutating the process
//!   environment, and the map lookup carries no platform to swap, so the
//!   win32 `notepad` platform default has no portable probe and the case
//!   pins the POSIX `nano` default only.
//! - `drainErrors` matches on the [`SettingsError`] fields; upstream's
//!   `toMatchObject` on `{ scope, path }` is the same shape minus the
//!   message.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
mod common;

use std::path::{Path, PathBuf};

use common::{empty_env, env_with};
use pi_agent_core::types::ThinkingLevel;
use pi_coding_agent::settings_manager::{
    DEFAULT_HTTP_IDLE_TIMEOUT_MS, DefaultProjectTrust, FullscreenExitOutput,
    InMemorySettingsStorage, MermaidRenderingMode, RetrySettings, Settings, SettingsManager,
    SettingsManagerCreateOptions, SettingsScope,
};
use pi_tui::components::scroll_view::ScrollViewScrollbar;
use pi_tui::terminal_image::{CapabilityOverrides, ImageProtocol};
use pi_tui::tui::TuiMode;
use serde_json::{Value, json};

// === fixtures ===============================================================

/// The per-test agent/project tree, upstream's `beforeEach` block: an agent
/// dir and a `<project>/.pi` dir under a scratch root that cleans up on
/// drop.
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
    fn manager(&self) -> SettingsManager<pi_coding_agent::settings_manager::FileSettingsStorage> {
        SettingsManager::create(
            &self.project_dir.to_string_lossy(),
            &self.agent_dir.to_string_lossy(),
            SettingsManagerCreateOptions::default(),
        )
    }

    fn manager_untrusted(
        &self,
    ) -> SettingsManager<pi_coding_agent::settings_manager::FileSettingsStorage> {
        SettingsManager::create(
            &self.project_dir.to_string_lossy(),
            &self.agent_dir.to_string_lossy(),
            SettingsManagerCreateOptions {
                project_trusted: Some(false),
            },
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

    fn write_global(&self, settings: &Value) {
        std::fs::write(
            self.global_settings_path(),
            serde_json::to_string(settings).expect("global settings serialize"),
        )
        .expect("global settings write");
    }

    fn write_project(&self, settings: &Value) {
        std::fs::write(
            self.project_settings_path(),
            serde_json::to_string(settings).expect("project settings serialize"),
        )
        .expect("project settings write");
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

/// The in-memory manager, upstream's `SettingsManager.inMemory(...)`.
fn in_memory(settings: &Value) -> SettingsManager<InMemorySettingsStorage> {
    SettingsManager::in_memory(
        &settings.as_object().expect("settings object").clone(),
        SettingsManagerCreateOptions::default(),
    )
}

// === describe("preserves externally added settings") ========================

#[tokio::test]
async fn should_preserve_enabled_models_when_changing_thinking_level() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "dark", "defaultModel": "claude-sonnet"}));
    let mut manager = fixture.manager();

    // Simulate user editing settings.json externally to add enabledModels
    let mut current = fixture.read_global();
    current["enabledModels"] = json!(["claude-opus-4-5", "gpt-5.2-codex"]);
    fixture.write_global(&current);

    // User changes thinking level via Shift+Tab
    manager.set_default_thinking_level(ThinkingLevel::High);
    manager.flush().await;

    let saved = fixture.read_global();
    assert_eq!(
        saved["enabledModels"],
        json!(["claude-opus-4-5", "gpt-5.2-codex"])
    );
    assert_eq!(saved["defaultThinkingLevel"], json!("high"));
    assert_eq!(saved["theme"], json!("dark"));
    assert_eq!(saved["defaultModel"], json!("claude-sonnet"));
}

#[tokio::test]
async fn should_preserve_custom_settings_when_changing_theme() {
    let fixture = fixture();
    fixture.write_global(&json!({"defaultModel": "claude-sonnet"}));
    let mut manager = fixture.manager();

    // User adds custom settings externally
    let mut current = fixture.read_global();
    current["shellPath"] = json!("/bin/zsh");
    current["extensions"] = json!(["/path/to/extension.ts"]);
    fixture.write_global(&current);

    // User changes theme
    manager.set_theme("light");
    manager.flush().await;

    let saved = fixture.read_global();
    assert_eq!(saved["shellPath"], json!("/bin/zsh"));
    assert_eq!(saved["extensions"], json!(["/path/to/extension.ts"]));
    assert_eq!(saved["theme"], json!("light"));
}

#[tokio::test]
async fn should_let_in_memory_changes_override_file_changes_for_same_key() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "dark"}));
    let mut manager = fixture.manager();

    // User externally sets thinking level to "low"
    let mut current = fixture.read_global();
    current["defaultThinkingLevel"] = json!("low");
    fixture.write_global(&current);

    // But then changes it via UI to "high"
    manager.set_default_thinking_level(ThinkingLevel::High);
    manager.flush().await;

    // In-memory change should win
    assert_eq!(fixture.read_global()["defaultThinkingLevel"], json!("high"));
}

// === describe("packages migration") =========================================

#[test]
fn should_keep_local_only_extensions_in_extensions_array() {
    let fixture = fixture();
    fixture.write_global(&json!({"extensions": ["/local/ext.ts", "./relative/ext.ts"]}));

    let manager = fixture.manager();

    assert_eq!(manager.get_packages(), Vec::<Value>::new());
    assert_eq!(
        manager.get_extension_paths(),
        vec!["/local/ext.ts".to_string(), "./relative/ext.ts".to_string()]
    );
}

#[test]
fn should_handle_packages_with_filtering_objects() {
    let fixture = fixture();
    fixture.write_global(&json!({
        "packages": [
            "npm:simple-pkg",
            {
                "source": "npm:shitty-extensions",
                "extensions": ["extensions/oracle.ts"],
                "skills": [],
            },
        ],
    }));

    let manager = fixture.manager();

    let packages = manager.get_packages();
    assert_eq!(packages.len(), 2);
    assert_eq!(packages[0], json!("npm:simple-pkg"));
    assert_eq!(
        packages[1],
        json!({
            "source": "npm:shitty-extensions",
            "extensions": ["extensions/oracle.ts"],
            "skills": [],
        })
    );
}

// === describe("reload") =====================================================

#[test]
fn should_reload_global_settings_from_disk() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "dark", "extensions": ["/before.ts"]}));

    let mut manager = fixture.manager();

    fixture.write_global(&json!({
        "theme": "light",
        "extensions": ["/after.ts"],
        "defaultModel": "claude-sonnet",
    }));

    manager.reload();

    assert_eq!(manager.get_theme().as_deref(), Some("light"));
    assert_eq!(manager.get_extension_paths(), vec!["/after.ts".to_string()]);
    assert_eq!(
        manager.get_default_model().as_deref(),
        Some("claude-sonnet")
    );
}

#[test]
fn should_keep_previous_settings_and_report_the_file_path_when_the_file_is_invalid() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "dark"}));

    let mut manager = fixture.manager();

    std::fs::write(fixture.global_settings_path(), "{ invalid json").expect("invalid rewrite");
    manager.reload();

    assert_eq!(manager.get_theme().as_deref(), Some("dark"));
    let errors = manager.drain_errors();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].scope, SettingsScope::Global);
    assert_eq!(
        errors[0].path.as_deref(),
        Some(fixture.global_settings_path().as_str())
    );
}

// === describe("theme setting") ==============================================

#[tokio::test]
async fn stores_slash_separated_automatic_theme_settings_separately_from_fixed_theme_names() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "light/dark"}));

    let mut manager = fixture.manager();

    assert_eq!(manager.get_theme(), None);
    assert_eq!(manager.get_theme_setting().as_deref(), Some("light/dark"));

    manager.set_theme("solarized-light/tokyo-night");
    manager.flush().await;

    assert_eq!(
        fixture.read_global()["theme"],
        json!("solarized-light/tokyo-night")
    );
}

// === describe("error tracking") =============================================

#[test]
fn should_collect_and_clear_load_errors_via_drain_errors() {
    let fixture = fixture();
    std::fs::write(fixture.global_settings_path(), "{ invalid global json")
        .expect("invalid global settings");
    std::fs::write(fixture.project_settings_path(), "{ invalid project json")
        .expect("invalid project settings");

    let mut manager = fixture.manager();
    let errors = manager.drain_errors();

    assert_eq!(errors.len(), 2);
    assert_eq!(errors[0].scope, SettingsScope::Global);
    assert_eq!(
        errors[0].path.as_deref(),
        Some(fixture.global_settings_path().as_str())
    );
    assert_eq!(errors[1].scope, SettingsScope::Project);
    assert_eq!(
        errors[1].path.as_deref(),
        Some(fixture.project_settings_path().as_str())
    );
    assert!(manager.drain_errors().is_empty());
}

// === describe("project trust") ==============================================

#[test]
fn should_skip_project_settings_when_project_is_not_trusted() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "global"}));
    fixture.write_project(&json!({"theme": "project"}));

    let manager = fixture.manager_untrusted();

    assert!(!manager.is_project_trusted());
    assert_eq!(manager.get_theme().as_deref(), Some("global"));
    assert_eq!(manager.get_project_settings(), Settings::new());
}

#[test]
fn should_reload_project_settings_after_trust_changes_to_true() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "global"}));
    fixture.write_project(&json!({"theme": "project"}));
    let mut manager = fixture.manager_untrusted();

    manager.set_project_trusted(true);

    assert!(manager.is_project_trusted());
    assert_eq!(manager.get_theme().as_deref(), Some("project"));
}

#[tokio::test]
async fn should_fail_project_settings_writes_when_project_is_not_trusted() {
    let fixture = fixture();
    fixture.write_project(&json!({"packages": ["npm:existing"]}));
    let mut manager = fixture.manager_untrusted();

    assert_eq!(
        manager.set_project_packages(&[json!("npm:new")]),
        Err("Project is not trusted; refusing to write project settings".to_string())
    );
    manager.flush().await;

    assert_eq!(manager.get_project_settings(), Settings::new());
    assert_eq!(
        fixture.read_project(),
        json!({"packages": ["npm:existing"]})
    );
}

#[test]
fn should_read_default_project_trust_from_global_settings_only() {
    let fixture = fixture();
    fixture.write_global(&json!({"defaultProjectTrust": "always"}));
    fixture.write_project(&json!({"defaultProjectTrust": "never"}));

    let manager = fixture.manager();

    assert_eq!(
        manager.get_default_project_trust(),
        DefaultProjectTrust::Always
    );
}

#[test]
fn should_default_invalid_project_trust_settings_to_ask() {
    let fixture = fixture();
    fixture.write_global(&json!({"defaultProjectTrust": "sometimes"}));

    let manager = fixture.manager();

    assert_eq!(
        manager.get_default_project_trust(),
        DefaultProjectTrust::Ask
    );
}

// === describe("project settings directory creation") ========================

#[test]
fn should_not_create_pi_folder_when_only_reading_project_settings() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "dark"}));
    let pi_dir = fixture.project_dir.join(".pi");
    std::fs::remove_dir_all(&pi_dir).expect(".pi folder removed");

    let manager = fixture.manager();

    assert!(!pi_dir.exists());
    assert_eq!(manager.get_theme().as_deref(), Some("dark"));
}

#[tokio::test]
async fn should_create_pi_folder_when_writing_project_settings() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "dark"}));
    let pi_dir = fixture.project_dir.join(".pi");
    std::fs::remove_dir_all(&pi_dir).expect(".pi folder removed");

    let mut manager = fixture.manager();

    assert!(!pi_dir.exists());

    manager
        .set_project_packages(&[json!({"source": "npm:test-pkg"})])
        .expect("trusted project write");
    manager.flush().await;

    assert!(pi_dir.exists());
    assert!(Path::new(&fixture.project_settings_path()).exists());
}

// === describe("terminal capability overrides") ==============================

#[test]
fn maps_explicit_values_and_omits_auto_values() {
    let get_overrides = |terminal: Value| {
        in_memory(&json!({"terminal": terminal})).get_terminal_capability_overrides()
    };

    assert_eq!(
        get_overrides(json!({"images": false, "trueColor": false, "hyperlinks": false})),
        CapabilityOverrides {
            images: Some(None),
            true_color: Some(false),
            hyperlinks: Some(false),
        }
    );
    assert_eq!(
        get_overrides(json!({"images": "kitty", "trueColor": true, "hyperlinks": true})),
        CapabilityOverrides {
            images: Some(Some(ImageProtocol::Kitty)),
            true_color: Some(true),
            hyperlinks: Some(true),
        }
    );
    assert_eq!(
        get_overrides(json!({"images": "auto", "trueColor": "auto", "hyperlinks": "auto"})),
        CapabilityOverrides::default()
    );
}

// === describe("retry settings") =============================================

#[test]
fn defaults_and_overrides_agent_retry_delay_cap() {
    assert_eq!(
        in_memory(&json!({})).get_retry_settings(),
        RetrySettings {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 2000,
            max_agent_delay_ms: 60_000,
        }
    );
    assert_eq!(
        in_memory(&json!({
            "retry": {"enabled": true, "maxRetries": 10, "baseDelayMs": 500, "maxAgentDelayMs": 5000},
        }))
        .get_retry_settings(),
        RetrySettings {
            enabled: true,
            max_retries: 10,
            base_delay_ms: 500,
            max_agent_delay_ms: 5000,
        }
    );
}

// === describe("httpIdleTimeoutMs") ==========================================

#[test]
fn should_default_to_5_minutes() {
    let fixture = fixture();
    let manager = fixture.manager();

    assert_eq!(
        manager.get_http_idle_timeout_ms(),
        Ok(DEFAULT_HTTP_IDLE_TIMEOUT_MS)
    );
}

#[test]
fn should_use_merged_global_and_project_settings() {
    let fixture = fixture();
    fixture.write_global(&json!({"httpIdleTimeoutMs": 300_000}));
    fixture.write_project(&json!({"httpIdleTimeoutMs": 0}));

    let manager = fixture.manager();

    assert_eq!(manager.get_http_idle_timeout_ms(), Ok(0));
}

#[test]
fn should_reject_invalid_timeout_values() {
    let fixture = fixture();
    fixture.write_global(&json!({"httpIdleTimeoutMs": -1}));
    let manager = fixture.manager();

    assert_eq!(
        manager.get_http_idle_timeout_ms(),
        Err("Invalid httpIdleTimeoutMs setting: -1".to_string())
    );
}

// === describe("externalEditor") =============================================

#[test]
fn should_resolve_editor_commands_by_precedence() {
    let visual_and_editor = env_with(&[("VISUAL", "vim"), ("EDITOR", "nano")]);
    assert_eq!(
        in_memory(&json!({"externalEditor": "code --wait"}))
            .get_external_editor_command_with(&visual_and_editor),
        "code --wait"
    );
    assert_eq!(
        in_memory(&json!({})).get_external_editor_command_with(&visual_and_editor),
        "vim"
    );

    let editor_only = env_with(&[("EDITOR", "emacs")]);
    assert_eq!(
        in_memory(&json!({})).get_external_editor_command_with(&editor_only),
        "emacs"
    );
}

#[test]
fn should_fall_back_to_platform_defaults() {
    // POSIX: the darwin/linux platform default is nano. The win32 "notepad"
    // branch rides `cfg!(windows)` at compile time and the map-backed
    // environment lookup carries no platform to swap, so it has no portable
    // probe from this suite.
    assert_eq!(
        in_memory(&json!({})).get_external_editor_command_with(&empty_env()),
        "nano"
    );
}

// === describe("TUI mode") ===================================================

#[tokio::test]
async fn defaults_to_regular_and_persists_fullscreen_mode() {
    let fixture = fixture();
    let mut manager = fixture.manager();

    assert_eq!(manager.get_tui_mode(), TuiMode::Regular);

    manager.set_tui_mode(TuiMode::Fullscreen);
    manager.flush().await;

    assert_eq!(manager.get_tui_mode(), TuiMode::Fullscreen);
    assert_eq!(fixture.read_global()["tuiMode"], json!("fullscreen"));
}

#[test]
fn falls_back_to_regular_for_unsupported_values() {
    let fixture = fixture();
    fixture.write_global(&json!({"tuiMode": "other"}));

    let manager = fixture.manager();

    assert_eq!(manager.get_tui_mode(), TuiMode::Regular);
}

#[test]
fn does_not_recognize_the_old_ui_mode_setting() {
    let fixture = fixture();
    fixture.write_global(&json!({"uiMode": "fullscreen"}));

    let manager = fixture.manager();

    assert_eq!(manager.get_tui_mode(), TuiMode::Regular);
}

// === it("validates and persists fullscreen settings") =======================

#[tokio::test]
async fn validates_and_persists_fullscreen_settings() {
    let fixture = fixture();
    let mut manager = fixture.manager();
    assert_eq!(
        manager.get_fullscreen_exit_output(),
        FullscreenExitOutput::Transcript
    );
    assert_eq!(
        manager.get_fullscreen_scrollbar(),
        ScrollViewScrollbar::Auto
    );
    assert!(manager.get_fullscreen_copy_on_select());

    manager.set_fullscreen_exit_output(FullscreenExitOutput::ResumeHint);
    manager.set_fullscreen_scrollbar(ScrollViewScrollbar::Hidden);
    manager.set_fullscreen_copy_on_select(false);
    manager.flush().await;
    let saved = fixture.read_global();
    assert_eq!(saved["fullscreenExitOutput"], json!("resume-hint"));
    assert_eq!(saved["fullscreenScrollbar"], json!("hidden"));
    assert_eq!(saved["fullscreenCopyOnSelect"], json!(false));

    fixture.write_global(&json!({
        "fullscreenExitOutput": "nothing",
        "fullscreenScrollbar": "sometimes",
    }));
    let reloaded_manager = fixture.manager();
    assert_eq!(
        reloaded_manager.get_fullscreen_exit_output(),
        FullscreenExitOutput::Transcript
    );
    assert_eq!(
        reloaded_manager.get_fullscreen_scrollbar(),
        ScrollViewScrollbar::Auto
    );
    assert!(reloaded_manager.get_fullscreen_copy_on_select());
}

// === describe("outputPad") ==================================================

#[tokio::test]
async fn should_default_to_1_and_persist_binary_values() {
    let fixture = fixture();
    let mut manager = fixture.manager();

    assert_eq!(manager.get_output_pad(), 1);

    manager.set_output_pad(0);
    manager.flush().await;

    assert_eq!(manager.get_output_pad(), 0);
    assert_eq!(fixture.read_global()["outputPad"], json!(0));
}

#[test]
fn should_treat_unsupported_output_pad_values_as_default_padding() {
    let fixture = fixture();
    fixture.write_global(&json!({"outputPad": 2}));

    let manager = fixture.manager();

    assert_eq!(manager.get_output_pad(), 1);
}

// === describe("markdown.mermaid") ===========================================

#[tokio::test]
async fn defaults_to_streaming_and_persists_rendering_modes() {
    let fixture = fixture();
    let mut manager = fixture.manager();

    assert_eq!(
        manager.get_mermaid_rendering_mode(),
        MermaidRenderingMode::Streaming
    );

    manager.set_mermaid_rendering_mode(MermaidRenderingMode::Final);
    manager.flush().await;

    assert_eq!(
        manager.get_mermaid_rendering_mode(),
        MermaidRenderingMode::Final
    );
    assert_eq!(fixture.read_global()["markdown"]["mermaid"], json!("final"));
}

#[test]
fn falls_back_to_streaming_for_unsupported_values() {
    let fixture = fixture();
    fixture.write_global(&json!({"markdown": {"mermaid": "sometimes"}}));

    assert_eq!(
        fixture.manager().get_mermaid_rendering_mode(),
        MermaidRenderingMode::Streaming
    );
}

// === describe("shellCommandPrefix") =========================================

#[test]
fn should_load_shell_command_prefix_from_settings() {
    let fixture = fixture();
    fixture.write_global(&json!({"shellCommandPrefix": "shopt -s expand_aliases"}));

    let manager = fixture.manager();

    assert_eq!(
        manager.get_shell_command_prefix().as_deref(),
        Some("shopt -s expand_aliases")
    );
}

#[test]
fn should_return_undefined_when_shell_command_prefix_is_not_set() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "dark"}));

    let manager = fixture.manager();

    assert_eq!(manager.get_shell_command_prefix(), None);
}

#[tokio::test]
async fn should_preserve_shell_command_prefix_when_saving_unrelated_settings() {
    let fixture = fixture();
    fixture.write_global(&json!({"shellCommandPrefix": "shopt -s expand_aliases"}));

    let mut manager = fixture.manager();
    manager.set_theme("light");
    manager.flush().await;

    let saved = fixture.read_global();
    assert_eq!(
        saved["shellCommandPrefix"],
        json!("shopt -s expand_aliases")
    );
    assert_eq!(saved["theme"], json!("light"));
}

// === describe("defaultTools") ===============================================

#[test]
fn loads_global_defaults_and_lets_project_settings_replace_them() {
    let fixture = fixture();
    fixture.write_global(&json!({"defaultTools": ["read", "bash"]}));

    assert_eq!(
        fixture.manager().get_default_tools(),
        Some(vec!["read".to_string(), "bash".to_string()])
    );

    fixture.write_project(&json!({"defaultTools": ["grep"]}));

    assert_eq!(
        fixture.manager().get_default_tools(),
        Some(vec!["grep".to_string()])
    );
}

#[test]
fn preserves_an_empty_tool_list() {
    assert_eq!(
        in_memory(&json!({"defaultTools": []})).get_default_tools(),
        Some(Vec::new())
    );
    assert_eq!(in_memory(&json!({})).get_default_tools(), None);
}

// === describe("getSessionDir") ==============================================

// The upstream suite nests this title under both the getSessionDir and
// getShellPath describes; the subject prefix keeps the two flat test names
// apart.

#[test]
fn session_dir_should_return_undefined_when_not_set() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "dark"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_session_dir(), None);
}

#[test]
fn should_return_global_session_dir() {
    let fixture = fixture();
    fixture.write_global(&json!({"sessionDir": "/tmp/sessions"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_session_dir().as_deref(), Some("/tmp/sessions"));
}

#[test]
fn should_return_project_session_dir_overriding_global() {
    let fixture = fixture();
    fixture.write_global(&json!({"sessionDir": "/global/sessions"}));
    fixture.write_project(&json!({"sessionDir": "./sessions"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_session_dir().as_deref(), Some("./sessions"));
}

#[test]
fn should_expand_tilde_in_session_dir() {
    let fixture = fixture();
    fixture.write_global(&json!({"sessionDir": "~/sessions"}));
    let manager = fixture.manager();
    let home = std::env::home_dir().expect("home directory resolves");
    assert_eq!(
        manager.get_session_dir().as_deref(),
        Some(home.join("sessions").to_string_lossy().as_ref())
    );
}

// === describe("getShellPath") ===============================================

#[test]
fn shell_path_should_return_undefined_when_not_set() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "dark"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_shell_path(), None);
}

#[test]
fn should_return_an_absolute_shell_path_unchanged() {
    let fixture = fixture();
    fixture.write_global(&json!({"shellPath": "/bin/zsh"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_shell_path().as_deref(), Some("/bin/zsh"));
}

#[test]
fn should_expand_tilde_in_shell_path() {
    let fixture = fixture();
    fixture.write_global(&json!({"shellPath": "~/.local/bin/agent-shell-sandbox"}));
    let manager = fixture.manager();
    let home = std::env::home_dir().expect("home directory resolves");
    assert_eq!(
        manager.get_shell_path().as_deref(),
        Some(
            home.join(".local/bin/agent-shell-sandbox")
                .to_string_lossy()
                .as_ref()
        )
    );
}

#[test]
fn should_expand_a_bare_tilde_in_shell_path() {
    let fixture = fixture();
    fixture.write_global(&json!({"shellPath": "~"}));
    let manager = fixture.manager();
    let home = std::env::home_dir().expect("home directory resolves");
    assert_eq!(
        manager.get_shell_path().as_deref(),
        Some(home.to_string_lossy().as_ref())
    );
}

// === issue #8337 UTF-8 BOM parsing (regression suite) =======================

mod issue_8337_utf8_bom_parsing {
    use super::*;
    use pi_coding_agent::utils::frontmatter::parse_frontmatter;
    use pi_coding_agent::utils::text::{SplitBom, split_bom};
    use yaml_rust2::Yaml;

    fn yaml_str<'a>(frontmatter: &'a Yaml, key: &str) -> Option<&'a str> {
        frontmatter
            .as_hash()?
            .get(&Yaml::String(key.to_string()))
            .and_then(Yaml::as_str)
    }

    #[tokio::test]
    async fn loads_frontmatter_and_settings_with_a_leading_bom() {
        assert_eq!(
            split_bom("\u{FEFF}content"),
            SplitBom {
                bom: "\u{FEFF}".to_string(),
                text: "content".to_string(),
            }
        );
        let document = "---\nname: demo\ndescription: Test\n---\nBody";
        let parsed = parse_frontmatter(&format!("\u{FEFF}{document}")).expect("frontmatter parses");
        assert_eq!(yaml_str(&parsed.frontmatter, "name"), Some("demo"));
        assert_eq!(yaml_str(&parsed.frontmatter, "description"), Some("Test"));
        assert_eq!(parsed.body, "Body");

        let fixture = fixture();
        std::fs::write(
            fixture.global_settings_path(),
            format!("\u{FEFF}{}", json!({"defaultModel": "global-model"})),
        )
        .expect("bommed global settings");
        std::fs::write(
            fixture.project_settings_path(),
            format!("\u{FEFF}{}", json!({"defaultProvider": "project-provider"})),
        )
        .expect("bommed project settings");

        let mut settings = fixture.manager();
        assert_eq!(
            settings.get_default_model().as_deref(),
            Some("global-model")
        );
        assert_eq!(
            settings.get_default_provider().as_deref(),
            Some("project-provider")
        );

        settings.set_theme("dark");
        settings.flush().await;
        let saved =
            std::fs::read_to_string(fixture.global_settings_path()).expect("saved settings");
        assert!(!saved.starts_with('\u{FEFF}'));
    }
}

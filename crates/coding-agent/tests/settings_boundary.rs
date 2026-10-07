//! Boundary tests binding the settings-manager branches the 1:1 suites leave
//! untested, at pin 60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::path::PathBuf;

use common::{empty_env, env_with};
use pi_agent_core::types::ThinkingLevel;
use pi_ai::types::Transport;
use pi_ai::utils::retry::DEFAULT_MAX_AGENT_RETRY_DELAY_MS;
use pi_coding_agent::config::home_dir;
use pi_coding_agent::settings_diagnostics::{SettingsDiagnostic, collect_settings_diagnostics};
use pi_coding_agent::settings_manager::{
    DEFAULT_HTTP_IDLE_TIMEOUT_MS, DefaultProjectTrust, FileSettingsStorage, FullscreenExitOutput,
    MermaidRenderingMode, ModelKey, QueueMode, Settings, SettingsLockFn, SettingsManager,
    SettingsManagerCreateOptions, SettingsScope, SettingsStorage, parse_http_idle_timeout_ms,
};
use pi_tui::components::scroll_view::ScrollViewScrollbar;
use pi_tui::terminal_image::{CapabilityOverrides, ImageProtocol};
use pi_tui::tui::TuiMode;
use serde_json::{Value, json};

// === fixtures ===============================================================

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

    fn manager_untrusted(&self) -> SettingsManager<FileSettingsStorage> {
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
        std::fs::write(self.global_settings_path(), settings.to_string())
            .expect("global settings write");
    }

    fn write_project(&self, settings: &Value) {
        std::fs::write(self.project_settings_path(), settings.to_string())
            .expect("project settings write");
    }

    fn write_raw_global(&self, content: &str) {
        std::fs::write(self.global_settings_path(), content).expect("raw global write");
    }

    fn write_raw_project(&self, content: &str) {
        std::fs::write(self.project_settings_path(), content).expect("raw project write");
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

    fn read_raw_project(&self) -> String {
        std::fs::read_to_string(self.project_settings_path()).expect("raw project read")
    }
}

// === the http idle timeout grammar ==========================================

#[test]
fn parse_http_idle_timeout_ms_pins_the_string_grammar() {
    assert_eq!(parse_http_idle_timeout_ms(&json!("disabled")), Some(0));
    assert_eq!(parse_http_idle_timeout_ms(&json!("DISABLED")), Some(0));
    assert_eq!(parse_http_idle_timeout_ms(&json!("")), None);
    assert_eq!(parse_http_idle_timeout_ms(&json!("  ")), None);
    assert_eq!(parse_http_idle_timeout_ms(&json!("12.9")), Some(12));
    assert_eq!(parse_http_idle_timeout_ms(&json!("1e3")), Some(1_000));
    assert_eq!(parse_http_idle_timeout_ms(&json!("abc")), None);
    assert_eq!(parse_http_idle_timeout_ms(&json!("-1")), None);
    assert_eq!(parse_http_idle_timeout_ms(&json!("nan")), None);
}

#[test]
fn parse_http_idle_timeout_ms_pins_the_number_grammar() {
    assert_eq!(parse_http_idle_timeout_ms(&json!(5.9)), Some(5));
    assert_eq!(parse_http_idle_timeout_ms(&json!(0)), Some(0));
    assert_eq!(parse_http_idle_timeout_ms(&json!(-1)), None);
    assert_eq!(parse_http_idle_timeout_ms(&json!(true)), None);
}

#[test]
fn get_http_idle_timeout_ms_reads_the_setting_and_defaults() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let manager = fixture.manager();
    assert_eq!(
        manager.get_http_idle_timeout_ms(),
        Ok(DEFAULT_HTTP_IDLE_TIMEOUT_MS)
    );

    fixture.write_global(&json!({"httpIdleTimeoutMs": "disabled"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_http_idle_timeout_ms(), Ok(0));

    fixture.write_global(&json!({"httpIdleTimeoutMs": 2500}));
    let manager = fixture.manager();
    assert_eq!(manager.get_http_idle_timeout_ms(), Ok(2_500));

    fixture.write_global(&json!({"httpIdleTimeoutMs": "abc"}));
    let manager = fixture.manager();
    assert_eq!(
        manager.get_http_idle_timeout_ms(),
        Err("Invalid httpIdleTimeoutMs setting: abc".to_string())
    );
}

#[test]
fn set_http_idle_timeout_ms_validates_and_stores_the_floor() {
    let fixture = fixture();
    let mut manager = fixture.manager();
    assert_eq!(
        manager.set_http_idle_timeout_ms(-1.0),
        Err("Invalid httpIdleTimeoutMs setting: -1".to_string())
    );
    manager
        .set_http_idle_timeout_ms(1500.7)
        .expect("valid timeout");
    assert_eq!(manager.get_http_idle_timeout_ms(), Ok(1_500));
    // the saved text carries the floored integer, the way upstream's
    // JSON.stringify writes Math.floor's JS number
    let saved = std::fs::read_to_string(fixture.global_settings_path()).expect("saved read");
    assert!(saved.contains("\"httpIdleTimeoutMs\": 1500"), "{saved}");

    // a floor beyond the integer range stays a float
    manager
        .set_http_idle_timeout_ms(1e30)
        .expect("valid timeout");
    let saved = std::fs::read_to_string(fixture.global_settings_path()).expect("saved read");
    assert!(saved.contains("1e+30"), "{saved}");
}

// === queue modes and transport ==============================================

#[test]
fn steering_and_follow_up_modes_round_trip() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert_eq!(manager.get_steering_mode(), QueueMode::OneAtATime);
    assert_eq!(manager.get_follow_up_mode(), QueueMode::OneAtATime);
    assert_eq!(QueueMode::All.as_str(), "all");
    assert_eq!(QueueMode::OneAtATime.as_str(), "one-at-a-time");

    manager.set_steering_mode(QueueMode::All);
    manager.set_follow_up_mode(QueueMode::All);
    assert_eq!(manager.get_steering_mode(), QueueMode::All);
    assert_eq!(manager.get_follow_up_mode(), QueueMode::All);

    fixture.write_global(&json!({"steeringMode": "all", "followUpMode": "one-at-a-time"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_steering_mode(), QueueMode::All);
    assert_eq!(manager.get_follow_up_mode(), QueueMode::OneAtATime);

    fixture.write_global(&json!({"steeringMode": "bogus"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_steering_mode(), QueueMode::OneAtATime);
}

#[test]
fn transport_reads_the_stored_value_and_defaults_to_auto() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let manager = fixture.manager();
    assert_eq!(manager.get_transport(), Transport::Auto);

    for (wire, expected) in [
        ("sse", Transport::Sse),
        ("websocket", Transport::Websocket),
        ("websocket-cached", Transport::WebsocketCached),
    ] {
        fixture.write_global(&json!({ "transport": wire }));
        let manager = fixture.manager();
        assert_eq!(manager.get_transport(), expected);
    }

    fixture.write_global(&json!({"transport": "carrier-pigeon"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_transport(), Transport::Auto);

    let mut writer = fixture.manager();
    writer.set_transport(Transport::Websocket);
    assert_eq!(writer.get_transport(), Transport::Websocket);
    assert_eq!(
        fixture.read_global().get("transport"),
        Some(&json!("websocket"))
    );
}

// === changelog, provider and model ==========================================

#[test]
fn changelog_version_round_trips() {
    let fixture = fixture();
    fixture.write_global(&json!({"lastChangelogVersion": "1.2.3"}));
    let mut manager = fixture.manager();
    assert_eq!(
        manager.get_last_changelog_version(),
        Some("1.2.3".to_string())
    );

    manager.set_last_changelog_version("9.9.9");
    assert_eq!(
        manager.get_last_changelog_version(),
        Some("9.9.9".to_string())
    );
    assert_eq!(
        fixture.read_global().get("lastChangelogVersion"),
        Some(&json!("9.9.9"))
    );
}

#[test]
fn default_provider_and_model_setters_write_the_file() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();

    manager.set_default_provider("anthropic");
    assert_eq!(
        manager.get_default_provider(),
        Some("anthropic".to_string())
    );
    manager.set_default_model("claude-x");
    assert_eq!(manager.get_default_model(), Some("claude-x".to_string()));

    manager.set_default_model_and_provider("openai", "gpt-y");
    assert_eq!(manager.get_default_provider(), Some("openai".to_string()));
    assert_eq!(manager.get_default_model(), Some("gpt-y".to_string()));
    assert_eq!(
        fixture.read_global(),
        json!({"defaultProvider": "openai", "defaultModel": "gpt-y"})
    );
}

// === thinking levels ========================================================

#[test]
fn default_thinking_level_reads_the_stored_value() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let manager = fixture.manager();
    assert_eq!(manager.get_default_thinking_level(), None);

    fixture.write_global(&json!({"defaultThinkingLevel": "high"}));
    let manager = fixture.manager();
    assert_eq!(
        manager.get_default_thinking_level(),
        Some(ThinkingLevel::High)
    );

    fixture.write_global(&json!({"defaultThinkingLevel": "bogus"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_default_thinking_level(), None);
}

#[test]
fn model_thinking_levels_get_set_all_and_remove() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert_eq!(manager.get_model_thinking_level("openai", "gpt"), None);
    assert!(manager.get_all_model_thinking_levels().is_empty());

    manager.set_model_thinking_level("openai", "gpt", ThinkingLevel::Max);
    assert_eq!(
        manager.get_model_thinking_level("openai", "gpt"),
        Some(ThinkingLevel::Max)
    );
    assert_eq!(
        manager.get_all_model_thinking_levels(),
        serde_json::from_str::<Settings>(r#"{"openai/gpt":"max"}"#).expect("levels object")
    );
    assert_eq!(
        fixture.read_global(),
        json!({"modelThinkingLevels": {"openai/gpt": "max"}})
    );

    manager.set_model_thinking_level("anthropic", "claude", ThinkingLevel::Low);
    manager.remove_model_thinking_level("openai", "gpt");
    assert_eq!(manager.get_model_thinking_level("openai", "gpt"), None);
    assert_eq!(
        manager.get_model_thinking_level("anthropic", "claude"),
        Some(ThinkingLevel::Low)
    );

    // removing the last entry drops the whole field
    manager.remove_model_thinking_level("anthropic", "claude");
    assert!(manager.get_all_model_thinking_levels().is_empty());
    assert_eq!(fixture.read_global().get("modelThinkingLevels"), None);

    // removing with no field present is a tracked no-op
    manager.remove_model_thinking_level("openai", "gpt");
    assert!(manager.drain_errors().is_empty());
    assert_eq!(manager.get_model_thinking_level("openai", "gpt"), None);
}

#[test]
fn model_thinking_level_reads_skip_invalid_values() {
    let fixture = fixture();
    fixture.write_global(
        &json!({"modelThinkingLevels": {"openai/gpt": 5, "anthropic/claude": "high"}}),
    );
    let manager = fixture.manager();
    assert_eq!(manager.get_model_thinking_level("openai", "gpt"), None);
    assert_eq!(
        manager.get_model_thinking_level("anthropic", "claude"),
        Some(ThinkingLevel::High)
    );
    assert_eq!(manager.get_model_thinking_level("openai", "missing"), None);
}

// === branch summary and retry ===============================================

#[test]
fn branch_summary_settings_read_with_defaults() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let manager = fixture.manager();
    let settings = manager.get_branch_summary_settings();
    assert_eq!(settings.reserve_tokens, 16_384);
    assert!(!settings.skip_prompt);
    assert!(!manager.get_branch_summary_skip_prompt());

    fixture.write_global(&json!({"branchSummary": {"reserveTokens": 99, "skipPrompt": true}}));
    let manager = fixture.manager();
    let settings = manager.get_branch_summary_settings();
    assert_eq!(settings.reserve_tokens, 99);
    assert!(settings.skip_prompt);
    assert!(manager.get_branch_summary_skip_prompt());
}

#[test]
fn retry_settings_read_with_defaults_and_set_retry_enabled() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    let settings = manager.get_retry_settings();
    assert!(settings.enabled);
    assert_eq!(settings.max_retries, 3);
    assert_eq!(settings.base_delay_ms, 2_000);
    assert_eq!(
        settings.max_agent_delay_ms,
        DEFAULT_MAX_AGENT_RETRY_DELAY_MS.cast_signed()
    );

    manager.set_retry_enabled(false);
    assert!(!manager.get_retry_enabled());
    assert_eq!(fixture.read_global(), json!({"retry": {"enabled": false}}));

    fixture.write_global(
        &json!({"retry": {"maxRetries": 7, "baseDelayMs": 500, "maxAgentDelayMs": 90_000}}),
    );
    let manager = fixture.manager();
    let settings = manager.get_retry_settings();
    assert_eq!(settings.max_retries, 7);
    assert_eq!(settings.base_delay_ms, 500);
    assert_eq!(settings.max_agent_delay_ms, 90_000);
}

#[test]
fn provider_retry_settings_read_the_nested_object() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let manager = fixture.manager();
    let settings = manager.get_provider_retry_settings();
    assert_eq!(settings.timeout_ms, None);
    assert_eq!(settings.max_retries, None);
    assert_eq!(settings.max_retry_delay_ms, 60_000);

    fixture.write_global(&json!({"retry": {"provider": {"timeoutMs": 1000, "maxRetries": 4, "maxRetryDelayMs": 9000}}}));
    let manager = fixture.manager();
    let settings = manager.get_provider_retry_settings();
    assert_eq!(settings.timeout_ms, Some(1_000));
    assert_eq!(settings.max_retries, Some(4));
    assert_eq!(settings.max_retry_delay_ms, 9_000);

    fixture.write_global(&json!({"retry": {"provider": {"maxRetryDelayMs": 500}}}));
    let manager = fixture.manager();
    assert_eq!(
        manager.get_provider_retry_settings().max_retry_delay_ms,
        500
    );
}

// === websocket timeout ======================================================

#[test]
fn websocket_connect_timeout_reads_the_grammar() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let manager = fixture.manager();
    assert_eq!(manager.get_websocket_connect_timeout_ms(), Ok(None));

    fixture.write_global(&json!({"websocketConnectTimeoutMs": "disabled"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_websocket_connect_timeout_ms(), Ok(Some(0)));

    fixture.write_global(&json!({"websocketConnectTimeoutMs": 2500}));
    let manager = fixture.manager();
    assert_eq!(manager.get_websocket_connect_timeout_ms(), Ok(Some(2_500)));

    fixture.write_global(&json!({"websocketConnectTimeoutMs": "abc"}));
    let manager = fixture.manager();
    assert_eq!(
        manager.get_websocket_connect_timeout_ms(),
        Err("Invalid websocketConnectTimeoutMs setting: abc".to_string())
    );
}

// === thinking block, cache-miss notices, quiet startup, changelog, telemetry

#[test]
fn boolean_toggles_round_trip_with_defaults() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert!(!manager.get_hide_thinking_block());
    assert!(!manager.get_show_cache_miss_notices());
    assert!(!manager.get_quiet_startup());
    assert!(!manager.get_collapse_changelog());
    assert!(manager.get_enable_install_telemetry());
    assert!(manager.get_enable_skill_commands());

    manager.set_hide_thinking_block(true);
    manager.set_show_cache_miss_notices(true);
    manager.set_quiet_startup(true);
    manager.set_collapse_changelog(true);
    manager.set_enable_install_telemetry(false);
    manager.set_enable_skill_commands(false);

    assert!(manager.get_hide_thinking_block());
    assert!(manager.get_show_cache_miss_notices());
    assert!(manager.get_quiet_startup());
    assert!(manager.get_collapse_changelog());
    assert!(!manager.get_enable_install_telemetry());
    assert!(!manager.get_enable_skill_commands());
    assert_eq!(
        fixture.read_global(),
        json!({
            "hideThinkingBlock": true,
            "showCacheMissNotices": true,
            "quietStartup": true,
            "collapseChangelog": true,
            "enableInstallTelemetry": false,
            "enableSkillCommands": false,
        })
    );

    fixture.write_global(&json!({"hideThinkingBlock": "yes", "enableSkillCommands": false}));
    let manager = fixture.manager();
    assert!(!manager.get_hide_thinking_block());
    assert!(!manager.get_enable_skill_commands());
}

// === external editor ========================================================

#[test]
fn external_editor_command_prefers_the_setting_then_the_environment() {
    let fixture = fixture();
    fixture.write_global(&json!({"externalEditor": "vim"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_external_editor_command(), "vim");

    fixture.write_global(&json!({"externalEditor": "  "}));
    let manager = fixture.manager();
    assert_eq!(
        manager.get_external_editor_command_with(&empty_env()),
        "nano"
    );
    assert_eq!(
        manager.get_external_editor_command_with(&env_with(&[("VISUAL", "vi")])),
        "vi"
    );
    assert_eq!(
        manager.get_external_editor_command_with(&env_with(&[("VISUAL", ""), ("EDITOR", "ed")])),
        "ed"
    );
    assert_eq!(
        manager.get_external_editor_command_with(&env_with(&[("VISUAL", "vi"), ("EDITOR", "ed")])),
        "vi"
    );
}

// === paths and commands =====================================================

#[test]
fn session_dir_and_shell_path_expand_a_leading_tilde() {
    let home = home_dir();
    let fixture = fixture();
    fixture.write_global(&json!({"sessionDir": "~/sessions", "shellPath": "~/bin/zsh"}));
    let mut manager = fixture.manager();
    assert_eq!(manager.get_session_dir(), Some(format!("{home}/sessions")));
    assert_eq!(manager.get_shell_path(), Some(format!("{home}/bin/zsh")));

    fixture.write_global(&json!({"sessionDir": 5}));
    let reader = fixture.manager();
    assert_eq!(reader.get_session_dir(), None);

    manager.set_shell_path(Some("/bin/bash"));
    assert_eq!(manager.get_shell_path(), Some("/bin/bash".to_string()));
    manager.set_shell_path(None);
    assert_eq!(manager.get_shell_path(), None);
    // upstream's undefined write drops the key instead of storing null
    assert_eq!(fixture.read_global().get("shellPath"), None);
}

#[test]
fn session_dir_and_shell_path_fall_back_on_an_invalid_file_url() {
    let fixture = fixture();
    fixture.write_global(
        &json!({"sessionDir": "file://evil/sessions", "shellPath": "file://evil/zsh"}),
    );
    let manager = fixture.manager();
    // the file:// branch of normalizePath rejects a non-empty host; the
    // getter falls back to the raw string
    assert_eq!(
        manager.get_session_dir(),
        Some("file://evil/sessions".to_string())
    );
    assert_eq!(
        manager.get_shell_path(),
        Some("file://evil/zsh".to_string())
    );
}

#[test]
fn a_non_object_nested_field_drops_the_nested_write() {
    let fixture = fixture();
    fixture.write_global(&json!({"terminal": "flat"}));
    let mut manager = fixture.manager();
    manager.set_show_images(false);
    assert!(manager.drain_errors().is_empty());
    // the nested key cannot land on a non-object entry; the merged view
    // still reads the untouched field
    assert_eq!(fixture.read_global().get("terminal"), Some(&json!("flat")));
    assert!(manager.get_show_images());
}

#[test]
fn shell_command_prefix_round_trips() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert_eq!(manager.get_shell_command_prefix(), None);

    manager.set_shell_command_prefix(Some("timeout 10"));
    assert_eq!(
        manager.get_shell_command_prefix(),
        Some("timeout 10".to_string())
    );
    manager.set_shell_command_prefix(None);
    assert_eq!(manager.get_shell_command_prefix(), None);
    assert_eq!(fixture.read_global().get("shellCommandPrefix"), None);
}

#[test]
fn npm_command_reads_the_array_and_drops_non_strings() {
    let fixture = fixture();
    fixture.write_global(&json!({"npmCommand": ["npm", "exec"]}));
    let mut manager = fixture.manager();
    assert_eq!(
        manager.get_npm_command(),
        Some(vec!["npm".to_string(), "exec".to_string()])
    );

    fixture.write_global(&json!({"npmCommand": [1, "pnpm"]}));
    let reader = fixture.manager();
    assert_eq!(reader.get_npm_command(), Some(vec!["pnpm".to_string()]));

    manager.set_npm_command(Some(&["pnpm".to_string(), "dlx".to_string()]));
    assert_eq!(
        manager.get_npm_command(),
        Some(vec!["pnpm".to_string(), "dlx".to_string()])
    );
    manager.set_npm_command(None);
    assert_eq!(manager.get_npm_command(), None);
    assert_eq!(fixture.read_global().get("npmCommand"), None);
}

// === analytics ==============================================================

#[test]
fn analytics_opt_in_generates_one_tracking_id() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert!(!manager.get_enable_analytics());
    assert_eq!(manager.get_tracking_id(), None);

    manager.set_enable_analytics(true);
    let id = manager.get_tracking_id().expect("tracking id");
    // the uuid v4 shape: 36 chars, version nibble 4, variant nibble 8/9/a/b
    assert_eq!(id.len(), 36);
    assert_eq!(&id[14..15], "4");
    assert!(matches!(&id[19..20], "8" | "9" | "a" | "b"));
    assert_eq!(
        fixture.read_global().get("trackingId"),
        Some(&json!(id.as_str()))
    );

    // the identifier survives toggling off and on and a second opt-in
    manager.set_enable_analytics(false);
    manager.set_enable_analytics(true);
    assert_eq!(manager.get_tracking_id(), Some(id.clone()));
    manager.set_enable_analytics(true);
    assert_eq!(manager.get_tracking_id(), Some(id));

    // a preset identifier is never regenerated
    fixture.write_global(&json!({"trackingId": "preset"}));
    let mut manager = fixture.manager();
    manager.set_enable_analytics(true);
    assert_eq!(manager.get_tracking_id(), Some("preset".to_string()));
}

// === packages and the resource path lists ===================================

#[test]
fn packages_round_trip_preserves_the_values() {
    let fixture = fixture();
    fixture.write_global(&json!({"packages": ["npm:x"]}));
    let mut manager = fixture.manager();
    assert_eq!(manager.get_packages(), vec![json!("npm:x")]);

    manager.set_packages(&[json!("npm:y"), json!(7)]);
    assert_eq!(manager.get_packages(), vec![json!("npm:y"), json!(7)]);
    assert_eq!(
        fixture.read_global().get("packages"),
        Some(&json!(["npm:y", 7]))
    );
}

#[test]
fn resource_path_lists_round_trip_and_drop_non_strings() {
    let fixture = fixture();
    fixture.write_global(&json!({"extensions": ["/a.ts", 5], "skills": ["/s.md"], "prompts": ["/p.md"], "themes": ["/t.json"]}));
    let manager = fixture.manager();
    assert_eq!(manager.get_extension_paths(), vec!["/a.ts".to_string()]);
    assert_eq!(manager.get_skill_paths(), vec!["/s.md".to_string()]);
    assert_eq!(
        manager.get_prompt_template_paths(),
        vec!["/p.md".to_string()]
    );
    assert_eq!(manager.get_theme_paths(), vec!["/t.json".to_string()]);

    let mut manager = fixture.manager();
    manager.set_extension_paths(&["/b.ts".to_string()]);
    manager.set_skill_paths(&["/s2.md".to_string()]);
    manager.set_prompt_template_paths(&["/p2.md".to_string()]);
    manager.set_theme_paths(&["/t2.json".to_string()]);

    assert_eq!(manager.get_extension_paths(), vec!["/b.ts".to_string()]);
    assert_eq!(manager.get_skill_paths(), vec!["/s2.md".to_string()]);
    assert_eq!(
        manager.get_prompt_template_paths(),
        vec!["/p2.md".to_string()]
    );
    assert_eq!(manager.get_theme_paths(), vec!["/t2.json".to_string()]);
    assert_eq!(
        fixture.read_global(),
        json!({
            "extensions": ["/b.ts"],
            "skills": ["/s2.md"],
            "prompts": ["/p2.md"],
            "themes": ["/t2.json"],
        })
    );
}

#[test]
fn project_resource_paths_write_the_project_scope() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();

    manager
        .set_project_packages(&[json!("npm:proj")])
        .expect("trusted write");
    manager
        .set_project_extension_paths(&["/e.ts".to_string()])
        .expect("trusted write");
    manager
        .set_project_skill_paths(&["/s.md".to_string()])
        .expect("trusted write");
    manager
        .set_project_prompt_template_paths(&["/p.md".to_string()])
        .expect("trusted write");
    manager
        .set_project_theme_paths(&["/t.json".to_string()])
        .expect("trusted write");

    assert_eq!(manager.get_packages(), vec![json!("npm:proj")]);
    assert_eq!(manager.get_extension_paths(), vec!["/e.ts".to_string()]);
    assert_eq!(manager.get_skill_paths(), vec!["/s.md".to_string()]);
    assert_eq!(
        manager.get_prompt_template_paths(),
        vec!["/p.md".to_string()]
    );
    assert_eq!(manager.get_theme_paths(), vec!["/t.json".to_string()]);
    assert_eq!(
        fixture.read_project(),
        json!({
            "packages": ["npm:proj"],
            "extensions": ["/e.ts"],
            "skills": ["/s.md"],
            "prompts": ["/p.md"],
            "themes": ["/t.json"],
        })
    );
}

#[test]
fn project_resource_paths_refuse_an_untrusted_project() {
    let fixture = fixture();
    let mut manager = fixture.manager_untrusted();
    let refusal = "Project is not trusted; refusing to write project settings".to_string();
    assert_eq!(manager.set_project_packages(&[]), Err(refusal.clone()));
    assert_eq!(
        manager.set_project_extension_paths(&[]),
        Err(refusal.clone())
    );
    assert_eq!(manager.set_project_skill_paths(&[]), Err(refusal.clone()));
    assert_eq!(
        manager.set_project_prompt_template_paths(&[]),
        Err(refusal.clone())
    );
    assert_eq!(manager.set_project_theme_paths(&[]), Err(refusal));
}

// === the default project trust posture ======================================

#[test]
fn default_project_trust_reads_the_global_scope_only() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    fixture.write_project(&json!({"defaultProjectTrust": "never"}));
    let mut manager = fixture.manager();
    assert_eq!(
        manager.get_default_project_trust(),
        DefaultProjectTrust::Ask
    );
    assert_eq!(DefaultProjectTrust::Always.as_str(), "always");
    assert_eq!(DefaultProjectTrust::Never.as_str(), "never");
    assert_eq!(DefaultProjectTrust::Ask.as_str(), "ask");

    manager.set_default_project_trust(DefaultProjectTrust::Always);
    assert_eq!(
        manager.get_default_project_trust(),
        DefaultProjectTrust::Always
    );
    assert_eq!(
        fixture.read_global().get("defaultProjectTrust"),
        Some(&json!("always"))
    );

    manager.set_default_project_trust(DefaultProjectTrust::Never);
    assert_eq!(
        manager.get_default_project_trust(),
        DefaultProjectTrust::Never
    );

    fixture.write_global(&json!({"defaultProjectTrust": "always"}));
    let manager = fixture.manager();
    assert_eq!(
        manager.get_default_project_trust(),
        DefaultProjectTrust::Always
    );

    fixture.write_global(&json!({"defaultProjectTrust": "bogus"}));
    let manager = fixture.manager();
    assert_eq!(
        manager.get_default_project_trust(),
        DefaultProjectTrust::Ask
    );
}

// === migrations =============================================================

#[test]
fn migrate_moves_queue_mode_to_steering_mode() {
    let fixture = fixture();
    fixture.write_global(&json!({"queueMode": "all"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_steering_mode(), QueueMode::All);
    assert_eq!(manager.merged().get("steeringMode"), Some(&json!("all")));
    assert_eq!(manager.merged().get("queueMode"), None);

    fixture.write_global(&json!({"queueMode": "one-at-a-time"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_steering_mode(), QueueMode::OneAtATime);

    // both present: the stored steering mode wins and nothing moves
    fixture.write_global(&json!({"queueMode": "all", "steeringMode": "one-at-a-time"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_steering_mode(), QueueMode::OneAtATime);
    assert_eq!(manager.merged().get("queueMode"), Some(&json!("all")));

    // the next save persists the migrated shape
    fixture.write_global(&json!({"queueMode": "all"}));
    let mut manager = fixture.manager();
    manager.set_theme("dark");
    let saved = fixture.read_global();
    assert_eq!(saved.get("steeringMode"), Some(&json!("all")));
    assert_eq!(saved.get("queueMode"), None);
    assert_eq!(saved.get("theme"), Some(&json!("dark")));
}

#[test]
fn migrate_maps_websockets_to_transport() {
    let fixture = fixture();
    fixture.write_global(&json!({"websockets": true}));
    let manager = fixture.manager();
    assert_eq!(manager.get_transport(), Transport::Websocket);
    assert_eq!(manager.merged().get("websockets"), None);

    fixture.write_global(&json!({"websockets": false}));
    let manager = fixture.manager();
    assert_eq!(manager.get_transport(), Transport::Sse);

    // an explicit transport is never clobbered
    fixture.write_global(&json!({"websockets": true, "transport": "sse"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_transport(), Transport::Sse);
    assert_eq!(manager.merged().get("websockets"), Some(&json!(true)));

    // a non-boolean legacy value is left alone
    fixture.write_global(&json!({"websockets": "yes"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_transport(), Transport::Auto);
    assert_eq!(manager.merged().get("websockets"), Some(&json!("yes")));
}

#[test]
fn migrate_flattens_the_skills_object() {
    let fixture = fixture();
    fixture.write_global(
        &json!({"skills": {"enableSkillCommands": false, "customDirectories": ["/a", "/b"]}}),
    );
    let manager = fixture.manager();
    assert_eq!(
        manager.get_skill_paths(),
        vec!["/a".to_string(), "/b".to_string()]
    );
    assert!(!manager.get_enable_skill_commands());
    assert_eq!(
        manager.merged().get("enableSkillCommands"),
        Some(&json!(false))
    );

    fixture.write_global(&json!({"skills": {"customDirectories": []}}));
    let manager = fixture.manager();
    assert!(manager.get_skill_paths().is_empty());
    assert_eq!(manager.merged().get("skills"), None);

    fixture.write_global(&json!({"skills": {"enableSkillCommands": true}}));
    let manager = fixture.manager();
    assert!(manager.get_enable_skill_commands());
    assert_eq!(manager.merged().get("skills"), None);

    // a top-level enableSkillCommands is not clobbered by the object's copy
    fixture.write_global(
        &json!({"skills": {"enableSkillCommands": false}, "enableSkillCommands": true}),
    );
    let manager = fixture.manager();
    assert!(manager.get_enable_skill_commands());

    // an array survives untouched
    fixture.write_global(&json!({"skills": ["/x"]}));
    let manager = fixture.manager();
    assert_eq!(manager.get_skill_paths(), vec!["/x".to_string()]);
}

#[test]
fn migrate_moves_retry_max_delay_into_the_provider() {
    let fixture = fixture();
    fixture.write_global(&json!({"retry": {"maxDelayMs": 500}}));
    let manager = fixture.manager();
    let retry = manager
        .merged()
        .get("retry")
        .cloned()
        .expect("retry object");
    assert_eq!(
        retry
            .get("provider")
            .and_then(|provider| provider.get("maxRetryDelayMs")),
        Some(&json!(500))
    );
    assert_eq!(retry.get("maxDelayMs"), None);
    assert_eq!(
        manager.get_provider_retry_settings().max_retry_delay_ms,
        500
    );

    // an existing provider override wins and the number is carried verbatim
    fixture.write_global(
        &json!({"retry": {"maxDelayMs": 500.5, "provider": {"maxRetryDelayMs": 900}}}),
    );
    let manager = fixture.manager();
    let retry = manager
        .merged()
        .get("retry")
        .cloned()
        .expect("retry object");
    assert_eq!(
        retry
            .get("provider")
            .and_then(|provider| provider.get("maxRetryDelayMs")),
        Some(&json!(900))
    );
    assert_eq!(retry.get("maxDelayMs"), None);

    // a null provider override migrates and keeps the provider's other keys
    fixture.write_global(&json!({"retry": {"maxDelayMs": 500, "provider": {"timeoutMs": 7, "maxRetryDelayMs": null}}}));
    let manager = fixture.manager();
    let settings = manager.get_provider_retry_settings();
    assert_eq!(settings.timeout_ms, Some(7));
    assert_eq!(settings.max_retry_delay_ms, 500);

    // a non-number maxDelayMs does not migrate but still deletes, and a
    // retry array is left alone
    fixture.write_global(&json!({"retry": {"maxDelayMs": "500"}}));
    let manager = fixture.manager();
    assert_eq!(manager.merged().get("retry"), Some(&json!({})));
    fixture.write_global(&json!({"retry": [1]}));
    let manager = fixture.manager();
    assert_eq!(manager.merged().get("retry"), Some(&json!([1])));
}

// === load, persist and reload error paths ===================================

#[test]
fn a_non_object_settings_file_loads_empty_without_error() {
    let fixture = fixture();
    fixture.write_raw_global("[1,2]");
    fixture.write_raw_project("null");
    let mut manager = fixture.manager();
    assert!(manager.drain_errors().is_empty());
    assert!(manager.merged().is_empty());
    assert!(manager.get_global_settings().is_empty());
    assert!(manager.get_project_settings().is_empty());
    assert_eq!(manager.get_theme_setting(), None);
}

#[test]
fn a_non_object_current_file_is_replaced_on_save() {
    let fixture = fixture();
    fixture.write_raw_global("[9]");
    let mut manager = fixture.manager();
    assert!(manager.drain_errors().is_empty());
    manager.set_theme("dark");
    assert_eq!(fixture.read_global(), json!({"theme": "dark"}));
}

#[test]
fn save_is_suppressed_while_a_load_error_stands() {
    let fixture = fixture();
    fixture.write_raw_global("{");
    let mut manager = fixture.manager();
    assert_eq!(manager.drain_errors().len(), 1);

    manager.set_theme("dark");
    assert!(manager.drain_errors().is_empty());
    assert_eq!(
        std::fs::read_to_string(fixture.global_settings_path()).expect("raw read"),
        "{"
    );
    // the in-memory view updated even though the write is suppressed
    assert_eq!(manager.get_theme_setting(), Some("dark".to_string()));
}

#[test]
fn a_failed_save_records_and_retries_with_tracking_intact() {
    let root = tempfile::tempdir().expect("scratch root");
    // the agent "dir" is a regular file: the save's create_dir_all fails
    let blocker = root.path().join("agent");
    std::fs::write(&blocker, b"not a directory").expect("blocker write");
    let project = root.path().join("project");
    std::fs::create_dir_all(project.join(".pi")).expect("project .pi dir");
    let mut manager = SettingsManager::<FileSettingsStorage>::create(
        &project.to_string_lossy(),
        &blocker.to_string_lossy(),
        SettingsManagerCreateOptions::default(),
    );
    assert!(manager.drain_errors().is_empty());

    manager.set_theme("dark");
    let errors = manager.drain_errors();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].scope, SettingsScope::Global);
    assert_eq!(
        errors[0].path.as_deref(),
        blocker.join("settings.json").to_str()
    );
    assert!(!errors[0].message.is_empty());
    assert!(!blocker.join("settings.json").exists());

    // the modified tracking survives the failed write: the retry persists
    // both fields once the directory exists
    std::fs::remove_file(&blocker).expect("blocker remove");
    std::fs::create_dir_all(&blocker).expect("agent dir create");
    manager.set_quiet_startup(true);
    assert!(manager.drain_errors().is_empty());
    assert_eq!(
        fixture_read(&blocker),
        json!({"theme": "dark", "quietStartup": true})
    );
}

fn fixture_read(agent_dir: &std::path::Path) -> Value {
    let content = std::fs::read_to_string(agent_dir.join("settings.json")).expect("settings read");
    serde_json::from_str(&content).expect("settings parse")
}

#[test]
fn a_failed_project_save_records_and_the_file_stays_absent() {
    let root = tempfile::tempdir().expect("scratch root");
    let cwd = root.path().join("project");
    std::fs::create_dir_all(&cwd).expect("project dir");
    // .pi is a regular file: the project save's create_dir_all fails
    std::fs::write(cwd.join(".pi"), b"not a directory").expect(".pi blocker write");
    let agent = root.path().join("agent");
    std::fs::create_dir_all(&agent).expect("agent dir");
    let mut manager = SettingsManager::<FileSettingsStorage>::create(
        &cwd.to_string_lossy(),
        &agent.to_string_lossy(),
        SettingsManagerCreateOptions::default(),
    );
    assert!(manager.drain_errors().is_empty());

    manager
        .set_project_extension_paths(&["/e.ts".to_string()])
        .expect("trusted project write");
    let errors = manager.drain_errors();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].scope, SettingsScope::Project);
    assert_eq!(
        errors[0].path.as_deref(),
        cwd.join(".pi").join("settings.json").to_str()
    );
    assert!(!cwd.join(".pi").join("settings.json").exists());
}

#[test]
fn project_save_is_suppressed_while_a_project_load_error_stands() {
    let fixture = fixture();
    fixture.write_raw_project("{");
    let mut manager = fixture.manager();
    assert_eq!(manager.drain_errors().len(), 1);

    manager
        .set_project_skill_paths(&["/s.md".to_string()])
        .expect("trusted project write");
    assert!(manager.drain_errors().is_empty());
    assert_eq!(fixture.read_raw_project(), "{");
}

#[test]
fn reload_keeps_previous_settings_and_records_errors() {
    let fixture = fixture();
    fixture.write_global(&json!({"defaultProvider": "anthropic"}));
    fixture.write_project(&json!({"quietStartup": true}));
    let mut manager = fixture.manager();
    assert!(manager.drain_errors().is_empty());

    fixture.write_raw_global("{");
    manager.reload();
    let errors = manager.drain_errors();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].scope, SettingsScope::Global);
    // the failed scope keeps its previous settings
    assert_eq!(
        manager.get_default_provider(),
        Some("anthropic".to_string())
    );
    assert!(manager.get_quiet_startup());

    fixture.write_raw_project("}");
    manager.reload();
    let errors = manager.drain_errors();
    assert_eq!(errors.len(), 2);
    assert!(
        errors
            .iter()
            .any(|error| error.scope == SettingsScope::Project)
    );
    assert!(
        errors
            .iter()
            .any(|error| error.scope == SettingsScope::Global)
    );
    assert_eq!(
        manager.get_default_provider(),
        Some("anthropic".to_string())
    );
    assert!(manager.get_quiet_startup());

    fixture.write_global(&json!({"defaultProvider": "openai"}));
    fixture.write_project(&json!({"quietStartup": false}));
    manager.reload();
    assert!(manager.drain_errors().is_empty());
    assert_eq!(manager.get_default_provider(), Some("openai".to_string()));
    assert!(!manager.get_quiet_startup());
}

// === the project trust flip =================================================

#[test]
fn set_project_trusted_transitions_reload_and_drop() {
    let fixture = fixture();
    fixture.write_project(&json!({"theme": "project-dark"}));
    let mut manager = fixture.manager_untrusted();
    assert!(!manager.is_project_trusted());
    assert_eq!(manager.get_theme_setting(), None);
    assert!(manager.get_project_settings().is_empty());

    // re-untrusting is a no-op
    manager.set_project_trusted(false);
    assert!(manager.drain_errors().is_empty());
    assert_eq!(manager.get_theme_setting(), None);

    manager.set_project_trusted(true);
    assert!(manager.is_project_trusted());
    assert_eq!(
        manager.get_theme_setting(),
        Some("project-dark".to_string())
    );
    assert_eq!(
        manager.get_project_settings().get("theme"),
        Some(&json!("project-dark"))
    );

    // re-trusting is a no-op
    manager.set_project_trusted(true);
    assert!(manager.drain_errors().is_empty());

    manager.set_project_trusted(false);
    assert_eq!(manager.get_theme_setting(), None);
    assert!(manager.get_project_settings().is_empty());

    // trusting over a broken file records the reload error
    fixture.write_raw_project("{");
    manager.set_project_trusted(true);
    let errors = manager.drain_errors();
    assert!(
        errors
            .iter()
            .any(|error| error.scope == SettingsScope::Project)
    );
    assert_eq!(manager.get_theme_setting(), None);
}

// === save-time failure paths on the file storage ============================

#[test]
fn a_corrupt_current_file_records_the_save_error() {
    let fixture = fixture();
    fixture.write_raw_global("{");
    let mut manager = fixture.manager();
    assert_eq!(manager.drain_errors().len(), 1);

    // clear the load error with a valid file, then corrupt it again: the
    // save's parse of the current content fails inside the lock
    fixture.write_global(&json!({}));
    manager.reload();
    manager.drain_errors();
    fixture.write_raw_global("{");
    manager.set_theme("dark");
    let errors = manager.drain_errors();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].scope, SettingsScope::Global);
    assert!(
        errors[0].message.starts_with("Failed to parse settings: "),
        "{}",
        errors[0].message
    );
    // the modified tracking survives for the next attempt; the session's
    // own modified field wins over the external edit
    fixture.write_global(&json!({"theme": "kept"}));
    manager.set_quiet_startup(true);
    assert!(manager.drain_errors().is_empty());
    let saved = fixture.read_global();
    assert_eq!(saved.get("theme"), Some(&json!("dark")));
    assert_eq!(saved.get("quietStartup"), Some(&json!(true)));
}

#[test]
fn a_lock_path_held_by_a_regular_file_records_the_save_error() {
    let root = tempfile::tempdir().expect("scratch root");
    let agent = root.path().join("agent");
    std::fs::create_dir_all(&agent).expect("agent dir");
    std::fs::write(agent.join("settings.json"), "{}").expect("settings write");
    std::fs::write(agent.join("settings.json.lock"), b"not a directory")
        .expect("lock blocker write");
    let project = root.path().join("project");
    std::fs::create_dir_all(project.join(".pi")).expect("project .pi dir");
    let mut manager = SettingsManager::<FileSettingsStorage>::create(
        &project.to_string_lossy(),
        &agent.to_string_lossy(),
        SettingsManagerCreateOptions::default(),
    );
    // the load already rides the contended lock and records the failure
    let errors = manager.drain_errors();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].message, "locked");

    // the load error suppresses saves; clearing the blocker and reloading
    // restores the write path
    std::fs::remove_file(agent.join("settings.json.lock")).expect("lock blocker remove");
    manager.reload();
    assert!(manager.drain_errors().is_empty());
    manager.set_theme("dark");
    assert!(manager.drain_errors().is_empty());
    assert_eq!(fixture_read(&agent).get("theme"), Some(&json!("dark")));
}

// === storage doubles: the SettingsLockFn error paths ========================

struct DeadStorage;

impl SettingsStorage for DeadStorage {
    fn with_lock(&self, _scope: SettingsScope, _f: SettingsLockFn<'_>) -> Result<(), String> {
        Err("lock backend failed".to_string())
    }
}

struct RefusingWriteStorage;

impl SettingsStorage for RefusingWriteStorage {
    fn with_lock(&self, _scope: SettingsScope, f: SettingsLockFn<'_>) -> Result<(), String> {
        let pending = f(None)?;
        if pending.is_some() {
            return Err("backend write refused".to_string());
        }
        Ok(())
    }
}

#[test]
fn a_dead_lock_backend_records_load_errors_for_both_scopes() {
    let mut manager =
        SettingsManager::from_storage(DeadStorage, SettingsManagerCreateOptions::default());
    let errors = manager.drain_errors();
    assert_eq!(errors.len(), 2);
    assert_eq!(errors[0].scope, SettingsScope::Global);
    assert_eq!(errors[0].message, "lock backend failed");
    assert_eq!(errors[0].path, None);
    assert_eq!(errors[1].scope, SettingsScope::Project);
    assert_eq!(errors[1].message, "lock backend failed");

    // the load error suppresses only the write: the in-memory merge ran
    manager.set_theme("dark");
    assert!(manager.drain_errors().is_empty());
    assert_eq!(manager.get_theme_setting(), Some("dark".to_string()));
}

#[test]
fn diagnostics_name_the_scope_when_the_storage_has_no_paths() {
    let mut manager =
        SettingsManager::from_storage(DeadStorage, SettingsManagerCreateOptions::default());
    let diagnostics = collect_settings_diagnostics(&mut manager);
    assert_eq!(
        diagnostics,
        vec![
            SettingsDiagnostic {
                diagnostic_type: "warning".to_string(),
                message: "Invalid global settings: lock backend failed".to_string(),
            },
            SettingsDiagnostic {
                diagnostic_type: "warning".to_string(),
                message: "Invalid project settings: lock backend failed".to_string(),
            },
        ]
    );
}

#[test]
fn a_failing_write_backend_records_the_save_error() {
    let mut manager = SettingsManager::from_storage(
        RefusingWriteStorage,
        SettingsManagerCreateOptions::default(),
    );
    assert!(manager.drain_errors().is_empty());

    manager.set_theme("dark");
    let errors = manager.drain_errors();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].scope, SettingsScope::Global);
    assert_eq!(errors[0].message, "backend write refused");
    assert_eq!(errors[0].path, None);
}

// === overrides and the scope views ==========================================

#[test]
fn apply_overrides_deep_merges_nested_objects() {
    let fixture = fixture();
    fixture.write_global(&json!({"tui": {"a": 1, "b": 2}, "packages": ["npm:x"], "theme": "dark"}));
    let mut manager = fixture.manager();

    let overrides = json!({"tui": {"b": 9}, "packages": ["npm:y"], "extra": true});
    manager.apply_overrides(overrides.as_object().expect("override object"));

    assert_eq!(manager.merged().get("tui"), Some(&json!({"a": 1, "b": 9})));
    assert_eq!(manager.merged().get("packages"), Some(&json!(["npm:y"])));
    assert_eq!(manager.merged().get("theme"), Some(&json!("dark")));
    assert_eq!(manager.merged().get("extra"), Some(&json!(true)));
    // overrides never touch the scopes or the file
    assert_eq!(
        fixture.read_global().get("tui"),
        Some(&json!({"a": 1, "b": 2}))
    );
}

#[test]
fn the_scope_views_return_the_loaded_clones() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "dark"}));
    fixture.write_project(&json!({"quietStartup": true}));
    let manager = fixture.manager();
    let global = json!({"theme": "dark"});
    let project = json!({"quietStartup": true});
    assert_eq!(
        manager.get_global_settings(),
        *global.as_object().expect("global object")
    );
    assert_eq!(
        manager.get_project_settings(),
        *project.as_object().expect("project object")
    );
}

// === the remaining typed accessors ==========================================

#[test]
fn terminal_capability_overrides_map_the_images_field() {
    let fixture = fixture();
    for (wire, images) in [
        ("kitty", Some(Some(ImageProtocol::Kitty))),
        ("iterm2", Some(Some(ImageProtocol::Iterm2))),
        ("unknown", None),
    ] {
        fixture.write_global(&json!({ "terminal": { "images": wire } }));
        let manager = fixture.manager();
        let expected = CapabilityOverrides {
            images,
            true_color: None,
            hyperlinks: None,
        };
        assert_eq!(manager.get_terminal_capability_overrides(), expected);
    }

    fixture.write_global(
        &json!({"terminal": {"images": false, "trueColor": true, "hyperlinks": false}}),
    );
    let manager = fixture.manager();
    assert_eq!(
        manager.get_terminal_capability_overrides(),
        CapabilityOverrides {
            images: Some(None),
            true_color: Some(true),
            hyperlinks: Some(false),
        }
    );
}

#[test]
fn show_images_and_width_round_trip() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let manager = fixture.manager();
    assert!(manager.get_show_images());
    assert_eq!(manager.get_image_width_cells(), 60);

    fixture.write_global(&json!({"terminal": {"showImages": false}}));
    let reader = fixture.manager();
    assert!(!reader.get_show_images());

    fixture.write_global(&json!({"terminal": {"imageWidthCells": 0.5}}));
    let reader = fixture.manager();
    assert_eq!(reader.get_image_width_cells(), 1);

    fixture.write_global(&json!({"terminal": {"imageWidthCells": 100.9}}));
    let reader = fixture.manager();
    assert_eq!(reader.get_image_width_cells(), 100);

    fixture.write_global(&json!({"terminal": {"imageWidthCells": "x"}}));
    let reader = fixture.manager();
    assert_eq!(reader.get_image_width_cells(), 60);

    let mut manager = fixture.manager();
    manager.set_show_images(false);
    assert!(!manager.get_show_images());
    manager.set_image_width_cells(0.5);
    assert_eq!(manager.get_image_width_cells(), 1);
    manager.set_image_width_cells(70.5);
    assert_eq!(manager.get_image_width_cells(), 70);
    assert_eq!(
        fixture.read_global().get("terminal"),
        Some(&json!({"showImages": false, "imageWidthCells": 70}))
    );
}

#[test]
fn clear_on_shrink_setting_beats_the_environment() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert!(!manager.get_clear_on_shrink_with(&empty_env()));
    assert!(manager.get_clear_on_shrink_with(&env_with(&[("PI_CLEAR_ON_SHRINK", "1")])));

    fixture.write_global(&json!({"terminal": {"clearOnShrink": false}}));
    let reader = fixture.manager();
    assert!(!reader.get_clear_on_shrink_with(&env_with(&[("PI_CLEAR_ON_SHRINK", "1")])));
    assert!(!reader.get_clear_on_shrink_with(&env_with(&[("PI_CLEAR_ON_SHRINK", "0")])));

    manager.set_clear_on_shrink(true);
    assert!(manager.get_clear_on_shrink());
    assert_eq!(
        fixture.read_global().get("terminal"),
        Some(&json!({"clearOnShrink": true}))
    );
}

#[test]
fn show_terminal_progress_round_trips() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert!(!manager.get_show_terminal_progress());
    manager.set_show_terminal_progress(true);
    assert!(manager.get_show_terminal_progress());
    assert_eq!(
        fixture.read_global().get("terminal"),
        Some(&json!({"showTerminalProgress": true}))
    );
}

#[test]
fn the_fullscreen_trio_round_trips() {
    let fixture = fixture();
    fixture.write_global(&json!({"tuiMode": "fullscreen", "fullscreenExitOutput": "resume-hint", "fullscreenScrollbar": "always"}));
    let mut manager = fixture.manager();
    assert_eq!(manager.get_tui_mode(), TuiMode::Fullscreen);
    assert_eq!(
        manager.get_fullscreen_exit_output(),
        FullscreenExitOutput::ResumeHint
    );
    assert_eq!(
        manager.get_fullscreen_scrollbar(),
        ScrollViewScrollbar::Always
    );

    manager.set_tui_mode(TuiMode::Regular);
    assert_eq!(manager.get_tui_mode(), TuiMode::Regular);
    manager.set_fullscreen_exit_output(FullscreenExitOutput::Transcript);
    assert_eq!(
        manager.get_fullscreen_exit_output(),
        FullscreenExitOutput::Transcript
    );
    manager.set_fullscreen_scrollbar(ScrollViewScrollbar::Hidden);
    assert_eq!(
        manager.get_fullscreen_scrollbar(),
        ScrollViewScrollbar::Hidden
    );
    manager.set_fullscreen_scrollbar(ScrollViewScrollbar::Always);
    assert_eq!(
        manager.get_fullscreen_scrollbar(),
        ScrollViewScrollbar::Always
    );
    manager.set_fullscreen_scrollbar(ScrollViewScrollbar::Auto);
    assert_eq!(
        manager.get_fullscreen_scrollbar(),
        ScrollViewScrollbar::Auto
    );
    assert_eq!(
        fixture.read_global(),
        json!({
            "tuiMode": "regular",
            "fullscreenExitOutput": "transcript",
            "fullscreenScrollbar": "auto",
        })
    );
}

#[test]
fn image_auto_resize_and_block_round_trip() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert!(manager.get_image_auto_resize());
    assert!(!manager.get_block_images());

    fixture.write_global(&json!({"images": {"autoResize": false}}));
    let reader = fixture.manager();
    assert!(!reader.get_image_auto_resize());

    manager.set_image_auto_resize(false);
    assert!(!manager.get_image_auto_resize());
    manager.set_block_images(true);
    assert!(manager.get_block_images());
    assert_eq!(
        fixture.read_global().get("images"),
        Some(&json!({"autoResize": false, "blockImages": true}))
    );
}

#[test]
fn double_escape_action_pins_the_default() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert_eq!(manager.get_double_escape_action(), "tree");

    for (wire, expected) in [("fork", "fork"), ("none", "none"), ("bogus", "tree")] {
        fixture.write_global(&json!({ "doubleEscapeAction": wire }));
        let manager = fixture.manager();
        assert_eq!(manager.get_double_escape_action(), expected);
    }

    manager.set_double_escape_action("none");
    assert_eq!(manager.get_double_escape_action(), "none");
    assert_eq!(
        fixture.read_global().get("doubleEscapeAction"),
        Some(&json!("none"))
    );
}

#[test]
fn tree_filter_mode_pins_the_vocabulary() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert_eq!(manager.get_tree_filter_mode(), "default");

    for mode in ["no-tools", "user-only", "labeled-only", "all"] {
        fixture.write_global(&json!({ "treeFilterMode": mode }));
        let manager = fixture.manager();
        assert_eq!(manager.get_tree_filter_mode(), mode);
    }

    fixture.write_global(&json!({"treeFilterMode": "bogus"}));
    let reader = fixture.manager();
    assert_eq!(reader.get_tree_filter_mode(), "default");

    manager.set_tree_filter_mode("all");
    assert_eq!(manager.get_tree_filter_mode(), "all");
    assert_eq!(
        fixture.read_global().get("treeFilterMode"),
        Some(&json!("all"))
    );
}

#[test]
fn hardware_cursor_setting_beats_the_environment() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert!(manager.get_show_hardware_cursor_with(&env_with(&[("PI_HARDWARE_CURSOR", "1")])));
    assert!(!manager.get_show_hardware_cursor_with(&empty_env()));

    fixture.write_global(&json!({"showHardwareCursor": false}));
    let reader = fixture.manager();
    assert!(!reader.get_show_hardware_cursor_with(&env_with(&[("PI_HARDWARE_CURSOR", "1")])));
    assert!(!reader.get_show_hardware_cursor_with(&env_with(&[("PI_HARDWARE_CURSOR", "0")])));

    fixture.write_global(&json!({"showHardwareCursor": true}));
    let reader = fixture.manager();
    assert!(reader.get_show_hardware_cursor_with(&empty_env()));

    manager.set_show_hardware_cursor(true);
    assert!(manager.get_show_hardware_cursor());
    assert_eq!(
        fixture.read_global().get("showHardwareCursor"),
        Some(&json!(true))
    );
}

#[test]
fn editor_padding_gets_and_clamps() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert_eq!(manager.get_editor_padding_x(), 0);

    fixture.write_global(&json!({"editorPaddingX": 2}));
    let reader = fixture.manager();
    assert_eq!(reader.get_editor_padding_x(), 2);

    manager.set_editor_padding_x(-1.5);
    assert_eq!(manager.get_editor_padding_x(), 0);
    manager.set_editor_padding_x(2.7);
    assert_eq!(manager.get_editor_padding_x(), 2);
    manager.set_editor_padding_x(9.0);
    assert_eq!(manager.get_editor_padding_x(), 3);
    assert_eq!(fixture.read_global().get("editorPaddingX"), Some(&json!(3)));
}

#[test]
fn output_pad_reads_and_writes() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert_eq!(manager.get_output_pad(), 1);

    fixture.write_global(&json!({"outputPad": 0}));
    let reader = fixture.manager();
    assert_eq!(reader.get_output_pad(), 0);

    manager.set_output_pad(0);
    assert_eq!(manager.get_output_pad(), 0);
    manager.set_output_pad(1);
    assert_eq!(manager.get_output_pad(), 1);
    assert_eq!(fixture.read_global().get("outputPad"), Some(&json!(1)));
}

#[test]
fn autocomplete_cap_clamps() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert_eq!(manager.get_autocomplete_max_visible(), 5);

    fixture.write_global(&json!({"autocompleteMaxVisible": 8}));
    let reader = fixture.manager();
    assert_eq!(reader.get_autocomplete_max_visible(), 8);

    manager.set_autocomplete_max_visible(1.9);
    assert_eq!(manager.get_autocomplete_max_visible(), 3);
    manager.set_autocomplete_max_visible(25.0);
    assert_eq!(manager.get_autocomplete_max_visible(), 20);
    manager.set_autocomplete_max_visible(10.5);
    assert_eq!(manager.get_autocomplete_max_visible(), 10);
    assert_eq!(
        fixture.read_global().get("autocompleteMaxVisible"),
        Some(&json!(10))
    );
}

#[test]
fn code_block_indent_defaults_to_two_spaces() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let manager = fixture.manager();
    assert_eq!(manager.get_code_block_indent(), "  ");

    fixture.write_global(&json!({"markdown": {"codeBlockIndent": "\t"}}));
    let manager = fixture.manager();
    assert_eq!(manager.get_code_block_indent(), "\t");
}

#[test]
fn mermaid_mode_pins_the_vocabulary() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert_eq!(
        manager.get_mermaid_rendering_mode(),
        MermaidRenderingMode::Streaming
    );

    fixture.write_global(&json!({"markdown": {"mermaid": "off"}}));
    let reader = fixture.manager();
    assert_eq!(
        reader.get_mermaid_rendering_mode(),
        MermaidRenderingMode::Off
    );

    fixture.write_global(&json!({"markdown": {"mermaid": "final"}}));
    let reader = fixture.manager();
    assert_eq!(
        reader.get_mermaid_rendering_mode(),
        MermaidRenderingMode::Final
    );

    fixture.write_global(&json!({"markdown": {"mermaid": "bogus"}}));
    let reader = fixture.manager();
    assert_eq!(
        reader.get_mermaid_rendering_mode(),
        MermaidRenderingMode::Streaming
    );

    manager.set_mermaid_rendering_mode(MermaidRenderingMode::Off);
    assert_eq!(
        manager.get_mermaid_rendering_mode(),
        MermaidRenderingMode::Off
    );
    assert_eq!(
        fixture.read_global().get("markdown"),
        Some(&json!({"mermaid": "off"}))
    );

    manager.set_mermaid_rendering_mode(MermaidRenderingMode::Final);
    assert_eq!(
        manager.get_mermaid_rendering_mode(),
        MermaidRenderingMode::Final
    );
    assert_eq!(
        fixture.read_global().get("markdown"),
        Some(&json!({"mermaid": "final"}))
    );

    manager.set_mermaid_rendering_mode(MermaidRenderingMode::Streaming);
    assert_eq!(
        manager.get_mermaid_rendering_mode(),
        MermaidRenderingMode::Streaming
    );
    assert_eq!(
        fixture.read_global().get("markdown"),
        Some(&json!({"mermaid": "streaming"}))
    );
}

#[test]
fn warnings_get_and_set() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let mut manager = fixture.manager();
    assert!(manager.get_warnings().is_empty());

    let warnings = json!({"border": true});
    manager.set_warnings(warnings.as_object().expect("warnings object"));
    assert_eq!(manager.get_warnings().get("border"), Some(&json!(true)));
    assert_eq!(
        fixture.read_global().get("warnings"),
        Some(&json!({"border": true}))
    );
}

#[test]
fn thinking_budgets_read_the_object() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let manager = fixture.manager();
    assert_eq!(manager.get_thinking_budgets(), None);

    fixture.write_global(&json!({"thinkingBudgets": {"high": 32_000}}));
    let manager = fixture.manager();
    assert_eq!(
        manager.get_thinking_budgets(),
        Some(
            json!({"high": 32_000})
                .as_object()
                .expect("budgets object")
                .clone()
        )
    );
}

#[test]
fn the_enabled_models_and_default_tools_lists_round_trip() {
    let fixture = fixture();
    fixture.write_global(&json!({}));
    let manager = fixture.manager();
    assert_eq!(manager.get_enabled_models(), None);
    assert_eq!(manager.get_default_tools(), None);

    fixture.write_global(&json!({"enabledModels": ["anthropic/*", 5], "defaultTools": ["read"]}));
    let reader = fixture.manager();
    assert_eq!(
        reader.get_enabled_models(),
        Some(vec!["anthropic/*".to_string()])
    );
    assert_eq!(reader.get_default_tools(), Some(vec!["read".to_string()]));

    let mut manager = fixture.manager();
    manager.set_enabled_models(Some(&["openai/*".to_string()]));
    assert_eq!(
        manager.get_enabled_models(),
        Some(vec!["openai/*".to_string()])
    );
    manager.set_enabled_models(None);
    assert_eq!(manager.get_enabled_models(), None);
    manager.set_default_tools(Some(&["bash".to_string()]));
    assert_eq!(manager.get_default_tools(), Some(vec!["bash".to_string()]));
    manager.set_default_tools(None);
    assert_eq!(manager.get_default_tools(), None);
    assert_eq!(fixture.read_global(), json!({}));
}

#[test]
fn the_theme_setting_distinguishes_the_fixed_theme() {
    let fixture = fixture();
    fixture.write_global(&json!({"theme": "dark"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_theme(), Some("dark".to_string()));

    fixture.write_global(&json!({"theme": "auto/light"}));
    let manager = fixture.manager();
    assert_eq!(manager.get_theme_setting(), Some("auto/light".to_string()));
    assert_eq!(manager.get_theme(), None);

    let mut manager = fixture.manager();
    manager.set_theme("light");
    assert_eq!(manager.get_theme(), Some("light".to_string()));
}

#[test]
fn debug_impls_name_the_paths_and_the_trust_flag() {
    let storage = FileSettingsStorage::new("/cwd", "/agent");
    let debug = format!("{storage:?}");
    assert!(debug.contains("/agent/settings.json"), "{debug}");
    assert!(debug.contains("/cwd/.pi/settings.json"), "{debug}");

    let fixture = fixture();
    let manager = fixture.manager();
    let debug = format!("{manager:?}");
    assert!(debug.contains("project_trusted: true"), "{debug}");
}

#[test]
fn the_model_key_spells_the_override_key() {
    let key = ModelKey {
        provider: "openai".to_string(),
        id: "gpt-x".to_string(),
    };
    assert_eq!(key.key(), "openai/gpt-x");
}

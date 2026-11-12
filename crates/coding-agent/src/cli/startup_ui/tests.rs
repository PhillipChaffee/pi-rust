//! The `first-time-setup.test.ts` predicate cases, ported 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`; the `analytics settings`
//! block lives in `tests/settings_boundary.rs`, where the settings slice
//! ported it.
//!
//! `first-time-setup-fork.test.ts` exercises upstream's package-rename
//! mechanism; the Rust build pins the official identity, so its subject has
//! no counterpart beyond the `is_official_distribution` false arm pinned
//! below.

#![expect(
    clippy::expect_used,
    reason = "the temp-dir setup panics the test when the filesystem cannot provide a scratch path"
)]

use super::*;

use crate::config::ENV_AGENT_DIR;

fn env(experimental: bool, agent_dir: Option<&str>) -> EnvLookup {
    let agent_dir = agent_dir.map(ToString::to_string);
    Box::new(move |key| {
        if key == "PI_EXPERIMENTAL" {
            Some(if experimental { "1" } else { "0" }.to_string())
        } else if key == ENV_AGENT_DIR {
            agent_dir.clone()
        } else {
            None
        }
    })
}

#[test]
fn runs_when_experimental_default_agent_dir_and_no_settings() {
    let dir = tempfile::tempdir().expect("tempdir");
    let settings_path = dir.path().join("settings.json");
    assert!(should_run_first_time_setup_with(
        &env(true, None),
        &settings_path
    ));
}

#[test]
fn does_not_run_when_experimental_features_are_disabled() {
    let dir = tempfile::tempdir().expect("tempdir");
    let settings_path = dir.path().join("settings.json");
    assert!(!should_run_first_time_setup_with(
        &env(false, None),
        &settings_path
    ));
}

#[test]
fn does_not_run_with_a_custom_agent_dir() {
    let dir = tempfile::tempdir().expect("tempdir");
    let settings_path = dir.path().join("settings.json");
    let custom_dir = dir.path().to_str().expect("utf-8 tempdir path");
    assert!(!should_run_first_time_setup_with(
        &env(true, Some(custom_dir)),
        &settings_path
    ));
}

#[test]
fn does_not_run_after_setup_completed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let settings_path = dir.path().join("settings.json");
    std::fs::write(&settings_path, "{}").expect("write settings");
    assert!(!should_run_first_time_setup_with(
        &env(true, None),
        &settings_path
    ));
}

#[test]
fn rejects_a_forked_distribution() {
    assert!(is_official_distribution(
        OFFICIAL_PACKAGE_NAME,
        APP_NAME,
        CONFIG_DIR_NAME
    ));
    assert!(!is_official_distribution(
        "@example/pi-coding-agent",
        APP_NAME,
        CONFIG_DIR_NAME
    ));
}

#[test]
fn plain_entry_reads_the_process_environment_and_settings_path() {
    assert_eq!(
        should_run_first_time_setup(),
        should_run_first_time_setup_with(&default_env_lookup(), &get_settings_path())
    );
}

/// The built-in dark palette on disk, a valid theme document for loader tests.
const DARK_JSON: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/modes/interactive/theme/dark.json"
);

/// The built-in light palette on disk.
const LIGHT_JSON: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/modes/interactive/theme/light.json"
);

fn theme_resource(path: &str, enabled: bool) -> ResolvedResource {
    ResolvedResource {
        path: path.to_string(),
        enabled,
        metadata: crate::package_manager::PathMetadata {
            source: "local".to_string(),
            scope: crate::source_info::SourceScope::User,
            origin: crate::source_info::SourceOrigin::TopLevel,
            base_dir: None,
        },
    }
}

#[test]
fn loads_only_enabled_themes() {
    let themes = load_themes(&[
        theme_resource(DARK_JSON, true),
        theme_resource(LIGHT_JSON, false),
    ]);
    let names: Vec<&str> = themes.iter().filter_map(|theme| theme.name()).collect();
    assert_eq!(names, vec!["dark"]);
}

#[test]
fn keeps_the_first_of_two_themes_with_one_name() {
    let themes = load_themes(&[
        theme_resource(DARK_JSON, true),
        theme_resource(DARK_JSON, true),
    ]);
    assert_eq!(themes.len(), 1);
    assert_eq!(themes[0].source_path(), Some(DARK_JSON));
}

#[test]
fn skips_broken_theme_files() {
    let themes = load_themes(&[
        theme_resource("/nonexistent/theme.json", true),
        theme_resource(DARK_JSON, true),
    ]);
    let names: Vec<&str> = themes.iter().filter_map(|theme| theme.name()).collect();
    assert_eq!(names, vec!["dark"]);
}

#[test]
fn skips_names_reserved_for_automatic_settings() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("slash.json");
    let content = std::fs::read_to_string(DARK_JSON).expect("read dark theme");
    std::fs::write(
        &path,
        content.replace("\"name\": \"dark\"", "\"name\": \"custom/dark\""),
    )
    .expect("write theme");
    assert!(load_themes(&[theme_resource(path.to_str().expect("utf-8 path"), true)]).is_empty());
}

#[tokio::test]
async fn resolves_themes_from_the_global_settings_array() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut settings = crate::settings_manager::Settings::new();
    settings.insert("themes".to_string(), serde_json::json!([DARK_JSON]));
    let manager = SettingsManager::in_memory(
        &settings,
        SettingsManagerCreateOptions {
            project_trusted: Some(false),
        },
    );
    let themes = load_startup_themes(
        &manager,
        dir.path().to_str().expect("utf-8 cwd"),
        dir.path().to_str().expect("utf-8 agent dir"),
    )
    .await;
    let names: Vec<&str> = themes.iter().filter_map(|theme| theme.name()).collect();
    assert_eq!(names, vec!["dark"]);
}

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

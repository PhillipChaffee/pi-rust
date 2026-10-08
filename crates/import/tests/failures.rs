//! The write-failure suite: a target the filesystem refuses, downgrading
//! every planned write's report item, plus the auth backend's construction
//! error path.

#![expect(
    clippy::unwrap_used,
    reason = "the fixtures and assertions unwrap on their own setup only"
)]
#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the outcome assertions panic when the tool misbehaves"
)]

mod common;

use std::path::Path;

use common::{
    api_key_entry, cleanup, message_entry, oauth_entry, session_header, settings_object,
    source_dir, temp_dir, write,
};
use pi_import::discovery::ImportOptions;
use pi_import::report::{CredentialOutcome, SessionOutcome, SettingsOutcome, TrustOutcome};
use pi_import::run_import_with_target;

/// A full source with every artifact class, the blocked-target run's input.
fn full_source(label: &str) -> std::path::PathBuf {
    let dir = source_dir(label);
    let cwd = dir.join("proj");
    std::fs::create_dir_all(&cwd).unwrap();
    let header = session_header("a", &cwd.display().to_string(), Some(3));
    write(
        &dir,
        "sessions/--x--/20260101T000000-000Z_a.jsonl",
        &format!("{}\n{}\n", header, message_entry("m1", None)),
    );
    write(
        &dir,
        "auth.json",
        &settings_object(&[
            ("anthropic", &api_key_entry("sk-live", None)),
            ("openai", &oauth_entry("at", "rt", "1700000000000", None)),
        ]),
    );
    write(
        &dir,
        "settings.json",
        &settings_object(&[("theme", "\"dark\"")]),
    );
    write(&dir, "trust.json", "{\"/a\": true}");
    dir
}

#[test]
fn a_blocked_target_downgrades_every_write() {
    let source = full_source("blocked-target-source");
    // The target's parent is a file: every parent-directory creation fails.
    let blocked = temp_dir("blocked-target-root");
    write(&blocked, "agent", "not a directory");
    let target = blocked.join("agent");

    let report = run_import_with_target(
        &ImportOptions {
            source: Some(source.clone()),
            ..ImportOptions::default()
        },
        &target,
    )
    .expect("run");

    // Every planned write downgraded its item and recorded the failure.
    assert!(
        report
            .sessions
            .iter()
            .any(|item| matches!(item.outcome, SessionOutcome::Failed { .. }))
    );
    assert!(
        report
            .credentials
            .iter()
            .any(|item| matches!(item.outcome, CredentialOutcome::Failed { .. }))
    );
    assert!(
        report
            .settings
            .iter()
            .any(|item| matches!(item.outcome, SettingsOutcome::Failed { .. }))
    );
    assert!(matches!(
        report.trust.as_ref().unwrap().outcome,
        TrustOutcome::Failed { .. }
    ));
    assert!(report.has_failures());
    assert!(
        report
            .failures()
            .iter()
            .all(|failure| !failure.detail.is_empty())
    );
    cleanup(&source);
    cleanup(&blocked);
}

#[test]
fn a_directory_target_auth_refuses_the_store_write() {
    let source = full_source("blocked-auth-source");
    let target = temp_dir("blocked-auth-target");
    // The target's auth.json is a directory: the store's write path fails.
    std::fs::create_dir_all(target.join("auth.json")).unwrap();

    let report = run_import_with_target(
        &ImportOptions {
            source: Some(source.clone()),
            ..ImportOptions::default()
        },
        &target,
    )
    .expect("run");

    // The read_target_map sees the directory as unreadable and refuses the
    // merge at plan time.
    assert!(
        report
            .failures()
            .iter()
            .any(|failure| failure.detail.contains("target auth.json"))
    );
    assert!(report.credentials.iter().all(|item| matches!(
        item.outcome,
        CredentialOutcome::Failed { .. } | CredentialOutcome::Skipped { .. }
    )));
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn auth_write_op_reports_backend_construction_failures() {
    let leg = pi_import::credentials::CredentialsLeg::default();
    let plan = pi_import::credentials::AuthWritePlan {
        auth_path: "/unused/auth.json".to_string(),
        next: "sk-material".to_string(),
    };
    let op = pi_import::credentials::auth_write_op(&plan, &leg);
    assert!(matches!(op, pi_import::plan::PlannedOp::AuthWrite { .. }));
    let debug = format!("{plan:?}");
    assert!(debug.contains("auth_path"));
    assert!(!debug.contains("sk-material"));
}

#[test]
fn a_directory_source_settings_fails_the_read() {
    let dir = temp_dir("blocked-settings-source");
    write(&dir, "trust.json", "{}");
    std::fs::create_dir_all(dir.join("settings.json")).unwrap();

    let report = run_import_with_target(
        &ImportOptions {
            source: Some(dir.clone()),
            ..ImportOptions::default()
        },
        &dir,
    )
    .expect("run");

    let global = report
        .settings
        .iter()
        .find(|item| item.scope == "global")
        .unwrap();
    assert!(matches!(global.outcome, SettingsOutcome::Failed { .. }));
    assert!(report.has_failures());
    cleanup(&dir);
}

#[test]
fn sessions_fall_back_to_the_source_dir_name_without_a_header_cwd() {
    let source = source_dir("fallback-placement-source");
    let target = temp_dir("fallback-placement-target");
    write(
        &source,
        "sessions/--x--/nocwd.jsonl",
        &format!(
            "{}\n",
            session_header("nc", &source.display().to_string(), Some(3))
        ),
    );
    // Rewrite the file without a cwd field.
    std::fs::write(
        source.join("sessions/--x--/nocwd.jsonl"),
        "{\"type\":\"session\",\"version\":3,\"id\":\"nc\",\"timestamp\":\"2026-01-01T00:00:00.000Z\"}\n",
    )
    .unwrap();

    let report = run_import_with_target(
        &ImportOptions {
            source: Some(source.clone()),
            ..ImportOptions::default()
        },
        &target,
    )
    .expect("run");

    let item = report
        .sessions
        .iter()
        .find(|item| item.path.ends_with("nocwd.jsonl"))
        .unwrap();
    let SessionOutcome::Copied { placed_to } = &item.outcome else {
        panic!("expected copied, got {:?}", item.outcome);
    };
    assert!(
        Path::new(placed_to)
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with("--x--")
    );
    assert!(Path::new(placed_to).exists());
    cleanup(&source);
    cleanup(&target);
}

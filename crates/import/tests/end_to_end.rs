#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::unwrap_used,
    reason = "the fixtures and assertions unwrap on their own setup only"
)]

//! The end-to-end suite: a full TS-pi source through the run, in place and
//! cross-dir, with the dry-run snapshot and the JSON report.

mod common;

use std::path::Path;

use common::{
    api_key_entry, cleanup, message_entry, oauth_entry, session_header, settings_object,
    source_dir, temp_dir, write,
};
use pi_import::discovery::ImportOptions;
use pi_import::report::SessionOutcome;
use pi_import::run_import_with_target;

/// A source agent dir with every artifact class the tool migrates.
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
    write(&dir, "stray.jsonl", &format!("{header}\n"));
    write(
        &dir,
        "auth.json",
        &settings_object(&[
            ("anthropic", &api_key_entry("sk-live", None)),
            ("openai", &oauth_entry("at", "rt", "1700000000123.5", None)),
        ]),
    );
    write(
        &dir,
        "settings.json",
        &settings_object(&[
            ("theme", "\"dark\""),
            ("queueMode", "\"all\""),
            ("packages", "[\"some-npm-pkg\"]"),
            ("extensions", "[\"/abs/ext.ts\"]"),
        ]),
    );
    write(&dir, "trust.json", "{\"/a\": true}");
    write(&dir, "extensions/my-ext.ts", "export default {};\n");
    dir
}

#[test]
fn in_place_run_normalizes_everything() {
    let dir = full_source("e2e-in-place");

    let report = run_import_with_target(
        &ImportOptions {
            source: Some(dir.clone()),
            ..ImportOptions::default()
        },
        &dir,
    )
    .expect("run");

    assert!(!report.has_failures());
    // Sessions validated in place, stray moved.
    assert_eq!(report.sessions.len(), 2);
    let stray = report
        .sessions
        .iter()
        .find(|item| item.path == "stray.jsonl")
        .expect("stray");
    assert!(matches!(stray.outcome, SessionOutcome::Copied { .. }));
    assert!(!dir.join("stray.jsonl").exists());
    // Credentials: both providers validate, the oauth one normalized.
    assert_eq!(report.credentials.len(), 2);
    assert!(
        report
            .credentials
            .iter()
            .any(|item| item.outcome == pi_import::report::CredentialOutcome::Normalized)
    );
    // Settings migrated the queueMode pair, kept the packages/extensions
    // entries, and enumerated both as skipped.
    let global = report
        .settings
        .iter()
        .find(|item| item.scope == "global")
        .expect("global");
    assert!(
        global
            .merged
            .contains(&"queueMode -> steeringMode".to_string())
    );
    assert_eq!(report.skipped.len(), 3);
    assert!(
        report
            .skipped
            .iter()
            .any(|item| item.kind == "package" && item.source == "some-npm-pkg")
    );
    assert!(
        report
            .skipped
            .iter()
            .any(|item| item.kind == "extension" && item.source == "/abs/ext.ts")
    );
    assert!(
        report
            .skipped
            .iter()
            .any(|item| item.source.ends_with("extensions/my-ext.ts"))
    );
    // The written settings carry the migrated form.
    let written = std::fs::read_to_string(dir.join("settings.json")).unwrap();
    assert!(written.contains("\"steeringMode\""));
    assert!(written.contains("\"packages\""));
    assert!(written.contains("\"extensions\""));
    // Trust unchanged.
    assert!(matches!(
        report.trust.as_ref().expect("trust").outcome,
        pi_import::report::TrustOutcome::Unchanged { entries: 1 }
    ));
    let rendered = pi_import::cli::render_text(&report);
    assert!(!rendered.contains("sk-live"));
    cleanup(&dir);
}

#[test]
fn cross_dir_run_populates_a_fresh_target() {
    let source = full_source("e2e-cross");
    let target = temp_dir("e2e-cross-target");

    let report = run_import_with_target(
        &ImportOptions {
            source: Some(source.clone()),
            ..ImportOptions::default()
        },
        &target,
    )
    .expect("run");

    assert!(
        report.failures().is_empty(),
        "failures: {:?}",
        report.failures()
    );
    assert!(target.join("sessions").is_dir());
    assert!(target.join("auth.json").exists());
    assert!(target.join("settings.json").exists());
    assert!(target.join("trust.json").exists());
    // The stray lands in the tree at the target and stays at the source.
    assert!(source.join("stray.jsonl").exists());
    let stored = std::fs::read_to_string(target.join("auth.json")).unwrap();
    assert!(stored.contains("\"expires\": 1700000000123"));
    let settings = std::fs::read_to_string(target.join("settings.json")).unwrap();
    assert!(settings.contains("\"steeringMode\""));
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn dry_run_touches_nothing() {
    let source = full_source("e2e-dry");
    let target = temp_dir("e2e-dry-target");
    let snapshot = |dir: &Path| -> Vec<(std::path::PathBuf, Option<String>)> {
        let mut files = Vec::new();
        for entry in walk(dir) {
            let content = std::fs::read_to_string(&entry).ok();
            files.push((entry, content));
        }
        files.sort();
        files
    };
    let before_source = snapshot(&source);

    let report = run_import_with_target(
        &ImportOptions {
            source: Some(source.clone()),
            dry_run: true,
            ..ImportOptions::default()
        },
        &target,
    )
    .expect("run");

    assert_eq!(snapshot(&source), before_source);
    assert!(report.dry_run);
    // The report still names the writes it would make.
    assert!(
        report
            .sessions
            .iter()
            .any(|item| matches!(item.outcome, SessionOutcome::Copied { .. }))
    );
    assert!(
        report
            .settings
            .iter()
            .any(|item| matches!(item.outcome, pi_import::report::SettingsOutcome::Written))
    );
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn json_report_parses_with_the_report_shape() {
    let source = full_source("e2e-json");
    let target = temp_dir("e2e-json-target");
    let report = run_import_with_target(
        &ImportOptions {
            source: Some(source.clone()),
            ..ImportOptions::default()
        },
        &target,
    )
    .expect("run");
    let value = report.to_json();
    let object = value.as_object().expect("object");
    for key in [
        "source",
        "target",
        "dryRun",
        "sessions",
        "credentials",
        "settings",
        "skipped",
    ] {
        assert!(object.contains_key(key), "missing {key}");
    }
    let sessions = object["sessions"].as_array().expect("sessions array");
    assert!(!sessions.is_empty());
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn a_second_cross_dir_run_is_idempotent() {
    let source = full_source("e2e-idempotent");
    let target = temp_dir("e2e-idempotent-target");

    let first = run_import_with_target(
        &ImportOptions {
            source: Some(source.clone()),
            ..ImportOptions::default()
        },
        &target,
    )
    .expect("first run");
    assert!(
        first.failures().is_empty(),
        "first failures: {:?}",
        first.failures()
    );
    let second = run_import_with_target(
        &ImportOptions {
            source: Some(source.clone()),
            ..ImportOptions::default()
        },
        &target,
    )
    .expect("second run");

    assert!(!second.has_failures());
    assert!(second.sessions.iter().all(|item| matches!(
        item.outcome,
        SessionOutcome::Present | SessionOutcome::Copied { .. } | SessionOutcome::Skipped { .. }
    )));
    let stored = std::fs::read_to_string(target.join("auth.json")).unwrap();
    assert!(stored.contains("sk-live"));
    cleanup(&source);
    cleanup(&target);
}

/// Every file under the dir, the snapshot's walk.
fn walk(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return files;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files.extend(walk(&path));
        } else {
            files.push(path);
        }
    }
    files
}

#[test]
fn env_run_dry_run_never_touches_the_real_agent_dir() {
    // The env-derived default path runs dry: no writes anywhere, the
    // report rendering its refusal or its empty run.
    let source = full_source("e2e-env-dry");
    let before = walk(&source);
    let report = pi_import::run_import(&ImportOptions {
        source: Some(source.clone()),
        dry_run: true,
        ..ImportOptions::default()
    })
    .expect("dry run never refuses a full source");
    assert!(report.dry_run);
    assert_eq!(walk(&source), before);
    cleanup(&source);
}

#[test]
fn package_objects_render_their_json_label() {
    let source = source_dir("e2e-package-object");
    write(
        &source,
        "settings.json",
        &settings_object(&[("packages", "[{\"source\":\"npm:@x/y\",\"filter\":\"*\"}]")]),
    );
    let target = temp_dir("e2e-package-object-target");
    let report = run_import_with_target(
        &ImportOptions {
            source: Some(source.clone()),
            ..ImportOptions::default()
        },
        &target,
    )
    .expect("run");
    let package = report
        .skipped
        .iter()
        .find(|item| item.kind == "package")
        .expect("package item");
    assert_eq!(package.source, "{\"source\":\"npm:@x/y\",\"filter\":\"*\"}");
    cleanup(&source);
    cleanup(&target);
}

//! The CLI's suite: the argv grammar, the renders, and the report's JSON.

#![expect(
    clippy::panic,
    reason = "the outcome assertions panic when the tool misbehaves"
)]

mod common;

use common::{
    api_key_entry, cleanup, oauth_entry, session_header, settings_object, source_dir, temp_dir,
    write,
};
use pi_import::report::{
    CredentialItem, CredentialOutcome, ImportReport, SessionItem, SessionOutcome, SettingsItem,
    SettingsOutcome, TrustItem, TrustOutcome,
};

#[test]
fn help_exits_zero() {
    assert_eq!(pi_import::cli::run_cli(vec!["--help".to_string()]), 0);
    assert_eq!(pi_import::cli::run_cli(vec!["-h".to_string()]), 0);
}

#[test]
fn usage_errors_exit_two() {
    let target = temp_dir("cli-usage-target");
    assert_eq!(
        pi_import::cli::run_cli_with_target(vec!["--unknown".to_string()], &target),
        2
    );
    assert_eq!(
        pi_import::cli::run_cli_with_target(vec!["--source".to_string()], &target),
        2
    );
    assert_eq!(
        pi_import::cli::run_cli_with_target(
            vec!["--source".to_string(), "--json".to_string()],
            &target
        ),
        2
    );
    // The refusal exits 2 with the explanation on stderr.
    let missing = temp_dir("cli-refused").join("nope");
    assert_eq!(
        pi_import::cli::run_cli_with_target(
            vec!["--source".to_string(), missing.display().to_string()],
            &target
        ),
        2
    );
    cleanup(&target);
}

#[test]
fn clean_run_exits_zero_and_failure_run_exits_one() {
    let source = source_dir("cli-clean");
    let target = temp_dir("cli-clean-target");
    write(&source, "trust.json", "{\"/a\": true}");
    let exit = pi_import::cli::run_cli_with_target(
        vec!["--source".to_string(), source.display().to_string()],
        &target,
    );
    assert_eq!(exit, 0);
    assert!(target.join("trust.json").exists());
    cleanup(&source);
    cleanup(&target);

    // With a failing artifact the exit is 1.
    let source = source_dir("cli-failure");
    write(&source, "trust.json", "{\"/a\": \"yes\"}");
    let target = temp_dir("cli-failure-target");
    let exit = pi_import::cli::run_cli_with_target(
        vec!["--source".to_string(), source.display().to_string()],
        &target,
    );
    assert_eq!(exit, 1);
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn json_render_prints_the_report() {
    let source = source_dir("cli-json");
    write(
        &source,
        "auth.json",
        &settings_object(&[("anthropic", &api_key_entry("sk-live", None))]),
    );
    let target = temp_dir("cli-json-target");
    let exit = pi_import::cli::run_cli_with_target(
        vec![
            "--source".to_string(),
            source.display().to_string(),
            "--json".to_string(),
        ],
        &target,
    );
    assert_eq!(exit, 0);
    cleanup(&target);
    cleanup(&source);
}

#[test]
fn text_render_shapes_the_sections() {
    let mut report = ImportReport::default();
    report.source = "/src".to_string();
    report.target = "/dst".to_string();
    report.project = Some("/proj".to_string());
    report.dry_run = true;
    report.sessions.push(SessionItem {
        path: "sessions/--x--/a.jsonl".to_string(),
        version: Some(3),
        outcome: SessionOutcome::Copied {
            placed_to: "/dst/sessions/--x--/a.jsonl".to_string(),
        },
    });
    report.credentials.push(CredentialItem {
        provider: "anthropic".to_string(),
        kind: Some("api_key".to_string()),
        outcome: CredentialOutcome::Imported,
    });
    report.credentials.push(CredentialItem {
        provider: "openai".to_string(),
        kind: Some("oauth".to_string()),
        outcome: CredentialOutcome::Normalized,
    });
    report.settings.push(SettingsItem {
        scope: "global".to_string(),
        path: "/dst/settings.json".to_string(),
        outcome: SettingsOutcome::Written,
        merged: vec!["queueMode -> steeringMode".to_string()],
        dropped: vec!["apiKeys (folded into auth.json)".to_string()],
        carried_keys: 4,
    });
    report.trust = Some(TrustItem {
        path: "/dst/trust.json".to_string(),
        outcome: TrustOutcome::Written { entries: 2 },
    });
    report.skipped.push(pi_import::report::SkippedItem {
        source: "extensions/my-ext.ts".to_string(),
        kind: "extension".to_string(),
        reason: "TS extensions do not execute on the Rust pi".to_string(),
    });
    report.push_failure("sessions/--x--/a.jsonl", "write failed: boom");

    let text = pi_import::cli::render_text(&report);
    assert!(text.contains("pi-import: TS-pi -> Rust-pi migration"));
    assert!(text.contains("  source: /src\n"));
    assert!(text.contains("  project: /proj\n"));
    assert!(text.contains("  mode: dry-run (no writes)\n"));
    assert!(text.contains("  copied   sessions/--x--/a.jsonl -> /dst/sessions/--x--/a.jsonl (v3)"));
    assert!(text.contains("  imported   anthropic (api_key)"));
    assert!(text.contains("  normalized openai (oauth, expires floored to an integer)"));
    assert!(text.contains("  summary: 1 imported (1 normalized), 0 kept, 0 failed"));
    assert!(text.contains("  global /dst/settings.json: written (merged: queueMode -> steeringMode) (dropped: apiKeys (folded into auth.json)) (4 keys)"));
    assert!(text.contains("  /dst/trust.json: copied (2 entries)"));
    assert!(text.contains("  extension extensions/my-ext.ts"));
    assert!(text.contains("failures\n  sessions/--x--/a.jsonl: write failed: boom"));

    let json = report.to_json();
    let Some(object) = json.as_object() else {
        panic!("the report serializes as a JSON object");
    };
    assert!(object.contains_key("sessions"));
    assert!(object.contains_key("credentials"));
    assert!(object.contains_key("settings"));
    assert!(object.contains_key("trust"));
    assert!(object.contains_key("skipped"));
    assert!(object.contains_key("failures"));
    assert_eq!(object["dryRun"], serde_json::Value::Bool(true));
}

#[test]
fn oauth_entries_render_their_kind() {
    let entry = oauth_entry("at", "rt", "1700000000000", None);
    assert!(entry.contains("\"oauth\""));
}

#[test]
fn session_header_fixture_carries_the_version() {
    let header = session_header("a", "/proj", Some(3));
    assert!(header.contains("\"version\":3"));
    let v1 = session_header("a", "/proj", None);
    assert!(!v1.contains("version"));
}

#[test]
fn project_flag_parses_through_the_cli() {
    let source = source_dir("cli-project");
    let target = temp_dir("cli-project-target");
    let project = temp_dir("cli-project-dir");
    write(
        &project,
        ".pi/settings.json",
        &settings_object(&[("theme", "\"dark\"")]),
    );
    let exit = pi_import::cli::run_cli_with_target(
        vec![
            "--source".to_string(),
            source.display().to_string(),
            "--project".to_string(),
            project.display().to_string(),
        ],
        &target,
    );
    assert_eq!(exit, 0);
    // The project file normalized in place (unchanged, no legacy keys).
    assert!(project.join(".pi/settings.json").exists());
    cleanup(&source);
    cleanup(&target);
    cleanup(&project);
}

#[test]
fn help_through_the_default_entry_never_touches_the_filesystem() {
    // The env-derived target is only read; --help returns before the run.
    assert_eq!(pi_import::cli::run_cli(vec!["--help".to_string()]), 0);
}

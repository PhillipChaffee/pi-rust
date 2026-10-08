//! The report model's suite: the failure downgrades, the JSON render, and
//! the text render's every outcome arm.

#![expect(
    clippy::unwrap_used,
    reason = "the fixtures and assertions unwrap on their own setup only"
)]

use pi_import::report::{
    CredentialItem, CredentialOutcome, Failure, ImportReport, SessionItem, SessionOutcome,
    SettingsItem, SettingsOutcome, TrustItem, TrustOutcome,
};

#[expect(
    clippy::too_many_lines,
    reason = "one fixture builder naming every outcome arm keeps the render assertions readable"
)]
fn sample() -> ImportReport {
    let mut report = ImportReport::default();
    report.source = "/src".to_string();
    report.target = "/dst".to_string();
    report.sessions.push(SessionItem {
        path: "sessions/--x--/a.jsonl".to_string(),
        version: Some(3),
        outcome: SessionOutcome::Copied {
            placed_to: "/dst/sessions/--x--/a.jsonl".to_string(),
        },
    });
    report.sessions.push(SessionItem {
        path: "sessions/--x--/b.jsonl".to_string(),
        version: None,
        outcome: SessionOutcome::Present,
    });
    report.sessions.push(SessionItem {
        path: "sessions/--x--/c.jsonl".to_string(),
        version: None,
        outcome: SessionOutcome::Skipped {
            reason: "no session header".to_string(),
        },
    });
    report.sessions.push(SessionItem {
        path: "sessions/--x--/d.jsonl".to_string(),
        version: None,
        outcome: SessionOutcome::Failed {
            reason: "io".to_string(),
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
    report.credentials.push(CredentialItem {
        provider: "groq".to_string(),
        kind: Some("api_key".to_string()),
        outcome: CredentialOutcome::Kept,
    });
    report.credentials.push(CredentialItem {
        provider: "skipme".to_string(),
        kind: None,
        outcome: CredentialOutcome::Skipped {
            reason: "null entry".to_string(),
        },
    });
    report.credentials.push(CredentialItem {
        provider: "bad".to_string(),
        kind: None,
        outcome: CredentialOutcome::Failed {
            reason: "invalid credential: does not match the api_key or oauth shape".to_string(),
        },
    });
    report.settings.push(SettingsItem {
        scope: "global".to_string(),
        path: "/dst/settings.json".to_string(),
        outcome: SettingsOutcome::Written,
        merged: vec!["queueMode -> steeringMode".to_string()],
        dropped: Vec::new(),
        carried_keys: 3,
    });
    report.settings.push(SettingsItem {
        scope: "project".to_string(),
        path: "/proj/.pi/settings.json".to_string(),
        outcome: SettingsOutcome::Unchanged,
        merged: Vec::new(),
        dropped: Vec::new(),
        carried_keys: 0,
    });
    report.settings.push(SettingsItem {
        scope: "project".to_string(),
        path: "/proj/.pi/settings.json".to_string(),
        outcome: SettingsOutcome::Skipped {
            reason: "no settings file".to_string(),
        },
        merged: Vec::new(),
        dropped: Vec::new(),
        carried_keys: 0,
    });
    report.settings.push(SettingsItem {
        scope: "project".to_string(),
        path: "/proj/.pi/settings.json".to_string(),
        outcome: SettingsOutcome::Failed {
            reason: "unreadable".to_string(),
        },
        merged: Vec::new(),
        dropped: Vec::new(),
        carried_keys: 0,
    });
    report.trust = Some(TrustItem {
        path: "/dst/trust.json".to_string(),
        outcome: TrustOutcome::Unchanged { entries: 2 },
    });
    report.skipped.push(pi_import::report::SkippedItem {
        source: "extensions/my-ext.ts".to_string(),
        kind: "extension".to_string(),
        reason: "TS extensions do not execute on the Rust pi".to_string(),
    });
    report.credential_sources = vec!["auth.json".to_string()];
    report.push_failure("sessions/--x--/d.jsonl", "io");
    report
}

#[test]
fn downgrades_fail_the_item_and_record_the_failure() {
    let mut report = ImportReport::default();
    report.sessions.push(SessionItem {
        path: "a.jsonl".to_string(),
        version: None,
        outcome: SessionOutcome::Present,
    });
    report.credentials.push(CredentialItem {
        provider: "anthropic".to_string(),
        kind: None,
        outcome: CredentialOutcome::Imported,
    });
    report.settings.push(SettingsItem {
        scope: "global".to_string(),
        path: "/dst/settings.json".to_string(),
        outcome: SettingsOutcome::Written,
        merged: Vec::new(),
        dropped: Vec::new(),
        carried_keys: 0,
    });
    report.trust = Some(TrustItem {
        path: "/dst/trust.json".to_string(),
        outcome: TrustOutcome::Written { entries: 1 },
    });

    report.fail_session(0, "write failed");
    assert_eq!(
        report.sessions[0].outcome,
        SessionOutcome::Failed {
            reason: "write failed".to_string()
        }
    );
    report.fail_credential(0, "lock failed");
    assert!(matches!(
        report.credentials[0].outcome,
        CredentialOutcome::Failed { .. }
    ));
    report.fail_settings(0, "mkdir failed");
    assert!(matches!(
        report.settings[0].outcome,
        SettingsOutcome::Failed { .. }
    ));
    report.fail_trust("copy failed");
    assert!(matches!(
        report.trust.as_ref().unwrap().outcome,
        TrustOutcome::Failed { .. }
    ));
    assert_eq!(report.failures().len(), 4);
    assert_eq!(
        report.failures()[0],
        Failure {
            artifact: "a.jsonl".to_string(),
            detail: "write failed".to_string(),
        }
    );
    assert!(report.has_failures());

    // Out-of-range indices are ignored.
    report.fail_session(99, "nope");
    report.fail_credential(99, "nope");
    report.fail_settings(99, "nope");
    assert_eq!(report.failures().len(), 4);
}

#[test]
fn every_outcome_arm_renders() {
    let report = sample();
    let text = pi_import::cli::render_text(&report);
    assert!(text.contains("  copied   sessions/--x--/a.jsonl -> /dst/sessions/--x--/a.jsonl (v3)"));
    assert!(text.contains("  present  sessions/--x--/b.jsonl"));
    assert!(text.contains("  skipped  sessions/--x--/c.jsonl (no session header)"));
    assert!(text.contains("  failed   sessions/--x--/d.jsonl (io)"));
    assert!(text.contains("  summary: 1 copied, 1 present, 1 skipped, 1 failed"));
    assert!(text.contains("credentials (from auth.json)"));
    assert!(text.contains("  imported   anthropic (api_key)"));
    assert!(text.contains("  normalized openai (oauth, expires floored to an integer)"));
    assert!(text.contains("  kept       groq"));
    assert!(text.contains("  skipped    skipme (null entry)"));
    assert!(text.contains(
        "  failed     bad (invalid credential: does not match the api_key or oauth shape)"
    ));
    assert!(text.contains("  summary: 1 imported (1 normalized), 1 kept, 1 failed"));
    assert!(text.contains(
        "  global /dst/settings.json: written (merged: queueMode -> steeringMode) (3 keys)"
    ));
    assert!(text.contains("  project /proj/.pi/settings.json: unchanged (0 keys)"));
    assert!(text.contains("  project /proj/.pi/settings.json: skipped (no settings file)"));
    assert!(text.contains("  project /proj/.pi/settings.json: failed (unreadable)"));
    assert!(text.contains("  /dst/trust.json: unchanged (2 entries)"));
    assert!(text.contains("  extension extensions/my-ext.ts"));
    assert!(text.contains("failures\n  sessions/--x--/d.jsonl: io"));
}

#[test]
fn every_outcome_arm_serializes() {
    let report = sample();
    let value = report.to_json();
    let text = value.to_string();
    assert!(text.contains("\"placedTo\""));
    assert!(text.contains("\"carriedKeys\""));
    assert!(text.contains("\"credentialSources\""));
    assert!(text.contains("\"dryRun\":false"));
}

#[test]
fn trust_and_session_absence_renders_placeholders() {
    let mut report = ImportReport::default();
    report.trust = Some(TrustItem {
        path: "/dst/trust.json".to_string(),
        outcome: TrustOutcome::Skipped {
            reason: "target already has trust.json".to_string(),
        },
    });
    let text = pi_import::cli::render_text(&report);
    assert!(text.contains("sessions: none"));
    assert!(text.contains("credentials: none"));
    assert!(text.contains("  /dst/trust.json: skipped (target already has trust.json)"));

    let mut report = ImportReport::default();
    report.trust = Some(TrustItem {
        path: "/dst/trust.json".to_string(),
        outcome: TrustOutcome::Written { entries: 5 },
    });
    let text = pi_import::cli::render_text(&report);
    assert!(text.contains("  /dst/trust.json: copied (5 entries)"));
}
#[test]
fn failing_trust_without_an_item_is_ignored() {
    let mut report = ImportReport::default();
    report.fail_trust("no store to fail");
    assert!(report.failures().is_empty());
    assert!(report.trust.is_none());
}

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::unwrap_used,
    reason = "the fixtures and assertions unwrap on their own setup only"
)]
#![expect(
    clippy::panic,
    reason = "the missing-provider guards panic when the tool misbehaves"
)]

//! The credentials leg's suite: typed validation, the `expires`
//! normalization, the legacy fold, the merge write, and the no-key-material
//! print rule.

mod common;

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use common::{api_key_entry, cleanup, oauth_entry, settings_object, source_dir, temp_dir, write};
use pi_import::discovery::ImportOptions;
use pi_import::report::{CredentialOutcome, ImportReport};
use pi_import::{credentials, run_import_with_target};

fn options(source: &Path) -> ImportOptions {
    ImportOptions {
        source: Some(source.to_path_buf()),
        ..ImportOptions::default()
    }
}

fn run(source: &Path, target: &Path) -> ImportReport {
    run_import_with_target(&options(source), target).expect("run")
}

#[test]
fn cross_dir_imports_valid_entries_and_preserves_extras() {
    let source = source_dir("cred-cross");
    let target = temp_dir("cred-cross-target");
    write(
        &source,
        "auth.json",
        &settings_object(&[
            (
                "anthropic",
                &api_key_entry("sk-source-anthropic", Some("{\"accountId\":\"acc\"}")),
            ),
            (
                "openai",
                &oauth_entry("at", "rt", "1700000000000", Some("\"plan\":\"pro\"")),
            ),
            ("broken", "{\"type\":\"api_key\",\"key\":42}"),
            ("empty", "null"),
        ]),
    );

    let report = run(&source, &target);

    assert_eq!(report.credential_sources, vec!["auth.json"]);
    let outcomes: Vec<(&str, &CredentialOutcome)> = report
        .credentials
        .iter()
        .map(|item| (item.provider.as_str(), &item.outcome))
        .collect();
    assert!(outcomes.contains(&("anthropic", &CredentialOutcome::Imported)));
    assert!(outcomes.contains(&("openai", &CredentialOutcome::Imported)));
    assert!(matches!(
        outcomes.iter().find(|(provider, _)| *provider == "broken"),
        Some((_, CredentialOutcome::Failed { .. }))
    ));
    assert!(matches!(
        outcomes.iter().find(|(provider, _)| *provider == "empty"),
        Some((_, CredentialOutcome::Skipped { .. }))
    ));

    let target_path = target.join("auth.json");
    assert!(target_path.exists());
    let stored = std::fs::read_to_string(&target_path).unwrap();
    assert!(stored.contains("\"anthropic\""));
    assert!(stored.contains("sk-source-anthropic"));
    // Unknown api-key fields ride verbatim.
    assert!(stored.contains("\"accountId\": \"acc\""));
    assert!(stored.contains("\"plan\": \"pro\""));
    // The failed and null entries do not land.
    assert!(!stored.contains("\"broken\""));
    assert!(!stored.contains("\"empty\""));
    // The store's file mode is owner-only.
    let mode = std::fs::metadata(&target_path)
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600, "mode: {mode:o}");
    assert!(report.has_failures());
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn fractional_expires_floors_to_the_integer() {
    let source = source_dir("cred-fractional");
    let target = temp_dir("cred-fractional-target");
    write(
        &source,
        "auth.json",
        &settings_object(&[
            (
                "fractional",
                &oauth_entry("at", "rt", "1700000000123.5", None),
            ),
            (
                "integral_float",
                &oauth_entry("at", "rt", "1700000000123.0", None),
            ),
            ("huge", &oauth_entry("at", "rt", "1e30", None)),
        ]),
    );

    let report = run(&source, &target);

    let find = |provider: &str| -> CredentialOutcome {
        report
            .credentials
            .iter()
            .find(|item| item.provider == provider)
            .map_or_else(|| panic!("missing {provider}"), |item| item.outcome.clone())
    };
    assert_eq!(find("fractional"), CredentialOutcome::Normalized);
    assert_eq!(find("integral_float"), CredentialOutcome::Normalized);
    assert!(matches!(find("huge"), CredentialOutcome::Failed { .. }));
    let stored = std::fs::read_to_string(target.join("auth.json")).unwrap();
    assert!(stored.contains("\"expires\": 1700000000123"));
    assert!(!stored.contains("1700000000123.5"));
    assert!(!stored.contains("\"huge\""));
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn legacy_fold_runs_only_without_auth_json() {
    let source = source_dir("cred-fold");
    let target = temp_dir("cred-fold-target");
    write(
        &source,
        "oauth.json",
        &settings_object(&[("anthropic", &oauth_entry("at", "rt", "1700000000000", None))]),
    );
    write(
        &source,
        "settings.json",
        &settings_object(&[
            ("theme", "\"dark\""),
            (
                "apiKeys",
                &settings_object(&[("groq", "\"gsk-source\""), ("skip", "42")]),
            ),
        ]),
    );

    let report = run(&source, &target);

    assert_eq!(
        report.credential_sources,
        vec!["oauth.json", "settings.json apiKeys"]
    );
    let stored = std::fs::read_to_string(target.join("auth.json")).unwrap();
    assert!(stored.contains("\"anthropic\""));
    assert!(stored.contains("\"groq\""));
    assert!(stored.contains("gsk-source"));
    assert!(!stored.contains("\"skip\""));
    // The source files stay untouched: no rename, no settings rewrite.
    assert!(source.join("oauth.json").exists());
    let source_settings = std::fs::read_to_string(source.join("settings.json")).unwrap();
    assert!(source_settings.contains("apiKeys"));
    // The settings write drops the folded key.
    let written = std::fs::read_to_string(target.join("settings.json")).unwrap();
    assert!(written.contains("\"theme\""));
    assert!(!written.contains("apiKeys"));
    let settings_item = report
        .settings
        .iter()
        .find(|item| item.scope == "global")
        .expect("global item");
    assert!(
        settings_item
            .dropped
            .iter()
            .any(|note| note.starts_with("apiKeys"))
    );
    assert!(!report.has_failures());
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn legacy_fold_oauth_wins_and_requires_string_keys() {
    let source = source_dir("cred-fold-precedence");
    let target = temp_dir("cred-fold-precedence-target");
    write(
        &source,
        "oauth.json",
        &settings_object(&[("dup", &oauth_entry("at", "rt", "1700000000000", None))]),
    );
    write(
        &source,
        "settings.json",
        &settings_object(&[(
            "apiKeys",
            &settings_object(&[("dup", "\"gsk-loser\""), ("only", "\"gsk-only\"")]),
        )]),
    );

    let report = run(&source, &target);

    let stored = std::fs::read_to_string(target.join("auth.json")).unwrap();
    assert_eq!(stored.matches("\"dup\"").count(), 1);
    assert!(stored.contains("\"oauth\""));
    assert!(!stored.contains("gsk-loser"));
    assert!(stored.contains("gsk-only"));
    assert!(!report.has_failures());
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn legacy_sources_are_ignored_when_auth_json_exists() {
    let source = source_dir("cred-fold-skip");
    let target = temp_dir("cred-fold-skip-target");
    write(
        &source,
        "auth.json",
        &settings_object(&[("anthropic", &api_key_entry("sk-live", None))]),
    );
    write(&source, "oauth.json", "{}");

    let report = run(&source, &target);

    assert_eq!(report.credential_sources, vec!["auth.json"]);
    assert_eq!(report.credentials.len(), 1);
    assert_eq!(report.credentials[0].provider, "anthropic");
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn cross_dir_merge_keeps_existing_providers() {
    let source = source_dir("cred-merge");
    let target = temp_dir("cred-merge-target");
    write(
        &source,
        "auth.json",
        &settings_object(&[
            ("x", &api_key_entry("sk-new", None)),
            ("y", &oauth_entry("at", "rt", "1700000000000", None)),
        ]),
    );
    write(
        &target,
        "auth.json",
        &settings_object(&[("x", &api_key_entry("sk-kept", None))]),
    );

    let report = run(&source, &target);

    let find = |provider: &str| -> CredentialOutcome {
        report
            .credentials
            .iter()
            .find(|item| item.provider == provider)
            .map_or_else(|| panic!("missing {provider}"), |item| item.outcome.clone())
    };
    assert_eq!(find("x"), CredentialOutcome::Kept);
    assert_eq!(find("y"), CredentialOutcome::Imported);
    let stored = std::fs::read_to_string(target.join("auth.json")).unwrap();
    assert!(stored.contains("sk-kept"));
    assert!(!stored.contains("sk-new"));
    assert!(stored.contains("\"oauth\""));
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn in_place_keeps_the_store_and_normalizes_in_place() {
    let dir = source_dir("cred-in-place");
    write(
        &dir,
        "auth.json",
        &settings_object(&[
            ("anthropic", &api_key_entry("sk-live", None)),
            ("openai", &oauth_entry("at", "rt", "1700000000123.5", None)),
        ]),
    );
    let before = std::fs::read_to_string(dir.join("auth.json")).unwrap();

    let report = run(&dir, &dir);

    let find = |provider: &str| -> CredentialOutcome {
        report
            .credentials
            .iter()
            .find(|item| item.provider == provider)
            .map_or_else(|| panic!("missing {provider}"), |item| item.outcome.clone())
    };
    assert_eq!(find("anthropic"), CredentialOutcome::Kept);
    assert_eq!(find("openai"), CredentialOutcome::Normalized);
    let after = std::fs::read_to_string(dir.join("auth.json")).unwrap();
    assert_ne!(before, after);
    assert!(after.contains("sk-live"));
    assert!(after.contains("\"expires\": 1700000000123"));
    cleanup(&dir);
}

#[test]
fn in_place_without_store_writes_the_fold() {
    let dir = source_dir("cred-fold-in-place");
    write(
        &dir,
        "oauth.json",
        &settings_object(&[("anthropic", &oauth_entry("at", "rt", "1700000000000", None))]),
    );

    let report = run(&dir, &dir);

    assert!(dir.join("auth.json").exists());
    let stored = std::fs::read_to_string(dir.join("auth.json")).unwrap();
    assert!(stored.contains("\"anthropic\""));
    assert!(dir.join("oauth.json").exists());
    assert!(!report.has_failures());
    cleanup(&dir);
}

#[test]
fn unreadable_sources_fail_the_leg() {
    let source = source_dir("cred-broken");
    let target = temp_dir("cred-broken-target");
    write(&source, "auth.json", "{not json");

    let report = run(&source, &target);

    assert!(report.has_failures());
    assert!(
        report
            .failures()
            .iter()
            .any(|failure| failure.artifact.ends_with("auth.json"))
    );
    assert!(!target.join("auth.json").exists());
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn non_object_target_refuses_the_merge() {
    let source = source_dir("cred-target-broken");
    let target = temp_dir("cred-target-broken-target");
    write(&source, "auth.json", &api_key_entry("sk-live", None));
    write(&target, "auth.json", "[1,2]");

    let report = run(&source, &target);

    assert!(report.has_failures());
    assert!(
        report
            .failures()
            .iter()
            .any(|failure| failure.detail.contains("target auth.json"))
    );
    let stored = std::fs::read_to_string(target.join("auth.json")).unwrap();
    assert_eq!(stored, "[1,2]");
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn reports_never_print_key_material() {
    let source = source_dir("cred-secret");
    let target = temp_dir("cred-secret-target");
    write(
        &source,
        "auth.json",
        &api_key_entry("sk-super-secret-value", None),
    );

    let report = run(&source, &target);
    let rendered = pi_import::cli::render_text(&report);
    let json = report.to_json().to_string();
    assert!(!rendered.contains("sk-super-secret-value"));
    assert!(!json.contains("sk-super-secret-value"));
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn fold_reads_bom_stripped_sources() {
    let source = source_dir("cred-bom");
    let target = temp_dir("cred-bom-target");
    write(
        &source,
        "auth.json",
        &format!(
            "\u{feff}{}",
            settings_object(&[("anthropic", &api_key_entry("sk-live", None))])
        ),
    );

    let report = run(&source, &target);

    assert_eq!(report.credentials[0].outcome, CredentialOutcome::Imported);
    let stored = std::fs::read_to_string(target.join("auth.json")).unwrap();
    assert!(stored.contains("sk-live"));
    assert!(!report.has_failures());
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn direct_leg_run_reports_without_a_run_context() {
    let dir = source_dir("cred-direct");
    write(&dir, "auth.json", &api_key_entry("sk-live", None));
    let discovery =
        pi_import::discovery::discover_with_target(&options(&dir), &dir).expect("recognized");
    let mut report = ImportReport::default();
    let leg = credentials::run_credentials(&discovery, &mut report);
    assert!(leg.write.is_none());
    assert!(leg.landed_items.is_empty());
    assert!(!leg.settings_api_keys_consumed);
    cleanup(&dir);
}

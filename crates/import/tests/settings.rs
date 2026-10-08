#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::unwrap_used,
    reason = "the fixtures and assertions unwrap on their own setup only"
)]

//! The settings leg's suite: the legacy-key migrations, the merged and
//! dropped notes, the write decisions, and the project scope.

mod common;

use std::path::Path;

use common::{cleanup, settings_object, source_dir, temp_dir, write};
use pi_import::discovery::ImportOptions;
use pi_import::report::SettingsOutcome;
use pi_import::{run_import_with_target, settings};

fn options(source: &Path) -> ImportOptions {
    ImportOptions {
        source: Some(source.to_path_buf()),
        ..ImportOptions::default()
    }
}

fn global_item(report: &pi_import::report::ImportReport) -> &pi_import::report::SettingsItem {
    report
        .settings
        .iter()
        .find(|item| item.scope == "global")
        .expect("global settings item")
}

#[test]
fn in_place_unchanged_leaves_the_file_alone() {
    let dir = source_dir("settings-unchanged");
    write(
        &dir,
        "settings.json",
        &settings_object(&[("theme", "\"dark\"")]),
    );
    let before = std::fs::read_to_string(dir.join("settings.json")).unwrap();

    let report = run_import_with_target(&options(&dir), &dir).expect("run");

    let item = global_item(&report);
    assert_eq!(item.outcome, SettingsOutcome::Unchanged);
    assert_eq!(item.merged, Vec::<String>::new());
    assert_eq!(item.dropped, Vec::<String>::new());
    assert_eq!(item.carried_keys, 1);
    assert_eq!(
        std::fs::read_to_string(dir.join("settings.json")).unwrap(),
        before
    );
    assert!(!report.has_failures());
    cleanup(&dir);
}

#[test]
fn in_place_migrates_the_legacy_keys_and_reports_the_pairs() {
    let dir = source_dir("settings-migrate");
    write(
        &dir,
        "settings.json",
        &settings_object(&[
            ("theme", "\"dark\""),
            ("queueMode", "\"all\""),
            ("websockets", "true"),
            ("retry", &settings_object(&[("maxDelayMs", "500")])),
            (
                "skills",
                &settings_object(&[
                    ("enableSkillCommands", "true"),
                    ("customDirectories", "[\"/x\"]"),
                ]),
            ),
        ]),
    );

    let report = run_import_with_target(&options(&dir), &dir).expect("run");

    let item = global_item(&report);
    assert_eq!(item.outcome, SettingsOutcome::Written);
    assert!(
        item.merged
            .contains(&"queueMode -> steeringMode".to_string())
    );
    assert!(item.merged.contains(&"websockets -> transport".to_string()));
    assert!(
        item.merged
            .contains(&"retry.maxDelayMs -> retry.provider.maxRetryDelayMs".to_string())
    );
    assert!(
        item.merged
            .contains(&"skills.customDirectories -> skills".to_string())
    );
    assert!(item.dropped.is_empty());
    let written = std::fs::read_to_string(dir.join("settings.json")).unwrap();
    assert!(written.contains("\"steeringMode\": \"all\""));
    assert!(written.contains("\"transport\": \"websocket\""));
    assert!(written.contains("\"maxRetryDelayMs\""));
    assert!(!written.contains("queueMode"));
    assert!(!written.contains("websockets"));
    assert!(!written.contains("maxDelayMs"));
    // The skills array replaced the object.
    assert!(written.contains("\"skills\": [\n    \"/x\"\n  ]"));
    cleanup(&dir);
}

#[test]
fn dropped_notes_name_the_unreplaced_keys() {
    let dir = source_dir("settings-dropped");
    write(
        &dir,
        "settings.json",
        &settings_object(&[
            (
                "retry",
                &settings_object(&[
                    ("maxDelayMs", "500"),
                    ("provider", &settings_object(&[("maxRetryDelayMs", "900")])),
                ]),
            ),
            (
                "skills",
                &settings_object(&[("enableSkillCommands", "true")]),
            ),
        ]),
    );

    let report = run_import_with_target(&options(&dir), &dir).expect("run");

    let item = global_item(&report);
    assert!(
        item.dropped
            .contains(&"retry.maxDelayMs (provider override already set)".to_string())
    );
    assert!(
        item.dropped
            .contains(&"skills (empty customDirectories)".to_string())
    );
    let written = std::fs::read_to_string(dir.join("settings.json")).unwrap();
    assert!(written.contains("\"maxRetryDelayMs\": 900"));
    assert!(!written.contains("maxDelayMs"));
    assert!(!written.contains("\"skills\""));
    cleanup(&dir);
}

#[test]
fn migrated_settings_are_idempotent_on_a_second_run() {
    let dir = source_dir("settings-idempotent");
    write(
        &dir,
        "settings.json",
        &settings_object(&[("queueMode", "\"all\"")]),
    );

    let _ = run_import_with_target(&options(&dir), &dir).expect("first run");
    let after_first = std::fs::read_to_string(dir.join("settings.json")).unwrap();
    let report = run_import_with_target(&options(&dir), &dir).expect("second run");

    assert_eq!(global_item(&report).outcome, SettingsOutcome::Unchanged);
    assert_eq!(
        std::fs::read_to_string(dir.join("settings.json")).unwrap(),
        after_first
    );
    cleanup(&dir);
}

#[test]
fn cross_dir_writes_when_absent_and_skips_when_present() {
    let source = source_dir("settings-cross");
    write(
        &source,
        "settings.json",
        &settings_object(&[("theme", "\"dark\"")]),
    );
    let target = temp_dir("settings-cross-target");
    let empty_target = temp_dir("settings-cross-empty");

    let first = run_import_with_target(&options(&source), &empty_target).expect("run");
    assert_eq!(global_item(&first).outcome, SettingsOutcome::Written);
    assert!(empty_target.join("settings.json").exists());

    // The other target lacks the recognition artifacts; seed one.
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(target.join("settings.json"), "{}").unwrap();
    let report = run_import_with_target(&options(&source), &target).expect("run");
    assert_eq!(
        global_item(&report).outcome,
        SettingsOutcome::Skipped {
            reason: "target already has settings.json".to_string()
        }
    );
    cleanup(&source);
    cleanup(&target);
    cleanup(&empty_target);
}

#[test]
fn project_scope_migrates_in_place_only_with_the_flag() {
    let dir = source_dir("settings-project");
    let project = temp_dir("settings-project-dir");
    write(
        &project,
        ".pi/settings.json",
        &settings_object(&[("queueMode", "\"one-at-a-time\"")]),
    );

    let without_project = run_import_with_target(&options(&dir), &dir).expect("run");
    assert_eq!(without_project.settings.len(), 1);

    let options = ImportOptions {
        source: Some(dir.clone()),
        project: Some(project.clone()),
        ..ImportOptions::default()
    };
    let report = run_import_with_target(&options, &dir).expect("run");
    assert_eq!(report.settings.len(), 2);
    let project_item = report
        .settings
        .iter()
        .find(|item| item.scope == "project")
        .expect("project item");
    assert_eq!(project_item.outcome, SettingsOutcome::Written);
    assert!(
        project_item
            .merged
            .contains(&"queueMode -> steeringMode".to_string())
    );
    let written = std::fs::read_to_string(project.join(".pi/settings.json")).unwrap();
    assert!(written.contains("\"steeringMode\""));
    cleanup(&dir);
    cleanup(&project);
}

#[test]
fn missing_project_settings_skip() {
    let dir = source_dir("settings-project-missing");
    let project = temp_dir("settings-project-missing-dir");

    let options = ImportOptions {
        source: Some(dir.clone()),
        project: Some(project.clone()),
        ..ImportOptions::default()
    };
    let report = run_import_with_target(&options, &dir).expect("run");
    let project_item = report
        .settings
        .iter()
        .find(|item| item.scope == "project")
        .expect("project item");
    assert_eq!(
        project_item.outcome,
        SettingsOutcome::Skipped {
            reason: "no settings file".to_string()
        }
    );
    cleanup(&dir);
    cleanup(&project);
}

#[test]
fn unreadable_and_non_object_settings_fail() {
    let dir = source_dir("settings-broken");
    write(&dir, "settings.json", "{not json");
    let report = run_import_with_target(&options(&dir), &dir).expect("run");
    assert!(matches!(
        global_item(&report).outcome,
        SettingsOutcome::Failed { .. }
    ));
    assert!(report.has_failures());
    cleanup(&dir);

    let dir = source_dir("settings-array");
    write(&dir, "settings.json", "[1,2]");
    let report = run_import_with_target(&options(&dir), &dir).expect("run");
    assert!(matches!(
        global_item(&report).outcome,
        SettingsOutcome::Failed { .. }
    ));
    // The array file stays untouched.
    assert_eq!(
        std::fs::read_to_string(dir.join("settings.json")).unwrap(),
        "[1,2]"
    );
    cleanup(&dir);
}

#[test]
fn bom_stripped_settings_migrate() {
    let dir = source_dir("settings-bom");
    write(
        &dir,
        "settings.json",
        &format!("\u{feff}{}", settings_object(&[("queueMode", "\"all\"")])),
    );
    let report = run_import_with_target(&options(&dir), &dir).expect("run");
    assert_eq!(global_item(&report).outcome, SettingsOutcome::Written);
    assert!(!report.has_failures());
    cleanup(&dir);
}

#[test]
fn direct_parse_helper_reports_its_errors() {
    assert!(settings::parse_object("{}").is_ok());
    assert!(settings::parse_object("{").is_err());
    assert!(settings::parse_object("[]").is_err());
}

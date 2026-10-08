#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::unwrap_used,
    reason = "the fixtures and assertions unwrap on their own setup only"
)]

//! The trust leg's suite: validation, the in-place and cross-dir
//! decisions, and the failure paths.

mod common;

use std::path::Path;

use common::{cleanup, source_dir, temp_dir, write};
use pi_import::discovery::ImportOptions;
use pi_import::report::TrustOutcome;
use pi_import::{run_import_with_target, trust};

fn options(source: &Path) -> ImportOptions {
    ImportOptions {
        source: Some(source.to_path_buf()),
        ..ImportOptions::default()
    }
}

fn trust_content() -> String {
    "{\"/a\": true,\n\"/b\": false,\n\"/c\": null}".to_string()
}

#[test]
fn in_place_validates_and_reports_entries() {
    let dir = source_dir("trust-in-place");
    write(&dir, "trust.json", &trust_content());
    let before = std::fs::read_to_string(dir.join("trust.json")).unwrap();

    let report = run_import_with_target(&options(&dir), &dir).expect("run");

    let item = report.trust.as_ref().expect("trust item");
    assert_eq!(item.path, dir.join("trust.json").display().to_string());
    assert_eq!(item.outcome, TrustOutcome::Unchanged { entries: 3 });
    assert_eq!(
        std::fs::read_to_string(dir.join("trust.json")).unwrap(),
        before
    );
    assert!(!report.has_failures());
    cleanup(&dir);
}

#[test]
fn cross_dir_copies_verbatim_and_skips_when_present() {
    let source = source_dir("trust-cross");
    write(&source, "trust.json", &trust_content());
    let target = temp_dir("trust-cross-target");

    let report = run_import_with_target(&options(&source), &target).expect("run");
    let item = report.trust.as_ref().expect("trust item");
    assert_eq!(item.outcome, TrustOutcome::Written { entries: 3 });
    assert_eq!(
        std::fs::read_to_string(target.join("trust.json")).unwrap(),
        trust_content()
    );

    let report = run_import_with_target(&options(&source), &target).expect("run again");
    let item = report.trust.as_ref().expect("trust item");
    assert_eq!(
        item.outcome,
        TrustOutcome::Skipped {
            reason: "target already has trust.json".to_string()
        }
    );
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn invalid_values_fail_the_leg() {
    let dir = source_dir("trust-bad-value");
    write(&dir, "trust.json", "{\"/a\": \"yes\"}");
    let report = run_import_with_target(&options(&dir), &dir).expect("run");
    let item = report.trust.as_ref().expect("trust item");
    assert!(matches!(item.outcome, TrustOutcome::Failed { .. }));
    assert!(
        report
            .failures()
            .iter()
            .any(|failure| failure.detail.contains("must be true, false, or null"))
    );
    cleanup(&dir);
}

#[test]
fn non_object_and_unreadable_stores_fail() {
    let dir = source_dir("trust-array");
    write(&dir, "trust.json", "[1]");
    let report = run_import_with_target(&options(&dir), &dir).expect("run");
    assert!(matches!(
        report.trust.as_ref().expect("item").outcome,
        TrustOutcome::Failed { .. }
    ));
    cleanup(&dir);

    let dir = source_dir("trust-broken");
    write(&dir, "trust.json", "{not");
    let report = run_import_with_target(&options(&dir), &dir).expect("run");
    assert!(matches!(
        report.trust.as_ref().expect("item").outcome,
        TrustOutcome::Failed { .. }
    ));
    cleanup(&dir);
}

#[test]
fn absent_store_reports_nothing() {
    let dir = source_dir("trust-absent");
    let report = run_import_with_target(&options(&dir), &dir).expect("run");
    assert!(report.trust.is_none());
    assert!(!report.has_failures());
    cleanup(&dir);
}

#[test]
fn direct_leg_run_plans_the_copy() {
    let source = source_dir("trust-direct");
    write(&source, "trust.json", &trust_content());
    let target = temp_dir("trust-direct-target");
    let discovery =
        pi_import::discovery::discover_with_target(&options(&source), &target).expect("recognized");
    let mut report = pi_import::report::ImportReport::default();
    let ops = trust::run_trust(&discovery, &mut report);
    assert_eq!(ops.len(), 1);
    ops[0].apply(false).expect("copy applies");
    assert!(target.join("trust.json").exists());
    cleanup(&source);
    cleanup(&target);
}

#[test]
fn a_directory_source_fails_the_read() {
    let dir = source_dir("trust-directory-source");
    std::fs::create_dir_all(dir.join("trust.json")).unwrap();
    let report = run_import_with_target(&options(&dir), &dir).expect("run");
    let item = report.trust.as_ref().expect("trust item");
    assert!(matches!(item.outcome, TrustOutcome::Failed { .. }));
    assert!(report.has_failures());
    cleanup(&dir);
}

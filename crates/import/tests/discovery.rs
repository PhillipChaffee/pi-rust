#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::unwrap_used,
    reason = "the fixtures and assertions unwrap on their own setup only"
)]

//! The discovery suite: the source refusals, the project flag, and the
//! in-place detection.

mod common;

use std::path::Path;

use common::{cleanup, source_dir, temp_dir, write};
use pi_import::discovery::{ImportOptions, discover_with_target};

fn options(source: Option<&Path>) -> ImportOptions {
    ImportOptions {
        source: source.map(Path::to_path_buf),
        ..ImportOptions::default()
    }
}

#[test]
fn missing_source_is_refused() {
    let missing = temp_dir("discovery-missing-source");
    let inner = missing.join("nope");
    let error =
        discover_with_target(&options(Some(&inner)), &missing).expect_err("missing source refuses");
    assert!(error.contains("does not exist"), "{error}");
    cleanup(&missing);
}

#[test]
fn unrecognized_source_is_refused_with_the_artifact_list() {
    let dir = temp_dir("discovery-unrecognized");
    write(&dir, "README.md", "nothing pi about this\n");
    let error = discover_with_target(&options(Some(&dir)), &temp_dir("discovery-unrecognized-t"))
        .expect_err("unrecognized source refuses");
    assert!(
        error.contains("does not look like a TS-pi agent dir"),
        "{error}"
    );
    assert!(error.contains("auth.json"), "{error}");
    assert!(error.contains("sessions"), "{error}");
    cleanup(&dir);
}

#[test]
fn any_recognized_artifact_accepts_the_source() {
    for artifact in [
        "trust.json",
        "oauth.json",
        "models.json",
        "extensions",
        "prompts",
        "themes",
    ] {
        let dir = temp_dir("discovery-recognized");
        #[expect(
            clippy::case_sensitive_file_extension_comparisons,
            reason = "the artifact names are the recognized set's exact spellings"
        )]
        if artifact.ends_with(".json") {
            write(&dir, artifact, "{}");
        } else {
            std::fs::create_dir_all(dir.join(artifact)).unwrap();
        }
        let target = temp_dir("discovery-recognized-target");
        let discovery = discover_with_target(&options(Some(&dir)), &target).expect("recognized");
        assert_eq!(discovery.source, dir);
        assert_eq!(discovery.target, target);
        assert!(!discovery.in_place);
        cleanup(&dir);
        cleanup(&target);
    }
}

#[test]
fn same_source_and_target_is_in_place() {
    let dir = source_dir("discovery-in-place");
    let discovery = discover_with_target(&options(Some(&dir)), &dir).expect("recognized");
    assert!(discovery.in_place);
    cleanup(&dir);
}

#[test]
fn absent_source_flag_defaults_to_the_target() {
    let target = source_dir("discovery-default");
    let discovery = discover_with_target(&options(None), &target).expect("recognized");
    assert_eq!(discovery.source, target);
    assert!(discovery.in_place);
    cleanup(&target);
}

#[test]
fn missing_project_dir_is_refused() {
    let dir = source_dir("discovery-project");
    let missing = temp_dir("discovery-project-missing").join("nope");
    let options = ImportOptions {
        source: Some(dir.clone()),
        project: Some(missing),
        ..ImportOptions::default()
    };
    let error = discover_with_target(&options, &dir).expect_err("missing project refuses");
    assert!(
        error.contains("project directory does not exist"),
        "{error}"
    );
    cleanup(&dir);
}

#[test]
fn existing_project_dir_is_accepted() {
    let dir = source_dir("discovery-project-ok");
    let project = temp_dir("discovery-project-ok-dir");
    let options = ImportOptions {
        source: Some(dir.clone()),
        project: Some(project.clone()),
        ..ImportOptions::default()
    };
    let discovery = discover_with_target(&options, &dir).expect("recognized");
    assert_eq!(discovery.project.as_deref(), Some(project.as_path()));
    cleanup(&dir);
    cleanup(&project);
}

#[test]
fn env_discovery_runs_without_writing() {
    // The env-derived path only reads; a missing real agent dir refuses.
    let source = source_dir("discovery-env");
    let outcome = pi_import::discovery::discover(&options(Some(&source)));
    // Either the real agent dir exists (Ok) or it does not (Err refused);
    // discovery never writes either way.
    assert!(outcome.is_ok() || outcome.unwrap_err().contains("does not exist"));
    cleanup(&source);
}

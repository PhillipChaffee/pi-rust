//! Boundary tests binding the trust-manager branches the 1:1 suite leaves
//! untested, at pin 60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::path::PathBuf;
use std::time::Duration;

use pi_coding_agent::file_lock::{acquire_sync_retrying, lock_dir_for};
use pi_coding_agent::trust_manager::{
    ProjectTrustStore, ProjectTrustUpdate, get_project_trust_options,
    get_project_trust_parent_path, has_trust_requiring_project_resources,
};
use serde_json::Value;

// === fixtures ===============================================================

fn canonical(path: &std::path::Path) -> String {
    std::fs::canonicalize(path)
        .expect("canonicalize")
        .to_string_lossy()
        .into_owned()
}

struct Fixture {
    root: tempfile::TempDir,
    agent_dir: PathBuf,
    project_dir: PathBuf,
}

fn fixture() -> Fixture {
    let root = tempfile::tempdir().expect("scratch root");
    let agent_dir = root.path().join("agent");
    let project_dir = root.path().join("project");
    std::fs::create_dir_all(&agent_dir).expect("agent dir");
    std::fs::create_dir_all(&project_dir).expect("project dir");
    Fixture {
        root,
        agent_dir,
        project_dir,
    }
}

impl Fixture {
    fn store(&self) -> ProjectTrustStore {
        ProjectTrustStore::new(&self.agent_dir.to_string_lossy())
    }

    fn trust_path(&self) -> String {
        // the store resolves the agent dir without canonicalizing, so the
        // path text it fails with is this form
        self.agent_dir
            .join("trust.json")
            .to_string_lossy()
            .into_owned()
    }
}

// === the trust prompt options ===============================================

#[test]
fn trust_options_offer_the_five_choices_with_a_parent() {
    let fixture = fixture();
    let cwd = canonical(&fixture.project_dir);
    let parent = canonical(fixture.root.path());

    let options = get_project_trust_options(&cwd, true);
    assert_eq!(options.len(), 5);

    assert_eq!(options[0].label, "Trust");
    assert!(options[0].trusted);
    assert_eq!(
        options[0].updates,
        vec![ProjectTrustUpdate {
            path: cwd.clone(),
            decision: Some(true),
        }]
    );
    assert_eq!(options[0].saved_path, Some(cwd.clone()));

    assert_eq!(options[1].label, format!("Trust parent folder ({parent})"));
    assert!(options[1].trusted);
    assert_eq!(
        options[1].updates,
        vec![
            ProjectTrustUpdate {
                path: parent.clone(),
                decision: Some(true),
            },
            ProjectTrustUpdate {
                path: cwd.clone(),
                decision: None,
            },
        ]
    );
    assert_eq!(options[1].saved_path, Some(parent));
    assert_eq!(options[2].label, "Trust (this session only)");
    assert!(options[2].trusted);
    assert!(options[2].updates.is_empty());
    assert_eq!(options[2].saved_path, None);

    assert_eq!(options[3].label, "Do not trust");
    assert!(!options[3].trusted);
    assert_eq!(
        options[3].updates,
        vec![ProjectTrustUpdate {
            path: cwd.clone(),
            decision: Some(false),
        }]
    );
    assert_eq!(options[3].saved_path, Some(cwd));

    assert_eq!(options[4].label, "Do not trust (this session only)");
    assert!(!options[4].trusted);
    assert!(options[4].updates.is_empty());
    assert_eq!(options[4].saved_path, None);
}

#[test]
fn trust_options_skip_the_parent_at_the_filesystem_root() {
    let with_session = get_project_trust_options("/", true);
    let labels: Vec<&str> = with_session
        .iter()
        .map(|option| option.label.as_str())
        .collect();
    assert_eq!(
        labels,
        vec![
            "Trust",
            "Trust (this session only)",
            "Do not trust",
            "Do not trust (this session only)",
        ]
    );

    let without_session = get_project_trust_options("/", false);
    let labels: Vec<&str> = without_session
        .iter()
        .map(|option| option.label.as_str())
        .collect();
    assert_eq!(labels, vec!["Trust", "Do not trust"]);
}

#[test]
fn trust_options_without_the_session_only_variants() {
    let fixture = fixture();
    let cwd = canonical(&fixture.project_dir);
    let parent = canonical(fixture.root.path());
    let options = get_project_trust_options(&cwd, false);
    let labels: Vec<&str> = options.iter().map(|option| option.label.as_str()).collect();
    assert_eq!(
        labels,
        vec![
            "Trust",
            &format!("Trust parent folder ({parent})"),
            "Do not trust"
        ]
    );
}

#[test]
fn the_parent_path_is_none_at_the_root() {
    assert_eq!(get_project_trust_parent_path("/"), None);

    let fixture = fixture();
    let cwd = canonical(&fixture.project_dir);
    let parent = canonical(fixture.root.path());
    assert_eq!(get_project_trust_parent_path(&cwd), Some(parent));
}

// === the store file grammar =================================================

#[test]
fn null_entries_are_climbed_past_and_preserved_on_write() {
    let fixture = fixture();
    let cwd = canonical(&fixture.project_dir);
    let parent = canonical(fixture.root.path());
    let trust_path = fixture.trust_path();
    std::fs::write(
        &trust_path,
        format!(r#"{{"{cwd}": null, "{parent}": true}}"#),
    )
    .expect("trust file write");

    let store = fixture.store();
    // the null at cwd climbs past to the parent's decision
    assert_eq!(store.get(&cwd).expect("trust get"), Some(true));

    // a write elsewhere preserves the null entry
    let other = canonical(&fixture.agent_dir);
    store.set(&other, Some(false)).expect("trust set");
    let content = std::fs::read_to_string(&trust_path).expect("trust read");
    let parsed: Value = serde_json::from_str(&content).expect("trust parse");
    assert_eq!(parsed.get(&cwd), Some(&Value::Null));
    assert_eq!(parsed.get(&parent), Some(&Value::Bool(true)));
    assert_eq!(parsed.get(&other), Some(&Value::Bool(false)));
}

#[test]
fn a_malformed_store_file_is_rejected_with_the_verbatim_messages() {
    let fixture = fixture();
    let trust_path = fixture.trust_path();
    let store = fixture.store();

    std::fs::write(&trust_path, "not json").expect("trust write");
    let error = store.get("/").expect_err("read failure");
    assert!(
        error
            .to_string()
            .starts_with(&format!("Failed to read trust store {trust_path}: ")),
        "{error}"
    );

    std::fs::write(&trust_path, "[1]").expect("trust write");
    let error = store.get("/").expect_err("read failure");
    assert_eq!(
        error.to_string(),
        format!("Invalid trust store {trust_path}: expected an object")
    );

    std::fs::write(&trust_path, r#"{"a": 5}"#).expect("trust write");
    let error = store.get("/").expect_err("read failure");
    assert_eq!(
        error.to_string(),
        format!(r#"Invalid trust store {trust_path}: value for "a" must be true, false, or null"#)
    );
}

// === the parent ascent and the canonicalize fallback ========================

#[test]
fn an_ascent_through_missing_paths_finds_the_lexical_parent() {
    let fixture = fixture();
    let store = fixture.store();
    let missing_parent = fixture.root.path().join("missing-parent");
    let missing_child = missing_parent.join("missing-child");
    let parent = missing_parent.to_string_lossy().into_owned();
    let child = missing_child.to_string_lossy().into_owned();

    // neither path exists, so the decision stores under the lexical form the
    // canonicalize fallback keeps, and the climb matches it
    store.set(&parent, Some(true)).expect("trust set");
    assert_eq!(store.get(&parent).expect("trust get"), Some(true));
    assert_eq!(store.get(&child).expect("trust get"), Some(true));
}

// === locking ================================================================

#[test]
fn lock_contention_waits_for_the_holder_to_release() {
    let fixture = fixture();
    let cwd = canonical(&fixture.project_dir);
    let trust_path = fixture.trust_path();
    let lock_dir = lock_dir_for(&trust_path);

    let guard = acquire_sync_retrying(&lock_dir).expect("held lock");
    let agent = fixture.agent_dir.to_string_lossy().into_owned();
    let worker_cwd = cwd.clone();
    let worker =
        std::thread::spawn(move || ProjectTrustStore::new(&agent).set(&worker_cwd, Some(true)));
    // the worker burns its first acquire attempts against the held lock
    std::thread::sleep(Duration::from_millis(100));
    guard.release().expect("lock release");
    worker.join().expect("worker join").expect("trust set");

    assert_eq!(fixture.store().get(&cwd).expect("trust get"), Some(true));
}

#[test]
fn an_unwritable_store_dir_fails_get_and_set() {
    let root = tempfile::tempdir().expect("scratch root");
    // the agent "dir" is a regular file: the store's create_dir_all fails
    let blocker = root.path().join("agent");
    std::fs::write(&blocker, b"not a directory").expect("blocker write");
    let store = ProjectTrustStore::new(&blocker.to_string_lossy());

    let debug = format!("{store:?}");
    assert!(debug.contains("trust.json"), "{debug}");

    let error = store.get("/").expect_err("get failure");
    assert!(!error.to_string().is_empty());

    let error = store.set("/", Some(true)).expect_err("set failure");
    assert!(!error.to_string().is_empty());
}

#[test]
fn a_store_file_that_is_a_directory_fails_the_read() {
    let fixture = fixture();
    let trust_path = fixture.trust_path();
    std::fs::create_dir_all(&trust_path).expect("trust.json as directory");
    let store = fixture.store();

    let error = store.get("/").expect_err("read failure");
    assert!(
        !error.to_string().starts_with("Invalid trust store"),
        "{error}"
    );
}

#[test]
fn a_lock_path_held_by_a_regular_file_fails_the_acquire() {
    let fixture = fixture();
    let cwd = canonical(&fixture.project_dir);
    let lock = format!("{}.lock", fixture.trust_path());
    std::fs::write(&lock, b"not a directory").expect("lock blocker write");
    let store = fixture.store();

    let error = store.get("/").expect_err("get failure");
    assert_eq!(error.to_string(), "locked");

    let error = store.set(&cwd, Some(true)).expect_err("set failure");
    assert_eq!(error.to_string(), "locked");
}

#[test]
fn a_readonly_store_file_fails_the_write() {
    let fixture = fixture();
    let cwd = canonical(&fixture.project_dir);
    let trust_path = fixture.trust_path();
    std::fs::write(&trust_path, "{}").expect("trust write");
    let mut permissions = std::fs::metadata(&trust_path)
        .expect("trust metadata")
        .permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(&trust_path, permissions).expect("trust chmod");
    let store = fixture.store();

    // the read succeeds, the write fails
    assert_eq!(store.get(&cwd).expect("trust get"), None);
    let error = store.set(&cwd, Some(true)).expect_err("set failure");
    assert!(!error.to_string().is_empty());
    assert_eq!(store.get(&cwd).expect("trust get"), None);
}

// === the resource probe =====================================================

#[test]
fn trust_requiring_resources_cover_the_config_dir_and_the_skill_climb() {
    let root = tempfile::tempdir().expect("scratch root");
    let project = root.path().join("project");
    std::fs::create_dir_all(&project).expect("project dir");
    let cwd = project.to_string_lossy().into_owned();

    assert!(!has_trust_requiring_project_resources(&cwd));

    std::fs::create_dir_all(project.join(".pi")).expect(".pi dir");
    std::fs::write(project.join(".pi").join("SYSTEM.md"), "").expect("SYSTEM.md write");
    assert!(has_trust_requiring_project_resources(&cwd));

    std::fs::remove_dir_all(project.join(".pi")).expect(".pi remove");
    assert!(!has_trust_requiring_project_resources(&cwd));

    // an ancestor's .agents/skills counts; the user's own does not
    std::fs::create_dir_all(root.path().join(".agents").join("skills"))
        .expect("ancestor skills dir");
    assert!(has_trust_requiring_project_resources(&cwd));
}

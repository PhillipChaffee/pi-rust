//! Upstream `test/trust-manager.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated.
//!
//! Upstream's `process.env.HOME` swap rides the workspace's no-env-mutation
//! rule: `detects trust-requiring project resources` drives the
//! `has_trust_requiring_project_resources_with_home` seam with the temp
//! directory standing in for `$HOME`. The store calls return
//! `Result<Option<bool>, TrustError>` — upstream's `null` "nothing decided"
//! is `Ok(None)` and its malformed-file throw is the `Err`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::fs;

use pi_coding_agent::trust_manager::{
    ProjectTrustStore, has_trust_requiring_project_resources_with_home,
};

#[test]
fn stores_decisions_and_inherits_from_parent_directories() {
    let temp = tempfile::tempdir().expect("temp dir");
    let agent_dir = temp.path().join("agent");
    let cwd = temp.path().join("project");
    fs::create_dir_all(&agent_dir).expect("agent dir");
    fs::create_dir_all(&cwd).expect("cwd dir");

    let store = ProjectTrustStore::new(&agent_dir.to_string_lossy());
    let parent_dir = temp.path().join("trusted-parent");
    let child_dir = parent_dir.join("project");
    fs::create_dir_all(&child_dir).expect("child dir");
    let parent = parent_dir.to_string_lossy().into_owned();
    let child = child_dir.to_string_lossy().into_owned();

    assert_eq!(store.get(&child).expect("trust get"), None);
    store.set(&parent, Some(true)).expect("trust set");
    assert_eq!(store.get(&child).expect("trust get"), Some(true));
    store.set(&child, Some(false)).expect("trust set");
    assert_eq!(store.get(&child).expect("trust get"), Some(false));
    // the null decision deletes the entry; the parent's trust governs again
    store.set(&child, None).expect("trust set");
    assert_eq!(store.get(&child).expect("trust get"), Some(true));
}

#[test]
fn detects_trust_requiring_project_resources() {
    let temp = tempfile::tempdir().expect("temp dir");
    let home = temp.path().to_string_lossy().into_owned();
    let cwd_dir = temp.path().join("project");
    fs::create_dir_all(&cwd_dir).expect("project dir");
    let cwd = cwd_dir.to_string_lossy().into_owned();

    fs::create_dir_all(temp.path().join(".pi").join("agent")).expect(".pi/agent dir");
    fs::create_dir_all(temp.path().join(".agents").join("skills")).expect(".agents/skills dir");
    // home's own .agents/skills is a trusted user resource, and .pi/agent
    // carries no trust-requiring entries
    assert!(!has_trust_requiring_project_resources_with_home(
        &home, &home
    ));
    assert!(!has_trust_requiring_project_resources_with_home(
        &cwd, &home
    ));

    fs::write(temp.path().join(".pi").join("settings.json"), "{}").expect("settings write");
    assert!(has_trust_requiring_project_resources_with_home(
        &home, &home
    ));
    fs::remove_file(temp.path().join(".pi").join("settings.json")).expect("settings remove");

    fs::create_dir_all(cwd_dir.join(".pi")).expect("project .pi dir");
    fs::write(cwd_dir.join(".pi").join("settings.json"), "{}").expect("project settings write");
    assert!(has_trust_requiring_project_resources_with_home(&cwd, &home));

    fs::remove_dir_all(cwd_dir.join(".pi")).expect("project .pi remove");
    fs::create_dir_all(cwd_dir.join(".agents").join("skills")).expect("project skills dir");
    assert!(has_trust_requiring_project_resources_with_home(&cwd, &home));
}

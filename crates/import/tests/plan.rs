#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::unwrap_used,
    reason = "the fixtures and assertions unwrap on their own setup only"
)]

//! The plan's suite: the apply paths, the dry-run branch, and the failure
//! conversions.

mod common;

use common::{cleanup, temp_dir, write};
use pi_import::plan::{OpTarget, PlannedOp};

#[test]
fn write_file_applies_the_mode() {
    use std::os::unix::fs::PermissionsExt;

    let dir = temp_dir("plan-write-mode");
    let path = dir.join("sub/auth.json");
    let op = PlannedOp::WriteFile {
        path: path.clone(),
        content: "{}".to_string(),
        mode: Some(0o600),
        targets: vec![OpTarget::Trust],
    };
    op.apply(false).expect("write applies");
    let mode = std::fs::metadata(&path).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "{}");
    cleanup(&dir);
}

#[test]
fn write_file_defaults_to_the_umask() {
    let dir = temp_dir("plan-write-default");
    let path = dir.join("settings.json");
    let op = PlannedOp::WriteFile {
        path: path.clone(),
        content: "{}".to_string(),
        mode: None,
        targets: Vec::new(),
    };
    op.apply(false).expect("write applies");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "{}");
    cleanup(&dir);
}

#[test]
fn copy_and_rename_create_parents() {
    let dir = temp_dir("plan-copy-rename");
    let from = write(&dir, "src/file.jsonl", "x\n");
    let copied = dir.join("deep/er/file.jsonl");
    let op = PlannedOp::CopyFile {
        from: from.clone(),
        to: copied.clone(),
        targets: vec![OpTarget::Session(0)],
    };
    op.apply(false).expect("copy applies");
    assert_eq!(std::fs::read_to_string(&copied).unwrap(), "x\n");

    let renamed = dir.join("other/file.jsonl");
    let op = PlannedOp::Rename {
        from,
        to: renamed.clone(),
        targets: vec![OpTarget::Session(1)],
    };
    op.apply(false).expect("rename applies");
    assert!(!dir.join("src/file.jsonl").exists());
    assert_eq!(std::fs::read_to_string(&renamed).unwrap(), "x\n");
    cleanup(&dir);
}

#[test]
fn dry_run_writes_nothing() {
    let dir = temp_dir("plan-dry");
    let op = PlannedOp::WriteFile {
        path: dir.join("out.json"),
        content: "{}".to_string(),
        mode: None,
        targets: Vec::new(),
    };
    op.apply(true).expect("dry run applies nothing");
    assert!(!dir.join("out.json").exists());
    cleanup(&dir);
}

#[test]
fn failures_carry_the_filesystem_message() {
    let dir = temp_dir("plan-failures");
    let op = PlannedOp::CopyFile {
        from: dir.join("missing.jsonl"),
        to: dir.join("out.jsonl"),
        targets: vec![OpTarget::Session(0)],
    };
    let error = op.apply(false).expect_err("missing source fails");
    assert!(!error.is_empty());

    // A parent that is a file fails the write's directory creation.
    write(&dir, "blocker", "x");
    let op = PlannedOp::WriteFile {
        path: dir.join("blocker/child.json"),
        content: "{}".to_string(),
        mode: None,
        targets: Vec::new(),
    };
    assert!(op.apply(false).is_err());
    cleanup(&dir);
}

#[test]
fn targets_lists_every_operation_kind() {
    let dir = temp_dir("plan-targets");
    let targets = vec![OpTarget::Session(0), OpTarget::Credential(1)];
    let op = PlannedOp::WriteFile {
        path: dir.join("out.json"),
        content: "{}".to_string(),
        mode: None,
        targets: targets.clone(),
    };
    assert_eq!(op.targets(), targets.as_slice());
    cleanup(&dir);
}

#[test]
fn write_file_to_a_directory_fails_the_open() {
    let dir = temp_dir("plan-write-dir");
    std::fs::create_dir_all(dir.join("occupied.json")).unwrap();
    let op = PlannedOp::WriteFile {
        path: dir.join("occupied.json"),
        content: "{}".to_string(),
        mode: Some(0o600),
        targets: Vec::new(),
    };
    assert!(op.apply(false).is_err());
    let op = PlannedOp::WriteFile {
        path: dir.join("occupied.json"),
        content: "{}".to_string(),
        mode: None,
        targets: Vec::new(),
    };
    assert!(op.apply(false).is_err());
    cleanup(&dir);
}

#[test]
fn rename_to_a_occupied_target_fails() {
    let dir = temp_dir("plan-rename-fail");
    let from = write(&dir, "a.jsonl", "x\n");
    // The destination is a non-empty directory: rename fails.
    std::fs::create_dir_all(dir.join("b.jsonl/inner")).unwrap();
    let op = PlannedOp::Rename {
        from,
        to: dir.join("b.jsonl"),
        targets: Vec::new(),
    };
    assert!(op.apply(false).is_err());
    cleanup(&dir);
}

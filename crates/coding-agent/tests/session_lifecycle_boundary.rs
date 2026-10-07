//! The lifecycle boundary suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: `open`'s cwd override, the
//! bounded header scan's full-load fallback, the io arm, `forkFrom`'s fresh
//! directory creation, and the session-cwd seam.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::fs;

use pi_coding_agent::session_cwd::SessionCwdSource;
use pi_coding_agent::session_manager::{SessionManager, SessionManagerError};

fn user_header(id: &str, cwd: &str) -> String {
    format!(
        r#"{{"type":"session","version":3,"id":"{id}","timestamp":"2026-01-01T00:00:00.000Z","cwd":"{cwd}"}}"#
    )
}

#[test]
fn the_cwd_override_skips_the_header_scan_and_the_directory_override_normalizes() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let session_file = format!("{temp}/session.jsonl");
    fs::write(&session_file, format!("{}\n", user_header("ovr", &temp))).expect("write session");

    let override_dir = format!("{temp}/other-sessions");
    let session =
        SessionManager::open(&session_file, Some(&override_dir), Some(&temp)).expect("open");
    assert_eq!(
        session.cwd(),
        &temp,
        "the override wins over the header cwd"
    );
    assert_eq!(
        session.session_dir(),
        override_dir,
        "the explicit directory normalizes as-is"
    );
    assert_eq!(session.session_id(), "ovr", "the entries still load");
}

#[test]
fn opening_a_non_session_file_reports_the_io_arm() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let error =
        SessionManager::open(&temp, None, None).expect_err("a directory is not a session file");
    assert!(
        matches!(error, SessionManagerError::Io(_)),
        "the directory read surfaces as io: {error}"
    );
}

#[test]
fn a_header_past_the_scan_limit_falls_back_to_the_full_load() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let session_file = format!("{temp}/big.jsonl");
    let mut content = format!("{}\n", "x".repeat(1024 * 1024 + 16));
    content.push_str(&user_header("big", &temp));
    content.push('\n');
    fs::write(&session_file, content).expect("write oversized file");

    let session =
        SessionManager::open(&session_file, None, None).expect("the scan falls back to the load");
    assert_eq!(
        session.session_id(),
        "big",
        "the full load still finds the header"
    );
    assert_eq!(session.cwd(), &temp, "the fallback header supplies the cwd");
}

#[test]
fn fork_from_creates_a_fresh_target_directory() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let source = format!("{temp}/source.jsonl");
    fs::write(&source, format!("{}\n", user_header("src", &temp))).expect("write source");

    let fresh_dir = format!("{temp}/fresh-sessions");
    let forked = SessionManager::fork_from(&source, &temp, Some(&fresh_dir), None).expect("fork");
    assert_eq!(
        forked.session_dir(),
        fresh_dir,
        "the target directory normalizes"
    );
    assert!(
        fs::exists(&fresh_dir).unwrap_or(false),
        "the fresh directory is created"
    );
    let file = forked.session_file().expect("the fork writes its file");
    let content = fs::read_to_string(file).expect("read fork file");
    assert!(
        content.contains(r#""parentSession""#),
        "the fork header links to the source"
    );
    assert!(
        content.contains(r#""cwd""#),
        "the fork header carries the target cwd"
    );
}

#[test]
fn the_session_cwd_seam_reads_the_manager() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    assert_eq!(
        SessionCwdSource::cwd(&session),
        temp,
        "the seam reads the manager cwd"
    );
    assert!(SessionCwdSource::session_file(&session).is_some());

    let memory = SessionManager::in_memory(None, None, None).expect("in-memory");
    assert_eq!(
        SessionCwdSource::session_file(&memory),
        None,
        "an in-memory session has no file for the seam"
    );
}

#[test]
fn a_oversized_file_without_a_session_header_reports_the_invalid_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let session_file = format!("{temp}/garbage.jsonl");
    let content = format!("{}\n", "x".repeat(1024 * 1024 + 16));
    fs::write(&session_file, content).expect("write oversized garbage");

    let error = SessionManager::open(&session_file, None, None).expect_err("no header, no session");
    assert!(
        matches!(error, SessionManagerError::InvalidSessionFile(_)),
        "the full load finds no header, so the file is invalid: {error}"
    );
}

#[test]
fn opening_an_empty_file_initializes_it_with_a_header() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let session_file = format!("{temp}/empty.jsonl");
    fs::write(&session_file, "").expect("write empty file");

    let session = SessionManager::open(&session_file, None, None).expect("open empty");
    assert_eq!(session.session_file(), Some(session_file.as_str()));
    let content = fs::read_to_string(&session_file).expect("read");
    assert!(
        content.contains(r#""type":"session""#),
        "the empty file gains a header: {content}"
    );
}

#[test]
fn opening_a_missing_file_starts_a_new_session_at_that_path() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let session_file = format!("{temp}/fresh.jsonl");
    let override_dir = format!("{temp}/sessions");
    let session =
        SessionManager::open(&session_file, Some(&override_dir), None).expect("open missing");
    assert_eq!(
        session.session_file(),
        Some(session_file.as_str()),
        "the explicit path is preserved"
    );
    assert_eq!(session.session_dir(), override_dir);
}

#[test]
fn a_directory_that_does_not_normalize_rejects_create_and_fork() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let create_error =
        SessionManager::create(&temp, Some("file://host/path"), None).expect_err("normalize");
    assert!(
        matches!(create_error, SessionManagerError::PathNormalize(_)),
        "{create_error}"
    );

    let source = format!("{temp}/source.jsonl");
    fs::write(&source, format!("{}\n", user_header("src", &temp))).expect("write source");
    let fork_error = SessionManager::fork_from(&source, &temp, Some("file://host/path"), None)
        .expect_err("normalize");
    assert!(
        matches!(fork_error, SessionManagerError::PathNormalize(_)),
        "{fork_error}"
    );
}

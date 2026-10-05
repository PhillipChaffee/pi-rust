//! Boundary tests for the missing-session-cwd surface: the error and prompt
//! bodies verbatim upstream, and the falsiness/existence branches the
//! upstream suites reach only through their persisted-session fixtures.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use pi_coding_agent::session_cwd::{
    SessionCwdIssue, SessionCwdSource, format_missing_session_cwd_error,
    format_missing_session_cwd_prompt, get_missing_session_cwd_issue,
};

/// The `SessionManager` stand-in for the branch fixtures.
struct StubSource {
    cwd: String,
    session_file: Option<String>,
}

impl SessionCwdSource for StubSource {
    fn cwd(&self) -> &str {
        &self.cwd
    }

    fn session_file(&self) -> Option<&str> {
        self.session_file.as_deref()
    }
}

fn issue() -> SessionCwdIssue {
    SessionCwdIssue {
        session_file: Some("/sessions/session.jsonl".to_string()),
        session_cwd: "/gone/cwd".to_string(),
        fallback_cwd: "/here/cwd".to_string(),
    }
}

#[test]
fn the_error_body_matches_upstream_verbatim() {
    assert_eq!(
        format_missing_session_cwd_error(&issue()),
        "Stored session working directory does not exist: /gone/cwd\nSession file: /sessions/session.jsonl\nCurrent working directory: /here/cwd"
    );
}

#[test]
fn the_error_body_drops_the_session_file_line_when_absent() {
    let bare = SessionCwdIssue {
        session_file: None,
        ..issue()
    };

    assert_eq!(
        format_missing_session_cwd_error(&bare),
        "Stored session working directory does not exist: /gone/cwd\nCurrent working directory: /here/cwd"
    );
}

#[test]
fn the_prompt_body_matches_upstream_verbatim() {
    assert_eq!(
        format_missing_session_cwd_prompt(&issue()),
        "cwd from session file does not exist\n/gone/cwd\n\ncontinue in current cwd\n/here/cwd"
    );
}

#[test]
fn an_empty_cwd_counts_as_absent() {
    // upstream's falsiness: `!sessionCwd` short-circuits before existsSync
    let source = StubSource {
        cwd: String::new(),
        session_file: Some("/sessions/session.jsonl".to_string()),
    };

    assert_eq!(get_missing_session_cwd_issue(&source, "/here/cwd"), None);
}

#[test]
fn an_existing_cwd_is_not_an_issue() {
    let existing = tempfile::tempdir().expect("scratch dir");
    let source = StubSource {
        cwd: existing.path().to_string_lossy().into_owned(),
        session_file: Some("/sessions/session.jsonl".to_string()),
    };

    assert_eq!(get_missing_session_cwd_issue(&source, "/here/cwd"), None);
}

#[test]
fn a_missing_cwd_without_a_session_file_is_not_an_issue() {
    let source = StubSource {
        cwd: "/gone/cwd".to_string(),
        session_file: None,
    };

    assert_eq!(get_missing_session_cwd_issue(&source, "/here/cwd"), None);
}

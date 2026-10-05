//! Upstream `test/session-cwd.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated.
//!
//! The three cases run against a stub source standing in for
//! `SessionManager.open` — the real manager reads the header cwd and file
//! path from the persisted session, and the JSONL session manager implements
//! [`SessionCwdSource`] when its port lands (map ticket "pi-coding-agent:
//! session manager and export"); that ticket's suites carry the file-parsing
//! half of these cases. The runtime-creation ordering of the third case
//! rides the agent-session-runtime port, which gates on this assert.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use pi_coding_agent::session_cwd::{
    SessionCwdIssue, SessionCwdSource, assert_session_cwd_exists, get_missing_session_cwd_issue,
};

/// The `SessionManager.open` stand-in: the persisted cwd and session file it
/// would report.
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

#[test]
fn detects_missing_session_cwd_from_persisted_sessions() {
    // upstream writes a session file whose header cwd points at a missing
    // directory and opens it; the stub carries the opened values
    let fallback = tempfile::tempdir().expect("scratch fallback dir");
    let missing = fallback.path().join("does-not-exist");
    let source = StubSource {
        cwd: missing.to_string_lossy().into_owned(),
        session_file: Some("/sessions/session.jsonl".to_string()),
    };

    let issue = get_missing_session_cwd_issue(&source, &fallback.path().to_string_lossy());

    assert_eq!(
        issue,
        Some(SessionCwdIssue {
            session_file: Some("/sessions/session.jsonl".to_string()),
            session_cwd: missing.to_string_lossy().into_owned(),
            fallback_cwd: fallback.path().to_string_lossy().into_owned(),
        })
    );
}

#[test]
fn supports_overriding_the_effective_cwd_when_opening_a_session() {
    // upstream opens with the fallback as the effective cwd override; the
    // session reports it and no issue remains
    let fallback = tempfile::tempdir().expect("scratch fallback dir");
    let missing = fallback.path().join("does-not-exist");
    let source = StubSource {
        cwd: fallback.path().to_string_lossy().into_owned(),
        session_file: Some("/sessions/session.jsonl".to_string()),
    };

    let issue = get_missing_session_cwd_issue(&source, &missing.to_string_lossy());

    assert_eq!(issue, None);
}

#[test]
fn fails_a_controlled_error_before_runtime_creation_when_the_stored_cwd_is_missing() {
    let fallback = tempfile::tempdir().expect("scratch fallback dir");
    let missing = fallback.path().join("does-not-exist");
    let source = StubSource {
        cwd: missing.to_string_lossy().into_owned(),
        session_file: Some("/sessions/session.jsonl".to_string()),
    };

    let result = assert_session_cwd_exists(&source, &fallback.path().to_string_lossy());

    let error = result.expect_err("the missing stored cwd is the controlled failure");
    assert_eq!(
        error.issue,
        SessionCwdIssue {
            session_file: Some("/sessions/session.jsonl".to_string()),
            session_cwd: missing.to_string_lossy().into_owned(),
            fallback_cwd: fallback.path().to_string_lossy().into_owned(),
        }
    );
    assert!(
        error
            .to_string()
            .starts_with("Stored session working directory does not exist: ",)
    );
}

#[test]
fn passes_when_the_stored_cwd_exists() {
    let fallback = tempfile::tempdir().expect("scratch fallback dir");
    let source = StubSource {
        cwd: fallback.path().to_string_lossy().into_owned(),
        session_file: Some("/sessions/session.jsonl".to_string()),
    };

    assert!(
        assert_session_cwd_exists(&source, &fallback.path().to_string_lossy()).is_ok(),
        "an existing stored cwd passes the gate"
    );
}

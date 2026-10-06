//! The missing-session-cwd detector, upstream's
//! `src/core/session-cwd.ts`: decide whether a session's stored cwd still
//! exists, and format the error and the prompt when it does not.
//!
//! The checks run against the [`SessionCwdSource`] seam rather than a
//! concrete session manager — upstream types the source as an interface for
//! the same reason, and the JSONL session manager implements it when its
//! port lands (map ticket "pi-coding-agent: session manager and export").

use std::fmt;
use std::path::Path;

/// What a session reports for the cwd check, upstream's
/// `SessionCwdSource`.
pub trait SessionCwdSource {
    /// The session's effective cwd, upstream's `getCwd()`. An empty string
    /// counts as absent, upstream's falsiness.
    fn cwd(&self) -> &str;

    /// The session file path, upstream's `getSessionFile()`; absent for an
    /// in-memory session.
    fn session_file(&self) -> Option<&str>;
}

/// The detected missing-cwd condition, upstream's `SessionCwdIssue`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionCwdIssue {
    /// The session file the stale cwd came from, upstream's `sessionFile`.
    pub session_file: Option<String>,
    /// The stored cwd that does not exist, upstream's `sessionCwd`.
    pub session_cwd: String,
    /// The cwd to fall back to, upstream's `fallbackCwd`.
    pub fallback_cwd: String,
}

/// Detect the missing-session-cwd condition, upstream's
/// `getMissingSessionCwdIssue`: no issue without a session file, no issue
/// with an absent (empty) or existing cwd.
#[must_use]
pub fn get_missing_session_cwd_issue(
    session_manager: &impl SessionCwdSource,
    fallback_cwd: &str,
) -> Option<SessionCwdIssue> {
    let session_file = session_manager.session_file()?;
    let session_cwd = session_manager.cwd();
    if session_cwd.is_empty() || Path::new(session_cwd).exists() {
        return None;
    }
    Some(SessionCwdIssue {
        session_file: Some(session_file.to_string()),
        session_cwd: session_cwd.to_string(),
        fallback_cwd: fallback_cwd.to_string(),
    })
}

/// The error body, upstream's `formatMissingSessionCwdError`: the
/// session-file line appears only when the issue carries one.
#[must_use]
pub fn format_missing_session_cwd_error(issue: &SessionCwdIssue) -> String {
    let session_file = issue
        .session_file
        .as_ref()
        .map_or_else(String::new, |file| format!("\nSession file: {file}"));
    format!(
        "Stored session working directory does not exist: {}{}\nCurrent working directory: {}",
        issue.session_cwd, session_file, issue.fallback_cwd
    )
}

/// The selector prompt body, upstream's `formatMissingSessionCwdPrompt`.
#[must_use]
pub fn format_missing_session_cwd_prompt(issue: &SessionCwdIssue) -> String {
    format!(
        "cwd from session file does not exist\n{}\n\ncontinue in current cwd\n{}",
        issue.session_cwd, issue.fallback_cwd
    )
}

/// The controlled error for a missing stored cwd, upstream's
/// `MissingSessionCwdError` (`name` `"MissingSessionCwdError"`, message from
/// [`format_missing_session_cwd_error`]).
#[derive(Debug)]
pub struct MissingSessionCwdError {
    /// The detected issue, upstream's `issue` field.
    pub issue: SessionCwdIssue,
}

impl MissingSessionCwdError {
    /// Wrap the issue, upstream's constructor.
    #[must_use]
    pub const fn new(issue: SessionCwdIssue) -> Self {
        Self { issue }
    }
}

impl fmt::Display for MissingSessionCwdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_missing_session_cwd_error(&self.issue))
    }
}

impl std::error::Error for MissingSessionCwdError {}

/// Fail when the stored session cwd does not exist, upstream's
/// `assertSessionCwdExists` — the gate callers run before creating a runtime
/// over a persisted session.
///
/// # Errors
/// The [`MissingSessionCwdError`] the detector produces, instead of the
/// upstream throw.
pub fn assert_session_cwd_exists(
    session_manager: &impl SessionCwdSource,
    fallback_cwd: &str,
) -> Result<(), MissingSessionCwdError> {
    get_missing_session_cwd_issue(session_manager, fallback_cwd)
        .map_or(Ok(()), |issue| Err(MissingSessionCwdError::new(issue)))
}

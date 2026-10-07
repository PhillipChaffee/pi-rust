//! The coding-agent session manager, upstream's `src/core/session-manager.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Manages conversation sessions as append-only trees stored in JSONL files.
//! Each entry has an id and parentId forming a tree; the leaf pointer tracks
//! the current position, appending creates a child of the leaf, and branching
//! moves the leaf to an earlier entry without modifying history.
//! [`build_session_context`] resolves the compaction-aware message list for
//! the LLM along the root-to-leaf path.
//!
//! The wire format is upstream's v3 session tree (the same shape
//! `pi-agent-core`'s JSONL backend reads as legacy-v3 input), so the header
//! codec and ISO-8601 helpers ride the substrate rather than a re-port, and
//! the sessions-root discovery reuses the #118 cwd encoding.
//!
//! Restatements: upstream's untyped entry objects tolerated arbitrary field
//! values; the typed port degrades unparseable entries to raw values that
//! still ride the file, the tree, and the export but cannot participate in
//! typed projections. Messages whose `content` is null or missing are
//! repaired at parse time to an empty block list (upstream repairs them in
//! the context projection; the typed message layer cannot represent null
//! content). `Date.parse`'s lenient non-ISO timestamps parse as absent.

pub mod appends;
pub mod branched;
pub mod context;
pub mod discovery;
pub mod entries;
pub mod file_entry;
pub mod lifecycle;
pub mod manager;
pub mod migrate;
pub mod parse;
pub mod tree;
pub mod types;

pub use context::{
    ByIdIndex, LeafId, SessionContext, SessionModel, build_context_entries, build_session_context,
    build_session_path, session_entry_to_context_messages,
};
pub use discovery::{
    SessionInfo, SessionListProgress, find_most_recent_session, get_default_session_dir,
};
pub use entries::{
    BranchSummaryEntry, CompactionEntry, CustomEntry, CustomMessageEntry, LabelEntry, MessageEntry,
    ModelChangeEntry, SessionEntry, SessionEntryBase, SessionInfoEntry, ThinkingLevelChangeEntry,
};
pub use file_entry::{FileEntry, ReadSessionHeaderError, SessionManagerError};
pub use manager::SessionManager;
pub use migrate::{get_latest_compaction_entry, migrate_session_entries, parse_session_entries};
pub use parse::{load_entries_from_file, read_session_header};
pub use tree::SessionTreeNode;
pub use types::{
    CURRENT_SESSION_VERSION, MAX_SESSION_HEADER_SCAN_BYTES, NewSessionOptions, SessionHeader,
};

use std::fs;
use std::path::Path;

use regex::Regex;

use pi_agent_core::harness::session::jsonl::codec::format_iso8601;

use crate::config::home_dir;
use crate::utils::paths::{PathInputOptions, normalize_path, resolve_path};

pub(crate) use migrate::generate_id;

/// Validate a session id, upstream's `assertValidSessionId`: non-empty,
/// alphanumeric with interior `.`, `-`, `_`, starting and ending
/// alphanumeric.
///
/// # Errors
/// [`SessionManagerError::InvalidSessionId`] when the id does not match,
/// instead of the upstream throw.
pub fn assert_valid_session_id(id: &str) -> Result<(), SessionManagerError> {
    let bytes = id.as_bytes();
    let middle_valid = bytes.len() <= 2
        || bytes[1..bytes.len() - 1]
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'));
    let valid = !bytes.is_empty()
        && bytes.first().is_some_and(u8::is_ascii_alphanumeric)
        && bytes.last().is_some_and(u8::is_ascii_alphanumeric)
        && middle_valid;
    if valid {
        Ok(())
    } else {
        Err(SessionManagerError::InvalidSessionId)
    }
}

/// The current Unix epoch milliseconds, upstream's `Date.now()`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "millisecond counts since the epoch fit i64 for any system clock this code will run on"
)]
pub(crate) fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_millis() as i64)
}

/// The current UTC timestamp, upstream's `new Date().toISOString()`.
pub(crate) fn now_iso8601() -> String {
    format_iso8601(now_millis())
}

/// The session filename's timestamp half, upstream's
/// `timestamp.replace(/[:.]/g, "-")`.
pub(crate) fn file_timestamp(timestamp: &str) -> String {
    timestamp.replace([':', '.'], "-")
}

/// Sanitize a session display name, upstream's
/// `name.replace(/[\r\n]+/g, " ").trim()`.
pub(crate) fn sanitize_session_name(name: &str) -> String {
    static NEWLINES: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        #[expect(
            clippy::expect_used,
            reason = "the newline class is a constant pattern; a compile failure here is a build bug, not a runtime condition"
        )]
        Regex::new(r"[\r\n]+").expect("static regex")
    });
    NEWLINES.replace_all(name, " ").trim().to_owned()
}

/// Append one JSON line to a file, upstream's `appendFileSync`.
pub(crate) fn append_line(path: &str, line: &str) -> Result<(), SessionManagerError> {
    use std::io::Write as _;
    let mut file = fs::OpenOptions::new().append(true).open(path)?;
    writeln!(file, "{line}")?;
    Ok(())
}

/// Resolve a path against the process cwd, upstream's `resolvePath` calls.
pub(crate) fn resolve_here(input: &str) -> String {
    resolve_path(input, &crate::config::process_cwd(), &home_dir())
}

/// Normalize a session directory or file path, upstream's `normalizePath`
/// calls, mapped onto the manager error taxonomy.
pub(crate) fn normalize_here(input: &str) -> Result<String, SessionManagerError> {
    normalize_path(input, &PathInputOptions::default())
        .map_err(|error| SessionManagerError::PathNormalize(error.to_string()))
}

/// Whether a path exists, upstream's `existsSync`.
pub(crate) fn path_exists(path: &str) -> bool {
    Path::new(path).exists()
}

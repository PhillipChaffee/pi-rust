//! The file-entry wrapper (header, typed entry, or raw value), the
//! entry accessors, and the manager error taxonomy.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use super::entries::SessionEntry;
use super::types::SessionHeader;

/// One parsed file line, upstream's `FileEntry` (`SessionHeader |
/// SessionEntry`).
///
/// The third variant carries the restatement: upstream's untyped parse kept
/// any JSON object a line produced, so a line outside the typed vocabulary —
/// an unknown type, or a known type with a malformed required field — still
/// rides the file, the tree, and every export. Such values cannot
/// participate in typed projections and are skipped where an id or parent
/// link would be needed. The serde pair is hand-written: a derived untagged
/// impl would match entry lines against the header shape (whose flattened
/// extras absorb unknown fields).
#[expect(
    clippy::large_enum_variant,
    reason = "the header variant mirrors upstream's FileEntry union; boxing would allocate on every line parse"
)]
#[derive(Clone, Debug, PartialEq)]
pub enum FileEntry {
    /// A session header, wire `"type": "session"`.
    Session(SessionHeader),
    /// A typed session tree entry.
    Entry(SessionEntry),
    /// Any other parseable line, preserved verbatim.
    Other(JsonValue),
}

impl Serialize for FileEntry {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Session(header) => header.serialize(serializer),
            Self::Entry(entry) => entry.serialize(serializer),
            Self::Other(value) => value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for FileEntry {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(super::parse::parse_file_entry(JsonValue::deserialize(
            deserializer,
        )?))
    }
}

impl FileEntry {
    /// The entry id, present for typed entries and raw values that carry a
    /// string `id` field.
    #[must_use]
    pub fn entry_id(&self) -> Option<&str> {
        match self {
            Self::Session(header) => Some(&header.id),
            Self::Entry(entry) => entry.base().id.as_deref(),
            Self::Other(value) => value.get("id").and_then(JsonValue::as_str),
        }
    }

    /// The parent id, `None` at a root; absent for raw values without one.
    #[must_use]
    pub fn entry_parent_id(&self) -> Option<&str> {
        match self {
            Self::Session(_) => None,
            Self::Entry(entry) => entry.base().parent_id.as_deref(),
            Self::Other(value) => value.get("parentId").and_then(JsonValue::as_str),
        }
    }

    /// The entry timestamp the tree sorts on, upstream's
    /// `new Date(entry.timestamp).getTime()`.
    #[must_use]
    pub fn entry_timestamp(&self) -> String {
        match self {
            Self::Session(header) => header.timestamp.clone(),
            Self::Entry(entry) => entry.base().timestamp.clone(),
            Self::Other(value) => value
                .get("timestamp")
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
                .to_owned(),
        }
    }
}

impl SessionEntry {
    /// The entry's shared base fields.
    #[must_use]
    pub const fn base(&self) -> &super::entries::SessionEntryBase {
        match self {
            Self::Message(e) => &e.base,
            Self::ThinkingLevelChange(e) => &e.base,
            Self::ModelChange(e) => &e.base,
            Self::Compaction(e) => &e.base,
            Self::BranchSummary(e) => &e.base,
            Self::Custom(e) => &e.base,
            Self::Label(e) => &e.base,
            Self::SessionInfo(e) => &e.base,
            Self::CustomMessage(e) => &e.base,
        }
    }

    /// The entry's shared base fields, mutably.
    #[expect(
        clippy::missing_const_for_fn,
        reason = "const fn cannot take a `&mut self` receiver on stable Rust; the lint cannot see that"
    )]
    pub fn base_mut(&mut self) -> &mut super::entries::SessionEntryBase {
        match self {
            Self::Message(e) => &mut e.base,
            Self::ThinkingLevelChange(e) => &mut e.base,
            Self::ModelChange(e) => &mut e.base,
            Self::Compaction(e) => &mut e.base,
            Self::BranchSummary(e) => &mut e.base,
            Self::Custom(e) => &mut e.base,
            Self::Label(e) => &mut e.base,
            Self::SessionInfo(e) => &mut e.base,
            Self::CustomMessage(e) => &mut e.base,
        }
    }

    /// The entry's wire type discriminator.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Message(_) => "message",
            Self::ThinkingLevelChange(_) => "thinking_level_change",
            Self::ModelChange(_) => "model_change",
            Self::Compaction(_) => "compaction",
            Self::BranchSummary(_) => "branch_summary",
            Self::Custom(_) => "custom",
            Self::Label(_) => "label",
            Self::SessionInfo(_) => "session_info",
            Self::CustomMessage(_) => "custom_message",
        }
    }
}

/// Why a session-manager operation failed; each message is upstream's
/// `Error` text verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionManagerError {
    /// `assertValidSessionId`'s rejection.
    InvalidSessionId,
    /// The opened file parsed but is not a pi session.
    InvalidSessionFile(String),
    /// A tree operation targeted a missing entry.
    EntryNotFound(String),
    /// A fork source was empty, invalid, or headerless.
    ForkSourceInvalid(String),
    /// A fork source had no header.
    ForkSourceHeaderless(String),
    /// A path normalization failed, upstream's `normalizePath` throw.
    PathNormalize(String),
    /// A filesystem operation failed.
    Io(String),
}

impl std::fmt::Display for SessionManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSessionId => f.write_str(
                "Session id must be non-empty, contain only alphanumeric characters, '-', '_', and '.', and start and end with an alphanumeric character",
            ),
            Self::InvalidSessionFile(path) => {
                write!(f, "Session file is not a valid pi session: {path}")
            }
            Self::EntryNotFound(id) => write!(f, "Entry {id} not found"),
            Self::ForkSourceInvalid(path) => {
                write!(f, "Cannot fork: source session file is empty or invalid: {path}")
            }
            Self::ForkSourceHeaderless(path) => write!(f, "Cannot fork: source session has no header: {path}"),
            Self::PathNormalize(message) | Self::Io(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for SessionManagerError {}

impl From<std::io::Error> for SessionManagerError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

/// Why reading a session header failed, upstream's
/// `SessionHeaderScanLimitError` plus the IO errors it propagates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadSessionHeaderError {
    /// The scan exceeded [`super::types::MAX_SESSION_HEADER_SCAN_BYTES`].
    ScanLimit(String),
    /// The file could not be read.
    Io(String),
}

impl std::fmt::Display for ReadSessionHeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScanLimit(path) => write!(
                f,
                "Session header exceeds {}-byte scan limit: {path}",
                super::types::MAX_SESSION_HEADER_SCAN_BYTES
            ),
            Self::Io(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ReadSessionHeaderError {}

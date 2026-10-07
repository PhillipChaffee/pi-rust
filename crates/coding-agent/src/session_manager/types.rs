//! The session file vocabulary, upstream's `src/core/session-manager.ts`
//! type surface: the header, the tree entry union, the file-entry wrapper,
//! and the manager error taxonomy.
//!
//! Upstream pin: `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use serde::{Deserialize, Serialize};

/// The JSON map shape the extras flatten rides, upstream's untyped object
/// remainder.
type JsonMap = serde_json::Map<String, serde_json::Value>;

/// The session format version this manager writes, upstream's
/// `CURRENT_SESSION_VERSION`.
pub const CURRENT_SESSION_VERSION: i64 = 3;

/// The byte budget the synchronous header discovery scans, upstream's
/// `MAX_SESSION_HEADER_SCAN_BYTES`: large enough for big cwd and custom
/// metadata fields, small enough to bound discovery work.
pub const MAX_SESSION_HEADER_SCAN_BYTES: u64 = 1024 * 1024;

/// The read chunk the bounded header scan consumes, upstream's
/// `SESSION_HEADER_READ_BUFFER_SIZE`.
pub(crate) const SESSION_HEADER_READ_BUFFER_SIZE: usize = 4096;

/// Options for a new session, upstream's `NewSessionOptions`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NewSessionOptions {
    /// An explicit session id; generated when absent.
    pub id: Option<String>,
    /// The parent session file when the new session forks one.
    pub parent_session: Option<String>,
}

/// The header line's wire shape, upstream's `SessionHeader` (`"type":
/// "session"`).
///
/// `version` is absent on v1 sessions; the other optional fields are absent
/// when a session does not carry them. The `"type"` field rides the
/// hand-written serde pair below so round-trips stay byte-stable.
#[expect(
    clippy::derive_partial_eq_without_eq,
    reason = "the extras flatten rides serde_json::Value, which is PartialEq-only; Eq would bar the round-trip fidelity"
)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionHeader {
    /// The format version; absent on v1 sessions, 3 on current ones.
    pub version: Option<i64>,
    /// The session id.
    pub id: String,
    /// The creation timestamp, ISO-8601 `Z` form.
    pub timestamp: String,
    /// The working directory the session started in; absent on headers that
    /// never recorded one.
    pub cwd: Option<String>,
    /// The parent session file when this session was forked or branched.
    pub parent_session: Option<String>,
    /// Fields outside the typed header, preserved verbatim through reloads
    /// and rewrites.
    pub extras: JsonMap,
}

impl SessionHeader {
    /// The header's cwd, upstream's `getSessionHeaderCwd`: present only when
    /// the field carried a string.
    #[must_use]
    pub fn cwd(&self) -> Option<&str> {
        self.cwd.as_deref()
    }
}

impl Serialize for SessionHeader {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap as _;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("type", "session")?;
        if let Some(version) = &self.version {
            map.serialize_entry("version", version)?;
        }
        map.serialize_entry("id", &self.id)?;
        if !self.timestamp.is_empty() {
            map.serialize_entry("timestamp", &self.timestamp)?;
        }
        if let Some(cwd) = &self.cwd {
            map.serialize_entry("cwd", cwd)?;
        }
        if let Some(parent_session) = &self.parent_session {
            map.serialize_entry("parentSession", parent_session)?;
        }
        for (key, value) in &self.extras {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for SessionHeader {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut map = JsonMap::deserialize(deserializer)?;
        map.remove("type");
        let version = match map.remove("version") {
            Some(value) => {
                Some(serde_json::from_value::<i64>(value).map_err(serde::de::Error::custom)?)
            }
            None => None,
        };
        let id = match map.remove("id") {
            Some(value) => {
                serde_json::from_value::<String>(value).map_err(serde::de::Error::custom)?
            }
            None => return Err(serde::de::Error::missing_field("id")),
        };
        let timestamp = match map.remove("timestamp") {
            Some(value) if !value.is_null() => {
                serde_json::from_value::<String>(value).map_err(serde::de::Error::custom)?
            }
            _ => String::new(),
        };
        let cwd = match map.remove("cwd") {
            Some(value) if !value.is_null() => {
                Some(serde_json::from_value::<String>(value).map_err(serde::de::Error::custom)?)
            }
            _ => None,
        };
        let parent_session = match map.remove("parentSession") {
            Some(value) if !value.is_null() => {
                Some(serde_json::from_value::<String>(value).map_err(serde::de::Error::custom)?)
            }
            _ => None,
        };
        Ok(Self {
            version,
            id,
            timestamp,
            cwd,
            parent_session,
            extras: map,
        })
    }
}

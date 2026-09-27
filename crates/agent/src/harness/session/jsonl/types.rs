//! The JSONL backend's format types, ported from upstream
//! `src/harness/session/jsonl/types.ts`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::harness::session::memory::NowFn;
use crate::harness::session::types::SessionMetadata;
use crate::harness::types::FileSystem;

/// The on-disk JSONL format's version, upstream's `JSONL_FORMAT_VERSION`.
pub const JSONL_FORMAT_VERSION: u32 = 4;

/// The durable state machine's version the format 4 files carry, upstream's
/// `JSONL_STORAGE_VERSION`.
pub const JSONL_STORAGE_VERSION: u32 = 1;

/// The file's first line, upstream's `JsonlStorageHeader`; the wire carries
/// `v`/`kind` discriminants and camelCase fields.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonlStorageHeader {
    /// The format version, wire `"v"`.
    pub v: u32,
    /// The record kind, wire `"kind": "header"`.
    pub kind: String,
    /// The session id.
    pub id: String,
    /// The durable state machine's version, wire `"storageVersion"`.
    pub storage_version: u32,
    /// Unix epoch milliseconds when the session was created, wire
    /// `"createdAt"`.
    pub created_at: i64,
    /// The session's working directory.
    pub cwd: String,
    /// The parent session id, when forked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    /// The legacy parent session path, when forked from a v3 ancestor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_parent_session_path: Option<String>,
    /// The sequence high-water mark snapshot rewrites record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_seq: Option<u64>,
}

/// The options one JSONL storage opens with, upstream's
/// `JsonlStorageOptions`.
#[derive(Clone)]
pub struct JsonlStorageOptions {
    /// The filesystem capability the storage reads and appends through.
    pub file_system: Arc<dyn FileSystem>,
    /// The addressed file path.
    pub path: String,
    /// The injected clock; the wall clock when omitted.
    pub now: Option<NowFn>,
}

impl std::fmt::Debug for JsonlStorageOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonlStorageOptions")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

/// The repo's session metadata, upstream's `JsonlSessionMetadata`.
///
/// [`SessionMetadata`] with a required `cwd` plus the file location and its
/// modification time. The erased session contract carries the base fields;
/// the repo's typed lifecycle carries the full record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonlSessionMetadata {
    /// The session id.
    pub id: String,
    /// Unix epoch milliseconds when the session was created.
    pub created_at: i64,
    /// The durable state machine's version.
    pub storage_version: u32,
    /// The session's working directory; required for file-backed sessions.
    pub cwd: String,
    /// The session file's addressed path, wire `"path"`.
    pub path: String,
    /// The file's modification time as Unix epoch milliseconds, wire
    /// `"modifiedAt"`.
    pub modified_at: i64,
    /// The parent session id, when forked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    /// The legacy parent session path, when the parent could not be
    /// resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_parent_session_path: Option<String>,
}

impl JsonlSessionMetadata {
    /// The erased base the session contract carries, upstream's
    /// `TMetadata extends SessionMetadata` upcast.
    #[must_use]
    pub fn base(&self) -> SessionMetadata {
        SessionMetadata {
            id: self.id.clone(),
            created_at: self.created_at,
            storage_version: self.storage_version,
            cwd: Some(self.cwd.clone()),
            parent_session_id: self.parent_session_id.clone(),
            legacy_parent_session_path: self.legacy_parent_session_path.clone(),
        }
    }
}

/// The create options one file-backed session accepts, upstream's
/// `JsonlSessionCreateOptions`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonlSessionCreateOptions {
    /// The requested session id; a uuidv7 is generated when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The parent session id, when the session descends from a fork.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    /// The session's working directory; the discovery scope the repo keys
    /// sessions under.
    pub cwd: String,
}

/// The list options one discovery accepts, upstream's
/// `JsonlSessionListOptions`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonlSessionListOptions {
    /// Restrict discovery to one cwd.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}

/// The repo's options, upstream's `JsonlSessionRepoOptions`.
#[derive(Clone)]
pub struct JsonlSessionRepoOptions {
    /// The filesystem capability discovery and lifecycle run through.
    pub file_system: Arc<dyn FileSystem>,
    /// The sessions root; each cwd gets one encoded directory under it.
    pub sessions_root: String,
    /// The injected clock; the wall clock when omitted.
    pub now: Option<NowFn>,
}

impl std::fmt::Debug for JsonlSessionRepoOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonlSessionRepoOptions")
            .field("sessions_root", &self.sessions_root)
            .finish_non_exhaustive()
    }
}

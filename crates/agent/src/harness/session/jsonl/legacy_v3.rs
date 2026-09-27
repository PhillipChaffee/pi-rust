//! The legacy v3 JSONL migration, ported from upstream
//! `src/harness/session/jsonl/legacy-v3.ts`.
//!
//! A captured legacy file exposes itself as repeatable logical v4 writes:
//! the scan builds structural indexes (reminted ids stable for the source
//! instance), derives current values and imported usage, and every
//! [`LegacyV3Source::writes`] pass reopens the path to materialize captured
//! records and resolve payload-specific references. Callers must not
//! replace or edit the source between passes. Upstream's async generators
//! restate as eager collections: each pass still reads the file exactly
//! once, so the observed read counts are unchanged.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use pi_ai::types::Usage;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};

use crate::harness::context::Context;
use crate::harness::session::commit::CommittedWrite;
use crate::harness::session::jsonl::codec::{
    LegacyV3SessionHeader, parse_iso8601_millis, parse_jsonl_session_header,
};
use crate::harness::session::jsonl::io::{JsonlError, file_value, read_jsonl_header};
use crate::harness::session::jsonl::types::{
    JSONL_FORMAT_VERSION, JSONL_STORAGE_VERSION, JsonlSessionMetadata, JsonlStorageHeader,
};
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::values::ValueSetWrite;
use crate::harness::types::{FileSystem, TextLineReader};
use crate::harness::utils::usage::{add_usage, empty_usage};
use crate::types::ThinkingLevel;

/// One retained record's structural row, upstream's
/// `Pick<CommittedEntryWrite, "id" | "parentId" | "seq">`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacyV3EntryRow {
    /// The reminted entry id.
    pub id: String,
    /// The resolved parent id; `None` at the root.
    pub parent_id: Option<String>,
    /// The imported sequence.
    pub seq: u64,
}

/// The retained v3 record types' shared shape, upstream's
/// `LegacyV3EntryBase`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LegacyV3EntryBase {
    /// The record's legacy id.
    pub id: String,
    /// The parent's legacy id; `None` at the root.
    pub parent_id: Option<String>,
    /// The record's ISO-8601 timestamp.
    pub timestamp: String,
}

/// One legacy v3 record, upstream's `LegacyV3Entry` union over `"type"`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum LegacyV3Entry {
    /// A transcript message, wire `"type": "message"`.
    #[serde(rename = "message")]
    Message {
        /// The base fields, flattened.
        #[serde(flatten)]
        base: LegacyV3EntryBase,
        /// The message payload.
        message: Box<crate::types::AgentMessage>,
    },
    /// An application-defined entry, wire `"type": "custom"`.
    #[serde(rename = "custom")]
    Custom {
        /// The base fields, flattened.
        #[serde(flatten)]
        base: LegacyV3EntryBase,
        /// The application's custom type discriminator.
        custom_type: String,
        /// The application-defined payload.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<JsonValue>,
    },
    /// A custom-role message record, wire `"type": "custom_message"`.
    #[serde(rename = "custom_message")]
    CustomMessage {
        /// The base fields, flattened.
        #[serde(flatten)]
        base: LegacyV3EntryBase,
        /// The message's custom type discriminator.
        custom_type: String,
        /// The message content, a plain string or content blocks.
        content: JsonValue,
        /// Implementation-specific details.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<JsonValue>,
        /// Whether the message displays in the transcript.
        display: bool,
    },
    /// A branch summary, wire `"type": "branch_summary"`.
    #[serde(rename = "branch_summary")]
    BranchSummary {
        /// The base fields, flattened.
        #[serde(flatten)]
        base: LegacyV3EntryBase,
        /// The summarized branch's source entry id.
        from_id: String,
        /// The generated summary.
        summary: String,
        /// Implementation-specific details.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<JsonValue>,
        /// The summarization model call's usage.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        /// Whether a navigation hook produced the entry.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },
    /// A compaction, wire `"type": "compaction"`.
    #[serde(rename = "compaction")]
    Compaction {
        /// The base fields, flattened.
        #[serde(flatten)]
        base: LegacyV3EntryBase,
        /// The generated summary.
        summary: String,
        /// The oldest entry the compaction retains, wire
        /// `"firstKeptEntryId"`.
        first_kept_entry_id: String,
        /// Estimated context tokens before compaction, wire
        /// `"tokensBefore"`.
        tokens_before: i64,
        /// Implementation-specific details.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<JsonValue>,
        /// The summarization model call's usage.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        /// Whether a compaction hook produced the entry.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },
    /// A model change, wire `"type": "model_change"`.
    #[serde(rename = "model_change")]
    ModelChange {
        /// The base fields, flattened.
        #[serde(flatten)]
        base: LegacyV3EntryBase,
        /// The new provider id.
        provider: String,
        /// The new model id.
        model_id: String,
    },
    /// A thinking-level change, wire `"type": "thinking_level_change"`.
    #[serde(rename = "thinking_level_change")]
    ThinkingLevelChange {
        /// The base fields, flattened.
        #[serde(flatten)]
        base: LegacyV3EntryBase,
        /// The new thinking level.
        thinking_level: ThinkingLevel,
    },
    /// An active-tools change, wire `"type": "active_tools_change"`.
    #[serde(rename = "active_tools_change")]
    ActiveToolsChange {
        /// The base fields, flattened.
        #[serde(flatten)]
        base: LegacyV3EntryBase,
        /// The newly active tool names.
        active_tool_names: Vec<String>,
    },
    /// A session-info record, wire `"type": "session_info"`.
    #[serde(rename = "session_info")]
    SessionInfo {
        /// The base fields, flattened.
        #[serde(flatten)]
        base: LegacyV3EntryBase,
        /// The session name; its absence clears the name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// A label record, wire `"type": "label"`.
    #[serde(rename = "label")]
    Label {
        /// The base fields, flattened.
        #[serde(flatten)]
        base: LegacyV3EntryBase,
        /// The labeled entry's legacy id.
        target_id: String,
        /// The label; its absence clears the target's label.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
}

impl LegacyV3Entry {
    /// The record's shared base fields.
    const fn base(&self) -> &LegacyV3EntryBase {
        match self {
            Self::Message { base, .. }
            | Self::Custom { base, .. }
            | Self::CustomMessage { base, .. }
            | Self::BranchSummary { base, .. }
            | Self::Compaction { base, .. }
            | Self::ModelChange { base, .. }
            | Self::ThinkingLevelChange { base, .. }
            | Self::ActiveToolsChange { base, .. }
            | Self::SessionInfo { base, .. }
            | Self::Label { base, .. } => base,
        }
    }

    /// The record's legacy id.
    fn id(&self) -> &str {
        &self.base().id
    }

    /// The record's parent legacy id.
    fn parent_id(&self) -> Option<&str> {
        self.base().parent_id.as_deref()
    }

    /// The record's ISO-8601 timestamp.
    fn timestamp(&self) -> &str {
        &self.base().timestamp
    }

    /// The wire type discriminant, upstream's `entry.type`.
    #[must_use]
    pub const fn record_type(&self) -> &'static str {
        match self {
            Self::Message { .. } => "message",
            Self::Custom { .. } => "custom",
            Self::CustomMessage { .. } => "custom_message",
            Self::BranchSummary { .. } => "branch_summary",
            Self::Compaction { .. } => "compaction",
            Self::ModelChange { .. } => "model_change",
            Self::ThinkingLevelChange { .. } => "thinking_level_change",
            Self::ActiveToolsChange { .. } => "active_tools_change",
            Self::SessionInfo { .. } => "session_info",
            Self::Label { .. } => "label",
        }
    }

    /// Whether the record's payload survives migration, upstream's
    /// `isRetainedEntry`: keep structure, labels, and configuration
    /// changes, but not conversation payloads.
    #[must_use]
    pub const fn is_retained(&self) -> bool {
        !matches!(
            self,
            Self::ModelChange { .. }
                | Self::ThinkingLevelChange { .. }
                | Self::ActiveToolsChange { .. }
                | Self::SessionInfo { .. }
                | Self::Label { .. }
        )
    }
}

/// The retained index entry's payload kind, upstream's
/// `RetainedLegacyV3IndexEntry`'s inner union.
#[derive(Clone, Debug)]
pub enum RetainedKind {
    /// A message record.
    Message,
    /// A custom-message record.
    CustomMessage,
    /// A custom entry.
    Custom,
    /// A branch summary carrying its source id.
    BranchSummary {
        /// The summarized branch's source entry id.
        from_id: String,
    },
    /// A compaction carrying its kept boundary.
    Compaction {
        /// The compaction's first kept entry id.
        first_kept_entry_id: String,
    },
}

/// The discarded index entry's payload kind, upstream's
/// `LegacyV3IndexEntry`'s discarded shapes: keep the mapping, not the
/// payload.
#[derive(Clone, Debug)]
pub enum DiscardedKind {
    /// A label record.
    Label {
        /// The labeled entry's legacy id.
        target_id: String,
        /// The label value.
        label: Option<String>,
    },
    /// A model change.
    ModelChange {
        /// The new provider id.
        provider: String,
        /// The new model id.
        model_id: String,
    },
    /// A thinking-level change.
    ThinkingLevelChange {
        /// The new thinking level.
        thinking_level: ThinkingLevel,
    },
    /// An active-tools change.
    ActiveToolsChange {
        /// The newly active tool names.
        active_tool_names: Vec<String>,
    },
    /// A session-info record.
    SessionInfo,
}

/// The structural index of one v3 record, upstream's `LegacyV3IndexEntry`.
#[derive(Clone, Debug)]
pub enum LegacyV3IndexEntry {
    /// A retained record with its reminted id and sequence.
    Retained {
        /// The record's legacy id.
        id: String,
        /// The record's parent legacy id.
        parent_id: Option<String>,
        /// The reminted id, upstream's `mappedId`.
        mapped_id: String,
        /// The imported sequence, upstream's `seq`.
        seq: u64,
        /// The payload kind.
        kind: RetainedKind,
    },
    /// A discarded record mapping to its nearest retained ancestor.
    Discarded {
        /// The record's legacy id.
        id: String,
        /// The record's parent legacy id.
        parent_id: Option<String>,
        /// The parent's mapped id, upstream's `mappedId`; `None` when the
        /// discarded record has no retained ancestor.
        mapped_id: Option<String>,
        /// The payload kind.
        kind: DiscardedKind,
    },
}

impl LegacyV3IndexEntry {
    /// The record's legacy id.
    fn id(&self) -> &str {
        match self {
            Self::Retained { id, .. } | Self::Discarded { id, .. } => id,
        }
    }

    /// The record's parent legacy id.
    fn parent_id(&self) -> Option<&str> {
        match self {
            Self::Retained { parent_id, .. } | Self::Discarded { parent_id, .. } => {
                parent_id.as_deref()
            }
        }
    }

    /// The mapped id: the reminted id for retained records, the parent's
    /// mapped id for discarded ones.
    fn mapped_id(&self) -> Option<&str> {
        match self {
            Self::Retained { mapped_id, .. } => Some(mapped_id),
            Self::Discarded { mapped_id, .. } => mapped_id.as_deref(),
        }
    }

    /// Whether the index entry retains its payload, upstream's
    /// `isRetainedEntry` over the index shape.
    #[must_use]
    pub const fn is_retained(&self) -> bool {
        matches!(self, Self::Retained { .. })
    }
}

/// The scan's inventory, upstream's `LegacyV3Inventory`.
struct LegacyV3Inventory {
    entries: BTreeMap<String, LegacyV3IndexEntry>,
    /// The records' file order, upstream's Map insertion order.
    order: Vec<String>,
    imported_usage: Usage,
    name: Option<String>,
    final_id: Option<String>,
    next_seq: u64,
}

/// The metadata fields the v3 header resolves to before the file location
/// attaches, upstream's `Omit<JsonlSessionMetadata, "path" | "modifiedAt">`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonlSessionMetadataBase {
    /// The session id.
    pub id: String,
    /// Unix epoch milliseconds when the session was created.
    pub created_at: i64,
    /// The durable state machine's version.
    pub storage_version: u32,
    /// The session's working directory.
    pub cwd: String,
    /// The resolved parent session id, when the parent header resolves.
    pub parent_session_id: Option<String>,
    /// The parent session's path, when it does not resolve.
    pub legacy_parent_session_path: Option<String>,
}

/// Resolves a legacy parent session path to its session id, upstream's
/// `resolveLegacyV3ParentSessionId`; unreadable or unrecognized parents
/// resolve to `None`.
async fn resolve_legacy_v3_parent_session_id(
    file_system: &dyn FileSystem,
    parent_session_path: &str,
    context: &Context,
) -> Option<String> {
    let lines = file_system
        .read_text_lines(parent_session_path, None, context)
        .await
        .ok()?;
    lines.first().and_then(|line| {
        parse_jsonl_session_header(line)
            .ok()
            .map(|parsed| match parsed {
                crate::harness::session::jsonl::codec::JsonlParsedSessionHeader::V4(header) => {
                    header.id
                }
                crate::harness::session::jsonl::codec::JsonlParsedSessionHeader::V3Legacy(
                    header,
                ) => header.id,
            })
    })
}

/// Builds the metadata fields one v3 header carries, upstream's
/// `metadataFromLegacyV3Header`.
///
/// # Errors
/// Never: unreadable parents degrade to the legacy path.
pub async fn metadata_from_legacy_v3_header(
    file_system: &dyn FileSystem,
    header: &LegacyV3SessionHeader,
    context: &Context,
) -> JsonlSessionMetadataBase {
    let mut metadata = JsonlSessionMetadataBase {
        id: header.id.clone(),
        created_at: parse_iso8601_millis(&header.timestamp).unwrap_or_default(),
        storage_version: JSONL_STORAGE_VERSION,
        cwd: header.cwd.clone(),
        parent_session_id: None,
        legacy_parent_session_path: None,
    };
    if let Some(parent_session) = &header.parent_session {
        metadata.parent_session_id =
            resolve_legacy_v3_parent_session_id(file_system, parent_session, context).await;
        if metadata.parent_session_id.is_none() {
            metadata.legacy_parent_session_path = Some(parent_session.clone());
        }
    }
    metadata
}

/// Builds the format-4 header one v3 file normalizes to, upstream's
/// `normalizeLegacyV3Header`.
pub async fn normalize_legacy_v3_header(
    file_system: &dyn FileSystem,
    header: &LegacyV3SessionHeader,
    context: &Context,
) -> JsonlStorageHeader {
    let base = metadata_from_legacy_v3_header(file_system, header, context).await;
    JsonlStorageHeader {
        v: JSONL_FORMAT_VERSION,
        kind: "header".to_owned(),
        id: base.id,
        storage_version: base.storage_version,
        created_at: base.created_at,
        cwd: base.cwd,
        parent_session_id: base.parent_session_id,
        legacy_parent_session_path: base.legacy_parent_session_path,
        next_seq: None,
    }
}

/// Parses one v3 record line, upstream's `parseLegacyV3Entry`: the record
/// type whitelist is the only validation.
///
/// # Errors
/// A [`JsonlError`] for invalid JSON and unsupported record types.
pub fn parse_legacy_v3_entry(line: &str) -> Result<LegacyV3Entry, JsonlError> {
    let value: JsonValue = serde_json::from_str(line)
        .map_err(|_| JsonlError("Invalid legacy v3 JSONL record: not valid JSON".to_owned()))?;
    serde_json::from_value(value.clone()).map_err(|_| {
        JsonlError(format!(
            "Unsupported legacy v3 record type: {}",
            value
                .as_object()
                .and_then(|record| record.get("type"))
                .and_then(JsonValue::as_str)
                .unwrap_or("undefined")
        ))
    })
}

/// The custom-role message one `custom_message` record imports, upstream's
/// `importedCustomMessage`: the fields carry over verbatim with the ISO
/// timestamp parsed to millis.
#[must_use]
pub fn imported_custom_message(entry: &LegacyV3Entry) -> crate::types::AgentMessage {
    let LegacyV3Entry::CustomMessage {
        base,
        custom_type,
        content,
        details,
        display,
    } = entry
    else {
        unreachable_custom_message();
    };
    let mut data = serde_json::Map::new();
    data.insert("customType".to_owned(), json!(custom_type));
    data.insert("content".to_owned(), content.clone());
    if let Some(details) = details {
        data.insert("details".to_owned(), details.clone());
    }
    data.insert("display".to_owned(), json!(display));
    crate::types::AgentMessage::Custom(crate::types::CustomAgentMessage {
        role: "custom".to_owned(),
        timestamp: parse_iso8601_millis(&base.timestamp).unwrap_or_default(),
        data,
    })
}

#[expect(
    clippy::panic,
    reason = "the caller matches on the custom-message variant first; the fallback is unreachable by construction"
)]
fn unreachable_custom_message() -> ! {
    panic!("imported_custom_message called on a non-custom-message record")
}

/// Resolves a legacy id to its imported id, upstream's `ResolveLegacyId`;
/// discarded records resolve to their nearest retained ancestor's minted
/// id.
fn create_legacy_id_resolver(
    entries: &BTreeMap<String, LegacyV3IndexEntry>,
) -> impl Fn(&str) -> Result<Option<String>, JsonlError> + '_ {
    // A discarded root maps to null, upstream's `mappedId: null`; only a
    // missing entry is a reference failure.
    move |legacy_id: &str| {
        entries.get(legacy_id).map_or_else(
            || {
                Err(JsonlError(format!(
                    "Missing legacy v3 entry reference: {legacy_id}"
                )))
            },
            |entry| Ok(entry.mapped_id().map(str::to_owned)),
        )
    }
}

/// Resolves a branch summary's source id, upstream's
/// `resolveBranchSummaryFromId`: the legacy `branchWithSummary()` encoded a
/// root source as the `"root"` sentinel instead of null.
fn resolve_branch_summary_from_id(
    resolve_legacy_id: &impl Fn(&str) -> Result<Option<String>, JsonlError>,
    legacy_from_id: &str,
) -> Result<Option<String>, JsonlError> {
    if legacy_from_id == "root" {
        return Ok(None);
    }
    resolve_legacy_id(legacy_from_id)
}

/// The context message one record projects, upstream's `projectContextMessage`.
fn project_context_message(
    entry: &LegacyV3Entry,
    resolve_legacy_id: &impl Fn(&str) -> Result<Option<String>, JsonlError>,
) -> Result<Option<crate::types::AgentMessage>, JsonlError> {
    match entry {
        LegacyV3Entry::Message { message, .. } => Ok(Some(*message.clone())),
        LegacyV3Entry::CustomMessage { .. } => Ok(Some(imported_custom_message(entry))),
        LegacyV3Entry::BranchSummary {
            base,
            from_id,
            summary,
            ..
        } => {
            if summary.is_empty() {
                return Ok(None);
            }
            let from_id = resolve_branch_summary_from_id(resolve_legacy_id, from_id)?;
            Ok(Some(custom_role_message(
                "branchSummary",
                &json!({ "summary": summary, "fromId": from_id }),
                parse_iso8601_millis(&base.timestamp).unwrap_or_default(),
            )))
        }
        LegacyV3Entry::Compaction {
            base,
            summary,
            tokens_before,
            ..
        } => Ok(Some(custom_role_message(
            "compactionSummary",
            &json!({ "summary": summary, "tokensBefore": tokens_before }),
            parse_iso8601_millis(&base.timestamp).unwrap_or_default(),
        ))),
        LegacyV3Entry::Custom { .. }
        | LegacyV3Entry::ModelChange { .. }
        | LegacyV3Entry::ThinkingLevelChange { .. }
        | LegacyV3Entry::ActiveToolsChange { .. }
        | LegacyV3Entry::SessionInfo { .. }
        | LegacyV3Entry::Label { .. } => Ok(None),
    }
}

/// The custom-role message the summary projections produce, upstream's
/// `createBranchSummaryMessage`/`createCompactionSummaryMessage` shapes.
fn custom_role_message(role: &str, data: &JsonValue, timestamp: i64) -> crate::types::AgentMessage {
    crate::types::AgentMessage::Custom(crate::types::CustomAgentMessage {
        role: role.to_owned(),
        timestamp,
        data: data.as_object().cloned().unwrap_or_default(),
    })
}

/// The physical ancestry walk one compaction's tail follows, upstream's
/// `retainedTailStructure`: from the compaction's parent through
/// `firstKeptEntryId`, inclusive, including discarded nodes.
fn retained_tail_structure<'a>(
    compaction: (&'a str, Option<&'a str>, &'a str),
    entries_by_id: &'a BTreeMap<String, LegacyV3IndexEntry>,
) -> Result<Vec<&'a LegacyV3IndexEntry>, JsonlError> {
    let (compaction_id, mut current_id, first_kept_entry_id) = compaction;
    let mut walked: Vec<&LegacyV3IndexEntry> = Vec::new();
    while let Some(current) = current_id {
        let Some(entry) = entries_by_id.get(current) else {
            // The scan guarantees every parent exists earlier in the file.
            return Err(JsonlError(format!(
                "Legacy v3 compaction {compaction_id} firstKeptEntryId is not on its parent branch: {first_kept_entry_id}"
            )));
        };
        walked.push(entry);
        if current == first_kept_entry_id {
            return Ok(walked);
        }
        current_id = entry.parent_id();
    }
    Err(JsonlError(format!(
        "Legacy v3 compaction {compaction_id} firstKeptEntryId is not on its parent branch: {first_kept_entry_id}"
    )))
}

/// The entry write one retained record normalizes to, upstream's
/// `normalizeRetainedEntry`.
fn normalize_retained_entry(
    entry: &LegacyV3Entry,
    indexed: &LegacyV3IndexEntry,
    retained_tail: Vec<crate::types::AgentMessage>,
    resolve_legacy_id: &impl Fn(&str) -> Result<Option<String>, JsonlError>,
) -> Result<CommittedWrite, JsonlError> {
    let LegacyV3IndexEntry::Retained { mapped_id, seq, .. } = indexed else {
        unreachable_retained();
    };
    // A parent whose chain holds no retained ancestor reparents the record
    // to the root, upstream's resolved null.
    let parent_id = match indexed.parent_id() {
        Some(parent) => resolve_legacy_id(parent)?,
        None => None,
    };
    let timestamp = parse_iso8601_millis(entry.timestamp()).unwrap_or_default();
    let id = mapped_id.clone();
    match entry {
        LegacyV3Entry::Message { message, .. } => Ok(CommittedWrite::Entry {
            entry: crate::harness::session::types::Entry::Message {
                id,
                parent_id,
                seq: *seq,
                timestamp,
                body: Box::new(crate::harness::session::types::MessageEntry {
                    message: *message.clone(),
                    terminate: None,
                }),
            },
        }),
        LegacyV3Entry::CustomMessage { .. } => Ok(CommittedWrite::Entry {
            entry: crate::harness::session::types::Entry::Message {
                id,
                parent_id,
                seq: *seq,
                timestamp,
                body: Box::new(crate::harness::session::types::MessageEntry {
                    message: imported_custom_message(entry),
                    terminate: None,
                }),
            },
        }),
        LegacyV3Entry::BranchSummary {
            base: _,
            from_id,
            summary,
            details,
            usage,
            from_hook,
        } => Ok(CommittedWrite::Entry {
            entry: crate::harness::session::types::Entry::BranchSummary {
                id,
                parent_id,
                seq: *seq,
                timestamp,
                body: crate::harness::session::types::BranchSummaryEntryBody {
                    from_id: resolve_branch_summary_from_id(resolve_legacy_id, from_id)?,
                    summary: summary.clone(),
                    details: details.clone(),
                    usage: *usage,
                    from_hook: from_hook.unwrap_or(false),
                },
            },
        }),
        LegacyV3Entry::Compaction {
            summary,
            details,
            usage,
            from_hook,
            tokens_before,
            ..
        } => Ok(CommittedWrite::Entry {
            entry: crate::harness::session::types::Entry::Compaction {
                id,
                parent_id,
                seq: *seq,
                timestamp,
                body: crate::harness::session::types::CompactionEntryBody {
                    summary: summary.clone(),
                    retained_tail,
                    tokens_before: *tokens_before,
                    details: details.clone(),
                    usage: *usage,
                    from_hook: from_hook.unwrap_or(false),
                },
            },
        }),
        LegacyV3Entry::Custom {
            custom_type, data, ..
        } => Ok(CommittedWrite::Entry {
            entry: crate::harness::session::types::Entry::Custom {
                id,
                parent_id,
                seq: *seq,
                timestamp,
                body: crate::harness::session::types::CustomEntryBody {
                    custom_type: custom_type.clone(),
                    data: data.clone(),
                },
            },
        }),
        LegacyV3Entry::ModelChange { .. }
        | LegacyV3Entry::ThinkingLevelChange { .. }
        | LegacyV3Entry::ActiveToolsChange { .. }
        | LegacyV3Entry::SessionInfo { .. }
        | LegacyV3Entry::Label { .. } => unreachable_retained(),
    }
}

#[expect(
    clippy::panic,
    reason = "normalize_retained_entry is only called on retained records; the fallback is unreachable by construction"
)]
fn unreachable_retained() -> ! {
    panic!("normalize_retained_entry called on a discarded record")
}

/// The lane configuration the selected ancestry reconstructs, upstream's
/// `selectedConfiguration`: the nearest change of each kind wins, and an
/// invalid partial configuration (missing model or thinking level) yields
/// nothing.
fn selected_configuration(
    entries_by_id: &BTreeMap<String, LegacyV3IndexEntry>,
    selected_id: Option<&str>,
) -> Option<LaneConfiguration> {
    use crate::harness::session::types::ModelIdentity;
    let mut remaining: BTreeSet<&str> = [
        "model_change",
        "thinking_level_change",
        "active_tools_change",
    ]
    .into_iter()
    .collect();
    let mut model: Option<ModelIdentity> = None;
    let mut thinking_level: Option<ThinkingLevel> = None;
    let mut active_tool_names: Option<Vec<String>> = None;
    let mut current_id = selected_id.map(str::to_owned);
    while let Some(current) = current_id
        && !remaining.is_empty()
    {
        let Some(entry) = entries_by_id.get(&current) else {
            break;
        };
        let record_type = match entry {
            LegacyV3IndexEntry::Retained { kind, .. } => match kind {
                RetainedKind::Message | RetainedKind::Custom | RetainedKind::CustomMessage => {
                    "message"
                }
                RetainedKind::BranchSummary { .. } => "branch_summary",
                RetainedKind::Compaction { .. } => "compaction",
            },
            LegacyV3IndexEntry::Discarded { kind, .. } => match kind {
                DiscardedKind::Label { .. } => "label",
                DiscardedKind::ModelChange { .. } => "model_change",
                DiscardedKind::ThinkingLevelChange { .. } => "thinking_level_change",
                DiscardedKind::ActiveToolsChange { .. } => "active_tools_change",
                DiscardedKind::SessionInfo => "session_info",
            },
        };
        // Consume the nearest change even when invalid: older values must
        // not become fallbacks.
        if remaining.remove(record_type) {
            match entry {
                LegacyV3IndexEntry::Discarded {
                    kind: DiscardedKind::ModelChange { provider, model_id },
                    ..
                } => {
                    model = Some(ModelIdentity {
                        provider: provider.clone(),
                        model_id: model_id.clone(),
                    });
                }
                LegacyV3IndexEntry::Discarded {
                    kind:
                        DiscardedKind::ThinkingLevelChange {
                            thinking_level: level,
                        },
                    ..
                } => {
                    thinking_level = Some(*level);
                }
                LegacyV3IndexEntry::Discarded {
                    kind:
                        DiscardedKind::ActiveToolsChange {
                            active_tool_names: names,
                        },
                    ..
                } => {
                    active_tool_names = Some(names.clone());
                }
                _ => {}
            }
        }
        current_id = entry.parent_id().map(str::to_owned);
    }
    Some(LaneConfiguration {
        model: model?,
        thinking_level: thinking_level?,
        active_tool_names: active_tool_names.unwrap_or_default(),
    })
}

/// Indexes one record during the scan, upstream's `indexLegacyV3Entry`:
/// `SessionManager` appends after an existing leaf and writes extracted
/// branches in parent order, so parents exist earlier in the file.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per wire record family mirrors upstream's switch; splitting would scatter the mapping"
)]
fn index_legacy_v3_entry(
    entry: &LegacyV3Entry,
    line_number: usize,
    seq: u64,
    entries: &BTreeMap<String, LegacyV3IndexEntry>,
) -> Result<LegacyV3IndexEntry, JsonlError> {
    let parent_id = entry.parent_id();
    let mapped_parent_id = match parent_id {
        None => None,
        Some(parent) => {
            // A discarded parent's mapped id chains to its own parent's
            // boundary, which reparents the child past the discarded node.
            let Some(parent_entry) = entries.get(parent) else {
                return Err(JsonlError(format!(
                    "Legacy v3 entry {} has a missing or forward parent at line {line_number}: {parent}",
                    entry.id()
                )));
            };
            parent_entry.mapped_id().map(str::to_owned)
        }
    };
    if !entry.is_retained() {
        return Ok(match entry {
            LegacyV3Entry::Label {
                base: _,
                target_id,
                label,
            } => LegacyV3IndexEntry::Discarded {
                id: entry.id().to_owned(),
                parent_id: parent_id.map(str::to_owned),
                mapped_id: mapped_parent_id,
                kind: DiscardedKind::Label {
                    target_id: target_id.clone(),
                    label: label.clone(),
                },
            },
            LegacyV3Entry::ModelChange {
                provider, model_id, ..
            } => LegacyV3IndexEntry::Discarded {
                id: entry.id().to_owned(),
                parent_id: parent_id.map(str::to_owned),
                mapped_id: mapped_parent_id,
                kind: DiscardedKind::ModelChange {
                    provider: provider.clone(),
                    model_id: model_id.clone(),
                },
            },
            LegacyV3Entry::ThinkingLevelChange { thinking_level, .. } => {
                LegacyV3IndexEntry::Discarded {
                    id: entry.id().to_owned(),
                    parent_id: parent_id.map(str::to_owned),
                    mapped_id: mapped_parent_id,
                    kind: DiscardedKind::ThinkingLevelChange {
                        thinking_level: *thinking_level,
                    },
                }
            }
            LegacyV3Entry::ActiveToolsChange {
                active_tool_names, ..
            } => LegacyV3IndexEntry::Discarded {
                id: entry.id().to_owned(),
                parent_id: parent_id.map(str::to_owned),
                mapped_id: mapped_parent_id,
                kind: DiscardedKind::ActiveToolsChange {
                    active_tool_names: active_tool_names.clone(),
                },
            },
            LegacyV3Entry::SessionInfo { .. } => LegacyV3IndexEntry::Discarded {
                id: entry.id().to_owned(),
                parent_id: parent_id.map(str::to_owned),
                mapped_id: mapped_parent_id,
                kind: DiscardedKind::SessionInfo,
            },
            _ => unreachable_discarded(),
        });
    }
    // Null denotes the root, not a retained node identity that can be
    // reminted.
    let mapped_id = pi_ai::utils::uuid::uuidv7(Some(
        u64::try_from(parse_iso8601_millis(entry.timestamp()).unwrap_or_default())
            .unwrap_or_default(),
    ))
    .map_err(|error| JsonlError(error.to_string()))?;
    Ok(match entry {
        LegacyV3Entry::BranchSummary { from_id, .. } => LegacyV3IndexEntry::Retained {
            id: entry.id().to_owned(),
            parent_id: parent_id.map(str::to_owned),
            mapped_id,
            seq,
            kind: RetainedKind::BranchSummary {
                from_id: from_id.clone(),
            },
        },
        LegacyV3Entry::Compaction {
            first_kept_entry_id,
            ..
        } => LegacyV3IndexEntry::Retained {
            id: entry.id().to_owned(),
            parent_id: parent_id.map(str::to_owned),
            mapped_id,
            seq,
            kind: RetainedKind::Compaction {
                first_kept_entry_id: first_kept_entry_id.clone(),
            },
        },
        _ => LegacyV3IndexEntry::Retained {
            id: entry.id().to_owned(),
            parent_id: parent_id.map(str::to_owned),
            mapped_id,
            seq,
            kind: match entry {
                LegacyV3Entry::Custom { .. } => RetainedKind::Custom,
                LegacyV3Entry::CustomMessage { .. } => RetainedKind::CustomMessage,
                _ => RetainedKind::Message,
            },
        },
    })
}

#[expect(
    clippy::panic,
    reason = "index_legacy_v3_entry only reaches the discarded arm for discarded records; the fallback is unreachable by construction"
)]
fn unreachable_discarded() -> ! {
    panic!("index_legacy_v3_entry reached the discarded arm for a retained record")
}

/// The usage row one record contributes to the imported ledger, upstream's
/// `legacyEntryUsage`.
fn legacy_entry_usage(entry: &LegacyV3Entry) -> Option<Usage> {
    match entry {
        LegacyV3Entry::Message { message, .. } => {
            let crate::types::AgentMessage::Standard(message) = message.as_ref() else {
                return None;
            };
            match message {
                pi_ai::types::Message::Assistant(assistant) => Some(assistant.usage),
                pi_ai::types::Message::ToolResult(tool_result) => tool_result.usage,
                pi_ai::types::Message::User(_) => None,
            }
        }
        LegacyV3Entry::Compaction { usage, .. } | LegacyV3Entry::BranchSummary { usage, .. } => {
            *usage
        }
        _ => None,
    }
}

/// Scans the complete v3 records, upstream's `readLegacyV3Inventory`.
///
/// # Errors
/// A [`JsonlError`] for read failures, duplicate ids, unsupported records,
/// and missing parents.
async fn read_legacy_v3_inventory(
    reader: &mut dyn TextLineReader,
    context: &Context,
) -> Result<LegacyV3Inventory, JsonlError> {
    let mut entries: BTreeMap<String, LegacyV3IndexEntry> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut next_seq: u64 = 1;
    let mut imported_usage = empty_usage();
    let mut name: Option<String> = None;
    let mut final_id: Option<String> = None;
    loop {
        let line = file_value(
            reader.read_line(context).await,
            "Failed to read legacy v3 source",
        )?;
        let Some(line) = line else { break };
        if !line.terminated {
            break;
        }
        let line_number = entries.len() + 2;
        let entry = parse_legacy_v3_entry(&line.text)?;
        if entries.contains_key(entry.id()) {
            return Err(JsonlError(format!(
                "Duplicate legacy v3 entry id: {}",
                entry.id()
            )));
        }
        let indexed = index_legacy_v3_entry(&entry, line_number, next_seq, &entries)?;
        if indexed.is_retained() {
            next_seq += 1;
        }
        entries.insert(entry.id().to_owned(), indexed);
        order.push(entry.id().to_owned());
        final_id = Some(entry.id().to_owned());
        if matches!(entry, LegacyV3Entry::SessionInfo { .. }) {
            let LegacyV3Entry::SessionInfo {
                name: record_name, ..
            } = &entry
            else {
                unreachable!("matched session_info");
            };
            name.clone_from(record_name);
        }
        if let Some(usage) = legacy_entry_usage(&entry) {
            imported_usage = add_usage(imported_usage, usage);
        }
    }
    Ok(LegacyV3Inventory {
        entries,
        order,
        imported_usage,
        name,
        final_id,
        next_seq,
    })
}

/// The derived current values one v3 file normalizes to, upstream's
/// `normalizeLegacyV3Values`.
fn normalize_legacy_v3_values(
    inventory: &LegacyV3Inventory,
) -> Result<Vec<CommittedWrite>, JsonlError> {
    use crate::harness::session::values::{
        branch_tip, entry_label, lane_config, lane_state, session_name,
    };
    let resolve_legacy_id = create_legacy_id_resolver(&inventory.entries);
    let mut next_seq = inventory.next_seq;
    let mut values: Vec<CommittedWrite> = Vec::new();

    let push_value = |write: ValueSetWrite, next_seq: &mut u64| {
        let seq = *next_seq;
        *next_seq += 1;
        CommittedWrite::ValueSet {
            seq,
            namespace: write.namespace,
            key: write.key,
            value: write.value,
        }
    };

    // session name: an absent or empty name clears it, upstream's
    // `if (name)` truthiness.
    if let Some(name) = inventory.name.as_deref().filter(|name| !name.is_empty()) {
        values.push(push_value(
            crate::harness::session::values::set_value(&session_name(), name.to_owned())
                .map_err(|error| JsonlError(error.to_string()))?,
            &mut next_seq,
        ));
    }

    // labels: the latest label per target, targets without retained
    // ancestors skipped.
    let mut labels: BTreeMap<String, String> = BTreeMap::new();
    for entry in inventory.entries.values() {
        let LegacyV3IndexEntry::Discarded {
            kind: DiscardedKind::Label { target_id, label },
            ..
        } = entry
        else {
            continue;
        };
        let Some(target) = resolve_legacy_id(target_id)? else {
            continue;
        };
        match label.as_deref() {
            // Upstream's `if (entry.label)` truthiness: an empty label
            // clears the target's label.
            Some(label) if !label.is_empty() => {
                labels.insert(target, label.to_owned());
            }
            _ => {
                labels.remove(&target);
            }
        }
    }
    for (target_id, label) in &labels {
        values.push(push_value(
            crate::harness::session::values::set_value(&entry_label(target_id), label.clone())
                .map_err(|error| JsonlError(error.to_string()))?,
            &mut next_seq,
        ));
    }

    // branch tip
    let tip = inventory
        .final_id
        .as_deref()
        .map(&resolve_legacy_id)
        .transpose()?
        .flatten();
    values.push(push_value(
        crate::harness::session::values::set_value(&branch_tip("main"), tip)
            .map_err(|error| JsonlError(error.to_string()))?,
        &mut next_seq,
    ));

    // configuration
    if let Some(configuration) =
        selected_configuration(&inventory.entries, inventory.final_id.as_deref())
    {
        values.push(push_value(
            crate::harness::session::values::set_value(&lane_config("main"), configuration)
                .map_err(|error| JsonlError(error.to_string()))?,
            &mut next_seq,
        ));
        values.push(push_value(
            crate::harness::session::values::set_value(
                &lane_state("main"),
                crate::harness::session::types::LaneState::default(),
            )
            .map_err(|error| JsonlError(error.to_string()))?,
            &mut next_seq,
        ));
    }
    Ok(values)
}

impl std::fmt::Debug for LegacyV3Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LegacyV3Source")
            .field("header", &self.header)
            .field("imported_usage", &self.imported_usage)
            .field("next_seq", &self.next_seq)
            .finish_non_exhaustive()
    }
}

/// A captured legacy file exposed as repeatable logical v4 writes, upstream's
/// `LegacyV3Source`.
pub struct LegacyV3Source {
    /// The normalized format-4 header.
    pub header: JsonlStorageHeader,
    /// The usage the file's records carry, upstream's `importedUsage`.
    pub imported_usage: Usage,
    /// The next sequence a later source write takes, upstream's `nextSeq`.
    pub next_seq: u64,
    /// The derived current values, upstream's `values`.
    pub values: Vec<CommittedWrite>,
    file_system: Arc<dyn FileSystem>,
    path: String,
    /// The structural index, upstream's `entries`.
    entries: BTreeMap<String, LegacyV3IndexEntry>,
    /// The records' file order, upstream's Map insertion order.
    order: Vec<String>,
}

impl LegacyV3Source {
    /// Scans the complete v3 records without modifying the file, upstream's
    /// `LegacyV3Source.read`: an unterminated final line is ignored, and the
    /// structural indexes, label/configuration metadata, and derived
    /// current values survive between scans.
    ///
    /// # Errors
    /// A [`JsonlError`] from the reads, the header parse, and the record
    /// validations.
    pub async fn read(
        file_system: Arc<dyn FileSystem>,
        path: &str,
        context: &Context,
    ) -> Result<Self, JsonlError> {
        let mut reader = file_value(
            file_system.open_text_line_reader(path, context).await,
            &format!("Failed to open legacy v3 source {path}"),
        )?;
        let result = Self::read_through(&mut *reader, file_system.clone(), path, context).await;
        reader.close(context).await;
        result
    }

    async fn read_through(
        reader: &mut dyn TextLineReader,
        file_system: Arc<dyn FileSystem>,
        path: &str,
        context: &Context,
    ) -> Result<Self, JsonlError> {
        let parsed = read_jsonl_header(reader, path, context).await?;
        let crate::harness::session::jsonl::codec::JsonlParsedSessionHeader::V3Legacy(header) =
            parsed
        else {
            return Err(JsonlError(format!(
                "Invalid legacy v3 JSONL storage {path}: expected format 3 header"
            )));
        };
        let inventory = read_legacy_v3_inventory(reader, context).await?;
        let values = normalize_legacy_v3_values(&inventory)?;
        Ok(Self {
            header: normalize_legacy_v3_header(file_system.as_ref(), &header, context).await,
            imported_usage: inventory.imported_usage,
            next_seq: inventory.next_seq + values.len() as u64,
            values,
            file_system,
            path: path.to_owned(),
            entries: inventory.entries,
            order: inventory.order,
        })
    }

    /// The retained records' structural rows, upstream's `entryStructures`.
    ///
    /// # Errors
    /// A missing legacy reference, which the scan already validated.
    pub fn entry_structures(&self) -> Result<Vec<LegacyV3EntryRow>, JsonlError> {
        let resolve_legacy_id = create_legacy_id_resolver(&self.entries);
        self.order
            .iter()
            .filter_map(|id| self.entries.get(id))
            .filter(|entry| entry.is_retained())
            .map(|entry| {
                let LegacyV3IndexEntry::Retained { mapped_id, seq, .. } = entry else {
                    unreachable_retained();
                };
                Ok(LegacyV3EntryRow {
                    id: mapped_id.clone(),
                    parent_id: entry
                        .parent_id()
                        .map(&resolve_legacy_id)
                        .transpose()?
                        .flatten(),
                    seq: *seq,
                })
            })
            .collect()
    }

    /// Resolves a legacy fork entry to its reminted id, upstream's
    /// `translateForkEntryId`.
    ///
    /// # Errors
    /// A [`JsonlError`] when the legacy id does not exist or is discarded.
    pub fn translate_fork_entry_id(&self, legacy_id: &str) -> Result<String, JsonlError> {
        let Some(entry) = self.entries.get(legacy_id) else {
            return Err(JsonlError(format!(
                "Legacy v3 fork entry does not exist: {legacy_id}"
            )));
        };
        let LegacyV3IndexEntry::Retained { mapped_id, .. } = entry else {
            return Err(JsonlError(format!(
                "Legacy v3 fork entry is not a retained entry: {legacy_id}"
            )));
        };
        Ok(mapped_id.clone())
    }

    /// The tail-entry ids the selected compactions reconstruct from,
    /// upstream's `collectRequiredTailMessageIds`.
    fn collect_required_tail_message_ids(
        &self,
        is_entry_selected: Option<&(dyn Fn(&str) -> bool + Send + Sync)>,
    ) -> BTreeSet<String> {
        let mut required_ids = BTreeSet::new();
        for entry in self.entries.values() {
            let LegacyV3IndexEntry::Retained {
                id,
                parent_id,
                kind:
                    RetainedKind::Compaction {
                        first_kept_entry_id,
                    },
                ..
            } = entry
            else {
                continue;
            };
            if let Some(is_entry_selected) = is_entry_selected
                && !is_entry_selected(entry.mapped_id().unwrap_or_default())
            {
                continue;
            }
            let Ok(tail_entries) = retained_tail_structure(
                (id, parent_id.as_deref(), first_kept_entry_id),
                &self.entries,
            ) else {
                continue;
            };
            for tail_entry in tail_entries {
                let can_produce_context_message = match tail_entry {
                    LegacyV3IndexEntry::Retained { kind, .. } => {
                        !matches!(kind, RetainedKind::Custom)
                    }
                    LegacyV3IndexEntry::Discarded { .. } => false,
                };
                if can_produce_context_message {
                    required_ids.insert(tail_entry.id().to_owned());
                }
            }
        }
        required_ids
    }

    /// The logical v4 writes one pass materializes, upstream's `writes`:
    /// normalized entries (optionally filtered by reminted id) followed by
    /// the derived current values.
    ///
    /// # Errors
    /// A [`JsonlError`] when the source changed under the pass or a
    /// reference does not resolve.
    pub async fn writes(
        &self,
        context: &Context,
        is_entry_selected: Option<&(dyn Fn(&str) -> bool + Send + Sync)>,
    ) -> Result<Vec<CommittedWrite>, JsonlError> {
        let required_tail_message_ids = self.collect_required_tail_message_ids(is_entry_selected);
        let resolve_legacy_id = create_legacy_id_resolver(&self.entries);
        let mut tail_messages_by_legacy_id: BTreeMap<String, crate::types::AgentMessage> =
            BTreeMap::new();
        let mut writes: Vec<CommittedWrite> = Vec::new();

        for (entry, indexed) in self.read_captured_entries(context).await? {
            if required_tail_message_ids.contains(entry.id())
                && let Some(message) = project_context_message(&entry, &resolve_legacy_id)?
            {
                tail_messages_by_legacy_id.insert(entry.id().to_owned(), message);
            }
            if !entry.is_retained() || !indexed.is_retained() {
                continue;
            }
            if let Some(is_entry_selected) = is_entry_selected
                && !is_entry_selected(indexed.mapped_id().unwrap_or_default())
            {
                continue;
            }
            let mut retained_tail: Vec<crate::types::AgentMessage> = Vec::new();
            if let LegacyV3IndexEntry::Retained {
                id,
                parent_id,
                kind:
                    RetainedKind::Compaction {
                        first_kept_entry_id,
                    },
                ..
            } = &indexed
            {
                let tail_entries = retained_tail_structure(
                    (id, parent_id.as_deref(), first_kept_entry_id),
                    &self.entries,
                )?;
                for ancestor in tail_entries {
                    if let Some(message) = tail_messages_by_legacy_id.get(ancestor.id()) {
                        retained_tail.push(message.clone());
                    }
                }
                retained_tail.reverse();
            }
            writes.push(normalize_retained_entry(
                &entry,
                &indexed,
                retained_tail,
                &resolve_legacy_id,
            )?);
        }
        writes.extend(self.values.iter().cloned());
        Ok(writes)
    }

    /// Replays the captured prefix and verifies its physical identities
    /// before materialization, upstream's `readCapturedEntries`.
    ///
    /// # Errors
    /// A [`JsonlError`] when the header or any record changed, or the file
    /// ended before the captured entries.
    async fn read_captured_entries(
        &self,
        context: &Context,
    ) -> Result<Vec<(LegacyV3Entry, LegacyV3IndexEntry)>, JsonlError> {
        let mut reader = file_value(
            self.file_system
                .open_text_line_reader(&self.path, context)
                .await,
            &format!("Failed to reopen legacy v3 source {}", self.path),
        )?;
        let result = self.read_captured_through(&mut *reader, context).await;
        reader.close(context).await;
        result
    }

    async fn read_captured_through(
        &self,
        reader: &mut dyn TextLineReader,
        context: &Context,
    ) -> Result<Vec<(LegacyV3Entry, LegacyV3IndexEntry)>, JsonlError> {
        use crate::harness::session::jsonl::codec::JsonlParsedSessionHeader;
        let parsed = read_jsonl_header(reader, &self.path, context).await?;
        let matches = matches!(&parsed, JsonlParsedSessionHeader::V3Legacy(header)
            if header.id == self.header.id && header.cwd == self.header.cwd);
        if !matches {
            return Err(JsonlError("Legacy v3 source header changed".to_owned()));
        }
        let mut captured: Vec<(LegacyV3Entry, LegacyV3IndexEntry)> = Vec::new();
        for indexed in self.order.iter().filter_map(|id| self.entries.get(id)) {
            let line = file_value(
                reader.read_line(context).await,
                "Failed to reread legacy v3 source",
            )?;
            let Some(line) = line.filter(|line| line.terminated) else {
                return Err(JsonlError(
                    "Legacy v3 source ended before captured entries".to_owned(),
                ));
            };
            let entry = parse_legacy_v3_entry(&line.text)?;
            if entry.id() != indexed.id() || entry.record_type() != indexed.record_type() {
                return Err(JsonlError("Legacy v3 source changed".to_owned()));
            }
            captured.push((entry, indexed.clone()));
        }
        Ok(captured)
    }
}

impl LegacyV3IndexEntry {
    /// The wire record type the index entry mirrors, upstream's
    /// `entry.type` compare in `readCapturedEntries`.
    const fn record_type(&self) -> &'static str {
        match self {
            Self::Retained { kind, .. } => match kind {
                RetainedKind::Message => "message",
                RetainedKind::Custom => "custom",
                RetainedKind::CustomMessage => "custom_message",
                RetainedKind::BranchSummary { .. } => "branch_summary",
                RetainedKind::Compaction { .. } => "compaction",
            },
            Self::Discarded { kind, .. } => match kind {
                DiscardedKind::Label { .. } => "label",
                DiscardedKind::ModelChange { .. } => "model_change",
                DiscardedKind::ThinkingLevelChange { .. } => "thinking_level_change",
                DiscardedKind::ActiveToolsChange { .. } => "active_tools_change",
                DiscardedKind::SessionInfo => "session_info",
            },
        }
    }
}

/// The repo's session metadata the v3 discovery builds, upstream's
/// `{ ...metadataFromLegacyV3Header(), path, modifiedAt }` spread.
#[must_use]
pub fn jsonl_metadata_from_base(
    base: JsonlSessionMetadataBase,
    path: String,
    modified_at: i64,
) -> JsonlSessionMetadata {
    JsonlSessionMetadata {
        id: base.id,
        created_at: base.created_at,
        storage_version: base.storage_version,
        cwd: base.cwd,
        path,
        modified_at,
        parent_session_id: base.parent_session_id,
        legacy_parent_session_path: base.legacy_parent_session_path,
    }
}

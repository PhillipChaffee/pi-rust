//! The typed session entry shapes, upstream's `SessionEntry` union members.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{Usage, UserContent};

/// The typed session entry's shared fields, upstream's `SessionEntryBase`.
///
/// The id is optional so pre-migration v1 entries (which carry none) parse;
/// [`crate::session_manager::SessionManager`] migrates them before anything
/// indexes the file. `timestamp` is absent-only-empty so a rewrite of a
/// timestampless parsed entry drops the field the way upstream's
/// `JSON.stringify` dropped `undefined`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionEntryBase {
    /// The tree-structure id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The parent entry's id; `null` at a root.
    pub parent_id: Option<String>,
    /// The entry timestamp, ISO-8601 `Z` form.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub timestamp: String,
    /// Fields outside the typed entry, preserved verbatim through reloads
    /// and rewrites.
    #[serde(flatten)]
    pub extras: JsonMap2,
}

/// Alias keeping the flatten imports readable in this module.
type JsonMap2 = serde_json::Map<String, JsonValue>;

/// A transcript message entry, wire `"type": "message"`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageEntry {
    /// The base fields, flattened.
    #[serde(flatten)]
    pub base: SessionEntryBase,
    /// The message payload; absent when the line carried none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<AgentMessage>,
    /// Fields outside the typed entry, preserved verbatim.
    #[serde(flatten)]
    pub extras: JsonMap2,
}

/// A thinking-level change, wire `"type": "thinking_level_change"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingLevelChangeEntry {
    /// The base fields, flattened.
    #[serde(flatten)]
    pub base: SessionEntryBase,
    /// The level the session switched to.
    pub thinking_level: String,
    /// Fields outside the typed entry, preserved verbatim.
    #[serde(flatten)]
    pub extras: JsonMap2,
}

/// A model change, wire `"type": "model_change"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelChangeEntry {
    /// The base fields, flattened.
    #[serde(flatten)]
    pub base: SessionEntryBase,
    /// The provider the session switched to.
    pub provider: String,
    /// The model id the session switched to.
    pub model_id: String,
    /// Fields outside the typed entry, preserved verbatim.
    #[serde(flatten)]
    pub extras: JsonMap2,
}

/// A compaction summary, wire `"type": "compaction"`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionEntry {
    /// The base fields, flattened.
    #[serde(flatten)]
    pub base: SessionEntryBase,
    /// The generated summary.
    pub summary: String,
    /// The first entry the compaction kept; absent on v1-migrated
    /// compactions whose index pointed past the transcript.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_kept_entry_id: Option<String>,
    /// The token count before compaction.
    pub tokens_before: i64,
    /// Extension-specific data (e.g. an `ArtifactIndex`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
    /// Usage from the LLM call(s) that generated this summary, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// True when an extension generated the summary, absent when pi did
    /// (backward compatible).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_hook: Option<bool>,
    /// Fields outside the typed entry, preserved verbatim.
    #[serde(flatten)]
    pub extras: JsonMap2,
}

/// A branch summary, wire `"type": "branch_summary"`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryEntry {
    /// The base fields, flattened.
    #[serde(flatten)]
    pub base: SessionEntryBase,
    /// The summarized branch's source entry id.
    pub from_id: String,
    /// The generated summary.
    pub summary: String,
    /// Extension-specific data (not sent to the LLM).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
    /// Usage from the LLM call that generated this summary, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// True when an extension generated the summary, absent or false when
    /// pi did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_hook: Option<bool>,
    /// Fields outside the typed entry, preserved verbatim.
    #[serde(flatten)]
    pub extras: JsonMap2,
}

/// An extension state entry, wire `"type": "custom"`.
///
/// Custom entries do not participate in the LLM context; extensions scan for
/// their `customType` on reload to reconstruct internal state.
#[expect(
    clippy::derive_partial_eq_without_eq,
    reason = "the extras flatten rides serde_json::Value, which is PartialEq-only; Eq would bar the round-trip fidelity"
)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomEntry {
    /// The extension's type discriminator.
    pub custom_type: String,
    /// The extension-defined payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<JsonValue>,
    /// The base fields, flattened.
    #[serde(flatten)]
    pub base: SessionEntryBase,
    /// Fields outside the typed entry, preserved verbatim.
    #[serde(flatten)]
    pub extras: JsonMap2,
}

/// A user-defined bookmark on an entry, wire `"type": "label"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LabelEntry {
    /// The base fields, flattened.
    #[serde(flatten)]
    pub base: SessionEntryBase,
    /// The labeled entry's id.
    pub target_id: String,
    /// The label text; absent when the entry clears the label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Fields outside the typed entry, preserved verbatim.
    #[serde(flatten)]
    pub extras: JsonMap2,
}

/// A session metadata entry, wire `"type": "session_info"` (e.g. a
/// user-defined display name).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfoEntry {
    /// The base fields, flattened.
    #[serde(flatten)]
    pub base: SessionEntryBase,
    /// The display name; absent when the entry clears the name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Fields outside the typed entry, preserved verbatim.
    #[serde(flatten)]
    pub extras: JsonMap2,
}

/// An extension-injected message entry that participates in the LLM context,
/// wire `"type": "custom_message"`.
#[expect(
    clippy::derive_partial_eq_without_eq,
    reason = "the extras flatten rides serde_json::Value, which is PartialEq-only; Eq would bar the round-trip fidelity"
)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessageEntry {
    /// The extension's type discriminator.
    pub custom_type: String,
    /// The message content, a plain string or text/image blocks; absent
    /// parses as the empty block list, upstream's `?? []`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<UserContent>,
    /// Whether the message displays in the transcript.
    #[serde(default)]
    pub display: bool,
    /// Extension-specific metadata (not sent to the LLM).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
    /// The base fields, flattened.
    #[serde(flatten)]
    pub base: SessionEntryBase,
    /// Fields outside the typed entry, preserved verbatim.
    #[serde(flatten)]
    pub extras: JsonMap2,
}

/// One session tree entry, upstream's `SessionEntry` union over `"type"`.
#[expect(
    clippy::large_enum_variant,
    reason = "the message variant is upstream's dominant entry kind; boxing would allocate on every transcript line parse"
)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionEntry {
    /// A transcript message.
    #[serde(rename = "message")]
    Message(MessageEntry),
    /// A thinking-level change.
    #[serde(rename = "thinking_level_change")]
    ThinkingLevelChange(ThinkingLevelChangeEntry),
    /// A model change.
    #[serde(rename = "model_change")]
    ModelChange(ModelChangeEntry),
    /// A compaction summary.
    #[serde(rename = "compaction")]
    Compaction(CompactionEntry),
    /// A branch summary.
    #[serde(rename = "branch_summary")]
    BranchSummary(BranchSummaryEntry),
    /// An extension state entry.
    #[serde(rename = "custom")]
    Custom(CustomEntry),
    /// A user-defined bookmark.
    #[serde(rename = "label")]
    Label(LabelEntry),
    /// A session metadata entry.
    #[serde(rename = "session_info")]
    SessionInfo(SessionInfoEntry),
    /// An extension-injected context message.
    #[serde(rename = "custom_message")]
    CustomMessage(CustomMessageEntry),
}

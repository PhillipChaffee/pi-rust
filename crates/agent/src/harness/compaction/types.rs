//! The compaction domain's shared type surface, ported from upstream
//! `src/harness/compaction/compaction.ts` and
//! `compaction/branch-summarization.ts`.
//!
//! The map's harness-foundations child carries the type surface its
//! signatures and hook payloads reference (`CompactionSettings`,
//! `CompactionPreparation`, `CompactResult`, the branch-summary pair, and
//! `FileOperations`); the compaction child owns the module's logic and the
//! rest of `utils.ts` (recorded on that ticket). The types ride the ported
//! suites elsewhere; this module has no upstream unit file of its own.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::types::AgentMessage;

/// The file operations a structural summary records, upstream's
/// `FileOperations` in `compaction/utils.ts` — sets at runtime, sorted
/// vectors on the wire.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileOperations {
    /// Files read by the summarized messages.
    pub read: Vec<String>,
    /// Files written or created by the summarized messages.
    pub written: Vec<String>,
    /// Files edited by the summarized messages.
    pub edited: Vec<String>,
}

/// Builds empty file operations, upstream's `createFileOps`.
#[must_use]
pub fn create_file_ops() -> FileOperations {
    FileOperations::default()
}

/// The sorted-vector wire form's set view, for extraction sites that
/// accumulate.
pub type FileOperationsSet = BTreeSet<String>;

/// Compaction thresholds and retention settings, upstream's
/// `CompactionSettings`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSettings {
    /// Enable automatic compaction decisions.
    pub enabled: bool,
    /// Tokens reserved for the summary prompt and output.
    pub reserve_tokens: u64,
    /// Approximate recent-context tokens to keep after compaction.
    pub keep_recent_tokens: u64,
}

/// Default compaction settings used by the harness, upstream's
/// `DEFAULT_COMPACTION_SETTINGS`.
pub const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = CompactionSettings {
    enabled: true,
    reserve_tokens: 16_384,
    keep_recent_tokens: 20_000,
};

/// Generated compaction data ready to be persisted as a compaction entry,
/// upstream's `CompactResult<T = JsonValue>`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactResult {
    /// Summary text that replaces compacted history in future context.
    pub summary: String,
    /// Estimated context tokens before compaction.
    pub tokens_before: i64,
    /// Usage from the LLM call(s) that generated this summary, if
    /// available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<pi_ai::types::Usage>,
    /// Recent messages retained after compaction and stored directly on
    /// the compaction entry.
    pub retained_tail: Vec<AgentMessage>,
    /// Optional implementation-specific details stored with the compaction
    /// entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
}

/// Prepared inputs for a compaction run, upstream's
/// `CompactionPreparation`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionPreparation {
    /// Messages summarized into the history summary.
    pub messages_to_summarize: Vec<AgentMessage>,
    /// Prefix messages summarized separately when compaction splits a
    /// turn.
    pub turn_prefix_messages: Vec<AgentMessage>,
    /// Recent messages retained after compaction and stored on the
    /// compaction entry.
    pub retained_tail: Vec<AgentMessage>,
    /// Whether compaction splits a turn.
    pub is_split_turn: bool,
    /// Estimated context tokens before compaction.
    pub tokens_before: i64,
    /// Previous compaction summary used for iterative updates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_summary: Option<String>,
    /// File operations extracted from summarized history.
    pub file_ops: FileOperations,
    /// Settings used to prepare compaction.
    pub settings: CompactionSettings,
}

/// Generated branch summary data ready to be persisted as a branch-summary
/// entry, upstream's `BranchSummaryResult`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryResult {
    /// The generated summary.
    pub summary: String,
    /// Usage from the summarization model call, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<pi_ai::types::Usage>,
    /// Files read while exploring the summarized branch.
    pub read_files: Vec<String>,
    /// Files modified while exploring the summarized branch.
    pub modified_files: Vec<String>,
}

/// Prepared branch content for summarization, upstream's
/// `BranchPreparation`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchPreparation {
    /// Messages selected for the branch summary.
    pub messages: Vec<AgentMessage>,
    /// File operations extracted from the branch.
    pub file_ops: FileOperations,
    /// Estimated token count for selected messages.
    pub total_tokens: i64,
}

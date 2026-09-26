//! The session contract surface, ported from upstream
//! `src/harness/session/types.ts`.
//!
//! Data shapes port 1:1 (entry and write unions, scans, the flat 13-leaf
//! durable operation state machine). The contract interfaces restate as
//! object-safe traits over erased value payloads: upstream's typed
//! `Value<T>`/`ValueList<T>` addresses keep their phantom types for
//! call-site typing in [`super::values`], while the trait methods carry
//! the runtime address plus JSON payloads, because durable values are
//! JSONL-persisted and the backends hold them as JSON at rest. Upstream's
//! `mutate<T>` generic restates as a callback returning
//! `Box<dyn Any + Send>` — a trait's generic method would forfeit the
//! object safety the harness stores `dyn Session` through; typed callers
//! downcast.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::types::BoxedFuture;
use pi_ai::types::{AssistantMessage, StopReason, Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::harness::compaction::types::CompactionSettings;
use crate::harness::context::Context;
use crate::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress, Write,
};
use crate::types::{AgentMessage, QueueMode, ThinkingLevel, ToolExecutionMode};

/// An assistant message settled past `pending`, upstream's
/// `SettledAssistantMessage = AssistantMessage & { stopReason: Exclude<
/// StopReason, "pending"> }`.
///
/// The port carries the settled reason as a field beside the message, so
/// the pending case is unrepresentable rather than a runtime lie.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettledAssistantMessage {
    /// The assistant message body.
    pub message: AssistantMessage,
    /// The settled stop reason; never `pending`.
    pub stop_reason: SettledStopReason,
}

/// The stop reasons a settled assistant message may carry, upstream's
/// `Exclude<StopReason, "pending">`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SettledStopReason {
    /// The model finished its turn.
    #[serde(rename = "stop")]
    Stop,
    /// The model hit a token limit.
    #[serde(rename = "length")]
    Length,
    /// The model called a tool.
    #[serde(rename = "toolUse")]
    ToolUse,
    /// The request failed.
    #[serde(rename = "error")]
    Error,
    /// The request was aborted.
    #[serde(rename = "aborted")]
    Aborted,
    /// The provider deferred the request and returned a durable handle.
    #[serde(rename = "deferred")]
    Deferred,
}

impl SettledStopReason {
    /// Narrows a stop reason, dropping `pending`.
    #[must_use]
    pub const fn from_stop_reason(stop_reason: StopReason) -> Option<Self> {
        match stop_reason {
            StopReason::Pending => None,
            StopReason::Stop => Some(Self::Stop),
            StopReason::Length => Some(Self::Length),
            StopReason::ToolUse => Some(Self::ToolUse),
            StopReason::Error => Some(Self::Error),
            StopReason::Aborted => Some(Self::Aborted),
            StopReason::Deferred => Some(Self::Deferred),
        }
    }

    /// The plain stop reason this settled value restates.
    #[must_use]
    pub const fn stop_reason(self) -> StopReason {
        match self {
            Self::Stop => StopReason::Stop,
            Self::Length => StopReason::Length,
            Self::ToolUse => StopReason::ToolUse,
            Self::Error => StopReason::Error,
            Self::Aborted => StopReason::Aborted,
            Self::Deferred => StopReason::Deferred,
        }
    }
}

/// Error raised by session and storage operations, the typed restatement
/// of upstream's thrown `Error`s in the session layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionError(pub String);

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SessionError {}

/// The entry types a session transcript holds, upstream's `EntryType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryType {
    /// A transcript message.
    #[default]
    #[serde(rename = "message")]
    Message,
    /// A compaction summary.
    #[serde(rename = "compaction")]
    Compaction,
    /// A branch summary.
    #[serde(rename = "branch_summary")]
    BranchSummary,
    /// An application-defined entry.
    #[serde(rename = "custom")]
    Custom,
}

/// The message entry body, upstream's `MessageEntry`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageEntry {
    /// The transcript message.
    pub message: AgentMessage,
    /// Whether this entry terminated the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminate: Option<bool>,
}

/// The compaction entry body, upstream's `CompactionEntry`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionEntryBody {
    /// The generated summary.
    pub summary: String,
    /// Recent messages stored directly on the compaction entry.
    pub retained_tail: Vec<AgentMessage>,
    /// Estimated context tokens before compaction.
    pub tokens_before: i64,
    /// Implementation-specific details stored with the entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
    /// Usage from the summarization model call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Whether a compaction hook produced this entry.
    pub from_hook: bool,
}

/// The branch-summary entry body, upstream's `BranchSummaryEntry`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryEntryBody {
    /// The entry id the summarized branch forked from, when any.
    pub from_id: Option<String>,
    /// The generated summary.
    pub summary: String,
    /// Implementation-specific details stored with the entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
    /// Usage from the summarization model call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Whether a navigation hook produced this entry.
    pub from_hook: bool,
}

/// The custom entry body, upstream's `CustomEntry`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomEntryBody {
    /// The application's custom type discriminator.
    pub custom_type: String,
    /// The application-defined payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<JsonValue>,
}

/// One transcript entry, upstream's `Entry` union over `EntryBase`.
///
/// The base fields flatten onto every variant body; the wire carries the
/// flat object with the `"type"` discriminator, exactly upstream's entry
/// shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum Entry {
    /// A transcript message, wire `"type": "message"`.
    #[serde(rename = "message")]
    Message {
        /// Stable entry id.
        id: String,
        /// The parent entry id; `None` at a branch root.
        parent_id: Option<String>,
        /// The storage-assigned sequence.
        seq: u64,
        /// The storage-assigned timestamp, Unix epoch milliseconds.
        timestamp: i64,
        /// The message and termination flag.
        #[serde(flatten)]
        body: Box<MessageEntry>,
    },
    /// A compaction summary, wire `"type": "compaction"`.
    #[serde(rename = "compaction")]
    Compaction {
        /// Stable entry id.
        id: String,
        /// The parent entry id; `None` at a branch root.
        parent_id: Option<String>,
        /// The storage-assigned sequence.
        seq: u64,
        /// The storage-assigned timestamp.
        timestamp: i64,
        /// The compaction data.
        #[serde(flatten)]
        body: CompactionEntryBody,
    },
    /// A branch summary, wire `"type": "branch_summary"`.
    #[serde(rename = "branch_summary")]
    BranchSummary {
        /// Stable entry id.
        id: String,
        /// The parent entry id; `None` at a branch root.
        parent_id: Option<String>,
        /// The storage-assigned sequence.
        seq: u64,
        /// The storage-assigned timestamp.
        timestamp: i64,
        /// The branch-summary data.
        #[serde(flatten)]
        body: BranchSummaryEntryBody,
    },
    /// An application-defined entry, wire `"type": "custom"`.
    #[serde(rename = "custom")]
    Custom {
        /// Stable entry id.
        id: String,
        /// The parent entry id; `None` at a branch root.
        parent_id: Option<String>,
        /// The storage-assigned sequence.
        seq: u64,
        /// The storage-assigned timestamp.
        timestamp: i64,
        /// The application-defined data.
        #[serde(flatten)]
        body: CustomEntryBody,
    },
}

impl Entry {
    /// The stable entry id, upstream's `EntryBase.id`.
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Message { id, .. }
            | Self::BranchSummary { id, .. }
            | Self::Compaction { id, .. }
            | Self::Custom { id, .. } => id,
        }
    }

    /// The parent entry id, upstream's `EntryBase.parentId`.
    #[must_use]
    pub fn parent_id(&self) -> Option<&str> {
        match self {
            Self::Message { parent_id, .. }
            | Self::Compaction { parent_id, .. }
            | Self::BranchSummary { parent_id, .. }
            | Self::Custom { parent_id, .. } => parent_id.as_deref(),
        }
    }

    /// The storage-assigned sequence, upstream's `EntryBase.seq`.
    #[must_use]
    pub const fn seq(&self) -> u64 {
        match self {
            Self::Message { seq, .. }
            | Self::Compaction { seq, .. }
            | Self::BranchSummary { seq, .. }
            | Self::Custom { seq, .. } => *seq,
        }
    }

    /// The storage-assigned timestamp, upstream's `EntryBase.timestamp`.
    #[must_use]
    pub const fn timestamp(&self) -> i64 {
        match self {
            Self::Message { timestamp, .. }
            | Self::Compaction { timestamp, .. }
            | Self::BranchSummary { timestamp, .. }
            | Self::Custom { timestamp, .. } => *timestamp,
        }
    }

    /// The entry type, upstream's `EntryBase.type`.
    #[must_use]
    pub const fn entry_type(&self) -> EntryType {
        match self {
            Self::Message { .. } => EntryType::Message,
            Self::Compaction { .. } => EntryType::Compaction,
            Self::BranchSummary { .. } => EntryType::BranchSummary,
            Self::Custom { .. } => EntryType::Custom,
        }
    }

    /// The application's custom type discriminator, upstream's
    /// `EntryBase.customType`, carried by custom entries only.
    #[must_use]
    pub fn custom_type(&self) -> Option<&str> {
        match self {
            Self::Custom { body, .. } => Some(&body.custom_type),
            _ => None,
        }
    }
}

/// An entry supplied to a transaction before storage assigns sequence and
/// timestamp, upstream's `NewEntry` — the same wire shape as [`Entry`]
/// minus `seq`/`timestamp`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum NewEntry {
    /// A transcript message, wire `"type": "message"`.
    #[serde(rename = "message")]
    Message {
        /// Stable entry id.
        id: String,
        /// The parent entry id; `None` at a branch root.
        parent_id: Option<String>,
        /// The message and termination flag.
        #[serde(flatten)]
        body: Box<MessageEntry>,
    },
    /// A compaction summary, wire `"type": "compaction"`.
    #[serde(rename = "compaction")]
    Compaction {
        /// Stable entry id.
        id: String,
        /// The parent entry id; `None` at a branch root.
        parent_id: Option<String>,
        /// The compaction data.
        #[serde(flatten)]
        body: CompactionEntryBody,
    },
    /// A branch summary, wire `"type": "branch_summary"`.
    #[serde(rename = "branch_summary")]
    BranchSummary {
        /// Stable entry id.
        id: String,
        /// The parent entry id; `None` at a branch root.
        parent_id: Option<String>,
        /// The branch-summary data.
        #[serde(flatten)]
        body: BranchSummaryEntryBody,
    },
    /// An application-defined entry, wire `"type": "custom"`.
    #[serde(rename = "custom")]
    Custom {
        /// Stable entry id.
        id: String,
        /// The parent entry id; `None` at a branch root.
        parent_id: Option<String>,
        /// The custom-entry data.
        #[serde(flatten)]
        body: CustomEntryBody,
    },
}

impl NewEntry {
    /// The stable entry id, upstream's `EntryBase.id`.
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Message { id, .. }
            | Self::Compaction { id, .. }
            | Self::BranchSummary { id, .. }
            | Self::Custom { id, .. } => id,
        }
    }

    /// The parent entry id, upstream's `EntryBase.parentId`.
    #[must_use]
    pub fn parent_id(&self) -> Option<&str> {
        match self {
            Self::Message { parent_id, .. }
            | Self::Compaction { parent_id, .. }
            | Self::BranchSummary { parent_id, .. }
            | Self::Custom { parent_id, .. } => parent_id.as_deref(),
        }
    }

    /// Materializes the new entry with storage-assigned sequence and
    /// timestamp, upstream's `materializeCommittedEntry`.
    #[must_use]
    pub fn materialize(self, seq: u64, timestamp: i64) -> Entry {
        match self {
            Self::Message {
                id,
                parent_id,
                body,
            } => Entry::Message {
                id,
                parent_id,
                seq,
                timestamp,
                body,
            },
            Self::Compaction {
                id,
                parent_id,
                body,
            } => Entry::Compaction {
                id,
                parent_id,
                seq,
                timestamp,
                body,
            },
            Self::BranchSummary {
                id,
                parent_id,
                body,
            } => Entry::BranchSummary {
                id,
                parent_id,
                seq,
                timestamp,
                body,
            },
            Self::Custom {
                id,
                parent_id,
                body,
            } => Entry::Custom {
                id,
                parent_id,
                seq,
                timestamp,
                body,
            },
        }
    }
}

/// Converts an application-defined custom entry into model context,
/// upstream's `EntryProjector`.
///
/// Returns `None` when the entry contributes nothing to the context.
#[derive(Clone)]
pub struct EntryProjector(pub EntryProjectorFn);

/// The projector callback over one entry and the call's context.
pub type EntryProjectorFn = Arc<
    dyn for<'a> Fn(&Entry, &'a Context) -> BoxedFuture<'a, Option<Vec<AgentMessage>>> + Send + Sync,
>;

impl std::fmt::Debug for EntryProjector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EntryProjector(..)")
    }
}

/// A lane's model configuration, upstream's `LaneConfiguration`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaneConfiguration {
    /// The configured provider and model id.
    pub model: ModelIdentity,
    /// The configured reasoning level.
    pub thinking_level: ThinkingLevel,
    /// The active tool names.
    pub active_tool_names: Vec<String>,
}

/// The provider/model pair a configuration carries, upstream's
/// `{ provider, modelId }` inline shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelIdentity {
    /// The provider id.
    pub provider: String,
    /// The model id.
    pub model_id: String,
}

/// The durable intent one operation serves, upstream's `OperationMeta`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationMeta {
    /// The durable operation id.
    pub operation_id: String,
    /// The lane the operation runs on.
    pub lane: String,
    /// The tip entry id the operation started from.
    pub source_tip_id: Option<String>,
    /// Unix timestamp in milliseconds when the operation was admitted.
    pub started_at: i64,
    /// The operation's intent.
    pub intent: OperationIntent,
}

/// The intent union, upstream's `OperationMeta["intent"]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum OperationIntent {
    /// A run over prompt entries.
    #[serde(rename = "run")]
    Run {
        /// The prompt entry ids the run replays.
        prompt_entry_ids: Vec<String>,
    },
    /// A compaction over history.
    #[serde(rename = "compaction")]
    Compaction {
        /// Caller instructions for the summary.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        custom_instructions: Option<String>,
    },
    /// A navigation to a target branch point.
    #[serde(rename = "navigation")]
    Navigation {
        /// The target entry id; `None` targets the branch root.
        target_id: Option<String>,
        /// Whether to summarize the abandoned branch.
        summarize: bool,
        /// An optional label for the target entry.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        /// Caller instructions for the optional summary.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        custom_instructions: Option<String>,
    },
}

/// The cancellation control state of an operation, upstream's `Control`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "status",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum Control {
    /// The operation runs.
    #[serde(rename = "running")]
    Running,
    /// Cancellation was requested; the operation is settling.
    #[serde(rename = "cancel_requested")]
    CancelRequested {
        /// Unix epoch milliseconds when cancellation was requested.
        requested_at: i64,
    },
}

/// A durable operation error, upstream's `OperationError`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationError {
    /// The stable error code.
    pub code: String,
    /// The human-readable failure message.
    pub message: String,
    /// Implementation-specific error details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
}

/// The terminal operation statuses, upstream's `TerminalStatus`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalStatus {
    /// The operation completed.
    Completed,
    /// The operation declined to act.
    Declined,
    /// The operation was aborted.
    Aborted,
    /// The operation failed.
    Failed,
}

/// The immutable lane-lived observation record one terminal transaction
/// writes, upstream's `OperationResultRecord`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationResultRecord {
    /// The durable operation id.
    pub operation_id: String,
    /// The operation kind, upstream's `OperationMeta["intent"]["kind"]`.
    pub kind: OperationKind,
    /// The terminal status.
    pub status: TerminalStatus,
    /// The operation error, when the operation failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<OperationError>,
    /// The lane tip the operation started from.
    pub from_tip_id: Option<String>,
    /// The lane tip the operation settled on.
    pub tip_id: Option<String>,
    /// Unix epoch milliseconds when the operation was admitted.
    pub started_at: i64,
    /// Unix epoch milliseconds when the operation settled.
    pub ended_at: i64,
}

/// The operation kinds, upstream's `OperationMeta["intent"]["kind"]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OperationKind {
    /// A model run.
    #[serde(rename = "run")]
    Run,
    /// A compaction.
    #[serde(rename = "compaction")]
    Compaction,
    /// A navigation.
    #[serde(rename = "navigation")]
    Navigation,
}

/// The continuation a checkpoint recorded, upstream's `Continuation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum Continuation {
    /// The run continues with an assistant request.
    #[serde(rename = "need_assistant")]
    NeedAssistant {
        /// Whether overflow recovery already replaced the request.
        overflow_recovery_used: bool,
    },
    /// The run may finish.
    #[serde(rename = "may_finish")]
    MayFinish {
        /// Whether the final assistant message joins the context.
        include_final_assistant: bool,
    },
}

/// The checkpoint payload, upstream's `CheckpointData`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointData {
    /// The recorded continuation.
    pub continuation: Continuation,
    /// The entry the checkpoint triggers from.
    pub trigger_entry_id: String,
}

/// The queued item kinds, upstream's `InboxItemKind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum InboxItemKind {
    /// A steering message.
    #[serde(rename = "steer")]
    Steer,
    /// A follow-up message.
    #[serde(rename = "followUp")]
    FollowUp,
    /// A next-run message.
    #[serde(rename = "nextRun")]
    NextRun,
    /// A queued write.
    #[serde(rename = "write")]
    Write,
}

/// One queued inbox item, upstream's `InboxItem`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InboxItem {
    /// The inbox entry id.
    pub entry_id: String,
    /// The queue the item sits in.
    pub kind: InboxItemKind,
}

/// The durable per-lane runtime state, upstream's `LaneState`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaneState {
    /// The live operation id, when one is admitted.
    pub current_operation_id: Option<String>,
    /// The last operation id, when one settled.
    pub last_operation_id: Option<String>,
    /// The queued items.
    pub inbox: Vec<InboxItem>,
}

/// A pending entry the session writes before its first commit, upstream's
/// `PendingEntry`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum PendingEntry {
    /// A transcript message, wire `"type": "message"`.
    #[serde(rename = "message")]
    Message {
        /// The message payload.
        payload: Box<AgentMessage>,
    },
    /// An application-defined entry, wire `"type": "custom"`.
    #[serde(rename = "custom")]
    Custom {
        /// The application's custom type discriminator.
        custom_type: String,
        /// The application-defined payload.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        payload: Option<JsonValue>,
    },
}

/// The retry policy fields the generation context normalizes to, upstream's
/// `NormalizedRetryPolicy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NormalizedRetryPolicy {
    /// The attempt budget.
    pub max_attempts: u32,
    /// The base delay in milliseconds.
    pub base_delay_ms: u64,
    /// The agent-level delay cap in ms.
    pub max_agent_delay_ms: u64,
}

/// The generation inputs one assistant step carries, upstream's
/// `GenerationContext`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationContext {
    /// The durable step id.
    pub step_id: String,
    /// The entry the step triggers from.
    pub trigger_entry_id: String,
    /// The lane configuration the step runs under.
    pub configuration: LaneConfiguration,
    /// The snapshotted stream options.
    pub stream_options: crate::harness::types::AgentHarnessStreamOptions,
    /// The normalized retry policy.
    pub retry_policy: NormalizedRetryPolicy,
    /// Whether overflow recovery already replaced the request.
    pub overflow_recovery_used: bool,
}

/// One planned or executed tool call in a batch, upstream's `ToolCall`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    /// Zero-based index in the assistant message's complete content array,
    /// not a filtered tool-call ordinal.
    pub source_index: u64,
    /// The reserved result-entry id.
    pub result_entry_id: String,
    /// The call's phase.
    #[serde(flatten)]
    status: ToolCallStatus,
}

/// The tool-call phase, upstream's `ToolCall["status"]` union.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ToolCallStatus {
    /// Planned but not yet admitted.
    #[serde(rename = "planned")]
    Planned,
    /// Effect dispatched; its outcome is unknown.
    #[serde(rename = "effect_pending")]
    EffectPending {
        /// The declared replay policy.
        replay: crate::types::ToolReplay,
    },
    /// The effect returned an outcome that has not committed yet.
    #[serde(rename = "outcome_ready")]
    OutcomeReady {
        /// Whether the call terminates the run.
        terminate: bool,
    },
    /// The result committed.
    #[serde(rename = "completed")]
    Completed {
        /// Whether the call terminated the run.
        terminate: bool,
    },
}

/// One durable tool batch, upstream's `ToolBatch`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolBatch {
    /// The assistant entry the batch runs from.
    pub assistant_entry_id: String,
    /// The lane configuration the batch runs under.
    pub configuration: LaneConfiguration,
    /// The invocation-local turn id.
    pub turn_id: String,
    /// The batch's calls.
    pub calls: Vec<ToolCall>,
}

/// The inputs one summary generation carries, upstream's `SummaryContext`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryContext {
    /// The reserved summary entry id.
    pub result_entry_id: String,
    /// The lane configuration the summary runs under.
    pub configuration: LaneConfiguration,
    /// The snapshotted stream options.
    pub stream_options: crate::harness::types::AgentHarnessStreamOptions,
    /// The normalized retry policy.
    pub retry_policy: NormalizedRetryPolicy,
}

/// The lane settings one operation snapshots, upstream's `RunSettings`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunSettings {
    /// The compaction settings snapshot.
    pub compaction: CompactionSettings,
    /// The steering queue mode.
    pub steering_mode: QueueMode,
    /// The follow-up queue mode.
    pub follow_up_mode: QueueMode,
    /// The tool execution mode.
    pub tool_execution: ToolExecutionMode,
}

/// The uniform scope every operation leaf carries, upstream's
/// `OperationScope`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationScope {
    /// The cancellation control.
    pub control: Control,
    /// The settings snapshot.
    pub settings: RunSettings,
    /// The latest assistant entry id, when one committed.
    pub latest_assistant_entry_id: Option<String>,
}

/// The shared backoff data every retry-wait leaf carries, upstream's
/// `RetryWait`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetryWait {
    /// The next attempt number.
    pub next_attempt: u32,
    /// Unix epoch milliseconds the wait ends at.
    pub not_before: i64,
    /// The error message the attempt failed with.
    pub error_message: String,
}

/// The boundary a summary task settles into, upstream's `ResultBoundary`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum ResultBoundary {
    /// Resume from the recorded checkpoint.
    #[serde(rename = "resume_checkpoint")]
    ResumeCheckpoint {
        /// The checkpoint to resume after.
        resume_after: CheckpointData,
    },
    /// Finish the operation.
    #[serde(rename = "finish")]
    Finish,
    /// Commit the prepared navigation.
    #[serde(rename = "commit_navigation")]
    CommitNavigation {
        /// The target entry id.
        target_id: String,
        /// An optional label for the target.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
}

/// A structural summary task, upstream's `SummaryTask`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryTask {
    /// The durable task id.
    pub task_id: String,
    /// The compaction trigger, when structural.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<CompactionReason>,
    /// Caller instructions for the summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_instructions: Option<String>,
    /// The boundary the task settles into.
    pub boundary: ResultBoundary,
}

/// The compaction trigger, upstream's `"manual" | "threshold" | "overflow"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionReason {
    /// A caller asked for compaction.
    Manual,
    /// The context threshold crossed.
    Threshold,
    /// The request overflowed the context window.
    Overflow,
}

/// The file operations a durable preparation records, upstream's
/// `DurableFileOperations` — sorted vectors on the wire.
pub type DurableFileOperations = crate::harness::compaction::types::FileOperations;

/// The durable structural preparation one summary task persists, upstream's
/// `DurableStructuralPreparation`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum DurableStructuralPreparation {
    /// A compaction preparation, wire `"kind": "compaction"`.
    #[serde(rename = "compaction")]
    Compaction {
        /// Messages summarized into the history summary.
        messages_to_summarize: Vec<AgentMessage>,
        /// Prefix messages summarized separately when compaction splits a
        /// turn.
        turn_prefix_messages: Vec<AgentMessage>,
        /// Recent messages retained after compaction.
        retained_tail: Vec<AgentMessage>,
        /// Whether compaction splits a turn.
        is_split_turn: bool,
        /// Estimated context tokens before compaction.
        tokens_before: i64,
        /// Previous compaction summary used for iterative updates.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        previous_summary: Option<String>,
        /// File operations extracted from summarized history.
        file_ops: DurableFileOperations,
        /// Settings used to prepare compaction.
        settings: CompactionSettings,
    },
    /// A branch summary preparation, wire `"kind": "branch_summary"`.
    #[serde(rename = "branch_summary")]
    BranchSummary {
        /// Messages selected for the branch summary.
        messages: Vec<AgentMessage>,
        /// File operations extracted from the branch.
        file_ops: DurableFileOperations,
        /// Estimated token count for selected messages.
        total_tokens: i64,
    },
}

/// The summary generation context, upstream's `SummaryGenerationScope`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryGenerationScope {
    /// The summary task.
    pub task: SummaryTask,
    /// The summary's generation inputs.
    pub summary_context: SummaryContext,
}

/// The deferred-step scope, upstream's `DeferredScope`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredScope {
    /// The cancellation control and settings snapshot.
    #[serde(flatten)]
    pub scope: OperationScope,
    /// The durable step id.
    pub step_id: String,
    /// The entry the deferred step polls from.
    pub source_entry_id: String,
    /// The poll counter.
    pub poll: u64,
    /// The lane configuration snapshot.
    pub configuration: LaneConfiguration,
    /// The stream options snapshot.
    pub stream_options: crate::harness::types::AgentHarnessStreamOptions,
}

/// The starting leaf, upstream's `StartingOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartingOperation {
    /// The uniform scope.
    pub scope: OperationScope,
}

/// The checkpoint leaf, upstream's `CheckpointOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The recorded checkpoint.
    #[serde(flatten)]
    pub checkpoint: CheckpointData,
}

/// The assistant-ready leaf, upstream's `AssistantReadyOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantReadyOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The generation inputs.
    pub generation_context: GenerationContext,
    /// The next attempt number.
    pub next_attempt: u32,
}

/// The assistant-effect-pending leaf, upstream's
/// `AssistantEffectPendingOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantEffectPendingOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The generation inputs.
    pub generation_context: GenerationContext,
    /// The attempt in flight.
    pub attempt: u32,
    /// The assistant entry id the response commits to.
    pub response_entry_id: String,
    /// The reserved usage row id.
    pub usage_id: String,
    /// The output limit the request intended.
    pub intended_output_limit: u64,
    /// The model's context window.
    pub context_window: u64,
}

/// The assistant-retry-wait leaf, upstream's `AssistantRetryWaitOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantRetryWaitOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The generation inputs.
    pub generation_context: GenerationContext,
    /// The backoff data.
    #[serde(flatten)]
    pub retry_wait: RetryWait,
}

/// The tools leaf, upstream's `ToolsOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The executing batch.
    pub batch: ToolBatch,
}

/// The deferred-suspended leaf, upstream's `DeferredSuspendedOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredSuspendedOperation {
    /// The deferred-step scope.
    #[serde(flatten)]
    pub deferred: DeferredScope,
}

/// The deferred-effect-pending leaf, upstream's
/// `DeferredEffectPendingOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredEffectPendingOperation {
    /// The deferred-step scope.
    #[serde(flatten)]
    pub scope: DeferredScope,
    /// The response entry the poll commits to.
    pub response_entry_id: String,
    /// The reserved usage row id.
    pub usage_id: String,
}

/// The summary-deciding leaf, upstream's `SummaryDecidingOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryDecidingOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The structural task.
    pub task: SummaryTask,
}

/// The summary-ready leaf, upstream's `SummaryReadyOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryReadyOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The structural task and generation inputs.
    #[serde(flatten)]
    pub generation: SummaryGenerationScope,
    /// The next attempt number.
    pub next_attempt: u32,
}

/// The summary-effect-pending leaf, upstream's
/// `SummaryEffectPendingOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryEffectPendingOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The structural task and generation inputs.
    #[serde(flatten)]
    pub generation: SummaryGenerationScope,
    /// The attempt in flight.
    pub attempt: u32,
    /// The effect request in flight, when one is admitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<SummaryEffectRequest>,
    /// The usage rows reserved by in-flight effects.
    pub usage_ids: Vec<String>,
}

/// The in-flight effect request of a summary step, upstream's
/// `SummaryGenerationEffectPending["request"]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryEffectRequest {
    /// The effect index.
    pub index: u64,
    /// The reserved usage row id.
    pub usage_id: String,
}

/// The summary-retry-wait leaf, upstream's `SummaryRetryWaitOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryRetryWaitOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The structural task and generation inputs.
    #[serde(flatten)]
    pub generation: SummaryGenerationScope,
    /// The backoff data.
    #[serde(flatten)]
    pub retry_wait: RetryWait,
}

/// The navigation-ready-to-commit leaf, upstream's
/// `NavigationReadyToCommitOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NavigationReadyToCommitOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The target entry id; `None` targets the branch root.
    pub target_id: Option<String>,
    /// An optional label for the target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// The flat durable operation state machine, upstream's `OperationState`:
/// exactly 13 family-neutral dispatcher leaves.
///
/// Tool batches stay the nested child collection, and cancellation rides
/// [`OperationScope`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "at", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum OperationState {
    /// A just-admitted operation.
    #[serde(rename = "starting")]
    Starting(StartingOperation),
    /// A checkpoint committed.
    #[serde(rename = "checkpoint")]
    Checkpoint(CheckpointOperation),
    /// An assistant request is ready to send.
    #[serde(rename = "assistant.ready")]
    AssistantReady(AssistantReadyOperation),
    /// An assistant effect is in flight.
    #[serde(rename = "assistant.effect_pending")]
    AssistantEffectPending(AssistantEffectPendingOperation),
    /// An assistant attempt is waiting to retry.
    #[serde(rename = "assistant.retry_wait")]
    AssistantRetryWait(AssistantRetryWaitOperation),
    /// Tools are executing.
    #[serde(rename = "tools")]
    Tools(ToolsOperation),
    /// A deferred response is suspended.
    #[serde(rename = "deferred.suspended")]
    DeferredSuspended(DeferredSuspendedOperation),
    /// A deferred poll is in flight.
    #[serde(rename = "deferred.effect_pending")]
    DeferredEffectPending(DeferredEffectPendingOperation),
    /// A structural task is deciding.
    #[serde(rename = "summary.deciding")]
    SummaryDeciding(SummaryDecidingOperation),
    /// A summary request is ready to send.
    #[serde(rename = "summary.ready")]
    SummaryReady(SummaryReadyOperation),
    /// A summary effect is in flight.
    #[serde(rename = "summary.effect_pending")]
    SummaryEffectPending(SummaryEffectPendingOperation),
    /// A summary attempt is waiting to retry.
    #[serde(rename = "summary.retry_wait")]
    SummaryRetryWait(SummaryRetryWaitOperation),
    /// A navigation is ready to commit.
    #[serde(rename = "navigation.ready_to_commit")]
    NavigationReadyToCommit(NavigationReadyToCommitOperation),
}

/// Copies only the uniform operation scope when constructing a successor
/// leaf, upstream's `operationScopeOf`.
#[must_use]
pub fn operation_scope_of(state: &OperationState) -> OperationScope {
    match state {
        OperationState::Starting(op) => op.scope.clone(),
        OperationState::Checkpoint(op) => op.scope.clone(),
        OperationState::AssistantReady(op) => op.scope.clone(),
        OperationState::AssistantEffectPending(op) => op.scope.clone(),
        OperationState::AssistantRetryWait(op) => op.scope.clone(),
        OperationState::Tools(op) => op.scope.clone(),
        OperationState::DeferredSuspended(op) => op.deferred.scope.clone(),
        OperationState::DeferredEffectPending(op) => op.scope.scope.clone(),
        OperationState::SummaryDeciding(op) => op.scope.clone(),
        OperationState::SummaryReady(op) => op.scope.clone(),
        OperationState::SummaryEffectPending(op) => op.scope.clone(),
        OperationState::SummaryRetryWait(op) => op.scope.clone(),
        OperationState::NavigationReadyToCommit(op) => op.scope.clone(),
    }
}

/// The list- and entry-scan read orders, upstream's `"asc" | "desc"`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryScanOrder {
    /// Oldest first; the default.
    #[default]
    Asc,
    /// Newest first.
    Desc,
}

/// A scan cursor positioned at one storage sequence, upstream's
/// `EntryCursor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryCursor {
    /// The sequence the cursor sits at.
    pub seq: u64,
}

/// A session-wide entry query, upstream's `EntryQuery`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryQuery {
    /// Restrict to one entry type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<EntryType>,
    /// Restrict to one custom type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_type: Option<String>,
    /// The read order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<EntryScanOrder>,
    /// The read limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
    /// The cursor to resume from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<EntryCursor>,
}

/// A storage-level entry scan, upstream's `EntryScan`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryScan {
    /// Restrict to one entry type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<EntryType>,
    /// Restrict to one custom type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_type: Option<String>,
    /// The lower sequence bound, inclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_seq: Option<u64>,
    /// The upper sequence bound, inclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_seq: Option<u64>,
    /// The read order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<EntryScanOrder>,
    /// The read limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
}

/// A storage-level usage scan, upstream's `UsageScan`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageScan {
    /// The lower sequence bound, inclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_seq: Option<u64>,
    /// The upper sequence bound, inclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_seq: Option<u64>,
    /// The read order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<EntryScanOrder>,
    /// The read limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
}

/// A branch-path scan, upstream's `BranchScan`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchScan {
    /// The entry the scan starts from, exclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<String>,
    /// Stop when the scan reaches this entry type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_at_type: Option<EntryType>,
    /// Stop when the scan reaches this entry id, exclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_at_id: Option<String>,
    /// Restrict to one entry type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<EntryType>,
    /// Restrict to one custom type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_type: Option<String>,
    /// The scan order; the branch path reads newest-first by default,
    /// upstream's `"newestFirst" | "oldestFirst"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<BranchScanOrder>,
    /// The scan limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
    /// The cursor to resume from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<EntryCursor>,
}

/// The branch-path scan orders, upstream's `"newestFirst" | "oldestFirst"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BranchScanOrder {
    /// Walk from the tip toward the root, wire `"newestFirst"`.
    #[serde(rename = "newestFirst")]
    NewestFirst,
    /// Walk from the root toward the tip, wire `"oldestFirst"`.
    #[serde(rename = "oldestFirst")]
    OldestFirst,
}

/// A branch-path scan with a required start, upstream's
/// `StorageBranchScan = BranchScan & { start: string }`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageBranchScan {
    /// The entry the scan starts from, exclusive.
    pub start: String,
    /// Stop when the scan reaches this entry type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_at_type: Option<EntryType>,
    /// Stop when the scan reaches this entry id, exclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_at_id: Option<String>,
    /// Restrict to one entry type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<EntryType>,
    /// Restrict to one custom type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_type: Option<String>,
    /// The scan order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<BranchScanOrder>,
    /// The scan limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
    /// The cursor to resume from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<EntryCursor>,
}

/// The structural view of one committed entry, upstream's
/// `EntryStructure`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryStructure {
    /// Stable entry id.
    pub id: String,
    /// The parent entry id; `None` at a branch root.
    pub parent_id: Option<String>,
    /// The storage-assigned sequence.
    pub seq: u64,
    /// The storage-assigned timestamp, Unix epoch milliseconds.
    pub timestamp: i64,
    /// The entry type, serialized as `type`.
    #[serde(rename = "type")]
    pub kind: EntryType,
    /// The application's custom type discriminator, when custom.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_type: Option<String>,
}

/// One recorded usage row, upstream's `UsageRow`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageRow {
    /// The usage row id.
    pub id: String,
    /// The storage-assigned sequence.
    pub seq: u64,
    /// The recorded usage.
    pub usage: Usage,
    /// The entry the usage attaches to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_id: Option<String>,
    /// Whether the row adjusts an earlier row rather than recording new
    /// consumption.
    pub adjustment: bool,
    /// Implementation-specific details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
}

/// A usage row before storage assigns its sequence, upstream's
/// `Omit<UsageRow, "seq">`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageWriteRow {
    /// The usage row id.
    pub id: String,
    /// The recorded usage.
    pub usage: Usage,
    /// The entry the usage attaches to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_id: Option<String>,
    /// Whether the row adjusts an earlier row rather than recording new
    /// usage.
    pub adjustment: bool,
    /// Implementation-specific usage details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
}

/// The session totals, upstream's `SessionStats`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStats {
    /// The number of transcript message entries.
    pub message_count: u64,
    /// The accumulated usage.
    pub usage: Usage,
}

/// The result of one committed transaction, upstream's `CommitResult`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommitResult {
    /// The first sequence the commit assigned.
    pub first_seq: u64,
    /// Every sequence the commit assigned, in write order.
    pub seqs: Vec<u64>,
    /// The storage-assigned timestamp, Unix epoch milliseconds.
    pub timestamp: i64,
    /// Session totals immediately after the commit applied.
    pub stats: SessionStats,
}

/// The metadata every session carries, upstream's `SessionMetadata`.
///
/// Upstream parameterizes `Session<TMetadata>` over an extension of this
/// shape; the contract traits erase to this default so `dyn Session`
/// stays object-safe, and backends downcast at their own seam.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMetadata {
    /// The session id.
    pub id: String,
    /// Unix epoch milliseconds when the session was created.
    pub created_at: i64,
    /// The storage layout version.
    pub storage_version: u32,
    /// The working directory the session was created in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// The parent session id, when forked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    /// The legacy parent session path, when forked from a file-backed
    /// ancestor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_parent_session_path: Option<String>,
}

/// The id generator a session allocates entry and operation ids with,
/// upstream's `IdGenerator`.
pub trait IdGenerator: Send + Sync {
    /// The next id; the timestamp, when supplied, seeds generators that
    /// derive sortable ids from time.
    fn next(&self, timestamp_ms: Option<i64>) -> String;
}

/// The callback-scoped mutation capability, upstream's
/// `SessionMutator = Omit<SessionMutation, "end">`: exactly one commit
/// attempt, no authority to release the session barrier.
pub trait SessionMutator: Send + Sync {
    /// The single commit attempt; a second attempt errors, upstream's
    /// `commit`.
    ///
    /// # Errors
    /// A `SessionError` when the commit fails or the capability already
    /// attempted one.
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>>;
}

/// The exclusive mutation capability, upstream's `SessionMutation`: adds
/// releasing the barrier.
pub trait SessionMutation: SessionMutator {
    /// Waits for any commit attempt, invalidates the capability, and
    /// releases the barrier, upstream's `end`.
    ///
    /// # Errors
    /// A `SessionError` when the barrier cannot be released.
    fn end(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>>;
}

/// The callback `Session::mutate` runs, upstream's `SessionMutationCallback`.
///
/// Upstream's type parameter `T` erases to `Box<dyn Any + Send>`, per the
/// contract-erasure decision recorded on the harness-foundations child:
/// callers downcast the boxed result. Upstream's one-shot contract restates
/// as a shared reference because the returned future borrows the
/// capability; the session runtime enforces the single call.
pub type SessionMutationCallback = Box<
    dyn for<'a> Fn(&'a dyn SessionMutator, &'a Context) -> BoxedFuture<'a, Box<dyn Any + Send>>
        + Send,
>;

/// One named branch's write surface, upstream's `Branch`.
pub trait Branch: Send + Sync {
    /// The branch name.
    fn name(&self) -> &str;

    /// The branch tip id, when any, upstream's `getTipId`.
    fn get_tip_id(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, SessionError>>;

    /// Scan the branch path, upstream's `findEntries`.
    fn find_entries(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>>;

    /// First match on the branch path, upstream's `findEntry`.
    fn find_entry(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>>;

    /// Append one message at the tip, upstream's `appendMessage`.
    fn append_message(
        &self,
        message: AgentMessage,
        context: &Context,
    ) -> BoxedFuture<'_, Result<String, SessionError>>;

    /// Append one custom entry at the tip, upstream's `appendCustomEntry`.
    fn append_custom_entry(
        &self,
        custom_type: &str,
        data: Option<JsonValue>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<String, SessionError>>;
}

/// The read-only session surface, upstream's `SessionReader`.
///
/// Typed payloads restate as JSON and erased addresses, per the contract-
/// erasure decision recorded on the harness-foundations child; typed
/// sugar lives in [`crate::harness::session::values`].
pub trait SessionReader: Send + Sync {
    /// The named entries, upstream's `getEntries`.
    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<BTreeMap<String, Entry>, SessionError>>;

    /// The session totals, upstream's `getStats`.
    fn get_stats(&self, context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>>;

    /// One stored value by erased address, upstream's `getValue<T>`.
    fn get_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>>;

    /// Values under a prefix address, upstream's `scanValues<T>`.
    fn scan_values(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>>;

    /// One list's elements, upstream's `readList<T>`.
    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>>;

    /// Scan a branch from an explicit entry while this reader capability
    /// remains valid, upstream's `scanBranch`.
    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>>;
}

/// The session contract, upstream's `Session<TMetadata extends
/// SessionMetadata>`, with the metadata erased to [`SessionMetadata`]
/// for object safety.
///
/// Every method carries the chord `Context` explicitly, the harness
/// context seam the foundations child settled.
pub trait Session: SessionReader {
    /// The session metadata, upstream's `metadata: TMetadata`.
    fn metadata(&self) -> &SessionMetadata;

    /// The id generator, upstream's `idGenerator: IdGenerator`.
    fn id_generator(&self) -> &dyn IdGenerator;

    /// One entry by id, upstream's `getEntry`.
    fn get_entry(
        &self,
        id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>>;

    /// The session name, upstream's `getName`.
    fn get_name(&self, context: &Context) -> BoxedFuture<'_, Result<Option<String>, SessionError>>;

    /// One entry's label, upstream's `getLabel`.
    fn get_label(
        &self,
        target_id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, SessionError>>;

    /// A session-wide entry query, upstream's `findEntries`.
    fn find_entries(
        &self,
        query: Option<&EntryQuery>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>>;

    /// First match session-wide, upstream's `findEntry`.
    fn find_entry(
        &self,
        query: Option<&EntryQuery>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>>;

    /// One named branch, upstream's `branch`.
    fn branch(
        &self,
        name: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Box<dyn Branch>>, SessionError>>;

    /// Create one branch at the named entry, upstream's `createBranch`.
    fn create_branch(
        &self,
        name: &str,
        at: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Branch>, SessionError>>;

    /// Take the exclusive mutation capability, upstream's `beginMutation`.
    fn begin_mutation(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn SessionMutation>, SessionError>>;

    /// Run the mutation callback with the callback-scoped capability,
    /// upstream's `mutate<T>`; the typed result restates as
    /// `Box<dyn Any + Send>` callers downcast.
    fn mutate(
        &self,
        mutation: SessionMutationCallback,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Any + Send>, SessionError>>;

    /// Set one value by erased address, upstream's `setValue<T>`.
    fn set_value(
        &self,
        address: &ValueAddress,
        next: JsonValue,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>>;

    /// Delete one value, upstream's `deleteValue<T>`.
    fn delete_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>>;

    /// Append one list element, upstream's `appendList<T>`.
    fn append_list(
        &self,
        address: &ListAddress,
        element: JsonValue,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>>;

    /// Delete one list, upstream's `deleteList<T>`.
    fn delete_list(
        &self,
        address: &ListAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>>;

    /// Set the session name, upstream's `setName`.
    fn set_name(
        &self,
        name: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>>;

    /// Set one entry label, upstream's `setLabel`.
    fn set_label(
        &self,
        target_id: &str,
        label: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>>;

    /// Close the session, upstream's `close`.
    fn close(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>>;
}

/// The durable storage contract, upstream's `Storage`, erased for object
/// safety the same way [`Session`] is.
pub trait Storage: Send + Sync {
    /// Commit one transaction, upstream's `commit`.
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>>;

    /// Named entries, upstream's `getEntries`.
    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<BTreeMap<String, Entry>, SessionError>>;

    /// One stored value, upstream's `getValue<T>`.
    fn get_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>>;

    /// Values under a prefix, upstream's `scanValues<T>`.
    fn scan_values(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>>;

    /// One list's elements, upstream's `readList<T>`.
    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>>;

    /// A branch scan with a required start, upstream's `scanBranch`.
    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>>;

    /// The structural view of a branch scan, upstream's
    /// `scanBranchStructure`.
    fn scan_branch_structure(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<EntryStructure>, SessionError>>;

    /// A flat entry scan, upstream's `scanEntries`.
    fn scan_entries(
        &self,
        query: &EntryScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>>;

    /// A flat usage scan, upstream's `scanUsage`.
    fn scan_usage(
        &self,
        query: &UsageScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<UsageRow>, SessionError>>;

    /// The session totals, upstream's `getStats`.
    fn get_stats(&self, context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>>;

    /// Close the storage, upstream's `close`.
    fn close(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>>;
}

#[cfg(test)]
mod tests;

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

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::types::{AssistantMessage, StopReason, Usage};
use pi_ai::types::BoxedFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::harness::compaction::types::{BranchPreparation, CompactionPreparation, CompactionSettings};
use crate::harness::context::Context;
use crate::harness::session::values::{
    EntryWrite, ListAddress, ListElement, ListReadOptions, ListWrite, UsageWrite,
    ValueAddress, ValueSetWrite, ValueDeleteWrite, ListAppendWrite, ListDeleteWrite,
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
    pub fn from_stop_reason(stop_reason: StopReason) -> Option<Self> {
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
    pub usage: Option<pi_ai::types::Usage>,
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
    pub usage: Option<pi_ai::types::Usage>,
    /// Whether a navigation hook produced this entry.
    pub from_hook: bool,
}

/// The custom entry body, upstream's `CustomEntry`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
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
        body: MessageEntry,
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
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
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
        body: MessageEntry,
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
pub struct EntryProjector(
    pub  Arc<
        dyn for<'a> Fn(&Entry, &'a Context) -> pi_ai::types::BoxedFuture<'a, Option<Vec<AgentMessage>>>
            + Send
            + Sync,
    >,
);

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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
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
#[serde(tag = "status", rename_all = "snake_case")]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationResultRecord {
    /// The durable operation id.
    pub operation_id: String,
    /// The operation kind, upstream's `OperationMeta["intent"]["kind"]`.
    pub kind: OperationIntentKind,
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
#[serde(tag = "kind", rename_all = "snake_case")]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryTask {
    /// The durable task id.
    pub task_id: String,
    /// The compaction trigger, when structural.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<SummaryReason>,
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

/// The summary generation context, upstream's `SummaryGenerationScope`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryGenerationScope {
    /// The summary task.
    pub task: SummaryTask,
    /// The summary's generation inputs.
    pub summary_context: SummaryContext,
}

/// The deferred-step scope, upstream's `DeferredScope`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartingOperation {
    /// The uniform scope.
    pub scope: OperationScope,
}

/// The checkpoint leaf, upstream's `CheckpointOperation`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The recorded checkpoint.
    #[serde(flatten)]
    pub checkpoint: CheckpointData,
}

/// The assistant-ready leaf, upstream's `AssistantReadyOperation`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The executing batch.
    pub batch: ToolBatch,
}

/// The deferred-suspended leaf, upstream's `DeferredSuspendedOperation`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredSuspendedOperation {
    /// The deferred-step scope.
    #[serde(flatten)]
    pub deferred: DeferredScope,
}

/// The deferred-effect-pending leaf, upstream's
/// `DeferredEffectPendingOperation`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryDecidingOperation {
    /// The uniform scope.
    pub scope: OperationScope,
    /// The structural task.
    pub task: SummaryTask,
}

/// The summary-ready leaf, upstream's `SummaryReadyOperation`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

/// The navigation-ready-to-commit leaf, upstream's
/// `NavigationReadyToCommitOperation`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
/// exactly 13 family-neutral dispatcher leaves. Tool batches stay the
/// nested child collection, and cancellation rides [`OperationScope`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
        OperationState::DeferredSuspended(op) => op.scope.scope.clone(),
        OperationState::DeferredEffectPending(op) => op.scope.scope.clone(),
        OperationState::SummaryDeciding(op) => op.scope.clone(),
        OperationState::SummaryReady(op) => op.scope.clone(),
        OperationState::SummaryEffectPending(op) => op.scope.clone(),
        OperationState::SummaryRetryWait(op) => op.scope.clone(),
        OperationState::NavigationReadyToCommit(op) => op.scope.clone(),
    }
}

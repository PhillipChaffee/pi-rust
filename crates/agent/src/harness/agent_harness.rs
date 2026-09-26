//! The harness surface, ported from upstream `src/harness/agent-harness.ts`.
//!
//! Upstream's interfaces restate as traits with boxed-future methods; its
//! discriminated unions restate as enums. The runtime constructor
//! (`AgentHarness = { create: createAgentHarness }`) rides the harness
//! runtime child — this module carries the type surface only.

use std::collections::BTreeMap;

use pi_ai::types::BoxedFuture;
use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, DeferredHandle, ImageContent, Message, Model,
    ToolResultMessage, Usage,
};
use pi_ai::utils::retry::RetryPolicy;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::sync::Arc;

use crate::harness::compaction::types::{
    BranchPreparation, BranchSummaryResult, CompactResult, CompactionSettings,
};
use crate::harness::context::Context;
use crate::harness::session::types::{
    BranchScan, CompactionReason, Entry, EntryProjector, LaneConfiguration, ModelIdentity,
    OperationError, OperationKind, OperationResultRecord, Session, SessionStats, UsageRow,
};
use crate::harness::types::{AgentHarnessStreamOptions, AgentHarnessStreamOptionsPatch};
use crate::types::{
    AgentMessage, AgentToolResult as AgentToolResultAlias, QueueMode, ThinkingLevel,
};

/// The tool result alias the harness surface uses, upstream's
/// `AgentToolResult<unknown>`.
pub type AgentToolResult = AgentToolResultAlias;

/// Convenience-only suspended run observation, constructed when the
/// harness runtime exposes public drive, upstream's `SuspendedRun`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SuspendedRun {
    /// The suspended operation's durable id.
    pub operation_id: String,
    /// The fixed `"suspended"` status.
    pub status: String,
    /// The deferred handle the run suspended on.
    pub deferred: DeferredHandle,
}

/// The run invocation result, upstream's `RunResult`.
///
/// The per-operation error unions restate over the one
/// [`HarnessError`](crate::harness::result::HarnessError) taxonomy; each
/// operation documents the variants it produces, and the taxonomy's
/// exhaustive `match` preserves the closed set the TS unions carry.
pub type RunResult = Result<RunOutcome, crate::harness::result::HarnessError>;

/// The success half of a run, upstream's
/// `OperationResultRecord | SuspendedRun`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunOutcome {
    /// The operation settled.
    Settled(OperationResultRecord),
    /// A suspended run.
    Suspended(SuspendedRun),
}

/// The compaction invocation result, upstream's `CompactionResult`.
pub type CompactionResult = Result<CompactionOutcome, crate::harness::result::HarnessError>;

/// The success half of a compaction, upstream's `{ compaction; run? }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionOutcome {
    /// The compaction's settled record.
    pub compaction: OperationResultRecord,
    /// The follow-up run, when one was admitted.
    pub run: Option<RunFollowUp>,
}

/// The follow-up run a compaction or navigation may admit, upstream's
/// `run?: OperationResultRecord | SuspendedRun`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunFollowUp {
    /// A settled run.
    Settled(OperationResultRecord),
    /// A suspended run.
    Suspended(SuspendedRun),
}

/// The navigation invocation result, upstream's `NavigationResult`.
pub type NavigationResult = Result<NavigationOutcome, crate::harness::result::HarnessError>;

/// The success half of a navigation, upstream's `{ navigation; run? }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NavigationOutcome {
    /// The navigation's settled record.
    pub navigation: OperationResultRecord,
    /// The follow-up run, when one was admitted.
    pub run: Option<RunFollowUp>,
}

/// The resume result, upstream's `ResumeResult`.
pub type ResumeResult = Result<RunOutcome, crate::harness::result::HarnessError>;

/// The queue result, upstream's `QueueResult`.
pub type QueueResult = Result<String, crate::harness::result::HarnessError>;

/// The cancel-queued result, upstream's
/// `CancelQueuedResult = Result<{ kind }, Closed>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CancelQueuedKind {
    /// The item was cancelled.
    #[serde(rename = "cancelled")]
    Cancelled,
    /// The item was already consumed.
    #[serde(rename = "already_consumed")]
    AlreadyConsumed,
    /// The item id did not resolve.
    #[serde(rename = "not_found")]
    NotFound,
}

/// The abort-request success value, upstream's
/// `{ operationId, newlyRequested, steer, followUp }`.
#[derive(Clone, Debug, PartialEq)]
pub struct AbortRequestOutcome {
    /// The operation id the abort was requested for.
    pub operation_id: String,
    /// Whether this call newly requested the abort.
    pub newly_requested: bool,
    /// Messages steered out of the aborted run.
    pub steer: Vec<AgentMessage>,
    /// Follow-up messages left queued.
    pub follow_up: Vec<AgentMessage>,
}

/// The abort result, upstream's `AbortResult`.
pub type AbortResult = Result<AbortOutcome, crate::harness::result::HarnessError>;

/// The abort success value, upstream's
/// `{ operationId, steer, followUp }`.
#[derive(Clone, Debug, PartialEq)]
pub struct AbortOutcome {
    /// The operation id the abort settled.
    pub operation_id: String,
    /// Messages steered out of the aborted run.
    pub steer: Vec<AgentMessage>,
    /// Follow-up messages left queued.
    pub follow_up: Vec<AgentMessage>,
}

/// The abort-request result, upstream's `AbortRequestResult`.
pub type AbortRequestResult = Result<AbortRequestOutcome, crate::harness::result::HarnessError>;

/// Navigation options, upstream's `NavigateOptions`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NavigateOptions {
    /// Whether to summarize the abandoned branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summarize: Option<bool>,
    /// An optional label for the target entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Caller instructions for the optional summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_instructions: Option<String>,
}

/// Compaction options, upstream's `compact`'s
/// `{ customInstructions? } | undefined`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactRequestOptions {
    /// Caller instructions for the summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_instructions: Option<String>,
}

/// The prompt messages a run may replay, upstream's
/// `prompt: AgentMessage | AgentMessage[]` plus the text prompt's images.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptMessagesPayload {
    /// A plain text prompt, with optional images.
    #[serde(rename_all = "camelCase")]
    Text {
        /// The prompt text.
        prompt: String,
        /// The attached images.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        images: Option<Vec<ImageContent>>,
    },
    /// One prebuilt message.
    Message(Box<AgentMessage>),
    /// Several prebuilt messages.
    Messages(Vec<AgentMessage>),
}

/// The operations a lane admits, upstream's `OperationRequest`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OperationRequest {
    /// A model run over a text or prebuilt prompt, wire
    /// `"kind": "prompt"`.
    #[serde(rename = "prompt", rename_all = "camelCase")]
    Prompt {
        /// A caller-supplied operation id, when one is pinned.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        operation_id: Option<String>,
        /// The prompt payload.
        #[serde(flatten)]
        prompt: Box<PromptMessagesPayload>,
    },
    /// A skill invocation, wire `"kind": "skill"`.
    #[serde(rename = "skill", rename_all = "camelCase")]
    Skill {
        /// A caller-supplied operation id, when one is pinned.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        operation_id: Option<String>,
        /// The skill name.
        name: String,
        /// Caller instructions appended to the skill's prompt.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        additional_instructions: Option<String>,
    },
    /// A prompt-template invocation, wire `"kind": "prompt_template"`.
    #[serde(rename = "prompt_template", rename_all = "camelCase")]
    PromptTemplate {
        /// A caller-supplied operation id, when one is pinned.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        operation_id: Option<String>,
        /// The template name.
        name: String,
        /// The template arguments.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<Vec<String>>,
    },
    /// A compaction, wire `"kind": "compaction"`.
    #[serde(rename = "compaction", rename_all = "camelCase")]
    Compaction {
        /// A caller-supplied operation id, when one is pinned.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        operation_id: Option<String>,
        /// Caller instructions for the summary.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        custom_instructions: Option<String>,
    },
    /// A navigation, wire `"kind": "navigation"`.
    #[serde(rename = "navigation", rename_all = "camelCase")]
    Navigation {
        /// A caller-supplied operation id, when one is pinned.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        operation_id: Option<String>,
        /// The target entry id; `None` targets the branch root.
        target_id: Option<String>,
        /// The navigation options.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        options: Option<NavigateOptions>,
    },
}

/// One admitted operation, upstream's `OperationAdmission`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationAdmission {
    /// The durable operation id.
    pub operation_id: String,
    /// The operation kind.
    pub kind: OperationKind,
    /// Unix epoch milliseconds when the operation was admitted.
    pub started_at: i64,
}

/// The admission error union, upstream's `OperationAdmissionError`.
///
/// The taxonomy collapses the union into [`crate::harness::result::HarnessError`];
/// admission produces `lane_busy`, `invalid_message`, `unknown_skill`,
/// `unknown_template`, `nothing_to_compact`, `invalid_navigation`,
/// `unknown_target`, and `closed`.
pub type OperationAdmissionError = crate::harness::result::HarnessError;

/// The admission result, upstream's `OperationAdmissionResult`.
pub type OperationAdmissionResult = Result<OperationAdmission, OperationAdmissionError>;

/// Drive options, upstream's `DriveOptions`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DriveOptions {
    /// The operation to drive.
    pub operation_id: String,
    /// Whether to resolve a retry wait by waiting it out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wait_for_retry: Option<bool>,
    /// Whether to resolve a deferred suspension by polling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poll_deferred: Option<bool>,
}

/// The drive outcomes, upstream's `DriveOutcome`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DriveOutcome {
    /// The operation settled, wire `"kind": "settled"`.
    #[serde(rename = "settled", rename_all = "camelCase")]
    Settled {
        /// The settled record.
        outcome: OperationResultRecord,
    },
    /// The operation waits on a caller decision, wire
    /// `"kind": "waiting"`.
    #[serde(rename = "waiting", rename_all = "camelCase")]
    Waiting {
        /// The waiting operation's id.
        operation_id: String,
        /// Which decision the drive waits on.
        #[serde(flatten)]
        reason: DriveWaitReason,
    },
}

/// The wait reasons a drive outcome carries, upstream's `DriveOutcome`'s
/// `"retry"` and `"deferred"` halves.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum DriveWaitReason {
    /// A retry wait, wire `"reason": "retry"`.
    #[serde(rename = "retry", rename_all = "camelCase")]
    Retry {
        /// Unix epoch milliseconds the retry starts at.
        not_before: i64,
    },
    /// A deferred suspension, wire `"reason": "deferred"`.
    #[serde(rename = "deferred", rename_all = "camelCase")]
    Deferred {
        /// The deferred handle the run suspended on.
        deferred: DeferredHandle,
    },
}

/// The drive result, upstream's `DriveResult`.
pub type DriveResult = Result<DriveOutcome, crate::harness::result::HarnessError>;

/// The watch handle contract, upstream's `WatchHandle<T>`.
pub trait WatchHandle<T>: Send + Sync {
    /// The current snapshot, upstream's `WatchHandle.snapshot`.
    fn snapshot(&self) -> T;

    /// Replaces the snapshot, upstream's `WatchHandle.snapshot` setter.
    fn set_snapshot(&self, snapshot: T);

    /// Starts delivering buffered events to the listener, exactly once,
    /// upstream's `start`.
    fn start(&self, listener: EventListener);

    /// Resnapshots the watch: drop in-flight events, mark the boundary,
    /// then hold new events until the boundary, upstream's `resnapshot`.
    fn resnapshot<'a>(
        &self,
        context: &'a Context,
    ) -> BoxedFuture<'a, Result<T, crate::harness::events::ListenerError>>;

    /// Unsubscribes the watcher, upstream's `unsubscribe`.
    fn unsubscribe(&self);
}

/// The live operation statuses, upstream's `OperationStatus`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationStatus {
    /// The operation runs.
    #[serde(rename = "running")]
    Running,
    /// The operation waits on a caller decision.
    #[serde(rename = "open")]
    Open,
    /// Cancellation was requested.
    #[serde(rename = "aborting")]
    Aborting,
}

/// One admitted operation's identity view, upstream's
/// `CurrentOperationInfo`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CurrentOperationInfo {
    /// The operation id.
    pub id: String,
    /// The operation kind.
    pub kind: OperationKind,
    /// Unix epoch milliseconds when the operation was admitted.
    pub started_at: i64,
    /// The live status.
    pub status: OperationStatus,
    /// The model identity the operation captured, when one was
    /// snapshotted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captured_model: Option<ModelIdentity>,
}

/// The execution view one lane reports, upstream's `LaneExecutionInfo`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaneExecutionInfo {
    /// The lane name.
    pub lane: String,
    /// The lane tip id, when any.
    pub tip_id: Option<String>,
    /// The configured model identity.
    pub configured_model: ModelIdentity,
    /// The live operation, when one is admitted.
    pub current: Option<CurrentOperationInfo>,
    /// The last operation id, when one settled.
    pub last_operation_id: Option<String>,
}

/// One lane's identity view, upstream's `LaneInfo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneInfo {
    /// The lane name.
    pub name: String,
    /// The lane tip id, when any.
    pub tip_id: Option<String>,
    /// The live operation, when one is admitted.
    pub operation: Option<CurrentOperationInfo>,
}

/// A tool snapshot in a lane snapshot, upstream's `LaneSnapshotTool`.
#[derive(Clone, Debug, PartialEq)]
pub enum LaneSnapshotTool {
    /// The call is executing.
    Running {
        /// The tool call id.
        tool_call_id: String,
        /// The tool name.
        tool_name: String,
        /// The call arguments.
        args: JsonValue,
        /// The partial result, when updates arrived.
        result: Option<AgentToolResult>,
    },
    /// The call settled.
    Settled {
        /// The tool call id.
        tool_call_id: String,
        /// The tool name.
        tool_name: String,
        /// The call arguments.
        args: JsonValue,
        /// The final result.
        result: AgentToolResult,
        /// Whether the execution errored.
        is_error: bool,
    },
}

/// One open operation's admission view, upstream's `OpenOperation`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenOperation {
    /// The lane name.
    pub lane: String,
    /// The operation id.
    pub operation_id: String,
    /// The operation kind.
    pub kind: OperationKind,
    /// Unix epoch milliseconds when the operation was admitted.
    pub started_at: i64,
    /// Whether cancellation was requested.
    pub aborting: Option<bool>,
}

/// One queued lane item, upstream's `LaneQueuedItem`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum LaneQueuedItem {
    /// A queued message, wire `"type": "message"`.
    #[serde(rename = "message")]
    Message {
        /// The inbox entry id.
        entry_id: String,
        /// The queue the item sits on.
        kind: crate::harness::session::types::InboxItemKind,
        /// The queued message.
        message: Box<AgentMessage>,
    },
    /// A queued custom write, wire `"type": "custom"`.
    #[serde(rename = "custom")]
    Custom {
        /// The inbox entry id.
        entry_id: String,
        /// The queue the item sits on (`write`).
        kind: crate::harness::session::types::InboxItemKind,
        /// The application's custom type.
        custom_type: String,
        /// The queued data.
        data: Option<JsonValue>,
    },
}

/// One lane's full snapshot, upstream's `LaneSnapshot`.
#[derive(Clone, Debug, PartialEq)]
pub struct LaneSnapshot {
    /// The lane name.
    pub lane: String,
    /// The branch transcript, tip path.
    pub transcript: Vec<Entry>,
    /// The lane tip id, when any.
    pub tip_id: Option<String>,
    /// The last settled record, when any.
    pub last_result: Option<OperationResultRecord>,
    /// The lane configuration.
    pub configuration: LaneConfiguration,
    /// The session totals.
    pub stats: SessionStats,
    /// The live operation view, when one is admitted.
    pub operation: Option<LiveOperationView>,
    /// The queued items.
    pub queues: Vec<LaneQueuedItem>,
    /// Whether the lane faulted.
    pub faulted: bool,
}

/// The live operation view a lane snapshot carries, upstream's
/// `LaneSnapshot["operation"]`.
#[derive(Clone, Debug, PartialEq)]
pub struct LiveOperationView {
    /// The operation id.
    pub id: String,
    /// The operation kind.
    pub kind: OperationKind,
    /// Unix epoch milliseconds when the operation was admitted.
    pub started_at: i64,
    /// The tip the operation started from.
    pub from_tip_id: Option<String>,
    /// The live status.
    pub status: OperationStatus,
    /// The live retry view, when the operation is retrying.
    pub retry: Option<RetryView>,
    /// The live deferred view, when suspended.
    pub deferred: Option<DeferredView>,
    /// The streaming assistant message, when one streams.
    pub streaming_message: Option<AssistantMessage>,
    /// The executing tool calls.
    pub running_tools: Vec<LaneSnapshotTool>,
}

/// The live retry view, upstream's
/// `LaneSnapshot["operation"]["retry"]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetryView {
    /// The attempt number.
    pub attempt: u32,
    /// The attempt budget.
    pub max_attempts: u32,
    /// Unix epoch milliseconds the retry starts at.
    pub next_attempt_at: i64,
}

/// The live deferred view, upstream's
/// `LaneSnapshot["operation"]["deferred"]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredView {
    /// The deferred handle.
    pub handle: DeferredHandle,
    /// The poll counter.
    pub poll: u64,
}

/// The session-level snapshot, upstream's `SessionSnapshot`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionSnapshot {
    /// The lane identity views.
    pub lanes: Vec<LaneInfo>,
    /// Whether the session faulted.
    pub faulted: bool,
}

/// The harness event payloads, upstream's `HarnessEventPayload`.
///
/// The port keeps one flat payload enum with the per-variant data; the
/// lane-scoped/global constraint (which payloads may carry a lane) is the
/// [`HarnessEvent`] wrapper's job, restated as its constructor and the
/// boundary suite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum HarnessEventPayload {
    /// A run invocation admitted, wire `"type": "run_start"`.
    #[serde(rename = "run_start", rename_all = "camelCase")]
    RunStart {
        /// The run id.
        run_id: String,
        /// Unix epoch milliseconds when the run started.
        started_at: i64,
    },
    /// A run resumed, wire `"type": "run_resume"`.
    #[serde(rename = "run_resume", rename_all = "camelCase")]
    RunResume {
        /// The run id.
        run_id: String,
    },
    /// A run suspended on a deferred response, wire `"run_suspend"`.
    #[serde(rename = "run_suspend", rename_all = "camelCase")]
    RunSuspend {
        /// The run id.
        run_id: String,
        /// The deferred handle.
        deferred: DeferredHandle,
        /// The poll counter.
        poll: u64,
    },
    /// Cancellation requested, wire `"type": "operation_abort"`.
    #[serde(rename = "operation_abort", rename_all = "camelCase")]
    OperationAbort {
        /// The operation id.
        operation_id: String,
        /// Messages steered out of the aborted run.
        steer: Vec<AgentMessage>,
        /// Follow-up messages left queued.
        follow_up: Vec<AgentMessage>,
    },
    /// A run ended, wire `"type": "run_end"`.
    #[serde(rename = "run_end", rename_all = "camelCase")]
    RunEnd {
        /// The run id.
        run_id: String,
        /// The tip the run started from.
        from_tip_id: Option<String>,
        /// The tip the run settled on.
        tip_id: Option<String>,
        /// Unix epoch milliseconds when the run ended.
        ended_at: i64,
        /// The terminal status and error.
        #[serde(flatten)]
        status: RunEndStatus,
    },
    /// A fault occurred, wire `"type": "fault"`.
    #[serde(rename = "fault", rename_all = "camelCase")]
    Fault {
        /// The stable fault code.
        code: String,
        /// The fault message.
        message: String,
    },
    /// A hook handler failed, wire `"type": "handler_error"`.
    #[serde(rename = "handler_error", rename_all = "camelCase")]
    HandlerError {
        /// The failure message.
        error: String,
        /// The stack trace, when captured.
        stack: Option<String>,
        /// Which surface failed.
        #[serde(flatten)]
        kind: HandlerErrorKind,
    },
    /// A turn started, wire `"type": "turn_start"`.
    #[serde(rename = "turn_start", rename_all = "camelCase")]
    TurnStart {
        /// The run id.
        run_id: String,
        /// The turn id.
        turn_id: String,
    },
    /// A turn ended, wire `"type": "turn_end"`.
    #[serde(rename = "turn_end", rename_all = "camelCase")]
    TurnEnd {
        /// The run id.
        run_id: String,
        /// The turn id.
        turn_id: String,
        /// The settled assistant message.
        message: AssistantMessage,
        /// The tool results of the batch.
        tool_results: Vec<ToolResultMessage>,
    },
    /// A retry was scheduled, wire `"type": "retry_scheduled"`.
    #[serde(rename = "retry_scheduled", rename_all = "camelCase")]
    RetryScheduled {
        /// The run id.
        run_id: String,
        /// The retryable step kind.
        step: String,
        /// The attempt number.
        attempt: u32,
        /// The attempt budget.
        max_attempts: u32,
        /// The delay in milliseconds.
        delay_ms: u64,
        /// Unix epoch milliseconds the retry starts at.
        not_before: i64,
        /// The error message the attempt failed with.
        error_message: String,
    },
    /// A retry attempt started, wire `"type": "retry_start"`.
    #[serde(rename = "retry_start", rename_all = "camelCase")]
    RetryStart {
        /// The run id.
        run_id: String,
        /// The retryable step kind.
        step: String,
        /// The attempt number.
        attempt: u32,
    },
    /// A retry attempt ended, wire `"type": "retry_end"`.
    #[serde(rename = "retry_end", rename_all = "camelCase")]
    RetryEnd {
        /// The run id.
        run_id: String,
        /// The retryable step kind.
        step: String,
        /// The attempt number.
        attempt: u32,
        /// Whether the attempt succeeded.
        success: bool,
        /// The final error message, when the step gave up.
        final_error: Option<String>,
    },
    /// A message was appended, wire `"type": "message_start"`.
    #[serde(rename = "message_start", rename_all = "camelCase")]
    MessageStart {
        /// The run id, when the message belongs to a run.
        run_id: Option<String>,
        /// The appended message.
        message: AgentMessage,
    },
    /// A message updated mid-stream, wire `"type": "message_update"`.
    #[serde(rename = "message_update", rename_all = "camelCase")]
    MessageUpdate {
        /// The run id.
        run_id: String,
        /// The streaming message.
        message: Box<AgentMessage>,
        /// The upstream stream event.
        event: Box<AssistantMessageEvent>,
        /// The frame the encoder produced, when any.
        frame: Option<pi_ai::utils::assistant_message_frame::AssistantMessageFrame>,
    },
    /// A message settled, wire `"type": "message_end"`.
    #[serde(rename = "message_end", rename_all = "camelCase")]
    MessageEnd {
        /// The run id, when the message belongs to a run.
        run_id: Option<String>,
        /// The settled message.
        message: AgentMessage,
        /// The committed entry id, when committed.
        entry_id: Option<String>,
    },
    /// A tool started, wire `"type": "tool_start"`.
    #[serde(rename = "tool_start", rename_all = "camelCase")]
    ToolStart {
        /// The run id.
        run_id: String,
        /// The turn id.
        turn_id: String,
        /// The tool call id.
        tool_call_id: String,
        /// The tool name.
        tool_name: String,
        /// The call arguments.
        args: JsonValue,
    },
    /// A tool published a partial result, wire `"type": "tool_update"`.
    #[serde(rename = "tool_update", rename_all = "camelCase")]
    ToolUpdate {
        /// The run id.
        run_id: String,
        /// The turn id.
        turn_id: String,
        /// The tool call id.
        tool_call_id: String,
        /// The tool name.
        tool_name: String,
        /// The partial result.
        partial_result: AgentToolResult,
    },
    /// A tool settled, wire `"type": "tool_end"`.
    #[serde(rename = "tool_end", rename_all = "camelCase")]
    ToolEnd {
        /// The run id.
        run_id: String,
        /// The turn id.
        turn_id: String,
        /// The tool call id.
        tool_call_id: String,
        /// The tool name.
        tool_name: String,
        /// The final result.
        result: AgentToolResult,
        /// Whether the execution errored.
        is_error: bool,
        /// Whether the tool asked to terminate.
        terminate: bool,
    },
    /// An entry committed, wire `"type": "entry_added"`.
    #[serde(rename = "entry_added", rename_all = "camelCase")]
    EntryAdded {
        /// The committed entry.
        entry: Entry,
    },
    /// Queues changed, wire `"type": "queue_update"`.
    #[serde(rename = "queue_update", rename_all = "camelCase")]
    QueueUpdate {
        /// The queued items.
        queues: Vec<LaneQueuedItem>,
    },
    /// A value changed, wire `"type": "value_update"`.
    #[serde(rename = "value_update", rename_all = "camelCase")]
    ValueUpdate {
        /// Which value changed.
        #[serde(flatten)]
        kind: ValueUpdateKind,
    },
    /// A configuration property changed, wire `"type": "config_update"`.
    #[serde(rename = "config_update", rename_all = "camelCase")]
    ConfigUpdate {
        /// Which property changed.
        #[serde(flatten)]
        property: ConfigUpdateKind,
    },
    /// Compaction started, wire `"type": "compaction_start"`.
    #[serde(rename = "compaction_start", rename_all = "camelCase")]
    CompactionStart {
        /// The run id.
        run_id: String,
        /// The compaction trigger.
        reason: CompactionReason,
        /// Unix epoch milliseconds when compaction started.
        started_at: i64,
    },
    /// Compaction ended, wire `"type": "compaction_end"`.
    #[serde(rename = "compaction_end", rename_all = "camelCase")]
    CompactionEnd {
        /// The run id.
        run_id: String,
        /// The compaction trigger.
        reason: CompactionReason,
        /// Unix epoch milliseconds when compaction ended.
        ended_at: i64,
        /// The terminal status and error.
        #[serde(flatten)]
        status: CompactionEndStatus,
    },
    /// A navigation started, wire `"type": "navigation_start"`.
    #[serde(rename = "navigation_start", rename_all = "camelCase")]
    NavigationStart {
        /// The run id.
        run_id: String,
        /// The target entry id.
        target_id: Option<String>,
        /// Unix epoch milliseconds when the navigation started.
        started_at: i64,
    },
    /// A navigation ended, wire `"type": "navigation_end"`.
    #[serde(rename = "navigation_end", rename_all = "camelCase")]
    NavigationEnd {
        /// The run id.
        run_id: String,
        /// The tip the navigation started from.
        from_tip_id: Option<String>,
        /// The tip the navigation settled on.
        tip_id: Option<String>,
        /// Unix epoch milliseconds when the navigation ended.
        ended_at: i64,
        /// The terminal status and error.
        #[serde(flatten)]
        status: NavigationEndStatus,
    },
    /// A lane was created, wire `"type": "lane_created"`.
    #[serde(rename = "lane_created", rename_all = "camelCase")]
    LaneCreated {
        /// The acquisition point, when one was requested.
        at: Option<String>,
    },
    /// Usage was recorded, wire `"type": "usage"`.
    #[serde(rename = "usage", rename_all = "camelCase")]
    Usage {
        /// The lane that recorded the usage.
        lane: String,
        /// The usage row.
        row: UsageRow,
        /// The totals after the row.
        totals: Usage,
    },
}

/// The run-end status, upstream's `run_end` status union.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RunEndStatus {
    /// The run completed.
    #[serde(rename = "completed")]
    Completed,
    /// The run was aborted.
    #[serde(rename = "aborted")]
    Aborted,
    /// The run failed.
    #[serde(rename = "failed")]
    Failed {
        /// The operation error.
        error: OperationError,
    },
}

/// The handler-error surface, upstream's `handler_error` kind union.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HandlerErrorKind {
    /// A hook handler failed, wire `"kind": "hook"`.
    #[serde(rename = "hook")]
    Hook {
        /// The hook name.
        hook: String,
    },
    /// An event listener failed, wire `"kind": "event"`.
    #[serde(rename = "event")]
    Event {
        /// The event type.
        event: String,
    },
}

/// The value-update discriminator, upstream's `value_update` union.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "value", rename_all = "snake_case")]
pub enum ValueUpdateKind {
    /// The session name changed, wire `"value": "session_name"`.
    #[serde(rename = "session_name", rename_all = "camelCase")]
    SessionName {
        /// The new name, `None` when cleared.
        name: Option<String>,
    },
    /// An entry label changed, wire `"value": "entry_label"`.
    #[serde(rename = "entry_label", rename_all = "camelCase")]
    EntryLabel {
        /// The labeled entry id.
        target_id: String,
        /// The new label, `None` when cleared.
        label: Option<String>,
    },
}

/// The lane-scoped config updates, upstream's
/// `LaneConfigEventPayload`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LaneConfigUpdate {
    /// The model changed, wire `"property": "model"`.
    #[serde(rename = "model", rename_all = "camelCase")]
    Model {
        /// The new identity.
        value: ModelIdentity,
        /// The previous value.
        previous: Option<ModelIdentity>,
    },
    /// The thinking level changed, wire `"property": "thinkingLevel"`.
    #[serde(rename = "thinkingLevel", rename_all = "camelCase")]
    ThinkingLevel {
        /// The new level.
        value: ThinkingLevel,
        /// The previous level.
        previous: ThinkingLevel,
    },
    /// The active tools changed, wire `"property": "activeTools"`.
    #[serde(rename = "activeTools", rename_all = "camelCase")]
    ActiveTools {
        /// The new set.
        value: Vec<String>,
        /// The previous set.
        previous: Vec<String>,
    },
}

/// The session-level config updates, upstream's
/// `GlobalConfigEventPayload`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GlobalConfigUpdate {
    /// The tools changed, wire `"property": "tools"`.
    Tools,
    /// The resources changed, wire `"property": "resources"`.
    Resources,
    /// The stream options changed, wire `"property": "streamOptions"`.
    #[serde(rename = "streamOptions", rename_all = "camelCase")]
    StreamOptions {
        /// The new options.
        value: AgentHarnessStreamOptions,
        /// The previous options.
        previous: AgentHarnessStreamOptions,
    },
    /// The retry policy changed, wire `"property": "retryPolicy"`.
    #[serde(rename = "retryPolicy", rename_all = "camelCase")]
    RetryPolicy {
        /// The new policy.
        value: RetryPolicy,
        /// The previous policy.
        previous: RetryPolicy,
    },
    /// The compaction settings changed, wire
    /// `"property": "compactionSettings"`.
    #[serde(rename = "compactionSettings", rename_all = "camelCase")]
    CompactionSettings {
        /// The new settings.
        value: CompactionSettings,
        /// The previous settings.
        previous: CompactionSettings,
    },
    /// The steering mode changed, wire `"property": "steeringMode"`.
    #[serde(rename = "steeringMode", rename_all = "camelCase")]
    SteeringMode {
        /// The new mode.
        value: QueueMode,
        /// The previous mode.
        previous: QueueMode,
    },
    /// The follow-up mode changed, wire `"property": "followUpMode"`.
    #[serde(rename = "followUpMode", rename_all = "camelCase")]
    FollowUpMode {
        /// The new mode.
        value: QueueMode,
        /// The previous mode.
        previous: QueueMode,
    },
}

/// The config-update discriminator, upstream's `ConfigEventPayload`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigUpdateKind {
    /// A lane-scoped property.
    Lane(LaneConfigUpdate),
    /// A session-level property.
    Global(GlobalConfigUpdate),
}

/// The compaction-end status, upstream's `compaction_end` status union.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CompactionEndStatus {
    /// The compaction committed its entry.
    #[serde(rename = "completed", rename_all = "camelCase")]
    Completed {
        /// The compaction entry id.
        entry_id: String,
    },
    /// The compaction declined.
    #[serde(rename = "declined")]
    Declined,
    /// The compaction aborted.
    #[serde(rename = "aborted")]
    Aborted,
    /// The compaction failed.
    #[serde(rename = "failed", rename_all = "camelCase")]
    Failed {
        /// The operation error.
        error: OperationError,
    },
}

/// The navigation-end status, upstream's `navigation_end` status union.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum NavigationEndStatus {
    /// The navigation completed, declined, or aborted; the status field
    /// carries which.
    #[serde(rename = "completed")]
    Completed,
    /// The navigation declined.
    #[serde(rename = "declined")]
    Declined,
    /// The navigation aborted.
    #[serde(rename = "aborted")]
    Aborted,
    /// The navigation failed.
    #[serde(rename = "failed", rename_all = "camelCase")]
    Failed {
        /// The operation error.
        error: OperationError,
    },
}

/// One delivered harness event, upstream's `HarnessEvent`.
///
/// Upstream types the lane-scoped/global constraint into the union (lane
/// events carry `lane`, global ones cannot, handler errors may); the port
/// restates it as [`HarnessEvent::lane_scoped`], which rejects global
/// payloads with a lane, and the boundary suite pins the split.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessEvent {
    /// The lane the event belongs to, when lane-scoped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<String>,
    /// Whether the event replays recovered work.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub recovery: bool,
    /// The payload.
    #[serde(flatten)]
    pub payload: HarnessEventPayload,
}

impl HarnessEvent {
    /// Builds a lane-scoped event, upstream's lane-carrying variants.
    ///
    /// # Errors
    /// A construction error when the payload is one of the harness-global
    /// payloads (`fault`, `value_update`, `usage`, or a global
    /// `config_update`), which cannot carry a lane.
    pub fn lane_scoped(
        lane: impl Into<String>,
        recovery: bool,
        payload: HarnessEventPayload,
    ) -> Result<Self, String> {
        if !payload.is_lane_scoped() {
            return Err(format!(
                "{} events are harness-global and cannot carry a lane",
                payload.event_type().as_str()
            ));
        }
        Ok(Self {
            lane: Some(lane.into()),
            recovery,
            payload,
        })
    }

    /// Builds a harness-global event, upstream's lane-less variants.
    ///
    /// # Errors
    /// A construction error when the payload is lane-scoped.
    pub fn global(payload: HarnessEventPayload) -> Result<Self, String> {
        if payload.is_lane_scoped() {
            return Err(format!(
                "{} events are lane-scoped and must carry a lane",
                payload.event_type().as_str()
            ));
        }
        Ok(Self {
            lane: None,
            recovery: false,
            payload,
        })
    }

    /// The event's type discriminator, upstream's
    /// `HarnessEvent["type"]`.
    #[must_use]
    pub const fn event_type(&self) -> HarnessEventType {
        self.payload.event_type()
    }
}

impl HarnessEventPayload {
    /// Whether the payload belongs to a lane, upstream's
    /// `LaneEventPayload` membership.
    #[must_use]
    pub const fn is_lane_scoped(&self) -> bool {
        !matches!(
            self,
            Self::Fault { .. }
                | Self::ValueUpdate { .. }
                | Self::Usage { .. }
                | Self::ConfigUpdate {
                    property: ConfigUpdateKind::Global(_)
                }
        )
    }

    /// The event's type discriminator, upstream's `HarnessEvent["type"]`.
    #[must_use]
    pub const fn event_type(&self) -> HarnessEventType {
        match self {
            Self::RunStart { .. } => HarnessEventType::RunStart,
            Self::RunResume { .. } => HarnessEventType::RunResume,
            Self::RunSuspend { .. } => HarnessEventType::RunSuspend,
            Self::OperationAbort { .. } => HarnessEventType::OperationAbort,
            Self::RunEnd { .. } => HarnessEventType::RunEnd,
            Self::Fault { .. } => HarnessEventType::Fault,
            Self::HandlerError { .. } => HarnessEventType::HandlerError,
            Self::TurnStart { .. } => HarnessEventType::TurnStart,
            Self::TurnEnd { .. } => HarnessEventType::TurnEnd,
            Self::RetryScheduled { .. } => HarnessEventType::RetryScheduled,
            Self::RetryStart { .. } => HarnessEventType::RetryStart,
            Self::RetryEnd { .. } => HarnessEventType::RetryEnd,
            Self::MessageStart { .. } => HarnessEventType::MessageStart,
            Self::MessageUpdate { .. } => HarnessEventType::MessageUpdate,
            Self::MessageEnd { .. } => HarnessEventType::MessageEnd,
            Self::ToolStart { .. } => HarnessEventType::ToolStart,
            Self::ToolUpdate { .. } => HarnessEventType::ToolUpdate,
            Self::ToolEnd { .. } => HarnessEventType::ToolEnd,
            Self::EntryAdded { .. } => HarnessEventType::EntryAdded,
            Self::QueueUpdate { .. } => HarnessEventType::QueueUpdate,
            Self::ValueUpdate { .. } => HarnessEventType::ValueUpdate,
            Self::ConfigUpdate { .. } => HarnessEventType::ConfigUpdate,
            Self::CompactionStart { .. } => HarnessEventType::CompactionStart,
            Self::CompactionEnd { .. } => HarnessEventType::CompactionEnd,
            Self::NavigationStart { .. } => HarnessEventType::NavigationStart,
            Self::NavigationEnd { .. } => HarnessEventType::NavigationEnd,
            Self::LaneCreated { .. } => HarnessEventType::LaneCreated,
            Self::Usage { .. } => HarnessEventType::Usage,
        }
    }
}

/// The event type discriminators, upstream's
/// `HarnessEventType = HarnessEvent["type"]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarnessEventType {
    /// A run invocation admitted.
    RunStart,
    /// A run resumed.
    RunResume,
    /// A run suspended on a deferred response.
    RunSuspend,
    /// Cancellation requested.
    OperationAbort,
    /// A run ended.
    RunEnd,
    /// A fault occurred.
    Fault,
    /// A hook handler or event listener failed.
    HandlerError,
    /// A turn started.
    TurnStart,
    /// A turn ended.
    TurnEnd,
    /// A retry was scheduled.
    RetryScheduled,
    /// A retry attempt started.
    RetryStart,
    /// A retry attempt ended.
    RetryEnd,
    /// A message was appended.
    MessageStart,
    /// A message updated mid-stream.
    MessageUpdate,
    /// A message settled.
    MessageEnd,
    /// A tool started.
    ToolStart,
    /// A tool published a partial result.
    ToolUpdate,
    /// A tool settled.
    ToolEnd,
    /// An entry committed.
    EntryAdded,
    /// Queues changed.
    QueueUpdate,
    /// A value changed.
    ValueUpdate,
    /// A configuration property changed.
    ConfigUpdate,
    /// Compaction started.
    CompactionStart,
    /// Compaction ended.
    CompactionEnd,
    /// A navigation started.
    NavigationStart,
    /// A navigation ended.
    NavigationEnd,
    /// A lane was created.
    LaneCreated,
    /// Usage was recorded.
    Usage,
}

impl HarnessEventType {
    /// The wire discriminator, upstream's `HarnessEventType` values.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RunStart => "run_start",
            Self::RunResume => "run_resume",
            Self::RunSuspend => "run_suspend",
            Self::OperationAbort => "operation_abort",
            Self::RunEnd => "run_end",
            Self::Fault => "fault",
            Self::HandlerError => "handler_error",
            Self::TurnStart => "turn_start",
            Self::TurnEnd => "turn_end",
            Self::RetryScheduled => "retry_scheduled",
            Self::RetryStart => "retry_start",
            Self::RetryEnd => "retry_end",
            Self::MessageStart => "message_start",
            Self::MessageUpdate => "message_update",
            Self::MessageEnd => "message_end",
            Self::ToolStart => "tool_start",
            Self::ToolUpdate => "tool_update",
            Self::ToolEnd => "tool_end",
            Self::EntryAdded => "entry_added",
            Self::QueueUpdate => "queue_update",
            Self::ValueUpdate => "value_update",
            Self::ConfigUpdate => "config_update",
            Self::CompactionStart => "compaction_start",
            Self::CompactionEnd => "compaction_end",
            Self::NavigationStart => "navigation_start",
            Self::NavigationEnd => "navigation_end",
            Self::LaneCreated => "lane_created",
            Self::Usage => "usage",
        }
    }
}

/// The event listener contract, upstream's `EventListener<TEvent>`: the
/// failure channel restates upstream's throw, and delivery isolates it
/// into a `handler_error` event.
pub type EventListener = Arc<
    dyn for<'a> Fn(
            &'a HarnessEvent,
            &'a Context,
        ) -> BoxedFuture<'a, Result<(), crate::harness::events::ListenerError>>
        + Send
        + Sync,
>;

/// The subscription contract, upstream's `Events`.
pub trait Events: Send + Sync {
    /// Subscribe to one event type; the returned handle unsubscribes,
    /// upstream's `on`.
    fn on(&self, event_type: HarnessEventType, listener: EventListener) -> Subscription;
}

/// The unsubscribe handle `Events::on` returns, upstream's `() => void`.
#[derive(Clone)]
pub struct Subscription {
    unsubscribe: Arc<dyn Fn() + Send + Sync>,
}

impl Subscription {
    /// Wraps one unsubscribe closure.
    #[must_use]
    pub fn new(unsubscribe: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self { unsubscribe }
    }

    /// Unsubscribes the listener; repeat calls are no-ops.
    pub fn unsubscribe(&self) {
        (self.unsubscribe)();
    }
}

impl std::fmt::Debug for Subscription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Subscription(..)")
    }
}

/// Resources made available to explicit invocation methods and
/// system-prompt callbacks, upstream's `Resources`.
pub type Resources = crate::harness::types::AgentHarnessResources;

/// The hook events and results, upstream's `HookMap`.
///
/// Each entry names its hook; the port keeps the per-name payloads as enum
/// variants and the registry aggregates over them with the same
/// fall-closed/fail-open semantics upstream's per-name methods carry.
#[derive(Clone, Debug, PartialEq)]
pub enum HookEvent {
    /// Before a run admits, upstream's `before_run`.
    BeforeRun {
        /// The prompt messages, with earlier hooks' injections folded in.
        prompt: Vec<AgentMessage>,
        /// The harness resources.
        resources: Resources,
    },
    /// Before a drive pass, upstream's `before_drive`.
    BeforeDrive {
        /// The operation kind being driven.
        operation: OperationKind,
    },
    /// Before a run settles, upstream's `before_run_end`.
    BeforeRunEnd {
        /// The run id.
        run_id: String,
        /// The messages the run produced.
        messages: Vec<AgentMessage>,
    },
    /// Before the model context is built, upstream's `transform_context`.
    TransformContext {
        /// The transcript messages.
        messages: Vec<AgentMessage>,
        /// The system prompt.
        system_prompt: String,
    },
    /// Before a provider request, upstream's `before_request`.
    BeforeRequest {
        /// The model the request targets.
        model: Model,
        /// The retryable step kind.
        step: StepKind,
        /// The attempt number.
        attempt: u32,
        /// The snapshotted stream options.
        stream_options: AgentHarnessStreamOptions,
    },
    /// Before the request payload serializes, upstream's `before_payload`.
    BeforePayload {
        /// The model the request targets.
        model: Model,
        /// The request payload.
        payload: JsonValue,
    },
    /// After a response settles, upstream's `after_response`.
    AfterResponse {
        /// The HTTP status, when the transport surfaced one.
        status: Option<u16>,
        /// The response headers, when captured.
        headers: Option<BTreeMap<String, String>>,
        /// The settled assistant message.
        message: crate::harness::session::types::SettledAssistantMessage,
    },
    /// Before a tool executes, upstream's `before_tool`.
    BeforeTool {
        /// The tool call id.
        tool_call_id: String,
        /// The tool name.
        tool_name: String,
        /// The validated arguments.
        args: BTreeMap<String, JsonValue>,
    },
    /// After a tool settles, upstream's `after_tool`.
    AfterTool {
        /// The tool call id.
        tool_call_id: String,
        /// The tool name.
        tool_name: String,
        /// The validated arguments.
        args: BTreeMap<String, JsonValue>,
        /// The tool result content.
        content: Vec<crate::types::AgentToolContent>,
        /// The tool result details.
        details: Option<JsonValue>,
        /// Whether the execution errored.
        is_error: bool,
        /// The usage the execution reported.
        usage: Option<Usage>,
    },
    /// Before a compaction runs, upstream's `before_compaction`.
    BeforeCompaction {
        /// The compaction trigger.
        reason: CompactionReason,
        /// The prepared inputs.
        preparation: crate::harness::compaction::types::CompactionPreparation,
        /// Caller instructions for the summary.
        custom_instructions: Option<String>,
    },
    /// Before a navigation runs, upstream's `before_navigation`.
    BeforeNavigation {
        /// The target entry id.
        target_id: String,
        /// The prepared branch content.
        preparation: BranchPreparation,
        /// Caller instructions for the summary.
        custom_instructions: Option<String>,
    },
}

/// The retryable step kinds, upstream's
/// `before_request["step"]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    /// The assistant request.
    Assistant,
    /// A deferred poll request.
    Deferred,
    /// A compaction summary request.
    Compaction,
    /// A branch-summary request.
    BranchSummary,
}

/// The hook results, upstream's `HookMap[TName]["result"]`.
#[derive(Clone, Debug, PartialEq)]
pub enum HookResult {
    /// `before_run`: injected messages, when any.
    BeforeRun(Option<BeforeRunResult>),
    /// `before_drive`: no result.
    BeforeDrive,
    /// `before_run_end`: a follow-up prompt, when any.
    BeforeRunEnd(Option<FollowUpResult>),
    /// `transform_context`: replacement messages and/or system prompt.
    TransformContext(Option<TransformContextResult>),
    /// `before_request`: a stream-options patch, when any.
    BeforeRequest(Option<BeforeRequestResult>),
    /// `before_payload`: the replacement payload.
    BeforePayload(Option<PayloadResult>),
    /// `after_response`: a replacement message, when any.
    AfterResponse(Option<Box<AfterResponseResult>>),
    /// `before_tool`: argument replacement and/or a block.
    BeforeTool(Option<BeforeToolResult>),
    /// `after_tool`: result patches, when any.
    AfterTool(Option<AfterToolResult>),
    /// `before_compaction`: decline or a replacement compaction.
    BeforeCompaction(Option<CompactionHookResult>),
    /// `before_navigation`: decline or a replacement summary.
    BeforeNavigation(Option<NavigationHookResult>),
}

/// `before_run`'s result, upstream's `{ messages? }`.
#[derive(Clone, Debug, PartialEq)]
pub struct BeforeRunResult {
    /// Messages injected ahead of the prompt.
    pub messages: Vec<AgentMessage>,
}

/// `before_run_end`'s result, upstream's `{ followUp? }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FollowUpResult {
    /// The follow-up prompt to run after the settled run.
    pub follow_up: String,
}

/// `transform_context`'s result, upstream's
/// `{ messages?, systemPrompt? }`.
#[derive(Clone, Debug, PartialEq)]
pub struct TransformContextResult {
    /// Replacement messages.
    pub messages: Option<Vec<AgentMessage>>,
    /// Replacement system prompt.
    pub system_prompt: Option<String>,
}

/// `before_request`'s result, upstream's `{ streamOptions? }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeforeRequestResult {
    /// The stream-options patch to apply.
    pub stream_options: AgentHarnessStreamOptionsPatch,
}

/// `before_payload`'s result, upstream's `{ payload }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PayloadResult {
    /// The replacement payload.
    pub payload: JsonValue,
}

/// `after_response`'s result, upstream's `{ message? }`.
#[derive(Clone, Debug, PartialEq)]
pub struct AfterResponseResult {
    /// The replacement settled message.
    pub message: Option<crate::harness::session::types::SettledAssistantMessage>,
}

/// `before_tool`'s result, upstream's
/// `{ args?, block?: { reason, terminate? } }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeforeToolResult {
    /// Replacement arguments.
    pub args: Option<BTreeMap<String, JsonValue>>,
    /// The block verdict, when the hook blocks the call.
    pub block: Option<ToolBlock>,
}

/// The block verdict, upstream's `{ reason, terminate? }`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolBlock {
    /// Why the call is blocked.
    pub reason: String,
    /// Whether blocking terminates the run.
    pub terminate: Option<bool>,
}

/// `after_tool`'s result, upstream's
/// `{ content?, details?, isError?, usage?, terminate? }`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AfterToolResult {
    /// Replacement content.
    pub content: Option<Vec<crate::types::AgentToolContent>>,
    /// Replacement details.
    pub details: Option<JsonValue>,
    /// Replacement error flag.
    pub is_error: Option<bool>,
    /// Replacement usage.
    pub usage: Option<Usage>,
    /// Replacement terminate hint.
    pub terminate: Option<bool>,
}

/// `before_compaction`'s result, upstream's
/// `{ decline?, compaction? }`.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionHookResult {
    /// Whether to decline the compaction.
    pub decline: Option<bool>,
    /// A hook-produced compaction result.
    pub compaction: Option<CompactResult>,
}

/// `before_navigation`'s result.
#[derive(Clone, Debug, PartialEq)]
pub struct NavigationHookResult {
    /// Whether to decline the navigation.
    pub decline: Option<bool>,
    /// A hook-produced summary.
    pub summary: Option<BranchSummaryResult>,
}

/// The hook names, upstream's `HookName`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HookName {
    /// Before a run admits.
    BeforeRun,
    /// Before a drive pass.
    BeforeDrive,
    /// Before a run settles.
    BeforeRunEnd,
    /// Before the model context builds.
    TransformContext,
    /// Before a provider request.
    BeforeRequest,
    /// Before the payload serializes.
    BeforePayload,
    /// After a response settles.
    AfterResponse,
    /// Before a tool executes.
    BeforeTool,
    /// After a tool settles.
    AfterTool,
    /// Before a compaction runs.
    BeforeCompaction,
    /// Before a navigation runs.
    BeforeNavigation,
}

impl HookName {
    /// The wire discriminator, upstream's `HookName` values.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BeforeRun => "before_run",
            Self::BeforeDrive => "before_drive",
            Self::BeforeRunEnd => "before_run_end",
            Self::TransformContext => "transform_context",
            Self::BeforeRequest => "before_request",
            Self::BeforePayload => "before_payload",
            Self::AfterResponse => "after_response",
            Self::BeforeTool => "before_tool",
            Self::AfterTool => "after_tool",
            Self::BeforeCompaction => "before_compaction",
            Self::BeforeNavigation => "before_navigation",
        }
    }
}

/// The hook invocation, upstream's
/// `HookInvocation<TName> = HookMap[TName]["event"] & { lane, runId }`.
#[derive(Clone, Debug, PartialEq)]
pub struct HookInvocation {
    /// The lane the hook fires on.
    pub lane: String,
    /// The durable operation id.
    pub run_id: String,
    /// The per-name payload.
    pub event: HookEvent,
}

/// The hook handler contract, upstream's `HookHandler<TName>`.
pub type HookHandler = Arc<
    dyn for<'a> Fn(
            &'a HookInvocation,
            &'a Context,
        ) -> BoxedFuture<'a, Result<HookResult, HookFailure>>
        + Send
        + Sync,
>;

/// The error a hook handler surfaces, upstream's handler throw.
pub type HookFailure = Box<dyn std::error::Error + Send + Sync>;

/// The hook subscription contract, upstream's `Hooks`.
pub trait Hooks: Send + Sync {
    /// Register one handler; the returned subscription unregisters,
    /// upstream's `on`.
    fn on(&self, name: HookName, handler: HookHandler, options: HookOptions) -> Subscription;
}

/// Hook registration options, upstream's `{ id?: string }`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HookOptions {
    /// Optional registration metadata surfaced in hook telemetry spans.
    pub id: Option<String>,
}

/// The harness options, upstream's
/// `AgentHarnessOptions<TContext>`.
///
/// The tool-context parameterization restates erased; see
/// [`crate::harness::types::ToolContext`].
pub struct AgentHarnessOptions {
    /// The session to attach to.
    pub session: Arc<dyn Session>,
    /// The models runtime.
    pub models: Arc<pi_ai::models::Models>,
    /// The model runs use.
    pub model: Model,
    /// The reasoning level. Defaults to upstream's `off`.
    pub thinking_level: Option<ThinkingLevel>,
    /// The tools active from the start.
    pub active_tool_names: Option<Vec<String>>,
    /// The tools the harness owns.
    pub tools: Option<Vec<crate::harness::types::AgentHarnessTool>>,
    /// The static tool context or provider, upstream's
    /// `toolContext?: TContext | ((context) => TContext | Promise<TContext>)`.
    pub tool_context: Option<crate::harness::types::AgentHarnessToolContextSource>,
    /// The system prompt, static or per-turn.
    pub system_prompt: Option<SystemPromptSource>,
    /// The resources.
    pub resources: Option<Resources>,
    /// The curated stream options.
    pub stream_options: Option<AgentHarnessStreamOptions>,
    /// The retry policy.
    pub retry: Option<RetryPolicy>,
    /// The compaction settings.
    pub compaction: Option<CompactionSettings>,
    /// The steering queue mode.
    pub steering_mode: Option<QueueMode>,
    /// The follow-up queue mode.
    pub follow_up_mode: Option<QueueMode>,
    /// The tool execution mode.
    pub tool_execution: Option<crate::types::ToolExecutionMode>,
    /// The transcript-to-provider conversion; the default is
    /// [`crate::harness::messages::convert_to_llm`].
    pub to_provider_messages: Option<ToProviderMessages>,
    /// The custom-entry projectors, by custom type.
    pub entry_projectors: Option<BTreeMap<String, EntryProjector>>,
}

/// The per-turn system-prompt provider, upstream's
/// `(toolContext, context) => string | Promise<string>`.
pub type PromptProvider = Arc<
    dyn Fn(crate::harness::types::ToolContext, &Context) -> BoxedFuture<'_, String> + Send + Sync,
>;

/// The system-prompt source, upstream's
/// `string | ((toolContext, context) => string | Promise<string>)`.
#[derive(Clone)]
pub enum SystemPromptSource {
    /// One static prompt.
    Static(String),
    /// A per-turn provider over the resolved tool context.
    Provided(PromptProvider),
}

impl std::fmt::Debug for SystemPromptSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static(prompt) => f
                .debug_tuple("SystemPromptSource::Static")
                .field(prompt)
                .finish(),
            Self::Provided(..) => f.write_str("SystemPromptSource::Provided(..)"),
        }
    }
}

/// The transcript-to-provider converter, upstream's `toProviderMessages`.
pub type ToProviderMessages = Arc<
    dyn for<'a> Fn(&'a [AgentMessage], &'a Context) -> BoxedFuture<'a, Vec<Message>> + Send + Sync,
>;

impl std::fmt::Debug for AgentHarnessOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentHarnessOptions")
            .finish_non_exhaustive()
    }
}

/// One lane's operation surface, upstream's `AgentLane`.
///
/// Every method carries the chord `Context` explicitly, the harness
/// context seam the foundations child settled.
pub trait AgentLane: Send + Sync {
    /// The lane name.
    fn name(&self) -> &str;

    /// The lane tip id, upstream's `getTipId`.
    fn get_tip_id(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, LaneOperationError>>;

    /// Scan the branch path, upstream's `findEntries`.
    fn find_entries(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, LaneOperationError>>;

    /// First match on the branch path, upstream's `findEntry`.
    fn find_entry(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, LaneOperationError>>;

    /// Append one message at the tip, upstream's `appendMessage`.
    fn append_message(
        &self,
        message: AgentMessage,
        context: &Context,
    ) -> BoxedFuture<'_, Result<String, LaneOperationError>>;

    /// Append one custom entry at the tip, upstream's `appendCustomEntry`.
    fn append_custom_entry(
        &self,
        custom_type: &str,
        data: Option<JsonValue>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<String, LaneOperationError>>;

    /// One operation's settled record, upstream's `getResult`.
    fn get_result(
        &self,
        operation_id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<OperationResultRecord>, LaneOperationError>>;

    /// Admit one operation, upstream's `accept`.
    fn accept(
        &self,
        request: OperationRequest,
        context: &Context,
    ) -> BoxedFuture<'_, Result<OperationAdmissionResult, LaneOperationError>>;

    /// Drive the current operation, upstream's `drive`.
    fn drive(
        &self,
        options: DriveOptions,
        context: &Context,
    ) -> BoxedFuture<'_, Result<DriveResult, LaneOperationError>>;

    /// Request cancellation of one operation, upstream's `requestAbort`.
    fn request_abort(
        &self,
        operation_id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<AbortRequestResult, LaneOperationError>>;

    /// The execution view, upstream's `inspectExecution`.
    fn inspect_execution(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<LaneExecutionInfo, LaneOperationError>>;

    /// Run a text prompt, upstream's `prompt(text, images)`.
    fn prompt_text(
        &self,
        text: &str,
        images: Option<Vec<ImageContent>>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<RunResult, LaneOperationError>>;

    /// Run a prebuilt message prompt, upstream's
    /// `prompt(message | messages)`.
    fn prompt_messages(
        &self,
        prompt: PromptMessagesPayload,
        context: &Context,
    ) -> BoxedFuture<'_, Result<RunResult, LaneOperationError>>;

    /// Invoke one skill, upstream's `skill`.
    fn skill(
        &self,
        name: &str,
        additional_instructions: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<RunResult, LaneOperationError>>;

    /// Invoke one prompt template, upstream's `promptFromTemplate`.
    fn prompt_from_template(
        &self,
        name: &str,
        args: Option<Vec<String>>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<RunResult, LaneOperationError>>;

    /// Compact the history, upstream's `compact`.
    fn compact(
        &self,
        options: Option<CompactRequestOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CompactionResult, LaneOperationError>>;

    /// Navigate the tree, upstream's `navigateTree`.
    fn navigate_tree(
        &self,
        target_id: Option<String>,
        options: Option<NavigateOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<NavigationResult, LaneOperationError>>;

    /// Resume a suspended run, upstream's `resume`.
    fn resume(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<ResumeResult, LaneOperationError>>;

    /// Abort the active operation, upstream's `abort`.
    fn abort(&self, context: &Context) -> BoxedFuture<'_, Result<AbortResult, LaneOperationError>>;

    /// Queue one steering message, upstream's `steer`.
    fn steer(
        &self,
        message: QueueMessage,
        context: &Context,
    ) -> BoxedFuture<'_, Result<QueueResult, LaneOperationError>>;

    /// Queue one follow-up message, upstream's `followUp`.
    fn follow_up(
        &self,
        message: QueueMessage,
        context: &Context,
    ) -> BoxedFuture<'_, Result<QueueResult, LaneOperationError>>;

    /// Queue one next-run message, upstream's `nextRun`.
    fn next_run(
        &self,
        message: QueueMessage,
        context: &Context,
    ) -> BoxedFuture<'_, Result<QueueResult, LaneOperationError>>;

    /// Cancel one queued item, upstream's `cancelQueued`.
    fn cancel_queued(
        &self,
        entry_id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CancelQueuedKind, CancelQueuedError>>;

    /// Record one usage row, upstream's `recordUsage`.
    fn record_usage(
        &self,
        usage: Usage,
        options: Option<RecordUsageOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<String, CancelQueuedError>>;

    /// Wait for the lane to idle, upstream's `waitForIdle`.
    fn wait_for_idle(&self, context: &Context) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// Run one callback when idle, upstream's `runWhenIdle`.
    fn run_when_idle(
        &self,
        callback: IdleCallback,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// The configured model, upstream's `getModel`.
    fn get_model(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Model>, LaneOperationError>>;

    /// Set the model, upstream's `setModel`.
    fn set_model(
        &self,
        model: ModelIdentity,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// The thinking level, upstream's `getThinkingLevel`.
    fn get_thinking_level(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<ThinkingLevel, LaneOperationError>>;

    /// Set the thinking level, upstream's `setThinkingLevel`.
    fn set_thinking_level(
        &self,
        level: ThinkingLevel,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// The active tools, upstream's `getActiveTools`.
    fn get_active_tools(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<String>, LaneOperationError>>;

    /// Set the active tools, upstream's `setActiveTools`.
    fn set_active_tools(
        &self,
        names: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// Watch the lane, upstream's `watch`.
    fn watch(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn WatchHandle<LaneSnapshot>>, LaneOperationError>>;
}

/// The message a queue method takes, upstream's
/// `message: string | AgentMessage`.
#[derive(Clone, Debug, PartialEq)]
pub enum QueueMessage {
    /// A plain text message.
    Text(String),
    /// A prebuilt message.
    Message(Box<AgentMessage>),
}

/// The images a queue message may carry, upstream's
/// `images: ImageContent[] | undefined`.
pub type QueueImages = Vec<ImageContent>;

/// Options for one usage row, upstream's
/// `{ entryId?, details? }`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecordUsageOptions {
    /// The entry the usage attaches to.
    pub entry_id: Option<String>,
    /// Implementation-specific details.
    pub details: Option<JsonValue>,
}

/// The error a lane operation reports, upstream's the `Result` error
/// unions' members.
///
/// The lane methods collapse the per-operation error unions into one
/// taxonomy; the per-operation unions restate as the
/// [`RunResult`]-family aliases above.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaneOperationError {
    /// The harness closed.
    Closed(crate::harness::result::HarnessError),
}

/// The cancel-queued error, upstream's `Closed` half of
/// `CancelQueuedResult`.
pub type CancelQueuedError = crate::harness::result::HarnessError;

/// The idle callback, upstream's `runWhenIdle`'s callback.
pub type IdleCallback = Arc<dyn Fn(&Context) -> BoxedFuture<'_, ()> + Send + Sync>;

/// Options for acquiring one lane, upstream's `AcquireLaneOptions`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AcquireLaneOptions {
    /// The entry to create the lane at; `None` creates at the current
    /// tip.
    pub create_at: Option<CreateAt>,
}

/// The `createAt` argument, upstream's `string | null`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CreateAt {
    /// Create at the branch root.
    Root,
    /// Create at the named entry.
    Entry(String),
}

/// The harness capability, upstream's `AgentHarness`.
pub trait AgentHarness: Send + Sync {
    /// Acquire one lane by name, upstream's `lane(name)`.
    fn lane(
        &self,
        name: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Arc<dyn AgentLane>, LaneOperationError>>;

    /// Acquire one lane with options, upstream's `lane(name, options)`.
    fn lane_with_options(
        &self,
        name: &str,
        options: AcquireLaneOptions,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Arc<dyn AgentLane>, LaneOperationError>>;

    /// The lane identity views, upstream's `lanes`.
    fn lanes(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<LaneInfo>, LaneOperationError>>;

    /// The session name, upstream's `getName`.
    fn get_name(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, LaneOperationError>>;

    /// Set the session name, upstream's `setName`.
    fn set_name(
        &self,
        name: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// One entry's label, upstream's `getLabel`.
    fn get_label(
        &self,
        target_id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, LaneOperationError>>;

    /// Set one entry label, upstream's `setLabel`.
    fn set_label(
        &self,
        target_id: &str,
        label: Option<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// The harness tools, upstream's `getTools`.
    fn get_tools(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<crate::harness::types::AgentHarnessTool>, LaneOperationError>>;

    /// Set the harness tools, upstream's `setTools`.
    fn set_tools(
        &self,
        tools: Vec<crate::harness::types::AgentHarnessTool>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// The resources, upstream's `getResources`.
    fn get_resources(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Resources, LaneOperationError>>;

    /// Set the resources, upstream's `setResources`.
    fn set_resources(
        &self,
        resources: Resources,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// The stream options, upstream's `getStreamOptions`.
    fn get_stream_options(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<AgentHarnessStreamOptions, LaneOperationError>>;

    /// Set the stream options, upstream's `setStreamOptions`.
    fn set_stream_options(
        &self,
        options: AgentHarnessStreamOptions,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// The retry policy, upstream's `getRetryPolicy`.
    fn get_retry_policy(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<RetryPolicy, LaneOperationError>>;

    /// Set the retry policy, upstream's `setRetryPolicy`.
    fn set_retry_policy(
        &self,
        policy: RetryPolicy,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// The compaction settings, upstream's `getCompactionSettings`.
    fn get_compaction_settings(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CompactionSettings, LaneOperationError>>;

    /// Set the compaction settings, upstream's `setCompactionSettings`.
    fn set_compaction_settings(
        &self,
        settings: CompactionSettings,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// The steering mode, upstream's `getSteeringMode`.
    fn get_steering_mode(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<QueueMode, LaneOperationError>>;

    /// Set the steering mode, upstream's `setSteeringMode`.
    fn set_steering_mode(
        &self,
        mode: QueueMode,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// The follow-up mode, upstream's `getFollowUpMode`.
    fn get_follow_up_mode(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<QueueMode, LaneOperationError>>;

    /// Set the follow-up mode, upstream's `setFollowUpMode`.
    fn set_follow_up_mode(
        &self,
        mode: QueueMode,
        context: &Context,
    ) -> BoxedFuture<'_, Result<(), LaneOperationError>>;

    /// Watch the session, upstream's `watchSession`.
    fn watch_session(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn WatchHandle<SessionSnapshot>>, LaneOperationError>>;

    /// The hook registry, upstream's `hooks`.
    fn hooks(&self) -> &dyn Hooks;

    /// The event bus, upstream's `events`.
    fn events(&self) -> &dyn Events;

    /// Close the harness, upstream's `close`.
    fn close(&self, context: &Context) -> BoxedFuture<'_, Result<(), LaneOperationError>>;
}

#[cfg(test)]
mod tests;

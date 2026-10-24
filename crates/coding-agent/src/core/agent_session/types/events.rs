//! The session event vocabulary, upstream's `AgentSessionEvent`.

use std::sync::Arc;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::ToolResultMessage;

use crate::compaction::CompactionResult;
use crate::extensions::reasons::CompactionReason;
use crate::session_manager::SessionEntry;

/// The summarization retry's source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SummarizationSource {
    /// A branch summary.
    BranchSummary,
    /// A compaction, carrying its trigger.
    Compaction(CompactionReason),
}

/// A session event: the agent events plus the session-specific ones.
#[expect(
    clippy::large_enum_variant,
    reason = "the variants mirror upstream's union; boxing would heap-allocate every forward"
)]
#[derive(Clone, Debug)]
pub enum AgentSessionEvent {
    /// The agent began processing.
    AgentStart,
    /// The agent finished, with the will-retry envelope.
    AgentEnd {
        /// The messages the run produced.
        messages: Vec<AgentMessage>,
        /// Whether the session will continue with an automatic retry.
        will_retry: bool,
    },
    /// A turn began.
    TurnStart,
    /// A turn ended.
    TurnEnd {
        /// The assistant message that completed the turn.
        message: AgentMessage,
        /// The turn's tool results.
        tool_results: Vec<ToolResultMessage>,
    },
    /// A message entered the transcript.
    MessageStart {
        /// The message that started.
        message: AgentMessage,
    },
    /// An assistant message streamed an update.
    MessageUpdate {
        /// The live partial message.
        message: AgentMessage,
        /// The protocol event that produced the update.
        assistant_message_event: AssistantMessageEvent,
    },
    /// A message reached its final form.
    MessageEnd {
        /// The final message.
        message: AgentMessage,
    },
    /// A tool execution began.
    ToolExecutionStart {
        /// The executing call's id.
        tool_call_id: String,
        /// The executing tool's name.
        tool_name: String,
        /// The validated arguments.
        args: serde_json::Value,
    },
    /// A tool streamed a partial result.
    ToolExecutionUpdate {
        /// The executing call's id.
        tool_call_id: String,
        /// The executing tool's name.
        tool_name: String,
        /// The validated arguments.
        args: serde_json::Value,
        /// The partial result.
        partial_result: pi_agent_core::types::AgentToolResult,
    },
    /// A tool execution finalized.
    ToolExecutionEnd {
        /// The finalized call's id.
        tool_call_id: String,
        /// The finalized tool's name.
        tool_name: String,
        /// The final result.
        result: pi_agent_core::types::AgentToolResult,
        /// Whether the result is an error.
        is_error: bool,
    },
    /// The run and every post-run continuation settled.
    AgentSettled,
    /// The pending steering/follow-up queues changed.
    QueueUpdate {
        /// The pending steering texts.
        steering: Vec<String>,
        /// The pending follow-up texts.
        follow_up: Vec<String>,
    },
    /// Compaction started.
    CompactionStart {
        /// Why compaction runs.
        reason: CompactionReason,
    },
    /// A custom entry landed in the session.
    EntryAppended {
        /// The appended entry.
        entry: SessionEntry,
    },
    /// The session display name changed.
    SessionInfoChanged {
        /// The new name; absent when cleared.
        name: Option<String>,
    },
    /// The thinking level changed.
    ThinkingLevelChanged {
        /// The applied level.
        level: pi_agent_core::types::ThinkingLevel,
    },
    /// Compaction ended.
    CompactionEnd {
        /// Why compaction ran.
        reason: CompactionReason,
        /// The result; absent on failure or cancellation.
        result: Option<CompactionResult>,
        /// Whether the compaction was cancelled.
        aborted: bool,
        /// Whether the interrupted turn retries.
        will_retry: bool,
        /// The failure message, absent on success or cancellation.
        error_message: Option<String>,
    },
    /// An automatic agent-turn retry was scheduled.
    AutoRetryStart {
        /// The attempt number.
        attempt: usize,
        /// The configured maximum.
        max_attempts: usize,
        /// The backoff delay.
        delay_ms: u64,
        /// The failure message that triggered the retry.
        error_message: String,
    },
    /// The automatic retry ended.
    AutoRetryEnd {
        /// Whether a later call completed normally.
        success: bool,
        /// The attempt count.
        attempt: usize,
        /// The final failure message, when unsuccessful.
        final_error: Option<String>,
    },
    /// A summarization retry was scheduled.
    SummarizationRetryScheduled {
        /// The attempt number.
        attempt: u32,
        /// The configured maximum.
        max_attempts: u32,
        /// The backoff delay.
        delay_ms: u64,
        /// The failure message that triggered the retry.
        error_message: String,
    },
    /// A summarization retry attempt started.
    SummarizationRetryAttemptStart {
        /// The summarization source.
        source: SummarizationSource,
    },
    /// The summarization retry loop finished.
    SummarizationRetryFinished,
    /// A bash execution streamed output.
    BashExecutionUpdate {
        /// The caller's optional identifier.
        id: Option<String>,
        /// The streamed delta.
        delta: String,
    },
}

/// A session event listener.
pub type AgentSessionEventListener = Arc<dyn Fn(AgentSessionEvent) + Send + Sync>;

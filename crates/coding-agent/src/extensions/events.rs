//! The extension-event union, upstream's `ExtensionEvent` in
//! `src/core/extensions/types.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Owned payloads; the runner threads them sequentially. Upstream mutates
//! some payloads in place (`before_provider_headers`, `tool_call` input);
//! those restatements pin the mutation through the runner's return or the
//! loop's owned-args contract.

use serde_json::Value;

pub use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::types::{AgentMessage, AgentToolResult};

use pi_ai::types::{AssistantMessageEvent, Model, ProviderHeaders, ToolResultMessage, Usage};

use crate::compaction::CompactionPreparation;
use crate::extensions::reasons::{
    CompactionReason, ForkPosition, InputSource, ModelSelectSource, SessionShutdownReason,
    SessionStartReason, StreamingBehavior, TreePreparation,
};
use crate::session_manager::{BranchSummaryEntry, CompactionEntry, SessionEntry};
use crate::system_prompt::BuildSystemPromptOptions;

/// The extension-event union, upstream's `ExtensionEvent`.
#[allow(
    clippy::large_enum_variant,
    reason = "the session_before_* payloads mirror upstream's shared objects"
)]
#[derive(Clone, Debug)]
pub enum ExtensionEvent {
    /// `project_trust` — the trust gate consults extensions.
    ProjectTrust {
        /// The project cwd.
        cwd: String,
    },
    /// `resources_discover` — extensions contribute resource paths.
    ResourcesDiscover {
        /// The cwd discovery runs against.
        cwd: String,
        /// Why discovery runs.
        reason: SessionStartReason,
    },
    /// `session_start`.
    SessionStart {
        /// Why the session started.
        reason: SessionStartReason,
        /// The file the replaced session wrote, when replaced.
        previous_session_file: Option<String>,
    },
    /// `session_info_changed`.
    SessionInfoChanged {
        /// The new display name, or none when cleared.
        name: Option<String>,
    },
    /// `session_before_switch`.
    SessionBeforeSwitch {
        /// Whether the switch creates or resumes.
        reason: SessionShutdownReason,
        /// The session file the switch targets.
        target_session_file: Option<String>,
    },
    /// `session_before_fork`.
    SessionBeforeFork {
        /// The entry being forked.
        entry_id: String,
        /// Whether the fork positions before or at the entry.
        position: ForkPosition,
    },
    /// `session_before_compact`.
    SessionBeforeCompact {
        /// The compaction preparation.
        preparation: CompactionPreparation,
        /// The branch entries the preparation came from.
        branch_entries: Vec<SessionEntry>,
        /// The caller's custom instructions.
        custom_instructions: Option<String>,
        /// Why compaction runs.
        reason: CompactionReason,
        /// Whether the interrupted turn retries after compaction.
        will_retry: bool,
        /// The signal cancelling the compaction.
        signal: Option<AbortSignal>,
    },
    /// `session_compact`.
    SessionCompact {
        /// The saved compaction entry.
        compaction_entry: CompactionEntry,
        /// Whether an extension supplied the compaction.
        from_extension: bool,
        /// Why compaction ran.
        reason: CompactionReason,
        /// Whether the interrupted turn retries.
        will_retry: bool,
    },
    /// `session_compact_failed`.
    SessionCompactFailed {
        /// Why compaction ran.
        reason: CompactionReason,
        /// The failure message; absent when aborted.
        error_message: Option<String>,
        /// Whether the compaction was cancelled.
        aborted: bool,
        /// Whether the interrupted turn would have retried.
        will_retry: bool,
        /// Whether an extension's compaction failed.
        from_extension: bool,
    },
    /// `session_shutdown`.
    SessionShutdown {
        /// Why the session shuts down.
        reason: SessionShutdownReason,
        /// The file the replacement session will write.
        target_session_file: Option<String>,
    },
    /// `session_before_tree`.
    SessionBeforeTree {
        /// The navigation preparation.
        preparation: TreePreparation,
        /// The signal cancelling the summarization.
        signal: Option<AbortSignal>,
    },
    /// `session_tree`.
    SessionTree {
        /// The leaf after navigation.
        new_leaf_id: Option<String>,
        /// The leaf before navigation.
        old_leaf_id: Option<String>,
        /// The saved branch summary entry, when one was written.
        summary_entry: Option<BranchSummaryEntry>,
        /// Whether an extension supplied the summary; absent without one.
        from_extension: Option<bool>,
    },
    /// `context` — the messages before each LLM call; handlers may replace.
    Context {
        /// The messages being sent.
        messages: Vec<AgentMessage>,
    },
    /// `before_provider_request` — the raw request payload before send.
    BeforeProviderRequest {
        /// The provider request payload.
        payload: Value,
    },
    /// `before_provider_headers` — request headers before send.
    BeforeProviderHeaders {
        /// The request headers; in-place mutation restates to the returned map.
        headers: ProviderHeaders,
    },
    /// `after_provider_response`.
    AfterProviderResponse {
        /// The response status.
        status: Option<u16>,
        /// The response headers.
        headers: ProviderHeaders,
    },
    /// `before_agent_start` — the prompt preflight, before the run starts.
    BeforeAgentStart {
        /// The expanded prompt text.
        prompt: String,
        /// The prompt images.
        images: Option<Vec<pi_ai::types::ImageContent>>,
        /// The base system prompt.
        system_prompt: String,
        /// The options the prompt was built with.
        system_prompt_options: BuildSystemPromptOptions,
        /// The run's signal.
        signal: Option<AbortSignal>,
    },
    /// `agent_start`.
    AgentStart,
    /// `agent_end`.
    AgentEnd {
        /// The messages the run produced.
        messages: Vec<AgentMessage>,
    },
    /// `agent_settled` — the run and every post-run continuation ended.
    AgentSettled,
    /// `turn_start`.
    TurnStart {
        /// The zero-based turn index within the run.
        turn_index: usize,
        /// The wall-clock timestamp, upstream's `Date.now()`.
        timestamp: i64,
    },
    /// `turn_end`.
    TurnEnd {
        /// The zero-based turn index within the run.
        turn_index: usize,
        /// The turn's assistant message.
        message: AgentMessage,
        /// The turn's tool results.
        tool_results: Vec<ToolResultMessage>,
    },
    /// `message_start`.
    MessageStart {
        /// The message that started.
        message: AgentMessage,
    },
    /// `message_update`.
    MessageUpdate {
        /// The live message.
        message: AgentMessage,
        /// The stream event that updated it.
        assistant_message_event: AssistantMessageEvent,
    },
    /// `message_end`.
    MessageEnd {
        /// The finalized message.
        message: AgentMessage,
    },
    /// `tool_execution_start`.
    ToolExecutionStart {
        /// The call id.
        tool_call_id: String,
        /// The tool name.
        tool_name: String,
        /// The validated arguments.
        args: Value,
    },
    /// `tool_execution_update`.
    ToolExecutionUpdate {
        /// The call id.
        tool_call_id: String,
        /// The tool name.
        tool_name: String,
        /// The validated arguments.
        args: Value,
        /// The partial result so far.
        partial_result: AgentToolResult,
    },
    /// `tool_execution_end`.
    ToolExecutionEnd {
        /// The call id.
        tool_call_id: String,
        /// The tool name.
        tool_name: String,
        /// The final result.
        result: AgentToolResult,
        /// Whether the execution failed.
        is_error: bool,
    },
    /// `model_select`.
    ModelSelect {
        /// The selected model.
        model: Model,
        /// The model before the selection.
        previous_model: Option<Model>,
        /// How the selection moved.
        source: ModelSelectSource,
    },
    /// `thinking_level_select`.
    ThinkingLevelSelect {
        /// The applied level.
        level: pi_agent_core::types::ThinkingLevel,
        /// The level before the change.
        previous_level: pi_agent_core::types::ThinkingLevel,
    },
    /// `user_bash` — a user-initiated bash command is about to run.
    UserBash {
        /// The command.
        command: String,
    },
    /// `input` — user input before expansion.
    Input {
        /// The input text.
        text: String,
        /// The input images.
        images: Option<Vec<pi_ai::types::ImageContent>>,
        /// Where the input came from.
        source: InputSource,
        /// The queue behavior while streaming.
        streaming_behavior: Option<StreamingBehavior>,
    },
    /// `tool_call` — a tool call is about to execute.
    ToolCall {
        /// The tool name.
        tool_name: String,
        /// The call id.
        tool_call_id: String,
        /// The raw arguments; in-place mutation restates to the loop's
        /// owned-args contract.
        input: Value,
    },
    /// `tool_result` — a tool result finalized.
    ToolResult {
        /// The tool name.
        tool_name: String,
        /// The call id.
        tool_call_id: String,
        /// The raw arguments.
        input: Value,
        /// The result content.
        content: Vec<pi_ai::types::AssistantBlock>,
        /// The result details.
        details: Value,
        /// Whether the execution failed.
        is_error: bool,
        /// The execution's own usage.
        usage: Option<Usage>,
    },
}

impl ExtensionEvent {
    /// The channel name, upstream's `event.type`.
    #[must_use]
    pub const fn channel(&self) -> &'static str {
        match self {
            Self::ProjectTrust { .. } => "project_trust",
            Self::ResourcesDiscover { .. } => "resources_discover",
            Self::SessionStart { .. } => "session_start",
            Self::SessionInfoChanged { .. } => "session_info_changed",
            Self::SessionBeforeSwitch { .. } => "session_before_switch",
            Self::SessionBeforeFork { .. } => "session_before_fork",
            Self::SessionBeforeCompact { .. } => "session_before_compact",
            Self::SessionCompact { .. } => "session_compact",
            Self::SessionCompactFailed { .. } => "session_compact_failed",
            Self::SessionShutdown { .. } => "session_shutdown",
            Self::SessionBeforeTree { .. } => "session_before_tree",
            Self::SessionTree { .. } => "session_tree",
            Self::Context { .. } => "context",
            Self::BeforeProviderRequest { .. } => "before_provider_request",
            Self::BeforeProviderHeaders { .. } => "before_provider_headers",
            Self::AfterProviderResponse { .. } => "after_provider_response",
            Self::BeforeAgentStart { .. } => "before_agent_start",
            Self::AgentStart => "agent_start",
            Self::AgentEnd { .. } => "agent_end",
            Self::AgentSettled => "agent_settled",
            Self::TurnStart { .. } => "turn_start",
            Self::TurnEnd { .. } => "turn_end",
            Self::MessageStart { .. } => "message_start",
            Self::MessageUpdate { .. } => "message_update",
            Self::MessageEnd { .. } => "message_end",
            Self::ToolExecutionStart { .. } => "tool_execution_start",
            Self::ToolExecutionUpdate { .. } => "tool_execution_update",
            Self::ToolExecutionEnd { .. } => "tool_execution_end",
            Self::ModelSelect { .. } => "model_select",
            Self::ThinkingLevelSelect { .. } => "thinking_level_select",
            Self::UserBash { .. } => "user_bash",
            Self::Input { .. } => "input",
            Self::ToolCall { .. } => "tool_call",
            Self::ToolResult { .. } => "tool_result",
        }
    }
}

#[cfg(test)]
mod tests;

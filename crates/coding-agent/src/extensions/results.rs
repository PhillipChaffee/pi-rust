//! The per-event handler results, upstream's result interfaces in
//! `src/core/extensions/types.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use serde_json::Value;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{ProviderHeaders, Usage};

use crate::compaction::CompactionResult;
use crate::extensions::reasons::CustomMessagePayload;

/// The generic emit's result for the `session_before_*` channels, upstream's
/// `SessionBeforeEventResult` family: the last defined result wins and a
/// cancelled result short-circuits.
#[derive(Clone, Debug, Default)]
pub enum SessionBeforeResult {
    /// `session_before_switch`.
    Switch {
        /// Whether the switch is cancelled.
        cancel: bool,
    },
    /// `session_before_fork`.
    Fork {
        /// Whether the fork is cancelled.
        cancel: bool,
        /// Skip restoring the conversation into agent state.
        skip_conversation_restore: bool,
    },
    /// `session_before_compact`.
    Compact {
        /// Whether compaction is cancelled.
        cancel: bool,
        /// An extension-supplied compaction result.
        compaction: Option<CompactionResult>,
    },
    /// `session_before_tree`.
    Tree {
        /// Whether navigation is cancelled.
        cancel: bool,
        /// An extension-supplied branch summary.
        summary: Option<ExtensionBranchSummary>,
        /// Overridden custom instructions.
        custom_instructions: Option<String>,
        /// Overridden replace-instructions flag.
        replace_instructions: Option<bool>,
        /// Overridden label.
        label: Option<String>,
    },
    /// No handler returned a result.
    #[default]
    None,
}

/// The extension-supplied branch summary, upstream's
/// `SessionBeforeTreeResult.summary`.
#[derive(Clone, Debug)]
pub struct ExtensionBranchSummary {
    /// The summary text.
    pub summary: String,
    /// Arbitrary details.
    pub details: Option<Value>,
    /// The generation usage.
    pub usage: Option<Usage>,
}

/// The `message_end` replacement result, upstream's `MessageEndEventResult`.
#[derive(Clone, Debug, Default)]
pub struct MessageEndResult {
    /// The replacement message; must keep the original role.
    pub message: Option<AgentMessage>,
}

/// The `tool_call` result, upstream's `ToolCallEventResult`.
#[derive(Clone, Debug, Default)]
pub struct ToolCallResult {
    /// Block tool execution.
    pub block: bool,
    /// The block reason.
    pub reason: Option<String>,
    /// Hint that the agent should stop after the current tool batch.
    pub terminate: bool,
}

/// The `tool_result` result, upstream's `ToolResultEventResult`.
#[derive(Clone, Debug, Default)]
pub struct ToolResultResult {
    /// Replacement content.
    pub content: Option<Vec<pi_ai::types::AssistantBlock>>,
    /// Replacement details.
    pub details: Option<Value>,
    /// Replacement error flag.
    pub is_error: Option<bool>,
    /// Replacement usage.
    pub usage: Option<Usage>,
}

/// The `context` result, upstream's `ContextEventResult`.
#[derive(Clone, Debug, Default)]
pub struct ContextResult {
    /// Replacement context messages.
    pub messages: Option<Vec<AgentMessage>>,
}

/// The `before_agent_start` result, upstream's `BeforeAgentStartEventResult`.
#[derive(Clone, Debug, Default)]
pub struct BeforeAgentStartResult {
    /// A custom message queued alongside the prompt.
    pub message: Option<CustomMessagePayload>,
    /// The replacement system prompt; handlers chain.
    pub system_prompt: Option<String>,
}

/// The `resources_discover` result, upstream's `ResourcesDiscoverResult`.
#[derive(Clone, Debug, Default)]
pub struct ResourcesDiscoverResult {
    /// Skill paths to load.
    pub skill_paths: Vec<String>,
    /// Prompt-template paths to load.
    pub prompt_paths: Vec<String>,
    /// Theme paths to load.
    pub theme_paths: Vec<String>,
}

/// The `input` result, upstream's `InputEventResult`.
#[derive(Clone, Debug)]
pub enum InputResult {
    /// The extension handled the input; the session drops it.
    Handled,
    /// The extension transformed the input.
    Transform {
        /// The replacement text.
        text: String,
        /// The replacement images; absent keeps the input's images.
        images: Option<Vec<pi_ai::types::ImageContent>>,
    },
    /// No handler acted.
    Continue,
}

/// The `user_bash` result, upstream's `UserBashEventResult`. The operations
/// half rides the bash executor's seam (#149).
#[derive(Clone, Debug, Default)]
pub struct UserBashResult {
    /// Whether the extension replaced execution entirely.
    pub handled: bool,
}

/// The `project_trust` decision, upstream's `ProjectTrustEventDecision`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectTrustDecision {
    /// The handler deferred to later handlers.
    Undecided,
    /// The project is trusted.
    Yes,
    /// The project is not trusted.
    No,
}

/// The `project_trust` result, upstream's `ProjectTrustEventResult`.
#[derive(Clone, Debug, Default)]
pub struct ProjectTrustResult {
    /// The decision; `None` reads as undecided.
    pub trusted: Option<ProjectTrustDecision>,
}

/// The per-event handler result, upstream's handler return union. The
/// generic emit reads only the `session_before_*` variants; the dedicated
/// emit methods read their own.
#[allow(
    clippy::large_enum_variant,
    reason = "the result payloads mirror upstream's handler returns"
)]
#[derive(Clone, Debug, Default)]
pub enum HandlerResult {
    /// A `session_before_*` result.
    SessionBefore(SessionBeforeResult),
    /// A `message_end` replacement.
    MessageEnd(MessageEndResult),
    /// A `tool_call` decision.
    ToolCall(ToolCallResult),
    /// A `tool_result` modification.
    ToolResult(ToolResultResult),
    /// A `context` replacement.
    Context(ContextResult),
    /// A `before_provider_request` replacement payload.
    BeforeProviderRequest(Value),
    /// A `before_provider_headers` mutation, restated as the replacement map.
    BeforeProviderHeaders(ProviderHeaders),
    /// A `before_agent_start` contribution.
    BeforeAgentStart(BeforeAgentStartResult),
    /// A `resources_discover` contribution.
    ResourcesDiscover(ResourcesDiscoverResult),
    /// An `input` action.
    Input(InputResult),
    /// A `user_bash` decision.
    UserBash(UserBashResult),
    /// A `project_trust` decision.
    ProjectTrust(ProjectTrustResult),
    /// The handler returned nothing.
    #[default]
    None,
}

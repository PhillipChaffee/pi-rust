//! The session's config and result types.

use std::sync::Arc;
use std::sync::Mutex;

use pi_ai::types::ImageContent;
use pi_ai::types::Model;

use crate::extensions::reasons::ExtensionMode;
use crate::extensions::reasons::InputSource;
use crate::extensions::reasons::SessionStartReason;
use crate::extensions::reasons::StreamingBehavior;
use crate::extensions::runner::CommandContextActions;
use crate::extensions::runner::ExtensionErrorListener;
use crate::model_resolver::ScopedModel;
use crate::session_manager::BranchSummaryEntry;

/// The service bundle an `AgentSession` is constructed from.
///
/// Restatement: upstream carries the settings manager beside the resource
/// loader; the port reads the loader's shared manager so both observe the
/// same writes.
pub struct AgentSessionConfig<S: crate::settings_manager::SettingsStorage + 'static> {
    /// The agent the session drives.
    pub agent: pi_agent_core::Agent,
    /// The session manager.
    pub session_manager: crate::session_manager::SessionManager,
    /// The resource loader, whose settings manager the session shares.
    pub resource_loader: crate::resource_loader::DefaultResourceLoader<S>,
    /// The working directory.
    pub cwd: String,
    /// Models to cycle through with Ctrl+P.
    pub scoped_models: Option<Vec<ScopedModel>>,
    /// The canonical model/auth runtime.
    pub model_runtime: crate::model_runtime::ModelRuntime,
    /// SDK custom tools registered outside extensions.
    pub custom_tools: Option<Vec<crate::extensions::types::ToolDefinition>>,
    /// Initial active built-in tool names. Default: read, bash, edit, write.
    pub initial_active_tool_names: Option<Vec<String>>,
    /// Optional allowlist of exposed tool names.
    pub allowed_tool_names: Option<Vec<String>>,
    /// Optional denylist of exposed tool names.
    pub excluded_tool_names: Option<Vec<String>>,
    /// Override base tools (useful for custom runtimes).
    pub base_tools_override:
        Option<std::collections::BTreeMap<String, pi_agent_core::types::AgentTool>>,
    /// Mutable slot the agent reads for the current extension runner.
    pub extension_runner_ref:
        Option<Arc<Mutex<Option<Arc<crate::extensions::runner::ExtensionRunner>>>>>,
    /// The session-start seed emitted when extensions bind.
    pub session_start_event: Option<SessionStartSeed>,
}

impl<S: crate::settings_manager::SettingsStorage + 'static> std::fmt::Debug
    for AgentSessionConfig<S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentSessionConfig")
            .field("session_manager", &self.session_manager)
            .field("cwd", &self.cwd)
            .field("scoped_models", &self.scoped_models.as_ref().map(Vec::len))
            .finish_non_exhaustive()
    }
}

/// The options one `prompt` call takes, upstream's `PromptOptions`.
#[derive(Clone, Default)]
pub struct PromptOptions {
    /// Whether skill commands and prompt templates expand. Default: true.
    pub expand_prompt_templates: Option<bool>,
    /// Image attachments.
    pub images: Option<Vec<ImageContent>>,
    /// The streaming delivery mode; required while streaming.
    pub streaming_behavior: Option<StreamingBehavior>,
    /// Where the input came from. Default: interactive.
    pub source: Option<InputSource>,
    /// Internal preflight observer, upstream's RPC hook.
    pub preflight_result: Option<Arc<dyn Fn(bool) + Send + Sync>>,
}

impl std::fmt::Debug for PromptOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptOptions")
            .field("expand_prompt_templates", &self.expand_prompt_templates)
            .field("images", &self.images)
            .field("streaming_behavior", &self.streaming_behavior)
            .field("source", &self.source)
            .field("preflight_result", &self.preflight_result.is_some())
            .finish()
    }
}

/// The persistence option shared by model/thinking mutations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelMutationOptions {
    /// Whether the new value persists to the global defaults. Default:
    /// session-only.
    pub persist: bool,
}

/// The model cycle direction, upstream's `"forward" | "backward"`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CycleDirection {
    /// The next model.
    #[default]
    Forward,
    /// The previous model.
    Backward,
}

/// The model cycle result, upstream's `ModelCycleResult`.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelCycleResult {
    /// The selected model.
    pub model: Model,
    /// The applied thinking level.
    pub thinking_level: pi_agent_core::types::ThinkingLevel,
    /// Whether the cycle ran over the scoped models.
    pub is_scoped: bool,
}

/// The session token totals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionTokens {
    /// Prompt tokens.
    pub input: u64,
    /// Completion tokens.
    pub output: u64,
    /// Prompt tokens served from cache.
    pub cache_read: u64,
    /// Prompt tokens written to cache.
    pub cache_write: u64,
    /// The sum of the four.
    pub total: u64,
}

/// Session statistics, upstream's `SessionStats`.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionStats {
    /// The session file, absent for in-memory sessions.
    pub session_file: Option<String>,
    /// The session id.
    pub session_id: String,
    /// User message count.
    pub user_messages: usize,
    /// Assistant message count.
    pub assistant_messages: usize,
    /// Tool call count.
    pub tool_calls: usize,
    /// Tool result count.
    pub tool_results: usize,
    /// All message entries.
    pub total_messages: usize,
    /// Token totals.
    pub tokens: SessionTokens,
    /// Provider-priced cost total.
    pub cost: f64,
    /// The context usage reading, absent without a model.
    pub context_usage: Option<crate::extensions::runtime::ContextUsage>,
}

/// The extension bindings one session accepts, upstream's
/// `ExtensionBindings` minus the UI context (which rides #132).
#[derive(Clone, Default)]
pub struct ExtensionBindings {
    /// The run mode.
    pub mode: Option<ExtensionMode>,
    /// The command-context replace/reload handlers.
    pub command_context_actions: Option<CommandContextActions>,
    /// The abort handler override.
    pub abort_handler: Option<Arc<dyn Fn() + Send + Sync>>,
    /// The shutdown handler.
    pub shutdown_handler: Option<Arc<dyn Fn() + Send + Sync>>,
    /// The extension error listener.
    pub on_error: Option<ExtensionErrorListener>,
}

impl std::fmt::Debug for ExtensionBindings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionBindings")
            .field("mode", &self.mode)
            .field(
                "command_context_actions",
                &self.command_context_actions.is_some(),
            )
            .field("abort_handler", &self.abort_handler.is_some())
            .field("shutdown_handler", &self.shutdown_handler.is_some())
            .field("on_error", &self.on_error.is_some())
            .finish()
    }
}

/// The session-start seed the first extension event carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionStartSeed {
    /// Why the session starts.
    pub reason: SessionStartReason,
    /// The file a replaced session wrote, when replaced.
    pub previous_session_file: Option<String>,
}

impl Default for SessionStartSeed {
    fn default() -> Self {
        Self {
            reason: SessionStartReason::Startup,
            previous_session_file: None,
        }
    }
}

/// The result of a tree navigation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NavigateTreeOutcome {
    /// The user text to restore to the editor, when the target was a user
    /// message.
    pub editor_text: Option<String>,
    /// Whether the navigation was cancelled.
    pub cancelled: bool,
    /// Whether the summarization was aborted.
    pub aborted: bool,
    /// The written branch-summary entry, when one was written.
    pub summary_entry: Option<BranchSummaryEntry>,
}

/// The bash execution options, upstream's `executeBash` options.
#[derive(Clone, Default)]
pub struct BashExecutionOptions {
    /// Whether the output stays out of the LLM context (the `!!` prefix).
    pub exclude_from_context: bool,
    /// The identifier carried by bash update events.
    pub id: Option<String>,
    /// Custom operations for remote execution.
    pub operations: Option<crate::tools::bash::BashOperations>,
}

impl std::fmt::Debug for BashExecutionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashExecutionOptions")
            .field("exclude_from_context", &self.exclude_from_context)
            .field("id", &self.id)
            .field("operations", &self.operations.is_some())
            .finish()
    }
}

/// The options one tree navigation takes, upstream's `navigateTree` options.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NavigateTreeOptions {
    /// Whether to summarize the abandoned branch.
    pub summarize: bool,
    /// Custom instructions for the summarizer.
    pub custom_instructions: Option<String>,
    /// Whether the instructions replace the default prompt.
    pub replace_instructions: bool,
    /// The label to attach.
    pub label: Option<String>,
}

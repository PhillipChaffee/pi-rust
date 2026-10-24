//! The product session, upstream's `src/core/agent-session.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`AgentSession`] owns the lifecycle shared by every run mode: the agent
//! subscription that persists messages and forwards extension events, the
//! model/thinking/queue mutations, compaction and retry, bash recording,
//! tree navigation, statistics/exports, and the tool registry the system
//! prompt is rebuilt from.
//!
//! Restatements: the mutable teammates (the session manager, settings
//! manager, resource loader) sit behind mutexes whose critical sections
//! never await; the extension runner is shared through an `Arc` and rebuilt
//! when bindings change; upstream's `AbortController` is the workspace
//! cancellation token, and extension events that carry an `AbortSignal` get
//! a bridged signal released when the operation settles; tool definitions
//! are adapted onto the plain `Agent`'s erased tool contract here.

mod access;
mod auto_compaction;
mod bash;
mod bindings;
mod compact;
mod compact_auth;
mod construct;
mod context_actions;
mod definition_tool;
mod dispatch;
mod extension_actions;
mod helpers;
mod hooks;
mod messages;
mod model;
mod prompt;
mod queue;
mod registry;
mod retry;
mod state;
mod stats;
mod thinking;
mod tree;
mod types;

pub use types::{
    AgentSessionConfig, AgentSessionEvent, AgentSessionEventListener, BashExecutionOptions,
    CycleDirection, ExtensionBindings, ModelCycleResult, ModelMutationOptions, NavigateTreeOptions,
    NavigateTreeOutcome, ParsedSkillBlock, PromptOptions, SessionStartSeed, SessionStats,
    SessionTokens, SummarizationSource, parse_skill_block,
};

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::sync::Weak;

use pi_agent_core::types::AgentTool;
use tokio_util::sync::CancellationToken;

use crate::extensions::reasons::ExtensionMode;
use crate::extensions::runner::CommandContextActions;
use crate::extensions::runner::ExtensionErrorListener;
use crate::extensions::runner::ExtensionRunner;
use crate::extensions::types::ToolDefinition;
use crate::model_resolver::ScopedModel;
use crate::model_runtime::ModelRuntime;
use crate::resource_loader::DefaultResourceLoader;
use crate::session_manager::SessionManager;
use crate::settings_manager::SettingsManager;
use crate::settings_manager::SettingsStorage;
use crate::system_prompt::BuildSystemPromptOptions;

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One tool-registry entry: the definition plus its provenance.
#[derive(Clone)]
pub(crate) struct ToolDefinitionEntry {
    pub(crate) definition: ToolDefinition,
    pub(crate) source_info: crate::source_info::SourceInfo,
}

impl std::fmt::Debug for ToolDefinitionEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolDefinitionEntry")
            .field("name", &self.definition.name)
            .finish_non_exhaustive()
    }
}

/// The product session, upstream's `AgentSession`.
pub struct AgentSession<S: SettingsStorage + 'static> {
    pub(crate) agent: pi_agent_core::Agent,
    pub(crate) session_manager: Mutex<SessionManager>,
    pub(crate) settings_manager: Arc<Mutex<SettingsManager<S>>>,
    pub(crate) resource_loader: Mutex<DefaultResourceLoader<S>>,
    pub(crate) cwd: String,
    pub(crate) model_runtime: ModelRuntime,
    pub(crate) scoped_models: Mutex<Vec<ScopedModel>>,
    pub(crate) initial_active_tool_names: Option<Vec<String>>,
    pub(crate) allowed_tool_names: Option<BTreeSet<String>>,
    pub(crate) excluded_tool_names: Option<BTreeSet<String>>,
    pub(crate) base_tools_override: Option<BTreeMap<String, AgentTool>>,
    pub(crate) custom_tools: Vec<ToolDefinition>,
    pub(crate) session_start_event: SessionStartSeed,
    pub(crate) extension_runner_ref: Option<Arc<Mutex<Option<Arc<ExtensionRunner>>>>>,
    pub(crate) event_listeners: Mutex<Vec<(Arc<()>, AgentSessionEventListener)>>,
    pub(crate) unsubscribe_agent: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    pub(crate) self_ref: Mutex<Weak<Self>>,
    pub(crate) is_agent_run_active: std::sync::atomic::AtomicBool,
    pub(crate) idle_tx: tokio::sync::watch::Sender<u64>,
    pub(crate) steering_messages: Mutex<Vec<String>>,
    pub(crate) follow_up_messages: Mutex<Vec<String>>,
    pub(crate) pending_next_turn_messages: Mutex<Vec<pi_agent_core::types::AgentMessage>>,
    pub(crate) pending_custom_messages: Mutex<Vec<pi_agent_core::types::AgentMessage>>,
    pub(crate) compaction_abort: Mutex<Option<CancellationToken>>,
    pub(crate) auto_compaction_abort: Mutex<Option<CancellationToken>>,
    pub(crate) branch_summary_abort: Mutex<Option<CancellationToken>>,
    pub(crate) retry_abort: Mutex<Option<CancellationToken>>,
    pub(crate) retry_attempt: std::sync::atomic::AtomicUsize,
    pub(crate) overflow_recovery_attempted: std::sync::atomic::AtomicBool,
    pub(crate) signal_watchers: Mutex<Vec<CancellationToken>>,
    pub(crate) bash_abort_controllers:
        Mutex<Vec<Arc<pi_agent_core::harness::context::AbortController>>>,
    pub(crate) pending_bash_messages: Mutex<Vec<pi_agent_core::types::AgentMessage>>,
    pub(crate) extension_runner: Mutex<Arc<ExtensionRunner>>,
    pub(crate) extension_mode: Mutex<ExtensionMode>,
    pub(crate) extension_command_context_actions: Mutex<Option<CommandContextActions>>,
    pub(crate) extension_abort_handler: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    pub(crate) extension_shutdown_handler: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    pub(crate) extension_error_listener: Mutex<Option<ExtensionErrorListener>>,
    pub(crate) extension_error_unsubscriber: Mutex<Option<Box<dyn Fn() + Send>>>,
    pub(crate) turn_index: std::sync::atomic::AtomicUsize,
    pub(crate) base_tool_definitions: Mutex<BTreeMap<String, ToolDefinition>>,
    pub(crate) tool_registry: Mutex<BTreeMap<String, AgentTool>>,
    pub(crate) tool_definitions: Mutex<BTreeMap<String, ToolDefinitionEntry>>,
    pub(crate) tool_prompt_snippets: Mutex<BTreeMap<String, String>>,
    pub(crate) tool_prompt_guidelines: Mutex<BTreeMap<String, Vec<String>>>,
    pub(crate) base_system_prompt: Mutex<String>,
    pub(crate) base_system_prompt_options: Mutex<BuildSystemPromptOptions>,
    pub(crate) system_prompt_override: Mutex<Option<String>>,
    pub(crate) last_assistant_message: Mutex<Option<pi_ai::types::AssistantMessage>>,
}

impl<S: SettingsStorage + 'static> std::fmt::Debug for AgentSession<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentSession")
            .field("cwd", &self.cwd)
            .field("session_id", &self.session_id())
            .finish_non_exhaustive()
    }
}

impl<S: SettingsStorage + 'static> AgentSession<S> {
    pub(crate) fn self_weak(&self) -> Weak<Self> {
        lock(&self.self_ref).clone()
    }

    /// The agent handle.
    #[must_use]
    pub const fn agent(&self) -> &pi_agent_core::Agent {
        &self.agent
    }

    /// The model runtime.
    #[must_use]
    pub const fn model_runtime(&self) -> &ModelRuntime {
        &self.model_runtime
    }

    /// The settings manager the loader shares.
    #[must_use]
    pub fn settings_manager(&self) -> Arc<Mutex<SettingsManager<S>>> {
        Arc::clone(&self.settings_manager)
    }

    /// Read the resource loader.
    pub fn with_resource_loader<R>(&self, f: impl FnOnce(&DefaultResourceLoader<S>) -> R) -> R {
        f(&lock(&self.resource_loader))
    }

    /// The loaded prompt templates.
    #[must_use]
    pub fn prompt_templates(&self) -> Vec<crate::prompt_templates::PromptTemplate> {
        lock(&self.resource_loader).get_prompts().prompts
    }

    /// The working directory.
    #[must_use]
    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    /// The controller set the compaction/tree slices read.
    pub(crate) fn is_any_compaction(&self) -> bool {
        lock(&self.compaction_abort).is_some()
            || lock(&self.auto_compaction_abort).is_some()
            || lock(&self.branch_summary_abort).is_some()
    }

    /// Whether the session is processing a run.
    #[must_use]
    pub fn is_streaming(&self) -> bool {
        self.is_agent_run_active
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether no run, compaction, or branch summary is active.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        !self.is_streaming() && !self.is_any_compaction()
    }

    /// Whether a compaction or branch summarization is running.
    #[must_use]
    pub fn is_compacting(&self) -> bool {
        self.is_any_compaction()
    }

    pub(crate) fn bump_idle(&self) {
        let generation = *self.idle_tx.borrow() + 1;
        let _ = self.idle_tx.send(generation);
    }

    pub(crate) fn resolve_idle_wait_if_idle(&self) {
        if self.is_idle() {
            self.bump_idle();
        }
    }

    /// Abort the current operation and wait for idle.
    pub async fn abort(&self) {
        self.abort_retry();
        self.abort_compaction();
        self.abort_branch_summary();
        self.agent.abort();
        self.wait_for_idle().await;
    }

    /// Wait until the session is idle.
    pub async fn wait_for_idle(&self) {
        if self.is_idle() {
            return;
        }
        let mut rx = self.idle_tx.subscribe();
        while !self.is_idle() {
            if rx.changed().await.is_err() {
                return;
            }
        }
    }
}

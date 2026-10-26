//! Construction: the service bundle, initial runner, and agent subscription.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;

use pi_agent_core::types::AgentEvent;
use tokio_util::sync::CancellationToken;

use super::lock;
use super::types::AgentSessionConfig;
use crate::extensions::api::ExtensionRuntime;
use crate::extensions::reasons::ExtensionMode;
use crate::extensions::runner::ExtensionRunner;
use crate::extensions::runtime::ContextActions;
use crate::model_registry::ModelRegistry;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Build the session over `config`.
    ///
    /// # Panics
    /// When the shared loader slot is empty, which only a concurrent reload
    /// can cause.
    #[must_use]
    #[expect(
        clippy::expect_used,
        clippy::too_many_lines,
        reason = "the loader is present outside the reload window; the 1:1 constructor runs just past the line budget"
    )]
    pub fn new(config: AgentSessionConfig<S>) -> Arc<Self> {
        let AgentSessionConfig {
            agent,
            session_manager,
            resource_loader,
            cwd,
            scoped_models,
            model_runtime,
            custom_tools,
            initial_active_tool_names,
            allowed_tool_names,
            excluded_tool_names,
            base_tools_override,
            extension_runner_ref,
            session_start_event,
        } = config;
        let settings_manager = lock(&resource_loader)
            .as_ref()
            .map(crate::resource_loader::DefaultResourceLoader::settings_manager)
            .expect("the loader is present at construction");
        let (idle_tx, _) = tokio::sync::watch::channel(0);
        let placeholder = ExtensionRunner::new(
            Vec::new(),
            Arc::new(ContextActions::default()),
            Arc::new(|| Ok(())),
            cwd.clone(),
            ExtensionMode::Print,
            None,
            session_manager.clone(),
            ModelRegistry::new(model_runtime.clone()),
            ExtensionRuntime::new(),
        );
        let session = Arc::new(Self {
            agent,
            session_manager: Mutex::new(session_manager),
            settings_manager,
            resource_loader,
            cwd,
            model_runtime,
            scoped_models: Mutex::new(scoped_models.unwrap_or_default()),
            initial_active_tool_names,
            allowed_tool_names: allowed_tool_names.map(|names| names.into_iter().collect()),
            excluded_tool_names: excluded_tool_names.map(|names| names.into_iter().collect()),
            base_tools_override,
            custom_tools: custom_tools.unwrap_or_default(),
            session_start_event: session_start_event.unwrap_or_default(),
            extension_runner_ref,
            event_listeners: Mutex::new(Vec::new()),
            unsubscribe_agent: Mutex::new(None),
            self_ref: Mutex::new(Weak::new()),
            is_agent_run_active: std::sync::atomic::AtomicBool::new(false),
            idle_tx,
            steering_messages: Mutex::new(Vec::new()),
            follow_up_messages: Mutex::new(Vec::new()),
            pending_next_turn_messages: Mutex::new(Vec::new()),
            pending_custom_messages: Mutex::new(Vec::new()),
            compaction_abort: Mutex::new(None),
            auto_compaction_abort: Mutex::new(None),
            branch_summary_abort: Mutex::new(None),
            retry_abort: Mutex::new(None),
            retry_attempt: std::sync::atomic::AtomicUsize::new(0),
            overflow_recovery_attempted: std::sync::atomic::AtomicBool::new(false),
            signal_watchers: Mutex::new(Vec::new()),
            bash_abort_controllers: Mutex::new(Vec::new()),
            pending_bash_messages: Mutex::new(Vec::new()),
            extension_runner: Mutex::new(Arc::new(placeholder)),
            extension_mode: Mutex::new(ExtensionMode::Print),
            extension_ui: Mutex::new(None),
            extension_command_context_actions: Mutex::new(None),
            extension_abort_handler: Mutex::new(None),
            extension_shutdown_handler: Mutex::new(None),
            extension_error_listener: Mutex::new(None),
            extension_error_unsubscriber: Mutex::new(None),
            turn_index: std::sync::atomic::AtomicUsize::new(0),
            base_tool_definitions: Mutex::new(BTreeMap::new()),
            tool_registry: Mutex::new(BTreeMap::new()),
            tool_definitions: Mutex::new(BTreeMap::new()),
            tool_prompt_snippets: Mutex::new(BTreeMap::new()),
            tool_prompt_guidelines: Mutex::new(BTreeMap::new()),
            base_system_prompt: Mutex::new(String::new()),
            base_system_prompt_options: Mutex::new(
                crate::system_prompt::BuildSystemPromptOptions::default(),
            ),
            system_prompt_override: Mutex::new(None),
            last_assistant_message: Mutex::new(None),
        });
        *lock(&session.self_ref) = Arc::downgrade(&session);

        let weak = Arc::downgrade(&session);
        let unsubscribe = session.agent.subscribe(Arc::new(
            move |event: AgentEvent, _token: CancellationToken| {
                let weak = weak.clone();
                Box::pin(async move {
                    if let Some(session) = weak.upgrade() {
                        session.handle_agent_event(event).await;
                    }
                })
            },
        ));
        *lock(&session.unsubscribe_agent) = Some(Box::new(unsubscribe));
        session.install_agent_tool_hooks();
        session.install_agent_next_turn_refresh();
        let initial_active_tool_names =
            session
                .initial_active_tool_names
                .clone()
                .unwrap_or_else(|| {
                    if session.base_tools_override.is_some() {
                        session
                            .base_tools_override
                            .as_ref()
                            .map_or_else(Vec::new, |overrides| overrides.keys().cloned().collect())
                    } else {
                        ["read", "bash", "edit", "write"]
                            .iter()
                            .map(|name| (*name).to_owned())
                            .collect()
                    }
                });
        session.build_runtime(Some(initial_active_tool_names), true);
        session
    }
}

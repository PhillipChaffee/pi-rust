//! Read-only state access and the session getters.

use std::sync::Arc;

use pi_ai::types::Model;

use super::lock;
use super::types::AgentSessionEvent;
use crate::extensions::runner::ExtensionRunner;
use crate::model_resolver::ScopedModel;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// The session id.
    #[must_use]
    pub fn session_id(&self) -> String {
        lock(&self.session_manager).session_id().to_owned()
    }

    /// The session file path, absent for in-memory sessions.
    #[must_use]
    pub fn session_file(&self) -> Option<String> {
        lock(&self.session_manager)
            .session_file()
            .map(str::to_owned)
    }

    /// The session display name.
    #[must_use]
    pub fn session_name(&self) -> Option<String> {
        lock(&self.session_manager).get_session_name()
    }

    /// The session header, upstream's `sessionManager.getHeader()`.
    #[must_use]
    pub fn session_header(&self) -> Option<crate::session_manager::SessionHeader> {
        lock(&self.session_manager).get_header().cloned()
    }

    /// The current leaf id, absent before any entry.
    #[must_use]
    pub fn leaf_id(&self) -> Option<String> {
        lock(&self.session_manager).get_leaf_id().map(str::to_owned)
    }

    /// The current model.
    #[must_use]
    pub fn model(&self) -> Option<Model> {
        Some(self.agent.state().model.clone())
    }

    /// The current thinking level.
    #[must_use]
    pub fn thinking_level(&self) -> pi_agent_core::types::ThinkingLevel {
        self.agent.state().thinking_level
    }

    /// The current effective system prompt.
    #[must_use]
    pub fn system_prompt(&self) -> String {
        self.agent.state().system_prompt.clone()
    }

    /// The scoped models.
    #[must_use]
    pub fn scoped_models(&self) -> Vec<ScopedModel> {
        lock(&self.scoped_models).clone()
    }

    /// Replace the scoped models.
    pub fn set_scoped_models(&self, models: Vec<ScopedModel>) {
        *lock(&self.scoped_models) = models;
    }

    /// Set the session display name.
    pub fn set_session_name(&self, name: &str) {
        let _ = lock(&self.session_manager).append_session_info(name);
        self.emit(AgentSessionEvent::SessionInfoChanged {
            name: self.session_name(),
        });
    }

    /// The extension runner.
    #[must_use]
    pub fn extension_runner(&self) -> Arc<ExtensionRunner> {
        Arc::clone(&lock(&self.extension_runner))
    }

    /// Whether extensions handle an event type.
    #[must_use]
    pub fn has_extension_handlers(&self, event_type: &str) -> bool {
        self.extension_runner().has_handlers(event_type)
    }

    /// The active tool names.
    #[must_use]
    pub fn active_tool_names(&self) -> Vec<String> {
        self.agent
            .state()
            .tools
            .iter()
            .map(|tool| tool.name().to_owned())
            .collect()
    }
}

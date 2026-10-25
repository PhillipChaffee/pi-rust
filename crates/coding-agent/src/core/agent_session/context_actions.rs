//! The context action bundle the extension runner binds.

use std::sync::Arc;

use super::lock;
use crate::extensions::runtime::ContextActions;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    pub(crate) fn context_actions(&self) -> ContextActions {
        let weak = self.self_weak();
        let weak_model = weak.clone();
        let weak_scoped = weak.clone();
        let weak_thinking = weak.clone();
        let weak_idle = weak.clone();
        let weak_trusted = weak.clone();
        let weak_signal = weak.clone();
        let weak_abort = weak.clone();
        let weak_pending = weak.clone();
        let weak_shutdown = weak.clone();
        let weak_usage = weak.clone();
        let weak_compact = weak.clone();
        let weak_prompt = weak.clone();
        let weak_options = weak;
        ContextActions {
            get_model: Some(Arc::new(move || {
                weak_model.upgrade().and_then(|session| session.model())
            })),
            get_scoped_models: Some(Arc::new(move || {
                weak_scoped
                    .upgrade()
                    .map_or_else(Vec::new, |session| session.scoped_models())
            })),
            get_thinking_level: Some(Arc::new(move || {
                weak_thinking
                    .upgrade()
                    .map_or(pi_agent_core::types::ThinkingLevel::Off, |session| {
                        session.thinking_level()
                    })
            })),
            is_idle: Some(Arc::new(move || {
                weak_idle.upgrade().is_none_or(|session| session.is_idle())
            })),
            is_project_trusted: Some(Arc::new(move || {
                weak_trusted
                    .upgrade()
                    .is_some_and(|session| lock(&session.settings_manager).is_project_trusted())
            })),
            get_signal: Some(Arc::new(move || {
                weak_signal
                    .upgrade()
                    .and_then(|session| session.run_signal())
            })),
            abort: Some(Arc::new(move || {
                if let Some(session) = weak_abort.upgrade() {
                    session.abort_now();
                    tokio::spawn(async move { session.wait_for_idle().await });
                }
            })),
            has_pending_messages: Some(Arc::new(move || {
                weak_pending
                    .upgrade()
                    .is_some_and(|session| session.pending_message_count() > 0)
            })),
            shutdown: Some(Arc::new(move || {
                if let Some(session) = weak_shutdown.upgrade()
                    && let Some(handler) = lock(&session.extension_shutdown_handler).clone()
                {
                    handler();
                }
            })),
            get_context_usage: Some(Arc::new(move || {
                weak_usage
                    .upgrade()
                    .and_then(|session| session.context_usage())
            })),
            compact: Some(Arc::new(move |instructions| {
                if let Some(session) = weak_compact.upgrade() {
                    tokio::spawn(async move {
                        let _ = session.compact(instructions).await;
                    });
                }
            })),
            get_system_prompt: Some(Arc::new(move || {
                weak_prompt
                    .upgrade()
                    .map_or_else(String::new, |session| session.system_prompt())
            })),
            get_system_prompt_options: Some(Arc::new(move || {
                weak_options.upgrade().map_or_else(
                    crate::system_prompt::BuildSystemPromptOptions::default,
                    |session| lock(&session.base_system_prompt_options).clone(),
                )
            })),
        }
    }
}

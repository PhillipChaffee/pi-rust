//! Reload, dispose invalidation, fork candidates, and the replaced context.

use super::lock;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::reasons::SessionShutdownReason;
use crate::extensions::reasons::SessionStartReason;
use crate::session_manager::FileEntry;
use crate::session_manager::SessionEntry;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Reload extensions, settings, and resources in place.
    ///
    /// # Panics
    /// When the resource loader is absent, which only a concurrent reload
    /// can cause.
    #[expect(
        clippy::expect_used,
        reason = "the loader is present outside the reload window; a concurrent reload is a caller bug"
    )]
    pub async fn reload(&self, before_session_start: Option<&(dyn Fn() + Send + Sync)>) {
        let old_runner = self.extension_runner();
        let flag_values = old_runner.get_flag_values();
        crate::extensions::runner::emit_session_shutdown_event(
            &old_runner,
            ExtensionEvent::SessionShutdown {
                reason: SessionShutdownReason::Reload,
                target_session_file: None,
            },
        )
        .await;
        old_runner.invalidate(None);
        lock(&self.settings_manager).reload();
        self.sync_queue_modes_from_settings();
        pi_ai::compat::reset_api_providers();
        let mut loader = lock(&self.resource_loader)
            .take()
            .expect("the resource loader is present before reload");
        loader.reload(None).await;
        *lock(&self.resource_loader) = Some(loader);
        self.build_runtime(Some(self.active_tool_names()), true);
        let runner = self.extension_runner();
        for (name, value) in flag_values {
            runner.set_flag_value(&name, value);
        }
        let has_bindings = lock(&self.extension_command_context_actions).is_some()
            || lock(&self.extension_shutdown_handler).is_some()
            || lock(&self.extension_error_listener).is_some();
        if has_bindings {
            if let Some(hook) = before_session_start {
                hook();
            }
            runner
                .emit(ExtensionEvent::SessionStart {
                    reason: SessionStartReason::Reload,
                    previous_session_file: None,
                })
                .await;
            self.extend_resources_from_extensions(SessionStartReason::Reload)
                .await;
        }
    }

    /// The command context plus the session send methods, upstream's
    /// `createReplacedSessionContext`.
    #[must_use]
    pub fn create_replaced_session_context(
        &self,
    ) -> crate::extensions::runner::ExtensionCommandContext {
        self.extension_runner().command_context()
    }

    /// Every user message with its entry id, for the fork selector.
    #[must_use]
    pub fn user_messages_for_forking(&self) -> Vec<(String, String)> {
        lock(&self.session_manager)
            .entries()
            .into_iter()
            .filter_map(|entry| {
                let FileEntry::Entry(SessionEntry::Message(message)) = entry else {
                    return None;
                };
                let Some(pi_agent_core::types::AgentMessage::Standard(
                    pi_ai::types::Message::User(user),
                )) = &message.message
                else {
                    return None;
                };
                let text = pi_ai::utils::text::content_text_with(&user.content, "");
                if text.is_empty() {
                    return None;
                }
                Some((message.base.id.clone().unwrap_or_default(), text))
            })
            .collect()
    }
}

//! Automatic agent-turn retry.

use pi_ai::types::AssistantMessage;
use std::sync::atomic::Ordering;
use tokio_util::sync::CancellationToken;

use super::lock;
use super::types::AgentSessionEvent;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Whether an error is retryable. Context overflow belongs to
    /// compaction, not retry.
    #[must_use]
    pub fn is_retryable_error(&self, message: &AssistantMessage) -> bool {
        if pi_ai::utils::overflow::is_context_overflow(
            message,
            self.model().map(|model| model.context_window),
        ) {
            return false;
        }
        pi_ai::utils::retry::is_retryable_assistant_error(message)
    }

    /// The current retry attempt, zero when not retrying.
    #[must_use]
    pub fn retry_attempt(&self) -> usize {
        self.retry_attempt.load(Ordering::Relaxed)
    }

    /// Whether auto-retry is in progress.
    #[must_use]
    pub fn is_retrying(&self) -> bool {
        lock(&self.retry_abort).is_some()
    }

    /// Whether auto-retry is enabled.
    #[must_use]
    pub fn auto_retry_enabled(&self) -> bool {
        lock(&self.settings_manager).get_retry_enabled()
    }

    /// Toggle auto-retry.
    pub fn set_auto_retry_enabled(&self, enabled: bool) {
        lock(&self.settings_manager).set_retry_enabled(enabled);
    }

    /// Cancel an in-progress retry.
    pub fn abort_retry(&self) {
        if let Some(token) = lock(&self.retry_abort).as_ref() {
            token.cancel();
        }
    }

    pub(crate) async fn prepare_retry(&self, message: &AssistantMessage) -> bool {
        let settings = lock(&self.settings_manager).get_retry_settings();
        if !settings.enabled {
            return false;
        }
        let attempt = self.retry_attempt.fetch_add(1, Ordering::Relaxed) + 1;
        let max_retries = usize::try_from(settings.max_retries.max(0)).unwrap_or(0);
        if attempt > max_retries {
            self.retry_attempt.store(attempt - 1, Ordering::Relaxed);
            return false;
        }
        let policy = pi_ai::utils::retry::RetryPolicy {
            enabled: settings.enabled,
            max_retries: u32::try_from(settings.max_retries.max(0)).unwrap_or(0),
            base_delay_ms: u64::try_from(settings.base_delay_ms.max(0)).unwrap_or(0),
            max_agent_delay_ms: Some(
                u64::try_from(settings.max_agent_delay_ms.max(0)).unwrap_or(0),
            ),
        };
        let delay_ms = pi_ai::utils::retry::retry_delay_ms(
            &policy,
            u32::try_from(attempt).unwrap_or(u32::MAX),
        );
        self.emit(AgentSessionEvent::AutoRetryStart {
            attempt,
            max_attempts: max_retries,
            delay_ms,
            error_message: message
                .error_message
                .clone()
                .unwrap_or_else(|| "Unknown error".to_owned()),
        });

        {
            let mut state = self.agent.state();
            if matches!(
                state.messages.last(),
                Some(pi_agent_core::types::AgentMessage::Standard(
                    pi_ai::types::Message::Assistant(_)
                ))
            ) {
                state.messages.pop();
            }
        }

        let token = CancellationToken::new();
        *lock(&self.retry_abort) = Some(token.clone());
        let slept = crate::utils::sleep::sleep(delay_ms, Some(&token)).await;
        *lock(&self.retry_abort) = None;
        if slept.is_err() {
            let attempt = self.retry_attempt.swap(0, Ordering::Relaxed);
            self.emit(AgentSessionEvent::AutoRetryEnd {
                success: false,
                attempt,
                final_error: Some("Retry cancelled".to_owned()),
            });
            return false;
        }
        true
    }
}

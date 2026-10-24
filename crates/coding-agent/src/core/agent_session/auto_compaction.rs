//! Threshold and overflow compaction, plus the abort handles.

use std::sync::atomic::Ordering;

use super::helpers::model_key;
use super::lock;
use super::types::AgentSessionEvent;
use crate::extensions::reasons::CompactionReason;
use crate::session_manager::FileEntry;
use crate::session_manager::SessionEntry;
use crate::settings_manager::SettingsStorage;
use tokio_util::sync::CancellationToken;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// The post-turn compaction ladder, upstream's `_checkCompaction`.
    #[expect(
        clippy::suspicious_operation_groupings,
        reason = "the assistant carries `model`, not `id`; the comparison is upstream's"
    )]
    pub(crate) async fn check_compaction(
        &self,
        message: &pi_ai::types::AssistantMessage,
        skip_aborted_check: bool,
    ) -> bool {
        let Some(model) = self.model() else {
            return false;
        };
        let settings =
            { lock(&self.settings_manager).get_compaction_settings(Some(&model_key(&model))) };
        let Ok(settings) = settings else {
            return false;
        };
        if !settings.enabled {
            return false;
        }
        if skip_aborted_check && message.stop_reason == pi_ai::types::StopReason::Aborted {
            return false;
        }
        let same_model = model.provider.0 == message.provider.0 && model.id == message.model;
        let overflow = same_model
            && pi_ai::utils::overflow::is_context_overflow(message, Some(model.context_window));
        let recoverable =
            same_model && pi_ai::utils::overflow::is_recoverable_length(message, model.max_tokens);
        if overflow || recoverable {
            if message.stop_reason == pi_ai::types::StopReason::Stop {
                return self
                    .run_auto_compaction(CompactionReason::Overflow, false)
                    .await;
            }
            if self.overflow_recovery_attempted.load(Ordering::Relaxed) {
                self.emit(AgentSessionEvent::CompactionEnd {
                    reason: CompactionReason::Overflow,
                    result: None,
                    aborted: false,
                    will_retry: false,
                    error_message: Some(
                        "Context overflow recovery failed after one compact-and-retry attempt. Try reducing context or switching to a larger-context model.".to_owned(),
                    ),
                });
                return false;
            }
            self.overflow_recovery_attempted
                .store(true, Ordering::Relaxed);
            self.drop_trailing_assistant();
            return self
                .run_auto_compaction(CompactionReason::Overflow, true)
                .await;
        }
        let tokens = if message.stop_reason == pi_ai::types::StopReason::Error {
            crate::compaction::estimate_context_tokens(&self.agent.state().messages).tokens
        } else {
            crate::compaction::calculate_context_tokens(&message.usage)
        };
        if crate::compaction::should_compact(tokens, model.context_window, &settings) {
            return self
                .run_auto_compaction(CompactionReason::Threshold, false)
                .await;
        }
        false
    }

    pub(crate) async fn run_auto_compaction(
        &self,
        reason: CompactionReason,
        will_retry: bool,
    ) -> bool {
        let token = CancellationToken::new();
        *lock(&self.auto_compaction_abort) = Some(token.clone());
        self.emit(AgentSessionEvent::CompactionStart { reason });
        let outcome = self.compact_once(reason, None, &token, will_retry).await;
        *lock(&self.auto_compaction_abort) = None;
        self.resolve_idle_wait_if_idle();
        match outcome {
            Ok(Some(result)) => {
                self.emit(AgentSessionEvent::CompactionEnd {
                    reason,
                    result: Some(result),
                    aborted: false,
                    will_retry,
                    error_message: None,
                });
                if will_retry {
                    self.drop_trailing_assistant();
                    return true;
                }
                self.agent.has_queued_messages()
            }
            Ok(None) => false,
            Err(error) => {
                let aborted = error == "Compaction cancelled";
                self.emit(AgentSessionEvent::CompactionEnd {
                    reason,
                    result: None,
                    aborted,
                    will_retry: false,
                    error_message: (!aborted).then(|| format!("Auto-compaction failed: {error}")),
                });
                false
            }
        }
    }

    pub(crate) fn drop_trailing_assistant(&self) {
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

    pub(crate) fn branch_entries(&self) -> Vec<SessionEntry> {
        lock(&self.session_manager)
            .get_branch(None)
            .into_iter()
            .filter_map(|entry| match entry {
                FileEntry::Entry(entry) => Some(entry.clone()),
                _ => None,
            })
            .collect()
    }

    /// Cancel an in-progress compaction.
    pub fn abort_compaction(&self) {
        if let Some(token) = lock(&self.compaction_abort).as_ref() {
            token.cancel();
        }
        if let Some(token) = lock(&self.auto_compaction_abort).as_ref() {
            token.cancel();
        }
    }

    /// Cancel an in-progress branch summarization.
    pub fn abort_branch_summary(&self) {
        if let Some(token) = lock(&self.branch_summary_abort).as_ref() {
            token.cancel();
        }
    }

    pub(crate) fn summarization_callbacks(
        &self,
        reason: CompactionReason,
    ) -> pi_ai::utils::retry::RetryCallbacks {
        let weak = self.self_weak();
        pi_ai::utils::retry::RetryCallbacks {
            on_retry_scheduled: Some(std::sync::Arc::new({
                let weak = weak.clone();
                move |attempt, max_attempts, delay_ms, error_message| {
                    if let Some(session) = weak.upgrade() {
                        session.emit(AgentSessionEvent::SummarizationRetryScheduled {
                            attempt,
                            max_attempts,
                            delay_ms,
                            error_message: error_message.to_owned(),
                        });
                    }
                }
            })),
            on_retry_attempt_start: Some(std::sync::Arc::new({
                let weak = weak.clone();
                move || {
                    if let Some(session) = weak.upgrade() {
                        session.emit(AgentSessionEvent::SummarizationRetryAttemptStart {
                            source: super::types::SummarizationSource::Compaction(reason),
                        });
                    }
                }
            })),
            on_retry_finished: Some(std::sync::Arc::new({
                move |_success, _attempt, _error| {
                    if let Some(session) = weak.upgrade() {
                        session.emit(AgentSessionEvent::SummarizationRetryFinished);
                    }
                }
            })),
        }
    }
}

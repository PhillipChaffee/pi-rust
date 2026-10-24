//! Prompting and the post-run loop.

use pi_agent_core::types::AgentMessage;
use pi_ai::types::Message;
use pi_ai::types::TextContent;
use pi_ai::types::UserBlock;
use pi_ai::types::UserContent;
use pi_ai::types::UserMessage;

use super::helpers::notify_preflight;
use super::helpers::now_ms;
use super::lock;
use super::types::AgentSessionEvent;
use super::types::PromptOptions;
use crate::extensions::reasons::StreamingBehavior;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Clear the queues and return their texts.
    #[must_use]
    pub fn clear_queue(&self) -> (Vec<String>, Vec<String>) {
        let steering = std::mem::take(&mut *lock(&self.steering_messages));
        let follow_up = std::mem::take(&mut *lock(&self.follow_up_messages));
        self.agent.clear_all_queues();
        self.emit_queue_update();
        (steering, follow_up)
    }

    /// Send a prompt to the agent.
    ///
    /// # Errors
    /// When the prompt is queued while streaming without a streaming
    /// behavior, or when no model is selected.
    pub async fn prompt(&self, text: String, options: Option<PromptOptions>) -> Result<(), String> {
        let options = options.unwrap_or_default();
        let expand = options.expand_prompt_templates.unwrap_or(true);
        let preflight = options.preflight_result.clone();
        let result = self
            .prompt_inner(text, options, expand, preflight.as_deref())
            .await;
        if result.is_err() {
            notify_preflight(preflight.as_deref(), false);
        }
        result
    }

    async fn prompt_inner(
        &self,
        text: String,
        options: PromptOptions,
        expand: bool,
        preflight: Option<&(dyn Fn(bool) + Send + Sync)>,
    ) -> Result<(), String> {
        let mut expanded = text;
        if expand {
            expanded = self.expand_skill_command(&expanded);
            expanded = crate::prompt_templates::expand_prompt_template(
                &expanded,
                &self.prompt_templates(),
            );
        }

        if self.is_streaming() {
            let Some(behavior) = options.streaming_behavior else {
                return Err("Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') to queue the message.".to_owned());
            };
            if behavior == StreamingBehavior::FollowUp {
                self.queue_follow_up(expanded, options.images);
            } else {
                self.queue_steer(expanded, options.images);
            }
            notify_preflight(preflight, true);
            return Ok(());
        }

        if self.model().is_none() {
            return Err(crate::auth_guidance::format_no_model_selected_message());
        }
        self.flush_pending_bash_messages();
        self.flush_pending_custom_messages();

        let mut content = vec![UserBlock::Text(TextContent {
            text: expanded,
            text_signature: None,
        })];
        if let Some(images) = options.images {
            content.extend(images.into_iter().map(UserBlock::Image));
        }
        let mut messages = vec![AgentMessage::Standard(Message::User(UserMessage {
            content: UserContent::Blocks(content),
            timestamp: now_ms(),
        }))];
        messages.extend(lock(&self.pending_next_turn_messages).drain(..));

        notify_preflight(preflight, true);
        self.run_agent_prompt(messages).await;
        Ok(())
    }

    pub(crate) async fn run_agent_prompt(&self, messages: Vec<AgentMessage>) {
        self.is_agent_run_active
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = self.agent.prompt(messages).await;
        while self.handle_post_agent_run().await {
            let _ = self.agent.continue_run().await;
        }
        self.is_agent_run_active
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.flush_pending_bash_messages();
        self.flush_pending_custom_messages();
        self.emit(AgentSessionEvent::AgentSettled);
    }

    async fn handle_post_agent_run(&self) -> bool {
        let message = lock(&self.last_assistant_message).take();
        let Some(message) = message else {
            return false;
        };
        if self.is_retryable_error(&message) && self.prepare_retry(&message).await {
            return true;
        }
        if self.check_compaction(&message, true).await {
            return true;
        }
        let attempt = self
            .retry_attempt
            .load(std::sync::atomic::Ordering::Relaxed);
        if message.stop_reason == pi_ai::types::StopReason::Error && attempt > 0 {
            self.emit(AgentSessionEvent::AutoRetryEnd {
                success: false,
                attempt,
                final_error: message.error_message.clone(),
            });
            self.retry_attempt
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
        self.agent.has_queued_messages()
    }
}

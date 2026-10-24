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
        if expand && text.starts_with('/') && self.try_execute_extension_command(&text).await {
            notify_preflight(preflight, true);
            return Ok(());
        }
        let source = options.source.unwrap_or_default();
        let Some((text, images)) = self
            .run_input_handlers(
                text,
                options.images,
                source,
                if self.is_streaming() {
                    options.streaming_behavior
                } else {
                    None
                },
            )
            .await
        else {
            notify_preflight(preflight, true);
            return Ok(());
        };
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
                self.queue_follow_up(expanded, images);
            } else {
                self.queue_steer(expanded, images);
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
        if let Some(images) = images {
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

    pub(crate) async fn run_input_handlers(
        &self,
        text: String,
        images: Option<Vec<pi_ai::types::ImageContent>>,
        source: crate::extensions::reasons::InputSource,
        streaming_behavior: Option<StreamingBehavior>,
    ) -> Option<(String, Option<Vec<pi_ai::types::ImageContent>>)> {
        let runner = self.extension_runner();
        if !runner.has_handlers("input") {
            return Some((text, images));
        }
        match runner
            .emit_input(text.clone(), images.clone(), source, streaming_behavior)
            .await
        {
            crate::extensions::results::InputResult::Handled => None,
            crate::extensions::results::InputResult::Transform {
                text: transformed,
                images: replaced,
            } => Some((transformed, replaced.or(images))),
            crate::extensions::results::InputResult::Continue => Some((text, images)),
        }
    }

    async fn try_execute_extension_command(&self, text: &str) -> bool {
        let (command_name, args) = text
            .split_once(' ')
            .map_or((&text[1..], ""), |(name, args)| (&name[1..], args));
        let runner = self.extension_runner();
        let Some(command) = runner.get_command(command_name) else {
            return false;
        };
        (command.handler)(args.to_owned(), runner.context()).await;
        true
    }

    pub(crate) fn throw_if_extension_command(&self, text: &str) -> Result<(), String> {
        let (command_name, _) = text
            .split_once(' ')
            .map_or((&text[1..], ""), |(name, args)| (&name[1..], args));
        if self.extension_runner().get_command(command_name).is_some() {
            return Err(format!(
                "Extension command \"/{command_name}\" cannot be queued. Use prompt() or execute the command when not streaming."
            ));
        }
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
        self.bump_idle();
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

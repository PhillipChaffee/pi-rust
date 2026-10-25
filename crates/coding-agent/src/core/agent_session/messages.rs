//! Custom/user message delivery and helper projections.

use pi_agent_core::types::AgentMessage;
use pi_ai::types::ImageContent;
use pi_ai::types::UserBlock;
use pi_ai::types::UserContent;

use super::lock;
use super::types::AgentSessionEvent;
use super::types::PromptOptions;
use crate::extensions::reasons::StreamingBehavior;
use crate::extensions::runner::DeliverAs;
use crate::extensions::runner::SendMessageOptions;
use crate::extensions::runner::SendUserMessageOptions;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Send a custom message to the session.
    ///
    /// # Errors
    /// When a triggered turn fails.
    pub async fn send_custom_message(
        &self,
        message: crate::extensions::reasons::CustomMessagePayload,
        options: Option<SendMessageOptions>,
    ) -> Result<(), String> {
        let custom = super::helpers::custom_message(
            message.custom_type,
            UserContent::Blocks(message.content),
            message.display,
            message.details,
        );
        let options = options.unwrap_or_default();
        if options.deliver_as == Some(DeliverAs::NextTurn) {
            lock(&self.pending_next_turn_messages).push(custom);
        } else if self.is_streaming() && options.trigger_turn != Some(false) {
            if options.deliver_as == Some(DeliverAs::FollowUp) {
                self.agent.follow_up(custom);
            } else {
                self.agent.steer(custom);
            }
        } else if options.trigger_turn == Some(true) {
            self.run_agent_prompt(vec![custom]).await;
        } else if self.is_streaming() {
            lock(&self.pending_custom_messages).push(custom);
        } else {
            self.append_custom_message(custom);
        }
        Ok(())
    }

    /// Send a user message, always triggering a turn.
    ///
    /// # Errors
    /// When the triggered turn fails.
    pub async fn send_user_message(
        &self,
        content: UserContent,
        options: Option<SendUserMessageOptions>,
    ) -> Result<(), String> {
        let (text, images) = split_user_content(content);
        let options = options.unwrap_or_default();
        self.prompt(
            text,
            Some(PromptOptions {
                expand_prompt_templates: Some(options.expand_prompt_templates.unwrap_or(false)),
                images,
                streaming_behavior: match options.deliver_as {
                    Some(DeliverAs::Steer) => Some(StreamingBehavior::Steer),
                    Some(DeliverAs::FollowUp) => Some(StreamingBehavior::FollowUp),
                    _ => None,
                },
                source: Some(crate::extensions::reasons::InputSource::Extension),
                preflight_result: None,
            }),
        )
        .await
    }

    pub(crate) fn append_custom_message(&self, message: AgentMessage) {
        self.agent.state().messages.push(message.clone());
        if let AgentMessage::Custom(custom) = &message {
            let custom_type = custom
                .field("customType")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let display = custom
                .field("display")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let _ = lock(&self.session_manager).append_custom_message_entry(
                &custom_type,
                custom_content(custom),
                display,
                custom.field("details").cloned(),
            );
        }
        self.emit(AgentSessionEvent::MessageStart {
            message: message.clone(),
        });
        self.emit(AgentSessionEvent::MessageEnd { message });
    }

    pub(crate) fn flush_pending_custom_messages(&self) {
        let pending: Vec<_> = lock(&self.pending_custom_messages).drain(..).collect();
        for message in pending {
            self.append_custom_message(message);
        }
    }
}

pub(super) fn custom_content(custom: &pi_agent_core::types::CustomAgentMessage) -> UserContent {
    custom
        .field("content")
        .and_then(|value| serde_json::from_value::<UserContent>(value.clone()).ok())
        .unwrap_or_else(|| UserContent::Blocks(Vec::new()))
}

pub(super) fn split_user_content(content: UserContent) -> (String, Option<Vec<ImageContent>>) {
    match content {
        UserContent::Text(text) => (text, None),
        UserContent::Blocks(blocks) => {
            let mut text = Vec::new();
            let mut images = Vec::new();
            for block in blocks {
                match block {
                    UserBlock::Text(block) => text.push(block.text),
                    UserBlock::Image(image) => images.push(image),
                }
            }
            (text.join("\n"), (!images.is_empty()).then_some(images))
        }
    }
}

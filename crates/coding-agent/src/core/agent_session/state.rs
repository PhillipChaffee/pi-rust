//! Subscription, event forwarding, persistence, and teardown.

use std::sync::Arc;

use pi_agent_core::types::AgentEvent;
use pi_agent_core::types::AgentMessage;
use pi_ai::types::Message;
use pi_ai::types::UserContent;

use super::lock;
use super::types::AgentSessionEvent;
use super::types::AgentSessionEventListener;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Subscribe to session events; the returned closure removes this
    /// listener.
    pub fn subscribe(&self, listener: AgentSessionEventListener) -> impl FnOnce() + use<S> {
        let token = Arc::new(());
        lock(&self.event_listeners).push((Arc::clone(&token), listener));
        let weak = self.self_weak();
        move || {
            if let Some(session) = weak.upgrade() {
                lock(&session.event_listeners)
                    .retain(|(registered, _)| !Arc::ptr_eq(registered, &token));
            }
        }
    }

    /// Emit one session event to every listener.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "listeners clone the event; the owned shape mirrors upstream's emit"
    )]
    pub(crate) fn emit(&self, event: AgentSessionEvent) {
        for (_, listener) in lock(&self.event_listeners).iter() {
            listener(event.clone());
        }
    }

    /// The pending steering texts.
    #[must_use]
    pub fn steering_messages(&self) -> Vec<String> {
        lock(&self.steering_messages).clone()
    }

    /// The pending follow-up texts.
    #[must_use]
    pub fn follow_up_messages(&self) -> Vec<String> {
        lock(&self.follow_up_messages).clone()
    }

    /// The pending steering and follow-up count.
    #[must_use]
    pub fn pending_message_count(&self) -> usize {
        lock(&self.steering_messages).len() + lock(&self.follow_up_messages).len()
    }

    pub(crate) fn emit_queue_update(&self) {
        self.emit(AgentSessionEvent::QueueUpdate {
            steering: lock(&self.steering_messages).clone(),
            follow_up: lock(&self.follow_up_messages).clone(),
        });
    }

    /// Disconnect from the agent and drop every listener, upstream's
    /// `dispose`.
    pub fn dispose(&self) {
        self.extension_runner().invalidate(Some(
            "This extension ctx is stale after session replacement or reload. Do not use a captured pi or command ctx after ctx.newSession(), ctx.fork(), ctx.switchSession(), or ctx.reload(). For newSession, fork, and switchSession, move post-replacement work into withSession and use the ctx passed to withSession. For reload, do not use the old ctx after await ctx.reload().".to_owned(),
        ));
        self.agent.abort();
        let unsubscribe = lock(&self.unsubscribe_agent).take();
        if let Some(unsubscribe) = unsubscribe {
            unsubscribe();
        }
        lock(&self.event_listeners).clear();
    }

    pub(crate) async fn handle_agent_event(&self, event: AgentEvent) {
        if let AgentEvent::MessageStart { message } = &event
            && let Some(text) = user_text(message)
            && !text.is_empty()
        {
            let steering_index = lock(&self.steering_messages)
                .iter()
                .position(|item| item == &text);
            if let Some(index) = steering_index {
                lock(&self.steering_messages).remove(index);
                self.emit_queue_update();
            } else {
                let follow_up_index = lock(&self.follow_up_messages)
                    .iter()
                    .position(|item| item == &text);
                if let Some(index) = follow_up_index {
                    lock(&self.follow_up_messages).remove(index);
                    self.emit_queue_update();
                }
            }
        }

        let replacement = self.dispatch_extension_event(&event).await;
        self.forward_event(&event);

        if let AgentEvent::MessageEnd { message } = &event {
            let message = replacement.as_ref().unwrap_or(message);
            self.persist_message(message);
            if let AgentMessage::Standard(Message::Assistant(assistant)) = message {
                *lock(&self.last_assistant_message) = Some(assistant.clone());
                if assistant.stop_reason != pi_ai::types::StopReason::Error
                    && assistant.stop_reason != pi_ai::types::StopReason::Length
                {
                    self.overflow_recovery_attempted
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                }
                let attempt = self
                    .retry_attempt
                    .load(std::sync::atomic::Ordering::Relaxed);
                if assistant.stop_reason != pi_ai::types::StopReason::Error && attempt > 0 {
                    self.emit(AgentSessionEvent::AutoRetryEnd {
                        success: true,
                        attempt,
                        final_error: None,
                    });
                    self.retry_attempt
                        .store(0, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        if matches!(event, AgentEvent::TurnEnd { .. }) {
            self.flush_pending_custom_messages();
        }
    }

    fn forward_event(&self, event: &AgentEvent) {
        match event {
            AgentEvent::AgentStart => self.emit(AgentSessionEvent::AgentStart),
            AgentEvent::AgentEnd { messages } => self.emit(AgentSessionEvent::AgentEnd {
                messages: messages.clone(),
                will_retry: self.will_retry_after_agent_end(messages),
            }),
            AgentEvent::TurnStart => self.emit(AgentSessionEvent::TurnStart),
            AgentEvent::TurnEnd {
                message,
                tool_results,
            } => self.emit(AgentSessionEvent::TurnEnd {
                message: message.clone(),
                tool_results: tool_results.clone(),
            }),
            AgentEvent::MessageStart { message } => self.emit(AgentSessionEvent::MessageStart {
                message: message.clone(),
            }),
            AgentEvent::MessageUpdate {
                message,
                assistant_message_event,
            } => self.emit(AgentSessionEvent::MessageUpdate {
                message: message.clone(),
                assistant_message_event: assistant_message_event.clone(),
            }),
            AgentEvent::MessageEnd { message } => self.emit(AgentSessionEvent::MessageEnd {
                message: message.clone(),
            }),
            AgentEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                args,
            } => self.emit(AgentSessionEvent::ToolExecutionStart {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                args: args.clone(),
            }),
            AgentEvent::ToolExecutionUpdate {
                tool_call_id,
                tool_name,
                args,
                partial_result,
            } => self.emit(AgentSessionEvent::ToolExecutionUpdate {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                args: args.clone(),
                partial_result: partial_result.clone(),
            }),
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                tool_name,
                result,
                is_error,
            } => self.emit(AgentSessionEvent::ToolExecutionEnd {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                result: result.clone(),
                is_error: *is_error,
            }),
        }
    }

    fn persist_message(&self, message: &AgentMessage) {
        match message {
            AgentMessage::Custom(custom) if custom.role == "custom" => {
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
            AgentMessage::Standard(_) => {
                let _ = lock(&self.session_manager).append_message(message.clone());
            }
            AgentMessage::Custom(_) => {}
        }
    }

    pub(crate) fn will_retry_after_agent_end(&self, messages: &[AgentMessage]) -> bool {
        let settings = lock(&self.settings_manager).get_retry_settings();
        let max_retries = usize::try_from(settings.max_retries.max(0)).unwrap_or(0);
        if !settings.enabled || max_retries == 0 {
            return false;
        }
        messages
            .iter()
            .rev()
            .find_map(|message| match message {
                AgentMessage::Standard(Message::Assistant(assistant)) => Some(
                    pi_ai::utils::retry::is_retryable_assistant_error(assistant)
                        && !pi_ai::utils::overflow::is_context_overflow(
                            assistant,
                            self.model().map(|model| model.context_window),
                        ),
                ),
                _ => None,
            })
            .unwrap_or(false)
    }
}

fn user_text(message: &AgentMessage) -> Option<String> {
    match message {
        AgentMessage::Standard(Message::User(user)) => {
            Some(pi_ai::utils::text::content_text_with(&user.content, ""))
        }
        _ => None,
    }
}

fn custom_content(custom: &pi_agent_core::types::CustomAgentMessage) -> UserContent {
    custom
        .field("content")
        .and_then(|value| serde_json::from_value::<UserContent>(value.clone()).ok())
        .unwrap_or(UserContent::Blocks(Vec::new()))
}

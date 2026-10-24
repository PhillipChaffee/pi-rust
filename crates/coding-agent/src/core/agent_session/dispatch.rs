//! Agent-event forwarding into the extension runner.

use pi_agent_core::types::AgentEvent;
use pi_agent_core::types::AgentMessage;

use crate::extensions::events::ExtensionEvent;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    #[expect(
        clippy::too_many_lines,
        reason = "the dispatch is upstream's one match over every agent event"
    )]
    pub(crate) async fn dispatch_extension_event(
        &self,
        event: &AgentEvent,
    ) -> Option<AgentMessage> {
        let runner = self.extension_runner();
        match event {
            AgentEvent::AgentStart => {
                self.turn_index
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                runner.emit(ExtensionEvent::AgentStart).await;
            }
            AgentEvent::AgentEnd { messages } => {
                runner
                    .emit(ExtensionEvent::AgentEnd {
                        messages: messages.clone(),
                    })
                    .await;
            }
            AgentEvent::TurnStart => {
                runner
                    .emit(ExtensionEvent::TurnStart {
                        turn_index: self.turn_index.load(std::sync::atomic::Ordering::Relaxed),
                        timestamp: super::helpers::now_ms(),
                    })
                    .await;
            }
            AgentEvent::TurnEnd {
                message,
                tool_results,
            } => {
                runner
                    .emit(ExtensionEvent::TurnEnd {
                        turn_index: self.turn_index.load(std::sync::atomic::Ordering::Relaxed),
                        message: message.clone(),
                        tool_results: tool_results.clone(),
                    })
                    .await;
                self.turn_index
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            AgentEvent::MessageStart { message } => {
                runner
                    .emit(ExtensionEvent::MessageStart {
                        message: message.clone(),
                    })
                    .await;
            }
            AgentEvent::MessageUpdate {
                message,
                assistant_message_event,
            } => {
                runner
                    .emit(ExtensionEvent::MessageUpdate {
                        message: message.clone(),
                        assistant_message_event: assistant_message_event.clone(),
                    })
                    .await;
            }
            AgentEvent::MessageEnd { message } => {
                if let Some(replacement) = runner.emit_message_end(message.clone()).await {
                    let replacement = normalize_replacement(replacement);
                    let mut state = self.agent.state();
                    if let Some(last) = state.messages.last_mut() {
                        *last = replacement.clone();
                    }
                    drop(state);
                    return Some(replacement);
                }
            }
            AgentEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                args,
            } => {
                runner
                    .emit(ExtensionEvent::ToolExecutionStart {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: tool_name.clone(),
                        args: args.clone(),
                    })
                    .await;
            }
            AgentEvent::ToolExecutionUpdate {
                tool_call_id,
                tool_name,
                args,
                partial_result,
            } => {
                runner
                    .emit(ExtensionEvent::ToolExecutionUpdate {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: tool_name.clone(),
                        args: args.clone(),
                        partial_result: partial_result.clone(),
                    })
                    .await;
            }
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                tool_name,
                result,
                is_error,
            } => {
                runner
                    .emit(ExtensionEvent::ToolExecutionEnd {
                        tool_call_id: tool_call_id.clone(),
                        tool_name: tool_name.clone(),
                        result: result.clone(),
                        is_error: *is_error,
                    })
                    .await;
            }
        }
        None
    }
}

const fn normalize_replacement(message: AgentMessage) -> AgentMessage {
    message
}

//! The session-event switch, upstream's `handleEvent`.
//!
//! Partial port: the transcript/status arms below cover the live run,
//! tool execution, compaction, and retry. The replays that need the cache
//! statistics and the summarization retry sub-events are recorded as
//! remaining in the ticket report.

use std::rc::Rc;

use crate::core::agent_session::AgentSessionEvent;
use crate::modes::interactive::components::status_indicator::StatusIndicatorKind;
use crate::settings_manager::SettingsStorageKind;

use super::InteractiveMode;

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Handle one session event, upstream's `handleEvent`.
    pub(crate) fn handle_event(&self, event: AgentSessionEvent) {
        match event {
            AgentSessionEvent::TurnEnd { .. } => {}
            AgentSessionEvent::AgentStart => {
                self.pending_tools.borrow_mut().clear();
                if let Some(handler) = self.retry_escape_handler.borrow_mut().take() {
                    self.default_editor.set_on_escape(Some(handler));
                }
            }
            AgentSessionEvent::TurnStart => {
                if self.working_visible.get()
                    && self
                        .active_status_indicator
                        .borrow()
                        .as_ref()
                        .is_none_or(|indicator| indicator.kind() != StatusIndicatorKind::Working)
                {
                    self.show_working_status_indicator();
                }
                self.tui.request_render(false);
            }
            AgentSessionEvent::QueueUpdate { .. } => {
                self.update_pending_messages_display();
                self.tui.request_render(false);
            }
            AgentSessionEvent::EntryAppended { entry } => {
                if let crate::session_manager::SessionEntry::Custom(custom) = entry {
                    self.add_custom_entry_to_chat(&crate::session_manager::SessionEntry::Custom(
                        custom.clone(),
                    ));
                    self.tui.request_render(false);
                }
            }
            AgentSessionEvent::SessionInfoChanged { .. } => {
                self.update_terminal_title();
                self.footer.invalidate();
                self.tui.request_render(false);
            }
            AgentSessionEvent::ThinkingLevelChanged { .. } => {
                self.footer.invalidate();
                self.update_editor_border_color();
            }
            AgentSessionEvent::MessageStart { message } => {
                self.add_message_to_chat(&message, false);
                self.tui.request_render(false);
            }
            AgentSessionEvent::MessageEnd { message } => {
                if let pi_agent_core::types::AgentMessage::Standard(
                    pi_ai::types::Message::Assistant(_),
                ) = &message
                {
                    *self.streaming_message.borrow_mut() = None;
                    *self.streaming_component.borrow_mut() = None;
                }
                self.tui.request_render(false);
            }
            AgentSessionEvent::AgentEnd { .. } => {
                self.clear_status_indicator(Some(StatusIndicatorKind::Working));
                *self.streaming_component.borrow_mut() = None;
                *self.streaming_message.borrow_mut() = None;
                self.pending_tools.borrow_mut().clear();
                self.tui.request_render(false);
            }
            AgentSessionEvent::AgentSettled => {
                if self.shutdown_requested.get() {
                    self.shutdown_flag
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            AgentSessionEvent::CompactionStart { .. } => {
                let indicator = self.build_status(
                    StatusIndicatorKind::Compaction,
                    "Compacting context...".to_owned(),
                    None,
                );
                self.show_status_indicator(indicator);
                self.tui.request_render(false);
            }
            AgentSessionEvent::CompactionEnd { will_retry, .. } => {
                self.clear_status_indicator(Some(StatusIndicatorKind::Compaction));
                self.flush_compaction_queue(will_retry);
                self.tui.request_render(false);
            }
            AgentSessionEvent::AutoRetryStart { .. } => {
                let indicator =
                    self.build_status(StatusIndicatorKind::Retry, "Retrying...".to_owned(), None);
                self.show_status_indicator(indicator);
                self.tui.request_render(false);
            }
            AgentSessionEvent::AutoRetryEnd { success, .. } => {
                self.clear_status_indicator(Some(StatusIndicatorKind::Retry));
                if !success {
                    self.show_error("Retry failed");
                }
                self.tui.request_render(false);
            }
            AgentSessionEvent::SummarizationRetryScheduled { error_message, .. } => {
                self.show_error(&error_message);
            }
            AgentSessionEvent::SummarizationRetryAttemptStart { .. } => {
                self.clear_status_indicator(Some(StatusIndicatorKind::Retry));
            }
            AgentSessionEvent::SummarizationRetryFinished => {
                self.clear_status_indicator(Some(StatusIndicatorKind::Retry));
            }
            AgentSessionEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                args,
            } => {
                let existing = self.pending_tools.borrow().get(&tool_call_id).cloned();
                let component = existing.unwrap_or_else(|| {
                    let component = Rc::new(
                        crate::modes::interactive::components::tool_execution::ToolExecutionComponent::new(
                            &tool_name,
                            &tool_call_id,
                            args,
                            self.tool_options(),
                            self.registered_tool_definition(&tool_name),
                            self.render_request(),
                            &self.session().cwd(),
                        ),
                    );
                    component.set_expanded(self.tool_output_expanded.get());
                    self.chat_container.add_child(component.clone());
                    self.pending_tools
                        .borrow_mut()
                        .insert(tool_call_id.clone(), Rc::clone(&component));
                    component
                });
                component.mark_execution_started();
                self.tui.request_render(false);
            }
            AgentSessionEvent::ToolExecutionUpdate {
                tool_call_id,
                partial_result,
                ..
            } => {
                if let Some(component) = self.pending_tools.borrow().get(&tool_call_id).cloned() {
                    component.update_result(partial_result, false, true);
                    self.tui.request_render(false);
                }
            }
            AgentSessionEvent::ToolExecutionEnd {
                tool_call_id,
                result,
                is_error,
                ..
            } => {
                if let Some(component) = self.pending_tools.borrow().get(&tool_call_id).cloned() {
                    component.update_result(result, is_error, false);
                    self.pending_tools.borrow_mut().remove(&tool_call_id);
                    self.tui.request_render(false);
                }
            }
            AgentSessionEvent::MessageUpdate { message, .. } => {
                let pi_agent_core::types::AgentMessage::Standard(pi_ai::types::Message::Assistant(
                    assistant,
                )) = &message
                else {
                    return;
                };
                if let Some(component) = self.streaming_component.borrow().clone() {
                    *self.streaming_message.borrow_mut() = Some(assistant.clone());
                    component.update_content(assistant, true);
                    self.tui.request_render(false);
                }
            }
            AgentSessionEvent::BashExecutionUpdate { .. } => {}
        }
    }

    /// The status indicator constructor the event arms share.
    pub(crate) fn build_status(
        &self,
        kind: StatusIndicatorKind,
        message: String,
        indicator: Option<pi_tui::components::loader::LoaderIndicatorOptions>,
    ) -> Rc<crate::modes::interactive::components::status_indicator::StatusIndicator> {
        let request = self.render_request();
        let spinner: pi_tui::components::ColorFn = std::sync::Arc::new(|text: &str| {
            crate::modes::interactive::theme::theme::theme().fg("accent", text)
        });
        let text_color: pi_tui::components::ColorFn = std::sync::Arc::new(|text: &str| {
            crate::modes::interactive::theme::theme::theme().fg("muted", text)
        });
        Rc::new(
            crate::modes::interactive::components::status_indicator::StatusIndicator::new(
                kind, request, spinner, text_color, message, indicator,
            ),
        )
    }
}

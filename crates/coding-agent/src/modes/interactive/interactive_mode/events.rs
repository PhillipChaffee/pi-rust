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
    pub fn handle_event(&self, event: AgentSessionEvent) {
        match event {
            AgentSessionEvent::TurnEnd { .. } => {}
            AgentSessionEvent::AgentStart => {
                self.pending_tools.borrow_mut().clear();
                if let Some(handler) = self.retry_escape_handler.borrow_mut().take() {
                    self.default_editor.set_on_escape(Some(handler));
                }
            }
            AgentSessionEvent::TurnStart => {
                if self
                    .settings()
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get_show_terminal_progress()
                {
                    self.set_terminal_progress(true);
                }
                if self.working_visible.get() {
                    if self
                        .active_status_indicator
                        .borrow()
                        .as_ref()
                        .is_none_or(|indicator| indicator.kind() != StatusIndicatorKind::Working)
                    {
                        self.show_working_status_indicator();
                    }
                } else {
                    self.clear_status_indicator(None);
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
            AgentSessionEvent::MessageEnd { .. } => {
                self.tui.request_render(false);
            }
            AgentSessionEvent::AgentEnd { .. } => {
                self.clear_status_indicator(Some(StatusIndicatorKind::Working));
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
                if self
                    .settings()
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get_show_terminal_progress()
                {
                    self.set_terminal_progress(true);
                }
                // Escape aborts the compaction while it runs; the handler is
                // restored at compaction_end.
                if self.auto_compaction_escape_handler.borrow().is_none() {
                    // Keep the operation's previous handler for
                    // compaction_end, upstream's `autoCompactionEscapeHandler`.
                    *self.auto_compaction_escape_handler.borrow_mut() =
                        self.default_editor.on_escape();
                    let session = self.session();
                    let handler: crate::modes::interactive::components::custom_editor::AppActionHandler =
                        Rc::new(move || {
                            session.abort_compaction();
                        });
                    self.default_editor.set_on_escape(Some(handler));
                }
                let indicator = self.build_status(
                    StatusIndicatorKind::Compaction,
                    "Compacting context...".to_owned(),
                    None,
                );
                self.show_status_indicator(indicator);
                self.tui.request_render(false);
            }
            AgentSessionEvent::CompactionEnd {
                reason,
                result,
                aborted,
                will_retry,
                error_message,
            } => {
                if self
                    .settings()
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get_show_terminal_progress()
                {
                    self.set_terminal_progress(false);
                }
                if let Some(handler) = self.auto_compaction_escape_handler.borrow_mut().take() {
                    self.default_editor.set_on_escape(Some(handler));
                }
                self.clear_status_indicator(Some(StatusIndicatorKind::Compaction));
                if aborted {
                    if reason == crate::extensions::reasons::CompactionReason::Manual {
                        self.show_error("Compaction cancelled");
                    } else {
                        self.show_status("Auto-compaction cancelled");
                    }
                } else if let Some(result) = result {
                    let entries: Vec<crate::session_manager::FileEntry> = {
                        let session = self.session();
                        let manager = crate::core::agent_session::lock(&session.session_manager);
                        manager
                            .build_context_entries()
                            .into_iter()
                            .cloned()
                            .collect()
                    };
                    self.chat_container.clear();
                    self.assistant_rows.borrow_mut().clear();
                    if !entries.is_empty() {
                        self.render_session_entries(&entries[1..], false, false);
                    }
                    let message = pi_agent_core::types::AgentMessage::Custom(
                        crate::messages::create_compaction_summary_message(
                            &result.summary,
                            i64::try_from(result.tokens_before).unwrap_or(i64::MAX),
                            &crate::session_manager::now_iso8601(),
                        ),
                    );
                    self.add_message_to_chat(&message, false);
                    self.add_compaction_cost_notice("Compaction", &result.usage);
                    self.footer.invalidate();
                } else if let Some(message) = error_message {
                    if reason == crate::extensions::reasons::CompactionReason::Manual {
                        self.show_error(&message);
                    } else {
                        self.chat_container
                            .add_child(Rc::new(pi_tui::components::Spacer::new()));
                        let line: Rc<dyn pi_tui::tui::Component> =
                            Rc::new(pi_tui::components::Text::with_padding(
                                crate::modes::interactive::theme::theme::theme()
                                    .fg("error", &message),
                                1,
                                0,
                            ));
                        self.chat_container.add_child(line);
                    }
                }
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
            AgentSessionEvent::MessageUpdate { .. } => {}
            AgentSessionEvent::BashExecutionUpdate { .. } => {}
        }
    }

    /// The status indicator constructor the event arms share.
    pub fn build_status(
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

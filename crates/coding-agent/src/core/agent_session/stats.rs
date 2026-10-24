//! Statistics, context usage, exports, and last-assistant text.

use pi_ai::types::AssistantBlock;
use pi_ai::types::Message;
use pi_ai::types::StopReason;

use super::lock;
use super::types::SessionStats;
use super::types::SessionTokens;
use crate::export_html::ExportOptions;
use crate::export_html::ThemeColorsSource;
use crate::session_manager::FileEntry;
use crate::session_manager::SessionEntry;
use crate::settings_manager::SettingsStorage;
use crate::usage_totals::add_usage_to_totals;
use crate::usage_totals::create_usage_totals;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Whether auto-compaction is enabled.
    #[must_use]
    pub fn auto_compaction_enabled(&self) -> bool {
        lock(&self.settings_manager).get_compaction_enabled()
    }

    /// Toggle auto-compaction.
    pub fn set_auto_compaction_enabled(&self, enabled: bool) {
        lock(&self.settings_manager).set_compaction_enabled(enabled);
    }

    /// Aggregate statistics over every session entry.
    #[must_use]
    pub fn session_stats(&self) -> SessionStats {
        let manager = lock(&self.session_manager);
        let mut stats = SessionStats {
            session_file: manager.session_file().map(str::to_owned),
            session_id: manager.session_id().to_owned(),
            user_messages: 0,
            assistant_messages: 0,
            tool_calls: 0,
            tool_results: 0,
            total_messages: 0,
            tokens: SessionTokens::default(),
            cost: 0.0,
            context_usage: None,
        };
        let mut totals = create_usage_totals();
        for entry in manager.entries() {
            match entry {
                FileEntry::Entry(SessionEntry::BranchSummary(branch)) => {
                    if let Some(usage) = &branch.usage {
                        add_usage_to_totals(&mut totals, usage);
                    }
                }
                FileEntry::Entry(SessionEntry::Compaction(compaction)) => {
                    if let Some(usage) = &compaction.usage {
                        add_usage_to_totals(&mut totals, usage);
                    }
                }
                FileEntry::Entry(SessionEntry::Message(entry)) => {
                    stats.total_messages += 1;
                    let Some(message) = &entry.message else {
                        continue;
                    };
                    match message {
                        pi_agent_core::types::AgentMessage::Standard(Message::User(_)) => {
                            stats.user_messages += 1;
                        }
                        pi_agent_core::types::AgentMessage::Standard(Message::ToolResult(
                            result,
                        )) => {
                            stats.tool_results += 1;
                            if let Some(usage) = &result.usage {
                                add_usage_to_totals(&mut totals, usage);
                            }
                        }
                        pi_agent_core::types::AgentMessage::Standard(Message::Assistant(
                            assistant,
                        )) => {
                            stats.assistant_messages += 1;
                            stats.tool_calls += assistant
                                .content
                                .iter()
                                .filter(|block| matches!(block, AssistantBlock::ToolCall(_)))
                                .count();
                            add_usage_to_totals(&mut totals, &assistant.usage);
                        }
                        pi_agent_core::types::AgentMessage::Custom(_) => {}
                    }
                }
                FileEntry::Session(_) | FileEntry::Entry(_) | FileEntry::Other(_) => {}
            }
        }
        drop(manager);
        stats.tokens = SessionTokens {
            input: totals.input,
            output: totals.output,
            cache_read: totals.cache_read,
            cache_write: totals.cache_write,
            total: totals.input + totals.output + totals.cache_read + totals.cache_write,
        };
        stats.cost = totals.cost;
        stats.context_usage = self.context_usage();
        stats
    }

    /// The context usage reading, upstream's `getContextUsage`.
    #[must_use]
    pub fn context_usage(&self) -> Option<crate::extensions::runtime::ContextUsage> {
        let model = self.model()?;
        if model.context_window == 0 {
            return None;
        }
        let manager = lock(&self.session_manager);
        let branch = manager.get_branch(None);
        let compaction_index = branch
            .iter()
            .rposition(|entry| matches!(entry, FileEntry::Entry(SessionEntry::Compaction(_))));
        if let Some(index) = compaction_index {
            let has_post_compaction_usage = branch[index + 1..].iter().rev().any(|entry| {
                let FileEntry::Entry(SessionEntry::Message(message_entry)) = entry else {
                    return false;
                };
                let Some(pi_agent_core::types::AgentMessage::Standard(Message::Assistant(
                    assistant,
                ))) = &message_entry.message
                else {
                    return false;
                };
                assistant.stop_reason != StopReason::Aborted
                    && assistant.stop_reason != StopReason::Error
                    && crate::compaction::calculate_context_tokens(&assistant.usage) > 0
            });
            if !has_post_compaction_usage {
                return Some(crate::extensions::runtime::ContextUsage {
                    tokens: None,
                    context_window: model.context_window,
                    percent: None,
                });
            }
        }
        drop(manager);
        let messages = self.agent.state().messages.clone();
        let estimate = crate::compaction::estimate_context_tokens(&messages);
        Some(crate::extensions::runtime::ContextUsage {
            tokens: Some(estimate.tokens),
            context_window: model.context_window,
            percent: Some(percent_of(estimate.tokens, model.context_window)),
        })
    }

    /// The last assistant message's trimmed text.
    #[must_use]
    pub fn last_assistant_text(&self) -> Option<String> {
        let state = self.agent.state();
        let message = state.messages.iter().rev().find_map(|message| {
            let pi_agent_core::types::AgentMessage::Standard(Message::Assistant(assistant)) =
                message
            else {
                return None;
            };
            if assistant.stop_reason == StopReason::Aborted && assistant.content.is_empty() {
                return None;
            }
            Some(assistant.clone())
        });
        drop(state);
        let message = message?;
        let text: String = message
            .content
            .iter()
            .filter_map(|block| match block {
                AssistantBlock::Text(content) => Some(content.text.as_str()),
                _ => None,
            })
            .collect();
        let trimmed = text.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    }

    /// Export the session to HTML.
    ///
    /// # Errors
    /// When the export fails.
    pub fn export_to_html(
        &self,
        output_path: Option<String>,
        theme_name: Option<String>,
        theme_source: &dyn ThemeColorsSource,
    ) -> Result<String, crate::export_html::ExportError> {
        let manager = lock(&self.session_manager);
        let state = self.agent.state();
        crate::export_html::export_session_to_html(
            &manager,
            Some(&state),
            ExportOptions {
                output_path,
                theme_name: theme_name.or_else(|| lock(&self.settings_manager).get_theme()),
                tool_renderer: None,
            },
            theme_source,
        )
    }

    /// Export the current branch to JSONL.
    ///
    /// # Errors
    /// When the write fails.
    pub fn export_to_jsonl(
        &self,
        output_path: Option<&str>,
    ) -> Result<String, crate::session_manager::SessionManagerError> {
        crate::session_export::export_session_to_jsonl(
            &lock(&self.session_manager),
            output_path,
            None,
        )
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the context-usage percent restates upstream's `(tokens / contextWindow) * 100`"
)]
fn percent_of(tokens: u64, window: u64) -> f64 {
    (tokens as f64 / window as f64) * 100.0
}

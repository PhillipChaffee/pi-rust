//! Tree navigation and branch summarization.

use pi_ai::types::Message;

use super::helpers::bridge_signal;
use super::lock;
use super::types::NavigateTreeOptions;
use super::types::NavigateTreeOutcome;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::reasons::TreePreparation;
use crate::extensions::results::SessionBeforeResult;
use crate::session_manager::FileEntry;
use crate::session_manager::SessionEntry;
use crate::settings_manager::SettingsStorage;
use tokio_util::sync::CancellationToken;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Navigate to another node in the session tree.
    ///
    /// # Errors
    /// When the session is streaming or compacting, no model is available
    /// for summarization, or the target entry is unknown.
    #[expect(
        clippy::too_many_lines,
        reason = "the navigation is upstream's one prepare/summarize/branch flow"
    )]
    pub async fn navigate_tree(
        &self,
        target_id: &str,
        options: NavigateTreeOptions,
    ) -> Result<NavigateTreeOutcome, String> {
        if self.is_streaming() {
            return Err(
                "Wait for the current response to finish before navigating the session tree."
                    .to_owned(),
            );
        }
        if self.is_compacting() {
            return Err("Wait for the current compaction or tree navigation to finish before navigating the session tree.".to_owned());
        }
        let old_leaf_id = lock(&self.session_manager).get_leaf_id().map(str::to_owned);
        if Some(target_id) == old_leaf_id.as_deref() {
            return Ok(NavigateTreeOutcome {
                cancelled: false,
                ..NavigateTreeOutcome::default()
            });
        }
        let target_entry = lock(&self.session_manager)
            .get_entry(target_id)
            .cloned()
            .ok_or_else(|| format!("Entry {target_id} not found"))?;
        let FileEntry::Entry(target_entry) = target_entry else {
            return Err(format!("Entry {target_id} not found"));
        };

        let collected = {
            let manager = lock(&self.session_manager);
            crate::compaction::collect_entries_for_branch_summary(
                &manager,
                old_leaf_id.as_deref(),
                target_id,
            )
        };
        let mut custom_instructions = options.custom_instructions;
        let mut replace_instructions = options.replace_instructions;
        let mut label = options.label;
        let preparation = TreePreparation {
            target_id: target_id.to_owned(),
            old_leaf_id: old_leaf_id.clone(),
            common_ancestor_id: collected.common_ancestor_id.clone(),
            entries_to_summarize: collected.entries.clone(),
            user_wants_summary: options.summarize,
            custom_instructions: custom_instructions.clone(),
            replace_instructions: Some(replace_instructions),
            label: label.clone(),
        };
        let token = CancellationToken::new();
        *lock(&self.branch_summary_abort) = Some(token.clone());

        let mut extension_summary = None;
        let mut from_extension = false;
        let runner = self.extension_runner();
        if runner.has_handlers("session_before_tree") {
            let signal = bridge_signal(Some(&token)).0;
            if let SessionBeforeResult::Tree {
                cancel,
                summary,
                custom_instructions: instructions,
                replace_instructions: replace,
                label: new_label,
            } = runner
                .emit(ExtensionEvent::SessionBeforeTree {
                    preparation,
                    signal,
                })
                .await
            {
                if cancel {
                    self.clear_branch_summary();
                    return Ok(NavigateTreeOutcome {
                        cancelled: true,
                        ..NavigateTreeOutcome::default()
                    });
                }
                if options.summarize {
                    extension_summary = summary;
                    from_extension = extension_summary.is_some();
                }
                if let Some(instructions) = instructions {
                    custom_instructions = Some(instructions);
                }
                if let Some(replace) = replace {
                    replace_instructions = replace;
                }
                if let Some(new_label) = new_label {
                    label = Some(new_label);
                }
            }
        }

        let mut summary_text = None;
        let mut summary_details = None;
        let mut summary_usage = None;
        if options.summarize && !collected.entries.is_empty() && extension_summary.is_none() {
            let model = self.model().ok_or("No model available for summarization")?;
            let Some(auth) = self.request_auth(&model, true).await else {
                self.clear_branch_summary();
                return Err(crate::auth_guidance::format_no_api_key_found_message(
                    &model.provider.0,
                ));
            };
            let reserve = lock(&self.settings_manager)
                .get_branch_summary_settings()
                .reserve_tokens;
            let retry = lock(&self.settings_manager).get_retry_settings();
            let policy = super::helpers::retry_policy(&retry);
            let callbacks =
                self.summarization_callbacks(crate::extensions::reasons::CompactionReason::Manual);
            let result = crate::compaction::generate_branch_summary(
                &collected.entries,
                &crate::compaction::GenerateBranchSummaryOptions {
                    model: &auth.model,
                    api_key: auth.api_key.as_deref(),
                    headers: auth.headers.as_ref(),
                    env: auth.env.as_ref(),
                    signal: &token,
                    custom_instructions: custom_instructions.as_deref(),
                    replace_instructions,
                    reserve_tokens: Some(reserve),
                    stream_fn: self.agent.stream_function().as_ref(),
                    retry: Some(&policy),
                    callbacks: Some(&callbacks),
                },
            )
            .await;
            if result.aborted {
                self.clear_branch_summary();
                return Ok(NavigateTreeOutcome {
                    cancelled: true,
                    aborted: true,
                    ..NavigateTreeOutcome::default()
                });
            }
            if let Some(error) = result.error {
                self.clear_branch_summary();
                return Err(error);
            }
            summary_text = result.summary;
            summary_usage = result.usage;
            summary_details = Some(serde_json::json!({
                "readFiles": result.read_files.unwrap_or_default(),
                "modifiedFiles": result.modified_files.unwrap_or_default(),
            }));
        } else if let Some(summary) = extension_summary {
            summary_text = Some(summary.summary);
            summary_details = summary.details;
            summary_usage = summary.usage;
        }

        let (new_leaf_id, editor_text) = new_leaf(&target_entry);
        let summarized = summary_text.is_some();
        let summary_entry = if let Some(summary) = summary_text {
            let entry_id = lock(&self.session_manager)
                .branch_with_summary(
                    new_leaf_id.as_deref(),
                    &summary,
                    summary_details,
                    Some(from_extension),
                    summary_usage,
                )
                .map_err(|error| error.to_string())?;
            if let Some(label) = label {
                let _ = lock(&self.session_manager).append_label_change(&entry_id, Some(&label));
            }
            lock(&self.session_manager)
                .get_entry(&entry_id)
                .and_then(|entry| match entry {
                    FileEntry::Entry(SessionEntry::BranchSummary(summary)) => Some(summary.clone()),
                    _ => None,
                })
        } else {
            if new_leaf_id.is_none() {
                lock(&self.session_manager).reset_leaf();
            } else if let Some(new_leaf) = &new_leaf_id {
                lock(&self.session_manager)
                    .branch(new_leaf)
                    .map_err(|error| error.to_string())?;
            }
            if let Some(label) = label {
                let _ = lock(&self.session_manager).append_label_change(target_id, Some(&label));
            }
            None
        };

        let context = lock(&self.session_manager).build_session_context();
        self.agent.set_messages(&context.messages);
        let new_leaf_id = lock(&self.session_manager).get_leaf_id().map(str::to_owned);
        runner
            .emit(ExtensionEvent::SessionTree {
                new_leaf_id,
                old_leaf_id,
                summary_entry: summary_entry.clone(),
                from_extension: summarized.then_some(from_extension),
            })
            .await;
        self.clear_branch_summary();
        Ok(NavigateTreeOutcome {
            editor_text,
            cancelled: false,
            aborted: false,
            summary_entry,
        })
    }

    fn clear_branch_summary(&self) {
        *lock(&self.branch_summary_abort) = None;
        self.resolve_idle_wait_if_idle();
    }
}

fn new_leaf(entry: &SessionEntry) -> (Option<String>, Option<String>) {
    match entry {
        SessionEntry::Message(message) => match &message.message {
            Some(pi_agent_core::types::AgentMessage::Standard(Message::User(user))) => (
                message.base.parent_id.clone(),
                Some(pi_ai::utils::text::content_text_with(&user.content, "")),
            ),
            _ => (message.base.id.clone(), None),
        },
        SessionEntry::CustomMessage(custom) => (
            custom.base.parent_id.clone(),
            custom
                .content
                .as_ref()
                .map(|content| pi_ai::utils::text::content_text_with(content, "")),
        ),
        _ => (entry.base().id.clone(), None),
    }
}

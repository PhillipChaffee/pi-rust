//! Manual compaction and the shared one-shot flow.

use super::compact_auth::RequestAuth;
use super::helpers::bridge_signal;
use super::helpers::model_key;
use super::helpers::retry_policy;
use super::lock;
use super::types::AgentSessionEvent;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::reasons::CompactionReason;
use crate::extensions::results::SessionBeforeResult;
use crate::session_manager::FileEntry;
use crate::session_manager::SessionEntry;
use crate::settings_manager::SettingsStorage;
use tokio_util::sync::CancellationToken;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Manually compact the session context.
    ///
    /// # Errors
    /// When no model/auth is available, the session is too small, an
    /// extension cancels, or the summary call fails.
    pub async fn compact(
        &self,
        custom_instructions: Option<String>,
    ) -> Result<crate::compaction::CompactionResult, String> {
        self.abort().await;
        let token = CancellationToken::new();
        *lock(&self.compaction_abort) = Some(token.clone());
        self.emit(AgentSessionEvent::CompactionStart {
            reason: CompactionReason::Manual,
        });
        let result = self
            .compact_once(CompactionReason::Manual, custom_instructions, &token, false)
            .await;
        *lock(&self.compaction_abort) = None;
        self.resolve_idle_wait_if_idle();
        match result {
            Ok(Some(result)) => {
                self.emit(AgentSessionEvent::CompactionEnd {
                    reason: CompactionReason::Manual,
                    result: Some(result.clone()),
                    aborted: false,
                    will_retry: false,
                    error_message: None,
                });
                Ok(result)
            }
            Ok(None) => Err(
                if matches!(
                    self.branch_entries().last(),
                    Some(SessionEntry::Compaction(_))
                ) {
                    "Already compacted".to_owned()
                } else {
                    "Nothing to compact (session too small)".to_owned()
                },
            ),
            Err(error) => {
                let aborted = error == "Compaction cancelled";
                self.emit(AgentSessionEvent::CompactionEnd {
                    reason: CompactionReason::Manual,
                    result: None,
                    aborted,
                    will_retry: false,
                    error_message: (!aborted).then(|| format!("Compaction failed: {error}")),
                });
                Err(error)
            }
        }
    }

    pub(crate) async fn compact_once(
        &self,
        reason: CompactionReason,
        custom_instructions: Option<String>,
        token: &CancellationToken,
        will_retry: bool,
    ) -> Result<Option<crate::compaction::CompactionResult>, String> {
        let Some(model) = self.model() else {
            return Err(crate::auth_guidance::format_no_model_selected_message());
        };
        let settings =
            lock(&self.settings_manager).get_compaction_settings(Some(&model_key(&model)))?;
        let Some(auth) = self
            .request_auth(&model, reason != CompactionReason::Overflow)
            .await
        else {
            return Err(crate::auth_guidance::format_no_api_key_found_message(
                &model.provider.0,
            ));
        };
        let branches = self.branch_entries();
        let Some(preparation) = crate::compaction::prepare_compaction(&branches, settings) else {
            return Ok(None);
        };

        let mut from_extension = false;
        let mut extension_compaction = None;
        let runner = self.extension_runner();
        if runner.has_handlers("session_before_compact") {
            let signal = bridge_signal(Some(token)).0;
            if let SessionBeforeResult::Compact { cancel, compaction } = runner
                .emit(ExtensionEvent::SessionBeforeCompact {
                    preparation: preparation.clone(),
                    branch_entries: branches,
                    custom_instructions: custom_instructions.clone(),
                    reason,
                    will_retry,
                    signal,
                })
                .await
            {
                if cancel {
                    return Err("Compaction cancelled".to_owned());
                }
                if compaction.is_some() {
                    extension_compaction = compaction;
                    from_extension = true;
                }
            }
        }

        let result = if let Some(compaction) = extension_compaction {
            compaction
        } else {
            self.default_compaction(
                preparation,
                &auth,
                custom_instructions.as_deref(),
                token,
                reason,
            )
            .await?
        };
        if token.is_cancelled() {
            return Err("Compaction cancelled".to_owned());
        }
        self.persist_compaction(&result, from_extension, reason, will_retry)
            .await;
        Ok(Some(result))
    }

    async fn default_compaction(
        &self,
        preparation: crate::compaction::CompactionPreparation,
        auth: &RequestAuth,
        custom_instructions: Option<&str>,
        token: &CancellationToken,
        reason: CompactionReason,
    ) -> Result<crate::compaction::CompactionResult, String> {
        let retry = lock(&self.settings_manager).get_retry_settings();
        let policy = retry_policy(&retry);
        let callbacks = self.summarization_callbacks(reason);
        crate::compaction::compact(
            preparation,
            &auth.model,
            auth.api_key.as_deref(),
            auth.headers.as_ref(),
            custom_instructions,
            Some(token),
            Some(self.thinking_level()),
            self.agent.stream_function().as_ref(),
            auth.env.as_ref(),
            Some(&policy),
            Some(&callbacks),
            None,
        )
        .await
        .map_err(|error| error.to_string())
    }

    async fn persist_compaction(
        &self,
        result: &crate::compaction::CompactionResult,
        from_extension: bool,
        reason: CompactionReason,
        will_retry: bool,
    ) {
        let tokens_before = i64::try_from(result.tokens_before).unwrap_or(i64::MAX);
        let _ = lock(&self.session_manager).append_compaction(
            &result.summary,
            &result.first_kept_entry_id,
            tokens_before,
            result.details.clone(),
            Some(from_extension),
            Some(result.usage),
        );
        let context = lock(&self.session_manager).build_session_context();
        self.agent.set_messages(&context.messages);
        let saved = lock(&self.session_manager)
            .get_branch(None)
            .into_iter()
            .rev()
            .find_map(|entry| match entry {
                FileEntry::Entry(SessionEntry::Compaction(compaction))
                    if compaction.summary == result.summary =>
                {
                    Some(compaction.clone())
                }
                _ => None,
            });
        if let Some(entry) = saved {
            self.extension_runner()
                .emit(ExtensionEvent::SessionCompact {
                    compaction_entry: entry,
                    from_extension,
                    reason,
                    will_retry,
                })
                .await;
        }
    }
}

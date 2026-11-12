//! The interactive session-share surface, upstream
//! `src/modes/interactive/session-share.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! The export → Radius upload → GitHub CLI gist ladder landed with the
//! session manager and export slice as [`crate::session_share`]; this module
//! is its upstream-faithful path under `modes::interactive` and re-exports
//! that port unchanged rather than keeping a second copy in sync.
//!
//! Restatement: upstream's `SessionShareContext` swaps a `BorderedLoader`
//! into the editor container, focuses it, and restores the editor on abort
//! or completion. The port restates that editor choreography as the
//! injected [`ShareUserInterface`], [`ShareProcessRunner`], and
//! [`ShareHttpClient`] seams — the loader's abort signal and editor swap
//! belong to the app frame, which is the remaining #131 slice — while the
//! export shape and the Radius/gist ladder stay 1:1.

pub use crate::session_share::{
    GistOutcome, RadiusUploadOutcome, ShareHttpClient, ShareProcessRunner, ShareSessionSource,
    ShareUserInterface, export_session_for_share, share_session,
};

use pi_ai::types::BoxedFuture;

use crate::core::agent_session::AgentSession;
use crate::export_html::ExportedTool;
use crate::session_manager::SessionManager;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> ShareSessionSource for std::sync::Arc<AgentSession<S>> {
    fn session_manager(&self) -> SessionManager {
        crate::core::agent_session::lock(&self.session_manager).clone()
    }

    fn system_prompt(&self) -> Option<String> {
        Some(AgentSession::system_prompt(self))
    }

    fn tools(&self) -> Vec<ExportedTool> {
        self.get_all_tools()
            .into_iter()
            .map(|tool| ExportedTool {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
            })
            .collect()
    }

    fn export_to_html(&mut self, file_path: &str) -> BoxedFuture<'_, Result<(), String>> {
        let file_path = file_path.to_owned();
        Box::pin(async move {
            AgentSession::export_to_html(
                self,
                Some(file_path),
                crate::modes::interactive::theme::theme::current_theme_name(),
                &crate::export_html::NoThemeSource,
            )
            .map(|_| ())
            .map_err(|error| error.to_string())
        })
    }

    fn has_radius_provider(&self) -> bool {
        self.model_runtime().get_provider("radius").is_some()
    }

    fn radius_token(&mut self) -> BoxedFuture<'_, Option<String>> {
        Box::pin(async move {
            let auth = self
                .model_runtime()
                .get_auth("radius", None)
                .await
                .ok()
                .flatten()?;
            if let Some(api_key) = auth.auth.api_key {
                return Some(api_key);
            }
            auth.auth.headers.as_ref().and_then(|headers| {
                headers
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                    .and_then(|(_, value)| value.as_ref())
                    .and_then(|value| value.strip_prefix("Bearer "))
                    .map(str::to_owned)
            })
        })
    }
}

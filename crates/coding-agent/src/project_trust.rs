//! The project-trust resolution flow, upstream's
//! `packages/coding-agent/src/core/project-trust.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The extension half of the flow — `emitProjectTrustEvent` and its
//! `LoadExtensionsResult` input — lands with the extension system (map
//! ticket "pi-coding-agent: extension system"), so the resolution takes the
//! extension decision through the [`ExtensionTrustGate`] seam and the
//! per-extension error reporting through the closure the runner would
//! receive. Everything else ports 1:1: the override, the resource probe,
//! the stored decision, the default posture, the selector prompt, and the
//! no-UI refusal.

use crate::config::{APP_NAME, CONFIG_DIR_NAME};
use crate::settings_manager::DefaultProjectTrust;
use crate::trust_manager::{
    ProjectTrustOption, ProjectTrustStore, ProjectTrustUpdate, get_project_trust_options,
    has_trust_requiring_project_resources,
};

/// The app mode, upstream's `AppMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    /// The interactive terminal session.
    Interactive,
    /// The one-shot print mode.
    Print,
    /// The JSON-event mode.
    Json,
    /// The RPC mode.
    Rpc,
}

/// The extension gate's outcome, upstream's `emitProjectTrustEvent` result:
/// the decision plus one error per failing extension path.
pub type ExtensionTrustResult = (Option<ExtensionTrustOutcome>, Vec<(String, String)>);

/// The extension system's trust gate, upstream's `emitProjectTrustEvent`
/// result narrowed to the shape the resolution consumes.
pub trait ExtensionTrustGate: Send + Sync {
    /// Offer the trust event to the loaded extensions, upstream's
    /// `emitProjectTrustEvent`.
    fn emit_project_trust<'a>(
        &'a self,
        event: &'a ExtensionTrustEvent,
    ) -> pi_ai::types::BoxedFuture<'a, ExtensionTrustResult>;
}

/// The trust event, upstream's `{ type: "project_trust", cwd }`.
#[derive(Debug, Clone)]
pub struct ExtensionTrustEvent {
    /// The cwd the trust question concerns.
    pub cwd: String,
}

/// The extensions' outcome, upstream's `{ trusted, remember }`.
#[derive(Debug, Clone, Copy)]
pub struct ExtensionTrustOutcome {
    /// `"yes"` trusts, anything else refuses.
    pub trusted_is_yes: bool,
    /// Whether to persist the decision.
    pub remember: bool,
}

/// The host context the selector needs, upstream's `ProjectTrustContext`
/// narrowed to `ui.select` and `hasUI` — the seam
/// [`crate::trust_manager::ProjectTrustSelector`] carries.
pub use crate::trust_manager::ProjectTrustSelector;

/// The resolution inputs, upstream's `ResolveProjectTrustedOptions`.
pub struct ResolveProjectTrustedOptions<'a> {
    /// The cwd to resolve trust for.
    pub cwd: &'a str,
    /// The trust store.
    pub trust_store: &'a ProjectTrustStore,
    /// A forced decision, upstream's `trustOverride`.
    pub trust_override: Option<bool>,
    /// The default posture, upstream's `defaultProjectTrust`.
    pub default_project_trust: Option<DefaultProjectTrust>,
    /// The extension gate, upstream's `extensionsResult`.
    pub extension_gate: Option<&'a dyn ExtensionTrustGate>,
    /// The selector, upstream's `projectTrustContext`.
    pub project_trust_context: &'a dyn ProjectTrustSelector,
    /// The per-extension error sink, upstream's `onExtensionError`.
    pub on_extension_error: Option<&'a dyn Fn(&str)>,
}

impl std::fmt::Debug for ResolveProjectTrustedOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolveProjectTrustedOptions")
            .field("cwd", &self.cwd)
            .field("trust_override", &self.trust_override)
            .field("default_project_trust", &self.default_project_trust)
            .finish_non_exhaustive()
    }
}

/// The prompt text, upstream's `formatProjectTrustPrompt`.
fn format_project_trust_prompt(cwd: &str) -> String {
    format!(
        "Trust project folder?\n{cwd}\n\nThis allows {APP_NAME} to load {CONFIG_DIR_NAME} settings and resources, install missing project packages, and execute project extensions."
    )
}

/// The selector's option list, upstream's `selectProjectTrustOption` with
/// the session-only variants included.
async fn select_project_trust_option(
    cwd: &str,
    selector: &dyn ProjectTrustSelector,
) -> Option<ProjectTrustOption> {
    let options = get_project_trust_options(cwd, true);
    let selected = selector
        .select(
            format_project_trust_prompt(cwd),
            options.iter().map(|option| option.label.clone()).collect(),
        )
        .await;
    options
        .into_iter()
        .find(|option| Some(&option.label) == selected.as_ref())
}

/// Save a prompt result's store updates, upstream's
/// `saveProjectTrustPromptResult`.
fn save_project_trust_prompt_result(
    trust_store: &ProjectTrustStore,
    result: &ProjectTrustOption,
) -> Result<(), crate::trust_manager::TrustError> {
    if result.updates.is_empty() {
        return Ok(());
    }
    trust_store.set_many(
        &result
            .updates
            .iter()
            .map(|update| ProjectTrustUpdate {
                path: update.path.clone(),
                decision: update.decision,
            })
            .collect::<Vec<_>>(),
    )
}

/// Resolve whether the project at `cwd` is trusted, upstream's
/// `resolveProjectTrusted`.
///
/// The override wins, unresourceful projects trust implicitly, extensions
/// answer first, then the store, then the default posture, then the prompt
/// — and a host without a UI refuses.
///
/// # Errors
/// The store writes the prompt saves can fail with.
pub async fn resolve_project_trusted(
    options: ResolveProjectTrustedOptions<'_>,
) -> Result<bool, crate::trust_manager::TrustError> {
    if let Some(trust_override) = options.trust_override {
        return Ok(trust_override);
    }
    if !has_trust_requiring_project_resources(options.cwd) {
        return Ok(true);
    }

    if let Some(gate) = options.extension_gate {
        let (result, errors) = gate
            .emit_project_trust(&ExtensionTrustEvent {
                cwd: options.cwd.to_string(),
            })
            .await;
        for (extension_path, error) in errors {
            if let Some(report) = options.on_extension_error {
                report(&format!(
                    "Extension \"{extension_path}\" project_trust error: {error}"
                ));
            }
        }
        if let Some(outcome) = result {
            let trusted = outcome.trusted_is_yes;
            if outcome.remember {
                options.trust_store.set(options.cwd, Some(trusted))?;
            }
            return Ok(trusted);
        }
    }

    let decision = options.trust_store.get(options.cwd)?;
    if decision.is_some() {
        return Ok(decision.unwrap_or(false));
    }

    match options
        .default_project_trust
        .unwrap_or(DefaultProjectTrust::Ask)
    {
        DefaultProjectTrust::Always => return Ok(true),
        DefaultProjectTrust::Never => return Ok(false),
        DefaultProjectTrust::Ask => {}
    }

    if !options.project_trust_context.has_ui() {
        return Ok(false);
    }

    let Some(selected) =
        select_project_trust_option(options.cwd, options.project_trust_context).await
    else {
        return Ok(false);
    };
    save_project_trust_prompt_result(options.trust_store, &selected)?;
    Ok(selected.trusted)
}

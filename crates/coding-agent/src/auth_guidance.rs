//! Login guidance strings, upstream's `src/core/auth-guidance.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::config::get_docs_path;

/// The provider display name standing in for a missing provider, upstream's
/// `UNKNOWN_PROVIDER`.
const UNKNOWN_PROVIDER: &str = "unknown";

/// Where to send a user who needs to log in, upstream's
/// `getProviderLoginHelp`.
#[must_use]
pub fn get_provider_login_help() -> String {
    [
        "Use /login to log into a provider via OAuth or API key. See:".to_string(),
        format!("  {}/providers.md", get_docs_path()),
        format!("  {}/models.md", get_docs_path()),
    ]
    .join("\n")
}

/// The message when no models are available, upstream's
/// `formatNoModelsAvailableMessage`.
#[must_use]
pub fn format_no_models_available_message() -> String {
    format!("No models available. {}", get_provider_login_help())
}

/// The message when no model is selected, upstream's
/// `formatNoModelSelectedMessage`.
#[must_use]
pub fn format_no_model_selected_message() -> String {
    format!(
        "No model selected.\n\n{}\n\nThen use /model to select a model.",
        get_provider_login_help()
    )
}

/// The message when a provider has no API key, upstream's
/// `formatNoApiKeyFoundMessage`; the unknown provider reads as "the
/// selected model".
#[must_use]
pub fn format_no_api_key_found_message(provider: &str) -> String {
    let provider_display = if provider == UNKNOWN_PROVIDER {
        "the selected model"
    } else {
        provider
    };
    format!(
        "No API key found for {provider_display}.\n\n{}",
        get_provider_login_help()
    )
}

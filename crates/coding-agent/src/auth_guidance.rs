//! Provider-login guidance strings, upstream's
//! `src/core/auth-guidance.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::config::get_docs_path;

const UNKNOWN_PROVIDER: &str = "unknown";

/// The docs-relative provider login help, upstream's `getProviderLoginHelp`.
#[must_use]
pub fn get_provider_login_help() -> String {
    let docs = get_docs_path();
    ["providers.md", "models.md"]
        .iter()
        .map(|name| docs.join(name).to_string_lossy().into_owned())
        .fold(
            String::from("Use /login to log into a provider via OAuth or API key. See:"),
            |joined, line| format!("{joined}\n  {line}"),
        )
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

/// The message when no API key is found, upstream's
/// `formatNoApiKeyFoundMessage`: the unknown provider reads as the
/// selected model.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_guidance_messages() {
        let help = get_provider_login_help();
        assert!(help.starts_with("Use /login to log into a provider via OAuth or API key. See:"));
        assert!(help.contains("/providers.md"));
        assert_eq!(
            format_no_models_available_message(),
            format!("No models available. {help}")
        );
        assert_eq!(
            format_no_model_selected_message(),
            format!("No model selected.\n\n{help}\n\nThen use /model to select a model.")
        );
        assert_eq!(
            format_no_api_key_found_message("openai"),
            format!("No API key found for openai.\n\n{help}")
        );
        assert_eq!(
            format_no_api_key_found_message("unknown"),
            format!("No API key found for the selected model.\n\n{help}")
        );
    }
}

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

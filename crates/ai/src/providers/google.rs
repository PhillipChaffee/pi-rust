//! The `google` provider factory, ported from
//! `packages/ai/src/providers/google.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `Google` provider, upstream's `googleProvider()`.
#[must_use]
pub fn google_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "google".to_owned(),
        name: Some("Google".to_owned()),
        base_url: Some("https://generativelanguage.googleapis.com/v1beta".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("Gemini API key", &["GEMINI_API_KEY"])),
            oauth: None,
        },
        models: get_builtin_models("google"),
        api: ProviderApi::Single(crate::api::google_generative_ai()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

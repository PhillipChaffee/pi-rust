//! The `mistral` provider factory, ported from
//! `packages/ai/src/providers/mistral.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `Mistral` provider, upstream's `mistralProvider()`.
#[must_use]
pub fn mistral_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "mistral".to_owned(),
        name: Some("Mistral".to_owned()),
        base_url: Some("https://api.mistral.ai".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("Mistral API key", &["MISTRAL_API_KEY"])),
            oauth: None,
        },
        models: get_builtin_models("mistral"),
        api: ProviderApi::Single(crate::api::mistral_conversations()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

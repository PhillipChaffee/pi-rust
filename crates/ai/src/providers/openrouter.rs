//! The OpenRouter provider factory, ported from
//! `packages/ai/src/providers/openrouter.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::api::ApiMap;
use crate::auth::helpers::{env_api_key_auth, lazy_oauth};
use crate::auth::oauth::load_openrouter_oauth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The OpenRouter provider, upstream's `openrouterProvider()`.
#[must_use]
pub fn openrouter_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "openrouter".to_owned(),
        name: Some("OpenRouter".to_owned()),
        base_url: Some("https://openrouter.ai/api/v1".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "OpenRouter API key",
                &["OPENROUTER_API_KEY"],
            )),
            oauth: Some(lazy_oauth(crate::auth::helpers::LazyOAuthInput {
                name: "OpenRouter OAuth".to_owned(),
                is_subscription: None,
                login_label: Some("Sign in with OpenRouter".to_owned()),
                load: Arc::new(|| load_openrouter_oauth()),
            })),
        },
        models: get_builtin_models("openrouter"),
        api: ProviderApi::ByApi(ApiMap::from([
            (
                "anthropic-messages".to_owned(),
                crate::api::anthropic_messages(),
            ),
            (
                "openai-completions".to_owned(),
                crate::api::openai_completions(),
            ),
        ])),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

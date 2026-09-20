//! The Fireworks provider factory, ported from
//! `packages/ai/src/providers/fireworks.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::api::ApiMap;
use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The Fireworks provider, upstream's `fireworksProvider()`.
#[must_use]
pub fn fireworks_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "fireworks".to_owned(),
        name: Some("Fireworks".to_owned()),
        base_url: Some("https://api.fireworks.ai/inference".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "Fireworks API key",
                &["FIREWORKS_API_KEY"],
            )),
            oauth: None,
        },
        models: get_builtin_models("fireworks"),
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

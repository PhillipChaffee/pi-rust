//! The `baseten` provider factory, ported from
//! `packages/ai/src/providers/baseten.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `Baseten` provider, upstream's `basetenProvider()`.
#[must_use]
pub fn baseten_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "baseten".to_owned(),
        name: Some("Baseten".to_owned()),
        base_url: Some("https://inference.baseten.co/v1".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("Baseten API key", &["BASETEN_API_KEY"])),
            oauth: None,
        },
        models: get_builtin_models("baseten"),
        api: ProviderApi::Single(crate::api::openai_completions()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

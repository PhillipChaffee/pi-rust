//! The `cerebras` provider factory, ported from
//! `packages/ai/src/providers/cerebras.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `Cerebras` provider, upstream's `cerebrasProvider()`.
#[must_use]
pub fn cerebras_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "cerebras".to_owned(),
        name: Some("Cerebras".to_owned()),
        base_url: Some("https://api.cerebras.ai/v1".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("Cerebras API key", &["CEREBRAS_API_KEY"])),
            oauth: None,
        },
        models: get_builtin_models("cerebras"),
        api: ProviderApi::Single(crate::api::openai_completions()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

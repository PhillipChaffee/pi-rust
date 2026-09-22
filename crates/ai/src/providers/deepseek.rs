//! The `deepseek` provider factory, ported from
//! `packages/ai/src/providers/deepseek.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `DeepSeek` provider, upstream's `deepseekProvider()`.
#[must_use]
pub fn deepseek_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "deepseek".to_owned(),
        name: Some("DeepSeek".to_owned()),
        base_url: Some("https://api.deepseek.com".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("DeepSeek API key", &["DEEPSEEK_API_KEY"])),
            oauth: None,
        },
        models: get_builtin_models("deepseek"),
        api: ProviderApi::Single(crate::api::openai_completions()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

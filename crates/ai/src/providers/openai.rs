//! The `openai` provider factory, ported from
//! `packages/ai/src/providers/openai.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `OpenAI` provider, upstream's `openaiProvider()`.
#[must_use]
pub fn openai_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "openai".to_owned(),
        name: Some("OpenAI".to_owned()),
        base_url: Some("https://api.openai.com/v1".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("OpenAI API key", &["OPENAI_API_KEY"])),
            oauth: None,
        },
        models: get_builtin_models("openai"),
        api: ProviderApi::Single(crate::api::openai_responses()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

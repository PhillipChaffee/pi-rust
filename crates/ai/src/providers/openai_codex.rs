//! The OpenAI Codex provider factory, ported from
//! `packages/ai/src/providers/openai-codex.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::lazy_oauth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The OpenAI Codex provider, upstream's `openaiCodexProvider()`.
#[must_use]
pub fn openai_codex_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "openai-codex".to_owned(),
        name: Some("OpenAI Codex".to_owned()),
        base_url: Some("https://chatgpt.com/backend-api".to_owned()),
        auth: ProviderAuth {
            api_key: None,
            oauth: Some(lazy_oauth(crate::auth::helpers::LazyOAuthInput {
                name: "OpenAI (ChatGPT Plus/Pro)".to_owned(),
                is_subscription: Some(true),
                login_label: None,
                load: Arc::new(|| crate::auth::oauth::load_openai_codex_oauth()),
            })),
        },
        models: get_builtin_models("openai-codex"),
        api: ProviderApi::Single(crate::api::openai_codex_responses()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

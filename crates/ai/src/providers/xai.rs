//! The xAI provider factory, ported from
//! `packages/ai/src/providers/xai.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::{env_api_key_auth, lazy_oauth};
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The xAI provider, upstream's `xaiProvider()`.
#[must_use]
pub fn xai_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "xai".to_owned(),
        name: Some("xAI".to_owned()),
        base_url: Some("https://api.x.ai/v1".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("xAI API key", &["XAI_API_KEY"])),
            oauth: Some(lazy_oauth(crate::auth::helpers::LazyOAuthInput {
                name: "xAI (Grok/X subscription)".to_owned(),
                is_subscription: Some(true),
                login_label: Some("Sign in with `SuperGrok` or X Premium".to_owned()),
                load: Arc::new(|| crate::auth::oauth::load_xai_oauth()),
            })),
        },
        models: get_builtin_models("xai"),
        api: ProviderApi::Single(crate::api::openai_responses()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

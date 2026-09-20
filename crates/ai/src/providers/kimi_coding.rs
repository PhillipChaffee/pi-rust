//! The Kimi For Coding provider factory, ported from
//! `packages/ai/src/providers/kimi-coding.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::{env_api_key_auth, lazy_oauth};
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The Kimi For Coding provider, upstream's `kimiCodingProvider()`.
#[must_use]
pub fn kimi_coding_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "kimi-coding".to_owned(),
        name: Some("Kimi For Coding".to_owned()),
        base_url: Some("https://api.kimi.com/coding".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("Kimi API key", &["KIMI_API_KEY"])),
            oauth: Some(lazy_oauth(crate::auth::helpers::LazyOAuthInput {
                name: "Kimi Code (subscription)".to_owned(),
                is_subscription: Some(true),
                login_label: Some("Sign in with Kimi Code".to_owned()),
                load: Arc::new(|| crate::auth::oauth::load_kimi_coding_oauth()),
            })),
        },
        models: get_builtin_models("kimi-coding"),
        api: ProviderApi::Single(crate::api::anthropic_messages()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

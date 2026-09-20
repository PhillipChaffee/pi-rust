//! The `minimax` provider factory, ported from
//! `packages/ai/src/providers/minimax.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `MiniMax` provider, upstream's `minimaxProvider()`.
#[must_use]
pub fn minimax_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "minimax".to_owned(),
        name: Some("MiniMax".to_owned()),
        base_url: Some("https://api.minimax.io/anthropic".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("MiniMax API key", &["MINIMAX_API_KEY"])),
            oauth: None,
        },
        models: get_builtin_models("minimax"),
        api: ProviderApi::Single(crate::api::anthropic_messages()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

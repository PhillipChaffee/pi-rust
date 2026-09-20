//! The `minimax-cn` provider factory, ported from
//! `packages/ai/src/providers/minimax-cn.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `MiniMax CN` provider, upstream's `minimaxCnProvider()`.
#[must_use]
pub fn minimax_cn_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "minimax-cn".to_owned(),
        name: Some("MiniMax CN".to_owned()),
        base_url: Some("https://api.minimaxi.com/anthropic".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "MiniMax CN API key",
                &["MINIMAX_CN_API_KEY"],
            )),
            oauth: None,
        },
        models: get_builtin_models("minimax-cn"),
        api: ProviderApi::Single(crate::api::anthropic_messages()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

//! The `zai-coding-cn` provider factory, ported from
//! `packages/ai/src/providers/zai-coding-cn.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `Z.AI Coding CN` provider, upstream's `zaiCodingCnProvider()`.
#[must_use]
pub fn zai_coding_cn_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "zai-coding-cn".to_owned(),
        name: Some("Z.AI Coding CN".to_owned()),
        base_url: Some("https://open.bigmodel.cn/api/coding/paas/v4".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "Z.AI Coding CN API key",
                &["ZAI_CODING_CN_API_KEY"],
            )),
            oauth: None,
        },
        models: get_builtin_models("zai-coding-cn"),
        api: ProviderApi::Single(crate::api::openai_completions()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

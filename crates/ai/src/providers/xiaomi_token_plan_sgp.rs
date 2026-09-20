//! The `xiaomi-token-plan-sgp` provider factory, ported from
//! `packages/ai/src/providers/xiaomi-token-plan-sgp.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `Xiaomi Token Plan SGP` provider, upstream's `xiaomiTokenPlanSgpProvider()`.
#[must_use]
pub fn xiaomi_token_plan_sgp_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "xiaomi-token-plan-sgp".to_owned(),
        name: Some("Xiaomi Token Plan SGP".to_owned()),
        base_url: Some("https://token-plan-sgp.xiaomimimo.com/v1".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "Xiaomi Token Plan SGP API key",
                &["XIAOMI_TOKEN_PLAN_SGP_API_KEY"],
            )),
            oauth: None,
        },
        models: get_builtin_models("xiaomi-token-plan-sgp"),
        api: ProviderApi::Single(crate::api::openai_completions()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

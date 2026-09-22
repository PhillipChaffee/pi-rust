//! The `qwen-token-plan-individual` provider factory, ported from
//! `packages/ai/src/providers/qwen-token-plan-individual.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `Qwen Token Plan Individual` provider, upstream's `qwenTokenPlanIndividualProvider()`.
#[must_use]
pub fn qwen_token_plan_individual_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "qwen-token-plan-individual".to_owned(),
        name: Some("Qwen Token Plan Individual".to_owned()),
        base_url: Some(
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1".to_owned(),
        ),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "Qwen Token Plan Individual API key",
                &["QWEN_TOKEN_PLAN_API_KEY"],
            )),
            oauth: None,
        },
        models: get_builtin_models("qwen-token-plan-individual"),
        api: ProviderApi::Single(crate::api::openai_completions()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

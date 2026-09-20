//! The `azure-openai-responses` provider factory, ported from
//! `packages/ai/src/providers/azure-openai-responses.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `Azure OpenAI` provider, upstream's `azureOpenAIResponsesProvider()`.
#[must_use]
pub fn azure_openai_responses_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "azure-openai-responses".to_owned(),
        name: Some("Azure OpenAI".to_owned()),
        base_url: None,
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "Azure OpenAI API key",
                &["AZURE_OPENAI_API_KEY"],
            )),
            oauth: None,
        },
        models: get_builtin_models("azure-openai-responses"),
        api: ProviderApi::Single(crate::api::azure_openai_responses()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

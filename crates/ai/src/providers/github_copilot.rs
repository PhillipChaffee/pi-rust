//! The GitHub Copilot provider factory, ported from
//! `packages/ai/src/providers/github-copilot.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::api::ApiMap;
use crate::auth::helpers::{env_api_key_auth, lazy_oauth};
use crate::auth::oauth::load_github_copilot_oauth;
use crate::auth::types::{Credential, ProviderAuth};
use crate::models::{
    CreateProviderOptions, FilterModelsFn, Provider, ProviderApi, create_provider,
};
use crate::providers::catalog::get_builtin_models;

/// Copilot's OAuth credential carries `availableModelIds`; when present and
/// well-formed it restricts the visible catalog, upstream's `filterModels`.
fn copilot_filter_models() -> FilterModelsFn {
    Arc::new(
        |models: Vec<crate::types::Model>, credential: Option<&Credential>| {
            let Some(credential) = credential else {
                return models;
            };
            let Some(oauth) = credential.as_oauth() else {
                return models;
            };
            let Some(serde_json::Value::Array(available)) = oauth.extra.get("availableModelIds")
            else {
                return models;
            };
            if !available.iter().all(serde_json::Value::is_string) {
                return models;
            }
            let available: BTreeSet<String> = available
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
                .collect();
            models
                .into_iter()
                .filter(|model| available.contains(&model.id))
                .collect()
        },
    )
}

/// The GitHub Copilot provider, upstream's `githubCopilotProvider()`.
#[must_use]
pub fn github_copilot_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "github-copilot".to_owned(),
        name: Some("GitHub Copilot".to_owned()),
        base_url: Some("https://api.individual.githubcopilot.com".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "GitHub Copilot token",
                &["COPILOT_GITHUB_TOKEN"],
            )),
            oauth: Some(lazy_oauth(crate::auth::helpers::LazyOAuthInput {
                name: "GitHub Copilot".to_owned(),
                is_subscription: Some(true),
                login_label: None,
                load: Arc::new(|| load_github_copilot_oauth()),
            })),
        },
        models: get_builtin_models("github-copilot"),
        api: ProviderApi::ByApi(ApiMap::from([
            (
                "anthropic-messages".to_owned(),
                crate::api::anthropic_messages(),
            ),
            (
                "openai-completions".to_owned(),
                crate::api::openai_completions(),
            ),
            (
                "openai-responses".to_owned(),
                crate::api::openai_responses(),
            ),
        ])),
        headers: None,
        fetch_models: None,
        filter_models: Some(copilot_filter_models()),
    }))
}

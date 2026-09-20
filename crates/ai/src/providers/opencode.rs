//! The OpenCode Zen provider factory, ported from
//! `packages/ai/src/providers/opencode.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::api::ApiMap;
use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;
use crate::providers::opencode_headers::with_opencode_session_header;

/// The OpenCode Zen provider, upstream's `opencodeProvider()`.
#[must_use]
pub fn opencode_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "opencode".to_owned(),
        name: Some("OpenCode Zen".to_owned()),
        base_url: None,
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("OpenCode API key", &["OPENCODE_API_KEY"])),
            oauth: None,
        },
        models: get_builtin_models("opencode"),
        api: ProviderApi::ByApi(ApiMap::from([
            (
                "anthropic-messages".to_owned(),
                with_opencode_session_header(crate::api::anthropic_messages()),
            ),
            (
                "google-generative-ai".to_owned(),
                with_opencode_session_header(crate::api::google_generative_ai()),
            ),
            (
                "openai-completions".to_owned(),
                with_opencode_session_header(crate::api::openai_completions()),
            ),
            (
                "openai-responses".to_owned(),
                with_opencode_session_header(crate::api::openai_responses()),
            ),
        ])),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

//! The Cloudflare Workers AI provider factory, ported from
//! `packages/ai/src/providers/cloudflare-workers-ai.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;
use crate::providers::cloudflare_auth::cloudflare_workers_ai_auth;
use crate::providers::cloudflare_stream::cloudflare_streams;

/// The Cloudflare Workers AI provider, upstream's
/// `cloudflareWorkersAIProvider()`.
#[must_use]
pub fn cloudflare_workers_ai_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "cloudflare-workers-ai".to_owned(),
        name: Some("Cloudflare Workers AI".to_owned()),
        base_url: None,
        auth: ProviderAuth {
            api_key: Some(cloudflare_workers_ai_auth()),
            oauth: None,
        },
        models: get_builtin_models("cloudflare-workers-ai"),
        api: ProviderApi::Single(cloudflare_streams(crate::api::openai_completions())),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

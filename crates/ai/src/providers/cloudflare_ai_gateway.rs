//! The Cloudflare AI Gateway provider factory, ported from
//! `packages/ai/src/providers/cloudflare-ai-gateway.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::api::ApiMap;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;
use crate::providers::cloudflare_auth::cloudflare_ai_gateway_auth;
use crate::providers::cloudflare_stream::cloudflare_streams;

/// The Cloudflare AI Gateway provider, upstream's
/// `cloudflareAIGatewayProvider()`.
///
/// The api map is pinned to all three APIs: models.dev's gateway catalog
/// drops and restores `workers-ai/*` (openai-completions) entries over time,
/// and inference from `models` alone would otherwise reject the
/// openai-completions entry whenever the generated catalog happens to
/// contain none.
#[must_use]
pub fn cloudflare_ai_gateway_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "cloudflare-ai-gateway".to_owned(),
        name: Some("Cloudflare AI Gateway".to_owned()),
        base_url: None,
        auth: ProviderAuth {
            api_key: Some(cloudflare_ai_gateway_auth()),
            oauth: None,
        },
        models: get_builtin_models("cloudflare-ai-gateway"),
        api: ProviderApi::ByApi(ApiMap::from([
            (
                "anthropic-messages".to_owned(),
                cloudflare_streams(crate::api::anthropic_messages()),
            ),
            (
                "openai-completions".to_owned(),
                cloudflare_streams(crate::api::openai_completions()),
            ),
            (
                "openai-responses".to_owned(),
                cloudflare_streams(crate::api::openai_responses()),
            ),
        ])),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

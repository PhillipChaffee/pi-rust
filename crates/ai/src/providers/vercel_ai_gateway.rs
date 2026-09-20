//! The `vercel-ai-gateway` provider factory, ported from
//! `packages/ai/src/providers/vercel-ai-gateway.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::env_api_key_auth;
use crate::auth::types::ProviderAuth;
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;

/// The `Vercel AI Gateway` provider, upstream's `vercelAIGatewayProvider()`.
#[must_use]
pub fn vercel_ai_gateway_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "vercel-ai-gateway".to_owned(),
        name: Some("Vercel AI Gateway".to_owned()),
        base_url: Some("https://ai-gateway.vercel.sh".to_owned()),
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth(
                "Vercel AI Gateway API key",
                &["AI_GATEWAY_API_KEY"],
            )),
            oauth: None,
        },
        models: get_builtin_models("vercel-ai-gateway"),
        api: ProviderApi::Single(crate::api::anthropic_messages()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

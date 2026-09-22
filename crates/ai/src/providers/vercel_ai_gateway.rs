//! The `vercel-ai-gateway` provider factory, ported from
//! `packages/ai/src/providers/vercel-ai-gateway.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Vercel AI Gateway` provider, upstream's `vercelAIGatewayProvider()`.
    vercel_ai_gateway_provider,
    "vercel-ai-gateway",
    "Vercel AI Gateway",
    Some("https://ai-gateway.vercel.sh"),
    "Vercel AI Gateway API key",
    ["AI_GATEWAY_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::anthropic_messages()),
);

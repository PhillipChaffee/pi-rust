//! The `zai` provider factory, ported from
//! `packages/ai/src/providers/zai.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Z.AI` provider, upstream's `zaiProvider()`.
    zai_provider,
    "zai",
    "Z.AI",
    Some("https://api.z.ai/api/coding/paas/v4"),
    "Z.AI API key",
    ["ZAI_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

//! The `mistral` provider factory, ported from
//! `packages/ai/src/providers/mistral.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Mistral` provider, upstream's `mistralProvider()`.
    mistral_provider,
    "mistral",
    "Mistral",
    Some("https://api.mistral.ai"),
    "Mistral API key",
    ["MISTRAL_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::mistral_conversations()),
);

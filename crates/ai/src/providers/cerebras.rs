//! The `cerebras` provider factory, ported from
//! `packages/ai/src/providers/cerebras.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Cerebras` provider, upstream's `cerebrasProvider()`.
    cerebras_provider,
    "cerebras",
    "Cerebras",
    Some("https://api.cerebras.ai/v1"),
    "Cerebras API key",
    ["CEREBRAS_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

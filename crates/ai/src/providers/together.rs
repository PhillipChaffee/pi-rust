//! The `together` provider factory, ported from
//! `packages/ai/src/providers/together.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Together` provider, upstream's `togetherProvider()`.
    together_provider,
    "together",
    "Together",
    Some("https://api.together.ai/v1"),
    "Together API key",
    ["TOGETHER_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

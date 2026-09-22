//! The `deepseek` provider factory, ported from
//! `packages/ai/src/providers/deepseek.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `DeepSeek` provider, upstream's `deepseekProvider()`.
    deepseek_provider,
    "deepseek",
    "DeepSeek",
    Some("https://api.deepseek.com"),
    "DeepSeek API key",
    ["DEEPSEEK_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

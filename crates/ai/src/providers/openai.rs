//! The `openai` provider factory, ported from
//! `packages/ai/src/providers/openai.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `OpenAI` provider, upstream's `openaiProvider()`.
    openai_provider,
    "openai",
    "OpenAI",
    Some("https://api.openai.com/v1"),
    "OpenAI API key",
    ["OPENAI_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_responses()),
);

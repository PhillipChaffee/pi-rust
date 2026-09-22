//! The `moonshotai` provider factory, ported from
//! `packages/ai/src/providers/moonshotai.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Moonshot AI` provider, upstream's `moonshotaiProvider()`.
    moonshotai_provider,
    "moonshotai",
    "Moonshot AI",
    Some("https://api.moonshot.ai/v1"),
    "Moonshot AI API key",
    ["MOONSHOT_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

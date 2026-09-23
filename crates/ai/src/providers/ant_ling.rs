//! The `ant-ling` provider factory, ported from
//! `packages/ai/src/providers/ant-ling.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Ant Ling` provider, upstream's `antLingProvider()`.
    ant_ling_provider,
    "ant-ling",
    "Ant Ling",
    Some("https://api.ant-ling.com/v1"),
    "Ant Ling API key",
    ["ANT_LING_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

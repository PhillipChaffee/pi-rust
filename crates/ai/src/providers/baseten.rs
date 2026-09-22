//! The `baseten` provider factory, ported from
//! `packages/ai/src/providers/baseten.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Baseten` provider, upstream's `basetenProvider()`.
    baseten_provider,
    "baseten",
    "Baseten",
    Some("https://inference.baseten.co/v1"),
    "Baseten API key",
    ["BASETEN_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

//! The Fireworks provider factory, ported from
//! `packages/ai/src/providers/fireworks.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The Fireworks provider, upstream's `fireworksProvider()`.
    fireworks_provider,
    "fireworks",
    "Fireworks",
    Some("https://api.fireworks.ai/inference"),
    "Fireworks API key",
    ["FIREWORKS_API_KEY"],
    crate::models::ProviderApi::ByApi(crate::api::ApiMap::from([ ( "anthropic-messages".to_owned(), crate::api::anthropic_messages(), ), ( "openai-completions".to_owned(), crate::api::openai_completions(), ), ])),
);

//! The `xiaomi` provider factory, ported from
//! `packages/ai/src/providers/xiaomi.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Xiaomi` provider, upstream's `xiaomiProvider()`.
    xiaomi_provider,
    "xiaomi",
    "Xiaomi",
    Some("https://api.xiaomimimo.com/v1"),
    "Xiaomi API key",
    ["XIAOMI_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

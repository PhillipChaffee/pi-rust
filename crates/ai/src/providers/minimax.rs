//! The `minimax` provider factory, ported from
//! `packages/ai/src/providers/minimax.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `MiniMax` provider, upstream's `minimaxProvider()`.
    minimax_provider,
    "minimax",
    "MiniMax",
    Some("https://api.minimax.io/anthropic"),
    "MiniMax API key",
    ["MINIMAX_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::anthropic_messages()),
);

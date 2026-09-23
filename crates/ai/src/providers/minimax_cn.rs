//! The `minimax-cn` provider factory, ported from
//! `packages/ai/src/providers/minimax-cn.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `MiniMax CN` provider, upstream's `minimaxCnProvider()`.
    minimax_cn_provider,
    "minimax-cn",
    "MiniMax CN",
    Some("https://api.minimaxi.com/anthropic"),
    "MiniMax CN API key",
    ["MINIMAX_CN_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::anthropic_messages()),
);

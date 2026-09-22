//! The `moonshotai-cn` provider factory, ported from
//! `packages/ai/src/providers/moonshotai-cn.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Moonshot AI CN` provider, upstream's `moonshotaiCnProvider()`.
    moonshotai_cn_provider,
    "moonshotai-cn",
    "Moonshot AI CN",
    Some("https://api.moonshot.cn/v1"),
    "Moonshot AI API key",
    ["MOONSHOT_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

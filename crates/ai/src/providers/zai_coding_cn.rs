//! The `zai-coding-cn` provider factory, ported from
//! `packages/ai/src/providers/zai-coding-cn.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Z.AI Coding CN` provider, upstream's `zaiCodingCnProvider()`.
    zai_coding_cn_provider,
    "zai-coding-cn",
    "Z.AI Coding CN",
    Some("https://open.bigmodel.cn/api/coding/paas/v4"),
    "Z.AI Coding CN API key",
    ["ZAI_CODING_CN_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

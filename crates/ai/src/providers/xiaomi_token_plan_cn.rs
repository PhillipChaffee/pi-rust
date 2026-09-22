//! The `xiaomi-token-plan-cn` provider factory, ported from
//! `packages/ai/src/providers/xiaomi-token-plan-cn.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Xiaomi Token Plan CN` provider, upstream's `xiaomiTokenPlanCnProvider()`.
    xiaomi_token_plan_cn_provider,
    "xiaomi-token-plan-cn",
    "Xiaomi Token Plan CN",
    Some("https://token-plan-cn.xiaomimimo.com/v1"),
    "Xiaomi Token Plan CN API key",
    ["XIAOMI_TOKEN_PLAN_CN_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

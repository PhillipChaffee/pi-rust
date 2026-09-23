//! The `qwen-token-plan-cn` provider factory, ported from
//! `packages/ai/src/providers/qwen-token-plan-cn.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Qwen Token Plan CN` provider, upstream's `qwenTokenPlanCnProvider()`.
    qwen_token_plan_cn_provider,
    "qwen-token-plan-cn",
    "Qwen Token Plan CN",
    Some("https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"),
    "Qwen Token Plan CN API key",
    ["QWEN_TOKEN_PLAN_CN_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

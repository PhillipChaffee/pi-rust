//! The `qwen-token-plan-individual` provider factory, ported from
//! `packages/ai/src/providers/qwen-token-plan-individual.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Qwen Token Plan Individual` provider, upstream's `qwenTokenPlanIndividualProvider()`.
    qwen_token_plan_individual_provider,
    "qwen-token-plan-individual",
    "Qwen Token Plan Individual",
    Some("https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"),
    "Qwen Token Plan Individual API key",
    ["QWEN_TOKEN_PLAN_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

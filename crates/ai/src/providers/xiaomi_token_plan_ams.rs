//! The `xiaomi-token-plan-ams` provider factory, ported from
//! `packages/ai/src/providers/xiaomi-token-plan-ams.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Xiaomi Token Plan AMS` provider, upstream's `xiaomiTokenPlanAmsProvider()`.
    xiaomi_token_plan_ams_provider,
    "xiaomi-token-plan-ams",
    "Xiaomi Token Plan AMS",
    Some("https://token-plan-ams.xiaomimimo.com/v1"),
    "Xiaomi Token Plan AMS API key",
    ["XIAOMI_TOKEN_PLAN_AMS_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

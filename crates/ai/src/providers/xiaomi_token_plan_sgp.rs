//! The `xiaomi-token-plan-sgp` provider factory, ported from
//! `packages/ai/src/providers/xiaomi-token-plan-sgp.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Xiaomi Token Plan SGP` provider, upstream's `xiaomiTokenPlanSgpProvider()`.
    xiaomi_token_plan_sgp_provider,
    "xiaomi-token-plan-sgp",
    "Xiaomi Token Plan SGP",
    Some("https://token-plan-sgp.xiaomimimo.com/v1"),
    "Xiaomi Token Plan SGP API key",
    ["XIAOMI_TOKEN_PLAN_SGP_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::openai_completions()),
);

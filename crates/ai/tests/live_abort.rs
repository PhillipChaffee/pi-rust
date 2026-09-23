//! The provider abort suites, ported from `packages/ai/test/abort.test.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: one `#[tokio::test]`
//! per upstream `it`, kept in upstream order.
//!
//! Like upstream's `describe.skipIf`, a block without its provider credential
//! returns early — an env var for the api-key providers, the Azure and
//! Bedrock credential guards for the cloud blocks, and the resolved
//! openai-codex token (credential store first, then env) for the Codex
//! block. The Anthropic block gates on `ANTHROPIC_OAUTH_TOKEN` per upstream,
//! not the credential store. Upstream passes only a block's extras and rides
//! compat's env-key resolution; the port hands every env-gated block the
//! gate's value as the api key — the value compat resolves anyway — because
//! the extras-carrying blocks dispatch through an adapter directly and the
//! adapters do not apply compat's env fallback. Upstream retries each probe
//! three times (`{ retry: 3 }`); the shared harness carries no retry wrapper,
//! so the port runs each probe once.

use pi_ai::api::google_shared::GoogleThinkingControl;
use pi_ai::types::{Api, Model, ThinkingLevel};

mod common;

use common::live;
use common::live::LiveOptions;

/// The options the Google block sends: the gate's key plus the enabled
/// thinking control, upstream's `{ thinking: { enabled: true } }`.
fn google_options(api_key: String) -> LiveOptions {
    LiveOptions {
        api_key: Some(api_key),
        google_thinking: Some(GoogleThinkingControl {
            enabled: true,
            budget_tokens: None,
            level: None,
        }),
        ..LiveOptions::default()
    }
}

/// The options the Anthropic block sends: the OAuth token as the key and
/// extended thinking at a 2048-token budget, upstream's
/// `{ thinkingEnabled: true, thinkingBudgetTokens: 2048 }`.
fn anthropic_options(api_key: String) -> LiveOptions {
    LiveOptions {
        api_key: Some(api_key),
        thinking_enabled: Some(true),
        thinking_budget_tokens: Some(2048),
        ..LiveOptions::default()
    }
}

/// The options the high-reasoning blocks send, upstream's
/// `{ reasoningEffort: "high" }` (Together AI and Baseten).
fn high_reasoning_options(api_key: String) -> LiveOptions {
    LiveOptions {
        api_key: Some(api_key),
        reasoning: Some(ThinkingLevel::High),
        ..LiveOptions::default()
    }
}

/// The gpt-4o-mini catalog model retargeted at the openai-completions wire
/// with its compat map stripped, upstream's
/// `{ ...getModel("openai", "gpt-4o-mini"), api: "openai-completions" }`
/// spread.
fn openai_completions_llm() -> Model {
    let mut llm = live::model("openai", "gpt-4o-mini");
    llm.api = Api::from("openai-completions");
    llm.compat = None;
    llm
}

/// The Azure model and options the block's probes share: the api key plus the
/// deployment override when `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` maps one for
/// gpt-4o-mini, upstream's `azureOptions`. Absent without the live Azure
/// credentials, upstream's `hasAzureOpenAICredentials` gate.
fn azure_model_and_options() -> Option<(Model, LiveOptions)> {
    if !live::has_azure_openai_credentials() {
        return None;
    }
    let api_key = live::env_key("AZURE_OPENAI_API_KEY")?;
    let llm = live::model("azure-openai-responses", "gpt-4o-mini");
    let azure_deployment_name = live::azure_deployment_name(&llm.id);
    Some((
        llm,
        LiveOptions {
            api_key: Some(api_key),
            azure_deployment_name,
            ..LiveOptions::default()
        },
    ))
}

/// Upstream `"Google Provider Abort"` / `"should abort mid-stream"`:
/// `google/gemini-2.5-flash` with thinking enabled.
#[tokio::test]
async fn google_abort_mid_stream() {
    let Some(api_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let llm = live::model("google", "gemini-2.5-flash");
    live::test_abort_signal(&llm, &google_options(api_key)).await;
}

/// Upstream `"Google Provider Abort"` / `"should handle immediate abort"`:
/// `google/gemini-2.5-flash` with thinking enabled.
#[tokio::test]
async fn google_immediate_abort() {
    let Some(api_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let llm = live::model("google", "gemini-2.5-flash");
    live::test_immediate_abort(&llm, &google_options(api_key)).await;
}

/// Upstream `"OpenAI Completions Provider Abort"` /
/// `"should abort mid-stream"`: the gpt-4o-mini catalog model forced onto the
/// openai-completions wire.
#[tokio::test]
async fn openai_completions_abort_mid_stream() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = openai_completions_llm();
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"OpenAI Completions Provider Abort"` /
/// `"should handle immediate abort"`: the gpt-4o-mini catalog model forced
/// onto the openai-completions wire.
#[tokio::test]
async fn openai_completions_immediate_abort() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = openai_completions_llm();
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"OpenAI Responses Provider Abort"` / `"should abort mid-stream"`:
/// `openai/gpt-5-mini`.
#[tokio::test]
async fn openai_responses_abort_mid_stream() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("openai", "gpt-5-mini");
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"OpenAI Responses Provider Abort"` /
/// `"should handle immediate abort"`: `openai/gpt-5-mini`.
#[tokio::test]
async fn openai_responses_immediate_abort() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("openai", "gpt-5-mini");
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Azure OpenAI Responses Provider Abort"` /
/// `"should abort mid-stream"`: `azure-openai-responses/gpt-4o-mini` with the
/// mapped deployment override.
#[tokio::test]
async fn azure_openai_responses_abort_mid_stream() {
    let Some((llm, options)) = azure_model_and_options() else {
        return;
    };
    live::test_abort_signal(&llm, &options).await;
}

/// Upstream `"Azure OpenAI Responses Provider Abort"` /
/// `"should handle immediate abort"`: `azure-openai-responses/gpt-4o-mini`
/// with the mapped deployment override.
#[tokio::test]
async fn azure_openai_responses_immediate_abort() {
    let Some((llm, options)) = azure_model_and_options() else {
        return;
    };
    live::test_immediate_abort(&llm, &options).await;
}

/// Upstream `"Anthropic Provider Abort"` / `"should abort mid-stream"`:
/// `anthropic/claude-sonnet-4-6` with extended thinking at a 2048-token
/// budget.
#[tokio::test]
async fn anthropic_abort_mid_stream() {
    let Some(api_key) = live::env_key("ANTHROPIC_OAUTH_TOKEN") else {
        return;
    };
    let llm = live::model("anthropic", "claude-sonnet-4-6");
    live::test_abort_signal(&llm, &anthropic_options(api_key)).await;
}

/// Upstream `"Anthropic Provider Abort"` / `"should handle immediate abort"`:
/// `anthropic/claude-sonnet-4-6` with extended thinking at a 2048-token
/// budget.
#[tokio::test]
async fn anthropic_immediate_abort() {
    let Some(api_key) = live::env_key("ANTHROPIC_OAUTH_TOKEN") else {
        return;
    };
    let llm = live::model("anthropic", "claude-sonnet-4-6");
    live::test_immediate_abort(&llm, &anthropic_options(api_key)).await;
}

/// Upstream `"Mistral Provider Abort"` / `"should abort mid-stream"`:
/// `mistral/devstral-medium-latest`.
#[tokio::test]
async fn mistral_abort_mid_stream() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let llm = live::model("mistral", "devstral-medium-latest");
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Mistral Provider Abort"` / `"should handle immediate abort"`:
/// `mistral/devstral-medium-latest`.
#[tokio::test]
async fn mistral_immediate_abort() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let llm = live::model("mistral", "devstral-medium-latest");
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Together AI Provider Abort"` / `"should abort mid-stream"`:
/// `together/moonshotai/Kimi-K2.6` at high reasoning effort.
#[tokio::test]
async fn together_abort_mid_stream() {
    let Some(api_key) = live::env_key("TOGETHER_API_KEY") else {
        return;
    };
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::test_abort_signal(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream `"Together AI Provider Abort"` / `"should handle immediate
/// abort"`: `together/moonshotai/Kimi-K2.6` at high reasoning effort.
#[tokio::test]
async fn together_immediate_abort() {
    let Some(api_key) = live::env_key("TOGETHER_API_KEY") else {
        return;
    };
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::test_immediate_abort(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream `"Baseten Provider Abort"` / `"should abort mid-stream"`:
/// `baseten/zai-org/GLM-5.2` at high reasoning effort.
#[tokio::test]
async fn baseten_abort_mid_stream() {
    let Some(api_key) = live::env_key("BASETEN_API_KEY") else {
        return;
    };
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    live::test_abort_signal(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream `"Baseten Provider Abort"` / `"should handle immediate abort"`:
/// `baseten/zai-org/GLM-5.2` at high reasoning effort.
#[tokio::test]
async fn baseten_immediate_abort() {
    let Some(api_key) = live::env_key("BASETEN_API_KEY") else {
        return;
    };
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    live::test_immediate_abort(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream `"MiniMax Provider Abort"` / `"should abort mid-stream"`:
/// `minimax/MiniMax-M2.7`.
#[tokio::test]
async fn minimax_abort_mid_stream() {
    let Some(api_key) = live::env_key("MINIMAX_API_KEY") else {
        return;
    };
    let llm = live::model("minimax", "MiniMax-M2.7");
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"MiniMax Provider Abort"` / `"should handle immediate abort"`:
/// `minimax/MiniMax-M2.7`.
#[tokio::test]
async fn minimax_immediate_abort() {
    let Some(api_key) = live::env_key("MINIMAX_API_KEY") else {
        return;
    };
    let llm = live::model("minimax", "MiniMax-M2.7");
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Xiaomi MiMo (API billing) Provider Abort"` /
/// `"should abort mid-stream"`: `xiaomi/mimo-v2.5-pro`.
#[tokio::test]
async fn xiaomi_abort_mid_stream() {
    let Some(api_key) = live::env_key("XIAOMI_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Xiaomi MiMo (API billing) Provider Abort"` /
/// `"should handle immediate abort"`: `xiaomi/mimo-v2.5-pro`.
#[tokio::test]
async fn xiaomi_immediate_abort() {
    let Some(api_key) = live::env_key("XIAOMI_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Xiaomi MiMo Token Plan (CN) Provider Abort"` /
/// `"should abort mid-stream"`: `xiaomi-token-plan-cn/mimo-v2.5-pro`.
#[tokio::test]
async fn xiaomi_token_plan_cn_abort_mid_stream() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Xiaomi MiMo Token Plan (CN) Provider Abort"` /
/// `"should handle immediate abort"`: `xiaomi-token-plan-cn/mimo-v2.5-pro`.
#[tokio::test]
async fn xiaomi_token_plan_cn_immediate_abort() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Xiaomi MiMo Token Plan (AMS) Provider Abort"` /
/// `"should abort mid-stream"`: `xiaomi-token-plan-ams/mimo-v2.5-pro`.
#[tokio::test]
async fn xiaomi_token_plan_ams_abort_mid_stream() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Xiaomi MiMo Token Plan (AMS) Provider Abort"` /
/// `"should handle immediate abort"`: `xiaomi-token-plan-ams/mimo-v2.5-pro`.
#[tokio::test]
async fn xiaomi_token_plan_ams_immediate_abort() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Xiaomi MiMo Token Plan (SGP) Provider Abort"` /
/// `"should abort mid-stream"`: `xiaomi-token-plan-sgp/mimo-v2.5-pro`.
#[tokio::test]
async fn xiaomi_token_plan_sgp_abort_mid_stream() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Xiaomi MiMo Token Plan (SGP) Provider Abort"` /
/// `"should handle immediate abort"`: `xiaomi-token-plan-sgp/mimo-v2.5-pro`.
#[tokio::test]
async fn xiaomi_token_plan_sgp_immediate_abort() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Qwen Token Plan Provider Abort"` / `"should abort mid-stream"`:
/// `qwen-token-plan/qwen3.7-max`.
#[tokio::test]
async fn qwen_token_plan_abort_mid_stream() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Qwen Token Plan Provider Abort"` / `"should handle immediate
/// abort"`: `qwen-token-plan/qwen3.7-max`.
#[tokio::test]
async fn qwen_token_plan_immediate_abort() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Qwen Token Plan Individual Provider Abort"` /
/// `"should abort mid-stream"`: `qwen-token-plan-individual/qwen3.8-max`,
/// gated on the same `QWEN_TOKEN_PLAN_API_KEY` as the shared-plan block.
#[tokio::test]
async fn qwen_token_plan_individual_abort_mid_stream() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Qwen Token Plan Individual Provider Abort"` /
/// `"should handle immediate abort"`:
/// `qwen-token-plan-individual/qwen3.8-max`, gated on the same
/// `QWEN_TOKEN_PLAN_API_KEY` as the shared-plan block.
#[tokio::test]
async fn qwen_token_plan_individual_immediate_abort() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Qwen Token Plan (CN) Provider Abort"` /
/// `"should abort mid-stream"`: `qwen-token-plan-cn/qwen3.7-max`.
#[tokio::test]
async fn qwen_token_plan_cn_abort_mid_stream() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Qwen Token Plan (CN) Provider Abort"` /
/// `"should handle immediate abort"`: `qwen-token-plan-cn/qwen3.7-max`.
#[tokio::test]
async fn qwen_token_plan_cn_immediate_abort() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Kimi For Coding Provider Abort"` / `"should abort mid-stream"`:
/// `kimi-coding/kimi-for-coding`.
#[tokio::test]
async fn kimi_for_coding_abort_mid_stream() {
    let Some(api_key) = live::env_key("KIMI_API_KEY") else {
        return;
    };
    let llm = live::model("kimi-coding", "kimi-for-coding");
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Kimi For Coding Provider Abort"` / `"should handle immediate
/// abort"`: `kimi-coding/kimi-for-coding`.
#[tokio::test]
async fn kimi_for_coding_immediate_abort() {
    let Some(api_key) = live::env_key("KIMI_API_KEY") else {
        return;
    };
    let llm = live::model("kimi-coding", "kimi-for-coding");
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Vercel AI Gateway Provider Abort"` / `"should abort
/// mid-stream"`: `vercel-ai-gateway/google/gemini-2.5-flash`.
#[tokio::test]
async fn vercel_ai_gateway_abort_mid_stream() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    live::test_abort_signal(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"Vercel AI Gateway Provider Abort"` / `"should handle immediate
/// abort"`: `vercel-ai-gateway/google/gemini-2.5-flash`.
#[tokio::test]
async fn vercel_ai_gateway_immediate_abort() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    live::test_immediate_abort(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `"OpenAI Codex Provider Abort"` / `"should abort mid-stream"`:
/// `openai-codex/gpt-5.5` riding the credential-store-resolved token,
/// upstream's per-it `it.skipIf(!openaiCodexToken)`.
#[tokio::test]
async fn openai_codex_abort_mid_stream() {
    let Some(token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::test_abort_signal(&llm, &LiveOptions::key(token)).await;
}

/// Upstream `"OpenAI Codex Provider Abort"` / `"should handle immediate
/// abort"`: `openai-codex/gpt-5.5` riding the credential-store-resolved
/// token, upstream's per-it `it.skipIf(!openaiCodexToken)`.
#[tokio::test]
async fn openai_codex_immediate_abort() {
    let Some(token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::test_immediate_abort(&llm, &LiveOptions::key(token)).await;
}

/// Upstream `"Amazon Bedrock Provider Abort"` / `"should abort mid-stream"`:
/// the Claude Sonnet 4.5 global inference profile at medium reasoning,
/// upstream's `{ reasoning: "medium" }`.
#[tokio::test]
async fn bedrock_abort_mid_stream() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    let options = LiveOptions {
        reasoning: Some(ThinkingLevel::Medium),
        ..LiveOptions::default()
    };
    live::test_abort_signal(&llm, &options).await;
}

/// Upstream `"Amazon Bedrock Provider Abort"` / `"should handle immediate
/// abort"`: the AWS credentials ride the provider's own resolution, upstream's
/// options-less probe.
#[tokio::test]
async fn bedrock_immediate_abort() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    live::test_immediate_abort(&llm, &LiveOptions::default()).await;
}

/// Upstream `"Amazon Bedrock Provider Abort"` / `"should handle abort then
/// new message"`: the aborted empty assistant rides context and the follow-up
/// still answers.
#[tokio::test]
async fn bedrock_abort_then_new_message() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    live::test_abort_then_new_message(&llm, &LiveOptions::default()).await;
}

//! Env-gated live token-accounting probes, ported from the upstream
//! `packages/ai/test/tokens.test.ts` `describe "Token Statistics on Abort"`
//! suites at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Gating contract: like upstream's `describe.skipIf`, a probe without its
//! provider credential returns early — env-keyed providers read their
//! variable through `live::env_key`, Azure, Bedrock, and the Cloudflare
//! backends through their credential guards, and the OAuth providers resolve
//! through `live::resolve_api_key` per test (upstream resolves the tokens
//! once at module load). Upstream's `retry: 3` and 30-second timeout have no
//! cargo equivalent, so a flaky provider fails the run instead of retrying.
//! The four Xiaomi probes are `it.skip` upstream and stay gated off here; see
//! their docs.

#![expect(
    clippy::expect_used,
    reason = "the tests pin live outcomes; an unexpected shape panics the test by design"
)]

mod common;
use common::live;

use pi_ai::api::google_shared::GoogleThinkingControl;
use pi_ai::types::{Api, ThinkingLevel};

/// The Google token-stats probe, upstream `describe "Google Provider"` / `it
/// "should include token stats when aborted mid-stream"` with thinking
/// enabled, upstream's `{ thinking: { enabled: true } }`.
#[tokio::test]
async fn google_token_stats_on_abort() {
    let Some(_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let llm = live::model("google", "gemini-2.5-flash");
    let options = live::LiveOptions {
        google_thinking: Some(GoogleThinkingControl {
            enabled: true,
            budget_tokens: None,
            level: None,
        }),
        ..live::LiveOptions::default()
    };
    live::test_tokens_on_abort(&llm, &options).await;
}

/// The OpenAI Completions token-stats probe, upstream `describe "OpenAI
/// Completions Provider"` / `it "should include token stats when aborted
/// mid-stream"`: the gpt-4o-mini catalog entry retargeted at the
/// openai-completions wire with its compat map stripped, upstream's
/// `{ ...getModel(...), api: "openai-completions" }` spread.
#[tokio::test]
async fn openai_completions_token_stats_on_abort() {
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let mut llm = live::model("openai", "gpt-4o-mini");
    llm.api = Api::from("openai-completions");
    llm.compat = None;
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The OpenAI Responses token-stats probe, upstream `describe "OpenAI
/// Responses Provider"` / `it "should include token stats when aborted
/// mid-stream"` at reasoning effort low, upstream's
/// `{ reasoningEffort: "low" }`.
#[tokio::test]
async fn openai_responses_token_stats_on_abort() {
    let Some(_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("openai", "gpt-5.4-mini");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::Low),
        ..live::LiveOptions::default()
    };
    live::test_tokens_on_abort(&llm, &options).await;
}

/// The Azure OpenAI Responses token-stats probe, upstream `describe "Azure
/// OpenAI Responses Provider"` / `it "should include token stats when aborted
/// mid-stream"`: the deployment name rides when the environment maps one for
/// the model, upstream's `azureDeploymentName`.
#[tokio::test]
async fn azure_openai_responses_token_stats_on_abort() {
    if !live::has_azure_openai_credentials() {
        return;
    }
    let llm = live::model("azure-openai-responses", "gpt-4o-mini");
    let mut options = live::LiveOptions::default();
    if let Some(deployment) = live::azure_deployment_name(&llm.id) {
        options.azure_deployment_name = Some(deployment);
    }
    live::test_tokens_on_abort(&llm, &options).await;
}

/// The Anthropic token-stats probe, upstream `describe "Anthropic Provider"`
/// / `it "should include token stats when aborted mid-stream"`: Anthropic
/// sends usage early, so the aborted message carries the full split.
#[tokio::test]
async fn anthropic_token_stats_on_abort() {
    if live::env_key("ANTHROPIC_API_KEY").is_none() {
        return;
    }
    let llm = live::model("anthropic", "claude-sonnet-4-6");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The xAI token-stats probe, upstream `describe "xAI Provider"` / `it
/// "should include token stats when aborted mid-stream"`.
#[tokio::test]
async fn xai_token_stats_on_abort() {
    if live::env_key("XAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xai", "grok-4.3");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Groq token-stats probe, upstream `describe "Groq Provider"` / `it
/// "should include token stats when aborted mid-stream"`.
#[tokio::test]
async fn groq_token_stats_on_abort() {
    if live::env_key("GROQ_API_KEY").is_none() {
        return;
    }
    let llm = live::model("groq", "openai/gpt-oss-20b");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Cerebras token-stats probe, upstream `describe "Cerebras Provider"` /
/// `it "should include token stats when aborted mid-stream"`: the probe runs
/// the first catalog model among the preferred ids, else the first model, and
/// an empty catalog is a test bug that panics, upstream's throw.
#[tokio::test]
async fn cerebras_token_stats_on_abort() {
    if live::env_key("CEREBRAS_API_KEY").is_none() {
        return;
    }
    let preferred = ["gpt-oss-120b", "zai-glm-4.7", "llama3.1-8b"];
    let catalog = live::models("cerebras");
    let llm = catalog
        .iter()
        .find(|model| preferred.contains(&model.id.as_str()))
        .or_else(|| catalog.first())
        .expect("No Cerebras models available");
    live::test_tokens_on_abort(llm, &live::LiveOptions::default()).await;
}

/// The Cloudflare Workers AI token-stats probe, upstream `describe
/// "Cloudflare Workers AI Provider"` / `it "should include token stats when
/// aborted mid-stream"`.
#[tokio::test]
async fn cloudflare_workers_ai_token_stats_on_abort() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Cloudflare AI Gateway token-stats probe, upstream `describe
/// "Cloudflare AI Gateway Provider"` / `it "should include token stats when
/// aborted mid-stream"`.
#[tokio::test]
async fn cloudflare_ai_gateway_token_stats_on_abort() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Hugging Face token-stats probe, upstream `describe "Hugging Face
/// Provider"` / `it "should include token stats when aborted mid-stream"`.
#[tokio::test]
async fn huggingface_token_stats_on_abort() {
    if live::env_key("HF_TOKEN").is_none() {
        return;
    }
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Together AI token-stats probe, upstream `describe "Together AI
/// Provider"` / `it "should include token stats when aborted mid-stream"`.
#[tokio::test]
async fn together_token_stats_on_abort() {
    if live::env_key("TOGETHER_API_KEY").is_none() {
        return;
    }
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Baseten token-stats probe, upstream `describe "Baseten Provider"` /
/// `it "should include token stats when aborted mid-stream"` at reasoning
/// effort high, upstream's `{ reasoningEffort: "high" }`.
#[tokio::test]
async fn baseten_token_stats_on_abort() {
    let Some(_key) = live::env_key("BASETEN_API_KEY") else {
        return;
    };
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    live::test_tokens_on_abort(&llm, &options).await;
}

/// The zAI token-stats probe, upstream `describe "zAI Provider"` / `it
/// "should include token stats when aborted mid-stream"`: z.ai only sends
/// usage in the final chunk, so the aborted message carries none.
#[tokio::test]
async fn zai_token_stats_on_abort() {
    if live::env_key("ZAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("zai", "glm-5.2");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Mistral token-stats probe, upstream `describe "Mistral Provider"` /
/// `it "should include token stats when aborted mid-stream"`.
#[tokio::test]
async fn mistral_token_stats_on_abort() {
    if live::env_key("MISTRAL_API_KEY").is_none() {
        return;
    }
    let llm = live::model("mistral", "devstral-medium-latest");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The MiniMax token-stats probe, upstream `describe "MiniMax Provider"` /
/// `it "should include token stats when aborted mid-stream"`: MiniMax M2.7
/// does not report token usage for aborted requests.
#[tokio::test]
async fn minimax_token_stats_on_abort() {
    if live::env_key("MINIMAX_API_KEY").is_none() {
        return;
    }
    let llm = live::model("minimax", "MiniMax-M2.7");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Kimi For Coding token-stats probe, upstream `describe "Kimi For Coding
/// Provider"` / `it "should include token stats when aborted mid-stream"`:
/// Kimi reports input tokens early but output tokens only in the final
/// chunk.
#[tokio::test]
async fn kimi_coding_token_stats_on_abort() {
    if live::env_key("KIMI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("kimi-coding", "kimi-for-coding");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Vercel AI Gateway token-stats probe, upstream `describe "Vercel AI
/// Gateway Provider"` / `it "should include token stats when aborted
/// mid-stream"`: the gateway only sends usage in the final chunk, so the
/// aborted message carries none.
#[tokio::test]
async fn vercel_ai_gateway_token_stats_on_abort() {
    if live::env_key("AI_GATEWAY_API_KEY").is_none() {
        return;
    }
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Xiaomi MiMo API-billing token-stats probe, upstream `describe "Xiaomi
/// MiMo (API billing) Provider"` — `it.skip` upstream: Xiaomi's
/// Anthropic-compatible stream does not populate usage in the
/// `message_start` event the way Anthropic does — usage only arrives at
/// `message_stop` upstream-side, so aborting mid-stream loses the input/output
/// token counts. The port keeps the probe gated off until the upstream stream
/// reports usage in `message_start`.
#[tokio::test]
async fn xiaomi_api_billing_aborted_token_stats_gated_off() {
    let Some(_key) = live::env_key("XIAOMI_API_KEY") else {
        return;
    };
}

/// The Xiaomi MiMo Token Plan (CN) token-stats probe, upstream `describe
/// "Xiaomi MiMo Token Plan (CN) Provider"` — `it.skip` upstream with the same
/// upstream limitation as the API-billing block: the streaming usage
/// applies to the Token Plan endpoints, so the port keeps the probe gated off
/// until the upstream stream reports usage in `message_start`.
#[tokio::test]
async fn xiaomi_token_plan_cn_aborted_token_stats_gated_off() {
    let Some(_key) = live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
}

/// The Xiaomi MiMo Token Plan (AMS) token-stats probe, upstream `describe
/// "Xiaomi MiMo Token Plan (AMS) Provider"` — `it.skip` upstream with the
/// same upstream limitation as the API-billing block: the streaming usage
/// limitation applies to the Token Plan endpoints, so the port keeps the
/// probe gated off until the upstream stream reports usage in `message_start`.
#[tokio::test]
async fn xiaomi_token_plan_ams_aborted_token_stats_gated_off() {
    let Some(_key) = live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY") else {
        return;
    };
}

/// The Xiaomi MiMo Token Plan (SGP) token-stats probe, upstream `describe
/// "Xiaomi MiMo Token Plan (SGP) Provider"` — `it.skip` upstream with the
/// same upstream limitation as the API-billing block: the streaming usage
/// limitation applies to the Token Plan endpoints, so the port keeps the
/// probe gated off until the upstream stream reports usage in `message_start`.
#[tokio::test]
async fn xiaomi_token_plan_sgp_aborted_token_stats_gated_off() {
    let Some(_key) = live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY") else {
        return;
    };
}

/// The Qwen Token Plan token-stats probe, upstream `describe "Qwen Token Plan
/// Provider"` / `it "should include token stats when aborted mid-stream"`.
#[tokio::test]
async fn qwen_token_plan_token_stats_on_abort() {
    if live::env_key("QWEN_TOKEN_PLAN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Qwen Token Plan Individual token-stats probe, upstream `describe
/// "Qwen Token Plan Individual Provider"` / `it "should include token stats
/// when aborted mid-stream"`.
#[tokio::test]
async fn qwen_token_plan_individual_token_stats_on_abort() {
    if live::env_key("QWEN_TOKEN_PLAN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Qwen Token Plan (CN) token-stats probe, upstream `describe "Qwen Token
/// Plan (CN) Provider"` / `it "should include token stats when aborted
/// mid-stream"`.
#[tokio::test]
async fn qwen_token_plan_cn_token_stats_on_abort() {
    if live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

/// The Anthropic OAuth token-stats probe, upstream `describe "Anthropic OAuth
/// Provider"` / `it "should include token stats when aborted mid-stream"`:
/// the pi credential store's OAuth token rides the request as the explicit
/// api key, upstream's `{ apiKey: anthropicOAuthToken }`.
#[tokio::test]
async fn anthropic_oauth_token_stats_on_abort() {
    let Some(key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-sonnet-4-6");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::key(key)).await;
}

/// The GitHub Copilot Haiku token-stats probe, upstream `describe "GitHub
/// Copilot Provider"` / `it "claude-haiku-4.5 - should include token stats
/// when aborted mid-stream"` over the Copilot OAuth credential.
#[tokio::test]
async fn github_copilot_haiku_token_stats_on_abort() {
    let Some(key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-haiku-4.5");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::key(key)).await;
}

/// The GitHub Copilot Sonnet token-stats probe, upstream `describe "GitHub
/// Copilot Provider"` / `it "claude-sonnet-4 - should include token stats
/// when aborted mid-stream"` over the Copilot OAuth credential.
#[tokio::test]
async fn github_copilot_sonnet_token_stats_on_abort() {
    let Some(key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::key(key)).await;
}

/// The OpenAI Codex token-stats probe, upstream `describe "OpenAI Codex
/// Provider"` / `it "gpt-5.5 - should include token stats when aborted
/// mid-stream"` over the Codex OAuth credential.
#[tokio::test]
async fn openai_codex_token_stats_on_abort() {
    let Some(key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::test_tokens_on_abort(&llm, &live::LiveOptions::key(key)).await;
}

/// The Amazon Bedrock token-stats probe, upstream `describe "Amazon Bedrock
/// Provider"` / `it "should include token stats when aborted mid-stream"`:
/// Bedrock only sends usage in the final chunk, so the aborted message
/// carries none.
#[tokio::test]
async fn amazon_bedrock_token_stats_on_abort() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    live::test_tokens_on_abort(&llm, &live::LiveOptions::default()).await;
}

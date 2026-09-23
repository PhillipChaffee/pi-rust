//! The `totalTokens` accounting suites, ported from the upstream
//! `packages/ai/test/total-tokens.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Gating contract: like upstream's `describe.skipIf` and `it.skipIf`, a
//! block without its provider credential returns early — single-var gates
//! read the environment through `live::env_key`, the Azure, Cloudflare, and
//! Bedrock gates ride the shared credential guards, and the OAuth gates
//! resolve through the pi credential store via `live::resolve_api_key` per
//! test. Every probe runs the two cache-primed completions and asserts
//! `total_tokens` equals the sum of its components; the Anthropic blocks
//! additionally require cache activity.

#![expect(
    clippy::print_stdout,
    reason = "the usage lines surface the wire's numbers, upstream's logUsage console.log"
)]
// binary and carries its own expects only for the lints its own code knows
// it trips; these cover the rest so the suite's gate runs clean without
// editing the shared file. Stale entries fail the build on their own.

use pi_ai::types::{Api, ThinkingLevel, Usage};

mod common;
use common::live::{self, LiveOptions};

/// The usage line block, upstream's `logUsage`: the wire's numbers plus the
/// computed component sum.
fn log_usage(label: &str, usage: &Usage) {
    let computed = usage.input + usage.output + usage.cache_read + usage.cache_write;
    println!("  {label}:");
    println!(
        "    input: {}, output: {}, cache_read: {}, cache_write: {}",
        usage.input, usage.output, usage.cache_read, usage.cache_write
    );
    println!(
        "    total_tokens: {}, computed: {}",
        usage.total_tokens, computed
    );
}

/// The shared block body: the two cache-primed completions, both usages
/// logged, both asserted against their component sum.
async fn run_total_tokens_case(
    label: &str,
    llm: &pi_ai::types::Model,
    options: &LiveOptions,
) -> (Usage, Usage) {
    println!("\n{label} / {}:", llm.id);
    let (first, second) = live::test_total_tokens_with_cache(llm, options).await;
    log_usage("First request", &first);
    log_usage("Second request", &second);
    live::assert_total_tokens_equals_components(&first);
    live::assert_total_tokens_equals_components(&second);
    (first, second)
}

/// The Anthropic-family cache-activity assert, upstream's per-Anthropic-it
/// extra.
fn assert_cache_activity(first: &Usage, second: &Usage) {
    let has_cache = second.cache_read > 0 || second.cache_write > 0 || first.cache_write > 0;
    assert!(has_cache, "Anthropic should have cache activity");
}

/// Upstream describe `Anthropic (API Key)`.
/// Upstream it `claude-sonnet-4-5 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn anthropic_api_key_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("ANTHROPIC_API_KEY") else {
        return;
    };
    let llm = live::model("anthropic", "claude-sonnet-4-5");

    let (first, second) =
        run_total_tokens_case("Anthropic", &llm, &LiveOptions::key(api_key)).await;
    assert_cache_activity(&first, &second);
}

/// Upstream describe `Anthropic (OAuth)`.
/// Upstream it `claude-sonnet-4 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn anthropic_oauth_total_tokens_equals_components() {
    let Some(token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-sonnet-4-6");

    let (first, second) =
        run_total_tokens_case("Anthropic OAuth", &llm, &LiveOptions::key(token)).await;
    assert_cache_activity(&first, &second);
}

/// Upstream describe `OpenAI Completions`.
/// Upstream it `gpt-4o-mini - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn openai_completions_total_tokens_equals_components() {
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let mut llm = live::model("openai", "gpt-4o-mini");
    llm.api = Api::from("openai-completions");

    run_total_tokens_case("OpenAI Completions", &llm, &LiveOptions::default()).await;
}

/// Upstream describe `OpenAI Responses`.
/// Upstream it `claude-haiku-4.5 - should return totalTokens equal to sum of components`.
/// (Upstream's it resolves openai/gpt-4o; the port keeps the code, not the title.)
#[tokio::test]
async fn openai_responses_total_tokens_equals_components() {
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("openai", "gpt-4o");

    run_total_tokens_case("OpenAI Responses", &llm, &LiveOptions::default()).await;
}

/// Upstream describe `Azure OpenAI Responses`.
/// Upstream it `gpt-4o-mini - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn azure_openai_responses_total_tokens_equals_components() {
    if !live::has_azure_openai_credentials() {
        return;
    }
    let llm = live::model("azure-openai-responses", "gpt-4o-mini");
    let options = live::azure_deployment_name(&llm.id).map_or_else(LiveOptions::default, |name| {
        LiveOptions {
            azure_deployment_name: Some(name),
            ..LiveOptions::default()
        }
    });

    run_total_tokens_case("Azure OpenAI Responses", &llm, &options).await;
}

/// Upstream describe `Google`.
/// Upstream it `gemini-2.5-flash - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn google_total_tokens_equals_components() {
    if live::env_key("GEMINI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("google", "gemini-2.5-flash");

    run_total_tokens_case("Google", &llm, &LiveOptions::default()).await;
}

/// Upstream describe `xAI`.
/// Upstream it `grok-4.3 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn xai_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("XAI_API_KEY") else {
        return;
    };
    let llm = live::model("xai", "grok-4.3");

    run_total_tokens_case("xAI", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `Groq`.
/// Upstream it `openai/gpt-oss-120b - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn groq_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("GROQ_API_KEY") else {
        return;
    };
    let llm = live::model("groq", "openai/gpt-oss-120b");

    run_total_tokens_case("Groq", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `Cerebras`.
/// Upstream it `gpt-oss-120b - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn cerebras_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("CEREBRAS_API_KEY") else {
        return;
    };
    let llm = live::model("cerebras", "gpt-oss-120b");

    run_total_tokens_case("Cerebras", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `Cloudflare Workers AI`.
/// Upstream it `@cf/moonshotai/kimi-k2.6 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn cloudflare_workers_ai_total_tokens_equals_components() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let Some(api_key) = live::env_key("CLOUDFLARE_API_KEY") else {
        return;
    };
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");

    run_total_tokens_case("Cloudflare Workers AI", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `Cloudflare AI Gateway`.
/// Upstream it `workers-ai/@cf/moonshotai/kimi-k2.6 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn cloudflare_ai_gateway_total_tokens_equals_components() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let Some(api_key) = live::env_key("CLOUDFLARE_API_KEY") else {
        return;
    };
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );

    run_total_tokens_case("Cloudflare AI Gateway", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `Hugging Face`.
/// Upstream it `Kimi-K2.5 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn huggingface_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("HF_TOKEN") else {
        return;
    };
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");

    run_total_tokens_case("Hugging Face", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `Together AI`.
/// Upstream it `Kimi-K2.6 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn together_ai_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("TOGETHER_API_KEY") else {
        return;
    };
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    let options = LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        api_key: Some(api_key),
        ..LiveOptions::default()
    };

    run_total_tokens_case("Together AI", &llm, &options).await;
}

/// Upstream describe `Baseten`.
/// Upstream it `GLM 5.2 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn baseten_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("BASETEN_API_KEY") else {
        return;
    };
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    let options = LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        api_key: Some(api_key),
        ..LiveOptions::default()
    };

    run_total_tokens_case("Baseten", &llm, &options).await;
}

/// Upstream describe `z.ai`.
/// Upstream it `glm-5.2 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn zai_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("ZAI_API_KEY") else {
        return;
    };
    let llm = live::model("zai", "glm-5.2");

    run_total_tokens_case("z.ai", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `Mistral`.
/// Upstream it `devstral-medium-latest - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn mistral_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let llm = live::model("mistral", "devstral-medium-latest");

    run_total_tokens_case("Mistral", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `MiniMax`.
/// Upstream it `MiniMax-M2.7 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn minimax_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("MINIMAX_API_KEY") else {
        return;
    };
    let llm = live::model("minimax", "MiniMax-M2.7");

    run_total_tokens_case("MiniMax", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `Xiaomi MiMo (API billing)`.
/// Upstream it `mimo-v2.5-pro - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn xiaomi_mimo_api_billing_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("XIAOMI_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi", "mimo-v2.5-pro");

    run_total_tokens_case("Xiaomi MiMo", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `Xiaomi MiMo Token Plan (CN)`.
/// Upstream it `mimo-v2.5-pro - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn xiaomi_token_plan_cn_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");

    run_total_tokens_case(
        "Xiaomi MiMo Token Plan CN",
        &llm,
        &LiveOptions::key(api_key),
    )
    .await;
}

/// Upstream describe `Xiaomi MiMo Token Plan (AMS)`.
/// Upstream it `mimo-v2.5-pro - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn xiaomi_token_plan_ams_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");

    run_total_tokens_case(
        "Xiaomi MiMo Token Plan AMS",
        &llm,
        &LiveOptions::key(api_key),
    )
    .await;
}

/// Upstream describe `Xiaomi MiMo Token Plan (SGP)`.
/// Upstream it `mimo-v2.5-pro - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn xiaomi_token_plan_sgp_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");

    run_total_tokens_case(
        "Xiaomi MiMo Token Plan SGP",
        &llm,
        &LiveOptions::key(api_key),
    )
    .await;
}

/// Upstream describe `Qwen Token Plan`.
/// Upstream it `qwen3.7-max - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn qwen_token_plan_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan", "qwen3.7-max");

    run_total_tokens_case("Qwen Token Plan", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `Qwen Token Plan Individual`.
/// Upstream it `qwen3.8-max - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn qwen_token_plan_individual_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");

    run_total_tokens_case(
        "Qwen Token Plan Individual",
        &llm,
        &LiveOptions::key(api_key),
    )
    .await;
}

/// Upstream describe `Qwen Token Plan (CN)`.
/// Upstream it `qwen3.7-max - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn qwen_token_plan_cn_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");

    run_total_tokens_case("Qwen Token Plan CN", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `Kimi For Coding`.
/// Upstream it `kimi-for-coding - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn kimi_for_coding_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("KIMI_API_KEY") else {
        return;
    };
    let llm = live::model("kimi-coding", "kimi-for-coding");

    run_total_tokens_case("Kimi For Coding", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `Vercel AI Gateway`.
/// Upstream it `google/gemini-2.5-flash - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn vercel_ai_gateway_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");

    run_total_tokens_case("Vercel AI Gateway", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `OpenRouter`.
/// Upstream it `anthropic/claude-sonnet-4 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn openrouter_anthropic_claude_sonnet_4_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let llm = live::model("openrouter", "anthropic/claude-sonnet-4");

    run_total_tokens_case("OpenRouter", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `OpenRouter`.
/// Upstream it `deepseek/deepseek-chat - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn openrouter_deepseek_chat_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let llm = live::model("openrouter", "deepseek/deepseek-chat");

    run_total_tokens_case("OpenRouter", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `OpenRouter`.
/// Upstream it `mistralai/mistral-small-3.2-24b-instruct - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn openrouter_mistral_small_3_2_24b_instruct_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let llm = live::model("openrouter", "mistralai/mistral-small-3.2-24b-instruct");

    run_total_tokens_case("OpenRouter", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `OpenRouter`.
/// Upstream it `google/gemini-2.5-flash - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn openrouter_google_gemini_2_5_flash_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let llm = live::model("openrouter", "google/gemini-2.5-flash");

    run_total_tokens_case("OpenRouter", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `OpenRouter`.
/// Upstream it `deepseek/deepseek-chat - should return totalTokens equal to sum of components`
/// (repeated it, upstream order kept).
#[tokio::test]
async fn openrouter_deepseek_chat_repeat_total_tokens_equals_components() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let llm = live::model("openrouter", "deepseek/deepseek-chat");

    run_total_tokens_case("OpenRouter", &llm, &LiveOptions::key(api_key)).await;
}

/// Upstream describe `GitHub Copilot (OAuth)`.
/// Upstream it `claude-haiku-4.5 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn github_copilot_claude_haiku_4_5_total_tokens() {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-haiku-4.5");

    run_total_tokens_case("GitHub Copilot", &llm, &LiveOptions::key(token)).await;
}

/// Upstream describe `GitHub Copilot (OAuth)`.
/// Upstream it `claude-sonnet-4 - should return totalTokens equal to sum of components`.
/// (Upstream's it resolves claude-sonnet-4.6; the port keeps the code, not the title.)
#[tokio::test]
async fn github_copilot_claude_sonnet_4_6_total_tokens() {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");

    run_total_tokens_case("GitHub Copilot", &llm, &LiveOptions::key(token)).await;
}

/// Upstream describe `Amazon Bedrock`.
/// Upstream it `claude-sonnet-4-5 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn amazon_bedrock_total_tokens_equals_components() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );

    run_total_tokens_case("Amazon Bedrock", &llm, &LiveOptions::default()).await;
}

/// Upstream describe `OpenAI Codex (OAuth)`.
/// Upstream it `gpt-5.5 - should return totalTokens equal to sum of components`.
#[tokio::test]
async fn openai_codex_oauth_total_tokens_equals_components() {
    let Some(token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");

    run_total_tokens_case("OpenAI Codex", &llm, &LiveOptions::key(token)).await;
}

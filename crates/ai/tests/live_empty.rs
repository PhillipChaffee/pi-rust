//! The empty-message live suites, ported from `packages/ai/test/empty.test.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: each gated provider
//! block probes an empty content array, an empty string, whitespace-only
//! content, and an empty assistant message in context, and the provider must
//! either answer gracefully or report a defined error.
//!
//! Upstream's `describe.skipIf(...)` gates restate as early returns: a probe
//! without its provider credential — an env key, a provider-credential
//! helper, or an OAuth key resolved through the pi credential store —
//! returns before resolving its model. Upstream resolves the OAuth tokens
//! once at module load; the port resolves per test, the skipIf semantics
//! preserved. Upstream's vitest `retry: 3, timeout: 30000` options have no
//! `#[tokio::test]` equivalent, so each probe runs once and untimed.

use pi_ai::types::{Api, ThinkingLevel};

mod common;
use common::live;

/// The env gate upstream's truthy `process.env.X` check restates to.
fn has_env(name: &str) -> bool {
    live::env_key(name).is_some()
}

/// Upstream `describe.skipIf(!process.env.GEMINI_API_KEY)` "Google Provider
/// Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn google_empty_content_array() {
    if !has_env("GEMINI_API_KEY") {
        return;
    }
    let llm = live::model("google", "gemini-2.5-flash");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.GEMINI_API_KEY)` "Google Provider
/// Empty Messages" > "should handle empty string content".
#[tokio::test]
async fn google_empty_string_content() {
    if !has_env("GEMINI_API_KEY") {
        return;
    }
    let llm = live::model("google", "gemini-2.5-flash");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.GEMINI_API_KEY)` "Google Provider
/// Empty Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn google_whitespace_only() {
    if !has_env("GEMINI_API_KEY") {
        return;
    }
    let llm = live::model("google", "gemini-2.5-flash");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.GEMINI_API_KEY)` "Google Provider
/// Empty Messages" > "should handle empty assistant message in conversation".
#[tokio::test]
async fn google_empty_assistant_message() {
    if !has_env("GEMINI_API_KEY") {
        return;
    }
    let llm = live::model("google", "gemini-2.5-flash");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Completions
/// Provider Empty Messages" > "should handle empty content array"; the port
/// retargets the catalog entry at `openai-completions`, upstream's
/// `getModel` return.
#[tokio::test]
async fn openai_completions_empty_content_array() {
    if !has_env("OPENAI_API_KEY") {
        return;
    }
    let mut llm = live::model("openai", "gpt-4o-mini");
    llm.api = Api::from("openai-completions");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Completions
/// Provider Empty Messages" > "should handle empty string content"; the port
/// retargets the catalog entry at `openai-completions`, upstream's
/// `getModel` return.
#[tokio::test]
async fn openai_completions_empty_string_content() {
    if !has_env("OPENAI_API_KEY") {
        return;
    }
    let mut llm = live::model("openai", "gpt-4o-mini");
    llm.api = Api::from("openai-completions");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Completions
/// Provider Empty Messages" > "should handle whitespace-only content"; the
/// port retargets the catalog entry at `openai-completions`, upstream's
/// `getModel` return.
#[tokio::test]
async fn openai_completions_whitespace_only() {
    if !has_env("OPENAI_API_KEY") {
        return;
    }
    let mut llm = live::model("openai", "gpt-4o-mini");
    llm.api = Api::from("openai-completions");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Completions
/// Provider Empty Messages" > "should handle empty assistant message in
/// conversation"; the port retargets the catalog entry at
/// `openai-completions`, upstream's `getModel` return.
#[tokio::test]
async fn openai_completions_empty_assistant_message() {
    if !has_env("OPENAI_API_KEY") {
        return;
    }
    let mut llm = live::model("openai", "gpt-4o-mini");
    llm.api = Api::from("openai-completions");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Responses
/// Provider Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn openai_responses_empty_content_array() {
    if !has_env("OPENAI_API_KEY") {
        return;
    }
    let llm = live::model("openai", "gpt-5-mini");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Responses
/// Provider Empty Messages" > "should handle empty string content".
#[tokio::test]
async fn openai_responses_empty_string_content() {
    if !has_env("OPENAI_API_KEY") {
        return;
    }
    let llm = live::model("openai", "gpt-5-mini");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Responses
/// Provider Empty Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn openai_responses_whitespace_only() {
    if !has_env("OPENAI_API_KEY") {
        return;
    }
    let llm = live::model("openai", "gpt-5-mini");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Responses
/// Provider Empty Messages" > "should handle empty assistant message in
/// conversation".
#[tokio::test]
async fn openai_responses_empty_assistant_message() {
    if !has_env("OPENAI_API_KEY") {
        return;
    }
    let llm = live::model("openai", "gpt-5-mini");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasAzureOpenAICredentials())` "Azure OpenAI
/// Responses Provider Empty Messages" > "should handle empty content array",
/// the deployment override riding upstream's `azureOptions`.
#[tokio::test]
async fn azure_openai_responses_empty_content_array() {
    if !live::has_azure_openai_credentials() {
        return;
    }
    let llm = live::model("azure-openai-responses", "gpt-4o-mini");
    let options = live::LiveOptions {
        azure_deployment_name: live::azure_deployment_name(&llm.id),
        ..live::LiveOptions::default()
    };
    live::test_empty_message(&llm, &options).await;
}

/// Upstream `describe.skipIf(!hasAzureOpenAICredentials())` "Azure OpenAI
/// Responses Provider Empty Messages" > "should handle empty string content",
/// the deployment override riding upstream's `azureOptions`.
#[tokio::test]
async fn azure_openai_responses_empty_string_content() {
    if !live::has_azure_openai_credentials() {
        return;
    }
    let llm = live::model("azure-openai-responses", "gpt-4o-mini");
    let options = live::LiveOptions {
        azure_deployment_name: live::azure_deployment_name(&llm.id),
        ..live::LiveOptions::default()
    };
    live::test_empty_string_message(&llm, &options).await;
}

/// Upstream `describe.skipIf(!hasAzureOpenAICredentials())` "Azure OpenAI
/// Responses Provider Empty Messages" > "should handle whitespace-only
/// content", the deployment override riding upstream's `azureOptions`.
#[tokio::test]
async fn azure_openai_responses_whitespace_only() {
    if !live::has_azure_openai_credentials() {
        return;
    }
    let llm = live::model("azure-openai-responses", "gpt-4o-mini");
    let options = live::LiveOptions {
        azure_deployment_name: live::azure_deployment_name(&llm.id),
        ..live::LiveOptions::default()
    };
    live::test_whitespace_only_message(&llm, &options).await;
}

/// Upstream `describe.skipIf(!hasAzureOpenAICredentials())` "Azure OpenAI
/// Responses Provider Empty Messages" > "should handle empty assistant
/// message in conversation", the deployment override riding upstream's
/// `azureOptions`.
#[tokio::test]
async fn azure_openai_responses_empty_assistant_message() {
    if !live::has_azure_openai_credentials() {
        return;
    }
    let llm = live::model("azure-openai-responses", "gpt-4o-mini");
    let options = live::LiveOptions {
        azure_deployment_name: live::azure_deployment_name(&llm.id),
        ..live::LiveOptions::default()
    };
    live::test_empty_assistant_message(&llm, &options).await;
}

/// Upstream `describe.skipIf(!process.env.ANTHROPIC_API_KEY)` "Anthropic
/// Provider Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn anthropic_empty_content_array() {
    if !has_env("ANTHROPIC_API_KEY") {
        return;
    }
    let llm = live::model("anthropic", "claude-haiku-4-5");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.ANTHROPIC_API_KEY)` "Anthropic
/// Provider Empty Messages" > "should handle empty string content".
#[tokio::test]
async fn anthropic_empty_string_content() {
    if !has_env("ANTHROPIC_API_KEY") {
        return;
    }
    let llm = live::model("anthropic", "claude-haiku-4-5");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.ANTHROPIC_API_KEY)` "Anthropic
/// Provider Empty Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn anthropic_whitespace_only() {
    if !has_env("ANTHROPIC_API_KEY") {
        return;
    }
    let llm = live::model("anthropic", "claude-haiku-4-5");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.ANTHROPIC_API_KEY)` "Anthropic
/// Provider Empty Messages" > "should handle empty assistant message in
/// conversation".
#[tokio::test]
async fn anthropic_empty_assistant_message() {
    if !has_env("ANTHROPIC_API_KEY") {
        return;
    }
    let llm = live::model("anthropic", "claude-haiku-4-5");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XAI_API_KEY)` "xAI Provider Empty
/// Messages" > "should handle empty content array".
#[tokio::test]
async fn xai_empty_content_array() {
    if !has_env("XAI_API_KEY") {
        return;
    }
    let llm = live::model("xai", "grok-4.3");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XAI_API_KEY)` "xAI Provider Empty
/// Messages" > "should handle empty string content".
#[tokio::test]
async fn xai_empty_string_content() {
    if !has_env("XAI_API_KEY") {
        return;
    }
    let llm = live::model("xai", "grok-4.3");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XAI_API_KEY)` "xAI Provider Empty
/// Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn xai_whitespace_only() {
    if !has_env("XAI_API_KEY") {
        return;
    }
    let llm = live::model("xai", "grok-4.3");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XAI_API_KEY)` "xAI Provider Empty
/// Messages" > "should handle empty assistant message in conversation".
#[tokio::test]
async fn xai_empty_assistant_message() {
    if !has_env("XAI_API_KEY") {
        return;
    }
    let llm = live::model("xai", "grok-4.3");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.GROQ_API_KEY)` "Groq Provider Empty
/// Messages" > "should handle empty content array".
#[tokio::test]
async fn groq_empty_content_array() {
    if !has_env("GROQ_API_KEY") {
        return;
    }
    let llm = live::model("groq", "openai/gpt-oss-20b");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.GROQ_API_KEY)` "Groq Provider Empty
/// Messages" > "should handle empty string content".
#[tokio::test]
async fn groq_empty_string_content() {
    if !has_env("GROQ_API_KEY") {
        return;
    }
    let llm = live::model("groq", "openai/gpt-oss-20b");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.GROQ_API_KEY)` "Groq Provider Empty
/// Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn groq_whitespace_only() {
    if !has_env("GROQ_API_KEY") {
        return;
    }
    let llm = live::model("groq", "openai/gpt-oss-20b");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.GROQ_API_KEY)` "Groq Provider Empty
/// Messages" > "should handle empty assistant message in conversation".
#[tokio::test]
async fn groq_empty_assistant_message() {
    if !has_env("GROQ_API_KEY") {
        return;
    }
    let llm = live::model("groq", "openai/gpt-oss-20b");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.CEREBRAS_API_KEY)` "Cerebras
/// Provider Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn cerebras_empty_content_array() {
    if !has_env("CEREBRAS_API_KEY") {
        return;
    }
    let llm = live::model("cerebras", "gpt-oss-120b");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.CEREBRAS_API_KEY)` "Cerebras
/// Provider Empty Messages" > "should handle empty string content".
#[tokio::test]
async fn cerebras_empty_string_content() {
    if !has_env("CEREBRAS_API_KEY") {
        return;
    }
    let llm = live::model("cerebras", "gpt-oss-120b");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.CEREBRAS_API_KEY)` "Cerebras
/// Provider Empty Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn cerebras_whitespace_only() {
    if !has_env("CEREBRAS_API_KEY") {
        return;
    }
    let llm = live::model("cerebras", "gpt-oss-120b");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.CEREBRAS_API_KEY)` "Cerebras
/// Provider Empty Messages" > "should handle empty assistant message in
/// conversation".
#[tokio::test]
async fn cerebras_empty_assistant_message() {
    if !has_env("CEREBRAS_API_KEY") {
        return;
    }
    let llm = live::model("cerebras", "gpt-oss-120b");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasCloudflareWorkersAICredentials())`
/// "Cloudflare Workers AI Provider Empty Messages" > "should handle empty
/// content array".
#[tokio::test]
async fn cloudflare_workers_ai_empty_content_array() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasCloudflareWorkersAICredentials())`
/// "Cloudflare Workers AI Provider Empty Messages" > "should handle empty
/// string content".
#[tokio::test]
async fn cloudflare_workers_ai_empty_string_content() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasCloudflareWorkersAICredentials())`
/// "Cloudflare Workers AI Provider Empty Messages" > "should handle
/// whitespace-only content".
#[tokio::test]
async fn cloudflare_workers_ai_whitespace_only() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasCloudflareWorkersAICredentials())`
/// "Cloudflare Workers AI Provider Empty Messages" > "should handle empty
/// assistant message in conversation".
#[tokio::test]
async fn cloudflare_workers_ai_empty_assistant_message() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasCloudflareAiGatewayCredentials())`
/// "Cloudflare AI Gateway Provider Empty Messages" > "should handle empty
/// content array".
#[tokio::test]
async fn cloudflare_ai_gateway_empty_content_array() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasCloudflareAiGatewayCredentials())`
/// "Cloudflare AI Gateway Provider Empty Messages" > "should handle empty
/// string content".
#[tokio::test]
async fn cloudflare_ai_gateway_empty_string_content() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasCloudflareAiGatewayCredentials())`
/// "Cloudflare AI Gateway Provider Empty Messages" > "should handle
/// whitespace-only content".
#[tokio::test]
async fn cloudflare_ai_gateway_whitespace_only() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasCloudflareAiGatewayCredentials())`
/// "Cloudflare AI Gateway Provider Empty Messages" > "should handle empty
/// assistant message in conversation".
#[tokio::test]
async fn cloudflare_ai_gateway_empty_assistant_message() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.HF_TOKEN)` "Hugging Face Provider
/// Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn huggingface_empty_content_array() {
    if !has_env("HF_TOKEN") {
        return;
    }
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.HF_TOKEN)` "Hugging Face Provider
/// Empty Messages" > "should handle empty string content".
#[tokio::test]
async fn huggingface_empty_string_content() {
    if !has_env("HF_TOKEN") {
        return;
    }
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.HF_TOKEN)` "Hugging Face Provider
/// Empty Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn huggingface_whitespace_only() {
    if !has_env("HF_TOKEN") {
        return;
    }
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.HF_TOKEN)` "Hugging Face Provider
/// Empty Messages" > "should handle empty assistant message in conversation".
#[tokio::test]
async fn huggingface_empty_assistant_message() {
    if !has_env("HF_TOKEN") {
        return;
    }
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.TOGETHER_API_KEY)` "Together AI
/// Provider Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn together_empty_content_array() {
    if !has_env("TOGETHER_API_KEY") {
        return;
    }
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.TOGETHER_API_KEY)` "Together AI
/// Provider Empty Messages" > "should handle empty string content".
#[tokio::test]
async fn together_empty_string_content() {
    if !has_env("TOGETHER_API_KEY") {
        return;
    }
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.TOGETHER_API_KEY)` "Together AI
/// Provider Empty Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn together_whitespace_only() {
    if !has_env("TOGETHER_API_KEY") {
        return;
    }
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.TOGETHER_API_KEY)` "Together AI
/// Provider Empty Messages" > "should handle empty assistant message in
/// conversation".
#[tokio::test]
async fn together_empty_assistant_message() {
    if !has_env("TOGETHER_API_KEY") {
        return;
    }
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.BASETEN_API_KEY)` "Baseten Provider
/// Empty Messages" > "should handle empty content array", upstream's
/// `{ reasoningEffort: "high" }` riding the pi reasoning level.
#[tokio::test]
async fn baseten_empty_content_array() {
    if !has_env("BASETEN_API_KEY") {
        return;
    }
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    live::test_empty_message(&llm, &options).await;
}

/// Upstream `describe.skipIf(!process.env.BASETEN_API_KEY)` "Baseten Provider
/// Empty Messages" > "should handle empty string content", upstream's
/// `{ reasoningEffort: "high" }` riding the pi reasoning level.
#[tokio::test]
async fn baseten_empty_string_content() {
    if !has_env("BASETEN_API_KEY") {
        return;
    }
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    live::test_empty_string_message(&llm, &options).await;
}

/// Upstream `describe.skipIf(!process.env.BASETEN_API_KEY)` "Baseten Provider
/// Empty Messages" > "should handle whitespace-only content", upstream's
/// `{ reasoningEffort: "high" }` riding the pi reasoning level.
#[tokio::test]
async fn baseten_whitespace_only() {
    if !has_env("BASETEN_API_KEY") {
        return;
    }
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    live::test_whitespace_only_message(&llm, &options).await;
}

/// Upstream `describe.skipIf(!process.env.BASETEN_API_KEY)` "Baseten Provider
/// Empty Messages" > "should handle empty assistant message in conversation",
/// upstream's `{ reasoningEffort: "high" }` riding the pi reasoning level.
#[tokio::test]
async fn baseten_empty_assistant_message() {
    if !has_env("BASETEN_API_KEY") {
        return;
    }
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    live::test_empty_assistant_message(&llm, &options).await;
}

/// Upstream `describe.skipIf(!process.env.ZAI_API_KEY)` "`zAI` Provider Empty
/// Messages" > "should handle empty content array".
#[tokio::test]
async fn zai_empty_content_array() {
    if !has_env("ZAI_API_KEY") {
        return;
    }
    let llm = live::model("zai", "glm-5.2");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.ZAI_API_KEY)` "`zAI` Provider Empty
/// Messages" > "should handle empty string content".
#[tokio::test]
async fn zai_empty_string_content() {
    if !has_env("ZAI_API_KEY") {
        return;
    }
    let llm = live::model("zai", "glm-5.2");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.ZAI_API_KEY)` "`zAI` Provider Empty
/// Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn zai_whitespace_only() {
    if !has_env("ZAI_API_KEY") {
        return;
    }
    let llm = live::model("zai", "glm-5.2");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.ZAI_API_KEY)` "`zAI` Provider Empty
/// Messages" > "should handle empty assistant message in conversation".
#[tokio::test]
async fn zai_empty_assistant_message() {
    if !has_env("ZAI_API_KEY") {
        return;
    }
    let llm = live::model("zai", "glm-5.2");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.MISTRAL_API_KEY)` "Mistral Provider
/// Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn mistral_empty_content_array() {
    if !has_env("MISTRAL_API_KEY") {
        return;
    }
    let llm = live::model("mistral", "devstral-medium-latest");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.MISTRAL_API_KEY)` "Mistral Provider
/// Empty Messages" > "should handle empty string content".
#[tokio::test]
async fn mistral_empty_string_content() {
    if !has_env("MISTRAL_API_KEY") {
        return;
    }
    let llm = live::model("mistral", "devstral-medium-latest");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.MISTRAL_API_KEY)` "Mistral Provider
/// Empty Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn mistral_whitespace_only() {
    if !has_env("MISTRAL_API_KEY") {
        return;
    }
    let llm = live::model("mistral", "devstral-medium-latest");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.MISTRAL_API_KEY)` "Mistral Provider
/// Empty Messages" > "should handle empty assistant message in conversation".
#[tokio::test]
async fn mistral_empty_assistant_message() {
    if !has_env("MISTRAL_API_KEY") {
        return;
    }
    let llm = live::model("mistral", "devstral-medium-latest");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.MINIMAX_API_KEY)` "MiniMax Provider
/// Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn minimax_empty_content_array() {
    if !has_env("MINIMAX_API_KEY") {
        return;
    }
    let llm = live::model("minimax", "MiniMax-M2.7");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.MINIMAX_API_KEY)` "MiniMax Provider
/// Empty Messages" > "should handle empty string content".
#[tokio::test]
async fn minimax_empty_string_content() {
    if !has_env("MINIMAX_API_KEY") {
        return;
    }
    let llm = live::model("minimax", "MiniMax-M2.7");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.MINIMAX_API_KEY)` "MiniMax Provider
/// Empty Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn minimax_whitespace_only() {
    if !has_env("MINIMAX_API_KEY") {
        return;
    }
    let llm = live::model("minimax", "MiniMax-M2.7");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.MINIMAX_API_KEY)` "MiniMax Provider
/// Empty Messages" > "should handle empty assistant message in conversation".
#[tokio::test]
async fn minimax_empty_assistant_message() {
    if !has_env("MINIMAX_API_KEY") {
        return;
    }
    let llm = live::model("minimax", "MiniMax-M2.7");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_API_KEY)` "Xiaomi MiMo (API
/// billing) Provider Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn xiaomi_empty_content_array() {
    if !has_env("XIAOMI_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_API_KEY)` "Xiaomi MiMo (API
/// billing) Provider Empty Messages" > "should handle empty string content".
#[tokio::test]
async fn xiaomi_empty_string_content() {
    if !has_env("XIAOMI_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_API_KEY)` "Xiaomi MiMo (API
/// billing) Provider Empty Messages" > "should handle whitespace-only
/// content".
#[tokio::test]
async fn xiaomi_whitespace_only() {
    if !has_env("XIAOMI_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_API_KEY)` "Xiaomi MiMo (API
/// billing) Provider Empty Messages" > "should handle empty assistant message
/// in conversation".
#[tokio::test]
async fn xiaomi_empty_assistant_message() {
    if !has_env("XIAOMI_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_CN_API_KEY)`
/// "Xiaomi MiMo Token Plan (CN) Provider Empty Messages" > "should handle
/// empty content array".
#[tokio::test]
async fn xiaomi_token_plan_cn_empty_content_array() {
    if !has_env("XIAOMI_TOKEN_PLAN_CN_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_CN_API_KEY)`
/// "Xiaomi MiMo Token Plan (CN) Provider Empty Messages" > "should handle
/// empty string content".
#[tokio::test]
async fn xiaomi_token_plan_cn_empty_string_content() {
    if !has_env("XIAOMI_TOKEN_PLAN_CN_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_CN_API_KEY)`
/// "Xiaomi MiMo Token Plan (CN) Provider Empty Messages" > "should handle
/// whitespace-only content".
#[tokio::test]
async fn xiaomi_token_plan_cn_whitespace_only() {
    if !has_env("XIAOMI_TOKEN_PLAN_CN_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_CN_API_KEY)`
/// "Xiaomi MiMo Token Plan (CN) Provider Empty Messages" > "should handle
/// empty assistant message in conversation".
#[tokio::test]
async fn xiaomi_token_plan_cn_empty_assistant_message() {
    if !has_env("XIAOMI_TOKEN_PLAN_CN_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_AMS_API_KEY)`
/// "Xiaomi MiMo Token Plan (AMS) Provider Empty Messages" > "should handle
/// empty content array".
#[tokio::test]
async fn xiaomi_token_plan_ams_empty_content_array() {
    if !has_env("XIAOMI_TOKEN_PLAN_AMS_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_AMS_API_KEY)`
/// "Xiaomi MiMo Token Plan (AMS) Provider Empty Messages" > "should handle
/// empty string content".
#[tokio::test]
async fn xiaomi_token_plan_ams_empty_string_content() {
    if !has_env("XIAOMI_TOKEN_PLAN_AMS_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_AMS_API_KEY)`
/// "Xiaomi MiMo Token Plan (AMS) Provider Empty Messages" > "should handle
/// whitespace-only content".
#[tokio::test]
async fn xiaomi_token_plan_ams_whitespace_only() {
    if !has_env("XIAOMI_TOKEN_PLAN_AMS_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_AMS_API_KEY)`
/// "Xiaomi MiMo Token Plan (AMS) Provider Empty Messages" > "should handle
/// empty assistant message in conversation".
#[tokio::test]
async fn xiaomi_token_plan_ams_empty_assistant_message() {
    if !has_env("XIAOMI_TOKEN_PLAN_AMS_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_SGP_API_KEY)`
/// "Xiaomi MiMo Token Plan (SGP) Provider Empty Messages" > "should handle
/// empty content array".
#[tokio::test]
async fn xiaomi_token_plan_sgp_empty_content_array() {
    if !has_env("XIAOMI_TOKEN_PLAN_SGP_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_SGP_API_KEY)`
/// "Xiaomi MiMo Token Plan (SGP) Provider Empty Messages" > "should handle
/// empty string content".
#[tokio::test]
async fn xiaomi_token_plan_sgp_empty_string_content() {
    if !has_env("XIAOMI_TOKEN_PLAN_SGP_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_SGP_API_KEY)`
/// "Xiaomi MiMo Token Plan (SGP) Provider Empty Messages" > "should handle
/// whitespace-only content".
#[tokio::test]
async fn xiaomi_token_plan_sgp_whitespace_only() {
    if !has_env("XIAOMI_TOKEN_PLAN_SGP_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_SGP_API_KEY)`
/// "Xiaomi MiMo Token Plan (SGP) Provider Empty Messages" > "should handle
/// empty assistant message in conversation".
#[tokio::test]
async fn xiaomi_token_plan_sgp_empty_assistant_message() {
    if !has_env("XIAOMI_TOKEN_PLAN_SGP_API_KEY") {
        return;
    }
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)` "Qwen
/// Token Plan Provider Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn qwen_token_plan_empty_content_array() {
    if !has_env("QWEN_TOKEN_PLAN_API_KEY") {
        return;
    }
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)` "Qwen
/// Token Plan Provider Empty Messages" > "should handle empty string
/// content".
#[tokio::test]
async fn qwen_token_plan_empty_string_content() {
    if !has_env("QWEN_TOKEN_PLAN_API_KEY") {
        return;
    }
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)` "Qwen
/// Token Plan Provider Empty Messages" > "should handle whitespace-only
/// content".
#[tokio::test]
async fn qwen_token_plan_whitespace_only() {
    if !has_env("QWEN_TOKEN_PLAN_API_KEY") {
        return;
    }
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)` "Qwen
/// Token Plan Provider Empty Messages" > "should handle empty assistant
/// message in conversation".
#[tokio::test]
async fn qwen_token_plan_empty_assistant_message() {
    if !has_env("QWEN_TOKEN_PLAN_API_KEY") {
        return;
    }
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)` "Qwen
/// Token Plan Individual Provider Empty Messages" > "should handle empty
/// content array".
#[tokio::test]
async fn qwen_token_plan_individual_empty_content_array() {
    if !has_env("QWEN_TOKEN_PLAN_API_KEY") {
        return;
    }
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)` "Qwen
/// Token Plan Individual Provider Empty Messages" > "should handle empty
/// string content".
#[tokio::test]
async fn qwen_token_plan_individual_empty_string_content() {
    if !has_env("QWEN_TOKEN_PLAN_API_KEY") {
        return;
    }
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)` "Qwen
/// Token Plan Individual Provider Empty Messages" > "should handle
/// whitespace-only content".
#[tokio::test]
async fn qwen_token_plan_individual_whitespace_only() {
    if !has_env("QWEN_TOKEN_PLAN_API_KEY") {
        return;
    }
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)` "Qwen
/// Token Plan Individual Provider Empty Messages" > "should handle empty
/// assistant message in conversation".
#[tokio::test]
async fn qwen_token_plan_individual_empty_assistant_message() {
    if !has_env("QWEN_TOKEN_PLAN_API_KEY") {
        return;
    }
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_CN_API_KEY)` "Qwen
/// Token Plan (CN) Provider Empty Messages" > "should handle empty content
/// array".
#[tokio::test]
async fn qwen_token_plan_cn_empty_content_array() {
    if !has_env("QWEN_TOKEN_PLAN_CN_API_KEY") {
        return;
    }
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_CN_API_KEY)` "Qwen
/// Token Plan (CN) Provider Empty Messages" > "should handle empty string
/// content".
#[tokio::test]
async fn qwen_token_plan_cn_empty_string_content() {
    if !has_env("QWEN_TOKEN_PLAN_CN_API_KEY") {
        return;
    }
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_CN_API_KEY)` "Qwen
/// Token Plan (CN) Provider Empty Messages" > "should handle whitespace-only
/// content".
#[tokio::test]
async fn qwen_token_plan_cn_whitespace_only() {
    if !has_env("QWEN_TOKEN_PLAN_CN_API_KEY") {
        return;
    }
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_CN_API_KEY)` "Qwen
/// Token Plan (CN) Provider Empty Messages" > "should handle empty assistant
/// message in conversation".
#[tokio::test]
async fn qwen_token_plan_cn_empty_assistant_message() {
    if !has_env("QWEN_TOKEN_PLAN_CN_API_KEY") {
        return;
    }
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.KIMI_API_KEY)` "Kimi For Coding
/// Provider Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn kimi_coding_empty_content_array() {
    if !has_env("KIMI_API_KEY") {
        return;
    }
    let llm = live::model("kimi-coding", "kimi-for-coding");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.KIMI_API_KEY)` "Kimi For Coding
/// Provider Empty Messages" > "should handle empty string content".
#[tokio::test]
async fn kimi_coding_empty_string_content() {
    if !has_env("KIMI_API_KEY") {
        return;
    }
    let llm = live::model("kimi-coding", "kimi-for-coding");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.KIMI_API_KEY)` "Kimi For Coding
/// Provider Empty Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn kimi_coding_whitespace_only() {
    if !has_env("KIMI_API_KEY") {
        return;
    }
    let llm = live::model("kimi-coding", "kimi-for-coding");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.KIMI_API_KEY)` "Kimi For Coding
/// Provider Empty Messages" > "should handle empty assistant message in
/// conversation".
#[tokio::test]
async fn kimi_coding_empty_assistant_message() {
    if !has_env("KIMI_API_KEY") {
        return;
    }
    let llm = live::model("kimi-coding", "kimi-for-coding");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.AI_GATEWAY_API_KEY)` "Vercel AI
/// Gateway Provider Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn vercel_ai_gateway_empty_content_array() {
    if !has_env("AI_GATEWAY_API_KEY") {
        return;
    }
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.AI_GATEWAY_API_KEY)` "Vercel AI
/// Gateway Provider Empty Messages" > "should handle empty string content".
#[tokio::test]
async fn vercel_ai_gateway_empty_string_content() {
    if !has_env("AI_GATEWAY_API_KEY") {
        return;
    }
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.AI_GATEWAY_API_KEY)` "Vercel AI
/// Gateway Provider Empty Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn vercel_ai_gateway_whitespace_only() {
    if !has_env("AI_GATEWAY_API_KEY") {
        return;
    }
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.AI_GATEWAY_API_KEY)` "Vercel AI
/// Gateway Provider Empty Messages" > "should handle empty assistant message
/// in conversation".
#[tokio::test]
async fn vercel_ai_gateway_empty_assistant_message() {
    if !has_env("AI_GATEWAY_API_KEY") {
        return;
    }
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasBedrockCredentials())` "Amazon Bedrock
/// Provider Empty Messages" > "should handle empty content array".
#[tokio::test]
async fn amazon_bedrock_empty_content_array() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    live::test_empty_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasBedrockCredentials())` "Amazon Bedrock
/// Provider Empty Messages" > "should handle empty string content".
#[tokio::test]
async fn amazon_bedrock_empty_string_content() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    live::test_empty_string_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasBedrockCredentials())` "Amazon Bedrock
/// Provider Empty Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn amazon_bedrock_whitespace_only() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    live::test_whitespace_only_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasBedrockCredentials())` "Amazon Bedrock
/// Provider Empty Messages" > "should handle empty assistant message in
/// conversation".
#[tokio::test]
async fn amazon_bedrock_empty_assistant_message() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    live::test_empty_assistant_message(&llm, &live::LiveOptions::default()).await;
}

/// Upstream `it.skipIf(!anthropicOAuthToken)` "Anthropic OAuth Provider Empty
/// Messages" > "should handle empty content array".
#[tokio::test]
async fn anthropic_oauth_empty_content_array() {
    let Some(token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-haiku-4-5");
    live::test_empty_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!anthropicOAuthToken)` "Anthropic OAuth Provider Empty
/// Messages" > "should handle empty string content".
#[tokio::test]
async fn anthropic_oauth_empty_string_content() {
    let Some(token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-haiku-4-5");
    live::test_empty_string_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!anthropicOAuthToken)` "Anthropic OAuth Provider Empty
/// Messages" > "should handle whitespace-only content".
#[tokio::test]
async fn anthropic_oauth_whitespace_only() {
    let Some(token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-haiku-4-5");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!anthropicOAuthToken)` "Anthropic OAuth Provider Empty
/// Messages" > "should handle empty assistant message in conversation".
#[tokio::test]
async fn anthropic_oauth_empty_assistant_message() {
    let Some(token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-haiku-4-5");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!githubCopilotToken)` "GitHub Copilot Provider Empty
/// Messages" > "claude-haiku-4.5 - should handle empty content array".
#[tokio::test]
async fn github_copilot_claude_haiku_4_5_empty_content_array() {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-haiku-4.5");
    live::test_empty_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!githubCopilotToken)` "GitHub Copilot Provider Empty
/// Messages" > "claude-haiku-4.5 - should handle empty string content".
#[tokio::test]
async fn github_copilot_claude_haiku_4_5_empty_string_content() {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-haiku-4.5");
    live::test_empty_string_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!githubCopilotToken)` "GitHub Copilot Provider Empty
/// Messages" > "claude-haiku-4.5 - should handle whitespace-only content".
#[tokio::test]
async fn github_copilot_claude_haiku_4_5_whitespace_only() {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-haiku-4.5");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!githubCopilotToken)` "GitHub Copilot Provider Empty
/// Messages" > "claude-haiku-4.5 - should handle empty assistant message in
/// conversation".
#[tokio::test]
async fn github_copilot_claude_haiku_4_5_empty_assistant_message() {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-haiku-4.5");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!githubCopilotToken)` "GitHub Copilot Provider Empty
/// Messages" > "claude-sonnet-4 - should handle empty content array",
/// probing the catalog's `claude-sonnet-4.6`.
#[tokio::test]
async fn github_copilot_claude_sonnet_4_empty_content_array() {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    live::test_empty_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!githubCopilotToken)` "GitHub Copilot Provider Empty
/// Messages" > "claude-sonnet-4 - should handle empty string content",
/// probing the catalog's `claude-sonnet-4.6`.
#[tokio::test]
async fn github_copilot_claude_sonnet_4_empty_string_content() {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    live::test_empty_string_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!githubCopilotToken)` "GitHub Copilot Provider Empty
/// Messages" > "claude-sonnet-4 - should handle whitespace-only content",
/// probing the catalog's `claude-sonnet-4.6`.
#[tokio::test]
async fn github_copilot_claude_sonnet_4_whitespace_only() {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!githubCopilotToken)` "GitHub Copilot Provider Empty
/// Messages" > "claude-sonnet-4 - should handle empty assistant message in
/// conversation", probing the catalog's `claude-sonnet-4.6`.
#[tokio::test]
async fn github_copilot_claude_sonnet_4_empty_assistant_message() {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!openaiCodexToken)` "OpenAI Codex Provider Empty
/// Messages" > "gpt-5.5 - should handle empty content array".
#[tokio::test]
async fn openai_codex_gpt_5_5_empty_content_array() {
    let Some(token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::test_empty_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!openaiCodexToken)` "OpenAI Codex Provider Empty
/// Messages" > "gpt-5.5 - should handle empty string content".
#[tokio::test]
async fn openai_codex_gpt_5_5_empty_string_content() {
    let Some(token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::test_empty_string_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!openaiCodexToken)` "OpenAI Codex Provider Empty
/// Messages" > "gpt-5.5 - should handle whitespace-only content".
#[tokio::test]
async fn openai_codex_gpt_5_5_whitespace_only() {
    let Some(token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::test_whitespace_only_message(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream `it.skipIf(!openaiCodexToken)` "OpenAI Codex Provider Empty
/// Messages" > "gpt-5.5 - should handle empty assistant message in
/// conversation".
#[tokio::test]
async fn openai_codex_gpt_5_5_empty_assistant_message() {
    let Some(token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::test_empty_assistant_message(&llm, &live::LiveOptions::key(token)).await;
}

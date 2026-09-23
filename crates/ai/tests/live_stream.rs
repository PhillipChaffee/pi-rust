//! The provider generate suites, ported from `packages/ai/test/stream.test.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: one `#[tokio::test]`
//! per upstream `it`, kept in upstream describe order.
//!
//! Like upstream's `describe.skipIf`, a block without its provider credential
//! returns early — an env var for the api-key providers, the Azure, Bedrock,
//! and Cloudflare credential guards for the cloud blocks, the Vertex
//! project-and-region pair (or the Vertex api key) for the Vertex block, the
//! resolved OAuth tokens (credential store first, then env) for the OAuth
//! blocks, and `PI_NO_LOCAL_LLM` unset plus an answering `ollama` binary for
//! the Ollama block. Upstream retries each probe three times (`{ retry: 3 }`);
//! the shared harness carries no retry wrapper, so the port runs each probe
//! once. Upstream env-gated blocks pass no options and ride compat's env-key
//! resolution; the port hands every env-gated block the gate's value as the
//! api key — the value compat resolves anyway — because the extras-carrying
//! probes dispatch through an adapter directly and the adapters do not apply
//! compat's env fallback. The three Bedrock claude-opus-4-6 probes are
//! bespoke: they capture the request payload through the `onPayload` hook.
//! The Ollama block folds upstream's `beforeAll`/`afterAll` (pull the model
//! when missing, serve, readiness poll, teardown) into its single test body.

#![expect(
    clippy::print_stdout,
    reason = "the Ollama setup surfaces its pull and readiness failures through the log, upstream's console.log/console.warn"
)]

use std::collections::BTreeMap;
use std::process::Stdio;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use pi_ai::api::anthropic_messages::AnthropicEffort;
use pi_ai::api::google_shared::GoogleThinkingControl;
use pi_ai::types::{
    Api, Context, Modality, Model, ModelCost, OnPayload, ProviderHeaders, ProviderId, StopReason,
    ThinkingLevel, Transport,
};
use serde_json::{Value, json};

mod common;
use common::live;
use common::live::LiveOptions;

/// The options a medium-reasoning thinking probe sends, upstream's
/// `{ reasoningEffort: "medium" }` plus the gate's credential.
fn medium_reasoning_options(api_key: String) -> LiveOptions {
    LiveOptions {
        api_key: Some(api_key),
        reasoning: Some(ThinkingLevel::Medium),
        ..LiveOptions::default()
    }
}

/// The options a high-reasoning thinking probe sends, upstream's
/// `{ reasoningEffort: "high" }` plus the gate's credential.
fn high_reasoning_options(api_key: String) -> LiveOptions {
    LiveOptions {
        api_key: Some(api_key),
        reasoning: Some(ThinkingLevel::High),
        ..LiveOptions::default()
    }
}

/// The options an xhigh-reasoning thinking probe sends, upstream's
/// `{ reasoningEffort: "xhigh" }` plus the resolved token.
fn xhigh_reasoning_options(api_key: String) -> LiveOptions {
    LiveOptions {
        api_key: Some(api_key),
        reasoning: Some(ThinkingLevel::Xhigh),
        ..LiveOptions::default()
    }
}

/// The options a budget-thinking probe sends, upstream's
/// `{ thinkingEnabled: true, thinkingBudgetTokens: 2048 }` plus the gate's
/// credential.
fn anthropic_budget_options(api_key: String) -> LiveOptions {
    LiveOptions {
        api_key: Some(api_key),
        thinking_enabled: Some(true),
        thinking_budget_tokens: Some(2048),
        ..LiveOptions::default()
    }
}

/// The options a managed-thinking probe sends, upstream's
/// `{ thinkingEnabled: true, reasoningEffort: "high" }` plus the gate's
/// credential; the reasoning level rides along the way upstream's extra rides
/// unread on the anthropic wire.
fn managed_thinking_options(api_key: String) -> LiveOptions {
    LiveOptions {
        api_key: Some(api_key),
        thinking_enabled: Some(true),
        reasoning: Some(ThinkingLevel::High),
        ..LiveOptions::default()
    }
}

/// The options an extended-thinking probe sends, upstream's
/// `{ thinkingEnabled: true }` plus the gate's credential.
fn thinking_enabled_options(api_key: String) -> LiveOptions {
    LiveOptions {
        api_key: Some(api_key),
        thinking_enabled: Some(true),
        ..LiveOptions::default()
    }
}

/// The options a Gemini thinking probe sends: the gate's credential plus the
/// enabled thinking control with a token budget, upstream's
/// `{ thinking: { enabled: true, budgetTokens: n } }`.
fn gemini_thinking_options(api_key: String, budget_tokens: i64) -> LiveOptions {
    LiveOptions {
        api_key: Some(api_key),
        google_thinking: Some(GoogleThinkingControl {
            enabled: true,
            budget_tokens: Some(budget_tokens),
            level: None,
        }),
        ..LiveOptions::default()
    }
}

/// The options a Vertex probe sends: the GCP project and region, upstream's
/// `vertexOptions`.
fn vertex_options(project: String, location: String) -> LiveOptions {
    LiveOptions {
        vertex_project: Some(project),
        vertex_location: Some(location),
        ..LiveOptions::default()
    }
}

/// The options a Vertex thinking probe sends: the GCP project and region plus
/// the enabled thinking control at a 1024-token budget and a provider-native
/// level, upstream's
/// `{ ...vertexOptions, thinking: { enabled: true, budgetTokens: 1024, level } }`.
fn vertex_thinking_options(project: String, location: String, level: &str) -> LiveOptions {
    LiveOptions {
        vertex_project: Some(project),
        vertex_location: Some(location),
        google_thinking: Some(GoogleThinkingControl {
            enabled: true,
            budget_tokens: Some(1024),
            level: Some(level.to_owned()),
        }),
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
/// credentials, upstream's `hasAzureOpenAICredentials` describe gate.
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

/// The BYOK headers a Cloudflare AI Gateway block sends: the upstream
/// provider's key as the Authorization bearer, upstream's
/// `{ headers: { Authorization: `Bearer ${key}` } }`. `None` unless the
/// gateway credentials and the upstream provider key are both configured,
/// upstream's combined `describe.skipIf` gate.
fn byok_headers(byok_env: &str) -> Option<ProviderHeaders> {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return None;
    }
    let byok_key = live::env_key(byok_env)?;
    Some(BTreeMap::from([(
        "Authorization".to_owned(),
        Some(format!("Bearer {byok_key}")),
    )]))
}

/// Upstream `describe.skipIf(!process.env.GEMINI_API_KEY)` "Gemini Provider
/// (gemini-2.5-flash)" / "should complete basic text generation".
#[tokio::test]
async fn gemini_basic_text_generation() {
    let Some(api_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let llm = live::model("google", "gemini-2.5-flash");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Gemini Provider (gemini-2.5-flash)" / "should handle tool
/// calling".
#[tokio::test]
async fn gemini_tool_calling() {
    let Some(api_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let llm = live::model("google", "gemini-2.5-flash");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Gemini Provider (gemini-2.5-flash)" / "should handle streaming".
#[tokio::test]
async fn gemini_streaming() {
    let Some(api_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let llm = live::model("google", "gemini-2.5-flash");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Gemini Provider (gemini-2.5-flash)" / "should handle thinking":
/// thinking enabled at a 1024-token budget.
#[tokio::test]
async fn gemini_thinking() {
    let Some(api_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let llm = live::model("google", "gemini-2.5-flash");
    live::handle_thinking(&llm, &gemini_thinking_options(api_key, 1024)).await;
}

/// Upstream "Gemini Provider (gemini-2.5-flash)" / "should handle multi-turn
/// with thinking and tools": thinking enabled at a 2048-token budget.
#[tokio::test]
async fn gemini_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let llm = live::model("google", "gemini-2.5-flash");
    live::multi_turn(&llm, &gemini_thinking_options(api_key, 2048)).await;
}

/// Upstream "Gemini Provider (gemini-2.5-flash)" / "should handle image
/// input".
#[tokio::test]
async fn gemini_image_input() {
    let Some(api_key) = live::env_key("GEMINI_API_KEY") else {
        return;
    };
    let llm = live::model("google", "gemini-2.5-flash");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe("Google Vertex Provider (gemini-3-flash-preview)")` /
/// `it.skipIf(!isVertexConfigured)` "should complete basic text generation":
/// the GCP project and region gate, upstream's `vertexOptions`.
#[tokio::test]
async fn vertex_basic_text_generation() {
    let (Some(project), Some(location)) = (
        live::env_key("GOOGLE_CLOUD_PROJECT").or_else(|| live::env_key("GCLOUD_PROJECT")),
        live::env_key("GOOGLE_CLOUD_LOCATION"),
    ) else {
        return;
    };
    let llm = live::model("google-vertex", "gemini-3-flash-preview");
    live::basic_text_generation(&llm, &vertex_options(project, location)).await;
}

/// Upstream "Google Vertex Provider (gemini-3-flash-preview)" /
/// `it.skipIf(!vertexApiKey)` "should complete basic text generation with
/// Vertex API key".
#[tokio::test]
async fn vertex_basic_text_generation_with_api_key() {
    let Some(api_key) = live::env_key("GOOGLE_CLOUD_API_KEY") else {
        return;
    };
    let llm = live::model("google-vertex", "gemini-3-flash-preview");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Google Vertex Provider (gemini-3-flash-preview)" /
/// `it.skipIf(!isVertexConfigured)` "should handle tool calling".
#[tokio::test]
async fn vertex_tool_calling() {
    let (Some(project), Some(location)) = (
        live::env_key("GOOGLE_CLOUD_PROJECT").or_else(|| live::env_key("GCLOUD_PROJECT")),
        live::env_key("GOOGLE_CLOUD_LOCATION"),
    ) else {
        return;
    };
    let llm = live::model("google-vertex", "gemini-3-flash-preview");
    live::handle_tool_call(&llm, &vertex_options(project, location)).await;
}

/// Upstream "Google Vertex Provider (gemini-3-flash-preview)" /
/// `it.skipIf(!isVertexConfigured)` "should handle thinking": thinking
/// enabled at a 1024-token budget with the provider-native low level,
/// upstream's `ThinkingLevel.LOW`.
#[tokio::test]
async fn vertex_thinking() {
    let (Some(project), Some(location)) = (
        live::env_key("GOOGLE_CLOUD_PROJECT").or_else(|| live::env_key("GCLOUD_PROJECT")),
        live::env_key("GOOGLE_CLOUD_LOCATION"),
    ) else {
        return;
    };
    let llm = live::model("google-vertex", "gemini-3-flash-preview");
    live::handle_thinking(&llm, &vertex_thinking_options(project, location, "LOW")).await;
}

/// Upstream "Google Vertex Provider (gemini-3-flash-preview)" /
/// `it.skipIf(!isVertexConfigured)` "should handle streaming".
#[tokio::test]
async fn vertex_streaming() {
    let (Some(project), Some(location)) = (
        live::env_key("GOOGLE_CLOUD_PROJECT").or_else(|| live::env_key("GCLOUD_PROJECT")),
        live::env_key("GOOGLE_CLOUD_LOCATION"),
    ) else {
        return;
    };
    let llm = live::model("google-vertex", "gemini-3-flash-preview");
    live::handle_streaming(&llm, &vertex_options(project, location)).await;
}

/// Upstream "Google Vertex Provider (gemini-3-flash-preview)" /
/// `it.skipIf(!isVertexConfigured)` "should handle multi-turn with thinking
/// and tools": thinking enabled at a 1024-token budget with the
/// provider-native medium level, upstream's `ThinkingLevel.MEDIUM`.
#[tokio::test]
async fn vertex_multi_turn_with_thinking_and_tools() {
    let (Some(project), Some(location)) = (
        live::env_key("GOOGLE_CLOUD_PROJECT").or_else(|| live::env_key("GCLOUD_PROJECT")),
        live::env_key("GOOGLE_CLOUD_LOCATION"),
    ) else {
        return;
    };
    let llm = live::model("google-vertex", "gemini-3-flash-preview");
    live::multi_turn(&llm, &vertex_thinking_options(project, location, "MEDIUM")).await;
}

/// Upstream "Google Vertex Provider (gemini-3-flash-preview)" /
/// `it.skipIf(!isVertexConfigured)` "should handle image input".
#[tokio::test]
async fn vertex_image_input() {
    let (Some(project), Some(location)) = (
        live::env_key("GOOGLE_CLOUD_PROJECT").or_else(|| live::env_key("GCLOUD_PROJECT")),
        live::env_key("GOOGLE_CLOUD_LOCATION"),
    ) else {
        return;
    };
    let llm = live::model("google-vertex", "gemini-3-flash-preview");
    live::handle_image(&llm, &vertex_options(project, location)).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Completions
/// Provider (gpt-4o-mini)" / "should complete basic text generation": the
/// catalog entry retargeted at the openai-completions wire.
#[tokio::test]
async fn openai_completions_basic_text_generation() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = openai_completions_llm();
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "OpenAI Completions Provider (gpt-4o-mini)" / "should handle tool
/// calling".
#[tokio::test]
async fn openai_completions_tool_calling() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = openai_completions_llm();
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "OpenAI Completions Provider (gpt-4o-mini)" / "should handle
/// streaming".
#[tokio::test]
async fn openai_completions_streaming() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = openai_completions_llm();
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "OpenAI Completions Provider (gpt-4o-mini)" / "should handle image
/// input".
#[tokio::test]
async fn openai_completions_image_input() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = openai_completions_llm();
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.DEEPSEEK_API_KEY)` "DeepSeek
/// Provider (deepseek-flash via OpenAI Completions)" / "should complete basic
/// text generation".
#[tokio::test]
async fn deepseek_basic_text_generation() {
    let Some(api_key) = live::env_key("DEEPSEEK_API_KEY") else {
        return;
    };
    let llm = live::model("deepseek", "deepseek-flash");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "DeepSeek Provider (deepseek-flash via OpenAI Completions)" /
/// "should handle tool calling".
#[tokio::test]
async fn deepseek_tool_calling() {
    let Some(api_key) = live::env_key("DEEPSEEK_API_KEY") else {
        return;
    };
    let llm = live::model("deepseek", "deepseek-flash");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "DeepSeek Provider (deepseek-flash via OpenAI Completions)" /
/// "should handle streaming".
#[tokio::test]
async fn deepseek_streaming() {
    let Some(api_key) = live::env_key("DEEPSEEK_API_KEY") else {
        return;
    };
    let llm = live::model("deepseek", "deepseek-flash");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "DeepSeek Provider (deepseek-flash via OpenAI Completions)" /
/// "should handle thinking mode" at high reasoning effort.
#[tokio::test]
async fn deepseek_thinking_mode() {
    let Some(api_key) = live::env_key("DEEPSEEK_API_KEY") else {
        return;
    };
    let llm = live::model("deepseek", "deepseek-flash");
    live::handle_thinking(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream "DeepSeek Provider (deepseek-flash via OpenAI Completions)" /
/// "should handle multi-turn with thinking and tools" at high reasoning
/// effort.
#[tokio::test]
async fn deepseek_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("DEEPSEEK_API_KEY") else {
        return;
    };
    let llm = live::model("deepseek", "deepseek-flash");
    live::multi_turn(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Responses
/// Provider (gpt-5.4)" / "should complete basic text generation".
#[tokio::test]
async fn openai_responses_basic_text_generation() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("openai", "gpt-5.4");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "OpenAI Responses Provider (gpt-5.4)" / "should handle tool
/// calling".
#[tokio::test]
async fn openai_responses_tool_calling() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("openai", "gpt-5.4");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "OpenAI Responses Provider (gpt-5.4)" / "should handle streaming".
#[tokio::test]
async fn openai_responses_streaming() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("openai", "gpt-5.4");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "OpenAI Responses Provider (gpt-5.4)" / "should handle thinking"
/// at high reasoning effort, upstream's `{ retry: 2 }`.
#[tokio::test]
async fn openai_responses_thinking() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("openai", "gpt-5.4");
    live::handle_thinking(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream "OpenAI Responses Provider (gpt-5.4)" / "should handle multi-turn
/// with thinking and tools" at high reasoning effort.
#[tokio::test]
async fn openai_responses_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("openai", "gpt-5.4");
    live::multi_turn(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream "OpenAI Responses Provider (gpt-5.4)" / "should handle image
/// input".
#[tokio::test]
async fn openai_responses_image_input() {
    let Some(api_key) = live::env_key("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("openai", "gpt-5.4");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.ANTHROPIC_API_KEY)` "Anthropic
/// Provider (claude-haiku-4-5)" / "should complete basic text generation"
/// with extended thinking enabled.
#[tokio::test]
async fn anthropic_basic_text_generation() {
    let Some(api_key) = live::env_key("ANTHROPIC_API_KEY") else {
        return;
    };
    let llm = live::model("anthropic", "claude-haiku-4-5");
    live::basic_text_generation(&llm, &thinking_enabled_options(api_key)).await;
}

/// Upstream "Anthropic Provider (claude-haiku-4-5)" / "should handle tool
/// calling".
#[tokio::test]
async fn anthropic_tool_calling() {
    let Some(api_key) = live::env_key("ANTHROPIC_API_KEY") else {
        return;
    };
    let llm = live::model("anthropic", "claude-haiku-4-5");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Anthropic Provider (claude-haiku-4-5)" / "should handle
/// streaming".
#[tokio::test]
async fn anthropic_streaming() {
    let Some(api_key) = live::env_key("ANTHROPIC_API_KEY") else {
        return;
    };
    let llm = live::model("anthropic", "claude-haiku-4-5");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Anthropic Provider (claude-haiku-4-5)" / "should handle image
/// input".
#[tokio::test]
async fn anthropic_image_input() {
    let Some(api_key) = live::env_key("ANTHROPIC_API_KEY") else {
        return;
    };
    let llm = live::model("anthropic", "claude-haiku-4-5");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe.skipIf(!hasAzureOpenAICredentials())` "Azure OpenAI
/// Responses Provider (gpt-4o-mini)" / "should complete basic text
/// generation" with the mapped deployment override.
#[tokio::test]
async fn azure_openai_responses_basic_text_generation() {
    let Some((llm, options)) = azure_model_and_options() else {
        return;
    };
    live::basic_text_generation(&llm, &options).await;
}

/// Upstream "Azure OpenAI Responses Provider (gpt-4o-mini)" / "should handle
/// tool calling" with the mapped deployment override.
#[tokio::test]
async fn azure_openai_responses_tool_calling() {
    let Some((llm, options)) = azure_model_and_options() else {
        return;
    };
    live::handle_tool_call(&llm, &options).await;
}

/// Upstream "Azure OpenAI Responses Provider (gpt-4o-mini)" / "should handle
/// streaming" with the mapped deployment override.
#[tokio::test]
async fn azure_openai_responses_streaming() {
    let Some((llm, options)) = azure_model_and_options() else {
        return;
    };
    live::handle_streaming(&llm, &options).await;
}

/// Upstream "Azure OpenAI Responses Provider (gpt-4o-mini)" / "should handle
/// image input" with the mapped deployment override.
#[tokio::test]
async fn azure_openai_responses_image_input() {
    let Some((llm, options)) = azure_model_and_options() else {
        return;
    };
    live::handle_image(&llm, &options).await;
}

/// Upstream `describe.skipIf(!process.env.XAI_API_KEY)` "xAI Provider
/// (grok-4.3 via OpenAI Responses)" / "should complete basic text
/// generation".
#[tokio::test]
async fn xai_basic_text_generation() {
    let Some(api_key) = live::env_key("XAI_API_KEY") else {
        return;
    };
    let llm = live::model("xai", "grok-4.3");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "xAI Provider (grok-4.3 via OpenAI Responses)" / "should handle
/// tool calling".
#[tokio::test]
async fn xai_tool_calling() {
    let Some(api_key) = live::env_key("XAI_API_KEY") else {
        return;
    };
    let llm = live::model("xai", "grok-4.3");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "xAI Provider (grok-4.3 via OpenAI Responses)" / "should handle
/// streaming".
#[tokio::test]
async fn xai_streaming() {
    let Some(api_key) = live::env_key("XAI_API_KEY") else {
        return;
    };
    let llm = live::model("xai", "grok-4.3");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "xAI Provider (grok-4.3 via OpenAI Responses)" / "should handle
/// thinking mode" at medium reasoning effort.
#[tokio::test]
async fn xai_thinking_mode() {
    let Some(api_key) = live::env_key("XAI_API_KEY") else {
        return;
    };
    let llm = live::model("xai", "grok-4.3");
    live::handle_thinking(&llm, &medium_reasoning_options(api_key)).await;
}

/// Upstream "xAI Provider (grok-4.3 via OpenAI Responses)" / "should handle
/// multi-turn with thinking and tools" at medium reasoning effort.
#[tokio::test]
async fn xai_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("XAI_API_KEY") else {
        return;
    };
    let llm = live::model("xai", "grok-4.3");
    live::multi_turn(&llm, &medium_reasoning_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.GROQ_API_KEY)` "Groq Provider
/// (gpt-oss-20b via OpenAI Completions)" / "should complete basic text
/// generation".
#[tokio::test]
async fn groq_basic_text_generation() {
    let Some(api_key) = live::env_key("GROQ_API_KEY") else {
        return;
    };
    let llm = live::model("groq", "openai/gpt-oss-20b");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Groq Provider (gpt-oss-20b via OpenAI Completions)" / "should
/// handle tool calling".
#[tokio::test]
async fn groq_tool_calling() {
    let Some(api_key) = live::env_key("GROQ_API_KEY") else {
        return;
    };
    let llm = live::model("groq", "openai/gpt-oss-20b");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Groq Provider (gpt-oss-20b via OpenAI Completions)" / "should
/// handle streaming".
#[tokio::test]
async fn groq_streaming() {
    let Some(api_key) = live::env_key("GROQ_API_KEY") else {
        return;
    };
    let llm = live::model("groq", "openai/gpt-oss-20b");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Groq Provider (gpt-oss-20b via OpenAI Completions)" / "should
/// handle thinking mode" at medium reasoning effort.
#[tokio::test]
async fn groq_thinking_mode() {
    let Some(api_key) = live::env_key("GROQ_API_KEY") else {
        return;
    };
    let llm = live::model("groq", "openai/gpt-oss-20b");
    live::handle_thinking(&llm, &medium_reasoning_options(api_key)).await;
}

/// Upstream "Groq Provider (gpt-oss-20b via OpenAI Completions)" / "should
/// handle multi-turn with thinking and tools" at medium reasoning effort.
#[tokio::test]
async fn groq_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("GROQ_API_KEY") else {
        return;
    };
    let llm = live::model("groq", "openai/gpt-oss-20b");
    live::multi_turn(&llm, &medium_reasoning_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.CEREBRAS_API_KEY)` "Cerebras
/// Provider (gpt-oss-120b via OpenAI Completions)" / "should complete basic
/// text generation".
#[tokio::test]
async fn cerebras_basic_text_generation() {
    let Some(api_key) = live::env_key("CEREBRAS_API_KEY") else {
        return;
    };
    let llm = live::model("cerebras", "gpt-oss-120b");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Cerebras Provider (gpt-oss-120b via OpenAI Completions)" /
/// "should handle tool calling".
#[tokio::test]
async fn cerebras_tool_calling() {
    let Some(api_key) = live::env_key("CEREBRAS_API_KEY") else {
        return;
    };
    let llm = live::model("cerebras", "gpt-oss-120b");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Cerebras Provider (gpt-oss-120b via OpenAI Completions)" /
/// "should handle streaming".
#[tokio::test]
async fn cerebras_streaming() {
    let Some(api_key) = live::env_key("CEREBRAS_API_KEY") else {
        return;
    };
    let llm = live::model("cerebras", "gpt-oss-120b");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Cerebras Provider (gpt-oss-120b via OpenAI Completions)" /
/// "should handle thinking mode" at medium reasoning effort.
#[tokio::test]
async fn cerebras_thinking_mode() {
    let Some(api_key) = live::env_key("CEREBRAS_API_KEY") else {
        return;
    };
    let llm = live::model("cerebras", "gpt-oss-120b");
    live::handle_thinking(&llm, &medium_reasoning_options(api_key)).await;
}

/// Upstream "Cerebras Provider (gpt-oss-120b via OpenAI Completions)" /
/// "should handle multi-turn with thinking and tools" at medium reasoning
/// effort.
#[tokio::test]
async fn cerebras_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("CEREBRAS_API_KEY") else {
        return;
    };
    let llm = live::model("cerebras", "gpt-oss-120b");
    live::multi_turn(&llm, &medium_reasoning_options(api_key)).await;
}

/// Upstream `describe.skipIf(!hasCloudflareWorkersAICredentials())`
/// "Cloudflare Workers AI Provider (Kimi K2.6 via OpenAI Completions)" /
/// "should complete basic text generation".
#[tokio::test]
async fn cloudflare_workers_ai_basic_text_generation() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    live::basic_text_generation(&llm, &LiveOptions::default()).await;
}

/// Upstream "Cloudflare Workers AI Provider (Kimi K2.6 via OpenAI
/// Completions)" / "should handle tool calling".
#[tokio::test]
async fn cloudflare_workers_ai_tool_calling() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    live::handle_tool_call(&llm, &LiveOptions::default()).await;
}

/// Upstream "Cloudflare Workers AI Provider (Kimi K2.6 via OpenAI
/// Completions)" / "should handle streaming".
#[tokio::test]
async fn cloudflare_workers_ai_streaming() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    live::handle_streaming(&llm, &LiveOptions::default()).await;
}

/// Upstream "Cloudflare Workers AI Provider (Kimi K2.6 via OpenAI
/// Completions)" / "should handle thinking mode" at medium reasoning effort.
#[tokio::test]
async fn cloudflare_workers_ai_thinking_mode() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    let options = LiveOptions {
        reasoning: Some(ThinkingLevel::Medium),
        ..LiveOptions::default()
    };
    live::handle_thinking(&llm, &options).await;
}

/// Upstream "Cloudflare Workers AI Provider (Kimi K2.6 via OpenAI
/// Completions)" / "should handle multi-turn with thinking and tools" at
/// medium reasoning effort.
#[tokio::test]
async fn cloudflare_workers_ai_multi_turn_with_thinking_and_tools() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    let options = LiveOptions {
        reasoning: Some(ThinkingLevel::Medium),
        ..LiveOptions::default()
    };
    live::multi_turn(&llm, &options).await;
}

/// Upstream `describe.skipIf(!hasCloudflareAiGatewayCredentials())`
/// "Cloudflare AI Gateway → Workers AI (Kimi K2.6 via /compat)" / "should
/// complete basic text generation".
#[tokio::test]
async fn cloudflare_ai_gateway_workers_ai_basic_text_generation() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    live::basic_text_generation(&llm, &LiveOptions::default()).await;
}

/// Upstream "Cloudflare AI Gateway → Workers AI (Kimi K2.6 via /compat)" /
/// "should handle tool calling".
#[tokio::test]
async fn cloudflare_ai_gateway_workers_ai_tool_calling() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    live::handle_tool_call(&llm, &LiveOptions::default()).await;
}

/// Upstream "Cloudflare AI Gateway → Workers AI (Kimi K2.6 via /compat)" /
/// "should handle streaming".
#[tokio::test]
async fn cloudflare_ai_gateway_workers_ai_streaming() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    live::handle_streaming(&llm, &LiveOptions::default()).await;
}

/// Upstream "Cloudflare AI Gateway → Workers AI (Kimi K2.6 via /compat)" /
/// "should handle thinking mode" at medium reasoning effort.
#[tokio::test]
async fn cloudflare_ai_gateway_workers_ai_thinking_mode() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    let options = LiveOptions {
        reasoning: Some(ThinkingLevel::Medium),
        ..LiveOptions::default()
    };
    live::handle_thinking(&llm, &options).await;
}

/// Upstream "Cloudflare AI Gateway → Workers AI (Kimi K2.6 via /compat)" /
/// "should handle multi-turn with thinking and tools" at medium reasoning
/// effort.
#[tokio::test]
async fn cloudflare_ai_gateway_workers_ai_multi_turn_with_thinking_and_tools() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    let options = LiveOptions {
        reasoning: Some(ThinkingLevel::Medium),
        ..LiveOptions::default()
    };
    live::multi_turn(&llm, &options).await;
}

/// Upstream `describe.skipIf(!hasCloudflareAiGatewayCredentials() ||
/// !process.env.OPENAI_API_KEY)` "Cloudflare AI Gateway → OpenAI BYOK
/// (gpt-5.1 via /openai responses)" / "should complete basic text
/// generation" with the BYOK Authorization header.
#[tokio::test]
async fn cloudflare_ai_gateway_openai_byok_basic_text_generation() {
    let Some(headers) = byok_headers("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("cloudflare-ai-gateway", "gpt-5.1");
    let options = LiveOptions {
        headers: Some(headers),
        ..LiveOptions::default()
    };
    live::basic_text_generation(&llm, &options).await;
}

/// Upstream "Cloudflare AI Gateway → OpenAI BYOK (gpt-5.1 via /openai
/// responses)" / "should handle tool calling" with the BYOK Authorization
/// header.
#[tokio::test]
async fn cloudflare_ai_gateway_openai_byok_tool_calling() {
    let Some(headers) = byok_headers("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("cloudflare-ai-gateway", "gpt-5.1");
    let options = LiveOptions {
        headers: Some(headers),
        ..LiveOptions::default()
    };
    live::handle_tool_call(&llm, &options).await;
}

/// Upstream "Cloudflare AI Gateway → OpenAI BYOK (gpt-5.1 via /openai
/// responses)" / "should handle streaming" with the BYOK Authorization
/// header.
#[tokio::test]
async fn cloudflare_ai_gateway_openai_byok_streaming() {
    let Some(headers) = byok_headers("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("cloudflare-ai-gateway", "gpt-5.1");
    let options = LiveOptions {
        headers: Some(headers),
        ..LiveOptions::default()
    };
    live::handle_streaming(&llm, &options).await;
}

/// Upstream "Cloudflare AI Gateway → OpenAI BYOK (gpt-5.1 via /openai
/// responses)" / "should handle thinking mode": the BYOK header plus
/// extended thinking at medium reasoning effort, upstream's
/// `thinkingOptions`.
#[tokio::test]
async fn cloudflare_ai_gateway_openai_byok_thinking_mode() {
    let Some(headers) = byok_headers("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("cloudflare-ai-gateway", "gpt-5.1");
    let options = LiveOptions {
        headers: Some(headers),
        thinking_enabled: Some(true),
        reasoning: Some(ThinkingLevel::Medium),
        ..LiveOptions::default()
    };
    live::handle_thinking(&llm, &options).await;
}

/// Upstream "Cloudflare AI Gateway → OpenAI BYOK (gpt-5.1 via /openai
/// responses)" / "should handle multi-turn with thinking and tools" with the
/// BYOK header plus extended thinking at medium reasoning effort.
#[tokio::test]
async fn cloudflare_ai_gateway_openai_byok_multi_turn_with_thinking_and_tools() {
    let Some(headers) = byok_headers("OPENAI_API_KEY") else {
        return;
    };
    let llm = live::model("cloudflare-ai-gateway", "gpt-5.1");
    let options = LiveOptions {
        headers: Some(headers),
        thinking_enabled: Some(true),
        reasoning: Some(ThinkingLevel::Medium),
        ..LiveOptions::default()
    };
    live::multi_turn(&llm, &options).await;
}

/// Upstream `describe.skipIf(!hasCloudflareAiGatewayCredentials() ||
/// !process.env.ANTHROPIC_API_KEY)` "Cloudflare AI Gateway → Anthropic BYOK
/// (claude-sonnet-4.5 via /anthropic messages)" / "should complete basic text
/// generation" with the BYOK Authorization header.
#[tokio::test]
async fn cloudflare_ai_gateway_anthropic_byok_basic_text_generation() {
    let Some(headers) = byok_headers("ANTHROPIC_API_KEY") else {
        return;
    };
    let llm = live::model("cloudflare-ai-gateway", "claude-sonnet-4.5");
    let options = LiveOptions {
        headers: Some(headers),
        ..LiveOptions::default()
    };
    live::basic_text_generation(&llm, &options).await;
}

/// Upstream "Cloudflare AI Gateway → Anthropic BYOK (claude-sonnet-4.5 via
/// /anthropic messages)" / "should handle tool calling" with the BYOK
/// Authorization header.
#[tokio::test]
async fn cloudflare_ai_gateway_anthropic_byok_tool_calling() {
    let Some(headers) = byok_headers("ANTHROPIC_API_KEY") else {
        return;
    };
    let llm = live::model("cloudflare-ai-gateway", "claude-sonnet-4.5");
    let options = LiveOptions {
        headers: Some(headers),
        ..LiveOptions::default()
    };
    live::handle_tool_call(&llm, &options).await;
}

/// Upstream "Cloudflare AI Gateway → Anthropic BYOK (claude-sonnet-4.5 via
/// /anthropic messages)" / "should handle streaming" with the BYOK
/// Authorization header.
#[tokio::test]
async fn cloudflare_ai_gateway_anthropic_byok_streaming() {
    let Some(headers) = byok_headers("ANTHROPIC_API_KEY") else {
        return;
    };
    let llm = live::model("cloudflare-ai-gateway", "claude-sonnet-4.5");
    let options = LiveOptions {
        headers: Some(headers),
        ..LiveOptions::default()
    };
    live::handle_streaming(&llm, &options).await;
}

/// Upstream "Cloudflare AI Gateway → Anthropic BYOK (claude-sonnet-4.5 via
/// /anthropic messages)" / "should handle thinking mode": the BYOK header
/// plus extended thinking at high reasoning effort, upstream's
/// `thinkingOptions`.
#[tokio::test]
async fn cloudflare_ai_gateway_anthropic_byok_thinking_mode() {
    let Some(headers) = byok_headers("ANTHROPIC_API_KEY") else {
        return;
    };
    let llm = live::model("cloudflare-ai-gateway", "claude-sonnet-4.5");
    let options = LiveOptions {
        headers: Some(headers),
        thinking_enabled: Some(true),
        reasoning: Some(ThinkingLevel::High),
        ..LiveOptions::default()
    };
    live::handle_thinking(&llm, &options).await;
}

/// Upstream "Cloudflare AI Gateway → Anthropic BYOK (claude-sonnet-4.5 via
/// /anthropic messages)" / "should handle multi-turn with thinking and
/// tools" with the BYOK header plus extended thinking at high reasoning
/// effort.
#[tokio::test]
async fn cloudflare_ai_gateway_anthropic_byok_multi_turn_with_thinking_and_tools() {
    let Some(headers) = byok_headers("ANTHROPIC_API_KEY") else {
        return;
    };
    let llm = live::model("cloudflare-ai-gateway", "claude-sonnet-4.5");
    let options = LiveOptions {
        headers: Some(headers),
        thinking_enabled: Some(true),
        reasoning: Some(ThinkingLevel::High),
        ..LiveOptions::default()
    };
    live::multi_turn(&llm, &options).await;
}

/// Upstream `describe.skipIf(!process.env.HF_TOKEN)` "Hugging Face Provider
/// (Kimi-K2.5 via OpenAI Completions)" / "should complete basic text
/// generation".
#[tokio::test]
async fn huggingface_basic_text_generation() {
    let Some(api_key) = live::env_key("HF_TOKEN") else {
        return;
    };
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Hugging Face Provider (Kimi-K2.5 via OpenAI Completions)" /
/// "should handle tool calling".
#[tokio::test]
async fn huggingface_tool_calling() {
    let Some(api_key) = live::env_key("HF_TOKEN") else {
        return;
    };
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Hugging Face Provider (Kimi-K2.5 via OpenAI Completions)" /
/// "should handle streaming".
#[tokio::test]
async fn huggingface_streaming() {
    let Some(api_key) = live::env_key("HF_TOKEN") else {
        return;
    };
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Hugging Face Provider (Kimi-K2.5 via OpenAI Completions)" /
/// "should handle thinking mode" at medium reasoning effort.
#[tokio::test]
async fn huggingface_thinking_mode() {
    let Some(api_key) = live::env_key("HF_TOKEN") else {
        return;
    };
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    live::handle_thinking(&llm, &medium_reasoning_options(api_key)).await;
}

/// Upstream "Hugging Face Provider (Kimi-K2.5 via OpenAI Completions)" /
/// "should handle multi-turn with thinking and tools" at medium reasoning
/// effort.
#[tokio::test]
async fn huggingface_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("HF_TOKEN") else {
        return;
    };
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    live::multi_turn(&llm, &medium_reasoning_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.TOGETHER_API_KEY)` "Together AI
/// Provider (Kimi-K2.6 via OpenAI Completions)" / "should complete basic text
/// generation".
#[tokio::test]
async fn together_basic_text_generation() {
    let Some(api_key) = live::env_key("TOGETHER_API_KEY") else {
        return;
    };
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Together AI Provider (Kimi-K2.6 via OpenAI Completions)" /
/// "should handle tool calling".
#[tokio::test]
async fn together_tool_calling() {
    let Some(api_key) = live::env_key("TOGETHER_API_KEY") else {
        return;
    };
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Together AI Provider (Kimi-K2.6 via OpenAI Completions)" /
/// "should handle streaming".
#[tokio::test]
async fn together_streaming() {
    let Some(api_key) = live::env_key("TOGETHER_API_KEY") else {
        return;
    };
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Together AI Provider (Kimi-K2.6 via OpenAI Completions)" /
/// "should handle thinking mode" at high reasoning effort.
#[tokio::test]
async fn together_thinking_mode() {
    let Some(api_key) = live::env_key("TOGETHER_API_KEY") else {
        return;
    };
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::handle_thinking(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream "Together AI Provider (Kimi-K2.6 via OpenAI Completions)" /
/// "should handle multi-turn with thinking and tools" at high reasoning
/// effort.
#[tokio::test]
async fn together_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("TOGETHER_API_KEY") else {
        return;
    };
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::multi_turn(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream "Together AI Provider (Kimi-K2.6 via OpenAI Completions)" /
/// "should handle image input".
#[tokio::test]
async fn together_image_input() {
    let Some(api_key) = live::env_key("TOGETHER_API_KEY") else {
        return;
    };
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.BASETEN_API_KEY)` "Baseten Provider
/// (GLM 5.2 via OpenAI Completions)" / "should complete basic text
/// generation" at high reasoning effort, upstream's block-wide `options`.
#[tokio::test]
async fn baseten_basic_text_generation() {
    let Some(api_key) = live::env_key("BASETEN_API_KEY") else {
        return;
    };
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    live::basic_text_generation(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream "Baseten Provider (GLM 5.2 via OpenAI Completions)" / "should
/// handle tool calling" at high reasoning effort.
#[tokio::test]
async fn baseten_tool_calling() {
    let Some(api_key) = live::env_key("BASETEN_API_KEY") else {
        return;
    };
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    live::handle_tool_call(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream "Baseten Provider (GLM 5.2 via OpenAI Completions)" / "should
/// handle streaming" at high reasoning effort.
#[tokio::test]
async fn baseten_streaming() {
    let Some(api_key) = live::env_key("BASETEN_API_KEY") else {
        return;
    };
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    live::handle_streaming(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream "Baseten Provider (GLM 5.2 via OpenAI Completions)" / "should
/// handle thinking mode" at high reasoning effort.
#[tokio::test]
async fn baseten_thinking_mode() {
    let Some(api_key) = live::env_key("BASETEN_API_KEY") else {
        return;
    };
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    live::handle_thinking(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream "Baseten Provider (GLM 5.2 via OpenAI Completions)" / "should
/// handle multi-turn with thinking and tools" at high reasoning effort.
#[tokio::test]
async fn baseten_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("BASETEN_API_KEY") else {
        return;
    };
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    live::multi_turn(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.NVIDIA_API_KEY)` "NVIDIA NIM
/// Provider (Nemotron 3 Super via OpenAI Completions)" / "should complete
/// basic text generation".
#[tokio::test]
async fn nvidia_basic_text_generation() {
    let Some(api_key) = live::env_key("NVIDIA_API_KEY") else {
        return;
    };
    let llm = live::model("nvidia", "nvidia/nemotron-3-super-120b-a12b");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "NVIDIA NIM Provider (Nemotron 3 Super via OpenAI Completions)" /
/// "should handle tool calling".
#[tokio::test]
async fn nvidia_tool_calling() {
    let Some(api_key) = live::env_key("NVIDIA_API_KEY") else {
        return;
    };
    let llm = live::model("nvidia", "nvidia/nemotron-3-super-120b-a12b");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "NVIDIA NIM Provider (Nemotron 3 Super via OpenAI Completions)" /
/// "should handle streaming".
#[tokio::test]
async fn nvidia_streaming() {
    let Some(api_key) = live::env_key("NVIDIA_API_KEY") else {
        return;
    };
    let llm = live::model("nvidia", "nvidia/nemotron-3-super-120b-a12b");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "NVIDIA NIM Provider (Nemotron 3 Super via OpenAI Completions)" /
/// "should handle thinking mode" at high reasoning effort.
#[tokio::test]
async fn nvidia_thinking_mode() {
    let Some(api_key) = live::env_key("NVIDIA_API_KEY") else {
        return;
    };
    let llm = live::model("nvidia", "nvidia/nemotron-3-super-120b-a12b");
    live::handle_thinking(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream "NVIDIA NIM Provider (Nemotron 3 Super via OpenAI Completions)" /
/// "should handle multi-turn with thinking and tools" at high reasoning
/// effort.
#[tokio::test]
async fn nvidia_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("NVIDIA_API_KEY") else {
        return;
    };
    let llm = live::model("nvidia", "nvidia/nemotron-3-super-120b-a12b");
    live::multi_turn(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.OPENROUTER_API_KEY)` "OpenRouter
/// Provider (glm-4.5v via OpenAI Completions)" / "should complete basic text
/// generation".
#[tokio::test]
async fn openrouter_basic_text_generation() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let llm = live::model("openrouter", "z-ai/glm-4.5v");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "OpenRouter Provider (glm-4.5v via OpenAI Completions)" / "should
/// handle tool calling".
#[tokio::test]
async fn openrouter_tool_calling() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let llm = live::model("openrouter", "z-ai/glm-4.5v");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "OpenRouter Provider (glm-4.5v via OpenAI Completions)" / "should
/// handle streaming".
#[tokio::test]
async fn openrouter_streaming() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let llm = live::model("openrouter", "z-ai/glm-4.5v");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "OpenRouter Provider (glm-4.5v via OpenAI Completions)" / "should
/// handle thinking mode" at medium reasoning effort.
#[tokio::test]
async fn openrouter_thinking_mode() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let llm = live::model("openrouter", "z-ai/glm-4.5v");
    live::handle_thinking(&llm, &medium_reasoning_options(api_key)).await;
}

/// Upstream "OpenRouter Provider (glm-4.5v via OpenAI Completions)" / "should
/// handle multi-turn with thinking and tools" at medium reasoning effort,
/// upstream's `{ retry: 2 }`.
#[tokio::test]
async fn openrouter_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let llm = live::model("openrouter", "z-ai/glm-4.5v");
    live::multi_turn(&llm, &medium_reasoning_options(api_key)).await;
}

/// Upstream "OpenRouter Provider (glm-4.5v via OpenAI Completions)" / "should
/// handle image input".
#[tokio::test]
async fn openrouter_image_input() {
    let Some(api_key) = live::env_key("OPENROUTER_API_KEY") else {
        return;
    };
    let llm = live::model("openrouter", "z-ai/glm-4.5v");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.AI_GATEWAY_API_KEY)` "Vercel AI
/// Gateway Provider (google/gemini-2.5-flash via Anthropic Messages)" /
/// "should complete basic text generation".
#[tokio::test]
async fn vercel_ai_gateway_gemini_basic_text_generation() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (google/gemini-2.5-flash via
/// Anthropic Messages)" / "should handle tool calling".
#[tokio::test]
async fn vercel_ai_gateway_gemini_tool_calling() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (google/gemini-2.5-flash via
/// Anthropic Messages)" / "should handle streaming".
#[tokio::test]
async fn vercel_ai_gateway_gemini_streaming() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (google/gemini-2.5-flash via
/// Anthropic Messages)" / "should handle image input".
#[tokio::test]
async fn vercel_ai_gateway_gemini_image_input() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (google/gemini-2.5-flash via
/// Anthropic Messages)" / "should handle multi-turn with tools".
#[tokio::test]
async fn vercel_ai_gateway_gemini_multi_turn_with_tools() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    live::multi_turn(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (anthropic/claude-opus-4.5 via
/// Anthropic Messages)" / "should complete basic text generation".
#[tokio::test]
async fn vercel_ai_gateway_claude_opus_basic_text_generation() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "anthropic/claude-opus-4.5");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (anthropic/claude-opus-4.5 via
/// Anthropic Messages)" / "should handle tool calling".
#[tokio::test]
async fn vercel_ai_gateway_claude_opus_tool_calling() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "anthropic/claude-opus-4.5");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (anthropic/claude-opus-4.5 via
/// Anthropic Messages)" / "should handle streaming".
#[tokio::test]
async fn vercel_ai_gateway_claude_opus_streaming() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "anthropic/claude-opus-4.5");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (anthropic/claude-opus-4.5 via
/// Anthropic Messages)" / "should handle image input".
#[tokio::test]
async fn vercel_ai_gateway_claude_opus_image_input() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "anthropic/claude-opus-4.5");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (anthropic/claude-opus-4.5 via
/// Anthropic Messages)" / "should handle multi-turn with tools".
#[tokio::test]
async fn vercel_ai_gateway_claude_opus_multi_turn_with_tools() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "anthropic/claude-opus-4.5");
    live::multi_turn(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (openai/gpt-5.1-codex-max via
/// Anthropic Messages)" / "should complete basic text generation".
#[tokio::test]
async fn vercel_ai_gateway_codex_max_basic_text_generation() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "openai/gpt-5.1-codex-max");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (openai/gpt-5.1-codex-max via
/// Anthropic Messages)" / "should handle tool calling".
#[tokio::test]
async fn vercel_ai_gateway_codex_max_tool_calling() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "openai/gpt-5.1-codex-max");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (openai/gpt-5.1-codex-max via
/// Anthropic Messages)" / "should handle streaming".
#[tokio::test]
async fn vercel_ai_gateway_codex_max_streaming() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "openai/gpt-5.1-codex-max");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (openai/gpt-5.1-codex-max via
/// Anthropic Messages)" / "should handle image input".
#[tokio::test]
async fn vercel_ai_gateway_codex_max_image_input() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "openai/gpt-5.1-codex-max");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Vercel AI Gateway Provider (openai/gpt-5.1-codex-max via
/// Anthropic Messages)" / "should handle multi-turn with tools".
#[tokio::test]
async fn vercel_ai_gateway_codex_max_multi_turn_with_tools() {
    let Some(api_key) = live::env_key("AI_GATEWAY_API_KEY") else {
        return;
    };
    let llm = live::model("vercel-ai-gateway", "openai/gpt-5.1-codex-max");
    live::multi_turn(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.ZAI_API_KEY)` "zAI Provider
/// (glm-5.2 via OpenAI Completions)" / "should complete basic text
/// generation".
#[tokio::test]
async fn zai_basic_text_generation() {
    let Some(api_key) = live::env_key("ZAI_API_KEY") else {
        return;
    };
    let llm = live::model("zai", "glm-5.2");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "zAI Provider (glm-5.2 via OpenAI Completions)" / "should handle
/// tool calling".
#[tokio::test]
async fn zai_tool_calling() {
    let Some(api_key) = live::env_key("ZAI_API_KEY") else {
        return;
    };
    let llm = live::model("zai", "glm-5.2");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "zAI Provider (glm-5.2 via OpenAI Completions)" / "should handle
/// streaming".
#[tokio::test]
async fn zai_streaming() {
    let Some(api_key) = live::env_key("ZAI_API_KEY") else {
        return;
    };
    let llm = live::model("zai", "glm-5.2");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "zAI Provider (glm-5.2 via OpenAI Completions)" / "should handle
/// thinking mode" at medium reasoning effort.
#[tokio::test]
async fn zai_thinking_mode() {
    let Some(api_key) = live::env_key("ZAI_API_KEY") else {
        return;
    };
    let llm = live::model("zai", "glm-5.2");
    live::handle_thinking(&llm, &medium_reasoning_options(api_key)).await;
}

/// Upstream "zAI Provider (glm-5.2 via OpenAI Completions)" / "should handle
/// multi-turn with thinking and tools" at medium reasoning effort.
#[tokio::test]
async fn zai_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("ZAI_API_KEY") else {
        return;
    };
    let llm = live::model("zai", "glm-5.2");
    live::multi_turn(&llm, &medium_reasoning_options(api_key)).await;
}

/// Upstream "zAI Provider (glm-5.2 via OpenAI Completions)" / "should handle
/// image input".
#[tokio::test]
async fn zai_image_input() {
    let Some(api_key) = live::env_key("ZAI_API_KEY") else {
        return;
    };
    let llm = live::model("zai", "glm-5.2");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.MISTRAL_API_KEY)` "Mistral Provider
/// (devstral-medium-latest)" / "should complete basic text generation".
#[tokio::test]
async fn mistral_devstral_basic_text_generation() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let llm = live::model("mistral", "devstral-medium-latest");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Mistral Provider (devstral-medium-latest)" / "should handle tool
/// calling".
#[tokio::test]
async fn mistral_devstral_tool_calling() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let llm = live::model("mistral", "devstral-medium-latest");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Mistral Provider (devstral-medium-latest)" / "should handle
/// streaming".
#[tokio::test]
async fn mistral_devstral_streaming() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let llm = live::model("mistral", "devstral-medium-latest");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Mistral Provider (devstral-medium-latest)" / "should handle
/// thinking mode": `mistral/mistral-small-2603` at high reasoning effort,
/// upstream's per-it `getModel`.
#[tokio::test]
async fn mistral_small_thinking_mode() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let llm = live::model("mistral", "mistral-small-2603");
    live::handle_thinking(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream "Mistral Provider (devstral-medium-latest)" / "should handle
/// multi-turn with thinking and tools": `mistral/mistral-small-2603` at high
/// reasoning effort.
#[tokio::test]
async fn mistral_small_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let llm = live::model("mistral", "mistral-small-2603");
    live::multi_turn(&llm, &high_reasoning_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.MISTRAL_API_KEY)` "Mistral Provider
/// (pixtral-12b with image support)" / "should complete basic text
/// generation".
#[tokio::test]
async fn mistral_pixtral_basic_text_generation() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let llm = live::model("mistral", "pixtral-12b");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Mistral Provider (pixtral-12b with image support)" / "should
/// handle tool calling".
#[tokio::test]
async fn mistral_pixtral_tool_calling() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let llm = live::model("mistral", "pixtral-12b");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Mistral Provider (pixtral-12b with image support)" / "should
/// handle streaming".
#[tokio::test]
async fn mistral_pixtral_streaming() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let llm = live::model("mistral", "pixtral-12b");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Mistral Provider (pixtral-12b with image support)" / "should
/// handle image input".
#[tokio::test]
async fn mistral_pixtral_image_input() {
    let Some(api_key) = live::env_key("MISTRAL_API_KEY") else {
        return;
    };
    let llm = live::model("mistral", "pixtral-12b");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.MINIMAX_API_KEY)` "MiniMax Provider
/// (MiniMax-M2.7 via Anthropic Messages)" / "should complete basic text
/// generation".
#[tokio::test]
async fn minimax_basic_text_generation() {
    let Some(api_key) = live::env_key("MINIMAX_API_KEY") else {
        return;
    };
    let llm = live::model("minimax", "MiniMax-M2.7");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "MiniMax Provider (MiniMax-M2.7 via Anthropic Messages)" /
/// "should handle tool calling".
#[tokio::test]
async fn minimax_tool_calling() {
    let Some(api_key) = live::env_key("MINIMAX_API_KEY") else {
        return;
    };
    let llm = live::model("minimax", "MiniMax-M2.7");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "MiniMax Provider (MiniMax-M2.7 via Anthropic Messages)" /
/// "should handle streaming".
#[tokio::test]
async fn minimax_streaming() {
    let Some(api_key) = live::env_key("MINIMAX_API_KEY") else {
        return;
    };
    let llm = live::model("minimax", "MiniMax-M2.7");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "MiniMax Provider (MiniMax-M2.7 via Anthropic Messages)" /
/// "should handle thinking mode" with extended thinking at a 2048-token
/// budget.
#[tokio::test]
async fn minimax_thinking_mode() {
    let Some(api_key) = live::env_key("MINIMAX_API_KEY") else {
        return;
    };
    let llm = live::model("minimax", "MiniMax-M2.7");
    live::handle_thinking(&llm, &anthropic_budget_options(api_key)).await;
}

/// Upstream "MiniMax Provider (MiniMax-M2.7 via Anthropic Messages)" /
/// "should handle multi-turn with thinking and tools" with extended thinking
/// at a 2048-token budget.
#[tokio::test]
async fn minimax_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("MINIMAX_API_KEY") else {
        return;
    };
    let llm = live::model("minimax", "MiniMax-M2.7");
    live::multi_turn(&llm, &anthropic_budget_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.KIMI_API_KEY)` "Kimi For Coding
/// Provider (kimi-for-coding via Anthropic Messages)" / "should complete
/// basic text generation".
#[tokio::test]
async fn kimi_for_coding_basic_text_generation() {
    let Some(api_key) = live::env_key("KIMI_API_KEY") else {
        return;
    };
    let llm = live::model("kimi-coding", "kimi-for-coding");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Kimi For Coding Provider (kimi-for-coding via Anthropic
/// Messages)" / "should handle tool calling".
#[tokio::test]
async fn kimi_for_coding_tool_calling() {
    let Some(api_key) = live::env_key("KIMI_API_KEY") else {
        return;
    };
    let llm = live::model("kimi-coding", "kimi-for-coding");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Kimi For Coding Provider (kimi-for-coding via Anthropic
/// Messages)" / "should handle streaming".
#[tokio::test]
async fn kimi_for_coding_streaming() {
    let Some(api_key) = live::env_key("KIMI_API_KEY") else {
        return;
    };
    let llm = live::model("kimi-coding", "kimi-for-coding");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Kimi For Coding Provider (kimi-for-coding via Anthropic
/// Messages)" / "should handle thinking mode" with extended thinking at a
/// 2048-token budget.
#[tokio::test]
async fn kimi_for_coding_thinking_mode() {
    let Some(api_key) = live::env_key("KIMI_API_KEY") else {
        return;
    };
    let llm = live::model("kimi-coding", "kimi-for-coding");
    live::handle_thinking(&llm, &anthropic_budget_options(api_key)).await;
}

/// Upstream "Kimi For Coding Provider (kimi-for-coding via Anthropic
/// Messages)" / "should handle multi-turn with thinking and tools" with
/// extended thinking at a 2048-token budget.
#[tokio::test]
async fn kimi_for_coding_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("KIMI_API_KEY") else {
        return;
    };
    let llm = live::model("kimi-coding", "kimi-for-coding");
    live::multi_turn(&llm, &anthropic_budget_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_API_KEY)` "Xiaomi MiMo (API
/// billing) Provider (Xiaomi MiMo-V2.5-Pro via Anthropic Messages)" /
/// "should complete basic text generation".
#[tokio::test]
async fn xiaomi_basic_text_generation() {
    let Some(api_key) = live::env_key("XIAOMI_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Xiaomi MiMo (API billing) Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages)" / "should handle tool calling".
#[tokio::test]
async fn xiaomi_tool_calling() {
    let Some(api_key) = live::env_key("XIAOMI_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Xiaomi MiMo (API billing) Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages)" / "should handle streaming".
#[tokio::test]
async fn xiaomi_streaming() {
    let Some(api_key) = live::env_key("XIAOMI_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Xiaomi MiMo (API billing) Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages)" / "should handle thinking mode": extended thinking
/// plus high reasoning effort, upstream's `thinkingOptions`.
#[tokio::test]
async fn xiaomi_thinking_mode() {
    let Some(api_key) = live::env_key("XIAOMI_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    live::handle_thinking(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream "Xiaomi MiMo (API billing) Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages)" / "should handle multi-turn with thinking and tools"
/// with extended thinking plus high reasoning effort.
#[tokio::test]
async fn xiaomi_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("XIAOMI_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    live::multi_turn(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_CN_API_KEY)`
/// "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via Anthropic
/// Messages, CN region)" / "should complete basic text generation".
#[tokio::test]
async fn xiaomi_token_plan_cn_basic_text_generation() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages, CN region)" / "should handle tool calling".
#[tokio::test]
async fn xiaomi_token_plan_cn_tool_calling() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages, CN region)" / "should handle streaming".
#[tokio::test]
async fn xiaomi_token_plan_cn_streaming() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages, CN region)" / "should handle thinking mode" with
/// extended thinking plus high reasoning effort.
#[tokio::test]
async fn xiaomi_token_plan_cn_thinking_mode() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    live::handle_thinking(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages, CN region)" / "should handle multi-turn with thinking
/// and tools" with extended thinking plus high reasoning effort.
#[tokio::test]
async fn xiaomi_token_plan_cn_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    live::multi_turn(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_AMS_API_KEY)`
/// "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via Anthropic
/// Messages, AMS region)" / "should complete basic text generation".
#[tokio::test]
async fn xiaomi_token_plan_ams_basic_text_generation() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages, AMS region)" / "should handle tool calling".
#[tokio::test]
async fn xiaomi_token_plan_ams_tool_calling() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages, AMS region)" / "should handle streaming".
#[tokio::test]
async fn xiaomi_token_plan_ams_streaming() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages, AMS region)" / "should handle thinking mode" with
/// extended thinking plus high reasoning effort.
#[tokio::test]
async fn xiaomi_token_plan_ams_thinking_mode() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    live::handle_thinking(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages, AMS region)" / "should handle multi-turn with
/// thinking and tools" with extended thinking plus high reasoning effort.
#[tokio::test]
async fn xiaomi_token_plan_ams_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    live::multi_turn(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_SGP_API_KEY)`
/// "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via Anthropic
/// Messages, SGP region)" / "should complete basic text generation".
#[tokio::test]
async fn xiaomi_token_plan_sgp_basic_text_generation() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages, SGP region)" / "should handle tool calling".
#[tokio::test]
async fn xiaomi_token_plan_sgp_tool_calling() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages, SGP region)" / "should handle streaming".
#[tokio::test]
async fn xiaomi_token_plan_sgp_streaming() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages, SGP region)" / "should handle thinking mode" with
/// extended thinking plus high reasoning effort.
#[tokio::test]
async fn xiaomi_token_plan_sgp_thinking_mode() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    live::handle_thinking(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream "Xiaomi MiMo Token Plan Provider (Xiaomi MiMo-V2.5-Pro via
/// Anthropic Messages, SGP region)" / "should handle multi-turn with
/// thinking and tools" with extended thinking plus high reasoning effort.
#[tokio::test]
async fn xiaomi_token_plan_sgp_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY") else {
        return;
    };
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    live::multi_turn(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)` "Qwen
/// Token Plan Provider (Qwen3.7-Max, international)" / "should complete basic
/// text generation".
#[tokio::test]
async fn qwen_token_plan_basic_text_generation() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Qwen Token Plan Provider (Qwen3.7-Max, international)" /
/// "should handle tool calling".
#[tokio::test]
async fn qwen_token_plan_tool_calling() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Qwen Token Plan Provider (Qwen3.7-Max, international)" /
/// "should handle streaming".
#[tokio::test]
async fn qwen_token_plan_streaming() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Qwen Token Plan Provider (Qwen3.7-Max, international)" /
/// "should handle thinking mode" with extended thinking plus high reasoning
/// effort.
#[tokio::test]
async fn qwen_token_plan_thinking_mode() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    live::handle_thinking(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream "Qwen Token Plan Provider (Qwen3.7-Max, international)" /
/// "should handle multi-turn with thinking and tools" with extended thinking
/// plus high reasoning effort.
#[tokio::test]
async fn qwen_token_plan_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    live::multi_turn(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream "Qwen Token Plan Individual Provider (Qwen3.8-Max,
/// international)" / "should complete basic text generation", gated on the
/// same `QWEN_TOKEN_PLAN_API_KEY` as the shared-plan block.
#[tokio::test]
async fn qwen_token_plan_individual_basic_text_generation() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Qwen Token Plan Individual Provider (Qwen3.8-Max,
/// international)" / "should handle tool calling", gated on the same
/// `QWEN_TOKEN_PLAN_API_KEY` as the shared-plan block.
#[tokio::test]
async fn qwen_token_plan_individual_tool_calling() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Qwen Token Plan Individual Provider (Qwen3.8-Max,
/// international)" / "should handle streaming", gated on the same
/// `QWEN_TOKEN_PLAN_API_KEY` as the shared-plan block.
#[tokio::test]
async fn qwen_token_plan_individual_streaming() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Qwen Token Plan Individual Provider (Qwen3.8-Max,
/// international)" / "should handle thinking mode" with extended thinking
/// plus high reasoning effort.
#[tokio::test]
async fn qwen_token_plan_individual_thinking_mode() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    live::handle_thinking(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream "Qwen Token Plan Individual Provider (Qwen3.8-Max,
/// international)" / "should handle multi-turn with thinking and tools" with
/// extended thinking plus high reasoning effort.
#[tokio::test]
async fn qwen_token_plan_individual_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    live::multi_turn(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_CN_API_KEY)` "Qwen
/// Token Plan Provider (Qwen3.7-Max, CN region)" / "should complete basic
/// text generation".
#[tokio::test]
async fn qwen_token_plan_cn_basic_text_generation() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Qwen Token Plan Provider (Qwen3.7-Max, CN region)" / "should
/// handle tool calling".
#[tokio::test]
async fn qwen_token_plan_cn_tool_calling() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Qwen Token Plan Provider (Qwen3.7-Max, CN region)" / "should
/// handle streaming".
#[tokio::test]
async fn qwen_token_plan_cn_streaming() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Qwen Token Plan Provider (Qwen3.7-Max, CN region)" / "should
/// handle thinking mode" with extended thinking plus high reasoning effort.
#[tokio::test]
async fn qwen_token_plan_cn_thinking_mode() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    live::handle_thinking(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream "Qwen Token Plan Provider (Qwen3.7-Max, CN region)" / "should
/// handle multi-turn with thinking and tools" with extended thinking plus
/// high reasoning effort.
#[tokio::test]
async fn qwen_token_plan_cn_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY") else {
        return;
    };
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    live::multi_turn(&llm, &managed_thinking_options(api_key)).await;
}

/// Upstream `describe.skipIf(!process.env.ANT_LING_API_KEY)` "Ant Ling
/// Provider (Ling 2.6 Flash via OpenAI Completions)" / "should complete basic
/// text generation".
#[tokio::test]
async fn ant_ling_basic_text_generation() {
    let Some(api_key) = live::env_key("ANT_LING_API_KEY") else {
        return;
    };
    let llm = live::model("ant-ling", "Ling-2.6-flash");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Ant Ling Provider (Ling 2.6 Flash via OpenAI Completions)" /
/// "should handle tool calling".
#[tokio::test]
async fn ant_ling_tool_calling() {
    let Some(api_key) = live::env_key("ANT_LING_API_KEY") else {
        return;
    };
    let llm = live::model("ant-ling", "Ling-2.6-flash");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Ant Ling Provider (Ling 2.6 Flash via OpenAI Completions)" /
/// "should handle streaming".
#[tokio::test]
async fn ant_ling_streaming() {
    let Some(api_key) = live::env_key("ANT_LING_API_KEY") else {
        return;
    };
    let llm = live::model("ant-ling", "Ling-2.6-flash");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Ant Ling Provider (Ling 2.6 Flash via OpenAI Completions)" /
/// "should handle thinking mode": `ant-ling/Ring-2.6-1T` at high reasoning
/// effort, upstream's per-it `ringModel`.
#[tokio::test]
async fn ant_ling_ring_thinking_mode() {
    let Some(api_key) = live::env_key("ANT_LING_API_KEY") else {
        return;
    };
    let ring_model = live::model("ant-ling", "Ring-2.6-1T");
    live::handle_thinking(&ring_model, &high_reasoning_options(api_key)).await;
}

/// Upstream `describe("Anthropic OAuth Provider (claude-sonnet-4-6)")` /
/// `it.skipIf(!anthropicOAuthToken)` "should complete basic text
/// generation": the pi credential store resolves the token first, then the
/// provider's env, upstream's `resolveApiKey`.
#[tokio::test]
async fn anthropic_oauth_sonnet_basic_text_generation() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-sonnet-4-6");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Anthropic OAuth Provider (claude-sonnet-4-6)" / "should handle
/// tool calling".
#[tokio::test]
async fn anthropic_oauth_sonnet_tool_calling() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-sonnet-4-6");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Anthropic OAuth Provider (claude-sonnet-4-6)" / "should handle
/// streaming".
#[tokio::test]
async fn anthropic_oauth_sonnet_streaming() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-sonnet-4-6");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Anthropic OAuth Provider (claude-sonnet-4-6)" / "should handle
/// thinking" with extended thinking enabled.
#[tokio::test]
async fn anthropic_oauth_sonnet_thinking() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-sonnet-4-6");
    live::handle_thinking(&llm, &thinking_enabled_options(api_key)).await;
}

/// Upstream "Anthropic OAuth Provider (claude-sonnet-4-6)" / "should handle
/// multi-turn with thinking and tools" with extended thinking enabled.
#[tokio::test]
async fn anthropic_oauth_sonnet_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-sonnet-4-6");
    live::multi_turn(&llm, &thinking_enabled_options(api_key)).await;
}

/// Upstream "Anthropic OAuth Provider (claude-sonnet-4-6)" / "should handle
/// image input".
#[tokio::test]
async fn anthropic_oauth_sonnet_image_input() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-sonnet-4-6");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe("Anthropic OAuth Provider (claude-opus-4-6 with
/// adaptive thinking)")` / `it.skipIf(!anthropicOAuthToken)` "should complete
/// basic text generation".
#[tokio::test]
async fn anthropic_oauth_opus_basic_text_generation() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-opus-4-6");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Anthropic OAuth Provider (claude-opus-4-6 with adaptive
/// thinking)" / "should handle tool calling".
#[tokio::test]
async fn anthropic_oauth_opus_tool_calling() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-opus-4-6");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Anthropic OAuth Provider (claude-opus-4-6 with adaptive
/// thinking)" / "should handle streaming".
#[tokio::test]
async fn anthropic_oauth_opus_streaming() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-opus-4-6");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "Anthropic OAuth Provider (claude-opus-4-6 with adaptive
/// thinking)" / "should handle adaptive thinking with effort high".
#[tokio::test]
async fn anthropic_oauth_opus_adaptive_thinking_effort_high() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-opus-4-6");
    let options = LiveOptions {
        api_key: Some(api_key),
        thinking_enabled: Some(true),
        effort: Some(AnthropicEffort::High),
        ..LiveOptions::default()
    };
    live::handle_thinking(&llm, &options).await;
}

/// Upstream "Anthropic OAuth Provider (claude-opus-4-6 with adaptive
/// thinking)" / "should handle adaptive thinking with effort medium".
#[tokio::test]
async fn anthropic_oauth_opus_adaptive_thinking_effort_medium() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-opus-4-6");
    let options = LiveOptions {
        api_key: Some(api_key),
        thinking_enabled: Some(true),
        effort: Some(AnthropicEffort::Medium),
        ..LiveOptions::default()
    };
    live::handle_thinking(&llm, &options).await;
}

/// Upstream "Anthropic OAuth Provider (claude-opus-4-6 with adaptive
/// thinking)" / "should handle multi-turn with adaptive thinking and tools"
/// at effort high.
#[tokio::test]
async fn anthropic_oauth_opus_multi_turn_with_adaptive_thinking_and_tools() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-opus-4-6");
    let options = LiveOptions {
        api_key: Some(api_key),
        thinking_enabled: Some(true),
        effort: Some(AnthropicEffort::High),
        ..LiveOptions::default()
    };
    live::multi_turn(&llm, &options).await;
}

/// Upstream "Anthropic OAuth Provider (claude-opus-4-6 with adaptive
/// thinking)" / "should handle image input".
#[tokio::test]
async fn anthropic_oauth_opus_image_input() {
    let Some(api_key) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-opus-4-6");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe("GitHub Copilot Provider (gpt-5.3-codex via OpenAI
/// Completions)")` / `it.skipIf(!githubCopilotToken)` "should complete basic
/// text generation": the pi credential store resolves the token first, then
/// the provider's env, upstream's `resolveApiKey`.
#[tokio::test]
async fn github_copilot_codex_basic_text_generation() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "gpt-5.3-codex");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "GitHub Copilot Provider (gpt-5.3-codex via OpenAI Completions)" /
/// "should handle tool calling".
#[tokio::test]
async fn github_copilot_codex_tool_calling() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "gpt-5.3-codex");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "GitHub Copilot Provider (gpt-5.3-codex via OpenAI Completions)" /
/// "should handle streaming".
#[tokio::test]
async fn github_copilot_codex_streaming() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "gpt-5.3-codex");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "GitHub Copilot Provider (gpt-5.3-codex via OpenAI Completions)" /
/// `it.skipIf(!githubCopilotToken)` "should handle thinking":
/// `github-copilot/gpt-5-mini` at high reasoning effort, upstream's per-it
/// `thinkingModel`, `{ retry: 2 }`.
#[tokio::test]
async fn github_copilot_gpt_5_mini_thinking() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let thinking_model = live::model("github-copilot", "gpt-5-mini");
    live::handle_thinking(&thinking_model, &high_reasoning_options(api_key)).await;
}

/// Upstream "GitHub Copilot Provider (gpt-5.3-codex via OpenAI Completions)" /
/// "should handle multi-turn with thinking and tools" on
/// `github-copilot/gpt-5-mini` at high reasoning effort.
#[tokio::test]
async fn github_copilot_gpt_5_mini_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let thinking_model = live::model("github-copilot", "gpt-5-mini");
    live::multi_turn(&thinking_model, &high_reasoning_options(api_key)).await;
}

/// Upstream "GitHub Copilot Provider (gpt-5.3-codex via OpenAI Completions)" /
/// "should handle image input".
#[tokio::test]
async fn github_copilot_codex_image_input() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "gpt-5.3-codex");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe("GitHub Copilot Provider (claude-sonnet-4 via Anthropic
/// Messages)")` / `it.skipIf(!githubCopilotToken)` "should complete basic
/// text generation".
#[tokio::test]
async fn github_copilot_claude_basic_text_generation() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "GitHub Copilot Provider (claude-sonnet-4 via Anthropic
/// Messages)" / "should handle tool calling".
#[tokio::test]
async fn github_copilot_claude_tool_calling() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "GitHub Copilot Provider (claude-sonnet-4 via Anthropic
/// Messages)" / "should handle streaming".
#[tokio::test]
async fn github_copilot_claude_streaming() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "GitHub Copilot Provider (claude-sonnet-4 via Anthropic
/// Messages)" / `it.skipIf(!githubCopilotToken)` "should handle thinking"
/// with extended thinking enabled, `{ retry: 2 }`.
#[tokio::test]
async fn github_copilot_claude_thinking() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    live::handle_thinking(&llm, &thinking_enabled_options(api_key)).await;
}

/// Upstream "GitHub Copilot Provider (claude-sonnet-4 via Anthropic
/// Messages)" / "should handle multi-turn with thinking and tools" with
/// extended thinking enabled.
#[tokio::test]
async fn github_copilot_claude_multi_turn_with_thinking_and_tools() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    live::multi_turn(&llm, &thinking_enabled_options(api_key)).await;
}

/// Upstream "GitHub Copilot Provider (claude-sonnet-4 via Anthropic
/// Messages)" / "should handle image input".
#[tokio::test]
async fn github_copilot_claude_image_input() {
    let Some(api_key) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe("OpenAI Codex Provider (gpt-5.5)")` /
/// `it.skipIf(!openaiCodexToken)` "should complete basic text generation":
/// the pi credential store resolves the token first, then the provider's env,
/// upstream's `resolveApiKey`.
#[tokio::test]
async fn openai_codex_basic_text_generation() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::basic_text_generation(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "OpenAI Codex Provider (gpt-5.5)" / "should handle tool calling".
#[tokio::test]
async fn openai_codex_tool_calling() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::handle_tool_call(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "OpenAI Codex Provider (gpt-5.5)" / "should handle streaming".
#[tokio::test]
async fn openai_codex_streaming() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::handle_streaming(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream "OpenAI Codex Provider (gpt-5.5)" / "should handle thinking with
/// reasoningEffort xhigh".
#[tokio::test]
async fn openai_codex_thinking_xhigh() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::handle_thinking(&llm, &xhigh_reasoning_options(api_key)).await;
}

/// Upstream "OpenAI Codex Provider (gpt-5.5)" / "should handle multi-turn
/// with thinking and tools" at reasoningEffort xhigh.
#[tokio::test]
async fn openai_codex_multi_turn_with_thinking_and_tools_xhigh() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::multi_turn(&llm, &xhigh_reasoning_options(api_key)).await;
}

/// Upstream "OpenAI Codex Provider (gpt-5.5)" / "should handle image input".
#[tokio::test]
async fn openai_codex_image_input() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    live::handle_image(&llm, &LiveOptions::key(api_key)).await;
}

/// Upstream `describe("OpenAI Codex Provider (gpt-5.5 via WebSocket)")` /
/// `it.skipIf(!openaiCodexToken)` "should complete basic text generation"
/// over the WebSocket transport, upstream's `wsOptions`.
#[tokio::test]
async fn openai_codex_websocket_basic_text_generation() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    let options = LiveOptions {
        api_key: Some(api_key),
        transport: Some(Transport::Websocket),
        ..LiveOptions::default()
    };
    live::basic_text_generation(&llm, &options).await;
}

/// Upstream "OpenAI Codex Provider (gpt-5.5 via WebSocket)" / "should handle
/// tool calling" over the WebSocket transport.
#[tokio::test]
async fn openai_codex_websocket_tool_calling() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    let options = LiveOptions {
        api_key: Some(api_key),
        transport: Some(Transport::Websocket),
        ..LiveOptions::default()
    };
    live::handle_tool_call(&llm, &options).await;
}

/// Upstream "OpenAI Codex Provider (gpt-5.5 via WebSocket)" / "should handle
/// streaming" over the WebSocket transport.
#[tokio::test]
async fn openai_codex_websocket_streaming() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    let options = LiveOptions {
        api_key: Some(api_key),
        transport: Some(Transport::Websocket),
        ..LiveOptions::default()
    };
    live::handle_streaming(&llm, &options).await;
}

/// Upstream "OpenAI Codex Provider (gpt-5.5 via WebSocket)" / "should handle
/// thinking with reasoningEffort xhigh" over the WebSocket transport.
#[tokio::test]
async fn openai_codex_websocket_thinking_xhigh() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    let options = LiveOptions {
        api_key: Some(api_key),
        transport: Some(Transport::Websocket),
        reasoning: Some(ThinkingLevel::Xhigh),
        ..LiveOptions::default()
    };
    live::handle_thinking(&llm, &options).await;
}

/// Upstream "OpenAI Codex Provider (gpt-5.5 via WebSocket)" / "should handle
/// multi-turn with thinking and tools" at reasoningEffort xhigh over the
/// WebSocket transport.
#[tokio::test]
async fn openai_codex_websocket_multi_turn_with_thinking_and_tools_xhigh() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    let options = LiveOptions {
        api_key: Some(api_key),
        transport: Some(Transport::Websocket),
        reasoning: Some(ThinkingLevel::Xhigh),
        ..LiveOptions::default()
    };
    live::multi_turn(&llm, &options).await;
}

/// Upstream "OpenAI Codex Provider (gpt-5.5 via WebSocket)" / "should handle
/// image input" over the WebSocket transport.
#[tokio::test]
async fn openai_codex_websocket_image_input() {
    let Some(api_key) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    let options = LiveOptions {
        api_key: Some(api_key),
        transport: Some(Transport::Websocket),
        ..LiveOptions::default()
    };
    live::handle_image(&llm, &options).await;
}

/// Upstream `describe.skipIf(!hasBedrockCredentials())` "Amazon Bedrock
/// Provider (claude-sonnet-4-5)" / "should complete basic text generation":
/// the AWS credentials ride the provider's own resolution, upstream's
/// options-less probe.
#[tokio::test]
async fn bedrock_basic_text_generation() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    live::basic_text_generation(&llm, &LiveOptions::default()).await;
}

/// Upstream "Amazon Bedrock Provider (claude-sonnet-4-5)" / "should handle
/// tool calling".
#[tokio::test]
async fn bedrock_tool_calling() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    live::handle_tool_call(&llm, &LiveOptions::default()).await;
}

/// Upstream "Amazon Bedrock Provider (claude-sonnet-4-5)" / "should handle
/// streaming".
#[tokio::test]
async fn bedrock_streaming() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    live::handle_streaming(&llm, &LiveOptions::default()).await;
}

/// Upstream "Amazon Bedrock Provider (claude-sonnet-4-5)" / "should handle
/// thinking" at Bedrock's `reasoning: "medium"`.
#[tokio::test]
async fn bedrock_thinking() {
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
    live::handle_thinking(&llm, &options).await;
}

/// Upstream "Amazon Bedrock Provider (claude-sonnet-4-5)" / "should handle
/// multi-turn with thinking and tools" at Bedrock's `reasoning: "high"`.
#[tokio::test]
async fn bedrock_multi_turn_with_thinking_and_tools() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    let options = LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..LiveOptions::default()
    };
    live::multi_turn(&llm, &options).await;
}

/// Upstream "Amazon Bedrock Provider (claude-sonnet-4-5)" / "should handle
/// image input".
#[tokio::test]
async fn bedrock_image_input() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    live::handle_image(&llm, &LiveOptions::default()).await;
}

/// The request-payload hook the Bedrock payload probes share: stores the
/// payload the adapter is about to send and keeps it unchanged, upstream's
/// `onPayload: (payload) => { capturedPayload = payload; }`.
fn payload_capture_hook(captured: Arc<Mutex<Option<Value>>>) -> OnPayload {
    OnPayload::new(move |payload, _model| {
        *captured.lock().unwrap_or_else(PoisonError::into_inner) = Some(payload.clone());
        Box::pin(async move { Some(payload) })
    })
}

/// Upstream `describe.skipIf(!hasBedrockCredentials())` "Amazon Bedrock
/// Provider (claude-opus-4-6 interleaved thinking)" / "should use adaptive
/// thinking without `anthropic_beta`": the tool-enabled request carries
/// adaptive thinking and the max-effort output config in
/// `additionalModelRequestFields`, with no `anthropic_beta` entry.
#[tokio::test]
async fn bedrock_uses_adaptive_thinking_without_anthropic_beta() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1");
    let captured = Arc::new(Mutex::new(None));
    let context = Context {
        system_prompt: Some("You are a helpful assistant that uses tools when asked.".to_owned()),
        messages: vec![live::user_message(
            "Think first, then calculate 15 + 27 using the math_operation tool.",
        )],
        tools: Some(vec![live::calculator_tool()]),
    };
    let options = LiveOptions {
        reasoning: Some(ThinkingLevel::Xhigh),
        interleaved_thinking: Some(true),
        on_payload: Some(payload_capture_hook(Arc::clone(&captured))),
        ..LiveOptions::default()
    };

    let response = live::complete(&llm, &context, &options).await;

    assert_ne!(
        response.stop_reason,
        StopReason::Error,
        "error: {:?}",
        response.error_message
    );
    let payload = captured
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    assert!(payload.is_some(), "the payload hook captured the request");
    let payload = payload.unwrap_or(Value::Null);
    assert_eq!(
        payload["additionalModelRequestFields"]["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(
        payload["additionalModelRequestFields"]["output_config"],
        json!({ "effort": "max" })
    );
    assert!(payload["additionalModelRequestFields"]["anthropic_beta"].is_null());
}

/// Upstream "Amazon Bedrock Provider (claude-opus-4-6 interleaved thinking)" /
/// "should pass requestMetadata to the SDK payload": the metadata object
/// reaches the SDK payload verbatim.
#[tokio::test]
async fn bedrock_request_metadata_rides_the_payload() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    let captured = Arc::new(Mutex::new(None));
    let context = Context {
        system_prompt: None,
        messages: vec![live::user_message("Say hi.")],
        tools: None,
    };
    let metadata = BTreeMap::from([
        ("app".to_owned(), json!("pi-test")),
        ("env".to_owned(), json!("ci")),
    ]);
    let options = LiveOptions {
        request_metadata: Some(metadata),
        on_payload: Some(payload_capture_hook(Arc::clone(&captured))),
        ..LiveOptions::default()
    };

    let response = live::complete(&llm, &context, &options).await;

    assert_ne!(
        response.stop_reason,
        StopReason::Error,
        "error: {:?}",
        response.error_message
    );
    let payload = captured
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    assert!(payload.is_some(), "the payload hook captured the request");
    let payload = payload.unwrap_or(Value::Null);
    assert_eq!(
        payload["requestMetadata"],
        json!({ "app": "pi-test", "env": "ci" })
    );
}

/// Upstream "Amazon Bedrock Provider (claude-opus-4-6 interleaved thinking)" /
/// "should omit requestMetadata from payload when not provided".
#[tokio::test]
async fn bedrock_omits_request_metadata_when_not_provided() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    let captured = Arc::new(Mutex::new(None));
    let context = Context {
        system_prompt: None,
        messages: vec![live::user_message("Say hi.")],
        tools: None,
    };
    let options = LiveOptions {
        on_payload: Some(payload_capture_hook(Arc::clone(&captured))),
        ..LiveOptions::default()
    };

    let response = live::complete(&llm, &context, &options).await;

    assert_ne!(
        response.stop_reason,
        StopReason::Error,
        "error: {:?}",
        response.error_message
    );
    let payload = captured
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    assert!(payload.is_some(), "the payload hook captured the request");
    let payload = payload.unwrap_or(Value::Null);
    assert!(payload.get("requestMetadata").is_none());
}

/// The hand-built Ollama model, upstream's `llm` literal in `beforeAll`.
fn ollama_model() -> Model {
    Model {
        id: "gpt-oss:20b".to_owned(),
        name: "Ollama GPT-OSS 20B".to_owned(),
        api: Api::from("openai-completions"),
        provider: ProviderId::from("ollama"),
        base_url: "http://localhost:11434/v1".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 16_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The spawned `ollama serve` child with its output discarded, upstream's
/// `spawn("ollama", ["serve"], { stdio: "ignore" })`.
fn spawn_ollama_serve() -> Option<std::process::Child> {
    std::process::Command::new("ollama")
        .arg("serve")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok()
}

/// Whether the local server answers on its port, upstream's readiness poll
/// over `/api/tags` with the initial 1s delay; bails after ~25s, upstream's
/// 30s `beforeAll` timeout.
async fn ollama_ready() -> bool {
    tokio::time::sleep(Duration::from_millis(1000)).await;
    for _ in 0..48 {
        if tokio::net::TcpStream::connect("127.0.0.1:11434")
            .await
            .is_ok()
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

/// Upstream `describe.skipIf(!ollamaInstalled)` "Ollama Provider (gpt-oss-20b
/// via OpenAI Completions)": upstream folds the block's five `it`s behind one
/// `beforeAll` (pull `gpt-oss:20b` when missing, serve, readiness poll) and
/// one `afterAll` (kill the server); the port folds that setup and teardown
/// around the five helper calls in this single test.
#[tokio::test]
async fn ollama_probes_run_the_five_helpers() {
    if live::env_key("PI_NO_LOCAL_LLM").is_some() {
        return;
    }
    if !std::process::Command::new("ollama")
        .arg("--version")
        .status()
        .is_ok_and(|status| status.success())
    {
        return;
    }
    let listed = std::process::Command::new("ollama")
        .arg("list")
        .output()
        .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).contains("gpt-oss:20b"));
    if !listed {
        println!("Pulling gpt-oss:20b model for Ollama tests...");
        if !std::process::Command::new("ollama")
            .args(["pull", "gpt-oss:20b"])
            .status()
            .is_ok_and(|status| status.success())
        {
            println!("Failed to pull gpt-oss:20b model, tests will be skipped");
            return;
        }
    }
    let Some(mut server) = spawn_ollama_serve() else {
        return;
    };
    if !ollama_ready().await {
        let _ = server.kill();
        println!("ollama serve never opened its port; skipping the Ollama probes");
        return;
    }

    let llm = ollama_model();
    let options = LiveOptions::key("test");
    let thinking_options = LiveOptions {
        reasoning: Some(ThinkingLevel::Medium),
        ..LiveOptions::key("test")
    };
    live::basic_text_generation(&llm, &options).await;
    live::handle_tool_call(&llm, &options).await;
    live::handle_streaming(&llm, &options).await;
    live::handle_thinking(&llm, &thinking_options).await;
    live::multi_turn(&llm, &thinking_options).await;

    let _ = server.kill();
}

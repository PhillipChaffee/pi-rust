//! Tool call without result tests, ported from `packages/ai/test/
//! tool-call-without-result.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: an assistant tool call with no
//! corresponding tool result — the aborted/cancelled-tool-call shape — must
//! be filtered out of the follow-up request instead of erroring, across
//! every provider the suite covers.
//!
//! Like upstream's `it.skipIf`, a probe without its provider credential
//! returns early: the env-var gates read the process environment exactly as
//! upstream's `skipIf(!process.env.X)`, the Azure/Bedrock/Cloudflare gates
//! ride the multi-var guards, and the OAuth-backed probes resolve through
//! [`common::live::resolve_api_key`] — the pi credential store first, then
//! the provider's env vars.

#![expect(
    clippy::expect_used,
    reason = "the tests pin live outcomes; an unexpected shape panics the test by design"
)]
#![expect(
    clippy::print_stdout,
    reason = "the probe logs the settled responses like upstream's console.log"
)]

use pi_ai::types::{AssistantBlock, Context, Message, Model, StopReason, ThinkingLevel, Tool};

mod common;
use common::live;

/// The calculate tool the probes ask for, upstream's `calculateTool` over
/// `Type.Object({ expression: Type.String(...) })`.
fn calculate_tool() -> Tool {
    Tool {
        name: "calculate".to_owned(),
        description: "Evaluate mathematical expressions".to_owned(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string", "description": "The mathematical expression to evaluate" }
            },
            "required": ["expression"]
        }),
        constrained_sampling: None,
    }
}

/// The orphaned-tool-call probe upstream's `testToolCallWithoutResult`: the
/// first turn must produce a tool call, the second — a user message with no
/// tool result supplied — must succeed, with either text or a fresh tool
/// call, and stop cleanly.
async fn test_tool_call_without_result(model: &Model, options: &live::LiveOptions) {
    // Step 1: create the context with the calculate tool, then ask the model
    // to make a tool call.
    let mut context = Context {
        system_prompt: Some(
            "You are a helpful assistant. Use the calculate tool when asked to perform \
             calculations."
                .to_owned(),
        ),
        messages: vec![live::user_message(
            "Please calculate 25 * 18 using the calculate tool.",
        )],
        tools: Some(vec![calculate_tool()]),
    };

    // Step 2: the assistant's response should contain a tool call.
    let first_response = live::complete(model, &context, options).await;
    println!(
        "First response: {}",
        serde_json::to_string_pretty(&first_response).expect("the first response serializes")
    );
    let has_tool_call = first_response
        .content
        .iter()
        .any(|block| matches!(block, AssistantBlock::ToolCall(_)));
    assert!(
        has_tool_call,
        "Expected assistant to make a tool call, but none was found"
    );
    context.messages.push(Message::Assistant(first_response));

    // Step 3: send a user message WITHOUT providing the tool result — the
    // aborted/cancelled-tool-call scenario.
    context
        .messages
        .push(live::user_message("Never mind, just tell me what is 2+2?"));

    // Step 4: the orphaned tool call must be filtered out, and the request
    // must succeed.
    let second_response = live::complete(model, &context, options).await;
    println!(
        "Second response: {}",
        serde_json::to_string_pretty(&second_response).expect("the second response serializes")
    );

    assert_ne!(
        second_response.stop_reason,
        StopReason::Error,
        "error: {:?}",
        second_response.error_message
    );
    assert!(
        !second_response.content.is_empty(),
        "the response carries content"
    );

    // The model may answer directly or make a new tool call — either is fine;
    // the important thing is it didn't fail with the orphaned tool call error.
    let text_content = second_response
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ");
    let tool_calls = second_response
        .content
        .iter()
        .filter(|block| matches!(block, AssistantBlock::ToolCall(_)))
        .count();
    assert!(
        tool_calls > 0 || !text_content.is_empty(),
        "tool calls or text content present"
    );
    println!("Answer: {text_content}");

    // The turn must stop cleanly or open a new tool call.
    assert!(
        matches!(
            second_response.stop_reason,
            StopReason::Stop | StopReason::ToolUse
        ),
        "stop reason is stop or toolUse: {:?}",
        second_response.stop_reason
    );
}

/// Upstream `describe.skipIf(!process.env.GEMINI_API_KEY)` "Google Provider" /
/// `it` "should filter out tool calls without corresponding tool results".
#[tokio::test]
async fn google_provider_should_filter_out_tool_calls_without_corresponding_tool_results() {
    if live::env_key("GEMINI_API_KEY").is_none() {
        return;
    }
    let model = live::model("google", "gemini-2.5-flash");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Completions
/// Provider" / `it` "should filter out tool calls without corresponding tool
/// results": the catalog model retargeted at the openai-completions wire,
/// upstream's `{ ...getModel(...), api }` spread.
#[tokio::test]
async fn openai_completions_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let model = Model {
        api: pi_ai::types::Api::from("openai-completions"),
        ..live::model("openai", "gpt-4o-mini")
    };
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.OPENAI_API_KEY)` "OpenAI Responses
/// Provider" / `it` "should filter out tool calls without corresponding tool
/// results".
#[tokio::test]
async fn openai_responses_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
{
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let model = live::model("openai", "gpt-5-mini");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasAzureOpenAICredentials())` "Azure OpenAI
/// Responses Provider" / `it` "should filter out tool calls without
/// corresponding tool results", with the deployment-name map override.
#[tokio::test]
async fn azure_openai_responses_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    if !live::has_azure_openai_credentials() {
        return;
    }
    let model = live::model("azure-openai-responses", "gpt-4o-mini");
    let options = live::LiveOptions {
        azure_deployment_name: live::azure_deployment_name(&model.id),
        ..live::LiveOptions::default()
    };
    test_tool_call_without_result(&model, &options).await;
}

/// Upstream `describe.skipIf(!process.env.ANTHROPIC_API_KEY)` "Anthropic
/// Provider" / `it` "should filter out tool calls without corresponding tool
/// results".
#[tokio::test]
async fn anthropic_provider_should_filter_out_tool_calls_without_corresponding_tool_results() {
    if live::env_key("ANTHROPIC_API_KEY").is_none() {
        return;
    }
    let model = live::model("anthropic", "claude-haiku-4-5");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XAI_API_KEY)` "xAI Provider" / `it`
/// "should filter out tool calls without corresponding tool results".
#[tokio::test]
async fn xai_provider_should_filter_out_tool_calls_without_corresponding_tool_results() {
    if live::env_key("XAI_API_KEY").is_none() {
        return;
    }
    let model = live::model("xai", "grok-4.3");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.GROQ_API_KEY)` "Groq Provider" /
/// `it` "should filter out tool calls without corresponding tool results".
#[tokio::test]
async fn groq_provider_should_filter_out_tool_calls_without_corresponding_tool_results() {
    if live::env_key("GROQ_API_KEY").is_none() {
        return;
    }
    let model = live::model("groq", "openai/gpt-oss-20b");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.CEREBRAS_API_KEY)` "Cerebras
/// Provider" / `it` "should filter out tool calls without corresponding tool
/// results".
#[tokio::test]
async fn cerebras_provider_should_filter_out_tool_calls_without_corresponding_tool_results() {
    if live::env_key("CEREBRAS_API_KEY").is_none() {
        return;
    }
    let model = live::model("cerebras", "gpt-oss-120b");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasCloudflareWorkersAICredentials())` "Cloudflare
/// Workers AI Provider" / `it` "should filter out tool calls without
/// corresponding tool results".
#[tokio::test]
async fn cloudflare_workers_ai_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let model = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasCloudflareAiGatewayCredentials())` "Cloudflare
/// AI Gateway Provider" / `it` "should filter out tool calls without
/// corresponding tool results".
#[tokio::test]
async fn cloudflare_ai_gateway_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let model = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.HF_TOKEN)` "Hugging Face Provider" /
/// `it` "should filter out tool calls without corresponding tool results".
#[tokio::test]
async fn hugging_face_provider_should_filter_out_tool_calls_without_corresponding_tool_results() {
    if live::env_key("HF_TOKEN").is_none() {
        return;
    }
    let model = live::model("huggingface", "moonshotai/Kimi-K2.5");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.TOGETHER_API_KEY)` "Together AI
/// Provider" / `it` "should filter out tool calls without corresponding tool
/// results", with the high reasoning effort.
#[tokio::test]
async fn together_ai_provider_should_filter_out_tool_calls_without_corresponding_tool_results() {
    if live::env_key("TOGETHER_API_KEY").is_none() {
        return;
    }
    let model = live::model("together", "moonshotai/Kimi-K2.6");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    test_tool_call_without_result(&model, &options).await;
}

/// Upstream `describe.skipIf(!process.env.BASETEN_API_KEY)` "Baseten Provider"
/// / `it` "should filter out tool calls without corresponding tool results",
/// with the high reasoning effort.
#[tokio::test]
async fn baseten_provider_should_filter_out_tool_calls_without_corresponding_tool_results() {
    if live::env_key("BASETEN_API_KEY").is_none() {
        return;
    }
    let model = live::model("baseten", "zai-org/GLM-5.2");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    test_tool_call_without_result(&model, &options).await;
}

/// Upstream `describe.skipIf(!process.env.ZAI_API_KEY)` "zAI Provider" / `it`
/// "should filter out tool calls without corresponding tool results".
#[tokio::test]
async fn zai_provider_should_filter_out_tool_calls_without_corresponding_tool_results() {
    if live::env_key("ZAI_API_KEY").is_none() {
        return;
    }
    let model = live::model("zai", "glm-5.2");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.MISTRAL_API_KEY)` "Mistral Provider"
/// / `it` "should filter out tool calls without corresponding tool results".
#[tokio::test]
async fn mistral_provider_should_filter_out_tool_calls_without_corresponding_tool_results() {
    if live::env_key("MISTRAL_API_KEY").is_none() {
        return;
    }
    let model = live::model("mistral", "devstral-medium-latest");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.MINIMAX_API_KEY)` "MiniMax Provider"
/// / `it` "should filter out tool calls without corresponding tool results".
#[tokio::test]
async fn minimax_provider_should_filter_out_tool_calls_without_corresponding_tool_results() {
    if live::env_key("MINIMAX_API_KEY").is_none() {
        return;
    }
    let model = live::model("minimax", "MiniMax-M2.7");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_API_KEY)` "Xiaomi MiMo (API
/// billing) Provider" / `it` "should filter out tool calls without
/// corresponding tool results".
#[tokio::test]
async fn xiaomi_mimo_api_billing_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    if live::env_key("XIAOMI_API_KEY").is_none() {
        return;
    }
    let model = live::model("xiaomi", "mimo-v2.5-pro");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_CN_API_KEY)`
/// "Xiaomi MiMo Token Plan (CN) Provider" / `it` "should filter out tool
/// calls without corresponding tool results".
#[tokio::test]
async fn xiaomi_mimo_token_plan_cn_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    if live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY").is_none() {
        return;
    }
    let model = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_AMS_API_KEY)`
/// "Xiaomi MiMo Token Plan (AMS) Provider" / `it` "should filter out tool
/// calls without corresponding tool results".
#[tokio::test]
async fn xiaomi_mimo_token_plan_ams_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    if live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY").is_none() {
        return;
    }
    let model = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.XIAOMI_TOKEN_PLAN_SGP_API_KEY)`
/// "Xiaomi MiMo Token Plan (SGP) Provider" / `it` "should filter out tool
/// calls without corresponding tool results".
#[tokio::test]
async fn xiaomi_mimo_token_plan_sgp_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    if live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY").is_none() {
        return;
    }
    let model = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)` "Qwen
/// Token Plan Provider" / `it` "should filter out tool calls without
/// corresponding tool results".
#[tokio::test]
async fn qwen_token_plan_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
{
    if live::env_key("QWEN_TOKEN_PLAN_API_KEY").is_none() {
        return;
    }
    let model = live::model("qwen-token-plan", "qwen3.7-max");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_API_KEY)` "Qwen
/// Token Plan Individual Provider" / `it` "should filter out tool calls
/// without corresponding tool results".
#[tokio::test]
async fn qwen_token_plan_individual_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    if live::env_key("QWEN_TOKEN_PLAN_API_KEY").is_none() {
        return;
    }
    let model = live::model("qwen-token-plan-individual", "qwen3.8-max");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.QWEN_TOKEN_PLAN_CN_API_KEY)` "Qwen
/// Token Plan (CN) Provider" / `it` "should filter out tool calls without
/// corresponding tool results".
#[tokio::test]
async fn qwen_token_plan_cn_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    if live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY").is_none() {
        return;
    }
    let model = live::model("qwen-token-plan-cn", "qwen3.7-max");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.KIMI_API_KEY)` "Kimi For Coding
/// Provider" / `it` "should filter out tool calls without corresponding tool
/// results".
#[tokio::test]
async fn kimi_for_coding_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
{
    if live::env_key("KIMI_API_KEY").is_none() {
        return;
    }
    let model = live::model("kimi-coding", "kimi-for-coding");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!process.env.AI_GATEWAY_API_KEY)` "Vercel AI
/// Gateway Provider" / `it` "should filter out tool calls without
/// corresponding tool results".
#[tokio::test]
async fn vercel_ai_gateway_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    if live::env_key("AI_GATEWAY_API_KEY").is_none() {
        return;
    }
    let model = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe.skipIf(!hasBedrockCredentials())` "Amazon Bedrock
/// Provider" / `it` "should filter out tool calls without corresponding tool
/// results".
#[tokio::test]
async fn amazon_bedrock_provider_should_filter_out_tool_calls_without_corresponding_tool_results() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let model = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    test_tool_call_without_result(&model, &live::LiveOptions::default()).await;
}

/// Upstream `describe` "Anthropic OAuth Provider" /
/// `it.skipIf(!anthropicOAuthToken)` "should filter out tool calls without
/// corresponding tool results".
#[tokio::test]
async fn anthropic_oauth_provider_should_filter_out_tool_calls_without_corresponding_tool_results()
{
    let Some(anthropic_oauth_token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let model = live::model("anthropic", "claude-haiku-4-5");
    test_tool_call_without_result(&model, &live::LiveOptions::key(anthropic_oauth_token)).await;
}

/// Upstream `describe` "GitHub Copilot Provider" /
/// `it.skipIf(!githubCopilotToken)` "claude-haiku-4.5 - should filter out
/// tool calls without corresponding tool results".
#[tokio::test]
async fn github_copilot_claude_haiku_4_5_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    let Some(github_copilot_token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let model = live::model("github-copilot", "claude-haiku-4.5");
    test_tool_call_without_result(&model, &live::LiveOptions::key(github_copilot_token)).await;
}

/// Upstream `describe` "GitHub Copilot Provider" /
/// `it.skipIf(!githubCopilotToken)` "claude-sonnet-4 - should filter out tool
/// calls without corresponding tool results".
#[tokio::test]
async fn github_copilot_claude_sonnet_4_should_filter_out_tool_calls_without_corresponding_tool_results()
 {
    let Some(github_copilot_token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let model = live::model("github-copilot", "claude-sonnet-4.6");
    test_tool_call_without_result(&model, &live::LiveOptions::key(github_copilot_token)).await;
}

/// Upstream `describe` "OpenAI Codex Provider" /
/// `it.skipIf(!openaiCodexToken)` "gpt-5.5 - should filter out tool calls
/// without corresponding tool results".
#[tokio::test]
async fn openai_codex_gpt_5_5_should_filter_out_tool_calls_without_corresponding_tool_results() {
    let Some(openai_codex_token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let model = live::model("openai-codex", "gpt-5.5");
    test_tool_call_without_result(&model, &live::LiveOptions::key(openai_codex_token)).await;
}

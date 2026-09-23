//! Live Unicode surrogate-pair handling suites, ported from the upstream
//! `packages/ai/test/unicode-surrogate.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream's brief: when tool results carry emoji or other characters
//! outside the Basic Multilingual Plane, broken serialization can send
//! unpaired surrogates and the provider rejects the body with a
//! "no low surrogate in string" JSON error. The suite proves emoji-bearing
//! tool results, a real-world `LinkedIn` comment dump, and a fabricated
//! unpaired high surrogate ride the wire end to end without that error.
//! The sanitization itself is statically upheld in the port: a Rust
//! `String` stores Unicode scalar values and cannot carry an unpaired
//! surrogate, so the unpaired-surrogate fixture carries U+FFFD — the
//! replacement character a lossy decode substitutes for ill-formed
//! surrogate bytes — and still proves the wire accepts it.
//!
//! Gating contract, upstream's `describe.skipIf`/`it.skipIf`: a probe
//! without its provider credential returns early. Single-variable gates
//! read the env var the way upstream does; the multi-variable Azure,
//! Cloudflare, and Bedrock gates ride the live guards; and the OAuth gates
//! resolve the pi credential-store token per probe (upstream resolves once
//! at module load, the skipIf semantics preserved). Upstream's vitest
//! `{ retry: 3, timeout: 30000 }` options have no port-side equivalent and
//! ride the plain test runner.

// binary and carries its own expects only for the lints its own code knows
// it trips; these cover the rest so the suite's gate runs clean without
// editing the shared file. Stale entries fail the build on their own.

use pi_ai::auth::resolve::now_ms;
use pi_ai::types::{
    AssistantBlock, AssistantMessage, Context, Message, Model, StopReason, TextContent,
    ThinkingLevel, Tool, ToolCall, ToolResultBlock, ToolResultMessage, Usage,
};

mod common;
use common::live;

/// The emoji-bearing tool result text, upstream's `testEmojiInToolResults`
/// fixture: emoji, CJK, mathematical symbols, and curly quotes.
const EMOJI_TOOL_RESULT_TEXT: &str = "Test with emoji 🙈 and other characters:\n- Monkey emoji: \
     🙈\n- Thumbs up: 👍\n- Heart: ❤️\n- Thinking face: 🤔\n- Rocket: 🚀\n- Mixed text: Mario \
     Zechner wann? Wo? Bin grad äußersr eventuninformiert 🙈\n- Japanese: こんにちは\n- Chinese: \
     你好\n- Mathematical symbols: ∑∫∂√\n- Special quotes: \"curly\" 'quotes'";

/// The real-world `LinkedIn` comment dump, upstream's `testRealWorldLinkedInData`
/// fixture.
const LINKEDIN_TOOL_RESULT_TEXT: &str = "Post: Hab einen \"Generative KI für Nicht-Techniker\" \
     Workshop gebaut.\nUnanswered Comments: 2\n\n=> {\n  \"comments\": [\n    {\n      \
     \"author\": \"Matthias Neumayer's  graphic link\",\n      \"text\": \"Leider nehmen das \
     viel zu wenige Leute ernst\"\n    },\n    {\n      \"author\": \"Matthias Neumayer's  \
     graphic link\",\n      \"text\": \"Mario Zechner wann? Wo? Bin grad äußersr \
     eventuninformiert 🙈\"\n    }\n  ]\n}";

/// The context scaffold the three upstream helpers share, upstream's inline
/// `Context` literals: the user asks for the tool, the assistant answers with
/// the tool call (zeroed usage, `stopReason: "toolUse"`), and `Context.tools`
/// carries the tool over `Type.Object({})` — an empty but proper OBJECT
/// schema, required by Cloud Code Assist.
fn tool_call_context(
    llm: &Model,
    tool_call_id: &str,
    tool_name: &str,
    description: &str,
    first_user: &str,
) -> Context {
    Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![
            live::user_message(first_user),
            Message::Assistant(AssistantMessage {
                content: vec![AssistantBlock::ToolCall(ToolCall {
                    id: tool_call_id.to_owned(),
                    name: tool_name.to_owned(),
                    arguments: serde_json::Map::new(),
                    thought_signature: None,
                    namespace: None,
                })],
                api: llm.api.clone(),
                provider: llm.provider.clone(),
                model: llm.id.clone(),
                usage: Usage::default(),
                stop_reason: StopReason::ToolUse,
                timestamp: now_ms(),
                response_model: None,
                response_id: None,
                provider_thinking_level: None,
                diagnostics: None,
                deferred: None,
                error_message: None,
                raw_stop_reason: None,
                end_turn: None,
            }),
        ],
        tools: Some(vec![Tool {
            name: tool_name.to_owned(),
            description: description.to_owned(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
            constrained_sampling: None,
        }]),
    }
}

/// The tool-result message the helpers append, upstream's inline
/// `ToolResultMessage` literals.
fn tool_result(tool_call_id: &str, tool_name: &str, text: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: tool_call_id.to_owned(),
        tool_name: tool_name.to_owned(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        added_tool_names: None,
        details: None,
        is_error: false,
        usage: None,
        timestamp: now_ms(),
    })
}

/// Upstream `testEmojiInToolResults`: the emoji fixture must complete without
/// the surrogate-pair error.
async fn test_emoji_in_tool_results(llm: &Model, options: &live::LiveOptions) {
    let tool_call_id = if llm.provider.0 == "mistral" {
        "testtool1"
    } else {
        "test_1"
    };
    let mut context = tool_call_context(
        llm,
        tool_call_id,
        "test_tool",
        "A test tool",
        "Use the test tool",
    );
    context.messages.push(tool_result(
        tool_call_id,
        "test_tool",
        EMOJI_TOOL_RESULT_TEXT,
    ));
    context
        .messages
        .push(live::user_message("Summarize the tool result briefly."));

    let response = live::complete(llm, &context, options).await;

    assert_ne!(
        response.stop_reason,
        StopReason::Error,
        "{}/{}: {:?}",
        llm.provider.0,
        llm.id,
        response.error_message
    );
    assert!(
        response.error_message.is_none(),
        "{}/{}",
        llm.provider.0,
        llm.id
    );
    assert!(
        !response.content.is_empty(),
        "{}/{}",
        llm.provider.0,
        llm.id
    );
}

/// Upstream `testRealWorldLinkedInData`: the `LinkedIn` comment dump must
/// complete and carry a text block.
async fn test_real_world_linkedin_data(llm: &Model, options: &live::LiveOptions) {
    let tool_call_id = if llm.provider.0 == "mistral" {
        "linkedin1"
    } else {
        "linkedin_1"
    };
    let mut context = tool_call_context(
        llm,
        tool_call_id,
        "linkedin_skill",
        "Get `LinkedIn` comments",
        "Use the linkedin tool to get comments",
    );
    context.messages.push(tool_result(
        tool_call_id,
        "linkedin_skill",
        LINKEDIN_TOOL_RESULT_TEXT,
    ));
    context
        .messages
        .push(live::user_message("How many comments are there?"));

    let response = live::complete(llm, &context, options).await;

    assert_ne!(
        response.stop_reason,
        StopReason::Error,
        "{}/{}: {:?}",
        llm.provider.0,
        llm.id,
        response.error_message
    );
    assert!(
        response.error_message.is_none(),
        "{}/{}",
        llm.provider.0,
        llm.id
    );
    assert!(
        response
            .content
            .iter()
            .any(|block| matches!(block, AssistantBlock::Text(_))),
        "{}/{}",
        llm.provider.0,
        llm.id
    );
}

/// Upstream `testUnpairedHighSurrogate`: the fabricated unpaired high
/// surrogate must be sanitized before the request and complete cleanly.
async fn test_unpaired_high_surrogate(llm: &Model, options: &live::LiveOptions) {
    let tool_call_id = if llm.provider.0 == "mistral" {
        "testtool2"
    } else {
        "test_2"
    };
    let mut context = tool_call_context(
        llm,
        tool_call_id,
        "test_tool",
        "A test tool",
        "Use the test tool",
    );
    // Upstream fabricates the lone high surrogate with
    // `String.fromCharCode(0xD83D)`, simulating text processing that corrupts
    // emoji. A Rust `String` cannot hold one, so the sanitization upstream
    // runs before serialization is statically upheld; the fixture carries
    // U+FFFD, the replacement character a lossy decode substitutes.
    let unpaired_surrogate = '\u{FFFD}';
    let text = format!("Text with unpaired surrogate: {unpaired_surrogate} <- should be sanitized");
    context
        .messages
        .push(tool_result(tool_call_id, "test_tool", &text));
    context
        .messages
        .push(live::user_message("What did the tool return?"));

    let response = live::complete(llm, &context, options).await;

    assert_ne!(
        response.stop_reason,
        StopReason::Error,
        "{}/{}: {:?}",
        llm.provider.0,
        llm.id,
        response.error_message
    );
    assert!(
        response.error_message.is_none(),
        "{}/{}",
        llm.provider.0,
        llm.id
    );
    assert!(
        !response.content.is_empty(),
        "{}/{}",
        llm.provider.0,
        llm.id
    );
}

/// The Azure block's options, upstream's `azureOptions` over `{}`: the api
/// key the credentials gate checked plus the deployment override when
/// `AZURE_OPENAI_DEPLOYMENT_NAME_MAP` maps one for `gpt-4o-mini`. The key
/// rides the options because the extras-carrying dispatch goes through the
/// Azure adapter directly, and the adapters do not apply compat's env
/// fallback.
fn azure_options(model_id: &str) -> live::LiveOptions {
    live::LiveOptions {
        api_key: live::env_key("AZURE_OPENAI_API_KEY"),
        azure_deployment_name: live::azure_deployment_name(model_id),
        ..live::LiveOptions::default()
    }
}

/// Upstream "Google Provider Unicode Handling" / "should handle emoji in tool
/// results".
#[tokio::test]
async fn google_should_handle_emoji_in_tool_results() {
    if live::env_key("GEMINI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("google", "gemini-2.5-flash");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Google Provider Unicode Handling" / "should handle real-world
/// `LinkedIn` comment data with emoji".
#[tokio::test]
async fn google_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("GEMINI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("google", "gemini-2.5-flash");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Google Provider Unicode Handling" / "should handle unpaired high
/// surrogate (0xD83D) in tool results".
#[tokio::test]
async fn google_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("GEMINI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("google", "gemini-2.5-flash");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "OpenAI Completions Provider Unicode Handling" / "should handle
/// emoji in tool results".
#[tokio::test]
async fn openai_completions_should_handle_emoji_in_tool_results() {
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("openai", "gpt-4o-mini");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "OpenAI Completions Provider Unicode Handling" / "should handle
/// real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn openai_completions_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("openai", "gpt-4o-mini");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "OpenAI Completions Provider Unicode Handling" / "should handle
/// unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn openai_completions_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("openai", "gpt-4o-mini");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "OpenAI Responses Provider Unicode Handling" / "should handle
/// emoji in tool results".
#[tokio::test]
async fn openai_responses_should_handle_emoji_in_tool_results() {
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("openai", "gpt-5-mini");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "OpenAI Responses Provider Unicode Handling" / "should handle
/// real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn openai_responses_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("openai", "gpt-5-mini");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "OpenAI Responses Provider Unicode Handling" / "should handle
/// unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn openai_responses_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("OPENAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("openai", "gpt-5-mini");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Azure OpenAI Responses Provider Unicode Handling" / "should
/// handle emoji in tool results".
#[tokio::test]
async fn azure_openai_responses_should_handle_emoji_in_tool_results() {
    if !live::has_azure_openai_credentials() {
        return;
    }
    let llm = live::model("azure-openai-responses", "gpt-4o-mini");
    let options = azure_options(&llm.id);
    test_emoji_in_tool_results(&llm, &options).await;
}

/// Upstream "Azure OpenAI Responses Provider Unicode Handling" / "should
/// handle real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn azure_openai_responses_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if !live::has_azure_openai_credentials() {
        return;
    }
    let llm = live::model("azure-openai-responses", "gpt-4o-mini");
    let options = azure_options(&llm.id);
    test_real_world_linkedin_data(&llm, &options).await;
}

/// Upstream "Azure OpenAI Responses Provider Unicode Handling" / "should
/// handle unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn azure_openai_responses_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if !live::has_azure_openai_credentials() {
        return;
    }
    let llm = live::model("azure-openai-responses", "gpt-4o-mini");
    let options = azure_options(&llm.id);
    test_unpaired_high_surrogate(&llm, &options).await;
}

/// Upstream "Anthropic Provider Unicode Handling" / "should handle emoji in
/// tool results".
#[tokio::test]
async fn anthropic_should_handle_emoji_in_tool_results() {
    if live::env_key("ANTHROPIC_API_KEY").is_none() {
        return;
    }
    let llm = live::model("anthropic", "claude-haiku-4-5");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Anthropic Provider Unicode Handling" / "should handle real-world
/// `LinkedIn` comment data with emoji".
#[tokio::test]
async fn anthropic_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("ANTHROPIC_API_KEY").is_none() {
        return;
    }
    let llm = live::model("anthropic", "claude-haiku-4-5");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Anthropic Provider Unicode Handling" / "should handle unpaired
/// high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn anthropic_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("ANTHROPIC_API_KEY").is_none() {
        return;
    }
    let llm = live::model("anthropic", "claude-haiku-4-5");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

// =========================================================================
// OAuth-backed providers (credentials from ~/.pi/agent/oauth.json)
// =========================================================================

/// Upstream "Anthropic OAuth Provider Unicode Handling" / "should handle emoji
/// in tool results".
#[tokio::test]
async fn anthropic_oauth_should_handle_emoji_in_tool_results() {
    let Some(token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-haiku-4-5");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream "Anthropic OAuth Provider Unicode Handling" / "should handle
/// real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn anthropic_oauth_should_handle_real_world_linkedin_comment_data_with_emoji() {
    let Some(token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-haiku-4-5");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream "Anthropic OAuth Provider Unicode Handling" / "should handle
/// unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn anthropic_oauth_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    let Some(token) = live::resolve_api_key("anthropic").await else {
        return;
    };
    let llm = live::model("anthropic", "claude-haiku-4-5");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream "GitHub Copilot Provider Unicode Handling" / "claude-haiku-4.5 -
/// should handle emoji in tool results".
#[tokio::test]
async fn github_copilot_claude_haiku_4_5_should_handle_emoji_in_tool_results() {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-haiku-4.5");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream "GitHub Copilot Provider Unicode Handling" / "claude-haiku-4.5 -
/// should handle real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn github_copilot_claude_haiku_4_5_should_handle_real_world_linkedin_comment_data_with_emoji()
{
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-haiku-4.5");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream "GitHub Copilot Provider Unicode Handling" / "claude-haiku-4.5 -
/// should handle unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn github_copilot_claude_haiku_4_5_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results()
 {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-haiku-4.5");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream "GitHub Copilot Provider Unicode Handling" / "claude-sonnet-4 -
/// should handle emoji in tool results".
#[tokio::test]
async fn github_copilot_claude_sonnet_4_should_handle_emoji_in_tool_results() {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream "GitHub Copilot Provider Unicode Handling" / "claude-sonnet-4 -
/// should handle real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn github_copilot_claude_sonnet_4_should_handle_real_world_linkedin_comment_data_with_emoji()
{
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream "GitHub Copilot Provider Unicode Handling" / "claude-sonnet-4 -
/// should handle unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn github_copilot_claude_sonnet_4_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results()
 {
    let Some(token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let llm = live::model("github-copilot", "claude-sonnet-4.6");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream "xAI Provider Unicode Handling" / "should handle emoji in tool
/// results".
#[tokio::test]
async fn xai_should_handle_emoji_in_tool_results() {
    if live::env_key("XAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xai", "grok-4.3");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "xAI Provider Unicode Handling" / "should handle real-world
/// `LinkedIn` comment data with emoji".
#[tokio::test]
async fn xai_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("XAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xai", "grok-4.3");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "xAI Provider Unicode Handling" / "should handle unpaired high
/// surrogate (0xD83D) in tool results".
#[tokio::test]
async fn xai_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("XAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xai", "grok-4.3");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Groq Provider Unicode Handling" / "should handle emoji in tool
/// results".
#[tokio::test]
async fn groq_should_handle_emoji_in_tool_results() {
    if live::env_key("GROQ_API_KEY").is_none() {
        return;
    }
    let llm = live::model("groq", "openai/gpt-oss-20b");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Groq Provider Unicode Handling" / "should handle real-world
/// `LinkedIn` comment data with emoji".
#[tokio::test]
async fn groq_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("GROQ_API_KEY").is_none() {
        return;
    }
    let llm = live::model("groq", "openai/gpt-oss-20b");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Groq Provider Unicode Handling" / "should handle unpaired high
/// surrogate (0xD83D) in tool results".
#[tokio::test]
async fn groq_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("GROQ_API_KEY").is_none() {
        return;
    }
    let llm = live::model("groq", "openai/gpt-oss-20b");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Cerebras Provider Unicode Handling" / "should handle emoji in
/// tool results".
#[tokio::test]
async fn cerebras_should_handle_emoji_in_tool_results() {
    if live::env_key("CEREBRAS_API_KEY").is_none() {
        return;
    }
    let llm = live::model("cerebras", "gpt-oss-120b");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Cerebras Provider Unicode Handling" / "should handle real-world
/// `LinkedIn` comment data with emoji".
#[tokio::test]
async fn cerebras_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("CEREBRAS_API_KEY").is_none() {
        return;
    }
    let llm = live::model("cerebras", "gpt-oss-120b");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Cerebras Provider Unicode Handling" / "should handle unpaired
/// high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn cerebras_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("CEREBRAS_API_KEY").is_none() {
        return;
    }
    let llm = live::model("cerebras", "gpt-oss-120b");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Cloudflare Workers AI Provider Unicode Handling" / "should handle
/// emoji in tool results".
#[tokio::test]
async fn cloudflare_workers_ai_should_handle_emoji_in_tool_results() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Cloudflare Workers AI Provider Unicode Handling" / "should handle
/// real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn cloudflare_workers_ai_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Cloudflare Workers AI Provider Unicode Handling" / "should handle
/// unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn cloudflare_workers_ai_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if !live::has_cloudflare_workers_ai_credentials() {
        return;
    }
    let llm = live::model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Cloudflare AI Gateway Provider Unicode Handling" / "should handle
/// emoji in tool results".
#[tokio::test]
async fn cloudflare_ai_gateway_should_handle_emoji_in_tool_results() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Cloudflare AI Gateway Provider Unicode Handling" / "should handle
/// real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn cloudflare_ai_gateway_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Cloudflare AI Gateway Provider Unicode Handling" / "should handle
/// unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn cloudflare_ai_gateway_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if !live::has_cloudflare_ai_gateway_credentials() {
        return;
    }
    let llm = live::model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    );
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Hugging Face Provider Unicode Handling" / "should handle emoji in
/// tool results".
#[tokio::test]
async fn hugging_face_should_handle_emoji_in_tool_results() {
    if live::env_key("HF_TOKEN").is_none() {
        return;
    }
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Hugging Face Provider Unicode Handling" / "should handle
/// real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn hugging_face_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("HF_TOKEN").is_none() {
        return;
    }
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Hugging Face Provider Unicode Handling" / "should handle unpaired
/// high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn hugging_face_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("HF_TOKEN").is_none() {
        return;
    }
    let llm = live::model("huggingface", "moonshotai/Kimi-K2.5");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Together AI Provider Unicode Handling" / "should handle emoji in
/// tool results".
#[tokio::test]
async fn together_ai_should_handle_emoji_in_tool_results() {
    if live::env_key("TOGETHER_API_KEY").is_none() {
        return;
    }
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    test_emoji_in_tool_results(&llm, &options).await;
}

/// Upstream "Together AI Provider Unicode Handling" / "should handle
/// real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn together_ai_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("TOGETHER_API_KEY").is_none() {
        return;
    }
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    test_real_world_linkedin_data(&llm, &options).await;
}

/// Upstream "Together AI Provider Unicode Handling" / "should handle unpaired
/// high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn together_ai_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("TOGETHER_API_KEY").is_none() {
        return;
    }
    let llm = live::model("together", "moonshotai/Kimi-K2.6");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    test_unpaired_high_surrogate(&llm, &options).await;
}

/// Upstream "Baseten Provider Unicode Handling" / "should handle emoji in tool
/// results".
#[tokio::test]
async fn baseten_should_handle_emoji_in_tool_results() {
    if live::env_key("BASETEN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    test_emoji_in_tool_results(&llm, &options).await;
}

/// Upstream "Baseten Provider Unicode Handling" / "should handle real-world
/// `LinkedIn` comment data with emoji".
#[tokio::test]
async fn baseten_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("BASETEN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    test_real_world_linkedin_data(&llm, &options).await;
}

/// Upstream "Baseten Provider Unicode Handling" / "should handle unpaired high
/// surrogate (0xD83D) in tool results".
#[tokio::test]
async fn baseten_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("BASETEN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("baseten", "zai-org/GLM-5.2");
    let options = live::LiveOptions {
        reasoning: Some(ThinkingLevel::High),
        ..live::LiveOptions::default()
    };
    test_unpaired_high_surrogate(&llm, &options).await;
}

/// Upstream "zAI Provider Unicode Handling" / "should handle emoji in tool
/// results".
#[tokio::test]
async fn zai_should_handle_emoji_in_tool_results() {
    if live::env_key("ZAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("zai", "glm-5.2");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "zAI Provider Unicode Handling" / "should handle real-world
/// `LinkedIn` comment data with emoji".
#[tokio::test]
async fn zai_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("ZAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("zai", "glm-5.2");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "zAI Provider Unicode Handling" / "should handle unpaired high
/// surrogate (0xD83D) in tool results".
#[tokio::test]
async fn zai_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("ZAI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("zai", "glm-5.2");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Mistral Provider Unicode Handling" / "should handle emoji in tool
/// results".
#[tokio::test]
async fn mistral_should_handle_emoji_in_tool_results() {
    if live::env_key("MISTRAL_API_KEY").is_none() {
        return;
    }
    let llm = live::model("mistral", "devstral-medium-latest");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Mistral Provider Unicode Handling" / "should handle real-world
/// `LinkedIn` comment data with emoji".
#[tokio::test]
async fn mistral_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("MISTRAL_API_KEY").is_none() {
        return;
    }
    let llm = live::model("mistral", "devstral-medium-latest");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Mistral Provider Unicode Handling" / "should handle unpaired high
/// surrogate (0xD83D) in tool results".
#[tokio::test]
async fn mistral_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("MISTRAL_API_KEY").is_none() {
        return;
    }
    let llm = live::model("mistral", "devstral-medium-latest");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "MiniMax Provider Unicode Handling" / "should handle emoji in tool
/// results".
#[tokio::test]
async fn minimax_should_handle_emoji_in_tool_results() {
    if live::env_key("MINIMAX_API_KEY").is_none() {
        return;
    }
    let llm = live::model("minimax", "MiniMax-M2.7");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "MiniMax Provider Unicode Handling" / "should handle real-world
/// `LinkedIn` comment data with emoji".
#[tokio::test]
async fn minimax_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("MINIMAX_API_KEY").is_none() {
        return;
    }
    let llm = live::model("minimax", "MiniMax-M2.7");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "MiniMax Provider Unicode Handling" / "should handle unpaired high
/// surrogate (0xD83D) in tool results".
#[tokio::test]
async fn minimax_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("MINIMAX_API_KEY").is_none() {
        return;
    }
    let llm = live::model("minimax", "MiniMax-M2.7");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Xiaomi MiMo (API billing) Provider Unicode Handling" / "should
/// handle emoji in tool results".
#[tokio::test]
async fn xiaomi_should_handle_emoji_in_tool_results() {
    if live::env_key("XIAOMI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Xiaomi MiMo (API billing) Provider Unicode Handling" / "should
/// handle real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn xiaomi_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("XIAOMI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Xiaomi MiMo (API billing) Provider Unicode Handling" / "should
/// handle unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn xiaomi_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("XIAOMI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xiaomi", "mimo-v2.5-pro");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Xiaomi MiMo Token Plan (CN) Provider Unicode Handling" / "should
/// handle emoji in tool results".
#[tokio::test]
async fn xiaomi_mimo_token_plan_cn_should_handle_emoji_in_tool_results() {
    if live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Xiaomi MiMo Token Plan (CN) Provider Unicode Handling" / "should
/// handle real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn xiaomi_mimo_token_plan_cn_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Xiaomi MiMo Token Plan (CN) Provider Unicode Handling" / "should
/// handle unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn xiaomi_mimo_token_plan_cn_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("XIAOMI_TOKEN_PLAN_CN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xiaomi-token-plan-cn", "mimo-v2.5-pro");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Xiaomi MiMo Token Plan (AMS) Provider Unicode Handling" / "should
/// handle emoji in tool results".
#[tokio::test]
async fn xiaomi_mimo_token_plan_ams_should_handle_emoji_in_tool_results() {
    if live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Xiaomi MiMo Token Plan (AMS) Provider Unicode Handling" / "should
/// handle real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn xiaomi_mimo_token_plan_ams_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Xiaomi MiMo Token Plan (AMS) Provider Unicode Handling" / "should
/// handle unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn xiaomi_mimo_token_plan_ams_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("XIAOMI_TOKEN_PLAN_AMS_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xiaomi-token-plan-ams", "mimo-v2.5-pro");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Xiaomi MiMo Token Plan (SGP) Provider Unicode Handling" / "should
/// handle emoji in tool results".
#[tokio::test]
async fn xiaomi_mimo_token_plan_sgp_should_handle_emoji_in_tool_results() {
    if live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Xiaomi MiMo Token Plan (SGP) Provider Unicode Handling" / "should
/// handle real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn xiaomi_mimo_token_plan_sgp_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Xiaomi MiMo Token Plan (SGP) Provider Unicode Handling" / "should
/// handle unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn xiaomi_mimo_token_plan_sgp_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("XIAOMI_TOKEN_PLAN_SGP_API_KEY").is_none() {
        return;
    }
    let llm = live::model("xiaomi-token-plan-sgp", "mimo-v2.5-pro");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Qwen Token Plan Provider Unicode Handling" / "should handle emoji
/// in tool results".
#[tokio::test]
async fn qwen_token_plan_should_handle_emoji_in_tool_results() {
    if live::env_key("QWEN_TOKEN_PLAN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Qwen Token Plan Provider Unicode Handling" / "should handle
/// real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn qwen_token_plan_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("QWEN_TOKEN_PLAN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Qwen Token Plan Provider Unicode Handling" / "should handle
/// unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn qwen_token_plan_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("QWEN_TOKEN_PLAN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("qwen-token-plan", "qwen3.7-max");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Qwen Token Plan Individual Provider Unicode Handling" / "should
/// handle emoji in tool results".
#[tokio::test]
async fn qwen_token_plan_individual_should_handle_emoji_in_tool_results() {
    if live::env_key("QWEN_TOKEN_PLAN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Qwen Token Plan Individual Provider Unicode Handling" / "should
/// handle real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn qwen_token_plan_individual_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("QWEN_TOKEN_PLAN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Qwen Token Plan Individual Provider Unicode Handling" / "should
/// handle unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn qwen_token_plan_individual_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("QWEN_TOKEN_PLAN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("qwen-token-plan-individual", "qwen3.8-max");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Qwen Token Plan (CN) Provider Unicode Handling" / "should handle
/// emoji in tool results".
#[tokio::test]
async fn qwen_token_plan_cn_should_handle_emoji_in_tool_results() {
    if live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Qwen Token Plan (CN) Provider Unicode Handling" / "should handle
/// real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn qwen_token_plan_cn_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Qwen Token Plan (CN) Provider Unicode Handling" / "should handle
/// unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn qwen_token_plan_cn_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("QWEN_TOKEN_PLAN_CN_API_KEY").is_none() {
        return;
    }
    let llm = live::model("qwen-token-plan-cn", "qwen3.7-max");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Kimi For Coding Provider Unicode Handling" / "should handle emoji
/// in tool results".
#[tokio::test]
async fn kimi_for_coding_should_handle_emoji_in_tool_results() {
    if live::env_key("KIMI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("kimi-coding", "kimi-for-coding");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Kimi For Coding Provider Unicode Handling" / "should handle
/// real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn kimi_for_coding_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("KIMI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("kimi-coding", "kimi-for-coding");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Kimi For Coding Provider Unicode Handling" / "should handle
/// unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn kimi_for_coding_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("KIMI_API_KEY").is_none() {
        return;
    }
    let llm = live::model("kimi-coding", "kimi-for-coding");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Vercel AI Gateway Provider Unicode Handling" / "should handle
/// emoji in tool results".
#[tokio::test]
async fn vercel_ai_gateway_should_handle_emoji_in_tool_results() {
    if live::env_key("AI_GATEWAY_API_KEY").is_none() {
        return;
    }
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Vercel AI Gateway Provider Unicode Handling" / "should handle
/// real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn vercel_ai_gateway_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if live::env_key("AI_GATEWAY_API_KEY").is_none() {
        return;
    }
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Vercel AI Gateway Provider Unicode Handling" / "should handle
/// unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn vercel_ai_gateway_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if live::env_key("AI_GATEWAY_API_KEY").is_none() {
        return;
    }
    let llm = live::model("vercel-ai-gateway", "google/gemini-2.5-flash");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Amazon Bedrock Provider Unicode Handling" / "should handle emoji
/// in tool results".
#[tokio::test]
async fn amazon_bedrock_should_handle_emoji_in_tool_results() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    test_emoji_in_tool_results(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Amazon Bedrock Provider Unicode Handling" / "should handle
/// real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn amazon_bedrock_should_handle_real_world_linkedin_comment_data_with_emoji() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    test_real_world_linkedin_data(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "Amazon Bedrock Provider Unicode Handling" / "should handle
/// unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn amazon_bedrock_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    if !live::has_bedrock_credentials() {
        return;
    }
    let llm = live::model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::default()).await;
}

/// Upstream "OpenAI Codex Provider Unicode Handling" / "gpt-5.5 - should
/// handle emoji in tool results".
#[tokio::test]
async fn openai_codex_gpt_5_5_should_handle_emoji_in_tool_results() {
    let Some(token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    test_emoji_in_tool_results(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream "OpenAI Codex Provider Unicode Handling" / "gpt-5.5 - should
/// handle real-world `LinkedIn` comment data with emoji".
#[tokio::test]
async fn openai_codex_gpt_5_5_should_handle_real_world_linkedin_comment_data_with_emoji() {
    let Some(token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    test_real_world_linkedin_data(&llm, &live::LiveOptions::key(token)).await;
}

/// Upstream "OpenAI Codex Provider Unicode Handling" / "gpt-5.5 - should
/// handle unpaired high surrogate (0xD83D) in tool results".
#[tokio::test]
async fn openai_codex_gpt_5_5_should_handle_unpaired_high_surrogate_0xd83d_in_tool_results() {
    let Some(token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let llm = live::model("openai-codex", "gpt-5.5");
    test_unpaired_high_surrogate(&llm, &live::LiveOptions::key(token)).await;
}

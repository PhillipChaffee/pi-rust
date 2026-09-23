//! Tool call ID normalization tests, ported from `packages/ai/test/
//! tool-call-id-normalization.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: tool call ids the OpenAI
//! Responses API generates (`{call_id}|{id}`, the `{id}` half 400+ chars with
//! `+`, `/`, `=`) must normalize when the history replays against other
//! providers. The prefilled-context probes carry the exact failing id from
//! the earendil-works/pi#1022 JSONL.
//!
//! Like upstream's `it.skipIf`, a probe without its credential returns early:
//! the OAuth-backed tokens resolve through [`common::live::resolve_api_key`],
//! the OpenRouter key through the provider's env vars.

#![expect(
    clippy::expect_used,
    reason = "the tests pin live outcomes; an unexpected shape panics the test by design"
)]
#![expect(
    clippy::print_stdout,
    reason = "the probe logs the generated tool call id like upstream's console.log"
)]

use pi_ai::auth::resolve::now_ms;
use pi_ai::compat::get_env_api_key;
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, Context, Message, ProviderId, StopReason, TextContent,
    Tool, ToolCall, ToolResultBlock, ToolResultMessage, Usage, UsageCost,
};
use serde_json::json;

mod common;
use common::live;

/// The echo tool the handoffs ride, upstream's `echoTool` over
/// `Type.Object({ message: Type.String(...) })`.
fn echo_tool() -> Tool {
    Tool {
        name: "echo".to_owned(),
        description: "Echoes the message back".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "message": { "type": "string", "description": "Message to echo back" }
            },
            "required": ["message"]
        }),
        constrained_sampling: None,
    }
}

/// The tool call block a copilot response carries, upstream's
/// `content.find(c => c.type === "toolCall")` plus the defined/type asserts.
fn copilot_tool_call(response: &AssistantMessage) -> ToolCall {
    response
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .expect("tool call block in the copilot response")
}

/// Upstream `describe` "Tool Call ID Normalization - Live Handoff" /
/// `it.skipIf(!copilotToken || !openrouterKey)` "github-copilot -> openrouter
/// should normalize pipe-separated IDs": generate the tool call with
/// github-copilot, answer it, then replay the history against openrouter
/// without a "`call_id` too long" error.
#[tokio::test]
async fn github_copilot_to_openrouter_should_normalize_pipe_separated_ids() {
    let Some(copilot_token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let Some(openrouter_key) = get_env_api_key("openrouter", None).filter(|key| !key.is_empty())
    else {
        return;
    };
    let copilot_model = live::model("github-copilot", "gpt-5.5");
    let openrouter_model = live::model("openrouter", "openai/gpt-5.5");

    // Step 1: generate the tool call with github-copilot.
    let user_message = live::user_message("Use the echo tool to echo 'hello world'");
    let first_context = Context {
        system_prompt: Some(
            "You are a helpful assistant. Use the echo tool when asked.".to_owned(),
        ),
        messages: vec![user_message.clone()],
        tools: Some(vec![echo_tool()]),
    };
    let assistant_response = live::complete(
        &copilot_model,
        &first_context,
        &live::LiveOptions::key(copilot_token),
    )
    .await;
    assert_eq!(
        assistant_response.stop_reason,
        StopReason::ToolUse,
        "Copilot error: {:?}",
        assistant_response.error_message
    );
    let tool_call = copilot_tool_call(&assistant_response);
    // The pipe-separated id is the OpenAI Responses format the handoff
    // normalizes, upstream's `expect(toolCall.id).toContain("|")`.
    assert!(
        tool_call.id.contains('|'),
        "pipe-separated id: {}",
        tool_call.id
    );
    println!(
        "Tool call ID from github-copilot: {}...",
        tool_call.id.chars().take(80).collect::<String>()
    );

    let tool_result = Message::ToolResult(ToolResultMessage {
        tool_call_id: tool_call.id.clone(),
        tool_name: "echo".to_owned(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: "hello world".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: now_ms(),
    });

    // Step 2: complete with openrouter over the openai-completions wire.
    let handoff_context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![
            user_message.clone(),
            Message::Assistant(assistant_response),
            tool_result,
            live::user_message("Say hi"),
        ],
        tools: Some(vec![echo_tool()]),
    };
    let openrouter_response = live::complete(
        &openrouter_model,
        &handoff_context,
        &live::LiveOptions::key(openrouter_key),
    )
    .await;

    // The handoff must not fail with a "call_id too long" error.
    assert_ne!(
        openrouter_response.stop_reason,
        StopReason::Error,
        "OpenRouter error: {:?}",
        openrouter_response.error_message
    );
    assert!(
        openrouter_response.error_message.is_none(),
        "OpenRouter error: {:?}",
        openrouter_response.error_message
    );
}

/// Upstream `describe` "Tool Call ID Normalization - Live Handoff" /
/// `it.skipIf(!copilotToken || !codexToken)` "github-copilot -> openai-codex
/// should normalize pipe-separated IDs": same handoff shape against the
/// openai-codex-responses wire.
#[tokio::test]
async fn github_copilot_to_openai_codex_should_normalize_pipe_separated_ids() {
    let Some(copilot_token) = live::resolve_api_key("github-copilot").await else {
        return;
    };
    let Some(codex_token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let copilot_model = live::model("github-copilot", "gpt-5.5");
    let codex_model = live::model("openai-codex", "gpt-5.5");

    // Step 1: generate the tool call with github-copilot.
    let user_message = live::user_message("Use the echo tool to echo 'test message'");
    let first_context = Context {
        system_prompt: Some(
            "You are a helpful assistant. Use the echo tool when asked.".to_owned(),
        ),
        messages: vec![user_message.clone()],
        tools: Some(vec![echo_tool()]),
    };
    let assistant_response = live::complete(
        &copilot_model,
        &first_context,
        &live::LiveOptions::key(copilot_token),
    )
    .await;
    assert_eq!(
        assistant_response.stop_reason,
        StopReason::ToolUse,
        "Copilot error: {:?}",
        assistant_response.error_message
    );
    let tool_call = copilot_tool_call(&assistant_response);

    let tool_result = Message::ToolResult(ToolResultMessage {
        tool_call_id: tool_call.id.clone(),
        tool_name: "echo".to_owned(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: "test message".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: now_ms(),
    });

    // Step 2: complete with openai-codex over the openai-codex-responses wire.
    let handoff_context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![
            user_message.clone(),
            Message::Assistant(assistant_response),
            tool_result,
            live::user_message("Say hi"),
        ],
        tools: Some(vec![echo_tool()]),
    };
    let codex_response = live::complete(
        &codex_model,
        &handoff_context,
        &live::LiveOptions::key(codex_token),
    )
    .await;

    // The handoff must not fail with an id validation error.
    assert_ne!(
        codex_response.stop_reason,
        StopReason::Error,
        "Codex error: {:?}",
        codex_response.error_message
    );
    assert!(
        codex_response.error_message.is_none(),
        "Codex error: {:?}",
        codex_response.error_message
    );
}

/// The exact tool call id from the issue #1022 JSONL, upstream's
/// `FAILING_TOOL_CALL_ID`.
const FAILING_TOOL_CALL_ID: &str = "call_pAYbIr76hXIjncD9UE4eGfnS|t5nnb2qYMFWGSsr13fhCd1CaCu3t3qONEPuOudu4HSVEtA8YJSL6FAZUxvoOoD792VIJWl91g87EdqsCWp9krVsdBysQoDaf9lMCLb8BS4EYi4gQd5kBQBYLlgD71PYwvf+TbMD9J9/5OMD42oxSRj8H+vRf78/l2Xla33LWz4nOgsddBlbvabICRs8GHt5C9PK5keFtzyi3lsyVKNlfduK3iphsZqs4MLv4zyGJnvZo/+QzShyk5xnMSQX/f98+aEoNflEApCdEOXipipgeiNWnpFSHbcwmMkZoJhURNu+JEz3xCh1mrXeYoN5o+trLL3IXJacSsLYXDrYTipZZbJFRPAucgbnjYBC+/ZzJOfkwCs+Gkw7EoZR7ZQgJ8ma+9586n4tT4cI8DEhBSZsWMjrCt8dxKg==";

/// The prefilled context carrying the failing id, upstream's
/// `buildPrefilledMessages`: user, assistant tool call, tool result, and the
/// follow-up user turn.
fn build_prefilled_messages() -> Vec<Message> {
    let mut arguments = serde_json::Map::new();
    arguments.insert("message".to_owned(), json!("hello"));
    let assistant_message = AssistantMessage {
        content: vec![AssistantBlock::ToolCall(ToolCall {
            id: FAILING_TOOL_CALL_ID.to_owned(),
            name: "echo".to_owned(),
            arguments,
            thought_signature: None,
            namespace: None,
        })],
        api: Api::from("openai-responses"),
        provider: ProviderId::from("github-copilot"),
        model: "gpt-5.2-codex".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage {
            input: 100,
            output: 50,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 150,
            cost: UsageCost::default(),
        },
        stop_reason: StopReason::ToolUse,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms() - 1500,
    };
    vec![
        Message::User(pi_ai::types::UserMessage {
            content: pi_ai::types::UserContent::Text(
                "Use the echo tool to echo 'hello'".to_owned(),
            ),
            timestamp: now_ms() - 2000,
        }),
        Message::Assistant(assistant_message),
        Message::ToolResult(ToolResultMessage {
            tool_call_id: FAILING_TOOL_CALL_ID.to_owned(),
            tool_name: "echo".to_owned(),
            content: vec![ToolResultBlock::Text(TextContent {
                text: "hello".to_owned(),
                text_signature: None,
            })],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: now_ms() - 1000,
        }),
        live::user_message("Say hi"),
    ]
}

/// Upstream `describe` "Tool Call ID Normalization - Prefilled Context" /
/// `it.skipIf(!openrouterKey)` "openrouter should handle prefilled context
/// with long pipe-separated IDs".
#[tokio::test]
async fn openrouter_should_handle_prefilled_context_with_long_pipe_separated_ids() {
    let Some(openrouter_key) = get_env_api_key("openrouter", None).filter(|key| !key.is_empty())
    else {
        return;
    };
    let model = live::model("openrouter", "openai/gpt-5.5");
    let context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: build_prefilled_messages(),
        tools: Some(vec![echo_tool()]),
    };

    let response = live::complete(&model, &context, &live::LiveOptions::key(openrouter_key)).await;

    // The replay must not fail with a "call_id too long" error.
    assert_ne!(
        response.stop_reason,
        StopReason::Error,
        "OpenRouter error: {:?}",
        response.error_message
    );
    if let Some(message) = &response.error_message {
        assert!(!message.contains("call_id"), "got: {message}");
        assert!(!message.contains("too long"), "got: {message}");
    }
}

/// Upstream `describe` "Tool Call ID Normalization - Prefilled Context" /
/// `it.skipIf(!codexToken)` "openai-codex should handle prefilled context
/// with long pipe-separated IDs".
#[tokio::test]
async fn openai_codex_should_handle_prefilled_context_with_long_pipe_separated_ids() {
    let Some(codex_token) = live::resolve_api_key("openai-codex").await else {
        return;
    };
    let model = live::model("openai-codex", "gpt-5.5");
    let context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: build_prefilled_messages(),
        tools: Some(vec![echo_tool()]),
    };

    let response = live::complete(&model, &context, &live::LiveOptions::key(codex_token)).await;

    // The replay must not fail with an id validation error.
    assert_ne!(
        response.stop_reason,
        StopReason::Error,
        "Codex error: {:?}",
        response.error_message
    );
    if let Some(message) = &response.error_message {
        assert!(!message.contains("id"), "got: {message}");
        assert!(!message.contains("additional characters"), "got: {message}");
    }
}

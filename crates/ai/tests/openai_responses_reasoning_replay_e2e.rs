//! Env-gated OpenAI Responses reasoning-replay E2E probes, ported from the
//! upstream `openai-responses-reasoning-replay-e2e.test.ts` live probes at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Like upstream's `describe.skipIf`, a probe without its credentials
//! returns early; the catalog-coverage assertion always runs. Upstream's
//! payload captures only feed console logging, so the port pins the
//! response shape the cases actually assert on.

#![expect(
    clippy::panic,
    reason = "the tests pin live outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::api::{anthropic_messages, openai_responses};
use pi_ai::env_api_keys::get_env_api_key;
use pi_ai::types::AssistantMessage;
use pi_ai::types::{
    AssistantBlock, Context, Message, SimpleStreamOptions, StopReason, TextContent, ThinkingLevel,
    Tool, ToolCall, ToolResultBlock, ToolResultMessage,
};

mod common;
use common::{assert_live_text_reply, builtin_model};

/// The number-doubling tool the replay probes send, upstream's `testTool`.
fn double_number_tool() -> Tool {
    Tool {
        name: "double_number".to_owned(),
        description: "Doubles a number and returns the result".to_owned(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "number", "description": "A number to double" } },
        }),
        constrained_sampling: None,
    }
}

/// The tool-result turn the replay histories carry.
fn tool_result(tool_call: &ToolCall, text: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: pi_ai::auth::resolve::now_ms(),
    })
}

fn user_message(text: &str) -> Message {
    Message::User(pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Text(text.to_owned()),
        timestamp: pi_ai::auth::resolve::now_ms(),
    })
}

/// The first tool call the probes' first turns produce, upstream's
/// `toolCallBlock` find with its missing-block panic.
fn first_tool_call(message: &AssistantMessage) -> ToolCall {
    message
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("model did not use the tool"))
}

/// The catalog-coverage assertion the live probes open with: the replay
/// probes' models ride the OpenAI Responses and Anthropic Messages wires.
#[test]
fn reasoning_replay_catalog_carries_the_probe_models() {
    for (provider, id, api) in [
        ("openai", "gpt-5-mini", "openai-responses"),
        ("openai", "gpt-5.5", "openai-responses"),
        ("anthropic", "claude-sonnet-4-5", "anthropic-messages"),
    ] {
        common::assert_catalog_api(provider, id, api);
    }
}

/// The high-reasoning options the replay probes send, upstream's
/// `reasoningEffort: "high"`.
fn high_reasoning_options(api_key: &str) -> SimpleStreamOptions {
    SimpleStreamOptions {
        api_key: Some(api_key.to_owned()),
        reasoning: Some(ThinkingLevel::High),
        ..SimpleStreamOptions::default()
    }
}

/// The handoff replay turn, upstream's per-case continuation: the carried
/// history streams at high reasoning through the follow-up until the model
/// answers with the tool result.
async fn replay_handoff(
    model: &pi_ai::types::Model,
    api_key: &str,
    assistant: AssistantMessage,
    tool_call: &ToolCall,
    user: &Message,
) -> AssistantMessage {
    openai_responses::stream_simple(
        model,
        &Context {
            system_prompt: Some("You are a helpful assistant. Answer concisely.".to_owned()),
            messages: vec![
                user.clone(),
                Message::Assistant(assistant),
                tool_result(tool_call, "42"),
                user_message("What was the result? Answer with just the number."),
            ],
            tools: Some(vec![double_number_tool()]),
        },
        Some(&high_reasoning_options(api_key)),
    )
    .result()
    .await
}

/// The first probe, upstream's "skips reasoning-only history after an
/// aborted turn": a reasoning-only assistant turn replayed with the
/// `aborted` stop reason must not 400 on the orphaned reasoning item.
#[tokio::test]
async fn skips_reasoning_only_history_after_an_aborted_turn() {
    let Some(api_key) = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|key| !key.trim().is_empty())
    else {
        return;
    };
    let model = builtin_model("openai", "gpt-5-mini");
    let tool = double_number_tool();
    let user = user_message("Use the double_number tool to double 21.");

    let assistant = openai_responses::stream_simple(
        &model,
        &Context {
            system_prompt: Some("You are a helpful assistant. Use the tool.".to_owned()),
            messages: vec![user.clone()],
            tools: Some(vec![tool.clone()]),
        },
        Some(&high_reasoning_options(&api_key)),
    )
    .result()
    .await;

    let Some(thinking) = assistant.content.iter().find_map(|block| match block {
        AssistantBlock::Thinking(thinking)
            if thinking
                .thinking_signature
                .as_deref()
                .is_some_and(|signature| !signature.is_empty()) =>
        {
            Some(block.clone())
        }
        _ => None,
    }) else {
        panic!("Missing thinking signature from OpenAI Responses");
    };
    let mut corrupted = assistant.clone();
    corrupted.content = vec![thinking];
    corrupted.stop_reason = StopReason::Aborted;

    let response = openai_responses::stream_simple(
        &model,
        &Context {
            system_prompt: Some("You are a helpful assistant.".to_owned()),
            messages: vec![
                user.clone(),
                Message::Assistant(corrupted),
                user_message("Say hello to confirm you can continue."),
            ],
            tools: Some(vec![tool]),
        },
        Some(&high_reasoning_options(&api_key)),
    )
    .result()
    .await;

    assert_ne!(
        response.stop_reason,
        StopReason::Error,
        "Error: {:?}",
        response.error_message
    );
    assert!(
        response.error_message.is_none(),
        "{:?}",
        response.error_message
    );
    assert!(
        !response.content.is_empty(),
        "the model answers after the replay"
    );
}

/// The same-provider model handoff, upstream's "handles same-provider
/// different-model handoff with tool calls": gpt-5-mini's reasoning and
/// function call replay through gpt-5.5 without an orphaned-reasoning 400.
#[tokio::test]
async fn handles_same_provider_different_model_handoff_with_tool_calls() {
    let Some(api_key) = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|key| !key.trim().is_empty())
    else {
        return;
    };
    let model_a = builtin_model("openai", "gpt-5-mini");
    let model_b = builtin_model("openai", "gpt-5.5");
    let user = user_message("Use the double_number tool to double 21.");

    let assistant = openai_responses::stream_simple(
        &model_a,
        &Context {
            system_prompt: Some(
                "You are a helpful assistant. Always use the tool when asked.".to_owned(),
            ),
            messages: vec![user.clone()],
            tools: Some(vec![double_number_tool()]),
        },
        Some(&high_reasoning_options(&api_key)),
    )
    .result()
    .await;

    let tool_call = first_tool_call(&assistant);

    let response = replay_handoff(&model_b, &api_key, assistant, &tool_call, &user).await;

    assert_live_text_reply(&response, "42");
}

/// The cross-provider handoff, upstream's "handles cross-provider handoff
/// from Anthropic to OpenAI": Claude's thinking and tool call replay
/// through an OpenAI Responses model without a pairing-history 400.
#[tokio::test]
async fn handles_cross_provider_handoff_from_anthropic_to_openai() {
    let Some(anthropic_api_key) =
        get_env_api_key("anthropic", None).filter(|key| !key.trim().is_empty())
    else {
        return;
    };
    let Some(openai_api_key) = get_env_api_key("openai", None).filter(|key| !key.trim().is_empty())
    else {
        return;
    };
    let anthropic_model = builtin_model("anthropic", "claude-sonnet-4-5");
    let openai_model = builtin_model("openai", "gpt-5.5");
    let user = user_message("Use the double_number tool to double 21.");

    let assistant = anthropic_messages::stream(
        &anthropic_model,
        &Context {
            system_prompt: Some(
                "You are a helpful assistant. Always use the tool when asked.".to_owned(),
            ),
            messages: vec![user.clone()],
            tools: Some(vec![double_number_tool()]),
        },
        Some(&anthropic_messages::AnthropicStreamOptions {
            api_key: Some(anthropic_api_key),
            thinking_enabled: Some(true),
            thinking_budget_tokens: Some(5000),
            ..anthropic_messages::AnthropicStreamOptions::default()
        }),
    )
    .result()
    .await;

    let tool_call = first_tool_call(&assistant);

    let response =
        replay_handoff(&openai_model, &openai_api_key, assistant, &tool_call, &user).await;

    assert_live_text_reply(&response, "42");
}

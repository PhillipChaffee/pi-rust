//! OpenAI Completions Anthropic-format cache-control markers, ported from
//! `packages/ai/test/openai-completions-cache-control-format.test.ts` at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: upstream captures the request through the mocked
//! SDK's `create(params)`; the port reads the same payload off the mock's
//! recorded request body.

#![expect(
    clippy::expect_used,
    reason = "the tests pin stream outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::api::openai_completions::{OpenAiCompletionsOptions, stream};
use pi_ai::http::MockHttpClient;
use pi_ai::types::{
    AssistantBlock, CacheRetention, Context, Message, Model, ModelCompat, TextContent, Tool,
    ToolResultBlock,
};
use serde_json::{Value, json};

mod common;
use common::{builtin_model, openai_done_chunk, openai_mock_with, recorded_body, user_message_now};

/// The read tool the cache-control suites send.
fn read_tool() -> Tool {
    Tool {
        name: "read".to_owned(),
        description: "Read a file".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"],
        }),
        constrained_sampling: None,
    }
}

/// The capture context the marker suites stream: system prompt, one user
/// turn, one tool.
fn marker_context() -> Context {
    Context {
        system_prompt: Some("System prompt".to_owned()),
        messages: vec![user_message_now("Hello")],
        tools: Some(vec![read_tool()]),
    }
}

/// Stream the context and take the request's payload; the mock is shared so
/// the recorded body and the options agree.
async fn capture(model: &Model, options: OpenAiCompletionsOptions) -> Value {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let mut options = options;
    options.transport_options = common::mock_transport(&mock);

    let _ = stream(model, &marker_context(), Some(&options))
        .result()
        .await;
    recorded_body(&mock)
}

/// The keyed options the marker suites send.
fn keyed() -> OpenAiCompletionsOptions {
    OpenAiCompletionsOptions {
        api_key: Some("test-key".to_owned()),
        ..OpenAiCompletionsOptions::default()
    }
}

/// The ephemeral marker.
fn ephemeral() -> Value {
    json!({ "type": "ephemeral" })
}

/// The Anthropic marker set upstream's `expectAnthropicCacheMarkers` pins:
/// the instruction message, the last tool definition, and the last
/// conversation message each carry `cache_control: { type: "ephemeral" }`.
fn expect_anthropic_cache_markers(params: &Value) {
    let messages = params["messages"].as_array().expect("messages");
    let instruction = messages
        .iter()
        .find(|message| matches!(message["role"].as_str(), Some("system" | "developer")))
        .expect("the instruction message");
    assert!(instruction["content"].is_array(), "got: {instruction}");
    assert_eq!(instruction["content"][0]["cache_control"], ephemeral());

    let tools = params["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["cache_control"], ephemeral());

    let last = messages.last().expect("the last message");
    assert_eq!(last["role"], json!("user"));
    assert!(last["content"].is_array());
    assert_eq!(last["content"][0]["cache_control"], ephemeral());
}

/// The custom-qwen model whose compat opts into Anthropic cache markers.
fn custom_qwen() -> Model {
    Model {
        id: "custom-qwen".to_owned(),
        name: "Custom Qwen".to_owned(),
        reasoning: true,
        context_window: 128_000,
        max_tokens: 32_000,
        compat: Some(
            serde_json::from_value::<ModelCompat>(json!({ "cacheControlFormat": "anthropic" }))
                .expect("compat map"),
        ),
        ..common::openai_catalog_model("openrouter", "auto")
    }
}

// ---------------------------------------------------------------------------
// Anthropic cache-control format
// ---------------------------------------------------------------------------

/// Markers ride when the model compat enables them.
#[tokio::test]
async fn applies_anthropic_style_cache_markers_when_model_compat_enables_them() {
    let params = capture(&custom_qwen(), keyed()).await;

    expect_anthropic_cache_markers(&params);
}

/// The OpenRouter Anthropic batch alias carries the markers from its catalog
/// compat.
#[tokio::test]
async fn preserves_anthropic_style_cache_markers_for_openrouter_anthropic_batch_aliases() {
    let model = builtin_model("openrouter", "anthropic/claude-fable-5.1:batch");
    let params = capture(&model, keyed()).await;

    expect_anthropic_cache_markers(&params);
}

/// With tool history the conversation marker lands on the tool-result turn
/// instead of a user turn.
#[tokio::test]
async fn moves_the_conversation_cache_marker_to_a_tool_result() {
    let model = builtin_model("openrouter", "anthropic/claude-fable-5.1:batch");
    let timestamp = pi_ai::auth::resolve::now_ms();
    let context = Context {
        system_prompt: Some("System prompt".to_owned()),
        messages: vec![
            common::user_message_at("Read the file", timestamp),
            Message::Assistant(pi_ai::types::AssistantMessage {
                content: vec![AssistantBlock::ToolCall(pi_ai::types::ToolCall {
                    id: "call_1".to_owned(),
                    name: "read".to_owned(),
                    arguments: serde_json::Map::from_iter([(
                        "path".to_owned(),
                        json!("README.md"),
                    )]),
                    thought_signature: None,
                    namespace: None,
                })],
                ..common::assistant_message_with_content(
                    "openai-completions",
                    "openrouter",
                    model.id.as_str(),
                    Vec::new(),
                )
            }),
            common::tool_result_message(
                "call_1",
                vec![ToolResultBlock::Text(TextContent {
                    text: "file contents".to_owned(),
                    text_signature: None,
                })],
                None,
            ),
        ],
        ..marker_context()
    };
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let options = common::keyed_openai_options(&mock);

    let _ = stream(&model, &context, Some(&options)).result().await;
    let params = recorded_body(&mock);

    let messages = params["messages"].as_array().expect("messages");
    let user_message = messages
        .iter()
        .find(|message| message["role"] == json!("user"))
        .expect("the user turn");
    assert_eq!(user_message["content"], json!("Read the file"));

    let tool_message = messages.last().expect("the tool turn");
    assert_eq!(tool_message["role"], json!("tool"));
    assert!(tool_message["content"].is_array());
    assert_eq!(tool_message["content"][0]["cache_control"], ephemeral());
}

/// `cacheRetention: none` drops every marker: the instruction message keeps
/// its string content, the tool definition carries no marker, and the last
/// message keeps its string content.
#[tokio::test]
async fn omits_anthropic_style_cache_markers_when_cache_retention_is_none() {
    let options = OpenAiCompletionsOptions {
        cache_retention: Some(CacheRetention::None),
        ..keyed()
    };

    let params = capture(&custom_qwen(), options).await;

    let messages = params["messages"].as_array().expect("messages");
    let instruction = messages
        .iter()
        .find(|message| matches!(message["role"].as_str(), Some("system" | "developer")))
        .expect("the instruction message");
    assert!(instruction["content"].is_string(), "got: {instruction}");
    let tools = params["tools"].as_array().expect("tools");
    assert!(tools[0].get("cache_control").is_none());
    let last = messages.last().expect("the last message");
    assert!(last["content"].is_string(), "got: {last}");
}

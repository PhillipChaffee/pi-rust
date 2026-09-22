//! Cross-provider message rewriting, ported from
//! `packages/ai/test/lax-message-content.test.ts` and
//! `packages/ai/test/transform-messages-copilot-openai-to-anthropic.test.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin rewrite outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::api::transform_messages::{transform_messages, transform_messages_lenient};
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, ImageContent, Message, Modality, Model, ProviderId,
    StopReason, TextContent, ThinkingContent, ToolCall, ToolResultBlock, ToolResultMessage,
    UserBlock, UserContent, UserMessage,
};
use serde_json::{Value, json};

/// The text-only model the lax suite runs, so the image-downgrade path —
/// the primary crash site for null tool-result content — executes.
fn text_only_model() -> Model {
    Model {
        id: "test-model".to_owned(),
        name: "Test Model".to_owned(),
        api: Api::from("openai-completions"),
        provider: ProviderId::from("openai"),
        base_url: "https://example.invalid/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 128_000,
        max_tokens: 16_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The lax entry, upstream `lax-message-content.test.ts`: the wire's
/// `content: null` and missing content become empty arrays instead of
/// crashing (issues pi #6259, #6276).
#[test]
fn normalizes_null_and_missing_content_to_an_empty_array_instead_of_crashing() {
    let messages = vec![
        json!({ "role": "user", "content": null, "timestamp": 1 }),
        serde_json::json!({
            "role": "assistant",
            "content": null,
            "api": "openai-completions",
            "provider": "openai",
            "model": "test-model",
            "usage": {
                "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
            },
            "stopReason": "stop",
            "timestamp": 1,
        }),
        serde_json::json!({
            "role": "toolResult",
            "toolCallId": "call_1",
            "toolName": "web_search",
            "isError": false,
            "timestamp": 1,
        }),
    ];

    let result = transform_messages_lenient(&messages, &text_only_model(), None);

    assert_eq!(result.len(), 3);
    for message in &result {
        let content_is_empty = match message {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => text.is_empty(),
                UserContent::Blocks(blocks) => blocks.is_empty(),
            },
            Message::Assistant(assistant) => assistant.content.is_empty(),
            Message::ToolResult(tool_result) => tool_result.content.is_empty(),
        };
        assert!(content_is_empty, "content must normalize empty");
    }
}

// -- OpenAI to Anthropic session migration for Copilot Claude ---------------

/// The anthropic ID normalizer, upstream's `anthropicNormalizeToolCallId`.
fn copilot_normalizer() -> impl Fn(&str, &Model, &AssistantMessage) -> String {
    |id: &str, _model: &_, _source: &_| {
        let normalized: String = id
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                    character
                } else {
                    '_'
                }
            })
            .collect();
        normalized.chars().take(64).collect()
    }
}

fn copilot_claude_model() -> Model {
    Model {
        id: "claude-sonnet-4.6".to_owned(),
        name: "Claude Sonnet 4.6".to_owned(),
        api: Api::from("anthropic-messages"),
        provider: ProviderId::from("github-copilot"),
        base_url: "https://api.individual.githubcopilot.com".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![Modality::Text, Modality::Image],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 128_000,
        max_tokens: 16_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// An assistant message from the source OpenAI session, upstream's
/// `makeAssistantMessage`.
fn openai_assistant(content: Vec<AssistantBlock>) -> AssistantMessage {
    AssistantMessage {
        content,
        api: Api::from("openai-responses"),
        provider: ProviderId::from("github-copilot"),
        model: "gpt-5".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: pi_ai::types::Usage::default(),
        stop_reason: StopReason::ToolUse,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

fn user_message(text: &str) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp: 1,
    })
}

fn tool_result(tool_call_id: &str, tool_name: &str, text: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: tool_call_id.to_owned(),
        tool_name: tool_name.to_owned(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 1,
    })
}

fn tool_call(id: &str, name: &str, arguments: &Value) -> ToolCall {
    ToolCall {
        id: id.to_owned(),
        name: name.to_owned(),
        arguments: arguments.as_object().cloned().expect("arguments object"),
        thought_signature: None,
        namespace: None,
    }
}

#[test]
fn converts_thinking_blocks_to_plain_text_when_source_model_differs() {
    let messages = vec![
        user_message("hello"),
        Message::Assistant(AssistantMessage {
            content: vec![
                AssistantBlock::Thinking(ThinkingContent {
                    thinking: "Let me think about this...".to_owned(),
                    thinking_signature: Some("reasoning_content".to_owned()),
                    redacted: None,
                }),
                AssistantBlock::Text(TextContent {
                    text: "Hi there!".to_owned(),
                    text_signature: None,
                }),
            ],
            api: Api::from("openai-completions"),
            provider: ProviderId::from("github-copilot"),
            model: "gpt-4o".to_owned(),
            ..openai_assistant(Vec::new())
        }),
    ];

    let result = transform_messages(
        messages,
        &copilot_claude_model(),
        Some(&copilot_normalizer()),
    );
    let assistant = assistant_message_of(&result).expect("assistant message");

    // The thinking block converts to text since the models differ.
    let thinking_blocks = assistant
        .content
        .iter()
        .filter(|block| matches!(block, AssistantBlock::Thinking(_)))
        .count();
    let text_blocks = assistant
        .content
        .iter()
        .filter(|block| matches!(block, AssistantBlock::Text(_)))
        .count();
    assert_eq!(thinking_blocks, 0);
    assert!(text_blocks >= 2);
}

#[test]
fn removes_thought_signature_from_tool_calls_when_migrating_between_models() {
    let mut signature_call = tool_call("call_123", "bash", &serde_json::json!({ "command": "ls" }));
    signature_call.thought_signature = Some(
        r#"{"type": "reasoning.encrypted", "id": "call_123", "data": "encrypted"}"#.to_owned(),
    );
    let messages = vec![
        user_message("run a command"),
        Message::Assistant(openai_assistant(vec![AssistantBlock::ToolCall(
            signature_call,
        )])),
        tool_result("call_123", "bash", "output"),
    ];

    let result = transform_messages(
        messages,
        &copilot_claude_model(),
        Some(&copilot_normalizer()),
    );
    let assistant = assistant_message_of(&result).expect("assistant message");
    let call = tool_call_of(&assistant.content).expect("tool call");

    assert_eq!(call.thought_signature, None);
}

#[test]
fn adds_synthetic_tool_results_for_trailing_orphaned_tool_calls() {
    let messages = vec![
        user_message("read the file"),
        Message::Assistant(openai_assistant(vec![AssistantBlock::ToolCall(tool_call(
            "call_123|fc_123",
            "read",
            &serde_json::json!({ "path": "README.md" }),
        ))])),
    ];

    let result = transform_messages(
        messages,
        &copilot_claude_model(),
        Some(&copilot_normalizer()),
    );
    let last = result.last().expect("the synthetic result");
    #[expect(
        clippy::panic,
        reason = "an unexpected trailing message panics the test by design"
    )]
    let Message::ToolResult(tool_result) = last else {
        panic!("the trailing message is the synthetic tool result");
    };
    assert_eq!(tool_result.tool_call_id, "call_123_fc_123");
    assert_eq!(tool_result.tool_name, "read");
    assert!(tool_result.is_error);
    assert_eq!(
        tool_result_content_json(&tool_result.content),
        json!([{ "type": "text", "text": "No result provided" }]),
    );
}

#[test]
fn adds_synthetic_results_only_for_trailing_tool_calls_still_missing_results() {
    let messages = vec![
        user_message("run commands"),
        Message::Assistant(openai_assistant(vec![
            AssistantBlock::ToolCall(tool_call(
                "call_1|fc_1",
                "read",
                &serde_json::json!({ "path": "README.md" }),
            )),
            AssistantBlock::ToolCall(tool_call(
                "call_2|fc_2",
                "bash",
                &serde_json::json!({ "command": "pwd" }),
            )),
        ])),
        tool_result("call_1|fc_1", "read", "done"),
    ];

    let result = transform_messages(
        messages,
        &copilot_claude_model(),
        Some(&copilot_normalizer()),
    );
    let synthetic: Vec<&ToolResultMessage> = result
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult(tool_result) if tool_result.is_error => Some(tool_result),
            _ => None,
        })
        .collect();

    assert_eq!(synthetic.len(), 1);
    assert_eq!(synthetic[0].tool_call_id, "call_2_fc_2");
    assert_eq!(synthetic[0].tool_name, "bash");
    assert_eq!(
        block_text(synthetic[0].content.first()),
        Some("No result provided"),
    );
}

fn assistant_message_of(messages: &[Message]) -> Option<&AssistantMessage> {
    messages.iter().find_map(|message| match message {
        Message::Assistant(assistant) => Some(assistant),
        _ => None,
    })
}

fn tool_call_of(content: &[AssistantBlock]) -> Option<&ToolCall> {
    content.iter().find_map(|block| match block {
        AssistantBlock::ToolCall(tool_call) => Some(tool_call),
        _ => None,
    })
}

fn block_text(block: Option<&ToolResultBlock>) -> Option<&str> {
    match block? {
        ToolResultBlock::Text(text) => Some(text.text.as_str()),
        ToolResultBlock::Image(_) => None,
    }
}

fn tool_result_content_json(content: &[ToolResultBlock]) -> Value {
    Value::Array(
        content
            .iter()
            .map(|block| match block {
                ToolResultBlock::Text(text) => json!({ "type": "text", "text": text.text }),
                ToolResultBlock::Image(image) => json!({ "type": "image", "data": image.data }),
            })
            .collect(),
    )
}

fn image_block() -> ImageContent {
    ImageContent {
        data: "aGk=".to_owned(),
        mime_type: "image/png".to_owned(),
    }
}

#[test]
fn downgrades_images_for_non_vision_models_in_user_blocks_and_tool_results() {
    let messages = vec![
        Message::User(UserMessage {
            content: UserContent::Blocks(vec![
                UserBlock::Text(text_block_value("look")),
                UserBlock::Image(image_block()),
                UserBlock::Image(image_block()),
            ]),
            timestamp: 1,
        }),
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "call_2".to_owned(),
            tool_name: "read".to_owned(),
            content: vec![ToolResultBlock::Image(image_block())],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 1,
        }),
    ];

    let result = transform_messages(messages, &text_only_model(), None);

    // Consecutive images collapse into one placeholder.
    let blocks = user_blocks_of(&result).expect("the blocks content survives the downgrade");
    assert_eq!(blocks.len(), 2);
    assert_eq!(
        user_text_of(blocks.first()).expect("the text block"),
        "look"
    );
    assert_eq!(
        user_text_of(blocks.last()).expect("the placeholder block"),
        "(image omitted: model does not support images)"
    );

    let tool_result = tool_result_at(&result, 1).expect("the tool result stays a tool result");
    assert_eq!(
        block_text(tool_result.content.first()),
        Some("(tool image omitted: model does not support images)"),
    );
}

fn text_block_value(text: &str) -> TextContent {
    TextContent {
        text: text.to_owned(),
        text_signature: None,
    }
}

fn user_blocks_of(messages: &[Message]) -> Option<&[UserBlock]> {
    messages.iter().find_map(|message| match message {
        Message::User(user) => match &user.content {
            UserContent::Blocks(blocks) => Some(blocks.as_slice()),
            UserContent::Text(_) => None,
        },
        _ => None,
    })
}

fn tool_result_at(messages: &[Message], index: usize) -> Option<&ToolResultMessage> {
    match messages.get(index)? {
        Message::ToolResult(result) => Some(result),
        _ => None,
    }
}

fn user_text_of(block: Option<&UserBlock>) -> Option<&str> {
    match block? {
        UserBlock::Text(text) => Some(text.text.as_str()),
        UserBlock::Image(_) => None,
    }
}

#[test]
fn a_hand_authored_placeholder_does_not_stack_with_the_downgraded_run() {
    let messages = vec![Message::User(UserMessage {
        content: UserContent::Blocks(vec![
            UserBlock::Text(text_block_value(
                "(image omitted: model does not support images)",
            )),
            UserBlock::Image(image_block()),
            UserBlock::Text(text_block_value("after")),
        ]),
        timestamp: 1,
    })];

    let result = transform_messages(messages, &text_only_model(), None);

    let blocks = user_blocks_of(&result).expect("the user turn keeps its blocks");
    assert_eq!(blocks.len(), 2);
    assert_eq!(
        user_text_of(blocks.first()).expect("the placeholder block"),
        "(image omitted: model does not support images)"
    );
    assert_eq!(
        user_text_of(blocks.last()).expect("the after block"),
        "after"
    );
}

#[test]
fn redacted_thinking_drops_across_models_and_survives_the_origin() {
    let redacted = ThinkingContent {
        thinking: "opaque".to_owned(),
        thinking_signature: Some("enc-payload".to_owned()),
        redacted: Some(true),
    };

    let cross_model = transform_messages(
        vec![Message::Assistant(openai_assistant(vec![
            AssistantBlock::Thinking(redacted.clone()),
        ]))],
        &copilot_claude_model(),
        Some(&copilot_normalizer()),
    );
    let assistant = assistant_message_of(&cross_model).expect("assistant message");
    assert!(
        !matches!(assistant.content.first(), Some(AssistantBlock::Thinking(_))),
        "the redacted block dropped across models"
    );

    let same_model_source = AssistantMessage {
        content: vec![AssistantBlock::Thinking(redacted)],
        api: Api::from("anthropic-messages"),
        provider: ProviderId::from("github-copilot"),
        model: "claude-sonnet-4.6".to_owned(),
        ..openai_assistant(Vec::new())
    };
    let same_model = transform_messages(
        vec![Message::Assistant(same_model_source)],
        &copilot_claude_model(),
        Some(&copilot_normalizer()),
    );
    let assistant = assistant_message_of(&same_model).expect("assistant message");
    assert!(matches!(
        assistant.content.first(),
        Some(AssistantBlock::Thinking(_))
    ));
}

#[test]
fn drops_empty_unsigned_thinking_and_errored_assistant_turns() {
    let messages = vec![
        user_message("hello"),
        Message::Assistant(openai_assistant(vec![AssistantBlock::Thinking(
            ThinkingContent {
                thinking: String::new(),
                thinking_signature: None,
                redacted: None,
            },
        )])),
        Message::Assistant(AssistantMessage {
            content: vec![AssistantBlock::Text(text_block_value("partial"))],
            stop_reason: StopReason::Error,
            error_message: Some("failed".to_owned()),
            ..openai_assistant(Vec::new())
        }),
    ];

    let result = transform_messages(messages, &copilot_claude_model(), None);

    // The empty unsigned thinking block leaves the assistant turn contentless
    // but kept; the errored turn is skipped entirely.
    assert_eq!(result.len(), 2);
    assert!(matches!(result[0], Message::User(_)));
    let assistant = assistant_message_of(&result).expect("the empty assistant turn survives");
    assert!(assistant.content.is_empty());
}

#[test]
fn lenient_entries_drop_untyped_values_and_keep_typed_content() {
    let messages = vec![
        json!(7),
        json!({ "role": "user", "content": "hi", "timestamp": 1 }),
    ];

    let result = transform_messages_lenient(&messages, &text_only_model(), None);

    assert_eq!(result.len(), 1);
    assert!(matches!(
        &result[0],
        Message::User(UserMessage { content: UserContent::Text(text), .. }) if text == "hi"
    ));
}

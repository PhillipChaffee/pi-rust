//! The context-token estimate port, from `test/context-estimate.test.ts`
//! plus the estimate case of `test/deferred-tools.test.ts`.

mod common;

use common::{assistant_message, usage, user_message};
use pi_ai::types::{
    Context, Message, StopReason, TextContent, Tool, ToolResultBlock, ToolResultMessage,
};
use pi_ai::utils::estimate::{
    estimate_context_tokens, estimate_message_tokens, estimate_messages_tokens,
};
use serde_json::json;

fn create_assistant(timestamp: i64, total_tokens: u64) -> Message {
    let mut message = assistant_message("kept");
    message.api = pi_ai::types::Api::from("openai-responses");
    message.provider = pi_ai::types::ProviderId::from("openai");
    message.timestamp = timestamp;
    message.usage = usage(total_tokens);
    Message::Assistant(message)
}

#[test]
fn ignores_stale_assistant_usage_after_a_newer_message_is_inserted_before_it() {
    let context = Context {
        system_prompt: Some(String::from("system")),
        messages: vec![
            user_message("summary", 200),
            create_assistant(100, 9_500),
            user_message(&"x".repeat(4_000), 300),
        ],
        tools: None,
    };

    let estimate = estimate_context_tokens(&context);
    assert_eq!(estimate.tokens, 1_005);
    assert_eq!(estimate.usage_tokens, 0);
    assert_eq!(estimate.trailing_tokens, 1_005);
    assert_eq!(estimate.last_usage_index, None);
    // The maxTokens clamp of buildBaseOptions rides with the
    // simple-options child; only the estimate half ports here.
}

#[test]
fn uses_assistant_usage_again_after_a_response_to_the_inserted_context() {
    let context = Context {
        system_prompt: None,
        messages: vec![
            user_message("summary", 200),
            create_assistant(100, 9_500),
            user_message("new prompt", 300),
            create_assistant(400, 2_000),
            user_message("tail", 500),
        ],
        tools: None,
    };

    let estimate = estimate_context_tokens(&context);
    assert_eq!(estimate.tokens, 2_001);
    assert_eq!(estimate.usage_tokens, 2_000);
    assert_eq!(estimate.trailing_tokens, 1);
    assert_eq!(estimate.last_usage_index, Some(3));
}

#[test]
fn the_bare_message_list_form_matches_the_context_form() {
    let messages = vec![
        user_message("summary", 200),
        create_assistant(100, 9_500),
        user_message("x", 300),
    ];
    let bare = estimate_messages_tokens(&messages);
    let in_context = estimate_context_tokens(&Context {
        system_prompt: None,
        messages,
        tools: None,
    });
    assert_eq!(bare.tokens, in_context.tokens);
    assert_eq!(bare.last_usage_index, in_context.last_usage_index);
}

#[test]
fn counts_definitions_marked_after_the_latest_usage_checkpoint() {
    let mut assistant = assistant_message("done");
    assistant.stop_reason = StopReason::Stop;
    assistant.usage = usage(100);
    assistant.usage.input = 50;
    assistant.usage.output = 50;
    let plain = estimate_context_tokens(&Context {
        system_prompt: None,
        messages: vec![Message::Assistant(assistant.clone()), user_message("x", 4)],
        tools: Some(Vec::new()),
    });
    let late_tool = Tool {
        name: String::from("late_tool"),
        description: "x".repeat(4_000),
        parameters: json!({}),
        constrained_sampling: None,
    };
    let marked = estimate_context_tokens(&Context {
        system_prompt: None,
        messages: vec![
            Message::Assistant(assistant),
            tool_result(vec![String::from("late_tool")]),
        ],
        tools: Some(vec![late_tool]),
    });

    assert!(marked.tokens > plain.tokens + 500);
    assert!(marked.trailing_tokens > plain.trailing_tokens + 500);
}

fn tool_result(added_tool_names: Vec<String>) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: String::from("call_1"),
        tool_name: String::from("base_tool"),
        content: vec![ToolResultBlock::Text(TextContent {
            text: String::from("done"),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: Some(added_tool_names),
        is_error: false,
        timestamp: 3,
    })
}

#[test]
fn message_token_estimates_count_characters_per_token() {
    let message = assistant_message("abcd");
    let estimate = estimate_message_tokens(&Message::Assistant(message));
    assert_eq!(estimate, 1, "four characters at four chars per token");
}

#[test]
fn the_component_sum_fills_in_when_usage_reports_no_total() {
    use pi_ai::utils::estimate::calculate_context_tokens;

    let reported = usage(120);
    assert_eq!(calculate_context_tokens(&reported), 120);
    let mut components = usage(0);
    components.input = 50;
    components.output = 60;
    components.cache_read = 30;
    components.cache_write = 20;
    assert_eq!(calculate_context_tokens(&components), 160);
}

#[test]
fn text_content_estimates_by_character_count() {
    use pi_ai::utils::estimate::estimate_text_and_image_content_tokens;
    use pi_ai::utils::estimate::estimate_text_tokens;

    // Four characters per token, rounded up.
    assert_eq!(estimate_text_tokens("abcd"), 1);
    assert_eq!(estimate_text_tokens("abcde"), 2);
    assert_eq!(estimate_text_tokens(""), 0);
    assert_eq!(estimate_text_and_image_content_tokens("abcde"), 2);
}

#[test]
fn user_content_counts_images_at_the_fixed_character_weight() {
    use pi_ai::types::{ImageContent, UserBlock, UserContent};
    use pi_ai::utils::estimate::estimate_text_and_image_content_tokens;

    let image = ImageContent {
        data: String::from("..."),
        mime_type: String::from("image/png"),
    };
    // 4800 image chars / 4 = 1200 tokens, text counted separately.
    let blocks = vec![
        UserBlock::Text(common::text_block("ab")),
        UserBlock::Image(image.clone()),
    ];
    assert_eq!(
        estimate_text_and_image_content_tokens(blocks.as_slice()),
        1_201
    );
    let only_image = vec![UserBlock::Image(image)];
    assert_eq!(
        estimate_text_and_image_content_tokens(only_image.as_slice()),
        1_200
    );
    assert_eq!(
        estimate_text_and_image_content_tokens(&UserContent::Text(String::from("abc"))),
        1
    );
    assert_eq!(
        estimate_text_and_image_content_tokens(&UserContent::Blocks(blocks)),
        1_201
    );
}

#[test]
fn tool_result_content_counts_text_and_images() {
    use pi_ai::types::{ImageContent, ToolResultBlock, ToolResultMessage};
    use pi_ai::utils::estimate::estimate_message_tokens;

    let result = ToolResultMessage {
        tool_call_id: String::from("call_1"),
        tool_name: String::from("tool"),
        content: vec![
            ToolResultBlock::Text(TextContent {
                text: String::from("abcd"),
                text_signature: None,
            }),
            ToolResultBlock::Image(ImageContent {
                data: String::from("..."),
                mime_type: String::from("image/png"),
            }),
        ],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 3,
    };
    assert_eq!(estimate_message_tokens(&Message::ToolResult(result)), 1_201);
}

#[test]
fn assistant_messages_estimate_thinking_and_tool_call_payloads() {
    use pi_ai::types::{AssistantBlock, ThinkingContent, ToolCall};
    use pi_ai::utils::estimate::estimate_message_tokens;
    use serde_json::json;

    let mut message = common::bare_assistant_message();
    message
        .content
        .push(AssistantBlock::Thinking(ThinkingContent {
            thinking: String::from("abcd"),
            thinking_signature: None,
            redacted: None,
        }));
    let mut arguments = serde_json::Map::new();
    arguments.insert(String::from("path"), json!("ab"));
    message.content.push(AssistantBlock::ToolCall(ToolCall {
        id: String::from("call_1"),
        name: String::from("read"),
        arguments,
        thought_signature: None,
        namespace: None,
    }));
    // thinking 4 chars + name 4 chars + serialized arguments 13 chars
    // ({"path":"ab"}) = 21 chars / 4 = 6 tokens.
    assert_eq!(estimate_message_tokens(&Message::Assistant(message)), 6);
}

//! The summary serialization suite, upstream
//! `test/compaction-serialization.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported 1:1.

use pi_ai::types::{
    AssistantMessage, Message, StopReason, TextContent, ToolResultBlock, ToolResultMessage,
    UserBlock, UserContent, UserMessage,
};

use pi_coding_agent::compaction::serialize_conversation;

fn tool_result(content_text: &str) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: "tc1".to_owned(),
        tool_name: "read".to_owned(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: content_text.to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 0,
    })
}

#[test]
fn truncates_long_tool_results() {
    let long_content = "x".repeat(5000);
    let messages = vec![tool_result(&long_content)];

    let result = serialize_conversation(&messages);

    assert!(result.contains("[Tool result]:"));
    assert!(result.contains("[... 3000 more characters truncated]"));
    assert!(!result.contains(&"x".repeat(3000)));
    // First 2000 chars should be present.
    assert!(result.contains(&"x".repeat(2000)));
}

#[test]
fn does_not_truncate_short_tool_results() {
    let short_content = "x".repeat(1500);
    let messages = vec![tool_result(&short_content)];

    let result = serialize_conversation(&messages);

    assert_eq!(result, format!("[Tool result]: {short_content}"));
    assert!(!result.contains("truncated"));
}

#[test]
fn does_not_truncate_assistant_or_user_messages() {
    let long_text = "y".repeat(5000);
    let messages = vec![
        Message::User(UserMessage {
            content: UserContent::Blocks(vec![UserBlock::Text(TextContent {
                text: long_text.clone(),
                text_signature: None,
            })]),
            timestamp: 0,
        }),
        Message::Assistant(AssistantMessage {
            content: vec![pi_ai::types::AssistantBlock::Text(TextContent {
                text: long_text.clone(),
                text_signature: None,
            })],
            api: pi_ai::types::Api::from(pi_ai::types::KnownApi::AnthropicMessages),
            provider: pi_ai::types::ProviderId("anthropic".to_owned()),
            model: "test".to_owned(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            diagnostics: None,
            usage: pi_ai::types::Usage {
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                cache_write_1h: None,
                reasoning: None,
                total_tokens: 0,
                cost: pi_ai::types::UsageCost {
                    input: 0.0,
                    output: 0.0,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0,
                },
            },
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        }),
    ];

    let result = serialize_conversation(&messages);

    assert!(!result.contains("truncated"));
    assert!(result.contains(&long_text));
}

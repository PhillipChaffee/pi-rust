//! Shared test fixtures for the utils belt suites, mirroring the shapes
//! upstream's `fauxAssistantMessage` and message helpers feed the same
//! code paths.

#![expect(
    dead_code,
    reason = "shared fixtures; each test binary uses the subset it needs"
)]
#![expect(
    unreachable_pub,
    reason = "the fixture module is compiled into every integration test binary as a private module"
)]

use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, Message, ProviderId, StopReason, TextContent, Usage,
    UsageCost, UserContent, UserMessage,
};

/// A provider-shaped usage block with the given total.
#[must_use]
pub fn usage(total_tokens: u64) -> Usage {
    Usage {
        input: total_tokens,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens,
        cost: UsageCost::default(),
    }
}

/// An assistant message carrying one text block, the faux provider's
/// success shape.
#[must_use]
pub fn assistant_message(text: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: Api::from("test-api"),
        provider: ProviderId::from("test-provider"),
        model: String::from("test-model"),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

/// An assistant message with overridden stop reason and error text.
#[must_use]
pub fn assistant_message_with(
    text: &str,
    stop_reason: StopReason,
    error_message: Option<String>,
) -> AssistantMessage {
    let mut message = assistant_message(text);
    message.stop_reason = stop_reason;
    message.error_message = error_message;
    message
}

/// A minimal assistant message with no content, for error/usage shapes.
#[must_use]
pub fn bare_assistant_message() -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: Api::from("test-api"),
        provider: ProviderId::from("test-provider"),
        model: String::from("test-model"),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

/// A user message with a plain-string content.
#[must_use]
pub fn user_message(text: &str, timestamp: i64) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp,
    })
}

/// A text block for content comparisons.
#[must_use]
pub fn text_block(text: &str) -> TextContent {
    TextContent {
        text: text.to_owned(),
        text_signature: None,
    }
}

/// An assistant message with only stop reason set, for retry shapes that
/// check the reason alone.
#[must_use]
pub fn aborted_message() -> AssistantMessage {
    assistant_message_with("", StopReason::Aborted, None)
}

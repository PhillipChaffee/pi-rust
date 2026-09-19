//! Context-token estimation, ported from
//! `packages/ai/src/utils/estimate.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! A cheap ceiling-free estimate of the context a request carries: the most
//! recent applicable assistant usage block anchors the total, trailing
//! messages after it are estimated from character counts, and the system
//! prompt, tool definitions, and tools loaded after the anchor add to the
//! trailing estimate.

use serde_json::Value;

use crate::types::{
    AssistantBlock, Context, Message, StopReason, Tool, ToolResultBlock, Usage, UserBlock,
    UserContent,
};
use crate::utils::error_body::safe_json_stringify;

/// The per-message context estimate: the anchored total, the anchored usage,
/// the estimated trailing tokens, and the anchor's index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextUsageEstimate {
    /// Estimated total context tokens.
    pub tokens: u64,
    /// Tokens reported by the most recent applicable assistant usage block.
    pub usage_tokens: u64,
    /// Estimated tokens after the most recent applicable assistant usage
    /// block.
    pub trailing_tokens: u64,
    /// Index of the applicable message that provided usage, or [`None`] when
    /// none exists.
    pub last_usage_index: Option<usize>,
}

const CHARS_PER_TOKEN: f64 = 4.0;
const ESTIMATED_IMAGE_CHARS: usize = 4800;

/// The context tokens a usage block reports: the provider's total when it
/// sent one, the component sum otherwise.
#[must_use]
pub const fn calculate_context_tokens(usage: &Usage) -> u64 {
    if usage.total_tokens > 0 {
        usage.total_tokens
    } else {
        usage.input + usage.output + usage.cache_read + usage.cache_write
    }
}

/// Estimate the token count of a text span from its character count.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "the chars-count / 4 ceil reproduces the JS number estimate; a non-negative count's ceil is non-negative and fits u64"
)]
#[must_use]
pub fn estimate_text_tokens(text: &str) -> u64 {
    (text.chars().count() as f64 / CHARS_PER_TOKEN).ceil() as u64
}

/// Content a text-and-image estimate reads: a plain string or blocks
/// carrying text and images, upstream's `string | (TextContent |
/// ImageContent)[]`.
pub trait TextImageContent {
    /// The character weight of this content, images at
    /// 4800 characters, the `ESTIMATED_IMAGE_CHARS` weight.
    fn text_and_image_chars(&self) -> usize;
}

impl TextImageContent for str {
    fn text_and_image_chars(&self) -> usize {
        self.chars().count()
    }
}

impl TextImageContent for UserContent {
    fn text_and_image_chars(&self) -> usize {
        match self {
            Self::Text(text) => text.chars().count(),
            Self::Blocks(blocks) => blocks.as_slice().text_and_image_chars(),
        }
    }
}

impl TextImageContent for [UserBlock] {
    fn text_and_image_chars(&self) -> usize {
        self.iter()
            .map(|block| match block {
                UserBlock::Text(text) => text.text.chars().count(),
                UserBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum()
    }
}

impl TextImageContent for [ToolResultBlock] {
    fn text_and_image_chars(&self) -> usize {
        self.iter()
            .map(|block| match block {
                ToolResultBlock::Text(text) => text.text.chars().count(),
                ToolResultBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum()
    }
}

/// Estimate the token count of text content, with images counted at
/// 4800 characters.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "the chars-count / 4 ceil reproduces the JS number estimate; a non-negative count's ceil is non-negative and fits u64"
)]
#[must_use]
pub fn estimate_text_and_image_content_tokens<T: TextImageContent + ?Sized>(content: &T) -> u64 {
    (content.text_and_image_chars() as f64 / CHARS_PER_TOKEN).ceil() as u64
}

/// Estimate the token count of one message from its character content.
#[must_use]
pub fn estimate_message_tokens(message: &Message) -> u64 {
    match message {
        Message::User(user) => estimate_text_and_image_content_tokens(&user.content),
        Message::ToolResult(result) => {
            estimate_text_and_image_content_tokens(result.content.as_slice())
        }
        Message::Assistant(assistant) => {
            let mut chars = 0usize;
            for block in &assistant.content {
                match block {
                    AssistantBlock::Text(text) => chars += text.text.chars().count(),
                    AssistantBlock::Thinking(thinking) => {
                        chars += thinking.thinking.chars().count();
                    }
                    AssistantBlock::ToolCall(tool_call) => {
                        chars += tool_call.name.chars().count();
                        chars +=
                            safe_json_stringify(&Value::Object(tool_call.arguments.clone())).len();
                    }
                }
            }
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                reason = "the chars-count / 4 ceil reproduces the JS number estimate; a non-negative count's ceil is non-negative and fits u64"
            )]
            let tokens = (chars as f64 / CHARS_PER_TOKEN).ceil() as u64;
            tokens
        }
    }
}

const fn message_timestamp(message: &Message) -> i64 {
    match message {
        Message::User(user) => user.timestamp,
        Message::Assistant(assistant) => assistant.timestamp,
        Message::ToolResult(result) => result.timestamp,
    }
}

/// The most recent assistant usage block that still describes the current
/// message prefix, with its index. A newer prefix message inserted after a
/// response — for example a compaction summary — invalidates that response's
/// usage for the current prefix.
fn get_last_assistant_usage_info(messages: &[Message]) -> Option<(&Usage, usize)> {
    let mut latest_prefix_timestamp = i64::MIN;
    let mut usage_info: Option<(&Usage, usize)> = None;

    for (index, message) in messages.iter().enumerate() {
        if let Message::Assistant(assistant) = message {
            let usage_applies_to_prefix = assistant.timestamp >= latest_prefix_timestamp;
            if usage_applies_to_prefix
                && assistant.stop_reason != StopReason::Aborted
                && assistant.stop_reason != StopReason::Error
                && calculate_context_tokens(&assistant.usage) > 0
            {
                usage_info = Some((&assistant.usage, index));
            }
        }
        latest_prefix_timestamp = latest_prefix_timestamp.max(message_timestamp(message));
    }

    usage_info
}

fn estimate_messages(messages: &[Message]) -> ContextUsageEstimate {
    if let Some((usage, index)) = get_last_assistant_usage_info(messages) {
        let usage_tokens = calculate_context_tokens(usage);
        let mut trailing_tokens = 0u64;
        for message in &messages[index + 1..] {
            trailing_tokens += estimate_message_tokens(message);
        }
        return ContextUsageEstimate {
            tokens: usage_tokens + trailing_tokens,
            usage_tokens,
            trailing_tokens,
            last_usage_index: Some(index),
        };
    }

    let mut tokens = 0u64;
    for message in messages {
        tokens += estimate_message_tokens(message);
    }
    ContextUsageEstimate {
        tokens,
        usage_tokens: 0,
        trailing_tokens: tokens,
        last_usage_index: None,
    }
}

fn estimate_tools_tokens(tools: Option<&[Tool]>) -> u64 {
    let Some(tools) = tools else {
        return 0;
    };
    if tools.is_empty() {
        return 0;
    }
    // serde_json serialization of plain data cannot fail; the fallback
    // covers the impossible path upstream guards with try/catch.
    let serialized =
        serde_json::to_string(tools).unwrap_or_else(|_| String::from("[unserializable]"));
    estimate_text_tokens(&serialized)
}

/// Estimate the context tokens of a bare message list, upstream's
/// `estimateContextTokens(messages)` array branch.
#[must_use]
pub fn estimate_messages_tokens(messages: &[Message]) -> ContextUsageEstimate {
    estimate_messages(messages)
}

/// Estimate the context tokens of a conversation: message estimation plus
/// the system prompt and tool definitions, and the tool definitions loaded
/// by tool results after the usage anchor.
#[must_use]
pub fn estimate_context_tokens(context: &Context) -> ContextUsageEstimate {
    let estimate = estimate_messages(&context.messages);

    if let Some(last_usage_index) = estimate.last_usage_index {
        let added_names: std::collections::BTreeSet<String> = context.messages
            [last_usage_index + 1..]
            .iter()
            .filter_map(|message| match message {
                Message::ToolResult(result) => result.added_tool_names.as_ref(),
                _ => None,
            })
            .flatten()
            .cloned()
            .collect();
        let added_tools = context.tools.as_ref().map(|tools| {
            tools
                .iter()
                .filter(|tool| added_names.contains(&tool.name))
                .cloned()
                .collect::<Vec<_>>()
        });
        let added_tool_tokens = estimate_tools_tokens(added_tools.as_deref());
        return ContextUsageEstimate {
            tokens: estimate.tokens + added_tool_tokens,
            usage_tokens: estimate.usage_tokens,
            trailing_tokens: estimate.trailing_tokens + added_tool_tokens,
            last_usage_index: estimate.last_usage_index,
        };
    }

    let prefix_tokens = context
        .system_prompt
        .as_ref()
        .map_or(0, |prompt| estimate_text_tokens(prompt))
        + estimate_tools_tokens(context.tools.as_deref());

    ContextUsageEstimate {
        tokens: estimate.tokens + prefix_tokens,
        usage_tokens: estimate.usage_tokens,
        trailing_tokens: estimate.trailing_tokens + prefix_tokens,
        last_usage_index: estimate.last_usage_index,
    }
}

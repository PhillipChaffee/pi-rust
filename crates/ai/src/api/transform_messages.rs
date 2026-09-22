//! Cross-provider message rewriting, ported from
//! `packages/ai/src/api/transform-messages.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The choke point every wire API runs before dispatching a request: image
//! downgrade for non-vision models, thinking-block replay rules, tool-call ID
//! normalization, and synthetic tool results for orphaned calls.
//!
//! Porting restatement: upstream also normalizes `content: null` and missing
//! content from untyped callers (issues pi #6259, #6276). A Rust
//! [`Message`] cannot hold a null content, so that normalization rides the
//! JSON boundary of [`transform_messages_lenient`]; the typed entry assumes
//! the invariant the type system upholds.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::auth::resolve::now_ms;
use crate::types::{
    AssistantBlock, AssistantMessage, ImageContent, Message, Modality, Model, TextContent,
    ToolCall, ToolResultBlock, ToolResultMessage, UserBlock,
};

/// The placeholder a non-vision user message's image downgrades to.
const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
/// The placeholder a non-vision tool-result image downgrades to.
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str =
    "(tool image omitted: model does not support images)";

/// The normalizer cross-provider session migration passes in, upstream's
/// `normalizeToolCallId?: (id, model, source) => string`.
pub type ToolCallIdNormalizer<'a> = dyn Fn(&str, &Model, &AssistantMessage) -> String + 'a;

/// The content shapes [`replace_images_with_placeholder`] walks: text or an
/// image, the block pair [`UserBlock`] and [`ToolResultBlock`] share.
#[derive(Clone)]
enum ContentBlock {
    Text(TextContent),
    Image(ImageContent),
}

impl From<UserBlock> for ContentBlock {
    fn from(block: UserBlock) -> Self {
        match block {
            UserBlock::Text(text) => Self::Text(text),
            UserBlock::Image(image) => Self::Image(image),
        }
    }
}

impl From<ContentBlock> for UserBlock {
    fn from(block: ContentBlock) -> Self {
        match block {
            ContentBlock::Text(text) => Self::Text(text),
            ContentBlock::Image(image) => Self::Image(image),
        }
    }
}

impl From<ToolResultBlock> for ContentBlock {
    fn from(block: ToolResultBlock) -> Self {
        match block {
            ToolResultBlock::Text(text) => Self::Text(text),
            ToolResultBlock::Image(image) => Self::Image(image),
        }
    }
}

impl From<ContentBlock> for ToolResultBlock {
    fn from(block: ContentBlock) -> Self {
        match block {
            ContentBlock::Text(text) => Self::Text(text),
            ContentBlock::Image(image) => Self::Image(image),
        }
    }
}

/// Replace image blocks with a text placeholder, collapsing consecutive
/// images into one placeholder, upstream's `replaceImagesWithPlaceholder`.
/// A text block equal to the placeholder keeps the run going, so a
/// hand-authored placeholder text does not stack.
fn replace_images_with_placeholder(
    content: Vec<ContentBlock>,
    placeholder: &str,
) -> Vec<ContentBlock> {
    let mut result = Vec::new();
    let mut previous_was_placeholder = false;
    for block in content {
        match block {
            ContentBlock::Image(_) => {
                if !previous_was_placeholder {
                    result.push(ContentBlock::Text(TextContent {
                        text: placeholder.to_owned(),
                        text_signature: None,
                    }));
                }
                previous_was_placeholder = true;
            }
            ContentBlock::Text(text) => {
                previous_was_placeholder = text.text == placeholder;
                result.push(ContentBlock::Text(text));
            }
        }
    }
    result
}

fn downgrade_unsupported_images(messages: Vec<Message>, model: &Model) -> Vec<Message> {
    if model.input.contains(&Modality::Image) {
        return messages;
    }
    messages
        .into_iter()
        .map(|message| match message {
            Message::User(mut user) => {
                if let crate::types::UserContent::Blocks(blocks) = user.content {
                    let downgraded = replace_images_with_placeholder(
                        blocks.into_iter().map(ContentBlock::from).collect(),
                        NON_VISION_USER_IMAGE_PLACEHOLDER,
                    );
                    user.content = crate::types::UserContent::Blocks(
                        downgraded.into_iter().map(UserBlock::from).collect(),
                    );
                }
                Message::User(user)
            }
            Message::ToolResult(mut result) => {
                let downgraded = replace_images_with_placeholder(
                    std::mem::take(&mut result.content)
                        .into_iter()
                        .map(ContentBlock::from)
                        .collect(),
                    NON_VISION_TOOL_IMAGE_PLACEHOLDER,
                );
                result.content = downgraded.into_iter().map(ToolResultBlock::from).collect();
                Message::ToolResult(result)
            }
            Message::Assistant(_) => message,
        })
        .collect()
}

/// Rewrite a conversation for the model about to receive it.
///
/// Normalizes tool-call IDs through `normalize_tool_call_id` when the source
/// model differs, downgrades images the model cannot see, drops or converts
/// thinking blocks per the replay rules, skips errored and aborted assistant
/// turns, and synthesizes `"No result provided"` tool results for calls the
/// conversation never answered.
///
/// The replay rules: redacted thinking survives only on its originating
/// model; signed thinking survives replay on the same model; empty or
/// unsigned thinking becomes plain text across models; tool calls lose
/// `thoughtSignature` across models and get their IDs normalized.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "the two passes mirror upstream's transformMessages one to one; the synthetic-result pass reads against it"
)]
pub fn transform_messages(
    messages: Vec<Message>,
    model: &Model,
    normalize_tool_call_id: Option<&ToolCallIdNormalizer<'_>>,
) -> Vec<Message> {
    let mut tool_call_id_map: BTreeMap<String, String> = BTreeMap::new();
    let image_aware_messages = downgrade_unsupported_images(messages, model);

    let transformed: Vec<Message> = image_aware_messages
        .into_iter()
        .map(|message| match message {
            Message::User(_) => message,
            Message::ToolResult(result) => {
                let Some(normalized_id) = tool_call_id_map.get(&result.tool_call_id) else {
                    return Message::ToolResult(result);
                };
                if *normalized_id == result.tool_call_id {
                    return Message::ToolResult(result);
                }
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: normalized_id.clone(),
                    ..result
                })
            }
            Message::Assistant(assistant) => {
                // Upstream compares the source provider/api/model ids; the
                // nursery's operator-grouping heuristic misreads
                // `assistant.model == model.id` as a typo.
                #[allow(
                    clippy::suspicious_operation_groupings,
                    reason = "the same-model check mirrors upstream's field-for-field comparison"
                )]
                let is_same_model = assistant.provider == model.provider
                    && assistant.api == model.api
                    && assistant.model == model.id;

                let content = assistant
                    .content
                    .iter()
                    .filter_map(|block| {
                        match block {
                            AssistantBlock::Thinking(thinking) => {
                                // Redacted thinking is opaque encrypted content, only
                                // valid for the same model; drop it cross-model.
                                if thinking.redacted.unwrap_or(false) {
                                    return is_same_model
                                        .then(|| AssistantBlock::Thinking(thinking.clone()));
                                }
                                // Same model: keep signed thinking even when the text
                                // is empty (OpenAI encrypted reasoning replays here).
                                if is_same_model
                                    && thinking
                                        .thinking_signature
                                        .as_deref()
                                        .is_some_and(|signature| !signature.is_empty())
                                {
                                    return Some(AssistantBlock::Thinking(thinking.clone()));
                                }
                                if thinking.thinking.trim().is_empty() {
                                    return None;
                                }
                                if is_same_model {
                                    return Some(AssistantBlock::Thinking(thinking.clone()));
                                }
                                Some(AssistantBlock::Text(TextContent {
                                    text: thinking.thinking.clone(),
                                    text_signature: None,
                                }))
                            }
                            AssistantBlock::Text(text) => {
                                if is_same_model {
                                    Some(AssistantBlock::Text(text.clone()))
                                } else {
                                    Some(AssistantBlock::Text(TextContent {
                                        text: text.text.clone(),
                                        text_signature: None,
                                    }))
                                }
                            }
                            AssistantBlock::ToolCall(tool_call) => {
                                let mut normalized = tool_call.clone();
                                if !is_same_model && normalized.thought_signature.is_some() {
                                    normalized.thought_signature = None;
                                }
                                if !is_same_model && let Some(normalize) = normalize_tool_call_id {
                                    let normalized_id = normalize(&tool_call.id, model, &assistant);
                                    if normalized_id != tool_call.id {
                                        tool_call_id_map
                                            .insert(tool_call.id.clone(), normalized_id.clone());
                                        normalized.id = normalized_id;
                                    }
                                }
                                Some(AssistantBlock::ToolCall(normalized))
                            }
                        }
                    })
                    .collect();

                Message::Assistant(AssistantMessage {
                    content,
                    ..assistant
                })
            }
        })
        .collect();

    // Second pass: insert synthetic empty tool results for orphaned calls.
    // This preserves thinking signatures and satisfies API requirements.
    let mut result: Vec<Message> = Vec::new();
    let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
    let mut existing_tool_result_ids: BTreeSet<String> = BTreeSet::new();

    for message in transformed {
        match message {
            Message::Assistant(assistant) => {
                insert_synthetic_tool_results(
                    &mut result,
                    &pending_tool_calls,
                    &existing_tool_result_ids,
                );
                pending_tool_calls.clear();
                existing_tool_result_ids.clear();

                // Errored and aborted assistant turns are incomplete: their
                // partial content would replay as API errors, so the model
                // retries from the last valid state.
                if matches!(
                    assistant.stop_reason,
                    crate::types::StopReason::Error | crate::types::StopReason::Aborted
                ) {
                    continue;
                }

                let tool_calls = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantBlock::ToolCall(tool_call) => Some(tool_call.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if !tool_calls.is_empty() {
                    pending_tool_calls = tool_calls;
                    existing_tool_result_ids = BTreeSet::new();
                }
                result.push(Message::Assistant(assistant));
            }
            Message::ToolResult(tool_result) => {
                existing_tool_result_ids.insert(tool_result.tool_call_id.clone());
                result.push(Message::ToolResult(tool_result));
            }
            Message::User(user) => {
                insert_synthetic_tool_results(
                    &mut result,
                    &pending_tool_calls,
                    &existing_tool_result_ids,
                );
                pending_tool_calls.clear();
                existing_tool_result_ids.clear();
                result.push(Message::User(user));
            }
        }
    }
    insert_synthetic_tool_results(&mut result, &pending_tool_calls, &existing_tool_result_ids);

    result
}

fn insert_synthetic_tool_results(
    result: &mut Vec<Message>,
    pending_tool_calls: &[ToolCall],
    existing_tool_result_ids: &BTreeSet<String>,
) {
    for call in pending_tool_calls {
        if !existing_tool_result_ids.contains(&call.id) {
            result.push(Message::ToolResult(ToolResultMessage {
                tool_call_id: call.id.clone(),
                tool_name: call.name.clone(),
                content: vec![ToolResultBlock::Text(TextContent {
                    text: "No result provided".to_owned(),
                    text_signature: None,
                })],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: true,
                timestamp: now_ms(),
            }));
        }
    }
}

/// The lenient entry for untyped callers, upstream's lax
/// `transformMessages` behavior.
///
/// Deserializes raw JSON messages, normalizing the wire's `content: null`
/// and missing `content` to empty arrays before the rewrite runs (issues pi
/// #6259, #6276). Messages that do not carry a parseable role are dropped —
/// upstream kept them verbatim only because its runtime types are erased.
#[must_use]
pub fn transform_messages_lenient(
    messages: &[Value],
    model: &Model,
    normalize_tool_call_id: Option<&ToolCallIdNormalizer<'_>>,
) -> Vec<Message> {
    let lenient_messages: Vec<Message> = messages
        .iter()
        .filter_map(|message| {
            let mut lenient = message.clone();
            if !lenient.is_object() {
                return None;
            }
            match lenient.get_mut("content") {
                Some(content) if content.is_null() => *content = Value::Array(Vec::new()),
                Some(_) => {}
                None => {
                    lenient["content"] = Value::Array(Vec::new());
                }
            }
            serde_json::from_value(lenient).ok()
        })
        .collect();

    transform_messages(lenient_messages, model, normalize_tool_call_id)
}

//! GitHub Copilot request headers, ported from
//! `packages/ai/src/api/github-copilot-headers.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::collections::BTreeMap;

use crate::types::{Message, ToolResultBlock, UserBlock, UserContent};

/// Copilot expects `X-Initiator` to indicate whether the request is
/// user-initiated or agent-initiated (a follow-up after assistant or tool
/// messages), upstream's `inferCopilotInitiator`.
#[must_use]
pub fn infer_copilot_initiator(messages: &[Message]) -> &'static str {
    if messages
        .last()
        .is_some_and(|last| !matches!(last, Message::User(_)))
    {
        "agent"
    } else {
        "user"
    }
}

/// Copilot requires the `Copilot-Vision-Request` header when the
/// conversation carries images, upstream's `hasCopilotVisionInput`.
#[must_use]
pub fn has_copilot_vision_input(messages: &[Message]) -> bool {
    messages.iter().any(|message| match message {
        Message::User(user) => matches!(
            &user.content,
            UserContent::Blocks(blocks)
                if blocks.iter().any(|block| matches!(block, UserBlock::Image(_)))
        ),
        Message::ToolResult(result) => result
            .content
            .iter()
            .any(|block| matches!(block, ToolResultBlock::Image(_))),
        Message::Assistant(_) => false,
    })
}

/// The per-request Copilot headers, upstream's `buildCopilotDynamicHeaders`.
#[must_use]
pub fn build_copilot_dynamic_headers(
    messages: &[Message],
    has_images: bool,
) -> BTreeMap<String, String> {
    let mut headers = BTreeMap::new();
    headers.insert(
        "X-Initiator".to_owned(),
        infer_copilot_initiator(messages).to_owned(),
    );
    headers.insert("Openai-Intent".to_owned(), "conversation-edits".to_owned());
    if has_images {
        headers.insert("Copilot-Vision-Request".to_owned(), "true".to_owned());
    }
    headers
}

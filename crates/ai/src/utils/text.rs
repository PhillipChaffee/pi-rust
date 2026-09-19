//! Text extraction from message content, ported from
//! `packages/ai/src/utils/text.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::types::{AssistantBlock, ToolResultBlock, UserBlock, UserContent};

/// Content shapes [`content_text`] reads: a plain string or a block list.
pub trait ContentTextSource {
    /// The text blocks of this content, in order.
    fn text_parts(&self) -> Vec<&str>;
}

impl ContentTextSource for str {
    fn text_parts(&self) -> Vec<&str> {
        vec![self]
    }
}

impl ContentTextSource for String {
    fn text_parts(&self) -> Vec<&str> {
        vec![self.as_str()]
    }
}

impl ContentTextSource for &[AssistantBlock] {
    fn text_parts(&self) -> Vec<&str> {
        self.iter()
            .filter_map(|block| match block {
                AssistantBlock::Text(content) => Some(content.text.as_str()),
                _ => None,
            })
            .collect()
    }
}

impl ContentTextSource for &[ToolResultBlock] {
    fn text_parts(&self) -> Vec<&str> {
        self.iter()
            .filter_map(|block| match block {
                ToolResultBlock::Text(content) => Some(content.text.as_str()),
                ToolResultBlock::Image(_) => None,
            })
            .collect()
    }
}

impl ContentTextSource for &[UserBlock] {
    fn text_parts(&self) -> Vec<&str> {
        self.iter()
            .filter_map(|block| match block {
                UserBlock::Text(content) => Some(content.text.as_str()),
                UserBlock::Image(_) => None,
            })
            .collect()
    }
}

impl ContentTextSource for UserContent {
    fn text_parts(&self) -> Vec<&str> {
        match self {
            Self::Text(text) => vec![text.as_str()],
            Self::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    UserBlock::Text(content) => Some(content.text.as_str()),
                    UserBlock::Image(_) => None,
                })
                .collect(),
        }
    }
}

/// Extract and join the text of a message's content.
#[must_use]
pub fn content_text(content: &impl ContentTextSource) -> String {
    content_text_with(content, "\n")
}

/// Extract and join the text of a message's content with a custom separator.
#[must_use]
pub fn content_text_with(content: &impl ContentTextSource, separator: &str) -> String {
    content.text_parts().join(separator)
}

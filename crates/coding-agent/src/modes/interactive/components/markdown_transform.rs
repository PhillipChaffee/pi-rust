//! Markdown transformer chaining, upstream
//! `src/modes/interactive/components/markdown-transform.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! A transformer that throws is skipped and the chain continues from the
//! Markdown it received, so one broken extension cannot blank a message.

use std::sync::Arc;

use pi_tui::components::markdown::TransformFn;

use crate::extensions::types::{
    MarkdownMessageType, MarkdownTransformContext, MarkdownTransformer,
};

/// Build the width-aware transform a [`pi_tui::components::Markdown`] row
/// runs, upstream's `createMarkdownTransform`.
#[must_use]
pub fn create_markdown_transform(
    message_type: MarkdownMessageType,
    is_streaming: bool,
    transformers: Arc<Vec<MarkdownTransformer>>,
) -> TransformFn {
    Arc::new(move |markdown, available_width| {
        let context = MarkdownTransformContext {
            message_type,
            is_streaming,
            available_width,
        };
        let mut transformed_markdown = markdown.to_string();
        for transformer in transformers.iter() {
            if let Ok(transformed) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                transformer(&transformed_markdown, &context)
            })) {
                transformed_markdown = transformed;
            }
        }
        transformed_markdown
    })
}

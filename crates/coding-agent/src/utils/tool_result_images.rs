//! Tool-result image normalization, upstream's
//! `src/utils/tool-result-images.ts`.

use base64::Engine as _;
use pi_ai::types::{ImageContent, TextContent};

use super::image_process::{ProcessImageOptions, ProcessImageResult, process_image};

/// A tool-result content block, upstream's `ToolResultContent`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolResultContent {
    /// A text block, upstream's `TextContent`.
    Text(TextContent),
    /// An image block, upstream's `ImageContent`.
    Image(ImageContent),
}

/// The normalization options, upstream's `NormalizeToolResultImagesOptions`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NormalizeToolResultImagesOptions {
    /// Whether oversized images are resized to inline provider limits.
    /// Default: true.
    pub auto_resize_images: Option<bool>,
}

/// Normalize image blocks returned by tool results.
///
/// The `read` tool and `@file` CLI attachments run their images through
/// `processImage`, but tools that produce images themselves (extensions,
/// MCP bridges, screenshot tools) hand back arbitrary base64 payloads that
/// go straight into session history and every subsequent provider request.
/// Oversized images make the provider reject the whole conversation, not
/// just the offending turn, so normalize them once as they enter history.
///
/// Returns an equal array when nothing changed so callers can skip
/// rewriting the result.
pub async fn normalize_tool_result_images(
    content: Vec<ToolResultContent>,
    options: Option<&NormalizeToolResultImagesOptions>,
) -> Vec<ToolResultContent> {
    if !content
        .iter()
        .any(|block| matches!(block, ToolResultContent::Image(_)))
    {
        return content;
    }

    let auto_resize_images = options
        .and_then(|options| options.auto_resize_images)
        .unwrap_or(true);
    let mut normalized: Vec<ToolResultContent> = Vec::new();

    for block in content {
        let ToolResultContent::Image(image) = &block else {
            normalized.push(block);
            continue;
        };

        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&image.data)
            .unwrap_or_default();
        let processed = process_image(
            bytes,
            &image.mime_type,
            Some(&ProcessImageOptions {
                auto_resize_images: Some(auto_resize_images),
                resize_options: None,
            }),
        )
        .await;
        let ProcessImageResult::Ok {
            data,
            mime_type,
            hints,
        } = processed
        else {
            // Unlike `read`, keep the original block. The tool already
            // produced this image and the failure may just be an
            // unavailable image backend, so passing it through preserves
            // the behavior tools have today instead of silently deleting
            // their output.
            normalized.push(block);
            continue;
        };

        if data == image.data && mime_type == image.mime_type && hints.is_empty() {
            normalized.push(block);
            continue;
        }

        normalized.push(ToolResultContent::Image(ImageContent { data, mime_type }));
        if !hints.is_empty() {
            normalized.push(ToolResultContent::Text(TextContent {
                text: hints.join("\n"),
                text_signature: None,
            }));
        }
    }

    normalized
}

//! The read tool, ported from upstream `src/harness/tools/read.ts`.

use std::sync::Arc;

use pi_ai::types::{BoxedFuture, ImageContent, TextContent, Tool};
use serde_json::json;

use crate::harness::context::Context;
use crate::harness::types::{AgentHarnessTool, AgentHarnessToolExecuteFn};
use crate::harness::utils::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationOptions, format_size, truncate_head,
};
use crate::types::AgentToolError;
use crate::types::{AgentToolContent, AgentToolResult};

use super::image::{detect_supported_image_mime_type, encode_base64};
use super::path_utils::resolve_read_tool_path;
use super::tool_context::{ExecutionToolContext, execution_tool_context, io_error};

/// The read tool's input schema, upstream's `readSchema`.
fn read_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (relative or absolute)"
            },
            "offset": {
                "type": "number",
                "description": "Line number to start reading from (1-indexed)"
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of lines to read"
            }
        },
        "required": ["path"]
    })
}

/// The parsed read-tool input, upstream's `ReadToolInput`.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadToolInput {
    /// The file to read.
    pub path: String,
    /// The 1-indexed line to start from.
    pub offset: Option<f64>,
    /// The maximum number of lines to read.
    pub limit: Option<f64>,
}

/// The read tool's structured details, upstream's `ReadToolDetails`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReadToolDetails {
    /// The truncation metadata, when the read truncated.
    pub truncation: Option<crate::harness::utils::truncate::TruncationResult>,
}

/// The injected image conversion outcome, upstream's
/// `ReadImageProcessorResult`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadImageProcessorResult {
    /// The converted image and its hints, upstream's `ok: true` shape.
    Ok {
        /// The converted image's base64 data.
        data: String,
        /// The converted image's MIME type.
        mime_type: String,
        /// The conversion hints appended to the text block.
        hints: Vec<String>,
    },
    /// The failure message rendered as text, upstream's `ok: false`.
    Failed {
        /// The failure message.
        message: String,
    },
}

/// The injected image conversion/resizing seam, upstream's
/// `ReadImageProcessor`.
pub type ReadImageProcessor = Arc<
    dyn for<'a> Fn(
            &'a [u8],
            &'a str,
            &'a ReadImageProcessorOptions,
            &'a Context,
        ) -> BoxedFuture<'a, Result<ReadImageProcessorResult, AgentToolError>>
        + Send
        + Sync,
>;

/// The image processor's options, upstream's
/// `{ autoResizeImages: boolean }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadImageProcessorOptions {
    /// Whether the processor should resize images.
    pub auto_resize_images: bool,
}

/// The read tool's options, upstream's `ReadToolOptions`.
#[derive(Clone, Default)]
pub struct ReadToolOptions {
    /// Whether an injected image processor should resize images. Defaults
    /// to `true`.
    pub auto_resize_images: Option<bool>,
    /// The optional image conversion/resizing implementation.
    pub image_processor: Option<ReadImageProcessor>,
}

impl std::fmt::Debug for ReadToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadToolOptions")
            .field("auto_resize_images", &self.auto_resize_images)
            .field("image_processor", &self.image_processor.is_some())
            .finish()
    }
}

/// Builds the read tool, upstream's `createReadTool`.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "the executor mirrors upstream's single createReadTool execute body: the image ladder and the offset/limit slicing read the same decoded state"
)]
pub fn create_read_tool<C: ExecutionToolContext>(
    options: Option<ReadToolOptions>,
) -> AgentHarnessTool {
    let options = options.unwrap_or_default();
    let execute: Arc<AgentHarnessToolExecuteFn> = Arc::new(
        move |_tool_call_id: &str,
              args: &serde_json::Value,
              _on_update,
              tool_context,
              _invocation,
              context: &Context| {
            let options = options.clone();
            Box::pin(async move {
                let input: ReadToolInput = parse_input(args)?;
                let context_holder = execution_tool_context::<C>(&tool_context)?;
                let env = context_holder.env();
                let absolute_path =
                    resolve_read_tool_path(env.as_ref(), &input.path, context).await?;
                let bytes = env
                    .read_binary_file(&absolute_path, context)
                    .await
                    .map_err(Box::<crate::harness::types::FileError>::from)?;
                if let Some(mime_type) = detect_supported_image_mime_type(&bytes) {
                    return image_result(&options, &bytes, mime_type, context).await;
                }

                let text_content = String::from_utf8_lossy(&bytes).into_owned();
                let all_lines: Vec<&str> = text_content.split('\n').collect();
                let total_file_lines = all_lines.len();
                let start_line = input
                    .offset
                    .map_or(0_usize, |offset| clamp_line_index(offset - 1.0));
                let start_line_display = start_line + 1;
                if start_line >= all_lines.len() {
                    return Err(io_error(format!(
                        "Offset {} is beyond end of file ({} lines total)",
                        format_number(input.offset.unwrap_or_default()),
                        all_lines.len()
                    )));
                }

                let (selected_content, user_limited_lines) = input.limit.map_or_else(
                    || (all_lines[start_line..].join("\n"), None),
                    |limit| {
                        let end_line = (start_line + clamp_line_index(limit)).min(all_lines.len());
                        (
                            all_lines[start_line..end_line].join("\n"),
                            Some(end_line - start_line),
                        )
                    },
                );

                let truncation = truncate_head(&selected_content, TruncationOptions::default());
                let output_text: String;
                let mut details = None;
                if truncation.metadata.first_line_exceeds_limit {
                    let first_line_size =
                        format_size(u64::try_from(all_lines[start_line].len()).unwrap_or(u64::MAX));
                    output_text = format!(
                        "[Line {} is {first_line_size}, exceeds {} limit. Use bash: sed -n '{}p' {} | head -c {}]",
                        start_line_display,
                        format_size(DEFAULT_MAX_BYTES),
                        start_line_display,
                        input.path,
                        DEFAULT_MAX_BYTES
                    );
                    details = Some(ReadToolDetails {
                        truncation: Some(truncation),
                    });
                } else if truncation.metadata.truncated {
                    let end_line_display = start_line_display
                        + usize::try_from(truncation.metadata.output_lines).unwrap_or(usize::MAX)
                        - 1;
                    let next_offset = end_line_display + 1;
                    output_text = if truncation.metadata.truncated_by
                        == Some(crate::harness::types::TruncatedBy::Lines)
                    {
                        format!(
                            "{}\n\n[Showing lines {}-{} of {}. Use offset={} to continue.]",
                            truncation.content,
                            start_line_display,
                            end_line_display,
                            total_file_lines,
                            next_offset
                        )
                    } else {
                        format!(
                            "{}\n\n[Showing lines {}-{} of {} ({} limit). Use offset={} to continue.]",
                            truncation.content,
                            start_line_display,
                            end_line_display,
                            total_file_lines,
                            format_size(DEFAULT_MAX_BYTES),
                            next_offset
                        )
                    };
                    details = Some(ReadToolDetails {
                        truncation: Some(truncation),
                    });
                } else if let Some(user_limited_lines) = user_limited_lines
                    && start_line + user_limited_lines < all_lines.len()
                {
                    let remaining = all_lines.len() - (start_line + user_limited_lines);
                    let next_offset = start_line + user_limited_lines + 1;
                    output_text = format!(
                        "{}\n\n[{} more lines in file. Use offset={} to continue.]",
                        truncation.content, remaining, next_offset
                    );
                } else {
                    output_text = truncation.content;
                }

                Ok(AgentToolResult {
                    content: vec![AgentToolContent::Text(TextContent {
                        text: output_text,
                        text_signature: None,
                    })],
                    details: serialize_details(details.as_ref()),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        },
    );
    AgentHarnessTool {
        tool: Tool {
            name: "read".to_owned(),
            description: format!(
                "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
                DEFAULT_MAX_BYTES / 1024
            ),
            parameters: read_schema(),
            constrained_sampling: None,
        },
        label: "read".to_owned(),
        prepare_arguments: None,
        execute,
        replay: None,
        execution_mode: None,
    }
}

/// The image branch, upstream's detection/processor/bmp/base64 ladder.
async fn image_result(
    options: &ReadToolOptions,
    bytes: &[u8],
    mime_type: &str,
    context: &Context,
) -> Result<AgentToolResult, AgentToolError> {
    if let Some(processor) = options.image_processor.as_ref() {
        let processor_options = ReadImageProcessorOptions {
            auto_resize_images: options.auto_resize_images.unwrap_or(true),
        };
        let processed = processor(bytes, mime_type, &processor_options, context).await?;
        match processed {
            ReadImageProcessorResult::Failed { message } => {
                return Ok(AgentToolResult {
                    content: vec![AgentToolContent::Text(TextContent {
                        text: format!("Read image file [{mime_type}]\n{message}"),
                        text_signature: None,
                    })],
                    details: serde_json::Value::Null,
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                });
            }
            ReadImageProcessorResult::Ok {
                data,
                mime_type: converted_mime_type,
                hints,
            } => {
                let hints = if hints.is_empty() {
                    String::new()
                } else {
                    format!("\n{}", hints.join("\n"))
                };
                return Ok(AgentToolResult {
                    content: vec![
                        AgentToolContent::Text(TextContent {
                            text: format!("Read image file [{converted_mime_type}]{hints}"),
                            text_signature: None,
                        }),
                        AgentToolContent::Image(ImageContent {
                            data,
                            mime_type: converted_mime_type,
                        }),
                    ],
                    details: serde_json::Value::Null,
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                });
            }
        }
    }
    if mime_type == "image/bmp" {
        return Ok(AgentToolResult {
            content: vec![AgentToolContent::Text(TextContent {
                text: "Read image file [image/bmp]\n[Image omitted: configure an imageProcessor to convert BMP images.]".to_owned(),
                text_signature: None,
            })],
            details: serde_json::Value::Null,
            usage: None,
            added_tool_names: None,
            terminate: None,
        });
    }
    Ok(AgentToolResult {
        content: vec![
            AgentToolContent::Text(TextContent {
                text: format!("Read image file [{mime_type}]"),
                text_signature: None,
            }),
            AgentToolContent::Image(ImageContent {
                data: encode_base64(bytes),
                mime_type: mime_type.to_owned(),
            }),
        ],
        details: serde_json::Value::Null,
        usage: None,
        added_tool_names: None,
        terminate: None,
    })
}

/// The `Math.max(0, offset - 1)` line index, restated over the JS slice's
/// toward-zero truncation for non-integer arguments.
const fn clamp_line_index(value: f64) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a fractional line index is meaningless; the JS slice truncates toward zero the same way"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "the value is clamped non-negative first"
    )]
    let clamped = value.max(0.0) as usize;
    clamped
}

/// The JS `String(number)` shape for the offsets the suite exercises:
/// integers print bare.
fn format_number(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "integral f64 values print bare like JS String(number)"
        )]
        let integer = value as i64;
        format!("{integer}")
    } else {
        format!("{value}")
    }
}

fn parse_input(args: &serde_json::Value) -> Result<ReadToolInput, AgentToolError> {
    #[derive(serde::Deserialize)]
    struct RawInput {
        path: String,
        offset: Option<f64>,
        limit: Option<f64>,
    }
    serde_json::from_value::<RawInput>(args.clone())
        .map(|raw| ReadToolInput {
            path: raw.path,
            offset: raw.offset,
            limit: raw.limit,
        })
        .map_err(|error| io_error(error.to_string()))
}

/// Serializes the read details, upstream's `ReadToolDetails` (the full
/// truncation result, flat).
fn serialize_details(details: Option<&ReadToolDetails>) -> serde_json::Value {
    details
        .and_then(|details| details.truncation.as_ref())
        .map_or_else(
            || serde_json::Value::Null,
            |truncation| {
                let metadata = &truncation.metadata;
                json!({
                    "truncation": {
                        "content": truncation.content,
                        "truncated": metadata.truncated,
                        "truncatedBy": metadata.truncated_by,
                        "totalLines": metadata.total_lines,
                        "totalBytes": metadata.total_bytes,
                        "outputLines": metadata.output_lines,
                        "outputBytes": metadata.output_bytes,
                        "lastLinePartial": metadata.last_line_partial,
                        "firstLineExceedsLimit": metadata.first_line_exceeds_limit,
                        "maxLines": metadata.max_lines,
                        "maxBytes": metadata.max_bytes,
                    }
                })
            },
        )
}

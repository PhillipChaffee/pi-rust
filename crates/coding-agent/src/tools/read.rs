//! The read tool, ported from upstream `src/core/tools/read.ts`.
//!
//! Text reads truncate through the shared [`super::truncate`] belt with the
//! actionable continuation notices; image reads detect the MIME type from
//! the file magic, convert and resize through the image belt, and attach
//! the image block with the non-vision note when the current model cannot
//! take images.

use std::fmt::Write as _;
use std::sync::Arc;

use pi_agent_core::harness::context::AbortSignal;
use pi_agent_core::types::{AgentToolContent, AgentToolError, AgentToolResult};
use pi_ai::types::{BoxedFuture, ImageContent, TextContent};
use serde_json::{Value, json};

use crate::extensions::types::{ExtensionContext, ToolDefinition};
use crate::utils::image_process::{ProcessImageOptions, ProcessImageResult, process_image};

use super::path_utils::resolve_read_path_async;
use super::tool_definition_wrapper::wrap_cwd_tool;
use super::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncatedBy, TruncationOptions, TruncationResult,
    format_size, truncate_head,
};
use super::{io_error, strict_sampling, tool_schema};

/// The read tool's input, upstream's `ReadToolInput`.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadToolInput {
    /// The path to read.
    pub path: String,
    /// The 1-indexed line to start from.
    pub offset: Option<f64>,
    /// The maximum number of lines to read.
    pub limit: Option<f64>,
}

/// The read tool's details, upstream's `ReadToolDetails`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReadToolDetails {
    /// The truncation metadata, when the output truncated.
    pub truncation: Option<TruncationResult>,
}

/// The pluggable operations for the read tool, upstream's
/// `ReadOperations`. Override to delegate file reading to remote systems.
#[derive(Clone)]
pub struct ReadOperations {
    /// Read file contents, upstream's `readFile` returning a Buffer.
    pub read_file: ReadReadFileFn,
    /// Check the file is readable (throw if not), upstream's `access`.
    pub access: ReadAccessFn,
    /// Detect the image MIME type, `None` for non-images, upstream's
    /// `detectImageMimeType`.
    pub detect_image_mime_type: Option<ReadDetectImageMimeFn>,
}

/// The erased `readFile` operation.
pub type ReadReadFileFn =
    Arc<dyn Fn(String) -> BoxedFuture<'static, Result<Vec<u8>, AgentToolError>> + Send + Sync>;
/// The erased `access` operation.
pub type ReadAccessFn =
    Arc<dyn Fn(String) -> BoxedFuture<'static, Result<(), AgentToolError>> + Send + Sync>;
/// The erased `detectImageMimeType` operation.
pub type ReadDetectImageMimeFn = Arc<
    dyn Fn(String) -> BoxedFuture<'static, Result<Option<&'static str>, AgentToolError>>
        + Send
        + Sync,
>;

impl std::fmt::Debug for ReadOperations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadOperations")
            .field(
                "detect_image_mime_type",
                &self.detect_image_mime_type.is_some(),
            )
            .finish_non_exhaustive()
    }
}

fn default_read_operations() -> ReadOperations {
    ReadOperations {
        read_file: Arc::new(|path| {
            Box::pin(async move { tokio::fs::read(&path).await.map_err(AgentToolError::from) })
        }),
        access: Arc::new(|path| {
            Box::pin(async move {
                tokio::fs::File::open(&path)
                    .await
                    .map(|_| ())
                    .map_err(AgentToolError::from)
            })
        }),
        detect_image_mime_type: Some(Arc::new(|path| {
            Box::pin(async move {
                crate::utils::mime::detect_supported_image_mime_type_from_file(&path)
                    .map_err(AgentToolError::from)
            })
        })),
    }
}

/// The read tool's options, upstream's `ReadToolOptions`.
#[derive(Clone, Default)]
pub struct ReadToolOptions {
    /// Whether to auto-resize images to the inline limits; default true.
    pub auto_resize_images: Option<bool>,
    /// Custom operations for file reading; default, local filesystem.
    pub operations: Option<ReadOperations>,
}

impl std::fmt::Debug for ReadToolOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadToolOptions")
            .field("auto_resize_images", &self.auto_resize_images)
            .field("operations", &self.operations.is_some())
            .finish()
    }
}

/// The read tool's system-prompt contribution, upstream's
/// `readToolSystemPromptContribution`.
pub const READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION: super::bash::SystemPromptContribution =
    super::bash::SystemPromptContribution {
        snippet: "Read file contents",
        guidelines: &["Use read to examine files instead of cat or sed."],
    };

/// The read tool's schema, upstream's `readSchema`.
fn read_schema() -> Value {
    tool_schema(
        &json!({
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
        }),
        &["path"],
    )
}

fn parse_input(params: &Value) -> Result<ReadToolInput, AgentToolError> {
    #[derive(serde::Deserialize)]
    struct RawInput {
        path: String,
        offset: Option<f64>,
        limit: Option<f64>,
    }
    serde_json::from_value::<RawInput>(params.clone())
        .map(|raw| ReadToolInput {
            path: raw.path,
            offset: raw.offset,
            limit: raw.limit,
        })
        .map_err(|error| io_error(error.to_string()))
}

/// The non-vision note, upstream's `getNonVisionImageNote`.
fn non_vision_image_note(model: Option<&pi_ai::types::Model>) -> Option<String> {
    let model = model?;
    if model.input.contains(&pi_ai::types::Modality::Image) {
        return None;
    }
    Some(
        "[Current model does not support images. The image will be omitted from this request.]"
            .to_owned(),
    )
}

/// The read tool's details wire shape with the absent member dropped.
fn details_wire(details: &ReadToolDetails) -> Value {
    details.truncation.as_ref().map_or_else(
        || Value::Null,
        |truncation| json!({ "truncation": serde_json::to_value(truncation).unwrap_or_default() }),
    )
}

/// The read tool's execution body, upstream's `createReadToolDefinition`
/// execute.
#[expect(
    clippy::too_many_lines,
    reason = "the body mirrors upstream's single execute: the image branch, the text slice, and the three truncation notices"
)]
async fn execute_read_tool(
    params: &Value,
    signal: Option<&AbortSignal>,
    ctx: Option<&dyn ExtensionContext>,
    ops: &ReadOperations,
    auto_resize_images: bool,
    cwd: &str,
) -> Result<AgentToolResult, AgentToolError> {
    let input = parse_input(params)?;
    if signal.is_some_and(AbortSignal::aborted) {
        return Err(io_error("Operation aborted"));
    }
    let effective_cwd = ctx
        .map(ExtensionContext::cwd)
        .filter(|ctx_cwd| !ctx_cwd.is_empty())
        .unwrap_or(cwd);
    let absolute_path = resolve_read_path_async(&input.path, effective_cwd)
        .await
        .map_err(|error| io_error(error.to_string()))?;
    if signal.is_some_and(AbortSignal::aborted) {
        return Err(io_error("Operation aborted"));
    }
    // Check if file exists and is readable.
    (ops.access)(absolute_path.clone()).await?;
    if signal.is_some_and(AbortSignal::aborted) {
        return Err(io_error("Operation aborted"));
    }
    let mime_type = match ops.detect_image_mime_type.as_ref() {
        Some(detect) => (detect)(absolute_path.clone()).await?,
        None => None,
    };
    let non_vision_image_note = non_vision_image_note(ctx.and_then(ExtensionContext::model));

    if let Some(mime_type) = mime_type {
        // Read image as binary.
        let buffer = (ops.read_file)(absolute_path.clone()).await?;
        let processed = process_image(
            buffer,
            mime_type,
            Some(&ProcessImageOptions {
                auto_resize_images: Some(auto_resize_images),
                ..ProcessImageOptions::default()
            }),
        )
        .await;
        let (content, details) = match processed {
            ProcessImageResult::Omitted { message } => {
                let mut note = format!("Read image file [{mime_type}]\n{message}");
                if let Some(extra) = non_vision_image_note.as_ref() {
                    note.push('\n');
                    note.push_str(extra);
                }
                (
                    vec![AgentToolContent::Text(TextContent {
                        text: note,
                        text_signature: None,
                    })],
                    ReadToolDetails::default(),
                )
            }
            ProcessImageResult::Ok {
                data,
                mime_type: processed_mime,
                hints,
            } => {
                let mut note = format!("Read image file [{processed_mime}]");
                if !hints.is_empty() {
                    note.push('\n');
                    note.push_str(&hints.join("\n"));
                }
                if let Some(extra) = non_vision_image_note.as_ref() {
                    note.push('\n');
                    note.push_str(extra);
                }
                (
                    vec![
                        AgentToolContent::Text(TextContent {
                            text: note,
                            text_signature: None,
                        }),
                        AgentToolContent::Image(ImageContent {
                            data,
                            mime_type: processed_mime,
                        }),
                    ],
                    ReadToolDetails::default(),
                )
            }
        };
        return finish(signal, content, &details);
    }

    // Read text content.
    let buffer = (ops.read_file)(absolute_path.clone()).await?;
    let text_content = String::from_utf8_lossy(&buffer).into_owned();
    let all_lines: Vec<&str> = text_content.split('\n').collect();
    let total_file_lines = all_lines.len();
    // Apply offset if specified. Convert from 1-indexed input to 0-indexed
    // array access.
    let start_line = input.offset.map_or(0, |offset| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a fractional offset floors to a whole line number, upstream's Math.max(0, offset)"
        )]
        let one_based = offset.max(0.0) as usize;
        one_based.saturating_sub(1)
    });
    let start_line_display = start_line + 1;
    // Check if offset is out of bounds.
    if start_line >= all_lines.len() {
        return Err(io_error(format!(
            "Offset {} is beyond end of file ({} lines total)",
            render_js_number(input.offset.unwrap_or_default()),
            total_file_lines
        )));
    }
    // If limit is specified by the user, honor it first. Otherwise
    // truncateHead decides.
    let (selected_content, user_limited_lines) = input.limit.map_or_else(
        || (all_lines[start_line..].join("\n"), None),
        |limit| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a fractional limit floors to a whole line count, upstream's Math.max(0, limit)"
            )]
            let limit = limit.max(0.0) as usize;
            let end_line = std::cmp::min(start_line.saturating_add(limit), all_lines.len());
            (
                all_lines[start_line..end_line].join("\n"),
                Some(end_line.saturating_sub(start_line)),
            )
        },
    );
    // Apply truncation, respecting both line and byte limits.
    let truncation = truncate_head(&selected_content, TruncationOptions::default());
    let output_text: String;
    let details: ReadToolDetails;
    if truncation.first_line_exceeds_limit {
        // First line alone exceeds the byte limit. Point the model at a
        // bash fallback.
        let first_line_size = format_size(all_lines[start_line].len());
        output_text = format!(
            "[Line {start_line_display} is {first_line_size}, exceeds {} limit. Use bash: sed -n '{start_line_display}p' {} | head -c {DEFAULT_MAX_BYTES}]",
            format_size(DEFAULT_MAX_BYTES),
            input.path
        );
        details = ReadToolDetails {
            truncation: Some(truncation),
        };
    } else if truncation.truncated {
        // Truncation occurred. Build an actionable continuation notice.
        let end_line_display = start_line_display + truncation.output_lines.saturating_sub(1);
        let next_offset = end_line_display + 1;
        let mut text = truncation.content.clone();
        if truncation.truncated_by == Some(TruncatedBy::Lines) {
            let _ = write!(
                text,
                "\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. Use offset={next_offset} to continue.]"
            );
        } else {
            let _ = write!(
                text,
                "\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} ({} limit). Use offset={next_offset} to continue.]",
                format_size(DEFAULT_MAX_BYTES)
            );
        }
        output_text = text;
        details = ReadToolDetails {
            truncation: Some(truncation),
        };
    } else if let Some(user_limited_lines) = user_limited_lines
        && start_line + user_limited_lines < all_lines.len()
    {
        // User-specified limit stopped early, but the file still has more
        // content.
        let remaining = all_lines.len() - (start_line + user_limited_lines);
        let next_offset = start_line + user_limited_lines + 1;
        output_text = format!(
            "{}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]",
            truncation.content
        );
        details = ReadToolDetails::default();
    } else {
        // No truncation and no remaining user-limited content.
        output_text = truncation.content;
        details = ReadToolDetails::default();
    }
    let content = vec![AgentToolContent::Text(TextContent {
        text: output_text,
        text_signature: None,
    })];
    finish(signal, content, &details)
}

/// The abort-checked settle both branches share.
fn finish(
    signal: Option<&AbortSignal>,
    content: Vec<AgentToolContent>,
    details: &ReadToolDetails,
) -> Result<AgentToolResult, AgentToolError> {
    if signal.is_some_and(AbortSignal::aborted) {
        return Err(io_error("Operation aborted"));
    }
    Ok(AgentToolResult {
        content,
        details: details_wire(details),
        usage: None,
        added_tool_names: None,
        terminate: None,
    })
}

/// The JS `String(number)` rendering for the offset error message.
fn render_js_number(value: f64) -> String {
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

/// Build the read tool definition, upstream's `createReadToolDefinition`.
#[must_use]
pub fn create_read_tool_definition(cwd: &str, options: Option<ReadToolOptions>) -> ToolDefinition {
    let options = options.unwrap_or_default();
    let auto_resize_images = options.auto_resize_images.unwrap_or(true);
    let ops = Arc::new(options.operations.unwrap_or_else(default_read_operations));
    let cwd = Arc::new(cwd.to_owned());

    ToolDefinition {
        name: "read".to_owned(),
        label: "read".to_owned(),
        description: format!(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
            DEFAULT_MAX_BYTES / 1024
        ),
        prompt_snippet: Some(READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION.snippet.to_owned()),
        prompt_guidelines: Some(
            READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION
                .guidelines
                .iter()
                .map(|guideline| (*guideline).to_owned())
                .collect(),
        ),
        parameters: read_schema(),
        constrained_sampling: Some(strict_sampling()),
        render_shell: None,
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(
            move |_tool_call_id: &str, params: &Value, signal, _on_update, ctx| {
                let ops = Arc::clone(&ops);
                let cwd = Arc::clone(&cwd);
                Box::pin(async move {
                    execute_read_tool(params, signal, ctx, &ops, auto_resize_images, &cwd).await
                })
            },
        ),
    }
}

/// Build the read tool, upstream's `createReadTool`.
#[must_use]
pub fn create_read_tool(
    cwd: &str,
    options: Option<ReadToolOptions>,
) -> pi_agent_core::harness::types::AgentHarnessTool {
    let definition = create_read_tool_definition(cwd, options);
    wrap_cwd_tool(definition)
}

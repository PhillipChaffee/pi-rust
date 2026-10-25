//! `@file` argument processing, upstream's `src/cli/file-processor.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream exits the process on a missing or unreadable file; the port
//! returns [`ProcessFileError`] and lets the binary entry decide, so the
//! library keeps no exit path.

use std::fmt;
use std::path::PathBuf;

use pi_ai::types::ImageContent;

use crate::tools::path_utils::resolve_read_path;
use crate::utils::image_process::{ProcessImageOptions, ProcessImageResult, process_image};
use crate::utils::mime::detect_supported_image_mime_type_from_file;
use crate::utils::text::strip_bom;

/// The pipeline options, upstream's `ProcessFileOptions`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcessFileOptions {
    /// Whether to auto-resize images to the inline limit. Default: true.
    pub auto_resize_images: Option<bool>,
}

/// The rendered files, upstream's `ProcessedFiles`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcessedFiles {
    /// The `<file>` text blocks, in argument order.
    pub text: String,
    /// The image attachments, in argument order.
    pub images: Vec<ImageContent>,
}

/// Why a file argument could not be processed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessFileError {
    /// The resolved path does not exist, upstream's `access` failure.
    NotFound(PathBuf),
    /// The file could not be read or its image type sniffed.
    Read {
        /// The path being processed.
        path: PathBuf,
        /// The underlying failure's message.
        message: String,
    },
}

impl fmt::Display for ProcessFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(path) => write!(formatter, "File not found: {}", path.display()),
            Self::Read { path, message } => {
                write!(
                    formatter,
                    "Could not read file {}: {message}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for ProcessFileError {}

fn read_error(path: &std::path::Path, error: impl fmt::Display) -> ProcessFileError {
    ProcessFileError::Read {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

/// Process `@file` arguments into text content and image attachments,
/// upstream's `processFileArguments`.
///
/// Empty files are skipped. A text file contributes a `<file>` block with its
/// BOM-stripped content; an image contributes an attachment plus either its
/// processing hints or an empty `<file>` reference in the text.
///
/// # Errors
/// [`ProcessFileError::NotFound`] when a path does not exist, and
/// [`ProcessFileError::Read`] when a file cannot be read, sniffed, or
/// decoded.
pub async fn process_file_arguments(
    file_args: &[String],
    cwd: &str,
    options: Option<&ProcessFileOptions>,
) -> Result<ProcessedFiles, ProcessFileError> {
    use std::fmt::Write as _;

    let mut text = String::new();
    let mut images: Vec<ImageContent> = Vec::new();

    for file_arg in file_args {
        let absolute_path = resolve_read_path(file_arg, cwd)
            .map_err(|error| read_error(std::path::Path::new(file_arg), error))?;
        let path = std::path::Path::new(&absolute_path);

        let stats = tokio::fs::metadata(path)
            .await
            .map_err(|_| ProcessFileError::NotFound(path.to_path_buf()))?;
        if stats.len() == 0 {
            continue;
        }

        let mime_type = detect_supported_image_mime_type_from_file(&absolute_path)
            .map_err(|error| read_error(path, error))?;

        if let Some(mime_type) = mime_type {
            let content = tokio::fs::read(path)
                .await
                .map_err(|error| read_error(path, error))?;
            let image_options = options.map_or_else(ProcessImageOptions::default, |options| {
                ProcessImageOptions {
                    auto_resize_images: options.auto_resize_images,
                    resize_options: None,
                }
            });
            match process_image(content, mime_type, Some(&image_options)).await {
                ProcessImageResult::Omitted { message } => {
                    let _ = writeln!(text, "<file name=\"{absolute_path}\">{message}</file>");
                }
                ProcessImageResult::Ok {
                    data,
                    mime_type,
                    hints,
                } => {
                    images.push(ImageContent { data, mime_type });
                    if hints.is_empty() {
                        let _ = writeln!(text, "<file name=\"{absolute_path}\"></file>");
                    } else {
                        let _ = writeln!(
                            text,
                            "<file name=\"{absolute_path}\">{}</file>",
                            hints.join("\n")
                        );
                    }
                }
            }
        } else {
            let content = tokio::fs::read(path)
                .await
                .map_err(|error| read_error(path, error))?;
            let content = String::from_utf8_lossy(&content);
            let _ = writeln!(
                text,
                "<file name=\"{absolute_path}\">\n{}\n</file>",
                strip_bom(&content)
            );
        }
    }

    Ok(ProcessedFiles { text, images })
}

#[cfg(test)]
mod tests;

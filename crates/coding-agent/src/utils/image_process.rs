//! The inline-image processing pipeline, upstream's
//! `src/utils/image-process.ts`.

use base64::Engine as _;

use super::image_convert::convert_image_bytes_to_png;
use super::image_resize::{ImageResizeOptions, format_dimension_note, resize_image};

/// The pipeline options, upstream's `ProcessImageOptions`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcessImageOptions {
    /// Whether to resize images to inline provider limits. Default: true.
    pub auto_resize_images: Option<bool>,
    /// Optional resize overrides. Uses resize defaults when omitted.
    pub resize_options: Option<ImageResizeOptions>,
}

/// The pipeline outcome, upstream's `ProcessImageResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessImageResult {
    /// The image survived normalization; hints ride alongside.
    Ok {
        /// The base64 payload.
        data: String,
        /// The payload's MIME type.
        mime_type: String,
        /// The conversion and dimension notes.
        hints: Vec<String>,
    },
    /// The image was dropped, with the caller-facing reason.
    Omitted {
        /// The message the caller reports.
        message: String,
    },
}

struct NormalizedImage {
    bytes: Vec<u8>,
    mime_type: String,
    converted_from: Option<String>,
}

fn base_mime_type(mime_type: &str) -> String {
    mime_type
        .split(';')
        .next()
        .unwrap_or(mime_type)
        .trim()
        .to_lowercase()
}

fn normalize_supported_image_mime_type(mime_type: &str) -> Option<&'static str> {
    match base_mime_type(mime_type).as_str() {
        "image/png" => Some("image/png"),
        "image/jpeg" | "image/jpg" => Some("image/jpeg"),
        "image/gif" => Some("image/gif"),
        "image/webp" => Some("image/webp"),
        _ => None,
    }
}

fn normalize_image(bytes: Vec<u8>, mime_type: &str) -> Option<NormalizedImage> {
    if let Some(normalized_mime_type) = normalize_supported_image_mime_type(mime_type) {
        return Some(NormalizedImage {
            bytes,
            mime_type: normalized_mime_type.to_string(),
            converted_from: None,
        });
    }

    let png_bytes = convert_image_bytes_to_png(&bytes)?;
    Some(NormalizedImage {
        bytes: png_bytes,
        mime_type: "image/png".to_string(),
        converted_from: Some(base_mime_type(mime_type)),
    })
}

fn conversion_hint(from: Option<&str>, to: &str) -> Option<String> {
    let from = from?;
    if from == to {
        return None;
    }
    Some(format!("[Image converted from {from} to {to}.]"))
}

/// Run an image through the inline pipeline, upstream's `processImage`:
/// convert anything the providers cannot inline, then (by default) resize
/// under the inline limits.
///
/// Returns [`ProcessImageResult::Omitted`] with the message the callers
/// present when the image cannot be converted or resized.
pub async fn process_image(
    bytes: Vec<u8>,
    mime_type: &str,
    options: Option<&ProcessImageOptions>,
) -> ProcessImageResult {
    let auto_resize_images = options
        .and_then(|options| options.auto_resize_images)
        .unwrap_or(true);
    let Some(normalized) = normalize_image(bytes, mime_type) else {
        return ProcessImageResult::Omitted {
            message: "[Image omitted: could not be converted to a supported inline image format.]"
                .to_string(),
        };
    };

    if auto_resize_images {
        let resized = resize_image(
            normalized.bytes,
            normalized.mime_type.clone(),
            options.and_then(|options| options.resize_options.clone()),
        )
        .await;
        let Some(resized) = resized else {
            return ProcessImageResult::Omitted {
                message: "[Image omitted: could not be resized below the inline image size limit.]"
                    .to_string(),
            };
        };

        let mut hints = Vec::new();
        if let Some(converted_hint) =
            conversion_hint(normalized.converted_from.as_deref(), &resized.mime_type)
        {
            hints.push(converted_hint);
        }
        if let Some(dimension_note) = format_dimension_note(&resized) {
            hints.push(dimension_note);
        }

        return ProcessImageResult::Ok {
            data: resized.data,
            mime_type: resized.mime_type,
            hints,
        };
    }

    let mut hints = Vec::new();
    if let Some(converted_hint) =
        conversion_hint(normalized.converted_from.as_deref(), &normalized.mime_type)
    {
        hints.push(converted_hint);
    }

    ProcessImageResult::Ok {
        data: base64::engine::general_purpose::STANDARD.encode(normalized.bytes),
        mime_type: normalized.mime_type,
        hints,
    }
}

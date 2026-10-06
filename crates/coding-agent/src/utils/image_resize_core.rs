//! The resize pipeline, upstream's `src/utils/image-resize-core.ts`.
//!
//! The WASM photon decode/resize/encode restate onto the `image` crate —
//! Lanczos3 sampling, PNG and quality-parameterized JPEG encodes — and the
//! candidate ladder ports 1:1: dimension clamp first, then the cheapest
//! encoding under the byte cap, then shrinking by a quarter until 1x1.

use std::io::Cursor;

use super::exif_orientation::apply_exif_orientation;
use base64::Engine as _;
use image::imageops::FilterType;
use image::{DynamicImage, ImageFormat};

/// The resize limits, upstream's `ImageResizeOptions` with its defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageResizeOptions {
    /// The widest allowed image; default 2000.
    pub max_width: Option<u32>,
    /// The tallest allowed image; default 2000.
    pub max_height: Option<u32>,
    /// The largest allowed base64 payload; default 4.5MB of payload, below
    /// Anthropic's 5MB limit.
    pub max_bytes: Option<usize>,
    /// The JPEG quality the ladder starts at; default 80.
    pub jpeg_quality: Option<u8>,
}

impl Default for ImageResizeOptions {
    fn default() -> Self {
        Self {
            max_width: Some(2000),
            max_height: Some(2000),
            // 4.5MB of base64 payload. Provides headroom below Anthropic's 5MB limit.
            max_bytes: Some(4_718_592),
            jpeg_quality: Some(80),
        }
    }
}

impl ImageResizeOptions {
    fn max_width(&self) -> u32 {
        self.max_width.unwrap_or(2000)
    }

    fn max_height(&self) -> u32 {
        self.max_height.unwrap_or(2000)
    }

    fn max_bytes(&self) -> usize {
        self.max_bytes.unwrap_or(4_718_592)
    }

    fn jpeg_quality(&self) -> u8 {
        self.jpeg_quality.unwrap_or(80)
    }
}

/// A resized image and the bookkeeping the dimension note needs, upstream's
/// `ResizedImage`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResizedImage {
    /// The base64 payload.
    pub data: String,
    /// The payload's MIME type.
    pub mime_type: String,
    /// The decoded image's width before any work.
    pub original_width: u32,
    /// The decoded image's height before any work.
    pub original_height: u32,
    /// The delivered width.
    pub width: u32,
    /// The delivered height.
    pub height: u32,
    /// Whether any resize happened.
    pub was_resized: bool,
}

/// One encoded candidate, upstream's `EncodedCandidate`.
struct EncodedCandidate {
    data: String,
    encoded_size: usize,
    mime_type: &'static str,
}

fn encode_candidate(buffer: &[u8], mime_type: &'static str) -> EncodedCandidate {
    let data = base64::engine::general_purpose::STANDARD.encode(buffer);
    EncodedCandidate {
        encoded_size: data.len(),
        data,
        mime_type,
    }
}

fn encode_png(buffer: &DynamicImage) -> Option<Vec<u8>> {
    let mut encoded = Cursor::new(Vec::new());
    buffer.write_to(&mut encoded, ImageFormat::Png).ok()?;
    Some(encoded.into_inner())
}

fn encode_jpeg(buffer: &DynamicImage, quality: u8) -> Option<Vec<u8>> {
    let mut sink = Cursor::new(Vec::new());
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut sink, quality);
    // JPEG carries no alpha; the encoder's own conversion drops it the way
    // photon's get_bytes_jpeg does.
    encoder
        .encode_image(&DynamicImage::ImageRgb8(buffer.to_rgb8()))
        .ok()?;
    Some(sink.into_inner())
}

/// Resize an image to fit within the specified max dimensions and encoded
/// file size, upstream's `resizeImageInProcess`.
///
/// Returns `None` when the bytes do not decode or no candidate gets under
/// `max_bytes`, photon's null both ways.
///
/// Strategy for staying under `max_bytes`:
/// 1. First resize to maxWidth/maxHeight
/// 2. Try both PNG and JPEG formats, pick the smaller one
/// 3. If still too large, try JPEG with decreasing quality
/// 4. If still too large, progressively reduce dimensions until 1x1
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "the candidate ladder is one loop upstream expresses as one function; splitting the shrink walk would scatter the byte-cap contract"
)]
pub fn resize_image_in_process(
    input_bytes: &[u8],
    mime_type: &str,
    options: Option<&ImageResizeOptions>,
) -> Option<ResizedImage> {
    let defaults = ImageResizeOptions::default();
    let opts = options.unwrap_or(&defaults);
    let input_base64_size = input_bytes.len().div_ceil(3) * 4;

    let decoded = image::load_from_memory(input_bytes).ok()?;
    let oriented = apply_exif_orientation(decoded.to_rgba8(), input_bytes);
    let image = DynamicImage::ImageRgba8(oriented);

    let original_width = image.width();
    let original_height = image.height();
    let format = mime_type.split('/').nth(1).unwrap_or("png");

    // Check if already within all limits (dimensions AND encoded size)
    if original_width <= opts.max_width()
        && original_height <= opts.max_height()
        && input_base64_size < opts.max_bytes()
    {
        return Some(ResizedImage {
            data: base64::engine::general_purpose::STANDARD.encode(input_bytes),
            mime_type: if mime_type.is_empty() {
                format!("image/{format}")
            } else {
                mime_type.to_string()
            },
            original_width,
            original_height,
            width: original_width,
            height: original_height,
            was_resized: false,
        });
    }

    // Calculate initial dimensions respecting max limits
    let mut target_width = original_width;
    let mut target_height = original_height;

    // The JS `Math.round(x)` runs on f64 operands the magnitudes of which
    // stay far inside the exact range, and a clamped target dimension
    // cannot exceed the source's, so the u32 cast loses nothing.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the scaled dimension is clamped against the source image's u32 dimensions; the f64 operands stay exact and non-negative"
    )]
    {
        if target_width > opts.max_width() {
            target_height = ((f64::from(target_height) * f64::from(opts.max_width()))
                / f64::from(target_width))
            .round() as u32;
            target_width = opts.max_width();
        }
        if target_height > opts.max_height() {
            target_width = ((f64::from(target_width) * f64::from(opts.max_height()))
                / f64::from(target_height))
            .round() as u32;
            target_height = opts.max_height();
        }
    }

    let try_encodings = |width: u32, height: u32, jpeg_qualities: &[u8]| -> Vec<EncodedCandidate> {
        let resized = image.resize_exact(width, height, FilterType::Lanczos3);
        let mut candidates =
            vec![encode_png(&resized).map(|bytes| encode_candidate(&bytes, "image/png"))];
        for quality in jpeg_qualities {
            candidates.push(
                encode_jpeg(&resized, *quality).map(|bytes| encode_candidate(&bytes, "image/jpeg")),
            );
        }
        candidates.into_iter().flatten().collect()
    };

    // Upstream's `Array.from(new Set([...]))`: dedup keeping first-seen order.
    let mut quality_steps = vec![opts.jpeg_quality()];
    for quality in [85u8, 70, 55, 40] {
        if !quality_steps.contains(&quality) {
            quality_steps.push(quality);
        }
    }
    let mut current_width = target_width;
    let mut current_height = target_height;

    loop {
        let candidates = try_encodings(current_width, current_height, &quality_steps);
        for candidate in candidates {
            if candidate.encoded_size < opts.max_bytes() {
                return Some(ResizedImage {
                    data: candidate.data,
                    mime_type: candidate.mime_type.to_string(),
                    original_width,
                    original_height,
                    width: current_width,
                    height: current_height,
                    was_resized: true,
                });
            }
        }

        if current_width == 1 && current_height == 1 {
            break;
        }

        // The shrink is floored and clamped to at least 1, so the u32 cast
        // loses nothing; the f64 operands stay far inside the exact range.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the shrink is floored and clamped to at least 1, so the u32 cast loses nothing"
        )]
        let (next_width, next_height) = {
            let next_width = if current_width == 1 {
                1
            } else {
                (f64::from(current_width) * 0.75).floor().max(1.0) as u32
            };
            let next_height = if current_height == 1 {
                1
            } else {
                (f64::from(current_height) * 0.75).floor().max(1.0) as u32
            };
            (next_width, next_height)
        };
        if next_width == current_width && next_height == current_height {
            break;
        }

        current_width = next_width;
        current_height = next_height;
    }

    None
}

/// Format a dimension note for resized images, upstream's
/// `formatDimensionNote` — it helps the model understand the coordinate
/// mapping.
#[must_use]
pub fn format_dimension_note(result: &ResizedImage) -> Option<String> {
    if !result.was_resized {
        return None;
    }

    let scale = f64::from(result.original_width) / f64::from(result.width);
    Some(format!(
        "[Image: original {}x{}, displayed at {}x{}. Multiply coordinates by {:.2} to map to original image.]",
        result.original_width, result.original_height, result.width, result.height, scale
    ))
}

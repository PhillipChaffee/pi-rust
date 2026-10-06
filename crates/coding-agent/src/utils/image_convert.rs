//! Image conversion to PNG, upstream's `src/utils/image-convert.ts`.
//!
//! The WASM photon decoder restate onto the `image` crate: any format the
//! crate decodes (the five inline-pipeline formats plus BMP) converts, and
//! the EXIF orientation applies to the decoded pixels before the PNG
//! encode.

use std::io::Cursor;

use base64::Engine as _;
use image::ImageFormat;

use super::exif_orientation::apply_exif_orientation;

/// Convert image bytes to PNG, upstream's `convertImageBytesToPng`.
///
/// Returns `None` when the bytes do not decode — photon's unavailable
/// module and its conversion failures read the same.
#[must_use]
pub fn convert_image_bytes_to_png(bytes: &[u8]) -> Option<Vec<u8>> {
    let decoded = image::load_from_memory(bytes).ok()?;
    let oriented = apply_exif_orientation(decoded.to_rgba8(), bytes);
    let mut encoded = Cursor::new(Vec::new());
    oriented.write_to(&mut encoded, ImageFormat::Png).ok()?;
    Some(encoded.into_inner())
}

/// Convert a base64 image payload to PNG for terminal display, upstream's
/// `convertToPng` — the Kitty graphics protocol requires PNG format
/// (f=100).
///
/// Already-PNG payloads pass through without a decode round-trip.
#[must_use]
pub fn convert_to_png(base64_data: &str, mime_type: &str) -> Option<(String, String)> {
    // Already PNG, no conversion needed
    if mime_type == "image/png" {
        return Some((base64_data.to_string(), mime_type.to_string()));
    }

    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(base64_data) else {
        return None;
    };
    let png_bytes = convert_image_bytes_to_png(&bytes)?;
    Some((
        base64::engine::general_purpose::STANDARD.encode(png_bytes),
        "image/png".to_string(),
    ))
}

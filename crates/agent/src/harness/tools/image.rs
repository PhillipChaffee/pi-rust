//! Image sniffing for the read tool, ported from upstream
//! `src/harness/tools/image.ts`.
//!
//! The detector reads magic bytes only — no decoder — and reports the five
//! provider-supported types; anything malformed, animated, or unsupported is
//! `None` so the read tool falls back to text handling. `encodeBase64`
//! restates onto the workspace `base64` crate (upstream hand-rolls the
//! alphabet walk because JavaScript has no byte-to-base64 builtin).

use base64::Engine as _;

/// The PNG magic, upstream's `PNG_SIGNATURE`.
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/// Detects a provider-supported image MIME type from magic bytes, upstream's
/// `detectSupportedImageMimeType`.
///
/// An animated PNG, and unsupported or malformed formats, report `None`.
#[must_use]
pub fn detect_supported_image_mime_type(buffer: &[u8]) -> Option<&'static str> {
    if starts_with(buffer, &[0xff, 0xd8, 0xff]) {
        return if buffer[3] == 0xf7 {
            None
        } else {
            Some("image/jpeg")
        };
    }
    if starts_with(buffer, &PNG_SIGNATURE) {
        return if is_png(buffer) && !is_animated_png(buffer) {
            Some("image/png")
        } else {
            None
        };
    }
    if starts_with_ascii(buffer, 0, b"GIF") {
        return Some("image/gif");
    }
    if starts_with_ascii(buffer, 0, b"RIFF") && starts_with_ascii(buffer, 8, b"WEBP") {
        return Some("image/webp");
    }
    if starts_with_ascii(buffer, 0, b"BM") && is_bmp(buffer) {
        return Some("image/bmp");
    }
    None
}

/// Encodes bytes as standard base64, upstream's `encodeBase64`
/// (`Buffer.prototype.toString("base64")`).
#[must_use]
pub fn encode_base64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn is_png(buffer: &[u8]) -> bool {
    buffer.len() >= 16
        && read_uint32_be(buffer, PNG_SIGNATURE.len()) == 13
        && starts_with_ascii(buffer, 12, b"IHDR")
}

fn is_animated_png(buffer: &[u8]) -> bool {
    let mut offset = PNG_SIGNATURE.len();
    while offset + 8 <= buffer.len() {
        let chunk_length = read_uint32_be(buffer, offset);
        let chunk_type_offset = offset + 4;
        if starts_with_ascii(buffer, chunk_type_offset, b"acTL") {
            return true;
        }
        if starts_with_ascii(buffer, chunk_type_offset, b"IDAT") {
            return false;
        }
        let next_offset = offset + 8 + chunk_length as usize + 4;
        if next_offset <= offset || next_offset > buffer.len() {
            return false;
        }
        offset = next_offset;
    }
    false
}

fn is_bmp(buffer: &[u8]) -> bool {
    if buffer.len() < 26 {
        return false;
    }
    let declared_file_size = read_uint32_le(buffer, 2);
    let pixel_data_offset = read_uint32_le(buffer, 10);
    let dib_header_size = read_uint32_le(buffer, 14);
    if declared_file_size != 0 && declared_file_size < 26 {
        return false;
    }
    if u64::from(pixel_data_offset) < 14 + u64::from(dib_header_size) {
        return false;
    }
    if declared_file_size != 0 && pixel_data_offset >= declared_file_size {
        return false;
    }

    let (color_planes, bits_per_pixel) = if dib_header_size == 12 {
        (read_uint16_le(buffer, 22), read_uint16_le(buffer, 24))
    } else if (40..=124).contains(&dib_header_size) {
        if buffer.len() < 30 {
            return false;
        }
        (read_uint16_le(buffer, 26), read_uint16_le(buffer, 28))
    } else {
        return false;
    };
    color_planes == 1 && matches!(bits_per_pixel, 1 | 4 | 8 | 16 | 24 | 32)
}

fn read_uint16_le(buffer: &[u8], offset: usize) -> u32 {
    u16::from_le_bytes([buffer[offset], buffer[offset + 1]]).into()
}

const fn read_uint32_be(buffer: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        buffer[offset],
        buffer[offset + 1],
        buffer[offset + 2],
        buffer[offset + 3],
    ])
}

const fn read_uint32_le(buffer: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        buffer[offset],
        buffer[offset + 1],
        buffer[offset + 2],
        buffer[offset + 3],
    ])
}

fn starts_with(buffer: &[u8], bytes: &[u8]) -> bool {
    buffer.len() >= bytes.len() && buffer.starts_with(bytes)
}

fn starts_with_ascii(buffer: &[u8], offset: usize, text: &[u8]) -> bool {
    buffer.len() >= offset + text.len() && &buffer[offset..offset + text.len()] == text
}

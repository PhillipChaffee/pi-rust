//! Image MIME sniffing from magic bytes, upstream's `src/utils/mime.ts`.

use std::fs::File;
use std::io::Read;

/// How many leading bytes the sniffer reads from a file.
const IMAGE_TYPE_SNIFF_BYTES: usize = 4100;

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/// Detect the image formats the inline pipeline supports from magic bytes.
///
/// APNG files read as null — upstream checks the `acTL` chunk before the
/// first `IDAT` — and the progressive-JPEG marker (`0xff 0xf7`) reads as
/// null with the plain `image/jpeg` magic. BMP validation walks the DIB
/// header so plain text beginning with `"BM"` does not sniff as an image.
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
    if starts_with_ascii(buffer, 0, "GIF") {
        return Some("image/gif");
    }
    if starts_with_ascii(buffer, 0, "RIFF") && starts_with_ascii(buffer, 8, "WEBP") {
        return Some("image/webp");
    }
    if starts_with_ascii(buffer, 0, "BM") && is_bmp(buffer) {
        return Some("image/bmp");
    }
    None
}

/// Sniff a file's first `IMAGE_TYPE_SNIFF_BYTES` bytes.
///
/// # Errors
/// An [`std::io::Error`] when the file cannot be opened or read.
pub fn detect_supported_image_mime_type_from_file(
    file_path: &str,
) -> std::io::Result<Option<&'static str>> {
    let mut file = File::open(file_path)?;
    let mut buffer = vec![0u8; IMAGE_TYPE_SNIFF_BYTES];
    let mut filled = 0usize;
    while filled < IMAGE_TYPE_SNIFF_BYTES {
        match file.read(&mut buffer[filled..])? {
            0 => break,
            read => filled += read,
        }
    }
    buffer.truncate(filled);
    Ok(detect_supported_image_mime_type(&buffer))
}

fn is_png(buffer: &[u8]) -> bool {
    buffer.len() >= 16
        && read_u32_be(buffer, PNG_SIGNATURE.len()) == 13
        && starts_with_ascii(buffer, 12, "IHDR")
}

fn is_animated_png(buffer: &[u8]) -> bool {
    let mut offset = PNG_SIGNATURE.len();
    while offset + 8 <= buffer.len() {
        let chunk_length = read_u32_be(buffer, offset);
        let chunk_type_offset = offset + 4;
        if starts_with_ascii(buffer, chunk_type_offset, "acTL") {
            return true;
        }
        if starts_with_ascii(buffer, chunk_type_offset, "IDAT") {
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

    let declared_file_size = read_u32_le(buffer, 2);
    let pixel_data_offset = read_u32_le(buffer, 10);
    let dib_header_size = read_u32_le(buffer, 14);
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
        (read_u16_le(buffer, 22), read_u16_le(buffer, 24))
    } else if (40..=124).contains(&dib_header_size) {
        if buffer.len() < 30 {
            return false;
        }
        (read_u16_le(buffer, 26), read_u16_le(buffer, 28))
    } else {
        return false;
    };

    color_planes == 1 && matches!(bits_per_pixel, 1 | 4 | 8 | 16 | 24 | 32)
}

fn read_u16_le(buffer: &[u8], offset: usize) -> u16 {
    let low = u16::from(buffer.get(offset).copied().unwrap_or(0));
    let high = u16::from(buffer.get(offset + 1).copied().unwrap_or(0));
    low | (high << 8)
}

fn read_u32_be(buffer: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        buffer.get(offset).copied().unwrap_or(0),
        buffer.get(offset + 1).copied().unwrap_or(0),
        buffer.get(offset + 2).copied().unwrap_or(0),
        buffer.get(offset + 3).copied().unwrap_or(0),
    ])
}

fn read_u32_le(buffer: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        buffer.get(offset).copied().unwrap_or(0),
        buffer.get(offset + 1).copied().unwrap_or(0),
        buffer.get(offset + 2).copied().unwrap_or(0),
        buffer.get(offset + 3).copied().unwrap_or(0),
    ])
}

fn starts_with(buffer: &[u8], bytes: &[u8]) -> bool {
    buffer.len() >= bytes.len() && buffer.starts_with(bytes)
}

fn starts_with_ascii(buffer: &[u8], offset: usize, text: &str) -> bool {
    buffer.len() >= offset + text.len()
        && buffer[offset..offset + text.len()]
            .iter()
            .zip(text.as_bytes())
            .all(|(byte, expected)| byte == expected)
}

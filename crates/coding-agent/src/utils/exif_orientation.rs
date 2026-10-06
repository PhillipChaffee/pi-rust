//! EXIF orientation parsing and application, upstream's
//! `src/utils/exif-orientation.ts`.
//!
//! The byte parsers port 1:1 — the JPEG marker walk, the WebP RIFF chunk
//! walk, and the TIFF IFD read all pin exact offsets. The application
//! restate onto the `image` crate's buffers: the WASM `PhotonImage` calls
//! become the equivalent `imageops` transforms, with the same destination
//! index math the rotations upstream hand-roll (the `rotate90` closure
//! folds to [`image::imageops::rotate90`] and its mirror to
//! [`image::imageops::rotate270`], verified against the index algebra).

use image::RgbaImage;
use image::imageops;

/// Read the orientation tag out of a TIFF header's IFD.
fn read_orientation_from_tiff(bytes: &[u8], tiff_start: usize) -> u16 {
    if tiff_start + 8 > bytes.len() {
        return 1;
    }

    let byte_order = ((u16::from(bytes[tiff_start])) << 8) | u16::from(bytes[tiff_start + 1]);
    let le = byte_order == 0x4949;

    let read16 = |pos: usize| -> u16 {
        if le {
            u16::from(bytes[pos]) | (u16::from(bytes[pos + 1]) << 8)
        } else {
            (u16::from(bytes[pos]) << 8) | u16::from(bytes[pos + 1])
        }
    };
    let read32 = |pos: usize| -> u32 {
        if le {
            u32::from(bytes[pos])
                | (u32::from(bytes[pos + 1]) << 8)
                | (u32::from(bytes[pos + 2]) << 16)
                | (u32::from(bytes[pos + 3]) << 24)
        } else {
            (u32::from(bytes[pos]) << 24)
                | (u32::from(bytes[pos + 1]) << 16)
                | (u32::from(bytes[pos + 2]) << 8)
                | u32::from(bytes[pos + 3])
        }
    };

    let ifd_offset = read32(tiff_start + 4);
    let ifd_start = tiff_start + usize::try_from(ifd_offset).unwrap_or(0);
    if ifd_start + 2 > bytes.len() {
        return 1;
    }

    let entry_count = read16(ifd_start);
    for entry in 0..entry_count {
        let entry_pos = ifd_start + 2 + entry as usize * 12;
        if entry_pos + 12 > bytes.len() {
            return 1;
        }

        if read16(entry_pos) == 0x0112 {
            let value = read16(entry_pos + 8);
            return if (1..=8).contains(&value) { value } else { 1 };
        }
    }

    1
}

/// Walk the JPEG marker chain to the EXIF TIFF header.
fn find_jpeg_tiff_offset(bytes: &[u8]) -> isize {
    let mut offset = 2usize;
    while offset < bytes.len().saturating_sub(1) {
        if bytes[offset] != 0xff {
            return -1;
        }
        let marker = bytes[offset + 1];
        if marker == 0xff {
            offset += 1;
            continue;
        }

        if marker == 0xe1 {
            if offset + 4 >= bytes.len() {
                return -1;
            }
            let segment_start = offset + 4;
            if segment_start + 6 > bytes.len() {
                return -1;
            }
            if has_exif_header(bytes, segment_start) {
                return (segment_start + 6).cast_signed();
            }
        }

        if offset + 4 > bytes.len() {
            return -1;
        }
        let length = ((u16::from(bytes[offset + 2])) << 8) | u16::from(bytes[offset + 3]);
        offset += 2 + length as usize;
    }

    -1
}

/// Walk the WebP RIFF chunk list to the EXIF chunk's TIFF header.
fn find_webp_tiff_offset(bytes: &[u8]) -> isize {
    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let chunk_id: String = bytes[offset..offset + 4]
            .iter()
            .map(|byte| *byte as char)
            .collect();
        let chunk_size = u32::from(bytes[offset + 4])
            | (u32::from(bytes[offset + 5]) << 8)
            | (u32::from(bytes[offset + 6]) << 16)
            | (u32::from(bytes[offset + 7]) << 24);
        let data_start = offset + 8;

        if chunk_id == "EXIF" {
            if data_start + chunk_size as usize > bytes.len() {
                return -1;
            }
            // Some WebP files have "Exif\0\0" prefix before the TIFF header
            let tiff_start = if chunk_size >= 6 && has_exif_header(bytes, data_start) {
                data_start + 6
            } else {
                data_start
            };
            return tiff_start.cast_signed();
        }

        // RIFF chunks are padded to even size
        offset = data_start + chunk_size as usize + (chunk_size % 2) as usize;
    }

    -1
}

fn has_exif_header(bytes: &[u8], offset: usize) -> bool {
    bytes.get(offset..offset + 6) == Some(&[0x45, 0x78, 0x69, 0x66, 0x00, 0x00])
}

/// The EXIF orientation value, 1 when the bytes carry none.
#[must_use]
pub fn get_exif_orientation(bytes: &[u8]) -> u16 {
    let tiff_offset = if bytes.len() >= 2 && bytes[0] == 0xff && bytes[1] == 0xd8 {
        // JPEG: starts with FF D8
        find_jpeg_tiff_offset(bytes)
    } else if bytes.len() >= 12
        && bytes[0] == 0x52
        && bytes[1] == 0x49
        && bytes[2] == 0x46
        && bytes[3] == 0x46
        && bytes[8] == 0x57
        && bytes[9] == 0x45
        && bytes[10] == 0x42
        && bytes[11] == 0x50
    {
        // WebP: starts with RIFF....WEBP
        find_webp_tiff_offset(bytes)
    } else {
        -1
    };

    if tiff_offset == -1 {
        return 1;
    }
    read_orientation_from_tiff(bytes, tiff_offset.cast_unsigned())
}

/// Apply the orientation the original bytes carry, upstream's
/// `applyExifOrientation`. The flips mutate; the rotations return a new
/// image, so the caller receives the owned result either way.
#[must_use]
pub fn apply_exif_orientation(image: RgbaImage, original_bytes: &[u8]) -> RgbaImage {
    let orientation = get_exif_orientation(original_bytes);
    if orientation == 1 {
        return image;
    }

    match orientation {
        2 => imageops::flip_horizontal(&image),
        3 => imageops::flip_vertical(&imageops::flip_horizontal(&image)),
        4 => imageops::flip_vertical(&image),
        5 => imageops::flip_horizontal(&imageops::rotate90(&image)),
        6 => imageops::rotate90(&image),
        7 => imageops::flip_horizontal(&imageops::rotate270(&image)),
        8 => imageops::rotate270(&image),
        _ => image,
    }
}

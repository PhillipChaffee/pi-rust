//! Boundary tests for the MIME sniffer's table walk: the APNG chunk scan,
//! the BMP DIB-header variants, and the short-buffer guards.

use pi_coding_agent::utils::mime::detect_supported_image_mime_type;

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

fn png_chunk(kind: &[u8], body_len: u32) -> Vec<u8> {
    let mut chunk = (body_len).to_be_bytes().to_vec();
    chunk.extend_from_slice(kind);
    chunk.extend(std::iter::repeat_n(0u8, body_len as usize + 4));
    chunk
}

fn ihdr_chunk() -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&13u32.to_be_bytes());
    ihdr.extend_from_slice(b"IHDR");
    ihdr.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 1, 8, 0, 0, 0, 0]);
    ihdr.extend_from_slice(&[0, 0, 0, 0]); // CRC
    ihdr
}

fn png_with_chunks(chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut png = PNG_SIGNATURE.to_vec();
    png.extend(ihdr_chunk());
    for chunk in chunks {
        png.extend(chunk);
    }
    png
}

#[test]
fn jpeg_magic_with_the_progressive_marker_reads_null() {
    assert_eq!(
        detect_supported_image_mime_type(&[0xff, 0xd8, 0xff, 0xf7]),
        None
    );
    assert_eq!(
        detect_supported_image_mime_type(&[0xff, 0xd8, 0xff, 0xe0]),
        Some("image/jpeg")
    );
    // A short JFIF prefix without the third byte is not a JPEG.
    assert_eq!(detect_supported_image_mime_type(&[0xff, 0xd8]), None);
}

#[test]
fn png_requires_the_ihdr_chunk() {
    // A well-formed IHDR reads image/png.
    assert_eq!(
        detect_supported_image_mime_type(&png_with_chunks(&[])),
        Some("image/png")
    );
    // An IHDR with a length other than 13 reads null, upstream's isPng gate.
    let mut wrong = PNG_SIGNATURE.to_vec();
    wrong.extend(png_chunk(b"IHDR", 4));
    assert_eq!(detect_supported_image_mime_type(&wrong), None);
}

#[test]
fn animated_png_reads_null_before_idat() {
    // acTL before IDAT marks the animation.
    let png = png_with_chunks(&[png_chunk(b"acTL", 4), png_chunk(b"IDAT", 4)]);
    assert_eq!(detect_supported_image_mime_type(&png), None);
    // IDAT first: a plain PNG even when acTL would come later.
    let png = png_with_chunks(&[png_chunk(b"IDAT", 4), png_chunk(b"acTL", 4)]);
    assert_eq!(detect_supported_image_mime_type(&png), Some("image/png"));
}

#[test]
fn gif_riff_and_webp_magics_read_their_types() {
    assert_eq!(
        detect_supported_image_mime_type(b"GIF89a"),
        Some("image/gif")
    );
    assert_eq!(
        detect_supported_image_mime_type(b"GIF87a"),
        Some("image/gif")
    );
    let mut webp = b"RIFF".to_vec();
    webp.extend_from_slice(&4u32.to_le_bytes());
    webp.extend_from_slice(b"WEBP");
    assert_eq!(detect_supported_image_mime_type(&webp), Some("image/webp"));
    // RIFF without WEBP is not a WebP.
    let mut riff = b"RIFF".to_vec();
    riff.extend_from_slice(&4u32.to_le_bytes());
    riff.extend_from_slice(b"WAVE");
    assert_eq!(detect_supported_image_mime_type(&riff), None);
}

fn bmp(dib_header_size: u32, planes: u16, bpp: u16) -> Vec<u8> {
    const FILE_SIZE: u32 = 58;
    let mut buffer = vec![0u8; FILE_SIZE as usize];
    buffer[0..2].copy_from_slice(b"BM");
    buffer[2..6].copy_from_slice(&FILE_SIZE.to_le_bytes());
    buffer[10..14].copy_from_slice(&54u32.to_le_bytes()); // pixel data offset
    buffer[14..18].copy_from_slice(&dib_header_size.to_le_bytes());
    // BITMAPINFOHEADER layout: width/height at 18, planes at 26, bpp at 28.
    // BITMAPCOREHEADER layout: width/height at 18, planes at 22, bpp at 24.
    let (plane_pos, bpp_pos) = if dib_header_size == 12 {
        (22usize, 24usize)
    } else {
        (26, 28)
    };
    buffer[plane_pos..plane_pos + 2].copy_from_slice(&planes.to_le_bytes());
    buffer[bpp_pos..bpp_pos + 2].copy_from_slice(&bpp.to_le_bytes());
    buffer
}

#[test]
fn bmp_reads_both_dib_header_shapes() {
    // BITMAPINFOHEADER (40) with one plane at 24bpp.
    assert_eq!(
        detect_supported_image_mime_type(&bmp(40, 1, 24)),
        Some("image/bmp")
    );
    // BITMAPCOREHEADER (12) with its own plane/bpp offsets.
    assert_eq!(
        detect_supported_image_mime_type(&bmp(12, 1, 24)),
        Some("image/bmp")
    );
    // Zero planes reject.
    assert_eq!(detect_supported_image_mime_type(&bmp(40, 0, 24)), None);
    // A bpp outside the table rejects.
    assert_eq!(detect_supported_image_mime_type(&bmp(40, 1, 12)), None);
    // A DIB header size outside 12 and 40..=124 rejects.
    assert_eq!(detect_supported_image_mime_type(&bmp(13, 1, 24)), None);
    assert_eq!(detect_supported_image_mime_type(&bmp(200, 1, 24)), None);
}

#[test]
fn bmp_size_consistency_gates() {
    // A declared file size below the header size rejects.
    let mut short = bmp(40, 1, 24);
    short[2..6].copy_from_slice(&10u32.to_le_bytes());
    assert_eq!(detect_supported_image_mime_type(&short), None);
    // A pixel-data offset inside the headers rejects.
    let mut overlapped = bmp(40, 1, 24);
    overlapped[10..14].copy_from_slice(&20u32.to_le_bytes());
    assert_eq!(detect_supported_image_mime_type(&overlapped), None);
    // A zero declared file size skips that gate.
    let mut headless = bmp(40, 1, 24);
    headless[2..6].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(
        detect_supported_image_mime_type(&headless),
        Some("image/bmp")
    );
    // A tiny buffer without the full header rejects.
    assert_eq!(detect_supported_image_mime_type(b"BM00"), None);
}

#[test]
fn short_buffers_never_panic() {
    // Every magic prefix shorter than the scan reads None, not a panic.
    for length in 0..12 {
        let slice = &PNG_SIGNATURE[..length.min(8)];
        assert_eq!(detect_supported_image_mime_type(slice), None);
    }
}

#[test]
fn a_dib_header_declared_past_the_buffer_reads_null() {
    // A 40-byte DIB declaration with fewer than 30 bytes on disk cannot
    // carry the color-plane fields, upstream's bounds check.
    let mut bmp = Vec::new();
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&0u32.to_le_bytes()); // declared file size
    bmp.extend_from_slice(&100u32.to_le_bytes()); // pixel data offset
    bmp.extend_from_slice(&40u32.to_le_bytes()); // DIB header size
    bmp.extend_from_slice(&[0u8; 12]); // 26 bytes total
    assert_eq!(detect_supported_image_mime_type(&bmp), None);
}

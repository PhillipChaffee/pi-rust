//! The tool-result-images suite, upstream's
//! `test/tool-result-images.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The PNG fixtures build by hand the way upstream's `node:zlib` builder
//! does; the identity cases assert array equality, the port of upstream's
//! `toBe(content)` identity contract.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use base64::Engine as _;
use pi_ai::types::{ImageContent, TextContent};
use pi_coding_agent::utils::tool_result_images::{
    NormalizeToolResultImagesOptions, ToolResultContent, normalize_tool_result_images,
};

const TINY_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

fn png_chunk(kind: &[u8], body: &[u8]) -> Vec<u8> {
    let mut header = Vec::new();
    // The fixture bodies are tens of bytes; the u32 cast loses nothing.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "fixture bodies are tens of bytes"
    )]
    let length = body.len() as u32;
    header.extend_from_slice(&length.to_be_bytes());
    header.extend_from_slice(kind);
    let mut checksum_input = Vec::new();
    checksum_input.extend_from_slice(kind);
    checksum_input.extend_from_slice(body);
    let mut chunk = header;
    chunk.extend_from_slice(body);
    chunk.extend_from_slice(&crc32(&checksum_input).to_be_bytes());
    chunk
}

/// The zlib crc32, upstream's `node:zlib` helper.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Build an 8-bit grayscale PNG of arbitrary dimensions without an encoder.
fn create_png(width: u32, height: u32) -> Vec<u8> {
    let mut ihdr = vec![0u8; 13];
    ihdr[0..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = 8; // bit depth
    ihdr[9] = 0; // color type: grayscale
    let mut raw = Vec::with_capacity((width as usize + 1) * height as usize);
    for row in 0..height {
        raw.push(0); // no filter
        raw.extend(std::iter::repeat_n(
            u8::try_from(row % 256).expect("fixture row"),
            width as usize,
        ));
    }
    // PNG's IDAT carries a zlib stream, not raw deflate.
    let mut deflate_input = Vec::new();
    let mut encoder =
        flate2::write::ZlibEncoder::new(&mut deflate_input, flate2::Compression::default());
    std::io::Write::write_all(&mut encoder, &raw).expect("deflate");
    std::io::Write::flush(&mut encoder).expect("deflate");
    drop(encoder);

    let mut png = vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    png.extend(png_chunk(b"IHDR", &ihdr));
    png.extend(png_chunk(b"IDAT", &deflate_input));
    png.extend(png_chunk(b"IEND", &[]));
    png
}

fn read_png_dimensions(base64_data: &str) -> (u32, u32) {
    let buffer = base64::engine::general_purpose::STANDARD
        .decode(base64_data)
        .expect("base64");
    (
        u32::from_be_bytes([buffer[16], buffer[17], buffer[18], buffer[19]]),
        u32::from_be_bytes([buffer[20], buffer[21], buffer[22], buffer[23]]),
    )
}

fn create_tiny_bmp_1x1_red_24bpp() -> Vec<u8> {
    const FILE_SIZE: u32 = 58;
    let mut buffer = vec![0u8; FILE_SIZE as usize];
    buffer[0..2].copy_from_slice(b"BM");
    buffer[2..6].copy_from_slice(&FILE_SIZE.to_le_bytes());
    buffer[10..14].copy_from_slice(&54u32.to_le_bytes());
    buffer[14..18].copy_from_slice(&40u32.to_le_bytes());
    buffer[18..22].copy_from_slice(&1i32.to_le_bytes());
    buffer[22..26].copy_from_slice(&1i32.to_le_bytes());
    buffer[26..28].copy_from_slice(&1u16.to_le_bytes());
    buffer[28..30].copy_from_slice(&24u16.to_le_bytes());
    buffer[30..34].copy_from_slice(&0u32.to_le_bytes());
    buffer[34..38].copy_from_slice(&4u32.to_le_bytes());
    buffer[56] = 0xff;
    buffer
}

fn image_block(bytes: &[u8], mime_type: &str) -> ToolResultContent {
    ToolResultContent::Image(ImageContent {
        data: base64::engine::general_purpose::STANDARD.encode(bytes),
        mime_type: mime_type.to_string(),
    })
}

fn text_block(text: &str) -> ToolResultContent {
    ToolResultContent::Text(TextContent {
        text: text.to_string(),
        text_signature: None,
    })
}

// === normalizeToolResultImages ==============================================

#[tokio::test]
async fn returns_the_original_array_when_there_are_no_image_blocks() {
    let content = vec![text_block("no images here")];
    let normalized = normalize_tool_result_images(content.clone(), None).await;
    assert_eq!(normalized, content);
}

#[tokio::test]
async fn returns_an_equal_array_when_images_are_already_within_limits() {
    let content = vec![
        text_block("screenshot"),
        image_block(
            &base64::engine::general_purpose::STANDARD
                .decode(TINY_PNG_BASE64)
                .expect("fixture"),
            "image/png",
        ),
    ];
    let normalized = normalize_tool_result_images(content.clone(), None).await;
    assert_eq!(normalized, content);
}

#[tokio::test]
async fn resizes_oversized_images_and_reports_the_original_dimensions() {
    let content = vec![image_block(&create_png(2400, 4800), "image/png")];

    let normalized = normalize_tool_result_images(content, None).await;

    assert_eq!(normalized.len(), 2);
    let ToolResultContent::Image(image) = &normalized[0] else {
        panic!("image block");
    };
    let (width, height) = read_png_dimensions(&image.data);
    assert!(width <= 2000);
    assert!(height <= 2000);
    let ToolResultContent::Text(note) = &normalized[1] else {
        panic!("text block");
    };
    assert!(note.text.contains("original 2400x4800"), "{}", note.text);
}

#[tokio::test]
async fn leaves_oversized_images_alone_when_auto_resize_is_disabled() {
    let content = vec![image_block(&create_png(2400, 4800), "image/png")];

    let normalized = normalize_tool_result_images(
        content.clone(),
        Some(&NormalizeToolResultImagesOptions {
            auto_resize_images: Some(false),
        }),
    )
    .await;

    assert_eq!(normalized, content);
}

#[tokio::test]
async fn converts_unsupported_image_formats_even_when_auto_resize_is_disabled() {
    let content = vec![image_block(&create_tiny_bmp_1x1_red_24bpp(), "image/bmp")];

    let normalized = normalize_tool_result_images(
        content,
        Some(&NormalizeToolResultImagesOptions {
            auto_resize_images: Some(false),
        }),
    )
    .await;

    assert_eq!(normalized.len(), 2);
    let ToolResultContent::Image(image) = &normalized[0] else {
        panic!("image block");
    };
    assert_eq!(image.mime_type, "image/png");
    let ToolResultContent::Text(note) = &normalized[1] else {
        panic!("text block");
    };
    assert_eq!(note.text, "[Image converted from image/bmp to image/png.]");
}

#[tokio::test]
async fn keeps_undecodable_images_instead_of_dropping_tool_output() {
    let content = vec![image_block(b"not-an-image", "image/png")];

    let normalized = normalize_tool_result_images(content.clone(), None).await;

    assert_eq!(normalized, content);
}

#[tokio::test]
async fn preserves_surrounding_text_blocks_and_their_order() {
    let content = vec![
        text_block("before"),
        image_block(&create_png(2400, 100), "image/png"),
        text_block("after"),
    ];

    let normalized = normalize_tool_result_images(content, None).await;

    let kinds: Vec<&str> = normalized
        .iter()
        .map(|block| match block {
            ToolResultContent::Text(_) => "text",
            ToolResultContent::Image(_) => "image",
        })
        .collect();
    assert_eq!(kinds, vec!["text", "image", "text", "text"]);
    assert_eq!(normalized[0], text_block("before"));
    assert_eq!(normalized[3], text_block("after"));
}

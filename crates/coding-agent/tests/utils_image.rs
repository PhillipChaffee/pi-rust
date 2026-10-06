//! The image-belt suites, upstream's `test/image-processing.test.ts` and
//! `test/image-process.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The photon WASM calls restate onto the `image` crate; the conversion
//! expectations pin magic bytes and dimensions rather than byte-identical
//! encoders, the way the photon-driven originals pin photon's outputs.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use base64::Engine as _;
use pi_coding_agent::utils::image_convert::convert_to_png;
use pi_coding_agent::utils::image_process::{ProcessImageOptions, process_image};
use pi_coding_agent::utils::image_resize::{
    ImageResizeOptions, format_dimension_note, resize_image,
};
use pi_coding_agent::utils::image_resize_core::resize_image_in_process;
use pi_coding_agent::utils::mime::detect_supported_image_mime_type;

// Small 2x2 red PNG image (base64) - generated with ImageMagick
const TINY_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACAQMAAABIeJ9nAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGUExURf8AAP///0EdNBEAAAABYktHRAH/Ai3eAAAAB3RJTUUH6gEOADM5Ddoh/wAAAAxJREFUCNdjYGBgAAAABAABJzQnCgAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyNi0wMS0xNFQwMDo1MTo1NyswMDowMOnKzHgAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjYtMDEtMTRUMDA6NTE6NTcrMDA6MDCYl3TEAAAAKHRFWHRkYXRlOnRpbWVzdGFtcAAyMDI2LTAxLTE0VDAwOjUxOjU3KzAwOjAwz4JVGwAAAABJRU5ErkJggg==";

// 2x3 solid-red PNG (74 bytes), synthesized from the zlib/struct layout.
const TWO_BY_THREE_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAADCAYAAAC56t6BAAAAEUlEQVR4nGP4z8DwH4QZMBgAoXkL9U3EmgcAAAAASUVORK5CYII=";

// Small 2x2 blue JPEG image (base64) - generated with ImageMagick
const TINY_JPEG: &str = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAACAAIDAREAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAVAQEBAAAAAAAAAAAAAAAAAAAGCf/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AD3VTB3/2Q==";

const TINY_JPEG_2X1: &str = "/9j/4AAQSkZJRgABAgAAAQABAAD/wAARCAABAAIDAREAAhEBAxEB/9sAQwADAgIDAgIDAwMDBAMDBAUIBQUEBAUKBwcGCAwKDAwLCgsLDQ4SEA0OEQ4LCxAWEBETFBUVFQwPFxgWFBgSFBUU/9sAQwEDBAQFBAUJBQUJFA0LDRQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQU/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD4H8Q/8h/Uv+vmX/0M1/o1wJ/ySWU/9g1D/wBNRMOM/wDkp8z/AOv9b/05I//Z";

// 100x100 gray PNG
const MEDIUM_PNG_100X100: &str = "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAAAAABVicqIAAAAAmJLR0QA/4ePzL8AAAAHdElNRQfqAQ4AMzkN2iH/AAAAP0lEQVRo3u3NQQEAAAQEMASXXYrz2gqst/Lm4ZBIJBKJRCKRSCQSiUQikUgkEolEIpFIJBKJRCKRSCQSiSTsAP1cAUZeKtreAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDI2LTAxLTE0VDAwOjUxOjU3KzAwOjAw6crMeAAAACV0RVh0ZGF0ZTptb2RpZnkAMjAyNi0wMS0xNFQwMDo1MTo1NyswMDowMJiXdMQAAAAodEVYdGRhdGU6dGltZXN0YW1wADIwMjYtMDEtMTRUMDA6NTE6NTcrMDA6MDDPglUbAAAAAElFTkSuQmCC";

// 200x200 colored PNG
const LARGE_PNG_200X200: &str = "iVBORw0KGgoAAAANSUhEUgAAAMgAAADIEAIAAAByquWKAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGYktHRP///////wlY99wAAAAHdElNRQfqAQ4AMzqU03BFAAAEEElEQVR42u3dvWqUURSF4XXMRAyIhSAWQvwnWtjaWusFx9bW1kKD/wELESxEiBjDjJdwysOC57mCXW32+zGcGbtdACpsdi+T8Xz1GABzm91xMl6sHgNgblzcT/Y+rh4DYG6cv0/2H64eA2BOEgI1fHQHaoyzbXIwVo8BMCcJgRqSEKgxft9Lrn5aPQbA3Ph1klw7Wj0GwJwkBGpYWECN8fMiub63egyAORcWUMPCAmqMH3eTG59XjwEwN76/S24+Wj0GwJwkBGpYWECN8e1fcmuzegyAORcWUMPCAmqM09vJ4dfVYwDMjS9vkzuPV48BMCcJgRqeSAZqjA/nyYP91WMAzElCoIYkBGqMk8Pk6HT1GABzm+3L5JIkBApIQqCGj+5AjfHmb/Lk8uoxAOYkIVBDEgI1LCygxmZ7nFyShEABFxZQw8ICaozXf5KnV1aPATDnwgJqWFhADQsLqOFnDUANFxZQw8ICaoxXZ8mzg9VjAMy5sIAaFhZQw3tYQA1/QgHUkIRADUkI1JCEQA1JCNSQhEANFxZQwzcsoIYkBGpIQqCGJARqSEKghiQEalhYQA1/QgHUcGEBNSwsoIYkBGq4sIAaFhZQw8ICaviGBdRwYQE1LCyghiQEariwgBoWFlDDe1hADS+OAjUkIVBDEgI1JCFQQxICNSQhUMOFBdTwDQuoIQmBGpIQqCEJgRqSEKghCYEaFhZQw4ujQA0XFlDDwgJqSEKghgsLqGFhATUsLKCGb1hADRcWUMPCAmpIQqCGCwuoYWEBNbyHBdTw4ihQQxICNSQhUEMSAjUkIVBDEgI1XFhADd+wgBqSEKghCYEakhCoIQmBGpIQqGFhATW8OArUcGEBNSwsoIYkBGq4sIAaFhZQw8ICaviGBdRwYQE1LCyghiQEariwgBoWFlDDe1hADS+OAjUkIVBDEgI1JCFQQxICNSQhUMOFBdTwDQuoIQmBGpIQqCEJgRqSEKghCYEa49lB8ups9RgAc14cBWpIQqCGhQXUkIRADRcWUMPCAmqMp1eS139WjwEw5xsWUEMSAjUsLKCGJARquLCAGhYWUGM8uZy8+bt6DIA5L44CNSQhUMMTyUCNcXSanByuHgNgThICNSQhUGM82E8+nK8eA2DOzxqAGpIQqDHuPE6+vF09BsDcOPyanN5ePQbAnCQEavgdFlBj3Nok3/6tHgNgzoujQA1JCNQYNx8l39+tHgNgbtz4nPy4u3oMgDlJCNSwsIAa4/pe8vNi9RgAc37WANSQhECNce0o+XWyegyAuXH1U/L73uoxAOYkIVDDwgJqjIORnG1XjwEw508ogBqSEKgx9h8m5+9XjwEwN/Y+Jhf3V48BMCcJgRpjPE+2x6vHAJgbSbLbrR4DYO4/GqiSgXN+ksgAAAAldEVYdGRhdGU6Y3JlYXRlADIwMjYtMDEtMTRUMDA6NTE6NTcrMDA6MDDpysx4AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDI2LTAxLTE0VDAwOjUxOjU3KzAwOjAwmJd0xAAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyNi0wMS0xNFQwMDo1MTo1NyswMDowMM+CVRsAAAAASUVORK5CYII=";

fn image_bytes(base64_data: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(base64_data)
        .expect("base64 fixture")
}

fn create_tiny_bmp_1x1_red_24bpp() -> Vec<u8> {
    // File size = 14 (BMP header) + 40 (DIB header) + 4 (pixel row) = 58
    const FILE_SIZE: u32 = 58;
    let mut buffer = vec![0u8; FILE_SIZE as usize];
    // BITMAPFILEHEADER
    buffer[0..2].copy_from_slice(b"BM");
    buffer[2..6].copy_from_slice(&FILE_SIZE.to_le_bytes());
    buffer[10..14].copy_from_slice(&54u32.to_le_bytes());
    // BITMAPINFOHEADER
    buffer[14..18].copy_from_slice(&40u32.to_le_bytes());
    buffer[18..22].copy_from_slice(&1i32.to_le_bytes()); // width
    buffer[22..26].copy_from_slice(&1i32.to_le_bytes()); // height (positive = bottom-up)
    buffer[26..28].copy_from_slice(&1u16.to_le_bytes()); // planes
    buffer[28..30].copy_from_slice(&24u16.to_le_bytes()); // bits per pixel
    buffer[30..34].copy_from_slice(&0u32.to_le_bytes()); // compression (BI_RGB)
    buffer[34..38].copy_from_slice(&4u32.to_le_bytes()); // image size (incl. padding)
    buffer[54..57].copy_from_slice(&[0x00, 0x00, 0xff]); // B, G, R
    buffer
}

fn expect_png_magic(base64_data: &str) {
    let buffer = base64::engine::general_purpose::STANDARD
        .decode(base64_data)
        .expect("base64");
    assert_eq!(&buffer[0..4], &[0x89, 0x50, 0x4e, 0x47]);
}

// === image-process.test.ts ==================================================

#[test]
fn detects_bmp_files_from_magic_bytes() {
    assert_eq!(
        detect_supported_image_mime_type(&create_tiny_bmp_1x1_red_24bpp()),
        Some("image/bmp")
    );
}

#[tokio::test]
async fn converts_bmp_files_to_png_attachments_when_auto_resize_is_disabled() {
    let result = process_image(
        create_tiny_bmp_1x1_red_24bpp(),
        "image/bmp",
        Some(&ProcessImageOptions {
            auto_resize_images: Some(false),
            resize_options: None,
        }),
    )
    .await;
    let pi_coding_agent::utils::image_process::ProcessImageResult::Ok {
        data,
        mime_type,
        hints,
    } = result
    else {
        panic!("the tiny BMP converts");
    };
    assert_eq!(mime_type, "image/png");
    assert!(hints.contains(&"[Image converted from image/bmp to image/png.]".to_string()));
    expect_png_magic(&data);
}

#[tokio::test]
async fn converts_bmp_files_before_auto_resizing() {
    let result = process_image(create_tiny_bmp_1x1_red_24bpp(), "image/bmp", None).await;
    let pi_coding_agent::utils::image_process::ProcessImageResult::Ok {
        data,
        mime_type,
        hints,
    } = result
    else {
        panic!("the tiny BMP converts");
    };
    assert_eq!(mime_type, "image/png");
    assert!(hints.contains(&"[Image converted from image/bmp to image/png.]".to_string()));
    expect_png_magic(&data);
}

#[test]
fn sniffs_the_supported_formats_and_rejects_the_rest() {
    assert_eq!(
        detect_supported_image_mime_type(&image_bytes(TINY_PNG)),
        Some("image/png")
    );
    assert_eq!(
        detect_supported_image_mime_type(&image_bytes(TINY_JPEG)),
        Some("image/jpeg")
    );
    // The BMP sniffer must reject plain text beginning with "BM".
    assert_eq!(
        detect_supported_image_mime_type(b"BM plain text, not an image"),
        None
    );
    assert_eq!(detect_supported_image_mime_type(b"plain text"), None);
}

// === image-processing.test.ts: convertToPng =================================

#[test]
fn convert_to_png_returns_original_data_for_png_input() {
    let result = convert_to_png(TINY_PNG, "image/png");
    let (data, mime_type) = result.expect("png passes through");
    assert_eq!(data, TINY_PNG);
    assert_eq!(mime_type, "image/png");
}

#[test]
fn convert_to_png_converts_jpeg_to_png() {
    let result = convert_to_png(TINY_JPEG, "image/jpeg");
    let (data, mime_type) = result.expect("jpeg converts");
    assert_eq!(mime_type, "image/png");
    expect_png_magic(&data);
}

#[test]
fn convert_to_png_applies_exif_orientation_after_an_xmp_app1_segment() {
    let jpeg = image_bytes(TINY_JPEG_2X1);
    let xmp_payload = b"http://ns.adobe.com/xap/1.0/\0<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>";
    let orientation_payload: Vec<u8> = b"Exif\0\0"
        .iter()
        .copied()
        .chain(
            // Little-endian TIFF with one IFD entry: orientation 6.
            hex("49492a0008000000010012010300010000000600000000000000"),
        )
        .collect();

    let app1 = |payload: &[u8]| {
        // The length fits u16: the fixtures are tens of bytes.
        let length = u16::try_from(payload.len() + 2).expect("fixture length");
        let mut segment = vec![0u8; payload.len() + 4];
        segment[0] = 0xff;
        segment[1] = 0xe1;
        segment[2..4].copy_from_slice(&length.to_be_bytes());
        segment[4..].copy_from_slice(payload);
        segment
    };

    let mut composed = jpeg[0..2].to_vec();
    composed.extend(app1(xmp_payload));
    composed.extend(app1(&orientation_payload));
    composed.extend(jpeg[2..].to_vec());
    let composed = base64::engine::general_purpose::STANDARD.encode(&composed);

    let result = convert_to_png(&composed, "image/jpeg");
    let (data, _) = result.expect("oriented jpeg converts");
    let png = base64::engine::general_purpose::STANDARD
        .decode(&data)
        .expect("png");
    let width = u32::from_be_bytes([png[16], png[17], png[18], png[19]]);
    let height = u32::from_be_bytes([png[20], png[21], png[22], png[23]]);
    // Orientation 6 (90 CW) swaps the 2x1 image's axes.
    assert_eq!(width, 1);
    assert_eq!(height, 2);
}

fn hex(hex_text: &str) -> Vec<u8> {
    (0..hex_text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&hex_text[at..at + 2], 16).expect("hex fixture"))
        .collect()
}

// === image-processing.test.ts: resizeImage ==================================

#[tokio::test]
async fn resize_keeps_caller_input_bytes_intact() {
    let input = image_bytes(TINY_PNG);
    let original_byte_length = input.len();
    let original_first_byte = input[0];

    let result = resize_image(
        input.clone(),
        "image/png".to_string(),
        Some(ImageResizeOptions {
            max_width: Some(100),
            max_height: Some(100),
            max_bytes: Some(1024 * 1024),
            jpeg_quality: None,
        }),
    )
    .await;

    let resized = result.expect("resize");
    assert_eq!(input.len(), original_byte_length);
    assert_eq!(input[0], original_first_byte);
    let _ = resized;
}

#[tokio::test]
async fn resize_returns_original_image_if_within_limits() {
    let result = resize_image(
        image_bytes(TINY_PNG),
        "image/png".to_string(),
        Some(ImageResizeOptions {
            max_width: Some(100),
            max_height: Some(100),
            max_bytes: Some(1024 * 1024),
            jpeg_quality: None,
        }),
    )
    .await;

    let resized = result.expect("resize");
    assert!(!resized.was_resized);
    assert_eq!(resized.data, TINY_PNG);
    assert_eq!(resized.original_width, 2);
    assert_eq!(resized.original_height, 2);
    assert_eq!(resized.width, 2);
    assert_eq!(resized.height, 2);
}

#[tokio::test]
async fn resize_handles_images_exceeding_dimension_limits() {
    let result = resize_image(
        image_bytes(MEDIUM_PNG_100X100),
        "image/png".to_string(),
        Some(ImageResizeOptions {
            max_width: Some(50),
            max_height: Some(50),
            max_bytes: Some(1024 * 1024),
            jpeg_quality: None,
        }),
    )
    .await;

    let resized = result.expect("resize");
    assert!(resized.was_resized);
    assert_eq!(resized.original_width, 100);
    assert_eq!(resized.original_height, 100);
    assert!(resized.width <= 50);
    assert!(resized.height <= 50);
}

#[tokio::test]
async fn resize_handles_images_exceeding_byte_limits() {
    let original = image_bytes(LARGE_PNG_200X200);
    let original_size = original.len();
    let encoded_length = LARGE_PNG_200X200.len();

    // Set maxBytes to less than the original encoded image size
    let result = resize_image(
        original,
        "image/png".to_string(),
        Some(ImageResizeOptions {
            max_width: Some(2000),
            max_height: Some(2000),
            max_bytes: Some(encoded_length * 9 / 10),
            jpeg_quality: None,
        }),
    )
    .await;

    let resized = result.expect("shrink");
    let result_buffer = base64::engine::general_purpose::STANDARD
        .decode(&resized.data)
        .expect("base64");
    assert!(result_buffer.len() < original_size);
    assert!(resized.data.len() < encoded_length);
}

#[tokio::test]
async fn resize_returns_null_when_image_cannot_be_resized_below_max_bytes() {
    let result = resize_image(
        image_bytes(LARGE_PNG_200X200),
        "image/png".to_string(),
        Some(ImageResizeOptions {
            max_width: Some(2000),
            max_height: Some(2000),
            max_bytes: Some(1),
            jpeg_quality: None,
        }),
    )
    .await;

    assert_eq!(result, None);
}

#[tokio::test]
async fn resize_handles_jpeg_input() {
    let result = resize_image(
        image_bytes(TINY_JPEG),
        "image/jpeg".to_string(),
        Some(ImageResizeOptions {
            max_width: Some(100),
            max_height: Some(100),
            max_bytes: Some(1024 * 1024),
            jpeg_quality: None,
        }),
    )
    .await;

    let resized = result.expect("resize");
    assert!(!resized.was_resized);
    assert_eq!(resized.original_width, 2);
    assert_eq!(resized.original_height, 2);
}

#[tokio::test]
async fn resize_returns_null_for_undecodable_bytes() {
    let result = resize_image(b"not an image".to_vec(), "image/png".to_string(), None).await;
    assert_eq!(result, None);
}

#[test]
fn resize_in_process_matches_the_awaited_entry() {
    // The spawn_blocking handoff must not change the pipeline's outcomes.
    let in_process = resize_image_in_process(
        &image_bytes(MEDIUM_PNG_100X100),
        "image/png",
        Some(&ImageResizeOptions {
            max_width: Some(50),
            max_height: Some(50),
            max_bytes: Some(1024 * 1024),
            jpeg_quality: None,
        }),
    );
    let resized = in_process.expect("resize");
    assert!(resized.was_resized);
    assert!(resized.width <= 50);
}

#[test]
fn the_shrink_floors_a_one_pixel_width() {
    // A 2x3 image under an impossible byte cap shrinks to 1x2 first: the
    // already-one width stays one while the height halves, upstream's
    // `Math.max(1, ...)` clamp, before the 1x1 gate ends the walk.
    let result = resize_image_in_process(
        &image_bytes(TWO_BY_THREE_PNG),
        "image/png",
        Some(&ImageResizeOptions {
            max_width: Some(2000),
            max_height: Some(2000),
            max_bytes: Some(1),
            jpeg_quality: None,
        }),
    );
    assert_eq!(result, None);
}

// === image-processing.test.ts: formatDimensionNote ==========================

#[test]
fn dimension_note_returns_null_for_non_resized_images() {
    let note = format_dimension_note(&pi_coding_agent::utils::image_resize_core::ResizedImage {
        data: String::new(),
        mime_type: "image/png".to_string(),
        original_width: 100,
        original_height: 100,
        width: 100,
        height: 100,
        was_resized: false,
    });
    assert_eq!(note, None);
}

#[test]
fn dimension_note_formats_resized_images() {
    let note = format_dimension_note(&pi_coding_agent::utils::image_resize_core::ResizedImage {
        data: String::new(),
        mime_type: "image/png".to_string(),
        original_width: 2000,
        original_height: 1000,
        width: 1000,
        height: 500,
        was_resized: true,
    })
    .expect("note");
    assert!(note.contains("original 2000x1000"));
    assert!(note.contains("displayed at 1000x500"));
    assert!(note.contains("2.00")); // scale factor
}

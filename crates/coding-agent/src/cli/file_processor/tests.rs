//! Boundary cases for `@file` processing, upstream's
//! `src/cli/file-processor.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. The file has no upstream
//! test; these pin the arms its callers rely on.

#![expect(
    clippy::expect_used,
    reason = "the temp-dir and fixture setup panics the test when the filesystem cannot provide the inputs"
)]

use super::*;

use base64::Engine as _;

/// A valid 1x1 RGBA PNG, the same fixture `utils_tool_result_images.rs` uses.
const TINY_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

fn cwd(dir: &tempfile::TempDir) -> String {
    dir.path().to_str().expect("utf-8 tempdir path").to_string()
}

#[tokio::test]
async fn renders_text_files_with_the_bom_stripped() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("note.md");
    std::fs::write(&path, "\u{FEFF}hello\nworld").expect("write note");
    let cwd = cwd(&dir);
    let absolute = resolve_read_path("note.md", &cwd).expect("resolves");

    let result = process_file_arguments(&["note.md".to_string()], &cwd, None)
        .await
        .expect("processes");

    assert_eq!(
        result.text,
        format!("<file name=\"{absolute}\">\nhello\nworld\n</file>\n")
    );
    assert!(result.images.is_empty());
}

#[tokio::test]
async fn skips_empty_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("empty.txt"), "").expect("write empty");

    let result = process_file_arguments(&["empty.txt".to_string()], &cwd(&dir), None)
        .await
        .expect("processes");

    assert_eq!(result.text, "");
    assert!(result.images.is_empty());
}

#[tokio::test]
async fn reports_missing_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    let error = process_file_arguments(&["missing.txt".to_string()], &cwd(&dir), None)
        .await
        .expect_err("missing file fails");
    assert!(matches!(error, ProcessFileError::NotFound(_)));
    assert!(error.to_string().starts_with("File not found: "));
}

#[tokio::test]
async fn attaches_images_and_references_them_in_the_text() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("pixel.png");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(TINY_PNG_BASE64)
        .expect("fixture decodes");
    std::fs::write(&path, bytes).expect("write png");
    let cwd = cwd(&dir);
    let absolute = resolve_read_path("pixel.png", &cwd).expect("resolves");
    let options = ProcessFileOptions {
        auto_resize_images: Some(false),
    };

    let result = process_file_arguments(&["pixel.png".to_string()], &cwd, Some(&options))
        .await
        .expect("processes");

    assert_eq!(result.text, format!("<file name=\"{absolute}\"></file>\n"));
    assert_eq!(result.images.len(), 1);
    assert_eq!(result.images[0].mime_type, "image/png");
    assert!(!result.images[0].data.is_empty());
}

#[tokio::test]
async fn reports_an_unprocessable_image_as_a_file_block() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("broken.gif");
    std::fs::write(&path, b"GIF89a").expect("write header");
    let cwd = cwd(&dir);
    let absolute = resolve_read_path("broken.gif", &cwd).expect("resolves");

    let result = process_file_arguments(&["broken.gif".to_string()], &cwd, None)
        .await
        .expect("processes");

    assert!(result.images.is_empty());
    assert_eq!(
        result.text,
        format!(
            "<file name=\"{absolute}\">[Image omitted: could not be resized below the inline image size limit.]</file>\n"
        )
    );
}

#[test]
fn formats_read_errors() {
    let error = ProcessFileError::Read {
        path: PathBuf::from("/tmp/x"),
        message: "boom".to_string(),
    };
    assert_eq!(error.to_string(), "Could not read file /tmp/x: boom");
}

#[tokio::test]
async fn reports_path_normalization_failures_as_read_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let error = process_file_arguments(&["file://example.com/x".to_string()], &cwd(&dir), None)
        .await
        .expect_err("non-local file URL fails");
    assert!(matches!(error, ProcessFileError::Read { .. }));
    assert!(error.to_string().starts_with("Could not read file "));
}

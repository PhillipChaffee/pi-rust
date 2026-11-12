//! The external-editor hand-off at pin `60e7e76`: the three upstream
//! `external-editor.test.ts` cases plus BOM/newline and spawn-failure
//! boundaries.

#![expect(
    clippy::expect_used,
    reason = "a missing capture file or temp path fails the assertion by design"
)]

use std::fs;
use std::path::{Path, PathBuf};

use pi_coding_agent::modes::interactive::external_editor::{
    ExternalEditorOptions, ExternalEditorResult, edit_in_external_editor,
};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/fake_external_editor.sh"
);

struct EditorCapture {
    file_path: PathBuf,
    content: String,
    entries: Vec<String>,
    directory_mode: Option<u32>,
}

#[expect(
    clippy::panic,
    reason = "the capture fixture must exist for the assertions to mean anything"
)]
fn capture_text(prefix: &Path, suffix: &str) -> String {
    let path = prefix.with_extension(suffix);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("capture {}: {error}", path.display()))
}

fn read_capture(prefix: &Path) -> EditorCapture {
    EditorCapture {
        file_path: PathBuf::from(capture_text(prefix, "path")),
        content: capture_text(prefix, "content"),
        entries: capture_text(prefix, "entries")
            .lines()
            .map(str::to_owned)
            .collect(),
        directory_mode: fs::read_to_string(prefix.with_extension("mode"))
            .ok()
            .and_then(|mode| u32::from_str_radix(mode.trim(), 8).ok()),
    }
}

async fn run_editor(mode: &str) -> (ExternalEditorResult, EditorCapture) {
    let temp = tempfile::tempdir().expect("temp dir");
    let capture_prefix = temp.path().join("capture");
    let command = format!("{FIXTURE} {} {mode}", capture_prefix.display());
    let result = edit_in_external_editor(ExternalEditorOptions {
        command,
        content: "original".to_string(),
    })
    .await
    .expect("editor hand-off");
    let capture = read_capture(&capture_prefix);
    (result, capture)
}

#[tokio::test]
async fn edits_a_prompt_inside_a_private_temporary_directory() {
    let (result, capture) = run_editor("--ok").await;
    let directory = capture.file_path.parent().expect("parent directory");

    assert_eq!(
        result,
        ExternalEditorResult::Complete {
            content: "edited".to_string()
        }
    );
    assert_eq!(directory.parent(), Some(std::env::temp_dir().as_path()));
    assert!(
        directory
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("pi-editor-"))
    );
    assert_eq!(
        capture.file_path.file_name().and_then(|name| name.to_str()),
        Some("prompt.md")
    );
    assert_eq!(capture.entries, ["prompt.md"]);
    assert_eq!(capture.content, "original");
    if cfg!(unix) {
        let mode = capture.directory_mode.expect("directory mode captured");
        assert_eq!(mode & 0o077, 0, "the directory is private");
    }
    assert!(!directory.exists(), "the temp directory is removed");
}

#[tokio::test]
async fn keeps_the_original_content_when_the_editor_exits_unsuccessfully() {
    let (result, capture) = run_editor("--fail").await;

    assert_eq!(result, ExternalEditorResult::Failed);
    assert!(
        !capture
            .file_path
            .parent()
            .expect("parent directory")
            .exists()
    );
}

#[tokio::test]
async fn returns_empty_content_when_the_editor_clears_the_prompt() {
    let (result, _capture) = run_editor("--empty").await;

    assert_eq!(
        result,
        ExternalEditorResult::Complete {
            content: String::new()
        }
    );
}

#[tokio::test]
async fn strips_a_bom_and_one_trailing_newline() {
    let (result, _capture) = run_editor("--bom").await;

    assert_eq!(
        result,
        ExternalEditorResult::Complete {
            content: "edited\n".to_string()
        },
        "the BOM and only the final newline are removed"
    );
}

#[tokio::test]
async fn reports_failed_when_the_editor_cannot_spawn() {
    let result = edit_in_external_editor(ExternalEditorOptions {
        command: "pi-editor-command-that-does-not-exist".to_string(),
        content: "original".to_string(),
    })
    .await
    .expect("the temp-file write still succeeds");

    assert_eq!(result, ExternalEditorResult::Failed);
}

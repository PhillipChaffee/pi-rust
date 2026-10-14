//! The read/write/edit/ls blocks of upstream's `tools.test.ts`, the
//! `edit tool fuzzy matching` and `edit tool CRLF handling` describes, the
//! `edit-tool-legacy-input` suite, and the built-in-tool describes of
//! `file-mutation-queue.test.ts`.

#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use pi_agent_core::harness::context::{background_context, with_cancel};
use pi_agent_core::types::AgentToolContent as Content;
use pi_agent_core::types::{AgentToolError, AgentToolResult};
use serde_json::json;

use crate::extensions::types::{CwdContext, ExtensionContext, ToolDefinition};
use crate::tools::bash::{BashExecOptions, BashExecOutcome, BashOperations, BashToolOptions};
use crate::tools::edit::create_edit_tool;
use crate::tools::edit::create_edit_tool_definition;
use crate::tools::find::{FindOperations, FindToolOptions};
use crate::tools::grep::{GrepOperations, GrepToolOptions};
use crate::tools::index::ToolsOptions;
use crate::tools::ls::create_ls_tool_definition;
use crate::tools::ls::{LsOperations, LsToolOptions};
use crate::tools::powershell::PowerShellToolOptions;
use crate::tools::read::create_read_tool_definition;
use crate::tools::read::{ReadOperations, ReadToolOptions};
use crate::tools::write::create_write_tool_definition;
use crate::tools::{
    edit::EditOperations, edit::EditToolOptions, write::WriteOperations, write::WriteToolOptions,
};

use super::helpers::{apply_unified_patch, block_on, text_output};

/// Runs a definition's execute directly, upstream's
/// `definition.execute(...)` calls.
pub(super) async fn run_definition(
    definition: &ToolDefinition,
    args: serde_json::Value,
    signal: Option<&pi_agent_core::harness::context::AbortSignal>,
    ctx: Option<&dyn ExtensionContext>,
) -> Result<AgentToolResult, AgentToolError> {
    (definition.execute)("test-call", &args, signal, None, ctx).await
}

/// The plain read invocation, the suite's default `run_definition` call
/// shape against the temp file.
async fn run_read(
    tool: &ToolDefinition,
    path: &std::path::Path,
) -> Result<AgentToolResult, AgentToolError> {
    run_definition(tool, json!({ "path": path.to_string_lossy() }), None, None).await
}

/// The first text block, the single-text-block reader the suites share.
pub(super) fn first_text(result: &AgentToolResult) -> String {
    match result.content.first() {
        Some(Content::Text(text)) => text.text.clone(),
        Some(Content::Image(_)) | None => String::new(),
    }
}

/// The image content block, the reader the read-tool image cases share.
pub(super) fn image_content(result: &AgentToolResult) -> Option<(&str, &str)> {
    result.content.iter().find_map(|part| match part {
        Content::Image(image) => Some((image.mime_type.as_str(), image.data.as_str())),
        Content::Text(_) => None,
    })
}

/// The 1x1 red 24bpp BMP fixture, upstream's `createTinyBmp1x1Red24bpp`.
pub(super) fn tiny_bmp_1x1_red_24bpp() -> Vec<u8> {
    let mut buffer = vec![0u8; 58];
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the BMP fixture is 58 bytes, far inside a u32 header field"
    )]
    let total_length = buffer.len() as u32;
    buffer[0] = b'B';
    buffer[1] = b'M';
    buffer[2..6].copy_from_slice(&total_length.to_le_bytes());
    buffer[10..14].copy_from_slice(&54u32.to_le_bytes());
    buffer[14..18].copy_from_slice(&40u32.to_le_bytes());
    buffer[18..22].copy_from_slice(&1i32.to_le_bytes());
    buffer[22..26].copy_from_slice(&1i32.to_le_bytes());
    buffer[26..28].copy_from_slice(&1u16.to_le_bytes());
    buffer[28..30].copy_from_slice(&24u16.to_le_bytes());
    buffer[56] = 0xff;
    buffer
}

const PNG_1X1_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg==";

// ---------------------------------------------------------------------------
// read tool
// ---------------------------------------------------------------------------

/// The disk-backed edit operations with the settle delays the concurrency
/// probes rely on, the shared shape of the two queued-write tests.
fn slow_disk_edit_operations() -> EditOperations {
    EditOperations {
        access: Arc::new(|path| {
            Box::pin(async move {
                tokio::fs::File::open(&path)
                    .await
                    .map(|_| ())
                    .map_err(AgentToolError::from)
            })
        }),
        read_file: Arc::new(|path| {
            Box::pin(async move {
                let buffer = tokio::fs::read(&path).await.map_err(AgentToolError::from)?;
                tokio::time::sleep(Duration::from_millis(30)).await;
                Ok(buffer)
            })
        }),
        write_file: Arc::new(|path, content| {
            Box::pin(async move {
                tokio::time::sleep(Duration::from_millis(30)).await;
                tokio::fs::write(&path, content.as_bytes())
                    .await
                    .map_err(AgentToolError::from)
            })
        }),
    }
}

#[test]
pub(super) fn read_reads_file_contents_that_fit_within_limits() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("test.txt");
        let content = "Hello, world!\nLine 2\nLine 3";
        std::fs::write(&test_file, content).unwrap();
        let tool = create_read_tool_definition("/", None);

        let result = run_read(&tool, &test_file).await.unwrap();
        let output = text_output(&result);
        assert_eq!(output, content);
        // No truncation message since file fits within limits
        assert!(!output.contains("Use offset="));
        assert_eq!(result.details, serde_json::Value::Null);
    });
}

#[test]
pub(super) fn read_handles_nonexistent_files() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("nonexistent.txt");
        let tool = create_read_tool_definition("/", None);

        let error = run_read(&tool, &test_file).await.unwrap_err();
        assert!(
            error.to_string().to_lowercase().contains("no such file")
                || error.to_string().contains("ENOENT"),
            "{}",
            error
        );
    });
}

#[test]
pub(super) fn read_truncates_files_exceeding_line_limit() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("large.txt");
        let lines: Vec<String> = (1..=2500).map(|i| format!("Line {i}")).collect();
        std::fs::write(&test_file, lines.join("\n")).unwrap();
        let tool = create_read_tool_definition("/", None);

        let result = run_read(&tool, &test_file).await.unwrap();
        let output = text_output(&result);
        assert!(output.contains("Line 1"));
        assert!(output.contains("Line 2000"));
        assert!(!output.contains("Line 2001"));
        assert!(output.contains("[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]"));
    });
}

#[test]
pub(super) fn read_truncates_when_byte_limit_exceeded() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("large-bytes.txt");
        let lines: Vec<String> = (1..=500)
            .map(|i| format!("Line {i}: {}", "x".repeat(200)))
            .collect();
        std::fs::write(&test_file, lines.join("\n")).unwrap();
        let tool = create_read_tool_definition("/", None);

        let result = run_read(&tool, &test_file).await.unwrap();
        let output = text_output(&result);
        assert!(output.contains("Line 1:"));
        // Should show byte limit message
        assert!(output.contains(" limit). Use offset="), "{output}");
    });
}

#[test]
pub(super) fn read_handles_offset_parameter() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("offset-test.txt");
        let lines: Vec<String> = (1..=100).map(|i| format!("Line {i}")).collect();
        std::fs::write(&test_file, lines.join("\n")).unwrap();
        let tool = create_read_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({ "path": test_file.to_string_lossy(), "offset": 51 }),
            None,
            None,
        )
        .await
        .unwrap();
        let output = text_output(&result);
        assert!(!output.contains("Line 50"));
        assert!(output.contains("Line 51"));
        assert!(output.contains("Line 100"));
        // No truncation message since file fits within limits
        assert!(!output.contains("Use offset="));
    });
}

#[test]
pub(super) fn read_handles_limit_parameter() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("limit-test.txt");
        let lines: Vec<String> = (1..=100).map(|i| format!("Line {i}")).collect();
        std::fs::write(&test_file, lines.join("\n")).unwrap();
        let tool = create_read_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({ "path": test_file.to_string_lossy(), "limit": 10 }),
            None,
            None,
        )
        .await
        .unwrap();
        let output = text_output(&result);
        assert!(output.contains("Line 1"));
        assert!(output.contains("Line 10"));
        assert!(!output.contains("Line 11"));
        assert!(output.contains("[90 more lines in file. Use offset=11 to continue.]"));
    });
}

#[test]
pub(super) fn read_handles_offset_and_limit_together() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("offset-limit-test.txt");
        let lines: Vec<String> = (1..=100).map(|i| format!("Line {i}")).collect();
        std::fs::write(&test_file, lines.join("\n")).unwrap();
        let tool = create_read_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({ "path": test_file.to_string_lossy(), "offset": 41, "limit": 20 }),
            None,
            None,
        )
        .await
        .unwrap();
        let output = text_output(&result);
        assert!(!output.contains("Line 40"));
        assert!(output.contains("Line 41"));
        assert!(output.contains("Line 60"));
        assert!(!output.contains("Line 61"));
        assert!(output.contains("[40 more lines in file. Use offset=61 to continue.]"));
    });
}

#[test]
pub(super) fn read_shows_error_when_offset_is_beyond_file_length() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("short.txt");
        std::fs::write(&test_file, "Line 1\nLine 2\nLine 3").unwrap();
        let tool = create_read_tool_definition("/", None);

        let error = run_definition(
            &tool,
            json!({ "path": test_file.to_string_lossy(), "offset": 100 }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Offset 100 is beyond end of file (3 lines total)"),
            "{}",
            error
        );
    });
}

#[test]
pub(super) fn read_includes_truncation_details_when_truncated() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("large-file.txt");
        let lines: Vec<String> = (1..=2500).map(|i| format!("Line {i}")).collect();
        std::fs::write(&test_file, lines.join("\n")).unwrap();
        let tool = create_read_tool_definition("/", None);

        let result = run_read(&tool, &test_file).await.unwrap();
        let details = &result.details;
        assert!(details.is_object());
        let truncation = &details["truncation"];
        assert_eq!(truncation["truncated"], true);
        assert_eq!(truncation["truncatedBy"], "lines");
        assert_eq!(truncation["totalLines"], 2500);
        assert_eq!(truncation["outputLines"], 2000);
    });
}

#[test]
pub(super) fn read_detects_image_mime_type_from_file_magic_not_extension() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("image.txt");
        std::fs::write(
            &test_file,
            base64::engine::general_purpose::STANDARD
                .decode(PNG_1X1_BASE64)
                .unwrap(),
        )
        .unwrap();
        let tool = create_read_tool_definition("/", None);

        let result = run_read(&tool, &test_file).await.unwrap();
        assert!(matches!(result.content.first(), Some(Content::Text(_))));
        assert!(text_output(&result).contains("Read image file [image/png]"));

        let image = image_content(&result).expect("image block");
        assert_eq!(image.0, "image/png");
        assert!(!image.1.is_empty());
    });
}

#[test]
pub(super) fn read_reads_bmp_files_from_disk_as_png_image_attachments() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("image.bmp");
        std::fs::write(&test_file, tiny_bmp_1x1_red_24bpp()).unwrap();
        let tool = create_read_tool_definition("/", None);

        let result = run_read(&tool, &test_file).await.unwrap();
        assert!(matches!(result.content.first(), Some(Content::Text(_))));
        assert!(text_output(&result).contains("Read image file [image/png]"));
        assert!(text_output(&result).contains("[Image converted from image/bmp to image/png.]"));
        let image = image_content(&result).expect("image block");
        assert_eq!(image.0, "image/png");
        // The PNG magic bytes.
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(image.1)
            .unwrap();
        assert_eq!(decoded[0], 0x89);
    });
}

#[test]
pub(super) fn read_treats_files_with_image_extension_but_non_image_content_as_text() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("not-an-image.png");
        std::fs::write(&test_file, "definitely not a png").unwrap();
        let tool = create_read_tool_definition("/", None);

        let result = run_read(&tool, &test_file).await.unwrap();
        let output = text_output(&result);
        assert!(output.contains("definitely not a png"));
        assert!(image_content(&result).is_none());
    });
}

// ---------------------------------------------------------------------------
// write tool
// ---------------------------------------------------------------------------

#[test]
pub(super) fn write_writes_file_contents() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("write-test.txt");
        let tool = create_write_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({ "path": test_file.to_string_lossy(), "content": "Test content" }),
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            first_text(&result),
            format!("Successfully wrote to {}", test_file.to_string_lossy())
        );
        assert_eq!(result.details, serde_json::Value::Null);
    });
}

#[test]
pub(super) fn write_creates_parent_directories() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("nested/dir/test.txt");
        let tool = create_write_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({ "path": test_file.to_string_lossy(), "content": "Nested content" }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(first_text(&result).contains("Successfully wrote"));
    });
}

// ---------------------------------------------------------------------------
// edit tool
// ---------------------------------------------------------------------------

#[test]
pub(super) fn edit_replaces_text_in_file() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("edit-test.txt");
        let original_content = "Hello, world!";
        std::fs::write(&test_file, original_content).unwrap();
        let tool = create_edit_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({
                "path": test_file.to_string_lossy(),
                "edits": [{ "oldText": "world", "newText": "testing" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        let output = text_output(&result);
        assert!(output.contains("Successfully replaced"));
        let details = &result.details;
        assert!(details["diff"].is_string());
        assert!(details["diff"].as_str().unwrap().contains("testing"));
        let patch = details["patch"].as_str().unwrap();
        assert!(patch.contains("--- "));
        assert!(patch.contains("+++ "));
        assert!(patch.contains("@@"));
        assert!(patch.contains("-Hello, world!"));
        assert!(patch.contains("+Hello, testing!"));
        // The patch applies back, upstream's `applyPatch` round trip.
        let applied = apply_unified_patch(original_content, patch).expect("patch applies");
        assert_eq!(applied, "Hello, testing!");
    });
}

#[test]
pub(super) fn edit_fails_if_text_not_found() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("edit-test.txt");
        std::fs::write(&test_file, "Hello, world!").unwrap();
        let tool = create_edit_tool_definition("/", None);

        let error = run_definition(
            &tool,
            json!({
                "path": test_file.to_string_lossy(),
                "edits": [{ "oldText": "nonexistent", "newText": "testing" }]
            }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("Could not find the exact text"),
            "{}",
            error
        );
    });
}

#[test]
pub(super) fn edit_includes_enoent_when_the_edit_target_does_not_exist() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let missing_file = dir.path().join("missing.txt");
        let missing = missing_file.to_string_lossy().into_owned();
        let tool = create_edit_tool_definition("/", None);

        let error = run_definition(
            &tool,
            json!({
                "path": missing,
                "edits": [{ "oldText": "hello", "newText": "world" }]
            }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("Could not edit file: {missing}. Error code: ENOENT.")
        );
    });
}

#[test]
pub(super) fn edit_fails_if_text_appears_multiple_times() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("edit-test.txt");
        std::fs::write(&test_file, "foo foo foo").unwrap();
        let tool = create_edit_tool_definition("/", None);

        let error = run_definition(
            &tool,
            json!({
                "path": test_file.to_string_lossy(),
                "edits": [{ "oldText": "foo", "newText": "bar" }]
            }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("Found 3 occurrences"),
            "{}",
            error
        );
    });
}

#[test]
pub(super) fn edit_replaces_multiple_disjoint_regions_in_one_call() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("edit-multi.txt");
        std::fs::write(&test_file, "alpha\nbeta\ngamma\ndelta\n").unwrap();
        let tool = create_edit_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({
                "path": test_file.to_string_lossy(),
                "edits": [
                    { "oldText": "alpha\n", "newText": "ALPHA\n" },
                    { "oldText": "gamma\n", "newText": "GAMMA\n" }
                ]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_output(&result).contains("Successfully replaced 2 block(s)"));
        assert_eq!(
            std::fs::read_to_string(&test_file).unwrap(),
            "ALPHA\nbeta\nGAMMA\ndelta\n"
        );
        let diff = result.details["diff"].as_str().unwrap();
        assert!(diff.contains("ALPHA"));
        assert!(diff.contains("GAMMA"));
    });
}

#[test]
pub(super) fn edit_collapses_large_unchanged_gaps_in_multi_edit_diffs() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("edit-multi-large-gap.txt");
        let lines: Vec<String> = (1..=600).map(|i| format!("line {i:0>3}")).collect();
        std::fs::write(&test_file, format!("{}\n", lines.join("\n"))).unwrap();
        let tool = create_edit_tool_definition("/", None);

        let result = run_definition(
            &tool,
            json!({
                "path": test_file.to_string_lossy(),
                "edits": [
                    { "oldText": "line 100\n", "newText": "LINE 100\n" },
                    { "oldText": "line 300\n", "newText": "LINE 300\n" },
                    { "oldText": "line 500\n", "newText": "LINE 500\n" }
                ]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        let diff = result.details["diff"].as_str().unwrap_or_default();
        assert!(diff.contains("LINE 100"));
        assert!(diff.contains("LINE 300"));
        assert!(diff.contains("LINE 500"));
        assert!(diff.contains("..."));
        assert!(!diff.contains("line 250"));
        assert!(diff.lines().count() < 50, "{} lines", diff.lines().count());
    });
}

#[test]
pub(super) fn edit_matches_edits_against_the_original_file_not_incrementally() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("edit-multi-original.txt");
        std::fs::write(&test_file, "foo\nbar\nbaz\n").unwrap();
        let tool = create_edit_tool_definition("/", None);

        run_definition(
            &tool,
            json!({
                "path": test_file.to_string_lossy(),
                "edits": [
                    { "oldText": "foo\n", "newText": "foo bar\n" },
                    { "oldText": "bar\n", "newText": "BAR\n" }
                ]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&test_file).unwrap(),
            "foo bar\nBAR\nbaz\n"
        );
    });
}

#[test]
pub(super) fn edit_fails_when_edits_is_empty() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("edit-empty-edits.txt");
        std::fs::write(&test_file, "hello\nworld\n").unwrap();
        let tool = create_edit_tool_definition("/", None);

        let error = run_definition(
            &tool,
            json!({ "path": test_file.to_string_lossy(), "edits": [] }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("edits must contain at least one replacement"),
            "{}",
            error
        );
    });
}

#[test]
pub(super) fn edit_fails_when_multi_edit_regions_overlap() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("edit-overlap.txt");
        std::fs::write(&test_file, "one\ntwo\nthree\n").unwrap();
        let tool = create_edit_tool_definition("/", None);

        let error = run_definition(
            &tool,
            json!({
                "path": test_file.to_string_lossy(),
                "edits": [
                    { "oldText": "one\ntwo\n", "newText": "ONE\nTWO\n" },
                    { "oldText": "two\nthree\n", "newText": "TWO\nTHREE\n" }
                ]
            }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("overlap"), "{}", error);
    });
}

#[test]
pub(super) fn edit_does_not_partially_apply_edits_when_one_edit_fails() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("edit-no-partial.txt");
        let original_content = "alpha\nbeta\ngamma\n";
        std::fs::write(&test_file, original_content).unwrap();
        let tool = create_edit_tool_definition("/", None);

        let error = run_definition(
            &tool,
            json!({
                "path": test_file.to_string_lossy(),
                "edits": [
                    { "oldText": "alpha\n", "newText": "ALPHA\n" },
                    { "oldText": "missing\n", "newText": "MISSING\n" }
                ]
            }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("Could not find"), "{}", error);
        assert_eq!(
            std::fs::read_to_string(&test_file).unwrap(),
            original_content
        );
    });
}

#[test]
#[cfg(unix)]
pub(super) fn edit_includes_eacces_for_read_only_files() {
    use std::os::unix::fs::PermissionsExt;

    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("edit-readonly.txt");
        std::fs::write(&test_file, "hello\n").unwrap();
        std::fs::set_permissions(&test_file, std::fs::Permissions::from_mode(0o444)).unwrap();
        let target = test_file.to_string_lossy().into_owned();
        let tool = create_edit_tool_definition("/", None);

        let error = run_definition(
            &tool,
            json!({
                "path": target,
                "edits": [{ "oldText": "hello", "newText": "world" }]
            }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("Error code: EACCES"),
            "{}",
            error
        );
    });
}

#[test]
pub(super) fn edit_includes_the_original_error_message_for_unknown_edit_access_errors() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let operations = EditOperations {
            access: Arc::new(|_path| {
                Box::pin(async { Err(AgentToolError::from(std::io::Error::other("disk offline"))) })
            }),
            read_file: Arc::new(|_path| Box::pin(async { Ok(b"hello\n".to_vec()) })),
            write_file: Arc::new(|_path, _content| Box::pin(async { Ok(()) })),
        };
        let tool = create_edit_tool_definition(
            dir.path().to_string_lossy().as_ref(),
            Some(EditToolOptions {
                operations: Some(operations),
            }),
        );

        let error = run_definition(
            &tool,
            json!({
                "path": "broken.txt",
                "edits": [{ "oldText": "hello", "newText": "world" }]
            }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Could not edit file: broken.txt. Error: disk offline."
        );
    });
}

// ---------------------------------------------------------------------------
// tool cwd resolution
// ---------------------------------------------------------------------------

#[test]
pub(super) fn read_uses_ctx_cwd_when_provided() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("ctx-cwd-read.txt");
        std::fs::write(&test_file, "hello from ctx.cwd").unwrap();
        let tool = create_read_tool_definition("/", None);
        let context = CwdContext {
            cwd: dir.path().to_string_lossy().into_owned(),
        };
        let result = run_definition(
            &tool,
            json!({ "path": "ctx-cwd-read.txt" }),
            None,
            Some(&context),
        )
        .await
        .unwrap();
        assert!(text_output(&result).contains("hello from ctx.cwd"));
    });
}

#[test]
pub(super) fn write_uses_ctx_cwd_when_provided() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let tool = create_write_tool_definition("/", None);
        let context = CwdContext {
            cwd: dir.path().to_string_lossy().into_owned(),
        };
        run_definition(
            &tool,
            json!({ "path": "ctx-cwd-write.txt", "content": "written via ctx.cwd" }),
            None,
            Some(&context),
        )
        .await
        .unwrap();
        let readback = std::fs::read_to_string(dir.path().join("ctx-cwd-write.txt")).unwrap();
        assert_eq!(readback, "written via ctx.cwd");
    });
}

#[test]
pub(super) fn edit_uses_ctx_cwd_when_provided() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("ctx-cwd-edit.txt");
        std::fs::write(&test_file, "old text").unwrap();
        let tool = create_edit_tool_definition("/", None);
        let context = CwdContext {
            cwd: dir.path().to_string_lossy().into_owned(),
        };
        run_definition(
            &tool,
            json!({
                "path": "ctx-cwd-edit.txt",
                "edits": [{ "oldText": "old text", "newText": "new text" }]
            }),
            None,
            Some(&context),
        )
        .await
        .unwrap();
        let readback = std::fs::read_to_string(&test_file).unwrap();
        assert_eq!(readback, "new text");
    });
}

#[test]
pub(super) fn ls_uses_ctx_cwd_when_provided() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ctx-cwd-ls.txt"), "list me").unwrap();
        let tool = create_ls_tool_definition("/", None);
        let context = CwdContext {
            cwd: dir.path().to_string_lossy().into_owned(),
        };
        let result = run_definition(&tool, json!({}), None, Some(&context))
            .await
            .unwrap();
        assert!(text_output(&result).contains("ctx-cwd-ls.txt"));
    });
}

// ---------------------------------------------------------------------------
// edit tool fuzzy matching
// ---------------------------------------------------------------------------

#[test]
pub(super) fn fuzzy_matches_text_with_trailing_whitespace_stripped() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("trailing-ws.txt");
        std::fs::write(&test_file, "line one   \nline two  \nline three\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let result = run_definition(
            &tool,
            json!({
                "path": "trailing-ws.txt",
                "edits": [{ "oldText": "line one\nline two\n", "newText": "replaced\n" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_output(&result).contains("Successfully replaced"));
        let content = std::fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "replaced\nline three\n");
    });
}

#[test]
pub(super) fn fuzzy_matches_fullwidth_punctuation_in_chinese_text() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("chinese-punctuation.txt");
        std::fs::write(&test_file, "你好，世界\n你好（世界）\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let result = run_definition(
            &tool,
            json!({
                "path": "chinese-punctuation.txt",
                "edits": [{ "oldText": "你好,世界\n你好(世界)\n", "newText": "你好，pi\n你好(pi)\n" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_output(&result).contains("Successfully replaced"));
        let content = std::fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "你好，pi\n你好(pi)\n");
    });
}

#[test]
pub(super) fn fuzzy_matches_compatibility_equivalent_unicode_forms() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("unicode-compatibility.txt");
        std::fs::write(&test_file, "ＡＢＣ１２３\ncafe\u{301}\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let result = run_definition(
            &tool,
            json!({
                "path": "unicode-compatibility.txt",
                "edits": [{ "oldText": "ABC123\ncaf\u{e9}\n", "newText": "XYZ789\ncoffee\n" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_output(&result).contains("Successfully replaced"));
        let content = std::fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "XYZ789\ncoffee\n");
    });
}

#[test]
pub(super) fn fuzzy_matches_smart_single_quotes_to_ascii_quotes() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("smart-quotes.txt");
        std::fs::write(&test_file, "console.log(\u{2018}hello\u{2019});\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let result = run_definition(
            &tool,
            json!({
                "path": "smart-quotes.txt",
                "edits": [{ "oldText": "console.log('hello');", "newText": "console.log('world');" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_output(&result).contains("Successfully replaced"));
        let content = std::fs::read_to_string(&test_file).unwrap();
        assert!(content.contains("world"));
    });
}

#[test]
pub(super) fn fuzzy_matches_smart_double_quotes_to_ascii_quotes() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("smart-double-quotes.txt");
        std::fs::write(&test_file, "const msg = \u{201C}Hello World\u{201D};\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let result = run_definition(
            &tool,
            json!({
                "path": "smart-double-quotes.txt",
                "edits": [{ "oldText": "const msg = \"Hello World\";", "newText": "const msg = \"Goodbye\";" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_output(&result).contains("Successfully replaced"));
        let content = std::fs::read_to_string(&test_file).unwrap();
        assert!(content.contains("Goodbye"));
    });
}

#[test]
pub(super) fn fuzzy_matches_unicode_dashes_to_ascii_hyphen() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("unicode-dashes.txt");
        std::fs::write(&test_file, "range: 1\u{2013}5\nbreak\u{2014}here\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let result = run_definition(
            &tool,
            json!({
                "path": "unicode-dashes.txt",
                "edits": [{ "oldText": "range: 1-5\nbreak-here", "newText": "range: 10-50\nbreak--here" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_output(&result).contains("Successfully replaced"));
        let content = std::fs::read_to_string(&test_file).unwrap();
        assert!(content.contains("10-50"));
    });
}

#[test]
pub(super) fn fuzzy_matches_non_breaking_space_to_regular_space() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("nbsp.txt");
        std::fs::write(&test_file, "hello\u{00A0}world\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let result = run_definition(
            &tool,
            json!({
                "path": "nbsp.txt",
                "edits": [{ "oldText": "hello world", "newText": "hello universe" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_output(&result).contains("Successfully replaced"));
        let content = std::fs::read_to_string(&test_file).unwrap();
        assert!(content.contains("universe"));
    });
}

#[test]
pub(super) fn fuzzy_prefers_exact_match_over_fuzzy_match() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("exact-preferred.txt");
        std::fs::write(&test_file, "const x = 'exact';\nconst y = 'other';\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let result = run_definition(
            &tool,
            json!({
                "path": "exact-preferred.txt",
                "edits": [{ "oldText": "const x = 'exact';", "newText": "const x = 'changed';" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_output(&result).contains("Successfully replaced"));
        let content = std::fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "const x = 'changed';\nconst y = 'other';\n");
    });
}

#[test]
pub(super) fn fuzzy_still_fails_when_text_is_not_found() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("no-match.txt");
        std::fs::write(&test_file, "completely different content\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let error = run_definition(
            &tool,
            json!({
                "path": "no-match.txt",
                "edits": [{ "oldText": "this does not exist", "newText": "replacement" }]
            }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("Could not find the exact text"),
            "{}",
            error
        );
    });
}

#[test]
pub(super) fn fuzzy_detects_duplicates_after_normalization() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("fuzzy-dups.txt");
        std::fs::write(&test_file, "hello world   \nhello world\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let error = run_definition(
            &tool,
            json!({
                "path": "fuzzy-dups.txt",
                "edits": [{ "oldText": "hello world", "newText": "replaced" }]
            }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("Found 2 occurrences"),
            "{}",
            error
        );
    });
}

#[test]
pub(super) fn fuzzy_supports_multi_edit_mode() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("fuzzy-multi.txt");
        std::fs::write(
            &test_file,
            "console.log(\u{2018}hello\u{2019});\nhello\u{00A0}world\n",
        )
        .unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        run_definition(
            &tool,
            json!({
                "path": "fuzzy-multi.txt",
                "edits": [
                    { "oldText": "console.log('hello');\n", "newText": "console.log('world');\n" },
                    { "oldText": "hello world\n", "newText": "hello universe\n" }
                ]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&test_file).unwrap(),
            "console.log('world');\nhello universe\n"
        );
    });
}

#[test]
pub(super) fn fuzzy_preserves_the_correct_occurrence_when_replacement_equals_a_nearby_line() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("fuzzy-preserve-duplicate-line.txt");
        let original_content = ["replace me   ", "after   ", ""].join("\n");
        std::fs::write(&test_file, &original_content).unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let result = run_definition(
            &tool,
            json!({
                "path": "fuzzy-preserve-duplicate-line.txt",
                "edits": [{ "oldText": "replace me\n", "newText": "after\n" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        let expected_content = ["after", "after   ", ""].join("\n");
        assert_eq!(
            std::fs::read_to_string(&test_file).unwrap(),
            expected_content
        );
        let patch = result.details["patch"].as_str().unwrap();
        let applied = apply_unified_patch(&original_content, patch).expect("patch applies");
        assert_eq!(applied, expected_content);
    });
}

#[test]
pub(super) fn fuzzy_preserves_untouched_lines_and_produces_an_applicable_patch() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("fuzzy-preserve-multi.txt");
        let original_content = [
            "keep before  ",
            "first target  ",
            "first after",
            "keep middle   ",
            "second target  ",
            "second after",
            "keep after  ",
            "",
        ]
        .join("\n");
        std::fs::write(&test_file, &original_content).unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let result = run_definition(
            &tool,
            json!({
                "path": "fuzzy-preserve-multi.txt",
                "edits": [
                    { "oldText": "first target\nfirst after", "newText": "FIRST\nFIRST2" },
                    { "oldText": "second target\nsecond after", "newText": "SECOND\nSECOND2" }
                ]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        let expected_content = [
            "keep before  ",
            "FIRST",
            "FIRST2",
            "keep middle   ",
            "SECOND",
            "SECOND2",
            "keep after  ",
            "",
        ]
        .join("\n");
        assert_eq!(
            std::fs::read_to_string(&test_file).unwrap(),
            expected_content
        );
        let patch = result.details["patch"].as_str().unwrap();
        let applied = apply_unified_patch(&original_content, patch).expect("patch applies");
        assert_eq!(applied, expected_content);
    });
}

// ---------------------------------------------------------------------------
// edit tool CRLF handling
// ---------------------------------------------------------------------------

#[test]
pub(super) fn crlf_matches_lf_oldtext_against_crlf_file_content() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("crlf-test.txt");
        std::fs::write(&test_file, "line one\r\nline two\r\nline three\r\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let result = run_definition(
            &tool,
            json!({
                "path": "crlf-test.txt",
                "edits": [{ "oldText": "line two\n", "newText": "replaced line\n" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(text_output(&result).contains("Successfully replaced"));
    });
}

#[test]
pub(super) fn crlf_preserves_crlf_line_endings_after_edit() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("crlf-preserve.txt");
        std::fs::write(&test_file, "first\r\nsecond\r\nthird\r\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        run_definition(
            &tool,
            json!({
                "path": "crlf-preserve.txt",
                "edits": [{ "oldText": "second\n", "newText": "REPLACED\n" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        let content = std::fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "first\r\nREPLACED\r\nthird\r\n");
    });
}

#[test]
pub(super) fn crlf_preserves_lf_line_endings_for_lf_files() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("lf-preserve.txt");
        std::fs::write(&test_file, "first\nsecond\nthird\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        run_definition(
            &tool,
            json!({
                "path": "lf-preserve.txt",
                "edits": [{ "oldText": "second\n", "newText": "REPLACED\n" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        let content = std::fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "first\nREPLACED\nthird\n");
    });
}

#[test]
pub(super) fn crlf_detects_duplicates_across_crlf_lf_variants() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("mixed-endings.txt");
        std::fs::write(&test_file, "hello\r\nworld\r\n---\r\nhello\nworld\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        let error = run_definition(
            &tool,
            json!({
                "path": "mixed-endings.txt",
                "edits": [{ "oldText": "hello\nworld\n", "newText": "replaced\n" }]
            }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("Found 2 occurrences"),
            "{}",
            error
        );
    });
}

#[test]
pub(super) fn crlf_preserves_utf8_bom_after_edit() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("bom-test.txt");
        std::fs::write(&test_file, "\u{FEFF}first\r\nsecond\r\nthird\r\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        run_definition(
            &tool,
            json!({
                "path": "bom-test.txt",
                "edits": [{ "oldText": "second\n", "newText": "REPLACED\n" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        let content = std::fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "\u{FEFF}first\r\nREPLACED\r\nthird\r\n");
    });
}

#[test]
pub(super) fn crlf_preserves_crlf_line_endings_and_bom_in_multi_edit_mode() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let test_file = dir.path().join("bom-crlf-multi.txt");
        std::fs::write(&test_file, "\u{FEFF}first\r\nsecond\r\nthird\r\nfourth\r\n").unwrap();
        let tool = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);

        run_definition(
            &tool,
            json!({
                "path": "bom-crlf-multi.txt",
                "edits": [
                    { "oldText": "second\n", "newText": "SECOND\n" },
                    { "oldText": "fourth\n", "newText": "FOURTH\n" }
                ]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        let content = std::fs::read_to_string(&test_file).unwrap();
        assert_eq!(content, "\u{FEFF}first\r\nSECOND\r\nthird\r\nFOURTH\r\n");
    });
}

// ---------------------------------------------------------------------------
// edit-tool-legacy-input
// ---------------------------------------------------------------------------

#[test]
pub(super) fn legacy_keeps_legacy_fields_out_of_the_public_schema() {
    let definition = create_edit_tool_definition(".", None);
    let properties = definition.parameters["properties"].as_object().unwrap();
    assert!(!properties.contains_key("oldText"));
    assert!(!properties.contains_key("newText"));
}

#[test]
pub(super) fn legacy_folds_top_level_old_text_new_text_into_edits() {
    let definition = create_edit_tool_definition(".", None);
    let prepared = (definition.prepare_arguments.as_ref().unwrap())(
        &json!({ "path": "file.txt", "oldText": "before", "newText": "after" }),
    )
    .unwrap();
    assert_eq!(
        prepared,
        json!({ "path": "file.txt", "edits": [{ "oldText": "before", "newText": "after" }] })
    );
}

#[test]
pub(super) fn legacy_appends_legacy_replacement_to_existing_edits() {
    let definition = create_edit_tool_definition(".", None);
    let prepared = (definition.prepare_arguments.as_ref().unwrap())(&json!({
        "path": "file.txt",
        "edits": [{ "oldText": "a", "newText": "b" }],
        "oldText": "c",
        "newText": "d"
    }))
    .unwrap();
    assert_eq!(
        prepared,
        json!({
            "path": "file.txt",
            "edits": [
                { "oldText": "a", "newText": "b" },
                { "oldText": "c", "newText": "d" }
            ]
        })
    );
}

#[test]
pub(super) fn legacy_passes_through_valid_input_unchanged() {
    let definition = create_edit_tool_definition(".", None);
    let input = json!({ "path": "file.txt", "edits": [{ "oldText": "a", "newText": "b" }] });
    let prepared = (definition.prepare_arguments.as_ref().unwrap())(&input).unwrap();
    assert_eq!(prepared, input);
}

#[test]
pub(super) fn legacy_passes_through_non_object_input_unchanged() {
    let definition = create_edit_tool_definition(".", None);
    let prepare = definition.prepare_arguments.as_ref().unwrap();
    assert_eq!(
        prepare(&serde_json::Value::Null).unwrap(),
        serde_json::Value::Null
    );
    assert_eq!(prepare(&json!("garbage")).unwrap(), json!("garbage"));
}

#[test]
pub(super) fn legacy_prepared_args_execute_correctly() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("legacy.txt");
        std::fs::write(&file_path, "before\n").unwrap();

        let definition = create_edit_tool_definition(dir.path().to_string_lossy().as_ref(), None);
        let prepared = (definition.prepare_arguments.as_ref().unwrap())(&json!({
            "path": "legacy.txt",
            "oldText": "before",
            "newText": "after"
        }))
        .unwrap();

        let result = (definition.execute)("tool-1", &prepared, None, None, None)
            .await
            .unwrap();
        assert_eq!(
            first_text(&result),
            "Successfully replaced 1 block(s) in legacy.txt."
        );
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "after\n");
    });
}

#[test]
pub(super) fn legacy_parses_edits_from_a_json_string() {
    let definition = create_edit_tool_definition(".", None);
    let prepared = (definition.prepare_arguments.as_ref().unwrap())(&json!({
        "path": "file.txt",
        "edits": "[{\"oldText\": \"a\", \"newText\": \"b\"}]"
    }))
    .unwrap();
    assert_eq!(
        prepared,
        json!({ "path": "file.txt", "edits": [{ "oldText": "a", "newText": "b" }] })
    );
}

#[test]
pub(super) fn legacy_leaves_edits_alone_when_the_string_is_not_valid_json() {
    let definition = create_edit_tool_definition(".", None);
    let prepared = (definition.prepare_arguments.as_ref().unwrap())(&json!({
        "path": "file.txt",
        "edits": "not json"
    }))
    .unwrap();
    assert_eq!(prepared, json!({ "path": "file.txt", "edits": "not json" }));
}

// ---------------------------------------------------------------------------
// file-mutation-queue.test.ts: built-in edit and write tools
// ---------------------------------------------------------------------------

#[test]
pub(super) fn mutation_queue_preserves_both_parallel_edits_on_the_same_file() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("parallel-edit.txt");
        std::fs::write(&file_path, "alpha\nbeta\ngamma\n").unwrap();

        let operations = slow_disk_edit_operations();
        let tool = create_edit_tool(
            dir.path().to_string_lossy().as_ref(),
            Some(EditToolOptions {
                operations: Some(operations),
            }),
        );

        let (first, second) = tokio::join!(
            run_tool_edit(&tool, &file_path, "alpha", "ALPHA"),
            run_tool_edit(&tool, &file_path, "beta", "BETA"),
        );
        first.unwrap();
        second.unwrap();

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "ALPHA\nBETA\ngamma\n");
    });
}

pub(super) async fn run_tool_edit(
    tool: &pi_agent_core::harness::types::AgentHarnessTool,
    file_path: &std::path::Path,
    old_text: &str,
    new_text: &str,
) -> Result<AgentToolResult, AgentToolError> {
    super::helpers::run_tool(
        tool,
        json!({
            "path": file_path.to_string_lossy(),
            "edits": [{ "oldText": old_text, "newText": new_text }]
        }),
        None,
        None,
        &background_context(),
    )
    .await
}

#[test]
pub(super) fn mutation_queue_shares_the_queue_between_edit_and_write() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("mixed.txt");
        std::fs::write(&file_path, "original\n").unwrap();

        let edit_operations = slow_disk_edit_operations();
        let write_operations = WriteOperations {
            mkdir: Arc::new(|_dir| Box::pin(async { Ok(()) })),
            write_file: Arc::new(|path, content| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    tokio::fs::write(&path, content.as_bytes())
                        .await
                        .map_err(AgentToolError::from)
                })
            }),
        };
        let edit_tool = create_edit_tool(
            dir.path().to_string_lossy().as_ref(),
            Some(EditToolOptions {
                operations: Some(edit_operations),
            }),
        );
        let write_tool = create_write_tool_definition(
            dir.path().to_string_lossy().as_ref(),
            Some(WriteToolOptions {
                operations: Some(write_operations),
            }),
        );

        let context = background_context();
        let edit = super::helpers::run_tool(
            &edit_tool,
            json!({
                "path": file_path.to_string_lossy(),
                "edits": [{ "oldText": "original", "newText": "edited" }]
            }),
            None,
            None,
            &context,
        );
        let write = run_definition(
            &write_tool,
            json!({ "path": file_path.to_string_lossy(), "content": "replacement\n" }),
            None,
            None,
        );
        let (edit, write) = tokio::join!(edit, write);
        edit.unwrap();
        write.unwrap();

        let readback = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(readback, "replacement\n");
    });
}

#[test]
pub(super) fn mutation_queue_keeps_write_queue_locked_while_an_aborted_write_is_still_in_flight() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("abort-write.txt");
        let first_write_started = tokio::sync::watch::channel(false);
        let finish_first_write = tokio::sync::watch::channel(false);
        let second_write_started = tokio::sync::watch::channel(false);
        let first_write_settled = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let write_operations = {
            let first_write_started = first_write_started.0.clone();
            let finish_first_write = finish_first_write.1.clone();
            let second_write_started = second_write_started.0.clone();
            let first_write_settled = Arc::clone(&first_write_settled);
            WriteOperations {
                mkdir: Arc::new(|_dir| Box::pin(async { Ok(()) })),
                write_file: Arc::new(move |path, content| {
                    let first_write_started = first_write_started.clone();
                    let mut finish_first_write = finish_first_write.clone();
                    let second_write_started = second_write_started.clone();
                    let first_write_settled = Arc::clone(&first_write_settled);
                    Box::pin(async move {
                        if content == "first\n" {
                            let _started = first_write_started.send(true);
                            while !*finish_first_write.borrow() {
                                if finish_first_write.changed().await.is_err() {
                                    break;
                                }
                            }
                            tokio::fs::write(&path, content.as_bytes())
                                .await
                                .map_err(AgentToolError::from)?;
                            first_write_settled.store(true, std::sync::atomic::Ordering::SeqCst);
                            return Ok(());
                        }

                        if content == "second\n" {
                            assert!(first_write_settled.load(std::sync::atomic::Ordering::SeqCst));
                            let _started = second_write_started.send(true);
                        }
                        tokio::fs::write(&path, content.as_bytes())
                            .await
                            .map_err(AgentToolError::from)
                    })
                }),
            }
        };
        let tool = create_write_tool_definition(
            dir.path().to_string_lossy().as_ref(),
            Some(WriteToolOptions {
                operations: Some(write_operations),
            }),
        );

        let (_context, controller) = with_cancel(&background_context());
        let signal = controller.signal().clone();
        let first = run_definition(
            &tool,
            json!({ "path": file_path.to_string_lossy(), "content": "first\n" }),
            Some(&signal),
            None,
        );
        tokio::pin!(first);
        // Drive the first write until its operation started.
        loop {
            if *first_write_started.1.borrow() {
                break;
            }
            tokio::select! {
                settled = &mut first => panic!("the first write settled early: {settled:?}"),
                () = tokio::time::sleep(Duration::from_millis(2)) => {}
            }
        }
        controller.abort_without_reason();

        let second = run_definition(
            &tool,
            json!({ "path": file_path.to_string_lossy(), "content": "second\n" }),
            None,
            None,
        );
        tokio::pin!(second);
        // The second write must stay queued while the aborted first write
        // is still in flight, upstream's `resolvesWithin(..., 20)` probe.
        let mut second_settled = false;
        tokio::select! {
            settled = &mut second => {
                second_settled = true;
                let _ = settled;
            }
            () = tokio::time::sleep(Duration::from_millis(20)) => {}
        }
        assert!(
            !second_settled,
            "the second write settled while the first was in flight"
        );
        assert!(
            !*second_write_started.1.borrow(),
            "the second write started while the first was in flight"
        );

        finish_first_write.0.send(true).unwrap();
        let first_error = first.await.unwrap_err();
        assert_eq!(first_error.to_string(), "Operation aborted");
        second.await.unwrap();

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "second\n");
    });
}

#[test]
fn the_options_and_operations_types_render_debug() {
    // The Debug impls are diagnostics-only surfaces; rendering each one pins
    // them without asserting a layout.
    let operations = BashOperations {
        exec: Arc::new(|_command: &str, _cwd: &str, _options: BashExecOptions| {
            Box::pin(async { Ok(BashExecOutcome { exit_code: Some(0) }) })
        }),
    };
    let grep_operations = GrepOperations {
        is_directory: Arc::new(|_path: String| Box::pin(async { None::<bool> })),
        read_file: Arc::new(|_path: String| Box::pin(async { Ok(String::new()) })),
    };
    let find_operations = FindOperations {
        exists: Arc::new(|_path: String| Box::pin(async { true })),
        glob: None,
    };
    let ls_operations = LsOperations {
        exists: Arc::new(|_path: String| Box::pin(async { true })),
        stat: Arc::new(|_path: String| Box::pin(async { None::<bool> })),
        read_dir: Arc::new(|_path: String| Box::pin(async { Ok(Vec::new()) })),
    };
    let read_operations = ReadOperations {
        read_file: Arc::new(|_path: String| {
            Box::pin(async { Ok::<Vec<u8>, AgentToolError>(Vec::new()) })
        }),
        access: Arc::new(|_path: String| Box::pin(async { Ok::<(), AgentToolError>(()) })),
        detect_image_mime_type: None,
    };
    let write_operations = WriteOperations {
        mkdir: Arc::new(|_dir: String| Box::pin(async { Ok::<(), AgentToolError>(()) })),
        write_file: Arc::new(|_path: String, _content: String| {
            Box::pin(async { Ok::<(), AgentToolError>(()) })
        }),
    };
    let edit_operations = EditOperations {
        access: Arc::new(|_path: String| Box::pin(async { Ok::<(), AgentToolError>(()) })),
        read_file: Arc::new(|_path: String| {
            Box::pin(async { Ok::<Vec<u8>, AgentToolError>(Vec::new()) })
        }),
        write_file: Arc::new(|_path: String, _content: String| {
            Box::pin(async { Ok::<(), AgentToolError>(()) })
        }),
    };
    let rendered = [
        format!(
            "{:?}",
            BashExecOptions {
                on_data: Arc::new(|_data: &[u8]| {}),
                signal: None,
                timeout: None,
                env: None,
            }
        ),
        format!("{operations:?}"),
        format!("{:?}", BashToolOptions::default()),
        format!("{:?}", GrepToolOptions::default()),
        format!("{grep_operations:?}"),
        format!("{:?}", FindToolOptions::default()),
        format!("{find_operations:?}"),
        format!("{:?}", LsToolOptions::default()),
        format!("{ls_operations:?}"),
        format!("{:?}", ReadToolOptions::default()),
        format!("{read_operations:?}"),
        format!("{:?}", WriteToolOptions::default()),
        format!("{write_operations:?}"),
        format!("{:?}", EditToolOptions::default()),
        format!("{edit_operations:?}"),
        format!("{:?}", ToolsOptions::default()),
        format!("{:?}", PowerShellToolOptions::default()),
    ];
    for rendering in rendered {
        assert!(!rendering.is_empty(), "{rendering}");
    }
}

#[test]
fn ls_lists_a_directory_through_the_default_operations() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("alpha.txt"), "").unwrap();
        std::fs::create_dir_all(dir.path().join("subdir")).unwrap();
        std::fs::write(dir.path().join("subdir").join("nested.txt"), "").unwrap();
        let tool = create_ls_tool_definition("/", None);
        let result = run_definition(
            &tool,
            json!({ "path": dir.path().to_string_lossy() }),
            None,
            None,
        )
        .await
        .unwrap();
        let output = first_text(&result);
        assert!(output.contains("alpha.txt"), "{output}");
        assert!(output.contains("subdir"), "{output}");
    });
}

#[test]
fn write_maps_a_failing_operation_to_a_tool_error() {
    block_on(async {
        let operations = WriteOperations {
            mkdir: Arc::new(|_dir: String| {
                Box::pin(async { Err(AgentToolError::from(std::io::Error::other("denied"))) })
            }),
            write_file: Arc::new(|_path: String, _content: String| Box::pin(async { Ok(()) })),
        };
        let tool = create_write_tool_definition(
            "/",
            Some(WriteToolOptions {
                operations: Some(operations),
            }),
        );
        let error = run_definition(
            &tool,
            json!({ "path": "out/put.txt", "content": "x" }),
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("denied"), "{error}");
    });
}

#[test]
fn ls_flags_the_entry_limit_and_wires_the_details() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        for index in 0..5 {
            std::fs::write(dir.path().join(format!("entry-{index}.txt")), "").unwrap();
        }
        let tool = create_ls_tool_definition("/", None);
        let result = run_definition(
            &tool,
            json!({ "path": dir.path().to_string_lossy(), "limit": 2 }),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(first_text(&result).contains("entries limit reached"));
        assert_eq!(result.details["entryLimitReached"], 2);
    });
}

#[test]
fn ls_rejects_a_file_path() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("plain-file.txt");
        std::fs::write(&file, "").unwrap();
        let tool = create_ls_tool_definition("/", None);
        let error = run_definition(&tool, json!({ "path": file.to_string_lossy() }), None, None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("Not a directory"), "{error}");
    });
}

/// The session-carrying context stub with a text-only model, the
/// non-vision model the read-image cases need.
struct TextOnlyModelContext {
    cwd: String,
}

impl ExtensionContext for TextOnlyModelContext {
    fn cwd(&self) -> &str {
        &self.cwd
    }

    fn model(&self) -> Option<&pi_ai::types::Model> {
        static MODEL: std::sync::OnceLock<pi_ai::types::Model> = std::sync::OnceLock::new();
        Some(MODEL.get_or_init(|| pi_ai::types::Model {
            id: "stub".to_owned(),
            name: "Stub".to_owned(),
            api: pi_ai::types::Api::from("openai-completions"),
            provider: pi_ai::types::ProviderId("stub".to_owned()),
            base_url: "https://example.test".to_owned(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![pi_ai::types::Modality::Text],
            cost: pi_ai::types::ModelCost {
                rates: pi_ai::types::ModelCostRates::default(),
                tiers: None,
            },
            context_window: 128_000,
            max_tokens: 8_192,
            sampling_params: None,
            headers: None,
            compat: None,
        }))
    }

    fn thinking_level(&self) -> Option<pi_agent_core::types::ThinkingLevel> {
        None
    }

    fn session_id(&self) -> Option<String> {
        None
    }

    fn session_file(&self) -> Option<String> {
        None
    }
}

#[test]
fn read_attaches_the_non_vision_note_when_the_model_cannot_take_images() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        // A real 1x1 red 24bpp BMP: the magic probe reads the file header.
        std::fs::write(dir.path().join("pixel.bmp"), tiny_bmp_1x1_red_24bpp()).unwrap();
        let tool = create_read_tool_definition("/", None);
        let context = TextOnlyModelContext {
            cwd: dir.path().to_string_lossy().into_owned(),
        };
        let result = run_definition(&tool, json!({ "path": "pixel.bmp" }), None, Some(&context))
            .await
            .unwrap();
        let output = first_text(&result);
        assert!(output.contains("does not support images"), "{output}");
        // The block still rides the result; the model layer omits it for the
        // request, upstream's note-and-image pairing.
        assert!(image_content(&result).is_some(), "{output}");
    });
}

#[test]
fn edit_reports_a_missing_file_with_the_access_error() {
    block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let tool = create_edit_tool_definition("/", None);
        let context = CwdContext {
            cwd: dir.path().to_string_lossy().into_owned(),
        };
        let error = run_definition(
            &tool,
            json!({
                "path": "no-such-file.txt",
                "edits": [{ "oldText": "a", "newText": "b" }]
            }),
            None,
            Some(&context),
        )
        .await
        .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("Could not edit file") || message.contains("no-such-file.txt"),
            "{message}"
        );
    });
}

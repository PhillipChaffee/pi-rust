//! Boundary tests for the compute-preview additions, upstream's
//! `edit-diff.ts` (the shared machinery rides pi-agent-core's edit-diff
//! suite; the coding-agent fork's `computeEditsDiff` cases bind here).

#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use pi_agent_core::harness::tools::edit_diff::Edit;

use crate::tools::edit_diff::{compute_edit_diff, compute_edits_diff, edit_access_error};

fn edits(pairs: &[(&str, &str)]) -> Vec<Edit> {
    pairs
        .iter()
        .map(|(old_text, new_text)| Edit {
            old_text: (*old_text).to_owned(),
            new_text: (*new_text).to_owned(),
        })
        .collect()
}

#[tokio::test]
async fn computes_the_display_diff_for_a_multi_edit() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("multi.txt");
    std::fs::write(&file_path, "alpha\nbeta\ngamma\n").unwrap();
    let path_string = file_path.to_string_lossy().into_owned();

    let result = compute_edits_diff(
        &path_string,
        &edits(&[("alpha\n", "ALPHA\n"), ("gamma\n", "GAMMA\n")]),
        "/",
    )
    .await
    .unwrap();
    assert!(result.diff.contains("+1 ALPHA"), "{}", result.diff);
    assert!(result.diff.contains("+3 GAMMA"), "{}", result.diff);
    assert_eq!(result.first_changed_line, Some(1));
}

#[tokio::test]
async fn includes_enoent_in_the_diff_preview_for_missing_files() {
    let dir = tempfile::tempdir().unwrap();
    let missing_file = dir.path().join("missing-preview.txt");
    let missing = missing_file.to_string_lossy().into_owned();

    let error = compute_edits_diff(&missing, &edits(&[("hello", "world")]), "/")
        .await
        .unwrap_err();
    assert_eq!(
        error,
        format!("Could not edit file: {missing}. Error code: ENOENT.")
    );
}

#[tokio::test]
#[cfg(unix)]
async fn includes_eacces_in_the_diff_preview_for_unreadable_files() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let unreadable_file = dir.path().join("unreadable-preview.txt");
    std::fs::write(&unreadable_file, "hello\n").unwrap();
    std::fs::set_permissions(&unreadable_file, std::fs::Permissions::from_mode(0o222)).unwrap();
    let unreadable = unreadable_file.to_string_lossy().into_owned();

    let error = compute_edits_diff(&unreadable, &edits(&[("hello", "world")]), "/")
        .await
        .unwrap_err();
    assert_eq!(
        error,
        format!("Could not edit file: {unreadable}. Error code: EACCES.")
    );
}

#[tokio::test]
async fn reports_the_edit_application_rejections() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("no-match.txt");
    std::fs::write(&file_path, "completely different content\n").unwrap();
    let path_string = file_path.to_string_lossy().into_owned();

    let error = compute_edits_diff(&path_string, &edits(&[("this does not exist", "x")]), "/")
        .await
        .unwrap_err();
    assert!(
        error.contains("Could not find the exact text in"),
        "{error}"
    );

    let empty_edits_error = compute_edits_diff(&path_string, &edits(&[]), "/")
        .await
        .unwrap_err();
    // The preview runs the shared application, whose zero-edit case falls
    // through to the no-change rejection; the tool's `edits must contain
    // at least one replacement` guard binds at validateEditInput instead.
    assert!(
        empty_edits_error.contains("No changes made to"),
        "{empty_edits_error}"
    );
}

#[tokio::test]
async fn single_edit_wrapper_routes_through_the_multi_edit_body() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("single.txt");
    std::fs::write(&file_path, "before\n").unwrap();
    let path_string = file_path.to_string_lossy().into_owned();

    let result = compute_edit_diff(&path_string, "before", "after", "/")
        .await
        .unwrap();
    assert!(result.diff.contains("+1 after"), "{}", result.diff);
    assert_eq!(result.first_changed_line, Some(1));
}

#[test]
fn the_access_error_message_classifies_by_code() {
    assert_eq!(
        edit_access_error("f.txt", &std::io::Error::from(std::io::ErrorKind::NotFound)),
        "Could not edit file: f.txt. Error code: ENOENT."
    );
    assert_eq!(
        edit_access_error(
            "f.txt",
            &std::io::Error::from(std::io::ErrorKind::PermissionDenied)
        ),
        "Could not edit file: f.txt. Error code: EACCES."
    );
    let other = std::io::Error::other("disk offline");
    assert_eq!(
        edit_access_error("f.txt", &other),
        "Could not edit file: f.txt. Error: disk offline."
    );
}

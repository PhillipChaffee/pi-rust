//! Edit-diff computation for the coding-agent tool surface, ported from
//! upstream `src/core/tools/edit-diff.ts`.
//!
//! The shared machinery — line-ending detection, LF normalization, the
//! fuzzy-match ladder, and the two diff generators — is upstream's
//! `packages/agent/src/harness/tools/edit-diff.ts` verbatim (the coding-agent
//! copy forks it, moving `stripBom` to `utils/text.ts` and adding the
//! compute-preview pair); it rides
//! [`pi_agent_core::harness::tools::edit_diff`] so the two crates carry one
//! hand-ported jsdiff. This module adds what the coding-agent fork adds:
//! [`compute_edits_diff`] and [`compute_edit_diff`], the preflight previews
//! the edit renderer draws before the tool executes.

use pi_agent_core::harness::tools::edit_diff::{
    Edit, apply_edits_to_normalized_content, generate_diff_string, normalize_to_lf,
};

use super::path_utils;

/// The computed preview, upstream's `EditDiffResult`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditDiffResult {
    /// The display diff with line numbers and context elision.
    pub diff: String,
    /// The first changed line number in the new file.
    pub first_changed_line: Option<usize>,
}

/// The single-file access failure message, upstream's
/// `Could not edit file: ${path}. ${errorMessage}.` — the error code when
/// the failure carries one, the message otherwise.
pub(crate) fn edit_access_error(path: &str, error: &std::io::Error) -> String {
    let error_message = match error.kind() {
        std::io::ErrorKind::NotFound => "Error code: ENOENT".to_owned(),
        std::io::ErrorKind::PermissionDenied => "Error code: EACCES".to_owned(),
        // The uncoded rendering, upstream's `String(error)` — the `Error:`
        // prefix JS `Error.prototype.toString` bakes in.
        _ => format!("Error: {error}"),
    };
    format!("Could not edit file: {path}. {error_message}.")
}

/// Compute the display diff for one or more edit operations without
/// applying them, upstream's `computeEditsDiff`. Used for preview rendering
/// in the TUI before the tool executes.
///
/// # Errors
/// The preflight failure message, upstream's `EditDiffError` — the access
/// failure or the edit application's rejection.
pub async fn compute_edits_diff(
    path: &str,
    edits: &[Edit],
    cwd: &str,
) -> Result<EditDiffResult, String> {
    let absolute_path = path_utils::resolve_to_cwd(path, cwd).map_err(|error| error.to_string())?;

    // Check if file exists and is readable
    if let Err(error) = tokio::fs::File::open(&absolute_path).await {
        return Err(edit_access_error(path, &error));
    }

    // Read the file
    let raw_content = tokio::fs::read(&absolute_path)
        .await
        .map_err(|error| error.to_string())?;
    let raw_content = String::from_utf8_lossy(&raw_content).into_owned();

    // Strip BOM before matching (LLM won't include invisible BOM in oldText)
    let content = crate::utils::text::strip_bom(&raw_content);
    let normalized_content = normalize_to_lf(content);
    let applied = apply_edits_to_normalized_content(&normalized_content, edits, path)?;
    let base_content = &applied.base_content;
    let new_content = &applied.new_content;

    // Generate the diff
    let generated = generate_diff_string(base_content, new_content, 4);
    Ok(EditDiffResult {
        diff: generated.diff,
        first_changed_line: generated.first_changed_line,
    })
}

/// Compute the diff for a single edit operation without applying it,
/// upstream's `computeEditDiff`. Kept as a convenience wrapper for
/// single-edit callers.
///
/// # Errors
/// [`compute_edits_diff`]'s preflight failure message.
pub async fn compute_edit_diff(
    path: &str,
    old_text: &str,
    new_text: &str,
    cwd: &str,
) -> Result<EditDiffResult, String> {
    compute_edits_diff(
        path,
        &[Edit {
            old_text: old_text.to_owned(),
            new_text: new_text.to_owned(),
        }],
        cwd,
    )
    .await
}

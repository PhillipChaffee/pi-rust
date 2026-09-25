//! Compatibility collector for callers that need one bounded final view,
//! ported from upstream `src/harness/utils/shell-output.ts`.
//!
//! Source-side capture, adaptive publication, and spilling remain owned by
//! the execution environment; this module folds the published view changes
//! into the final [`ShellCaptureResult`] callers consume.

use std::sync::{Arc, Mutex};

use crate::harness::context::Context;
use crate::harness::types::{ExecutionEnv, ExecutionError, ShellExecOptions, ShellOutputView};
use crate::harness::utils::output_capture::{ShellUpdateListener, apply_shell_output_update};
use crate::harness::utils::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, truncate_tail,
};

/// The progress view of one shell capture, upstream's
/// `ShellCaptureProgress`.
#[derive(Clone, Debug, Default)]
pub struct ShellCaptureProgress {
    /// The retained output text.
    pub output: String,
    /// The truncation metadata, restated with the content copy the
    /// progress view carries upstream.
    pub truncation: TruncationResult,
    /// The execution-environment-local file preserving complete output,
    /// when the capture spilled.
    pub full_output_path: Option<String>,
    /// Bytes of the partially truncated last line.
    pub last_line_bytes: u64,
}

/// Options for [`execute_shell_with_capture`], upstream's
/// `ShellCaptureOptions extends Omit<ShellExecOptions, "capture" |
/// "onUpdate">`.
#[derive(Clone, Default)]
pub struct ShellCaptureOptions {
    /// Working directory for the command.
    pub cwd: Option<String>,
    /// Environment variables for the command.
    pub env: Option<std::collections::BTreeMap<String, String>>,
    /// Whether to inherit the execution environment's default variables.
    pub inherit_env: Option<bool>,
    /// Timeout in seconds.
    pub timeout: Option<f64>,
    /// Called with each incremental chunk and a progress getter. A
    /// metadata-only update and a post-cap replacement carry no new
    /// incremental chunk.
    pub on_chunk: Option<Arc<OnShellChunk>>,
    /// Return shell execution failures with captured output instead of as
    /// a failed [`Result`].
    pub return_execution_errors: bool,
}

impl std::fmt::Debug for ShellCaptureOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShellCaptureOptions")
            .field("cwd", &self.cwd)
            .field("env", &self.env)
            .field("inherit_env", &self.inherit_env)
            .field("timeout", &self.timeout)
            .field("on_chunk", &self.on_chunk.is_some())
            .field("return_execution_errors", &self.return_execution_errors)
            .finish()
    }
}

/// The per-chunk callback, upstream's
/// `onChunk?: (chunk, getProgress, context) => void`.
pub type OnShellChunk = dyn Fn(&str, &dyn Fn() -> ShellCaptureProgress, &Context) + Send + Sync;

/// The bounded final view one shell capture returns, upstream's
/// `ShellCaptureResult`.
#[derive(Debug, Default)]
pub struct ShellCaptureResult {
    /// The retained output text.
    pub output: String,
    /// The truncation metadata with the content copy.
    pub truncation: TruncationResult,
    /// The spill file path, when the capture spilled.
    pub full_output_path: Option<String>,
    /// Bytes of the partially truncated last line.
    pub last_line_bytes: u64,
    /// The process exit code; absent when the execution failed or was
    /// cancelled.
    pub exit_code: Option<i32>,
    /// Whether the execution was cancelled.
    pub cancelled: bool,
    /// Whether the output was truncated.
    pub truncated: bool,
    /// The execution failure, when the caller asked for execution errors
    /// inline.
    pub execution_error: Option<ExecutionError>,
}

fn progress_from(output: &ShellOutputView) -> ShellCaptureProgress {
    ShellCaptureProgress {
        output: output.text.clone(),
        truncation: TruncationResult {
            content: output.text.clone(),
            metadata: output.metadata.truncation.clone(),
        },
        full_output_path: output.metadata.spill_path.clone(),
        last_line_bytes: output.metadata.last_line_bytes.unwrap_or(0),
    }
}

/// Folds each published update into the shared compatibility view and
/// forwards incremental chunks to `on_chunk`; the view slot's guard
/// releases before the callback runs so its progress getter may re-lock.
fn shell_capture_update_sink(
    capture_output: Arc<Mutex<Option<ShellOutputView>>>,
    on_chunk: Option<Arc<OnShellChunk>>,
) -> ShellUpdateListener {
    Arc::new(move |update, update_context| {
        let mut slot = capture_output
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = slot.clone();
        let next = apply_shell_output_update(previous.as_ref(), update);
        // A metadata-only update and a post-cap replacement contain no
        // new incremental chunk. Reporting their complete view would
        // duplicate bytes for callers that accumulate this
        // compatibility callback.
        let chunk = match update {
            crate::harness::types::ShellOutputUpdate::Append { text, .. }
            | crate::harness::types::ShellOutputUpdate::Slide { text, .. } => Some(text.clone()),
            crate::harness::types::ShellOutputUpdate::Replace { .. } if previous.is_none() => {
                Some(next.text.clone())
            }
            _ => None,
        };
        *slot = Some(next);
        // The progress getter re-locks the shared view, so the update
        // slot's guard must release before the callback runs.
        drop(slot);
        if let (Some(chunk), Some(on_chunk)) = (chunk, on_chunk.as_ref()) {
            let getter_output = Arc::clone(&capture_output);
            on_chunk(
                &chunk,
                &move || {
                    let guard = getter_output
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    guard
                        .as_ref()
                        .map_or_else(ShellCaptureProgress::default, progress_from)
                },
                update_context,
            );
        }
    })
}

/// Executes `command` on the environment with default capture limits and
/// spilling, folding the published view into one final result.
///
/// # Errors
/// The environment's execution error when the command fails, unless the
/// run aborted (a cancelled result returns instead) or
/// `return_execution_errors` carries the failure as captured output.
pub async fn execute_shell_with_capture(
    env: &dyn ExecutionEnv,
    command: &str,
    options: Option<ShellCaptureOptions>,
    context: &Context,
) -> Result<ShellCaptureResult, ExecutionError> {
    let options = options.unwrap_or_default();
    let on_chunk = options.on_chunk.clone();
    let shared_output: Arc<Mutex<Option<ShellOutputView>>> = Arc::new(Mutex::new(None));
    let capture_output = Arc::clone(&shared_output);
    let exec_options = ShellExecOptions {
        cwd: options.cwd.clone(),
        env: options.env.clone(),
        inherit_env: options.inherit_env,
        timeout: options.timeout,
        capture: Some(crate::harness::types::ShellOutputCaptureOptions {
            limits: crate::harness::types::ShellOutputLimits {
                max_bytes: DEFAULT_MAX_BYTES,
                max_lines: DEFAULT_MAX_LINES,
                retain: Some(crate::harness::types::ShellOutputRetention::Tail),
            },
            spill: true,
        }),
        on_update: Some(shell_capture_update_sink(capture_output, on_chunk)),
    };
    let result = env.exec(command, Some(exec_options), context).await;

    let output = {
        let mut slot = shared_output
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        slot.take().unwrap_or_else(|| {
            let empty = truncate_tail(
                "",
                crate::harness::utils::truncate::TruncationOptions::default(),
            );
            ShellOutputView {
                text: empty.content,
                metadata: crate::harness::types::ShellOutputMetadata {
                    truncation: empty.metadata,
                    spill_path: None,
                    last_line_bytes: None,
                },
            }
        })
    };
    let progress = progress_from(&output);
    match result {
        Err(error) => {
            if error.code == crate::harness::types::ExecutionErrorCode::Aborted
                || context
                    .abort_signal()
                    .is_some_and(|signal| signal.aborted())
            {
                return Ok(ShellCaptureResult {
                    exit_code: None,
                    cancelled: true,
                    truncated: progress.truncation.metadata.truncated,
                    output: progress.output,
                    truncation: progress.truncation,
                    full_output_path: progress.full_output_path,
                    last_line_bytes: progress.last_line_bytes,
                    ..ShellCaptureResult::default()
                });
            }
            if options.return_execution_errors {
                return Ok(ShellCaptureResult {
                    exit_code: None,
                    cancelled: false,
                    truncated: progress.truncation.metadata.truncated,
                    execution_error: Some(error),
                    output: progress.output,
                    truncation: progress.truncation,
                    full_output_path: progress.full_output_path,
                    last_line_bytes: progress.last_line_bytes,
                });
            }
            Err(error)
        }
        Ok(result) => Ok(ShellCaptureResult {
            output: progress.output,
            truncation: progress.truncation,
            full_output_path: progress.full_output_path,
            last_line_bytes: progress.last_line_bytes,
            exit_code: Some(result.exit_code),
            cancelled: false,
            truncated: result.truncation.truncated,
            execution_error: None,
        }),
    }
}

/// The sanitized binary-output alias, upstream's
/// `export { sanitizeShellOutput as sanitizeBinaryOutput }`.
pub use crate::harness::utils::output_capture::sanitize_shell_output as sanitize_binary_output;

#[cfg(test)]
mod tests;

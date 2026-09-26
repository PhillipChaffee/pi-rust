//! The compatibility collector suite, ported from the behaviors upstream's
//! `shell-output.ts` carries (the execution-environment suites exercise
//! the end-to-end path; upstream has no dedicated unit file).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::sync::{Arc, Mutex};

use pi_chord::context::background_context;

use crate::harness::env::nodejs::NodeExecutionEnv;
use crate::harness::utils::shell_output::ShellCaptureProgress;
use crate::harness::utils::shell_output::{ShellCaptureOptions, execute_shell_with_capture};

fn env_at(root: &std::path::Path) -> NodeExecutionEnv {
    NodeExecutionEnv::new(root.to_string_lossy().into_owned(), None, None)
}

/// `execute_shell_with_capture` folds the bounded view and spills the
/// complete output through the environment, upstream's integration path.
#[tokio::test]
async fn executes_with_capture_folds_and_spills() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path());
    let result = execute_shell_with_capture(
        &env,
        "yes line | head -n 15000",
        None,
        &background_context(),
    )
    .await
    .expect("capture");
    assert!(result.truncated);
    let full_output_path = result.full_output_path.clone().expect("spill");
    let full_output = read_text_file(&env, &full_output_path).await;
    assert!(full_output.split('\n').count() > 10_000);
    assert!(result.output.len() < full_output.len());
    assert!(result.truncation.truncated);
}

/// The per-chunk callback reports incremental chunks, and the progress
/// getter carries the accumulated view.
#[tokio::test]
async fn the_on_chunk_callback_reports_incremental_chunks() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path());
    let chunks: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let result = execute_shell_with_capture(
        &env,
        "printf abc; printf def",
        Some(ShellCaptureOptions {
            on_chunk: Some({
                let chunks = Arc::clone(&chunks);
                Arc::new(
                    move |chunk: &str, progress: &dyn Fn() -> ShellCaptureProgress, _context| {
                        chunks.lock().expect("chunks lock").push(chunk.to_owned());
                        let _ = progress().output.len();
                    },
                )
            }),
            ..ShellCaptureOptions::default()
        }),
        &background_context(),
    )
    .await
    .expect("capture");
    let recorded = chunks.lock().expect("chunks lock").clone();
    assert!(!recorded.is_empty());
    assert!(result.output.contains("abc"));
}

async fn read_text_file(env: &NodeExecutionEnv, path: &str) -> String {
    use crate::harness::types::FileSystem as _;
    let context = background_context();
    env.read_text_file(path, &context)
        .await
        .expect("read spill")
}

/// The option bundle's debug view restates the callback presence without
/// exposing closure internals.
#[test]
fn the_capture_options_debug_view_reports_the_callback_presence() {
    let plain = ShellCaptureOptions::default();
    assert!(format!("{plain:?}").contains("on_chunk: false"));
    let with_callback = ShellCaptureOptions {
        on_chunk: Some(Arc::new(
            |_chunk: &str, _progress: &dyn Fn() -> ShellCaptureProgress, _context| (),
        )),
        ..ShellCaptureOptions::default()
    };
    assert!(format!("{with_callback:?}").contains("on_chunk: true"));
}

/// A run with no output folds the empty view, upstream's empty-view
/// fallback when nothing was published.
#[tokio::test]
async fn an_output_free_run_folds_the_empty_view() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path());
    let result = execute_shell_with_capture(&env, "true", None, &background_context())
        .await
        .expect("capture");
    assert!(result.output.is_empty());
    assert_eq!(result.exit_code, Some(0));
    assert!(!result.truncated);
    assert!(!result.cancelled);
    assert_eq!(result.full_output_path, None);
    assert_eq!(result.last_line_bytes, 0);
    assert!(result.execution_error.is_none());
}

/// A spawn failure surfaces inline when `return_execution_errors` carries
/// it, upstream's inline-error mode.
#[tokio::test]
async fn a_failed_run_returns_the_error_inline_when_requested() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path());
    let result = execute_shell_with_capture(
        &env,
        "true",
        Some(ShellCaptureOptions {
            cwd: Some("/nonexistent-dir-for-shell-capture-test".to_owned()),
            return_execution_errors: true,
            ..ShellCaptureOptions::default()
        }),
        &background_context(),
    )
    .await
    .expect("inline error result");
    let error = result.execution_error.expect("inline execution error");
    assert_eq!(
        error.code,
        crate::harness::types::ExecutionErrorCode::SpawnError
    );
    assert_eq!(result.exit_code, None);
    assert!(!result.cancelled);
    assert!(result.output.is_empty());
}

/// A spawn failure surfaces as the failed result without the inline flag,
/// upstream's default error propagation.
#[tokio::test]
async fn a_failed_run_surfaces_the_error_without_the_flag() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path());
    let error = execute_shell_with_capture(
        &env,
        "true",
        Some(ShellCaptureOptions {
            cwd: Some("/nonexistent-dir-for-shell-capture-test".to_owned()),
            ..ShellCaptureOptions::default()
        }),
        &background_context(),
    )
    .await
    .expect_err("spawn failure");
    assert_eq!(
        error.code,
        crate::harness::types::ExecutionErrorCode::SpawnError
    );
}

/// An aborted run folds a cancelled result instead of the error, upstream's
/// cancelled fold.
#[tokio::test]
async fn an_aborted_run_folds_a_cancelled_result() {
    let root = tempfile::tempdir().expect("temp root");
    let env = env_at(root.path());
    let (context, controller) = pi_chord::context::with_cancel(&background_context());
    let aborter = controller.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        aborter.abort("cancelled by the test");
    });
    let result = execute_shell_with_capture(&env, "sleep 30", None, &context)
        .await
        .expect("cancelled capture");
    assert!(result.cancelled);
    assert_eq!(result.exit_code, None);
    assert!(!result.truncated);
    assert!(result.execution_error.is_none());
}

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

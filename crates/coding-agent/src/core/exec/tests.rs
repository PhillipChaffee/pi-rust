#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use super::*;

#[tokio::test]
async fn executes_a_command_and_collects_both_pipes() {
    let result = exec_command("sh", &["-c", "echo out; echo err >&2; exit 0"], ".", None)
        .await
        .unwrap();
    assert_eq!(result.stdout.trim_end(), "out");
    assert_eq!(result.stderr.trim_end(), "err");
    assert_eq!(result.code, 0);
    assert!(!result.killed);
}

#[tokio::test]
async fn the_exit_code_surfaces_without_a_kill() {
    let result = exec_command("sh", &["-c", "exit 7"], ".", None)
        .await
        .unwrap();
    assert_eq!(result.code, 7);
    assert!(!result.killed);
}

#[tokio::test]
async fn the_working_directory_option_resolves_relative_paths() {
    let dir = tempfile::tempdir().unwrap();
    let options = ExecOptions {
        cwd: Some(dir.path().to_string_lossy().into_owned()),
        ..ExecOptions::default()
    };
    let result = exec_command("sh", &["-c", "pwd"], ".", Some(options))
        .await
        .unwrap();
    assert!(
        result
            .stdout
            .contains(dir.path().file_name().unwrap().to_str().unwrap())
    );
}

#[tokio::test]
async fn a_timeout_kills_the_command_and_reports_it() {
    let options = ExecOptions {
        timeout: Some(100),
        ..ExecOptions::default()
    };
    let result = exec_command("sh", &["-c", "sleep 30"], ".", Some(options))
        .await
        .unwrap();
    assert!(result.killed, "the timeout kills the command tree");
}

#[tokio::test]
async fn an_abort_signal_kills_the_command_and_reports_it() {
    let context = pi_agent_core::harness::context::background_context();
    let (_context, controller) = pi_agent_core::harness::context::with_cancel(&context);
    let options = ExecOptions {
        signal: Some(controller.signal().clone()),
        ..ExecOptions::default()
    };
    let run = tokio::spawn(exec_command("sh", &["-c", "sleep 30"], ".", Some(options)));
    // Give the spawn a beat to register the listener, then abort.
    tokio::time::sleep(Duration::from_millis(100)).await;
    controller.abort_without_reason();
    let result = run.await.unwrap().unwrap();
    assert!(result.killed, "the abort kills the command tree");
}

#[tokio::test]
async fn a_spawn_failure_is_an_error() {
    let error = exec_command("pi-coding-agent-no-such-binary", &[], ".", None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
}

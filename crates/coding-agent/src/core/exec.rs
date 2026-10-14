//! Shared command execution for extensions and custom tools, ported from
//! upstream `src/core/exec.ts`.
//!
//! The command spawns without a shell and the wait never hangs on inherited
//! stdio handles (upstream's `waitForChildProcess`); the abort signal and
//! the timeout kill the process — SIGTERM first, SIGKILL after the
//! five-second force window. The result always resolves: a wait error
//! resolves code 0, upstream's `code ?? 0` arm.

use std::sync::Arc;
use std::time::Duration;

use pi_agent_core::harness::context::AbortSignal;

use crate::utils::child_process::{OnChildChunk, wait_for_pipes};

/// How long after SIGTERM the force kill waits, upstream's 5000 ms
/// `setTimeout`.
const FORCE_KILL_GRACE_MS: u64 = 5_000;

/// The boxed pipe reader [`wait_for_pipes`]'s generic bounds accept.
type BoxReader = Box<dyn tokio::io::AsyncRead + Unpin + Send>;

/// Box a pipe reader into the type the wait core consumes.
fn box_reader<R: tokio::io::AsyncRead + Unpin + Send + 'static>(pipe: R) -> BoxReader {
    Box::new(pipe)
}

/// The execution options, upstream's `ExecOptions`.
#[derive(Clone, Debug, Default)]
pub struct ExecOptions {
    /// The abort signal that cancels the command.
    pub signal: Option<AbortSignal>,
    /// The timeout in milliseconds.
    pub timeout: Option<u64>,
    /// The working directory.
    pub cwd: Option<String>,
}

/// The execution outcome, upstream's `ExecResult`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecResult {
    /// The collected stdout.
    pub stdout: String,
    /// The collected stderr.
    pub stderr: String,
    /// The exit code, 0 when the wait resolved without one, upstream's
    /// `code ?? 0`.
    pub code: i32,
    /// Whether the command was killed (by abort or timeout), upstream's
    /// `killed`.
    pub killed: bool,
}

/// Execute a command and return stdout/stderr/code, upstream's `execCommand`.
///
/// No shell, `["ignore", "pipe", "pipe"]` stdio, and the pipe-drain wait so
/// detached descendants holding the pipes cannot hang the resolve.
///
/// # Errors
/// When the child cannot spawn, upstream's spawn failure.
pub async fn exec_command(
    command: &str,
    args: &[&str],
    _cwd: &str,
    options: Option<ExecOptions>,
) -> Result<ExecResult, std::io::Error> {
    let options = options.unwrap_or_default();
    let mut cmd = tokio::process::Command::new(command);
    cmd.args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(cwd) = options.cwd.as_deref() {
        cmd.current_dir(cwd);
    }
    let mut child = cmd.spawn()?;

    let stdout_chunks = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let stderr_chunks = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let stdout_sink = Arc::clone(&stdout_chunks);
    let stderr_sink = Arc::clone(&stderr_chunks);
    let collect: OnChildChunk = Arc::new(move |chunk: &[u8], is_stderr: bool| {
        let sink = if is_stderr {
            stderr_sink.as_ref()
        } else {
            stdout_sink.as_ref()
        };
        sink.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend_from_slice(chunk);
    });

    // The kill is one shot, upstream's `killProcess` guard: the abort
    // signal, the timeout, and the five-second force kill each race to
    // fire it once.
    let killed_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let kill_child = {
        let child_id = child.id();
        let killed_flag = Arc::clone(&killed_flag);
        move || {
            if killed_flag.swap(true, std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            if let Some(pid) = child_id {
                crate::utils::shell::kill_process_tree(pid);
            }
            // The force kill, upstream's delayed SIGKILL: it fires from its
            // own task and never blocks the resolve.
            if let Some(pid) = child_id {
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(FORCE_KILL_GRACE_MS)).await;
                    crate::utils::shell::kill_process_tree(pid);
                });
            }
        }
    };

    // The abort arm, upstream's `addEventListener("abort", killProcess, {
    // once: true })` with its pre-aborted check.
    let abort_task = options.signal.as_ref().map(|signal| {
        let signal = signal.clone();
        let kill = kill_child.clone();
        tokio::spawn(async move {
            if !signal.aborted() {
                signal.wait().await;
            }
            kill();
        })
    });

    // The timeout arm, upstream's `setTimeout(() => killProcess(), timeout)`.
    let timeout_task = options
        .timeout
        .filter(|timeout| *timeout > 0)
        .map(|timeout| {
            let kill = kill_child.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(timeout)).await;
                kill();
            })
        });

    let stdout_pipe: Option<BoxReader> = child.stdout.take().map(box_reader);
    let stderr_pipe: Option<BoxReader> = child.stderr.take().map(box_reader);
    let exit = async move { child.wait().await.ok().and_then(|settled| settled.code()) };
    let code = wait_for_pipes(
        stdout_pipe.unwrap_or_else(|| box_reader(tokio::io::empty())),
        stderr_pipe.unwrap_or_else(|| box_reader(tokio::io::empty())),
        exit,
        collect,
    )
    .await;

    if let Some(abort_task) = abort_task {
        abort_task.abort();
    }
    if let Some(timeout_task) = timeout_task {
        timeout_task.abort();
    }

    let stdout = String::from_utf8_lossy(
        &stdout_chunks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
    .into_owned();
    let stderr = String::from_utf8_lossy(
        &stderr_chunks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
    .into_owned();

    Ok(ExecResult {
        stdout,
        stderr,
        code: code.unwrap_or(0),
        killed: killed_flag.load(std::sync::atomic::Ordering::SeqCst),
    })
}

#[cfg(test)]
mod tests {
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
}

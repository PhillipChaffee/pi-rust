//! Clipboard command execution, upstream's `src/utils/clipboard-command.ts`.
//!
//! One command, one bounded result: a timed-out or failing command answers
//! `None`, an empty buffer is a successful result. Clipboard writers can
//! daemonize — when input is piped they get no stdout pipe to retain,
//! upstream's stdio shape.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

/// The command runner the clipboard belts drive, upstream's
/// `runClipboardCommand` module import — the seam the suites replace with
/// scripted results the way upstream's `vi.mock` replaces the module.
pub trait ClipboardCommandRunner: Send + Sync {
    /// Run one clipboard command; `None` means it failed. The options carry
    /// the input and the timeout, upstream's third parameter.
    fn run<'a>(
        &'a self,
        command: &'a str,
        args: &'a [String],
        options: &'a ClipboardCommandOptions,
    ) -> Pin<Box<dyn Future<Output = Option<Vec<u8>>> + 'a>>;
}

/// The process runner, the production default.
#[derive(Debug, Default)]
pub struct ProcessClipboardCommandRunner;

impl ClipboardCommandRunner for ProcessClipboardCommandRunner {
    fn run<'a>(
        &'a self,
        command: &'a str,
        args: &'a [String],
        options: &'a ClipboardCommandOptions,
    ) -> Pin<Box<dyn Future<Output = Option<Vec<u8>>> + 'a>> {
        Box::pin(async move { run_clipboard_command(command, args, Some(options)).await })
    }
}

/// The reader options, upstream's `{ timeoutMs: 5000 }` reads.
#[must_use]
pub fn read_options(timeout_ms: Option<u64>) -> ClipboardCommandOptions {
    ClipboardCommandOptions {
        input: None,
        timeout_ms,
        ..ClipboardCommandOptions::default()
    }
}

/// The per-command options, upstream's third parameter.
#[derive(Debug, Clone, Default)]
pub struct ClipboardCommandOptions {
    /// Text to pipe to the writer's stdin; a writer call gets no stdout
    /// pipe, upstream's stdio shape.
    pub input: Option<String>,
    /// Kill the command after this long; default 3000ms.
    pub timeout_ms: Option<u64>,
    /// Give up once more than this many output bytes arrive; default
    /// 50MiB.
    pub max_buffer_bytes: Option<usize>,
}

impl ClipboardCommandOptions {
    fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms.unwrap_or(3000))
    }

    fn max_buffer_bytes(&self) -> usize {
        self.max_buffer_bytes.unwrap_or(50 * 1024 * 1024)
    }
}

/// Run one clipboard command, upstream's `runClipboardCommand`.
///
/// `None` means the command failed — spawn failure, non-zero exit, the
/// timeout, or an output over the buffer cap; an empty buffer is a
/// successful result.
pub async fn run_clipboard_command(
    command: &str,
    args: &[String],
    options: Option<&ClipboardCommandOptions>,
) -> Option<Vec<u8>> {
    let options = options.cloned().unwrap_or_default();
    // Clipboard writers can daemonize. Do not give them output pipes to retain.
    let mut cmd = Command::new(command);
    cmd.args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(if options.input.is_none() {
            std::process::Stdio::piped()
        } else {
            std::process::Stdio::null()
        })
        .stderr(std::process::Stdio::null());
    let mut child = cmd.spawn().ok()?;
    let stdin = child.stdin.take();
    if let Some(input) = options.input.clone() {
        // A writer may exit before consuming all input; the write races the
        // wait and its errors die with the pipe, upstream's stdin error
        // listener.
        tokio::task::spawn(async move {
            if let Some(mut stdin) = stdin {
                let _written = stdin.write_all(input.as_bytes()).await;
                let _flushed = stdin.flush().await;
            }
        });
    }

    let mut stdout = child.stdout.take();
    let mut chunks: Vec<u8> = Vec::new();
    let mut exited: Option<Option<i32>> = None;
    let mut drained = stdout.is_none();
    let deadline = tokio::time::Instant::now() + options.timeout();

    loop {
        if exited.is_some() && drained {
            break;
        }
        let mut buffer = [0u8; 8192];
        let next_chunk = async {
            match stdout.as_mut() {
                Some(stdout) => stdout.read(&mut buffer).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            status = child.wait(), if exited.is_none() => {
                exited = Some(status.ok().and_then(|settled| settled.code()));
            }
            () = tokio::time::sleep_until(deadline) => {
                let _killed = child.start_kill();
                // Reap the killed child; tokio leaves a zombie otherwise.
                let _reaped = child.wait().await;
                return None;
            }
            read = next_chunk, if !drained => {
                match read {
                    Ok(0) | Err(_) => drained = true,
                    Ok(read) => {
                        chunks.extend_from_slice(&buffer[..read]);
                        if chunks.len() > options.max_buffer_bytes() {
                            let _killed = child.start_kill();
                            // Reap the killed child; tokio leaves a zombie otherwise.
                            let _reaped = child.wait().await;
                            return None;
                        }
                    }
                }
            }
        }
    }
    match exited.flatten() {
        Some(0) => Some(chunks),
        _ => None,
    }
}

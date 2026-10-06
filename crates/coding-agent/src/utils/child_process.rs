//! Synchronous and awaited child processes, upstream's
//! `src/utils/child-process.ts`.
//!
//! The `crossSpawn` detour exists upstream for win32 argument escaping; the
//! map's Windows exclusion makes the node spawn the only branch, so the
//! wrappers collapse onto `std::process::Command` / `tokio::process`.
//!
//! [`wait_for_child_process`] carries the pipe-drain contract: a
//! short-lived child can exit while a detached descendant keeps its
//! stdout/stderr pipe open (earendil-works/pi#5303), so after exit the wait
//! holds for the pipes to fall idle — the grace timer re-arms on every
//! chunk — instead of truncating a tail still being written.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::process::Child;

/// How long the post-exit stdio wait holds when the pipes go quiet,
/// upstream's `EXIT_STDIO_GRACE_MS`.
const EXIT_STDIO_GRACE_MS: u64 = 100;

/// The spawn-sync options upstream passes per call site.
///
/// Whether output is captured (the `which` probes) or discarded (the
/// extended-attribute writes), and how long the child may run before the
/// watchdog kills it.
#[derive(Debug, Clone)]
pub struct SpawnSyncOptions {
    /// Capture stdout as UTF-8; `false` is upstream's `stdio: "ignore"`.
    pub capture_output: bool,
    /// Kill the child when it runs longer, upstream's `timeout` option.
    pub timeout_ms: Option<u64>,
}

impl SpawnSyncOptions {
    /// Discard all output, upstream's `stdio: "ignore"` spawn with UTF-8
    /// encoding.
    pub const IGNORE_OUTPUT: Self = Self {
        capture_output: false,
        timeout_ms: None,
    };
}

/// The outcome of a synchronous spawn, upstream's
/// `SpawnSyncReturns<string>` reduced to the fields its callers read.
///
/// The exit status, absent when the child was killed or failed to spawn,
/// and the UTF-8 stdout.
#[derive(Debug, Clone, Default)]
pub struct SpawnSyncOutcome {
    /// The child's exit code; `None` on spawn failure or a watchdog kill.
    pub status: Option<i32>,
    /// The captured stdout, UTF-8 decoded like upstream's `encoding`.
    pub stdout: String,
}

/// Run one child to completion synchronously, upstream's `spawnProcessSync`
/// at its callers' options.
///
/// The stdout pipe drains on its own thread, the way node's `spawnSync`
/// reads while it waits, so a child writing more than the pipe buffer
/// cannot deadlock the wait.
///
/// # Panics
/// Never: the drain-buffer lock only guards a local vec whose guard the
/// same thread holds once.
#[must_use]
pub fn spawn_process_sync(
    command: &str,
    args: &[&str],
    options: &SpawnSyncOptions,
) -> SpawnSyncOutcome {
    let mut cmd = std::process::Command::new(command);
    cmd.args(args);
    if options.capture_output {
        cmd.stdout(std::process::Stdio::piped());
    } else {
        cmd.stdout(std::process::Stdio::null());
    }
    cmd.stderr(std::process::Stdio::null());
    cmd.stdin(std::process::Stdio::null());
    let Ok(mut child) = cmd.spawn() else {
        return SpawnSyncOutcome::default();
    };

    // The reader thread runs the drain concurrently with the exit wait.
    let stdout_pipe = child.stdout.take().filter(|_| options.capture_output);
    let drained = stdout_pipe.map(|mut pipe| {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let reader_buffer = Arc::clone(&buffer);
        let reader = std::thread::spawn(move || {
            // A poisoned lock means a previous read panicked mid-drain; the
            // partial tail still reads.
            let mut sink = reader_buffer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _ignored = std::io::Read::read_to_end(&mut pipe, &mut sink);
        });
        (buffer, reader)
    });

    let timeout = options.timeout_ms.map(Duration::from_millis);
    let started = Instant::now();
    let pid = child.id();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if timeout.is_some_and(|limit| started.elapsed() >= limit) {
                    super::shell::kill_process_tree(pid);
                    break child.wait().ok();
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => break None,
        }
    };
    if let Some((buffer, reader)) = drained {
        let _joined = reader.join();
        let stdout = buffer
            .lock()
            .map(|sink| String::from_utf8_lossy(&sink).into_owned())
            .unwrap_or_default();
        SpawnSyncOutcome {
            status: status.and_then(|status| status.code()),
            stdout,
        }
    } else {
        SpawnSyncOutcome {
            status: status.and_then(|status| status.code()),
            stdout: String::new(),
        }
    }
}

/// Which pipe a drain event arrived on.
#[derive(Clone, Copy)]
enum Drain {
    /// A chunk arrived on stdout.
    StdoutChunk,
    /// A chunk arrived on stderr.
    StderrChunk,
    /// The stdout pipe reached EOF, upstream's `end` event.
    StdoutEnd,
    /// The stderr pipe reached EOF, upstream's `end` event.
    StderrEnd,
}

/// Wait for a child process to terminate without hanging on inherited
/// stdio handles, upstream's `waitForChildProcess`.
///
/// After `exit` the pipes must fall idle before the result settles: the
/// grace timer re-arms on every chunk, so an actively writing descendant
/// keeps the drain alive, while a quiet inherited handle releases after
/// the `EXIT_STDIO_GRACE_MS` constant. Both pipes reaching EOF with the exit settle
/// immediately, upstream's `close` event. The child's exit code is `None`
/// when a signal killed it; spawn failures surface at spawn time in Rust
/// and the callers handle them there, upstream's rejected promise.
pub async fn wait_for_child_process(child: Child) -> Option<i32> {
    let mut child = child;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let mut stdout_ended = stdout.is_none();
    let mut stderr_ended = stderr.is_none();

    let (sender, mut drain_rx) = tokio::sync::mpsc::unbounded_channel();
    if let Some(pipe) = stdout {
        tokio::task::spawn(drain_pipe(
            pipe,
            sender.clone(),
            Drain::StdoutChunk,
            Drain::StdoutEnd,
        ));
    }
    if let Some(pipe) = stderr {
        tokio::task::spawn(drain_pipe(
            pipe,
            sender.clone(),
            Drain::StderrChunk,
            Drain::StderrEnd,
        ));
    }
    // The wait loop reads only the end markers; the guard keeps a fully
    // drained channel from being polled.
    drop(sender);

    let grace = Duration::from_millis(EXIT_STDIO_GRACE_MS);
    let mut exit_code: Option<Option<i32>> = None;
    // The idle timer only runs post-exit, upstream's armIdleTimer guards.
    let mut idle_deadline: Option<tokio::time::Instant> = None;

    loop {
        if exit_code.is_some() && stdout_ended && stderr_ended {
            break;
        }
        let next_drain = async {
            if stdout_ended && stderr_ended {
                std::future::pending().await
            } else {
                drain_rx.recv().await
            }
        };
        tokio::select! {
            status = child.wait(), if exit_code.is_none() => {
                exit_code = Some(status.ok().and_then(|settled| settled.code()));
                idle_deadline = Some(tokio::time::Instant::now() + grace);
            }
            drain = next_drain => {
                match drain {
                    Some(Drain::StdoutChunk | Drain::StderrChunk) => {
                        // Output is still arriving after exit; defer
                        // finalizing so the tail is not destroyed mid-write,
                        // upstream's onData re-arm.
                        if exit_code.is_some() {
                            idle_deadline = Some(tokio::time::Instant::now() + grace);
                        }
                    }
                    Some(Drain::StdoutEnd) => stdout_ended = true,
                    Some(Drain::StderrEnd) => stderr_ended = true,
                    // Unreachable while a drain task lives: both ends have
                    // been delivered and the guard pends the recv.
                    None => {}
                }
            }
            () = tokio::time::sleep_until(idle_deadline.unwrap_or_else(tokio::time::Instant::now)),
                if exit_code.is_some() && idle_deadline.is_some() => break,
        }
    }
    exit_code.flatten()
}

/// Drain one pipe to EOF on its own task, delivering chunk notifications
/// and then the end marker to the shared wait channel.
async fn drain_pipe<R>(
    mut pipe: R,
    sender: tokio::sync::mpsc::UnboundedSender<Drain>,
    chunk: Drain,
    end: Drain,
) where
    R: tokio::io::AsyncRead + Unpin + 'static,
{
    let mut buffer = [0u8; 8192];
    loop {
        match tokio::io::AsyncReadExt::read(&mut pipe, &mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                if sender.send(chunk).is_err() {
                    return;
                }
            }
        }
    }
    let _closed = sender.send(end);
}

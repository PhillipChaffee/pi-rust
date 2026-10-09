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
    /// A chunk arrived on either pipe.
    Chunk,
    /// The stdout pipe reached EOF, upstream's `end` event.
    StdoutEnd,
    /// The stderr pipe reached EOF, upstream's `end` event.
    StderrEnd,
}

/// The chunk listener the streaming wait feeds, shared by the bash tool's
/// operations (upstream's shared `onData` over both pipes). The boolean is
/// `true` for stderr chunks.
pub type OnChildChunk = Arc<dyn Fn(&[u8], bool) + Send + Sync>;

/// The pipe-and-exit wait core both wrappers share and the 5303 regression
/// drives directly.
///
/// `exit` resolves once with the exit code (`None` on a signal), the two
/// readers deliver every chunk to `on_chunk` and then an end marker. The
/// loop settles when the exit has landed and both pipes have fallen
/// idle — the grace timer re-arms on every chunk, so an actively writing
/// descendant keeps the drain alive while a quiet inherited handle releases
/// after `EXIT_STDIO_GRACE_MS`, upstream's re-armed grace timer over the
/// same stream events.
pub async fn wait_for_pipes<R1, R2, E>(
    stdout: R1,
    stderr: R2,
    exit: E,
    on_chunk: OnChildChunk,
) -> Option<i32>
where
    R1: tokio::io::AsyncRead + Unpin + Send + 'static,
    R2: tokio::io::AsyncRead + Unpin + Send + 'static,
    E: Future<Output = Option<i32>>,
{
    let (sender, mut drain_rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::task::spawn(drain_pipe(
        stdout,
        sender.clone(),
        Drain::Chunk,
        Drain::StdoutEnd,
        Arc::clone(&on_chunk),
        false,
    ));
    tokio::task::spawn(drain_pipe(
        stderr,
        sender,
        Drain::Chunk,
        Drain::StderrEnd,
        on_chunk,
        true,
    ));

    let mut exit = Box::pin(exit);
    // The outer `None` is "not yet exited"; the inner `None` is a signal
    // death, upstream's `null` exit code.
    let mut exit_code: Option<Option<i32>> = None;
    let mut stdout_ended = false;
    let mut stderr_ended = false;
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
            code = &mut exit, if exit_code.is_none() => {
                exit_code = Some(code);
                idle_deadline = Some(tokio::time::Instant::now() + grace());
            }
            drain = next_drain => {
                match drain {
                    Some(Drain::Chunk) => {
                        // Output is still arriving after exit; defer
                        // finalizing so the tail is not destroyed mid-write,
                        // upstream's onData re-arm.
                        if exit_code.is_some() {
                            idle_deadline = Some(tokio::time::Instant::now() + grace());
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

/// The post-exit idle grace, [`EXIT_STDIO_GRACE_MS`] as a duration.
const fn grace() -> Duration {
    Duration::from_millis(EXIT_STDIO_GRACE_MS)
}

/// Wait for a child process to terminate without hanging on inherited
/// stdio handles, upstream's `waitForChildProcess`.
///
/// The pipes' data is not consumed — callers attach their own listeners
/// (upstream's `on("data")`) or use [`wait_for_child_process_streaming`].
pub async fn wait_for_child_process(child: Child) -> Option<i32> {
    let mut child = child;
    let stdout: Box<dyn tokio::io::AsyncRead + Unpin + Send> = match child.stdout.take() {
        Some(pipe) => Box::new(pipe),
        None => Box::new(tokio::io::empty()),
    };
    let stderr: Box<dyn tokio::io::AsyncRead + Unpin + Send> = match child.stderr.take() {
        Some(pipe) => Box::new(pipe),
        None => Box::new(tokio::io::empty()),
    };
    let exit = async move { child.wait().await.ok().and_then(|settled| settled.code()) };
    wait_for_pipes(
        stdout,
        stderr,
        exit,
        Arc::new(|_chunk: &[u8], _is_stderr: bool| {}),
    )
    .await
}

/// Wait for a child process, delivering every stdout/stderr chunk to
/// `on_chunk` — the form the built-in shell operations run, upstream's data
/// listeners over both pipes plus the shared wait.
pub async fn wait_for_child_process_streaming(child: Child, on_chunk: OnChildChunk) -> Option<i32> {
    let mut child = child;
    let stdout: Box<dyn tokio::io::AsyncRead + Unpin + Send> = match child.stdout.take() {
        Some(pipe) => Box::new(pipe),
        None => Box::new(tokio::io::empty()),
    };
    let stderr: Box<dyn tokio::io::AsyncRead + Unpin + Send> = match child.stderr.take() {
        Some(pipe) => Box::new(pipe),
        None => Box::new(tokio::io::empty()),
    };
    let exit = async move { child.wait().await.ok().and_then(|settled| settled.code()) };
    wait_for_pipes(stdout, stderr, exit, on_chunk).await
}

/// Drain one pipe to EOF on its own task, delivering each chunk to the
/// shared listener and then the end marker to the wait channel.
async fn drain_pipe<R>(
    mut pipe: R,
    sender: tokio::sync::mpsc::UnboundedSender<Drain>,
    chunk: Drain,
    end: Drain,
    on_chunk: OnChildChunk,
    is_stderr: bool,
) where
    R: tokio::io::AsyncRead + Unpin + 'static,
{
    let mut buffer = [0u8; 8192];
    loop {
        match tokio::io::AsyncReadExt::read(&mut pipe, &mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                on_chunk(&buffer[..read], is_stderr);
                if sender.send(chunk).is_err() {
                    return;
                }
            }
        }
    }
    let _closed = sender.send(end);
}

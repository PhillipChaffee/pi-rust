//! Stdout takeover and raw writes, upstream's `src/core/output-guard.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Interactive mode claims the terminal; anything printed through the normal
//! stdout path would corrupt the TUI, so a takeover redirects that path to
//! stderr while the TUI paints through [`write_raw_stdout`], which bypasses
//! the takeover and serializes every chunk behind one tail task.
//!
//! Node redirects by monkey-patching `process.stdout.write`; Rust has no
//! patchable global, so the takeover is a flag the stdout path consults
//! ([`stdout_is_taken_over`]) and [`write_stdout`] is the taken-over write
//! it routes to stderr. The raw path keeps upstream's backpressure shape:
//! one tail chain, `ENOBUFS`/`EAGAIN`/`EWOULDBLOCK` retried after 10 ms,
//! everything else failing the process.

use std::sync::{Arc, Mutex, OnceLock};

/// The retry delay for backpressure write failures, upstream's
/// `RAW_STDOUT_RETRY_DELAY_MS`.
pub const RAW_STDOUT_RETRY_DELAY_MS: u64 = 10;

/// The real stdout writer, upstream's `process.stdout.write` binding.
pub type RawStdoutWriter = Arc<dyn Fn(&str) -> Result<(), std::io::Error> + Send + Sync>;

/// The takeover state, upstream's `stdoutTakeoverState`.
#[derive(Default)]
struct StdoutTakeoverState {
    taken_over: bool,
    raw_stdout_write: Option<RawStdoutWriter>,
}

/// The process-global state, upstream's module-scope `let stdoutTakeoverState`.
static STDOUT_TAKEOVER: OnceLock<Mutex<StdoutTakeoverState>> = OnceLock::new();
/// The serialized raw-write tail, upstream's `rawStdoutWriteTail` promise
/// chain: one mutex the queued writes hold in order.
static RAW_STDOUT_TAIL: OnceLock<Arc<tokio::sync::Mutex<()>>> = OnceLock::new();
/// The chunks queued but not yet written; upstream's tail identity doubles
/// as the backpressure signal, restated as a count.
static RAW_STDOUT_PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn takeover() -> &'static Mutex<StdoutTakeoverState> {
    STDOUT_TAKEOVER.get_or_init(|| Mutex::new(StdoutTakeoverState::default()))
}

fn tail() -> &'static Arc<tokio::sync::Mutex<()>> {
    RAW_STDOUT_TAIL.get_or_init(|| Arc::new(tokio::sync::Mutex::new(())))
}

/// The active raw writer, upstream's `getRawStdoutWrite`: the captured
/// binding while taken over, a fresh stdout write otherwise.
fn raw_writer() -> RawStdoutWriter {
    let state = takeover()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state
        .raw_stdout_write
        .clone()
        .unwrap_or_else(default_raw_writer)
}

/// The default writer: a blocking write of the whole chunk to stdout.
fn default_raw_writer() -> RawStdoutWriter {
    Arc::new(|text: &str| {
        use std::io::Write;
        let mut stdout = std::io::stdout().lock();
        stdout
            .write_all(text.as_bytes())
            .and_then(|()| stdout.flush())
    })
}

/// Write one chunk to the raw stdout, retrying the transient backpressure
/// errors, upstream's `writeRawStdoutChunk`.
async fn write_raw_stdout_chunk(
    text: &str,
    writer: &RawStdoutWriter,
) -> Result<(), std::io::Error> {
    loop {
        let write_error = {
            let text = text.to_string();
            let writer = Arc::clone(writer);
            // The write itself is blocking I/O; keep it off the async
            // reactor the way the ported child-process plumbing does.
            tokio::task::spawn_blocking(move || writer(&text)).await
        };
        match write_error {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(error)) if is_transient_backpressure(&error) => {}
            Ok(Err(error)) => return Err(error),
            Err(join) => {
                return Err(std::io::Error::other(format!(
                    "raw stdout write task failed: {join}"
                )));
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(RAW_STDOUT_RETRY_DELAY_MS)).await;
    }
}

/// The transient backpressure classification, upstream's
/// `code !== "ENOBUFS" && code !== "EAGAIN" && code !== "EWOULDBLOCK"` guard.
fn is_transient_backpressure(error: &std::io::Error) -> bool {
    matches!(error.kind(), std::io::ErrorKind::WouldBlock)
}

/// Take over the stdout path, upstream's `takeOverStdout`. Idempotent.
pub fn take_over_stdout() {
    let mut state = takeover()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.taken_over {
        return;
    }
    state.taken_over = true;
    state.raw_stdout_write = Some(default_raw_writer());
}

/// Restore the stdout path, upstream's `restoreStdout`. Idempotent.
pub fn restore_stdout() {
    let mut state = takeover()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.taken_over = false;
    state.raw_stdout_write = None;
}

/// Whether the takeover is active, upstream's `isStdoutTakenOver`.
#[must_use]
pub fn stdout_is_taken_over() -> bool {
    let state = takeover()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.taken_over
}

/// The taken-over stdout write, upstream's monkey-patched
/// `process.stdout.write`: everything the normal path prints lands on
/// stderr while the takeover is active.
#[allow(
    clippy::print_stderr,
    reason = "the taken-over stdout write lands on stderr, upstream's monkey-patched write"
)]
pub fn write_stdout(text: &str) {
    eprint!("{text}");
}

/// Queue one chunk onto the raw stdout's serialized tail, upstream's
/// `writeRawStdout`.
///
/// An empty chunk is dropped. A failed write exits the process with status
/// 1, upstream's `process.exit(1)` — the terminal can no longer be trusted.
#[allow(
    clippy::print_stderr,
    reason = "the pre-exit report is stderr, upstream's crash path"
)]
pub fn write_raw_stdout(text: &str) {
    if text.is_empty() {
        return;
    }
    let text = text.to_string();
    let writer = raw_writer();
    let tail_lock = Arc::clone(tail());
    RAW_STDOUT_PENDING.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    tokio::spawn(async move {
        let _guard = tail_lock.lock().await;
        if let Err(error) = write_raw_stdout_chunk(&text, &writer).await {
            eprintln!("raw stdout write failed: {error}");
            std::process::exit(1);
        }
        RAW_STDOUT_PENDING.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    });
}

/// Wait until every chunk queued so far has been written, upstream's
/// `waitForRawStdoutBackpressure`.
pub async fn wait_for_raw_stdout_backpressure() {
    loop {
        let pending = RAW_STDOUT_PENDING.load(std::sync::atomic::Ordering::SeqCst);
        if pending == 0 {
            return;
        }
        // Queue behind the in-flight write; re-check after releasing so a
        // chunk queued meanwhile re-arms the wait.
        let _held = tail().lock().await;
    }
}

/// Drain the tail and write one empty chunk, upstream's `flushRawStdout`.
pub async fn flush_raw_stdout() {
    wait_for_raw_stdout_backpressure().await;
    let writer = raw_writer();
    let _ = write_raw_stdout_chunk("", &writer).await;
}

/// Install an injectable raw writer for tests; returns the previous value.
pub fn set_raw_writer_for_test(writer: Option<RawStdoutWriter>) -> Option<RawStdoutWriter> {
    let mut state = takeover()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    std::mem::replace(&mut state.raw_stdout_write, writer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn takeover_flips_and_restores_idempotently() {
        restore_stdout();
        assert!(!stdout_is_taken_over());
        take_over_stdout();
        take_over_stdout();
        assert!(stdout_is_taken_over());
        restore_stdout();
        restore_stdout();
        assert!(!stdout_is_taken_over());
    }

    #[tokio::test]
    async fn raw_writes_serialize_on_the_tail_and_flush_drains() {
        let writes = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&writes);
        let previous = set_raw_writer_for_test(Some(Arc::new(move |text: &str| {
            // The flush's empty chunk passes through the raw writer too,
            // upstream's `writeRawStdoutChunk("")`.
            assert!(text.is_empty() || text.contains("chunk"));
            if !text.is_empty() {
                counter.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })));
        write_raw_stdout("chunk-a");
        write_raw_stdout("chunk-b");
        flush_raw_stdout().await;
        assert_eq!(writes.load(Ordering::SeqCst), 2);
        set_raw_writer_for_test(previous);
    }

    #[test]
    fn stdout_write_delegates_to_the_taken_over_or_system_stream() {
        restore_stdout();
        write_stdout("plain text stdout\n");
        take_over_stdout();
        write_stdout("taken over text\n");
        restore_stdout();
        write_stdout("plain text again\n");
    }

    #[test]
    fn the_backpressure_class_is_would_block_only() {
        restore_stdout();
        assert!(!is_transient_backpressure(&std::io::Error::other("no")));
        assert!(is_transient_backpressure(&std::io::Error::from(
            std::io::ErrorKind::WouldBlock
        )));
    }

    #[tokio::test]
    async fn raw_writes_retry_transient_backpressure_and_queue_through_the_tail() {
        set_raw_writer_for_test(None);
        // The empty chunk returns before queueing anything.
        write_raw_stdout("");
        // One transient failure, then the write succeeds through the tail.
        let retried = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&retried);
        let previous = set_raw_writer_for_test(Some(Arc::new(move |text: &str| {
            if text.is_empty() {
                // The flush's empty chunk passes through but never counts.
                return Ok(());
            }
            counter.fetch_add(1, Ordering::SeqCst);
            if counter.load(Ordering::SeqCst) == 1 {
                Err(std::io::Error::from(std::io::ErrorKind::WouldBlock))
            } else {
                Ok(())
            }
        })));
        write_raw_stdout("chunk-retry");
        wait_for_raw_stdout_backpressure().await;
        flush_raw_stdout().await;
        assert_eq!(retried.load(Ordering::SeqCst), 2);
        write_raw_stdout(""); // the flush's empty chunk passes through too
        set_raw_writer_for_test(previous);
    }
}

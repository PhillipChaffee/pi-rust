//! The `5303-bash-output-truncation` regression, ported against
//! [`wait_for_pipes`]'s core: after exit, a descendant holding the stdout
//! pipe open must not bin the output still being written — the grace
//! re-arms on every chunk — and must release after the grace once the pipe
//! goes quiet.
//!
//! Upstream drives the timings with fake timers; the paused tokio clock
//! auto-advances while the in-memory pipe leaves the runtime idle, so the
//! restatement pins the two behaviors with wall-clock margins around the
//! 100 ms grace: chunks land well inside it, and the settle is sampled
//! before the earliest broken-early resolution.

#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::sync::Arc;
use std::time::Duration;

use tokio::io::AsyncWriteExt as _;

use pi_agent_core::types::AgentToolError;

use crate::utils::child_process::wait_for_pipes;

/// The post-exit idle grace, [`crate::utils::child_process`]'s
/// `EXIT_STDIO_GRACE_MS`.
const GRACE: Duration = Duration::from_millis(100);

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn captures_output_emitted_after_exit_while_a_descendant_holds_stdout_open() {
    let (mut writer, reader) = tokio::io::duplex(64);
    let (exit_tx, exit_rx) = tokio::sync::oneshot::channel::<Option<i32>>();
    let chunks = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let sink = Arc::clone(&chunks);
    let wait = tokio::spawn(wait_for_pipes(
        reader,
        tokio::io::empty(),
        async move { exit_rx.await.unwrap_or(None) },
        Arc::new(move |chunk: &[u8], _is_stderr: bool| {
            sink.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(String::from_utf8_lossy(chunk).into_owned());
        }),
    ));
    writer.write_all(b"HEAD\n").await.unwrap();
    // The exit event; the write end stays open — the descendant's handle.
    exit_tx.send(Some(0)).unwrap();

    // Six ticks at 30 ms spacing: each inside the 100 ms grace the
    // re-arm keeps alive, upstream's 50 ms advance + write cadence.
    for index in 1..=6usize {
        tokio::time::sleep(Duration::from_millis(30)).await;
        writer
            .write_all(format!("TICK{index}\n").as_bytes())
            .await
            .unwrap();
    }

    // The broken wait resolved at exit + grace; the re-armed one cannot
    // settle before the last chunk's grace elapses.
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        !wait.is_finished(),
        "the wait settled while the pipe was still writing"
    );

    let settled = tokio::time::timeout(Duration::from_secs(2), wait)
        .await
        .expect("the wait released after the grace")
        .unwrap();
    assert_eq!(settled, Some(0));
    let collected = chunks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .join("");
    assert!(collected.contains("HEAD"));
    assert!(collected.contains("TICK6"));
    let _unused: Result<(), AgentToolError> = Ok(());
    drop(writer);
}

#[tokio::test(flavor = "current_thread", start_paused = false)]
async fn resolves_after_the_grace_when_a_descendant_holds_stdout_open_but_stays_quiet() {
    let (mut writer, reader) = tokio::io::duplex(64);
    let (exit_tx, exit_rx) = tokio::sync::oneshot::channel::<Option<i32>>();
    let wait = tokio::spawn(wait_for_pipes(
        reader,
        tokio::io::empty(),
        async move { exit_rx.await.unwrap_or(None) },
        Arc::new(|_chunk: &[u8], _is_stderr: bool| {}),
    ));
    writer.write_all(b"DONE\n").await.unwrap();
    exit_tx.send(Some(0)).unwrap();

    // The broken wait resolved immediately on exit; the held-open pipe
    // keeps the wait alive for the grace.
    tokio::time::sleep(GRACE / 2).await;
    assert!(!wait.is_finished(), "the wait settled before the grace");

    let settled = tokio::time::timeout(Duration::from_secs(2), wait)
        .await
        .expect("the wait released after the grace")
        .unwrap();
    assert_eq!(settled, Some(0));
    drop(writer);
}

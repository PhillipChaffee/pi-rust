//! Abort-aware sleeping, upstream's `src/utils/sleep.ts`.
//!
//! The abort surface rides the same `CancellationToken` the pi-ai crate's
//! abort belt carries, so the sleep races the token's cancellation with the
//! paused-clock timer the stack decision pins.

use std::time::Duration;

use tokio_util::sync::CancellationToken;

/// The error an aborted sleep rejects with, upstream's `new
/// Error("Aborted")`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SleepAborted;

impl std::fmt::Display for SleepAborted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Aborted")
    }
}

impl std::error::Error for SleepAborted {}

/// Sleep `ms`, giving up when the signal aborts.
///
/// # Errors
/// [`SleepAborted`] when the signal was already aborted or aborts first.
pub async fn sleep(ms: u64, signal: Option<&CancellationToken>) -> Result<(), SleepAborted> {
    if let Some(signal) = signal {
        if signal.is_cancelled() {
            return Err(SleepAborted);
        }
        return tokio::select! {
            () = signal.cancelled() => Err(SleepAborted),
            () = tokio::time::sleep(Duration::from_millis(ms)) => Ok(()),
        };
    }
    tokio::time::sleep(Duration::from_millis(ms)).await;
    Ok(())
}

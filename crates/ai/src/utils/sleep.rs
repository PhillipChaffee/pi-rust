//! An abortable sleep, ported from
//! `packages/ai/src/utils/sleep.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatement: the TypeScript helper rejects with the signal's
//! `reason`, which an `AbortSignal` may carry; a `CancellationToken` carries
//! no reason payload, so cancellation surfaces as the standard
//! `AbortError` (see [`crate::utils::abort`]).

use tokio_util::sync::CancellationToken;

use crate::utils::abort::AbortError;

/// Sleep for `ms` milliseconds, stopping early when the token aborts.
///
/// # Errors
/// Returns [`AbortError`] when the token aborts during or before the sleep.
pub async fn sleep(ms: u64, signal: &CancellationToken) -> Result<(), AbortError> {
    tokio::select! {
        () = signal.cancelled() => Err(AbortError),
        () = tokio::time::sleep(std::time::Duration::from_millis(ms)) => Ok(()),
    }
}

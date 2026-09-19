//! Operation-local abort plumbing, ported from
//! `packages/ai/src/utils/abort.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The AbortSignal surface ports to [`tokio_util::sync::CancellationToken`]
//! per the stack decision. Porting restatement: TypeScript lets a raced
//! operation keep running as an abandoned promise whose later rejection is
//! still observed; a dropped Rust future stops running, so the abandoned
//! operation is cancelled outright — the same user-visible stop, with no
//! orphaned work left behind.

use std::future::Future;

use tokio_util::sync::CancellationToken;

/// The abort reason a cancellation carries.
///
/// An `AbortSignal`'s `reason` field has no `CancellationToken` counterpart,
/// so every cancellation surfaces as the standard abort reason, upstream's
/// default `AbortError`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbortError;

impl AbortError {
    /// The message upstream's default abort reason carries.
    pub const MESSAGE: &'static str = "The operation was aborted";
    /// The error name upstream's abort reason carries.
    pub const NAME: &'static str = "AbortError";
}

impl std::fmt::Display for AbortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(Self::MESSAGE)
    }
}

impl std::error::Error for AbortError {}

/// Create an operation-local token for public APIs whose signal is optional:
/// the caller's token when supplied, a fresh uncancelled token otherwise.
#[must_use]
pub fn operation_signal(signal: Option<&CancellationToken>) -> CancellationToken {
    signal.cloned().unwrap_or_default()
}

/// Why a raced operation stopped: the abort, or the operation's own error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaceError<E> {
    /// The signal aborted before the operation settled.
    Aborted(AbortError),
    /// The operation itself failed.
    Operation(E),
}

impl<E: std::fmt::Display> std::fmt::Display for RaceError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Aborted(abort) => std::fmt::Display::fmt(abort, f),
            Self::Operation(error) => std::fmt::Display::fmt(error, f),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for RaceError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Aborted(abort) => Some(abort),
            Self::Operation(error) => Some(error),
        }
    }
}

/// Wait for an operation, stopping when its token aborts.
///
/// # Errors
/// Returns [`RaceError::Aborted`] when the signal aborts first — an already
/// aborted signal rejects without waiting — and [`RaceError::Operation`] with
/// the operation's own failure otherwise.
pub async fn race_with_abort_signal<T, E>(
    operation: impl Future<Output = Result<T, E>>,
    signal: &CancellationToken,
) -> Result<T, RaceError<E>> {
    // An already-aborted signal rejects without waiting, upstream's early
    // check before the race registers.
    if signal.is_cancelled() {
        return Err(RaceError::Aborted(AbortError));
    }
    tokio::select! {
        () = signal.cancelled() => Err(RaceError::Aborted(AbortError)),
        result = operation => result.map_err(RaceError::Operation),
    }
}

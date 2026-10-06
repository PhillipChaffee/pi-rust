//! Operation-local abort plumbing, upstream's `src/utils/abort.ts`.
//!
//! Upstream keeps a near-identical copy of this belt in the ai package;
//! the port reuses the ai crate's ported vocabulary ([`AbortError`] and
//! [`RaceError`]) and adds the optional-signal wrapper this file's
//! `raceWithAbortSignal` carries over it (upstream accepts `signal:
//! AbortSignal | undefined` and passes the operation through untouched
//! when it is absent).
//!
//! The AbortSignal surface ports to [`CancellationToken`]; see the ai
//! crate's abort module for the abandoned-operation restatement note.

pub use pi_ai::utils::abort::{AbortError, RaceError};

use std::future::Future;

use tokio_util::sync::CancellationToken;

/// Normalize an optional public signal without imposing a deadline: the
/// caller's token when supplied, a fresh uncancelled token otherwise.
#[must_use]
pub fn operation_signal(signal: Option<&CancellationToken>) -> CancellationToken {
    pi_ai::utils::abort::operation_signal(signal)
}

/// Stop waiting on abort while observing the abandoned operation through
/// settlement.
///
/// An absent signal waits the operation out: upstream's pass-through branch.
///
/// # Errors
/// [`RaceError::Aborted`] when the signal aborts first — an already aborted
/// signal rejects without waiting — and [`RaceError::Operation`] with the
/// operation's own failure otherwise.
pub async fn race_with_abort_signal<T, E>(
    operation: impl Future<Output = Result<T, E>>,
    signal: Option<&CancellationToken>,
) -> Result<T, RaceError<E>> {
    match signal {
        Some(signal) => pi_ai::utils::abort::race_with_abort_signal(operation, signal).await,
        None => operation.await.map_err(RaceError::Operation),
    }
}

//! The mutation line serializing one Session's read-modify-write jobs,
//! ported from upstream `src/harness/session/mutation-line.ts`.
//!
//! Upstream chains a promise tail; the port restates the line as a FIFO
//! mutex: [`MutationLine::acquire`] parks until the line is free and
//! re-checks the seal after the grant (a job queued when close sealed the
//! line still fails, upstream's queued-job check), and the guard's release
//! is the tail advance. [`MutationLine::seal`] latches the first error and
//! the close path drains by acquiring once.

use std::sync::{Arc, Mutex, PoisonError};

use crate::harness::session::types::SessionError;

/// The latched seal error plus the FIFO job line, upstream's `tail` promise
/// and `sealedError` slot.
#[derive(Debug, Default)]
struct LineInner {
    line: Arc<tokio::sync::Mutex<()>>,
    sealed: Mutex<Option<SessionError>>,
}

/// Serializes complete read-modify-write jobs for one Session, upstream's
/// `MutationLine`.
///
/// Clone the handle freely: every clone shares the same line and seal.
#[derive(Clone, Debug, Default)]
pub struct MutationLine {
    inner: Arc<LineInner>,
}

/// One acquired job's hold on the line; the barrier releases on drop.
#[derive(Debug)]
pub struct MutationLineGuard {
    _guard: tokio::sync::OwnedMutexGuard<()>,
}

impl MutationLine {
    /// A fresh, unsealed line.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The latched seal error, when close has sealed the line.
    #[must_use]
    pub fn sealed_error(&self) -> Option<SessionError> {
        self.inner
            .sealed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Latch the seal error; the first seal wins, upstream's `seal`.
    pub fn seal(&self, error: SessionError) {
        let mut sealed = self
            .inner
            .sealed
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if sealed.is_none() {
            *sealed = Some(error);
        }
    }

    /// Acquire the line for one job, upstream's `run`'s grant.
    ///
    /// # Errors
    /// The latched seal error, checked before parking (upstream's call-time
    /// check) and again after the grant (upstream's execution-time check for
    /// jobs queued before the seal).
    pub async fn acquire(&self) -> Result<MutationLineGuard, SessionError> {
        if let Some(error) = self.sealed_error() {
            return Err(error);
        }
        let guard = self.inner.line.clone().lock_owned().await;
        if let Some(error) = self.sealed_error() {
            return Err(error);
        }
        Ok(MutationLineGuard { _guard: guard })
    }

    /// Wait until the line is free, ignoring the seal, upstream's `seal`
    /// returning the tail: the close path drains the granted job without
    /// failing on its own latch.
    pub async fn drain(&self) {
        drop(self.inner.line.clone().lock_owned().await);
    }
}

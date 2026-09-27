//! The mutation line serializing one Session's read-modify-write jobs,
//! ported from upstream `src/harness/session/mutation-line.ts`.
//!
//! Upstream chains a promise tail; the port restates the line as a FIFO
//! mutex: [`MutationLine::run`] parks until the line is free and re-checks
//! the seal after the grant (a job queued when close sealed the line still
//! fails, upstream's queued-job check), [`MutationLine::acquire`] is the
//! grant-only form the explicit mutation barrier holds across `end`, and
//! [`MutationLine::seal`] latches the first error and returns the tail the
//! close path drains.

use std::future::Future;
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

    /// Latch the seal error without draining, the close path's synchronous
    /// prologue; the first seal wins.
    pub fn latch(&self, error: SessionError) {
        let mut sealed = self
            .inner
            .sealed
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if sealed.is_none() {
            *sealed = Some(error);
        }
    }

    /// Latch the seal error and return the tail, upstream's `seal`: the
    /// first seal wins, and awaiting the returned value drains the granted
    /// job. The future owns the line handle.
    pub fn seal(&self, error: SessionError) -> impl Future<Output = ()> {
        self.latch(error);
        let line = self.inner.line.clone();
        async move {
            drop(line.lock_owned().await);
        }
    }

    /// Wait until the line is free, ignoring the seal: the close path's
    /// drain after its own latch.
    pub async fn drain(&self) {
        drop(self.inner.line.clone().lock_owned().await);
    }

    /// Run one complete read-modify-write job on the line, upstream's `run`.
    ///
    /// # Errors
    /// The latched seal error, checked before parking (upstream's call-time
    /// check) and again after the grant (upstream's execution-time check for
    /// jobs queued before the seal); the operation's own error, which does
    /// not stop later jobs.
    pub async fn run<T, F, Fut>(&self, operation: F) -> Result<T, SessionError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, SessionError>>,
    {
        if let Some(error) = self.sealed_error() {
            return Err(error);
        }
        let guard = self.inner.line.clone().lock_owned().await;
        if let Some(error) = self.sealed_error() {
            return Err(error);
        }
        let outcome = operation().await;
        drop(guard);
        outcome
    }

    /// Acquire the line for one job, the grant-only form the explicit
    /// mutation barrier holds until `end`.
    ///
    /// # Errors
    /// The latched seal error, checked before parking and again after the
    /// grant.
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
}

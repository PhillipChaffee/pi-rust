//! The test-only storage decorator that deterministically parks admitted
//! commits, ported from upstream
//! `src/harness/session/testing/gating-storage.ts`.
//!
#![expect(
    clippy::significant_drop_tightening,
    reason = "the gate's queue and waiters drain under one lock; the sends must fire under it, upstream's splice-and-resolve"
)]
//!

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use pi_ai::types::BoxedFuture;
use tokio::sync::oneshot;

use crate::harness::context::Context;
use crate::harness::session::testing::storage_decorator::{StorageDecorator, storage_forwards};
use crate::harness::session::types::{
    CommitResult, Entry, EntryScan, EntryStructure, SessionError, SessionStats, Storage,
    StorageBranchScan, UsageRow, UsageScan,
};
use crate::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress, Write,
};

/// The settle-or-discard signal one parked commit's release carries,
/// upstream's `released` promise resolved by `release()` and rejected by
/// `drop(error)`.
type ReleaseSignal = oneshot::Sender<Result<(), SessionError>>;

/// One parked commit, upstream's `ParkedCommit`: the release signal the
/// gate feeds and the landing receiver the releaser waits on.
struct ParkedCommit {
    release: ReleaseSignal,
    landing_rx: oneshot::Receiver<Result<(), SessionError>>,
}

/// One pending `waitPending` waiter, upstream's `PendingWaiter`.
struct PendingWaiter {
    count: usize,
    tx: oneshot::Sender<Result<(), SessionError>>,
}

/// The queue and waiters one gate owns, upstream's `queue`/`waiters`.
#[derive(Default)]
struct GateState {
    queue: VecDeque<ParkedCommit>,
    waiters: Vec<PendingWaiter>,
}

/// The test-only gate parking admitted commits, upstream's `GatingStorage`.
pub struct GatingStorage {
    base: StorageDecorator,
    armed: AtomicBool,
    state: Mutex<GateState>,
    discarded: AtomicBool,
}

impl std::fmt::Debug for GatingStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatingStorage").finish_non_exhaustive()
    }
}

impl GatingStorage {
    /// A gate over the delegate, upstream's constructor.
    #[must_use]
    pub fn new(delegate: Arc<dyn Storage>) -> Self {
        Self {
            base: StorageDecorator::new(delegate),
            armed: AtomicBool::new(false),
            state: Mutex::new(GateState::default()),
            discarded: AtomicBool::new(false),
        }
    }

    /// Fixture setup bypasses gating until explicitly armed, upstream's
    /// `arm`.
    pub fn arm(&self) {
        self.armed.store(true, Ordering::Release);
    }

    /// The parked-commit count, upstream's `pending`.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .queue
            .len()
    }

    /// Wait until at least `count` commits are parked, upstream's
    /// `waitPending`.
    ///
    /// # Errors
    /// A `SessionError::CommitDiscarded` after [`Self::discard`], or a
    /// `SessionError::Message` (upstream's `RangeError`) for a non-positive
    /// count.
    pub async fn wait_pending(&self, count: usize) -> Result<(), SessionError> {
        if count < 1 {
            return Err(SessionError::Message(
                "Pending commit count must be a positive safe integer".to_owned(),
            ));
        }
        if self.discarded.load(Ordering::Acquire) {
            return Err(SessionError::CommitDiscarded(
                "storage discarded".to_owned(),
            ));
        }
        {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if state.queue.len() >= count {
                return Ok(());
            }
        }
        let (tx, rx) = oneshot::channel::<Result<(), SessionError>>();
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .waiters
            .push(PendingWaiter { count, tx });
        rx.await.unwrap_or_else(|_| {
            Err(SessionError::CommitDiscarded(
                "storage discarded".to_owned(),
            ))
        })
    }

    /// Release `count` commits in FIFO order and wait until each write
    /// lands, upstream's `next`.
    ///
    /// # Errors
    /// A `SessionError::CommitDiscarded` after [`Self::discard`], or a
    /// `SessionError::Message` (upstream's `RangeError`) for a non-positive
    /// count.
    pub async fn next(&self, count: usize) -> Result<(), SessionError> {
        if count < 1 {
            return Err(SessionError::Message(
                "Released commit count must be a positive safe integer".to_owned(),
            ));
        }
        for _ in 0..count {
            self.wait_pending(1).await?;
            let parked = {
                let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                state.queue.pop_front()
            };
            let Some(parked) = parked else {
                return Err(SessionError::Message("No parked commit".to_owned()));
            };
            let _ = parked.release.send(Ok(()));
            parked
                .landing_rx
                .await
                .unwrap_or_else(|_| Err(SessionError::Message("No parked commit".to_owned())))?;
        }
        Ok(())
    }

    /// Drop parked commits and permanently reject every later commit,
    /// upstream's `discard`.
    pub fn discard(&self) {
        if self.discarded.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        for parked in state.queue.drain(..) {
            let _ = parked.release.send(Err(SessionError::CommitDiscarded(
                "commit discarded".to_owned(),
            )));
        }
        for waiter in state.waiters.drain(..) {
            let _ = waiter.tx.send(Err(SessionError::CommitDiscarded(
                "commit discarded".to_owned(),
            )));
        }
    }

    fn notify_waiters(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        for index in (0..state.waiters.len()).rev() {
            if state.queue.len() < state.waiters[index].count {
                continue;
            }
            let waiter = state.waiters.remove(index);
            let _ = waiter.tx.send(Ok(()));
        }
    }
}

impl Storage for GatingStorage {
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        if self.discarded.load(Ordering::Acquire) {
            return Box::pin(std::future::ready(Err(SessionError::CommitDiscarded(
                "commit rejected: storage discarded".to_owned(),
            ))));
        }
        if !self.armed.load(Ordering::Acquire) {
            return self.base.commit_forward(writes, context);
        }
        let (release_tx, release_rx) = oneshot::channel::<Result<(), SessionError>>();
        let (landing_tx, landing_rx) = oneshot::channel::<Result<(), SessionError>>();
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .queue
            .push_back(ParkedCommit {
                release: release_tx,
                landing_rx,
            });
        self.notify_waiters();
        let context = context.clone();
        Box::pin(async move {
            if release_rx
                .await
                .unwrap_or_else(|_| {
                    Err(SessionError::CommitDiscarded("commit discarded".to_owned()))
                })
                .is_err()
            {
                let _ = landing_tx.send(Err(SessionError::CommitDiscarded(
                    "commit discarded".to_owned(),
                )));
                return Err(SessionError::CommitDiscarded(
                    "commit rejected: storage discarded".to_owned(),
                ));
            }
            if self.discarded.load(Ordering::Acquire) {
                let _ = landing_tx.send(Err(SessionError::CommitDiscarded(
                    "commit discarded".to_owned(),
                )));
                return Err(SessionError::CommitDiscarded(
                    "commit rejected: storage discarded".to_owned(),
                ));
            }
            let result = self.base.commit_forward(writes, &context).await;
            match result {
                Ok(result) => {
                    let _ = landing_tx.send(Ok(()));
                    Ok(result)
                }
                Err(error) => {
                    let _ = landing_tx.send(Err(error.clone()));
                    Err(error)
                }
            }
        })
    }

    storage_forwards!();
}

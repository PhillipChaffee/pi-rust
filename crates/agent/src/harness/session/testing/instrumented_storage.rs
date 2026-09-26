//! The test-only transparent Storage decorator that records commit
//! admission, ported from upstream
//! `src/harness/session/testing/instrumented-storage.ts`.

use std::sync::{Arc, Mutex, PoisonError};

use pi_ai::types::BoxedFuture;

use crate::harness::context::Context;
use crate::harness::session::testing::storage_decorator::{StorageDecorator, storage_forwards};
use crate::harness::session::types::{
    CommitResult, Entry, EntryScan, EntryStructure, SessionError, SessionStats, Storage,
    StorageBranchScan, UsageRow, UsageScan,
};
use crate::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress, Write,
};

/// The transparent decorator recording every commit's write set, upstream's
/// `InstrumentedStorage`.
pub struct InstrumentedStorage {
    base: StorageDecorator,
    commit_attempts: Mutex<Vec<Vec<Write>>>,
}

impl std::fmt::Debug for InstrumentedStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstrumentedStorage")
            .finish_non_exhaustive()
    }
}

impl InstrumentedStorage {
    /// A decorator over the delegate, upstream's constructor.
    #[must_use]
    pub fn new(delegate: Arc<dyn Storage>) -> Self {
        Self {
            base: StorageDecorator::new(delegate),
            commit_attempts: Mutex::new(Vec::new()),
        }
    }

    /// Every commit's write set in admission order, upstream's
    /// `getCommitAttempts`.
    #[must_use]
    pub fn get_commit_attempts(&self) -> Vec<Vec<Write>> {
        self.commit_attempts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Forget the recorded attempts, upstream's `clearCommitAttempts`.
    pub fn clear_commit_attempts(&self) {
        self.commit_attempts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }
}

impl Storage for InstrumentedStorage {
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        self.commit_attempts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(writes.clone());
        self.base.commit_forward(writes, context)
    }

    storage_forwards!();
}

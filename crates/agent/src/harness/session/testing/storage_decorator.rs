//! The test-only forwarding base for decorators that alter one part of
//! Storage behavior, ported from upstream
//! `src/harness/session/testing/storage-decorator.ts`.

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::types::BoxedFuture;

use crate::harness::context::Context;
use crate::harness::session::types::{
    CommitResult, Entry, EntryScan, EntryStructure, SessionError, SessionStats, Storage,
    StorageBranchScan, UsageRow, UsageScan,
};
use crate::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress, Write,
};

/// The forwarding base the decorators compose, upstream's
/// `StorageDecorator`.
pub struct StorageDecorator {
    delegate: Arc<dyn Storage>,
}

impl std::fmt::Debug for StorageDecorator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageDecorator").finish_non_exhaustive()
    }
}

impl StorageDecorator {
    /// A decorator over the delegate.
    #[must_use]
    pub fn new(delegate: Arc<dyn Storage>) -> Self {
        Self { delegate }
    }

    /// The delegate the decorator forwards to.
    #[must_use]
    pub fn delegate(&self) -> &Arc<dyn Storage> {
        &self.delegate
    }

    /// The forwarded commit, upstream's base `commit`.
    #[must_use]
    pub fn commit_forward(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        self.delegate.commit(writes, context)
    }

    /// The forwarded named-entry read, upstream's base `getEntries`.
    #[must_use]
    pub fn get_entries_forward(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<BTreeMap<String, Entry>, SessionError>> {
        self.delegate.get_entries(ids, context)
    }

    /// The forwarded value read, upstream's base `getValue`.
    #[must_use]
    pub fn get_value_forward(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        self.delegate.get_value(address, context)
    }

    /// The forwarded value scan, upstream's base `scanValues`.
    #[must_use]
    pub fn scan_values_forward(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        self.delegate.scan_values(prefix, context)
    }

    /// The forwarded list read, upstream's base `readList`.
    #[must_use]
    pub fn read_list_forward(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        self.delegate.read_list(address, options, context)
    }

    /// The forwarded branch scan, upstream's base `scanBranch`.
    #[must_use]
    pub fn scan_branch_forward(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.delegate.scan_branch(query, context)
    }

    /// The forwarded structural branch scan, upstream's base
    /// `scanBranchStructure`.
    #[must_use]
    pub fn scan_branch_structure_forward(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<EntryStructure>, SessionError>> {
        self.delegate.scan_branch_structure(query, context)
    }

    /// The forwarded flat entry scan, upstream's base `scanEntries`.
    #[must_use]
    pub fn scan_entries_forward(
        &self,
        query: &EntryScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.delegate.scan_entries(query, context)
    }

    /// The forwarded usage scan, upstream's base `scanUsage`.
    #[must_use]
    pub fn scan_usage_forward(
        &self,
        query: &UsageScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<UsageRow>, SessionError>> {
        self.delegate.scan_usage(query, context)
    }

    /// The forwarded totals read, upstream's base `getStats`.
    #[must_use]
    pub fn get_stats_forward(
        &self,
        context: &Context,
    ) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        self.delegate.get_stats(context)
    }

    /// The forwarded close, upstream's base `close`.
    #[must_use]
    pub fn close_forward(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        self.delegate.close(context)
    }
}

/// The read-side Storage forwards the decorators share, upstream's class
/// inheritance: each decorator embeds a [`StorageDecorator`] and expands
/// this inside its `impl Storage`, overriding only the methods it alters.
macro_rules! storage_forwards {
    () => {
        fn get_entries(
            &self,
            ids: Vec<String>,
            context: &Context,
        ) -> BoxedFuture<'_, Result<std::collections::BTreeMap<String, Entry>, SessionError>> {
            self.base.get_entries_forward(ids, context)
        }

        fn get_value(
            &self,
            address: &ValueAddress,
            context: &Context,
        ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
            self.base.get_value_forward(address, context)
        }

        fn scan_values(
            &self,
            prefix: &ValueAddress,
            context: &Context,
        ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
            self.base.scan_values_forward(prefix, context)
        }

        fn read_list(
            &self,
            address: &ListAddress,
            options: Option<ListReadOptions>,
            context: &Context,
        ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
            self.base.read_list_forward(address, options, context)
        }

        fn scan_branch(
            &self,
            query: &StorageBranchScan,
            context: &Context,
        ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
            self.base.scan_branch_forward(query, context)
        }

        fn scan_branch_structure(
            &self,
            query: &StorageBranchScan,
            context: &Context,
        ) -> BoxedFuture<'_, Result<Vec<EntryStructure>, SessionError>> {
            self.base.scan_branch_structure_forward(query, context)
        }

        fn scan_entries(
            &self,
            query: &EntryScan,
            context: &Context,
        ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
            self.base.scan_entries_forward(query, context)
        }

        fn scan_usage(
            &self,
            query: &UsageScan,
            context: &Context,
        ) -> BoxedFuture<'_, Result<Vec<UsageRow>, SessionError>> {
            self.base.scan_usage_forward(query, context)
        }

        fn get_stats(
            &self,
            context: &Context,
        ) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
            self.base.get_stats_forward(context)
        }

        fn close(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
            self.base.close_forward(context)
        }
    };
}

pub(crate) use storage_forwards;

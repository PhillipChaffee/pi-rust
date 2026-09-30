//! The `SqliteStorage` conformance suite, ported 1:1 from upstream
//! `test/storage-conformance.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: `createStorageConformance`'s
//! 21 runner-independent Storage cases over the `:memory:` fixture.
//!
//! Fixture restatements, upstream storage-conformance.test.ts:34-61: the
//! database opens synchronously (the port's driver seam is sync — recorded
//! adapter delta); upstream's factory `catch` closes the database before
//! rethrowing a construction failure, a path the port's fixture carries as
//! the seed helpers' `expect` panics (nothing constructed closes on that
//! path); and the dispose closes the storage then the database, upstream's
//! `try { await storage.close(BACKGROUND_CONTEXT); } finally { db.close(); }`,
//! carried by [`FixtureDisposeStorage`] because the harness's
//! [`StorageFixture::dispose`] closes the storage alone.
//!
//! Registration restatement: upstream nests vitest `describe(group)`/
//! `it(name)`; the port registers one test per case through the
//! `storage_case!` macro over the suite's shared runner, a different shape
//! from the agent crate's registration file (upstream itself duplicates the
//! tiny runner per backend test file).

#![expect(
    clippy::expect_used,
    reason = "the fixture opens the database and applies the schema; a construction failure panics the case by design"
)]

mod support;

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_agent_core::harness::context::Context;
use pi_agent_core::harness::session::testing::conformance::storage::StorageFixtureFactory;
use pi_agent_core::harness::session::testing::{
    ConformanceCase, StorageFixture, create_storage_conformance,
};
use pi_agent_core::harness::session::types::{
    CommitResult, Entry, EntryScan, EntryStructure, SessionError, SessionStats, Storage,
    StorageBranchScan, UsageRow, UsageScan,
};
use pi_agent_core::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress, Write,
};
use pi_agent_core::types::BoxedFuture;
use pi_session_backend_sqlite_node::sqlite::types::SqliteDatabase;
use pi_session_backend_sqlite_node::{
    SqliteStorage, SqliteStorageOptions, apply_initial_schema, create_rusqlite_factory,
};

use support::{NOW, fixed_clock, insert_conformance_session_row};

/// The seeded session id, upstream's `SESSION_ID`.
const SESSION_ID: &str = "session";

/// The fixture's storage handle: every call forwards, and `close` releases
/// the storage then the database — upstream's fixture dispose
/// `try { await storage.close(BACKGROUND_CONTEXT); } finally { db.close(); }`.
/// The harness's [`StorageFixture::dispose`] closes the storage only, so the
/// database close rides the storage's close here; the finally block's throw
/// replaces the outcome, so a database-close failure supersedes the
/// storage's result.
struct FixtureDisposeStorage {
    storage: Arc<dyn Storage>,
    db: Arc<dyn SqliteDatabase>,
}

impl Storage for FixtureDisposeStorage {
    fn commit(
        &self,
        writes: Vec<Write>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<CommitResult, SessionError>> {
        self.storage.commit(writes, context)
    }

    fn get_entries(
        &self,
        ids: Vec<String>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<BTreeMap<String, Entry>, SessionError>> {
        self.storage.get_entries(ids, context)
    }

    fn get_value(
        &self,
        address: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        self.storage.get_value(address, context)
    }

    fn scan_values(
        &self,
        prefix: &ValueAddress,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        self.storage.scan_values(prefix, context)
    }

    fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        self.storage.read_list(address, options, context)
    }

    fn scan_branch(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.storage.scan_branch(query, context)
    }

    fn scan_branch_structure(
        &self,
        query: &StorageBranchScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<EntryStructure>, SessionError>> {
        self.storage.scan_branch_structure(query, context)
    }

    fn scan_entries(
        &self,
        query: &EntryScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        self.storage.scan_entries(query, context)
    }

    fn scan_usage(
        &self,
        query: &UsageScan,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<UsageRow>, SessionError>> {
        self.storage.scan_usage(query, context)
    }

    fn get_stats(&self, context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        self.storage.get_stats(context)
    }

    fn close(&self, context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        let storage = Arc::clone(&self.storage);
        let db = Arc::clone(&self.db);
        let context = context.clone();
        Box::pin(async move {
            let storage_result = storage.close(&context).await;
            db.close()
                .map_err(|error| SessionError::Message(error.to_string()))?;
            storage_result
        })
    }
}

/// The fixture factory, upstream's storage-conformance.test.ts:36-53: one
/// `:memory:` container under the initial schema with the seeded `sessions`
/// row (`created_at` NOW, the crate's storage version, `next_seq` 1), and
/// the storage clocked at NOW.
fn storage_fixture_factory() -> StorageFixtureFactory {
    Arc::new(|| {
        let db: Arc<dyn SqliteDatabase> = Arc::from(
            create_rusqlite_factory()
                .open(":memory:")
                .expect("fixture :memory: open"),
        );
        apply_initial_schema(db.as_ref()).expect("fixture initial schema");
        insert_conformance_session_row(db.as_ref());
        let storage = SqliteStorage::new(
            Arc::clone(&db),
            &SqliteStorageOptions {
                session_id: SESSION_ID.to_owned(),
                now: Some(fixed_clock(NOW)),
            },
        );
        Box::pin(async move {
            StorageFixture::new(Arc::new(FixtureDisposeStorage {
                storage: Arc::new(storage),
                db,
            }))
        })
    })
}

fn storage_conformance_cases() -> Vec<ConformanceCase> {
    create_storage_conformance(&storage_fixture_factory())
}

/// Runs one registered case by its group and name, upstream's
/// `it(testCase.name, () => testCase.run())` under the group's `describe`.
macro_rules! storage_case {
    ($case_test:ident, $group:literal, $case_name:literal) => {
        #[tokio::test]
        async fn $case_test() {
            support::run_conformance_case(&storage_conformance_cases(), $group, $case_name).await;
        }
    };
}

storage_case!(
    storage_commits_mixed_writes_atomically_in_write_order,
    "transactions",
    "commits mixed writes atomically in write order"
);
storage_case!(
    storage_rolls_back_every_store_when_a_mixed_transaction_fails,
    "transactions",
    "rolls back every store when a mixed transaction fails"
);
storage_case!(
    storage_preserves_overwritten_and_deleted_values_when_a_transaction_fails,
    "transactions",
    "preserves overwritten and deleted values when a transaction fails"
);
storage_case!(
    storage_enforces_one_shared_entry_and_usage_id_namespace,
    "transactions",
    "enforces one shared entry and usage id namespace"
);
storage_case!(
    storage_resolves_parents_only_from_prior_entries_and_earlier_writes,
    "transactions",
    "resolves parents only from prior entries and earlier writes"
);
storage_case!(
    storage_places_pending_content_under_its_reserved_entry_id,
    "transactions",
    "places pending content under its reserved entry id"
);
storage_case!(
    storage_sets_replaces_deletes_and_recreates_values_without_tombstones,
    "values",
    "sets, replaces, deletes, and recreates values without tombstones"
);
storage_case!(
    storage_applies_same_transaction_value_and_list_operations_in_write_order,
    "values",
    "applies same-transaction value and list operations in write order"
);
storage_case!(
    storage_does_not_change_historical_stores_during_value_only_commits,
    "values",
    "does not change historical stores during value-only commits"
);
storage_case!(
    storage_pages_appends_by_global_sequence_and_deletes_whole_lists,
    "lists",
    "pages appends by global sequence and deletes whole lists"
);
storage_case!(
    storage_clamps_one_read_page_without_limiting_list_growth,
    "lists",
    "clamps one read page without limiting list growth"
);
storage_case!(
    storage_commits_mixed_list_writes_atomically_and_rolls_them_back_with_siblings,
    "lists",
    "commits mixed list writes atomically and rolls them back with siblings"
);
storage_case!(
    storage_stores_custom_entries_with_and_without_data,
    "entry queries",
    "stores custom entries with and without data"
);
storage_case!(
    storage_scans_global_entries_with_explicit_ranges_filters_orders_and_limits,
    "entry queries",
    "scans global entries with explicit ranges, filters, orders, and limits"
);
storage_case!(
    storage_applies_stops_before_filters_and_cursors_before_limits,
    "branch queries",
    "applies stops before filters and cursors before limits"
);
storage_case!(
    storage_returns_branch_structure_without_payload_fields,
    "branch queries",
    "returns branch structure without payload fields"
);
storage_case!(
    storage_applies_branch_query_semantics_to_structure_scans,
    "branch queries",
    "applies branch query semantics to structure scans"
);
storage_case!(
    storage_scans_the_usage_ledger_with_explicit_ranges_orders_and_limits,
    "usage and stats",
    "scans the usage ledger with explicit ranges, orders, and limits"
);
storage_case!(
    storage_keeps_stats_equal_to_message_count_and_ledger_totals,
    "usage and stats",
    "keeps stats equal to message count and ledger totals"
);
storage_case!(
    storage_serializes_back_to_back_commits_in_admission_order,
    "serialization",
    "serializes back-to-back commits in admission order"
);
storage_case!(
    storage_seals_admission_drains_admitted_commits_and_closes_idempotently,
    "lifecycle",
    "seals admission, drains admitted commits, and closes idempotently"
);
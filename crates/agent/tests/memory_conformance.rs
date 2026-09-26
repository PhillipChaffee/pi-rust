//! The memory backend's conformance suite, ported 1:1 from upstream
//! `test/harness/memory-conformance.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the runner-independent
//! Storage and `SessionRepo` conformance cases registered against
//! `MemoryStorage`/`MemorySessionRepo`, plus the reopen commit-statistics
//! case.
//!
//! Upstream registers the cases through vitest's `describe`/`it`; the port
//! restates the registration as one test per case driven by
//! [`run_case`], which panics when the named case is missing or fails.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod session_common;
use session_common::*;

use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::memory::{
    MemorySessionRepo, MemorySessionRepoOptions, MemoryStorage, MemoryStorageOptions,
};
use pi_agent_core::harness::session::session::StorageBackedSession;
use pi_agent_core::harness::session::testing::{
    ConformanceCase, RepoFixture, StorageFixture, create_session_repo_conformance,
    create_session_repo_streaming_fork_conformance, create_storage_conformance,
};
use pi_agent_core::harness::session::types::SessionRepo;
use pi_agent_core::harness::session::{commit as session_writes, values as stored_values};

fn storage_fixture_factory()
-> Arc<dyn Fn() -> pi_ai::types::BoxedFuture<'static, StorageFixture> + Send + Sync> {
    Arc::new(|| {
        let storage = MemoryStorage::new(MemoryStorageOptions {
            now: Some(fixed_clock(NOW)),
        });
        Box::pin(async move { StorageFixture::new(Arc::new(storage)) })
    })
}

fn storage_conformance_cases() -> Vec<ConformanceCase> {
    create_storage_conformance(&storage_fixture_factory())
}

fn repo_fixture_factory()
-> Arc<dyn Fn() -> pi_ai::types::BoxedFuture<'static, RepoFixture> + Send + Sync> {
    Arc::new(|| {
        let repo = Arc::new(MemorySessionRepo::new(MemorySessionRepoOptions {
            now: Some(fixed_clock(NOW)),
        }));
        let closer = repo.clone();
        Box::pin(async move {
            RepoFixture::new(
                repo,
                Some(Arc::new(move || {
                    let closer = closer.clone();
                    Box::pin(async move {
                        let _ = closer.close(&background_context()).await;
                    })
                })),
            )
        })
    })
}

fn repo_conformance_cases() -> Vec<ConformanceCase> {
    [
        create_session_repo_conformance(&repo_fixture_factory()),
        create_session_repo_streaming_fork_conformance(&repo_fixture_factory()),
    ]
    .concat()
}

/// Runs one registered case by its group and name, upstream's vitest
/// registration.
///
/// # Panics
/// When the named case does not exist or the case fails.
async fn run_case(cases: &[ConformanceCase], group: &str, name: &str) {
    let case = cases
        .iter()
        .find(|case| case.group == group && case.name == name)
        .unwrap_or_else(|| panic!("missing conformance case {group}/{name}"));
    case.run().await;
}

/// One test per registered Storage case, upstream's `registerConformance`.
macro_rules! storage_case_test {
    ($test_name:ident, $group:literal, $case_name:literal) => {
        #[tokio::test]
        async fn $test_name() {
            run_case(&storage_conformance_cases(), $group, $case_name).await;
        }
    };
}

/// One test per registered `SessionRepo` case.
macro_rules! repo_case_test {
    ($test_name:ident, $group:literal, $case_name:literal) => {
        #[tokio::test]
        async fn $test_name() {
            run_case(&repo_conformance_cases(), $group, $case_name).await;
        }
    };
}

storage_case_test!(
    storage_commits_mixed_writes_atomically_in_write_order,
    "transactions",
    "commits mixed writes atomically in write order"
);
storage_case_test!(
    storage_rolls_back_every_store_when_a_mixed_transaction_fails,
    "transactions",
    "rolls back every store when a mixed transaction fails"
);
storage_case_test!(
    storage_preserves_overwritten_and_deleted_values_when_a_transaction_fails,
    "transactions",
    "preserves overwritten and deleted values when a transaction fails"
);
storage_case_test!(
    storage_enforces_one_shared_entry_and_usage_id_namespace,
    "transactions",
    "enforces one shared entry and usage id namespace"
);
storage_case_test!(
    storage_resolves_parents_only_from_prior_entries_and_earlier_writes,
    "transactions",
    "resolves parents only from prior entries and earlier writes"
);
storage_case_test!(
    storage_places_pending_content_under_its_reserved_entry_id,
    "transactions",
    "places pending content under its reserved entry id"
);
storage_case_test!(
    storage_sets_replaces_deletes_and_recreates_values_without_tombstones,
    "values",
    "sets, replaces, deletes, and recreates values without tombstones"
);
storage_case_test!(
    storage_applies_same_transaction_value_and_list_operations_in_write_order,
    "values",
    "applies same-transaction value and list operations in write order"
);
storage_case_test!(
    storage_does_not_change_historical_stores_during_value_only_commits,
    "values",
    "does not change historical stores during value-only commits"
);
storage_case_test!(
    storage_pages_appends_by_global_sequence_and_deletes_whole_lists,
    "lists",
    "pages appends by global sequence and deletes whole lists"
);
storage_case_test!(
    storage_clamps_one_read_page_without_limiting_list_growth,
    "lists",
    "clamps one read page without limiting list growth"
);
storage_case_test!(
    storage_commits_mixed_list_writes_atomically_and_rolls_them_back_with_siblings,
    "lists",
    "commits mixed list writes atomically and rolls them back with siblings"
);
storage_case_test!(
    storage_stores_custom_entries_with_and_without_data,
    "entry queries",
    "stores custom entries with and without data"
);
storage_case_test!(
    storage_scans_global_entries_with_explicit_ranges_filters_orders_and_limits,
    "entry queries",
    "scans global entries with explicit ranges, filters, orders, and limits"
);
storage_case_test!(
    storage_applies_stops_before_filters_and_cursors_before_limits,
    "branch queries",
    "applies stops before filters and cursors before limits"
);
storage_case_test!(
    storage_returns_branch_structure_without_payload_fields,
    "branch queries",
    "returns branch structure without payload fields"
);
storage_case_test!(
    storage_applies_branch_query_semantics_to_structure_scans,
    "branch queries",
    "applies branch query semantics to structure scans"
);
storage_case_test!(
    storage_scans_the_usage_ledger_with_explicit_ranges_orders_and_limits,
    "usage and stats",
    "scans the usage ledger with explicit ranges, orders, and limits"
);
storage_case_test!(
    storage_keeps_stats_equal_to_message_count_and_ledger_totals,
    "usage and stats",
    "keeps stats equal to message count and ledger totals"
);
storage_case_test!(
    storage_serializes_back_to_back_commits_in_admission_order,
    "serialization",
    "serializes back-to-back commits in admission order"
);
storage_case_test!(
    storage_seals_admission_drains_admitted_commits_and_closes_idempotently,
    "lifecycle",
    "seals admission, drains admitted commits, and closes idempotently"
);

repo_case_test!(
    repo_creates_a_session_with_no_implicit_branch_and_rejects_duplicate_ids,
    "lifecycle",
    "creates a session with no implicit branch and rejects duplicate ids"
);
repo_case_test!(
    repo_close_drains_an_acquired_scope_and_rejects_a_queued_mutation_callback,
    "lifecycle",
    "close drains an acquired scope and rejects a queued mutation callback"
);
repo_case_test!(
    repo_lists_metadata_and_preserves_state_across_close_and_reopen,
    "lifecycle",
    "lists metadata and preserves state across close and reopen"
);
repo_case_test!(
    repo_deletes_closed_sessions_without_affecting_other_sessions,
    "lifecycle",
    "deletes closed sessions without affecting other sessions"
);
repo_case_test!(
    repo_rejects_opening_an_already_open_session,
    "ownership",
    "rejects opening an already-open session"
);
repo_case_test!(
    repo_rejects_pending_assistant_messages_without_changing_the_tree,
    "messages",
    "rejects pending assistant messages without changing the tree"
);
repo_case_test!(
    repo_preserves_every_settled_assistant_stop_reason,
    "messages",
    "preserves every settled assistant stop reason"
);
repo_case_test!(
    repo_tree_forks_a_fresh_session_before_first_attachment,
    "forks",
    "tree-forks a fresh session before first attachment"
);
repo_case_test!(
    repo_rejects_a_data_only_branch_and_releases_its_destination_id,
    "forks",
    "rejects a data-only branch and releases its destination id"
);
repo_case_test!(
    repo_forks_one_named_configured_branch_with_scoped_values_and_a_zero_ledger,
    "forks",
    "forks one named configured branch with scoped values and a zero ledger"
);
repo_case_test!(
    repo_enforces_branch_ancestry_for_at_and_before_placement,
    "forks",
    "enforces branch ancestry for at and before placement"
);
repo_case_test!(
    repo_forks_a_closed_source_session,
    "forks",
    "forks a closed source session"
);
repo_case_test!(
    repo_forks_the_whole_configured_tree_with_fresh_lane_state,
    "forks",
    "forks the whole configured tree with fresh lane state"
);
repo_case_test!(
    repo_rejects_only_surviving_unknown_reserved_scalar_state,
    "forks",
    "rejects only surviving unknown reserved scalar state"
);
repo_case_test!(
    repo_publishes_create_when_it_reserves_a_shared_destination_id_first,
    "fork coordination",
    "publishes create when it reserves a shared destination id first"
);
repo_case_test!(
    repo_publishes_fork_when_it_reserves_a_shared_destination_id_first,
    "fork coordination",
    "publishes fork when it reserves a shared destination id first"
);
repo_case_test!(
    repo_captures_one_coherent_boundary_between_source_commits,
    "fork coordination",
    "captures one coherent boundary between source commits"
);
repo_case_test!(
    repo_fork_application_lists_open_source_copies_lists_at_distinct_addresses,
    "fork application lists (open source)",
    "tree fork copies lists at distinct addresses"
);
repo_case_test!(
    repo_fork_application_lists_open_source_copies_only_survivors_after_list_deletion_and_reappend,
    "fork application lists (open source)",
    "tree fork copies only survivors after list deletion and reappend"
);
repo_case_test!(
    repo_fork_application_lists_open_source_preserves_list_element_sequences_including_gaps,
    "fork application lists (open source)",
    "tree fork preserves list element sequences including gaps"
);
repo_case_test!(
    repo_fork_application_lists_open_source_continues_asc_pagination_using_source_cursors,
    "fork application lists (open source)",
    "tree fork continues asc pagination using source cursors"
);
repo_case_test!(
    repo_fork_application_lists_open_source_continues_desc_pagination_using_source_cursors,
    "fork application lists (open source)",
    "tree fork continues desc pagination using source cursors"
);
repo_case_test!(
    repo_fork_application_lists_closed_source_copies_lists_at_distinct_addresses,
    "fork application lists (closed source)",
    "tree fork copies lists at distinct addresses"
);
repo_case_test!(
    repo_fork_application_lists_closed_source_copies_only_survivors_after_list_deletion_and_reappend,
    "fork application lists (closed source)",
    "tree fork copies only survivors after list deletion and reappend"
);
repo_case_test!(
    repo_fork_application_lists_closed_source_preserves_list_element_sequences_including_gaps,
    "fork application lists (closed source)",
    "tree fork preserves list element sequences including gaps"
);
repo_case_test!(
    repo_fork_application_lists_closed_source_continues_asc_pagination_using_source_cursors,
    "fork application lists (closed source)",
    "tree fork continues asc pagination using source cursors"
);
repo_case_test!(
    repo_fork_application_lists_closed_source_continues_desc_pagination_using_source_cursors,
    "fork application lists (closed source)",
    "tree fork continues desc pagination using source cursors"
);
repo_case_test!(
    repo_branch_fork_application_state_open_source_excludes_overwritten_and_unchanged_application_values,
    "branch fork application state (open source)",
    "excludes overwritten and unchanged application values"
);
repo_case_test!(
    repo_branch_fork_application_state_open_source_excludes_deleted_reappended_and_untouched_application_lists,
    "branch fork application state (open source)",
    "excludes deleted/reappended and untouched application lists"
);
repo_case_test!(
    repo_branch_fork_application_state_closed_source_excludes_overwritten_and_unchanged_application_values,
    "branch fork application state (closed source)",
    "excludes overwritten and unchanged application values"
);
repo_case_test!(
    repo_branch_fork_application_state_closed_source_excludes_deleted_reappended_and_untouched_application_lists,
    "branch fork application state (closed source)",
    "excludes deleted/reappended and untouched application lists"
);
repo_case_test!(
    repo_fork_lane_validation_ignores_malformed_unrelated_lanes,
    "fork lane validation",
    "ignores malformed unrelated lanes"
);

#[tokio::test]
async fn includes_historical_totals_in_the_first_commit_after_session_reopen() {
    let repo = MemorySessionRepo::new(MemorySessionRepoOptions {
        now: Some(fixed_clock(NOW)),
    });
    let session = repo
        .create(
            pi_agent_core::harness::session::types::SessionCreateOptions::default(),
            &background_context(),
        )
        .await
        .expect("create");
    let usage = pi_ai::types::Usage {
        input: 1,
        output: 2,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 3,
        cost: pi_ai::types::UsageCost::default(),
    };
    session
        .mutate(
            StorageBackedSession::commit_writes_callback(vec![
                stored_values::Write::Entry(Box::new(session_writes::insert_entry(
                    pi_agent_core::harness::session::types::NewEntry::Message {
                        id: "history".to_owned(),
                        parent_id: None,
                        body: Box::new(pi_agent_core::harness::session::types::MessageEntry {
                            message: user_message("history"),
                            terminate: None,
                        }),
                    },
                ))),
                stored_values::Write::Usage(session_writes::insert_usage(
                    pi_agent_core::harness::session::types::UsageWriteRow {
                        id: "usage".to_owned(),
                        usage,
                        entry_id: None,
                        adjustment: false,
                        details: None,
                    },
                )),
            ]),
            &background_context(),
        )
        .await
        .expect("seed commit");
    session.close(&background_context()).await.expect("close");
    let reopened = repo
        .open(session.metadata(), &background_context())
        .await
        .expect("reopen");
    let result = reopened
        .mutate(
            StorageBackedSession::commit_writes_callback(vec![stored_values::Write::ValueSet(
                stored_values::set_value(&stored_values::session_name(), "reopened".to_owned())
                    .expect("write"),
            )]),
            &background_context(),
        )
        .await
        .expect("reopen commit");
    let result =
        pi_agent_core::harness::session::testing::conformance::downcast_commit_result(result);
    assert_eq!(
        result.stats,
        pi_agent_core::harness::session::types::SessionStats {
            message_count: 1,
            usage,
        },
    );
    assert_eq!(
        result.stats,
        reopened
            .get_stats(&background_context())
            .await
            .expect("stats"),
    );
    reopened.close(&background_context()).await.expect("close");
    repo.close(&background_context()).await.expect("close repo");
}

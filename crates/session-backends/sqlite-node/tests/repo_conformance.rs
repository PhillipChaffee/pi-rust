//! The `SqliteSessionRepo` conformance suites, ported 1:1 from upstream
//! `test/repo-conformance.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`:
//! `createSessionRepoConformance`'s 17 cases over both layouts — one
//! `{id}.sqlite` container per session, and one shared `sessions.sqlite`
//! container per repository.
//!
//! The streaming-fork cases stay out of both registrations, upstream's WP08
//! exclusion (`createSessionRepoStreamingForkConformance` is not part of
//! `createSessionRepoConformance`) carried as the same split.
//!
//! Fixture restatement, upstream repo-conformance.test.ts:28-57: each
//! factory builds a fresh temp directory per case and the cleanup removes
//! it; the port's close hook closes the repository first (upstream's
//! cleanup only removes the directory — the port closes the repo there too
//! so no connection outlives the case) and the temp directory's drop is
//! that removal.
//!
//! Registration restatement: upstream nests vitest `describe`/`it` twice
//! (`SqliteSessionRepo conformance` and `SqliteSessionRepo shared-container
//! conformance`); the port registers one test per case through the
//! `repo_case!` and `shared_repo_case!` macros over the suite's shared
//! runner, a different shape from the agent crate's registration file
//! (upstream itself duplicates the tiny runner per backend test file).

#![expect(
    clippy::expect_used,
    reason = "the fixture builds the temp directory and repo; a construction failure panics the case by design"
)]

mod support;

use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::testing::conformance::session_repo::RepoFixtureFactory;
use pi_agent_core::harness::session::testing::{
    ConformanceCase, RepoFixture, create_session_repo_conformance,
};
use pi_session_backend_sqlite_node::{SqliteSessionRepo, SqliteSessionRepoOptions};

use support::{NOW, fixed_clock};

/// Builds one layout's fixture factory: the repo over a fresh temp directory
/// per case, its close hook closing the repo and then dropping the
/// directory, upstream's
/// `createConformanceRepo`/`createSharedContainerConformanceRepo` plus the
/// `cleanupConformanceRepo`/`cleanupSharedContainerConformanceRepo` rm.
fn fixture_factory_of(
    options_for: impl Fn(&str) -> SqliteSessionRepoOptions + Send + Sync + 'static,
) -> RepoFixtureFactory {
    Arc::new(move || {
        let directory = tempfile::tempdir().expect("fixture temp directory");
        let directory_path = directory.path().to_str().expect("utf8 temp directory");
        let options = options_for(directory_path);
        let repo = Arc::new(SqliteSessionRepo::new(options));
        let closer = Arc::clone(&repo);
        let directory = Arc::new(directory);
        Box::pin(async move {
            RepoFixture::new(
                repo,
                Some(Arc::new(move || {
                    let closer = Arc::clone(&closer);
                    let directory = Arc::clone(&directory);
                    Box::pin(async move {
                        // The port delta vs upstream's directory-only cleanup:
                        // the repo closes here first.
                        let _ = closer.close(&background_context()).await;
                        drop(directory);
                    })
                })),
            )
        })
    })
}

fn per_file_repo_fixture_factory() -> RepoFixtureFactory {
    fixture_factory_of(|directory| support::repo_options(directory, fixed_clock(NOW)))
}

fn shared_container_repo_fixture_factory() -> RepoFixtureFactory {
    fixture_factory_of(|directory| {
        support::shared_repo_options(
            directory,
            &format!("{directory}/sessions.sqlite"),
            fixed_clock(NOW),
        )
    })
}

fn repo_conformance_cases() -> Vec<ConformanceCase> {
    create_session_repo_conformance(&per_file_repo_fixture_factory())
}

fn shared_container_repo_conformance_cases() -> Vec<ConformanceCase> {
    create_session_repo_conformance(&shared_container_repo_fixture_factory())
}

/// Runs one per-file-layout case by its group and name, upstream's
/// `it(testCase.name, () => testCase.run())` under the layout's `describe`.
macro_rules! repo_case {
    ($test_name:ident, $group:literal, $case_name:literal) => {
        #[tokio::test]
        async fn $test_name() {
            support::run_conformance_case(&repo_conformance_cases(), $group, $case_name).await;
        }
    };
}

/// Runs one shared-container-layout case by its group and name.
macro_rules! shared_repo_case {
    ($test_name:ident, $group:literal, $case_name:literal) => {
        #[tokio::test]
        async fn $test_name() {
            support::run_conformance_case(
                &shared_container_repo_conformance_cases(),
                $group,
                $case_name,
            )
            .await;
        }
    };
}

repo_case!(
    repo_creates_a_session_with_no_implicit_branch_and_rejects_duplicate_ids,
    "lifecycle",
    "creates a session with no implicit branch and rejects duplicate ids"
);
repo_case!(
    repo_close_drains_an_acquired_scope_and_rejects_a_queued_mutation_callback,
    "lifecycle",
    "close drains an acquired scope and rejects a queued mutation callback"
);
repo_case!(
    repo_lists_metadata_and_preserves_state_across_close_and_reopen,
    "lifecycle",
    "lists metadata and preserves state across close and reopen"
);
repo_case!(
    repo_deletes_closed_sessions_without_affecting_other_sessions,
    "lifecycle",
    "deletes closed sessions without affecting other sessions"
);
repo_case!(
    repo_rejects_opening_an_already_open_session,
    "ownership",
    "rejects opening an already-open session"
);
repo_case!(
    repo_rejects_pending_assistant_messages_without_changing_the_tree,
    "messages",
    "rejects pending assistant messages without changing the tree"
);
repo_case!(
    repo_preserves_every_settled_assistant_stop_reason,
    "messages",
    "preserves every settled assistant stop reason"
);
repo_case!(
    repo_tree_forks_a_fresh_session_before_first_attachment,
    "forks",
    "tree-forks a fresh session before first attachment"
);
repo_case!(
    repo_rejects_a_data_only_branch_and_releases_its_destination_id,
    "forks",
    "rejects a data-only branch and releases its destination id"
);
repo_case!(
    repo_forks_one_named_configured_branch_with_scoped_values_and_a_zero_ledger,
    "forks",
    "forks one named configured branch with scoped values and a zero ledger"
);
repo_case!(
    repo_enforces_branch_ancestry_for_at_and_before_placement,
    "forks",
    "enforces branch ancestry for at and before placement"
);
repo_case!(
    repo_forks_a_closed_source_session,
    "forks",
    "forks a closed source session"
);
repo_case!(
    repo_forks_the_whole_configured_tree_with_fresh_lane_state,
    "forks",
    "forks the whole configured tree with fresh lane state"
);
repo_case!(
    repo_rejects_only_surviving_unknown_reserved_scalar_state,
    "forks",
    "rejects only surviving unknown reserved scalar state"
);
repo_case!(
    repo_captures_one_coherent_boundary_between_source_commits,
    "fork coordination",
    "captures one coherent boundary between source commits"
);
repo_case!(
    repo_publishes_create_when_it_reserves_a_shared_destination_id_first,
    "fork coordination",
    "publishes create when it reserves a shared destination id first"
);
repo_case!(
    repo_publishes_fork_when_it_reserves_a_shared_destination_id_first,
    "fork coordination",
    "publishes fork when it reserves a shared destination id first"
);

shared_repo_case!(
    shared_container_repo_creates_a_session_with_no_implicit_branch_and_rejects_duplicate_ids,
    "lifecycle",
    "creates a session with no implicit branch and rejects duplicate ids"
);
shared_repo_case!(
    shared_container_repo_close_drains_an_acquired_scope_and_rejects_a_queued_mutation_callback,
    "lifecycle",
    "close drains an acquired scope and rejects a queued mutation callback"
);
shared_repo_case!(
    shared_container_repo_lists_metadata_and_preserves_state_across_close_and_reopen,
    "lifecycle",
    "lists metadata and preserves state across close and reopen"
);
shared_repo_case!(
    shared_container_repo_deletes_closed_sessions_without_affecting_other_sessions,
    "lifecycle",
    "deletes closed sessions without affecting other sessions"
);
shared_repo_case!(
    shared_container_repo_rejects_opening_an_already_open_session,
    "ownership",
    "rejects opening an already-open session"
);
shared_repo_case!(
    shared_container_repo_rejects_pending_assistant_messages_without_changing_the_tree,
    "messages",
    "rejects pending assistant messages without changing the tree"
);
shared_repo_case!(
    shared_container_repo_preserves_every_settled_assistant_stop_reason,
    "messages",
    "preserves every settled assistant stop reason"
);
shared_repo_case!(
    shared_container_repo_tree_forks_a_fresh_session_before_first_attachment,
    "forks",
    "tree-forks a fresh session before first attachment"
);
shared_repo_case!(
    shared_container_repo_rejects_a_data_only_branch_and_releases_its_destination_id,
    "forks",
    "rejects a data-only branch and releases its destination id"
);
shared_repo_case!(
    shared_container_repo_forks_one_named_configured_branch_with_scoped_values_and_a_zero_ledger,
    "forks",
    "forks one named configured branch with scoped values and a zero ledger"
);
shared_repo_case!(
    shared_container_repo_enforces_branch_ancestry_for_at_and_before_placement,
    "forks",
    "enforces branch ancestry for at and before placement"
);
shared_repo_case!(
    shared_container_repo_forks_a_closed_source_session,
    "forks",
    "forks a closed source session"
);
shared_repo_case!(
    shared_container_repo_forks_the_whole_configured_tree_with_fresh_lane_state,
    "forks",
    "forks the whole configured tree with fresh lane state"
);
shared_repo_case!(
    shared_container_repo_rejects_only_surviving_unknown_reserved_scalar_state,
    "forks",
    "rejects only surviving unknown reserved scalar state"
);
shared_repo_case!(
    shared_container_repo_captures_one_coherent_boundary_between_source_commits,
    "fork coordination",
    "captures one coherent boundary between source commits"
);
shared_repo_case!(
    shared_container_repo_publishes_create_when_it_reserves_a_shared_destination_id_first,
    "fork coordination",
    "publishes create when it reserves a shared destination id first"
);
shared_repo_case!(
    shared_container_repo_publishes_fork_when_it_reserves_a_shared_destination_id_first,
    "fork coordination",
    "publishes fork when it reserves a shared destination id first"
);

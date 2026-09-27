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

// The shared runner and case list live in
// `conformance_case_registration.rs`, included below; the factories above
// stay per-suite.
include!("common/conformance_case_registration.rs");

// The memory backend registers the fork-destination-reservation cases the
// jsonl repo's conformance registration leaves out, upstream's
// createSessionRepoForkConformance composition.
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

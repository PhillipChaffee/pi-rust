//! The `JsonlStorage` and `JsonlSessionRepo` conformance suites, ported 1:1
//! from upstream `test/harness/jsonl-storage-conformance.test.ts` and
//! `test/harness/jsonl-session-repo-conformance.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. The repo suite registers
//! lifecycle, ownership, message, fork-behavior, fork-source-snapshot, and
//! streaming-fork cases; the fork-destination-reservation cases stay out,
//! matching upstream's jsonl registration.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod jsonl_common;
use jsonl_common::jsonl_repo;

use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::jsonl::storage::JsonlStorage;
use pi_agent_core::harness::session::jsonl::types::{
    JSONL_FORMAT_VERSION, JsonlStorageHeader, JsonlStorageOptions,
};
use pi_agent_core::harness::session::testing::{
    ConformanceCase, RepoFixture, StorageFixture, create_session_repo_fork_behavior_conformance,
    create_session_repo_fork_source_snapshot_conformance,
    create_session_repo_lifecycle_conformance, create_session_repo_message_conformance,
    create_session_repo_ownership_conformance, create_session_repo_streaming_fork_conformance,
    create_storage_conformance,
};
use pi_agent_core::harness::types::FileSystem;

use jsonl_common::WrappedEnv;

fn storage_fixture_factory()
-> Arc<dyn Fn() -> pi_ai::types::BoxedFuture<'static, StorageFixture> + Send + Sync> {
    Arc::new(|| {
        let root = jsonl_common::TempRoot::new();
        let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
        let options = JsonlStorageOptions {
            file_system,
            path: "session.jsonl".to_owned(),
            now: Some(Arc::new(move || jsonl_common::NOW)),
        };
        let header = JsonlStorageHeader {
            v: JSONL_FORMAT_VERSION,
            kind: "header".to_owned(),
            id: "session".to_owned(),
            storage_version: 1,
            created_at: jsonl_common::NOW,
            cwd: "/workspace".to_owned(),
            ..Default::default()
        };
        Box::pin(async move {
            let storage = JsonlStorage::create(&options, header, Vec::new(), &background_context())
                .await
                .expect("fixture storage");
            let _root = root;
            StorageFixture::new(Arc::new(storage))
        })
    })
}

fn repo_fixture_factory()
-> Arc<dyn Fn() -> pi_ai::types::BoxedFuture<'static, RepoFixture> + Send + Sync> {
    Arc::new(|| {
        let root = jsonl_common::TempRoot::new();
        let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
        let repo = Arc::new(jsonl_common::CwdScopedRepo::new(jsonl_repo(file_system)));
        let closer = repo.clone();
        Box::pin(async move {
            let _root = root;
            RepoFixture::new(
                repo,
                Some(Arc::new(move || {
                    let closer = closer.clone();
                    Box::pin(async move {
                        closer.close(&background_context()).await;
                    })
                })),
            )
        })
    })
}

fn storage_conformance_cases() -> Vec<ConformanceCase> {
    create_storage_conformance(&storage_fixture_factory())
}

fn repo_conformance_cases() -> Vec<ConformanceCase> {
    [
        create_session_repo_lifecycle_conformance(&repo_fixture_factory()),
        create_session_repo_ownership_conformance(&repo_fixture_factory()),
        create_session_repo_message_conformance(&repo_fixture_factory()),
        create_session_repo_fork_behavior_conformance(&repo_fixture_factory()),
        create_session_repo_fork_source_snapshot_conformance(&repo_fixture_factory()),
        create_session_repo_streaming_fork_conformance(&repo_fixture_factory()),
    ]
    .concat()
}

// The shared runner and case list live in
// `conformance_case_registration.rs`, included below; the factories above
// stay per-suite.
include!("common/conformance_case_registration.rs");

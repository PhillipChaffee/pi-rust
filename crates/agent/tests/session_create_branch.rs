//! The Session.createBranch suite, ported 1:1 from upstream
//! `test/harness/session-create-branch.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod session_common;
use session_common::*;

use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::commit as session_writes;
use pi_agent_core::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use pi_agent_core::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use pi_agent_core::harness::session::types::{
    CustomEntryBody, NewEntry, Session, SessionError, SessionReader,
};
use pi_agent_core::harness::session::values::{Write, branch_tip, lane_config, lane_state};

fn create_session() -> StorageBackedSession {
    StorageBackedSession::new(
        pi_agent_core::harness::session::types::SessionMetadata {
            id: "session".to_owned(),
            created_at: 1,
            storage_version: 1,
            cwd: None,
            parent_session_id: None,
            legacy_parent_session_path: None,
        },
        Arc::new(MemoryStorage::new(MemoryStorageOptions {
            now: Some(fixed_clock(10)),
        })),
        StorageBackedSessionOptions::default(),
    )
}

#[tokio::test]
async fn creates_only_the_data_branch_at_a_validated_target() {
    let session = create_session();
    commit_session(
        &session,
        vec![Write::Entry(Box::new(session_writes::insert_entry(
            NewEntry::Custom {
                id: "target".to_owned(),
                parent_id: None,
                body: CustomEntryBody {
                    custom_type: "target".to_owned(),
                    data: None,
                },
            },
        )))],
    )
    .await
    .expect("seed");

    let branch = session
        .create_branch("main", Some("target".to_owned()), &background_context())
        .await
        .expect("create branch");
    assert_eq!(
        branch.get_tip_id(&background_context()).await.expect("tip"),
        Some("target".to_owned()),
    );
    assert_eq!(
        session
            .get_value(&branch_tip("main").address, &background_context())
            .await
            .expect("value")
            .expect("stored")
            .value,
        serde_json::json!("target"),
    );
    assert!(
        session
            .get_value(&lane_config("main").address, &background_context())
            .await
            .expect("value")
            .is_none(),
    );
    assert!(
        session
            .get_value(&lane_state("main").address, &background_context())
            .await
            .expect("value")
            .is_none(),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn validates_names_and_non_null_targets() {
    let session = create_session();
    let rejected = session.create_branch("", None, &background_context()).await;
    assert!(matches!(
        rejected.err().expect("empty name"),
        SessionError::InvalidBranch { branch, reason }
            if branch.is_empty() && reason.contains("must not be empty")
    ),);
    let rejected = session
        .create_branch("bad\u{0}name", None, &background_context())
        .await;
    assert!(matches!(
        rejected.err().expect("nul name"),
        SessionError::InvalidBranch { branch, .. } if branch.contains('\u{0}')
    ),);
    let rejected = session
        .create_branch("main", Some("missing".to_owned()), &background_context())
        .await;
    assert_eq!(
        rejected.err().expect("missing target"),
        SessionError::UnknownTarget {
            target_id: "missing".to_owned(),
        },
    );
    assert!(
        session
            .branch("main", &background_context())
            .await
            .expect("branch")
            .is_none(),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn rejects_duplicates_atomically_including_concurrent_creation() {
    let session = create_session();
    let (first, second) = tokio::join!(
        session.create_branch("main", None, &background_context()),
        session.create_branch("main", None, &background_context()),
    );
    let outcomes = [first.is_ok(), second.is_ok()];
    assert_eq!(outcomes.iter().filter(|outcome| **outcome).count(), 1);
    let rejected = if first.is_err() { first } else { second };
    assert!(
        matches!(rejected.err().expect("rejected creation"), SessionError::BranchExists { branch } if branch == "main"),
    );
    assert!(
        session
            .branch("main", &background_context())
            .await
            .expect("branch")
            .is_some(),
    );
    session.close(&background_context()).await.expect("close");
}

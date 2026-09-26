//! The Branch suite, ported 1:1 from upstream `test/harness/branch.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod session_common;
use session_common::*;

use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::commit as session_writes;
use pi_agent_core::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use pi_agent_core::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use pi_agent_core::harness::session::types::{
    CustomEntryBody, Entry, IdGenerator, NewEntry, Session, SessionError,
};
use pi_agent_core::harness::session::values::{Write, branch_tip, set_value as set_value_write};

/// The entry-id counter the deterministic generator drives, upstream's
/// `let nextId = 1` module variable.
struct CountingGenerator {
    next_id: std::sync::atomic::AtomicUsize,
}

impl IdGenerator for CountingGenerator {
    fn next(&self, _timestamp_ms: Option<i64>) -> String {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
            + 1;
        format!("entry-{id}")
    }
}

fn counting_generator() -> Arc<dyn IdGenerator> {
    Arc::new(CountingGenerator {
        next_id: std::sync::atomic::AtomicUsize::new(0),
    })
}

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
        StorageBackedSessionOptions {
            id_generator: Some(counting_generator()),
            ..Default::default()
        },
    )
}

#[tokio::test]
async fn is_absent_until_explicitly_created_and_session_has_no_implicit_branch_surface() {
    let session = create_session();
    assert!(
        session
            .branch("main", &background_context())
            .await
            .expect("branch")
            .is_none(),
    );
    // The `in session` structural checks upstream pins restate compile-time:
    // the `Session` trait has no getTipId/appendMessage/findEntriesOnBranch
    // methods, so the surface absence is enforced by the trait itself.

    let branch = session
        .create_branch("main", None, &background_context())
        .await
        .expect("create branch");
    assert_eq!(branch.name(), "main");
    assert_eq!(
        branch.get_tip_id(&background_context()).await.expect("tip"),
        None
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn directly_appends_immutable_entries_and_advances_only_its_own_tip() {
    let session = create_session();
    let main = session
        .create_branch("main", None, &background_context())
        .await
        .expect("create main");
    let review = session
        .create_branch("review", None, &background_context())
        .await
        .expect("create review");
    let message = serde_json::from_value::<pi_agent_core::types::AgentMessage>(serde_json::json!({
        "role": "user",
        "content": "hello",
        "timestamp": 1,
    }))
    .expect("user message wire");

    let message_id = main
        .append_message(message.clone(), &background_context())
        .await
        .expect("append");
    let custom_id = main
        .append_custom_entry(
            "note",
            Some(serde_json::json!({ "ok": true })),
            &background_context(),
        )
        .await
        .expect("append");
    let review_id = review
        .append_custom_entry("review", None, &background_context())
        .await
        .expect("append");

    assert_eq!(
        main.get_tip_id(&background_context()).await.expect("tip"),
        Some(custom_id.clone())
    );
    assert_eq!(
        review.get_tip_id(&background_context()).await.expect("tip"),
        Some(review_id)
    );
    assert_eq!(
        main.find_entries(
            Some(&pi_agent_core::harness::session::types::BranchScan {
                order: Some(pi_agent_core::harness::session::types::BranchScanOrder::OldestFirst),
                ..Default::default()
            }),
            &background_context(),
        )
        .await
        .expect("entries")
        .into_iter()
        .map(|entry| entry.id().to_owned())
        .collect::<Vec<_>>(),
        [message_id.clone(), custom_id],
    );
    let found = main
        .find_entry(
            Some(&pi_agent_core::harness::session::types::BranchScan {
                kind: Some(pi_agent_core::harness::session::types::EntryType::Message),
                ..Default::default()
            }),
            &background_context(),
        )
        .await
        .expect("find entry")
        .expect("found");
    let Entry::Message { body, id, .. } = found else {
        panic!("expected a message entry");
    };
    assert_eq!(id, message_id);
    assert_eq!(body.message, message);
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn supports_explicit_starts_without_moving_the_branch_tip() {
    let session = create_session();
    commit_session(
        &session,
        vec![
            Write::Entry(Box::new(session_writes::insert_entry(NewEntry::Custom {
                id: "root".to_owned(),
                parent_id: None,
                body: CustomEntryBody {
                    custom_type: "root".to_owned(),
                    data: None,
                },
            }))),
            Write::Entry(Box::new(session_writes::insert_entry(NewEntry::Custom {
                id: "left".to_owned(),
                parent_id: Some("root".to_owned()),
                body: CustomEntryBody {
                    custom_type: "left".to_owned(),
                    data: None,
                },
            }))),
            Write::Entry(Box::new(session_writes::insert_entry(NewEntry::Custom {
                id: "right".to_owned(),
                parent_id: Some("root".to_owned()),
                body: CustomEntryBody {
                    custom_type: "right".to_owned(),
                    data: None,
                },
            }))),
            Write::ValueSet(
                set_value_write(&branch_tip("main"), Some("left".to_owned())).expect("write"),
            ),
        ],
    )
    .await
    .expect("seed");
    let branch = session
        .branch("main", &background_context())
        .await
        .expect("branch")
        .expect("main branch");

    assert_eq!(
        branch
            .find_entries(
                Some(&pi_agent_core::harness::session::types::BranchScan {
                    start: Some("right".to_owned()),
                    order: Some(
                        pi_agent_core::harness::session::types::BranchScanOrder::OldestFirst
                    ),
                    ..Default::default()
                }),
                &background_context(),
            )
            .await
            .expect("entries")
            .into_iter()
            .map(|entry| entry.id().to_owned())
            .collect::<Vec<_>>(),
        ["root", "right"],
    );
    assert_eq!(
        branch.get_tip_id(&background_context()).await.expect("tip"),
        Some("left".to_owned())
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn rejects_pending_assistant_messages_before_committing() {
    let session = create_session();
    let branch = session
        .create_branch("main", None, &background_context())
        .await
        .expect("create branch");
    let rejected = branch
        .append_message(assistant_message("pending", ""), &background_context())
        .await;
    assert_eq!(
        rejected.expect_err("pending rejected"),
        SessionError::PendingAssistantMessage,
    );
    assert_eq!(
        branch.get_tip_id(&background_context()).await.expect("tip"),
        None
    );
    session.close(&background_context()).await.expect("close");
}

//! The session layer's boundary suite: the restated branches no upstream
//! test pins directly — the memory backend's call-time semantics, the
//! mutation line's queue and seal edges, the abandoned commit outcome, the
//! value-scan ordering, and the commit validation messages.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::significant_drop_tightening,
    reason = "the mutation-line test's parked guard must outlive the assertions that pin its hold"
)]

mod session_common;
use session_common::*;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::commit::{
    CommittedWrite, prepare_storage_commit, validate_committed_writes,
};
use pi_agent_core::harness::session::in_memory_storage_state::InMemoryStorageState;
use pi_agent_core::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use pi_agent_core::harness::session::mutation_line::MutationLine;
use pi_agent_core::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use pi_agent_core::harness::session::testing::{StorageFixture, create_storage_conformance};
use pi_agent_core::harness::session::types::SessionRepo;
use pi_agent_core::harness::session::types::{
    CustomEntryBody, NewEntry, Session, SessionCreateOptions, SessionError, Storage, UsageWriteRow,
};
use pi_agent_core::harness::session::values::Write;
use pi_agent_core::harness::session::values::{self as stored_values};

fn memory_storage() -> Arc<MemoryStorage> {
    Arc::new(MemoryStorage::new(MemoryStorageOptions {
        now: Some(fixed_clock(NOW)),
    }))
}

fn session_with(storage: Arc<MemoryStorage>) -> StorageBackedSession {
    StorageBackedSession::new(metadata(), storage, StorageBackedSessionOptions::default())
}

#[tokio::test]
async fn memory_operations_apply_at_the_call_and_reject_after_close() {
    let storage = memory_storage();
    let write = stored_values::set_value(&stored_values::session_name(), "applied".to_owned())
        .expect("write");
    let pending = storage.commit(vec![Write::ValueSet(write)], &background_context());
    // The application is observable before the future is polled: a read in
    // the same synchronous stretch sees the committed value, upstream's
    // enqueue-then-settle boundary.
    assert_eq!(
        stored_values::StoredValue {
            namespace: stored_values::session_name().address.namespace.clone(),
            key: stored_values::session_name().address.key.clone(),
            value: serde_json::json!("applied"),
            seq: 1,
        },
        storage
            .get_value(
                &stored_values::session_name().address,
                &background_context()
            )
            .await
            .expect("value")
            .expect("stored"),
    );
    pending.await.expect("commit");

    storage.close(&background_context()).await.expect("close");
    let rejected = storage.commit(Vec::new(), &background_context()).await;
    assert_eq!(
        rejected.expect_err("closed"),
        SessionError::Message("MemoryStorage is closed".to_owned()),
    );
}

#[tokio::test]
async fn the_mutation_line_seals_queued_jobs_and_drains_the_granted_one() {
    let line = MutationLine::new();
    let granted_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let granted_done = Arc::new(tokio::sync::Notify::new());
    let started_flag = granted_started.clone();
    let done_signal = granted_done.clone();
    let granted = tokio::spawn({
        let line = line.clone();
        async move {
            let guard = line.acquire().await.expect("grant");
            started_flag.store(true, Ordering::Release);
            done_signal.notified().await;
            drop(guard);
        }
    });

    let queued = tokio::spawn({
        let line = line.clone();
        async move { line.acquire().await }
    });
    tokio::task::yield_now().await;
    assert!(
        granted_started.load(Ordering::Acquire),
        "the first job grants immediately",
    );
    line.seal(SessionError::Message("sealed".to_owned()));
    // Release the granted job first: the queued job then takes the line and
    // rejects on the seal it now sees, upstream's queued-job check.
    granted_done.notify_waiters();
    granted.await.expect("granted task");
    let queued = queued.await.expect("queued task");
    assert!(
        matches!(queued.expect_err("sealed job"), SessionError::Message(message) if message == "sealed"),
        "the queued job rejects with the seal error after its grant",
    );

    // The drain waits for the granted job without failing on the latch.
    line.drain().await;
}

#[tokio::test]
async fn an_abandoned_commit_attempt_settles_the_guard_for_end() {
    let session = session_with(memory_storage());
    let mutation = session
        .begin_mutation(&background_context())
        .await
        .expect("begin");
    let write =
        stored_values::set_value(&stored_values::session_name(), "lost".to_owned()).expect("write");
    // The commit future is created and dropped without polling: the guard
    // consumption stays observable and `end` does not hang.
    let abandoned = mutation.commit(vec![Write::ValueSet(write)], &background_context());
    drop(abandoned);
    mutation.end(&background_context()).await.expect("end");
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn value_scans_order_keys_by_code_point() {
    let storage = memory_storage();
    let prefix = stored_values::generic_value("test.order", "");
    storage
        .commit(
            vec![
                Write::ValueSet(
                    stored_values::set_value(&prefix, serde_json::json!(0)).expect("write"),
                ),
                Write::ValueSet(
                    stored_values::set_value(
                        &stored_values::generic_value("test.order", "a"),
                        serde_json::json!(1),
                    )
                    .expect("write"),
                ),
                Write::ValueSet(
                    stored_values::set_value(
                        &stored_values::generic_value("test.order", "\u{e000}"),
                        serde_json::json!(2),
                    )
                    .expect("write"),
                ),
                Write::ValueSet(
                    stored_values::set_value(
                        &stored_values::generic_value("test.order", "\u{10000}"),
                        serde_json::json!(3),
                    )
                    .expect("write"),
                ),
            ],
            &background_context(),
        )
        .await
        .expect("commit");

    let stored = storage
        .scan_values(&prefix.address, &background_context())
        .await
        .expect("scan");
    assert_eq!(
        stored
            .iter()
            .map(|value| value.key.as_str())
            .collect::<Vec<_>>(),
        ["", "a", "\u{e000}", "\u{10000}"],
    );
    storage.close(&background_context()).await.expect("close");
}

#[test]
fn commit_validation_pins_the_upstream_messages() {
    let mut state = InMemoryStorageState::new();
    let entry = NewEntry::Custom {
        id: "entry".to_owned(),
        parent_id: None,
        body: CustomEntryBody {
            custom_type: "note".to_owned(),
            data: None,
        },
    };
    let prepared = prepare_storage_commit(
        vec![Write::Entry(Box::new(
            pi_agent_core::harness::session::commit::insert_entry(entry),
        ))],
        1,
        NOW,
    );
    state.apply_validated(&prepared.writes);

    // A duplicate id in a later transaction.
    let duplicate = prepare_storage_commit(
        vec![Write::Usage(
            pi_agent_core::harness::session::commit::insert_usage(UsageWriteRow {
                id: "entry".to_owned(),
                usage: pi_ai::types::Usage::default(),
                entry_id: None,
                adjustment: false,
                details: None,
            }),
        )],
        2,
        NOW,
    );
    let error = validate_committed_writes(&duplicate.writes, 2, &state).expect_err("duplicate");
    assert_eq!(error.to_string(), "Duplicate entry or usage id: entry",);

    // A missing parent.
    let orphan = NewEntry::Custom {
        id: "orphan".to_owned(),
        parent_id: Some("missing".to_owned()),
        body: CustomEntryBody {
            custom_type: "note".to_owned(),
            data: None,
        },
    };
    let prepared = prepare_storage_commit(
        vec![Write::Entry(Box::new(
            pi_agent_core::harness::session::commit::insert_entry(orphan),
        ))],
        2,
        NOW,
    );
    let error = validate_committed_writes(&prepared.writes, 2, &state).expect_err("orphan");
    assert_eq!(error.to_string(), "Missing parent entry: missing");

    // A non-monotonic sequence: the validator receives a write whose
    // sequence is below the transaction's first.
    let committed = vec![CommittedWrite::ValueSet {
        seq: 1,
        namespace: "test.value".to_owned(),
        key: String::new(),
        value: serde_json::json!(1),
    }];
    let error = validate_committed_writes(&committed, 2, &state).expect_err("non-monotonic");
    assert_eq!(error.to_string(), "Non-monotonic storage sequence: 1");
}

#[tokio::test]
async fn the_facade_admits_no_operation_after_close() {
    let repo = memory_repo();
    let session = repo
        .create(
            SessionCreateOptions {
                id: Some("session".to_owned()),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("create");
    let metadata = session.metadata().clone();
    session.close(&background_context()).await.expect("close");

    let rejected = session
        .set_value(
            &stored_values::session_name().address,
            serde_json::json!("late"),
            &background_context(),
        )
        .await;
    assert!(
        rejected
            .expect_err("closed")
            .to_string()
            .contains("Session is closed"),
    );
    let reopened = repo
        .open(&metadata, &background_context())
        .await
        .expect("reopen");
    assert!(
        reopened
            .get_name(&background_context())
            .await
            .expect("name")
            .is_none(),
        "the rejected operation must not have landed",
    );
    reopened.close(&background_context()).await.expect("close");
    repo.close(&background_context()).await.expect("close repo");
}

/// The storage conformance creator yields the upstream case count; a
/// dropped case fails this boundary test.
#[test]
fn the_storage_conformance_creator_yields_the_upstream_case_count() {
    let factory: Arc<dyn Fn() -> pi_ai::types::BoxedFuture<'static, StorageFixture> + Send + Sync> =
        Arc::new(|| {
            let storage: Arc<dyn Storage> =
                Arc::new(MemoryStorage::new(MemoryStorageOptions::default()));
            Box::pin(async move { StorageFixture::new(storage) })
        });
    let cases = create_storage_conformance(&factory);
    assert_eq!(cases.len(), 21);
}

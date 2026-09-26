//! The `InstrumentedStorage` suite, ported 1:1 from upstream
//! `test/harness/instrumented-storage.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::too_many_lines,
    reason = "the delegation case mirrors upstream's read-by-read assertions in one body"
)]

mod session_common;
use session_common::*;

use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use pi_agent_core::harness::session::testing::InstrumentedStorage;
use pi_agent_core::harness::session::types::{
    CommitResult, EntryScan, SessionError, SessionStats, Storage, StorageBranchScan, UsageScan,
};
use pi_agent_core::harness::session::values::{
    self as stored_values, Write, set_value as set_value_write,
};

/// The parked-commit queue the controlled delegate's hook shares with the
/// resolver, upstream's `ControlledCommitStorage`.
struct ParkedCommits {
    pending: std::sync::Mutex<
        std::collections::VecDeque<
            tokio::sync::oneshot::Sender<Result<CommitResult, SessionError>>,
        >,
    >,
}

impl ParkedCommits {
    fn resolve_next(&self, result: CommitResult) {
        let pending = self
            .pending
            .lock()
            .expect("pending lock")
            .pop_front()
            .expect("no pending commit");
        let _ = pending.send(Ok(result));
    }
}

fn empty_stats() -> SessionStats {
    SessionStats {
        message_count: 0,
        usage: pi_ai::types::Usage::default(),
    }
}

/// The parked-commit delegate builder, upstream's `ControlledCommitStorage`
/// constructor: commits park until [`ParkedCommits::resolve_next`] fires.
fn parked_commit_delegate() -> (Arc<HookedStorage>, Arc<ParkedCommits>) {
    let parked = Arc::new(ParkedCommits {
        pending: std::sync::Mutex::new(std::collections::VecDeque::new()),
    });
    let delegate = Arc::new(HookedStorage {
        base: Arc::new(MemoryStorage::new(MemoryStorageOptions::default())),
        commit_hook: Box::new({
            let parked = parked.clone();
            move |_base: Arc<MemoryStorage>, _writes: Vec<Write>, _context| {
                // Upstream records the parked commit in the synchronous
                // commit call; the push happens before the future is
                // returned.
                let (tx, rx) = tokio::sync::oneshot::channel();
                parked.pending.lock().expect("pending lock").push_back(tx);
                Box::pin(async move { rx.await.expect("commit resolver") })
            }
        }),
    });
    (delegate, parked)
}

/// A name write, the suites' shorthand.
fn name_write(value: &str) -> Write {
    Write::ValueSet(
        set_value_write(&stored_values::session_name(), value.to_owned()).expect("write"),
    )
}

#[tokio::test]
async fn records_commit_attempts_synchronously_in_admission_order_before_settlement() {
    let (delegate, parked) = parked_commit_delegate();
    let storage = InstrumentedStorage::new(delegate.clone());
    let first_transaction = vec![name_write("first")];
    let second_transaction = vec![name_write("second")];

    let first_commit = storage.commit(first_transaction.clone(), &background_context());
    assert_eq!(
        storage.get_commit_attempts(),
        vec![first_transaction.clone()]
    );
    let second_commit = storage.commit(second_transaction.clone(), &background_context());
    assert_eq!(
        storage.get_commit_attempts(),
        vec![first_transaction.clone(), second_transaction.clone()],
    );

    let first_result = CommitResult {
        first_seq: 1,
        seqs: vec![1],
        timestamp: 10,
        stats: empty_stats(),
    };
    parked.resolve_next(first_result.clone());
    assert_eq!(first_commit.await.expect("first commit"), first_result);
    assert_eq!(
        storage.get_commit_attempts(),
        vec![first_transaction, second_transaction.clone()],
    );
    let second_result = CommitResult {
        first_seq: 2,
        seqs: vec![2],
        timestamp: 20,
        stats: empty_stats(),
    };
    parked.resolve_next(second_result.clone());
    assert_eq!(second_commit.await.expect("second commit"), second_result);
    storage.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn records_the_transaction_reference_passed_to_the_delegate() {
    let (delegate, parked) = parked_commit_delegate();
    let storage = InstrumentedStorage::new(delegate.clone());
    let transaction = vec![name_write("value")];

    let commit = storage.commit(transaction.clone(), &background_context());
    assert_eq!(storage.get_commit_attempts()[0], transaction);
    parked.resolve_next(CommitResult {
        first_seq: 1,
        seqs: vec![1],
        timestamp: 10,
        stats: empty_stats(),
    });
    commit.await.expect("commit");
    storage.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn clears_recorded_attempts_between_phases_without_affecting_the_delegate() {
    let delegate = MemoryStorage::new(MemoryStorageOptions {
        now: Some(fixed_clock(100)),
    });
    let storage = InstrumentedStorage::new(Arc::new(delegate));
    storage
        .commit(vec![name_write("first")], &background_context())
        .await
        .expect("commit");

    storage.clear_commit_attempts();
    assert!(storage.get_commit_attempts().is_empty());
    assert_eq!(
        storage
            .get_value(
                &stored_values::session_name().address,
                &background_context()
            )
            .await
            .expect("value")
            .expect("stored")
            .value,
        serde_json::json!("first"),
    );

    let second_transaction = vec![name_write("second")];
    storage
        .commit(second_transaction.clone(), &background_context())
        .await
        .expect("commit");
    assert_eq!(storage.get_commit_attempts(), vec![second_transaction]);
    storage.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn delegates_every_read_and_query_without_recording_synthetic_writes() {
    let delegate = Arc::new(MemoryStorage::new(MemoryStorageOptions {
        now: Some(fixed_clock(100)),
    }));
    let storage = InstrumentedStorage::new(delegate.clone());
    let events = stored_values::generic_list("test.events", "");
    storage
        .commit(
            vec![
                Write::Entry(Box::new(
                    pi_agent_core::harness::session::commit::insert_entry(
                        pi_agent_core::harness::session::types::NewEntry::Custom {
                            id: "root".to_owned(),
                            parent_id: None,
                            body: pi_agent_core::harness::session::types::CustomEntryBody {
                                custom_type: "note".to_owned(),
                                data: None,
                            },
                        },
                    ),
                )),
                name_write("session"),
                Write::ListAppend(
                    stored_values::append_list(&events, serde_json::json!("event")).expect("write"),
                ),
                Write::Usage(pi_agent_core::harness::session::commit::insert_usage(
                    pi_agent_core::harness::session::types::UsageWriteRow {
                        id: "usage".to_owned(),
                        adjustment: false,
                        usage: pi_ai::types::Usage {
                            input: 1,
                            output: 2,
                            cache_read: 0,
                            cache_write: 0,
                            cache_write_1h: None,
                            reasoning: None,
                            total_tokens: 3,
                            cost: pi_ai::types::UsageCost::default(),
                        },
                        entry_id: None,
                        details: None,
                    },
                )),
            ],
            &background_context(),
        )
        .await
        .expect("commit");

    assert_eq!(
        storage
            .get_entries(vec!["root".to_owned()], &background_context())
            .await
            .expect("entries"),
        delegate
            .get_entries(vec!["root".to_owned()], &background_context())
            .await
            .expect("entries"),
    );
    assert_eq!(
        storage
            .get_value(
                &stored_values::session_name().address,
                &background_context()
            )
            .await
            .expect("value"),
        delegate
            .get_value(
                &stored_values::session_name().address,
                &background_context()
            )
            .await
            .expect("value"),
    );
    assert_eq!(
        storage
            .scan_values(
                &stored_values::session_name().address,
                &background_context()
            )
            .await
            .expect("values"),
        delegate
            .scan_values(
                &stored_values::session_name().address,
                &background_context()
            )
            .await
            .expect("values"),
    );
    assert_eq!(
        storage
            .read_list(&events.address, None, &background_context())
            .await
            .expect("list"),
        delegate
            .read_list(&events.address, None, &background_context())
            .await
            .expect("list"),
    );
    assert_eq!(
        storage
            .scan_branch(
                &StorageBranchScan {
                    start: "root".to_owned(),
                    ..Default::default()
                },
                &background_context(),
            )
            .await
            .expect("branch"),
        delegate
            .scan_branch(
                &StorageBranchScan {
                    start: "root".to_owned(),
                    ..Default::default()
                },
                &background_context(),
            )
            .await
            .expect("branch"),
    );
    assert_eq!(
        storage
            .scan_branch_structure(
                &StorageBranchScan {
                    start: "root".to_owned(),
                    ..Default::default()
                },
                &background_context(),
            )
            .await
            .expect("structure"),
        delegate
            .scan_branch_structure(
                &StorageBranchScan {
                    start: "root".to_owned(),
                    ..Default::default()
                },
                &background_context(),
            )
            .await
            .expect("structure"),
    );
    assert_eq!(
        storage
            .scan_entries(&harness_scan_asc(), &background_context(),)
            .await
            .expect("entries"),
        delegate
            .scan_entries(&harness_scan_asc(), &background_context(),)
            .await
            .expect("entries"),
    );
    assert_eq!(
        storage
            .scan_usage(&harness_usage_asc(), &background_context(),)
            .await
            .expect("usage"),
        delegate
            .scan_usage(&harness_usage_asc(), &background_context(),)
            .await
            .expect("usage"),
    );
    assert_eq!(
        storage
            .get_stats(&background_context())
            .await
            .expect("stats"),
        delegate
            .get_stats(&background_context())
            .await
            .expect("stats"),
    );
    assert_eq!(storage.get_commit_attempts().len(), 1);
    storage.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn records_list_appends_without_reading_the_target_list() {
    let delegate = Arc::new(MemoryStorage::new(MemoryStorageOptions {
        now: Some(fixed_clock(100)),
    }));
    let storage = InstrumentedStorage::new(delegate.clone());
    let events = stored_values::generic_list("test.events", "");

    storage
        .commit(
            vec![Write::ListAppend(
                stored_values::append_list(&events, serde_json::json!("event")).expect("write"),
            )],
            &background_context(),
        )
        .await
        .expect("commit");

    // Upstream spies on readList and asserts it never ran; the memory
    // backend's list append never reads, so the attempt record carries the
    // append write untouched.
    assert_eq!(
        storage.get_commit_attempts(),
        vec![vec![Write::ListAppend(stored_values::ListAppendWrite {
            kind: "list".to_owned(),
            op: "append".to_owned(),
            namespace: "test.events".to_owned(),
            key: String::new(),
            value: serde_json::json!("event"),
        },)]],
    );
    storage.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn delegates_close_idempotence_and_admitted_commit_draining() {
    let delegate = MemoryStorage::new(MemoryStorageOptions {
        now: Some(fixed_clock(100)),
    });
    let storage = InstrumentedStorage::new(Arc::new(delegate));
    let admitted = storage.commit(vec![name_write("admitted")], &background_context());

    let first_close = storage.close(&background_context());
    let second_close = storage.close(&background_context());
    admitted.await.expect("admitted commit");
    let (first_closed, second_closed) = tokio::join!(first_close, second_close);
    assert!(first_closed.is_ok() && second_closed.is_ok());
    let rejected = storage.get_stats(&background_context()).await;
    assert!(
        rejected
            .expect_err("closed storage")
            .to_string()
            .contains("MemoryStorage is closed"),
    );
    assert_eq!(storage.get_commit_attempts().len(), 1);
}

/// The ascending entry scan the delegation test compares with.
fn harness_scan_asc() -> EntryScan {
    EntryScan {
        order: Some(pi_agent_core::harness::session::types::EntryScanOrder::Asc),
        ..Default::default()
    }
}

/// The ascending usage scan the delegation test compares with.
fn harness_usage_asc() -> UsageScan {
    UsageScan {
        order: Some(pi_agent_core::harness::session::types::EntryScanOrder::Asc),
        ..Default::default()
    }
}

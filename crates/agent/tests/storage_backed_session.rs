//! The `StorageBackedSession` suite, ported 1:1 from upstream
//! `test/harness/storage-backed-session.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod session_common;
use session_common::storage_backed_session;
use session_common::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::commit as session_writes;
use pi_agent_core::harness::session::memory::{MemoryStorage, MemoryStorageOptions};
use pi_agent_core::harness::session::session::{StorageBackedSession, StorageBackedSessionOptions};
use pi_agent_core::harness::session::testing::{GatingStorage, InstrumentedStorage};
use pi_agent_core::harness::session::types::{
    CustomEntryBody, Entry, IdGenerator, MessageEntry, NewEntry, Session, SessionError,
    SessionMutator, SessionReader, Storage, StorageBranchScan, UsageWriteRow,
};
use pi_agent_core::harness::session::values::{
    self as stored_values, Write, set_value as set_value_write,
};
use serde_json::json;

fn memory_storage() -> Arc<MemoryStorage> {
    Arc::new(MemoryStorage::new(MemoryStorageOptions {
        now: Some(fixed_clock(NOW)),
    }))
}

fn scalar() -> stored_values::Value<String> {
    stored_values::value("test.application.scalar", "").expect("address")
}

fn events() -> stored_values::ValueList<String> {
    stored_values::list("test.application.events", "").expect("address")
}

/// The uuidv7 timestamp an id carries, upstream's `decodeTimestamp`.
fn decode_timestamp(id: &str) -> u64 {
    u64::from_str_radix(&id.replace('-', "")[..12], 16).expect("uuid timestamp")
}

#[tokio::test]
async fn delegates_typed_values_directly_without_validation_or_cloning() {
    let storage = InstrumentedStorage::new(memory_storage());
    let storage = Arc::new(storage);
    let session = storage_backed_session(storage.clone());
    let data = json!({ "nested": ["original"] });
    let transaction = vec![
        Write::Entry(Box::new(session_writes::insert_entry(NewEntry::Custom {
            id: ENTRY_ID.to_owned(),
            parent_id: None,
            body: CustomEntryBody {
                custom_type: "note".to_owned(),
                data: Some(data.clone()),
            },
        }))),
        Write::ValueSet(
            set_value_write(
                &stored_values::generic_value("test.value", "state"),
                data.clone(),
            )
            .expect("write"),
        ),
    ];

    let result = commit_session(&session, transaction.clone())
        .await
        .expect("commit");

    assert_eq!(storage.get_commit_attempts()[0], transaction);
    let entry = session
        .get_entries(vec![ENTRY_ID.to_owned()], &background_context())
        .await
        .expect("entries")
        .get(ENTRY_ID)
        .cloned()
        .expect("entry");
    let Entry::Custom {
        body,
        seq,
        timestamp,
        ..
    } = &entry
    else {
        panic!("expected custom entry");
    };
    assert_eq!(*seq, result.seqs[0]);
    assert_eq!(*timestamp, NOW);
    assert_eq!(body.data, Some(data));
    assert_eq!(
        session
            .get_value(
                &stored_values::generic_value("test.value", "state").address,
                &background_context()
            )
            .await
            .expect("value")
            .expect("stored")
            .value,
        json!({ "nested": ["original"] }),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn composes_bound_values_and_lists_atomically_with_entries_and_usage() {
    let storage = InstrumentedStorage::new(memory_storage());
    let storage = Arc::new(storage);
    let session = storage_backed_session(storage.clone());
    let scalar = scalar();
    let events = events();

    let result = session
        .mutate(
            StorageBackedSession::commit_writes_callback(vec![
                Write::Entry(Box::new(session_writes::insert_entry(NewEntry::Custom {
                    id: ENTRY_ID.to_owned(),
                    parent_id: None,
                    body: CustomEntryBody {
                        custom_type: "note".to_owned(),
                        data: None,
                    },
                }))),
                Write::ValueSet(set_value_write(&scalar, "state".to_owned()).expect("write")),
                Write::ListAppend(
                    stored_values::append_list(&events, "event".to_owned()).expect("write"),
                ),
                Write::Usage(session_writes::insert_usage(UsageWriteRow {
                    id: "usage".to_owned(),
                    adjustment: false,
                    usage: pi_ai::types::Usage {
                        input: 1,
                        output: 1,
                        cache_read: 0,
                        cache_write: 0,
                        cache_write_1h: None,
                        reasoning: None,
                        total_tokens: 2,
                        cost: pi_ai::types::UsageCost::default(),
                    },
                    entry_id: None,
                    details: None,
                })),
            ]),
            &background_context(),
        )
        .await
        .map(pi_agent_core::harness::session::testing::conformance::downcast_commit_result)
        .expect("commit");

    assert_eq!(result.seqs.len(), 4);
    assert_eq!(
        session
            .get_value(&scalar.address, &background_context())
            .await
            .expect("value")
            .expect("stored")
            .seq,
        result.seqs[1],
    );
    assert_eq!(
        session
            .read_list(&events.address, None, &background_context())
            .await
            .expect("list"),
        vec![stored_values::ListElement {
            seq: result.seqs[2],
            value: serde_json::json!("event"),
        }],
    );
    assert_eq!(storage.get_commit_attempts().len(), 1);
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn serializes_read_modify_write_callbacks_on_the_single_session_line() {
    let session = storage_backed_session(memory_storage());
    let counter = stored_values::value::<u64>("test.counter", "").expect("address");
    let increment = {
        let session = session.clone();
        let counter = counter.clone();
        move || {
            let session = session.clone();
            let counter = counter.clone();
            async move {
                let next = session
                    .get_value(&counter.address, &background_context())
                    .await
                    .expect("read")
                    .and_then(|stored| stored.value.as_u64())
                    .unwrap_or_default()
                    + 1;
                session
                    .mutate(
                        StorageBackedSession::commit_writes_callback(vec![Write::ValueSet(
                            set_value_write(&counter, next).expect("write"),
                        )]),
                        &background_context(),
                    )
                    .await
                    .expect("commit");
                next
            }
        }
    };

    let (first, second) = tokio::join!(increment(), increment());
    assert_eq!((first, second), (1, 2));
    assert_eq!(
        session
            .get_value(&counter.address, &background_context())
            .await
            .expect("read")
            .expect("stored")
            .value,
        serde_json::json!(2),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn keeps_separate_direct_reads_and_writes_deliberately_non_atomic() {
    let session = storage_backed_session(memory_storage());
    let counter = stored_values::value::<u64>("test.counter", "").expect("address");
    // The both-read gate upstream's `deferred()` pair restates: both
    // increments read before either writes, so the shared read total is 1.
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let increment = {
        let session = session.clone();
        let counter = counter.clone();
        let barrier = barrier.clone();
        move || {
            let session = session.clone();
            let counter = counter.clone();
            let barrier = barrier.clone();
            async move {
                let next = session
                    .get_value(&counter.address, &background_context())
                    .await
                    .expect("read")
                    .and_then(|stored| stored.value.as_u64())
                    .unwrap_or_default()
                    + 1;
                barrier.wait().await;
                session
                    .set_value(
                        &counter.address,
                        serde_json::json!(next),
                        &background_context(),
                    )
                    .await
                    .expect("write");
            }
        }
    };

    tokio::join!(increment(), increment());
    assert_eq!(
        session
            .get_value(&counter.address, &background_context())
            .await
            .expect("read")
            .expect("stored")
            .value,
        serde_json::json!(1),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn queues_a_nested_public_writer_until_its_owning_callback_returns() {
    let session = storage_backed_session(memory_storage());
    let settled = Arc::new(AtomicBool::new(false));
    // The callback hands the spawned writer's handle back through shared
    // state: awaiting the nested writer inside the callback is the
    // documented deadlock, so the test awaits it after the callback returns.
    let nested_handle: Arc<tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>> =
        Arc::new(tokio::sync::Mutex::new(None));
    {
        let settled = settled.clone();
        let callback_session = session.clone();
        let nested_handle = nested_handle.clone();
        session
            .mutate(
                Box::new(
                    move |_mutator: &dyn SessionMutator,
                          _context|
                          -> pi_ai::types::BoxedFuture<
                        '_,
                        Result<Box<dyn std::any::Any + Send>, SessionError>,
                    > {
                        let session = callback_session.clone();
                        let settled = settled.clone();
                        let nested_handle = nested_handle.clone();
                        Box::pin(async move {
                            let flag = settled.clone();
                            let task = tokio::spawn(async move {
                                session
                                    .set_value(
                                        &stored_values::session_name().address,
                                        serde_json::json!("nested"),
                                        &background_context(),
                                    )
                                    .await
                                    .expect("nested write");
                                flag.store(true, Ordering::Release);
                            });
                            tokio::task::yield_now().await;
                            assert!(
                                !settled.load(Ordering::Acquire),
                                "the nested writer must queue behind the callback",
                            );
                            *nested_handle.lock().await = Some(task);
                            let done: Box<dyn std::any::Any + Send> = Box::new(());
                            Ok(done)
                        })
                    },
                ),
                &background_context(),
            )
            .await
            .expect("mutate");
    }
    // The nested writer runs only after the callback released the line.
    let task = nested_handle.lock().await.take().expect("nested handle");
    task.await.expect("nested task");
    assert!(settled.load(Ordering::Acquire));
    assert_eq!(
        session.get_name(&background_context()).await.expect("name"),
        Some("nested".to_owned()),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn exposes_either_side_of_an_atomic_multi_write_commit_to_direct_reads() {
    let first = stored_values::value::<String>("test.atomic", "first").expect("address");
    let second = stored_values::value::<String>("test.atomic", "second").expect("address");
    let storage = Arc::new(GatingStorage::new(memory_storage()));
    storage
        .commit(
            vec![
                Write::ValueSet(set_value_write(&first, "old".to_owned()).expect("write")),
                Write::ValueSet(set_value_write(&second, "old".to_owned()).expect("write")),
            ],
            &background_context(),
        )
        .await
        .expect("setup");
    storage.arm();
    let session = storage_backed_session(storage.clone());
    let committing = {
        let session = session.clone();
        let first = first.clone();
        let second = second.clone();
        tokio::spawn(async move {
            commit_session(
                &session,
                vec![
                    Write::ValueSet(set_value_write(&first, "new".to_owned()).expect("write")),
                    Write::ValueSet(set_value_write(&second, "new".to_owned()).expect("write")),
                ],
            )
            .await
            .expect("commit");
        })
    };
    storage.wait_pending(1).await.expect("parked");

    assert_eq!(
        session
            .get_value(&first.address, &background_context())
            .await
            .expect("read")
            .expect("stored")
            .value,
        serde_json::json!("old"),
    );
    assert_eq!(
        session
            .get_value(&second.address, &background_context())
            .await
            .expect("read")
            .expect("stored")
            .value,
        serde_json::json!("old"),
    );
    storage.next(1).await.expect("released");
    committing.await.expect("committing task");
    assert_eq!(
        session
            .get_value(&first.address, &background_context())
            .await
            .expect("read")
            .expect("stored")
            .value,
        serde_json::json!("new"),
    );
    assert_eq!(
        session
            .get_value(&second.address, &background_context())
            .await
            .expect("read")
            .expect("stored")
            .value,
        serde_json::json!("new"),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn holds_the_explicit_session_barrier_through_commit_until_end() {
    let storage = Arc::new(InstrumentedStorage::new(memory_storage()));
    let session = storage_backed_session(storage.clone());
    let mutation = session
        .begin_mutation(&background_context())
        .await
        .expect("begin");
    let queued_started = Arc::new(AtomicBool::new(false));
    let queued = queued_mutate_probe(Arc::new(session.clone()), queued_started.clone());

    tokio::task::yield_now().await;
    assert!(!queued_started.load(Ordering::Acquire));
    assert!(
        mutation
            .get_value(
                &stored_values::session_name().address,
                &background_context()
            )
            .await
            .expect("read")
            .is_none(),
    );
    let result = mutation
        .commit(Vec::new(), &background_context())
        .await
        .expect("commit");
    assert!(result.seqs.is_empty());
    assert!(!queued_started.load(Ordering::Acquire));
    assert!(
        mutation
            .get_value(
                &stored_values::session_name().address,
                &background_context()
            )
            .await
            .expect("read")
            .is_none(),
    );
    mutation.end(&background_context()).await.expect("end");
    queued.await.expect("queued task");
    assert!(queued_started.load(Ordering::Acquire));
    assert_eq!(storage.get_commit_attempts().len(), 1);
    let rejected = mutation
        .get_entries(Vec::new(), &background_context())
        .await;
    assert!(
        rejected
            .expect_err("invalidated mutator")
            .to_string()
            .contains("outside its mutation callback"),
    );
    let rejected = mutation.commit(Vec::new(), &background_context()).await;
    assert!(
        rejected
            .expect_err("invalidated mutator")
            .to_string()
            .contains("outside its mutation callback"),
    );
    mutation
        .end(&background_context())
        .await
        .expect("idempotent end");
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn allows_direct_reads_to_observe_a_committed_value_before_the_mutation_scope_ends() {
    let session = storage_backed_session(memory_storage());
    let mutation = session
        .begin_mutation(&background_context())
        .await
        .expect("begin");
    assert!(
        session
            .get_value(
                &stored_values::session_name().address,
                &background_context()
            )
            .await
            .expect("read")
            .is_none(),
    );
    mutation
        .commit(
            vec![Write::ValueSet(
                set_value_write(&stored_values::session_name(), "visible".to_owned())
                    .expect("write"),
            )],
            &background_context(),
        )
        .await
        .expect("commit");
    let queued_started = Arc::new(AtomicBool::new(false));
    let queued = queued_mutate_probe(Arc::new(session.clone()), queued_started.clone());

    assert_eq!(
        session
            .get_value(
                &stored_values::session_name().address,
                &background_context()
            )
            .await
            .expect("read")
            .expect("stored")
            .value,
        serde_json::json!("visible"),
    );
    assert!(!queued_started.load(Ordering::Acquire));
    mutation.end(&background_context()).await.expect("end");
    queued.await.expect("queued task");
    assert!(queued_started.load(Ordering::Acquire));
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn ends_an_explicit_mutation_without_committing_and_lets_close_finish() {
    let storage = Arc::new(InstrumentedStorage::new(memory_storage()));
    let session = storage_backed_session(storage.clone());
    let mutation = session
        .begin_mutation(&background_context())
        .await
        .expect("begin");
    let closed = Arc::new(AtomicBool::new(false));
    let closed_flag = closed.clone();
    let closing_session = session.clone();
    let closing = tokio::spawn(async move {
        closing_session
            .close(&background_context())
            .await
            .expect("close");
        closed_flag.store(true, Ordering::Release);
    });

    tokio::task::yield_now().await;
    assert!(!closed.load(Ordering::Acquire));
    mutation.end(&background_context()).await.expect("end");
    closing.await.expect("closing task");
    assert!(closed.load(Ordering::Acquire));
    assert!(storage.get_commit_attempts().is_empty());
}

#[tokio::test]
async fn exposes_explicit_branch_scans_through_the_session_and_callback_scoped_mutator() {
    let storage = memory_storage();
    let session = storage_backed_session(storage);
    let child_id = "00000000-0000-7000-8000-000000000002";
    commit_session(
        &session,
        vec![
            Write::Entry(Box::new(session_writes::insert_entry(NewEntry::Custom {
                id: ENTRY_ID.to_owned(),
                parent_id: None,
                body: CustomEntryBody {
                    custom_type: "root".to_owned(),
                    data: None,
                },
            }))),
            Write::Entry(Box::new(session_writes::insert_entry(NewEntry::Custom {
                id: child_id.to_owned(),
                parent_id: Some(ENTRY_ID.to_owned()),
                body: CustomEntryBody {
                    custom_type: "child".to_owned(),
                    data: None,
                },
            }))),
        ],
    )
    .await
    .expect("seed");

    let scanned = session
        .scan_branch(
            &StorageBranchScan {
                start: child_id.to_owned(),
                order: Some(pi_agent_core::harness::session::types::BranchScanOrder::OldestFirst),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("scan");
    assert_eq!(
        scanned.iter().map(Entry::id).collect::<Vec<_>>(),
        [ENTRY_ID, child_id],
    );
    let captured = Arc::new(tokio::sync::Mutex::new(None::<Box<dyn SessionMutator>>));
    {
        let captured = captured.clone();
        session
            .mutate(
                Box::new(
                    move |mutator: &dyn SessionMutator,
                          _context|
                          -> pi_ai::types::BoxedFuture<
                        '_,
                        Result<Box<dyn std::any::Any + Send>, SessionError>,
                    > {
                        let captured = captured.clone();
                        Box::pin(async move {
                            let scanned = mutator
                                .scan_branch(
                                    &StorageBranchScan {
                                        start: child_id.to_owned(),
                                        limit: Some(1),
                                        ..Default::default()
                                    },
                                    &background_context(),
                                )
                                .await
                                .expect("scan");
                            assert_eq!(
                                scanned.iter().map(Entry::id).collect::<Vec<_>>(),
                                [child_id],
                            );
                            *captured.lock().await = None;
                            let done: Box<dyn std::any::Any + Send> = Box::new(());
                            Ok(done)
                        })
                    },
                ),
                &background_context(),
            )
            .await
            .expect("mutate");
    }
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn rejects_pending_assistant_entries_at_the_durable_session_write_boundary() {
    let storage = Arc::new(InstrumentedStorage::new(memory_storage()));
    let session = storage_backed_session(storage.clone());
    let pending = assistant_message("pending", "");

    let rejected = commit_session(
        &session,
        vec![Write::Entry(Box::new(session_writes::insert_entry(
            NewEntry::Message {
                id: ENTRY_ID.to_owned(),
                parent_id: None,
                body: Box::new(MessageEntry {
                    message: pending,
                    terminate: None,
                }),
            },
        )))],
    )
    .await;
    assert_eq!(
        rejected.expect_err("pending rejected"),
        SessionError::PendingAssistantMessage,
    );
    assert!(storage.get_commit_attempts().is_empty());
    assert!(
        session
            .get_entries(vec![ENTRY_ID.to_owned()], &background_context())
            .await
            .expect("entries")
            .is_empty(),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn trusts_typed_custom_messages_without_repository_schema_registration() {
    let storage = memory_storage();
    let session = storage_backed_session(storage);
    let message = serde_json::from_value::<pi_agent_core::types::AgentMessage>(serde_json::json!({
        "role": "custom",
        "customType": "notice",
        "content": "maintenance",
        "display": true,
        "timestamp": NOW,
    }))
    .expect("custom message wire");
    let entry = NewEntry::Message {
        id: ENTRY_ID.to_owned(),
        parent_id: None,
        body: Box::new(MessageEntry {
            message: message.clone(),
            terminate: None,
        }),
    };

    let result = commit_session(
        &session,
        vec![Write::Entry(Box::new(session_writes::insert_entry(
            entry.clone(),
        )))],
    )
    .await
    .expect("commit");

    let stored = session
        .get_entries(vec![ENTRY_ID.to_owned()], &background_context())
        .await
        .expect("entries")
        .get(ENTRY_ID)
        .cloned()
        .expect("entry");
    assert_eq!(
        stored,
        Entry::Message {
            id: ENTRY_ID.to_owned(),
            parent_id: None,
            seq: result.first_seq,
            timestamp: result.timestamp,
            body: Box::new(MessageEntry {
                message,
                terminate: None,
            }),
        },
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn serializes_mutations_permits_one_commit_attempt_and_invalidates_the_mutator() {
    let storage = Arc::new(InstrumentedStorage::new(memory_storage()));
    let session = storage_backed_session(storage.clone());
    let captured: Arc<tokio::sync::Mutex<Option<Box<dyn SessionMutator>>>> =
        Arc::new(tokio::sync::Mutex::new(None));

    let captured_for_callback = captured.clone();
    session
        .mutate(
            Box::new(
                move |mutator: &dyn SessionMutator,
                      context|
                      -> pi_ai::types::BoxedFuture<
                    '_,
                    Result<Box<dyn std::any::Any + Send>, SessionError>,
                > {
                    let captured = captured_for_callback.clone();
                    Box::pin(async move {
                        *captured.lock().await = None;
                        assert!(
                            mutator
                                .get_value(&stored_values::session_name().address, context)
                                .await
                                .expect("read")
                                .is_none(),
                        );
                        mutator
                            .commit(
                                vec![Write::ValueSet(
                                    set_value_write(
                                        &stored_values::session_name(),
                                        "committed".to_owned(),
                                    )
                                    .expect("write"),
                                )],
                                context,
                            )
                            .await
                            .expect("commit");
                        let rejected = mutator.commit(Vec::new(), context).await;
                        assert!(
                            rejected
                                .expect_err("second attempt")
                                .to_string()
                                .contains("commit already attempted"),
                        );
                        let done: Box<dyn std::any::Any + Send> = Box::new(());
                        Ok(done)
                    })
                },
            ),
            &background_context(),
        )
        .await
        .expect("mutate");

    assert_eq!(storage.get_commit_attempts().len(), 1);
    assert_eq!(
        session.get_name(&background_context()).await.expect("name"),
        Some("committed".to_owned()),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn consumes_the_commit_guard_when_the_first_commit_fails() {
    let storage = Arc::new(InstrumentedStorage::new(memory_storage()));
    let session = storage_backed_session(storage.clone());
    let transaction = vec![Write::Entry(Box::new(session_writes::insert_entry(
        NewEntry::Custom {
            id: ENTRY_ID.to_owned(),
            parent_id: Some("missing".to_owned()),
            body: CustomEntryBody {
                custom_type: "note".to_owned(),
                data: None,
            },
        },
    )))];

    session
        .mutate(
            Box::new(
                move |mutator: &dyn SessionMutator,
                      context|
                      -> pi_ai::types::BoxedFuture<
                    '_,
                    Result<Box<dyn std::any::Any + Send>, SessionError>,
                > {
                    let transaction = transaction.clone();
                    Box::pin(async move {
                        let rejected = mutator.commit(transaction, context).await;
                        assert!(
                            rejected
                                .expect_err("missing parent")
                                .to_string()
                                .contains("Missing parent entry"),
                        );
                        let rejected = mutator.commit(Vec::new(), context).await;
                        assert!(
                            rejected
                                .expect_err("second attempt")
                                .to_string()
                                .contains("commit already attempted"),
                        );
                        let done: Box<dyn std::any::Any + Send> = Box::new(());
                        Ok(done)
                    })
                },
            ),
            &background_context(),
        )
        .await
        .expect("mutate");
    assert_eq!(storage.get_commit_attempts().len(), 1);
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn mints_distinct_follower_ids_with_the_leader_timestamp() {
    let session = storage_backed_session(memory_storage());
    let leader_timestamp = 0x0123_4567_89ab_u64;
    let leader = session
        .id_generator()
        .next(Some(i64::try_from(leader_timestamp).expect("timestamp")));
    let followers = [
        session
            .id_generator()
            .next(Some(i64::try_from(leader_timestamp).expect("timestamp"))),
        session
            .id_generator()
            .next(Some(i64::try_from(leader_timestamp).expect("timestamp"))),
    ];

    let ids = [leader, followers[0].clone(), followers[1].clone()];
    assert_eq!(
        ids.iter()
            .map(|id| decode_timestamp(id))
            .collect::<Vec<_>>(),
        [leader_timestamp, leader_timestamp, leader_timestamp],
    );
    let distinct: std::collections::BTreeSet<String> = ids.into_iter().collect();
    assert_eq!(distinct.len(), 3);
    session.close(&background_context()).await.expect("close");
}

/// The deterministic id generator the injected-generator test pins, upstream's
/// `{ next: (timestampMs) => ... }` object.
struct CountingIdGenerator {
    next_id: std::sync::atomic::AtomicUsize,
}

impl IdGenerator for CountingIdGenerator {
    fn next(&self, timestamp_ms: Option<i64>) -> String {
        let id = self.next_id.fetch_add(1, Ordering::AcqRel) + 1;
        format!(
            "{}:{id}",
            timestamp_ms.map_or_else(|| "now".to_owned(), |ts| ts.to_string())
        )
    }
}

#[tokio::test]
async fn accepts_an_injected_id_generator_for_deterministic_execution_tests() {
    let id_generator = Arc::new(CountingIdGenerator {
        next_id: std::sync::atomic::AtomicUsize::new(0),
    });
    let session = StorageBackedSession::new(
        metadata(),
        memory_storage(),
        StorageBackedSessionOptions {
            id_generator: Some(id_generator.clone()),
            ..Default::default()
        },
    );

    assert_eq!(session.id_generator().next(Some(7)), "7:1");
    assert_eq!(session.id_generator().next(None), "now:2");
    session.close(&background_context()).await.expect("close");
}

/// Whether the id reads as a uuidv7, upstream's uuid regex; the version
/// nibble is 7 and the variant nibble is one of 8/9/a/b.
fn is_uuid_v7(id: &str) -> bool {
    let hex: String = id.chars().filter(|c| *c != '-').collect();
    if id.len() != 36
        || id.as_bytes()[8] != b'-'
        || id.as_bytes()[13] != b'-'
        || id.as_bytes()[18] != b'-'
        || id.as_bytes()[23] != b'-'
        || hex.len() != 32
        || !hex.chars().all(|c| c.is_ascii_hexdigit())
    {
        return false;
    }
    let version = hex.chars().nth(12).expect("version nibble");
    let variant = hex.chars().nth(16).expect("variant nibble");
    version == '7' && matches!(variant, '8' | '9' | 'a' | 'b')
}

#[tokio::test]
async fn exposes_metadata_directly_and_the_shared_uuidv7_id_generator() {
    let source_metadata = metadata();
    let session = storage_backed_session(memory_storage());
    assert_eq!(session.metadata(), &source_metadata);
    assert!(is_uuid_v7(&session.id_generator().next(None)));
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn closes_idempotently_and_rejects_operations_not_admitted_before_close() {
    let storage = memory_storage();
    let session = storage_backed_session(storage);
    let (first_close, second_close) = (
        session.close(&background_context()),
        session.close(&background_context()),
    );
    let (first_closed, second_closed) = tokio::join!(first_close, second_close);
    assert!(first_closed.is_ok() && second_closed.is_ok());

    let rejected = session
        .mutate(
            StorageBackedSession::commit_writes_callback(Vec::new()),
            &background_context(),
        )
        .await;
    assert!(
        rejected
            .expect_err("closed session")
            .to_string()
            .contains("Session is closed"),
    );
    let rejected = session.get_entries(Vec::new(), &background_context()).await;
    assert!(
        rejected
            .expect_err("closed session")
            .to_string()
            .contains("Session is closed"),
    );
    let rejected = session
        .get_value(
            &stored_values::session_name().address,
            &background_context(),
        )
        .await;
    assert!(
        rejected
            .expect_err("closed session")
            .to_string()
            .contains("Session is closed"),
    );
    let rejected = session
        .scan_values(
            &stored_values::session_name().address,
            &background_context(),
        )
        .await;
    assert!(
        rejected
            .expect_err("closed session")
            .to_string()
            .contains("Session is closed"),
    );
    let rejected = session
        .scan_branch(
            &StorageBranchScan {
                start: ENTRY_ID.to_owned(),
                ..Default::default()
            },
            &background_context(),
        )
        .await;
    assert!(
        rejected
            .expect_err("closed session")
            .to_string()
            .contains("Session is closed"),
    );
}

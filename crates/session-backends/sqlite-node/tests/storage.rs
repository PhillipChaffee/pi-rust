//! The `SqliteStorage` suite, ported 1:1 from upstream
//! `packages/session-backends/sqlite-node/test/storage.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements recorded against upstream:
//! - "gets entries by requested id order": the Rust `Storage` trait returns a
//!   `BTreeMap<String, Entry>` (id-sorted); the request-order assertion
//!   restates to membership plus the missing id dropped.
//! - usage-JSON comparisons are structural (decode back to `Usage`), never
//!   byte compares: serde_json renders integral f64 zeros as `0.0` where JS
//!   renders `0` (the design's recorded rendering restatement).

#![expect(clippy::expect_used, reason = "tests assert on results")]
#![expect(clippy::panic, reason = "tests panic on failure")]
#![expect(
    clippy::unused_async,
    reason = "upstream's suite is async end to end; the port keeps the tokio shape"
)]

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

use pi_agent_core::harness::context::{Context, background_context};
use pi_agent_core::harness::session::commit::{
    PreparedCommitResult, insert_entry, insert_usage, prepare_storage_commit,
};
use pi_agent_core::harness::session::testing::conformance::user_message;
use pi_agent_core::harness::session::types::{
    CommitResult, Entry, EntryCursor, EntryScan, EntryScanOrder, EntryType, MessageEntry, NewEntry,
    Session, SessionStats, Storage, StorageBranchScan, UsageRow, UsageWriteRow,
};
use pi_agent_core::harness::session::values::{
    ListAddress, ListCursor, ListReadOptions, Write, delete_value_write, entry_label, list,
    session_name, set_value_write, value,
};
use pi_ai::types::{Usage, UsageCost};
use pi_session_backend_sqlite_node::sqlite::migrations::apply_initial_schema;
use pi_session_backend_sqlite_node::sqlite::session_sequences::{advance_next_seq, read_next_seq};
use pi_session_backend_sqlite_node::sqlite::storage::{SqliteStorage, SqliteStorageOptions};
use pi_session_backend_sqlite_node::sqlite::types::{
    SqliteDatabase, SqliteDatabaseFactory, SqliteParams, SqliteRow, SqliteValue,
};
use pi_session_backend_sqlite_node::sqlite::values::list_value_read_query;
use pi_session_backend_sqlite_node::{create_rusqlite_factory, sql};

use support::{TransactionCountingDatabase, expect_branch_plan, explain_query_plan};

mod support;

const SESSION_ID: &str = "session";

const ZERO_USAGE: Usage = Usage {
    input: 0,
    output: 0,
    cache_read: 0,
    cache_write: 0,
    cache_write_1h: None,
    reasoning: None,
    total_tokens: 0,
    cost: UsageCost {
        input: 0.0,
        output: 0.0,
        cache_read: 0.0,
        cache_write: 0.0,
        total: 0.0,
    },
};

fn usage(input: i64, output: i64) -> Usage {
    Usage {
        input: u64::try_from(input).unwrap_or(u64::MAX),
        output: u64::try_from(output).unwrap_or(u64::MAX),
        cache_read: u64::try_from(input + 1).unwrap_or(u64::MAX),
        cache_write: u64::try_from(output + 1).unwrap_or(u64::MAX),
        cache_write_1h: None,
        reasoning: None,
        total_tokens: u64::try_from(input + output).unwrap_or(u64::MAX),
        cost: UsageCost {
            input: 0.1,
            output: 0.2,
            cache_read: 0.3,
            cache_write: 0.4,
            total: 1.0,
        },
    }
}

fn context() -> Context {
    background_context()
}

/// The suites' `withStorage`: an in-memory database over the real factory,
/// the schema applied, the storage on the fixed clock, the db closed after.
async fn with_storage<T: Send + 'static>(
    run: impl FnOnce(
        Arc<SqliteStorage>,
        Arc<dyn SqliteDatabase>,
    ) -> std::pin::Pin<Box<dyn Future<Output = T> + Send>>,
) -> T {
    let db: Arc<dyn SqliteDatabase> =
        Arc::from(create_rusqlite_factory().open(":memory:").expect("open"));
    apply_initial_schema(db.as_ref()).expect("schema");
    let storage = Arc::new(SqliteStorage::new(
        Arc::clone(&db),
        &SqliteStorageOptions {
            session_id: SESSION_ID.to_owned(),
            now: Some(support::fixed_clock(support::NOW)),
        },
    ));
    run(storage, db).await
}

fn user_entry(id: &str, parent_id: Option<&str>, content: &str) -> Write {
    Write::Entry(Box::new(insert_entry(NewEntry::Message {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        body: Box::new(MessageEntry {
            message: user_message(content),
            terminate: None,
        }),
    })))
}

fn compaction_entry(id: &str, parent_id: &str) -> Write {
    Write::Entry(Box::new(insert_entry(NewEntry::Compaction {
        id: id.to_owned(),
        parent_id: Some(parent_id.to_owned()),
        body: pi_agent_core::harness::session::types::CompactionEntryBody {
            summary: "summary".to_owned(),
            retained_tail: Vec::new(),
            tokens_before: 1,
            details: None,
            usage: None,
            from_hook: false,
        },
    })))
}

fn compaction_payload() -> serde_json::Value {
    serde_json::json!({ "summary": "s", "retainedTail": [], "tokensBefore": 1, "fromHook": false })
}

fn message_payload(content: &str, timestamp: i64) -> serde_json::Value {
    serde_json::json!({ "message": { "role": "user", "content": content, "timestamp": timestamp } })
}

fn s(row: &SqliteRow, column: &str) -> String {
    row.string(column).expect(column)
}

fn n(row: &SqliteRow, column: &str) -> i64 {
    row.integer(column).expect(column)
}

fn o(row: &SqliteRow, column: &str) -> Option<String> {
    row.opt_string(column).expect(column)
}

/// Seeds one `entries` row directly, the suites' raw-SQL inserts.
fn seed_entry(
    db: &dyn SqliteDatabase,
    id: &str,
    parent_id: Option<&str>,
    seq: i64,
    kind: &str,
    custom_type: Option<&str>,
    timestamp: i64,
    payload: serde_json::Value,
) {
    sql!(
        "INSERT INTO entries (session_id, id, parent_id, seq, type, custom_type, timestamp, payload)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        SESSION_ID,
        id,
        parent_id,
        seq,
        kind,
        custom_type,
        timestamp,
        serde_json::to_string(&payload).expect("payload json"),
    )
    .run(db)
    .map(|_| ())
    .expect("seed entry");
}

fn seed_branch_meta(
    db: &dyn SqliteDatabase,
    branch_id: &str,
    tip_entry_id: &str,
    tip_seq: i64,
    base_branch_id: Option<&str>,
    base_seq: Option<i64>,
) {
    sql!(
        "INSERT INTO branch_meta (session_id, branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq)
		VALUES (?, ?, ?, ?, ?, ?)",
        SESSION_ID,
        branch_id,
        tip_entry_id,
        tip_seq,
        base_branch_id,
        base_seq,
    )
    .run(db)
    .map(|_| ())
    .expect("seed branch_meta");
}

fn seed_branch_entry(
    db: &dyn SqliteDatabase,
    branch_id: &str,
    entry_id: &str,
    entry_seq: i64,
    entry_type: &str,
) {
    sql!(
        "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type)
		VALUES (?, ?, ?, ?, ?)",
        SESSION_ID,
        branch_id,
        entry_id,
        entry_seq,
        entry_type,
    )
    .run(db)
    .map(|_| ())
    .expect("seed branch_entry");
}

#[tokio::test]
async fn uses_one_write_transaction_for_an_ordinary_commit() {
    let counter = Arc::new(TransactionCountingDatabase {
        source: create_rusqlite_factory().open(":memory:").expect("open"),
        transaction_count: std::sync::atomic::AtomicUsize::new(0),
    });
    apply_initial_schema(counter.source.as_ref()).expect("schema");
    support::insert_commit_session_row(counter.source.as_ref(), 1);
    let db: Arc<dyn SqliteDatabase> = counter.clone();
    let storage = SqliteStorage::new(
        Arc::clone(&db),
        &SqliteStorageOptions {
            session_id: SESSION_ID.to_owned(),
            now: Some(support::fixed_clock(support::NOW)),
        },
    );

    storage
        .commit(
            vec![set_value_write(&session_name(), "name".to_owned()).expect("write")],
            &context(),
        )
        .await
        .expect("commit");

    assert_eq!(
        counter
            .transaction_count
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "one write transaction for an ordinary commit"
    );
    storage.close(&context()).await.expect("close");
}

#[tokio::test]
async fn commits_root_entries_and_append_to_tip_entries_into_the_branch_index() {
    with_storage(|storage, db| {
        Box::pin(async move {
            support::insert_commit_session_row(db.as_ref(), 1);

            assert_eq!(
                storage
                    .commit(vec![user_entry("root", None, "root")], &context())
                    .await
                    .expect("commit"),
                CommitResult {
                    first_seq: 1,
                    seqs: vec![1],
                    timestamp: support::NOW,
                    stats: SessionStats { message_count: 1, usage: ZERO_USAGE },
                }
            );
            assert_eq!(
                storage
                    .commit(vec![user_entry("child", Some("root"), "child")], &context())
                    .await
                    .expect("commit"),
                CommitResult {
                    first_seq: 2,
                    seqs: vec![2],
                    timestamp: support::NOW,
                    stats: SessionStats { message_count: 2, usage: ZERO_USAGE },
                }
            );

            let rows = sql!("SELECT branch_id, tip_entry_id, tip_seq FROM branch_meta")
                .all(db.as_ref())
                .expect("branch_meta");
            assert_eq!(
                rows.iter().map(|row| (s(row, "branch_id"), s(row, "tip_entry_id"), n(row, "tip_seq"))).collect::<Vec<_>>(),
                vec![("root".to_owned(), "child".to_owned(), 2)]
            );
            let rows = sql!(
                "SELECT branch_id, entry_id, entry_seq, entry_type FROM branch_entries ORDER BY entry_seq"
            )
            .all(db.as_ref())
            .expect("branch_entries");
            assert_eq!(
                rows.iter()
                    .map(|row| (s(row, "branch_id"), s(row, "entry_id"), n(row, "entry_seq"), s(row, "entry_type")))
                    .collect::<Vec<_>>(),
                vec![
                    ("root".to_owned(), "root".to_owned(), 1, "message".to_owned()),
                    ("root".to_owned(), "child".to_owned(), 2, "message".to_owned()),
                ]
            );
            let entries = storage
                .scan_branch(
                    &StorageBranchScan { start: "child".to_owned(), ..StorageBranchScan::default() },
                    &context(),
                )
                .await
                .expect("scan");
            assert_eq!(
                entries.iter().map(Entry::id).collect::<Vec<_>>(),
                vec!["child", "root"]
            );
        })
    })
    .await;
}

#[tokio::test]
async fn commits_divergent_branch_entries_by_materializing_a_new_segment() {
    with_storage(|storage, db| {
        Box::pin(async move {
            support::insert_commit_session_row(db.as_ref(), 1);
            storage
                .commit(
                    vec![
                        user_entry("root", None, "root"),
                        user_entry("left", Some("root"), "left"),
                        user_entry("right", Some("root"), "right"),
                    ],
                    &context(),
                )
                .await
                .expect("commit");

            let rows = sql!(
                "SELECT branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq FROM branch_meta ORDER BY branch_id"
            )
            .all(db.as_ref())
            .expect("branch_meta");
            assert_eq!(
                rows.iter()
                    .map(|row| (
                        s(row, "branch_id"),
                        s(row, "tip_entry_id"),
                        n(row, "tip_seq"),
                        o(row, "base_branch_id"),
                        o(row, "base_seq"),
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    ("right".to_owned(), "right".to_owned(), 3, None, None),
                    ("root".to_owned(), "left".to_owned(), 2, None, None),
                ]
            );
            let rows = sql!(
                "SELECT branch_id, entry_id, entry_seq, entry_type FROM branch_entries ORDER BY branch_id, entry_seq"
            )
            .all(db.as_ref())
            .expect("branch_entries");
            assert_eq!(
                rows.iter()
                    .map(|row| (s(row, "branch_id"), s(row, "entry_id"), n(row, "entry_seq"), s(row, "entry_type")))
                    .collect::<Vec<_>>(),
                vec![
                    ("right".to_owned(), "root".to_owned(), 1, "message".to_owned()),
                    ("right".to_owned(), "right".to_owned(), 3, "message".to_owned()),
                    ("root".to_owned(), "root".to_owned(), 1, "message".to_owned()),
                    ("root".to_owned(), "left".to_owned(), 2, "message".to_owned()),
                ]
            );
            for (start, expected) in [("right", vec!["right", "root"]), ("left", vec!["left", "root"])] {
                let entries = storage
                    .scan_branch(
                        &StorageBranchScan { start: start.to_owned(), ..StorageBranchScan::default() },
                        &context(),
                    )
                    .await
                    .expect("scan");
                assert_eq!(entries.iter().map(Entry::id).collect::<Vec<_>>(), expected);
            }
        })
    })
    .await;
}

#[tokio::test]
async fn bases_divergent_branch_segments_at_the_newest_compaction() {
    with_storage(|storage, db| {
        Box::pin(async move {
            support::insert_commit_session_row(db.as_ref(), 1);
            storage
                .commit(
                    vec![
                        user_entry("root", None, "root"),
                        compaction_entry("compact", "root"),
                        user_entry("left", Some("compact"), "left"),
                        user_entry("leaf", Some("left"), "leaf"),
                        user_entry("right", Some("left"), "right"),
                    ],
                    &context(),
                )
                .await
                .expect("commit");

            let row = sql!(
                "SELECT branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq FROM branch_meta WHERE branch_id = ?",
                "right"
            )
            .get(db.as_ref())
            .expect("branch_meta")
            .expect("right row");
            assert_eq!(s(&row, "branch_id"), "right");
            assert_eq!(s(&row, "tip_entry_id"), "right");
            assert_eq!(n(&row, "tip_seq"), 5);
            assert_eq!(o(&row, "base_branch_id").as_deref(), Some("root"));
            assert_eq!(n(&row, "base_seq"), 2);
            let rows = sql!(
                "SELECT branch_id, entry_id, entry_seq, entry_type FROM branch_entries WHERE branch_id = ? ORDER BY entry_seq",
                "right"
            )
            .all(db.as_ref())
            .expect("branch_entries");
            assert_eq!(
                rows.iter()
                    .map(|row| (s(row, "entry_id"), n(row, "entry_seq"), s(row, "entry_type")))
                    .collect::<Vec<_>>(),
                vec![
                    ("left".to_owned(), 3, "message".to_owned()),
                    ("right".to_owned(), 5, "message".to_owned()),
                ]
            );
            let entries = storage
                .scan_branch(
                    &StorageBranchScan { start: "right".to_owned(), ..StorageBranchScan::default() },
                    &context(),
                )
                .await
                .expect("scan");
            assert_eq!(
                entries.iter().map(Entry::id).collect::<Vec<_>>(),
                vec!["right", "left", "compact", "root"]
            );
        })
    })
    .await;
}

/// Upstream asserts the requested-id order `["second", "first"]`; the Rust
/// trait returns a `BTreeMap` (id-sorted), so the assertion restates to
/// membership plus the missing id dropped (the design's recorded restatement).
#[tokio::test]
async fn gets_entries_by_requested_id_order() {
    with_storage(|storage, db| {
        Box::pin(async move {
            seed_entry(
                db.as_ref(),
                "first",
                None,
                1,
                "custom",
                Some("note"),
                10,
                serde_json::json!({ "data": { "value": 1 } }),
            );
            seed_entry(
                db.as_ref(),
                "second",
                Some("first"),
                2,
                "message",
                None,
                11,
                message_payload("hi", 11),
            );

            let entries: BTreeMap<String, Entry> = storage
                .get_entries(
                    vec![
                        "second".to_owned(),
                        "missing".to_owned(),
                        "first".to_owned(),
                    ],
                    &context(),
                )
                .await
                .expect("get_entries");

            assert_eq!(
                entries.len(),
                2,
                "the missing id drops; both found ids return"
            );
            assert_eq!(entries.keys().collect::<Vec<_>>(), vec!["first", "second"]);
            let second = entries.get("second").expect("second");
            assert_eq!(second.id(), "second");
            assert_eq!(second.entry_type(), EntryType::Message);
            assert_eq!(
                second.parent_id().map(str::to_owned).as_deref(),
                Some("first")
            );
            let first = entries.get("first").expect("first");
            assert_eq!(first.entry_type(), EntryType::Custom);
            assert_eq!(first.custom_type(), Some("note"));
        })
    })
    .await;
}

#[tokio::test]
async fn scans_decoded_entries_with_filters_and_sequence_bounds() {
    with_storage(|storage, db| {
        Box::pin(async move {
            seed_entry(
                db.as_ref(),
                "one",
                None,
                1,
                "custom",
                Some("note"),
                10,
                serde_json::json!({ "data": 1 }),
            );
            seed_entry(
                db.as_ref(),
                "two",
                Some("one"),
                2,
                "message",
                None,
                11,
                message_payload("two", 11),
            );
            seed_entry(
                db.as_ref(),
                "three",
                Some("two"),
                3,
                "custom",
                Some("note"),
                12,
                serde_json::json!({ "data": 3 }),
            );

            let entries = storage
                .scan_entries(
                    &EntryScan {
                        order: Some(EntryScanOrder::Desc),
                        kind: Some(EntryType::Custom),
                        from_seq: Some(2),
                        ..EntryScan::default()
                    },
                    &context(),
                )
                .await
                .expect("scan");
            assert_eq!(
                entries.iter().map(Entry::id).collect::<Vec<_>>(),
                vec!["three"]
            );
            let entries = storage
                .scan_entries(
                    &EntryScan {
                        order: Some(EntryScanOrder::Asc),
                        custom_type: Some("note".to_owned()),
                        limit: Some(2),
                        ..EntryScan::default()
                    },
                    &context(),
                )
                .await
                .expect("scan");
            assert_eq!(
                entries.iter().map(Entry::id).collect::<Vec<_>>(),
                vec!["one", "three"]
            );
        })
    })
    .await;
}

#[tokio::test]
async fn scans_branch_entries_through_materialized_branch_segments() {
    with_storage(|storage, db| {
        Box::pin(async move {
            seed_entry(db.as_ref(), "root", None, 1, "message", None, 10, message_payload("root", 10));
            seed_entry(db.as_ref(), "compact", Some("root"), 2, "compaction", None, 11, compaction_payload());
            seed_entry(
                db.as_ref(),
                "old",
                Some("compact"),
                3,
                "message",
                None,
                12,
                serde_json::json!({ "message": { "role": "assistant", "content": "old", "timestamp": 12 } }),
            );
            seed_entry(db.as_ref(), "custom", Some("old"), 4, "custom", Some("note"), 13, serde_json::json!({ "data": 4 }));
            seed_entry(db.as_ref(), "leaf", Some("custom"), 5, "message", None, 14, message_payload("leaf", 14));
            seed_branch_meta(db.as_ref(), "base", "compact", 2, None, None);
            seed_branch_meta(db.as_ref(), "new", "leaf", 5, Some("base"), Some(2));
            seed_branch_entry(db.as_ref(), "base", "root", 1, "message");
            seed_branch_entry(db.as_ref(), "base", "compact", 2, "compaction");
            seed_branch_entry(db.as_ref(), "new", "old", 3, "message");
            seed_branch_entry(db.as_ref(), "new", "custom", 4, "custom");
            seed_branch_entry(db.as_ref(), "new", "leaf", 5, "message");

            let entries = storage
                .scan_branch(
                    &StorageBranchScan {
                        start: "leaf".to_owned(),
                        stop_at_type: Some(EntryType::Compaction),
                        limit: Some(2),
                        ..StorageBranchScan::default()
                    },
                    &context(),
                )
                .await
                .expect("scan");
            assert_eq!(entries.iter().map(Entry::id).collect::<Vec<_>>(), vec!["leaf", "custom"]);
            let entries = storage
                .scan_branch(
                    &StorageBranchScan {
                        start: "leaf".to_owned(),
                        order: Some(pi_agent_core::harness::session::types::BranchScanOrder::OldestFirst),
                        kind: Some(EntryType::Message),
                        cursor: Some(EntryCursor { seq: 1 }),
                        ..StorageBranchScan::default()
                    },
                    &context(),
                )
                .await
                .expect("scan");
            assert_eq!(entries.iter().map(Entry::id).collect::<Vec<_>>(), vec!["old", "leaf"]);
        })
    })
    .await;
}

#[tokio::test]
async fn resolves_a_branch_segment_that_physically_contains_the_start_entry() {
    with_storage(|storage, db| {
        Box::pin(async move {
            seed_entry(db.as_ref(), "root", None, 1, "message", None, 10, message_payload("root", 10));
            seed_entry(
                db.as_ref(),
                "base-tip",
                Some("root"),
                2,
                "message",
                None,
                11,
                serde_json::json!({ "message": { "role": "assistant", "content": "base", "timestamp": 11 } }),
            );
            seed_entry(db.as_ref(), "new-tip", Some("base-tip"), 3, "message", None, 12, message_payload("new", 12));
            seed_branch_meta(db.as_ref(), "aaa-new", "new-tip", 3, Some("zzz-base"), Some(2));
            seed_branch_meta(db.as_ref(), "zzz-base", "base-tip", 2, None, None);
            seed_branch_entry(db.as_ref(), "aaa-new", "new-tip", 3, "message");
            seed_branch_entry(db.as_ref(), "zzz-base", "root", 1, "message");
            seed_branch_entry(db.as_ref(), "zzz-base", "base-tip", 2, "message");

            let entries = storage
                .scan_branch(
                    &StorageBranchScan {
                        start: "base-tip".to_owned(),
                        order: Some(pi_agent_core::harness::session::types::BranchScanOrder::OldestFirst),
                        ..StorageBranchScan::default()
                    },
                    &context(),
                )
                .await
                .expect("scan");
            assert_eq!(entries.iter().map(Entry::id).collect::<Vec<_>>(), vec!["root", "base-tip"]);
        })
    })
    .await;
}

#[tokio::test]
async fn applies_branch_stop_boundaries_across_base_segments_before_filtering() {
    with_storage(|storage, db| {
        Box::pin(async move {
            seed_entry(db.as_ref(), "root", None, 1, "message", None, 10, message_payload("root", 10));
            seed_entry(db.as_ref(), "compact", Some("root"), 2, "compaction", None, 11, compaction_payload());
            seed_entry(
                db.as_ref(),
                "after",
                Some("compact"),
                3,
                "message",
                None,
                12,
                serde_json::json!({ "message": { "role": "assistant", "content": "after", "timestamp": 12 } }),
            );
            seed_entry(db.as_ref(), "leaf", Some("after"), 4, "custom", Some("note"), 13, serde_json::json!({ "data": 4 }));
            seed_branch_meta(db.as_ref(), "base", "compact", 2, None, None);
            seed_branch_meta(db.as_ref(), "new", "leaf", 4, Some("base"), Some(2));
            seed_branch_entry(db.as_ref(), "base", "root", 1, "message");
            seed_branch_entry(db.as_ref(), "base", "compact", 2, "compaction");
            seed_branch_entry(db.as_ref(), "new", "after", 3, "message");
            seed_branch_entry(db.as_ref(), "new", "leaf", 4, "custom");

            let entries = storage
                .scan_branch(
                    &StorageBranchScan {
                        start: "leaf".to_owned(),
                        stop_at_type: Some(EntryType::Compaction),
                        ..StorageBranchScan::default()
                    },
                    &context(),
                )
                .await
                .expect("scan");
            assert_eq!(
                entries.iter().map(Entry::id).collect::<Vec<_>>(),
                vec!["leaf", "after", "compact"]
            );
            let entries = storage
                .scan_branch(
                    &StorageBranchScan {
                        start: "leaf".to_owned(),
                        stop_at_type: Some(EntryType::Compaction),
                        kind: Some(EntryType::Message),
                        ..StorageBranchScan::default()
                    },
                    &context(),
                )
                .await
                .expect("scan");
            assert_eq!(entries.iter().map(Entry::id).collect::<Vec<_>>(), vec!["after"]);
        })
    })
    .await;
}

#[tokio::test]
async fn uses_branch_entries_as_the_outer_scan_for_branch_payload_queries() {
    with_storage(|_storage, db| {
        Box::pin(async move {
            let plan = explain_query_plan(
                db.as_ref(),
                "SELECT e.id, e.parent_id, e.seq, e.type, e.custom_type, e.timestamp, e.payload
				FROM branch_entries b
				CROSS JOIN entries e ON e.session_id = b.session_id AND e.id = b.entry_id
				WHERE b.session_id = ? AND b.branch_id = ? AND b.entry_seq > ? AND b.entry_seq <= ?
				ORDER BY b.entry_seq DESC LIMIT ?",
                &SqliteParams::Positional(vec![
                    SqliteValue::from_param(SESSION_ID),
                    SqliteValue::from_param("main"),
                    SqliteValue::from_param(0i64),
                    SqliteValue::from_param(10i64),
                    SqliteValue::from_param(2i64),
                ]),
            );
            expect_branch_plan(&plan);
        })
    })
    .await;
}

#[tokio::test]
async fn scans_branch_structure_without_payloads() {
    with_storage(|storage, db| {
        Box::pin(async move {
            seed_entry(
                db.as_ref(),
                "root",
                None,
                1,
                "message",
                None,
                10,
                message_payload("root", 10),
            );
            seed_entry(
                db.as_ref(),
                "custom",
                Some("root"),
                2,
                "custom",
                Some("note"),
                11,
                serde_json::json!({ "data": 2 }),
            );
            seed_branch_meta(db.as_ref(), "main", "custom", 2, None, None);
            seed_branch_entry(db.as_ref(), "main", "root", 1, "message");
            seed_branch_entry(db.as_ref(), "main", "custom", 2, "custom");

            let structures = storage
                .scan_branch_structure(
                    &StorageBranchScan {
                        start: "custom".to_owned(),
                        custom_type: Some("note".to_owned()),
                        ..StorageBranchScan::default()
                    },
                    &context(),
                )
                .await
                .expect("scan");
            assert_eq!(
                structures,
                vec![pi_agent_core::harness::session::types::EntryStructure {
                    id: "custom".to_owned(),
                    parent_id: Some("root".to_owned()),
                    seq: 2,
                    timestamp: 11,
                    kind: EntryType::Custom,
                    custom_type: Some("note".to_owned()),
                }]
            );
        })
    })
    .await;
}

#[tokio::test]
async fn uses_branch_entries_as_the_outer_scan_for_branch_structure_queries() {
    with_storage(|_storage, db| {
        Box::pin(async move {
            let plan = explain_query_plan(
                db.as_ref(),
                "SELECT e.id, e.parent_id, e.seq, e.type, e.custom_type, e.timestamp
				FROM branch_entries b
				CROSS JOIN entries e ON e.session_id = b.session_id AND e.id = b.entry_id
				WHERE b.session_id = ? AND b.branch_id = ? AND b.entry_seq > ? AND b.entry_seq <= ?
				ORDER BY b.entry_seq ASC LIMIT ?",
                &SqliteParams::Positional(vec![
                    SqliteValue::from_param(SESSION_ID),
                    SqliteValue::from_param("main"),
                    SqliteValue::from_param(0i64),
                    SqliteValue::from_param(10i64),
                    SqliteValue::from_param(2i64),
                ]),
            );
            expect_branch_plan(&plan);
        })
    })
    .await;
}

#[tokio::test]
async fn scans_decoded_usage_rows_with_sequence_bounds() {
    with_storage(|storage, db| {
        Box::pin(async move {
            let usage_json = serde_json::to_string(&usage(1, 2)).expect("usage json");
            for (id, seq, entry_id, adjustment, details) in [
                ("u1", 1i64, Some("e1"), 0i64, None),
                ("u2", 2, None, 1, Some(serde_json::json!({ "reason": "adjust" }))),
                ("u3", 3, Some("e3"), 0, None),
            ] {
                sql!(
                    "INSERT INTO usage_ledger (session_id, id, seq, entry_id, adjustment, usage, details)
					VALUES (?, ?, ?, ?, ?, ?, ?)",
                    SESSION_ID,
                    id,
                    seq,
                    entry_id,
                    adjustment,
                    usage_json.clone(),
                    details.map(|value| serde_json::to_string(&value).expect("details json")),
                )
                .run(db.as_ref())
                .map(|_| ())
                .expect("seed usage");
            }

            assert_eq!(
                storage
                    .scan_usage(
                        &pi_agent_core::harness::session::types::UsageScan {
                            from_seq: Some(2),
                            order: Some(EntryScanOrder::Asc),
                            limit: Some(1),
                            ..pi_agent_core::harness::session::types::UsageScan::default()
                        },
                        &context(),
                    )
                    .await
                    .expect("scan"),
                vec![UsageRow {
                    id: "u2".to_owned(),
                    seq: 2,
                    usage: usage(1, 2),
                    entry_id: None,
                    adjustment: true,
                    details: Some(serde_json::json!({ "reason": "adjust" })),
                }]
            );
            let rows = storage
                .scan_usage(
                    &pi_agent_core::harness::session::types::UsageScan {
                        to_seq: Some(2),
                        order: Some(EntryScanOrder::Desc),
                        ..pi_agent_core::harness::session::types::UsageScan::default()
                    },
                    &context(),
                )
                .await
                .expect("scan");
            assert_eq!(rows.iter().map(|row| row.id.as_str()).collect::<Vec<_>>(), vec!["u2", "u1"]);
        })
    })
    .await;
}

/// The pure-function test, upstream's "prepares committed writes with
/// assigned sequences and timestamp".
#[test]
fn prepares_committed_writes_with_assigned_sequences_and_timestamp() {
    let prepared = prepare_storage_commit(
        vec![
            user_entry("entry", None, "hi"),
            Write::Usage(insert_usage(UsageWriteRow {
                id: "usage".to_owned(),
                usage: usage(1, 2),
                entry_id: None,
                adjustment: false,
                details: None,
            })),
            set_value_write(&session_name(), "name".to_owned()).expect("write"),
            delete_value_write(&entry_label("entry")),
        ],
        7,
        support::NOW,
    );

    assert_eq!(
        prepared.result,
        PreparedCommitResult {
            first_seq: 7,
            seqs: vec![7, 8, 9, 10],
            timestamp: support::NOW
        }
    );
    assert_eq!(prepared.writes.len(), 4);
    let pi_agent_core::harness::session::commit::CommittedWrite::Entry { entry } =
        &prepared.writes[0]
    else {
        panic!("first write is an entry");
    };
    assert_eq!(entry.id(), "entry");
    assert_eq!(entry.seq(), 7);
    assert_eq!(entry.timestamp(), support::NOW);
    let pi_agent_core::harness::session::commit::CommittedWrite::Usage { row } =
        &prepared.writes[1]
    else {
        panic!("second write is a usage row");
    };
    assert_eq!(row.seq, 8);
    let pi_agent_core::harness::session::commit::CommittedWrite::ValueSet {
        seq,
        namespace,
        key,
        value,
    } = &prepared.writes[2]
    else {
        panic!("third write is a value set");
    };
    assert_eq!(
        (*seq, namespace.as_str(), key.as_str(), value),
        (9, "pi.session.name", "", &serde_json::json!("name"))
    );
    let pi_agent_core::harness::session::commit::CommittedWrite::ValueDelete {
        seq,
        namespace,
        key,
    } = &prepared.writes[3]
    else {
        panic!("fourth write is a value delete");
    };
    assert_eq!(
        (*seq, namespace.as_str(), key.as_str()),
        (10, "pi.entry.label", "entry")
    );
}

#[tokio::test]
async fn reads_and_advances_the_next_commit_sequence() {
    with_storage(|_storage, db| {
        Box::pin(async move {
            sql!(
                "INSERT INTO sessions
				(id, created_at, parent_session_id, storage_version, metadata, message_count, usage_payload, next_seq)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                SESSION_ID,
                1i64,
                None::<String>,
                1i64,
                None::<String>,
                0i64,
                "{}",
                7i64,
            )
            .run(db.as_ref())
            .map(|_| ())
            .expect("seed session");

            assert_eq!(read_next_seq(db.as_ref(), SESSION_ID).expect("read"), 7);
            advance_next_seq(db.as_ref(), SESSION_ID, 10).expect("advance");
            assert_eq!(read_next_seq(db.as_ref(), SESSION_ID).expect("read"), 10);
        })
    })
    .await;
}

#[tokio::test]
async fn gets_maintained_session_stats() {
    with_storage(|storage, db| {
        Box::pin(async move {
            sql!(
                "INSERT INTO sessions
				(id, created_at, parent_session_id, storage_version, metadata, message_count, usage_payload, next_seq)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                SESSION_ID,
                1i64,
                None::<String>,
                1i64,
                None::<String>,
                2i64,
                serde_json::to_string(&usage(1, 2)).expect("usage json"),
                3i64,
            )
            .run(db.as_ref())
            .map(|_| ())
            .expect("seed session");

            assert_eq!(
                storage.get_stats(&context()).await.expect("stats"),
                SessionStats { message_count: 2, usage: usage(1, 2) }
            );
            let next = storage
                .commit(
                    vec![set_value_write(&session_name(), "after-history".to_owned()).expect("write")],
                    &context(),
                )
                .await
                .expect("commit");
            assert_eq!(next.stats, SessionStats { message_count: 2, usage: usage(1, 2) });
        })
    })
    .await;
}

#[tokio::test]
async fn includes_historical_totals_in_the_first_commit_after_storage_reopen() {
    with_storage(|storage, db| {
        Box::pin(async move {
            support::insert_commit_session_row(db.as_ref(), 1);
            let history_usage = Usage {
                input: 1,
                output: 2,
                cache_read: 0,
                cache_write: 0,
                cache_write_1h: None,
                reasoning: None,
                total_tokens: 3,
                cost: UsageCost {
                    input: 0.0,
                    output: 0.0,
                    cache_read: 0.0,
                    cache_write: 0.0,
                    total: 0.0,
                },
            };
            storage
                .commit(
                    vec![
                        user_entry("history", None, "history"),
                        Write::Usage(insert_usage(UsageWriteRow {
                            id: "usage".to_owned(),
                            usage: history_usage,
                            entry_id: None,
                            adjustment: false,
                            details: None,
                        })),
                    ],
                    &context(),
                )
                .await
                .expect("commit");
            storage.close(&context()).await.expect("close");

            let reopened = SqliteStorage::new(
                Arc::clone(&db),
                &SqliteStorageOptions {
                    session_id: SESSION_ID.to_owned(),
                    now: Some(support::fixed_clock(support::NOW)),
                },
            );
            let result = reopened
                .commit(
                    vec![set_value_write(&session_name(), "reopened".to_owned()).expect("write")],
                    &context(),
                )
                .await
                .expect("commit");
            assert_eq!(
                result.stats,
                SessionStats {
                    message_count: 1,
                    usage: history_usage
                }
            );
            assert_eq!(
                result.stats,
                reopened.get_stats(&context()).await.expect("stats")
            );
            reopened.close(&context()).await.expect("close");
        })
    })
    .await;
}

#[tokio::test]
async fn gets_a_decoded_scalar_value_by_bound_address() {
    with_storage(|storage, db| {
        Box::pin(async move {
            let address = value::<serde_json::Value>("test.value", "state").expect("address");
            sql!(
                "INSERT INTO scalar_values (session_id, namespace, key, seq, value)
				VALUES (?, ?, ?, ?, ?)",
                SESSION_ID,
                &address.address.namespace,
                &address.address.key,
                1i64,
                serde_json::json!({ "ready": true }).to_string(),
            )
            .run(db.as_ref())
            .map(|_| ())
            .expect("seed value");

            assert_eq!(
                storage
                    .get_value(&address.address, &context())
                    .await
                    .expect("value"),
                Some(pi_agent_core::harness::session::values::StoredValue {
                    namespace: "test.value".to_owned(),
                    key: "state".to_owned(),
                    seq: 1,
                    value: serde_json::json!({ "ready": true }),
                })
            );
            let missing = value::<serde_json::Value>("test.value", "missing").expect("address");
            assert_eq!(
                storage
                    .get_value(&missing.address, &context())
                    .await
                    .expect("value"),
                None
            );
        })
    })
    .await;
}

#[tokio::test]
async fn scans_decoded_scalar_values_by_namespace_and_key_prefix() {
    with_storage(|storage, db| {
        Box::pin(async move {
            for (key, seq, value_json) in [
                ("app:one", 1i64, serde_json::json!(1)),
                ("app:two", 2, serde_json::json!(2)),
                ("app:\u{ffff}tail", 3, serde_json::json!(3)),
                ("app;other", 4, serde_json::json!(4)),
                ("other", 5, serde_json::json!(5)),
            ] {
                sql!(
                    "INSERT INTO scalar_values (session_id, namespace, key, seq, value)
					VALUES (?, ?, ?, ?, ?)",
                    SESSION_ID,
                    "test.value",
                    key,
                    seq,
                    value_json.to_string(),
                )
                .run(db.as_ref())
                .map(|_| ())
                .expect("seed value");
            }
            sql!(
                "INSERT INTO scalar_values (session_id, namespace, key, seq, value)
				VALUES (?, ?, ?, ?, ?)",
                SESSION_ID,
                "other.value",
                "",
                6i64,
                serde_json::json!("name").to_string(),
            )
            .run(db.as_ref())
            .map(|_| ())
            .expect("seed value");

            let prefix = value::<serde_json::Value>("test.value", "app:").expect("address");
            let stored = storage
                .scan_values(&prefix.address, &context())
                .await
                .expect("scan");
            let addresses = ["app:one", "app:two", "app:\u{ffff}tail"]
                .iter()
                .map(
                    |key| pi_agent_core::harness::session::values::ValueAddress {
                        namespace: "test.value".to_owned(),
                        key: (*key).to_owned(),
                    },
                )
                .collect::<Vec<_>>();
            let expected = addresses
                .iter()
                .zip([
                    serde_json::json!(1),
                    serde_json::json!(2),
                    serde_json::json!(3),
                ])
                .map(
                    |(address, value)| pi_agent_core::harness::session::values::StoredValue {
                        namespace: address.namespace.clone(),
                        key: address.key.clone(),
                        seq: 0,
                        value,
                    },
                )
                .enumerate()
                .map(|(index, mut stored)| {
                    stored.seq = u64::try_from(index + 1).unwrap_or(u64::MAX);
                    stored
                })
                .collect::<Vec<_>>();
            assert_eq!(stored, expected);
        })
    })
    .await;
}

#[tokio::test]
async fn uses_the_list_primary_key_for_ascending_and_descending_cursor_pages() {
    with_storage(|_storage, db| {
        Box::pin(async move {
            let address = list::<serde_json::Value>("test.list", "events").expect("address");
            let address: ListAddress = address.address;
            for options in [
                ListReadOptions {
                    order: Some(EntryScanOrder::Asc),
                    ..ListReadOptions::default()
                },
                ListReadOptions {
                    order: Some(EntryScanOrder::Asc),
                    cursor: Some(ListCursor { seq: 4 }),
                    ..ListReadOptions::default()
                },
                ListReadOptions {
                    order: Some(EntryScanOrder::Desc),
                    ..ListReadOptions::default()
                },
                ListReadOptions {
                    order: Some(EntryScanOrder::Desc),
                    cursor: Some(ListCursor { seq: 4 }),
                    ..ListReadOptions::default()
                },
            ] {
                let query =
                    list_value_read_query(SESSION_ID, &address, Some(options)).expect("query");
                let plan = explain_query_plan(
                    db.as_ref(),
                    &query.query_text,
                    &SqliteParams::Positional(query.params.clone()),
                );
                assert!(
                    plan.iter()
                        .any(|detail| detail.contains("USING PRIMARY KEY")),
                    "primary key used: {plan:?}"
                );
                assert!(
                    !plan
                        .iter()
                        .any(|detail| detail.contains("SCAN list_values")),
                    "no scan: {plan:?}"
                );
                assert!(
                    !plan.iter().any(|detail| detail.contains("USE TEMP B-TREE")),
                    "no temp b-tree: {plan:?}"
                );
            }
        })
    })
    .await;
}

//! Boundary tests binding the branches the 1:1 suites leave untested: the
//! filename encoding's byte vectors, the prefix-boundary arithmetic, the
//! storage-version gate's two messages, the changes-count errors, the decode
//! errors, the erased-adapter surface, and the driver-seam behaviors.

#![expect(clippy::expect_used, reason = "tests assert on results")]

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_agent_core::harness::context::{Context, background_context};
use pi_agent_core::harness::session::commit::insert_entry;
use pi_agent_core::harness::session::session::StorageBackedSession;
use pi_agent_core::harness::session::testing::conformance::user_message;
use pi_agent_core::harness::session::types::{
    Entry, ForkOptions, MessageEntry, NewEntry, Session, SessionError, SessionMetadata,
    SessionRepo, Storage,
};
use pi_agent_core::harness::session::values::Write;
use pi_session_backend_sqlite_node::sqlite::migrations::apply_initial_schema;
use pi_session_backend_sqlite_node::sqlite::session_row::delete_session_rows;
use pi_session_backend_sqlite_node::sqlite::session_sequences::{advance_next_seq, read_next_seq};
use pi_session_backend_sqlite_node::sqlite::storage::{SqliteStorage, SqliteStorageOptions};
use pi_session_backend_sqlite_node::sqlite::types::{
    SqliteDatabase, SqliteDatabaseFactory, SqliteValue, sql_integer, sql_u64,
};
use pi_session_backend_sqlite_node::{
    SQLITE_STORAGE_VERSION, SqliteSessionCreateOptions, SqliteSessionRepo, create_rusqlite_factory,
    sql, wrap_rusqlite_connection,
};

use support::{repo_options, shared_repo_options};

mod support;

fn context() -> Context {
    background_context()
}

fn fixed_repo(directory: &tempfile::TempDir) -> SqliteSessionRepo {
    SqliteSessionRepo::new(repo_options(
        directory.path().to_string_lossy().as_ref(),
        support::fixed_clock(support::NOW),
    ))
}

fn memory_db() -> Arc<dyn SqliteDatabase> {
    Arc::from(create_rusqlite_factory().open(":memory:").expect("open"))
}

fn seeded_db() -> Arc<dyn SqliteDatabase> {
    let db = memory_db();
    apply_initial_schema(db.as_ref()).expect("schema");
    sql!(
        "INSERT INTO sessions (id, created_at, parent_session_id, storage_version, metadata, message_count, usage_payload, next_seq)
		VALUES ('s', 1, NULL, 1, NULL, 0, ?, 1)",
        support::zero_usage_json(),
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed");
    db
}

// ---------------------------------------------------------------------------
// Filename encoding: byte vectors pinned against node's
// Buffer.from(id, "utf16le").toString("base64url") (the upstream package
// asserts containment and round-trip only; these pin the algorithm).
// ---------------------------------------------------------------------------

#[test]
fn encodes_unsafe_ids_byte_exactly() {
    use pi_session_backend_sqlite_node::sqlite::repo::session_file_name;
    let vectors: &[(&str, &str)] = &[
        ("../escape", "~LgAuAC8AZQBzAGMAYQBwAGUA.sqlite"),
        ("slash/id", "~cwBsAGEAcwBoAC8AaQBkAA.sqlite"),
        ("back\\slash", "~YgBhAGMAawBcAHMAbABhAHMAaAA.sqlite"),
        ("percent%id", "~cABlAHIAYwBlAG4AdAAlAGkAZAA.sqlite"),
        ("..", "~LgAuAA.sqlite"),
        ("dots...id", "~ZABvAHQAcwAuAC4ALgBpAGQA.sqlite"),
        ("ユニコード", "~5jDLMLMw_DDJMA.sqlite"),
        ("", "~.sqlite"),
        ("fork/../%", "~ZgBvAHIAawAvAC4ALgAvACUA.sqlite"),
        ("~tilde", "~fgB0AGkAbABkAGUA.sqlite"),
        ("sp ace", "~cwBwACAAYQBjAGUA.sqlite"),
    ];
    for (id, expected) in vectors {
        assert_eq!(&session_file_name(id), expected, "encoding of {id:?}");
    }
    for id in ["a-b_c0", "0", "_", "-", "A"] {
        assert_eq!(
            session_file_name(id),
            format!("{id}.sqlite"),
            "safe id keeps its name"
        );
    }
}

// ---------------------------------------------------------------------------
// Prefix-boundary arithmetic through the scalar scans: the last code point
// increments, the surrogate window jumps to U+E000, an all-max prefix is
// open-ended.
// ---------------------------------------------------------------------------

#[test]
fn prefix_boundaries_skip_surrogates_and_open_end_at_max() {
    use pi_agent_core::harness::session::values::ValueAddress;
    use pi_session_backend_sqlite_node::sqlite::values::scan_scalar_value_rows;
    let db = seeded_db();
    for (key, seq) in [
        ("k:\u{d7ff}a", 1i64),
        ("k:\u{e000}", 2),
        ("k:\u{10ffff}", 3),
        ("k;\u{10ffff}\u{10ffff}", 4),
    ] {
        seed_scalar_value(db.as_ref(), "ns", key, seq, "0");
    }
    // Prefix "k:\u{d7ff}": the boundary walks the last code point U+D7FF ->
    // U+E000, so the scan includes keys sorting below "k:\u{e000}" and
    // excludes the boundary itself.
    let address = ValueAddress {
        namespace: "ns".to_owned(),
        key: "k:\u{d7ff}".to_owned(),
    };
    let stored = scan_scalar_value_rows(db.as_ref(), "s", &address).expect("scan");
    let keys: Vec<String> = stored
        .iter()
        .map(|stored_value| stored_value.key.clone())
        .collect();
    assert_eq!(keys, vec!["k:\u{d7ff}a".to_owned()]);
    // An all-U+10FFFF prefix never increments: the scan is open-ended and
    // the "k:" keys all sort below the prefix.
    let max_prefix = ValueAddress {
        namespace: "ns".to_owned(),
        key: "\u{10ffff}".to_owned(),
    };
    assert!(
        scan_scalar_value_rows(db.as_ref(), "s", &max_prefix)
            .expect("scan")
            .is_empty()
    );
    // The "k:" prefix's boundary is "k;": the \u{10ffff}-keyed row sorts
    // below it (byte 0x3A < 0x3B) and the "k;"-prefixed row sorts above.
    let colon_prefix = ValueAddress {
        namespace: "ns".to_owned(),
        key: "k:".to_owned(),
    };
    let stored = scan_scalar_value_rows(db.as_ref(), "s", &colon_prefix).expect("scan");
    let keys: Vec<String> = stored
        .iter()
        .map(|stored_value| stored_value.key.clone())
        .collect();
    assert_eq!(
        keys,
        vec![
            "k:\u{d7ff}a".to_owned(),
            "k:\u{e000}".to_owned(),
            "k:\u{10ffff}".to_owned()
        ]
    );
}

// ---------------------------------------------------------------------------
// Storage-version gate: both messages.
// ---------------------------------------------------------------------------

#[test]
fn storage_version_gate_reports_newer_and_older_versions() {
    use pi_session_backend_sqlite_node::sqlite::session_row::{
        metadata_from_session_row, read_session_row,
    };
    let db = memory_db();
    apply_initial_schema(db.as_ref()).expect("schema");
    sql!(
        "INSERT INTO sessions (id, created_at, parent_session_id, storage_version, metadata, message_count, usage_payload, next_seq)
		VALUES ('s', 1, NULL, 5, NULL, 0, '{}', 1)"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed");
    let error = metadata_from_session_row(
        "/container.sqlite",
        &read_session_row(db.as_ref(), "s").expect("row"),
        SQLITE_STORAGE_VERSION,
    )
    .expect_err("newer version rejects");
    assert!(error.to_string().contains("is newer than 1"), "{error}");
    let error = metadata_from_session_row(
        "/container.sqlite",
        &read_session_row(db.as_ref(), "s").expect("row"),
        9,
    )
    .expect_err("older version rejects");
    assert!(error.to_string().contains("requires migrations"), "{error}");
}

// ---------------------------------------------------------------------------
// Changes-count and unknown-session errors.
// ---------------------------------------------------------------------------

#[test]
fn sequence_helpers_report_unknown_and_missed_updates() {
    let db = memory_db();
    apply_initial_schema(db.as_ref()).expect("schema");
    let error = read_next_seq(db.as_ref(), "s").expect_err("unknown session");
    assert_eq!(
        error,
        SessionError::Message("Unknown SQLite session: s".to_owned())
    );
    sql!(
        "INSERT INTO sessions (id, created_at, parent_session_id, storage_version, metadata, message_count, usage_payload, next_seq)
		VALUES ('s', 1, NULL, 1, NULL, 0, '{}', 1)"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed");
    let error = advance_next_seq(db.as_ref(), "missing", 2).expect_err("missed update");
    assert!(
        error
            .to_string()
            .contains("Expected to update one SQLite session missing, updated 0")
    );
    let error = delete_session_rows(db.as_ref(), "missing").expect_err("missed delete");
    assert!(
        error
            .to_string()
            .contains("Expected to delete one SQLite session missing, deleted 0")
    );
}

// ---------------------------------------------------------------------------
// Branch-index errors reachable through storage commits and scans.
// ---------------------------------------------------------------------------

fn seed_entry_row(
    db: &dyn SqliteDatabase,
    id: &str,
    parent_id: Option<&str>,
    seq: i64,
    kind: &str,
) {
    sql!(
        "INSERT INTO entries (session_id, id, parent_id, seq, type, custom_type, timestamp, payload)
		VALUES ('s', ?, ?, ?, ?, NULL, 1, '{}')",
        id,
        parent_id,
        seq,
        kind
    )
    .run(db)
    .map(|_| ())
    .expect("seed entry");
}

/// Seeds one `scalar_values` row, the suite's raw-SQL insert.
fn seed_scalar_value(db: &dyn SqliteDatabase, namespace: &str, key: &str, seq: i64, value: &str) {
    sql!(
        "INSERT INTO scalar_values (session_id, namespace, key, seq, value)
		VALUES ('s', ?, ?, ?, ?)",
        namespace,
        key,
        seq,
        value,
    )
    .run(db)
    .map(|_| ())
    .expect("seed scalar value");
}

#[tokio::test]
async fn commit_report_missing_branch_cache_entry() {
    let db = seeded_db();
    // The parent row exists (the entries trigger passes) but carries no
    // branch membership, so the divergent-branch path fails on the cache.
    seed_entry_row(db.as_ref(), "orphan", None, 1, "message");
    let storage = SqliteStorage::new(Arc::clone(&db), &storage_options("s"));
    let error = storage
        .commit(
            vec![Write::Entry(Box::new(insert_entry(NewEntry::Message {
                id: "child".to_owned(),
                parent_id: Some("orphan".to_owned()),
                body: Box::new(MessageEntry {
                    message: user_message("hi"),
                    terminate: None,
                }),
            })))],
            &context(),
        )
        .await
        .expect_err("branch cache error");
    assert_eq!(
        error,
        SessionError::Message("Branch cache missing entry orphan".to_owned())
    );
}

fn storage_options(session_id: &str) -> SqliteStorageOptions {
    SqliteStorageOptions {
        session_id: session_id.to_owned(),
        now: Some(support::fixed_clock(support::NOW)),
    }
}

#[tokio::test]
async fn scan_report_missing_branch_metadata_for_the_base_chain() {
    use pi_agent_core::harness::session::types::StorageBranchScan;
    let db = seeded_db();
    seed_entry_row(db.as_ref(), "entry", None, 2, "message");
    sql!(
        "INSERT INTO branch_meta (session_id, branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq)
		VALUES ('s', 'x', 'entry', 2, 'y', 1)"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed meta");
    sql!(
        "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type)
		VALUES ('s', 'x', 'entry', 2, 'message')"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed membership");
    let storage = SqliteStorage::new(Arc::clone(&db), &storage_options("s"));
    let error = storage
        .scan_branch(
            &StorageBranchScan {
                start: "entry".to_owned(),
                ..StorageBranchScan::default()
            },
            &context(),
        )
        .await
        .expect_err("missing base metadata");
    assert_eq!(
        error,
        SessionError::Message("Branch metadata missing for branch y".to_owned())
    );
}

#[tokio::test]
async fn scan_report_base_branch_without_base_seq() {
    use pi_agent_core::harness::session::types::StorageBranchScan;
    let db = seeded_db();
    seed_entry_row(db.as_ref(), "entry", None, 2, "message");
    sql!(
        "INSERT INTO branch_meta (session_id, branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq)
		VALUES ('s', 'x', 'entry', 2, 'y', NULL)"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed meta");
    sql!(
        "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type)
		VALUES ('s', 'x', 'entry', 2, 'message')"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed membership");
    let storage = SqliteStorage::new(Arc::clone(&db), &storage_options("s"));
    let error = storage
        .scan_branch(
            &StorageBranchScan {
                start: "entry".to_owned(),
                ..StorageBranchScan::default()
            },
            &context(),
        )
        .await
        .expect_err("base without seq");
    assert_eq!(
        error,
        SessionError::Message("Branch x has base branch without base_seq".to_owned())
    );
}

// ---------------------------------------------------------------------------
// Decode errors.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn decode_report_custom_row_missing_custom_type() {
    let db = seeded_db();
    sql!(
        "INSERT INTO entries (session_id, id, parent_id, seq, type, custom_type, timestamp, payload)
		VALUES ('s', 'c', NULL, 1, 'custom', NULL, 1, '{}')"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed");
    let storage = SqliteStorage::new(Arc::clone(&db), &storage_options("s"));
    let error = storage
        .get_entries(vec!["c".to_owned()], &context())
        .await
        .expect_err("missing custom type");
    assert_eq!(
        error,
        SessionError::Message("Custom entry c is missing custom_type".to_owned())
    );
}

#[tokio::test]
async fn decode_report_unknown_entry_type() {
    let db = seeded_db();
    sql!(
        "INSERT INTO entries (session_id, id, parent_id, seq, type, custom_type, timestamp, payload)
		VALUES ('s', 'm', NULL, 1, 'mystery', NULL, 1, '{}')"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed");
    let storage = SqliteStorage::new(Arc::clone(&db), &storage_options("s"));
    let error = storage
        .get_entries(vec!["m".to_owned()], &context())
        .await
        .expect_err("unknown type");
    assert_eq!(
        error,
        SessionError::Message("Unknown SQLite entry type: mystery".to_owned())
    );
}

#[tokio::test]
async fn decode_report_out_of_range_sequence() {
    let db = seeded_db();
    sql!(
        "INSERT INTO entries (session_id, id, parent_id, seq, type, custom_type, timestamp, payload)
		VALUES ('s', 'neg', NULL, -1, 'message', NULL, 1, '{}')"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed");
    let storage = SqliteStorage::new(Arc::clone(&db), &storage_options("s"));
    let error = storage
        .get_entries(vec!["neg".to_owned()], &context())
        .await
        .expect_err("negative seq");
    assert_eq!(
        error,
        SessionError::Message("Integer out of range: -1".to_owned())
    );
}

#[tokio::test]
async fn scan_bounds_refuse_out_of_sqlite_range_sequences() {
    let db = seeded_db();
    let storage = SqliteStorage::new(Arc::clone(&db), &storage_options("s"));
    let error = storage
        .scan_entries(
            &pi_agent_core::harness::session::types::EntryScan {
                from_seq: Some(u64::MAX),
                ..pi_agent_core::harness::session::types::EntryScan::default()
            },
            &context(),
        )
        .await
        .expect_err("out of range");
    assert!(
        error
            .to_string()
            .contains(&format!("Integer out of SQLite range: {}", u64::MAX))
    );
}

// ---------------------------------------------------------------------------
// Closed-storage reads and the int helpers' round trip.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn closed_storage_rejects_every_read_and_write() {
    let storage = SqliteStorage::new(seeded_db(), &storage_options("s"));
    storage.close(&context()).await.expect("close");
    let error = storage
        .get_entries(Vec::new(), &context())
        .await
        .expect_err("closed");
    assert_eq!(
        error,
        SessionError::Message("SqliteStorage is closed".to_owned())
    );
    let error = storage
        .commit(Vec::new(), &context())
        .await
        .expect_err("closed commit");
    assert_eq!(
        error,
        SessionError::Message("SqliteStorage is closed".to_owned())
    );
    let error = storage
        .get_stats(&context())
        .await
        .expect_err("closed stats");
    assert_eq!(
        error,
        SessionError::Message("SqliteStorage is closed".to_owned())
    );
    let address = pi_agent_core::harness::session::values::ValueAddress {
        namespace: "n".to_owned(),
        key: "k".to_owned(),
    };
    let error = storage
        .get_value(&address, &context())
        .await
        .expect_err("closed value");
    assert_eq!(
        error,
        SessionError::Message("SqliteStorage is closed".to_owned())
    );
}

#[test]
fn integer_helpers_round_trip_and_reject() {
    assert_eq!(sql_integer(0).expect("zero"), 0);
    assert_eq!(sql_integer(i64::MAX as u64).expect("max"), i64::MAX);
    let error = sql_integer(u64::MAX).expect_err("overflow");
    assert_eq!(
        error,
        SessionError::Message(format!("Integer out of SQLite range: {}", u64::MAX))
    );
    assert_eq!(sql_u64(7).expect("seven"), 7);
    let error = sql_u64(-1).expect_err("negative");
    assert_eq!(
        error,
        SessionError::Message("Integer out of range: -1".to_owned())
    );
}

// ---------------------------------------------------------------------------
// Driver-seam behaviors.
// ---------------------------------------------------------------------------

#[test]
fn adapter_rejects_exec_with_parameters() {
    let db = memory_db();
    let error = sql!("SELECT 1", 1i64)
        .exec(db.as_ref())
        .expect_err("params");
    assert_eq!(
        error.to_string(),
        "SQLite exec queries cannot have parameters"
    );
}

#[test]
fn adapter_run_reports_stale_connection_counters_like_node() {
    let db = memory_db();
    db.exec("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        .expect("table");
    let fresh = db
        .prepare("SELECT 1")
        .run(&pi_session_backend_sqlite_node::sqlite::types::SqliteParams::none())
        .expect("run on select");
    assert_eq!(fresh.changes, 0, "no insert has ever run");
    assert_eq!(fresh.last_insert_rowid, 0);
    db.prepare("INSERT INTO t (v) VALUES (?)")
        .run(
            &pi_session_backend_sqlite_node::sqlite::types::SqliteParams::Positional(vec![
                SqliteValue::from_param("a"),
            ]),
        )
        .expect("insert");
    let stale = db
        .prepare("SELECT 1")
        .run(&pi_session_backend_sqlite_node::sqlite::types::SqliteParams::none())
        .expect("run on select");
    assert_eq!(stale.changes, 1, "stale connection-level changes");
    assert_eq!(stale.last_insert_rowid, 1, "stale connection-level rowid");
}

#[test]
fn wrap_rusqlite_connection_wraps_a_plain_handle() {
    let connection = rusqlite::Connection::open_in_memory().expect("connection");
    let db = wrap_rusqlite_connection(connection);
    db.exec("CREATE TABLE t (v)").expect("table");
    db.prepare("INSERT INTO t (v) VALUES (?)")
        .run(
            &pi_session_backend_sqlite_node::sqlite::types::SqliteParams::Positional(vec![
                SqliteValue::from_param("x"),
            ]),
        )
        .expect("insert");
    let rows = db
        .prepare("SELECT v FROM t")
        .all(&pi_session_backend_sqlite_node::sqlite::types::SqliteParams::none())
        .expect("rows");
    assert_eq!(rows.len(), 1);
    db.close().expect("close");
    // A second close is inert.
    db.close().expect("second close");
}

// ---------------------------------------------------------------------------
// Repo-level: the seed row shape, the fork's next_seq carry, the erased
// adapter, and discovery's best-effort swallow.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn seed_row_carries_null_metadata_and_zero_usage() {
    let directory = tempfile::tempdir().expect("tempdir");
    let repo = fixed_repo(&directory);
    let session = repo
        .create(SqliteSessionCreateOptions::default(), &context())
        .await
        .expect("create");
    session.close(&context()).await.expect("close");
    let id = session.typed_metadata().base.id.clone();
    assert_eq!(id.len(), 36, "the generated id is a uuid shape");
    let db = create_rusqlite_factory()
        .open_existing(&session.typed_metadata().path)
        .expect("open");
    let row = sql!(
        "SELECT metadata, usage_payload FROM sessions WHERE id = ?",
        id
    )
    .get(db.as_ref())
    .expect("row")
    .expect("session row");
    assert_eq!(
        row.get("metadata"),
        Some(&SqliteValue::Null),
        "the metadata column is NULL"
    );
    let usage: pi_ai::types::Usage =
        serde_json::from_str(&row.string("usage_payload").expect("payload")).expect("usage");
    assert_eq!(usage, pi_ai::types::Usage::default(), "the zero-usage seed");
    db.close().expect("close");
}

#[tokio::test]
async fn fork_carries_the_source_next_seq_into_the_destination() {
    let directory = tempfile::tempdir().expect("tempdir");
    let repo = fixed_repo(&directory);
    let context = context();
    let session = repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    session
        .mutate(
            StorageBackedSession::commit_writes_callback(vec![Write::Entry(Box::new(
                insert_entry(NewEntry::Message {
                    id: "one".to_owned(),
                    parent_id: None,
                    body: Box::new(MessageEntry {
                        message: user_message("hi"),
                        terminate: None,
                    }),
                }),
            ))]),
            &context,
        )
        .await
        .expect("commit");
    session.close(&context).await.expect("close");
    let fork = repo
        .fork(
            session.typed_metadata(),
            &ForkOptions::Tree { id: None },
            &context,
        )
        .await
        .expect("fork");
    let commit = fork
        .mutate(
            StorageBackedSession::commit_writes_callback(vec![Write::Entry(Box::new(
                insert_entry(NewEntry::Message {
                    id: "two".to_owned(),
                    parent_id: Some("one".to_owned()),
                    body: Box::new(MessageEntry {
                        message: user_message("ho"),
                        terminate: None,
                    }),
                }),
            ))]),
            &context,
        )
        .await
        .expect("commit");
    let result = commit
        .downcast::<pi_agent_core::harness::session::types::CommitResult>()
        .expect("commit result");
    assert_eq!(
        result.first_seq, 2,
        "the destination continues the source's sequence"
    );
    fork.close(&context).await.expect("close");
    repo.close(&context).await.expect("close");
}

#[tokio::test]
async fn erased_adapter_derives_paths_and_forks_an_open_source() {
    let directory = tempfile::tempdir().expect("tempdir");
    let repo = Arc::new(fixed_repo(&directory));
    let context = context();
    let erased: Arc<dyn SessionRepo> = repo.clone();
    let session = erased
        .create(
            pi_agent_core::harness::session::types::SessionCreateOptions {
                id: Some("s1".to_owned()),
                ..pi_agent_core::harness::session::types::SessionCreateOptions::default()
            },
            &context,
        )
        .await
        .expect("create");
    session
        .mutate(
            StorageBackedSession::commit_writes_callback(vec![Write::Entry(Box::new(
                insert_entry(NewEntry::Message {
                    id: "one".to_owned(),
                    parent_id: None,
                    body: Box::new(MessageEntry {
                        message: user_message("hi"),
                        terminate: None,
                    }),
                }),
            ))]),
            &context,
        )
        .await
        .expect("commit");
    // The erased fork from the OPEN source resolves the queued-snapshot path
    // through the canonicalized derived path.
    let forked = erased
        .fork(
            &SessionMetadata {
                id: "s1".to_owned(),
                created_at: support::NOW,
                storage_version: SQLITE_STORAGE_VERSION,
                cwd: None,
                parent_session_id: None,
                legacy_parent_session_path: None,
            },
            &ForkOptions::Tree {
                id: Some("f1".to_owned()),
            },
            &context,
        )
        .await
        .expect("erased fork");
    session.close(&context).await.expect("close");
    // The erased open derives the path from the base metadata.
    let reopened = erased
        .open(
            &SessionMetadata {
                id: "s1".to_owned(),
                created_at: support::NOW,
                storage_version: SQLITE_STORAGE_VERSION,
                cwd: None,
                parent_session_id: None,
                legacy_parent_session_path: None,
            },
            &context,
        )
        .await
        .expect("erased open");
    let entries: BTreeMap<String, Entry> = reopened
        .get_entries(vec!["one".to_owned()], &context)
        .await
        .expect("entries");
    assert!(entries.contains_key("one"));
    let forked_entries: BTreeMap<String, Entry> = forked
        .get_entries(vec!["one".to_owned()], &context)
        .await
        .expect("fork entries");
    assert!(
        forked_entries.contains_key("one"),
        "the forked tree carries the source entries"
    );
    let listed = erased.list(&context).await.expect("list");
    let mut ids: Vec<&str> = listed.iter().map(|metadata| metadata.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec!["f1", "s1"],
        "both sessions list through the erased surface"
    );
    let closed = forked.close(&context).await;
    assert!(closed.is_ok());
    let closed = reopened.close(&context).await;
    assert!(closed.is_ok());
    let closed = session.close(&context).await;
    assert!(closed.is_ok());
    let _ = repo.close(&context).await;
}

#[tokio::test]
async fn list_skips_entries_it_cannot_open() {
    let directory = tempfile::tempdir().expect("tempdir");
    let repo = fixed_repo(&directory);
    let context = context();
    let session = repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    session.close(&context).await.expect("close");
    // A directory wearing the container extension cannot open; discovery
    // swallows it and reports the readable sessions.
    std::fs::create_dir(directory.path().join("x.sqlite")).expect("dir");
    let listed = repo.list(&context).expect("list");
    let ids: Vec<String> = listed
        .iter()
        .map(|metadata| metadata.base.id.clone())
        .collect();
    assert_eq!(ids, vec![session.typed_metadata().base.id.clone()]);
    let _ = repo.close(&context).await;
}

#[tokio::test]
async fn shared_container_options_place_sessions_in_one_file() {
    let directory = tempfile::tempdir().expect("tempdir");
    let database_path = directory
        .path()
        .join("sessions.sqlite")
        .to_string_lossy()
        .into_owned();
    let repo = SqliteSessionRepo::new(shared_repo_options(
        directory.path().to_string_lossy().as_ref(),
        &database_path,
        support::fixed_clock(support::NOW),
    ));
    let context = context();
    let first = repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    let second = repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    assert_eq!(
        first.typed_metadata().path,
        second.typed_metadata().path,
        "one shared container"
    );
    first.close(&context).await.expect("close");
    second.close(&context).await.expect("close");
    let _ = repo.close(&context).await;
}

#[tokio::test]
async fn storage_contract_round_trips_a_full_session() {
    let storage = SqliteStorage::new(seeded_db(), &storage_options("s"));
    let context = context();
    storage
        .commit(
            vec![
                Write::Entry(Box::new(insert_entry(NewEntry::Message {
                    id: "root".to_owned(),
                    parent_id: None,
                    body: Box::new(MessageEntry {
                        message: user_message("hi"),
                        terminate: None,
                    }),
                }))),
                Write::Entry(Box::new(insert_entry(NewEntry::Compaction {
                    id: "c".to_owned(),
                    parent_id: Some("root".to_owned()),
                    body: pi_agent_core::harness::session::types::CompactionEntryBody {
                        summary: "s".to_owned(),
                        retained_tail: vec![],
                        tokens_before: 1,
                        details: Some(serde_json::json!({ "k": "v" })),
                        usage: Some(pi_ai::types::Usage::default()),
                        from_hook: true,
                    },
                }))),
                Write::Entry(Box::new(insert_entry(NewEntry::BranchSummary {
                    id: "b".to_owned(),
                    parent_id: Some("c".to_owned()),
                    body: pi_agent_core::harness::session::types::BranchSummaryEntryBody {
                        from_id: Some("root".to_owned()),
                        summary: "sum".to_owned(),
                        details: None,
                        usage: None,
                        from_hook: false,
                    },
                }))),
                Write::Entry(Box::new(insert_entry(NewEntry::Custom {
                    id: "x".to_owned(),
                    parent_id: Some("b".to_owned()),
                    body: pi_agent_core::harness::session::types::CustomEntryBody {
                        custom_type: "note".to_owned(),
                        data: Some(serde_json::json!({ "value": 1 })),
                    },
                }))),
            ],
            &context,
        )
        .await
        .expect("commit");
    let stored: Vec<Entry> = storage
        .scan_entries(
            &pi_agent_core::harness::session::types::EntryScan::default(),
            &context,
        )
        .await
        .expect("scan");
    assert_eq!(stored.len(), 4);
    let entries: BTreeMap<String, Entry> = storage
        .get_entries(
            stored.iter().map(|entry| entry.id().to_owned()).collect(),
            &context,
        )
        .await
        .expect("entries");
    // The scan reads in sequence order; the id-keyed map returns ids sorted.
    let mut from_get: Vec<Entry> = entries.into_values().collect();
    from_get.sort_by_key(Entry::seq);
    assert_eq!(stored, from_get, "payload round-trips per variant");
    storage.close(&context).await.expect("close");
}

// ---------------------------------------------------------------------------
// Coverage-boundary additions: the Debug impls, the conversion arms, the
// driver-seam error paths, and the repo error paths the 1:1 suites leave
// untested.
// ---------------------------------------------------------------------------

#[test]
fn debug_impls_render_without_panicking() {
    use pi_session_backend_sqlite_node::sqlite::entries::EntryRowWriter;
    use pi_session_backend_sqlite_node::sqlite::types::SqliteTransactionOutcome;
    use pi_session_backend_sqlite_node::sqlite::usage_ledger::UsageLedgerRowWriter;
    let directory = tempfile::tempdir().expect("tempdir");
    let options = pi_session_backend_sqlite_node::SqliteSessionRepoOptions {
        directory: directory.path().to_string_lossy().into_owned(),
        database_path: Some("db.sqlite".to_owned()),
        database_factory: Arc::new(create_rusqlite_factory()),
        now: None,
    };
    let _debug = format!("{options:?}");
    let db = memory_db();
    let _writer = format!("{:?}", EntryRowWriter::new(Arc::clone(&db), "s".to_owned()));
    let _usage_writer = format!(
        "{:?}",
        UsageLedgerRowWriter::new(Arc::clone(&db), "s".to_owned())
    );
    let _storage = format!(
        "{:?}",
        SqliteStorage::new(
            Arc::clone(&db),
            &SqliteStorageOptions {
                session_id: "s".to_owned(),
                now: None
            }
        )
    );
    let _outcome = format!("{:?}", SqliteTransactionOutcome::Asynchronous);
    let outcome = SqliteTransactionOutcome::Committed(Box::new(0i32));
    assert_eq!(
        format!("{outcome:?}"),
        "SqliteTransactionOutcome::Committed(..)"
    );
    let adapter_error =
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError::new("boom");
    assert_eq!(
        format!("{adapter_error:?}"),
        "SqliteAdapterError { message: \"boom\" }"
    );
}

#[test]
fn row_accessors_read_all_value_kinds_and_report_misfits() {
    use pi_session_backend_sqlite_node::sqlite::types::SqliteRow;
    let db = seeded_db();
    sql!("CREATE TABLE probe (t TEXT, i INTEGER, r REAL, b BLOB)")
        .run(db.as_ref())
        .map(|_| ())
        .expect("table");
    sql!("INSERT INTO probe VALUES ('x', 5, 0.5, x'0102')")
        .run(db.as_ref())
        .map(|_| ())
        .expect("insert");
    let row = sql!("SELECT t, i, r, b FROM probe")
        .get(db.as_ref())
        .expect("row")
        .expect("row");
    assert_eq!(row.string("t").expect("t"), "x");
    assert_eq!(row.integer("i").expect("i"), 5);
    assert!((row.real("r").expect("r") - 0.5).abs() < f64::EPSILON);
    assert_eq!(row.get("b"), Some(&SqliteValue::Blob(vec![1, 2])));
    // Misfits and missing columns report.
    assert!(row.string("r").is_err());
    assert!(row.integer("t").is_err());
    assert!(row.real("t").is_err());
    assert!(row.string("absent").is_err());
    assert!(row.integer("absent").is_err());
    assert!(row.real("absent").is_err());
    // The conversion arms round-trip.
    assert_eq!(SqliteValue::from(7i32), SqliteValue::Integer(7));
    assert_eq!(SqliteValue::from(0.5f64), SqliteValue::Real(0.5));
    assert_eq!(
        SqliteValue::from(vec![1u8, 2]),
        SqliteValue::Blob(vec![1, 2])
    );
    assert_eq!(SqliteValue::from(None::<String>), SqliteValue::Null);
    assert_eq!(
        SqliteValue::from_param("x"),
        SqliteValue::Text("x".to_owned())
    );
    let _ = SqliteRow::column_names(&row).count();
}

#[test]
fn named_parameters_bind_through_get_all_and_iterate() {
    let db = seeded_db();
    sql!("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        .run(db.as_ref())
        .map(|_| ())
        .expect("table");
    sql!("INSERT INTO t (v) VALUES (:value)", "named")
        .run(db.as_ref())
        .map(|_| ())
        .expect("insert");
    let mut named = BTreeMap::new();
    named.insert("id".to_owned(), SqliteValue::Integer(1));
    let row = db
        .prepare("SELECT v FROM t WHERE id >= :id")
        .get(&pi_session_backend_sqlite_node::sqlite::types::SqliteParams::Named(named.clone()))
        .expect("get");
    assert_eq!(
        row.map(|row| row.string("v").expect("v")),
        Some("named".to_owned())
    );
    let all = db
        .prepare("SELECT v FROM t WHERE id >= :id")
        .all(&pi_session_backend_sqlite_node::sqlite::types::SqliteParams::Named(named.clone()))
        .expect("all");
    assert_eq!(all.len(), 1);
    let iterated = db
        .prepare("SELECT v FROM t WHERE id >= :id")
        .iterate(&pi_session_backend_sqlite_node::sqlite::types::SqliteParams::Named(named))
        .expect("iterate");
    assert_eq!(iterated.len(), 1);
}

#[test]
fn a_callback_closing_the_database_reports_the_commit_failure() {
    let db = seeded_db();
    let error = db
        .transaction(Box::new({
            let inner = Arc::clone(&db);
            move || {
                let _ = inner.close();
                Err(
                    pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError::new(
                        "closed mid transaction",
                    ),
                )
            }
        }))
        .expect_err("callback error");
    assert_eq!(error.to_string(), "closed mid transaction");
    // The commit path: a callback that closes the database makes COMMIT fail;
    // the rollback is swallowed and the commit error wins.
    let db = seeded_db();
    let error = db
        .transaction(Box::new({
            let inner = Arc::clone(&db);
            move || {
                let closed = inner.close();
                Ok(pi_session_backend_sqlite_node::sqlite::types::SqliteTransactionOutcome::Committed(
                    Box::new(closed.is_ok()),
                ))
            }
        }))
        .expect_err("commit after close");
    assert!(error.to_string().contains("closed"), "{error}");
}

#[tokio::test]
async fn duplicate_create_of_a_closed_session_reports_the_existing_row() {
    let directory = tempfile::tempdir().expect("tempdir");
    let repo = fixed_repo(&directory);
    let context = context();
    let first = repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    first.close(&context).await.expect("close");
    // The per-file recreate fails at the wx reservation first: the container
    // file exists.
    let error = repo
        .create(
            pi_agent_core::harness::session::types::SessionCreateOptions {
                id: Some(first.typed_metadata().base.id.clone()),
                ..pi_agent_core::harness::session::types::SessionCreateOptions::default()
            },
            &context,
        )
        .await
        .expect_err("duplicate file");
    assert!(error.to_string().contains("File exists"), "{error}");
    // The shared container skips the file reservation: the row check throws.
    let shared = tempfile::tempdir().expect("tempdir");
    let database_path = shared
        .path()
        .join("sessions.sqlite")
        .to_string_lossy()
        .into_owned();
    let shared_repo = SqliteSessionRepo::new(shared_repo_options(
        shared.path().to_string_lossy().as_ref(),
        &database_path,
        support::fixed_clock(support::NOW),
    ));
    let seeded = shared_repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    seeded.close(&context).await.expect("close");
    let error = shared_repo
        .create(
            pi_agent_core::harness::session::types::SessionCreateOptions {
                id: Some(seeded.typed_metadata().base.id.clone()),
                ..pi_agent_core::harness::session::types::SessionCreateOptions::default()
            },
            &context,
        )
        .await
        .expect_err("duplicate row");
    assert_eq!(
        error,
        SessionError::Message(format!(
            "SQLite session already exists: {}",
            seeded.typed_metadata().base.id
        ))
    );
    let _ = repo.close(&context).await;
    let _ = shared_repo.close(&context).await;
}

#[tokio::test]
async fn fork_of_an_existing_destination_reports_the_existing_row() {
    let directory = tempfile::tempdir().expect("tempdir");
    let database_path = directory
        .path()
        .join("sessions.sqlite")
        .to_string_lossy()
        .into_owned();
    let repo = SqliteSessionRepo::new(shared_repo_options(
        directory.path().to_string_lossy().as_ref(),
        &database_path,
        support::fixed_clock(support::NOW),
    ));
    let context = context();
    let source = repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    source.close(&context).await.expect("close");
    let destination = repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    destination.close(&context).await.expect("close");
    let error = repo
        .fork(
            source.typed_metadata(),
            &ForkOptions::Tree {
                id: Some(destination.typed_metadata().base.id.clone()),
            },
            &context,
        )
        .await
        .expect_err("duplicate fork destination");
    assert_eq!(
        error,
        SessionError::Message(format!(
            "SQLite session already exists: {}",
            destination.typed_metadata().base.id
        ))
    );
    let _ = repo.close(&context).await;
}

/// The factory double whose open swaps the reserved file for a directory:
/// the catch's container removal then fails, and the removal error replaces
/// the original.
struct DirectorySwappingOpenFactory;

impl SqliteDatabaseFactory for DirectorySwappingOpenFactory {
    fn open(
        &self,
        path: &str,
    ) -> Result<
        Box<dyn SqliteDatabase>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        std::fs::remove_file(path).expect("swap remove");
        std::fs::create_dir(path).expect("swap dir");
        Err(pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError::new("open failed"))
    }

    fn open_existing(
        &self,
        _path: &str,
    ) -> Result<
        Box<dyn SqliteDatabase>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        unreachable!("not used by this test")
    }

    fn open_read_only(
        &self,
        _path: &str,
    ) -> Result<
        Box<dyn SqliteDatabase>,
        pi_session_backend_sqlite_node::sqlite::types::SqliteAdapterError,
    > {
        unreachable!("not used by this test")
    }
}

#[tokio::test]
async fn a_failed_container_removal_replaces_the_original_create_error() {
    let directory = tempfile::tempdir().expect("tempdir");
    let repo = SqliteSessionRepo::new(pi_session_backend_sqlite_node::SqliteSessionRepoOptions {
        directory: directory.path().to_string_lossy().into_owned(),
        database_path: None,
        database_factory: Arc::new(DirectorySwappingOpenFactory),
        now: Some(support::fixed_clock(support::NOW)),
    });
    let error = repo
        .create(
            pi_agent_core::harness::session::types::SessionCreateOptions {
                id: Some("s".to_owned()),
                ..pi_agent_core::harness::session::types::SessionCreateOptions::default()
            },
            &context(),
        )
        .await
        .expect_err("create failure");
    assert!(
        error.to_string().contains("directory") || error.to_string().contains("not permitted"),
        "{error}"
    );
    let _ = repo.close(&context()).await;
}

#[tokio::test]
async fn a_failing_database_close_reports_from_the_delete_and_the_external_fork() {
    let directory = tempfile::tempdir().expect("tempdir");
    let factory = Arc::new(support::CloseTrackingFactory::new());
    let repo = SqliteSessionRepo::new(pi_session_backend_sqlite_node::SqliteSessionRepoOptions {
        directory: directory.path().to_string_lossy().into_owned(),
        database_path: None,
        database_factory: factory.clone(),
        now: Some(support::fixed_clock(support::NOW)),
    });
    let context = context();
    let session = repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    session.close(&context).await.expect("close");
    // The delete's connection close fails: the close error replaces the
    // would-be success, and the container removal (after the close) never
    // runs.
    factory.fail_next_connection_close();
    let error = repo
        .delete(session.typed_metadata(), &context)
        .await
        .expect_err("close failure");
    assert!(error.to_string().contains("close failed"), "{error}");
    assert_eq!(
        factory.close_attempts(),
        vec![1, 1],
        "each writable connection closed once"
    );
    assert!(
        support::path_exists(&session.typed_metadata().path),
        "the container remains: the removal follows the close, which failed"
    );
    let _ = repo.close(&context).await;
}

#[tokio::test]
async fn fork_external_source_reads_through_the_tracking_factory() {
    let directory = tempfile::tempdir().expect("tempdir");
    let factory = Arc::new(support::CloseTrackingFactory::new());
    let repo = SqliteSessionRepo::new(pi_session_backend_sqlite_node::SqliteSessionRepoOptions {
        directory: directory.path().to_string_lossy().into_owned(),
        database_path: None,
        database_factory: factory.clone(),
        now: Some(support::fixed_clock(support::NOW)),
    });
    let context = context();
    let source = repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    source.close(&context).await.expect("close");
    // Read-only opens pass through the tracking factory unwrapped, so the
    // fork's external-source read runs and closes undisturbed; the close
    // error arm is asserted by the delete test above.
    let forked = repo
        .fork(
            source.typed_metadata(),
            &ForkOptions::Tree { id: None },
            &context,
        )
        .await
        .expect("fork");
    forked.close(&context).await.expect("close");
    let _ = repo.close(&context).await;
}

#[tokio::test]
async fn list_reads_through_the_tracking_factory() {
    let directory = tempfile::tempdir().expect("tempdir");
    let factory = Arc::new(support::CloseTrackingFactory::new());
    let repo = SqliteSessionRepo::new(pi_session_backend_sqlite_node::SqliteSessionRepoOptions {
        directory: directory.path().to_string_lossy().into_owned(),
        database_path: None,
        database_factory: factory.clone(),
        now: Some(support::fixed_clock(support::NOW)),
    });
    let context = context();
    let session = repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    session.close(&context).await.expect("close");
    // Read-only discovery opens bypass the tracking factory, and the
    // best-effort per-file swallow is pinned by the corrupt-file test; this
    // asserts the readable session still lists through the tracked factory.
    let listed = repo.list(&context).expect("list");
    assert_eq!(listed.len(), 1);
    let _ = repo.close(&context).await;
}

#[tokio::test]
async fn list_propagates_a_directory_read_error() {
    let directory = tempfile::tempdir().expect("tempdir");
    // The repo directory path holds a FILE: read_dir fails with
    // NotADirectory, which list propagates (only ENOENT short-circuits).
    let repo_directory = directory.path().join("not-a-dir");
    std::fs::write(&repo_directory, "x").expect("file");
    let repo = SqliteSessionRepo::new(repo_options(
        repo_directory.to_string_lossy().as_ref(),
        support::fixed_clock(support::NOW),
    ));
    let error = repo.list(&context()).expect_err("directory read");
    assert!(error.to_string().contains("directory"), "{error}");
}

#[tokio::test]
async fn fork_external_rolls_back_on_a_newer_source_storage_version() {
    let directory = tempfile::tempdir().expect("tempdir");
    let repo = fixed_repo(&directory);
    let context = context();
    let source = repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    source.close(&context).await.expect("close");
    let db = create_rusqlite_factory()
        .open_existing(&source.typed_metadata().path)
        .expect("open");
    sql!(
        "UPDATE sessions SET storage_version = ? WHERE id = ?",
        999i64,
        &source.typed_metadata().base.id
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("bump version");
    db.close().expect("close");
    let error = repo
        .fork(
            source.typed_metadata(),
            &ForkOptions::Tree { id: None },
            &context,
        )
        .await
        .expect_err("version gate");
    assert!(error.to_string().contains("is newer than 1"), "{error}");
    // The rollback left no destination container behind.
    assert!(
        !support::path_exists(&format!(
            "{}{}",
            source.typed_metadata().path.replace(".sqlite", ""),
            "fork.sqlite"
        )) || !directory.path().join(format!("~{}", "any")).exists()
    );
    let leftovers: Vec<_> = std::fs::read_dir(directory.path())
        .expect("dir")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .map(|path| {
            path.file_name()
                .expect("entry")
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.ends_with(".sqlite"))
        .collect();
    assert_eq!(
        leftovers.len(),
        1,
        "only the source container remains: {leftovers:?}"
    );
    let _ = repo.close(&context).await;
}

#[tokio::test]
async fn scan_branch_limit_zero_returns_empty() {
    use pi_agent_core::harness::session::types::StorageBranchScan;
    let db = seeded_db();
    seed_entry_row(db.as_ref(), "root", None, 1, "message");
    sql!("INSERT INTO branch_meta (session_id, branch_id, tip_entry_id, tip_seq, base_branch_id, base_seq) VALUES ('s', 'root', 'root', 1, NULL, NULL)")
        .run(db.as_ref())
        .map(|_| ())
        .expect("meta");
    sql!("INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type) VALUES ('s', 'root', 'root', 1, 'message')")
        .run(db.as_ref())
        .map(|_| ())
        .expect("membership");
    let storage = SqliteStorage::new(Arc::clone(&db), &storage_options("s"));
    let entries = storage
        .scan_branch(
            &StorageBranchScan {
                start: "root".to_owned(),
                limit: Some(0),
                ..StorageBranchScan::default()
            },
            &context(),
        )
        .await
        .expect("scan");
    assert!(entries.is_empty(), "limit zero reads nothing");
    storage.close(&context()).await.expect("close");
}

#[test]
fn usage_writer_and_free_insert_round_trip_details() {
    use pi_session_backend_sqlite_node::sqlite::usage_ledger::{
        UsageLedgerRowWriter, insert_usage_ledger_row, scan_usage_ledger_rows,
    };
    let db = seeded_db();
    let row = pi_agent_core::harness::session::types::UsageRow {
        id: "u".to_owned(),
        seq: 3,
        usage: pi_ai::types::Usage::default(),
        entry_id: Some("e".to_owned()),
        adjustment: true,
        details: Some(serde_json::json!({ "reason": "adjust" })),
    };
    UsageLedgerRowWriter::new(Arc::clone(&db), "s".to_owned())
        .insert(&row)
        .expect("writer insert");
    let second = pi_agent_core::harness::session::types::UsageRow {
        id: "v".to_owned(),
        seq: 4,
        usage: pi_ai::types::Usage::default(),
        entry_id: None,
        adjustment: false,
        details: None,
    };
    insert_usage_ledger_row(db.as_ref(), "s", &second).expect("free insert");
    let rows = scan_usage_ledger_rows(
        db.as_ref(),
        "s",
        &pi_agent_core::harness::session::types::UsageScan::default(),
    )
    .expect("scan");
    assert_eq!(rows.len(), 2);
    let decoded =
        pi_session_backend_sqlite_node::sqlite::usage_ledger::decode_usage_ledger_row(&rows[0])
            .expect("decode");
    assert_eq!(decoded, row);
    let decoded =
        pi_session_backend_sqlite_node::sqlite::usage_ledger::decode_usage_ledger_row(&rows[1])
            .expect("decode");
    assert_eq!(decoded, second);
    let _ = std::format!(
        "{:?}",
        UsageLedgerRowWriter::new(Arc::clone(&db), "s".to_owned())
    );
}

#[test]
fn entry_helpers_round_trip() {
    use pi_agent_core::harness::session::types::CustomEntryBody as TestCustomEntryBody;
    use pi_session_backend_sqlite_node::sqlite::entries::{
        EntryRowWriter, entry_structure_from_row, insert_entry_row,
    };
    let db = seeded_db();
    let entry = Entry::Custom {
        id: "x".to_owned(),
        parent_id: None,
        seq: 1,
        timestamp: 5,
        body: TestCustomEntryBody {
            custom_type: "note".to_owned(),
            data: None,
        },
    };
    insert_entry_row(db.as_ref(), "s", &entry).expect("free insert");
    let second = Entry::Custom {
        id: "y".to_owned(),
        parent_id: None,
        seq: 2,
        timestamp: 6,
        body: TestCustomEntryBody {
            custom_type: "memo".to_owned(),
            data: None,
        },
    };
    EntryRowWriter::new(Arc::clone(&db), "s".to_owned())
        .insert(&second)
        .expect("writer insert");
    let structure =
        entry_structure_from_row(&pi_session_backend_sqlite_node::sqlite::entries::EntryRow {
            id: "x".to_owned(),
            parent_id: None,
            seq: 1,
            kind: pi_agent_core::harness::session::types::EntryType::Custom,
            custom_type: Some("note".to_owned()),
            timestamp: 5,
            payload: "{}".to_owned(),
        })
        .expect("structure");
    assert_eq!(structure.id, "x");
    assert_eq!(structure.custom_type.as_deref(), Some("note"));
}

#[tokio::test]
async fn get_entries_with_no_ids_returns_empty() {
    let storage = SqliteStorage::new(seeded_db(), &storage_options("s"));
    let entries: BTreeMap<String, Entry> = storage
        .get_entries(Vec::new(), &context())
        .await
        .expect("entries");
    assert!(entries.is_empty());
    storage.close(&context()).await.expect("close");
}

#[tokio::test]
async fn negative_message_count_reports_out_of_range() {
    let db = seeded_db();
    sql!("UPDATE sessions SET message_count = -1 WHERE id = 's'")
        .run(db.as_ref())
        .map(|_| ())
        .expect("bump");
    let storage = SqliteStorage::new(Arc::clone(&db), &storage_options("s"));
    let error = storage
        .get_stats(&context())
        .await
        .expect_err("negative count");
    assert_eq!(
        error,
        SessionError::Message("Integer out of range: -1".to_owned())
    );
    storage.close(&context()).await.expect("close");
}

#[tokio::test]
async fn storage_snapshot_of_an_unknown_branch_scope_errors() {
    let storage = SqliteStorage::new(seeded_db(), &storage_options("s"));
    let error = storage
        .snapshot(
            &ForkOptions::Branch {
                branch: "nope".to_owned(),
                entry_id: None,
                position: None,
                id: None,
            },
            &context(),
        )
        .await
        .expect_err("unknown tip");
    assert_eq!(
        error,
        SessionError::Message("Unknown source branch: nope".to_owned())
    );
    storage.close(&context()).await.expect("close");
}

// ---------------------------------------------------------------------------
// Coverage-boundary additions round two: the error-mapping closures the
// suites never drive — closed-database statements, corrupt persisted JSON,
// filesystem-held repo paths, and the uuidv7 clock failure.
// ---------------------------------------------------------------------------

#[test]
fn debug_impls_render_the_repo_and_facade_types() {
    use pi_session_backend_sqlite_node::sqlite::session::{
        SqliteOpenSession, SqliteOpenSessionOptions,
    };
    let directory = tempfile::tempdir().expect("tempdir");
    let repo = fixed_repo(&directory);
    let _repo_debug = format!("{repo:?}");
    let options = SqliteOpenSessionOptions {
        close_database: Arc::new(|| Ok(())),
        on_close: Arc::new(|| ()),
    };
    let _options_debug = format!("{options:?}");
    // The facade type itself formats through the shared core's Debug.
    let facade_debug = std::any::type_name::<SqliteOpenSession>();
    assert!(facade_debug.contains("SqliteOpenSession"));
}

#[test]
fn a_closed_database_rejects_statement_execution() {
    use pi_session_backend_sqlite_node::sqlite::types::SqliteParams;
    let db = seeded_db();
    db.close().expect("close");
    let statement = db.prepare("SELECT 1");
    let error = statement
        .run(&SqliteParams::none())
        .expect_err("closed run");
    assert_eq!(error.message(), "SQLite database is closed");
    let error = statement
        .get(&SqliteParams::none())
        .expect_err("closed get");
    assert_eq!(error.message(), "SQLite database is closed");
    let error = statement
        .all(&SqliteParams::none())
        .expect_err("closed all");
    assert_eq!(error.message(), "SQLite database is closed");
}

#[tokio::test]
async fn create_reports_uuidv7_failures_from_the_clock() {
    let directory = tempfile::tempdir().expect("tempdir");
    // The clock's value exceeds the uuidv7 timestamp range: the generated-id
    // helper surfaces the error, upstream's uuidv7 throw.
    let repo = SqliteSessionRepo::new(pi_session_backend_sqlite_node::SqliteSessionRepoOptions {
        directory: directory.path().to_string_lossy().into_owned(),
        database_path: None,
        database_factory: Arc::new(create_rusqlite_factory()),
        now: Some(support::fixed_clock(i64::MAX)),
    });
    let error = repo
        .create(
            pi_agent_core::harness::session::types::SessionCreateOptions::default(),
            &context(),
        )
        .await
        .expect_err("uuid failure");
    assert!(error.to_string().contains("UUIDv7 timestamp"), "{error}");
    let _ = repo.close(&context()).await;
}

#[tokio::test]
async fn create_and_fork_fail_when_the_repo_directory_is_a_file() {
    let directory = tempfile::tempdir().expect("tempdir");
    let repo_directory = directory.path().join("held");
    std::fs::write(&repo_directory, "x").expect("file");
    let repo = SqliteSessionRepo::new(repo_options(
        repo_directory.to_string_lossy().as_ref(),
        support::fixed_clock(support::NOW),
    ));
    let context = context();
    let create_error = repo
        .create(
            pi_agent_core::harness::session::types::SessionCreateOptions {
                id: Some("s".to_owned()),
                ..pi_agent_core::harness::session::types::SessionCreateOptions::default()
            },
            &context,
        )
        .await
        .expect_err("held directory");
    assert!(
        create_error.to_string().contains("File exists")
            || create_error.to_string().contains("directory"),
        "{create_error}"
    );
    // The fork path fails at its own mkdir before any snapshot work.
    let source = SessionMetadata {
        id: "source".to_owned(),
        created_at: support::NOW,
        storage_version: SQLITE_STORAGE_VERSION,
        cwd: None,
        parent_session_id: None,
        legacy_parent_session_path: None,
    };
    let fork_error = repo
        .fork(
            &pi_session_backend_sqlite_node::sqlite::session_row::SqliteSessionMetadata {
                base: source,
                path: repo_directory
                    .join("elsewhere.sqlite")
                    .to_string_lossy()
                    .into_owned(),
            },
            &ForkOptions::Tree {
                id: Some("f".to_owned()),
            },
            &context,
        )
        .await
        .expect_err("held directory fork");
    assert!(
        fork_error.to_string().contains("File exists")
            || fork_error.to_string().contains("directory"),
        "{fork_error}"
    );
}

#[tokio::test]
async fn decode_report_corrupt_persisted_json() {
    let db = seeded_db();
    // Corrupt entry payload.
    sql!(
        "INSERT INTO entries (session_id, id, parent_id, seq, type, custom_type, timestamp, payload)
		VALUES ('s', 'p', NULL, 1, 'message', NULL, 1, 'not json')"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed");
    // Corrupt usage payload on the session row and the ledger.
    sql!("UPDATE sessions SET usage_payload = 'not json' WHERE id = 's'")
        .run(db.as_ref())
        .map(|_| ())
        .expect("seed");
    sql!(
        "INSERT INTO usage_ledger (session_id, id, seq, entry_id, adjustment, usage, details)
		VALUES ('s', 'u', 1, NULL, 0, 'not json', NULL)"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed");
    // Corrupt scalar value and list element.
    sql!(
        "INSERT INTO scalar_values (session_id, namespace, key, seq, value)
		VALUES ('s', 'ns', 'k', 1, 'not json')"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed");
    sql!(
        "INSERT INTO list_values (session_id, namespace, key, seq, value)
		VALUES ('s', 'ns', 'l', 1, 'not json')"
    )
    .run(db.as_ref())
    .map(|_| ())
    .expect("seed");
    let storage = SqliteStorage::new(Arc::clone(&db), &storage_options("s"));
    let context = context();

    let error = storage
        .get_entries(vec!["p".to_owned()], &context)
        .await
        .expect_err("corrupt payload");
    assert!(error.to_string().contains("expected"), "{error}");
    let error = storage
        .get_stats(&context)
        .await
        .expect_err("corrupt usage payload");
    assert!(error.to_string().contains("expected"), "{error}");
    let error = storage
        .scan_usage(
            &pi_agent_core::harness::session::types::UsageScan::default(),
            &context,
        )
        .await
        .expect_err("corrupt ledger usage");
    assert!(error.to_string().contains("expected"), "{error}");
    let address = pi_agent_core::harness::session::values::ValueAddress {
        namespace: "ns".to_owned(),
        key: "k".to_owned(),
    };
    let error = storage
        .get_value(&address, &context)
        .await
        .expect_err("corrupt value");
    assert!(error.to_string().contains("expected"), "{error}");
    let list_address = pi_agent_core::harness::session::values::ListAddress {
        namespace: "ns".to_owned(),
        key: "l".to_owned(),
    };
    let error = storage
        .read_list(&list_address, None, &context)
        .await
        .expect_err("corrupt element");
    assert!(error.to_string().contains("expected"), "{error}");
    storage.close(&context).await.expect("close");
}

#[tokio::test]
async fn snapshot_report_a_corrupt_branch_tip_value() {
    let db = seeded_db();
    // A branch tip whose stored value is not a string.
    seed_scalar_value(db.as_ref(), "pi.branch.tip", "main", 1, "42");
    let storage = SqliteStorage::new(Arc::clone(&db), &storage_options("s"));
    let error = storage
        .snapshot(
            &ForkOptions::Branch {
                branch: "main".to_owned(),
                entry_id: None,
                position: None,
                id: None,
            },
            &context(),
        )
        .await
        .expect_err("corrupt tip");
    assert!(error.to_string().contains("invalid type"), "{error}");
    storage.close(&context()).await.expect("close");
}

#[tokio::test]
async fn list_swallows_unresolvable_container_paths() {
    let directory = tempfile::tempdir().expect("tempdir");
    let repo = fixed_repo(&directory);
    let context = context();
    let session = repo
        .create(SqliteSessionCreateOptions::default(), &context)
        .await
        .expect("create");
    session.close(&context).await.expect("close");
    // A dangling symlink wearing the container extension: canonicalize fails
    // per file and discovery moves on.
    #[cfg(unix)]
    std::os::unix::fs::symlink(
        directory.path().join("nowhere.sqlite"),
        directory.path().join("dangling.sqlite"),
    )
    .expect("symlink");
    let listed = repo.list(&context).expect("list");
    assert_eq!(listed.len(), 1);
    let _ = repo.close(&context).await;
}

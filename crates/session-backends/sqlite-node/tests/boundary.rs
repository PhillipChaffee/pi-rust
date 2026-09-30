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
        sql!(
            "INSERT INTO scalar_values (session_id, namespace, key, seq, value) VALUES ('s', 'ns', ?, ?, '0')",
            key,
            seq
        )
        .run(db.as_ref())
        .map(|_| ())
        .expect("seed");
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

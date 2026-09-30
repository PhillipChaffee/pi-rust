//! The `SqliteSessionRepo` suite, ported 1:1 from upstream
//! `packages/session-backends/sqlite-node/test/repo.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` — 22 `it()` declarations, 23
//! runtime tests (the two-layouts loop stays one loop).
//!
//! Restatements recorded for this file, per the port design (§10):
//! - usage-JSON assertions are structural: the seeded `usage_payload`
//!   decodes back to `Usage::default()`; serde_json renders f64 zeros as
//!   `0.0` where JS renders `0`, so upstream's byte-equal
//!   `JSON.stringify(zeroUsage())` compare never becomes a byte compare.
//! - the deferred-promise open gate (upstream's `GatedOpenExistingFactory`)
//!   is support's thread gate: the delete runs on a worker thread via
//!   `Handle::block_on` and parks inside `open_existing` over std channels
//!   while the main thread drives the create/open/fork exclusions.
//! - the live-source WAL snapshot tests run both layouts through one loop
//!   (upstream's per-layout `it`), with support's `SnapshotBoundaryFactory`
//!   hooking the first `FROM sessions` read and the writer committing
//!   through an independent writable connection.
//! - the close-failure identity compares restate promise identity and
//!   `rejects.toBe(error)` as `PartialEq` on the stored
//!   `Result<(), SqliteRepoCloseError>` values.
//! - upstream's `repo.list` awaits a promise for interface uniformity; the
//!   port's `list` is sync (the factory seam is sync), and upstream's
//!   `realpath` is `std::fs::canonicalize`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "the tests panic on failure")]

mod support;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::commit::{insert_entry, insert_usage};
use pi_agent_core::harness::session::session::StorageBackedSession;
use pi_agent_core::harness::session::testing::conformance;
use pi_agent_core::harness::session::types::{
    BranchScanOrder, ForkOptions, LaneConfiguration, LaneState, MessageEntry, ModelIdentity,
    NewEntry, Session, SessionError, SessionMetadata, SessionReader, StorageBranchScan,
    UsageWriteRow,
};
use pi_agent_core::harness::session::values::{self, Write};
use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::types::{Usage, UsageCost, UserContent, UserMessage};
use pi_session_backend_sqlite_node::sqlite::repo::SqliteRepoCloseError;
use pi_session_backend_sqlite_node::sqlite::session_row::SqliteSessionMetadata;
use pi_session_backend_sqlite_node::sqlite::types::{
    SqliteAdapterError, SqliteDatabase, SqliteDatabaseFactory, SqliteTransactionOutcome,
};
use pi_session_backend_sqlite_node::{
    SqliteOpenSession, SqliteSessionCreateOptions, SqliteSessionRepo, SqliteSessionRepoOptions,
    create_rusqlite_factory, sql,
};

use support::{NOW, fixed_clock};

/// One fixture temp directory, upstream's `withTempDir`'s `mkdtemp`; the
/// guard removes the tree recursively on drop, the finally `rm`.
fn temp_dir() -> (tempfile::TempDir, String) {
    let directory = tempfile::tempdir().expect("fixture temp directory");
    let path = directory
        .path()
        .to_str()
        .expect("utf8 temp directory")
        .to_owned();
    (directory, path)
}

/// Opens one raw writable connection for the direct SQL probes, upstream's
/// `withDb`.
fn with_db<T>(path: &str, run: impl FnOnce(&dyn SqliteDatabase) -> T) -> T {
    let db = create_rusqlite_factory().open(path).expect("raw db open");
    let result = run(db.as_ref());
    db.close().expect("raw db close");
    result
}

/// One `realpath` compare, upstream's `await realpath(...)`.
fn canonical_path(path: &str) -> String {
    std::fs::canonicalize(path)
        .expect("realpath")
        .to_string_lossy()
        .into_owned()
}

/// One path under the fixture directory, upstream's `join`.
fn join(directory: &str, leaves: &[&str]) -> String {
    let mut path = std::path::PathBuf::from(directory);
    for leaf in leaves {
        path.push(leaf);
    }
    path.to_string_lossy().into_owned()
}

/// The per-file repo most tests build, upstream's
/// `new SqliteSessionRepo({ directory, databaseFactory, now: () => <n> })`.
fn repo(directory: &str, now: i64) -> SqliteSessionRepo {
    SqliteSessionRepo::new(support::repo_options(directory, fixed_clock(now)))
}

/// The shared-container repo, upstream's `databasePath` option.
fn shared_repo(directory: &str, database_path: &str, now: i64) -> SqliteSessionRepo {
    SqliteSessionRepo::new(support::shared_repo_options(
        directory,
        database_path,
        fixed_clock(now),
    ))
}

/// The explicit-id create options, upstream's `repo.create({ id })`.
fn create_options(id: &str) -> SqliteSessionCreateOptions {
    SqliteSessionCreateOptions {
        id: Some(id.to_owned()),
        parent_session_id: None,
    }
}

/// The branch-scope fork options, upstream's
/// `{ scope: "branch", branch, entryId?, id }` literals.
fn branch_fork(branch: &str, entry_id: Option<&str>, id: &str) -> ForkOptions {
    conformance::branch_fork(branch, entry_id, None, Some(id))
}

/// The tree-scope fork options, upstream's `{ scope: "tree", id }`.
fn tree_fork(id: &str) -> ForkOptions {
    conformance::tree_fork(id)
}

/// upstream's `TEST_LANE_CONFIGURATION` (repo.test.ts:11-15).
fn test_lane_configuration() -> LaneConfiguration {
    LaneConfiguration {
        model: ModelIdentity {
            provider: "test".to_owned(),
            model_id: "test".to_owned(),
        },
        thinking_level: ThinkingLevel::Off,
        active_tool_names: Vec::new(),
    }
}

/// upstream's `IDLE_LANE_STATE` (repo.test.ts:16-20).
const fn idle_lane_state() -> LaneState {
    LaneState {
        current_operation_id: None,
        last_operation_id: None,
        inbox: Vec::new(),
    }
}

/// One custom-entry write without a data payload, upstream's
/// `insertEntry({ id, parentId, type: "custom", customType })`.
fn custom_entry_write(id: &str, parent_id: Option<&str>, custom_type: &str) -> Write {
    conformance::insert_entry_write(conformance::custom_entry(
        id,
        parent_id.map(str::to_owned),
        custom_type,
        None,
    ))
}

/// One message-entry write with the plain-string content the fixtures carry,
/// upstream's `insertEntry({ ..., type: "message", message: { role: "user",
/// content, timestamp } })`.
fn message_entry_write(id: &str, parent_id: Option<&str>, content: &str, timestamp: i64) -> Write {
    Write::Entry(Box::new(insert_entry(NewEntry::Message {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        body: Box::new(MessageEntry {
            message: AgentMessage::Standard(pi_ai::types::Message::User(UserMessage {
                content: UserContent::Text(content.to_owned()),
                timestamp,
            })),
            terminate: None,
        }),
    })))
}

/// One session-name write, upstream's `setValue(sessionName, name)`.
fn set_name_write(name: &str) -> Write {
    Write::ValueSet(values::set_value(&values::session_name(), name.to_owned()).expect("write"))
}

/// One usage-ledger write, upstream's `insertUsage({ id: "usage",
/// adjustment: false, usage })`.
fn usage_row_write() -> Write {
    Write::Usage(insert_usage(UsageWriteRow {
        id: "usage".to_owned(),
        usage: Usage {
            input: 1,
            output: 1,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 2,
            cost: UsageCost::default(),
        },
        entry_id: None,
        adjustment: false,
        details: None,
    }))
}

/// Commits one transaction through the facade, upstream's
/// `session.mutate((mutator) => mutator.commit(writes, BACKGROUND_CONTEXT))`.
async fn commit_writes(session: &SqliteOpenSession, writes: Vec<Write>) {
    session
        .mutate(
            StorageBackedSession::commit_writes_callback(writes),
            &background_context(),
        )
        .await
        .expect("commit");
}

/// The writer transaction the live-fork tests commit at the snapshot
/// boundary, upstream's `commitLaterSourceState` (repo.test.ts:210-234): an
/// entry child, its branch row, the branch tip update, three scalar
/// upserts, and the sessions counters update — six statements in one
/// transaction.
fn commit_later_source_state(db: &Arc<dyn SqliteDatabase>) -> Result<(), SessionError> {
    let tip = values::branch_tip("main").address;
    let name = values::session_name().address;
    let label = values::entry_label("root").address;
    let transaction_db = Arc::clone(db);
    db.transaction(Box::new(move || {
        sql!(
            "INSERT INTO entries (session_id, id, parent_id, seq, type, custom_type, timestamp, payload)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            "source",
            "child",
            "root",
            7i64,
            "message",
            None::<String>,
            2i64,
            serde_json::json!({ "message": { "role": "user", "content": "after", "timestamp": 2 } }).to_string(),
        )
        .run(transaction_db.as_ref())?;
        sql!(
            "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type)
			VALUES (?, ?, ?, ?, ?)",
            "source",
            "root",
            "child",
            7i64,
            "message",
        )
        .run(transaction_db.as_ref())?;
        sql!(
            "UPDATE branch_meta SET tip_entry_id = ?, tip_seq = ?
			WHERE session_id = ? AND branch_id = ?",
            "child",
            7i64,
            "source",
            "root",
        )
        .run(transaction_db.as_ref())?;
        for (namespace, key, seq, value) in [
            (tip.namespace.clone(), tip.key.clone(), 8i64, "child"),
            (name.namespace.clone(), name.key.clone(), 9i64, "after"),
            (label.namespace.clone(), label.key.clone(), 10i64, "after-label"),
        ] {
            sql!(
                "INSERT INTO scalar_values (session_id, namespace, key, seq, value)
				VALUES (?, ?, ?, ?, ?)
				ON CONFLICT(session_id, namespace, key) DO UPDATE SET seq = excluded.seq, value = excluded.value",
                "source",
                namespace,
                key,
                seq,
                serde_json::to_string(value)
                    .map_err(|error| SqliteAdapterError::new(error.to_string()))?,
            )
            .run(transaction_db.as_ref())?;
        }
        sql!("UPDATE sessions SET message_count = ?, next_seq = ? WHERE id = ?", 1i64, 11i64, "source")
            .run(transaction_db.as_ref())?;
        Ok(SqliteTransactionOutcome::Committed(Box::new(())))
    }))
    .map_err(SessionError::from)?;
    Ok(())
}

#[tokio::test]
async fn creates_one_branchless_initialized_session_file() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, NOW);

    let session = repo
        .create(create_options("session"), &background_context())
        .await
        .expect("create");
    let metadata = session.typed_metadata();
    assert_eq!(metadata.base.id, "session");
    assert_eq!(metadata.base.created_at, NOW);
    assert_eq!(metadata.base.storage_version, 1);
    assert_eq!(
        metadata.path,
        canonical_path(&join(&directory, &["session.sqlite"]))
    );

    with_db(&metadata.path, |db| {
        let count = sql!("SELECT COUNT(*) AS count FROM sessions")
            .get(db)
            .expect("sessions count")
            .expect("count row");
        assert_eq!(count.integer("count").expect("count"), 1);
        let row = sql!(
            "SELECT message_count, usage_payload, next_seq FROM sessions WHERE id = ?",
            "session"
        )
        .get(db)
        .expect("session row")
        .expect("session row");
        assert_eq!(row.integer("message_count").expect("message_count"), 0);
        // Structural usage compare (recorded restatement): serde_json renders
        // f64 zeros as 0.0 where JS renders 0.
        let usage: Usage =
            serde_json::from_str(&row.string("usage_payload").expect("usage_payload"))
                .expect("usage payload");
        assert_eq!(usage, Usage::default());
        assert_eq!(row.integer("next_seq").expect("next_seq"), 1);
        let scalars =
            sql!("SELECT namespace, key, seq, value FROM scalar_values WHERE session_id = ? ORDER BY seq", "session")
                .all(db)
                .expect("scalar values");
        assert!(scalars.is_empty());
        let list_count = sql!(
            "SELECT COUNT(*) AS count FROM list_values WHERE session_id = ?",
            "session"
        )
        .get(db)
        .expect("list count")
        .expect("count row");
        assert_eq!(list_count.integer("count").expect("count"), 0);
    });
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn exposes_explicit_branch_scans_through_the_open_session_facade() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, NOW);
    let session = repo
        .create(create_options("session"), &background_context())
        .await
        .expect("create");
    commit_writes(
        &session,
        vec![
            custom_entry_write("root", None, "root"),
            custom_entry_write("child", Some("root"), "child"),
        ],
    )
    .await;

    let scanned = session
        .scan_branch(
            &StorageBranchScan {
                start: "child".to_owned(),
                order: Some(BranchScanOrder::OldestFirst),
                ..StorageBranchScan::default()
            },
            &background_context(),
        )
        .await
        .expect("scan");
    assert_eq!(
        conformance::ids(&scanned),
        vec!["root".to_owned(), "child".to_owned()]
    );
    session.close(&background_context()).await.expect("close");
    let rejected = session
        .scan_branch(
            &StorageBranchScan {
                start: "child".to_owned(),
                ..StorageBranchScan::default()
            },
            &background_context(),
        )
        .await;
    assert!(
        rejected
            .expect_err("closed scan")
            .to_string()
            .contains("Session is closed"),
    );
}

#[tokio::test]
async fn commits_an_explicit_mutation_through_the_open_session_facade() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, NOW);
    let session = repo
        .create(create_options("session"), &background_context())
        .await
        .expect("create");
    let mutation = session
        .begin_mutation(&background_context())
        .await
        .expect("begin mutation");
    let result = mutation
        .commit(
            vec![Write::ValueSet(
                values::set_value(&values::session_name(), "explicit".to_owned()).expect("write"),
            )],
            &background_context(),
        )
        .await
        .expect("commit");

    assert_eq!(result.seqs.len(), 1);
    let stored = mutation
        .get_value(&values::session_name().address, &background_context())
        .await
        .expect("value")
        .expect("stored value");
    assert_eq!(stored.value, serde_json::json!("explicit"));
    mutation.end(&background_context()).await.expect("end");
    assert_eq!(
        session.get_name(&background_context()).await.expect("name"),
        Some("explicit".to_owned()),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn rejects_duplicate_create_without_deleting_the_existing_database() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, 1);
    let session = repo
        .create(create_options("session"), &background_context())
        .await
        .expect("create");
    let metadata = session.typed_metadata().clone();

    let rejected = repo
        .create(create_options("session"), &background_context())
        .await;
    assert!(rejected.is_err(), "duplicate create must reject");
    with_db(&metadata.path, |db| {
        let count = sql!("SELECT COUNT(*) AS count FROM sessions")
            .get(db)
            .expect("sessions count")
            .expect("count row");
        assert_eq!(count.integer("count").expect("count"), 1);
    });
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn lists_an_open_session_without_storage_layer_ownership_state() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, 1);
    let session = repo
        .create(create_options("session"), &background_context())
        .await
        .expect("create");
    let metadata = session.typed_metadata().clone();

    let listed = repo.list(&background_context()).expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].base.id, "session");
    assert_eq!(listed[0].path, metadata.path);
    with_db(&metadata.path, |db| {
        let row = sql!(
            "SELECT name FROM sqlite_master WHERE type = ? AND name = ?",
            "table",
            "writer_lease"
        )
        .get(db)
        .expect("sqlite master");
        assert!(row.is_none());
    });
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn ignores_a_stale_writer_lease_table_from_a_pre_wp07_wip_database() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, 1);
    let created = repo
        .create(create_options("session"), &background_context())
        .await
        .expect("create");
    let created_metadata = created.typed_metadata().clone();
    created.close(&background_context()).await.expect("close");
    with_db(&created_metadata.path, |db| {
        db.exec(
            "CREATE TABLE writer_lease (session_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, fence INTEGER NOT NULL, expires_at_ms INTEGER NOT NULL) WITHOUT ROWID",
        )
        .expect("create writer_lease");
        sql!(
            "INSERT INTO writer_lease (session_id, owner_id, fence, expires_at_ms)
			VALUES (?, ?, ?, ?)",
            "session",
            "stale",
            7i64,
            999i64,
        )
        .run(db)
        .expect("insert lease");
    });

    let reopened = repo
        .open(&created_metadata, &background_context())
        .await
        .expect("reopen");
    reopened
        .set_name(Some("works".to_owned()), &background_context())
        .await
        .expect("set name");
    reopened.close(&background_context()).await.expect("close");
    with_db(&created_metadata.path, |db| {
        let row = sql!(
            "SELECT owner_id, fence FROM writer_lease WHERE session_id = ?",
            "session"
        )
        .get(db)
        .expect("lease row")
        .expect("lease row");
        assert_eq!(row.string("owner_id").expect("owner_id"), "stale");
        assert_eq!(row.integer("fence").expect("fence"), 7);
    });
}

#[tokio::test]
async fn skips_corrupt_and_incompatible_files_during_list_discovery() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, 1);
    let session = repo
        .create(create_options("session"), &background_context())
        .await
        .expect("create");
    let metadata = session.typed_metadata().clone();
    session.close(&background_context()).await.expect("close");
    std::fs::write(
        join(&directory, &["corrupt.sqlite"]),
        "not a sqlite database",
    )
    .expect("write corrupt file");
    with_db(&metadata.path, |db| {
        sql!(
            "UPDATE sessions SET storage_version = ? WHERE id = ?",
            999i64,
            "session"
        )
        .run(db)
        .expect("bump storage version");
    });

    let listed = repo.list(&background_context()).expect("list");
    assert!(listed.is_empty());
}

#[tokio::test]
async fn does_not_remove_a_pre_existing_non_database_file_when_create_fails() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, 1);
    let path = join(&directory, &["session.sqlite"]);
    std::fs::write(&path, "not a sqlite database").expect("pre-write non-database file");

    let rejected = repo
        .create(create_options("session"), &background_context())
        .await;
    assert!(
        rejected.is_err(),
        "create over a non-database file must reject"
    );
    assert!(support::path_exists(&path));
}

#[tokio::test]
async fn rejects_delete_for_missing_files_and_deletes_a_closed_session() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, 1);
    let session = repo
        .create(create_options("session"), &background_context())
        .await
        .expect("create");
    let metadata = session.typed_metadata().clone();
    session.close(&background_context()).await.expect("close");

    let missing = SqliteSessionMetadata {
        base: metadata.base.clone(),
        path: join(&directory, &["missing.sqlite"]),
    };
    let rejected = repo.delete(&missing, &background_context()).await;
    assert!(rejected.is_err(), "delete of a missing file must reject");
    repo.delete(&metadata, &background_context())
        .await
        .expect("delete");
    assert!(!support::path_exists(&metadata.path));
}

#[tokio::test]
async fn closes_open_sessions_through_repo_close_and_rejects_later_operations() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, 1);
    let session = repo
        .create(create_options("session"), &background_context())
        .await
        .expect("create");

    repo.close(&background_context()).await.expect("close");

    let rejected = session.get_stats(&background_context()).await;
    assert!(
        rejected
            .expect_err("closed session")
            .to_string()
            .contains("Session is closed"),
    );
    let rejected = repo.list(&background_context());
    assert!(
        rejected
            .expect_err("closed repo")
            .to_string()
            .contains("SqliteSessionRepo is closed"),
    );
    let rejected = repo
        .create(create_options("other"), &background_context())
        .await;
    assert!(
        rejected
            .expect_err("closed repo")
            .to_string()
            .contains("SqliteSessionRepo is closed"),
    );
    let again = repo.close(&background_context()).await;
    assert!(again.is_ok(), "repo close is idempotent");
}

#[tokio::test]
async fn opens_a_session_through_the_version_gate_and_rejects_a_duplicate_local_handle() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, 1);
    let created = repo
        .create(create_options("session"), &background_context())
        .await
        .expect("create");
    let metadata = created.typed_metadata().clone();
    created.close(&background_context()).await.expect("close");

    let opened = repo
        .open(&metadata, &background_context())
        .await
        .expect("open");
    let opened_metadata = opened.typed_metadata();
    assert_eq!(opened_metadata.base.id, "session");
    assert_eq!(opened_metadata.path, metadata.path);
    let rejected = repo.open(&metadata, &background_context()).await;
    assert!(
        rejected
            .expect_err("duplicate handle")
            .to_string()
            .contains("already open"),
    );
    opened.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn isolates_sessions_stored_in_one_shared_sqlite_container() {
    let (_temp_guard, directory) = temp_dir();
    let database_path = join(&directory, &["sessions.sqlite"]);
    let repo = shared_repo(&directory, &database_path, NOW);
    let left = repo
        .create(create_options("left"), &background_context())
        .await
        .expect("create left");
    let right = repo
        .create(create_options("right"), &background_context())
        .await
        .expect("create right");

    assert_eq!(left.typed_metadata().path, canonical_path(&database_path));
    assert_eq!(right.typed_metadata().path, canonical_path(&database_path));
    commit_writes(
        &left,
        vec![
            custom_entry_write("left-root", None, "left"),
            set_name_write("left-name"),
        ],
    )
    .await;
    commit_writes(
        &right,
        vec![
            custom_entry_write("right-root", None, "right"),
            set_name_write("right-name"),
        ],
    )
    .await;

    let listed_ids: Vec<String> = repo
        .list(&background_context())
        .expect("list")
        .iter()
        .map(|metadata| metadata.base.id.clone())
        .collect();
    let mut sorted_ids = listed_ids.clone();
    sorted_ids.sort();
    assert_eq!(sorted_ids, vec!["left".to_owned(), "right".to_owned()]);
    let left_value = left
        .get_value(&values::session_name().address, &background_context())
        .await
        .expect("left value")
        .expect("stored value");
    assert_eq!(left_value.value, serde_json::json!("left-name"));
    let right_value = right
        .get_value(&values::session_name().address, &background_context())
        .await
        .expect("right value")
        .expect("stored value");
    assert_eq!(right_value.value, serde_json::json!("right-name"));
    let left_entries = left
        .get_entries(
            vec!["left-root".to_owned(), "right-root".to_owned()],
            &background_context(),
        )
        .await
        .expect("left entries");
    assert!(!left_entries.contains_key("right-root"));
    let right_entries = right
        .get_entries(
            vec!["left-root".to_owned(), "right-root".to_owned()],
            &background_context(),
        )
        .await
        .expect("right entries");
    assert!(!right_entries.contains_key("left-root"));

    let (left_closed, right_closed) = tokio::join!(
        left.close(&background_context()),
        right.close(&background_context())
    );
    left_closed.expect("close left");
    right_closed.expect("close right");
}

#[tokio::test]
async fn does_not_copy_usage_ledger_rows_when_forking() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, NOW);
    let source = repo
        .create(create_options("source"), &background_context())
        .await
        .expect("create source");
    commit_writes(
        &source,
        vec![
            custom_entry_write("root", None, "root"),
            message_entry_write("child", Some("root"), "child", 1),
            conformance::tip_write("main", Some("child")),
            conformance::lane_config_write("main", &test_lane_configuration()),
            conformance::lane_state_write("main", &idle_lane_state()),
            usage_row_write(),
        ],
    )
    .await;

    let source_metadata = source.typed_metadata().clone();
    source
        .close(&background_context())
        .await
        .expect("close source");
    let fork = repo
        .fork(
            &source_metadata,
            &branch_fork("main", Some("child"), "fork"),
            &background_context(),
        )
        .await
        .expect("fork");
    let fork_metadata = fork.typed_metadata().clone();

    with_db(&fork_metadata.path, |db| {
        let count = sql!(
            "SELECT COUNT(*) AS count FROM usage_ledger WHERE session_id = ?",
            "fork"
        )
        .get(db)
        .expect("ledger count")
        .expect("count row");
        assert_eq!(count.integer("count").expect("count"), 0);
    });
    let (_, _) = tokio::join!(
        source.close(&background_context()),
        fork.close(&background_context())
    );
}

#[tokio::test]
async fn does_not_create_databases_for_missing_open_list_fork_or_delete_targets() {
    let (_temp_guard, root) = temp_dir();
    let directory = join(&root, &["missing-directory"]);
    let repo = repo(&directory, 1);
    let missing = SqliteSessionMetadata {
        base: SessionMetadata {
            id: "missing".to_owned(),
            created_at: 1,
            storage_version: 1,
            cwd: None,
            parent_session_id: None,
            legacy_parent_session_path: None,
        },
        path: join(&directory, &["missing.sqlite"]),
    };

    let listed = repo.list(&background_context()).expect("list");
    assert!(listed.is_empty());
    assert!(!support::path_exists(&directory));
    let rejected = repo.open(&missing, &background_context()).await;
    assert!(rejected.is_err(), "open of a missing target must reject");
    let rejected = repo.delete(&missing, &background_context()).await;
    assert!(rejected.is_err(), "delete of a missing target must reject");
    let rejected = repo
        .fork(&missing, &tree_fork("fork"), &background_context())
        .await;
    assert!(rejected.is_err(), "fork of a missing target must reject");
    assert!(!support::path_exists(&missing.path));
    assert!(!support::path_exists(&join(&directory, &["fork.sqlite"])));
}

#[tokio::test]
async fn reserves_deletion_against_local_create_open_and_fork_destinations() {
    let (_temp_guard, directory) = temp_dir();
    let (factory, entered_rx, release_tx) = support::GatedOpenExistingFactory::new();
    let repo = Arc::new(SqliteSessionRepo::new(SqliteSessionRepoOptions {
        directory: directory.clone(),
        database_path: None,
        database_factory: factory.clone(),
        now: Some(fixed_clock(1)),
    }));
    let target = repo
        .create(create_options("target"), &background_context())
        .await
        .expect("create target");
    let source = repo
        .create(create_options("source"), &background_context())
        .await
        .expect("create source");
    let target_metadata = target.typed_metadata().clone();
    let source_metadata = source.typed_metadata().clone();
    let rejected = repo.delete(&target_metadata, &background_context()).await;
    assert!(
        rejected
            .expect_err("open-target delete")
            .to_string()
            .contains("already open"),
    );
    // The host closes a worker before deletion; only same-repository
    // exclusion is promised here.
    target
        .close(&background_context())
        .await
        .expect("close target");
    factory.arm();

    // The delete runs on a worker thread and parks inside its open_existing
    // at the gate, while this thread drives the exclusions.
    let runtime_handle = tokio::runtime::Handle::current();
    let worker_repo = Arc::clone(&repo);
    let worker_metadata = target_metadata.clone();
    let worker = std::thread::spawn(move || {
        runtime_handle.block_on(worker_repo.delete(&worker_metadata, &background_context()))
    });

    entered_rx.recv().expect("gate entered");
    let rejected = repo
        .create(create_options("target"), &background_context())
        .await;
    assert!(
        rejected
            .expect_err("reserved create")
            .to_string()
            .contains("already open"),
    );
    let rejected = repo.open(&target_metadata, &background_context()).await;
    assert!(
        rejected
            .expect_err("reserved open")
            .to_string()
            .contains("already open"),
    );
    let rejected = repo
        .fork(
            &source_metadata,
            &tree_fork("target"),
            &background_context(),
        )
        .await;
    assert!(
        rejected
            .expect_err("reserved fork")
            .to_string()
            .contains("already open"),
    );
    release_tx.send(()).expect("release");
    let delete_result = worker.join().expect("worker join");
    assert!(
        delete_result.is_ok(),
        "the released delete succeeds: {delete_result:?}"
    );
    source
        .close(&background_context())
        .await
        .expect("close source");
}

#[tokio::test]
async fn deletes_only_the_selected_session_from_a_shared_container() {
    let (_temp_guard, directory) = temp_dir();
    let database_path = join(&directory, &["shared.sqlite"]);
    let repo = shared_repo(&directory, &database_path, 1);
    let removed = repo
        .create(create_options("removed"), &background_context())
        .await
        .expect("create removed");
    let retained = repo
        .create(create_options("retained"), &background_context())
        .await
        .expect("create retained");
    removed
        .set_name(Some("removed".to_owned()), &background_context())
        .await
        .expect("set name");
    retained
        .set_name(Some("retained".to_owned()), &background_context())
        .await
        .expect("set name");
    let removed_metadata = removed.typed_metadata().clone();
    removed
        .close(&background_context())
        .await
        .expect("close removed");

    repo.delete(&removed_metadata, &background_context())
        .await
        .expect("delete");

    let listed_ids: Vec<String> = repo
        .list(&background_context())
        .expect("list")
        .iter()
        .map(|metadata| metadata.base.id.clone())
        .collect();
    assert_eq!(listed_ids, vec!["retained".to_owned()]);
    assert_eq!(
        retained
            .get_name(&background_context())
            .await
            .expect("name"),
        Some("retained".to_owned()),
    );
    let rejected = repo.open(&removed_metadata, &background_context()).await;
    assert!(
        rejected
            .expect_err("deleted session open")
            .to_string()
            .contains("Unknown SQLite session"),
    );
    retained
        .close(&background_context())
        .await
        .expect("close retained");
}

#[tokio::test]
async fn removes_per_file_wal_and_shm_sidecars() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, 1);
    let session = repo
        .create(create_options("session"), &background_context())
        .await
        .expect("create");
    let metadata = session.typed_metadata().clone();
    session.close(&background_context()).await.expect("close");
    std::fs::write(format!("{}-wal", metadata.path), "").expect("write wal sidecar");
    std::fs::write(format!("{}-shm", metadata.path), "").expect("write shm sidecar");

    repo.delete(&metadata, &background_context())
        .await
        .expect("delete");

    assert!(!support::path_exists(&metadata.path));
    assert!(!support::path_exists(&format!("{}-wal", metadata.path)));
    assert!(!support::path_exists(&format!("{}-shm", metadata.path)));
}

#[tokio::test]
async fn creates_the_parent_of_a_custom_shared_container_path() {
    let (_temp_guard, directory) = temp_dir();
    let database_path = join(&directory, &["nested", "containers", "sessions.sqlite"]);
    let repo = SqliteSessionRepo::new(SqliteSessionRepoOptions {
        directory: join(&directory, &["unrelated"]),
        database_path: Some(database_path.clone()),
        database_factory: Arc::new(create_rusqlite_factory()),
        now: Some(fixed_clock(1)),
    });

    let session = repo
        .create(create_options("session"), &background_context())
        .await
        .expect("create");

    assert_eq!(
        session.typed_metadata().path,
        canonical_path(&database_path)
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn encodes_unsafe_explicit_ids_inside_the_repository_directory_and_round_trips_them() {
    let (_temp_guard, directory) = temp_dir();
    let repo = repo(&directory, 1);
    let ids = [
        "../escape",
        "slash/id",
        "back\\slash",
        "percent%id",
        "..",
        "dots...id",
        "ユニコード",
    ];
    let mut metadata = Vec::new();
    for id in ids {
        let session = repo
            .create(
                SqliteSessionCreateOptions {
                    id: Some(id.to_owned()),
                    parent_session_id: None,
                },
                &background_context(),
            )
            .await
            .expect("create");
        metadata.push(session.typed_metadata().clone());
        session.close(&background_context()).await.expect("close");
    }
    let canonical_directory = canonical_path(&directory);
    for stored in &metadata {
        // Upstream's relative(canonicalDirectory, stored.path) containment
        // checks: the stored path stays inside the canonical directory.
        let from_directory = Path::new(&stored.path)
            .strip_prefix(&canonical_directory)
            .map_or_else(
                |_| stored.path.clone(),
                |relative| relative.to_string_lossy().into_owned(),
            );
        assert!(
            !Path::new(&from_directory).is_absolute(),
            "{} escaped: {}",
            stored.base.id,
            from_directory
        );
        assert!(
            !from_directory.starts_with(".."),
            "{} escaped: {}",
            stored.base.id,
            from_directory,
        );
        assert_eq!(
            Path::new(&stored.path)
                .parent()
                .map(|parent| parent.to_string_lossy().into_owned()),
            Some(canonical_directory.clone()),
        );
    }
    let mut listed_ids: Vec<String> = repo
        .list(&background_context())
        .expect("list")
        .iter()
        .map(|metadata| metadata.base.id.clone())
        .collect();
    listed_ids.sort();
    let mut expected_ids: Vec<String> = ids.iter().map(ToString::to_string).collect();
    expected_ids.sort();
    assert_eq!(listed_ids, expected_ids);
    for stored in &metadata {
        let reopened = repo
            .open(stored, &background_context())
            .await
            .expect("reopen");
        assert_eq!(reopened.typed_metadata().base.id, stored.base.id);
        reopened.close(&background_context()).await.expect("close");
    }
    let fork = repo
        .fork(&metadata[0], &tree_fork("fork/../%"), &background_context())
        .await
        .expect("fork");
    let fork_metadata = fork.typed_metadata();
    assert_eq!(fork_metadata.base.id, "fork/../%");
    assert_eq!(
        fork_metadata.base.parent_session_id,
        Some("../escape".to_owned())
    );
    assert_eq!(
        Path::new(&fork_metadata.path)
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned()),
        Some(canonical_directory.clone()),
    );
    fork.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn uses_exact_physical_identity_for_foreign_fork_sources_and_rejects_foreign_writable_metadata()
 {
    let (_temp_guard, root) = temp_dir();
    let left_repo = repo(&join(&root, &["left"]), 1);
    let right_repo = repo(&join(&root, &["right"]), 1);
    let left = left_repo
        .create(create_options("same"), &background_context())
        .await
        .expect("create left");
    let right = right_repo
        .create(create_options("same"), &background_context())
        .await
        .expect("create right");
    commit_writes(
        &left,
        vec![
            custom_entry_write("left-root", None, "left"),
            conformance::tip_write("main", Some("left-root")),
            conformance::lane_config_write("main", &test_lane_configuration()),
            conformance::lane_state_write("main", &idle_lane_state()),
            set_name_write("left"),
        ],
    )
    .await;
    commit_writes(
        &right,
        vec![
            custom_entry_write("right-root", None, "right"),
            conformance::tip_write("main", Some("right-root")),
            set_name_write("right"),
        ],
    )
    .await;

    let left_metadata = left.typed_metadata().clone();
    let fork = right_repo
        .fork(
            &left_metadata,
            &branch_fork("main", None, "fork-left"),
            &background_context(),
        )
        .await
        .expect("fork");
    let entries = fork
        .find_entries(Some(&conformance::asc_query()), &background_context())
        .await
        .expect("entries");
    assert_eq!(conformance::ids(&entries), vec!["left-root".to_owned()]);
    assert_eq!(
        fork.get_name(&background_context()).await.expect("name"),
        Some("left".to_owned())
    );
    right
        .close(&background_context())
        .await
        .expect("close right");
    let rejected = right_repo.open(&left_metadata, &background_context()).await;
    assert!(
        rejected
            .expect_err("foreign open")
            .to_string()
            .contains("outside this repository"),
    );
    let rejected = right_repo
        .delete(&left_metadata, &background_context())
        .await;
    assert!(
        rejected
            .expect_err("foreign delete")
            .to_string()
            .contains("outside this repository"),
    );
    let (left_closed, fork_closed) = tokio::join!(
        left.close(&background_context()),
        fork.close(&background_context())
    );
    left_closed.expect("close left");
    fork_closed.expect("close fork");
}

#[expect(
    clippy::too_many_lines,
    reason = "the per-layout fork body mirrors upstream's it body step for step; splitting the interleaving would scatter one ordering"
)]
async fn run_live_fork_layouts() {
    for layout in ["per-file", "shared"] {
        let (_temp_guard, directory) = temp_dir();
        let database_path = match layout {
            "shared" => Some(join(&directory, &["shared.sqlite"])),
            _ => None,
        };
        let worker_repo = database_path.as_ref().map_or_else(
            || repo(&directory, 1),
            |path| shared_repo(&directory, path, 1),
        );
        let source = worker_repo
            .create(create_options("source"), &background_context())
            .await
            .expect("create source");
        commit_writes(
            &source,
            vec![
                custom_entry_write("root", None, "root"),
                conformance::tip_write("main", Some("root")),
                conformance::lane_config_write("main", &test_lane_configuration()),
                conformance::lane_state_write("main", &idle_lane_state()),
                set_name_write("before"),
                conformance::label_write("root", "before-label"),
            ],
        )
        .await;
        let source_metadata = source.typed_metadata().clone();
        let writer_db: Arc<dyn SqliteDatabase> = Arc::from(
            create_rusqlite_factory()
                .open_existing(&source_metadata.path)
                .expect("writer db"),
        );
        writer_db
            .exec("PRAGMA busy_timeout = 5000;")
            .expect("writer busy timeout");
        let later_commit_completed = Arc::new(AtomicBool::new(false));
        let hook_db = Arc::clone(&writer_db);
        let hook_completed = Arc::clone(&later_commit_completed);
        let after_snapshot_established: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            if hook_completed.load(Ordering::SeqCst) {
                return;
            }
            commit_later_source_state(&hook_db).expect("later commit");
            hook_completed.store(true, Ordering::SeqCst);
        });
        let boundary_factory = Arc::new(support::SnapshotBoundaryFactory::new(
            after_snapshot_established,
        ));
        let boundary_count = Arc::clone(&boundary_factory);
        let server_repo = Arc::new(SqliteSessionRepo::new(SqliteSessionRepoOptions {
            directory: directory.clone(),
            database_path: database_path.clone(),
            database_factory: boundary_factory,
            now: Some(fixed_clock(2)),
        }));

        let first = server_repo
            .fork(
                &source_metadata,
                &branch_fork("main", None, "first-fork"),
                &background_context(),
            )
            .await
            .expect("first fork");

        assert!(
            later_commit_completed.load(Ordering::SeqCst),
            "the writer's later commit must have run at the snapshot boundary",
        );
        assert_eq!(
            boundary_count.read_only_open_count.load(Ordering::SeqCst),
            1
        );
        let entries = first
            .find_entries(Some(&conformance::asc_query()), &background_context())
            .await
            .expect("entries");
        assert_eq!(conformance::ids(&entries), vec!["root".to_owned()]);
        assert_eq!(
            first.get_name(&background_context()).await.expect("name"),
            Some("before".to_owned())
        );
        assert_eq!(
            first
                .get_label("root", &background_context())
                .await
                .expect("label"),
            Some("before-label".to_owned()),
        );
        let branch = first
            .branch("main", &background_context())
            .await
            .expect("branch")
            .expect("main branch");
        assert_eq!(
            branch.get_tip_id(&background_context()).await.expect("tip"),
            Some("root".to_owned())
        );
        assert_eq!(
            first
                .get_stats(&background_context())
                .await
                .expect("stats")
                .message_count,
            0
        );

        let second = server_repo
            .fork(
                &source_metadata,
                &branch_fork("main", None, "second-fork"),
                &background_context(),
            )
            .await
            .expect("second fork");
        assert_eq!(
            boundary_count.read_only_open_count.load(Ordering::SeqCst),
            2
        );
        let entries = second
            .find_entries(Some(&conformance::asc_query()), &background_context())
            .await
            .expect("entries");
        assert_eq!(
            conformance::ids(&entries),
            vec!["root".to_owned(), "child".to_owned()]
        );
        assert_eq!(
            second.get_name(&background_context()).await.expect("name"),
            Some("after".to_owned())
        );
        assert_eq!(
            second
                .get_label("root", &background_context())
                .await
                .expect("label"),
            Some("after-label".to_owned()),
        );
        let branch = second
            .branch("main", &background_context())
            .await
            .expect("branch")
            .expect("main branch");
        assert_eq!(
            branch.get_tip_id(&background_context()).await.expect("tip"),
            Some("child".to_owned())
        );
        assert_eq!(
            second
                .get_stats(&background_context())
                .await
                .expect("stats")
                .message_count,
            1
        );

        writer_db.close().expect("writer close");
        let (source_closed, first_closed, second_closed) = tokio::join!(
            source.close(&background_context()),
            first.close(&background_context()),
            second.close(&background_context()),
        );
        source_closed.expect("close source");
        first_closed.expect("close first");
        second_closed.expect("close second");
    }
}

#[tokio::test]
async fn forks_a_live_source_through_one_coherent_read_only_wal_snapshot() {
    run_live_fork_layouts().await;
}

#[tokio::test]
async fn waits_for_every_session_close_and_reports_one_or_multiple_failures_after_settlement() {
    let (_temp_guard, directory) = temp_dir();
    let multiple_factory = Arc::new(support::CloseTrackingFactory::new());
    let multiple_repo = Arc::new(SqliteSessionRepo::new(SqliteSessionRepoOptions {
        directory: join(&directory, &["multiple"]),
        database_path: None,
        database_factory: multiple_factory.clone(),
        now: Some(fixed_clock(1)),
    }));
    multiple_repo
        .create(create_options("first"), &background_context())
        .await
        .expect("create first");
    multiple_repo
        .create(create_options("second"), &background_context())
        .await
        .expect("create second");
    multiple_repo
        .create(create_options("third"), &background_context())
        .await
        .expect("create third");
    let first_error = SessionError::Message("first close failed".to_owned());
    let second_error = SessionError::Message("second close failed".to_owned());
    multiple_factory.fail_close(0, first_error.clone());
    multiple_factory.fail_close(1, second_error.clone());

    let multiple_close = multiple_repo.close(&background_context()).await;
    let second_close = multiple_repo.close(&background_context()).await;
    // The second close returns the same stored result, upstream's
    // `.toBe(multipleClose)` promise identity.
    assert_eq!(second_close, multiple_close);

    let Err(failure) = &multiple_close else {
        panic!("repo close must fail");
    };
    assert!(
        matches!(
            failure,
            SqliteRepoCloseError::Aggregate(errors)
                if *errors == vec![first_error.clone(), second_error.clone()],
        ),
        "two failures aggregate in close order: {failure:?}",
    );
    assert_eq!(failure.to_string(), "Failed to close SQLite Sessions");
    assert_eq!(multiple_factory.close_attempts(), vec![1, 1, 1]);

    let single_factory = Arc::new(support::CloseTrackingFactory::new());
    let single_repo = Arc::new(SqliteSessionRepo::new(SqliteSessionRepoOptions {
        directory: join(&directory, &["single"]),
        database_path: None,
        database_factory: single_factory.clone(),
        now: Some(fixed_clock(1)),
    }));
    single_repo
        .create(create_options("first"), &background_context())
        .await
        .expect("create first");
    single_repo
        .create(create_options("second"), &background_context())
        .await
        .expect("create second");
    let single_error = SessionError::Message("single close failed".to_owned());
    single_factory.fail_close(0, single_error.clone());

    let single_close = single_repo.close(&background_context()).await;
    // One failure rethrows alone, upstream's `rejects.toBe(singleError)`.
    assert_eq!(
        single_close,
        Err(SqliteRepoCloseError::Single(single_error))
    );
    assert_eq!(single_factory.close_attempts(), vec![1, 1]);
}

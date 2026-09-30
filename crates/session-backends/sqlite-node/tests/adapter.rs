//! The node:sqlite adapter suite, upstream's
//! `test/adapter.test.ts` `describe("node:sqlite adapter")`, ported 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatements: upstream's factory methods return Promises for interface
//! uniformity, so these tests were `async` only in form — the port's factory
//! is sync `Result` and the bodies stay sync under `#[tokio::test]` for
//! parity. The rejected async transaction callback (upstream's thenable
//! detection) reifies as `SqliteTransactionOutcome::Asynchronous`.

#![expect(clippy::expect_used, reason = "tests assert on results")]

mod support;

use std::any::Any;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use pi_session_backend_sqlite_node::create_rusqlite_factory;
use pi_session_backend_sqlite_node::sqlite::types::{
    SqliteAdapterError, SqliteDatabase, SqliteDatabaseFactory, SqliteParams, SqliteRunResult,
    SqliteTransactionOutcome, SqliteValue,
};

use support::path_exists;

/// Runs the body inside a fresh temp directory, upstream's `withTempDir`
/// (`mkdtemp(join(tmpdir(), "pi-sqlite-adapter-"))` + `rm -rf` in a finally);
/// the tempfile guard removes the tree on drop, including on unwind.
fn with_temp_dir(run: impl FnOnce(&str)) {
    let directory = tempfile::Builder::new()
        .prefix("pi-sqlite-adapter-")
        .tempdir()
        .expect("temp dir");
    run(directory.path().to_str().expect("temp dir path"));
}

/// Boxes the callback's value into the seam's `Any` outcome and downcasts it
/// back, upstream's generic `transaction<T>(callback: () => T): T` over the
/// port's erased callback.
fn transaction_typed<T: Any + Send>(
    db: &dyn SqliteDatabase,
    callback: impl FnOnce() -> Result<T, SqliteAdapterError> + Send + 'static,
) -> Result<T, SqliteAdapterError> {
    let outcome = db.transaction(Box::new(move || {
        callback().map(|value| {
            let committed: Box<dyn Any + Send> = Box::new(value);
            SqliteTransactionOutcome::Committed(committed)
        })
    }))?;
    outcome.downcast::<T>().map_or_else(
        |_| {
            Err(SqliteAdapterError::new(
                "transaction callback returned an unexpected type",
            ))
        },
        |value| Ok(*value),
    )
}

#[tokio::test]
async fn does_not_create_files_for_existing_or_read_only_opens() {
    with_temp_dir(|directory| {
        let path = Path::new(directory).join("missing % #.sqlite");
        let path = path.to_str().expect("utf8 path");
        let factory = create_rusqlite_factory();

        assert!(factory.open_existing(path).is_err());
        assert!(!path_exists(path));
        assert!(factory.open_read_only(path).is_err());
        assert!(!path_exists(path));
    });
}

#[tokio::test]
async fn opens_an_existing_database_read_write_or_read_only_without_changing_its_mode() {
    with_temp_dir(|directory| {
        let path = Path::new(directory).join("existing % #.sqlite");
        let path = path.to_str().expect("utf8 path");
        let factory = create_rusqlite_factory();
        let created = factory.open(path).expect("open");
        created
            .exec("CREATE TABLE values_table (value INTEGER NOT NULL)")
            .expect("create table");
        created.close().expect("close created");

        let writable = factory.open_existing(path).expect("open existing");
        writable
            .exec("INSERT INTO values_table (value) VALUES (1)")
            .expect("insert");
        writable.close().expect("close writable");
        let read_only = factory.open_read_only(path).expect("open read only");
        // Upstream wraps these assertions in try/finally so the read-only
        // handle closes even on failure; a panicking Rust test unwinds the
        // whole test, so the close runs on the success path only.
        let rows = read_only
            .prepare("SELECT value FROM values_table")
            .all(&SqliteParams::none())
            .expect("all");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].column_names().collect::<Vec<_>>(), vec!["value"]);
        assert_eq!(rows[0].get("value"), Some(&SqliteValue::Integer(1)));
        assert!(
            read_only
                .exec("INSERT INTO values_table (value) VALUES (2)")
                .is_err()
        );
        read_only.close().expect("close read only");
    });
}

#[tokio::test]
async fn commits_a_synchronous_transaction_and_returns_its_result() {
    let db: Arc<dyn SqliteDatabase> =
        Arc::from(create_rusqlite_factory().open(":memory:").expect("open"));
    db.exec("CREATE TABLE values_table (value INTEGER NOT NULL)")
        .expect("create table");

    let db_handle = Arc::clone(&db);
    let result = transaction_typed(db.as_ref(), move || {
        db_handle
            .prepare("INSERT INTO values_table (value) VALUES (?)")
            .run(&SqliteParams::Positional(vec![SqliteValue::Integer(42)]))?;
        Ok("committed".to_owned())
    })
    .expect("transaction");
    assert_eq!(result, "committed");

    let row = db
        .prepare("SELECT value FROM values_table")
        .get(&SqliteParams::none())
        .expect("get")
        .expect("row");
    assert_eq!(row.column_names().collect::<Vec<_>>(), vec!["value"]);
    assert_eq!(row.get("value"), Some(&SqliteValue::Integer(42)));

    db.close().expect("close");
}

#[tokio::test]
async fn forwards_positional_and_named_statement_parameters() {
    let db = create_rusqlite_factory().open(":memory:").expect("open");
    db.exec("CREATE TABLE values_table (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        .expect("create table");

    let run = db
        .prepare("INSERT INTO values_table (value) VALUES (?)")
        .run(&SqliteParams::Positional(vec![SqliteValue::Text(
            "positional".to_owned(),
        )]))
        .expect("insert positional");
    assert_eq!(
        run,
        SqliteRunResult {
            changes: 1,
            last_insert_rowid: 1
        }
    );

    let mut named = BTreeMap::new();
    named.insert("value".to_owned(), SqliteValue::Text("named".to_owned()));
    let run = db
        .prepare("INSERT INTO values_table (value) VALUES (:value)")
        .run(&SqliteParams::Named(named))
        .expect("insert named");
    assert_eq!(
        run,
        SqliteRunResult {
            changes: 1,
            last_insert_rowid: 2
        }
    );

    let row = db
        .prepare("SELECT value FROM values_table WHERE id = ?")
        .get(&SqliteParams::Positional(vec![SqliteValue::Integer(1)]))
        .expect("get")
        .expect("row");
    assert_eq!(row.column_names().collect::<Vec<_>>(), vec!["value"]);
    assert_eq!(
        row.get("value"),
        Some(&SqliteValue::Text("positional".to_owned()))
    );

    let mut named = BTreeMap::new();
    named.insert("id".to_owned(), SqliteValue::Integer(2));
    let rows = db
        .prepare("SELECT value FROM values_table WHERE id >= :id ORDER BY id")
        .all(&SqliteParams::Named(named))
        .expect("all");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].column_names().collect::<Vec<_>>(), vec!["value"]);
    assert_eq!(
        rows[0].get("value"),
        Some(&SqliteValue::Text("named".to_owned()))
    );

    db.close().expect("close");
}

#[tokio::test]
async fn rejects_asynchronous_transaction_callbacks() {
    let db: Arc<dyn SqliteDatabase> =
        Arc::from(create_rusqlite_factory().open(":memory:").expect("open"));
    db.exec("CREATE TABLE values_table (value INTEGER NOT NULL)")
        .expect("create table");

    // Upstream's callback is `async`: its first segment (the insert) runs
    // synchronously before the promise is observed; the port's rejected shape
    // runs the same insert then reports `Asynchronous`.
    let db_handle = Arc::clone(&db);
    let rejected = db.transaction(Box::new(move || {
        db_handle
            .prepare("INSERT INTO values_table (value) VALUES (?)")
            .run(&SqliteParams::Positional(vec![SqliteValue::Integer(42)]))?;
        Ok(SqliteTransactionOutcome::Asynchronous)
    }));
    let error = rejected.expect_err("transaction rejected");
    assert!(
        error
            .message()
            .contains("SQLite transaction callbacks must be synchronous")
    );
    // Upstream drains one microtask (`await Promise.resolve()`) before
    // reading; the Rust callback already ran to completion synchronously, so
    // there is nothing to drain.
    let rows = db
        .prepare("SELECT value FROM values_table")
        .all(&SqliteParams::none())
        .expect("all");
    assert!(rows.is_empty());

    db.close().expect("close");
}

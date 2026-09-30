//! The SQLite driver binding, upstream's `src/index.ts` node:sqlite adapter
//! restated over rusqlite.
//!
//! `rusqlite::Connection` is Send but not Sync, so the shared handle sits
//! behind a `Mutex<Option<Connection>>`; the `Option` is what makes
//! [`SqliteDatabase::close`] implementable (the connection must be consumed
//! to close it). Statements hold the handle's `Arc` and prepare per execution
//! through the connection's statement cache — upstream's adapter also
//! prepares fresh per call, and the once-prepared writer classes reuse the
//! same cache.
#![expect(
    clippy::significant_drop_tightening,
    reason = "the connection guard must span the prepare-execute-decode sequence; early drops would release the shared connection mid-statement"
)]

use std::any::Any;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use rusqlite::{Connection, OpenFlags, OptionalExtension, Params, ToSql};

use crate::sqlite::types::{
    SqliteAdapterError, SqliteDatabase, SqliteDatabaseFactory, SqliteParams, SqliteRow,
    SqliteRunResult, SqliteStatement, SqliteTransactionCallback, SqliteTransactionOutcome,
    SqliteValue,
};

/// The adapter's error for a use-after-close, this port's own message (node
/// throws a generic invalid-state error; no upstream test pins it).
const CLOSED_MESSAGE: &str = "SQLite database is closed";

type SharedConnection = Arc<Mutex<Option<Connection>>>;

fn lock_shared(shared: &SharedConnection) -> MutexGuard<'_, Option<Connection>> {
    shared.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A prepared statement over the shared connection, upstream's
/// `NodeSqliteStatement`.
#[derive(Debug)]
pub struct RusqliteStatement {
    connection: SharedConnection,
    sql: String,
}

impl SqliteStatement for RusqliteStatement {
    fn run(&self, params: &SqliteParams) -> Result<SqliteRunResult, SqliteAdapterError> {
        match params {
            SqliteParams::Positional(values) => {
                self.run_with(rusqlite::params_from_iter(values.iter()))
            }
            SqliteParams::Named(map) => self.run_with(named_binding(map).as_slice()),
        }
    }

    fn get(&self, params: &SqliteParams) -> Result<Option<SqliteRow>, SqliteAdapterError> {
        match params {
            SqliteParams::Positional(values) => {
                self.get_with(rusqlite::params_from_iter(values.iter()))
            }
            SqliteParams::Named(map) => self.get_with(named_binding(map).as_slice()),
        }
    }

    fn all(&self, params: &SqliteParams) -> Result<Vec<SqliteRow>, SqliteAdapterError> {
        match params {
            SqliteParams::Positional(values) => {
                self.materialized(rusqlite::params_from_iter(values.iter()))
            }
            SqliteParams::Named(map) => self.materialized(named_binding(map).as_slice()),
        }
    }

    fn iterate(&self, params: &SqliteParams) -> Result<Vec<SqliteRow>, SqliteAdapterError> {
        match params {
            SqliteParams::Positional(values) => {
                self.materialized(rusqlite::params_from_iter(values.iter()))
            }
            SqliteParams::Named(map) => self.materialized(named_binding(map).as_slice()),
        }
    }
}

impl RusqliteStatement {
    fn run_with<P: Params>(&self, params: P) -> Result<SqliteRunResult, SqliteAdapterError> {
        let guard = lock_shared(&self.connection);
        let connection = guard
            .as_ref()
            .ok_or_else(|| SqliteAdapterError::new(CLOSED_MESSAGE))?;
        let mut statement = connection.prepare_cached(&self.sql)?;
        if statement.readonly() {
            // node:sqlite runs row-returning statements and reports the stale
            // connection-level counters; rusqlite's execute would reject them
            // with ExecuteReturnedResults. The rows are stepped and discarded.
            let mut rows = statement.query(params)?;
            while rows.next()?.is_some() {}
            Ok(SqliteRunResult {
                changes: connection.changes(),
                last_insert_rowid: connection.last_insert_rowid(),
            })
        } else {
            let changes = statement.execute(params)?;
            Ok(SqliteRunResult {
                changes: u64::try_from(changes)
                    .map_err(|_| SqliteAdapterError::new("Statement changes overflow u64"))?,
                last_insert_rowid: connection.last_insert_rowid(),
            })
        }
    }

    fn get_with<P: Params>(&self, params: P) -> Result<Option<SqliteRow>, SqliteAdapterError> {
        let guard = lock_shared(&self.connection);
        let connection = guard
            .as_ref()
            .ok_or_else(|| SqliteAdapterError::new(CLOSED_MESSAGE))?;
        let mut statement = connection.prepare_cached(&self.sql)?;
        let names: Vec<String> = statement
            .column_names()
            .iter()
            .map(ToString::to_string)
            .collect();
        Ok(statement
            .query_row(params, |row| build_row(&names, row))
            .optional()?)
    }

    fn materialized<P: Params>(&self, params: P) -> Result<Vec<SqliteRow>, SqliteAdapterError> {
        let guard = lock_shared(&self.connection);
        let connection = guard
            .as_ref()
            .ok_or_else(|| SqliteAdapterError::new(CLOSED_MESSAGE))?;
        let mut statement = connection.prepare_cached(&self.sql)?;
        let names: Vec<String> = statement
            .column_names()
            .iter()
            .map(ToString::to_string)
            .collect();
        let rows = statement
            .query_map(params, |row| build_row(&names, row))?
            .collect::<rusqlite::Result<Vec<SqliteRow>>>()?;
        Ok(rows)
    }
}

/// The named-parameter slice rusqlite binds against `:name` markers, the
/// `&[(S, T)]` shape of its sealed `Params` trait.
fn named_binding(map: &BTreeMap<String, SqliteValue>) -> Vec<(&str, &dyn ToSql)> {
    map.iter()
        .map(|(name, value)| {
            let value: &dyn ToSql = value;
            (name.as_str(), value)
        })
        .collect()
}

fn build_row(names: &[String], row: &rusqlite::Row<'_>) -> rusqlite::Result<SqliteRow> {
    let mut columns = Vec::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        let value = row.get_ref(index)?;
        columns.push((name.clone(), SqliteValue::from(&value)));
    }
    Ok(SqliteRow::from_columns(columns))
}

/// A SQLite database over rusqlite, upstream's `NodeSqliteDatabase`.
#[derive(Debug)]
pub struct RusqliteDatabase {
    connection: SharedConnection,
}

impl RusqliteDatabase {
    fn new(connection: Connection) -> Self {
        Self {
            connection: Arc::new(Mutex::new(Some(connection))),
        }
    }

    fn with_connection<T>(
        &self,
        read: impl FnOnce(&Connection) -> Result<T, SqliteAdapterError>,
    ) -> Result<T, SqliteAdapterError> {
        let guard = lock_shared(&self.connection);
        let connection = guard
            .as_ref()
            .ok_or_else(|| SqliteAdapterError::new(CLOSED_MESSAGE))?;
        read(connection)
    }
}

impl SqliteDatabase for RusqliteDatabase {
    fn exec(&self, sql: &str) -> Result<(), SqliteAdapterError> {
        self.with_connection(|connection| Ok(connection.execute_batch(sql)?))
    }

    fn prepare(&self, sql: &str) -> Box<dyn SqliteStatement> {
        Box::new(RusqliteStatement {
            connection: self.connection.clone(),
            sql: sql.to_owned(),
        })
    }

    fn transaction(
        &self,
        callback: SqliteTransactionCallback,
    ) -> Result<Box<dyn Any + Send>, SqliteAdapterError> {
        // The begin sits outside the catch: a begin failure propagates raw,
        // like upstream's `sql\`BEGIN IMMEDIATE\`.exec(this)` before `try`.
        self.exec("BEGIN IMMEDIATE")?;
        match callback() {
            Ok(SqliteTransactionOutcome::Committed(value)) => match self.exec("COMMIT") {
                Ok(()) => Ok(value),
                Err(commit_error) => {
                    // Rollback failures are swallowed so the commit error wins,
                    // upstream's catch-comment verbatim.
                    let _ = self.exec("ROLLBACK");
                    Err(commit_error)
                }
            },
            Ok(SqliteTransactionOutcome::Asynchronous) => {
                let _ = self.exec("ROLLBACK");
                Err(SqliteAdapterError::new(
                    "SQLite transaction callbacks must be synchronous",
                ))
            }
            Err(callback_error) => {
                let _ = self.exec("ROLLBACK");
                Err(callback_error)
            }
        }
    }

    fn close(&self) -> Result<(), SqliteAdapterError> {
        let mut guard = lock_shared(&self.connection);
        if let Some(connection) = guard.take() {
            connection
                .close()
                .map_err(|(_, error)| SqliteAdapterError::from(error))?;
        }
        Ok(())
    }
}

/// Wraps an existing rusqlite connection in the seam, upstream's
/// `wrapNodeSqliteDatabase`.
#[must_use]
pub fn wrap_rusqlite_connection(connection: Connection) -> Box<dyn SqliteDatabase> {
    Box::new(RusqliteDatabase::new(connection))
}

/// The SQLite connection factory, upstream's `createNodeSqliteFactory`.
///
/// Open modes carry explicit flags: the create path uses read-write + create,
/// the no-create path drops create, and the read-only path drops write — so
/// no method ever URI-decodes the path (rusqlite's default flags include
/// `SQLITE_OPEN_URI`, which upstream's `openExisting` relied on for its
/// `mode=rw` URI).
#[derive(Debug, Default)]
pub struct RusqliteDatabaseFactory;

impl SqliteDatabaseFactory for RusqliteDatabaseFactory {
    fn open(&self, path: &str) -> Result<Box<dyn SqliteDatabase>, SqliteAdapterError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        Ok(Box::new(RusqliteDatabase::new(
            Connection::open_with_flags(Path::new(path), flags)?,
        )))
    }

    fn open_existing(&self, path: &str) -> Result<Box<dyn SqliteDatabase>, SqliteAdapterError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        Ok(Box::new(RusqliteDatabase::new(
            Connection::open_with_flags(Path::new(path), flags)?,
        )))
    }

    fn open_read_only(&self, path: &str) -> Result<Box<dyn SqliteDatabase>, SqliteAdapterError> {
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        Ok(Box::new(RusqliteDatabase::new(
            Connection::open_with_flags(Path::new(path), flags)?,
        )))
    }
}

/// The factory constructor, upstream's `createNodeSqliteFactory`.
#[must_use]
pub const fn create_rusqlite_factory() -> RusqliteDatabaseFactory {
    RusqliteDatabaseFactory
}

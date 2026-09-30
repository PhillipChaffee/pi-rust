//! The structural SQLite capability seam, upstream's `src/sqlite/types.ts`.
//!
//! Upstream defines structural interfaces (`SqliteDatabase`, `SqliteStatement`,
//! `SqliteDatabaseFactory`) so test doubles can duck-type them; this port keeps
//! them as traits because the tests wrap the seam (transaction counting, gated
//! opens, statement interception, close tracking). The value, parameter, row,
//! and result shapes restate node:sqlite's built-in value model, which
//! TypeScript gets from the language itself.

use std::any::Any;
use std::collections::BTreeMap;

use rusqlite::ToSql;
use rusqlite::types::{ToSqlOutput, Value, ValueRef};

/// A value bound into or read out of a SQLite statement, the port's
/// `node:sqlite` `SQLInputValue`/row-value model.
#[derive(Clone, Debug, PartialEq)]
pub enum SqliteValue {
    /// SQL NULL.
    Null,
    /// SQL INTEGER.
    Integer(i64),
    /// SQL REAL.
    Real(f64),
    /// SQL TEXT.
    Text(String),
    /// SQL BLOB.
    Blob(Vec<u8>),
}

impl ToSql for SqliteValue {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        let value = match self {
            Self::Null => Value::Null,
            Self::Integer(value) => Value::Integer(*value),
            Self::Real(value) => Value::Real(*value),
            Self::Text(value) => Value::Text(value.clone()),
            Self::Blob(value) => Value::Blob(value.clone()),
        };
        Ok(ToSqlOutput::Owned(value))
    }
}

impl From<&ValueRef<'_>> for SqliteValue {
    fn from(value: &ValueRef<'_>) -> Self {
        match value {
            ValueRef::Null => Self::Null,
            ValueRef::Integer(value) => Self::Integer(*value),
            ValueRef::Real(value) => Self::Real(*value),
            // node:sqlite decodes TEXT through a lossy UTF-8 view; the same
            // tolerance keeps a foreign container readable instead of erroring.
            ValueRef::Text(bytes) => Self::Text(String::from_utf8_lossy(bytes).into_owned()),
            ValueRef::Blob(bytes) => Self::Blob(bytes.to_vec()),
        }
    }
}

impl SqliteValue {
    /// Widens one interpolated expression into a parameter, the crate's
    /// `sql!` macro's interpolation rule (upstream: a template interpolation
    /// becomes a `?` parameter).
    pub fn from_param(value: impl Into<Self>) -> Self {
        value.into()
    }
}

impl From<&str> for SqliteValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<&String> for SqliteValue {
    fn from(value: &String) -> Self {
        Self::Text(value.clone())
    }
}

impl From<String> for SqliteValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<i64> for SqliteValue {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<i32> for SqliteValue {
    fn from(value: i32) -> Self {
        Self::Integer(i64::from(value))
    }
}

impl From<f64> for SqliteValue {
    fn from(value: f64) -> Self {
        Self::Real(value)
    }
}

impl From<Vec<u8>> for SqliteValue {
    fn from(value: Vec<u8>) -> Self {
        Self::Blob(value)
    }
}

impl<T: Into<Self>> From<Option<T>> for SqliteValue {
    fn from(value: Option<T>) -> Self {
        value.map_or(Self::Null, Into::into)
    }
}

/// Result of a prepared SQLite statement execution, upstream's `SqliteRunResult`.
///
/// Upstream declares `lastInsertRowid` optional; the adapter always reports the
/// connection-level `sqlite3_last_insert_rowid` value (0 when no insert has
/// ever run on the connection, stale otherwise — verified against node:sqlite,
/// which returns the same connection-level value even for UPDATE/DELETE).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SqliteRunResult {
    /// Rows changed by the statement, upstream's `changes`.
    pub changes: u64,
    /// The connection's most recent insert rowid, upstream's `lastInsertRowid`.
    pub last_insert_rowid: i64,
}

/// An object-shaped result row, the port of node:sqlite's row objects.
///
/// Columns keep their result order; lookups are by column name, which is all
/// the upstream `row.col` object access expresses.
#[derive(Clone, Debug, PartialEq)]
pub struct SqliteRow {
    columns: Vec<(String, SqliteValue)>,
}

impl SqliteRow {
    /// Builds a row from its ordered (name, value) columns; the driver
    /// binding's constructor.
    pub(crate) const fn from_columns(columns: Vec<(String, SqliteValue)>) -> Self {
        Self { columns }
    }

    /// The declared column names, in result order.
    pub fn column_names(&self) -> impl Iterator<Item = &str> {
        self.columns.iter().map(|(name, _)| name.as_str())
    }

    /// The raw value bound to a column, `row.col` in object terms.
    #[must_use]
    pub fn get(&self, column: &str) -> Option<&SqliteValue> {
        self.columns
            .iter()
            .find(|(name, _)| name == column)
            .map(|(_, value)| value)
    }

    /// A TEXT column decoded to a string.
    ///
    /// # Errors
    /// When the column is absent or does not hold TEXT.
    pub fn string(&self, column: &str) -> Result<String, SqliteAdapterError> {
        match self.get(column) {
            Some(SqliteValue::Text(value)) => Ok(value.clone()),
            Some(value) => Err(unexpected_type(column, value)),
            None => Err(missing_column(column)),
        }
    }

    /// A TEXT column decoded to a string, absent mapping to `None`.
    ///
    /// # Errors
    /// When the column does not hold TEXT.
    pub fn opt_string(&self, column: &str) -> Result<Option<String>, SqliteAdapterError> {
        match self.get(column) {
            None | Some(SqliteValue::Null) => Ok(None),
            Some(SqliteValue::Text(value)) => Ok(Some(value.clone())),
            Some(value) => Err(unexpected_type(column, value)),
        }
    }

    /// An INTEGER column.
    ///
    /// # Errors
    /// When the column is absent or does not hold an INTEGER.
    pub fn integer(&self, column: &str) -> Result<i64, SqliteAdapterError> {
        match self.get(column) {
            Some(SqliteValue::Integer(value)) => Ok(*value),
            Some(value) => Err(unexpected_type(column, value)),
            None => Err(missing_column(column)),
        }
    }

    /// A nullable INTEGER column.
    ///
    /// # Errors
    /// When the column does not hold an INTEGER.
    pub fn opt_integer(&self, column: &str) -> Result<Option<i64>, SqliteAdapterError> {
        match self.get(column) {
            None | Some(SqliteValue::Null) => Ok(None),
            Some(SqliteValue::Integer(value)) => Ok(Some(*value)),
            Some(value) => Err(unexpected_type(column, value)),
        }
    }

    /// A REAL column.
    ///
    /// # Errors
    /// When the column is absent or does not hold a REAL.
    pub fn real(&self, column: &str) -> Result<f64, SqliteAdapterError> {
        match self.get(column) {
            Some(SqliteValue::Real(value)) => Ok(*value),
            Some(value) => Err(unexpected_type(column, value)),
            None => Err(missing_column(column)),
        }
    }
}

fn missing_column(column: &str) -> SqliteAdapterError {
    SqliteAdapterError::new(format!("Row has no column {column}"))
}

fn unexpected_type(column: &str, value: &SqliteValue) -> SqliteAdapterError {
    SqliteAdapterError::new(format!("Unexpected type for column {column}: {value:?}"))
}

/// Statement parameters, upstream's variadic `run(...params)`.
///
/// A single object argument is a named-parameter record
/// (`isNamedParameters`); the runtime shape check restates as this
/// type-level distinction.
#[derive(Clone, Debug, PartialEq)]
pub enum SqliteParams {
    /// Positional `?` parameters.
    Positional(Vec<SqliteValue>),
    /// Named (`:name`) parameters.
    Named(BTreeMap<String, SqliteValue>),
}

impl SqliteParams {
    /// No parameters.
    #[must_use]
    pub const fn none() -> Self {
        Self::Positional(Vec::new())
    }
}

/// An error from the SQLite seam: a driver failure or one of the adapter's
/// own contract errors. Upstream throws plain exceptions here; the port
/// carries the message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SqliteAdapterError {
    message: String,
}

impl SqliteAdapterError {
    /// An adapter error carrying `message` verbatim.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The error message, upstream's exception message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for SqliteAdapterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for SqliteAdapterError {}

impl From<rusqlite::Error> for SqliteAdapterError {
    fn from(error: rusqlite::Error) -> Self {
        Self::new(error.to_string())
    }
}

impl From<pi_agent_core::harness::session::types::SessionError> for SqliteAdapterError {
    fn from(error: pi_agent_core::harness::session::types::SessionError) -> Self {
        Self::new(error.to_string())
    }
}

impl From<SqliteAdapterError> for pi_agent_core::harness::session::types::SessionError {
    fn from(error: SqliteAdapterError) -> Self {
        Self::Message(error.message().to_owned())
    }
}

/// Widens a contract `u64` (sequence, scan bound, limit) to the INTEGER the
/// column binds. SQLite INTEGER is i64; upstream's JS numbers silently lose
/// precision past 2^53 and this port refuses instead.
///
/// # Errors
/// `Integer out of SQLite range: {value}` when the value exceeds `i64::MAX`.
pub fn sql_integer(
    value: u64,
) -> Result<i64, pi_agent_core::harness::session::types::SessionError> {
    i64::try_from(value).map_err(|_| {
        pi_agent_core::harness::session::types::SessionError::Message(format!(
            "Integer out of SQLite range: {value}"
        ))
    })
}

/// Narrows a read-back INTEGER to the contract's `u64`.
///
/// # Errors
/// `Integer out of range: {value}` when negative — reachable only from a
/// foreign or corrupt container.
pub fn sql_u64(value: i64) -> Result<u64, pi_agent_core::harness::session::types::SessionError> {
    u64::try_from(value).map_err(|_| {
        pi_agent_core::harness::session::types::SessionError::Message(format!(
            "Integer out of range: {value}"
        ))
    })
}

/// A prepared SQLite statement capability, upstream's `SqliteStatement`.
pub trait SqliteStatement: Send + Sync {
    /// Execute a statement that changes rows, upstream's `run`.
    ///
    /// # Errors
    /// A driver failure, including prepare errors (preparation is deferred to
    /// first execution).
    fn run(&self, params: &SqliteParams) -> Result<SqliteRunResult, SqliteAdapterError>;
    /// Read the first row, upstream's `get`.
    ///
    /// # Errors
    /// A driver failure, including prepare errors.
    fn get(&self, params: &SqliteParams) -> Result<Option<SqliteRow>, SqliteAdapterError>;
    /// Read every row, upstream's `all`.
    ///
    /// # Errors
    /// A driver failure, including prepare errors.
    fn all(&self, params: &SqliteParams) -> Result<Vec<SqliteRow>, SqliteAdapterError>;
    /// Read every row; upstream's `iterate` materializes because a streaming
    /// iterator cannot hold the shared connection lock in this port.
    ///
    /// # Errors
    /// A driver failure, including prepare errors.
    fn iterate(&self, params: &SqliteParams) -> Result<Vec<SqliteRow>, SqliteAdapterError>;
}

/// The outcome a transaction callback reports, upstream's callback return.
///
/// Upstream detects a thenable return at runtime (`isAsyncResult`) and throws
/// `TypeError("SQLite transaction callbacks must be synchronous")`; the
/// rejected shape reifies as [`SqliteTransactionOutcome::Asynchronous`], which
/// the adapter rejects with that exact message after rolling back.
pub enum SqliteTransactionOutcome {
    /// The callback's value, committed.
    Committed(Box<dyn Any + Send>),
    /// The rejected async shape; the adapter rolls back and errors.
    Asynchronous,
}

impl std::fmt::Debug for SqliteTransactionOutcome {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Committed(_) => formatter.write_str("SqliteTransactionOutcome::Committed(..)"),
            Self::Asynchronous => formatter.write_str("SqliteTransactionOutcome::Asynchronous"),
        }
    }
}

/// A synchronous transaction body, upstream's `() => T` callback.
pub type SqliteTransactionCallback =
    Box<dyn FnOnce() -> Result<SqliteTransactionOutcome, SqliteAdapterError> + Send>;

/// A SQLite database capability, upstream's `SqliteDatabase`.
///
/// The port runs on the same single-threaded substrate as upstream's event
/// loop: a transaction body must not interleave with other statements on the
/// same connection, and callers never issue statements concurrently with a
/// transaction body. (Holding the adapter lock across the callback would
/// deadlock the callback's own prepare calls.)
pub trait SqliteDatabase: Send + Sync {
    /// Execute one or more statements, ignoring any rows, upstream's `exec`.
    ///
    /// # Errors
    /// A driver failure.
    fn exec(&self, sql: &str) -> Result<(), SqliteAdapterError>;
    /// Prepare a statement, upstream's `prepare`.
    fn prepare(&self, sql: &str) -> Box<dyn SqliteStatement>;
    /// Run a synchronous write transaction, upstream's `transaction`: begin
    /// immediate, run the callback, commit; roll back and rethrow on callback
    /// error or the rejected async shape.
    ///
    /// # Errors
    /// The begin, commit, or callback error; the async-callback contract
    /// error.
    fn transaction(
        &self,
        callback: SqliteTransactionCallback,
    ) -> Result<Box<dyn Any + Send>, SqliteAdapterError>;
    /// Close the connection, upstream's `close`. Idempotent here: a second
    /// close returns success.
    ///
    /// # Errors
    /// A driver failure while closing.
    fn close(&self) -> Result<(), SqliteAdapterError>;
}

/// A SQLite connection factory, upstream's `SqliteDatabaseFactory`.
pub trait SqliteDatabaseFactory: Send + Sync {
    /// Open a writable database, creating it when absent, upstream's `open`.
    ///
    /// # Errors
    /// A driver failure.
    fn open(&self, path: &str) -> Result<Box<dyn SqliteDatabase>, SqliteAdapterError>;
    /// Open a writable database without creating it, upstream's `openExisting`.
    ///
    /// # Errors
    /// A driver failure, including a missing file.
    fn open_existing(&self, path: &str) -> Result<Box<dyn SqliteDatabase>, SqliteAdapterError>;
    /// Open an existing database read-only, upstream's `openReadOnly`.
    ///
    /// # Errors
    /// A driver failure, including a missing file.
    fn open_read_only(&self, path: &str) -> Result<Box<dyn SqliteDatabase>, SqliteAdapterError>;
}

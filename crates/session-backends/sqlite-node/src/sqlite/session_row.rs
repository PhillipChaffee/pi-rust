//! The `sessions` row, upstream's `src/sqlite/session/session-row.ts`.

use pi_agent_core::harness::session::types::{SessionError, SessionMetadata};
use pi_ai::types::Usage;

use crate::sql;
use crate::sqlite::types::{SqliteDatabase, SqliteRow};

/// The `sessions` table row, upstream's `SessionRow`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionRow {
    /// The session id.
    pub id: String,
    /// The creation timestamp, epoch milliseconds.
    pub created_at: i64,
    /// The fork parent, when the session is a fork.
    pub parent_session_id: Option<String>,
    /// The schema version the row was written under.
    pub storage_version: i64,
    /// The always-null metadata column.
    pub metadata: Option<String>,
    /// The maintained message count.
    pub message_count: i64,
    /// The maintained usage JSON.
    pub usage_payload: String,
    /// The next storage sequence the commit pipeline would assign.
    pub next_seq: i64,
}

/// The session metadata this backend carries, upstream's
/// `SqliteSessionMetadata` extending `SessionMetadata`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SqliteSessionMetadata {
    /// The base contract fields.
    pub base: SessionMetadata,
    /// The SQLite container/shard path containing this session.
    pub path: String,
}

impl SqliteSessionMetadata {
    /// The erased contract metadata, upstream's spread into the base shape.
    #[must_use]
    pub fn base(&self) -> SessionMetadata {
        self.base.clone()
    }
}

/// The all-zero usage the session row seeds with, upstream's `zeroUsage()`.
#[must_use]
pub fn zero_usage() -> Usage {
    Usage::default()
}

fn session_row(row: &SqliteRow) -> Result<SessionRow, SessionError> {
    Ok(SessionRow {
        id: row.string("id")?,
        created_at: row.integer("created_at")?,
        parent_session_id: row.opt_string("parent_session_id")?,
        storage_version: row.integer("storage_version")?,
        metadata: row.opt_string("metadata")?,
        message_count: row.integer("message_count")?,
        usage_payload: row.string("usage_payload")?,
        next_seq: row.integer("next_seq")?,
    })
}

/// Reads one session row, upstream's `readSessionRow`.
///
/// # Errors
/// `Unknown SQLite session: {id}` when absent; a driver failure otherwise.
pub fn read_session_row(
    db: &dyn SqliteDatabase,
    session_id: &str,
) -> Result<SessionRow, SessionError> {
    let row = sql!(
        "SELECT id, created_at, parent_session_id, storage_version, metadata,
			message_count, usage_payload, next_seq
		FROM sessions
		WHERE id = ?",
        session_id
    )
    .get(db)?
    .ok_or_else(|| SessionError::Message(format!("Unknown SQLite session: {session_id}")))?;
    session_row(&row)
}

/// Reads every session row, upstream's `readAllSessionRows`.
///
/// # Errors
/// A driver failure.
pub fn read_all_session_rows(db: &dyn SqliteDatabase) -> Result<Vec<SessionRow>, SessionError> {
    let rows = sql!(
        "SELECT id, created_at, parent_session_id, storage_version, metadata,
			message_count, usage_payload, next_seq
		FROM sessions"
    )
    .all(db)?;
    rows.iter().map(session_row).collect()
}

/// Whether the row exists, upstream's `hasSessionRow`.
///
/// # Errors
/// A driver failure.
pub fn has_session_row(db: &dyn SqliteDatabase, session_id: &str) -> Result<bool, SessionError> {
    Ok(sql!("SELECT id FROM sessions WHERE id = ?", session_id)
        .get(db)?
        .is_some())
}

/// Builds the session metadata from a row, upstream's
/// `metadataFromSessionRow`: the storage-version gate first.
///
/// # Errors
/// `SQLite session storage version {row} is newer than {current}` when the
/// container is newer than this backend; `SQLite session storage version
/// {row} requires migrations` when older.
pub fn metadata_from_session_row(
    path: &str,
    row: &SessionRow,
    current_storage_version: u32,
) -> Result<SqliteSessionMetadata, SessionError> {
    let current = i64::from(current_storage_version);
    if row.storage_version > current {
        return Err(SessionError::Message(format!(
            "SQLite session storage version {} is newer than {current}",
            row.storage_version
        )));
    }
    if row.storage_version < current {
        return Err(SessionError::Message(format!(
            "SQLite session storage version {} requires migrations",
            row.storage_version
        )));
    }
    Ok(SqliteSessionMetadata {
        base: SessionMetadata {
            id: row.id.clone(),
            created_at: row.created_at,
            storage_version: u32::try_from(row.storage_version).map_err(|_| {
                SessionError::Message(format!("Integer out of range: {}", row.storage_version))
            })?,
            cwd: None,
            parent_session_id: row.parent_session_id.clone(),
            legacy_parent_session_path: None,
        },
        path: path.to_owned(),
    })
}

/// Inserts the seed session row, upstream's `insertSessionRow`: zero usage,
/// zero message count, and a NULL metadata column.
///
/// # Errors
/// A driver failure or an out-of-range integer.
pub fn insert_session_row(
    db: &dyn SqliteDatabase,
    metadata: &SqliteSessionMetadata,
    storage_version: u32,
    next_seq: u64,
) -> Result<(), SessionError> {
    sql!(
        "INSERT INTO sessions
			(id, created_at, parent_session_id, storage_version, metadata, message_count, usage_payload, next_seq)
		VALUES (
			?,
			?,
			?,
			?,
			?,
			?,
			?,
			?
		)",
        &metadata.base.id,
        metadata.base.created_at,
        metadata.base.parent_session_id.clone(),
        i64::from(storage_version),
        None::<String>,
        0i64,
        serde_json::to_string(&zero_usage()).map_err(|error| SessionError::Message(error.to_string()))?,
        crate::sqlite::types::sql_integer(next_seq)?,
    )
    .run(db)?;
    Ok(())
}

/// Deletes every row of one session in upstream's order, upstream's
/// `deleteSessionRows`.
///
/// # Errors
/// `Expected to delete one SQLite session {id}, deleted {n}` when the final
/// delete misses; a driver failure otherwise.
pub fn delete_session_rows(db: &dyn SqliteDatabase, session_id: &str) -> Result<(), SessionError> {
    sql!("DELETE FROM entries WHERE session_id = ?", session_id).run(db)?;
    sql!("DELETE FROM scalar_values WHERE session_id = ?", session_id).run(db)?;
    sql!("DELETE FROM list_values WHERE session_id = ?", session_id).run(db)?;
    sql!("DELETE FROM usage_ledger WHERE session_id = ?", session_id).run(db)?;
    sql!(
        "DELETE FROM branch_entries WHERE session_id = ?",
        session_id
    )
    .run(db)?;
    sql!("DELETE FROM branch_meta WHERE session_id = ?", session_id).run(db)?;
    let result = sql!("DELETE FROM sessions WHERE id = ?", session_id).run(db)?;
    if result.changes != 1 {
        return Err(SessionError::Message(format!(
            "Expected to delete one SQLite session {session_id}, deleted {}",
            result.changes
        )));
    }
    Ok(())
}

//! The `sessions.next_seq` counter, upstream's
//! `src/sqlite/session/session-sequences.ts`.

use pi_agent_core::harness::session::types::SessionError;

use crate::sql;
use crate::sqlite::types::{SqliteDatabase, sql_integer, sql_u64};

/// Reads the next storage sequence, upstream's `readNextSeq`.
///
/// # Errors
/// `Unknown SQLite session: {id}` when absent; a driver failure otherwise.
pub fn read_next_seq(db: &dyn SqliteDatabase, session_id: &str) -> Result<u64, SessionError> {
    let row = sql!("SELECT next_seq FROM sessions WHERE id = ?", session_id)
        .get(db)?
        .ok_or_else(|| SessionError::Message(format!("Unknown SQLite session: {session_id}")))?;
    sql_u64(row.integer("next_seq")?)
}

/// Advances the next storage sequence, upstream's `advanceNextSeq`.
///
/// # Errors
/// `Expected to update one SQLite session {id}, updated {n}` when the update
/// misses; a driver failure otherwise.
pub fn advance_next_seq(
    db: &dyn SqliteDatabase,
    session_id: &str,
    next_seq: u64,
) -> Result<(), SessionError> {
    let result = sql!(
        "UPDATE sessions SET next_seq = ? WHERE id = ?",
        sql_integer(next_seq)?,
        session_id
    )
    .run(db)?;
    if result.changes != 1 {
        return Err(SessionError::Message(format!(
            "Expected to update one SQLite session {session_id}, updated {}",
            result.changes
        )));
    }
    Ok(())
}

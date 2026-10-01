//! The maintained `sessions` stats, upstream's
//! `src/sqlite/session/session-stats.ts`.

use pi_agent_core::harness::session::types::{SessionError, SessionStats};
use pi_ai::types::Usage;

use crate::sql;
use crate::sqlite::session_row::read_session_row;
use crate::sqlite::types::SqliteDatabase;

/// Reads the maintained stats, upstream's `readSessionStats`: the message
/// count column plus the usage JSON.
///
/// # Errors
/// `Unknown SQLite session: {id}` when absent; a driver or JSON failure
/// otherwise.
pub fn read_session_stats(
    db: &dyn SqliteDatabase,
    session_id: &str,
) -> Result<SessionStats, SessionError> {
    let row = read_session_row(db, session_id)?;
    let usage = serde_json::from_str::<Usage>(&row.usage_payload)
        .map_err(|error| SessionError::Message(error.to_string()))?;
    let message_count = u64::try_from(row.message_count).map_err(|_| {
        SessionError::Message(format!("Integer out of range: {}", row.message_count))
    })?;
    Ok(SessionStats {
        message_count,
        usage,
    })
}

/// Bumps the message count, upstream's `incrementMessageCount`.
///
/// # Errors
/// A driver failure.
pub fn increment_message_count(
    db: &dyn SqliteDatabase,
    session_id: &str,
) -> Result<(), SessionError> {
    sql!(
        "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
        session_id
    )
    .run(db)?;
    Ok(())
}

/// Adds one usage into the maintained usage JSON, upstream's
/// `addUsageToSessionStats`: read-modify-write in the caller's transaction.
///
/// # Errors
/// `Unknown SQLite session: {id}` when absent; a driver or JSON failure
/// otherwise.
pub fn add_usage_to_session_stats(
    db: &dyn SqliteDatabase,
    session_id: &str,
    usage: &Usage,
) -> Result<(), SessionError> {
    let current = read_session_stats(db, session_id)?.usage;
    let next = pi_agent_core::harness::utils::usage::add_usage(current, *usage);
    sql!(
        "UPDATE sessions SET usage_payload = ? WHERE id = ?",
        serde_json::to_string(&next).map_err(|error| SessionError::Message(error.to_string()))?,
        session_id
    )
    .run(db)?;
    Ok(())
}

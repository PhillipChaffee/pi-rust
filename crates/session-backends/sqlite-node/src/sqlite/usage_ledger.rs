//! The `usage_ledger` rows, upstream's
//! `src/sqlite/session/usage-ledger.ts`.

use pi_agent_core::harness::session::types::{EntryScanOrder, SessionError, UsageRow, UsageScan};

use std::sync::Arc;

use crate::sql;
use crate::sqlite::sql::{SqlQuery, join_sql_fragments};
use crate::sqlite::types::{SqliteDatabase, SqliteRow, SqliteValue, sql_integer, sql_u64};

/// The `usage_ledger` table row, upstream's `UsageLedgerRow`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsageLedgerRow {
    /// The usage row id.
    pub id: String,
    /// The storage-assigned sequence.
    pub seq: i64,
    /// The entry the usage belongs to, when any.
    pub entry_id: Option<String>,
    /// The adjustment flag as 0/1, upstream's integer binding.
    pub adjustment: i64,
    /// The usage JSON.
    pub usage: String,
    /// The details JSON, when any.
    pub details: Option<String>,
}

const INSERT_USAGE_LEDGER_SQL: &str =
    "INSERT INTO usage_ledger (session_id, id, seq, entry_id, adjustment, usage, details)
	VALUES (?, ?, ?, ?, ?, ?, ?)";

fn usage_ledger_row_params(
    session_id: &str,
    row: &UsageRow,
) -> Result<Vec<SqliteValue>, SessionError> {
    Ok(vec![
        SqliteValue::from_param(session_id),
        SqliteValue::from_param(row.id.as_str()),
        SqliteValue::Integer(sql_integer(row.seq)?),
        SqliteValue::from_param(row.entry_id.clone()),
        SqliteValue::Integer(i64::from(row.adjustment)),
        SqliteValue::from_param(
            serde_json::to_string(&row.usage)
                .map_err(|error| SessionError::Message(error.to_string()))?,
        ),
        match &row.details {
            Some(details) => SqliteValue::from_param(
                serde_json::to_string(details)
                    .map_err(|error| SessionError::Message(error.to_string()))?,
            ),
            None => SqliteValue::Null,
        },
    ])
}

/// The once-prepared usage insert, upstream's `UsageLedgerRowWriter`.
///
/// The port prepares per execution through the connection's statement cache,
/// so the writer carries the handle and session id and clones into the
/// commit closure.
#[derive(Clone)]
pub struct UsageLedgerRowWriter {
    db: Arc<dyn SqliteDatabase>,
    session_id: String,
}

impl std::fmt::Debug for UsageLedgerRowWriter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("UsageLedgerRowWriter")
            .finish_non_exhaustive()
    }
}

impl UsageLedgerRowWriter {
    /// Prepares the writer over one session, upstream's constructor.
    #[must_use]
    pub fn new(db: Arc<dyn SqliteDatabase>, session_id: String) -> Self {
        Self { db, session_id }
    }

    /// Inserts one usage row, upstream's `insert`.
    ///
    /// # Errors
    /// A driver failure or an out-of-range integer.
    pub fn insert(&self, row: &UsageRow) -> Result<(), SessionError> {
        SqlQuery {
            query_text: INSERT_USAGE_LEDGER_SQL.to_owned(),
            params: usage_ledger_row_params(&self.session_id, row)?,
        }
        .run(self.db.as_ref())
        .map(|_| ())
    }
}

/// Inserts one usage row, preparing per call, upstream's
/// `insertUsageLedgerRow`.
///
/// # Errors
/// A driver failure or an out-of-range integer.
pub fn insert_usage_ledger_row(
    db: &dyn SqliteDatabase,
    session_id: &str,
    row: &UsageRow,
) -> Result<(), SessionError> {
    SqlQuery {
        query_text: INSERT_USAGE_LEDGER_SQL.to_owned(),
        params: usage_ledger_row_params(session_id, row)?,
    }
    .run(db)
    .map(|_| ())
}

fn usage_ledger_row(row: &SqliteRow) -> Result<UsageLedgerRow, SessionError> {
    Ok(UsageLedgerRow {
        id: row.string("id")?,
        seq: row.integer("seq")?,
        entry_id: row.opt_string("entry_id")?,
        adjustment: row.integer("adjustment")?,
        usage: row.string("usage")?,
        details: row.opt_string("details")?,
    })
}

fn parse_json<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, SessionError> {
    serde_json::from_str(text).map_err(|error| SessionError::Message(error.to_string()))
}

/// Decodes a ledger row into the contract shape, upstream's
/// `decodeUsageLedgerRow`.
///
/// # Errors
/// A JSON failure.
pub fn decode_usage_ledger_row(row: &UsageLedgerRow) -> Result<UsageRow, SessionError> {
    Ok(UsageRow {
        id: row.id.clone(),
        seq: sql_u64(row.seq)?,
        usage: parse_json(&row.usage)?,
        entry_id: row.entry_id.clone(),
        adjustment: row.adjustment != 0,
        details: row.details.as_deref().map(parse_json).transpose()?,
    })
}

/// Scans the ledger with sequence bounds, upstream's `scanUsageLedgerRows`.
///
/// # Errors
/// A driver failure or an out-of-range integer.
pub fn scan_usage_ledger_rows(
    db: &dyn SqliteDatabase,
    session_id: &str,
    query: &UsageScan,
) -> Result<Vec<UsageLedgerRow>, SessionError> {
    let mut filters = vec![sql!("session_id = ?", session_id)];
    if let Some(from_seq) = query.from_seq {
        filters.push(sql!("seq >= ?", sql_integer(from_seq)?));
    }
    if let Some(to_seq) = query.to_seq {
        filters.push(sql!("seq <= ?", sql_integer(to_seq)?));
    }
    let order = match query.order {
        Some(EntryScanOrder::Desc) => " ORDER BY seq DESC",
        _ => " ORDER BY seq ASC",
    };
    let limit = match query.limit {
        Some(limit) => sql!(" LIMIT ?", sql_integer(limit)?),
        None => SqlQuery::empty(),
    };
    let mut composed = sql!(
        "SELECT id, seq, entry_id, adjustment, usage, details
		FROM usage_ledger WHERE "
    );
    composed.push(join_sql_fragments(filters, " AND "));
    composed.push(SqlQuery::text(order));
    composed.push(limit);
    let rows = composed.all(db)?;
    rows.iter().map(usage_ledger_row).collect()
}

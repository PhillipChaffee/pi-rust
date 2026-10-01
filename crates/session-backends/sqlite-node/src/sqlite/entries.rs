//! The `entries` rows and payload projection, upstream's
//! `src/sqlite/session/entries.ts`.

use pi_agent_core::harness::session::types::{
    CustomEntryBody, Entry, EntryScan, EntryScanOrder, EntryStructure, EntryType, SessionError,
};

use std::sync::Arc;

use crate::sql;
use crate::sqlite::sql::{SqlQuery, join_sql_fragments};
use crate::sqlite::types::{SqliteDatabase, SqliteRow, SqliteValue, sql_integer, sql_u64};

/// The `entries` table row, upstream's `EntryRow`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryRow {
    /// The entry id.
    pub id: String,
    /// The parent entry id; `None` at a branch root.
    pub parent_id: Option<String>,
    /// The storage-assigned sequence.
    pub seq: u64,
    /// The entry type's wire name.
    pub kind: EntryType,
    /// The custom type, present only on custom entries.
    pub custom_type: Option<String>,
    /// The storage-assigned timestamp, epoch milliseconds.
    pub timestamp: i64,
    /// The payload JSON: the entry minus id/parentId/seq/timestamp/type/customType.
    pub payload: String,
}

/// The custom entry's stored payload, upstream's
/// `StoredEntryPayload<CustomEntry>` — `customType` rides the column, so the
/// payload carries only the data.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CustomEntryPayload {
    /// The application-defined payload, when any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

pub(crate) const fn wire_of(kind: EntryType) -> &'static str {
    match kind {
        EntryType::Message => "message",
        EntryType::Compaction => "compaction",
        EntryType::BranchSummary => "branch_summary",
        EntryType::Custom => "custom",
    }
}

pub(crate) fn kind_of(wire: &str) -> Result<EntryType, SessionError> {
    match wire {
        "message" => Ok(EntryType::Message),
        "compaction" => Ok(EntryType::Compaction),
        "branch_summary" => Ok(EntryType::BranchSummary),
        "custom" => Ok(EntryType::Custom),
        // Upstream's decode switch silently yields `undefined` for an unknown
        // type (a type lie); the port errors instead. Recorded choice.
        _ => Err(SessionError::Message(format!(
            "Unknown SQLite entry type: {wire}"
        ))),
    }
}

/// Projects the stored payload for one entry, upstream's `entryPayload`:
/// the entry minus id/parentId/seq/timestamp/type/customType.
///
/// # Errors
/// A JSON serialization failure.
pub fn entry_payload(entry: &Entry) -> Result<String, SessionError> {
    let text = match entry {
        Entry::Message { body, .. } => {
            serde_json::to_string(body.as_ref()).map_err(|error| json_error(&error))?
        }
        Entry::Compaction { body, .. } => {
            serde_json::to_string(body).map_err(|error| json_error(&error))?
        }
        Entry::BranchSummary { body, .. } => {
            serde_json::to_string(body).map_err(|error| json_error(&error))?
        }
        Entry::Custom { body, .. } => serde_json::to_string(&CustomEntryPayload {
            data: body.data.clone(),
        })
        .map_err(|error| json_error(&error))?,
    };
    Ok(text)
}

fn json_error(error: &serde_json::Error) -> SessionError {
    SessionError::Message(error.to_string())
}

const INSERT_ENTRY_SQL: &str =
    "INSERT INTO entries (session_id, id, parent_id, seq, type, custom_type, timestamp, payload)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?)";

fn entry_row_params(session_id: &str, entry: &Entry) -> Result<Vec<SqliteValue>, SessionError> {
    let (id, parent_id, seq, timestamp, kind, custom_type) = match entry {
        Entry::Message {
            id,
            parent_id,
            seq,
            timestamp,
            ..
        }
        | Entry::Compaction {
            id,
            parent_id,
            seq,
            timestamp,
            ..
        }
        | Entry::BranchSummary {
            id,
            parent_id,
            seq,
            timestamp,
            ..
        } => (id, parent_id, *seq, *timestamp, entry.entry_type(), None),
        Entry::Custom {
            id,
            parent_id,
            seq,
            timestamp,
            body,
            ..
        } => (
            id,
            parent_id,
            *seq,
            *timestamp,
            entry.entry_type(),
            Some(body.custom_type.clone()),
        ),
    };
    Ok(vec![
        SqliteValue::from_param(session_id),
        SqliteValue::from_param(id.as_str()),
        SqliteValue::from_param(parent_id.clone()),
        SqliteValue::Integer(sql_integer(seq)?),
        SqliteValue::from_param(wire_of(kind)),
        SqliteValue::from_param(custom_type),
        SqliteValue::Integer(timestamp),
        SqliteValue::from_param(entry_payload(entry)?),
    ])
}

/// The once-prepared entry insert, upstream's `EntryRowWriter`.
///
/// The port prepares per execution through the connection's statement cache,
/// so the writer carries the handle and session id and clones into the
/// commit closure.
#[derive(Clone)]
pub struct EntryRowWriter {
    db: Arc<dyn SqliteDatabase>,
    session_id: String,
}

impl std::fmt::Debug for EntryRowWriter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EntryRowWriter")
            .finish_non_exhaustive()
    }
}

impl EntryRowWriter {
    /// Prepares the writer over one session, upstream's constructor.
    #[must_use]
    pub fn new(db: Arc<dyn SqliteDatabase>, session_id: String) -> Self {
        Self { db, session_id }
    }

    /// Inserts one entry, upstream's `insert`.
    ///
    /// # Errors
    /// A driver failure, an out-of-range integer, or a JSON failure.
    pub fn insert(&self, entry: &Entry) -> Result<(), SessionError> {
        SqlQuery {
            query_text: INSERT_ENTRY_SQL.to_owned(),
            params: entry_row_params(&self.session_id, entry)?,
        }
        .run(self.db.as_ref())
        .map(|_| ())
    }
}

/// Inserts one entry row, preparing per call, upstream's `insertEntryRow`.
///
/// # Errors
/// A driver failure, an out-of-range integer, or a JSON failure.
pub fn insert_entry_row(
    db: &dyn SqliteDatabase,
    session_id: &str,
    entry: &Entry,
) -> Result<(), SessionError> {
    SqlQuery {
        query_text: INSERT_ENTRY_SQL.to_owned(),
        params: entry_row_params(session_id, entry)?,
    }
    .run(db)
    .map(|_| ())
}

fn parse_payload<T: serde::de::DeserializeOwned>(row: &EntryRow) -> Result<T, SessionError> {
    serde_json::from_str(&row.payload).map_err(|error| json_error(&error))
}

/// Decodes a row into the entry, upstream's `decodeEntryRow`.
///
/// # Errors
/// `Custom entry {id} is missing custom_type` when a custom row lost its
/// column; a JSON failure otherwise.
pub fn decode_entry_row(row: &EntryRow) -> Result<Entry, SessionError> {
    let base = (
        row.id.clone(),
        row.parent_id.clone(),
        row.seq,
        row.timestamp,
    );
    match row.kind {
        EntryType::Message => Ok(Entry::Message {
            id: base.0,
            parent_id: base.1,
            seq: base.2,
            timestamp: base.3,
            body: Box::new(parse_payload(row)?),
        }),
        EntryType::Compaction => Ok(Entry::Compaction {
            id: base.0,
            parent_id: base.1,
            seq: base.2,
            timestamp: base.3,
            body: parse_payload(row)?,
        }),
        EntryType::BranchSummary => Ok(Entry::BranchSummary {
            id: base.0,
            parent_id: base.1,
            seq: base.2,
            timestamp: base.3,
            body: parse_payload(row)?,
        }),
        EntryType::Custom => {
            let custom_type = row.custom_type.clone().ok_or_else(|| {
                SessionError::Message(format!("Custom entry {} is missing custom_type", row.id))
            })?;
            let payload: CustomEntryPayload = parse_payload(row)?;
            Ok(Entry::Custom {
                id: base.0,
                parent_id: base.1,
                seq: base.2,
                timestamp: base.3,
                body: CustomEntryBody {
                    custom_type,
                    data: payload.data,
                },
            })
        }
    }
}

/// Builds the structural view of one row, upstream's `entryStructureFromRow`.
///
/// # Errors
/// An out-of-range integer.
pub fn entry_structure_from_row(row: &EntryRow) -> Result<EntryStructure, SessionError> {
    Ok(EntryStructure {
        id: row.id.clone(),
        parent_id: row.parent_id.clone(),
        seq: row.seq,
        timestamp: row.timestamp,
        kind: row.kind,
        custom_type: row.custom_type.clone(),
    })
}

pub(crate) fn entry_row(row: &SqliteRow) -> Result<EntryRow, SessionError> {
    Ok(EntryRow {
        id: row.string("id")?,
        parent_id: row.opt_string("parent_id")?,
        seq: sql_u64(row.integer("seq")?)?,
        kind: kind_of(&row.string("type")?)?,
        custom_type: row.opt_string("custom_type")?,
        timestamp: row.integer("timestamp")?,
        payload: row.string("payload")?,
    })
}

/// Builds the structural view from a raw result row, the structure scan's
/// decode, upstream's `decodeEntryStructureRow`.
///
/// # Errors
/// An out-of-range integer or an unknown entry type.
pub(crate) fn entry_structure_row(row: &SqliteRow) -> Result<EntryStructure, SessionError> {
    Ok(EntryStructure {
        id: row.string("id")?,
        parent_id: row.opt_string("parent_id")?,
        seq: sql_u64(row.integer("seq")?)?,
        timestamp: row.integer("timestamp")?,
        kind: kind_of(&row.string("type")?)?,
        custom_type: row.opt_string("custom_type")?,
    })
}

/// Decodes a batch of rows, the shared read helper (upstream inlines
/// `.map(decodeEntryRow)` at each read site).
pub(crate) fn decode_entry_rows(rows: &[EntryRow]) -> Result<Vec<Entry>, SessionError> {
    rows.iter().map(decode_entry_row).collect()
}

/// Reads entries by id, upstream's `readEntryRows`; the returned order
/// follows the database's, not the request's.
///
/// # Errors
/// A driver or decode failure.
pub fn read_entry_rows(
    db: &dyn SqliteDatabase,
    session_id: &str,
    ids: &[String],
) -> Result<Vec<EntryRow>, SessionError> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = join_sql_fragments(
        ids.iter().map(|id| SqlQuery::param(id.as_str())).collect(),
        ", ",
    );
    let mut query = sql!(
        "SELECT id, parent_id, seq, type, custom_type, timestamp, payload
		FROM entries
		WHERE session_id = ? AND id IN (",
        session_id
    );
    query.push(placeholders);
    query.push(SqlQuery::text(")"));
    let rows = query.all(db)?;
    rows.iter().map(entry_row).collect()
}

/// Reads every entry in sequence order, upstream's `readAllEntryRows`.
///
/// # Errors
/// A driver or decode failure.
pub fn read_all_entry_rows(
    db: &dyn SqliteDatabase,
    session_id: &str,
) -> Result<Vec<EntryRow>, SessionError> {
    let rows = sql!(
        "SELECT id, parent_id, seq, type, custom_type, timestamp, payload
		FROM entries WHERE session_id = ? ORDER BY seq ASC",
        session_id
    )
    .all(db)?;
    rows.iter().map(entry_row).collect()
}

/// Scans entries with filters and sequence bounds, upstream's
/// `scanEntryRows`.
///
/// # Errors
/// A driver failure or an out-of-range integer.
pub fn scan_entry_rows(
    db: &dyn SqliteDatabase,
    session_id: &str,
    query: &EntryScan,
) -> Result<Vec<EntryRow>, SessionError> {
    let mut filters = vec![sql!("session_id = ?", session_id)];
    if let Some(kind) = query.kind {
        filters.push(sql!("type = ?", wire_of(kind)));
    }
    if let Some(custom_type) = &query.custom_type {
        filters.push(sql!("custom_type = ?", custom_type.as_str()));
    }
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
        "SELECT id, parent_id, seq, type, custom_type, timestamp, payload
		FROM entries WHERE "
    );
    composed.push(join_sql_fragments(filters, " AND "));
    composed.push(SqlQuery::text(order));
    composed.push(limit);
    let rows = composed.all(db)?;
    rows.iter().map(entry_row).collect()
}

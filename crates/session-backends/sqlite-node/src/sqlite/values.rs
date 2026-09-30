//! The `scalar_values` and `list_values` rows, upstream's
//! `src/sqlite/session/values.ts`.

use pi_agent_core::harness::session::types::{EntryScanOrder, SessionError};
use pi_agent_core::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress, resolve_list_read_options,
};

use crate::sql;
use crate::sqlite::sql::SqlQuery;
use crate::sqlite::types::{SqliteDatabase, SqliteRow, sql_integer, sql_u64};

/// The `scalar_values` table row, upstream's `ScalarValueRow`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarValueRow {
    /// The addressed namespace.
    pub namespace: String,
    /// The addressed key.
    pub key: String,
    /// The sequence the value was written at.
    pub seq: i64,
    /// The value JSON.
    pub value: String,
}

/// The `list_values` element row, upstream's `ListValueRow`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListValueRow {
    /// The element's sequence.
    pub seq: i64,
    /// The element value JSON.
    pub value: String,
}

/// Upserts one scalar value, upstream's `setScalarValueRow`.
///
/// # Errors
/// A driver failure or an out-of-range integer.
pub fn set_scalar_value_row(
    db: &dyn SqliteDatabase,
    session_id: &str,
    namespace: &str,
    key: &str,
    seq: u64,
    stored_value: &serde_json::Value,
) -> Result<(), SessionError> {
    sql!(
        "INSERT INTO scalar_values (session_id, namespace, key, seq, value)
	VALUES (?, ?, ?, ?, ?)
	ON CONFLICT(session_id, namespace, key) DO UPDATE SET seq = excluded.seq, value = excluded.value",
        session_id,
        namespace,
        key,
        sql_integer(seq)?,
        serde_json::to_string(stored_value)
            .map_err(|error| SessionError::Message(error.to_string()))?,
    )
    .run(db)
    .map(|_| ())?;
    Ok(())
}

/// Deletes one scalar value, upstream's `deleteScalarValueRow`.
///
/// # Errors
/// A driver failure.
pub fn delete_scalar_value_row(
    db: &dyn SqliteDatabase,
    session_id: &str,
    namespace: &str,
    key: &str,
) -> Result<(), SessionError> {
    sql!(
        "DELETE FROM scalar_values WHERE session_id = ? AND namespace = ? AND key = ?",
        session_id,
        namespace,
        key
    )
    .run(db)
    .map(|_| ())?;
    Ok(())
}

/// Appends one list element, upstream's `appendListValueRow`.
///
/// # Errors
/// A driver failure or an out-of-range integer.
pub fn append_list_value_row(
    db: &dyn SqliteDatabase,
    session_id: &str,
    namespace: &str,
    key: &str,
    seq: u64,
    element: &serde_json::Value,
) -> Result<(), SessionError> {
    sql!(
        "INSERT INTO list_values (session_id, namespace, key, seq, value) VALUES (?, ?, ?, ?, ?)",
        session_id,
        namespace,
        key,
        sql_integer(seq)?,
        serde_json::to_string(element).map_err(|error| SessionError::Message(error.to_string()))?,
    )
    .run(db)
    .map(|_| ())?;
    Ok(())
}

/// Deletes every element of one list, upstream's `deleteListValueRows`.
///
/// # Errors
/// A driver failure.
pub fn delete_list_value_rows(
    db: &dyn SqliteDatabase,
    session_id: &str,
    namespace: &str,
    key: &str,
) -> Result<(), SessionError> {
    sql!(
        "DELETE FROM list_values WHERE session_id = ? AND namespace = ? AND key = ?",
        session_id,
        namespace,
        key
    )
    .run(db)
    .map(|_| ())?;
    Ok(())
}

fn decode_scalar_value_row(
    address: &ValueAddress,
    row: &SqliteRow,
) -> Result<StoredValue, SessionError> {
    let namespace = row.string("namespace")?;
    let key = row.string("key")?;
    if namespace != address.namespace || key != address.key {
        return Err(SessionError::Message(format!(
            "Expected value {}:{}, found {}:{}",
            address.namespace, address.key, namespace, key
        )));
    }
    let value = row.string("value")?;
    Ok(StoredValue {
        namespace,
        key,
        seq: sql_u64(row.integer("seq")?)?,
        value: serde_json::from_str(&value)
            .map_err(|error| SessionError::Message(error.to_string()))?,
    })
}

/// Reads one scalar value by its bound address, upstream's
/// `readScalarValueRow`.
///
/// # Errors
/// `Expected value {namespace}:{key}, found {row_namespace}:{row_key}` on an
/// address mismatch; a driver or JSON failure otherwise.
pub fn read_scalar_value_row(
    db: &dyn SqliteDatabase,
    session_id: &str,
    address: &ValueAddress,
) -> Result<Option<StoredValue>, SessionError> {
    let row = sql!(
        "SELECT namespace, key, seq, value FROM scalar_values
	WHERE session_id = ? AND namespace = ? AND key = ?",
        session_id,
        &address.namespace,
        &address.key
    )
    .get(db)?;
    row.map(|row| decode_scalar_value_row(address, &row))
        .transpose()
}

/// Reads every scalar value in sequence order, upstream's
/// `readAllScalarValueRows`.
///
/// # Errors
/// A driver or JSON failure.
pub fn read_all_scalar_value_rows(
    db: &dyn SqliteDatabase,
    session_id: &str,
) -> Result<Vec<StoredValue>, SessionError> {
    let rows = sql!(
        "SELECT namespace, key, seq, value FROM scalar_values
	WHERE session_id = ? ORDER BY seq ASC",
        session_id
    )
    .all(db)?;
    rows.iter()
        .map(|row| {
            let value = row.string("value")?;
            Ok(StoredValue {
                namespace: row.string("namespace")?,
                key: row.string("key")?,
                seq: sql_u64(row.integer("seq")?)?,
                value: serde_json::from_str(&value)
                    .map_err(|error| SessionError::Message(error.to_string()))?,
            })
        })
        .collect()
}

/// The exclusive upper bound a prefix scan stops below, upstream's
/// `nextPrefixBoundary`: walk code points from the end and increment the
/// first one that is not `U+10FFFF`, jumping the surrogate window
/// (`U+D7FF..U+E000`) straight to `U+E000`; an empty or all-max prefix is
/// open-ended. Upstream's defensive `Invalid value key prefix` throw is
/// unreachable here — a `String` cannot hold lone surrogates.
fn next_prefix_boundary(prefix: &str) -> Option<String> {
    if prefix.is_empty() {
        return None;
    }
    let code_points: Vec<char> = prefix.chars().collect();
    for index in (0..code_points.len()).rev() {
        let code_point = code_points[index] as u32;
        if code_point < 0x10_ffff {
            let next_code_point = if (0xd7ff..0xe000).contains(&code_point) {
                0xe000
            } else {
                code_point + 1
            };
            let mut boundary = code_points[..index].iter().collect::<String>();
            // The arithmetic above cannot produce a surrogate or exceed
            // U+10FFFF, so the char conversion always succeeds.
            boundary.push(char::from_u32(next_code_point)?);
            return Some(boundary);
        }
    }
    None
}

/// Scans scalar values under one namespace and key prefix, upstream's
/// `scanScalarValueRows`.
///
/// # Errors
/// A driver or JSON failure.
pub fn scan_scalar_value_rows(
    db: &dyn SqliteDatabase,
    session_id: &str,
    prefix: &ValueAddress,
) -> Result<Vec<StoredValue>, SessionError> {
    let upper_bound = next_prefix_boundary(&prefix.key);
    let rows = match upper_bound {
        None => sql!(
            "SELECT namespace, key, seq, value FROM scalar_values
		WHERE session_id = ? AND namespace = ? AND key >= ?
		ORDER BY key ASC",
            session_id,
            &prefix.namespace,
            &prefix.key
        )
        .all(db)?,
        Some(upper_bound) => sql!(
            "SELECT namespace, key, seq, value FROM scalar_values
		WHERE session_id = ? AND namespace = ? AND key >= ? AND key < ?
		ORDER BY key ASC",
            session_id,
            &prefix.namespace,
            &prefix.key,
            upper_bound
        )
        .all(db)?,
    };
    rows.iter()
        .map(|row| {
            let address = ValueAddress {
                namespace: row.string("namespace")?,
                key: row.string("key")?,
            };
            decode_scalar_value_row(&address, row)
        })
        .collect()
}

/// Builds the list-read query, upstream's `listValueReadQuery`: the four
/// order/cursor variants over the list primary key, with the resolved
/// options' limit.
///
/// # Errors
/// The limit error from [`resolve_list_read_options`]; an out-of-range
/// integer otherwise.
pub fn list_value_read_query(
    session_id: &str,
    address: &ListAddress,
    options: Option<ListReadOptions>,
) -> Result<SqlQuery, SessionError> {
    let options = resolve_list_read_options(options)?;
    let limit = sql_integer(options.limit)?;
    Ok(match (&options.cursor, options.order) {
        (Some(cursor), EntryScanOrder::Asc) => sql!(
            "SELECT seq, value FROM list_values
		WHERE session_id = ? AND namespace = ? AND key = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
            session_id,
            &address.namespace,
            &address.key,
            sql_integer(cursor.seq)?,
            limit
        ),
        (Some(cursor), EntryScanOrder::Desc) => sql!(
            "SELECT seq, value FROM list_values
		WHERE session_id = ? AND namespace = ? AND key = ? AND seq < ? ORDER BY seq DESC LIMIT ?",
            session_id,
            &address.namespace,
            &address.key,
            sql_integer(cursor.seq)?,
            limit
        ),
        (None, EntryScanOrder::Asc) => sql!(
            "SELECT seq, value FROM list_values
		WHERE session_id = ? AND namespace = ? AND key = ? ORDER BY seq ASC LIMIT ?",
            session_id,
            &address.namespace,
            &address.key,
            limit
        ),
        (None, EntryScanOrder::Desc) => sql!(
            "SELECT seq, value FROM list_values
		WHERE session_id = ? AND namespace = ? AND key = ? ORDER BY seq DESC LIMIT ?",
            session_id,
            &address.namespace,
            &address.key,
            limit
        ),
    })
}

/// Reads list elements, upstream's `readListValueRows`.
///
/// # Errors
/// The query builder's errors; a driver or JSON failure otherwise.
pub fn read_list_value_rows(
    db: &dyn SqliteDatabase,
    session_id: &str,
    address: &ListAddress,
    options: Option<ListReadOptions>,
) -> Result<Vec<ListElement>, SessionError> {
    let query = list_value_read_query(session_id, address, options)?;
    let rows = query.all(db)?;
    rows.iter()
        .map(|row| {
            let value = row.string("value")?;
            Ok(ListElement {
                seq: sql_u64(row.integer("seq")?)?,
                value: serde_json::from_str(&value)
                    .map_err(|error| SessionError::Message(error.to_string()))?,
            })
        })
        .collect()
}

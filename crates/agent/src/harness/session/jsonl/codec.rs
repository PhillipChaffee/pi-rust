//! The JSONL session-header codec, ported from upstream
//! `src/harness/session/jsonl/codec.ts`: the format-4 and legacy-v3 header
//! shapes and the line parser that discriminates them.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::harness::session::jsonl::types::{JSONL_FORMAT_VERSION, JsonlStorageHeader};

/// The legacy v3 file's first line, upstream's `LegacyV3SessionHeader`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LegacyV3SessionHeader {
    /// The record type, wire `"type": "session"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The format version, wire `"version": 3`.
    pub version: u32,
    /// The session id.
    pub id: String,
    /// The creation time as an ISO-8601 string, wire `"timestamp"`.
    pub timestamp: String,
    /// The session's working directory.
    pub cwd: String,
    /// The parent session's file path, when forked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
}

/// The header format one first line parses to, upstream's
/// `JsonlParsedSessionHeader`.
#[derive(Clone, Debug)]
pub enum JsonlParsedSessionHeader {
    /// A format-4 storage header.
    V4(JsonlStorageHeader),
    /// A legacy v3 session header.
    V3Legacy(LegacyV3SessionHeader),
}

fn is_record(value: &JsonValue) -> bool {
    value.is_object()
}

fn is_safe_integer_at_least(value: &JsonValue, minimum: i64) -> bool {
    value
        .as_i64()
        .is_some_and(|int| int >= minimum && int <= 9_007_199_254_740_991)
}

/// Whether the value reads as a legacy v3 session header, upstream's
/// `isLegacyV3SessionHeader`.
#[must_use]
pub fn is_legacy_v3_session_header(value: &JsonValue) -> bool {
    let Some(record) = value.as_object() else {
        return false;
    };
    if !is_record(value)
        || record.get("type").and_then(JsonValue::as_str) != Some("session")
        || record.get("version").and_then(JsonValue::as_u64) != Some(3)
        || !record.get("id").is_some_and(JsonValue::is_string)
        || !record.get("cwd").is_some_and(JsonValue::is_string)
        || !record.get("timestamp").is_some_and(|timestamp| {
            parse_iso8601_millis(timestamp.as_str().unwrap_or_default()).is_some()
        })
    {
        return false;
    }
    record.get("parentSession").is_none_or(JsonValue::is_string)
}

/// Whether the value reads as a format-4 storage header, upstream's
/// `isJsonlStorageHeader`.
#[must_use]
pub fn is_jsonl_storage_header(value: &JsonValue) -> bool {
    let Some(record) = value.as_object() else {
        return false;
    };
    if !is_record(value)
        || record.get("kind").and_then(JsonValue::as_str) != Some("header")
        || record.get("v").and_then(JsonValue::as_u64) != Some(u64::from(JSONL_FORMAT_VERSION))
        || !record.get("id").is_some_and(JsonValue::is_string)
        || !record.get("cwd").is_some_and(JsonValue::is_string)
        || !is_safe_integer_at_least(record.get("storageVersion").unwrap_or(&JsonValue::Null), 1)
        || !is_safe_integer_at_least(record.get("createdAt").unwrap_or(&JsonValue::Null), 0)
        || record
            .get("nextSeq")
            .is_some_and(|next_seq| !is_safe_integer_at_least(next_seq, 1))
        || !record
            .get("parentSessionId")
            .is_none_or(JsonValue::is_string)
        || !record
            .get("legacyParentSessionPath")
            .is_none_or(JsonValue::is_string)
    {
        return false;
    }
    true
}

/// Parses one first line, upstream's `parseJsonlSessionHeader`.
///
/// # Errors
/// The failure messages upstream throws: not-valid-JSON and unsupported
/// header.
pub fn parse_jsonl_session_header(
    line: &str,
) -> Result<JsonlParsedSessionHeader, SessionHeaderError> {
    let value: JsonValue = match serde_json::from_str(line) {
        Ok(value) => value,
        Err(_) => {
            return Err(SessionHeaderError(
                "Invalid JSONL session header: not valid JSON".to_owned(),
            ));
        }
    };
    if is_jsonl_storage_header(&value) {
        let header = serde_json::from_value(value)
            .map_err(|_| SessionHeaderError("Unsupported JSONL session header".to_owned()))?;
        return Ok(JsonlParsedSessionHeader::V4(header));
    }
    if is_legacy_v3_session_header(&value) {
        let header = serde_json::from_value(value)
            .map_err(|_| SessionHeaderError("Unsupported JSONL session header".to_owned()))?;
        return Ok(JsonlParsedSessionHeader::V3Legacy(header));
    }
    Err(SessionHeaderError(
        "Unsupported JSONL session header".to_owned(),
    ))
}

/// The header parse failure, upstream's thrown `Error`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionHeaderError(pub String);

impl std::fmt::Display for SessionHeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SessionHeaderError {}

/// Parses an ISO-8601 timestamp to Unix epoch milliseconds, the
/// `Date.parse` reads the v3 header's `timestamp` field rides.
///
/// Accepts the `YYYY-MM-DDTHH:MM:SS(.sss)?Z` forms the fixtures and
/// `toISOString` produce; returns `None` when the string does not parse.
#[must_use]
pub fn parse_iso8601_millis(timestamp: &str) -> Option<i64> {
    let timestamp = timestamp.strip_suffix('Z').unwrap_or(timestamp);
    let (date, time) = timestamp.split_once('T')?;
    let mut date_parts = date.split('-');
    let year: i64 = date_parts.next()?.parse().ok()?;
    let month: i64 = date_parts.next()?.parse().ok()?;
    let day: i64 = date_parts.next()?.parse().ok()?;
    if date_parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let mut time_parts = time.split(':');
    let hour: i64 = time_parts.next()?.parse().ok()?;
    let minute: i64 = time_parts.next()?.parse().ok()?;
    let seconds_and_fraction = time_parts.next()?;
    let (second_part, millis) = match seconds_and_fraction.split_once('.') {
        Some((seconds, fraction)) => {
            let padded = format!("{fraction}000");
            let millis: i64 = padded.get(0..3)?.parse().ok()?;
            (seconds.parse::<i64>().ok()?, millis)
        }
        None => (seconds_and_fraction.parse::<i64>().ok()?, 0),
    };
    if time_parts.next().is_some()
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
        || !(0..=59).contains(&second_part)
    {
        return None;
    }
    Some(
        days_from_civil(year, month, day) * 86_400_000
            + hour * 3_600_000
            + minute * 60_000
            + second_part * 1_000
            + millis,
    )
}

/// The days-from-civil algorithm (Howard Hinnant) mapping a calendar date
/// to days since 1970-01-01, the ISO timestamp's date half.
const fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month_shift = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * month_shift + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}
/// Formats Unix epoch milliseconds as the ISO-8601 `Z` string
/// `toISOString` produces, the session filename's timestamp half.
#[must_use]
pub fn format_iso8601(millis: i64) -> String {
    let days = millis.div_euclid(86_400_000);
    let within_day = millis.rem_euclid(86_400_000);
    let (year, month, day) = civil_from_days(days);
    let hour = within_day / 3_600_000;
    let minute = (within_day % 3_600_000) / 60_000;
    let second = (within_day % 60_000) / 1_000;
    let fraction = within_day % 1_000;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{fraction:03}Z")
}

/// The civil-from-days algorithm (Howard Hinnant) mapping days since
/// 1970-01-01 to a calendar date.
const fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let days = days + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let day_of_era = days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_shift = (5 * day_of_year + 2) / 153;
    let month = if month_shift < 10 {
        month_shift + 3
    } else {
        month_shift - 9
    };
    (
        if month <= 2 { year + 1 } else { year },
        month,
        day_of_year - (153 * month_shift + 2) / 5 + 1,
    )
}

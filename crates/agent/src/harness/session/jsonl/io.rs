//! The JSONL publication and transaction codec, ported from upstream
//! `src/harness/session/jsonl/io.ts`.
//!
//! Atomic publication stages content into a `.tmp` sibling and renames it
//! over the destination; a failure discards the staging and rethrows. The
//! transaction codec persists committed writes in upstream's flat wire
//! shape: the entry and usage families carry their materialized record's
//! fields beside `kind`, the value and list families an `op`.

use std::future::Future;
use std::sync::Arc;

use pi_ai::types::BoxedFuture;
use serde_json::{Value as JsonValue, json};

use crate::harness::context::Context;
use crate::harness::session::commit::CommittedWrite;
use crate::harness::session::jsonl::codec::{JsonlParsedSessionHeader, parse_jsonl_session_header};
use crate::harness::session::jsonl::types::JsonlStorageHeader;
use crate::harness::types::{FileContent, FileError, FileSystem, RemoveOptions, TextLineReader};

/// The JSONL layer's thrown error, upstream's `new Error(message, { cause })`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonlError(pub String);

impl std::fmt::Display for JsonlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for JsonlError {}

impl From<JsonlError> for crate::harness::session::types::SessionError {
    fn from(error: JsonlError) -> Self {
        Self::Message(error.0)
    }
}

/// Unwraps one filesystem result into the thrown-error restatement,
/// upstream's `fileValue`: the action prefixes the failure message.
///
/// # Errors
/// A [`JsonlError`] carrying `{action}: {message}`.
pub fn file_value<T>(result: Result<T, FileError>, action: &str) -> Result<T, JsonlError> {
    result.map_err(|error| JsonlError(format!("{action}: {}", error.message)))
}

/// Reads and parses the file's first line, upstream's `readJsonlHeader`.
///
/// # Errors
/// The read failure prefixed with the action, a missing, unterminated, or
/// empty header line, or an unsupported header.
pub async fn read_jsonl_header(
    reader: &mut dyn TextLineReader,
    path: &str,
    context: &Context,
) -> Result<JsonlParsedSessionHeader, JsonlError> {
    let line = file_value(
        reader.read_line(context).await,
        &format!("Failed to read JSONL storage {path}"),
    )?;
    let Some(line) = line else {
        return Err(JsonlError(format!(
            "Invalid JSONL storage {path}: missing header"
        )));
    };
    if !line.terminated || line.text.is_empty() {
        return Err(JsonlError(format!(
            "Invalid JSONL storage {path}: missing header"
        )));
    }
    parse_jsonl_session_header(&line.text).map_err(|error| {
        JsonlError(format!(
            "Invalid JSONL storage {path}: invalid header ({error})"
        ))
    })
}

/// Validates and decodes one committed write from its wire object,
/// upstream's `parseCommittedWrite`.
///
/// # Errors
/// A [`JsonlError`] for non-object writes, non-safe-integer sequences and
/// timestamps, unknown kinds, or unknown operations.
pub fn parse_committed_write(value: &JsonValue) -> Result<CommittedWrite, JsonlError> {
    let Some(record) = value.as_object() else {
        return Err(JsonlError("Invalid JSONL transaction write".to_owned()));
    };
    let seq = record
        .get("seq")
        .and_then(JsonValue::as_u64)
        .filter(|seq| (1..=9_007_199_254_740_991).contains(seq))
        .ok_or_else(|| JsonlError("Invalid JSONL write seq".to_owned()))?;
    match record.get("kind").and_then(JsonValue::as_str) {
        Some("entry") => {
            if record
                .get("timestamp")
                .and_then(JsonValue::as_i64)
                .is_none_or(|timestamp| timestamp < 0)
            {
                return Err(JsonlError("Invalid JSONL entry timestamp".to_owned()));
            }
            let entry: crate::harness::session::types::Entry =
                serde_json::from_value(value.clone())
                    .map_err(|_| JsonlError("Invalid JSONL transaction write".to_owned()))?;
            Ok(CommittedWrite::Entry { entry })
        }
        Some("usage") => {
            let row: crate::harness::session::types::UsageRow =
                serde_json::from_value(value.clone())
                    .map_err(|_| JsonlError("Invalid JSONL transaction write".to_owned()))?;
            Ok(CommittedWrite::Usage { row })
        }
        Some("value") => match record.get("op").and_then(JsonValue::as_str) {
            Some("set") => Ok(CommittedWrite::ValueSet {
                seq,
                namespace: string_field(record, "namespace")?,
                key: string_field(record, "key")?,
                value: record.get("value").cloned().unwrap_or(JsonValue::Null),
            }),
            Some("delete") => Ok(CommittedWrite::ValueDelete {
                seq,
                namespace: string_field(record, "namespace")?,
                key: string_field(record, "key")?,
            }),
            other => Err(JsonlError(format!(
                "Invalid JSONL value operation: {}",
                other.unwrap_or("undefined")
            ))),
        },
        Some("list") => match record.get("op").and_then(JsonValue::as_str) {
            Some("append") => Ok(CommittedWrite::ListAppend {
                seq,
                namespace: string_field(record, "namespace")?,
                key: string_field(record, "key")?,
                value: record.get("value").cloned().unwrap_or(JsonValue::Null),
            }),
            Some("delete") => Ok(CommittedWrite::ListDelete {
                seq,
                namespace: string_field(record, "namespace")?,
                key: string_field(record, "key")?,
            }),
            other => Err(JsonlError(format!(
                "Invalid JSONL list operation: {}",
                other.unwrap_or("undefined")
            ))),
        },
        other => Err(JsonlError(format!(
            "Invalid JSONL write kind: {}",
            other.unwrap_or("undefined")
        ))),
    }
}

fn string_field(
    record: &serde_json::Map<String, JsonValue>,
    field: &str,
) -> Result<String, JsonlError> {
    record
        .get(field)
        .and_then(JsonValue::as_str)
        .map(str::to_owned)
        .ok_or_else(|| JsonlError("Invalid JSONL transaction write".to_owned()))
}

/// Parses one transaction line, upstream's `parseJsonlTransaction`: a JSON
/// array is a multi-write transaction, any other JSON value is a
/// single-write transaction.
///
/// # Errors
/// A [`JsonlError`] for invalid JSON and per-write validation failures.
pub fn parse_jsonl_transaction(line: &str) -> Result<Vec<CommittedWrite>, JsonlError> {
    let value: JsonValue = serde_json::from_str(line)
        .map_err(|_| JsonlError("Invalid JSONL transaction: not valid JSON".to_owned()))?;
    match value {
        JsonValue::Array(writes) => writes.iter().map(parse_committed_write).collect(),
        value => Ok(vec![parse_committed_write(&value)?]),
    }
}

/// The wire object one committed write serializes to, upstream's
/// `JSON.stringify` of the spread committed write.
///
/// # Panics
/// Never: the wire shapes are fixed object constructions over
/// serde-serializable records.
#[must_use]
#[expect(
    clippy::expect_used,
    reason = "Value serialization of these closed wire shapes cannot fail: serde_json only errors for non-self-describing formats"
)]
pub fn committed_write_wire(write: &CommittedWrite) -> JsonValue {
    match write {
        CommittedWrite::Entry { entry } => {
            let mut wire = serde_json::to_value(entry).expect("entry wire");
            wire["kind"] = json!("entry");
            wire
        }
        CommittedWrite::Usage { row } => {
            let mut wire = serde_json::to_value(row).expect("usage row wire");
            wire["kind"] = json!("usage");
            wire
        }
        CommittedWrite::ValueSet {
            seq,
            namespace,
            key,
            value,
        } => json!({
            "kind": "value",
            "op": "set",
            "seq": seq,
            "namespace": namespace,
            "key": key,
            "value": value,
        }),
        CommittedWrite::ValueDelete {
            seq,
            namespace,
            key,
        } => json!({
            "kind": "value",
            "op": "delete",
            "seq": seq,
            "namespace": namespace,
            "key": key,
        }),
        CommittedWrite::ListAppend {
            seq,
            namespace,
            key,
            value,
        } => json!({
            "kind": "list",
            "op": "append",
            "seq": seq,
            "namespace": namespace,
            "key": key,
            "value": value,
        }),
        CommittedWrite::ListDelete {
            seq,
            namespace,
            key,
        } => json!({
            "kind": "list",
            "op": "delete",
            "seq": seq,
            "namespace": namespace,
            "key": key,
        }),
    }
}

/// Serializes one transaction line, upstream's `serializeJsonlTransaction`:
/// one write serializes as a bare object, several as an array.
#[must_use]
pub fn serialize_jsonl_transaction(writes: &[CommittedWrite]) -> String {
    let wire: Vec<JsonValue> = writes.iter().map(committed_write_wire).collect();
    match wire.as_slice() {
        [only] => only.to_string(),
        _ => JsonValue::Array(wire).to_string(),
    }
}

/// The raw-content appender [`publish_file_atomically`] hands out,
/// upstream's `append` closure.
pub type AtomicAppend =
    Arc<dyn Fn(String) -> BoxedFuture<'static, Result<(), JsonlError>> + Send + Sync>;

/// The transaction-line appender [`publish_jsonl`] hands out, upstream's
/// `writeTransactions((writes) => append(...))`.
pub type TransactionAppender =
    Arc<dyn Fn(&[CommittedWrite]) -> BoxedFuture<'static, Result<(), JsonlError>> + Send + Sync>;

/// Publish only after the callback succeeds, upstream's
/// `publishFileAtomically`.
///
/// Content stages into a `.tmp` sibling and renames over the destination; a
/// failure discards the staging and rethrows. The callback must await each
/// append before returning, upstream's documented contract.
///
/// # Errors
/// A [`JsonlError`] from the staging, append, or rename failures, or the
/// callback's own.
pub async fn publish_file_atomically<F, Fut>(
    file_system: &Arc<dyn FileSystem>,
    destination_path: &str,
    context: &Context,
    write_content: F,
) -> Result<(), JsonlError>
where
    F: FnOnce(AtomicAppend) -> Fut,
    Fut: Future<Output = Result<(), JsonlError>>,
{
    let temp_path = format!("{destination_path}.tmp");
    let result = publish_body(
        file_system,
        destination_path,
        &temp_path,
        context,
        write_content,
    )
    .await;
    if result.is_err() {
        // Upstream's catch awaits the remove unguarded; the port swallows a
        // staging-cleanup failure so the caller's error always surfaces.
        let _ = file_system
            .remove(
                &temp_path,
                Some(RemoveOptions {
                    force: Some(true),
                    ..Default::default()
                }),
                context,
            )
            .await;
    }
    result
}

async fn publish_body<F, Fut>(
    file_system: &Arc<dyn FileSystem>,
    destination_path: &str,
    temp_path: &str,
    context: &Context,
    write_content: F,
) -> Result<(), JsonlError>
where
    F: FnOnce(AtomicAppend) -> Fut,
    Fut: Future<Output = Result<(), JsonlError>>,
{
    file_value(
        file_system
            .write_file(temp_path, FileContent::Text(String::new()), context)
            .await,
        &format!("Failed to stage JSONL storage {destination_path}"),
    )?;
    let append: AtomicAppend = {
        let file_system = file_system.clone();
        let temp_path = temp_path.to_owned();
        let destination = destination_path.to_owned();
        let context = context.clone();
        Arc::new(move |content: String| {
            let file_system = file_system.clone();
            let temp_path = temp_path.clone();
            let destination = destination.clone();
            let context = context.clone();
            Box::pin(async move {
                file_value(
                    file_system
                        .append_file(&temp_path, FileContent::Text(content), &context)
                        .await,
                    &format!("Failed to append JSONL storage {destination}"),
                )
            })
        })
    };
    write_content(append).await?;
    file_value(
        file_system
            .rename_file(temp_path, destination_path, context)
            .await,
        &format!("Failed to publish JSONL storage {destination_path}"),
    )
}

/// Stream a header and complete transactions through the shared atomic
/// publisher, upstream's `publishJsonl`.
///
/// # Panics
/// Never: the header is a closed serde-serializable shape whose `Value`
/// serialization cannot fail.
///
/// # Errors
/// The [`publish_file_atomically`] failures.
pub async fn publish_jsonl<F, Fut>(
    file_system: &Arc<dyn FileSystem>,
    destination_path: &str,
    header: &JsonlStorageHeader,
    context: &Context,
    write_transactions: F,
) -> Result<(), JsonlError>
where
    F: FnOnce(TransactionAppender) -> Fut,
    Fut: Future<Output = Result<(), JsonlError>>,
{
    // The header is a closed serde-serializable shape; Value serialization
    // cannot fail for self-describing formats.
    #[expect(
        clippy::expect_used,
        reason = "the header's Value serialization cannot fail"
    )]
    let header_line = format!("{}\n", serde_json::to_string(header).expect("header wire"));
    publish_file_atomically(
        file_system,
        destination_path,
        context,
        |append| async move {
            append(header_line).await?;
            let transaction_appender: TransactionAppender = Arc::new({
                let append = append.clone();
                move |writes: &[CommittedWrite]| {
                    let line = format!("{}\n", serialize_jsonl_transaction(writes));
                    append(line)
                }
            });
            write_transactions(transaction_appender).await
        },
    )
    .await
}

/// The torn-tail helper the storage's open path shares with its tests,
/// upstream's `splitCompleteLines`.
#[must_use]
pub fn split_complete_lines(content: &str) -> (Vec<String>, bool) {
    if let Some(complete) = content.strip_suffix('\n') {
        return (complete.split('\n').map(str::to_owned).collect(), false);
    }
    content.rfind('\n').map_or_else(
        || (Vec::new(), true),
        |last_newline| {
            (
                content[..last_newline]
                    .split('\n')
                    .map(str::to_owned)
                    .collect(),
                true,
            )
        },
    )
}

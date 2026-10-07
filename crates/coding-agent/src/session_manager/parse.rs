//! The JSONL parse and the bounded header scan, upstream's
//! `loadEntriesFromFile`, `parseSessionEntryLine`, `parseSessionHeaderCandidate`,
//! and `readSessionHeader`.
//!
//! Upstream pin: `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::fs;
use std::io::{Read as _, Write as _};
use std::path::Path;

use serde_json::Value as JsonValue;

use super::entries::{MessageEntry, SessionEntry};
use super::file_entry::{FileEntry, ReadSessionHeaderError};
use super::types::{MAX_SESSION_HEADER_SCAN_BYTES, SESSION_HEADER_READ_BUFFER_SIZE, SessionHeader};
use crate::utils::paths::{PathInputOptions, normalize_path};

/// Repair a standard-role message whose content is null or missing to an
/// empty block list, upstream's `sessionEntryToContextMessages` repair
/// applied at parse time: the typed message layer cannot represent null
/// content, and the repair keeps such hand-edited messages in the context
/// instead of dropping them.
fn repair_null_content_message(entry: &mut JsonValue) {
    let Some(message) = entry.get_mut("message") else {
        return;
    };
    let role = message.get("role").and_then(JsonValue::as_str);
    if !matches!(role, Some("user" | "assistant" | "toolResult")) {
        return;
    }
    if message.get("content").is_none_or(JsonValue::is_null)
        && let Some(message) = message.as_object_mut()
    {
        message.insert("content".to_owned(), JsonValue::Array(Vec::new()));
    }
}

/// Parse one line's JSON into a file entry; the caller skips `None` (blank
/// or malformed lines), upstream's `parseSessionEntryLine` contract.
pub(crate) fn parse_session_entry_line(line: &str) -> Option<FileEntry> {
    if line.trim().is_empty() {
        return None;
    }
    serde_json::from_str::<JsonValue>(line)
        .ok()
        .map(parse_file_entry)
}

/// Parse one line's JSON value into a file entry, upstream's untyped
/// `JSON.parse` shape: a session-typed object becomes the header, a typed
/// entry becomes the entry, anything else rides as a raw value.
pub(crate) fn parse_file_entry(mut value: JsonValue) -> FileEntry {
    if value.get("type").and_then(JsonValue::as_str) == Some("session") {
        return serde_json::from_value::<SessionHeader>(value.clone())
            .map_or_else(|_| FileEntry::Other(value), FileEntry::Session);
    }
    repair_null_content_message(&mut value);
    serde_json::from_value::<SessionEntry>(value.clone())
        .map_or_else(|_| FileEntry::Other(value), FileEntry::Entry)
}

/// Load a session file's entries, upstream's `loadEntriesFromFile`.
///
/// The whole file must lead with a valid session header (first parsed entry
/// `"type": "session"` with a string id) or nothing is returned. A non-empty
/// tail without a terminating newline is repaired in place.
#[must_use]
pub fn load_entries_from_file(file_path: &str) -> Vec<FileEntry> {
    let resolved_file_path = normalize_path(file_path, &PathInputOptions::default())
        .unwrap_or_else(|_| file_path.to_owned());
    let Ok(bytes) = fs::read(&resolved_file_path) else {
        return Vec::new();
    };
    let content = String::from_utf8_lossy(&bytes).into_owned();

    let mut entries: Vec<FileEntry> = Vec::new();
    let mut ends_with_newline = true;
    for line in content.split('\n') {
        ends_with_newline = line.is_empty();
        if let Some(entry) = parse_session_entry_line(line) {
            entries.push(entry);
        }
    }

    // Validate session header before repairing the file: the first parsed
    // entry must be a session header (upstream's `type !== "session" ||
    // typeof id !== "string"` gate; the typed parse already guarantees the
    // string id).
    if entries.is_empty() || !matches!(entries.first(), Some(FileEntry::Session(_))) {
        return Vec::new();
    }

    if !ends_with_newline
        && let Ok(mut file) = fs::OpenOptions::new()
            .append(true)
            .open(&resolved_file_path)
    {
        let _ = file.write_all(b"\n");
    }
    entries
}

/// Why a bounded header scan stopped, upstream's read result.
pub(crate) enum HeaderCandidate {
    /// Keep scanning: blank or malformed line.
    KeepScanning,
    /// A parsed non-header entry: not a session.
    NotHeader,
    /// The session header.
    Header(SessionHeader),
}

/// Inspect one line while searching for the first parsed session entry,
/// upstream's `parseSessionHeaderCandidate`.
fn parse_session_header_candidate(line: &str) -> HeaderCandidate {
    if line.trim().is_empty() {
        return HeaderCandidate::KeepScanning;
    }
    let Ok(value) = serde_json::from_str::<JsonValue>(line) else {
        return HeaderCandidate::KeepScanning;
    };
    if value.get("type").and_then(JsonValue::as_str) != Some("session")
        || !value.get("id").is_some_and(JsonValue::is_string)
    {
        return HeaderCandidate::NotHeader;
    }
    serde_json::from_value::<SessionHeader>(value)
        .map_or_else(|_| HeaderCandidate::NotHeader, HeaderCandidate::Header)
}

/// Evaluate the trailing partial line at end-of-file, upstream's
/// `decoder.end()` flush.
fn finish_header_scan(tail: &[u8]) -> Option<SessionHeader> {
    let line = String::from_utf8_lossy(tail).into_owned();
    match parse_session_header_candidate(&line) {
        HeaderCandidate::Header(header) => Some(header),
        _ => None,
    }
}

/// Read a file's session header with the bounded scan, upstream's
/// `readSessionHeader`.
///
/// Blank and malformed lines are skipped; the first parsed entry decides: a
/// session header returns it, a non-header entry returns `None`. A header
/// line ending exactly at the scan limit is accepted; any additional byte
/// fails with [`ReadSessionHeaderError::ScanLimit`].
///
/// # Errors
/// [`ReadSessionHeaderError::ScanLimit`] past the bounded scan,
/// [`ReadSessionHeaderError::Io`] when the file cannot be read.
pub fn read_session_header(
    file_path: &Path,
) -> Result<Option<SessionHeader>, ReadSessionHeaderError> {
    let display = file_path.display().to_string();
    let mut file =
        fs::File::open(file_path).map_err(|error| ReadSessionHeaderError::Io(error.to_string()))?;

    let mut scanned: Vec<u8> = Vec::with_capacity(SESSION_HEADER_READ_BUFFER_SIZE * 2);
    let mut processed = 0usize;
    let mut chunk = [0u8; SESSION_HEADER_READ_BUFFER_SIZE];
    // The 1 MiB bound fits usize on every platform this port targets.
    let scan_limit = usize::try_from(MAX_SESSION_HEADER_SCAN_BYTES).unwrap_or(usize::MAX);

    while scanned.len() < scan_limit {
        let read_length = usize::min(chunk.len(), scan_limit - scanned.len());
        let bytes_read = file
            .read(&mut chunk[..read_length])
            .map_err(|error| ReadSessionHeaderError::Io(error.to_string()))?;
        if bytes_read == 0 {
            return Ok(finish_header_scan(&scanned[processed..]));
        }
        scanned.extend_from_slice(&chunk[..bytes_read]);

        while let Some(newline) = scanned[processed..].iter().position(|byte| *byte == b'\n') {
            let line =
                String::from_utf8_lossy(&scanned[processed..processed + newline]).into_owned();
            processed += newline + 1;
            match parse_session_header_candidate(&line) {
                HeaderCandidate::KeepScanning => {}
                HeaderCandidate::NotHeader => return Ok(None),
                HeaderCandidate::Header(header) => return Ok(Some(header)),
            }
        }
    }

    // Probe for EOF so a final header without a newline is allowed when it
    // ends exactly at the scan limit. Any additional byte exceeds the
    // bounded scan.
    let mut probe = [0u8; 1];
    match file.read(&mut probe) {
        Ok(0) => Ok(finish_header_scan(&scanned[processed..])),
        Ok(_) => Err(ReadSessionHeaderError::ScanLimit(display)),
        Err(error) => Err(ReadSessionHeaderError::Io(error.to_string())),
    }
}

/// Read a session header for discovery, upstream's
/// `readSessionHeaderForDiscovery`: best-effort, unreadable or oversized
/// files are not sessions.
pub(crate) fn read_session_header_for_discovery(file_path: &Path) -> Option<SessionHeader> {
    read_session_header(file_path).ok().flatten()
}

/// Whether a message entry carries an assistant payload, upstream's
/// `e.type === "message" && e.message.role === "assistant"` flush gate.
pub(crate) fn has_assistant_entry(entries: &[FileEntry]) -> bool {
    entries.iter().any(|entry| {
        matches!(
            entry,
            FileEntry::Entry(SessionEntry::Message(MessageEntry {
                message: Some(pi_agent_core::types::AgentMessage::Standard(
                    pi_ai::types::Message::Assistant(_)
                )),
                ..
            }))
        )
    })
}

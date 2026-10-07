//! The session format migrations, upstream's `migrateV1ToV2`,
//! `migrateV2ToV3`, and `migrateToCurrentVersion` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::collections::HashSet;

use serde_json::Value as JsonValue;

use super::entries::{CompactionEntry, SessionEntry};
use super::file_entry::FileEntry;

/// Generate a unique short id (8 hex chars, collision-checked), upstream's
/// `generateId`: a UUIDv4's first 8 chars, retried up to 100 times against
/// the taken predicate, falling back to a full hyphenated UUID.
pub(crate) fn generate_id(is_taken: impl Fn(&str) -> bool) -> String {
    for _ in 0..100 {
        let id = short_uuid_v4();
        if !is_taken(&id) {
            return id;
        }
    }
    uuid::Uuid::new_v4().hyphenated().to_string()
}

/// A UUIDv4's first 8 hex characters, upstream's `randomUUID().slice(0, 8)`.
fn short_uuid_v4() -> String {
    let mut simple = uuid::Uuid::new_v4().simple().to_string();
    simple.truncate(8);
    simple
}

/// Migrate v1 → v2: add id/parentId tree structure, upstream's
/// `migrateV1ToV2`. Mutates in place.
///
/// The compaction kept-index resolution reads the ids assigned so far —
/// upstream resolves `entries[firstKeptEntryIndex]` mid-loop, so a
/// forward-pointing index observes an unassigned (absent) id, exactly what
/// the parallel assigned-ids array reproduces.
fn migrate_v1_to_v2(entries: &mut [FileEntry]) {
    let mut taken: HashSet<String> = HashSet::new();
    let mut prev_id: Option<String> = None;
    let mut assigned_ids: Vec<Option<String>> = std::vec::from_elem(None, entries.len());

    for (position, entry) in entries.iter_mut().enumerate() {
        if let FileEntry::Session(header) = entry {
            header.version = Some(2);
            continue;
        }

        let id = generate_id(|id| taken.contains(id));
        taken.insert(id.clone());
        assigned_ids[position] = Some(id.clone());
        let parent_id = prev_id.take();
        prev_id = Some(id.clone());
        assign_tree_fields(entry, id, parent_id);

        // Convert firstKeptEntryIndex to firstKeptEntryId for compaction.
        let kept_index = match entry {
            FileEntry::Entry(SessionEntry::Compaction(compaction)) => compaction
                .extras
                .get("firstKeptEntryIndex")
                .and_then(JsonValue::as_i64),
            FileEntry::Other(value) => value.get("firstKeptEntryIndex").and_then(JsonValue::as_i64),
            _ => None,
        };
        if let Some(index) = kept_index {
            // A negative index (hand-edited file) misses the assigned ids
            // the way upstream's array read of a negative index returns
            // undefined; usize::MAX is never assigned.
            let target = assigned_ids
                .get(usize::try_from(index).unwrap_or(usize::MAX))
                .cloned()
                .flatten();
            match entry {
                FileEntry::Entry(SessionEntry::Compaction(compaction)) => {
                    compaction.first_kept_entry_id = target;
                    compaction.extras.remove("firstKeptEntryIndex");
                }
                FileEntry::Other(value) => {
                    if let Some(object) = value.as_object_mut() {
                        if let Some(target) = target {
                            object.insert("firstKeptEntryId".to_owned(), JsonValue::String(target));
                        }
                        object.remove("firstKeptEntryIndex");
                    }
                }
                _ => {}
            }
        }
    }
}

/// Assign the freshly generated tree fields to a typed entry or a raw value,
/// the in-place half of upstream's v1 migration.
fn assign_tree_fields(entry: &mut FileEntry, id: String, parent_id: Option<String>) {
    match entry {
        FileEntry::Entry(typed) => {
            let base = typed.base_mut();
            base.id = Some(id);
            base.parent_id = parent_id;
        }
        FileEntry::Other(value) => {
            if let Some(object) = value.as_object_mut() {
                object.insert("id".to_owned(), JsonValue::String(id));
                object.insert(
                    "parentId".to_owned(),
                    parent_id.map_or(JsonValue::Null, JsonValue::String),
                );
            }
        }
        FileEntry::Session(_) => {}
    }
}

/// Migrate v2 → v3: rename the hookMessage role to custom, upstream's
/// `migrateV2ToV3`. Mutates in place.
fn migrate_v2_to_v3(entries: &mut [FileEntry]) {
    for entry in entries.iter_mut() {
        match entry {
            FileEntry::Session(header) => header.version = Some(3),
            FileEntry::Entry(SessionEntry::Message(message_entry)) => {
                if let Some(pi_agent_core::types::AgentMessage::Custom(custom)) =
                    message_entry.message.as_mut()
                    && custom.role == "hookMessage"
                {
                    custom.role = String::from("custom");
                }
            }
            FileEntry::Other(value) => {
                if value.get("type").and_then(JsonValue::as_str) == Some("message")
                    && value
                        .get("message")
                        .and_then(|message| message.get("role"))
                        .and_then(JsonValue::as_str)
                        == Some("hookMessage")
                    && let Some(object) =
                        value.get_mut("message").and_then(JsonValue::as_object_mut)
                {
                    object.insert("role".to_owned(), JsonValue::String("custom".to_owned()));
                }
            }
            FileEntry::Entry(_) => {}
        }
    }
}

/// Run all necessary migrations to bring entries to the current version,
/// upstream's `migrateToCurrentVersion`. Mutates in place; returns true when
/// any migration was applied.
pub(crate) fn migrate_to_current_version(entries: &mut [FileEntry]) -> bool {
    let version = entries
        .iter()
        .find_map(|entry| match entry {
            FileEntry::Session(header) => Some(header.version.unwrap_or(1)),
            _ => None,
        })
        .unwrap_or(1);

    if version >= super::types::CURRENT_SESSION_VERSION {
        return false;
    }

    if version < 2 {
        migrate_v1_to_v2(entries);
    }
    if version < 3 {
        migrate_v2_to_v3(entries);
    }

    true
}

/// Migrate entries to the current version, upstream's exported
/// `migrateSessionEntries` (exported for testing).
pub fn migrate_session_entries(entries: &mut [FileEntry]) -> bool {
    migrate_to_current_version(entries)
}

/// Parse a JSONL transcript's lines into entries, skipping malformed lines,
/// upstream's exported `parseSessionEntries` (exported for compaction).
///
/// The split stays on `\n` (upstream's `split("\n")`): a `\r` survives the
/// trim and breaks the line's JSON parse, so CRLF files degrade to empty
/// exactly as upstream's do.
#[expect(
    clippy::str_split_at_newline,
    reason = "upstream splits on \\n verbatim; lines() would strip a \\r and silently widen the parse past upstream's behavior"
)]
#[must_use]
pub fn parse_session_entries(content: &str) -> Vec<FileEntry> {
    content
        .trim()
        .split('\n')
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            serde_json::from_str::<JsonValue>(line)
                .ok()
                .map(super::parse::parse_file_entry)
        })
        .collect()
}

/// The latest compaction entry in a transcript, upstream's
/// `getLatestCompactionEntry`.
#[must_use]
pub fn get_latest_compaction_entry(entries: &[FileEntry]) -> Option<&FileEntry> {
    entries.iter().rev().find(|entry| {
        matches!(
            entry,
            FileEntry::Entry(SessionEntry::Compaction(CompactionEntry { .. }))
        )
    })
}

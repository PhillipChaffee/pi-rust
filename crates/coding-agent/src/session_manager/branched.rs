//! Branched-session extraction, upstream's `createBranchedSession` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::collections::{HashMap, HashSet};

use serde_json::Value as JsonValue;

use pi_ai::utils::uuid::uuidv7;

use super::entries::{LabelEntry, SessionEntry};
use super::file_entry::{FileEntry, SessionManagerError};
use super::manager::SessionManager;
use super::parse::has_assistant_entry;
use super::types::{CURRENT_SESSION_VERSION, SessionHeader};
use super::{file_timestamp, now_iso8601};

impl SessionManager {
    /// Create a new session file containing only the path from root to the
    /// specified leaf, upstream's `createBranchedSession`: useful for
    /// extracting a single conversation path from a branched session.
    /// Returns the new session file path, or absent when not persisting.
    ///
    /// # Errors
    /// [`SessionManagerError::EntryNotFound`] when the leaf is unknown.
    #[expect(
        clippy::too_many_lines,
        reason = "the function restates upstream's createBranchedSession step for step; splitting the label re-chaining would scatter one contract"
    )]
    pub fn create_branched_session(
        &mut self,
        leaf_id: &str,
    ) -> Result<Option<String>, SessionManagerError> {
        let previous_session_file = self.session_file.clone();
        let path: Vec<FileEntry> = self
            .get_branch(Some(leaf_id))
            .into_iter()
            .cloned()
            .collect();
        if path.is_empty() {
            return Err(SessionManagerError::EntryNotFound(leaf_id.to_owned()));
        }

        // Filter out LabelEntry from path — recreate them from the resolved
        // map. Because labels are real tree entries, later entries can be
        // children of labels; removing labels requires re-chaining the
        // retained path to avoid orphaned subtrees.
        let mut path_without_labels: Vec<FileEntry> = Vec::new();
        let mut replacement_by_label_id: HashMap<String, String> = HashMap::new();
        let mut pending_label_ids: Vec<String> = Vec::new();
        let mut path_parent_id: Option<String> = None;
        for entry in &path {
            let Some(id) = entry.entry_id() else { continue };
            let id = id.to_owned();
            if matches!(entry, FileEntry::Entry(SessionEntry::Label(_))) {
                pending_label_ids.push(id);
                continue;
            }
            for label_id in std::mem::take(&mut pending_label_ids) {
                replacement_by_label_id.insert(label_id, id.clone());
            }
            let mut cloned = entry.clone();
            rechain_entry(
                &mut cloned,
                path_parent_id.as_deref(),
                &replacement_by_label_id,
            );
            path_parent_id = Some(id);
            path_without_labels.push(cloned);
        }

        let new_session_id =
            uuidv7(None).map_err(|error| SessionManagerError::Io(error.to_string()))?;
        let timestamp = now_iso8601();
        let new_session_file = std::path::Path::new(&self.session_dir)
            .join(format!(
                "{}_{}.jsonl",
                file_timestamp(&timestamp),
                new_session_id
            ))
            .display()
            .to_string();

        let header = SessionHeader {
            version: Some(CURRENT_SESSION_VERSION),
            id: new_session_id.clone(),
            timestamp,
            cwd: Some(self.cwd.clone()),
            parent_session: self
                .persist
                .then(|| previous_session_file.clone())
                .flatten(),
            extras: serde_json::Map::default(),
        };

        // Collect labels for entries in the path.
        let mut path_entry_ids: HashSet<String> = path_without_labels
            .iter()
            .filter_map(FileEntry::entry_id)
            .map(str::to_owned)
            .collect();
        let labels_to_write: Vec<(String, String, String)> = self
            .labels_by_id
            .iter()
            .filter(|(target, _, _)| path_entry_ids.contains(target))
            .cloned()
            .collect();

        let last_entry_id = path_without_labels
            .last()
            .and_then(FileEntry::entry_id)
            .map(str::to_owned);
        if self.persist {
            // Build label entries.
            let mut parent_id = last_entry_id;
            let mut label_entries: Vec<FileEntry> = Vec::new();
            for (target_id, label, label_timestamp) in labels_to_write {
                let id = super::generate_id(|id| path_entry_ids.contains(id));
                path_entry_ids.insert(id.clone());
                label_entries.push(label_entry(
                    &id,
                    parent_id.as_deref(),
                    &label_timestamp,
                    &target_id,
                    &label,
                ));
                parent_id = Some(id);
            }

            self.file_entries = Vec::new();
            self.file_entries.push(FileEntry::Session(header));
            self.file_entries.extend(path_without_labels);
            self.file_entries.extend(label_entries);
            self.session_id = new_session_id;
            self.session_file = Some(new_session_file.clone());
            self.build_index();

            // Only write the file now if it contains an assistant message.
            // Otherwise defer to persist, which creates the file on the
            // first assistant response, matching the newSession contract
            // and avoiding the duplicate-header bug when the no-assistant
            // guard later resets the flush.
            if has_assistant_entry(&self.file_entries) {
                self.rewrite_file()?;
                self.flushed = true;
            } else {
                self.flushed = false;
            }

            return Ok(Some(new_session_file));
        }

        // In-memory mode: replace current session with the path + labels.
        let mut label_entries: Vec<FileEntry> = Vec::new();
        let mut parent_id = last_entry_id;
        for (target_id, label, label_timestamp) in labels_to_write {
            let id = super::generate_id(|id| {
                path_entry_ids.contains(id)
                    || label_entries
                        .iter()
                        .any(|entry| entry.entry_id() == Some(id))
            });
            label_entries.push(label_entry(
                &id,
                parent_id.as_deref(),
                &label_timestamp,
                &target_id,
                &label,
            ));
            parent_id = Some(id);
        }
        self.file_entries = Vec::new();
        self.file_entries.push(FileEntry::Session(header));
        self.file_entries.extend(path_without_labels);
        self.file_entries.extend(label_entries);
        self.session_id = new_session_id;
        self.build_index();
        Ok(None)
    }
}

/// Re-chain one retained path entry onto its new parent, upstream's
/// `{ ...entry, parentId }` spread with the compaction kept-id remap.
fn rechain_entry(
    entry: &mut FileEntry,
    path_parent_id: Option<&str>,
    replacements: &HashMap<String, String>,
) {
    match entry {
        FileEntry::Entry(typed) => {
            typed.base_mut().parent_id = path_parent_id.map(str::to_owned);
            if let SessionEntry::Compaction(compaction) = typed
                && let Some(kept) = &compaction.first_kept_entry_id
                && let Some(replacement) = replacements.get(kept)
            {
                compaction.first_kept_entry_id = Some(replacement.clone());
            }
        }
        FileEntry::Other(value) => {
            if let Some(object) = value.as_object_mut() {
                object.insert(
                    "parentId".to_owned(),
                    path_parent_id.map_or(JsonValue::Null, |parent| {
                        JsonValue::String(parent.to_owned())
                    }),
                );
                if object.get("type").and_then(JsonValue::as_str) == Some("compaction")
                    && let Some(kept) = object.get("firstKeptEntryId").and_then(JsonValue::as_str)
                    && let Some(replacement) = replacements.get(kept)
                {
                    object.insert(
                        "firstKeptEntryId".to_owned(),
                        JsonValue::String(replacement.clone()),
                    );
                }
            }
        }
        FileEntry::Session(_) => {}
    }
}

/// One rebuilt label entry, upstream's label-entry literal.
fn label_entry(
    id: &str,
    parent_id: Option<&str>,
    timestamp: &str,
    target_id: &str,
    label: &str,
) -> FileEntry {
    FileEntry::Entry(SessionEntry::Label(LabelEntry {
        base: super::entries::SessionEntryBase {
            id: Some(id.to_owned()),
            parent_id: parent_id.map(str::to_owned),
            timestamp: timestamp.to_owned(),
            extras: serde_json::Map::default(),
        },
        target_id: target_id.to_owned(),
        label: Some(label.to_owned()),
        extras: serde_json::Map::default(),
    }))
}

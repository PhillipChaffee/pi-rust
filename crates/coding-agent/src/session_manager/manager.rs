//! The session manager state machine, upstream's private `SessionManager`
//! constructor and file mechanics at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::Path;

use pi_ai::utils::uuid::uuidv7;

use super::discovery::{get_default_session_dir_path, session_cwd_matches};
use super::entries::SessionEntry;
use super::file_entry::{FileEntry, SessionManagerError};
use super::migrate::migrate_to_current_version;
use super::parse::{has_assistant_entry, load_entries_from_file};
use super::types::{CURRENT_SESSION_VERSION, NewSessionOptions, SessionHeader};
use super::{append_line, file_timestamp, normalize_here, now_iso8601, path_exists, resolve_here};

/// Manages conversation sessions as append-only trees stored in JSONL files,
/// upstream's `SessionManager`.
///
/// Each session entry has an id and parentId forming a tree structure. The
/// leaf pointer tracks the current position. Appending creates a child of
/// the current leaf. Branching moves the leaf to an earlier entry, allowing
/// new branches without modifying history. Use
/// [`SessionManager::build_session_context`] to get the resolved message
/// list for the LLM, which handles compaction summaries and follows the path
/// from root to current leaf.
#[derive(Debug, Clone)]
pub struct SessionManager {
    pub(crate) session_id: String,
    pub(crate) session_file: Option<String>,
    pub(crate) session_dir: String,
    pub(crate) cwd: String,
    pub(crate) persist: bool,
    pub(crate) flushed: bool,
    pub(crate) file_entries: Vec<FileEntry>,
    pub(crate) by_id: HashMap<String, usize>,
    /// Resolved labels as `(target id, label, label-entry timestamp)` in the
    /// label entries' file order — upstream's `labelsById` +
    /// `labelTimestampsById` maps, whose insertion order the branched-session
    /// label rebuild preserves. Re-setting a target replaces in place, the
    /// way `Map.set` keeps an existing key's position.
    pub(crate) labels_by_id: Vec<(String, String, String)>,
    pub(crate) leaf_id: Option<String>,
}

impl SessionManager {
    /// The private constructor, upstream's `new SessionManager(...)`.
    pub(crate) fn new(
        cwd: &str,
        session_dir: &str,
        session_file: Option<&str>,
        persist: bool,
        new_session_options: Option<&NewSessionOptions>,
        preloaded_file_entries: Option<Vec<FileEntry>>,
    ) -> Result<Self, SessionManagerError> {
        let mut manager = Self {
            session_id: String::new(),
            session_file: None,
            session_dir: normalize_here(session_dir)?,
            cwd: resolve_here(cwd),
            persist,
            flushed: false,
            file_entries: Vec::new(),
            by_id: HashMap::new(),
            labels_by_id: Vec::new(),
            leaf_id: None,
        };

        if manager.persist && !manager.session_dir.is_empty() && !path_exists(&manager.session_dir)
        {
            fs::create_dir_all(&manager.session_dir)?;
        }

        if let Some(session_file) = session_file {
            manager.set_session_file_inner(session_file, preloaded_file_entries)?;
        } else if preloaded_file_entries
            .as_ref()
            .is_some_and(|entries| !entries.is_empty())
        {
            manager.load_entries(
                preloaded_file_entries.unwrap_or_default(),
                new_session_options,
            )?;
        } else {
            manager.new_session(new_session_options.cloned())?;
        }
        Ok(manager)
    }

    /// Switch to a different session file (used for resume and branching),
    /// upstream's `setSessionFile`.
    ///
    /// # Errors
    /// [`SessionManagerError::InvalidSessionFile`] when the file exists but
    /// is not a pi session.
    pub fn set_session_file(&mut self, session_file: &str) -> Result<(), SessionManagerError> {
        self.set_session_file_inner(session_file, None)
    }

    fn set_session_file_inner(
        &mut self,
        session_file: &str,
        preloaded_file_entries: Option<Vec<FileEntry>>,
    ) -> Result<(), SessionManagerError> {
        let resolved = resolve_here(session_file);
        self.session_file = Some(resolved.clone());
        if path_exists(&resolved) {
            let entries =
                preloaded_file_entries.unwrap_or_else(|| load_entries_from_file(&resolved));

            // If file was empty, initialize it with a valid session header.
            // If it was non-empty but did not parse as a pi session, fail
            // without modifying it.
            if entries.is_empty() {
                if fs::metadata(&resolved)?.len() > 0 {
                    return Err(SessionManagerError::InvalidSessionFile(resolved));
                }
                self.new_session(None)?;
                self.session_file = Some(resolved);
                self.rewrite_file()?;
                self.flushed = true;
                return Ok(());
            }

            self.load_entries(entries, None)?;
            self.flushed = true;
        } else {
            self.new_session(None)?;
            self.session_file = Some(resolved); // preserve explicit path from --session flag
        }
        Ok(())
    }

    /// Start a new session, upstream's `newSession`. Returns the session
    /// file path when persisting.
    ///
    /// # Errors
    /// [`SessionManagerError::InvalidSessionId`] when `options.id` is
    /// invalid.
    pub fn new_session(
        &mut self,
        options: Option<NewSessionOptions>,
    ) -> Result<Option<String>, SessionManagerError> {
        if let Some(id) = options.as_ref().and_then(|options| options.id.as_deref()) {
            super::assert_valid_session_id(id)?;
        }
        self.session_id = match options.as_ref().and_then(|options| options.id.clone()) {
            Some(id) => id,
            None => uuidv7(None).map_err(|error| SessionManagerError::Io(error.to_string()))?,
        };
        let timestamp = now_iso8601();
        let header = FileEntry::Session(SessionHeader {
            version: Some(CURRENT_SESSION_VERSION),
            id: self.session_id.clone(),
            timestamp: timestamp.clone(),
            cwd: Some(self.cwd.clone()),
            parent_session: options.and_then(|options| options.parent_session),
            extras: serde_json::Map::default(),
        });
        self.file_entries = vec![header];
        self.by_id.clear();
        self.labels_by_id.clear();
        self.leaf_id = None;
        self.flushed = false;

        let session_file = self.persist.then(|| {
            let file = Path::new(&self.session_dir)
                .join(format!(
                    "{}_{}.jsonl",
                    file_timestamp(&timestamp),
                    self.session_id
                ))
                .display()
                .to_string();
            self.session_file = Some(file.clone());
            file
        });
        Ok(session_file)
    }

    /// Load parsed entries into the manager, upstream's `_loadEntries`.
    fn load_entries(
        &mut self,
        entries: Vec<FileEntry>,
        options: Option<&NewSessionOptions>,
    ) -> Result<(), SessionManagerError> {
        let header_index = entries
            .iter()
            .position(|entry| matches!(entry, FileEntry::Session(_)));

        if let Some(index) = header_index {
            let Some(FileEntry::Session(header)) = entries.get(index) else {
                unreachable!("position matched a session header");
            };
            let session_id = header.id.clone();
            self.file_entries = entries;
            self.session_id = session_id;

            if migrate_to_current_version(&mut self.file_entries) {
                self.rewrite_file()?;
            }
        } else {
            self.new_session(options.cloned())?;
            self.file_entries.extend(entries);
        }

        self.build_index();
        Ok(())
    }

    /// Rebuild the id index and resolved labels, upstream's `_buildIndex`.
    pub(crate) fn build_index(&mut self) {
        self.by_id.clear();
        self.labels_by_id.clear();
        self.leaf_id = None;
        for (index, entry) in self.file_entries.iter().enumerate() {
            let Some(id) = indexed_entry_id(entry) else {
                continue;
            };
            self.by_id.insert(id.to_owned(), index);
            self.leaf_id = Some(id.to_owned());
            // Capture the label update before mutating: the entry borrow and
            // the label-table write are disjoint fields, but the label
            // helpers take `&mut self`.
            let update = match entry {
                FileEntry::Entry(SessionEntry::Label(label)) => Some((
                    label.target_id.clone(),
                    label.label.clone().filter(|value| !value.is_empty()),
                    label.base.timestamp.clone(),
                )),
                _ => None,
            };
            if let Some((target, label, timestamp)) = update {
                match label {
                    Some(text) => set_label(&mut self.labels_by_id, &target, &text, &timestamp),
                    None => clear_label(&mut self.labels_by_id, &target),
                }
            }
        }
    }

    /// Set (or replace in place) one resolved label, upstream's
    /// `labelsById.set` — an existing target keeps its position.
    pub(crate) fn set_label(&mut self, target_id: &str, label: &str, timestamp: &str) {
        set_label(&mut self.labels_by_id, target_id, label, timestamp);
    }

    /// Clear one resolved label, upstream's `labelsById.delete`.
    pub(crate) fn clear_label(&mut self, target_id: &str) {
        clear_label(&mut self.labels_by_id, target_id);
    }

    /// Rewrite the session file from the in-memory entries, upstream's
    /// `_rewriteFile`. No-op when not persisting.
    pub(crate) fn rewrite_file(&self) -> Result<(), SessionManagerError> {
        if !self.persist {
            return Ok(());
        }
        let Some(session_file) = &self.session_file else {
            return Ok(());
        };
        let mut file = fs::File::create(session_file)?;
        for entry in &self.file_entries {
            let line = serde_json::to_string(entry)
                .map_err(|error| SessionManagerError::Io(error.to_string()))?;
            writeln!(file, "{line}")?;
        }
        Ok(())
    }

    /// Write one entry line to the session file under the flush rules,
    /// upstream's `_persist`: entries stay in memory until the first
    /// assistant message, then the whole buffer flushes once (exclusive
    /// create, upstream's `"wx"`) and later entries append. The caller has
    /// already pushed the entry, so the assistant gate scans it.
    pub(crate) fn persist_entry(&mut self, line: &str) -> Result<(), SessionManagerError> {
        if !self.persist {
            return Ok(());
        }
        let Some(session_file) = self.session_file.clone() else {
            return Ok(());
        };

        if !has_assistant_entry(&self.file_entries) {
            if self.flushed {
                append_line(&session_file, line)?;
            } else {
                // Mark as not flushed so when assistant arrives, all entries
                // get written.
                self.flushed = false;
            }
            return Ok(());
        }

        if self.flushed {
            append_line(&session_file, line)?;
        } else {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&session_file)?;
            for entry in &self.file_entries {
                let line = serde_json::to_string(entry)
                    .map_err(|error| SessionManagerError::Io(error.to_string()))?;
                writeln!(file, "{line}")?;
            }
            self.flushed = true;
        }
        Ok(())
    }

    /// Append a typed entry as a child of the current leaf and advance the
    /// leaf, upstream's `_appendEntry`. Returns the entry id.
    pub(crate) fn append_entry(
        &mut self,
        entry: SessionEntry,
    ) -> Result<String, SessionManagerError> {
        let id = entry.base().id.clone().unwrap_or_default();
        let line = serde_json::to_string(&entry)
            .map_err(|error| SessionManagerError::Io(error.to_string()))?;
        self.file_entries.push(FileEntry::Entry(entry));
        let index = self.file_entries.len() - 1;
        if !id.is_empty() {
            self.by_id.insert(id.clone(), index);
        }
        self.leaf_id = Some(id.clone());
        self.persist_entry(&line)?;
        Ok(id)
    }

    /// Whether the session directory is the default one for the cwd,
    /// upstream's `usesDefaultSessionDir` (file mechanics half).
    pub(crate) fn uses_default_session_dir_value(&self) -> bool {
        self.session_dir
            == get_default_session_dir_path(
                &self.cwd,
                &crate::config::get_agent_dir().display().to_string(),
            )
            .display()
            .to_string()
    }

    /// Whether a cwd filter applies for discovery on this directory,
    /// upstream's `filterCwd` computation shared by `continueRecent` and
    /// `list`.
    pub(crate) fn discovery_filters_cwd(session_dir: Option<&str>, dir: &str, cwd: &str) -> bool {
        session_dir.is_some()
            && dir
                != get_default_session_dir_path(
                    cwd,
                    &crate::config::get_agent_dir().display().to_string(),
                )
                .display()
                .to_string()
    }

    /// Whether a session's cwd matches the resolved cwd, the listing-side
    /// re-export of the shared matcher.
    pub(crate) fn cwd_matches(session_cwd: &str, resolved_cwd: &str) -> bool {
        session_cwd_matches(Some(session_cwd), resolved_cwd)
    }

    /// The next entry's base fields, upstream's per-append literals.
    pub(crate) fn next_base(&self) -> super::entries::SessionEntryBase {
        super::entries::SessionEntryBase {
            id: Some(super::generate_id(|id| self.by_id.contains_key(id))),
            parent_id: self.leaf_id.clone(),
            timestamp: now_iso8601(),
            extras: serde_json::Map::default(),
        }
    }
}

/// Set (or replace in place) one resolved label, upstream's
/// `labelsById.set` — an existing target keeps its position. A free function
/// over the label table so the index rebuild can call it while iterating the
/// entry list (the methods take `&mut self`, which would fight the iterator
/// borrow).
pub(crate) fn set_label(
    labels: &mut Vec<(String, String, String)>,
    target_id: &str,
    label: &str,
    timestamp: &str,
) {
    let resolved = (target_id.to_owned(), label.to_owned(), timestamp.to_owned());
    match labels.iter().position(|(target, _, _)| target == target_id) {
        Some(position) => labels[position] = resolved,
        None => labels.push(resolved),
    }
}

/// Clear one resolved label, upstream's `labelsById.delete`.
pub(crate) fn clear_label(labels: &mut Vec<(String, String, String)>, target_id: &str) {
    labels.retain(|(target, _, _)| target != target_id);
}

/// The entry id the tree and index key on, present for typed entries and raw
/// values carrying one.
fn indexed_entry_id(entry: &FileEntry) -> Option<&str> {
    match entry {
        FileEntry::Session(_) => None,
        _ => entry.entry_id(),
    }
}

//! The entry-append surface and accessors, upstream's `appendMessage`
//! through `appendCustomMessageEntry`, `appendLabelChange`, and the simple
//! getters at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use serde_json::Value as JsonValue;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{Usage, UserContent};

use super::entries::{LabelEntry, SessionEntry, SessionEntryBase};
use super::file_entry::FileEntry;
use super::file_entry::SessionManagerError;
use super::manager::SessionManager;
use super::{now_iso8601, sanitize_session_name};

impl SessionManager {
    /// Whether the manager persists to a file, upstream's `isPersisted`.
    #[must_use]
    pub const fn is_persisted(&self) -> bool {
        self.persist
    }

    /// The session's resolved cwd, upstream's `getCwd`.
    #[must_use]
    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    /// The session directory, upstream's `getSessionDir`.
    #[must_use]
    pub fn session_dir(&self) -> &str {
        &self.session_dir
    }

    /// Whether the session lives in the default directory for its cwd,
    /// upstream's `usesDefaultSessionDir`.
    #[must_use]
    pub fn uses_default_session_dir(&self) -> bool {
        self.uses_default_session_dir_value()
    }

    /// The session id, upstream's `getSessionId`.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The session file path, upstream's `getSessionFile`; absent for an
    /// in-memory session.
    #[must_use]
    pub fn session_file(&self) -> Option<&str> {
        self.session_file.as_deref()
    }

    /// Append a message as child of current leaf, then advance leaf, upstream's
    /// `appendMessage`. Returns the entry id.
    ///
    /// Does not allow writing compaction-summary and branch-summary messages
    /// directly: those are top-level entries appended via
    /// [`SessionManager::append_compaction`] and
    /// [`SessionManager::branch_with_summary`], upstream's documented
    /// contract.
    ///
    /// # Errors
    /// [`SessionManagerError::Io`] when the flushed write races an existing
    /// file, upstream's exclusive-create throw.
    pub fn append_message(&mut self, message: AgentMessage) -> Result<String, SessionManagerError> {
        let entry = SessionEntry::Message(super::entries::MessageEntry {
            base: SessionEntryBase {
                id: Some(super::generate_id(|id| self.by_id.contains_key(id))),
                parent_id: self.leaf_id.clone(),
                timestamp: now_iso8601(),
                extras: serde_json::Map::default(),
            },
            message: Some(message),
            extras: serde_json::Map::default(),
        });
        self.append_entry(entry)
    }

    /// Append a thinking level change as child of current leaf, then advance
    /// leaf, upstream's `appendThinkingLevelChange`. Returns the entry id.
    ///
    /// # Errors
    /// [`SessionManagerError::Io`] when the flushed write races an existing
    /// file.
    pub fn append_thinking_level_change(
        &mut self,
        thinking_level: &str,
    ) -> Result<String, SessionManagerError> {
        let entry = SessionEntry::ThinkingLevelChange(super::entries::ThinkingLevelChangeEntry {
            base: self.next_base(),
            thinking_level: thinking_level.to_owned(),
            extras: serde_json::Map::default(),
        });
        self.append_entry(entry)
    }

    /// Append a model change as child of current leaf, then advance leaf,
    /// upstream's `appendModelChange`. Returns the entry id.
    ///
    /// # Errors
    /// [`SessionManagerError::Io`] when the flushed write races an existing
    /// file.
    pub fn append_model_change(
        &mut self,
        provider: &str,
        model_id: &str,
    ) -> Result<String, SessionManagerError> {
        let entry = SessionEntry::ModelChange(super::entries::ModelChangeEntry {
            base: self.next_base(),
            provider: provider.to_owned(),
            model_id: model_id.to_owned(),
            extras: serde_json::Map::default(),
        });
        self.append_entry(entry)
    }

    /// Append a compaction summary as child of current leaf, then advance
    /// leaf, upstream's `appendCompaction`. Returns the entry id.
    ///
    /// # Errors
    /// [`SessionManagerError::Io`] when the flushed write races an existing
    /// file.
    pub fn append_compaction(
        &mut self,
        summary: &str,
        first_kept_entry_id: &str,
        tokens_before: i64,
        details: Option<JsonValue>,
        from_hook: Option<bool>,
        usage: Option<Usage>,
    ) -> Result<String, SessionManagerError> {
        let entry = SessionEntry::Compaction(super::entries::CompactionEntry {
            base: self.next_base(),
            summary: summary.to_owned(),
            first_kept_entry_id: Some(first_kept_entry_id.to_owned()),
            tokens_before,
            details,
            usage,
            from_hook,
            extras: serde_json::Map::default(),
        });
        self.append_entry(entry)
    }

    /// Append a custom entry (for extensions) as child of current leaf, then
    /// advance leaf, upstream's `appendCustomEntry`. Returns the entry id.
    ///
    /// # Errors
    /// [`SessionManagerError::Io`] when the flushed write races an existing
    /// file.
    pub fn append_custom_entry(
        &mut self,
        custom_type: &str,
        data: Option<JsonValue>,
    ) -> Result<String, SessionManagerError> {
        let entry = SessionEntry::Custom(super::entries::CustomEntry {
            custom_type: custom_type.to_owned(),
            data,
            base: self.next_base(),
            extras: serde_json::Map::default(),
        });
        self.append_entry(entry)
    }

    /// Append a session info entry (e.g. display name), upstream's
    /// `appendSessionInfo`. Returns the entry id.
    ///
    /// # Errors
    /// [`SessionManagerError::Io`] when the flushed write races an existing
    /// file.
    pub fn append_session_info(&mut self, name: &str) -> Result<String, SessionManagerError> {
        let sanitized_name = sanitize_session_name(name);
        let entry = SessionEntry::SessionInfo(super::entries::SessionInfoEntry {
            base: self.next_base(),
            name: Some(sanitized_name),
            extras: serde_json::Map::default(),
        });
        self.append_entry(entry)
    }

    /// The current session name from the latest `session_info` entry,
    /// upstream's `getSessionName`: the reverse scan stops at the first
    /// `session_info` entry, so an explicit clear resolves to absent instead
    /// of falling through to an older name.
    #[must_use]
    pub fn get_session_name(&self) -> Option<String> {
        for entry in self.entries().into_iter().rev() {
            if let FileEntry::Entry(SessionEntry::SessionInfo(info)) = entry {
                return info
                    .name
                    .as_deref()
                    .map(str::trim)
                    .filter(|trimmed| !trimmed.is_empty())
                    .map(str::to_owned);
            }
        }
        None
    }

    /// Append a custom message entry (for extensions) that participates in
    /// LLM context, upstream's `appendCustomMessageEntry`. Returns the entry
    /// id.
    ///
    /// # Errors
    /// [`SessionManagerError::Io`] when the flushed write races an existing
    /// file.
    pub fn append_custom_message_entry(
        &mut self,
        custom_type: &str,
        content: UserContent,
        display: bool,
        details: Option<JsonValue>,
    ) -> Result<String, SessionManagerError> {
        let entry = SessionEntry::CustomMessage(super::entries::CustomMessageEntry {
            custom_type: custom_type.to_owned(),
            content: Some(content),
            display,
            details,
            base: self.next_base(),
            extras: serde_json::Map::default(),
        });
        self.append_entry(entry)
    }

    /// Set or clear a label on an entry, upstream's `appendLabelChange`.
    /// Labels are user-defined markers for bookmarking/navigation; an absent
    /// or empty label clears.
    ///
    /// # Errors
    /// [`SessionManagerError::EntryNotFound`] when `target_id` is unknown,
    /// upstream's throw.
    pub fn append_label_change(
        &mut self,
        target_id: &str,
        label: Option<&str>,
    ) -> Result<String, SessionManagerError> {
        if !self.by_id.contains_key(target_id) {
            return Err(SessionManagerError::EntryNotFound(target_id.to_owned()));
        }
        let entry = SessionEntry::Label(LabelEntry {
            base: self.next_base(),
            target_id: target_id.to_owned(),
            label: label.map(str::to_owned),
            extras: serde_json::Map::default(),
        });
        let timestamp = entry.base().timestamp.clone();
        let id = self.append_entry(entry)?;
        match label.filter(|value| !value.is_empty()) {
            Some(label) => self.set_label(target_id, label, &timestamp),
            None => self.clear_label(target_id),
        }
        Ok(id)
    }
}

//! The context projections, upstream's `buildSessionPath`, `buildContextEntries`,
//! `buildSessionContext`, and `sessionEntryToContextMessages` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::collections::HashMap;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::Message;

use super::entries::{CompactionEntry, SessionEntry};
use super::file_entry::FileEntry;
use crate::messages::{
    create_branch_summary_message, create_compaction_summary_message, create_custom_message,
};

/// The id → file-entry index, upstream's `byId` map.
pub type ByIdIndex = HashMap<String, usize>;

/// The leaf pointer argument, upstream's `leafId?: string | null`:
/// [`LeafId::Default`] selects the newest entry, [`LeafId::None`] selects the
/// empty path, and [`LeafId::Id`] selects that entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafId<'a> {
    /// Upstream's absent `leafId`.
    Default,
    /// Upstream's explicit `null`.
    None,
    /// An explicit entry id.
    Id(&'a str),
}

/// The resolved session context, upstream's `SessionContext`.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionContext {
    /// The compaction-aware message list along the root-to-leaf path.
    pub messages: Vec<AgentMessage>,
    /// The latest thinking level along the path, `"off"` by default.
    pub thinking_level: String,
    /// The latest model along the path, when any carried one.
    pub model: Option<SessionModel>,
}

/// The model reference a session context carries, upstream's
/// `{ provider, modelId } | null`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionModel {
    /// The provider id.
    pub provider: String,
    /// The model id.
    pub model_id: String,
}

/// The leaf-path walk, upstream's `buildSessionPath`: from the selected leaf
/// (or the newest entry when the leaf is the upstream-absent form) back to
/// root, reversed.
#[must_use]
pub fn build_session_path<'a>(
    entries: &'a [FileEntry],
    leaf_id: LeafId<'_>,
    by_id: &ByIdIndex,
) -> Vec<&'a FileEntry> {
    let leaf = match leaf_id {
        LeafId::None => return Vec::new(),
        LeafId::Id(id) => by_id.get(id).and_then(|index| entries.get(*index)),
        LeafId::Default => entries.last(),
    };
    let Some(leaf) = leaf else {
        return Vec::new();
    };

    let mut path: Vec<&FileEntry> = Vec::new();
    let mut current: Option<&FileEntry> = Some(leaf);
    while let Some(entry) = current {
        path.push(entry);
        current = entry
            .entry_parent_id()
            .and_then(|parent| by_id.get(parent))
            .and_then(|index| entries.get(*index));
    }
    path.reverse();
    path
}

/// The compaction-aware, active session entry list, upstream's exported
/// `buildContextEntries`.
///
/// It follows the selected leaf path; when the path contains compaction
/// entries the latest compaction is represented by the compaction entry
/// itself, followed by the kept entries starting at `firstKeptEntryId` and
/// all entries after it — older summarized entries are omitted.
#[must_use]
pub fn build_context_entries<'a>(
    entries: &'a [FileEntry],
    leaf_id: LeafId<'_>,
    by_id: &ByIdIndex,
) -> Vec<&'a FileEntry> {
    let path = build_session_path(entries, leaf_id, by_id);
    let Some(compaction) = path.iter().rev().find(|entry| {
        matches!(
            entry,
            FileEntry::Entry(SessionEntry::Compaction(CompactionEntry { .. }))
        )
    }) else {
        return path;
    };
    // An id-less compaction cannot locate itself in the path; upstream's
    // `findIndex` over `undefined` ids cannot either (it would match an
    // arbitrary id-less entry, which the typed port declines to do).
    let Some(compaction_id) = compaction.entry_id().map(str::to_owned) else {
        return path;
    };
    let Some(compaction_index) = path
        .iter()
        .position(|entry| entry.entry_id() == Some(compaction_id.as_str()))
    else {
        return path;
    };
    let first_kept = match compaction {
        FileEntry::Entry(SessionEntry::Compaction(compaction_entry)) => {
            compaction_entry.first_kept_entry_id.clone()
        }
        _ => return path,
    };

    let mut context_entries: Vec<&FileEntry> = vec![compaction];
    let mut found_first_kept = false;
    for entry in &path[..compaction_index] {
        if first_kept
            .as_deref()
            .is_some_and(|kept| entry.entry_id() == Some(kept))
        {
            found_first_kept = true;
        }
        if found_first_kept {
            context_entries.push(entry);
        }
    }
    context_entries.extend(path[compaction_index + 1..].iter().copied());
    context_entries
}

/// The path's thinking level and model settings, upstream's
/// `getSessionContextSettings`.
fn get_session_context_settings(path: &[&FileEntry]) -> (String, Option<SessionModel>) {
    let mut thinking_level = "off".to_owned();
    let mut model: Option<SessionModel> = None;

    for entry in path {
        match entry {
            FileEntry::Entry(SessionEntry::ThinkingLevelChange(change)) => {
                thinking_level.clone_from(&change.thinking_level);
            }
            FileEntry::Entry(SessionEntry::ModelChange(change)) => {
                model = Some(SessionModel {
                    provider: change.provider.clone(),
                    model_id: change.model_id.clone(),
                });
            }
            FileEntry::Entry(SessionEntry::Message(message)) => {
                if let Some(AgentMessage::Standard(Message::Assistant(assistant))) =
                    &message.message
                {
                    model = Some(SessionModel {
                        provider: assistant.provider.0.clone(),
                        model_id: assistant.model.clone(),
                    });
                }
            }
            _ => {}
        }
    }

    (thinking_level, model)
}

/// Project one selected session entry into LLM/runtime messages, upstream's
/// exported `sessionEntryToContextMessages`. Plain custom entries are
/// display/state entries and do not participate in the context.
#[must_use]
pub fn session_entry_to_context_messages(entry: &FileEntry) -> Vec<AgentMessage> {
    let FileEntry::Entry(entry) = entry else {
        return Vec::new();
    };
    typed_entry_to_context_messages(entry)
}

/// The typed-union projection [`session_entry_to_context_messages`] runs for
/// `FileEntry::Entry` values.
///
/// Borrowed so entry scans (compaction cut points, turn-start searches) do
/// not clone the whole entry per probe.
#[must_use]
pub fn typed_entry_to_context_messages(entry: &SessionEntry) -> Vec<AgentMessage> {
    match entry {
        SessionEntry::Message(message_entry) => message_entry
            .message
            .clone()
            .map(|message| vec![message])
            .unwrap_or_default(),
        SessionEntry::CustomMessage(custom) => {
            vec![AgentMessage::Custom(create_custom_message(
                &custom.custom_type,
                custom
                    .content
                    .clone()
                    .unwrap_or(pi_ai::types::UserContent::Blocks(Vec::new())),
                custom.display,
                custom.details.clone(),
                &custom.base.timestamp,
            ))]
        }
        SessionEntry::BranchSummary(branch) if !branch.summary.is_empty() => {
            vec![AgentMessage::Custom(create_branch_summary_message(
                &branch.summary,
                &branch.from_id,
                &branch.base.timestamp,
            ))]
        }
        SessionEntry::Compaction(compaction) => {
            vec![AgentMessage::Custom(create_compaction_summary_message(
                &compaction.summary,
                compaction.tokens_before,
                &compaction.base.timestamp,
            ))]
        }
        _ => Vec::new(),
    }
}

/// Build the session context from entries using tree traversal, upstream's
/// exported `buildSessionContext`.
///
/// With a leaf id, walks from that entry to root; handles compaction and
/// branch summaries along the path.
#[must_use]
pub fn build_session_context(
    entries: &[FileEntry],
    leaf_id: LeafId<'_>,
    by_id: &ByIdIndex,
) -> SessionContext {
    let path = build_session_path(entries, leaf_id, by_id);
    let (thinking_level, model) = get_session_context_settings(&path);
    let messages = build_context_entries(entries, leaf_id, by_id)
        .into_iter()
        .flat_map(session_entry_to_context_messages)
        .collect();
    SessionContext {
        messages,
        thinking_level,
        model,
    }
}

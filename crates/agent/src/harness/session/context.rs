//! The session-to-model-context projection, ported from upstream
//! `src/harness/session/context.ts`: the compaction-checkpoint slice, the
//! entry-to-message mapping, and the custom-entry projectors.

use std::collections::BTreeMap;

use crate::harness::context::Context;
use crate::harness::session::types::{
    CompactionEntryBody, Entry, EntryProjector, EntryType, SessionError,
};
use crate::types::{AgentMessage, CustomAgentMessage};

/// The context-build options, upstream's `SessionContextBuildOptions`.
#[derive(Clone, Debug, Default)]
pub struct SessionContextBuildOptions {
    /// The custom-entry projectors, by custom type, upstream's
    /// `entryProjectors?`.
    pub entry_projectors: Option<BTreeMap<String, EntryProjector>>,
}

/// Trims a branch path to the latest compaction checkpoint plus the entries
/// after it, upstream's `buildContextEntries`.
#[must_use]
pub fn build_context_entries(path_entries: &[Entry]) -> Vec<Entry> {
    let mut checkpoint: Option<usize> = None;
    for (index, entry) in path_entries.iter().enumerate().rev() {
        if entry.entry_type() == EntryType::Compaction {
            checkpoint = Some(index);
            break;
        }
    }
    checkpoint.map_or_else(
        || path_entries.to_vec(),
        |index| {
            let mut entries = Vec::with_capacity(path_entries.len() - index);
            entries.push(path_entries[index].clone());
            entries.extend(path_entries[index + 1..].iter().cloned());
            entries
        },
    )
}

/// Whether a message contributes to the model context, upstream's
/// `isContextMessage`: assistant responses that failed, aborted, or went
/// deferred do not.
const fn is_context_message(message: &AgentMessage) -> bool {
    let AgentMessage::Standard(message) = message else {
        return true;
    };
    let pi_ai::types::Message::Assistant(assistant) = message else {
        return true;
    };
    !matches!(
        assistant.stop_reason,
        pi_ai::types::StopReason::Error
            | pi_ai::types::StopReason::Aborted
            | pi_ai::types::StopReason::Deferred
    )
}

/// A custom-role context message, the object shape
/// `createCompactionSummaryMessage`/`createBranchSummaryMessage` produce
/// upstream — a role-discriminated message the model context carries as-is.
fn custom_role_message(
    role: &str,
    timestamp: i64,
    data: serde_json::Map<String, serde_json::Value>,
) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage {
        role: role.to_owned(),
        timestamp,
        data,
    })
}

/// Maps one entry to its context messages, upstream's
/// `sessionEntryToContextMessages`.
#[must_use]
pub fn session_entry_to_context_messages(entry: &Entry) -> Vec<AgentMessage> {
    match entry {
        Entry::Message { body, .. } => {
            if is_context_message(&body.message) {
                vec![body.message.clone()]
            } else {
                Vec::new()
            }
        }
        Entry::Compaction { body, .. } => {
            let CompactionEntryBody {
                summary,
                retained_tail,
                tokens_before,
                ..
            } = body;
            let mut data = serde_json::Map::new();
            data.insert("summary".to_owned(), summary.clone().into());
            data.insert("tokensBefore".to_owned(), (*tokens_before).into());
            let mut messages = vec![custom_role_message(
                "compactionSummary",
                entry.timestamp(),
                data,
            )];
            messages.extend(
                retained_tail
                    .iter()
                    .filter(|message| is_context_message(message))
                    .cloned(),
            );
            messages
        }
        Entry::BranchSummary { body, .. } => {
            let mut data = serde_json::Map::new();
            data.insert("summary".to_owned(), body.summary.clone().into());
            data.insert(
                "fromId".to_owned(),
                body.from_id
                    .clone()
                    .map_or(serde_json::Value::Null, serde_json::Value::String),
            );
            if body.summary.is_empty() {
                Vec::new()
            } else {
                vec![custom_role_message(
                    "branchSummary",
                    entry.timestamp(),
                    data,
                )]
            }
        }
        Entry::Custom { .. } => Vec::new(),
    }
}

/// Builds the model context from a branch path, upstream's
/// `buildSessionContext`.
///
/// # Errors
/// A custom entry projector's failure, propagated.
pub async fn build_session_context(
    path_entries: &[Entry],
    options: Option<&SessionContextBuildOptions>,
    context: &Context,
) -> Result<Vec<AgentMessage>, SessionError> {
    let default_options = SessionContextBuildOptions::default();
    let options = options.unwrap_or(&default_options);
    let mut messages: Vec<AgentMessage> = Vec::new();
    for entry in &build_context_entries(path_entries) {
        match entry.entry_type() {
            EntryType::Custom => {
                let Some(custom_type) = entry.custom_type() else {
                    continue;
                };
                let Some(projector) = options
                    .entry_projectors
                    .as_ref()
                    .and_then(|projectors| projectors.get(custom_type))
                else {
                    continue;
                };
                if let Some(projected) = (projector.0)(entry, context).await? {
                    messages.extend(projected);
                }
            }
            _ => messages.extend(session_entry_to_context_messages(entry)),
        }
    }
    Ok(messages)
}

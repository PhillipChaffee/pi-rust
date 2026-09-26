//! The session-to-model-context builder, ported from upstream
//! `src/harness/session/context.ts`.
//!
//! The map's harness-foundations child landed the type surface this module's
//! signatures reference; the session-layer child owns `session/context.ts`
//! and extends this module with its suite (`session-context.test.ts`,
//! recorded on that ticket). The compaction-segment selection and the
//! context-message filters restate 1:1.

use std::collections::BTreeMap;

use pi_ai::types::Message;
use pi_ai::types::StopReason;

use crate::harness::context::Context;
use crate::harness::messages::Timestamp;
use crate::harness::messages::create_branch_summary_message;
use crate::harness::messages::create_compaction_summary_message;
use crate::harness::session::types::Entry;
use crate::harness::session::types::EntryProjector;
use crate::harness::session::types::EntryType;
use crate::types::AgentMessage;
use crate::types::CustomAgentMessage;

/// The custom-entry projectors a context build consults, upstream's
/// `SessionContextBuildOptions`.
#[derive(Clone, Debug, Default)]
pub struct SessionContextBuildOptions {
    /// The projectors, by custom type. Absent custom types drop.
    pub entry_projectors: BTreeMap<String, EntryProjector>,
}

/// Narrows a branch path to the entries the model context holds, upstream's
/// `buildContextEntries`: the last compaction entry plus everything after
/// it, or the whole path when no compaction entry sits on it.
#[must_use]
pub fn build_context_entries(path_entries: &[Entry]) -> Vec<Entry> {
    let mut compaction_index = None;
    for index in (0..path_entries.len()).rev() {
        if path_entries[index].entry_type() == EntryType::Compaction {
            compaction_index = Some(index);
            break;
        }
    }
    match compaction_index {
        None => path_entries.to_vec(),
        Some(compaction_index) => {
            let mut entries = Vec::with_capacity(path_entries.len() - compaction_index);
            entries.push(path_entries[compaction_index].clone());
            entries.extend(path_entries[compaction_index + 1..].iter().cloned());
            entries
        }
    }
}

/// Whether a transcript message contributes to the model context, upstream's
/// `isContextMessage`: every non-assistant message, and assistant messages
/// that did not end in `error`, `aborted`, or `deferred`.
#[must_use]
pub fn is_context_message(message: &AgentMessage) -> bool {
    match message {
        AgentMessage::Standard(Message::Assistant(assistant)) => !matches!(
            assistant.stop_reason,
            StopReason::Error | StopReason::Aborted | StopReason::Deferred
        ),
        AgentMessage::Standard(_) | AgentMessage::Custom(_) => true,
    }
}

/// Converts one entry to the model-context messages it holds, upstream's
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
        Entry::Compaction {
            body, timestamp, ..
        } => {
            let mut messages = vec![summary_agent_message_compaction(
                &body.summary,
                body.tokens_before,
                *timestamp,
            )];
            messages.extend(
                body.retained_tail
                    .iter()
                    .filter(|message| is_context_message(message))
                    .cloned(),
            );
            messages
        }
        Entry::BranchSummary {
            body, timestamp, ..
        } => {
            if body.summary.is_empty() {
                Vec::new()
            } else {
                vec![summary_agent_message_branch(&body.summary, body.from_id.clone(), *timestamp)]
            }
        }
        Entry::Custom { .. } => Vec::new(),
    }
}

fn summary_agent_message_compaction(summary: &str, tokens_before: i64, timestamp: i64) -> AgentMessage {
    let message = create_compaction_summary_message(summary, tokens_before, Timestamp::Millis(timestamp));
    custom_agent_message_from_wire(&message)
}

fn summary_agent_message_branch(summary: &str, from_id: Option<String>, timestamp: i64) -> AgentMessage {
    let message = create_branch_summary_message(summary, from_id, Timestamp::Millis(timestamp));
    custom_agent_message_from_wire(&message)
}

/// Wraps one landed summary-message constructor's output as the custom-role
/// [`AgentMessage`] the transcript carries. The constructor's serde shape is
/// the custom message's wire shape; deserializing through
/// [`CustomAgentMessage`] extracts `role`/`timestamp` first-class and keeps
/// every other field as owned JSON data.
fn custom_agent_message_from_wire<T: serde::Serialize + serde::de::DeserializeOwned>(
    message: &T,
) -> AgentMessage {
    let wire = serde_json::to_value(message)
        .unwrap_or_else(|error| unreachable!("summary message serialization failed: {error}"));
    let custom: CustomAgentMessage = serde_json::from_value(wire)
        .unwrap_or_else(|error| unreachable!("summary message deserialization failed: {error}"));
    AgentMessage::Custom(custom)
}

/// Builds the model context one branch path holds, upstream's
/// `buildSessionContext`: non-custom entries convert through
/// [`session_entry_to_context_messages`], custom entries consult their
/// projector.
pub async fn build_session_context(
    path_entries: &[Entry],
    options: Option<&SessionContextBuildOptions>,
    context: &Context,
) -> Vec<AgentMessage> {
    let default_options = SessionContextBuildOptions::default();
    let options = options.unwrap_or(&default_options);
    let mut messages: Vec<AgentMessage> = Vec::new();
    for entry in build_context_entries(path_entries) {
        let Entry::Custom { body, .. } = &entry else {
            messages.extend(session_entry_to_context_messages(&entry));
            continue;
        };
        if let Some(projector) = options.entry_projectors.get(&body.custom_type) {
            if let Some(projected) = (projector.0)(&entry, context).await {
                messages.extend(projected);
            }
        }
    }
    messages
}
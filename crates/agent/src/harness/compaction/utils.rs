//! The compaction belt, ported from upstream
//! `src/harness/compaction/utils.ts`: the file-operation accumulator, the
//! file-list computation, and the conversation serializer the summary
//! prompts embed.

use std::collections::BTreeSet;

use pi_ai::types::{AssistantBlock, Message};
use serde_json::Value as JsonValue;

use crate::types::AgentMessage;

use super::types::{CompactionDetails, FileOperations};

/// The runtime file-operation accumulator, upstream's `FileOperations` in
/// `compaction/utils.ts` — three path sets.
///
/// The durable [`FileOperations`] in `types.rs` is the sorted-vector wire
/// form the summary task persists; `durable_file_operations` converts.
#[derive(Clone, Debug, Default)]
pub struct FileOpsAccumulator {
    /// Files read but not necessarily modified.
    pub read: FileOperationsSet,
    /// Files written by full-file write operations.
    pub written: FileOperationsSet,
    /// Files modified by edit operations.
    pub edited: FileOperationsSet,
}

/// The set view one accumulator side carries, upstream's `Set<string>`.
pub type FileOperationsSet = BTreeSet<String>;

/// Builds an empty file-operation accumulator, upstream's `createFileOps`.
#[must_use]
pub fn create_file_ops_accumulator() -> FileOpsAccumulator {
    FileOpsAccumulator::default()
}

/// Add file operations from assistant tool calls to an accumulator,
/// upstream's `extractFileOpsFromMessage`.
///
/// The `read`/`write`/`edit` tool calls carry the touched path in their
/// `path` argument; every other tool name and every non-assistant message
/// contributes nothing.
pub fn extract_file_ops_from_message(message: &AgentMessage, file_ops: &mut FileOpsAccumulator) {
    let AgentMessage::Standard(Message::Assistant(assistant)) = message else {
        return;
    };
    for block in &assistant.content {
        let AssistantBlock::ToolCall(call) = block else {
            continue;
        };
        let Some(path) = call.arguments.get("path").and_then(JsonValue::as_str) else {
            continue;
        };
        // Upstream's `if (!path)` skips the empty string like any missing
        // path.
        if path.is_empty() {
            continue;
        }
        match call.name.as_str() {
            "read" => {
                file_ops.read.insert(path.to_owned());
            }
            "write" => {
                file_ops.written.insert(path.to_owned());
            }
            "edit" => {
                file_ops.edited.insert(path.to_owned());
            }
            _ => {}
        }
    }
}

/// Converts an accumulator into the durable sorted-vector wire form,
/// upstream's `durableFileOperations` in the runtime's drive/structural
/// module.
///
/// A `BTreeSet` iterates sorted, the sort the wire form carries.
#[must_use]
pub fn durable_file_operations(file_ops: &FileOpsAccumulator) -> FileOperations {
    FileOperations {
        read: file_ops.read.iter().cloned().collect(),
        written: file_ops.written.iter().cloned().collect(),
        edited: file_ops.edited.iter().cloned().collect(),
    }
}

/// Compute sorted read-only and modified file lists from accumulated
/// operations, upstream's `computeFileLists`.
///
/// The port takes the durable shape: a compaction preparation carries
/// [`FileOperations`], and the generated summaries re-derive the lists
/// from it. A modified file (written or edited) never appears in the
/// read-only list; duplicates collapse and both lists sort.
#[must_use]
pub fn compute_file_lists(file_ops: &FileOperations) -> CompactionDetails {
    let modified: FileOperationsSet = file_ops
        .written
        .iter()
        .chain(file_ops.edited.iter())
        .cloned()
        .collect();
    let read_only: FileOperationsSet = file_ops
        .read
        .iter()
        .filter(|file| !modified.contains(*file))
        .cloned()
        .collect();
    CompactionDetails {
        read_files: read_only.into_iter().collect(),
        modified_files: modified.into_iter().collect(),
    }
}

/// Format file lists as summary metadata tags, upstream's
/// `formatFileOperations`.
///
/// One `<read-files>` and one `<modified-files>` section, blank-line
/// separated, prefixed by a blank line for appending to a summary.
#[must_use]
pub fn format_file_operations(read_files: &[String], modified_files: &[String]) -> String {
    let mut sections: Vec<String> = Vec::new();
    if !read_files.is_empty() {
        sections.push(format!(
            "<read-files>\n{}\n</read-files>",
            read_files.join("\n")
        ));
    }
    if !modified_files.is_empty() {
        sections.push(format!(
            "<modified-files>\n{}\n</modified-files>",
            modified_files.join("\n")
        ));
    }
    if sections.is_empty() {
        return String::new();
    }
    format!("\n\n{}", sections.join("\n\n"))
}

const TOOL_RESULT_MAX_CHARS: usize = 2000;

/// Upstream's `safeJsonStringify`: the JSON text, with a stand-in when
/// serialization fails. Upstream's `undefined` branch cannot fire — a
/// `serde_json` value always serializes or fails outright.
pub(crate) fn safe_json_stringify<T: serde::Serialize + ?Sized>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "[unserializable]".to_owned())
}

/// The tail-truncation the tool-result serializer applies, upstream's
/// `truncateForSummary`.
///
/// Lengths count characters, upstream's `.length` UTF-16 units restated as
/// chars (the suites' fixtures are ASCII, where they agree).
fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_owned();
    }
    let truncated_chars = char_count - max_chars;
    let kept: String = text.chars().take(max_chars).collect();
    format!("{kept}\n\n[... {truncated_chars} more characters truncated]")
}

/// Serialize LLM messages to plain text for summarization prompts,
/// upstream's `serializeConversation`.
///
/// One `[User]`/`[Assistant thinking]`/`[Assistant]`/`[Assistant tool
/// calls]`/`[Tool result]` paragraph per contributing message,
/// blank-line separated.
#[must_use]
pub fn serialize_conversation(messages: &[Message]) -> String {
    let mut parts: Vec<String> = Vec::new();

    for message in messages {
        match message {
            Message::User(user) => {
                let content = pi_ai::utils::text::content_text_with(&user.content, "");
                if !content.is_empty() {
                    parts.push(format!("[User]: {content}"));
                }
            }
            Message::Assistant(assistant) => {
                let mut thinking_parts: Vec<&str> = Vec::new();
                let mut tool_calls: Vec<String> = Vec::new();

                for block in &assistant.content {
                    match block {
                        AssistantBlock::Thinking(content) => {
                            thinking_parts.push(&content.thinking);
                        }
                        AssistantBlock::ToolCall(call) => {
                            let arguments = call
                                .arguments
                                .iter()
                                .map(|(key, value)| format!("{key}={}", safe_json_stringify(value)))
                                .collect::<Vec<_>>()
                                .join(", ");
                            tool_calls.push(format!("{}({arguments})", call.name));
                        }
                        AssistantBlock::Text(_) => {}
                    }
                }

                if !thinking_parts.is_empty() {
                    parts.push(format!(
                        "[Assistant thinking]: {}",
                        thinking_parts.join("\n")
                    ));
                }
                if assistant
                    .content
                    .iter()
                    .any(|block| matches!(block, AssistantBlock::Text(_)))
                {
                    parts.push(format!(
                        "[Assistant]: {}",
                        pi_ai::utils::text::content_text(&assistant.content.as_slice())
                    ));
                }
                if !tool_calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]: {}", tool_calls.join("; ")));
                }
            }
            Message::ToolResult(result) => {
                let content = pi_ai::utils::text::content_text_with(&result.content.as_slice(), "");
                if !content.is_empty() {
                    parts.push(format!(
                        "[Tool result]: {}",
                        truncate_for_summary(&content, TOOL_RESULT_MAX_CHARS)
                    ));
                }
            }
        }
    }

    parts.join("\n\n")
}

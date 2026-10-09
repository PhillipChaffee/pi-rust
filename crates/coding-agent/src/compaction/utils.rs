//! Shared utilities for compaction and branch summarization, upstream
//! `src/core/compaction/utils.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::collections::HashSet;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{AssistantBlock, Message};
use pi_ai::utils::text::{content_text, content_text_with};

// ============================================================================
// File Operation Tracking
// ============================================================================

/// File paths touched by tool calls in the summarized range, upstream's
/// `FileOperations`. The sets dedupe; ordering never survives
/// [`compute_file_lists`], which sorts both outputs.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FileOperations {
    /// Paths opened by `read` tool calls.
    pub read: HashSet<String>,
    /// Paths written by full-file `write` tool calls.
    pub written: HashSet<String>,
    /// Paths modified by `edit` tool calls.
    pub edited: HashSet<String>,
}

/// Create an empty file-operation accumulator, upstream's `createFileOps`.
#[must_use]
pub fn create_file_ops() -> FileOperations {
    FileOperations::default()
}

/// Extract file operations from tool calls in an assistant message, upstream's
/// `extractFileOpsFromMessage`.
///
/// Only the `read`/`write`/`edit` tools with a string `path` argument
/// contribute; every other message is ignored.
pub fn extract_file_ops_from_message(message: &AgentMessage, file_ops: &mut FileOperations) {
    let AgentMessage::Standard(Message::Assistant(assistant)) = message else {
        return;
    };
    for block in &assistant.content {
        let AssistantBlock::ToolCall(call) = block else {
            continue;
        };
        let Some(pi_ai::types::JsonValue::String(path)) = call.arguments.get("path") else {
            continue;
        };
        match call.name.as_str() {
            "read" => {
                file_ops.read.insert(path.clone());
            }
            "write" => {
                file_ops.written.insert(path.clone());
            }
            "edit" => {
                file_ops.edited.insert(path.clone());
            }
            _ => {}
        }
    }
}

/// The sorted file lists a summary records, upstream's
/// `{ readFiles, modifiedFiles }`. `modifiedFiles` merges the edited and
/// written sets; `readFiles` keeps only files that were never modified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileLists {
    /// Files only read, not modified.
    pub read_files: Vec<String>,
    /// Files modified by write or edit calls.
    pub modified_files: Vec<String>,
}

/// Compute final file lists from file operations, upstream's `computeFileLists`.
#[must_use]
pub fn compute_file_lists(file_ops: &FileOperations) -> FileLists {
    let modified: HashSet<&String> = file_ops.edited.union(&file_ops.written).collect();
    let read_files = js_sort(
        file_ops
            .read
            .iter()
            .filter(|file| !modified.contains(*file))
            .cloned()
            .collect(),
    );
    let modified_files = js_sort(modified.into_iter().cloned().collect());
    FileLists {
        read_files,
        modified_files,
    }
}

/// Format file lists as the summary's XML tags, upstream's
/// `formatFileOperations`. The result is either empty or begins with the
/// blank line that separates it from the summary body.
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

/// `Array.prototype.sort()`'s default order, UTF-16 code-unit comparison.
/// Restated here because file lists land in summary text the sessions store.
fn js_sort(mut values: Vec<String>) -> Vec<String> {
    values.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    values
}

// ============================================================================
// Message Serialization
// ============================================================================

/// Maximum characters for a tool result in serialized summaries, upstream's
/// `TOOL_RESULT_MAX_CHARS`.
const TOOL_RESULT_MAX_CHARS: usize = 2000;

/// Truncate text to a maximum character length for summarization, upstream's
/// `truncateForSummary`. Keeps the beginning and appends a truncation marker
/// naming how many characters were dropped.
///
/// Lengths count characters, upstream's `.length` UTF-16 units restated as
/// chars (the suites' fixtures are ASCII, where they agree).
fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_owned();
    }
    let truncated_chars = char_count - max_chars;
    let prefix: String = text.chars().take(max_chars).collect();
    format!("{prefix}\n\n[... {truncated_chars} more characters truncated]")
}

/// One serialized tool call's arguments, upstream's
/// `` `${k}=${JSON.stringify(v)}` `` pairs. `serde_json`'s `preserve_order`
/// keeps insertion order like JS objects do.
fn tool_call_args_text(arguments: &serde_json::Map<String, serde_json::Value>) -> String {
    arguments
        .iter()
        .map(|(key, value)| format!("{key}={}", safe_json_stringify(value)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `JSON.stringify`, restated. `JsonValue` values never fail to serialize,
/// so the fallback arm is unreachable.
fn safe_json_stringify(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "[unserializable]".to_owned())
}

/// Serialize LLM messages to text for summarization, upstream's
/// `serializeConversation`.
///
/// The flat text prevents the model from treating the request as a
/// conversation to continue. Call [`crate::messages::convert_to_llm`] first to
/// handle the custom message types. Tool results truncate to keep the
/// summarization request within reasonable token budgets; full content is not
/// needed for summarization.
#[must_use]
pub fn serialize_conversation(messages: &[Message]) -> String {
    let mut parts: Vec<String> = Vec::new();

    for msg in messages {
        match msg {
            Message::User(user) => {
                let content = content_text_with(&user.content, "");
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
                            tool_calls.push(format!(
                                "{}({})",
                                call.name,
                                tool_call_args_text(&call.arguments)
                            ));
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
                        content_text(&assistant.content.as_slice())
                    ));
                }
                if !tool_calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]: {}", tool_calls.join("; ")));
                }
            }
            Message::ToolResult(result) => {
                let content = content_text_with(&result.content.as_slice(), "");
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

// ============================================================================
// Summarization System Prompt
// ============================================================================

/// The system prompt every summarization request carries, upstream's
/// `SUMMARIZATION_SYSTEM_PROMPT`.
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

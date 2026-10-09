//! The custom-message constructors and LLM converters the session projection
//! and compaction need, from upstream `src/core/messages.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream declares the four coding-agent custom roles
//! (`bashExecution`, `custom`, `branchSummary`, `compactionSummary`) via
//! declaration merging; the substrate carries them as owned-JSON
//! [`CustomAgentMessage`] values, so each constructor builds that shape with
//! the wire field names verbatim. `convertToLlm` and `bashExecutionToText`
//! read those same owned-JSON shapes back out; they ride this module because
//! compaction (#123) consumes them and AgentSession (#125) follows.

use std::fmt::Write as _;

use serde_json::{Map, Value as JsonValue};

use pi_agent_core::harness::session::jsonl::codec::parse_iso8601_millis;
use pi_agent_core::types::{AgentMessage, CustomAgentMessage};
use pi_ai::types::{Message, UserContent, UserMessage};

/// The prefix a compaction summary carries into the LLM context, upstream's
/// `COMPACTION_SUMMARY_PREFIX`.
pub const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";

/// The suffix a compaction summary carries into the LLM context, upstream's
/// `COMPACTION_SUMMARY_SUFFIX`.
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";

/// The prefix a branch summary carries into the LLM context, upstream's
/// `BRANCH_SUMMARY_PREFIX`.
pub const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";

/// The suffix a branch summary carries into the LLM context, upstream's
/// `BRANCH_SUMMARY_SUFFIX`.
pub const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";

/// Millis for an entry timestamp, upstream's `new Date(timestamp).getTime()`.
///
/// An unparseable timestamp degrades to `0` instead of the NaN upstream
/// would carry: the custom-message wire contract requires a number, and a
/// NaN would fail the substrate's custom-message deserialization on reload.
fn timestamp_millis(timestamp: &str) -> i64 {
    parse_iso8601_millis(timestamp).unwrap_or(0)
}

/// A branch-summary message for the LLM context, upstream's
/// `createBranchSummaryMessage` (wire role `branchSummary`).
#[must_use]
pub fn create_branch_summary_message(
    summary: &str,
    from_id: &str,
    timestamp: &str,
) -> CustomAgentMessage {
    let mut data = Map::new();
    data.insert("summary".to_owned(), JsonValue::String(summary.to_owned()));
    data.insert("fromId".to_owned(), JsonValue::String(from_id.to_owned()));
    CustomAgentMessage {
        role: "branchSummary".to_owned(),
        timestamp: timestamp_millis(timestamp),
        data,
    }
}

/// A compaction-summary message for the LLM context, upstream's
/// `createCompactionSummaryMessage` (wire role `compactionSummary`).
#[must_use]
pub fn create_compaction_summary_message(
    summary: &str,
    tokens_before: i64,
    timestamp: &str,
) -> CustomAgentMessage {
    let mut data = Map::new();
    data.insert("summary".to_owned(), JsonValue::String(summary.to_owned()));
    data.insert("tokensBefore".to_owned(), JsonValue::from(tokens_before));
    CustomAgentMessage {
        role: "compactionSummary".to_owned(),
        timestamp: timestamp_millis(timestamp),
        data,
    }
}

/// A custom message converted to the agent-message format, upstream's
/// `createCustomMessage` (wire role `custom`).
#[must_use]
pub fn create_custom_message(
    custom_type: &str,
    content: UserContent,
    display: bool,
    details: Option<JsonValue>,
    timestamp: &str,
) -> CustomAgentMessage {
    let mut data = Map::new();
    data.insert(
        "customType".to_owned(),
        JsonValue::String(custom_type.to_owned()),
    );
    data.insert(
        "content".to_owned(),
        serde_json::to_value(content).unwrap_or_default(),
    );
    data.insert("display".to_owned(), JsonValue::from(display));
    if let Some(details) = details {
        data.insert("details".to_owned(), details);
    }
    CustomAgentMessage {
        role: "custom".to_owned(),
        timestamp: timestamp_millis(timestamp),
        data,
    }
}

/// Convert a bash-execution message to user-message text for the LLM context,
/// upstream's `bashExecutionToText`.
///
/// The message is the owned-JSON `bashExecution` shape: `command`, `output`,
/// `exitCode`, `cancelled`, `truncated`, and the optional `fullOutputPath`.
#[must_use]
pub fn bash_execution_to_text(message: &CustomAgentMessage) -> String {
    let field = |name: &str| -> String {
        message
            .field(name)
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let mut text = format!("Ran `{}`\n", field("command"));
    let output = field("output");
    if output.is_empty() {
        text.push_str("(no output)");
    } else {
        let _ = write!(text, "```\n{output}\n```");
    }
    if message.field("cancelled").and_then(JsonValue::as_bool) == Some(true) {
        text.push_str("\n\n(command cancelled)");
    } else {
        let exit_code = message.field("exitCode").and_then(JsonValue::as_i64);
        if let Some(exit_code) = exit_code.filter(|code| *code != 0) {
            let _ = write!(text, "\n\nCommand exited with code {exit_code}");
        }
    }
    if message.field("truncated").and_then(JsonValue::as_bool) == Some(true) {
        let full_output_path = field("fullOutputPath");
        if !full_output_path.is_empty() {
            let _ = write!(
                text,
                "\n\n[Output truncated. Full output: {full_output_path}]"
            );
        }
    }
    text
}

/// Transform agent messages (including the coding-agent custom types) to
/// LLM-compatible messages, upstream's `convertToLlm`.
///
/// Used by the Agent's `transformToLlm` hook (prompt calls and queued
/// messages), compaction's summary generation, and custom extensions and
/// tools. A bash execution with `excludeFromContext` (the `!!` prefix) drops
/// out of the context entirely.
#[must_use]
pub fn convert_to_llm(messages: &[AgentMessage]) -> Vec<Message> {
    messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Standard(standard) => Some(standard.clone()),
            AgentMessage::Custom(custom) => match custom.role.as_str() {
                "bashExecution" => {
                    if custom
                        .field("excludeFromContext")
                        .and_then(JsonValue::as_bool)
                        == Some(true)
                    {
                        return None;
                    }
                    Some(Message::User(UserMessage {
                        content: UserContent::Text(bash_execution_to_text(custom)),
                        timestamp: custom.timestamp,
                    }))
                }
                "custom" => Some(Message::User(UserMessage {
                    content: match custom.field("content") {
                        Some(JsonValue::String(text)) => {
                            UserContent::Blocks(vec![pi_ai::types::UserBlock::Text(
                                pi_ai::types::TextContent {
                                    text: text.clone(),
                                    text_signature: None,
                                },
                            )])
                        }
                        Some(content) => serde_json::from_value(content.clone())
                            .unwrap_or_else(|_| UserContent::Blocks(Vec::new())),
                        None => UserContent::Blocks(Vec::new()),
                    },
                    timestamp: custom.timestamp,
                })),
                "branchSummary" => Some(Message::User(UserMessage {
                    content: UserContent::Text(format!(
                        "{BRANCH_SUMMARY_PREFIX}{}{BRANCH_SUMMARY_SUFFIX}",
                        custom
                            .field("summary")
                            .and_then(JsonValue::as_str)
                            .unwrap_or_default()
                    )),
                    timestamp: custom.timestamp,
                })),
                "compactionSummary" => Some(Message::User(UserMessage {
                    content: UserContent::Text(format!(
                        "{COMPACTION_SUMMARY_PREFIX}{}{COMPACTION_SUMMARY_SUFFIX}",
                        custom
                            .field("summary")
                            .and_then(JsonValue::as_str)
                            .unwrap_or_default()
                    )),
                    timestamp: custom.timestamp,
                })),
                _ => None,
            },
        })
        .collect()
}

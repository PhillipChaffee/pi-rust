//! The custom-message constructors the session projection needs, from
//! upstream `src/core/messages.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream declares the four coding-agent custom roles
//! (`bashExecution`, `custom`, `branchSummary`, `compactionSummary`) via
//! declaration merging; the substrate carries them as owned-JSON
//! [`CustomAgentMessage`] values, so each constructor builds that shape with
//! the wire field names verbatim. The rest of `messages.ts` — the summary
//! prefixes, `bashExecutionToText`, and `convertToLlm` — rides the
//! compaction and AgentSession slices, which consume it.

use serde_json::{Map, Value as JsonValue};

use pi_agent_core::harness::session::jsonl::codec::parse_iso8601_millis;
use pi_agent_core::types::CustomAgentMessage;
use pi_ai::types::UserContent;

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

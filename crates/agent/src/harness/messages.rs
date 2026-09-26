//! Harness message helpers, ported from upstream `src/harness/messages.ts`.
//!
//! The four custom message shapes (`bashExecution`, `custom`,
//! `branchSummary`, `compactionSummary`) ride the crate's
//! [`AgentMessage::Custom`] variant as owned JSON data (the scaffold
//! child's declaration-merging restatement), so this module carries the
//! typed view over that data: the constructors build the wire objects, the
//! `TryFrom` conversions round-trip them field-for-field, and
//! [`convert_to_llm`] matches on the role discriminators exactly as
//! upstream's switch does.

use std::fmt::Write as _;

use pi_ai::types::{ImageContent, Message, TextContent, UserBlock, UserContent, UserMessage};
use serde::{Deserialize, Serialize};

use crate::types::{AgentMessage, CustomAgentMessage};

/// The text block wrapper around one compaction summary, upstream's
/// `COMPACTION_SUMMARY_PREFIX`.
pub const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";

/// The closing wrapper around one compaction summary, upstream's
/// `COMPACTION_SUMMARY_SUFFIX`.
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";

/// The text block wrapper around one branch summary, upstream's
/// `BRANCH_SUMMARY_PREFIX`.
pub const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";

/// The closing wrapper around one branch summary, upstream's
/// `BRANCH_SUMMARY_SUFFIX`.
pub const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";

/// The content union a custom message carries, upstream's
/// `string | (TextContent | ImageContent)[]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CustomMessageContent {
    /// One text string.
    Text(String),
    /// Full typed content blocks.
    Blocks(Vec<CustomMessageBlock>),
}

/// One content block of a custom message, upstream's
/// `TextContent | ImageContent`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CustomMessageBlock {
    /// A text block.
    Text(TextContent),
    /// An image block.
    Image(ImageContent),
}

/// The `custom` role message, upstream's `CustomMessage<T>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessage {
    /// The role discriminator.
    pub role: String,
    /// The application's custom type discriminator.
    pub custom_type: String,
    /// The message content.
    pub content: CustomMessageContent,
    /// Whether UIs render the message.
    pub display: bool,
    /// Arbitrary application details, upstream's `details?: T`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

/// The `bashExecution` role message, upstream's `BashExecutionMessage`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashExecutionMessage {
    /// The role discriminator.
    pub role: String,
    /// The command that ran.
    pub command: String,
    /// The captured output.
    pub output: String,
    /// The exit code; absent when the command never ran to exit.
    pub exit_code: Option<i32>,
    /// Whether the command was cancelled.
    pub cancelled: bool,
    /// Whether the output was truncated.
    pub truncated: bool,
    /// The file preserving complete output when truncated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
    /// Whether the message is excluded from model context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_from_context: Option<bool>,
}

/// The `branchSummary` role message, upstream's `BranchSummaryMessage`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryMessage {
    /// The role discriminator.
    pub role: String,
    /// The generated summary.
    pub summary: String,
    /// The entry id the summarized branch forked from, when known.
    pub from_id: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

/// The `compactionSummary` role message, upstream's
/// `CompactionSummaryMessage`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSummaryMessage {
    /// The role discriminator.
    pub role: String,
    /// The generated summary.
    pub summary: String,
    /// Estimated context tokens before compaction.
    pub tokens_before: i64,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
}

/// Formats a bash-execution message as model context, upstream's
/// `bashExecutionToText`.
#[must_use]
pub fn bash_execution_to_text(msg: &BashExecutionMessage) -> String {
    let mut text = format!("Ran `{}`\n", msg.command);
    if msg.output.is_empty() {
        text.push_str("(no output)");
    } else {
        let _ = write!(text, "```\n{}\n```", msg.output);
    }
    if msg.cancelled {
        text.push_str("\n\n(command cancelled)");
    } else if msg.exit_code.is_some_and(|exit_code| exit_code != 0) {
        let _ = write!(
            text,
            "\n\nCommand exited with code {}",
            msg.exit_code.unwrap_or_default()
        );
    }
    if msg.truncated
        && let Some(full_output_path) = &msg.full_output_path
    {
        let _ = write!(
            text,
            "\n\n[Output truncated. Full output: {full_output_path}]"
        );
    }
    text
}

/// Builds a branch-summary message, upstream's `createBranchSummaryMessage`.
#[must_use]
pub fn create_branch_summary_message(
    summary: impl Into<String>,
    from_id: Option<String>,
    timestamp: Timestamp,
) -> BranchSummaryMessage {
    BranchSummaryMessage {
        role: "branchSummary".to_owned(),
        summary: summary.into(),
        from_id,
        timestamp: timestamp.millis(),
    }
}

/// Builds a compaction-summary message, upstream's
/// `createCompactionSummaryMessage`.
#[must_use]
pub fn create_compaction_summary_message(
    summary: impl Into<String>,
    tokens_before: i64,
    timestamp: Timestamp,
) -> CompactionSummaryMessage {
    CompactionSummaryMessage {
        role: "compactionSummary".to_owned(),
        summary: summary.into(),
        tokens_before,
        timestamp: timestamp.millis(),
    }
}

/// Builds a custom message, upstream's `createCustomMessage`.
#[must_use]
pub fn create_custom_message(
    custom_type: impl Into<String>,
    content: CustomMessageContent,
    display: bool,
    details: Option<serde_json::Value>,
    timestamp: Timestamp,
) -> CustomMessage {
    CustomMessage {
        role: "custom".to_owned(),
        custom_type: custom_type.into(),
        content,
        display,
        details,
        timestamp: timestamp.millis(),
    }
}

/// The timestamp argument the message constructors take, upstream's
/// `string | number`.
#[derive(Clone, Debug)]
pub enum Timestamp {
    /// Unix epoch milliseconds.
    Millis(i64),
    /// An RFC 2822 or ISO-8601 date string, parsed per JavaScript's
    /// `new Date` on construction.
    Date(String),
}

impl From<i64> for Timestamp {
    fn from(value: i64) -> Self {
        Self::Millis(value)
    }
}

impl Timestamp {
    /// The Unix epoch milliseconds, upstream's
    /// `typeof timestamp === "number" ? timestamp : new Date(timestamp).getTime()`.
    #[must_use]
    pub fn millis(self) -> i64 {
        match self {
            Self::Millis(millis) => millis,
            Self::Date(date) => parse_date_millis(&date),
        }
    }
}

/// Parses one RFC 2822 or ISO-8601 date string to Unix epoch
/// milliseconds, upstream's `new Date(timestamp).getTime()`.
///
/// The supported subset covers the formats the harness round-trips:
/// ISO-8601 dates and date-times with optional fractional seconds and a
/// `Z` or numeric UTC offset. Input the parser cannot read restates
/// JavaScript's `NaN` as `i64::MIN`.
#[must_use]
pub fn parse_date_millis(date: &str) -> i64 {
    let invalid = i64::MIN;
    let parse_int = |slice: Option<&str>| slice.and_then(|slice| slice.parse::<i64>().ok());
    let Some(year) = parse_int(date.get(0..4)) else {
        return invalid;
    };
    let Some(month) = date
        .get(5..7)
        .and_then(|slice| slice.parse::<u32>().ok())
        .filter(|month| (1..=12).contains(month))
    else {
        return invalid;
    };
    let Some(day) = date
        .get(8..10)
        .and_then(|slice| slice.parse::<u32>().ok())
        .filter(|day| (1..=31).contains(day))
    else {
        return invalid;
    };

    let mut hour = 0u32;
    let mut minute = 0u32;
    let mut second = 0u32;
    let mut millis = 0i64;
    let mut offset_ms = 0i64;

    let rest = &date[10..];
    if !rest.is_empty() {
        let Some(rest) = rest.strip_prefix(['T', 't', ' ']) else {
            return invalid;
        };
        let parse_two = |slice: Option<&str>| {
            slice
                .and_then(|slice| slice.get(0..2))
                .and_then(|pair| pair.parse::<u32>().ok())
        };
        let Some(parsed_hour) = parse_two(Some(rest)) else {
            return invalid;
        };
        hour = parsed_hour;
        let after_hour = &rest[2..];
        let after_hour = after_hour.strip_prefix(':').unwrap_or(after_hour);
        if after_hour.len() >= 2 {
            let Some(parsed_minute) = parse_two(Some(after_hour)) else {
                return invalid;
            };
            minute = parsed_minute;
            let after_minute = &after_hour[2..];
            let after_minute = after_minute.strip_prefix(':').unwrap_or(after_minute);
            if !after_minute.is_empty() && after_minute.as_bytes()[0].is_ascii_digit() {
                let Some(parsed_second) = parse_two(Some(after_minute)) else {
                    return invalid;
                };
                second = parsed_second;
                let after_second = &after_minute[2..];
                let (fraction, offset_part) = after_second.strip_prefix('.').map_or_else(
                    || (String::new(), after_second.to_owned()),
                    |fraction| {
                        let digits: String =
                            fraction.chars().take_while(char::is_ascii_digit).collect();
                        (digits.clone(), fraction[digits.len()..].to_owned())
                    },
                );
                if !fraction.is_empty() {
                    let mut scaled = fraction;
                    while scaled.len() < 3 {
                        scaled.push('0');
                    }
                    millis = scaled[..3].parse::<i64>().unwrap_or(0);
                }
                offset_ms = parse_offset(&offset_part).unwrap_or(i64::MIN);
                if offset_ms == i64::MIN
                    && !offset_part.is_empty()
                    && offset_part != "Z"
                    && offset_part != "z"
                {
                    return invalid;
                }
            } else {
                offset_ms = match parse_offset(after_minute) {
                    Ok(offset_ms) => offset_ms,
                    Err(()) => return invalid,
                };
            }
        } else {
            offset_ms = match parse_offset(after_hour) {
                Ok(offset_ms) => offset_ms,
                Err(()) => return invalid,
            };
        }
    }

    epoch_ms_for(year, month, day, hour, minute, second, millis, offset_ms)
}

/// Parses the numeric UTC offset tail (`Z`, `±HH`, `±HH:MM`), upstream's
/// JavaScript offset handling; `Z` and an absent tail are zero.
fn parse_offset(rest: &str) -> Result<i64, ()> {
    if rest.is_empty() || rest == "Z" || rest == "z" {
        return Ok(0);
    }
    let (sign, rest) = match rest.as_bytes().first() {
        Some(b'+') => (1i64, &rest[1..]),
        Some(b'-') => (-1i64, &rest[1..]),
        _ => return Err(()),
    };
    let parse_two = |slice: &str| slice.get(0..2).and_then(|part| part.parse::<i64>().ok());
    let hours = parse_two(rest).ok_or(())?;
    let minutes = rest
        .get(3..5)
        .and_then(|part| part.parse::<i64>().ok())
        .unwrap_or(0);
    Ok(sign * (hours * 3_600_000 + minutes * 60_000))
}

/// Days-from-civil to epoch milliseconds, Howard Hinnant's algorithm, the
/// proleptic-Gregorian arithmetic JavaScript's `Date` carries.
#[expect(
    clippy::too_many_arguments,
    reason = "the calendar inputs are the wire's date shape; bundling them hides it"
)]
fn epoch_ms_for(
    year: i64,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    millis: i64,
    offset_ms: i64,
) -> i64 {
    let y = year - i64::from(month <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (i64::from(month) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    days * 86_400_000
        + i64::from(hour) * 3_600_000
        + i64::from(minute) * 60_000
        + i64::from(second) * 1_000
        + millis
        - offset_ms
}

/// Converts harness messages into LLM messages, upstream's `convertToLlm`.
///
/// Bash executions render through [`bash_execution_to_text`], custom
/// messages pass their content, branch and compaction summaries wrap their
/// text in the summary constants, standard messages pass through, and
/// anything else drops.
#[must_use]
pub fn convert_to_llm(messages: &[AgentMessage]) -> Vec<Message> {
    messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Standard(standard) => Some(standard.clone()),
            AgentMessage::Custom(custom) => match custom.role.as_str() {
                "bashExecution" => {
                    let Ok(msg) = BashExecutionMessage::try_from(custom) else {
                        return None;
                    };
                    if msg.exclude_from_context.unwrap_or(false) {
                        return None;
                    }
                    Some(Message::User(UserMessage {
                        content: UserContent::Text(bash_execution_to_text(&msg)),
                        timestamp: msg.timestamp,
                    }))
                }
                "custom" => {
                    let Ok(msg) = CustomMessage::try_from(custom) else {
                        return None;
                    };
                    let content = match msg.content {
                        CustomMessageContent::Text(text) => UserContent::Text(text),
                        CustomMessageContent::Blocks(blocks) => UserContent::Blocks(
                            blocks
                                .into_iter()
                                .map(|block| match block {
                                    CustomMessageBlock::Text(TextContent { text, .. }) => {
                                        UserBlock::Text(TextContent {
                                            text,
                                            text_signature: None,
                                        })
                                    }
                                    CustomMessageBlock::Image(image) => UserBlock::Image(image),
                                })
                                .collect(),
                        ),
                    };
                    Some(Message::User(UserMessage {
                        content,
                        timestamp: msg.timestamp,
                    }))
                }
                "branchSummary" => {
                    let Ok(msg) = BranchSummaryMessage::try_from(custom) else {
                        return None;
                    };
                    Some(Message::User(UserMessage {
                        content: UserContent::Text(format!(
                            "{BRANCH_SUMMARY_PREFIX}{}{BRANCH_SUMMARY_SUFFIX}",
                            msg.summary
                        )),
                        timestamp: msg.timestamp,
                    }))
                }
                "compactionSummary" => {
                    let Ok(msg) = CompactionSummaryMessage::try_from(custom) else {
                        return None;
                    };
                    Some(Message::User(UserMessage {
                        content: UserContent::Text(format!(
                            "{COMPACTION_SUMMARY_PREFIX}{}{COMPACTION_SUMMARY_SUFFIX}",
                            msg.summary
                        )),
                        timestamp: msg.timestamp,
                    }))
                }
                _ => None,
            },
        })
        .collect()
}

impl TryFrom<&CustomAgentMessage> for BashExecutionMessage {
    type Error = ();

    fn try_from(custom: &CustomAgentMessage) -> Result<Self, Self::Error> {
        let wire = serde_json::to_value(custom).map_err(|_| ())?;
        serde_json::from_value(wire).map_err(|_| ())
    }
}

impl TryFrom<&CustomAgentMessage> for CustomMessage {
    type Error = ();

    fn try_from(custom: &CustomAgentMessage) -> Result<Self, Self::Error> {
        let wire = serde_json::to_value(custom).map_err(|_| ())?;
        serde_json::from_value(wire).map_err(|_| ())
    }
}

impl TryFrom<&CustomAgentMessage> for BranchSummaryMessage {
    type Error = ();

    fn try_from(custom: &CustomAgentMessage) -> Result<Self, Self::Error> {
        let wire = serde_json::to_value(custom).map_err(|_| ())?;
        serde_json::from_value(wire).map_err(|_| ())
    }
}

impl TryFrom<&CustomAgentMessage> for CompactionSummaryMessage {
    type Error = ();

    fn try_from(custom: &CustomAgentMessage) -> Result<Self, Self::Error> {
        let wire = serde_json::to_value(custom).map_err(|_| ())?;
        serde_json::from_value(wire).map_err(|_| ())
    }
}

#[cfg(test)]
mod tests;

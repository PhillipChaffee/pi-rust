//! Context compaction for long sessions, upstream
//! `src/core/compaction/compaction.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Pure functions for compaction logic. The session manager handles I/O, and
//! after compaction the session is reloaded.

use pi_agent_core::types::{AgentMessage, StreamFn, ThinkingLevel};
use pi_ai::types::{
    AssistantBlock, CacheRetention, Context, JsonValue, Message, Model, ProviderEnv,
    ProviderHeaders, SimpleStreamOptions, StopReason, TransportOptions, Usage, UserContent,
    UserMessage,
};
use pi_ai::utils::retry::{RetryCallbacks, RetryPolicy, retry_assistant_call};
use pi_ai::utils::text::content_text;
use pi_ai::utils::uuid::uuidv7;
use std::fmt::Write as _;
use tokio_util::sync::CancellationToken;

use crate::messages::convert_to_llm;
use crate::session_manager::context::{ByIdIndex, LeafId, build_session_context};
use crate::session_manager::entries::SessionEntry;
use crate::session_manager::file_entry::FileEntry;
use crate::session_manager::typed_entry_to_context_messages;

use super::utils::{
    FileOperations, SUMMARIZATION_SYSTEM_PROMPT, compute_file_lists, create_file_ops,
    extract_file_ops_from_message, format_file_operations, serialize_conversation,
};

// ============================================================================
// File Operation Tracking
// ============================================================================

/// Details stored in `CompactionEntry.details` for file tracking, upstream's
/// `CompactionDetails`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionDetails {
    /// Files only read in the summarized range.
    pub read_files: Vec<String>,
    /// Files modified in the summarized range.
    pub modified_files: Vec<String>,
}

/// Extract file operations from messages and the previous compaction entry,
/// upstream's `extractFileOperations`.
fn extract_file_operations(
    messages: &[AgentMessage],
    entries: &[SessionEntry],
    prev_compaction_index: Option<usize>,
) -> FileOperations {
    let mut file_ops = create_file_ops();

    // Collect from previous compaction's details (if pi-generated).
    if let Some(SessionEntry::Compaction(prev_compaction)) =
        prev_compaction_index.and_then(|index| entries.get(index))
    {
        // The fromHook field is kept for session-file compatibility: only
        // pi-generated compactions carry the file lists. The detail lists
        // read tolerantly, upstream's `Array.isArray` skips; non-string
        // items drop, where upstream's Set<string> would hold them.
        if prev_compaction.from_hook != Some(true)
            && let Some(details) = &prev_compaction.details
        {
            if let Some(read_files) = details.get("readFiles").and_then(JsonValue::as_array) {
                for file in read_files.iter().filter_map(JsonValue::as_str) {
                    file_ops.read.insert(file.to_owned());
                }
            }
            if let Some(modified_files) = details.get("modifiedFiles").and_then(JsonValue::as_array)
            {
                for file in modified_files.iter().filter_map(JsonValue::as_str) {
                    file_ops.edited.insert(file.to_owned());
                }
            }
        }
    }

    // Extract from tool calls in messages.
    for msg in messages {
        extract_file_ops_from_message(msg, &mut file_ops);
    }

    file_ops
}

// ============================================================================
// Message Extraction
// ============================================================================

/// Extract an `AgentMessage` from an entry if it produces one, upstream's
/// `getMessageFromEntryForCompaction`. Compaction entries return nothing: the
/// summary message they project is the context itself, not summarized
/// content.
fn get_message_from_entry_for_compaction(entry: &SessionEntry) -> Option<AgentMessage> {
    if matches!(entry, SessionEntry::Compaction(_)) {
        return None;
    }
    typed_entry_to_context_messages(entry).into_iter().next()
}

/// Result from [`compact`] — the session manager adds uuid/parentUuid when
/// saving, upstream's `CompactionResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionResult {
    /// The generated summary, file lists appended.
    pub summary: String,
    /// The id of the first entry the compaction kept.
    pub first_kept_entry_id: String,
    /// The estimated context token count before compaction.
    pub tokens_before: u64,
    /// The estimated tokens after compaction, when computed.
    pub estimated_tokens_after: Option<u64>,
    /// Usage from the LLM call(s) that generated this summary, if available.
    pub usage: Usage,
    /// Extension-specific data (e.g. an `ArtifactIndex`, version markers for
    /// structured compaction).
    pub details: Option<JsonValue>,
}

/// Merge two usage records, upstream's `combineUsage`. The optional split
/// fields (`cacheWrite1h`, `reasoning`) appear when either side carries them,
/// summing absent arms as zero.
fn combine_usage(first: &Usage, second: &Usage) -> Usage {
    let optional_sum = |a: Option<u64>, b: Option<u64>| -> Option<u64> {
        (a.is_some() || b.is_some()).then(|| a.unwrap_or(0) + b.unwrap_or(0))
    };
    Usage {
        input: first.input + second.input,
        output: first.output + second.output,
        cache_read: first.cache_read + second.cache_read,
        cache_write: first.cache_write + second.cache_write,
        cache_write_1h: optional_sum(first.cache_write_1h, second.cache_write_1h),
        reasoning: optional_sum(first.reasoning, second.reasoning),
        total_tokens: first.total_tokens + second.total_tokens,
        cost: pi_ai::types::UsageCost {
            input: first.cost.input + second.cost.input,
            output: first.cost.output + second.cost.output,
            cache_read: first.cost.cache_read + second.cost.cache_read,
            cache_write: first.cost.cache_write + second.cost.cache_write,
            total: first.cost.total + second.cost.total,
        },
    }
}

// ============================================================================
// Types
// ============================================================================

/// The compaction settings, upstream's `CompactionSettings` — the resolved
/// shape [`crate::settings_manager::SettingsManager::get_compaction_settings`]
/// returns.
pub use crate::settings_manager::CompactionSettings;

/// The built-in compaction settings, upstream's `DEFAULT_COMPACTION_SETTINGS`.
pub const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = CompactionSettings {
    enabled: true,
    reserve_tokens: 16384,
    keep_recent_tokens: 20000,
};

// ============================================================================
// Token calculation
// ============================================================================

/// Calculate total context tokens from usage, upstream's
/// `calculateContextTokens`. Uses the native `totalTokens` field when
/// non-zero, falling back to computing from components.
#[must_use]
pub const fn calculate_context_tokens(usage: &Usage) -> u64 {
    if usage.total_tokens == 0 {
        usage.input + usage.output + usage.cache_read + usage.cache_write
    } else {
        usage.total_tokens
    }
}

/// Get usage from an assistant message if available, upstream's
/// `getAssistantUsage`. Skips aborted, error, and all-zero usage messages as
/// they don't have valid usage data.
fn get_assistant_usage(msg: &AgentMessage) -> Option<Usage> {
    let AgentMessage::Standard(Message::Assistant(assistant)) = msg else {
        return None;
    };
    if assistant.stop_reason == StopReason::Aborted
        || assistant.stop_reason == StopReason::Error
        || calculate_context_tokens(&assistant.usage) == 0
    {
        return None;
    }
    Some(assistant.usage)
}

/// Find the last valid assistant message usage from session entries, upstream's
/// `getLastAssistantUsage`.
#[must_use]
pub fn get_last_assistant_usage(entries: &[SessionEntry]) -> Option<Usage> {
    entries.iter().rev().find_map(|entry| match entry {
        SessionEntry::Message(message) => message.message.as_ref().and_then(get_assistant_usage),
        _ => None,
    })
}

/// The context token estimate's breakdown, upstream's `ContextUsageEstimate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextUsageEstimate {
    /// The total estimated context tokens.
    pub tokens: u64,
    /// The portion anchored on the last assistant usage.
    pub usage_tokens: u64,
    /// The portion estimated for messages after the last usage.
    pub trailing_tokens: u64,
    /// The index of the message the usage anchored on, when any.
    pub last_usage_index: Option<usize>,
}

/// The last valid assistant usage in the message list, with its index,
/// upstream's `getLastAssistantUsageInfo`.
fn get_last_assistant_usage_info(messages: &[AgentMessage]) -> Option<(Usage, usize)> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| get_assistant_usage(message).map(|usage| (usage, index)))
}

/// Estimate context tokens from messages, upstream's `estimateContextTokens`.
///
/// Uses the last assistant usage when available; if there are messages after
/// the last usage, estimates their tokens with [`estimate_tokens`].
#[must_use]
pub fn estimate_context_tokens(messages: &[AgentMessage]) -> ContextUsageEstimate {
    match get_last_assistant_usage_info(messages) {
        None => {
            let estimated: u64 = messages.iter().map(estimate_tokens).sum();
            ContextUsageEstimate {
                tokens: estimated,
                usage_tokens: 0,
                trailing_tokens: estimated,
                last_usage_index: None,
            }
        }
        Some((usage, index)) => {
            let usage_tokens = calculate_context_tokens(&usage);
            let trailing_tokens: u64 = messages[index + 1..].iter().map(estimate_tokens).sum();
            ContextUsageEstimate {
                tokens: usage_tokens + trailing_tokens,
                usage_tokens,
                trailing_tokens,
                last_usage_index: Some(index),
            }
        }
    }
}

/// Check if compaction should trigger based on context usage, upstream's
/// `shouldCompact`.
#[must_use]
pub fn should_compact(
    context_tokens: u64,
    context_window: u64,
    settings: &CompactionSettings,
) -> bool {
    if !settings.enabled {
        return false;
    }
    let reserve_tokens = u64::try_from(settings.reserve_tokens).unwrap_or(0);
    context_tokens > context_window.saturating_sub(reserve_tokens)
}

// ============================================================================
// Cut point detection
// ============================================================================

/// The estimated-image block's char budget, upstream's `ESTIMATED_IMAGE_CHARS`.
const ESTIMATED_IMAGE_CHARS: usize = 4800;

/// The char count the estimates measure, upstream's `.length` UTF-16 units
/// restated as chars (the suites' fixtures are ASCII, where they agree).
fn char_units(text: &str) -> usize {
    text.chars().count()
}

/// `Math.ceil(chars / 4)`, the token heuristic's rounding.
const fn tokens_for_chars(chars: usize) -> u64 {
    chars.div_ceil(4) as u64
}

/// The text/image char budget of the duck-typed `string | blocks` content the
/// user, custom, and tool-result messages carry, upstream's
/// `estimateTextAndImageContentChars`.
fn text_and_image_content_chars(content: &UserContent) -> usize {
    match content {
        UserContent::Text(text) => char_units(text),
        UserContent::Blocks(blocks) => blocks
            .iter()
            .map(|block| match block {
                pi_ai::types::UserBlock::Text(content) => char_units(&content.text),
                pi_ai::types::UserBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum(),
    }
}

/// The same budget over the owned-JSON content a `custom` message carries,
/// upstream's untyped block scan.
fn custom_content_chars(content: Option<&JsonValue>) -> usize {
    match content {
        Some(JsonValue::String(text)) => char_units(text),
        Some(JsonValue::Array(blocks)) => blocks
            .iter()
            .map(
                |block| match block.get("type").and_then(JsonValue::as_str) {
                    Some("text") => block
                        .get("text")
                        .and_then(JsonValue::as_str)
                        .map_or(0, char_units),
                    Some("image") => ESTIMATED_IMAGE_CHARS,
                    _ => 0,
                },
            )
            .sum(),
        _ => 0,
    }
}

/// Estimate token count for a message using the chars/4 heuristic, upstream's
/// `estimateTokens`. This is conservative (overestimates tokens).
#[must_use]
pub fn estimate_tokens(message: &AgentMessage) -> u64 {
    let chars = match message {
        AgentMessage::Standard(Message::User(user)) => text_and_image_content_chars(&user.content),
        AgentMessage::Standard(Message::Assistant(assistant)) => {
            let mut chars = 0;
            for block in &assistant.content {
                match block {
                    AssistantBlock::Text(content) => chars += char_units(&content.text),
                    AssistantBlock::Thinking(content) => chars += char_units(&content.thinking),
                    AssistantBlock::ToolCall(call) => {
                        chars += call.name.len()
                            + serde_json::to_string(&call.arguments)
                                .unwrap_or_else(|_| "[unserializable]".to_owned())
                                .len();
                    }
                }
            }
            chars
        }
        AgentMessage::Standard(Message::ToolResult(result)) => result
            .content
            .iter()
            .map(|block| match block {
                pi_ai::types::ToolResultBlock::Text(content) => char_units(&content.text),
                pi_ai::types::ToolResultBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum(),
        AgentMessage::Custom(custom) => match custom.role.as_str() {
            "custom" => custom_content_chars(custom.field("content")),
            "bashExecution" => {
                custom
                    .field("command")
                    .and_then(JsonValue::as_str)
                    .map_or(0, char_units)
                    + custom
                        .field("output")
                        .and_then(JsonValue::as_str)
                        .map_or(0, char_units)
            }
            "branchSummary" | "compactionSummary" => custom
                .field("summary")
                .and_then(JsonValue::as_str)
                .map_or(0, char_units),
            _ => 0,
        },
    };
    tokens_for_chars(chars)
}

/// Whether a message may host a cut point, upstream's `isCutPointMessage`.
/// Tool results never do: they must follow their tool call.
fn is_cut_point_message(message: &AgentMessage) -> bool {
    match message {
        AgentMessage::Standard(Message::ToolResult(_)) => false,
        AgentMessage::Standard(_) => true,
        AgentMessage::Custom(custom) => matches!(
            custom.role.as_str(),
            "bashExecution" | "custom" | "branchSummary" | "compactionSummary"
        ),
    }
}

/// Whether a message starts a turn, upstream's `isTurnStartMessage`.
fn is_turn_start_message(message: &AgentMessage) -> bool {
    match message {
        AgentMessage::Standard(Message::User(_)) => true,
        AgentMessage::Standard(_) => false,
        AgentMessage::Custom(custom) => matches!(
            custom.role.as_str(),
            "bashExecution" | "custom" | "branchSummary" | "compactionSummary"
        ),
    }
}

/// Whether an entry hosts a turn-start message, upstream's
/// `isTurnStartEntry`.
fn is_turn_start_entry(entry: &SessionEntry) -> bool {
    if matches!(entry, SessionEntry::Compaction(_)) {
        return false;
    }
    typed_entry_to_context_messages(entry)
        .iter()
        .any(is_turn_start_message)
}

/// Find valid cut points: indices of context-visible user-like or assistant
/// messages, upstream's `findValidCutPoints`. Never cut at tool results (they
/// must follow their tool call). When we cut at an assistant message with
/// tool calls, its tool results follow it and will be kept.
fn find_valid_cut_points(
    entries: &[SessionEntry],
    start_index: usize,
    end_index: usize,
) -> Vec<usize> {
    let mut cut_points = Vec::new();
    for (index, entry) in entries.iter().enumerate().take(end_index).skip(start_index) {
        if matches!(entry, SessionEntry::Compaction(_)) {
            continue;
        }
        if typed_entry_to_context_messages(entry)
            .iter()
            .any(is_cut_point_message)
        {
            cut_points.push(index);
        }
    }
    cut_points
}

/// Find the context-visible user-role message that starts the turn containing
/// the given entry index, upstream's `findTurnStartIndex`. Returns `-1` when
/// no turn start is found before the index.
#[expect(
    clippy::cast_possible_wrap,
    reason = "the -1 sentinel restates upstream's number; usize indices cannot reach i64's range"
)]
#[must_use]
pub fn find_turn_start_index(
    entries: &[SessionEntry],
    entry_index: usize,
    start_index: usize,
) -> i64 {
    for index in (start_index..=entry_index).rev() {
        if is_turn_start_entry(&entries[index]) {
            return index as i64;
        }
    }
    -1
}

/// The cut-point result, upstream's `CutPointResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutPointResult {
    /// Index of the first entry to keep.
    pub first_kept_entry_index: usize,
    /// Index of the user message that starts the turn being split, or `-1`
    /// when not splitting.
    pub turn_start_index: i64,
    /// Whether this cut splits a turn (cut point is not a user message).
    pub is_split_turn: bool,
}

/// Find the cut point in session entries that keeps approximately
/// `keep_recent_tokens`, upstream's `findCutPoint`.
///
/// Algorithm: walk backwards from newest, accumulating estimated message
/// sizes. Stop when we've accumulated at least `keep_recent_tokens`. Cut at
/// that point. Can cut at user OR assistant messages (never tool results).
/// When cutting at an assistant message with tool calls, its tool results
/// come after and will be kept.
///
/// Only considers entries between `start_index` (inclusive) and `end_index`
/// (exclusive).
#[must_use]
pub fn find_cut_point(
    entries: &[SessionEntry],
    start_index: usize,
    end_index: usize,
    keep_recent_tokens: u64,
) -> CutPointResult {
    let cut_points = find_valid_cut_points(entries, start_index, end_index);

    if cut_points.is_empty() {
        return CutPointResult {
            first_kept_entry_index: start_index,
            turn_start_index: -1,
            is_split_turn: false,
        };
    }

    // Walk backwards from newest, accumulating estimated message sizes.
    let mut accumulated_tokens: u64 = 0;
    let mut cut_index = cut_points[0]; // Default: keep from first message (not header)

    for index in (start_index..end_index).rev() {
        let message_tokens: u64 = typed_entry_to_context_messages(&entries[index])
            .iter()
            .map(estimate_tokens)
            .sum();
        if message_tokens == 0 {
            continue;
        }
        accumulated_tokens += message_tokens;

        // Check if we've exceeded the budget.
        if accumulated_tokens >= keep_recent_tokens {
            // Find the closest valid cut point at or after this entry.
            if let Some(closest) = cut_points.iter().find(|cut_point| **cut_point >= index) {
                cut_index = *closest;
            }
            break;
        }
    }

    // Scan backwards from cut_index to include adjacent metadata entries that
    // do not affect context. Stop at compaction boundaries or context-visible
    // entries.
    while cut_index > start_index {
        let prev_entry = &entries[cut_index - 1];
        if matches!(prev_entry, SessionEntry::Compaction(_))
            || !typed_entry_to_context_messages(prev_entry).is_empty()
        {
            break;
        }
        cut_index -= 1;
    }

    // Determine if this is a split turn.
    let cut_entry = &entries[cut_index];
    let starts_turn = is_turn_start_entry(cut_entry);
    let turn_start_index = if starts_turn {
        -1
    } else {
        find_turn_start_index(entries, cut_index, start_index)
    };

    CutPointResult {
        first_kept_entry_index: cut_index,
        turn_start_index,
        is_split_turn: !starts_turn && turn_start_index != -1,
    }
}

// ============================================================================
// Summarization
// ============================================================================

const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.\n\nUse this EXACT format:\n\n## Goal\n[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements mentioned by user]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Current work]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [Ordered list of what should happen next]\n\n## Critical Context\n- [Any data, examples, or references needed to continue]\n- [Or \"(none)\" if not applicable]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

const UPDATE_SUMMARIZATION_PROMPT: &str = "The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.\n\nUpdate the existing structured summary with new information. RULES:\n- PRESERVE all existing information from the previous summary\n- ADD new progress, decisions, and context from the new messages\n- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed\n- UPDATE \"Next Steps\" based on what was accomplished\n- PRESERVE exact file paths, function names, and error messages\n- If something is no longer relevant, you may remove it\n\nUse this EXACT format:\n\n## Goal\n[Preserve existing goals, add new ones if the task expanded]\n\n## Constraints & Preferences\n- [Preserve existing, add new ones discovered]\n\n## Progress\n### Done\n- [x] [Include previously done items AND newly completed items]\n\n### In Progress\n- [ ] [Current work - update based on progress]\n\n### Blocked\n- [Current blockers - remove if resolved]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale] (preserve all previous, add new)\n\n## Next Steps\n1. [Update based on current state]\n\n## Critical Context\n- [Preserve important context, add new if needed]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = "This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.\n\nSummarize the prefix to provide context for the retained suffix:\n\n## Original Request\n[What did the user ask for in this turn?]\n\n## Early Progress\n- [Key decisions and work done in the prefix]\n\n## Context for Suffix\n- [Information needed to understand the retained recent work]\n\nBe concise. Focus on what's needed to understand the kept suffix.";

/// Why compaction or one of its summarization calls failed; the message is
/// upstream's thrown `Error` text verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionError {
    /// The upstream error message.
    pub message: String,
}

impl std::fmt::Display for CompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CompactionError {}

/// Returns an error message when a summarization response cannot safely be
/// persisted, upstream's `getSummarizationFailure`. A length stop contains
/// partial text and must not become a session checkpoint.
#[must_use]
pub fn get_summarization_failure(
    response: &pi_ai::types::AssistantMessage,
    label: &str,
) -> Option<String> {
    if response.stop_reason == StopReason::Error {
        return Some(format!(
            "{label} failed: {}",
            response.error_message.as_deref().unwrap_or("Unknown error")
        ));
    }
    if response.stop_reason == StopReason::Length {
        return Some(format!(
            "{label} failed: generation hit the token cap and the summary is incomplete"
        ));
    }
    None
}

/// The agent thinking level's request form, upstream's `thinkingLevel !==
/// "off"` gate: `off` maps to an absent `reasoning` field, every other level
/// passes through.
const fn stream_reasoning(level: ThinkingLevel) -> Option<pi_ai::types::ThinkingLevel> {
    match level {
        ThinkingLevel::Off => None,
        ThinkingLevel::Minimal => Some(pi_ai::types::ThinkingLevel::Minimal),
        ThinkingLevel::Low => Some(pi_ai::types::ThinkingLevel::Low),
        ThinkingLevel::Medium => Some(pi_ai::types::ThinkingLevel::Medium),
        ThinkingLevel::High => Some(pi_ai::types::ThinkingLevel::High),
        ThinkingLevel::Xhigh => Some(pi_ai::types::ThinkingLevel::Xhigh),
        ThinkingLevel::Max => Some(pi_ai::types::ThinkingLevel::Max),
    }
}

/// The options one summarization request carries, upstream's
/// `createSummarizationOptions`.
#[expect(
    clippy::too_many_arguments,
    reason = "the parameter list mirrors upstream's createSummarizationOptions()"
)]
fn create_summarization_options(
    model: &Model,
    max_tokens: u64,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
    env: Option<&ProviderEnv>,
    signal: Option<&CancellationToken>,
    thinking_level: Option<ThinkingLevel>,
    session_id: Option<&str>,
) -> SimpleStreamOptions {
    let reasoning = thinking_level
        .filter(|level| *level != ThinkingLevel::Off && model.reasoning)
        .and_then(stream_reasoning);
    SimpleStreamOptions {
        transport_options: TransportOptions {
            signal: signal.cloned(),
            ..TransportOptions::default()
        },
        api_key: api_key.map(str::to_owned),
        headers: headers.cloned(),
        env: env.cloned(),
        max_tokens: Some(max_tokens),
        session_id: session_id.map(str::to_owned),
        reasoning,
        ..SimpleStreamOptions::default()
    }
}

/// Shared choke point for every compaction/branch-summary summarization call,
/// upstream's `completeSummarization`.
///
/// Wraps the single LLM call in [`retry_assistant_call`] so transient stream
/// drops (e.g. `terminated`, socket close) honor the configured retry policy
/// instead of failing the whole compaction on the first attempt. Deterministic
/// errors and aborts return immediately (see [`retry_assistant_call`]).
pub async fn complete_summarization(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    stream_fn: Option<&StreamFn>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&RetryCallbacks>,
) -> pi_ai::types::AssistantMessage {
    // Avoid cache writes for one-off summaries. Reuse caller-supplied
    // routing when available; callers without a session ID, including branch
    // summaries, receive a fresh routing ID. Upstream throws when uuidv7
    // fails; the only failure is the format's timestamp range, which the
    // process clock cannot reach, so the id is silently absent instead.
    let request_options = SimpleStreamOptions {
        cache_retention: Some(CacheRetention::None),
        session_id: options.session_id.clone().or_else(|| uuidv7(None).ok()),
        ..options.clone()
    };
    let produce = || async {
        match stream_fn {
            Some(stream_fn) => {
                stream_fn(model, context, Some(&request_options))
                    .result()
                    .await
            }
            None => pi_ai::compat::complete_simple(model, context, Some(&request_options)).await,
        }
    };
    retry_assistant_call(
        produce,
        retry,
        request_options.transport_options.signal.as_ref(),
        callbacks,
    )
    .await
}

/// The usage one summary generation reports, upstream's inline
/// `{ text, usage }` return.
#[derive(Debug, Clone, PartialEq)]
pub struct SummaryWithUsage {
    /// The generated summary text.
    pub text: String,
    /// The provider usage of the summarization call.
    pub usage: Usage,
}

/// Generate a summary of the conversation using the LLM, upstream's
/// `generateSummary` — [`generate_summary_with_usage`] without the usage.
///
/// # Errors
/// The summarization request's failed settlement, or a tool-call attempt.
#[expect(
    clippy::too_many_arguments,
    reason = "the positional signature mirrors upstream's generateSummary*(); the request plumbing rides the tail parameters"
)]
pub async fn generate_summary(
    current_messages: &[AgentMessage],
    model: &Model,
    reserve_tokens: i64,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
    signal: Option<&CancellationToken>,
    custom_instructions: Option<&str>,
    previous_summary: Option<&str>,
    thinking_level: Option<ThinkingLevel>,
    stream_fn: Option<&StreamFn>,
    env: Option<&ProviderEnv>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&RetryCallbacks>,
    session_id: Option<&str>,
) -> Result<String, CompactionError> {
    generate_summary_with_usage(
        current_messages,
        model,
        reserve_tokens,
        api_key,
        headers,
        signal,
        custom_instructions,
        previous_summary,
        thinking_level,
        stream_fn,
        env,
        retry,
        callbacks,
        session_id,
    )
    .await
    .map(|summary| summary.text)
}

/// Build the provider context for a standalone summary request, upstream's
/// `buildSummarizationContext`. Branch summarization reuses it for its own
/// prompt shape.
pub(crate) fn build_summarization_context(prompt_text: &str) -> Context {
    Context {
        system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_owned()),
        messages: vec![Message::User(UserMessage {
            content: UserContent::Blocks(vec![pi_ai::types::UserBlock::Text(
                pi_ai::types::TextContent {
                    text: prompt_text.to_owned(),
                    text_signature: None,
                },
            )]),
            timestamp: now_millis(),
        })],
        tools: None,
    }
}

/// `Date.now()`, the summarization context's user-message timestamp.
fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        // A pre-epoch system clock cannot happen in practice; 0 keeps the
        // timestamp a number the way upstream's Date.now() always is.
        .map_or(0, |duration| {
            i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
        })
}

/// The output-token cap one summarization request gets, upstream's
/// `Math.min(Math.floor(fraction * reserveTokens), model.maxTokens || Infinity)`.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the cap restates JS number arithmetic over validated non-negative settings"
)]
fn summary_max_tokens(reserve_tokens: i64, fraction: f64, model: &Model) -> u64 {
    let reserve_cap = (fraction * reserve_tokens as f64).floor();
    if reserve_cap <= 0.0 {
        return 0;
    }
    if model.max_tokens > 0 {
        (reserve_cap as u64).min(model.max_tokens)
    } else {
        reserve_cap as u64
    }
}

/// Generate or update a conversation summary and return its provider usage,
/// upstream's `generateSummaryWithUsage`. If previousSummary is provided,
/// uses the update prompt to merge.
///
/// # Errors
/// The summarization request's failed settlement (`Summarization failed:
/// ...`), or a tool-call attempt (`Summarization attempted to call a tool`).
#[expect(
    clippy::too_many_arguments,
    reason = "the positional signature mirrors upstream's generateSummaryWithUsage(); the request plumbing rides the tail parameters"
)]
pub async fn generate_summary_with_usage(
    current_messages: &[AgentMessage],
    model: &Model,
    reserve_tokens: i64,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
    signal: Option<&CancellationToken>,
    custom_instructions: Option<&str>,
    previous_summary: Option<&str>,
    thinking_level: Option<ThinkingLevel>,
    stream_fn: Option<&StreamFn>,
    env: Option<&ProviderEnv>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&RetryCallbacks>,
    session_id: Option<&str>,
) -> Result<SummaryWithUsage, CompactionError> {
    let max_tokens = summary_max_tokens(reserve_tokens, 0.8, model);

    // Use update prompt if we have a previous summary, otherwise initial
    // prompt.
    let mut base_prompt = if previous_summary.is_some() {
        UPDATE_SUMMARIZATION_PROMPT.to_owned()
    } else {
        SUMMARIZATION_PROMPT.to_owned()
    };
    if let Some(custom_instructions) = custom_instructions {
        let _ = write!(base_prompt, "\n\nAdditional focus: {custom_instructions}");
    }

    // Serialize conversation to text so model doesn't try to continue it.
    // Convert to LLM messages first (handles custom types like bashExecution,
    // custom, etc.).
    let llm_messages = convert_to_llm(current_messages);
    let conversation_text = serialize_conversation(&llm_messages);

    // Build the prompt with conversation wrapped in tags.
    let mut prompt_text = format!("<conversation>\n{conversation_text}\n</conversation>\n\n");
    if let Some(previous_summary) = previous_summary {
        let _ = write!(
            prompt_text,
            "<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        );
    }
    prompt_text.push_str(&base_prompt);

    let completion_options = create_summarization_options(
        model,
        max_tokens,
        api_key,
        headers,
        env,
        signal,
        thinking_level,
        session_id,
    );

    let response = complete_summarization(
        model,
        &build_summarization_context(&prompt_text),
        &completion_options,
        stream_fn,
        retry,
        callbacks,
    )
    .await;

    if let Some(failure) = get_summarization_failure(&response, "Summarization") {
        return Err(CompactionError { message: failure });
    }
    if response
        .content
        .iter()
        .any(|block| matches!(block, AssistantBlock::ToolCall(_)))
    {
        return Err(CompactionError {
            message: "Summarization attempted to call a tool".to_owned(),
        });
    }

    let text_content = content_text(&response.content.as_slice());

    Ok(SummaryWithUsage {
        text: text_content,
        usage: response.usage,
    })
}

// ============================================================================
// Compaction Preparation (for extensions)
// ============================================================================

/// The data one compaction run needs, upstream's `CompactionPreparation`.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPreparation {
    /// UUID of first entry to keep.
    pub first_kept_entry_id: String,
    /// Messages that will be summarized and discarded.
    pub messages_to_summarize: Vec<AgentMessage>,
    /// Messages that will be turned into turn prefix summary (if splitting).
    pub turn_prefix_messages: Vec<AgentMessage>,
    /// Whether this is a split turn (cut point in middle of turn).
    pub is_split_turn: bool,
    /// The estimated context token count before compaction.
    pub tokens_before: u64,
    /// Summary from previous compaction, for iterative update.
    pub previous_summary: Option<String>,
    /// File operations extracted from messagesToSummarize.
    pub file_ops: FileOperations,
    /// Compaction settings from settings.json.
    pub settings: CompactionSettings,
}

/// The `turnStartIndex` `-1` sentinel as a slice index, upstream's
/// `cutPoint.turnStartIndex`. Only read when the cut split a turn, where the
/// sentinel cannot appear.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the sentinel guard clamps to a non-negative index usize holds"
)]
fn turn_start_index(cut_point: &CutPointResult) -> usize {
    cut_point.turn_start_index.max(0) as usize
}

/// Compute the compaction preparation over a session path, upstream's
/// `prepareCompaction`.
///
/// Returns `None` when the path ends in a compaction, nothing new is
/// summarizable, or the first kept entry lacks an id (the session needs
/// migration).
#[must_use]
pub fn prepare_compaction(
    path_entries: &[SessionEntry],
    settings: CompactionSettings,
) -> Option<CompactionPreparation> {
    if let Some(last) = path_entries.last()
        && matches!(last, SessionEntry::Compaction(_))
    {
        return None;
    }

    let prev_compaction_index: Option<usize> = path_entries
        .iter()
        .rposition(|entry| matches!(entry, SessionEntry::Compaction(_)));

    let (previous_summary, boundary_start) = prev_compaction_index.map_or((None, 0), |index| {
        let SessionEntry::Compaction(prev_compaction) = &path_entries[index] else {
            unreachable!("rposition matched a compaction entry");
        };
        let previous_summary = Some(prev_compaction.summary.clone());
        // Upstream's findIndex matches `entry.id === prevCompaction.firstKeptEntryId`;
        // when the previous compaction lacks the id, no typed entry matches
        // and the boundary falls to the compaction itself (an id-less entry
        // could match upstream, which the typed port declines the same way
        // the context projection does).
        let first_kept_entry_index = path_entries.iter().position(|entry| {
            entry.base().id.is_some()
                && entry.base().id.as_deref() == prev_compaction.first_kept_entry_id.as_deref()
        });
        (
            previous_summary,
            first_kept_entry_index.unwrap_or(index + 1),
        )
    });
    let boundary_end = path_entries.len();

    let tokens_before = {
        let file_entries: Vec<FileEntry> = path_entries
            .iter()
            .map(|entry| FileEntry::Entry(entry.clone()))
            .collect();
        let by_id: ByIdIndex = file_entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.entry_id().map(|id| (id.to_owned(), index)))
            .collect();
        let context = build_session_context(&file_entries, LeafId::Default, &by_id);
        estimate_context_tokens(&context.messages).tokens
    };

    let cut_point = find_cut_point(
        path_entries,
        boundary_start,
        boundary_end,
        settings.keep_recent_tokens.cast_unsigned(),
    );

    // Get UUID of first kept entry.
    let first_kept_entry = &path_entries[cut_point.first_kept_entry_index];
    let Some(first_kept_entry_id) = first_kept_entry.base().id.clone() else {
        return None; // Session needs migration
    };

    let history_end: usize = if cut_point.is_split_turn {
        turn_start_index(&cut_point)
    } else {
        cut_point.first_kept_entry_index
    };

    // Messages to summarize (will be discarded after summary).
    let mut messages_to_summarize: Vec<AgentMessage> = Vec::new();
    for entry in &path_entries[boundary_start..history_end] {
        if let Some(msg) = get_message_from_entry_for_compaction(entry) {
            messages_to_summarize.push(msg);
        }
    }

    // Messages for turn prefix summary (if splitting a turn).
    let mut turn_prefix_messages: Vec<AgentMessage> = Vec::new();
    if cut_point.is_split_turn {
        for entry in &path_entries[turn_start_index(&cut_point)..cut_point.first_kept_entry_index] {
            if let Some(msg) = get_message_from_entry_for_compaction(entry) {
                turn_prefix_messages.push(msg);
            }
        }
    }

    if messages_to_summarize.is_empty() && turn_prefix_messages.is_empty() {
        return None;
    }

    // Extract file operations from messages and previous compaction.
    let mut file_ops =
        extract_file_operations(&messages_to_summarize, path_entries, prev_compaction_index);

    // Also extract file ops from turn prefix if splitting.
    if cut_point.is_split_turn {
        for msg in &turn_prefix_messages {
            extract_file_ops_from_message(msg, &mut file_ops);
        }
    }

    Some(CompactionPreparation {
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        is_split_turn: cut_point.is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
        settings,
    })
}

// ============================================================================
// Main compaction function
// ============================================================================

/// Generate summaries for compaction using prepared data, upstream's
/// `compact`. Returns [`CompactionResult`] — the session manager adds
/// uuid/parentUuid when saving.
///
/// # Errors
/// Either summarization call's failed settlement, a tool-call attempt, or a
/// first kept entry without a UUID (the session may need migration).
#[expect(
    clippy::too_many_arguments,
    reason = "the positional signature mirrors upstream's compact(); the request plumbing rides the tail parameters"
)]
pub async fn compact(
    preparation: CompactionPreparation,
    model: &Model,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
    custom_instructions: Option<&str>,
    signal: Option<&CancellationToken>,
    thinking_level: Option<ThinkingLevel>,
    stream_fn: Option<&StreamFn>,
    env: Option<&ProviderEnv>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&RetryCallbacks>,
    session_id: Option<&str>,
) -> Result<CompactionResult, CompactionError> {
    let CompactionPreparation {
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
        settings,
    } = preparation;

    // Generate summaries and merge into one.
    let (summary, summary_usage) = if is_split_turn && !turn_prefix_messages.is_empty() {
        let (history_text, history_usage) = generate_split_history(
            &messages_to_summarize,
            model,
            &settings,
            api_key,
            headers,
            signal,
            custom_instructions,
            previous_summary.as_deref(),
            thinking_level,
            stream_fn,
            env,
            retry,
            callbacks,
            session_id,
        )
        .await?;
        let turn_prefix_result = generate_turn_prefix_summary(
            &turn_prefix_messages,
            model,
            settings.reserve_tokens,
            api_key,
            headers,
            env,
            signal,
            thinking_level,
            stream_fn,
            retry,
            callbacks,
            session_id,
        )
        .await?;
        // Merge into single summary.
        let summary = format!(
            "{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{}",
            turn_prefix_result.text
        );
        let summary_usage = match history_usage {
            Some(history_usage) => combine_usage(&history_usage, &turn_prefix_result.usage),
            None => turn_prefix_result.usage,
        };
        (summary, summary_usage)
    } else {
        // Just generate history summary.
        let result = generate_summary_with_usage(
            &messages_to_summarize,
            model,
            settings.reserve_tokens,
            api_key,
            headers,
            signal,
            custom_instructions,
            previous_summary.as_deref(),
            thinking_level,
            stream_fn,
            env,
            retry,
            callbacks,
            session_id,
        )
        .await?;
        (result.text, result.usage)
    };

    // Compute file lists and append to summary.
    let file_lists = compute_file_lists(&file_ops);
    let summary = format!(
        "{summary}{}",
        format_file_operations(&file_lists.read_files, &file_lists.modified_files)
    );

    if first_kept_entry_id.is_empty() {
        return Err(CompactionError {
            message: "First kept entry has no UUID - session may need migration".to_owned(),
        });
    }

    Ok(CompactionResult {
        summary,
        first_kept_entry_id,
        tokens_before,
        estimated_tokens_after: None,
        usage: summary_usage,
        details: Some(
            serde_json::to_value(CompactionDetails {
                read_files: file_lists.read_files,
                modified_files: file_lists.modified_files,
            })
            .unwrap_or(JsonValue::Null),
        ),
    })
}

/// The history half of a split-turn compaction, upstream's inline
/// `messagesToSummarize.length > 0` guard: when the split left nothing to
/// summarize the text falls back to `"No prior history."` and no usage
/// reports.
#[expect(
    clippy::too_many_arguments,
    reason = "the parameter list mirrors upstream's inline split-turn call"
)]
async fn generate_split_history(
    messages_to_summarize: &[AgentMessage],
    model: &Model,
    settings: &CompactionSettings,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
    signal: Option<&CancellationToken>,
    custom_instructions: Option<&str>,
    previous_summary: Option<&str>,
    thinking_level: Option<ThinkingLevel>,
    stream_fn: Option<&StreamFn>,
    env: Option<&ProviderEnv>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&RetryCallbacks>,
    session_id: Option<&str>,
) -> Result<(String, Option<Usage>), CompactionError> {
    if messages_to_summarize.is_empty() {
        return Ok(("No prior history.".to_owned(), None));
    }
    let history_result = generate_summary_with_usage(
        messages_to_summarize,
        model,
        settings.reserve_tokens,
        api_key,
        headers,
        signal,
        custom_instructions,
        previous_summary,
        thinking_level,
        stream_fn,
        env,
        retry,
        callbacks,
        session_id,
    )
    .await?;
    Ok((history_result.text, Some(history_result.usage)))
}

/// Generate a summary for a turn prefix (when splitting a turn), upstream's
/// `generateTurnPrefixSummary`.
///
/// # Errors
/// The summarization request's failed settlement, or a tool-call attempt.
#[expect(
    clippy::too_many_arguments,
    reason = "the positional signature mirrors upstream's generateTurnPrefixSummary(); the request plumbing rides the tail parameters"
)]
async fn generate_turn_prefix_summary(
    messages: &[AgentMessage],
    model: &Model,
    reserve_tokens: i64,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
    env: Option<&ProviderEnv>,
    signal: Option<&CancellationToken>,
    thinking_level: Option<ThinkingLevel>,
    stream_fn: Option<&StreamFn>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&RetryCallbacks>,
    session_id: Option<&str>,
) -> Result<SummaryWithUsage, CompactionError> {
    // Smaller budget for turn prefix.
    let max_tokens = summary_max_tokens(reserve_tokens, 0.5, model);
    let llm_messages = convert_to_llm(messages);
    let conversation_text = serialize_conversation(&llm_messages);
    let prompt_text = format!(
        "<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
    );

    let response = complete_summarization(
        model,
        &build_summarization_context(&prompt_text),
        &create_summarization_options(
            model,
            max_tokens,
            api_key,
            headers,
            env,
            signal,
            thinking_level,
            session_id,
        ),
        stream_fn,
        retry,
        callbacks,
    )
    .await;

    if let Some(failure) = get_summarization_failure(&response, "Turn prefix summarization") {
        return Err(CompactionError { message: failure });
    }
    if response
        .content
        .iter()
        .any(|block| matches!(block, AssistantBlock::ToolCall(_)))
    {
        return Err(CompactionError {
            message: "Turn prefix summarization attempted to call a tool".to_owned(),
        });
    }

    Ok(SummaryWithUsage {
        text: content_text(&response.content.as_slice()),
        usage: response.usage,
    })
}

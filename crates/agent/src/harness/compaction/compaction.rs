//! The compaction logic, ported from upstream
//! `src/harness/compaction/compaction.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The token estimates and cut points, the threshold check, the summary
//! prompts, and the `prepare`/`compact` pair whose caller-owned request
//! boundary ([`SummaryRequest`]) restates the injectable summarization call.

use std::fmt::Write as _;
use std::sync::Arc;

use pi_ai::auth::resolve::now_ms;
use pi_ai::models::{Models, WithTransforms};
use pi_ai::types::{
    AssistantBlock, AssistantMessage, BoxedFuture, CacheRetention, Context as AiContext, Message,
    Model, SimpleStreamOptions, StopReason, TextContent, TransportOptions, Usage, UserBlock,
    UserContent, UserMessage,
};
use pi_ai::utils::retry::{RetryCallbacks, RetryPolicy, retry_assistant_call};
use pi_ai::utils::text::content_text;
use pi_ai::utils::uuid::uuidv7;
use serde_json::Value as JsonValue;

use crate::agent_loop::stream_reasoning;
use crate::harness::context::{Context, get_telemetry_context, request_signal};
use crate::harness::messages::{
    convert_to_llm, create_branch_summary_message, create_compaction_summary_message,
};
use crate::harness::session::context::{build_context_entries, session_entry_to_context_messages};
use crate::harness::session::types::{Entry, MessageEntry};
use crate::harness::types::{CompactionError, CompactionErrorCode};
use crate::harness::utils::usage::add_usage;
use crate::types::{AgentMessage, ThinkingLevel};

use super::types::{CompactResult, CompactionPreparation, CompactionSettings};
pub use super::utils::serialize_conversation;
use super::utils::{
    FileOpsAccumulator, compute_file_lists, durable_file_operations, extract_file_ops_from_message,
    format_file_operations, safe_json_stringify,
};

/// File-operation details stored on generated compaction entries,
/// upstream's `CompactionDetails`; the entry's `details` JSON carries the
/// camelCase wire names.
pub use super::types::CompactionDetails;

/// The caller-owned one-request boundary the summary generators send
/// through, upstream's `SummaryRequest` —
/// `(aiContext, options, context) => Promise<AssistantMessage>`.
///
/// The chord context is the request's cancellation and telemetry scope.
/// Per the crate's stored-closure convention the future is `'static`:
/// implementations clone their inputs into it.
pub type SummaryRequest = Arc<
    dyn Fn(&AiContext, &SimpleStreamOptions, &Context) -> BoxedFuture<'static, AssistantMessage>
        + Send
        + Sync,
>;

/// The text and usage one summarization response yields, upstream's
/// `{ text, usage }` return of `generateSummaryWithUsage`.
#[derive(Clone, Debug, PartialEq)]
pub struct SummaryWithUsage {
    /// The summary text.
    pub text: String,
    /// The usage the summarizing call reported.
    pub usage: Usage,
}

/// Estimated context-token usage for a message list, upstream's
/// `ContextUsageEstimate`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextUsageEstimate {
    /// Estimated total context tokens.
    pub tokens: u64,
    /// Tokens reported by the most recent assistant usage block.
    pub usage_tokens: u64,
    /// Estimated tokens after the most recent assistant usage block.
    pub trailing_tokens: u64,
    /// Index of the message that provided usage, or `None` when none
    /// exists.
    pub last_usage_index: Option<usize>,
}

/// Cut point selected for compaction, upstream's `CutPointResult`.
///
/// Upstream carries `isSplitTurn` beside the `-1`-or-index
/// `turnStartIndex`; the split flag is exactly the index's presence, so
/// the port carries the index and derives the flag
/// ([`CutPointResult::is_split_turn`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CutPointResult {
    /// Index of the first entry retained after compaction.
    pub first_kept_entry_index: usize,
    /// Index of the turn-start entry when the cut splits a turn, otherwise
    /// `None`.
    pub turn_start_index: Option<usize>,
}

impl CutPointResult {
    /// Whether the selected cut point splits an in-progress turn, upstream's
    /// `isSplitTurn`.
    #[must_use]
    pub const fn is_split_turn(&self) -> bool {
        self.turn_start_index.is_some()
    }
}

/// The summary generation inputs the `WithRequest` entry takes, upstream's
/// `SummaryGenerationOptions`.
#[derive(Clone, Debug)]
pub struct SummaryGenerationOptions {
    /// The model the summary request runs on.
    pub model: Model,
    /// Tokens reserved for summary prompt and output.
    pub reserve_tokens: u64,
    /// Optional instructions appended to the summary prompt.
    pub custom_instructions: Option<String>,
    /// The previous compaction summary, updating it instead of starting
    /// fresh.
    pub previous_summary: Option<String>,
    /// The thinking level the summary request carries.
    pub thinking_level: Option<ThinkingLevel>,
}

/// The compaction generation inputs the `WithRequest` entry takes,
/// upstream's `CompactGenerationOptions`.
#[derive(Clone, Debug)]
pub struct CompactGenerationOptions {
    /// The model the summary requests run on.
    pub model: Model,
    /// Optional instructions appended to the summary prompt.
    pub custom_instructions: Option<String>,
    /// The thinking level the summary requests carry.
    pub thinking_level: Option<ThinkingLevel>,
}

/// The models-backed request boundary the `Models`-taking entries build,
/// upstream's `(aiContext, options, context) => completeSimpleWithRetries(...)`
/// arrow at the three call sites. Branch summarization imports it like
/// upstream's `./compaction.ts` import.
pub(crate) fn models_summary_request(
    models: &Models,
    model: &Model,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&RetryCallbacks>,
) -> SummaryRequest {
    let models = models.clone();
    let model = model.clone();
    let retry = retry.copied();
    let callbacks = callbacks.cloned();
    Arc::new(
        move |ai_context: &AiContext, options: &SimpleStreamOptions, request_context: &Context| {
            let ai_context = ai_context.clone();
            let options = options.clone();
            let request_context = request_context.clone();
            let models = models.clone();
            let model = model.clone();
            let callbacks = callbacks.clone();
            Box::pin(async move {
                complete_simple_with_retries(
                    &models,
                    &model,
                    &ai_context,
                    &options,
                    retry.as_ref(),
                    callbacks.as_ref(),
                    &request_context,
                )
                .await
            })
        },
    )
}

fn safe_json_stringify_length(value: &impl serde::Serialize) -> usize {
    safe_json_stringify(value).len()
}

/// The char count the estimates measure, upstream's `.length` UTF-16 units
/// restated as chars (the suites' fixtures are ASCII, where they agree).
fn char_units(text: &str) -> usize {
    text.chars().count()
}

/// The estimated-image block's char budget, upstream's
/// `ESTIMATED_IMAGE_CHARS`.
const ESTIMATED_IMAGE_CHARS: usize = 4800;

/// `Math.ceil(chars / 4)`, the token heuristic's rounding.
const fn tokens_for_chars(chars: usize) -> u64 {
    chars.div_ceil(4) as u64
}

/// The text/image char budget of the duck-typed `string | blocks` content
/// the custom and user messages carry, upstream's
/// `estimateTextAndImageContentChars`.
fn text_and_image_content_chars(content: &JsonValue) -> usize {
    match content {
        JsonValue::String(text) => char_units(text),
        JsonValue::Array(blocks) => blocks
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

fn user_content_chars(content: &UserContent) -> usize {
    match content {
        UserContent::Text(text) => char_units(text),
        UserContent::Blocks(blocks) => blocks
            .iter()
            .map(|block| match block {
                UserBlock::Text(content) => char_units(&content.text),
                UserBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum(),
    }
}

fn tool_result_content_chars(content: &[pi_ai::types::ToolResultBlock]) -> usize {
    content
        .iter()
        .map(|block| match block {
            pi_ai::types::ToolResultBlock::Text(content) => char_units(&content.text),
            pi_ai::types::ToolResultBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
        })
        .sum()
}

/// The message's Unix-millisecond timestamp, upstream's `message.timestamp`
/// on every `AgentMessage` shape.
const fn message_timestamp(message: &AgentMessage) -> i64 {
    match message {
        AgentMessage::Standard(Message::User(user)) => user.timestamp,
        AgentMessage::Standard(Message::Assistant(assistant)) => assistant.timestamp,
        AgentMessage::Standard(Message::ToolResult(result)) => result.timestamp,
        AgentMessage::Custom(custom) => custom.timestamp,
    }
}

/// Estimate token count for one message using a conservative character
/// heuristic, upstream's `estimateTokens`.
///
/// A quarter of the message's char count: text, thinking, tool-call names
/// and arguments, images at the fixed estimate, and summary texts whole.
#[must_use]
pub fn estimate_tokens(message: &AgentMessage) -> u64 {
    match message {
        AgentMessage::Standard(Message::User(user)) => {
            tokens_for_chars(user_content_chars(&user.content))
        }
        AgentMessage::Standard(Message::Assistant(assistant)) => {
            let mut chars = 0;
            for block in &assistant.content {
                match block {
                    AssistantBlock::Text(content) => chars += char_units(&content.text),
                    AssistantBlock::Thinking(content) => chars += char_units(&content.thinking),
                    AssistantBlock::ToolCall(call) => {
                        chars +=
                            char_units(&call.name) + safe_json_stringify_length(&call.arguments);
                    }
                }
            }
            tokens_for_chars(chars)
        }
        AgentMessage::Standard(Message::ToolResult(result)) => {
            tokens_for_chars(tool_result_content_chars(&result.content))
        }
        AgentMessage::Custom(custom) => {
            let chars = match custom.role.as_str() {
                "custom" => custom
                    .field("content")
                    .map_or(0, text_and_image_content_chars),
                "bashExecution" => {
                    let command = custom
                        .field("command")
                        .and_then(JsonValue::as_str)
                        .unwrap_or_default();
                    let output = custom
                        .field("output")
                        .and_then(JsonValue::as_str)
                        .unwrap_or_default();
                    char_units(command) + char_units(output)
                }
                "branchSummary" | "compactionSummary" => custom
                    .field("summary")
                    .and_then(JsonValue::as_str)
                    .map_or(0, char_units),
                // Upstream's fall-through: a role the switch does not know
                // estimates to zero.
                _ => 0,
            };
            tokens_for_chars(chars)
        }
    }
}

/// Calculate total context tokens from provider usage, upstream's
/// `calculateContextTokens`: the reported total when present, the
/// component sum otherwise.
#[must_use]
pub const fn calculate_context_tokens(usage: &Usage) -> u64 {
    if usage.total_tokens > 0 {
        usage.total_tokens
    } else {
        usage.input + usage.output + usage.cache_read + usage.cache_write
    }
}

/// The usage of one settled assistant message, upstream's
/// `getAssistantUsage`: aborted and error responses carry no usage, and a
/// zero total counts as none.
fn get_assistant_usage(message: &AgentMessage) -> Option<Usage> {
    let AgentMessage::Standard(Message::Assistant(assistant)) = message else {
        return None;
    };
    if assistant.stop_reason == StopReason::Aborted || assistant.stop_reason == StopReason::Error {
        return None;
    }
    if calculate_context_tokens(&assistant.usage) > 0 {
        Some(assistant.usage)
    } else {
        None
    }
}

/// Return usage from the last valid assistant message in session entries,
/// upstream's `getLastAssistantUsage`.
#[must_use]
pub fn get_last_assistant_usage(entries: &[Entry]) -> Option<Usage> {
    entries.iter().rev().find_map(|entry| match entry {
        Entry::Message { body, .. } => get_assistant_usage(&body.message),
        _ => None,
    })
}

fn get_last_assistant_usage_info(messages: &[AgentMessage]) -> Option<(Usage, usize)> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| get_assistant_usage(message).map(|usage| (usage, index)))
}

/// Estimate context tokens for messages using provider usage when
/// available, upstream's `estimateContextTokens`.
///
/// The last usage block's total plus the trailing messages' estimates, or
/// the whole list's estimate when no usage block exists.
#[must_use]
pub fn estimate_context_tokens(messages: &[AgentMessage]) -> ContextUsageEstimate {
    let Some((usage, index)) = get_last_assistant_usage_info(messages) else {
        let estimated = messages.iter().map(estimate_tokens).sum();
        return ContextUsageEstimate {
            tokens: estimated,
            usage_tokens: 0,
            trailing_tokens: estimated,
            last_usage_index: None,
        };
    };

    let usage_tokens = calculate_context_tokens(&usage);
    let trailing_tokens = messages[index + 1..].iter().map(estimate_tokens).sum();

    ContextUsageEstimate {
        tokens: usage_tokens + trailing_tokens,
        usage_tokens,
        trailing_tokens,
        last_usage_index: Some(index),
    }
}

/// Return whether context usage exceeds the configured compaction
/// threshold, upstream's `shouldCompact`.
#[must_use]
pub const fn should_compact(
    context_tokens: u64,
    context_window: u64,
    settings: &CompactionSettings,
) -> bool {
    if !settings.enabled {
        return false;
    }
    // The difference runs in i64, JS's number domain: a window below the
    // reserve (upstream's negative difference) still reports `true` for any
    // non-negative token count.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "token counts are far below i64::MAX; the i64 difference preserves JS's signed comparison when the window is below the reserve"
    )]
    let threshold = context_window as i64 - settings.reserve_tokens as i64;
    #[expect(
        clippy::cast_possible_wrap,
        reason = "token counts are far below i64::MAX; the comparison is sign-preserving"
    )]
    {
        context_tokens as i64 > threshold
    }
}

fn find_valid_cut_points(entries: &[Entry], start_index: usize, end_index: usize) -> Vec<usize> {
    let mut cut_points: Vec<usize> = Vec::new();
    for (offset, entry) in entries[start_index..end_index].iter().enumerate() {
        let index = start_index + offset;
        match entry {
            Entry::Message { body, .. } => match &body.message {
                AgentMessage::Standard(Message::ToolResult(_)) => {}
                AgentMessage::Standard(_) => cut_points.push(index),
                AgentMessage::Custom(custom) => match custom.role.as_str() {
                    "bashExecution" | "custom" | "branchSummary" | "compactionSummary" => {
                        cut_points.push(index);
                    }
                    _ => {}
                },
            },
            Entry::Compaction { .. } | Entry::Custom { .. } => {}
            Entry::BranchSummary { .. } => cut_points.push(index),
        }
    }
    cut_points
}

/// Find the user-visible message that starts the turn containing an entry,
/// upstream's `findTurnStartIndex`.
///
/// The nearest preceding branch summary, user message, or bash execution
/// at or before `entry_index`, stopping at `start_index`.
#[must_use]
pub fn find_turn_start_index(
    entries: &[Entry],
    entry_index: usize,
    start_index: usize,
) -> Option<usize> {
    for index in (start_index..=entry_index).rev() {
        match &entries[index] {
            Entry::BranchSummary { .. } => return Some(index),
            Entry::Message { body, .. } => match &body.message {
                AgentMessage::Standard(Message::User(_)) => return Some(index),
                AgentMessage::Custom(custom) if custom.role == "bashExecution" => {
                    return Some(index);
                }
                _ => {}
            },
            _ => {}
        }
    }
    None
}

/// Find the compaction cut point that keeps approximately the requested
/// recent-token budget, upstream's `findCutPoint`.
#[must_use]
pub fn find_cut_point(
    entries: &[Entry],
    start_index: usize,
    end_index: usize,
    keep_recent_tokens: u64,
) -> CutPointResult {
    let cut_points = find_valid_cut_points(entries, start_index, end_index);

    if cut_points.is_empty() {
        return CutPointResult {
            first_kept_entry_index: start_index,
            turn_start_index: None,
        };
    }
    let mut accumulated_tokens = 0;
    let mut cut_index = cut_points[0];

    for index in (start_index..end_index).rev() {
        let Entry::Message { body, .. } = &entries[index] else {
            continue;
        };
        accumulated_tokens += estimate_tokens(&body.message);
        if accumulated_tokens >= keep_recent_tokens {
            cut_index = cut_points
                .iter()
                .copied()
                .find(|&point| point >= index)
                .unwrap_or(cut_index);
            break;
        }
    }
    while cut_index > start_index {
        match &entries[cut_index - 1] {
            Entry::Compaction { .. } | Entry::Message { .. } => break,
            _ => cut_index -= 1,
        }
    }
    let cut_entry = &entries[cut_index];
    let is_user_message = matches!(
        cut_entry,
        Entry::Message { body, .. }
            if matches!(body.message, AgentMessage::Standard(Message::User(_)))
    );
    let turn_start_index = if is_user_message {
        None
    } else {
        find_turn_start_index(entries, cut_index, start_index)
    };

    CutPointResult {
        first_kept_entry_index: cut_index,
        turn_start_index,
    }
}

/// The system prompt every summarization request carries, upstream's
/// `SUMMARIZATION_SYSTEM_PROMPT`.
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.\n\nUse this EXACT format:\n\n## Goal\n[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements mentioned by user]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Current work]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [Ordered list of what should happen next]\n\n## Critical Context\n- [Any data, examples, or references needed to continue]\n- [Or \"(none)\" if not applicable]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

const UPDATE_SUMMARIZATION_PROMPT: &str = "The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.\n\nUpdate the existing structured summary with new information. RULES:\n- PRESERVE all existing information from the previous summary\n- ADD new progress, decisions, and context from the new messages\n- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed\n- UPDATE \"Next Steps\" based on what was accomplished\n- PRESERVE exact file paths, function names, and error messages\n- If something is no longer relevant, you may remove it\n\nUse this EXACT format:\n\n## Goal\n[Preserve existing goals, add new ones if the task expanded]\n\n## Constraints & Preferences\n- [Preserve existing, add new ones discovered]\n\n## Progress\n### Done\n- [x] [Include previously done items AND newly completed items]\n\n### In Progress\n- [ ] [Current work - update based on progress]\n\n### Blocked\n- [Current blockers - remove if resolved]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale] (preserve all previous, add new)\n\n## Next Steps\n1. [Update based on current state]\n\n## Critical Context\n- [Preserve important context, add new if needed]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = "This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.\n\nSummarize the prefix to provide context for the retained suffix:\n\n## Original Request\n[What did the user ask for in this turn?]\n\n## Early Progress\n- [Key decisions and work done in the prefix]\n\n## Context for Suffix\n- [Information needed to understand the retained recent work]\n\nBe concise. Focus on what's needed to understand the kept suffix.";

/// Options for one summarization request, upstream's
/// `createSummaryRequestOptions`.
///
/// The caller's options with the standalone-request isolation applied —
/// the context's abort signal, the context's telemetry parent, no cache
/// writes that cannot be reused, and a fresh session id when absent.
#[must_use]
pub fn create_summary_request_options(
    options: SimpleStreamOptions,
    context: &Context,
) -> SimpleStreamOptions {
    SimpleStreamOptions {
        transport_options: TransportOptions {
            signal: request_signal(context),
            ..options.transport_options.clone()
        },
        telemetry_context: Some(get_telemetry_context(context)),
        cache_retention: Some(CacheRetention::None),
        // Upstream throws when uuidv7 fails; the only failure is the format's
        // timestamp range, which the process clock cannot reach, so the id is
        // silently absent instead.
        session_id: options.session_id.clone().or_else(|| uuidv7(None).ok()),
        ..options
    }
}

/// Complete a simple summarization request under the retry policy,
/// upstream's `completeSimpleWithRetries`.
///
/// Summaries are standalone requests, so the options isolate routing and
/// avoid cache writes that cannot be reused
/// ([`create_summary_request_options`]).
pub async fn complete_simple_with_retries(
    models: &Models,
    model: &Model,
    ai_context: &AiContext,
    options: &SimpleStreamOptions,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&RetryCallbacks>,
    context: &Context,
) -> AssistantMessage {
    let request_options = WithTransforms {
        options: create_summary_request_options(options.clone(), context),
        transform_headers: None,
    };
    retry_assistant_call(
        || models.complete_simple(model, ai_context, Some(&request_options)),
        retry,
        request_options.options.transport_options.signal.as_ref(),
        callbacks,
    )
    .await
}

/// Generate or update a conversation summary for compaction, upstream's
/// `generateSummary`: [`generate_summary_with_usage`] without the usage.
///
/// # Errors
/// The summarization request's aborted or failed settlement.
#[expect(
    clippy::too_many_arguments,
    reason = "the positional signature mirrors upstream's generateSummary*(); the WithRequest variant collapses the tail into options"
)]
pub async fn generate_summary(
    current_messages: &[AgentMessage],
    models: &Models,
    model: &Model,
    reserve_tokens: u64,
    custom_instructions: Option<String>,
    previous_summary: Option<String>,
    thinking_level: Option<ThinkingLevel>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&RetryCallbacks>,
    context: &Context,
) -> Result<String, CompactionError> {
    generate_summary_with_usage(
        current_messages,
        models,
        model,
        reserve_tokens,
        custom_instructions,
        previous_summary,
        thinking_level,
        retry,
        callbacks,
        context,
    )
    .await
    .map(|summary| summary.text)
}

/// Generate or update a conversation summary and return its provider
/// usage, upstream's `generateSummaryWithUsage`.
///
/// # Errors
/// The summarization request's aborted or failed settlement.
#[expect(
    clippy::too_many_arguments,
    reason = "the positional signature mirrors upstream's generateSummary*(); the WithRequest variant collapses the tail into options"
)]
pub async fn generate_summary_with_usage(
    current_messages: &[AgentMessage],
    models: &Models,
    model: &Model,
    reserve_tokens: u64,
    custom_instructions: Option<String>,
    previous_summary: Option<String>,
    thinking_level: Option<ThinkingLevel>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&RetryCallbacks>,
    context: &Context,
) -> Result<SummaryWithUsage, CompactionError> {
    let options = SummaryGenerationOptions {
        model: model.clone(),
        reserve_tokens,
        custom_instructions,
        previous_summary,
        thinking_level,
    };
    let request = models_summary_request(models, model, retry, callbacks);
    generate_summary_with_request(current_messages, &options, &request, context).await
}

/// The output-token cap one summary request carries: the fraction of the
/// reserve clamped to the model's output cap, upstream's
/// `Math.min(Math.floor(fraction * reserveTokens), model.maxTokens > 0 ? model.maxTokens : Infinity)`.
fn clamped_output_tokens(fraction: f64, reserve_tokens: u64, model_max_tokens: u64) -> u64 {
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "the fraction-of-reserve floor reproduces the JS number cap; a non-negative floor is non-negative and fits u64"
    )]
    let fraction_of_reserve = (fraction * reserve_tokens as f64).floor() as u64;
    if model_max_tokens > 0 {
        fraction_of_reserve.min(model_max_tokens)
    } else {
        fraction_of_reserve
    }
}

/// The request options one summary call sends, upstream's inline
/// `completionOptions`: the clamped output cap, plus the reasoning level
/// only for reasoning models with thinking enabled.
fn summary_completion_options(
    max_tokens: u64,
    model: &Model,
    thinking_level: Option<ThinkingLevel>,
) -> SimpleStreamOptions {
    let reasoning = thinking_level
        .filter(|level| *level != ThinkingLevel::Off && model.reasoning)
        .and_then(stream_reasoning);
    SimpleStreamOptions {
        max_tokens: Some(max_tokens),
        reasoning,
        ..SimpleStreamOptions::default()
    }
}

/// The one-message user turn the summarization prompts send, upstream's
/// `summarizationMessages`.
fn summarization_messages(prompt_text: String) -> Vec<Message> {
    vec![Message::User(UserMessage {
        content: UserContent::Blocks(vec![UserBlock::Text(TextContent {
            text: prompt_text,
            text_signature: None,
        })]),
        timestamp: now_ms(),
    })]
}

/// The aborted/failed settlement one summarization response carries,
/// upstream's per-site stop-reason checks with their verbs.
///
/// `Err` carries the `CompactionError`; the caller's error-code pair and
/// verb ride the parameters.
fn summarization_settlement(
    response: &AssistantMessage,
    aborted_message: &str,
    failed_prefix: &str,
) -> Result<(), CompactionError> {
    if response.stop_reason == StopReason::Aborted {
        return Err(CompactionError::new(
            CompactionErrorCode::Aborted,
            response
                .error_message
                .clone()
                .unwrap_or_else(|| aborted_message.to_owned()),
            None,
        ));
    }
    if response.stop_reason == StopReason::Error {
        return Err(CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            format!(
                "{failed_prefix}: {}",
                response
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "Unknown error".to_owned())
            ),
            None,
        ));
    }
    Ok(())
}

/// Generate one summary through a caller-owned one-request boundary,
/// upstream's `generateSummaryWithRequest`.
///
/// # Errors
/// [`CompactionErrorCode::Aborted`] when the request aborted (upstream's
/// `"Summarization aborted"` default), [`CompactionErrorCode::SummarizationFailed`]
/// when it failed (`"Summarization failed: ..."`).
pub async fn generate_summary_with_request(
    current_messages: &[AgentMessage],
    options: &SummaryGenerationOptions,
    request: &SummaryRequest,
    context: &Context,
) -> Result<SummaryWithUsage, CompactionError> {
    let SummaryGenerationOptions {
        model,
        reserve_tokens,
        custom_instructions,
        previous_summary,
        thinking_level,
    } = options;
    let max_tokens = clamped_output_tokens(0.8, *reserve_tokens, model.max_tokens);
    let mut base_prompt = if previous_summary.is_some() {
        UPDATE_SUMMARIZATION_PROMPT
    } else {
        SUMMARIZATION_PROMPT
    }
    .to_owned();
    if let Some(custom_instructions) = custom_instructions {
        base_prompt = format!("{base_prompt}\n\nAdditional focus: {custom_instructions}");
    }
    let llm_messages = convert_to_llm(current_messages);
    let conversation_text = serialize_conversation(&llm_messages);
    let mut prompt_text = format!("<conversation>\n{conversation_text}\n</conversation>\n\n");
    if let Some(previous_summary) = previous_summary {
        let _ = write!(
            prompt_text,
            "<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        );
    }
    prompt_text += &base_prompt;

    let completion_options = summary_completion_options(max_tokens, model, *thinking_level);

    let response = request(
        &AiContext {
            system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_owned()),
            messages: summarization_messages(prompt_text),
            tools: None,
        },
        &create_summary_request_options(completion_options, context),
        context,
    )
    .await;
    summarization_settlement(&response, "Summarization aborted", "Summarization failed")?;

    Ok(SummaryWithUsage {
        text: content_text(&response.content.as_slice()),
        usage: response.usage,
    })
}

/// The file-operation extraction one preparation runs, upstream's
/// `extractFileOperations`: the previous compaction entry's details seed
/// the accumulator, then the summarized messages' tool calls add to it.
fn extract_file_operations(
    messages: &[AgentMessage],
    entries: &[Entry],
    prev_compaction_index: Option<usize>,
) -> FileOpsAccumulator {
    let mut file_ops = FileOpsAccumulator::default();
    if let Some(index) = prev_compaction_index
        && let Entry::Compaction { body, .. } = &entries[index]
        && let Some(details) = body.details.as_ref().filter(|details| details.is_object())
    {
        for path in details
            .get("readFiles")
            .and_then(JsonValue::as_array)
            .into_iter()
            .flatten()
            .filter_map(JsonValue::as_str)
        {
            file_ops.read.insert(path.to_owned());
        }
        for path in details
            .get("modifiedFiles")
            .and_then(JsonValue::as_array)
            .into_iter()
            .flatten()
            .filter_map(JsonValue::as_str)
        {
            file_ops.edited.insert(path.to_owned());
        }
    }
    for message in messages {
        extract_file_ops_from_message(message, &mut file_ops);
    }
    file_ops
}

/// The `AgentMessage` one transcript entry contributes to a summary range,
/// upstream's `getMessageFromEntry`: message entries carry their message,
/// branch summaries and compactions rebuild their context-message shapes,
/// custom entries contribute nothing.
fn get_message_from_entry(entry: &Entry) -> Option<AgentMessage> {
    match entry {
        Entry::Message { body, .. } => Some(body.message.clone()),
        Entry::BranchSummary { body, .. } => Some(custom_message_agent_message(
            &create_branch_summary_message(
                body.summary.clone(),
                body.from_id.clone(),
                entry.timestamp().into(),
            ),
        )),
        Entry::Compaction { body, .. } => Some(custom_message_agent_message(
            &create_compaction_summary_message(
                body.summary.clone(),
                body.tokens_before,
                entry.timestamp().into(),
            ),
        )),
        Entry::Custom { .. } => None,
    }
}

/// [`get_message_from_entry`] with compaction entries skipped, upstream's
/// `getMessageFromEntryForCompaction`: an inner compaction's summary
/// already stands in the retained tail, so a further compaction range
/// never re-summarizes it.
fn get_message_from_entry_for_compaction(entry: &Entry) -> Option<AgentMessage> {
    if matches!(entry, Entry::Compaction { .. }) {
        return None;
    }
    get_message_from_entry(entry)
}

/// The custom-role `AgentMessage` a typed message constructor produces,
/// upstream's object literal: the wire object's `role`/`timestamp` lift to
/// the variant's fields and the rest rides `data` verbatim.
pub(crate) fn custom_message_agent_message(message: &impl serde::Serialize) -> AgentMessage {
    let mut wire = serde_json::to_value(message)
        .ok()
        .and_then(|wire| wire.as_object().cloned())
        .unwrap_or_default();
    let role = wire
        .get("role")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_owned();
    let timestamp = wire
        .get("timestamp")
        .and_then(JsonValue::as_i64)
        .unwrap_or_default();
    wire.remove("role");
    wire.remove("timestamp");
    AgentMessage::Custom(crate::types::CustomAgentMessage {
        role,
        timestamp,
        data: wire,
    })
}

/// The compaction-skipping message extraction one entry range runs,
/// upstream's three `getMessageFromEntryForCompaction` loops.
fn range_messages(entries: &[Entry], from: usize, to: usize) -> Vec<AgentMessage> {
    entries[from..to]
        .iter()
        .filter_map(get_message_from_entry_for_compaction)
        .collect()
}

/// The virtual message entries one previous compaction's retained tail
/// re-enters the compactable range as, upstream's `virtualRetainedEntries`:
/// the chain hangs off the compaction entry and reuses its sequence.
fn virtual_retained_entries(id: &str, seq: u64, retained_tail: &[AgentMessage]) -> Vec<Entry> {
    retained_tail
        .iter()
        .enumerate()
        .map(|(index, message)| Entry::Message {
            id: format!("{id}:retained:{index}"),
            parent_id: Some(if index == 0 {
                id.to_owned()
            } else {
                format!("{id}:retained:{}", index - 1)
            }),
            seq,
            timestamp: message_timestamp(message),
            body: Box::new(MessageEntry {
                message: message.clone(),
                terminate: None,
            }),
        })
        .collect()
}

/// Prepare session entries for compaction, or return `None` when
/// compaction is not applicable, upstream's `prepareCompaction`.
///
/// The latest compaction's summary seeds an iterative update, its retained
/// tail re-enters the compactable range as virtual entries, and the cut
/// point splits the range into summarized history, optional turn prefix,
/// and retained tail.
///
/// # Errors
/// Never; upstream types the failure for its callers' signature. The port
/// carries the shape.
pub fn prepare_compaction(
    path_entries: &[Entry],
    settings: CompactionSettings,
) -> Result<Option<CompactionPreparation>, CompactionError> {
    if path_entries.is_empty() || matches!(path_entries.last(), Some(Entry::Compaction { .. })) {
        return Ok(None);
    }

    let prev_compaction_index = path_entries
        .iter()
        .rposition(|entry| matches!(entry, Entry::Compaction { .. }));
    // The scan above matched this index on the compaction variant; the
    // defensive arm restates the same predicate and never fires.
    let prev_compaction = prev_compaction_index.and_then(|index| match &path_entries[index] {
        Entry::Compaction { id, seq, body, .. } => Some((index, id.as_str(), *seq, body)),
        _ => None,
    });

    let (previous_summary, compactable_entries) = match prev_compaction {
        Some((prev_compaction_index, id, seq, compaction_body)) => {
            let mut combined = virtual_retained_entries(id, seq, &compaction_body.retained_tail);
            combined.extend(path_entries[prev_compaction_index + 1..].iter().cloned());
            (Some(compaction_body.summary.clone()), combined)
        }
        None => (None, path_entries.to_vec()),
    };
    let boundary_end = compactable_entries.len();

    #[expect(
        clippy::cast_possible_wrap,
        reason = "token estimates are far below i64::MAX; the durable preparation's tokens_before is i64"
    )]
    let tokens_before = estimate_context_tokens(
        &build_context_entries(path_entries)
            .iter()
            .flat_map(session_entry_to_context_messages)
            .collect::<Vec<_>>(),
    )
    .tokens as i64;

    let cut_point = find_cut_point(
        &compactable_entries,
        0,
        boundary_end,
        settings.keep_recent_tokens,
    );
    let history_end = cut_point
        .turn_start_index
        .unwrap_or(cut_point.first_kept_entry_index);
    let messages_to_summarize = range_messages(&compactable_entries, 0, history_end);
    let turn_prefix_messages =
        cut_point
            .turn_start_index
            .map_or_else(Vec::new, |turn_start_index| {
                range_messages(
                    &compactable_entries,
                    turn_start_index,
                    cut_point.first_kept_entry_index,
                )
            });
    let retained_tail = range_messages(
        &compactable_entries,
        cut_point.first_kept_entry_index,
        boundary_end,
    );
    let mut file_ops =
        extract_file_operations(&messages_to_summarize, path_entries, prev_compaction_index);
    if cut_point.is_split_turn() {
        for message in &turn_prefix_messages {
            extract_file_ops_from_message(message, &mut file_ops);
        }
    }

    Ok(Some(CompactionPreparation {
        messages_to_summarize,
        turn_prefix_messages,
        retained_tail,
        is_split_turn: cut_point.is_split_turn(),
        tokens_before,
        previous_summary,
        file_ops: durable_file_operations(&file_ops),
        settings,
    }))
}

/// Generate compaction summary data from prepared session history,
/// upstream's `compact`.
///
/// The history summary, a split turn's prefix summary, and the
/// file-operation tags combine into the persisted result.
///
/// # Errors
/// Either summarization request's aborted or failed settlement.
#[expect(
    clippy::too_many_arguments,
    reason = "the positional signature mirrors upstream's compact(); the WithRequest variant collapses the model/retry/callbacks tail"
)]
pub async fn compact(
    preparation: &CompactionPreparation,
    models: &Models,
    model: &Model,
    custom_instructions: Option<String>,
    thinking_level: Option<ThinkingLevel>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&RetryCallbacks>,
    context: &Context,
) -> Result<CompactResult, CompactionError> {
    let options = CompactGenerationOptions {
        model: model.clone(),
        custom_instructions,
        thinking_level,
    };
    let request = models_summary_request(models, model, retry, callbacks);
    compact_with_request(preparation, &options, &request, context).await
}

/// Generate compaction data through a caller-owned boundary for each
/// provider request, upstream's `compactWithRequest`.
///
/// # Errors
/// Either summarization request's aborted or failed settlement; the split
/// turn's turn-prefix messages carry their own verbs
/// (`"Turn prefix summarization ..."`) upstream-verbatim.
pub async fn compact_with_request(
    preparation: &CompactionPreparation,
    options: &CompactGenerationOptions,
    request: &SummaryRequest,
    context: &Context,
) -> Result<CompactResult, CompactionError> {
    let CompactionPreparation {
        messages_to_summarize,
        turn_prefix_messages,
        retained_tail,
        is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
        settings,
    } = preparation;

    let (summary, summary_usage) = if *is_split_turn && !turn_prefix_messages.is_empty() {
        let (history_text, history_usage) = if messages_to_summarize.is_empty() {
            ("No prior history.".to_owned(), None)
        } else {
            let history_result = generate_summary_with_request(
                messages_to_summarize,
                &SummaryGenerationOptions {
                    model: options.model.clone(),
                    reserve_tokens: settings.reserve_tokens,
                    custom_instructions: options.custom_instructions.clone(),
                    previous_summary: previous_summary.clone(),
                    thinking_level: options.thinking_level,
                },
                request,
                context,
            )
            .await?;
            (history_result.text, Some(history_result.usage))
        };
        let turn_prefix_result = generate_turn_prefix_summary(
            turn_prefix_messages,
            settings.reserve_tokens,
            options,
            request,
            context,
        )
        .await?;
        (
            format!(
                "{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{}",
                turn_prefix_result.text
            ),
            history_usage.map_or(turn_prefix_result.usage, |history_usage| {
                add_usage(history_usage, turn_prefix_result.usage)
            }),
        )
    } else {
        let summary_result = generate_summary_with_request(
            messages_to_summarize,
            &SummaryGenerationOptions {
                model: options.model.clone(),
                reserve_tokens: settings.reserve_tokens,
                custom_instructions: options.custom_instructions.clone(),
                previous_summary: previous_summary.clone(),
                thinking_level: options.thinking_level,
            },
            request,
            context,
        )
        .await?;
        (summary_result.text, summary_result.usage)
    };

    let file_lists = compute_file_lists(file_ops);
    let summary = format!(
        "{summary}{}",
        format_file_operations(&file_lists.read_files, &file_lists.modified_files)
    );

    Ok(CompactResult {
        summary,
        tokens_before: *tokens_before,
        usage: Some(summary_usage),
        retained_tail: retained_tail.clone(),
        details: Some(serde_json::to_value(&file_lists).unwrap_or_default()),
    })
}

/// Generate the split turn's prefix summary, upstream's
/// `generateTurnPrefixSummary`: the same request shape with the
/// turn-prefix prompt and half the reserve as the output cap.
///
/// # Errors
/// [`CompactionErrorCode::Aborted`] when the request aborted (upstream's
/// `"Turn prefix summarization aborted"` default),
/// [`CompactionErrorCode::SummarizationFailed`] when it failed
/// (`"Turn prefix summarization failed: ..."`).
async fn generate_turn_prefix_summary(
    messages: &[AgentMessage],
    reserve_tokens: u64,
    options: &CompactGenerationOptions,
    request: &SummaryRequest,
    context: &Context,
) -> Result<SummaryWithUsage, CompactionError> {
    let max_tokens = clamped_output_tokens(0.5, reserve_tokens, options.model.max_tokens);
    let llm_messages = convert_to_llm(messages);
    let conversation_text = serialize_conversation(&llm_messages);
    let prompt_text = format!(
        "<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
    );
    let completion_options =
        summary_completion_options(max_tokens, &options.model, options.thinking_level);

    let response = request(
        &AiContext {
            system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_owned()),
            messages: summarization_messages(prompt_text),
            tools: None,
        },
        &create_summary_request_options(completion_options, context),
        context,
    )
    .await;
    summarization_settlement(
        &response,
        "Turn prefix summarization aborted",
        "Turn prefix summarization failed",
    )?;

    Ok(SummaryWithUsage {
        text: content_text(&response.content.as_slice()),
        usage: response.usage,
    })
}

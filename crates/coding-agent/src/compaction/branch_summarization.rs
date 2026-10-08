//! Branch summarization for tree navigation, upstream
//! `src/core/compaction/branch-summarization.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! When navigating to a different point in the session tree, this generates
//! a summary of the branch being left so context isn't lost.

use pi_agent_core::types::{AgentMessage, StreamFn};
use pi_ai::types::{
    AssistantBlock, Model, ProviderEnv, ProviderHeaders, SimpleStreamOptions, StopReason,
    TransportOptions, Usage,
};
use pi_ai::utils::retry::{RetryCallbacks, RetryPolicy};
use pi_ai::utils::text::content_text;
use tokio_util::sync::CancellationToken;

use crate::messages::{
    convert_to_llm, create_branch_summary_message, create_compaction_summary_message,
    create_custom_message,
};
use crate::session_manager::SessionManager;
use crate::session_manager::entries::SessionEntry;
use crate::session_manager::file_entry::FileEntry;

use super::compaction::{
    build_summarization_context, complete_summarization, estimate_tokens, get_summarization_failure,
};
use super::utils::{
    FileOperations, compute_file_lists, create_file_ops, extract_file_ops_from_message,
    format_file_operations, serialize_conversation,
};

// ============================================================================
// Types
// ============================================================================

/// The branch summarization's outcome, upstream's `BranchSummaryResult`.
/// Exactly one of the terminal fields carries the outcome.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BranchSummaryResult {
    /// The generated summary, preamble and file lists included.
    pub summary: Option<String>,
    /// Usage from the LLM call that generated the summary.
    pub usage: Option<Usage>,
    /// Files only read in the summarized branch.
    pub read_files: Option<Vec<String>>,
    /// Files modified in the summarized branch.
    pub modified_files: Option<Vec<String>>,
    /// True when the request was aborted.
    pub aborted: bool,
    /// The failure message when the summary could not be produced.
    pub error: Option<String>,
}

/// Details stored in `BranchSummaryEntry.details` for file tracking, upstream's
/// `BranchSummaryDetails`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryDetails {
    /// Files only read in the summarized branch.
    pub read_files: Vec<String>,
    /// Files modified in the summarized branch.
    pub modified_files: Vec<String>,
}

/// The entries prepared for summarization, upstream's `BranchPreparation`.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchPreparation {
    /// Messages extracted for summarization, in chronological order.
    pub messages: Vec<AgentMessage>,
    /// File operations extracted from tool calls.
    pub file_ops: FileOperations,
    /// Total estimated tokens in messages.
    pub total_tokens: u64,
}

/// The entries collected for one branch-summary navigation, upstream's
/// `CollectEntriesResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectEntriesResult {
    /// Entries to summarize, in chronological order.
    pub entries: Vec<SessionEntry>,
    /// Common ancestor between old and new position, if any.
    pub common_ancestor_id: Option<String>,
}

/// The options one branch-summary generation takes, upstream's
/// `GenerateBranchSummaryOptions`.
pub struct GenerateBranchSummaryOptions<'a> {
    /// Model to use for summarization.
    pub model: &'a Model,
    /// API key for the model.
    pub api_key: Option<&'a str>,
    /// Request headers for the model.
    pub headers: Option<&'a ProviderHeaders>,
    /// Provider-scoped environment values for the model.
    pub env: Option<&'a ProviderEnv>,
    /// Abort signal for cancellation.
    pub signal: &'a CancellationToken,
    /// Optional custom instructions for summarization.
    pub custom_instructions: Option<&'a str>,
    /// If true, customInstructions replaces the default prompt instead of
    /// being appended.
    pub replace_instructions: bool,
    /// Tokens reserved when selecting branch history; upstream's default is
    /// 16384.
    pub reserve_tokens: Option<i64>,
    /// Optional session stream function. Used to preserve SDK request
    /// behavior without mutating agent state.
    pub stream_fn: Option<&'a StreamFn>,
    /// Retry policy for transient summarization errors. Reuses coding-agent's
    /// `settings.retry`.
    pub retry: Option<&'a RetryPolicy>,
    /// Optional callbacks for retry reporting (e.g. TUI retry indicators).
    pub callbacks: Option<&'a RetryCallbacks>,
}

impl std::fmt::Debug for GenerateBranchSummaryOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerateBranchSummaryOptions")
            .field("model", &self.model.id)
            .field("api_key", &self.api_key)
            .field("headers", &self.headers)
            .field("env", &self.env)
            .field("custom_instructions", &self.custom_instructions)
            .field("replace_instructions", &self.replace_instructions)
            .field("reserve_tokens", &self.reserve_tokens)
            .field("stream_fn", &self.stream_fn.is_some())
            .field("retry", &self.retry)
            .field("callbacks", &self.callbacks.is_some())
            .finish()
    }
}

// ============================================================================
// Entry Collection
// ============================================================================

/// Collect entries that should be summarized when navigating from one
/// position to another, upstream's `collectEntriesForBranchSummary`.
///
/// Walks from oldLeafId back to the common ancestor with targetId, collecting
/// entries along the way. Does NOT stop at compaction boundaries — those are
/// included and their summaries become context.
///
/// The walk continues through raw (untyped) file entries, whose parent links
/// the tree honors; only typed entries collect, since the summarizer projects
/// the typed union.
#[must_use]
pub fn collect_entries_for_branch_summary(
    session: &SessionManager,
    old_leaf_id: Option<&str>,
    target_id: &str,
) -> CollectEntriesResult {
    // If no old position, nothing to summarize.
    let Some(old_leaf_id) = old_leaf_id else {
        return CollectEntriesResult {
            entries: Vec::new(),
            common_ancestor_id: None,
        };
    };

    // Find common ancestor (deepest node that's on both paths).
    let old_path: std::collections::HashSet<String> = session
        .get_branch(Some(old_leaf_id))
        .iter()
        .filter_map(|entry| entry.entry_id().map(str::to_owned))
        .collect();
    let target_path = session.get_branch(Some(target_id));

    // targetPath is root-first, so iterate backwards to find deepest common
    // ancestor.
    let mut common_ancestor_id: Option<String> = None;
    for entry in target_path.iter().rev() {
        if let Some(id) = entry.entry_id()
            && old_path.contains(id)
        {
            common_ancestor_id = Some(id.to_owned());
            break;
        }
    }

    // Collect entries from old leaf back to common ancestor.
    let mut entries: Vec<SessionEntry> = Vec::new();
    let mut current: Option<String> = Some(old_leaf_id.to_owned());

    while let Some(id) = current {
        if common_ancestor_id.as_deref() == Some(id.as_str()) {
            break;
        }
        let Some(entry) = session.get_entry(&id) else {
            break;
        };
        if let FileEntry::Entry(typed) = entry {
            entries.push(typed.clone());
        }
        current = entry.entry_parent_id().map(str::to_owned);
    }

    // Reverse to get chronological order.
    entries.reverse();

    CollectEntriesResult {
        entries,
        common_ancestor_id,
    }
}

// ============================================================================
// Entry to Message Conversion
// ============================================================================

/// Extract an `AgentMessage` from a session entry, upstream's
/// `getMessageFromEntry`. Similar to compaction's extraction but also handles
/// compaction entries.
fn get_message_from_entry(entry: &SessionEntry) -> Option<AgentMessage> {
    match entry {
        SessionEntry::Message(message_entry) => {
            // Skip tool results - context is in assistant's tool call.
            match &message_entry.message {
                Some(AgentMessage::Standard(pi_ai::types::Message::ToolResult(_))) | None => None,
                Some(message) => Some(message.clone()),
            }
        }
        SessionEntry::CustomMessage(custom) => Some(AgentMessage::Custom(create_custom_message(
            &custom.custom_type,
            custom
                .content
                .clone()
                .unwrap_or(pi_ai::types::UserContent::Blocks(Vec::new())),
            custom.display,
            custom.details.clone(),
            &custom.base.timestamp,
        ))),
        SessionEntry::BranchSummary(branch) => Some(AgentMessage::Custom(
            create_branch_summary_message(&branch.summary, &branch.from_id, &branch.base.timestamp),
        )),
        SessionEntry::Compaction(compaction) => {
            Some(AgentMessage::Custom(create_compaction_summary_message(
                &compaction.summary,
                compaction.tokens_before,
                &compaction.base.timestamp,
            )))
        }
        // These don't contribute to conversation content.
        SessionEntry::ThinkingLevelChange(_)
        | SessionEntry::ModelChange(_)
        | SessionEntry::Custom(_)
        | SessionEntry::Label(_)
        | SessionEntry::SessionInfo(_) => None,
    }
}

/// Prepare entries for summarization with token budget, upstream's
/// `prepareBranchEntries`.
///
/// Walks entries from NEWEST to OLDEST, adding messages until we hit the
/// token budget. This ensures we keep the most recent context when the branch
/// is too long. File operations collect from all entries first (cumulative
/// tracking from nested branch summaries, pi-generated ones only), then from
/// the assistant tool calls that fit.
#[expect(
    clippy::cast_precision_loss,
    reason = "the 0.9 summary-fit margin restates upstream's number comparison"
)]
#[must_use]
pub fn prepare_branch_entries(entries: &[SessionEntry], token_budget: i64) -> BranchPreparation {
    let mut messages: Vec<AgentMessage> = Vec::new();
    let mut file_ops = create_file_ops();
    let mut total_tokens: u64 = 0;

    // First pass: collect file ops from ALL entries (even if they don't fit
    // in token budget). This ensures we capture cumulative file tracking from
    // nested branch summaries. Only extract from pi-generated summaries, not
    // extension-generated ones.
    for entry in entries {
        if let SessionEntry::BranchSummary(branch) = entry {
            if branch.from_hook == Some(true) {
                continue;
            }
            if let Some(details) = &branch.details {
                // The detail lists read tolerantly, upstream's
                // `Array.isArray` skips.
                if let Some(read_files) = details
                    .get("readFiles")
                    .and_then(serde_json::Value::as_array)
                {
                    for file in read_files.iter().filter_map(serde_json::Value::as_str) {
                        file_ops.read.insert(file.to_owned());
                    }
                }
                // Modified files go into edited for proper deduplication with
                // the written set.
                if let Some(modified_files) = details
                    .get("modifiedFiles")
                    .and_then(serde_json::Value::as_array)
                {
                    for file in modified_files.iter().filter_map(serde_json::Value::as_str) {
                        file_ops.edited.insert(file.to_owned());
                    }
                }
            }
        }
    }

    // Second pass: walk from newest to oldest, adding messages until token
    // budget.
    for entry in entries.iter().rev() {
        let Some(message) = get_message_from_entry(entry) else {
            continue;
        };

        // Extract file ops from assistant messages (tool calls).
        extract_file_ops_from_message(&message, &mut file_ops);

        let tokens = estimate_tokens(&message);

        // Check budget before adding.
        if token_budget > 0
            && total_tokens.saturating_add(tokens) > u64::try_from(token_budget).unwrap_or(u64::MAX)
        {
            // If this is a summary entry, try to fit it anyway as it's
            // important context.
            if matches!(
                entry,
                SessionEntry::Compaction(_) | SessionEntry::BranchSummary(_)
            ) && (total_tokens as f64) < token_budget as f64 * 0.9
            {
                messages.push(message);
                total_tokens += tokens;
            }
            // Stop - we've hit the budget.
            break;
        }

        messages.push(message);
        total_tokens += tokens;
    }

    messages.reverse();
    BranchPreparation {
        messages,
        file_ops,
        total_tokens,
    }
}

// ============================================================================
// Summary Generation
// ============================================================================

const BRANCH_SUMMARY_PREAMBLE: &str = "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n";

const BRANCH_SUMMARY_PROMPT: &str = "Create a structured summary of this conversation branch for context when returning later.\n\nUse this EXACT format:\n\n## Goal\n[What was the user trying to accomplish in this branch?]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements mentioned]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Work that was started but not finished]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [What should happen next to continue this work]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

/// Generate a summary of abandoned branch entries, upstream's
/// `generateBranchSummary`.
pub async fn generate_branch_summary(
    entries: &[SessionEntry],
    options: &GenerateBranchSummaryOptions<'_>,
) -> BranchSummaryResult {
    // Token budget = context window minus reserved space for prompt +
    // response.
    let reserve_tokens = options.reserve_tokens.unwrap_or(16384);
    let context_window: i64 = if options.model.context_window == 0 {
        128_000
    } else {
        i64::try_from(options.model.context_window).unwrap_or(i64::MAX)
    };
    let token_budget = context_window - reserve_tokens;

    let preparation = prepare_branch_entries(entries, token_budget);

    if preparation.messages.is_empty() {
        return BranchSummaryResult {
            summary: Some("No content to summarize".to_owned()),
            ..BranchSummaryResult::default()
        };
    }

    // Transform to LLM-compatible messages, then serialize to text.
    // Serialization prevents the model from treating it as a conversation to
    // continue.
    let llm_messages = convert_to_llm(&preparation.messages);
    let conversation_text = serialize_conversation(&llm_messages);

    // Build prompt.
    let instructions = if options.replace_instructions && options.custom_instructions.is_some() {
        options.custom_instructions.unwrap_or_default().to_owned()
    } else if let Some(custom_instructions) = options.custom_instructions {
        format!("{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom_instructions}")
    } else {
        BRANCH_SUMMARY_PROMPT.to_owned()
    };
    let prompt_text =
        format!("<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}");

    let max_tokens: u64 = if options.model.max_tokens > 0 {
        4096.min(options.model.max_tokens)
    } else {
        4096
    };

    // Call LLM for summarization. Prefer the session stream function so SDK
    // request behavior (timeouts, retries, attribution headers) stays
    // consistent without running through agent state/events. Retried via
    // complete_summarization so transient stream drops reuse the configured
    // retry policy.
    let context = build_summarization_context(&prompt_text);
    let request_options = SimpleStreamOptions {
        transport_options: TransportOptions {
            signal: Some(options.signal.clone()),
            ..TransportOptions::default()
        },
        api_key: options.api_key.map(str::to_owned),
        headers: options.headers.cloned(),
        env: options.env.cloned(),
        max_tokens: Some(max_tokens),
        ..SimpleStreamOptions::default()
    };
    let response = complete_summarization(
        options.model,
        &context,
        &request_options,
        options.stream_fn,
        options.retry,
        options.callbacks,
    )
    .await;

    // Check if aborted or errored.
    if response.stop_reason == StopReason::Aborted {
        return BranchSummaryResult {
            aborted: true,
            ..BranchSummaryResult::default()
        };
    }
    if let Some(failure) = get_summarization_failure(&response, "Branch summarization") {
        return BranchSummaryResult {
            error: Some(failure),
            ..BranchSummaryResult::default()
        };
    }
    if response
        .content
        .iter()
        .any(|block| matches!(block, AssistantBlock::ToolCall(_)))
    {
        return BranchSummaryResult {
            error: Some("Branch summarization attempted to call a tool".to_owned()),
            ..BranchSummaryResult::default()
        };
    }

    let mut summary = content_text(&response.content.as_slice());

    // Prepend preamble to provide context about the branch summary.
    summary = format!("{BRANCH_SUMMARY_PREAMBLE}{summary}");

    // Compute file lists and append to summary.
    let file_lists = compute_file_lists(&preparation.file_ops);
    summary.push_str(&format_file_operations(
        &file_lists.read_files,
        &file_lists.modified_files,
    ));

    let summary = if summary.is_empty() {
        "No summary generated".to_owned()
    } else {
        summary
    };

    BranchSummaryResult {
        summary: Some(summary),
        usage: Some(response.usage),
        read_files: Some(file_lists.read_files),
        modified_files: Some(file_lists.modified_files),
        aborted: false,
        error: None,
    }
}

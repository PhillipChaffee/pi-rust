//! Branch summarization, ported from upstream
//! `src/harness/compaction/branch-summarization.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The abandoned-branch entry collection, the budgeted entry preparation,
//! and the summary generation through the compaction module's request
//! boundary.

use std::collections::BTreeSet;
use std::sync::Arc;

use pi_ai::auth::resolve::now_ms;
use pi_ai::models::Models;
use pi_ai::types::{
    BoxedFuture, Context as AiContext, Message, Model, SimpleStreamOptions, StopReason,
    TextContent, UserBlock, UserContent, UserMessage,
};
use pi_ai::utils::retry::{RetryCallbacks, RetryPolicy};
use pi_ai::utils::text::content_text;
use serde_json::Value as JsonValue;

use crate::harness::compaction::compaction::{
    SUMMARIZATION_SYSTEM_PROMPT, SummaryRequest, create_summary_request_options,
    custom_message_agent_message, estimate_tokens, models_summary_request,
};
use crate::harness::context::Context;
use crate::harness::messages::{
    convert_to_llm, create_branch_summary_message, create_compaction_summary_message,
};
use crate::harness::session::types::{Branch, BranchScan, Entry, Session, SessionError};
use crate::harness::types::{BranchSummaryError, BranchSummaryErrorCode};
use crate::types::AgentMessage;

use super::types::{BranchPreparation, BranchSummaryResult};
use super::utils::{
    FileOpsAccumulator, compute_file_lists, durable_file_operations, extract_file_ops_from_message,
    format_file_operations, serialize_conversation,
};

/// The file-operation details a generated branch summary entry stores,
/// upstream's `BranchSummaryDetails`.
///
/// The entry's `details` wire object carries `readFiles`/`modifiedFiles`,
/// the pair [`BranchSummaryResult`] returns.
pub type BranchSummaryDetails = super::types::CompactionDetails;

/// The narrow branch surface the collector reads, upstream's
/// `Pick<Branch, "findEntries">`.
pub trait BranchPathReader: Send + Sync {
    /// The branch path, upstream's `Branch.findEntries`.
    fn find_entries(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>>;
}

impl<T: Branch + ?Sized> BranchPathReader for T {
    fn find_entries(
        &self,
        query: Option<&BranchScan>,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        Branch::find_entries(self, query, context)
    }
}

/// The narrow session surface the collector reads, upstream's
/// `Pick<Session, "getEntry">`.
pub trait SessionEntryReader: Send + Sync {
    /// One entry by id, upstream's `Session.getEntry`.
    fn get_entry(
        &self,
        id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>>;
}

impl<T: Session + ?Sized> SessionEntryReader for T {
    fn get_entry(
        &self,
        id: &str,
        context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        Session::get_entry(self, id, context)
    }
}

/// Entries selected for branch summarization, upstream's
/// `CollectEntriesResult`.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectEntriesResult {
    /// Entries to summarize in chronological order.
    pub entries: Vec<Entry>,
    /// Deepest common ancestor between the previous tip and target entry.
    pub common_ancestor_id: Option<String>,
}

/// Options for generating a branch summary, upstream's
/// `GenerateBranchSummaryOptions`.
#[derive(Clone, Debug)]
pub struct GenerateBranchSummaryOptions {
    /// Provider collection the summarization request goes through; owns
    /// auth resolution.
    pub models: Arc<Models>,
    /// Model used for summarization.
    pub model: Model,
    /// Optional instructions appended to or replacing the default prompt.
    pub custom_instructions: Option<String>,
    /// Replace the default prompt with custom instructions instead of
    /// appending them. Default `false`.
    pub replace_instructions: Option<bool>,
    /// Tokens reserved for prompt and model output. Default `16384`.
    pub reserve_tokens: Option<u64>,
    /// Optional retry policy for transient summarization errors.
    pub retry: Option<RetryPolicy>,
    /// Optional callbacks for retry reporting.
    pub callbacks: Option<RetryCallbacks>,
}

/// Collect entries that should be summarized before navigating to a
/// different session tree entry, upstream's `collectEntriesForBranchSummary`.
///
/// The abandoned branch side runs from the old tip down to (excluding) the
/// deepest common ancestor with the target path.
///
/// # Errors
/// Upstream's `Corrupt session: entry ... not found` throw restates as
/// [`SessionError::Message`], and the readers' failures propagate.
pub async fn collect_entries_for_branch_summary(
    branch: &dyn BranchPathReader,
    session: &dyn SessionEntryReader,
    old_tip_id: Option<&str>,
    target_id: &str,
    context: &Context,
) -> Result<CollectEntriesResult, SessionError> {
    let Some(old_tip_id) = old_tip_id else {
        return Ok(CollectEntriesResult {
            entries: Vec::new(),
            common_ancestor_id: None,
        });
    };
    let old_path: BTreeSet<String> = branch
        .find_entries(
            Some(&BranchScan {
                start: Some(old_tip_id.to_owned()),
                ..BranchScan::default()
            }),
            context,
        )
        .await?
        .into_iter()
        .map(|entry| entry.id().to_owned())
        .collect();
    let target_path = branch
        .find_entries(
            Some(&BranchScan {
                start: Some(target_id.to_owned()),
                ..BranchScan::default()
            }),
            context,
        )
        .await?;
    let mut common_ancestor_id: Option<String> = None;
    for entry in &target_path {
        if old_path.contains(entry.id()) {
            common_ancestor_id = Some(entry.id().to_owned());
            break;
        }
    }
    let mut entries: Vec<Entry> = Vec::new();
    let mut current: Option<String> = Some(old_tip_id.to_owned());

    while let Some(id) = current {
        if common_ancestor_id
            .as_ref()
            .is_some_and(|ancestor| ancestor == &id)
        {
            break;
        }
        let entry = session.get_entry(&id, context).await?.ok_or_else(|| {
            SessionError::Message(format!("Corrupt session: entry {id} not found"))
        })?;
        current = entry.parent_id().map(str::to_owned);
        entries.push(entry);
    }
    entries.reverse();

    Ok(CollectEntriesResult {
        entries,
        common_ancestor_id,
    })
}

/// The `AgentMessage` one transcript entry contributes to a branch
/// summary, upstream's `getMessageFromEntry` in `branch-summarization.ts`:
/// tool-result messages and custom entries contribute nothing, the
/// difference from the compaction module's extractor.
fn get_message_from_entry(entry: &Entry) -> Option<AgentMessage> {
    match entry {
        Entry::Message { body, .. } => match &body.message {
            AgentMessage::Standard(Message::ToolResult(_)) => None,
            _ => Some(body.message.clone()),
        },
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

/// Prepare branch entries for summarization within an optional token
/// budget, upstream's `prepareBranchEntries`.
///
/// Earlier branch summaries' file-operation details seed the accumulator,
/// then the branch's visible messages accumulate from the tip down until
/// the budget is crossed — a compaction or branch-summary entry still
/// rides in while the total sits below nine tenths of the budget. A zero
/// budget is unlimited; upstream's negative budget (a context window
/// below the reserve) takes the same unlimited case.
#[must_use]
pub fn prepare_branch_entries(entries: &[Entry], token_budget: u64) -> BranchPreparation {
    let mut file_ops = FileOpsAccumulator::default();
    for entry in entries {
        let Entry::BranchSummary { body, .. } = entry else {
            continue;
        };
        let Some(details) = body.details.as_ref().filter(|details| details.is_object()) else {
            continue;
        };
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
    let mut messages: Vec<AgentMessage> = Vec::new();
    let mut total_tokens: u64 = 0;
    for entry in entries.iter().rev() {
        let Some(message) = get_message_from_entry(entry) else {
            continue;
        };
        extract_file_ops_from_message(&message, &mut file_ops);

        let tokens = estimate_tokens(&message);
        if token_budget > 0 && total_tokens + tokens > token_budget {
            if matches!(entry, Entry::Compaction { .. } | Entry::BranchSummary { .. })
                // The nine-tenths fraction restates exactly over integers:
                // `totalTokens < tokenBudget * 0.9` is `10 * total < 9 *
                // budget`.
                && total_tokens * 10 < token_budget * 9
            {
                messages.push(message);
                total_tokens += tokens;
            }
            break;
        }

        messages.push(message);
        total_tokens += tokens;
    }
    messages.reverse();

    #[expect(
        clippy::cast_possible_wrap,
        reason = "token estimates are far below i64::MAX; the durable preparation's total_tokens is i64"
    )]
    let total_tokens = total_tokens as i64;

    BranchPreparation {
        messages,
        file_ops: durable_file_operations(&file_ops),
        total_tokens,
    }
}

const BRANCH_SUMMARY_PREAMBLE: &str = "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n";

const BRANCH_SUMMARY_PROMPT: &str = "Create a structured summary of this conversation branch for context when returning later.\n\nUse this EXACT format:\n\n## Goal\n[What was the user trying to accomplish in this branch?]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements mentioned]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Work that was started but not finished]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [What should happen next to continue this work]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

/// Options for generating a branch summary, upstream's
/// `PreparedBranchSummaryOptions`.
#[derive(Clone, Debug, Default)]
pub struct PreparedBranchSummaryOptions {
    /// Optional instructions appended to or replacing the default prompt.
    pub custom_instructions: Option<String>,
    /// Replace the default prompt with custom instructions instead of
    /// appending them. Default `false`.
    pub replace_instructions: Option<bool>,
}

/// Generate a summary for abandoned branch entries, upstream's
/// `generateBranchSummary`.
///
/// The budget takes the model's context window minus the reserved tokens,
/// then the prepared generation runs through the models-backed request
/// boundary.
///
/// # Errors
/// The summarization request's aborted or failed settlement.
pub async fn generate_branch_summary(
    entries: &[Entry],
    options: &GenerateBranchSummaryOptions,
    context: &Context,
) -> Result<BranchSummaryResult, BranchSummaryError> {
    let GenerateBranchSummaryOptions {
        models,
        model,
        custom_instructions,
        replace_instructions,
        reserve_tokens,
        retry,
        callbacks,
    } = options;
    let reserve_tokens = reserve_tokens.unwrap_or(16_384);
    let context_window = if model.context_window > 0 {
        model.context_window
    } else {
        128_000
    };
    // Upstream's `contextWindow - reserveTokens` can go negative, the
    // unlimited budget; the saturating restatement clamps to zero, the
    // same unlimited case.
    let token_budget = context_window.saturating_sub(reserve_tokens);
    let preparation = prepare_branch_entries(entries, token_budget);
    let request = models_summary_request(models, model, retry.as_ref(), callbacks.as_ref());
    generate_branch_summary_with_request(
        &preparation,
        &PreparedBranchSummaryOptions {
            custom_instructions: custom_instructions.clone(),
            replace_instructions: *replace_instructions,
        },
        &request,
        context,
    )
    .await
}

/// Generate a prepared branch summary through a caller-owned one-request
/// boundary, upstream's `generateBranchSummaryWithRequest`.
///
/// The preamble prefixes the generated text and the file-operation tags
/// append.
///
/// # Errors
/// [`BranchSummaryErrorCode::Aborted`] when the request aborted (upstream's
/// `"Branch summary aborted"` default),
/// [`BranchSummaryErrorCode::SummarizationFailed`] when it failed
/// (`"Branch summary failed: ..."`).
pub async fn generate_branch_summary_with_request(
    preparation: &BranchPreparation,
    options: &PreparedBranchSummaryOptions,
    request: &SummaryRequest,
    context: &Context,
) -> Result<BranchSummaryResult, BranchSummaryError> {
    let PreparedBranchSummaryOptions {
        custom_instructions,
        replace_instructions,
    } = options;
    let BranchPreparation {
        messages, file_ops, ..
    } = preparation;
    if messages.is_empty() {
        return Ok(BranchSummaryResult {
            summary: "No content to summarize".to_owned(),
            usage: None,
            read_files: Vec::new(),
            modified_files: Vec::new(),
        });
    }
    let llm_messages = convert_to_llm(messages);
    let conversation_text = serialize_conversation(&llm_messages);
    let instructions = match (custom_instructions, replace_instructions.unwrap_or(false)) {
        (Some(custom), true) => custom.clone(),
        (Some(custom), false) => format!("{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom}"),
        (None, _) => BRANCH_SUMMARY_PROMPT.to_owned(),
    };
    let prompt_text =
        format!("<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}");

    let completion_options = SimpleStreamOptions {
        max_tokens: Some(2048),
        ..SimpleStreamOptions::default()
    };

    let response = request(
        &AiContext {
            system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_owned()),
            messages: vec![Message::User(UserMessage {
                content: UserContent::Blocks(vec![UserBlock::Text(TextContent {
                    text: prompt_text,
                    text_signature: None,
                })]),
                timestamp: now_ms(),
            })],
            tools: None,
        },
        &create_summary_request_options(completion_options, context),
        context,
    )
    .await;
    if response.stop_reason == StopReason::Aborted {
        return Err(BranchSummaryError::new(
            BranchSummaryErrorCode::Aborted,
            response
                .error_message
                .unwrap_or_else(|| "Branch summary aborted".to_owned()),
            None,
        ));
    }
    if response.stop_reason == StopReason::Error {
        return Err(BranchSummaryError::new(
            BranchSummaryErrorCode::SummarizationFailed,
            format!(
                "Branch summary failed: {}",
                response
                    .error_message
                    .unwrap_or_else(|| "Unknown error".to_owned())
            ),
            None,
        ));
    }

    let mut summary = content_text(&response.content.as_slice());
    summary = format!("{BRANCH_SUMMARY_PREAMBLE}{summary}");
    let file_lists = compute_file_lists(file_ops);
    summary += &format_file_operations(&file_lists.read_files, &file_lists.modified_files);

    Ok(BranchSummaryResult {
        summary: if summary.is_empty() {
            "No summary generated".to_owned()
        } else {
            summary
        },
        usage: Some(response.usage),
        read_files: file_lists.read_files,
        modified_files: file_lists.modified_files,
    })
}

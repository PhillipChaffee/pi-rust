//! The branch-summarization suite, ported 1:1 from upstream
//! `test/harness/branch-summarization.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, with boundary tests on top
//! binding the untested paths (the corrupt-session error, the budget
//! selection, and the generated-summary shapes).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod compaction_common;
use compaction_common::*;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;

use pi_agent_core::harness::compaction::branch_summarization::PreparedBranchSummaryOptions;
use pi_agent_core::harness::compaction::branch_summarization::{
    BranchPathReader, CollectEntriesResult, GenerateBranchSummaryOptions, SessionEntryReader,
    collect_entries_for_branch_summary, generate_branch_summary,
    generate_branch_summary_with_request, prepare_branch_entries,
};
use pi_agent_core::harness::compaction::compaction::SummaryRequest;
use pi_agent_core::harness::compaction::types::BranchPreparation;
use pi_agent_core::harness::context::{Context, background_context};
use pi_agent_core::harness::session::types::{BranchScan, Entry, MessageEntry, SessionError};
use pi_agent_core::harness::types::BranchSummaryErrorCode;
use pi_agent_core::types::AgentMessage;
use pi_ai::providers::faux::{
    FauxAssistantMessageOptions, FauxProviderState, FauxResponseStep, faux_assistant_message,
};
use pi_ai::types::{
    AssistantBlock, BoxedFuture, Context as AiContext, Message, Model, SimpleStreamOptions,
    StopReason, ToolCall,
};
use serde_json::json;

/// The user message fixture, upstream's `message`.
fn message(text: &str) -> AgentMessage {
    create_user_message(text)
}

/// The message entry fixture, upstream's `messageEntry`; the timestamp is
/// the sequence like upstream's `timestamp: seq`.
fn message_entry(id: &str, parent_id: Option<&str>, text: &str, seq: u64) -> Entry {
    Entry::Message {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        seq,
        #[expect(
            clippy::cast_possible_wrap,
            reason = "the sequence doubles as the timestamp like upstream's `timestamp: seq`; seq stays far below i64::MAX"
        )]
        timestamp: seq as i64,
        body: Box::new(MessageEntry {
            message: message(text),
            terminate: None,
        }),
    }
}

/// The in-memory branch/session reader, upstream's `branchReader`: a map
/// of entries; `findEntries` walks the parent chain and `getEntry` looks
/// one up.
struct BranchReader {
    by_id: BTreeMap<String, Entry>,
    /// The ids `getEntry` reports missing, the corrupt-session fixture.
    missing: BTreeSet<String>,
}

impl BranchReader {
    fn new(entries: &[Entry]) -> Self {
        Self {
            by_id: entries
                .iter()
                .map(|entry| (entry.id().to_owned(), entry.clone()))
                .collect(),
            missing: BTreeSet::new(),
        }
    }
}

impl BranchPathReader for BranchReader {
    fn find_entries(
        &self,
        query: Option<&BranchScan>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        let start = query.and_then(|query| query.start.clone());
        Box::pin(async move {
            let mut path = Vec::new();
            let mut current = start;
            while let Some(id) = current {
                let entry = self
                    .by_id
                    .get(&id)
                    .ok_or_else(|| SessionError::Message(format!("Unknown entry {id}")))?;
                current = entry.parent_id().map(str::to_owned);
                path.push(entry.clone());
            }
            Ok(path)
        })
    }
}

impl SessionEntryReader for BranchReader {
    fn get_entry(
        &self,
        id: &str,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        let found = self
            .by_id
            .get(id)
            .cloned()
            .filter(|_| !self.missing.contains(id));
        Box::pin(async move { Ok(found) })
    }
}

#[tokio::test]
async fn collects_the_abandoned_side_of_a_branch_in_chronological_order() {
    let root = message_entry("root", None, "root", 1);
    let common = message_entry("common", Some("root"), "common", 2);
    let abandoned1 = message_entry("abandoned-1", Some("common"), "abandoned 1", 3);
    let abandoned2 = message_entry("abandoned-2", Some("abandoned-1"), "abandoned 2", 4);
    let target = message_entry("target", Some("common"), "target", 5);
    let reader = BranchReader::new(&[root, common, abandoned1.clone(), abandoned2.clone(), target]);

    let result = collect_entries_for_branch_summary(
        &reader,
        &reader,
        Some("abandoned-2"),
        "target",
        &background_context(),
    )
    .await
    .expect("collection succeeds");
    assert_eq!(result.common_ancestor_id.as_deref(), Some("common"));
    let ids: Vec<&str> = result.entries.iter().map(Entry::id).collect();
    assert_eq!(ids, ["abandoned-1", "abandoned-2"]);
    assert!(!result.entries.iter().any(|entry| entry.id() == "root"));
}

#[tokio::test]
async fn returns_no_entries_when_there_was_no_previous_leaf() {
    let target = message_entry("target", None, "target", 1);
    let reader = BranchReader::new(&[target]);
    assert_eq!(
        collect_entries_for_branch_summary(&reader, &reader, None, "target", &background_context())
            .await
            .expect("collection succeeds"),
        CollectEntriesResult {
            entries: Vec::new(),
            common_ancestor_id: None,
        }
    );
}

// ---------------------------------------------------------------------------
// Boundary tests: the paths upstream's two-case suite does not reach.
// ---------------------------------------------------------------------------

/// The options the factories record, the file-local read.
fn seen(
    snapshot: &Arc<Mutex<Vec<SimpleStreamOptions>>>,
) -> MutexGuard<'_, Vec<SimpleStreamOptions>> {
    snapshot.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The factory step that records the request options each call saw.
fn options_recorder(seen_options: &Arc<Mutex<Vec<SimpleStreamOptions>>>) -> FauxResponseStep {
    let seen_options = Arc::clone(seen_options);
    FauxResponseStep::Factory(Arc::new(
        move |_context: &AiContext,
              options: Option<&SimpleStreamOptions>,
              _state: &FauxProviderState,
              _model: &Model| {
            if let Some(options) = options {
                seen(&seen_options).push(options.clone());
            }
            let response = faux_assistant_message(
                "## Goal\nTest summary",
                FauxAssistantMessageOptions::default(),
            );
            Box::pin(async move { Ok(response) })
        },
    ))
}

/// The request boundary that records the prompt text, upstream's
/// `(context) => { promptText = context.messages[0]... }` factory restated
/// over the request seam.
fn prompt_request(seen_prompt: &Arc<Mutex<String>>) -> SummaryRequest {
    let seen_prompt = Arc::clone(seen_prompt);
    Arc::new(
        move |ai_context: &AiContext, _options: &SimpleStreamOptions, _context: &Context| {
            let prompt = ai_context
                .messages
                .first()
                .map_or_else(String::new, |message| match message {
                    Message::User(user) => match &user.content {
                        pi_ai::types::UserContent::Blocks(blocks) => blocks
                            .iter()
                            .find_map(|block| match block {
                                pi_ai::types::UserBlock::Text(text) => Some(text.text.clone()),
                                pi_ai::types::UserBlock::Image(_) => None,
                            })
                            .unwrap_or_default(),
                        pi_ai::types::UserContent::Text(text) => text.clone(),
                    },
                    _ => String::new(),
                });
            *seen_prompt.lock().unwrap_or_else(PoisonError::into_inner) = prompt;
            let response = faux_assistant_message(
                "## Goal\nBranch body",
                FauxAssistantMessageOptions::default(),
            );
            Box::pin(async move { response })
        },
    )
}

/// The queued-response request boundary, the stub seam the `WithRequest`
/// tests drive.
fn queued_request(response: pi_ai::types::AssistantMessage) -> SummaryRequest {
    Arc::new(
        move |_ai_context: &AiContext, _options: &SimpleStreamOptions, _context: &Context| {
            let response = response.clone();
            Box::pin(async move { response })
        },
    )
}

/// The failed/aborted request boundary, upstream's
/// `fauxAssistantMessage("", { stopReason })` stub.
fn failing_request(stop_reason: StopReason, error_message: Option<&str>) -> SummaryRequest {
    let response = faux_assistant_message(
        "",
        FauxAssistantMessageOptions {
            stop_reason: Some(stop_reason),
            error_message: error_message.map(str::to_owned),
            ..FauxAssistantMessageOptions::default()
        },
    );
    queued_request(response)
}

/// The read-a-file transcript the preparation scenarios use: a user, a
/// read tool call, and a trailing assistant message.
fn read_a_file_transcript() -> Vec<Entry> {
    let user = create_message_entry(create_user_message("read a file"), None);
    let assistant_message = {
        let mut message =
            create_assistant_message("calling tool", create_mock_usage(100, 50, 0, 0));
        message.content = vec![AssistantBlock::ToolCall(ToolCall {
            id: "tool-1".to_owned(),
            name: "read".to_owned(),
            arguments: json!({ "path": "src/index.ts" })
                .as_object()
                .expect("object")
                .clone(),
            thought_signature: None,
            namespace: None,
        })];
        message
    };
    let assistant = create_message_entry(
        AgentMessage::Standard(Message::Assistant(assistant_message)),
        Some(entry_id(&user)),
    );
    vec![user, assistant]
}

#[tokio::test]
async fn reports_a_corrupt_session_when_the_walk_hits_a_missing_entry() {
    let root = message_entry("root", None, "root", 1);
    let target = message_entry("target", Some("root"), "target", 2);
    // The abandoned side's parent chain leaves the session map: the walk
    // finds the chain through the branch but cannot fetch one entry.
    let abandoned = message_entry("abandoned", Some("target"), "abandoned", 3);
    let branch_reader = BranchReader::new(&[root.clone(), target.clone(), abandoned]);
    let mut corrupt = BranchReader::new(&[target]);
    corrupt.missing.insert("abandoned".to_owned());

    let error = collect_entries_for_branch_summary(
        &branch_reader,
        &corrupt,
        Some("abandoned"),
        "target",
        &background_context(),
    )
    .await
    .expect_err("the walk reports the corrupt session");
    assert!(
        matches!(error, SessionError::Message(message) if message == "Corrupt session: entry abandoned not found")
    );
}

#[tokio::test]
async fn the_entry_budget_selects_messages_from_the_tip_down() {
    // The branch-summary entry carries the file-operation details the
    // preparation seeds from; upstream's seeding loop reads only
    // `branch_summary` entries.
    let branch_summary = create_branch_summary_entry_with_details(
        None,
        "from",
        "branch summary",
        json!({ "readFiles": ["old-read.ts"], "modifiedFiles": ["old-edit.ts"] }),
    );
    let user = create_message_entry(create_user_message("user"), Some(entry_id(&branch_summary)));
    let assistant = create_message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "assistant with a fairly long reply",
            create_mock_usage(100, 50, 0, 0),
        ))),
        Some(entry_id(&user)),
    );
    let tool_result = create_message_entry(
        AgentMessage::Standard(Message::ToolResult(pi_ai::types::ToolResultMessage {
            tool_call_id: "call-1".to_owned(),
            tool_name: "read".to_owned(),
            content: vec![pi_ai::types::ToolResultBlock::Text(
                pi_ai::types::TextContent {
                    text: "tool output".to_owned(),
                    text_signature: None,
                },
            )],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: NOW,
        })),
        Some(entry_id(&assistant)),
    );
    let entries = [branch_summary, user, assistant, tool_result];

    // A zero budget is unlimited: every visible message rides in, in
    // chronological order, the tool result contributes nothing, and the
    // branch summary's details seed the file operations.
    let unlimited = prepare_branch_entries(&entries, 0);
    let roles: Vec<&str> = unlimited.messages.iter().map(message_role).collect();
    assert_eq!(roles, ["branchSummary", "user", "assistant"]);
    assert!(unlimited.file_ops.read.contains(&"old-read.ts".to_owned()));
    assert!(
        unlimited
            .file_ops
            .edited
            .contains(&"old-edit.ts".to_owned())
    );

    // A budget the compaction entry crosses with the total already at or
    // past nine tenths excludes it: ten tokens against a budget of eleven
    // is past the 0.9 mark.
    let tight_budget = prepare_branch_entries(&entries, 11);
    let roles: Vec<&str> = tight_budget.messages.iter().map(message_role).collect();
    assert_eq!(roles, ["user", "assistant"]);

    // One token of headroom below the mark lets the boundary compaction
    // ride in.
    let roomy_budget = prepare_branch_entries(&entries, 12);
    let roles: Vec<&str> = roomy_budget.messages.iter().map(message_role).collect();
    assert_eq!(roles, ["branchSummary", "user", "assistant"]);
}

#[tokio::test]
async fn the_generated_branch_summary_carries_the_preamble_and_file_tags() {
    let preparation = prepare_branch_entries(&read_a_file_transcript(), 0);
    assert!(
        preparation
            .file_ops
            .read
            .contains(&"src/index.ts".to_owned())
    );

    let result = generate_branch_summary_with_request(
        &preparation,
        &PreparedBranchSummaryOptions::default(),
        &queued_request(faux_assistant_message(
            "## Goal\nBranch body",
            FauxAssistantMessageOptions::default(),
        )),
        &background_context(),
    )
    .await
    .expect("summary succeeds");
    assert!(result
        .summary
        .starts_with("The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n## Goal\nBranch body"));
    assert!(
        result
            .summary
            .contains("<read-files>\nsrc/index.ts\n</read-files>")
    );
    assert_eq!(result.read_files, ["src/index.ts".to_owned()]);
    assert!(result.modified_files.is_empty());
    assert!(result.usage.is_some());
}

#[tokio::test]
async fn returns_no_content_to_summarize_without_a_request() {
    let preparation = BranchPreparation {
        messages: Vec::new(),
        file_ops: pi_agent_core::harness::compaction::types::FileOperations::default(),
        total_tokens: 0,
    };
    let requested = Arc::new(Mutex::new(false));
    let requested_probe = Arc::clone(&requested);
    let request: SummaryRequest = Arc::new(
        move |_ai_context: &AiContext, _options: &SimpleStreamOptions, _context: &Context| {
            *requested_probe
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = true;
            let response =
                faux_assistant_message("unreachable", FauxAssistantMessageOptions::default());
            Box::pin(async move { response })
        },
    );

    let result = generate_branch_summary_with_request(
        &preparation,
        &PreparedBranchSummaryOptions::default(),
        &request,
        &background_context(),
    )
    .await
    .expect("summary succeeds");
    assert_eq!(result.summary, "No content to summarize");
    assert_eq!(result.usage, None);
    assert!(result.read_files.is_empty());
    assert!(result.modified_files.is_empty());
    assert!(!*requested.lock().unwrap_or_else(PoisonError::into_inner));
}

#[tokio::test]
async fn instructions_replace_or_append_the_default_prompt() {
    let preparation = prepare_branch_entries(&read_a_file_transcript(), 0);
    let seen_prompt = Arc::new(Mutex::new(String::new()));

    generate_branch_summary_with_request(
        &preparation,
        &PreparedBranchSummaryOptions {
            custom_instructions: Some("custom text".to_owned()),
            replace_instructions: Some(true),
        },
        &prompt_request(&seen_prompt),
        &background_context(),
    )
    .await
    .expect("summary succeeds");
    let prompt = seen_prompt
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    assert!(prompt.ends_with("\n\ncustom text"));
    assert!(!prompt.contains("Additional focus:"));

    generate_branch_summary_with_request(
        &preparation,
        &PreparedBranchSummaryOptions {
            custom_instructions: Some("focus".to_owned()),
            replace_instructions: None,
        },
        &prompt_request(&seen_prompt),
        &background_context(),
    )
    .await
    .expect("summary succeeds");
    let prompt = seen_prompt
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    assert!(prompt.contains("Additional focus: focus"));
    assert!(prompt.contains("## Goal"));
}

#[tokio::test]
async fn returns_error_results_for_failed_or_aborted_branch_summaries() {
    let preparation = prepare_branch_entries(&read_a_file_transcript(), 0);

    let error = generate_branch_summary_with_request(
        &preparation,
        &PreparedBranchSummaryOptions::default(),
        &failing_request(StopReason::Error, Some("boom")),
        &background_context(),
    )
    .await
    .expect_err("the failed request errors");
    assert!(matches!(
        error.code,
        BranchSummaryErrorCode::SummarizationFailed
    ));
    assert_eq!(error.message, "Branch summary failed: boom");

    // An aborted response with no error message carries the default verb.
    let aborted = generate_branch_summary_with_request(
        &preparation,
        &PreparedBranchSummaryOptions::default(),
        &failing_request(StopReason::Aborted, None),
        &background_context(),
    )
    .await
    .expect_err("the aborted request errors");
    assert!(matches!(aborted.code, BranchSummaryErrorCode::Aborted));
    assert_eq!(aborted.message, "Branch summary aborted");
}

#[tokio::test]
async fn generate_branch_summary_runs_through_the_models_collection() {
    let user = create_message_entry(create_user_message("hello"), None);
    let models = test_models();
    let faux = create_faux_model(&models, false, 8192);
    let seen_options = Arc::new(Mutex::new(Vec::new()));
    faux.set_responses([options_recorder(&seen_options)]);

    let result = generate_branch_summary(
        &[user],
        &GenerateBranchSummaryOptions {
            models: Arc::new(models),
            model: faux.first_model(),
            custom_instructions: None,
            replace_instructions: None,
            reserve_tokens: None,
            retry: None,
            callbacks: None,
        },
        &background_context(),
    )
    .await
    .expect("summary succeeds");
    assert!(result.summary.contains("## Goal\nTest summary"));
    assert!(
        result
            .summary
            .contains("The user explored a different conversation branch")
    );
    // The one-message budget rides the model's 200000-token window minus
    // the default 16384 reserve.
    assert_eq!(seen(&seen_options).len(), 1);
}

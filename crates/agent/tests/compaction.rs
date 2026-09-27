//! The compaction suite, ported 1:1 from upstream
//! `test/harness/compaction.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream's `createModelsWithSimpleResponses` stub (a `Models` whose
//! `completeSimple` dequeues scripted responses) restates through the
//! `WithRequest` boundary — the port's seam for a caller-owned request;
//! every other test drives the real faux provider through a `Models`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod compaction_common;
use compaction_common::*;

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;

use pi_agent_core::harness::compaction::compaction::{
    CompactGenerationOptions, CompactionDetails, SummaryRequest, calculate_context_tokens, compact,
    compact_with_request, estimate_context_tokens, estimate_tokens, find_cut_point,
    find_turn_start_index, generate_summary, generate_summary_with_usage, get_last_assistant_usage,
    prepare_compaction, serialize_conversation, should_compact,
};
use pi_agent_core::harness::compaction::types::{
    CompactionPreparation, CompactionSettings, DEFAULT_COMPACTION_SETTINGS, FileOperations,
};
use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::context::build_session_context;
use pi_agent_core::harness::session::types::Entry;
use pi_agent_core::harness::types::CompactionErrorCode;
use pi_agent_core::types::{AgentMessage, ThinkingLevel};
use pi_ai::providers::faux::{
    FauxAssistantMessageOptions, FauxProviderState, FauxResponseStep, faux_assistant_message,
};
use pi_ai::types::{
    AssistantBlock, CacheRetention, Context as AiContext, Message, SimpleStreamOptions, StopReason,
    ToolCall, UserBlock, UserContent,
};
use serde_json::json;

/// The options one faux factory saw, the test-local read over
/// [`SeenOptions`].
fn seen(snapshot: &SeenOptions) -> MutexGuard<'_, Vec<SimpleStreamOptions>> {
    snapshot.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The scripted single-text response step, upstream's
/// `fauxAssistantMessage(...)` queue entry.
fn text_response(text: &str) -> FauxResponseStep {
    faux_assistant_message(text, FauxAssistantMessageOptions::default()).into()
}

/// The scripted single-text response message, the recorder call sites'
/// fixture value.
fn text_message(text: &str) -> pi_ai::types::AssistantMessage {
    faux_assistant_message(text, FauxAssistantMessageOptions::default())
}

/// The scripted response step with overrides, upstream's
/// `fauxAssistantMessage("", { stopReason, errorMessage })`.
fn status_response(
    text: &str,
    stop_reason: StopReason,
    error_message: Option<&str>,
) -> FauxResponseStep {
    faux_assistant_message(
        text,
        FauxAssistantMessageOptions {
            stop_reason: Some(stop_reason),
            error_message: error_message.map(str::to_owned),
            ..FauxAssistantMessageOptions::default()
        },
    )
    .into()
}

/// The factory step that records the request options each call saw,
/// upstream's `(_context, options) => { seenOptions.push(options); ... }`.
fn options_recorder(
    seen_options: &SeenOptions,
    response: pi_ai::types::AssistantMessage,
) -> FauxResponseStep {
    let seen_options = Arc::clone(seen_options);
    FauxResponseStep::Factory(Arc::new(
        move |_context: &AiContext,
              options: Option<&SimpleStreamOptions>,
              _state: &FauxProviderState,
              _model: &pi_ai::types::Model| {
            if let Some(options) = options {
                seen(&seen_options).push(options.clone());
            }
            let response = response.clone();
            Box::pin(async move { Ok(response) })
        },
    ))
}

/// The factory step that records the prompt text, upstream's
/// `(context) => { promptText = context.messages[0]... }`.
fn prompt_recorder(seen_prompt: &Arc<Mutex<String>>) -> FauxResponseStep {
    let seen_prompt = Arc::clone(seen_prompt);
    FauxResponseStep::Factory(Arc::new(
        move |context: &AiContext,
              _options: Option<&SimpleStreamOptions>,
              _state: &FauxProviderState,
              _model: &pi_ai::types::Model| {
            let prompt =
                context
                    .messages
                    .first()
                    .map_or_else(String::new, |message| match message {
                        Message::User(user) => match &user.content {
                            UserContent::Blocks(blocks) => blocks
                                .iter()
                                .find_map(|block| match block {
                                    UserBlock::Text(text) => Some(text.text.clone()),
                                    UserBlock::Image(_) => None,
                                })
                                .unwrap_or_default(),
                            UserContent::Text(text) => text.clone(),
                        },
                        _ => String::new(),
                    });
            *seen_prompt.lock().unwrap_or_else(PoisonError::into_inner) = prompt;
            let message = faux_assistant_message(
                "## Goal\nTest summary",
                FauxAssistantMessageOptions::default(),
            );
            Box::pin(async move { Ok(message) })
        },
    ))
}

/// The compaction preparation fixture upstream's inline literals build.
fn preparation(
    messages: &[AgentMessage],
    is_split_turn: bool,
    tokens_before: i64,
    reserve_tokens: u64,
) -> CompactionPreparation {
    CompactionPreparation {
        messages_to_summarize: messages.to_vec(),
        turn_prefix_messages: if is_split_turn {
            messages.to_vec()
        } else {
            Vec::new()
        },
        retained_tail: messages.to_vec(),
        is_split_turn,
        tokens_before,
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: CompactionSettings {
            enabled: true,
            reserve_tokens,
            keep_recent_tokens: 20,
        },
    }
}

#[tokio::test]
async fn calculates_total_context_tokens_from_usage() {
    assert_eq!(
        calculate_context_tokens(&create_mock_usage(1000, 500, 200, 100)),
        1800
    );
    assert_eq!(calculate_context_tokens(&create_mock_usage(0, 0, 0, 0)), 0);
}

#[tokio::test]
async fn checks_compaction_threshold() {
    let settings = CompactionSettings {
        enabled: true,
        reserve_tokens: 10_000,
        keep_recent_tokens: 20_000,
    };
    assert!(should_compact(95_000, 100_000, &settings));
    assert!(!should_compact(89_000, 100_000, &settings));
    assert!(!should_compact(
        95_000,
        100_000,
        &CompactionSettings {
            enabled: false,
            ..settings
        }
    ));
}

#[tokio::test]
async fn finds_a_cut_point_based_on_token_differences() {
    let mut entries: Vec<Entry> = Vec::new();
    let mut parent_id: Option<String> = None;
    for i in 0..10 {
        let user =
            create_message_entry(create_user_message(&format!("User {i}")), parent_id.clone());
        entries.push(user.clone());
        let assistant = create_message_entry(
            AgentMessage::Standard(Message::Assistant(create_assistant_message(
                &format!("Assistant {i}"),
                create_mock_usage(0, 100, (i + 1) * 1000, 0),
            ))),
            Some(entry_id(&user)),
        );
        entries.push(assistant.clone());
        parent_id = Some(entry_id(&assistant));
    }

    let result = find_cut_point(&entries, 0, entries.len(), 2500);
    assert!(matches!(
        entries[result.first_kept_entry_index],
        Entry::Message { .. }
    ));
}

#[tokio::test]
async fn covers_cut_point_and_turn_start_edge_cases() {
    let first_custom = create_custom_entry("first", None);
    let second_custom = create_custom_entry("second", Some(entry_id(&first_custom)));
    let result = find_cut_point(&[first_custom.clone(), second_custom.clone()], 0, 2, 1);
    assert_eq!(result.first_kept_entry_index, 0);
    assert_eq!(result.turn_start_index, None);
    assert!(!result.is_split_turn());

    let branch_summary =
        create_branch_summary_entry(Some(entry_id(&second_custom)), "branch", "branch summary");
    assert_eq!(
        find_turn_start_index(&[first_custom.clone(), branch_summary.clone()], 1, 0),
        Some(1)
    );
    assert_eq!(
        find_turn_start_index(&[first_custom.clone(), second_custom], 1, 0),
        None
    );

    let result = find_cut_point(&[first_custom, branch_summary], 0, 2, 1);
    assert_eq!(result.first_kept_entry_index, 0);

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
        None,
    );
    let result = find_cut_point(std::slice::from_ref(&tool_result), 0, 1, 1);
    assert_eq!(result.first_kept_entry_index, 0);
    assert_eq!(result.turn_start_index, None);
    assert!(!result.is_split_turn());

    let user = create_message_entry(create_user_message("user"), None);
    let compaction = create_compaction_entry("summary", Some(entry_id(&user)));
    let assistant = create_message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "assistant",
            create_mock_usage(100, 50, 0, 0),
        ))),
        Some(entry_id(&compaction)),
    );
    let result = find_cut_point(&[user, compaction, assistant], 0, 3, 1);
    assert_eq!(result.first_kept_entry_index, 2);
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the body is upstream's single `it` transcribed: eight message fixtures and their assertions in one scenario"
)]
async fn estimates_tokens_and_context_usage_across_supported_message_roles() {
    let usage = create_mock_usage(10, 5, 3, 2);
    let assistant = create_assistant_message("assistant", usage);
    let assistant_with_thinking_and_tool = {
        let mut message = assistant.clone();
        message.content = vec![
            AssistantBlock::Thinking(pi_ai::types::ThinkingContent {
                thinking: "thinking".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantBlock::ToolCall(ToolCall {
                id: "call-1".to_owned(),
                name: "read".to_owned(),
                arguments: json!({ "path": "file.ts" })
                    .as_object()
                    .expect("object")
                    .clone(),
                thought_signature: None,
                namespace: None,
            }),
        ];
        message
    };
    let custom_string = AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage {
        role: "custom".to_owned(),
        timestamp: NOW,
        data: json!({
            "customType": "note",
            "content": "custom text",
            "display": true,
        })
        .as_object()
        .expect("object")
        .clone(),
    });
    let tool_result_with_image =
        AgentMessage::Standard(Message::ToolResult(pi_ai::types::ToolResultMessage {
            tool_call_id: "call-1".to_owned(),
            tool_name: "read".to_owned(),
            content: vec![
                pi_ai::types::ToolResultBlock::Text(pi_ai::types::TextContent {
                    text: "tool text".to_owned(),
                    text_signature: None,
                }),
                pi_ai::types::ToolResultBlock::Image(pi_ai::types::ImageContent {
                    data: "abc".to_owned(),
                    mime_type: "image/png".to_owned(),
                }),
            ],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: NOW,
        }));
    let bash_execution = AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage {
        role: "bashExecution".to_owned(),
        timestamp: NOW,
        data: json!({
            "command": "npm run check",
            "output": "ok",
            "exitCode": 0,
            "cancelled": false,
            "truncated": false,
        })
        .as_object()
        .expect("object")
        .clone(),
    });
    let branch_summary_message = AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage {
        role: "branchSummary".to_owned(),
        timestamp: NOW,
        data: json!({ "summary": "branch", "fromId": "x" })
            .as_object()
            .expect("object")
            .clone(),
    });
    let compaction_summary_message =
        AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage {
            role: "compactionSummary".to_owned(),
            timestamp: NOW,
            data: json!({ "summary": "compact", "tokensBefore": 123 })
                .as_object()
                .expect("object")
                .clone(),
        });

    assert!(
        estimate_tokens(&AgentMessage::Standard(Message::User(
            pi_ai::types::UserMessage {
                content: UserContent::Text("plain user".to_owned()),
                timestamp: NOW,
            }
        ))) > 0
    );
    assert!(
        estimate_tokens(&AgentMessage::Standard(Message::Assistant(
            assistant_with_thinking_and_tool
        ))) > 0
    );
    assert!(estimate_tokens(&custom_string) > 0);
    assert!(estimate_tokens(&tool_result_with_image) > 1000);
    assert!(estimate_tokens(&bash_execution) > 0);
    assert!(estimate_tokens(&branch_summary_message) > 0);
    assert!(estimate_tokens(&compaction_summary_message) > 0);
    // Upstream feeds a `{ role: "unknown" }` message the switch does not
    // know; the port's custom variant carries the same unknown role.
    assert_eq!(
        estimate_tokens(&AgentMessage::Custom(
            pi_agent_core::types::CustomAgentMessage {
                role: "unknown".to_owned(),
                timestamp: NOW,
                data: serde_json::Map::default(),
            }
        )),
        0
    );
    assert_eq!(
        get_last_assistant_usage(&[
            create_message_entry(create_user_message("user"), None),
            create_message_entry(
                AgentMessage::Standard(Message::Assistant(assistant.clone())),
                None
            ),
        ]),
        Some(usage)
    );
    let aborted = {
        let mut message = assistant.clone();
        message.stop_reason = StopReason::Aborted;
        create_message_entry(AgentMessage::Standard(Message::Assistant(message)), None)
    };
    let failed = {
        let mut message = assistant.clone();
        message.stop_reason = StopReason::Error;
        create_message_entry(AgentMessage::Standard(Message::Assistant(message)), None)
    };
    assert_eq!(get_last_assistant_usage(&[aborted, failed]), None);
    let partial = create_assistant_message("partial", create_mock_usage(0, 0, 0, 0));
    assert_eq!(
        get_last_assistant_usage(&[
            create_message_entry(create_user_message("user"), None),
            create_message_entry(
                AgentMessage::Standard(Message::Assistant(assistant.clone())),
                None
            ),
            create_message_entry(AgentMessage::Standard(Message::Assistant(partial)), None),
        ]),
        Some(usage)
    );
    assert_eq!(
        estimate_context_tokens(&[create_user_message("no usage")]).last_usage_index,
        None
    );
    let estimate = estimate_context_tokens(&[
        AgentMessage::Standard(Message::Assistant(assistant.clone())),
        create_user_message("tail"),
    ]);
    assert_eq!(estimate.usage_tokens, 20);
    assert_eq!(estimate.last_usage_index, Some(0));

    let estimate = estimate_context_tokens(&[
        create_user_message("Hello"),
        AgentMessage::Standard(Message::Assistant(assistant)),
        create_user_message("continue"),
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "Partial thinking",
            create_mock_usage(0, 0, 0, 0),
        ))),
    ]);
    assert_eq!(estimate.usage_tokens, 20);
    assert_eq!(estimate.last_usage_index, Some(1));
    assert!(estimate.trailing_tokens > 0);
    assert_eq!(estimate.tokens, 20 + estimate.trailing_tokens);
}

#[tokio::test]
async fn builds_session_context_with_a_compaction_entry() {
    let u1 = create_message_entry(create_user_message("1"), None);
    let a1 = create_message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "a",
            create_mock_usage(100, 50, 0, 0),
        ))),
        Some(entry_id(&u1)),
    );
    let u2 = create_message_entry(create_user_message("2"), Some(entry_id(&a1)));
    let a2 = create_message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "b",
            create_mock_usage(100, 50, 0, 0),
        ))),
        Some(entry_id(&u2)),
    );
    let compaction = create_compaction_entry_with(
        "Summary of 1,a,2,b",
        Some(entry_id(&a2)),
        vec![
            create_user_message("2"),
            AgentMessage::Standard(Message::Assistant(create_assistant_message(
                "b",
                create_mock_usage(100, 50, 0, 0),
            ))),
        ],
        None,
    );
    let u3 = create_message_entry(create_user_message("3"), Some(entry_id(&compaction)));
    let a3 = create_message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "c",
            create_mock_usage(100, 50, 0, 0),
        ))),
        Some(entry_id(&u3)),
    );
    let loaded = build_session_context(
        &[u1, a1, u2, a2, compaction, u3, a3],
        None,
        &background_context(),
    )
    .await
    .expect("context builds");
    assert_eq!(loaded.len(), 5);
    assert_eq!(message_role(&loaded[0]), "compactionSummary");
    let roles: Vec<&str> = loaded.iter().map(message_role).collect();
    assert_eq!(
        roles,
        [
            "compactionSummary",
            "user",
            "assistant",
            "user",
            "assistant"
        ]
    );
}

#[tokio::test]
async fn prepares_compaction_using_the_latest_compaction_summary_as_previous_summary() {
    let u1 = create_message_entry(create_user_message("user msg 1"), None);
    let a1 = create_message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "assistant msg 1",
            create_mock_usage(100, 50, 0, 0),
        ))),
        Some(entry_id(&u1)),
    );
    let u2 = create_message_entry(create_user_message("user msg 2"), Some(entry_id(&a1)));
    let a2 = create_message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "assistant msg 2",
            create_mock_usage(5000, 1000, 0, 0),
        ))),
        Some(entry_id(&u2)),
    );
    let compaction1 = create_compaction_entry("First summary", Some(entry_id(&a2)));
    let u3 = create_message_entry(
        create_user_message("user msg 3"),
        Some(entry_id(&compaction1)),
    );
    let a3 = create_message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "assistant msg 3",
            create_mock_usage(8000, 2000, 0, 0),
        ))),
        Some(entry_id(&u3)),
    );
    let path_entries = [u1, a1, u2, a2, compaction1, u3, a3];
    let preparation = prepare_compaction(&path_entries, DEFAULT_COMPACTION_SETTINGS)
        .expect("prepare succeeds")
        .expect("preparation exists");
    assert_eq!(
        preparation.previous_summary.as_deref(),
        Some("First summary")
    );
    assert!(!preparation.retained_tail.is_empty());
    let context_messages = build_session_context(&path_entries, None, &background_context())
        .await
        .expect("context builds");
    #[expect(
        clippy::cast_possible_wrap,
        reason = "token estimates are far below i64::MAX; the preparation's tokens_before is i64"
    )]
    let tokens_before = estimate_context_tokens(&context_messages).tokens as i64;
    assert_eq!(preparation.tokens_before, tokens_before);
}

#[tokio::test]
async fn carries_a_previous_compactions_retained_tail_into_the_next_preparation() {
    // The expected tail reuses these constructed messages — upstream's
    // `toEqual` compares the same instances, and a rebuilt assistant
    // message carries a fresh timestamp.
    let retained_user = create_user_message("retained user");
    let retained_assistant = AgentMessage::Standard(Message::Assistant(create_assistant_message(
        "retained assistant",
        create_mock_usage(100, 50, 0, 0),
    )));
    let new_user = create_user_message("new user");
    let new_assistant = AgentMessage::Standard(Message::Assistant(create_assistant_message(
        "new assistant",
        create_mock_usage(100, 50, 0, 0),
    )));
    let compaction = create_compaction_entry_with(
        "previous summary",
        None,
        vec![retained_user.clone(), retained_assistant.clone()],
        None,
    );
    let user = create_message_entry(new_user.clone(), Some(entry_id(&compaction)));
    let assistant = create_message_entry(new_assistant.clone(), Some(entry_id(&user)));

    let preparation = prepare_compaction(
        &[compaction, user, assistant],
        CompactionSettings {
            enabled: true,
            reserve_tokens: 100,
            keep_recent_tokens: 1,
        },
    )
    .expect("prepare succeeds")
    .expect("preparation exists");
    assert_eq!(
        preparation.previous_summary.as_deref(),
        Some("previous summary")
    );
    let concatenated: Vec<AgentMessage> = preparation
        .messages_to_summarize
        .iter()
        .chain(preparation.turn_prefix_messages.iter())
        .chain(preparation.retained_tail.iter())
        .cloned()
        .collect();
    assert_eq!(
        concatenated,
        vec![retained_user, retained_assistant, new_user, new_assistant]
    );
}

#[tokio::test]
async fn prepares_split_turn_compaction_with_prior_file_operation_details() {
    let u1 = create_message_entry(create_user_message("user msg 1"), None);
    let assistant_message = {
        let mut message =
            create_assistant_message("assistant msg 1", create_mock_usage(100, 50, 0, 0));
        message.content = vec![AssistantBlock::ToolCall(ToolCall {
            id: "tool-1".to_owned(),
            name: "write".to_owned(),
            arguments: json!({ "path": "written.ts" })
                .as_object()
                .expect("object")
                .clone(),
            thought_signature: None,
            namespace: None,
        })];
        message
    };
    let a1 = create_message_entry(
        AgentMessage::Standard(Message::Assistant(assistant_message)),
        Some(entry_id(&u1)),
    );
    let compaction1 = create_compaction_entry_with(
        "First summary",
        Some(entry_id(&a1)),
        Vec::new(),
        Some(
            json!({ "readFiles": ["old-read.ts"], "modifiedFiles": ["old-edit.ts", "written.ts"] }),
        ),
    );
    let u2 = create_message_entry(
        create_user_message("large turn"),
        Some(entry_id(&compaction1)),
    );
    let a2 = create_message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "large assistant message",
            create_mock_usage(100, 50, 0, 0),
        ))),
        Some(entry_id(&u2)),
    );
    let preparation = prepare_compaction(
        &[u1, a1, compaction1, u2, a2],
        CompactionSettings {
            enabled: true,
            reserve_tokens: 100,
            keep_recent_tokens: 1,
        },
    )
    .expect("prepare succeeds")
    .expect("preparation exists");

    assert_eq!(
        preparation.previous_summary.as_deref(),
        Some("First summary")
    );
    assert!(preparation.is_split_turn);
    let roles: Vec<&str> = preparation
        .turn_prefix_messages
        .iter()
        .map(message_role)
        .collect();
    assert_eq!(roles, ["user"]);
    assert!(
        preparation
            .file_ops
            .read
            .contains(&"old-read.ts".to_owned())
    );
    assert!(
        preparation
            .file_ops
            .edited
            .contains(&"old-edit.ts".to_owned())
    );
    assert!(
        preparation
            .file_ops
            .edited
            .contains(&"written.ts".to_owned())
    );
}

#[tokio::test]
async fn does_not_prepare_compaction_when_there_is_nothing_valid_to_compact() {
    let compaction = create_compaction_entry("already compacted", None);
    assert_eq!(
        prepare_compaction(&[compaction], DEFAULT_COMPACTION_SETTINGS).expect("prepare succeeds"),
        None
    );
    assert_eq!(
        prepare_compaction(&[], DEFAULT_COMPACTION_SETTINGS).expect("prepare succeeds"),
        None
    );
}

#[tokio::test]
async fn serializes_conversation_with_truncated_tool_results() {
    let long_content = "x".repeat(5000);
    let messages = vec![Message::ToolResult(pi_ai::types::ToolResultMessage {
        tool_call_id: "tc1".to_owned(),
        tool_name: "read".to_owned(),
        content: vec![pi_ai::types::ToolResultBlock::Text(
            pi_ai::types::TextContent {
                text: long_content,
                text_signature: None,
            },
        )],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: NOW,
    })];
    let result = serialize_conversation(&messages);
    assert!(result.contains("[Tool result]:"));
    assert!(result.contains("[... 3000 more characters truncated]"));
}

#[tokio::test]
async fn passes_reasoning_through_generate_summary_only_for_reasoning_models_with_thinking_enabled()
{
    let messages = vec![create_user_message("Summarize this.")];
    let seen_options: SeenOptions = Arc::new(Mutex::new(Vec::new()));
    let models = test_models();

    let faux_reasoning = create_faux_model(&models, true, 8192);
    faux_reasoning.set_responses([options_recorder(
        &seen_options,
        text_message("## Goal\nTest summary"),
    )]);
    let reasoning_model = faux_reasoning.first_model();
    generate_summary(
        &messages,
        &models,
        &reasoning_model,
        2000,
        None,
        None,
        Some(ThinkingLevel::Medium),
        None,
        None,
        &background_context(),
    )
    .await
    .expect("summary succeeds");
    assert_eq!(
        seen(&seen_options)[0].reasoning,
        Some(pi_ai::types::ThinkingLevel::Medium)
    );

    let faux_off = create_faux_model(&models, true, 8192);
    faux_off.set_responses([options_recorder(
        &seen_options,
        text_message("## Goal\nTest summary"),
    )]);
    let off_model = faux_off.first_model();
    generate_summary(
        &messages,
        &models,
        &off_model,
        2000,
        None,
        None,
        Some(ThinkingLevel::Off),
        None,
        None,
        &background_context(),
    )
    .await
    .expect("summary succeeds");
    assert_eq!(seen(&seen_options)[1].reasoning, None);

    let faux_non_reasoning = create_faux_model(&models, false, 8192);
    faux_non_reasoning.set_responses([options_recorder(
        &seen_options,
        text_message("## Goal\nTest summary"),
    )]);
    let non_reasoning_model = faux_non_reasoning.first_model();
    generate_summary(
        &messages,
        &models,
        &non_reasoning_model,
        2000,
        None,
        None,
        Some(ThinkingLevel::Medium),
        None,
        None,
        &background_context(),
    )
    .await
    .expect("summary succeeds");
    assert_eq!(seen(&seen_options)[2].reasoning, None);
}

#[tokio::test]
async fn includes_previous_summaries_and_custom_instructions_in_generate_summary_prompts() {
    let messages = vec![create_user_message("Summarize this.")];
    let seen_prompt = Arc::new(Mutex::new(String::new()));
    let models = test_models();
    let faux = create_faux_model(&models, false, 8192);
    faux.set_responses([prompt_recorder(&seen_prompt)]);

    let summary = generate_summary_with_usage(
        &messages,
        &models,
        &faux.first_model(),
        2000,
        Some("focus".to_owned()),
        Some("old summary".to_owned()),
        None,
        None,
        None,
        &background_context(),
    )
    .await
    .expect("summary succeeds");

    assert!(summary.text.contains("Test summary"));
    assert!(summary.usage.input > 0);
    assert!(summary.usage.output > 0);
    assert_eq!(
        summary.usage.total_tokens,
        summary.usage.input
            + summary.usage.output
            + summary.usage.cache_read
            + summary.usage.cache_write
    );
    let prompt_text = seen_prompt
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    assert!(prompt_text.contains("<previous-summary>\nold summary\n</previous-summary>"));
    assert!(prompt_text.contains("Additional focus: focus"));
}

#[tokio::test]
async fn preserves_the_string_result_from_generate_summary() {
    let messages = vec![create_user_message("Summarize this.")];
    let models = test_models();
    let faux = create_faux_model(&models, false, 8192);
    faux.set_responses([text_response("## Goal\nTest summary")]);

    let summary = generate_summary(
        &messages,
        &models,
        &faux.first_model(),
        2000,
        None,
        None,
        None,
        None,
        None,
        &background_context(),
    )
    .await
    .expect("summary succeeds");
    assert_eq!(summary, "## Goal\nTest summary");
}

#[tokio::test]
async fn returns_error_results_for_failed_or_aborted_summary_generations() {
    let messages = vec![create_user_message("Summarize this.")];
    let models = test_models();
    let error_faux = create_faux_model(&models, false, 8192);
    error_faux.set_responses([status_response("", StopReason::Error, Some("boom"))]);
    let error_result = generate_summary(
        &messages,
        &models,
        &error_faux.first_model(),
        2000,
        None,
        None,
        None,
        None,
        None,
        &background_context(),
    )
    .await;
    let Err(error) = error_result else {
        panic!("expected an error result");
    };
    assert_eq!(error.code, CompactionErrorCode::SummarizationFailed);
    assert_eq!(error.message, "Summarization failed: boom");

    let aborted_faux = create_faux_model(&models, false, 8192);
    aborted_faux.set_responses([status_response("", StopReason::Aborted, Some("stopped"))]);
    let aborted_result = generate_summary(
        &messages,
        &models,
        &aborted_faux.first_model(),
        2000,
        None,
        None,
        None,
        None,
        None,
        &background_context(),
    )
    .await;
    let Err(error) = aborted_result else {
        panic!("expected an error result");
    };
    assert_eq!(error.code, CompactionErrorCode::Aborted);
    assert_eq!(error.message, "stopped");
}

#[tokio::test]
async fn clamps_compaction_summary_max_tokens_to_the_model_output_cap() {
    let messages = vec![create_user_message("Summarize this.")];
    let seen_options: SeenOptions = Arc::new(Mutex::new(Vec::new()));
    let models = test_models();
    let faux = create_faux_model(&models, false, 128_000);
    faux.set_responses([
        options_recorder(&seen_options, text_message("## Goal\nTest summary")),
        options_recorder(&seen_options, text_message("## Goal\nTest summary")),
    ]);
    let preparation = preparation(&messages, true, 600_000, 500_000);

    compact(
        &preparation,
        &models,
        &faux.first_model(),
        None,
        None,
        None,
        None,
        &background_context(),
    )
    .await
    .expect("compact succeeds");

    let max_tokens: Vec<Option<u64>> = seen(&seen_options)
        .iter()
        .map(|options| options.max_tokens)
        .collect();
    assert_eq!(max_tokens, [Some(128_000), Some(128_000)]);
    let cache_retention: Vec<Option<CacheRetention>> = seen(&seen_options)
        .iter()
        .map(|options| options.cache_retention)
        .collect();
    assert_eq!(
        cache_retention,
        [Some(CacheRetention::None), Some(CacheRetention::None)]
    );
    let session_ids: Vec<Option<String>> = seen(&seen_options)
        .iter()
        .map(|options| options.session_id.clone())
        .collect();
    assert!(session_ids.iter().all(Option::is_some));
    assert_ne!(session_ids[0], session_ids[1]);
}

#[tokio::test]
async fn retains_per_request_retries_for_non_harness_compaction_callers() {
    let messages = vec![create_user_message("Summarize this.")];
    let preparation = preparation(&messages, false, 100, 2_000);
    let models = test_models();
    let faux = create_faux_model(&models, false, 8192);
    faux.set_responses([
        status_response("", StopReason::Error, Some("rate limit exceeded")),
        text_response("recovered summary"),
    ]);

    let result = compact(
        &preparation,
        &models,
        &faux.first_model(),
        None,
        None,
        Some(&pi_ai::utils::retry::RetryPolicy {
            enabled: true,
            max_retries: 1,
            base_delay_ms: 0,
            max_agent_delay_ms: None,
        }),
        None,
        &background_context(),
    )
    .await
    .expect("compact succeeds");
    assert!(result.summary.contains("recovered summary"));
}

#[tokio::test]
async fn returns_compaction_error_results_without_throwing() {
    let messages = vec![create_user_message("Summarize this.")];
    let preparation = preparation(&messages, false, 100, 2_000);
    let models = test_models();
    let history_faux = create_faux_model(&models, false, 8192);
    history_faux.set_responses([status_response(
        "",
        StopReason::Error,
        Some("history failed"),
    )]);
    let result = compact(
        &preparation,
        &models,
        &history_faux.first_model(),
        None,
        None,
        None,
        None,
        &background_context(),
    )
    .await;
    let Err(error) = result else {
        panic!("expected an error result");
    };
    assert_eq!(error.code, CompactionErrorCode::SummarizationFailed);
    assert_eq!(error.message, "Summarization failed: history failed");
}

#[tokio::test]
async fn combines_usage_for_split_turn_compaction_summaries() {
    let messages = vec![create_user_message("Summarize this.")];
    let models = test_models();
    let faux = create_faux_model(&models, false, 8192);
    let model = faux.first_model();
    let history_usage = create_mock_usage(1, 2, 3, 4);
    let turn_prefix_usage = create_mock_usage(5, 6, 7, 8);
    // Upstream stubs `Models.completeSimple` to dequeue these two
    // responses; the port intercepts at the WithRequest boundary the stub
    // models.
    let remaining = Arc::new(Mutex::new(VecDeque::from(vec![
        {
            let mut message =
                faux_assistant_message("history summary", FauxAssistantMessageOptions::default());
            message.usage = history_usage;
            message
        },
        {
            let mut message = faux_assistant_message(
                "turn prefix summary",
                FauxAssistantMessageOptions::default(),
            );
            message.usage = turn_prefix_usage;
            message
        },
    ])));
    let request: SummaryRequest = Arc::new(
        move |_ai_context: &AiContext,
              _options: &SimpleStreamOptions,
              _context: &pi_agent_core::harness::context::Context| {
            let response = remaining
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .pop_front()
                .expect("No faux completeSimple response queued");
            Box::pin(async move { response })
        },
    );
    let preparation = CompactionPreparation {
        messages_to_summarize: messages.clone(),
        turn_prefix_messages: messages.clone(),
        is_split_turn: true,
        tokens_before: 100,
        retained_tail: messages.clone(),
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: CompactionSettings {
            enabled: true,
            reserve_tokens: 2_000,
            keep_recent_tokens: 20,
        },
    };

    let result = compact_with_request(
        &preparation,
        &CompactGenerationOptions {
            model,
            custom_instructions: None,
            thinking_level: None,
        },
        &request,
        &background_context(),
    )
    .await
    .expect("compact succeeds");

    assert_eq!(result.usage, Some(create_mock_usage(6, 8, 10, 12)));
}

#[tokio::test]
async fn passes_reasoning_through_turn_prefix_summaries_when_enabled() {
    let messages = vec![create_user_message("Summarize this.")];
    let seen_options: SeenOptions = Arc::new(Mutex::new(Vec::new()));
    let models = test_models();
    let faux = create_faux_model(&models, true, 8192);
    faux.set_responses([options_recorder(
        &seen_options,
        text_message("## Original Request\nTest summary"),
    )]);
    let preparation = CompactionPreparation {
        messages_to_summarize: Vec::new(),
        turn_prefix_messages: messages.clone(),
        retained_tail: messages.clone(),
        is_split_turn: true,
        tokens_before: 100,
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: CompactionSettings {
            enabled: true,
            reserve_tokens: 2_000,
            keep_recent_tokens: 20,
        },
    };

    compact(
        &preparation,
        &models,
        &faux.first_model(),
        None,
        Some(ThinkingLevel::High),
        None,
        None,
        &background_context(),
    )
    .await
    .expect("compact succeeds");

    assert_eq!(
        seen(&seen_options)[0].reasoning,
        Some(pi_ai::types::ThinkingLevel::High)
    );
}

#[tokio::test]
async fn returns_turn_prefix_compaction_errors_without_throwing() {
    let messages = vec![create_user_message("Summarize this.")];
    let preparation = CompactionPreparation {
        messages_to_summarize: Vec::new(),
        turn_prefix_messages: messages.clone(),
        retained_tail: messages.clone(),
        is_split_turn: true,
        tokens_before: 100,
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: CompactionSettings {
            enabled: true,
            reserve_tokens: 2_000,
            keep_recent_tokens: 20,
        },
    };
    let models = test_models();
    let faux = create_faux_model(&models, false, 8192);
    faux.set_responses([status_response(
        "",
        StopReason::Error,
        Some("prefix failed"),
    )]);

    let result = compact(
        &preparation,
        &models,
        &faux.first_model(),
        None,
        None,
        None,
        None,
        &background_context(),
    )
    .await;
    let Err(error) = result else {
        panic!("expected an error result");
    };
    assert_eq!(error.code, CompactionErrorCode::SummarizationFailed);
    assert_eq!(
        error.message,
        "Turn prefix summarization failed: prefix failed"
    );

    let aborted_faux = create_faux_model(&models, false, 8192);
    aborted_faux.set_responses([status_response(
        "",
        StopReason::Aborted,
        Some("prefix stopped"),
    )]);
    let result = compact(
        &preparation,
        &models,
        &aborted_faux.first_model(),
        None,
        None,
        None,
        None,
        &background_context(),
    )
    .await;
    let Err(error) = result else {
        panic!("expected an error result");
    };
    assert_eq!(error.code, CompactionErrorCode::Aborted);
    assert_eq!(error.message, "prefix stopped");
}

#[tokio::test]
async fn returns_a_compaction_result_with_file_details() {
    let u1 = create_message_entry(create_user_message("read a file"), None);
    let assistant_message = {
        let mut message =
            create_assistant_message("calling tool", create_mock_usage(1000, 200, 0, 0));
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
    let a1 = create_message_entry(
        AgentMessage::Standard(Message::Assistant(assistant_message)),
        Some(entry_id(&u1)),
    );
    let u2 = create_message_entry(create_user_message("continue"), Some(entry_id(&a1)));
    let a2 = create_message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "done",
            create_mock_usage(4000, 500, 0, 0),
        ))),
        Some(entry_id(&u2)),
    );
    let preparation = prepare_compaction(&[u1, a1, u2, a2], DEFAULT_COMPACTION_SETTINGS)
        .expect("prepare succeeds")
        .expect("preparation exists");
    let models = test_models();
    let faux = create_faux_model(&models, false, 8192);
    faux.set_responses([text_response("## Goal\nTest summary")]);
    let result = compact(
        &preparation,
        &models,
        &faux.first_model(),
        None,
        None,
        None,
        None,
        &background_context(),
    )
    .await
    .expect("compact succeeds");
    assert!(!result.summary.is_empty());
    assert!(result.usage.expect("usage present").total_tokens > 0);
    assert!(!result.retained_tail.is_empty());
    let details: CompactionDetails =
        serde_json::from_value(result.details.expect("details present")).expect("details shape");
    // The whole path sits inside the recent-token budget, so nothing was
    // summarized and both lists are empty.
    assert_eq!(details, CompactionDetails::default());
}

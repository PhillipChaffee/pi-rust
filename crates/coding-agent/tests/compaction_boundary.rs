//! The compaction boundary suite: branches the 1:1 suites do not reach,
//! pinned against upstream at `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` —
//! the estimate arms, the cut-point metadata scan-back, the file-op
//! extraction from previous-compaction details, the split-turn usage merge,
//! and the LLM converter's custom-role arms.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod common;

use std::sync::{Arc, Mutex};

use pi_agent_core::types::{AgentMessage, CustomAgentMessage, StreamFn};
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, AssistantMessageEvent, ImageContent, KnownApi, Message,
    ProviderId, StopReason, TextContent, ThinkingContent, ToolCall, ToolResultBlock,
    ToolResultMessage, Usage, UserBlock, UserContent, UserMessage,
};
use pi_ai::utils::event_stream::create_assistant_message_event_stream;

use pi_coding_agent::compaction::{
    CompactionPreparation, CompactionSettings, DEFAULT_COMPACTION_SETTINGS, FileOperations,
    calculate_context_tokens, estimate_context_tokens, estimate_tokens, find_cut_point,
    find_turn_start_index, get_summarization_failure, prepare_compaction,
};
use pi_coding_agent::messages::{
    COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX, bash_execution_to_text, convert_to_llm,
    create_custom_message,
};
use pi_coding_agent::session_manager::entries::{
    CompactionEntry, MessageEntry, SessionEntry, SessionEntryBase,
};

const fn usage_of(input: u64, output: u64, total: u64) -> Usage {
    Usage {
        input,
        output,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: total,
        cost: pi_ai::types::UsageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    }
}

fn assistant_usage_message(stop_reason: StopReason) -> AgentMessage {
    AgentMessage::Standard(Message::Assistant(AssistantMessage {
        content: Vec::new(),
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("anthropic".to_owned()),
        model: "claude-sonnet-4-5".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage_of(100, 50, 150),
        stop_reason,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }))
}

#[test]
fn calculate_context_tokens_prefers_the_native_total() {
    let usage = usage_of(100, 50, 999);
    assert_eq!(calculate_context_tokens(&usage), 999);
}

#[test]
fn error_and_aborted_usage_are_skipped_by_the_estimator() {
    let error = estimate_context_tokens(&[assistant_usage_message(StopReason::Error)]);
    assert_eq!(error.usage_tokens, 0);
    let aborted = estimate_context_tokens(&[assistant_usage_message(StopReason::Aborted)]);
    assert_eq!(aborted.usage_tokens, 0);
    // A valid assistant message anchors.
    let anchored = estimate_context_tokens(&[assistant_usage_message(StopReason::Stop)]);
    assert_eq!(anchored.usage_tokens, 150);
    assert_eq!(anchored.last_usage_index, Some(0));
}

#[test]
fn deferred_stop_reason_counts_as_valid_usage() {
    // Only aborted/error are skipped; every other settlement counts.
    let deferred = estimate_context_tokens(&[assistant_usage_message(StopReason::Deferred)]);
    assert_eq!(deferred.usage_tokens, 150);
}

fn bash_execution(exclude_from_context: bool, exit_code: Option<i64>) -> CustomAgentMessage {
    let mut data = serde_json::Map::new();
    data.insert(
        "command".to_owned(),
        serde_json::Value::String("ls -la".to_owned()),
    );
    data.insert(
        "output".to_owned(),
        serde_json::Value::String("file-one\nfile-two".to_owned()),
    );
    data.insert(
        "exitCode".to_owned(),
        exit_code.map_or(serde_json::Value::Null, serde_json::Value::from),
    );
    data.insert("cancelled".to_owned(), serde_json::Value::Bool(false));
    data.insert("truncated".to_owned(), serde_json::Value::Bool(false));
    if exclude_from_context {
        data.insert(
            "excludeFromContext".to_owned(),
            serde_json::Value::Bool(true),
        );
    }
    CustomAgentMessage {
        role: "bashExecution".to_owned(),
        timestamp: 0,
        data,
    }
}

#[test]
fn bash_execution_to_text_renders_the_command_block() {
    let text = bash_execution_to_text(&bash_execution(false, Some(0)));
    assert_eq!(text, "Ran `ls -la`\n```\nfile-one\nfile-two\n```");
}

#[test]
fn bash_execution_to_text_renders_the_no_output_form() {
    let mut message = bash_execution(false, Some(0));
    message.data.insert(
        "output".to_owned(),
        serde_json::Value::String(String::new()),
    );
    let text = bash_execution_to_text(&message);
    assert_eq!(text, "Ran `ls -la`\n(no output)");
}

#[test]
fn bash_execution_to_text_renders_cancelled_and_exit_code() {
    let mut cancelled = bash_execution(false, Some(0));
    cancelled
        .data
        .insert("cancelled".to_owned(), serde_json::Value::Bool(true));
    assert!(bash_execution_to_text(&cancelled).ends_with("(command cancelled)"));

    let failed = bash_execution_to_text(&bash_execution(false, Some(2)));
    assert!(failed.ends_with("\n\nCommand exited with code 2"));

    // exitCode 0 renders neither suffix.
    let success = bash_execution_to_text(&bash_execution(false, Some(0)));
    assert!(!success.contains("exited with code"));
    assert!(!success.contains("cancelled"));
}

#[test]
fn bash_execution_to_text_renders_the_truncation_pointer() {
    let mut message = bash_execution(false, Some(0));
    message
        .data
        .insert("truncated".to_owned(), serde_json::Value::Bool(true));
    message.data.insert(
        "fullOutputPath".to_owned(),
        serde_json::Value::String("/tmp/full.log".to_owned()),
    );
    let text = bash_execution_to_text(&message);
    assert!(text.ends_with("\n\n[Output truncated. Full output: /tmp/full.log]"));

    // truncated without a path renders no pointer, upstream's `&&`.
    let mut pathless = bash_execution(false, Some(0));
    pathless
        .data
        .insert("truncated".to_owned(), serde_json::Value::Bool(true));
    assert!(!bash_execution_to_text(&pathless).contains("Output truncated"));
}

#[test]
fn convert_to_llm_drops_excluded_bash_executions() {
    let messages = vec![AgentMessage::Custom(bash_execution(true, Some(0)))];
    assert!(convert_to_llm(&messages).is_empty());
}

#[test]
fn convert_to_llm_wraps_bash_executions_in_user_text() {
    let messages = vec![AgentMessage::Custom(bash_execution(false, None))];
    let converted = convert_to_llm(&messages);
    assert_eq!(converted.len(), 1);
    match &converted[0] {
        Message::User(user) => {
            match &user.content {
                UserContent::Text(text) => assert!(text.starts_with("Ran `ls -la`\n")),
                UserContent::Blocks(_) => panic!("bash execution converts to user text"),
            }
            assert_eq!(user.timestamp, 0);
        }
        _ => panic!("bash execution converts to a user message"),
    }
}

#[test]
fn convert_to_llm_wraps_custom_string_content_into_a_text_block() {
    let custom = create_custom_message(
        "test",
        UserContent::Text("hello".to_owned()),
        true,
        None,
        "2026-01-01T00:00:00.000Z",
    );
    let messages = vec![AgentMessage::Custom(custom)];
    let converted = convert_to_llm(&messages);
    match &converted[0] {
        Message::User(user) => match &user.content {
            UserContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UserBlock::Text(_)));
            }
            UserContent::Text(_) => panic!("string content becomes a text block array"),
        },
        _ => panic!("custom converts to a user message"),
    }
}

#[test]
fn convert_to_llm_carries_branch_and_compaction_summaries_in_their_prefixes() {
    let mut data = serde_json::Map::new();
    data.insert(
        "summary".to_owned(),
        serde_json::Value::String("did things".to_owned()),
    );
    let branch = CustomAgentMessage {
        role: "branchSummary".to_owned(),
        timestamp: 5,
        data: data.clone(),
    };
    let compaction = CustomAgentMessage {
        role: "compactionSummary".to_owned(),
        timestamp: 6,
        data,
    };
    let converted = convert_to_llm(&[
        AgentMessage::Custom(branch),
        AgentMessage::Custom(compaction),
    ]);
    assert_eq!(converted.len(), 2);
    let texts: Vec<String> = converted
        .iter()
        .map(|message| match message {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => text.clone(),
                UserContent::Blocks(_) => panic!("summary converts to user text"),
            },
            _ => panic!("summary converts to a user message"),
        })
        .collect();
    assert!(texts[0].starts_with(
        "The following is a summary of a branch that this conversation came back from:"
    ));
    assert!(texts[0].ends_with("<summary>\ndid things</summary>"));
    assert!(texts[1].starts_with(
        "The conversation history before this point was compacted into the following summary:"
    ));
    assert!(texts[1].ends_with("<summary>\ndid things\n</summary>"));
}

#[test]
fn estimate_tokens_arms_match_the_role_budgets() {
    // bashExecution: command + output chars.
    let bash = AgentMessage::Custom(bash_execution(false, None));
    let bash_chars = "ls -la".chars().count() + "file-one\nfile-two".chars().count();
    assert_eq!(estimate_tokens(&bash), bash_chars.div_ceil(4) as u64);

    // branchSummary/compactionSummary: summary chars.
    let mut data = serde_json::Map::new();
    data.insert(
        "summary".to_owned(),
        serde_json::Value::String("abcd".to_owned()),
    );
    let branch = AgentMessage::Custom(CustomAgentMessage {
        role: "branchSummary".to_owned(),
        timestamp: 0,
        data: data.clone(),
    });
    assert_eq!(estimate_tokens(&branch), 1);

    // custom string content.
    let mut content = serde_json::Map::new();
    content.insert(
        "content".to_owned(),
        serde_json::Value::String("abcdef".to_owned()),
    );
    let custom = AgentMessage::Custom(CustomAgentMessage {
        role: "custom".to_owned(),
        timestamp: 0,
        data: content,
    });
    assert_eq!(estimate_tokens(&custom), 2);

    // custom block content with an image (4800 chars) and text.
    let mut blocks = serde_json::Map::new();
    blocks.insert(
        "content".to_owned(),
        serde_json::Value::Array(vec![
            serde_json::json!({ "type": "image" }),
            serde_json::json!({ "type": "text", "text": "xy" }),
        ]),
    );
    let image_custom = AgentMessage::Custom(CustomAgentMessage {
        role: "custom".to_owned(),
        timestamp: 0,
        data: blocks,
    });
    assert_eq!(
        estimate_tokens(&image_custom),
        (4800usize + 2).div_ceil(4) as u64
    );

    // Unknown custom roles estimate zero, upstream's fall-through.
    let unknown = AgentMessage::Custom(CustomAgentMessage {
        role: "mystery".to_owned(),
        timestamp: 0,
        data: serde_json::Map::new(),
    });
    assert_eq!(estimate_tokens(&unknown), 0);

    // toolResult with an image.
    let result = AgentMessage::Standard(Message::ToolResult(ToolResultMessage {
        tool_call_id: "tc".to_owned(),
        tool_name: "read".to_owned(),
        content: vec![
            ToolResultBlock::Image(ImageContent {
                data: "x".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
            ToolResultBlock::Text(TextContent {
                text: "hi".to_owned(),
                text_signature: None,
            }),
        ],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 0,
    }));
    assert_eq!(estimate_tokens(&result), (4800usize + 2).div_ceil(4) as u64);
}

#[test]
fn estimate_tokens_uses_the_assistant_block_budgets() {
    let message = AgentMessage::Standard(Message::Assistant(AssistantMessage {
        content: vec![
            AssistantBlock::Text(TextContent {
                text: "abcd".to_owned(),
                text_signature: None,
            }),
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "ef".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantBlock::ToolCall(ToolCall {
                id: "tc".to_owned(),
                name: "read".to_owned(),
                namespace: None,
                arguments: serde_json::from_str(r#"{"path":"a.md"}"#).expect("args"),
                thought_signature: None,
            }),
        ],
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("anthropic".to_owned()),
        model: "m".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage_of(0, 0, 0),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }));
    let args_json = serde_json::to_string(&serde_json::json!({"path":"a.md"})).expect("json");
    let chars = 4 + 2 + "read".len() + args_json.len();
    assert_eq!(estimate_tokens(&message), chars.div_ceil(4) as u64);
}

#[test]
fn find_turn_start_index_returns_minus_one_without_a_turn_start() {
    let mut chain = common::compaction::EntryChain::new();
    // Only assistant messages: no turn start before the index.
    let entries = vec![
        chain.message_entry(assistant_message_text("a")),
        chain.message_entry(assistant_message_text("b")),
    ];
    assert_eq!(find_turn_start_index(&entries, 1, 0), -1);
}

fn assistant_message_text(text: &str) -> AgentMessage {
    let _ = text;
    AgentMessage::Standard(Message::Assistant(AssistantMessage {
        content: Vec::new(),
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("anthropic".to_owned()),
        model: "m".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage_of(0, 0, 0),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }))
}

#[test]
fn find_cut_point_includes_adjacent_metadata_entries() {
    let mut chain = common::compaction::EntryChain::new();
    // A context-invisible entry directly before the cut point pulls the cut
    // back to include it (upstream's scan-back loop).
    let label = SessionEntry::Label(pi_coding_agent::session_manager::entries::LabelEntry {
        base: SessionEntryBase {
            id: Some("label".to_owned()),
            parent_id: None,
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        target_id: "x".to_owned(),
        label: Some("note".to_owned()),
        extras: serde_json::Map::new(),
    });
    let entries = vec![
        chain.message_entry(user_message_of("u1")),
        label,
        chain.message_entry(user_message_of("u2")),
        chain.message_entry(assistant_message_text("a")),
        chain.message_entry(user_message_of("u3")),
    ];
    // Budget 2 lands the cut at u2 (index 2); the scan-back then pulls the
    // context-invisible label (index 1) into the kept range, and the split
    // turn reports the user message that started it.
    let result = find_cut_point(&entries, 0, entries.len(), 2);
    assert_eq!(result.first_kept_entry_index, 1);
    assert_eq!(result.turn_start_index, 0);
    assert!(result.is_split_turn);
}

fn user_message_of(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp: 0,
    }))
}

#[test]
fn prepare_compaction_reads_file_lists_from_the_previous_pi_generated_compaction() {
    let mut chain = common::compaction::EntryChain::new();
    let u1 = chain.message_entry(user_message_of("u1"));
    let u1_id = match &u1 {
        SessionEntry::Message(message) => message.base.id.clone().expect("id"),
        _ => unreachable!(),
    };
    let compaction = SessionEntry::Compaction(CompactionEntry {
        base: SessionEntryBase {
            id: Some("comp".to_owned()),
            parent_id: Some(u1_id.clone()),
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        summary: "prev".to_owned(),
        first_kept_entry_id: Some(u1_id),
        tokens_before: 100,
        details: Some(serde_json::json!({
            "readFiles": ["a.txt"],
            "modifiedFiles": ["b.txt"]
        })),
        usage: None,
        from_hook: None,
        extras: serde_json::Map::new(),
    });
    let path = vec![
        u1,
        compaction,
        chain.message_entry(user_message_of(&"u2 ".repeat(200))),
    ];

    let settings = CompactionSettings {
        keep_recent_tokens: 1,
        ..DEFAULT_COMPACTION_SETTINGS
    };
    let preparation = prepare_compaction(&path, settings).expect("preparation");

    // The previous compaction's file lists seed the accumulator, so b.txt
    // counts modified even though the summarized range never touches it.
    assert!(preparation.file_ops.read.contains("a.txt"));
    assert!(preparation.file_ops.edited.contains("b.txt"));
}

#[test]
fn prepare_compaction_skips_extension_generated_previous_details() {
    let mut chain = common::compaction::EntryChain::new();
    let u1 = chain.message_entry(user_message_of("u1"));
    let u1_id = match &u1 {
        SessionEntry::Message(message) => message.base.id.clone().expect("id"),
        _ => unreachable!(),
    };
    let compaction = SessionEntry::Compaction(CompactionEntry {
        base: SessionEntryBase {
            id: Some("comp".to_owned()),
            parent_id: Some(u1_id.clone()),
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        summary: "prev".to_owned(),
        first_kept_entry_id: Some(u1_id),
        tokens_before: 100,
        details: Some(serde_json::json!({
            "readFiles": ["a.txt"],
            "modifiedFiles": ["b.txt"]
        })),
        usage: None,
        from_hook: Some(true),
        extras: serde_json::Map::new(),
    });
    let path = vec![
        u1,
        compaction,
        chain.message_entry(user_message_of(&"u2 ".repeat(200))),
    ];

    let settings = CompactionSettings {
        keep_recent_tokens: 1,
        ..DEFAULT_COMPACTION_SETTINGS
    };
    let preparation = prepare_compaction(&path, settings).expect("preparation");
    assert!(!preparation.file_ops.read.contains("a.txt"));
    assert!(!preparation.file_ops.edited.contains("b.txt"));
}

#[test]
fn prepare_compaction_tolerates_partial_details() {
    let mut chain = common::compaction::EntryChain::new();
    let u1 = chain.message_entry(user_message_of("u1"));
    let u1_id = match &u1 {
        SessionEntry::Message(message) => message.base.id.clone().expect("id"),
        _ => unreachable!(),
    };
    let compaction = SessionEntry::Compaction(CompactionEntry {
        base: SessionEntryBase {
            id: Some("comp".to_owned()),
            parent_id: Some(u1_id.clone()),
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        summary: "prev".to_owned(),
        first_kept_entry_id: Some(u1_id),
        tokens_before: 100,
        // Only modifiedFiles: upstream's Array.isArray skips the missing
        // list and keeps the present one.
        details: Some(serde_json::json!({ "modifiedFiles": ["b.txt"] })),
        usage: None,
        from_hook: None,
        extras: serde_json::Map::new(),
    });
    let path = vec![
        u1,
        compaction,
        chain.message_entry(user_message_of(&"u2 ".repeat(200))),
    ];

    let settings = CompactionSettings {
        keep_recent_tokens: 1,
        ..DEFAULT_COMPACTION_SETTINGS
    };
    let preparation = prepare_compaction(&path, settings).expect("preparation");
    assert!(!preparation.file_ops.read.contains("a.txt"));
    assert!(preparation.file_ops.edited.contains("b.txt"));
}

#[test]
fn get_summarization_failure_pins_the_error_and_length_messages() {
    let mut response = AssistantMessage {
        content: Vec::new(),
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("p".to_owned()),
        model: "m".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage_of(0, 0, 0),
        stop_reason: StopReason::Error,
        deferred: None,
        error_message: Some("boom".to_owned()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    };
    assert_eq!(
        get_summarization_failure(&response, "Summarization").as_deref(),
        Some("Summarization failed: boom")
    );
    response.error_message = None;
    assert_eq!(
        get_summarization_failure(&response, "Summarization").as_deref(),
        Some("Summarization failed: Unknown error")
    );
    response.stop_reason = StopReason::Length;
    assert_eq!(
        get_summarization_failure(&response, "Branch summarization").as_deref(),
        Some(
            "Branch summarization failed: generation hit the token cap and the summary is incomplete"
        )
    );
    response.stop_reason = StopReason::Stop;
    assert!(get_summarization_failure(&response, "Summarization").is_none());
}

/// The split-turn compaction's fixture model.
fn model_fixture() -> pi_ai::types::Model {
    pi_ai::types::Model {
        id: "m".to_owned(),
        name: "m".to_owned(),
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("p".to_owned()),
        base_url: "https://example.test".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![],
        cost: pi_ai::types::ModelCost {
            rates: pi_ai::types::ModelCostRates {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 200_000,
        max_tokens: 8192,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The split-turn merge mock: serves the default stop response for both
/// summarization calls and records the settled messages.
struct TwoCallMock {
    captured: Mutex<Vec<AssistantMessage>>,
    default: AssistantMessage,
}

#[tokio::test]
async fn split_turn_compaction_merges_history_and_prefix_usage() {
    let message = AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: "part".to_owned(),
            text_signature: None,
        })],
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("p".to_owned()),
        model: "m".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage {
            input: 3,
            output: 4,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: Some(1),
            total_tokens: 7,
            cost: pi_ai::types::UsageCost {
                input: 0.5,
                output: 1.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 1.5,
            },
        },
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    };
    let mock = Arc::new(TwoCallMock {
        captured: Mutex::new(Vec::new()),
        default: message,
    });
    let mock_for_stream = Arc::clone(&mock);
    let stream_fn: StreamFn = Arc::new(move |_model, _context, _options| {
        mock_for_stream
            .captured
            .lock()
            .expect("captured lock")
            .push(mock_for_stream.default.clone());
        let stream = create_assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: mock_for_stream.default.stop_reason,
            message: mock_for_stream.default.clone(),
        });
        stream
    });

    let preparation = CompactionPreparation {
        first_kept_entry_id: "keep".to_owned(),
        messages_to_summarize: vec![user_message_of("history")],
        turn_prefix_messages: vec![user_message_of("prefix")],
        is_split_turn: true,
        tokens_before: 100,
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: CompactionSettings {
            enabled: true,
            reserve_tokens: 2000,
            keep_recent_tokens: 20,
        },
    };

    let result = pi_coding_agent::compaction::compact(
        preparation,
        &model_fixture(),
        None,
        None,
        None,
        None,
        None,
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("compaction");

    // The merged summary joins the two texts with the split-turn separator.
    assert!(
        result
            .summary
            .starts_with("part\n\n---\n\n**Turn Context (split turn):**\n\npart")
    );
    // combineUsage sums both calls, keeping the optional reasoning arm.
    assert_eq!(result.usage.input, 6);
    assert_eq!(result.usage.output, 8);
    assert_eq!(result.usage.total_tokens, 14);
    assert_eq!(result.usage.reasoning, Some(2));
    assert!((result.usage.cost.total - 3.0).abs() < f64::EPSILON);
    // The details carry the file lists.
    let details = result.details.expect("details");
    assert_eq!(details["readFiles"], serde_json::json!([]));
    assert_eq!(details["modifiedFiles"], serde_json::json!([]));
    assert_eq!(mock.captured.lock().expect("captured lock").len(), 2);
}

#[test]
fn prepare_compaction_returns_none_when_the_path_ends_in_a_compaction() {
    let mut chain = common::compaction::EntryChain::new();
    let u1 = chain.message_entry(user_message_of("u1"));
    let u1_id = match &u1 {
        SessionEntry::Message(message) => message.base.id.clone().expect("id"),
        _ => unreachable!(),
    };
    let path = vec![u1, chain.compaction_entry("done", &u1_id)];
    assert!(prepare_compaction(&path, DEFAULT_COMPACTION_SETTINGS).is_none());
}

#[test]
fn prepare_compaction_returns_none_when_nothing_is_summarizable() {
    let mut chain = common::compaction::EntryChain::new();
    let path = vec![chain.message_entry(user_message_of("tiny"))];
    assert!(prepare_compaction(&path, DEFAULT_COMPACTION_SETTINGS).is_none());
}

/// The prefix/suffix constants re-exported for the converter tests; the
/// format helper keeps the expected strings honest in one place.
#[test]
fn summary_prefixes_compose_with_the_suffix() {
    let composed = format!("{COMPACTION_SUMMARY_PREFIX}BODY{COMPACTION_SUMMARY_SUFFIX}");
    assert!(composed.starts_with("The conversation history before this point was compacted"));
    assert!(composed.ends_with("<summary>\nBODY\n</summary>"));
}

/// A message entry with no message payload projects nothing and cannot host
/// a cut point (upstream's `sessionEntryToContextMessages(entry)[0]` guard).
#[test]
fn empty_message_entries_never_host_cut_points() {
    let empty = SessionEntry::Message(MessageEntry {
        base: SessionEntryBase {
            id: Some("empty".to_owned()),
            parent_id: None,
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        message: None,
        extras: serde_json::Map::new(),
    });
    assert!(pi_coding_agent::session_manager::typed_entry_to_context_messages(&empty).is_empty());
    let result = find_cut_point(&[empty], 0, 1, 1000);
    assert_eq!(result.first_kept_entry_index, 0);
    assert_eq!(result.turn_start_index, -1);
    assert!(!result.is_split_turn);
}

// ============================================================================
// The file-op belt and the serializer's remaining arms
// ============================================================================

fn tool_call(name: &str, path: &str) -> AssistantBlock {
    AssistantBlock::ToolCall(ToolCall {
        id: "tc".to_owned(),
        name: name.to_owned(),
        namespace: None,
        arguments: serde_json::from_str(&format!(r#"{{"path":"{path}"}}"#)).expect("args"),
        thought_signature: None,
    })
}

fn assistant_with_blocks(blocks: Vec<AssistantBlock>) -> AgentMessage {
    AgentMessage::Standard(Message::Assistant(AssistantMessage {
        content: blocks,
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("p".to_owned()),
        model: "m".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage_of(0, 0, 0),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }))
}

#[test]
fn extract_file_ops_from_message_collects_read_write_edit_paths() {
    let mut file_ops = pi_coding_agent::compaction::create_file_ops();
    let message = assistant_with_blocks(vec![
        tool_call("read", "a.txt"),
        tool_call("write", "b.txt"),
        tool_call("edit", "c.txt"),
        tool_call("bash", "ignored"), // non-belt tool, no `path` match
        tool_call("read", "a.txt"),   // duplicate collapses
    ]);
    pi_coding_agent::compaction::extract_file_ops_from_message(&message, &mut file_ops);
    assert_eq!(file_ops.read, std::iter::once("a.txt".to_owned()).collect());
    assert_eq!(
        file_ops.written,
        std::iter::once("b.txt".to_owned()).collect()
    );
    assert_eq!(
        file_ops.edited,
        std::iter::once("c.txt".to_owned()).collect()
    );

    // A tool call without a string path contributes nothing.
    let mut file_ops = pi_coding_agent::compaction::create_file_ops();
    let message = AgentMessage::Standard(Message::Assistant(AssistantMessage {
        content: vec![AssistantBlock::ToolCall(ToolCall {
            id: "tc".to_owned(),
            name: "read".to_owned(),
            namespace: None,
            arguments: serde_json::from_str(r#"{"file":"a.txt"}"#).expect("args"),
            thought_signature: None,
        })],
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("p".to_owned()),
        model: "m".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage_of(0, 0, 0),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }));
    pi_coding_agent::compaction::extract_file_ops_from_message(&message, &mut file_ops);
    assert!(file_ops.read.is_empty());

    // Non-assistant messages contribute nothing.
    pi_coding_agent::compaction::extract_file_ops_from_message(
        &user_message_of("hi"),
        &mut file_ops,
    );
    assert!(file_ops.read.is_empty());
}

#[test]
fn compute_file_lists_merges_written_and_edited_and_excludes_reads() {
    let mut file_ops = pi_coding_agent::compaction::create_file_ops();
    file_ops.read.insert("read-only.txt".to_owned());
    file_ops.read.insert("b.txt".to_owned());
    file_ops.written.insert("b.txt".to_owned());
    file_ops.edited.insert("a.txt".to_owned());
    file_ops.edited.insert("b.txt".to_owned());

    let lists = pi_coding_agent::compaction::compute_file_lists(&file_ops);
    // modified = edited ∪ written, sorted; read-only excludes modified.
    assert_eq!(
        lists.modified_files,
        vec!["a.txt".to_owned(), "b.txt".to_owned()]
    );
    assert_eq!(lists.read_files, ["read-only.txt".to_owned()]);
}

#[test]
fn format_file_operations_composes_the_xml_sections() {
    // Both sections, blank-line separated.
    let both = pi_coding_agent::compaction::format_file_operations(
        &["r.txt".to_owned()],
        &["m.txt".to_owned()],
    );
    assert_eq!(
        both,
        "\n\n<read-files>\nr.txt\n</read-files>\n\n<modified-files>\nm.txt\n</modified-files>"
    );

    // Single sections.
    let reads = pi_coding_agent::compaction::format_file_operations(&["r.txt".to_owned()], &[]);
    assert_eq!(reads, "\n\n<read-files>\nr.txt\n</read-files>");
    let modified = pi_coding_agent::compaction::format_file_operations(&[], &["m.txt".to_owned()]);
    assert_eq!(modified, "\n\n<modified-files>\nm.txt\n</modified-files>");

    // Empty in, empty out.
    assert_eq!(
        pi_coding_agent::compaction::format_file_operations(&[], &[]),
        ""
    );
}

#[test]
fn serialize_conversation_renders_the_assistant_sections() {
    let message = AgentMessage::Standard(Message::Assistant(AssistantMessage {
        content: vec![
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "pondering".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantBlock::Text(TextContent {
                text: "the answer".to_owned(),
                text_signature: None,
            }),
            tool_call("read", "a.md"),
        ],
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("p".to_owned()),
        model: "m".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage_of(0, 0, 0),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }));

    let serialized =
        pi_coding_agent::compaction::serialize_conversation(&convert_to_llm(&[message]));
    assert!(serialized.contains("[Assistant thinking]: pondering"));
    assert!(serialized.contains("[Assistant]: the answer"));
    assert!(serialized.contains("[Assistant tool calls]: read(path=\"a.md\")"));

    // Multiple thinking blocks join with a newline; multiple tool calls with
    // `; `.
    let message = AgentMessage::Standard(Message::Assistant(AssistantMessage {
        content: vec![
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "one".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "two".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            tool_call("read", "a.md"),
            tool_call("edit", "b.md"),
        ],
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("p".to_owned()),
        model: "m".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage_of(0, 0, 0),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }));
    let serialized =
        pi_coding_agent::compaction::serialize_conversation(&convert_to_llm(&[message]));
    assert!(serialized.contains("[Assistant thinking]: one\ntwo"));
    assert!(serialized.contains("read(path=\"a.md\"); edit(path=\"b.md\")"));

    // No text block, no [Assistant] line.
    let message = assistant_with_blocks(vec![tool_call("read", "a.md")]);
    let serialized =
        pi_coding_agent::compaction::serialize_conversation(&convert_to_llm(&[message]));
    assert!(!serialized.contains("[Assistant]:"));
}

#[test]
fn serialize_conversation_joins_parts_with_blank_lines() {
    let converted = convert_to_llm(&[
        user_message_of("hi"),
        assistant_with_blocks(vec![AssistantBlock::Text(TextContent {
            text: "hello".to_owned(),
            text_signature: None,
        })]),
    ]);
    let serialized = pi_coding_agent::compaction::serialize_conversation(&converted);
    assert_eq!(serialized, "[User]: hi\n\n[Assistant]: hello");

    // Blank content contributes no part, upstream's `if (content)` guard.
    let empty = AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Text(String::new()),
        timestamp: 0,
    }));
    let converted = convert_to_llm(&[empty, user_message_of("only")]);
    assert_eq!(
        pi_coding_agent::compaction::serialize_conversation(&converted),
        "[User]: only"
    );
}

// ============================================================================
// The compat fallback path and the zero-reserve clamp
// ============================================================================

/// The faux registration guard upstream's `registrations` bookkeeping
/// restates; the provider unregisters when the test ends.
struct RegistrationGuard(pi_ai::providers::faux::FauxProviderRegistration);

impl std::ops::Deref for RegistrationGuard {
    type Target = pi_ai::providers::faux::FauxProviderRegistration;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for RegistrationGuard {
    fn drop(&mut self) {
        self.0.unregister();
    }
}

/// Without a streamFn the summarization routes through
/// [`pi_ai::compat::complete_simple`]; the faux provider scripts the
/// response.
#[tokio::test]
async fn generate_summary_routes_complete_simple_through_the_compat_registry() {
    let registration = RegistrationGuard(pi_ai::compat::register_faux_provider(
        pi_ai::providers::faux::RegisterFauxProviderOptions {
            models: vec![pi_ai::providers::faux::FauxModelDefinition {
                id: "faux-summary".to_owned(),
                ..pi_ai::providers::faux::FauxModelDefinition::default()
            }],
            ..pi_ai::providers::faux::RegisterFauxProviderOptions::default()
        },
    ));
    registration.set_responses([pi_ai::providers::faux::faux_assistant_message(
        vec![pi_ai::providers::faux::faux_text("## Goal\nfaux summary")],
        pi_ai::providers::faux::FauxAssistantMessageOptions::default(),
    )
    .into()]);

    let summary = pi_coding_agent::compaction::generate_summary(
        &[user_message_of("summarize me")],
        &registration.first_model(),
        2000,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    assert_eq!(summary, "## Goal\nfaux summary");
}

/// A zero reserve floors the output cap at zero, upstream's
/// `Math.floor(0.8 * 0)`.
#[tokio::test]
async fn a_zero_reserve_clamps_the_output_cap_to_zero() {
    let mock = Arc::new(TwoCallMock::default());
    let mock_for_stream = Arc::clone(&mock);
    let stream_fn: StreamFn = Arc::new(move |_model, _context, options| {
        if let Some(options) = options {
            assert_eq!(options.max_tokens, Some(0));
        }
        let stream = create_assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: mock_for_stream.default.stop_reason,
            message: mock_for_stream.default.clone(),
        });
        stream
    });

    let summary = pi_coding_agent::compaction::generate_summary(
        &[user_message_of("summarize me")],
        &model_fixture(),
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(&stream_fn),
        None,
        None,
        None,
        None,
    )
    .await
    .expect("summary");

    assert_eq!(summary, "part");
}

impl Default for TwoCallMock {
    fn default() -> Self {
        Self {
            captured: Mutex::new(Vec::new()),
            default: AssistantMessage {
                content: vec![AssistantBlock::Text(TextContent {
                    text: "part".to_owned(),
                    text_signature: None,
                })],
                api: Api::from(KnownApi::AnthropicMessages),
                provider: ProviderId("p".to_owned()),
                model: "m".to_owned(),
                response_model: None,
                response_id: None,
                provider_thinking_level: None,
                diagnostics: None,
                usage: usage_of(0, 0, 0),
                stop_reason: StopReason::Stop,
                deferred: None,
                error_message: None,
                raw_stop_reason: None,
                end_turn: None,
                timestamp: 0,
            },
        }
    }
}

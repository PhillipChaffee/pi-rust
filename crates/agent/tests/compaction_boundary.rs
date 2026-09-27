//! The compaction boundary suite: the paths upstream's
//! `test/harness/compaction.test.ts` does not reach — the cut-point role
//! arms, the estimator's per-block edges, the serializer's truncation
//! boundaries, and the request-option isolation.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod compaction_common;
use compaction_common::*;

use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::harness::compaction::compaction::{
    calculate_context_tokens, compact, create_summary_request_options, estimate_context_tokens,
    estimate_tokens, find_cut_point, find_turn_start_index, should_compact,
};
use pi_agent_core::harness::compaction::types::{
    CompactionPreparation, CompactionSettings, DEFAULT_COMPACTION_SETTINGS, FileOperations,
};
use pi_agent_core::harness::compaction::utils::{
    compute_file_lists, create_file_ops_accumulator, durable_file_operations,
    extract_file_ops_from_message, format_file_operations, serialize_conversation,
};
use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::types::{BranchSummaryEntryBody, Entry};
use pi_agent_core::types::{AgentMessage, CustomAgentMessage, ThinkingLevel};
use pi_ai::providers::faux::{
    FauxAssistantMessageOptions, FauxProviderState, FauxResponseStep, faux_assistant_message,
};
use pi_ai::types::{
    AssistantBlock, Context as AiContext, Message, Model, SimpleStreamOptions, TextContent,
    ToolCall, ToolResultBlock, ToolResultMessage, UserBlock, UserContent, UserMessage,
};
use std::sync::MutexGuard;
use std::sync::PoisonError;

fn seen(snapshot: &SeenOptions) -> MutexGuard<'_, Vec<SimpleStreamOptions>> {
    snapshot.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The factory step that records the request options each call saw.
fn options_recorder(seen_options: &SeenOptions) -> FauxResponseStep {
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

/// The custom-role message one entry carries, the duck-typed shapes the
/// estimator walks.
fn custom_message(role: &str, data: &serde_json::Value) -> AgentMessage {
    AgentMessage::Custom(CustomAgentMessage {
        role: role.to_owned(),
        timestamp: NOW,
        data: data.as_object().expect("object").clone(),
    })
}

fn message_entry(message: AgentMessage, parent_id: Option<String>) -> Entry {
    create_message_entry(message, parent_id)
}

#[test]
fn a_branch_summary_message_entry_is_a_cut_point() {
    // The walk skips empty messages until the branch-summary message
    // crosses the budget: the first cut point at or past it is itself.
    let user = message_entry(create_user_message(""), None);
    let branch_summary_message = message_entry(
        custom_message(
            "branchSummary",
            &serde_json::json!({ "summary": "branch", "fromId": "x" }),
        ),
        Some(entry_id(&user)),
    );
    let silent_assistant = message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "",
            create_mock_usage(0, 0, 0, 0),
        ))),
        Some(entry_id(&branch_summary_message)),
    );

    let result = find_cut_point(&[user, branch_summary_message, silent_assistant], 0, 3, 1);
    assert_eq!(result.first_kept_entry_index, 1);
}

#[test]
fn an_unknown_custom_role_is_never_a_cut_point() {
    // The budget is never crossed, so the cut stays at the first valid cut
    // point: the user message, not the unknown role before it.
    let unknown = message_entry(custom_message("unknown", &serde_json::json!({})), None);
    let user = message_entry(create_user_message("user"), Some(entry_id(&unknown)));

    let result = find_cut_point(&[unknown, user], 0, 2, u64::MAX);
    assert_eq!(result.first_kept_entry_index, 1);
}

#[test]
fn the_turn_start_walk_runs_past_compactions_and_assistant_messages() {
    let user = message_entry(create_user_message("user"), None);
    let assistant = message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "a",
            create_mock_usage(100, 50, 0, 0),
        ))),
        Some(entry_id(&user)),
    );
    let compaction = create_compaction_entry("summary", Some(entry_id(&assistant)));
    let trailing_assistant = message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "b",
            create_mock_usage(100, 50, 0, 0),
        ))),
        Some(entry_id(&compaction)),
    );

    assert_eq!(
        find_turn_start_index(&[user, assistant, compaction, trailing_assistant], 3, 0),
        Some(0)
    );
}

#[test]
fn a_bash_execution_message_starts_a_turn() {
    let bash = message_entry(
        custom_message(
            "bashExecution",
            &serde_json::json!({ "command": "ls", "output": "", "exitCode": 0, "cancelled": false, "truncated": false }),
        ),
        None,
    );
    let assistant = message_entry(
        AgentMessage::Standard(Message::Assistant(create_assistant_message(
            "a",
            create_mock_usage(100, 50, 0, 0),
        ))),
        Some(entry_id(&bash)),
    );

    assert_eq!(find_turn_start_index(&[bash, assistant], 1, 0), Some(0));
}

#[test]
fn a_cut_below_a_custom_entry_walks_the_decrement() {
    // The budget is never crossed, so the cut stays at the first cut
    // point; the walk back steps over the custom entry to reach it.
    let custom = create_custom_entry("note", None);
    let user = message_entry(create_user_message("user"), Some(entry_id(&custom)));

    let result = find_cut_point(&[custom, user], 0, 2, u64::MAX);
    assert_eq!(result.first_kept_entry_index, 0);
}

#[test]
fn the_threshold_compares_in_the_signed_number_domain() {
    let settings = CompactionSettings {
        enabled: true,
        reserve_tokens: 16_384,
        keep_recent_tokens: 20_000,
    };
    // A window below the reserve makes the difference negative; any
    // non-negative token count exceeds it.
    assert!(should_compact(0, 0, &settings));
    // The sum path: a zero reported total falls back to the components.
    assert_eq!(
        calculate_context_tokens(&create_mock_usage(10, 5, 3, 2)),
        20
    );
}

#[test]
fn the_file_lists_dedup_sort_and_subtract() {
    let mut file_ops = create_file_ops_accumulator();
    file_ops.read.insert("b.ts".to_owned());
    file_ops.read.insert("a.ts".to_owned());
    file_ops.written.insert("a.ts".to_owned());
    file_ops.written.insert("c.ts".to_owned());
    file_ops.written.insert("c.ts".to_owned());

    let lists = compute_file_lists(&durable_file_operations(&file_ops));
    assert_eq!(lists.read_files, ["b.ts".to_owned()]);
    assert_eq!(lists.modified_files, ["a.ts".to_owned(), "c.ts".to_owned()]);

    // A read of a modified file drops from the read-only list, and the
    // durable form carries sorted vectors.
    let durable = durable_file_operations(&file_ops);
    assert_eq!(durable.read, ["a.ts".to_owned(), "b.ts".to_owned()]);
}

#[test]
fn the_file_operation_tags_render_each_shape() {
    let read_only = ["r.ts".to_owned()];
    let modified = ["m.ts".to_owned()];
    assert_eq!(
        format_file_operations(&read_only, &[]),
        "\n\n<read-files>\nr.ts\n</read-files>"
    );
    assert_eq!(
        format_file_operations(&[], &modified),
        "\n\n<modified-files>\nm.ts\n</modified-files>"
    );
    assert_eq!(
        format_file_operations(&read_only, &modified),
        "\n\n<read-files>\nr.ts\n</read-files>\n\n<modified-files>\nm.ts\n</modified-files>"
    );
    assert_eq!(format_file_operations(&[], &[]), "");
}

#[test]
fn the_conversation_serializer_renders_each_message_shape() {
    let user_blocks = vec![Message::User(UserMessage {
        content: UserContent::Blocks(vec![
            UserBlock::Text(TextContent {
                text: "a".to_owned(),
                text_signature: None,
            }),
            UserBlock::Text(TextContent {
                text: "b".to_owned(),
                text_signature: None,
            }),
        ]),
        timestamp: NOW,
    })];
    assert_eq!(serialize_conversation(&user_blocks), "[User]: ab");

    let empty_user = vec![Message::User(UserMessage {
        content: UserContent::Text(String::new()),
        timestamp: NOW,
    })];
    assert_eq!(serialize_conversation(&empty_user), "");

    let assistant = vec![Message::Assistant({
        let mut message = create_assistant_message("reply", create_mock_usage(0, 0, 0, 0));
        message.content = vec![
            AssistantBlock::Thinking(pi_ai::types::ThinkingContent {
                thinking: "h1".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantBlock::Thinking(pi_ai::types::ThinkingContent {
                thinking: "h2".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantBlock::Text(TextContent {
                text: "reply".to_owned(),
                text_signature: None,
            }),
            AssistantBlock::ToolCall(ToolCall {
                id: "call-1".to_owned(),
                name: "read".to_owned(),
                arguments: serde_json::json!({ "path": "f.ts", "n": 3 })
                    .as_object()
                    .expect("object")
                    .clone(),
                thought_signature: None,
                namespace: None,
            }),
        ];
        message
    })];
    assert_eq!(
        serialize_conversation(&assistant),
        "[Assistant thinking]: h1\nh2\n\n[Assistant]: reply\n\n[Assistant tool calls]: read(path=\"f.ts\", n=3)"
    );

    let boundary = "x".repeat(2000);
    let over = "x".repeat(2001);
    let tool_result = |text: &str| {
        vec![Message::ToolResult(ToolResultMessage {
            tool_call_id: "tc1".to_owned(),
            tool_name: "read".to_owned(),
            content: vec![ToolResultBlock::Text(TextContent {
                text: text.to_owned(),
                text_signature: None,
            })],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: NOW,
        })]
    };
    assert_eq!(
        serialize_conversation(&tool_result(&boundary)),
        format!("[Tool result]: {boundary}")
    );
    assert_eq!(
        serialize_conversation(&tool_result(&over)),
        format!("[Tool result]: {boundary}\n\n[... 1 more characters truncated]")
    );
}

#[test]
fn the_estimator_walks_each_block_shape() {
    // An image-only user message estimates at the fixed image budget.
    let image_user = AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Blocks(vec![UserBlock::Image(pi_ai::types::ImageContent {
            data: "abc".to_owned(),
            mime_type: "image/png".to_owned(),
        })]),
        timestamp: NOW,
    }));
    assert_eq!(estimate_tokens(&image_user), 4800 / 4);

    // An assistant with no content estimates to zero.
    assert_eq!(
        estimate_tokens(&AgentMessage::Standard(Message::Assistant(
            create_assistant_message("", create_mock_usage(0, 0, 0, 0))
        ))),
        0
    );

    // A custom message without a content field, and a bash execution with
    // neither command nor output, estimate to zero.
    assert_eq!(
        estimate_tokens(&custom_message(
            "custom",
            &serde_json::json!({ "customType": "note" })
        )),
        0
    );
    assert_eq!(
        estimate_tokens(&custom_message(
            "bashExecution",
            &serde_json::json!({ "exitCode": 0, "cancelled": false, "truncated": false })
        )),
        0
    );

    // The context estimate sums the trailing messages after the last usage
    // block.
    let assistant = AgentMessage::Standard(Message::Assistant(create_assistant_message(
        "abcd",
        create_mock_usage(10, 5, 3, 2),
    )));
    let estimate = estimate_context_tokens(&[
        create_user_message("12345678"),
        assistant,
        create_user_message("x"),
    ]);
    assert_eq!(estimate.usage_tokens, 20);
    assert_eq!(estimate.last_usage_index, Some(1));
    assert_eq!(estimate.trailing_tokens, 1);
    assert_eq!(estimate.tokens, 21);
}

#[tokio::test]
async fn the_summary_request_options_isolate_the_request() {
    let context = background_context();
    let options = SimpleStreamOptions {
        session_id: Some("kept".to_owned()),
        ..SimpleStreamOptions::default()
    };

    let isolated = create_summary_request_options(options, &context);
    // An existing session id is kept, not regenerated.
    assert_eq!(isolated.session_id.as_deref(), Some("kept"));
    assert_eq!(
        isolated.cache_retention,
        Some(pi_ai::types::CacheRetention::None)
    );
    // The telemetry parent is the context's, here the shared no-op.
    assert!(isolated.telemetry_context.is_some());
    // A context without an abort signal carries no cancellation token.
    assert!(isolated.transport_options.signal.is_none());

    // A signal-bearing context links one.
    let (signal_context, _controller) =
        pi_agent_core::harness::context::with_cancel(&background_context());
    let linked = create_summary_request_options(SimpleStreamOptions::default(), &signal_context);
    assert!(linked.transport_options.signal.is_some());

    let regenerated = create_summary_request_options(SimpleStreamOptions::default(), &context);
    let fresh = regenerated.session_id.expect("a fresh session id");
    let fresh_again = create_summary_request_options(SimpleStreamOptions::default(), &context)
        .session_id
        .expect("a fresh session id");
    assert_ne!(fresh, fresh_again);
}

#[tokio::test]
async fn the_summary_output_caps_take_the_reserve_fractions_below_the_model_cap() {
    let seen_options: SeenOptions = Arc::new(Mutex::new(Vec::new()));
    let models = test_models();
    let faux = create_faux_model(&models, false, 128_000);
    faux.set_responses([
        options_recorder(&seen_options),
        options_recorder(&seen_options),
    ]);

    let messages = vec![create_user_message("Summarize this.")];
    let preparation = CompactionPreparation {
        messages_to_summarize: messages.clone(),
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
        Some(ThinkingLevel::Off),
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
    // The history request takes four fifths of the reserve, the turn
    // prefix half, both far below the model's 128000 cap.
    assert_eq!(max_tokens, [Some(1_600), Some(1_000)]);
}

#[test]
fn the_file_op_extraction_walks_the_tool_call_names() {
    let tool_call = |name: &str, path: &str| {
        AssistantBlock::ToolCall(ToolCall {
            id: "call-1".to_owned(),
            name: name.to_owned(),
            arguments: serde_json::json!({ "path": path })
                .as_object()
                .expect("object")
                .clone(),
            thought_signature: None,
            namespace: None,
        })
    };
    let message = AgentMessage::Standard(Message::Assistant({
        let mut assistant = create_assistant_message("", create_mock_usage(0, 0, 0, 0));
        assistant.content = vec![
            tool_call("read", "r.ts"),
            tool_call("write", "w.ts"),
            tool_call("edit", "e.ts"),
            tool_call("list", "ignored.ts"),
            tool_call("read", "r.ts"),
        ];
        assistant
    }));

    let mut file_ops = create_file_ops_accumulator();
    extract_file_ops_from_message(&message, &mut file_ops);
    let durable = durable_file_operations(&file_ops);
    assert_eq!(durable.read, ["r.ts".to_owned()]);
    assert_eq!(durable.written, ["w.ts".to_owned()]);
    assert_eq!(durable.edited, ["e.ts".to_owned()]);

    // Non-assistant messages and tool calls without a path contribute
    // nothing.
    let mut file_ops = create_file_ops_accumulator();
    extract_file_ops_from_message(&create_user_message("user"), &mut file_ops);
    extract_file_ops_from_message(
        &AgentMessage::Standard(Message::Assistant({
            let mut assistant = create_assistant_message("", create_mock_usage(0, 0, 0, 0));
            assistant.content = vec![tool_call("read", "")];
            assistant
        })),
        &mut file_ops,
    );
    assert_eq!(
        durable_file_operations(&file_ops),
        FileOperations::default()
    );
}

#[test]
fn a_custom_entry_never_carries_a_message() {
    // The entry extractors skip custom entries; the shape is bound through
    // the cut-point walk, which treats them as neither cut points nor
    // turn starts.
    let custom = create_custom_entry("note", None);
    let entries = [custom];
    assert_eq!(find_turn_start_index(&entries, 0, 0), None);
    let result = find_cut_point(&entries, 0, 1, 1);
    assert_eq!(result.first_kept_entry_index, 0);
    assert!(!result.is_split_turn());
}

#[test]
fn the_branch_summary_fixture_rides_the_entry_body() {
    // The branch-summary entry body the cut-point tests build carries its
    // summary through the entry, upstream's inline literal.
    let entry = create_branch_summary_entry(None, "branch", "branch summary");
    let Entry::BranchSummary {
        body: BranchSummaryEntryBody {
            summary, from_id, ..
        },
        ..
    } = &entry
    else {
        panic!("expected a branch-summary entry");
    };
    assert_eq!(summary, "branch summary");
    assert_eq!(from_id.as_deref(), Some("branch"));
}

#[test]
fn the_message_entry_fixture_rides_the_body() {
    let entry = create_message_entry(create_user_message("user"), None);
    let Entry::Message { body, .. } = &entry else {
        panic!("expected a message entry");
    };
    assert_eq!(message_role(&body.message), "user");
    assert_eq!(body.terminate, None);
}

#[test]
fn the_default_settings_carry_upstreams_values() {
    assert_eq!(
        DEFAULT_COMPACTION_SETTINGS,
        CompactionSettings {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
        }
    );
}

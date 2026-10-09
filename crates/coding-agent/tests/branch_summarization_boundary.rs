//! The branch-summarization boundary suite: the entry collection and
//! preparation machinery the 1:1 suite does not reach (upstream exercises
//! them through the AgentSession suites, which ride their own tickets),
//! pinned against upstream at `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use std::sync::{Arc, Mutex};

use pi_agent_core::types::{AgentMessage, StreamFn};
use pi_ai::providers::faux::{FauxAssistantMessageOptions, faux_assistant_message};
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, AssistantMessageEvent, KnownApi, Model, ProviderId,
    StopReason,
};
use pi_ai::utils::event_stream::create_assistant_message_event_stream;
use pi_ai::utils::uuid::uuidv7;
use tokio_util::sync::CancellationToken;

use pi_coding_agent::compaction::{
    BranchSummaryDetails, GenerateBranchSummaryOptions, collect_entries_for_branch_summary,
    estimate_tokens, generate_branch_summary, prepare_branch_entries,
};
use pi_coding_agent::session_manager::SessionManager;

fn model() -> Model {
    Model {
        id: "test-model".to_owned(),
        name: "Test Model".to_owned(),
        api: Api::from(KnownApi::AnthropicMessages),
        provider: ProviderId("anthropic".to_owned()),
        base_url: "https://api.anthropic.com".to_owned(),
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

/// A tree the collection walks: root `r` with children `a`, `b`; `a` has
/// children `a1`, `a2`. The manager builds it in file order.
fn manager_with_tree() -> SessionManager {
    let entries: Vec<pi_coding_agent::session_manager::FileEntry> = vec![
        message_entry("r", None, "root request", 1),
        message_entry("a", Some("r"), "branch a request", 2),
        message_entry("a1", Some("a"), "branch a1 request", 3),
        message_entry("a2", Some("a1"), "branch a2 request", 4),
        message_entry("b", Some("r"), "branch b request", 5),
    ];
    let header = r#"{"type":"session","id":"s","version":3,"timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#;
    let mut content = header.to_owned();
    for entry in &entries {
        content.push('\n');
        content.push_str(&serde_json::to_string(entry).expect("entry serializes"));
    }
    let dir = std::env::temp_dir().join(format!(
        "pi-branch-boundary-{}",
        uuidv7(None).expect("uuid")
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("session.jsonl");
    std::fs::write(&path, &content).expect("write session");
    SessionManager::open(path.to_str().expect("path"), None, None).expect("session opens")
}

fn message_entry(
    id: &str,
    parent: Option<&str>,
    text: &str,
    timestamp: i64,
) -> pi_coding_agent::session_manager::FileEntry {
    use pi_coding_agent::session_manager::entries::{MessageEntry, SessionEntry, SessionEntryBase};
    pi_coding_agent::session_manager::FileEntry::Entry(SessionEntry::Message(MessageEntry {
        base: SessionEntryBase {
            id: Some(id.to_owned()),
            parent_id: parent.map(str::to_owned),
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        message: Some(AgentMessage::Standard(pi_ai::types::Message::User(
            pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text(text.to_owned()),
                timestamp,
            },
        ))),
        extras: serde_json::Map::new(),
    }))
}

#[test]
fn collection_walks_from_the_old_leaf_to_the_common_ancestor() {
    let session = manager_with_tree();

    let result = collect_entries_for_branch_summary(&session, Some("a2"), "b");

    // The abandoned path a2 -> a1 -> a, oldest first; the common ancestor is
    // the root both branches share.
    let ids: Vec<&str> = result
        .entries
        .iter()
        .map(|entry| entry.base().id.as_deref().expect("id"))
        .collect();
    assert_eq!(ids, vec!["a", "a1", "a2"]);
    assert_eq!(result.common_ancestor_id.as_deref(), Some("r"));
}

#[test]
fn collection_without_an_old_position_collects_nothing() {
    let session = manager_with_tree();
    let result = collect_entries_for_branch_summary(&session, None, "b");
    assert!(result.entries.is_empty());
    assert!(result.common_ancestor_id.is_none());
}

#[test]
fn collection_when_the_old_leaf_sits_on_the_target_path() {
    let session = manager_with_tree();
    // Navigating to an ancestor of the current leaf: the common ancestor is
    // the target itself and everything below it summarizes.
    let result = collect_entries_for_branch_summary(&session, Some("a2"), "a");
    let ids: Vec<&str> = result
        .entries
        .iter()
        .map(|entry| entry.base().id.as_deref().expect("id"))
        .collect();
    assert_eq!(ids, vec!["a1", "a2"]);
    assert_eq!(result.common_ancestor_id.as_deref(), Some("a"));
}

/// The pi-generated branch-summary details the first preparation pass reads.
fn branch_summary_entry_with_details(
    from_hook: Option<bool>,
    details: Option<serde_json::Value>,
) -> pi_coding_agent::session_manager::entries::SessionEntry {
    use pi_coding_agent::session_manager::entries::{
        BranchSummaryEntry, SessionEntry, SessionEntryBase,
    };
    SessionEntry::BranchSummary(BranchSummaryEntry {
        base: SessionEntryBase {
            id: Some("bs".to_owned()),
            parent_id: None,
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        from_id: "x".to_owned(),
        summary: "explored".to_owned(),
        details,
        usage: None,
        from_hook,
        extras: serde_json::Map::new(),
    })
}

#[test]
fn preparation_seeds_file_ops_from_pi_generated_branch_details() {
    let entries = vec![
        branch_summary_entry_with_details(
            None,
            Some(serde_json::json!({
                "readFiles": ["kept-a.txt"],
                "modifiedFiles": ["kept-b.txt"]
            })),
        ),
        branch_summary_entry_with_details(
            Some(true),
            Some(serde_json::json!({
                "readFiles": ["skipped-a.txt"],
                "modifiedFiles": ["skipped-b.txt"]
            })),
        ),
    ];

    let preparation = prepare_branch_entries(&entries, 0);

    assert!(preparation.file_ops.read.contains("kept-a.txt"));
    assert!(preparation.file_ops.edited.contains("kept-b.txt"));
    assert!(!preparation.file_ops.read.contains("skipped-a.txt"));
    assert!(!preparation.file_ops.edited.contains("skipped-b.txt"));
}

#[test]
fn preparation_walks_newest_to_oldest_under_the_budget() {
    use pi_agent_core::types::AgentMessage;
    use pi_coding_agent::session_manager::entries::{MessageEntry, SessionEntry, SessionEntryBase};

    let entry = |id: &str, text: &str| -> SessionEntry {
        SessionEntry::Message(MessageEntry {
            base: SessionEntryBase {
                id: Some(id.to_owned()),
                parent_id: None,
                timestamp: String::new(),
                extras: serde_json::Map::new(),
            },
            message: Some(AgentMessage::Standard(pi_ai::types::Message::User(
                pi_ai::types::UserMessage {
                    content: pi_ai::types::UserContent::Text(text.to_owned()),
                    timestamp: 0,
                },
            ))),
            extras: serde_json::Map::new(),
        })
    };
    let entries = vec![entry("old", "0123456789"), entry("new", "0123456789")];
    // 10 chars -> 3 tokens each; a 4-token budget fits the newest and stops
    // before the oldest.
    let preparation = prepare_branch_entries(&entries, 4);
    assert_eq!(preparation.messages.len(), 1);
    match &preparation.messages[0] {
        AgentMessage::Standard(pi_ai::types::Message::User(user)) => {
            assert_eq!(
                user.content,
                pi_ai::types::UserContent::Text("0123456789".to_owned())
            );
        }
        _ => panic!("newest message kept"),
    }
    assert_eq!(preparation.total_tokens, 3);
}

#[test]
fn preparation_unbounded_budget_keeps_every_message_chronologically() {
    use pi_coding_agent::session_manager::entries::SessionEntry;

    let entries: Vec<SessionEntry> = (0..5)
        .map(|index| {
            let mut base = branch_summary_entry_with_details(None, None);
            if let SessionEntry::BranchSummary(branch) = &mut base {
                branch.base.id = Some(format!("id-{index}"));
            }
            base
        })
        .collect();
    let preparation = prepare_branch_entries(&entries, 0);
    // Branch summaries project as their summary messages, chronological.
    assert_eq!(preparation.messages.len(), 5);
    let first = &preparation.messages[0];
    let AgentMessage::Custom(custom) = first else {
        panic!("branch summary message");
    };
    assert_eq!(custom.role, "branchSummary");
    assert_eq!(
        preparation.total_tokens,
        "explored".chars().count().div_ceil(4) as u64 * 5
    );
}

#[test]
fn estimate_tokens_matches_the_char_quarter_heuristic() {
    use pi_coding_agent::session_manager::entries::{MessageEntry, SessionEntry, SessionEntryBase};
    let entry = SessionEntry::Message(MessageEntry {
        base: SessionEntryBase {
            id: None,
            parent_id: None,
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        message: Some(AgentMessage::Standard(pi_ai::types::Message::User(
            pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("abcdefghij".to_owned()),
                timestamp: 0,
            },
        ))),
        extras: serde_json::Map::new(),
    });
    let message = pi_coding_agent::session_manager::typed_entry_to_context_messages(&entry)
        .into_iter()
        .next()
        .expect("message");
    assert_eq!(estimate_tokens(&message), 3);
}

/// The aborted-arm mock upstream's abort cases rest.
fn aborted_stream() -> StreamFn {
    Arc::new(move |_model, _context, _options| {
        let stream = create_assistant_message_event_stream();
        let mut aborted = faux_assistant_message("", FauxAssistantMessageOptions::default());
        aborted.stop_reason = StopReason::Aborted;
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error: aborted,
        });
        stream
    })
}

fn text_response(text: &str) -> AssistantMessage {
    let mut message = faux_assistant_message("", FauxAssistantMessageOptions::default());
    message.content = vec![AssistantBlock::Text(pi_ai::types::TextContent {
        text: text.to_owned(),
        text_signature: None,
    })];
    message.api = Api::from(KnownApi::AnthropicMessages);
    message.provider = ProviderId("anthropic".to_owned());
    "test-model".clone_into(&mut message.model);
    message
}

fn settled_stream(message: AssistantMessage) -> StreamFn {
    Arc::new(move |_model, _context, _options| {
        let stream = create_assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: message.stop_reason,
            message: message.clone(),
        });
        stream
    })
}

fn entries() -> Vec<pi_coding_agent::session_manager::entries::SessionEntry> {
    use pi_coding_agent::session_manager::entries::{MessageEntry, SessionEntry, SessionEntryBase};
    vec![SessionEntry::Message(MessageEntry {
        base: SessionEntryBase {
            id: Some("branch-user".to_owned()),
            parent_id: None,
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        message: Some(AgentMessage::Standard(pi_ai::types::Message::User(
            pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("Abandoned request".to_owned()),
                timestamp: 1,
            },
        ))),
        extras: serde_json::Map::new(),
    })]
}

fn options<'a>(
    model: &'a Model,
    signal: &'a CancellationToken,
    stream_fn: Option<&'a StreamFn>,
    custom_instructions: Option<&'a str>,
    replace_instructions: bool,
) -> GenerateBranchSummaryOptions<'a> {
    GenerateBranchSummaryOptions {
        model,
        api_key: None,
        headers: None,
        env: None,
        signal,
        custom_instructions,
        replace_instructions,
        reserve_tokens: None,
        stream_fn,
        retry: None,
        callbacks: None,
    }
}

#[tokio::test]
async fn generation_reports_aborts() {
    let model = model();
    let signal = CancellationToken::new();
    let stream_fn = aborted_stream();
    let result = generate_branch_summary(
        &entries(),
        &options(&model, &signal, Some(&stream_fn), None, false),
    )
    .await;
    assert!(result.aborted);
    assert!(result.summary.is_none());
    assert!(result.error.is_none());
}

#[tokio::test]
async fn generation_without_messages_reports_the_empty_summary() {
    let model = model();
    let signal = CancellationToken::new();
    let result = generate_branch_summary(&[], &options(&model, &signal, None, None, false)).await;
    assert_eq!(result.summary.as_deref(), Some("No content to summarize"));
    assert!(result.usage.is_none());
    assert!(result.read_files.is_none());
    assert!(result.modified_files.is_none());
}

#[tokio::test]
async fn generation_prepends_the_preamble_and_reports_file_lists() {
    let model = model();
    let signal = CancellationToken::new();

    // A read tool call in the summarized range feeds the file lists.
    let with_tool_call = {
        use pi_coding_agent::session_manager::entries::{
            MessageEntry, SessionEntry, SessionEntryBase,
        };
        vec![SessionEntry::Message(MessageEntry {
            base: SessionEntryBase {
                id: Some("assistant".to_owned()),
                parent_id: None,
                timestamp: String::new(),
                extras: serde_json::Map::new(),
            },
            message: Some(AgentMessage::Standard(pi_ai::types::Message::Assistant(
                AssistantMessage {
                    content: vec![AssistantBlock::ToolCall(pi_ai::types::ToolCall {
                        id: "tc".to_owned(),
                        name: "read".to_owned(),
                        namespace: None,
                        arguments: serde_json::from_str(r#"{"path":"notes.md"}"#).expect("args"),
                        thought_signature: None,
                    })],
                    api: Api::from(KnownApi::AnthropicMessages),
                    provider: ProviderId("anthropic".to_owned()),
                    model: "m".to_owned(),
                    response_model: None,
                    response_id: None,
                    provider_thinking_level: None,
                    diagnostics: None,
                    usage: pi_ai::types::Usage {
                        input: 0,
                        output: 0,
                        cache_read: 0,
                        cache_write: 0,
                        cache_write_1h: None,
                        reasoning: None,
                        total_tokens: 0,
                        cost: pi_ai::types::UsageCost {
                            input: 0.0,
                            output: 0.0,
                            cache_read: 0.0,
                            cache_write: 0.0,
                            total: 0.0,
                        },
                    },
                    stop_reason: StopReason::Stop,
                    deferred: None,
                    error_message: None,
                    raw_stop_reason: None,
                    end_turn: None,
                    timestamp: 0,
                },
            ))),
            extras: serde_json::Map::new(),
        })]
    };

    let stream_fn = settled_stream(text_response("the summary"));
    let result = generate_branch_summary(
        &with_tool_call,
        &options(&model, &signal, Some(&stream_fn), None, false),
    )
    .await;

    let summary = result.summary.expect("summary");
    assert!(
        summary.starts_with(
            "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\nthe summary"
        )
    );
    // The read tool call lands in the appended file lists.
    assert!(summary.contains("<read-files>\nnotes.md\n</read-files>"));
    assert_eq!(
        result.read_files.as_deref(),
        Some(&["notes.md".to_owned()][..])
    );
    assert!(result.usage.is_some());
}

#[tokio::test]
async fn generation_appends_and_replaces_custom_instructions() {
    let model = model();
    let signal = CancellationToken::new();
    let captured = Arc::new(Mutex::new(Vec::new()));

    let captured_for_stream = Arc::clone(&captured);
    let stream_fn: StreamFn = Arc::new(move |_model, context, _options| {
        captured_for_stream
            .lock()
            .expect("captured lock")
            .push(context.clone());
        let stream = create_assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: text_response("s"),
        });
        stream
    });

    // Appended mode: the default prompt plus the focus line.
    generate_branch_summary(
        &entries(),
        &options(&model, &signal, Some(&stream_fn), Some("focus here"), false),
    )
    .await;
    let appended =
        serde_json::to_string(&captured.lock().expect("captured lock")[0]).expect("json");
    assert!(appended.contains("Create a structured summary of this conversation branch"));
    assert!(appended.contains("Additional focus: focus here"));

    // Replace mode: only the custom instructions ride the prompt.
    generate_branch_summary(
        &entries(),
        &options(&model, &signal, Some(&stream_fn), Some("only this"), true),
    )
    .await;
    let replaced =
        serde_json::to_string(&captured.lock().expect("captured lock")[1]).expect("json");
    assert!(replaced.contains("only this"));
    assert!(!replaced.contains("Create a structured summary of this conversation branch"));

    // Replace mode without instructions falls back to the default prompt.
    generate_branch_summary(
        &entries(),
        &options(&model, &signal, Some(&stream_fn), None, true),
    )
    .await;
    let fallback =
        serde_json::to_string(&captured.lock().expect("captured lock")[2]).expect("json");
    assert!(fallback.contains("Create a structured summary of this conversation branch"));

    // The system prompt every request carries.
    for context in captured.lock().expect("captured lock").iter() {
        assert_eq!(
            context.system_prompt.as_deref(),
            Some(pi_coding_agent::compaction::SUMMARIZATION_SYSTEM_PROMPT)
        );
    }
}

#[test]
fn branch_summary_details_serialize_with_the_wire_names() {
    let details = BranchSummaryDetails {
        read_files: vec!["a".to_owned()],
        modified_files: vec!["b".to_owned()],
    };
    let json = serde_json::to_value(&details).expect("serialize");
    assert_eq!(json["readFiles"], serde_json::json!(["a"]));
    assert_eq!(json["modifiedFiles"], serde_json::json!(["b"]));
    let round: BranchSummaryDetails = serde_json::from_value(json).expect("deserialize");
    assert_eq!(round, details);
}
// ============================================================================
// The projection arms prepareBranchEntries walks for summary entries
// ============================================================================

fn session_user_entry(
    id: &str,
    text: &str,
) -> pi_coding_agent::session_manager::entries::SessionEntry {
    use pi_coding_agent::session_manager::entries::{MessageEntry, SessionEntry, SessionEntryBase};
    SessionEntry::Message(MessageEntry {
        base: SessionEntryBase {
            id: Some(id.to_owned()),
            parent_id: None,
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        message: Some(AgentMessage::Standard(pi_ai::types::Message::User(
            pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text(text.to_owned()),
                timestamp: 0,
            },
        ))),
        extras: serde_json::Map::new(),
    })
}

fn compaction_entry(summary: &str) -> pi_coding_agent::session_manager::entries::SessionEntry {
    use pi_coding_agent::session_manager::entries::{
        CompactionEntry, SessionEntry, SessionEntryBase,
    };
    SessionEntry::Compaction(CompactionEntry {
        base: SessionEntryBase {
            id: Some("comp".to_owned()),
            parent_id: None,
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        summary: summary.to_owned(),
        first_kept_entry_id: None,
        tokens_before: 100,
        details: None,
        usage: None,
        from_hook: None,
        extras: serde_json::Map::new(),
    })
}

fn custom_message_entry(content: &str) -> pi_coding_agent::session_manager::entries::SessionEntry {
    use pi_coding_agent::session_manager::entries::{
        CustomMessageEntry, SessionEntry, SessionEntryBase,
    };
    SessionEntry::CustomMessage(CustomMessageEntry {
        base: SessionEntryBase {
            id: Some("cm".to_owned()),
            parent_id: None,
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        custom_type: "test".to_owned(),
        content: Some(pi_ai::types::UserContent::Text(content.to_owned())),
        display: true,
        details: None,
        extras: serde_json::Map::new(),
    })
}

fn role_of(message: &AgentMessage) -> &'static str {
    match message {
        AgentMessage::Custom(custom) => match custom.role.as_str() {
            "compactionSummary" => "compactionSummary",
            "custom" => "custom",
            "branchSummary" => "branchSummary",
            other => panic!("unexpected custom role {other}"),
        },
        AgentMessage::Standard(pi_ai::types::Message::User(_)) => "user",
        AgentMessage::Standard(_) => panic!("unexpected message kind"),
    }
}

#[test]
fn preparation_projects_compaction_and_custom_entries() {
    use pi_coding_agent::session_manager::entries::SessionEntry;

    let entries: Vec<SessionEntry> = vec![
        branch_summary_entry_with_details(None, None),
        compaction_entry("the compacted history"),
        custom_message_entry("extension injected"),
    ];

    let preparation = prepare_branch_entries(&entries, 0);
    let roles: Vec<&str> = preparation.messages.iter().map(role_of).collect();
    // Chronological order: branch summary, compaction summary, custom.
    assert_eq!(roles, vec!["branchSummary", "compactionSummary", "custom"]);
    assert!(preparation.total_tokens > 0);
}

#[test]
fn preparation_skips_metadata_entries() {
    use pi_agent_core::types::AgentMessage;
    use pi_coding_agent::session_manager::entries::{
        ModelChangeEntry, SessionEntry, SessionEntryBase,
    };

    let entries: Vec<SessionEntry> = vec![
        SessionEntry::ModelChange(ModelChangeEntry {
            base: SessionEntryBase {
                id: None,
                parent_id: None,
                timestamp: String::new(),
                extras: serde_json::Map::new(),
            },
            provider: "anthropic".to_owned(),
            model_id: "m".to_owned(),
            extras: serde_json::Map::new(),
        }),
        compaction_entry("the compacted history"),
    ];

    let preparation = prepare_branch_entries(&entries, 0);
    assert_eq!(preparation.messages.len(), 1);
    assert!(
        matches!(&preparation.messages[0], AgentMessage::Custom(custom) if custom.role == "compactionSummary")
    );
}

#[test]
fn preparation_budgeted_summary_entries_fit_when_below_ninety_percent() {
    use pi_coding_agent::session_manager::entries::SessionEntry;

    // A small newest message accumulates little; the older compaction
    // summary then overflows the budget outright but still fits because the
    // accumulated total sits under 90% of the budget, upstream's
    // `totalTokens < tokenBudget * 0.9` rule.
    let entries: Vec<SessionEntry> = vec![
        compaction_entry(&"c".repeat(4000)),
        session_user_entry("small", "0123456789"),
    ];

    let preparation = prepare_branch_entries(&entries, 1000);
    assert_eq!(preparation.messages.len(), 2);
    assert!(matches!(
        &preparation.messages[0],
        AgentMessage::Custom(custom) if custom.role == "compactionSummary"
    ));
}

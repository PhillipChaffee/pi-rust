//! The compaction core suite, upstream `test/compaction.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported 1:1.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "the env-gated live tests log the summary the way upstream's console.log lines do"
)]

mod common;

use common::compaction::{EntryChain, assistant_message, extract_text, mock_usage, user_message};

use pi_agent_core::types::AgentMessage;

use pi_coding_agent::compaction::{
    CompactionSettings, DEFAULT_COMPACTION_SETTINGS, calculate_context_tokens,
    estimate_context_tokens, find_cut_point, get_last_assistant_usage, prepare_compaction,
    should_compact,
};
use pi_coding_agent::session_manager::context::{
    ByIdIndex, LeafId, SessionContext, SessionModel, build_session_context,
};
use pi_coding_agent::session_manager::entries::{CompactionEntry, SessionEntry, SessionEntryBase};
use pi_coding_agent::session_manager::file_entry::FileEntry;
use pi_coding_agent::session_manager::{migrate_session_entries, parse_session_entries};

// ============================================================================
// Test fixtures
// ============================================================================

fn load_large_session_entries() -> Vec<SessionEntry> {
    let session_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/large-session.jsonl"
    );
    let content = std::fs::read_to_string(session_path).expect("fixture read");
    let mut entries = parse_session_entries(&content);
    migrate_session_entries(&mut entries); // Add id/parentId for v1 fixtures
    entries
        .into_iter()
        .filter_map(|entry| match entry {
            FileEntry::Entry(typed) => Some(typed),
            _ => None,
        })
        .collect()
}

/// The free-function `buildSessionContext(entries)` call upstream's tests
/// make: the typed union wrapped into file entries, indexed, default leaf.
fn build_typed_session_context(entries: &[SessionEntry]) -> SessionContext {
    let file_entries: Vec<FileEntry> = entries
        .iter()
        .map(|entry| FileEntry::Entry(entry.clone()))
        .collect();
    let by_id: ByIdIndex = file_entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| entry.entry_id().map(|id| (id.to_owned(), index)))
        .collect();
    build_session_context(&file_entries, LeafId::Default, &by_id)
}

fn message_role(entry: &SessionEntry) -> Option<&'static str> {
    match entry {
        SessionEntry::Message(message) => message.message.as_ref().map(|message| match message {
            AgentMessage::Standard(pi_ai::types::Message::User(_)) => "user",
            AgentMessage::Standard(pi_ai::types::Message::Assistant(_)) => "assistant",
            _ => "",
        }),
        _ => None,
    }
}

fn first_summary(messages: &[AgentMessage]) -> &str {
    match &messages[0] {
        AgentMessage::Custom(custom) => {
            assert_eq!(custom.role, "compactionSummary");
            custom
                .field("summary")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
        }
        AgentMessage::Standard(_) => panic!("expected compactionSummary"),
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[test]
fn calculates_total_context_tokens_from_usage() {
    let usage = mock_usage(1000, 500, 200, 100);
    assert_eq!(calculate_context_tokens(&usage), 1800);
}

#[test]
fn handles_zero_values() {
    let usage = mock_usage(0, 0, 0, 0);
    assert_eq!(calculate_context_tokens(&usage), 0);
}

#[test]
fn finds_the_last_non_aborted_assistant_message_usage() {
    let mut chain = EntryChain::new();
    let entries = vec![
        chain.message_entry(user_message("Hello")),
        chain.message_entry(assistant_message("Hi", Some(mock_usage(100, 50, 0, 0)))),
        chain.message_entry(user_message("How are you?")),
        chain.message_entry(assistant_message("Good", Some(mock_usage(200, 100, 0, 0)))),
    ];

    let usage = get_last_assistant_usage(&entries).expect("usage present");
    assert_eq!(usage.input, 200);
}

#[test]
fn skips_aborted_messages() {
    let mut chain = EntryChain::new();
    let aborted = match assistant_message("Aborted", Some(mock_usage(300, 150, 0, 0))) {
        AgentMessage::Standard(pi_ai::types::Message::Assistant(mut message)) => {
            message.stop_reason = pi_ai::types::StopReason::Aborted;
            AgentMessage::Standard(pi_ai::types::Message::Assistant(message))
        }
        _ => unreachable!("assistant_message builds an assistant message"),
    };

    let entries = vec![
        chain.message_entry(user_message("Hello")),
        chain.message_entry(assistant_message("Hi", Some(mock_usage(100, 50, 0, 0)))),
        chain.message_entry(user_message("How are you?")),
        chain.message_entry(aborted),
    ];

    let usage = get_last_assistant_usage(&entries).expect("usage present");
    assert_eq!(usage.input, 100);
}

#[test]
fn skips_all_zero_assistant_usage() {
    let mut chain = EntryChain::new();
    let entries = vec![
        chain.message_entry(user_message("Hello")),
        chain.message_entry(assistant_message("Hi", Some(mock_usage(100, 50, 0, 0)))),
        chain.message_entry(user_message("continue")),
        chain.message_entry(assistant_message("Partial", Some(mock_usage(0, 0, 0, 0)))),
    ];

    let usage = get_last_assistant_usage(&entries).expect("usage present");
    assert_eq!(usage.input, 100);
}

#[test]
fn returns_none_if_no_assistant_messages() {
    let mut chain = EntryChain::new();
    let entries = vec![chain.message_entry(user_message("Hello"))];
    assert!(get_last_assistant_usage(&entries).is_none());
}

#[test]
fn uses_the_last_non_zero_assistant_usage_as_the_context_anchor() {
    let messages = vec![
        user_message("Hello"),
        assistant_message("Hi", Some(mock_usage(100, 50, 0, 0))),
        user_message("continue"),
        assistant_message("Partial thinking", Some(mock_usage(0, 0, 0, 0))),
    ];

    let estimate = estimate_context_tokens(&messages);

    assert_eq!(estimate.usage_tokens, 150);
    assert_eq!(estimate.last_usage_index, Some(1));
    assert!(estimate.trailing_tokens > 0);
    assert_eq!(estimate.tokens, 150 + estimate.trailing_tokens);
}

#[test]
fn returns_true_when_context_exceeds_threshold() {
    let settings = CompactionSettings {
        enabled: true,
        reserve_tokens: 10000,
        keep_recent_tokens: 20000,
    };

    assert!(should_compact(95000, 100_000, &settings));
    assert!(!should_compact(89000, 100_000, &settings));
}

#[test]
fn returns_false_when_disabled() {
    let settings = CompactionSettings {
        enabled: false,
        reserve_tokens: 10000,
        keep_recent_tokens: 20000,
    };

    assert!(!should_compact(95000, 100_000, &settings));
}

#[test]
fn finds_cut_point_based_on_actual_token_differences() {
    let mut chain = EntryChain::new();
    let mut entries: Vec<SessionEntry> = Vec::new();
    for i in 0..10 {
        entries.push(chain.message_entry(user_message(&format!("User {i}"))));
        entries.push(chain.message_entry(assistant_message(
            &format!("Assistant {i}"),
            Some(mock_usage(0, 100, (i + 1) * 1000, 0)),
        )));
    }

    // 20 entries, last assistant has 10000 tokens. keepRecentTokens = 2500:
    // keep entries where diff < 2500.
    let result = find_cut_point(&entries, 0, entries.len(), 2500);

    // Should cut at a valid cut point (user or assistant message).
    assert!(matches!(
        entries[result.first_kept_entry_index],
        SessionEntry::Message(_)
    ));
    assert!(matches!(
        message_role(&entries[result.first_kept_entry_index]),
        Some("user" | "assistant")
    ));
}

#[test]
fn returns_start_index_if_no_valid_cut_points_in_range() {
    let mut chain = EntryChain::new();
    let entries = vec![chain.message_entry(assistant_message("a", None))];
    let result = find_cut_point(&entries, 0, entries.len(), 1000);
    assert_eq!(result.first_kept_entry_index, 0);
}

#[test]
fn keeps_everything_if_all_messages_fit_within_budget() {
    let mut chain = EntryChain::new();
    let entries = vec![
        chain.message_entry(user_message("1")),
        chain.message_entry(assistant_message("a", Some(mock_usage(0, 50, 500, 0)))),
        chain.message_entry(user_message("2")),
        chain.message_entry(assistant_message("b", Some(mock_usage(0, 50, 1000, 0)))),
    ];

    let result = find_cut_point(&entries, 0, entries.len(), 50000);
    assert_eq!(result.first_kept_entry_index, 0);
}

#[test]
fn indicates_split_turn_when_cutting_at_assistant_message() {
    let mut chain = EntryChain::new();
    // Create a scenario where we cut at an assistant message mid-turn.
    let entries = vec![
        chain.message_entry(user_message("Turn 1")),
        chain.message_entry(assistant_message("A1", Some(mock_usage(0, 100, 1000, 0)))),
        chain.message_entry(user_message("Turn 2")), // index 2
        chain.message_entry(assistant_message("A2-1", Some(mock_usage(0, 100, 5000, 0)))), // index 3
        chain.message_entry(assistant_message("A2-2", Some(mock_usage(0, 100, 8000, 0)))), // index 4
        chain.message_entry(assistant_message(
            "A2-3",
            Some(mock_usage(0, 100, 10000, 0)),
        )), // index 5
    ];

    // With keepRecentTokens = 3000, should cut somewhere in Turn 2.
    let result = find_cut_point(&entries, 0, entries.len(), 3000);

    // If cut at assistant message (not user), should indicate split turn.
    if message_role(&entries[result.first_kept_entry_index]) == Some("assistant") {
        assert!(result.is_split_turn);
        assert_eq!(result.turn_start_index, 2); // Turn 2 starts at index 2
    }
}

#[test]
fn budgets_context_visible_custom_message_entries() {
    let mut chain = EntryChain::new();
    let entries = vec![
        chain.message_entry(user_message("hi")),
        chain.message_entry(assistant_message("hello", None)),
        chain.custom_message_entry(&"x".repeat(4000)),
        chain.message_entry(assistant_message("ok", None)),
    ];

    let tiny_budget = find_cut_point(&entries, 0, entries.len(), 1);
    assert_eq!(tiny_budget.first_kept_entry_index, 3);
    assert!(tiny_budget.is_split_turn);
    assert_eq!(tiny_budget.turn_start_index, 2);

    let custom_fits_budget = find_cut_point(&entries, 0, entries.len(), 2);
    assert_eq!(custom_fits_budget.first_kept_entry_index, 2);
    assert!(!custom_fits_budget.is_split_turn);
    assert_eq!(custom_fits_budget.turn_start_index, -1);
}

// ============================================================================
// buildSessionContext
// ============================================================================

#[test]
fn loads_all_messages_when_no_compaction() {
    let mut chain = EntryChain::new();
    let entries = vec![
        chain.message_entry(user_message("1")),
        chain.message_entry(assistant_message("a", None)),
        chain.message_entry(user_message("2")),
        chain.message_entry(assistant_message("b", None)),
    ];

    let loaded = build_typed_session_context(&entries);
    assert_eq!(loaded.messages.len(), 4);
    assert_eq!(loaded.thinking_level, "off");
    assert_eq!(
        loaded.model,
        Some(SessionModel {
            provider: "anthropic".to_owned(),
            model_id: "claude-sonnet-4-5".to_owned(),
        })
    );
}

#[test]
fn handles_single_compaction() {
    let mut chain = EntryChain::new();
    // IDs: u1=test-id-0, a1=test-id-1, u2=test-id-2, a2=test-id-3,
    // compaction=test-id-4, u3=test-id-5, a3=test-id-6.
    let u1 = chain.message_entry(user_message("1"));
    let a1 = chain.message_entry(assistant_message("a", None));
    let u2 = chain.message_entry(user_message("2"));
    let a2 = chain.message_entry(assistant_message("b", None));
    let u2_id = match &u2 {
        SessionEntry::Message(message) => message.base.id.clone().expect("id"),
        _ => unreachable!("message entry"),
    };
    let compaction = chain.compaction_entry("Summary of 1,a,2,b", &u2_id); // keep from u2 onwards
    let u3 = chain.message_entry(user_message("3"));
    let a3 = chain.message_entry(assistant_message("c", None));

    let entries = vec![u1, a1, u2, a2, compaction, u3, a3];

    let loaded = build_typed_session_context(&entries);
    // summary + kept (u2, a2) + after (u3, a3) = 5
    assert_eq!(loaded.messages.len(), 5);
    assert!(first_summary(&loaded.messages).contains("Summary of 1,a,2,b"));
}

#[test]
fn handles_multiple_compactions_only_latest_matters() {
    let mut chain = EntryChain::new();
    // First batch.
    let u1 = chain.message_entry(user_message("1"));
    let a1 = chain.message_entry(assistant_message("a", None));
    let u1_id = match &u1 {
        SessionEntry::Message(message) => message.base.id.clone().expect("id"),
        _ => unreachable!("message entry"),
    };
    let compact1 = chain.compaction_entry("First summary", &u1_id);
    // Second batch.
    let u2 = chain.message_entry(user_message("2"));
    let b = chain.message_entry(assistant_message("b", None));
    let u3 = chain.message_entry(user_message("3"));
    let u3_id = match &u3 {
        SessionEntry::Message(message) => message.base.id.clone().expect("id"),
        _ => unreachable!("message entry"),
    };
    let c = chain.message_entry(assistant_message("c", None));
    let compact2 = chain.compaction_entry("Second summary", &u3_id); // keep from u3 onwards
    // After second compaction.
    let u4 = chain.message_entry(user_message("4"));
    let d = chain.message_entry(assistant_message("d", None));

    let entries = vec![u1, a1, compact1, u2, b, u3, c, compact2, u4, d];

    let loaded = build_typed_session_context(&entries);
    // summary + kept from u3 (u3, c) + after (u4, d) = 5
    assert_eq!(loaded.messages.len(), 5);
    assert!(first_summary(&loaded.messages).contains("Second summary"));
}

#[test]
fn keeps_all_messages_when_first_kept_entry_id_is_first_entry() {
    let mut chain = EntryChain::new();
    let u1 = chain.message_entry(user_message("1"));
    let u1_id = match &u1 {
        SessionEntry::Message(message) => message.base.id.clone().expect("id"),
        _ => unreachable!("message entry"),
    };
    let a1 = chain.message_entry(assistant_message("a", None));
    let compact1 = chain.compaction_entry("First summary", &u1_id); // keep from first entry
    let u2 = chain.message_entry(user_message("2"));
    let b = chain.message_entry(assistant_message("b", None));

    let entries = vec![u1, a1, compact1, u2, b];

    let loaded = build_typed_session_context(&entries);
    // summary + all messages (u1, a1, u2, b) = 5
    assert_eq!(loaded.messages.len(), 5);
}

#[test]
fn tracks_model_and_thinking_level_changes() {
    let mut chain = EntryChain::new();
    let entries = vec![
        chain.message_entry(user_message("1")),
        chain.model_change_entry("openai", "gpt-4"),
        chain.message_entry(assistant_message("a", None)),
        chain.thinking_level_entry("high"),
    ];

    let loaded = build_typed_session_context(&entries);
    // model_change is later overwritten by assistant message's model info
    assert_eq!(
        loaded.model,
        Some(SessionModel {
            provider: "anthropic".to_owned(),
            model_id: "claude-sonnet-4-5".to_owned(),
        })
    );
    assert_eq!(loaded.thinking_level, "high");
}

// ============================================================================
// prepareCompaction with previous compaction
// ============================================================================

#[test]
fn skips_repeated_compactions_when_kept_messages_still_fit() {
    let mut chain = EntryChain::new();
    let u1 = chain.message_entry(user_message("user msg 1 (summarized by compaction1)"));
    let a1 = chain.message_entry(assistant_message("assistant msg 1", None));
    let u2 = chain.message_entry(user_message("user msg 2 - kept by compaction1"));
    let u2_id = match &u2 {
        SessionEntry::Message(message) => message.base.id.clone().expect("id"),
        _ => unreachable!("message entry"),
    };
    let a2 = chain.message_entry(assistant_message("assistant msg 2", None));
    let u3 = chain.message_entry(user_message("user msg 3 - kept by compaction1"));
    let a3 = chain.message_entry(assistant_message(
        "assistant msg 3",
        Some(mock_usage(5000, 1000, 0, 0)),
    ));
    let compaction1 = chain.compaction_entry("First summary", &u2_id);
    let u4 = chain.message_entry(user_message("user msg 4 (new after compaction1)"));
    let a4 = chain.message_entry(assistant_message(
        "assistant msg 4",
        Some(mock_usage(8000, 2000, 0, 0)),
    ));

    let path_entries = vec![u1, a1, u2, a2, u3, a3, compaction1, u4, a4];
    let preparation = prepare_compaction(&path_entries, DEFAULT_COMPACTION_SETTINGS);

    assert!(preparation.is_none());
}

#[test]
fn re_summarize_previously_kept_messages_when_the_recent_window_moves_past_them() {
    let mut chain = EntryChain::new();
    let u1 = chain.message_entry(user_message(
        &"user msg 1 (summarized by compaction1)".repeat(4),
    ));
    let a1 = chain.message_entry(assistant_message(&"assistant msg 1".repeat(4), None));
    let u2 = chain.message_entry(user_message(
        &"user msg 2 - kept by compaction1 ".repeat(12),
    ));
    let u2_id = match &u2 {
        SessionEntry::Message(message) => message.base.id.clone().expect("id"),
        _ => unreachable!("message entry"),
    };
    let a2 = chain.message_entry(assistant_message(&"assistant msg 2 ".repeat(12), None));
    let u3 = chain.message_entry(user_message(
        &"user msg 3 - kept by compaction1 ".repeat(12),
    ));
    let a3 = chain.message_entry(assistant_message(
        &"assistant msg 3 ".repeat(12),
        Some(mock_usage(5000, 1000, 0, 0)),
    ));
    let compaction1 = chain.compaction_entry("First summary", &u2_id);
    let u4 = chain.message_entry(user_message(
        &"user msg 4 (new after compaction1) ".repeat(12),
    ));
    let a4 = chain.message_entry(assistant_message(
        &"assistant msg 4 ".repeat(12),
        Some(mock_usage(8000, 2000, 0, 0)),
    ));

    let path_entries = vec![u1, a1, u2, a2, u3, a3, compaction1, u4, a4];

    let settings = CompactionSettings {
        keep_recent_tokens: 100,
        ..DEFAULT_COMPACTION_SETTINGS
    };
    let preparation = prepare_compaction(&path_entries, settings).expect("preparation present");

    let summarized_text = extract_text(&preparation.messages_to_summarize);
    assert!(summarized_text.contains("user msg 2 - kept by compaction1"));
    assert!(summarized_text.contains("user msg 3 - kept by compaction1"));
    assert!(!summarized_text.contains("First summary"));
    assert_eq!(
        preparation.previous_summary.as_deref(),
        Some("First summary")
    );
}

// ============================================================================
// Integration tests with real session data
// ============================================================================

#[test]
fn parses_the_large_session() {
    let entries = load_large_session_entries();
    assert!(entries.len() > 100);

    let message_count = entries
        .iter()
        .filter(|entry| matches!(entry, SessionEntry::Message(_)))
        .count();
    assert!(message_count > 100);
}

#[test]
fn finds_cut_point_in_large_session() {
    let entries = load_large_session_entries();
    let result = find_cut_point(
        &entries,
        0,
        entries.len(),
        DEFAULT_COMPACTION_SETTINGS.keep_recent_tokens as u64,
    );

    // Cut point should be at a message entry (user or assistant).
    assert!(matches!(
        entries[result.first_kept_entry_index],
        SessionEntry::Message(_)
    ));
    assert!(matches!(
        message_role(&entries[result.first_kept_entry_index]),
        Some("user" | "assistant")
    ));
}

#[test]
fn loads_session_correctly() {
    let entries = load_large_session_entries();
    let loaded = build_typed_session_context(&entries);

    assert!(loaded.messages.len() > 100);
    assert!(loaded.model.is_some());
}

// ============================================================================
// LLM integration tests (skipped without API key)
// ============================================================================

fn anthropic_model() -> pi_ai::types::Model {
    pi_ai::providers::catalog::get_builtin_model("anthropic", "claude-sonnet-4-5")
        .expect("builtin model")
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC_OAUTH_TOKEN is set, upstream's describe.skipIf"]
async fn generates_a_compaction_result_for_the_large_session() {
    let Some(token) = std::env::var("ANTHROPIC_OAUTH_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
    else {
        eprintln!("skipping: ANTHROPIC_OAUTH_TOKEN not set");
        return;
    };
    let entries = load_large_session_entries();
    let model = anthropic_model();

    let preparation =
        prepare_compaction(&entries, DEFAULT_COMPACTION_SETTINGS).expect("preparation present");

    let compaction_result = pi_coding_agent::compaction::compact(
        preparation,
        &model,
        Some(&token),
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
    .expect("compaction succeeds");

    assert!(compaction_result.summary.chars().count() > 100);
    assert!(!compaction_result.first_kept_entry_id.is_empty());
    assert!(compaction_result.tokens_before > 0);

    println!(
        "Summary length: {}",
        compaction_result.summary.chars().count()
    );
    println!(
        "First kept entry ID: {}",
        compaction_result.first_kept_entry_id
    );
    println!("Tokens before: {}", compaction_result.tokens_before);
    println!("\n--- SUMMARY ---\n");
    println!("{}", compaction_result.summary);
}

#[tokio::test]
#[ignore = "env-gated: runs when ANTHROPIC_OAUTH_TOKEN is set, upstream's describe.skipIf"]
async fn produces_valid_session_after_compaction() {
    let Some(token) = std::env::var("ANTHROPIC_OAUTH_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
    else {
        eprintln!("skipping: ANTHROPIC_OAUTH_TOKEN not set");
        return;
    };
    let entries = load_large_session_entries();
    let loaded = build_typed_session_context(&entries);
    let model = anthropic_model();

    let preparation =
        prepare_compaction(&entries, DEFAULT_COMPACTION_SETTINGS).expect("preparation present");

    let compaction_result = pi_coding_agent::compaction::compact(
        preparation,
        &model,
        Some(&token),
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
    .expect("compaction succeeds");

    // Simulate appending compaction to entries by creating a proper entry.
    let parent_id = entries
        .last()
        .and_then(|entry| entry.base().id.clone())
        .expect("last entry id");
    let compaction_entry = SessionEntry::Compaction(CompactionEntry {
        base: SessionEntryBase {
            id: Some("compaction-test-id".to_owned()),
            parent_id: Some(parent_id),
            timestamp: String::new(),
            extras: serde_json::Map::new(),
        },
        summary: compaction_result.summary.clone(),
        first_kept_entry_id: Some(compaction_result.first_kept_entry_id.clone()),
        tokens_before: compaction_result.tokens_before.cast_signed(),
        details: compaction_result.details.clone(),
        usage: Some(compaction_result.usage),
        from_hook: None,
        extras: serde_json::Map::new(),
    });
    let mut new_entries = entries.clone();
    new_entries.push(compaction_entry);
    let reloaded = build_typed_session_context(&new_entries);

    // Should have summary + kept messages.
    assert!(reloaded.messages.len() < loaded.messages.len());
    assert!(first_summary(&reloaded.messages).contains(&compaction_result.summary));

    println!("Original messages: {}", loaded.messages.len());
    println!("After compaction: {}", reloaded.messages.len());
}

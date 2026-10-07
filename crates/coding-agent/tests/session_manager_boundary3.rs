//! The session-manager boundary suite, third pass at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the simple getters, the
//! label table's load-time clear arm, the tree's child sorting (including
//! upstream's unparseable-timestamp tie), and the post-flush append rule.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::fs;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{
    AssistantBlock, AssistantMessage, KnownApi, Message, ProviderId, StopReason, Usage, UsageCost,
};

use pi_coding_agent::session_manager::SessionManager;

const fn usage() -> Usage {
    Usage {
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 0,
        cost: UsageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    }
}

fn assistant_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::Assistant(AssistantMessage {
        content: vec![AssistantBlock::Text(pi_ai::types::TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: KnownApi::AnthropicMessages.into(),
        provider: ProviderId("anthropic".to_owned()),
        model: "test".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: usage(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 42,
    }))
}

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(pi_ai::types::UserMessage {
        content: pi_ai::types::UserContent::Text(text.to_owned()),
        timestamp: 42,
    }))
}

#[test]
fn the_empty_in_memory_session_reports_its_defaults() {
    let session = SessionManager::in_memory(None, None, None).expect("in-memory");
    assert!(
        !session.is_persisted(),
        "an in-memory session never persists"
    );
    assert_eq!(
        session.session_dir(),
        "",
        "an in-memory session has no directory"
    );
    assert_eq!(session.session_file(), None);
    assert!(
        session.get_header().is_some(),
        "the constructor still opens a session header"
    );
    assert!(
        session.build_context_entries().is_empty(),
        "the empty leaf resolves to no entries"
    );
    assert_eq!(
        session.get_session_name(),
        None,
        "no session_info means no name"
    );
    assert!(session.get_tree().is_empty());
}

#[test]
fn the_persisted_getters_and_the_session_name_round_trip() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    assert!(session.is_persisted());
    assert_eq!(session.session_dir(), temp);
    assert!(session.session_file().is_some());

    session
        .append_session_info("  the name  ")
        .expect("session info");
    assert_eq!(
        session.get_session_name().as_deref(),
        Some("the name"),
        "the name trims"
    );
}

#[test]
fn entries_appended_after_the_flush_land_in_the_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    session
        .append_message(assistant_message("flush"))
        .expect("assistant flushes the buffer");
    let file = session.session_file().expect("file").to_owned();

    session
        .append_thinking_level_change("high")
        .expect("thinking");
    let lines = fs::read_to_string(&file).expect("read").lines().count();
    assert_eq!(lines, 3, "header + assistant + the appended thinking entry");

    session.append_message(user_message("again")).expect("user");
    let lines = fs::read_to_string(&file).expect("read").lines().count();
    assert_eq!(
        lines, 4,
        "the append path writes line by line after the flush"
    );
}

#[test]
fn a_label_entry_without_a_label_clears_on_load() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = format!("{}/labeled.jsonl", dir.path().display());
    fs::write(
        &path,
        concat!(
            r#"{"type":"session","version":3,"id":"lbl","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#, "\n",
            r#"{"type":"custom","id":"e1","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z","customType":"x"}"#, "\n",
            r#"{"type":"label","id":"l1","parentId":"e1","timestamp":"2026-01-01T00:00:01.000Z","targetId":"e1"}"#, "\n",
        ),
    )
    .expect("write labeled file");

    let mut session = SessionManager::open(&path, None, None).expect("open");
    assert_eq!(
        session.get_label("e1"),
        None,
        "an absent label clears the target"
    );
    assert!(
        session.get_tree()[0].label.is_none(),
        "the tree node reports no label"
    );

    // A loaded file is flushed before its first assistant message, so the
    // next entry rides the append path, not the buffer.
    session
        .append_session_info("the name")
        .expect("session info");
    let lines = fs::read_to_string(&path).expect("read").lines().count();
    assert_eq!(
        lines, 4,
        "the appended entry lands in the already-flushed file"
    );
}

#[test]
fn tree_children_sort_by_timestamp_and_unparseable_ones_tie() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = format!("{}/branched.jsonl", dir.path().display());
    fs::write(
        &path,
        concat!(
            r#"{"type":"session","version":3,"id":"tree","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#, "\n",
            r#"{"type":"custom","id":"r","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z","customType":"root"}"#, "\n",
            r#"{"type":"custom","id":"b","parentId":"r","timestamp":"2026-01-02T00:00:00.000Z","customType":"late"}"#, "\n",
            r#"{"type":"custom","id":"a","parentId":"r","timestamp":"2026-01-01T00:00:00.000Z","customType":"early"}"#, "\n",
            r#"{"type":"custom","id":"g","parentId":"r","timestamp":"not-a-date","customType":"garbage"}"#, "\n",
        ),
    )
    .expect("write tree file");

    let session = SessionManager::open(&path, None, None).expect("open");
    let tree = session.get_tree();
    assert_eq!(tree.len(), 1, "one root");
    let ids: Vec<&str> = tree[0]
        .children
        .iter()
        .filter_map(|node| node.entry.entry_id())
        .collect();
    // Upstream's comparator turns an unparseable timestamp into NaN, which
    // sorts as equal: the parseable children sort, and the garbage child
    // keeps its file position relative to them.
    assert_eq!(
        ids,
        vec!["a", "b", "g"],
        "timestamps sort first; the garbage timestamp ties"
    );
}

#[test]
fn branching_to_an_unknown_summary_entry_errors() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    session
        .append_message(user_message("hello"))
        .expect("append");

    let error = session
        .branch_with_summary(Some("zzz"), "summary", None, None, None)
        .expect_err("unknown id");
    assert_eq!(
        error.to_string(),
        "Entry zzz not found",
        "the branch summary reports the unknown entry like branch does"
    );
}

#[test]
fn a_self_parented_or_parentless_entry_roots_the_tree() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = format!("{}/roots.jsonl", dir.path().display());
    fs::write(
        &path,
        concat!(
            r#"{"type":"session","version":3,"id":"roots","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#, "\n",
            r#"{"type":"custom","id":"s","parentId":"s","timestamp":"2026-01-01T00:00:00.000Z","customType":"self"}"#, "\n",
            r#"{"type":"custom","id":"p","timestamp":"2026-01-01T00:00:01.000Z","customType":"parentless"}"#, "\n",
        ),
    )
    .expect("write roots file");

    let session = SessionManager::open(&path, None, None).expect("open");
    let tree = session.get_tree();
    assert_eq!(
        tree.len(),
        2,
        "a self-parent and a parentless entry both root"
    );
    let ids: Vec<Option<&str>> = tree.iter().map(|node| node.entry.entry_id()).collect();
    assert_eq!(
        ids,
        vec![Some("s"), Some("p")],
        "file order keeps the roots"
    );
    assert!(tree[0].children.is_empty() && tree[1].children.is_empty());
}

#[test]
fn branch_with_summary_from_a_reset_leaf_parents_at_root() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    session
        .append_message(assistant_message("flush"))
        .expect("append");
    session.reset_leaf();

    let summary_id = session
        .branch_with_summary(None, "the summary", None, None, None)
        .expect("branch");
    let summary = session.get_entry(&summary_id).expect("the summary entry");
    assert_eq!(
        summary.entry_parent_id(),
        None,
        "the summary parents at the root"
    );
    assert_eq!(session.get_leaf_id(), Some(summary_id.as_str()));
    let _ = first;
    let file = session.session_file().expect("file");
    let lines = fs::read_to_string(file).expect("read").lines().count();
    assert_eq!(lines, 4, "header + two entries + the appended summary");
}

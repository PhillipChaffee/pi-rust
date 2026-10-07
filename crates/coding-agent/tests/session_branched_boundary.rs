//! The branched-session boundary suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the label re-chaining and
//! compaction kept-id remap through removed labels, raw-value path entries,
//! the flush gates, and the unknown-leaf error.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use std::fs;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{Message, UserContent, UserMessage};

use pi_coding_agent::session_manager::{
    FileEntry, SessionEntry, SessionManager, SessionManagerError,
};

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp: 42,
    }))
}

fn assistant_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::Assistant(pi_ai::types::AssistantMessage {
        content: vec![pi_ai::types::AssistantBlock::Text(
            pi_ai::types::TextContent {
                text: text.to_owned(),
                text_signature: None,
            },
        )],
        api: pi_ai::types::KnownApi::AnthropicMessages.into(),
        provider: pi_ai::types::ProviderId("anthropic".to_owned()),
        model: "test".to_owned(),
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
        stop_reason: pi_ai::types::StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 42,
    }))
}

fn wire_types(session: &SessionManager) -> Vec<String> {
    session
        .entries()
        .iter()
        .filter_map(|entry| match entry {
            FileEntry::Entry(SessionEntry::Message(_)) => Some("message".to_owned()),
            FileEntry::Entry(SessionEntry::Label(_)) => Some("label".to_owned()),
            FileEntry::Entry(SessionEntry::Compaction(_)) => Some("compaction".to_owned()),
            FileEntry::Other(value) => value["type"].as_str().map(str::to_owned),
            _ => None,
        })
        .collect()
}

#[test]
fn an_unknown_leaf_errors_before_any_file_is_touched() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    let error = session
        .create_branched_session("zzz")
        .expect_err("unknown leaf");
    assert_eq!(error, SessionManagerError::EntryNotFound("zzz".to_owned()));
}

#[test]
fn a_label_inside_the_path_re_chains_its_children_onto_the_next_entry() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    let first = session
        .append_message(user_message("root"))
        .expect("append");
    let label_id = session
        .append_label_change(&first, Some("bookmark"))
        .expect("label");
    let child = session
        .append_message(user_message("under label"))
        .expect("append");
    session.branch(&child).expect("branch");

    let branched = session
        .create_branched_session(&child)
        .expect("branch")
        .expect("persisted");
    assert!(
        !fs::exists(&branched).unwrap_or(false),
        "no assistant in the path defers the write"
    );
    assert_eq!(session.session_file(), Some(branched.as_str()));
    assert_eq!(
        wire_types(&session),
        vec!["message", "message", "label"],
        "the label rebuilds at the tail"
    );
    let ids: Vec<Option<&str>> = session
        .entries()
        .iter()
        .map(|entry| entry.entry_id())
        .collect();
    assert_eq!(
        ids[..2],
        [Some(first.as_str()), Some(child.as_str())],
        "the two retained entries keep their ids"
    );
    let child_entry = &session.entries()[1];
    assert_eq!(
        child_entry.entry_parent_id(),
        Some(first.as_str()),
        "the label's child re-chains onto the kept entry"
    );
    let rebuilt = session.entries().iter().find_map(|entry| match entry {
        FileEntry::Entry(SessionEntry::Label(label)) => Some(label),
        _ => None,
    });
    let rebuilt = rebuilt.expect("the label rebuilds");
    assert_ne!(
        rebuilt.base.id.as_deref(),
        Some(label_id.as_str()),
        "the rebuilt label carries a fresh id"
    );
    assert_eq!(rebuilt.target_id, first, "the label keeps its target");
    assert_eq!(
        rebuilt.base.parent_id.as_deref(),
        Some(child.as_str()),
        "the rebuilt label chains onto the path tail"
    );
    assert_eq!(
        session.get_label(&first).map(str::to_owned).as_deref(),
        Some("bookmark")
    );
}

#[test]
fn a_compaction_whose_kept_id_named_a_removed_label_remaps_to_the_replacement() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    let first = session
        .append_message(user_message("root"))
        .expect("append");
    let label_id = session
        .append_label_change(&first, Some("bookmark"))
        .expect("label");
    let second = session
        .append_message(user_message("kept"))
        .expect("append");
    let compaction_id = session
        .append_compaction("summary", &label_id, 10, None, None, None)
        .expect("compaction");
    session.branch(&compaction_id).expect("branch");

    session
        .create_branched_session(&compaction_id)
        .expect("branch");
    let compaction = session.entries().iter().find_map(|entry| match entry {
        FileEntry::Entry(SessionEntry::Compaction(compaction)) => Some(compaction),
        _ => None,
    });
    let compaction = compaction.expect("the compaction rides the path");
    assert_eq!(
        compaction.first_kept_entry_id.as_deref(),
        Some(second.as_str()),
        "the kept id pointed at the removed label, so it remaps to the label's replacement"
    );
    let _ = label_id;
}

#[test]
fn raw_path_entries_re_chain_and_raw_compactions_carry_their_kept_ids() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = format!("{}/raw-path.jsonl", dir.path().display());
    fs::write(
        &path,
        concat!(
            r#"{"type":"session","version":3,"id":"raw","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#, "\n",
            r#"{"type":"custom","id":"e1","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z","customType":"root"}"#, "\n",
            r#"{"type":"mystery","id":"m1","parentId":"e1","timestamp":"2026-01-01T00:00:01.000Z","value":1}"#, "\n",
            r#"{"type":"compaction","id":"c1","parentId":"m1","timestamp":"2026-01-01T00:00:02.000Z","summary":"s","tokensBefore":1,"firstKeptEntryId":"m1"}"#, "\n",
        ),
    )
    .expect("write raw-path file");
    let mut session = SessionManager::open(&path, None, None).expect("open");

    session.create_branched_session("c1").expect("branch");
    let entries = session.entries();
    assert_eq!(
        entries.len(),
        3,
        "the three retained entries ride the branch"
    );
    let mystery = &entries[1];
    assert_eq!(mystery.entry_id(), Some("m1"));
    assert_eq!(
        mystery.entry_parent_id(),
        Some("e1"),
        "the raw value re-chains onto the kept root"
    );
    let raw_compaction = &entries[2];
    assert_eq!(
        raw_compaction.entry_parent_id(),
        Some("m1"),
        "the raw compaction chains onto the raw entry"
    );
    match raw_compaction {
        FileEntry::Entry(SessionEntry::Compaction(compaction)) => {
            assert_eq!(
                compaction.first_kept_entry_id.as_deref(),
                Some("m1"),
                "the kept id stays: its target was kept"
            );
        }
        FileEntry::Other(value) => {
            assert_eq!(
                value["firstKeptEntryId"], "m1",
                "the kept id stays: its target was kept"
            );
        }
        _ => panic!("the compaction rides the branch"),
    }
}

#[test]
fn a_persisted_branch_with_an_assistant_in_the_path_writes_immediately() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    session
        .append_message(user_message("hello"))
        .expect("append");
    session
        .append_message(assistant_message("reply"))
        .expect("append");
    let second = session
        .append_message(user_message("kept"))
        .expect("append");
    session.branch(&second).expect("branch");

    let branched = session
        .create_branched_session(&second)
        .expect("branch")
        .expect("persisted");
    assert!(
        fs::exists(&branched).unwrap_or(false),
        "the path carries an assistant message, so the file flushes now"
    );
    let lines: Vec<String> = fs::read_to_string(&branched)
        .expect("read")
        .lines()
        .map(str::to_owned)
        .collect();
    assert_eq!(lines.len(), 4, "header + the three path entries");
    assert!(
        lines[2].contains(r#""role":"assistant""#),
        "the assistant rides the file"
    );
}

#[test]
fn a_persisted_branch_without_an_assistant_defers_the_write() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    session
        .append_message(assistant_message("flush"))
        .expect("append");
    session.branch(&first).expect("branch");

    let branched = session
        .create_branched_session(&first)
        .expect("branch")
        .expect("persisted");
    assert!(
        !fs::exists(&branched).unwrap_or(false),
        "the path carries no assistant message, so the write defers"
    );
    // The first assistant append flushes the whole buffered branch.
    session
        .append_message(assistant_message("again"))
        .expect("append");
    let lines: Vec<String> = fs::read_to_string(&branched)
        .expect("read")
        .lines()
        .map(str::to_owned)
        .collect();
    assert_eq!(
        lines.len(),
        3,
        "header + the buffered entry + the assistant"
    );
    assert!(
        lines[1].contains(&first),
        "the branch entry writes with the flush"
    );
}

#[test]
fn an_in_memory_branch_replaces_the_session_without_a_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::in_memory(Some(&temp), None, None).expect("in-memory");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    session
        .append_label_change(&first, Some("bookmark"))
        .expect("label");
    let second = session
        .append_message(user_message("kept"))
        .expect("append");
    session.branch(&second).expect("branch");

    let branched = session.create_branched_session(&second).expect("branch");
    assert_eq!(branched, None, "in-memory branches persist nothing");
    assert_eq!(session.session_file(), None, "still no file");
    let rebuilt = session.entries().iter().find_map(|entry| match entry {
        FileEntry::Entry(SessionEntry::Label(label)) => Some(label),
        _ => None,
    });
    let rebuilt = rebuilt.expect("the label rebuilds");
    assert_eq!(rebuilt.target_id, first, "the label keeps its target");
    assert_eq!(
        rebuilt.base.parent_id.as_deref(),
        Some(second.as_str()),
        "the rebuilt label chains onto the path tail"
    );
    assert_eq!(
        session.get_label(&first).map(str::to_owned).as_deref(),
        Some("bookmark")
    );
}

#[test]
fn a_raw_compaction_that_cannot_parse_still_remaps_through_the_labels() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = format!("{}/raw-compaction.jsonl", dir.path().display());
    fs::write(
        &path,
        concat!(
            r#"{"type":"session","version":3,"id":"rawc","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#, "\n",
            r#"{"type":"custom","id":"e1","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z","customType":"root"}"#, "\n",
            r#"{"type":"label","id":"l1","parentId":"e1","timestamp":"2026-01-01T00:00:01.000Z","targetId":"e1","label":"x"}"#, "\n",
            r#"{"type":"compaction","id":"c1","parentId":"l1","timestamp":"2026-01-01T00:00:02.000Z","summary":"s","firstKeptEntryId":"l1"}"#, "\n",
        ),
    )
    .expect("write raw-compaction file");
    let mut session = SessionManager::open(&path, None, None).expect("open");

    session.create_branched_session("c1").expect("branch");
    let entries = session.entries();
    assert_eq!(entries.len(), 3, "two retained entries + the rebuilt label");
    let raw_compaction = &entries[1];
    if let FileEntry::Other(value) = raw_compaction {
        assert_eq!(
            value["type"], "compaction",
            "the tokensBefore-less compaction rides raw"
        );
        assert_eq!(
            value["firstKeptEntryId"], "c1",
            "the kept id pointed at the removed label, so it remaps to the label's replacement"
        );
        assert_eq!(
            value["parentId"], "e1",
            "the raw compaction re-chains onto the kept root"
        );
    } else {
        panic!("the tokensBefore-less compaction rides as a raw value");
    }
}

#[test]
fn a_raw_compaction_without_a_kept_id_still_re_chains() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = format!("{}/raw-keptless.jsonl", dir.path().display());
    fs::write(
        &path,
        concat!(
            r#"{"type":"session","version":3,"id":"rk","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#, "\n",
            r#"{"type":"label","id":"l1","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z","targetId":"l1","label":"x"}"#, "\n",
            r#"{"type":"compaction","id":"c1","parentId":"l1","timestamp":"2026-01-01T00:00:01.000Z","summary":"s"}"#, "\n",
        ),
    )
    .expect("write raw-keptless file");
    let mut session = SessionManager::open(&path, None, None).expect("open");

    session.create_branched_session("c1").expect("branch");
    let entries = session.entries();
    assert_eq!(
        entries.len(),
        1,
        "the self-targeting label drops with no rebuild: its target left the path"
    );
    if let FileEntry::Other(value) = &entries[0] {
        assert_eq!(value["type"], "compaction");
        assert_eq!(
            value.get("firstKeptEntryId"),
            None,
            "the compaction never carried a kept id and gains none"
        );
        assert_eq!(
            value["parentId"],
            serde_json::Value::Null,
            "the raw compaction re-chains onto the root"
        );
    } else {
        panic!("the tokensBefore-less compaction rides as a raw value");
    }
}

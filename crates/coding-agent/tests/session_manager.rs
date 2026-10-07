//! The session-manager core behaviors at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: append/branch flows, the
//! flush rules, label resolution, tree shape, the context projection, and
//! the invalid-session-file contract the `--session` suite drives.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
use std::fs;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{Message, UserContent, UserMessage};

use pi_coding_agent::session_manager::{
    CURRENT_SESSION_VERSION, FileEntry, NewSessionOptions, SessionEntry, SessionManager,
    SessionManagerError, SessionModel,
};

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp: 0,
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
            input: 1,
            output: 1,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 2,
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
        timestamp: 0,
    }))
}

#[test]
fn appends_chain_from_the_leaf_and_advance_it() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    let second = session
        .append_message(user_message("again"))
        .expect("append");

    assert_eq!(session.get_leaf_id(), Some(second.as_str()));
    assert_eq!(session.get_branch(None).len(), 2);
    let parent = session
        .get_entry(&second)
        .and_then(FileEntry::entry_parent_id)
        .expect("parent");
    assert_eq!(parent, first);
    assert_eq!(session.entries().len(), 2);
}

#[test]
fn branch_moves_the_leaf_without_touching_history() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    session
        .append_message(user_message("abandoned"))
        .expect("append");

    session.branch(&first).expect("branch");
    let branched = session
        .append_message(user_message("new path"))
        .expect("append");

    assert_eq!(session.entries().len(), 3);
    let parent = session
        .get_entry(&branched)
        .and_then(FileEntry::entry_parent_id)
        .expect("parent");
    assert_eq!(parent, first);
    assert_eq!(session.get_branch(None).len(), 2);
    assert_eq!(
        session.get_children(&first).len(),
        2,
        "both branches are children"
    );
}

#[test]
fn branch_with_unknown_entry_fails_with_upstreams_message() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let error = session.branch("missing").expect_err("branch fails");
    assert_eq!(
        error,
        SessionManagerError::EntryNotFound("missing".to_owned())
    );
    assert_eq!(error.to_string(), "Entry missing not found");
}

#[test]
fn branch_with_summary_records_the_abandoned_path_source() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");

    let summary_id = session
        .branch_with_summary(None, "the summary", None, None, None)
        .expect("branch with summary");

    let entry = session.get_entry(&summary_id).cloned().expect("entry");
    let FileEntry::Entry(SessionEntry::BranchSummary(summary)) = &entry else {
        panic!("branch summary entry, got {entry:?}");
    };
    assert_eq!(
        summary.from_id, first,
        "the abandoned path's leaf sources the summary"
    );
    assert_eq!(summary.summary, "the summary");
    assert_eq!(summary.base.parent_id, None);
    assert_eq!(session.get_leaf_id(), Some(summary_id.as_str()));
}

#[test]
fn branch_from_an_entry_carries_its_id_as_the_summary_source() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    session
        .branch_with_summary(Some(&first), "back", None, None, None)
        .expect("branch with summary");
    let entry = session.get_leaf_entry().expect("leaf").clone();
    let FileEntry::Entry(SessionEntry::BranchSummary(summary)) = &entry else {
        panic!("branch summary entry");
    };
    assert_eq!(summary.from_id, first);
}

#[test]
fn labels_set_and_clear_through_the_resolved_map() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");

    let error = session
        .append_label_change("missing", Some("x"))
        .expect_err("unknown target");
    assert_eq!(error.to_string(), "Entry missing not found");

    let label_id = session
        .append_label_change(&first, Some("checkpoint"))
        .expect("label");
    assert_eq!(session.get_label(&first), Some("checkpoint"));

    // A later label on the same target replaces in place, upstream's Map.set.
    session
        .append_label_change(&first, Some("renamed"))
        .expect("label");
    assert_eq!(session.get_label(&first), Some("renamed"));

    session.append_label_change(&first, None).expect("clear");
    assert_eq!(session.get_label(&first), None);
    assert!(
        session.get_entry(&label_id).is_some(),
        "the label entries stay in the file"
    );
    assert_eq!(session.entries().len(), 4, "message + three label entries");
}

#[test]
fn the_tree_carries_resolved_labels_and_orphans_root() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    let second = session
        .append_message(user_message("again"))
        .expect("append");
    session
        .append_label_change(&second, Some("current"))
        .expect("label");

    let tree = session.get_tree();
    assert_eq!(
        tree.len(),
        1,
        "one root; the label entry chains from the leaf"
    );
    let root = &tree[0];
    let FileEntry::Entry(session_entry_message) = &root.entry else {
        panic!("root is the first message");
    };
    match session_entry_message {
        SessionEntry::Message(message) => {
            assert_eq!(message.base.id.as_deref(), Some(first.as_str()));
        }
        other => panic!("message entry, got {other:?}"),
    }
    // The label appends as a child of the second message (the leaf when it
    // was written) and resolves onto that target's node.
    assert_eq!(
        root.children.len(),
        1,
        "the second message is the first's only child"
    );
    let second_node = &root.children[0];
    assert_eq!(
        second_node.label.as_deref(),
        Some("current"),
        "labels resolve onto their targets"
    );
    assert!(
        !second_node
            .label_timestamp
            .as_ref()
            .expect("timestamp")
            .is_empty()
    );
    assert_eq!(
        second_node.children.len(),
        1,
        "the label entry chains from its target"
    );
    assert_eq!(
        second_node.children[0].label, None,
        "the label entry itself is unlabeled"
    );
}

#[test]
fn orphaned_entries_root_in_the_tree() {
    use pi_coding_agent::session_manager::SessionEntry;
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("session.jsonl");
    // Hand-fabricate an orphan whose parent does not exist.
    fs::write(
        &path,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"s1\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"{}\"}}\n{{\"type\":\"message\",\"id\":\"m1\",\"parentId\":null,\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"message\":{{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":0}}}}\n{{\"type\":\"message\",\"id\":\"orphan\",\"parentId\":\"ghost\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"message\":{{\"role\":\"user\",\"content\":\"o\",\"timestamp\":0}}}}\n",
            dir.path().display()
        ),
    )
    .expect("write");
    let session = SessionManager::open(&path.display().to_string(), None, None).expect("open");

    let tree = session.get_tree();
    assert_eq!(tree.len(), 2, "the orphan roots beside the main chain");
    assert!(matches!(
        &tree[1].entry,
        FileEntry::Entry(SessionEntry::Message(_))
    ));
}

#[test]
fn session_info_names_sanitize_and_clear() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    session
        .append_session_info("hello\nworld\r\nagain")
        .expect("name");
    // Regression #5996: newlines do not reach the stored name.
    assert_eq!(
        session.get_session_name().as_deref(),
        Some("hello world again")
    );

    session.append_session_info("  spaced  ").expect("name");
    assert_eq!(session.get_session_name().as_deref(), Some("spaced"));

    session.append_session_info("").expect("clear");
    assert_eq!(
        session.get_session_name(),
        None,
        "empty names explicitly clear the title"
    );
}

#[test]
fn the_context_projection_resolves_settings_and_messages() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    session
        .append_message(user_message("hello"))
        .expect("append");
    session
        .append_model_change("anthropic", "claude")
        .expect("model");
    session
        .append_thinking_level_change("high")
        .expect("thinking");
    session
        .append_message(assistant_message("hi there"))
        .expect("append");

    let context = session.build_session_context();
    assert_eq!(context.thinking_level, "high");
    // The trailing assistant message overwrites the model_change value,
    // upstream's settings-walk order.
    assert_eq!(
        context.model,
        Some(SessionModel {
            provider: "anthropic".to_owned(),
            model_id: "test".to_owned(),
        })
    );
    assert_eq!(
        context.messages.len(),
        2,
        "settings entries carry no messages"
    );
    assert!(matches!(
        context.messages[0],
        AgentMessage::Standard(Message::User(_))
    ));
    assert!(matches!(
        context.messages[1],
        AgentMessage::Standard(Message::Assistant(_))
    ));
}

#[test]
fn compaction_projects_the_kept_tail() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    session.append_message(user_message("old")).expect("append");
    let kept = session
        .append_message(user_message("kept"))
        .expect("append");
    let _last = session
        .append_message(user_message("after"))
        .expect("append");
    let first_kept = session
        .get_entry(&kept)
        .and_then(FileEntry::entry_id)
        .expect("kept id")
        .to_owned();

    let compaction_id = session
        .append_compaction("the summary", &first_kept, 100, None, None, None)
        .expect("compaction");

    let entries = session.build_context_entries();
    assert_eq!(
        entries.len(),
        3,
        "compaction entry, kept entry, post-compaction entry"
    );
    assert!(matches!(
        entries[0],
        FileEntry::Entry(SessionEntry::Compaction(_))
    ));
    let context = session.build_session_context();
    assert_eq!(
        context.messages.len(),
        3,
        "the compaction summary + kept + after messages"
    );
    assert!(
        matches!(&context.messages[0], AgentMessage::Custom(custom) if custom.role == "compactionSummary"),
        "the compaction entry projects its summary message"
    );
    assert_eq!(
        session.get_leaf_id(),
        Some(compaction_id.as_str()),
        "the compaction advances the leaf"
    );
}

#[test]
fn the_persist_rules_hold_the_file_until_an_assistant_message() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    let session_file = session.session_file().expect("file").to_owned();

    session
        .append_message(user_message("hello"))
        .expect("append");
    assert!(
        !fs::exists(&session_file).unwrap_or(false),
        "no file before the first assistant message"
    );

    session
        .append_message(assistant_message("hi"))
        .expect("append");
    let content = fs::read_to_string(&session_file).expect("file exists now");
    assert_eq!(
        content.lines().count(),
        3,
        "the buffered header and entries flush with the assistant"
    );
    let header: serde_json::Value =
        serde_json::from_str(content.lines().next().expect("header")).expect("json");
    assert_eq!(header["type"], "session");
    assert_eq!(header["version"], CURRENT_SESSION_VERSION);

    session
        .append_message(user_message("later"))
        .expect("append");
    let content = fs::read_to_string(&session_file).expect("read");
    assert_eq!(content.lines().count(), 4, "later entries append");
}

#[test]
fn opening_an_empty_file_initializes_a_header_and_opening_a_non_session_file_fails() {
    let dir = tempfile::tempdir().expect("temp dir");
    let empty = dir.path().join("empty.jsonl");
    fs::write(&empty, "").expect("write");

    let session = SessionManager::open(&empty.display().to_string(), None, None).expect("open");
    let header = session.get_header().expect("header initialized").clone();
    assert_eq!(header.version, Some(CURRENT_SESSION_VERSION));
    let content = fs::read_to_string(&empty).expect("read");
    assert_eq!(
        content.lines().count(),
        1,
        "the header was rewritten into the empty file"
    );

    let invalid = dir.path().join("not-a-session.log");
    let original = "{\"type\":\"event\",\"data\":\"not a session\"}\n";
    fs::write(&invalid, original).expect("write");
    let error =
        SessionManager::open(&invalid.display().to_string(), None, None).expect_err("open fails");
    assert!(
        error
            .to_string()
            .starts_with("Session file is not a valid pi session: "),
        "the message is upstream's: {error}"
    );
    assert_eq!(
        fs::read_to_string(&invalid).expect("read"),
        original,
        "the file is preserved"
    );
}

#[test]
fn opening_preserves_unknown_fields_and_rewrites_keep_them() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("session.jsonl");
    fs::write(
        &path,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"s1\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"{}\",\"custom\":\"kept\"}}\n",
            dir.path().display()
        ),
    )
    .expect("write");

    let mut session = SessionManager::open(&path.display().to_string(), None, None).expect("open");
    let header = session.get_header().expect("header");
    assert_eq!(
        header.extras.get("custom"),
        Some(&serde_json::json!("kept"))
    );
    session
        .append_message(assistant_message("hi"))
        .expect("append");

    let content = fs::read_to_string(&path).expect("read");
    assert!(
        content.contains("\"custom\":\"kept\""),
        "rewrites preserve unknown header fields: {content}"
    );
}

#[test]
fn opening_a_v1_file_migrates_and_rewrites() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("legacy.jsonl");
    fs::write(
        &path,
        "{\"type\":\"session\",\"id\":\"legacy\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"/tmp\"}\n{\"type\":\"message\",\"timestamp\":\"2026-01-01T00:00:01.000Z\",\"message\":{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":0}}\n",
    )
    .expect("write");

    let session = SessionManager::open(&path.display().to_string(), None, None).expect("open");
    let header = session.get_header().expect("header");
    assert_eq!(header.version, Some(CURRENT_SESSION_VERSION));
    assert_eq!(session.entries().len(), 1);
    let content = fs::read_to_string(&path).expect("read");
    assert!(
        content.contains("\"version\":3"),
        "the rewrite stamps the current version"
    );
    assert!(
        content.contains("\"parentId\":"),
        "the rewrite adds the tree structure"
    );
}

#[test]
fn create_branched_session_rebuilds_the_path_with_a_fresh_header() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    session
        .append_message(user_message("side"))
        .expect("append");
    session
        .append_message(assistant_message("hi"))
        .expect("append");
    let second = session
        .append_message(user_message("kept"))
        .expect("append");
    session.branch(&second).expect("branch");
    session
        .append_label_change(&first, Some("root label"))
        .expect("label");
    let previous_file = session.session_file().expect("previous file").to_owned();

    let branched_file = session
        .create_branched_session(&second)
        .expect("branch")
        .expect("persisted");
    assert!(branched_file.contains('_'), "timestamped name");
    assert_eq!(session.session_file(), Some(branched_file.as_str()));

    let content = fs::read_to_string(&branched_file).expect("read");
    let lines: Vec<&str> = content.lines().collect();
    let header: serde_json::Value = serde_json::from_str(lines[0]).expect("json");
    assert_eq!(header["type"], "session");
    assert_eq!(header["version"], CURRENT_SESSION_VERSION);
    assert_eq!(
        header["parentSession"].as_str(),
        Some(previous_file.as_str()),
        "the branched session links its source"
    );
    // The path is root → kept (four entries), re-chained linearly, plus one
    // label entry appended after the last path entry.
    assert_eq!(
        lines.len(),
        6,
        "header + 4 path entries + 1 label: {content}"
    );
    let entry_ids: Vec<String> = lines[1..]
        .iter()
        .map(|line| {
            serde_json::from_str::<serde_json::Value>(line).expect("json")["id"]
                .as_str()
                .expect("id")
                .to_owned()
        })
        .collect();
    assert_eq!(entry_ids[0], first, "the path starts at the root entry");
    assert_eq!(entry_ids[3], second, "the path ends at the branch leaf");
    let label: serde_json::Value = serde_json::from_str(lines[5]).expect("json");
    assert_eq!(label["type"], "label");
    assert_eq!(label["targetId"], first);
    assert_eq!(
        label["parentId"], entry_ids[3],
        "the label chains after the last path entry"
    );
    assert_eq!(label["label"], "root label");
    assert_eq!(session.session_id(), header["id"].as_str().expect("id"));
}

#[test]
fn in_memory_create_branched_session_replaces_the_session() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    session
        .append_message(user_message("dropped"))
        .expect("append");
    session
        .append_message(user_message("kept"))
        .expect("append");
    session.branch(&first).expect("branch");
    session
        .append_label_change(&first, Some("root"))
        .expect("label");

    assert_eq!(
        session.create_branched_session(&first).expect("branch"),
        None
    );
    assert_eq!(
        session.entries().len(),
        2,
        "path + label, the dropped entries are gone"
    );
    let header = session.get_header().expect("header");
    assert_eq!(
        header.parent_session, None,
        "in-memory branched sessions carry no parent link"
    );
    assert!(
        session.get_label(&first).is_some(),
        "labels survive the rebuild"
    );
}

#[test]
fn new_session_validates_the_explicit_id() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let error = session
        .new_session(Some(NewSessionOptions {
            id: Some("-bad".to_owned()),
            parent_session: None,
        }))
        .expect_err("invalid id");
    assert_eq!(error, SessionManagerError::InvalidSessionId);
}

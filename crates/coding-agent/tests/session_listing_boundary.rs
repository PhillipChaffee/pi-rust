//! The listing boundary suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the message-text extraction
//! arms `buildSessionInfo` walks, the headerless-file exclusion, and the
//! directory-entry skips of most-recent discovery.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::fs;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{
    ImageContent, KnownApi, Message, ProviderId, TextContent, ToolCall, ToolResultBlock,
    ToolResultMessage, UserBlock, UserContent, UserMessage,
};

use pi_coding_agent::session_manager::{
    FileEntry, MessageEntry, SessionEntry, SessionEntryBase, SessionManager,
    find_most_recent_session,
};

const TS: &str = "2026-01-01T00:00:00.000Z";

fn base(id: &str, parent: Option<&str>) -> SessionEntryBase {
    SessionEntryBase {
        id: Some(id.to_owned()),
        parent_id: parent.map(str::to_owned),
        timestamp: TS.to_owned(),
        extras: serde_json::Map::default(),
    }
}

fn message_entry(id: &str, parent: Option<&str>, message: Option<AgentMessage>) -> FileEntry {
    FileEntry::Entry(SessionEntry::Message(MessageEntry {
        base: base(id, parent),
        message,
        extras: serde_json::Map::default(),
    }))
}

fn write_jsonl(path: &str, lines: &[FileEntry]) {
    let body: String = lines
        .iter()
        .map(|entry| serde_json::to_string(entry).expect("serialize entry"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(path, format!("{body}\n")).expect("write session file");
}

#[tokio::test(flavor = "current_thread")]
async fn the_listing_extracts_block_text_and_skips_non_text_messages() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let path = format!("{temp}/rich.jsonl");

    let user = AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Blocks(vec![
            UserBlock::Text(TextContent {
                text: "hello".to_owned(),
                text_signature: None,
            }),
            UserBlock::Image(ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ]),
        timestamp: 100,
    }));
    let assistant = AgentMessage::Standard(Message::Assistant(tool_call_assistant(200)));
    let tool_result = AgentMessage::Standard(Message::ToolResult(ToolResultMessage {
        tool_call_id: "call-1".to_owned(),
        tool_name: "rich".to_owned(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: "done".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 300,
    }));
    let custom = AgentMessage::Custom(pi_coding_agent::messages::create_custom_message(
        "ext.note",
        UserContent::Text("note".to_owned()),
        true,
        None,
        TS,
    ));

    let header = FileEntry::Session(
        serde_json::from_str(&format!(
            r#"{{"type":"session","version":3,"id":"rich","timestamp":"{TS}","cwd":"{temp}"}}"#
        ))
        .expect("header"),
    );
    let lines = vec![
        header,
        message_entry("e1", None, Some(user)),
        message_entry("e2", Some("e1"), Some(assistant)),
        message_entry("e3", Some("e2"), Some(tool_result)),
        message_entry("e4", Some("e3"), Some(custom)),
        message_entry("e5", Some("e4"), None),
        FileEntry::Other(serde_json::json!({"type": "mystery", "id": "e6", "parentId": "e5"})),
    ];
    write_jsonl(&path, &lines);

    let sessions = SessionManager::list_all_from_dir(&temp, None)
        .await
        .expect("list");
    assert_eq!(sessions.len(), 1, "the rich session lists");
    let info = &sessions[0];
    assert_eq!(info.id, "rich");
    assert_eq!(
        info.message_count, 5,
        "every message entry counts, text or not"
    );
    assert_eq!(
        info.first_message, "hello",
        "the first user text block wins; images do not"
    );
    assert_eq!(
        info.all_messages_text, "hello",
        "tool results, custom messages, and bare entries drop"
    );
    assert_eq!(
        info.modified, 200,
        "the latest user/assistant timestamp reports; tool results do not"
    );
}

fn tool_call_assistant(timestamp: i64) -> pi_ai::types::AssistantMessage {
    pi_ai::types::AssistantMessage {
        content: vec![pi_ai::types::AssistantBlock::ToolCall(ToolCall {
            id: "call-1".to_owned(),
            name: "rich".to_owned(),
            arguments: serde_json::Map::new(),
            thought_signature: None,
            namespace: None,
        })],
        api: KnownApi::AnthropicMessages.into(),
        provider: ProviderId("anthropic".to_owned()),
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
        stop_reason: pi_ai::types::StopReason::ToolUse,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp,
    }
}

#[tokio::test(flavor = "current_thread")]
async fn a_file_that_does_not_open_with_a_header_is_not_a_session() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let path = format!("{temp}/junk.jsonl");
    fs::write(&path, r#"{"type":"mystery","id":"j1"}"#).expect("write junk");
    let sessions = SessionManager::list_all_from_dir(&temp, None)
        .await
        .expect("list");
    assert!(sessions.is_empty(), "the headerless file excludes itself");
}

#[test]
fn most_recent_discovery_skips_non_jsonl_and_headerless_files() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    fs::write(format!("{temp}/notes.txt"), "not a session").expect("write notes");
    fs::write(format!("{temp}/broken.jsonl"), "garbage\n").expect("write broken");
    let valid = format!("{temp}/valid.jsonl");
    write_jsonl(&valid, &[FileEntry::Session(
        serde_json::from_str(r#"{"type":"session","version":3,"id":"ok","timestamp":"2026-01-01T00:00:00.000Z"}"#)
            .expect("header"),
    )]);

    let found = find_most_recent_session(&temp, None).expect("the valid session wins");
    assert_eq!(found, valid);
}

#[tokio::test(flavor = "current_thread")]
async fn listing_a_non_directory_reports_no_sessions() {
    let dir = tempfile::tempdir().expect("temp dir");
    let file = format!("{}/plain.txt", dir.path().display());
    fs::write(&file, "a file, not a directory").expect("write file");
    let sessions = SessionManager::list_all_from_dir(&file, None)
        .await
        .expect("list");
    assert!(sessions.is_empty(), "a file path lists nothing");
}

#[tokio::test(flavor = "current_thread")]
async fn a_garbage_header_timestamp_falls_back_to_the_file_mtime() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let path = format!("{temp}/stale.jsonl");
    write_jsonl(
        &path,
        &[FileEntry::Session(
            serde_json::from_str(
                r#"{"type":"session","version":3,"id":"stale","timestamp":"not-a-date"}"#,
            )
            .expect("header"),
        )],
    );
    let sessions = SessionManager::list_all_from_dir(&temp, None)
        .await
        .expect("list");
    assert_eq!(sessions.len(), 1);
    assert_eq!(
        sessions[0].created, None,
        "the unparseable header timestamp reports absent"
    );
    assert!(
        sessions[0].modified > 1_500_000_000_000,
        "no message activity and no header time fall back to the mtime: {}",
        sessions[0].modified
    );
}

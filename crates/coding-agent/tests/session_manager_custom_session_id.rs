//! The session-manager custom-id suite, upstream's
//! `test/session-manager/custom-session-id.test.ts` ported 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use pi_coding_agent::session_manager::assert_valid_session_id;
use pi_coding_agent::session_manager::{NewSessionOptions, SessionManager};

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_millis()
        .try_into()
        .expect("epoch millis fit u64")
}

#[test]
fn uses_the_provided_id_instead_of_generating_one() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    session
        .new_session(Some(NewSessionOptions {
            id: Some("my-custom-id".to_owned()),
            parent_session: None,
        }))
        .expect("new session");
    assert_eq!(session.session_id(), "my-custom-id");
}

#[test]
fn uses_the_provided_id_when_creating_an_in_memory_session() {
    let session = SessionManager::in_memory(
        None,
        Some(NewSessionOptions {
            id: Some("memory-session-id".to_owned()),
            parent_session: None,
        }),
        None,
    )
    .expect("in-memory");
    assert_eq!(session.session_id(), "memory-session-id");
    assert_eq!(
        session.get_header().expect("header").id,
        "memory-session-id"
    );
    assert_eq!(session.session_file(), None);
}

#[test]
fn allows_alphanumeric_session_ids_with_interior_punctuation() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    session
        .new_session(Some(NewSessionOptions {
            id: Some("abc-123_def.456".to_owned()),
            parent_session: None,
        }))
        .expect("new session");
    assert_eq!(session.session_id(), "abc-123_def.456");
}

#[test]
fn rejects_invalid_custom_session_ids() {
    for id in [
        "", "-abc", "abc-", "_abc", "abc_", ".abc", "abc.", "abc/def", "abc\\def", "abc def",
    ] {
        assert!(assert_valid_session_id(id).is_err(), "rejected: {id}");
        let error = assert_valid_session_id(id).expect_err(id);
        assert!(
            error
                .to_string()
                .starts_with("Session id must be non-empty, contain only alphanumeric characters"),
            "the message is upstream's: {error}"
        );
    }
}

#[test]
fn generates_a_uuid_v7_id_when_no_id_is_provided() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    session.new_session(None).expect("new session");
    let id = session.session_id();
    assert!(!id.is_empty());
    assert!(is_uuid_v7(id), "uuidv7-shaped: {id}");
}

#[test]
fn generates_a_uuid_v7_id_when_options_is_provided_without_id() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    session
        .new_session(Some(NewSessionOptions {
            id: None,
            parent_session: Some("parent.jsonl".to_owned()),
        }))
        .expect("new session");
    assert!(is_uuid_v7(session.session_id()));
}

#[test]
fn includes_the_custom_id_in_the_session_header() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    session
        .new_session(Some(NewSessionOptions {
            id: Some("header-test-id".to_owned()),
            parent_session: None,
        }))
        .expect("new session");
    assert_eq!(session.get_header().expect("header").id, "header-test-id");
}

#[test]
fn generates_a_uuid_v7_id_when_constructed_without_an_explicit_id() {
    let session = SessionManager::in_memory(None, None, None).expect("in-memory");
    assert!(is_uuid_v7(session.session_id()));
    assert_eq!(
        session.get_header().expect("header").id,
        session.session_id()
    );
}

#[test]
fn uses_the_provided_id_when_creating_a_persisted_session() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let session = SessionManager::create(
        &temp,
        Some(&temp),
        Some(NewSessionOptions {
            id: Some("created-session-id".to_owned()),
            parent_session: None,
        }),
    )
    .expect("create");

    assert_eq!(session.session_id(), "created-session-id");
    assert_eq!(
        session.get_header().expect("header").id,
        "created-session-id"
    );
    let session_file = session.session_file().expect("file");
    assert!(session_file.contains("created-session-id"));
    let basename = session_file.rsplit('/').next().expect("basename");
    assert!(
        basename.ends_with("_created-session-id.jsonl")
            && basename.len()
                == "0000-00-00T00-00-00-000Z".len() + "_created-session-id.jsonl".len(),
        "filename shape: {basename}"
    );
    assert!(
        !fs::exists(session_file).unwrap_or(false),
        "no file until an assistant message flushes"
    );
}

#[test]
fn generates_a_uuid_v7_id_when_creating_a_branched_session() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let first_id = session
        .append_message(pi_agent_core::types::AgentMessage::Standard(
            pi_ai::types::Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Blocks(vec![pi_ai::types::UserBlock::Text(
                    pi_ai::types::TextContent {
                        text: "hello".to_owned(),
                        text_signature: None,
                    },
                )]),
                timestamp: now_millis().cast_signed(),
            }),
        ))
        .expect("append");

    session.create_branched_session(&first_id).expect("branch");

    assert!(is_uuid_v7(session.session_id()));
    assert_eq!(
        session.get_header().expect("header").id,
        session.session_id()
    );
}

#[test]
fn generates_a_uuid_v7_id_when_forking_from_another_session_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let source_path = format!("{temp}/source.jsonl");
    let lines = [
        format!(
            r#"{{"type":"session","version":3,"id":"legacy-session-id","timestamp":"{}","cwd":"{temp}"}}"#,
            pi_agent_core::harness::session::jsonl::codec::format_iso8601(
                now_millis().cast_signed()
            )
        ),
        format!(
            r#"{{"type":"message","id":"entry-1","parentId":null,"timestamp":"{}","message":{{"role":"assistant","content":[{{"type":"text","text":"hello"}}],"api":"openai-responses","provider":"openai","model":"gpt-5.4","usage":{{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}}}},"stopReason":"stop","timestamp":{}}}}}"#,
            pi_agent_core::harness::session::jsonl::codec::format_iso8601(
                now_millis().cast_signed()
            ),
            now_millis().cast_signed()
        ),
    ];
    fs::write(&source_path, lines.join("\n")).expect("write fixture");

    let forked = SessionManager::fork_from(&source_path, &temp, Some(&temp), None).expect("fork");
    let header = forked.get_header().expect("header");
    assert!(is_uuid_v7(&header.id));
    assert_eq!(header.parent_session.as_deref(), Some(source_path.as_str()));
}

#[test]
fn uses_the_provided_id_when_forking_from_another_session_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let source_path = format!("{temp}/source.jsonl");
    fs::write(
        &source_path,
        format!(
            r#"{{"type":"session","version":3,"id":"source-session-id","timestamp":"{}","cwd":"{temp}"}}"#,
            pi_agent_core::harness::session::jsonl::codec::format_iso8601(now_millis().cast_signed())
        ),
    )
    .expect("write fixture");

    let forked = SessionManager::fork_from(
        &source_path,
        &temp,
        Some(&temp),
        Some(NewSessionOptions {
            id: Some("forked-session-id".to_owned()),
            parent_session: None,
        }),
    )
    .expect("fork");
    let header = forked.get_header().expect("header");
    assert_eq!(header.id, "forked-session-id");
    assert_eq!(header.parent_session.as_deref(), Some(source_path.as_str()));
    let session_file = forked.session_file().expect("file");
    let basename = session_file.rsplit('/').next().expect("basename");
    assert!(
        basename.ends_with("_forked-session-id.jsonl"),
        "filename shape: {basename}"
    );
}

/// The RFC 9562 UUIDv7 shape, the upstream suite's `UUID_V7_RE`.
fn is_uuid_v7(id: &str) -> bool {
    let bytes = id.as_bytes();
    bytes.len() == 36
        && bytes[14] == b'7'
        && bytes.iter().enumerate().all(|(index, byte)| {
            matches!(byte, b'0'..=b'9' | b'a'..=b'f')
                || *byte == b'-' && matches!(index, 8 | 13 | 18 | 23)
        })
        && matches!(bytes[19], b'8'..=b'b')
}

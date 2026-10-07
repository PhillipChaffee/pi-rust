//! The session-manager boundary suite, second pass: the append surface's
//! wire shapes, the error taxonomy's messages, the discovery info fields,
//! and the projection arms the 1:1 suites do not reach, pinned against
//! upstream at `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
use std::fs;
use std::future::Future;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{Message, UserContent, UserMessage};

use pi_coding_agent::session_manager::{
    FileEntry, SessionEntry, SessionManager, SessionManagerError, load_entries_from_file,
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
        timestamp: 42,
    }))
}

// ---------------------------------------------------------------------------
// The append surface's wire shapes.
// ---------------------------------------------------------------------------

#[test]
fn every_append_method_writes_its_wire_shape() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    session
        .append_message(assistant_message("flush"))
        .expect("flush");
    let file = session.session_file().expect("file").to_owned();

    session
        .append_thinking_level_change("high")
        .expect("thinking");
    session
        .append_model_change("openai", "gpt-5")
        .expect("model");
    session
        .append_compaction(
            "the summary",
            "entry-1",
            100,
            Some(serde_json::json!({"k": 1})),
            Some(false),
            None,
        )
        .expect("compaction");
    session
        .append_custom_entry("ext.state", Some(serde_json::json!({"v": 2})))
        .expect("custom");
    session
        .append_custom_message_entry(
            "ext.note",
            UserContent::Text("note".to_owned()),
            true,
            Some(serde_json::json!({"d": 3})),
        )
        .expect("custom message");

    let lines: Vec<serde_json::Value> = fs::read_to_string(&file)
        .expect("read")
        .lines()
        .map(|line| serde_json::from_str(line).expect("json"))
        .collect();
    let last = |field: &str| lines[lines.len() - 1][field].clone();
    let thinking = &lines[lines.len() - 5];
    assert_eq!(thinking["type"], "thinking_level_change");
    assert_eq!(thinking["thinkingLevel"], "high");
    let model = &lines[lines.len() - 4];
    assert_eq!(model["type"], "model_change");
    assert_eq!(model["provider"], "openai");
    assert_eq!(model["modelId"], "gpt-5");
    let compaction = &lines[lines.len() - 3];
    assert_eq!(compaction["type"], "compaction");
    assert_eq!(compaction["summary"], "the summary");
    assert_eq!(compaction["firstKeptEntryId"], "entry-1");
    assert_eq!(compaction["tokensBefore"], 100);
    assert_eq!(compaction["details"], serde_json::json!({"k": 1}));
    assert_eq!(
        compaction["fromHook"], false,
        "an explicit false serializes"
    );
    let custom = &lines[lines.len() - 2];
    assert_eq!(custom["type"], "custom");
    assert_eq!(custom["customType"], "ext.state");
    assert_eq!(custom["data"], serde_json::json!({"v": 2}));
    let custom_message = &lines[lines.len() - 1];
    assert_eq!(custom_message["type"], "custom_message");
    assert_eq!(custom_message["customType"], "ext.note");
    assert_eq!(custom_message["content"], "note");
    assert_eq!(custom_message["display"], true);
    assert_eq!(custom_message["details"], serde_json::json!({"d": 3}));
    let _ = last("");
}

#[test]
fn absent_optionals_drop_from_the_wire() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    session
        .append_message(assistant_message("flush"))
        .expect("flush");
    let file = session.session_file().expect("file").to_owned();

    session
        .append_compaction("bare", "entry-1", 5, None, None, None)
        .expect("compaction");
    session.append_custom_entry("bare", None).expect("custom");

    let lines: Vec<serde_json::Value> = fs::read_to_string(&file)
        .expect("read")
        .lines()
        .map(|line| serde_json::from_str(line).expect("json"))
        .collect();
    let compaction = &lines[lines.len() - 2];
    assert!(
        compaction.get("details").is_none(),
        "absent details drop: {compaction}"
    );
    assert!(compaction.get("usage").is_none());
    assert!(compaction.get("fromHook").is_none());
    let custom = &lines[lines.len() - 1];
    assert!(custom.get("data").is_none(), "absent data drops: {custom}");
}

// ---------------------------------------------------------------------------
// The error taxonomy's messages.
// ---------------------------------------------------------------------------

#[test]
fn the_error_taxonomy_carries_upstreams_messages() {
    assert_eq!(
        SessionManagerError::InvalidSessionId.to_string(),
        "Session id must be non-empty, contain only alphanumeric characters, '-', '_', and '.', and start and end with an alphanumeric character"
    );
    assert_eq!(
        SessionManagerError::InvalidSessionFile("/x/y.jsonl".to_owned()).to_string(),
        "Session file is not a valid pi session: /x/y.jsonl"
    );
    assert_eq!(
        SessionManagerError::EntryNotFound("e1".to_owned()).to_string(),
        "Entry e1 not found"
    );
    assert_eq!(
        SessionManagerError::ForkSourceInvalid("/src.jsonl".to_owned()).to_string(),
        "Cannot fork: source session file is empty or invalid: /src.jsonl"
    );
    assert_eq!(
        SessionManagerError::ForkSourceHeaderless("/src.jsonl".to_owned()).to_string(),
        "Cannot fork: source session has no header: /src.jsonl"
    );
    assert_eq!(
        SessionManagerError::Io("boom".to_owned()).to_string(),
        "boom"
    );
}

#[test]
fn fork_rejects_empty_and_headerless_sources() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let empty = format!("{temp}/empty.jsonl");
    fs::write(&empty, "").expect("write");
    let error =
        SessionManager::fork_from(&empty, &temp, Some(&temp), None).expect_err("empty source");
    assert_eq!(
        error.to_string(),
        format!("Cannot fork: source session file is empty or invalid: {empty}")
    );

    let headerless = format!("{temp}/headerless.jsonl");
    fs::write(
        &headerless,
        "{\"type\":\"message\",\"id\":\"m1\",\"parentId\":null,\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"message\":{\"role\":\"user\",\"content\":\"x\",\"timestamp\":0}}\n",
    )
    .expect("write");
    // Upstream's load gate rejects a file whose first entry is not a session
    // header before the headerless arm can fire, so a headerless file also
    // reports "empty or invalid" (the Headerless arm stays defensive).
    let error = SessionManager::fork_from(&headerless, &temp, Some(&temp), None)
        .expect_err("headerless source");
    assert_eq!(
        error.to_string(),
        format!("Cannot fork: source session file is empty or invalid: {headerless}")
    );
}

#[test]
fn fork_creates_the_target_directory_and_reopens_the_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let source = format!("{temp}/source.jsonl");
    let nested = format!("{temp}/nested/sessions");
    fs::write(
        &source,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"src\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"{temp}\"}}\n{{\"type\":\"message\",\"id\":\"m1\",\"parentId\":null,\"timestamp\":\"2026-01-01T00:00:01.000Z\",\"message\":{{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":0}}}}\n"
        ),
    )
    .expect("write");

    let forked = SessionManager::fork_from(&source, &temp, Some(&nested), None).expect("fork");
    assert_eq!(
        forked.entries().len(),
        1,
        "the header is excluded from the entries"
    );
    assert_ne!(
        forked.session_id(),
        "src",
        "the fork keeps its own fresh id"
    );
    assert_eq!(forked.build_session_context().messages.len(), 1);
    let file = forked.session_file().expect("file").to_owned();
    assert!(
        file.starts_with(&nested),
        "the fork lands in the (created) target directory: {file}"
    );
    let content = fs::read_to_string(&file).expect("read");
    assert!(
        content.contains(&format!("\"parentSession\":\"{source}\"")),
        "the header links the source: {content}"
    );
}

// ---------------------------------------------------------------------------
// Discovery info fields.
// ---------------------------------------------------------------------------

#[test]
fn the_listing_reports_names_counts_and_text_fields() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let mut session = SessionManager::create(&temp, Some(&temp), None).expect("create");
    let file = session.session_file().expect("file").to_owned();
    session
        .append_message(user_message("first user text"))
        .expect("append");
    session
        .append_message(assistant_message("assistant text"))
        .expect("append");
    session
        .append_message(user_message("second"))
        .expect("append");
    session.append_session_info("the name").expect("name");

    let sessions = futures_block_on(SessionManager::list_all_from_dir(&temp, None)).expect("list");
    assert_eq!(sessions.len(), 1);
    let info = &sessions[0];
    assert_eq!(info.path, file);
    assert_eq!(info.id, session.session_id());
    assert_eq!(info.cwd, temp);
    assert_eq!(info.name.as_deref(), Some("the name"));
    assert_eq!(info.message_count, 3);
    assert_eq!(info.first_message, "first user text");
    assert_eq!(
        info.all_messages_text,
        "first user text assistant text second"
    );
    assert!(info.created.is_some());
    assert_eq!(info.parent_session_path, None);
}

#[test]
fn a_session_without_messages_reports_the_placeholder() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let path = format!("{temp}/bare.jsonl");
    fs::write(
        &path,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"bare\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"{temp}\"}}\n"
        ),
    )
    .expect("write");
    let sessions = futures_block_on(SessionManager::list_all_from_dir(&temp, None)).expect("list");
    let info = &sessions[0];
    assert_eq!(info.first_message, "(no messages)");
    assert_eq!(info.message_count, 0);
    assert_eq!(info.all_messages_text, "");
}

#[test]
fn a_garbage_header_timestamp_falls_back_to_the_mtime() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let path = format!("{temp}/odd.jsonl");
    fs::write(
        &path,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"odd\",\"timestamp\":\"not-a-date\",\"cwd\":\"{temp}\"}}\n"
        ),
    )
    .expect("write");
    let sessions = futures_block_on(SessionManager::list_all_from_dir(&temp, None)).expect("list");
    let info = &sessions[0];
    assert_eq!(info.created, None, "the unparseable created reports absent");
    assert!(info.modified > 0, "the mtime fallback drives modified");
}

#[test]
fn a_session_with_a_parent_link_reports_it() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let path = format!("{temp}/child.jsonl");
    fs::write(
        &path,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"child\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"{temp}\",\"parentSession\":\"{temp}/parent.jsonl\"}}\n"
        ),
    )
    .expect("write");
    let sessions = futures_block_on(SessionManager::list_all_from_dir(&temp, None)).expect("list");
    assert_eq!(
        sessions[0].parent_session_path.as_deref(),
        Some(format!("{temp}/parent.jsonl").as_str())
    );
}

#[test]
fn listing_a_missing_directory_returns_empty() {
    let sessions = futures_block_on(SessionManager::list_all_from_dir(
        "/definitely/missing/dir",
        None,
    ))
    .expect("list");
    assert!(sessions.is_empty());
}

// ---------------------------------------------------------------------------
// Projection and tree arms.
// ---------------------------------------------------------------------------

#[test]
fn the_leaf_walk_and_children_handles_missing_targets() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    session
        .append_message(user_message("again"))
        .expect("append");

    assert!(session.get_entry("missing").is_none());
    assert!(session.get_children("missing").is_empty());
    assert!(session.get_label("missing").is_none());
    assert!(session.get_leaf_entry().is_some());
    assert_eq!(
        session.get_branch(Some(&first)).len(),
        1,
        "an explicit walk starts at the entry"
    );
    assert!(session.get_branch(Some("missing")).is_empty());

    session.reset_leaf();
    assert!(session.get_leaf_id().is_none());
    let appended = session
        .append_message(user_message("new root"))
        .expect("append");
    let entry = session.get_entry(&appended).expect("entry").clone();
    assert!(
        matches!(&entry, FileEntry::Entry(SessionEntry::Message(message)) if message.base.parent_id.is_none()),
        "a reset leaf starts a new root"
    );
}

#[test]
fn unknown_and_other_entries_project_no_messages() {
    // Raw values and info entries ride the entries list; the projection
    // yields no messages for them.
    let with_raw = SessionManager::in_memory(
        None,
        None,
        Some(vec![
            FileEntry::Session(pi_coding_agent::session_manager::SessionHeader {
                version: Some(3),
                id: "s".to_owned(),
                timestamp: "2026-01-01T00:00:00.000Z".to_owned(),
                cwd: None,
                parent_session: None,
                extras: serde_json::Map::default(),
            }),
            FileEntry::Other(serde_json::json!({"type":"mystery","id":"m1","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z"})),
            FileEntry::Entry(SessionEntry::SessionInfo(pi_coding_agent::session_manager::SessionInfoEntry {
                base: pi_coding_agent::session_manager::SessionEntryBase {
                    id: Some("i1".to_owned()),
                    parent_id: Some("m1".to_owned()),
                    timestamp: "2026-01-01T00:00:00.000Z".to_owned(),
                    extras: serde_json::Map::default(),
                },
                name: Some("named".to_owned()),
                extras: serde_json::Map::default(),
            })),
        ]),
    )
    .expect("in-memory");

    let context = with_raw.build_session_context();
    assert!(
        context.messages.is_empty(),
        "raw and info entries project no messages"
    );
    assert_eq!(context.thinking_level, "off");
    assert_eq!(context.model, None);
}

#[test]
fn an_empty_branch_summary_projects_no_message() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    let first = session
        .append_message(user_message("hello"))
        .expect("append");
    session
        .branch_with_summary(Some(&first), "", None, None, None)
        .expect("branch");

    let context = session.build_session_context();
    assert_eq!(
        context.messages.len(),
        1,
        "the empty summary is skipped upstream's truthy check"
    );

    // A branch from a null leaf re-parents the summary at the root; the
    // append then advances the leaf onto the summary entry, so the context
    // replays just the summary message.
    let mut parked = SessionManager::in_memory(None, None, None).expect("in-memory");
    parked
        .append_message(user_message("hello"))
        .expect("append");
    let summary_id = parked
        .branch_with_summary(None, "summary", None, None, None)
        .expect("branch");
    assert_eq!(parked.get_leaf_id(), Some(summary_id.as_str()));
    let context = parked.build_session_context();
    assert_eq!(context.messages.len(), 1);
    assert!(
        matches!(&context.messages[0], AgentMessage::Custom(custom) if custom.role == "branchSummary")
    );
}

#[test]
fn a_custom_message_entry_projects_its_custom_message() {
    let mut session = SessionManager::in_memory(None, None, None).expect("in-memory");
    session
        .append_message(user_message("hello"))
        .expect("append");
    session
        .append_custom_message_entry(
            "ext.note",
            UserContent::Text("injected".to_owned()),
            false,
            None,
        )
        .expect("custom message");

    let context = session.build_session_context();
    assert_eq!(context.messages.len(), 2);
    let AgentMessage::Custom(custom) = &context.messages[1] else {
        panic!("custom message projected");
    };
    assert_eq!(custom.role, "custom");
    assert_eq!(
        custom.field("customType"),
        Some(&serde_json::json!("ext.note"))
    );
    assert_eq!(
        custom.field("content"),
        Some(&serde_json::json!("injected"))
    );
    assert_eq!(custom.field("display"), Some(&serde_json::json!(false)));
}

// ---------------------------------------------------------------------------
// Constructor and file-mechanics arms.
// ---------------------------------------------------------------------------

#[test]
fn uses_default_session_dir_compares_against_the_encoded_default() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let custom = SessionManager::create(&temp, Some(&temp), None).expect("create");
    assert!(
        !custom.uses_default_session_dir(),
        "a custom directory is not the default"
    );

    // The comparison target is the encoded default for the cwd under the
    // ambient agent dir; the encoded shape carries the #118 encoding.
    let default_path = pi_coding_agent::config::default_session_dir_path(
        &temp,
        &pi_coding_agent::config::get_agent_dir()
            .display()
            .to_string(),
    );
    let encoded = default_path
        .file_name()
        .expect("encoded")
        .to_string_lossy()
        .into_owned();
    assert!(
        encoded.starts_with("--") && encoded.ends_with("--"),
        "the encoded cwd wraps in dashes: {encoded}"
    );
}

#[test]
fn an_explicit_parent_session_lands_in_the_header() {
    let session = SessionManager::in_memory(
        None,
        Some(pi_coding_agent::session_manager::NewSessionOptions {
            id: None,
            parent_session: Some("/parent/path.jsonl".to_owned()),
        }),
        None,
    )
    .expect("in-memory");
    assert_eq!(
        session
            .get_header()
            .expect("header")
            .parent_session
            .as_deref(),
        Some("/parent/path.jsonl")
    );
}

#[test]
fn set_session_file_switches_to_another_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let temp = dir.path().display().to_string();
    let a = format!("{temp}/a.jsonl");
    let b = format!("{temp}/b.jsonl");
    fs::write(
        &a,
        format!(
            "{{\"type\":\"session\",\"version\":3,\"id\":\"a\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"{temp}\"}}\n"
        ),
    )
    .expect("write");

    let mut session = SessionManager::open(&a, None, None).expect("open");
    assert_eq!(session.session_id(), "a");
    session.set_session_file(&b).expect("switch");
    assert_eq!(session.session_file(), Some(b.as_str()));
    assert_eq!(
        session.get_header().expect("header").id,
        session.session_id()
    );
    // The fresh file stays empty until an assistant message flushes.
    assert!(!fs::exists(&b).unwrap_or(false));
    session
        .append_message(assistant_message("hi"))
        .expect("append");
    assert!(fs::exists(&b).unwrap_or(false));
}

#[test]
fn in_memory_entries_without_a_header_start_fresh_and_extend() {
    let entries = vec![
        FileEntry::Other(
            serde_json::json!({"type":"mystery","id":"m1","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z"}),
        ),
        FileEntry::Other(
            serde_json::json!({"type":"mystery","id":"m2","parentId":"m1","timestamp":"2026-01-01T00:00:01.000Z"}),
        ),
    ];
    let session = SessionManager::in_memory(None, None, Some(entries)).expect("in-memory");
    assert_eq!(
        session.entries().len(),
        2,
        "the header precedes the carried entries"
    );
    assert_eq!(
        session.get_entry("m2").and_then(FileEntry::entry_parent_id),
        Some("m1")
    );
    assert_eq!(session.get_leaf_id(), Some("m2"));
}

#[test]
fn load_entries_from_file_reads_blank_and_malformed_lines_leniently() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("lenient.jsonl");
    fs::write(
        &path,
        format!(
            "\n   \n{{not json}}\n{}\n{{\"type\":\"message\",\"id\":\"m1\",\"parentId\":null,\"timestamp\":\"2026-01-01T00:00:01.000Z\",\"message\":{{\"role\":\"user\",\"content\":\"hi\",\"timestamp\":0}}}}\n",
            serde_json::json!({"type": "session", "version": 3, "id": "s", "timestamp": "2026-01-01T00:00:00.000Z", "cwd": "/tmp"})
        ),
    )
    .expect("write");
    let entries = load_entries_from_file(&path.display().to_string());
    assert_eq!(
        entries.len(),
        2,
        "blank, whitespace, and malformed lines skip"
    );
}

/// The current-thread runtime helper.
fn futures_block_on<T>(future: impl Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(future)
}

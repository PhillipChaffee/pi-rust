//! The JSONL v3 migration suite, ported 1:1 from upstream (tests assert by panicking)
//! `test/harness/jsonl-v3-migration.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
#![expect(
    clippy::cast_precision_loss,
    reason = "the fixture usage costs are small integers; the f64 widening cannot lose them"
)]

use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::jsonl::storage::JsonlStorage;
use pi_agent_core::harness::session::jsonl::types::{
    JSONL_FORMAT_VERSION, JSONL_STORAGE_VERSION, JsonlSessionListOptions,
};
use pi_agent_core::harness::session::types::{
    BranchScan, BranchScanOrder, EntryQuery, EntryScan, EntryScanOrder, ForkOptions, Session,
    Storage,
};
use pi_agent_core::harness::session::values::{
    Write, branch_tip, entry_label, lane_config, lane_state, session_name, set_value,
};
use pi_agent_core::harness::types::FileSystem;
use serde_json::{Value as JsonValue, json};

mod jsonl_common;
use jsonl_common::{NOW, WrappedEnv, jsonl_repo};

use pi_agent_core::harness::session::types::{
    BranchSummaryEntryBody, CompactionEntryBody, CustomEntryBody, Entry, MessageEntry,
};

fn entry_message(entry: &Entry) -> &MessageEntry {
    match entry {
        Entry::Message { body, .. } => body,
        _ => panic!("expected a message entry"),
    }
}

fn entry_custom(entry: &Entry) -> &CustomEntryBody {
    match entry {
        Entry::Custom { body, .. } => body,
        _ => panic!("expected a custom entry"),
    }
}

fn entry_branch_summary(entry: &Entry) -> &BranchSummaryEntryBody {
    match entry {
        Entry::BranchSummary { body, .. } => body,
        _ => panic!("expected a branch-summary entry"),
    }
}

fn entry_compaction(entry: &Entry) -> &CompactionEntryBody {
    match entry {
        Entry::Compaction { body, .. } => body,
        _ => panic!("expected a compaction entry"),
    }
}

const FIRST_MESSAGE: &str = "first";
const SECOND_MESSAGE: &str = "second";

fn iso(millis: i64) -> String {
    pi_agent_core::harness::session::jsonl::codec::format_iso8601(millis)
}

fn uuid_timestamp(id: &str) -> u64 {
    let digits: String = id.chars().filter(|c| *c != '-').collect();
    u64::from_str_radix(&digits[..12], 16).expect("uuid timestamp")
}

fn user_message(text: &str, timestamp: i64) -> JsonValue {
    json!({
        "role": "user",
        "content": [{ "type": "text", "text": text }],
        "timestamp": timestamp,
    })
}

fn assistant_message(text: &str, timestamp: i64, usage: &JsonValue) -> JsonValue {
    json!({
        "role": "assistant",
        "content": [{ "type": "text", "text": text }],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "usage": usage,
        "stopReason": "stop",
        "timestamp": timestamp,
    })
}

fn tool_result_message(timestamp: i64, usage: &JsonValue) -> JsonValue {
    json!({
        "role": "toolResult",
        "toolCallId": "call-1",
        "toolName": "test",
        "content": [{ "type": "text", "text": "result" }],
        "usage": usage,
        "isError": false,
        "timestamp": timestamp,
    })
}

fn usage(factor: i64) -> JsonValue {
    json!({
        "input": factor,
        "output": factor * 2,
        "cacheRead": factor * 3,
        "cacheWrite": factor * 4,
        "cacheWrite1h": factor * 5,
        "reasoning": factor * 6,
        "totalTokens": factor * 10,
        "cost": {
            "input": factor as f64,
            "output": (factor * 2) as f64,
            "cacheRead": (factor * 3) as f64,
            "cacheWrite": (factor * 4) as f64,
            "total": (factor * 10) as f64,
        },
    })
}

fn simple_usage(factor: i64) -> JsonValue {
    json!({
        "input": factor,
        "output": factor * 2,
        "cacheRead": factor * 3,
        "cacheWrite": factor * 4,
        "totalTokens": factor * 10,
        "cost": {
            "input": factor as f64,
            "output": (factor * 2) as f64,
            "cacheRead": (factor * 3) as f64,
            "cacheWrite": (factor * 4) as f64,
            "total": (factor * 10) as f64,
        },
    })
}

/// The shared fixture the migration suite drives: the failable-rename
/// filesystem, the repo, and the legacy-v3 writer, upstream's
/// `writeLegacyV3Fixture`.
struct Suite {
    _root: jsonl_common::TempRoot,
    file_system: Arc<WrappedEnv>,
    env: Arc<dyn FileSystem>,
    repo: pi_agent_core::harness::session::jsonl::repo::JsonlSessionRepo,
    fixture_id: usize,
}

impl Suite {
    fn new() -> Self {
        let root = jsonl_common::TempRoot::new();
        let file_system = WrappedEnv::new(root.path().to_owned());
        let env: Arc<dyn FileSystem> = file_system.clone();
        let repo = jsonl_repo(file_system.clone());
        Self {
            _root: root,
            file_system,
            env,
            repo,
            fixture_id: 0,
        }
    }

    async fn write_legacy_v3_fixture(
        &mut self,
        records: &[JsonValue],
        parent_session: Option<&str>,
    ) -> (String, String) {
        let directory = self
            .env
            .join_path(
                &["sessions".to_owned(), "--workspace--".to_owned()],
                &background_context(),
            )
            .await
            .expect("directory join");
        self.env
            .create_dir(&directory, None, &background_context())
            .await
            .expect("directory create");
        let relative = self
            .env
            .join_path(
                &[directory, "legacy.jsonl".to_owned()],
                &background_context(),
            )
            .await
            .expect("file join");
        let path = self
            .env
            .absolute_path(&relative, &background_context())
            .await
            .expect("absolute path");
        let mut lines = vec![
            serde_json::json!({
                "type": "session",
                "version": 3,
                "id": "legacy",
                "timestamp": iso(NOW),
                "cwd": "/workspace",
            })
            .to_string(),
        ];
        if let Some(parent_session) = parent_session {
            let mut header: JsonValue = serde_json::from_str(&lines[0]).expect("header");
            header["parentSession"] = json!(parent_session);
            lines[0] = header.to_string();
        }
        lines.extend(records.iter().map(JsonValue::to_string));
        let content = format!("{}\n", lines.join("\n"));
        std::fs::write(&path, &content).expect("fixture write");
        self.fixture_id += 1;
        (path, content)
    }

    async fn discover(
        &self,
    ) -> pi_agent_core::harness::session::jsonl::types::JsonlSessionMetadata {
        let metadata = self
            .repo
            .list(
                Some(JsonlSessionListOptions {
                    cwd: Some("/workspace".to_owned()),
                }),
                &background_context(),
            )
            .await
            .expect("list");
        metadata
            .into_iter()
            .next()
            .expect("legacy fixture discovered")
    }

    async fn open(
        &self,
        metadata: &pi_agent_core::harness::session::jsonl::types::JsonlSessionMetadata,
    ) -> Box<dyn Session> {
        self.repo
            .open(metadata, &background_context())
            .await
            .expect("open")
            .0
    }

    async fn main_tip(session: &dyn Session) -> Option<String> {
        let branch = session
            .branch("main", &background_context())
            .await
            .expect("branch")
            .expect("imported main Branch");
        branch.get_tip_id(&background_context()).await.expect("tip")
    }

    /// Discovers the fixture's metadata and opens it, the migration
    /// suite's repeated preamble.
    async fn open_discovered(&self) -> Box<dyn Session> {
        let metadata = self.discover().await;
        self.open(&metadata).await
    }

    /// The session's asc entries with the expected count pinned.
    async fn imported_chain(session: &dyn Session, expected_len: usize) -> Vec<Entry> {
        let entries = Self::entries_asc(session).await;
        assert_eq!(entries.len(), expected_len);
        entries
    }

    /// The child entry links to the parent entry, upstream's
    /// `expect(second.parentId).toBe(first.id)`.
    fn assert_child_parent(child: &Entry, parent: &Entry) {
        assert_eq!(
            child.parent_id().map(str::to_owned),
            Some(parent.id().to_owned())
        );
    }

    async fn entries_asc(session: &dyn Session) -> Vec<Entry> {
        session
            .find_entries(
                Some(&EntryQuery {
                    order: Some(EntryScanOrder::Asc),
                    ..Default::default()
                }),
                &background_context(),
            )
            .await
            .expect("entries")
    }
}

/// The imported configuration the config-change fixtures pin, upstream's
/// `laneConfig("main")` value assertions.
async fn assert_lane_config(session: &dyn Session, model_id: &str, tools: &[&str]) {
    let config = session
        .get_value(&lane_config("main").address, &background_context())
        .await
        .expect("lane config")
        .expect("stored config");
    assert_eq!(
        config.value,
        json!({
            "model": { "provider": "anthropic", "modelId": model_id },
            "thinkingLevel": "high",
            "activeToolNames": tools,
        }),
    );
}

/// The fresh idle lane state every configured import pins, upstream's
/// `laneState("main")` value assertion.
async fn assert_lane_state_fresh(session: &dyn Session) {
    let state = session
        .get_value(&lane_state("main").address, &background_context())
        .await
        .expect("lane state")
        .expect("stored state");
    assert_eq!(
        state.value,
        json!({ "currentOperationId": null, "lastOperationId": null, "inbox": [] }),
    );
}

/// Both reserved values absent, upstream's data-only assertions.
async fn assert_lane_values_absent(session: &dyn Session) {
    assert!(
        session
            .get_value(&lane_state("main").address, &background_context())
            .await
            .expect("lane state")
            .is_none(),
    );
    assert!(
        session
            .get_value(&lane_config("main").address, &background_context())
            .await
            .expect("lane config")
            .is_none(),
    );
}

/// The main tip equals the named entry, upstream's
/// `expect(await mainTip(fork)).toBe(entry.id)`.
async fn assert_main_tip_is(session: &dyn Session, entry_id: &str) {
    assert_eq!(Suite::main_tip(session).await, Some(entry_id.to_owned()));
}

/// The entry's stored label, upstream's `getLabel`.
async fn stored_label(session: &dyn Session, entry_id: &str) -> Option<String> {
    session
        .get_label(entry_id, &background_context())
        .await
        .expect("label")
}

/// The message-record wire shape every fixture repeats.
fn message_record(id: &str, parent_id: Option<&str>, timestamp: i64, text: &str) -> JsonValue {
    json!({
        "type": "message",
        "id": id,
        "parentId": parent_id,
        "timestamp": iso(timestamp),
        "message": user_message(text, timestamp),
    })
}

/// The assistant-message record wire shape.
fn assistant_record(
    id: &str,
    parent_id: Option<&str>,
    timestamp: i64,
    text: &str,
    usage: &JsonValue,
) -> JsonValue {
    json!({
        "type": "message",
        "id": id,
        "parentId": parent_id,
        "timestamp": iso(timestamp),
        "message": assistant_message(text, timestamp, usage),
    })
}

fn model_change_record(
    id: &str,
    parent_id: Option<&str>,
    timestamp: i64,
    provider: &str,
    model_id: &str,
) -> JsonValue {
    json!({
        "type": "model_change",
        "id": id,
        "parentId": parent_id,
        "timestamp": iso(timestamp),
        "provider": provider,
        "modelId": model_id,
    })
}

fn thinking_level_record(
    id: &str,
    parent_id: Option<&str>,
    timestamp: i64,
    level: &str,
) -> JsonValue {
    json!({
        "type": "thinking_level_change",
        "id": id,
        "parentId": parent_id,
        "timestamp": iso(timestamp),
        "thinkingLevel": level,
    })
}

fn active_tools_record(
    id: &str,
    parent_id: Option<&str>,
    timestamp: i64,
    names: &[&str],
) -> JsonValue {
    json!({
        "type": "active_tools_change",
        "id": id,
        "parentId": parent_id,
        "timestamp": iso(timestamp),
        "activeToolNames": names,
    })
}

fn session_info_record(
    id: &str,
    parent_id: Option<&str>,
    timestamp: i64,
    name: Option<&str>,
) -> JsonValue {
    let mut record = json!({
        "type": "session_info",
        "id": id,
        "parentId": parent_id,
        "timestamp": iso(timestamp),
    });
    if let Some(name) = name {
        record["name"] = json!(name);
    }
    record
}

fn label_record(
    id: &str,
    parent_id: Option<&str>,
    timestamp: i64,
    target_id: &str,
    label: Option<&str>,
) -> JsonValue {
    let mut record = json!({
        "type": "label",
        "id": id,
        "parentId": parent_id,
        "timestamp": iso(timestamp),
        "targetId": target_id,
    });
    if let Some(label) = label {
        record["label"] = json!(label);
    }
    record
}

fn compaction_record(
    id: &str,
    parent_id: Option<&str>,
    timestamp: i64,
    summary: &str,
    first_kept: &str,
    tokens_before: i64,
) -> JsonValue {
    json!({
        "type": "compaction",
        "id": id,
        "parentId": parent_id,
        "timestamp": iso(timestamp),
        "summary": summary,
        "firstKeptEntryId": first_kept,
        "tokensBefore": tokens_before,
    })
}

fn branch_summary_record(
    id: &str,
    parent_id: Option<&str>,
    timestamp: i64,
    from_id: &str,
    summary: &str,
) -> JsonValue {
    json!({
        "type": "branch_summary",
        "id": id,
        "parentId": parent_id,
        "timestamp": iso(timestamp),
        "fromId": from_id,
        "summary": summary,
    })
}

/// The asc usage scan the migration suite repeats, upstream's
/// `scanUsage({ order: "asc" })`.
async fn scan_usage_asc(
    storage: &dyn Storage,
) -> Vec<pi_agent_core::harness::session::types::UsageRow> {
    storage
        .scan_usage(
            &pi_agent_core::harness::session::types::UsageScan {
                order: Some(EntryScanOrder::Asc),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("usage scan")
}

fn zero_usage() -> JsonValue {
    json!({
        "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
        "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 },
    })
}

#[tokio::test]
async fn discovers_legacy_v3_session_files_without_rewriting_them() {
    let mut suite = Suite::new();
    let (path, content) = suite
        .write_legacy_v3_fixture(&[], Some("/old-session.jsonl"))
        .await;
    let metadata = suite.discover().await;
    let after = std::fs::read_to_string(&path).expect("after read");

    assert_eq!(metadata.id, "legacy");
    assert_eq!(metadata.created_at, NOW);
    assert_eq!(metadata.storage_version, JSONL_STORAGE_VERSION);
    assert_eq!(metadata.cwd, "/workspace");
    assert_eq!(metadata.path, path);
    assert_eq!(
        metadata.legacy_parent_session_path.as_deref(),
        Some("/old-session.jsonl")
    );
    assert_ne!(metadata.modified_at, 0);
    assert_eq!(after, content);
}

#[tokio::test]
async fn resolves_an_available_v3_parent_path_to_its_session_id() {
    let mut suite = Suite::new();
    let parent_path = suite
        .env
        .absolute_path("parent-v3.jsonl", &background_context())
        .await
        .expect("parent path");
    let parent_header = serde_json::json!({
        "type": "session",
        "version": 3,
        "id": "legacy-parent",
        "timestamp": iso(NOW - 1_000),
        "cwd": "/workspace",
    });
    std::fs::write(&parent_path, format!("{parent_header}\n")).expect("parent write");
    suite.write_legacy_v3_fixture(&[], Some(&parent_path)).await;
    let metadata = suite.discover().await;

    assert_eq!(metadata.id, "legacy");
    assert_eq!(metadata.parent_session_id.as_deref(), Some("legacy-parent"));
    assert!(metadata.legacy_parent_session_path.is_none());
    let (session, opened_metadata) = suite
        .repo
        .open(&metadata, &background_context())
        .await
        .expect("open");
    assert_eq!(
        opened_metadata.parent_session_id.as_deref(),
        Some("legacy-parent")
    );
    assert!(opened_metadata.legacy_parent_session_path.is_none());
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn resolves_an_available_v4_parent_path_to_its_session_id() {
    let mut suite = Suite::new();
    let parent_path = suite
        .env
        .absolute_path("parent-v4.jsonl", &background_context())
        .await
        .expect("parent path");
    let parent_header = serde_json::json!({
        "v": 4,
        "kind": "header",
        "id": "current-parent",
        "storageVersion": JSONL_STORAGE_VERSION,
        "createdAt": NOW - 1_000,
        "cwd": "/workspace",
    });
    std::fs::write(&parent_path, format!("{parent_header}\n")).expect("parent write");
    suite.write_legacy_v3_fixture(&[], Some(&parent_path)).await;
    let metadata = suite.discover().await;

    assert_eq!(
        metadata.parent_session_id.as_deref(),
        Some("current-parent")
    );
    assert!(metadata.legacy_parent_session_path.is_none());
    let (session, _) = suite
        .repo
        .open(&metadata, &background_context())
        .await
        .expect("open");
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn preserves_an_invalid_parent_path_as_legacy_metadata() {
    let mut suite = Suite::new();
    let parent_path = suite
        .env
        .absolute_path("invalid-parent.jsonl", &background_context())
        .await
        .expect("parent path");
    std::fs::write(&parent_path, "{\"not\":\"a session header\"}\n").expect("parent write");
    suite.write_legacy_v3_fixture(&[], Some(&parent_path)).await;
    let metadata = suite.discover().await;

    assert_eq!(
        metadata.legacy_parent_session_path.as_deref(),
        Some(parent_path.as_str())
    );
    assert!(metadata.parent_session_id.is_none());
}

fn fork_fixture_records() -> Vec<JsonValue> {
    vec![
        message_record("message-1", None, NOW + 1_000, "fork me"),
        label_record(
            "label-1",
            Some("message-1"),
            NOW + 2_000,
            "message-1",
            Some("Fork point"),
        ),
        session_info_record(
            "session-info",
            Some("label-1"),
            NOW + 3_000,
            Some("Imported fork"),
        ),
        assistant_record(
            "message-2",
            Some("session-info"),
            NOW + 4_000,
            "forked",
            &simple_usage(1),
        ),
    ]
}

/// The forked-destination assertions, upstream's `expectForkedState`.
async fn expect_forked_state(fork: &dyn Session) {
    let entries = Suite::entries_asc(fork).await;
    assert_eq!(entries.len(), 2);
    assert_eq!(
        serde_json::to_value(&entry_message(&entries[0]).message).expect("first message wire"),
        user_message("fork me", NOW + 1_000),
    );
    assert_eq!(entries[0].parent_id(), None);
    assert_eq!(
        serde_json::to_value(&entry_message(&entries[1]).message).expect("second message wire"),
        assistant_message("forked", NOW + 4_000, &simple_usage(1)),
    );
    Suite::assert_child_parent(&entries[1], &entries[0]);
    assert_ne!(entries[0].id(), "message-1");
    assert_ne!(entries[1].id(), "message-2");
    assert_eq!(
        Suite::main_tip(fork).await,
        Some(entries[1].id().to_owned())
    );
    assert_eq!(
        fork.get_name(&background_context()).await.expect("name"),
        Some("Imported fork".to_owned()),
    );
    assert_eq!(
        fork.get_label(entries[0].id(), &background_context())
            .await
            .expect("label"),
        Some("Fork point".to_owned()),
    );
    assert!(
        fork.get_value(&lane_state("main").address, &background_context())
            .await
            .expect("lane state")
            .is_none(),
    );
    assert!(
        fork.get_value(&lane_config("main").address, &background_context())
            .await
            .expect("lane config")
            .is_none(),
    );
    let stats = fork.get_stats(&background_context()).await.expect("stats");
    assert_eq!(stats.message_count, 2);
    assert_eq!(
        serde_json::to_value(stats.usage).expect("usage wire"),
        zero_usage()
    );
}

#[tokio::test]
async fn forks_a_closed_source_into_a_complete_v4_destination_without_rewriting_it() {
    let mut suite = Suite::new();
    let records = fork_fixture_records();
    let (path, content) = suite.write_legacy_v3_fixture(&records, None).await;
    let metadata = suite.discover().await;

    let (fork, fork_metadata) = suite
        .repo
        .fork(
            &metadata,
            &ForkOptions::Tree {
                id: Some("closed-fork".to_owned()),
            },
            &background_context(),
        )
        .await
        .expect("fork");

    assert_eq!(
        std::fs::read_to_string(&path).expect("source read"),
        content
    );
    assert_eq!(fork_metadata.id, "closed-fork");
    let header_line: Vec<String> = suite
        .env
        .read_text_lines(&fork_metadata.path, None, &background_context())
        .await
        .expect("fork header read");
    let header: JsonValue =
        serde_json::from_str(header_line.first().expect("header line")).expect("header parse");
    assert_eq!(header["v"], json!(4));
    assert_eq!(header["kind"], json!("header"));
    assert_eq!(header["id"], json!("closed-fork"));
    assert_eq!(header["storageVersion"], json!(JSONL_STORAGE_VERSION));
    assert_eq!(header["parentSessionId"], json!("legacy"));
    expect_forked_state(fork.as_ref()).await;

    fork.close(&background_context()).await.expect("fork close");
    let destination = JsonlStorage::open(
        &pi_agent_core::harness::session::jsonl::types::JsonlStorageOptions {
            file_system: suite.env.clone(),
            path: fork_metadata.path.clone(),
            now: Some(Arc::new(move || NOW)),
        },
        &background_context(),
    )
    .await
    .expect("destination open");
    let usage_rows = scan_usage_asc(&destination).await;
    assert_eq!(usage_rows, []);
    destination
        .close(&background_context())
        .await
        .expect("destination close");
    assert_eq!(
        std::fs::read_to_string(&path).expect("source read"),
        content
    );
}

#[tokio::test]
async fn forks_a_configured_closed_source_at_its_main_tip_when_entry_id_is_omitted() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                model_change_record("model", None, NOW + 1_000, "anthropic", "claude-sonnet-4-5"),
                thinking_level_record("thinking", Some("model"), NOW + 2_000, "high"),
                message_record("tip", Some("thinking"), NOW + 3_000, "fork me"),
            ],
            None,
        )
        .await;
    let metadata = suite.discover().await;

    let (fork, _) = suite
        .repo
        .fork(
            &metadata,
            &ForkOptions::Branch {
                branch: "main".to_owned(),
                entry_id: None,
                position: None,
                id: Some("branch-fork".to_owned()),
            },
            &background_context(),
        )
        .await
        .expect("fork");
    let entries = Suite::entries_asc(fork.as_ref()).await;
    assert_eq!(entries.len(), 1);
    assert_eq!(
        Suite::main_tip(fork.as_ref()).await,
        Some(entries[0].id().to_owned())
    );
    let config = fork
        .get_value(&lane_config("main").address, &background_context())
        .await
        .expect("lane config")
        .expect("stored config");
    assert_eq!(
        config.value,
        json!({
            "model": { "provider": "anthropic", "modelId": "claude-sonnet-4-5" },
            "thinkingLevel": "high",
            "activeToolNames": [],
        }),
    );
    let state = fork
        .get_value(&lane_state("main").address, &background_context())
        .await
        .expect("lane state")
        .expect("stored state");
    assert_eq!(
        state.value,
        json!({ "currentOperationId": null, "lastOperationId": null, "inbox": [] }),
    );
    fork.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn forks_a_configured_closed_source_at_an_original_legacy_entry_id() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                model_change_record("model", None, NOW + 1_000, "anthropic", "claude-sonnet-4-5"),
                thinking_level_record("thinking", Some("model"), NOW + 2_000, "high"),
                message_record("message-1", Some("thinking"), NOW + 3_000, "fork me"),
                assistant_record(
                    "message-2",
                    Some("message-1"),
                    NOW + 4_000,
                    "forked",
                    &simple_usage(1),
                ),
                message_record("message-3", Some("message-2"), NOW + 5_000, "fork me"),
            ],
            None,
        )
        .await;
    let metadata = suite.discover().await;

    let (fork, _) = suite
        .repo
        .fork(
            &metadata,
            &ForkOptions::Branch {
                branch: "main".to_owned(),
                entry_id: Some("message-2".to_owned()),
                position: None,
                id: Some("entry-fork".to_owned()),
            },
            &background_context(),
        )
        .await
        .expect("fork");
    let entries = Suite::entries_asc(fork.as_ref()).await;
    assert_eq!(entries.len(), 2);
    Suite::assert_child_parent(&entries[1], &entries[0]);
    assert_eq!(
        serde_json::to_value(&entry_message(&entries[1]).message).expect("message wire"),
        assistant_message("forked", NOW + 4_000, &simple_usage(1)),
    );
    assert_main_tip_is(fork.as_ref(), entries[1].id()).await;
    fork.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn rejects_an_open_v3_source_until_a_non_empty_commit_persists_its_format_4_ids() {
    let mut suite = Suite::new();
    let (path, content) = suite
        .write_legacy_v3_fixture(&fork_fixture_records(), None)
        .await;
    let metadata = suite.discover().await;
    let source = suite.open(&metadata).await;
    let source_entries = Suite::entries_asc(source.as_ref()).await;

    let forked = suite
        .repo
        .fork(
            &metadata,
            &ForkOptions::Tree {
                id: Some("open-fork".to_owned()),
            },
            &background_context(),
        )
        .await;
    assert!(
        forked
            .err()
            .expect("fork rejected")
            .to_string()
            .contains("Cannot fork an open legacy v3 JSONL session")
    );
    assert_eq!(
        std::fs::read_to_string(&path).expect("source read"),
        content
    );

    source
        .set_name(Some("Upgraded source".to_owned()), &background_context())
        .await
        .expect("set name");
    let (fork, _) = suite
        .repo
        .fork(
            &metadata,
            &ForkOptions::Tree {
                id: Some("open-fork".to_owned()),
            },
            &background_context(),
        )
        .await
        .expect("fork after upgrade");

    assert_eq!(Suite::entries_asc(fork.as_ref()).await, source_entries);
    assert_eq!(
        fork.get_name(&background_context()).await.expect("name"),
        Some("Upgraded source".to_owned()),
    );
    source
        .close(&background_context())
        .await
        .expect("source close");
    fork.close(&background_context()).await.expect("fork close");
}

#[tokio::test]
async fn opens_an_empty_legacy_session_with_a_data_only_main_branch() {
    let mut suite = Suite::new();
    suite.write_legacy_v3_fixture(&[], None).await;
    let metadata = suite.discover().await;
    let session = suite.open(&metadata).await;

    assert_eq!(
        session
            .find_entries(None, &background_context())
            .await
            .expect("entries"),
        [],
    );
    assert_eq!(Suite::main_tip(session.as_ref()).await, None);
    assert_lane_values_absent(session.as_ref()).await;
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn reports_embedded_legacy_usage_without_creating_usage_rows_or_rewriting_the_source() {
    let mut suite = Suite::new();
    let (path, content) = suite
        .write_legacy_v3_fixture(
            &[
                serde_json::json!({
                    "type": "message",
                    "id": "assistant",
                    "parentId": null,
                    "timestamp": iso(NOW + 1_000),
                    "message": assistant_message("answer", NOW + 1_000, &usage(1)),
                }),
                serde_json::json!({
                    "type": "message",
                    "id": "tool-result",
                    "parentId": "assistant",
                    "timestamp": iso(NOW + 2_000),
                    "message": tool_result_message(NOW + 2_000, &usage(10)),
                }),
                serde_json::json!({
                    "type": "compaction",
                    "id": "compaction",
                    "parentId": "tool-result",
                    "timestamp": iso(NOW + 3_000),
                    "summary": "Earlier context",
                    "firstKeptEntryId": "assistant",
                    "tokensBefore": 1_000,
                    "usage": usage(100),
                }),
                serde_json::json!({
                    "type": "branch_summary",
                    "id": "branch-summary",
                    "parentId": "compaction",
                    "timestamp": iso(NOW + 4_000),
                    "fromId": "assistant",
                    "summary": "Abandoned branch",
                    "usage": usage(1_000),
                }),
            ],
            None,
        )
        .await;

    let storage = JsonlStorage::open(
        &pi_agent_core::harness::session::jsonl::types::JsonlStorageOptions {
            file_system: suite.env.clone(),
            path: path.clone(),
            now: Some(Arc::new(move || NOW)),
        },
        &background_context(),
    )
    .await
    .expect("storage open");

    let stats = storage
        .get_stats(&background_context())
        .await
        .expect("stats");
    assert_eq!(stats.message_count, 2);
    assert_eq!(
        serde_json::to_value(stats.usage).expect("usage wire"),
        usage(1_111)
    );
    let usage_rows = scan_usage_asc(&storage).await;
    assert_eq!(usage_rows, []);
    storage.close(&background_context()).await.expect("close");
    assert_eq!(
        std::fs::read_to_string(&path).expect("source read"),
        content
    );
}

fn usage_fixture_records() -> Vec<JsonValue> {
    vec![assistant_record(
        "assistant",
        None,
        NOW + 1_000,
        "imported answer",
        &simple_usage(1),
    )]
}

#[tokio::test]
async fn writes_one_zero_valued_usage_adjustment_when_converting_a_session_without_imported_usage()
{
    let mut suite = Suite::new();
    let (path, _content) = suite.write_legacy_v3_fixture(&[], None).await;
    let storage = JsonlStorage::open(
        &pi_agent_core::harness::session::jsonl::types::JsonlStorageOptions {
            file_system: suite.env.clone(),
            path,
            now: Some(Arc::new(move || NOW)),
        },
        &background_context(),
    )
    .await
    .expect("storage open");

    let name_write =
        set_value(&session_name(), "Converted session".to_owned()).expect("name write");
    storage
        .commit(vec![Write::ValueSet(name_write)], &background_context())
        .await
        .expect("convert commit");

    let usage_rows = storage
        .scan_usage(
            &pi_agent_core::harness::session::types::UsageScan {
                order: Some(EntryScanOrder::Asc),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("usage rows");
    assert_eq!(usage_rows.len(), 1);
    assert_eq!(
        serde_json::to_value(usage_rows[0].usage).expect("usage wire"),
        zero_usage()
    );
    assert!(usage_rows[0].adjustment);
    assert_eq!(
        usage_rows[0].details,
        Some(serde_json::json!({ "source": "v3-import" }))
    );
    assert!(usage_rows[0].entry_id.is_none());
    storage.close(&background_context()).await.expect("close");
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the conversion assertions walk the converted file line by line, upstream's single test body"
)]
async fn converts_to_v4_and_preserves_the_first_caller_transaction_with_one_usage_adjustment() {
    let mut suite = Suite::new();
    let (path, _content) = suite
        .write_legacy_v3_fixture(&usage_fixture_records(), None)
        .await;
    let options = pi_agent_core::harness::session::jsonl::types::JsonlStorageOptions {
        file_system: suite.env.clone(),
        path: path.clone(),
        now: Some(Arc::new(move || NOW)),
    };
    let storage = JsonlStorage::open(&options, &background_context())
        .await
        .expect("storage open");
    let imported_entries = storage
        .scan_entries(
            &EntryScan {
                order: Some(EntryScanOrder::Asc),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("imported entries");
    let stats_before = storage
        .get_stats(&background_context())
        .await
        .expect("stats before");

    let name_write =
        set_value(&session_name(), "Converted session".to_owned()).expect("name write");
    let committed = storage
        .commit(
            vec![Write::ValueSet(name_write.clone())],
            &background_context(),
        )
        .await
        .expect("convert commit");

    assert_eq!(committed.seqs.len(), 1);
    assert_eq!(committed.first_seq, committed.seqs[0]);
    assert_eq!(
        storage
            .get_value(&session_name().address, &background_context())
            .await
            .expect("stored name")
            .expect("stored name value")
            .seq,
        committed.seqs[0],
    );
    let usage_rows = storage
        .scan_usage(
            &pi_agent_core::harness::session::types::UsageScan {
                order: Some(EntryScanOrder::Asc),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("usage rows");
    assert_eq!(usage_rows.len(), 1);
    let adjustment = &usage_rows[0];
    assert_eq!(
        serde_json::to_value(adjustment.usage).expect("usage wire"),
        simple_usage(1),
    );
    assert!(adjustment.adjustment);
    assert_eq!(
        adjustment.details,
        Some(serde_json::json!({ "source": "v3-import" }))
    );
    assert!(adjustment.entry_id.is_none());
    assert!(!committed.seqs.contains(&adjustment.seq));
    assert_eq!(committed.stats, stats_before);
    assert_eq!(
        storage
            .get_stats(&background_context())
            .await
            .expect("stats"),
        stats_before
    );

    let converted: Vec<String> = std::fs::read_to_string(&path)
        .expect("converted file read")
        .trim_end()
        .split('\n')
        .map(str::to_owned)
        .collect();
    let header: JsonValue =
        serde_json::from_str(converted.first().expect("header line")).expect("header parse");
    assert_eq!(header["v"], json!(JSONL_FORMAT_VERSION));
    assert_eq!(header["kind"], json!("header"));
    assert_eq!(header["id"], json!("legacy"));
    assert_eq!(header["storageVersion"], json!(JSONL_STORAGE_VERSION));
    assert_eq!(header["createdAt"], json!(NOW));
    assert_eq!(header["cwd"], json!("/workspace"));
    let transaction: JsonValue = serde_json::from_str(converted.last().expect("transaction line"))
        .expect("transaction parse");
    let expected_adjustment = serde_json::json!({
        "kind": "usage",
        "id": adjustment.id,
        "usage": serde_json::to_value(adjustment.usage).expect("usage wire"),
        "adjustment": true,
        "details": { "source": "v3-import" },
        "seq": adjustment.seq,
    });
    let expected_value = serde_json::json!({
        "kind": "value",
        "op": "set",
        "seq": committed.seqs[0],
        "namespace": session_name().address.namespace,
        "key": session_name().address.key,
        "value": "Converted session",
    });
    assert_eq!(
        transaction,
        serde_json::json!([expected_adjustment, expected_value])
    );
    storage.close(&background_context()).await.expect("close");

    let reopened = JsonlStorage::open(&options, &background_context())
        .await
        .expect("reopen");
    let scanned = reopened
        .scan_entries(
            &EntryScan {
                order: Some(EntryScanOrder::Asc),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("scanned entries");
    assert_eq!(scanned, imported_entries);
    let imported_entry = imported_entries.first().expect("imported entry");
    assert_eq!(
        reopened
            .get_value(&branch_tip("main").address, &background_context())
            .await
            .expect("branch tip")
            .expect("stored tip")
            .value,
        serde_json::json!(imported_entry.id()),
    );
    assert!(
        reopened
            .get_value(&lane_state("main").address, &background_context())
            .await
            .expect("lane state")
            .is_none(),
    );
    assert_eq!(
        reopened
            .get_value(&session_name().address, &background_context())
            .await
            .expect("session name")
            .expect("stored name")
            .value,
        serde_json::json!("Converted session"),
    );
    assert_eq!(
        reopened
            .scan_usage(
                &pi_agent_core::harness::session::types::UsageScan {
                    order: Some(EntryScanOrder::Asc),
                    ..Default::default()
                },
                &background_context(),
            )
            .await
            .expect("usage rows"),
        usage_rows,
    );
    assert_eq!(
        reopened
            .get_stats(&background_context())
            .await
            .expect("stats"),
        stats_before
    );
    reopened.close(&background_context()).await.expect("close");
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the failure assertions walk every read surface, upstream's single test body"
)]
async fn leaves_the_v3_source_and_live_state_unchanged_when_atomic_publication_fails() {
    let mut suite = Suite::new();
    let (path, content) = suite
        .write_legacy_v3_fixture(&usage_fixture_records(), None)
        .await;
    let options = pi_agent_core::harness::session::jsonl::types::JsonlStorageOptions {
        file_system: suite.env.clone(),
        path: path.clone(),
        now: Some(Arc::new(move || NOW)),
    };
    let storage = JsonlStorage::open(&options, &background_context())
        .await
        .expect("storage open");
    let entries_before = storage
        .scan_entries(
            &EntryScan {
                order: Some(EntryScanOrder::Asc),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("entries before");
    let leaf_before = storage
        .get_value(&branch_tip("main").address, &background_context())
        .await
        .expect("leaf before")
        .expect("normalized main Branch tip");
    let lane_state_before = storage
        .get_value(&lane_state("main").address, &background_context())
        .await
        .expect("lane state before");
    let name_before = storage
        .get_value(&session_name().address, &background_context())
        .await
        .expect("name before");
    let stats_before = storage
        .get_stats(&background_context())
        .await
        .expect("stats before");
    let usage_before = scan_usage_asc(&storage).await;

    suite.file_system.set_fail_rename(true);
    let name_write =
        set_value(&session_name(), "Converted session".to_owned()).expect("name write");
    let committed = storage
        .commit(vec![Write::ValueSet(name_write)], &background_context())
        .await;
    assert!(
        committed
            .expect_err("publication failure")
            .to_string()
            .contains(&format!("Failed to publish JSONL storage {path}"))
    );
    suite.file_system.set_fail_rename(false);

    assert_eq!(
        std::fs::read_to_string(&path).expect("source read"),
        content
    );
    let entries_after = storage
        .scan_entries(
            &EntryScan {
                order: Some(EntryScanOrder::Asc),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("entries after");
    assert_eq!(entries_after, entries_before);
    assert_eq!(
        storage
            .get_value(&branch_tip("main").address, &background_context())
            .await
            .expect("leaf after"),
        Some(leaf_before.clone()),
    );
    assert_eq!(
        storage
            .get_value(&lane_state("main").address, &background_context())
            .await
            .expect("lane state after"),
        lane_state_before,
    );
    assert_eq!(
        storage
            .get_value(&session_name().address, &background_context())
            .await
            .expect("name after"),
        name_before,
    );
    assert_eq!(
        storage
            .get_stats(&background_context())
            .await
            .expect("stats"),
        stats_before
    );
    assert_eq!(
        storage
            .scan_usage(
                &pi_agent_core::harness::session::types::UsageScan::default(),
                &background_context(),
            )
            .await
            .expect("usage after"),
        usage_before,
    );

    let name_write =
        set_value(&session_name(), "Converted session".to_owned()).expect("name write");
    let committed = storage
        .commit(vec![Write::ValueSet(name_write)], &background_context())
        .await
        .expect("retry commit");

    assert_eq!(committed.first_seq, leaf_before.seq + 2);
    assert_eq!(committed.seqs, [committed.first_seq]);
    let stored = storage
        .get_value(&session_name().address, &background_context())
        .await
        .expect("stored name")
        .expect("stored name value");
    assert_eq!(stored.seq, committed.first_seq);
    assert_eq!(stored.value, serde_json::json!("Converted session"));
    let usage_rows = storage
        .scan_usage(
            &pi_agent_core::harness::session::types::UsageScan {
                order: Some(EntryScanOrder::Asc),
                ..Default::default()
            },
            &background_context(),
        )
        .await
        .expect("usage rows");
    assert_eq!(usage_rows.len(), 1);
    assert!(usage_rows[0].adjustment);
    assert_eq!(
        usage_rows[0].details,
        Some(serde_json::json!({ "source": "v3-import" }))
    );
    assert_eq!(
        storage
            .get_stats(&background_context())
            .await
            .expect("stats"),
        stats_before
    );
    storage.close(&background_context()).await.expect("close");
}

fn configuration_changes() -> Vec<JsonValue> {
    vec![
        model_change_record(
            "model-change",
            Some("message-1"),
            NOW + 2_000,
            "anthropic",
            "claude-sonnet-4-5",
        ),
        thinking_level_record("thinking-change", Some("model-change"), NOW + 3_000, "high"),
        active_tools_record(
            "active-tools-change",
            Some("thinking-change"),
            NOW + 4_000,
            &["read", "bash"],
        ),
    ]
}

#[tokio::test]
async fn reparents_a_retained_child_through_configuration_changes() {
    let mut suite = Suite::new();
    let mut records = vec![message_record(
        "message-1",
        None,
        NOW + 1_000,
        FIRST_MESSAGE,
    )];
    records.extend(configuration_changes());
    records.push(message_record(
        "message-2",
        Some("active-tools-change"),
        NOW + 5_000,
        SECOND_MESSAGE,
    ));
    suite.write_legacy_v3_fixture(&records, None).await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 2).await;
    assert_eq!(entries[0].parent_id(), None);
    assert_eq!(
        serde_json::to_value(&entry_message(&entries[0]).message).expect("first message wire"),
        user_message(FIRST_MESSAGE, NOW + 1_000),
    );
    Suite::assert_child_parent(&entries[1], &entries[0]);
    assert_eq!(
        serde_json::to_value(&entry_message(&entries[1]).message).expect("second message wire"),
        user_message(SECOND_MESSAGE, NOW + 5_000),
    );
    assert!(entries[1].seq() > entries[0].seq());
    assert_main_tip_is(session.as_ref(), entries[1].id()).await;
    assert_lane_config(session.as_ref(), "claude-sonnet-4-5", &["read", "bash"]).await;
    assert_lane_state_fresh(session.as_ref()).await;
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn retains_only_configuration_changes_on_the_selected_physical_branch() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                message_record("root", None, NOW + 1_000, FIRST_MESSAGE),
                model_change_record(
                    "selected-model",
                    Some("root"),
                    NOW + 2_000,
                    "anthropic",
                    "selected",
                ),
                thinking_level_record(
                    "selected-thinking",
                    Some("selected-model"),
                    NOW + 3_000,
                    "high",
                ),
                model_change_record(
                    "abandoned-model",
                    Some("root"),
                    NOW + 4_000,
                    "openai",
                    "abandoned",
                ),
                message_record(
                    "selected-tip",
                    Some("selected-thinking"),
                    NOW + 5_000,
                    SECOND_MESSAGE,
                ),
            ],
            None,
        )
        .await;
    let metadata = suite.discover().await;
    let session = suite.open(&metadata).await;

    assert_lane_config(session.as_ref(), "selected", &[]).await;
    assert_lane_state_fresh(session.as_ref()).await;
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn leaves_main_data_only_for_missing_model() {
    leaves_main_data_only(vec![thinking_level_record(
        "thinking",
        Some("root"),
        NOW + 3_000,
        "high",
    )])
    .await;
}

#[tokio::test]
async fn leaves_main_data_only_for_missing_thinking_level() {
    leaves_main_data_only(vec![model_change_record(
        "model",
        Some("root"),
        NOW + 2_000,
        "anthropic",
        "selected",
    )])
    .await;
}

/// The it.each body, upstream's "leaves main data-only for $name".
async fn leaves_main_data_only(changes: Vec<JsonValue>) {
    let mut suite = Suite::new();
    let mut records = vec![message_record("root", None, NOW + 1_000, FIRST_MESSAGE)];
    let last_change_id = changes
        .last()
        .and_then(|change| change["id"].as_str())
        .map_or_else(|| "root".to_owned(), str::to_owned);
    records.extend(changes);
    records.push(message_record(
        "tip",
        Some(&last_change_id),
        NOW + 5_000,
        SECOND_MESSAGE,
    ));
    suite.write_legacy_v3_fixture(&records, None).await;
    let metadata = suite.discover().await;
    let session = suite.open(&metadata).await;

    assert!(
        session
            .get_value(&lane_config("main").address, &background_context())
            .await
            .expect("lane config")
            .is_none(),
    );
    assert!(
        session
            .get_value(&lane_state("main").address, &background_context())
            .await
            .expect("lane state")
            .is_none(),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn resolves_the_main_tip_through_trailing_configuration_changes() {
    let mut suite = Suite::new();
    let mut records = vec![message_record(
        "message-1",
        None,
        NOW + 1_000,
        FIRST_MESSAGE,
    )];
    records.extend(configuration_changes());
    suite.write_legacy_v3_fixture(&records, None).await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 1).await;
    assert_main_tip_is(session.as_ref(), entries[0].id()).await;
    assert_lane_config(session.as_ref(), "claude-sonnet-4-5", &["read", "bash"]).await;
    assert_lane_state_fresh(session.as_ref()).await;
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn imports_session_info_as_the_current_name_without_retaining_a_tree_entry() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[session_info_record(
                "session-info",
                None,
                NOW + 1_000,
                Some("Imported session"),
            )],
            None,
        )
        .await;
    let metadata = suite.discover().await;
    let session = suite.open(&metadata).await;

    assert_eq!(
        session.get_name(&background_context()).await.expect("name"),
        Some("Imported session".to_owned()),
    );
    assert_eq!(
        session
            .find_entries(None, &background_context())
            .await
            .expect("entries"),
        [],
    );
    assert_eq!(Suite::main_tip(session.as_ref()).await, None);
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn uses_the_latest_session_info_and_resolves_tree_structure_through_discarded_records() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                message_record("message-1", None, NOW + 1_000, FIRST_MESSAGE),
                session_info_record(
                    "session-info-1",
                    Some("message-1"),
                    NOW + 2_000,
                    Some("Earlier name"),
                ),
                message_record(
                    "message-2",
                    Some("session-info-1"),
                    NOW + 3_000,
                    SECOND_MESSAGE,
                ),
                session_info_record(
                    "session-info-2",
                    Some("message-2"),
                    NOW + 4_000,
                    Some("Latest name"),
                ),
            ],
            None,
        )
        .await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 2).await;
    Suite::assert_child_parent(&entries[1], &entries[0]);
    assert_eq!(
        session.get_name(&background_context()).await.expect("name"),
        Some("Latest name".to_owned()),
    );
    assert_main_tip_is(session.as_ref(), entries[1].id()).await;
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn clears_the_session_name_with_a_missing_name_field() {
    clears_session_name(None).await;
}

#[tokio::test]
async fn clears_the_session_name_with_an_empty_name_field() {
    clears_session_name(Some("")).await;
}

/// The it.each body, upstream's "clears the session name with $name".
async fn clears_session_name(name: Option<&str>) {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                session_info_record("session-info-1", None, NOW + 1_000, Some("Earlier name")),
                session_info_record("session-info-2", Some("session-info-1"), NOW + 2_000, name),
            ],
            None,
        )
        .await;
    let metadata = suite.discover().await;
    let session = suite.open(&metadata).await;

    assert!(
        session
            .get_value(&session_name().address, &background_context())
            .await
            .expect("session name")
            .is_none(),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn imports_a_label_for_its_remapped_entry_without_retaining_a_tree_node() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                message_record("message-1", None, NOW + 1_000, "label me"),
                label_record(
                    "label-1",
                    Some("message-1"),
                    NOW + 2_000,
                    "message-1",
                    Some("Important"),
                ),
            ],
            None,
        )
        .await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 1).await;
    assert_eq!(
        session
            .get_value(&entry_label(entries[0].id()).address, &background_context())
            .await
            .expect("entry label")
            .expect("stored label")
            .value,
        serde_json::json!("Important"),
    );
    assert_main_tip_is(session.as_ref(), entries[0].id()).await;
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn skips_a_label_whose_target_has_no_retained_ancestor() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                session_info_record("root-session-info", None, NOW + 1_000, None),
                label_record(
                    "root-label",
                    Some("root-session-info"),
                    NOW + 2_000,
                    "root-session-info",
                    Some("Skipped label"),
                ),
                message_record(
                    "message-1",
                    Some("root-label"),
                    NOW + 3_000,
                    "retained message",
                ),
            ],
            None,
        )
        .await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 1).await;
    assert_eq!(entries[0].parent_id(), None);
    assert!(
        session
            .get_label(entries[0].id(), &background_context())
            .await
            .expect("label")
            .is_none(),
    );
    assert_main_tip_is(session.as_ref(), entries[0].id()).await;
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn uses_the_latest_label_after_remapping_discarded_targets() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                message_record("message-1", None, NOW + 1_000, FIRST_MESSAGE),
                session_info_record("session-info", Some("message-1"), NOW + 2_000, None),
                label_record(
                    "label-1",
                    Some("session-info"),
                    NOW + 3_000,
                    "session-info",
                    Some("Earlier label"),
                ),
                label_record(
                    "label-2",
                    Some("label-1"),
                    NOW + 4_000,
                    "message-1",
                    Some("Latest label"),
                ),
                message_record("message-2", Some("label-2"), NOW + 5_000, SECOND_MESSAGE),
            ],
            None,
        )
        .await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 2).await;
    Suite::assert_child_parent(&entries[1], &entries[0]);
    assert_eq!(
        stored_label(session.as_ref(), entries[0].id()).await,
        Some("Latest label".to_owned())
    );
    assert_main_tip_is(session.as_ref(), entries[1].id()).await;
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn clears_a_label_with_a_missing_label_field() {
    clears_label(None).await;
}

#[tokio::test]
async fn clears_a_label_with_an_empty_label_field() {
    clears_label(Some("")).await;
}

/// The it.each body, upstream's "clears a label with $label".
async fn clears_label(label: Option<&str>) {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                message_record("message-1", None, NOW + 1_000, "clear my label"),
                label_record(
                    "label-1",
                    Some("message-1"),
                    NOW + 2_000,
                    "message-1",
                    Some("Earlier label"),
                ),
                label_record("label-2", Some("label-1"), NOW + 3_000, "message-1", label),
            ],
            None,
        )
        .await;
    let metadata = suite.discover().await;
    let session = suite.open(&metadata).await;

    let entries = session
        .find_entries(None, &background_context())
        .await
        .expect("entries");
    assert_eq!(entries.len(), 1);
    assert!(
        session
            .get_label(entries[0].id(), &background_context())
            .await
            .expect("label")
            .is_none(),
    );
    assert_main_tip_is(session.as_ref(), entries[0].id()).await;
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn imports_a_custom_entry_without_rewriting_opaque_data_references() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                message_record("message-1", None, NOW + 1_000, FIRST_MESSAGE),
                json!({
                    "type": "custom",
                    "id": "custom-1",
                    "parentId": "message-1",
                    "timestamp": iso(NOW + 2_000),
                    "customType": "checkpoint",
                    "data": { "legacyReference": "message-1", "nested": { "legacyReference": "custom-1" } },
                }),
            ],
            None,
        )
        .await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 2).await;
    assert_eq!(entries[0].seq(), 1);
    let custom = &entries[1];
    let custom_body = entry_custom(custom);
    assert_eq!(custom.custom_type(), Some("checkpoint"));
    Suite::assert_child_parent(custom, &entries[0]);
    assert_eq!(custom.seq(), 2);
    assert_eq!(custom.timestamp(), NOW + 2_000);
    assert_eq!(
        custom_body.data,
        Some(serde_json::json!({
            "legacyReference": "message-1",
            "nested": { "legacyReference": "custom-1" },
        })),
    );
    assert!(is_uuidv7(custom.id()));
    assert_eq!(uuid_timestamp(custom.id()).cast_signed(), NOW + 2_000);
    assert_eq!(
        Suite::main_tip(session.as_ref()).await,
        Some(custom.id().to_owned())
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn imports_a_custom_message_as_a_current_message_entry() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                message_record("message-1", None, NOW + 1_000, FIRST_MESSAGE),
                json!({
                    "type": "custom_message",
                    "id": "custom-message-1",
                    "parentId": "message-1",
                    "timestamp": iso(NOW + 2_000),
                    "customType": "status",
                    "content": [{ "type": "text", "text": "legacy custom message" }],
                    "details": { "status": "complete" },
                    "display": false,
                }),
            ],
            None,
        )
        .await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 2).await;
    assert_eq!(entries[0].seq(), 1);
    let custom_message = &entries[1];
    assert_eq!(
        serde_json::to_value(&entry_message(custom_message).message).expect("message wire"),
        json!({
            "role": "custom",
            "customType": "status",
            "content": [{ "type": "text", "text": "legacy custom message" }],
            "details": { "status": "complete" },
            "display": false,
            "timestamp": NOW + 2_000,
        }),
    );
    Suite::assert_child_parent(custom_message, &entries[0]);
    assert_eq!(custom_message.seq(), 2);
    assert_eq!(custom_message.timestamp(), NOW + 2_000);
    assert!(is_uuidv7(custom_message.id()));
    assert_eq!(
        uuid_timestamp(custom_message.id()).cast_signed(),
        NOW + 2_000
    );
    assert_eq!(
        Suite::main_tip(session.as_ref()).await,
        Some(custom_message.id().to_owned())
    );
    assert_eq!(
        session
            .get_stats(&background_context())
            .await
            .expect("stats")
            .message_count,
        2,
    );
    session.close(&background_context()).await.expect("close");
}

fn is_uuidv7(id: &str) -> bool {
    let hex: String = id.chars().filter(|c| *c != '-').collect();
    id.len() == 36
        && id.chars().filter(|c| *c == '-').count() == 4
        && [8, 13, 18, 23]
            .iter()
            .all(|dash| id.as_bytes()[*dash] == b'-')
        && hex.len() == 32
        && hex.chars().all(|c| c.is_ascii_hexdigit())
        && hex.as_bytes()[12] == b'7'
        && matches!(hex.as_bytes()[16], b'8' | b'9' | b'a' | b'b')
}

fn branch_summary_fixture(from_hook: Option<bool>) -> Vec<JsonValue> {
    let mut summary = branch_summary_record(
        "summary",
        Some("branch-point"),
        NOW + 3_000,
        "branch-point",
        "Summary of the abandoned branch",
    );
    summary["details"] = json!({ "reason": "navigation" });
    summary["usage"] = simple_usage(1);
    if let Some(from_hook) = from_hook {
        summary["fromHook"] = json!(from_hook);
    }
    vec![
        message_record("branch-point", None, NOW + 1_000, "Try the first approach"),
        assistant_record(
            "abandoned-response",
            Some("branch-point"),
            NOW + 2_000,
            "Implemented the first approach",
            &simple_usage(2),
        ),
        summary,
    ]
}

#[tokio::test]
async fn preserves_branch_summary_payload_and_remaps_references() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(&branch_summary_fixture(None), None)
        .await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 3).await;
    Suite::assert_child_parent(&entries[1], &entries[0]);
    let summary = &entries[2];
    let summary_body = entry_branch_summary(summary);
    Suite::assert_child_parent(summary, &entries[0]);
    assert_eq!(summary.seq(), 3);
    assert_eq!(summary.timestamp(), NOW + 3_000);
    assert_eq!(summary_body.from_id.as_deref(), Some(entries[0].id()));
    assert_eq!(summary_body.summary, "Summary of the abandoned branch");
    assert_eq!(
        summary_body.details,
        Some(serde_json::json!({ "reason": "navigation" }))
    );
    assert_eq!(
        serde_json::to_value(summary_body.usage).expect("usage wire"),
        simple_usage(1),
    );
    assert!(!summary_body.from_hook);
    assert!(is_uuidv7(summary.id()));
    assert_eq!(uuid_timestamp(summary.id()).cast_signed(), NOW + 3_000);
    assert_eq!(
        Suite::main_tip(session.as_ref()).await,
        Some(summary.id().to_owned())
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn preserves_an_explicit_branch_summary_from_hook_flag() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(&branch_summary_fixture(Some(true)), None)
        .await;
    let metadata = suite.discover().await;
    let session = suite.open(&metadata).await;

    let entries = Suite::entries_asc(session.as_ref()).await;
    assert!(entry_branch_summary(&entries[2]).from_hook);
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn normalizes_branch_summary_from_id_root_to_null() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[branch_summary_record(
                "summary",
                None,
                NOW + 3_000,
                "root",
                "Summary from the root",
            )],
            None,
        )
        .await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 1).await;
    assert!(entry_branch_summary(&entries[0]).from_id.is_none());
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn rejects_a_missing_branch_summary_from_id() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[branch_summary_record(
                "summary",
                None,
                NOW + 3_000,
                "missing-legacy-entry",
                "Summary from a missing source",
            )],
            None,
        )
        .await;
    let metadata = suite.discover().await;

    let opened = suite.repo.open(&metadata, &background_context()).await;
    assert!(
        opened
            .err()
            .expect("open rejected")
            .to_string()
            .contains("Missing legacy v3 entry reference: missing-legacy-entry")
    );
}

#[tokio::test]
async fn keeps_branch_summary_from_id_null_when_a_discarded_source_has_no_retained_ancestor() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                model_change_record(
                    "model-change",
                    None,
                    NOW + 1_000,
                    "anthropic",
                    "claude-sonnet-4-5",
                ),
                branch_summary_record(
                    "summary",
                    Some("model-change"),
                    NOW + 3_000,
                    "model-change",
                    "Summary from the root",
                ),
            ],
            None,
        )
        .await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 1).await;
    assert_eq!(entries[0].parent_id(), None);
    assert!(entry_branch_summary(&entries[0]).from_id.is_none());
    session.close(&background_context()).await.expect("close");
}

fn compaction_fixture(from_hook: Option<bool>) -> Vec<JsonValue> {
    let mut compaction = compaction_record(
        "compaction",
        Some("retained-message"),
        NOW + 3_000,
        "Summary of the earlier context",
        "retained-message",
        12_000,
    );
    compaction["details"] = json!({ "strategy": "default" });
    compaction["usage"] = simple_usage(100);
    if let Some(from_hook) = from_hook {
        compaction["fromHook"] = json!(from_hook);
    }
    vec![
        message_record("excluded-message", None, NOW + 1_000, "old context"),
        message_record(
            "retained-message",
            Some("excluded-message"),
            NOW + 2_000,
            "retain this context",
        ),
        compaction,
    ]
}

#[tokio::test]
async fn materializes_the_retained_tail_and_preserves_the_compaction_payload() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(&compaction_fixture(None), None)
        .await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 3).await;
    Suite::assert_child_parent(&entries[1], &entries[0]);
    assert_eq!(entries[1].seq(), 2);
    let compaction = &entries[2];
    let compaction_body = entry_compaction(compaction);
    Suite::assert_child_parent(compaction, &entries[1]);
    assert_eq!(compaction.seq(), 3);
    assert_eq!(compaction.timestamp(), NOW + 3_000);
    assert_eq!(compaction_body.summary, "Summary of the earlier context");
    assert_eq!(
        serde_json::to_value(&compaction_body.retained_tail).expect("tail wire"),
        serde_json::json!([user_message("retain this context", NOW + 2_000)]),
    );
    assert_eq!(compaction_body.tokens_before, 12_000);
    assert_eq!(
        compaction_body.details,
        Some(serde_json::json!({ "strategy": "default" }))
    );
    assert_eq!(
        serde_json::to_value(compaction_body.usage).expect("usage wire"),
        simple_usage(100),
    );
    assert!(!compaction_body.from_hook);
    assert!(is_uuidv7(compaction.id()));
    assert_eq!(uuid_timestamp(compaction.id()).cast_signed(), NOW + 3_000);
    assert_eq!(
        Suite::main_tip(session.as_ref()).await,
        Some(compaction.id().to_owned())
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn builds_the_retained_tail_from_the_compaction_branch_rather_than_physical_order() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                message_record("main-1", None, NOW + 10_000, "first main-branch message"),
                message_record(
                    "other-1",
                    Some("main-1"),
                    NOW + 11_000,
                    "first unrelated-branch message",
                ),
                message_record(
                    "main-2",
                    Some("main-1"),
                    NOW + 12_000,
                    "second main-branch message",
                ),
                message_record(
                    "other-2",
                    Some("other-1"),
                    NOW + 13_000,
                    "second unrelated-branch message",
                ),
                compaction_record(
                    "compaction",
                    Some("main-2"),
                    NOW + 14_000,
                    "Summary before the retained main branch",
                    "main-1",
                    8_000,
                ),
            ],
            None,
        )
        .await;
    let metadata = suite.discover().await;
    let session = suite.open(&metadata).await;

    let entries = Suite::entries_asc(session.as_ref()).await;
    let compaction = entries
        .iter()
        .find(|entry| matches!(entry, Entry::Compaction { .. }))
        .expect("compaction entry");
    assert_eq!(
        serde_json::to_value(&entry_compaction(compaction).retained_tail).expect("tail wire"),
        serde_json::json!([
            user_message("first main-branch message", NOW + 10_000),
            user_message("second main-branch message", NOW + 12_000),
        ]),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn projects_every_supported_legacy_node_in_a_retained_tail() {
    let mut suite = Suite::new();
    let ordinary_timestamp = NOW + 30_000;
    suite
        .write_legacy_v3_fixture(
            &[
                message_record(
                    "ordinary-message",
                    None,
                    ordinary_timestamp,
                    "ordinary retained message",
                ),
                json!({
                    "type": "custom_message",
                    "id": "custom-message",
                    "parentId": "ordinary-message",
                    "timestamp": iso(NOW + 31_000),
                    "customType": "notice",
                    "content": [{ "type": "text", "text": "custom retained message" }],
                    "details": { "status": "complete" },
                    "display": false,
                }),
                json!({
                    "type": "custom",
                    "id": "plain-custom",
                    "parentId": "custom-message",
                    "timestamp": iso(NOW + 32_000),
                    "customType": "checkpoint",
                    "data": { "ignored": true },
                }),
                branch_summary_record(
                    "branch-summary",
                    Some("plain-custom"),
                    NOW + 33_000,
                    "ordinary-message",
                    "Earlier branch work",
                ),
                compaction_record(
                    "older-compaction",
                    Some("branch-summary"),
                    NOW + 34_000,
                    "Older compacted context",
                    "ordinary-message",
                    4_000,
                ),
                compaction_record(
                    "final-compaction",
                    Some("older-compaction"),
                    NOW + 35_000,
                    "Final compacted context",
                    "ordinary-message",
                    8_000,
                ),
            ],
            None,
        )
        .await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 6).await;
    let ordinary = entries.first().expect("ordinary entry");
    assert_eq!(
        uuid_timestamp(ordinary.id()).cast_signed(),
        ordinary_timestamp
    );
    let final_compaction = entries.last().expect("final compaction");
    assert!(matches!(final_compaction, Entry::Compaction { .. }));
    assert_eq!(
        serde_json::to_value(&entry_compaction(final_compaction).retained_tail).expect("tail wire"),
        serde_json::json!([
            user_message("ordinary retained message", ordinary_timestamp),
            {
                "role": "custom",
                "customType": "notice",
                "content": [{ "type": "text", "text": "custom retained message" }],
                "details": { "status": "complete" },
                "display": false,
                "timestamp": NOW + 31_000,
            },
            {
                "role": "branchSummary",
                "summary": "Earlier branch work",
                "fromId": ordinary.id(),
                "timestamp": NOW + 33_000,
            },
            {
                "role": "compactionSummary",
                "summary": "Older compacted context",
                "tokensBefore": 4_000,
                "timestamp": NOW + 34_000,
            },
        ]),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn rejects_a_missing_first_kept_entry_id() {
    rejects_bad_first_kept_entry_id("missing-entry").await;
}

#[tokio::test]
async fn rejects_an_off_branch_first_kept_entry_id() {
    rejects_bad_first_kept_entry_id("other-branch").await;
}

/// The it.each body, upstream's "rejects a $boundary firstKeptEntryId".
async fn rejects_bad_first_kept_entry_id(first_kept_entry_id: &str) {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                message_record("root", None, NOW + 20_000, "old context"),
                message_record(
                    "main-branch",
                    Some("root"),
                    NOW + 21_000,
                    "retain this context",
                ),
                message_record("other-branch", Some("root"), NOW + 22_000, "old context"),
                compaction_record(
                    "compaction",
                    Some("main-branch"),
                    NOW + 23_000,
                    "Summary before the retained main branch",
                    first_kept_entry_id,
                    8_000,
                ),
            ],
            None,
        )
        .await;
    let metadata = suite.discover().await;

    let opened = suite.repo.open(&metadata, &background_context()).await;
    assert!(
        opened
            .err()
            .expect("open rejected")
            .to_string()
            .contains("firstKeptEntryId is not on its parent branch")
    );
}

#[tokio::test]
async fn preserves_an_explicit_compaction_from_hook_flag() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(&compaction_fixture(Some(true)), None)
        .await;
    let metadata = suite.discover().await;
    let session = suite.open(&metadata).await;

    let entries = Suite::entries_asc(session.as_ref()).await;
    let compaction = entries
        .iter()
        .find(|entry| matches!(entry, Entry::Compaction { .. }))
        .expect("compaction entry");
    assert!(entry_compaction(compaction).from_hook);
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn remaps_a_legacy_message_chain_and_exposes_it_through_current_apis() {
    let mut suite = Suite::new();
    suite
        .write_legacy_v3_fixture(
            &[
                message_record("message-1", None, NOW + 1_000, FIRST_MESSAGE),
                message_record("message-2", Some("message-1"), NOW + 2_000, SECOND_MESSAGE),
            ],
            None,
        )
        .await;
    let session = suite.open_discovered().await;
    let entries = Suite::imported_chain(session.as_ref(), 2).await;
    assert_eq!(entries[0].parent_id(), None);
    assert_eq!(entries[0].seq(), 1);
    assert_eq!(entries[0].timestamp(), NOW + 1_000);
    assert_eq!(
        serde_json::to_value(&entry_message(&entries[0]).message).expect("first message wire"),
        user_message(FIRST_MESSAGE, NOW + 1_000),
    );
    Suite::assert_child_parent(&entries[1], &entries[0]);
    assert_eq!(entries[1].seq(), 2);
    assert_eq!(entries[1].timestamp(), NOW + 2_000);
    assert_eq!(
        serde_json::to_value(&entry_message(&entries[1]).message).expect("second message wire"),
        user_message(SECOND_MESSAGE, NOW + 2_000),
    );
    assert_main_tip_is(session.as_ref(), entries[1].id()).await;
    let branch = session
        .branch("main", &background_context())
        .await
        .expect("branch")
        .expect("imported main Branch");
    let branch_entries = branch
        .find_entries(
            Some(&BranchScan {
                order: Some(BranchScanOrder::OldestFirst),
                ..Default::default()
            }),
            &background_context(),
        )
        .await
        .expect("branch entries");
    assert_eq!(
        branch_entries.iter().map(Entry::id).collect::<Vec<_>>(),
        [entries[0].id(), entries[1].id()],
    );
    assert_eq!(
        session
            .get_stats(&background_context())
            .await
            .expect("stats")
            .message_count,
        2,
    );
    session.close(&background_context()).await.expect("close");
}

const OPENING_TIMESTAMP: i64 = NOW + 1_234;

/// The opening fixture, upstream's "opening a legacy v3 message session"
/// `beforeEach`.
async fn opening_fixture() -> (Suite, Box<dyn Session>, String, String, i64) {
    let mut suite = Suite::new();
    // The record's timestamp (the remint id's clock) differs from the
    // message's own timestamp, upstream's opening fixture.
    let (path, content) = suite
        .write_legacy_v3_fixture(
            &[json!({
                "type": "message",
                "id": "message-1",
                "parentId": null,
                "timestamp": iso(OPENING_TIMESTAMP),
                "message": user_message("hello", NOW + 1_000),
            })],
            None,
        )
        .await;
    let before_mtime = suite
        .env
        .file_info(&path, &background_context())
        .await
        .expect("file info")
        .mtime_ms;
    let metadata = suite.discover().await;
    let session = suite.open(&metadata).await;
    (suite, session, path, content, before_mtime)
}

#[tokio::test]
async fn exposes_the_legacy_message_through_the_current_entry_api() {
    let (_suite, session, _path, _content, _mtime) = opening_fixture().await;
    let entries = Suite::entries_asc(session.as_ref()).await;
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].parent_id(), None);
    assert_eq!(entries[0].seq(), 1);
    assert_eq!(entries[0].timestamp(), OPENING_TIMESTAMP);
    assert_eq!(
        serde_json::to_value(&entry_message(&entries[0]).message).expect("message wire"),
        user_message("hello", NOW + 1_000),
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn remints_the_entry_id_as_a_uuidv7_with_the_legacy_timestamp() {
    let (_suite, session, _path, _content, _mtime) = opening_fixture().await;
    let entries = Suite::entries_asc(session.as_ref()).await;
    assert!(is_uuidv7(entries[0].id()));
    assert_eq!(
        uuid_timestamp(entries[0].id()).cast_signed(),
        OPENING_TIMESTAMP
    );
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn initializes_a_data_only_main_branch_at_the_imported_entry() {
    let (_suite, session, _path, _content, _mtime) = opening_fixture().await;
    let entries = Suite::entries_asc(session.as_ref()).await;
    assert_main_tip_is(session.as_ref(), entries[0].id()).await;
    assert_lane_values_absent(session.as_ref()).await;
    session.close(&background_context()).await.expect("close");
}

#[tokio::test]
async fn leaves_the_legacy_source_untouched_after_open_and_close() {
    let (suite, session, path, content, before_mtime) = opening_fixture().await;
    session.close(&background_context()).await.expect("close");

    assert_eq!(
        std::fs::read_to_string(&path).expect("source read"),
        content
    );
    assert_eq!(
        suite
            .env
            .file_info(&path, &background_context())
            .await
            .expect("file info")
            .mtime_ms,
        before_mtime,
    );
}

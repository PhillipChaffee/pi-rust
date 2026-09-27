//! The `JsonlStorage` persistence and torn-tail suite, ported 1:1 from
//! upstream `test/harness/jsonl-storage.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::commit::{insert_entry, insert_usage};
use pi_agent_core::harness::session::jsonl::storage::JsonlStorage;
use pi_agent_core::harness::session::jsonl::types::{JsonlStorageHeader, JsonlStorageOptions};
use pi_agent_core::harness::session::types::{NewEntry, Storage};
use pi_agent_core::harness::session::values::Write;
use pi_agent_core::harness::session::values::{
    ValueList, append_list, branch_tip, delete_list, session_name, set_value,
};
use pi_agent_core::harness::types::{FileContent, FileSystem};
use serde_json::json;

mod jsonl_common;
use jsonl_common::WrappedEnv;

const NOW: i64 = 1_700_000_000_000;

fn header(id: &str) -> JsonlStorageHeader {
    JsonlStorageHeader {
        v: pi_agent_core::harness::session::jsonl::types::JSONL_FORMAT_VERSION,
        kind: "header".to_owned(),
        id: id.to_owned(),
        storage_version: 1,
        created_at: NOW,
        cwd: "/workspace".to_owned(),
        ..Default::default()
    }
}

fn storage_options(file_system: &Arc<dyn FileSystem>, path: &str) -> JsonlStorageOptions {
    JsonlStorageOptions {
        file_system: file_system.clone(),
        path: path.to_owned(),
        now: Some(Arc::new(move || NOW)),
    }
}

fn user_entry(id: &str) -> Write {
    Write::Entry(Box::new(insert_entry(NewEntry::Message {
        id: id.to_owned(),
        parent_id: None,
        body: Box::new(pi_agent_core::harness::session::types::MessageEntry {
            message: serde_json::from_value(json!({
                "role": "user",
                "content": id,
                "timestamp": 1,
            }))
            .expect("user message wire"),
            terminate: None,
        }),
    })))
}

fn usage_write() -> Write {
    Write::Usage(insert_usage(
        pi_agent_core::harness::session::types::UsageWriteRow {
            id: "usage".to_owned(),
            usage: serde_json::from_value(json!({
                "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 3,
                "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
            }))
            .expect("usage wire"),
            entry_id: Some("root".to_owned()),
            adjustment: false,
            details: None,
        },
    ))
}

fn events() -> ValueList<String> {
    pi_agent_core::harness::session::values::list::<String>("test.events", "")
        .expect("list address")
}

#[tokio::test]
async fn replays_whole_list_deletion_without_resurrecting_earlier_appends() {
    let root = jsonl_common::TempRoot::new();
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let options = storage_options(&file_system, "list-delete.jsonl");
    let events = events();
    let storage = JsonlStorage::create(
        &options,
        header("list-delete"),
        Vec::new(),
        &background_context(),
    )
    .await
    .expect("create");
    storage
        .commit(
            vec![
                Write::ListAppend(append_list(&events, "first".to_owned()).expect("append write")),
                Write::ListAppend(append_list(&events, "second".to_owned()).expect("append write")),
            ],
            &background_context(),
        )
        .await
        .expect("first commit");
    storage
        .commit(
            vec![Write::ListDelete(delete_list(&events))],
            &background_context(),
        )
        .await
        .expect("delete commit");
    storage.close(&background_context()).await.expect("close");

    let reopened = JsonlStorage::open(&options, &background_context())
        .await
        .expect("reopen");
    assert_eq!(
        reopened
            .read_list(&events.address, None, &background_context())
            .await
            .expect("read list"),
        [],
    );
    let recreated = reopened
        .commit(
            vec![Write::ListAppend(
                append_list(&events, "after".to_owned()).expect("append write"),
            )],
            &background_context(),
        )
        .await
        .expect("recreated commit");
    assert_eq!(recreated.first_seq, 4);
    assert_eq!(
        reopened
            .read_list(&events.address, None, &background_context())
            .await
            .expect("read list"),
        [pi_agent_core::harness::session::values::ListElement {
            seq: 4,
            value: serde_json::json!("after"),
        }],
    );
    reopened.close(&background_context()).await.expect("close");
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the replay assertions walk the wire file line by line, upstream's single test body"
)]
async fn writes_one_line_per_transaction_and_replays_stamped_state() {
    let root = jsonl_common::TempRoot::new();
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let options = storage_options(&file_system, "session.jsonl");
    let storage = JsonlStorage::create(
        &options,
        header("round-trip"),
        Vec::new(),
        &background_context(),
    )
    .await
    .expect("create");
    let committed = storage
        .commit(
            vec![
                Write::Entry(Box::new(insert_entry(NewEntry::Message {
                    id: "root".to_owned(),
                    parent_id: None,
                    body: Box::new(pi_agent_core::harness::session::types::MessageEntry {
                        message: serde_json::from_value(json!({
                            "role": "user",
                            "content": "hello",
                            "timestamp": 1,
                        }))
                        .expect("user message wire"),
                        terminate: None,
                    }),
                }))),
                Write::ValueSet(
                    set_value(&branch_tip("main"), Some("root".to_owned()))
                        .expect("branch tip write"),
                ),
                usage_write(),
            ],
            &background_context(),
        )
        .await
        .expect("first commit");
    let name_write = set_value(&session_name(), "name".to_owned()).expect("name write");
    storage
        .commit(
            vec![Write::ValueSet(name_write.clone())],
            &background_context(),
        )
        .await
        .expect("second commit");

    let lines: Vec<String> =
        std::fs::read_to_string(std::path::Path::new(root.path()).join("session.jsonl"))
            .expect("file read")
            .trim_end()
            .split('\n')
            .map(str::to_owned)
            .collect();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&lines[0]).expect("header line"),
        serde_json::to_value(header("round-trip")).expect("header wire"),
    );
    let first_transaction: serde_json::Value =
        serde_json::from_str(&lines[1]).expect("transaction line");
    assert!(
        first_transaction.is_array() && first_transaction.as_array().expect("array").len() == 3
    );
    let second_transaction: serde_json::Value =
        serde_json::from_str(&lines[2]).expect("transaction line");
    assert!(!second_transaction.is_array());
    storage.close(&background_context()).await.expect("close");

    let reopened = JsonlStorage::open(&options, &background_context())
        .await
        .expect("reopen");
    let entries = reopened
        .get_entries(vec!["root".to_owned()], &background_context())
        .await
        .expect("entries");
    let entry = entries.get("root").expect("root entry");
    assert_eq!(
        serde_json::to_value(entry).expect("entry wire"),
        json!({
            "id": "root",
            "parentId": null,
            "type": "message",
            "message": { "role": "user", "content": "hello", "timestamp": 1 },
            "seq": committed.seqs[0],
            "timestamp": committed.timestamp,
        }),
    );
    let tip = reopened
        .get_value(&branch_tip("main").address, &background_context())
        .await
        .expect("branch tip value")
        .expect("stored tip");
    assert_eq!(
        (
            tip.namespace.as_str(),
            tip.key.as_str(),
            tip.value.clone(),
            tip.seq
        ),
        (
            "pi.branch.tip",
            "main",
            serde_json::json!("root"),
            committed.seqs[1]
        ),
    );
    let usage = reopened
        .scan_usage(
            &pi_agent_core::harness::session::types::UsageScan::default(),
            &background_context(),
        )
        .await
        .expect("usage scan");
    assert_eq!(
        usage
            .iter()
            .map(|row| (row.id.clone(), row.seq))
            .collect::<Vec<_>>(),
        [("usage".to_owned(), committed.seqs[2])],
    );
    let historical_stats: pi_agent_core::harness::session::types::SessionStats =
        serde_json::from_value(json!({
            "messageCount": 1,
            "usage": {
                "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 3,
                "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0 },
            },
        }))
        .expect("stats wire");
    assert_eq!(
        reopened
            .get_stats(&background_context())
            .await
            .expect("stats"),
        historical_stats
    );
    let next = reopened
        .commit(Vec::new(), &background_context())
        .await
        .expect("empty commit");
    assert_eq!(next.first_seq, 5);
    assert_eq!(next.stats, historical_stats);
    reopened.close(&background_context()).await.expect("close");
}

fn torn_entry_line(id: &str) -> String {
    format!(
        "{}",
        serde_json::json!({
            "kind": "entry",
            "id": id,
            "parentId": null,
            "type": "message",
            "message": { "role": "user", "content": id, "timestamp": 1 },
            "seq": 2,
            "timestamp": NOW,
        })
    )
}

#[tokio::test]
async fn discards_an_unterminated_final_object_line_and_truncates_before_admitting_writes() {
    let root = jsonl_common::TempRoot::new();
    let (file_system, options, prefix) = seed_at(&root).await;
    file_system
        .append_file(
            "session.jsonl",
            FileContent::Text(torn_entry_line("torn")),
            &background_context(),
        )
        .await
        .expect("torn append");

    let reopened = JsonlStorage::open(&options, &background_context())
        .await
        .expect("reopen");
    let entries = reopened
        .get_entries(
            vec!["kept".to_owned(), "torn".to_owned()],
            &background_context(),
        )
        .await
        .expect("entries");
    assert!(!entries.contains_key("torn"));
    assert_eq!(entries.get("kept").expect("kept entry").id(), "kept");
    assert_eq!(
        std::fs::read_to_string(std::path::Path::new(root.path()).join("session.jsonl"))
            .expect("file read"),
        prefix,
    );
    assert!(
        !std::path::Path::new(root.path())
            .join("session.jsonl.tmp")
            .exists()
    );

    let next = reopened
        .commit(vec![user_entry("after")], &background_context())
        .await
        .expect("after commit");
    assert_eq!(next.first_seq, 2);
    let entries = reopened
        .get_entries(vec!["after".to_owned()], &background_context())
        .await
        .expect("entries");
    assert_eq!(entries.get("after").expect("after entry").seq(), 2);
    reopened.close(&background_context()).await.expect("close");
}

/// The seed variant tied to a kept root, since the torn suite's fixtures
/// outlive the seed's temp guard.
async fn seed_at(
    root: &jsonl_common::TempRoot,
) -> (Arc<dyn FileSystem>, JsonlStorageOptions, String) {
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let options = storage_options(&file_system, "session.jsonl");
    let storage = JsonlStorage::create(&options, header("torn"), Vec::new(), &background_context())
        .await
        .expect("create");
    storage
        .commit(vec![user_entry("kept")], &background_context())
        .await
        .expect("seed commit");
    storage.close(&background_context()).await.expect("close");
    let prefix = std::fs::read_to_string(std::path::Path::new(root.path()).join("session.jsonl"))
        .expect("prefix read");
    (file_system, options, prefix)
}

#[tokio::test]
async fn discards_a_torn_array_line_wholly_including_list_elements() {
    let root = jsonl_common::TempRoot::new();
    let (file_system, options, prefix) = seed_at(&root).await;
    let events = events();
    let torn = serde_json::json!([
        serde_json::json!({
            "kind": "entry",
            "id": "torn-a",
            "parentId": null,
            "type": "message",
            "message": { "role": "user", "content": "torn-a", "timestamp": 1 },
            "seq": 2,
            "timestamp": NOW,
        }),
        serde_json::to_value(set_value(&session_name(), "lost".to_owned()).expect("name write"))
            .expect("wire"),
        serde_json::to_value(append_list(&events, "lost".to_owned()).expect("append write"))
            .expect("wire"),
    ]);
    let torn_line = torn.to_string();
    file_system
        .append_file(
            "session.jsonl",
            FileContent::Text(torn_line),
            &background_context(),
        )
        .await
        .expect("torn append");

    let reopened = JsonlStorage::open(&options, &background_context())
        .await
        .expect("reopen");
    let entries = reopened
        .get_entries(vec!["torn-a".to_owned()], &background_context())
        .await
        .expect("entries");
    assert!(!entries.contains_key("torn-a"));
    assert!(
        reopened
            .get_value(&session_name().address, &background_context())
            .await
            .expect("session name")
            .is_none(),
    );
    assert_eq!(
        reopened
            .read_list(&events.address, None, &background_context())
            .await
            .expect("read list"),
        [],
    );
    assert_eq!(
        std::fs::read_to_string(std::path::Path::new(root.path()).join("session.jsonl"))
            .expect("file read"),
        prefix,
    );
    reopened.close(&background_context()).await.expect("close");
}

async fn corrupted_open_rejects(
    root: &jsonl_common::TempRoot,
    corrupted: &str,
    expected_fragment: &str,
) {
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let options = storage_options(&file_system, "session.jsonl");
    std::fs::write(
        std::path::Path::new(root.path()).join("session.jsonl"),
        corrupted,
    )
    .expect("corrupt write");

    let opened = JsonlStorage::open(&options, &background_context()).await;
    let error = opened.expect_err("open rejected").to_string();
    assert!(
        error.contains(expected_fragment),
        "unexpected error: {error}"
    );
    assert_eq!(
        std::fs::read_to_string(std::path::Path::new(root.path()).join("session.jsonl"))
            .expect("file read"),
        corrupted,
        "the open must not rewrite the file",
    );
}

#[tokio::test]
async fn rejects_a_malformed_interior_line_without_rewriting() {
    let root = jsonl_common::TempRoot::new();
    let (_file_system, _options, prefix) = seed_at(&root).await;
    let name_write = set_value(&session_name(), "after".to_owned()).expect("name write");
    let corrupted = format!(
        "{prefix}not-json\n{}\n",
        serde_json::to_string(&Write::ValueSet(name_write)).expect("wire"),
    );
    corrupted_open_rejects(&root, &corrupted, "line 3").await;
}

#[tokio::test]
async fn rejects_the_unsupported_pre_wp01_scalar_record_spelling() {
    let root = jsonl_common::TempRoot::new();
    let (_file_system, _options, prefix) = seed_at(&root).await;
    let legacy_kind = ["reg", "ister"].join("");
    let corrupted = format!(
        "{prefix}{}\n",
        serde_json::json!({
            "kind": legacy_kind,
            "op": "set",
            "seq": 2,
            "namespace": "legacy.value",
            "key": "state",
            "value": true,
        })
    );
    corrupted_open_rejects(&root, &corrupted, "line 3").await;
}

#[tokio::test]
async fn rejects_a_complete_malformed_final_line_without_rewriting() {
    let root = jsonl_common::TempRoot::new();
    let (_file_system, _options, prefix) = seed_at(&root).await;
    corrupted_open_rejects(&root, &format!("{prefix}not-json\n"), "line 3").await;
}

#[tokio::test]
async fn rejects_a_complete_final_line_with_invalid_transaction_framing() {
    let root = jsonl_common::TempRoot::new();
    let (_file_system, _options, prefix) = seed_at(&root).await;
    let corrupted = format!(
        "{prefix}{}\n",
        serde_json::json!({ "kind": "nope", "seq": 2 })
    );
    corrupted_open_rejects(&root, &corrupted, "line 3").await;
}

#[tokio::test]
async fn rejects_an_unterminated_header() {
    let root = jsonl_common::TempRoot::new();
    let file_system: Arc<dyn FileSystem> = WrappedEnv::new(root.path().to_owned());
    let options = storage_options(&file_system, "session.jsonl");
    let header_wire = serde_json::to_string(&header("torn")).expect("header wire");
    std::fs::write(
        std::path::Path::new(root.path()).join("session.jsonl"),
        &header_wire[..header_wire.len() - 4],
    )
    .expect("header write");

    let opened = JsonlStorage::open(&options, &background_context()).await;
    assert!(
        opened
            .expect_err("open rejected")
            .to_string()
            .contains("missing header")
    );
}

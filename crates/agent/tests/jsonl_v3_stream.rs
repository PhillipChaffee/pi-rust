//! The streaming legacy-v3 normalization suite, ported 1:1 from upstream
//! `test/harness/jsonl-v3-stream.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. Upstream's async-generator
//! `collect` restates as one eager `writes` call per pass; the pass still
//! reads the header and each captured record exactly once, so the
//! observed line-read counts carry over.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod jsonl_common;
use jsonl_common::WrappedEnv;

use std::sync::Arc;

use pi_agent_core::harness::context::background_context;
use pi_agent_core::harness::session::jsonl::codec::{LegacyV3SessionHeader, format_iso8601};
use pi_agent_core::harness::session::jsonl::legacy_v3::LegacyV3Source;
use pi_agent_core::harness::types::{FileContent, FileSystem};
use serde_json::Value as JsonValue;

const NOW: i64 = 1_700_000_000_000;

fn header() -> LegacyV3SessionHeader {
    LegacyV3SessionHeader {
        kind: "session".to_owned(),
        version: 3,
        id: "legacy".to_owned(),
        timestamp: format_iso8601(NOW),
        cwd: "/workspace".to_owned(),
        parent_session: None,
    }
}

fn message(id: &str, parent_id: Option<&str>, content: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "message",
        "id": id,
        "parentId": parent_id,
        "timestamp": format_iso8601(NOW),
        "message": { "role": "user", "content": content, "timestamp": NOW },
    })
}

struct Fixture {
    _root: jsonl_common::TempRoot,
    path: String,
    file_system: Arc<WrappedEnv>,
    source: LegacyV3Source,
}

async fn fixture(records: &[serde_json::Value], suffix: &str, fixture_id: usize) -> Fixture {
    let root = jsonl_common::TempRoot::new();
    let file_system = WrappedEnv::new(root.path().to_owned());
    let path = format!("legacy-{fixture_id}.jsonl");
    let mut lines = vec![serde_json::to_string(&header()).expect("header wire")];
    lines.extend(records.iter().map(JsonValue::to_string));
    let content = format!("{}\n{}", lines.join("\n"), suffix);
    std::fs::write(std::path::Path::new(root.path()).join(&path), content).expect("fixture write");
    let env: Arc<dyn FileSystem> = file_system.clone();
    let source = LegacyV3Source::read(env, &path, &background_context())
        .await
        .expect("source read");
    Fixture {
        _root: root,
        path,
        file_system,
        source,
    }
}

async fn collect(
    source: &LegacyV3Source,
    selected: Option<&(dyn Fn(&str) -> bool + Send + Sync)>,
) -> Vec<pi_agent_core::harness::session::commit::CommittedWrite> {
    source
        .writes(&background_context(), selected)
        .await
        .expect("writes")
}

#[tokio::test]
async fn materializes_a_selected_compaction_with_its_branch_local_tail() {
    let fixture = fixture(
        &[
            message("before", None, "before"),
            serde_json::json!({ "type": "session_info", "id": "boundary", "parentId": "before", "timestamp": format_iso8601(NOW) }),
            message("kept", Some("boundary"), "Unicode é漢字"),
            message("other", Some("before"), "not on this branch"),
            serde_json::json!({
                "type": "compaction",
                "id": "selected",
                "parentId": "kept",
                "timestamp": format_iso8601(NOW),
                "summary": "selected",
                "firstKeptEntryId": "boundary",
                "tokensBefore": 20,
                "fromHook": true,
                "details": { "a": 2 },
            }),
        ],
        "",
        0,
    )
    .await;
    let selected_id = fixture
        .source
        .entry_structures()
        .expect("structures")
        .last()
        .expect("last structure")
        .id
        .clone();
    assert_eq!(
        collect(&fixture.source, Some(&|_id: &str| false)).await,
        fixture.source.values,
    );
    fixture.file_system.reset_line_reads();
    let writes = collect(&fixture.source, Some(&|id: &str| id == selected_id)).await;
    assert_eq!(writes.len(), 2);
    let pi_agent_core::harness::session::commit::CommittedWrite::Entry { entry } = &writes[0]
    else {
        panic!("expected an entry write");
    };
    let pi_agent_core::harness::session::types::Entry::Compaction { id, seq, body, .. } = entry
    else {
        panic!("expected a compaction entry");
    };
    assert_eq!(id, &selected_id);
    assert_eq!(*seq, 4);
    assert!(body.from_hook);
    assert_eq!(body.details, Some(serde_json::json!({ "a": 2 })));
    assert_eq!(
        serde_json::to_value(&body.retained_tail).expect("tail wire"),
        serde_json::json!([{ "role": "user", "content": "Unicode é漢字", "timestamp": NOW }]),
    );
    assert_eq!(fixture.file_system.line_reads(), 6);
}

#[tokio::test]
async fn reuses_an_earlier_cached_message_in_selected_compaction_tails_on_different_branches() {
    let fixture = fixture(
        &[
            message("root", None, "root"),
            message("left", Some("root"), "left"),
            serde_json::json!({
                "type": "compaction",
                "id": "left-compaction",
                "parentId": "left",
                "timestamp": format_iso8601(NOW),
                "summary": "left summary",
                "firstKeptEntryId": "root",
                "tokensBefore": 10,
            }),
            message("right", Some("root"), "right"),
            serde_json::json!({
                "type": "compaction",
                "id": "right-compaction",
                "parentId": "right",
                "timestamp": format_iso8601(NOW),
                "summary": "right summary",
                "firstKeptEntryId": "root",
                "tokensBefore": 20,
            }),
        ],
        "",
        1,
    )
    .await;
    let structures = fixture.source.entry_structures().expect("structures");
    let selected: std::collections::BTreeSet<String> = [2usize, 4]
        .into_iter()
        .map(|index| structures[index].id.clone())
        .collect();
    fixture.file_system.reset_line_reads();
    let writes = collect(&fixture.source, Some(&|id: &str| selected.contains(id))).await;
    assert_eq!(writes.len(), 3);
    let tails: Vec<Vec<pi_agent_core::types::AgentMessage>> = writes[..2]
        .iter()
        .map(|write| match write {
            pi_agent_core::harness::session::commit::CommittedWrite::Entry { entry } => match entry
            {
                pi_agent_core::harness::session::types::Entry::Compaction { body, .. } => {
                    body.retained_tail.clone()
                }
                _ => panic!("expected a compaction entry"),
            },
            _ => panic!("expected an entry write"),
        })
        .collect();
    assert_eq!(
        serde_json::to_value(&tails[0]).expect("tail wire"),
        serde_json::json!([
            { "role": "user", "content": "root", "timestamp": NOW },
            { "role": "user", "content": "left", "timestamp": NOW },
        ]),
    );
    assert_eq!(
        serde_json::to_value(&tails[1]).expect("tail wire"),
        serde_json::json!([
            { "role": "user", "content": "root", "timestamp": NOW },
            { "role": "user", "content": "right", "timestamp": NOW },
        ]),
    );
    assert_eq!(fixture.file_system.line_reads(), 6);
    let again = collect(&fixture.source, Some(&|id: &str| selected.contains(id))).await;
    assert_eq!(again, writes);
}

#[tokio::test]
async fn ignores_a_torn_tail_and_emits_only_the_captured_complete_records_after_later_appends() {
    let fixture = fixture(
        &[message("a", None, "a")],
        &message("torn", Some("a"), "torn").to_string(),
        2,
    )
    .await;
    let appended = format!("\n{}\n", message("later", Some("torn"), "later"));
    fixture
        .file_system
        .append_file(
            &fixture.path,
            FileContent::Text(appended),
            &background_context(),
        )
        .await
        .expect("append");
    fixture.file_system.reset_line_reads();
    let writes = collect(&fixture.source, None).await;
    assert_eq!(writes.len(), 2);
    assert_eq!(fixture.file_system.line_reads(), 2);
    assert_eq!(fixture.source.next_seq, 3);
}

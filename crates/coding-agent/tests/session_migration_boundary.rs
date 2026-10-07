//! The migration and projection boundary suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the v1 migration arms that
//! ride raw values, the exported migrate/compaction helpers, and the leaf
//! selection arms of the context projections.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
use pi_agent_core::types::AgentMessage;
use serde_json::json;

use pi_coding_agent::session_manager::{
    ByIdIndex, FileEntry, LeafId, SessionEntry, SessionManager, build_context_entries,
    build_session_path, get_latest_compaction_entry, load_entries_from_file,
    migrate_session_entries, parse_session_entries, session_entry_to_context_messages,
};

const TS: &str = "2026-01-01T00:00:00.000Z";

fn entry(wire: serde_json::Value) -> SessionEntry {
    match serde_json::from_value::<FileEntry>(wire).expect("typed entry") {
        FileEntry::Entry(entry) => entry,
        other => panic!("expected a typed entry, got {other:?}"),
    }
}

fn read_lines(path: &str) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .expect("read session file")
        .lines()
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect()
}

#[test]
fn the_v1_migration_assigns_raw_values_and_renames_hook_messages() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = format!("{}/v1.jsonl", dir.path().display());
    std::fs::write(
        &path,
        concat!(
            r#"{"type":"session","id":"v1"}"#,
            "\n",
            r#"{"type":"mystery","value":1}"#,
            "\n",
            r#"{"type":"compaction","summary":"s","firstKeptEntryIndex":1}"#,
            "\n",
            r#"{"type":"message","message":{"role":"hookMessage","content":123}}"#,
            "\n",
        ),
    )
    .expect("write v1 file");

    let session = SessionManager::open(&path, None, None).expect("open migrates and rewrites");
    assert_eq!(session.session_id(), "v1");
    let lines = read_lines(&path);

    let mystery = &lines[1];
    assert!(
        mystery["id"].is_string(),
        "the raw value gains a generated tree id"
    );
    assert_eq!(
        mystery["parentId"],
        serde_json::Value::Null,
        "the first entry parents at the root"
    );

    let compaction = &lines[2];
    assert_eq!(
        compaction["firstKeptEntryId"], mystery["id"],
        "the kept index resolves to the assigned id"
    );
    assert!(
        compaction.get("firstKeptEntryIndex").is_none(),
        "the index drops after the rename"
    );
    assert_eq!(
        compaction["parentId"], mystery["id"],
        "the raw compaction chains onto the raw entry"
    );

    let hook = &lines[3];
    assert_eq!(
        hook["message"]["role"], "custom",
        "the raw hookMessage renames on the way to v3"
    );
    assert!(hook["id"].is_string());
    assert_eq!(hook["parentId"], compaction["id"]);
}

#[test]
fn the_exported_helpers_migrate_raw_only_input_and_find_compactions() {
    let mut entries = parse_session_entries(r#"{"type":"mystery","value":1}"#);
    assert!(
        migrate_session_entries(&mut entries),
        "headerless input migrates as v1"
    );
    assert!(entries[0].entry_id().is_some(), "the raw value gains an id");
    assert_eq!(get_latest_compaction_entry(&entries), None);

    let with_compaction = parse_session_entries(concat!(
        r#"{"type":"session","id":"s"}"#,
        "\n",
        r#"{"type":"compaction","summary":"s","tokensBefore":3,"id":"c1","parentId":null}"#,
        "\n",
    ));
    assert!(matches!(
        get_latest_compaction_entry(&with_compaction),
        Some(FileEntry::Entry(SessionEntry::Compaction(_)))
    ));
}

#[test]
fn the_leaf_selections_resolve_to_empty_paths_for_null_default_and_unknown_leaves() {
    let message = entry(json!({
        "type": "message", "id": "a", "parentId": null, "timestamp": TS,
        "message": {"role": "user", "content": "hi", "timestamp": 1}
    }));
    let child = entry(json!({
        "type": "message", "id": "b", "parentId": "a", "timestamp": TS,
        "message": {"role": "user", "content": "more", "timestamp": 2}
    }));
    let entries = vec![FileEntry::Entry(message), FileEntry::Entry(child)];
    let by_id: ByIdIndex = [("a".to_owned(), 0usize), ("b".to_owned(), 1)]
        .into_iter()
        .collect();

    assert!(
        build_session_path(&entries, LeafId::None, &by_id).is_empty(),
        "the null leaf is the empty path"
    );
    assert!(
        build_session_path(&[], LeafId::Default, &by_id).is_empty(),
        "no entries, no path"
    );
    assert_eq!(
        build_session_path(&entries, LeafId::Default, &by_id).len(),
        2,
        "the default leaf is the newest entry"
    );
    assert!(build_session_path(&entries, LeafId::Id("zzz"), &by_id).is_empty());
}

#[test]
fn an_idless_compaction_cannot_anchor_the_context_projection() {
    let message = entry(json!({
        "type": "message", "id": "a", "parentId": null, "timestamp": TS,
        "message": {"role": "user", "content": "hi", "timestamp": 1}
    }));
    let compaction = entry(json!({
        "type": "compaction", "summary": "s", "tokensBefore": 0, "parentId": "a", "timestamp": TS
    }));
    let entries = vec![FileEntry::Entry(message), FileEntry::Entry(compaction)];
    let by_id: ByIdIndex = ByIdIndex::from([("a".to_owned(), 0usize)]);

    let path = build_session_path(&entries, LeafId::Default, &by_id);
    assert_eq!(
        path.len(),
        2,
        "the id-less compaction still walks as the leaf"
    );
    let context = build_context_entries(&entries, LeafId::Default, &by_id);
    assert_eq!(
        context.len(),
        path.len(),
        "the projection declines to anchor and returns the path"
    );
}

#[test]
fn the_custom_message_context_carries_details() {
    let custom = entry(json!({
        "type": "custom_message", "id": "x", "parentId": null, "timestamp": TS,
        "customType": "ext.note", "content": "note", "display": true, "details": {"d": 1}
    }));
    let messages = session_entry_to_context_messages(&FileEntry::Entry(custom));
    let Some(AgentMessage::Custom(custom)) = messages.first() else {
        panic!("the custom message projects");
    };
    assert_eq!(
        custom.data["details"],
        json!({"d": 1}),
        "the details ride the data map"
    );
}

#[test]
fn a_null_content_message_repairs_to_an_empty_block_list_at_parse_time() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = format!("{}/null-content.jsonl", dir.path().display());
    std::fs::write(
        &path,
        concat!(
            r#"{"type":"session","version":3,"id":"nullc","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}"#, "\n",
            r#"{"type":"message","id":"e1","parentId":null,"timestamp":"2026-01-01T00:00:01.000Z","message":{"role":"user","content":null,"timestamp":1}}"#, "\n",
        ),
    )
    .expect("write null-content file");

    let entries = load_entries_from_file(&path);
    assert_eq!(entries.len(), 2, "the repaired message stays in the file");
    if let Some(FileEntry::Entry(SessionEntry::Message(message))) = entries.last() {
        assert!(
            message.message.is_some(),
            "the repaired entry parses as a typed message"
        );
    } else {
        panic!("the repaired entry parses as a typed message");
    }
}

#[test]
fn an_unreadable_file_loads_as_nothing() {
    let entries = load_entries_from_file("/nonexistent/path/session.jsonl");
    assert!(
        entries.is_empty(),
        "a read failure degrades to the empty load"
    );
}

#[test]
fn in_memory_entries_migrate_and_rewrite_is_a_no_op_without_a_file() {
    let mut entries = parse_session_entries(concat!(
        r#"{"type":"session","id":"v1mem"}"#,
        "\n",
        r#"{"type":"custom","customType":"ext"}"#,
        "\n",
    ));
    assert!(
        migrate_session_entries(&mut entries),
        "the in-memory file migrates"
    );
    let session =
        SessionManager::in_memory(None, None, Some(entries)).expect("in-memory with entries");
    assert_eq!(session.session_id(), "v1mem");
    let raw = session
        .entries()
        .first()
        .copied()
        .expect("the raw entry rides");
    assert!(
        raw.entry_id().is_some(),
        "the migration assigned the tree id in memory"
    );
    assert_eq!(
        session.session_file(),
        None,
        "the rewrite is a no-op without a file"
    );
}

#[test]
fn a_missing_file_reports_the_io_arm_of_the_header_scan() {
    let error = pi_coding_agent::session_manager::read_session_header(std::path::Path::new(
        "/nonexistent/session.jsonl",
    ))
    .expect_err("missing file");
    assert!(
        matches!(
            error,
            pi_coding_agent::session_manager::ReadSessionHeaderError::Io(_)
        ),
        "{error}"
    );
}

#[test]
fn a_non_path_input_falls_back_to_itself_and_loads_nothing() {
    let entries = load_entries_from_file("file://host/missing.jsonl");
    assert!(
        entries.is_empty(),
        "the un-normalizable path degrades to the empty load"
    );
}

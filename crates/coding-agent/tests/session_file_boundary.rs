//! Boundary tests for the session file layer: the v1→v3 migrations, the
//! lenient JSONL load, the trailing-newline repair, and the bounded header
//! scan, pinned against upstream's behavior at the pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
use std::fs;
use std::path::Path;

use pi_coding_agent::session_manager::{
    CURRENT_SESSION_VERSION, FileEntry, SessionEntry, load_entries_from_file,
    migrate_session_entries, parse_session_entries, read_session_header,
};

fn write_lines(path: &Path, lines: &[String]) {
    fs::write(path, lines.join("\n")).expect("write fixture");
}

fn session_header_json(id: &str, version: Option<i64>) -> String {
    let version_part = version.map_or_else(String::new, |value| format!(r#""version":{value},"#));
    format!(
        r#"{{"type":"session",{version_part}"id":"{id}","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/tmp"}}"#
    )
}

fn message_entry_json(id: &str, parent: Option<&str>, text: &str) -> String {
    let parent_part = parent.map_or_else(|| "null".to_owned(), |id| format!(r#""{id}""#));
    format!(
        r#"{{"type":"message","id":"{id}","parentId":{parent_part},"timestamp":"2026-01-01T00:00:01.000Z","message":{{"role":"user","content":"{text}","timestamp":0}}}}"#
    )
}

#[test]
fn v1_entries_without_ids_migrate_to_a_chain() {
    let mut entries = parse_session_entries(
        &[
            session_header_json("legacy", None),
            r#"{"type":"message","timestamp":"2026-01-01T00:00:01.000Z","message":{"role":"user","content":"hi","timestamp":0}}"#.to_owned(),
            r#"{"type":"message","timestamp":"2026-01-01T00:00:02.000Z","message":{"role":"user","content":"again","timestamp":0}}"#.to_owned(),
        ]
        .join("\n"),
    );
    // Pre-migration the entries parse typed with no ids.
    let FileEntry::Entry(SessionEntry::Message(first)) = &entries[1] else {
        panic!("v1 message parses typed");
    };
    assert_eq!(first.base.id, None);

    assert!(migrate_session_entries(&mut entries));

    let FileEntry::Session(header) = &entries[0] else {
        panic!("header first");
    };
    assert_eq!(header.version, Some(CURRENT_SESSION_VERSION));
    let FileEntry::Entry(SessionEntry::Message(first)) = &entries[1] else {
        panic!("v1 message migrated");
    };
    let FileEntry::Entry(SessionEntry::Message(second)) = &entries[2] else {
        panic!("second v1 message migrated");
    };
    assert_eq!(second.base.parent_id.as_deref(), first.base.id.as_deref());
    assert_eq!(first.base.parent_id, None);
}

#[test]
fn v1_compaction_kept_index_resolves_to_the_assigned_id() {
    let mut entries = parse_session_entries(
        &[
            session_header_json("legacy", None),
            message_entry_json("seed", None, "hi"),
            r#"{"type":"compaction","summary":"s","tokensBefore":5,"firstKeptEntryIndex":1,"timestamp":"2026-01-01T00:00:02.000Z"}"#.to_owned(),
        ]
        .join("\n"),
    );
    assert!(migrate_session_entries(&mut entries));
    let FileEntry::Entry(SessionEntry::Compaction(compaction)) = &entries[2] else {
        panic!("compaction migrated");
    };
    let FileEntry::Entry(SessionEntry::Message(message)) = &entries[1] else {
        panic!("message migrated");
    };
    assert_eq!(
        compaction.first_kept_entry_id.as_deref(),
        message.base.id.as_deref()
    );
    assert!(compaction.extras.get("firstKeptEntryIndex").is_none());
}

#[test]
fn v1_forward_kept_index_stays_absent() {
    // Upstream resolves entries[index].id mid-loop: a forward-pointing index
    // observes an unassigned id, so firstKeptEntryId stays absent.
    let mut entries = parse_session_entries(
        &[
            session_header_json("legacy", None),
            r#"{"type":"compaction","summary":"s","tokensBefore":5,"firstKeptEntryIndex":2,"timestamp":"2026-01-01T00:00:02.000Z"}"#.to_owned(),
            message_entry_json("seed", None, "hi"),
        ]
        .join("\n"),
    );
    assert!(migrate_session_entries(&mut entries));
    let FileEntry::Entry(SessionEntry::Compaction(compaction)) = &entries[1] else {
        panic!("compaction migrated");
    };
    assert_eq!(compaction.first_kept_entry_id, None);
}

#[test]
fn v2_hook_message_role_renames_to_custom() {
    let mut entries = parse_session_entries(
        &[
            session_header_json("legacy", Some(2)),
            r#"{"type":"message","id":"m1","parentId":null,"timestamp":"2026-01-01T00:00:01.000Z","message":{"role":"hookMessage","customType":"x","content":"hi","display":true,"timestamp":0}}"#.to_owned(),
        ]
        .join("\n"),
    );
    assert!(migrate_session_entries(&mut entries));
    let FileEntry::Entry(SessionEntry::Message(message)) = &entries[1] else {
        panic!("message parsed");
    };
    let Some(pi_agent_core::types::AgentMessage::Custom(custom)) = &message.message else {
        panic!("custom message parsed");
    };
    assert_eq!(custom.role, "custom");
    assert_eq!(custom.field("customType"), Some(&serde_json::json!("x")));
}

#[test]
fn v3_file_migrates_nothing() {
    let mut entries = parse_session_entries(
        &[
            session_header_json("current", Some(3)),
            message_entry_json("m1", None, "hi"),
        ]
        .join("\n"),
    );
    assert!(!migrate_session_entries(&mut entries));
}

#[test]
fn load_rejects_a_file_whose_first_entry_is_not_a_session_header() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("not-a-session.log");
    let original = "{\"type\":\"event\",\"data\":\"not a session\"}\n";
    fs::write(&path, original).expect("write");
    assert!(load_entries_from_file(&path.display().to_string()).is_empty());
    assert_eq!(fs::read_to_string(&path).expect("read"), original);
}

#[test]
fn load_repairs_a_missing_trailing_newline() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("session.jsonl");
    write_lines(
        &path,
        &[
            session_header_json("id", Some(3)),
            message_entry_json("m1", None, "hi"),
        ],
    );
    let entries = load_entries_from_file(&path.display().to_string());
    assert_eq!(entries.len(), 2);
    let content = fs::read_to_string(&path).expect("read");
    assert!(
        content.ends_with('\n'),
        "the unterminated tail is repaired in place"
    );
}

#[test]
fn load_preserves_unknown_entry_shapes() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("session.jsonl");
    write_lines(
        &path,
        &[
            session_header_json("id", Some(3)),
            r#"{"type":"extension-thing","id":"x1","parentId":null,"timestamp":"2026-01-01T00:00:01.000Z","payload":{"a":1}}"#.to_owned(),
            message_entry_json("m1", Some("x1"), "hi"),
        ],
    );
    let entries = load_entries_from_file(&path.display().to_string());
    assert_eq!(entries.len(), 3);
    let FileEntry::Other(raw) = &entries[1] else {
        panic!("unknown shape rides as a raw value");
    };
    assert_eq!(raw.get("payload"), Some(&serde_json::json!({"a": 1})));
}

#[test]
fn known_type_with_a_malformed_required_field_degrades_to_raw() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("session.jsonl");
    // A message entry whose message payload is not an object: upstream keeps
    // the raw object; the port rides it as a raw value the tree and index
    // skip.
    write_lines(
        &path,
        &[
            session_header_json("id", Some(3)),
            r#"{"type":"message","id":"m1","parentId":null,"timestamp":"2026-01-01T00:00:01.000Z","message":"not an object"}"#.to_owned(),
        ],
    );
    let entries = load_entries_from_file(&path.display().to_string());
    assert!(matches!(entries[1], FileEntry::Other(_)));
}

#[test]
fn header_scan_reads_the_first_session_header() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("session.jsonl");
    write_lines(
        &path,
        &[
            "// not json".to_owned(),
            session_header_json("first", Some(3)),
            message_entry_json("m1", None, "hi"),
        ],
    );
    let header = read_session_header(&path).expect("scan");
    assert_eq!(header.expect("header").id, "first");
}

#[test]
fn header_scan_stops_at_the_first_parsed_non_header_entry() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("session.jsonl");
    write_lines(
        &path,
        &[
            r#"{"type":"message","id":"m0","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z","message":{"role":"user","content":"hi","timestamp":0}}"#.to_owned(),
            session_header_json("never", Some(3)),
        ],
    );
    assert_eq!(read_session_header(&path).expect("scan"), None);
}

#[test]
fn header_scan_returns_none_at_eof() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("session.jsonl");
    write_lines(&path, &["   ".to_owned(), "// still not json".to_owned()]);
    assert_eq!(read_session_header(&path).expect("scan"), None);
}

#[test]
fn header_scan_limits_long_unterminated_leading_content() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("session.jsonl");
    let long_line = "x".repeat(1024 * 1024 + 16);
    fs::write(
        &path,
        format!("{long_line}\n{}", session_header_json("late", Some(3))),
    )
    .expect("write");
    let error = read_session_header(&path).expect_err("scan limit");
    assert!(
        error.to_string().contains("1048576-byte scan limit"),
        "the limit message is upstream's: {error}"
    );
}

#[test]
fn header_scan_accepts_a_final_header_ending_exactly_at_the_limit() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("session.jsonl");
    // Pad one unterminated leading line so the final header line ends
    // exactly at the scan limit with no trailing newline.
    let header = session_header_json("boundary", Some(3));
    let padding = "y".repeat(1024 * 1024 - header.len() - 1);
    fs::write(&path, format!("{padding}\n{header}")).expect("write");
    let header = read_session_header(&path).expect("scan");
    assert_eq!(header.expect("header").id, "boundary");
}

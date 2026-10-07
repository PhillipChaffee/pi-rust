//! The file-entry boundary suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the accessor arms across the
//! three `FileEntry` variants, the hand-written serde pairs, and the header's
//! optional-field wire shapes.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
use serde_json::json;

use pi_coding_agent::session_manager::{
    FileEntry, ReadSessionHeaderError, SessionEntry, SessionHeader, SessionManagerError,
};

const TS: &str = "2026-01-01T00:00:00.000Z";

fn entry(wire: serde_json::Value) -> SessionEntry {
    match serde_json::from_value::<FileEntry>(wire).expect("typed entry") {
        FileEntry::Entry(entry) => entry,
        other => panic!("expected a typed entry, got {other:?}"),
    }
}

#[test]
fn the_file_entry_accessors_cover_all_three_variants() {
    let header = FileEntry::Session(SessionHeader {
        id: "s1".to_owned(),
        timestamp: TS.to_owned(),
        ..SessionHeader::default()
    });
    assert_eq!(header.entry_id(), Some("s1"), "the header ids as itself");
    assert_eq!(header.entry_parent_id(), None);
    assert_eq!(header.entry_timestamp(), TS);

    let typed = FileEntry::Entry(entry(json!({
        "type": "custom", "id": "c1", "parentId": "p1", "timestamp": TS, "customType": "x"
    })));
    assert_eq!(typed.entry_id(), Some("c1"));
    assert_eq!(typed.entry_parent_id(), Some("p1"));
    assert_eq!(typed.entry_timestamp(), TS);

    let raw =
        FileEntry::Other(json!({"type": "mystery", "id": "o1", "parentId": "p2", "timestamp": TS}));
    assert_eq!(
        raw.entry_id(),
        Some("o1"),
        "a raw value ids from its own fields"
    );
    assert_eq!(raw.entry_parent_id(), Some("p2"));
    assert_eq!(raw.entry_timestamp(), TS);

    let bare = FileEntry::Other(json!({"type": "mystery"}));
    assert_eq!(bare.entry_id(), None);
    assert_eq!(bare.entry_parent_id(), None);
    assert_eq!(
        bare.entry_timestamp(),
        "",
        "a missing raw timestamp degrades to empty"
    );
}

#[test]
fn the_file_entry_serde_pair_round_trips_all_three_variants() {
    let raw = FileEntry::Other(json!({"type": "mystery", "id": "o1"}));
    let text = serde_json::to_string(&raw).expect("serialize");
    assert_eq!(
        serde_json::from_str::<FileEntry>(&text).expect("deserialize"),
        raw
    );

    let header = FileEntry::Session(SessionHeader {
        id: "s1".to_owned(),
        timestamp: TS.to_owned(),
        ..SessionHeader::default()
    });
    let text = serde_json::to_string(&header).expect("serialize");
    assert_eq!(
        serde_json::from_str::<FileEntry>(&text).expect("deserialize"),
        header
    );

    let typed = FileEntry::Entry(entry(json!({
        "type": "thinking_level_change", "id": "t1", "parentId": null, "timestamp": TS, "thinkingLevel": "high"
    })));
    let text = serde_json::to_string(&typed).expect("serialize");
    assert_eq!(
        serde_json::from_str::<FileEntry>(&text).expect("deserialize"),
        typed
    );
}

#[test]
fn the_header_wire_omits_absent_fields_and_keeps_extras() {
    let minimal = SessionHeader::default();
    assert_eq!(
        serde_json::to_string(&minimal).expect("serialize"),
        r#"{"type":"session","id":""}"#,
        "the default header serializes the always-present empty id"
    );

    let full = SessionHeader {
        parent_session: Some("parent.jsonl".to_owned()),
        extras: serde_json::Map::from_iter([("custom".to_owned(), json!("x"))]),
        ..SessionHeader {
            id: "s1".to_owned(),
            timestamp: TS.to_owned(),
            ..Default::default()
        }
    };
    let text = serde_json::to_string(&full).expect("serialize");
    assert!(text.contains(r#""parentSession":"parent.jsonl""#), "{text}");
    assert!(text.contains(r#""custom":"x""#), "{text}");
    assert_eq!(
        serde_json::from_str::<SessionHeader>(&text).expect("deserialize"),
        full
    );
}

#[test]
fn the_header_parse_repairs_null_fields_and_rejects_missing_ids() {
    let nulls: SessionHeader = serde_json::from_str(
        r#"{"type":"session","id":"s1","timestamp":null,"cwd":null,"parentSession":null}"#,
    )
    .expect("null optional fields parse");
    assert_eq!(nulls.timestamp, "", "a null timestamp repairs to empty");
    assert_eq!(nulls.cwd, None);
    assert_eq!(nulls.parent_session, None);

    let missing_id = serde_json::from_str::<SessionHeader>(r#"{"type":"session","timestamp":"t"}"#);
    assert!(missing_id.is_err(), "a header without an id is rejected");
}

#[test]
fn the_error_taxonomy_displays_and_converts() {
    let io = SessionManagerError::from(std::io::Error::other("boom"));
    assert_eq!(io, SessionManagerError::Io("boom".to_owned()));
    assert_eq!(
        ReadSessionHeaderError::Io("gone".to_owned()).to_string(),
        "gone"
    );
    assert_eq!(
        SessionManagerError::PathNormalize("bad path".to_owned()).to_string(),
        "bad path"
    );
    assert_eq!(
        SessionManagerError::Io("io".to_owned()).to_string(),
        "io",
        "the io arm reprints the underlying message"
    );
}

#[test]
fn every_entry_variant_exposes_base_type_and_mut_base() {
    let ts = "2026-01-01T00:00:00.000Z";
    let message = json!({
        "type": "message", "id": "m1", "parentId": null, "timestamp": ts,
        "message": {"role": "user", "content": "hi", "timestamp": 1}
    });
    let wires = [
        (message, "message"),
        (
            json!({"type": "thinking_level_change", "id": "t1", "parentId": null, "timestamp": ts, "thinkingLevel": "high"}),
            "thinking_level_change",
        ),
        (
            json!({"type": "model_change", "id": "d1", "parentId": null, "timestamp": ts, "provider": "openai", "modelId": "gpt-5"}),
            "model_change",
        ),
        (
            json!({"type": "compaction", "id": "c1", "parentId": null, "timestamp": ts, "summary": "s", "tokensBefore": 3}),
            "compaction",
        ),
        (
            json!({"type": "branch_summary", "id": "b1", "parentId": null, "timestamp": ts, "fromId": "m1", "summary": "sum"}),
            "branch_summary",
        ),
        (
            json!({"type": "custom", "id": "u1", "parentId": null, "timestamp": ts, "customType": "ext"}),
            "custom",
        ),
        (
            json!({"type": "label", "id": "l1", "parentId": null, "timestamp": ts, "targetId": "m1", "label": "x"}),
            "label",
        ),
        (
            json!({"type": "session_info", "id": "s1", "parentId": null, "timestamp": ts, "name": "the name"}),
            "session_info",
        ),
        (
            json!({"type": "custom_message", "id": "k1", "parentId": null, "timestamp": ts, "customType": "ext.note", "content": "n", "display": true}),
            "custom_message",
        ),
    ];
    for (wire, expected_type) in wires {
        let mut entry = entry(wire);
        assert_eq!(entry.type_name(), expected_type, "the wire discriminator");
        assert_eq!(entry.base().timestamp, ts, "the shared base reads");
        entry.base_mut().timestamp = "mutated".to_owned();
        assert_eq!(
            entry.base().timestamp,
            "mutated",
            "the mutable base writes through"
        );
    }
}

#[test]
fn a_header_with_a_malformed_typed_field_rejects() {
    for wire in [
        r#"{"type":"session","id":"s1","version":"x"}"#,
        r#"{"type":"session","id":"s1","timestamp":5}"#,
        r#"{"type":"session","id":"s1","cwd":5}"#,
        r#"{"type":"session","id":"s1","parentSession":5}"#,
    ] {
        let parsed = serde_json::from_str::<SessionHeader>(wire);
        assert!(parsed.is_err(), "a non-typed field value rejects: {wire}");
    }
}

//! The snapshot fork builder's unit tests: the tree/branch projections and
//! the source-snapshot validations over synthetic snapshots, the surface the
//! streaming backends (the JSONL child) consume.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use super::{ForkDestinationSnapshot, ForkSourceSnapshot, create_fork_snapshot};
use crate::harness::session::types::{CustomEntryBody, Entry, ForkPosition};
use crate::harness::session::values::StoredValue;
use serde_json::json;

fn custom_entry(id: &str, parent_id: Option<&str>, seq: u64, custom_type: &str) -> Entry {
    Entry::Custom {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        seq,
        timestamp: 1,
        body: CustomEntryBody {
            custom_type: custom_type.to_owned(),
            data: None,
        },
    }
}

fn stored(namespace: &str, key: &str, value: serde_json::Value, seq: u64) -> StoredValue {
    StoredValue {
        namespace: namespace.to_owned(),
        key: key.to_owned(),
        value,
        seq,
    }
}

/// The configured lane trio a source snapshot carries, upstream's
/// `branch.tip`/`lane.config`/`lane.state` rows.
fn lane_rows(branch: &str, tip: Option<&str>) -> Vec<StoredValue> {
    let tip_value = tip.map_or(serde_json::Value::Null, |tip| json!(tip));
    vec![
        stored("pi.branch.tip", branch, tip_value, 10),
        stored(
            "pi.lane.config",
            branch,
            serde_json::to_value(crate::harness::session::types::LaneConfiguration {
                model: crate::harness::session::types::ModelIdentity {
                    provider: "provider".to_owned(),
                    model_id: "model".to_owned(),
                },
                thinking_level: crate::types::ThinkingLevel::Off,
                active_tool_names: vec!["read".to_owned()],
            })
            .expect("config wire"),
            11,
        ),
        stored(
            "pi.lane.state",
            branch,
            json!({
                "currentOperationId": serde_json::Value::Null,
                "lastOperationId": serde_json::Value::Null,
                "inbox": [],
            }),
            12,
        ),
    ]
}

#[test]
fn tree_forks_copy_every_entry_and_project_surviving_scalars_with_fresh_sequences() {
    let source = ForkSourceSnapshot {
        entries: vec![
            custom_entry("a", None, 1, "root"),
            custom_entry("b", Some("a"), 2, "child"),
        ],
        scalar_values: vec![
            stored("pi.session.name", "", json!("kept"), 3),
            stored("app.state", "", json!("copied"), 4),
        ],
        entries_complete: None,
    };

    let ForkDestinationSnapshot {
        entries,
        scalar_values,
        next_seq,
    } = create_fork_snapshot(
        &source,
        &crate::harness::session::types::ForkOptions::Tree { id: None },
    )
    .expect("tree fork");

    assert_eq!(entries.len(), 2);
    let sequences: Vec<u64> = entries.values().map(Entry::seq).collect();
    assert_eq!(*sequences.iter().max().expect("entries"), 2);
    // The scalars keep their contents and land on fresh sequences above the
    // copied tree's.
    let namespaced: Vec<(&str, &serde_json::Value)> = scalar_values
        .iter()
        .map(|stored| (stored.namespace.as_str(), &stored.value))
        .collect();
    assert_eq!(
        namespaced,
        [
            ("pi.session.name", &json!("kept")),
            ("app.state", &json!("copied"))
        ],
    );
    // The counter rides past every projected scalar, upstream's `nextSeq++`.
    assert_eq!(next_seq, 5);
    assert_eq!(
        scalar_values
            .iter()
            .map(|stored| stored.seq)
            .collect::<Vec<_>>(),
        [3, 4],
    );
}

#[test]
fn branch_forks_project_the_destination_tip_and_reset_lane_state() {
    let source = ForkSourceSnapshot {
        entries: vec![
            custom_entry("a", None, 1, "root"),
            custom_entry("b", Some("a"), 2, "child"),
            custom_entry("c", Some("a"), 3, "sibling"),
        ],
        scalar_values: [
            lane_rows("main", Some("b")),
            lane_rows("other", Some("c")),
            vec![
                stored("pi.entry.label", "b", json!("kept label"), 20),
                stored("pi.entry.label", "c", json!("dropped label"), 21),
                stored("pi.result", "op", json!("dropped"), 22),
                stored("pi.op.meta", "op", json!("dropped"), 23),
                stored("app.state", "", json!("branch forks drop me"), 24),
            ],
        ]
        .concat(),
        entries_complete: None,
    };

    let options = crate::harness::session::types::ForkOptions::Branch {
        branch: "main".to_owned(),
        entry_id: Some("b".to_owned()),
        position: Some(ForkPosition::At),
        id: None,
    };
    let fork = create_fork_snapshot(&source, &options).expect("branch fork");

    assert_eq!(fork.entries.len(), 2);
    // The copied branch's tip rides the destination plan; the other branch's
    // tip row does not survive.
    let tips: Vec<&StoredValue> = fork
        .scalar_values
        .iter()
        .filter(|stored| stored.namespace == "pi.branch.tip")
        .collect();
    assert_eq!(tips.len(), 1);
    assert_eq!(tips[0].key, "main");
    assert_eq!(tips[0].value, json!("b"));
    // The label rides its copied entry; the sibling's label drops; results,
    // operation state, and application values drop on branch forks.
    let labels: Vec<&StoredValue> = fork
        .scalar_values
        .iter()
        .filter(|stored| stored.namespace == "pi.entry.label")
        .collect();
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0].key, "b");
    assert!(
        fork.scalar_values
            .iter()
            .all(|stored| stored.namespace != "pi.result"
                && stored.namespace != "pi.op.meta"
                && stored.namespace != "app.state"),
    );
}

#[test]
fn partial_snapshots_skip_tip_existence_checks() {
    let source = ForkSourceSnapshot {
        entries: vec![],
        scalar_values: lane_rows("main", Some("gone")),
        entries_complete: Some(false),
    };

    let fork = create_fork_snapshot(
        &source,
        &crate::harness::session::types::ForkOptions::Branch {
            branch: "main".to_owned(),
            entry_id: None,
            position: None,
            id: None,
        },
    );
    // The tip's entry is absent from the partial snapshot: the branch fork
    // still walks, and fails on the missing ancestry rather than the tip.
    assert!(fork.is_err());
}

#[test]
fn validations_pin_the_upstream_messages() {
    // A configured lane without a branch tip row.
    let source = ForkSourceSnapshot {
        entries: vec![],
        scalar_values: vec![stored(
            "pi.lane.config",
            "main",
            json!({ "model": { "provider": "p", "modelId": "m" }, "thinkingLevel": "off", "activeToolNames": [] }),
            1,
        )],
        entries_complete: None,
    };
    let error = create_fork_snapshot(
        &source,
        &crate::harness::session::types::ForkOptions::Tree { id: None },
    )
    .expect_err("missing tip");
    assert_eq!(
        error.to_string(),
        "Source session branch \"main\" is missing branch.tip"
    );

    // A tip whose lane state is missing.
    let source = ForkSourceSnapshot {
        entries: vec![],
        scalar_values: vec![
            stored("pi.branch.tip", "main", json!("b"), 1),
            stored("pi.lane.config", "main", json!({}), 2),
        ],
        entries_complete: None,
    };
    let error = create_fork_snapshot(
        &source,
        &crate::harness::session::types::ForkOptions::Tree { id: None },
    )
    .expect_err("incomplete lane");
    assert_eq!(
        error.to_string(),
        "Source session branch \"main\" has incomplete lane state"
    );

    // A branch fork of an unconfigured branch.
    let source = ForkSourceSnapshot {
        entries: vec![],
        scalar_values: vec![stored("pi.branch.tip", "main", serde_json::Value::Null, 1)],
        entries_complete: None,
    };
    let error = create_fork_snapshot(
        &source,
        &crate::harness::session::types::ForkOptions::Branch {
            branch: "main".to_owned(),
            entry_id: None,
            position: None,
            id: None,
        },
    )
    .expect_err("unconfigured branch");
    assert_eq!(
        error.to_string(),
        "Source branch \"main\" is not a configured AgentLane"
    );

    // A complete snapshot whose tip entry is unknown.
    let source = ForkSourceSnapshot {
        entries: vec![],
        scalar_values: lane_rows("main", Some("gone")),
        entries_complete: None,
    };
    let error = create_fork_snapshot(
        &source,
        &crate::harness::session::types::ForkOptions::Tree { id: None },
    )
    .expect_err("unknown tip");
    assert_eq!(
        error.to_string(),
        "Source session branch \"main\" has an unknown tip"
    );
}

#[test]
fn unknown_reserved_namespaces_reject_through_the_snapshot_path() {
    let source = ForkSourceSnapshot {
        entries: vec![custom_entry("a", None, 1, "root")],
        scalar_values: [
            lane_rows("main", Some("a")),
            vec![stored("pi.unknown", "", json!(true), 30)],
        ]
        .concat(),
        entries_complete: None,
    };
    let error = create_fork_snapshot(
        &source,
        &crate::harness::session::types::ForkOptions::Tree { id: None },
    )
    .expect_err("reserved namespace");
    assert_eq!(
        error.to_string(),
        "Unknown reserved fork namespace: pi.unknown"
    );
}

#[test]
fn empty_snapshots_fork_to_empty_destinations() {
    let source = ForkSourceSnapshot::default();
    let fork = create_fork_snapshot(
        &source,
        &crate::harness::session::types::ForkOptions::Tree { id: None },
    )
    .expect("empty tree fork");
    assert!(fork.entries.is_empty());
    assert!(fork.scalar_values.is_empty());
    assert_eq!(fork.next_seq, 1);
}

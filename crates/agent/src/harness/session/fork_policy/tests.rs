//! The fork-policy unit tests: the ancestry walk and the reserved-namespace
//! projection over synthetic sources, the branches the memory/JSONL
//! conformance suites exercise through whole backends.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use super::{
    BranchForkSource, ForkCurrentStatePlan, ForkStateWrite, project_fork_current_state_write,
    select_branch_fork,
};
use crate::harness::session::types::{ForkOptions, ForkPosition, SessionError};
use serde_json::json;
use std::collections::BTreeSet;

fn branch_options(
    branch: &str,
    entry_id: Option<&str>,
    position: Option<ForkPosition>,
) -> ForkOptions {
    ForkOptions::Branch {
        branch: branch.to_owned(),
        entry_id: entry_id.map(str::to_owned),
        position,
        id: None,
    }
}

/// The parent map one synthetic tree reads; a missing id reads as no parent.
fn parent_map(parents: &[(&str, Option<&str>)]) -> impl Fn(&str) -> Option<Option<String>> {
    let map: std::collections::HashMap<String, Option<String>> = parents
        .iter()
        .map(|(id, parent)| ((*id).to_owned(), parent.map(str::to_owned)))
        .collect();
    move |entry_id: &str| map.get(entry_id).cloned()
}

#[test]
fn selects_the_requested_path_from_the_tip() {
    let mut selected = BTreeSet::new();
    let mut source = BranchForkSource {
        tip: Some(Some("c".to_owned())),
        get_parent: &parent_map(&[("c", Some("b")), ("b", Some("a")), ("a", None)]),
        select_entry: &mut |entry_id| {
            selected.insert(entry_id.to_owned());
        },
    };
    let plan = select_branch_fork(
        &branch_options("main", Some("b"), Some(ForkPosition::At)),
        &mut source,
    )
    .expect("walk");
    assert_eq!(
        plan,
        ForkCurrentStatePlan::Branch {
            branch: "main".to_owned(),
            destination_tip: Some("b".to_owned()),
        },
    );
    assert_eq!(selected, BTreeSet::from(["b".to_owned(), "a".to_owned()]));
}

#[test]
fn before_placement_selects_up_to_the_parent() {
    let mut selected = BTreeSet::new();
    let mut source = BranchForkSource {
        tip: Some(Some("c".to_owned())),
        get_parent: &parent_map(&[("c", Some("b")), ("b", Some("a")), ("a", None)]),
        select_entry: &mut |entry_id| {
            selected.insert(entry_id.to_owned());
        },
    };
    let plan = select_branch_fork(
        &branch_options("main", Some("b"), Some(ForkPosition::Before)),
        &mut source,
    )
    .expect("walk");
    assert_eq!(
        plan,
        ForkCurrentStatePlan::Branch {
            branch: "main".to_owned(),
            destination_tip: Some("a".to_owned()),
        },
    );
    assert_eq!(selected, BTreeSet::from(["a".to_owned()]));
}

#[test]
fn an_empty_branch_forks_at_its_null_tip() {
    let mut selected = BTreeSet::new();
    let mut source = BranchForkSource {
        tip: Some(None),
        get_parent: &parent_map(&[]),
        select_entry: &mut |entry_id| {
            selected.insert(entry_id.to_owned());
        },
    };
    let plan = select_branch_fork(&branch_options("empty", None, None), &mut source).expect("walk");
    assert_eq!(
        plan,
        ForkCurrentStatePlan::Branch {
            branch: "empty".to_owned(),
            destination_tip: None,
        },
    );
    assert!(selected.is_empty());
}

#[test]
fn unknown_branches_and_off_branch_entries_reject() {
    let mut source = BranchForkSource {
        tip: None,
        get_parent: &parent_map(&[]),
        select_entry: &mut |_| {},
    };
    let error = select_branch_fork(&branch_options("missing", None, None), &mut source)
        .expect_err("unknown branch");
    assert_eq!(error.to_string(), "Unknown source branch: missing");

    let mut source = BranchForkSource {
        tip: Some(Some("c".to_owned())),
        get_parent: &parent_map(&[("c", Some("b")), ("b", Some("a")), ("a", None)]),
        select_entry: &mut |_| {},
    };
    let error = select_branch_fork(
        &branch_options("main", Some("x"), Some(ForkPosition::At)),
        &mut source,
    )
    .expect_err("off-branch entry");
    assert_eq!(
        error.to_string(),
        "Fork entry x is not on source branch \"main\""
    );
}

#[test]
fn corrupt_ancestries_reject() {
    let mut source = BranchForkSource {
        tip: Some(Some("c".to_owned())),
        get_parent: &parent_map(&[("c", Some("gone"))]),
        select_entry: &mut |_| {},
    };
    let error = select_branch_fork(&branch_options("main", None, None), &mut source)
        .expect_err("corrupt ancestry");
    // The walk steps to the missing parent and names it, upstream's
    // `missing parent ${entryId}`.
    assert_eq!(
        error.to_string(),
        "Corrupt source branch: missing parent gone"
    );
}

#[test]
fn projects_the_reserved_namespaces_by_scope() {
    let tree = ForkCurrentStatePlan::Tree;
    let branch = |branch: &str, tip: Option<&str>| ForkCurrentStatePlan::Branch {
        branch: branch.to_owned(),
        destination_tip: tip.map(str::to_owned),
    };
    let is_copied = |entry_id: &str| entry_id == "kept";
    let value_set = |namespace: &str, key: &str| ForkStateWrite::ValueSet {
        seq: 1,
        namespace: namespace.to_owned(),
        key: key.to_owned(),
        value: json!("value"),
    };

    // The session name always survives; labels ride their entries.
    let write = value_set("pi.session.name", "");
    assert_eq!(
        project_fork_current_state_write(&write, &branch("main", None), &is_copied)
            .expect("projection"),
        Some(write),
    );
    let label = value_set("pi.entry.label", "kept");
    assert_eq!(
        project_fork_current_state_write(&label, &branch("main", None), &is_copied)
            .expect("projection"),
        Some(label.clone()),
    );
    let dropped_label = value_set("pi.entry.label", "gone");
    assert_eq!(
        project_fork_current_state_write(&dropped_label, &branch("main", None), &is_copied)
            .expect("projection"),
        None,
    );

    // Branch tips collapse to the destination tip on branch forks.
    let tip = value_set("pi.branch.tip", "main");
    assert_eq!(
        project_fork_current_state_write(&tip, &branch("main", Some("kept")), &is_copied)
            .expect("projection"),
        Some(ForkStateWrite::ValueSet {
            seq: 1,
            namespace: "pi.branch.tip".to_owned(),
            key: "main".to_owned(),
            value: json!("kept"),
        }),
    );
    let other_tip = value_set("pi.branch.tip", "other");
    assert_eq!(
        project_fork_current_state_write(&other_tip, &branch("main", None), &is_copied)
            .expect("projection"),
        None,
    );

    // Lane configs ride the copied branch; lane states reset to idle.
    let config = value_set("pi.lane.config", "main");
    assert_eq!(
        project_fork_current_state_write(&config, &branch("main", None), &is_copied)
            .expect("projection"),
        Some(config.clone()),
    );
    let state = value_set("pi.lane.state", "main");
    match project_fork_current_state_write(&state, &branch("main", None), &is_copied)
        .expect("projection")
    {
        Some(ForkStateWrite::ValueSet { value, .. }) => {
            assert_eq!(
                value,
                json!({ "currentOperationId": None::<String>, "lastOperationId": None::<String>, "inbox": [] })
            );
        }
        _ => panic!("expected the idle lane state"),
    }

    // Results, operation, and pending state never survive.
    for namespace in ["pi.result", "pi.op.meta", "pi.pending.entry"] {
        let write = value_set(namespace, "key");
        assert_eq!(
            project_fork_current_state_write(&write, &tree, &is_copied).expect("projection"),
            None,
        );
    }

    // Unknown reserved namespaces reject; application namespaces ride tree
    // forks only.
    for namespace in ["pi", "pi.unknown"] {
        let write = value_set(namespace, "key");
        let error = project_fork_current_state_write(&write, &tree, &is_copied)
            .expect_err("reserved namespace");
        assert!(matches!(error, SessionError::Message(_)));
    }
    let application = value_set("app.state", "");
    assert_eq!(
        project_fork_current_state_write(&application, &tree, &is_copied).expect("projection"),
        Some(application.clone()),
    );
    assert_eq!(
        project_fork_current_state_write(&application, &branch("main", None), &is_copied)
            .expect("projection"),
        None,
    );
}

#[test]
fn list_appends_project_like_values() {
    let tree = ForkCurrentStatePlan::Tree;
    let append = |namespace: &str| ForkStateWrite::ListAppend {
        seq: 2,
        namespace: namespace.to_owned(),
        key: "key".to_owned(),
        value: json!("element"),
    };
    assert_eq!(
        project_fork_current_state_write(&append("app.events"), &tree, &|_| true)
            .expect("projection"),
        Some(append("app.events")),
    );
    assert_eq!(
        project_fork_current_state_write(&append("pi.op.meta"), &tree, &|_| true)
            .expect("projection"),
        None,
    );
}

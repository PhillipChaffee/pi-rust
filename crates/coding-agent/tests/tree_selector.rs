//! `tree-selector.test.ts` at pin `60e7e76`, ported 1:1, plus boundaries
//! (#208).

#![expect(
    clippy::expect_used,
    reason = "the fixtures are valid by construction; a malformed one should fail the test, not the suite"
)]

#[path = "common/selectors.rs"]
mod selectors;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use pi_agent_core::harness::session::jsonl::codec::parse_iso8601_millis;
use pi_coding_agent::modes::interactive::components::tree_selector::{
    FilterMode, TreeSelectorComponent, TreeSelectorOptions, format_label_timestamp,
    format_label_timestamp_at,
};
use pi_coding_agent::session_manager::{FileEntry, SessionEntry, SessionTreeNode};
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::tui::{Component, Focusable};
use pi_tui::utils::visible_width;
use serde_json::{Value, json};

use selectors::{output, setup};

fn entry(value: Value) -> FileEntry {
    FileEntry::Entry(serde_json::from_value::<SessionEntry>(value).expect("valid session entry"))
}

fn user_message(id: &str, parent: Option<&str>, content: &str) -> FileEntry {
    entry(json!({
        "type": "message",
        "id": id,
        "parentId": parent,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {"role": "user", "content": content, "timestamp": 0}
    }))
}

fn assistant_message(id: &str, parent: Option<&str>, text: &str) -> FileEntry {
    entry(json!({
        "type": "message",
        "id": id,
        "parentId": parent,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": "claude-sonnet-4",
            "usage": {
                "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}
            },
            "stopReason": "stop",
            "timestamp": 0
        }
    }))
}

fn tool_call_only_assistant(id: &str, parent: Option<&str>) -> FileEntry {
    entry(json!({
        "type": "message",
        "id": id,
        "parentId": parent,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": format!("tc-{id}"), "name": "read", "arguments": {"path": "test.ts"}}],
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": "claude-sonnet-4",
            "usage": {
                "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}
            },
            "stopReason": "toolUse",
            "timestamp": 0
        }
    }))
}

fn model_change(id: &str, parent: Option<&str>) -> FileEntry {
    entry(json!({
        "type": "model_change",
        "id": id,
        "parentId": parent,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "provider": "anthropic",
        "modelId": "claude-sonnet-4"
    }))
}

fn thinking_change(id: &str, parent: Option<&str>) -> FileEntry {
    entry(json!({
        "type": "thinking_level_change",
        "id": id,
        "parentId": parent,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "thinkingLevel": "high"
    }))
}

fn build_tree(entries: Vec<FileEntry>) -> Vec<SessionTreeNode> {
    if entries.is_empty() {
        return Vec::new();
    }
    let nodes: Vec<SessionTreeNode> = entries
        .into_iter()
        .map(|entry| SessionTreeNode {
            entry,
            children: Vec::new(),
            label: None,
            label_timestamp: None,
        })
        .collect();
    let by_id: HashMap<String, usize> = nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| node.entry.entry_id().map(|id| (id.to_string(), index)))
        .collect();
    let mut child_indices: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut roots = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        match node.entry.entry_parent_id() {
            Some(parent) => {
                if let Some(&parent_index) = by_id.get(parent) {
                    child_indices[parent_index].push(index);
                }
            }
            None => roots.push(index),
        }
    }
    let mut slots: Vec<Option<SessionTreeNode>> = nodes.into_iter().map(Some).collect();
    roots
        .into_iter()
        .map(|index| take_node(index, &child_indices, &mut slots))
        .collect()
}

fn take_node(
    index: usize,
    child_indices: &[Vec<usize>],
    slots: &mut [Option<SessionTreeNode>],
) -> SessionTreeNode {
    let mut node = slots[index].take().expect("node");
    node.children = child_indices[index]
        .iter()
        .map(|&child| take_node(child, child_indices, slots))
        .collect();
    node
}

fn selector(tree: &[SessionTreeNode], leaf: Option<&str>) -> Rc<TreeSelectorComponent> {
    TreeSelectorComponent::new(
        tree,
        leaf,
        24,
        TreeSelectorOptions {
            on_select: Rc::new(|_| {}),
            on_cancel: Rc::new(|| {}),
            on_label_change: None,
            initial_selected_id: None,
            initial_filter_mode: None,
        },
    )
}

fn selected_id(selector: &TreeSelectorComponent) -> Option<String> {
    selector.get_tree_list().get_selected_entry_id()
}

#[test]
fn focuses_nearest_visible_ancestor_for_a_model_change_leaf() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
        user_message("user-2", Some("asst-1"), "active branch"),
        model_change("model-1", Some("user-2")),
        user_message("user-3", Some("asst-1"), "sibling branch"),
    ]);
    let selector = selector(&tree, Some("model-1"));
    assert_eq!(selected_id(&selector).as_deref(), Some("user-2"));
}

#[test]
fn focuses_nearest_visible_ancestor_for_a_thinking_change_leaf() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
        user_message("user-2", Some("asst-1"), "active branch"),
        thinking_change("thinking-1", Some("user-2")),
        user_message("user-3", Some("asst-1"), "sibling branch"),
    ]);
    let selector = selector(&tree, Some("thinking-1"));
    assert_eq!(selected_id(&selector).as_deref(), Some("user-2"));
}

#[test]
fn switches_to_the_nearest_visible_user_message_in_user_only_mode() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
        user_message("user-2", Some("asst-1"), "active branch"),
        assistant_message("asst-2", Some("user-2"), "response"),
        user_message("user-3", Some("asst-1"), "sibling branch"),
    ]);
    let selector = selector(&tree, Some("asst-2"));
    let list = selector.get_tree_list();
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("asst-2"));

    selector.handle_input("\x15");
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-2"));
}

#[test]
fn returns_to_the_nearest_visible_ancestor_when_switching_back_to_default() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
        user_message("user-2", Some("asst-1"), "active branch"),
        assistant_message("asst-2", Some("user-2"), "response"),
        user_message("user-3", Some("asst-1"), "sibling branch"),
    ]);
    let selector = selector(&tree, Some("asst-2"));
    let list = selector.get_tree_list();

    selector.handle_input("\x15");
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-2"));
    selector.handle_input("\x04");
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-2"));
}

#[test]
fn renders_semantic_help_rows_without_truncating_narrow_controls() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
    ]);
    let selector = selector(&tree, Some("asst-1"));
    let lines: Vec<String> = selector
        .render(30)
        .iter()
        .map(|line| strip_ansi(line))
        .collect();
    let plain = lines.join("\n");
    for expected in ["branch", "copy", "filters", "cycle", "label time"] {
        assert!(plain.contains(expected), "missing {expected}: {plain}");
    }
    assert!(!plain.contains("..."));
    assert!(lines.iter().all(|line| visible_width(line) <= 30));
}

#[test]
fn copies_the_full_selected_message_with_ctrl_x() {
    setup(&[]);
    let message = format!("{}\nsecond line", "long message ".repeat(30));
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), &message),
    ]);
    let selector = selector(&tree, Some("asst-1"));
    let copied = Rc::new(RefCell::new(None));
    let captured = Rc::clone(&copied);
    selector.set_on_copy(Rc::new(move |text| {
        *captured.borrow_mut() = text.map(str::to_string);
    }));

    selector.handle_input("\x18");
    assert_eq!(copied.borrow().as_deref(), Some(message.as_str()));
}

#[test]
fn toggles_label_timestamps_for_labeled_nodes() {
    setup(&[]);
    let mut tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
    ]);
    tree[0].label = Some("checkpoint".to_string());
    tree[0].label_timestamp = Some("2026-03-28T14:32:00.000Z".to_string());
    let selector = selector(&tree, Some("asst-1"));
    let list = selector.get_tree_list();

    let rendered = output(list.as_ref(), 200);
    assert!(rendered.contains("[checkpoint]"));
    assert!(!rendered.contains("14:32"));
    assert!(!rendered.contains("[+label time]"));

    selector.handle_input("T");
    let rendered = output(list.as_ref(), 200);
    assert!(rendered.contains(&format!(
        "{} ",
        format_label_timestamp("2026-03-28T14:32:00.000Z")
    )));
    assert!(rendered.contains("[+label time]"));
}

#[test]
fn preserves_selection_through_an_empty_labeled_filter() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
        user_message("user-2", Some("asst-1"), "bye"),
        assistant_message("asst-2", Some("user-2"), "goodbye"),
    ]);
    let selector = selector(&tree, Some("asst-2"));
    let list = selector.get_tree_list();
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("asst-2"));

    selector.handle_input("\x0c");
    assert!(list.get_selected_entry_id().is_none());

    selector.handle_input("\x04");
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("asst-2"));
}

#[test]
fn preserves_selection_through_multiple_empty_filter_switches() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
    ]);
    let selector = selector(&tree, Some("asst-1"));
    let list = selector.get_tree_list();

    selector.handle_input("\x0c");
    assert!(list.get_selected_entry_id().is_none());
    selector.handle_input("\x0c");
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("asst-1"));
    selector.handle_input("\x0c");
    assert!(list.get_selected_entry_id().is_none());
    selector.handle_input("\x04");
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("asst-1"));
}

const UP: &str = "\x1b[A";
const DOWN: &str = "\x1b[B";
const CTRL_LEFT: &str = "\x1b[1;5D";
const CTRL_RIGHT: &str = "\x1b[1;5C";
const ALT_LEFT: &str = "\x1b[1;3D";
const ALT_RIGHT: &str = "\x1b[1;3C";

fn branching_tree() -> Vec<SessionTreeNode> {
    build_tree(vec![
        user_message("user-1", None, "first message"),
        assistant_message("asst-1", Some("user-1"), "response 1"),
        user_message("user-2", Some("asst-1"), "second message"),
        assistant_message("asst-2", Some("user-2"), "response 2"),
        user_message("user-3a", Some("asst-2"), "branch A start"),
        assistant_message("asst-3a", Some("user-3a"), "branch A response"),
        user_message("user-4a", Some("asst-3a"), "branch A deep"),
        assistant_message("asst-4a", Some("user-4a"), "branch A leaf"),
        user_message("user-3b", Some("asst-2"), "branch B start"),
        assistant_message("asst-3b", Some("user-3b"), "branch B response"),
        user_message("user-4b", Some("asst-3b"), "branch B deep"),
    ])
}

#[test]
fn ctrl_right_unfolds_then_jumps_to_the_branch_segment_leaf() {
    setup(&[]);
    let tree = branching_tree();
    let selector = selector(&tree, Some("asst-4a"));
    let list = selector.get_tree_list();

    selector.handle_input(CTRL_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3a"));
    selector.handle_input(CTRL_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3a"));

    selector.handle_input(DOWN);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3b"));
    selector.handle_input(UP);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3a"));

    selector.handle_input(CTRL_RIGHT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3a"));
    selector.handle_input(DOWN);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("asst-3a"));
    selector.handle_input(CTRL_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3a"));
    selector.handle_input(CTRL_RIGHT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("asst-4a"));
}

#[test]
fn alt_arrows_alias_fold_and_unfold_navigation() {
    setup(&[]);
    let tree = branching_tree();
    let selector = selector(&tree, Some("asst-4a"));
    let list = selector.get_tree_list();

    selector.handle_input(ALT_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3a"));
    selector.handle_input(ALT_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3a"));
    selector.handle_input(ALT_RIGHT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3a"));
    selector.handle_input(ALT_RIGHT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("asst-4a"));
}

#[test]
fn folding_the_root_hides_the_subtree_and_preserves_nested_folds() {
    setup(&[]);
    let tree = branching_tree();
    let selector = selector(&tree, Some("asst-4a"));
    let list = selector.get_tree_list();

    selector.handle_input(CTRL_LEFT);
    selector.handle_input(CTRL_LEFT);
    selector.handle_input(CTRL_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-1"));
    selector.handle_input(CTRL_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-1"));
    selector.handle_input(DOWN);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-1"));

    selector.handle_input(CTRL_RIGHT);
    selector.handle_input(CTRL_RIGHT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3a"));
    selector.handle_input(DOWN);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3b"));
}

#[test]
fn folds_and_navigates_a_non_active_branch() {
    setup(&[]);
    let tree = branching_tree();
    let selector = selector(&tree, Some("asst-4a"));
    let list = selector.get_tree_list();

    let mut found = false;
    for _ in 0..20 {
        selector.handle_input(DOWN);
        if list.get_selected_entry_id().as_deref() == Some("user-3b") {
            found = true;
            break;
        }
    }
    assert!(found);
    selector.handle_input(CTRL_RIGHT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-4b"));
    selector.handle_input(CTRL_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3b"));
    selector.handle_input(CTRL_LEFT);
    selector.handle_input(CTRL_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-1"));
}

#[test]
fn folds_and_navigates_multiple_roots() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "first root"),
        assistant_message("asst-1", Some("user-1"), "response 1"),
        user_message("user-2", None, "second root"),
        assistant_message("asst-2", Some("user-2"), "response 2"),
    ]);
    let selector = selector(&tree, Some("asst-1"));
    let list = selector.get_tree_list();

    assert_eq!(list.get_selected_entry_id().as_deref(), Some("asst-1"));
    selector.handle_input(CTRL_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-1"));
    selector.handle_input(CTRL_LEFT);
    selector.handle_input(DOWN);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-2"));
    selector.handle_input(CTRL_RIGHT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("asst-2"));
    selector.handle_input(CTRL_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-2"));
    selector.handle_input(CTRL_LEFT);
    selector.handle_input(CTRL_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-2"));
}

#[test]
fn folding_the_root_hides_descendants_filtered_out_of_the_view() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        tool_call_only_assistant("tool-asst-1", Some("user-1")),
        user_message("user-2", Some("tool-asst-1"), "follow up"),
        assistant_message("asst-2", Some("user-2"), "response"),
    ]);
    let selector = selector(&tree, Some("asst-2"));
    let list = selector.get_tree_list();

    selector.handle_input(CTRL_LEFT);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-1"));
    selector.handle_input(CTRL_LEFT);
    selector.handle_input(DOWN);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-1"));
}

#[test]
fn search_resets_fold_state() {
    setup(&[]);
    let tree = branching_tree();
    let selector = selector(&tree, Some("asst-4a"));
    let list = selector.get_tree_list();

    selector.handle_input(CTRL_LEFT);
    selector.handle_input(CTRL_LEFT);
    selector.handle_input(DOWN);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-3b"));
    selector.handle_input("b");
    selector.handle_input("\x1b");

    let mut current = String::new();
    for _ in 0..20 {
        selector.handle_input(DOWN);
        current = list.get_selected_entry_id().unwrap_or_default();
        if current == "user-3a" {
            break;
        }
    }
    assert_eq!(current, "user-3a");
    selector.handle_input(DOWN);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("asst-3a"));
}

#[test]
fn filter_mode_change_resets_fold_state() {
    setup(&[]);
    let tree = branching_tree();
    let selector = selector(&tree, Some("asst-4a"));
    let list = selector.get_tree_list();

    selector.handle_input(CTRL_LEFT);
    selector.handle_input(CTRL_LEFT);
    selector.handle_input("\x15");
    selector.handle_input("\x04");

    let mut current = String::new();
    for _ in 0..20 {
        selector.handle_input(DOWN);
        current = list.get_selected_entry_id().unwrap_or_default();
        if current == "user-3a" {
            break;
        }
    }
    assert_eq!(current, "user-3a");
    selector.handle_input(DOWN);
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("asst-3a"));
}

#[test]
fn empty_tree_renders_the_empty_state_and_ignores_navigation() {
    setup(&[]);
    let cancelled = Rc::new(std::cell::Cell::new(0));
    let captured = Rc::clone(&cancelled);
    let selector = TreeSelectorComponent::new(
        &[],
        None,
        24,
        TreeSelectorOptions {
            on_select: Rc::new(|_| {}),
            on_cancel: Rc::new(move || captured.set(captured.get() + 1)),
            on_label_change: None,
            initial_selected_id: None,
            initial_filter_mode: None,
        },
    );
    let list = selector.get_tree_list();
    let rendered = output(selector.as_ref(), 40);
    assert!(rendered.contains("No entries found"));
    assert!(rendered.contains("(0/0)"));
    assert!(list.get_selected_node().is_none());

    for key in [
        UP, DOWN, CTRL_LEFT, CTRL_RIGHT, "\x1b[6~", "\r", "\x18", "L", "T", "x",
    ] {
        selector.handle_input(key);
    }
    assert!(list.get_selected_node().is_none());
    selector.handle_input("\x1b");
    selector.handle_input("\x1b");
    assert_eq!(cancelled.get(), 1);
    assert!(!selector.render(10).is_empty());
}

#[test]
fn every_entry_kind_renders_its_row() {
    setup(&[("app.tree.filter.cycleBackward", &["ctrl+b"])]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        entry(json!({
            "type": "custom_message", "id": "custom-msg", "parentId": "user-1",
            "timestamp": "2026-01-01T00:00:00.000Z", "customType": "hint",
            "content": "custom body", "display": true
        })),
        entry(json!({
            "type": "compaction", "id": "compact", "parentId": "custom-msg",
            "timestamp": "2026-01-01T00:00:00.000Z", "summary": "summary",
            "tokensBefore": 12_345
        })),
        entry(json!({
            "type": "branch_summary", "id": "branch", "parentId": "compact",
            "timestamp": "2026-01-01T00:00:00.000Z", "fromId": "user-1",
            "summary": "branch body"
        })),
        model_change("model", Some("branch")),
        thinking_change("thinking", Some("model")),
        entry(json!({
            "type": "custom", "id": "custom", "parentId": "thinking",
            "timestamp": "2026-01-01T00:00:00.000Z", "customType": "state"
        })),
        entry(json!({
            "type": "label", "id": "label-entry", "parentId": "custom",
            "timestamp": "2026-01-01T00:00:00.000Z", "targetId": "user-1",
            "label": "bookmark"
        })),
        entry(json!({
            "type": "session_info", "id": "info", "parentId": "label-entry",
            "timestamp": "2026-01-01T00:00:00.000Z", "name": "named session"
        })),
        entry(json!({
            "type": "session_info", "id": "info-empty", "parentId": "info",
            "timestamp": "2026-01-01T00:00:00.000Z"
        })),
        FileEntry::Other(json!({"type": "mystery", "id": "raw", "parentId": "info-empty"})),
    ]);
    let selector = TreeSelectorComponent::new(
        &tree,
        Some("raw"),
        24,
        TreeSelectorOptions {
            on_select: Rc::new(|_| {}),
            on_cancel: Rc::new(|| {}),
            on_label_change: None,
            initial_selected_id: None,
            initial_filter_mode: Some(FilterMode::All),
        },
    );
    let rendered = output(selector.as_ref(), 200);
    for expected in [
        "user: hello",
        "[hint]: custom body",
        "[compaction: 12k tokens]",
        "[branch summary]: branch body",
        "[model: claude-sonnet-4]",
        "[thinking: high]",
        "[custom: state]",
        "[label: bookmark]",
        "[title: named session]",
        "[title: empty]",
        "[all]",
    ] {
        assert!(
            rendered.contains(expected),
            "missing {expected}: {rendered}"
        );
    }

    selector.handle_input("\x0f");
    assert!(!output(selector.as_ref(), 200).contains(" [no-tools]"));
    selector.handle_input("\x0f");
    assert!(output(selector.as_ref(), 200).contains("[no-tools]"));
    selector.handle_input("\x0f");
    assert!(output(selector.as_ref(), 200).contains("[user]"));
    selector.handle_input("\x0f");
    assert!(output(selector.as_ref(), 200).contains("[labeled]"));
    selector.handle_input("\x0f");
    assert!(output(selector.as_ref(), 200).contains("[all]"));
    selector.handle_input("\x02");
    assert!(output(selector.as_ref(), 200).contains("[labeled]"));
}

#[test]
fn tool_rows_reuse_the_tool_call_arguments() {
    setup(&[]);
    let calls = [
        (
            "read",
            json!({"path": "/tmp/readme", "offset": 2, "limit": 3}),
        ),
        ("write", json!({"path": "/tmp/out"})),
        ("edit", json!({"file_path": "/tmp/edit"})),
        ("bash", json!({"command": "echo hello"})),
        ("grep", json!({"pattern": "needle", "path": "/tmp"})),
        ("find", json!({"pattern": "*.rs"})),
        ("ls", json!({"path": "/tmp"})),
        (
            "custom",
            json!({"value": "a very long custom argument value that truncates"}),
        ),
        ("short", json!({"x": 1})),
    ];
    let content: Vec<Value> = calls
        .iter()
        .map(|(name, arguments)| {
            json!({"type": "toolCall", "id": format!("tc-{name}"), "name": name, "arguments": arguments})
        })
        .collect();
    let mut entries = vec![entry(json!({
        "type": "message", "id": "asst", "parentId": null,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {
            "role": "assistant", "content": content,
            "api": "anthropic-messages", "provider": "anthropic", "model": "claude-sonnet-4",
            "usage": {
                "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}
            },
            "stopReason": "toolUse", "timestamp": 0
        }
    }))];
    let mut parent = "asst".to_string();
    for (name, _) in &calls {
        let id = format!("result-{name}");
        entries.push(entry(json!({
            "type": "message", "id": id, "parentId": parent,
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {
                "role": "toolResult", "toolCallId": format!("tc-{name}"), "toolName": name,
                "content": [{"type": "text", "text": format!("output {name}")}],
                "isError": false, "timestamp": 0
            }
        })));
        parent = id;
    }
    let tree = build_tree(entries);
    let selector = TreeSelectorComponent::new(
        &tree,
        Some(&parent),
        60,
        TreeSelectorOptions {
            on_select: Rc::new(|_| {}),
            on_cancel: Rc::new(|| {}),
            on_label_change: None,
            initial_selected_id: None,
            initial_filter_mode: Some(FilterMode::All),
        },
    );
    let rendered = output(selector.as_ref(), 240);
    for expected in [
        "[read: /tmp/readme:2-4]",
        "[write: /tmp/out]",
        "[edit: /tmp/edit]",
        "[bash: echo hello]",
        "[grep: /needle/ in /tmp]",
        "[find: *.rs in .]",
        "[ls: /tmp]",
        "[custom: {",
        "[short: {\"x\":1}]",
    ] {
        assert!(
            rendered.contains(expected),
            "missing {expected}: {rendered}"
        );
    }
}

#[test]
fn label_input_edits_cancels_and_clears_labels() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
    ]);
    let changes = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&changes);
    let selector = TreeSelectorComponent::new(
        &tree,
        Some("asst-1"),
        24,
        TreeSelectorOptions {
            on_select: Rc::new(|_| {}),
            on_cancel: Rc::new(|| {}),
            on_label_change: Some(Rc::new(move |id, label| {
                captured
                    .borrow_mut()
                    .push((id.to_string(), label.map(str::to_string)));
            })),
            initial_selected_id: Some("user-1".to_string()),
            initial_filter_mode: None,
        },
    );
    selector.set_focused(true);
    selector.handle_input("L");
    assert!(output(selector.as_ref(), 80).contains("Label (empty to remove):"));

    selector.handle_input("checkpoint");
    selector.handle_input("\r");
    assert_eq!(
        changes.borrow().as_slice(),
        &[("user-1".to_string(), Some("checkpoint".to_string()))]
    );
    assert!(output(selector.as_ref(), 120).contains("[checkpoint]"));

    selector.handle_input("L");
    selector.handle_input("\x1b");
    assert_eq!(changes.borrow().len(), 1);
    assert!(output(selector.as_ref(), 120).contains("Session Tree"));

    selector.handle_input("L");
    for _ in 0.."checkpoint".len() {
        selector.handle_input("\x7f");
    }
    selector.handle_input("\r");
    assert_eq!(
        changes.borrow().as_slice(),
        &[
            ("user-1".to_string(), Some("checkpoint".to_string())),
            ("user-1".to_string(), None),
        ]
    );
    assert!(!output(selector.as_ref(), 120).contains("[checkpoint]"));
}

#[test]
fn label_timestamp_shapes_follow_the_utc_calendar() {
    let now = parse_iso8601_millis("2026-10-26T12:00:00.000Z").expect("now");
    assert_eq!(
        format_label_timestamp_at("2026-10-26T09:05:00.000Z", now),
        "09:05"
    );
    assert_eq!(
        format_label_timestamp_at("2026-03-28T14:32:00.000Z", now),
        "3/28 14:32"
    );
    assert_eq!(
        format_label_timestamp_at("2025-12-31T23:59:00.000Z", now),
        "25/12/31 23:59"
    );
    assert_eq!(format_label_timestamp_at("not-a-date", now), "not-a-date");
    assert_eq!(
        format_label_timestamp_at("1969-07-20T20:17:00.000Z", now),
        "69/7/20 20:17"
    );
}

#[test]
fn search_filters_by_label_and_entry_text_and_clears_before_cancel() {
    setup(&[]);
    let mut tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
        user_message("user-2", Some("asst-1"), "needle in a haystack"),
    ]);
    tree[0].label = Some("start".to_string());
    let cancelled = Rc::new(std::cell::Cell::new(0));
    let captured = Rc::clone(&cancelled);
    let selector = TreeSelectorComponent::new(
        &tree,
        Some("user-2"),
        24,
        TreeSelectorOptions {
            on_select: Rc::new(|_| {}),
            on_cancel: Rc::new(move || captured.set(captured.get() + 1)),
            on_label_change: None,
            initial_selected_id: None,
            initial_filter_mode: None,
        },
    );
    let list = selector.get_tree_list();

    selector.handle_input("needle");
    assert_eq!(list.get_search_query(), "needle");
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-2"));
    assert!(output(selector.as_ref(), 120).contains("needle"));

    selector.handle_input("\x7f");
    assert_eq!(list.get_search_query(), "needl");
    selector.handle_input("\x1b");
    assert_eq!(list.get_search_query(), "");
    assert_eq!(cancelled.get(), 0);
    selector.handle_input("start");
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-1"));
    selector.handle_input("\x1b");
    selector.handle_input("\x1b");
    assert_eq!(cancelled.get(), 1);
}

#[test]
fn page_and_confirm_report_the_selected_entry() {
    setup(&[]);
    let tree = branching_tree();
    let selected = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&selected);
    let selector = TreeSelectorComponent::new(
        &tree,
        Some("asst-4a"),
        10,
        TreeSelectorOptions {
            on_select: Rc::new(move |id| captured.borrow_mut().push(id.to_string())),
            on_cancel: Rc::new(|| {}),
            on_label_change: None,
            initial_selected_id: None,
            initial_filter_mode: None,
        },
    );
    selector.handle_input("\x1b[6~");
    selector.handle_input("\x1b[5~");
    let before_confirm = selected_id(&selector).unwrap_or_default();
    selector.handle_input("\r");
    assert_eq!(selected.borrow().as_slice(), &[before_confirm]);
}

#[test]
fn copy_returns_none_for_rows_without_text() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        model_change("model-1", Some("user-1")),
    ]);
    let selector = selector(&tree, Some("model-1"));
    let copied = Rc::new(RefCell::new(Some("sentinel".to_string())));
    let captured = Rc::clone(&copied);
    selector.set_on_copy(Rc::new(move |text| {
        *captured.borrow_mut() = text.map(str::to_string);
    }));
    selector.handle_input("\x18");
    assert!(copied.borrow().is_none());
}

#[test]
fn copy_falls_back_to_assistant_error_and_reads_custom_kinds() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        entry(json!({
            "type": "message", "id": "error", "parentId": "user-1",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {
                "role": "assistant", "content": [],
                "api": "anthropic-messages", "provider": "anthropic", "model": "claude-sonnet-4",
                "usage": {
                    "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                    "totalTokens": 0,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}
                },
                "stopReason": "error", "errorMessage": "boom", "timestamp": 0
            }
        })),
        entry(json!({
            "type": "message", "id": "aborted", "parentId": "error",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {
                "role": "assistant", "content": [],
                "api": "anthropic-messages", "provider": "anthropic", "model": "claude-sonnet-4",
                "usage": {
                    "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                    "totalTokens": 0,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}
                },
                "stopReason": "aborted", "timestamp": 0
            }
        })),
        entry(json!({
            "type": "message", "id": "bash", "parentId": "aborted",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {"role": "bashExecution", "command": "ls -la", "timestamp": 0}
        })),
        entry(json!({
            "type": "compaction", "id": "compact", "parentId": "bash",
            "timestamp": "2026-01-01T00:00:00.000Z", "summary": "compact summary",
            "tokensBefore": 500
        })),
        entry(json!({
            "type": "branch_summary", "id": "branch", "parentId": "compact",
            "timestamp": "2026-01-01T00:00:00.000Z", "fromId": "user-1",
            "summary": "branch summary text"
        })),
        entry(json!({
            "type": "custom_message", "id": "custom-msg", "parentId": "branch",
            "timestamp": "2026-01-01T00:00:00.000Z", "customType": "hint",
            "content": [{"type": "text", "text": "custom text"}], "display": true
        })),
    ]);
    let selector = TreeSelectorComponent::new(
        &tree,
        Some("custom-msg"),
        60,
        TreeSelectorOptions {
            on_select: Rc::new(|_| {}),
            on_cancel: Rc::new(|| {}),
            on_label_change: None,
            initial_selected_id: Some("error".to_string()),
            initial_filter_mode: Some(FilterMode::All),
        },
    );
    let copied = Rc::new(RefCell::new(Vec::new()));
    let captured = Rc::clone(&copied);
    selector.set_on_copy(Rc::new(move |text| {
        captured.borrow_mut().push(text.map(str::to_string));
    }));
    let list = selector.get_tree_list();

    let rendered = output(selector.as_ref(), 160);
    assert!(rendered.contains("boom"));
    assert!(rendered.contains("(aborted)"));
    assert!(rendered.contains("[bash]: ls -la"));
    assert!(rendered.contains("[compaction: 1k tokens]"));
    assert!(rendered.contains("[branch summary]: branch summary text"));
    assert!(rendered.contains("[hint]: custom text"));

    selector.handle_input("\x18");
    selector.handle_input(DOWN);
    selector.handle_input("\x18");
    assert!(copied.borrow()[1].is_none());
    for expected in [
        "ls -la",
        "compact summary",
        "branch summary text",
        "custom text",
    ] {
        selector.handle_input(DOWN);
        selector.handle_input("\x18");
        assert_eq!(
            copied.borrow().last().and_then(|text| text.as_deref()),
            Some(expected)
        );
    }
    assert!(list.get_selected_node().is_some());
}

#[test]
fn horizontal_viewport_keeps_rows_within_the_requested_width() {
    setup(&[]);
    let message = "word ".repeat(80);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), &message),
        user_message("user-2", Some("asst-1"), "follow up"),
    ]);
    let selector = selector(&tree, Some("user-2"));
    let list = selector.get_tree_list();
    let lines: Vec<String> = selector
        .render(20)
        .iter()
        .map(|line| strip_ansi(line))
        .collect();
    assert!(lines.iter().all(|line| visible_width(line) <= 20));
    let rendered = lines.join("\n");
    assert!(rendered.contains("user: hello"));
    assert!(rendered.contains("follow up"));

    let rendered = output(list.as_ref(), 12);
    assert!(rendered.lines().all(|line| visible_width(line) <= 12));
}

#[test]
fn initial_selection_ids_and_labels_round_trip() {
    setup(&[]);
    let mut tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
        user_message("user-2", Some("asst-1"), "bye"),
    ]);
    tree[2].label = Some("last".to_string());
    let selector = TreeSelectorComponent::new(
        &tree,
        Some("asst-1"),
        24,
        TreeSelectorOptions {
            on_select: Rc::new(|_| {}),
            on_cancel: Rc::new(|| {}),
            on_label_change: None,
            initial_selected_id: Some("missing".to_string()),
            initial_filter_mode: None,
        },
    );
    let list = selector.get_tree_list();
    assert!(list.wants_input());
    assert_eq!(list.get_selected_entry_id().as_deref(), Some("user-2"));
    assert_eq!(list.get_selected_label().as_deref(), Some("last"));
    let node = list.get_selected_node().expect("selected");
    assert_eq!(node.entry.entry_id(), Some("user-2"));
    assert_eq!(node.label.as_deref(), Some("last"));
    assert!(node.children.is_empty());

    list.update_node_label("missing", Some("x"), Some("2026-01-01T00:00:00.000Z"));
    list.update_node_label("user-1", Some("first"), Some("2026-01-01T00:00:00.000Z"));
    assert_eq!(
        list.get_selected_label().as_deref(),
        Some("last"),
        "only the selected node's label is read"
    );
}

#[test]
fn component_focus_children_and_invalidate_delegate_to_the_container() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
    ]);
    let selector = selector(&tree, Some("asst-1"));
    assert!(selector.wants_input());
    assert!(!selector.is_focused());
    selector.set_focused(true);
    assert!(selector.is_focused());
    assert!(!selector.children().is_empty());
    assert!(format!("{selector:?}").contains("TreeSelectorComponent"));
    assert!(format!("{:?}", selector.get_tree_list()).contains("TreeList"));
    selector.invalidate();
    assert!(selector.as_focusable().is_some());

    selector.handle_input("L");
    assert!(output(selector.as_ref(), 80).contains("Label (empty to remove):"));
    selector.set_focused(false);
    selector.handle_input("\x1b");
    assert!(output(selector.as_ref(), 80).contains("Session Tree"));
}

#[test]
fn direct_filter_toggles_and_small_content_paths() {
    setup(&[]);
    let tree = build_tree(vec![
        entry(json!({
            "type": "message", "id": "user-blocks", "parentId": null,
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {
                "role": "user", "timestamp": 0,
                "content": [{"type": "text", "text": "block text"}, {"type": "image", "data": "AA==", "mimeType": "image/png"}]
            }
        })),
        entry(json!({
            "type": "message", "id": "assistant-thinking", "parentId": "user-blocks",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "answer"}],
                "api": "anthropic-messages", "provider": "anthropic", "model": "claude-sonnet-4",
                "usage": {
                    "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                    "totalTokens": 0,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}
                },
                "stopReason": "stop", "timestamp": 0
            }
        })),
        tool_call_only_assistant("tool-only", Some("assistant-thinking")),
    ]);
    let selector = selector(&tree, Some("tool-only"));
    let list = selector.get_tree_list();
    let rendered = output(selector.as_ref(), 120);
    assert!(rendered.contains("block text"));
    assert!(rendered.contains("answer"));
    assert!(!rendered.contains("tc-tool-only"));

    selector.handle_input("\x14");
    assert_eq!(list.filter_mode(), FilterMode::NoTools);
    selector.handle_input("\x14");
    assert_eq!(list.filter_mode(), FilterMode::Default);
    selector.handle_input("\x01");
    assert_eq!(list.filter_mode(), FilterMode::All);
    selector.handle_input("\x01");
    assert_eq!(list.filter_mode(), FilterMode::Default);
    selector.handle_input("\x15");
    assert_eq!(list.filter_mode(), FilterMode::UserOnly);
    selector.handle_input("\x04");
    assert_eq!(list.filter_mode(), FilterMode::Default);
}

#[test]
fn search_reaches_every_entry_kind() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        entry(json!({
            "type": "custom_message", "id": "custom-msg", "parentId": "user-1",
            "timestamp": "2026-01-01T00:00:00.000Z", "customType": "hint",
            "content": "custom body", "display": true
        })),
        entry(json!({
            "type": "compaction", "id": "compact", "parentId": "custom-msg",
            "timestamp": "2026-01-01T00:00:00.000Z", "summary": "summary",
            "tokensBefore": 500
        })),
        entry(json!({
            "type": "branch_summary", "id": "branch", "parentId": "compact",
            "timestamp": "2026-01-01T00:00:00.000Z", "fromId": "user-1",
            "summary": "branch body"
        })),
        model_change("model", Some("branch")),
        thinking_change("thinking", Some("model")),
        entry(json!({
            "type": "custom", "id": "custom", "parentId": "thinking",
            "timestamp": "2026-01-01T00:00:00.000Z", "customType": "state"
        })),
        entry(json!({
            "type": "label", "id": "label-entry", "parentId": "custom",
            "timestamp": "2026-01-01T00:00:00.000Z", "targetId": "user-1",
            "label": "bookmark"
        })),
        entry(json!({
            "type": "session_info", "id": "info", "parentId": "label-entry",
            "timestamp": "2026-01-01T00:00:00.000Z", "name": "named session"
        })),
    ]);
    let selector = TreeSelectorComponent::new(
        &tree,
        Some("info"),
        24,
        TreeSelectorOptions {
            on_select: Rc::new(|_| {}),
            on_cancel: Rc::new(|| {}),
            on_label_change: None,
            initial_selected_id: None,
            initial_filter_mode: Some(FilterMode::All),
        },
    );
    let list = selector.get_tree_list();
    for (query, expected) in [
        ("custom body", "custom-msg"),
        ("compaction", "compact"),
        ("branch body", "branch"),
        ("claude", "model"),
        ("high", "thinking"),
        ("state", "custom"),
        ("bookmark", "label-entry"),
        ("named session", "info"),
    ] {
        selector.handle_input(query);
        assert_eq!(
            list.get_selected_entry_id().as_deref(),
            Some(expected),
            "query {query}"
        );
        selector.handle_input("\x1b");
    }
}

#[test]
fn selected_root_reconstructs_its_children() {
    setup(&[]);
    let tree = build_tree(vec![
        user_message("user-1", None, "hello"),
        assistant_message("asst-1", Some("user-1"), "hi"),
    ]);
    let selector = TreeSelectorComponent::new(
        &tree,
        Some("asst-1"),
        24,
        TreeSelectorOptions {
            on_select: Rc::new(|_| {}),
            on_cancel: Rc::new(|| {}),
            on_label_change: None,
            initial_selected_id: Some("user-1".to_string()),
            initial_filter_mode: None,
        },
    );
    let node = selector.get_tree_list().get_selected_node().expect("root");
    assert_eq!(node.entry.entry_id(), Some("user-1"));
    assert_eq!(node.children.len(), 1);
    assert_eq!(node.children[0].entry.entry_id(), Some("asst-1"));
}

#[test]
fn assistant_without_content_or_error_renders_no_content() {
    setup(&[]);
    let tree = build_tree(vec![entry(json!({
        "type": "message", "id": "empty", "parentId": null,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {
            "role": "assistant", "content": [],
            "api": "anthropic-messages", "provider": "anthropic", "model": "claude-sonnet-4",
            "usage": {
                "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}
            },
            "stopReason": "stop", "timestamp": 0
        }
    }))]);
    let selector = selector(&tree, Some("empty"));
    let rendered = output(selector.as_ref(), 120);
    assert!(rendered.contains("(no content)"));
}

#[test]
fn horizontal_viewport_pans_a_deep_selected_row() {
    setup(&[]);
    let tree = branching_tree();
    let selector = selector(&tree, Some("asst-4a"));
    let lines: Vec<String> = selector
        .render(18)
        .iter()
        .map(|line| strip_ansi(line))
        .collect();
    assert!(lines.iter().all(|line| visible_width(line) <= 18));
    assert!(lines.join("\n").contains("assistant:"));
}

#[test]
fn help_wraps_when_a_single_item_exceeds_the_width() {
    setup(&[]);
    let tree = build_tree(vec![user_message("user-1", None, "hello")]);
    let selector = selector(&tree, Some("user-1"));
    let lines = selector.render(5);
    assert!(lines.iter().all(|line| visible_width(line) <= 5));
}

#[test]
fn messages_without_payloads_render_without_panicking() {
    setup(&[]);
    let tree = build_tree(vec![entry(json!({
        "type": "message", "id": "empty-message", "parentId": null,
        "timestamp": "2026-01-01T00:00:00.000Z"
    }))]);
    let selector = selector(&tree, Some("empty-message"));
    let rendered = output(selector.as_ref(), 80);
    assert!(rendered.contains("(1/1)"));
}

// __APPEND__

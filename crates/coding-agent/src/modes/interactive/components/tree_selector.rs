//! The session tree selector, upstream's
//! `src/modes/interactive/components/tree-selector.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#208).
//!
//! The component takes the resolved [`SessionTreeNode`] snapshot and reports
//! navigation, copy, and label decisions; forking and persistence stay with
//! the interactive frame.
//!
//! Restatements: label timestamps render in UTC because std carries no
//! local-time conversion; the empty-tree cancel timer is host-owned (the
//! frame checks the tree before mounting); [`TreeSelectorComponent::new`]
//! takes an options bundle for upstream's trailing optional parameters.

use std::cell::{Cell, RefCell};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::rc::Rc;

use pi_agent_core::harness::session::jsonl::codec::parse_iso8601_millis;
use pi_agent_core::types::AgentMessage;
use pi_ai::types::{
    AssistantBlock, AssistantMessage, Message, StopReason, ToolResultBlock, UserBlock, UserContent,
};
use pi_tui::components::input::Input;
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container, Focusable};
use pi_tui::utils::{slice_by_column, truncate_to_width, visible_width, wrap_text_with_ansi};

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::{KeyTextFormatOptions, format_key_text, key_hint};
use crate::modes::interactive::theme::theme::{bold, italic, theme};
use crate::session_manager::entries::MessageEntry;
use crate::session_manager::{FileEntry, SessionEntry, SessionTreeNode, now_iso8601, now_millis};
use crate::tools::render_utils::shorten_path;

/// One ancestor branch point's gutter.
#[derive(Clone, Debug, PartialEq, Eq)]
struct GutterInfo {
    /// The display indent level where the connector was shown.
    position: usize,
    /// Whether to paint `│` (true) or a space.
    show: bool,
}

/// One flattened tree row.
#[derive(Clone, Debug)]
struct FlatNode {
    /// Index into the owned tree arena.
    node: usize,
    /// Indentation level; each level is three characters.
    indent: usize,
    /// Whether to show the `├─`/`└─` connector.
    show_connector: bool,
    /// Whether this row is its parent's last child.
    is_last: bool,
    /// Gutter state for each ancestor branch point.
    gutters: Vec<GutterInfo>,
    /// True for a root under the virtual branching root.
    is_virtual_root_child: bool,
}

/// One row handed to the horizontal viewport renderer.
struct HorizontalViewportRow {
    gutter: String,
    body: String,
    anchor_col: usize,
    body_width: usize,
    is_selected: bool,
}

/// The tree-gutter width the horizontal viewport always keeps visible.
const TREE_GUTTER_WIDTH: usize = 2;
/// The minimum selected-row content kept visible after its anchor.
const MIN_VISIBLE_ANCHOR_CONTENT_WIDTH: usize = 4;
/// The maximum selected-row content kept visible after its anchor.
const MAX_VISIBLE_ANCHOR_CONTENT_WIDTH: usize = 20;
/// The minimum anchor context kept visible when panning.
const MIN_ANCHOR_CONTEXT_WIDTH: usize = 2;
/// The maximum anchor context kept visible when panning.
const MAX_ANCHOR_CONTEXT_WIDTH: usize = 12;

/// Filter mode for tree display, upstream's `FilterMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilterMode {
    /// Hide settings/bookkeeping entries.
    #[default]
    Default,
    /// The default view minus tool results.
    NoTools,
    /// User messages only.
    UserOnly,
    /// Labeled entries only.
    LabeledOnly,
    /// Show every entry.
    All,
}

/// A tool call extracted from an assistant message, for tool-result rows.
#[derive(Clone, Debug)]
struct ToolCallInfo {
    name: String,
    arguments: serde_json::Map<String, serde_json::Value>,
}

/// One node in the owned tree arena.
struct OwnedNode {
    entry: FileEntry,
    parent: Option<usize>,
    children: Vec<usize>,
    label: RefCell<Option<String>>,
    label_timestamp: RefCell<Option<String>>,
}

/// The owned tree and the flattened rows derived from it.
struct OwnedTree {
    nodes: Vec<OwnedNode>,
    roots: Vec<usize>,
    id_index: HashMap<String, usize>,
}

impl OwnedTree {
    /// Clone the borrowed tree into stable indices, parents before children.
    fn new(tree: &[SessionTreeNode]) -> Self {
        struct Pending<'a> {
            node: &'a SessionTreeNode,
            parent: Option<usize>,
        }

        let mut nodes: Vec<OwnedNode> = Vec::new();
        let mut roots: Vec<usize> = Vec::new();
        let mut stack: Vec<Pending<'_>> = tree
            .iter()
            .rev()
            .map(|node| Pending { node, parent: None })
            .collect();

        while let Some(pending) = stack.pop() {
            let index = nodes.len();
            nodes.push(OwnedNode {
                entry: pending.node.entry.clone(),
                parent: pending.parent,
                children: Vec::new(),
                label: RefCell::new(pending.node.label.clone()),
                label_timestamp: RefCell::new(pending.node.label_timestamp.clone()),
            });
            if let Some(parent) = pending.parent {
                nodes[parent].children.push(index);
            } else {
                roots.push(index);
            }
            for child in pending.node.children.iter().rev() {
                stack.push(Pending {
                    node: child,
                    parent: Some(index),
                });
            }
        }

        let mut id_index = HashMap::new();
        for (index, node) in nodes.iter().enumerate() {
            if let Some(id) = node.entry.entry_id() {
                id_index.insert(id.to_string(), index);
            }
        }

        Self {
            nodes,
            roots,
            id_index,
        }
    }

    fn node(&self, index: usize) -> Option<&OwnedNode> {
        self.nodes.get(index)
    }

    fn entry_id(&self, index: usize) -> Option<&str> {
        self.node(index).and_then(|node| node.entry.entry_id())
    }
}

impl std::fmt::Debug for TreeList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeList")
            .field("nodes", &self.tree.nodes.len())
            .field("filtered", &self.filtered_nodes.borrow().len())
            .finish_non_exhaustive()
    }
}

/// A confirm callback carrying the selected entry id.
type SelectCallback = Rc<dyn Fn(&str)>;
/// A dismissal callback.
type CancelCallback = Rc<dyn Fn()>;
/// A copy callback carrying the selected text.
type CopyCallback = Rc<dyn Fn(Option<&str>)>;
/// A label-edit callback carrying the entry id and current label.
type LabelCallback = Rc<dyn Fn(&str, Option<&str>)>;
/// One flatten/recalculate stack row.
type StackEntry = (usize, usize, bool, bool, bool, Vec<GutterInfo>, bool);

/// The list of tree rows, upstream's `TreeList`.
pub struct TreeList {
    tree: OwnedTree,
    flat_nodes: RefCell<Vec<FlatNode>>,
    filtered_nodes: RefCell<Vec<usize>>,
    selected_index: Cell<usize>,
    current_leaf_id: Option<String>,
    current_leaf_node: Option<usize>,
    max_visible_lines: usize,
    filter_mode: Cell<FilterMode>,
    search_query: RefCell<String>,
    tool_call_map: RefCell<HashMap<String, ToolCallInfo>>,
    multiple_roots: Cell<bool>,
    show_label_timestamps: Cell<bool>,
    active_path: RefCell<HashSet<usize>>,
    visible_parent_map: RefCell<HashMap<usize, Option<usize>>>,
    visible_children_map: RefCell<HashMap<Option<usize>, Vec<usize>>>,
    last_selected_node: Cell<Option<usize>>,
    folded_nodes: RefCell<HashSet<usize>>,
    parser: KeyParser,
    /// Called with the selected entry id on confirm.
    pub on_select: RefCell<Option<SelectCallback>>,
    /// Called when the list is dismissed.
    pub on_cancel: RefCell<Option<CancelCallback>>,
    /// Called with the copied text, or `None` when the row has none.
    pub on_copy: RefCell<Option<CopyCallback>>,
    /// Called when the label editor opens.
    pub on_label_edit: RefCell<Option<LabelCallback>>,
}

impl TreeList {
    fn new(
        tree: &[SessionTreeNode],
        current_leaf_id: Option<&str>,
        max_visible_lines: usize,
        initial_selected_id: Option<&str>,
        initial_filter_mode: Option<FilterMode>,
    ) -> Self {
        let owned = OwnedTree::new(tree);
        let current_leaf_node = current_leaf_id.and_then(|id| owned.id_index.get(id).copied());
        let multiple_roots = owned.roots.len() > 1;
        let list = Self {
            tree: owned,
            flat_nodes: RefCell::new(Vec::new()),
            filtered_nodes: RefCell::new(Vec::new()),
            selected_index: Cell::new(0),
            current_leaf_id: current_leaf_id.map(str::to_string),
            current_leaf_node,
            max_visible_lines,
            filter_mode: Cell::new(initial_filter_mode.unwrap_or_default()),
            search_query: RefCell::new(String::new()),
            tool_call_map: RefCell::new(HashMap::new()),
            multiple_roots: Cell::new(multiple_roots),
            show_label_timestamps: Cell::new(false),
            active_path: RefCell::new(HashSet::new()),
            visible_parent_map: RefCell::new(HashMap::new()),
            visible_children_map: RefCell::new(HashMap::new()),
            last_selected_node: Cell::new(None),
            folded_nodes: RefCell::new(HashSet::new()),
            parser: KeyParser::new(),
            on_select: RefCell::new(None),
            on_cancel: RefCell::new(None),
            on_copy: RefCell::new(None),
            on_label_edit: RefCell::new(None),
        };
        let flat = list.flatten_tree();
        *list.flat_nodes.borrow_mut() = flat;
        list.build_active_path();
        list.apply_filter();

        let target = initial_selected_id
            .and_then(|id| list.tree.id_index.get(id).copied())
            .or(current_leaf_node);
        let index = list.nearest_visible_from(target);
        list.selected_index.set(index);
        if let Some(&flat_index) = list.filtered_nodes.borrow().get(index) {
            list.last_selected_node.set(
                list.flat_nodes
                    .borrow()
                    .get(flat_index)
                    .map(|flat| flat.node),
            );
        }
        list
    }

    /// The current search text.
    #[must_use]
    pub fn get_search_query(&self) -> String {
        self.search_query.borrow().clone()
    }

    /// The selected node, reconstructed from the owned tree.
    #[must_use]
    pub fn get_selected_node(&self) -> Option<SessionTreeNode> {
        let flat_index = *self
            .filtered_nodes
            .borrow()
            .get(self.selected_index.get())?;
        let node = self.flat_nodes.borrow().get(flat_index)?.node;
        self.to_session_tree_node(node)
    }

    /// The selected entry id, when the row carries one.
    #[must_use]
    pub fn get_selected_entry_id(&self) -> Option<String> {
        let flat_index = *self
            .filtered_nodes
            .borrow()
            .get(self.selected_index.get())?;
        let node = self.flat_nodes.borrow().get(flat_index)?.node;
        self.tree.entry_id(node).map(str::to_string)
    }

    /// The selected row's label, when it has one.
    #[must_use]
    pub fn get_selected_label(&self) -> Option<String> {
        let flat_index = *self
            .filtered_nodes
            .borrow()
            .get(self.selected_index.get())?;
        let node = self.flat_nodes.borrow().get(flat_index)?.node;
        self.tree
            .node(node)
            .and_then(|node| node.label.borrow().clone())
    }

    /// The active filter mode.
    #[must_use]
    pub const fn filter_mode(&self) -> FilterMode {
        self.filter_mode.get()
    }

    /// Replace a node's label and timestamp.
    pub fn update_node_label(
        &self,
        entry_id: &str,
        label: Option<&str>,
        label_timestamp: Option<&str>,
    ) {
        let Some(index) = self.tree.id_index.get(entry_id).copied() else {
            return;
        };
        let Some(node) = self.tree.nodes.get(index) else {
            return;
        };
        *node.label.borrow_mut() = label.map(str::to_string);
        *node.label_timestamp.borrow_mut() =
            label.map(|_| label_timestamp.map_or_else(now_iso8601, str::to_string));
    }

    /// Render the selected row's copy text through `on_copy`.
    pub fn copy_selected(&self) {
        let text = self
            .get_selected_node()
            .and_then(|node| Self::get_entry_copy_text(&node));
        let callback = self.on_copy.borrow().clone();
        if let Some(callback) = callback {
            callback(text.as_deref());
        }
    }

    fn flatten_tree(&self) -> Vec<FlatNode> {
        let multiple_roots = self.multiple_roots.get();
        let mut contains_active: HashMap<usize, bool> = HashMap::new();
        for index in (0..self.tree.nodes.len()).rev() {
            let mut has = self.current_leaf_node == Some(index);
            if let Some(node) = self.tree.nodes.get(index) {
                for child in &node.children {
                    if contains_active.get(child).copied().unwrap_or(false) {
                        has = true;
                    }
                }
            }
            contains_active.insert(index, has);
        }

        let mut ordered_roots = self.tree.roots.clone();
        ordered_roots
            .sort_by_key(|index| Reverse(contains_active.get(index).copied().unwrap_or(false)));

        let mut result = Vec::new();
        let mut tool_calls = HashMap::new();
        let mut stack: Vec<StackEntry> = Vec::new();
        for (position, index) in ordered_roots.iter().enumerate().rev() {
            let is_last = position == ordered_roots.len() - 1;
            stack.push((
                *index,
                usize::from(multiple_roots),
                multiple_roots,
                multiple_roots,
                is_last,
                Vec::new(),
                multiple_roots,
            ));
        }

        while let Some((
            node_index,
            indent,
            just_branched,
            show_connector,
            is_last,
            gutters,
            virtual_root,
        )) = stack.pop()
        {
            let Some(node) = self.tree.nodes.get(node_index) else {
                continue;
            };
            extract_tool_calls(&node.entry, &mut tool_calls);
            let children = node.children.clone();
            result.push(FlatNode {
                node: node_index,
                indent,
                show_connector,
                is_last,
                gutters: gutters.clone(),
                is_virtual_root_child: virtual_root,
            });

            let multiple_children = children.len() > 1;
            let mut ordered_children = children;
            ordered_children
                .sort_by_key(|child| Reverse(contains_active.get(child).copied().unwrap_or(false)));
            let child_indent = if multiple_children || (just_branched && indent > 0) {
                indent + 1
            } else {
                indent
            };
            let connector_displayed = show_connector && !virtual_root;
            let current_display_indent = if multiple_roots {
                indent.saturating_sub(1)
            } else {
                indent
            };
            let connector_position = current_display_indent.saturating_sub(1);
            let mut child_gutters = gutters;
            if connector_displayed {
                child_gutters.push(GutterInfo {
                    position: connector_position,
                    show: !is_last,
                });
            }
            for (position, child) in ordered_children.iter().enumerate().rev() {
                let child_is_last = position == ordered_children.len() - 1;
                stack.push((
                    *child,
                    child_indent,
                    multiple_children,
                    multiple_children,
                    child_is_last,
                    child_gutters.clone(),
                    false,
                ));
            }
        }
        *self.tool_call_map.borrow_mut() = tool_calls;
        result
    }

    fn build_active_path(&self) {
        let mut active = HashSet::new();
        let mut current = self.current_leaf_node;
        while let Some(index) = current {
            active.insert(index);
            current = self.tree.node(index).and_then(|node| node.parent);
        }
        *self.active_path.borrow_mut() = active;
    }

    fn apply_filter(&self) {
        if !self.filtered_nodes.borrow().is_empty()
            && let Some(&flat_index) = self.filtered_nodes.borrow().get(self.selected_index.get())
            && let Some(flat) = self.flat_nodes.borrow().get(flat_index)
        {
            self.last_selected_node.set(Some(flat.node));
        }

        let search_tokens: Vec<String> = self
            .search_query
            .borrow()
            .to_lowercase()
            .split_whitespace()
            .map(str::to_string)
            .collect();
        let flat_nodes = self.flat_nodes.borrow();
        let mut filtered: Vec<usize> = Vec::new();
        for (flat_index, flat) in flat_nodes.iter().enumerate() {
            let Some(node) = self.tree.node(flat.node) else {
                continue;
            };
            let is_current_leaf = self
                .current_leaf_id
                .as_deref()
                .is_some_and(|id| node.entry.entry_id() == Some(id));
            if !is_current_leaf && hides_tool_call_only_assistant(&node.entry) {
                continue;
            }

            let is_settings = is_settings_entry(&node.entry);
            let passes_filter = match self.filter_mode.get() {
                FilterMode::UserOnly => is_user_message(&node.entry),
                FilterMode::NoTools => !is_settings && !is_tool_result(&node.entry),
                FilterMode::LabeledOnly => node.label.borrow().is_some(),
                FilterMode::All => true,
                FilterMode::Default => !is_settings,
            };
            if !passes_filter {
                continue;
            }

            if !search_tokens.is_empty() {
                let searchable = get_searchable_text(node).to_lowercase();
                if !search_tokens.iter().all(|token| searchable.contains(token)) {
                    continue;
                }
            }
            filtered.push(flat_index);
        }

        if !self.folded_nodes.borrow().is_empty() {
            let mut skip = HashSet::new();
            for flat in flat_nodes.iter() {
                let Some(node) = self.tree.node(flat.node) else {
                    continue;
                };
                if let Some(parent) = node.parent
                    && (self.folded_nodes.borrow().contains(&parent) || skip.contains(&parent))
                {
                    skip.insert(flat.node);
                }
            }
            filtered.retain(|&flat_index| {
                flat_nodes
                    .get(flat_index)
                    .is_some_and(|flat| !skip.contains(&flat.node))
            });
        }
        drop(flat_nodes);

        *self.filtered_nodes.borrow_mut() = filtered;
        self.recalculate_visual_structure();

        if let Some(last) = self.last_selected_node.get() {
            self.selected_index
                .set(self.nearest_visible_from(Some(last)));
        } else {
            let length = self.filtered_nodes.borrow().len();
            self.selected_index
                .set(self.selected_index.get().min(length.saturating_sub(1)));
        }

        if !self.filtered_nodes.borrow().is_empty()
            && let Some(&flat_index) = self.filtered_nodes.borrow().get(self.selected_index.get())
            && let Some(flat) = self.flat_nodes.borrow().get(flat_index)
        {
            self.last_selected_node.set(Some(flat.node));
        }
    }

    fn nearest_visible_from(&self, node: Option<usize>) -> usize {
        let filtered = self.filtered_nodes.borrow();
        if filtered.is_empty() {
            return 0;
        }
        let flat_nodes = self.flat_nodes.borrow();
        let positions: HashMap<usize, usize> = filtered
            .iter()
            .enumerate()
            .filter_map(|(position, &flat_index)| {
                flat_nodes.get(flat_index).map(|flat| (flat.node, position))
            })
            .collect();
        let mut current = node;
        while let Some(index) = current {
            if let Some(position) = positions.get(&index) {
                return *position;
            }
            current = self.tree.node(index).and_then(|node| node.parent);
        }
        filtered.len().saturating_sub(1)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the pass mirrors upstream recalculateVisualStructure; the visible-ancestor map and the flatten-compatible indentation rules belong in one place"
    )]
    fn recalculate_visual_structure(&self) {
        let filtered = self.filtered_nodes.borrow().clone();
        if filtered.is_empty() {
            self.visible_parent_map.borrow_mut().clear();
            self.visible_children_map.borrow_mut().clear();
            return;
        }

        let (visible_parent, visible_children, filtered_nodes) = {
            let flat_nodes = self.flat_nodes.borrow();
            let visible: HashSet<usize> = filtered
                .iter()
                .filter_map(|&flat_index| flat_nodes.get(flat_index).map(|flat| flat.node))
                .collect();

            let find_visible_ancestor = |node_index: usize| -> Option<usize> {
                let mut current = self.tree.node(node_index).and_then(|node| node.parent);
                while let Some(index) = current {
                    if visible.contains(&index) {
                        return Some(index);
                    }
                    current = self.tree.node(index).and_then(|node| node.parent);
                }
                None
            };

            let mut visible_parent: HashMap<usize, Option<usize>> = HashMap::new();
            let mut visible_children: HashMap<Option<usize>, Vec<usize>> = HashMap::new();
            visible_children.insert(None, Vec::new());
            let mut filtered_nodes: HashMap<usize, usize> = HashMap::new();
            for &flat_index in &filtered {
                let Some(flat) = flat_nodes.get(flat_index) else {
                    continue;
                };
                let ancestor = find_visible_ancestor(flat.node);
                visible_parent.insert(flat.node, ancestor);
                visible_children
                    .entry(ancestor)
                    .or_default()
                    .push(flat.node);
                filtered_nodes.insert(flat.node, flat_index);
            }
            (visible_parent, visible_children, filtered_nodes)
        };

        let visible_roots = visible_children.get(&None).cloned().unwrap_or_default();
        self.multiple_roots.set(visible_roots.len() > 1);
        let multiple_roots = self.multiple_roots.get();
        let mut stack: Vec<StackEntry> = Vec::new();
        for (position, node) in visible_roots.iter().enumerate().rev() {
            let is_last = position == visible_roots.len() - 1;
            stack.push((
                *node,
                usize::from(multiple_roots),
                multiple_roots,
                multiple_roots,
                is_last,
                Vec::new(),
                multiple_roots,
            ));
        }

        while let Some((
            node_index,
            indent,
            just_branched,
            show_connector,
            is_last,
            gutters,
            virtual_root,
        )) = stack.pop()
        {
            let Some(&flat_index) = filtered_nodes.get(&node_index) else {
                continue;
            };
            let children = visible_children
                .get(&Some(node_index))
                .cloned()
                .unwrap_or_default();
            {
                let mut flat_nodes = self.flat_nodes.borrow_mut();
                if let Some(flat) = flat_nodes.get_mut(flat_index) {
                    flat.indent = indent;
                    flat.show_connector = show_connector;
                    flat.is_last = is_last;
                    flat.gutters.clone_from(&gutters);
                    flat.is_virtual_root_child = virtual_root;
                }
            }

            let multiple_children = children.len() > 1;
            let child_indent = if multiple_children || (just_branched && indent > 0) {
                indent + 1
            } else {
                indent
            };
            let connector_displayed = show_connector && !virtual_root;
            let current_display_indent = if multiple_roots {
                indent.saturating_sub(1)
            } else {
                indent
            };
            let connector_position = current_display_indent.saturating_sub(1);
            let mut child_gutters = gutters;
            if connector_displayed {
                child_gutters.push(GutterInfo {
                    position: connector_position,
                    show: !is_last,
                });
            }
            for (position, child) in children.iter().enumerate().rev() {
                let child_is_last = position == children.len() - 1;
                stack.push((
                    *child,
                    child_indent,
                    multiple_children,
                    multiple_children,
                    child_is_last,
                    child_gutters.clone(),
                    false,
                ));
            }
        }

        *self.visible_parent_map.borrow_mut() = visible_parent;
        *self.visible_children_map.borrow_mut() = visible_children;
    }

    fn is_foldable(&self, node_index: usize) -> bool {
        let children = self
            .visible_children_map
            .borrow()
            .get(&Some(node_index))
            .cloned()
            .unwrap_or_default();
        if children.is_empty() {
            return false;
        }
        let parent = self
            .visible_parent_map
            .borrow()
            .get(&node_index)
            .copied()
            .flatten();
        let Some(parent) = parent else {
            return true;
        };
        self.visible_children_map
            .borrow()
            .get(&Some(parent))
            .is_some_and(|siblings| siblings.len() > 1)
    }

    fn find_branch_segment_start(&self, direction: BranchDirection) -> usize {
        let Some(&flat_index) = self.filtered_nodes.borrow().get(self.selected_index.get()) else {
            return self.selected_index.get();
        };
        let Some(selected_node) = self
            .flat_nodes
            .borrow()
            .get(flat_index)
            .map(|flat| flat.node)
        else {
            return self.selected_index.get();
        };
        let flat_nodes = self.flat_nodes.borrow();
        let index_by_node: HashMap<usize, usize> = self
            .filtered_nodes
            .borrow()
            .iter()
            .enumerate()
            .filter_map(|(position, &flat_index)| {
                flat_nodes.get(flat_index).map(|flat| (flat.node, position))
            })
            .collect();
        drop(flat_nodes);

        match direction {
            BranchDirection::Down => {
                let mut current = selected_node;
                loop {
                    let children = self
                        .visible_children_map
                        .borrow()
                        .get(&Some(current))
                        .cloned()
                        .unwrap_or_default();
                    if children.is_empty() {
                        return index_by_node
                            .get(&current)
                            .copied()
                            .unwrap_or_else(|| self.selected_index.get());
                    }
                    if children.len() > 1 {
                        return index_by_node
                            .get(&children[0])
                            .copied()
                            .unwrap_or_else(|| self.selected_index.get());
                    }
                    current = children[0];
                }
            }
            BranchDirection::Up => {
                let mut current = selected_node;
                loop {
                    let parent = self
                        .visible_parent_map
                        .borrow()
                        .get(&current)
                        .copied()
                        .flatten();
                    let Some(parent) = parent else {
                        return index_by_node
                            .get(&current)
                            .copied()
                            .unwrap_or_else(|| self.selected_index.get());
                    };
                    let children = self
                        .visible_children_map
                        .borrow()
                        .get(&Some(parent))
                        .cloned()
                        .unwrap_or_default();
                    if children.len() > 1
                        && let Some(segment_start) = index_by_node.get(&current).copied()
                        && segment_start < self.selected_index.get()
                    {
                        return segment_start;
                    }
                    current = parent;
                }
            }
        }
    }

    fn get_status_labels(&self) -> String {
        let mut labels = match self.filter_mode.get() {
            FilterMode::NoTools => " [no-tools]".to_string(),
            FilterMode::UserOnly => " [user]".to_string(),
            FilterMode::LabeledOnly => " [labeled]".to_string(),
            FilterMode::All => " [all]".to_string(),
            FilterMode::Default => String::new(),
        };
        if self.show_label_timestamps.get() {
            labels.push_str(" [+label time]");
        }
        labels
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the row painter mirrors upstream render; prefix positions, fold markers, labels, and horizontal panning are one visual unit"
    )]
    fn render_rows(&self, width: usize) -> Vec<String> {
        let filtered = self.filtered_nodes.borrow().clone();
        let mut lines = Vec::new();
        if filtered.is_empty() {
            lines.push(truncate_to_width(
                &theme().fg("muted", "  No entries found"),
                width,
                "",
                false,
            ));
            lines.push(truncate_to_width(
                &theme().fg("muted", &format!("  (0/0){}", self.get_status_labels())),
                width,
                "",
                false,
            ));
            return lines;
        }

        let max_visible = self.max_visible_lines.max(1);
        let start_index = self
            .selected_index
            .get()
            .saturating_sub(max_visible / 2)
            .min(filtered.len().saturating_sub(max_visible));
        let end_index = (start_index + max_visible).min(filtered.len());
        let flat_nodes = self.flat_nodes.borrow();
        let multiple_roots = self.multiple_roots.get();
        let folded = self.folded_nodes.borrow();
        let active = self.active_path.borrow();
        let mut rows = Vec::new();

        for position in start_index..end_index {
            let Some(&flat_index) = filtered.get(position) else {
                continue;
            };
            let Some(flat) = flat_nodes.get(flat_index) else {
                continue;
            };
            let Some(node) = self.tree.node(flat.node) else {
                continue;
            };
            let is_selected = position == self.selected_index.get();
            let cursor = if is_selected {
                theme().fg("accent", "› ")
            } else {
                "  ".to_string()
            };
            let display_indent = if multiple_roots {
                flat.indent.saturating_sub(1)
            } else {
                flat.indent
            };
            let connector = if flat.show_connector && !flat.is_virtual_root_child {
                if flat.is_last { "└─ " } else { "├─ " }
            } else {
                ""
            };
            let connector_position =
                (!connector.is_empty()).then(|| display_indent.saturating_sub(1));
            let mut prefix = String::new();
            let is_folded = folded.contains(&flat.node);
            for char_index in 0..display_indent * 3 {
                let level = char_index / 3;
                let position_in_level = char_index % 3;
                if let Some(gutter) = flat.gutters.iter().find(|gutter| gutter.position == level) {
                    prefix.push(if position_in_level == 0 && gutter.show {
                        '│'
                    } else {
                        ' '
                    });
                } else if connector_position == Some(level) {
                    if position_in_level == 0 {
                        prefix.push(if flat.is_last { '└' } else { '├' });
                    } else if position_in_level == 1 {
                        prefix.push(if is_folded {
                            '⊞'
                        } else if self.is_foldable(flat.node) {
                            '⊟'
                        } else {
                            '─'
                        });
                    } else {
                        prefix.push(' ');
                    }
                } else {
                    prefix.push(' ');
                }
            }

            let shows_fold_in_connector = flat.show_connector && !flat.is_virtual_root_child;
            let fold_marker = if is_folded && !shows_fold_in_connector {
                theme().fg("accent", "⊞ ")
            } else {
                String::new()
            };
            let path_marker = if active.contains(&flat.node) {
                theme().fg("accent", "• ")
            } else {
                String::new()
            };
            let label_text = node.label.borrow().clone();
            let label = label_text.as_ref().map_or_else(String::new, |label| {
                theme().fg("warning", &format!("[{label}] "))
            });
            let label_timestamp = if self.show_label_timestamps.get() && label_text.is_some() {
                node.label_timestamp
                    .borrow()
                    .as_ref()
                    .map_or_else(String::new, |timestamp| {
                        theme().fg("muted", &format!("{} ", format_label_timestamp(timestamp)))
                    })
            } else {
                String::new()
            };
            let content = self.get_entry_display_text(node, is_selected);
            let prefix_part = theme().fg("dim", &prefix) + &fold_marker + &path_marker;
            let anchor_col = visible_width(&prefix_part);
            let mut gutter = cursor;
            let mut body = prefix_part + &label + &label_timestamp + &content;
            if is_selected {
                gutter = theme().bg("selectedBg", &gutter);
                body = theme().bg("selectedBg", &body);
            }
            let body_width = visible_width(&body);
            rows.push(HorizontalViewportRow {
                gutter,
                body,
                anchor_col,
                body_width,
                is_selected,
            });
        }
        drop(active);
        drop(folded);
        drop(flat_nodes);

        lines.extend(render_horizontal_viewport(&rows, width));
        lines.push(truncate_to_width(
            &theme().fg(
                "muted",
                &format!(
                    "  ({}/{}){}",
                    self.selected_index.get() + 1,
                    filtered.len(),
                    self.get_status_labels()
                ),
            ),
            width,
            "",
            false,
        ));
        lines
    }

    fn get_entry_display_text(&self, node: &OwnedNode, is_selected: bool) -> String {
        let result = match &node.entry {
            FileEntry::Entry(SessionEntry::Message(MessageEntry {
                message: Some(message),
                ..
            })) => match message {
                AgentMessage::Standard(Message::User(user)) => {
                    let content = normalize(&content_text(&user.content));
                    theme().fg("accent", "user: ") + &content
                }
                AgentMessage::Standard(Message::Assistant(assistant)) => {
                    assistant_display_text(assistant)
                }
                AgentMessage::Standard(Message::ToolResult(result)) => self
                    .tool_call_map
                    .borrow()
                    .get(&result.tool_call_id)
                    .map_or_else(
                        || theme().fg("muted", &format!("[{}]", result.tool_name)),
                        |tool_call| {
                            theme().fg(
                                "muted",
                                &format_tool_call(&tool_call.name, &tool_call.arguments),
                            )
                        },
                    ),
                AgentMessage::Custom(custom) if custom.role == "bashExecution" => {
                    let command = custom
                        .field("command")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default();
                    theme().fg("dim", &format!("[bash]: {}", normalize(command)))
                }
                AgentMessage::Custom(custom) => theme().fg("dim", &format!("[{}]", custom.role)),
            },
            FileEntry::Entry(SessionEntry::CustomMessage(custom)) => {
                let content = custom
                    .content
                    .as_ref()
                    .map_or_else(String::new, content_text);
                theme().fg("customMessageLabel", &format!("[{}]: ", custom.custom_type))
                    + &normalize(&content)
            }
            FileEntry::Entry(SessionEntry::Compaction(compaction)) => {
                let tokens = rounded_k_tokens(compaction.tokens_before);
                theme().fg("borderAccent", &format!("[compaction: {tokens}k tokens]"))
            }
            FileEntry::Entry(SessionEntry::BranchSummary(summary)) => {
                theme().fg("warning", "[branch summary]: ") + &normalize(&summary.summary)
            }
            FileEntry::Entry(SessionEntry::ModelChange(change)) => {
                theme().fg("dim", &format!("[model: {}]", change.model_id))
            }
            FileEntry::Entry(SessionEntry::ThinkingLevelChange(change)) => {
                theme().fg("dim", &format!("[thinking: {}]", change.thinking_level))
            }
            FileEntry::Entry(SessionEntry::Custom(custom)) => {
                theme().fg("dim", &format!("[custom: {}]", custom.custom_type))
            }
            FileEntry::Entry(SessionEntry::Label(label)) => {
                let text = label.label.as_deref().unwrap_or("(cleared)");
                theme().fg("dim", &format!("[label: {text}]"))
            }
            FileEntry::Entry(SessionEntry::SessionInfo(info)) => info.name.as_ref().map_or_else(
                || {
                    theme().fg("dim", "[title: ")
                        + &italic(&theme().fg("dim", "empty"))
                        + &theme().fg("dim", "]")
                },
                |name| {
                    theme().fg("dim", "[title: ")
                        + &theme().fg("dim", name)
                        + &theme().fg("dim", "]")
                },
            ),
            FileEntry::Entry(SessionEntry::Message(_))
            | FileEntry::Session(_)
            | FileEntry::Other(_) => String::new(),
        };
        if is_selected { bold(&result) } else { result }
    }

    fn get_entry_copy_text(node: &SessionTreeNode) -> Option<String> {
        let text = match &node.entry {
            FileEntry::Entry(SessionEntry::Message(MessageEntry {
                message: Some(message),
                ..
            })) => match message {
                AgentMessage::Standard(Message::User(user)) => content_text(&user.content),
                AgentMessage::Standard(Message::Assistant(assistant)) => {
                    let content = assistant_content_text(&assistant.content);
                    if content.trim().is_empty() {
                        assistant.error_message.clone().unwrap_or_default()
                    } else {
                        content
                    }
                }
                AgentMessage::Standard(Message::ToolResult(result)) => {
                    tool_result_content_text(&result.content)
                }
                AgentMessage::Custom(custom) if custom.role == "bashExecution" => custom
                    .field("command")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                AgentMessage::Custom(_) => String::new(),
            },
            FileEntry::Entry(SessionEntry::CustomMessage(custom)) => custom
                .content
                .as_ref()
                .map_or_else(String::new, content_text),
            FileEntry::Entry(SessionEntry::Compaction(compaction)) => compaction.summary.clone(),
            FileEntry::Entry(SessionEntry::BranchSummary(summary)) => summary.summary.clone(),
            _ => String::new(),
        };
        if text.trim().is_empty() {
            None
        } else {
            Some(text)
        }
    }

    fn to_session_tree_node(&self, root: usize) -> Option<SessionTreeNode> {
        let mut order = Vec::new();
        let mut stack = vec![root];
        while let Some(index) = stack.pop() {
            order.push(index);
            if let Some(node) = self.tree.node(index) {
                stack.extend(node.children.iter().rev().copied());
            }
        }
        let mut built: HashMap<usize, SessionTreeNode> = HashMap::new();
        for &index in order.iter().rev() {
            let Some(node) = self.tree.node(index) else {
                continue;
            };
            let children: Vec<SessionTreeNode> = node
                .children
                .iter()
                .filter_map(|child| built.remove(child))
                .collect();
            built.insert(
                index,
                SessionTreeNode {
                    entry: node.entry.clone(),
                    children,
                    label: node.label.borrow().clone(),
                    label_timestamp: node.label_timestamp.borrow().clone(),
                },
            );
        }
        built.remove(&root)
    }
}

/// The direction of a branch-segment jump.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BranchDirection {
    Up,
    Down,
}

fn render_horizontal_viewport(rows: &[HorizontalViewportRow], width: usize) -> Vec<String> {
    let viewport_width = width.saturating_sub(TREE_GUTTER_WIDTH);
    let max_body_width = rows.iter().map(|row| row.body_width).max().unwrap_or(0);
    let max_horizontal_scroll = max_body_width.saturating_sub(viewport_width);
    let selected_row = rows.iter().find(|row| row.is_selected);
    let mut horizontal_scroll = 0;
    if let Some(selected_row) = selected_row
        && max_horizontal_scroll > 0
    {
        let min_visible = MIN_VISIBLE_ANCHOR_CONTENT_WIDTH
            .max(viewport_width / 3)
            .min(MAX_VISIBLE_ANCHOR_CONTENT_WIDTH);
        if selected_row.anchor_col > viewport_width.saturating_sub(min_visible) {
            let anchor_context = MIN_ANCHOR_CONTEXT_WIDTH
                .max(viewport_width / 4)
                .min(MAX_ANCHOR_CONTEXT_WIDTH);
            horizontal_scroll =
                max_horizontal_scroll.min(selected_row.anchor_col.saturating_sub(anchor_context));
        }
    }

    rows.iter()
        .map(|row| {
            let line = if horizontal_scroll > 0 {
                format!(
                    "{}{}\x1b[0m",
                    row.gutter,
                    slice_by_column(&row.body, horizontal_scroll, viewport_width, true)
                )
            } else {
                format!("{}{}", row.gutter, row.body)
            };
            truncate_to_width(&line, width, "", false)
        })
        .collect()
}

fn normalize(text: &str) -> String {
    text.replace(['\n', '\t'], " ").trim().to_string()
}

fn content_text(content: &UserContent) -> String {
    match content {
        UserContent::Text(text) => text.clone(),
        UserContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                UserBlock::Text(text) => Some(text.text.as_str()),
                UserBlock::Image(_) => None,
            })
            .collect(),
    }
}

fn assistant_content_text(content: &[AssistantBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Text(text) => Some(text.text.as_str()),
            AssistantBlock::Thinking(_) | AssistantBlock::ToolCall(_) => None,
        })
        .collect()
}

fn tool_result_content_text(content: &[ToolResultBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ToolResultBlock::Text(text) => Some(text.text.as_str()),
            ToolResultBlock::Image(_) => None,
        })
        .collect()
}

fn assistant_display_text(assistant: &AssistantMessage) -> String {
    let text = normalize(&assistant_content_text(&assistant.content));
    if !text.is_empty() {
        return theme().fg("success", "assistant: ") + &text;
    }
    if assistant.stop_reason == StopReason::Aborted {
        return theme().fg("success", "assistant: ") + &theme().fg("muted", "(aborted)");
    }
    if let Some(error) = &assistant.error_message {
        let error: String = normalize(error).chars().take(80).collect();
        return theme().fg("success", "assistant: ") + &theme().fg("error", &error);
    }
    theme().fg("success", "assistant: ") + &theme().fg("muted", "(no content)")
}

const fn rounded_k_tokens(tokens_before: i64) -> i64 {
    if tokens_before >= 0 {
        tokens_before.saturating_add(500) / 1000
    } else {
        tokens_before.saturating_sub(500) / 1000
    }
}

fn extract_tool_calls(entry: &FileEntry, calls: &mut HashMap<String, ToolCallInfo>) {
    let FileEntry::Entry(SessionEntry::Message(MessageEntry {
        message: Some(AgentMessage::Standard(Message::Assistant(assistant))),
        ..
    })) = entry
    else {
        return;
    };
    for block in &assistant.content {
        if let AssistantBlock::ToolCall(call) = block {
            calls.insert(
                call.id.clone(),
                ToolCallInfo {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                },
            );
        }
    }
}

fn message_role(message: &AgentMessage) -> &str {
    match message {
        AgentMessage::Standard(Message::User(_)) => "user",
        AgentMessage::Standard(Message::Assistant(_)) => "assistant",
        AgentMessage::Standard(Message::ToolResult(_)) => "toolResult",
        AgentMessage::Custom(custom) => &custom.role,
    }
}

fn get_searchable_text(node: &OwnedNode) -> String {
    let mut parts = Vec::new();
    if let Some(label) = node.label.borrow().as_ref() {
        parts.push(label.clone());
    }
    match &node.entry {
        FileEntry::Entry(SessionEntry::Message(MessageEntry {
            message: Some(message),
            ..
        })) => {
            parts.push(message_role(message).to_string());
            match message {
                AgentMessage::Standard(Message::User(user)) => {
                    parts.push(content_text(&user.content));
                }
                AgentMessage::Standard(Message::Assistant(assistant)) => {
                    parts.push(assistant_content_text(&assistant.content));
                }
                AgentMessage::Standard(Message::ToolResult(result)) => {
                    parts.push(tool_result_content_text(&result.content));
                }
                AgentMessage::Custom(custom) => {
                    if let Some(command) =
                        custom.field("command").and_then(serde_json::Value::as_str)
                    {
                        parts.push(command.to_string());
                    }
                }
            }
        }
        FileEntry::Entry(SessionEntry::CustomMessage(custom)) => {
            parts.push(custom.custom_type.clone());
            if let Some(content) = &custom.content {
                parts.push(content_text(content));
            }
        }
        FileEntry::Entry(SessionEntry::Compaction(_)) => parts.push("compaction".to_string()),
        FileEntry::Entry(SessionEntry::BranchSummary(summary)) => {
            parts.push("branch summary".to_string());
            parts.push(summary.summary.clone());
        }
        FileEntry::Entry(SessionEntry::SessionInfo(info)) => {
            parts.push("title".to_string());
            if let Some(name) = &info.name {
                parts.push(name.clone());
            }
        }
        FileEntry::Entry(SessionEntry::ModelChange(change)) => {
            parts.push("model".to_string());
            parts.push(change.model_id.clone());
        }
        FileEntry::Entry(SessionEntry::ThinkingLevelChange(change)) => {
            parts.push("thinking".to_string());
            parts.push(change.thinking_level.clone());
        }
        FileEntry::Entry(SessionEntry::Custom(custom)) => {
            parts.push("custom".to_string());
            parts.push(custom.custom_type.clone());
        }
        FileEntry::Entry(SessionEntry::Label(label)) => {
            parts.push("label".to_string());
            parts.push(label.label.clone().unwrap_or_default());
        }
        FileEntry::Entry(SessionEntry::Message(_))
        | FileEntry::Session(_)
        | FileEntry::Other(_) => {}
    }
    parts.join(" ")
}

const fn is_settings_entry(entry: &FileEntry) -> bool {
    matches!(
        entry,
        FileEntry::Entry(
            SessionEntry::Label(_)
                | SessionEntry::Custom(_)
                | SessionEntry::ModelChange(_)
                | SessionEntry::ThinkingLevelChange(_)
                | SessionEntry::SessionInfo(_)
        )
    )
}

const fn is_user_message(entry: &FileEntry) -> bool {
    matches!(
        entry,
        FileEntry::Entry(SessionEntry::Message(MessageEntry {
            message: Some(AgentMessage::Standard(Message::User(_))),
            ..
        }))
    )
}

const fn is_tool_result(entry: &FileEntry) -> bool {
    matches!(
        entry,
        FileEntry::Entry(SessionEntry::Message(MessageEntry {
            message: Some(AgentMessage::Standard(Message::ToolResult(_))),
            ..
        }))
    )
}

fn hides_tool_call_only_assistant(entry: &FileEntry) -> bool {
    let FileEntry::Entry(SessionEntry::Message(MessageEntry {
        message: Some(AgentMessage::Standard(Message::Assistant(assistant))),
        ..
    })) = entry
    else {
        return false;
    };
    let has_text = assistant.content.iter().any(|block| match block {
        AssistantBlock::Text(text) => !text.text.trim().is_empty(),
        AssistantBlock::Thinking(_) | AssistantBlock::ToolCall(_) => false,
    });
    let error_or_aborted = !matches!(
        assistant.stop_reason,
        StopReason::Stop | StopReason::ToolUse
    );
    !has_text && !error_or_aborted
}

fn format_tool_call(name: &str, arguments: &serde_json::Map<String, serde_json::Value>) -> String {
    let arg_string = |keys: &[&str]| -> String {
        keys.iter()
            .find_map(|key| arguments.get(*key).and_then(serde_json::Value::as_str))
            .unwrap_or_default()
            .to_string()
    };
    let argument_or = |key: &str, fallback: &str| -> String {
        arguments
            .get(key)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or(fallback)
            .to_string()
    };
    match name {
        "read" => {
            let path = shorten_path(&arg_string(&["path", "file_path"]));
            let offset = arguments.get("offset").and_then(serde_json::Value::as_i64);
            let limit = arguments.get("limit").and_then(serde_json::Value::as_i64);
            let mut display = path;
            if offset.is_some() || limit.is_some() {
                let start = offset.unwrap_or(1);
                let end = limit.map(|limit| start.saturating_add(limit).saturating_sub(1));
                match end {
                    Some(end) => {
                        let _ = write!(display, ":{start}-{end}");
                    }
                    None => {
                        let _ = write!(display, ":{start}");
                    }
                }
            }
            format!("[read: {display}]")
        }
        "write" => format!(
            "[write: {}]",
            shorten_path(&arg_string(&["path", "file_path"]))
        ),
        "edit" => format!(
            "[edit: {}]",
            shorten_path(&arg_string(&["path", "file_path"]))
        ),
        "bash" => {
            let raw = arg_string(&["command"]);
            let command: String = normalize(&raw).chars().take(50).collect();
            let ellipsis = if raw.chars().count() > 50 { "..." } else { "" };
            format!("[bash: {command}{ellipsis}]")
        }
        "grep" => {
            let pattern = arg_string(&["pattern"]);
            let path = shorten_path(&argument_or("path", "."));
            format!("[grep: /{pattern}/ in {path}]")
        }
        "find" => {
            let pattern = arg_string(&["pattern"]);
            let path = shorten_path(&argument_or("path", "."));
            format!("[find: {pattern} in {path}]")
        }
        "ls" => format!("[ls: {}]", shorten_path(&argument_or("path", "."))),
        _ => {
            let full = serde_json::Value::Object(arguments.clone()).to_string();
            let display: String = full.chars().take(40).collect();
            let ellipsis = if full.chars().count() > 40 { "..." } else { "" };
            format!("[{name}: {display}{ellipsis}]")
        }
    }
}

/// The display form of a label timestamp, upstream's
/// `formatLabelTimestamp`, rendered from the timestamp's UTC fields.
///
/// An unparseable timestamp renders verbatim.
#[must_use]
pub fn format_label_timestamp(timestamp: &str) -> String {
    format_label_timestamp_at(timestamp, now_millis())
}

/// [`format_label_timestamp`] against a fixed `now`, for deterministic tests.
#[must_use]
pub fn format_label_timestamp_at(timestamp: &str, now_ms: i64) -> String {
    let Some(millis) = parse_iso8601_millis(timestamp) else {
        return timestamp.to_string();
    };
    let (year, month, day, hour, minute) = utc_parts(millis);
    let (now_year, now_month, now_day, _, _) = utc_parts(now_ms);
    let time = format!("{hour:02}:{minute:02}");
    if year == now_year && month == now_month && day == now_day {
        return time;
    }
    if year == now_year {
        return format!("{month}/{day} {time}");
    }
    format!("{}/{}/{} {time}", year.rem_euclid(100), month, day)
}

fn utc_parts(millis: i64) -> (i64, u32, u32, u32, u32) {
    const DAY_MILLIS: i64 = 86_400_000;
    let days = millis.div_euclid(DAY_MILLIS);
    let day_millis = millis.rem_euclid(DAY_MILLIS);
    let (year, month, day) = civil_from_days(days);
    let hour = u32::try_from(day_millis / 3_600_000).unwrap_or(0);
    let minute = u32::try_from((day_millis % 3_600_000) / 60_000).unwrap_or(0);
    (year, month, day, hour, minute)
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    (
        year + i64::from(month <= 2),
        u32::try_from(month).unwrap_or(1),
        u32::try_from(day).unwrap_or(1),
    )
}

/// The tree list's search line, upstream `SearchLine`.
struct SearchLine {
    tree_list: Rc<TreeList>,
}

impl Component for SearchLine {
    fn render(&self, width: usize) -> Vec<String> {
        let query = self.tree_list.get_search_query();
        let line = if query.is_empty() {
            format!("  {}", theme().fg("muted", "Type to search:"))
        } else {
            format!(
                "  {} {}",
                theme().fg("muted", "Type to search:"),
                theme().fg("accent", &query)
            )
        };
        vec![truncate_to_width(&line, width, "", false)]
    }
}

/// The semantic tree help rows, upstream `TreeHelp`.
struct TreeHelp;

/// One help row's actions and label.
struct HelpItem {
    keys: &'static [&'static str],
    label: &'static str,
    label_first: bool,
}

const TREE_HELP_ITEMS: &[HelpItem] = &[
    HelpItem {
        keys: &["tui.select.up", "tui.select.down"],
        label: "move",
        label_first: false,
    },
    HelpItem {
        keys: &["tui.editor.cursorLeft", "tui.editor.cursorRight"],
        label: "page",
        label_first: false,
    },
    HelpItem {
        keys: &["app.tree.foldOrUp", "app.tree.unfoldOrDown"],
        label: "branch",
        label_first: false,
    },
    HelpItem {
        keys: &["app.message.copy"],
        label: "copy",
        label_first: false,
    },
    HelpItem {
        keys: &["app.tree.editLabel"],
        label: "label",
        label_first: false,
    },
    HelpItem {
        keys: &["app.tree.toggleLabelTimestamp"],
        label: "label time",
        label_first: false,
    },
    HelpItem {
        keys: &[
            "app.tree.filter.default",
            "app.tree.filter.noTools",
            "app.tree.filter.userOnly",
            "app.tree.filter.labeledOnly",
            "app.tree.filter.all",
        ],
        label: "filters",
        label_first: true,
    },
    HelpItem {
        keys: &[
            "app.tree.filter.cycleForward",
            "app.tree.filter.cycleBackward",
        ],
        label: "cycle",
        label_first: true,
    },
];

impl Component for TreeHelp {
    fn render(&self, width: usize) -> Vec<String> {
        let items: Vec<String> = TREE_HELP_ITEMS
            .iter()
            .map(|item| {
                let text = format_help_keys(item.keys);
                if text.is_empty() {
                    item.label.to_string()
                } else if item.label_first {
                    format!("{} {text}", item.label)
                } else {
                    format!("{text} {}", item.label)
                }
            })
            .collect();

        let available_width = width.max(1);
        let indent = "  ";
        let separator = " · ";
        let mut lines = Vec::new();
        let mut current = String::new();
        for item in items {
            let candidate = if current.is_empty() {
                let indented = format!("{indent}{item}");
                if visible_width(&indented) <= available_width {
                    indented
                } else {
                    item.clone()
                }
            } else {
                format!("{current}{separator}{item}")
            };
            if current.is_empty() || visible_width(&candidate) <= available_width {
                current = candidate;
                continue;
            }
            lines.extend(wrap_text_with_ansi(current.trim_end(), available_width));
            let indented = format!("{indent}{item}");
            current = if visible_width(&indented) <= available_width {
                indented
            } else {
                item
            };
        }
        if !current.is_empty() {
            lines.extend(wrap_text_with_ansi(current.trim_end(), available_width));
        }
        lines
            .into_iter()
            .map(|line| theme().fg("muted", &line))
            .collect()
    }
}

fn format_help_keys(actions: &[&str]) -> String {
    let keys = {
        let keybindings = get_keybindings();
        let mut keys = Vec::new();
        for action in actions {
            if let Some(key) = keybindings.get_keys(action).into_iter().next() {
                keys.push(key);
            }
        }
        keys
    };
    if keys.is_empty() {
        return String::new();
    }
    replace_help_words(&format_key_text(
        &compact_raw_keys(&keys),
        KeyTextFormatOptions::default(),
    ))
}

fn compact_raw_keys(keys: &[String]) -> String {
    if keys.len() == 1 {
        return keys[0].clone();
    }
    let parts: Vec<(&str, &str)> = keys
        .iter()
        .map(|key| {
            key.rfind('+').map_or(("", key.as_str()), |index| {
                (&key[..=index], &key[index + 1..])
            })
        })
        .collect();
    let prefix = parts[0].0;
    if !prefix.is_empty() && parts.iter().all(|part| part.0 == prefix) {
        let suffixes: Vec<&str> = parts.iter().map(|part| part.1).collect();
        return format!("{prefix}{}", suffixes.join("/"));
    }
    keys.join("/")
}

fn replace_help_words(text: &str) -> String {
    let mut output = String::new();
    let mut word = String::new();
    for character in text.chars() {
        if character.is_ascii_alphanumeric() {
            word.push(character);
        } else {
            output.push_str(help_word(&word));
            word.clear();
            output.push(character);
        }
    }
    output.push_str(help_word(&word));
    output
}

fn help_word(word: &str) -> &str {
    match word {
        "pageUp" => "pgup",
        "pageDown" => "pgdn",
        "up" => "↑",
        "down" => "↓",
        "left" => "←",
        "right" => "→",
        _ => word,
    }
}

/// The label editor, upstream `LabelInput`.
struct LabelInput {
    input: Input,
    entry_id: String,
    focused: Cell<bool>,
    on_submit: RefCell<Option<LabelCallback>>,
    on_cancel: RefCell<Option<CancelCallback>>,
    parser: KeyParser,
}

impl LabelInput {
    fn new(entry_id: &str, current_label: Option<&str>) -> Self {
        let input = Input::new();
        if let Some(label) = current_label {
            input.set_value(label.to_string());
        }
        Self {
            input,
            entry_id: entry_id.to_string(),
            focused: Cell::new(false),
            on_submit: RefCell::new(None),
            on_cancel: RefCell::new(None),
            parser: KeyParser::new(),
        }
    }
}

impl Component for LabelInput {
    fn render(&self, width: usize) -> Vec<String> {
        let indent = "  ";
        let available_width = width.saturating_sub(indent.len());
        let mut lines = vec![truncate_to_width(
            &format!(
                "{indent}{}",
                theme().fg("muted", "Label (empty to remove):")
            ),
            width,
            "",
            false,
        )];
        lines.extend(
            self.input
                .render(available_width)
                .into_iter()
                .map(|line| truncate_to_width(&format!("{indent}{line}"), width, "", false)),
        );
        lines.push(truncate_to_width(
            &format!(
                "{indent}{}  {}",
                key_hint("tui.select.confirm", "save"),
                key_hint("tui.select.cancel", "cancel")
            ),
            width,
            "",
            false,
        ));
        lines
    }

    fn handle_input(&self, data: &str) {
        let matches = |action: &str| get_keybindings().matches(&self.parser, data, action);
        if matches("tui.select.confirm") {
            let value = self.input.get_value().trim().to_string();
            let label = if value.is_empty() { None } else { Some(value) };
            let callback = self.on_submit.borrow().clone();
            if let Some(callback) = callback {
                callback(&self.entry_id, label.as_deref());
            }
        } else if matches("tui.select.cancel") {
            let callback = self.on_cancel.borrow().clone();
            if let Some(callback) = callback {
                callback();
            }
        } else {
            self.input.handle_input(data);
        }
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }
}

impl Focusable for LabelInput {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
        self.input.set_focused(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

impl std::fmt::Debug for LabelInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LabelInput").finish_non_exhaustive()
    }
}

/// The options for [`TreeSelectorComponent::new`].
pub struct TreeSelectorOptions {
    /// Called with the confirmed entry id.
    pub on_select: SelectCallback,
    /// Called on dismissal.
    pub on_cancel: CancelCallback,
    /// Called after a label change.
    pub on_label_change: Option<LabelCallback>,
    /// The row to select first; defaults to the current leaf.
    pub initial_selected_id: Option<String>,
    /// The opening filter mode.
    pub initial_filter_mode: Option<FilterMode>,
}

impl std::fmt::Debug for TreeSelectorOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeSelectorOptions")
            .field("initial_selected_id", &self.initial_selected_id)
            .field("initial_filter_mode", &self.initial_filter_mode)
            .finish_non_exhaustive()
    }
}

/// The session tree selector, upstream `TreeSelectorComponent`.
pub struct TreeSelectorComponent {
    tree_list: Rc<TreeList>,
    label_input: RefCell<Option<Rc<LabelInput>>>,
    label_input_container: Rc<Container>,
    tree_container: Rc<Container>,
    on_label_change: Option<LabelCallback>,
    focused: Cell<bool>,
    container: Container,
}

impl TreeSelectorComponent {
    /// Build the selector over a resolved session tree.
    #[must_use]
    pub fn new(
        tree: &[SessionTreeNode],
        current_leaf_id: Option<&str>,
        terminal_height: usize,
        options: TreeSelectorOptions,
    ) -> Rc<Self> {
        let max_visible_lines = (terminal_height / 2).max(5);
        let tree_list = Rc::new(TreeList::new(
            tree,
            current_leaf_id,
            max_visible_lines,
            options.initial_selected_id.as_deref(),
            options.initial_filter_mode,
        ));
        *tree_list.on_select.borrow_mut() = Some(options.on_select);
        *tree_list.on_cancel.borrow_mut() = Some(options.on_cancel);

        let tree_container = Rc::new(Container::default());
        let tree_component: Rc<dyn Component> = tree_list.clone();
        tree_container.add_child(tree_component);
        let label_input_container = Rc::new(Container::default());

        let container = Container::default();
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(DynamicBorder::themed()));
        container.add_child(Rc::new(Text::with_padding(bold("  Session Tree"), 1, 0)));
        container.add_child(Rc::new(TreeHelp));
        container.add_child(Rc::new(SearchLine {
            tree_list: Rc::clone(&tree_list),
        }));
        container.add_child(Rc::new(DynamicBorder::themed()));
        container.add_child(Rc::new(Spacer::new()));
        let tree_container_component: Rc<dyn Component> = tree_container.clone();
        container.add_child(tree_container_component);
        let label_container_component: Rc<dyn Component> = label_input_container.clone();
        container.add_child(label_container_component);
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(DynamicBorder::themed()));

        let selector = Rc::new(Self {
            tree_list,
            label_input: RefCell::new(None),
            label_input_container,
            tree_container,
            on_label_change: options.on_label_change,
            focused: Cell::new(false),
            container,
        });
        let weak = Rc::downgrade(&selector);
        *selector.tree_list.on_label_edit.borrow_mut() = Some(Rc::new(move |entry_id, label| {
            if let Some(selector) = weak.upgrade() {
                selector.show_label_input(entry_id, label);
            }
        }));
        selector
    }

    /// The focusable tree list inside the selector, upstream's `getTreeList`.
    #[must_use]
    pub fn get_tree_list(&self) -> Rc<TreeList> {
        Rc::clone(&self.tree_list)
    }

    /// Install the copy callback, upstream's public `onCopy`.
    pub fn set_on_copy(&self, callback: CopyCallback) {
        *self.tree_list.on_copy.borrow_mut() = Some(callback);
    }

    fn show_label_input(self: &Rc<Self>, entry_id: &str, current_label: Option<&str>) {
        let label_input = Rc::new(LabelInput::new(entry_id, current_label));
        let weak = Rc::downgrade(self);
        *label_input.on_submit.borrow_mut() = Some(Rc::new(move |entry_id, label| {
            if let Some(selector) = weak.upgrade() {
                selector.tree_list.update_node_label(entry_id, label, None);
                if let Some(callback) = &selector.on_label_change {
                    callback(entry_id, label);
                }
                selector.hide_label_input();
            }
        }));
        let weak = Rc::downgrade(self);
        *label_input.on_cancel.borrow_mut() = Some(Rc::new(move || {
            if let Some(selector) = weak.upgrade() {
                selector.hide_label_input();
            }
        }));

        label_input.set_focused(self.focused.get());
        self.tree_container.clear();
        self.label_input_container.clear();
        let component: Rc<dyn Component> = label_input.clone();
        self.label_input_container.add_child(component);
        *self.label_input.borrow_mut() = Some(label_input);
    }

    fn hide_label_input(&self) {
        *self.label_input.borrow_mut() = None;
        self.label_input_container.clear();
        self.tree_container.clear();
        let component: Rc<dyn Component> = self.tree_list.clone();
        self.tree_container.add_child(component);
    }
}

impl Component for TreeSelectorComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_input(&self, data: &str) {
        let label_input = self.label_input.borrow().clone();
        if let Some(label_input) = label_input {
            label_input.handle_input(data);
        } else {
            self.tree_list.handle_input(data);
        }
    }

    fn invalidate(&self) {
        self.container.invalidate();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }
}

impl Focusable for TreeSelectorComponent {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
        if let Some(label_input) = self.label_input.borrow().as_ref() {
            label_input.set_focused(focused);
        }
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

impl std::fmt::Debug for TreeSelectorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeSelectorComponent")
            .finish_non_exhaustive()
    }
}

impl Component for TreeList {
    fn render(&self, width: usize) -> Vec<String> {
        self.render_rows(width)
    }

    fn invalidate(&self) {}

    fn wants_input(&self) -> bool {
        true
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the input ladder mirrors upstream handleInput; the action order is the contract and splitting it would hide precedence"
    )]
    fn handle_input(&self, data: &str) {
        let matches = |action: &str| get_keybindings().matches(&self.parser, data, action);
        let filtered_len = self.filtered_nodes.borrow().len();

        if matches("tui.select.up") {
            if filtered_len == 0 {
                return;
            }
            let selected = self.selected_index.get();
            self.selected_index.set(if selected == 0 {
                filtered_len - 1
            } else {
                selected - 1
            });
        } else if matches("tui.select.down") {
            if filtered_len == 0 {
                return;
            }
            let selected = self.selected_index.get();
            self.selected_index.set(if selected == filtered_len - 1 {
                0
            } else {
                selected + 1
            });
        } else if matches("app.tree.foldOrUp") {
            let current = self.selected_node_index();
            if current.is_some_and(|node| {
                self.is_foldable(node) && !self.folded_nodes.borrow().contains(&node)
            }) {
                if let Some(node) = current {
                    self.folded_nodes.borrow_mut().insert(node);
                }
                self.apply_filter();
            } else {
                self.selected_index
                    .set(self.find_branch_segment_start(BranchDirection::Up));
            }
        } else if matches("app.tree.unfoldOrDown") {
            let current = self.selected_node_index();
            if current.is_some_and(|node| self.folded_nodes.borrow().contains(&node)) {
                if let Some(node) = current {
                    self.folded_nodes.borrow_mut().remove(&node);
                }
                self.apply_filter();
            } else {
                self.selected_index
                    .set(self.find_branch_segment_start(BranchDirection::Down));
            }
        } else if matches("tui.editor.cursorLeft") || matches("tui.select.pageUp") {
            self.selected_index.set(
                self.selected_index
                    .get()
                    .saturating_sub(self.max_visible_lines),
            );
        } else if matches("tui.editor.cursorRight") || matches("tui.select.pageDown") {
            self.selected_index.set(
                (self.selected_index.get() + self.max_visible_lines)
                    .min(filtered_len.saturating_sub(1)),
            );
        } else if matches("tui.select.confirm") {
            if let Some(callback) = self.on_select.borrow().clone()
                && let Some(entry_id) = self.get_selected_entry_id()
            {
                callback(&entry_id);
            }
        } else if matches("app.message.copy") {
            self.copy_selected();
        } else if matches("tui.select.cancel") {
            if self.search_query.borrow().is_empty() {
                let callback = self.on_cancel.borrow().clone();
                if let Some(callback) = callback {
                    callback();
                }
            } else {
                self.search_query.borrow_mut().clear();
                self.folded_nodes.borrow_mut().clear();
                self.apply_filter();
            }
        } else if matches("app.tree.filter.default") {
            self.filter_mode.set(FilterMode::Default);
            self.folded_nodes.borrow_mut().clear();
            self.apply_filter();
        } else if matches("app.tree.filter.noTools") {
            self.filter_mode
                .set(if self.filter_mode.get() == FilterMode::NoTools {
                    FilterMode::Default
                } else {
                    FilterMode::NoTools
                });
            self.folded_nodes.borrow_mut().clear();
            self.apply_filter();
        } else if matches("app.tree.filter.userOnly") {
            self.filter_mode
                .set(if self.filter_mode.get() == FilterMode::UserOnly {
                    FilterMode::Default
                } else {
                    FilterMode::UserOnly
                });
            self.folded_nodes.borrow_mut().clear();
            self.apply_filter();
        } else if matches("app.tree.filter.labeledOnly") {
            self.filter_mode
                .set(if self.filter_mode.get() == FilterMode::LabeledOnly {
                    FilterMode::Default
                } else {
                    FilterMode::LabeledOnly
                });
            self.folded_nodes.borrow_mut().clear();
            self.apply_filter();
        } else if matches("app.tree.filter.all") {
            self.filter_mode
                .set(if self.filter_mode.get() == FilterMode::All {
                    FilterMode::Default
                } else {
                    FilterMode::All
                });
            self.folded_nodes.borrow_mut().clear();
            self.apply_filter();
        } else if matches("app.tree.filter.cycleBackward") {
            let modes = [
                FilterMode::Default,
                FilterMode::NoTools,
                FilterMode::UserOnly,
                FilterMode::LabeledOnly,
                FilterMode::All,
            ];
            let current = modes
                .iter()
                .position(|mode| *mode == self.filter_mode.get())
                .unwrap_or(0);
            self.filter_mode
                .set(modes[(current + modes.len() - 1) % modes.len()]);
            self.folded_nodes.borrow_mut().clear();
            self.apply_filter();
        } else if matches("app.tree.filter.cycleForward") {
            let modes = [
                FilterMode::Default,
                FilterMode::NoTools,
                FilterMode::UserOnly,
                FilterMode::LabeledOnly,
                FilterMode::All,
            ];
            let current = modes
                .iter()
                .position(|mode| *mode == self.filter_mode.get())
                .unwrap_or(0);
            self.filter_mode.set(modes[(current + 1) % modes.len()]);
            self.folded_nodes.borrow_mut().clear();
            self.apply_filter();
        } else if matches("tui.editor.deleteCharBackward") {
            if !self.search_query.borrow().is_empty() {
                let _ = self.search_query.borrow_mut().pop();
                self.folded_nodes.borrow_mut().clear();
                self.apply_filter();
            }
        } else if matches("app.tree.editLabel") {
            if let Some(callback) = self.on_label_edit.borrow().clone()
                && let Some(entry_id) = self.get_selected_entry_id()
            {
                let label = self.get_selected_label();
                callback(&entry_id, label.as_deref());
            }
        } else if matches("app.tree.toggleLabelTimestamp") {
            self.show_label_timestamps
                .set(!self.show_label_timestamps.get());
        } else if !data.chars().any(|character| {
            let code = u32::from(character);
            code < 32 || code == 0x7f || (0x80..=0x9f).contains(&code)
        }) && !data.is_empty()
        {
            self.search_query.borrow_mut().push_str(data);
            self.folded_nodes.borrow_mut().clear();
            self.apply_filter();
        }
    }
}

impl TreeList {
    fn selected_node_index(&self) -> Option<usize> {
        let flat_index = *self
            .filtered_nodes
            .borrow()
            .get(self.selected_index.get())?;
        self.flat_nodes
            .borrow()
            .get(flat_index)
            .map(|flat| flat.node)
    }
}

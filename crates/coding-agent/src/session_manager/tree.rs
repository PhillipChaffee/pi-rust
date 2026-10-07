//! Tree traversal and branching, upstream's `getTree`, `getBranch`,
//! `branch`, `resetLeaf`, and `branchWithSummary` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::cmp::Ordering;
use std::collections::HashMap;

use serde_json::Value as JsonValue;

use pi_agent_core::harness::session::jsonl::codec::parse_iso8601_millis;
use pi_ai::types::Usage;

use super::context::{LeafId, build_context_entries, build_session_context};
use super::entries::{BranchSummaryEntry, SessionEntry, SessionEntryBase};
use super::file_entry::FileEntry;
use super::manager::SessionManager;
use super::now_iso8601;
use super::types::SessionHeader;

/// The session tree node, upstream's `SessionTreeNode` — a defensive copy of
/// the session structure with resolved labels.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionTreeNode {
    /// The entry copy.
    pub entry: FileEntry,
    /// Child nodes, oldest first.
    pub children: Vec<Self>,
    /// The resolved label for this entry, if any.
    pub label: Option<String>,
    /// The timestamp of the latest label change for this entry, if any.
    pub label_timestamp: Option<String>,
}

/// One entry's assembly record for the tree build.
struct TreeData {
    entry: FileEntry,
    label: Option<String>,
    label_timestamp: Option<String>,
    child_indices: Vec<usize>,
    parent_index: Option<usize>,
    assembled: Vec<SessionTreeNode>,
}

impl SessionManager {
    /// The current leaf id, upstream's `getLeafId`; absent before any entry
    /// or after [`SessionManager::reset_leaf`].
    #[must_use]
    pub fn get_leaf_id(&self) -> Option<&str> {
        self.leaf_id.as_deref()
    }

    /// The current leaf entry, upstream's `getLeafEntry`.
    #[must_use]
    pub fn get_leaf_entry(&self) -> Option<&FileEntry> {
        let leaf_id = self.leaf_id.as_deref()?;
        self.by_id
            .get(leaf_id)
            .and_then(|index| self.file_entries.get(*index))
    }

    /// The entry with the given id, upstream's `getEntry`.
    #[must_use]
    pub fn get_entry(&self, id: &str) -> Option<&FileEntry> {
        self.by_id
            .get(id)
            .and_then(|index| self.file_entries.get(*index))
    }

    /// All direct children of an entry, upstream's `getChildren` (file order,
    /// the upstream map's insertion order).
    #[must_use]
    pub fn get_children(&self, parent_id: &str) -> Vec<&FileEntry> {
        self.file_entries
            .iter()
            .filter(|entry| {
                !matches!(entry, FileEntry::Session(_))
                    && entry.entry_parent_id() == Some(parent_id)
            })
            .collect()
    }

    /// The label for an entry, upstream's `getLabel`.
    #[must_use]
    pub fn get_label(&self, id: &str) -> Option<&str> {
        self.labels_by_id
            .iter()
            .find(|(target, _, _)| target == id)
            .map(|(_, label, _)| label.as_str())
    }

    /// Walk from an entry to root, returning all entries in path order,
    /// upstream's `getBranch`. Includes all entry types; use
    /// [`SessionManager::build_session_context`] to get the resolved messages
    /// for the LLM.
    #[must_use]
    pub fn get_branch(&self, from_id: Option<&str>) -> Vec<&FileEntry> {
        let start_id = from_id.or(self.leaf_id.as_deref());
        let mut path: Vec<&FileEntry> = Vec::new();
        let mut current = start_id.and_then(|id| self.get_entry(id));
        while let Some(entry) = current {
            path.push(entry);
            current = entry
                .entry_parent_id()
                .and_then(|parent| self.get_entry(parent));
        }
        path.reverse();
        path
    }

    /// The compaction-aware entry list for context/rendering, upstream's
    /// `buildContextEntries` method form: tree traversal from the current
    /// leaf.
    #[must_use]
    pub fn build_context_entries(&self) -> Vec<&FileEntry> {
        build_context_entries(&self.file_entries, self.leaf_arg(), &self.by_id)
    }

    /// The session context (what gets sent to the LLM), upstream's
    /// `buildSessionContext` method form: tree traversal from the current
    /// leaf.
    #[must_use]
    pub fn build_session_context(&self) -> super::context::SessionContext {
        build_session_context(&self.file_entries, self.leaf_arg(), &self.by_id)
    }

    /// The leaf pointer as the module-level projection's argument: the
    /// manager's absent leaf is the empty path, upstream's explicit `null`.
    fn leaf_arg(&self) -> LeafId<'_> {
        self.leaf_id.as_deref().map_or(LeafId::None, LeafId::Id)
    }

    /// The session header, upstream's `getHeader`.
    #[must_use]
    pub fn get_header(&self) -> Option<&SessionHeader> {
        self.file_entries.iter().find_map(|entry| match entry {
            FileEntry::Session(header) => Some(header),
            _ => None,
        })
    }

    /// All session entries (excludes the header), upstream's `getEntries`.
    /// The session is append-only: entries cannot be modified or deleted.
    #[must_use]
    pub fn entries(&self) -> Vec<&FileEntry> {
        self.file_entries
            .iter()
            .filter(|entry| !matches!(entry, FileEntry::Session(_)))
            .collect()
    }

    /// The session as a tree structure, upstream's `getTree` — a defensive
    /// copy of all entries. A well-formed session has exactly one root
    /// (first entry with `parentId === null`); orphaned entries (broken
    /// parent chain) also return as roots. Children sort by timestamp; the
    /// assembly walks iteratively so deep trees cannot overflow the stack.
    #[must_use]
    pub fn get_tree(&self) -> Vec<SessionTreeNode> {
        let entries = self.entries();
        let mut data: Vec<TreeData> = Vec::with_capacity(entries.len());
        let mut node_map: HashMap<String, usize> = HashMap::new();

        // Create nodes with resolved labels. The `&entry` pattern derefs the
        // Vec<&FileEntry> element to the shared reference the node copy takes.
        for &entry in &entries {
            let Some(id) = entry.entry_id() else { continue };
            let id = id.to_owned();
            let label = self.get_label(&id).map(str::to_owned);
            let label_timestamp = self
                .labels_by_id
                .iter()
                .find(|(target, _, _)| target == &id)
                .map(|(_, _, timestamp)| timestamp.clone());
            node_map.insert(id, data.len());
            data.push(TreeData {
                entry: entry.clone(),
                label,
                label_timestamp,
                child_indices: Vec::new(),
                parent_index: None,
                assembled: Vec::new(),
            });
        }

        // Build the parent links. `parentId === null` and self-parents root;
        // parents missing from the map orphan-root.
        let mut roots: Vec<usize> = Vec::new();
        for &entry in &entries {
            let Some(id) = entry.entry_id() else { continue };
            let Some(&index) = node_map.get(id) else {
                continue;
            };
            let parent = match entry.entry_parent_id() {
                None => None,
                Some(parent) if parent == id => None,
                Some(parent) => node_map.get(parent).copied(),
            };
            match parent {
                None => roots.push(index),
                Some(parent_index) => {
                    data[parent_index].child_indices.push(index);
                    data[index].parent_index = Some(parent_index);
                }
            }
        }

        // Assemble iteratively (post-order: children complete before their
        // parent), sorting each node's children at completion the way
        // upstream sorts in place during its iterative walk.
        let mut ordered_roots: Vec<SessionTreeNode> = Vec::new();
        let mut stack: Vec<(usize, bool)> =
            roots.iter().rev().map(|index| (*index, false)).collect();
        while let Some((index, expanded)) = stack.pop() {
            if expanded {
                let entry =
                    std::mem::replace(&mut data[index].entry, FileEntry::Other(JsonValue::Null));
                let label = data[index].label.take();
                let label_timestamp = data[index].label_timestamp.take();
                let children = std::mem::take(&mut data[index].assembled);
                let parent_index = data[index].parent_index;
                let mut assembled = SessionTreeNode {
                    entry,
                    children,
                    label,
                    label_timestamp,
                };
                assembled.children.sort_by(|left, right| {
                    match (
                        parse_iso8601_millis(&left.entry.entry_timestamp()),
                        parse_iso8601_millis(&right.entry.entry_timestamp()),
                    ) {
                        (Some(left), Some(right)) => left.cmp(&right),
                        // Upstream's comparator returns NaN on unparseable
                        // timestamps, which `Array.prototype.sort` treats as
                        // equal.
                        _ => Ordering::Equal,
                    }
                });
                match parent_index {
                    Some(parent) => data[parent].assembled.push(assembled),
                    None => ordered_roots.push(assembled),
                }
            } else {
                stack.push((index, true));
                for child in data[index].child_indices.iter().rev() {
                    stack.push((*child, false));
                }
            }
        }

        ordered_roots
    }

    /// Start a new branch from an earlier entry, upstream's `branch`. Moves
    /// the leaf pointer; existing entries are not modified or deleted.
    ///
    /// # Errors
    /// [`super::file_entry::SessionManagerError::EntryNotFound`] when the id
    /// is unknown.
    pub fn branch(
        &mut self,
        branch_from_id: &str,
    ) -> Result<(), super::file_entry::SessionManagerError> {
        if !self.by_id.contains_key(branch_from_id) {
            return Err(super::file_entry::SessionManagerError::EntryNotFound(
                branch_from_id.to_owned(),
            ));
        }
        self.leaf_id = Some(branch_from_id.to_owned());
        Ok(())
    }

    /// Reset the leaf pointer to null (before any entries), upstream's
    /// `resetLeaf`: the next append creates a new root entry. Use this when
    /// navigating to re-edit the first user message.
    pub fn reset_leaf(&mut self) {
        self.leaf_id = None;
    }

    /// Start a new branch with a summary of the abandoned path, upstream's
    /// `branchWithSummary`. Same as [`SessionManager::branch`], but also
    /// appends a `branch_summary` entry that captures context from the
    /// abandoned conversation path.
    ///
    /// # Errors
    /// [`super::file_entry::SessionManagerError::EntryNotFound`] when
    /// `branch_from_id` is given and unknown.
    pub fn branch_with_summary(
        &mut self,
        branch_from_id: Option<&str>,
        summary: &str,
        details: Option<JsonValue>,
        from_hook: Option<bool>,
        usage: Option<Usage>,
    ) -> Result<String, super::file_entry::SessionManagerError> {
        if let Some(id) = branch_from_id
            && !self.by_id.contains_key(id)
        {
            return Err(super::file_entry::SessionManagerError::EntryNotFound(
                id.to_owned(),
            ));
        }
        let from_id = self.leaf_id.clone().unwrap_or_else(|| "root".to_owned());
        self.leaf_id = branch_from_id.map(str::to_owned);
        let entry = SessionEntry::BranchSummary(BranchSummaryEntry {
            base: SessionEntryBase {
                id: Some(super::generate_id(|id| self.by_id.contains_key(id))),
                parent_id: branch_from_id.map(str::to_owned),
                timestamp: now_iso8601(),
                extras: serde_json::Map::default(),
            },
            from_id,
            summary: summary.to_owned(),
            details,
            usage,
            from_hook,
            extras: serde_json::Map::default(),
        });
        self.append_entry(entry)
    }
}

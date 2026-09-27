//! The complete materialized session state behind the memory backend,
//! ported from upstream `src/harness/session/in-memory-storage-state.ts`.
//!
//! Upstream keeps the class intentionally unsuitable for database backends
//! and long-running sessions; the port carries the same shape so the JSONL
//! child shares it. Upstream's insertion-ordered maps restate as a
//! hash-plus-sequence-vector pair for entries and sort-on-read maps for
//! values and usage; key ordering uses Rust's `str` ordering, which matches
//! upstream's code-point comparison.
//!
//! Upstream serializes commits through a promise queue; the port applies
//! each commit synchronously under the caller's lock, so admission order is
//! call order and no queue remains.

use std::collections::{BTreeMap, HashMap};

use crate::harness::session::commit::{
    CommitValidationState, CommittedWrite, PreparedCommit, prepare_storage_commit,
    validate_committed_writes,
};
use crate::harness::session::fork_policy::{
    BranchForkSource, ForkCurrentStatePlan, ForkStateWrite, project_fork_current_state_write,
    select_branch_fork,
};
use crate::harness::session::types::{
    BranchScanOrder, Entry, EntryScan, EntryScanOrder, EntryStructure, EntryType, ForkOptions,
    SessionError, SessionStats, StorageBranchScan, UsageRow, UsageScan,
};
use crate::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress, Write, branch_tip,
    lane_config, lane_state, resolve_list_read_options,
};
use crate::harness::utils::usage::{add_usage, empty_usage};

/// One stored list's elements plus the address the fork projection reads,
/// upstream's `StoredListSnapshot`.
#[derive(Clone, Debug, PartialEq)]
struct StoredListSnapshot {
    namespace: String,
    key: String,
    elements: Vec<ListElement>,
}

/// The fork plan one in-memory fork walks, upstream's `MemoryForkPlan`: the
/// branch plan with the copied entry ids, or the whole tree.
enum MemoryForkPlan {
    /// Copy the whole tree.
    Tree,
    /// Copy one branch's selected path.
    Branch {
        /// The copied branch name.
        branch: String,
        /// The destination tip after the copy.
        destination_tip: Option<String>,
        /// The selected entry ids.
        entry_ids: std::collections::BTreeSet<String>,
    },
}

/// The physical map key one addressed row lives under, upstream's
/// `physicalKey`.
fn physical_key(namespace: &str, key: &str) -> String {
    format!("{namespace}\u{0}{key}")
}

/// Complete materialized session state for the memory and JSONL backends,
/// upstream's `InMemoryStorageState`.
#[derive(Debug)]
pub struct InMemoryStorageState {
    entries: HashMap<String, Entry>,
    entries_by_seq: Vec<Entry>,
    scalar_values: BTreeMap<String, StoredValue>,
    list_values: BTreeMap<String, StoredListSnapshot>,
    usage: BTreeMap<String, UsageRow>,
    stats: SessionStats,
    next_seq: u64,
}

impl Default for InMemoryStorageState {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            entries_by_seq: Vec::new(),
            scalar_values: BTreeMap::new(),
            list_values: BTreeMap::new(),
            usage: BTreeMap::new(),
            stats: SessionStats {
                message_count: 0,
                usage: empty_usage(),
            },
            next_seq: 1,
        }
    }
}

impl InMemoryStorageState {
    /// Empty state with the sequence counter at 1.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare one commit against the current sequence, upstream's
    /// `prepareCommit`.
    ///
    /// # Errors
    /// The duplicate/parent validation failures of
    /// [`validate_committed_writes`].
    pub fn prepare_commit(
        &mut self,
        writes: Vec<Write>,
        timestamp: i64,
    ) -> Result<PreparedCommit, SessionError> {
        let prepared = prepare_storage_commit(writes, self.next_seq, timestamp);
        self.validate_committed(&prepared.writes)?;
        Ok(prepared)
    }

    /// Validate committed writes against this state, upstream's
    /// `validateCommitted`.
    ///
    /// # Errors
    /// The duplicate/parent validation failures of
    /// [`validate_committed_writes`].
    pub fn validate_committed(&self, writes: &[CommittedWrite]) -> Result<(), SessionError> {
        validate_committed_writes(writes, self.next_seq, self)
    }

    /// Apply writes already accepted by [`Self::validate_committed`] and
    /// return the post-apply totals, upstream's `applyValidated`.
    pub fn apply_validated(&mut self, writes: &[CommittedWrite]) -> SessionStats {
        for write in writes {
            match write {
                CommittedWrite::Entry { entry } => {
                    if entry.entry_type() == EntryType::Message {
                        self.stats.message_count += 1;
                    }
                    let entry = entry.clone();
                    self.entries.insert(entry.id().to_owned(), entry.clone());
                    self.entries_by_seq.push(entry);
                }
                CommittedWrite::Usage { row } => {
                    self.stats.usage = add_usage(self.stats.usage, row.usage);
                    self.usage.insert(row.id.clone(), row.clone());
                }
                CommittedWrite::ValueDelete { namespace, key, .. } => {
                    self.scalar_values.remove(&physical_key(namespace, key));
                }
                CommittedWrite::ListDelete { namespace, key, .. } => {
                    self.list_values.remove(&physical_key(namespace, key));
                }
                CommittedWrite::ValueSet {
                    seq,
                    namespace,
                    key,
                    value,
                } => {
                    self.apply_fork_state_write(&ForkStateWrite::ValueSet {
                        seq: *seq,
                        namespace: namespace.clone(),
                        key: key.clone(),
                        value: value.clone(),
                    });
                }
                CommittedWrite::ListAppend {
                    seq,
                    namespace,
                    key,
                    value,
                } => {
                    self.apply_fork_state_write(&ForkStateWrite::ListAppend {
                        seq: *seq,
                        namespace: namespace.clone(),
                        key: key.clone(),
                        value: value.clone(),
                    });
                }
            }
            self.next_seq = write.seq() + 1;
        }
        self.stats.clone()
    }

    /// Build the destination state for one fork, upstream's `createFork`.
    ///
    /// # Errors
    /// A `SessionError::Message` when the source branch is unknown, its
    /// ancestry is corrupt, the requested entry is off the branch, the
    /// branch is not a configured `AgentLane`, or a surviving write sits in
    /// an unknown reserved namespace.
    pub fn create_fork(&self, options: &ForkOptions) -> Result<Self, SessionError> {
        let plan = self.select_fork_plan(options)?;

        let is_entry_copied = |entry_id: &str| match &plan {
            MemoryForkPlan::Tree => true,
            MemoryForkPlan::Branch { entry_ids, .. } => entry_ids.contains(entry_id),
        };
        let fork_plan = fork_current_state_plan(&plan);
        let mut destination = Self::default();
        let mut message_count = 0;
        for entry in &self.entries_by_seq {
            if !is_entry_copied(entry.id()) {
                continue;
            }
            if entry.entry_type() == EntryType::Message {
                message_count += 1;
            }
            destination
                .entries
                .insert(entry.id().to_owned(), entry.clone());
            destination.entries_by_seq.push(entry.clone());
        }
        destination.stats.message_count = message_count;

        for stored in self.scalar_values.values() {
            let projected = project_fork_current_state_write(
                &ForkStateWrite::ValueSet {
                    seq: stored.seq,
                    namespace: stored.namespace.clone(),
                    key: stored.key.clone(),
                    value: stored.value.clone(),
                },
                &fork_plan,
                &is_entry_copied,
            )?;
            if let Some(projected) = projected {
                destination.apply_fork_state_write(&projected);
            }
        }

        for stored in self.list_values.values() {
            for element in &stored.elements {
                let projected = project_fork_current_state_write(
                    &ForkStateWrite::ListAppend {
                        seq: element.seq,
                        namespace: stored.namespace.clone(),
                        key: stored.key.clone(),
                        value: element.value.clone(),
                    },
                    &fork_plan,
                    &is_entry_copied,
                )?;
                if let Some(projected) = projected {
                    destination.apply_fork_state_write(&projected);
                }
            }
        }
        destination.next_seq = self.next_seq;
        Ok(destination)
    }

    fn select_fork_plan(&self, options: &ForkOptions) -> Result<MemoryForkPlan, SessionError> {
        let ForkOptions::Branch { branch, .. } = options else {
            return Ok(MemoryForkPlan::Tree);
        };
        let mut entry_ids = std::collections::BTreeSet::new();
        let get_parent = |entry_id: &str| {
            self.entries
                .get(entry_id)
                .map(|entry| entry.parent_id().map(str::to_owned))
        };
        let mut select_entry = |entry_id: &str| {
            entry_ids.insert(entry_id.to_owned());
        };
        let tip = self
            .get_value(&branch_tip(branch).address)
            .map(|stored| stored.value.as_str().map(str::to_owned));
        let plan = select_branch_fork(
            options,
            &mut BranchForkSource {
                tip,
                get_parent: &get_parent,
                select_entry: &mut select_entry,
            },
        )?;
        if self.get_value(&lane_config(branch).address).is_none()
            || self.get_value(&lane_state(branch).address).is_none()
        {
            return Err(SessionError::Message(format!(
                "Source branch {branch:?} is not a configured AgentLane"
            )));
        }
        Ok(match plan {
            ForkCurrentStatePlan::Tree => MemoryForkPlan::Tree,
            ForkCurrentStatePlan::Branch {
                branch: plan_branch,
                destination_tip,
            } => MemoryForkPlan::Branch {
                branch: plan_branch,
                destination_tip,
                entry_ids,
            },
        })
    }

    fn apply_fork_state_write(&mut self, write: &ForkStateWrite) {
        match write {
            ForkStateWrite::ValueSet {
                seq,
                namespace,
                key,
                value,
            } => {
                self.scalar_values.insert(
                    physical_key(namespace, key),
                    StoredValue {
                        namespace: namespace.clone(),
                        key: key.clone(),
                        value: value.clone(),
                        seq: *seq,
                    },
                );
            }
            ForkStateWrite::ListAppend {
                seq,
                namespace,
                key,
                value,
            } => {
                let element = ListElement {
                    seq: *seq,
                    value: value.clone(),
                };
                let stored = self
                    .list_values
                    .entry(physical_key(namespace, key))
                    .or_insert_with(|| StoredListSnapshot {
                        namespace: namespace.clone(),
                        key: key.clone(),
                        elements: Vec::new(),
                    });
                stored.elements.push(element);
            }
        }
    }

    /// Raise the sequence high-water mark, upstream's `advanceNextSeq` (the
    /// JSONL backend's reopen path).
    ///
    /// # Errors
    /// A `SessionError::Message` when the mark is below 1.
    pub fn advance_next_seq(&mut self, next_seq: u64) -> Result<(), SessionError> {
        if next_seq < 1 {
            return Err(SessionError::Message(format!(
                "Invalid storage sequence high-water mark: {next_seq}"
            )));
        }
        self.next_seq = self.next_seq.max(next_seq);
        Ok(())
    }

    /// Named entries by id, upstream's `getEntries`.
    #[must_use]
    pub fn get_entries(&self, ids: &[String]) -> BTreeMap<String, Entry> {
        let mut found = BTreeMap::new();
        for id in ids {
            if let Some(entry) = self.entries.get(id) {
                found.insert(id.clone(), entry.clone());
            }
        }
        found
    }

    /// One stored value, upstream's `getValue`.
    #[must_use]
    pub fn get_value(&self, address: &ValueAddress) -> Option<StoredValue> {
        self.scalar_values
            .get(&physical_key(&address.namespace, &address.key))
            .cloned()
    }

    /// Values under a prefix address, upstream's `scanValues`, key-ordered.
    #[must_use]
    pub fn scan_values(&self, prefix: &ValueAddress) -> Vec<StoredValue> {
        let mut stored: Vec<StoredValue> = self
            .scalar_values
            .values()
            .filter(|stored| {
                stored.namespace == prefix.namespace && stored.key.starts_with(&prefix.key)
            })
            .cloned()
            .collect();
        stored.sort_by(|left, right| left.key.cmp(&right.key));
        stored
    }

    /// One list's elements, upstream's `readList`.
    ///
    /// # Errors
    /// A `SessionError::Message` when the read options are invalid.
    pub fn read_list(
        &self,
        address: &ListAddress,
        options: Option<ListReadOptions>,
    ) -> Result<Vec<ListElement>, SessionError> {
        let resolved = resolve_list_read_options(options)?;
        let empty: Vec<ListElement> = Vec::new();
        let elements = self
            .list_values
            .get(&physical_key(&address.namespace, &address.key))
            .map_or(empty.as_slice(), |stored| stored.elements.as_slice());
        let mut filtered: Vec<ListElement> = elements
            .iter()
            .filter(|element| {
                resolved.cursor.is_none_or(|cursor| {
                    if resolved.order == EntryScanOrder::Asc {
                        element.seq > cursor.seq
                    } else {
                        element.seq < cursor.seq
                    }
                })
            })
            .cloned()
            .collect();
        if resolved.order == EntryScanOrder::Desc {
            filtered.reverse();
        }
        Ok(filtered
            .into_iter()
            .take(usize::try_from(resolved.limit).unwrap_or(usize::MAX))
            .collect())
    }

    /// Walk one branch path, upstream's `scanBranch`.
    ///
    /// # Errors
    /// A `SessionError::Message` for an unknown start or a corrupt
    /// ancestry.
    pub fn scan_branch(&self, query: &StorageBranchScan) -> Result<Vec<Entry>, SessionError> {
        let start = self
            .entries
            .get(&query.start)
            .ok_or_else(|| SessionError::Message(format!("Unknown branch start: {}", query.start)))?
            .clone();

        let mut path: Vec<Entry> = Vec::new();
        let mut current = start;
        loop {
            path.push(current.clone());
            let Some(parent_id) = current.parent_id() else {
                break;
            };
            current = self
                .entries
                .get(parent_id)
                .ok_or_else(|| SessionError::Message("Corrupt branch: missing parent".to_owned()))?
                .clone();
        }
        if query.order == Some(BranchScanOrder::OldestFirst) {
            path.reverse();
        }

        let mut stopped: Vec<Entry> = Vec::new();
        for candidate in &path {
            stopped.push(candidate.clone());
            if Some(candidate.id()) == query.stop_at_id.as_deref()
                || query
                    .stop_at_type
                    .is_some_and(|stop_at| candidate.entry_type() == stop_at)
            {
                break;
            }
        }
        let filtered: Vec<Entry> = stopped
            .into_iter()
            .filter(|candidate| query.kind.is_none_or(|kind| candidate.entry_type() == kind))
            .filter(|candidate| {
                query
                    .custom_type
                    .as_deref()
                    .is_none_or(|custom_type| candidate.custom_type() == Some(custom_type))
            })
            .filter(|candidate| {
                query.cursor.is_none_or(|cursor| {
                    if query.order == Some(BranchScanOrder::OldestFirst) {
                        candidate.seq() > cursor.seq
                    } else {
                        candidate.seq() < cursor.seq
                    }
                })
            })
            .collect();
        Ok(match query.limit {
            None => filtered,
            Some(limit) => filtered
                .into_iter()
                .take(usize::try_from(limit).unwrap_or(usize::MAX))
                .collect(),
        })
    }

    /// The structural view of a branch scan, upstream's
    /// `scanBranchStructure`.
    ///
    /// # Errors
    /// The [`Self::scan_branch`] failures.
    pub fn scan_branch_structure(
        &self,
        query: &StorageBranchScan,
    ) -> Result<Vec<EntryStructure>, SessionError> {
        Ok(self
            .scan_branch(query)?
            .into_iter()
            .map(|entry| EntryStructure {
                id: entry.id().to_owned(),
                parent_id: entry.parent_id().map(str::to_owned),
                seq: entry.seq(),
                timestamp: entry.timestamp(),
                kind: entry.entry_type(),
                custom_type: entry.custom_type().map(str::to_owned),
            })
            .collect())
    }

    /// A flat entry scan, upstream's `scanEntries`.
    #[must_use]
    pub fn scan_entries(&self, query: &EntryScan) -> Vec<Entry> {
        let limit = usize::try_from(query.limit.unwrap_or(u64::MAX)).unwrap_or(usize::MAX);
        let mut entries: Vec<Entry> = Vec::new();
        let descending = query.order == Some(EntryScanOrder::Desc);
        let indices: Vec<usize> = if descending {
            (0..self.entries_by_seq.len()).rev().collect()
        } else {
            (0..self.entries_by_seq.len()).collect()
        };
        for index in indices {
            if entries.len() >= limit {
                break;
            }
            let entry = &self.entries_by_seq[index];
            if query.kind.is_none_or(|kind| entry.entry_type() == kind)
                && query
                    .custom_type
                    .as_deref()
                    .is_none_or(|custom_type| entry.custom_type() == Some(custom_type))
                && query
                    .from_seq
                    .is_none_or(|from_seq| entry.seq() >= from_seq)
                && query.to_seq.is_none_or(|to_seq| entry.seq() <= to_seq)
            {
                entries.push(entry.clone());
            }
        }
        entries
    }

    /// A flat usage scan, upstream's `scanUsage`.
    #[must_use]
    pub fn scan_usage(&self, query: &UsageScan) -> Vec<UsageRow> {
        let mut rows: Vec<UsageRow> = self
            .usage
            .values()
            .filter(|row| query.from_seq.is_none_or(|from_seq| row.seq >= from_seq))
            .filter(|row| query.to_seq.is_none_or(|to_seq| row.seq <= to_seq))
            .cloned()
            .collect();
        rows.sort_by(|left, right| match query.order {
            Some(EntryScanOrder::Desc) => right.seq.cmp(&left.seq),
            _ => left.seq.cmp(&right.seq),
        });
        match query.limit {
            None => rows,
            Some(limit) => rows
                .into_iter()
                .take(usize::try_from(limit).unwrap_or(usize::MAX))
                .collect(),
        }
    }

    /// The session totals, upstream's `getStats`.
    #[must_use]
    pub fn get_stats(&self) -> SessionStats {
        self.stats.clone()
    }

    /// The next storage sequence, upstream's `getNextSeq`.
    #[must_use]
    pub const fn get_next_seq(&self) -> u64 {
        self.next_seq
    }
}

impl CommitValidationState for InMemoryStorageState {
    fn has_entry_or_usage_id(&self, id: &str) -> bool {
        self.entries.contains_key(id) || self.usage.contains_key(id)
    }

    fn has_entry_id(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }
}

fn fork_current_state_plan(plan: &MemoryForkPlan) -> ForkCurrentStatePlan {
    match plan {
        MemoryForkPlan::Tree => ForkCurrentStatePlan::Tree,
        MemoryForkPlan::Branch {
            branch,
            destination_tip,
            ..
        } => ForkCurrentStatePlan::Branch {
            branch: branch.clone(),
            destination_tip: destination_tip.clone(),
        },
    }
}

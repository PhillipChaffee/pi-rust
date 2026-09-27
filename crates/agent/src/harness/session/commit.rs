//! The storage commit pipeline, ported from upstream
//! `src/harness/session/commit.ts`.
//!
//! It carries the committed-write union with storage-assigned sequence and
//! timestamp, the prepare step, and the duplicate-id/parent validation
//! every backend runs against its own state.

use serde::{Deserialize, Serialize};

use crate::harness::session::types::{Entry, NewEntry, SessionError, UsageRow, UsageWriteRow};
use crate::harness::session::values::{EntryWrite, UsageWrite, Write};

/// One committed write, upstream's `CommittedWrite` union.
///
/// The write families carry their storage-assigned sequence. The entry and
/// usage families hold their materialized records; the value and list
/// families hold the addressed payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all_fields = "camelCase")]
pub enum CommittedWrite {
    /// An inserted entry, wire `"kind": "entry"`.
    #[serde(rename = "entry")]
    Entry {
        /// The materialized entry.
        entry: Entry,
    },
    /// An inserted usage row, wire `"kind": "usage"`.
    #[serde(rename = "usage")]
    Usage {
        /// The materialized usage row.
        row: UsageRow,
    },
    /// A value set, wire `"kind": "value"`, `"op": "set"`.
    #[serde(rename = "value_set")]
    ValueSet {
        /// The write's sequence.
        seq: u64,
        /// The addressed namespace.
        namespace: String,
        /// The addressed key.
        key: String,
        /// The stored value.
        value: serde_json::Value,
    },
    /// A value delete, wire `"kind": "value"`, `"op": "delete"`.
    #[serde(rename = "value_delete")]
    ValueDelete {
        /// The write's sequence.
        seq: u64,
        /// The addressed namespace.
        namespace: String,
        /// The addressed key.
        key: String,
    },
    /// A list append, wire `"kind": "list"`, `"op": "append"`.
    #[serde(rename = "list_append")]
    ListAppend {
        /// The write's sequence.
        seq: u64,
        /// The addressed namespace.
        namespace: String,
        /// The addressed key.
        key: String,
        /// The appended element.
        value: serde_json::Value,
    },
    /// A list delete, wire `"kind": "list"`, `"op": "delete"`.
    #[serde(rename = "list_delete")]
    ListDelete {
        /// The write's sequence.
        seq: u64,
        /// The addressed namespace.
        namespace: String,
        /// The addressed key.
        key: String,
    },
}

impl CommittedWrite {
    /// The write's storage-assigned sequence, upstream's `write.seq`.
    #[must_use]
    pub const fn seq(&self) -> u64 {
        match self {
            Self::Entry { entry } => entry.seq(),
            Self::Usage { row } => row.seq,
            Self::ValueSet { seq, .. }
            | Self::ValueDelete { seq, .. }
            | Self::ListAppend { seq, .. }
            | Self::ListDelete { seq, .. } => *seq,
        }
    }

    /// The stable record id the entry and usage families carry, upstream's
    /// `write.id`; `None` for the value and list families.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        match self {
            Self::Entry { entry } => Some(entry.id()),
            Self::Usage { row } => Some(&row.id),
            _ => None,
        }
    }

    /// The entry family's parent id, upstream's `write.parentId`; `None`
    /// for every other family.
    #[must_use]
    pub fn parent_id(&self) -> Option<&str> {
        match self {
            Self::Entry { entry } => entry.parent_id(),
            _ => None,
        }
    }

    /// Whether the write is the entry family, upstream's
    /// `write.kind === "entry"`.
    #[must_use]
    pub const fn is_entry(&self) -> bool {
        matches!(self, Self::Entry { .. })
    }
}

/// The commit a backend validates before applying, upstream's
/// `PreparedCommit`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedCommit {
    /// The committed writes in write order.
    pub writes: Vec<CommittedWrite>,
    /// The sequence and timestamp assignment, upstream's
    /// `Omit<CommitResult, "stats">`.
    pub result: PreparedCommitResult,
}

/// The sequence and timestamp assignment one commit carries before the
/// backend's post-apply totals, upstream's `Omit<CommitResult, "stats">`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedCommitResult {
    /// The first sequence the commit assigned.
    pub first_seq: u64,
    /// Every sequence the commit assigned, in write order.
    pub seqs: Vec<u64>,
    /// The storage-assigned timestamp, Unix epoch milliseconds.
    pub timestamp: i64,
}

/// The duplicate/parent checks one backend answers from its own state,
/// upstream's `CommitValidationState`.
pub trait CommitValidationState {
    /// Whether an entry or usage row already carries the id.
    fn has_entry_or_usage_id(&self, id: &str) -> bool;

    /// Whether an entry already carries the id.
    fn has_entry_id(&self, id: &str) -> bool;
}

/// Builds an entry-insert write, upstream's `insertEntry`.
#[must_use]
pub fn insert_entry(entry: NewEntry) -> EntryWrite {
    EntryWrite {
        kind: "entry".to_owned(),
        entry,
    }
}

/// Builds a usage-insert write, upstream's `insertUsage`.
#[must_use]
pub fn insert_usage(row: UsageWriteRow) -> UsageWrite {
    UsageWrite {
        kind: "usage".to_owned(),
        row,
    }
}

/// Materializes one write with its assigned sequence, upstream's
/// `commitWrite`.
#[must_use]
pub fn commit_write(write: Write, seq: u64, timestamp: i64) -> CommittedWrite {
    match write {
        Write::Entry(entry_write) => CommittedWrite::Entry {
            entry: entry_write.entry.materialize(seq, timestamp),
        },
        Write::Usage(usage_write) => CommittedWrite::Usage {
            row: UsageRow {
                id: usage_write.row.id,
                seq,
                usage: usage_write.row.usage,
                entry_id: usage_write.row.entry_id,
                adjustment: usage_write.row.adjustment,
                details: usage_write.row.details,
            },
        },
        Write::ValueSet(value_set) => CommittedWrite::ValueSet {
            seq,
            namespace: value_set.namespace,
            key: value_set.key,
            value: value_set.value,
        },
        Write::ValueDelete(value_delete) => CommittedWrite::ValueDelete {
            seq,
            namespace: value_delete.namespace,
            key: value_delete.key,
        },
        Write::ListAppend(list_append) => CommittedWrite::ListAppend {
            seq,
            namespace: list_append.namespace,
            key: list_append.key,
            value: list_append.value,
        },
        Write::ListDelete(list_delete) => CommittedWrite::ListDelete {
            seq,
            namespace: list_delete.namespace,
            key: list_delete.key,
        },
    }
}

/// Assigns sequences and the timestamp to a transaction's writes, upstream's
/// `prepareStorageCommit`.
#[must_use]
pub fn prepare_storage_commit(
    writes: Vec<Write>,
    first_seq: u64,
    timestamp: i64,
) -> PreparedCommit {
    let committed_writes: Vec<CommittedWrite> = writes
        .into_iter()
        .enumerate()
        .map(|(index, write)| commit_write(write, first_seq + index as u64, timestamp))
        .collect();
    PreparedCommit {
        result: PreparedCommitResult {
            first_seq,
            seqs: committed_writes.iter().map(CommittedWrite::seq).collect(),
            timestamp,
        },
        writes: committed_writes,
    }
}

/// Validates committed writes against a backend's state, upstream's
/// `validateCommittedWrites`.
///
/// # Errors
/// A `SessionError::Message` for a non-monotonic sequence, a duplicate
/// entry/usage id, or a missing parent entry.
pub fn validate_committed_writes(
    writes: &[CommittedWrite],
    first_seq: u64,
    state: &dyn CommitValidationState,
) -> Result<(), SessionError> {
    let mut previous_seq = first_seq.checked_sub(1);
    let mut transaction_ids: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut transaction_entry_ids: std::collections::HashSet<&str> =
        std::collections::HashSet::new();
    for write in writes {
        if previous_seq.is_some_and(|previous| write.seq() <= previous) {
            return Err(SessionError::Message(format!(
                "Non-monotonic storage sequence: {}",
                write.seq()
            )));
        }
        previous_seq = Some(write.seq());
        let Some(id) = write.id() else {
            continue;
        };
        if state.has_entry_or_usage_id(id) || transaction_ids.contains(id) {
            return Err(SessionError::Message(format!(
                "Duplicate entry or usage id: {id}"
            )));
        }
        if write.is_entry()
            && let Some(parent_id) = write.parent_id()
            && !state.has_entry_id(parent_id)
            && !transaction_entry_ids.contains(parent_id)
        {
            return Err(SessionError::Message(format!(
                "Missing parent entry: {parent_id}"
            )));
        }
        transaction_ids.insert(id);
        if write.is_entry() {
            transaction_entry_ids.insert(id);
        }
    }
    Ok(())
}

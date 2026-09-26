//! The snapshot-based fork builder, ported from upstream
//! `src/harness/session/fork.ts`.
//!
//! It materializes the logical destination state from a complete source
//! snapshot. The memory backend walks its live state directly; streaming
//! backends that supply a bounded snapshot build through this module.

use std::collections::BTreeMap;

use crate::harness::session::fork_policy::{
    BranchForkSource, ForkCurrentStatePlan, ForkStateWrite, project_fork_current_state_write,
    select_branch_fork,
};
use crate::harness::session::types::{Entry, ForkOptions, SessionError};
use crate::harness::session::values::{StoredValue, lane_config_namespace, lane_state_namespace};

/// The source state one snapshot fork reads, upstream's
/// `ForkSourceSnapshot`.
#[derive(Clone, Debug, Default)]
pub struct ForkSourceSnapshot {
    /// The source conversation tree, complete unless
    /// [`ForkSourceSnapshot::entries_complete`] says otherwise.
    pub entries: Vec<Entry>,
    /// The source's surviving scalar rows.
    pub scalar_values: Vec<StoredValue>,
    /// `Some(false)` when the backend supplied only the requested branch
    /// rather than the full tree, upstream's `entriesComplete?`.
    pub entries_complete: Option<bool>,
}

/// The complete logical state for a forked destination session, upstream's
/// `ForkDestinationSnapshot`.
#[derive(Clone, Debug, Default)]
pub struct ForkDestinationSnapshot {
    /// The copied conversation tree.
    pub entries: BTreeMap<String, Entry>,
    /// The projected surviving scalar rows, with fresh destination
    /// sequences.
    pub scalar_values: Vec<StoredValue>,
    /// The destination's next storage sequence.
    pub next_seq: u64,
}

/// Builds the complete logical state for a forked destination session,
/// upstream's `createForkSnapshot`.
///
/// # Errors
/// A `SessionError::Message` when the source snapshot is inconsistent for
/// the requested scope: a configured lane without a branch tip, an
/// incomplete lane, an unknown tip, or a surviving unknown reserved
/// namespace.
pub fn create_fork_snapshot(
    source: &ForkSourceSnapshot,
    options: &ForkOptions,
) -> Result<ForkDestinationSnapshot, SessionError> {
    let source_entries: BTreeMap<String, Entry> = source
        .entries
        .iter()
        .map(|entry| (entry.id().to_owned(), entry.clone()))
        .collect();
    let source_tips: Vec<&StoredValue> = source
        .scalar_values
        .iter()
        .filter(|stored| stored.namespace == "pi.branch.tip")
        .collect();
    validate_fork_source_snapshot(source, &source_entries, &source_tips, options)?;

    let (entry_ids, plan) = select_fork_contents(&source_entries, &source_tips, options)?;
    let entries: BTreeMap<String, Entry> = entry_ids
        .iter()
        .map(|id| {
            let entry = source_entries
                .get(id)
                .cloned()
                .unwrap_or_else(|| unreachable_selected_entry(id));
            (id.clone(), entry)
        })
        .collect();

    let mut scalar_values: Vec<StoredValue> = Vec::new();
    let mut next_seq = entries.values().map(Entry::seq).max().unwrap_or(0) + 1;
    for stored in &source.scalar_values {
        let projected = project_fork_current_state_write(
            &ForkStateWrite::ValueSet {
                seq: stored.seq,
                namespace: stored.namespace.clone(),
                key: stored.key.clone(),
                value: stored.value.clone(),
            },
            &plan,
            &|entry_id: &str| entry_ids.contains(entry_id),
        )?;
        if let Some(ForkStateWrite::ValueSet {
            namespace,
            key,
            value,
            ..
        }) = projected
        {
            scalar_values.push(StoredValue {
                namespace,
                key,
                value,
                seq: next_seq,
            });
            next_seq += 1;
        }
    }

    Ok(ForkDestinationSnapshot {
        entries,
        scalar_values,
        next_seq,
    })
}

fn select_fork_contents(
    source_entries: &BTreeMap<String, Entry>,
    source_tips: &[&StoredValue],
    options: &ForkOptions,
) -> Result<(std::collections::BTreeSet<String>, ForkCurrentStatePlan), SessionError> {
    let mut entry_ids = std::collections::BTreeSet::new();
    match options {
        ForkOptions::Tree { .. } => {
            entry_ids.extend(source_entries.keys().cloned());
            Ok((entry_ids, ForkCurrentStatePlan::Tree))
        }
        ForkOptions::Branch { branch, .. } => {
            let source_tip = source_tips
                .iter()
                .find(|stored| stored.key == *branch)
                .map(|stored| stored.value.as_str().map(str::to_owned));
            let mut source = BranchForkSource {
                tip: source_tip,
                get_parent: &|entry_id: &str| {
                    source_entries
                        .get(entry_id)
                        .map(|entry| entry.parent_id().map(str::to_owned))
                },
                select_entry: &mut |entry_id: &str| {
                    entry_ids.insert(entry_id.to_owned());
                },
            };
            let plan = select_branch_fork(options, &mut source)?;
            Ok((entry_ids, plan))
        }
    }
}

fn validate_fork_source_snapshot(
    source: &ForkSourceSnapshot,
    source_entries: &BTreeMap<String, Entry>,
    source_tips: &[&StoredValue],
    options: &ForkOptions,
) -> Result<(), SessionError> {
    let source_tip_keys: std::collections::BTreeSet<&str> = source_tips
        .iter()
        .map(|stored| stored.key.as_str())
        .collect();

    for stored in &source.scalar_values {
        if (stored.namespace == lane_config_namespace()
            || stored.namespace == lane_state_namespace())
            && !source_tip_keys.contains(stored.key.as_str())
        {
            return Err(SessionError::Message(format!(
                "Source session branch {:?} is missing branch.tip",
                stored.key
            )));
        }
    }
    for tip in source_tips {
        let find = |namespace: &str| {
            source
                .scalar_values
                .iter()
                .any(|stored| stored.namespace == namespace && stored.key == tip.key)
        };
        let configuration = find(lane_config_namespace());
        let state = find(lane_state_namespace());
        if configuration != state {
            return Err(SessionError::Message(format!(
                "Source session branch {:?} has incomplete lane state",
                tip.key
            )));
        }
        if let ForkOptions::Branch { branch, .. } = options
            && tip.key == *branch
            && !configuration
        {
            return Err(SessionError::Message(format!(
                "Source branch {branch:?} is not a configured AgentLane"
            )));
        }
        if (source.entries_complete.unwrap_or(true) || matches!(options, ForkOptions::Tree { .. }))
            && let Some(tip_value) = tip.value.as_str()
            && !source_entries.contains_key(tip_value)
        {
            return Err(SessionError::Message(format!(
                "Source session branch {:?} has an unknown tip",
                tip.key
            )));
        }
    }
    Ok(())
}

#[expect(
    clippy::panic,
    reason = "select_fork_contents selects exactly the ids it read from the source map; the fallback is unreachable by construction"
)]
fn unreachable_selected_entry(id: &str) -> ! {
    panic!("selected fork entry {id} missing from the source snapshot")
}

//! The JSONL fork, ported from upstream
//! `src/harness/session/jsonl/fork.ts`.
//!
//! One pass indexes the source (parent links, current-row sequences, lane
//! inventory — not payloads), validates the requested fork, and a second
//! pass streams only the selected writes into an atomically published
//! format-4 destination. Copied sequences and the source's nextSeq are
//! preserved; usage and open-operation state are excluded. Source files
//! must not be replaced or edited between passes; later append-only writes
//! are excluded by the captured sequence boundary or the legacy record
//! count. Upstream's async generators restate as eager collections; each
//! pass still reads the file exactly once.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::Value as JsonValue;

use crate::harness::context::Context;
use crate::harness::session::commit::CommittedWrite;
use crate::harness::session::fork_policy::{
    BranchForkSource, ForkCurrentStatePlan, ForkStateWrite, project_fork_current_state_write,
    select_branch_fork,
};
use crate::harness::session::jsonl::io::{
    JsonlError, file_value, parse_jsonl_transaction, publish_jsonl, read_jsonl_header,
};
use crate::harness::session::jsonl::legacy_v3::LegacyV3Source;
use crate::harness::session::jsonl::types::{JSONL_STORAGE_VERSION, JsonlStorageHeader};
use crate::harness::session::types::ForkOptions;
use crate::harness::types::{FileSystem, TextLineReader};

/// The source identity one fork reads, upstream's `JsonlForkSourceMetadata`.
#[derive(Clone, Debug)]
pub struct JsonlForkSourceMetadata {
    /// The session id.
    pub id: String,
    /// The session's working directory.
    pub cwd: String,
    /// The session file's addressed path.
    pub path: String,
}

fn physical_key(namespace: &str, key: &str) -> String {
    format!("{namespace}\u{0}{key}")
}

/// The index one fork builds, upstream's `JsonlForkIndex`: entry parent
/// links, current-row sequences, and lane inventory, not entry payloads.
#[derive(Debug, Default)]
struct JsonlForkIndex {
    current_scalar_seqs: BTreeMap<String, u64>,
    branch_tips: BTreeMap<String, Option<String>>,
    first_surviving_list_seqs: BTreeMap<String, u64>,
    entry_parents: BTreeMap<String, Option<String>>,
    copied_entry_ids: std::cell::RefCell<BTreeSet<String>>,
    lane_configs: BTreeSet<String>,
    lane_states: BTreeSet<String>,
}

impl JsonlForkIndex {
    fn apply_entry(&mut self, id: &str, parent_id: Option<&str>) {
        self.entry_parents
            .insert(id.to_owned(), parent_id.map(str::to_owned));
    }

    fn apply_writes(&mut self, writes: &[CommittedWrite]) {
        for write in writes {
            match write {
                CommittedWrite::Entry { entry } => {
                    self.apply_entry(entry.id(), entry.parent_id());
                }
                CommittedWrite::ValueSet {
                    seq,
                    namespace,
                    key,
                    ..
                }
                | CommittedWrite::ValueDelete {
                    seq,
                    namespace,
                    key,
                    ..
                } => {
                    let physical = physical_key(namespace, key);
                    match write {
                        CommittedWrite::ValueSet { .. } => {
                            self.current_scalar_seqs.insert(physical, *seq);
                        }
                        _ => {
                            self.current_scalar_seqs.remove(&physical);
                        }
                    }
                    let tip_value = match write {
                        CommittedWrite::ValueSet { value, .. } => value.as_str().map(str::to_owned),
                        _ => None,
                    };
                    self.apply_lane_value(
                        namespace,
                        key,
                        matches!(write, CommittedWrite::ValueSet { .. }),
                        tip_value,
                    );
                }
                CommittedWrite::ListAppend {
                    seq,
                    namespace,
                    key,
                    ..
                }
                | CommittedWrite::ListDelete {
                    seq,
                    namespace,
                    key,
                    ..
                } => {
                    let physical = physical_key(namespace, key);
                    match write {
                        CommittedWrite::ListAppend { .. } => {
                            self.first_surviving_list_seqs
                                .entry(physical)
                                .or_insert(*seq);
                        }
                        _ => {
                            self.first_surviving_list_seqs.remove(&physical);
                        }
                    }
                }
                CommittedWrite::Usage { .. } => {}
            }
        }
    }

    fn apply_lane_value(
        &mut self,
        namespace: &str,
        key: &str,
        present: bool,
        tip_value: Option<String>,
    ) {
        match namespace {
            "pi.branch.tip" => {
                if present {
                    self.branch_tips.insert(key.to_owned(), tip_value);
                } else {
                    self.branch_tips.remove(key);
                }
            }
            "pi.lane.config" => {
                if present {
                    self.lane_configs.insert(key.to_owned());
                } else {
                    self.lane_configs.remove(key);
                }
            }
            "pi.lane.state" => {
                if present {
                    self.lane_states.insert(key.to_owned());
                } else {
                    self.lane_states.remove(key);
                }
            }
            _ => {}
        }
    }

    /// The branch tip, upstream's `getBranchTip`: `None` for an unknown
    /// branch, `Some(None)` for a branch whose tip is the root.
    #[expect(
        clippy::option_option,
        reason = "the three states mirror upstream's string | null | undefined: no such branch, an empty branch, and the branch tip id"
    )]
    fn get_branch_tip(&self, branch: &str) -> Option<Option<String>> {
        self.branch_tips.get(branch).cloned()
    }

    fn has_complete_lane(&self, branch: &str) -> bool {
        self.lane_configs.contains(branch) && self.lane_states.contains(branch)
    }

    fn get_current_scalar_seq(&self, namespace: &str, key: &str) -> Option<u64> {
        self.current_scalar_seqs
            .get(&physical_key(namespace, key))
            .copied()
    }

    fn is_surviving_list_element(&self, namespace: &str, key: &str, seq: u64) -> bool {
        self.first_surviving_list_seqs
            .get(&physical_key(namespace, key))
            .is_some_and(|first_seq| seq >= *first_seq)
    }

    #[expect(
        clippy::option_option,
        reason = "the three states mirror upstream's string | null | undefined: no such entry, a root parent, and the parent id"
    )]
    fn get_parent(&self, entry_id: &str) -> Option<Option<String>> {
        self.entry_parents.get(entry_id).cloned()
    }

    fn select_entry(&self, entry_id: &str) {
        self.copied_entry_ids
            .borrow_mut()
            .insert(entry_id.to_owned());
    }

    fn copied_entry_ids_snapshot(&self) -> BTreeSet<String> {
        self.copied_entry_ids.borrow().clone()
    }
}

/// Validates the fork source's format-4 header, upstream's
/// `readJsonlForkHeader`.
///
/// # Errors
/// A [`JsonlError`] for a non-v4 header, an identity mismatch, or an
/// unsupported storage version.
async fn read_jsonl_fork_header(
    reader: &mut dyn TextLineReader,
    source: &JsonlForkSourceMetadata,
    context: &Context,
) -> Result<JsonlStorageHeader, JsonlError> {
    use crate::harness::session::jsonl::codec::JsonlParsedSessionHeader;
    let parsed = read_jsonl_header(reader, &source.path, context).await?;
    let JsonlParsedSessionHeader::V4(header) = parsed else {
        return Err(JsonlError(format!(
            "Invalid JSONL storage {}: expected format 4 header",
            source.path
        )));
    };
    if header.id != source.id || header.cwd != source.cwd {
        return Err(JsonlError(format!(
            "Session identity does not match header: {}",
            source.id
        )));
    }
    if header.storage_version != JSONL_STORAGE_VERSION {
        return Err(JsonlError(format!(
            "Session {} uses unsupported storage version {}",
            source.id, header.storage_version
        )));
    }
    Ok(header)
}

/// Whether one transaction reaches the fork's sequence boundary, upstream's
/// `reachesForkBoundary`.
///
/// # Errors
/// A [`JsonlError`] when the transaction straddles the boundary.
fn reaches_fork_boundary(
    writes: &[CommittedWrite],
    stop_before_seq: Option<u64>,
) -> Result<bool, JsonlError> {
    let Some(stop_before_seq) = stop_before_seq else {
        return Ok(false);
    };
    if writes.is_empty() {
        return Ok(false);
    }
    let first = writes[0].seq();
    let last = writes[writes.len() - 1].seq();
    if first >= stop_before_seq {
        return Ok(true);
    }
    if last >= stop_before_seq {
        return Err(JsonlError(format!(
            "JSONL transaction crosses fork sequence boundary {stop_before_seq}"
        )));
    }
    Ok(false)
}

/// Reads complete transactions after the header, never splitting a
/// transaction at the sequence boundary, upstream's
/// `readJsonlForkTransactions` as an eager collection.
///
/// # Errors
/// A [`JsonlError`] from the reads and the transaction parses.
async fn read_jsonl_fork_transactions(
    reader: &mut dyn TextLineReader,
    path: &str,
    stop_before_seq: Option<u64>,
    context: &Context,
) -> Result<Vec<Vec<CommittedWrite>>, JsonlError> {
    let mut transactions: Vec<Vec<CommittedWrite>> = Vec::new();
    loop {
        let line = file_value(
            reader.read_line(context).await,
            &format!("Failed to read JSONL fork source {path}"),
        )?;
        let Some(line) = line.filter(|line| line.terminated) else {
            break;
        };
        let writes = parse_jsonl_transaction(&line.text)?;
        if reaches_fork_boundary(&writes, stop_before_seq)? {
            break;
        }
        transactions.push(writes);
    }
    Ok(transactions)
}

/// Validates the source lanes and returns the fork plan, upstream's
/// `selectJsonlFork`.
///
/// # Errors
/// The [`select_branch_fork`](crate::harness::session::fork_policy::select_branch_fork)
/// failures and the not-a-configured-`AgentLane` rejection.
fn select_jsonl_fork(
    index: &JsonlForkIndex,
    options: &ForkOptions,
) -> Result<ForkCurrentStatePlan, JsonlError> {
    let ForkOptions::Branch { branch, .. } = options else {
        return Ok(ForkCurrentStatePlan::Tree);
    };
    let tip = index.get_branch_tip(branch);
    let mut source = BranchForkSource {
        tip,
        get_parent: &|entry_id: &str| index.get_parent(entry_id),
        select_entry: &mut |entry_id: &str| index.select_entry(entry_id),
    };
    let plan =
        select_branch_fork(options, &mut source).map_err(|error| JsonlError(error.to_string()))?;
    if !index.has_complete_lane(branch) {
        return Err(JsonlError(format!(
            "Source branch {branch:?} is not a configured AgentLane"
        )));
    }
    Ok(plan)
}

/// Projects one source write into the destination, upstream's
/// `projectJsonlForkWrite`.
///
/// # Errors
/// A [`JsonlError`] from the current-state projection (an unknown reserved
/// namespace).
fn project_jsonl_fork_write(
    write: &CommittedWrite,
    index: &JsonlForkIndex,
    plan: &ForkCurrentStatePlan,
    is_entry_copied: &(dyn Fn(&str) -> bool + Send + Sync),
) -> Result<Option<CommittedWrite>, JsonlError> {
    match write {
        CommittedWrite::Entry { entry } => Ok(is_entry_copied(entry.id()).then(|| write.clone())),
        CommittedWrite::ValueSet {
            seq,
            namespace,
            key,
            value,
        } => {
            if index.get_current_scalar_seq(namespace, key) != Some(*seq) {
                return Ok(None);
            }
            project_state_write(
                &ForkStateWrite::ValueSet {
                    seq: *seq,
                    namespace: namespace.clone(),
                    key: key.clone(),
                    value: value.clone(),
                },
                plan,
                is_entry_copied,
            )
        }
        CommittedWrite::ListAppend {
            seq,
            namespace,
            key,
            value,
        } => {
            if !index.is_surviving_list_element(namespace, key, *seq) {
                return Ok(None);
            }
            project_state_write(
                &ForkStateWrite::ListAppend {
                    seq: *seq,
                    namespace: namespace.clone(),
                    key: key.clone(),
                    value: value.clone(),
                },
                plan,
                is_entry_copied,
            )
        }
        CommittedWrite::Usage { .. }
        | CommittedWrite::ValueDelete { .. }
        | CommittedWrite::ListDelete { .. } => Ok(None),
    }
}

/// Projects one current-state write and converts the result back to its
/// committed shape, upstream's inline double-match.
fn project_state_write(
    write: &ForkStateWrite,
    plan: &ForkCurrentStatePlan,
    is_entry_copied: &(dyn Fn(&str) -> bool + Send + Sync),
) -> Result<Option<CommittedWrite>, JsonlError> {
    let projected = project_fork_current_state_write(write, plan, is_entry_copied)
        .map_err(|error| JsonlError(error.to_string()))?;
    Ok(projected.map(|projected| match projected {
        ForkStateWrite::ValueSet {
            seq,
            namespace,
            key,
            value,
        } => CommittedWrite::ValueSet {
            seq,
            namespace,
            key,
            value,
        },
        ForkStateWrite::ListAppend {
            seq,
            namespace,
            key,
            value,
        } => CommittedWrite::ListAppend {
            seq,
            namespace,
            key,
            value,
        },
    }))
}

/// Prepared fork input: format-4 file metadata or an already-normalized
/// legacy source, upstream's `JsonlForkInput`.
#[derive(Clone, Debug)]
pub enum JsonlForkInput {
    /// An open source storage, stopped at the captured sequence boundary.
    Open {
        /// The source's identity.
        metadata: JsonlForkSourceMetadata,
        /// The captured commit-queue sequence boundary, upstream's
        /// `nextSeq`.
        next_seq: u64,
    },
    /// A closed source scanned to its end.
    Closed {
        /// The source's identity.
        metadata: JsonlForkSourceMetadata,
    },
    /// A legacy v3 source captured through its normalized read.
    LegacyV3(Arc<LegacyV3Source>),
}

/// The file metadata the file-backed fork inputs carry, upstream's
/// `input.metadata` access.
fn fork_input_metadata(input: &JsonlForkInput) -> &JsonlForkSourceMetadata {
    match input {
        JsonlForkInput::Open { metadata, .. } | JsonlForkInput::Closed { metadata } => metadata,
        JsonlForkInput::LegacyV3(_) => unreachable_legacy_index(),
    }
}

/// Builds the index one fork selects against, upstream's `indexForkInput`.
///
/// # Errors
/// A [`JsonlError`] from the source reads and the fork-header validation.
async fn index_fork_input(
    input: &JsonlForkInput,
    file_system: &Arc<dyn FileSystem>,
    context: &Context,
) -> Result<(JsonlForkIndex, u64), JsonlError> {
    let mut index = JsonlForkIndex::default();
    if let JsonlForkInput::LegacyV3(normalized) = input {
        for entry in normalized.entry_structures()? {
            index.apply_entry(&entry.id, entry.parent_id.as_deref());
        }
        index.apply_writes(&normalized.values);
        return Ok((index, normalized.next_seq));
    }
    let metadata = fork_input_metadata(input);
    let mut reader = file_value(
        file_system
            .open_text_line_reader(&metadata.path, context)
            .await,
        &format!("Failed to open JSONL fork source {}", metadata.path),
    )?;
    let result = index_fork_through(&mut *reader, input, metadata, context).await;
    reader.close(context).await;
    result
}

async fn index_fork_through(
    reader: &mut dyn TextLineReader,
    input: &JsonlForkInput,
    metadata: &JsonlForkSourceMetadata,
    context: &Context,
) -> Result<(JsonlForkIndex, u64), JsonlError> {
    let mut index = JsonlForkIndex::default();
    let header = read_jsonl_fork_header(reader, metadata, context).await?;
    let stop_before_seq = match input {
        JsonlForkInput::Open { next_seq, .. } => Some(*next_seq),
        _ => None,
    };
    let mut highest_complete_seq = 0;
    for writes in
        read_jsonl_fork_transactions(reader, &metadata.path, stop_before_seq, context).await?
    {
        if let Some(last) = writes.last() {
            highest_complete_seq = last.seq();
        }
        index.apply_writes(&writes);
    }
    let next_seq = match input {
        JsonlForkInput::Open { next_seq, .. } => *next_seq,
        _ => header.next_seq.unwrap_or(1).max(highest_complete_seq + 1),
    };
    Ok((index, next_seq))
}

#[expect(
    clippy::panic,
    reason = "index_fork_input dispatches the legacy case before this helper; the fallback is unreachable by construction"
)]
fn unreachable_legacy_index() -> ! {
    panic!("index_fork_input reached the file path for a legacy input")
}

/// Yields the source writes; the caller owns the final projection and
/// filtering, upstream's `streamForkWrites` as an eager collection.
///
/// # Errors
/// A [`JsonlError`] from the source reads and the v3 normalization.
async fn stream_fork_writes(
    input: &JsonlForkInput,
    file_system: &Arc<dyn FileSystem>,
    stop_before_seq: u64,
    is_entry_copied: &(dyn Fn(&str) -> bool + Send + Sync),
    context: &Context,
) -> Result<Vec<CommittedWrite>, JsonlError> {
    if let JsonlForkInput::LegacyV3(normalized) = input {
        // V3 filters early to avoid rereading messages and reconstructing
        // unselected compaction tails; V4 already stores complete payloads.
        // Both formats pass through the projection for final filtering.
        return normalized
            .writes(context, Some(is_entry_copied))
            .await
            .map(|writes| {
                writes
                    .into_iter()
                    .filter(|write| {
                        matches!(
                            write,
                            CommittedWrite::Entry { .. } | CommittedWrite::ValueSet { .. }
                        )
                    })
                    .collect()
            });
    }
    let metadata = fork_input_metadata(input);
    let mut reader = file_value(
        file_system
            .open_text_line_reader(&metadata.path, context)
            .await,
        &format!("Failed to open JSONL fork source {}", metadata.path),
    )?;
    let result = stream_fork_through(&mut *reader, metadata, stop_before_seq, context).await;
    reader.close(context).await;
    result
}

async fn stream_fork_through(
    reader: &mut dyn TextLineReader,
    metadata: &JsonlForkSourceMetadata,
    stop_before_seq: u64,
    context: &Context,
) -> Result<Vec<CommittedWrite>, JsonlError> {
    read_jsonl_fork_header(reader, metadata, context).await?;
    let transactions =
        read_jsonl_fork_transactions(reader, &metadata.path, Some(stop_before_seq), context)
            .await?;
    Ok(transactions.into_iter().flatten().collect())
}

/// The fork options [`run_jsonl_fork`] takes, upstream's inline options
/// object.
pub struct JsonlForkRun {
    /// The prepared fork input.
    pub input: JsonlForkInput,
    /// The filesystem capability both passes read through.
    pub file_system: Arc<dyn FileSystem>,

    /// The destination file's addressed path.
    pub destination_path: String,
    /// The destination header without its sequence mark, upstream's
    /// `destinationHeader`.
    pub destination_header: JsonlStorageHeader,
    /// What the fork copies.
    pub fork: ForkOptions,
}

impl std::fmt::Debug for JsonlForkRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonlForkRun")
            .field("input", &self.input)
            .field("destination_path", &self.destination_path)
            .field("destination_header", &self.destination_header)
            .field("fork", &self.fork)
            .finish_non_exhaustive()
    }
}

/// Index the source, validate the requested fork, and stream selected
/// entries and current state into an atomically published format-4
/// destination without modifying the source, upstream's `runJsonlFork`.
///
/// Copied sequences and the source's nextSeq are preserved while usage and
/// open-operation state are excluded. Does not open the destination
/// Session.
///
/// # Errors
/// A [`JsonlError`] from the indexing, the selection, the projection, and
/// the publication.
pub async fn run_jsonl_fork(options: JsonlForkRun, context: &Context) -> Result<(), JsonlError> {
    let (index, next_seq) = index_fork_input(&options.input, &options.file_system, context).await?;
    let fork = match (&options.input, &options.fork) {
        (
            JsonlForkInput::LegacyV3(normalized),
            ForkOptions::Branch {
                entry_id: Some(entry_id),
                ..
            },
        ) => {
            let mut translated = options.fork.clone();
            if let ForkOptions::Branch { entry_id: slot, .. } = &mut translated {
                *slot = Some(normalized.translate_fork_entry_id(entry_id)?);
            }
            translated
        }
        _ => options.fork.clone(),
    };
    let plan = select_jsonl_fork(&index, &fork).map_err(|error| JsonlError(error.to_string()))?;
    let copied = index.copied_entry_ids_snapshot();
    let plan_for_closure = plan.clone();
    let is_entry_copied = move |entry_id: &str| match &plan_for_closure {
        ForkCurrentStatePlan::Tree => true,
        ForkCurrentStatePlan::Branch { .. } => copied.contains(entry_id),
    };
    let mut destination_header = options.destination_header.clone();
    destination_header.next_seq = Some(next_seq);
    let source_writes = stream_fork_writes(
        &options.input,
        &options.file_system,
        next_seq,
        &is_entry_copied,
        context,
    )
    .await?;
    let mut projected_writes: Vec<CommittedWrite> = Vec::new();
    for write in &source_writes {
        if let Some(projected) = project_jsonl_fork_write(write, &index, &plan, &is_entry_copied)? {
            projected_writes.push(projected);
        }
    }
    publish_jsonl(
        &options.file_system,
        &options.destination_path,
        &destination_header,
        context,
        move |append| async move {
            // Each projected write appends as its own single-write
            // transaction, upstream's `await append([projected])`.
            for write in &projected_writes {
                append(std::slice::from_ref(write)).await?;
            }
            Ok(())
        },
    )
    .await
}

/// The JSON value the v3 tip projection reads; the index's tip tracking is
/// presence-only, upstream's `string | null` value union.
pub type TipValue = JsonValue;

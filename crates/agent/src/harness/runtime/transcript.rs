//! The transcript helpers the lane and the drive procedures share, ported
//! from upstream `src/harness/runtime/transcript.ts`.
//!
//! `readBoundedContext` consults the context builder the runtime-foundations
//! child carries in [`crate::harness::session::context`]; the session-child
//! reservation stands.

use std::sync::Arc;

use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::LaneQueuedItem;
use crate::harness::context::Context;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::from_arc;
use crate::harness::runtime::types::lane_error;
use crate::harness::runtime::types::LaneError;
use crate::harness::session::context::SessionContextBuildOptions;
use crate::harness::session::context::build_session_context;
use crate::harness::session::types::BranchScanOrder;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::Entry;
use crate::harness::session::types::EntryType;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionInvariantError;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::StorageBranchScan;
use crate::harness::session::values::pending_entry;
use crate::harness::session::values::StoredValue;
use crate::harness::runtime::lane::Lane;
use crate::types::AgentMessage;

/// Links a list of new entries into one parent chain, upstream's
/// `chainEntries`: each entry's parent is the previous item's id, or the
/// given parent for the first.
#[must_use]
pub fn chain_entries(parent_id: Option<String>, items: Vec<NewEntry>) -> Vec<NewEntry> {
    let mut parent_id = parent_id;
    items
        .into_iter()
        .map(|mut entry| {
            set_entry_parent(&mut entry, parent_id.clone());
            parent_id = Some(entry.id().to_owned());
            entry
        })
        .collect()
}

fn set_entry_parent(entry: &mut NewEntry, parent_id: Option<String>) {
    match entry {
        NewEntry::Message { parent_id: at, .. }
        | NewEntry::Compaction { parent_id: at, .. }
        | NewEntry::BranchSummary { parent_id: at, .. }
        | NewEntry::Custom { parent_id: at, .. } => *at = parent_id,
    }
}

/// The lifecycle events one committed entry publishes, upstream's
/// `entryLifecycleEvents`: a message entry announces start, end, and the
/// commit; every other entry announces the commit.
#[must_use]
pub fn entry_lifecycle_events(entry: Entry, lane: &str, run_id: Option<&str>) -> Vec<HarnessEvent> {
    let message_events = match &entry {
        Entry::Message { body, id, .. } => {
            let run_id = run_id.map(str::to_owned);
            let message_start = HarnessEvent::lane_scoped(
                lane,
                false,
                HarnessEventPayload::MessageStart {
                    run_id: run_id.clone(),
                    message: body.message.clone(),
                },
            )
            .unwrap_or_else(|error| unreachable!("message_start is lane-scoped: {error}"));
            let message_end = HarnessEvent::lane_scoped(
                lane,
                false,
                HarnessEventPayload::MessageEnd {
                    run_id,
                    message: body.message.clone(),
                    entry_id: Some(id.clone()),
                },
            )
            .unwrap_or_else(|error| unreachable!("message_end is lane-scoped: {error}"));
            vec![message_start, message_end]
        }
        Entry::Compaction { .. } | Entry::BranchSummary { .. } | Entry::Custom { .. } => Vec::new(),
    };
    let entry_added = HarnessEvent::lane_scoped(lane, false, HarnessEventPayload::EntryAdded { entry })
        .unwrap_or_else(|error| unreachable!("entry_added is lane-scoped: {error}"));
    let mut events = message_events;
    events.push(entry_added);
    events
}

/// The lifecycle events a commit's entries publish, upstream's
/// `committedEntryEvents`: each entry materializes with its storage-assigned
/// sequence and the commit's timestamp.
#[must_use]
pub fn committed_entry_events(
    entries: &[NewEntry],
    commit: &CommitResult,
    lane: &str,
    run_id: Option<&str>,
    first_write_index: usize,
) -> Vec<HarnessEvent> {
    let mut events = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        let seq = match commit.seqs.get(first_write_index + index) {
            Some(seq) => *seq,
            None => unreachable!("commit carries one sequence per write"),
        };
        let materialized = entry.clone().materialize(seq, commit.timestamp);
        events.extend(entry_lifecycle_events(materialized, lane, run_id));
    }
    events
}

/// Reads the entries one run's operation covers, upstream's
/// `readBoundedEntries`: the branch path from the tip to the last compaction
/// entry, oldest first.
pub async fn read_bounded_entries(
    lane: &Lane,
    drive: &Drive,
) -> Result<ContinueOperationResult<Vec<Entry>>, LaneError> {
    lane.continue_operation(
        |state: crate::harness::runtime::types::LaneState, session: Arc<dyn crate::harness::session::types::Session>, context| {
            Box::pin(async move {
                let Some(tip_id) = state.tip_id else {
                    return Err(Arc::new(SessionInvariantError(
                        "Run operation has no Branch tip".to_owned(),
                    )));
                };
                let entries = session
                    .scan_branch(
                        &StorageBranchScan {
                            start: tip_id,
                            stop_at_type: Some(EntryType::Compaction),
                            order: Some(BranchScanOrder::NewestFirst),
                            ..Default::default()
                        },
                        &context,
                    )
                    .await
                    .map_err(lane_error)?;
                let mut entries = entries;
                entries.reverse();
                Ok(OperationCommand::Return { result: entries })
            })
                as BoxedFuture<'static, Result<OperationCommand<Vec<Entry>>, LaneError>>
        },
        &drive.context,
    )
    .await
}

/// Reads the model context one run's operation covers, upstream's
/// `readBoundedContext`.
pub async fn read_bounded_context(
    lane: &Lane,
    drive: &Drive,
) -> Result<ContinueOperationResult<Vec<AgentMessage>>, LaneError> {
    let entries = read_bounded_entries(lane, drive).await?;
    match entries {
        ContinueOperationResult::CancelRequested => Ok(ContinueOperationResult::CancelRequested),
        ContinueOperationResult::Result { value } => {
            let options = SessionContextBuildOptions {
                entry_projectors: lane.read_config().entry_projectors,
            };
            let value = build_session_context(&value, Some(&options), &drive.context).await;
            Ok(ContinueOperationResult::Result { value })
        }
    }
}

/// Reads the queued items one lane's inbox holds, upstream's
/// `readLaneQueues`: each pending payload becomes its queue view.
pub async fn read_lane_queues(
    reader: &dyn SessionReader,
    inbox: &[InboxItem],
    context: &Context,
) -> Result<Vec<LaneQueuedItem>, SessionError> {
    let mut queues = Vec::with_capacity(inbox.len());
    for item in inbox {
        let stored = reader
            .get_value(&pending_entry(&item.entry_id).address, context)
            .await?;
        let Some(stored) = stored else {
            return Err(SessionInvariantError(format!(
                "Pending {} entry {} is missing its payload",
                inbox_item_kind(&item.kind),
                item.entry_id
            ))
            .into());
        };
        let pending: PendingEntry = serde_json::from_value(stored.value)
            .map_err(|error| SessionInvariantError(format!("Pending entry payload is malformed: {error}")))?;
        match pending {
            PendingEntry::Message { payload } => {
                queues.push(LaneQueuedItem::Message {
                    entry_id: item.entry_id.clone(),
                    kind: item.kind,
                    message: Box::new(*payload),
                });
            }
            PendingEntry::Custom { custom_type, payload } => {
                if item.kind != InboxItemKind::Write {
                    return Err(SessionInvariantError(format!(
                        "Pending {} entry {} is not a message",
                        inbox_item_kind(&item.kind),
                        item.entry_id
                    ))
                    .into());
                }
                queues.push(LaneQueuedItem::Custom {
                    entry_id: item.entry_id.clone(),
                    kind: item.kind,
                    custom_type,
                    data: payload,
                });
            }
        }
    }
    Ok(queues)
}

fn inbox_item_kind(kind: &InboxItemKind) -> &'static str {
    match kind {
        InboxItemKind::Steer => "steer",
        InboxItemKind::FollowUp => "followUp",
        InboxItemKind::NextRun => "nextRun",
        InboxItemKind::Write => "write",
    }
}

/// Reads the pending message payloads named ids carry, upstream's
/// `readPendingMessages`.
pub async fn read_pending_messages(
    reader: &dyn SessionReader,
    ids: &[String],
    description: &str,
    context: &Context,
) -> Result<Vec<(String, AgentMessage)>, SessionError> {
    let missing = |entry_id: &str| {
        SessionInvariantError(format!(
            "{description} {entry_id} is missing its message payload"
        ))
    };
    let mut messages = Vec::with_capacity(ids.len());
    for entry_id in ids {
        let value = reader.get_value(&pending_entry(entry_id).address, context).await?;
        let stored: StoredValue = value.ok_or_else(|| missing(entry_id))?;
        let pending: PendingEntry = serde_json::from_value(stored.value).map_err(|_| missing(entry_id))?;
        let PendingEntry::Message { payload } = pending else {
            return Err(missing(entry_id).into());
        };
        messages.push((entry_id.clone(), *payload));
    }
    Ok(messages)
}
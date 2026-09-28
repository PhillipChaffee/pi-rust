//! The boundary mediation the checkpoint and structural procedures share,
//! ported from upstream `src/harness/runtime/drive/boundary.ts`.
//!
//! A boundary replans the lane's inbox input
//! (`plan_boundary_inbox`), builds the assistant-ready leaf renewed work
//! commits (`assistant_ready_at_boundary`), derives the placement's
//! events (`boundary_placement_events`), and finishes a run through the
//! `before_run_end` hook (`finish_run_boundary`).
//!
//! Both placement write orders fix the commit's sequence numbers the event
//! builders read: entry inserts, then the selected pending-entry deletes,
//! then the branch-tip set, and in the finish arm the follow-up entry rides
//! after the placement writes.

use std::sync::Arc;

use pi_ai::types::Message;
use pi_ai::types::UserContent;
use pi_ai::types::UserMessage;
use pi_ai::utils::retry::DEFAULT_MAX_AGENT_RETRY_DELAY_MS;

use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HookEvent;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::agent_harness::HookResult;
use crate::harness::agent_harness::LaneQueuedItem;
use crate::harness::agent_harness::RunEndStatus;
use crate::harness::agent_harness::lane_scoped_event;
use crate::harness::context::Context;
use crate::harness::runtime::clock::now_ms;
use crate::harness::runtime::drive::hook_error_to_lane_error;
use crate::harness::runtime::drive::terminal::operation_cleanup_writes;
use crate::harness::runtime::drive::terminal::operation_result_record;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::transcript::committed_entry_events;
use crate::harness::runtime::transcript::entry_lifecycle_events;
use crate::harness::runtime::transcript::read_bounded_context;
use crate::harness::runtime::transcript::read_lane_queues;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::LanePatch;
use crate::harness::runtime::types::LaneState;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::runtime::types::ProcedureResult;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::commit::insert_entry;
use crate::harness::session::types::AssistantReadyOperation;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::CustomEntryBody;
use crate::harness::session::types::Entry;
use crate::harness::session::types::GenerationContext;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::InboxItemKind;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::NormalizedRetryPolicy;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::PendingEntry;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::TerminalStatus;
use crate::harness::session::types::operation_scope_of;
use crate::harness::session::values::Write;
use crate::harness::session::values::branch_tip;
use crate::harness::session::values::delete_value_write;
use crate::harness::session::values::pending_entry;
use crate::harness::session::values::set_value_write;
use crate::types::AgentMessage;
use crate::types::QueueMode;

/// The mediation result telling the caller the run may finish, upstream's
/// `BoundaryFinishPending` (`{ kind: "finish_pending"; entryIds }`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BoundaryFinishPending {
    /// The placed entries the finished run leaves committed.
    pub entry_ids: Vec<String>,
}

/// One boundary's selected and materialized lane-owned input, upstream's
/// `BoundaryPlacement`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BoundaryPlacement {
    /// The chained entries the placement inserts.
    pub entries: Vec<NewEntry>,
    /// The transaction's writes: entry inserts, the selected pending-entry
    /// deletes, and the branch-tip set when any entry placed.
    pub writes: Vec<Write>,
    /// The tip after the placement; with no entries placed, the incoming
    /// tip unchanged.
    pub tip_id: Option<String>,
    /// The remaining inbox after removing the selected items.
    pub inbox: Vec<InboxItem>,
    /// The last projecting entry's id, when one placed — the generation
    /// trigger.
    pub trigger_entry_id: Option<String>,
    /// The remaining inbox's queue read, when any item was selected.
    pub queues: Option<Vec<LaneQueuedItem>>,
}

/// The hook's follow-up prompt with its reserved entry id, upstream's
/// `{ id, message: { role: "user", content, timestamp } }` literal.
#[derive(Clone)]
struct FollowUpEntry {
    id: String,
    message: AgentMessage,
}

/// The inbox position of one selected item, upstream's
/// `state.inbox.indexOf(a) - state.inbox.indexOf(b)` sort comparator — the
/// selected items are the inbox's own elements, so identity pins the index.
fn inbox_index(inbox: &[InboxItem], item: &InboxItem) -> usize {
    inbox
        .iter()
        .position(|candidate| std::ptr::eq(candidate, item))
        .unwrap_or(usize::MAX)
}

/// The queue-kind name the pending-payload invariants carry, upstream's
/// `${item.kind}` interpolation.
const fn inbox_kind_name(kind: InboxItemKind) -> &'static str {
    match kind {
        InboxItemKind::Steer => "steer",
        InboxItemKind::FollowUp => "followUp",
        InboxItemKind::NextRun => "nextRun",
        InboxItemKind::Write => "write",
    }
}

/// Reads the selected items' pending payloads, upstream's `load`: each item
/// pairs with its stored payload, and a missing payload or a non-message
/// payload under a steer/follow-up item is an invariant.
///
/// # Errors
/// The value read's storage error; the invariants below; a pending payload
/// the session's pending-entry shape cannot decode.
async fn load_pending<'a>(
    reader: &dyn SessionReader,
    selected: &[&'a InboxItem],
    context: &Context,
) -> Result<Vec<(&'a InboxItem, PendingEntry)>, LaneError> {
    let mut pending = Vec::with_capacity(selected.len());
    for item in selected {
        let stored = reader
            .get_value(&pending_entry(&item.entry_id).address, context)
            .await
            .map_err(lane_error)?;
        let Some(stored) = stored else {
            return Err(lane_error(SessionError::Invariant(format!(
                "Pending {} entry {} is missing its payload",
                inbox_kind_name(item.kind),
                item.entry_id
            ))));
        };
        // The message check reads the raw discriminator first, upstream's
        // `stored.value.type !== "message"` probe.
        if item.kind != InboxItemKind::Write
            && stored.value.get("type").is_none_or(|tag| tag != "message")
        {
            return Err(lane_error(SessionError::Invariant(format!(
                "Queued {} entry {} is not a message",
                inbox_kind_name(item.kind),
                item.entry_id
            ))));
        }
        let payload: PendingEntry = serde_json::from_value(stored.value).map_err(|error| {
            lane_error(SessionError::Invariant(format!(
                "Pending entry payload is malformed: {error}"
            )))
        })?;
        pending.push((*item, payload));
    }
    Ok(pending)
}

/// Derives the normalized retry policy from the lane's configured policy,
/// upstream's `normalizedRetryPolicy`: the attempt budget is
/// `retry.enabled ? retry.maxRetries + 1 : 1`, the base delay copies
/// through, and the agent delay cap falls back to
/// `DEFAULT_MAX_AGENT_RETRY_DELAY_MS` when the policy leaves it unset.
#[must_use]
pub(crate) fn normalized_retry_policy(lane: &Lane) -> NormalizedRetryPolicy {
    let retry = lane.read_config().retry_policy;
    NormalizedRetryPolicy {
        max_attempts: if retry.enabled {
            retry.max_retries.saturating_add(1)
        } else {
            1
        },
        base_delay_ms: retry.base_delay_ms,
        max_agent_delay_ms: retry
            .max_agent_delay_ms
            .unwrap_or(DEFAULT_MAX_AGENT_RETRY_DELAY_MS),
    }
}

/// Builds the assistant-ready leaf one renewed boundary commits, upstream's
/// `assistantReadyAtBoundary`: a fresh step id from the session's id
/// generator, the trigger entry, the lane state's configuration snapshot
/// and stream options, the normalized retry policy, and attempt 1.
#[must_use]
pub(crate) fn assistant_ready_at_boundary(
    lane: &Lane,
    state: &LaneState,
    scope: OperationScope,
    trigger_entry_id: String,
    overflow_recovery_used: bool,
) -> AssistantReadyOperation {
    let config = lane.read_config();
    AssistantReadyOperation {
        scope,
        generation_context: GenerationContext {
            step_id: lane.session().id_generator().next(None),
            trigger_entry_id,
            configuration: state.configuration.clone(),
            stream_options: config.stream_options,
            retry_policy: normalized_retry_policy(lane),
            overflow_recovery_used,
        },
        next_attempt: 1,
    }
}

/// Selects and materializes one boundary's lane-owned input without
/// committing it, upstream's `planBoundaryInbox`.
///
/// The selection keeps the inbox's order: steer items per the steering
/// mode, write items, and — only when `follow_up_when_no_trigger` holds
/// and no selected payload projects a trigger — follow-up items re-merged
/// into inbox order with a full payload reload. The returned `queues`
/// describe the remaining inbox, not the selection.
///
/// # Errors
/// The invariants `` `Pending {kind} entry {id} is missing its payload` ``
/// and `` `Queued {kind} entry {id} is not a message` `` when a selected
/// payload is missing or mistyped, the underlying reads' storage errors,
/// and the branch-tip write's payload serialization.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single planBoundaryInbox method"
)]
pub(crate) async fn plan_boundary_inbox(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    state: &LaneState,
    scope: OperationScope,
    reader: &dyn SessionReader,
    tip_id: Option<String>,
    follow_up_when_no_trigger: bool,
) -> Result<BoundaryPlacement, LaneError> {
    let steer: Vec<&InboxItem> = state
        .inbox
        .iter()
        .filter(|item| item.kind == InboxItemKind::Steer)
        .collect();
    let selected_steer: Vec<&InboxItem> = if scope.settings.steering_mode == QueueMode::All {
        steer
    } else {
        steer.into_iter().take(1).collect()
    };
    // Inbox order preserved; next-run items never select here.
    let mut selected: Vec<&InboxItem> = state
        .inbox
        .iter()
        .filter(|item| {
            item.kind == InboxItemKind::Write
                || selected_steer
                    .iter()
                    .any(|candidate| candidate.entry_id == item.entry_id)
        })
        .collect();
    let projects = |value: &PendingEntry| match value {
        PendingEntry::Message { .. } => true,
        PendingEntry::Custom { custom_type, .. } => lane
            .read_config()
            .entry_projectors
            .contains_key(custom_type),
    };
    let mut pending = load_pending(reader, &selected, &drive.context).await?;
    if follow_up_when_no_trigger && !pending.iter().any(|(_, value)| projects(value)) {
        let follow_up: Vec<&InboxItem> = state
            .inbox
            .iter()
            .filter(|item| item.kind == InboxItemKind::FollowUp)
            .collect();
        let selected_follow_up: Vec<&InboxItem> = if scope.settings.follow_up_mode == QueueMode::All
        {
            follow_up
        } else {
            follow_up.into_iter().take(1).collect()
        };
        // Re-merge the fallback into inbox order, upstream's
        // `[...selected, ...selectedFollowUp].sort((a, b) => indexOf(a) - indexOf(b))`.
        let mut merged = selected.clone();
        merged.extend(selected_follow_up);
        merged.sort_by_key(|item| inbox_index(&state.inbox, item));
        selected = merged;
        // A full reload: a payload may have changed between the two loads.
        pending = load_pending(reader, &selected, &drive.context).await?;
    }

    let mut parent_id = tip_id;
    let mut trigger_entry_id: Option<String> = None;
    let mut entries: Vec<NewEntry> = Vec::with_capacity(pending.len());
    for (item, value) in &pending {
        let entry = match value {
            PendingEntry::Message { payload } => NewEntry::Message {
                id: item.entry_id.clone(),
                parent_id: parent_id.clone(),
                body: Box::new(MessageEntry {
                    message: (**payload).clone(),
                    terminate: None,
                }),
            },
            PendingEntry::Custom {
                custom_type,
                payload,
            } => NewEntry::Custom {
                id: item.entry_id.clone(),
                parent_id: parent_id.clone(),
                body: CustomEntryBody {
                    custom_type: custom_type.clone(),
                    data: payload.clone(),
                },
            },
        };
        parent_id = Some(item.entry_id.clone());
        if projects(value) {
            trigger_entry_id = Some(item.entry_id.clone());
        }
        entries.push(entry);
    }
    let inbox: Vec<InboxItem> = state
        .inbox
        .iter()
        .filter(|item| !selected.iter().any(|s| s.entry_id == item.entry_id))
        .cloned()
        .collect();
    // The queues describe the remaining inbox, not the selected items.
    let queues = if selected.is_empty() {
        None
    } else {
        Some(
            read_lane_queues(reader, &inbox, &drive.context)
                .await
                .map_err(lane_error)?,
        )
    };
    let mut writes: Vec<Write> = entries
        .iter()
        .map(|entry| Write::Entry(Box::new(insert_entry(entry.clone()))))
        .collect();
    for item in &selected {
        writes.push(delete_value_write(&pending_entry(&item.entry_id)));
    }
    if !entries.is_empty() {
        writes.push(
            set_value_write(&branch_tip(lane.name()), parent_id.clone()).map_err(lane_error)?,
        );
    }
    Ok(BoundaryPlacement {
        entries,
        writes,
        tip_id: parent_id,
        inbox,
        trigger_entry_id,
        queues,
    })
}

/// Builds the events one committed placement publishes, upstream's
/// `boundaryPlacementEvents`: the entries' lifecycle events read from
/// `first_write_index`, plus the remaining inbox's `queue_update` event
/// when the placement read queues.
#[must_use]
pub(crate) fn boundary_placement_events(
    placement: &BoundaryPlacement,
    commit: &CommitResult,
    first_write_index: usize,
    lane: &str,
    run_id: &str,
) -> Vec<HarnessEvent> {
    let mut events = committed_entry_events(
        &placement.entries,
        commit,
        lane,
        Some(run_id),
        first_write_index,
    );
    if let Some(queues) = &placement.queues {
        events.push(lane_scoped_event(
            lane,
            false,
            "queue_update",
            HarnessEventPayload::QueueUpdate {
                queues: queues.clone(),
            },
        ));
    }
    events
}

/// Replans after `before_run_end` and commits either renewed work or the
/// terminal run result, upstream's `finishRunBoundary`.
///
/// The hook's follow-up joins the commit only while the re-planned entry
/// ids equal `planned_entry_ids` positionally — input that arrived during
/// the hook discards the follow-up.
///
/// `capability` is the finish-boundary leaf the caller matched (the
/// checkpoint or a summary leaf); it rides `state.operation` inside the
/// landed planners, so the parameter only pins the caller's narrowed
/// dispatch. `continuation` is the run's `may_finish` continuation, whose
/// `include_final_assistant` gates the final-assistant invariant.
///
/// # Errors
/// The invariants `` `Completed run has no tip` ``, `` `Completed run is
/// missing its final assistant` ``, and the terminal record's failed/error
/// invariant, plus the placement's errors, the hook run's rejection, and
/// the commit's storage and delivery errors.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single finishRunBoundary method"
)]
pub(crate) async fn finish_run_boundary(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    _capability: &OperationState,
    continuation: Continuation,
    planned_entry_ids: &[String],
    pending_events: Vec<HarnessEvent>,
) -> Result<ProcedureResult, LaneError> {
    let bounded = read_bounded_context(lane, drive).await?;
    let ContinueOperationResult::Result { value: messages } = bounded else {
        return Ok(ProcedureResult::Continue);
    };
    let hook = lane
        .hooks()
        .run_with_gate(
            HookName::BeforeRunEnd,
            HookInvocation {
                lane: lane.name().to_owned(),
                run_id: drive.operation_id.clone(),
                event: HookEvent::BeforeRunEnd {
                    run_id: drive.operation_id.clone(),
                    messages,
                },
            },
            &drive.gate,
            &drive.context,
        )
        .await
        .map_err(|error| hook_error_to_lane_error(error, drive))?;
    let HookResult::BeforeRunEnd(hook_follow_up) = hook else {
        unreachable!("before_run_end returns its own result variant")
    };
    // The id reserves before the clock reads, upstream's object-literal
    // evaluation order.
    let follow_up = hook_follow_up.map(|result| FollowUpEntry {
        id: lane.session().id_generator().next(None),
        message: AgentMessage::Standard(Message::User(UserMessage {
            content: UserContent::Text(result.follow_up),
            timestamp: now_ms(),
        })),
    });

    let planner_lane = Arc::clone(lane);
    let planner_drive = Arc::clone(drive);
    // The planner closure is 'static; the ids cross as an owned vector.
    let planned_entry_ids = planned_entry_ids.to_vec();
    let result = lane
        .continue_operation::<ProcedureResult, _>(
            move |state, session, _context| {
                let lane = Arc::clone(&planner_lane);
                let drive = Arc::clone(&planner_drive);
                let planned_entry_ids = planned_entry_ids.clone();
                let pending_events = pending_events.clone();
                let follow_up = follow_up.clone();
                Box::pin(async move {
                    let Some(current) = state.operation.as_ref() else {
                        unreachable!("continueOperation planner runs under an active operation")
                    };
                    let scope = operation_scope_of(&current.state);
                    let placement = plan_boundary_inbox(
                        &lane,
                        &drive,
                        &state,
                        scope.clone(),
                        session.as_ref(),
                        state.tip_id.clone(),
                        true,
                    )
                    .await?;
                    if let Some(trigger_entry_id) = placement.trigger_entry_id.clone() {
                        return Ok(OperationCommand::Commit {
                            writes: placement.writes.clone(),
                            operation_state: OperationState::AssistantReady(
                                assistant_ready_at_boundary(
                                    &lane,
                                    &state,
                                    scope,
                                    trigger_entry_id,
                                    false,
                                ),
                            ),
                            lane: Some(LanePatch {
                                tip_id: Some(placement.tip_id.clone()),
                                inbox: Some(placement.inbox.clone()),
                                ..LanePatch::default()
                            }),
                            materialize: Arc::new(|_: &CommitResult| ProcedureResult::Continue),
                            events: Some(Arc::new({
                                let pending_events = pending_events.clone();
                                let placement = placement.clone();
                                let lane_name = lane.name().to_owned();
                                let run_id = drive.operation_id.clone();
                                move |commit: &CommitResult| {
                                    let mut events = pending_events.clone();
                                    events.extend(boundary_placement_events(
                                        &placement, commit, 0, &lane_name, &run_id,
                                    ));
                                    events
                                }
                            })),
                        });
                    }
                    // Input that arrived during the hook changed the plan;
                    // a stale follow-up drops, upstream's `hookPlanIsCurrent`.
                    let hook_plan_is_current = placement.entries.len() == planned_entry_ids.len()
                        && placement
                            .entries
                            .iter()
                            .zip(planned_entry_ids.iter())
                            .all(|(entry, planned)| entry.id() == planned.as_str());
                    if hook_plan_is_current && let Some(follow_up) = &follow_up {
                        let entry = NewEntry::Message {
                            id: follow_up.id.clone(),
                            parent_id: placement.tip_id.clone(),
                            body: Box::new(MessageEntry {
                                message: follow_up.message.clone(),
                                terminate: None,
                            }),
                        };
                        let entry_write_index = placement.writes.len();
                        let mut writes = placement.writes.clone();
                        writes.push(Write::Entry(Box::new(insert_entry(entry))));
                        writes.push(
                            set_value_write(&branch_tip(lane.name()), Some(follow_up.id.clone()))
                                .map_err(lane_error)?,
                        );
                        return Ok(OperationCommand::Commit {
                            writes,
                            operation_state: OperationState::AssistantReady(
                                assistant_ready_at_boundary(
                                    &lane,
                                    &state,
                                    scope,
                                    follow_up.id.clone(),
                                    false,
                                ),
                            ),
                            lane: Some(LanePatch {
                                tip_id: Some(Some(follow_up.id.clone())),
                                inbox: Some(placement.inbox.clone()),
                                ..LanePatch::default()
                            }),
                            materialize: Arc::new(|_: &CommitResult| ProcedureResult::Continue),
                            events: Some(Arc::new({
                                let pending_events = pending_events.clone();
                                let placement = placement.clone();
                                let lane_name = lane.name().to_owned();
                                let run_id = drive.operation_id.clone();
                                let follow_up = follow_up.clone();
                                move |commit: &CommitResult| {
                                    let mut events = pending_events.clone();
                                    events.extend(boundary_placement_events(
                                        &placement, commit, 0, &lane_name, &run_id,
                                    ));
                                    let seq = commit.seqs.get(entry_write_index).map_or_else(
                                        || unreachable!("commit carries one sequence per write"),
                                        |seq| *seq,
                                    );
                                    let committed = Entry::Message {
                                        id: follow_up.id.clone(),
                                        parent_id: placement.tip_id.clone(),
                                        seq,
                                        timestamp: commit.timestamp,
                                        body: Box::new(MessageEntry {
                                            message: follow_up.message.clone(),
                                            terminate: None,
                                        }),
                                    };
                                    events.extend(entry_lifecycle_events(
                                        committed,
                                        &lane_name,
                                        Some(&run_id),
                                    ));
                                    events
                                }
                            })),
                        });
                    }
                    let Some(tip_id) = placement.tip_id.clone() else {
                        return Err(lane_error(SessionError::Invariant(
                            "Completed run has no tip".to_owned(),
                        )));
                    };
                    if matches!(
                        continuation,
                        Continuation::MayFinish {
                            include_final_assistant: true
                        }
                    ) && scope.latest_assistant_entry_id.is_none()
                    {
                        return Err(lane_error(SessionError::Invariant(
                            "Completed run is missing its final assistant".to_owned(),
                        )));
                    }
                    let record = operation_result_record(
                        &current.meta,
                        TerminalStatus::Completed,
                        Some(tip_id.clone()),
                        None,
                    )?;
                    let cleanup = operation_cleanup_writes(
                        session.as_ref(),
                        &drive.operation_id,
                        &current.state,
                        &drive.context,
                    )
                    .await?;
                    let ended_at = record.ended_at;
                    Ok(OperationCommand::Finish {
                        writes: [placement.writes.clone(), cleanup].concat(),
                        record: record.clone(),
                        lane: Some(LanePatch {
                            tip_id: Some(Some(tip_id.clone())),
                            inbox: Some(placement.inbox.clone()),
                            ..LanePatch::default()
                        }),
                        materialize: Arc::new(move |_: &CommitResult| ProcedureResult::Settled {
                            outcome: record.clone(),
                        }),
                        events: Some(Arc::new({
                            let pending_events = pending_events.clone();
                            let placement = placement.clone();
                            let lane_name = lane.name().to_owned();
                            let run_id = drive.operation_id.clone();
                            let source_tip_id = current.meta.source_tip_id.clone();
                            move |commit: &CommitResult| {
                                let mut events = pending_events.clone();
                                events.extend(boundary_placement_events(
                                    &placement, commit, 0, &lane_name, &run_id,
                                ));
                                events.push(lane_scoped_event(
                                    &lane_name,
                                    false,
                                    "run_end",
                                    HarnessEventPayload::RunEnd {
                                        run_id: run_id.clone(),
                                        from_tip_id: source_tip_id.clone(),
                                        tip_id: Some(tip_id.clone()),
                                        ended_at,
                                        status: RunEndStatus::Completed,
                                    },
                                ));
                                events
                            }
                        })),
                    })
                })
            },
            &drive.context,
        )
        .await?;
    Ok(match result {
        ContinueOperationResult::CancelRequested => ProcedureResult::Continue,
        ContinueOperationResult::Result { value } => value,
    })
}

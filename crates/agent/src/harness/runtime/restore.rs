//! The lane restore, ported from upstream `src/harness/runtime/restore.ts`.
//!
//! The map's drive child owns `restore.ts`; the runtime-foundations child
//! carries it because constructing a lane from durable storage is the lane
//! suite's entry point (recorded on both tickets). The storage
//! classification, the intent/state match, and the invariant messages
//! restate 1:1. Reads ride the session contract's reader half — inside a
//! mutation the session barrier guarantees no other writer interleaves, so
//! reads through the outer session see the same committed state upstream's
//! mutation-line reader does.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::harness::context::Context;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::OperationIntent;
use crate::harness::session::types::OperationMeta;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::Session;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionInvariantError;
use crate::harness::session::types::SessionMutationCallback;
use crate::harness::session::types::SessionReader;
use crate::harness::session::values::branch_tip;
use crate::harness::session::values::branch_tip_inventory_prefix;
use crate::harness::session::values::lane_config;
use crate::harness::session::values::lane_state;
use crate::harness::session::values::operation_meta;
use crate::harness::session::values::operation_state;
use crate::harness::session::values::StoredValue;
use crate::harness::runtime::types::any_payload;
use crate::harness::runtime::types::LaneState;
use crate::harness::runtime::types::LiveOperation;

/// Whether a durable leaf belongs to a summary family, upstream's
/// `isSummaryState`.
#[must_use]
fn is_summary_state(state: &OperationState) -> bool {
    matches!(
        state,
        OperationState::SummaryDeciding(_)
            | OperationState::SummaryReady(_)
            | OperationState::SummaryEffectPending(_)
            | OperationState::SummaryRetryWait(_)
    )
}

/// The state leaf's summary task, when the leaf is a summary family member,
/// upstream's `state.task` reads under the summary-state narrow.
#[must_use]
fn summary_task(state: &OperationState) -> Option<&crate::harness::session::types::SummaryTask> {
    match state {
        OperationState::SummaryDeciding(leaf) => Some(&leaf.task),
        OperationState::SummaryReady(leaf) => Some(&leaf.generation.task),
        OperationState::SummaryEffectPending(leaf) => Some(&leaf.generation.task),
        OperationState::SummaryRetryWait(leaf) => Some(&leaf.generation.task),
        _ => None,
    }
}

fn summary_boundary(state: &OperationState) -> Option<&ResultBoundary> {
    summary_task(state).map(|task| &task.boundary)
}

/// Whether a durable leaf matches the intent it was admitted under,
/// upstream's `stateMatchesIntent`.
#[must_use]
pub fn state_matches_intent(meta: &OperationMeta, state: &OperationState) -> bool {
    match &meta.intent {
        OperationIntent::Compaction { .. } => {
            is_summary_state(state) && matches!(summary_boundary(state), Some(ResultBoundary::Finish))
        }
        OperationIntent::Navigation {
            summarize,
            target_id,
            label,
            custom_instructions,
        } => match state {
            OperationState::NavigationReadyToCommit(leaf) => {
                !*summarize && &leaf.target_id == target_id && &leaf.label == label
            }
            other => {
                let Some(task) = summary_task(other) else {
                    return false;
                };
                let ResultBoundary::CommitNavigation {
                    target_id: boundary_target,
                    label: boundary_label,
                } = &task.boundary
                else {
                    return false;
                };
                *summarize
                    && target_id.as_ref() == Some(boundary_target)
                    && label == boundary_label
                    && &task.custom_instructions == custom_instructions
            }
        },
        OperationIntent::Run { .. } => {
            !matches!(state, OperationState::NavigationReadyToCommit(_))
                && (!is_summary_state(state)
                    || matches!(summary_boundary(state), Some(ResultBoundary::ResumeCheckpoint { .. })))
        }
    }
}

/// The storage classification one lane resolves to, upstream's
/// `ClassifiedLaneStorage`.
pub enum ClassifiedLaneStorage {
    /// No lane values persisted.
    Absent,
    /// Only a branch tip persisted; the lane was never configured.
    Branch {
        /// The branch tip value.
        tip: StoredValue,
    },
    /// A fully configured lane.
    Lane {
        /// The branch tip value.
        tip: StoredValue,
        /// The lane configuration value.
        configuration: StoredValue,
        /// The durable lane record value.
        lane_state: StoredValue,
    },
}

fn classify_lane_storage(
    lane: &str,
    tip: Option<StoredValue>,
    configuration: Option<StoredValue>,
    lane_state: Option<StoredValue>,
) -> Result<ClassifiedLaneStorage, SessionInvariantError> {
    if tip.is_none() && configuration.is_none() && lane_state.is_none() {
        return Ok(ClassifiedLaneStorage::Absent);
    }
    if tip.is_some() && configuration.is_none() && lane_state.is_none() {
        return Ok(ClassifiedLaneStorage::Branch {
            tip: match tip {
                Some(tip) => tip,
                None => return Err(SessionInvariantError(String::new())),
            },
        });
    }
    let Some(tip) = tip else {
        return Err(SessionInvariantError(format!(
            "Lane {lane:?} is missing branch.tip"
        )));
    };
    let Some(configuration) = configuration else {
        return Err(SessionInvariantError(format!(
            "Lane {lane:?} is missing lane.config"
        )));
    };
    let Some(lane_state) = lane_state else {
        return Err(SessionInvariantError(format!(
            "Lane {lane:?} is missing lane.state"
        )));
    };
    Ok(ClassifiedLaneStorage::Lane {
        tip,
        configuration,
        lane_state,
    })
}

/// Reads and classifies one lane's storage values, upstream's
/// `readLaneStorage`.
pub async fn read_lane_storage(
    reader: &dyn SessionReader,
    lane: &str,
    context: &Context,
) -> Result<ClassifiedLaneStorage, SessionError> {
    let (tip, configuration, lane_state) = tokio::join!(
        reader.get_value(&branch_tip(lane).address, context),
        reader.get_value(&lane_config(lane).address, context),
        reader.get_value(&lane_state(lane).address, context),
    );
    classify_lane_storage(lane, tip?, configuration?, lane_state?)
        .map_err(|error| SessionError(error.to_string()))
}

/// The payload one restore mutation carries back: the boxed-Any contract
/// has no error channel, so thrown errors ride the payload.
enum RestoreOutcome {
    /// One lane's restored state.
    Lane(Box<LaneState>),
    /// Every configured lane's restored state.
    Lanes(BTreeMap<String, LaneState>),
    /// The restore threw; upstream's propagated mutation-callback error.
    Threw(SessionError),
}

/// Restores every complete configured lane in one coherent session read,
/// upstream's `restoreSession`.
pub async fn restore_session(
    session: Arc<dyn Session>,
    context: &Context,
) -> Result<BTreeMap<String, LaneState>, SessionError> {
    let callback_session = Arc::clone(&session);
    let mutation: SessionMutationCallback = Box::new(move |_mutator, callback_context| {
        let session = Arc::clone(&callback_session);
        Box::pin(async move {
            let outcome: Result<BTreeMap<String, LaneState>, SessionError> = async {
                let (tips, configurations, states) = tokio::join!(
                    session.scan_values(&branch_tip_inventory_prefix().address, callback_context),
                    session.scan_values(&lane_config("").address, callback_context),
                    session.scan_values(&lane_state("").address, callback_context),
                );
                let mut tip_by_lane: BTreeMap<String, StoredValue> = BTreeMap::new();
                for value in tips? {
                    tip_by_lane.insert(value.key.clone(), value);
                }
                let mut configuration_by_lane: BTreeMap<String, StoredValue> = BTreeMap::new();
                for value in configurations? {
                    configuration_by_lane.insert(value.key.clone(), value);
                }
                let mut state_by_lane: BTreeMap<String, StoredValue> = BTreeMap::new();
                for value in states? {
                    state_by_lane.insert(value.key.clone(), value);
                }
                let names: std::collections::BTreeSet<String> = tip_by_lane
                    .keys()
                    .chain(configuration_by_lane.keys())
                    .chain(state_by_lane.keys())
                    .cloned()
                    .collect();
                let mut restored: BTreeMap<String, LaneState> = BTreeMap::new();
                for lane in names {
                    let classified = classify_lane_storage(
                        &lane,
                        tip_by_lane.get(&lane).cloned(),
                        configuration_by_lane.get(&lane).cloned(),
                        state_by_lane.get(&lane).cloned(),
                    )
                    .map_err(|error| SessionError(error.to_string()))?;
                    let ClassifiedLaneStorage::Lane {
                        tip,
                        configuration,
                        lane_state,
                    } = classified
                    else {
                        continue;
                    };
                    let state = restore_lane_state(
                        session.as_ref(),
                        &lane,
                        &tip,
                        &configuration,
                        &lane_state,
                        callback_context,
                    )
                    .await?;
                    restored.insert(lane, state);
                }
                Ok(restored)
            }
            .await;
            match outcome {
                Ok(restored) => any_payload(RestoreOutcome::Lanes(restored)),
                Err(error) => any_payload(RestoreOutcome::Threw(error)),
            }
        })
    });
    let restored = session.mutate(mutation, context).await?;
    let restored = restored
        .downcast::<RestoreOutcome>()
        .map_err(|_| SessionError("restore_session's callback returns a restore outcome".to_owned()))?;
    match *restored {
        RestoreOutcome::Lanes(restored) => Ok(restored),
        RestoreOutcome::Threw(error) => Err(error),
        RestoreOutcome::Lane(_) => Err(SessionError(
            "restore_session's callback returns a lane map".to_owned(),
        )),
    }
}

/// Restores one configured lane without starting work or interpreting its
/// state, upstream's `restoreLane`.
pub async fn restore_lane(
    session: Arc<dyn Session>,
    lane: &str,
    context: &Context,
) -> Result<LaneState, SessionError> {
    let lane_owned = lane.to_owned();
    let callback_session = Arc::clone(&session);
    let mutation: SessionMutationCallback = Box::new(move |_mutator, callback_context| {
        let session = Arc::clone(&callback_session);
        let lane = lane_owned.clone();
        Box::pin(async move {
            let outcome: Result<LaneState, SessionError> = async {
                let stored = read_lane_storage(session.as_ref(), &lane, callback_context).await?;
                let ClassifiedLaneStorage::Lane {
                    tip,
                    configuration,
                    lane_state,
                } = stored
                else {
                    return Err(match stored {
                        ClassifiedLaneStorage::Absent => SessionInvariantError(format!(
                            "Lane {lane:?} is missing branch.tip"
                        )),
                        ClassifiedLaneStorage::Branch { .. } => SessionInvariantError(format!(
                            "Lane {lane:?} is missing lane.config"
                        )),
                        ClassifiedLaneStorage::Lane { .. } => unreachable!("matched above"),
                    });
                };
                Ok(restore_lane_state(
                    session.as_ref(),
                    &lane,
                    &tip,
                    &configuration,
                    &lane_state,
                    callback_context,
                )
                .await?)
            }
            .await;
            match outcome {
                Ok(state) => any_payload(RestoreOutcome::Lane(Box::new(state))),
                Err(error) => any_payload(RestoreOutcome::Threw(error.into())),
            }
        })
    });
    let restored = session.mutate(mutation, context).await?;
    let restored = restored
        .downcast::<RestoreOutcome>()
        .map_err(|_| SessionError("restore_lane's callback returns a restore outcome".to_owned()))?;
    match *restored {
        RestoreOutcome::Lane(state) => Ok(*state),
        RestoreOutcome::Threw(error) => Err(error),
        RestoreOutcome::Lanes(_) => Err(SessionError(
            "restore_lane's callback returns the lane state".to_owned(),
        )),
    }
}

/// Restores the lane state from classified storage, upstream's
/// `restoreLaneState`.
pub async fn restore_lane_state(
    reader: &dyn SessionReader,
    lane: &str,
    tip: &StoredValue,
    configuration: &StoredValue,
    lane_state: &StoredValue,
    context: &Context,
) -> Result<LaneState, SessionError> {
    let configuration: LaneConfiguration = serde_json::from_value(configuration.value.clone())
        .map_err(|error| SessionInvariantError(format!("Lane {lane:?} config is malformed: {error}")))?;
    let durable: crate::harness::session::types::LaneState = serde_json::from_value(lane_state.value.clone())
        .map_err(|error| SessionInvariantError(format!("Lane {lane:?} state is malformed: {error}")))?;
    let tip_id: Option<String> = serde_json::from_value(tip.value.clone())
        .map_err(|error| SessionInvariantError(format!("Lane {lane:?} tip is malformed: {error}")))?;

    let operation = match durable.current_operation_id {
        None => None,
        Some(operation_id) => {
            let (meta, state) = tokio::join!(
                reader.get_value(&operation_meta(&operation_id).address, context),
                reader.get_value(&operation_state(&operation_id).address, context),
            );
            let meta = meta?.ok_or_else(|| {
                SessionInvariantError(format!("Operation {operation_id} is missing op.meta"))
            })?;
            let state = state?.ok_or_else(|| {
                SessionInvariantError(format!("Operation {operation_id} is missing op.state"))
            })?;
            let meta: OperationMeta = serde_json::from_value(meta.value).map_err(|error| {
                SessionInvariantError(format!("Operation {operation_id} meta is malformed: {error}"))
            })?;
            let state: OperationState = serde_json::from_value(state.value).map_err(|error| {
                SessionInvariantError(format!("Operation {operation_id} state is malformed: {error}"))
            })?;
            if meta.operation_id != operation_id {
                return Err(SessionInvariantError(format!(
                    "Operation {operation_id} metadata names operation {:?}",
                    meta.operation_id
                ))
                .into());
            }
            if meta.lane != lane {
                return Err(SessionInvariantError(format!(
                    "Operation {operation_id} belongs to lane {:?}, not {lane:?}",
                    meta.lane
                ))
                .into());
            }
            if !state_matches_intent(&meta, &state) {
                return Err(SessionInvariantError(format!(
                    "Operation {operation_id} intent {} does not match state {}",
                    intent_kind(&meta),
                    state.at()
                ))
                .into());
            }
            Some(LiveOperation { meta, state })
        }
    };

    Ok(LaneState {
        tip_id,
        configuration,
        inbox: durable.inbox,
        last_operation_id: durable.last_operation_id,
        operation,
    })
}

fn intent_kind(meta: &OperationMeta) -> &'static str {
    match meta.intent {
        OperationIntent::Run { .. } => "run",
        OperationIntent::Compaction { .. } => "compaction",
        OperationIntent::Navigation { .. } => "navigation",
    }
}
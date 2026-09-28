//! The orphaned-effect recovery procedures, ported from upstream
//! `src/harness/runtime/drive/recovery.ts`.
//!
//! `recover_assistant_generation` settles an orphaned assistant request
//! from its bounded committed frame prefix without another provider call;
//! `recover_cancelled_assistant_effect` synthetically settles one
//! cancelled orphaned assistant or deferred effect under its reserved ids.
//! Both publish the recovery `message_start`/`message_end` pair and settle
//! through the response module.
use std::sync::Arc;

use pi_ai::types::Api;
use pi_ai::types::AssistantMessage;
use pi_ai::types::Message;
use pi_ai::types::ProviderId;
use pi_ai::types::StopReason;
use pi_ai::types::Usage;
use pi_ai::types::UsageCost;
use pi_ai::utils::assistant_message_frame::AssistantMessageFrame;
use pi_ai::utils::assistant_message_frame::reduce_assistant_message_frames;

use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::lane_scoped_event;
use crate::harness::runtime::clock::now_ms;
use crate::harness::runtime::drive::response::ResponseIntent;
use crate::harness::runtime::drive::response::ResponseOptions;
use crate::harness::runtime::drive::response::publish_response;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::progress::read_assistant_frames;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::runtime::types::ProcedureResult;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::ModelIdentity;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SettledAssistantMessage;
use crate::harness::session::types::SettledStopReason;
use crate::types::AgentMessage;

/// The zero usage the interrupted settlement reports, upstream's
/// `ZERO_USAGE` literal. The two `Option` fields upstream's object does not
/// carry stay `None` — the wire omits them, as upstream's undefined fields
/// do.
const ZERO_USAGE: Usage = Usage {
    input: 0,
    output: 0,
    cache_read: 0,
    cache_write: 0,
    cache_write_1h: None,
    reasoning: None,
    total_tokens: 0,
    cost: UsageCost {
        input: 0.0,
        output: 0.0,
        cache_read: 0.0,
        cache_write: 0.0,
        total: 0.0,
    },
};

/// Builds the interrupted settlement, upstream's `interruptedAssistantMessage`:
/// the committed partial restated as the `error` stop with [`ZERO_USAGE`] and
/// the interruption warning, or — when no start frame committed — a fresh
/// empty message under the identity's provider and model carrying a fresh
/// timestamp. Both halves settle consistently: the message's
/// [`StopReason::Error`] mirrors the returned [`SettledStopReason::Error`].
fn interrupted_assistant_message(
    identity: &ModelIdentity,
    partial: Option<AssistantMessage>,
) -> SettledAssistantMessage {
    let warning = "Assistant request was interrupted. The preceding content is the latest committed partial; newer live output may be missing and the external outcome is unknown.";
    let mut message = partial.unwrap_or_else(|| AssistantMessage {
        content: Vec::new(),
        api: Api("unknown".to_owned()),
        provider: ProviderId(identity.provider.clone()),
        model: identity.model_id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: ZERO_USAGE,
        stop_reason: StopReason::Error,
        deferred: None,
        error_message: Some(warning.to_owned()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms(),
    });
    message.usage = ZERO_USAGE;
    message.stop_reason = StopReason::Error;
    message.error_message = Some(warning.to_owned());
    SettledAssistantMessage {
        message,
        stop_reason: SettledStopReason::Error,
    }
}

/// Publishes the recovery `message_start`/`message_end` pair, upstream's
/// `lane.emitBatch([...])` both procedures run: the interrupted message
/// announces under the run id, the end event names the response entry, and
/// both carry the lane-scoped recovery flag.
///
/// # Errors
/// The batch delivery's failure, upstream's rejected `emitBatch` promise.
async fn emit_recovery_events(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    message: &SettledAssistantMessage,
    response_entry_id: &str,
) -> Result<(), LaneError> {
    let lane_name = lane.name().to_owned();
    let run_id = drive.operation_id.clone();
    let start = lane_scoped_event(
        &lane_name,
        true,
        "message_start",
        HarnessEventPayload::MessageStart {
            run_id: Some(run_id.clone()),
            message: AgentMessage::Standard(Message::Assistant(message.message.clone())),
        },
    );
    let end = lane_scoped_event(
        &lane_name,
        true,
        "message_end",
        HarnessEventPayload::MessageEnd {
            run_id: Some(run_id),
            message: AgentMessage::Standard(Message::Assistant(message.message.clone())),
            entry_id: Some(response_entry_id.to_owned()),
        },
    );
    (lane.emit_batch())(vec![start, end], drive.context.clone()).await
}

/// Settle an orphaned assistant request from its bounded committed frame
/// prefix without another provider call, upstream's
/// `recoverAssistantGeneration`.
///
/// The frames under the reserved response entry reduce to the committed
/// partial; the settlement restates it as the interrupted `error` stop with
/// zero usage — a fresh `"unknown"`-API message when no start frame
/// committed — the `message_start`/`message_end` recovery pair publishes,
/// and the response module settles the leaf. Cancellation requested before
/// the frame read downgrades to a plain continue, upstream's
/// `continueOperation` contract.
///
/// # Errors
/// The malformed-frame error restated from
/// `reduce_assistant_message_frames`, the frame read's and recovery
/// events' storage and delivery errors, and the response publication's
/// errors.
pub(crate) async fn recover_assistant_generation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    generation: &AssistantEffectPendingOperation,
) -> Result<ProcedureResult, LaneError> {
    let planner_drive = Arc::clone(drive);
    let planner_entry_id = generation.response_entry_id.clone();
    let frames = lane
        .continue_operation::<Vec<AssistantMessageFrame>, _>(
            move |_state, reader, _context| {
                let drive = Arc::clone(&planner_drive);
                let response_entry_id = planner_entry_id.clone();
                Box::pin(async move {
                    let frames = read_assistant_frames(
                        reader.as_ref(),
                        &drive.operation_id,
                        &response_entry_id,
                        &drive.context,
                    )
                    .await
                    .map_err(lane_error)?;
                    Ok(OperationCommand::Return { result: frames })
                })
            },
            &drive.context,
        )
        .await?;
    let ContinueOperationResult::Result { value: frames } = frames else {
        return Ok(ProcedureResult::Continue);
    };

    let partial = reduce_assistant_message_frames(frames)
        .map_err(|message| lane_error(SessionError::Message(message)))?;
    let message =
        interrupted_assistant_message(&generation.generation_context.configuration.model, partial);
    emit_recovery_events(lane, drive, &message, &generation.response_entry_id).await?;
    publish_response(
        lane,
        drive,
        ResponseIntent::Assistant(generation.clone()),
        message,
        ResponseOptions { recovery: true },
    )
    .await
}

/// Synthetically settle one cancelled orphaned assistant or deferred effect
/// under its reserved ids, upstream's `recoverCancelledAssistantEffect`.
///
/// The settle runs even when cancellation is requested, upstream's
/// `settleOperation` contract — there is no cancel-refused arm. The
/// interrupted settlement takes the leaf's own identity: the assistant
/// leaf's generation configuration, the deferred leaf's deferred scope
/// configuration.
///
/// # Errors
/// The malformed-frame error restated from
/// `reduce_assistant_message_frames`, the frame read's and recovery
/// events' storage and delivery errors, and the response publication's
/// errors.
pub(crate) async fn recover_cancelled_assistant_effect(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    effect: &OperationState,
) -> Result<ProcedureResult, LaneError> {
    let (identity, response_entry_id, intent) = match effect {
        OperationState::AssistantEffectPending(leaf) => (
            &leaf.generation_context.configuration.model,
            leaf.response_entry_id.clone(),
            ResponseIntent::Assistant(leaf.clone()),
        ),
        OperationState::DeferredEffectPending(leaf) => (
            &leaf.scope.configuration.model,
            leaf.response_entry_id.clone(),
            ResponseIntent::Deferred(leaf.clone()),
        ),
        _ => unreachable!("recoverCancelledAssistantEffect runs under an effect-pending leaf"),
    };

    let planner_drive = Arc::clone(drive);
    let planner_entry_id = response_entry_id.clone();
    let frames = lane
        .settle_operation::<Vec<AssistantMessageFrame>, _>(
            move |_state, _current, _meta, reader, _context| {
                let drive = Arc::clone(&planner_drive);
                let response_entry_id = planner_entry_id.clone();
                Box::pin(async move {
                    let frames = read_assistant_frames(
                        reader.as_ref(),
                        &drive.operation_id,
                        &response_entry_id,
                        &drive.context,
                    )
                    .await
                    .map_err(lane_error)?;
                    Ok(OperationCommand::Return { result: frames })
                })
            },
            &drive.context,
        )
        .await?;

    let partial = reduce_assistant_message_frames(frames)
        .map_err(|message| lane_error(SessionError::Message(message)))?;
    let message = interrupted_assistant_message(identity, partial);
    emit_recovery_events(lane, drive, &message, &response_entry_id).await?;
    publish_response(
        lane,
        drive,
        intent,
        message,
        ResponseOptions { recovery: true },
    )
    .await
}

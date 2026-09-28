//! The response settlement procedures, ported from upstream
//! `src/harness/runtime/drive/response.ts`.
//!
//! `publish_response` classifies one settled assistant message and
//! commits its disposition — the response entry, its usage row, and the
//! next durable leaf or terminal failure. `publish_configuration_failure`
//! fails a run before any response id is reserved. `open_assistant_response`
//! opens the frame-progress lifecycle a generation streams through.
//!
//! Porting restatement: upstream's observer callbacks throw the frame
//! encoder's errors and reject `emitBatch` promises, both aborting the
//! stream; the [`AssistantStreamObserver`] trait has no error channel, so
//! the first failure is held in the lifecycle's shared state, later
//! callbacks no-op (upstream's dead stream), and the close capability
//! raises it after the seal and drain — the same point
//! `stream_harness_assistant`'s caller awaits in its cleanup.
#![expect(
    dead_code,
    reason = "the generation, recovery, and deferred children call these procedures and open the lifecycle; until they land the crate sees no caller"
)]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;

use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::BoxedFuture;
use pi_ai::types::Message;
use pi_ai::types::StopReason;
use pi_ai::utils::assistant_message_frame::AssistantMessageFrame;
use pi_ai::utils::assistant_message_frame::AssistantMessageFrameEncoder;
use pi_ai::utils::overflow::is_context_overflow;
use pi_ai::utils::overflow::is_recoverable_length;
use pi_ai::utils::retry::RetryPolicy;
use pi_ai::utils::retry::is_retryable_assistant_error;
use pi_ai::utils::retry::retry_delay_ms;

use crate::harness::agent_harness::HarnessEvent;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HookEvent;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::agent_harness::HookResult;
use crate::harness::agent_harness::RunEndStatus;
use crate::harness::agent_harness::global_event;
use crate::harness::agent_harness::lane_scoped_event;
use crate::harness::context::Context;
use crate::harness::execution::assistant::AfterResponseHook;
use crate::harness::execution::assistant::AssistantStreamObserver;
use crate::harness::gate::AbortRequested;
use crate::harness::hooks::HookRunError;
use crate::harness::runtime::clock::now_ms;
use crate::harness::runtime::drive::retry::retry_not_before;
use crate::harness::runtime::drive::structural::prepare_overflow_compaction;
use crate::harness::runtime::drive::terminal::operation_cleanup_writes;
use crate::harness::runtime::drive::terminal::operation_result_record;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::progress::ProgressChannel;
use crate::harness::runtime::progress::open_frame_progress;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::EventsFn;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::LanePatch;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::runtime::types::ProcedureResult;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::commit::insert_entry;
use crate::harness::session::commit::insert_usage;
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::AssistantRetryWaitOperation;
use crate::harness::session::types::CheckpointData;
use crate::harness::session::types::CheckpointOperation;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::CompactionReason;
use crate::harness::session::types::Continuation;
use crate::harness::session::types::Control;
use crate::harness::session::types::DeferredEffectPendingOperation;
use crate::harness::session::types::DeferredScope;
use crate::harness::session::types::DeferredSuspendedOperation;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::MessageEntry;
use crate::harness::session::types::NewEntry;
use crate::harness::session::types::NormalizedRetryPolicy;
use crate::harness::session::types::OperationError;
use crate::harness::session::types::OperationScope;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::ResultBoundary;
use crate::harness::session::types::RetryWait;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SettledAssistantMessage;
use crate::harness::session::types::SettledStopReason;
use crate::harness::session::types::SummaryDecidingOperation;
use crate::harness::session::types::SummaryTask;
use crate::harness::session::types::TerminalStatus;
use crate::harness::session::types::ToolBatch;
use crate::harness::session::types::ToolCall;
use crate::harness::session::types::ToolsOperation;
use crate::harness::session::types::UsageRow;
use crate::harness::session::types::UsageWriteRow;
use crate::harness::session::values::Write;
use crate::harness::session::values::branch_tip;
use crate::harness::session::values::delete_list;
use crate::harness::session::values::operation_preparation;
use crate::harness::session::values::pending_assistant_frames;
use crate::harness::session::values::set_value_write;
use crate::types::AgentMessage;

/// The two effect-pending leaves one settled response publishes, upstream's
/// `ResponseIntent` (`AssistantEffectPendingOperation |
/// DeferredEffectPendingOperation`).
#[derive(Clone, Debug)]
pub(crate) enum ResponseIntent {
    /// The `assistant.effect_pending` leaf.
    Assistant(AssistantEffectPendingOperation),
    /// The `deferred.effect_pending` leaf.
    Deferred(DeferredEffectPendingOperation),
}

impl ResponseIntent {
    /// The response entry the settlement commits to, upstream's
    /// `intent.responseEntryId`.
    fn response_entry_id(&self) -> &str {
        match self {
            Self::Assistant(leaf) => &leaf.response_entry_id,
            Self::Deferred(leaf) => &leaf.response_entry_id,
        }
    }

    /// The reserved usage row id, upstream's `intent.usageId`.
    fn usage_id(&self) -> &str {
        match self {
            Self::Assistant(leaf) => &leaf.usage_id,
            Self::Deferred(leaf) => &leaf.usage_id,
        }
    }
}

/// The settlement options, upstream's `{ recovery?: true }`: recovery mode
/// settles under the reserved ids without touching the retry budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ResponseOptions {
    /// Whether the settlement is crash recovery, upstream's `recovery`.
    pub recovery: bool,
}

/// The stream lifecycle one assistant response opens, upstream's
/// `AssistantResponseLifecycle`.
pub(crate) struct AssistantResponseLifecycle {
    /// The stream observer publishing the frame progress and message
    /// events, upstream's `observer`.
    pub observer: Arc<dyn AssistantStreamObserver>,
    /// The after-response settlement binding, upstream's `afterResponse`.
    pub after_response: AfterResponseHook,
    /// Closes the lifecycle, upstream's `close()`: seals the frame progress
    /// and awaits its writes.
    pub close: CloseAssistantResponse,
}

impl std::fmt::Debug for AssistantResponseLifecycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AssistantResponseLifecycle(..)")
    }
}

/// The close capability of [`AssistantResponseLifecycle`], upstream's
/// `close(): Promise<void>` — `Ok` when the progress drained and no stream
/// failure was held, its write failure or the held frame-encoder/emit
/// failure otherwise.
pub(crate) type CloseAssistantResponse =
    Arc<dyn Fn() -> BoxedFuture<'static, Result<(), LaneError>> + Send + Sync>;

/// The request family one settled response belongs to, upstream's
/// `"assistant" | "deferred"` source discriminator.
#[derive(Clone, Copy)]
enum ResponseSource {
    /// The assistant generation.
    Assistant,
    /// The deferred poll.
    Deferred,
}

impl ResponseSource {
    /// The capitalized request name the error templates interpolate,
    /// upstream's `source === "assistant" ? "Assistant" : "Deferred"`.
    const fn request_name(self) -> &'static str {
        match self {
            Self::Assistant => "Assistant",
            Self::Deferred => "Deferred",
        }
    }
}

/// The response leaf one settlement transaction re-read, upstream's
/// `current` narrowed from the fresh operation state the settle planner
/// receives.
enum ResponseLeaf<'a> {
    /// The `assistant.effect_pending` leaf.
    Assistant(&'a AssistantEffectPendingOperation),
    /// The `deferred.effect_pending` leaf.
    Deferred(&'a DeferredEffectPendingOperation),
}

impl<'a> ResponseLeaf<'a> {
    /// Narrows the dispatched leaf, upstream's `current` typed by the
    /// `ResponseIntent` capability.
    ///
    /// # Panics
    /// When the live operation is not one of the two response leaves — the
    /// caller passes the intent the settled state must match.
    fn narrow(state: &'a OperationState) -> Self {
        match state {
            OperationState::AssistantEffectPending(leaf) => Self::Assistant(leaf),
            OperationState::DeferredEffectPending(leaf) => Self::Deferred(leaf),
            _ => unreachable!("settleOperation planner runs under the dispatched response leaf"),
        }
    }

    /// The request family the error templates interpolate, upstream's
    /// `source`.
    const fn source(&self) -> ResponseSource {
        match self {
            Self::Assistant(_) => ResponseSource::Assistant,
            Self::Deferred(_) => ResponseSource::Deferred,
        }
    }

    /// The uniform scope the leaf carries, upstream's `current` scope reads.
    fn scope(&self) -> OperationScope {
        match self {
            Self::Assistant(leaf) => leaf.scope.clone(),
            Self::Deferred(leaf) => leaf.scope.scope.clone(),
        }
    }

    /// The configuration snapshot the next leaf carries, upstream's
    /// `configuration`.
    const fn configuration(&self) -> &LaneConfiguration {
        match self {
            Self::Assistant(leaf) => &leaf.generation_context.configuration,
            Self::Deferred(leaf) => &leaf.scope.configuration,
        }
    }

    /// The turn id the events and tool batch carry, upstream's `turnId`:
    /// the step id for assistant responses, the poll-local
    /// `<stepId>:poll:<poll>` id for deferred polls.
    fn turn_id(&self) -> String {
        match self {
            Self::Assistant(leaf) => leaf.generation_context.step_id.clone(),
            Self::Deferred(leaf) => format!("{}:poll:{}", leaf.scope.step_id, leaf.scope.poll),
        }
    }

    /// The uniform scope one settled response's next leaf carries with the
    /// response entry recorded, upstream's
    /// `{ ...operationScopeOf(current), latestAssistantEntryId }`.
    fn settlement_scope(&self, response_entry_id: &str) -> OperationScope {
        let mut scope = self.scope();
        scope.latest_assistant_entry_id = Some(response_entry_id.to_owned());
        scope
    }
}

/// The shared stream state the observer, the after-response binding, and
/// the close capability hold, upstream's `openAssistantResponse` closure
/// captures.
struct ResponseStreamShared {
    /// The frame encoder, upstream's `frameEncoder`.
    encoder: Mutex<AssistantMessageFrameEncoder>,
    /// The first stream failure — a frame-encoder error or an event
    /// delivery failure — upstream's throw aborting the stream.
    failure: Mutex<Option<LaneError>>,
    /// The frame progress channel, upstream's `progress`.
    progress: ProgressChannel<AssistantMessageFrame>,
}

impl ResponseStreamShared {
    /// Whether an earlier event already failed, upstream's stream being
    /// dead after its first throw.
    fn failed(&self) -> bool {
        self.failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }

    /// Encodes one event, upstream's `frameEncoder.encode(event)`.
    ///
    /// # Errors
    /// The frame-encoder protocol error, upstream's throw.
    fn encode(
        &self,
        event: &pi_ai::types::AssistantMessageEvent,
    ) -> Result<Option<AssistantMessageFrame>, LaneError> {
        let mut encoder = self.encoder.lock().unwrap_or_else(PoisonError::into_inner);
        encoder
            .encode(event.clone())
            .map_err(|message| lane_error(SessionError::Message(message)))
    }

    /// Records the first stream failure; later ones yield to it, upstream's
    /// first throw aborting the stream.
    fn hold_failure(&self, error: LaneError) {
        let mut failure = self.failure.lock().unwrap_or_else(PoisonError::into_inner);
        if failure.is_none() {
            *failure = Some(error);
        }
    }

    /// Takes the held stream failure, upstream's one-shot error surfacing.
    fn take_failure(&self) -> Option<LaneError> {
        self.failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }
}

/// Seals the frame progress, awaits its writes, and raises the held stream
/// failure, upstream's `close`.
///
/// # Errors
/// The newest frame write's commit failure, or the held frame-encoder/emit
/// failure the stream carried.
async fn close_response_stream(shared: &ResponseStreamShared) -> Result<(), LaneError> {
    shared.progress.seal();
    shared.progress.drain().await?;
    shared.take_failure().map_or(Ok(()), Err)
}

/// The frame-progress observer one response lifecycle installs, upstream's
/// observer object literal: each event's frame encodes and persists before
/// the message event publishes.
struct ResponseObserver {
    /// The shared encoder, failure cell, and progress channel.
    shared: Arc<ResponseStreamShared>,
    /// The lane the message events publish through.
    lane: Arc<Lane>,
    /// The lane name the events carry, upstream's `eventContext.lane`.
    lane_name: String,
    /// The run id the events carry, upstream's `eventContext.runId`.
    run_id: String,
    /// The committed entry id the end event carries, upstream's
    /// `entryId`.
    response_entry_id: String,
    /// Whether the events replay recovered work, upstream's
    /// `eventContext.recovery`.
    recovery: bool,
}

impl ResponseObserver {
    /// Publishes one message event through the lane's batch emitter; a
    /// delivery failure joins the held stream failure, upstream's rejected
    /// `emitBatch` promise aborting the stream.
    async fn emit(&self, event: HarnessEvent, context: &Context) {
        if let Err(failure) = (self.lane.emit_batch())(vec![event], context.clone()).await {
            self.shared.hold_failure(failure);
        }
    }

    /// Publishes the stream's start, upstream's `observer.start`.
    async fn publish_start(
        &self,
        message: AssistantMessage,
        event: pi_ai::types::AssistantMessageEvent,
        context: Context,
    ) {
        if self.shared.failed() {
            return;
        }
        match self.shared.encode(&event) {
            Ok(Some(frame)) => self.shared.progress.write(frame),
            Ok(None) => {}
            Err(failure) => {
                self.shared.hold_failure(failure);
                return;
            }
        }
        let published = lane_scoped_event(
            &self.lane_name,
            self.recovery,
            "message_start",
            HarnessEventPayload::MessageStart {
                run_id: Some(self.run_id.clone()),
                message: AgentMessage::Standard(Message::Assistant(message)),
            },
        );
        self.emit(published, &context).await;
    }

    /// Publishes one partial update, upstream's `observer.update`; the
    /// encoded frame rides the event only when the encoder produced one.
    async fn publish_update(
        &self,
        message: AssistantMessage,
        event: pi_ai::types::AssistantMessageEvent,
        context: Context,
    ) {
        if self.shared.failed() {
            return;
        }
        let frame = match self.shared.encode(&event) {
            Ok(frame) => frame,
            Err(failure) => {
                self.shared.hold_failure(failure);
                return;
            }
        };
        if let Some(frame) = &frame {
            self.shared.progress.write(frame.clone());
        }
        let published = lane_scoped_event(
            &self.lane_name,
            self.recovery,
            "message_update",
            HarnessEventPayload::MessageUpdate {
                run_id: self.run_id.clone(),
                message: Box::new(AgentMessage::Standard(Message::Assistant(message))),
                event: Box::new(event),
                frame,
            },
        );
        self.emit(published, &context).await;
    }

    /// Publishes the stream's settlement, upstream's `observer.end`; the
    /// message is the final, possibly hook-replaced settlement.
    async fn publish_end(&self, message: SettledAssistantMessage, context: Context) {
        if self.shared.failed() {
            return;
        }
        let published = lane_scoped_event(
            &self.lane_name,
            self.recovery,
            "message_end",
            HarnessEventPayload::MessageEnd {
                run_id: Some(self.run_id.clone()),
                message: AgentMessage::Standard(Message::Assistant(message.message.clone())),
                entry_id: Some(self.response_entry_id.clone()),
            },
        );
        self.emit(published, &context).await;
    }
}

impl AssistantStreamObserver for ResponseObserver {
    fn start(
        &self,
        message: AssistantMessage,
        event: &pi_ai::types::AssistantMessageEvent,
        context: &Context,
    ) -> BoxedFuture<'_, ()> {
        // The callbacks own their inputs, the [`AfterResponseHook`]
        // precedent — the boxed future is `'static` and the trait's
        // elided lifetime accepts it.
        Box::pin(self.publish_start(message, event.clone(), context.clone()))
    }

    fn update(
        &self,
        message: AssistantMessage,
        event: &pi_ai::types::AssistantMessageEvent,
        context: &Context,
    ) -> BoxedFuture<'_, ()> {
        Box::pin(self.publish_update(message, event.clone(), context.clone()))
    }

    fn end(&self, message: &SettledAssistantMessage, context: &Context) -> BoxedFuture<'_, ()> {
        Box::pin(self.publish_end(message.clone(), context.clone()))
    }
}

/// Opens the response lifecycle for one response entry, upstream's
/// `openAssistantResponse`: frame progress under the operation's ownership,
/// the message-event observer, the `after_response` hook binding, and the
/// close future.
///
/// The after-response binding seals and drains the frame progress before
/// the hook runs, upstream's `await close()`; a held stream failure raises
/// there first, where upstream's throw had already aborted the stream.
#[must_use]
pub(crate) fn open_assistant_response(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    response_entry_id: &str,
    recovery: bool,
) -> AssistantResponseLifecycle {
    let progress = open_frame_progress(lane, drive, response_entry_id);
    let shared = Arc::new(ResponseStreamShared {
        encoder: Mutex::new(AssistantMessageFrameEncoder::default()),
        failure: Mutex::new(None),
        progress,
    });
    let observer: Arc<dyn AssistantStreamObserver> = Arc::new(ResponseObserver {
        shared: Arc::clone(&shared),
        lane: Arc::clone(lane),
        lane_name: lane.name().to_owned(),
        run_id: drive.operation_id.clone(),
        response_entry_id: response_entry_id.to_owned(),
        recovery,
    });

    let close: CloseAssistantResponse = {
        let shared = Arc::clone(&shared);
        Arc::new(move || {
            let shared = Arc::clone(&shared);
            Box::pin(async move { close_response_stream(&shared).await })
        })
    };

    let after_response: AfterResponseHook = {
        let shared = Arc::clone(&shared);
        let lane = Arc::clone(lane);
        let drive = Arc::clone(drive);
        let lane_name = lane.name().to_owned();
        let run_id = drive.operation_id.clone();
        Arc::new(move |message, metadata, context| {
            let shared = Arc::clone(&shared);
            let lane = Arc::clone(&lane);
            let drive = Arc::clone(&drive);
            let lane_name = lane_name.clone();
            let run_id = run_id.clone();
            Box::pin(async move {
                close_response_stream(&shared).await?;
                let result = match lane
                    .hooks()
                    .run_with_gate(
                        HookName::AfterResponse,
                        HookInvocation {
                            lane: lane_name,
                            run_id,
                            event: HookEvent::AfterResponse {
                                status: metadata.status,
                                headers: metadata.headers,
                                message: message.clone(),
                            },
                        },
                        &drive.gate,
                        &context,
                    )
                    .await
                {
                    Ok(result) => result,
                    Err(HookRunError::GateAborted(abort)) => {
                        return Err(Box::from(abort));
                    }
                    Err(HookRunError::Aborted(_)) => {
                        return Err(Box::from(AbortRequested {
                            cancellation: drive.abort_cancellation(),
                        }));
                    }
                    Err(error) => {
                        return Err(Box::from(error));
                    }
                };
                let HookResult::AfterResponse(result) = result else {
                    unreachable!("after_response returns its own result variant")
                };
                Ok(result.and_then(|result| result.message).unwrap_or(message))
            })
        })
    };

    AssistantResponseLifecycle {
        observer,
        after_response,
        close,
    }
}

/// Publish a non-retryable request-configuration failure before reserving
/// response ids, upstream's `publishConfigurationFailure`.
///
/// The capability is one of the four leaves the publication accepts —
/// `assistant.ready`, `assistant.retry_wait`, `deferred.suspended`,
/// `deferred.effect_pending`, upstream's `ConfigurationFailureState` bound;
/// the planner re-reads the lane's live leaf. Cancellation requested before
/// the planner downgrades to a plain continue, upstream's
/// `continueOperation` contract.
///
/// # Errors
/// The invariant `` `Failed run has no Branch tip` ``, the failed-result
/// record invariant, the cleanup's storage errors, and the commit's storage
/// and delivery errors.
pub(crate) async fn publish_configuration_failure(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    _capability: &OperationState,
    error: OperationError,
) -> Result<ProcedureResult, LaneError> {
    let planner_lane = Arc::clone(lane);
    let planner_drive = Arc::clone(drive);
    let planner_error = error.clone();
    let published = lane
        .continue_operation::<ProcedureResult, _>(
            move |state, session, _context| {
                let lane = Arc::clone(&planner_lane);
                let drive = Arc::clone(&planner_drive);
                let error = planner_error.clone();
                Box::pin(async move {
                    let Some(operation) = state.operation.as_ref() else {
                        unreachable!("continueOperation planner runs under an active operation")
                    };
                    let Some(tip_id) = state.tip_id.clone() else {
                        return Err(lane_error(SessionError::Invariant(
                            "Failed run has no Branch tip".to_owned(),
                        )));
                    };
                    let record = operation_result_record(
                        &operation.meta,
                        TerminalStatus::Failed,
                        Some(tip_id.clone()),
                        Some(error.clone()),
                    )?;
                    let cleanup = operation_cleanup_writes(
                        session.as_ref(),
                        &drive.operation_id,
                        &operation.state,
                        &drive.context,
                    )
                    .await?;
                    let lane_name = lane.name().to_owned();
                    let run_id = drive.operation_id.clone();
                    let from_tip_id = operation.meta.source_tip_id.clone();
                    let ended_at = record.ended_at;
                    Ok(OperationCommand::Finish {
                        writes: cleanup,
                        record: record.clone(),
                        lane: None,
                        materialize: Arc::new(move |_: &CommitResult| ProcedureResult::Settled {
                            outcome: record.clone(),
                        }),
                        events: Some(Arc::new(move |_commit: &CommitResult| {
                            vec![lane_scoped_event(
                                &lane_name,
                                false,
                                "run_end",
                                HarnessEventPayload::RunEnd {
                                    run_id: run_id.clone(),
                                    from_tip_id: from_tip_id.clone(),
                                    tip_id: Some(tip_id.clone()),
                                    ended_at,
                                    status: RunEndStatus::Failed {
                                        error: error.clone(),
                                    },
                                },
                            )]
                        })),
                    })
                })
            },
            &drive.context,
        )
        .await?;
    Ok(match published {
        ContinueOperationResult::CancelRequested => ProcedureResult::Continue,
        ContinueOperationResult::Result { value } => value,
    })
}

/// The timestamp a reserved response entry id encodes, upstream's
/// `uuidV7Timestamp`: the first 48 bits of the UUIDv7, the two hex runs
/// before and after the first dash. The parse is strict where upstream's
/// `Number.parseInt` is lenient — a malformed reserved id is the invariant,
/// and 48 bits never exceed the safe-integer range the upstream check
/// guards.
///
/// # Errors
/// The [`SessionError::Invariant`] `` `Invalid reserved UUIDv7 {id}` ``
/// when the id is shorter than the timestamp fields or not hexadecimal.
fn uuid_v7_timestamp(id: &str) -> Result<i64, LaneError> {
    let timestamp = id
        .get(..8)
        .zip(id.get(9..13))
        .and_then(|(head, tail)| i64::from_str_radix(&format!("{head}{tail}"), 16).ok());
    let Some(timestamp) = timestamp else {
        return Err(lane_error(SessionError::Invariant(format!(
            "Invalid reserved UUIDv7 {id}"
        ))));
    };
    Ok(timestamp)
}

/// The wire spelling of a settled stop reason, upstream's
/// `${message.stopReason}` interpolation — the serde renames pin the same
/// strings.
const fn stop_reason_wire(stop_reason: SettledStopReason) -> &'static str {
    match stop_reason {
        SettledStopReason::Stop => "stop",
        SettledStopReason::Length => "length",
        SettledStopReason::ToolUse => "toolUse",
        SettledStopReason::Error => "error",
        SettledStopReason::Aborted => "aborted",
        SettledStopReason::Deferred => "deferred",
    }
}

/// Builds one provider failure, upstream's `providerError`: the settled
/// message's own error text, or the stop-reason fallback naming the request
/// family.
fn provider_error(source: ResponseSource, message: &SettledAssistantMessage) -> OperationError {
    OperationError {
        code: "assistant_error".to_owned(),
        message: message.message.error_message.clone().unwrap_or_else(|| {
            format!(
                "{} request ended with {}",
                source.request_name(),
                stop_reason_wire(message.stop_reason)
            )
        }),
        details: None,
    }
}

/// Rewrites one settled response as the `error` stop, upstream's
/// `normalizeError`: the wrapper's stop reason and the message body's move
/// together, so the committed entry matches the settlement.
fn normalize_error(
    message: SettledAssistantMessage,
    error_message: String,
) -> SettledAssistantMessage {
    SettledAssistantMessage {
        stop_reason: SettledStopReason::Error,
        message: AssistantMessage {
            stop_reason: StopReason::Error,
            error_message: Some(error_message),
            ..message.message
        },
    }
}

/// Rewrites one settled response as the `aborted` stop with the request
/// family's fallback text, upstream's `normalizeAborted`.
fn normalize_aborted(
    source: ResponseSource,
    message: SettledAssistantMessage,
) -> SettledAssistantMessage {
    let error_message = message
        .message
        .error_message
        .clone()
        .unwrap_or_else(|| format!("{} request was cancelled", source.request_name()));
    SettledAssistantMessage {
        stop_reason: SettledStopReason::Aborted,
        message: AssistantMessage {
            stop_reason: StopReason::Aborted,
            error_message: Some(error_message),
            ..message.message
        },
    }
}

/// Whether one deferred response carries a usable durable handle, upstream's
/// `deferredHandleIsValid`: the deferred stop, a non-empty provider id, and
/// the handle matching the generation's identity and the response's wire
/// API — the handle's `api` string compares against the message's API id,
/// both rendering the same wire value.
fn deferred_handle_is_valid(
    message: &SettledAssistantMessage,
    generation: &AssistantEffectPendingOperation,
) -> bool {
    let Some(handle) = &message.message.deferred else {
        return false;
    };
    let identity = &generation.generation_context.configuration.model;
    message.stop_reason == SettledStopReason::Deferred
        && !handle.id.is_empty()
        && handle.provider == identity.provider
        && handle.model_id == identity.model_id
        && handle.api.as_str() == message.message.api.0.as_str()
}

/// The planned phase one batch call starts in, upstream's
/// `{ status: "planned", sourceIndex, resultEntryId }` literal. The phase
/// field is private to the session types module, so the literal restates
/// through the wire shape both sides already serialize; the planned phase
/// carries no payload.
fn planned_tool_call(source_index: u64, result_entry_id: &str) -> ToolCall {
    let wire = serde_json::json!({
        "sourceIndex": source_index,
        "resultEntryId": result_entry_id,
        "status": "planned",
    });
    match serde_json::from_value::<ToolCall>(wire) {
        Ok(call) => call,
        Err(error) => unreachable!("the planned tool call round-trips: {error}"),
    }
}

/// The delay one failed attempt waits, upstream's `retryDelayMs(policy,
/// attempt)` over the generation context's normalized policy: pi-ai's delay
/// re-derives through the equivalent [`RetryPolicy`] (`maxAttempts − 1`
/// retries, the normalized delay cap made explicit), the same seam
/// [`retry_not_before`] and the structural module build.
fn policy_delay_ms(policy: &NormalizedRetryPolicy, attempt: u32) -> u64 {
    let policy = RetryPolicy {
        enabled: true,
        max_retries: policy.max_attempts.saturating_sub(1),
        base_delay_ms: policy.base_delay_ms,
        max_agent_delay_ms: Some(policy.max_agent_delay_ms),
    };
    retry_delay_ms(&policy, attempt)
}

/// Classify one settled response and commit its durable disposition,
/// upstream's `publishResponse`.
///
/// The classification ladder is first-match: cancellation normalizes the
/// response as aborted and checkpoints; an aborted response under running
/// control is the invariant; overflow arms the prepared compaction's
/// `summary.deciding` task or fails; a deferred stop suspends the run on a
/// valid handle or fails; an error stop retries within the attempt budget
/// (recovery mode retries non-retryable errors too) or fails; otherwise the
/// tool calls open the batch, a tool-use stop without calls fails, and a
/// bare stop checkpoints. The settled — possibly rewritten — message always
/// commits with its usage row and the branch tip, whatever the disposition.
///
/// The settle runs even when cancellation is requested, upstream's
/// `settleOperation` contract.
///
/// # Errors
/// The invariants `` `{Assistant|Deferred} response is aborted while
/// durable control is running` ``, `` `Response settlement has no durable
/// disposition` ``, `` `Response settlement is missing its next state` ``,
/// `` `Invalid reserved UUIDv7 {id}` ``, the failed-result record
/// invariant, the overflow preparation's and cleanup's storage errors, and
/// the commit's storage and delivery errors.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single publishResponse method and its classification ladder"
)]
pub(crate) async fn publish_response(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    intent: ResponseIntent,
    response: SettledAssistantMessage,
    options: ResponseOptions,
) -> Result<ProcedureResult, LaneError> {
    let overflow = match &intent {
        ResponseIntent::Assistant(leaf) => {
            is_context_overflow(&response.message, Some(leaf.context_window))
                || is_recoverable_length(&response.message, leaf.intended_output_limit)
        }
        ResponseIntent::Deferred(_) => false,
    };
    let overflow_preparation = match (&intent, overflow) {
        (ResponseIntent::Assistant(leaf), true)
            if !leaf.generation_context.overflow_recovery_used =>
        {
            prepare_overflow_compaction(lane, drive, leaf).await?
        }
        _ => None,
    };
    let planner_lane = Arc::clone(lane);
    let planner_drive = Arc::clone(drive);
    let planned_intent = intent;
    let planned_response = response;
    lane.settle_operation::<ProcedureResult, _>(
        move |state, current, meta, reader, _context| {
            let intent = planned_intent.clone();
            let response = planned_response.clone();
            let overflow_preparation = overflow_preparation.clone();
            let lane = Arc::clone(&planner_lane);
            let drive = Arc::clone(&planner_drive);
            let meta = meta.clone();
            let current = current.clone();
            Box::pin(async move {
                let leaf = ResponseLeaf::narrow(&current);
                let source = leaf.source();
                let response_entry_id = intent.response_entry_id().to_owned();
                let usage_id = intent.usage_id().to_owned();
                let configuration = leaf.configuration().clone();
                let turn_id = leaf.turn_id();
                let scope = leaf.settlement_scope(&response_entry_id);

                let mut committed = response;
                let mut settled: Option<OperationState> = None;
                let mut failure: Option<OperationError> = None;

                if matches!(leaf.scope().control, Control::CancelRequested { .. }) {
                    committed = normalize_aborted(source, committed);
                    settled = Some(OperationState::Checkpoint(CheckpointOperation {
                        scope: scope.clone(),
                        checkpoint: CheckpointData {
                            continuation: Continuation::MayFinish {
                                include_final_assistant: true,
                            },
                            trigger_entry_id: response_entry_id.clone(),
                        },
                    }));
                } else if committed.stop_reason == SettledStopReason::Aborted {
                    return Err(lane_error(SessionError::Invariant(format!(
                        "{} response is aborted while durable control is running",
                        source.request_name(),
                    ))));
                } else if matches!(leaf, ResponseLeaf::Assistant(_)) && overflow {
                    let ResponseLeaf::Assistant(assistant) = &leaf else {
                        unreachable!("the overflow arm reads the assistant leaf")
                    };
                    let error_message =
                        committed.message.error_message.clone().unwrap_or_else(|| {
                            "Assistant request exceeded the context window".to_owned()
                        });
                    committed = normalize_error(committed, error_message);
                    if assistant.generation_context.overflow_recovery_used
                        || overflow_preparation.is_none()
                    {
                        failure = Some(provider_error(source, &committed));
                    } else {
                        let Some(preparation) = overflow_preparation.as_ref() else {
                            unreachable!("the preparation gate bound Some");
                        };
                        settled = Some(OperationState::SummaryDeciding(SummaryDecidingOperation {
                            scope: scope.clone(),
                            task: SummaryTask {
                                task_id: preparation.task_id.clone(),
                                reason: Some(CompactionReason::Overflow),
                                custom_instructions: None,
                                boundary: ResultBoundary::ResumeCheckpoint {
                                    resume_after: CheckpointData {
                                        continuation: Continuation::NeedAssistant {
                                            overflow_recovery_used: true,
                                        },
                                        trigger_entry_id: assistant
                                            .generation_context
                                            .trigger_entry_id
                                            .clone(),
                                    },
                                },
                            },
                        }));
                    }
                } else if committed.stop_reason == SettledStopReason::Deferred {
                    match leaf {
                        ResponseLeaf::Assistant(assistant) => {
                            if deferred_handle_is_valid(&committed, assistant) {
                                settled = Some(OperationState::DeferredSuspended(
                                    DeferredSuspendedOperation {
                                        deferred: DeferredScope {
                                            scope: scope.clone(),
                                            step_id: assistant.generation_context.step_id.clone(),
                                            source_entry_id: response_entry_id.clone(),
                                            poll: 0,
                                            configuration: configuration.clone(),
                                            stream_options: assistant
                                                .generation_context
                                                .stream_options
                                                .clone(),
                                        },
                                    },
                                ));
                            } else {
                                committed = normalize_error(
                                    committed,
                                    "Provider returned an invalid deferred handle".to_owned(),
                                );
                                failure = Some(provider_error(source, &committed));
                            }
                        }
                        ResponseLeaf::Deferred(poll) => {
                            settled = Some(OperationState::DeferredSuspended(
                                DeferredSuspendedOperation {
                                    deferred: DeferredScope {
                                        scope: scope.clone(),
                                        step_id: poll.scope.step_id.clone(),
                                        source_entry_id: response_entry_id.clone(),
                                        poll: poll.scope.poll,
                                        configuration: configuration.clone(),
                                        stream_options: poll.scope.stream_options.clone(),
                                    },
                                },
                            ));
                        }
                    }
                } else if committed.stop_reason == SettledStopReason::Error {
                    match leaf {
                        ResponseLeaf::Assistant(assistant) => {
                            if (options.recovery
                                || is_retryable_assistant_error(&committed.message))
                                && assistant.attempt
                                    < assistant.generation_context.retry_policy.max_attempts
                            {
                                settled = Some(OperationState::AssistantRetryWait(
                                    AssistantRetryWaitOperation {
                                        scope: scope.clone(),
                                        generation_context: assistant.generation_context.clone(),
                                        retry_wait: RetryWait {
                                            next_attempt: assistant.attempt.saturating_add(1),
                                            not_before: retry_not_before(
                                                &assistant.generation_context.retry_policy,
                                                assistant.attempt,
                                                now_ms(),
                                            ),
                                            error_message: committed
                                                .message
                                                .error_message
                                                .clone()
                                                .unwrap_or_else(|| {
                                                    "Assistant request failed".to_owned()
                                                }),
                                        },
                                    },
                                ));
                            } else {
                                failure = Some(provider_error(source, &committed));
                            }
                        }
                        ResponseLeaf::Deferred(_) => {
                            failure = Some(provider_error(source, &committed));
                        }
                    }
                } else {
                    #[expect(
                        clippy::option_if_let_else,
                        reason = "the closure-free match keeps the invariant arm a match arm instead of a coverage-counted closure"
                    )]
                    let calls: Vec<u64> = committed
                        .message
                        .content
                        .iter()
                        .enumerate()
                        .filter_map(|(source_index, block)| {
                            matches!(block, AssistantBlock::ToolCall(_)).then_some(source_index)
                        })
                        .map(|source_index| match u64::try_from(source_index) {
                            Ok(index) => index,
                            Err(_) => unreachable!("content indices fit u64"),
                        })
                        .collect();
                    if !calls.is_empty() {
                        let timestamp = uuid_v7_timestamp(&response_entry_id)?;
                        let planned = calls
                            .iter()
                            .map(|source_index| {
                                planned_tool_call(
                                    *source_index,
                                    &reader.id_generator().next(Some(timestamp)),
                                )
                            })
                            .collect();
                        settled = Some(OperationState::Tools(ToolsOperation {
                            scope: scope.clone(),
                            batch: ToolBatch {
                                assistant_entry_id: response_entry_id.clone(),
                                configuration: configuration.clone(),
                                turn_id: turn_id.clone(),
                                calls: planned,
                            },
                        }));
                    } else if committed.stop_reason == SettledStopReason::ToolUse {
                        committed = normalize_error(
                            committed,
                            "Provider reported tool use without any tool calls".to_owned(),
                        );
                        failure = Some(provider_error(source, &committed));
                    } else {
                        settled = Some(OperationState::Checkpoint(CheckpointOperation {
                            scope: scope.clone(),
                            checkpoint: CheckpointData {
                                continuation: Continuation::MayFinish {
                                    include_final_assistant: true,
                                },
                                trigger_entry_id: response_entry_id.clone(),
                            },
                        }));
                    }
                }

                let response_entry = NewEntry::Message {
                    id: response_entry_id.clone(),
                    parent_id: state.tip_id.clone(),
                    body: Box::new(MessageEntry {
                        message: AgentMessage::Standard(Message::Assistant(
                            committed.message.clone(),
                        )),
                        terminate: None,
                    }),
                };
                let usage_row = UsageWriteRow {
                    id: usage_id,
                    usage: committed.message.usage,
                    entry_id: Some(response_entry_id.clone()),
                    adjustment: false,
                    details: None,
                };
                if settled.is_none() && failure.is_none() {
                    return Err(lane_error(SessionError::Invariant(
                        "Response settlement has no durable disposition".to_owned(),
                    )));
                }
                let record = failure
                    .as_ref()
                    .map(|failure| {
                        operation_result_record(
                            &meta,
                            TerminalStatus::Failed,
                            Some(response_entry_id.clone()),
                            Some(failure.clone()),
                        )
                    })
                    .transpose()?;
                let cleanup = match &record {
                    Some(_) => {
                        operation_cleanup_writes(
                            reader.as_ref(),
                            &drive.operation_id,
                            &current,
                            &drive.context,
                        )
                        .await?
                    }
                    None => Vec::new(),
                };
                let mut writes = vec![
                    Write::Entry(Box::new(insert_entry(response_entry.clone()))),
                    Write::Usage(insert_usage(usage_row.clone())),
                    set_value_write(&branch_tip(lane.name()), Some(response_entry_id.clone()))
                        .map_err(lane_error)?,
                ];
                if record.is_none() {
                    writes.push(Write::ListDelete(delete_list(&pending_assistant_frames(
                        &drive.operation_id,
                        &response_entry_id,
                    ))));
                } else {
                    writes.extend(cleanup);
                }
                if settled
                    .as_ref()
                    .is_some_and(|settled| settled.at() == "summary.deciding")
                    && let Some(preparation) = &overflow_preparation
                {
                    writes.push(
                        set_value_write(
                            &operation_preparation(&drive.operation_id, &preparation.task_id),
                            preparation.preparation.clone(),
                        )
                        .map_err(lane_error)?,
                    );
                }

                let lane_name = lane.name().to_owned();
                let run_id = drive.operation_id.clone();
                let from_tip_id = meta.source_tip_id.clone();
                let (is_assistant, is_deferred_effect, attempt) = match &leaf {
                    ResponseLeaf::Assistant(leaf) => (true, false, leaf.attempt),
                    ResponseLeaf::Deferred(_) => (false, true, 0),
                };
                let delay_ms = match &leaf {
                    ResponseLeaf::Assistant(leaf) => {
                        policy_delay_ms(&leaf.generation_context.retry_policy, leaf.attempt)
                    }
                    ResponseLeaf::Deferred(_) => 0,
                };
                let events: EventsFn = Arc::new({
                    let recovery = options.recovery;
                    let committed = committed.clone();
                    let settled = settled.clone();
                    let record = record.clone();
                    let failure = failure.clone();
                    let response_entry = response_entry.clone();
                    let usage_row = usage_row.clone();
                    let response_entry_id = response_entry_id.clone();
                    move |commit: &CommitResult| {
                        let seq = commit.seqs.first().map_or_else(
                            || unreachable!("commit carries one sequence per write"),
                            |seq| *seq,
                        );
                        let entry = response_entry.clone().materialize(seq, commit.timestamp);
                        #[expect(
                            clippy::option_if_let_else,
                            reason = "the closure-free match keeps the invariant arm a match arm instead of a coverage-counted closure"
                        )]
                        let usage_seq = match commit.seqs.get(1) {
                            Some(seq) => *seq,
                            None => unreachable!("commit carries one sequence per write"),
                        };
                        let mut batch = vec![
                            lane_scoped_event(
                                &lane_name,
                                recovery,
                                "entry_added",
                                HarnessEventPayload::EntryAdded { entry },
                            ),
                            global_event(
                                "usage",
                                HarnessEventPayload::Usage {
                                    lane: lane_name.clone(),
                                    row: UsageRow {
                                        id: usage_row.id.clone(),
                                        seq: usage_seq,
                                        usage: usage_row.usage,
                                        entry_id: usage_row.entry_id.clone(),
                                        adjustment: usage_row.adjustment,
                                        details: usage_row.details.clone(),
                                    },
                                    totals: commit.stats.usage,
                                },
                            ),
                        ];
                        let settled_at = settled.as_ref().map(OperationState::at);
                        if is_assistant {
                            if !recovery
                                && attempt > 1
                                && settled_at != Some("assistant.retry_wait")
                            {
                                let success = !matches!(
                                    committed.stop_reason,
                                    SettledStopReason::Error | SettledStopReason::Aborted
                                );
                                let final_error = (!success).then(|| {
                                    committed.message.error_message.clone().unwrap_or_else(|| {
                                        format!(
                                            "Assistant request ended with {}",
                                            stop_reason_wire(committed.stop_reason)
                                        )
                                    })
                                });
                                batch.push(lane_scoped_event(
                                    &lane_name,
                                    false,
                                    "retry_end",
                                    HarnessEventPayload::RetryEnd {
                                        run_id: run_id.clone(),
                                        step: turn_id.clone(),
                                        attempt,
                                        success,
                                        final_error,
                                    },
                                ));
                            }
                            if !recovery && settled_at == Some("assistant.retry_wait") {
                                let Some(OperationState::AssistantRetryWait(retry)) = &settled
                                else {
                                    unreachable!("the scheduled event reads the retry leaf")
                                };
                                batch.push(lane_scoped_event(
                                    &lane_name,
                                    false,
                                    "retry_scheduled",
                                    HarnessEventPayload::RetryScheduled {
                                        run_id: run_id.clone(),
                                        step: turn_id.clone(),
                                        attempt: retry.retry_wait.next_attempt,
                                        max_attempts: retry
                                            .generation_context
                                            .retry_policy
                                            .max_attempts,
                                        delay_ms,
                                        not_before: retry.retry_wait.not_before,
                                        error_message: retry.retry_wait.error_message.clone(),
                                    },
                                ));
                            }
                            if !recovery
                                && settled_at != Some("tools")
                                && settled_at != Some("assistant.retry_wait")
                            {
                                batch.push(lane_scoped_event(
                                    &lane_name,
                                    false,
                                    "turn_end",
                                    HarnessEventPayload::TurnEnd {
                                        run_id: run_id.clone(),
                                        turn_id: turn_id.clone(),
                                        message: committed.message.clone(),
                                        tool_results: Vec::new(),
                                    },
                                ));
                            }
                            if settled_at == Some("summary.deciding") {
                                batch.push(lane_scoped_event(
                                    &lane_name,
                                    false,
                                    "compaction_start",
                                    HarnessEventPayload::CompactionStart {
                                        run_id: run_id.clone(),
                                        reason: CompactionReason::Overflow,
                                        started_at: commit.timestamp,
                                    },
                                ));
                            }
                        } else if settled_at != Some("tools") {
                            batch.push(lane_scoped_event(
                                &lane_name,
                                recovery,
                                "turn_end",
                                HarnessEventPayload::TurnEnd {
                                    run_id: run_id.clone(),
                                    turn_id: turn_id.clone(),
                                    message: committed.message.clone(),
                                    tool_results: Vec::new(),
                                },
                            ));
                        }
                        if (is_deferred_effect || !recovery)
                            && settled_at == Some("deferred.suspended")
                            && let Some(deferred) = committed.message.deferred.clone()
                        {
                            let Some(OperationState::DeferredSuspended(suspended)) = &settled
                            else {
                                unreachable!("the suspend event reads the deferred leaf")
                            };
                            batch.push(lane_scoped_event(
                                &lane_name,
                                is_deferred_effect && recovery,
                                "run_suspend",
                                HarnessEventPayload::RunSuspend {
                                    run_id: run_id.clone(),
                                    deferred,
                                    poll: suspended.deferred.poll,
                                },
                            ));
                        }
                        if let (Some(record), Some(failure)) = (&record, &failure) {
                            batch.push(lane_scoped_event(
                                &lane_name,
                                false,
                                "run_end",
                                HarnessEventPayload::RunEnd {
                                    run_id: run_id.clone(),
                                    from_tip_id: from_tip_id.clone(),
                                    tip_id: Some(response_entry_id.clone()),
                                    ended_at: record.ended_at,
                                    status: RunEndStatus::Failed {
                                        error: failure.clone(),
                                    },
                                },
                            ));
                        }
                        batch
                    }
                });
                if let Some(record) = record {
                    let materialize_record = record.clone();
                    return Ok(OperationCommand::Finish {
                        writes,
                        record,
                        lane: Some(LanePatch {
                            tip_id: Some(Some(response_entry_id)),
                            ..LanePatch::default()
                        }),
                        materialize: Arc::new(move |_: &CommitResult| ProcedureResult::Settled {
                            outcome: materialize_record.clone(),
                        }),
                        events: Some(events),
                    });
                }
                let Some(settled) = settled else {
                    return Err(lane_error(SessionError::Invariant(
                        "Response settlement is missing its next state".to_owned(),
                    )));
                };
                Ok(OperationCommand::Commit {
                    writes,
                    operation_state: settled,
                    lane: Some(LanePatch {
                        tip_id: Some(Some(response_entry_id)),
                        ..LanePatch::default()
                    }),
                    materialize: Arc::new(|_: &CommitResult| ProcedureResult::Continue),
                    events: Some(events),
                })
            })
        },
        &drive.context,
    )
    .await
}

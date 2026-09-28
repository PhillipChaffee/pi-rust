//! The deferred-poll procedure, ported from upstream
//! `src/harness/runtime/drive/deferred.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! `run_deferred` advances one deferred leaf: the suspended leaf polls
//! its suspended handle when the pass carries a permit — one intent
//! commit, one gated long-poll, one settlement through the response
//! module — and reports the durable wait without one; the effect-pending
//! leaf replaces its orphaned poll the same way under fresh ids at the
//! same poll number, deleting the old poll's frames with the intent
//! commit. `read_deferred_source_handle` validates the source assistant
//! entry's deferred handle under the reserved ids, and the reconcile
//! procedure consumes it.
//!
//! Porting restatements: the permit counter is the pass's atomic
//! [`Drive::deferred_permits`], consumed in the intent commit's
//! materialization (upstream's `materialize` decrement), so a parked or
//! discarded commit leaves the permit unspent. The poll's provider options
//! ride pi-ai's `DeferredFetchOptions` — `wait: 0`, the gate-linked
//! cancellation token, the `before_payload` hook, and the captured
//! response metadata — and the after-response binding late-reads the
//! capture cell where upstream reassigns its `metadata` local.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::sync::atomic::Ordering;

use pi_ai::models::WithTransforms;
use pi_ai::types::DeferredFetchOptions;
use pi_ai::types::DeferredHandle;
use pi_ai::types::DeferredRequest;
use pi_ai::types::Message;
use pi_ai::types::Model;
use pi_ai::types::OnPayload;
use pi_ai::types::OnResponse;
use pi_ai::types::ProviderResponse;
use pi_ai::types::StopReason;
use pi_ai::types::TransportOptions;

use pi_chord::context::with_abort_signal;

use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::DriveWaitReason;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HookEvent;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::agent_harness::HookResult;
use crate::harness::agent_harness::StepKind;
use crate::harness::agent_harness::lane_scoped_event;
use crate::harness::context::Context;
use crate::harness::context::get_telemetry_context;
use crate::harness::context::request_signal;
use crate::harness::execution::assistant::AfterResponseCallback;
use crate::harness::execution::assistant::AssistantResponseMetadata;
use crate::harness::execution::assistant::consume_assistant_stream;
use crate::harness::gate::AbortRequested;
use crate::harness::gate::GateRejection;
use crate::harness::hooks::HookRunError;
use crate::harness::hooks::apply_stream_options_patch;
use crate::harness::runtime::clock::now_ms;
use crate::harness::runtime::drive::response::ResponseIntent;
use crate::harness::runtime::drive::response::ResponseOptions;
use crate::harness::runtime::drive::response::open_assistant_response;
use crate::harness::runtime::drive::response::publish_configuration_failure;
use crate::harness::runtime::drive::response::publish_response;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::EventsFn;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::runtime::types::ProcedureResult;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::DeferredEffectPendingOperation;
use crate::harness::session::types::DeferredScope;
use crate::harness::session::types::DeferredSuspendedOperation;
use crate::harness::session::types::Entry;
use crate::harness::session::types::ModelIdentity;
use crate::harness::session::types::OperationError;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::SettledAssistantMessage;
use crate::harness::session::types::operation_scope_of;
use crate::harness::session::values::Write;
use crate::harness::session::values::delete_list;
use crate::harness::session::values::pending_assistant_frames;
use crate::harness::types::AgentHarnessStreamOptions;
use serde_json::Value as JsonValue;

/// The two leaves the dispatcher routes to the deferred procedure,
/// upstream's `DeferredLeaf` (`DeferredSuspendedOperation |
/// DeferredEffectPendingOperation`).
#[derive(Clone, Debug)]
pub(crate) enum DeferredLeaf {
    /// The `deferred.suspended` leaf.
    Suspended(DeferredSuspendedOperation),
    /// The `deferred.effect_pending` leaf.
    EffectPending(DeferredEffectPendingOperation),
}

impl DeferredLeaf {
    /// The leaf's deferred-step scope, upstream's leaf field reads.
    const fn deferred_scope(&self) -> &DeferredScope {
        match self {
            Self::Suspended(leaf) => &leaf.deferred,
            Self::EffectPending(leaf) => &leaf.scope,
        }
    }

    /// The leaf restated as the durable state, the capability
    /// [`publish_configuration_failure`] receives.
    fn operation_state(&self) -> OperationState {
        match self {
            Self::Suspended(leaf) => OperationState::DeferredSuspended(leaf.clone()),
            Self::EffectPending(leaf) => OperationState::DeferredEffectPending(leaf.clone()),
        }
    }
}

/// The ready poll one preparation builds, upstream's
/// `PreparedDeferredPoll`: the validated handle, the resolved model, the
/// poll number this request advances to, and the poll-local stream options
/// with the deferred flag forced off.
struct PreparedDeferredPoll {
    /// The source handle the poll fetches, upstream's `source`.
    source: DeferredHandle,
    /// The resolved model the poll streams from, upstream's `model`.
    model: Model,
    /// The poll number this request advances to, upstream's `poll`.
    poll: u64,
    /// The patched poll options, upstream's `streamOptions`.
    stream_options: AgentHarnessStreamOptions,
}

/// What the poll preparation produced, upstream's `DeferredPreparation`.
#[expect(
    clippy::large_enum_variant,
    reason = "mirrors upstream's discriminated union carrying the ready poll's model and options whole"
)]
enum DeferredPreparation {
    /// The poll may run, upstream's `ready` arm.
    Ready(PreparedDeferredPoll),
    /// Cancellation won the source-handle read, upstream's
    /// `cancel_requested`.
    CancelRequested,
    /// The pass carries no permit, upstream's `waiting` arm.
    Waiting {
        /// The handle the pass waits on.
        source: DeferredHandle,
    },
    /// The captured model is unavailable, upstream's
    /// `configuration_failure`.
    ConfigurationFailure,
}

/// Builds one missing-model failure, upstream's `configurationError`: the
/// identity itself rides the details.
fn configuration_error(identity: &ModelIdentity) -> OperationError {
    OperationError {
        code: "model_unavailable".to_owned(),
        message: "The configured model is unavailable in this process".to_owned(),
        details: Some(serde_json::to_value(identity).unwrap_or_default()),
    }
}

/// Validates the source assistant entry's deferred handle under the
/// reserved ids, upstream's `readDeferredSourceHandle`.
///
/// # Errors
/// The invariants `` `Deferred source {entryId} is missing its assistant
/// handle` `` — a non-message entry, a non-assistant role, a non-deferred
/// stop, or a missing handle — and `` `Deferred source {entryId} has an
/// invalid handle` `` (empty id, or provider/model id/api mismatching the
/// leaf's configuration and the entry's wire API), plus the underlying
/// read's storage errors.
pub(crate) async fn read_deferred_source_handle(
    reader: &dyn SessionReader,
    deferred: &DeferredLeaf,
    context: &Context,
) -> Result<DeferredHandle, LaneError> {
    let scope = deferred.deferred_scope();
    let entries = reader
        .get_entries(vec![scope.source_entry_id.clone()], context)
        .await
        .map_err(lane_error)?;
    let message = entries.get(&scope.source_entry_id).and_then(|source| {
        let Entry::Message { body, .. } = source else {
            return None;
        };
        let crate::types::AgentMessage::Standard(Message::Assistant(message)) = &body.message
        else {
            return None;
        };
        (message.stop_reason == StopReason::Deferred && message.deferred.is_some())
            .then_some(message)
    });
    let Some(message) = message else {
        return Err(lane_error(SessionError::Invariant(format!(
            "Deferred source {} is missing its assistant handle",
            scope.source_entry_id
        ))));
    };
    let Some(handle) = message.deferred.as_ref() else {
        unreachable!("the deferred filter bound Some");
    };
    let identity = &scope.configuration.model;
    if handle.id.is_empty()
        || handle.provider != identity.provider
        || handle.model_id != identity.model_id
        || handle.api.as_str() != message.api.0.as_str()
    {
        return Err(lane_error(SessionError::Invariant(format!(
            "Deferred source {} has an invalid handle",
            scope.source_entry_id
        ))));
    }
    Ok(handle.clone())
}

/// Reads the source handle through the lane's mutation line, upstream's
/// `readSourceHandle` — a pure read that still honours the
/// cancel-refused contract.
///
/// # Errors
/// [`read_deferred_source_handle`]'s invariants and storage errors.
async fn read_source_handle(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    deferred: &DeferredLeaf,
) -> Result<ContinueOperationResult<DeferredHandle>, LaneError> {
    let planner_drive = Arc::clone(drive);
    let planner_deferred = deferred.clone();
    lane.continue_operation::<DeferredHandle, _>(
        move |_state, reader, _context| {
            let drive = Arc::clone(&planner_drive);
            let deferred = planner_deferred.clone();
            Box::pin(async move {
                let handle =
                    read_deferred_source_handle(reader.as_ref(), &deferred, &drive.context).await?;
                Ok(OperationCommand::Return { result: handle })
            })
        },
        &drive.context,
    )
    .await
}

/// Resolves one leaf into the poll preparation, upstream's
/// `prepareDeferredPoll`: the validated source handle, the permit gate,
/// the model resolution, and the `before_request` hook's patch over the
/// deferred-disabled options.
///
/// # Errors
/// [`read_source_handle`]'s errors, and the hook run's failures outside
/// the abort path — the abort carries the spine-recognized
/// [`AbortRequested`].
async fn prepare_deferred_poll(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    expected: &DeferredLeaf,
) -> Result<DeferredPreparation, LaneError> {
    let source = read_source_handle(lane, drive, expected).await?;
    let source = match source {
        ContinueOperationResult::CancelRequested => {
            return Ok(DeferredPreparation::CancelRequested);
        }
        ContinueOperationResult::Result { value } => value,
    };
    if drive.deferred_permits.load(Ordering::SeqCst) == 0 {
        return Ok(DeferredPreparation::Waiting { source });
    }

    let identity = &expected.deferred_scope().configuration.model;
    let Some(model) = lane.models().model(&identity.provider, &identity.model_id) else {
        return Ok(DeferredPreparation::ConfigurationFailure);
    };
    let base_options = AgentHarnessStreamOptions {
        deferred: Some(DeferredRequest::Enabled(false)),
        ..expected.deferred_scope().stream_options.clone()
    };
    let poll = match expected {
        DeferredLeaf::Suspended(leaf) => leaf.deferred.poll.saturating_add(1),
        DeferredLeaf::EffectPending(leaf) => leaf.scope.poll,
    };
    let hook = lane
        .hooks()
        .run_with_gate(
            HookName::BeforeRequest,
            HookInvocation {
                lane: lane.name().to_owned(),
                run_id: drive.operation_id.clone(),
                event: HookEvent::BeforeRequest {
                    model: model.clone(),
                    step: StepKind::Deferred,
                    // The hook event's attempt field is u32 where the
                    // durable poll counter is u64; only this wire event
                    // narrows it.
                    attempt: u32::try_from(poll).unwrap_or(u32::MAX),
                    stream_options: base_options.clone(),
                },
            },
            &drive.gate,
            &drive.context,
        )
        .await;
    let hook = match hook {
        Ok(hook) => hook,
        Err(HookRunError::GateAborted(abort)) => return Err(lane_error(abort)),
        Err(HookRunError::Aborted(_)) => {
            return Err(lane_error(AbortRequested {
                cancellation: drive.abort_cancellation(),
            }));
        }
        Err(error) => return Err(lane_error(error)),
    };
    let HookResult::BeforeRequest(patch) = hook else {
        unreachable!("before_request returns its own result variant")
    };
    let patched = patch.map_or_else(
        || base_options.clone(),
        |patch| apply_stream_options_patch(&base_options, &patch.stream_options),
    );
    let mut stream_options = patched;
    stream_options.deferred = Some(DeferredRequest::Enabled(false));
    Ok(DeferredPreparation::Ready(PreparedDeferredPoll {
        source,
        model,
        poll,
        stream_options,
    }))
}

/// Commits the fresh poll intent and consumes the pass's permit in its
/// materialization, upstream's `publishPollIntent`: both reserved ids mint
/// from one timestamp seed, the effect-pending replacement deletes the old
/// poll's frames, and the `run_resume`/`turn_start` pair publishes with
/// the poll-local turn id.
///
/// # Errors
/// The commit's storage and delivery errors.
async fn publish_poll_intent(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    deferred: &DeferredLeaf,
    prepared: &PreparedDeferredPoll,
    recovery: bool,
) -> Result<ContinueOperationResult<DeferredEffectPendingOperation>, LaneError> {
    let at = now_ms();
    let response_entry_id = lane.session().id_generator().next(Some(at));
    let usage_id = lane.session().id_generator().next(Some(at));
    let writes = match deferred {
        DeferredLeaf::EffectPending(leaf) => vec![Write::ListDelete(delete_list(
            &pending_assistant_frames(&drive.operation_id, &leaf.response_entry_id),
        ))],
        DeferredLeaf::Suspended(_) => Vec::new(),
    };
    let step_id = deferred.deferred_scope().step_id.clone();
    let source_entry_id = deferred.deferred_scope().source_entry_id.clone();
    let configuration = deferred.deferred_scope().configuration.clone();
    let stream_options = deferred.deferred_scope().stream_options.clone();
    let poll = prepared.poll;
    let lane_name = lane.name().to_owned();
    let run_id = drive.operation_id.clone();
    let turn_id = format!("{step_id}:poll:{poll}");
    let events: EventsFn = Arc::new(move |_commit: &CommitResult| {
        vec![
            lane_scoped_event(
                &lane_name,
                recovery,
                "run_resume",
                HarnessEventPayload::RunResume {
                    run_id: run_id.clone(),
                },
            ),
            lane_scoped_event(
                &lane_name,
                recovery,
                "turn_start",
                HarnessEventPayload::TurnStart {
                    run_id: run_id.clone(),
                    turn_id: turn_id.clone(),
                },
            ),
        ]
    });
    let planner_drive = Arc::clone(drive);
    lane.continue_operation::<DeferredEffectPendingOperation, _>(
        move |state, _session, _context| {
            let writes = writes.clone();
            let response_entry_id = response_entry_id.clone();
            let usage_id = usage_id.clone();
            let step_id = step_id.clone();
            let source_entry_id = source_entry_id.clone();
            let configuration = configuration.clone();
            let stream_options = stream_options.clone();
            let drive = Arc::clone(&planner_drive);
            let events = Arc::clone(&events);
            Box::pin(async move {
                let Some(current) = state.operation.as_ref() else {
                    unreachable!("continueOperation planner runs under an active operation")
                };
                let scope = operation_scope_of(&current.state);
                let next = DeferredEffectPendingOperation {
                    scope: DeferredScope {
                        scope,
                        step_id,
                        source_entry_id,
                        poll,
                        configuration,
                        stream_options,
                    },
                    response_entry_id,
                    usage_id,
                };
                Ok(OperationCommand::Commit {
                    writes,
                    operation_state: OperationState::DeferredEffectPending(next.clone()),
                    lane: None,
                    materialize: Arc::new(move |_: &CommitResult| {
                        // The permit spends when the fresh intent commit
                        // lands and materializes — a parked or discarded
                        // commit leaves it unspent, upstream's
                        // `materialize` decrement.
                        drive.deferred_permits.fetch_sub(1, Ordering::SeqCst);
                        next.clone()
                    }),
                    events: Some(events),
                })
            })
        },
        &drive.context,
    )
    .await
}

/// Streams one deferred long-poll through the response lifecycle, upstream's
/// `performDeferredPoll`: the gate admits the lazy stream, the options
/// carry the one-status-check `wait: 0` with the gate-linked signal and
/// the `before_payload` hook, the response hook captures the wire
/// metadata, and the close always runs after the stream settles.
///
/// # Errors
/// The admission refusal (the abort rides the spine's catch), the stream's
/// protocol and hook failures, and the close's write or held-stream
/// failure — which replaces the stream's own, upstream's `finally` throw.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single performDeferredPoll method and its option assembly"
)]
async fn perform_deferred_poll(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    prepared: &PreparedDeferredPoll,
    intent: &DeferredEffectPendingOperation,
    recovery: bool,
) -> Result<SettledAssistantMessage, LaneError> {
    let response = open_assistant_response(lane, drive, &intent.response_entry_id, recovery);
    let metadata = Arc::new(Mutex::new(AssistantResponseMetadata::default()));
    let admitted_context = with_abort_signal(drive.gate.signal().clone(), &drive.context);
    let on_payload = {
        let lane = Arc::clone(lane);
        let drive = Arc::clone(drive);
        // The hook sees the drive's signal-stripped context, upstream's
        // `drive.context` — the stream's own options keep the gate-linked
        // signal below.
        let hook_context = drive.context.clone();
        OnPayload::new(move |payload: JsonValue, request_model: Model| {
            let lane = Arc::clone(&lane);
            let drive = Arc::clone(&drive);
            let hook_context = hook_context.clone();
            Box::pin(async move {
                match lane
                    .hooks()
                    .run_with_gate(
                        HookName::BeforePayload,
                        HookInvocation {
                            lane: lane.name().to_owned(),
                            run_id: drive.operation_id.clone(),
                            event: HookEvent::BeforePayload {
                                model: request_model,
                                payload,
                            },
                        },
                        &drive.gate,
                        &hook_context,
                    )
                    .await
                {
                    Ok(HookResult::BeforePayload(result)) => result.map(|result| result.payload),
                    Ok(_) => unreachable!("before_payload returns its own result variant"),
                    // The OnPayload seam carries no error channel; the
                    // request's linked cancellation token aborts it when the
                    // signal aborted, upstream's onPayload throw reaching the
                    // same settlement.
                    Err(_) => None,
                }
            })
        })
    };
    let on_response = {
        let cell = Arc::clone(&metadata);
        OnResponse::new(move |response: ProviderResponse, _model: Model| {
            *cell.lock().unwrap_or_else(PoisonError::into_inner) = AssistantResponseMetadata {
                status: Some(response.status),
                headers: Some(response.headers),
            };
            Box::pin(async {})
        })
    };
    let options = WithTransforms {
        options: DeferredFetchOptions {
            transport_options: TransportOptions {
                http_client: None,
                signal: request_signal(&admitted_context),
                on_payload: Some(on_payload),
                on_response: Some(on_response),
            },
            telemetry_context: Some(get_telemetry_context(&admitted_context)),
            headers: prepared.stream_options.headers.clone().map(|headers| {
                headers
                    .into_iter()
                    .map(|(key, value)| (key, Some(value)))
                    .collect()
            }),
            timeout_ms: prepared.stream_options.timeout_ms,
            max_retries: prepared.stream_options.max_retries,
            max_retry_delay_ms: prepared.stream_options.max_retry_delay_ms,
            wait: Some(0),
            ..DeferredFetchOptions::default()
        },
        transform_headers: None,
    };
    let stream = {
        let models = Arc::clone(lane.models());
        let model = prepared.model.clone();
        let source = prepared.source.clone();
        match drive
            .gate
            .admit(move || models.stream_deferred(&model, &source, Some(&options)))
        {
            Ok(stream) => stream,
            Err(GateRejection::AbortRequested(abort)) => return Err(lane_error(abort)),
            Err(GateRejection::Closed(error)) => return Err(lane_error(error)),
        }
    };
    let after_response = response.after_response.clone();
    let cell = Arc::clone(&metadata);
    let after_response: AfterResponseCallback = Arc::new(move |message, context| {
        let metadata = cell.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let hook = Arc::clone(&after_response);
        Box::pin(async move { hook(message, metadata, context).await })
    });

    let settled = consume_assistant_stream(
        &stream,
        response.observer.as_ref(),
        Some(after_response),
        &drive.context,
    )
    .await;
    let closed = ((response.close)()).await;
    match closed {
        Err(error) => Err(error),
        Ok(()) => settled.map_err(lane_error),
    }
}

/// Prepares, commits, streams, and settles one deferred poll, upstream's
/// `pollDeferred`: the cancelled and no-permit preparations report without
/// touching the state, the configuration failure publishes before ids are
/// reserved, and the ready preparation runs the gated poll through the
/// response module's classifier.
///
/// # Errors
/// [`prepare_deferred_poll`]'s, [`publish_poll_intent`]'s,
/// [`perform_deferred_poll`]'s, [`publish_configuration_failure`]'s, and
/// [`publish_response`]'s errors.
async fn poll_deferred(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    expected: &DeferredLeaf,
    recovery: bool,
) -> Result<ProcedureResult, LaneError> {
    let prepared = prepare_deferred_poll(lane, drive, expected).await?;
    match prepared {
        DeferredPreparation::CancelRequested => Ok(ProcedureResult::Continue),
        DeferredPreparation::Waiting { source } => Ok(ProcedureResult::Waiting {
            outcome: DriveOutcome::Waiting {
                operation_id: drive.operation_id.clone(),
                reason: DriveWaitReason::Deferred { deferred: source },
            },
        }),
        DeferredPreparation::ConfigurationFailure => {
            publish_configuration_failure(
                lane,
                drive,
                &expected.operation_state(),
                configuration_error(&expected.deferred_scope().configuration.model),
            )
            .await
        }
        DeferredPreparation::Ready(prepared) => {
            let intent = publish_poll_intent(lane, drive, expected, &prepared, recovery).await?;
            let ContinueOperationResult::Result { value: intent } = intent else {
                return Ok(ProcedureResult::Continue);
            };
            let response = perform_deferred_poll(lane, drive, &prepared, &intent, recovery).await?;
            publish_response(
                lane,
                drive,
                ResponseIntent::Deferred(intent),
                response,
                ResponseOptions { recovery },
            )
            .await
        }
    }
}

/// Poll one durably suspended deferred response when this pass carries a
/// permit, upstream's `runDeferredSuspended`: without a permit the pass
/// reports the durable wait on the suspended handle, with one the poll
/// streams from the captured options and settles through the response
/// module.
///
/// # Errors
/// [`poll_deferred`]'s errors.
pub(crate) async fn run_deferred_suspended(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    deferred: DeferredSuspendedOperation,
) -> Result<ProcedureResult, LaneError> {
    poll_deferred(lane, drive, &DeferredLeaf::Suspended(deferred), false).await
}

/// Replace one orphaned unknown-outcome poll under fresh ids at the same
/// poll number when this pass carries a permit, upstream's
/// `recoverDeferredPoll`: the old frames delete with the fresh intent
/// commit, and the settlement runs in recovery mode.
///
/// # Errors
/// [`poll_deferred`]'s errors.
pub(crate) async fn recover_deferred_poll(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    deferred: DeferredEffectPendingOperation,
) -> Result<ProcedureResult, LaneError> {
    poll_deferred(lane, drive, &DeferredLeaf::EffectPending(deferred), true).await
}

/// Advance one deferred suspended or effect-pending leaf, upstream's
/// `runDeferred`.
///
/// # Errors
/// [`poll_deferred`]'s errors — the source-handle invariants
/// `` `Deferred source {entryId} is missing its assistant handle` `` and
/// `` `Deferred source {entryId} has an invalid handle` `` among them.
pub(crate) async fn run_deferred(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    deferred: DeferredLeaf,
) -> Result<ProcedureResult, LaneError> {
    match deferred {
        DeferredLeaf::Suspended(leaf) => run_deferred_suspended(lane, drive, leaf).await,
        DeferredLeaf::EffectPending(leaf) => recover_deferred_poll(lane, drive, leaf).await,
    }
}

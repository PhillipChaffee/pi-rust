//! The assistant generation procedure, ported from upstream
//! `src/harness/runtime/drive/generation.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! `run_generation` advances one ready or retry-wait leaf: the ready leaf
//! resolves the configured model and tools, publishes the durable generation
//! intent — whose response and usage ids mint from one timestamp seed —
//! streams the response through the pass's gate, and settles it through the
//! response module; the retry-wait leaf waits out or returns its durable
//! wait, then re-commits the ready leaf.
//!
//! Porting restatement: the request hook returns a stream with no error
//! channel, so an admission refusal cannot cross it the way upstream's
//! throw does. The refusal records into a shared cell and the hook ends the
//! stream with an aborted settlement; only a cancellation landing between
//! the transform hook's own admission and the request's reaches this arm
//! (upstream's window between the two `admit` checks). The generation
//! discards the settlement and raises the recorded rejection — upstream's
//! throw propagating to the spine's abort catch — so the durable outcome is
//! the recovery reconciliation either way.

use super::gate_rejection_error;
use super::hook_error_box;
use super::hook_error_to_lane_error;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;

use pi_ai::models::WithTransforms;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Context as AiContext;
use pi_ai::types::Model;
use pi_ai::types::StopReason;
use pi_ai::types::Tool;
use pi_ai::utils::event_stream::assistant_message_event_stream;

use crate::harness::agent_harness::DriveOutcome;
use crate::harness::agent_harness::DriveWaitReason;
use crate::harness::agent_harness::HarnessEventPayload;
use crate::harness::agent_harness::HookEvent;
use crate::harness::agent_harness::HookInvocation;
use crate::harness::agent_harness::HookName;
use crate::harness::agent_harness::HookResult;
use crate::harness::agent_harness::StepKind;
use crate::harness::agent_harness::SystemPromptSource;
use crate::harness::agent_harness::ToProviderMessages;
use crate::harness::agent_harness::lane_scoped_event;
use crate::harness::context::Context;
use crate::harness::context::get_telemetry_context;
use crate::harness::context::request_signal;
use crate::harness::context::with_abort_signal;
use crate::harness::execution::assistant::AssistantRequestContext;
use crate::harness::execution::assistant::AssistantRequestHook;
use crate::harness::execution::assistant::AssistantStreamError;
use crate::harness::execution::assistant::BeforePayloadHook;
use crate::harness::execution::assistant::HarnessAssistantStreamConfig;
use crate::harness::execution::assistant::TransformContextHook;
use crate::harness::execution::assistant::stream_harness_assistant;
use crate::harness::gate::AbortRequested;
use crate::harness::hooks::HookRunError;
use crate::harness::hooks::apply_stream_options_patch;
use crate::harness::runtime::clock::now_ms;
use crate::harness::runtime::drive::response::ResponseIntent;
use crate::harness::runtime::drive::response::ResponseOptions;
use crate::harness::runtime::drive::response::open_assistant_response;
use crate::harness::runtime::drive::response::publish_configuration_failure;
use crate::harness::runtime::drive::response::publish_response;
use crate::harness::runtime::drive::retry::wait_until;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::transcript::read_bounded_context;
use crate::harness::runtime::types::ContinueOperationResult;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::OperationCommand;
use crate::harness::runtime::types::ProcedureResult;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::types::AssistantEffectPendingOperation;
use crate::harness::session::types::AssistantReadyOperation;
use crate::harness::session::types::AssistantRetryWaitOperation;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::OperationError;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SettledAssistantMessage;
use crate::harness::session::types::operation_scope_of;
use crate::harness::types::AgentHarnessStreamOptions;
use crate::harness::types::AgentHarnessTool;
use crate::harness::types::AgentHarnessToolContextSource;
use crate::types::AgentMessage;
use serde_json::Value as JsonValue;

/// The two leaves the dispatcher routes to the generation procedure,
/// upstream's `AssistantReadyOperation | AssistantRetryWaitOperation`
/// union parameter.
#[derive(Clone, Debug)]
pub(crate) enum GenerationLeaf {
    /// The `assistant.ready` leaf.
    Ready(AssistantReadyOperation),
    /// The `assistant.retry_wait` leaf.
    RetryWait(AssistantRetryWaitOperation),
}

/// The resolved request inputs one ready attempt streams with, upstream's
/// `PreparedGeneration`.
struct PreparedGeneration {
    /// The resolved model, upstream's `model`.
    model: Model,
    /// The provider tool surface built from the active tool names, upstream's
    /// `tools`.
    tools: Vec<Tool>,
    /// The bounded transcript the request reads, upstream's `messages`.
    messages: Vec<AgentMessage>,
    /// The resolved system prompt, upstream's `systemPrompt`.
    system_prompt: String,
    /// The stream options after the `before_request` patch, upstream's
    /// `streamOptions`.
    stream_options: AgentHarnessStreamOptions,
    /// The transcript-to-provider conversion, upstream's
    /// `toProviderMessages`.
    to_provider_messages: ToProviderMessages,
}

/// What one ready attempt resolved to, upstream's `GenerationPreparation`.
#[expect(
    clippy::large_enum_variant,
    reason = "mirrors upstream's discriminated union carrying the prepared request whole; boxing it would obscure the three-way match"
)]
enum GenerationPreparation {
    /// The resolved request, upstream's `kind: "ready"`.
    Ready(PreparedGeneration),
    /// The configuration cannot serve the request, upstream's
    /// `kind: "configuration_failure"`.
    ConfigurationFailure(OperationError),
    /// Cancellation won before preparation finished, upstream's
    /// `kind: "cancel_requested"`.
    CancelRequested,
}

/// The unavailable-configuration codes, upstream's `configurationError`'s
/// `"model_unavailable" | "configured_tools_unavailable"` parameter.
const MODEL_UNAVAILABLE: &str = "model_unavailable";
const CONFIGURED_TOOLS_UNAVAILABLE: &str = "configured_tools_unavailable";

/// Builds one configuration failure, upstream's `configurationError`: the
/// stable code, the per-code message, and the caller's details.
fn configuration_error(code: &str, details: JsonValue) -> OperationError {
    let message = if code == MODEL_UNAVAILABLE {
        "The configured model is unavailable in this process"
    } else {
        "One or more configured tools are unavailable in this process"
    };
    OperationError {
        code: code.to_owned(),
        message: message.to_owned(),
        details: Some(details),
    }
}

/// Resolves the configured system prompt, upstream's `resolveSystemPrompt`:
/// an unset prompt is the empty string, a static prompt stands, and a
/// per-turn provider receives the resolved tool context — upstream's
/// `undefined` tool context restates as [`ToolContext`]'s `None`.
async fn resolve_system_prompt(lane: &Lane, context: &Context) -> String {
    let config = lane.read_config();
    match config.system_prompt {
        None => String::new(),
        Some(SystemPromptSource::Static(prompt)) => prompt,
        Some(SystemPromptSource::Provided(provider)) => {
            let tool_context = match &config.tool_context {
                None => None,
                Some(AgentHarnessToolContextSource::Static(value)) => value.clone(),
                Some(AgentHarnessToolContextSource::Resolved(resolver)) => resolver(context).await,
            };
            provider(tool_context, context).await
        }
    }
}

/// Resolves the model, tools, bounded context, and stream options one ready
/// attempt needs, upstream's `prepareGeneration`. The model lookup and the
/// active tool names check the live registry against the generation
/// context's captured configuration; a missing capture fails before any id
/// is reserved, upstream's `configuration_failure`.
///
/// # Errors
/// The invariant `` `Configured tool {name} disappeared during resolution` ``
/// when a name the availability check passed vanishes before its provider
/// surface builds, the bounded context's storage errors, and the
/// `before_request` hook's failure.
async fn prepare_generation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    generation: &AssistantReadyOperation,
) -> Result<GenerationPreparation, LaneError> {
    let identity = &generation.generation_context.configuration.model;
    let Some(model) = lane.models().model(&identity.provider, &identity.model_id) else {
        let details = match serde_json::to_value(identity) {
            Ok(value) => value,
            Err(error) => unreachable!("the model identity serializes: {error}"),
        };
        return Ok(GenerationPreparation::ConfigurationFailure(
            configuration_error(MODEL_UNAVAILABLE, details),
        ));
    };

    let config = lane.read_config();
    let tools_by_name: BTreeMap<&str, &AgentHarnessTool> = config
        .tools
        .iter()
        .map(|tool| (tool.tool.name.as_str(), tool))
        .collect();
    let missing_tools: Vec<String> = generation
        .generation_context
        .configuration
        .active_tool_names
        .iter()
        .filter(|name| !tools_by_name.contains_key(name.as_str()))
        .cloned()
        .collect();
    if !missing_tools.is_empty() {
        return Ok(GenerationPreparation::ConfigurationFailure(
            configuration_error(
                CONFIGURED_TOOLS_UNAVAILABLE,
                serde_json::json!({ "tools": missing_tools }),
            ),
        ));
    }
    let tools: Vec<Tool> = generation
        .generation_context
        .configuration
        .active_tool_names
        .iter()
        .map(|name| {
            let Some(tool) = tools_by_name.get(name.as_str()) else {
                return Err(lane_error(SessionError::Invariant(format!(
                    "Configured tool {name} disappeared during resolution"
                ))));
            };
            Ok(tool.tool.clone())
        })
        .collect::<Result<Vec<_>, LaneError>>()?;

    let messages = read_bounded_context(lane, drive).await?;
    let ContinueOperationResult::Result { value: messages } = messages else {
        return Ok(GenerationPreparation::CancelRequested);
    };
    let system_prompt = resolve_system_prompt(lane, &drive.context).await;
    let hook = lane
        .hooks()
        .run_with_gate(
            HookName::BeforeRequest,
            HookInvocation {
                lane: lane.name().to_owned(),
                run_id: drive.operation_id.clone(),
                event: HookEvent::BeforeRequest {
                    model: model.clone(),
                    step: StepKind::Assistant,
                    attempt: generation.next_attempt,
                    stream_options: generation.generation_context.stream_options.clone(),
                },
            },
            &drive.gate,
            &drive.context,
        )
        .await
        .map_err(|error| hook_error_to_lane_error(error, drive))?;
    let HookResult::BeforeRequest(patch) = hook else {
        unreachable!("before_request returns its own result variant");
    };
    let stream_options = patch.map_or_else(
        || generation.generation_context.stream_options.clone(),
        |patch| {
            apply_stream_options_patch(
                &generation.generation_context.stream_options,
                &patch.stream_options,
            )
        },
    );
    Ok(GenerationPreparation::Ready(PreparedGeneration {
        model,
        tools,
        messages,
        system_prompt,
        stream_options,
        to_provider_messages: config.to_provider_messages,
    }))
}

/// Publishes the durable generation intent, upstream's
/// `publishGenerationIntent`: one commit moves the ready leaf to
/// `assistant.effect_pending` under the attempt in flight, the model's
/// provider-reported output limit and context window, and the response and
/// usage entry ids — both minted from the same timestamp seed so the pair
/// sorts together. The first attempt's commit announces the turn; retries
/// continue silently.
///
/// Cancellation requested before the planner downgrades to
/// [`ContinueOperationResult::CancelRequested`], upstream's
/// `continueOperation` contract.
///
/// # Errors
/// The commit's storage and delivery errors.
async fn publish_generation_intent(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    ready: &AssistantReadyOperation,
    prepared: &PreparedGeneration,
) -> Result<ContinueOperationResult<AssistantEffectPendingOperation>, LaneError> {
    let at = now_ms();
    let response_entry_id = lane.session().id_generator().next(Some(at));
    let usage_id = lane.session().id_generator().next(Some(at));
    let generation_context = ready.generation_context.clone();
    let attempt = ready.next_attempt;
    let intended_output_limit = prepared.model.max_tokens;
    let context_window = prepared.model.context_window;
    let first_attempt = ready.next_attempt == 1;
    let turn_start = lane_scoped_event(
        lane.name(),
        false,
        "turn_start",
        HarnessEventPayload::TurnStart {
            run_id: drive.operation_id.clone(),
            turn_id: ready.generation_context.step_id.clone(),
        },
    );
    let published = lane
        .continue_operation::<AssistantEffectPendingOperation, _>(
            move |state, _session, _context| {
                let generation_context = generation_context.clone();
                let response_entry_id = response_entry_id.clone();
                let usage_id = usage_id.clone();
                let turn_start = turn_start.clone();
                Box::pin(async move {
                    let Some(current) = state.operation.as_ref() else {
                        unreachable!("continueOperation planner runs under an active operation")
                    };
                    let pending = AssistantEffectPendingOperation {
                        scope: operation_scope_of(&current.state),
                        generation_context,
                        attempt,
                        response_entry_id,
                        usage_id,
                        intended_output_limit,
                        context_window,
                    };
                    Ok(OperationCommand::Commit {
                        writes: Vec::new(),
                        operation_state: OperationState::AssistantEffectPending(pending.clone()),
                        lane: None,
                        materialize: Arc::new(move |_: &CommitResult| pending.clone()),
                        events: Some(Arc::new(move |_: &CommitResult| {
                            if first_attempt {
                                vec![turn_start.clone()]
                            } else {
                                Vec::new()
                            }
                        })),
                    })
                })
            },
            &drive.context,
        )
        .await?;
    Ok(published)
}

/// Streams one assistant response under the reserved ids, upstream's
/// `performGeneration`: the response lifecycle opens, the stream runs with
/// the context-transform, payload, after-response, and request hooks
/// admitted through the pass's gate, and the lifecycle closes whether or
/// not the stream faulted, upstream's `finally`.
///
/// # Errors
/// The stream's protocol and hook errors, and the lifecycle close's frame
/// write failure.
#[expect(
    clippy::too_many_lines,
    reason = "the port mirrors upstream's single performGeneration method and its hook closures"
)]
async fn perform_generation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    intent: &AssistantEffectPendingOperation,
    prepared: &PreparedGeneration,
) -> Result<SettledAssistantMessage, LaneError> {
    let response = open_assistant_response(lane, drive, &intent.response_entry_id, false);
    let refusal: Arc<Mutex<Option<LaneError>>> = Arc::new(Mutex::new(None));

    let transform_context: TransformContextHook = {
        let lane = Arc::clone(lane);
        let drive = Arc::clone(drive);
        Arc::new(
            move |request_context: AssistantRequestContext, context: &Context| {
                let lane = Arc::clone(&lane);
                let drive = Arc::clone(&drive);
                let lane_name = lane.name().to_owned();
                let run_id = drive.operation_id.clone();
                Box::pin(async move {
                    let result = match lane
                        .hooks()
                        .run_with_gate(
                            HookName::TransformContext,
                            HookInvocation {
                                lane: lane_name,
                                run_id,
                                event: HookEvent::TransformContext {
                                    messages: request_context.messages.clone(),
                                    system_prompt: request_context.system_prompt.clone(),
                                },
                            },
                            &drive.gate,
                            context,
                        )
                        .await
                    {
                        Ok(result) => result,
                        Err(HookRunError::GateAborted(abort)) => {
                            return Err(hook_error_box(abort));
                        }
                        Err(HookRunError::Aborted(_)) => {
                            return Err(hook_error_box(AbortRequested {
                                cancellation: drive.abort_cancellation(),
                            }));
                        }
                        Err(error) => return Err(hook_error_box(error)),
                    };
                    let HookResult::TransformContext(result) = result else {
                        unreachable!("transform_context returns its own result variant");
                    };
                    Ok(AssistantRequestContext {
                        messages: result
                            .as_ref()
                            .and_then(|result| result.messages.clone())
                            .unwrap_or(request_context.messages),
                        system_prompt: result
                            .and_then(|result| result.system_prompt)
                            .unwrap_or(request_context.system_prompt),
                    })
                })
            },
        )
    };

    let before_payload: BeforePayloadHook = {
        let lane = Arc::clone(lane);
        let drive = Arc::clone(drive);
        Arc::new(
            move |payload: JsonValue, request_model: Model, context: Context| {
                let lane = Arc::clone(&lane);
                let drive = Arc::clone(&drive);
                let lane_name = lane.name().to_owned();
                let run_id = drive.operation_id.clone();
                Box::pin(async move {
                    match lane
                        .hooks()
                        .run_with_gate(
                            HookName::BeforePayload,
                            HookInvocation {
                                lane: lane_name,
                                run_id,
                                event: HookEvent::BeforePayload {
                                    model: request_model,
                                    payload,
                                },
                            },
                            &drive.gate,
                            &context,
                        )
                        .await
                    {
                        Ok(HookResult::BeforePayload(result)) => {
                            result.map(|result| result.payload)
                        }
                        Ok(_) => unreachable!("before_payload returns its own result variant"),
                        // The OnPayload seam carries no error channel; the
                        // request's linked cancellation token aborts it when the
                        // signal aborted, upstream's onPayload throw reaching the
                        // same settlement.
                        Err(_) => None,
                    }
                })
            },
        )
    };

    let request: AssistantRequestHook = {
        let lane = Arc::clone(lane);
        let drive = Arc::clone(drive);
        let model = prepared.model.clone();
        let refusal = Arc::clone(&refusal);
        Arc::new(
            move |ai_context: &AiContext,
                  options: pi_ai::types::SimpleStreamOptions,
                  context: &Context| {
                let lane = Arc::clone(&lane);
                let drive = Arc::clone(&drive);
                let model = model.clone();
                let refusal = Arc::clone(&refusal);
                let session_id = lane.session().metadata().id.clone();
                let lane_name = lane.name().to_owned();
                // The request runs against a context cancelled by either the
                // caller's signal or the gate's, upstream's
                // `withAbortSignal(drive.gate.signal, context)`.
                let admitted = with_abort_signal(drive.gate.signal().clone(), context);
                let signal = request_signal(&admitted);
                Box::pin(async move {
                    let mut options = options;
                    options.session_id = Some(format!("{session_id}:{lane_name}"));
                    options.transport_options.signal = signal;
                    options.telemetry_context = Some(get_telemetry_context(&admitted));
                    let request_options = WithTransforms {
                        options,
                        transform_headers: None,
                    };
                    match drive.gate.admit(|| {
                        lane.models()
                            .stream_simple(&model, ai_context, Some(&request_options))
                    }) {
                        Ok(stream) => stream,
                        Err(rejection) => {
                            // The refusal cannot cross the hook's stream
                            // return, upstream's thrown `AbortRequested`; it
                            // records for the caller and the stream ends
                            // aborted — the settlement only the discard path
                            // below observes.
                            let error = gate_rejection_error(rejection);
                            *refusal.lock().unwrap_or_else(PoisonError::into_inner) =
                                Some(Arc::clone(&error));
                            let stream = assistant_message_event_stream();
                            let settlement = refused_stream_settlement(&model);
                            stream.push(AssistantMessageEvent::Error {
                                reason: StopReason::Aborted,
                                error: settlement.clone(),
                            });
                            stream.end(Some(&settlement));
                            stream
                        }
                    }
                })
            },
        )
    };

    let config = HarnessAssistantStreamConfig {
        model: prepared.model.clone(),
        system_prompt: prepared.system_prompt.clone(),
        tools: Some(prepared.tools.clone()),
        thinking_level: intent.generation_context.configuration.thinking_level,
        stream_options: prepared.stream_options.clone(),
        transform_context: Some(transform_context),
        to_provider_messages: prepared.to_provider_messages.clone(),
        before_payload: Some(before_payload),
        after_response: Some(Arc::clone(&response.after_response)),
        request,
        observer: Arc::clone(&response.observer),
    };

    let settled = match stream_harness_assistant(&prepared.messages, &config, &drive.context).await
    {
        Ok(settled) => settled,
        Err(AssistantStreamError::Hook(error)) => {
            // The boxed error keeps its concrete type, so the spine's
            // abort catch recognizes the carried [`AbortRequested`];
            // every other hook failure faults the pass, upstream's
            // rethrow.
            (response.close)().await?;
            return Err(Arc::from(error));
        }
        Err(error) => {
            (response.close)().await?;
            return Err(lane_error(error));
        }
    };
    (response.close)().await?;
    let rejection = refusal
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    if let Some(rejection) = rejection {
        return Err(rejection);
    }
    Ok(settled)
}

/// The aborted settlement the refused request hook ends its stream with;
/// upstream has no counterpart because its throw precedes any settlement.
/// The discard path reads only the stop reason, so the fields restate the
/// request's model identity and upstream's `normalizeAborted` fallback
/// text.
///
/// The synthetic settlement also ends the port's stream normally: the
/// response lifecycle runs the `after_response` admission and the
/// observer's end, so a `message_end` surfaces for a message upstream
/// never announced — the durable outcome matches, since the generation
/// discards the settlement and raises the recorded rejection.
fn refused_stream_settlement(model: &Model) -> pi_ai::types::AssistantMessage {
    pi_ai::types::AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: pi_ai::types::Usage::default(),
        stop_reason: StopReason::Aborted,
        deferred: None,
        error_message: Some("Assistant request was cancelled".to_owned()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms(),
    }
}

/// Advance one durable assistant retry wait according to this pass's local
/// wait policy, upstream's `runRetryWait`.
///
/// With waiting disabled the due wait returns the durable waiting outcome
/// without a timer or a commit; with waiting enabled the wait sleeps inside
/// a gate permit — an abort rejects through the gate so the spine awaits
/// the cancellation. At or past the deadline one commit re-enters the ready
/// leaf and announces `retry_start`.
///
/// # Errors
/// The wait's abort rejection and the commit's storage and delivery errors.
pub(crate) async fn run_retry_wait(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    generation: &AssistantRetryWaitOperation,
) -> Result<ProcedureResult, LaneError> {
    if now_ms() < generation.retry_wait.not_before {
        if !drive.wait_for_retry {
            return Ok(ProcedureResult::Waiting {
                outcome: DriveOutcome::Waiting {
                    operation_id: drive.operation_id.clone(),
                    reason: DriveWaitReason::Retry {
                        not_before: generation.retry_wait.not_before,
                    },
                },
            });
        }
        match drive
            .gate
            .admit(|| wait_until(generation.retry_wait.not_before, drive))
        {
            Ok(wait) => wait.await?,
            Err(rejection) => return Err(gate_rejection_error(rejection)),
        }
    }

    let generation_context = generation.generation_context.clone();
    let next_attempt = generation.retry_wait.next_attempt;
    let retry_start = lane_scoped_event(
        lane.name(),
        false,
        "retry_start",
        HarnessEventPayload::RetryStart {
            run_id: drive.operation_id.clone(),
            step: generation.generation_context.step_id.clone(),
            attempt: generation.retry_wait.next_attempt,
        },
    );
    let published = lane
        .continue_operation::<ProcedureResult, _>(
            move |state, _session, _context| {
                let generation_context = generation_context.clone();
                let retry_start = retry_start.clone();
                Box::pin(async move {
                    let Some(current) = state.operation.as_ref() else {
                        unreachable!("continueOperation planner runs under an active operation")
                    };
                    let ready = AssistantReadyOperation {
                        scope: operation_scope_of(&current.state),
                        generation_context,
                        next_attempt,
                    };
                    Ok(OperationCommand::Commit {
                        writes: Vec::new(),
                        operation_state: OperationState::AssistantReady(ready),
                        lane: None,
                        materialize: Arc::new(|_: &CommitResult| ProcedureResult::Continue),
                        events: Some(Arc::new(move |_: &CommitResult| vec![retry_start.clone()])),
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

/// Execute one ready assistant generation or advance its durable retry
/// wait, upstream's `runGeneration`.
///
/// The ready leaf resolves the request, publishes the intent, streams the
/// response, and settles it; a configuration failure publishes before any
/// id is reserved; a cancellation winning preparation or intent publication
/// downgrades to a plain continue, upstream's cancel-refused arms.
///
/// # Errors
/// The configuration resolution's invariants, the hook failures outside
/// the abort path, and the response publication's errors.
pub(crate) async fn run_generation(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    generation: GenerationLeaf,
) -> Result<ProcedureResult, LaneError> {
    match generation {
        GenerationLeaf::RetryWait(retry) => run_retry_wait(lane, drive, &retry).await,
        GenerationLeaf::Ready(ready) => {
            let prepared = prepare_generation(lane, drive, &ready).await?;
            let prepared = match prepared {
                GenerationPreparation::ConfigurationFailure(error) => {
                    return publish_configuration_failure(
                        lane,
                        drive,
                        &OperationState::AssistantReady(ready),
                        error,
                    )
                    .await;
                }
                GenerationPreparation::CancelRequested => return Ok(ProcedureResult::Continue),
                GenerationPreparation::Ready(prepared) => prepared,
            };

            let intent = publish_generation_intent(lane, drive, &ready, &prepared).await?;
            let ContinueOperationResult::Result { value: intent } = intent else {
                return Ok(ProcedureResult::Continue);
            };
            let response = perform_generation(lane, drive, &intent, &prepared).await?;
            publish_response(
                lane,
                drive,
                ResponseIntent::Assistant(intent),
                response,
                ResponseOptions { recovery: false },
            )
            .await
        }
    }
}

//! The low-level agent loop, upstream's `src/agent-loop.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`agent_loop`] and [`agent_loop_continue`] start a run and return the
//! event stream; [`run_agent_loop`] and [`run_agent_loop_continue`] are the
//! awaited entries that take an explicit [`AgentEventSink`]. The loop works
//! with [`AgentMessage`] throughout and converts to LLM messages only at the
//! provider-call boundary.
//!
//! Porting restatements:
//! - the run is fired on a spawned task, upstream's `void runAgentLoop(...)`;
//!   the returned stream settles on the run's `agent_end` event;
//! - upstream's `AbortSignal` is the workspace's cancellation token;
//! - parallel tool execution spawns one task per prepared call, the
//!   JavaScript event loop's interleaving restated as runtime scheduling;
//!   `tool_execution_end` is emitted by the completing task while
//!   tool-result messages are emitted afterwards in assistant source order;
//! - `beforeToolCall` receives an owned snapshot of the validated arguments,
//!   so hook-side mutation cannot flow into execution the way JavaScript's
//!   shared object let it;
//! - a tool's partial-result updates drain through the sink after the
//!   execution outcome exists, in the order the tool emitted them — upstream
//!   pushes at update time and awaits settlement after, and no suite in this
//!   slice pins the cross-tool interleaving;
//! - a spawned tool task that panics yields an error tool result carrying
//!   the join error, the analog of upstream catching a tool throw; the
//!   sequential path propagates a panic instead, `Result` being the tool's
//!   error channel;
//! - a default stream-fn resolution failure leaves the returned stream open
//!   forever, upstream's rejected promise that never ends the stream;
//!   [`run_agent_loop`] and [`run_agent_loop_continue`] surface that error
//!   directly.

use std::sync::{Arc, Mutex, PoisonError, atomic::AtomicBool, atomic::Ordering};

use pi_ai::auth::resolve::now_ms;
use pi_ai::types::{
    AssistantBlock, AssistantMessage, AssistantMessageEvent, Context, Message, SimpleStreamOptions,
    StopReason, Tool, ToolCall, ToolResultMessage,
};
use pi_ai::utils::event_stream::{AssistantMessageEventStream, EventStream};
use pi_ai::utils::validation::validate_tool_arguments;
use tokio_util::sync::CancellationToken;

use crate::stream_fn::get_default_stream_fn;
use crate::types::{
    AfterToolCallContext, AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool,
    AgentToolContent, AgentToolError, AgentToolResult, BeforeToolCallContext, BoxedFuture,
    PrepareNextTurnContext, StreamFn, ThinkingLevel, ToolExecutionMode,
};

/// The sink a run reports events through, upstream's `AgentEventSink`.
///
/// The loop awaits every delivery before proceeding, so a sink that awaits
/// listeners (the `Agent` class's) back-pressures the run exactly where
/// upstream's awaited emissions do.
pub type AgentEventSink = Arc<dyn Fn(AgentEvent) -> BoxedFuture<'static, ()> + Send + Sync>;

/// The stream a loop run reports on: [`AgentEvent`]s in emission order,
/// settling to the messages the run produced on `agent_end`.
pub type AgentEventStream = EventStream<AgentEvent, Vec<AgentMessage>>;

/// Why a loop failed to start, upstream's thrown `Error`s.
#[derive(Debug)]
pub enum AgentLoopError {
    /// [`agent_loop_continue`] on an empty context, upstream's
    /// `Cannot continue: no messages in context`.
    NoMessages,
    /// [`agent_loop_continue`] whose last context message is an assistant
    /// message, upstream's `Cannot continue from message role: assistant`.
    ContinueFromAssistant,
    /// No stream function was supplied and none is configured; wraps
    /// [`NoDefaultStreamFn`](crate::stream_fn::NoDefaultStreamFn).
    NoDefaultStreamFn(crate::stream_fn::NoDefaultStreamFn),
}

impl std::fmt::Display for AgentLoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMessages => f.write_str("Cannot continue: no messages in context"),
            Self::ContinueFromAssistant => {
                f.write_str("Cannot continue from message role: assistant")
            }
            Self::NoDefaultStreamFn(error) => std::fmt::Display::fmt(error, f),
        }
    }
}

impl std::error::Error for AgentLoopError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoDefaultStreamFn(error) => Some(error),
            _ => None,
        }
    }
}

impl From<crate::stream_fn::NoDefaultStreamFn> for AgentLoopError {
    fn from(error: crate::stream_fn::NoDefaultStreamFn) -> Self {
        Self::NoDefaultStreamFn(error)
    }
}

/// Start an agent loop with new prompt messages, upstream's `agentLoop`.
///
/// The prompts are added to the context and events are emitted for them; the
/// returned stream settles on `agent_end` with the messages the run
/// produced.
///
/// `stream_fn: None` resolves through the process-global default at run
/// start. When no default is configured the run dies after emitting the
/// prompt events and the stream is never ended — a consumer iterating it
/// waits forever, upstream's rejected promise. Pass a `stream_fn` or
/// configure [`set_default_stream_fn`](crate::stream_fn::set_default_stream_fn);
/// [`run_agent_loop`] surfaces the error directly instead.
///
/// # Panics
/// Called outside a tokio runtime context: the spawned run needs one.
#[must_use]
pub fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<CancellationToken>,
    stream_fn: Option<StreamFn>,
) -> AgentEventStream {
    let stream = create_agent_stream();
    let sink = sink_for_stream(&stream);
    let end_stream = stream.clone();
    tokio::spawn(async move {
        if let Ok(messages) = Box::pin(run_agent_loop(
            prompts, context, config, sink, signal, stream_fn,
        ))
        .await
        {
            end_stream.end(Some(&messages));
            // Otherwise the stream stays open, upstream's rejected-promise
            // statement.
        }
    });
    stream
}

/// Continue an agent loop from the current context without adding a new
/// message, upstream's `agentLoopContinue`. Used for retries — the context
/// already holds the user message or tool results.
///
/// **Important:** the last message in context must convert to a `user` or
/// `toolResult` message via `convert_to_llm`; if it doesn't, the provider
/// rejects the request. This cannot be validated here since
/// `convert_to_llm` is only called once per turn.
///
/// # Errors
/// [`AgentLoopError::NoMessages`] on an empty context and
/// [`AgentLoopError::ContinueFromAssistant`] when the last message is an
/// assistant message, both before any run starts.
///
/// `stream_fn: None` resolves through the process-global default at run
/// start; an unconfigured default leaves the returned stream open forever
/// ([`agent_loop`] documents the degradation).
///
/// # Panics
/// Called outside a tokio runtime context.
pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<CancellationToken>,
    stream_fn: Option<StreamFn>,
) -> Result<AgentEventStream, AgentLoopError> {
    validate_continue_context(&context)?;
    let stream = create_agent_stream();
    let sink = sink_for_stream(&stream);
    let end_stream = stream.clone();
    tokio::spawn(async move {
        if let Ok(messages) = Box::pin(run_agent_loop_continue(
            context, config, sink, signal, stream_fn,
        ))
        .await
        {
            end_stream.end(Some(&messages));
        }
    });
    Ok(stream)
}

/// Run the loop with new prompt messages and report every event through
/// `emit`, upstream's `runAgentLoop`. Returns the messages the run produced.
///
/// # Errors
/// [`AgentLoopError::NoDefaultStreamFn`] when `stream_fn` is `None` and no
/// process-global default is configured; the `agent_start`/`turn_start` and
/// prompt events have already been emitted by then, matching the upstream
/// evaluation order.
pub async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Option<CancellationToken>,
    stream_fn: Option<StreamFn>,
) -> Result<Vec<AgentMessage>, AgentLoopError> {
    let mut new_messages: Vec<AgentMessage> = prompts.clone();
    let current_context = AgentContext {
        system_prompt: context.system_prompt.clone(),
        tools: context.tools.clone(),
        messages: {
            let mut messages = context.messages;
            messages.extend(prompts.iter().cloned());
            messages
        },
    };

    emit_event(&emit, AgentEvent::AgentStart).await;
    emit_event(&emit, AgentEvent::TurnStart).await;
    for prompt in &prompts {
        emit_event(
            &emit,
            AgentEvent::MessageStart {
                message: prompt.clone(),
            },
        )
        .await;
        emit_event(
            &emit,
            AgentEvent::MessageEnd {
                message: prompt.clone(),
            },
        )
        .await;
    }

    let stream_fn = resolve_stream_fn(stream_fn)?;
    run_loop(
        current_context,
        &mut new_messages,
        config,
        signal,
        &emit,
        &stream_fn,
    )
    .await;
    Ok(new_messages)
}

/// Continue the loop from the current context and report every event through
/// `emit`, upstream's `runAgentLoopContinue`.
///
/// Returns the messages this run produced — pre-existing context messages
/// are not included.
///
/// # Errors
/// The same context validation [`agent_loop_continue`] applies, plus
/// [`AgentLoopError::NoDefaultStreamFn`] when `stream_fn` is `None` and no
/// default is configured.
pub async fn run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Option<CancellationToken>,
    stream_fn: Option<StreamFn>,
) -> Result<Vec<AgentMessage>, AgentLoopError> {
    validate_continue_context(&context)?;

    let mut new_messages: Vec<AgentMessage> = Vec::new();
    let current_context = context;

    emit_event(&emit, AgentEvent::AgentStart).await;
    emit_event(&emit, AgentEvent::TurnStart).await;

    let stream_fn = resolve_stream_fn(stream_fn)?;
    run_loop(
        current_context,
        &mut new_messages,
        config,
        signal,
        &emit,
        &stream_fn,
    )
    .await;
    Ok(new_messages)
}

/// The stream a loop run reports on: `agent_end` settles it with the run's
/// messages.
fn create_agent_stream() -> AgentEventStream {
    EventStream::new(
        |event| matches!(event, AgentEvent::AgentEnd { .. }),
        |event| match event {
            AgentEvent::AgentEnd { messages } => Some(messages.clone()),
            _ => None,
        },
    )
}

/// The sink that feeds a run's events into its stream.
fn sink_for_stream(stream: &AgentEventStream) -> AgentEventSink {
    let stream = stream.clone();
    Arc::new(move |event| {
        let stream = stream.clone();
        Box::pin(async move { stream.push(event) })
    })
}

async fn emit_event(emit: &AgentEventSink, event: AgentEvent) {
    (emit)(event).await;
}

/// Resolve the run's stream function, upstream's `streamFn ??
/// getDefaultStreamFn()`.
fn resolve_stream_fn(stream_fn: Option<StreamFn>) -> Result<StreamFn, AgentLoopError> {
    match stream_fn {
        Some(stream_fn) => Ok(stream_fn),
        None => Ok(get_default_stream_fn()?),
    }
}

/// The two continue-entry validations, upstream-verbatim.
fn validate_continue_context(context: &AgentContext) -> Result<(), AgentLoopError> {
    if context.messages.is_empty() {
        return Err(AgentLoopError::NoMessages);
    }
    if matches!(
        context.messages.last(),
        Some(AgentMessage::Standard(Message::Assistant(_)))
    ) {
        return Err(AgentLoopError::ContinueFromAssistant);
    }
    Ok(())
}

/// Queued steering/follow-up messages join the transcript before the next
/// assistant response: each one's start/end events emit, then it lands in
/// the context and the run's messages.
async fn inject_pending_messages(
    context: &mut AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    pending_messages: &mut Vec<AgentMessage>,
    emit: &AgentEventSink,
) {
    for message in std::mem::take(pending_messages) {
        emit_event(
            emit,
            AgentEvent::MessageStart {
                message: message.clone(),
            },
        )
        .await;
        emit_event(
            emit,
            AgentEvent::MessageEnd {
                message: message.clone(),
            },
        )
        .await;
        context.messages.push(message.clone());
        new_messages.push(message);
    }
}

/// The turn and run end when the assistant response failed or was aborted:
/// the turn closes with no tool results and the run ends immediately.
async fn end_run_on_error(
    message: &AssistantMessage,
    new_messages: &[AgentMessage],
    emit: &AgentEventSink,
) {
    emit_event(
        emit,
        AgentEvent::TurnEnd {
            message: AgentMessage::Standard(Message::Assistant(message.clone())),
            tool_results: Vec::new(),
        },
    )
    .await;
    emit_event(
        emit,
        AgentEvent::AgentEnd {
            messages: new_messages.to_vec(),
        },
    )
    .await;
}

/// The turn's tool-call phase: a "length" stop fails every truncated call
/// instead of executing potentially borked ones; otherwise the batch
/// executes, its tool-result messages are appended to the context and the
/// run's messages, and the batch's early-termination hint is reported. A
/// message with no tool calls terminates with `(empty, true)` — the batch
/// phase ends, upstream's `hasMoreToolCalls = false` default.
async fn run_tool_batch(
    context: &mut AgentContext,
    message: &AssistantMessage,
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
    emit: &AgentEventSink,
    new_messages: &mut Vec<AgentMessage>,
) -> (Vec<ToolResultMessage>, bool) {
    let tool_calls = assistant_tool_calls(message);
    if tool_calls.is_empty() {
        return (Vec::new(), true);
    }

    // A "length" stop means the output was cut off by the token limit, so
    // every tool call in the message may carry truncated arguments. Fail them
    // all instead of executing potentially borked calls.
    let executed_tool_batch = if message.stop_reason == StopReason::Length {
        fail_tool_calls_from_truncated_message(&tool_calls, emit).await
    } else {
        execute_tool_calls(context, message, config, signal, emit).await
    };
    for result in &executed_tool_batch.messages {
        let message = AgentMessage::Standard(Message::ToolResult(result.clone()));
        context.messages.push(message.clone());
        new_messages.push(message);
    }
    (executed_tool_batch.messages, executed_tool_batch.terminate)
}

/// Main loop logic shared by both run entries, upstream's `runLoop`.
///
/// The outer loop continues while queued follow-up messages arrive after the
/// agent would stop; the inner loop processes tool calls and steering
/// messages.
async fn run_loop(
    initial_context: AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    initial_config: AgentLoopConfig,
    signal: Option<CancellationToken>,
    emit: &AgentEventSink,
    stream_fn: &StreamFn,
) {
    let mut context = initial_context;
    let mut config = initial_config;
    let mut last_completed_turn: Option<PrepareNextTurnContext> = None;
    // Check for steering messages at start (user may have typed while waiting)
    let mut pending_messages = poll_steering(&config).await;

    'outer: loop {
        let mut has_more_tool_calls = true;

        // Inner loop: process tool calls and steering messages
        while has_more_tool_calls || !pending_messages.is_empty() {
            if let Some(turn) = last_completed_turn.take() {
                if let Some(prepare) = &config.prepare_next_turn
                    && let Some(next_turn) = prepare(turn).await
                {
                    apply_turn_update(&mut context, &mut config, next_turn);
                }
                // Preparation can be long-running (for example, compaction). Pick
                // up steering queued while it ran. Only poll again if the earlier
                // poll returned nothing; otherwise one-at-a-time mode would
                // deliver two messages in this turn.
                if pending_messages.is_empty() {
                    pending_messages = poll_steering(&config).await;
                }
                emit_event(emit, AgentEvent::TurnStart).await;
            }

            // Process pending messages (inject before next assistant response)
            inject_pending_messages(&mut context, new_messages, &mut pending_messages, emit).await;

            // Stream assistant response
            let message =
                stream_assistant_response(&mut context, &config, signal.as_ref(), emit, stream_fn)
                    .await;
            new_messages.push(AgentMessage::Standard(Message::Assistant(message.clone())));

            if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
                end_run_on_error(&message, new_messages, emit).await;
                return;
            }

            // Check for tool calls
            let (tool_results, terminated) = run_tool_batch(
                &mut context,
                &message,
                &config,
                signal.as_ref(),
                emit,
                new_messages,
            )
            .await;
            // No tool calls means the batch terminates the tool phase, even
            // though its terminate hint is false.
            has_more_tool_calls = !terminated;

            emit_event(
                emit,
                AgentEvent::TurnEnd {
                    message: AgentMessage::Standard(Message::Assistant(message.clone())),
                    tool_results: tool_results.clone(),
                },
            )
            .await;

            let completed_turn = PrepareNextTurnContext {
                message: message.clone(),
                tool_results,
                context: context.clone(),
                new_messages: new_messages.clone(),
            };

            if let Some(should_stop) = &config.should_stop_after_turn
                && should_stop(completed_turn.clone()).await
            {
                emit_event(
                    emit,
                    AgentEvent::AgentEnd {
                        messages: new_messages.clone(),
                    },
                )
                .await;
                return;
            }

            last_completed_turn = Some(completed_turn);
            pending_messages = poll_steering(&config).await;
        }

        // Agent would stop here. Check for follow-up messages.
        let follow_up_messages = match &config.get_follow_up_messages {
            Some(hook) => hook().await,
            None => Vec::new(),
        };
        if !follow_up_messages.is_empty() {
            // Set as pending so inner loop processes them
            pending_messages = follow_up_messages;
            continue 'outer;
        }

        // No more messages, exit
        break;
    }

    emit_event(
        emit,
        AgentEvent::AgentEnd {
            messages: new_messages.clone(),
        },
    )
    .await;
}

/// The tool-call blocks of an assistant message, in content order.
fn assistant_tool_calls(message: &AssistantMessage) -> Vec<ToolCall> {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .collect()
}

/// Steering messages queued for the loop, upstream's
/// `(await config.getSteeringMessages?.()) || []`.
async fn poll_steering(config: &AgentLoopConfig) -> Vec<AgentMessage> {
    match &config.get_steering_messages {
        Some(hook) => hook().await,
        None => Vec::new(),
    }
}

/// The simple-request `reasoning` value a thinking level maps to: the agent
/// level adds `off`, which the request options carry as an absent field.
/// Shared by the loop's turn updates and the `Agent`'s loop-config builder.
pub(crate) const fn stream_reasoning(level: ThinkingLevel) -> Option<pi_ai::types::ThinkingLevel> {
    match level {
        ThinkingLevel::Off => None,
        ThinkingLevel::Minimal => Some(pi_ai::types::ThinkingLevel::Minimal),
        ThinkingLevel::Low => Some(pi_ai::types::ThinkingLevel::Low),
        ThinkingLevel::Medium => Some(pi_ai::types::ThinkingLevel::Medium),
        ThinkingLevel::High => Some(pi_ai::types::ThinkingLevel::High),
        ThinkingLevel::Xhigh => Some(pi_ai::types::ThinkingLevel::Xhigh),
        ThinkingLevel::Max => Some(pi_ai::types::ThinkingLevel::Max),
    }
}

/// A `prepare_next_turn` update applied to the loop's live state: the
/// replacement context replaces the whole one, the model and thinking level
/// replace the config's, and absent fields keep the current values.
fn apply_turn_update(
    context: &mut AgentContext,
    config: &mut AgentLoopConfig,
    next_turn: crate::types::AgentLoopTurnUpdate,
) {
    if let Some(next_context) = next_turn.context {
        *context = next_context;
    }
    if let Some(model) = next_turn.model {
        config.model = model;
    }
    if let Some(thinking_level) = next_turn.thinking_level {
        config.stream_options.reasoning = stream_reasoning(thinking_level);
    }
}

/// Stream an assistant response from the LLM.
///
/// This is where `AgentMessage`s get transformed to LLM messages: the
/// transform hook runs at the [`AgentMessage`] level, then `convert_to_llm`
/// produces the provider-call transcript.
async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
    emit: &AgentEventSink,
    stream_fn: &StreamFn,
) -> AssistantMessage {
    let (llm_context, options) = build_llm_request(context, config, signal).await;
    let response = stream_fn(&config.model, &llm_context, Some(&options));
    // Set once the `start` event added the live partial to the transcript;
    // every later partial replaces that entry, upstream's shared accumulator.
    let mut added_partial = false;

    while let Some(event) = response.next().await {
        match &event {
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => {
                return settle_final_message(&response, context, emit, added_partial).await;
            }
            AssistantMessageEvent::Start { partial } => {
                let partial = partial.clone();
                context
                    .messages
                    .push(AgentMessage::Standard(Message::Assistant(partial.clone())));
                added_partial = true;
                emit_event(
                    emit,
                    AgentEvent::MessageStart {
                        message: AgentMessage::Standard(Message::Assistant(partial)),
                    },
                )
                .await;
            }
            event => {
                // The remaining protocol events are partial updates.
                let Some(partial) = event_partial(event).cloned() else {
                    continue;
                };
                if added_partial {
                    replace_last_message(
                        context,
                        AgentMessage::Standard(Message::Assistant(partial.clone())),
                    );
                    emit_event(
                        emit,
                        AgentEvent::MessageUpdate {
                            assistant_message_event: event.clone(),
                            message: AgentMessage::Standard(Message::Assistant(partial)),
                        },
                    )
                    .await;
                }
            }
        }
    }

    // The stream ended without a terminal event; awaiting the result is
    // upstream's never-settling promise when a stream ends mid-flight.
    settle_final_message(&response, context, emit, added_partial).await
}

/// The provider-call inputs: the transformed transcript converted to LLM
/// messages, and the request options with the API key resolved for the
/// current turn (important for expiring tokens).
async fn build_llm_request(
    context: &AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
) -> (Context, SimpleStreamOptions) {
    // Apply context transform if configured (AgentMessage[] -> AgentMessage[])
    let mut messages = context.messages.clone();
    if let Some(transform) = &config.transform_context {
        messages = transform(messages, signal.cloned()).await;
    }

    // Convert to LLM-compatible messages (AgentMessage[] -> Message[])
    let llm_messages = (config.convert_to_llm)(messages).await;

    let llm_context = Context {
        system_prompt: Some(context.system_prompt.clone()),
        messages: llm_messages,
        tools: context.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| tool.tool.clone())
                .collect::<Vec<Tool>>()
        }),
    };

    let resolved_api_key = match &config.get_api_key {
        Some(get_api_key) => get_api_key(&config.model.provider).await,
        None => None,
    }
    .or_else(|| config.stream_options.api_key.clone());

    let mut options = config.stream_options.clone();
    options.api_key = resolved_api_key;
    options.transport_options.signal = signal.cloned();
    (llm_context, options)
}

/// Settle a finished stream: swap the live partial for the final message (or
/// push it when none was added), emit the transcript events, and return the
/// final message, upstream's `done`/`error` arm and the mid-flight-end tail.
async fn settle_final_message(
    response: &AssistantMessageEventStream,
    context: &mut AgentContext,
    emit: &AgentEventSink,
    added_partial: bool,
) -> AssistantMessage {
    let final_message = response.result().await;
    let final_event = AgentMessage::Standard(Message::Assistant(final_message.clone()));
    if added_partial {
        replace_last_message(context, final_event.clone());
    } else {
        context.messages.push(final_event.clone());
        emit_event(
            emit,
            AgentEvent::MessageStart {
                message: final_event.clone(),
            },
        )
        .await;
    }
    emit_event(
        emit,
        AgentEvent::MessageEnd {
            message: final_event,
        },
    )
    .await;
    final_message
}

/// The live partial message a mid-stream protocol event carries.
const fn event_partial(event: &AssistantMessageEvent) -> Option<&AssistantMessage> {
    match event {
        AssistantMessageEvent::Start { partial }
        | AssistantMessageEvent::TextStart { partial, .. }
        | AssistantMessageEvent::TextDelta { partial, .. }
        | AssistantMessageEvent::TextEnd { partial, .. }
        | AssistantMessageEvent::ThinkingStart { partial, .. }
        | AssistantMessageEvent::ThinkingDelta { partial, .. }
        | AssistantMessageEvent::ThinkingEnd { partial, .. }
        | AssistantMessageEvent::ToolcallStart { partial, .. }
        | AssistantMessageEvent::ToolcallDelta { partial, .. }
        | AssistantMessageEvent::ToolcallEnd { partial, .. } => Some(partial),
        AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => None,
    }
}

/// The transcript's last entry is the live streaming partial; every update
/// replaces it, upstream's `context.messages[len - 1] = partialMessage`.
fn replace_last_message(context: &mut AgentContext, message: AgentMessage) {
    if let Some(last) = context.messages.last_mut() {
        *last = message;
    }
}

/// The batch a tool-execution phase produces: the tool-result messages in
/// assistant source order, and whether the batch ends the run.
struct ExecutedToolCallBatch {
    messages: Vec<ToolResultMessage>,
    terminate: bool,
}

/// Emit `tool_execution_start` for a call, carrying its raw arguments.
async fn emit_tool_execution_start(tool_call: &ToolCall, emit: &AgentEventSink) {
    emit_event(
        emit,
        AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: raw_arguments(tool_call),
        },
    )
    .await;
}

/// Fail all tool calls from an assistant message that was truncated by the
/// output token limit. Streamed tool-call arguments are finalized with a
/// best-effort JSON salvage parser, so a truncated message can yield tool
/// calls whose arguments parse and validate but are silently incomplete.
/// None of them are safe to execute; report each as an error so the model
/// can re-issue them.
async fn fail_tool_calls_from_truncated_message(
    tool_calls: &[ToolCall],
    emit: &AgentEventSink,
) -> ExecutedToolCallBatch {
    let mut messages: Vec<ToolResultMessage> = Vec::new();
    for tool_call in tool_calls {
        emit_tool_execution_start(tool_call, emit).await;
        let finalized = FinalizedToolCallOutcome {
            tool_call: tool_call.clone(),
            result: error_tool_result(&format!(
                "Tool call \"{}\" was not executed: the response hit the output token limit, \
                 so its arguments may be truncated. Re-issue the tool call with complete \
                 arguments.",
                tool_call.name
            )),
            is_error: true,
        };
        emit_tool_execution_end(&finalized, emit).await;
        let tool_result_message = create_tool_result_message(&finalized);
        emit_tool_result_message(&tool_result_message, emit).await;
        messages.push(tool_result_message);
    }
    ExecutedToolCallBatch {
        messages,
        terminate: false,
    }
}

/// A tool call's raw arguments as the event payloads carry them.
fn raw_arguments(tool_call: &ToolCall) -> serde_json::Value {
    serde_json::Value::Object(tool_call.arguments.clone())
}

/// Execute tool calls from an assistant message.
async fn execute_tool_calls(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
    emit: &AgentEventSink,
) -> ExecutedToolCallBatch {
    let tool_calls = assistant_tool_calls(assistant_message);
    let has_sequential_tool_call = tool_calls.iter().any(|tool_call| {
        context
            .tools
            .as_ref()
            .and_then(|tools| tools.iter().find(|tool| tool.name() == tool_call.name))
            .and_then(|tool| tool.execution_mode)
            == Some(ToolExecutionMode::Sequential)
    });
    if config.tool_execution == Some(ToolExecutionMode::Sequential) || has_sequential_tool_call {
        execute_tool_calls_sequential(
            context,
            assistant_message,
            &tool_calls,
            config,
            signal,
            emit,
        )
        .await
    } else {
        execute_tool_calls_parallel(
            context,
            assistant_message,
            &tool_calls,
            config,
            signal,
            emit,
        )
        .await
    }
}

/// Execute tool calls one at a time: each call is prepared, executed, and
/// finalized before the next starts.
async fn execute_tool_calls_sequential(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
    emit: &AgentEventSink,
) -> ExecutedToolCallBatch {
    let mut finalized_calls: Vec<FinalizedToolCallOutcome> = Vec::new();
    let mut messages: Vec<ToolResultMessage> = Vec::new();

    for tool_call in tool_calls {
        emit_tool_execution_start(tool_call, emit).await;

        let finalized =
            match prepare_tool_call(context, assistant_message, tool_call, config, signal).await {
                Preparation::Immediate { result, is_error } => FinalizedToolCallOutcome {
                    tool_call: tool_call.clone(),
                    result,
                    is_error,
                },
                Preparation::Prepared(prepared) => {
                    let executed = execute_prepared_tool_call(&prepared, signal, emit).await;
                    finalize_executed_tool_call(
                        context,
                        assistant_message,
                        &prepared,
                        executed,
                        config,
                        signal,
                    )
                    .await
                }
            };

        emit_tool_execution_end(&finalized, emit).await;
        let tool_result_message = create_tool_result_message(&finalized);
        emit_tool_result_message(&tool_result_message, emit).await;
        finalized_calls.push(finalized);
        messages.push(tool_result_message);

        if is_aborted(signal) {
            break;
        }
    }

    ExecutedToolCallBatch {
        messages,
        terminate: should_terminate_tool_batch(&finalized_calls),
    }
}

/// One entry of a parallel batch: already-finalized outcomes (blocked calls,
/// unknown tools) and prepared calls whose execution runs concurrently.
#[expect(
    clippy::large_enum_variant,
    reason = "the finalized outcome is the entry's payload, mirroring the upstream shape; boxing would add allocation churn on every tool call"
)]
enum ParallelEntry {
    Immediate(FinalizedToolCallOutcome),
    Deferred(Box<PreparedToolCall>),
}

/// Execute tool calls concurrently: prepared calls run on spawned tasks, each
/// emitting its `tool_execution_end` as it finalizes; the tool-result
/// messages are emitted afterwards in assistant source order.
async fn execute_tool_calls_parallel(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
    emit: &AgentEventSink,
) -> ExecutedToolCallBatch {
    let mut entries: Vec<ParallelEntry> = Vec::new();

    for tool_call in tool_calls {
        emit_tool_execution_start(tool_call, emit).await;

        match prepare_tool_call(context, assistant_message, tool_call, config, signal).await {
            Preparation::Immediate { result, is_error } => {
                let finalized = FinalizedToolCallOutcome {
                    tool_call: tool_call.clone(),
                    result,
                    is_error,
                };
                emit_tool_execution_end(&finalized, emit).await;
                entries.push(ParallelEntry::Immediate(finalized));
                if is_aborted(signal) {
                    break;
                }
            }
            Preparation::Prepared(prepared) => {
                entries.push(ParallelEntry::Deferred(prepared));
                if is_aborted(signal) {
                    break;
                }
            }
        }
    }

    // Deferred executions run concurrently; slots keep assistant source order.
    let mut slots: Vec<Option<FinalizedToolCallOutcome>> = Vec::new();
    let mut handles: Vec<(usize, tokio::task::JoinHandle<FinalizedToolCallOutcome>)> = Vec::new();
    for entry in entries {
        match entry {
            ParallelEntry::Immediate(finalized) => slots.push(Some(finalized)),
            ParallelEntry::Deferred(prepared) => {
                let slot_index = slots.len();
                slots.push(None);
                let handle = spawn_prepared_execution(
                    *prepared,
                    context,
                    assistant_message,
                    config,
                    signal,
                    emit,
                );
                handles.push((slot_index, handle));
            }
        }
    }
    for (slot_index, handle) in handles {
        match handle.await {
            Ok(finalized) => slots[slot_index] = Some(finalized),
            // A tool that panicked instead of returning `Err` still owes the
            // transcript a tool result, upstream's caught-throw statement.
            Err(join_error) => {
                let finalized = FinalizedToolCallOutcome {
                    tool_call: tool_calls[slot_index].clone(),
                    result: error_tool_result(&join_error.to_string()),
                    is_error: true,
                };
                emit_tool_execution_end(&finalized, emit).await;
                slots[slot_index] = Some(finalized);
            }
        }
    }

    #[allow(
        clippy::expect_used,
        reason = "every deferred execution settles its slot: the spawned task fills it or the join-error branch does"
    )]
    let ordered_finalized_calls: Vec<FinalizedToolCallOutcome> = slots
        .into_iter()
        .map(|slot| slot.expect("every deferred execution settles its slot"))
        .collect();
    let mut messages: Vec<ToolResultMessage> = Vec::new();
    for finalized in &ordered_finalized_calls {
        let tool_result_message = create_tool_result_message(finalized);
        emit_tool_result_message(&tool_result_message, emit).await;
        messages.push(tool_result_message);
    }

    ExecutedToolCallBatch {
        messages,
        terminate: should_terminate_tool_batch(&ordered_finalized_calls),
    }
}

/// Run one prepared call to its finalized outcome on its own task: execute,
/// apply the `after_tool_call` override, and emit `tool_execution_end` as
/// soon as the call settles, upstream's deferred entry.
fn spawn_prepared_execution(
    prepared: PreparedToolCall,
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
    emit: &AgentEventSink,
) -> tokio::task::JoinHandle<FinalizedToolCallOutcome> {
    let context = context.clone();
    let assistant_message = assistant_message.clone();
    let config = config.clone();
    let signal = signal.cloned();
    let emit = emit.clone();
    tokio::spawn(async move {
        if is_aborted(signal.as_ref()) {
            let finalized = FinalizedToolCallOutcome {
                tool_call: prepared.tool_call.clone(),
                result: error_tool_result("Operation aborted"),
                is_error: true,
            };
            emit_tool_execution_end(&finalized, &emit).await;
            return finalized;
        }
        let executed = execute_prepared_tool_call(&prepared, signal.as_ref(), &emit).await;
        let finalized = finalize_executed_tool_call(
            &context,
            &assistant_message,
            &prepared,
            executed,
            &config,
            signal.as_ref(),
        )
        .await;
        emit_tool_execution_end(&finalized, &emit).await;
        finalized
    })
}

/// A tool call that cleared preflight and is ready to execute.
struct PreparedToolCall {
    tool_call: ToolCall,
    tool: AgentTool,
    args: serde_json::Value,
}

/// A tool call's outcome once its execution settled.
struct ExecutedToolCallOutcome {
    result: AgentToolResult,
    is_error: bool,
}

/// A tool call with its finalized result, ready for the transcript.
struct FinalizedToolCallOutcome {
    tool_call: ToolCall,
    result: AgentToolResult,
    is_error: bool,
}

/// What preparing a tool call produced: a call ready to execute, or an
/// immediate error outcome that skips execution. The prepared side is boxed:
/// the enum lives on every prepared call's transient path and the tool
/// surface (schema document, executor) is the batch's hot-path payload.
#[expect(
    clippy::large_enum_variant,
    reason = "the error outcome carries its full tool result by value, mirroring the upstream shape"
)]
enum Preparation {
    Prepared(Box<PreparedToolCall>),
    Immediate {
        result: AgentToolResult,
        is_error: bool,
    },
}

/// The batch early-termination rule: every finalized result in the batch
/// must hint termination.
fn should_terminate_tool_batch(finalized_calls: &[FinalizedToolCallOutcome]) -> bool {
    !finalized_calls.is_empty()
        && finalized_calls
            .iter()
            .all(|finalized| finalized.result.terminate == Some(true))
}

/// Apply the tool's pre-validation argument shim, upstream's
/// `prepareToolCallArguments`.
fn prepare_tool_call_arguments(
    tool: &AgentTool,
    tool_call: &ToolCall,
) -> Result<ToolCall, AgentToolError> {
    let Some(prepare_shim) = &tool.prepare_arguments else {
        return Ok(tool_call.clone());
    };
    let arguments = serde_json::Value::Object(tool_call.arguments.clone());
    let shimmed_arguments = prepare_shim(&arguments)?;
    // The shim contract requires an object that matches the tool's schema;
    // anything else is stated as the shim's own error.
    let Some(shimmed) = shimmed_arguments.as_object() else {
        return Err("prepareArguments must return an object".into());
    };
    Ok(ToolCall {
        arguments: shimmed.clone(),
        ..tool_call.clone()
    })
}

/// Prepare one tool call: resolve the tool, apply the argument shim, validate
/// arguments against the tool schema, and run the `before_tool_call` gate.
/// Failures at any stage are immediate error results.
async fn prepare_tool_call(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_call: &ToolCall,
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
) -> Preparation {
    let Some(tool) = context
        .tools
        .as_ref()
        .and_then(|tools| tools.iter().find(|tool| tool.name() == tool_call.name))
    else {
        return Preparation::Immediate {
            result: error_tool_result(&format!("Tool {} not found", tool_call.name)),
            is_error: true,
        };
    };

    let prepared_tool_call = match prepare_tool_call_arguments(tool, tool_call) {
        Ok(prepared) => prepared,
        Err(error) => {
            return Preparation::Immediate {
                result: error_tool_result(&error.to_string()),
                is_error: true,
            };
        }
    };
    let validated_args = match validate_tool_arguments(&tool.tool, &prepared_tool_call) {
        Ok(args) => args,
        Err(error) => {
            return Preparation::Immediate {
                result: error_tool_result(&error.to_string()),
                is_error: true,
            };
        }
    };
    if let Some(before_tool_call) = &config.before_tool_call {
        let before_result = before_tool_call(
            BeforeToolCallContext {
                assistant_message: assistant_message.clone(),
                tool_call: tool_call.clone(),
                args: validated_args.clone(),
                context: context.clone(),
            },
            signal.cloned(),
        )
        .await;
        if is_aborted(signal) {
            return Preparation::Immediate {
                result: error_tool_result("Operation aborted"),
                is_error: true,
            };
        }
        if let Some(before_result) = before_result
            && before_result.block == Some(true)
        {
            let mut result = error_tool_result(
                before_result
                    .reason
                    .as_deref()
                    .unwrap_or("Tool execution was blocked"),
            );
            if before_result.terminate == Some(true) {
                result.terminate = Some(true);
            }
            return Preparation::Immediate {
                result,
                is_error: true,
            };
        }
    }
    if is_aborted(signal) {
        return Preparation::Immediate {
            result: error_tool_result("Operation aborted"),
            is_error: true,
        };
    }
    Preparation::Prepared(Box::new(PreparedToolCall {
        tool_call: tool_call.clone(),
        tool: tool.clone(),
        args: validated_args,
    }))
}

/// Execute a prepared call, forwarding the tool's partial-result updates as
/// `tool_execution_update` events. Calls after the tool settles are ignored;
/// the accepted ones drain through the sink in emission order before the
/// outcome is returned, so the caller never finalizes over an undelivered
/// update.
async fn execute_prepared_tool_call(
    prepared: &PreparedToolCall,
    signal: Option<&CancellationToken>,
    emit: &AgentEventSink,
) -> ExecutedToolCallOutcome {
    let updates: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let accepting = Arc::new(AtomicBool::new(true));

    let on_update = {
        let updates = Arc::clone(&updates);
        let accepting = Arc::clone(&accepting);
        let tool_call_id = prepared.tool_call.id.clone();
        let tool_name = prepared.tool_call.name.clone();
        let args = raw_arguments(&prepared.tool_call);
        Arc::new(move |partial_result: &AgentToolResult| {
            if !accepting.load(Ordering::Relaxed) {
                return;
            }
            updates.lock().unwrap_or_else(PoisonError::into_inner).push(
                AgentEvent::ToolExecutionUpdate {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    args: args.clone(),
                    partial_result: partial_result.clone(),
                },
            );
        })
    };

    let executed = (prepared.tool.execute)(
        &prepared.tool_call.id,
        &prepared.args,
        signal,
        Some(on_update.as_ref()),
    )
    .await;
    accepting.store(false, Ordering::Relaxed);
    let outcome = match executed {
        Ok(result) => ExecutedToolCallOutcome {
            result,
            is_error: false,
        },
        Err(error) => ExecutedToolCallOutcome {
            result: error_tool_result(&error.to_string()),
            is_error: true,
        },
    };
    let update_events: Vec<AgentEvent> =
        std::mem::take(&mut *updates.lock().unwrap_or_else(PoisonError::into_inner));
    for event in update_events {
        emit_event(emit, event).await;
    }
    outcome
}

/// Apply the `after_tool_call` override pass to an executed call, upstream's
/// `finalizeExecutedToolCall`. Each `Some` field replaces the executed value
/// in full; the hook's contract forbids failure, so there is no error path.
async fn finalize_executed_tool_call(
    context: &AgentContext,
    assistant_message: &AssistantMessage,
    prepared: &PreparedToolCall,
    executed: ExecutedToolCallOutcome,
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
) -> FinalizedToolCallOutcome {
    let mut result = executed.result;
    let mut is_error = executed.is_error;

    if let Some(after_tool_call) = &config.after_tool_call
        && let Some(after_result) = after_tool_call(
            AfterToolCallContext {
                assistant_message: assistant_message.clone(),
                tool_call: prepared.tool_call.clone(),
                args: prepared.args.clone(),
                result: result.clone(),
                is_error,
                context: context.clone(),
            },
            signal.cloned(),
        )
        .await
    {
        if let Some(content) = after_result.content {
            result.content = content;
        }
        if let Some(details) = after_result.details {
            result.details = details;
        }
        if let Some(usage) = after_result.usage {
            result.usage = Some(usage);
        }
        if let Some(terminate) = after_result.terminate {
            result.terminate = Some(terminate);
        }
        if let Some(after_is_error) = after_result.is_error {
            is_error = after_is_error;
        }
    }

    FinalizedToolCallOutcome {
        tool_call: prepared.tool_call.clone(),
        result,
        is_error,
    }
}

/// The loop's error tool result, upstream's `createErrorToolResult`.
fn error_tool_result(message: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![AgentToolContent::Text(pi_ai::types::TextContent {
            text: message.to_owned(),
            text_signature: None,
        })],
        details: serde_json::json!({}),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

/// Whether the run's abort token is cancelled.
fn is_aborted(signal: Option<&CancellationToken>) -> bool {
    signal.is_some_and(CancellationToken::is_cancelled)
}

/// Emit `tool_execution_end` for a finalized call.
async fn emit_tool_execution_end(finalized: &FinalizedToolCallOutcome, emit: &AgentEventSink) {
    emit_event(
        emit,
        AgentEvent::ToolExecutionEnd {
            tool_call_id: finalized.tool_call.id.clone(),
            tool_name: finalized.tool_call.name.clone(),
            result: finalized.result.clone(),
            is_error: finalized.is_error,
        },
    )
    .await;
}

/// The tool-result message a finalized call produces, upstream's
/// `createToolResultMessage`. Untyped tools can return results without
/// content or details; the shapes normalize so no null enters session
/// history or provider payloads, and an empty `addedToolNames` omits the
/// field.
fn create_tool_result_message(finalized: &FinalizedToolCallOutcome) -> ToolResultMessage {
    let result = &finalized.result;
    ToolResultMessage {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        content: result.content.clone(),
        details: (!result.details.is_null()).then(|| result.details.clone()),
        usage: result.usage,
        added_tool_names: result
            .added_tool_names
            .as_ref()
            .filter(|names| !names.is_empty())
            .cloned(),
        is_error: finalized.is_error,
        timestamp: now_ms(),
    }
}

/// Emit a tool result's transcript entry, upstream's `emitToolResultMessage`.
async fn emit_tool_result_message(tool_result_message: &ToolResultMessage, emit: &AgentEventSink) {
    let message = AgentMessage::Standard(Message::ToolResult(tool_result_message.clone()));
    emit_event(
        emit,
        AgentEvent::MessageStart {
            message: message.clone(),
        },
    )
    .await;
    emit_event(emit, AgentEvent::MessageEnd { message }).await;
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "the tests pin outcomes; an unexpected result panics the test by design"
    )]

    use super::*;
    use crate::types::AgentLoopTurnUpdate;

    /// A minimal assistant message for the pure-helper assertions; the
    /// provider shapes are arbitrary, only the struct shape matters.
    fn unit_assistant_message() -> AssistantMessage {
        serde_json::from_value(serde_json::json!({
            "content": [],
            "api": "google-generative-ai",
            "provider": "google",
            "model": "unit",
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 },
            },
            "stopReason": "pending",
            "timestamp": 0,
        }))
        .expect("a well-formed assistant message literal")
    }

    fn unit_config() -> AgentLoopConfig {
        AgentLoopConfig {
            stream_options: SimpleStreamOptions {
                api_key: Some("unit-key".to_string()),
                ..SimpleStreamOptions::default()
            },
            model: unit_model("unit"),
            convert_to_llm: Arc::new(|_messages| Box::pin(async move { Vec::new() })),
            transform_context: None,
            get_api_key: None,
            should_stop_after_turn: None,
            prepare_next_turn: None,
            get_steering_messages: None,
            get_follow_up_messages: None,
            tool_execution: None,
            before_tool_call: None,
            after_tool_call: None,
        }
    }

    fn unit_model(id: &str) -> pi_ai::types::Model {
        serde_json::from_value(serde_json::json!({
            "id": id,
            "name": id,
            "api": "google-generative-ai",
            "provider": "google",
            "baseUrl": "https://unit.example",
            "reasoning": true,
            "input": ["text", "image"],
            "cost": { "input": 1.0, "output": 2.0, "cacheRead": 0.5, "cacheWrite": 3.0 },
            "contextWindow": 4096,
            "maxTokens": 1024,
        }))
        .expect("a well-formed model literal")
    }

    #[test]
    fn stream_reasoning_maps_every_agent_level_to_the_request_reasoning() {
        // The agent level adds `off`, which the request carries as an absent
        // field; every other level passes through.
        let expected = [
            (ThinkingLevel::Off, None),
            (
                ThinkingLevel::Minimal,
                Some(pi_ai::types::ThinkingLevel::Minimal),
            ),
            (ThinkingLevel::Low, Some(pi_ai::types::ThinkingLevel::Low)),
            (
                ThinkingLevel::Medium,
                Some(pi_ai::types::ThinkingLevel::Medium),
            ),
            (ThinkingLevel::High, Some(pi_ai::types::ThinkingLevel::High)),
            (
                ThinkingLevel::Xhigh,
                Some(pi_ai::types::ThinkingLevel::Xhigh),
            ),
            (ThinkingLevel::Max, Some(pi_ai::types::ThinkingLevel::Max)),
        ];
        for (level, reasoning) in expected {
            assert_eq!(stream_reasoning(level), reasoning);
        }
    }

    #[test]
    fn apply_turn_update_replaces_the_presented_fields_and_keeps_the_absent_ones() {
        let mut context = AgentContext {
            system_prompt: "first".to_string(),
            messages: Vec::new(),
            tools: None,
        };
        let mut config = unit_config();
        let next_context = AgentContext {
            system_prompt: "second".to_string(),
            messages: Vec::new(),
            tools: None,
        };
        apply_turn_update(
            &mut context,
            &mut config,
            AgentLoopTurnUpdate {
                context: Some(next_context),
                model: Some(unit_model("other")),
                thinking_level: Some(ThinkingLevel::Low),
            },
        );
        assert_eq!(context.system_prompt, "second");
        assert_eq!(config.model.id, "other");
        assert_eq!(
            config.stream_options.reasoning,
            Some(pi_ai::types::ThinkingLevel::Low)
        );

        // An all-default update keeps every live value.
        apply_turn_update(&mut context, &mut config, AgentLoopTurnUpdate::default());
        assert_eq!(context.system_prompt, "second");
        assert_eq!(config.model.id, "other");
        assert_eq!(
            config.stream_options.reasoning,
            Some(pi_ai::types::ThinkingLevel::Low)
        );
    }

    #[test]
    fn event_partial_answers_the_live_accumulator_and_none_for_the_terminal_events() {
        let partial = unit_assistant_message();
        assert!(
            event_partial(&AssistantMessageEvent::Start {
                partial: partial.clone(),
            })
            .is_some()
        );
        assert!(
            event_partial(&AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "x".to_string(),
                partial: partial.clone(),
            })
            .is_some()
        );
        assert!(
            event_partial(&AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message: partial.clone(),
            })
            .is_none()
        );
        assert!(
            event_partial(&AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: partial,
            })
            .is_none()
        );
    }
}

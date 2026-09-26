//! The stateful `Agent` wrapper, upstream's `src/agent.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`Agent`] owns the current transcript, emits lifecycle events, executes
//! tools through the low-level loop, and exposes queueing APIs for steering
//! and follow-up messages. The settlement contract carries over verbatim:
//! `agent_end` is the final emitted event for a run, but the agent does not
//! become idle until every awaited listener for that event has settled and
//! the run's runtime state is cleared.
//!
//! Porting restatements:
//! - the mutable state, the listener registry, the two queues, and the
//!   active run live behind locks; every critical section is synchronous and
//!   no lock is held across an await, the single event loop's interleaving
//!   restated;
//! - listeners are dispatched from a snapshot taken at emission time, in
//!   registration order — upstream iterates its live `Set`;
//! - upstream's `AbortController` is the workspace cancellation token;
//! - a run is owned by a spawned lifecycle task, so an un-awaited `prompt`
//!   future still completes the run's settlement (`handleRunFailure` and
//!   `finishRun` run inside the task), and the executor runs on a further
//!   nested task so a panicking closure unwinds into the failure lifecycle
//!   instead of taking the caller down;
//! - a run failure resolves the prompt: the failed turn lands in the
//!   transcript with stop reason `error` or `aborted`, the full lifecycle
//!   (`message_start` through `agent_end`) is emitted, and
//!   `state.error_message` carries the message — upstream's
//!   caught-and-handled rejection;
//! - a panicking closure (a stream fn or hook violating its must-not-throw
//!   contract, upstream's synchronous throw) reaches the same failure
//!   lifecycle with the panic payload as the error message;
//! - a stream function omitted at construction resolves through the
//!   process-global default at run start (the agent-loop child's precedent),
//!   so a missing default surfaces through the run's failure lifecycle
//!   instead of the constructor;
//! - `continue` restates as `continue_run` and `followUp` as `follow_up`;
//! - the `state.tools`/`state.messages` copy-on-assign accessors restate as
//!   [`Agent::set_tools`] and [`Agent::set_messages`], which clone the
//!   provided slice — caller-side mutation of the original never reaches the
//!   runtime, and vice versa;
//! - `waitForIdle` waits on a generation counter the finishing run bumps
//!   after clearing the active run, upstream's per-run promise;
//! - the option hooks that upstream types with the abort signal
//!   (`shouldStopAfterTurn`, `prepareNextTurn`,
//!   `prepareNextTurnWithContext`) keep that shape; the loop's signal-less
//!   shapes are wrapped with the active run's token at call time, upstream's
//!   `this.signal` reads.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use pi_ai::auth::resolve::now_ms;
use pi_ai::types::Api;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::Message;
use pi_ai::types::Model;
use pi_ai::types::ModelCost;
use pi_ai::types::OnPayload;
use pi_ai::types::OnResponse;
use pi_ai::types::ProviderId;
use pi_ai::types::SimpleStreamOptions;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::types::ThinkingBudgets;
use pi_ai::types::Transport;
use pi_ai::types::TransportOptions;
use pi_ai::types::Usage;
use pi_ai::types::UserBlock;
use pi_ai::types::UserContent;
use pi_ai::types::UserMessage;
use tokio::sync::watch;
use tokio::task::JoinError;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::agent_loop::AgentEventSink;
use crate::agent_loop::AgentLoopError;
use crate::agent_loop::run_agent_loop;
use crate::agent_loop::run_agent_loop_continue;
use crate::agent_loop::stream_reasoning;
use crate::types::AfterToolCall;
use crate::types::AgentContext;
use crate::types::AgentEvent;
use crate::types::AgentListener;
use crate::types::AgentLoopConfig;
use crate::types::AgentMessage;
use crate::types::AgentPrepareNextTurn;
use crate::types::AgentPrepareNextTurnWithContext;
use crate::types::AgentShouldStopAfterTurn;
use crate::types::AgentState;
use crate::types::AgentTool;
use crate::types::BeforeToolCall;
use crate::types::BoxedFuture;
use crate::types::ConvertToLlm;
use crate::types::GetApiKey;
use crate::types::GetFollowUpMessages;
use crate::types::GetSteeringMessages;
use crate::types::PrepareNextTurn;
use crate::types::PrepareNextTurnContext;
use crate::types::PromptInput;
use crate::types::QueueMode;
use crate::types::ShouldStopAfterTurn;
use crate::types::ShouldStopAfterTurnContext;
use crate::types::StreamFn;
use crate::types::ThinkingLevel;
use crate::types::ToolExecutionMode;
use crate::types::TransformContext;

/// The unknown default model, upstream's `DEFAULT_MODEL`.
fn unknown_model() -> Model {
    Model {
        id: "unknown".to_owned(),
        name: "unknown".to_owned(),
        api: Api::from("unknown"),
        provider: ProviderId::from("unknown"),
        base_url: String::new(),
        reasoning: false,
        thinking_level_map: None,
        input: Vec::new(),
        cost: ModelCost::default(),
        context_window: 0,
        max_tokens: 0,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The default agent-message to LLM-message converter, upstream's
/// `defaultConvertToLlm`: keep user, assistant, and tool-result messages;
/// drop everything else (UI-only custom messages included).
fn default_convert_to_llm(messages: Vec<AgentMessage>) -> Vec<Message> {
    messages
        .into_iter()
        .filter_map(|message| match message {
            AgentMessage::Standard(
                standard @ (Message::User(_) | Message::Assistant(_) | Message::ToolResult(_)),
            ) => Some(standard),
            _ => None,
        })
        .collect()
}

/// The error the `Agent`'s entry points surface, upstream's thrown `Error`s.
#[derive(Debug)]
pub enum AgentError {
    /// `prompt` while a run is active.
    PromptWhileStreaming,
    /// `continue` while a run is active.
    ContinueWhileStreaming,
    /// `reset` while a run is active.
    ResetWhileStreaming,
    /// `continue` on an empty transcript.
    NoMessagesToContinueFrom,
    /// `continue` on an assistant tail with both queues empty.
    ContinueFromAssistant,
    /// A failure escaped the run's own failure path — a listener that
    /// panicked while the failure lifecycle was being emitted, upstream's
    /// rejection escaping `handleRunFailure`. Carries the panic payload.
    RunFailed(String),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // The message texts are the upstream contract verbatim.
            Self::PromptWhileStreaming => f.write_str(
                "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion.",
            ),
            Self::ContinueWhileStreaming => {
                f.write_str("Agent is already processing. Wait for completion before continuing.")
            }
            Self::ResetWhileStreaming => {
                f.write_str("Agent is already processing. Wait for completion before resetting.")
            }
            Self::NoMessagesToContinueFrom => f.write_str("No messages to continue from"),
            Self::ContinueFromAssistant => {
                f.write_str("Cannot continue from message role: assistant")
            }
            Self::RunFailed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for AgentError {}

/// The initial state the `Agent` options carry, upstream's `initialState`
/// partial: absent fields take the constructor's defaults (empty prompt, the
/// unknown model, thinking level `Off`).
#[derive(Clone, Debug, Default)]
pub struct AgentInitialState {
    /// System prompt sent with each model request.
    pub system_prompt: String,
    /// Active model used for future turns. `None`: the unknown default
    /// model.
    pub model: Option<Model>,
    /// Requested reasoning level for future turns. `None`: `Off`.
    pub thinking_level: Option<ThinkingLevel>,
    /// Available tools.
    pub tools: Vec<AgentTool>,
    /// Conversation transcript.
    pub messages: Vec<AgentMessage>,
}

/// Options for constructing an [`Agent`], upstream's `AgentOptions`; every
/// absent field takes upstream's default.
#[derive(Clone, Default)]
pub struct AgentOptions {
    /// The state the agent starts with.
    pub initial_state: Option<AgentInitialState>,
    /// The agent-message to LLM-message conversion. Default: keep user,
    /// assistant, and tool-result messages, drop the rest.
    pub convert_to_llm: Option<ConvertToLlm>,
    /// Optional transform applied to the context before `convert_to_llm`.
    pub transform_context: Option<TransformContext>,
    /// The stream function. `None` resolves through the process-global
    /// default at run start; a missing default surfaces through the run's
    /// failure lifecycle, not the constructor.
    pub stream_fn: Option<StreamFn>,
    /// Dynamic API-key resolution for each LLM call.
    pub get_api_key: Option<GetApiKey>,
    /// Request-payload hook forwarded on every provider call.
    pub on_payload: Option<OnPayload>,
    /// Response hook forwarded on every provider call.
    pub on_response: Option<OnResponse>,
    /// Pre-execution gate for each tool call.
    pub before_tool_call: Option<BeforeToolCall>,
    /// Post-execution override pass for each tool result.
    pub after_tool_call: Option<AfterToolCall>,
    /// Graceful stop after the current turn; receives the run's token.
    pub should_stop_after_turn: Option<AgentShouldStopAfterTurn>,
    /// Signal-only turn preparation; ignored when
    /// `prepare_next_turn_with_context` is set.
    pub prepare_next_turn: Option<AgentPrepareNextTurn>,
    /// Context-aware turn preparation; wins over `prepare_next_turn` when
    /// both are set, upstream-verbatim.
    pub prepare_next_turn_with_context: Option<AgentPrepareNextTurnWithContext>,
    /// How queued steering messages drain. Default: `OneAtATime`.
    pub steering_mode: Option<QueueMode>,
    /// How queued follow-up messages drain. Default: `OneAtATime`.
    pub follow_up_mode: Option<QueueMode>,
    /// Session identifier forwarded to providers for cache-aware backends.
    pub session_id: Option<String>,
    /// Optional per-level thinking token budgets forwarded to the stream
    /// function.
    pub thinking_budgets: Option<ThinkingBudgets>,
    /// Preferred transport forwarded to the stream function. Default:
    /// `Auto`.
    pub transport: Option<Transport>,
    /// Optional cap for provider-requested retry delays.
    pub max_retry_delay_ms: Option<u64>,
    /// Tool execution strategy for assistant messages with multiple tool
    /// calls. Default: `Parallel`.
    pub tool_execution: Option<ToolExecutionMode>,
}

impl std::fmt::Debug for AgentOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The hook closures do not debug; the option struct is request
        // plumbing, so the non-exhaustive finish covers it.
        f.debug_struct("AgentOptions").finish_non_exhaustive()
    }
}

/// One registered listener plus its unsubscribe identity; the token's
/// pointer is the membership the returned unsubscribe closure removes, the
/// listener `Set` restated as a registration-ordered list.
struct ListenerEntry {
    token: Arc<()>,
    listener: AgentListener,
}

/// The queued steering/follow-up message store, upstream's
/// `PendingMessageQueue`.
struct PendingMessageQueue {
    messages: Vec<AgentMessage>,
    mode: QueueMode,
}

impl PendingMessageQueue {
    const fn new(mode: QueueMode) -> Self {
        Self {
            messages: Vec::new(),
            mode,
        }
    }

    fn enqueue(&mut self, message: AgentMessage) {
        self.messages.push(message);
    }

    const fn has_items(&self) -> bool {
        !self.messages.is_empty()
    }

    fn drain(&mut self) -> Vec<AgentMessage> {
        match self.mode {
            QueueMode::All => std::mem::take(&mut self.messages),
            QueueMode::OneAtATime => {
                if self.messages.is_empty() {
                    Vec::new()
                } else {
                    vec![self.messages.remove(0)]
                }
            }
        }
    }

    fn clear(&mut self) {
        self.messages.clear();
    }
}

/// The run currently executing, upstream's `ActiveRun`. Its waiters settle
/// through the idle generation watch, not a per-run handle.
struct ActiveRun {
    token: CancellationToken,
}

/// The `Agent` runtime shared by every handle clone. All fields are plain
/// data or closures; every mutable part sits behind a lock whose critical
/// sections never await.
struct AgentInner {
    state: Mutex<AgentState>,
    listeners: Mutex<Vec<ListenerEntry>>,
    steering_queue: Mutex<PendingMessageQueue>,
    follow_up_queue: Mutex<PendingMessageQueue>,
    active_run: Mutex<Option<ActiveRun>>,
    /// Bumped by every finishing run after the active run clears;
    /// `wait_for_idle` waits for the change, upstream's per-run promise.
    idle_tx: watch::Sender<u64>,
    session_id: Mutex<Option<String>>,

    convert_to_llm: ConvertToLlm,
    transform_context: Option<TransformContext>,
    stream_function: Option<StreamFn>,
    get_api_key: Option<GetApiKey>,
    on_payload: Option<OnPayload>,
    on_response: Option<OnResponse>,
    before_tool_call: Option<BeforeToolCall>,
    after_tool_call: Option<AfterToolCall>,
    should_stop_after_turn: Option<AgentShouldStopAfterTurn>,
    prepare_next_turn: Option<AgentPrepareNextTurn>,
    prepare_next_turn_with_context: Option<AgentPrepareNextTurnWithContext>,
    thinking_budgets: Option<ThinkingBudgets>,
    transport: Transport,
    max_retry_delay_ms: Option<u64>,
    tool_execution: ToolExecutionMode,
}

/// Which loop entry a run drives, plus the prompt messages and the steering
/// skip flag the entry carries, upstream's `runPromptMessages` options.
enum RunKind {
    /// `run_agent_loop` with the given messages; the flag restates upstream's
    /// `skipInitialSteeringPoll` (the `continue` steering-tail path).
    Prompt(Vec<AgentMessage>, bool),
    /// `run_agent_loop_continue` from the current transcript.
    Continue,
}

/// Stateful wrapper around the low-level agent loop.
///
/// `Agent` owns the current transcript, emits lifecycle events, executes
/// tools, and exposes queueing APIs for steering and follow-up messages.
/// Upstream runs on one JavaScript event loop; the port shares the runtime
/// through the handle ([`Clone`]), so prompts, aborts, and queueing
/// interleave the way the event loop's scheduling did.
#[derive(Clone)]
pub struct Agent(Arc<AgentInner>);

impl Agent {
    /// Build an agent from `options`.
    #[must_use]
    pub fn new(options: AgentOptions) -> Self {
        let initial = options.initial_state.unwrap_or_default();
        let state = AgentState {
            system_prompt: initial.system_prompt,
            model: initial.model.unwrap_or_else(unknown_model),
            thinking_level: initial.thinking_level.unwrap_or(ThinkingLevel::Off),
            tools: initial.tools,
            messages: initial.messages,
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: BTreeSet::new(),
            error_message: None,
        };
        let (idle_tx, _idle_rx) = watch::channel(0);
        Self(Arc::new(AgentInner {
            state: Mutex::new(state),
            listeners: Mutex::new(Vec::new()),
            steering_queue: Mutex::new(PendingMessageQueue::new(
                options.steering_mode.unwrap_or(QueueMode::OneAtATime),
            )),
            follow_up_queue: Mutex::new(PendingMessageQueue::new(
                options.follow_up_mode.unwrap_or(QueueMode::OneAtATime),
            )),
            active_run: Mutex::new(None),
            idle_tx,
            session_id: Mutex::new(options.session_id),
            convert_to_llm: options.convert_to_llm.unwrap_or_else(|| {
                Arc::new(|messages| Box::pin(async move { default_convert_to_llm(messages) }))
            }),
            transform_context: options.transform_context,
            stream_function: options.stream_fn,
            get_api_key: options.get_api_key,
            on_payload: options.on_payload,
            on_response: options.on_response,
            before_tool_call: options.before_tool_call,
            after_tool_call: options.after_tool_call,
            should_stop_after_turn: options.should_stop_after_turn,
            prepare_next_turn: options.prepare_next_turn,
            prepare_next_turn_with_context: options.prepare_next_turn_with_context,
            thinking_budgets: options.thinking_budgets,
            transport: options.transport.unwrap_or(Transport::Auto),
            max_retry_delay_ms: options.max_retry_delay_ms,
            tool_execution: options
                .tool_execution
                .unwrap_or(ToolExecutionMode::Parallel),
        }))
    }

    /// Subscribe to agent lifecycle events.
    ///
    /// Listener promises are awaited in subscription order and are included
    /// in the current run's settlement. Listeners also receive the active
    /// run's cancellation token. `agent_end` is the final emitted event for
    /// a run, but the agent does not become idle until all awaited listeners
    /// for that event have settled. Returns the unsubscribe closure.
    pub fn subscribe(&self, listener: AgentListener) -> impl FnOnce() {
        let token = Arc::new(());
        self.0
            .listeners
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(ListenerEntry {
                token: Arc::clone(&token),
                listener,
            });
        let inner = Arc::clone(&self.0);
        move || {
            inner
                .listeners
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .retain(|entry| !Arc::ptr_eq(&entry.token, &token));
        }
    }

    /// Current agent state, the live view behind the lock. Do not hold the
    /// guard across an await.
    pub fn state(&self) -> MutexGuard<'_, AgentState> {
        self.0.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Set the system prompt sent with each model request.
    pub fn set_system_prompt(&self, system_prompt: impl Into<String>) {
        self.0
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .system_prompt = system_prompt.into();
    }

    /// Set the active model used for future turns.
    pub fn set_model(&self, model: Model) {
        self.0
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .model = model;
    }

    /// Set the requested reasoning level for future turns.
    pub fn set_thinking_level(&self, thinking_level: ThinkingLevel) {
        self.0
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .thinking_level = thinking_level;
    }

    /// Replace the available tools. The collection is cloned into storage —
    /// the copy-on-assign statement: caller-side mutation of the original
    /// never reaches the runtime, and vice versa.
    pub fn set_tools(&self, tools: &[AgentTool]) {
        self.0
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .tools = tools.to_vec();
    }

    /// Replace the conversation transcript. The collection is cloned into
    /// storage — the same copy-on-assign statement as [`Agent::set_tools`].
    pub fn set_messages(&self, messages: &[AgentMessage]) {
        self.0
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .messages = messages.to_vec();
    }

    /// Controls how queued steering messages drain.
    #[must_use]
    pub fn steering_mode(&self) -> QueueMode {
        self.0
            .steering_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .mode
    }

    /// Controls how queued steering messages drain.
    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.0
            .steering_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .mode = mode;
    }

    /// Controls how queued follow-up messages drain.
    #[must_use]
    pub fn follow_up_mode(&self) -> QueueMode {
        self.0
            .follow_up_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .mode
    }

    /// Controls how queued follow-up messages drain.
    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.0
            .follow_up_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .mode = mode;
    }

    /// Queue a message to be injected after the current assistant turn
    /// finishes.
    pub fn steer(&self, message: AgentMessage) {
        self.0
            .steering_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .enqueue(message);
    }

    /// Queue a message to run only after the agent would otherwise stop.
    pub fn follow_up(&self, message: AgentMessage) {
        self.0
            .follow_up_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .enqueue(message);
    }

    /// Remove all queued steering messages.
    pub fn clear_steering_queue(&self) {
        self.0
            .steering_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    /// Remove all queued follow-up messages.
    pub fn clear_follow_up_queue(&self) {
        self.0
            .follow_up_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    /// Remove all queued steering and follow-up messages.
    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    /// Returns true when either queue still contains pending messages.
    #[must_use]
    pub fn has_queued_messages(&self) -> bool {
        self.0
            .steering_queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .has_items()
            || self
                .0
                .follow_up_queue
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .has_items()
    }

    /// The session identifier forwarded to providers, when one is set.
    #[must_use]
    pub fn session_id(&self) -> Option<String> {
        self.0
            .session_id
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Set the session identifier; `None` clears it.
    pub fn set_session_id(&self, session_id: Option<String>) {
        *self
            .0
            .session_id
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = session_id;
    }

    /// Active cancellation token for the current run, if any.
    #[must_use]
    pub fn signal(&self) -> Option<CancellationToken> {
        self.0.current_token()
    }

    /// Abort the current run, if one is active.
    pub fn abort(&self) {
        if let Some(run) = self
            .0
            .active_run
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
        {
            run.token.cancel();
        }
    }

    /// Resolve when the current run and all awaited event listeners have
    /// finished. This resolves after the `agent_end` listeners settle.
    /// Immediate when no run is active.
    #[must_use]
    pub fn wait_for_idle(&self) -> BoxedFuture<'static, ()> {
        let inner = Arc::clone(&self.0);
        Box::pin(async move {
            let mut rx = inner.idle_tx.subscribe();
            let _seen = *rx.borrow_and_update();
            loop {
                let active = inner
                    .active_run
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .is_some();
                if !active {
                    return;
                }
                if rx.changed().await.is_err() {
                    return;
                }
            }
        })
    }

    /// Clear transcript state, runtime state, and queued messages.
    ///
    /// # Errors
    /// [`AgentError::ResetWhileStreaming`] when a run is active; the
    /// transcript and queues are left untouched, upstream-verbatim.
    pub fn reset(&self) -> Result<(), AgentError> {
        if self
            .0
            .active_run
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
        {
            return Err(AgentError::ResetWhileStreaming);
        }
        {
            let mut state = self.0.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.messages = Vec::new();
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls = BTreeSet::new();
            state.error_message = None;
        }
        self.clear_follow_up_queue();
        self.clear_steering_queue();
        Ok(())
    }

    /// Start a new prompt from text, a single message, or a batch of
    /// messages.
    ///
    /// # Errors
    /// [`AgentError::PromptWhileStreaming`] when a run is already active —
    /// use `steer` or `follow_up` to queue messages, or wait for completion.
    /// A run that fails (an unresolvable stream function, a panicking
    /// closure) still resolves the prompt: the failure lifecycle is emitted
    /// and the failed turn lands in the transcript.
    ///
    /// # Panics
    /// Called outside a tokio runtime context: the run is a spawned task.
    pub fn prompt(
        &self,
        input: impl Into<PromptInput>,
    ) -> BoxedFuture<'static, Result<(), AgentError>> {
        let inner = Arc::clone(&self.0);
        let messages = normalize_prompt_input(input.into());
        Box::pin(async move {
            let token = inner.register_run(AgentError::PromptWhileStreaming)?;
            let handle = tokio::spawn(run_lifecycle(
                inner,
                RunKind::Prompt(messages, false),
                token,
            ));
            finish_join(handle).await
        })
    }

    /// Continue from the current transcript. The last message must not be an
    /// assistant message: on an assistant tail the drained steering queue
    /// (then the follow-up queue) becomes the prompt instead, upstream-
    /// verbatim, one-at-a-time semantics included.
    ///
    /// # Errors
    /// [`AgentError::ContinueWhileStreaming`] when a run is active,
    /// [`AgentError::NoMessagesToContinueFrom`] on an empty transcript,
    /// [`AgentError::ContinueFromAssistant`] on an assistant tail with both
    /// queues empty.
    ///
    /// # Panics
    /// Called outside a tokio runtime context.
    #[must_use]
    pub fn continue_run(&self) -> BoxedFuture<'static, Result<(), AgentError>> {
        let inner = Arc::clone(&self.0);
        Box::pin(async move {
            let run = {
                let last = inner
                    .state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .messages
                    .last()
                    .cloned();
                let Some(last) = last else {
                    return Err(AgentError::NoMessagesToContinueFrom);
                };
                if matches!(last, AgentMessage::Standard(Message::Assistant(_))) {
                    let steering = inner
                        .steering_queue
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .drain();
                    if steering.is_empty() {
                        let follow_ups = inner
                            .follow_up_queue
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .drain();
                        if follow_ups.is_empty() {
                            return Err(AgentError::ContinueFromAssistant);
                        }
                        RunKind::Prompt(follow_ups, false)
                    } else {
                        RunKind::Prompt(steering, true)
                    }
                } else {
                    RunKind::Continue
                }
            };
            let token = inner.register_run(AgentError::ContinueWhileStreaming)?;
            let handle = tokio::spawn(run_lifecycle(inner, run, token));
            finish_join(handle).await
        })
    }
}

/// The failure an event delivery hits when no run is active: the sink a run
/// reports through only exists inside a registered run, so this path is a
/// caller bug. The message text is the upstream contract verbatim.
#[expect(
    clippy::panic,
    reason = "the loop's sink only exists inside a registered run; this path is a caller bug, not a runtime condition"
)]
fn listener_outside_run() -> ! {
    panic!("Agent listener invoked outside active run")
}

impl AgentInner {
    /// The active run's cancellation token, if any.
    fn current_token(&self) -> Option<CancellationToken> {
        self.active_run
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map(|run| run.token.clone())
    }

    /// Register a run, upstream's `runWithLifecycle` entry: claim the
    /// active-run slot and set the runtime state flags in one section.
    /// `busy_error` selects which already-processing error the entry point
    /// reports.
    fn register_run(&self, busy_error: AgentError) -> Result<CancellationToken, AgentError> {
        let token = {
            let mut active = self
                .active_run
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if active.is_some() {
                return Err(busy_error);
            }
            let token = CancellationToken::new();
            *active = Some(ActiveRun {
                token: token.clone(),
            });
            drop(active);
            token
        };
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.is_streaming = true;
        state.streaming_message = None;
        state.error_message = None;
        drop(state);
        Ok(token)
    }

    /// Reduce internal state for a loop event, then await listeners.
    ///
    /// `agent_end` only means no further loop events will be emitted. The
    /// run is considered idle later, after all awaited listeners for
    /// `agent_end` finish and `finish_run` clears runtime-owned state.
    async fn process_event(&self, event: AgentEvent) {
        {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            match &event {
                AgentEvent::MessageStart { message }
                | AgentEvent::MessageUpdate { message, .. } => {
                    state.streaming_message = Some(message.clone());
                }
                AgentEvent::MessageEnd { message } => {
                    state.streaming_message = None;
                    state.messages.push(message.clone());
                }
                AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                    state.pending_tool_calls.insert(tool_call_id.clone());
                }
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                    state.pending_tool_calls.remove(tool_call_id);
                }
                AgentEvent::TurnEnd { message, .. } => {
                    if let AgentMessage::Standard(Message::Assistant(assistant)) = message
                        && let Some(error) = &assistant.error_message
                    {
                        state.error_message = Some(error.clone());
                    }
                }
                AgentEvent::AgentEnd { .. } => {
                    state.streaming_message = None;
                }
                AgentEvent::AgentStart
                | AgentEvent::TurnStart
                | AgentEvent::ToolExecutionUpdate { .. } => {}
            }
        }

        let token: CancellationToken = {
            let Some(run) = self
                .active_run
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .as_ref()
                .map(|run| run.token.clone())
            else {
                listener_outside_run();
            };
            run
        };
        let listeners: Vec<AgentListener> = self
            .listeners
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|entry| Arc::clone(&entry.listener))
            .collect();
        for listener in listeners {
            (listener)(event.clone(), token.clone()).await;
        }
    }

    /// Clear transcript-runtime state and settle the run's waiters, upstream's
    /// `finishRun`: the run slot clears before the idle generation bumps, so
    /// a woken waiter observes the agent idle.
    fn finish_run(&self) {
        {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls = BTreeSet::new();
        }
        let _run = self
            .active_run
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let generation = *self.idle_tx.borrow() + 1;
        let _ = self.idle_tx.send(generation);
    }

    /// The context snapshot a run starts from, upstream's
    /// `createContextSnapshot`.
    fn create_context_snapshot(&self) -> AgentContext {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        AgentContext {
            system_prompt: state.system_prompt.clone(),
            messages: state.messages.clone(),
            tools: Some(state.tools.clone()),
        }
    }

    /// The loop configuration a run starts from, upstream's
    /// `createLoopConfig`. The steering hook restates upstream's captured
    /// `skipInitialSteeringPoll` mutable as a shared flag consumed on the
    /// first poll; the signal-taking option hooks are wrapped onto the
    /// loop's signal-less shapes with the active run's token, upstream's
    /// `this.signal` reads.
    fn create_loop_config(self: &Arc<Self>, skip_initial_steering_poll: bool) -> AgentLoopConfig {
        let (model, reasoning) = {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            (state.model.clone(), stream_reasoning(state.thinking_level))
        };
        let session_id = self
            .session_id
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();

        let get_steering_messages = self.steering_hook(skip_initial_steering_poll);
        let get_follow_up_messages = self.follow_up_hook();
        let should_stop_after_turn = self.should_stop_hook();
        let prepare_next_turn = self.prepare_next_turn_hook();

        AgentLoopConfig {
            stream_options: SimpleStreamOptions {
                transport_options: TransportOptions {
                    signal: None,
                    on_payload: self.on_payload.clone(),
                    on_response: self.on_response.clone(),
                    http_client: None,
                },
                reasoning,
                session_id,
                transport: Some(self.transport),
                thinking_budgets: self.thinking_budgets,
                max_retry_delay_ms: self.max_retry_delay_ms,
                ..SimpleStreamOptions::default()
            },
            model,
            convert_to_llm: Arc::clone(&self.convert_to_llm),
            transform_context: self.transform_context.clone(),
            get_api_key: self.get_api_key.clone(),
            should_stop_after_turn,
            prepare_next_turn,
            get_steering_messages: Some(get_steering_messages),
            get_follow_up_messages: Some(get_follow_up_messages),
            tool_execution: Some(self.tool_execution),
            before_tool_call: self.before_tool_call.clone(),
            after_tool_call: self.after_tool_call.clone(),
        }
    }

    /// The event sink a run reports through: every event reduces the
    /// agent's state and is dispatched to the awaited listeners, upstream's
    /// `processEvents` closure.
    fn event_sink(self: &Arc<Self>) -> AgentEventSink {
        let inner = Arc::clone(self);
        Arc::new(move |event| {
            let inner = Arc::clone(&inner);
            Box::pin(async move { inner.process_event(event).await })
        })
    }

    /// The steering hook a run polls, upstream's `getSteeringMessages`
    /// closure: the first poll returns nothing when the run started from
    /// drained steering messages, upstream's `skipInitialSteeringPoll`
    /// mutable restated as a shared flag consumed once.
    fn steering_hook(self: &Arc<Self>, skip_initial_steering_poll: bool) -> GetSteeringMessages {
        let inner = Arc::clone(self);
        let skip = Arc::new(AtomicBool::new(skip_initial_steering_poll));
        Arc::new(move || {
            let inner = Arc::clone(&inner);
            let skip = Arc::clone(&skip);
            Box::pin(async move {
                if skip.swap(false, Ordering::Relaxed) {
                    return Vec::new();
                }
                inner
                    .steering_queue
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .drain()
            })
        })
    }

    /// The follow-up hook a run polls, upstream's `getFollowUpMessages`.
    fn follow_up_hook(self: &Arc<Self>) -> GetFollowUpMessages {
        let inner = Arc::clone(self);
        Arc::new(move || {
            let inner = Arc::clone(&inner);
            Box::pin(async move {
                inner
                    .follow_up_queue
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .drain()
            })
        })
    }

    /// The loop-shaped stop hook with the run's token injected at call time,
    /// wrapping the option hook, upstream's `async (context) => await
    /// shouldStopAfterTurn(context, this.signal)`.
    fn should_stop_hook(self: &Arc<Self>) -> Option<ShouldStopAfterTurn> {
        self.should_stop_after_turn
            .clone()
            .map(|hook| -> ShouldStopAfterTurn {
                let inner = Arc::clone(self);
                Arc::new(move |context: ShouldStopAfterTurnContext| {
                    let inner = Arc::clone(&inner);
                    let hook = Arc::clone(&hook);
                    Box::pin(async move { hook(context, inner.current_token()).await })
                })
            })
    }

    /// The loop-shaped turn-preparation hook: the context-aware option hook
    /// wins when both are set, upstream-verbatim; the signal-only shape gets
    /// the run's token in place of the context.
    fn prepare_next_turn_hook(self: &Arc<Self>) -> Option<PrepareNextTurn> {
        self.prepare_next_turn_with_context.clone().map_or_else(
            || {
                self.prepare_next_turn
                    .clone()
                    .map(|hook| -> PrepareNextTurn {
                        let inner = Arc::clone(self);
                        Arc::new(move |_context: PrepareNextTurnContext| {
                            let inner = Arc::clone(&inner);
                            let hook = Arc::clone(&hook);
                            Box::pin(async move { hook(inner.current_token()).await })
                        })
                    })
            },
            |with_context| {
                let inner = Arc::clone(self);
                let hook: PrepareNextTurn = Arc::new(move |context: PrepareNextTurnContext| {
                    let inner = Arc::clone(&inner);
                    let hook = Arc::clone(&with_context);
                    Box::pin(async move { hook(context, inner.current_token()).await })
                });
                Some(hook)
            },
        )
    }

    /// The emitted failure turn, upstream's `handleRunFailure`: a synthetic
    /// assistant message carrying the run's error, run through the same
    /// lifecycle events a streamed turn would produce. `aborted` picks the
    /// stop reason.
    async fn handle_run_failure(&self, message: String, aborted: bool) {
        let (api, provider, model) = {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            (
                state.model.api.clone(),
                state.model.provider.clone(),
                state.model.id.clone(),
            )
        };
        let failure = AssistantMessage {
            content: vec![AssistantBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            })],
            api,
            provider,
            model,
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: if aborted {
                StopReason::Aborted
            } else {
                StopReason::Error
            },
            deferred: None,
            error_message: Some(message),
            raw_stop_reason: None,
            end_turn: None,
            timestamp: now_ms(),
        };
        let failure_message = AgentMessage::Standard(Message::Assistant(failure.clone()));
        self.process_event(AgentEvent::MessageStart {
            message: failure_message.clone(),
        })
        .await;
        self.process_event(AgentEvent::MessageEnd {
            message: failure_message.clone(),
        })
        .await;
        self.process_event(AgentEvent::TurnEnd {
            message: failure_message.clone(),
            tool_results: Vec::new(),
        })
        .await;
        self.process_event(AgentEvent::AgentEnd {
            messages: vec![failure_message],
        })
        .await;
    }
}

/// Normalize the prompt input, upstream's `normalizePromptInput`: text
/// becomes one user message with a text block followed by any images;
/// array inputs pass through.
fn normalize_prompt_input(input: PromptInput) -> Vec<AgentMessage> {
    match input {
        PromptInput::Many(messages) => messages,
        PromptInput::One(message) => vec![message],
        PromptInput::Text { text, images } => {
            let mut content = vec![UserBlock::Text(TextContent {
                text,
                text_signature: None,
            })];
            if !images.is_empty() {
                content.extend(images.into_iter().map(UserBlock::Image));
            }
            vec![AgentMessage::Standard(Message::User(UserMessage {
                content: UserContent::Blocks(content),
                timestamp: now_ms(),
            }))]
        }
    }
}

/// The run's lifecycle owner: execute, route failures through the failure
/// lifecycle, then clear the runtime state — so an un-awaited `prompt`
/// future still settles its run, upstream's `finally` ordering. Returns the
/// panic payload when the failure path itself panics (a listener throwing
/// mid-failure), upstream's rejection escaping `handleRunFailure`.
async fn run_lifecycle(
    inner: Arc<AgentInner>,
    run: RunKind,
    token: CancellationToken,
) -> Result<(), String> {
    let outcome = tokio::spawn(run_executor(Arc::clone(&inner), run, token.clone())).await;
    let failure = match outcome {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(error.to_string()),
        Err(join_error) if join_error.is_panic() => Some(panic_message(join_error)),
        Err(_) => Some("Agent run task was cancelled".to_owned()),
    };
    let Some(message) = failure else {
        inner.finish_run();
        return Ok(());
    };
    // The failure handler runs on its own task so a panicking listener
    // cannot skip the settlement below; `finish_run` runs before the panic
    // is surfaced, upstream's `finally` after a throwing `handleRunFailure`.
    let escaped = match tokio::spawn({
        let inner = Arc::clone(&inner);
        async move {
            inner
                .handle_run_failure(message, token.is_cancelled())
                .await;
        }
    })
    .await
    {
        Ok(()) => None,
        Err(join_error) if join_error.is_panic() => Some(panic_message(join_error)),
        Err(_) => Some("Agent run task was cancelled".to_owned()),
    };
    inner.finish_run();
    escaped.map_or(Ok(()), Err)
}

/// The executor a run drives: the loop entry with the context snapshot, the
/// per-run loop config, the event sink, and the run's token, upstream's
/// `runWithLifecycle` executor closure.
async fn run_executor(
    inner: Arc<AgentInner>,
    run: RunKind,
    token: CancellationToken,
) -> Result<(), AgentLoopError> {
    let skip_initial_steering_poll = matches!(run, RunKind::Prompt(_, true));
    let context = inner.create_context_snapshot();
    let config = inner.create_loop_config(skip_initial_steering_poll);
    let sink = inner.event_sink();
    let stream_fn = inner.stream_function.clone();
    match run {
        RunKind::Prompt(messages, _) => {
            run_agent_loop(messages, context, config, sink, Some(token), stream_fn)
                .await
                .map(|_| ())
        }
        RunKind::Continue => run_agent_loop_continue(context, config, sink, Some(token), stream_fn)
            .await
            .map(|_| ()),
    }
}

/// Map the lifecycle task's join result onto the entry point's return: the
/// task handles every failure itself, so only a panic escaping the failure
/// path rejects, upstream's rejection escaping `handleRunFailure`.
async fn finish_join(handle: JoinHandle<Result<(), String>>) -> Result<(), AgentError> {
    match handle.await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(message)) => Err(AgentError::RunFailed(message)),
        Err(join_error) if join_error.is_panic() => {
            Err(AgentError::RunFailed(panic_message(join_error)))
        }
        Err(_) => Err(AgentError::RunFailed(
            "Agent run task was cancelled".to_owned(),
        )),
    }
}

/// The panic payload's message, upstream's `String(error)` for the values a
/// panicking closure carries; non-string payloads report as `panicked`.
fn panic_message(join_error: JoinError) -> String {
    let payload = join_error.into_panic();
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        return (*text).to_owned();
    }
    if let Some(text) = payload.downcast_ref::<String>() {
        return text.clone();
    }
    "panicked".to_owned()
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field(
                "state",
                &self.0.state.lock().unwrap_or_else(PoisonError::into_inner),
            )
            .finish_non_exhaustive()
    }
}

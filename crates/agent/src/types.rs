//! The core type surface of the agent runtime, upstream's `src/types.ts`.
//!
//! [`StreamFn`] is the transport abstraction the whole crate is built on: the
//! agent loop and the `Agent` class consume any implementation that can turn
//! a model, a context, and simple stream options into an
//! `AssistantMessageEventStream` (`pi-ai`'s `Models.streamSimple` satisfies
//! the shape). [`AgentMessage`] extends `pi-ai`'s message model with the
//! app-defined custom messages, [`AgentEvent`] is the event vocabulary every
//! consumer observes, and [`AgentTool`] is the tool contract the loop
//! executes.

use std::collections::BTreeSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, Context, Message, Model, SimpleStreamOptions, Tool,
    ToolCall, ToolResultMessage, Usage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use tokio_util::sync::CancellationToken;

/// The boxed future the crate's async contracts return, pi-ai's
/// [`BoxedFuture`](pi_ai::types::BoxedFuture) re-used so one lifetime
/// convention covers both crates.
pub type BoxedFuture<'a, T> = pi_ai::types::BoxedFuture<'a, T>;

/// Stream function used by the agent loop, upstream's `StreamFn`.
///
/// `Models.streamSimple` satisfies this shape. Contract, upstream-verbatim:
/// - must not throw or return a rejected promise for request/model/runtime
///   failures;
/// - must return an `AssistantMessageEventStream`;
/// - failures must be encoded in the returned stream via protocol events and
///   a final `AssistantMessage` with stop reason `error` or `aborted` and an
///   error message.
///
/// The boxed-fn shape follows pi-ai's `StreamFunction` precedent: the
/// sync-return lazy-stream statement of upstream's
/// `AssistantMessageEventStream | Promise<AssistantMessageEventStream>`. It
/// is an [`Arc`] rather than a `Box` because the same function value is
/// handed to every loop run — upstream hands one function value around; the
/// `Arc` is that statement — and the process-global default
/// ([`crate::stream_fn`]) must be able to hand out copies.
pub type StreamFn = Arc<
    dyn Fn(&Model, &Context, Option<&SimpleStreamOptions>) -> AssistantMessageEventStream
        + Send
        + Sync,
>;

/// Configuration for how tool calls from a single assistant message are
/// executed, upstream's `ToolExecutionMode`.
///
/// - `Sequential`: each tool call is prepared, executed, and finalized before
///   the next one starts.
/// - `Parallel`: tool calls are prepared sequentially, then allowed tools
///   execute concurrently. `tool_execution_end` is emitted in tool completion
///   order after each tool is finalized, while tool-result message artifacts
///   are emitted later in assistant source order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionMode {
    /// Execute tool calls one by one.
    Sequential,
    /// Prepare sequentially, execute concurrently; the loop default.
    #[default]
    Parallel,
}

/// Controls how many queued user messages are injected when the agent loop
/// reaches a queue drain point, upstream's `QueueMode`.
///
/// - `All`: drain and inject every queued message at that point.
/// - `OneAtATime`: drain and inject only the oldest queued message, leaving
///   the rest queued for later drain points.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueueMode {
    /// Drain and inject every queued message at each drain point.
    #[default]
    All,
    /// Drain and inject only the oldest queued message at each drain point.
    OneAtATime,
}

/// Thinking/reasoning level for models that support it, upstream's
/// `ThinkingLevel`.
///
/// `Xhigh` and `Max` are only supported by selected model families; use
/// model thinking-level metadata from pi-ai to detect support for a concrete
/// model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingLevel {
    /// No thinking.
    #[default]
    Off,
    /// Minimal thinking budget.
    Minimal,
    /// Low thinking budget.
    Low,
    /// Medium thinking budget.
    Medium,
    /// High thinking budget.
    High,
    /// Extra-high budget, selected model families only.
    #[serde(rename = "xhigh")]
    Xhigh,
    /// Maximum budget, selected model families only.
    Max,
}

/// A single tool-call content block emitted by an assistant message,
/// upstream's `AgentToolCall` — the `Extract` of the `toolCall` block.
pub type AgentToolCall = ToolCall;

/// The content a tool returns: text and image blocks. Upstream types this as
/// the `(TextContent | ImageContent)[]` element union with no name; pi-ai's
/// tool-result block enum is that union.
pub type AgentToolContent = pi_ai::types::ToolResultBlock;

/// Result returned from `before_tool_call`, upstream's `BeforeToolCallResult`.
///
/// A `block` call prevents the tool from executing; the loop emits an error
/// tool result instead, with `reason` as its text (a default blocked message
/// when `reason` is omitted). The `terminate` hint participates in the
/// batch early-termination rule only when every finalized tool result in the
/// batch sets it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BeforeToolCallResult {
    /// Prevent the tool from executing; the loop emits an error tool result.
    /// Omitted: the call is not blocked.
    pub block: Option<bool>,
    /// The text shown in the error result of a blocked call. Omitted: the
    /// loop's default blocked message.
    pub reason: Option<String>,
    /// Hint that the agent should stop after the current tool batch when this
    /// call is blocked. Omitted: no early-termination hint.
    pub terminate: Option<bool>,
}

/// Partial override returned from `after_tool_call`, upstream's
/// `AfterToolCallResult`.
///
/// Merge semantics are field-by-field, upstream-verbatim: an `Some` field
/// replaces the executed tool result's value in full (there is no deep merge
/// for content, details, or usage); `None` keeps the original executed
/// values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AfterToolCallResult {
    /// Replaces the tool result content array in full.
    pub content: Option<Vec<AgentToolContent>>,
    /// Replaces the tool result details value in full.
    pub details: Option<serde_json::Value>,
    /// Replaces the tool result error flag.
    pub is_error: Option<bool>,
    /// Usage from the final tool execution itself, if available. Not used for
    /// main LLM context accounting.
    pub usage: Option<Usage>,
    /// Hint that the agent should stop after the current tool batch. Omitted:
    /// no early-termination hint.
    pub terminate: Option<bool>,
}

/// Context passed to `before_tool_call`, upstream's `BeforeToolCallContext`.
/// No `PartialEq`: the snapshot carries execute closures.
#[derive(Clone, Debug)]
pub struct BeforeToolCallContext {
    /// The assistant message that requested the tool call.
    pub assistant_message: AssistantMessage,
    /// The raw tool-call block from `assistant_message.content`.
    pub tool_call: AgentToolCall,
    /// Validated tool arguments for the target tool schema.
    pub args: serde_json::Value,
    /// Current agent context at the time the tool call is prepared.
    pub context: AgentContext,
}

/// Context passed to `after_tool_call`, upstream's `AfterToolCallContext`.
/// No `PartialEq`: the snapshot carries execute closures.
#[derive(Clone, Debug)]
pub struct AfterToolCallContext {
    /// The assistant message that requested the tool call.
    pub assistant_message: AssistantMessage,
    /// The raw tool-call block from `assistant_message.content`.
    pub tool_call: AgentToolCall,
    /// Validated tool arguments for the target tool schema.
    pub args: serde_json::Value,
    /// The executed tool result before any `after_tool_call` overrides are
    /// applied.
    pub result: AgentToolResult,
    /// Whether the executed tool result is currently treated as an error.
    pub is_error: bool,
    /// Current agent context at the time the tool call is finalized.
    pub context: AgentContext,
}

/// Context passed to `should_stop_after_turn`, upstream's
/// `ShouldStopAfterTurnContext`. No `PartialEq`: the snapshot carries execute
/// closures.
#[derive(Clone, Debug)]
pub struct ShouldStopAfterTurnContext {
    /// The assistant message that completed the turn.
    pub message: AssistantMessage,
    /// Tool result messages passed to the preceding `turn_end` event.
    pub tool_results: Vec<ToolResultMessage>,
    /// Current agent context after the turn's assistant message and tool
    /// results have been appended.
    pub context: AgentContext,
    /// Messages that this loop invocation will return if it exits at this
    /// point. Prompt runs include the initial prompt messages; continuation
    /// runs do not include pre-existing context messages.
    pub new_messages: Vec<AgentMessage>,
}

/// Replacement runtime state used by the agent loop before starting another
/// provider request, upstream's `AgentLoopTurnUpdate`.
#[derive(Clone, Debug, Default)]
pub struct AgentLoopTurnUpdate {
    /// Context for the next provider request. `None`: keep the current one.
    pub context: Option<AgentContext>,
    /// Model for the next provider request. `None`: keep the current one.
    pub model: Option<Model>,
    /// Thinking level for the next provider request. `None`: keep the
    /// current one.
    pub thinking_level: Option<ThinkingLevel>,
}

/// Context passed to `prepare_next_turn`, upstream's
/// `PrepareNextTurnContext` — declared as an empty extension of
/// `ShouldStopAfterTurnContext` upstream; the alias restates the identity.
pub type PrepareNextTurnContext = ShouldStopAfterTurnContext;

/// Converts agent messages to LLM-compatible messages before each LLM call,
/// upstream's `AgentLoopConfig.convertToLlm`.
///
/// Each [`AgentMessage`] must be converted to a user, assistant, or
/// tool-result message the LLM understands; messages that cannot be
/// converted (UI-only notifications, status messages) are filtered out.
///
/// Contract: must not throw or reject — return a safe fallback instead.
/// Throwing interrupts the low-level agent loop without producing a normal
/// event sequence.
pub type ConvertToLlm =
    Arc<dyn Fn(Vec<AgentMessage>) -> BoxedFuture<'static, Vec<Message>> + Send + Sync>;

/// Optional transform applied to the context before `convert_to_llm`,
/// upstream's `AgentLoopConfig.transformContext`.
///
/// For operations that work at the [`AgentMessage`] level: context-window
/// management (pruning old messages), injecting context from external
/// sources. Contract: must not throw or reject — return the original
/// messages or another safe fallback instead.
pub type TransformContext = Arc<
    dyn Fn(Vec<AgentMessage>, Option<CancellationToken>) -> BoxedFuture<'static, Vec<AgentMessage>>
        + Send
        + Sync,
>;

/// Resolves an API key dynamically for each LLM call, upstream's
/// `AgentLoopConfig.getApiKey`.
///
/// Useful for short-lived OAuth tokens (e.g. GitHub Copilot) that may expire
/// during long-running tool-execution phases. Contract: must not throw or
/// reject; return `None` when no key is available.
pub type GetApiKey = Arc<dyn Fn(&str) -> BoxedFuture<'static, Option<String>> + Send + Sync>;

/// Called after each turn fully completes and `turn_end` has been emitted,
/// upstream's `AgentLoopConfig.shouldStopAfterTurn`.
///
/// `true` makes the loop emit `agent_end` and exit before polling steering
/// or follow-up queues, without starting another LLM call; the current
/// assistant response and any tool executions finish normally. The hook sees
/// the completed-turn context and runs before `prepare_next_turn`. Use it to
/// request a graceful stop after the current turn, e.g. before context gets
/// too full. Contract: must not throw or reject — throwing interrupts the
/// low-level agent loop without producing a normal event sequence.
pub type ShouldStopAfterTurn =
    Arc<dyn Fn(ShouldStopAfterTurnContext) -> BoxedFuture<'static, bool> + Send + Sync>;

/// Called after `turn_end` when the loop will continue, immediately before
/// the next turn starts, upstream's `AgentLoopConfig.prepareNextTurn`.
///
/// Return replacement context/model/thinking state to affect that turn;
/// `None` keeps the current context and configuration.
pub type PrepareNextTurn = Arc<
    dyn Fn(PrepareNextTurnContext) -> BoxedFuture<'static, Option<AgentLoopTurnUpdate>>
        + Send
        + Sync,
>;

/// Returns steering messages to inject into the conversation mid-run,
/// upstream's `AgentLoopConfig.getSteeringMessages`.
///
/// Called after the current assistant turn finishes executing its tool
/// calls, unless `should_stop_after_turn` exits first. Returned messages are
/// added to the context before the next LLM call; tool calls from the
/// current assistant message are not skipped. Use this for steering the
/// agent while it works. Contract: must not throw or reject — return an
/// empty vector when no steering messages are available.
pub type GetSteeringMessages =
    Arc<dyn Fn() -> BoxedFuture<'static, Vec<AgentMessage>> + Send + Sync>;

/// Returns follow-up messages to process after the agent would otherwise
/// stop, upstream's `AgentLoopConfig.getFollowUpMessages`.
///
/// Called when the agent has no more tool calls and no steering messages.
/// Returned messages are added to the context and the agent continues with
/// another turn. Use this for follow-up messages that should wait until the
/// agent finishes. Contract: must not throw or reject — return an empty
/// vector when no follow-up messages are available.
pub type GetFollowUpMessages =
    Arc<dyn Fn() -> BoxedFuture<'static, Vec<AgentMessage>> + Send + Sync>;

/// Called before a tool is executed, after arguments have been validated,
/// upstream's `AgentLoopConfig.beforeToolCall`.
///
/// Return `Some` with `block: true` to prevent execution; the loop emits an
/// error tool result instead. A blocked result can also set `terminate: true`
/// to participate in the batch early-termination rule. The hook receives the
/// agent abort token and is responsible for honoring it.
pub type BeforeToolCall = Arc<
    dyn Fn(
            BeforeToolCallContext,
            Option<CancellationToken>,
        ) -> BoxedFuture<'static, Option<BeforeToolCallResult>>
        + Send
        + Sync,
>;

/// Called after a tool finishes executing, before `tool_execution_end` and
/// tool-result message events are emitted, upstream's
/// `AgentLoopConfig.afterToolCall`.
///
/// Return an [`AfterToolCallResult`] to override parts of the executed tool
/// result; `None` keeps the executed result as-is. Any omitted fields keep
/// their original values — no deep merge is performed. The hook receives the
/// agent abort token and is responsible for honoring it.
pub type AfterToolCall = Arc<
    dyn Fn(
            AfterToolCallContext,
            Option<CancellationToken>,
        ) -> BoxedFuture<'static, Option<AfterToolCallResult>>
        + Send
        + Sync,
>;

/// The low-level agent loop's configuration, upstream's `AgentLoopConfig`.
///
/// Upstream `extends SimpleStreamOptions`; the port carries those fields in
/// [`stream_options`](AgentLoopConfig::stream_options) and forwards the
/// struct as a whole to the [`StreamFn`] — the config is the simple-request
/// options of every provider call it drives.
#[derive(Clone)]
pub struct AgentLoopConfig {
    /// The simple-request options each provider call this config drives
    /// receives.
    pub stream_options: SimpleStreamOptions,
    /// The model every provider request uses until a turn update replaces it.
    pub model: Model,
    /// The agent-message to LLM-message conversion the loop runs before each
    /// provider call. See [`ConvertToLlm`] for the contract.
    pub convert_to_llm: ConvertToLlm,
    /// Optional transform applied to the context before `convert_to_llm`.
    pub transform_context: Option<TransformContext>,
    /// Dynamic API-key resolution for each LLM call.
    pub get_api_key: Option<GetApiKey>,
    /// Graceful stop after the current turn.
    pub should_stop_after_turn: Option<ShouldStopAfterTurn>,
    /// Replacement context/model/thinking state for the next turn.
    pub prepare_next_turn: Option<PrepareNextTurn>,
    /// Steering messages to inject into the conversation mid-run.
    pub get_steering_messages: Option<GetSteeringMessages>,
    /// Follow-up messages to process after the agent would otherwise stop.
    pub get_follow_up_messages: Option<GetFollowUpMessages>,
    /// Tool execution mode. Omitted: the loop's default, `Parallel`.
    pub tool_execution: Option<ToolExecutionMode>,
    /// Pre-execution gate for each tool call, after argument validation.
    pub before_tool_call: Option<BeforeToolCall>,
    /// Post-execution override pass for each tool result.
    pub after_tool_call: Option<AfterToolCall>,
}

impl std::fmt::Debug for AgentLoopConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Hook closures do not debug; the config's data fields do. Pinning the
        // non-exhaustive shape here keeps the derive off a struct that cannot
        // have one.
        f.debug_struct("AgentLoopConfig")
            .field("stream_options", &self.stream_options)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

/// A custom app message, upstream's `CustomAgentMessages[keyof
/// CustomAgentMessages]`.
///
/// Upstream apps add message types via declaration merging; Rust has no
/// equivalent, so the extension surface is this owned-JSON shape. The wire
/// contract it preserves: a custom message is a JSON object whose `role` is
/// a non-standard discriminator, whose `timestamp` is milliseconds since the
/// Unix epoch, and whose remaining fields are arbitrary. Every custom
/// message type the coding agent declares at the pin
/// (`bashExecution`, `custom`, `branchSummary`, `compactionSummary`) carries
/// both, and the session layer persists messages as raw JSON lines, so
/// `role`/`timestamp` are first-class fields here and every other field
/// rides [`data`](CustomAgentMessage::data) verbatim — a serialized custom
/// message round-trips field-for-field through the JSONL layer.
///
/// Apps that want typed access write conversion helpers over `data` (the
/// coding-agent port carries its message structs in its own crate); the
/// `convert_to_llm` hook is where custom messages are converted or filtered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomAgentMessage {
    /// The non-standard role discriminator, e.g. `bashExecution` or
    /// `custom`. A role equal to one of the standard message roles would
    /// make the object indistinguishable from an LLM message and is a
    /// construction bug.
    pub role: String,
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
    /// Every other field of the message object, verbatim.
    pub data: serde_json::Map<String, serde_json::Value>,
}

impl CustomAgentMessage {
    /// Read one of the message's non-`role`/`timestamp` fields.
    #[must_use]
    pub fn field(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key)
    }
}

impl Serialize for CustomAgentMessage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap as _;
        let mut map = serializer.serialize_map(Some(self.data.len() + 2))?;
        map.serialize_entry("role", &self.role)?;
        map.serialize_entry("timestamp", &self.timestamp)?;
        for (key, value) in &self.data {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for CustomAgentMessage {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut object = serde_json::Map::<String, serde_json::Value>::deserialize(deserializer)?;
        let role = object
            .remove("role")
            .ok_or_else(|| serde::de::Error::missing_field("role"))?;
        let role = String::deserialize(role).map_err(|_| {
            serde::de::Error::custom("a custom agent message's role must be a string")
        })?;
        let timestamp = object
            .remove("timestamp")
            .ok_or_else(|| serde::de::Error::missing_field("timestamp"))?;
        let timestamp = i64::deserialize(timestamp).map_err(|_| {
            serde::de::Error::custom("a custom agent message's timestamp must be a number")
        })?;
        Ok(Self {
            role,
            timestamp,
            data: object,
        })
    }
}

/// One message of the agent's transcript, upstream's `AgentMessage` — the
/// LLM message union extended with the app-defined custom messages.
///
/// Standard roles deserialize by their wire `role` discriminators
/// (pi-ai's `Message`); any object with a non-standard `role` and a
/// `timestamp` is a [`AgentMessage::Custom`] variant. See [`CustomAgentMessage`] for the
/// extension design.
#[expect(
    clippy::large_enum_variant,
    reason = "the message model mirrors upstream: an assistant message carries the full usage accounting beside one-line custom messages"
)]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum AgentMessage {
    /// A user, assistant, or tool-result LLM message.
    Standard(Message),
    /// An app-defined custom message.
    Custom(CustomAgentMessage),
}

/// Public agent state, upstream's `AgentState`.
///
/// Upstream's `tools` and `messages` are accessor properties so
/// implementations can copy assigned arrays before storing them; the port
/// states that as plain fields plus setter methods on the owner (upstream's
/// `Agent`), which clone before storing. `is_streaming` remains
/// true until awaited `agent_end` listeners settle. No `PartialEq`: the
/// tools carry execute closures.
#[derive(Clone, Debug)]
pub struct AgentState {
    /// System prompt sent with each model request.
    pub system_prompt: String,
    /// Active model used for future turns.
    pub model: Model,
    /// Requested reasoning level for future turns.
    pub thinking_level: ThinkingLevel,
    /// Available tools. Assigning a new value copies the top-level array.
    pub tools: Vec<AgentTool>,
    /// Conversation transcript. Assigning a new value copies the top-level
    /// array.
    pub messages: Vec<AgentMessage>,
    /// True while the agent is processing a prompt or continuation; remains
    /// true until awaited `agent_end` listeners settle.
    pub is_streaming: bool,
    /// Partial assistant message for the current streamed response, if any.
    pub streaming_message: Option<AgentMessage>,
    /// Tool call ids currently executing.
    pub pending_tool_calls: BTreeSet<String>,
    /// Error message from the most recent failed or aborted assistant turn,
    /// if any.
    pub error_message: Option<String>,
}

/// Final or partial result produced by a tool, upstream's
/// `AgentToolResult<T>`.
///
/// The erased runtime path is `T = JsonValue` (upstream's `any`): the loop
/// serializes tool details into the tool-result message's JSON details.
/// Authoring-side tools with typed details restate over this struct with
/// their own `T` (the tools child carries that surface).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolResult<TDetails = serde_json::Value> {
    /// Text or image content returned to the model.
    pub content: Vec<AgentToolContent>,
    /// Arbitrary structured details for logs or UI rendering.
    pub details: TDetails,
    /// Usage from the final tool execution itself, if available. Not used
    /// for main LLM context accounting.
    pub usage: Option<Usage>,
    /// Names of tools introduced by this result and available from this
    /// transcript point onward.
    pub added_tool_names: Option<Vec<String>>,
    /// Hint that the agent should stop after the current tool batch. Early
    /// termination only happens when every finalized tool result in the
    /// batch sets this to true.
    pub terminate: Option<bool>,
}

/// Callback used by tools to stream partial execution updates, upstream's
/// `AgentToolUpdateCallback<T>`.
///
/// The callback is scoped to the current `execute` invocation; calls made
/// after the tool settles are ignored by the loop.
pub type AgentToolUpdateCallback<'a, TDetails = serde_json::Value> =
    &'a (dyn Fn(&AgentToolResult<TDetails>) + Send + Sync);

/// The error a tool's execution surfaces, upstream's `execute` throw.
///
/// Upstream tools throw on failure instead of encoding errors in `content`;
/// the loop catches and converts the throw into an error tool result.
pub type AgentToolError = Box<dyn std::error::Error + Send + Sync>;

/// The erased tool contract the runtime stores, upstream's
/// `AgentTool<TSchema, any>`: validated arguments in, result (or partial
/// updates) out, abort honored by the tool.
///
/// Upstream throws on failure instead of encoding errors in `content`; the
/// port states that as the error half of the future.
pub type AgentToolExecuteFn = dyn for<'a> Fn(
        &'a str,
        &'a serde_json::Value,
        Option<&'a CancellationToken>,
        Option<AgentToolUpdateCallback<'a>>,
    ) -> BoxedFuture<'a, Result<AgentToolResult, AgentToolError>>
    + Send
    + Sync;

/// Recovery policy for an effect whose durable intent exists but whose
/// outcome is unknown, upstream's `replay?: "never" | "safe"`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolReplay {
    /// The effect must not re-run; recovery reports it as unknown.
    Never,
    /// The effect is safe to re-run during recovery.
    #[default]
    Safe,
}

/// Tool definition used by the agent runtime, upstream's
/// `AgentTool<TParameters extends TSchema, TDetails>`.
///
/// pi-ai's `Tool` extended with the label, the pre-validation argument shim,
/// the executor, the replay policy, and the per-tool execution-mode
/// override. The typebox `Static<TParameters>` parameterization restates
/// over plain JSON Schemas (pi-ai's tool-args substrate validates against
/// the schema document [`tool`](AgentTool::tool) carries); the erased
/// runtime type carries `JsonValue` arguments and details.
#[derive(Clone)]
pub struct AgentTool {
    /// The pi-ai tool surface: name, description, parameter JSON Schema
    /// document, constrained-sampling setting.
    pub tool: Tool,
    /// Human-readable label for UI display.
    pub label: String,
    /// Optional compatibility shim for raw tool-call arguments before schema
    /// validation; must return an object that matches the tool's schema.
    pub prepare_arguments: Option<AgentToolPrepareArguments>,
    /// Execute the tool call. Throw on failure instead of encoding errors in
    /// `content`.
    pub execute: Arc<AgentToolExecuteFn>,
    /// Recovery policy for an effect whose outcome is unknown. Omitted: the
    /// loop's default, `Safe`.
    pub replay: Option<ToolReplay>,
    /// Per-tool execution-mode override. `Sequential`: this tool must
    /// execute one at a time with other tool calls; `Parallel`: this tool
    /// can execute concurrently. Omitted: the config's execution mode
    /// applies.
    pub execution_mode: Option<ToolExecutionMode>,
}

impl std::fmt::Debug for AgentTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The closures do not debug; the declarative surface does, and the
        // non-exhaustive finish names the executor's presence.
        f.debug_struct("AgentTool")
            .field("tool", &self.tool)
            .field("label", &self.label)
            .field("prepare_arguments", &self.prepare_arguments.is_some())
            .field("execute", &())
            .field("replay", &self.replay)
            .field("execution_mode", &self.execution_mode)
            .finish_non_exhaustive()
    }
}

/// The pre-validation argument shim, upstream's
/// `AgentTool.prepareArguments`.
pub type AgentToolPrepareArguments =
    Arc<dyn Fn(&serde_json::Value) -> Result<serde_json::Value, AgentToolError> + Send + Sync>;

impl AgentTool {
    /// The tool name, upstream's `AgentTool.name` inherited from `Tool`.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.tool.name
    }
}

/// Context snapshot passed into the low-level agent loop, upstream's
/// `AgentContext`. No `PartialEq`: the tools carry execute closures.
#[derive(Clone, Debug, Default)]
pub struct AgentContext {
    /// System prompt included with the request.
    pub system_prompt: String,
    /// Transcript visible to the model.
    pub messages: Vec<AgentMessage>,
    /// Tools available for this run. Omitted: no tools this run.
    pub tools: Option<Vec<AgentTool>>,
}

/// Events emitted by the agent runtime for UI updates, upstream's
/// `AgentEvent`.
///
/// `agent_end` is the last event emitted for a run, but awaited
/// `Agent.subscribe()` listeners for that event are still part of run
/// settlement — the agent becomes idle only after those listeners finish
/// (the Agent-class child carries that settlement).
#[expect(
    clippy::large_enum_variant,
    reason = "the event payloads mirror upstream: message_update carries the shared live partial plus the protocol event that produced it"
)]
#[derive(Clone, Debug, PartialEq)]
pub enum AgentEvent {
    /// The agent began processing, upstream `agent_start`.
    AgentStart,
    /// The agent finished; carries the messages the run produced, upstream
    /// `agent_end`.
    AgentEnd {
        /// The messages the completed run produced.
        messages: Vec<AgentMessage>,
    },
    /// A turn began — one assistant response plus any tool calls/results,
    /// upstream `turn_start`.
    TurnStart,
    /// The turn ended, upstream `turn_end`.
    TurnEnd {
        /// The assistant message that completed the turn.
        message: AgentMessage,
        /// The tool result messages of the turn.
        tool_results: Vec<ToolResultMessage>,
    },
    /// A message entered the transcript, upstream `message_start`; emitted
    /// for user, assistant, and tool-result messages.
    MessageStart {
        /// The message that started.
        message: AgentMessage,
    },
    /// An assistant message streamed an update, upstream `message_update`.
    /// Only emitted for assistant messages during streaming.
    MessageUpdate {
        /// The live partial message.
        message: AgentMessage,
        /// The assistant-message protocol event that produced the update.
        assistant_message_event: AssistantMessageEvent,
    },
    /// A message reached its final form, upstream `message_end`.
    MessageEnd {
        /// The final message.
        message: AgentMessage,
    },
    /// A tool execution began, upstream `tool_execution_start`.
    ToolExecutionStart {
        /// The executing call's id.
        tool_call_id: String,
        /// The executing tool's name.
        tool_name: String,
        /// The validated call arguments.
        args: serde_json::Value,
    },
    /// A tool streamed a partial result, upstream `tool_execution_update`.
    ToolExecutionUpdate {
        /// The executing call's id.
        tool_call_id: String,
        /// The executing tool's name.
        tool_name: String,
        /// The validated call arguments.
        args: serde_json::Value,
        /// The partial result the tool streamed.
        partial_result: AgentToolResult,
    },
    /// A tool execution finalized, upstream `tool_execution_end`.
    ToolExecutionEnd {
        /// The finalized call's id.
        tool_call_id: String,
        /// The finalized tool's name.
        tool_name: String,
        /// The finalized tool result, after any `after_tool_call` override.
        result: AgentToolResult,
        /// Whether the finalized result is an error.
        is_error: bool,
    },
}

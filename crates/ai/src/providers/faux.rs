//! The faux provider, ported from `packages/ai/src/providers/faux.ts` at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! An in-memory fake provider: a queue of scripted responses — assistant
//! messages or factories receiving the request context — streams through the
//! real event protocol with usage estimation from the serialized context,
//! simulated prompt caching per session, paced token deltas, aborts, and the
//! deferred-response machinery (`stream_deferred` submissions, pending
//! fetches that keep returning the handle, in-band redemption failures).
//! The test substrate the downstream coding-agent port reuses.
//!
//! Porting restatements: `Math.random` becomes a system-random draw per id
//! and per chunk size; the `queueMicrotask` stream bodies run as spawned
//! tokio tasks, and the unpaced chunk pacing is a scheduler yield so pacing
//! sleeps ride the tokio clock and tests can pause it; `AbortSignal` is the
//! crate's `CancellationToken`; chunk cuts land on UTF-8 char boundaries
//! where upstream slices UTF-16 units (the suites' fixtures are ASCII, where
//! the two agree); string lengths in the token estimates are byte lengths.
//! The registry-glue entry lives with the compat surface:
//! [`faux_provider`] builds the standalone `Provider`, [`create_faux_core`]
//! exposes the pieces compat's `register_faux_provider` wraps, and
//! [`FauxProviderRegistration`] is the registration handle that function
//! hands back over the compat api-registry.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::types::{
    Api, AssistantBlock, AssistantMessage, AssistantMessageEvent, CacheRetention, Context,
    DeferredHandle, ImageContent, Message, Modality, Model, ModelCost, ProviderId,
    SimpleStreamOptions, StopReason, StreamOptions, TextContent, ThinkingContent, ToolCall,
    ToolResultMessage, UserBlock, UserContent,
};
use crate::utils::event_stream::{
    AssistantMessageEventStream, create_assistant_message_event_stream,
};
use crate::utils::provider_retry::ProviderRequestError;
use crate::{models, types};

/// The default wire API of a faux response, upstream's `DEFAULT_API`.
pub const DEFAULT_API: &str = "faux";
/// The default provider id, upstream's `DEFAULT_PROVIDER`.
pub const DEFAULT_PROVIDER: &str = "faux";
/// The default model id, upstream's `DEFAULT_MODEL_ID`.
pub const DEFAULT_MODEL_ID: &str = "faux-1";
/// The default model name, upstream's `DEFAULT_MODEL_NAME`.
pub const DEFAULT_MODEL_NAME: &str = "Faux Model";
/// The default base URL, upstream's `DEFAULT_BASE_URL` — a port that never
/// answers, since the faux provider performs no network I/O.
pub const DEFAULT_BASE_URL: &str = "http://localhost:0";
/// The default minimum chunk size, upstream's `DEFAULT_MIN_TOKEN_SIZE`.
pub const DEFAULT_MIN_TOKEN_SIZE: usize = 3;
/// The default maximum chunk size, upstream's `DEFAULT_MAX_TOKEN_SIZE`.
pub const DEFAULT_MAX_TOKEN_SIZE: usize = 5;

/// The default usage every faux-built message carries before its estimate
/// overwrites it, upstream's `DEFAULT_USAGE`.
fn default_usage() -> types::Usage {
    types::Usage::default()
}

/// One faux model definition, upstream's `FauxModelDefinition`.
#[derive(Clone, Debug, Default)]
pub struct FauxModelDefinition {
    /// The model id.
    pub id: String,
    /// The display name. Default: the id.
    pub name: Option<String>,
    /// Whether the model supports reasoning. Default: `false`.
    pub reasoning: Option<bool>,
    /// The input modalities. Default: text and image.
    pub input: Option<Vec<Modality>>,
    /// The pricing. Default: all-zero rates.
    pub cost: Option<ModelCost>,
    /// The context window in tokens. Default: 128000.
    pub context_window: Option<u64>,
    /// The maximum output tokens. Default: 16384.
    pub max_tokens: Option<u64>,
}

/// The content one faux response carries: text, thinking, or tool-call
/// blocks, upstream's `FauxContentBlock`.
pub type FauxContentBlock = AssistantBlock;

/// A text block, upstream's `fauxText`.
#[must_use]
pub fn faux_text(text: impl Into<String>) -> FauxContentBlock {
    FauxContentBlock::Text(TextContent {
        text: text.into(),
        text_signature: None,
    })
}

/// A thinking block, upstream's `fauxThinking`.
#[must_use]
pub fn faux_thinking(thinking: impl Into<String>) -> FauxContentBlock {
    FauxContentBlock::Thinking(ThinkingContent {
        thinking: thinking.into(),
        thinking_signature: None,
        redacted: None,
    })
}

/// A tool-call block, upstream's `fauxToolCall`; `id` defaults to a fresh
/// `tool:`-prefixed random id.
#[must_use]
pub fn faux_tool_call(
    name: impl Into<String>,
    arguments: serde_json::Map<String, serde_json::Value>,
    id: Option<String>,
) -> FauxContentBlock {
    FauxContentBlock::ToolCall(ToolCall {
        id: id.unwrap_or_else(|| random_id("tool")),
        name: name.into(),
        arguments,
        thought_signature: None,
        namespace: None,
    })
}

/// The content shapes [`faux_assistant_message`] accepts: a plain string
/// (one text block), one block, or a block array, upstream's
/// `string | FauxContentBlock | FauxContentBlock[]`.
pub trait IntoFauxContent {
    /// Normalize the content to block form, upstream's
    /// `normalizeFauxAssistantContent`.
    fn into_faux_content(self) -> Vec<FauxContentBlock>;
}

impl IntoFauxContent for &str {
    fn into_faux_content(self) -> Vec<FauxContentBlock> {
        vec![faux_text(self)]
    }
}

impl IntoFauxContent for String {
    fn into_faux_content(self) -> Vec<FauxContentBlock> {
        vec![faux_text(self)]
    }
}

impl IntoFauxContent for FauxContentBlock {
    fn into_faux_content(self) -> Vec<FauxContentBlock> {
        vec![self]
    }
}

impl IntoFauxContent for Vec<FauxContentBlock> {
    fn into_faux_content(self) -> Vec<FauxContentBlock> {
        self
    }
}

/// The construction options of [`faux_assistant_message`], upstream's inline
/// `{ stopReason?, deferred?, errorMessage?, responseId?, timestamp? }`.
#[derive(Clone, Debug, Default)]
pub struct FauxAssistantMessageOptions {
    /// The stop reason. Default: `stop`.
    pub stop_reason: Option<StopReason>,
    /// The deferred handle the response carries.
    pub deferred: Option<DeferredHandle>,
    /// The failure description.
    pub error_message: Option<String>,
    /// The provider-specific response id.
    pub response_id: Option<String>,
    /// Unix-millisecond timestamp. Default: now.
    pub timestamp: Option<i64>,
}

/// An assistant message carrying the faux provider's default api, provider,
/// model, and usage, upstream's `fauxAssistantMessage`.
#[must_use]
pub fn faux_assistant_message(
    content: impl IntoFauxContent,
    options: FauxAssistantMessageOptions,
) -> AssistantMessage {
    AssistantMessage {
        content: content.into_faux_content(),
        api: Api::from(DEFAULT_API),
        provider: ProviderId::from(DEFAULT_PROVIDER),
        model: DEFAULT_MODEL_ID.to_owned(),
        response_model: None,
        response_id: options.response_id,
        provider_thinking_level: None,
        diagnostics: None,
        usage: default_usage(),
        stop_reason: options.stop_reason.unwrap_or(StopReason::Stop),
        deferred: options.deferred,
        error_message: options.error_message,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: options
            .timestamp
            .unwrap_or_else(crate::auth::resolve::now_ms),
    }
}

/// The shared observable state of one faux core, upstream's
/// `FauxProviderState`.
#[derive(Debug, Default)]
pub struct FauxProviderState {
    call_count: std::sync::atomic::AtomicU64,
    deferred_fetch_count: std::sync::atomic::AtomicU64,
    cancelled_deferred: Mutex<Vec<DeferredHandle>>,
}

impl FauxProviderState {
    /// How many streams this core dispatched, upstream's `callCount`.
    #[must_use]
    pub fn call_count(&self) -> u64 {
        self.call_count.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// How many deferred fetches this core dispatched, upstream's
    /// `deferredFetchCount`.
    #[must_use]
    pub fn deferred_fetch_count(&self) -> u64 {
        self.deferred_fetch_count
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The handles cancellation recorded, upstream's `cancelledDeferred`.
    #[must_use]
    pub fn cancelled_deferred(&self) -> Vec<DeferredHandle> {
        self.cancelled_deferred
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn record_cancelled(&self, handle: DeferredHandle) {
        self.cancelled_deferred
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(handle);
    }
}

/// A scripted response factory, upstream's `FauxResponseFactory`.
///
/// Receives the request context, the stream options, the shared state, and
/// the request model, and produces the assistant message. Fails with any
/// error; the failure surfaces as the stream's terminal error message.
pub type FauxResponseFactory = Arc<
    dyn for<'a> Fn(
            &'a Context,
            Option<&'a SimpleStreamOptions>,
            &'a FauxProviderState,
            &'a Model,
        ) -> types::BoxedFuture<'a, Result<AssistantMessage, FauxFactoryError>>
        + Send
        + Sync,
>;

/// The failure a response factory fails with, upstream's untyped `throw`.
pub type FauxFactoryError = Box<dyn std::error::Error + Send + Sync>;

/// One queued response step, upstream's
/// `FauxResponseStep = AssistantMessage | FauxResponseFactory`.
#[expect(
    clippy::large_enum_variant,
    reason = "the union mirrors upstream: a scripted message carries the full usage model while a factory holds one closure"
)]
#[derive(Clone)]
pub enum FauxResponseStep {
    /// A complete assistant message.
    Message(AssistantMessage),
    /// A factory evaluated per request.
    Factory(FauxResponseFactory),
}

impl std::fmt::Debug for FauxResponseStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message(message) => f.debug_tuple("Message").field(message).finish(),
            Self::Factory(_) => f.write_str("Factory(..)"),
        }
    }
}

impl From<AssistantMessage> for FauxResponseStep {
    fn from(message: AssistantMessage) -> Self {
        Self::Message(message)
    }
}

/// The deferred-behavior options, upstream's
/// `RegisterFauxProviderOptions.deferred`.
#[derive(Clone, Debug, Default)]
pub struct FauxDeferredOptions {
    /// Number of fetches that return the original handle before the scripted
    /// response becomes ready, upstream's `pendingFetches`.
    pub pending_fetches: Option<u64>,
    /// Milliseconds the returned handle asks callers to wait before the
    /// first poll, upstream's `pollAfterMs`.
    pub poll_after_ms: Option<u64>,
}

/// The chunk-size bounds, upstream's `RegisterFauxProviderOptions.tokenSize`.
#[derive(Clone, Debug, Default)]
pub struct FauxTokenSize {
    /// Minimum chunk size. Default: [`DEFAULT_MIN_TOKEN_SIZE`].
    pub min: Option<usize>,
    /// Maximum chunk size. Default: [`DEFAULT_MAX_TOKEN_SIZE`].
    pub max: Option<usize>,
}

/// The construction options of [`create_faux_core`], upstream's
/// `RegisterFauxProviderOptions`.
#[derive(Clone, Debug, Default)]
pub struct RegisterFauxProviderOptions {
    /// The wire-API id to register under. Default: a fresh random `faux:` id
    /// per core.
    pub api: Option<String>,
    /// The provider id. Default: [`DEFAULT_PROVIDER`].
    pub provider: Option<String>,
    /// The model definitions. Default: one faux-1 model.
    pub models: Vec<FauxModelDefinition>,
    /// The deferred-response behavior.
    pub deferred: Option<FauxDeferredOptions>,
    /// Stream pacing in tokens per second; `None` or non-positive streams
    /// unpaced.
    pub tokens_per_second: Option<f64>,
    /// The chunk-size bounds.
    pub token_size: Option<FauxTokenSize>,
}

/// A deferred-response entry, upstream's `deferredResponses` map value.
struct DeferredEntry {
    handle: DeferredHandle,
    step: FauxResponseStep,
    context: Context,
    options: Option<SimpleStreamOptions>,
    model: Model,
    pending_fetches: u64,
    cancelled: bool,
    final_message: Option<AssistantMessage>,
}

impl Clone for DeferredEntry {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            step: self.step.clone(),
            context: self.context.clone(),
            options: self.options.clone(),
            model: self.model.clone(),
            pending_fetches: self.pending_fetches,
            cancelled: self.cancelled,
            final_message: self.final_message.clone(),
        }
    }
}

/// The shared state of one faux core, upstream's `createFauxCore` closure
/// state.
struct FauxCoreInner {
    api: String,
    provider: String,
    models: Vec<Model>,
    min_token_size: usize,
    max_token_size: usize,
    tokens_per_second: Option<f64>,
    deferred_options: Option<FauxDeferredOptions>,
    state: FauxProviderState,
    pending_responses: Mutex<VecDeque<FauxResponseStep>>,
    prompt_cache: Mutex<BTreeMap<String, String>>,
    deferred_responses: tokio::sync::Mutex<BTreeMap<String, DeferredEntry>>,
}

/// A faux provider core, upstream's `createFauxCore` return value: models,
/// the scripted queue, and the streaming machinery behind one shared state.
///
/// Implements [`types::ProviderStreams`], so it plugs into
/// `create_provider` directly.
#[derive(Clone)]
pub struct FauxCore(Arc<FauxCoreInner>);

impl std::fmt::Debug for FauxCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FauxCore")
            .field("api", &self.0.api)
            .field("provider", &self.0.provider)
            .field("models", &self.0.models.len())
            .finish_non_exhaustive()
    }
}

/// The system randomness of [`random_id`] and the chunk sizing, upstream's
/// `Math.random`.
#[expect(
    clippy::expect_used,
    reason = "system entropy is a platform facility; an entropy failure is a platform breakage, not a runtime condition"
)]
fn random_u64() -> u64 {
    let mut bytes = [0u8; 8];
    getrandom::fill(&mut bytes).expect("system random source");
    u64::from_le_bytes(bytes)
}

/// An opaque random id, upstream's `randomId`: `prefix:now:random`.
fn random_id(prefix: &str) -> String {
    let random = random_u64();
    format!("{prefix}:{}:{random:016x}", crate::auth::resolve::now_ms())
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "the chars-count / 4 ceil reproduces the JS number estimate; a non-negative count's ceil is non-negative and fits u64"
)]
#[must_use]
fn estimate_tokens(text: &str) -> u64 {
    (text.chars().count() as f64 / 4.0).ceil() as u64
}

fn content_to_text(content: &UserContent) -> String {
    match content {
        UserContent::Text(text) => text.clone(),
        UserContent::Blocks(blocks) => blocks
            .iter()
            .map(|block| match block {
                UserBlock::Text(text) => text.text.clone(),
                UserBlock::Image(image) => {
                    format!("[image:{}:{}]", image.mime_type, image.data.len())
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn image_to_text(image: &ImageContent) -> String {
    format!("[image:{}:{}]", image.mime_type, image.data.len())
}

fn assistant_content_to_text(content: &[AssistantBlock]) -> String {
    content
        .iter()
        .map(|block| match block {
            AssistantBlock::Text(text) => text.text.clone(),
            AssistantBlock::Thinking(thinking) => thinking.thinking.clone(),
            AssistantBlock::ToolCall(tool_call) => format!(
                "{}:{}",
                tool_call.name,
                serde_json::to_string(&tool_call.arguments).unwrap_or_default()
            ),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_result_to_text(message: &ToolResultMessage) -> String {
    let mut parts = vec![message.tool_name.clone()];
    for block in &message.content {
        match block {
            types::ToolResultBlock::Text(text) => parts.push(text.text.clone()),
            types::ToolResultBlock::Image(image) => parts.push(image_to_text(image)),
        }
    }
    parts.join("\n")
}

fn message_to_text(message: &Message) -> String {
    match message {
        Message::User(user) => content_to_text(&user.content),
        Message::Assistant(assistant) => assistant_content_to_text(&assistant.content),
        Message::ToolResult(tool_result) => tool_result_to_text(tool_result),
    }
}

fn serialize_context(context: &Context) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(system_prompt) = &context.system_prompt {
        parts.push(format!("system:{system_prompt}"));
    }
    for message in &context.messages {
        parts.push(format!(
            "{}:{}",
            message_role(message),
            message_to_text(message)
        ));
    }
    if let Some(tools) = context.tools.as_ref().filter(|tools| !tools.is_empty()) {
        parts.push(format!(
            "tools:{}",
            serde_json::to_string(tools).unwrap_or_default()
        ));
    }
    parts.join("\n\n")
}

const fn message_role(message: &Message) -> &str {
    match message {
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::ToolResult(_) => "toolResult",
    }
}

fn common_prefix_length(a: &str, b: &str) -> usize {
    let length = a.len().min(b.len());
    let mut index = 0;
    while index < length {
        let (a_bytes, b_bytes) = (a.as_bytes(), b.as_bytes());
        if a_bytes[index] != b_bytes[index] {
            break;
        }
        // Cuts land on char boundaries: walk back to one before comparing
        // past a code point's tail bytes.
        index += 1;
        while index < length && !a.is_char_boundary(index) {
            index += 1;
        }
    }
    index
}

/// The usage estimate applied to every resolved faux message, upstream's
/// `withUsageEstimate`: prompt tokens from the serialized context, output
/// tokens from the assistant content, and simulated prompt caching per
/// session id (never when `cacheRetention` is `none`).
fn with_usage_estimate(
    message: AssistantMessage,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
    prompt_cache: &Mutex<BTreeMap<String, String>>,
) -> AssistantMessage {
    let prompt_text = serialize_context(context);
    let prompt_tokens = estimate_tokens(&prompt_text);
    let output_tokens = estimate_tokens(&assistant_content_to_text(&message.content));
    let mut input = prompt_tokens;
    let mut cache_read = 0u64;
    let mut cache_write = 0u64;
    let session_id = options.and_then(|options| options.session_id.clone());

    if let Some(session_id) = session_id {
        // `undefined` retention still caches: only the explicit `none` opts
        // out, upstream's `options?.cacheRetention !== "none"`.
        if options.and_then(|options| options.cache_retention) != Some(CacheRetention::None) {
            let mut cache = prompt_cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(previous_prompt) = cache.get(&session_id) {
                let cached_chars = common_prefix_length(previous_prompt, &prompt_text);
                cache_read = estimate_tokens(&previous_prompt[..cached_chars]);
                cache_write = estimate_tokens(&prompt_text[cached_chars..]);
                input = prompt_tokens.saturating_sub(cache_read);
            } else {
                cache_write = prompt_tokens;
            }
            cache.insert(session_id, prompt_text);
        }
    }

    AssistantMessage {
        usage: types::Usage {
            input,
            output: output_tokens,
            cache_read,
            cache_write,
            total_tokens: input + output_tokens + cache_read + cache_write,
            cost: types::UsageCost::default(),
            ..message.usage
        },
        ..message
    }
}

/// An opaque random chunk-size draw, upstream's `Math.random` bound to the
/// byte-length arithmetic the sizing splits on.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the chunk-size draw only needs a bounded sample; a truncated draw is still a valid size"
)]
fn random_usize() -> usize {
    random_u64() as usize
}

/// The chunk sizing of a scripted stream, upstream's
/// `splitStringByTokenSize`: random sizes between the bounds, four
/// characters (bytes) per token, cutting on char boundaries.
fn split_string_by_token_size(
    text: &str,
    min_token_size: usize,
    max_token_size: usize,
) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut index = 0;
    while index < text.len() {
        let token_size = min_token_size + random_usize() % (max_token_size - min_token_size + 1);
        let char_size = std::cmp::max(1, token_size * 4);
        let mut end = std::cmp::min(index + char_size, text.len());
        // A chunk never splits a code point: extend to the boundary, where
        // upstream slices UTF-16 units and may split a surrogate pair.
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }
        chunks.push(text[index..end].to_owned());
        index = end;
    }
    if chunks.is_empty() {
        vec![String::new()]
    } else {
        chunks
    }
}

/// The api/provider/model rewrite of a resolved response, upstream's
/// `cloneMessage`: the scripted message's own ids never reach the caller.
fn clone_message(
    message: AssistantMessage,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    AssistantMessage {
        api: Api::from(api),
        provider: ProviderId::from(provider),
        model: model_id.to_owned(),
        ..message
    }
}

fn create_deferred_message(model: &Model, handle: &DeferredHandle) -> AssistantMessage {
    faux_core_message(
        &model.api.0,
        &model.provider.0,
        &model.id,
        StopReason::Deferred,
        None,
        Some(handle.clone()),
    )
}

fn create_error_message(
    message: &str,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    faux_core_message(
        api,
        provider,
        model_id,
        StopReason::Error,
        Some(message.to_owned()),
        None,
    )
}

/// The empty-content message a control path produces: deferred handles and
/// errors — content arrives only through the scripted response.
fn faux_core_message(
    api: &str,
    provider: &str,
    model_id: &str,
    stop_reason: StopReason,
    error_message: Option<String>,
    deferred: Option<DeferredHandle>,
) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: Api::from(api),
        provider: ProviderId::from(provider),
        model: model_id.to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: default_usage(),
        stop_reason,
        deferred,
        error_message,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: crate::auth::resolve::now_ms(),
    }
}

fn create_aborted_message(partial: &AssistantMessage) -> AssistantMessage {
    AssistantMessage {
        stop_reason: StopReason::Aborted,
        error_message: Some("Request was aborted".to_owned()),
        timestamp: crate::auth::resolve::now_ms(),
        ..partial.clone()
    }
}

/// The response the faux provider reports to the `onResponse` hooks,
/// upstream's `{ status: 200, headers: {} }`.
const fn faux_response() -> types::ProviderResponse {
    types::ProviderResponse {
        status: 200,
        headers: BTreeMap::new(),
    }
}

/// The pacing pause before one delta, upstream's `scheduleChunk`: a timer
/// at the estimated token rate, or a bare scheduling boundary when unpaced.
#[expect(
    clippy::cast_precision_loss,
    reason = "the pacing delay is a heuristic, the estimated tokens over the configured rate; the token estimate already rounds"
)]
async fn schedule_chunk(chunk: &str, tokens_per_second: Option<f64>) {
    match tokens_per_second {
        Some(tokens_per_second) if tokens_per_second > 0.0 => {
            let delay = (estimate_tokens(chunk) as f64 / tokens_per_second) * 1000.0;
            tokio::time::sleep(std::time::Duration::from_secs_f64(delay / 1000.0)).await;
        }
        _ => tokio::task::yield_now().await,
    }
}

/// Emit a mid-stream abort, upstream's abort arms: the error event carries
/// the aborted message and the stream ends resolving to it. Returns the
/// message the caller's early return rides on.
fn push_aborted(
    stream: &AssistantMessageEventStream,
    partial: &AssistantMessage,
) -> AssistantMessage {
    let aborted = create_aborted_message(partial);
    stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Aborted,
        error: aborted.clone(),
    });
    stream.end(Some(&aborted));
    aborted
}

/// Emit an error-terminal failure, upstream's catch blocks: the error event
/// carries the failing message and the stream resolves to it.
fn push_error_terminal(stream: &AssistantMessageEventStream, message: &AssistantMessage) {
    stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: message.clone(),
    });
    stream.end(Some(message));
}
/// Stream a scripted message as deltas over the event protocol, upstream's
/// `streamWithDeltas`: start, per-block start/delta/end (tool-call arguments
/// streaming as raw JSON chunks), then done — or an error event when the
/// scripted message itself failed or the request aborted.
#[expect(
    clippy::too_many_lines,
    reason = "the delta protocol mirrors upstream's streamWithDeltas block ladder one to one"
)]
async fn stream_with_deltas(
    stream: &AssistantMessageEventStream,
    message: AssistantMessage,
    min_token_size: usize,
    max_token_size: usize,
    tokens_per_second: Option<f64>,
    signal: Option<&tokio_util::sync::CancellationToken>,
) -> Result<(), FauxFactoryError> {
    let mut partial = message.clone();
    partial.content = Vec::new();
    partial.stop_reason = StopReason::Pending;
    if signal.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
        push_aborted(stream, &partial);
        return Ok(());
    }

    stream.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });

    for index in 0..message.content.len() {
        if signal.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
            push_aborted(stream, &partial);
            return Ok(());
        }

        let block = message.content[index].clone();

        if let AssistantBlock::Thinking(block) = block {
            partial
                .content
                .push(FauxContentBlock::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: None,
                    redacted: None,
                }));
            stream.push(AssistantMessageEvent::ThinkingStart {
                content_index: index as u64,
                partial: partial.clone(),
            });
            for chunk in split_string_by_token_size(&block.thinking, min_token_size, max_token_size)
            {
                schedule_chunk(&chunk, tokens_per_second).await;
                if signal.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
                    push_aborted(stream, &partial);
                    return Ok(());
                }
                if let AssistantBlock::Thinking(partial_block) = &mut partial.content[index] {
                    partial_block.thinking.push_str(&chunk);
                }
                stream.push(AssistantMessageEvent::ThinkingDelta {
                    content_index: index as u64,
                    delta: chunk,
                    partial: partial.clone(),
                });
            }
            stream.push(AssistantMessageEvent::ThinkingEnd {
                content_index: index as u64,
                content: block.thinking.clone(),
                partial: partial.clone(),
            });
            continue;
        }

        if let AssistantBlock::Text(block) = block.clone() {
            partial.content.push(FauxContentBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            }));
            stream.push(AssistantMessageEvent::TextStart {
                content_index: index as u64,
                partial: partial.clone(),
            });
            for chunk in split_string_by_token_size(&block.text, min_token_size, max_token_size) {
                schedule_chunk(&chunk, tokens_per_second).await;
                if signal.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
                    push_aborted(stream, &partial);
                    return Ok(());
                }
                if let AssistantBlock::Text(partial_block) = &mut partial.content[index] {
                    partial_block.text.push_str(&chunk);
                }
                stream.push(AssistantMessageEvent::TextDelta {
                    content_index: index as u64,
                    delta: chunk,
                    partial: partial.clone(),
                });
            }
            stream.push(AssistantMessageEvent::TextEnd {
                content_index: index as u64,
                content: block.text.clone(),
                partial: partial.clone(),
            });
            continue;
        }

        if let AssistantBlock::ToolCall(block) = block {
            partial.content.push(FauxContentBlock::ToolCall(ToolCall {
                id: block.id.clone(),
                name: block.name.clone(),
                arguments: serde_json::Map::new(),
                thought_signature: None,
                namespace: None,
            }));
            stream.push(AssistantMessageEvent::ToolcallStart {
                content_index: index as u64,
                partial: partial.clone(),
            });
            let arguments_json = serde_json::to_string(&block.arguments).unwrap_or_default();
            for chunk in split_string_by_token_size(&arguments_json, min_token_size, max_token_size)
            {
                schedule_chunk(&chunk, tokens_per_second).await;
                if signal.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
                    push_aborted(stream, &partial);
                    return Ok(());
                }
                stream.push(AssistantMessageEvent::ToolcallDelta {
                    content_index: index as u64,
                    delta: chunk,
                    partial: partial.clone(),
                });
            }
            if let AssistantBlock::ToolCall(partial_block) = &mut partial.content[index] {
                partial_block.arguments.clone_from(&block.arguments);
            }
            stream.push(AssistantMessageEvent::ToolcallEnd {
                content_index: index as u64,
                tool_call: block,
                partial: partial.clone(),
            });
        }
    }

    if message.stop_reason == StopReason::Pending {
        return Err("Faux response ended without a stop reason".into());
    }
    if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
        stream.push(AssistantMessageEvent::Error {
            reason: message.stop_reason,
            error: message.clone(),
        });
        stream.end(Some(&message));
        return Ok(());
    }

    stream.push(AssistantMessageEvent::Done {
        reason: message.stop_reason,
        message: message.clone(),
    });
    stream.end(Some(&message));
    Ok(())
}

impl FauxCore {
    /// The wire-API id this core registers under.
    #[must_use]
    pub fn api(&self) -> &str {
        &self.0.api
    }

    /// The provider id this core serves, upstream's `provider`.
    #[must_use]
    pub fn provider(&self) -> &str {
        &self.0.provider
    }

    /// The faux models, upstream's `models`.
    #[must_use]
    pub fn models(&self) -> &[Model] {
        &self.0.models
    }

    /// The first model, upstream's `getModel()`.
    ///
    /// # Panics
    /// Never: the model list carries at least one model by construction.
    #[must_use]
    pub fn first_model(&self) -> Model {
        self.0.models[0].clone()
    }

    /// One model by id, upstream's `getModel(id)`.
    #[must_use]
    pub fn model(&self, model_id: &str) -> Option<Model> {
        self.0
            .models
            .iter()
            .find(|candidate| candidate.id == model_id)
            .cloned()
    }

    /// The shared observable state, upstream's `state`.
    #[must_use]
    pub fn state(&self) -> &FauxProviderState {
        &self.0.state
    }

    /// Replace the queued responses, upstream's `setResponses`.
    pub fn set_responses<I: IntoIterator<Item = FauxResponseStep>>(&self, responses: I) {
        *self
            .0
            .pending_responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = responses.into_iter().collect();
    }

    /// Append queued responses, upstream's `appendResponses`.
    pub fn append_responses<I: IntoIterator<Item = FauxResponseStep>>(&self, responses: I) {
        self.0
            .pending_responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(responses);
    }

    /// How many responses remain queued, upstream's
    /// `getPendingResponseCount()`.
    #[must_use]
    pub fn get_pending_response_count(&self) -> usize {
        self.0
            .pending_responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// The queued responses, upstream's `pendingResponses`.
    fn pending_responses(&self) -> MutexGuard<'_, VecDeque<FauxResponseStep>> {
        self.0
            .pending_responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The response a queued step resolves to: factories run, messages pass
    /// through, both get the api/provider/model rewrite and the usage
    /// estimate, upstream's `resolveResponse`.
    async fn resolve_response(
        &self,
        step: FauxResponseStep,
        context: &Context,
        stream_options: Option<&SimpleStreamOptions>,
        request_model: &Model,
    ) -> Result<AssistantMessage, FauxFactoryError> {
        let resolved = match step {
            FauxResponseStep::Message(message) => message,
            FauxResponseStep::Factory(factory) => {
                factory(context, stream_options, &self.0.state, request_model).await?
            }
        };
        Ok(with_usage_estimate(
            clone_message(resolved, &self.0.api, &self.0.provider, &request_model.id),
            context,
            stream_options,
            &self.0.prompt_cache,
        ))
    }

    /// The stream body, upstream's `queueMicrotask` callback: the response
    /// hook, the exhausted-queue error, the deferred registration, or the
    /// scripted response streamed as deltas.
    async fn run_stream(
        self,
        outer: AssistantMessageEventStream,
        step: Option<FauxResponseStep>,
        model: Model,
        context: Context,
        stream_options: Option<SimpleStreamOptions>,
    ) {
        let outcome: Result<(), FauxFactoryError> = async {
            Self::fire_response_hook(
                stream_options
                    .as_ref()
                    .and_then(|options| options.transport_options.on_response.as_ref()),
                &model,
            )
            .await;
            let Some(step) = step else {
                let message = with_usage_estimate(
                    create_error_message(
                        "No more faux responses queued",
                        &self.0.api,
                        &self.0.provider,
                        &model.id,
                    ),
                    &context,
                    stream_options.as_ref(),
                    &self.0.prompt_cache,
                );
                push_error_terminal(&outer, &message);
                return Ok(());
            };

            if stream_options
                .as_ref()
                .and_then(|options| options.deferred.as_ref())
                .is_some()
            {
                let mut handle = DeferredHandle {
                    provider: model.provider.0.clone(),
                    model_id: model.id.clone(),
                    api: model.api.0.clone(),
                    id: random_id("deferred"),
                    expires_at: None,
                    poll_after_ms: None,
                    data: None,
                };
                if let Some(poll_after_ms) = self
                    .0
                    .deferred_options
                    .as_ref()
                    .and_then(|deferred| deferred.poll_after_ms)
                {
                    handle.poll_after_ms = Some(poll_after_ms);
                }
                // The pending-fetch count comes from the core's construction
                // options, not the request, upstream's
                // `options.deferred?.pendingFetches` closure capture.
                let pending_fetches = self
                    .0
                    .deferred_options
                    .as_ref()
                    .and_then(|deferred| deferred.pending_fetches)
                    .unwrap_or(0);
                self.0.deferred_responses.lock().await.insert(
                    handle.id.clone(),
                    DeferredEntry {
                        handle: handle.clone(),
                        step,
                        context: context.clone(),
                        options: stream_options.clone(),
                        model: model.clone(),
                        pending_fetches,
                        cancelled: false,
                        final_message: None,
                    },
                );
                self.stream_message_as_deltas(
                    &outer,
                    create_deferred_message(&model, &handle),
                    stream_options
                        .as_ref()
                        .and_then(|options| options.transport_options.signal.as_ref()),
                )
                .await?;
                return Ok(());
            }

            let message = self
                .resolve_response(step, &context, stream_options.as_ref(), &model)
                .await?;
            self.stream_message_as_deltas(
                &outer,
                message,
                stream_options
                    .as_ref()
                    .and_then(|options| options.transport_options.signal.as_ref()),
            )
            .await?;
            Ok(())
        }
        .await;
        if let Err(error) = outcome {
            let message =
                create_error_message(&error.to_string(), &self.0.api, &self.0.provider, &model.id);
            push_error_terminal(&outer, &message);
        }
    }

    /// Fire the response hook an operation carries, upstream's awaited
    /// `options?.onResponse?.({ status: 200, headers: {} }, model)`.
    async fn fire_response_hook(on_response: Option<&types::OnResponse>, model: &Model) {
        if let Some(on_response) = on_response {
            on_response
                .clone()
                .call(faux_response(), model.clone())
                .await;
        }
    }
    /// Stream `message` as deltas with the core's pacing bounds, the
    /// `streamWithDeltas` tail every dispatch path shares.
    async fn stream_message_as_deltas(
        &self,
        outer: &AssistantMessageEventStream,
        message: AssistantMessage,
        signal: Option<&tokio_util::sync::CancellationToken>,
    ) -> Result<(), FauxFactoryError> {
        stream_with_deltas(
            outer,
            message,
            self.0.min_token_size,
            self.0.max_token_size,
            self.0.tokens_per_second,
            signal,
        )
        .await
    }
    /// The stream entry, upstream's `stream`: pops the next scripted step,
    /// counts the call, and runs the body on a spawned task so the returned
    /// stream is live before the first event lands.
    fn stream_inner(
        &self,
        model: &Model,
        context: &Context,
        stream_options: Option<SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let outer = create_assistant_message_event_stream();
        let step = self.pending_responses().pop_front();
        self.0
            .state
            .call_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let core = self.clone();
        let task_stream = outer.clone();
        let model = model.clone();
        let context = context.clone();
        tokio::spawn(async move {
            core.run_stream(task_stream, step, model, context, stream_options)
                .await;
        });
        outer
    }
}

/// Builds a faux core, upstream's `createFauxCore`.
#[must_use]
pub fn create_faux_core(options: RegisterFauxProviderOptions) -> FauxCore {
    let api = options.api.unwrap_or_else(|| random_id(DEFAULT_API));
    let provider = options
        .provider
        .unwrap_or_else(|| DEFAULT_PROVIDER.to_owned());
    let min_token_size = std::cmp::max(
        1,
        std::cmp::min(
            options
                .token_size
                .as_ref()
                .and_then(|size| size.min)
                .unwrap_or(DEFAULT_MIN_TOKEN_SIZE),
            options
                .token_size
                .as_ref()
                .and_then(|size| size.max)
                .unwrap_or(DEFAULT_MAX_TOKEN_SIZE),
        ),
    );
    let max_token_size = std::cmp::max(
        min_token_size,
        options
            .token_size
            .as_ref()
            .and_then(|size| size.max)
            .unwrap_or(DEFAULT_MAX_TOKEN_SIZE),
    );
    let definitions: Vec<FauxModelDefinition> = if options.models.is_empty() {
        vec![FauxModelDefinition {
            id: DEFAULT_MODEL_ID.to_owned(),
            name: Some(DEFAULT_MODEL_NAME.to_owned()),
            reasoning: Some(false),
            input: Some(vec![Modality::Text, Modality::Image]),
            cost: Some(ModelCost::default()),
            context_window: Some(128_000),
            max_tokens: Some(16_384),
        }]
    } else {
        options.models
    };
    let models: Vec<Model> = definitions
        .iter()
        .map(|definition| Model {
            id: definition.id.clone(),
            name: definition
                .name
                .clone()
                .unwrap_or_else(|| definition.id.clone()),
            api: Api::from(api.as_str()),
            provider: ProviderId::from(provider.as_str()),
            base_url: DEFAULT_BASE_URL.to_owned(),
            reasoning: definition.reasoning.unwrap_or(false),
            thinking_level_map: None,
            input: definition
                .input
                .clone()
                .unwrap_or_else(|| vec![Modality::Text, Modality::Image]),
            cost: definition.cost.clone().unwrap_or_default(),
            context_window: definition.context_window.unwrap_or(128_000),
            max_tokens: definition.max_tokens.unwrap_or(16_384),
            sampling_params: None,
            headers: None,
            compat: None,
        })
        .collect();
    FauxCore(Arc::new(FauxCoreInner {
        api,
        provider,
        models,
        min_token_size,
        max_token_size,
        tokens_per_second: options.tokens_per_second,
        deferred_options: options.deferred,
        state: FauxProviderState::default(),
        pending_responses: Mutex::new(VecDeque::new()),
        prompt_cache: Mutex::new(BTreeMap::new()),
        deferred_responses: tokio::sync::Mutex::new(BTreeMap::new()),
    }))
}

impl types::ProviderStreams for FauxCore {
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        self.stream_inner(
            model,
            context,
            options.map(SimpleStreamOptions::from_stream),
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        self.stream_inner(model, context, options.cloned())
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: Option<&types::DeferredFetchOptions>,
    ) -> Option<AssistantMessageEventStream> {
        let outer = create_assistant_message_event_stream();
        self.0
            .state
            .deferred_fetch_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let core = self.clone();
        let task_stream = outer.clone();
        let model = model.clone();
        let handle = handle.clone();
        let fetch_options = options.cloned();
        tokio::spawn(async move {
            core.run_fetch_deferred(task_stream, model, handle, fetch_options)
                .await;
        });
        Some(outer)
    }

    fn cancel_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        options: Option<&'a types::DeferredCancelOptions>,
    ) -> types::BoxedFuture<'a, Result<(), ProviderRequestError>> {
        let core = self.clone();
        let model = model.clone();
        let handle = handle.clone();
        let options = options.cloned();
        Box::pin(async move {
            core.0.state.record_cancelled(handle.clone());
            if let Some(entry) = core.0.deferred_responses.lock().await.get_mut(&handle.id) {
                entry.cancelled = true;
            }
            Self::fire_response_hook(
                options
                    .as_ref()
                    .and_then(|options| options.transport_options.on_response.as_ref()),
                &model,
            )
            .await;
            Ok(())
        })
    }

    fn supports_fetch_deferred(&self) -> bool {
        true
    }

    fn supports_cancel_deferred(&self) -> bool {
        true
    }
}

impl FauxCore {
    /// The deferred-fetch body, upstream's `fetchDeferred` microtask: the
    /// response hook, the unknown-handle and cancelled checks, pending
    /// fetches that keep returning the handle, and the once-resolved final
    /// message streamed as deltas.
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the deferred map's guard spans the once-only resolution: concurrent fetches share one entry and one final message, upstream's unlocked single-threaded read-modify-write"
    )]
    async fn run_fetch_deferred(
        self,
        outer: AssistantMessageEventStream,
        model: Model,
        handle: DeferredHandle,
        fetch_options: Option<types::DeferredFetchOptions>,
    ) {
        let outcome: Result<(), FauxFactoryError> = async {
            Self::fire_response_hook(
                fetch_options
                    .as_ref()
                    .and_then(|options| options.transport_options.on_response.as_ref()),
                &model,
            )
            .await;
            let entry = {
                let map = self.0.deferred_responses.lock().await;
                map.get(&handle.id).cloned()
            };
            let Some(entry) = entry else {
                return Err(format!("Unknown faux deferred response: {}", handle.id).into());
            };
            if entry.handle.provider != handle.provider
                || entry.handle.model_id != handle.model_id
                || entry.handle.api != handle.api
            {
                return Err(format!("Unknown faux deferred response: {}", handle.id).into());
            }
            if entry.cancelled {
                return Err(format!("Faux deferred response was cancelled: {}", handle.id).into());
            }

            if entry.pending_fetches > 0 {
                {
                    let mut map = self.0.deferred_responses.lock().await;
                    if let Some(entry) = map.get_mut(&handle.id) {
                        entry.pending_fetches -= 1;
                    }
                }
                self.stream_message_as_deltas(
                    &outer,
                    create_deferred_message(&model, &entry.handle),
                    fetch_options
                        .as_ref()
                        .and_then(|options| options.transport_options.signal.as_ref()),
                )
                .await?;
                return Ok(());
            }

            let final_message = {
                let mut map = self.0.deferred_responses.lock().await;
                let Some(entry) = map.get_mut(&handle.id) else {
                    return Err(format!("Unknown faux deferred response: {}", handle.id).into());
                };
                if entry.final_message.is_none() {
                    let submission = entry.options.clone().map(|mut submission| {
                        submission.deferred = None;
                        submission.transport_options.signal = None;
                        submission.transport_options.on_response = None;
                        submission
                    });
                    entry.final_message = Some(
                        match self
                            .resolve_response(
                                entry.step.clone(),
                                &entry.context,
                                submission.as_ref(),
                                &entry.model,
                            )
                            .await
                        {
                            Ok(message) => message,
                            Err(error) => create_error_message(
                                &error.to_string(),
                                &self.0.api,
                                &self.0.provider,
                                &entry.model.id,
                            ),
                        },
                    );
                }
                entry.final_message.clone()
            }
            .ok_or_else(|| format!("Unknown faux deferred response: {}", handle.id))
            .map_err(FauxFactoryError::from)?;
            self.stream_message_as_deltas(
                &outer,
                final_message,
                fetch_options
                    .as_ref()
                    .and_then(|options| options.transport_options.signal.as_ref()),
            )
            .await?;
            Ok(())
        }
        .await;
        if let Err(error) = outcome {
            let message =
                create_error_message(&error.to_string(), &self.0.api, &self.0.provider, &model.id);
            push_error_terminal(&outer, &message);
        }
    }
}

/// The standalone faux provider, upstream's `fauxProvider`: a core behind a
/// `Provider` built with ambient api-key auth that always resolves
/// configured, ready for `Models.set_provider`.
///
/// ```rust
/// use pi_ai::models::create_models;
/// use pi_ai::providers::faux::{
///     faux_assistant_message, faux_provider, FauxAssistantMessageOptions, RegisterFauxProviderOptions,
/// };
///
/// let faux = faux_provider(RegisterFauxProviderOptions::default());
/// let models = create_models(None);
/// models.set_provider(std::sync::Arc::new(faux.provider.clone()));
/// faux.set_responses([faux_assistant_message("hi", FauxAssistantMessageOptions::default()).into()]);
/// ```
#[must_use]
pub fn faux_provider(options: RegisterFauxProviderOptions) -> FauxProviderHandle {
    let core = create_faux_core(options);
    let provider = models::create_provider(models::CreateProviderOptions {
        id: core.provider().to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: crate::auth::types::ProviderAuth {
            api_key: Some(crate::auth::types::ApiKeyAuth {
                name: "Faux".to_owned(),
                login: None,
                check: None,
                resolve: {
                    let resolve: crate::auth::types::ApiKeyResolveFn =
                        Arc::new(|_input: crate::auth::types::ApiKeyAuthInput| {
                            Box::pin(async { Ok(Some(crate::auth::types::AuthResult::default())) })
                        });
                    resolve
                },
            }),
            oauth: None,
        },
        models: core.models().to_vec(),
        fetch_models: None,
        filter_models: None,
        api: models::ProviderApi::Single(Arc::new(core.clone())),
    });
    FauxProviderHandle { provider, core }
}

/// A faux provider built with [`faux_provider`], upstream's
/// `FauxProviderHandle`: the `Provider` plus the core's scripting surface.
pub struct FauxProviderHandle {
    /// The provider for `Models.set_provider`, upstream's `provider`.
    pub provider: models::ProviderImpl,
    core: FauxCore,
}

impl std::fmt::Debug for FauxProviderHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FauxProviderHandle")
            .field("provider", &self.provider)
            .field("core", &self.core)
            .finish()
    }
}

impl FauxProviderHandle {
    /// The core, for the stream entry points and scripting surface.
    #[must_use]
    pub const fn core(&self) -> &FauxCore {
        &self.core
    }

    /// The wire-API id the core registered under, upstream's `api`.
    #[must_use]
    pub fn api(&self) -> &str {
        self.core.api()
    }

    /// The faux models, upstream's `models`.
    #[must_use]
    pub fn models(&self) -> &[Model] {
        self.core.models()
    }

    /// The first model, upstream's `getModel()`.
    #[must_use]
    pub fn first_model(&self) -> Model {
        self.core.first_model()
    }

    /// One model by id, upstream's `getModel(id)`.
    #[must_use]
    pub fn model(&self, model_id: &str) -> Option<Model> {
        self.core.model(model_id)
    }

    /// The shared observable state, upstream's `state`.
    #[must_use]
    pub fn state(&self) -> &FauxProviderState {
        self.core.state()
    }

    /// Replace the queued responses, upstream's `setResponses`.
    pub fn set_responses<I: IntoIterator<Item = FauxResponseStep>>(&self, responses: I) {
        self.core.set_responses(responses);
    }

    /// Append queued responses, upstream's `appendResponses`.
    pub fn append_responses<I: IntoIterator<Item = FauxResponseStep>>(&self, responses: I) {
        self.core.append_responses(responses);
    }

    /// How many responses remain queued, upstream's
    /// `getPendingResponseCount()`.
    #[must_use]
    pub fn get_pending_response_count(&self) -> usize {
        self.core.get_pending_response_count()
    }
}

/// A faux provider registered into the compat api-registry, upstream's
/// `FauxProviderRegistration`.
///
/// The core's scripting surface plus the `unregister` scoped to the source
/// id the registration was made under. Constructed only by
/// [`crate::compat::register_faux_provider`], upstream's definition site.
pub struct FauxProviderRegistration {
    core: FauxCore,
    source_id: String,
}

impl std::fmt::Debug for FauxProviderRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FauxProviderRegistration")
            .field("api", &self.core.api())
            .field("source_id", &self.source_id)
            .finish()
    }
}

impl FauxProviderRegistration {
    /// The registration handle, upstream's object literal in
    /// `registerFauxProvider`.
    pub(crate) const fn new(core: FauxCore, source_id: String) -> Self {
        Self { core, source_id }
    }

    /// The wire-API id the registration serves, upstream's `api`.
    #[must_use]
    pub fn api(&self) -> &str {
        self.core.api()
    }

    /// The faux models, upstream's `models`.
    #[must_use]
    pub fn models(&self) -> &[Model] {
        self.core.models()
    }

    /// The first model, upstream's `getModel()`.
    #[must_use]
    pub fn first_model(&self) -> Model {
        self.core.first_model()
    }

    /// One model by id, upstream's `getModel(id)`.
    #[must_use]
    pub fn model(&self, model_id: &str) -> Option<Model> {
        self.core.model(model_id)
    }

    /// The shared observable state, upstream's `state`.
    #[must_use]
    pub fn state(&self) -> &FauxProviderState {
        self.core.state()
    }

    /// Replace the queued responses, upstream's `setResponses`.
    pub fn set_responses<I: IntoIterator<Item = FauxResponseStep>>(&self, responses: I) {
        self.core.set_responses(responses);
    }

    /// Append queued responses, upstream's `appendResponses`.
    pub fn append_responses<I: IntoIterator<Item = FauxResponseStep>>(&self, responses: I) {
        self.core.append_responses(responses);
    }

    /// How many responses remain queued, upstream's
    /// `getPendingResponseCount()`.
    #[must_use]
    pub fn get_pending_response_count(&self) -> usize {
        self.core.get_pending_response_count()
    }

    /// Remove the registration, upstream's `unregister`: every api-registry
    /// entry made under this registration's source id is dropped, so the
    /// compat dispatch falls back to its no-provider error.
    pub fn unregister(&self) {
        crate::compat::unregister_api_providers(&self.source_id);
    }
}

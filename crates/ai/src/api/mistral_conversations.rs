//! The Mistral Conversations wire API, ported from
//! `packages/ai/src/api/mistral-conversations.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements:
//! - Despite the api id, the adapter speaks Chat Completions:
//!   `POST {baseUrl}/v1/chat/completions` over SSE — one request, no retry;
//!   upstream accepts `maxRetries` in its options type and never wires it.
//! - The camelCase "SDK-style" payload `onPayload` sees converts to the
//!   `snake_case` wire body afterwards, upstream's `toMistralWirePayload`; a
//!   hook replacement rides through the same conversion.
//! - `stripSymbolKeys` vanishes: serde JSON values carry no symbol-keyed
//!   decorations, the `TypeBox` artifacts upstream deletes before send.
//! - `sanitizeSurrogates` is statically upheld: Rust strings are valid
//!   UTF-8.
//! - Upstream's `AbortSignal.timeout`/`AbortSignal.any` collapse into the
//!   seam request's `timeout_ms` (defaulting to 60 s when unset) plus the
//!   cancellation token; the timeout surfaces with the upstream wording
//!   "The operation was aborted due to timeout".
//! - The hand-rolled `TextDecoder`/event-boundary matcher collapses into the
//!   shared [`SseStream`](crate::http::sse) decoder (recorded on the
//!   HttpClient-seam ticket); the `[DONE]` terminator and the object-with-
//!   `choices` validation stay here.
//! - `MistralHttpError` collapses into its formatted message: the seam has
//!   no `statusText`, so a body-less failure reads
//!   `Mistral API error (<status>): Request failed with status <status>`.

use std::collections::HashMap;

use bytes::Bytes;
use serde_json::{Value, json};

use crate::api::constrained_sampling::{
    get_json_schema_tool_parameters, resolve_json_schema_strict_sampling,
};
use crate::api::simple_options::build_base_options;
use crate::api::transform_messages::{ToolCallIdNormalizer, transform_messages};
use crate::api::wire_common::{
    call_response_hook, close_current_block, initial_output, missing_api_key_message,
    setup_error_stream, spawn_adapter_stream, upsert_header,
};
use crate::http::client::{HttpError, HttpMethod, HttpRequest, read_body_text};
use crate::types::{
    AssistantBlock, AssistantMessage, AssistantMessageEvent, Context, Message, Modality, Model,
    ModelThinkingLevel, SimpleStreamOptions, StopReason, StreamOptions, Tool, ToolCall,
    ToolResultBlock, UserBlock, UserContent,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::hash::short_hash;
use crate::utils::json_parse::{parse_json_with_repair, parse_streaming_json};
use crate::utils::pi_user_agent::get_pi_user_agent;

/// The normalized tool-call id length Mistral's wire expects.
const MISTRAL_TOOL_CALL_ID_LENGTH: usize = 9;
/// The error-body cap, upstream's `MAX_MISTRAL_ERROR_BODY_CHARS`.
const MAX_MISTRAL_ERROR_BODY_CHARS: usize = 4000;
/// The default request timeout, upstream's `AbortSignal.timeout` fallback.
const DEFAULT_TIMEOUT_MS: u64 = 60_000;

/// The tool choice Mistral accepts, upstream's
/// `MistralOptions["toolChoice"]`: the shared choices plus `any`,
/// `required`, and the OpenAI-style forced function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MistralToolChoice {
    /// Let the provider decide.
    Auto,
    /// Never call tools.
    None,
    /// The model must call a tool, upstream's `"any"`.
    Any,
    /// The model must call some tool, upstream's `"required"`.
    Required,
    /// Force a specific function, upstream's
    /// `{ type: "function", function: { name } }`.
    Function {
        /// The function to force.
        name: String,
    },
}

impl MistralToolChoice {
    /// The wire shape, upstream's `mapToolChoice` passthrough value.
    fn to_wire(&self) -> Value {
        match self {
            Self::Auto => json!("auto"),
            Self::None => json!("none"),
            Self::Any => json!("any"),
            Self::Required => json!("required"),
            Self::Function { name } => json!({
                "type": "function",
                "function": { "name": name },
            }),
        }
    }
}

/// The reasoning-effort values the reasoning-effort models accept, upstream's
/// `MistralReasoningEffort = "none" | "high"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MistralReasoningEffort {
    /// No reasoning.
    None,
    /// Full reasoning.
    High,
}

impl MistralReasoningEffort {
    /// The wire spelling.
    const fn as_wire(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::High => "high",
        }
    }
}

/// The prompt mode for Magistral-family reasoning, upstream's
/// `promptMode?: "reasoning"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MistralPromptMode {
    /// The model reasons into a `thinking` content channel.
    Reasoning,
}

impl MistralPromptMode {
    /// The wire spelling.
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Reasoning => "reasoning",
        }
    }
}

/// The adapter-facing options, upstream's `MistralOptions`.
#[derive(Clone, Debug, Default)]
pub struct MistralStreamOptions {
    /// The transport seam: the request's HTTP client, cancellation token,
    /// and lifecycle callbacks.
    pub transport_options: crate::types::TransportOptions,
    /// The API key, overriding credential resolution.
    pub api_key: Option<String>,
    /// Explicit parent context for telemetry produced by this logical request.
    pub telemetry_context: Option<pi_telemetry::TelemetryHandle>,
    /// Provider-scoped environment values; these take precedence over the
    /// process environment.
    pub env: Option<crate::types::ProviderEnv>,
    /// Custom HTTP headers merged with provider defaults.
    pub headers: Option<crate::types::ProviderHeaders>,
    /// HTTP request timeout in milliseconds; defaults to 60 s when unset.
    pub timeout_ms: Option<u64>,
    /// Accepted upstream-side and never used: this adapter issues one
    /// request with no client-side retry.
    pub max_retries: Option<u32>,
    /// Accepted upstream-side and never used.
    pub max_retry_delay_ms: Option<u64>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Arbitrary sampling parameters; upstream's Mistral options shape has
    /// no `samplingParams` field, so the merge is dropped.
    pub sampling_params: Option<std::collections::BTreeMap<String, Value>>,
    /// Maximum output tokens.
    pub max_tokens: Option<u64>,
    /// Preferred transport; SSE is the only transport this adapter speaks.
    pub transport: Option<crate::types::Transport>,
    /// Prompt cache retention preference. Default: `"short"`.
    pub cache_retention: Option<crate::types::CacheRetention>,
    /// Optional session identifier; rides as `prompt_cache_key` and the
    /// `x-affinity` header when caching is on.
    pub session_id: Option<String>,
    /// WebSocket connect timeout in milliseconds.
    pub websocket_connect_timeout_ms: Option<u64>,
    /// Optional metadata to include in API requests.
    pub metadata: Option<std::collections::BTreeMap<String, Value>>,
    /// The tool choice, upstream's broader Mistral union.
    pub tool_choice: Option<MistralToolChoice>,
    /// The reasoning control for Magistral-family models.
    pub prompt_mode: Option<MistralPromptMode>,
    /// The reasoning-effort control for Mistral Small/Medium and GLM
    /// models.
    pub reasoning_effort: Option<MistralReasoningEffort>,
}

impl From<StreamOptions> for MistralStreamOptions {
    fn from(options: StreamOptions) -> Self {
        let StreamOptions {
            transport_options,
            api_key,
            telemetry_context,
            env,
            headers,
            timeout_ms,
            max_retries: _,
            max_retry_delay_ms: _,
            temperature,
            sampling_params: _,
            max_tokens,
            transport,
            cache_retention,
            session_id,
            websocket_connect_timeout_ms: _,
            metadata,
        } = options;
        Self {
            transport_options,
            api_key,
            telemetry_context,
            env,
            headers,
            timeout_ms,
            max_retries: None,
            max_retry_delay_ms: None,
            temperature,
            sampling_params: None,
            max_tokens,
            transport,
            cache_retention,
            session_id,
            websocket_connect_timeout_ms: None,
            metadata,
            tool_choice: None,
            prompt_mode: None,
            reasoning_effort: None,
        }
    }
}

/// The Mistral Conversations streams, upstream's `mistralConversationsApi()`.
#[derive(Debug, Default)]
pub struct MistralStreams;

impl crate::api::wire_common::TransportCarrier for MistralStreamOptions {
    fn transport_options(&self) -> &crate::types::TransportOptions {
        &self.transport_options
    }
}

crate::api::wire_common::forward_provider_streams!(MistralStreams, MistralStreamOptions);

/// Stream an assistant response, upstream's `stream` export.
#[must_use]
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&MistralStreamOptions>,
) -> AssistantMessageEventStream {
    spawn_adapter_stream(
        model,
        context,
        options.cloned(),
        |model, _| initial_output(model),
        |model, context, options, output, events| {
            Box::pin(run_stream(model, context, options, output, events))
        },
    )
}

/// Stream a simple assistant response, upstream's `streamSimple` export:
/// picks the reasoning control per model family and resolves the pi level.
///
/// Upstream throws synchronously when the key is missing; the port encodes
/// that failure as a settled error stream per the stream contract.
#[must_use]
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let Some(api_key) = options
        .as_ref()
        .and_then(|options| options.api_key.as_deref())
    else {
        return setup_error_stream(model, &missing_api_key_message(&model.provider));
    };

    let mut base =
        MistralStreamOptions::from(build_base_options(model, context, options, Some(api_key)));
    base.tool_choice = options
        .and_then(|options| options.tool_choice)
        .map(|choice| match choice {
            crate::types::ToolChoice::Auto => MistralToolChoice::Auto,
            crate::types::ToolChoice::None => MistralToolChoice::None,
        });

    let clamped = options
        .and_then(|options| options.reasoning)
        .map(|reasoning| {
            crate::models::clamp_thinking_level(model, ModelThinkingLevel::from(reasoning))
        });
    let reasoning = match clamped {
        Some(ModelThinkingLevel::Off) | None => None,
        Some(level) => Some(level),
    };
    let should_use_reasoning = model.reasoning && reasoning.is_some();

    if should_use_reasoning && uses_prompt_mode_reasoning(model) {
        base.prompt_mode = Some(MistralPromptMode::Reasoning);
    }
    if should_use_reasoning && uses_reasoning_effort(model) {
        base.reasoning_effort = Some(map_reasoning_effort(model, reasoning));
    }
    stream(model, context, Some(&base))
}

async fn run_stream(
    model: &Model,
    context: &Context,
    options: &MistralStreamOptions,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    let Some(api_key) = options.api_key.clone().filter(|key| !key.is_empty()) else {
        return Err(missing_api_key_message(&model.provider));
    };

    // The normalizer state is synchronous scratch; the scoped block drops it
    // before the first await so the stream future stays Send.
    let transformed = {
        let normalizer_state = std::cell::RefCell::new(create_tool_call_id_normalizer());
        let normalizer: &ToolCallIdNormalizer<'_> =
            &|id: &str, _model: &Model, _source: &AssistantMessage| {
                normalizer_state.borrow_mut()(id)
            };
        transform_messages(context.messages.clone(), model, Some(normalizer))
    };

    let mut payload = build_chat_payload(model, context, &transformed, options)?;
    if let Some(hook) = &options.transport_options.on_payload {
        payload = hook
            .call(payload.clone(), model.clone())
            .await
            .unwrap_or(payload);
    }

    let response = dispatch_stream_request(model, &payload, options, &api_key).await?;
    // The start event pushes only after the HTTP response succeeds:
    // pre-request failures settle without a start.
    events.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });
    let mut scratch = ToolCallScratch::default();
    consume_chat_stream(model, response, output, events, &mut scratch).await?;
    finish_stream(options, output, events)
}

/// The streaming scratch buffers, upstream's `partialArgs` fields plus the
/// tool-block dedup map. The blocks on the accumulator never carry them.
#[derive(Default)]
struct ToolCallScratch {
    /// Dedup key (stream index or call id) to content index, insertion
    /// ordered like upstream's Map.
    blocks: Vec<(ToolBlockKey, usize)>,
    /// The accumulated raw argument JSON per content index.
    partial_args: HashMap<usize, String>,
}

/// The dedup key of a streamed tool call, upstream's
/// `toolCall.index ?? callId`.
#[derive(Clone, PartialEq, Eq, Hash)]
enum ToolBlockKey {
    /// The stream's `index` field.
    Index(u64),
    /// The derived call id when no index rides.
    Id(String),
}

/// Build the camelCase chat payload, upstream's `buildChatPayload`; the
/// `snake_case` wire conversion happens in [`to_wire_payload`].
///
/// # Errors
/// When a tool requires strict sampling that cannot be resolved, or when the
/// request is already aborted.
fn build_chat_payload(
    model: &Model,
    context: &Context,
    transformed_messages: &[Message],
    options: &MistralStreamOptions,
) -> Result<Value, String> {
    let supports_images = model.input.contains(&Modality::Image);
    let mut messages = to_chat_messages(transformed_messages, supports_images);
    if let Some(system_prompt) = &context.system_prompt {
        messages.insert(0, json!({ "role": "system", "content": system_prompt }));
    }

    let mut payload = json!({
        "model": model.id,
        "stream": true,
        "messages": messages,
    });
    #[expect(
        clippy::expect_used,
        reason = "the json! literal above is an object by construction"
    )]
    let object = payload.as_object_mut().expect("payload object");

    let tools = context.tools.as_deref().unwrap_or_default();
    if !tools.is_empty() {
        object.insert("tools".to_owned(), to_function_tools(tools)?);
    }
    if let Some(temperature) = options.temperature {
        object.insert("temperature".to_owned(), json!(temperature));
    }
    if let Some(max_tokens) = options.max_tokens {
        object.insert("maxTokens".to_owned(), json!(max_tokens));
    }
    if let Some(tool_choice) = &options.tool_choice {
        object.insert("toolChoice".to_owned(), tool_choice.clone().to_wire());
    }
    if let Some(prompt_mode) = &options.prompt_mode {
        object.insert("promptMode".to_owned(), json!(prompt_mode.as_wire()));
    }
    if let Some(reasoning_effort) = &options.reasoning_effort {
        object.insert(
            "reasoningEffort".to_owned(),
            json!(reasoning_effort.as_wire()),
        );
    }
    if let Some(session_id) = should_use_prompt_caching(options) {
        object.insert("promptCacheKey".to_owned(), json!(session_id));
    }

    if options.transport_options.signal().is_cancelled() {
        return Err("Request aborted".to_owned());
    }
    Ok(payload)
}

/// Whether session-affinity prompt caching applies and the session id,
/// upstream's `shouldUsePromptCaching`.
fn should_use_prompt_caching(options: &MistralStreamOptions) -> Option<&str> {
    if options.cache_retention == Some(crate::types::CacheRetention::None) {
        return None;
    }
    options.session_id.as_deref()
}

/// Convert the camelCase payload to the `snake_case` wire body, upstream's
/// `toMistralWirePayload`.
fn to_wire_payload(payload: &Value) -> Value {
    let Some(object) = payload.as_object() else {
        return payload.clone();
    };
    let mut wire = object.clone();
    for (source, target) in [
        ("topP", "top_p"),
        ("maxTokens", "max_tokens"),
        ("randomSeed", "random_seed"),
        ("responseFormat", "response_format"),
        ("toolChoice", "tool_choice"),
        ("presencePenalty", "presence_penalty"),
        ("frequencyPenalty", "frequency_penalty"),
        ("parallelToolCalls", "parallel_tool_calls"),
        ("reasoningEffort", "reasoning_effort"),
        ("promptMode", "prompt_mode"),
        ("promptCacheKey", "prompt_cache_key"),
        ("safePrompt", "safe_prompt"),
    ] {
        remap_property(&mut wire, source, target);
    }
    if let Some(messages) = wire.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages.iter_mut() {
            *message = to_wire_message(message);
        }
    }
    if let Some(response_format) = wire.get_mut("response_format")
        && let Some(response_format) = response_format.as_object_mut()
    {
        remap_property(response_format, "jsonSchema", "json_schema");
        if let Some(json_schema) = response_format.get_mut("json_schema")
            && let Some(json_schema) = json_schema.as_object_mut()
        {
            remap_property(json_schema, "schemaDefinition", "schema");
        }
    }
    Value::Object(wire)
}

fn to_wire_message(message: &Value) -> Value {
    let mut wire = message.as_object().cloned().unwrap_or_default();
    remap_property(&mut wire, "toolCalls", "tool_calls");
    remap_property(&mut wire, "toolCallId", "tool_call_id");
    if let Some(content) = wire.get_mut("content").and_then(Value::as_array_mut) {
        for chunk in content.iter_mut() {
            let Some(object) = chunk.as_object_mut() else {
                continue;
            };
            for (source, target) in [
                ("imageUrl", "image_url"),
                ("documentUrl", "document_url"),
                ("documentName", "document_name"),
                ("fileId", "file_id"),
                ("referenceIds", "reference_ids"),
                ("inputAudio", "input_audio"),
            ] {
                remap_property(&mut *object, source, target);
            }
        }
    }
    Value::Object(wire)
}

fn remap_property(record: &mut serde_json::Map<String, Value>, source: &str, target: &str) {
    if let Some(value) = record.remove(source) {
        record.insert(target.to_owned(), value);
    }
}

/// The chat messages in Mistral's wire shape, upstream's `toChatMessages`.
#[expect(
    clippy::too_many_lines,
    reason = "the per-role conversion mirrors upstream's toChatMessages arm for arm"
)]
fn to_chat_messages(messages: &[Message], supports_images: bool) -> Vec<Value> {
    let mut result: Vec<Value> = Vec::new();

    for msg in messages {
        match msg {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => {
                    result.push(json!({ "role": "user", "content": text }));
                }
                UserContent::Blocks(blocks) => {
                    let had_images = blocks
                        .iter()
                        .any(|block| matches!(block, UserBlock::Image(_)));
                    let content: Vec<Value> = blocks
                        .iter()
                        .filter(|block| match block {
                            UserBlock::Text(_) => true,
                            UserBlock::Image(_) => supports_images,
                        })
                        .map(|block| match block {
                            UserBlock::Text(text) => json!({ "type": "text", "text": text.text }),
                            UserBlock::Image(image) => json!({
                                "type": "image_url",
                                "imageUrl": format!("data:{};base64,{}", image.mime_type, image.data),
                            }),
                        })
                        .collect();
                    if !content.is_empty() {
                        result.push(json!({ "role": "user", "content": content }));
                        continue;
                    }
                    if had_images && !supports_images {
                        result.push(json!({
                            "role": "user",
                            "content": "(image omitted: model does not support images)",
                        }));
                    }
                }
            },
            Message::Assistant(assistant) => {
                let mut content_parts: Vec<Value> = Vec::new();
                let mut tool_calls: Vec<Value> = Vec::new();

                for block in &assistant.content {
                    match block {
                        AssistantBlock::Text(text) => {
                            if !text.text.trim().is_empty() {
                                content_parts.push(json!({ "type": "text", "text": text.text }));
                            }
                        }
                        AssistantBlock::Thinking(thinking)
                            if !thinking.thinking.trim().is_empty() =>
                        {
                            content_parts.push(json!({
                                "type": "thinking",
                                "thinking": [{ "type": "text", "text": thinking.thinking }],
                            }));
                        }
                        AssistantBlock::Thinking(_) => {}
                        AssistantBlock::ToolCall(call) => {
                            tool_calls.push(json!({
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": Value::Object(call.arguments.clone()).to_string(),
                                },
                                "index": 0,
                            }));
                        }
                    }
                }

                let mut assistant_message = serde_json::Map::new();
                assistant_message.insert("role".to_owned(), json!("assistant"));
                // `prefix: false` is always present on replayed turns.
                assistant_message.insert("prefix".to_owned(), json!(false));
                if !content_parts.is_empty() {
                    assistant_message.insert("content".to_owned(), json!(content_parts));
                }
                if !tool_calls.is_empty() {
                    assistant_message.insert("toolCalls".to_owned(), json!(tool_calls));
                }
                if !content_parts.is_empty() || !tool_calls.is_empty() {
                    result.push(Value::Object(assistant_message));
                }
            }
            Message::ToolResult(result_message) => {
                let text_result = result_message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ToolResultBlock::Text(text) => Some(text.text.as_str()),
                        ToolResultBlock::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_images = result_message
                    .content
                    .iter()
                    .any(|block| matches!(block, ToolResultBlock::Image(_)));
                let tool_text = build_tool_result_text(
                    &text_result,
                    has_images,
                    supports_images,
                    result_message.is_error,
                );
                let mut tool_content = vec![json!({ "type": "text", "text": tool_text })];
                if supports_images {
                    for block in &result_message.content {
                        if let ToolResultBlock::Image(image) = block {
                            tool_content.push(json!({
                                "type": "image_url",
                                "imageUrl": format!("data:{};base64,{}", image.mime_type, image.data),
                            }));
                        }
                    }
                }
                result.push(json!({
                    "role": "tool",
                    "toolCallId": result_message.tool_call_id,
                    "name": result_message.tool_name,
                    "content": tool_content,
                }));
            }
        }
    }

    result
}

/// The tool-result text with the error/omitted-image wording, upstream's
/// `buildToolResultText`.
fn build_tool_result_text(
    text: &str,
    has_images: bool,
    supports_images: bool,
    is_error: bool,
) -> String {
    let trimmed = text.trim();
    let error_prefix = if is_error { "[tool error] " } else { "" };

    if !trimmed.is_empty() {
        let image_suffix = if has_images && !supports_images {
            "\n[tool image omitted: model does not support images]"
        } else {
            ""
        };
        return format!("{error_prefix}{trimmed}{image_suffix}");
    }

    if has_images {
        if supports_images {
            return if is_error {
                "[tool error] (see attached image)".to_owned()
            } else {
                "(see attached image)".to_owned()
            };
        }
        return if is_error {
            "[tool error] (image omitted: model does not support images)".to_owned()
        } else {
            "(image omitted: model does not support images)".to_owned()
        };
    }

    if is_error {
        "[tool error] (no tool output)".to_owned()
    } else {
        "(no tool output)".to_owned()
    }
}

/// The function tools, upstream's `toFunctionTools`: the strictness
/// resolution and parameters ride the shared constrained-sampling seam.
///
/// # Errors
/// When a tool requires strict sampling that cannot be resolved.
fn to_function_tools(tools: &[Tool]) -> Result<Value, String> {
    let mut wire = Vec::with_capacity(tools.len());
    for tool in tools {
        let strict = resolve_json_schema_strict_sampling(tool, true)?;
        let parameters = get_json_schema_tool_parameters(tool, strict).map_err(|error| error.0)?;
        wire.push(json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
                "strict": strict.unwrap_or(false),
            },
        }));
    }
    Ok(json!(wire))
}

/// The memoizing tool-call id normalizer, upstream's
/// `createMistralToolCallIdNormalizer`: a bidirectional map so every
/// occurrence of one id derives identically, with collision attempts walking
/// until the candidate is unowned.
fn create_tool_call_id_normalizer() -> impl FnMut(&str) -> String {
    let mut id_map: HashMap<String, String> = HashMap::new();
    let mut reverse_map: HashMap<String, String> = HashMap::new();
    move |id: &str| -> String {
        if let Some(existing) = id_map.get(id) {
            return existing.clone();
        }
        let mut attempt = 0;
        loop {
            let candidate = derive_tool_call_id(id, attempt);
            let owner = reverse_map.get(&candidate).cloned();
            match owner {
                None => {
                    reverse_map.insert(candidate.clone(), id.to_owned());
                    id_map.insert(id.to_owned(), candidate);
                    return id.to_owned();
                }
                Some(owner) if owner == id => {
                    id_map.insert(id.to_owned(), candidate);
                    return id.to_owned();
                }
                Some(_) => attempt += 1,
            }
        }
    }
}

/// The 9-character id derivation, upstream's `deriveMistralToolCallId`.
fn derive_tool_call_id(id: &str, attempt: u32) -> String {
    let normalized: String = id.chars().filter(char::is_ascii_alphanumeric).collect();
    if attempt == 0 && normalized.chars().count() == MISTRAL_TOOL_CALL_ID_LENGTH {
        return normalized;
    }
    let seed_base = if normalized.is_empty() {
        id.to_owned()
    } else {
        normalized
    };
    let seed = if attempt == 0 {
        seed_base
    } else {
        format!("{seed_base}:{attempt}")
    };
    short_hash(&seed)
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(MISTRAL_TOOL_CALL_ID_LENGTH)
        .collect()
}

/// The streaming state machine, upstream's `consumeChatStream`.
async fn consume_chat_stream(
    model: &Model,
    response: crate::http::client::HttpResponse,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
    scratch: &mut ToolCallScratch,
) -> Result<(), String> {
    let mut sse = crate::http::sse::SseStream::new(response.body);
    let mut current_block: Option<bool> = None;
    while let Some(sse_event) = sse.next().await.map_err(mistral_error_from_http)? {
        let data = sse_event.data.trim().to_owned();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }
        let chunk = parse_json_with_repair(&data)
            .map_err(|error| format!("Invalid Mistral streaming event: {error}"))?;
        if !chunk.is_object() || !chunk.get("choices").is_some_and(Value::is_array) {
            return Err("Invalid Mistral streaming event".to_owned());
        }

        // Mistral's streamed CompletionChunk carries an id field; keep the
        // first non-empty one, the stable response identifier per stream.
        if let Some(id) = chunk.get("id").and_then(Value::as_str)
            && !id.is_empty()
            && output.response_id.as_ref().is_none_or(String::is_empty)
        {
            output.response_id = Some(id.to_owned());
        }

        if let Some(usage) = chunk.get("usage") {
            let prompt_tokens = wire_u64(usage, "prompt_tokens");
            let cached = cached_prompt_tokens(usage, prompt_tokens);
            output.usage.input = prompt_tokens.saturating_sub(cached);
            output.usage.output = wire_u64(usage, "completion_tokens");
            output.usage.cache_read = cached;
            output.usage.cache_write = 0;
            output.usage.total_tokens = match wire_u64(usage, "total_tokens") {
                0 => {
                    output.usage.input
                        + output.usage.output
                        + output.usage.cache_read
                        + output.usage.cache_write
                }
                total => total,
            };
            crate::models::calculate_cost(model, &mut output.usage);
        }

        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            continue;
        };

        // A null/empty finish reason is a no-op, upstream's truthiness gate.
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str)
            && !reason.is_empty()
        {
            output.raw_stop_reason = Some(reason.to_owned());
            let (stop_reason, error_message) = map_chat_stop_reason(Some(reason));
            output.stop_reason = stop_reason;
            if let Some(error_message) = error_message {
                output.error_message = Some(error_message);
            }
        }

        let delta = choice.get("delta");
        if let Some(content) = delta.and_then(|delta| delta.get("content"))
            && !content.is_null()
        {
            apply_content_delta(content, output, &mut current_block, events);
        }

        let tool_calls = delta
            .and_then(|delta| delta.get("tool_calls"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for tool_call in &tool_calls {
            apply_tool_call_delta(tool_call, output, &mut current_block, scratch, events);
        }
    }

    // The loop settled: close the open text/thinking block, then finalize
    // every tool block with its fully parsed arguments.
    close_current_block(&mut current_block, output, events);
    let ordered: Vec<usize> = scratch.blocks.iter().map(|(_, index)| *index).collect();
    for &content_index in &ordered {
        let Some(block) = output.content.get_mut(content_index) else {
            continue;
        };
        let AssistantBlock::ToolCall(tool_call) = block else {
            continue;
        };
        tool_call.arguments = parse_streaming_json_args(scratch.partial_args.get(&content_index));
    }
    for (content_index, tool_call) in finalized_tool_calls(output, &ordered, scratch) {
        events.push(AssistantMessageEvent::ToolcallEnd {
            content_index: content_index as u64,
            tool_call,
            partial: output.clone(),
        });
    }
    Ok(())
}

/// The authoritative tool calls re-parsed from the scratch buffers, in
/// content order.
fn finalized_tool_calls(
    output: &AssistantMessage,
    ordered: &[usize],
    scratch: &ToolCallScratch,
) -> Vec<(usize, ToolCall)> {
    let _ = scratch;
    ordered
        .iter()
        .filter_map(|index| {
            let AssistantBlock::ToolCall(call) = &output.content[*index] else {
                return None;
            };
            Some((*index, call.clone()))
        })
        .collect()
}

/// The upstream helper on a stream that never carries one; the empty parse
/// shape keeps the call contract.
fn parse_streaming_json_args(partial: Option<&String>) -> serde_json::Map<String, Value> {
    parse_streaming_json(partial.map(String::as_str))
        .as_object()
        .cloned()
        .unwrap_or_default()
}

/// The parsed arguments map at each delta, upstream's per-delta
/// `parseStreamingJson` re-parse.
fn apply_content_delta(
    content: &Value,
    output: &mut AssistantMessage,
    current_block: &mut Option<bool>,
    events: &AssistantMessageEventStream,
) {
    match content {
        Value::String(text) => {
            apply_text_delta(text, output, current_block, events);
        }
        Value::Array(items) => {
            for item in items {
                match item {
                    Value::String(text) => apply_text_delta(text, output, current_block, events),
                    Value::Object(object) => match object.get("type").and_then(Value::as_str) {
                        Some("thinking") => {
                            let delta_text = object
                                .get("thinking")
                                .and_then(Value::as_array)
                                .map(|parts| {
                                    parts
                                        .iter()
                                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                                        .collect::<Vec<_>>()
                                        .join("")
                                })
                                .unwrap_or_default();
                            if delta_text.is_empty() {
                                continue;
                            }
                            apply_thinking_delta(&delta_text, output, current_block, events);
                        }
                        Some("text") => {
                            let text = object.get("text").and_then(Value::as_str).unwrap_or("");
                            apply_text_delta(text, output, current_block, events);
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

fn apply_text_delta(
    text: &str,
    output: &mut AssistantMessage,
    current_block: &mut Option<bool>,
    events: &AssistantMessageEventStream,
) {
    crate::api::wire_common::open_stream_block(current_block, output, events, false);
    let content_index = output.content.len() as u64 - 1;
    if let Some(AssistantBlock::Text(block)) = output.content.last_mut() {
        block.text.push_str(text);
    }
    events.push(AssistantMessageEvent::TextDelta {
        content_index,
        delta: text.to_owned(),
        partial: output.clone(),
    });
}

fn apply_thinking_delta(
    thinking: &str,
    output: &mut AssistantMessage,
    current_block: &mut Option<bool>,
    events: &AssistantMessageEventStream,
) {
    crate::api::wire_common::open_stream_block(current_block, output, events, true);
    let content_index = output.content.len() as u64 - 1;
    if let Some(AssistantBlock::Thinking(block)) = output.content.last_mut() {
        block.thinking.push_str(thinking);
    }
    events.push(AssistantMessageEvent::ThinkingDelta {
        content_index,
        delta: thinking.to_owned(),
        partial: output.clone(),
    });
}

/// One streamed tool-call delta: closes any open block, dedups by stream
/// index or call id, accumulates the raw arguments, and re-parses.
fn apply_tool_call_delta(
    tool_call: &Value,
    output: &mut AssistantMessage,
    current_block: &mut Option<bool>,
    scratch: &mut ToolCallScratch,
    events: &AssistantMessageEventStream,
) {
    if current_block.is_some() {
        close_current_block(current_block, output, events);
    }
    let raw_id = tool_call.get("id").and_then(Value::as_str);
    let call_id = match raw_id {
        Some(id) if !id.is_empty() && id != "null" => id.to_owned(),
        _ => derive_tool_call_id(
            &format!(
                "toolcall:{}",
                tool_call.get("index").and_then(Value::as_u64).unwrap_or(0)
            ),
            0,
        ),
    };
    let key = tool_call
        .get("index")
        .and_then(Value::as_u64)
        .map_or_else(|| ToolBlockKey::Id(call_id.clone()), ToolBlockKey::Index);

    let existing_index = scratch
        .blocks
        .iter()
        .find(|(key_entry, _)| *key_entry == key)
        .map(|(_, index)| *index);
    let content_index = if let Some(index) = existing_index {
        index
    } else {
        let name = tool_call
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        output.content.push(AssistantBlock::ToolCall(ToolCall {
            id: call_id,
            name: name.to_owned(),
            arguments: serde_json::Map::new(),
            thought_signature: None,
            namespace: None,
        }));
        let index = output.content.len() - 1;
        scratch.blocks.push((key, index));
        events.push(AssistantMessageEvent::ToolcallStart {
            content_index: index as u64,
            partial: output.clone(),
        });
        index
    };

    let arguments = tool_call
        .get("function")
        .and_then(|function| function.get("arguments"));
    let args_delta = match arguments {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Object(object)) => Value::Object(object.clone()).to_string(),
        Some(_) | None => String::new(),
    };
    let entry = scratch.partial_args.entry(content_index).or_default();
    entry.push_str(&args_delta);
    let parsed = parse_streaming_json(Some(entry.as_str()))
        .as_object()
        .cloned()
        .unwrap_or_default();
    if let Some(AssistantBlock::ToolCall(block)) = output.content.get_mut(content_index) {
        block.arguments = parsed;
    }
    events.push(AssistantMessageEvent::ToolcallDelta {
        content_index: content_index as u64,
        delta: args_delta,
        partial: output.clone(),
    });
}

/// The reasoning-effort model families, upstream's `usesReasoningEffort`.
fn uses_reasoning_effort(model: &Model) -> bool {
    model.id == "mistral-small-2603"
        || model.id == "mistral-small-latest"
        || model.id.starts_with("mistral-medium-")
        || model.id == "zai-glm-5-2"
}

/// The Magistral-family prompt-mode rule, upstream's
/// `usesPromptModeReasoning`.
fn uses_prompt_mode_reasoning(model: &Model) -> bool {
    model.reasoning && !uses_reasoning_effort(model)
}

/// The reasoning effort for the clamped level, upstream's
/// `mapReasoningEffort`: the model's mapped value when it names an effort,
/// `high` otherwise.
fn map_reasoning_effort(
    model: &Model,
    level: Option<ModelThinkingLevel>,
) -> MistralReasoningEffort {
    let mapped = level
        .and_then(|level| {
            model
                .thinking_level_map
                .as_ref()
                .and_then(|map| map.get(&level))
                .cloned()
        })
        .and_then(|value| value)
        .and_then(|value| parse_effort(&value));
    mapped.unwrap_or(MistralReasoningEffort::High)
}

/// The cached-token probe across the four spellings plus the two flat
/// keys, clamped to `[0, promptTokens]`, upstream's
/// `getMistralCachedPromptTokens`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the token count clamps to non-negative and truncates like upstream's Number coercion"
)]
fn cached_prompt_tokens(usage: &Value, prompt_tokens: u64) -> u64 {
    let raw = usage
        .get("promptTokensDetails")
        .and_then(|details| details.get("cachedTokens"))
        .cloned()
        .or_else(|| {
            usage
                .get("prompt_tokens_details")
                .and_then(|details| details.get("cached_tokens"))
                .cloned()
        })
        .or_else(|| {
            usage
                .get("promptTokenDetails")
                .and_then(|details| details.get("cachedTokens"))
                .cloned()
        })
        .or_else(|| {
            usage
                .get("prompt_token_details")
                .and_then(|details| details.get("cached_tokens"))
                .cloned()
        })
        .or_else(|| usage.get("numCachedTokens").cloned())
        .or_else(|| usage.get("num_cached_tokens").cloned());
    let value = raw
        .and_then(|raw| raw.as_f64())
        .filter(|value| value.is_finite())
        .unwrap_or(0.0);
    let cached = value.max(0.0) as u64;
    prompt_tokens.min(cached)
}

fn parse_effort(value: &str) -> Option<MistralReasoningEffort> {
    match value {
        "none" => Some(MistralReasoningEffort::None),
        "high" => Some(MistralReasoningEffort::High),
        _ => None,
    }
}

/// The finish-reason mapping, upstream's `mapChatStopReason`.
fn map_chat_stop_reason(reason: Option<&str>) -> (StopReason, Option<String>) {
    match reason {
        None | Some("stop") => (StopReason::Stop, None),
        Some("length" | "model_length") => (StopReason::Length, None),
        Some("tool_calls") => (StopReason::ToolUse, None),
        Some("error") => (
            StopReason::Error,
            Some("Provider stopped with: error".to_owned()),
        ),
        Some(reason) => (
            StopReason::Error,
            Some(format!("Provider stopped with: {reason}")),
        ),
    }
}

fn wire_u64(value: &Value, key: &str) -> u64 {
    value
        .get(key)
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .max(0)
        .cast_unsigned()
}

/// The chat-completions URL, upstream's URL join: the base path keeps its
/// trailing slash and `v1/chat/completions` rides after it.
fn chat_completions_url(model: &Model) -> Result<String, String> {
    let base = url::Url::parse(&model.base_url).map_err(|error| format!("invalid URL: {error}"))?;
    let path = format!(
        "{trimmed}/v1/chat/completions",
        trimmed = base.path().trim_end_matches('/')
    );
    let mut url = base;
    url.set_path(&path);
    Ok(url.to_string())
}

/// Assemble the request headers, upstream's `buildMistralHeaders`: the pi
/// user agent, SSE accept, bearer auth, JSON content type, then the model
/// and caller overrides (a `None` value deletes), with the session-affinity
/// header riding only when caching is on and no override owns the name.
fn build_request_headers(
    model: &Model,
    api_key: &str,
    options: &MistralStreamOptions,
) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = vec![
        ("User-Agent".to_owned(), get_pi_user_agent()),
        ("accept".to_owned(), "text/event-stream".to_owned()),
        ("authorization".to_owned(), format!("Bearer {api_key}")),
        ("content-type".to_owned(), "application/json".to_owned()),
    ];
    if let Some(model_headers) = &model.headers {
        for (name, value) in model_headers {
            upsert_header(&mut headers, name, value);
        }
    }
    if let Some(options_headers) = &options.headers {
        // Mistral's override semantics: a `None` value deletes the header,
        // any value sets it.
        for (name, value) in options_headers {
            match value {
                Some(value) => upsert_header(&mut headers, name, value),
                None => headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name)),
            }
        }
    }
    if should_use_prompt_caching(options).is_some()
        && !has_header_override(model.headers.as_ref(), "x-affinity")
        && !options.headers.as_ref().is_some_and(|headers| {
            headers
                .keys()
                .any(|name| name.eq_ignore_ascii_case("x-affinity"))
        })
    {
        let session_id = options.session_id.clone().unwrap_or_default();
        upsert_header(&mut headers, "x-affinity", &session_id);
    }
    headers
}

fn has_header_override(
    headers: Option<&std::collections::BTreeMap<String, String>>,
    target: &str,
) -> bool {
    headers.is_some_and(|headers| headers.keys().any(|name| name.eq_ignore_ascii_case(target)))
}

/// Dispatch the stream request, upstream's `requestMistralStream`: one
/// POST, the response hook before the ok check, and the body-only-when-ok
/// rule.
async fn dispatch_stream_request(
    model: &Model,
    payload: &Value,
    options: &MistralStreamOptions,
    api_key: &str,
) -> Result<crate::http::client::HttpResponse, String> {
    let http_client = options.transport_options.client();
    let request = HttpRequest {
        method: HttpMethod::Post,
        url: chat_completions_url(model)?,
        headers: build_request_headers(model, api_key, options),
        body: Some(Bytes::from(to_wire_payload(payload).to_string())),
        timeout_ms: Some(options.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS)),
        signal: options.transport_options.signal(),
    };
    let response = http_client
        .execute(request)
        .await
        .map_err(mistral_error_from_http)?;

    call_response_hook(
        &options.transport_options,
        model,
        response.status,
        &response.headers,
    )
    .await;

    if !(200..300).contains(&response.status) {
        let body_text = read_body_text(response.body).await.unwrap_or_default();
        return Err(mistral_http_error_message(response.status, &body_text));
    }
    Ok(response)
}

/// The `MistralHttpError` message, upstream's status + body fold: the body
/// carries the truncation, a body-less failure carries the status fallback
/// (the seam has no `statusText`).
fn mistral_http_error_message(status: u16, body: &str) -> String {
    if body.trim().is_empty() {
        return format!("Mistral API error ({status}): Request failed with status {status}");
    }
    format!(
        "Mistral API error ({status}): {}",
        crate::utils::error_body::truncate_error_text(body.trim(), MAX_MISTRAL_ERROR_BODY_CHARS)
    )
}

fn mistral_error_from_http(error: HttpError) -> String {
    match error {
        HttpError::Aborted => crate::utils::abort::AbortError::MESSAGE.to_owned(),
        // The seam's display form must keep the word the retry and
        // timeout tests match on.
        HttpError::Timeout => "The operation was aborted due to timeout".to_owned(),
        HttpError::Transport(message) => message,
        HttpError::InvalidUrl(url) => format!("invalid URL: {url}"),
    }
}

/// The post-loop settlement, upstream's tail of the try block.
fn finish_stream(
    options: &MistralStreamOptions,
    output: &AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), String> {
    if options.transport_options.signal().is_cancelled() {
        return Err("Request was aborted".to_owned());
    }
    if output.stop_reason == StopReason::Pending {
        return Err("Mistral stream ended without a finish reason".to_owned());
    }
    if matches!(output.stop_reason, StopReason::Aborted | StopReason::Error) {
        return Err(output
            .error_message
            .clone()
            .unwrap_or_else(|| "An unknown error occurred".to_owned()));
    }

    events.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output.clone(),
    });
    Ok(())
}

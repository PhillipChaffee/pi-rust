//! The pi-messages wire API, ported from `packages/ai/src/api/pi-messages.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Streams pi's own message protocol directly to a backend: the request is a
//! single POST of `{ model, context, options }` to `<baseUrl>/messages`, the
//! response is an SSE stream of serialized assistant-message events plus a
//! terminal `done`/`error` event. This is the wire protocol spoken by the
//! Radius gateway, but any backend implementing it can be used, e.g. via a
//! models.json custom provider with `"api": "pi-messages"`.
//!
//! Porting restatements:
//! - The wire frames carry the same event vocabulary as pi's own protocol
//!   without the `partial` field; the converter holds one mutable
//!   accumulator and attaches it to every emitted event.
//! - The CRLF-normalizing incremental `TextDecoder` loop collapses into the
//!   shared [`SseStream`](crate::http::sse) decoder; the first-`data:`-line
//!   rule, the `[DONE]` skip, and the trailing-frame parse stay here.
//! - The seam carries no `statusText`, so the response-error message reads
//!   `"{status}: {message-or-body} ({code})"`; the diagnostic details carry
//!   everything else upstream kept.
//! - `streamSimple` cannot smuggle the pi-messages-only `debug` flag through
//!   the shared simple options (upstream casts the struct); `debug` rides on
//!   the adapter-local options only.

use std::collections::HashMap;

use bytes::Bytes;
use serde_json::{Value, json};

use crate::api::wire_common::{call_response_hook, sse_error_message, upsert_header};
use crate::http::client::{HttpMethod, HttpRequest};
use crate::types::AssistantMessageDiagnostic;
use crate::types::{
    AssistantBlock, AssistantMessage, AssistantMessageEvent, Context, Model, SimpleStreamOptions,
    StopReason, StreamOptions, TextContent, ThinkingContent, ToolCall,
};
use crate::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use crate::utils::json_parse::{parse_json_with_repair, parse_streaming_json};
use crate::utils::provider_env::get_provider_env_value;

/// The tool choice pi-messages accepts, upstream's
/// `PiMessagesOptions["toolChoice"]`: the shared choices plus `required` and
/// the OpenAI-style forced function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PiMessagesToolChoice {
    /// Let the provider decide.
    Auto,
    /// Never call tools.
    None,
    /// The model must call some tool, upstream's `"required"`.
    Required,
    /// Force a specific function, upstream's
    /// `{ type: "function", function: { name } }`.
    Function {
        /// The function to force.
        name: String,
    },
}

impl PiMessagesToolChoice {
    /// The wire shape, upstream's passthrough `toolChoice` value.
    #[must_use]
    pub fn to_wire(self) -> Value {
        match self {
            Self::Auto => json!("auto"),
            Self::None => json!("none"),
            Self::Required => json!("required"),
            Self::Function { name } => json!({
                "type": "function",
                "function": { "name": name },
            }),
        }
    }
}

/// Impact summary of a server-side message rewrite (e.g. a gateway policy),
/// upstream's `PiMessagesRewriteImpact`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiMessagesRewriteImpact {
    /// The policy that rewrote the context.
    pub policy_id: String,
    /// The policy version that rewrote it.
    pub policy_version: u64,
    /// Whether the rewrite changed anything.
    pub changed: bool,
    /// The token-count delta the rewrite produced.
    pub token_count_change: i64,
    /// The message-count delta the rewrite produced.
    pub message_count_change: i64,
    /// Whether the system prompt changed.
    pub system_prompt_changed: bool,
}

/// Serialized assistant-message event as sent by a pi-messages backend,
/// upstream's `PiMessagesEvent`.
///
/// The wire carries pi's own protocol without the `partial`. Terminal
/// `reason` values are pi stop-reason subsets; `"deferred"` never appears on
/// this wire.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum PiMessagesEvent {
    /// The stream opened.
    #[serde(rename = "start")]
    Start,
    /// A text block opened, wire `"text_start"`.
    #[serde(rename = "text_start", rename_all = "camelCase")]
    TextStart {
        /// The block's index in `content`.
        content_index: u64,
    },
    /// Text grew, wire `"text_delta"`.
    #[serde(rename = "text_delta", rename_all = "camelCase")]
    TextDelta {
        /// The block's index in `content`.
        content_index: u64,
        /// The appended text.
        delta: String,
    },
    /// A text block closed authoritatively, wire `"text_end"`.
    #[serde(rename = "text_end", rename_all = "camelCase")]
    TextEnd {
        /// The block's index in `content`.
        content_index: u64,
        /// The authoritative full text.
        content: String,
        /// The text signature, serialized as `contentSignature`.
        content_signature: Option<String>,
    },
    /// A thinking block opened, wire `"thinking_start"`.
    #[serde(rename = "thinking_start", rename_all = "camelCase")]
    ThinkingStart {
        /// The block's index in `content`.
        content_index: u64,
    },
    /// Thinking grew, wire `"thinking_delta"`.
    #[serde(rename = "thinking_delta", rename_all = "camelCase")]
    ThinkingDelta {
        /// The block's index in `content`.
        content_index: u64,
        /// The appended thinking text.
        delta: String,
    },
    /// A thinking block closed authoritatively, wire `"thinking_end"`.
    #[serde(rename = "thinking_end", rename_all = "camelCase")]
    ThinkingEnd {
        /// The block's index in `content`.
        content_index: u64,
        /// The authoritative full thinking text.
        content: String,
        /// The thinking signature, serialized as `contentSignature`.
        content_signature: Option<String>,
        /// Whether the thinking was redacted, serialized as `redacted`.
        redacted: Option<bool>,
    },
    /// A tool call opened, wire `"toolcall_start"`.
    #[serde(rename = "toolcall_start", rename_all = "camelCase")]
    ToolcallStart {
        /// The block's index in `content`.
        content_index: u64,
        /// The tool call id.
        id: String,
        /// The tool name, serialized as `toolName`.
        tool_name: String,
    },
    /// Tool-call arguments grew, wire `"toolcall_delta"`.
    #[serde(rename = "toolcall_delta", rename_all = "camelCase")]
    ToolcallDelta {
        /// The block's index in `content`.
        content_index: u64,
        /// The appended JSON update.
        delta: String,
    },
    /// A tool call closed authoritatively, wire `"toolcall_end"`.
    #[serde(rename = "toolcall_end", rename_all = "camelCase")]
    ToolcallEnd {
        /// The block's index in `content`.
        content_index: u64,
        /// The authoritative tool call.
        tool_call: ToolCall,
    },
    /// The stream finished successfully, with the wire-verbatim usage.
    #[serde(rename = "done", rename_all = "camelCase")]
    Done {
        /// Why the stream finished.
        reason: StopReason,
        /// The provider-reported usage.
        usage: crate::types::Usage,
        /// The provider response identifier, serialized as `responseId`.
        response_id: Option<String>,
        /// The provider-native effort level, serialized as
        /// `providerThinkingLevel`.
        provider_thinking_level: Option<String>,
        /// The rewrite impact when a gateway policy rewrote the messages.
        rewrite: Option<PiMessagesRewriteImpact>,
    },
    /// The stream failed or was aborted, with the failing message's usage.
    #[serde(rename = "error", rename_all = "camelCase")]
    Error {
        /// The failure kind.
        reason: StopReason,
        /// The provider-reported usage.
        usage: crate::types::Usage,
        /// The failure description, serialized as `errorMessage`.
        error_message: Option<String>,
        /// The provider response identifier, serialized as `responseId`.
        response_id: Option<String>,
        /// The provider-native effort level, serialized as
        /// `providerThinkingLevel`.
        provider_thinking_level: Option<String>,
        /// The rewrite impact when a gateway policy rewrote the messages.
        rewrite: Option<PiMessagesRewriteImpact>,
    },
}

/// The adapter-facing options, upstream's `PiMessagesOptions`.
#[derive(Clone, Debug, Default)]
pub struct PiMessagesStreamOptions {
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
    /// Accepted upstream-side and never used: this adapter issues one
    /// request with no timeout and no client-side retry.
    pub timeout_ms: Option<u64>,
    /// Accepted upstream-side and never used.
    pub max_retries: Option<u32>,
    /// Accepted upstream-side and never used.
    pub max_retry_delay_ms: Option<u64>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Arbitrary sampling parameters; upstream's pi-messages options shape
    /// has no samplingParams field, so the merge is dropped.
    pub sampling_params: Option<std::collections::BTreeMap<String, Value>>,
    /// Maximum output tokens.
    pub max_tokens: Option<u64>,
    /// Preferred transport; SSE is the only transport this adapter speaks.
    pub transport: Option<crate::types::Transport>,
    /// Prompt cache retention preference; the backend defaults apply when
    /// unset, upstream's `resolveCacheRetention`.
    pub cache_retention: Option<crate::types::CacheRetention>,
    /// Optional session identifier.
    pub session_id: Option<String>,
    /// Optional metadata to include in API requests.
    pub metadata: Option<std::collections::BTreeMap<String, Value>>,
    /// The pi thinking level for this request.
    pub reasoning: Option<crate::types::ThinkingLevel>,
    /// The tool choice, upstream's broader pi-messages union.
    pub tool_choice: Option<PiMessagesToolChoice>,
    /// Ask the backend for debug metadata (e.g. routing response headers).
    pub debug: bool,
}

impl From<StreamOptions> for PiMessagesStreamOptions {
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
            metadata: _,
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
            metadata: None,
            reasoning: None,
            tool_choice: None,
            debug: false,
        }
    }
}

/// The pi-messages streams, upstream's `piMessagesApi()`.
#[derive(Debug, Default)]
pub struct PiMessagesStreams;

crate::api::wire_common::forward_provider_streams!(PiMessagesStreams, PiMessagesStreamOptions);

/// Stream an assistant response, upstream's `stream` export: one POST of
/// `{ model, context, options }` and an SSE rehydration of the backend's
/// serialized assistant-message events.
///
/// Every failure mode — missing key, bad URL, HTTP error, parse error,
/// missing terminal event — settles as a terminal error event, upstream's
/// `createErrorEvent`.
#[must_use]
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&PiMessagesStreamOptions>,
) -> AssistantMessageEventStream {
    let events = assistant_message_event_stream();
    let forward = events.clone();
    let model = model.clone();
    let context = context.clone();
    let options = options.cloned();
    tokio::spawn(async move {
        let options = options.unwrap_or_default();
        run_stream_entry(&model, &context, &options, &forward).await;
    });
    events
}

/// Stream a simple assistant response, upstream's `streamSimple` export: the
/// shared options carry everything pi-messages reads, so the mapping is a
/// rename.
#[must_use]
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    let base = PiMessagesStreamOptions::from(crate::api::simple_options::build_base_options(
        model, context, options, None,
    ));
    let tool_choice = options
        .and_then(|options| options.tool_choice)
        .map(|choice| match choice {
            crate::types::ToolChoice::Auto => PiMessagesToolChoice::Auto,
            crate::types::ToolChoice::None => PiMessagesToolChoice::None,
        });
    stream(
        model,
        context,
        Some(&PiMessagesStreamOptions {
            reasoning: options.and_then(|options| options.reasoning),
            tool_choice,
            ..base
        }),
    )
}

/// A failed response's structured details, the port of
/// `PiMessagesResponseError`: the formatted message, the wire `error.code`
/// when a string, and the diagnostic details the failure carries.
struct PiMessagesResponseError {
    message: String,
    diagnostic_details: std::collections::BTreeMap<String, Value>,
}

/// The failure a stream carries before the error event is built.
struct StreamFailure {
    message: String,
    response_error: Option<PiMessagesResponseError>,
}

impl StreamFailure {
    const fn plain(message: String) -> Self {
        Self {
            message,
            response_error: None,
        }
    }
}

/// Build the request URL, upstream's trailing-slash strip plus the debug
/// query parameter.
fn messages_url(model: &Model, debug: bool) -> String {
    let mut url = format!("{}/messages", model.base_url.trim_end_matches('/'));
    if debug {
        url.push_str("?debug=1");
    }
    url
}

/// The cache-retention the request body carries: the explicit option wins,
/// else the legacy env opt-in maps `long`, else nothing (backend defaults).
fn resolve_cache_retention(
    cache_retention: Option<crate::types::CacheRetention>,
    env: Option<&crate::types::ProviderEnv>,
) -> Option<crate::types::CacheRetention> {
    if let Some(cache_retention) = cache_retention {
        return Some(cache_retention);
    }
    if get_provider_env_value("PI_CACHE_RETENTION", env).as_deref() == Some("long") {
        return Some(crate::types::CacheRetention::Long);
    }
    None
}

/// The `{ model, context, options }` body, upstream's payload: option keys
/// ride only when set, matching JSON.stringify's undefined drop.
fn build_payload(
    model: &Model,
    context: &Context,
    options: &PiMessagesStreamOptions,
) -> Result<Value, StreamFailure> {
    if options.transport_options.signal().is_cancelled() {
        return Err(StreamFailure::plain("Request aborted".to_owned()));
    }
    let mut options_object = serde_json::Map::new();
    if let Some(temperature) = options.temperature {
        options_object.insert("temperature".to_owned(), json!(temperature));
    }
    if let Some(max_tokens) = options.max_tokens {
        options_object.insert("maxTokens".to_owned(), json!(max_tokens));
    }
    if let Some(reasoning) = options.reasoning {
        options_object.insert("reasoning".to_owned(), json!(reasoning));
    }
    if let Some(cache_retention) =
        resolve_cache_retention(options.cache_retention, options.env.as_ref())
    {
        options_object.insert("cacheRetention".to_owned(), json!(cache_retention));
    }
    if let Some(session_id) = &options.session_id {
        options_object.insert("sessionId".to_owned(), json!(session_id));
    }
    if let Some(tool_choice) = &options.tool_choice {
        options_object.insert("toolChoice".to_owned(), tool_choice.clone().to_wire());
    }

    Ok(json!({
        "model": model.id,
        "context": context,
        "options": Value::Object(options_object),
    }))
}

async fn run_stream(
    model: &Model,
    context: &Context,
    options: &PiMessagesStreamOptions,
    events: &AssistantMessageEventStream,
) -> Result<(), StreamFailure> {
    let Some(api_key) = options.api_key.as_deref().filter(|key| !key.is_empty()) else {
        return Err(StreamFailure::plain(format!(
            "No API key provided for provider \"{}\"",
            model.provider.0
        )));
    };

    let url = messages_url(model, options.debug);
    let mut payload = build_payload(model, context, options)?;
    if let Some(hook) = &options.transport_options.on_payload {
        payload = hook
            .call(payload.clone(), model.clone())
            .await
            .unwrap_or(payload);
    }

    let response = dispatch_stream_request(model, &url, &payload, options, api_key).await?;

    // The wire speaks pi's own event protocol without the `partial`; the
    // converter rehydrates the shared accumulator and terminates on the
    // first done/error.
    let mut converter = EventConverter::new(model);
    let mut sse = crate::http::sse::SseStream::new(response.body);
    while let Some(sse_event) = sse
        .next()
        .await
        .map_err(|error| StreamFailure::plain(sse_error_message(error)))?
    {
        let Some(wire_event) = parse_wire_event(&sse_event.data).map_err(StreamFailure::plain)?
        else {
            continue;
        };
        let event = converter.convert(wire_event);
        let terminal = matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        );
        events.push(event);
        if terminal {
            return Ok(());
        }
    }

    Err(StreamFailure::plain(format!(
        "{} stream ended without a terminal event",
        model.provider.0
    )))
}

async fn dispatch_stream_request(
    model: &Model,
    url: &str,
    payload: &Value,
    options: &PiMessagesStreamOptions,
    api_key: &str,
) -> Result<crate::http::client::HttpResponse, StreamFailure> {
    let http_client = options.transport_options.client();
    let mut headers: Vec<(String, String)> = vec![
        ("authorization".to_owned(), format!("Bearer {api_key}")),
        ("accept".to_owned(), "text/event-stream".to_owned()),
        ("content-type".to_owned(), "application/json".to_owned()),
    ];
    // Caller headers override same-named defaults; a `None` value drops out
    // rather than deleting, upstream's providerHeadersToRecord spread.
    if let Some(record) =
        crate::utils::headers::provider_headers_to_record(options.headers.as_ref())
    {
        for (name, value) in record {
            upsert_header(&mut headers, &name, &value);
        }
    }
    let request = HttpRequest {
        method: HttpMethod::Post,
        url: url.to_owned(),
        headers,
        body: Some(Bytes::from(payload.to_string())),
        timeout_ms: options.timeout_ms,
        signal: options.transport_options.signal(),
    };
    let response = http_client
        .execute(request)
        .await
        .map_err(|error| match error {
            crate::http::client::HttpError::Aborted => {
                StreamFailure::plain("The operation was aborted".to_owned())
            }
            other => StreamFailure::plain(other.to_string()),
        })?;

    call_response_hook(
        &options.transport_options,
        model,
        response.status,
        &response.headers,
    )
    .await;

    if !(200..300).contains(&response.status) {
        let body = crate::http::client::read_body_text(response.body)
            .await
            .unwrap_or_default();
        let error = response_error(model, url, response.status, &body);
        return Err(StreamFailure {
            message: error.message.clone(),
            response_error: Some(error),
        });
    }
    Ok(response)
}

/// The wire's parsed error body, upstream's `parsePiMessagesErrorBody`: only
/// a JSON object whose `error` is a non-null, non-array object counts.
fn parse_error_body(body: &str) -> Option<Value> {
    let parsed: Value = serde_json::from_str(body).ok()?;
    parsed
        .get("error")
        .is_some_and(Value::is_object)
        .then_some(parsed)
}

/// The response failure, upstream's `createPiMessagesResponseError`: the
/// formatted message, the wire code, and the redacted diagnostic details.
fn response_error(model: &Model, url: &str, status: u16, body: &str) -> PiMessagesResponseError {
    let error_body = parse_error_body(body);
    let message = error_body
        .as_ref()
        .and_then(|body| body.pointer("/error/message"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let code = error_body
        .as_ref()
        .and_then(|body| body.pointer("/error/code"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let suffix = message.unwrap_or_else(|| body.to_owned());
    let code_suffix = code
        .as_deref()
        .map(|code| format!(" ({code})"))
        .unwrap_or_default();

    let mut diagnostic_details = std::collections::BTreeMap::new();
    diagnostic_details.insert("version".to_owned(), json!(1));
    diagnostic_details.insert("provider".to_owned(), json!(model.provider.0));
    diagnostic_details.insert("model".to_owned(), json!(model.id));
    diagnostic_details.insert("url".to_owned(), json!(url));
    diagnostic_details.insert("status".to_owned(), json!(status));
    if let Some(error_body) = &error_body {
        diagnostic_details.insert(
            "error".to_owned(),
            error_body.get("error").cloned().unwrap_or(Value::Null),
        );
    } else {
        diagnostic_details.insert("body".to_owned(), json!(truncate_diagnostic_string(body)));
    }
    diagnostic_details.insert(
        "timestampMs".to_owned(),
        json!(crate::auth::resolve::now_ms()),
    );

    PiMessagesResponseError {
        message: format!("{status}: {suffix}{code_suffix}"),
        diagnostic_details,
    }
}

/// The 8192-char diagnostic cap with the trailing ellipsis, upstream's
/// `truncateDiagnosticString`.
fn truncate_diagnostic_string(value: &str) -> String {
    const MAX_LENGTH: usize = 8192;
    if value.chars().count() > MAX_LENGTH {
        let kept: String = value.chars().take(MAX_LENGTH).collect();
        format!("{kept}…")
    } else {
        value.to_owned()
    }
}

/// Parse one SSE frame's data into the wire event, upstream's
/// `parsePiMessagesEvent`: the first `data:` line counts, `[DONE]` is
/// skipped, and frames without one are skipped.
fn parse_wire_event(data: &str) -> Result<Option<PiMessagesEvent>, String> {
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    let value = parse_json_with_repair(data).map_err(|error| error.to_string())?;
    serde_json::from_value(value).map_err(|error| error.to_string())
}

/// The event rehydrator, upstream's `createEventConverter`: one mutable
/// accumulator plus the raw tool-call JSON buffers.
struct EventConverter {
    partial: AssistantMessage,
    tool_json: HashMap<u64, String>,
}

/// Place a block at the wire's content index, the JS array-assignment
/// semantics upstream relies on: an index at the end extends, a lower index
/// overwrites, and a hole past the end carries empty text placeholders (a
/// shape only malformed streams can produce).
fn place_content(content: &mut Vec<AssistantBlock>, index: u64, block: AssistantBlock) {
    let Ok(index) = usize::try_from(index) else {
        return;
    };
    while content.len() < index {
        content.push(AssistantBlock::Text(TextContent {
            text: String::new(),
            text_signature: None,
        }));
    }
    if content.len() == index {
        content.push(block);
    } else {
        content[index] = block;
    }
}

/// The accumulator position a wire content index addresses, upstream's
/// JS array indexing.
fn content_position(index: u64) -> usize {
    usize::try_from(index).unwrap_or(usize::MAX)
}

impl EventConverter {
    fn new(model: &Model) -> Self {
        Self {
            partial: crate::api::wire_common::initial_output(model),
            tool_json: HashMap::new(),
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the event conversion mirrors upstream's createEventConverter in one match"
    )]
    fn convert(&mut self, event: PiMessagesEvent) -> AssistantMessageEvent {
        match event {
            PiMessagesEvent::Start => AssistantMessageEvent::Start {
                partial: self.partial.clone(),
            },
            PiMessagesEvent::TextStart { content_index } => {
                place_content(
                    &mut self.partial.content,
                    content_index,
                    AssistantBlock::Text(TextContent {
                        text: String::new(),
                        text_signature: None,
                    }),
                );
                AssistantMessageEvent::TextStart {
                    content_index,
                    partial: self.partial.clone(),
                }
            }
            PiMessagesEvent::TextDelta {
                content_index,
                delta,
            } => {
                if let Some(AssistantBlock::Text(block)) = self
                    .partial
                    .content
                    .get_mut(content_position(content_index))
                {
                    block.text.push_str(&delta);
                }
                AssistantMessageEvent::TextDelta {
                    content_index,
                    delta,
                    partial: self.partial.clone(),
                }
            }
            PiMessagesEvent::TextEnd {
                content_index,
                content,
                content_signature,
            } => {
                if let Some(AssistantBlock::Text(block)) = self
                    .partial
                    .content
                    .get_mut(content_position(content_index))
                {
                    content.clone_into(&mut block.text);
                    block.text_signature = content_signature;
                }
                AssistantMessageEvent::TextEnd {
                    content_index,
                    content,
                    partial: self.partial.clone(),
                }
            }
            PiMessagesEvent::ThinkingStart { content_index } => {
                place_content(
                    &mut self.partial.content,
                    content_index,
                    AssistantBlock::Thinking(ThinkingContent {
                        thinking: String::new(),
                        thinking_signature: None,
                        redacted: None,
                    }),
                );
                AssistantMessageEvent::ThinkingStart {
                    content_index,
                    partial: self.partial.clone(),
                }
            }
            PiMessagesEvent::ThinkingDelta {
                content_index,
                delta,
            } => {
                if let Some(AssistantBlock::Thinking(block)) = self
                    .partial
                    .content
                    .get_mut(content_position(content_index))
                {
                    block.thinking.push_str(&delta);
                }
                AssistantMessageEvent::ThinkingDelta {
                    content_index,
                    delta,
                    partial: self.partial.clone(),
                }
            }
            PiMessagesEvent::ThinkingEnd {
                content_index,
                content,
                content_signature,
                redacted,
            } => {
                if let Some(AssistantBlock::Thinking(block)) = self
                    .partial
                    .content
                    .get_mut(content_position(content_index))
                {
                    content.clone_into(&mut block.thinking);
                    block.thinking_signature = content_signature;
                    block.redacted = redacted;
                }
                AssistantMessageEvent::ThinkingEnd {
                    content_index,
                    content,
                    partial: self.partial.clone(),
                }
            }
            PiMessagesEvent::ToolcallStart {
                content_index,
                id,
                tool_name,
            } => {
                place_content(
                    &mut self.partial.content,
                    content_index,
                    AssistantBlock::ToolCall(ToolCall {
                        id,
                        name: tool_name,
                        arguments: serde_json::Map::new(),
                        thought_signature: None,
                        namespace: None,
                    }),
                );
                self.tool_json.insert(content_index, String::new());
                AssistantMessageEvent::ToolcallStart {
                    content_index,
                    partial: self.partial.clone(),
                }
            }
            PiMessagesEvent::ToolcallDelta {
                content_index,
                delta,
            } => {
                let json = self.tool_json.entry(content_index).or_default();
                json.push_str(&delta);
                let parsed = parse_streaming_json(Some(json.as_str()))
                    .as_object()
                    .cloned()
                    .unwrap_or_default();
                if let Some(AssistantBlock::ToolCall(block)) = self
                    .partial
                    .content
                    .get_mut(content_position(content_index))
                {
                    block.arguments = parsed;
                }
                AssistantMessageEvent::ToolcallDelta {
                    content_index,
                    delta,
                    partial: self.partial.clone(),
                }
            }
            PiMessagesEvent::ToolcallEnd {
                content_index,
                tool_call,
            } => {
                // The authoritative wire tool call merges field-wise over the
                // partial: the wire's `thoughtSignature`/`namespace` win when
                // present, absent fields keep the accumulated values. A close
                // without an open is a protocol violation; upstream's
                // Object.assign on the missing index crashes into the same
                // catch that settles the stream with an error.
                let merged = match self
                    .partial
                    .content
                    .get_mut(content_position(content_index))
                {
                    Some(AssistantBlock::ToolCall(block)) => {
                        tool_call.id.clone_into(&mut block.id);
                        tool_call.name.clone_into(&mut block.name);
                        tool_call.arguments.clone_into(&mut block.arguments);
                        if tool_call.thought_signature.is_some() {
                            tool_call
                                .thought_signature
                                .clone_into(&mut block.thought_signature);
                        }
                        if tool_call.namespace.is_some() {
                            block.namespace = tool_call.namespace;
                        }
                        Some(block.clone())
                    }
                    _ => None,
                };
                if let Some(tool_call) = merged {
                    self.tool_json.remove(&content_index);
                    AssistantMessageEvent::ToolcallEnd {
                        content_index,
                        tool_call,
                        partial: self.partial.clone(),
                    }
                } else {
                    let mut failing = self.partial.clone();
                    failing.stop_reason = StopReason::Error;
                    failing.error_message = Some("Invalid pi-messages event sequence".to_owned());
                    AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error: failing,
                    }
                }
            }
            PiMessagesEvent::Done {
                reason,
                usage,
                response_id,
                provider_thinking_level,
                rewrite,
            } => {
                self.partial.stop_reason = reason;
                self.partial.usage = usage;
                self.partial.response_id = response_id;
                if let Some(provider_thinking_level) = provider_thinking_level {
                    self.partial.provider_thinking_level = Some(provider_thinking_level);
                }
                self.append_rewrite_diagnostic(rewrite);
                AssistantMessageEvent::Done {
                    reason,
                    message: self.partial.clone(),
                }
            }
            PiMessagesEvent::Error {
                reason,
                usage,
                error_message,
                response_id,
                provider_thinking_level,
                rewrite,
            } => {
                self.partial.stop_reason = reason;
                self.partial.usage = usage;
                self.partial.error_message = error_message;
                self.partial.response_id = response_id;
                if let Some(provider_thinking_level) = provider_thinking_level {
                    self.partial.provider_thinking_level = Some(provider_thinking_level);
                }
                self.append_rewrite_diagnostic(rewrite);
                AssistantMessageEvent::Error {
                    reason,
                    error: self.partial.clone(),
                }
            }
        }
    }

    fn append_rewrite_diagnostic(&mut self, rewrite: Option<PiMessagesRewriteImpact>) {
        let Some(rewrite) = rewrite else {
            return;
        };
        let mut details = std::collections::BTreeMap::new();
        details.insert("policyId".to_owned(), json!(rewrite.policy_id));
        details.insert("policyVersion".to_owned(), json!(rewrite.policy_version));
        details.insert("changed".to_owned(), json!(rewrite.changed));
        details.insert(
            "tokenCountChange".to_owned(),
            json!(rewrite.token_count_change),
        );
        details.insert(
            "messageCountChange".to_owned(),
            json!(rewrite.message_count_change),
        );
        details.insert(
            "systemPromptChanged".to_owned(),
            json!(rewrite.system_prompt_changed),
        );
        let diagnostic = AssistantMessageDiagnostic {
            kind: "pi_messages_rewrite".to_owned(),
            timestamp: crate::auth::resolve::now_ms(),
            error: None,
            details: Some(details),
        };
        crate::utils::diagnostics::append_assistant_message_diagnostic(
            &mut self.partial,
            diagnostic,
        );
    }
}

async fn run_stream_entry(
    model: &Model,
    context: &Context,
    options: &PiMessagesStreamOptions,
    forward: &AssistantMessageEventStream,
) {
    match run_stream(model, context, options, forward).await {
        Ok(()) => {
            // The terminal wire event already settled the final result.
            forward.end(None);
        }
        Err(failure) => {
            let aborted = options.transport_options.signal().is_cancelled();
            let reason = if aborted {
                StopReason::Aborted
            } else {
                StopReason::Error
            };
            let mut failing = AssistantMessage {
                content: Vec::new(),
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_model: None,
                response_id: None,
                provider_thinking_level: None,
                diagnostics: None,
                usage: crate::types::Usage::default(),
                stop_reason: reason,
                deferred: None,
                error_message: Some(failure.message),
                raw_stop_reason: None,
                end_turn: None,
                timestamp: crate::auth::resolve::now_ms(),
            };
            // Only genuine response failures carry the diagnostic; an
            // aborted turn reports none.
            if !aborted && let Some(response_error) = failure.response_error {
                failing.diagnostics = Some(vec![AssistantMessageDiagnostic {
                    kind: "pi_messages_response_failure".to_owned(),
                    timestamp: crate::auth::resolve::now_ms(),
                    error: Some(crate::types::DiagnosticErrorInfo {
                        name: Some("PiMessagesResponseError".to_owned()),
                        message: response_error.message.clone(),
                        stack: None,
                        code: None,
                    }),
                    details: Some(response_error.diagnostic_details),
                }]);
            }
            forward.push(AssistantMessageEvent::Error {
                reason,
                error: failing.clone(),
            });
            forward.end(Some(&failing));
        }
    }
}

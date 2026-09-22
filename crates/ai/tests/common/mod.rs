//! Shared test fixtures for the utils belt suites, mirroring the shapes
//! upstream's `fauxAssistantMessage` and message helpers feed the same
//! code paths.

#![expect(
    dead_code,
    reason = "shared fixtures; each test binary uses the subset it needs"
)]
#![expect(
    unreachable_pub,
    reason = "the fixture module is compiled into every integration test binary as a private module"
)]
#![expect(
    clippy::expect_used,
    reason = "the block helper panics on a runtime build failure by design; tests pin outcomes"
)]

pub mod auth_fixtures;
pub mod auth_guards;
pub mod auth_interaction;
pub mod auth_json;
pub mod oauth_fixtures;
pub mod paused_clock;
pub mod radius_fixtures;
pub mod seam_forms;

use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, Message, ProviderId, StopReason, TextContent,
    ThinkingContent, ToolResultBlock, ToolResultMessage, Usage, UsageCost, UserContent,
    UserMessage,
};

/// A provider-shaped usage block with the given total.
#[must_use]
pub fn usage(total_tokens: u64) -> Usage {
    Usage {
        input: total_tokens,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens,
        cost: UsageCost::default(),
    }
}

/// An assistant message carrying one text block, the faux provider's
/// success shape.
#[must_use]
pub fn assistant_message(text: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: Api::from("test-api"),
        provider: ProviderId::from("test-provider"),
        model: String::from("test-model"),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

/// An assistant message with overridden stop reason and error text.
#[must_use]
pub fn assistant_message_with(
    text: &str,
    stop_reason: StopReason,
    error_message: Option<String>,
) -> AssistantMessage {
    let mut message = assistant_message(text);
    message.stop_reason = stop_reason;
    message.error_message = error_message;
    message
}

/// A minimal assistant message with no content, for error/usage shapes.
#[must_use]
pub fn bare_assistant_message() -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: Api::from("test-api"),
        provider: ProviderId::from("test-provider"),
        model: String::from("test-model"),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

/// A user message with a plain-string content.
#[must_use]
pub fn user_message(text: &str, timestamp: i64) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp,
    })
}

/// A text block for content comparisons.
#[must_use]
pub fn text_block(text: &str) -> TextContent {
    TextContent {
        text: text.to_owned(),
        text_signature: None,
    }
}

/// An assistant message with only stop reason set, for retry shapes that
/// check the reason alone.
#[must_use]
pub fn aborted_message() -> AssistantMessage {
    assistant_message_with("", StopReason::Aborted, None)
}

// --- The Models-runtime fixtures the #28 runtime suites share ---

use std::future::Future;
use std::sync::{Arc, Mutex};

use pi_ai::auth::types::{ApiKeyAuth, ApiKeyAuthInput, ProviderAuth};
use pi_ai::types::{
    BoxedFuture, Context, DeferredCancelOptions, DeferredFetchOptions, DeferredHandle, Model,
    ProviderStreams, SimpleStreamOptions, StreamOptions,
};

/// Run a future to completion on a fresh current-thread runtime, the seam
/// sync tests use to drive the async surface.
pub fn block<F: Future>(future: F) -> F::Output {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    runtime.block_on(future)
}

/// The minimal chat model every runtime suite streams against.
#[must_use]
pub fn fixture_model() -> Model {
    Model {
        id: "m".to_owned(),
        name: "m".to_owned(),
        api: Api::from("test-api"),
        provider: ProviderId::from("p"),
        base_url: "https://example.test/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 1000,
        max_tokens: 100,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The settled assistant message a fixture stream ends with.
#[must_use]
pub fn message_fixture(model: &Model, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

/// Ambient api-key auth whose resolve always succeeds, the configured stub.
#[must_use]
pub fn ambient_auth() -> ProviderAuth {
    ProviderAuth {
        api_key: Some(ApiKeyAuth {
            name: "Ambient".to_owned(),
            login: None,
            check: None,
            resolve: Arc::new(|_input: ApiKeyAuthInput| {
                Box::pin(async move { Ok(Some(pi_ai::auth::types::AuthResult::default())) })
            }),
        }),
        oauth: None,
    }
}

/// The empty context every stream request carries.
#[must_use]
pub fn context() -> Context {
    Context::default()
}

/// The deferred handle the deferred dispatch tests route through.
#[must_use]
pub fn deferred_handle() -> DeferredHandle {
    DeferredHandle {
        provider: "p".to_owned(),
        model_id: "m".to_owned(),
        api: "test-api".to_owned(),
        id: "id".to_owned(),
        expires_at: None,
        poll_after_ms: None,
        data: None,
    }
}

/// A stream that settles immediately with the fixture message.
#[must_use]
pub fn end_with(model: &Model) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
    let stream = pi_ai::utils::event_stream::assistant_message_event_stream();
    stream.end(Some(&message_fixture(model, StopReason::Stop)));
    stream
}

/// The deferred-capable streams fixture: streams end with the fixture
/// message, fetches carry the handle through, and both fetch and cancel are
/// counted so suites can assert the dispatch reached the provider.
pub struct DeferredStreams {
    /// How many fetches reached the fixture.
    pub fetches: Arc<Mutex<u64>>,
    /// How many cancels reached the fixture.
    pub cancels: Arc<Mutex<u64>>,
}

impl DeferredStreams {
    /// The fixture with its two counters, the shape suites assert on.
    #[must_use]
    pub fn new() -> (Self, Arc<Mutex<u64>>, Arc<Mutex<u64>>) {
        let fetches = Arc::new(Mutex::new(0));
        let cancels = Arc::new(Mutex::new(0));
        (
            Self {
                fetches: Arc::clone(&fetches),
                cancels: Arc::clone(&cancels),
            },
            fetches,
            cancels,
        )
    }

    /// The fixture when only the behavior matters, not the counters.
    #[must_use]
    pub fn uncounted() -> Self {
        Self {
            fetches: Arc::new(Mutex::new(0)),
            cancels: Arc::new(Mutex::new(0)),
        }
    }
}

impl ProviderStreams for DeferredStreams {
    fn stream(
        &self,
        model: &Model,
        _context: &Context,
        _options: Option<&StreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        end_with(model)
    }

    fn stream_simple(
        &self,
        model: &Model,
        _context: &Context,
        _options: Option<&SimpleStreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        end_with(model)
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        _options: Option<&DeferredFetchOptions>,
    ) -> Option<pi_ai::utils::event_stream::AssistantMessageEventStream> {
        *self
            .fetches
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
        let stream = pi_ai::utils::event_stream::assistant_message_event_stream();
        let mut message = message_fixture(model, StopReason::Stop);
        message.deferred = Some(handle.clone());
        stream.end(Some(&message));
        Some(stream)
    }

    fn cancel_deferred<'a>(
        &'a self,
        _model: &'a Model,
        _handle: &'a DeferredHandle,
        _options: Option<&'a DeferredCancelOptions>,
    ) -> BoxedFuture<'a, Result<(), pi_ai::utils::provider_retry::ProviderRequestError>> {
        *self
            .cancels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
        Box::pin(async { Ok(()) })
    }

    fn supports_fetch_deferred(&self) -> bool {
        true
    }

    fn supports_cancel_deferred(&self) -> bool {
        true
    }
}

/// The minimal image model every image suite streams against.
#[must_use]
pub fn image_model(provider: &str, id: &str) -> pi_ai::types::ImagesModel {
    pi_ai::types::ImagesModel {
        id: id.to_owned(),
        name: id.to_owned(),
        api: pi_ai::types::ImagesApi::from("test-images"),
        provider: pi_ai::types::ImagesProviderId::from(provider),
        base_url: "https://example.test/v1".to_owned(),
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        output: vec![pi_ai::types::Modality::Image],
        cost: pi_ai::types::ModelCost {
            rates: pi_ai::types::ModelCostRates::default(),
            tiers: None,
        },
        sampling_params: None,
        headers: None,
    }
}

/// The settled image result a fixture image api returns.
#[must_use]
pub fn ok_images_result(model: &pi_ai::types::ImagesModel) -> pi_ai::types::AssistantImages {
    pi_ai::types::AssistantImages {
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        output: vec![pi_ai::types::ImagesBlock::Image(
            pi_ai::types::ImageContent {
                data: "aGk=".to_owned(),
                mime_type: "image/png".to_owned(),
            },
        )],
        response_id: None,
        usage: None,
        stop_reason: pi_ai::types::ImagesStopReason::Stop,
        error_message: None,
        timestamp: pi_ai::auth::resolve::now_ms(),
    }
}

/// The image-generation context every image suite sends.
#[must_use]
pub fn images_context() -> pi_ai::types::ImagesContext {
    pi_ai::types::ImagesContext {
        input: vec![pi_ai::types::ImagesBlock::Text(TextContent {
            text: "a red circle".to_owned(),
            text_signature: None,
        })],
    }
}

// --- The Anthropic Messages fixtures the #30 suites share ---

use serde_json::json;

use pi_ai::http::MockHttpClient;
use pi_ai::types::{OnPayload, Tool, ToolCall, TransportOptions};

/// The plain tool the deferred-reference suites keep immediate.
#[must_use]
pub fn lookup_tool() -> Tool {
    Tool {
        name: "lookup".to_owned(),
        description: "Look up.".to_owned(),
        parameters: json!({ "type": "object", "properties": {} }),
        constrained_sampling: None,
    }
}

/// The tool the deferred-reference suites load through the result marker.
#[must_use]
pub fn late_tool() -> Tool {
    Tool {
        name: "late_tool".to_owned(),
        description: "Loaded late.".to_owned(),
        parameters: json!({ "type": "object", "properties": {} }),
        constrained_sampling: None,
    }
}

/// The todo tool the OAuth CC-casing suites rename.
#[must_use]
pub fn todo_tool() -> Tool {
    Tool {
        name: "todowrite".to_owned(),
        description: "Write a todo.".to_owned(),
        parameters: json!({ "type": "object", "properties": {} }),
        constrained_sampling: None,
    }
}

/// The keyed stream options the mock suites send: the mock as the transport
/// and a static test key.
#[must_use]
pub fn keyed_anthropic_options(
    mock: &MockHttpClient,
) -> pi_ai::api::anthropic_messages::AnthropicStreamOptions {
    pi_ai::api::anthropic_messages::AnthropicStreamOptions {
        transport_options: mock_transport(mock),
        api_key: Some("test-key".to_owned()),
        ..pi_ai::api::anthropic_messages::AnthropicStreamOptions::default()
    }
}

/// The keyed simple-stream options the simple-entry mock suites send.
#[must_use]
pub fn keyed_simple_options(mock: &MockHttpClient) -> SimpleStreamOptions {
    SimpleStreamOptions {
        transport_options: mock_transport(mock),
        api_key: Some("test-key".to_owned()),
        ..SimpleStreamOptions::default()
    }
}

/// A successful SSE run: the `message_start` opener, the given block events, a
/// final stop-reason delta with the given usage, and `message_stop`.
#[must_use]
pub fn anthropic_done_run(
    id: &str,
    block_events: Vec<(&'static str, String)>,
    stop_reason: &str,
    usage: impl serde::Serialize,
) -> Vec<(&'static str, String)> {
    let mut events = vec![message_start_event(
        id,
        json!({ "input_tokens": 1, "output_tokens": 0 }),
    )];
    events.extend(block_events);
    events.push(message_delta_event(
        json!({ "stop_reason": stop_reason }),
        Some(usage),
    ));
    events.push(message_stop_event());
    events
}

/// A hand-built Anthropic Messages model, the shape the upstream auth-token
/// and conformance suites construct.
#[must_use]
pub fn anthropic_model() -> Model {
    Model {
        id: "claude-test".to_owned(),
        name: "Claude Test".to_owned(),
        api: Api::from("anthropic-messages"),
        provider: ProviderId::from("anthropic"),
        base_url: "https://api.anthropic.com".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 100_000,
        max_tokens: 4096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// Resolve one generated catalog model, upstream's `getModel(provider, id)`.
#[must_use]
pub fn builtin_model(provider: &str, id: &str) -> Model {
    pi_ai::providers::all::builtin_models_of(provider)
        .into_iter()
        .find(|model| model.id == id)
        .expect("the catalog carries the model a suite names")
}

/// A user message with a plain-string content and a fixed timestamp.
#[must_use]
pub fn user_message_at(text: &str, timestamp: i64) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp,
    })
}

/// A user message with the current wall clock, like upstream test contexts.
#[must_use]
pub fn user_message_now(text: &str) -> Message {
    user_message_at(text, pi_ai::auth::resolve::now_ms())
}

/// An assistant message with the given thinking block, the replay fixtures
/// the signature-compat suites build.
#[must_use]
pub fn thinking_assistant_message(
    api: &str,
    provider: &str,
    model: &str,
    thinking: &str,
    signature: &str,
) -> AssistantMessage {
    assistant_message_with_content(
        api,
        provider,
        model,
        vec![AssistantBlock::Thinking(ThinkingContent {
            thinking: thinking.to_owned(),
            thinking_signature: Some(signature.to_owned()),
            redacted: None,
        })],
    )
}

/// An assistant message carrying one tool call, the tool-use turn the
/// migration suites replay.
#[must_use]
pub fn tool_call_assistant_message(
    api: &str,
    provider: &str,
    model: &str,
    tool_call: ToolCall,
) -> AssistantMessage {
    assistant_message_with_content(
        api,
        provider,
        model,
        vec![AssistantBlock::ToolCall(tool_call)],
    )
}

/// The anthropic messages context the payload-capture suites send.
#[must_use]
pub fn anthropic_context() -> Context {
    Context {
        system_prompt: Some("System prompt.".to_owned()),
        messages: vec![user_message_now("Hello")],
        tools: None,
    }
}

/// An assistant message with the given content, the history-turn fixture the
/// request-conversion suites replay.
#[must_use]
pub fn assistant_message_with_content(
    api: &str,
    provider: &str,
    model: &str,
    content: Vec<AssistantBlock>,
) -> AssistantMessage {
    AssistantMessage {
        content,
        api: Api::from(api),
        provider: ProviderId::from(provider),
        model: model.to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

/// A tool-result message with the given content, the history-turn fixture.
#[must_use]
pub fn tool_result_message(
    tool_call_id: &str,
    content: Vec<ToolResultBlock>,
    added_tool_names: Option<Vec<String>>,
) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: tool_call_id.to_owned(),
        tool_name: "tool".to_owned(),
        content,
        details: None,
        usage: None,
        added_tool_names,
        is_error: false,
        timestamp: 1,
    })
}

/// The first recorded request's decoded JSON body, the shape the
/// request-shape suites assert on.
#[must_use]
pub fn recorded_body(mock: &MockHttpClient) -> serde_json::Value {
    serde_json::from_slice(
        mock.recorded()[0]
            .body
            .as_ref()
            .expect("the mock request carries a body"),
    )
    .expect("the request body is JSON")
}

/// The value of the first recorded request's named header, case-insensitive.
#[must_use]
pub fn recorded_header(mock: &MockHttpClient, name: &str) -> Option<String> {
    mock.recorded()[0]
        .headers
        .iter()
        .find(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
}

/// Mount the seam mock the stream suites inject; the route answers
/// anthropic message requests with the given SSE body.
#[must_use]
pub fn anthropic_mock(events: &[(&str, String)]) -> MockHttpClient {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, events);
    mock
}

/// The (event, data) pairs `sse_response` takes from the string-data event
/// fixtures the anthropic suites build.
#[must_use]
pub fn sse_pairs<'a>(events: &'a [(&'a str, String)]) -> Vec<(&'a str, &'a str)> {
    events
        .iter()
        .map(|(event, data)| (*event, data.as_str()))
        .collect()
}

/// Mount onto an existing mock, for suites that assert the recorded request.
pub fn anthropic_mock_with(mock: &MockHttpClient, events: &[(&str, String)]) {
    let pairs: Vec<(&str, &str)> = events
        .iter()
        .map(|(event, data)| (*event, data.as_str()))
        .collect();
    mock.on(|request| request.url.contains("/v1/messages"))
        .respond(pi_ai::http::sse_response(200, &pairs));
}

/// The barest successful stream, for request-shape capture suites: auth
/// resolution threads its headers through and the stream settles.
#[must_use]
pub fn minimal_anthropic_done() -> Vec<(&'static str, String)> {
    vec![
        message_start_event("msg_test", json!({ "input_tokens": 1, "output_tokens": 0 })),
        message_delta_event(
            json!({ "stop_reason": "end_turn" }),
            Some(json!({ "input_tokens": 1, "output_tokens": 1 })),
        ),
        message_stop_event(),
    ]
}

/// The `message_start` frame: the stream opens with the given response id
/// and usage block.
#[must_use]
pub fn message_start_event(id: &str, usage: impl serde::Serialize) -> (&'static str, String) {
    (
        "message_start",
        json!({
            "type": "message_start",
            "message": { "id": id, "usage": usage },
        })
        .to_string(),
    )
}

/// The `message_start` frame that names the serving model, the shape the
/// fallback-repricing suites stream.
#[must_use]
pub fn message_start_from_serving_model(
    serving_model: &str,
    id: &str,
    usage: impl serde::Serialize,
) -> (&'static str, String) {
    (
        "message_start",
        json!({
            "type": "message_start",
            "message": { "id": id, "model": serving_model, "usage": usage },
        })
        .to_string(),
    )
}

/// The `message_delta` frame: a stop-reason delta with an optional partial
/// usage update.
#[must_use]
pub fn message_delta_event(
    delta: impl serde::Serialize,
    usage: Option<impl serde::Serialize>,
) -> (&'static str, String) {
    let mut body = json!({ "type": "message_delta", "delta": delta });
    if let Some(usage) = usage {
        body["usage"] = json!(usage);
    }
    ("message_delta", body.to_string())
}

/// The `message_stop` frame.
#[must_use]
pub fn message_stop_event() -> (&'static str, String) {
    (
        "message_stop",
        json!({ "type": "message_stop" }).to_string(),
    )
}

/// The `content_block_start` frame: the block opens at the given wire index.
#[must_use]
pub fn block_start_event(
    index: u64,
    content_block: impl serde::Serialize,
) -> (&'static str, String) {
    (
        "content_block_start",
        json!({
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        })
        .to_string(),
    )
}

/// The `content_block_delta` frame: the delta lands at the given wire index.
#[must_use]
pub fn block_delta_event(index: u64, delta: impl serde::Serialize) -> (&'static str, String) {
    (
        "content_block_delta",
        json!({ "type": "content_block_delta", "index": index, "delta": delta }).to_string(),
    )
}

/// The `content_block_stop` frame: the block at the given wire index closes.
#[must_use]
pub fn block_stop_event(index: u64) -> (&'static str, String) {
    (
        "content_block_stop",
        json!({ "type": "content_block_stop", "index": index }).to_string(),
    )
}

/// The captured on-payload hook: records the payload an adapter is about to
/// send and keeps it unchanged, upstream's throwing `PayloadCaptured` hook
/// minus the throw — the empty mock fails the request instead.
#[must_use]
pub fn payload_capture() -> (OnPayload, CapturedPayload) {
    let captured = Arc::new(Mutex::new(None));
    let slot = Arc::clone(&captured);
    (
        OnPayload::new(move |payload, _model| {
            *slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(payload);
            Box::pin(async { None })
        }),
        captured,
    )
}

/// Wait for the captured payload, the shape upstream's capture helpers
/// assert on.
#[must_use]
pub fn captured_payload(captured: &CapturedPayload) -> serde_json::Value {
    captured
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("expected payload capture before request failure")
}

/// The payload slot a capture hook fills, shared between the hook and the
/// asserting test.
pub type CapturedPayload = Arc<Mutex<Option<serde_json::Value>>>;

/// The wire options every mock-stream test injects: the mock as the
/// transport and no credential resolution.
#[must_use]
pub fn mock_transport(mock: &MockHttpClient) -> TransportOptions {
    TransportOptions {
        http_client: Some(Arc::new(mock.clone())),
        ..TransportOptions::default()
    }
}

// --- The OpenAI Completions fixtures the openai-family suites share ---

use pi_ai::api::openai_completions::OpenAiCompletionsOptions;

/// The plain openai-completions model the retry and raw-stop-reason suites
/// stream against, the shape upstream's shared retry-suite model carries.
#[must_use]
pub fn openai_completions_model() -> Model {
    Model {
        id: "test-model".to_owned(),
        name: "Test Model".to_owned(),
        api: Api::from("openai-completions"),
        provider: ProviderId::from("opencode-go"),
        base_url: "https://opencode.ai/zen/go/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 1000,
        max_tokens: 100,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The catalog model retargeted at the openai-completions wire, the
/// `{ ...getModel(...), api: "openai-completions" }` spread the upstream
/// suites run for OpenAI and OpenRouter catalog entries.
#[must_use]
pub fn openai_catalog_model(provider: &str, id: &str) -> Model {
    Model {
        api: Api::from("openai-completions"),
        ..builtin_model(provider, id)
    }
}

/// A chunk stream's SSE body: each chunk one `data:` frame, closed by the
/// `data: [DONE]` sentinel, the shape `data: [DONE]` ends the shared decoder
/// on.
#[must_use]
pub fn openai_sse_body(chunks: &[serde_json::Value]) -> String {
    use std::fmt::Write as _;
    let mut body = String::new();
    for chunk in chunks {
        let _ = write!(body, "data: {chunk}\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

/// The SSE response an OpenAI completions stream returns over the given
/// chunks, the mock body the chunk-suite mounts.
#[must_use]
pub fn openai_sse_response(chunks: &[serde_json::Value]) -> pi_ai::http::MockResponse {
    pi_ai::http::MockResponse::status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(openai_sse_body(chunks))
}

/// The standard final chunk the capture suites stream: an empty delta with
/// `stop` plus the zeroed usage details block.
#[must_use]
pub fn openai_done_chunk() -> serde_json::Value {
    json!({
        "choices": [{ "delta": {}, "finish_reason": "stop" }],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "prompt_tokens_details": { "cached_tokens": 0 },
            "completion_tokens_details": { "reasoning_tokens": 0 },
        },
    })
}

/// Mount the seam mock the openai-completions suites inject: the route
/// answers `/chat/completions` with the chunks' SSE body.
#[must_use]
pub fn openai_mock(chunks: &[serde_json::Value]) -> MockHttpClient {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, chunks);
    mock
}

/// Mount onto an existing mock, for suites that assert the recorded request.
pub fn openai_mock_with(mock: &MockHttpClient, chunks: &[serde_json::Value]) {
    mock.on(|request| request.url.contains("/chat/completions"))
        .respond(openai_sse_response(chunks));
}

/// The keyed openai-completions options the mock suites send: the mock as the
/// transport and a static test key.
#[must_use]
pub fn keyed_openai_options(mock: &MockHttpClient) -> OpenAiCompletionsOptions {
    OpenAiCompletionsOptions {
        transport_options: mock_transport(mock),
        api_key: Some("test-key".to_owned()),
        ..OpenAiCompletionsOptions::default()
    }
}

/// The recorded body of the request at `index`, the shape the multi-request
/// suites (retry sequences, second-run replays) assert on.
#[must_use]
pub fn recorded_body_at(mock: &MockHttpClient, index: usize) -> serde_json::Value {
    serde_json::from_slice(
        mock.recorded()[index]
            .body
            .as_ref()
            .expect("the mock request carries a body"),
    )
    .expect("the request body is JSON")
}

/// Drain the event stream and settle its final message, upstream's
/// drain-then-`result()` shape.
///
/// The result future joins the drain so its waiter registers before the
/// completing event settles: the wire APIs call `end(None)` after the `done`
/// event, which clears an already-settled result, so a `result()` awaited
/// only after the drain would wait forever.
pub async fn drain_and_settle(
    stream: &pi_ai::utils::event_stream::AssistantMessageEventStream,
) -> AssistantMessage {
    let ((), message) = tokio::join!(
        async { while stream.next().await.is_some() {} },
        stream.result()
    );
    message
}

/// The catalog probe the live E2E suites open with: the named model rides
/// the named wire API.
pub fn assert_catalog_api(provider: &str, id: &str, api: &str) {
    let model = builtin_model(provider, id);
    assert_eq!(model.api, Api::from(api), "{provider}/{id}");
}

/// The settled message's text blocks joined, the live-probe reply reader.
#[must_use]
pub fn live_response_text(message: &AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

/// Assert a live probe's settled reply carried the expected text, upstream's
/// `responseText` containment check.
pub fn assert_live_text_reply(message: &AssistantMessage, needle: &str) {
    let text = live_response_text(message);
    assert!(text.contains(needle), "got: {text}");
}

/// Assert a stream settled as a credential-setup failure that dispatched
/// nothing: the missing-key error message and an untouched mock, the shape
/// the wire-API credential suites pin.
pub fn assert_setup_error_without_dispatch(result: &AssistantMessage, mock: &MockHttpClient) {
    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.as_deref().expect("the setup error");
    assert!(
        message.contains("No API key for provider"),
        "got: {message}"
    );
    assert_eq!(mock.request_count(), 0);
}

/// The recorded request's named header values, duplicates and case order
/// preserved, the multi-value header assertions' reader.
#[must_use]
pub fn recorded_header_values(mock: &MockHttpClient, name: &str) -> Vec<String> {
    mock.recorded()[0]
        .headers
        .iter()
        .filter(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
        .collect()
}

// --- The OpenAI Responses fixtures the responses-family suites share ---

use pi_ai::api::openai_responses::OpenAiResponsesOptions;

/// The Models runtime the live E2E probes stream through, upstream's
/// `createModels` over the builtin provider registry.
#[must_use]
pub fn models_runtime() -> pi_ai::models::Models {
    let models = pi_ai::models::create_models(Some(pi_ai::models::CreateModelsOptions::default()));
    for provider in pi_ai::providers::all::builtin_providers() {
        models.set_provider(provider);
    }
    models
}

/// The minimal terminal run the capture suites stream: one
/// `response.completed` frame naming an id and the `completed` status.
#[must_use]
pub fn openai_responses_completed_event() -> serde_json::Value {
    json!({
        "type": "response.completed",
        "response": { "id": "resp_test", "status": "completed" },
    })
}

/// A Responses event stream's SSE body: each event one `data:` frame, closed
/// by the `data: [DONE]` sentinel the shared decoder skips.
#[must_use]
pub fn openai_responses_sse_body(events: &[serde_json::Value]) -> String {
    use std::fmt::Write as _;
    let mut body = String::new();
    for event in events {
        let _ = write!(body, "data: {event}\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

/// The SSE response an OpenAI Responses stream returns over the given
/// events, the mock body the responses-family suites mount.
#[must_use]
pub fn openai_responses_sse_response(events: &[serde_json::Value]) -> pi_ai::http::MockResponse {
    pi_ai::http::MockResponse::status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(openai_responses_sse_body(events))
}

/// Mount the seam mock the openai-responses suites inject: the route answers
/// `/responses` with the events' SSE body.
#[must_use]
pub fn openai_responses_mock(events: &[serde_json::Value]) -> MockHttpClient {
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, events);
    mock
}

/// Mount onto an existing mock, for suites that assert the recorded request.
pub fn openai_responses_mock_with(mock: &MockHttpClient, events: &[serde_json::Value]) {
    mock.on(|request| request.url.contains("/responses"))
        .respond(openai_responses_sse_response(events));
}

/// The keyed openai-responses options the mock suites send: the mock as the
/// transport and a static test key.
#[must_use]
pub fn keyed_openai_responses_options(mock: &MockHttpClient) -> OpenAiResponsesOptions {
    OpenAiResponsesOptions {
        transport_options: mock_transport(mock),
        api_key: Some("test-key".to_owned()),
        ..OpenAiResponsesOptions::default()
    }
}

//! OpenAI Completions conformance suites, ported from the upstream
//! `openai-completions-*` test files at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, grouped per upstream file:
//! prompt-cache, response-model, thinking-as-text, thinking-token-budget,
//! tool-choice, tool-result-images, reasoning-details, raw-stop-reason,
//! empty-tools, vllm-priority, the error-body-regression tier, and the
//! openai-completions tier of sampling-options.
//!
//! Porting restatements: upstream captures request payloads through the
//! mocked SDK's `create(params)` and `onPayload` hooks; the port reads the
//! same payload off the [`MockHttpClient`] seam's recorded request body, and
//! the `convertMessages` calls capture the request the stream dispatches.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the tests pin stream outcomes; an unexpected shape panics the test by design"
)]

use std::sync::Arc;

use pi_ai::api::openai_completions::{
    OpenAiCompletionsOptions, OpenAiToolChoice, stream, stream_simple,
};
use pi_ai::http::MockHttpClient;
use pi_ai::types::{
    AssistantBlock, CacheRetention, Context, Message, Modality, Model, ModelCompat,
    ModelThinkingLevel, ProviderId, SimpleStreamOptions, StopReason, TextContent, ThinkingBudgets,
    ThinkingContent, ThinkingLevel, Tool, ToolCall, ToolChoice, ToolResultBlock,
};
use serde_json::{Value, json};

mod common;
use common::{
    builtin_model, openai_done_chunk, openai_mock_with, recorded_body, recorded_body_at,
    user_message_now,
};

// ---------------------------------------------------------------------------
// Shared capture helpers
// ---------------------------------------------------------------------------

/// Stream the context and take the request's payload; the mock is mounted
/// fresh and shared by the options.
async fn capture(model: &Model, context: &Context, mut options: OpenAiCompletionsOptions) -> Value {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    options.transport_options = common::mock_transport(&mock);
    let _ = stream(model, context, Some(&options)).result().await;
    recorded_body(&mock)
}

/// Stream the context through `stream_simple` and take the request's payload.
async fn capture_simple(
    model: &Model,
    context: &Context,
    mut options: SimpleStreamOptions,
) -> Value {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    options.transport_options = common::mock_transport(&mock);
    let _ = stream_simple(model, context, Some(&options)).result().await;
    recorded_body(&mock)
}

/// The keyed options against the given mock.
fn keyed(mock: &MockHttpClient) -> OpenAiCompletionsOptions {
    common::keyed_openai_options(mock)
}

/// The keyed simple options against the given mock.
fn keyed_simple(mock: &MockHttpClient) -> SimpleStreamOptions {
    SimpleStreamOptions {
        transport_options: common::mock_transport(mock),
        api_key: Some("test".to_owned()),
        ..SimpleStreamOptions::default()
    }
}

/// Capture the keyed stream's request payload and mock: the options come
/// back configured by the suite.
async fn capture_keyed(
    model: &Model,
    configure: impl FnOnce(&mut OpenAiCompletionsOptions),
) -> (Value, MockHttpClient) {
    let mut options = keyed(&MockHttpClient::new());
    configure(&mut options);
    capture_request(model, options).await
}

/// Stream the raw-stop-reason chunks through the zai catalog model on the
/// simple entry, the per-case setup the finish-reason suites repeat.
async fn stream_zai_simple(chunks: &[Value]) -> pi_ai::types::AssistantMessage {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, chunks);
    let model = builtin_model("zai", "glm-5.2");
    let context = tool_context("Hi", None);
    stream_simple(&model, &context, Some(&keyed_simple(&mock)))
        .result()
        .await
}

/// Capture the keyed simple stream's request payload, the options configured
/// by the suite.
async fn capture_keyed_simple(
    model: &Model,
    context: &Context,
    configure: impl FnOnce(&mut SimpleStreamOptions),
) -> Value {
    let mut options = keyed_simple(&MockHttpClient::new());
    configure(&mut options);
    capture_simple(model, context, options).await
}

/// The named recorded request header equals the expected value.
fn assert_header(mock: &MockHttpClient, name: &str, expected: &str) {
    assert_eq!(
        common::recorded_header(mock, name).as_deref(),
        Some(expected),
        "{name}"
    );
}

/// The named recorded request header is absent.
fn assert_no_header(mock: &MockHttpClient, name: &str) {
    assert!(common::recorded_header(mock, name).is_none(), "{name}");
}

/// A compat map from the upstream test's JSON spelling.
fn compat_map(compat: Value) -> ModelCompat {
    serde_json::from_value::<ModelCompat>(compat).expect("compat map")
}

/// A plain tool with the given JSON-schema parameters.
fn tool(name: &str, parameters: Value) -> Tool {
    Tool {
        name: name.to_owned(),
        description: format!("{name} tool."),
        parameters,
        constrained_sampling: None,
    }
}

/// The read tool the tool-history suites send.
fn read_tool() -> Tool {
    tool(
        "read",
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"],
        }),
    )
}

/// The zeroed usage block the replay fixtures carry.
fn empty_usage() -> pi_ai::types::Usage {
    pi_ai::types::Usage::default()
}

/// A tool-result message with the given text content.
fn tool_result_turn(tool_call_id: &str, text: &str) -> Message {
    common::tool_result_message(
        tool_call_id,
        vec![ToolResultBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        None,
    )
}

/// The catalog model retargeted at the proxy base URL the session-affinity
/// suites stream against.
fn proxied(model: Model) -> Model {
    Model {
        base_url: "https://proxy.example.com/v1".to_owned(),
        ..model
    }
}

/// The instructed context the instruction-routing suites send: the system
/// prompt over one user turn.
fn instructed_context() -> Context {
    Context {
        system_prompt: Some("Follow instructions.".to_owned()),
        messages: vec![user_message_now("Hi")],
        ..Context::default()
    }
}

/// The history assistant turn carrying the given blocks, the replay fixture
/// the reasoning-replay suites build.
fn history_turn(api: &str, provider: &str, model_id: &str, blocks: Vec<AssistantBlock>) -> Message {
    Message::Assistant(pi_ai::types::AssistantMessage {
        content: blocks,
        usage: empty_usage(),
        ..common::assistant_message_with_content(api, provider, model_id, Vec::new())
    })
}

// ---------------------------------------------------------------------------
// prompt cache (upstream openai-completions-prompt-cache)
// ---------------------------------------------------------------------------

/// The gpt-4o-mini catalog entry retargeted at the completions wire, with the
/// upstream overrides applied.
fn gpt4o_mini() -> Model {
    common::openai_catalog_model("openai", "gpt-4o-mini")
}

fn gpt4o_mini_with_compat(compat: Value) -> Model {
    Model {
        compat: Some(compat_map(compat)),
        ..gpt4o_mini()
    }
}

/// Stream the session/retention setup against the sys prompt context and take
/// the payload and the mock for header assertions.
async fn capture_request(
    model: &Model,
    options: OpenAiCompletionsOptions,
) -> (Value, MockHttpClient) {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let mut options = options;
    options.transport_options = common::mock_transport(&mock);
    let context = Context {
        system_prompt: Some("sys".to_owned()),
        messages: vec![user_message_now("hi")],
        ..Context::default()
    };
    let _ = stream(model, &context, Some(&options)).result().await;
    (recorded_body(&mock), mock)
}

#[tokio::test]
async fn prompt_cache_key_rides_for_direct_openai_requests_when_caching_is_enabled() {
    let (payload, _mock) = capture_keyed(&gpt4o_mini(), |options| {
        options.session_id = Some("session-123".to_owned());
    })
    .await;

    assert_eq!(payload["prompt_cache_key"], json!("session-123"));
    assert!(payload.get("prompt_cache_retention").is_none());
}

#[tokio::test]
async fn prompt_cache_retention_is_24h_when_cache_retention_is_long() {
    let (payload, _) = capture_keyed(&gpt4o_mini(), |options| {
        options.cache_retention = Some(CacheRetention::Long);
        options.session_id = Some("session-456".to_owned());
    })
    .await;

    assert_eq!(payload["prompt_cache_key"], json!("session-456"));
    assert_eq!(payload["prompt_cache_retention"], json!("24h"));
}

#[tokio::test]
async fn prompt_cache_key_clamps_to_openai_s_64_character_limit() {
    let (payload, _) = capture_keyed(&gpt4o_mini(), |options| {
        options.session_id = Some("x".repeat(67));
    })
    .await;

    assert_eq!(payload["prompt_cache_key"], json!("x".repeat(64)));
}

#[tokio::test]
async fn prompt_cache_fields_are_omitted_when_cache_retention_is_none() {
    let (payload, _) = capture_keyed(&gpt4o_mini(), |options| {
        options.cache_retention = Some(CacheRetention::None);
        options.session_id = Some("session-789".to_owned());
    })
    .await;

    assert!(payload.get("prompt_cache_key").is_none());
    assert!(payload.get("prompt_cache_retention").is_none());
}

#[tokio::test]
async fn prompt_cache_fields_are_omitted_for_non_openai_base_urls_without_long_retention() {
    let model = gpt4o_mini_with_compat(json!({ "supportsLongCacheRetention": false }));
    let model = proxied(model);
    let (payload, _) = capture_keyed(&model, |options| {
        options.cache_retention = Some(CacheRetention::Long);
        options.session_id = Some("session-proxy".to_owned());
    })
    .await;

    assert!(payload.get("prompt_cache_key").is_none());
    assert!(payload.get("prompt_cache_retention").is_none());
}

/// Porting restatement: the env rides the request's provider-scoped
/// [`pi_ai::types::ProviderEnv`] — the port of upstream's
/// `PI_CACHE_RETENTION` process-env read.
#[tokio::test]
async fn pi_cache_retention_env_drives_direct_openai_requests() {
    let (payload, _) = capture_keyed(&gpt4o_mini(), |options| {
        options.env =
            Some(std::iter::once(("PI_CACHE_RETENTION".to_owned(), "long".to_owned())).collect());
        options.session_id = Some("session-env".to_owned());
    })
    .await;

    assert_eq!(payload["prompt_cache_key"], json!("session-env"));
    assert_eq!(payload["prompt_cache_retention"], json!("24h"));
}

#[tokio::test]
async fn session_affinity_headers_follow_the_compat_flag() {
    let model = gpt4o_mini_with_compat(json!({ "sendSessionAffinityHeaders": true }));
    let model = proxied(model);
    let (_, mock) = capture_keyed(&model, |options| {
        options.session_id = Some("session-affinity".to_owned());
    })
    .await;

    assert_header(&mock, "session_id", "session-affinity");
    assert_header(&mock, "x-client-request-id", "session-affinity");
    assert_header(&mock, "x-session-affinity", "session-affinity");
}

/// The Fireworks session-affinity header rides for the model id and its
/// router alias.
#[tokio::test]
async fn sends_fireworks_session_affinity_for_the_model_and_router_alias() {
    for model_id in [
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/routers/glm-5p2-fast",
    ] {
        let model = builtin_model("fireworks", model_id);
        let (_, mock) = capture_keyed(&model, |options| {
            options.session_id = Some("fireworks-session".to_owned());
        })
        .await;

        assert_header(&mock, "x-session-affinity", "fireworks-session");
    }
}

#[tokio::test]
async fn uses_openai_nosession_format_when_configured() {
    let model = gpt4o_mini_with_compat(json!({
        "sendSessionAffinityHeaders": true,
        "sessionAffinityFormat": "openai-nosession",
    }));
    let (payload, mock) = capture_keyed(&model, |options| {
        options.session_id = Some("session-nosession".to_owned());
    })
    .await;

    assert!(payload.get("session_id").is_none());
    assert_eq!(payload["prompt_cache_key"], json!("session-nosession"));
    assert_no_header(&mock, "session_id");
    assert_header(&mock, "x-client-request-id", "session-nosession");
    assert_header(&mock, "x-session-affinity", "session-nosession");
    assert_no_header(&mock, "x-session-id");
}

#[tokio::test]
async fn uses_openrouter_session_affinity_header_when_configured() {
    let model = gpt4o_mini_with_compat(json!({
        "sendSessionAffinityHeaders": true,
        "sessionAffinityFormat": "openrouter",
    }));
    let model = proxied(model);
    let (payload, mock) = capture_keyed(&model, |options| {
        options.session_id = Some("session-proxy".to_owned());
    })
    .await;

    assert!(payload.get("session_id").is_none());
    assert!(payload.get("prompt_cache_key").is_none());
    assert_header(&mock, "x-session-id", "session-proxy");
    assert_no_header(&mock, "session_id");
    assert_no_header(&mock, "x-client-request-id");
    assert_no_header(&mock, "x-session-affinity");
}

#[tokio::test]
async fn sends_openrouter_session_affinity_header_by_default_for_builtin_models() {
    let model = builtin_model("openrouter", "auto");
    let (payload, mock) = capture_keyed(&model, |options| {
        options.session_id = Some("session-openrouter".to_owned());
    })
    .await;

    assert!(payload.get("session_id").is_none());
    assert!(payload.get("prompt_cache_key").is_none());
    assert_header(&mock, "x-session-id", "session-openrouter");
    assert_no_header(&mock, "session_id");
    assert_no_header(&mock, "x-client-request-id");
    assert_no_header(&mock, "x-session-affinity");
}

#[tokio::test]
async fn omits_openrouter_session_affinity_data_when_disabled() {
    let model = Model {
        provider: ProviderId::from("openrouter"),
        base_url: "https://openrouter.ai/api/v1".to_owned(),
        ..gpt4o_mini_with_compat(json!({ "sendSessionAffinityHeaders": false }))
    };
    let (payload, mock) = capture_keyed(&model, |options| {
        options.session_id = Some("session-openrouter".to_owned());
    })
    .await;

    assert!(payload.get("session_id").is_none());
    assert!(payload.get("prompt_cache_key").is_none());
    assert_no_header(&mock, "x-session-id");
}

#[tokio::test]
async fn omits_session_affinity_headers_when_cache_retention_is_none() {
    let model = gpt4o_mini_with_compat(json!({ "sendSessionAffinityHeaders": true }));
    let model = proxied(model);
    let (_, mock) = capture_keyed(&model, |options| {
        options.cache_retention = Some(CacheRetention::None);
        options.session_id = Some("session-affinity".to_owned());
    })
    .await;

    assert_no_header(&mock, "session_id");
    assert_no_header(&mock, "x-client-request-id");
    assert_no_header(&mock, "x-session-affinity");
}

#[tokio::test]
async fn explicit_headers_override_generated_session_affinity_headers() {
    let model = gpt4o_mini_with_compat(json!({ "sendSessionAffinityHeaders": true }));
    let model = proxied(model);
    let (_, mock) = capture_keyed(&model, |options| {
        options.session_id = Some("session-affinity".to_owned());
        options.headers = Some(
            [
                ("session_id".to_owned(), Some("override-session".to_owned())),
                (
                    "x-client-request-id".to_owned(),
                    Some("override-request".to_owned()),
                ),
                (
                    "x-session-affinity".to_owned(),
                    Some("override-affinity".to_owned()),
                ),
            ]
            .into_iter()
            .collect(),
        );
    })
    .await;

    assert_header(&mock, "session_id", "override-session");
    assert_header(&mock, "x-client-request-id", "override-request");
    assert_header(&mock, "x-session-affinity", "override-affinity");
}

// ---------------------------------------------------------------------------
// response model (upstream openai-completions-response-model)
// ---------------------------------------------------------------------------

/// Router/virtual ids (e.g. OpenRouter `auto`) keep `model` pinned to the
/// requested id and surface the routed concrete id on `responseModel`.
fn open_router_auto() -> Model {
    Model {
        id: "openrouter/auto".to_owned(),
        name: "OpenRouter Auto".to_owned(),
        base_url: "https://openrouter.ai/api/v1".to_owned(),
        context_window: 200_000,
        max_tokens: 8192,
        ..common::openai_catalog_model("openrouter", "auto")
    }
}

/// Stream the given chunks to settlement, the shape the response-model suite
/// pins.
async fn complete_chunks(chunks: Vec<Value>) -> pi_ai::types::AssistantMessage {
    let model = open_router_auto();
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &chunks);
    let context = Context {
        messages: vec![user_message_now("hi")],
        ..Context::default()
    };
    let options = OpenAiCompletionsOptions {
        api_key: Some("test".to_owned()),
        transport_options: common::mock_transport(&mock),
        ..OpenAiCompletionsOptions::default()
    };
    stream(&model, &context, Some(&options)).result().await
}

fn routed_chunk(delta: &Value, finish: Option<&str>, usage: Option<Value>) -> Value {
    let mut choice = json!({ "index": 0, "delta": delta });
    if let Some(finish) = finish {
        choice["finish_reason"] = json!(finish);
    }
    let mut chunk = json!({ "id": "chatcmpl-1", "choices": [choice] });
    if let Some(usage) = usage {
        chunk["usage"] = usage;
    }
    chunk
}

/// Stamp a serving model onto every chunk, the upstream chunk fixtures'
/// `model` field.
fn inject_model(chunks: Vec<Value>, model: &str) -> Vec<Value> {
    chunks
        .into_iter()
        .map(|mut chunk| {
            chunk["model"] = json!(model);
            chunk
        })
        .collect()
}

fn routed_usage(prompt: u64, completion: u64) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_tokens_details": { "cached_tokens": 0 },
        "completion_tokens_details": { "reasoning_tokens": 0 },
    })
}

#[tokio::test]
async fn surfaces_routed_chunk_model_on_response_model_without_changing_model() {
    let routed = vec![
        routed_chunk(&json!({ "content": "hi" }), None, None),
        routed_chunk(&json!({}), Some("stop"), Some(routed_usage(10, 5))),
    ];
    let message = complete_chunks(inject_model(routed, "anthropic/claude-opus-4.8")).await;

    assert_eq!(message.model, "openrouter/auto");
    assert_eq!(
        message.response_model.as_deref(),
        Some("anthropic/claude-opus-4.8")
    );
    assert_eq!(message.provider.0, "openrouter");
    assert_eq!(message.stop_reason, StopReason::Stop);
}

#[tokio::test]
async fn leaves_response_model_unset_when_chunks_echo_the_requested_id() {
    let chunks = vec![
        routed_chunk(&json!({ "content": "hi" }), None, None),
        routed_chunk(&json!({}), Some("stop"), Some(routed_usage(1, 1))),
    ];
    let message = complete_chunks(inject_model(chunks, "openrouter/auto")).await;

    assert_eq!(message.model, "openrouter/auto");
    assert_eq!(message.response_model, None);
}

#[tokio::test]
async fn ignores_empty_or_missing_chunk_model() {
    let message = complete_chunks(vec![
        routed_chunk(&json!({ "content": "hi" }), None, None),
        {
            let mut chunk = routed_chunk(&json!({ "content": "!" }), None, None);
            chunk["model"] = json!("");
            chunk
        },
        routed_chunk(&json!({}), Some("stop"), Some(routed_usage(1, 2))),
    ])
    .await;

    assert_eq!(message.model, "openrouter/auto");
    assert_eq!(message.response_model, None);
}

// ---------------------------------------------------------------------------
// thinking as text replay (upstream openai-completions-thinking-as-text)
// ---------------------------------------------------------------------------

/// The full compat belt the thinking-as-text suite constructs, with
/// `requiresThinkingAsText` enabled.
fn thinking_as_text_compat() -> ModelCompat {
    compat_map(json!({
        "supportsStore": true,
        "supportsDeveloperRole": true,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "supportsFinishReason": true,
        "maxTokensField": "max_completion_tokens",
        "requiresToolResultName": false,
        "requiresAssistantAfterToolResult": false,
        "requiresThinkingAsText": true,
        "requiresReasoningContentOnAssistantMessages": false,
        "thinkingFormat": "openai",
        "openRouterRouting": {},
        "vercelGatewayRouting": {},
        "chatTemplateKwargs": {},
        "chatTemplateArgs": {},
        "zaiToolStream": false,
        "supportsThinkingTokenBudget": false,
        "supportsStrictMode": true,
        "supportsOpenAIGrammarTools": false,
        "sendSessionAffinityHeaders": false,
        "sessionAffinityFormat": "openai",
        "supportsLongCacheRetention": true,
    }))
}

fn repro_model() -> Model {
    Model {
        id: "repro-model".to_owned(),
        name: "Repro Model".to_owned(),
        provider: ProviderId::from("repro-provider"),
        base_url: "http://127.0.0.1:1".to_owned(),
        reasoning: true,
        context_window: 128_000,
        max_tokens: 4096,
        compat: Some(thinking_as_text_compat()),
        ..common::openai_catalog_model("openai", "gpt-4o-mini")
    }
}

fn thinking_as_text_context(content: Vec<AssistantBlock>) -> Context {
    let timestamp = pi_ai::auth::resolve::now_ms();
    Context {
        messages: vec![
            Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("hello".to_owned()),
                timestamp,
            }),
            Message::Assistant(pi_ai::types::AssistantMessage {
                usage: empty_usage(),
                ..common::assistant_message_with_content(
                    "openai-completions",
                    "repro-provider",
                    "repro-model",
                    content,
                )
            }),
            Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("continue".to_owned()),
                timestamp,
            }),
        ],
        ..Context::default()
    }
}

fn text_part(text: &str) -> Value {
    json!({ "type": "text", "text": text })
}

/// Same-model thinking rides as assistant text parts next to the answer.
#[tokio::test]
async fn serializes_same_model_thinking_plus_text_replay_as_assistant_text_parts() {
    let model = repro_model();
    let context = thinking_as_text_context(vec![
        AssistantBlock::Thinking(ThinkingContent {
            thinking: "internal reasoning".to_owned(),
            thinking_signature: None,
            redacted: None,
        }),
        AssistantBlock::Text(TextContent {
            text: "visible answer".to_owned(),
            text_signature: None,
        }),
    ]);

    let payload = capture(&model, &context, keyed(&MockHttpClient::new())).await;

    assert_eq!(
        payload["messages"][1],
        json!({
            "role": "assistant",
            "content": [text_part("internal reasoning"), text_part("visible answer")],
        })
    );
}

#[tokio::test]
async fn serializes_same_model_thinking_only_replay_as_assistant_text_parts() {
    let model = repro_model();
    let context = thinking_as_text_context(vec![AssistantBlock::Thinking(ThinkingContent {
        thinking: "internal reasoning".to_owned(),
        thinking_signature: None,
        redacted: None,
    })]);

    let payload = capture(&model, &context, keyed(&MockHttpClient::new())).await;

    assert_eq!(
        payload["messages"][1],
        json!({
            "role": "assistant",
            "content": [text_part("internal reasoning")],
        })
    );
}

/// The replayed thinking-plus-text turn reaches the endpoint and the stream
/// settles with `done`.
#[tokio::test]
async fn reaches_the_endpoint_when_replay_contains_both_thinking_and_text() {
    let mock = MockHttpClient::new();
    openai_mock_with(
        &mock,
        &[
            json!({
                "id": "chatcmpl-repro",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "repro-model",
                "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "ok" }, "finish_reason": null }],
            }),
            json!({
                "id": "chatcmpl-repro",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "repro-model",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1 },
            }),
        ],
    );
    let options = OpenAiCompletionsOptions {
        api_key: Some("test-key".to_owned()),
        transport_options: common::mock_transport(&mock),
        ..OpenAiCompletionsOptions::default()
    };
    let context = thinking_as_text_context(vec![
        AssistantBlock::Thinking(ThinkingContent {
            thinking: "internal reasoning".to_owned(),
            thinking_signature: None,
            redacted: None,
        }),
        AssistantBlock::Text(TextContent {
            text: "visible answer".to_owned(),
            text_signature: None,
        }),
    ]);
    let model = Model {
        base_url: "http://127.0.0.1:1".to_owned(),
        ..repro_model()
    };
    let stream = stream(&model, &context, Some(&options));

    let (terminal_was_done, message) = tokio::join!(
        async {
            let mut terminal_was_done = false;
            while let Some(event) = stream.next().await {
                terminal_was_done =
                    matches!(event, pi_ai::types::AssistantMessageEvent::Done { .. });
            }
            terminal_was_done
        },
        stream.result()
    );

    assert_eq!(mock.request_count(), 1);
    let body = recorded_body(&mock);
    assert_eq!(
        body["messages"][1],
        json!({
            "role": "assistant",
            "content": [text_part("internal reasoning"), text_part("visible answer")],
        })
    );
    assert!(terminal_was_done, "the terminal event is done");
    assert_eq!(message.stop_reason, StopReason::Stop);
}

// ---------------------------------------------------------------------------
// thinking token budget (upstream openai-completions-thinking-token-budget)
// ---------------------------------------------------------------------------

fn vllm_budget_model(compat: Value) -> Model {
    Model {
        id: "zai-org/glm-5.2".to_owned(),
        name: "GLM 5.2 (local vLLM)".to_owned(),
        provider: ProviderId::from("local-vllm"),
        base_url: "http://localhost:8000/v1".to_owned(),
        reasoning: true,
        context_window: 262_144,
        max_tokens: 16_384,
        compat: Some(compat_map(compat)),
        ..common::openai_catalog_model("openai", "gpt-4o-mini")
    }
}

fn vllm_budget_options(
    reasoning: Option<ThinkingLevel>,
    budgets: Option<ThinkingBudgets>,
    max_tokens: Option<u64>,
) -> SimpleStreamOptions {
    SimpleStreamOptions {
        api_key: Some("test".to_owned()),
        reasoning,
        thinking_budgets: budgets,
        max_tokens,
        ..SimpleStreamOptions::default()
    }
}

async fn capture_budget(
    model: &Model,
    reasoning: Option<ThinkingLevel>,
    budgets: Option<ThinkingBudgets>,
    max_tokens: Option<u64>,
) -> Value {
    let context = Context {
        messages: vec![user_message_now("Hi")],
        ..Context::default()
    };
    capture_simple(
        model,
        &context,
        vllm_budget_options(reasoning, budgets, max_tokens),
    )
    .await
}

#[tokio::test]
async fn sends_the_configured_budget_for_the_requested_level() {
    let model = vllm_budget_model(json!({
        "thinkingFormat": "zai",
        "supportsThinkingTokenBudget": true,
    }));
    let budgets = ThinkingBudgets {
        medium: Some(4096),
        ..ThinkingBudgets::default()
    };

    let payload = capture_budget(&model, Some(ThinkingLevel::Medium), Some(budgets), None).await;

    assert_eq!(payload["thinking_token_budget"], json!(4096));
}

#[tokio::test]
async fn omits_the_budget_when_neither_the_field_nor_the_alias_is_set() {
    let model = vllm_budget_model(json!({ "thinkingFormat": "zai" }));
    let budgets = ThinkingBudgets {
        medium: Some(4096),
        ..ThinkingBudgets::default()
    };

    let payload = capture_budget(&model, Some(ThinkingLevel::Medium), Some(budgets), None).await;

    assert!(payload.get("thinking_token_budget").is_none());
    assert!(payload.get("thinking_budget").is_none());
    assert!(payload.get("thinking_budget_tokens").is_none());
}

#[tokio::test]
async fn omits_the_budget_when_thinking_is_off() {
    let model = vllm_budget_model(json!({
        "thinkingFormat": "zai",
        "supportsThinkingTokenBudget": true,
    }));
    let budgets = ThinkingBudgets {
        high: Some(8192),
        ..ThinkingBudgets::default()
    };

    let payload = capture_budget(&model, None, Some(budgets), None).await;

    assert!(payload.get("thinking_token_budget").is_none());
}

#[tokio::test]
async fn clamps_xhigh_and_max_to_the_high_budget() {
    let model = vllm_budget_model(json!({
        "thinkingFormat": "zai",
        "supportsThinkingTokenBudget": true,
    }));
    let budgets = ThinkingBudgets {
        high: Some(8192),
        ..ThinkingBudgets::default()
    };

    let xhigh = capture_budget(&model, Some(ThinkingLevel::Xhigh), Some(budgets), None).await;
    let max = capture_budget(&model, Some(ThinkingLevel::Max), Some(budgets), None).await;

    assert_eq!(xhigh["thinking_token_budget"], json!(8192));
    assert_eq!(max["thinking_token_budget"], json!(8192));
}

#[tokio::test]
async fn leaves_room_for_the_answer_when_the_budget_meets_the_response_ceiling() {
    let model = vllm_budget_model(json!({
        "thinkingFormat": "zai",
        "supportsThinkingTokenBudget": true,
    }));

    let payload = capture_budget(&model, Some(ThinkingLevel::High), None, None).await;

    assert_eq!(payload["thinking_token_budget"], json!(16_384 - 1024));
}

#[tokio::test]
async fn uses_the_caller_max_tokens_as_the_ceiling_when_it_is_lower_than_the_model_cap() {
    let model = vllm_budget_model(json!({
        "thinkingFormat": "zai",
        "supportsThinkingTokenBudget": true,
    }));
    let budgets = ThinkingBudgets {
        high: Some(8192),
        ..ThinkingBudgets::default()
    };

    let payload =
        capture_budget(&model, Some(ThinkingLevel::High), Some(budgets), Some(4096)).await;

    assert_eq!(payload["thinking_token_budget"], json!(4096 - 1024));
}

#[tokio::test]
async fn sends_the_compat_budget_field_when_thinking_token_budget_field_is_set() {
    let budgets = ThinkingBudgets {
        medium: Some(4096),
        ..ThinkingBudgets::default()
    };
    for (field, wire) in [
        ("thinking_budget", "thinking_budget"),
        ("thinking_budget_tokens", "thinking_budget_tokens"),
    ] {
        let model = vllm_budget_model(json!({
            "thinkingFormat": "qwen",
            "thinkingTokenBudgetField": field,
        }));

        let payload =
            capture_budget(&model, Some(ThinkingLevel::Medium), Some(budgets), None).await;

        assert_eq!(payload[wire], json!(4096), "{wire}");
        assert!(payload.get("thinking_token_budget").is_none());
    }
}

#[tokio::test]
async fn lets_thinking_token_budget_field_win_over_the_boolean_alias() {
    let model = vllm_budget_model(json!({
        "thinkingFormat": "zai",
        "supportsThinkingTokenBudget": true,
        "thinkingTokenBudgetField": "thinking_budget",
    }));
    let budgets = ThinkingBudgets {
        medium: Some(4096),
        ..ThinkingBudgets::default()
    };

    let payload = capture_budget(&model, Some(ThinkingLevel::Medium), Some(budgets), None).await;

    assert_eq!(payload["thinking_budget"], json!(4096));
    assert!(payload.get("thinking_token_budget").is_none());
}

#[tokio::test]
async fn puts_the_clamped_budget_in_chat_template_kwargs_when_var_is_thinking_budget() {
    let model = vllm_budget_model(json!({
        "thinkingFormat": "chat-template",
        "chatTemplateKwargs": {
            "enable_thinking": { "$var": "thinking.enabled" },
            "thinking_budget": { "$var": "thinking.budget" },
        },
    }));

    let payload = capture_budget(&model, Some(ThinkingLevel::High), None, None).await;

    assert_eq!(
        payload["chat_template_kwargs"],
        json!({ "enable_thinking": true, "thinking_budget": 16_384 - 1024 })
    );
    assert!(payload.get("thinking_token_budget").is_none());
}

#[tokio::test]
async fn omits_thinking_budget_from_chat_template_kwargs_when_thinking_is_off() {
    let model = vllm_budget_model(json!({
        "thinkingFormat": "chat-template",
        "chatTemplateKwargs": {
            "enable_thinking": { "$var": "thinking.enabled" },
            "thinking_budget": { "$var": "thinking.budget" },
        },
    }));

    let payload = capture_budget(&model, None, None, None).await;

    assert_eq!(
        payload["chat_template_kwargs"],
        json!({ "enable_thinking": false })
    );
}

// ---------------------------------------------------------------------------
// tool choice (upstream openai-completions-tool-choice)
// ---------------------------------------------------------------------------

fn ping_tool() -> Tool {
    tool(
        "ping",
        json!({
            "type": "object",
            "properties": { "ok": { "type": "boolean" } },
            "required": ["ok"],
        }),
    )
}

fn tool_context(text: &str, tools: Option<Vec<Tool>>) -> Context {
    Context {
        messages: vec![user_message_now(text)],
        tools,
        ..Context::default()
    }
}

/// Restatement of upstream's `toolChoice: "required"` cast: the pi simple
/// entry only carries the neutral `auto`/`none` choices, so the required
/// choice rides the typed [`OpenAiToolChoice::Required`] on
/// [`OpenAiCompletionsOptions`] through `stream`.
#[tokio::test]
async fn forwards_a_required_tool_choice_to_the_payload() {
    let model = gpt4o_mini();
    let context = tool_context("Call ping with ok=true", Some(vec![ping_tool()]));
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let options = OpenAiCompletionsOptions {
        api_key: Some("test".to_owned()),
        tool_choice: Some(OpenAiToolChoice::Required),
        transport_options: common::mock_transport(&mock),
        ..OpenAiCompletionsOptions::default()
    };

    let _ = stream(&model, &context, Some(&options)).result().await;
    let payload = recorded_body(&mock);

    assert_eq!(payload["tool_choice"], json!("required"));
    let tools = payload["tools"].as_array().expect("tools");
    assert!(!tools.is_empty());
}

#[tokio::test]
async fn includes_tool_choice_when_no_tools_are_provided() {
    let model = gpt4o_mini();
    let context = tool_context("Summarize the conversation", None);
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let options = SimpleStreamOptions {
        api_key: Some("test".to_owned()),
        tool_choice: Some(ToolChoice::None),
        transport_options: common::mock_transport(&mock),
        ..SimpleStreamOptions::default()
    };

    let _ = stream_simple(&model, &context, Some(&options))
        .result()
        .await;
    let payload = recorded_body(&mock);

    assert_eq!(payload["tool_choice"], json!("none"));
    assert!(payload.get("tools").is_none());
}

#[tokio::test]
async fn omits_strict_when_compat_disables_strict_mode() {
    let model = gpt4o_mini_with_compat(json!({ "supportsStrictMode": false }));
    let context = tool_context("Call ping with ok=true", Some(vec![ping_tool()]));
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let options = keyed_simple(&mock);

    let _ = stream_simple(&model, &context, Some(&options))
        .result()
        .await;
    let payload = recorded_body(&mock);

    let function = &payload["tools"][0]["function"];
    assert!(function.is_object(), "got: {function}");
    assert!(function.get("strict").is_none());
}

#[tokio::test]
async fn maps_groq_qwen_reasoning_levels_to_default_reasoning_effort() {
    let model = builtin_model("groq", "qwen/qwen3.6-27b");
    let context = tool_context("Hi", None);
    let payload = capture_keyed_simple(&model, &context, |options| {
        options.reasoning = Some(ThinkingLevel::Medium);
    })
    .await;

    assert_eq!(payload["reasoning_effort"], json!("default"));
}

#[tokio::test]
async fn keeps_normal_reasoning_effort_for_groq_models_without_compat_mapping() {
    let model = builtin_model("groq", "openai/gpt-oss-20b");
    let context = tool_context("Hi", None);
    let payload = capture_keyed_simple(&model, &context, |options| {
        options.reasoning = Some(ThinkingLevel::Medium);
    })
    .await;

    assert_eq!(payload["reasoning_effort"], json!("medium"));
}

#[tokio::test]
async fn enables_tool_stream_for_supported_zai_models_with_tools() {
    let model = builtin_model("zai", "glm-5.2");
    let context = tool_context("Call ping with ok=true", Some(vec![ping_tool()]));
    let payload = capture_simple(&model, &context, keyed_simple(&MockHttpClient::new())).await;

    assert_eq!(payload["tool_stream"], json!(true));
}

/// The catalog pins z.ai's tool-stream support.
#[test]
fn stores_zai_tool_stream_support_in_model_compat_metadata() {
    for (provider, id) in [
        ("zai", "glm-4.7"),
        ("zai", "glm-5-turbo"),
        ("zai", "glm-5.2"),
    ] {
        let model = builtin_model(provider, id);
        assert_eq!(
            model
                .compat
                .as_ref()
                .and_then(|compat| compat.zai_tool_stream),
            Some(true),
            "{provider}/{id}"
        );
    }
}

/// The z.ai effort metadata: the level map the models-dev generation pinned,
/// restated for the catalog the port carries (zai-coding-cn carries
/// `glm-5.3`; `glm-5.2` lives on the `zai` provider).
#[test]
fn stores_zai_effort_metadata() {
    let glm52_map = [
        (ModelThinkingLevel::Off, Some("none")),
        (ModelThinkingLevel::Minimal, None),
        (ModelThinkingLevel::Low, None),
        (ModelThinkingLevel::Medium, None),
        (ModelThinkingLevel::High, Some("high")),
        (ModelThinkingLevel::Xhigh, None),
        (ModelThinkingLevel::Max, Some("max")),
    ];
    let glm53_map = [
        (ModelThinkingLevel::Off, None),
        (ModelThinkingLevel::Minimal, None),
        (ModelThinkingLevel::Low, Some("low")),
        (ModelThinkingLevel::Medium, None),
        (ModelThinkingLevel::High, Some("high")),
        (ModelThinkingLevel::Xhigh, None),
        (ModelThinkingLevel::Max, Some("max")),
    ];
    let map = |entries: &[(ModelThinkingLevel, Option<&str>)]| -> pi_ai::types::ThinkingLevelMap {
        entries
            .iter()
            .map(|(level, effort)| (*level, effort.map(str::to_owned)))
            .collect()
    };

    for provider in ["zai", "zai-coding-cn"] {
        if provider == "zai" {
            for id in ["glm-5.2", "glm-5.2-highspeed"] {
                let model = builtin_model(provider, id);
                assert_eq!(
                    model
                        .compat
                        .as_ref()
                        .and_then(|compat| compat.supports_reasoning_effort),
                    Some(true),
                    "{provider}/{id}"
                );
                assert_eq!(model.thinking_level_map.as_ref(), Some(&map(&glm52_map)));
            }
        }
        let glm53 = builtin_model(provider, "glm-5.3");
        assert_eq!(
            glm53
                .compat
                .as_ref()
                .and_then(|compat| compat.supports_reasoning_effort),
            Some(true)
        );
        assert_eq!(glm53.thinking_level_map.as_ref(), Some(&map(&glm53_map)));
    }
}

#[tokio::test]
async fn maps_zai_glm_5_2_thinking_levels_to_reasoning_effort() {
    let model = builtin_model("zai", "glm-5.2");
    let cases = [
        (ThinkingLevel::Low, "high"),
        (ThinkingLevel::Medium, "high"),
        (ThinkingLevel::High, "high"),
        (ThinkingLevel::Max, "max"),
    ];

    for (reasoning, effort) in cases {
        let context = tool_context("Hi", None);
        let payload = capture_keyed_simple(&model, &context, |options| {
            options.reasoning = Some(reasoning);
        })
        .await;

        assert_eq!(
            payload["thinking"],
            json!({ "type": "enabled", "clear_thinking": false })
        );
        assert_eq!(payload["reasoning_effort"], json!(effort));
    }
}

#[tokio::test]
async fn preserves_zai_thinking_when_replaying_reasoning_content() {
    let model = builtin_model("zai", "glm-5.2");
    let context = Context {
        messages: vec![
            user_message_now("Read README.md"),
            history_turn(
                "openai-completions",
                "zai",
                "glm-5.2",
                vec![
                    AssistantBlock::Thinking(ThinkingContent {
                        thinking: "prior reasoning".to_owned(),
                        thinking_signature: Some("reasoning_content".to_owned()),
                        redacted: None,
                    }),
                    AssistantBlock::ToolCall(ToolCall {
                        id: "call_1".to_owned(),
                        name: "read".to_owned(),
                        arguments: serde_json::Map::from_iter([(
                            "path".to_owned(),
                            json!("README.md"),
                        )]),
                        thought_signature: None,
                        namespace: None,
                    }),
                ],
            ),
            tool_result_turn("call_1", "contents"),
            user_message_now("Continue"),
        ],
        ..Context::default()
    };
    let payload = capture_keyed_simple(&model, &context, |options| {
        options.reasoning = Some(ThinkingLevel::High);
    })
    .await;

    let replayed = payload["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == json!("assistant"))
        .expect("the replayed assistant");
    assert_eq!(replayed["reasoning_content"], json!("prior reasoning"));
    assert_eq!(
        payload["thinking"],
        json!({ "type": "enabled", "clear_thinking": false })
    );
}

#[tokio::test]
async fn omits_zai_glm_5_2_reasoning_effort_when_thinking_is_off() {
    let model = builtin_model("zai", "glm-5.2");
    let context = tool_context("Hi", None);

    let payload = capture_simple(&model, &context, keyed_simple(&MockHttpClient::new())).await;

    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
    assert!(payload.get("reasoning_effort").is_none());
}

/// The compat override rides the catalog model's own compat object.
#[tokio::test]
async fn respects_explicit_zai_tool_stream_compat_override() {
    let base = builtin_model("zai", "glm-5.2");
    let mut compat = base.compat.clone().expect("catalog compat");
    compat.zai_tool_stream = Some(true);
    let model = Model {
        compat: Some(compat),
        ..base
    };
    let context = tool_context("Call ping with ok=true", Some(vec![ping_tool()]));

    let payload = capture_simple(&model, &context, keyed_simple(&MockHttpClient::new())).await;

    assert_eq!(payload["tool_stream"], json!(true));
}

#[tokio::test]
async fn omits_tool_stream_when_no_tools_are_provided() {
    let model = builtin_model("zai", "glm-5.2");
    let context = tool_context("Hi", None);

    let payload = capture_simple(&model, &context, keyed_simple(&MockHttpClient::new())).await;

    assert!(payload.get("tool_stream").is_none());
}

#[tokio::test]
async fn maps_non_standard_provider_finish_reason_values_to_stop_reason_error() {
    let chunks = vec![
        json!({ "choices": [{ "delta": { "content": "partial" }, "finish_reason": null }] }),
        json!({
            "choices": [{ "delta": {}, "finish_reason": "network_error" }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "prompt_tokens_details": { "cached_tokens": 0 },
                "completion_tokens_details": { "reasoning_tokens": 0 },
            },
        }),
    ];
    let result = stream_zai_simple(&chunks).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Provider finish_reason: network_error")
    );
}

#[tokio::test]
async fn ignores_null_stream_chunks_from_openai_compatible_providers() {
    let chunks = vec![
        Value::Null,
        json!({
            "id": "chatcmpl-test",
            "choices": [{ "delta": { "content": "OK" }, "finish_reason": null }],
        }),
        json!({
            "id": "chatcmpl-test",
            "choices": [{ "delta": {}, "finish_reason": "stop" }],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "prompt_tokens_details": { "cached_tokens": 0 },
                "completion_tokens_details": { "reasoning_tokens": 0 },
            },
        }),
    ];
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &chunks);
    let model = gpt4o_mini();
    let context = tool_context("Reply with exactly OK", None);

    let result = stream_simple(&model, &context, Some(&keyed_simple(&mock)))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.error_message, None);
    assert_eq!(result.response_id.as_deref(), Some("chatcmpl-test"));
    assert_eq!(result.usage.total_tokens, 4);
    assert_eq!(
        result.content,
        vec![AssistantBlock::Text(TextContent {
            text: "OK".to_owned(),
            text_signature: None,
        })]
    );
}

#[tokio::test]
async fn errors_when_a_stream_ends_after_only_null_finish_reason_chunks() {
    let chunks = vec![
        json!({
            "id": "chatcmpl-truncated",
            "choices": [{ "delta": { "content": "partial answer" }, "finish_reason": null }],
        }),
        json!({
            "id": "chatcmpl-truncated",
            "choices": [{ "delta": { "content": "partial answer" }, "finish_reason": null }],
        }),
    ];
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &chunks);
    let model = gpt4o_mini();
    let context = tool_context("Reply with a longer sentence", None);

    let result = stream_simple(&model, &context, Some(&keyed_simple(&mock)))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Stream ended without finish_reason")
    );
}

#[tokio::test]
async fn accepts_streams_without_finish_reason_when_compat_disables_it() {
    let chunks = vec![json!({
        "id": "chatcmpl-no-finish-reason",
        "choices": [{ "delta": { "content": "complete answer" }, "finish_reason": null }],
    })];
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &chunks);
    let model = gpt4o_mini_with_compat(json!({ "supportsFinishReason": false }));
    let context = tool_context("Reply with a complete answer", None);

    let result = stream_simple(&model, &context, Some(&keyed_simple(&mock)))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.error_message, None);
    assert_eq!(
        result.content,
        vec![AssistantBlock::Text(TextContent {
            text: "complete answer".to_owned(),
            text_signature: None,
        })]
    );
}

#[tokio::test]
async fn ignores_empty_custom_objects_on_function_tool_call_deltas() {
    let chunks = vec![json!({
        "id": "chatcmpl-empty-custom",
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "read", "arguments": "{\"path\":\"README.md\"}" },
                    "custom": {},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    })];
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &chunks);
    let model = gpt4o_mini();
    let context = tool_context("Read README.md", Some(vec![read_tool()]));

    let result = stream_simple(&model, &context, Some(&keyed_simple(&mock)))
        .result()
        .await;

    assert_eq!(
        result.content,
        vec![AssistantBlock::ToolCall(ToolCall {
            id: "call_1".to_owned(),
            name: "read".to_owned(),
            arguments: serde_json::Map::from_iter([("path".to_owned(), json!("README.md"))]),
            thought_signature: None,
            namespace: None,
        })]
    );
}

#[tokio::test]
async fn coalesces_tool_call_deltas_by_stable_index_when_provider_mutates_ids_mid_stream() {
    let chunks = vec![
        json!({
            "id": "chatcmpl-kimi-bad-stream",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "functions.read:0",
                        "type": "function",
                        "function": { "name": "read", "arguments": "" },
                    }],
                },
                "finish_reason": null,
            }],
        }),
        json!({
            "id": "chatcmpl-kimi-bad-stream",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "chatcmpl-tool-a",
                        "type": "function",
                        "function": { "name": null, "arguments": "{\"path\":\"README" },
                    }],
                },
                "finish_reason": null,
            }],
        }),
        json!({
            "id": "chatcmpl-kimi-bad-stream",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "chatcmpl-tool-b",
                        "type": "function",
                        "function": { "name": null, "arguments": ".md\"}" },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": { "cached_tokens": 0 },
                "completion_tokens_details": { "reasoning_tokens": 0 },
            },
        }),
    ];
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &chunks);
    let model = gpt4o_mini();
    let context = tool_context("Read README.md", Some(vec![read_tool()]));
    let stream = stream_simple(&model, &context, Some(&keyed_simple(&mock)));

    let (tool_call_content_indexes, result) = tokio::join!(
        async {
            let mut tool_call_content_indexes: Vec<u64> = Vec::new();
            while let Some(event) = stream.next().await {
                match event {
                    pi_ai::types::AssistantMessageEvent::ToolcallStart {
                        content_index, ..
                    }
                    | pi_ai::types::AssistantMessageEvent::ToolcallDelta {
                        content_index, ..
                    }
                    | pi_ai::types::AssistantMessageEvent::ToolcallEnd { content_index, .. } => {
                        tool_call_content_indexes.push(content_index);
                    }
                    _ => {}
                }
            }
            tool_call_content_indexes
        },
        stream.result()
    );
    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(tool_call_content_indexes, [0, 0, 0, 0, 0]);
    assert_eq!(result.content.len(), 1);
    let AssistantBlock::ToolCall(tool_call) = &result.content[0] else {
        panic!("expected a tool call");
    };
    assert_eq!(tool_call.id, "functions.read:0");
    assert_eq!(tool_call.name, "read");
    assert_eq!(
        tool_call.arguments,
        serde_json::Map::from_iter([("path".to_owned(), json!("README.md"))])
    );
}

#[expect(
    clippy::too_many_lines,
    reason = "the upstream case streams one chunk set with three mixed deltas; splitting it would scatter the event-count assertions"
)]
#[tokio::test]
async fn accumulates_mixed_content_reasoning_and_parallel_tool_call_deltas_independently() {
    let chunks = vec![
        json!({
            "id": "chatcmpl-mixed-deltas",
            "choices": [{
                "delta": {
                    "content": "answer 1",
                    "reasoning_content": "think 1",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "tc_read_initial",
                            "type": "function",
                            "function": { "name": "read", "arguments": "{\"path\":\"README" },
                        },
                        {
                            "index": 1,
                            "id": "tc_grep_initial",
                            "type": "function",
                            "function": { "name": "grep", "arguments": "{\"pattern\":\"PATTERN" },
                        },
                        {
                            "id": "tc_list_no_index",
                            "type": "function",
                            "function": { "name": "list", "arguments": "{\"path\":\"packages" },
                        },
                        {
                            "id": "tc_write_no_index",
                            "type": "function",
                            "function": { "name": "write", "arguments": "{\"path\":\"out" },
                        },
                    ],
                },
                "finish_reason": null,
            }],
        }),
        json!({
            "id": "chatcmpl-mixed-deltas",
            "choices": [{
                "delta": {
                    "content": " answer 2",
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "tc_grep_changed",
                            "type": "function",
                            "function": { "arguments": "\",\"path\":\"src" },
                        },
                        {
                            "id": "tc_write_no_index",
                            "type": "function",
                            "function": { "arguments": ".txt\",\"content\":\"ok\"}" },
                        },
                        {
                            "id": "tc_list_no_index",
                            "type": "function",
                            "function": { "arguments": "/ai\"}" },
                        },
                    ],
                },
                "finish_reason": null,
            }],
        }),
        json!({
            "id": "chatcmpl-mixed-deltas",
            "choices": [{
                "delta": {
                    "content": "\n",
                    "reasoning_content": " think 2",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "tc_read_changed",
                            "type": "function",
                            "function": { "arguments": ".md\"}" },
                        },
                        { "index": 1, "type": "function", "function": { "arguments": "\"}" } },
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "prompt_tokens_details": { "cached_tokens": 0 },
                "completion_tokens_details": { "reasoning_tokens": 2 },
            },
        }),
    ];
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &chunks);
    let model = gpt4o_mini();
    let tools = vec![
        tool(
            "read",
            json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }),
        ),
        tool(
            "grep",
            json!({
                "type": "object",
                "properties": { "pattern": { "type": "string" }, "path": { "type": "string" } },
                "required": ["pattern", "path"],
            }),
        ),
        tool(
            "list",
            json!({ "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] }),
        ),
        tool(
            "write",
            json!({
                "type": "object",
                "properties": { "path": { "type": "string" }, "content": { "type": "string" } },
                "required": ["path", "content"],
            }),
        ),
    ];
    let context = tool_context("Think, answer, and use tools.", Some(tools));
    let stream = stream_simple(&model, &context, Some(&keyed_simple(&mock)));
    let mut tool_events_by_index: std::collections::BTreeMap<u64, Vec<&'static str>> =
        std::collections::BTreeMap::new();
    let mut counts: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();

    let ((), result) = tokio::join!(
        async {
            use pi_ai::types::AssistantMessageEvent as Event;
            while let Some(event) = stream.next().await {
                let (name, content_index) = match &event {
                    Event::Start { .. } => ("start", None),
                    Event::TextStart { content_index, .. } => ("text_start", Some(*content_index)),
                    Event::TextDelta { content_index, .. } => ("text_delta", Some(*content_index)),
                    Event::TextEnd { content_index, .. } => ("text_end", Some(*content_index)),
                    Event::ThinkingStart { content_index, .. } => {
                        ("thinking_start", Some(*content_index))
                    }
                    Event::ThinkingDelta { content_index, .. } => {
                        ("thinking_delta", Some(*content_index))
                    }
                    Event::ThinkingEnd { content_index, .. } => {
                        ("thinking_end", Some(*content_index))
                    }
                    Event::ToolcallStart { content_index, .. } => {
                        ("toolcall_start", Some(*content_index))
                    }
                    Event::ToolcallDelta { content_index, .. } => {
                        ("toolcall_delta", Some(*content_index))
                    }
                    Event::ToolcallEnd { content_index, .. } => {
                        ("toolcall_end", Some(*content_index))
                    }
                    Event::Done { .. } | Event::Error { .. } => ("terminal", None),
                };
                *counts.entry(name).or_default() += 1;
                if let Some(index) = content_index
                    && matches!(name, "toolcall_start" | "toolcall_delta" | "toolcall_end")
                {
                    tool_events_by_index.entry(index).or_default().push(name);
                }
            }
        },
        stream.result()
    );

    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(counts["text_start"], 1);
    assert_eq!(counts["text_delta"], 3);
    assert_eq!(counts["text_end"], 1);
    assert_eq!(counts["thinking_start"], 1);
    assert_eq!(counts["thinking_delta"], 2);
    assert_eq!(counts["thinking_end"], 1);
    assert_eq!(counts["toolcall_start"], 4);
    assert_eq!(counts["toolcall_delta"], 9);
    assert_eq!(counts["toolcall_end"], 4);
    for (index, expected) in [
        (
            2,
            vec![
                "toolcall_start",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_end",
            ],
        ),
        (
            3,
            vec![
                "toolcall_start",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_end",
            ],
        ),
        (
            4,
            vec![
                "toolcall_start",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_end",
            ],
        ),
        (
            5,
            vec![
                "toolcall_start",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_end",
            ],
        ),
    ] {
        assert_eq!(
            tool_events_by_index.get(&index),
            Some(&expected),
            "index {index}"
        );
    }

    assert_eq!(result.content.len(), 6);
    assert_eq!(
        result.content[0],
        AssistantBlock::Text(TextContent {
            text: "answer 1 answer 2\n".to_owned(),
            text_signature: None,
        })
    );
    assert_eq!(
        result.content[1],
        AssistantBlock::Thinking(ThinkingContent {
            thinking: "think 1 think 2".to_owned(),
            thinking_signature: Some("reasoning_content".to_owned()),
            redacted: None,
        })
    );
    let expected_calls = [
        (
            "tc_read_initial",
            "read",
            serde_json::Map::from_iter([("path".to_owned(), json!("README.md"))]),
        ),
        (
            "tc_grep_initial",
            "grep",
            serde_json::Map::from_iter([
                ("pattern".to_owned(), json!("PATTERN")),
                ("path".to_owned(), json!("src")),
            ]),
        ),
        (
            "tc_list_no_index",
            "list",
            serde_json::Map::from_iter([("path".to_owned(), json!("packages/ai"))]),
        ),
        (
            "tc_write_no_index",
            "write",
            serde_json::Map::from_iter([
                ("path".to_owned(), json!("out.txt")),
                ("content".to_owned(), json!("ok")),
            ]),
        ),
    ];
    for (index, (id, name, arguments)) in expected_calls.into_iter().enumerate() {
        let AssistantBlock::ToolCall(tool_call) = &result.content[index + 2] else {
            panic!("block {index} is not a tool call");
        };
        assert_eq!(tool_call.id, id, "block {index}");
        assert_eq!(tool_call.name, name, "block {index}");
        assert_eq!(tool_call.arguments, arguments, "block {index}");
    }
}

#[tokio::test]
async fn uses_system_messages_for_non_openai_anthropic_openrouter_reasoning_instructions() {
    let model = builtin_model("openrouter", "deepseek/deepseek-v4-pro");
    let context = instructed_context();

    let payload = capture_simple(&model, &context, keyed_simple(&MockHttpClient::new())).await;

    assert_eq!(payload["messages"][0]["role"], json!("system"));
}

#[tokio::test]
async fn keeps_developer_messages_for_openai_and_anthropic_openrouter_batch_instructions() {
    for (provider, id) in [
        ("openrouter", "openai/gpt-5.2-codex"),
        ("openrouter", "anthropic/claude-fable-5.1:batch"),
    ] {
        let model = builtin_model(provider, id);
        let context = Context {
            system_prompt: Some("Follow instructions.".to_owned()),
            messages: vec![user_message_now("Hi")],
            ..Context::default()
        };

        let payload = capture_simple(&model, &context, keyed_simple(&MockHttpClient::new())).await;

        assert_eq!(
            payload["messages"][0]["role"],
            json!("developer"),
            "{}",
            model.id
        );
    }
}

#[tokio::test]
async fn keeps_developer_messages_for_openai_reasoning_model_instructions() {
    let model = common::openai_catalog_model("openai", "gpt-5.5");
    let context = instructed_context();

    let payload = capture_simple(&model, &context, keyed_simple(&MockHttpClient::new())).await;

    assert_eq!(payload["messages"][0]["role"], json!("developer"));
}

/// Porting restatement: upstream pins this model as
/// `getModel("openai", "gpt-5.5")` retargeted at the completions wire.
#[test]
fn stores_openrouter_kimi_k2_6_reasoning_replay_compat_in_builtin_metadata() {
    let model = builtin_model("openrouter", "moonshotai/kimi-k2.6");
    let compat = model.compat.as_ref().expect("catalog compat");
    assert_eq!(compat.supports_developer_role, Some(false));
    assert_eq!(
        compat.requires_reasoning_content_on_assistant_messages,
        Some(true)
    );
}

#[test]
fn stores_xiaomi_mimo_reasoning_replay_compat_in_builtin_metadata() {
    for provider in [
        "xiaomi",
        "xiaomi-token-plan-cn",
        "xiaomi-token-plan-ams",
        "xiaomi-token-plan-sgp",
    ] {
        let model = builtin_model(provider, "mimo-v2.5-pro");
        let compat = model.compat.as_ref().expect("catalog compat");
        assert_eq!(
            compat.requires_reasoning_content_on_assistant_messages,
            Some(true)
        );
        assert_eq!(
            compat.thinking_format,
            Some(pi_ai::types::ThinkingFormat::Deepseek)
        );
        assert_eq!(compat.max_tokens_field, None);
        assert_eq!(compat.supports_developer_role, None);
    }
}

#[test]
fn stores_qwen_token_plan_reasoning_replay_compat_in_builtin_metadata() {
    for provider in [
        "qwen-token-plan",
        "qwen-token-plan-cn",
        "qwen-token-plan-individual",
    ] {
        let model = builtin_model(provider, "qwen3.7-max");
        let compat = model.compat.as_ref().expect("catalog compat");
        assert_eq!(
            compat.thinking_format,
            Some(pi_ai::types::ThinkingFormat::Qwen)
        );
        assert_eq!(
            compat.requires_reasoning_content_on_assistant_messages,
            None
        );
        assert_eq!(compat.supports_developer_role, Some(false));
        assert_eq!(compat.supports_store, Some(false));
    }
}

#[tokio::test]
async fn replays_xiaomi_mimo_assistant_tool_calls_with_empty_reasoning_content() {
    let model = builtin_model("xiaomi", "mimo-v2.5-pro");
    let context = Context {
        messages: vec![
            user_message_now("Read README.md"),
            history_turn(
                "openai-completions",
                "xiaomi",
                "mimo-v2.5-pro",
                vec![AssistantBlock::ToolCall(ToolCall {
                    id: "call_1".to_owned(),
                    name: "read".to_owned(),
                    arguments: serde_json::Map::from_iter([(
                        "path".to_owned(),
                        json!("README.md"),
                    )]),
                    thought_signature: None,
                    namespace: None,
                })],
            ),
            tool_result_turn("call_1", "contents"),
        ],
        ..Context::default()
    };
    let payload = capture_keyed_simple(&model, &context, |options| {
        options.reasoning = Some(ThinkingLevel::High);
    })
    .await;

    let replayed = payload["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == json!("assistant"))
        .expect("the replayed assistant");
    assert_eq!(replayed["role"], json!("assistant"));
    assert_eq!(replayed["reasoning_content"], json!(""));
    assert_eq!(payload["thinking"], json!({ "type": "enabled" }));
    assert_eq!(payload["reasoning_effort"], json!("high"));
}

#[tokio::test]
async fn normalizes_opencode_go_reasoning_deltas_to_reasoning_content_for_replay() {
    let chunks = vec![json!({
        "id": "chatcmpl-opencode-go-reasoning",
        "choices": [{ "delta": { "reasoning": "think" }, "finish_reason": "stop" }],
    })];
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &chunks);
    let model = common::openai_catalog_model("opencode-go", "kimi-k2.6");
    let context = tool_context("Use reasoning.", None);

    let result = stream_simple(&model, &context, Some(&keyed_simple(&mock)))
        .result()
        .await;

    assert_eq!(
        result.content,
        vec![AssistantBlock::Thinking(ThinkingContent {
            thinking: "think".to_owned(),
            thinking_signature: Some("reasoning_content".to_owned()),
            redacted: None,
        })]
    );
}

#[tokio::test]
async fn keeps_non_opencode_go_reasoning_deltas_on_the_original_reasoning_field() {
    let chunks = vec![json!({
        "id": "chatcmpl-reasoning",
        "choices": [{ "delta": { "reasoning": "think" }, "finish_reason": "stop" }],
    })];
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &chunks);
    let model = gpt4o_mini();
    let context = tool_context("Use reasoning.", None);

    let result = stream_simple(&model, &context, Some(&keyed_simple(&mock)))
        .result()
        .await;

    assert_eq!(
        result.content,
        vec![AssistantBlock::Thinking(ThinkingContent {
            thinking: "think".to_owned(),
            thinking_signature: Some("reasoning".to_owned()),
            redacted: None,
        })]
    );
}

/// Porting restatement: upstream calls `convertMessages` directly with a full
/// compat object; the port captures the request the stream dispatches with
/// the same compat.
#[tokio::test]
async fn replays_opencode_go_reasoning_thinking_blocks_as_reasoning_content() {
    let model = Model {
        compat: Some(compat_map(json!({
            "supportsStore": false,
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": true,
            "supportsUsageInStreaming": true,
            "supportsFinishReason": true,
            "maxTokensField": "max_completion_tokens",
            "requiresToolResultName": false,
            "requiresAssistantAfterToolResult": false,
            "requiresThinkingAsText": false,
            "requiresReasoningContentOnAssistantMessages": false,
            "thinkingFormat": "openai",
            "openRouterRouting": {},
            "vercelGatewayRouting": {},
            "chatTemplateKwargs": {},
            "chatTemplateArgs": {},
            "zaiToolStream": false,
            "supportsStrictMode": true,
            "supportsOpenAIGrammarTools": false,
            "sendSessionAffinityHeaders": false,
            "sessionAffinityFormat": "openai",
            "supportsLongCacheRetention": true,
        }))),
        ..common::openai_catalog_model("opencode-go", "kimi-k2.6")
    };
    let context = Context {
        messages: vec![Message::Assistant(pi_ai::types::AssistantMessage {
            content: vec![
                AssistantBlock::Thinking(ThinkingContent {
                    thinking: "think".to_owned(),
                    thinking_signature: Some("reasoning".to_owned()),
                    redacted: None,
                }),
                AssistantBlock::ToolCall(ToolCall {
                    id: "call_1".to_owned(),
                    name: "read".to_owned(),
                    arguments: serde_json::Map::from_iter([(
                        "path".to_owned(),
                        json!("README.md"),
                    )]),
                    thought_signature: None,
                    namespace: None,
                }),
            ],
            usage: empty_usage(),
            ..common::assistant_message_with_content(
                "openai-completions",
                "opencode-go",
                "kimi-k2.6",
                Vec::new(),
            )
        })],
        ..Context::default()
    };

    let payload = capture(&model, &context, keyed(&MockHttpClient::new())).await;

    let replayed = payload["messages"][0].as_object().expect("assistant");
    assert_eq!(replayed["role"], json!("assistant"));
    assert_eq!(replayed["reasoning_content"], json!("think"));
    assert!(replayed.get("reasoning").is_none());
}

#[tokio::test]
async fn sends_thinking_disabled_for_opencode_go_kimi_k2_6_when_thinking_is_off() {
    let model = common::openai_catalog_model("opencode-go", "kimi-k2.6");
    let context = tool_context("Hi", None);

    let payload = capture_simple(&model, &context, keyed_simple(&MockHttpClient::new())).await;

    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
    assert!(payload.get("reasoning_effort").is_none());
}

#[tokio::test]
async fn sends_thinking_enabled_for_opencode_go_kimi_k2_6_when_thinking_is_enabled() {
    let model = common::openai_catalog_model("opencode-go", "kimi-k2.6");
    let context = tool_context("Hi", None);
    let payload = capture_keyed_simple(&model, &context, |options| {
        options.reasoning = Some(ThinkingLevel::High);
    })
    .await;

    assert_eq!(payload["thinking"], json!({ "type": "enabled" }));
    assert!(payload.get("reasoning_effort").is_none());
}

#[tokio::test]
async fn omits_disabled_thinking_for_moonshot_kimi_k2_7_code_models() {
    for provider in ["moonshotai", "moonshotai-cn"] {
        let model = builtin_model(provider, "kimi-k2.7-code");
        let context = tool_context("Hi", None);

        let payload = capture_simple(&model, &context, keyed_simple(&MockHttpClient::new())).await;

        assert!(payload.get("thinking").is_none(), "{provider}");
        assert!(payload.get("reasoning_effort").is_none(), "{provider}");
    }
}

#[tokio::test]
async fn keeps_disabled_thinking_for_moonshot_kimi_k2_6_when_thinking_is_off() {
    let model = builtin_model("moonshotai-cn", "kimi-k2.6");
    let context = tool_context("Hi", None);

    let payload = capture_simple(&model, &context, keyed_simple(&MockHttpClient::new())).await;

    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
    assert!(payload.get("reasoning_effort").is_none());
}

#[tokio::test]
async fn sends_max_tokens_for_opencode_completions_models() {
    for provider in ["opencode-go", "opencode"] {
        let model = common::openai_catalog_model(provider, "kimi-k2.6");
        assert_eq!(
            model
                .compat
                .as_ref()
                .and_then(|compat| compat.max_tokens_field),
            Some(pi_ai::types::MaxTokensField::MaxTokens)
        );
        let context = tool_context("Hi", None);
        let payload = capture_keyed_simple(&model, &context, |options| {
            options.max_tokens = Some(123);
        })
        .await;

        assert_eq!(payload["max_tokens"], json!(123), "{provider}");
        assert!(payload.get("max_completion_tokens").is_none(), "{provider}");
    }
}

fn deepseek_custom_model(base_url: &str, id: &str) -> Model {
    Model {
        id: id.to_owned(),
        name: "Custom DeepSeek Model".to_owned(),
        provider: ProviderId::from("custom-deepseek"),
        base_url: base_url.to_owned(),
        ..gpt4o_mini()
    }
}

#[tokio::test]
async fn sends_max_tokens_for_builtin_and_custom_deepseek_api_models() {
    let native = [
        builtin_model("deepseek", "deepseek-flash"),
        builtin_model("deepseek", "deepseek-v4-pro"),
    ];
    let custom = [
        deepseek_custom_model("https://api.deepseek.com", "custom-deepseek-model"),
        deepseek_custom_model(
            "https://API.DeepSeek.COM",
            "custom-uppercase-deepseek-model",
        ),
    ];
    for model in &native {
        assert_eq!(
            model
                .compat
                .as_ref()
                .and_then(|compat| compat.max_tokens_field),
            Some(pi_ai::types::MaxTokensField::MaxTokens)
        );
    }

    for model in native.iter().chain(custom.iter()) {
        let context = tool_context("Hi", None);
        let payload = capture_keyed_simple(model, &context, |options| {
            options.max_tokens = Some(123);
        })
        .await;

        assert_eq!(payload["max_tokens"], json!(123), "{}", model.id);
        assert!(
            payload.get("max_completion_tokens").is_none(),
            "{}",
            model.id
        );
    }
}

#[tokio::test]
async fn sends_max_tokens_for_zai_completions_models() {
    for id in ["glm-5-turbo", "glm-5.2"] {
        let model = builtin_model("zai", id);
        assert_eq!(
            model
                .compat
                .as_ref()
                .and_then(|compat| compat.max_tokens_field),
            Some(pi_ai::types::MaxTokensField::MaxTokens)
        );
        let context = tool_context("Hi", None);
        let payload = capture_keyed_simple(&model, &context, |options| {
            options.max_tokens = Some(123);
        })
        .await;

        assert_eq!(payload["max_tokens"], json!(123), "{id}");
        assert!(payload.get("max_completion_tokens").is_none(), "{id}");
    }
}

/// Porting restatement: upstream's compat dispatcher routes this model by its
/// catalog api; the completions suite pins the completions entry directly.
#[tokio::test]
async fn omits_reasoning_effort_for_opencode_grok_build() {
    let model = builtin_model("opencode", "grok-build-0.1");
    let context = tool_context("Hi", None);
    let payload = capture_keyed_simple(&model, &context, |options| {
        options.reasoning = Some(ThinkingLevel::High);
    })
    .await;

    assert!(payload.get("reasoning_effort").is_none());
}

#[tokio::test]
async fn does_not_double_count_reasoning_tokens_in_completion_usage() {
    let chunks = vec![json!({
        "id": "chatcmpl-reasoning-usage",
        "choices": [{ "delta": {}, "finish_reason": "stop" }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 33,
            "prompt_tokens_details": { "cached_tokens": 0 },
            "completion_tokens_details": { "reasoning_tokens": 21 },
        },
    })];
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &chunks);
    let model = gpt4o_mini();
    let context = tool_context("Use reasoning.", None);

    let result = stream_simple(&model, &context, Some(&keyed_simple(&mock)))
        .result()
        .await;

    assert_eq!(result.usage.input, 10);
    assert_eq!(result.usage.output, 33);
    assert_eq!(result.usage.total_tokens, 43);
}

fn cache_write_usage_chunk(id: &str, per_choice: bool) -> Vec<Value> {
    let usage = json!({
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "prompt_tokens_details": { "cached_tokens": 50, "cache_write_tokens": 30 },
        "completion_tokens_details": { "reasoning_tokens": 0 },
    });
    if per_choice {
        vec![
            json!({ "id": id, "choices": [{ "delta": { "content": "OK" }, "finish_reason": null }] }),
            json!({ "id": id, "choices": [{ "delta": {}, "finish_reason": "stop", "usage": usage }] }),
        ]
    } else {
        vec![
            json!({ "id": id, "choices": [{ "delta": { "content": "OK" }, "finish_reason": null }] }),
            json!({ "id": id, "choices": [{ "delta": {}, "finish_reason": "stop" }], "usage": usage }),
        ]
    }
}

async fn cache_write_result(chunks: Vec<Value>) -> pi_ai::types::AssistantMessage {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &chunks);
    let model = gpt4o_mini();
    let context = tool_context("Reply with exactly OK", None);
    stream_simple(&model, &context, Some(&keyed_simple(&mock)))
        .result()
        .await
}

/// `cached_tokens` is documented as cache reads; `cache_write_tokens` is
/// separate.
#[tokio::test]
async fn preserves_prompt_tokens_details_cache_fields_from_chunk_usage() {
    let result = cache_write_result(cache_write_usage_chunk("chatcmpl-cache-write", false)).await;

    assert_eq!(result.usage.input, 20);
    assert_eq!(result.usage.cache_read, 50);
    assert_eq!(result.usage.cache_write, 30);
    assert_eq!(result.usage.total_tokens, 105);
}

#[tokio::test]
async fn preserves_prompt_tokens_details_cache_fields_from_choice_usage_fallback() {
    let result =
        cache_write_result(cache_write_usage_chunk("chatcmpl-cache-write-choice", true)).await;

    assert_eq!(result.usage.input, 20);
    assert_eq!(result.usage.cache_read, 50);
    assert_eq!(result.usage.cache_write, 30);
    assert_eq!(result.usage.total_tokens, 105);
}

#[tokio::test]
async fn uses_openrouter_reasoning_object_instead_of_reasoning_effort() {
    let model = builtin_model("openrouter", "deepseek/deepseek-r1");
    let context = tool_context("Hi", None);
    let payload = capture_keyed_simple(&model, &context, |options| {
        options.reasoning = Some(ThinkingLevel::High);
    })
    .await;

    assert_eq!(payload["reasoning"], json!({ "effort": "high" }));
    assert!(payload.get("reasoning_effort").is_none());
}

#[tokio::test]
async fn uses_configurable_chat_template_boolean_thinking_kwargs() {
    let model = Model {
        id: "deepseek-ai/DeepSeek-V3.1".to_owned(),
        name: "DeepSeek V3.1 via vLLM".to_owned(),
        reasoning: true,
        compat: Some(compat_map(json!({
            "thinkingFormat": "chat-template",
            "supportsReasoningEffort": false,
            "chatTemplateKwargs": { "thinking": { "$var": "thinking.enabled" } },
        }))),
        ..gpt4o_mini()
    };

    for (reasoning, expected) in [
        (Some(ThinkingLevel::High), json!(true)),
        (None, json!(false)),
    ] {
        let context = tool_context("Hi", None);
        let payload = capture_keyed_simple(&model, &context, |options| {
            options.reasoning = reasoning;
        })
        .await;

        assert_eq!(
            payload["chat_template_kwargs"],
            json!({ "thinking": expected })
        );
        assert!(payload.get("thinking").is_none());
        assert!(payload.get("reasoning_effort").is_none());
    }
}

#[tokio::test]
async fn uses_qwen_chat_template_thinking_kwargs() {
    let model = Model {
        id: "Qwen/Qwen3-Coder".to_owned(),
        name: "Qwen3 Coder via vLLM".to_owned(),
        reasoning: true,
        compat: Some(compat_map(json!({
            "thinkingFormat": "qwen-chat-template",
            "supportsReasoningEffort": false,
        }))),
        ..gpt4o_mini()
    };

    for (reasoning, expected) in [
        (Some(ThinkingLevel::High), json!(true)),
        (None, json!(false)),
    ] {
        let context = tool_context("Hi", None);
        let payload = capture_keyed_simple(&model, &context, |options| {
            options.reasoning = reasoning;
        })
        .await;

        assert_eq!(
            payload["chat_template_kwargs"],
            json!({ "enable_thinking": expected, "preserve_thinking": true })
        );
        assert!(payload.get("reasoning_effort").is_none());
    }
}

#[tokio::test]
async fn uses_configurable_chat_template_effort_kwargs_with_static_kwargs() {
    let model = Model {
        id: "unsloth/gpt-oss-120b-GGUF".to_owned(),
        name: "GPT OSS via vLLM".to_owned(),
        reasoning: true,
        thinking_level_map: Some(
            std::iter::once((ModelThinkingLevel::Xhigh, Some("max".to_owned()))).collect(),
        ),
        compat: Some(compat_map(json!({
            "thinkingFormat": "chat-template",
            "supportsReasoningEffort": false,
            "chatTemplateKwargs": {
                "preserve_thinking": true,
                "reasoning_effort": { "$var": "thinking.effort", "omitWhenOff": true },
            },
        }))),
        ..gpt4o_mini()
    };
    let context = tool_context("Hi", None);
    let payload = capture_keyed_simple(&model, &context, |options| {
        options.reasoning = Some(ThinkingLevel::Xhigh);
    })
    .await;

    assert_eq!(
        payload["chat_template_kwargs"],
        json!({ "preserve_thinking": true, "reasoning_effort": "max" })
    );
    assert!(payload.get("reasoning_effort").is_none());
}

#[tokio::test]
async fn uses_ant_ling_compatibility_metadata() {
    let model = builtin_model("ant-ling", "Ring-2.6-1T");
    let compat = model.compat.as_ref().expect("catalog compat");
    assert_eq!(compat.supports_store, Some(false));
    assert_eq!(compat.supports_developer_role, Some(false));
    assert_eq!(compat.supports_reasoning_effort, Some(false));
    assert_eq!(
        compat.max_tokens_field,
        Some(pi_ai::types::MaxTokensField::MaxTokens)
    );
    assert_eq!(
        compat.thinking_format,
        Some(pi_ai::types::ThinkingFormat::AntLing)
    );
    assert_eq!(compat.supports_long_cache_retention, Some(false));
    assert_eq!(compat.supports_strict_mode, None);
    assert_eq!(
        compat.requires_reasoning_content_on_assistant_messages,
        None
    );

    let context = Context {
        system_prompt: Some("Follow instructions.".to_owned()),
        messages: vec![user_message_now("Hi")],
        tools: None,
    };
    let payload = capture_keyed_simple(&model, &context, |options| {
        options.max_tokens = Some(123);
        options.reasoning = Some(ThinkingLevel::High);
        options.cache_retention = Some(CacheRetention::Long);
        options.session_id = Some("ant-ling-session".to_owned());
    })
    .await;

    assert_eq!(payload["max_tokens"], json!(123));
    assert!(payload.get("max_completion_tokens").is_none());
    assert_eq!(payload["messages"][0]["role"], json!("system"));
    assert_eq!(payload["reasoning"], json!({ "effort": "high" }));
    assert!(payload.get("reasoning_effort").is_none());
    assert!(payload.get("store").is_none());
    assert!(payload.get("prompt_cache_key").is_none());
    assert!(payload.get("prompt_cache_retention").is_none());
}

/// Porting restatement: upstream drives Ring through the direct `stream`
/// entry so the unmapped `medium` effort reaches the wire unclamped — the
/// simple entry clamps it to `high`, a level Ring's map spells.
#[tokio::test]
async fn omits_ant_ling_reasoning_for_unmapped_direct_reasoning_efforts_and_non_reasoning_models() {
    let ring = builtin_model("ant-ling", "Ring-2.6-1T");
    let mut options = keyed(&MockHttpClient::new());
    options.reasoning_effort = Some(ThinkingLevel::Medium);

    let payload = capture(&ring, &tool_context("Hi", None), options).await;

    assert!(payload.get("reasoning").is_none());

    let ling = builtin_model("ant-ling", "Ling-2.6-flash");
    let payload = capture_keyed_simple(&ling, &tool_context("Hi", None), |options| {
        options.reasoning = Some(ThinkingLevel::High);
    })
    .await;

    assert!(payload.get("reasoning").is_none());
}

// ---------------------------------------------------------------------------
// tool result images (upstream openai-completions-tool-result-images)
// ---------------------------------------------------------------------------

fn image_read_result_turn(tool_call_id: &str) -> Message {
    common::tool_result_message(
        tool_call_id,
        vec![
            ToolResultBlock::Text(TextContent {
                text: "Read image file [image/png]".to_owned(),
                text_signature: None,
            }),
            ToolResultBlock::Image(pi_ai::types::ImageContent {
                data: "ZmFrZQ==".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ],
        None,
    )
}

fn image_input_model() -> Model {
    Model {
        input: vec![Modality::Text, Modality::Image],
        ..gpt4o_mini()
    }
}

#[tokio::test]
async fn batches_tool_result_images_after_consecutive_tool_results() {
    let model = image_input_model();
    let timestamp = pi_ai::auth::resolve::now_ms();
    let context = Context {
        messages: vec![
            Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("Read the images".to_owned()),
                timestamp: timestamp - 2,
            }),
            history_turn(
                "openai-completions",
                "openai",
                "gpt-4o-mini",
                vec![
                    AssistantBlock::ToolCall(ToolCall {
                        id: "tool-1".to_owned(),
                        name: "read".to_owned(),
                        arguments: serde_json::Map::from_iter([(
                            "path".to_owned(),
                            json!("img-1.png"),
                        )]),
                        thought_signature: None,
                        namespace: None,
                    }),
                    AssistantBlock::ToolCall(ToolCall {
                        id: "tool-2".to_owned(),
                        name: "read".to_owned(),
                        arguments: serde_json::Map::from_iter([(
                            "path".to_owned(),
                            json!("img-2.png"),
                        )]),
                        thought_signature: None,
                        namespace: None,
                    }),
                ],
            ),
            image_read_result_turn("tool-1"),
            image_read_result_turn("tool-2"),
        ],
        ..Context::default()
    };

    let payload = capture(&model, &context, keyed(&MockHttpClient::new())).await;

    let roles: Vec<&str> = payload["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .map(|message| message["role"].as_str().expect("role"))
        .collect();
    assert_eq!(roles, ["user", "assistant", "tool", "tool", "user"]);

    let image_message = payload["messages"]
        .as_array()
        .expect("messages")
        .last()
        .expect("the image turn");
    assert_eq!(image_message["role"], json!("user"));
    assert!(image_message["content"].is_array());
    let image_parts = image_message["content"]
        .as_array()
        .expect("content parts")
        .iter()
        .filter(|part| part["type"] == json!("image_url"))
        .count();
    assert_eq!(image_parts, 2);
}

#[tokio::test]
async fn uses_no_tool_output_placeholder_for_empty_tool_results_without_images() {
    let model = image_input_model();
    let timestamp = pi_ai::auth::resolve::now_ms();
    let context = Context {
        messages: vec![
            Message::User(pi_ai::types::UserMessage {
                content: pi_ai::types::UserContent::Text("Run the command".to_owned()),
                timestamp: timestamp - 1,
            }),
            history_turn(
                "openai-completions",
                "openai",
                "gpt-4o-mini",
                vec![AssistantBlock::ToolCall(ToolCall {
                    id: "tool-1".to_owned(),
                    name: "bash".to_owned(),
                    arguments: serde_json::Map::from_iter([("command".to_owned(), json!("true"))]),
                    thought_signature: None,
                    namespace: None,
                })],
            ),
            common::tool_result_message(
                "tool-1",
                vec![ToolResultBlock::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                })],
                None,
            ),
        ],
        ..Context::default()
    };

    let payload = capture(&model, &context, keyed(&MockHttpClient::new())).await;

    let tool_message = payload["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == json!("tool"))
        .expect("the tool turn");
    assert_eq!(tool_message["content"], json!("(no tool output)"));
}

// ---------------------------------------------------------------------------
// reasoning details (upstream openai-completions-reasoning-details)
// ---------------------------------------------------------------------------

const REASONING_DETAIL_JSON: &str =
    r#"{"type":"reasoning.encrypted","id":"call_1","data":"encrypted-signature"}"#;
const SIGNED_TEXT_DETAIL_JSON: &str = r#"{"type":"reasoning.text","text":"I should call the read tool.","signature":"sha256:signed-text","id":"reasoning-text-1","format":"anthropic-claude-v1","index":0}"#;
const SUMMARY_DETAIL_JSON: &str = r#"{"type":"reasoning.summary","summary":"Decided to inspect the requested file.","id":"reasoning-summary-1","format":"anthropic-claude-v1","index":1}"#;

fn reasoning_details_model() -> Model {
    Model {
        id: "google/gemini-test".to_owned(),
        name: "Gemini Test".to_owned(),
        base_url: "https://openrouter.ai/api/v1".to_owned(),
        reasoning: true,
        context_window: 100_000,
        max_tokens: 4096,
        ..common::openai_catalog_model("openrouter", "auto")
    }
}

fn details_chunk(delta: &Value, finish_reason: Option<&str>) -> Value {
    let mut choice = json!({ "index": 0, "delta": delta });
    if let Some(finish) = finish_reason {
        choice["finish_reason"] = json!(finish);
    }
    json!({
        "id": "chatcmpl-test",
        "model": "google/gemini-test",
        "choices": [choice],
    })
}

fn details_tool_call_chunk() -> Value {
    details_chunk(
        &json!({
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": { "name": "read", "arguments": "{\"path\":\"README.md\"}" },
            }],
        }),
        None,
    )
}

/// Mount a route that answers the first request with `first` and the second
/// with `second`, the mock-side shape of upstream's `chunkSets` queue.
fn mount_two_runs(mock: &MockHttpClient, first: &[Value], second: &[Value]) {
    mock.on(|request| request.url.contains("/chat/completions"))
        .respond_sequence(vec![
            common::openai_sse_response(first),
            common::openai_sse_response(second),
        ]);
}

/// The reasoning-details two-run fixture: one streamed encrypted detail, a
/// tool call closing it, then the second run answering with text — the shape
/// the signature-replay cases ride.
fn mounted_reasoning_two_runs() -> MockHttpClient {
    let reasoning_detail = encrypted_detail();
    let mock = MockHttpClient::new();
    mount_two_runs(
        &mock,
        &[
            details_chunk(&json!({ "reasoning_details": [reasoning_detail] }), None),
            details_tool_call_chunk(),
            details_chunk(&json!({}), Some("tool_calls")),
        ],
        &[
            details_chunk(&json!({ "content": "ok" }), None),
            details_chunk(&json!({}), Some("stop")),
        ],
    );
    mock
}

async fn run_details_stream(
    mock: &MockHttpClient,
    messages: Vec<Message>,
) -> pi_ai::types::AssistantMessage {
    let model = reasoning_details_model();
    let context = Context {
        messages,
        tools: Some(vec![read_tool()]),
        ..Context::default()
    };
    let options = OpenAiCompletionsOptions {
        api_key: Some("test".to_owned()),
        transport_options: common::mock_transport(mock),
        ..OpenAiCompletionsOptions::default()
    };
    stream(&model, &context, Some(&options)).result().await
}

fn detail_block_content(message: &pi_ai::types::AssistantMessage) -> (String, Option<String>) {
    for block in &message.content {
        if let AssistantBlock::Thinking(thinking) = block {
            return (
                thinking.thinking.clone(),
                thinking.thinking_signature.clone(),
            );
        }
    }
    panic!("expected a thinking block");
}

/// Replay the settled assistant message and assert the second run carries
/// the detail, the shape the reasoning-replay cases end with.
async fn replay_and_assert_details(
    mock: &MockHttpClient,
    first: pi_ai::types::AssistantMessage,
    reasoning_detail: Value,
) {
    let _second = run_details_stream(mock, vec![Message::Assistant(first)]).await;
    let payload = recorded_body_at(mock, 1);
    assert_eq!(
        assistant_payload_message(&payload)["reasoning_details"],
        json!([reasoning_detail])
    );
}

/// The assistant message a replayed payload's `messages` array carries.
fn assistant_payload_message(payload: &Value) -> Value {
    payload["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == json!("assistant"))
        .cloned()
        .expect("the replayed assistant")
}

fn encrypted_detail() -> Value {
    serde_json::from_str(REASONING_DETAIL_JSON).expect("the encrypted detail")
}

fn signed_text_detail() -> Value {
    serde_json::from_str(SIGNED_TEXT_DETAIL_JSON).expect("the signed text detail")
}

fn summary_detail() -> Value {
    serde_json::from_str(SUMMARY_DETAIL_JSON).expect("the summary detail")
}

#[tokio::test]
async fn preserves_reasoning_details_in_the_thinking_signature() {
    let reasoning_detail = encrypted_detail();
    let mock = mounted_reasoning_two_runs();

    let first = run_details_stream(&mock, Vec::new()).await;
    let (thinking, signature) = detail_block_content(&first);
    assert_eq!(thinking, "");
    assert_eq!(
        signature.as_deref(),
        Some(
            serde_json::to_string(&vec![reasoning_detail.clone()])
                .expect("serializes")
                .as_str()
        )
    );
    let AssistantBlock::ToolCall(tool_call) = &first.content[1] else {
        panic!("expected the tool call");
    };
    assert_eq!(tool_call.id, "call_1");
    assert_eq!(tool_call.name, "read");
    assert_eq!(
        tool_call.arguments,
        serde_json::Map::from_iter([("path".to_owned(), json!("README.md"))])
    );

    replay_and_assert_details(&mock, first, reasoning_detail).await;
}

/// Older stored assistant messages carry the encrypted detail on the tool
/// call's `thoughtSignature` instead of a thinking block.
#[tokio::test]
async fn falls_back_to_encrypted_tool_call_signatures_for_older_stored_messages() {
    let reasoning_detail = encrypted_detail();
    let mock = mounted_reasoning_two_runs();

    let mut first = run_details_stream(&mock, Vec::new()).await;
    first
        .content
        .retain(|block| !matches!(block, AssistantBlock::Thinking(_)));
    let signature = serde_json::to_string(&reasoning_detail).expect("serializes");
    for block in &mut first.content {
        if let AssistantBlock::ToolCall(tool_call) = block {
            tool_call.thought_signature = Some(signature.clone());
        }
    }

    replay_and_assert_details(&mock, first, reasoning_detail).await;
}

#[tokio::test]
async fn preserves_signed_text_and_summary_reasoning_details_in_their_original_sequence() {
    let reasoning_detail = encrypted_detail();
    let signed_text = signed_text_detail();
    let summary = summary_detail();
    let signed_text_text = signed_text["text"]
        .as_str()
        .expect("the detail text")
        .to_owned();
    let mock = MockHttpClient::new();
    mount_two_runs(
        &mock,
        &[
            details_chunk(
                &json!({
                    "reasoning": signed_text_text,
                    "reasoning_details": [signed_text.clone()],
                }),
                None,
            ),
            details_chunk(
                &json!({ "reasoning_details": [reasoning_detail.clone(), summary.clone()] }),
                None,
            ),
            details_tool_call_chunk(),
            details_chunk(&json!({}), Some("tool_calls")),
        ],
        &[
            details_chunk(&json!({ "content": "ok" }), None),
            details_chunk(&json!({}), Some("stop")),
        ],
    );

    let first = run_details_stream(&mock, Vec::new()).await;
    let expected_details = vec![
        signed_text.clone(),
        reasoning_detail.clone(),
        summary.clone(),
    ];
    let (thinking, signature) = detail_block_content(&first);
    assert_eq!(thinking, signed_text_text);
    assert_eq!(
        signature.as_deref(),
        Some(
            serde_json::to_string(&expected_details)
                .expect("serializes")
                .as_str()
        )
    );

    let _second = run_details_stream(&mock, vec![Message::Assistant(first)]).await;

    let payload = recorded_body_at(&mock, 1);
    let assistant = assistant_payload_message(&payload);
    assert_eq!(assistant["reasoning_details"], json!(expected_details));
    assert!(assistant.get("reasoning").is_none());
}

/// Consecutive text and summary deltas merge in place before replay; the
/// encrypted entry stays discrete.
#[tokio::test]
async fn merges_consecutive_text_and_summary_reasoning_details_deltas_before_replay() {
    let text_delta = json!({ "type": "reasoning.text", "text": "The", "index": 0 });
    let text_delta_with_signature = json!({
        "type": "reasoning.text",
        "text": " user wants the time.",
        "signature": "sha256:text-signature",
        "format": "openai-responses-v1",
        "index": 0,
    });
    let summary_delta = json!({ "type": "reasoning.summary", "summary": "Looked", "index": 0 });
    let summary_delta_with_format = json!({
        "type": "reasoning.summary",
        "summary": " up time.",
        "format": "openai-responses-v1",
        "index": 0,
    });
    let reasoning_detail = encrypted_detail();
    let later_summary_delta = json!({
        "type": "reasoning.summary",
        "summary": "After encrypted block.",
        "format": "openai-responses-v1",
        "index": 0,
    });
    let expected_details = json!([
        {
            "type": "reasoning.text",
            "text": "The user wants the time.",
            "index": 0,
            "signature": "sha256:text-signature",
            "format": "openai-responses-v1",
        },
        {
            "type": "reasoning.summary",
            "summary": "Looked up time.",
            "index": 0,
            "format": "openai-responses-v1",
        },
        reasoning_detail,
        later_summary_delta,
    ]);
    let mock = MockHttpClient::new();
    mount_two_runs(
        &mock,
        &[
            details_chunk(&json!({ "reasoning_details": [text_delta] }), None),
            details_chunk(
                &json!({ "reasoning_details": [text_delta_with_signature] }),
                None,
            ),
            details_chunk(&json!({ "reasoning_details": [summary_delta] }), None),
            details_chunk(
                &json!({ "reasoning_details": [summary_delta_with_format] }),
                None,
            ),
            details_chunk(&json!({ "reasoning_details": [encrypted_detail()] }), None),
            details_chunk(&json!({ "reasoning_details": [later_summary_delta] }), None),
            details_tool_call_chunk(),
            details_chunk(&json!({}), Some("tool_calls")),
        ],
        &[
            details_chunk(&json!({ "content": "ok" }), None),
            details_chunk(&json!({}), Some("stop")),
        ],
    );

    let first = run_details_stream(&mock, Vec::new()).await;
    let (thinking, signature) = detail_block_content(&first);
    assert_eq!(thinking, "");
    assert_eq!(
        signature.as_deref(),
        Some(expected_details.to_string().as_str())
    );

    let _second = run_details_stream(&mock, vec![Message::Assistant(first)]).await;

    let payload = recorded_body_at(&mock, 1);
    assert_eq!(
        assistant_payload_message(&payload)["reasoning_details"],
        expected_details
    );
}

// ---------------------------------------------------------------------------
// raw stop reasons (upstream openai-completions-raw-stop-reason)
// ---------------------------------------------------------------------------

fn raw_stop_model() -> Model {
    Model {
        id: "test-model".to_owned(),
        name: "Test Model".to_owned(),
        base_url: "https://api.openai.com/v1".to_owned(),
        reasoning: false,
        context_window: 128_000,
        max_tokens: 4096,
        ..common::openai_catalog_model("openai", "gpt-4o-mini")
    }
}

async fn raw_stop_result(finish_reason: &str, id: &str) -> pi_ai::types::AssistantMessage {
    let mock = MockHttpClient::new();
    openai_mock_with(
        &mock,
        &[json!({
            "id": id,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason }],
        })],
    );
    let model = raw_stop_model();
    let context = Context {
        messages: vec![user_message_now("hello")],
        ..Context::default()
    };
    let options = OpenAiCompletionsOptions {
        api_key: Some("test".to_owned()),
        transport_options: common::mock_transport(&mock),
        ..OpenAiCompletionsOptions::default()
    };
    stream(&model, &context, Some(&options)).result().await
}

#[tokio::test]
async fn preserves_raw_finish_reasons_for_successful_stops() {
    let message = raw_stop_result("stop", "chatcmpl-1").await;

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("stop"));
    assert_eq!(message.error_message, None);
}

#[tokio::test]
async fn preserves_raw_finish_reasons_for_provider_error_stops() {
    let message = raw_stop_result("content_filter", "chatcmpl-2").await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("content_filter"));
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider finish_reason: content_filter")
    );
}

// ---------------------------------------------------------------------------
// empty tools handling (upstream openai-completions-empty-tools)
// ---------------------------------------------------------------------------

/// The empty-tools result for the given context and options: the request's
/// payload after the stream settles.
async fn empty_tools_payload(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
) -> Value {
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let mut options = options.clone();
    options.transport_options = common::mock_transport(&mock);
    let _ = stream_simple(model, context, Some(&options)).result().await;
    recorded_body(&mock)
}

fn empty_tools_options() -> SimpleStreamOptions {
    SimpleStreamOptions {
        api_key: Some("test".to_owned()),
        ..SimpleStreamOptions::default()
    }
}

/// Porting restatement: the Cloudflare auth resolution replaces upstream's
/// process-env resolution — the resolved headers and env ride the request
/// options the same way the provider factory threads them.
async fn resolved_cloudflare_gateway(model_id: &str) -> (Model, pi_ai::types::ProviderHeaders) {
    let auth = pi_ai::providers::cloudflare_auth::cloudflare_ai_gateway_auth();
    let env: pi_ai::types::ProviderEnv = [
        ("CLOUDFLARE_API_KEY", "cf-token"),
        ("CLOUDFLARE_ACCOUNT_ID", "account-id"),
        ("CLOUDFLARE_GATEWAY_ID", "gateway-id"),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value.to_owned()))
    .collect();
    let input = pi_ai::auth::types::ApiKeyAuthInput {
        ctx: Arc::new(EnvContext(env.clone())),
        credential: None,
        signal: tokio_util::sync::CancellationToken::new(),
    };
    let resolved = (auth.resolve)(input)
        .await
        .expect("resolve")
        .expect("configured");
    let model = pi_ai::providers::cloudflare_stream::resolve_cloudflare_model(
        &builtin_model("cloudflare-ai-gateway", model_id),
        Some(&env),
    );
    (model, resolved.auth.headers.expect("gateway headers"))
}

/// The env fixture the Cloudflare resolution drives.
struct EnvContext(pi_ai::types::ProviderEnv);

impl pi_ai::auth::types::AuthContext for EnvContext {
    fn env(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }

    fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

#[tokio::test]
async fn omits_tools_field_when_context_tools_is_an_empty_array() {
    let model = gpt4o_mini();
    let context = Context {
        messages: vec![user_message_now("hi")],
        tools: Some(Vec::new()),
        ..Context::default()
    };

    let payload = empty_tools_payload(&model, &context, &empty_tools_options()).await;

    assert!(payload.get("tools").is_none());
}

#[tokio::test]
async fn omits_tools_field_when_context_tools_is_undefined() {
    let model = gpt4o_mini();
    let context = Context {
        messages: vec![user_message_now("hi")],
        tools: None,
        ..Context::default()
    };

    let payload = empty_tools_payload(&model, &context, &empty_tools_options()).await;

    assert!(payload.get("tools").is_none());
}

#[tokio::test]
async fn sends_default_max_tokens() {
    let model = gpt4o_mini();
    let context = Context {
        messages: vec![user_message_now("hi")],
        ..Context::default()
    };

    let payload = empty_tools_payload(&model, &context, &empty_tools_options()).await;

    assert!(payload.get("max_tokens").is_none());
    assert_eq!(payload["max_completion_tokens"], json!(model.max_tokens));
}

#[tokio::test]
async fn sends_explicit_max_tokens() {
    let model = gpt4o_mini();
    let context = Context {
        messages: vec![user_message_now("hi")],
        ..Context::default()
    };
    let options = SimpleStreamOptions {
        api_key: Some("test".to_owned()),
        max_tokens: Some(1234),
        ..SimpleStreamOptions::default()
    };

    let payload = empty_tools_payload(&model, &context, &options).await;

    assert!(payload.get("max_tokens").is_none());
    assert_eq!(payload["max_completion_tokens"], json!(1234));
}

#[tokio::test]
async fn clamps_default_max_tokens_to_remaining_context() {
    let model = Model {
        context_window: 10_000,
        max_tokens: 8000,
        ..gpt4o_mini()
    };
    let context = Context {
        messages: vec![user_message_now("x".repeat(8000).as_str())],
        ..Context::default()
    };

    let payload = empty_tools_payload(&model, &context, &empty_tools_options()).await;

    assert!(payload.get("max_tokens").is_none());
    assert_eq!(payload["max_completion_tokens"], json!(3904));
}

#[tokio::test]
async fn clamps_explicit_max_tokens_to_remaining_context() {
    let model = Model {
        context_window: 10_000,
        max_tokens: 8000,
        ..gpt4o_mini()
    };
    let context = Context {
        messages: vec![user_message_now("x".repeat(8000).as_str())],
        ..Context::default()
    };
    let options = SimpleStreamOptions {
        api_key: Some("test".to_owned()),
        max_tokens: Some(7000),
        ..SimpleStreamOptions::default()
    };

    let payload = empty_tools_payload(&model, &context, &options).await;

    assert!(payload.get("max_tokens").is_none());
    assert_eq!(payload["max_completion_tokens"], json!(3904));
}

#[tokio::test]
async fn uses_conservative_fields_for_cloudflare_ai_gateway_compat_models() {
    let (model, headers) = resolved_cloudflare_gateway("workers-ai/@cf/moonshotai/kimi-k2.6").await;
    let context = Context {
        system_prompt: Some("You are helpful.".to_owned()),
        messages: vec![user_message_now("hi")],
        ..Context::default()
    };
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let options = SimpleStreamOptions {
        api_key: Some("unused".to_owned()),
        headers: Some(headers),
        max_tokens: Some(1234),
        reasoning: Some(ThinkingLevel::High),
        transport_options: common::mock_transport(&mock),
        ..SimpleStreamOptions::default()
    };

    let _ = stream_simple(&model, &context, Some(&options))
        .result()
        .await;
    let payload = recorded_body(&mock);

    assert_eq!(payload["messages"][0]["role"], json!("system"));
    assert_eq!(payload["max_tokens"], json!(1234));
    assert!(payload.get("max_completion_tokens").is_none());
    assert!(payload.get("reasoning_effort").is_none());
    assert!(payload.get("store").is_none());

    // The gateway key rides cf-aig-authorization. Port note: upstream's SDK
    // suppresses its own `Authorization` default when the caller headers set
    // it to null; the port's seam dispatch prepends the dummy
    // `Authorization: Bearer unused` before the caller headers merge, so the
    // wire still carries that dummy pair.
    assert_header(&mock, "cf-aig-authorization", "Bearer cf-token");
}

#[tokio::test]
async fn resolves_cloudflare_ai_gateway_base_url_through_provider_auth() {
    let (model, headers) = resolved_cloudflare_gateway("workers-ai/@cf/moonshotai/kimi-k2.6").await;
    let context = Context {
        messages: vec![user_message_now("hi")],
        ..Context::default()
    };
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let options = SimpleStreamOptions {
        api_key: Some("unused".to_owned()),
        headers: Some(headers),
        transport_options: common::mock_transport(&mock),
        ..SimpleStreamOptions::default()
    };

    let _ = stream_simple(&model, &context, Some(&options))
        .result()
        .await;

    assert_eq!(
        mock.recorded()[0].url,
        format!("{}/chat/completions", model.base_url)
    );
}

#[tokio::test]
async fn preserves_inline_upstream_authorization_for_cloudflare_byok_requests() {
    let (model, mut headers) = resolved_cloudflare_gateway("gpt-5.1").await;
    headers.insert(
        "Authorization".to_owned(),
        Some("Bearer upstream-token".to_owned()),
    );
    let context = Context {
        messages: vec![user_message_now("hi")],
        ..Context::default()
    };
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let options = SimpleStreamOptions {
        api_key: Some("unused".to_owned()),
        headers: Some(headers),
        transport_options: common::mock_transport(&mock),
        ..SimpleStreamOptions::default()
    };

    let _ = stream_simple(&model, &context, Some(&options))
        .result()
        .await;

    // Port note: upstream's SDK defaultHeaders override its `Authorization`
    // default; the port's seam prepends the dummy bearer first, so the
    // caller's value rides as the second `Authorization` pair.
    let recorded = mock.recorded();
    let authorizations: Vec<&str> = recorded[0]
        .headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("Authorization"))
        .map(|(_, value)| value.as_str())
        .collect();
    assert!(
        authorizations.contains(&"Bearer upstream-token"),
        "got: {authorizations:?}"
    );
    assert_header(&mock, "cf-aig-authorization", "Bearer cf-token");
}

#[tokio::test]
async fn sends_session_affinity_headers_for_workers_ai_through_cloudflare_ai_gateway() {
    let (model, headers) = resolved_cloudflare_gateway("workers-ai/@cf/moonshotai/kimi-k2.6").await;
    let context = Context {
        messages: vec![user_message_now("hi")],
        ..Context::default()
    };
    let mock = MockHttpClient::new();
    openai_mock_with(&mock, &[openai_done_chunk()]);
    let options = SimpleStreamOptions {
        api_key: Some("unused".to_owned()),
        headers: Some(headers),
        session_id: Some("session-1".to_owned()),
        transport_options: common::mock_transport(&mock),
        ..SimpleStreamOptions::default()
    };

    let _ = stream_simple(&model, &context, Some(&options))
        .result()
        .await;

    assert_header(&mock, "session_id", "session-1");
    assert_header(&mock, "x-client-request-id", "session-1");
    assert_header(&mock, "x-session-affinity", "session-1");
}

#[tokio::test]
async fn still_emits_empty_tools_for_anthropic_litellm_proxy_when_history_has_tools() {
    let model = gpt4o_mini();
    let context = Context {
        messages: vec![
            user_message_now("use the tool"),
            history_turn(
                "openai-completions",
                "openai",
                "gpt-4o-mini",
                vec![AssistantBlock::ToolCall(ToolCall {
                    id: "t1".to_owned(),
                    name: "noop".to_owned(),
                    arguments: serde_json::Map::new(),
                    thought_signature: None,
                    namespace: None,
                })],
            ),
            common::tool_result_message(
                "t1",
                vec![ToolResultBlock::Text(TextContent {
                    text: "done".to_owned(),
                    text_signature: None,
                })],
                None,
            ),
        ],
        tools: Some(Vec::new()),
        ..Context::default()
    };
    let options = SimpleStreamOptions {
        api_key: Some("test".to_owned()),
        ..SimpleStreamOptions::default()
    };

    let payload = empty_tools_payload(&model, &context, &options).await;

    assert!(payload["tools"].is_array());
    assert_eq!(payload["tools"], json!([]));
}

// ---------------------------------------------------------------------------
// vllm priority (upstream openai-completions-vllm-priority)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sends_compat_vllm_priority_as_the_top_level_priority_request_field() {
    let model = Model {
        compat: Some(compat_map(json!({ "vllmPriority": 10 }))),
        ..gpt4o_mini()
    };
    let context = Context {
        system_prompt: Some("sys".to_owned()),
        messages: vec![user_message_now("hi")],
        ..Context::default()
    };

    let payload = capture(&model, &context, keyed(&MockHttpClient::new())).await;

    assert_eq!(payload["priority"], json!(10));
}

#[tokio::test]
async fn omits_priority_when_vllm_priority_is_not_set() {
    let context = Context {
        system_prompt: Some("sys".to_owned()),
        messages: vec![user_message_now("hi")],
        ..Context::default()
    };

    let payload = capture(&gpt4o_mini(), &context, keyed(&MockHttpClient::new())).await;

    assert!(payload.get("priority").is_none());
}

// ---------------------------------------------------------------------------
// error body passthrough (upstream provider-error-body-regression, the
// openai-completions tier)
// ---------------------------------------------------------------------------

fn error_body_context() -> Context {
    Context {
        system_prompt: Some(String::new()),
        messages: vec![Message::User(pi_ai::types::UserMessage {
            content: pi_ai::types::UserContent::Blocks(vec![pi_ai::types::UserBlock::Text(
                TextContent {
                    text: "hi".to_owned(),
                    text_signature: None,
                },
            )]),
            timestamp: 0,
        })],
        tools: Some(Vec::new()),
    }
}

fn error_body_model() -> Model {
    Model {
        id: "test-model".to_owned(),
        name: "Test Model".to_owned(),
        base_url: "https://openrouter.ai/api/v1".to_owned(),
        reasoning: false,
        context_window: 1000,
        max_tokens: 100,
        ..common::openai_catalog_model("openrouter", "auto")
    }
}

async fn error_body_result(body: Value) -> pi_ai::types::AssistantMessage {
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/chat/completions"))
        .respond(pi_ai::http::json_response(403, &body));
    let model = error_body_model();
    let options = OpenAiCompletionsOptions {
        api_key: Some("test".to_owned()),
        transport_options: common::mock_transport(&mock),
        ..OpenAiCompletionsOptions::default()
    };
    stream(&model, &error_body_context(), Some(&options))
        .result()
        .await
}

/// The body-blind text provider surfaces status + body.
#[tokio::test]
async fn openai_completions_surfaces_status_and_body() {
    let output = error_body_result(json!({ "error": "blocked by gateway WAF" })).await;

    assert_eq!(output.stop_reason, StopReason::Error);
    let message = output.error_message.expect("the failure message");
    assert!(message.contains("403"), "got: {message}");
    assert!(message.contains("blocked by gateway WAF"), "got: {message}");
    assert_ne!(message, "403 status code (no body)");
}

/// OpenRouter returns the extra reason under `metadata.raw`, which the
/// parsed body already surfaces; the manual append must not duplicate it.
#[tokio::test]
async fn openai_completions_does_not_double_print_the_openrouter_metadata_raw_extra() {
    let output = error_body_result(json!({
        "message": "Provider returned error",
        "code": 403,
        "metadata": { "raw": "upstream WAF blocked policy XYZ" },
    }))
    .await;

    let message = output.error_message.expect("the failure message");
    assert!(
        message.contains("upstream WAF blocked policy XYZ"),
        "got: {message}"
    );
    let occurrences = message.matches("upstream WAF blocked policy XYZ").count();
    assert_eq!(occurrences, 1);
}

// ---------------------------------------------------------------------------
// sampling options (upstream sampling-options, the openai-completions tier)
// ---------------------------------------------------------------------------

fn sampling_model() -> Model {
    Model {
        id: "custom-model".to_owned(),
        name: "Custom Model".to_owned(),
        provider: ProviderId::from("custom-provider"),
        base_url: "http://127.0.0.1:9/v1".to_owned(),
        reasoning: false,
        context_window: 128_000,
        max_tokens: 16_384,
        ..gpt4o_mini()
    }
}

fn sampling_options(params: Option<serde_json::Map<String, Value>>) -> SimpleStreamOptions {
    SimpleStreamOptions {
        api_key: Some("fake-key".to_owned()),
        sampling_params: params.map(std::collections::BTreeMap::from_iter),
        ..SimpleStreamOptions::default()
    }
}

async fn capture_sampling(model: &Model, options: SimpleStreamOptions) -> Value {
    let context = Context {
        messages: vec![user_message_now("Hello")],
        ..Context::default()
    };
    capture_simple(model, &context, options).await
}

#[tokio::test]
async fn merges_stream_option_sampling_params_into_the_request_body() {
    let options = sampling_options(Some(
        [
            ("top_p".to_owned(), json!(0.95)),
            ("top_k".to_owned(), json!(0)),
            ("min_p".to_owned(), json!(0)),
        ]
        .into_iter()
        .collect(),
    ));

    let payload = capture_sampling(&sampling_model(), options).await;

    assert_eq!(payload["top_p"], json!(0.95));
    assert_eq!(payload["top_k"], json!(0));
    assert_eq!(payload["min_p"], json!(0));
}

#[tokio::test]
async fn omits_sampling_params_when_neither_options_nor_model_set_them() {
    let payload = capture_sampling(&sampling_model(), sampling_options(None)).await;

    assert!(payload.get("temperature").is_none());
    assert!(payload.get("top_p").is_none());
}

#[tokio::test]
async fn applies_model_level_sampling_params() {
    let model = Model {
        sampling_params: Some(
            [
                ("temperature".to_owned(), json!(1.0)),
                ("top_p".to_owned(), json!(0.95)),
            ]
            .into_iter()
            .collect(),
        ),
        ..sampling_model()
    };

    let payload = capture_sampling(&model, sampling_options(None)).await;

    assert_eq!(payload["temperature"], json!(1.0));
    assert_eq!(payload["top_p"], json!(0.95));
}

#[tokio::test]
async fn merges_stream_option_keys_over_model_level_keys() {
    let model = Model {
        sampling_params: Some(
            [
                ("top_p".to_owned(), json!(0.95)),
                ("min_p".to_owned(), json!(0.05)),
            ]
            .into_iter()
            .collect(),
        ),
        ..sampling_model()
    };
    let options = sampling_options(Some(
        std::iter::once(("top_p".to_owned(), json!(0.5))).collect(),
    ));

    let payload = capture_sampling(&model, options).await;

    assert_eq!(payload["top_p"], json!(0.5));
    assert_eq!(payload["min_p"], json!(0.05));
}

#[tokio::test]
async fn stream_option_sampling_params_override_named_request_fields() {
    let options = SimpleStreamOptions {
        api_key: Some("fake-key".to_owned()),
        temperature: Some(0.0),
        sampling_params: Some(std::iter::once(("temperature".to_owned(), json!(1.0))).collect()),
        ..SimpleStreamOptions::default()
    };

    let payload = capture_sampling(&sampling_model(), options).await;

    assert_eq!(payload["temperature"], json!(1.0));
}

// ---------------------------------------------------------------------------
// openrouter reasoning options (upstream openrouter-reasoning-options.test.ts)
// ---------------------------------------------------------------------------

/// The openrouter-wired reasoning model the upstream file drives, the
/// `{ ...model, compat: { thinkingFormat: "openrouter" } }` fixture over
/// `streamSimple`.
fn openrouter_model(thinking_level_map: Option<pi_ai::types::ThinkingLevelMap>) -> Model {
    Model {
        id: "stealth/ox-alpha".to_owned(),
        name: "Ox Alpha".to_owned(),
        api: pi_ai::types::Api::from("openai-completions"),
        provider: ProviderId::from("openrouter"),
        base_url: "https://example.invalid/v1".to_owned(),
        reasoning: true,
        thinking_level_map,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 128_000,
        max_tokens: 4096,
        compat: Some(compat_map(json!({ "thinkingFormat": "openrouter" }))),
        ..gpt4o_mini()
    }
}

/// The level map upstream's `getOpenRouterThinkingLevelMap` builds for a
/// mandatory reasoning model over the supported efforts: `off` carries the
/// null marker, only the supported efforts name their effort.
fn mandatory_level_map(
    entries: &[(ModelThinkingLevel, Option<&str>)],
) -> pi_ai::types::ThinkingLevelMap {
    entries
        .iter()
        .map(|(level, effort)| (*level, effort.map(str::to_owned)))
        .collect()
}

/// Capture the request payload the openrouter model sends on the simple
/// entry, upstream's `capturePayload` through the seam mock.
async fn capture_openrouter_payload(model: &Model, reasoning: Option<ThinkingLevel>) -> Value {
    let context = Context {
        messages: vec![user_message_now("Hello")],
        ..Context::default()
    };
    capture_keyed_simple(model, &context, |options| {
        options.reasoning = reasoning;
    })
    .await
}

/// A mandatory-reasoning model omits `reasoning` from a background call
/// that does not request it: the level map's null `off` marker drops the
/// object instead of spelling a disable.
#[tokio::test]
async fn omits_reasoning_when_a_background_call_does_not_request_it() {
    let payload = capture_openrouter_payload(
        &openrouter_model(Some(mandatory_level_map(&[
            (ModelThinkingLevel::Off, None),
            (ModelThinkingLevel::Minimal, None),
            (ModelThinkingLevel::Low, Some("low")),
            (ModelThinkingLevel::Medium, None),
            (ModelThinkingLevel::High, Some("high")),
            (ModelThinkingLevel::Xhigh, None),
            (ModelThinkingLevel::Max, Some("max")),
        ]))),
        None,
    )
    .await;

    assert!(payload.get("reasoning").is_none());
}

/// A mandatory-reasoning model still carries an explicitly selected
/// supported effort through the nested reasoning object.
#[tokio::test]
async fn still_sends_an_explicitly_selected_supported_effort() {
    let payload = capture_openrouter_payload(
        &openrouter_model(Some(mandatory_level_map(&[
            (ModelThinkingLevel::Off, None),
            (ModelThinkingLevel::Low, Some("low")),
            (ModelThinkingLevel::High, Some("high")),
            (ModelThinkingLevel::Max, Some("max")),
        ]))),
        Some(ThinkingLevel::Low),
    )
    .await;

    assert_eq!(payload["reasoning"], json!({ "effort": "low" }));
}

/// An optional model without effort controls keeps the explicit disable:
/// the missing map spells `reasoning.effort: "none"`.
#[tokio::test]
async fn continues_to_explicitly_disable_reasoning_for_optional_models() {
    let payload = capture_openrouter_payload(&openrouter_model(None), None).await;

    assert_eq!(payload["reasoning"], json!({ "effort": "none" }));
}

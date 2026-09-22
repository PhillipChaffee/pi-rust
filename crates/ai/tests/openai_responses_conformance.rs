//! OpenAI Responses conformance suites, ported from the upstream
//! `openai-responses-*` test files at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, grouped per upstream file:
//! compat, empty tool result, foreign tool-call id, message id, tool-call
//! namespaces, partial-json cleanup, terminal event, tool result images, and
//! the openai-responses tier of provider-error-body-regression.
//!
//! Porting restatements: upstream captures request payloads through the
//! mocked SDK's `create(params)` and `onPayload` hooks and stubbed
//! `globalThis.fetch` calls; the port reads the same payload and headers off
//! the [`MockHttpClient`] seam's recorded requests. The direct
//! `processResponsesStream` calls ride a canned [`HttpResponse`] whose body
//! frames the same events, and upstream's push spy becomes a drain of the
//! processor's event stream.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the tests pin stream outcomes; an unexpected shape panics the test by design"
)]
#![expect(
    clippy::float_cmp,
    reason = "the service-tier cost assertions pin exact upstream arithmetic outcomes"
)]

use std::collections::BTreeSet;

use bytes::Bytes;
use pi_ai::api::azure_openai_responses::AzureOpenAiResponsesOptions;
use pi_ai::api::openai_responses::{self, OpenAiResponsesOptions, ReasoningSummary};
use pi_ai::api::openai_responses_shared::{
    ConvertResponsesMessagesOptions, OpenAiResponsesStreamOptions, convert_responses_messages,
    process_responses_stream,
};
use pi_ai::http::{HttpByteStream, HttpResponse, MockHttpClient};
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, AssistantMessageEvent, CacheRetention, Context, Message,
    Modality, Model, ModelCompat, ProviderId, StopReason, TextContent, ThinkingContent,
    ThinkingLevel, Tool, ToolCall, ToolResultBlock, ToolResultMessage, TransportOptions, UserBlock,
    UserContent, UserMessage,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde_json::{Value, json};

mod common;
use common::{
    builtin_model, openai_responses_completed_event, openai_responses_mock_with, recorded_body,
    recorded_body_at, recorded_header, user_message_now,
};

use pi_ai::utils::hash::short_hash;

// ---------------------------------------------------------------------------
// Shared capture helpers
// ---------------------------------------------------------------------------

/// Stream the context and take the request's payload; the mock is mounted
/// fresh and shared by the options, upstream's `onPayload` capture.
async fn capture(model: &Model, context: &Context, mut options: OpenAiResponsesOptions) -> Value {
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    options.transport_options = common::mock_transport(&mock);
    let stream = openai_responses::stream(model, context, Some(&options));
    let _ = common::drain_and_settle(&stream).await;
    recorded_body(&mock)
}

/// Stream the session/retention setup against the sys-prompt context and
/// return the mock for header assertions, upstream's
/// `captureOpenAIResponseHeaders`.
async fn capture_headers(model: &Model, mut options: OpenAiResponsesOptions) -> MockHttpClient {
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    options.transport_options = common::mock_transport(&mock);
    let context = Context {
        system_prompt: Some("sys".to_owned()),
        messages: vec![user_message_now("hi")],
        ..Context::default()
    };
    let stream = openai_responses::stream(model, &context, Some(&options));
    let _ = common::drain_and_settle(&stream).await;
    mock
}

/// The keyed options against the given mock.
fn keyed(mock: &MockHttpClient) -> OpenAiResponsesOptions {
    common::keyed_openai_responses_options(mock)
}

/// A compat map from the upstream test's JSON spelling.
fn compat_map(compat: Value) -> ModelCompat {
    serde_json::from_value::<ModelCompat>(compat).expect("compat map")
}

/// The openai gpt-5.4 catalog entry, upstream's `getModel("openai",
/// "gpt-5.4")` default.
fn gpt54() -> Model {
    builtin_model("openai", "gpt-5.4")
}

/// A copy of the model retargeted at another provider and base URL, the
/// `{ ...getModel(...), provider, baseUrl }` spread the upstream suites run.
fn retargeted(model: &Model, provider: &str, base_url: &str) -> Model {
    Model {
        provider: ProviderId::from(provider),
        base_url: base_url.to_owned(),
        ..model.clone()
    }
}

/// A copy of the model with a compat override, the
/// `{ ...getModel(...), compat }` spread.
fn with_compat(model: &Model, compat: Value) -> Model {
    Model {
        compat: Some(compat_map(compat)),
        ..model.clone()
    }
}

/// The zeroed usage block the replay fixtures carry.
fn empty_usage() -> pi_ai::types::Usage {
    pi_ai::types::Usage::default()
}

/// A tool with the given JSON-schema parameters, upstream's typebox schemas.
fn tool(name: &str, parameters: Value) -> Tool {
    Tool {
        name: name.to_owned(),
        description: format!("{name} tool."),
        parameters,
        constrained_sampling: None,
    }
}

// ---------------------------------------------------------------------------
// compat (upstream openai-responses-compat)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn omits_reasoning_when_no_reasoning_is_requested() {
    let model = builtin_model("github-copilot", "gpt-5-mini");
    let payload = capture(&model, &sys_hi_context(), keyed(&MockHttpClient::new())).await;

    assert!(payload.get("reasoning").is_none());
}

#[tokio::test]
async fn forwards_required_tool_choice() {
    let model = gpt54();
    let context = Context {
        messages: vec![user_message_now(
            "Do not call ping. Respond with text instead.",
        )],
        tools: Some(vec![tool(
            "ping",
            json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"],
            }),
        )]),
        ..Context::default()
    };
    let mut options = keyed(&MockHttpClient::new());
    options.tool_choice = Some(json!("required"));

    let payload = capture(&model, &context, options).await;

    assert_eq!(payload["tool_choice"], json!("required"));
    assert_eq!(payload["tools"][0]["name"], json!("ping"));
}

/// The `getModel("cloudflare-ai-gateway", "gpt-5.6-sol")` entry the strict
/// mode test pins.
fn gpt56_sol_cloudflare() -> Model {
    builtin_model("cloudflare-ai-gateway", "gpt-5.6-sol")
}

#[tokio::test]
async fn sets_strict_mode_explicitly_for_cloudflare_openai_responses_tools() {
    let model = gpt56_sol_cloudflare();
    assert_eq!(
        model
            .compat
            .as_ref()
            .and_then(|compat| compat.supports_strict_mode),
        Some(true)
    );
    let context = Context {
        messages: vec![user_message_now("Use a tool.")],
        tools: Some(vec![
            tool(
                "ordinary",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "offset": { "type": "number" },
                    },
                    "required": ["path"],
                }),
            ),
            Tool {
                name: "constrained".to_owned(),
                description: "A constrained tool".to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"],
                }),
                constrained_sampling: Some(pi_ai::types::ConstrainedSamplingSetting::Config(
                    pi_ai::types::ConstrainedSamplingConfig::JsonSchema {
                        strict: pi_ai::types::Strictness::Prefer,
                    },
                )),
            },
        ]),
        ..Context::default()
    };

    let payload = capture(&model, &context, keyed(&MockHttpClient::new())).await;

    let tools = payload["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["name"], json!("ordinary"));
    assert_eq!(tools[0]["strict"], json!(false));
    assert_eq!(tools[1]["name"], json!("constrained"));
    assert_eq!(tools[1]["strict"], json!(true));
}

/// The catalog ids that send `reasoning.effort: "none"` when no reasoning is
/// requested, upstream's `sends none reasoning effort` `it.each`.
const NONE_EFFORT_MODEL_IDS: [&str; 10] = [
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
];

#[tokio::test]
async fn sends_none_reasoning_effort_when_no_reasoning_is_requested() {
    for model_id in NONE_EFFORT_MODEL_IDS {
        let model = builtin_model("openai", model_id);
        let payload = capture(&model, &sys_hi_context(), keyed(&MockHttpClient::new())).await;

        assert_eq!(
            payload["reasoning"],
            json!({ "effort": "none" }),
            "{model_id}"
        );
    }
}

/// The catalog ids whose off state is unsupported, so no reasoning field is
/// sent at all, upstream's `omits reasoning effort` `it.each`.
const OFF_UNSUPPORTED_MODEL_IDS: [&str; 7] = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    "gpt-5.2-pro",
    "gpt-5.4-pro",
    "gpt-5.5-pro",
];

#[tokio::test]
async fn omits_reasoning_effort_when_off_is_unsupported() {
    for model_id in OFF_UNSUPPORTED_MODEL_IDS {
        let model = builtin_model("openai", model_id);
        let payload = capture(&model, &sys_hi_context(), keyed(&MockHttpClient::new())).await;

        assert!(payload.get("reasoning").is_none(), "{model_id}");
    }
}

/// The three session-affinity headers, upstream's `CapturedHeaders` reads.
type AffinityHeaders = (Option<String>, Option<String>, Option<String>);

fn affinity_headers(mock: &MockHttpClient) -> AffinityHeaders {
    (
        recorded_header(mock, "session_id"),
        recorded_header(mock, "x-client-request-id"),
        recorded_header(mock, "x-session-id"),
    )
}

/// Stream the affinity setup and return its headers plus payload, upstream's
/// `captureOpenAIResponseHeaders` with the tests' onPayload capture folded
/// in.
async fn capture_affinity(
    model: &Model,
    options: OpenAiResponsesOptions,
) -> (AffinityHeaders, Value) {
    let mock = capture_headers(model, options).await;
    let payload = recorded_body(&mock);
    (affinity_headers(&mock), payload)
}

#[tokio::test]
async fn sets_cache_affinity_headers_for_official_requests_with_a_session_id() {
    let mut options = keyed(&MockHttpClient::new());
    options.session_id = Some("session-123".to_owned());

    let (session_id, client_request_id, x_session_id) =
        affinity_headers(&capture_headers(&gpt54(), options).await);

    assert_eq!(session_id.as_deref(), Some("session-123"));
    assert_eq!(client_request_id.as_deref(), Some("session-123"));
    assert_eq!(x_session_id, None);
}

#[tokio::test]
async fn clamps_prompt_cache_key_to_the_64_character_limit() {
    let mut options = keyed(&MockHttpClient::new());
    options.session_id = Some("x".repeat(67));

    let payload = capture(&gpt54(), &sys_hi_context(), options).await;

    assert_eq!(payload["prompt_cache_key"], json!("x".repeat(64)));
}

#[tokio::test]
async fn sets_cache_affinity_headers_for_proxy_requests_with_a_session_id() {
    let model = retargeted(&gpt54(), "opencode", "https://proxy.example.com/v1");
    let mut options = keyed(&MockHttpClient::new());
    options.session_id = Some("session-123".to_owned());

    let (session_id, client_request_id, _x_session_id) =
        affinity_headers(&capture_headers(&model, options).await);

    assert_eq!(session_id.as_deref(), Some("session-123"));
    assert_eq!(client_request_id.as_deref(), Some("session-123"));
}

#[tokio::test]
async fn uses_openrouter_session_affinity_header_when_configured() {
    let model = with_compat(
        &retargeted(&gpt54(), "proxy", "https://proxy.example.com/v1"),
        json!({ "sessionAffinityFormat": "openrouter" }),
    );
    let mut options = keyed(&MockHttpClient::new());
    options.session_id = Some("session-proxy".to_owned());

    let ((session_id, client_request_id, x_session_id), payload) =
        capture_affinity(&model, options).await;
    assert_eq!(session_id, None);
    assert_eq!(client_request_id, None);
    assert_eq!(x_session_id.as_deref(), Some("session-proxy"));
    assert!(payload.get("session_id").is_none());
    assert_eq!(payload["prompt_cache_key"], json!("session-proxy"));
}

#[tokio::test]
async fn auto_detects_openrouter_session_affinity_header_for_openrouter_endpoints() {
    let model = retargeted(&gpt54(), "openrouter", "https://openrouter.ai/api/v1");
    let mut options = keyed(&MockHttpClient::new());
    options.session_id = Some("session-openrouter".to_owned());

    let ((session_id, client_request_id, x_session_id), payload) =
        capture_affinity(&model, options).await;
    assert_eq!(session_id, None);
    assert_eq!(client_request_id, None);
    assert_eq!(x_session_id.as_deref(), Some("session-openrouter"));
    assert!(payload.get("session_id").is_none());
    assert_eq!(payload["prompt_cache_key"], json!("session-openrouter"));
}

#[tokio::test]
async fn uses_openai_no_session_format_when_configured() {
    let model = with_compat(
        &retargeted(&gpt54(), "proxy", "https://proxy.example.com/v1"),
        json!({ "sessionAffinityFormat": "openai-nosession" }),
    );
    let mut options = keyed(&MockHttpClient::new());
    options.session_id = Some("session-proxy".to_owned());

    let ((session_id, client_request_id, x_session_id), payload) =
        capture_affinity(&model, options).await;
    assert_eq!(session_id, None);
    assert_eq!(client_request_id.as_deref(), Some("session-proxy"));
    assert_eq!(x_session_id, None);
    assert!(payload.get("session_id").is_none());
    assert_eq!(payload["prompt_cache_key"], json!("session-proxy"));
}

#[tokio::test]
async fn uses_openai_no_session_format_for_opencode_models() {
    let model = builtin_model("opencode", "gpt-5.4");
    assert_eq!(
        model
            .compat
            .as_ref()
            .and_then(|compat| compat.session_affinity_format),
        Some(pi_ai::types::SessionAffinityFormat::OpenaiNosession)
    );
    let mut options = keyed(&MockHttpClient::new());
    options.session_id = Some("session-opencode".to_owned());

    let ((session_id, client_request_id, x_session_id), payload) =
        capture_affinity(&model, options).await;
    assert_eq!(session_id, None);
    assert_eq!(client_request_id.as_deref(), Some("session-opencode"));
    assert_eq!(x_session_id, None);
    assert_eq!(payload["prompt_cache_key"], json!("session-opencode"));
}

#[tokio::test]
async fn can_omit_the_session_id_header_while_preserving_other_affinity_data() {
    let model = with_compat(
        &retargeted(&gpt54(), "opencode", "https://proxy.example.com/v1"),
        json!({ "sessionAffinityFormat": "openai-nosession" }),
    );
    let mut options = keyed(&MockHttpClient::new());
    options.session_id = Some("session-123".to_owned());

    let ((session_id, client_request_id, _x_session_id), payload) =
        capture_affinity(&model, options).await;
    assert_eq!(session_id, None);
    assert_eq!(client_request_id.as_deref(), Some("session-123"));
    assert_eq!(payload["prompt_cache_key"], json!("session-123"));
}

#[tokio::test]
async fn lets_explicit_headers_override_the_default_affinity_headers() {
    let mut options = keyed(&MockHttpClient::new());
    options.session_id = Some("session-123".to_owned());
    options.headers = Some(
        [
            ("session_id".to_owned(), Some("override-session".to_owned())),
            (
                "x-client-request-id".to_owned(),
                Some("override-request".to_owned()),
            ),
        ]
        .into_iter()
        .collect(),
    );

    let ((session_id, client_request_id, _x_session_id), _payload) =
        capture_affinity(&gpt54(), options).await;
    assert_eq!(session_id.as_deref(), Some("override-session"));
    assert_eq!(client_request_id.as_deref(), Some("override-request"));
}

#[tokio::test]
async fn omits_affinity_headers_when_cache_retention_is_none() {
    let mut options = keyed(&MockHttpClient::new());
    options.cache_retention = Some(CacheRetention::None);
    options.session_id = Some("session-123".to_owned());

    let (session_id, client_request_id, _x_session_id) =
        affinity_headers(&capture_headers(&gpt54(), options).await);

    assert_eq!(session_id, None);
    assert_eq!(client_request_id, None);
}

/// The terminal run the service-tier suites stream: `response.completed`
/// with the tier and the 100k-token usage, upstream's `sse` fixture.
fn completed_with_service_tier(service_tier: &str) -> Value {
    json!({
        "type": "response.completed",
        "response": {
            "status": "completed",
            "service_tier": service_tier,
            "usage": {
                "input_tokens": 100_000,
                "output_tokens": 100_000,
                "total_tokens": 200_000,
                "input_tokens_details": { "cached_tokens": 0 },
            },
        },
    })
}

/// Stream the tier setup and settle the final message.
async fn capture_tier_result(model: &Model, service_tier: &str) -> AssistantMessage {
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[completed_with_service_tier(service_tier)]);
    let mut options = keyed(&mock);
    options.service_tier = Some(service_tier.to_owned());
    let stream = openai_responses::stream(model, &sys_hi_context(), Some(&options));
    common::drain_and_settle(&stream).await
}

#[tokio::test]
async fn applies_service_tier_cost_multipliers() {
    // (model id, tier, multiplier), upstream's `it.each` rows.
    let cases: [(&str, &str, f64); 3] = [
        ("gpt-5.4", "priority", 2.0),
        ("gpt-5.5", "priority", 2.5),
        ("gpt-5.5", "flex", 0.5),
    ];
    for (model_id, service_tier, multiplier) in cases {
        let model = builtin_model("openai", model_id);
        let result = capture_tier_result(&model, service_tier).await;

        // calculate_cost prices per million tokens, then the tier scales
        // every component; the total recomputes from the parts.
        let base = |rate: f64| rate / 1_000_000.0 * 100_000.0;
        let expected_input = base(model.cost.rates.input) * multiplier;
        let expected_output = base(model.cost.rates.output) * multiplier;
        assert_eq!(result.usage.cost.input, expected_input, "{model_id}");
        assert_eq!(result.usage.cost.output, expected_output, "{model_id}");
        assert_eq!(
            result.usage.cost.total,
            expected_input + expected_output,
            "{model_id}"
        );
    }
}

#[tokio::test]
async fn sends_max_output_tokens_by_default() {
    let mut options = keyed(&MockHttpClient::new());
    options.max_tokens = Some(1024);

    let payload = capture(&gpt54(), &sys_hi_context(), options).await;

    assert_eq!(payload["max_output_tokens"], json!(1024));
}

#[tokio::test]
async fn omits_max_output_tokens_when_supports_max_output_tokens_is_false() {
    let base = gpt54();
    let mut compat = base.compat.clone().unwrap_or_default();
    compat.supports_max_output_tokens = Some(false);
    let model = Model {
        compat: Some(compat),
        ..base
    };
    let mut options = keyed(&MockHttpClient::new());
    options.max_tokens = Some(1024);

    let payload = capture(&model, &sys_hi_context(), options).await;

    assert!(payload.get("max_output_tokens").is_none());
}

// ---------------------------------------------------------------------------
// empty tool result (upstream openai-responses-empty-tool-result)
// ---------------------------------------------------------------------------

fn sys_hi_context() -> Context {
    Context {
        system_prompt: Some("sys".to_owned()),
        messages: vec![user_message_now("hi")],
        ..Context::default()
    }
}

#[tokio::test]
async fn uses_no_tool_output_placeholder_for_empty_tool_results_without_images() {
    let model = builtin_model("openai", "gpt-4o-mini");
    let now = pi_ai::auth::resolve::now_ms();
    let assistant = AssistantMessage {
        content: vec![AssistantBlock::ToolCall(ToolCall {
            id: "tool-1".to_owned(),
            name: "bash".to_owned(),
            arguments: serde_json::Map::from_iter([("command".to_owned(), json!("true"))]),
            thought_signature: None,
            namespace: None,
        })],
        usage: empty_usage(),
        stop_reason: StopReason::ToolUse,
        ..common::assistant_message_with_content(
            "openai-responses",
            "openai",
            "gpt-4o-mini",
            Vec::new(),
        )
    };
    let context = Context {
        messages: vec![
            common::user_message_at("Run the command", now - 1),
            Message::Assistant(assistant),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "tool-1".to_owned(),
                tool_name: "bash".to_owned(),
                content: vec![ToolResultBlock::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                })],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: false,
                timestamp: now + 1,
            }),
        ],
        ..Context::default()
    };

    let input = convert_responses_messages(&model, &context, &openai_tool_call_providers(), None)
        .expect("conversion");

    let function_call_output = input
        .iter()
        .find(|item| item["type"] == json!("function_call_output"))
        .expect("function_call_output item");
    assert_eq!(function_call_output["output"], json!("(no tool output)"));
    let serialized = function_call_output.to_string();
    assert!(!serialized.contains("see attached image"));
}

fn openai_tool_call_providers() -> BTreeSet<String> {
    ["openai", "openai-codex", "opencode"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

// ---------------------------------------------------------------------------
// foreign tool-call id (upstream openai-responses-foreign-toolcall-id)
// ---------------------------------------------------------------------------

const COPILOT_RAW_TOOL_CALL_ID: &str = "call_4VnzVawQXPB9MgYib7CiQFEY|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt6SLUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD070Du16q1glydVjWR0SBJsE6TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wkmFqHeqTyEei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcbi2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nxj+TnEkReA==";

#[tokio::test]
async fn hashes_foreign_copilot_tool_item_ids_into_a_bounded_codex_safe_shape() {
    let model = builtin_model("openai-codex", "gpt-5.5");
    let now = pi_ai::auth::resolve::now_ms();
    let assistant = common::assistant_message_with_content(
        "openai-responses",
        "github-copilot",
        "gpt-5.5",
        vec![AssistantBlock::ToolCall(ToolCall {
            id: COPILOT_RAW_TOOL_CALL_ID.to_owned(),
            name: "edit".to_owned(),
            arguments: serde_json::Map::from_iter([(
                "path".to_owned(),
                json!("src/styles/app.css"),
            )]),
            thought_signature: None,
            namespace: None,
        })],
    );
    let tool_result = Message::ToolResult(ToolResultMessage {
        tool_call_id: COPILOT_RAW_TOOL_CALL_ID.to_owned(),
        tool_name: "edit".to_owned(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: "ok".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: now - 1000,
    });
    let context = Context {
        system_prompt: Some("You are concise.".to_owned()),
        messages: vec![
            common::user_message_at("Use the tool.", now - 3000),
            Message::Assistant(assistant),
            tool_result,
        ],
        ..Context::default()
    };

    let input = convert_responses_messages(&model, &context, &openai_tool_call_providers(), None)
        .expect("conversion");

    let function_call = input
        .iter()
        .find(|item| item["type"] == json!("function_call"))
        .expect("function_call item");
    let item_part = COPILOT_RAW_TOOL_CALL_ID
        .split('|')
        .nth(1)
        .expect("the pipe-split item id");
    let expected_item_id = format!("fc_{}", short_hash(item_part));
    assert_eq!(function_call["id"], json!(expected_item_id));
    assert!(function_call["id"].as_str().expect("id").len() <= 64);
    let id = function_call["id"].as_str().expect("the id string");
    assert!(id.starts_with("fc_"));
    assert!(
        id[3..]
            .chars()
            .all(|character| character.is_ascii_alphanumeric())
    );
}

// ---------------------------------------------------------------------------
// message id (upstream openai-responses-message-id)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generates_unique_fallback_message_ids_for_multiple_text_blocks() {
    let model = builtin_model("openai-codex", "gpt-5.5");
    let now = pi_ai::auth::resolve::now_ms();
    let assistant = common::assistant_message_with_content(
        "anthropic-messages",
        "anthropic",
        "claude-opus-4-8",
        vec![
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "private reasoning".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantBlock::Text(TextContent {
                text: "visible answer".to_owned(),
                text_signature: None,
            }),
        ],
    );
    let context = Context {
        system_prompt: Some("You are concise.".to_owned()),
        messages: vec![
            common::user_message_at("hello", now - 2000),
            Message::Assistant(assistant),
        ],
        ..Context::default()
    };

    let input = convert_responses_messages(&model, &context, &openai_tool_call_providers(), None)
        .expect("conversion");

    let message_ids: Vec<&str> = input
        .iter()
        .filter(|item| item["type"] == json!("message") && item["id"].is_string())
        .filter_map(|item| item["id"].as_str())
        .collect();
    assert_eq!(message_ids, ["msg_pi_1", "msg_pi_1_1"]);
    assert_eq!(
        message_ids.iter().collect::<BTreeSet<_>>().len(),
        message_ids.len()
    );
}

// ---------------------------------------------------------------------------
// tool-call namespaces (upstream openai-responses-namespace)
// ---------------------------------------------------------------------------

/// The hand-built model upstream's namespace suite constructs.
fn namespace_model() -> Model {
    Model {
        id: "gpt-5.4".to_owned(),
        name: "GPT-5.4".to_owned(),
        api: Api::from("openai-responses"),
        provider: ProviderId::from("openai"),
        base_url: "https://api.openai.com/v1".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 400_000,
        max_tokens: 128_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The fresh accumulator a processor run starts from, upstream's
/// `createOutput`.
fn fresh_output(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: empty_usage(),
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: pi_ai::auth::resolve::now_ms(),
    }
}

/// The canned SSE response a direct processor run consumes.
fn responses_response(events: &[Value]) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: vec![("content-type".to_owned(), "text/event-stream".to_owned())],
        body: HttpByteStream::from_chunks(vec![Ok(Bytes::from(
            common::openai_responses_sse_body(events),
        ))]),
    }
}

/// Run the events through the shared processor into a fresh accumulator,
/// returning the outcome, the accumulator, and the pushed events. The
/// explicit `end` closes the stream so the buffered-event drain terminates,
/// standing in for upstream's push spy.
async fn run_events(
    events: &[Value],
    model: &Model,
    options: Option<&OpenAiResponsesStreamOptions>,
) -> (
    Result<(), String>,
    AssistantMessage,
    Vec<AssistantMessageEvent>,
) {
    let mut output = fresh_output(model);
    let stream = pi_ai::utils::event_stream::assistant_message_event_stream();
    let outcome = process_responses_stream(
        responses_response(events),
        &mut output,
        &stream,
        model,
        options,
    )
    .await;
    stream.end(None);
    let mut pushed = Vec::new();
    while let Some(event) = stream.next().await {
        pushed.push(event);
    }
    (outcome, output, pushed)
}

fn function_call_namespace_events() -> Vec<Value> {
    vec![
        json!({
            "type": "response.output_item.added",
            "sequence_number": 0,
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": "lookup",
                "arguments": "",
            },
        }),
        json!({
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": "lookup",
                "arguments": "{\"value\":\"hello\"}",
                "namespace": "dynamic_tools",
            },
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 2,
            "response": { "id": "resp_test", "status": "completed" },
        }),
    ]
}

fn custom_tool_call_namespace_events() -> Vec<Value> {
    vec![
        json!({
            "type": "response.output_item.added",
            "sequence_number": 0,
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "id": "ctc_test",
                "call_id": "call_test",
                "name": "query",
                "input": "",
            },
        }),
        json!({
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "id": "ctc_test",
                "call_id": "call_test",
                "name": "query",
                "input": "hello",
                "namespace": "dynamic_tools",
            },
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 2,
            "response": { "id": "resp_test", "status": "completed" },
        }),
    ]
}

/// The `{"query": "input"}` grammar map, upstream's
/// `grammarToolInputProperties`.
#[tokio::test]
async fn omits_an_absent_error_message() {
    let model = namespace_model();

    let (outcome, output, _pushed) =
        run_events(&function_call_namespace_events(), &model, None).await;

    assert!(outcome.is_ok());
    assert!(output.error_message.is_none());
}

#[tokio::test]
async fn round_trips_a_function_namespace_received_only_on_output_item_done() {
    let model = namespace_model();

    let (outcome, output, _pushed) =
        run_events(&function_call_namespace_events(), &model, None).await;
    assert!(outcome.is_ok());

    let tool_call = &output.content[0];
    let AssistantBlock::ToolCall(tool_call) = tool_call else {
        panic!("expected toolCall block");
    };
    assert_eq!(tool_call.id, "call_test|fc_test");
    assert_eq!(tool_call.name, "lookup");
    assert_eq!(
        tool_call.arguments,
        serde_json::Map::from_iter([("value".to_owned(), json!("hello"))])
    );
    assert_eq!(tool_call.namespace.as_deref(), Some("dynamic_tools"));

    let context = Context {
        messages: vec![Message::Assistant(output)],
        ..Context::default()
    };
    let replayed = convert_responses_messages(&model, &context, &openai_only(), None)
        .expect("conversion")
        .into_iter()
        .find(|item| item["type"] == json!("function_call"))
        .expect("function_call item");
    assert_eq!(replayed["id"], json!("fc_test"));
    assert_eq!(replayed["call_id"], json!("call_test"));
    assert_eq!(replayed["name"], json!("lookup"));
    assert_eq!(replayed["arguments"], json!("{\"value\":\"hello\"}"));
    assert_eq!(replayed["namespace"], json!("dynamic_tools"));
}

#[tokio::test]
async fn round_trips_a_custom_tool_namespace_received_only_on_output_item_done() {
    let model = namespace_model();
    let properties = grammar_properties();

    let (outcome, output, _pushed) = run_events(
        &custom_tool_call_namespace_events(),
        &model,
        Some(&OpenAiResponsesStreamOptions {
            grammar_tool_input_properties: Some(properties.clone()),
            ..OpenAiResponsesStreamOptions::default()
        }),
    )
    .await;
    assert!(outcome.is_ok());

    let tool_call = &output.content[0];
    let AssistantBlock::ToolCall(tool_call) = tool_call else {
        panic!("expected toolCall block");
    };
    assert_eq!(tool_call.id, "call_test|ctc_test");
    assert_eq!(tool_call.name, "query");
    assert_eq!(
        tool_call.arguments,
        serde_json::Map::from_iter([("input".to_owned(), json!("hello"))])
    );
    assert_eq!(tool_call.namespace.as_deref(), Some("dynamic_tools"));

    let context = Context {
        messages: vec![Message::Assistant(output)],
        ..Context::default()
    };
    let replayed = convert_responses_messages(
        &model,
        &context,
        &openai_only(),
        Some(&ConvertResponsesMessagesOptions {
            grammar_tool_input_properties: Some(properties),
            ..ConvertResponsesMessagesOptions::default()
        }),
    )
    .expect("conversion")
    .into_iter()
    .find(|item| item["type"] == json!("custom_tool_call"))
    .expect("custom_tool_call item");
    assert_eq!(replayed["id"], json!("ctc_test"));
    assert_eq!(replayed["call_id"], json!("call_test"));
    assert_eq!(replayed["name"], json!("query"));
    assert_eq!(replayed["input"], json!("hello"));
    assert_eq!(replayed["namespace"], json!("dynamic_tools"));
}

#[tokio::test]
async fn drops_namespaces_when_the_target_cannot_replay_their_load_items() {
    let base = namespace_model();
    let target_models: [(&str, Model); 3] = [
        (
            "gpt-5.2",
            Model {
                id: "gpt-5.2".to_owned(),
                name: "GPT-5.2".to_owned(),
                ..base.clone()
            },
        ),
        (
            "azure",
            Model {
                provider: ProviderId::from("azure-openai-responses"),
                ..base.clone()
            },
        ),
        (
            "codex",
            Model {
                api: Api::from("openai-codex-responses"),
                provider: ProviderId::from("openai-codex"),
                id: "gpt-5.3-codex-spark".to_owned(),
                name: "GPT-5.3 Codex Spark".to_owned(),
                ..base.clone()
            },
        ),
    ];
    for (label, target_model) in target_models {
        let mut output = fresh_output(&base);
        output.content = vec![
            AssistantBlock::ToolCall(ToolCall {
                id: "call_function|fc_test".to_owned(),
                name: "lookup".to_owned(),
                arguments: serde_json::Map::from_iter([("value".to_owned(), json!("hello"))]),
                thought_signature: None,
                namespace: Some("dynamic_tools".to_owned()),
            }),
            AssistantBlock::ToolCall(ToolCall {
                id: "call_custom|ctc_test".to_owned(),
                name: "query".to_owned(),
                arguments: serde_json::Map::from_iter([("input".to_owned(), json!("hello"))]),
                thought_signature: None,
                namespace: Some("dynamic_tools".to_owned()),
            }),
        ];
        let context = Context {
            messages: vec![Message::Assistant(output)],
            ..Context::default()
        };

        let replayed = convert_responses_messages(
            &target_model,
            &context,
            &openai_only(),
            Some(&ConvertResponsesMessagesOptions {
                grammar_tool_input_properties: Some(grammar_properties()),
                ..ConvertResponsesMessagesOptions::default()
            }),
        )
        .expect("conversion");

        let function_call = replayed
            .iter()
            .find(|item| item["type"] == json!("function_call"))
            .expect("function_call item");
        assert!(function_call.get("namespace").is_none(), "{label}");
        let custom_tool_call = replayed
            .iter()
            .find(|item| item["type"] == json!("custom_tool_call"))
            .expect("custom_tool_call item");
        assert!(custom_tool_call.get("namespace").is_none(), "{label}");
    }
}

#[tokio::test]
async fn does_not_add_a_namespace_to_ordinary_function_calls() {
    let model = namespace_model();
    let mut output = fresh_output(&model);
    output.content.push(AssistantBlock::ToolCall(ToolCall {
        id: "call_test|fc_test".to_owned(),
        name: "lookup".to_owned(),
        arguments: serde_json::Map::from_iter([("value".to_owned(), json!("hello"))]),
        thought_signature: None,
        namespace: None,
    }));
    let context = Context {
        messages: vec![Message::Assistant(output)],
        ..Context::default()
    };

    let replayed = convert_responses_messages(&model, &context, &openai_only(), None)
        .expect("conversion")
        .into_iter()
        .find(|item| item["type"] == json!("function_call"))
        .expect("function_call item");

    assert!(replayed.get("namespace").is_none());
}

fn openai_only() -> BTreeSet<String> {
    std::iter::once("openai".to_owned()).collect()
}

fn grammar_properties() -> std::collections::BTreeMap<String, String> {
    std::iter::once(("query".to_owned(), "input".to_owned())).collect()
}

// ---------------------------------------------------------------------------
// partial json cleanup (upstream openai-responses-partial-json-cleanup)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn removes_the_partial_arguments_scratch_from_persisted_tool_calls() {
    let model = Model {
        id: "gpt-5-mini".to_owned(),
        name: "GPT-5 Mini".to_owned(),
        ..namespace_model()
    };
    let arguments_json = "{\"path\":\"README.md\",\"content\":\"updated\"}";
    let events = vec![
        json!({
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": "edit",
                "arguments": "",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "delta": "{\"path\":\"README.md\"",
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "delta": ",\"content\":\"updated\"}",
        }),
        json!({
            "type": "response.function_call_arguments.done",
            "arguments": arguments_json,
        }),
        json!({
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": "edit",
                "arguments": arguments_json,
            },
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 5,
            "response": { "id": "resp_test", "status": "completed" },
        }),
    ];

    let (outcome, output, pushed) = run_events(&events, &model, None).await;
    assert!(outcome.is_ok());

    assert_eq!(output.content.len(), 1);
    let AssistantBlock::ToolCall(persisted) = &output.content[0] else {
        panic!("expected toolCall block");
    };
    // Upstream additionally asserts the scratch `partialJson` field is gone;
    // the Rust block type never carries the scratch — it lives in the
    // processor state and drops when the item closes.
    assert_eq!(
        json!(persisted.arguments),
        json!({ "path": "README.md", "content": "updated" })
    );
    let toolcall_end = pushed.iter().find_map(|event| match event {
        AssistantMessageEvent::ToolcallEnd { tool_call, .. } => Some(tool_call),
        _ => None,
    });
    let toolcall_end = toolcall_end.expect("toolcall_end event");
    // Upstream asserts identity with the persisted block; the Rust event
    // carries a clone, so the restated assertion is field equality.
    assert_eq!(toolcall_end.id, persisted.id);
    assert_eq!(toolcall_end.name, persisted.name);
    assert_eq!(toolcall_end.arguments, persisted.arguments);
}

// ---------------------------------------------------------------------------
// terminal event (upstream openai-responses-terminal-event)
// ---------------------------------------------------------------------------

/// The upstream regression context: an empty system prompt, one block-text
/// user turn, and no tools.
fn empty_tools_hi_context() -> Context {
    Context {
        system_prompt: Some(String::new()),
        messages: vec![Message::User(UserMessage {
            content: UserContent::Blocks(vec![UserBlock::Text(TextContent {
                text: "hi".to_owned(),
                text_signature: None,
            })]),
            timestamp: 0,
        })],
        tools: Some(Vec::new()),
    }
}

/// The hand-built model upstream's terminal-event suite constructs.
fn terminal_model() -> Model {
    Model {
        id: "gpt-5-mini".to_owned(),
        name: "GPT-5 Mini".to_owned(),
        ..namespace_model()
    }
}

fn early_eof_events() -> Vec<Value> {
    vec![
        json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": { "id": "resp_early_eof" },
        }),
        json!({
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": { "type": "reasoning", "id": "rs_early_eof", "summary": [] },
        }),
        json!({
            "type": "response.reasoning_text.delta",
            "sequence_number": 2,
            "output_index": 0,
            "content_index": 0,
            "item_id": "rs_early_eof",
            "delta": "partial reasoning before the stream ends",
        }),
    ]
}

fn completed_events() -> Vec<Value> {
    vec![json!({
        "type": "response.completed",
        "sequence_number": 0,
        "response": {
            "id": "resp_completed",
            "status": "completed",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 7,
                "total_tokens": 27,
                "input_tokens_details": { "cached_tokens": 2, "cache_write_tokens": 3 },
            },
        },
    })]
}

fn incomplete_events(reason: &str) -> Vec<Value> {
    vec![json!({
        "type": "response.incomplete",
        "sequence_number": 0,
        "response": {
            "id": "resp_incomplete",
            "status": "incomplete",
            "incomplete_details": { "reason": reason },
            "usage": {
                "input_tokens": 30,
                "output_tokens": 12,
                "total_tokens": 42,
                "input_tokens_details": { "cached_tokens": 5 },
            },
        },
    })]
}

fn failed_events() -> Vec<Value> {
    vec![json!({
        "type": "response.failed",
        "sequence_number": 0,
        "response": {
            "id": "resp_failed",
            "status": "failed",
            "error": { "code": "server_error", "message": "boom" },
        },
    })]
}

fn phased_message_events(phases: [&str; 2], terminal_status: &str) -> Vec<Value> {
    let mut events = vec![
        json!({
            "type": "response.output_item.added",
            "sequence_number": 0,
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_phase",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
                "phase": phases[0],
            },
        }),
        json!({
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_phase",
                "role": "assistant",
                "status": "completed",
                "content": [{ "type": "output_text", "text": "answer", "annotations": [] }],
                "phase": phases[1],
            },
        }),
    ];
    events.push(if terminal_status == "incomplete" {
        json!({
            "type": "response.incomplete",
            "sequence_number": 2,
            "response": {
                "id": "resp_phase",
                "status": "incomplete",
                "incomplete_details": { "reason": "max_output_tokens" },
            },
        })
    } else {
        json!({
            "type": "response.completed",
            "sequence_number": 2,
            "response": { "id": "resp_phase", "status": "completed" },
        })
    });
    events
}

/// The stop reasons the pushed events' partials carried, upstream's push
/// spy over `event.partial.stopReason`.
fn partial_stop_reasons(pushed: &[AssistantMessageEvent]) -> Vec<StopReason> {
    pushed
        .iter()
        .filter_map(|event| match event {
            AssistantMessageEvent::Start { partial }
            | AssistantMessageEvent::TextStart { partial, .. }
            | AssistantMessageEvent::TextDelta { partial, .. }
            | AssistantMessageEvent::TextEnd { partial, .. }
            | AssistantMessageEvent::ThinkingStart { partial, .. }
            | AssistantMessageEvent::ThinkingDelta { partial, .. }
            | AssistantMessageEvent::ThinkingEnd { partial, .. }
            | AssistantMessageEvent::ToolcallStart { partial, .. }
            | AssistantMessageEvent::ToolcallDelta { partial, .. }
            | AssistantMessageEvent::ToolcallEnd { partial, .. } => Some(partial.stop_reason),
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => None,
        })
        .collect()
}

#[tokio::test]
async fn rejects_streams_that_end_before_a_terminal_response_event() {
    let model = terminal_model();

    let (outcome, _output, _pushed) = run_events(&early_eof_events(), &model, None).await;

    assert_eq!(
        outcome.expect_err("the early stream rejects"),
        "OpenAI Responses stream ended before a terminal response event"
    );
}

/// Drain the stream's events and settle its result, the collect variant of
/// the shared drain helper.
async fn drain_events_and_settle(
    stream: &AssistantMessageEventStream,
) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
    let (events, result) = tokio::join!(
        async {
            let mut collected = Vec::new();
            while let Some(event) = stream.next().await {
                collected.push(event);
            }
            collected
        },
        stream.result()
    );
    (events, result)
}

#[tokio::test]
async fn emits_an_error_final_result_when_the_wrapper_stream_ends_before_a_terminal_response_event()
{
    let model = terminal_model();
    let context = empty_tools_hi_context();
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &early_eof_events());
    let options = keyed(&mock);

    let stream = openai_responses::stream(&model, &context, Some(&options));
    let (events, result) = drain_events_and_settle(&stream).await;

    let initial = events.first().expect("the start event");
    let AssistantMessageEvent::Start { partial } = initial else {
        panic!("expected start event");
    };
    assert_eq!(partial.stop_reason, StopReason::Pending);
    let last = events.last().expect("a last event");
    assert!(
        matches!(last, AssistantMessageEvent::Error { .. }),
        "{last:?}"
    );
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("OpenAI Responses stream ended before a terminal response event")
    );
}

#[tokio::test]
async fn tracks_message_phases() {
    let model = terminal_model();
    // (phases, expected partial stop reasons), upstream's `it.each` rows.
    let cases: [([&str; 2], [StopReason; 2]); 3] = [
        (
            ["commentary", "commentary"],
            [StopReason::Pending, StopReason::Pending],
        ),
        (
            ["final_answer", "final_answer"],
            [StopReason::Stop, StopReason::Stop],
        ),
        (
            ["commentary", "final_answer"],
            [StopReason::Pending, StopReason::Stop],
        ),
    ];
    for (phases, expected) in cases {
        let (_outcome, output, pushed) =
            run_events(&phased_message_events(phases, "completed"), &model, None).await;

        assert_eq!(partial_stop_reasons(&pushed), expected, "{phases:?}");
        assert_eq!(output.stop_reason, StopReason::Stop);
    }
}

#[tokio::test]
async fn replaces_a_provisional_final_answer_stop_with_an_incomplete_terminal_reason() {
    let model = terminal_model();

    let (_outcome, output, pushed) = run_events(
        &phased_message_events(["final_answer", "final_answer"], "incomplete"),
        &model,
        None,
    )
    .await;

    assert_eq!(
        partial_stop_reasons(&pushed),
        [StopReason::Stop, StopReason::Stop]
    );
    assert_eq!(output.stop_reason, StopReason::Length);
}

#[tokio::test]
async fn finalizes_completed_terminal_events_as_stop() {
    let model = terminal_model();

    let (outcome, output, _pushed) = run_events(&completed_events(), &model, None).await;
    assert!(outcome.is_ok());

    assert_eq!(output.response_id.as_deref(), Some("resp_completed"));
    assert_eq!(output.stop_reason, StopReason::Stop);
    assert_eq!(output.raw_stop_reason.as_deref(), Some("completed"));
    assert_eq!(output.usage.input, 15);
    assert_eq!(output.usage.output, 7);
    assert_eq!(output.usage.cache_read, 2);
    assert_eq!(output.usage.cache_write, 3);
    assert_eq!(output.usage.total_tokens, 27);
}

#[tokio::test]
async fn finalizes_incomplete_terminal_events_as_length_stops() {
    let model = terminal_model();

    let (outcome, output, _pushed) =
        run_events(&incomplete_events("max_output_tokens"), &model, None).await;
    assert!(outcome.is_ok());

    assert_eq!(output.response_id.as_deref(), Some("resp_incomplete"));
    assert_eq!(output.stop_reason, StopReason::Length);
    assert_eq!(
        output.raw_stop_reason.as_deref(),
        Some("incomplete.max_output_tokens")
    );
    assert_eq!(output.usage.input, 25);
    assert_eq!(output.usage.output, 12);
    assert_eq!(output.usage.cache_read, 5);
    assert_eq!(output.usage.cache_write, 0);
    assert_eq!(output.usage.total_tokens, 42);
}

#[tokio::test]
async fn finalizes_content_filtered_incomplete_responses_as_errors() {
    let model = terminal_model();

    let (outcome, output, _pushed) =
        run_events(&incomplete_events("content_filter"), &model, None).await;
    assert!(outcome.is_ok());

    assert_eq!(output.stop_reason, StopReason::Error);
    assert_eq!(
        output.raw_stop_reason.as_deref(),
        Some("incomplete.content_filter")
    );
    assert_eq!(
        output.error_message.as_deref(),
        Some("Response incomplete: content_filter")
    );
}

#[tokio::test]
async fn preserves_unknown_provider_incomplete_reasons_as_errors() {
    let model = terminal_model();

    let (outcome, output, _pushed) =
        run_events(&incomplete_events("max_time_limit"), &model, None).await;
    assert!(outcome.is_ok());

    assert_eq!(output.stop_reason, StopReason::Error);
    assert_eq!(
        output.raw_stop_reason.as_deref(),
        Some("incomplete.max_time_limit")
    );
    assert_eq!(
        output.error_message.as_deref(),
        Some("Response incomplete: max_time_limit")
    );
}

#[tokio::test]
async fn rejects_failed_terminal_events_with_the_provider_error() {
    let model = terminal_model();

    let (outcome, output, _pushed) = run_events(&failed_events(), &model, None).await;

    assert_eq!(
        outcome.expect_err("the failed terminal rejects"),
        "server_error: boom"
    );
    assert_eq!(output.raw_stop_reason.as_deref(), Some("failed"));
}

// ---------------------------------------------------------------------------
// tool result images (upstream openai-responses-tool-result-images)
// ---------------------------------------------------------------------------

/// Which adapter the flow streams through, upstream's per-provider
/// `describe.skipIf` cases.
#[derive(Clone, Copy, Debug)]
enum ImagesProvider {
    /// `getModel("openai", "gpt-5-mini")`.
    OpenAi,
    /// `getModel("azure-openai-responses", "gpt-4o-mini")`.
    Azure,
    /// `getModel("github-copilot", "gpt-5-mini")`.
    Copilot,
}

/// The `get_circle_with_description` tool, upstream's `getImageTool`.
fn get_circle_tool() -> Tool {
    tool(
        "get_circle_with_description",
        json!({ "type": "object", "properties": {} }),
    )
}

/// A minimal PNG stand-in for upstream's `test/data/red-circle.png` fixture:
/// the conversion reads only the mime and the base64 data, so the pixel
/// content never reaches an assertion.
const RED_IMAGE_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

fn image_tool_result_turn(tool_call_id: &str, tool_name: &str, timestamp: i64) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: tool_call_id.to_owned(),
        tool_name: tool_name.to_owned(),
        content: vec![
            ToolResultBlock::Text(TextContent {
                text: "A red circle with a diameter of 100 pixels.".to_owned(),
                text_signature: None,
            }),
            ToolResultBlock::Image(pi_ai::types::ImageContent {
                data: RED_IMAGE_BASE64.to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp,
    })
}

/// Turn 1's canned run: a function call for the circle tool, upstream's
/// live `complete()` result shape.
fn tool_call_run_events() -> Vec<Value> {
    vec![
        json!({ "type": "response.created", "response": { "id": "resp_imgs" } }),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_imgs",
                "call_id": "call_imgs",
                "name": "get_circle_with_description",
                "arguments": "",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "{}",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_imgs",
                "call_id": "call_imgs",
                "name": "get_circle_with_description",
                "arguments": "{}",
            },
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp_imgs",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "input_tokens_details": { "cached_tokens": 0 },
                },
            },
        }),
    ]
}

/// Turn 2's canned run: a completed response, upstream's live `stop` shape.
fn completed_run_events() -> Vec<Value> {
    vec![json!({
        "type": "response.completed",
        "response": {
            "id": "resp_imgs_2",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18,
                "input_tokens_details": { "cached_tokens": 0 },
            },
        },
    })]
}

/// Stream one images turn through the provider's adapter, upstream's
/// `complete(model, context, options)` dispatch.
async fn stream_images_turn(
    provider: ImagesProvider,
    model: &Model,
    mock: &MockHttpClient,
    context: &Context,
) -> AssistantMessage {
    match provider {
        ImagesProvider::OpenAi | ImagesProvider::Copilot => {
            let mut options = common::keyed_openai_responses_options(mock);
            options.reasoning_effort = Some(ThinkingLevel::Low);
            let stream = openai_responses::stream(model, context, Some(&options));
            common::drain_and_settle(&stream).await
        }
        ImagesProvider::Azure => {
            let options = AzureOpenAiResponsesOptions {
                transport_options: common::mock_transport(mock),
                api_key: Some("test-key".to_owned()),
                azure_base_url: Some("https://my-resource.openai.azure.com".to_owned()),
                ..AzureOpenAiResponsesOptions::default()
            };
            let stream = pi_ai::api::azure_openai_responses::stream(model, context, Some(&options));
            common::drain_and_settle(&stream).await
        }
    }
}

/// The two-turn flow upstream's verifier runs, ending in the second
/// request's captured payload.
async fn tool_result_images_payload(provider: ImagesProvider, model: &Model) -> Value {
    assert!(
        model.input.contains(&Modality::Image),
        "the fixture must support image input; upstream skips the rest"
    );
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/responses"))
        .respond_sequence(vec![
            common::openai_responses_sse_response(&tool_call_run_events()),
            common::openai_responses_sse_response(&completed_run_events()),
        ]);

    let mut context = Context {
        system_prompt: Some(
            "You are a helpful assistant that always uses the provided tool when asked.".to_owned(),
        ),
        messages: vec![user_message_now(
            "Call get_circle_with_description, then describe both the tool text and the image. Mention the color and shape.",
        )],
        tools: Some(vec![get_circle_tool()]),
    };

    let first = stream_images_turn(provider, model, &mock, &context).await;
    assert_eq!(first.stop_reason, StopReason::ToolUse, "turn 1");
    let AssistantBlock::ToolCall(tool_call) = &first.content[0] else {
        panic!("expected the streamed tool call");
    };

    context.messages.push(Message::Assistant(first.clone()));
    context.messages.push(image_tool_result_turn(
        &tool_call.id,
        &tool_call.name,
        pi_ai::auth::resolve::now_ms(),
    ));

    let second = stream_images_turn(provider, model, &mock, &context).await;
    assert_eq!(second.stop_reason, StopReason::Stop, "turn 2");
    assert!(second.error_message.is_none(), "turn 2");

    recorded_body_at(&mock, 1)
}

/// The request-shape half of upstream's verifier: the tool result's text and
/// image ride the `function_call_output` item, and nothing user-role follows.
/// The live-response text assertions (the model naming the color and shape)
/// are upstream's credential-gated half and stay unported.
fn assert_images_stay_in_function_call_output(payload: &Value) {
    let response_input = payload["input"].as_array().expect("the input array");
    let function_call_output_index = response_input
        .iter()
        .position(|item| item["type"] == json!("function_call_output"))
        .expect("function_call_output item");
    let output = &response_input[function_call_output_index]["output"];
    let output_items = output.as_array().expect("the output content array");
    let text_item = output_items
        .iter()
        .find(|item| item["type"] == json!("input_text"))
        .expect("input_text item");
    let image_item = output_items
        .iter()
        .find(|item| item["type"] == json!("input_image"))
        .expect("input_image item");
    assert!(
        text_item["text"]
            .as_str()
            .expect("text")
            .contains("A red circle with a diameter of 100 pixels.")
    );
    assert!(
        image_item["image_url"]
            .as_str()
            .expect("image_url")
            .starts_with("data:image/png;base64,")
    );
    let later_user_messages = response_input[function_call_output_index + 1..]
        .iter()
        .filter(|item| item["role"] == json!("user"))
        .count();
    assert_eq!(later_user_messages, 0);
}

#[tokio::test]
async fn openai_gpt_5_mini_sends_tool_result_images_in_function_call_output() {
    let payload = tool_result_images_payload(
        ImagesProvider::OpenAi,
        &builtin_model("openai", "gpt-5-mini"),
    )
    .await;
    assert_images_stay_in_function_call_output(&payload);
}

#[tokio::test]
async fn azure_gpt_4o_mini_sends_tool_result_images_in_function_call_output() {
    let payload = tool_result_images_payload(
        ImagesProvider::Azure,
        &builtin_model("azure-openai-responses", "gpt-4o-mini"),
    )
    .await;
    assert_images_stay_in_function_call_output(&payload);
}

#[tokio::test]
async fn github_copilot_gpt_5_mini_sends_tool_result_images_in_function_call_output() {
    let payload = tool_result_images_payload(
        ImagesProvider::Copilot,
        &builtin_model("github-copilot", "gpt-5-mini"),
    )
    .await;
    assert_images_stay_in_function_call_output(&payload);
}

// ---------------------------------------------------------------------------
// provider error body regression, responses tier
// (upstream provider-error-body-regression)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_responses_keeps_the_prefix_and_surfaces_the_body() {
    let model = Model {
        id: "gpt-test".to_owned(),
        name: "GPT Test".to_owned(),
        ..terminal_model()
    };
    let context = empty_tools_hi_context();
    let mock = MockHttpClient::new();
    mock.on(|request| request.url.contains("/responses"))
        .respond(pi_ai::http::json_response(
            403,
            &json!({ "error": "blocked by gateway WAF" }),
        ));
    let options = keyed(&mock);

    let stream = openai_responses::stream(&model, &context, Some(&options));
    let result = common::drain_and_settle(&stream).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the failure message");
    assert!(message.contains("OpenAI API error (403)"), "got: {message}");
    assert!(message.contains("blocked by gateway WAF"), "got: {message}");
}
// ---------------------------------------------------------------------------
// Port-added: malformed and edge event sequences through the shared
// Responses processor, and the wire-level request fields the upstream
// compat suites reach only implicitly
// ---------------------------------------------------------------------------

async fn run_skipping(events: Vec<Value>) -> (Result<(), String>, AssistantMessage) {
    let (outcome, output, _) = run_events(&events, &gpt54(), None).await;
    (outcome, output)
}

/// The argument-delta events for one tool call item, upstream's function-call
/// streaming shape.
fn tool_call_added(output_index: u64) -> Value {
    json!({
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "",
        },
    })
}

/// Port-added: argument deltas and dones that reference absent slots,
/// missing fields, or missing scratch state are skipped and the stream still
/// settles.
#[tokio::test]
async fn orphaned_argument_events_skip_without_touching_the_message() {
    let events = vec![
        tool_call_added(0),
        // Deltas and dones for an index with no tool call.
        json!({"type": "response.function_call_arguments.delta", "output_index": 7, "delta": "{\"x\":"}),
        json!({"type": "response.function_call_arguments.done", "output_index": 9, "arguments": "{}"}),
        // Deltas with no delta field.
        json!({"type": "response.function_call_arguments.delta", "output_index": 0}),
        json!({"type": "response.function_call_arguments.done", "output_index": 0}),
        json!({
            "type": "response.completed",
            "response": { "id": "resp_t", "status": "completed" },
        }),
    ];
    let (outcome, output) = run_skipping(events).await;
    assert!(outcome.is_ok(), "{outcome:?}");
    let call = output
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::ToolCall(call) => Some(call),
            _ => None,
        })
        .expect("the tool call");
    assert_eq![call.id, "call_1|fc_1"];
}

/// Port-added: custom tool input deltas and dones on a function-shaped call
/// skip, and the done event's full input replaces the streamed prefix.
#[tokio::test]
async fn custom_input_events_on_function_shaped_calls_skip() {
    let events = vec![
        tool_call_added(0),
        json!({"type": "response.custom_tool_call_input.delta", "output_index": 0, "delta": "zz"}),
        json!({"type": "response.custom_tool_call_input.done", "output_index": 0}),
        json!({"type": "response.custom_tool_call_input.delta"}),
        json!({
            "type": "response.completed",
            "response": { "id": "resp_t", "status": "completed" },
        }),
    ];
    let (outcome, output) = run_skipping(events).await;
    assert!(outcome.is_ok(), "{outcome:?}");
    let call = output
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::ToolCall(call) => Some(call),
            _ => None,
        })
        .expect("the function call");
    assert_eq![call.name, "lookup"];
}

/// Port-added: terminal status normalization — an unknown status drops the
/// field, a null response rides, and a non-object response drops.
#[tokio::test]
async fn terminal_response_shapes_normalize_like_the_wire() {
    // An unknown status fails with the unhandled-status message.
    let events = vec![
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
        }),
        json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "hi" }),
        json!({
            "type": "response.completed",
            "response": { "id": "resp_t", "status": "weird_status" },
        }),
    ];
    let (outcome, _output) = run_skipping(events).await;
    assert_eq![
        outcome.err().as_deref(),
        Some("Unhandled stop reason: weird_status")
    ];

    // A null response and a non-object response ride; the status read falls
    // back to the bare stop.
    for response in [Value::Null, json!("a string")] {
        let events = vec![
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
            }),
            json!({ "type": "response.output_text.delta", "item_id": "msg_1", "delta": "hi" }),
            json!({ "type": "response.completed", "response": response }),
        ];
        let (outcome, output) = run_skipping(events).await;
        assert!(outcome.is_ok(), "{outcome:?} {response}");
        assert_eq![output.stop_reason, StopReason::Stop, "{response}"];
        assert_eq!(message_text(&output), "hi", "{response}");
    }
}

/// A one-user-message context, the request-shape suites' fixture.
fn hello_context() -> Context {
    Context {
        messages: vec![user_message_now("Hello")],
        ..Context::default()
    }
}

/// The settled message's text blocks joined, the shape-assert reader.
fn message_text(message: &AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantBlock::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<String>()
}

/// Port-added: a `response.incomplete` without error details fails with the
/// no-provider-reason message; unknown statuses keep their provider text.
#[tokio::test]
async fn incomplete_terminal_events_spell_their_provider_reasons() {
    let events = vec![json!({
        "type": "response.incomplete",
        "response": { "id": "resp_t", "status": "incomplete" },
    })];
    let (outcome, output) = run_skipping(events).await;
    assert!(outcome.is_ok(), "{outcome:?}");
    assert_eq![output.stop_reason, StopReason::Error];
    let message = output.error_message.expect("the incomplete failure");
    assert![
        message.contains("Response incomplete without a provider reason"),
        "{message}"
    ];

    let events = vec![json!({
        "type": "response.incomplete",
        "response": { "id": "resp_t", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"} },
    })];
    // The max-output reason maps to the length stop.
    let (outcome, output) = run_skipping(events).await;
    assert!(outcome.is_ok(), "{outcome:?}");
    assert_eq![output.stop_reason, StopReason::Length];
    assert_eq![
        output.raw_stop_reason.as_deref(),
        Some("incomplete.max_output_tokens")
    ];
}

/// Port-added: a second summary delta joins the thinking text with a blank
/// line; an unknown output index skips.
#[tokio::test]
async fn summary_deltas_join_with_blank_lines_and_orphans_skip() {
    let events = vec![
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "reasoning", "id": "rs_1", "summary": [], "content": [] },
        }),
        json!({ "type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "one" }),
        json!({ "type": "response.reasoning_summary_text.delta", "output_index": 9, "delta": "orphan" }),
        json!({ "type": "response.reasoning_summary_part.done", "output_index": 0, "summary_index": 0 }),
        json!({ "type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "two" }),
        json!({
            "type": "response.completed",
            "response": { "id": "resp_t", "status": "completed" },
        }),
    ];
    let (outcome, output) = run_skipping(events).await;
    assert!(outcome.is_ok(), "{outcome:?}");
    let thinking = output
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::Thinking(thinking) => Some(thinking),
            _ => None,
        })
        .expect("the thinking block");
    assert_eq![thinking.thinking, "one\n\ntwo"];
}

/// Port-added: a reasoning item done with an empty summary falls back to the
/// content text.
#[tokio::test]
async fn reasoning_items_read_the_content_text_when_the_summary_is_empty() {
    let events = vec![
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "content": [{ "type": "reasoning_text", "text": "the content text" }],
            },
        }),
        json!({
            "type": "response.completed",
            "response": { "id": "resp_t", "status": "completed" },
        }),
    ];
    let (outcome, output) = run_skipping(events).await;
    assert!(outcome.is_ok(), "{outcome:?}");
    let thinking = output
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::Thinking(thinking) => Some(thinking),
            _ => None,
        })
        .expect("the thinking block");
    assert_eq![thinking.thinking, "the content text"];
}

/// Port-added: refusal text parts read their refusal wording.
#[tokio::test]
async fn refusal_text_parts_read_their_refusal_wording() {
    let events = vec![
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": { "type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": [] },
        }),
        json!({
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "delta": "",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{ "type": "refusal", "refusal": "I cannot help with that." }],
            },
        }),
        json!({
            "type": "response.completed",
            "response": { "id": "resp_t", "status": "completed" },
        }),
    ];
    let (outcome, output) = run_skipping(events).await;
    assert!(outcome.is_ok(), "{outcome:?}");
    assert![
        message_text(&output).contains("I cannot help with that."),
        "{}",
        message_text(&output)
    ];
}

/// Port-added: block user messages send `input_image` parts and images on
/// vision-capable models; a message whose blocks are all non-text skips.
#[tokio::test]
async fn user_image_blocks_ride_the_input_image_parts() {
    let mut model = gpt54();
    model.input = vec![Modality::Text, Modality::Image];
    let context = Context {
        messages: vec![
            Message::User(UserMessage {
                content: UserContent::Blocks(vec![
                    UserBlock::Text(common::text_block("look")),
                    UserBlock::Image(pi_ai::types::ImageContent {
                        data: "aGk=".to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ]),
                timestamp: 1,
            }),
            Message::User(UserMessage {
                content: UserContent::Blocks(vec![]),
                timestamp: 2,
            }),
        ],
        ..Context::default()
    };
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let options = OpenAiResponsesOptions {
        transport_options: common::mock_transport(&mock),
        api_key: Some("test-key".to_owned()),
        ..OpenAiResponsesOptions::default()
    };
    let _ = openai_responses::stream(&model, &context, Some(&options))
        .result()
        .await;
    let body = recorded_body(&mock);
    let input = body["input"].as_array().expect("the input items");
    let image_item = input
        .iter()
        .flat_map(|item| item["content"].as_array().cloned().unwrap_or_default())
        .find(|part| part["type"] == json!("input_image"))
        .expect("the input_image part");
    assert_eq!(image_item["image_url"], json!("data:image/png;base64,aGk="));
}

/// Port-added: a legacy text signature id rides as the item id; a v1 JSON
/// signature with a phase carries the phase and a >64-char id hashes.
#[tokio::test]
async fn legacy_message_ids_hash_and_signature_phases_ride() {
    let context = Context {
        messages: vec![
            Message::User(UserMessage {
                content: UserContent::Text("hello".to_owned()),
                timestamp: 1,
            }),
            Message::Assistant(common::assistant_message_with_content(
                "openai-responses",
                "openai",
                "gpt-5.4",
                vec![
                    AssistantBlock::Text(TextContent {
                        text: "legacy".to_owned(),
                        text_signature: Some("legacy-id".to_owned()),
                    }),
                    AssistantBlock::Text(TextContent {
                        text: "phased".to_owned(),
                        text_signature: Some(
                            json!({
                                "v": 1,
                                "id": "i".repeat(80),
                                "phase": "final_answer",
                            })
                            .to_string(),
                        ),
                    }),
                ],
            )),
        ],
        ..Context::default()
    };
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let mut options = keyed(&mock);
    options.transport_options = common::mock_transport(&mock);
    let _ = openai_responses::stream(&gpt54(), &context, Some(&options))
        .result()
        .await;
    let body = recorded_body(&mock);
    let message_item = body["input"]
        .as_array()
        .expect("the input items")
        .iter()
        .find(|item| item["role"] == json!("assistant"))
        .expect("the replayed assistant item");
    // The plain legacy signature rides as the item id; the long v1 id hashes
    // into the msg_ shape with its phase on the item.
    let input = body["input"].as_array().expect("the input items");
    let legacy_item = input
        .iter()
        .find(|item| item.get("id") == Some(&json!("legacy-id")))
        .expect("the legacy id item");
    assert_eq!(legacy_item["role"], json!("assistant"));
    let phased = input
        .iter()
        .find(|item| item.get("phase") == Some(&json!("final_answer")))
        .expect("the phased item");
    let id = phased["id"].as_str().expect("the hashed id");
    assert![id.starts_with("msg_"), "the long id hashes: {id}"];
    let _ = message_item;
}

/// Port-added: a malformed text signature fails the request with the parse
/// error before dispatch.
#[tokio::test]
async fn a_malformed_text_signature_fails_before_dispatch() {
    let context = Context {
        messages: vec![
            Message::User(UserMessage {
                content: UserContent::Text("hello".to_owned()),
                timestamp: 1,
            }),
            Message::Assistant(common::assistant_message_with_content(
                "openai-responses",
                "openai",
                "gpt-5.4",
                vec![AssistantBlock::Thinking(ThinkingContent {
                    thinking: "thought".to_owned(),
                    thinking_signature: Some("not json".to_owned()),
                    redacted: None,
                })],
            )),
        ],
        ..Context::default()
    };
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let mut options = keyed(&mock);
    options.transport_options = common::mock_transport(&mock);
    let result = openai_responses::stream(&gpt54(), &context, Some(&options))
        .result()
        .await;
    assert_eq![result.stop_reason, StopReason::Error];
    let message = result.error_message.expect("the parse failure");
    assert![
        message.contains("key must be a string") || message.contains("expected"),
        "{message}"
    ];
    assert_eq!(mock.request_count(), 0);
}

/// Port-added: long session ids clamp to the wire's 64 characters through
/// the sanitize/hash path.
#[tokio::test]
async fn long_session_ids_clamp_and_hash_into_the_wire_shape() {
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let mut options = keyed(&mock);
    options.transport_options = common::mock_transport(&mock);
    options.session_id = Some(long_session_id());
    let _ = openai_responses::stream(&gpt54(), &hello_context(), Some(&options))
        .result()
        .await;
    let body = recorded_body(&mock);
    let key = body["prompt_cache_key"].as_str().expect("the cache key");
    assert_eq!(key.chars().count(), 64);
    assert![key.starts_with('s'), "{key}"];
}

fn long_session_id() -> String {
    "s".repeat(80)
}

/// Port-added: the wire-level request extras — temperature, sampling params,
/// the reasoning effort spellings, and the summary value — ride their fields.
#[tokio::test]
async fn the_responses_request_extras_ride_their_wire_fields() {
    for (effort, expected) in [
        (ThinkingLevel::Minimal, "minimal"),
        (ThinkingLevel::Max, "max"),
    ] {
        let mock = MockHttpClient::new();
        openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
        let options = OpenAiResponsesOptions {
            transport_options: common::mock_transport(&mock),
            api_key: Some("test-key".to_owned()),
            reasoning_effort: Some(effort),
            reasoning_summary: Some(ReasoningSummary::Concise),
            temperature: Some(0.2),
            sampling_params: Some(
                std::iter::once(("presence_penalty".to_owned(), json!(0.5))).collect(),
            ),
            ..OpenAiResponsesOptions::default()
        };
        let _ = openai_responses::stream(&gpt54(), &hello_context(), Some(&options))
            .result()
            .await;
        let body = recorded_body(&mock);
        assert_eq!(body["reasoning"]["effort"], json!(expected), "{expected}");
        assert_eq!(body["reasoning"]["summary"], json!("concise"));
        assert_eq!(body["presence_penalty"], json!(0.5));
    }
}

/// Port-added: long retention sends the 24h retention field on capable
/// models, and the explicit prompt-cache mode sends its options.
#[tokio::test]
async fn the_prompt_cache_extras_ride_their_request_fields() {
    // Long retention with the 24h spelling on the catalog model.
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let mut options = keyed(&mock);
    options.transport_options = common::mock_transport(&mock);
    options.cache_retention = Some(CacheRetention::Long);
    options.session_id = Some("cache-session".to_owned());
    let _ = openai_responses::stream(&gpt54(), &hello_context(), Some(&options))
        .result()
        .await;
    let body = recorded_body(&mock);
    assert_eq!(body["prompt_cache_retention"], json!("24h"));

    // The explicit prompt-cache mode sends mode/ttl options instead.
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let model = with_compat(
        &gpt54(),
        json!({"supportsExplicitPromptCacheMode": true, "supportsLongCacheRetention": true}),
    );
    let mut options = keyed(&mock);
    options.transport_options = common::mock_transport(&mock);
    options.cache_retention = Some(CacheRetention::Long);
    let _ = openai_responses::stream(&model, &hello_context(), Some(&options))
        .result()
        .await;
    let body = recorded_body(&mock);
    assert_eq!(body["prompt_cache_options"]["ttl"], json!("30m"));

    // Under no-cache retention the explicit mode sends the explicit mode.
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let mut options = keyed(&mock);
    options.transport_options = common::mock_transport(&mock);
    options.cache_retention = Some(CacheRetention::None);
    let _ = openai_responses::stream(&model, &hello_context(), Some(&options))
        .result()
        .await;
    let body = recorded_body(&mock);
    assert_eq!(body["prompt_cache_options"]["mode"], json!("explicit"));
}

/// Port-added: the `PI_CACHE_RETENTION` env drives direct requests when the
/// options carry none.
#[tokio::test]
async fn the_pi_cache_retention_env_drives_the_responses_requests() {
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let mut options = keyed(&mock);
    options.transport_options = common::mock_transport(&mock);
    options.env =
        Some(std::iter::once(("PI_CACHE_RETENTION".to_owned(), "long".to_owned())).collect());
    let _ = openai_responses::stream(&gpt54(), &hello_context(), Some(&options))
        .result()
        .await;
    let body = recorded_body(&mock);
    assert_eq!(body["prompt_cache_retention"], json!("24h"));
}

/// Port-added: the tool-search deferred mode rides the compat flag and keeps
/// the immediate tools on the request.
#[tokio::test]
async fn the_tool_search_deferred_mode_rides_the_compat_flag() {
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let model = with_compat(&gpt54(), json!({"supportsToolSearch": true}));
    let context = Context {
        messages: hello_context().messages,
        tools: Some(vec![common::lookup_tool(), common::late_tool()]),
        ..Context::default()
    };
    let mut options = keyed(&mock);
    options.transport_options = common::mock_transport(&mock);
    let _ = openai_responses::stream(&model, &context, Some(&options))
        .result()
        .await;
    let body = recorded_body(&mock);
    // No tool results in history: every tool rides immediately.
    let lookup = body["tools"]
        .as_array()
        .expect("the tools array")
        .iter()
        .any(|tool| tool["name"] == json!("lookup"));
    assert![lookup];
}

/// Port-added: an aborted signal at finish fails the responses stream
/// aborted, and a terminal-free stream fails with the lifecycle message.
#[tokio::test]
async fn the_responses_stream_lifecycle_failures_surface_their_messages() {
    // A stream that ends without a terminal event.
    let mock = MockHttpClient::new();
    openai_responses_mock_with(
        &mock,
        &[json!({
            "type": "response.output_text.delta",
            "item_id": "msg_x",
            "delta": "hi",
        })],
    );
    let mut options = keyed(&mock);
    options.transport_options = common::mock_transport(&mock);
    let result = openai_responses::stream(&gpt54(), &hello_context(), Some(&options))
        .result()
        .await;
    assert_eq![result.stop_reason, StopReason::Error];

    // A signal cancelled at finish reports the abort.
    let token = tokio_util::sync::CancellationToken::new();
    let cancel_token = token.clone();
    let mock = MockHttpClient::new();
    openai_responses_mock_with(&mock, &[openai_responses_completed_event()]);
    let options = OpenAiResponsesOptions {
        transport_options: TransportOptions {
            http_client: Some(std::sync::Arc::new(mock.clone())),
            signal: Some(token),
            on_response: Some(pi_ai::types::OnResponse::new(move |_response, _model| {
                cancel_token.cancel();
                Box::pin(async {})
            })),
            ..TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        ..OpenAiResponsesOptions::default()
    };
    let result = openai_responses::stream(&gpt54(), &hello_context(), Some(&options))
        .result()
        .await;
    assert_eq![result.stop_reason, StopReason::Aborted];
    assert_eq!(result.error_message.as_deref(), Some("Request was aborted"));
}

//! The Bedrock Converse Stream suite, ported from
//! `packages/ai/test/bedrock-raw-stop-reason.test.ts`,
//! `packages/ai/test/bedrock-thinking-payload.test.ts`,
//! `packages/ai/test/bedrock-convert-messages.test.ts`,
//! `packages/ai/test/bedrock-cache-write-1h-cost.test.ts`,
//! `packages/ai/test/bedrock-error-metadata.test.ts`,
//! `packages/ai/test/bedrock-credentials.test.ts`,
//! `packages/ai/test/bedrock-endpoint-resolution.test.ts`,
//! `packages/ai/test/bedrock-custom-headers.test.ts`, and
//! `packages/ai/test/bedrock-redacted-reasoning.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements:
//! - The mocked `BedrockRuntimeClient` is a seam recorder: it captures the
//!   command input and the resolved client config per send (upstream's
//!   `constructorCalls` and `ConverseStreamCommand.input` probes) and plays a
//!   fixed event stream or failure. The `captureClientConfig` assertions read
//!   `resolve_client_config` output, the constructor-capture analog; payload
//!   assertions read the `onPayload` capture.
//! - The `streamSimple` header regression (upstream's VC4) drives the same
//!   option mapping — `build_base_options` into the adapter options and the
//!   client-config resolution — rather than a real `stream_simple` send: the
//!   runtime seam rides the adapter-local options, so a direct simple-entry
//!   stream cannot carry the mock and would reach the real SDK.
//! - Unknown user/assistant content types are statically upheld: the message
//!   enums are closed, so the skip branches upstream guards can never
//!   receive foreign blocks, and the unknown-only-message fixtures carry no
//!   port.
//! - Surrogate sanitization is statically upheld: Rust strings are valid
//!   UTF-8, so the half-surrogate fixtures upstream turns into `<empty>`
//!   cannot occur here.
//! - The streamed `redactedContent` blobs ride the wire events as base64
//!   strings; the replay path carries them as byte arrays, the JSON command
//!   input's shape for the blob.

#![expect(
    clippy::expect_used,
    reason = "the tests pin adapter outcomes; an unexpected shape panics the test by design"
)]

mod common;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pi_ai::api::bedrock_converse_stream::{
    BedrockRuntime, BedrockStreamFailure, format_bedrock_error, stream as stream_bedrock,
};
use pi_ai::api::bedrock_options::{
    BedrockClientConfig, BedrockStreamOptions, resolve_client_config,
    resolve_client_config_with_process_env,
};
use pi_ai::types::{
    BoxedFuture, CacheRetention, ConstrainedSamplingConfig, ConstrainedSamplingSetting, Context,
    Message, Model, SimpleStreamOptions, StopReason, Strictness, ThinkingBudgets, ThinkingContent,
    ThinkingLevel, Tool, TransportOptions,
};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

/// The base64 fixture upstream streams as `redactedContent` blobs.
const REDACTED_BASE64: &str = "cnNuXzVaVnJpZjRKMGJYSXFtV2RsZWRqN1FJRmVOaWtSUWJF";
/// The validation failure message the SDK exception fixtures carry.
const VALIDATION_MESSAGE: &str = "The provided model identifier is invalid.";
/// The request id the metadata fixtures carry.
const REQUEST_ID: &str = "11111111-2222-3333-4444-555555555555";

/// What the seam plays for one send, upstream's `send` fixtures.
#[derive(Clone, Debug)]
enum Outcome {
    /// A send failure, upstream's rejecting `send()`.
    Failure(BedrockStreamFailure),
    /// A reply: the response metadata plus the event frames, upstream's
    /// `stream` generator.
    Events {
        /// The response's request id, upstream's `$metadata.requestId`.
        request_id: Option<String>,
        /// The HTTP status, upstream's `$metadata.httpStatusCode`.
        status: Option<u16>,
        /// The frames the iterator yields, mid-stream failures included.
        events: Vec<Result<Value, BedrockStreamFailure>>,
    },
}

/// The seam recorder: one recorded command input and client config per send,
/// upstream's `bedrockMock` fixtures.
#[derive(Debug)]
struct MockBedrockRuntime {
    /// The canned send outcome; the default fails like upstream's
    /// `"mock send"` rejection.
    outcome: Mutex<Option<Outcome>>,
    /// The (command input, client config) pairs per send.
    recorded: Mutex<Vec<(Value, BedrockClientConfig)>>,
}

impl MockBedrockRuntime {
    /// The recorder with an all-`Ok` event run, status 200.
    fn events(events: Vec<Value>) -> Arc<Self> {
        Self::with(Outcome::Events {
            request_id: None,
            status: Some(200),
            events: events.into_iter().map(Ok).collect(),
        })
    }

    /// The recorder with the given canned outcome.
    fn with(outcome: Outcome) -> Arc<Self> {
        Arc::new(Self {
            outcome: Mutex::new(Some(outcome)),
            recorded: Mutex::new(Vec::new()),
        })
    }

    /// The send failure the capture tests drive, upstream's `"mock send"`
    /// rejection.
    fn failing_send() -> Arc<Self> {
        Self::with(Outcome::Failure(BedrockStreamFailure::plain(
            "mock send".to_owned(),
        )))
    }

    /// The command input and client config of the recorded send.
    fn recorded(&self) -> (Value, BedrockClientConfig) {
        self.recorded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .first()
            .cloned()
            .expect("the mock seam recorded one send")
    }
}

impl BedrockRuntime for MockBedrockRuntime {
    fn converse_stream(
        &self,
        input: Value,
        config: BedrockClientConfig,
        _signal: CancellationToken,
    ) -> BoxedFuture<
        'static,
        Result<pi_ai::api::bedrock_converse_stream::BedrockStreamReply, BedrockStreamFailure>,
    > {
        self.recorded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((input, config));
        let outcome = self
            .outcome
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        Box::pin(async move {
            match outcome {
                Some(Outcome::Failure(failure)) => Err(failure),
                Some(Outcome::Events {
                    request_id,
                    status,
                    events,
                }) => Ok(pi_ai::api::bedrock_converse_stream::BedrockStreamReply {
                    request_id,
                    status,
                    events: Box::pin(futures_util::stream::iter(events)),
                }),
                None => Err(BedrockStreamFailure::plain("mock send".to_owned())),
            }
        })
    }
}

/// The options every mock-stream test sends: cache retention off and the
/// recorder as the runtime seam.
fn options_with(runtime: &Arc<MockBedrockRuntime>) -> BedrockStreamOptions {
    let runtime: Arc<dyn BedrockRuntime> = runtime.clone();
    BedrockStreamOptions {
        cache_retention: Some(CacheRetention::None),
        runtime: Some(runtime),
        ..BedrockStreamOptions::default()
    }
}

/// The options the cache-point tests send: cache retention rides the short
/// default so the cache points inject.
fn caching_options(runtime: &Arc<MockBedrockRuntime>) -> BedrockStreamOptions {
    let runtime: Arc<dyn BedrockRuntime> = runtime.clone();
    BedrockStreamOptions {
        runtime: Some(runtime),
        ..BedrockStreamOptions::default()
    }
}

/// The failing-send recorder the payload-capture tests drive.
fn failure_runtime() -> Arc<MockBedrockRuntime> {
    MockBedrockRuntime::failing_send()
}

/// The one-user-message context the stream fixtures run, upstream's
/// `context` literals.
fn context(text: &str) -> Context {
    Context {
        system_prompt: None,
        messages: vec![common::user_message_now(text)],
        tools: None,
    }
}

/// The base64 of the byte array the JSON replay path carries.
fn redacted_bytes() -> Vec<u8> {
    pi_ai::api::bedrock_converse_stream::base64_to_bytes(REDACTED_BASE64)
        .expect("the redacted fixture decodes")
}

// --- upstream bedrock-raw-stop-reason.test.ts ---

/// Raw Bedrock stop reasons survive on successful stops, and provider error
/// stops surface with the upstream wording.
#[tokio::test]
async fn preserves_raw_bedrock_stop_reasons() {
    let model = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");

    let runtime = MockBedrockRuntime::events(vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({ "messageStop": { "stopReason": "end_turn" } }),
    ]);
    let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(message.raw_stop_reason.as_deref(), Some("end_turn"));
    assert!(message.error_message.is_none());

    let runtime = MockBedrockRuntime::events(vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({ "messageStop": { "stopReason": "guardrail_intervened" } }),
    ]);
    let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.raw_stop_reason.as_deref(),
        Some("guardrail_intervened")
    );
    assert_eq!(
        message.error_message.as_deref(),
        Some("Provider stopped with: guardrail_intervened")
    );
}

// --- upstream bedrock-thinking-payload.test.ts ---

/// The model fixture with the catalog base and the test id/name override,
/// upstream's `getModel` plus spread.
fn renamed_model(base: &Model, id: &str, name: &str) -> Model {
    let mut model = base.clone();
    id.clone_into(&mut model.id);
    name.clone_into(&mut model.name);
    model
}

/// The payload the `onPayload` hook captured before the mock send failed,
/// upstream's `capturePayload` helper.
async fn capture_payload(
    model: &Model,
    context: &Context,
    options: &BedrockStreamOptions,
) -> Value {
    let (on_payload, captured) = common::payload_capture();
    let mut options = options.clone();
    options.transport_options.on_payload = Some(on_payload);
    let _ = stream_bedrock(model, context, Some(&options))
        .result()
        .await;
    common::captured_payload(&captured)
}

/// The adaptive-thinking payload pins for Opus 4.8, Opus 5, Sonnet 5, and
/// Fable 5, upstream's first thinking tests.
#[tokio::test]
async fn uses_adaptive_thinking_for_the_claude_5_families() {
    let base = common::builtin_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1");
    let models = [
        renamed_model(
            &base,
            "global.anthropic.claude-opus-4-8-v1",
            "Claude Opus 4.8 (Global)",
        ),
        common::builtin_model("amazon-bedrock", "global.anthropic.claude-fable-5"),
        common::builtin_model("amazon-bedrock", "global.anthropic.claude-sonnet-5"),
        common::builtin_model("amazon-bedrock", "global.anthropic.claude-opus-5"),
    ];

    let mut options = options_with(&failure_runtime());
    options.reasoning = Some(ThinkingLevel::High);
    for model in &models {
        let payload = capture_payload(model, &context("Hello"), &options).await;
        assert_eq!(
            payload.pointer("/additionalModelRequestFields/thinking"),
            Some(&json!({ "type": "adaptive", "display": "summarized" })),
            "{} adaptive thinking",
            model.id
        );
        assert_eq!(
            payload.pointer("/additionalModelRequestFields/output_config"),
            Some(&json!({ "effort": "high" })),
            "{} effort",
            model.id
        );
        assert_eq!(
            payload.pointer("/additionalModelRequestFields/anthropic_beta"),
            None,
            "{} no beta",
            model.id
        );
    }
}

/// `xhigh` maps to the native effort for the models that support it,
/// upstream's xhigh tests.
#[tokio::test]
async fn maps_xhigh_reasoning_to_effort_xhigh_for_native_models() {
    let base = common::builtin_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1");
    let models = [
        renamed_model(
            &base,
            "global.anthropic.claude-opus-4-8-v1",
            "Claude Opus 4.8 (Global)",
        ),
        common::builtin_model("amazon-bedrock", "global.anthropic.claude-opus-5"),
        common::builtin_model("amazon-bedrock", "global.anthropic.claude-fable-5"),
    ];

    let mut options = options_with(&failure_runtime());
    options.reasoning = Some(ThinkingLevel::Xhigh);
    for model in &models {
        let payload = capture_payload(model, &context("Hello"), &options).await;
        assert_eq!(
            payload.pointer("/additionalModelRequestFields/output_config"),
            Some(&json!({ "effort": "xhigh" })),
            "{} native xhigh",
            model.id
        );
    }
}

/// `GovCloud` model ids and regions drop the `display` field; non-adaptive
/// Claude rides the fixed budget plus the interleaved beta.
///
/// Upstream's `GovCloud` tests.
#[tokio::test]
async fn omits_display_for_govcloud_targets() {
    // GovCloud model id on non-adaptive Claude thinking.
    let base = common::builtin_model(
        "amazon-bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    let model = renamed_model(
        &base,
        "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "Claude Sonnet 4.5 (GovCloud)",
    );
    let mut options = options_with(&failure_runtime());
    options.reasoning = Some(ThinkingLevel::High);
    let payload = capture_payload(&model, &context("Hello"), &options).await;
    assert_eq!(
        payload.pointer("/additionalModelRequestFields/thinking"),
        Some(&json!({ "type": "enabled", "budget_tokens": 16384 }))
    );
    assert_eq!(
        payload.pointer("/additionalModelRequestFields/anthropic_beta"),
        Some(&json!(["interleaved-thinking-2025-05-14"]))
    );

    // GovCloud region on adaptive Claude thinking.
    let base = common::builtin_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1");
    let model = renamed_model(
        &base,
        "global.anthropic.claude-opus-4-8-v1",
        "Claude Opus 4.8 (Global)",
    );
    let mut options = options_with(&failure_runtime());
    options.reasoning = Some(ThinkingLevel::High);
    options.region = Some("us-gov-west-1".to_owned());
    let payload = capture_payload(&model, &context("Hello"), &options).await;
    assert_eq!(
        payload.pointer("/additionalModelRequestFields/thinking"),
        Some(&json!({ "type": "adaptive" }))
    );
    assert_eq!(
        payload.pointer("/additionalModelRequestFields/output_config"),
        Some(&json!({ "effort": "high" }))
    );
    assert_eq!(
        payload.pointer("/additionalModelRequestFields/anthropic_beta"),
        None
    );
}

/// Application inference profiles decide off `model.name` when the ARN lacks
/// the model name: adaptive thinking, cache points, and the fixed-budget
/// fallback, upstream's profile tests.
#[tokio::test]
async fn application_inference_profiles_decide_off_model_name() {
    let arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile";
    let base = common::builtin_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1");

    // Adaptive thinking rides the profile name.
    let model = renamed_model(&base, arn, "Claude Opus 4.6");
    let mut options = options_with(&failure_runtime());
    options.reasoning = Some(ThinkingLevel::High);
    let payload = capture_payload(&model, &context("Hello"), &options).await;
    assert_eq!(
        payload.pointer("/additionalModelRequestFields/thinking"),
        Some(&json!({ "type": "adaptive", "display": "summarized" }))
    );
    assert_eq!(
        payload.pointer("/additionalModelRequestFields/output_config"),
        Some(&json!({ "effort": "high" }))
    );

    // Cache points inject through model.name.
    let model = renamed_model(&base, arn, "Claude Sonnet 4.6");
    let caching_context = Context {
        system_prompt: Some("You are helpful.".to_owned()),
        messages: vec![common::user_message_now("Hello")],
        tools: None,
    };
    let payload = capture_payload(
        &model,
        &caching_context,
        &caching_options(&failure_runtime()),
    )
    .await;
    let system = payload
        .get("system")
        .and_then(Value::as_array)
        .expect("system blocks");
    assert_eq!(system.len(), 2);
    assert!(system[1].get("cachePoint").is_some(), "system cache point");
    let last_content = payload
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.last())
        .and_then(|message| message.get("content"))
        .and_then(Value::as_array)
        .and_then(|content| content.last())
        .expect("last message content");
    assert!(last_content.get("cachePoint").is_some(), "user cache point");

    // Fixed-budget thinking falls back through the profile name.
    let sonnet_base = common::builtin_model(
        "amazon-bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    let model = renamed_model(&sonnet_base, arn, "Claude Sonnet 4.5");
    let mut options = options_with(&failure_runtime());
    options.reasoning = Some(ThinkingLevel::High);
    let payload = capture_payload(&model, &context("Hello"), &options).await;
    let thinking = payload
        .pointer("/additionalModelRequestFields/thinking")
        .expect("thinking fields");
    assert_eq!(
        thinking.get("type").and_then(Value::as_str),
        Some("enabled")
    );
    assert!(
        thinking.get("budget_tokens").is_some_and(Value::is_number),
        "a budget rides"
    );
    assert_eq!(
        payload.pointer("/additionalModelRequestFields/anthropic_beta"),
        Some(&json!(["interleaved-thinking-2025-05-14"]))
    );
}

// --- upstream bedrock-convert-messages.test.ts ---

/// The hand-built strict-capable Claude fixture, upstream's `baseModel`.
fn convert_messages_model() -> Model {
    Model {
        id: "us.anthropic.claude-sonnet-4-5-20250929-v1:0".to_owned(),
        name: "Claude Sonnet 4.5 (US)".to_owned(),
        api: pi_ai::types::Api::from("bedrock-converse-stream"),
        provider: pi_ai::types::ProviderId::from("amazon-bedrock"),
        base_url: "https://bedrock-runtime.us-east-1.amazonaws.com".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text, pi_ai::types::Modality::Image],
        cost: pi_ai::types::ModelCost {
            rates: pi_ai::types::ModelCostRates {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
            tiers: None,
        },
        context_window: 200_000,
        max_tokens: 64_000,
        sampling_params: None,
        headers: None,
        compat: Some(pi_ai::types::ModelCompat {
            supports_strict_mode: Some(true),
            ..pi_ai::types::ModelCompat::default()
        }),
    }
}

/// The Nova fixture, upstream's `novaModel`.
fn nova_model() -> Model {
    let mut model = convert_messages_model();
    "amazon.nova-lite-v1:0".clone_into(&mut model.id);
    "Nova Lite".clone_into(&mut model.name);
    model.reasoning = false;
    model.compat = None;
    model
}

/// The lookup tool with the given strictness, upstream's constrained-sampling
/// fixture.
fn strict_lookup_tool(strict: Option<Strictness>) -> Tool {
    Tool {
        name: "lookup".to_owned(),
        description: "Look up a value".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
        }),
        constrained_sampling: strict.map(|strict| {
            ConstrainedSamplingSetting::Config(ConstrainedSamplingConfig::JsonSchema { strict })
        }),
    }
}

/// Native strict tool use gates by model capability, upstream's constrained
/// sampling test.
#[tokio::test]
async fn gates_native_strict_tool_use_by_model_capability() {
    let model = convert_messages_model();
    let nova = nova_model();

    let context = Context {
        system_prompt: None,
        messages: vec![common::user_message_now("Use the tool")],
        tools: Some(vec![strict_lookup_tool(Some(Strictness::Require))]),
    };
    let payload = capture_payload(&model, &context, &options_with(&failure_runtime())).await;
    assert_eq!(
        payload.pointer("/toolConfig/tools/0/toolSpec/strict"),
        Some(&json!(true))
    );

    let context = Context {
        system_prompt: None,
        messages: vec![common::user_message_now("Use the tool")],
        tools: Some(vec![strict_lookup_tool(Some(Strictness::Prefer))]),
    };
    let payload = capture_payload(&nova, &context, &options_with(&failure_runtime())).await;
    assert_eq!(payload.pointer("/toolConfig/tools/0/toolSpec/strict"), None);
}

/// Streamed tool arguments keep empty property names, upstream's tool
/// arguments test.
#[tokio::test]
async fn preserves_empty_property_names_in_streamed_tool_arguments() {
    let model = convert_messages_model();
    let runtime = MockBedrockRuntime::events(vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": { "toolUse": { "toolUseId": "tool-1", "name": "edit" } },
            }
        }),
        json!({
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {
                    "toolUse": {
                        "input": "{\"path\":\"/workspace/foobar/file.js\",\"edits\":[{\"oldText\":\"first\",\"newText\":\"updated first\"},{\"oldText\":\"second\",\"newText\":\"updated second\",\"\":\"\"}]}",
                    }
                },
            }
        }),
        json!({ "contentBlockStop": { "contentBlockIndex": 0 } }),
        json!({ "messageStop": { "stopReason": "tool_use" } }),
    ]);
    let message = stream_bedrock(
        &model,
        &context("Use the tool"),
        Some(&options_with(&runtime)),
    )
    .result()
    .await;

    assert_eq!(message.stop_reason, StopReason::ToolUse);
    assert_eq!(
        serde_json::to_value(&message.content[0]).expect("serializable block"),
        json!({
            "type": "toolCall",
            "id": "tool-1",
            "name": "edit",
            "arguments": {
                "path": "/workspace/foobar/file.js",
                "edits": [
                    { "oldText": "first", "newText": "updated first" },
                    { "oldText": "second", "newText": "updated second", "": "" },
                ],
            },
        })
    );
}

/// Blank user content collapses to the `<empty>` placeholder, and blank
/// blocks drop when other content remains, upstream's placeholder tests.
#[tokio::test]
async fn replaces_blank_user_content_with_the_placeholder() {
    let model = convert_messages_model();

    // A blank user string.
    let context = Context {
        system_prompt: None,
        messages: vec![common::user_message_now("   ")],
        tools: None,
    };
    let payload = capture_payload(&model, &context, &options_with(&failure_runtime())).await;
    assert_eq!(
        payload.pointer("/messages/0/content"),
        Some(&json!([{ "text": "<empty>" }]))
    );

    // Blank blocks drop when another block keeps the message.
    let context = Context {
        system_prompt: None,
        messages: vec![Message::User(pi_ai::types::UserMessage {
            content: pi_ai::types::UserContent::Blocks(vec![
                pi_ai::types::UserBlock::Text(common::text_block("")),
                pi_ai::types::UserBlock::Text(common::text_block("hello")),
            ]),
            timestamp: 1,
        })],
        tools: None,
    };
    let payload = capture_payload(&model, &context, &options_with(&failure_runtime())).await;
    assert_eq!(
        payload.pointer("/messages/0/content"),
        Some(&json!([{ "text": "hello" }]))
    );
}

/// Blank tool results collapse to the placeholder, and assistant turns whose
/// every block filtered drop entirely, upstream's filter tests.
#[tokio::test]
async fn filters_blank_tool_results_and_assistant_turns() {
    let model = convert_messages_model();

    // Blank tool result content becomes the placeholder.
    let context = Context {
        system_prompt: None,
        messages: vec![common::tool_result_message(
            "tool-1",
            vec![pi_ai::types::ToolResultBlock::Text(common::text_block(""))],
            None,
        )],
        tools: None,
    };
    let payload = capture_payload(&model, &context, &options_with(&failure_runtime())).await;
    assert_eq!(
        payload.pointer("/messages/0/content/0/toolResult/content"),
        Some(&json!([{ "text": "<empty>" }]))
    );

    // An assistant turn whose every block filtered drops the message.
    let assistant = common::assistant_message_with_content(
        "bedrock-converse-stream",
        "amazon-bedrock",
        &convert_messages_model().id,
        vec![pi_ai::types::AssistantBlock::Text(common::text_block(""))],
    );
    let context = Context {
        system_prompt: None,
        messages: vec![Message::Assistant(assistant)],
        tools: None,
    };
    let payload = capture_payload(&model, &context, &options_with(&failure_runtime())).await;
    let messages = payload
        .get("messages")
        .and_then(Value::as_array)
        .expect("messages");
    assert!(messages.is_empty(), "the emptied assistant turn skips");
}

/// The empty-key removal only rewrites the replayed Bedrock input, never the
/// caller's message, upstream's `sanitizeBedrockDocument` test.
#[tokio::test]
async fn removes_empty_property_names_only_from_the_replayed_input() {
    let model = convert_messages_model();
    let tool_call = pi_ai::types::ToolCall {
        id: "tool-1".to_owned(),
        name: "edit".to_owned(),
        arguments: serde_json::from_value(json!({
            "path": "/workspace/foobar/file.js",
            "edits": [
                { "oldText": "first", "newText": "updated first" },
                { "oldText": "second", "newText": "updated second", "": "" },
            ],
        }))
        .expect("arguments parse"),
        thought_signature: None,
        namespace: None,
    };
    let assistant = common::assistant_message_with_content(
        "bedrock-converse-stream",
        "amazon-bedrock",
        &model.id,
        vec![pi_ai::types::AssistantBlock::ToolCall(tool_call.clone())],
    );
    let context = Context {
        system_prompt: None,
        messages: vec![
            Message::Assistant(assistant),
            common::tool_result_message(
                "tool-1",
                vec![pi_ai::types::ToolResultBlock::Text(common::text_block(
                    "done",
                ))],
                None,
            ),
            common::user_message_now("Continue"),
        ],
        tools: None,
    };
    let payload = capture_payload(&model, &context, &options_with(&failure_runtime())).await;

    assert_eq!(
        payload.pointer("/messages/0/content/0/toolUse/input"),
        Some(&json!({
            "path": "/workspace/foobar/file.js",
            "edits": [
                { "oldText": "first", "newText": "updated first" },
                { "oldText": "second", "newText": "updated second" },
            ],
        }))
    );
    // The caller's arguments keep the empty key.
    assert_eq!(
        tool_call
            .arguments
            .get("edits")
            .and_then(Value::as_array)
            .and_then(|edits| edits.get(1))
            .and_then(|edit| edit.get("")),
        Some(&json!(""))
    );
}

// --- upstream bedrock-cache-write-1h-cost.test.ts ---

/// The 1h cache details price at 2x input while the total cache write rides
/// verbatim, upstream's pricing regression.
#[expect(
    clippy::suboptimal_flops,
    reason = "upstream's cost arithmetic is kept verbatim; mul_add rounding would differ"
)]
#[tokio::test]
async fn prices_the_1h_cache_details_at_2x() {
    let model = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");
    let runtime = MockBedrockRuntime::events(vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({
            "metadata": {
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 5,
                    "totalTokens": 1_000_105,
                    "cacheWriteInputTokens": 1_000_000,
                    "cacheDetails": [
                        { "ttl": "1h", "inputTokens": 150_000 },
                        { "ttl": "5m", "inputTokens": 600_000 },
                        { "ttl": "1h", "inputTokens": 250_000 },
                    ],
                }
            }
        }),
        json!({ "messageStop": { "stopReason": "end_turn" } }),
    ]);
    let message = stream_bedrock(&model, &context("hi"), Some(&options_with(&runtime)))
        .result()
        .await;

    assert_eq!(message.usage.cache_write, 1_000_000);
    assert_eq!(message.usage.cache_write_1h, Some(400_000));
    let rates = model.cost.rates;
    let expected = (600_000.0 * rates.cache_write + 400_000.0 * rates.input * 2.0) / 1_000_000.0;
    assert!(
        (message.usage.cost.cache_write - expected).abs() < 1e-9,
        "cache write cost {} vs {expected}",
        message.usage.cost.cache_write
    );
    assert_eq!(message.usage.cache_read, 0);
    assert_eq!(message.usage.input, 100);
    assert_eq!(message.usage.output, 5);
}

// --- upstream bedrock-error-metadata.test.ts ---

/// The service exception the SDK's `handleError` path throws, upstream's
/// `makeServiceException`.
fn service_exception(name: &str, status: u16, request_id: Option<&str>) -> BedrockStreamFailure {
    BedrockStreamFailure {
        name: Some(name.to_owned()),
        message: VALIDATION_MESSAGE.to_owned(),
        status: Some(status),
        body: None,
        request_id: request_id.map(str::to_owned),
    }
}

/// The settled message's `bedrock_response_failure` diagnostic.
fn failure_diagnostic(
    message: &pi_ai::types::AssistantMessage,
) -> &pi_ai::types::AssistantMessageDiagnostic {
    message
        .diagnostics
        .as_ref()
        .expect("diagnostics present")
        .iter()
        .find(|diagnostic| diagnostic.kind == "bedrock_response_failure")
        .expect("the failure diagnostic rides")
}

fn detail_entries(entries: &[(&str, Value)]) -> BTreeMap<String, Value> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect()
}

/// A non-2xx from the send carries status, error code, and request id, and
/// `errorMessage` stays byte-identical for the retry classifier.
#[tokio::test]
async fn records_send_failure_diagnostics() {
    let model = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");
    let runtime = MockBedrockRuntime::with(Outcome::Failure(service_exception(
        "ValidationException",
        400,
        Some(REQUEST_ID),
    )));
    let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    let diagnostic = failure_diagnostic(&message);
    assert_eq!(
        diagnostic.details,
        Some(detail_entries(&[
            ("status", json!(400)),
            ("errorCode", json!("ValidationException")),
            ("requestId", json!(REQUEST_ID)),
        ]))
    );
    assert!(diagnostic.error.is_none());
    // The diagnostic carries exactly the type/timestamp/details shape,
    // upstream's key-set assertion.
    let serialized = serde_json::to_value(diagnostic).expect("serializable diagnostic");
    let mut keys: Vec<&str> = serialized
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, vec!["details", "timestamp", "type"]);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Validation error: The provided model identifier is invalid.")
    );
}

/// Mid-stream exceptions report what the frame carried: the bare literal
/// loses the code, the unmodeled error name reports, transport names never
/// do.
#[tokio::test]
async fn mid_stream_failures_report_only_what_the_frame_carried() {
    let model = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");
    let failing_stream = |failure| Outcome::Events {
        request_id: Some(REQUEST_ID.to_owned()),
        status: Some(200),
        events: vec![
            Ok(json!({ "messageStart": { "role": "assistant" } })),
            Err(failure),
        ],
    };

    // A bare exception literal: only the request id survives.
    let runtime = MockBedrockRuntime::with(failing_stream(BedrockStreamFailure {
        name: None,
        message: "Too many requests, please wait.".to_owned(),
        status: None,
        body: None,
        request_id: None,
    }));
    let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
        .result()
        .await;
    assert_eq!(
        failure_diagnostic(&message).details,
        Some(detail_entries(&[("requestId", json!(REQUEST_ID))]))
    );

    // An unmodeled error named after the frame's error code reports.
    let runtime = MockBedrockRuntime::with(failing_stream(BedrockStreamFailure {
        name: Some("ModelStreamErrorException".to_owned()),
        message: "Model stream terminated unexpectedly.".to_owned(),
        status: None,
        body: None,
        request_id: None,
    }));
    let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
        .result()
        .await;
    assert_eq!(
        failure_diagnostic(&message).details,
        Some(detail_entries(&[
            ("errorCode", json!("ModelStreamErrorException")),
            ("requestId", json!(REQUEST_ID)),
        ]))
    );

    // A transport failure name is not a provider error code.
    let runtime = MockBedrockRuntime::with(failing_stream(BedrockStreamFailure {
        name: Some("TimeoutError".to_owned()),
        message: "Connection timed out after 1000 ms".to_owned(),
        status: None,
        body: None,
        request_id: None,
    }));
    let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
        .result()
        .await;
    assert_eq!(
        failure_diagnostic(&message).details,
        Some(detail_entries(&[("requestId", json!(REQUEST_ID))]))
    );

    // No provider metadata at all: no diagnostic.
    let runtime = MockBedrockRuntime::with(Outcome::Failure(BedrockStreamFailure::plain(
        "socket hang up".to_owned(),
    )));
    let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("socket hang up"));
    assert!(message.diagnostics.is_none(), "no diagnostic rides");
}

/// Over-long header-derived values drop; the SDK's `Unknown` placeholder
/// never reports as a code; aborted turns emit no diagnostic.
#[tokio::test]
async fn diagnostic_value_bounds_and_aborted_turns() {
    let model = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");

    // Values past the 200-char bound drop rather than truncate.
    let runtime = MockBedrockRuntime::with(Outcome::Failure(BedrockStreamFailure {
        name: Some(format!("{}Exception", "E".repeat(5000))),
        message: VALIDATION_MESSAGE.to_owned(),
        status: Some(400),
        body: None,
        request_id: Some("R".repeat(5000)),
    }));
    let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
        .result()
        .await;
    assert_eq!(
        failure_diagnostic(&message).details,
        Some(detail_entries(&[("status", json!(400))]))
    );

    // The SDK's Unknown placeholder is omitted from the code.
    let runtime = MockBedrockRuntime::with(Outcome::Failure(service_exception(
        "Unknown",
        403,
        Some(REQUEST_ID),
    )));
    let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
        .result()
        .await;
    assert_eq!(
        failure_diagnostic(&message).details,
        Some(detail_entries(&[
            ("status", json!(403)),
            ("requestId", json!(REQUEST_ID))
        ]))
    );

    // An aborted turn emits no diagnostic even with full metadata.
    let signal = CancellationToken::new();
    signal.cancel();
    let runtime = MockBedrockRuntime::with(Outcome::Failure(service_exception(
        "ValidationException",
        400,
        Some(REQUEST_ID),
    )));
    let mut options = options_with(&runtime);
    options.transport_options = TransportOptions {
        signal: Some(signal),
        ..TransportOptions::default()
    };
    let message = stream_bedrock(&model, &context("hello"), Some(&options))
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert!(message.diagnostics.is_none(), "no diagnostic rides");
}

// --- upstream bedrock-credentials.test.ts ---

/// The ambient `AWS_PROFILE` seam value the profile tests drive, the
/// process-environment leg upstream stubs.
fn ambient_profile(name: &'static str) -> impl Fn(&str) -> Option<String> {
    move |env_name: &str| (env_name == "AWS_PROFILE").then(|| name.to_owned())
}

fn env_of(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect()
}

fn env_options(env: BTreeMap<String, String>) -> BedrockStreamOptions {
    BedrockStreamOptions {
        env: Some(env),
        ..BedrockStreamOptions::default()
    }
}

/// Explicit and scoped profiles beat ambient access keys; ambient profiles
/// leave them in place, upstream's credential-priority tests.
#[test]
fn credential_priority_walks_explicit_scoped_and_ambient_profiles() {
    let model = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");
    let keys = env_of(&[
        ("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE"),
        ("AWS_SECRET_ACCESS_KEY", "secretexample"),
    ]);

    // Explicit profile: the keys stay off the config.
    let mut options = env_options(keys.clone());
    options.profile = Some("explicit-profile".to_owned());
    let config = resolve_client_config(&model, &options);
    assert_eq!(config.profile.as_deref(), Some("explicit-profile"));
    assert!(config.credentials.is_none());

    // Scoped env profile: the keys stay off the config.
    let mut env = keys.clone();
    env.insert("AWS_PROFILE".to_owned(), "scoped-profile".to_owned());
    let config = resolve_client_config(&model, &env_options(env));
    assert_eq!(config.profile.as_deref(), Some("scoped-profile"));
    assert!(config.credentials.is_none());

    // No profile: the ambient access keys ride.
    let config = resolve_client_config(&model, &env_options(keys.clone()));
    assert_eq!(config.profile, None);
    assert_eq!(
        config.credentials.as_ref().map(|credentials| (
            credentials.access_key_id.as_str(),
            credentials.secret_access_key.as_str()
        )),
        Some(("AKIAEXAMPLE", "secretexample"))
    );

    // An ambient profile leaves the keys in place: the SDK chain weighs them
    // together.
    let config = resolve_client_config_with_process_env(
        &model,
        &env_options(keys),
        &ambient_profile("ambient-profile"),
    );
    assert_eq!(config.profile.as_deref(), Some("ambient-profile"));
    assert_eq!(
        config.credentials.as_ref().map(|credentials| (
            credentials.access_key_id.as_str(),
            credentials.secret_access_key.as_str()
        )),
        Some(("AKIAEXAMPLE", "secretexample"))
    );
}

// --- upstream bedrock-endpoint-resolution.test.ts ---

/// The endpoint-resolution table: EU catalog URLs, the region suppression,
/// the ARN extraction, and the bearer-token path, upstream's endpoint tests.
#[test]
fn endpoint_resolution_walks_regions_profiles_and_arns() {
    let eu = common::builtin_model(
        "amazon-bedrock",
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    );
    let us = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");

    // EU inference profiles pin the EU runtime URL.
    assert_eq!(
        eu.base_url,
        "https://bedrock-runtime.eu-central-1.amazonaws.com"
    );

    // AWS_REGION suppresses endpoint pinning on standard hosts.
    let config = resolve_client_config(&us, &env_options(env_of(&[("AWS_REGION", "us-east-2")])));
    assert_eq!(config.region.as_deref(), Some("us-east-2"));
    assert!(config.endpoint.is_none());

    // The EU endpoint derives the region when nothing else configures one.
    let config = resolve_client_config(&eu, &BedrockStreamOptions::default());
    assert_eq!(
        config.endpoint.as_deref(),
        Some("https://bedrock-runtime.eu-central-1.amazonaws.com")
    );
    assert_eq!(config.region.as_deref(), Some("eu-central-1"));

    // Explicit and scoped profiles keep the pinned endpoint; an ambient
    // profile drops it.
    let options = BedrockStreamOptions {
        profile: Some("bedrock-profile".to_owned()),
        ..BedrockStreamOptions::default()
    };
    let config = resolve_client_config(&eu, &options);
    assert_eq!(config.profile.as_deref(), Some("bedrock-profile"));
    assert_eq!(
        config.endpoint.as_deref(),
        Some("https://bedrock-runtime.eu-central-1.amazonaws.com")
    );
    assert_eq!(config.region.as_deref(), Some("eu-central-1"));

    let options = BedrockStreamOptions {
        env: Some(env_of(&[("AWS_PROFILE", "scoped-bedrock-profile")])),
        ..BedrockStreamOptions::default()
    };
    let config = resolve_client_config(&eu, &options);
    assert_eq!(config.profile.as_deref(), Some("scoped-bedrock-profile"));
    assert_eq!(
        config.endpoint.as_deref(),
        Some("https://bedrock-runtime.eu-central-1.amazonaws.com")
    );
    assert_eq!(config.region.as_deref(), Some("eu-central-1"));

    let config = resolve_client_config_with_process_env(
        &eu,
        &BedrockStreamOptions::default(),
        &ambient_profile("ambient-bedrock-profile"),
    );
    assert_eq!(config.profile.as_deref(), Some("ambient-bedrock-profile"));
    assert!(config.endpoint.is_none());
    assert!(config.region.is_none());

    // Custom Bedrock endpoints pass through with the configured region.
    let mut custom = us.clone();
    custom.base_url = "https://bedrock-vpc.example.com".to_owned();
    let config = resolve_client_config(
        &custom,
        &env_options(env_of(&[("AWS_REGION", "us-west-2")])),
    );
    assert_eq!(
        config.endpoint.as_deref(),
        Some("https://bedrock-vpc.example.com")
    );
    assert_eq!(config.region.as_deref(), Some("us-west-2"));

    // ARN-embedded regions win over AWS_REGION, commercial and GovCloud.
    let mut arn = us.clone();
    arn.id =
        "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123".to_owned();
    let config = resolve_client_config(&arn, &env_options(env_of(&[("AWS_REGION", "us-east-1")])));
    assert_eq!(config.region.as_deref(), Some("us-west-2"));

    let mut gov = us.clone();
    gov.id =
        "arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:application-inference-profile/abc123"
            .to_owned();
    let config = resolve_client_config(&gov, &env_options(env_of(&[("AWS_REGION", "us-east-1")])));
    assert_eq!(config.region.as_deref(), Some("us-gov-west-1"));

    // An ambient profile rides custom model ids without bearer auth.
    let mut profile_arn = us.clone();
    profile_arn.id =
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/example".to_owned();
    let config = resolve_client_config_with_process_env(
        &profile_arn,
        &BedrockStreamOptions::default(),
        &ambient_profile("bedrock-profile"),
    );
    assert_eq!(config.profile.as_deref(), Some("bedrock-profile"));
    assert!(config.token.is_none());
    assert!(config.auth_scheme_preference.is_none());

    // The generic API-key option becomes the Bedrock bearer token.
    let options = BedrockStreamOptions {
        api_key: Some("bedrock-api-key".to_owned()),
        ..BedrockStreamOptions::default()
    };
    let config = resolve_client_config(&us, &options);
    assert_eq!(config.token.as_deref(), Some("bedrock-api-key"));
    assert_eq!(
        config.auth_scheme_preference.as_deref(),
        Some("httpBearerAuth")
    );
}

// --- upstream bedrock-custom-headers.test.ts ---

/// Reserved headers skip case-insensitively while allowed ones ride,
/// upstream's VC1/VC2.
#[test]
fn custom_headers_skip_reserved_names_case_insensitively() {
    let model = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");
    let options = BedrockStreamOptions {
        headers: Some(BTreeMap::from([
            ("authorization".to_owned(), Some("evil".to_owned())),
            ("x-amz-date".to_owned(), Some("evil".to_owned())),
            ("x-allowed".to_owned(), Some("ok".to_owned())),
            ("Authorization".to_owned(), Some("evil2".to_owned())),
            ("X-Amz-Date".to_owned(), Some("evil2".to_owned())),
            ("HOST".to_owned(), Some("evil3".to_owned())),
        ])),
        ..BedrockStreamOptions::default()
    };
    let config = resolve_client_config(&model, &options);
    assert_eq!(
        config.headers,
        vec![("x-allowed".to_owned(), "ok".to_owned())]
    );
}

/// Caller headers ride the pre-signing config, and empty header sets add no
/// entries, upstream's VC3.
#[tokio::test]
async fn custom_headers_ride_and_empty_sets_add_nothing() {
    let model = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");

    // A caller header rides the recorded client config.
    let runtime = failure_runtime();
    let mut options = options_with(&runtime);
    options.headers = Some(BTreeMap::from([(
        "x-custom".to_owned(),
        Some("v".to_owned()),
    )]));
    let _ = stream_bedrock(&model, &context("hello"), Some(&options))
        .result()
        .await;
    let (_, config) = runtime.recorded();
    assert_eq!(
        config.headers,
        vec![("x-custom".to_owned(), "v".to_owned())]
    );

    // An empty header set adds no entries.
    let config = resolve_client_config(
        &model,
        &BedrockStreamOptions {
            headers: Some(BTreeMap::new()),
            ..BedrockStreamOptions::default()
        },
    );
    assert!(config.headers.is_empty(), "no middleware entries");
}

/// `streamSimple` forwards caller headers end-to-end, upstream's VC4
/// regression guard, through the option mapping the simple entry builds.
#[test]
fn stream_simple_forwards_headers_end_to_end() {
    let model = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");
    let simple = SimpleStreamOptions {
        headers: Some(BTreeMap::from([(
            "x-custom".to_owned(),
            Some("v".to_owned()),
        )])),
        ..SimpleStreamOptions::default()
    };
    let base = BedrockStreamOptions::from(pi_ai::api::simple_options::build_base_options(
        &model,
        &context("hello"),
        Some(&simple),
        None,
    ));
    let config = resolve_client_config(&model, &base);
    assert_eq!(
        config.headers,
        vec![("x-custom".to_owned(), "v".to_owned())]
    );
}

// --- upstream bedrock-redacted-reasoning.test.ts ---

/// The GPT-5.6 Terra fixture, upstream's `gptModel`.
fn gpt_model() -> Model {
    Model {
        id: "global.openai.gpt-5.6-terra".to_owned(),
        name: "GPT-5.6 Terra (Global)".to_owned(),
        api: pi_ai::types::Api::from("bedrock-converse-stream"),
        provider: pi_ai::types::ProviderId::from("amazon-bedrock"),
        base_url: "https://bedrock-runtime.ap-northeast-1.amazonaws.com".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: pi_ai::types::ModelCost {
            rates: pi_ai::types::ModelCostRates {
                input: 1.25,
                output: 10.0,
                cache_read: 0.125,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 400_000,
        max_tokens: 128_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// Mirrors the `ConverseStream` frames GPT-5.6 emits: encrypted reasoning,
/// then text, upstream's `redactedReasoningEvents`.
fn redacted_reasoning_events() -> Vec<Value> {
    vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": { "reasoningContent": { "redactedContent": REDACTED_BASE64 } },
            }
        }),
        json!({ "contentBlockStop": { "contentBlockIndex": 0 } }),
        json!({ "contentBlockDelta": { "contentBlockIndex": 1, "delta": { "text": "done" } } }),
        json!({ "contentBlockStop": { "contentBlockIndex": 1 } }),
        json!({ "messageStop": { "stopReason": "end_turn" } }),
    ]
}

/// The redacted thinking block the response carries.
fn thinking_block(message: &pi_ai::types::AssistantMessage) -> &ThinkingContent {
    let Some(pi_ai::types::AssistantBlock::Thinking(thinking)) = message
        .content
        .iter()
        .find(|block| matches!(block, pi_ai::types::AssistantBlock::Thinking(_)))
    else {
        unreachable!("expected a thinking block");
    };
    thinking
}

/// Redacted reasoning streams through without failing, the payload rides
/// `thinkingSignature`, and a missing stop still flushes.
#[tokio::test]
async fn redacted_reasoning_streams_and_flushes() {
    let model = gpt_model();

    // The stream settles with reasoning preceding the answer.
    let runtime = MockBedrockRuntime::events(redacted_reasoning_events());
    let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
        .result()
        .await;
    assert_ne!(
        message.stop_reason,
        StopReason::Error,
        "{}",
        message.error_message.clone().unwrap_or_default()
    );
    let kinds: Vec<&str> = message
        .content
        .iter()
        .map(|block| match block {
            pi_ai::types::AssistantBlock::Text(_) => "text",
            pi_ai::types::AssistantBlock::Thinking(_) => "thinking",
            pi_ai::types::AssistantBlock::ToolCall(_) => "toolCall",
        })
        .collect();
    assert_eq!(kinds, vec!["thinking", "text"]);
    assert_eq!(
        serde_json::to_value(&message.content[1]).expect("serializable"),
        json!({ "type": "text", "text": "done" })
    );

    // The opaque payload rides thinkingSignature with redacted: true, the
    // placeholder marks the block once, and no scratch buffer persists.
    let thinking = thinking_block(&message);
    assert_eq!(thinking.redacted, Some(true));
    assert_eq!(
        thinking.thinking_signature.as_deref(),
        Some(REDACTED_BASE64)
    );
    assert_eq!(thinking.thinking, "[Reasoning redacted]");

    // No contentBlockStop: the finalize path flushes the same payload.
    let runtime = MockBedrockRuntime::events(vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": { "reasoningContent": { "redactedContent": REDACTED_BASE64 } },
            }
        }),
        json!({ "messageStop": { "stopReason": "end_turn" } }),
    ]);
    let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
        .result()
        .await;
    let thinking = thinking_block(&message);
    assert_eq!(
        thinking.thinking_signature.as_deref(),
        Some(REDACTED_BASE64)
    );
}

/// Encrypted reasoning joins across deltas, and the placeholder marks the
/// block exactly once, upstream's split-deltas test.
#[tokio::test]
async fn joins_redacted_reasoning_across_deltas() {
    let model = gpt_model();
    let bytes = redacted_bytes();
    let (head, tail) = (&bytes[..7], &bytes[7..]);
    let runtime = MockBedrockRuntime::events(vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": { "reasoningContent": { "redactedContent":
                    pi_ai::api::bedrock_converse_stream::bytes_to_base64(head) } },
            }
        }),
        json!({
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": { "reasoningContent": { "redactedContent":
                    pi_ai::api::bedrock_converse_stream::bytes_to_base64(tail) } },
            }
        }),
        json!({ "contentBlockStop": { "contentBlockIndex": 0 } }),
        json!({ "messageStop": { "stopReason": "end_turn" } }),
    ]);
    let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
        .result()
        .await;
    let thinking = thinking_block(&message);
    assert_eq!(
        thinking.thinking_signature.as_deref(),
        Some(REDACTED_BASE64)
    );
    assert_eq!(thinking.thinking, "[Reasoning redacted]");
}

/// Replayed redacted reasoning rides `reasoningContent.redactedContent`,
/// ahead of the text or toolUse block it belongs to, upstream's replay
/// tests.
#[tokio::test]
async fn replays_redacted_reasoning_as_redacted_content() {
    let model = gpt_model();
    let assistant_text = common::assistant_message_with_content(
        "bedrock-converse-stream",
        "amazon-bedrock",
        &model.id,
        vec![
            pi_ai::types::AssistantBlock::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some(REDACTED_BASE64.to_owned()),
                redacted: Some(true),
            }),
            pi_ai::types::AssistantBlock::Text(common::text_block("done")),
        ],
    );
    let tool_call = pi_ai::types::ToolCall {
        id: "tool-1".to_owned(),
        name: "read".to_owned(),
        arguments: serde_json::from_value(json!({ "path": "/tmp/a.txt" })).expect("arguments"),
        thought_signature: None,
        namespace: None,
    };
    let assistant_tool = common::assistant_message_with_content(
        "bedrock-converse-stream",
        "amazon-bedrock",
        &model.id,
        vec![
            pi_ai::types::AssistantBlock::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some(REDACTED_BASE64.to_owned()),
                redacted: Some(true),
            }),
            pi_ai::types::AssistantBlock::ToolCall(tool_call),
        ],
    );

    // Text continuation.
    let context = Context {
        system_prompt: None,
        messages: vec![
            common::user_message_now("hello"),
            Message::Assistant(assistant_text),
            common::user_message_now("continue"),
        ],
        tools: None,
    };
    let payload = capture_payload(&model, &context, &options_with(&failure_runtime())).await;
    let assistant = payload
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.iter().find(|m| m["role"] == "assistant"))
        .expect("assistant turn");
    assert_eq!(
        assistant["content"],
        json!([
            { "reasoningContent": { "redactedContent": json!(redacted_bytes()) } },
            { "text": "done" },
        ])
    );

    // Tool continuation, the payload ahead of the toolUse.
    let context = Context {
        system_prompt: None,
        messages: vec![
            common::user_message_now("read the file"),
            Message::Assistant(assistant_tool),
            common::tool_result_message(
                "tool-1",
                vec![pi_ai::types::ToolResultBlock::Text(common::text_block(
                    "file body",
                ))],
                None,
            ),
        ],
        tools: None,
    };
    let payload = capture_payload(&model, &context, &options_with(&failure_runtime())).await;
    let assistant = payload
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.iter().find(|m| m["role"] == "assistant"))
        .expect("assistant turn");
    assert_eq!(
        assistant["content"],
        json!([
            { "reasoningContent": { "redactedContent": json!(redacted_bytes()) } },
            { "toolUse": { "toolUseId": "tool-1", "name": "read", "input": { "path": "/tmp/a.txt" } } },
        ])
    );
}

// --- the format rule the error-metadata suite rides on ---

/// The failure formatting table: the body surface, the data-retention hint,
/// and the prefix rule, the contracts the retry and overflow classifiers
/// match on, upstream's `formatBedrockError`.
#[test]
fn formats_bedrock_errors() {
    // A gateway 403 surfaces its body instead of the SDK placeholder.
    assert_eq!(
        format_bedrock_error(
            Some("Unknown"),
            "UnknownError",
            Some(403),
            Some("forbidden")
        ),
        "403: forbidden"
    );
    // A body the message already carries stays in the message.
    assert_eq!(
        format_bedrock_error(None, "gateway said 403: denied", Some(403), Some("denied")),
        "gateway said 403: denied"
    );
    // A data-retention failure points at the docs.
    let formatted = format_bedrock_error(
        Some("ValidationException"),
        "data retention mode 'default' is not available for this model",
        None,
        None,
    );
    assert!(formatted.starts_with("Validation error: "), "{formatted}");
    assert!(
        formatted.ends_with(
            " See https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html for supported data retention modes."
        ),
        "{formatted}"
    );
    // A transport name never gains a prefix.
    assert_eq!(
        format_bedrock_error(Some("TimeoutError"), "socket hang up", None, None),
        "socket hang up"
    );
}

// --- the modeled exception frames, image gates, budgets, and the simple entry ---

/// The five modeled exception frames throw into the catch path with the
/// stable prefix wording the retry and overflow classifiers match on.
#[tokio::test]
async fn modeled_exception_frames_surface_with_the_prefix_wording() {
    let model = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");
    for (key, name, prefix) in [
        (
            "internalServerException",
            "InternalServerException",
            "Internal server error",
        ),
        (
            "modelStreamErrorException",
            "ModelStreamErrorException",
            "Model stream error",
        ),
        (
            "validationException",
            "ValidationException",
            "Validation error",
        ),
        (
            "throttlingException",
            "ThrottlingException",
            "Throttling error",
        ),
        (
            "serviceUnavailableException",
            "ServiceUnavailableException",
            "Service unavailable",
        ),
    ] {
        let runtime = MockBedrockRuntime::events(vec![
            json!({ "messageStart": { "role": "assistant" } }),
            json!({ key: { "name": name, "message": "the model failed" } }),
        ]);
        let message = stream_bedrock(&model, &context("hello"), Some(&options_with(&runtime)))
            .result()
            .await;
        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(
            message.error_message.as_deref(),
            Some(format!("{prefix}: the model failed").as_str()),
            "{key}"
        );
    }
}

/// The image blocks gate the MIME type and the base64 data with the
/// upstream wording, upstream's `createImageBlock` throws.
#[tokio::test]
async fn image_blocks_gate_the_mime_type_and_the_base64_data() {
    let model = convert_messages_model();
    let image_context = |data: &str, mime_type: &str| Context {
        system_prompt: None,
        messages: vec![Message::User(pi_ai::types::UserMessage {
            content: pi_ai::types::UserContent::Blocks(vec![pi_ai::types::UserBlock::Image(
                pi_ai::types::ImageContent {
                    data: data.to_owned(),
                    mime_type: mime_type.to_owned(),
                },
            )]),
            timestamp: 1,
        })],
        tools: None,
    };

    let failure = pi_ai::api::bedrock_converse_stream::convert_messages(
        &image_context("AAAA", "image/tiff"),
        &model,
        CacheRetention::None,
        None,
    )
    .expect_err("the unknown mime type");
    assert_eq!(failure, "Unknown image type: image/tiff");

    let failure = pi_ai::api::bedrock_converse_stream::convert_messages(
        &image_context("!!!", "image/png"),
        &model,
        CacheRetention::None,
        None,
    )
    .expect_err("the invalid base64");
    assert_eq!(failure, "Invalid base64 image data: image/png");
}

/// The fixed-budget table: the extended levels clamp to the high budget and
/// the caller budgets override, upstream's default table plus override.
#[test]
fn the_budget_table_clamps_the_extended_levels_and_the_custom_budgets_override() {
    let base = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");
    let model = renamed_model(
        &base,
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "Claude 3.7 Sonnet",
    );
    let fields = |reasoning: Option<ThinkingLevel>, budgets: Option<&ThinkingBudgets>| {
        pi_ai::api::bedrock_converse_stream::build_additional_model_request_fields(
            &model, reasoning, budgets, None, None, None,
        )
        .expect("the claude thinking fields")
    };
    let budget_tokens = |fields: &Value| {
        fields
            .pointer("/thinking/budget_tokens")
            .and_then(Value::as_u64)
            .expect("the budget tokens")
    };

    assert_eq!(
        budget_tokens(&fields(Some(ThinkingLevel::Minimal), None)),
        1024
    );
    assert_eq!(budget_tokens(&fields(Some(ThinkingLevel::Low), None)), 2048);
    assert_eq!(
        budget_tokens(&fields(Some(ThinkingLevel::Medium), None)),
        8192
    );
    assert_eq!(
        budget_tokens(&fields(Some(ThinkingLevel::High), None)),
        16384
    );
    assert_eq!(
        budget_tokens(&fields(Some(ThinkingLevel::Xhigh), None)),
        16384,
        "xhigh clamps to the high budget"
    );
    assert_eq!(
        budget_tokens(&fields(Some(ThinkingLevel::Max), None)),
        16384,
        "max clamps to the high budget"
    );

    let custom = ThinkingBudgets {
        high: Some(2048),
        minimal: Some(128),
        ..ThinkingBudgets::default()
    };
    assert_eq!(
        budget_tokens(&fields(Some(ThinkingLevel::High), Some(&custom))),
        2048,
        "the caller budget overrides"
    );
    assert_eq!(
        budget_tokens(&fields(Some(ThinkingLevel::Xhigh), Some(&custom))),
        2048,
        "the extended levels ride the clamped level's row"
    );
    assert_eq!(
        budget_tokens(&fields(Some(ThinkingLevel::Minimal), Some(&custom))),
        128,
        "the minimal row overrides"
    );
}

/// The loopback simple-entry model: the pinned refused endpoint keeps the
/// real SDK adapter send a local failure, upstream's credential-free test
/// client construction.
fn loopback_simple_model(base: &Model, id: &str, name: &str) -> Model {
    let mut model = renamed_model(base, id, name);
    "http://127.0.0.1:9".clone_into(&mut model.base_url);
    model
}

/// The captured simple-entry payload and the settled message.
async fn capture_simple_stream(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
) -> (Value, pi_ai::types::AssistantMessage) {
    let (on_payload, captured) = common::payload_capture();
    let mut options = options.clone();
    options.transport_options.on_payload = Some(on_payload);
    let message =
        pi_ai::api::bedrock_converse_stream::stream_simple(model, context, Some(&options))
            .result()
            .await;
    (common::captured_payload(&captured), message)
}

/// The skip-auth env fixture the simple-entry streams resolve with.
fn simple_options(
    tool_choice: Option<pi_ai::types::ToolChoice>,
    reasoning: Option<ThinkingLevel>,
) -> SimpleStreamOptions {
    SimpleStreamOptions {
        transport_options: TransportOptions::default(),
        env: Some(BTreeMap::from([
            ("AWS_BEDROCK_SKIP_AUTH".to_owned(), "1".to_owned()),
            ("AWS_REGION".to_owned(), "us-east-1".to_owned()),
            ("NO_PROXY".to_owned(), "*".to_owned()),
        ])),
        tool_choice,
        reasoning,
        ..SimpleStreamOptions::default()
    }
}

/// The simple entry maps the shared tool choices onto the Bedrock choices
/// and settles the failed loopback send as an error, upstream's
/// `streamSimple` toolChoice narrowing.
#[tokio::test]
async fn the_simple_stream_maps_the_shared_tool_choices() {
    let base = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");
    let model = loopback_simple_model(&base, "us.anthropic.claude-opus-4-8", "Claude Opus 4.8");
    let tool = strict_lookup_tool(None);
    let context = Context {
        system_prompt: None,
        messages: vec![common::user_message_now("hello")],
        tools: Some(vec![tool]),
    };

    let (payload, message) = capture_simple_stream(
        &model,
        &context,
        &simple_options(
            Some(pi_ai::types::ToolChoice::Auto),
            Some(ThinkingLevel::High),
        ),
    )
    .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        payload.pointer("/toolConfig/toolChoice"),
        Some(&json!({ "auto": {} })),
        "the Auto choice maps to the Bedrock auto form"
    );

    let (payload, message) = capture_simple_stream(
        &model,
        &context,
        &simple_options(Some(pi_ai::types::ToolChoice::None), None),
    )
    .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(
        payload.get("toolConfig").is_none(),
        "the None choice drops the tool config: {payload}"
    );
}

/// The simple entry's claude budget path: the non-adaptive claude model
/// rides the fixed-budget table with the model cap maxTokens, and the
/// adaptive families pass the level through.
#[tokio::test]
async fn the_simple_stream_claude_paths_shape_the_thinking_fields() {
    let base = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");

    // The non-adaptive claude rides the fixed-budget table.
    let (payload, message) = capture_simple_stream(
        &loopback_simple_model(
            &base,
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "Claude 3.7 Sonnet",
        ),
        &context("hello"),
        &simple_options(None, Some(ThinkingLevel::High)),
    )
    .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        payload.pointer("/additionalModelRequestFields/thinking/type"),
        Some(&json!("enabled"))
    );
    assert!(
        payload
            .pointer("/additionalModelRequestFields/thinking/budget_tokens")
            .and_then(Value::as_u64)
            .is_some(),
        "the budget rides: {payload}"
    );

    // The adaptive family passes the level through verbatim.
    let (payload, _message) = capture_simple_stream(
        &loopback_simple_model(&base, "us.anthropic.claude-opus-4-8", "Claude Opus 4.8"),
        &context("hello"),
        &simple_options(None, Some(ThinkingLevel::Medium)),
    )
    .await;
    assert_eq!(
        payload.pointer("/additionalModelRequestFields/thinking"),
        Some(&json!({ "type": "adaptive", "display": "summarized" }))
    );

    // The non-claude model carries no additional fields.
    let nova = loopback_simple_model(&nova_model(), "amazon.nova-lite-v1:0", "Nova Lite");
    let (payload, _message) = capture_simple_stream(
        &nova,
        &context("hello"),
        &simple_options(None, Some(ThinkingLevel::High)),
    )
    .await;
    assert_eq!(
        payload.get("additionalModelRequestFields"),
        None,
        "the non-claude model carries no thinking fields"
    );
}

/// The public client-config wrappers resolve through the process-env seam
/// with the option fields, upstream's option plumb-through.
#[test]
fn the_client_config_wrappers_resolve_the_option_fields() {
    let model = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");
    let options = BedrockStreamOptions {
        region: Some("us-west-2".to_owned()),
        profile: Some("bedrock-profile".to_owned()),
        headers: Some(BTreeMap::from([
            ("x-custom".to_owned(), Some("v".to_owned())),
            ("authorization".to_owned(), None),
        ])),
        bearer_token: Some("bearer".to_owned()),
        api_key: Some("api-key".to_owned()),
        cache_retention: Some(CacheRetention::Long),
        ..BedrockStreamOptions::default()
    };

    assert_eq!(
        pi_ai::api::bedrock_options::resolve_region(&model, &options),
        "us-west-2"
    );
    // A standard runtime host with a configured region rides ambient, so no
    // explicit endpoint is pinned.
    assert_eq!(
        pi_ai::api::bedrock_options::resolve_endpoint(&model, &options),
        None,
        "the standard runtime host rides the region-built endpoint"
    );

    // A non-standard base URL pins as the explicit endpoint.
    let pinned = Model {
        base_url: "https://proxy.example.com".to_owned(),
        ..model.clone()
    };
    assert_eq!(
        pi_ai::api::bedrock_options::resolve_endpoint(&pinned, &options),
        Some("https://proxy.example.com".to_owned()),
        "the non-standard base URL pins"
    );

    // The scoped profile beats the ambient AWS_PROFILE chain.
    let scoped = BedrockStreamOptions {
        env: Some(BTreeMap::from([(
            "AWS_PROFILE".to_owned(),
            "scoped-profile".to_owned(),
        )])),
        ..BedrockStreamOptions::default()
    };
    assert_eq!(
        pi_ai::api::bedrock_options::resolve_profile(&scoped),
        Some("scoped-profile".to_owned()),
        "the provider-scoped profile rides"
    );

    let config = resolve_client_config(&model, &options);
    assert_eq!(config.region.as_deref(), Some("us-west-2"));
    assert_eq!(
        config.token.as_deref(),
        Some("bearer"),
        "the explicit bearer token beats the api key"
    );

    // Without the explicit bearer token, the api key doubles as one.
    let config = resolve_client_config(
        &model,
        &BedrockStreamOptions {
            bearer_token: None,
            api_key: Some("api-key".to_owned()),
            ..BedrockStreamOptions::default()
        },
    );
    assert_eq!(config.token.as_deref(), Some("api-key"));

    // The cache retention resolution honors the long opt-in.
    let options = BedrockStreamOptions {
        cache_retention: Some(CacheRetention::Long),
        env: Some(BTreeMap::from([(
            "PI_CACHE_RETENTION".to_owned(),
            "long".to_owned(),
        )])),
        ..BedrockStreamOptions::default()
    };
    assert_eq!(
        pi_ai::api::bedrock_options::resolve_cache_retention(&options),
        CacheRetention::Long
    );
}

/// The claude thinking replay keeps its signature, upstream's
/// `supportsThinkingSignature` gate.
#[tokio::test]
async fn the_claude_thinking_replay_keeps_its_signature() {
    let model = common::builtin_model("amazon-bedrock", "us.anthropic.claude-opus-4-8");
    let context = Context {
        system_prompt: None,
        messages: vec![Message::Assistant(pi_ai::types::AssistantMessage {
            content: vec![pi_ai::types::AssistantBlock::Thinking(ThinkingContent {
                thinking: "ponder".to_owned(),
                thinking_signature: Some("sig-1".to_owned()),
                redacted: None,
            })],
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            diagnostics: None,
            usage: pi_ai::types::Usage::default(),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 1,
        })],
        tools: None,
    };

    let wire = pi_ai::api::bedrock_converse_stream::convert_messages(
        &context,
        &model,
        CacheRetention::None,
        None,
    )
    .expect("the claude replay converts");

    let reasoning = &wire[0]["content"][0]["reasoningContent"]["reasoningText"];
    assert_eq!(reasoning["text"], json!("ponder"));
    assert_eq!(reasoning["signature"], json!("sig-1"));
}

/// The seam reply debugs with its metadata fields only.
#[test]
fn the_stream_reply_debugs_with_its_metadata() {
    let reply = pi_ai::api::bedrock_converse_stream::BedrockStreamReply {
        request_id: Some(REQUEST_ID.to_owned()),
        status: Some(200),
        events: Box::pin(futures_util::stream::iter(Vec::new())),
    };
    let debug = format!("{reply:?}");
    assert!(debug.contains("request_id"), "{debug}");
    assert!(debug.contains("status: Some(200)"), "{debug}");
}

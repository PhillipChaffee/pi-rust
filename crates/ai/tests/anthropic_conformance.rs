//! Anthropic conformance suites, ported from the upstream
//! `anthropic-*-compat`, `anthropic-thinking-*`,
//! `anthropic-mid-conversation-effort`, and
//! `anthropic-adaptive-thinking-models` test files at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream captured payloads through throwing `onPayload` hooks and a
//! loopback HTTP server; the port records the payload through the
//! [`OnPayload`] hook and the request through the [`MockHttpClient`] seam.

#![expect(
    clippy::expect_used,
    reason = "the tests pin stream outcomes; an unexpected shape panics the test by design"
)]

use std::sync::Arc;

use pi_ai::api::anthropic_messages::{
    AnthropicEffort, AnthropicStreamOptions, stream, stream_simple,
};
use pi_ai::http::MockHttpClient;
use pi_ai::types::{
    AssistantBlock, CacheRetention, Context, Message, Modality, Model, ModelCompat,
    SimpleStreamOptions, StopReason, TextContent, ThinkingLevel, Tool,
};
use serde_json::{Value, json};

mod common;
use common::{builtin_model, mock_transport, payload_capture, user_message_now};

// ---------------------------------------------------------------------------
// Adaptive thinking model metadata (upstream anthropic-adaptive-thinking-models)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn marks_builtin_anthropic_messages_models_that_use_adaptive_thinking() {
    let flagged_models: Vec<String> = pi_ai::providers::all::builtin_providers()
        .iter()
        .flat_map(|provider| provider.get_models().unwrap_or_default())
        .filter(|model| model.api == pi_ai::types::Api::from("anthropic-messages"))
        .filter(|model| {
            model
                .compat
                .as_ref()
                .and_then(|compat| compat.force_adaptive_thinking)
                == Some(true)
        })
        .map(|model| format!("{}/{}", model.provider.0, model.id))
        .collect();

    let expected = [
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-4-8",
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",
        "cloudflare-ai-gateway/claude-fable-5",
        "fireworks/accounts/fireworks/models/deepseek-v4-flash-0731",
        "fireworks/accounts/fireworks/models/gpt-oss-120b",
        "fireworks/accounts/fireworks/models/qwen3p8-max",
        "kimi-coding/kimi-for-coding",
        "kimi-coding/k3",
        "kimi-coding/kimi-for-coding-highspeed",
        "opencode/claude-opus-4-8",
        "opencode/claude-opus-5",
        "vercel-ai-gateway/anthropic/claude-opus-4.8",
        "vercel-ai-gateway/anthropic/claude-opus-5",
        "vercel-ai-gateway/anthropic/claude-sonnet-5",
    ];
    for model_id in expected {
        assert!(
            flagged_models.iter().any(|flagged| flagged == model_id),
            "expected {model_id} among {flagged_models:?}"
        );
    }
    // Regression for pi #9323: Fireworks uses catalog effort metadata and
    // verified fallbacks, not a fixed set of adaptive model names.
    let family = regex::Regex::new(
        r"opus[-.](4[-.][678]|5)|sonnet[-.]4[-.]6|sonnet[-.]5|fable[-.]5|kimi-coding/",
    )
    .expect("the family filter regex");
    let outside_families: Vec<&String> = flagged_models
        .iter()
        .filter(|model_id| !(model_id.starts_with("fireworks/") || family.is_match(model_id)))
        .collect();
    assert!(
        outside_families.is_empty(),
        "adaptive flags outside the expected families: {outside_families:?}"
    );
}

// ---------------------------------------------------------------------------
// 1h cache write cost (upstream anthropic-cache-write-1h-cost)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn prices_the_1h_portion_at_2x_input_and_the_rest_at_the_5m_rate() {
    let (cost, cache_write_1h, cache_write) = stream_cache_write_1h_cost(Some(json!({
        "ephemeral_5m_input_tokens": 600_000,
        "ephemeral_1h_input_tokens": 400_000,
    })))
    .await;
    assert_eq!(cache_write, 1_000_000);
    assert_eq!(cache_write_1h, 400_000);
    // 600k * 6.25/Mtok + 400k * 10/Mtok = 3.75 + 4.0 = 7.75
    assert!((cost - 7.75).abs() < 1e-10, "got: {cost}");
}

#[tokio::test]
async fn falls_back_to_the_5m_rate_when_no_breakdown_is_reported() {
    let (cost, cache_write_1h, cache_write) = stream_cache_write_1h_cost(None).await;
    assert_eq!(cache_write, 1_000_000);
    assert_eq!(cache_write_1h, 0);
    // 1M * 6.25/Mtok = 6.25
    assert!((cost - 6.25).abs() < 1e-10, "got: {cost}");
}

async fn stream_cache_write_1h_cost(cache_creation: Option<Value>) -> (f64, u64, u64) {
    let mut start_usage = json!({
        "input_tokens": 100,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 1_000_000,
    });
    if let Some(cache_creation) = cache_creation {
        start_usage["cache_creation"] = cache_creation;
    }
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(
        &mock,
        &[
            (
                "message_start",
                json!({ "type": "message_start", "message": { "id": "msg_test", "usage": start_usage } })
                    .to_string(),
            ),
            common::block_start_event(0, json!({ "type": "text", "text": "" })),
            common::block_delta_event(0, json!({ "type": "text_delta", "text": "Hi" })),
            common::block_stop_event(0),
            common::message_delta_event(json!({ "stop_reason": "end_turn" }), Some(json!({ "input_tokens": 100, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 1_000_000, }))),
            common::message_stop_event(),
        ],
    );
    let model = builtin_model("anthropic", "claude-opus-4-8");
    let context = Context {
        system_prompt: None,
        messages: vec![user_message_now("hi")],
        tools: None,
    };
    let options = AnthropicStreamOptions {
        transport_options: mock_transport(&mock),
        ..AnthropicStreamOptions::default()
    };
    let result = stream(&model, &context, Some(&options)).result().await;
    (
        result.usage.cost.cache_write,
        result.usage.cache_write_1h.unwrap_or(0),
        result.usage.cache_write,
    )
}

// ---------------------------------------------------------------------------
// Eager tool input streaming compat (upstream anthropic-eager-tool-input-compat)
// ---------------------------------------------------------------------------

fn lookup_tool() -> Tool {
    Tool {
        name: "lookup".to_owned(),
        description: "Look up a value".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
        }),
        constrained_sampling: None,
    }
}

fn schema_compatibility_tool() -> Tool {
    Tool {
        parameters: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false,
            "title": "LookupInput",
        }),
        ..lookup_tool()
    }
}

fn strict_tool() -> Tool {
    Tool {
        parameters: json!({
            "type": "object",
            "title": "StrictLookupInput",
            "properties": {
                "value": { "type": "string" },
                "optional": { "type": "number" },
            },
            "required": ["value"],
        }),
        constrained_sampling: Some(pi_ai::types::ConstrainedSamplingSetting::Config(
            pi_ai::types::ConstrainedSamplingConfig::JsonSchema {
                strict: pi_ai::types::Strictness::Prefer,
            },
        )),
        ..lookup_tool()
    }
}

#[tokio::test]
async fn sends_per_tool_eager_input_streaming_by_default() {
    let (headers, body) = capture_tool_request(None, [lookup_tool()].as_slice()).await;
    assert_eq!(body["tools"][0]["eager_input_streaming"], Value::Bool(true));
    assert!(recorded_beta_header(&headers).is_none(), "got: {headers:?}");
}

#[tokio::test]
async fn uses_the_legacy_fine_grained_tool_streaming_beta_when_eager_streaming_is_disabled() {
    let (headers, body) = capture_tool_request(
        Some(json!({ "supportsEagerToolInputStreaming": false })),
        [lookup_tool()].as_slice(),
    )
    .await;
    assert_eq!(body["tools"][0].get("eager_input_streaming"), None);
    assert_eq!(
        recorded_beta_header(&headers).as_deref(),
        Some("fine-grained-tool-streaming-2025-05-14"),
    );
}

#[tokio::test]
async fn does_not_send_the_legacy_beta_when_there_are_no_tools() {
    let (headers, body) = capture_tool_request(
        Some(json!({ "supportsEagerToolInputStreaming": false })),
        &[],
    )
    .await;
    assert_eq!(body.get("tools"), None);
    assert!(recorded_beta_header(&headers).is_none(), "got: {headers:?}");
}

#[tokio::test]
async fn only_sends_the_full_input_schema_for_strict_json_schema_tools() {
    let (legacy_headers, legacy_body) = capture_tool_request(
        Some(json!({ "supportsStrictTools": true })),
        [schema_compatibility_tool()].as_slice(),
    )
    .await;
    assert!(recorded_beta_header(&legacy_headers).is_none());
    assert_eq!(
        &legacy_body["tools"][0]["input_schema"],
        &json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
        }),
    );

    let (_strict_headers, strict_body) = capture_tool_request(
        Some(json!({ "supportsStrictTools": true })),
        [strict_tool()].as_slice(),
    )
    .await;
    assert_eq!(strict_body["tools"][0]["strict"], Value::Bool(true));
    let input_schema = &strict_body["tools"][0]["input_schema"];
    assert_eq!(input_schema["additionalProperties"], Value::Bool(false));
    assert_eq!(input_schema["required"], json!(["value", "optional"]));
    assert_eq!(
        input_schema["properties"]["optional"],
        json!({ "anyOf": [{ "type": "number" }, { "type": "null" }] }),
    );
    assert_eq!(input_schema["title"], "StrictLookupInput");
}

async fn capture_tool_request(
    compat: Option<Value>,
    tools: &[Tool],
) -> (Vec<(String, String)>, Value) {
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &minimal_done_events());
    let mut compat_json = json!({ "forceAdaptiveThinking": true });
    if let Some(overrides) = compat
        && let (Some(base), Some(overrides)) = (compat_json.as_object_mut(), overrides.as_object())
    {
        for (key, value) in overrides {
            base.insert(key.clone(), value.clone());
        }
    }
    let model = Model {
        id: "claude-opus-4-8".to_owned(),
        name: "Claude Opus 4.8".to_owned(),
        api: pi_ai::types::Api::from("anthropic-messages"),
        provider: pi_ai::types::ProviderId::from("test-anthropic"),
        base_url: "https://api.anthropic.com".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 200_000,
        max_tokens: 32_000,
        sampling_params: None,
        headers: None,
        compat: Some(serde_json::from_value::<ModelCompat>(compat_json).expect("compat map")),
    };
    let context = Context {
        system_prompt: None,
        messages: vec![user_message_now("Use the tool")],
        tools: if tools.is_empty() {
            None
        } else {
            Some(tools.to_vec())
        },
    };
    let mut options = common::keyed_anthropic_options(&mock);
    options.cache_retention = Some(CacheRetention::None);

    let _ = stream(&model, &context, Some(&options)).result().await;

    let request = &mock.recorded()[0];
    let body: Value =
        serde_json::from_slice(request.body.as_ref().expect("request body")).expect("body JSON");
    (request.headers.clone(), body)
}

/// The barest successful stream, for request-shape capture suites.
fn minimal_done_events() -> Vec<(&'static str, String)> {
    vec![
        common::message_start_event("msg_test", json!({ "input_tokens": 1, "output_tokens": 0 })),
        common::message_delta_event(
            json!({ "stop_reason": "end_turn" }),
            Some(json!({ "input_tokens": 1, "output_tokens": 1 })),
        ),
        common::message_stop_event(),
    ]
}

fn recorded_beta_header(headers: &[(String, String)]) -> Option<String> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .map(|(_, value)| value.clone())
}

// ---------------------------------------------------------------------------
// Empty thinking signature compat (upstream anthropic-empty-thinking-signature-compat)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn converts_empty_signature_thinking_to_text_by_default() {
    let payload = capture_thinking_replay(None, "", "internal reasoning").await;
    let assistant = assistant_content(&payload);
    assert_eq!(
        assistant,
        &json!([{ "type": "text", "text": "internal reasoning" }]),
    );
}

#[tokio::test]
async fn preserves_empty_thinking_text_when_the_signature_is_present() {
    let payload = capture_thinking_replay(None, "signed-thinking", "").await;
    let assistant = assistant_content(&payload);
    assert_eq!(
        assistant,
        &json!([{ "type": "thinking", "thinking": "", "signature": "signed-thinking" }]),
    );
}

#[tokio::test]
async fn preserves_empty_signature_thinking_when_allow_empty_signature_is_enabled() {
    let payload = capture_thinking_replay(
        Some(json!({ "allowEmptySignature": true })),
        " ",
        "internal reasoning",
    )
    .await;
    let assistant = assistant_content(&payload);
    assert_eq!(
        assistant,
        &json!([{ "type": "thinking", "thinking": "internal reasoning", "signature": "" }]),
    );
}

/// Regression for pi #9323: Fireworks emits unsigned thinking that must
/// survive replay.
#[tokio::test]
async fn preserves_unsigned_thinking_for_fireworks_models() {
    for model_id in [
        "accounts/fireworks/models/deepseek-v4-flash-0731",
        "accounts/fireworks/models/deepseek-v4-flash-vision-exp",
        "accounts/fireworks/models/deepseek-v4-pro-0813",
        "accounts/fireworks/models/qwen3p8-max",
        "accounts/fireworks/models/qwen3p8-2p4t-a95b",
        "accounts/fireworks/models/kimi-k2p6",
    ] {
        let model = builtin_model("fireworks", model_id);
        assert!(
            model
                .compat
                .as_ref()
                .and_then(|compat| compat.allow_empty_signature)
                == Some(true),
            "{model_id} allowEmptySignature"
        );
        let mut context = thinking_replay_context("", "internal reasoning", "fireworks", model_id);
        if let Message::Assistant(assistant) = &mut context.messages[1] {
            assistant.content.push(AssistantBlock::Text(TextContent {
                text: "answer".to_owned(),
                text_signature: None,
            }));
        }
        let payload = capture_simple_payload(&model, context, simple_capture_options()).await;
        assert_eq!(
            assistant_content(&payload),
            &json!([
                { "type": "thinking", "thinking": "internal reasoning", "signature": "" },
                { "type": "text", "text": "answer" },
            ]),
            "{model_id}"
        );
    }
}

/// Regression for pi #9323: opting into unsigned replay must not change
/// cross-model conversion.
#[tokio::test]
async fn still_converts_cross_model_fireworks_thinking_to_text() {
    let model = builtin_model(
        "fireworks",
        "accounts/fireworks/models/deepseek-v4-flash-0731",
    );
    let context = thinking_replay_context(
        "",
        "internal reasoning",
        "fireworks",
        "accounts/fireworks/models/kimi-k2p6",
    );
    let payload = capture_simple_payload(&model, context, simple_capture_options()).await;
    let assistant = assistant_content(&payload);
    assert_eq!(
        assistant,
        &json!([{ "type": "text", "text": "internal reasoning" }])
    );
}

#[tokio::test]
async fn allows_empty_signatures_for_kimi_coding_k3() {
    let model = builtin_model("kimi-coding", "k3");
    assert_eq!(
        model
            .compat
            .as_ref()
            .and_then(|compat| compat.allow_empty_signature),
        Some(true)
    );
    let payload = capture_simple_payload(
        &model,
        thinking_replay_context(" ", "internal reasoning", "kimi-coding", "k3"),
        simple_capture_options(),
    )
    .await;
    let assistant = assistant_content(&payload);
    assert_eq!(
        assistant,
        &json!([{ "type": "thinking", "thinking": "internal reasoning", "signature": "" }]),
    );
}

fn assistant_content(payload: &Value) -> &Value {
    payload
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| {
            messages
                .iter()
                .find(|message| message["role"] == "assistant")
        })
        .and_then(|message| message.get("content"))
        .expect("assistant message content")
}

async fn capture_thinking_replay(
    compat: Option<Value>,
    thinking_signature: &str,
    thinking: &str,
) -> Value {
    let model = Model {
        id: "mimo-v2.5-pro".to_owned(),
        name: "MiMo-V2.5-Pro".to_owned(),
        api: pi_ai::types::Api::from("anthropic-messages"),
        provider: pi_ai::types::ProviderId::from("xiaomi-token-plan-ams"),
        base_url: "http://127.0.0.1:9/anthropic".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 1_048_576,
        max_tokens: 1024,
        sampling_params: None,
        headers: None,
        compat: compat
            .map(|compat| serde_json::from_value::<ModelCompat>(compat).expect("compat map")),
    };
    let context = thinking_replay_context(
        thinking_signature,
        thinking,
        "xiaomi-token-plan-ams",
        "mimo-v2.5-pro",
    );
    capture_simple_payload(&model, context, simple_capture_options()).await
}

fn thinking_replay_context(
    thinking_signature: &str,
    thinking: &str,
    provider: &str,
    model_id: &str,
) -> Context {
    let assistant = common::thinking_assistant_message(
        "anthropic-messages",
        provider,
        model_id,
        thinking,
        thinking_signature,
    );
    Context {
        system_prompt: None,
        messages: vec![
            user_message_now("first"),
            Message::Assistant(assistant),
            user_message_now("second"),
        ],
        tools: None,
    }
}

fn simple_capture_options() -> SimpleStreamOptions {
    SimpleStreamOptions {
        api_key: Some("fake-key".to_owned()),
        ..SimpleStreamOptions::default()
    }
}

// ---------------------------------------------------------------------------
// forceAdaptiveThinking compat (upstream anthropic-force-adaptive-thinking)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sends_legacy_thinking_payload_for_custom_model_ids_by_default() {
    let payload = capture_custom_model_thinking(None, Some(ThinkingLevel::Medium)).await;
    assert_eq!(payload["thinking"]["type"], "enabled");
    assert_eq!(payload.get("output_config"), None);
}

#[tokio::test]
async fn sends_adaptive_thinking_payload_when_compat_force_adaptive_thinking_is_true() {
    let payload = capture_custom_model_thinking(
        Some(json!({ "forceAdaptiveThinking": true })),
        Some(ThinkingLevel::Medium),
    )
    .await;
    assert_eq!(
        payload["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(payload["output_config"], json!({ "effort": "medium" }));
}

#[tokio::test]
async fn uses_adaptive_thinking_with_native_xhigh_effort_for_claude_fable_5() {
    let model = builtin_model("anthropic", "claude-fable-5");
    let payload = capture_simple_payload(
        &model,
        user_context(),
        reasoning_options(ThinkingLevel::Xhigh),
    )
    .await;
    assert_eq!(
        payload["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(payload["output_config"], json!({ "effort": "xhigh" }));
}

#[tokio::test]
async fn uses_adaptive_thinking_effort_without_a_token_budget_for_kimi_coding() {
    for (model_id, level, effort) in [
        ("kimi-for-coding", ThinkingLevel::Medium, "medium"),
        ("k3", ThinkingLevel::Max, "max"),
        ("kimi-for-coding-highspeed", ThinkingLevel::Medium, "medium"),
    ] {
        let model = builtin_model("kimi-coding", model_id);
        let payload =
            capture_simple_payload(&model, user_context(), reasoning_options(level)).await;
        assert_eq!(
            payload["thinking"],
            json!({ "type": "adaptive", "display": "summarized" }),
            "{model_id}"
        );
        assert_eq!(
            payload["output_config"],
            json!({ "effort": effort }),
            "{model_id}"
        );
    }
}

#[tokio::test]
async fn allows_builtin_adaptive_models_to_opt_out_with_compat_force_adaptive_thinking_false() {
    let mut model = builtin_model("anthropic", "claude-opus-4-8");
    model.compat = Some(
        serde_json::from_value::<ModelCompat>(json!({ "forceAdaptiveThinking": false }))
            .expect("compat map"),
    );
    let payload = capture_simple_payload(
        &model,
        user_context(),
        reasoning_options(ThinkingLevel::Medium),
    )
    .await;
    assert_eq!(payload["thinking"]["type"], "enabled");
    assert_eq!(payload.get("output_config"), None);
}

#[tokio::test]
async fn preserves_thinking_type_disabled_when_reasoning_is_off_regardless_of_override() {
    let payload =
        capture_custom_model_thinking(Some(json!({ "forceAdaptiveThinking": true })), None).await;
    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
    assert_eq!(payload.get("output_config"), None);
}

async fn capture_custom_model_thinking(
    compat: Option<Value>,
    level: Option<ThinkingLevel>,
) -> Value {
    let model = vendor_proxy_model(
        "vendor--claude-opus-latest",
        "Vendor Proxy Opus Latest",
        compat,
    );
    let mut options = simple_capture_options();
    options.reasoning = level;
    capture_simple_payload(&model, user_context(), options).await
}

// ---------------------------------------------------------------------------
// Mid-conversation effort (upstream anthropic-mid-conversation-effort)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reconstructs_an_exact_historical_marker_prefix_and_appends_the_current_marker() {
    let model = managed_model();
    let first = capture_managed_effort(&model, vec![user_message_now("one")], Some("low")).await;
    let second = capture_managed_effort(
        &model,
        vec![
            user_message_now("one"),
            Message::Assistant(managed_assistant(&model, "low")),
            user_message_now("two"),
        ],
        Some("high"),
    )
    .await;

    let first_messages = first.payload["messages"].as_array().expect("messages");
    assert_eq!(
        first_messages,
        &vec![
            json!({ "role": "user", "content": "one" }),
            json!({ "role": "system", "content": [], "output_config": { "effort": "low" } }),
        ],
    );
    let second_messages = second.payload["messages"].as_array().expect("messages");
    assert_eq!(
        &second_messages[..first_messages.len()],
        first_messages.as_slice(),
    );
    assert_eq!(
        second_messages.last().expect("final marker"),
        &json!({ "role": "system", "content": [], "output_config": { "effort": "high" } }),
    );
    assert_eq!(first.payload["output_config"], json!({ "effort": "high" }));
    assert_eq!(second.payload["output_config"], json!({ "effort": "high" }));
    assert_eq!(
        second.payload["thinking"],
        json!({
            "type": "adaptive",
            "display": "summarized",
            "block_binding": { "prefix_mismatch_behavior": "drop_block" },
        }),
    );
    assert_eq!(
        first.message.provider_thinking_level.as_deref(),
        Some("low")
    );
}

#[tokio::test]
async fn preserves_native_effort_levels() {
    for effort in ["low", "medium", "high", "xhigh", "max"] {
        let model = managed_model();
        let capture =
            capture_managed_effort(&model, vec![user_message_now("one")], Some(effort)).await;
        let system_messages: Vec<&Value> = capture.payload["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .filter(|message| message["role"] == "system")
            .collect();
        assert_eq!(
            system_messages,
            vec![&json!({
                "role": "system",
                "content": [],
                "output_config": { "effort": effort },
            })],
            "{effort}"
        );
        assert_eq!(
            capture.message.provider_thinking_level,
            Some(effort.to_owned()),
            "{effort}"
        );
    }
}

#[tokio::test]
async fn defaults_omitted_effort_to_high_and_still_enables_drop_block() {
    let model = managed_model();
    let capture = capture_managed_effort(&model, vec![user_message_now("one")], None).await;
    assert_eq!(
        capture.payload["messages"]
            .as_array()
            .expect("messages")
            .last()
            .expect("final marker"),
        &json!({ "role": "system", "content": [], "output_config": { "effort": "high" } }),
    );
    assert_eq!(
        capture.payload["thinking"]["block_binding"]["prefix_mismatch_behavior"],
        "drop_block"
    );
    assert_eq!(
        capture.message.provider_thinking_level,
        Some("high".to_owned())
    );
}

#[tokio::test]
async fn does_not_invent_markers_for_legacy_or_other_provider_assistants() {
    let model = managed_model();
    let legacy = managed_assistant(&model, "");
    let mut other_provider = managed_assistant(&model, "low");
    other_provider.provider = pi_ai::types::ProviderId::from("other-provider");
    let capture = capture_managed_effort(
        &model,
        vec![
            user_message_now("one"),
            Message::Assistant(legacy),
            user_message_now("two"),
            Message::Assistant(other_provider),
            user_message_now("three"),
        ],
        Some("medium"),
    )
    .await;
    let system_messages: Vec<&Value> = capture.payload["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .filter(|message| message["role"] == "system")
        .collect();
    assert_eq!(
        system_messages,
        vec![&json!({
            "role": "system",
            "content": [],
            "output_config": { "effort": "medium" },
        })],
    );
}

#[tokio::test]
async fn leaves_unsupported_models_on_top_level_effort() {
    let mut model = managed_model();
    model.compat = Some(
        serde_json::from_value::<ModelCompat>(json!({ "forceAdaptiveThinking": true }))
            .expect("compat map"),
    );
    let capture = capture_managed_effort(&model, vec![user_message_now("one")], Some("low")).await;
    assert_eq!(
        capture.payload["messages"],
        json!([{ "role": "user", "content": "one" }])
    );
    assert_eq!(capture.payload["output_config"], json!({ "effort": "low" }));
    assert_eq!(
        capture.payload["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(capture.message.provider_thinking_level, None);
}

#[tokio::test]
async fn sends_the_effort_and_binding_beta_headers() {
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &minimal_done_events());
    let model = managed_model();
    let mut options = common::keyed_anthropic_options(&mock);
    options.cache_retention = Some(CacheRetention::None);
    options.thinking_enabled = Some(true);
    let result = stream(&model, &user_context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    let beta = recorded_beta_header(&mock.recorded()[0].headers).expect("the beta header");
    assert!(
        beta.contains("mid-conversation-output-config-2026-07-01"),
        "got: {beta}"
    );
    assert!(
        beta.contains("thinking-binding-controls-2026-08-01"),
        "got: {beta}"
    );
}

#[tokio::test]
async fn generates_exact_model_and_transport_gates() {
    let direct = builtin_model("anthropic", "claude-fable-5-1");
    let open_router = builtin_model("openrouter", "anthropic/claude-fable-5.1");
    let unsupported = builtin_model("anthropic", "claude-opus-4-8");
    assert_eq!(
        direct
            .compat
            .as_ref()
            .and_then(|compat| compat.supports_mid_convo_effort),
        Some(true)
    );
    assert_eq!(
        direct
            .thinking_level_map
            .as_ref()
            .and_then(|map| map.get(&pi_ai::types::ModelThinkingLevel::Off)),
        Some(&None)
    );
    assert_eq!(
        open_router.api,
        pi_ai::types::Api::from("anthropic-messages")
    );
    assert_eq!(open_router.base_url, "https://openrouter.ai/api");
    assert_eq!(
        open_router
            .compat
            .as_ref()
            .and_then(|compat| compat.supports_mid_convo_effort),
        Some(true)
    );
    assert_eq!(
        unsupported
            .compat
            .as_ref()
            .and_then(|compat| compat.supports_mid_convo_effort),
        None
    );
    assert_eq!(
        builtin_model("anthropic", "claude-opus-5")
            .compat
            .as_ref()
            .and_then(|compat| compat.allowed_fallback_models.clone()),
        None
    );
}

fn managed_model() -> Model {
    Model {
        id: "claude-fable-5-1".to_owned(),
        name: "Claude Fable 5.1".to_owned(),
        api: pi_ai::types::Api::from("anthropic-messages"),
        provider: pi_ai::types::ProviderId::from("anthropic"),
        base_url: "http://127.0.0.1:9".to_owned(),
        reasoning: true,
        thinking_level_map: Some(
            [
                (pi_ai::types::ModelThinkingLevel::Off, None),
                (
                    pi_ai::types::ModelThinkingLevel::Minimal,
                    Some("low".to_owned()),
                ),
                (
                    pi_ai::types::ModelThinkingLevel::Low,
                    Some("low".to_owned()),
                ),
                (
                    pi_ai::types::ModelThinkingLevel::Medium,
                    Some("medium".to_owned()),
                ),
                (
                    pi_ai::types::ModelThinkingLevel::High,
                    Some("high".to_owned()),
                ),
                (
                    pi_ai::types::ModelThinkingLevel::Max,
                    Some("max".to_owned()),
                ),
            ]
            .into_iter()
            .collect(),
        ),
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 200_000,
        max_tokens: 32_000,
        sampling_params: None,
        headers: None,
        compat: Some(
            serde_json::from_value::<ModelCompat>(json!({
                "forceAdaptiveThinking": true,
                "supportsMidConvoEffort": true,
            }))
            .expect("compat map"),
        ),
    }
}

fn managed_assistant(model: &Model, level: &str) -> pi_ai::types::AssistantMessage {
    let mut assistant = common::thinking_assistant_message(
        "anthropic-messages",
        &model.provider.0,
        &model.id,
        "reasoning",
        "signature",
    );
    assistant.content.push(AssistantBlock::Text(TextContent {
        text: "answer".to_owned(),
        text_signature: None,
    }));
    assistant.provider_thinking_level = if level.is_empty() {
        None
    } else {
        Some(level.to_owned())
    };
    assistant.stop_reason = StopReason::Stop;
    assistant
}

struct ManagedCapture {
    payload: Value,
    message: pi_ai::types::AssistantMessage,
}

async fn capture_managed_effort(
    model: &Model,
    messages: Vec<Message>,
    effort: Option<&str>,
) -> ManagedCapture {
    let mock = MockHttpClient::new();
    let (hook, captured) = payload_capture();
    let options = AnthropicStreamOptions {
        transport_options: pi_ai::types::TransportOptions {
            http_client: Some(Arc::new(mock)),
            on_payload: Some(hook),
            ..pi_ai::types::TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        cache_retention: Some(CacheRetention::None),
        thinking_enabled: Some(true),
        effort: effort.map(|effort| AnthropicEffort::try_from(effort).expect("effort")),
        ..AnthropicStreamOptions::default()
    };
    let context = Context {
        system_prompt: None,
        messages,
        tools: None,
    };
    let message = stream(model, &context, Some(&options)).result().await;
    ManagedCapture {
        payload: common::captured_payload(&captured),
        message,
    }
}

// ---------------------------------------------------------------------------
// Temperature compat (upstream anthropic-temperature-compat)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn omits_temperature_for_claude_opus_4_7() {
    let payload = capture_temperature(builtin_model("anthropic", "claude-opus-4-7"), 0.0).await;
    assert_eq!(payload.get("temperature"), None);
}

#[tokio::test]
async fn omits_temperature_for_claude_opus_4_8() {
    let payload = capture_temperature(builtin_model("anthropic", "claude-opus-4-8"), 0.0).await;
    assert_eq!(payload.get("temperature"), None);
}

#[tokio::test]
async fn omits_default_temperature_for_claude_opus_4_7() {
    let payload = capture_temperature(builtin_model("anthropic", "claude-opus-4-7"), 1.0).await;
    assert_eq!(payload.get("temperature"), None);
}

#[tokio::test]
async fn keeps_temperature_for_claude_opus_4_6() {
    let payload = capture_temperature(builtin_model("anthropic", "claude-opus-4-6"), 0.0).await;
    assert_eq!(payload["temperature"], json!(0.0));
}

#[tokio::test]
async fn keeps_temperature_for_claude_sonnet_4_6() {
    let payload = capture_temperature(builtin_model("anthropic", "claude-sonnet-4-6"), 0.0).await;
    assert_eq!(payload["temperature"], json!(0.0));
}

#[tokio::test]
async fn omits_temperature_for_custom_models_with_supports_temperature_disabled() {
    let model = vendor_proxy_model(
        "vendor--claude-opus-4-7",
        "Vendor Proxy Opus 4.7",
        Some(json!({ "supportsTemperature": false })),
    );
    let payload = capture_temperature(model, 0.0).await;
    assert_eq!(payload.get("temperature"), None);
}

/// A vendor-proxy model the gate tests stream through: the id mirrors
/// corporate proxy schemes such as `anthropic--claude-opus-latest`.
fn vendor_proxy_model(id: &str, name: &str, compat: Option<Value>) -> Model {
    Model {
        id: id.to_owned(),
        name: name.to_owned(),
        api: pi_ai::types::Api::from("anthropic-messages"),
        provider: pi_ai::types::ProviderId::from("vendor-proxy"),
        base_url: "http://127.0.0.1:9".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 200_000,
        max_tokens: 32_000,
        sampling_params: None,
        headers: None,
        compat: compat
            .map(|compat| serde_json::from_value::<ModelCompat>(compat).expect("compat map")),
    }
}

async fn capture_temperature(model: Model, temperature: f64) -> Value {
    let mut options = simple_capture_options();
    options.temperature = Some(temperature);
    capture_simple_payload(&model, user_context(), options).await
}

// ---------------------------------------------------------------------------
// Thinking disable payload (upstream anthropic-thinking-disable)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sends_thinking_type_disabled_for_budget_based_reasoning_models_when_thinking_is_off() {
    let payload =
        capture_reasoning_or_disabled(builtin_model("anthropic", "claude-sonnet-4-5"), None).await;
    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
    assert_eq!(payload.get("output_config"), None);
}

#[tokio::test]
async fn sends_thinking_type_disabled_for_adaptive_reasoning_models_when_thinking_is_off() {
    let payload =
        capture_reasoning_or_disabled(builtin_model("anthropic", "claude-opus-4-6"), None).await;
    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
    assert_eq!(payload.get("output_config"), None);
}

#[tokio::test]
async fn sends_thinking_type_disabled_for_claude_opus_4_8_when_thinking_is_off() {
    let payload =
        capture_reasoning_or_disabled(builtin_model("anthropic", "claude-opus-4-8"), None).await;
    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
    assert_eq!(payload.get("output_config"), None);
}

#[tokio::test]
async fn omits_thinking_type_disabled_for_claude_fable_5_when_thinking_is_off() {
    let payload =
        capture_reasoning_or_disabled(builtin_model("anthropic", "claude-fable-5"), None).await;
    assert_eq!(payload.get("thinking"), None);
    assert_eq!(payload.get("output_config"), None);
}

#[tokio::test]
async fn uses_adaptive_thinking_for_claude_opus_4_8_when_reasoning_is_enabled() {
    let payload = capture_reasoning_or_disabled(
        builtin_model("anthropic", "claude-opus-4-8"),
        Some(ThinkingLevel::High),
    )
    .await;
    assert_eq!(
        payload["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(payload["output_config"], json!({ "effort": "high" }));
}

#[tokio::test]
async fn uses_adaptive_thinking_for_claude_sonnet_5_when_reasoning_is_enabled() {
    let payload = capture_reasoning_or_disabled(
        builtin_model("anthropic", "claude-sonnet-5"),
        Some(ThinkingLevel::High),
    )
    .await;
    assert_eq!(
        payload["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(payload["output_config"], json!({ "effort": "high" }));
}

#[tokio::test]
async fn maps_xhigh_reasoning_to_effort_xhigh_for_claude_opus_4_8() {
    let payload = capture_reasoning_or_disabled(
        builtin_model("anthropic", "claude-opus-4-8"),
        Some(ThinkingLevel::Xhigh),
    )
    .await;
    assert_eq!(
        payload["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(payload["output_config"], json!({ "effort": "xhigh" }));
}

async fn capture_reasoning_or_disabled(model: Model, level: Option<ThinkingLevel>) -> Value {
    let mut options = simple_capture_options();
    options.reasoning = level;
    capture_simple_payload(&model, user_context(), options).await
}

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

fn user_context() -> Context {
    Context {
        system_prompt: None,
        messages: vec![user_message_now("Hello")],
        tools: None,
    }
}

fn reasoning_options(level: ThinkingLevel) -> SimpleStreamOptions {
    SimpleStreamOptions {
        api_key: Some("fake-key".to_owned()),
        reasoning: Some(level),
        ..SimpleStreamOptions::default()
    }
}

/// Capture the payload a simple request produces before its dispatch: the
/// hook records it and the empty mock fails the request, upstream's
/// throwing `PayloadCaptured` hook minus the throw.
async fn capture_simple_payload(
    model: &Model,
    context: Context,
    mut options: SimpleStreamOptions,
) -> Value {
    let mock = MockHttpClient::new();
    let (hook, captured) = payload_capture();
    options.transport_options = pi_ai::types::TransportOptions {
        http_client: Some(Arc::new(mock)),
        on_payload: Some(hook),
        ..options.transport_options
    };
    let _ = stream_simple(model, &context, Some(&options))
        .result()
        .await;
    common::captured_payload(&captured)
}

// ---------------------------------------------------------------------------
// Wire-shape branches the suites above leave uncovered
// ---------------------------------------------------------------------------

fn simple_tool() -> Tool {
    Tool {
        name: "lookup".to_owned(),
        description: "Look up a value".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
        }),
        constrained_sampling: None,
    }
}

/// Stream a tool-result turn with the given content and tools, retention
/// off, and capture the request body, the shape the tool-result suites share.
async fn capture_tool_result_turn(
    model: &Model,
    result_content: Vec<pi_ai::types::ToolResultBlock>,
    tools: Vec<Tool>,
) -> Value {
    capture_tool_result_turn_with_names(model, result_content, tools, None).await
}

/// The variant the deferred-reference suites drive, with load markers.
async fn capture_tool_result_turn_with_names(
    model: &Model,
    result_content: Vec<pi_ai::types::ToolResultBlock>,
    tools: Vec<Tool>,
    added_tool_names: Option<Vec<String>>,
) -> Value {
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &minimal_done_events());
    let context = Context {
        system_prompt: None,
        messages: vec![
            user_message_now("use the tool"),
            common::tool_result_message("call_1", result_content, added_tool_names),
        ],
        tools: Some(tools),
    };
    let mut options = common::keyed_anthropic_options(&mock);
    options.cache_retention = Some(CacheRetention::None);
    let _ = stream(model, &context, Some(&options)).result().await;
    recorded_request_body(&mock)
}

#[tokio::test]
async fn tool_results_join_text_and_route_images_into_block_arrays() {
    let model = Model {
        api: pi_ai::types::Api::from("anthropic-messages"),
        provider: pi_ai::types::ProviderId::from("test-anthropic"),
        ..simple_anthropic_model()
    };
    let body = capture_tool_result_turn(
        &model,
        vec![
            pi_ai::types::ToolResultBlock::Text(TextContent {
                text: "first".to_owned(),
                text_signature: None,
            }),
            pi_ai::types::ToolResultBlock::Text(TextContent {
                text: "second".to_owned(),
                text_signature: None,
            }),
        ],
        vec![simple_tool()],
    )
    .await;

    let tool_turn = &body["messages"]
        .as_array()
        .expect("messages")
        .last()
        .expect("tool-result turn")["content"];
    // Text-only tool results join into one string.
    assert_eq!(tool_turn[0]["type"], "tool_result");
    assert_eq!(tool_turn[0]["content"], json!("first\nsecond"));
}

#[tokio::test]
async fn image_only_tool_results_gain_the_placeholder_text_block() {
    let body = capture_tool_result_turn(
        &Model {
            input: vec![Modality::Text, Modality::Image],
            ..simple_anthropic_model()
        },
        vec![pi_ai::types::ToolResultBlock::Image(
            pi_ai::types::ImageContent {
                data: "aGVsbG8=".to_owned(),
                mime_type: "image/png".to_owned(),
            },
        )],
        vec![simple_tool()],
    )
    .await;

    let tool_turn = &body["messages"]
        .as_array()
        .expect("messages")
        .last()
        .expect("tool-result turn")["content"];
    assert_eq!(tool_turn[0]["type"], "tool_result");
    assert_eq!(tool_turn[0]["content"][0]["type"], "text");
    assert_eq!(tool_turn[0]["content"][0]["text"], "(see attached image)");
    assert_eq!(tool_turn[0]["content"][1]["type"], "image");
    // No displaced references: the ordinary content has no siblings.
    assert_eq!(tool_turn.as_array().expect("content blocks").len(), 1);
}

#[tokio::test]
async fn user_image_blocks_convert_to_base64_sources_and_cache_control_lands_on_the_last_block() {
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &minimal_done_events());
    let model = Model {
        input: vec![Modality::Text, Modality::Image],
        ..simple_anthropic_model()
    };
    let context = Context {
        system_prompt: None,
        messages: vec![Message::User(pi_ai::types::UserMessage {
            content: pi_ai::types::UserContent::Blocks(vec![
                pi_ai::types::UserBlock::Text(TextContent {
                    text: "what is this?".to_owned(),
                    text_signature: None,
                }),
                pi_ai::types::UserBlock::Image(pi_ai::types::ImageContent {
                    data: "aGVsbG8=".to_owned(),
                    mime_type: "image/jpeg".to_owned(),
                }),
            ]),
            timestamp: 1,
        })],
        tools: None,
    };
    let options = common::keyed_anthropic_options(&mock);

    let _ = stream(&model, &context, Some(&options)).result().await;

    let body = recorded_request_body(&mock);
    let last_user = body["messages"]
        .as_array()
        .expect("messages")
        .last()
        .expect("last message");
    assert_eq!(last_user["content"][1]["type"], "image");
    assert_eq!(last_user["content"][1]["source"]["type"], "base64");
    assert_eq!(
        last_user["content"][1]["source"]["media_type"],
        "image/jpeg"
    );
    // cache_control rides the last user block.
    assert!(last_user["content"][1].get("cache_control").is_some());
}

/// Tool results carrying `addedToolNames` route their load markers into
/// `tool_reference` blocks and displace the ordinary content into siblings.
#[tokio::test]
async fn deferred_tool_results_carry_tool_reference_blocks() {
    let model = Model {
        id: "claude-opus-4-8".to_owned(),
        api: pi_ai::types::Api::from("anthropic-messages"),
        provider: pi_ai::types::ProviderId::from("anthropic"),
        ..simple_anthropic_model()
    };
    let body = capture_tool_result_turn_with_names(
        &model,
        vec![pi_ai::types::ToolResultBlock::Text(TextContent {
            text: "loaded".to_owned(),
            text_signature: None,
        })],
        vec![simple_tool(), common::late_tool()],
        Some(vec!["late_tool".to_owned()]),
    )
    .await;

    let tool_turn = &body["messages"]
        .as_array()
        .expect("messages")
        .last()
        .expect("tool-result turn")["content"];
    // The tool_result block carries the reference; the ordinary content
    // rides as the sibling block behind it.
    assert_eq!(tool_turn[0]["type"], "tool_result");
    assert_eq!(tool_turn[0]["content"][0]["type"], json!("tool_reference"),);
    assert_eq!(tool_turn[0]["content"][0]["tool_name"], json!("late_tool"));
    assert_eq!(tool_turn[1]["type"], "text");
    assert_eq!(tool_turn[1]["text"], "loaded");
    // The deferred tool definition rides with `defer_loading: true` while
    // the immediate tools stay plain.
    let tools = body["tools"].as_array().expect("tools");
    let late = tools
        .iter()
        .find(|tool| tool["name"] == "late_tool")
        .expect("the deferred tool definition");
    assert_eq!(late["defer_loading"], Value::Bool(true));
}

#[tokio::test]
async fn cc_tool_names_round_trip_through_the_mock() {
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(
        &mock,
        &[
            common::message_start_event(
                "msg_test",
                json!({ "input_tokens": 1, "output_tokens": 0 }),
            ),
            common::block_start_event(
                0,
                json!({ "type": "tool_use", "id": "toolu_1", "name": "TodoWrite", "input": {} }),
            ),
            common::block_stop_event(0),
            common::message_delta_event(
                json!({ "stop_reason": "tool_use" }),
                Some(json!({ "input_tokens": 1, "output_tokens": 1 })),
            ),
            common::message_stop_event(),
        ],
    );
    let model = Model {
        api: pi_ai::types::Api::from("anthropic-messages"),
        provider: pi_ai::types::ProviderId::from("anthropic"),
        ..simple_anthropic_model()
    };
    let todo_tool = Tool {
        name: "todowrite".to_owned(),
        description: "Write a todo item".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "task": { "type": "string" } },
            "required": ["task"],
        }),
        constrained_sampling: None,
    };
    let context = Context {
        system_prompt: Some("Use the todowrite tool.".to_owned()),
        messages: vec![user_message_now("Add a todo")],
        tools: Some(vec![todo_tool]),
    };
    let options = AnthropicStreamOptions {
        transport_options: mock_transport(&mock),
        api_key: Some("sk-ant-oat-oauth-token".to_owned()),
        ..AnthropicStreamOptions::default()
    };
    let result = stream(&model, &context, Some(&options)).result().await;

    assert_eq!(result.stop_reason, StopReason::ToolUse);
    let headers = &mock.recorded()[0].headers;
    let header = |name: &str| {
        headers
            .iter()
            .find(|(name_, _)| name_.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    };
    // The OAuth request carries bearer auth, the CC identity, and the
    // OAuth beta features.
    assert_eq!(
        header("Authorization").as_deref(),
        Some("Bearer sk-ant-oat-oauth-token")
    );
    assert!(header("x-api-key").is_none());
    assert_eq!(header("user-agent").as_deref(), Some("claude-cli/2.1.251"));
    let beta = header("anthropic-beta").expect("oauth betas");
    assert!(beta.contains("oauth-2025-04-20"), "got: {beta}");
    // The outbound tool definition is renamed to CC casing.
    let body = recorded_request_body(&mock);
    assert_eq!(body["tools"][0]["name"], json!("TodoWrite"));
    // The inbound call maps back to the context's spelling.
    let tool_call = result.content.iter().find_map(|block| match block {
        AssistantBlock::ToolCall(tool_call) => Some(tool_call.clone()),
        _ => None,
    });
    assert_eq!(tool_call.expect("tool call").name, "todowrite");
}

fn simple_anthropic_model() -> Model {
    Model {
        id: "claude-test".to_owned(),
        name: "Claude Test".to_owned(),
        api: pi_ai::types::Api::from("anthropic-messages"),
        provider: pi_ai::types::ProviderId::from("anthropic"),
        base_url: "https://api.anthropic.com".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 100_000,
        max_tokens: 4096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn recorded_request_body(mock: &MockHttpClient) -> Value {
    serde_json::from_slice(
        mock.recorded()[0]
            .body
            .as_ref()
            .expect("anthropic request carries a body"),
    )
    .expect("body JSON")
}

/// The Copilot branch of the header assembly, upstream's `createClient`
/// copilot arm.
#[tokio::test]
async fn copilot_requests_carry_the_bearer_and_dynamic_headers() {
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &minimal_done_events());
    let model = Model {
        api: pi_ai::types::Api::from("anthropic-messages"),
        provider: pi_ai::types::ProviderId::from("github-copilot"),
        ..simple_anthropic_model()
    };
    let context = Context {
        system_prompt: None,
        messages: vec![user_message_now("hello")],
        tools: None,
    };
    let options = copilot_options(&mock);
    let _ = stream(&model, &context, Some(&options)).result().await;

    let headers = &mock.recorded()[0].headers;
    let header = |name: &str| {
        headers
            .iter()
            .find(|(name_, _)| name_.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    };
    assert_eq!(
        header("Authorization").as_deref(),
        Some("Bearer copilot-token")
    );
    assert_eq!(header("X-Initiator").as_deref(), Some("user"));
    assert_eq!(
        header("Openai-Intent").as_deref(),
        Some("conversation-edits")
    );
    assert!(header("Copilot-Vision-Request").is_none());
}

/// The agent-initiator variant: the last message is not a user turn, and a
/// vision input adds the Copilot-Vision-Request header.
#[tokio::test]
async fn copilot_headers_flip_to_agent_and_carry_vision() {
    let (initiator, vision) =
        copilot_header_probe(vec![Message::ToolResult(pi_ai::types::ToolResultMessage {
            tool_call_id: "call_1".to_owned(),
            tool_name: "read".to_owned(),
            content: vec![pi_ai::types::ToolResultBlock::Image(
                pi_ai::types::ImageContent {
                    data: "aGVsbG8=".to_owned(),
                    mime_type: "image/png".to_owned(),
                },
            )],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 1,
        })])
        .await;
    assert_eq!(initiator, Some("agent".to_owned()));
    assert_eq!(vision.as_deref(), Some("true"));
}

/// The copilot-credential options the copilot header suites send.
fn copilot_options(mock: &MockHttpClient) -> AnthropicStreamOptions {
    AnthropicStreamOptions {
        transport_options: mock_transport(mock),
        api_key: Some("copilot-token".to_owned()),
        ..AnthropicStreamOptions::default()
    }
}
async fn copilot_header_probe(messages: Vec<Message>) -> (Option<String>, Option<String>) {
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &minimal_done_events());
    let model = Model {
        api: pi_ai::types::Api::from("anthropic-messages"),
        provider: pi_ai::types::ProviderId::from("github-copilot"),
        ..simple_anthropic_model()
    };
    let context = Context {
        system_prompt: None,
        messages,
        tools: None,
    };
    let mut options = copilot_options(&mock);
    options.cache_retention = Some(CacheRetention::None);
    let _ = stream(&model, &context, Some(&options)).result().await;

    let headers = &mock.recorded()[0].headers;
    let header = |name: &str| {
        headers
            .iter()
            .find(|(name_, _)| name_.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    };
    (header("X-Initiator"), header("Copilot-Vision-Request"))
}

/// Stream a request through the given seam mock and settle the result, the
/// shape the SDK-error suites share.
async fn error_stream_probe(route: impl FnOnce(&MockHttpClient)) -> pi_ai::types::AssistantMessage {
    let mock = MockHttpClient::new();
    route(&mock);
    let options = AnthropicStreamOptions {
        transport_options: mock_transport(&mock),
        api_key: Some("anthropic-key".to_owned()),
        ..AnthropicStreamOptions::default()
    };
    let context = Context {
        system_prompt: None,
        messages: vec![user_message_now("Hello")],
        tools: None,
    };
    stream(&simple_anthropic_model(), &context, Some(&options))
        .result()
        .await
}

/// The retry and seam error surfaces, upstream's SDK error mapping.
#[tokio::test]
async fn error_bodies_fold_into_the_stream_error_message() {
    let result = error_stream_probe(|mock| {
        mock.on(|request| request.url.contains("/v1/messages"))
            .respond(pi_ai::http::json_response(
                401,
                &json!({ "type": "error", "error": { "type": "authentication_error", "message": "invalid x-api-key" } }),
            ));
    })
    .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the provider error message");
    // The pinned SDK's message shape: `{status} {parsed body}`.
    assert!(
        message.contains("401") && message.contains("authentication_error"),
        "got: {message}"
    );
}

/// A transport failure (no route) surfaces as a non-retryable stream error
/// without a status.
#[tokio::test]
async fn transport_failures_surface_their_transport_message() {
    // No route mounted: the mock fails like a dead server.
    let result = error_stream_probe(|_mock| {}).await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the transport error");
    assert!(message.contains("no mock route matched"), "got: {message}");
}

/// A missing credential fails with the upstream assert message.
#[tokio::test]
async fn a_missing_credential_fails_the_stream_with_the_assert_message() {
    let model = simple_anthropic_model();
    let context = Context {
        system_prompt: None,
        messages: vec![user_message_now("Hello")],
        tools: None,
    };
    let options = AnthropicStreamOptions::default();
    let result = stream(&model, &context, Some(&options)).result().await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("No API key for provider: anthropic"),
    );
}

/// `stream_simple` without reasoning sends thinking off and keeps the
/// neutral tool choice.
#[tokio::test]
async fn simple_requests_without_reasoning_disable_thinking() {
    let mock = MockHttpClient::new();
    let (hook, captured) = payload_capture();
    common::anthropic_mock_with(&mock, &minimal_done_events());
    let options = SimpleStreamOptions {
        transport_options: pi_ai::types::TransportOptions {
            http_client: Some(Arc::new(mock.clone())),
            on_payload: Some(hook),
            ..pi_ai::types::TransportOptions::default()
        },
        api_key: Some("test-key".to_owned()),
        tool_choice: Some(pi_ai::types::ToolChoice::Auto),
        ..SimpleStreamOptions::default()
    };
    let mut model = simple_anthropic_model();
    model.reasoning = true;
    let context = Context {
        system_prompt: None,
        messages: vec![user_message_now("Hello")],
        tools: Some(vec![simple_tool()]),
    };
    let _ = stream_simple(&model, &context, Some(&options))
        .result()
        .await;

    let payload = common::captured_payload(&captured);
    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
    assert_eq!(payload["tool_choice"], json!({ "type": "auto" }));
}

/// The forced adaptive-thinking map: the model's thinkingLevelMap picks the
/// native effort spelling.
#[tokio::test]
async fn adaptive_models_map_their_thinking_level_entries() {
    let mut model = simple_anthropic_model();
    model.reasoning = true;
    model.compat = Some(
        serde_json::from_value::<ModelCompat>(json!({ "forceAdaptiveThinking": true }))
            .expect("compat map"),
    );
    model.thinking_level_map = Some(
        std::iter::once((
            pi_ai::types::ModelThinkingLevel::Medium,
            Some("high".to_owned()),
        ))
        .collect(),
    );
    let payload = capture_simple_payload(
        &model,
        user_context(),
        reasoning_options(ThinkingLevel::Medium),
    )
    .await;
    assert_eq!(payload["output_config"], json!({ "effort": "high" }));
}

/// Metadata rides `user_id` into the request metadata.
#[tokio::test]
async fn request_metadata_carries_the_user_id() {
    let mut options = simple_capture_options();
    options.metadata =
        Some(std::iter::once(("user_id".to_owned(), Value::String("user_1".to_owned()))).collect());
    let model = simple_anthropic_model();
    let payload = capture_simple_payload(&model, user_context(), options).await;
    assert_eq!(payload["metadata"], json!({ "user_id": "user_1" }));
}

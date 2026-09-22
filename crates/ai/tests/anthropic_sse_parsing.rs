//! Anthropic raw SSE parsing, ported from
//! `packages/ai/test/anthropic-sse-parsing.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The fake `@anthropic-ai/sdk` clients upstream injects through
//! `options.client` become [`MockHttpClient`] seam routes; the params
//! assertions upstream reads off the fake client become recorded-request
//! header and body assertions.

#![expect(
    clippy::expect_used,
    reason = "the tests pin stream outcomes; an unexpected shape panics the test by design"
)]

use std::sync::Arc;

use pi_ai::api::anthropic_messages::{AnthropicStreamOptions, stream as stream_anthropic};
use pi_ai::http::MockHttpClient;
use pi_ai::types::{Context, OnPayload, StopReason, Tool};
use serde_json::json;

mod common;
use common::{
    anthropic_mock, anthropic_mock_with, builtin_model, mock_transport, user_message_at,
    user_message_now,
};

fn minimal_anthropic_events() -> Vec<(&'static str, String)> {
    vec![
        common::message_start_event(
            "msg_test",
            json!({ "input_tokens": 12, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, }),
        ),
        common::block_start_event(0, json!({ "type": "text", "text": "" })),
        common::block_delta_event(0, json!({ "type": "text_delta", "text": "Hello" })),
        common::block_stop_event(0),
        common::message_delta_event(
            json!({ "stop_reason": "end_turn" }),
            Some(
                json!({ "input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, }),
            ),
        ),
        common::message_stop_event(),
    ]
}

fn sse_options(mock: &MockHttpClient) -> AnthropicStreamOptions {
    AnthropicStreamOptions {
        transport_options: mock_transport(mock),
        ..AnthropicStreamOptions::default()
    }
}

fn recorded_body(mock: &MockHttpClient) -> serde_json::Value {
    let request = &mock.recorded()[0];
    serde_json::from_slice(
        request
            .body
            .as_ref()
            .expect("anthropic request carries a body"),
    )
    .expect("the request body is JSON")
}

fn recorded_header(mock: &MockHttpClient, name: &str) -> Option<String> {
    mock.recorded()[0]
        .headers
        .iter()
        .find(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
}

/// Upstream injects a fake SDK client through `options.client`; the port
/// injects the mock seam transport with no credential.
#[tokio::test]
async fn fails_safely_when_anthropic_falls_back_after_output_begins() {
    let mock = anthropic_mock(&[
        common::message_start_from_serving_model(
            "claude-opus-5",
            "msg_fallback",
            json!({ "input_tokens": 1, "output_tokens": 0 }),
        ),
        common::block_start_event(0, json!({ "type": "text", "text": "partial" })),
        common::block_stop_event(0),
        common::block_start_event(
            1,
            json!({ "type": "fallback", "from": { "model": "claude-opus-5" }, "to": { "model": "claude-opus-4-8" }, }),
        ),
    ]);
    let model = builtin_model("anthropic", "claude-opus-5");
    let context = Context {
        messages: vec![user_message_at("Hello", 1)],
        ..Context::default()
    };
    let result = stream_anthropic(&model, &context, Some(&sse_options(&mock)))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    let message = result.error_message.expect("the fallback error message");
    assert!(
        message.contains("unsupported mid-output model fallback"),
        "got: {message}"
    );
}

#[tokio::test]
async fn forces_streaming_after_an_on_payload_replacement() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &minimal_anthropic_events());
    let options = AnthropicStreamOptions {
        transport_options: pi_ai::types::TransportOptions {
            http_client: Some(Arc::new(mock.clone())),
            on_payload: Some(OnPayload::new(|payload, _model| {
                let mut replacement = payload;
                replacement["stream"] = serde_json::json!(false);
                Box::pin(async move { Some(replacement) })
            })),
            ..pi_ai::types::TransportOptions::default()
        },
        ..AnthropicStreamOptions::default()
    };
    let model = builtin_model("anthropic", "claude-fable-5-1");
    let context = Context {
        messages: vec![user_message_at("Hello", 1)],
        ..Context::default()
    };

    let _ = stream_anthropic(&model, &context, Some(&options))
        .result()
        .await;

    let body = recorded_body(&mock);
    assert_eq!(body.get("stream"), Some(&serde_json::json!(true)));
}

#[tokio::test]
async fn omits_the_interleaved_thinking_beta_when_thinking_is_disabled() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &minimal_anthropic_events());
    let options = AnthropicStreamOptions {
        transport_options: mock_transport(&mock),
        thinking_enabled: Some(false),
        ..AnthropicStreamOptions::default()
    };
    let model = builtin_model("openrouter", "anthropic/claude-3-haiku");
    let context = Context {
        messages: vec![user_message_at("Hello", 1)],
        ..Context::default()
    };

    let _ = stream_anthropic(&model, &context, Some(&options))
        .result()
        .await;

    let beta = recorded_header(&mock, "anthropic-beta");
    assert!(beta.is_none(), "got: {beta:?}");
}

#[tokio::test]
async fn passes_managed_beta_features_to_injected_clients() {
    let mock = MockHttpClient::new();
    anthropic_mock_with(&mock, &minimal_anthropic_events());
    let options = sse_options(&mock);
    let model = builtin_model("anthropic", "claude-fable-5-1");
    let context = Context {
        messages: vec![user_message_at("Hello", 1)],
        ..Context::default()
    };

    let result = stream_anthropic(&model, &context, Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    let beta = recorded_header(&mock, "anthropic-beta").expect("the managed betas header");
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
async fn uses_the_serving_model_input_transformations_from_the_final_stream_event() {
    let mut events = minimal_anthropic_events();
    events[0] = (
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_transformations",
                "model": "claude-fable-5-1",
                "usage": { "input_tokens": 12, "output_tokens": 0 },
                "input_transformations": [
                    { "type": "thinking_dropped", "path": "messages.1.content.0", "reason": "prefix_binding_mismatch" },
                ],
            },
        })
        .to_string(),
    );
    let mut delta =
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&events[4].1)
            .expect("the message_delta event parses");
    delta.insert(
        "input_transformations".to_owned(),
        json!([
            { "type": "thinking_dropped", "path": "messages.3.content.0", "reason": "model_binding_mismatch" },
        ]),
    );
    events[4] = (
        "message_delta",
        serde_json::Value::Object(delta).to_string(),
    );

    let mock = anthropic_mock(&events);
    let options = sse_options(&mock);
    let model = builtin_model("anthropic", "claude-fable-5-1");
    let context = Context {
        messages: vec![user_message_at("Hello", 1)],
        ..Context::default()
    };

    let result = stream_anthropic(&model, &context, Some(&options))
        .result()
        .await;

    let diagnostics = result.diagnostics.expect("the transformation diagnostic");
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].kind, "anthropic_input_transformations");
    let details = diagnostics[0].details.as_ref().expect("the details map");
    assert_eq!(
        details.get("transformations"),
        Some(&serde_json::json!([
            {
                "type": "thinking_dropped",
                "path": "messages.3.content.0",
                "reason": "model_binding_mismatch",
            },
        ])),
    );
}

#[tokio::test]
async fn repairs_malformed_sse_json_and_malformed_streamed_tool_json() {
    let model = builtin_model("anthropic", "claude-haiku-4-5");
    let tool = Tool {
        name: "edit".to_owned(),
        description: "Edit a file.".to_owned(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "text": { "type": "string" },
            },
            "required": ["path", "text"],
        }),
        constrained_sampling: None,
    };
    let context = Context {
        system_prompt: None,
        messages: vec![user_message_now("Use the edit tool.")],
        tools: Some(vec![tool]),
    };

    // String.raw upstream: an invalid `A\H` escape and a literal tab inside
    // the streamed partial JSON, both of which the JSON repair fixes.
    let malformed_tool_json_delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"A\H\",\"text\":\"col1	col2\"}"}}"#
        .to_string();
    let mock = anthropic_mock(&[
        common::message_start_event(
            "msg_test",
            json!({ "input_tokens": 12, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, }),
        ),
        common::block_start_event(
            0,
            json!({ "type": "tool_use", "id": "toolu_test", "name": "edit", "input": {} }),
        ),
        ("content_block_delta", malformed_tool_json_delta),
        common::block_stop_event(0),
        common::message_delta_event(
            json!({ "stop_reason": "tool_use" }),
            Some(
                json!({ "input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, }),
            ),
        ),
        common::message_stop_event(),
    ]);
    let options = sse_options(&mock);

    let result = stream_anthropic(&model, &context, Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(result.error_message, None);
    let tool_call = result.content.iter().find_map(|block| match block {
        pi_ai::types::AssistantBlock::ToolCall(tool_call) => Some(tool_call.clone()),
        _ => None,
    });
    let tool_call = tool_call.expect("the repaired tool call");
    assert_eq!(
        serde_json::Value::Object(tool_call.arguments),
        serde_json::json!({
            "path": "A\\H",
            "text": "col1\tcol2",
        }),
    );
}

#[tokio::test]
async fn preserves_content_from_content_block_start_events() {
    let model = builtin_model("anthropic", "claude-haiku-4-5");
    let context = Context {
        messages: vec![user_message_now("Say hello.")],
        ..Context::default()
    };
    let mock = anthropic_mock(&[
        common::message_start_event(
            "msg_initial_content",
            json!({ "input_tokens": 12, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, }),
        ),
        common::block_start_event(0, json!({ "type": "text", "text": "Initial text" })),
        common::block_delta_event(0, json!({ "type": "text_delta", "text": " plus delta" })),
        common::block_stop_event(0),
        common::block_start_event(
            1,
            json!({ "type": "thinking", "thinking": "Initial thinking", "signature": "initial signature", }),
        ),
        common::block_delta_event(
            1,
            json!({ "type": "thinking_delta", "thinking": " plus delta" }),
        ),
        common::block_delta_event(
            1,
            json!({ "type": "signature_delta", "signature": " plus delta" }),
        ),
        common::block_stop_event(1),
        common::message_delta_event(
            json!({ "stop_reason": "end_turn" }),
            Some(
                json!({ "input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, }),
            ),
        ),
        common::message_stop_event(),
    ]);
    let options = sse_options(&mock);

    let result = stream_anthropic(&model, &context, Some(&options))
        .result()
        .await;

    assert_eq!(
        result.content,
        vec![
            pi_ai::types::AssistantBlock::Text(pi_ai::types::TextContent {
                text: "Initial text plus delta".to_owned(),
                text_signature: None,
            }),
            pi_ai::types::AssistantBlock::Thinking(pi_ai::types::ThinkingContent {
                thinking: "Initial thinking plus delta".to_owned(),
                thinking_signature: Some("initial signature plus delta".to_owned()),
                redacted: None,
            }),
        ],
    );
}

/// Stream the built events through a fresh seam mock and settle the result,
/// the shape the event-mutation suites share.
async fn events_run(
    events: Vec<(&'static str, String)>,
    model: &pi_ai::types::Model,
    context: &Context,
) -> pi_ai::types::AssistantMessage {
    let mock = anthropic_mock(&events);
    let options = sse_options(&mock);
    stream_anthropic(model, context, Some(&options))
        .result()
        .await
}

#[tokio::test]
async fn preserves_refusal_stop_details_from_message_delta() {
    let model = builtin_model("anthropic", "claude-fable-5");
    let context = Context {
        messages: vec![user_message_now("blocked request")],
        ..Context::default()
    };
    let explanation = "This request triggered restrictions on violative cyber content and was blocked under Anthropic's Usage Policy. To learn more, provide feedback, or request an exemption based on how you use Claude, visit our help center: https://support.claude.com/en/articles/14604842-real-time-cyber-safeguards-on-claude.";
    let mock = anthropic_mock(&[
        common::message_start_event(
            "msg_01XFUDYJgAACzvnptvVoYEL",
            json!({ "input_tokens": 412, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, }),
        ),
        common::message_delta_event(
            json!({ "stop_reason": "refusal", "stop_details": { "type": "refusal", "category": "cyber", "explanation": explanation }, }),
            Some(
                json!({ "input_tokens": 412, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, }),
            ),
        ),
        common::message_stop_event(),
    ]);
    let options = sse_options(&mock);

    let result = stream_anthropic(&model, &context, Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.raw_stop_reason.as_deref(), Some("refusal"));
    assert_eq!(result.error_message.as_deref(), Some(explanation));
}

#[tokio::test]
async fn preserves_sensitive_stop_reasons_with_a_descriptive_error_message() {
    let model = builtin_model("anthropic", "claude-haiku-4-5");
    let context = Context {
        messages: vec![user_message_now("blocked request")],
        ..Context::default()
    };
    let mock = anthropic_mock(&[
        common::message_start_event(
            "msg_sensitive",
            json!({ "input_tokens": 12, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, }),
        ),
        common::message_delta_event(
            json!({ "stop_reason": "sensitive" }),
            Some(
                json!({ "input_tokens": 12, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, }),
            ),
        ),
        common::message_stop_event(),
    ]);
    let options = sse_options(&mock);

    let result = stream_anthropic(&model, &context, Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.raw_stop_reason.as_deref(), Some("sensitive"));
    assert_eq!(
        result.error_message.as_deref(),
        Some("Provider stopped with: sensitive")
    );
}

#[tokio::test]
async fn treats_message_delta_without_usage_as_a_no_op_for_usage_accumulation() {
    let model = builtin_model("anthropic", "claude-haiku-4-5");
    let context = Context {
        messages: vec![user_message_now("Say hello.")],
        ..Context::default()
    };
    let events: Vec<(&str, String)> = minimal_anthropic_events()
        .into_iter()
        .map(|(event, data)| {
            if event == "message_delta" {
                (
                    event,
                    json!({
                        "type": "message_delta",
                        "delta": { "stop_reason": "end_turn" },
                    })
                    .to_string(),
                )
            } else {
                (event, data)
            }
        })
        .collect();
    let result = events_run(events, &model, &context).await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.error_message, None);
    assert_eq!(
        result.content,
        vec![pi_ai::types::AssistantBlock::Text(
            pi_ai::types::TextContent {
                text: "Hello".to_owned(),
                text_signature: None,
            }
        )],
    );
    assert_eq!(result.usage.input, 12);
    assert_eq!(result.usage.total_tokens, 12);
}

#[tokio::test]
async fn ignores_unknown_sse_events_after_message_stop() {
    let model = builtin_model("anthropic", "claude-haiku-4-5");
    let context = Context {
        messages: vec![user_message_now("Say hello.")],
        ..Context::default()
    };
    let mut events = minimal_anthropic_events();
    events.push(("done", "[DONE]".to_owned()));
    events.push(("proxy.stats", "not json".to_owned()));
    let result = events_run(events, &model, &context).await;

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.error_message, None);
    assert_eq!(
        result.content,
        vec![pi_ai::types::AssistantBlock::Text(
            pi_ai::types::TextContent {
                text: "Hello".to_owned(),
                text_signature: None,
            }
        )],
    );
}

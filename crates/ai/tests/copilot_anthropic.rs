//! Copilot Claude via Anthropic Messages, ported from
//! `test/github-copilot-anthropic.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: the Copilot catalog's
//! adaptive-thinking maps, the Bearer auth with the Copilot header stack, and
//! the interleaved-thinking beta omitted for adaptive-thinking models.
//!
//! Seam adaptation: upstream mocks the Anthropic SDK and reads
//! `mockState.constructorOpts`/`createParams`; the port reads the seam mock's
//! recorded request — headers play the constructor's `defaultHeaders`, the
//! body plays `createParams`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::sync::Arc;

use pi_ai::api::anthropic_messages as anthropic;
use pi_ai::http::MockHttpClient;
use pi_ai::http::mock::RecordedRequest;
use pi_ai::models::get_supported_thinking_levels;
use pi_ai::types::{Context, Model, ModelThinkingLevel};
use serde_json::json;

fn copilot_model(id: &str) -> Model {
    common::builtin_model("github-copilot", id)
}

/// One map entry, read without moving the model.
fn mapped(model: &Model, level: ModelThinkingLevel) -> Option<String> {
    model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&level))
        .cloned()
        .unwrap_or_default()
}

fn context() -> Context {
    Context {
        system_prompt: Some(String::from("You are a helpful assistant.")),
        messages: vec![common::user_message_now("Hello")],
        tools: None,
    }
}

/// Upstream case 1: the Copilot Claude models carry the adaptive-thinking
/// maps the corporate gateway accepts.
#[test]
fn applies_copilot_specific_adaptive_thinking_effort_overrides() {
    let opus47 = copilot_model("claude-opus-4.7");
    assert_eq!(
        mapped(&opus47, ModelThinkingLevel::Minimal),
        Some("low".to_owned())
    );
    assert_eq!(
        mapped(&opus47, ModelThinkingLevel::Xhigh),
        Some("xhigh".to_owned())
    );
    assert_eq!(
        mapped(&opus47, ModelThinkingLevel::Max),
        Some("max".to_owned())
    );
    let supported = get_supported_thinking_levels(&opus47);
    assert!(supported.contains(&ModelThinkingLevel::Xhigh));
    assert!(supported.contains(&ModelThinkingLevel::Max));

    let opus5 = copilot_model("claude-opus-5");
    assert_eq!(opus5.api, pi_ai::types::Api::from("anthropic-messages"));
    assert_eq!(opus5.context_window, 1_000_000);
    assert_eq!(
        mapped(&opus5, ModelThinkingLevel::Minimal),
        Some("low".to_owned())
    );
    assert_eq!(
        mapped(&opus5, ModelThinkingLevel::Xhigh),
        Some("xhigh".to_owned())
    );
    assert_eq!(
        mapped(&opus5, ModelThinkingLevel::Max),
        Some("max".to_owned())
    );
    let supported = get_supported_thinking_levels(&opus5);
    assert!(supported.contains(&ModelThinkingLevel::Xhigh));
    assert!(supported.contains(&ModelThinkingLevel::Max));

    let sonnet46 = copilot_model("claude-sonnet-4.6");
    assert_eq!(
        mapped(&sonnet46, ModelThinkingLevel::Minimal),
        Some("low".to_owned())
    );
    assert_eq!(
        mapped(&sonnet46, ModelThinkingLevel::Max),
        Some("max".to_owned())
    );
    let supported = get_supported_thinking_levels(&sonnet46);
    assert!(supported.contains(&ModelThinkingLevel::Max));
    assert!(!supported.contains(&ModelThinkingLevel::Xhigh));
}

/// Capture the one request the empty mock fails: headers and body as the
/// adapter assembled them, upstream's constructor/params reads.
async fn capture_request(
    model: &Model,
    options: anthropic::AnthropicStreamOptions,
) -> RecordedRequest {
    let mock = MockHttpClient::new();
    let mut options = options;
    options.transport_options.http_client = Some(Arc::new(mock.clone()));
    let _ = anthropic::stream(model, &context(), Some(&options))
        .result()
        .await;
    let recorded = mock.recorded();
    assert_eq!(recorded.len(), 1, "one request dispatched");
    recorded.into_iter().next().expect("one request dispatched")
}

fn header_value<'a>(request: &'a RecordedRequest, name: &str) -> Option<&'a str> {
    request
        .headers
        .iter()
        .find(|(key, _value)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

/// Upstream case 2: Bearer auth, the Copilot static headers, the dynamic
/// per-request headers, and a valid Anthropic Messages payload.
#[tokio::test]
async fn uses_bearer_auth_copilot_headers_and_a_valid_anthropic_payload() {
    let model = copilot_model("claude-sonnet-4.6");
    let request = capture_request(
        &model,
        anthropic::AnthropicStreamOptions {
            api_key: Some(String::from("tid_copilot_session_test_token")),
            ..anthropic::AnthropicStreamOptions::default()
        },
    )
    .await;

    assert_eq!(
        header_value(&request, "Authorization"),
        Some("Bearer tid_copilot_session_test_token")
    );
    // Copilot static headers from the model's own header table.
    let user_agent = header_value(&request, "User-Agent").expect("the Copilot user agent");
    assert!(user_agent.contains("GitHubCopilotChat"), "{user_agent}");
    assert_eq!(
        header_value(&request, "Copilot-Integration-Id"),
        Some("vscode-chat")
    );
    // Dynamic headers.
    assert_eq!(header_value(&request, "X-Initiator"), Some("user"));
    assert_eq!(
        header_value(&request, "Openai-Intent"),
        Some("conversation-edits")
    );

    // The payload is valid Anthropic Messages format.
    let body: serde_json::Value =
        serde_json::from_slice(request.body.as_deref().expect("a request body"))
            .expect("the payload parses");
    let betas = body
        .get("betas")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(
        !betas
            .iter()
            .any(|beta| beta == "fine-grained-tool-streaming-2025-05-14")
    );
    assert_eq!(body["model"], json!("claude-sonnet-4.6"));
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["max_tokens"], json!(model.max_tokens));
    assert!(body["messages"].is_array());
}

/// Upstream case 3: requesting the interleaved-thinking beta on an
/// adaptive-thinking model must not add the beta.
#[tokio::test]
async fn omits_interleaved_thinking_beta_for_adaptive_thinking_models() {
    let model = copilot_model("claude-sonnet-4.6");
    let request = capture_request(
        &model,
        anthropic::AnthropicStreamOptions {
            api_key: Some(String::from("tid_copilot_session_test_token")),
            interleaved_thinking: Some(true),
            ..anthropic::AnthropicStreamOptions::default()
        },
    )
    .await;

    let body: serde_json::Value =
        serde_json::from_slice(request.body.as_deref().expect("a payload"))
            .expect("the payload parses");
    let betas = body
        .get("betas")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(
        !betas
            .iter()
            .any(|beta| beta == "interleaved-thinking-2025-05-14")
    );
}

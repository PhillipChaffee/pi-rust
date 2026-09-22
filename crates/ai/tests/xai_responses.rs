//! xAI Responses suites, ported from the upstream `xai-responses.test.ts` at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: upstream captures the request through a stubbed
//! `globalThis.fetch`; the port reads the same URL, headers, and body off the
//! [`MockHttpClient`] seam's recorded request. The two completions-tier
//! User-Agent cases upstream also carries exercise the openai-completions
//! wire and belong to the completions suites, so they are not ported here.

#![expect(
    clippy::expect_used,
    reason = "the tests pin stream outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::api::openai_responses::{self, OpenAiResponsesOptions};
use pi_ai::http::MockHttpClient;
use pi_ai::models::get_supported_thinking_levels;
use pi_ai::types::{
    Api, CacheRetention, Context, Message, Model, ModelThinkingLevel, ProviderId, StopReason,
    ThinkingLevel, UserContent, UserMessage,
};
use pi_ai::utils::pi_user_agent::get_pi_user_agent;
use serde_json::json;

mod common;
use common::{builtin_model, openai_responses_completed_event, openai_responses_mock_with};

/// Stream the capture and assert the clean stop, upstream's `captureRequest`
/// (whose fetch stub also asserts the settled stop reason).
async fn capture_request(
    model: &Model,
    context: &Context,
    mock: &MockHttpClient,
    mut options: OpenAiResponsesOptions,
) {
    openai_responses_mock_with(mock, &[openai_responses_completed_event()]);
    options.transport_options = common::mock_transport(mock);
    let stream = openai_responses::stream(model, context, Some(&options));
    let result = common::drain_and_settle(&stream).await;
    assert_eq!(
        result.stop_reason,
        StopReason::Stop,
        "{:?}",
        result.error_message
    );
}

/// The first recorded request's URL, upstream's `CapturedRequest.url`.
fn captured_url(mock: &MockHttpClient) -> String {
    mock.recorded()[0].url.clone()
}

/// The first recorded request's named header, case-insensitive.
fn recorded_header(mock: &MockHttpClient, name: &str) -> Option<String> {
    common::recorded_header(mock, name)
}

/// The hello-world context upstream's captures send.
fn hello_context() -> Context {
    Context {
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("hello".to_owned()),
            timestamp: 1,
        })],
        ..Context::default()
    }
}

/// The keyed options upstream's captures send: the xAI key over the default
/// transport.
fn xai_options() -> OpenAiResponsesOptions {
    OpenAiResponsesOptions {
        api_key: Some("xai-test-token".to_owned()),
        ..OpenAiResponsesOptions::default()
    }
}

#[tokio::test]
async fn excludes_retired_and_redundant_models_from_the_builtin_catalog() {
    let models = pi_ai::providers::all::builtin_models_of("xai");
    let ids: Vec<&str> = models.iter().map(|model| model.id.as_str()).collect();
    for model_id in [
        "grok-3",
        "grok-3-fast",
        "grok-4.20-0309-non-reasoning",
        "grok-4.20-0309-reasoning",
        "grok-build-0.1",
        "grok-code-fast-1",
    ] {
        assert!(
            !ids.contains(&model_id),
            "{model_id} must not ship in the catalog"
        );
    }
}

#[tokio::test]
async fn routes_every_builtin_xai_model_through_responses() {
    for model in pi_ai::providers::all::builtin_models_of("xai") {
        assert_eq!(model.api, Api::from("openai-responses"), "{}", model.id);
    }
    let levels = |id: &str| get_supported_thinking_levels(&builtin_model("xai", id));
    assert_eq!(
        levels("grok-4.5"),
        vec![
            ModelThinkingLevel::Low,
            ModelThinkingLevel::Medium,
            ModelThinkingLevel::High
        ]
    );
    assert_eq!(
        levels("grok-4.6"),
        vec![
            ModelThinkingLevel::Low,
            ModelThinkingLevel::Medium,
            ModelThinkingLevel::High,
            ModelThinkingLevel::Xhigh,
        ]
    );
    assert_eq!(
        levels("grok-4.3"),
        vec![
            ModelThinkingLevel::Off,
            ModelThinkingLevel::Low,
            ModelThinkingLevel::Medium,
            ModelThinkingLevel::High,
        ]
    );
}

#[tokio::test]
async fn uses_responses_with_bearer_auth_and_xai_compatible_request_fields() {
    let model = builtin_model("xai", "grok-4.5");
    let context = Context {
        system_prompt: Some("You are a careful coding assistant.".to_owned()),
        ..hello_context()
    };
    let mut options = xai_options();
    options.session_id = Some("pi-session-123".to_owned());
    options.cache_retention = Some(CacheRetention::Long);
    options.reasoning_effort = Some(ThinkingLevel::Medium);
    let mock = MockHttpClient::new();
    capture_request(&model, &context, &mock, options).await;

    assert_eq!(captured_url(&mock), "https://api.x.ai/v1/responses");
    assert_eq!(
        recorded_header(&mock, "authorization").as_deref(),
        Some("Bearer xai-test-token")
    );
    assert_eq!(
        recorded_header(&mock, "user-agent").as_deref(),
        Some(get_pi_user_agent().as_str())
    );
    assert_eq!(
        recorded_header(&mock, "session_id").as_deref(),
        Some("pi-session-123")
    );
    let body = common::recorded_body(&mock);
    assert_eq!(body["model"], json!("grok-4.5"));
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["prompt_cache_key"], json!("pi-session-123"));
    assert_eq!(body["reasoning"]["effort"], json!("medium"));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert!(body.get("prompt_cache_retention").is_none());
    assert!(
        body["input"]
            .as_array()
            .expect("the input list")
            .iter()
            .any(|item| item["role"] == json!("developer")
                && item["content"] == json!("You are a careful coding assistant."))
    );
}

#[tokio::test]
async fn requests_encrypted_reasoning_without_an_effort_override() {
    let model = builtin_model("xai", "grok-4.5");
    let context = hello_context();
    let mock = MockHttpClient::new();
    capture_request(&model, &context, &mock, xai_options()).await;

    let body = common::recorded_body(&mock);
    assert_eq!(body["model"], json!("grok-4.5"));
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert!(body.get("reasoning").is_none());
}

#[tokio::test]
async fn uses_responses_for_grok_4_6_with_xhigh_effort_and_encrypted_reasoning() {
    let model = builtin_model("xai", "grok-4.6");
    let context = Context {
        system_prompt: Some("You are a careful coding assistant.".to_owned()),
        ..hello_context()
    };
    let mut options = xai_options();
    options.reasoning_effort = Some(ThinkingLevel::Xhigh);
    let mock = MockHttpClient::new();
    capture_request(&model, &context, &mock, options).await;

    assert_eq!(captured_url(&mock), "https://api.x.ai/v1/responses");
    let body = common::recorded_body(&mock);
    assert_eq!(body["model"], json!("grok-4.6"));
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
}

#[tokio::test]
async fn uses_responses_for_grok_4_3() {
    let model = builtin_model("xai", "grok-4.3");
    let context = hello_context();
    let mut options = xai_options();
    options.reasoning_effort = Some(ThinkingLevel::Low);
    let mock = MockHttpClient::new();
    capture_request(&model, &context, &mock, options).await;

    assert_eq!(captured_url(&mock), "https://api.x.ai/v1/responses");
    let body = common::recorded_body(&mock);
    assert_eq!(body["model"], json!("grok-4.3"));
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(body["reasoning"]["effort"], json!("low"));
}

#[tokio::test]
async fn uses_the_pi_user_agent_by_default_for_responses_requests() {
    let model = Model {
        provider: ProviderId::from("openai"),
        base_url: "https://api.openai.com/v1".to_owned(),
        ..builtin_model("xai", "grok-4.5")
    };
    let context = hello_context();
    let mock = MockHttpClient::new();
    capture_request(&model, &context, &mock, xai_options()).await;

    assert_eq!(
        recorded_header(&mock, "user-agent").as_deref(),
        Some(get_pi_user_agent().as_str())
    );
}

#[tokio::test]
async fn lets_explicit_headers_override_the_default_responses_user_agent() {
    let model = builtin_model("xai", "grok-4.5");
    let context = hello_context();
    let mut options = xai_options();
    options.headers =
        Some(std::iter::once(("User-Agent".to_owned(), Some("custom-agent".to_owned()))).collect());
    let mock = MockHttpClient::new();
    capture_request(&model, &context, &mock, options).await;

    assert_eq!(
        recorded_header(&mock, "user-agent").as_deref(),
        Some("custom-agent")
    );
}

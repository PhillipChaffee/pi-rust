//! Anthropic auth token handling, ported from
//! `packages/ai/test/anthropic-auth-token.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream mocked the `@anthropic-ai/sdk` constructor and read
//! `constructorOpts`/`createParams` back; the port mounts the seam mock and
//! reads the recorded request's headers and body.

#![expect(
    clippy::expect_used,
    reason = "the tests pin stream outcomes; an unexpected shape panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::api::anthropic_messages::{AnthropicStreamOptions, stream as stream_anthropic};
use pi_ai::auth::types::{ApiKeyAuthInput, AuthContext};
use pi_ai::http::MockHttpClient;
use pi_ai::models::{CreateModelsOptions, ModelsSimpleStreamOptions};
use pi_ai::providers::anthropic::anthropic_provider;
use pi_ai::types::{
    Context, Message, ProviderHeaders, SimpleStreamOptions, TransportOptions, UserContent,
    UserMessage,
};
use serde_json::Value;

mod common;
use common::anthropic_model;

const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";
const ANTHROPIC_OAUTH_TOKEN_ENV: &str = "ANTHROPIC_OAUTH_TOKEN";

/// The env-only auth context the resolution tests drive, upstream's inline
/// `{ env: async (name) => ..., fileExists: async () => false }` context.
#[derive(Clone, Debug, Default)]
struct EnvAuthContext(BTreeMap<String, String>);

impl AuthContext for EnvAuthContext {
    fn env(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }

    fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

fn env_context(entries: &[(&str, &str)]) -> Arc<dyn AuthContext> {
    Arc::new(EnvAuthContext(
        entries
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect(),
    ))
}

/// The models runtime wired to the anthropic provider under the given env
/// auth context, the shape the auth-token suites stream through.
fn env_models(env: &[(&str, &str)]) -> pi_ai::models::Models {
    let models = pi_ai::models::create_models(Some(CreateModelsOptions {
        auth_context: Some(env_context(env)),
        ..CreateModelsOptions::default()
    }));
    models.set_provider(anthropic_provider());
    models
}

/// The seam-only simple options the auth-token suites send: the mock as the
/// transport and no credential of their own.
fn seam_options(mock: &MockHttpClient) -> ModelsSimpleStreamOptions {
    ModelsSimpleStreamOptions {
        options: SimpleStreamOptions {
            transport_options: TransportOptions {
                http_client: Some(Arc::new(mock.clone())),
                ..TransportOptions::default()
            },
            ..SimpleStreamOptions::default()
        },
        transform_headers: None,
    }
}

fn context() -> Context {
    Context {
        system_prompt: Some("System prompt.".to_owned()),
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("Hello".to_owned()),
            timestamp: pi_ai::auth::resolve::now_ms(),
        })],
        tools: None,
    }
}

fn recorded_header(mock: &MockHttpClient, name: &str) -> Option<String> {
    mock.recorded()[0]
        .headers
        .iter()
        .find(|(header, _)| header.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.clone())
}

fn kimi_coding_model() -> pi_ai::types::Model {
    let mut model = anthropic_model();
    "kimi-for-coding".clone_into(&mut model.id);
    "Kimi For Coding".clone_into(&mut model.name);
    model.provider = pi_ai::types::ProviderId::from("kimi-coding");
    "https://api.kimi.com/coding".clone_into(&mut model.base_url);
    model
}

#[tokio::test]
async fn resolves_anthropic_auth_token_as_a_bearer_authorization_header() {
    let provider = anthropic_provider();
    let resolve = provider
        .auth()
        .api_key
        .as_ref()
        .expect("anthropic api-key auth")
        .resolve
        .clone();
    let auth = (resolve)(ApiKeyAuthInput {
        ctx: env_context(&[
            ("ANTHROPIC_AUTH_TOKEN", "auth-token"),
            ("ANTHROPIC_OAUTH_TOKEN", "oauth-token"),
            ("ANTHROPIC_API_KEY", "api-key"),
        ]),
        credential: None,
        signal: pi_ai::utils::abort::operation_signal(None),
    })
    .await
    .expect("resolve")
    .expect("resolved");

    assert_eq!(auth.source.as_deref(), Some(ANTHROPIC_AUTH_TOKEN_ENV));
    assert_eq!(
        auth.auth
            .headers
            .as_ref()
            .and_then(|headers| headers.get("Authorization")),
        Some(&Some("Bearer auth-token".to_owned())),
    );
    assert_eq!(auth.auth.api_key, None);
}

#[tokio::test]
async fn preserves_anthropic_oauth_token_as_oauth_shaped_api_auth() {
    let provider = anthropic_provider();
    let resolve = provider
        .auth()
        .api_key
        .as_ref()
        .expect("anthropic api-key auth")
        .resolve
        .clone();
    let auth = (resolve)(ApiKeyAuthInput {
        ctx: env_context(&[
            ("ANTHROPIC_OAUTH_TOKEN", "oauth-token"),
            ("ANTHROPIC_API_KEY", "api-key"),
        ]),
        credential: None,
        signal: pi_ai::utils::abort::operation_signal(None),
    })
    .await
    .expect("resolve")
    .expect("resolved");

    assert_eq!(auth.source.as_deref(), Some(ANTHROPIC_OAUTH_TOKEN_ENV));
    assert_eq!(auth.auth.api_key.as_deref(), Some("oauth-token"));
}

#[tokio::test]
async fn uses_authorization_headers_without_oauth_mode_request_shaping() {
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let options = AnthropicStreamOptions {
        transport_options: common::mock_transport(&mock),
        headers: Some(ProviderHeaders::from([(
            "Authorization".to_owned(),
            Some("Bearer gateway-token".to_owned()),
        )])),
        ..AnthropicStreamOptions::default()
    };
    let result = stream_anthropic(&anthropic_model(), &context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, pi_ai::types::StopReason::Stop);
    assert!(recorded_header(&mock, "x-api-key").is_none());
    assert_eq!(
        recorded_header(&mock, "Authorization").as_deref(),
        Some("Bearer gateway-token")
    );
    let beta = recorded_header(&mock, "anthropic-beta");
    assert!(
        beta.as_ref()
            .is_none_or(|beta| !beta.contains("oauth-2025-04-20")),
        "got: {beta:?}"
    );
    let body: Value =
        serde_json::from_slice(mock.recorded()[0].body.as_ref().expect("body")).expect("body JSON");
    assert!(
        body["system"]
            .as_array()
            .expect("system blocks")
            .iter()
            .any(|block| block["text"] == "System prompt."),
    );
}

#[tokio::test]
async fn threads_auth_context_anthropic_auth_token_through_request_headers() {
    let models = env_models(&[(ANTHROPIC_AUTH_TOKEN_ENV, "ctx-token")]);
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let options = seam_options(&mock);

    let result = models
        .stream_simple(&anthropic_model(), &context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, pi_ai::types::StopReason::Stop);
    assert!(recorded_header(&mock, "x-api-key").is_none());
    assert_eq!(
        recorded_header(&mock, "Authorization").as_deref(),
        Some("Bearer ctx-token")
    );
    let beta = recorded_header(&mock, "anthropic-beta");
    assert!(
        beta.as_ref()
            .is_none_or(|beta| !beta.contains("oauth-2025-04-20")),
        "got: {beta:?}"
    );
}

#[tokio::test]
async fn preserves_oauth_request_shaping_for_anthropic_oauth_token() {
    let models = env_models(&[(ANTHROPIC_OAUTH_TOKEN_ENV, "sk-ant-oat-test")]);
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let options = seam_options(&mock);

    let result = models
        .stream_simple(&anthropic_model(), &context(), Some(&options))
        .result()
        .await;

    assert_eq!(result.stop_reason, pi_ai::types::StopReason::Stop);
    assert!(recorded_header(&mock, "x-api-key").is_none());
    assert_eq!(
        recorded_header(&mock, "Authorization").as_deref(),
        Some("Bearer sk-ant-oat-test")
    );
    let beta = recorded_header(&mock, "anthropic-beta").expect("oauth betas");
    assert!(beta.contains("oauth-2025-04-20"), "got: {beta}");
    assert!(beta.contains("claude-code-20250219"), "got: {beta}");
    let body: Value =
        serde_json::from_slice(mock.recorded()[0].body.as_ref().expect("body")).expect("body JSON");
    assert!(body["system"].as_array().is_some_and(|blocks| {
        blocks.iter().any(|block| {
            block["text"] == "You are Claude Code, Anthropic's official CLI for Claude."
        })
    }));
}

#[tokio::test]
async fn lets_explicit_request_headers_override_anthropic_auth_token() {
    let models = env_models(&[(ANTHROPIC_AUTH_TOKEN_ENV, "ctx-token")]);
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let mut options = seam_options(&mock);
    options.options.headers = Some(ProviderHeaders::from([(
        "Authorization".to_owned(),
        Some("Bearer explicit-token".to_owned()),
    )]));

    let _ = models
        .stream_simple(&anthropic_model(), &context(), Some(&options))
        .result()
        .await;

    assert_eq!(
        recorded_header(&mock, "Authorization").as_deref(),
        Some("Bearer explicit-token")
    );
}

// -- Anthropic-compatible user agents ---------------------------------------

#[tokio::test]
async fn uses_pis_user_agent_by_default_for_anthropic_messages_requests() {
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let options = AnthropicStreamOptions {
        transport_options: common::mock_transport(&mock),
        api_key: Some("anthropic-key".to_owned()),
        ..AnthropicStreamOptions::default()
    };
    let _ = stream_anthropic(&anthropic_model(), &context(), Some(&options))
        .result()
        .await;

    assert_eq!(
        recorded_header(&mock, "User-Agent"),
        Some(pi_ai::utils::pi_user_agent::get_pi_user_agent()),
    );
}

#[tokio::test]
async fn lets_explicit_headers_override_the_default_anthropic_messages_user_agent() {
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let options = AnthropicStreamOptions {
        transport_options: common::mock_transport(&mock),
        api_key: Some("kimi-key".to_owned()),
        headers: Some(ProviderHeaders::from([(
            "User-Agent".to_owned(),
            Some("custom-client".to_owned()),
        )])),
        ..AnthropicStreamOptions::default()
    };
    let _ = stream_anthropic(&kimi_coding_model(), &context(), Some(&options))
        .result()
        .await;

    assert_eq!(
        recorded_header(&mock, "User-Agent").as_deref(),
        Some("custom-client")
    );
}

#[tokio::test]
async fn preserves_explicit_anthropic_beta_header_replacement() {
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let options = AnthropicStreamOptions {
        transport_options: common::mock_transport(&mock),
        api_key: Some("anthropic-key".to_owned()),
        headers: Some(ProviderHeaders::from([(
            "anthropic-beta".to_owned(),
            Some("custom-beta".to_owned()),
        )])),
        ..AnthropicStreamOptions::default()
    };
    let _ = stream_anthropic(&anthropic_model(), &context(), Some(&options))
        .result()
        .await;

    assert_eq!(
        recorded_header(&mock, "anthropic-beta").as_deref(),
        Some("custom-beta")
    );
}

#[tokio::test]
async fn preserves_explicit_anthropic_beta_header_suppression() {
    let mock = MockHttpClient::new();
    common::anthropic_mock_with(&mock, &common::minimal_anthropic_done());
    let options = AnthropicStreamOptions {
        transport_options: common::mock_transport(&mock),
        api_key: Some("anthropic-key".to_owned()),
        headers: Some(ProviderHeaders::from([("anthropic-beta".to_owned(), None)])),
        ..AnthropicStreamOptions::default()
    };
    let _ = stream_anthropic(&anthropic_model(), &context(), Some(&options))
        .result()
        .await;

    assert_eq!(recorded_header(&mock, "anthropic-beta"), None);
}

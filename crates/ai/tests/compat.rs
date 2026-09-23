//! The compat legacy-API fallback suite, ported from
//! `packages/ai/test/compat-env.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`: an unknown provider dispatches
//! through the legacy api-registry, with the explicit request key passing
//! through the env-key merge untouched.

use pi_ai::compat::{ApiProvider, complete, register_api_provider, reset_api_providers};
use pi_ai::types::{
    Api, Context, Message, Model, ProviderId, StreamOptions, UserContent, UserMessage,
};

mod common;

/// The openai-responses model behind a custom provider, upstream's `model`.
fn model() -> Model {
    Model {
        api: Api::from("openai-responses"),
        provider: ProviderId::from("custom-openai"),
        base_url: "https://example.test/v1".to_owned(),
        context_window: 128_000,
        max_tokens: 4096,
        ..common::fixture_model()
    }
}

/// The single user-message context, upstream's `context`.
fn context() -> Context {
    Context {
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("hi".to_owned()),
            timestamp: pi_ai::auth::resolve::now_ms(),
        })],
        ..Context::default()
    }
}

/// The unknown provider dispatches through the legacy api-registry and the
/// explicit request key reaches the provider untouched, upstream's
/// `dispatches unknown providers through the legacy API registry`.
#[tokio::test]
async fn dispatches_unknown_providers_through_the_legacy_api_registry() {
    let _guard = common::registry_guard().await;
    reset_api_providers();
    let capture = common::registry_capture();
    register_api_provider(
        ApiProvider {
            api: Api::from("openai-responses"),
            streams: capture.streams.clone(),
        },
        None,
    );

    let model = model();
    complete(
        &model,
        &context(),
        Some(&StreamOptions {
            api_key: Some("request-key".to_owned()),
            ..StreamOptions::default()
        }),
    )
    .await;

    assert_eq!(capture.captured_key().as_deref(), Some("request-key"));
    reset_api_providers();
}

//! The compat registry boundary suite, binding the branches the upstream
//! suites leave untested at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`:
//! the no-clobber builtin registration, the reset, the source-scoped
//! unregister, the registry's api check, the unregistered-api failure, and
//! the env-key merge branches (`withEnvApiKey`).

#![expect(
    clippy::expect_used,
    reason = "the tests pin dispatch outcomes; an unexpected shape panics the test by design"
)]

use std::collections::BTreeMap;

use pi_ai::compat::{
    ApiProvider, builtin_apis, get_api_provider, get_api_providers, register_api_provider,
    register_built_in_api_providers, reset_api_providers, stream, stream_simple,
    unregister_api_providers,
};
use pi_ai::types::{
    Api, Context, Model, ProviderId, SimpleStreamOptions, StopReason, StreamOptions,
};

mod common;

/// The model dispatching through a registry provider, with the provider id
/// and env map the env-key merge reads.
fn registry_model(provider: &str) -> Model {
    Model {
        api: Api::from("openai-responses"),
        provider: ProviderId::from(provider),
        base_url: "https://example.test/v1".to_owned(),
        ..common::fixture_model()
    }
}

/// A registered override for the openai-responses api under `source_id`,
/// upstream's extension-shaped registration.
fn register_capture(source_id: Option<&str>) -> common::RegistryCapture {
    let capture = common::registry_capture();
    register_api_provider(
        ApiProvider {
            api: Api::from("openai-responses"),
            streams: capture.streams.clone(),
        },
        source_id,
    );
    capture
}

/// The env-resolved api key merges into options that lack one, the
/// `withEnvApiKey` branch upstream's `hasExplicitApiKey` guard leaves open.
#[tokio::test]
async fn the_env_resolved_api_key_merges_into_options() {
    let _guard = common::registry_guard().await;
    reset_api_providers();
    let capture = register_capture(None);

    let model = registry_model("xai");
    let options = SimpleStreamOptions {
        env: Some(BTreeMap::from([(
            "XAI_API_KEY".to_owned(),
            "env-key".to_owned(),
        )])),
        ..SimpleStreamOptions::default()
    };
    stream_simple(&model, &Context::default(), Some(&options))
        .result()
        .await;

    assert_eq!(capture.captured_key().as_deref(), Some("env-key"));
    reset_api_providers();
}

/// A blank explicit key is no key: the env lookup still runs, upstream's
/// `hasExplicitApiKey` trim check.
#[tokio::test]
async fn a_blank_explicit_key_still_resolves_from_the_env() {
    let _guard = common::registry_guard().await;
    reset_api_providers();
    let capture = register_capture(None);

    let model = registry_model("xai");
    let options = SimpleStreamOptions {
        api_key: Some("   ".to_owned()),
        env: Some(BTreeMap::from([(
            "XAI_API_KEY".to_owned(),
            "env-key".to_owned(),
        )])),
        ..SimpleStreamOptions::default()
    };
    stream_simple(&model, &Context::default(), Some(&options))
        .result()
        .await;

    assert_eq!(capture.captured_key().as_deref(), Some("env-key"));
    reset_api_providers();
}

/// The ambient-auth marker is left alone: a provider marked
/// `<authenticated>` resolves credentials itself, upstream's
/// `AMBIENT_AUTH_MARKER` guard.
#[tokio::test]
async fn the_ambient_auth_marker_is_left_alone() {
    let _guard = common::registry_guard().await;
    reset_api_providers();
    let capture = register_capture(None);

    let model = registry_model("xai");
    let options = SimpleStreamOptions {
        env: Some(BTreeMap::from([(
            "XAI_API_KEY".to_owned(),
            "<authenticated>".to_owned(),
        )])),
        ..SimpleStreamOptions::default()
    };
    stream_simple(&model, &Context::default(), Some(&options))
        .result()
        .await;

    assert_eq!(capture.captured_key(), None);
    reset_api_providers();
}

/// Registering the builtins never clobbers an existing entry — compat may
/// load after an override stands — upstream's `registerBuiltInApiProviders`
/// guard: the override keeps serving an unknown provider's dispatch.
#[tokio::test]
async fn builtin_registration_never_clobbers_an_override() {
    let _guard = common::registry_guard().await;
    reset_api_providers();
    let capture = register_capture(Some("test-source"));
    register_built_in_api_providers();

    let model = registry_model("custom-openai");
    let options = SimpleStreamOptions {
        api_key: Some("override-key".to_owned()),
        ..SimpleStreamOptions::default()
    };
    stream_simple(&model, &Context::default(), Some(&options))
        .result()
        .await;

    assert_eq!(capture.captured_key().as_deref(), Some("override-key"));
    reset_api_providers();
}

/// The reset restores the builtin api implementations, upstream's
/// `resetApiProviders`.
#[tokio::test]
async fn the_reset_restores_the_builtin_registrations() {
    let _guard = common::registry_guard().await;
    register_capture(None);
    reset_api_providers();

    let providers = get_api_providers();
    assert_eq!(providers.len(), 10);
    for api in builtin_apis().iter().map(|(api, _)| Api::from(*api)) {
        assert!(get_api_provider(&api).is_some(), "{api} stays registered");
    }
}

/// Unregistering by source id removes exactly that source's registrations,
/// upstream's `unregisterApiProviders`: the override replaced the builtin,
/// so the api is unregistered until the reset.
#[tokio::test]
async fn unregistering_by_source_id_scopes_the_removal() {
    let _guard = common::registry_guard().await;
    reset_api_providers();
    register_capture(Some("test-source"));
    register_built_in_api_providers();

    unregister_api_providers("test-source");
    let model = registry_model("xai");
    let options = SimpleStreamOptions::default();
    let message = stream_simple(&model, &Context::default(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("No API provider registered for api: openai-responses")
    );
    reset_api_providers();
}

/// A registered provider reached with a mismatched model's api settles the
/// stream as an error carrying upstream's `Mismatched api` message, the
/// `wrapStream` check the registry re-adds.
#[tokio::test]
async fn a_mismatched_api_fails_the_stream_with_the_registry_message() {
    let _guard = common::registry_guard().await;
    reset_api_providers();
    register_capture(None);

    let model = Model {
        api: Api::from("anthropic-messages"),
        ..registry_model("xai")
    };
    let streams = get_api_provider(&Api::from("openai-responses")).expect("the override");
    let message = streams
        .stream(&model, &Context::default(), None)
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Mismatched api: anthropic-messages expected openai-responses")
    );
    reset_api_providers();
}

/// An api with no registration settles the dispatch as an error carrying
/// upstream's `No API provider registered` message, the sync throw the
/// stream-based dispatch re-expresses.
#[tokio::test]
async fn an_unregistered_api_fails_with_the_dispatch_message() {
    let _guard = common::registry_guard().await;
    reset_api_providers();

    let model = Model {
        api: Api::from("unheard-of"),
        ..registry_model("xai")
    };
    let options = StreamOptions::default();
    let message = stream(&model, &Context::default(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("No API provider registered for api: unheard-of")
    );
    reset_api_providers();
}

/// An unregistered api fails the simple dispatch with the same message.
#[tokio::test]
async fn an_unregistered_api_fails_the_simple_dispatch_with_the_dispatch_message() {
    let _guard = common::registry_guard().await;
    reset_api_providers();

    let model = Model {
        api: Api::from("unheard-of"),
        ..registry_model("xai")
    };
    let options = SimpleStreamOptions::default();
    let message = stream_simple(&model, &Context::default(), Some(&options))
        .result()
        .await;

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("No API provider registered for api: unheard-of")
    );
    reset_api_providers();
}

/// Every builtin api id carries an implementation after the reset.
#[tokio::test]
async fn every_builtin_api_carries_an_implementation() {
    let _guard = common::registry_guard().await;
    reset_api_providers();
    for (api, _) in builtin_apis() {
        assert!(
            get_api_provider(&Api::from(*api)).is_some(),
            "{api} carries a builtin implementation"
        );
    }
    reset_api_providers();
}

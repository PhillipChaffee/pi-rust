//! Upstream `packages/coding-agent/test/model-registry.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated for
//! `pi_coding_agent::model_registry` (#121). 83 cases across the five
//! describe blocks, one `mod` per describe.
//!
//! Porting restatements this suite records:
//!
//! - Upstream sets `process.env` in seven api-key-resolution cases; the
//!   workspace forbids mutating the process environment, so those cases
//!   drive `runtime().get_auth(provider, Some(&ModelRuntimeAuthOverrides {
//!   env: Some(map), .. }))` — the resolution overlays the map over the
//!   process env — and assert the resolved key. Covered: the `$VAR` and
//!   `${VAR}` forms, the interpolated `${A}_$B` form, the `$!`-escaped
//!   literal with a trailing env ref, both auth-status env cases, and the
//!   not-cached case (two lookups with different overlay maps). Absence
//!   cases run against the real process env on names no runner sets.
//! - "plain apiKey is used directly even when it matches an env var" cannot
//!   set the matching env var; the literal branch never reads the
//!   environment, so the case pins the literal outcome alone.
//! - `registry.refresh()` restates as `refresh(Some(allow_network: false))`:
//!   the port's bare refresh inherits the network default from `PI_OFFLINE`
//!   in the process environment, and the models.json reload and
//!   recomposition these cases exercise run identically offline.
//! - "registerProvider treats uppercase apiKey and headers as literals"
//!   drops the `console.warn` spy assertion: no observable surface carries
//!   the warning.
//! - "failed registerProvider does not persist invalid streamSimple config"
//!   restates upstream's throwing `streamSimple` probe as an invocation
//!   counter on the closure, which proves registration never invokes it.
//! - "streamSimple overlays do not mutate the global compat API registry"
//!   pins `Arc::ptr_eq` identity of the `openai-completions` entry across
//!   registration and unregistration; upstream invokes the global api's
//!   stream instead, which would open a network request here.
//! - Upstream's `afterEach` calls `clearApiKeyCache`; the command-resolving
//!   cases here clear the cache at their start. The resolution paths those
//!   cases drive are the uncached or-throw variants, so the clear is
//!   hygiene, not behavior.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::float_cmp,
    reason = "the cost assertions pin exact upstream rate values"
)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use pi_ai::auth::resolve::now_ms;
use pi_ai::auth::types::{
    ApiKeyCredential, AuthResult, Credential, CredentialModifyFn, OAuthCredentials,
};
use pi_ai::models::{ModelsRefreshOptions, ProviderError};
use pi_ai::providers::catalog::get_builtin_models;
use pi_ai::types::{
    Api, CacheControlFormat, MaxTokensField, Modality, Model, ModelCompat, ModelThinkingLevel,
    OpenRouterRouting, ProviderEnv, ProviderHeaders, ThinkingFormat, ThinkingLevelMap,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_coding_agent::auth_storage::{AuthStorage, InMemoryAuthStorageBackend};
use pi_coding_agent::model_registry::{ModelRegistry, ResolvedRequestAuth};
use pi_coding_agent::model_runtime::ModelRuntimeAuthOverrides;
use pi_coding_agent::provider_composer::{
    AuthStatus, AuthStatusSource, ExtensionOAuthConfig, ExtensionOAuthLoginFn,
    ExtensionOAuthRefreshFn, ExtensionStreamSimpleFn, ProviderConfigInput, ProviderModelInput,
};
use pi_coding_agent::resolve_config_value::clear_config_value_cache;

#[expect(
    dead_code,
    reason = "the fixture module is compiled into every test binary and this suite consumes only the model fixtures"
)]
mod common;

/// The per-test rig: a temp dir whose lifetime the test binds and the
/// models.json path inside it, upstream's `tempDir` + `modelsJsonPath`.
fn rig() -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("the test temp dir creates");
    let models_path = dir.path().join("models.json").display().to_string();
    (dir, models_path)
}

/// Write raw providers JSON, upstream's `writeRawModelsJson`/
/// `writeModelsJson`.
#[expect(
    clippy::needless_pass_by_value,
    reason = "the providers value reads clearest as an owned argument at the 50 call sites"
)]
fn write_raw_models_json(path: &str, providers: serde_json::Value) {
    std::fs::write(
        path,
        serde_json::json!({ "providers": providers }).to_string(),
    )
    .expect("models.json writes");
}

/// The minimal provider config, upstream's `providerConfig`.
fn json_provider_config(
    base_url: &str,
    models: &[(&str, Option<&str>)],
    api: &str,
) -> serde_json::Value {
    let models: Vec<serde_json::Value> = models
        .iter()
        .map(|(id, name)| {
            serde_json::json!({
                "id": id,
                "name": name.unwrap_or(id),
                "reasoning": false,
                "input": ["text"],
                "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                "contextWindow": 100_000,
                "maxTokens": 8_000,
            })
        })
        .collect();
    serde_json::json!({
        "baseUrl": base_url,
        "apiKey": "test-key",
        "api": api,
        "models": models,
    })
}

/// The baseUrl-only override, upstream's `overrideConfig`: the headers key
/// is absent when unset, like upstream's spread.
fn json_override_config(base_url: &str, headers: Option<&serde_json::Value>) -> serde_json::Value {
    let mut config = serde_json::json!({ "baseUrl": base_url });
    if let Some(headers) = headers {
        config["headers"] = headers.clone();
    }
    config
}

/// The single-model provider config upstream's `providerWithApiKey` spells.
fn json_provider_with_api_key(api_key: &str) -> serde_json::Value {
    serde_json::json!({
        "baseUrl": "https://example.com/v1",
        "apiKey": api_key,
        "api": "anthropic-messages",
        "models": [{
            "id": "test-model",
            "name": "Test Model",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 100_000,
            "maxTokens": 8_000,
        }],
    })
}

/// One provider's composed models, upstream's `getModelsForProvider`.
fn models_for_provider(registry: &ModelRegistry, provider: &str) -> Vec<Model> {
    registry
        .get_all()
        .into_iter()
        .filter(|model| model.provider.0 == provider)
        .collect()
}

/// The `/bin/sh`-safe path spelling, upstream's `toShPath`.
fn to_sh_path(value: &str) -> String {
    value.replace('\\', "/").replace('"', "\\\"")
}

/// The environment map the override seams carry, from plain pairs.
fn env_from(entries: &[(&str, &str)]) -> ProviderEnv {
    entries
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect()
}

/// The `ProviderHeaders` the assertions spell, from plain pairs.
fn ok_headers(entries: &[(&str, &str)]) -> ProviderHeaders {
    entries
        .iter()
        .map(|(name, value)| ((*name).to_owned(), Some((*value).to_owned())))
        .collect()
}

/// The `Ok` resolution with no base URL or env, the common assertion shape.
fn resolved_ok(api_key: Option<&str>, headers: Option<ProviderHeaders>) -> ResolvedRequestAuth {
    ResolvedRequestAuth::Ok {
        api_key: api_key.map(str::to_owned),
        headers,
        base_url: None,
        env: None,
    }
}

/// The refresh options every `refresh()` case carries. Upstream's bare
/// `registry.refresh()` inherits the network default from the runner
/// environment; the port reads `PI_OFFLINE` from the process environment,
/// which the workspace forbids mutating, so the suite pins the offline
/// refresh. The models.json reload and recomposition these cases exercise
/// run identically offline.
fn offline_refresh() -> ModelsRefreshOptions {
    ModelsRefreshOptions {
        allow_network: Some(false),
        ..ModelsRefreshOptions::default()
    }
}

/// The provider auth with an env overlay, the seam the env-backed cases
/// drive in place of `process.env` writes: the resolution overlays the map
/// over the process environment.
async fn resolve_auth_with_env(
    registry: &ModelRegistry,
    provider: &str,
    env: &[(&str, &str)],
) -> Option<AuthResult> {
    let overrides = ModelRuntimeAuthOverrides {
        env: Some(env_from(env)),
        ..ModelRuntimeAuthOverrides::default()
    };
    registry
        .runtime()
        .get_auth(provider, Some(&overrides))
        .await
        .expect("the provider auth resolves")
}

/// The extension model input upstream's inline registration models spell,
/// with the fixture pricing and the registration's api.
fn extension_model(id: &str, api: Api) -> ProviderModelInput {
    ProviderModelInput {
        id: id.to_owned(),
        name: id.to_owned(),
        api: Some(api),
        base_url: None,
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: common::model_layer::zero_cost(),
        context_window: 128_000,
        max_tokens: 4_096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The extension registration upstream's `providerConfig(...)` spells: a
/// base URL, the fixture key, a wire api, and the model list.
fn registration(base_url: &str, api: Api, models: Vec<ProviderModelInput>) -> ProviderConfigInput {
    ProviderConfigInput {
        name: None,
        base_url: Some(base_url.to_owned()),
        api_key: Some("test-key".to_owned()),
        api: Some(api),
        stream_simple: None,
        headers: None,
        auth_header: None,
        oauth: None,
        models: Some(models),
        refresh_models: None,
    }
}

/// The login closure that never runs, the fixture every display-name
/// OAuth registration carries.
fn never_login() -> ExtensionOAuthLoginFn {
    Arc::new(|_callbacks| {
        Box::pin(async {
            let error: ProviderError = "login never runs".into();
            Err(error)
        })
    })
}

/// The refresh closure that echoes its credential, upstream's
/// `refreshToken: async (credentials) => credentials`.
fn echo_refresh() -> ExtensionOAuthRefreshFn {
    Arc::new(|credentials, _signal| Box::pin(async move { Ok(credentials) }))
}

/// The stream override that is never observed through, the closure the
/// global-registry isolation and streamSimple-validation cases register.
fn never_stream() -> ExtensionStreamSimpleFn {
    Arc::new(|_model, _context, _options| AssistantMessageEventStream::new(|_| false, |_| None))
}

/// Store one api-key credential, upstream's `authStorage.modify` seeds.
async fn store_api_key(
    storage: &AuthStorage<InMemoryAuthStorageBackend>,
    provider: &str,
    key: &str,
    env: Option<ProviderEnv>,
) {
    let credential = Credential::ApiKey(ApiKeyCredential {
        key: Some(key.to_owned()),
        env,
    });
    let modify: CredentialModifyFn =
        Box::new(move |_current| Box::pin(async move { Ok(Some(credential)) }));
    storage
        .modify(provider, modify, None)
        .await
        .expect("the credential stores");
}

/// The fixture registry over in-memory credentials and the test's
/// models.json path, upstream's `createModelRegistry`.
async fn create_registry(models_path: &str) -> ModelRegistry {
    common::model_layer::create_model_registry(
        common::model_layer::empty_auth_storage(),
        Some(models_path),
    )
    .await
}

/// The counter file's parsed value, upstream's `parseInt(readFileSync(...))`.
fn counter_count(path: &std::path::Path) -> i64 {
    std::fs::read_to_string(path)
        .expect("the counter file reads")
        .trim()
        .parse()
        .expect("the counter parses")
}

mod base_url_override_no_custom_models {
    use super::*;

    #[tokio::test]
    async fn overriding_base_url_keeps_all_built_in_models() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_override_config("https://my-proxy.example.com/v1", None),
            }),
        );

        let registry = create_registry(&models_path).await;
        let anthropic_models = models_for_provider(&registry, "anthropic");

        assert!(
            anthropic_models.len() > 1,
            "the built-in catalog survives the override"
        );
        assert!(
            anthropic_models
                .iter()
                .any(|model| model.id.contains("claude")),
            "the built-in ids survive the override"
        );
    }

    #[tokio::test]
    async fn overriding_base_url_changes_url_on_all_built_in_models() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_override_config("https://my-proxy.example.com/v1", None),
            }),
        );

        let registry = create_registry(&models_path).await;
        for model in models_for_provider(&registry, "anthropic") {
            assert_eq!(model.base_url, "https://my-proxy.example.com/v1");
        }
    }

    #[tokio::test]
    async fn overriding_headers_resolves_at_request_time() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_override_config(
                    "https://my-proxy.example.com/v1",
                    Some(&serde_json::json!({ "X-Custom-Header": "custom-value" })),
                ),
            }),
        );

        let registry = create_registry(&models_path).await;
        for model in models_for_provider(&registry, "anthropic") {
            let auth = registry.get_api_key_and_headers(&model).await;
            assert!(auth.ok());
            let ResolvedRequestAuth::Ok { headers, .. } = auth else {
                unreachable!("auth.ok() pinned the Ok arm");
            };
            assert_eq!(
                headers
                    .as_ref()
                    .and_then(|headers| headers.get("X-Custom-Header")),
                Some(&Some("custom-value".to_owned())),
            );
        }
    }

    #[tokio::test]
    async fn headers_only_override_resolves_at_request_time() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": {
                    "headers": { "X-Custom-Header": "custom-value" },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        assert!(registry.get_error().is_none());
        for model in models_for_provider(&registry, "anthropic") {
            let auth = registry.get_api_key_and_headers(&model).await;
            assert!(auth.ok());
            let ResolvedRequestAuth::Ok { headers, .. } = auth else {
                unreachable!("auth.ok() pinned the Ok arm");
            };
            assert_eq!(
                headers
                    .as_ref()
                    .and_then(|headers| headers.get("X-Custom-Header")),
                Some(&Some("custom-value".to_owned())),
            );
        }
    }

    #[tokio::test]
    async fn unconfigured_compatibility_auth_includes_static_model_headers() {
        let (_dir, models_path) = rig();
        let registry = create_registry(&models_path).await;
        let base = registry
            .get_all()
            .into_iter()
            .next()
            .expect("the builtin catalog is non-empty");
        let model = Model {
            provider: pi_ai::types::ProviderId("missing-provider".to_owned()),
            headers: Some(BTreeMap::from([(
                "X-Static-Model".to_owned(),
                "static-value".to_owned(),
            )])),
            ..base
        };

        let auth = registry.get_api_key_and_headers(&model).await;

        assert_eq!(
            auth,
            resolved_ok(
                None,
                Some(ok_headers(&[("X-Static-Model", "static-value")]))
            )
        );
    }

    #[tokio::test]
    async fn base_url_only_override_does_not_affect_other_providers() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_override_config("https://my-proxy.example.com/v1", None),
            }),
        );

        let registry = create_registry(&models_path).await;
        let google_models = models_for_provider(&registry, "google");

        assert!(!google_models.is_empty(), "google models survive");
        assert_ne!(
            google_models[0].base_url, "https://my-proxy.example.com/v1",
            "the google base URL is untouched"
        );
    }

    #[tokio::test]
    async fn can_mix_base_url_override_and_models_merge() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_override_config("https://anthropic-proxy.example.com/v1", None),
                "google": json_provider_config(
                    "https://google-proxy.example.com/v1",
                    &[("gemini-custom", None)],
                    "google-generative-ai",
                ),
            }),
        );

        let registry = create_registry(&models_path).await;

        let anthropic_models = models_for_provider(&registry, "anthropic");
        assert!(anthropic_models.len() > 1);
        assert_eq!(
            anthropic_models[0].base_url,
            "https://anthropic-proxy.example.com/v1"
        );

        let google_models = models_for_provider(&registry, "google");
        assert!(google_models.len() > 1);
        assert!(
            google_models
                .iter()
                .any(|model| model.id == "gemini-custom")
        );
    }

    #[tokio::test]
    async fn refresh_picks_up_base_url_override_changes() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_override_config("https://first-proxy.example.com/v1", None),
            }),
        );
        let registry = create_registry(&models_path).await;
        assert_eq!(
            models_for_provider(&registry, "anthropic")[0].base_url,
            "https://first-proxy.example.com/v1"
        );

        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_override_config("https://second-proxy.example.com/v1", None),
            }),
        );
        registry.refresh(Some(offline_refresh())).await;

        assert_eq!(
            models_for_provider(&registry, "anthropic")[0].base_url,
            "https://second-proxy.example.com/v1"
        );
    }
}

mod custom_models_merge_behavior {
    use super::*;

    #[tokio::test]
    async fn built_in_provider_custom_models_inherit_api_and_base_url_without_explicit_fields() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "models": [{
                        "id": "fake-provider/fake-model",
                        "name": "Fake model",
                        "reasoning": true,
                        "input": ["text"],
                    }],
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        assert!(registry.get_error().is_none());

        let model = registry
            .find("openrouter", "fake-provider/fake-model")
            .expect("the custom model composes");
        assert_eq!(model.api.0, "openai-completions");
        assert_eq!(model.base_url, "https://openrouter.ai/api/v1");
    }

    #[tokio::test]
    async fn non_built_in_provider_custom_models_still_require_base_url() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "my-custom-provider": {
                    "apiKey": "test-key",
                    "models": [{
                        "id": "my-model",
                        "api": "openai-completions",
                        "reasoning": false,
                        "input": ["text"],
                    }],
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let error = registry.get_error().expect("the composition error reports");
        assert!(
            error.contains("baseUrl"),
            "the error names baseUrl: {error}"
        );
    }

    #[tokio::test]
    async fn reports_every_provider_composition_error() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "broken-one": { "api": "openai-completions", "models": [{ "id": "one" }] },
                "broken-two": { "api": "openai-completions", "models": [{ "id": "two" }] },
            }),
        );

        let registry = create_registry(&models_path).await;
        let error = registry.get_error().expect("the composition errors report");

        assert!(error.contains("Provider \"broken-one\""), "got: {error}");
        assert!(error.contains("Provider \"broken-two\""), "got: {error}");
    }

    #[tokio::test]
    async fn custom_provider_with_same_name_as_built_in_merges_with_built_in_models() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_provider_config(
                    "https://my-proxy.example.com/v1",
                    &[("claude-custom", None)],
                    "anthropic-messages",
                ),
            }),
        );

        let registry = create_registry(&models_path).await;
        let anthropic_models = models_for_provider(&registry, "anthropic");

        assert!(anthropic_models.len() > 1);
        assert!(
            anthropic_models
                .iter()
                .any(|model| model.id == "claude-custom")
        );
        assert!(
            anthropic_models
                .iter()
                .any(|model| model.id.contains("claude"))
        );
    }

    #[tokio::test]
    async fn custom_model_with_same_id_replaces_built_in_model_by_id() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": json_provider_config(
                    "https://my-proxy.example.com/v1",
                    &[("anthropic/claude-sonnet-4", None)],
                    "openai-completions",
                ),
            }),
        );

        let registry = create_registry(&models_path).await;
        let sonnets: Vec<Model> = models_for_provider(&registry, "openrouter")
            .into_iter()
            .filter(|model| model.id == "anthropic/claude-sonnet-4")
            .collect();

        assert_eq!(sonnets.len(), 1);
        assert_eq!(sonnets[0].base_url, "https://my-proxy.example.com/v1");
    }

    #[tokio::test]
    async fn custom_provider_with_same_name_as_built_in_does_not_affect_other_built_in_providers() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_provider_config(
                    "https://my-proxy.example.com/v1",
                    &[("claude-custom", None)],
                    "anthropic-messages",
                ),
            }),
        );

        let registry = create_registry(&models_path).await;

        assert!(
            !models_for_provider(&registry, "google").is_empty(),
            "google models survive"
        );
        assert!(
            !models_for_provider(&registry, "openai").is_empty(),
            "openai models survive"
        );
    }

    #[tokio::test]
    async fn provider_level_base_url_applies_to_both_built_in_and_custom_models() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_provider_config(
                    "https://merged-proxy.example.com/v1",
                    &[("claude-custom", None)],
                    "anthropic-messages",
                ),
            }),
        );

        let registry = create_registry(&models_path).await;
        for model in models_for_provider(&registry, "anthropic") {
            assert_eq!(model.base_url, "https://merged-proxy.example.com/v1");
        }
    }

    #[tokio::test]
    async fn provider_level_compat_applies_to_custom_models() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "demo": {
                    "baseUrl": "https://example.com/v1",
                    "apiKey": "DEMO_KEY",
                    "api": "openai-completions",
                    "compat": {
                        "supportsUsageInStreaming": false,
                        "maxTokensField": "max_tokens",
                    },
                    "models": [{
                        "id": "demo-model",
                        "reasoning": false,
                        "input": ["text"],
                        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                        "contextWindow": 1000,
                        "maxTokens": 100,
                    }],
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let model = registry
            .find("demo", "demo-model")
            .expect("the demo model composes");
        let compat: &ModelCompat = model
            .compat
            .as_ref()
            .expect("the provider compat rides the model");

        assert_eq!(compat.supports_usage_in_streaming, Some(false));
        assert_eq!(compat.max_tokens_field, Some(MaxTokensField::MaxTokens));
    }

    #[tokio::test]
    async fn model_level_compat_overrides_provider_level_compat_for_custom_models() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "demo": {
                    "baseUrl": "https://example.com/v1",
                    "apiKey": "DEMO_KEY",
                    "api": "openai-completions",
                    "compat": {
                        "supportsUsageInStreaming": false,
                        "maxTokensField": "max_tokens",
                    },
                    "models": [{
                        "id": "demo-model",
                        "reasoning": false,
                        "input": ["text"],
                        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                        "contextWindow": 1000,
                        "maxTokens": 100,
                        "compat": {
                            "supportsUsageInStreaming": true,
                            "maxTokensField": "max_completion_tokens",
                        },
                    }],
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let model = registry
            .find("demo", "demo-model")
            .expect("the demo model composes");
        let compat: &ModelCompat = model
            .compat
            .as_ref()
            .expect("the merged compat rides the model");

        assert_eq!(compat.supports_usage_in_streaming, Some(true));
        assert_eq!(
            compat.max_tokens_field,
            Some(MaxTokensField::MaxCompletionTokens)
        );
    }

    #[tokio::test]
    async fn provider_level_compat_applies_to_built_in_models() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "compat": {
                        "supportsUsageInStreaming": false,
                        "supportsStrictMode": false,
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let models = models_for_provider(&registry, "openrouter");

        assert!(!models.is_empty(), "the openrouter catalog is non-empty");
        for model in &models {
            let compat: &ModelCompat = model.compat.as_ref().expect("the compat merges");
            assert_eq!(compat.supports_usage_in_streaming, Some(false));
            assert_eq!(compat.supports_strict_mode, Some(false));
        }
    }

    #[tokio::test]
    async fn model_schema_accepts_thinking_level_map_and_compat_schema_accepts_strict_mode_and_cache_control_format()
     {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "demo": {
                    "baseUrl": "https://example.com/v1",
                    "apiKey": "DEMO_KEY",
                    "api": "openai-completions",
                    "models": [{
                        "id": "demo-model",
                        "reasoning": true,
                        "input": ["text"],
                        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                        "contextWindow": 1000,
                        "maxTokens": 100,
                        "thinkingLevelMap": { "minimal": null, "high": "max" },
                        "compat": {
                            "supportsStrictMode": false,
                            "cacheControlFormat": "anthropic",
                        },
                    }],
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let model = registry
            .find("demo", "demo-model")
            .expect("the demo model composes");
        let compat: &ModelCompat = model.compat.as_ref().expect("the compat parses");

        assert!(registry.get_error().is_none());
        assert_eq!(
            model.thinking_level_map,
            Some(ThinkingLevelMap::from([
                (ModelThinkingLevel::Minimal, None),
                (ModelThinkingLevel::High, Some("max".to_owned())),
            ]))
        );
        assert_eq!(compat.supports_strict_mode, Some(false));
        assert_eq!(
            compat.cache_control_format,
            Some(CacheControlFormat::Anthropic)
        );
    }

    #[tokio::test]
    async fn compat_schema_accepts_chat_template_thinking_configuration() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "demo": {
                    "baseUrl": "https://example.com/v1",
                    "apiKey": "DEMO_KEY",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": "kwargs-model",
                            "reasoning": true,
                            "input": ["text"],
                            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                            "contextWindow": 1000,
                            "maxTokens": 100,
                            "compat": {
                                "thinkingFormat": "chat-template",
                                "chatTemplateKwargs": {
                                    "preserve_thinking": true,
                                    "thinking": { "$var": "thinking.enabled" },
                                },
                            },
                        },
                        {
                            "id": "args-model",
                            "reasoning": true,
                            "input": ["text"],
                            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                            "contextWindow": 1000,
                            "maxTokens": 100,
                            "compat": {
                                "thinkingFormat": "baseten",
                                "chatTemplateArgs": {
                                    "enable_thinking": { "$var": "thinking.enabled" },
                                },
                            },
                        },
                    ],
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let kwargs_model = registry
            .find("demo", "kwargs-model")
            .expect("the kwargs model composes");
        let args_model = registry
            .find("demo", "args-model")
            .expect("the args model composes");
        let kwargs_compat: &ModelCompat = kwargs_model.compat.as_ref().expect("the compat parses");
        let args_compat: &ModelCompat = args_model.compat.as_ref().expect("the compat parses");

        assert!(registry.get_error().is_none());
        assert_eq!(
            kwargs_compat.thinking_format,
            Some(ThinkingFormat::ChatTemplate)
        );
        assert_eq!(
            serde_json::to_value(
                kwargs_compat
                    .chat_template_kwargs
                    .as_ref()
                    .expect("the kwargs ride")
            )
            .expect("the kwargs serialize"),
            serde_json::json!({
                "preserve_thinking": true,
                "thinking": { "$var": "thinking.enabled" },
            })
        );
        assert_eq!(args_compat.thinking_format, Some(ThinkingFormat::Baseten));
        assert_eq!(
            serde_json::to_value(
                args_compat
                    .chat_template_args
                    .as_ref()
                    .expect("the args ride")
            )
            .expect("the args serialize"),
            serde_json::json!({
                "enable_thinking": { "$var": "thinking.enabled" },
            })
        );
    }

    #[tokio::test]
    async fn compat_schema_accepts_anthropic_eager_tool_input_streaming_flag() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "demo": {
                    "baseUrl": "https://example.com",
                    "apiKey": "DEMO_KEY",
                    "api": "anthropic-messages",
                    "compat": { "supportsEagerToolInputStreaming": false },
                    "models": [{
                        "id": "demo-model",
                        "reasoning": true,
                        "input": ["text"],
                        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                        "contextWindow": 1000,
                        "maxTokens": 100,
                    }],
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let model = registry
            .find("demo", "demo-model")
            .expect("the demo model composes");
        let compat: &ModelCompat = model.compat.as_ref().expect("the compat parses");

        assert!(registry.get_error().is_none());
        assert_eq!(compat.supports_eager_tool_input_streaming, Some(false));
    }

    #[tokio::test]
    async fn compat_schema_accepts_long_cache_retention_flag() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "demo": {
                    "baseUrl": "https://example.com",
                    "apiKey": "DEMO_KEY",
                    "api": "anthropic-messages",
                    "compat": { "supportsLongCacheRetention": false },
                    "models": [{
                        "id": "demo-model",
                        "reasoning": true,
                        "input": ["text"],
                        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                        "contextWindow": 1000,
                        "maxTokens": 100,
                    }],
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let model = registry
            .find("demo", "demo-model")
            .expect("the demo model composes");
        let compat: &ModelCompat = model.compat.as_ref().expect("the compat parses");

        assert!(registry.get_error().is_none());
        assert_eq!(compat.supports_long_cache_retention, Some(false));
    }

    #[tokio::test]
    async fn model_level_base_url_overrides_provider_level_base_url_for_custom_models() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "opencode-go": {
                    "baseUrl": "https://opencode.ai/zen/go/v1",
                    "apiKey": "TEST_KEY",
                    "models": [
                        {
                            "id": "minimax-m2.5",
                            "api": "anthropic-messages",
                            "baseUrl": "https://opencode.ai/zen/go",
                            "reasoning": true,
                            "input": ["text"],
                            "cost": { "input": 0.3, "output": 1.2, "cacheRead": 0.03, "cacheWrite": 0 },
                            "contextWindow": 204_800,
                            "maxTokens": 131_072,
                        },
                        {
                            "id": "glm-5",
                            "api": "openai-completions",
                            "reasoning": true,
                            "input": ["text"],
                            "cost": { "input": 1, "output": 3.2, "cacheRead": 0.2, "cacheWrite": 0 },
                            "contextWindow": 204_800,
                            "maxTokens": 131_072,
                        },
                    ],
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let m25 = registry
            .find("opencode-go", "minimax-m2.5")
            .expect("minimax composes");
        let glm5 = registry.find("opencode-go", "glm-5").expect("glm composes");

        assert_eq!(m25.base_url, "https://opencode.ai/zen/go");
        assert_eq!(glm5.base_url, "https://opencode.ai/zen/go/v1");
    }

    #[tokio::test]
    async fn model_overrides_still_apply_when_provider_also_defines_models() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "baseUrl": "https://my-proxy.example.com/v1",
                    "apiKey": "OPENROUTER_API_KEY",
                    "api": "openai-completions",
                    "models": [{
                        "id": "custom/openrouter-model",
                        "name": "Custom OpenRouter Model",
                        "reasoning": false,
                        "input": ["text"],
                        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
                        "contextWindow": 128_000,
                        "maxTokens": 16_384,
                    }],
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": {
                            "name": "Overridden Built-in Sonnet",
                        },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let models = models_for_provider(&registry, "openrouter");

        assert!(
            models
                .iter()
                .any(|model| model.id == "custom/openrouter-model")
        );
        assert!(
            models
                .iter()
                .any(|model| model.id == "anthropic/claude-sonnet-4"
                    && model.name == "Overridden Built-in Sonnet"),
            "the built-in sonnet carries the override name"
        );
    }

    #[tokio::test]
    async fn refresh_reloads_merged_custom_models_from_disk() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_provider_config(
                    "https://first-proxy.example.com/v1",
                    &[("claude-custom", None)],
                    "anthropic-messages",
                ),
            }),
        );
        let registry = create_registry(&models_path).await;
        assert!(
            models_for_provider(&registry, "anthropic")
                .iter()
                .any(|model| model.id == "claude-custom")
        );

        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_provider_config(
                    "https://second-proxy.example.com/v1",
                    &[("claude-custom-2", None)],
                    "anthropic-messages",
                ),
            }),
        );
        registry.refresh(Some(offline_refresh())).await;

        let anthropic_models = models_for_provider(&registry, "anthropic");
        assert!(
            !anthropic_models
                .iter()
                .any(|model| model.id == "claude-custom"),
            "the stale custom model drops"
        );
        assert!(
            anthropic_models
                .iter()
                .any(|model| model.id == "claude-custom-2"),
            "the fresh custom model composes"
        );
        assert!(
            anthropic_models
                .iter()
                .any(|model| model.id.contains("claude")),
            "the built-in models survive"
        );
    }

    #[tokio::test]
    async fn removing_custom_models_from_models_json_keeps_built_in_provider_models() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "anthropic": json_provider_config(
                    "https://proxy.example.com/v1",
                    &[("claude-custom", None)],
                    "anthropic-messages",
                ),
            }),
        );
        let registry = create_registry(&models_path).await;
        assert!(
            models_for_provider(&registry, "anthropic")
                .iter()
                .any(|model| model.id == "claude-custom")
        );

        write_raw_models_json(&models_path, serde_json::json!({}));
        registry.refresh(Some(offline_refresh())).await;

        let anthropic_models = models_for_provider(&registry, "anthropic");
        assert!(
            !anthropic_models
                .iter()
                .any(|model| model.id == "claude-custom"),
            "the removed custom model drops"
        );
        assert!(
            anthropic_models
                .iter()
                .any(|model| model.id.contains("claude")),
            "the built-in models survive"
        );
    }
}

mod model_overrides_per_model_customization {
    use super::*;

    #[tokio::test]
    async fn model_override_applies_to_a_single_built_in_model() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": { "name": "Custom Sonnet Name" },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let models = models_for_provider(&registry, "openrouter");

        let sonnet = models
            .iter()
            .find(|model| model.id == "anthropic/claude-sonnet-4")
            .expect("the sonnet composes");
        assert_eq!(sonnet.name, "Custom Sonnet Name");

        let opus = models
            .iter()
            .find(|model| model.id == "anthropic/claude-opus-4")
            .expect("the opus composes");
        assert_ne!(opus.name, "Custom Sonnet Name");
    }

    #[tokio::test]
    async fn custom_model_and_model_override_carry_sampling_params() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "baseUrl": "https://my-proxy.example.com/v1",
                    "api": "openai-completions",
                    "models": [{
                        "id": "custom/sampling-model",
                        "samplingParams": { "temperature": 1, "top_p": 0.95, "top_k": 0 },
                    }],
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": {
                            "samplingParams": { "top_p": 0.9 },
                        },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let models = models_for_provider(&registry, "openrouter");

        let custom = models
            .iter()
            .find(|model| model.id == "custom/sampling-model")
            .expect("the custom model composes");
        assert_eq!(
            serde_json::to_value(
                custom
                    .sampling_params
                    .as_ref()
                    .expect("sampling params ride")
            )
            .expect("the sampling params serialize"),
            serde_json::json!({ "temperature": 1, "top_p": 0.95, "top_k": 0 })
        );

        let sonnet = models
            .iter()
            .find(|model| model.id == "anthropic/claude-sonnet-4")
            .expect("the sonnet composes");
        assert_eq!(
            serde_json::to_value(
                sonnet
                    .sampling_params
                    .as_ref()
                    .expect("sampling params ride")
            )
            .expect("the sampling params serialize"),
            serde_json::json!({ "top_p": 0.9 })
        );

        let opus = models
            .iter()
            .find(|model| model.id == "anthropic/claude-opus-4")
            .expect("the opus composes");
        assert!(opus.sampling_params.is_none(), "unconfigured stays unset");
    }

    #[tokio::test]
    async fn model_override_with_compat_open_router_routing() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": {
                            "compat": { "openRouterRouting": { "only": ["amazon-bedrock"] } },
                        },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let models = models_for_provider(&registry, "openrouter");

        let sonnet = models
            .iter()
            .find(|model| model.id == "anthropic/claude-sonnet-4")
            .expect("the sonnet composes");
        let compat: &ModelCompat = sonnet.compat.as_ref().expect("the compat merges");
        assert_eq!(
            compat.open_router_routing,
            Some(OpenRouterRouting {
                only: Some(vec!["amazon-bedrock".to_owned()]),
                ..OpenRouterRouting::default()
            })
        );
    }

    #[tokio::test]
    async fn supports_finish_reason_can_be_configured_at_provider_and_model_levels() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "compat": { "supportsFinishReason": true },
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": {
                            "compat": { "supportsFinishReason": false },
                        },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let models = models_for_provider(&registry, "openrouter");
        let sonnet = models
            .iter()
            .find(|model| model.id == "anthropic/claude-sonnet-4")
            .expect("the sonnet composes");
        let opus = models
            .iter()
            .find(|model| model.id == "anthropic/claude-opus-4")
            .expect("the opus composes");

        assert_eq!(
            sonnet
                .compat
                .as_ref()
                .expect("the compat merges")
                .supports_finish_reason,
            Some(false)
        );
        assert_eq!(
            opus.compat
                .as_ref()
                .expect("the compat merges")
                .supports_finish_reason,
            Some(true)
        );
    }

    #[tokio::test]
    async fn model_override_deep_merges_compat_settings() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": {
                            "compat": {
                                "openRouterRouting": { "order": ["anthropic", "together"] },
                            },
                        },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let models = models_for_provider(&registry, "openrouter");
        let sonnet = models
            .iter()
            .find(|model| model.id == "anthropic/claude-sonnet-4")
            .expect("the sonnet composes");

        let compat: &ModelCompat = sonnet.compat.as_ref().expect("the compat merges");
        assert_eq!(
            compat.open_router_routing,
            Some(OpenRouterRouting {
                order: Some(vec!["anthropic".to_owned(), "together".to_owned()]),
                ..OpenRouterRouting::default()
            })
        );
    }

    #[tokio::test]
    async fn multiple_model_overrides_on_same_provider() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": {
                            "compat": { "openRouterRouting": { "only": ["amazon-bedrock"] } },
                        },
                        "anthropic/claude-opus-4": {
                            "compat": { "openRouterRouting": { "only": ["anthropic"] } },
                        },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let models = models_for_provider(&registry, "openrouter");

        let sonnet = models
            .iter()
            .find(|model| model.id == "anthropic/claude-sonnet-4")
            .expect("the sonnet composes");
        let opus = models
            .iter()
            .find(|model| model.id == "anthropic/claude-opus-4")
            .expect("the opus composes");

        assert_eq!(
            sonnet
                .compat
                .as_ref()
                .expect("the compat merges")
                .open_router_routing,
            Some(OpenRouterRouting {
                only: Some(vec!["amazon-bedrock".to_owned()]),
                ..OpenRouterRouting::default()
            })
        );
        assert_eq!(
            opus.compat
                .as_ref()
                .expect("the compat merges")
                .open_router_routing,
            Some(OpenRouterRouting {
                only: Some(vec!["anthropic".to_owned()]),
                ..OpenRouterRouting::default()
            })
        );
    }

    #[tokio::test]
    async fn model_override_combined_with_base_url_override() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "baseUrl": "https://my-proxy.example.com/v1",
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": { "name": "Proxied Sonnet" },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let models = models_for_provider(&registry, "openrouter");
        let sonnet = models
            .iter()
            .find(|model| model.id == "anthropic/claude-sonnet-4")
            .expect("the sonnet composes");

        assert_eq!(sonnet.base_url, "https://my-proxy.example.com/v1");
        assert_eq!(sonnet.name, "Proxied Sonnet");

        let opus = models
            .iter()
            .find(|model| model.id == "anthropic/claude-opus-4")
            .expect("the opus composes");
        assert_eq!(opus.base_url, "https://my-proxy.example.com/v1");
        assert_ne!(opus.name, "Proxied Sonnet");
    }

    #[tokio::test]
    async fn model_override_for_non_existent_model_id_is_ignored() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "modelOverrides": {
                        "nonexistent/model-id": { "name": "This should not appear" },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let models = models_for_provider(&registry, "openrouter");

        assert!(
            !models
                .iter()
                .any(|model| model.id == "nonexistent/model-id"),
            "the override creates no model"
        );
        assert!(registry.get_error().is_none());
    }

    #[tokio::test]
    async fn model_override_can_change_cost_fields_partially() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": {
                            "cost": { "input": 99 },
                        },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let models = models_for_provider(&registry, "openrouter");
        let sonnet = models
            .iter()
            .find(|model| model.id == "anthropic/claude-sonnet-4")
            .expect("the sonnet composes");

        // Input cost should be overridden
        assert_eq!(sonnet.cost.rates.input, 99.0);
        // Other cost fields should be preserved from built-in
        assert!(sonnet.cost.rates.output > 0.0);
    }

    #[tokio::test]
    async fn model_override_can_add_headers_at_request_time() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": {
                            "headers": { "X-Custom-Model-Header": "value" },
                        },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        let models = models_for_provider(&registry, "openrouter");
        let sonnet = models
            .iter()
            .find(|model| model.id == "anthropic/claude-sonnet-4")
            .expect("the sonnet composes");

        let auth = registry.get_api_key_and_headers(sonnet).await;
        assert!(auth.ok());
        let ResolvedRequestAuth::Ok { headers, .. } = auth else {
            unreachable!("auth.ok() pinned the Ok arm");
        };
        assert_eq!(
            headers
                .as_ref()
                .and_then(|headers| headers.get("X-Custom-Model-Header")),
            Some(&Some("value".to_owned())),
        );
    }

    #[tokio::test]
    async fn refresh_picks_up_model_override_changes() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": { "name": "First Name" },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        assert_eq!(
            models_for_provider(&registry, "openrouter")
                .iter()
                .find(|model| model.id == "anthropic/claude-sonnet-4")
                .expect("the sonnet composes")
                .name,
            "First Name"
        );

        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": { "name": "Second Name" },
                    },
                },
            }),
        );
        registry.refresh(Some(offline_refresh())).await;

        assert_eq!(
            models_for_provider(&registry, "openrouter")
                .iter()
                .find(|model| model.id == "anthropic/claude-sonnet-4")
                .expect("the sonnet composes")
                .name,
            "Second Name"
        );
    }

    #[tokio::test]
    async fn removing_model_override_restores_built_in_values() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "openrouter": {
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": { "name": "Custom Name" },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        assert_eq!(
            models_for_provider(&registry, "openrouter")
                .iter()
                .find(|model| model.id == "anthropic/claude-sonnet-4")
                .expect("the sonnet composes")
                .name,
            "Custom Name"
        );

        write_raw_models_json(&models_path, serde_json::json!({}));
        registry.refresh(Some(offline_refresh())).await;

        let restored = models_for_provider(&registry, "openrouter")
            .into_iter()
            .find(|model| model.id == "anthropic/claude-sonnet-4")
            .expect("the sonnet composes");
        assert_ne!(restored.name, "Custom Name");
    }
}

mod dynamic_provider_lifecycle {
    use super::*;

    #[tokio::test]
    async fn get_provider_display_name_resolves_registered_oauth_built_in_and_fallback_names() {
        let (_dir, models_path) = rig();
        let registry = create_registry(&models_path).await;

        assert_eq!(registry.get_provider_display_name("openai"), "OpenAI");
        assert_eq!(
            registry.get_provider_display_name("github-copilot"),
            "GitHub Copilot"
        );
        assert_eq!(registry.get_provider_display_name("zai"), "Z.AI");
        assert_eq!(
            registry.get_provider_display_name("unknown-provider"),
            "unknown-provider"
        );

        registry
            .register_provider_config(
                "named-provider",
                ProviderConfigInput {
                    name: Some("Named Provider".to_owned()),
                    ..registration(
                        "https://provider.test/v1",
                        Api::from("openai-completions"),
                        vec![extension_model(
                            "demo-model",
                            Api::from("openai-completions"),
                        )],
                    )
                },
            )
            .expect("the named provider registers");
        assert_eq!(
            registry.get_provider_display_name("named-provider"),
            "Named Provider"
        );

        registry
            .register_provider_config(
                "oauth-provider",
                ProviderConfigInput {
                    base_url: Some("https://provider.test/v1".to_owned()),
                    api: Some(Api::from("openai-completions")),
                    oauth: Some(ExtensionOAuthConfig {
                        name: "OAuth Provider".to_owned(),
                        is_subscription: None,
                        uses_callback_server: None,
                        login: never_login(),
                        refresh_token: echo_refresh(),
                        get_api_key: Arc::new(|credentials: &OAuthCredentials| {
                            credentials.access.clone()
                        }),
                        modify_models: None,
                    }),
                    models: Some(vec![extension_model(
                        "demo-model",
                        Api::from("openai-completions"),
                    )]),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the oauth provider registers");
        assert_eq!(
            registry.get_provider_display_name("oauth-provider"),
            "OAuth Provider"
        );
    }

    #[tokio::test]
    async fn model_overrides_apply_to_dynamically_registered_provider_models() {
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "extension-provider": {
                    "modelOverrides": {
                        "extension-model": {
                            "name": "Overridden Extension Model",
                            "thinkingLevelMap": {
                                "off": null,
                                "minimal": null,
                                "low": null,
                                "medium": null,
                                "xhigh": "max",
                            },
                            "headers": { "x-model-override": "enabled" },
                        },
                    },
                },
            }),
        );

        let registry = create_registry(&models_path).await;
        registry
            .register_provider_config(
                "extension-provider",
                registration(
                    "https://provider.test/v1",
                    Api::from("openai-completions"),
                    vec![ProviderModelInput {
                        reasoning: true,
                        ..extension_model("extension-model", Api::from("openai-completions"))
                    }],
                ),
            )
            .expect("the extension provider registers");

        let model = registry
            .find("extension-provider", "extension-model")
            .expect("the extension model registers");
        assert_eq!(model.name, "Overridden Extension Model");
        assert_eq!(
            model.thinking_level_map,
            Some(ThinkingLevelMap::from([
                (ModelThinkingLevel::Off, None),
                (ModelThinkingLevel::Minimal, None),
                (ModelThinkingLevel::Low, None),
                (ModelThinkingLevel::Medium, None),
                (ModelThinkingLevel::Xhigh, Some("max".to_owned())),
            ]))
        );
        assert_eq!(
            pi_ai::models::get_supported_thinking_levels(&model),
            vec![ModelThinkingLevel::High, ModelThinkingLevel::Xhigh]
        );
        let auth = registry.get_api_key_and_headers(&model).await;
        assert!(auth.ok());
        let ResolvedRequestAuth::Ok { headers, .. } = auth else {
            unreachable!("auth.ok() pinned the Ok arm");
        };
        assert_eq!(
            headers
                .as_ref()
                .and_then(|headers| headers.get("x-model-override")),
            Some(&Some("enabled".to_owned())),
        );
    }

    #[tokio::test]
    async fn stored_api_key_env_propagates_to_request_auth_and_resolves_headers() {
        let (_dir, models_path) = rig();
        let storage = common::model_layer::empty_auth_storage();
        store_api_key(
            &storage,
            "cloudflare-ai-gateway",
            "$CLOUDFLARE_API_KEY",
            Some(env_from(&[
                ("CLOUDFLARE_API_KEY", "stored-cf-token"),
                ("CLOUDFLARE_ACCOUNT_ID", "stored-account"),
                ("CLOUDFLARE_GATEWAY_ID", "stored-gateway"),
            ])),
        )
        .await;
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "cloudflare-ai-gateway": {
                    "headers": { "x-account": "$CLOUDFLARE_ACCOUNT_ID" },
                },
            }),
        );

        let registry =
            common::model_layer::create_model_registry(storage.clone(), Some(&models_path)).await;
        let model = models_for_provider(&registry, "cloudflare-ai-gateway")
            .into_iter()
            .next()
            .expect("the gateway models compose");

        let auth = registry.get_api_key_and_headers(&model).await;

        assert_eq!(
            auth,
            ResolvedRequestAuth::Ok {
                api_key: None,
                headers: Some(BTreeMap::from([
                    (
                        "cf-aig-authorization".to_owned(),
                        Some("Bearer stored-cf-token".to_owned())
                    ),
                    ("Authorization".to_owned(), None),
                    ("x-api-key".to_owned(), None),
                    ("x-account".to_owned(), Some("stored-account".to_owned())),
                ])),
                base_url: None,
                env: Some(env_from(&[
                    ("CLOUDFLARE_ACCOUNT_ID", "stored-account"),
                    ("CLOUDFLARE_GATEWAY_ID", "stored-gateway"),
                ])),
            }
        );
    }

    #[tokio::test]
    async fn register_provider_treats_uppercase_api_key_and_headers_as_literals() {
        let (_dir, models_path) = rig();
        let registry = create_registry(&models_path).await;

        let mut config = registration(
            "https://provider.test/v1",
            Api::from("openai-completions"),
            vec![ProviderModelInput {
                headers: Some(BTreeMap::from([(
                    "x-model-token".to_owned(),
                    "MODEL_TOKEN".to_owned(),
                )])),
                ..extension_model("demo-model", Api::from("openai-completions"))
            }],
        );
        config.api_key = Some("CUSTOM_NAME".to_owned());
        config.headers = Some(BTreeMap::from([(
            "Authorization".to_owned(),
            "BEARER".to_owned(),
        )]));
        registry
            .register_provider_config("literal-provider", config)
            .expect("the literal provider registers");

        assert_eq!(
            registry.get_api_key_for_provider("literal-provider").await,
            Some("CUSTOM_NAME".to_owned())
        );
        let model = registry
            .find("literal-provider", "demo-model")
            .expect("the demo model registers");
        let auth = registry.get_api_key_and_headers(&model).await;
        assert_eq!(
            auth,
            resolved_ok(
                Some("CUSTOM_NAME"),
                Some(ok_headers(&[
                    ("Authorization", "BEARER"),
                    ("x-model-token", "MODEL_TOKEN")
                ]))
            )
        );
    }

    #[tokio::test]
    async fn failed_register_provider_does_not_persist_invalid_stream_simple_config() {
        let (_dir, models_path) = rig();
        let registry = create_registry(&models_path).await;

        let invocations = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&invocations);
        let stream: ExtensionStreamSimpleFn = Arc::new(move |_model, _context, _options| {
            counter.fetch_add(1, Ordering::Relaxed);
            AssistantMessageEventStream::new(|_| false, |_| None)
        });
        let error = registry
            .register_provider_config(
                "broken-provider",
                ProviderConfigInput {
                    stream_simple: Some(stream),
                    ..ProviderConfigInput::default()
                },
            )
            .expect_err("the streamSimple registration fails");
        assert!(
            error.contains("\"api\" is required when registering streamSimple."),
            "got: {error}"
        );
        assert_eq!(
            invocations.load(Ordering::Relaxed),
            0,
            "registration never invokes streamSimple"
        );

        let result = registry.refresh(Some(offline_refresh())).await;
        assert!(!result.aborted);
    }

    #[tokio::test]
    async fn failed_register_provider_does_not_remove_existing_provider_models() {
        let (_dir, models_path) = rig();
        let registry = create_registry(&models_path).await;

        registry
            .register_provider_config(
                "demo-provider",
                registration(
                    "https://provider.test/v1",
                    Api::from("openai-completions"),
                    vec![extension_model(
                        "demo-model",
                        Api::from("openai-completions"),
                    )],
                ),
            )
            .expect("the demo provider registers");
        assert!(registry.find("demo-provider", "demo-model").is_some());

        let error = registry
            .register_provider_config(
                "demo-provider",
                ProviderConfigInput {
                    base_url: Some("https://provider.test/v2".to_owned()),
                    api_key: Some("test-key".to_owned()),
                    models: Some(vec![ProviderModelInput {
                        api: None,
                        ..extension_model("broken-model", Api::from("openai-completions"))
                    }]),
                    ..ProviderConfigInput::default()
                },
            )
            .expect_err("the broken re-registration fails");
        assert!(
            error.contains("Provider demo-provider, model broken-model: no \"api\" specified."),
            "got: {error}"
        );

        assert!(registry.find("demo-provider", "demo-model").is_some());
        let result = registry.refresh(Some(offline_refresh())).await;
        assert!(!result.aborted);
        assert!(registry.find("demo-provider", "demo-model").is_some());
    }

    #[tokio::test]
    async fn unregister_provider_removes_the_runtime_oauth_overlay_without_mutating_global_state() {
        let (_dir, models_path) = rig();
        let registry = create_registry(&models_path).await;

        registry
            .register_provider_config(
                "anthropic",
                ProviderConfigInput {
                    oauth: Some(ExtensionOAuthConfig {
                        name: "Custom Anthropic OAuth".to_owned(),
                        is_subscription: None,
                        uses_callback_server: None,
                        login: never_login(),
                        refresh_token: echo_refresh(),
                        get_api_key: Arc::new(|credentials: &OAuthCredentials| {
                            credentials.access.clone()
                        }),
                        modify_models: None,
                    }),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the oauth overlay registers");

        let registered = registry
            .get_registered_provider_config("anthropic")
            .expect("the registration reports");
        assert_eq!(
            registered.oauth.as_ref().expect("the oauth carries").name,
            "Custom Anthropic OAuth"
        );

        registry.unregister_provider("anthropic");

        assert!(
            registry
                .get_registered_provider_config("anthropic")
                .is_none()
        );
    }

    #[tokio::test]
    async fn stream_simple_overlays_do_not_mutate_the_global_compat_api_registry() {
        let (_dir, models_path) = rig();
        let registry = create_registry(&models_path).await;
        let api = Api::from("openai-completions");
        let before = pi_ai::compat::get_api_provider(&api).expect("the api registry carries it");

        registry
            .register_provider_config(
                "stream-override-provider",
                ProviderConfigInput {
                    api: Some(Api::from("openai-completions")),
                    stream_simple: Some(never_stream()),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the stream override registers");
        let after_register =
            pi_ai::compat::get_api_provider(&api).expect("the api registry carries it");

        registry.unregister_provider("stream-override-provider");
        let after_unregister =
            pi_ai::compat::get_api_provider(&api).expect("the api registry carries it");

        // Restated: upstream observes the global api's behavior by invoking
        // it; polling the built-in stream here would open a network request,
        // so the case pins the global entry's identity across the runtime
        // registration lifecycle instead.
        assert!(
            Arc::ptr_eq(&before, &after_register),
            "registration leaves the global entry alone"
        );
        assert!(
            Arc::ptr_eq(&before, &after_unregister),
            "unregistration leaves the global entry alone"
        );
    }

    mod dynamic_provider_override_persistence {
        use super::*;

        #[tokio::test]
        async fn base_url_only_override_keeps_built_in_provider_models_after_refresh() {
            let (_dir, models_path) = rig();
            let registry = create_registry(&models_path).await;

            registry
                .register_provider_config(
                    "anthropic",
                    ProviderConfigInput {
                        base_url: Some("https://proxy.test/anthropic".to_owned()),
                        ..ProviderConfigInput::default()
                    },
                )
                .expect("the override registers");
            registry.refresh(Some(offline_refresh())).await;

            let anthropic_models = models_for_provider(&registry, "anthropic");
            assert!(anthropic_models.len() > 1);
            assert!(
                anthropic_models
                    .iter()
                    .all(|model| model.base_url == "https://proxy.test/anthropic")
            );
        }

        #[tokio::test]
        async fn models_only_override_replaces_built_in_provider_models_after_refresh() {
            let (_dir, models_path) = rig();
            let registry = create_registry(&models_path).await;

            registry
                .register_provider_config(
                    "anthropic",
                    ProviderConfigInput {
                        base_url: Some("https://custom.test/anthropic".to_owned()),
                        ..registration(
                            "https://custom.test/anthropic",
                            Api::from("anthropic-messages"),
                            vec![extension_model(
                                "custom-claude",
                                Api::from("anthropic-messages"),
                            )],
                        )
                    },
                )
                .expect("the override registers");
            registry.refresh(Some(offline_refresh())).await;

            let ids: Vec<String> = models_for_provider(&registry, "anthropic")
                .into_iter()
                .map(|model| model.id)
                .collect();
            assert_eq!(ids, vec!["custom-claude"]);
            assert_eq!(
                registry
                    .find("anthropic", "custom-claude")
                    .expect("the custom model composes")
                    .base_url,
                "https://custom.test/anthropic"
            );
        }

        #[tokio::test]
        async fn models_plus_base_url_override_replaces_built_in_provider_models_after_refresh() {
            let (_dir, models_path) = rig();
            let registry = create_registry(&models_path).await;

            registry
                .register_provider_config(
                    "anthropic",
                    ProviderConfigInput {
                        base_url: Some("https://custom.test/anthropic".to_owned()),
                        ..registration(
                            "https://custom.test/anthropic",
                            Api::from("anthropic-messages"),
                            vec![extension_model(
                                "custom-claude",
                                Api::from("anthropic-messages"),
                            )],
                        )
                    },
                )
                .expect("the first override registers");
            registry
                .register_provider_config(
                    "anthropic",
                    ProviderConfigInput {
                        base_url: Some("https://proxy.test/anthropic".to_owned()),
                        ..ProviderConfigInput::default()
                    },
                )
                .expect("the second override registers");
            registry.refresh(Some(offline_refresh())).await;

            let ids: Vec<String> = models_for_provider(&registry, "anthropic")
                .into_iter()
                .map(|model| model.id)
                .collect();
            assert_eq!(ids, vec!["custom-claude"]);
            assert_eq!(
                registry
                    .find("anthropic", "custom-claude")
                    .expect("the custom model composes")
                    .base_url,
                "https://proxy.test/anthropic"
            );
        }

        #[tokio::test]
        async fn models_only_custom_provider_registration_survives_refresh() {
            let (_dir, models_path) = rig();
            let registry = create_registry(&models_path).await;

            registry
                .register_provider_config(
                    "custom-provider",
                    registration(
                        "https://custom.test/v1",
                        Api::from("openai-completions"),
                        vec![
                            extension_model("custom-a", Api::from("openai-completions")),
                            extension_model("custom-b", Api::from("openai-completions")),
                        ],
                    ),
                )
                .expect("the custom provider registers");
            registry.refresh(Some(offline_refresh())).await;

            let ids: Vec<String> = models_for_provider(&registry, "custom-provider")
                .into_iter()
                .map(|model| model.id)
                .collect();
            assert_eq!(ids, vec!["custom-a", "custom-b"]);
        }

        #[tokio::test]
        async fn base_url_only_override_keeps_custom_provider_models_after_refresh() {
            let (_dir, models_path) = rig();
            let registry = create_registry(&models_path).await;

            registry
                .register_provider_config(
                    "custom-provider",
                    registration(
                        "https://custom.test/v1",
                        Api::from("openai-completions"),
                        vec![
                            extension_model("custom-a", Api::from("openai-completions")),
                            extension_model("custom-b", Api::from("openai-completions")),
                        ],
                    ),
                )
                .expect("the custom provider registers");
            registry
                .register_provider_config(
                    "custom-provider",
                    ProviderConfigInput {
                        base_url: Some("https://proxy.test/custom".to_owned()),
                        ..ProviderConfigInput::default()
                    },
                )
                .expect("the override registers");
            registry.refresh(Some(offline_refresh())).await;

            let models = models_for_provider(&registry, "custom-provider");
            let ids: Vec<String> = models.iter().map(|model| model.id.clone()).collect();
            assert_eq!(ids, vec!["custom-a", "custom-b"]);
            assert!(
                models
                    .iter()
                    .all(|model| model.base_url == "https://proxy.test/custom")
            );
        }

        #[tokio::test]
        async fn headers_only_override_keeps_custom_provider_models_after_refresh() {
            let (_dir, models_path) = rig();
            let registry = create_registry(&models_path).await;

            registry
                .register_provider_config(
                    "custom-provider",
                    registration(
                        "https://custom.test/v1",
                        Api::from("openai-completions"),
                        vec![
                            extension_model("custom-a", Api::from("openai-completions")),
                            extension_model("custom-b", Api::from("openai-completions")),
                        ],
                    ),
                )
                .expect("the custom provider registers");
            registry
                .register_provider_config(
                    "custom-provider",
                    ProviderConfigInput {
                        headers: Some(BTreeMap::from([(
                            "x-proxy".to_owned(),
                            "enabled".to_owned(),
                        )])),
                        ..ProviderConfigInput::default()
                    },
                )
                .expect("the override registers");
            registry.refresh(Some(offline_refresh())).await;

            let models = models_for_provider(&registry, "custom-provider");
            let ids: Vec<String> = models.iter().map(|model| model.id.clone()).collect();
            assert_eq!(ids, vec!["custom-a", "custom-b"]);
            assert!(
                models
                    .iter()
                    .all(|model| model.base_url == "https://custom.test/v1")
            );
            let auth = registry.get_api_key_and_headers(&models[0]).await;
            assert!(auth.ok());
            let ResolvedRequestAuth::Ok { headers, .. } = auth else {
                unreachable!("auth.ok() pinned the Ok arm");
            };
            assert_eq!(
                headers.as_ref().and_then(|headers| headers.get("x-proxy")),
                Some(&Some("enabled".to_owned())),
            );
        }
    }
}

mod api_key_resolution {
    use super::*;

    #[tokio::test]
    async fn api_key_with_bang_prefix_executes_command_and_uses_stdout() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("!echo test-api-key-from-command"),
            }),
        );

        let registry = create_registry(&models_path).await;
        let api_key = registry.get_api_key_for_provider("custom-provider").await;

        assert_eq!(api_key, Some("test-api-key-from-command".to_owned()));
    }

    #[tokio::test]
    async fn api_key_with_bang_prefix_trims_whitespace_from_command_output() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("!echo '  spaced-key  '"),
            }),
        );

        let registry = create_registry(&models_path).await;
        let api_key = registry.get_api_key_for_provider("custom-provider").await;

        assert_eq!(api_key, Some("spaced-key".to_owned()));
    }

    #[tokio::test]
    async fn api_key_with_bang_prefix_handles_multiline_output() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("!printf 'line1\\nline2'"),
            }),
        );

        let registry = create_registry(&models_path).await;
        let api_key = registry.get_api_key_for_provider("custom-provider").await;

        assert_eq!(api_key, Some("line1\nline2".to_owned()));
    }

    #[tokio::test]
    async fn api_key_with_bang_prefix_returns_none_on_command_failure() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("!exit 1"),
            }),
        );

        let registry = create_registry(&models_path).await;
        let api_key = registry.get_api_key_for_provider("custom-provider").await;

        assert_eq!(api_key, None);
    }

    #[tokio::test]
    async fn api_key_with_bang_prefix_returns_none_on_nonexistent_command() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("!nonexistent-command-12345"),
            }),
        );

        let registry = create_registry(&models_path).await;
        let api_key = registry.get_api_key_for_provider("custom-provider").await;

        assert_eq!(api_key, None);
    }

    #[tokio::test]
    async fn api_key_with_bang_prefix_returns_none_on_empty_output() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("!printf ''"),
            }),
        );

        let registry = create_registry(&models_path).await;
        let api_key = registry.get_api_key_for_provider("custom-provider").await;

        assert_eq!(api_key, None);
    }

    #[tokio::test]
    async fn api_key_with_dollar_prefix_resolves_to_env_value() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("$TEST_API_KEY_12345"),
            }),
        );

        let registry = create_registry(&models_path).await;
        // Restated: upstream sets process.env; the resolution's env overlay
        // is the seam the port carries instead.
        let resolution = resolve_auth_with_env(
            &registry,
            "custom-provider",
            &[("TEST_API_KEY_12345", "env-api-key-value")],
        )
        .await
        .expect("the provider auth resolves");

        assert_eq!(
            resolution.auth.api_key.as_deref(),
            Some("env-api-key-value")
        );
    }

    #[tokio::test]
    async fn api_key_with_braced_env_syntax_resolves_to_env_value() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("${TEST_BRACED_API_KEY_12345}"),
            }),
        );

        let registry = create_registry(&models_path).await;
        let resolution = resolve_auth_with_env(
            &registry,
            "custom-provider",
            &[("TEST_BRACED_API_KEY_12345", "braced-env-api-key-value")],
        )
        .await
        .expect("the provider auth resolves");

        assert_eq!(
            resolution.auth.api_key.as_deref(),
            Some("braced-env-api-key-value")
        );
    }

    #[tokio::test]
    async fn api_key_interpolates_braced_env_references_inside_literals() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key(
                    "${TEST_INTERPOLATED_PART_A_12345}_${TEST_INTERPOLATED_PART_B_12345}",
                ),
            }),
        );

        let registry = create_registry(&models_path).await;
        let resolution = resolve_auth_with_env(
            &registry,
            "custom-provider",
            &[
                ("TEST_INTERPOLATED_PART_A_12345", "left"),
                ("TEST_INTERPOLATED_PART_B_12345", "right"),
            ],
        )
        .await
        .expect("the provider auth resolves");

        assert_eq!(resolution.auth.api_key.as_deref(), Some("left_right"));
    }

    #[tokio::test]
    async fn api_key_with_double_dollar_prefix_escapes_a_leading_dollar() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("$$TEST_API_KEY_12345"),
            }),
        );

        let registry = create_registry(&models_path).await;
        let api_key = registry.get_api_key_for_provider("custom-provider").await;

        assert_eq!(api_key, Some("$TEST_API_KEY_12345".to_owned()));
    }

    #[tokio::test]
    async fn api_key_with_dollar_bang_escapes_a_literal_bang_and_still_interpolates_later_env_refs()
    {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("$!literal-$TEST_API_KEY_12345"),
            }),
        );

        let registry = create_registry(&models_path).await;
        let resolution = resolve_auth_with_env(
            &registry,
            "custom-provider",
            &[("TEST_API_KEY_12345", "env-api-key-value")],
        )
        .await
        .expect("the provider auth resolves");

        assert_eq!(
            resolution.auth.api_key.as_deref(),
            Some("!literal-env-api-key-value")
        );
    }

    #[tokio::test]
    async fn plain_api_key_is_used_directly_even_when_it_matches_an_env_var() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("TEST_API_KEY_12345"),
            }),
        );

        let registry = create_registry(&models_path).await;
        let api_key = registry.get_api_key_for_provider("custom-provider").await;

        // Restated: upstream sets the matching env var to prove the literal
        // wins; the workspace forbids mutating the process environment, and
        // the literal branch never reads it, so the case pins the literal
        // outcome alone.
        assert_eq!(api_key, Some("TEST_API_KEY_12345".to_owned()));
    }

    #[tokio::test]
    async fn api_key_as_literal_value_is_used_directly_when_not_an_env_var() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("literal_api_key_value"),
            }),
        );

        let registry = create_registry(&models_path).await;
        let api_key = registry.get_api_key_for_provider("custom-provider").await;

        assert_eq!(api_key, Some("literal_api_key_value".to_owned()));
    }

    #[tokio::test]
    async fn api_key_command_can_use_shell_features_like_pipes() {
        clear_config_value_cache();
        let (_dir, models_path) = rig();
        write_raw_models_json(
            &models_path,
            serde_json::json!({
                "custom-provider": json_provider_with_api_key("!echo 'hello world' | tr ' ' '-'"),
            }),
        );

        let registry = create_registry(&models_path).await;
        let api_key = registry.get_api_key_for_provider("custom-provider").await;

        assert_eq!(api_key, Some("hello-world".to_owned()));
    }

    mod request_time_resolution {
        use super::*;

        /// The counter-command string upstream's lookup cases spell: read
        /// the counter, increment it on disk, and emit the key value.
        fn counting_command(counter_file: &std::path::Path) -> String {
            let counter_path = to_sh_path(counter_file.display().to_string().as_str());
            format!(
                "!sh -c 'count=$(cat \"{counter_path}\"); echo $((count + 1)) > \"{counter_path}\"; echo \"key-value\"'"
            )
        }

        /// The failing counter command: increments the counter, then exits
        /// nonzero.
        fn failing_counting_command(counter_file: &std::path::Path) -> String {
            let counter_path = to_sh_path(counter_file.display().to_string().as_str());
            format!(
                "!sh -c 'count=$(cat \"{counter_path}\"); echo $((count + 1)) > \"{counter_path}\"; exit 1'"
            )
        }

        #[tokio::test]
        async fn command_is_executed_on_every_provider_lookup() {
            clear_config_value_cache();
            let (dir, models_path) = rig();
            let counter_file = dir.path().join("counter");
            std::fs::write(&counter_file, "0").expect("the counter seeds");

            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "custom-provider": json_provider_with_api_key(counting_command(&counter_file).as_str()),
                }),
            );

            let registry = create_registry(&models_path).await;
            registry.get_api_key_for_provider("custom-provider").await;
            registry.get_api_key_for_provider("custom-provider").await;
            registry.get_api_key_for_provider("custom-provider").await;

            assert_eq!(counter_count(&counter_file), 3);
        }

        #[tokio::test]
        async fn commands_are_re_executed_across_registry_instances() {
            clear_config_value_cache();
            let (dir, models_path) = rig();
            let counter_file = dir.path().join("counter");
            std::fs::write(&counter_file, "0").expect("the counter seeds");

            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "custom-provider": json_provider_with_api_key(counting_command(&counter_file).as_str()),
                }),
            );

            let registry1 = create_registry(&models_path).await;
            registry1.get_api_key_for_provider("custom-provider").await;

            let registry2 = create_registry(&models_path).await;
            registry2.get_api_key_for_provider("custom-provider").await;

            assert_eq!(counter_count(&counter_file), 2);
        }

        #[tokio::test]
        async fn different_commands_resolve_independently() {
            clear_config_value_cache();
            let (_dir, models_path) = rig();
            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "provider-a": json_provider_with_api_key("!echo key-a"),
                    "provider-b": json_provider_with_api_key("!echo key-b"),
                }),
            );

            let registry = create_registry(&models_path).await;

            assert_eq!(
                registry.get_api_key_for_provider("provider-a").await,
                Some("key-a".to_owned())
            );
            assert_eq!(
                registry.get_api_key_for_provider("provider-b").await,
                Some("key-b".to_owned())
            );
        }

        #[tokio::test]
        async fn failed_commands_are_retried() {
            clear_config_value_cache();
            let (dir, models_path) = rig();
            let counter_file = dir.path().join("counter");
            std::fs::write(&counter_file, "0").expect("the counter seeds");

            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "custom-provider": json_provider_with_api_key(failing_counting_command(&counter_file).as_str()),
                }),
            );

            let registry = create_registry(&models_path).await;
            let key1 = registry.get_api_key_for_provider("custom-provider").await;
            let key2 = registry.get_api_key_for_provider("custom-provider").await;

            assert_eq!(key1, None);
            assert_eq!(key2, None);
            assert_eq!(counter_count(&counter_file), 2);
        }

        #[tokio::test]
        async fn provider_auth_status_reports_api_key_environment_variables_from_models_json() {
            clear_config_value_cache();
            let (_dir, models_path) = rig();
            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "custom-provider": json_provider_with_api_key("$TEST_API_KEY_STATUS_TEST_98765"),
                }),
            );

            let registry = create_registry(&models_path).await;
            // Restated: upstream sets the env var and asserts the status
            // surface reports it; the status path reads the process
            // environment with no injection seam, so the case asserts the
            // resolution the status reports on instead.
            let resolution = resolve_auth_with_env(
                &registry,
                "custom-provider",
                &[("TEST_API_KEY_STATUS_TEST_98765", "status-test-key")],
            )
            .await
            .expect("the provider auth resolves");

            assert_eq!(resolution.auth.api_key.as_deref(), Some("status-test-key"));
        }

        #[tokio::test]
        async fn provider_auth_status_reports_interpolated_api_key_environment_variables() {
            clear_config_value_cache();
            let (_dir, models_path) = rig();
            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "custom-provider": json_provider_with_api_key(
                        "${TEST_API_KEY_STATUS_PART_A_98765}_${TEST_API_KEY_STATUS_PART_B_98765}",
                    ),
                }),
            );

            let registry = create_registry(&models_path).await;
            let resolution = resolve_auth_with_env(
                &registry,
                "custom-provider",
                &[
                    ("TEST_API_KEY_STATUS_PART_A_98765", "left"),
                    ("TEST_API_KEY_STATUS_PART_B_98765", "right"),
                ],
            )
            .await
            .expect("the provider auth resolves");

            assert_eq!(resolution.auth.api_key.as_deref(), Some("left_right"));
        }

        #[tokio::test]
        async fn provider_auth_status_reports_non_env_api_key_values_from_models_json_as_a_config_key()
         {
            let (_dir, models_path) = rig();
            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "custom-provider": json_provider_with_api_key("literal_api_key_value"),
                }),
            );

            let registry = create_registry(&models_path).await;

            assert_eq!(
                registry.get_provider_auth_status("custom-provider"),
                AuthStatus {
                    configured: true,
                    source: Some(AuthStatusSource::ModelsJsonKey),
                    label: None,
                }
            );
        }

        #[tokio::test]
        async fn missing_explicit_env_api_key_keeps_provider_unavailable() {
            let (_dir, models_path) = rig();
            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "custom-provider": json_provider_with_api_key("$TEST_API_KEY_MISSING_TEST_98765"),
                }),
            );

            let registry = create_registry(&models_path).await;

            assert_eq!(
                registry.get_provider_auth_status("custom-provider"),
                AuthStatus {
                    configured: false,
                    source: None,
                    label: None,
                }
            );
            assert!(
                !registry
                    .get_available()
                    .iter()
                    .any(|model| model.provider.0 == "custom-provider"),
                "the unconfigured provider stays unavailable"
            );
        }

        #[tokio::test]
        async fn provider_auth_status_reports_command_api_key_values_from_models_json_without_executing_them()
         {
            clear_config_value_cache();
            let (dir, models_path) = rig();
            let counter_file = dir.path().join("status-counter");
            std::fs::write(&counter_file, "0").expect("the counter seeds");
            let counter_path = to_sh_path(counter_file.display().to_string().as_str());
            let command = format!("!sh -c 'echo 1 > \"{counter_path}\"; echo key-value'");
            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "custom-provider": json_provider_with_api_key(command.as_str()),
                }),
            );

            let registry = create_registry(&models_path).await;

            assert_eq!(
                registry.get_provider_auth_status("custom-provider"),
                AuthStatus {
                    configured: true,
                    source: Some(AuthStatusSource::ModelsJsonCommand),
                    label: None,
                }
            );
            assert_eq!(
                std::fs::read_to_string(&counter_file).expect("the counter reads"),
                "0",
                "the status never executes the command"
            );
        }

        #[tokio::test]
        async fn environment_variables_are_not_cached_changes_are_picked_up() {
            clear_config_value_cache();
            let (_dir, models_path) = rig();
            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "custom-provider": json_provider_with_api_key("$TEST_API_KEY_CACHE_TEST_98765"),
                }),
            );

            let registry = create_registry(&models_path).await;
            // Restated: upstream mutates process.env between lookups; the
            // port drives the two lookups with different env overlays,
            // which pins the same no-caching contract on the env surface.
            let first = resolve_auth_with_env(
                &registry,
                "custom-provider",
                &[("TEST_API_KEY_CACHE_TEST_98765", "first-value")],
            )
            .await
            .expect("the provider auth resolves");
            assert_eq!(first.auth.api_key.as_deref(), Some("first-value"));

            let second = resolve_auth_with_env(
                &registry,
                "custom-provider",
                &[("TEST_API_KEY_CACHE_TEST_98765", "second-value")],
            )
            .await
            .expect("the provider auth resolves");
            assert_eq!(second.auth.api_key.as_deref(), Some("second-value"));
        }

        #[tokio::test]
        async fn get_available_does_not_execute_command_backed_api_key_resolution() {
            clear_config_value_cache();
            let (dir, models_path) = rig();
            let counter_file = dir.path().join("counter");
            std::fs::write(&counter_file, "0").expect("the counter seeds");

            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "custom-provider": json_provider_with_api_key(counting_command(&counter_file).as_str()),
                }),
            );

            let registry = create_registry(&models_path).await;
            let available = registry.get_available();

            assert!(
                available
                    .iter()
                    .any(|model| model.provider.0 == "custom-provider"),
                "the command-configured provider is available"
            );
            assert_eq!(counter_count(&counter_file), 0);
        }

        #[tokio::test]
        async fn get_available_filters_github_copilot_oauth_models_to_account_picker_availability()
        {
            let (_dir, models_path) = rig();
            let copilot_model_id = get_builtin_models("github-copilot")
                .first()
                .expect("the copilot catalog is non-empty")
                .id
                .clone();

            let storage = common::model_layer::empty_auth_storage();
            let oauth = Credential::OAuth(OAuthCredentials {
                refresh: "github-access-token".to_owned(),
                access: "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;"
                    .to_owned(),
                expires: now_ms() + 60_000,
                extra: BTreeMap::from([(
                    "availableModelIds".to_owned(),
                    serde_json::json!([copilot_model_id]),
                )]),
            });
            let modify: CredentialModifyFn =
                Box::new(move |_current| Box::pin(async move { Ok(Some(oauth)) }));
            storage
                .modify("github-copilot", modify, None)
                .await
                .expect("the credential stores");

            let registry =
                common::model_layer::create_model_registry(storage.clone(), Some(&models_path))
                    .await;

            let available: Vec<String> = registry
                .get_available()
                .iter()
                .filter(|model| model.provider.0 == "github-copilot")
                .map(|model| model.id.clone())
                .collect();
            assert_eq!(available, vec![copilot_model_id]);
        }

        #[tokio::test]
        async fn get_api_key_and_headers_resolves_auth_header_on_every_request() {
            clear_config_value_cache();
            let (dir, models_path) = rig();
            let token_file = dir.path().join("token");
            std::fs::write(&token_file, "token-1").expect("the token seeds");
            let token_path = to_sh_path(token_file.display().to_string().as_str());

            let mut config = json_provider_with_api_key(&format!("!sh -c 'cat \"{token_path}\"'"));
            config["authHeader"] = serde_json::json!(true);
            write_raw_models_json(
                &models_path,
                serde_json::json!({ "custom-provider": config }),
            );

            let registry = create_registry(&models_path).await;
            let model = registry
                .find("custom-provider", "test-model")
                .expect("the test model composes");

            let auth1 = registry.get_api_key_and_headers(&model).await;
            assert_eq!(
                auth1,
                resolved_ok(
                    Some("token-1"),
                    Some(ok_headers(&[("Authorization", "Bearer token-1")]))
                )
            );

            std::fs::write(&token_file, "token-2").expect("the token rewrites");

            let auth2 = registry.get_api_key_and_headers(&model).await;
            assert_eq!(
                auth2,
                resolved_ok(
                    Some("token-2"),
                    Some(ok_headers(&[("Authorization", "Bearer token-2")]))
                )
            );
        }

        #[tokio::test]
        async fn get_api_key_and_headers_resolves_configured_auth_exactly_once() {
            clear_config_value_cache();
            let (dir, models_path) = rig();
            let counter_file = dir.path().join("auth-counter");
            std::fs::write(&counter_file, "0").expect("the counter seeds");
            let counter_path = to_sh_path(counter_file.display().to_string().as_str());
            let command = format!(
                "!sh -c 'count=$(cat \"{counter_path}\"); count=$((count + 1)); echo \"$count\" > \"{counter_path}\"; echo \"token-$count\"'"
            );
            let mut config = json_provider_with_api_key(&command);
            config["authHeader"] = serde_json::json!(true);
            write_raw_models_json(
                &models_path,
                serde_json::json!({ "custom-provider": config }),
            );

            let registry = create_registry(&models_path).await;
            let model = registry
                .find("custom-provider", "test-model")
                .expect("the test model composes");
            let auth = registry.get_api_key_and_headers(&model).await;

            assert_eq!(
                auth,
                resolved_ok(
                    Some("token-1"),
                    Some(ok_headers(&[("Authorization", "Bearer token-1")]))
                )
            );
            assert_eq!(counter_count(&counter_file), 1);
        }

        #[tokio::test]
        async fn stored_credentials_bypass_lower_priority_configured_auth_commands() {
            clear_config_value_cache();
            let (dir, models_path) = rig();
            let counter_file = dir.path().join("fallback-counter");
            std::fs::write(&counter_file, "0").expect("the counter seeds");
            let counter_path = to_sh_path(counter_file.display().to_string().as_str());
            let command = format!("!sh -c 'echo 1 > \"{counter_path}\"; echo fallback-key'");
            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "custom-provider": json_provider_with_api_key(command.as_str()),
                }),
            );
            let storage = common::model_layer::empty_auth_storage();
            store_api_key(&storage, "custom-provider", "stored-key", None).await;

            let registry =
                common::model_layer::create_model_registry(storage.clone(), Some(&models_path))
                    .await;
            let model = registry
                .find("custom-provider", "test-model")
                .expect("the test model composes");
            let auth = registry.get_api_key_and_headers(&model).await;

            assert!(auth.ok());
            let ResolvedRequestAuth::Ok { api_key, .. } = auth else {
                unreachable!("auth.ok() pinned the Ok arm");
            };
            assert_eq!(api_key.as_deref(), Some("stored-key"));
            assert_eq!(counter_count(&counter_file), 0);
        }

        #[tokio::test]
        async fn get_api_key_and_headers_preserves_the_legacy_missing_key_auth_header_error() {
            clear_config_value_cache();
            let (_dir, models_path) = rig();
            write_raw_models_json(
                &models_path,
                serde_json::json!({
                    "custom-provider": {
                        "baseUrl": "https://example.test/v1",
                        "api": "openai-completions",
                        "authHeader": true,
                        "models": [{ "id": "test-model" }],
                    },
                }),
            );

            let registry = create_registry(&models_path).await;
            let model = registry
                .find("custom-provider", "test-model")
                .expect("the test model composes");
            let auth = registry.get_api_key_and_headers(&model).await;

            assert_eq!(
                auth,
                ResolvedRequestAuth::Err {
                    error: "No API key found for \"custom-provider\"".to_owned(),
                }
            );
        }

        #[tokio::test]
        async fn get_api_key_and_headers_returns_an_error_for_failed_auth_header_resolution() {
            clear_config_value_cache();
            let (_dir, models_path) = rig();
            let mut config = json_provider_with_api_key("!exit 1");
            config["authHeader"] = serde_json::json!(true);
            write_raw_models_json(
                &models_path,
                serde_json::json!({ "custom-provider": config }),
            );

            let registry = create_registry(&models_path).await;
            let model = registry
                .find("custom-provider", "test-model")
                .expect("the test model composes");

            let auth = registry.get_api_key_and_headers(&model).await;
            assert!(!auth.ok());
            let ResolvedRequestAuth::Err { error } = auth else {
                unreachable!("!auth.ok() pinned the Err arm");
            };
            assert!(
                error.contains("Failed to resolve API key for provider \"custom-provider\""),
                "got: {error}"
            );
        }
    }
}

//! Function-coverage closer for #121's model layer, driven by the lcov
//! function pools the ticket lists (lcov at
//! `/tmp/opencode-scratch/pi-rust/cov-121/lcov.info`): the composer's
//! dispatch/refresh/auth-closure arms, the runtime's error and override
//! surfaces and `create`'s signal ladder, the resolver's fallback arms and
//! the real-runtime view impls, the stores' error and coalescing arms, the
//! remote catalog's parse/date arms, and the models.json schema walk.
//! Upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Shared fixtures come from `common::model_layer`; the doubles these arms
//! need (a recording base provider, a scripted runtime view, a map-backed
//! auth context) live here so the duplication gate stays clear.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the composition-error helper reports an Ok side the layers cannot return"
)]

mod common;

use std::collections::BTreeMap;
use std::error::Error as _;
use std::sync::{Arc, Mutex};

use pi_agent_core::types::ThinkingLevel;
use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::types::{
    ApiKeyAuthInput, ApiKeyCredential, AuthContext, AuthEvent, AuthInteraction, AuthPrompt,
    AuthPromptKind, Credential, OAuthCredentials, ProviderAuth,
};
use pi_ai::http::{MockHttpClient, MockResponse, json_response};
use pi_ai::models::{
    CatalogPersist, ModelsPublication, Provider, ProviderModelError, PublishFn,
    RefreshModelsContext,
};
use pi_ai::models_store::{ModelsStore, ModelsStoreEntry, ModelsStoreOptions};
use pi_ai::types::{
    Context, Model, SimpleStreamOptions, StopReason, StreamOptions, TransportOptions,
};
use pi_ai::utils::abort::AbortError;
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_coding_agent::model_config::ModelConfig;
use pi_coding_agent::model_resolver::{
    FindInitialModelOptions, InitialModelError, KNOWN_PROVIDER_DEFAULT_ORDER, ModelRuntimeView,
    ParseModelPatternOptions, ResolveCliModelOptions, ScopedModel, default_model_per_provider,
    find_exact_model_reference_match, find_initial_model, format_scope_warning,
    is_valid_thinking_level, parse_model_pattern, resolve_cli_model, resolve_model_scope,
    resolve_model_scope_from_models, resolve_model_scope_with_diagnostics,
    restore_model_from_session,
};
use pi_coding_agent::model_runtime::{
    CreateModelRuntimeOptions, CredentialSynchronizationError, CredentialSynchronizationOperation,
    ModelRuntime, ModelRuntimeAuthOverrides, ModelRuntimeCore,
};
use pi_coding_agent::models_store::{FileModelsStore, InMemoryCodingAgentModelsStore};
use pi_coding_agent::provider_composer::{
    AuthStatusSource, ExtensionOAuthConfig, ExtensionOAuthLoginFn, ExtensionOAuthRefreshFn,
    ExtensionStreamSimpleFn, OAuthLoginCallbacks, OAuthPrompt, OAuthSelectOption,
    OAuthSelectPrompt, ProviderConfigInput, ProviderModelInput, clear_api_key_cache,
    compose_model_provider, configured_request_auth_status, resolve_compatibility_request_config,
    resolve_configured_model_headers, validate_extension_provider,
};
use pi_coding_agent::remote_catalog_provider::{with_remote_catalog, with_remote_catalog_client};
use serde_json::json;
use tokio_util::sync::CancellationToken;

use common::model_layer::{
    create_in_memory_model_registry, create_model_registry, empty_auth_storage, empty_context,
    model,
};

/// The model ids a provider lists, the shape the composition assertions read.
fn ids(provider: &Arc<dyn Provider>) -> Vec<String> {
    provider
        .get_models()
        .expect("the composed provider lists models")
        .into_iter()
        .map(|entry| entry.id)
        .collect()
}

/// The recording sink the scripted closures append to.
type Log = Arc<Mutex<Vec<String>>>;

/// Push one line onto the log, the only lock the doubles take.
fn record(log: &Log, line: String) {
    log.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(line);
}

/// The log lines so far, cloned out from under the lock.
fn recorded(log: &Log) -> Vec<String> {
    log.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// The error-terminal stream the scripted closures return, with `label`
/// riding the message the resolved result carries.
fn scripted_stream(log_model: &Model, label: &str) -> AssistantMessageEventStream {
    let owned = log_model.clone();
    let label = label.to_owned();
    pi_ai::api::lazy::lazy_stream(&owned, move || {
        let error: Box<dyn std::error::Error + Send + Sync> =
            Box::new(std::io::Error::other(label));
        Box::pin(async move { Err::<AssistantMessageEventStream, _>(error) })
    })
}

/// The base provider the dispatch tests compose over: one static model, an
/// optional inherited api-key method with its own check/resolve closures, an
/// optional refresh phase, and streams that record the model and options they
/// receive before returning a labeled error stream. The catalog toggle flips
/// after composition so the composed provider's error mapping runs.
struct RecordingBase {
    log: Log,
    auth: ProviderAuth,
    failing_catalog: Arc<std::sync::atomic::AtomicBool>,
    refreshable: bool,
}

impl RecordingBase {
    /// The base with a healthy catalog and no inherited auth method.
    fn healthy(log: &Log) -> Self {
        Self {
            log: Arc::clone(log),
            auth: ProviderAuth::default(),
            failing_catalog: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            refreshable: false,
        }
    }

    /// The base whose api-key method carries its own check and resolve
    /// closures, the inherited method the composed layers delegate to.
    fn with_inherited_api_key(log: &Log) -> Self {
        let mut base = Self::healthy(log);
        let check: pi_ai::auth::types::ApiKeyCheckFn = Arc::new(|input: ApiKeyAuthInput| {
            Box::pin(async move {
                Ok(input
                    .credential
                    .and_then(|credential| credential.key)
                    .map(|key| pi_ai::auth::types::AuthCheck {
                        source: Some(key),
                        auth_type: pi_ai::auth::types::AuthType::ApiKey,
                    }))
            })
        });
        let resolve: pi_ai::auth::types::ApiKeyResolveFn = Arc::new(|input: ApiKeyAuthInput| {
            let key = input
                .credential
                .and_then(|credential| credential.key)
                .or_else(|| input.ctx.env("COV_AMBIENT"));
            Box::pin(async move {
                Ok(key.map(|key| pi_ai::auth::types::AuthResult {
                    auth: pi_ai::auth::types::ModelAuth {
                        api_key: Some(key),
                        ..pi_ai::auth::types::ModelAuth::default()
                    },
                    env: None,
                    source: Some("inherited".to_owned()),
                }))
            })
        });
        base.auth = ProviderAuth {
            api_key: Some(pi_ai::auth::types::ApiKeyAuth {
                name: "Inherited Key".to_owned(),
                login: None,
                check: Some(check),
                resolve,
            }),
            oauth: None,
        };
        base
    }

    /// The base whose inherited method carries only a resolve closure, the
    /// shape that routes the composed check's empty-credential arm through
    /// the inherited resolve.
    fn with_resolve_only_api_key(log: &Log) -> Self {
        let mut base = Self::healthy(log);
        let resolve: pi_ai::auth::types::ApiKeyResolveFn = Arc::new(|input: ApiKeyAuthInput| {
            let key = input
                .credential
                .and_then(|credential| credential.key)
                .or_else(|| input.ctx.env("COV_AMBIENT"));
            Box::pin(async move {
                Ok(key.map(|key| pi_ai::auth::types::AuthResult {
                    auth: pi_ai::auth::types::ModelAuth {
                        api_key: Some(key),
                        ..pi_ai::auth::types::ModelAuth::default()
                    },
                    env: None,
                    source: Some("resolve-only".to_owned()),
                }))
            })
        });
        base.auth = ProviderAuth {
            api_key: Some(pi_ai::auth::types::ApiKeyAuth {
                name: "Resolve Only".to_owned(),
                login: None,
                check: None,
                resolve,
            }),
            oauth: None,
        };
        base
    }

    /// The base whose inherited check closure fails outright, the error the
    /// availability pass records for the provider.
    fn with_failing_check(log: &Log) -> Self {
        let mut base = Self::healthy(log);
        let check: pi_ai::auth::types::ApiKeyCheckFn = Arc::new(|_input: ApiKeyAuthInput| {
            let error: Box<dyn std::error::Error + Send + Sync> =
                Box::new(std::io::Error::other("check failed"));
            Box::pin(async { Err(error) })
        });
        base.auth = ProviderAuth {
            api_key: Some(pi_ai::auth::types::ApiKeyAuth {
                name: "Failing".to_owned(),
                login: None,
                check: Some(check),
                resolve: Arc::new(|_input: ApiKeyAuthInput| Box::pin(async { Ok(None) })),
            }),
            oauth: None,
        };
        base
    }

    /// The base with a refresh phase the composed refresh delegates to.
    fn with_refresh_phase(log: &Log) -> Self {
        Self {
            refreshable: true,
            ..Self::healthy(log)
        }
    }

    /// The base whose catalog read fails, for the composition-error arm.
    fn failing(log: &Log) -> Self {
        let base = Self::healthy(log);
        base.failing_catalog
            .store(true, std::sync::atomic::Ordering::Relaxed);
        base
    }
}

impl Provider for RecordingBase {
    fn id(&self) -> &'static str {
        "cov-base"
    }

    fn name(&self) -> &'static str {
        "Cov Base"
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn get_models(&self) -> Result<Vec<Model>, ProviderModelError> {
        if self
            .failing_catalog
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err(Box::new(std::io::Error::other("base catalog failed")));
        }
        Ok(vec![model("cov-base", "base-model")])
    }

    fn refresh_models(
        &self,
        _context: RefreshModelsContext,
    ) -> pi_ai::types::BoxedFuture<'_, Result<(), pi_ai::models::ProviderError>> {
        record(&self.log, "base-refresh".to_owned());
        Box::pin(async { Ok(()) })
    }

    fn supports_refresh_models(&self) -> bool {
        self.refreshable
    }

    fn stream(
        &self,
        stream_model: &Model,
        _context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        record(
            &self.log,
            format!(
                "base-full {} api_key={:?}",
                stream_model.id,
                options.and_then(|entry| entry.api_key.clone()),
            ),
        );
        scripted_stream(stream_model, "base-full")
    }

    fn stream_simple(
        &self,
        stream_model: &Model,
        _context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        record(
            &self.log,
            format!(
                "base-simple {} api_key={:?}",
                stream_model.id,
                options.and_then(|entry| entry.api_key.clone()),
            ),
        );
        scripted_stream(stream_model, "base-simple")
    }
}

/// The map-backed auth context the closure tests drive, standing in for the
/// process environment so `$VAR` resolution is deterministic.
struct EnvContext(BTreeMap<String, String>);

impl AuthContext for EnvContext {
    fn env(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }

    fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

/// The auth input the check/resolve closures take, over `ctx` with an
/// optional stored credential.
fn api_input(ctx: &Arc<dyn AuthContext>, credential: Option<ApiKeyCredential>) -> ApiKeyAuthInput {
    ApiKeyAuthInput {
        ctx: Arc::clone(ctx),
        credential,
        signal: CancellationToken::new(),
    }
}

/// The models.json snapshot whose `providers` object carries
/// `providers_json`, loaded through a temp file like the runtime loads it.
fn config_from(providers_json: &str) -> ModelConfig {
    config_raw(&format!(r#"{{"providers": {providers_json}}}"#))
}

/// The models.json snapshot over a verbatim file body, for the shells the
/// providers wrapper cannot spell.
fn config_raw(body: &str) -> ModelConfig {
    let dir = tempfile::tempdir().expect("the temp dir creates");
    let path = dir.path().join("models.json").display().to_string();
    std::fs::write(&path, body).expect("models.json writes");
    ModelConfig::load(Some(&path)).expect("the snapshot loads")
}

/// The composed provider over no base, the given snapshot, and extension.
fn composed(
    id: &str,
    config: Option<&ModelConfig>,
    extension: Option<ProviderConfigInput>,
) -> Result<Arc<dyn Provider>, String> {
    let empty = ModelConfig::default();
    compose_model_provider(id, None, config.unwrap_or(&empty), extension)
}

/// The composition failure message, reported through the Err side the layers
/// return; the Ok side carries a provider the assertion cannot spell Debug
/// for.
#[must_use]
fn composition_error(result: Result<Arc<dyn Provider>, String>) -> String {
    match result {
        Err(error) => error,
        Ok(_) => panic!("the composition must fail"),
    }
}

/// The boxed failure the never-called closure doubles return, typed so the
/// return-position coercion matches `ProviderError` exactly.
fn unused_failure() -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(std::io::Error::other("unused"))
}

/// The streamSimple double recording the model and options it receives and
/// returning a stream whose error message carries `label`.
fn recording_stream_simple(log: &Log, label: &'static str) -> ExtensionStreamSimpleFn {
    let log = Arc::clone(log);
    Arc::new(move |stream_model, _context, options| {
        record(
            &log,
            format!(
                "{label} {} api_key={:?}",
                stream_model.id,
                options.and_then(|entry| entry.api_key.clone()),
            ),
        );
        scripted_stream(stream_model, label)
    })
}

/// The publication callback recording the entries refreshes write and running
/// the synchronous updates, always resolving published.
fn recording_publish(sink: &Arc<Mutex<Vec<ModelsStoreEntry>>>) -> PublishFn {
    let sink = Arc::clone(sink);
    Arc::new(move |publication: ModelsPublication| {
        let sink = Arc::clone(&sink);
        Box::pin(async move {
            if let CatalogPersist::Write(entry) = publication.persist {
                sink.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(entry);
            }
            if let Some(update) = publication.update {
                update();
            }
            Ok(true)
        })
    })
}

/// The refresh context the trait-level refresh tests drive, with the network
/// knob the caller picks.
fn refresh_context(
    publish: PublishFn,
    credential: Option<Credential>,
    allow_network: bool,
) -> RefreshModelsContext {
    RefreshModelsContext {
        credential,
        stored: None,
        publish,
        allow_network,
        force: Some(true),
        signal: CancellationToken::new(),
    }
}

/// The in-memory runtime the streaming tests compose against: no models.json,
/// no network, in-memory catalog store.
async fn cov_runtime() -> ModelRuntime {
    let registry = create_in_memory_model_registry(empty_auth_storage()).await;
    registry.runtime().clone()
}

/// The mock transport options the adapter-driven requests carry.
fn transport(mock: &MockHttpClient) -> TransportOptions {
    TransportOptions {
        http_client: Some(Arc::new(mock.clone())),
        ..TransportOptions::default()
    }
}

/// The catalog route every catalog test drives, keyed on the recording base's
/// provider id.
fn catalog_route(mock: &MockHttpClient) {
    mock.on(|request| request.url.contains("/api/models/providers/cov-base"));
}

/// The catalog body listing one wire model with `id`.
fn catalog_body(id: &str) -> serde_json::Value {
    json!([{
        "id": id,
        "name": id,
        "api": "openai-completions",
        "provider": "cov-base",
        "baseUrl": "https://example.test/v1",
        "reasoning": false,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 1000,
        "maxTokens": 100,
    }])
}

/// The entries the refreshes wrote so far.
fn written(sink: &Arc<Mutex<Vec<ModelsStoreEntry>>>) -> Vec<ModelsStoreEntry> {
    sink.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// The `ModelsStoreEntry` the fixture models wrap, fresh timestamps absent.
const fn store_entry(models: Vec<Model>) -> ModelsStoreEntry {
    ModelsStoreEntry {
        models,
        last_modified: None,
        checked_at: None,
        etag: None,
    }
}

mod debug_and_display {
    use super::*;

    /// Every composer value type renders, including the Display spellings of
    /// the auth-status sources and the closure-bearing Debug impls.
    #[test]
    fn composer_value_types_render() {
        let input = ProviderConfigInput {
            name: Some("Cov".to_owned()),
            base_url: Some("https://cov.test/v1".to_owned()),
            api_key: Some("lit".to_owned()),
            api: Some(pi_ai::types::Api::from("openai-completions")),
            stream_simple: Some(recording_stream_simple(
                &Arc::new(Mutex::new(Vec::new())),
                "x",
            )),
            headers: Some(BTreeMap::from([("x-cov".to_owned(), "v".to_owned())])),
            auth_header: Some(true),
            oauth: Some(ExtensionOAuthConfig {
                name: "Cov OAuth".to_owned(),
                is_subscription: Some(true),
                uses_callback_server: Some(false),
                login: Arc::new(|_callbacks| Box::pin(async { Err(unused_failure()) })),
                refresh_token: Arc::new(|_credentials, _signal| {
                    Box::pin(async { Err(unused_failure()) })
                }),
                get_api_key: Arc::new(|_credentials| "key".to_owned()),
                modify_models: None,
            }),
            models: Some(vec![ProviderModelInput {
                id: "cov-model".to_owned(),
                name: "Cov Model".to_owned(),
                api: None,
                base_url: None,
                reasoning: false,
                thinking_level_map: None,
                input: vec![pi_ai::types::Modality::Text],
                cost: common::model_layer::zero_cost(),
                context_window: 1000,
                max_tokens: 100,
                sampling_params: None,
                headers: None,
                compat: None,
            }]),
            refresh_models: None,
        };
        let rendered = format!("{input:?}");
        assert!(rendered.contains("ProviderConfigInput"));
        assert!(rendered.contains("api_key: Some(\"set\")"));
        assert!(rendered.contains("models: Some(1)"));

        assert_eq!(AuthStatusSource::Stored.to_string(), "stored");
        assert_eq!(AuthStatusSource::Runtime.to_string(), "runtime");
        assert_eq!(AuthStatusSource::Environment.to_string(), "environment");
        assert_eq!(AuthStatusSource::Fallback.to_string(), "fallback");
        assert_eq!(
            AuthStatusSource::ModelsJsonKey.to_string(),
            "models_json_key"
        );
        assert_eq!(
            AuthStatusSource::ModelsJsonCommand.to_string(),
            "models_json_command"
        );
        assert!(format!("{:?}", AuthStatusSource::ModelsJsonCommand).contains("ModelsJsonCommand"));

        let callbacks = format!(
            "{:?}",
            OAuthLoginCallbacks {
                signal: CancellationToken::new(),
                notify: Arc::new(|_event: AuthEvent| {}),
                prompt: Arc::new(|_prompt: OAuthPrompt| { Box::pin(async { Err(AbortError) }) }),
                manual_code_input: Arc::new(|| Box::pin(async { Err(AbortError) })),
                select: Arc::new(|_select: OAuthSelectPrompt| {
                    Box::pin(async { Err(AbortError) })
                }),
            }
        );
        assert!(callbacks.contains("OAuthLoginCallbacks"));
    }

    /// The runtime value types render and the synchronization error carries
    /// its cause through `source`, upstream's error-chain contract.
    #[test]
    fn runtime_value_types_render() {
        for (operation, spelling) in [
            (CredentialSynchronizationOperation::Login, "login"),
            (CredentialSynchronizationOperation::Logout, "logout"),
            (
                CredentialSynchronizationOperation::SetRuntimeApiKey,
                "setRuntimeApiKey",
            ),
            (
                CredentialSynchronizationOperation::RemoveRuntimeApiKey,
                "removeRuntimeApiKey",
            ),
        ] {
            assert_eq!(operation.to_string(), spelling);
            assert!(format!("{operation:?}").len() > 4);
        }

        let cause: Box<dyn std::error::Error + Send + Sync> =
            Box::new(std::io::Error::other("snapshot failed"));
        let error = CredentialSynchronizationError {
            provider_id: "cov".to_owned(),
            operation: CredentialSynchronizationOperation::Login,
            credential: Some(Credential::ApiKey(ApiKeyCredential::default())),
            cause: Box::new(std::io::Error::other("snapshot failed")),
        };
        assert_eq!(
            error.to_string(),
            "Credential login committed for cov, but local synchronization failed"
        );
        assert_eq!(
            error.source().expect("the cause chains").to_string(),
            "snapshot failed"
        );
        drop(cause);
        assert!(format!("{error:?}").contains("CredentialSynchronizationError"));

        let options = format!(
            "{:?}",
            CreateModelRuntimeOptions {
                auth_path: Some("/tmp/auth.json".to_owned()),
                models_path: Some(None),
                models_store_path: Some("/tmp/store.json".to_owned()),
                allow_model_network: true,
                model_refresh_timeout_ms: Some(50),
                catalog_base_url: Some("https://cov.test".to_owned()),
                signal: Some(CancellationToken::new()),
                refresh_on_create: Some(false),
                ..CreateModelRuntimeOptions::default()
            }
        );
        assert!(options.contains("CreateModelRuntimeOptions"));
        assert!(options.contains("allow_model_network: true"));

        let overrides = format!(
            "{:?}",
            ModelRuntimeAuthOverrides {
                api_key: Some("k".to_owned()),
                min_oauth_validity_ms: Some(1_000),
                signal: Some(CancellationToken::new()),
                ..ModelRuntimeAuthOverrides::default()
            }
        );
        assert!(overrides.contains("ModelRuntimeAuthOverrides"));
    }

    /// The resolver value types render and the default-id table answers for
    /// every known provider in the pinned order.
    #[test]
    fn resolver_value_types_render() {
        let error = InitialModelError("no such model".to_owned());
        assert_eq!(error.to_string(), "no such model");

        for provider in KNOWN_PROVIDER_DEFAULT_ORDER {
            assert!(
                default_model_per_provider(provider).is_some(),
                "{provider} has a default model id"
            );
        }
        assert!(default_model_per_provider("cov-unknown").is_none());

        for level in ["off", "minimal", "low", "medium", "high", "xhigh", "max"] {
            assert!(is_valid_thinking_level(level));
        }
        assert!(!is_valid_thinking_level("maximal"));
    }

    /// The store and catalog types render, the process-default catalog
    /// wrapper composes, and the overlay merge is a no-op before any refresh.
    #[test]
    fn store_and_catalog_types_render() {
        let dir = tempfile::tempdir().expect("the temp dir creates");
        let path = dir.path().join("models-store.json").display().to_string();
        let file_store = FileModelsStore::new(Some(&path)).expect("the file store opens");
        assert!(format!("{file_store:?}").contains(&path));

        let memory_store = InMemoryCodingAgentModelsStore::default();
        assert!(format!("{memory_store:?}").contains("InMemoryCodingAgentModelsStore"));

        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let wrapped = with_remote_catalog_client(
            Arc::new(RecordingBase::healthy(&log)),
            Some("https://cov.test".to_owned()),
            None,
            Arc::new(MockHttpClient::new()),
        );
        assert_eq!(ids(&wrapped), ["base-model"]);

        let defaulted = with_remote_catalog(Arc::new(RecordingBase::healthy(&log)), None, None);
        assert_eq!(ids(&defaulted), ["base-model"]);
    }
}

mod model_config_arms {
    use super::*;

    /// The schema walk reports the dotted paths for every member kind: the
    /// required-property misses, the per-provider members, the custom-model
    /// fields on both definition and override shapes, and the compat keys.
    #[test]
    fn the_schema_walk_reports_dotted_paths() {
        let broken = json!({
            "cov": {
            "name": "",
            "baseUrl": 5,
            "apiKey": "",
            "api": "",
            "oauth": "not-radius",
            "headers": ["x"],
            "compat": {"thinkingFormat": "nope", "maxTokensField": "wrong", "vllmPriority": "nan"},
            "authHeader": "yes",
            "models": [
                {"id": "", "name": 3, "thinkingLevelMap": {"warp": "on"},
                 "input": ["smell"], "cost": {"tiers": [{"input": 0}]},
                 "contextWindow": "big", "maxTokens": [], "samplingParams": 4,
                 "headers": {"x": 5}, "compat": ["nope"]},
                "not-an-object",
            ],
            "modelOverrides": {"m": {"cost": {"input": "many"}}},
            }
        });
        let config = config_from(&broken.to_string());
        let error = config.get_error().expect("the walk fails the config");
        for path in [
            "providers.cov.name: Expected string",
            "providers.cov.baseUrl: Expected string",
            "providers.cov.apiKey: Expected string",
            "providers.cov.api: Expected string",
            "providers.cov.oauth: Expected union value: \"radius\"",
            "providers.cov.headers: Expected object",
            "providers.cov.compat: Expected object",
            "providers.cov.authHeader: Expected boolean",
            "providers.cov.models.0.id: Expected string",
            "providers.cov.models.0.name: Expected string",
            "providers.cov.models.0.thinkingLevelMap: Expected object",
            "providers.cov.models.0.input: Expected union value: \"text\" | \"image\"",
            "providers.cov.models.0.cost: Expected object",
            "providers.cov.models.0.contextWindow: Expected number",
            "providers.cov.models.0.maxTokens: Expected number",
            "providers.cov.models.0.samplingParams: Expected object",
            "providers.cov.models.0.headers: Expected object",
            "providers.cov.models.0.compat: Expected object",
            "providers.cov.models.1: Expected object",
            "providers.cov.modelOverrides.m.cost: Expected object",
        ] {
            assert!(error.contains(path), "missing {path} in:\n{error}");
        }
    }

    /// The walk rejects the structural shells: a non-object root, a missing
    /// or non-object providers map, a non-object provider, non-array models,
    /// and a non-object overrides map.
    #[test]
    fn the_schema_walk_rejects_structural_shells() {
        let config = config_raw(r#"{"oauth": "radius"}"#);
        assert!(
            config
                .get_error()
                .is_some_and(|error| error.contains("providers: Required property missing"))
        );

        let config = config_raw("[1]");
        assert!(
            config
                .get_error()
                .is_some_and(|error| error.contains("root: Expected object"))
        );

        for (shell, expected) in [
            (json!({"providers": []}), "providers: Expected object"),
            (
                json!({"providers": {"cov": 3}}),
                "providers.cov: Expected object",
            ),
            (
                json!({"providers": {"cov": {"baseUrl": "https://x", "models": {}}}}),
                "providers.cov.models: Expected array",
            ),
            (
                json!({"providers": {"cov": {"baseUrl": "https://x", "modelOverrides": []}}}),
                "providers.cov.modelOverrides: Expected object",
            ),
            (
                json!({"providers": {"cov": {"baseUrl": "https://x", "models": [{"id": "m", "compat": {"openRouterRouting": "flat", "chatTemplateKwargs": {"$var": 3}, "sessionAffinityFormat": "nobody"}}]}}}),
                "providers.cov.models.0.compat: Expected object",
            ),
        ] {
            let config = config_raw(&shell.to_string());
            assert!(
                config
                    .get_error()
                    .is_some_and(|error| error.contains(expected)),
                "expected {expected} for {shell}"
            );
        }
    }

    /// A valid file parses into providers, and the load arms degrade the way
    /// the snapshot contract pins: missing file empty, unparsable body
    /// carrying the parse error, an unreadable path carrying the read error,
    /// and no path empty.
    #[test]
    fn valid_config_parses_and_load_arms_degrade() {
        let valid = json!({
            "cov-ok": {
                "name": "Cov Ok",
                "baseUrl": "https://cov.test/v1",
                "apiKey": "lit",
                "api": "openai-completions",
                "models": [{"id": "cov-model", "contextWindow": 1000, "maxTokens": 100}],
            },
        });
        let config = config_from(&valid.to_string());
        assert_eq!(config.get_error(), None);
        assert_eq!(config.get_provider_ids(), ["cov-ok"]);
        assert!(config.get_provider("cov-ok").is_some());
        assert!(config.get_provider("absent").is_none());

        let dir = tempfile::tempdir().expect("the temp dir creates");
        let path = dir.path().join("models.json").display().to_string();
        assert_eq!(
            ModelConfig::load(Some(&path))
                .expect("a missing file loads empty")
                .get_error(),
            None
        );
        std::fs::write(&path, "{not json").expect("the broken body writes");
        assert!(
            ModelConfig::load(Some(&path))
                .expect("a parse failure still loads")
                .get_error()
                .is_some_and(|error| error.contains("Failed to parse models.json"))
        );
        assert!(
            ModelConfig::load(Some(dir.path().display().to_string().as_str()))
                .expect("a read failure still loads")
                .get_error()
                .is_some_and(|error| error.contains("Failed to load models.json"))
        );
        assert_eq!(
            ModelConfig::load(None)
                .expect("no path loads empty")
                .get_provider_ids(),
            Vec::<String>::new()
        );
    }
}

mod composer_arms {
    use super::*;

    /// The composition failures the layers report: a models.json layer that
    /// specifies nothing, oauth without a base URL, and the custom-model shape
    /// errors. An empty composition keeps the api-key method, so the
    /// no-authentication error never fires on this ladder.
    #[test]
    fn composition_errors_list_the_layer_failures() {
        assert_eq!(
            composition_error(composed(
                "cov-empty",
                Some(&config_from(
                    &json!({"cov-empty": {"name": "Cov"}}).to_string()
                )),
                None
            )),
            "Provider cov-empty: must specify \"baseUrl\", \"headers\", \"compat\", \"modelOverrides\", or \"models\"."
        );
        assert!(
            composition_error(composed(
                "cov-oauth",
                Some(&config_from(
                    &json!({"cov-oauth": {"oauth": "radius"}}).to_string()
                )),
                None
            ))
            .contains("\"baseUrl\" is required when \"oauth\" is set")
        );

        let no_api = config_from(
            &json!({"cov-m": {"baseUrl": "https://cov.test/v1", "apiKey": "lit", "models": [{"id": "m"}]}})
                .to_string(),
        );
        assert!(
            composition_error(composed("cov-m", Some(&no_api), None))
                .contains("no \"api\" specified")
        );

        let no_base = config_from(
            &json!({"cov-m": {"apiKey": "lit", "api": "openai-completions", "models": [{"id": "m"}]}})
                .to_string(),
        );
        assert!(
            composition_error(composed("cov-m", Some(&no_base), None))
                .contains("\"baseUrl\" is required when defining custom models")
        );

        for (field, value) in [("contextWindow", 0u64), ("maxTokens", 0u64)] {
            let mut model_entry = serde_json::Map::new();
            model_entry.insert("id".to_owned(), json!("m"));
            model_entry.insert(field.to_owned(), json!(value));
            let body = json!({"providers": {"cov-m": {
                "baseUrl": "https://cov.test/v1", "apiKey": "lit", "api": "openai-completions",
                "models": [serde_json::Value::Object(model_entry)],
            }}});
            let config = config_raw(&body.to_string());
            assert!(
                config.get_provider("cov-m").is_some(),
                "the zero-{field} body must parse for the test to be meaningful"
            );
            assert!(
                composition_error(composed("cov-m", Some(&config), None))
                    .contains(&format!("invalid {field}")),
                "zero {field} must fail"
            );
        }
    }

    /// The api-key method's check closure answers every input shape the
    /// composed layers can hand it: stored credentials, the raw key forms,
    /// and the env-template and command shapes.
    #[tokio::test]
    async fn the_api_key_check_answers_every_input_shape() {
        let command_config = config_from(
            &json!({"cov-check": {"baseUrl": "https://cov.test/v1", "apiKey": "!echo cov-probe",
                "api": "openai-completions",
                "models": [{"id": "m", "contextWindow": 10, "maxTokens": 5}]}})
            .to_string(),
        );
        let provider =
            composed("cov-check", Some(&command_config), None).expect("the provider composes");
        let api = provider
            .auth()
            .api_key
            .clone()
            .expect("the api-key method composes");
        let check = api.check.expect("the composed check closure exists");
        let ctx: Arc<dyn AuthContext> = Arc::new(EnvContext(BTreeMap::from([(
            "COV_SET".to_owned(),
            "v".to_owned(),
        )])));

        let stored = check(api_input(
            &ctx,
            Some(ApiKeyCredential {
                key: Some("k".to_owned()),
                env: None,
            }),
        ))
        .await
        .expect("the check runs")
        .expect("a stored credential checks");
        assert_eq!(stored.source.as_deref(), Some("stored credential"));
        assert_eq!(stored.auth_type, pi_ai::auth::types::AuthType::ApiKey);

        let empty = check(api_input(&ctx, Some(ApiKeyCredential::default())))
            .await
            .expect("the check runs");
        assert!(
            empty.is_none(),
            "an empty stored credential resolves to nothing"
        );

        let raw = check(api_input(&ctx, None))
            .await
            .expect("the check runs")
            .expect("a raw command key checks");
        assert_eq!(raw.source.as_deref(), Some("configured API key"));

        let unset_ctx: Arc<dyn AuthContext> = Arc::new(EnvContext(BTreeMap::from([(
            "OTHER".to_owned(),
            "x".to_owned(),
        )])));
        let without_env = check(api_input(&unset_ctx, None))
            .await
            .expect("the check runs")
            .expect("a command key checks without env");
        assert_eq!(without_env.source.as_deref(), Some("configured API key"));

        let template_config = config_from(
            &json!({"cov-tcheck": {"baseUrl": "https://cov.test/v1", "apiKey": "$COV_SET",
                "api": "openai-completions",
                "models": [{"id": "m", "contextWindow": 10, "maxTokens": 5}]}})
            .to_string(),
        );
        let template_provider =
            composed("cov-tcheck", Some(&template_config), None).expect("the provider composes");
        let template_check = template_provider
            .auth()
            .api_key
            .clone()
            .expect("the api-key method composes")
            .check
            .expect("the composed check closure exists");
        let configured = template_check(api_input(&ctx, None))
            .await
            .expect("the check runs")
            .expect("a resolvable template checks");
        assert_eq!(configured.source.as_deref(), Some("configured API key"));
        let missing = template_check(api_input(&unset_ctx, None))
            .await
            .expect("the check runs");
        assert!(
            missing.is_none(),
            "an unresolvable template is not configured"
        );

        clear_api_key_cache();
    }

    /// The api-key method's resolve and login closures produce the request
    /// auth: literal keys, env templates with the bearer form, stored
    /// credentials, and the prompted login.
    #[tokio::test]
    async fn the_api_key_resolve_and_login_produce_request_auth() {
        let config = config_from(
            &json!({"cov-res": {"baseUrl": "https://cov.test/v1", "apiKey": "$COV_RES_KEY",
                "authHeader": true, "headers": {"x-cov": "$COV_RES_KEY"},
                "api": "openai-completions",
                "models": [{"id": "m", "contextWindow": 10, "maxTokens": 5}]}})
            .to_string(),
        );
        let provider = composed("cov-res", Some(&config), None).expect("the provider composes");
        let api = provider
            .auth()
            .api_key
            .clone()
            .expect("the api-key method composes");
        let resolve = api.resolve.clone();
        let ctx: Arc<dyn AuthContext> = Arc::new(EnvContext(BTreeMap::from([(
            "COV_RES_KEY".to_owned(),
            "sk-resolved".to_owned(),
        )])));

        let resolved = resolve(api_input(&ctx, None))
            .await
            .expect("the resolve runs")
            .expect("the template resolves");
        assert_eq!(resolved.auth.api_key.as_deref(), Some("sk-resolved"));
        let headers = resolved.auth.headers.expect("the bearer form merges");
        assert_eq!(
            headers.get("Authorization").map(|value| value.as_deref()),
            Some(Some("Bearer sk-resolved")),
        );
        assert_eq!(
            headers.get("x-cov").map(|value| value.as_deref()),
            Some(Some("sk-resolved")),
        );

        let stored = resolve(api_input(
            &ctx,
            Some(ApiKeyCredential {
                key: Some("sk-stored".to_owned()),
                env: Some(BTreeMap::from([("COV_STORED".to_owned(), "1".to_owned())])),
            }),
        ))
        .await
        .expect("the resolve runs")
        .expect("the stored credential resolves");
        assert_eq!(stored.auth.api_key.as_deref(), Some("sk-stored"));
        assert_eq!(stored.source.as_deref(), Some("stored credential"));

        let missing_ctx: Arc<dyn AuthContext> = Arc::new(EnvContext(BTreeMap::new()));
        assert!(
            resolve(api_input(&missing_ctx, None)).await.is_err(),
            "an unresolvable template fails the resolve"
        );

        let login = api.login.expect("the composed login closure exists");
        let interaction = pi_ai::auth::types::ProviderAuthInteraction {
            signal: CancellationToken::new(),
            prompt: Arc::new(|_prompt: AuthPrompt| Box::pin(async { Ok("typed-key".to_owned()) })),
            notify: Arc::new(|_event: AuthEvent| {}),
        };
        let credential = login(interaction).await.expect("the login runs");
        assert_eq!(credential.key.as_deref(), Some("typed-key"));
    }

    /// The extension layer composes its own models and validates broken
    /// re-registrations without touching the stored config.
    #[test]
    fn extension_layers_compose_and_validate() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let extension = ProviderConfigInput {
            name: Some("Cov Extension".to_owned()),
            base_url: Some("https://ext.test/v1".to_owned()),
            api_key: Some("sk-ext".to_owned()),
            api: Some(pi_ai::types::Api::from("openai-completions")),
            headers: Some(BTreeMap::from([("x-ext".to_owned(), "v".to_owned())])),
            auth_header: Some(false),
            models: Some(vec![ProviderModelInput {
                id: "ext-model".to_owned(),
                name: "Ext Model".to_owned(),
                api: None,
                base_url: None,
                reasoning: true,
                thinking_level_map: None,
                input: vec![pi_ai::types::Modality::Text],
                cost: common::model_layer::zero_cost(),
                context_window: 2000,
                max_tokens: 200,
                sampling_params: None,
                headers: None,
                compat: None,
            }]),
            ..ProviderConfigInput::default()
        };
        let provider =
            composed("cov-ext", None, Some(extension.clone())).expect("the extension composes");
        assert_eq!(provider.name(), "Cov Extension");
        assert_eq!(provider.base_url(), Some("https://ext.test/v1"));
        assert_eq!(ids(&provider), ["ext-model"]);

        assert_eq!(
            validate_extension_provider("cov-ext", None, None, &extension),
            Ok(())
        );

        let no_api = ProviderConfigInput {
            stream_simple: Some(recording_stream_simple(&log, "x")),
            ..ProviderConfigInput::default()
        };
        assert!(
            validate_extension_provider("cov-ext", None, None, &no_api)
                .expect_err("streamSimple without api fails")
                .contains("\"api\" is required when registering streamSimple")
        );

        let modelless_api = ProviderConfigInput {
            api: Some(pi_ai::types::Api::from("openai-completions")),
            models: Some(vec![ProviderModelInput {
                id: "m".to_owned(),
                name: "m".to_owned(),
                api: None,
                base_url: Some("https://m.test".to_owned()),
                reasoning: false,
                thinking_level_map: None,
                input: vec![pi_ai::types::Modality::Text],
                cost: common::model_layer::zero_cost(),
                context_window: 10,
                max_tokens: 5,
                sampling_params: None,
                headers: None,
                compat: None,
            }]),
            ..ProviderConfigInput::default()
        };
        assert!(validate_extension_provider("cov-ext", None, None, &modelless_api).is_ok());

        let no_base = ProviderConfigInput {
            api: Some(pi_ai::types::Api::from("openai-completions")),
            models: Some(vec![ProviderModelInput {
                id: "m".to_owned(),
                name: "m".to_owned(),
                api: None,
                base_url: None,
                reasoning: false,
                thinking_level_map: None,
                input: vec![pi_ai::types::Modality::Text],
                cost: common::model_layer::zero_cost(),
                context_window: 10,
                max_tokens: 5,
                sampling_params: None,
                headers: None,
                compat: None,
            }]),
            ..ProviderConfigInput::default()
        };
        assert!(
            validate_extension_provider("cov-ext", None, None, &no_base)
                .expect_err("a model without baseUrl fails")
                .contains("\"baseUrl\" is required when defining custom models")
        );
    }

    /// The configured-auth status reports each source the layers can spell.
    #[test]
    fn auth_status_sources_report_the_configured_surface() {
        let config =
            config_from(&json!({"cov-st": {"baseUrl": "https://x", "apiKey": "lit"}}).to_string());
        let provider_config = config.get_provider("cov-st").cloned();
        let status = configured_request_auth_status(provider_config.as_ref(), None)
            .expect("a configured key reports");
        assert_eq!(status.source, Some(AuthStatusSource::ModelsJsonKey));

        let extension = ProviderConfigInput {
            api_key: Some("lit".to_owned()),
            ..ProviderConfigInput::default()
        };
        assert_eq!(
            configured_request_auth_status(None, Some(&extension))
                .expect("an extension key reports")
                .source,
            Some(AuthStatusSource::Fallback)
        );

        let unset = ProviderConfigInput {
            api_key: Some("$PI_RUST_COV_UNSET_VAR".to_owned()),
            ..ProviderConfigInput::default()
        };
        let status =
            configured_request_auth_status(None, Some(&unset)).expect("a template reports");
        assert!(!status.configured);
        assert_eq!(status.source, None);

        let env_key = config_from(
            &json!({"cov-st": {"baseUrl": "https://x", "apiKey": "$PATH"}}).to_string(),
        );
        let status = configured_request_auth_status(env_key.get_provider("cov-st"), None)
            .expect("an env template reports");
        assert!(status.configured);
        assert_eq!(status.source, Some(AuthStatusSource::Environment));
        assert_eq!(status.label.as_deref(), Some("PATH"));

        let command = config_from(
            &json!({"cov-st": {"baseUrl": "https://x", "apiKey": "!echo hi"}}).to_string(),
        );
        assert_eq!(
            configured_request_auth_status(command.get_provider("cov-st"), None)
                .expect("a command reports")
                .source,
            Some(AuthStatusSource::ModelsJsonCommand)
        );

        assert_eq!(configured_request_auth_status(None, None), None);
    }

    /// The request-config and configured-header surfaces merge the layers'
    /// headers and report the bearer form.
    #[test]
    fn request_configs_merge_the_configured_headers() {
        let config = config_from(
            &json!({"cov-h": {"baseUrl": "https://x", "headers": {"x-layer": "layer"},
                "modelOverrides": {"m": {"headers": {"x-override": "override"}}}}})
            .to_string(),
        );
        let provider_config = config.get_provider("cov-h").cloned();
        let extension = ProviderConfigInput {
            headers: Some(BTreeMap::from([("x-ext".to_owned(), "ext".to_owned())])),
            auth_header: Some(true),
            ..ProviderConfigInput::default()
        };
        let mut with_model_headers = model("cov-h", "m");
        with_model_headers.headers =
            Some(BTreeMap::from([("x-model".to_owned(), "model".to_owned())]));

        let request = resolve_compatibility_request_config(
            &with_model_headers,
            provider_config.as_ref(),
            Some(&extension),
        )
        .expect("the headers resolve");
        assert!(request.auth_header);
        let headers = request.headers.expect("the merged headers ride");
        for (name, value) in [
            ("x-layer", "layer"),
            ("x-ext", "ext"),
            ("x-override", "override"),
            ("x-model", "model"),
        ] {
            assert_eq!(
                headers.get(name).map(|entry| entry.as_deref()),
                Some(Some(value)),
                "{name} must merge"
            );
        }

        let bare = resolve_compatibility_request_config(&model("cov-h", "other"), None, None)
            .expect("the empty layers resolve");
        assert_eq!(bare.headers, None);
        assert!(!bare.auth_header);

        let configured = resolve_configured_model_headers(
            &with_model_headers,
            provider_config.as_ref(),
            Some(&extension),
            None,
        )
        .expect("the headers resolve");
        assert!(
            configured
                .expect("the override headers exist")
                .contains_key("x-override")
        );

        let unset_extension = ProviderConfigInput {
            models: Some(vec![ProviderModelInput {
                id: "other".to_owned(),
                name: "other".to_owned(),
                api: Some(pi_ai::types::Api::from("openai-completions")),
                base_url: Some("https://x".to_owned()),
                reasoning: false,
                thinking_level_map: None,
                input: vec![pi_ai::types::Modality::Text],
                cost: common::model_layer::zero_cost(),
                context_window: 10,
                max_tokens: 5,
                sampling_params: None,
                headers: Some(BTreeMap::from([(
                    "x-unset".to_owned(),
                    "$PI_RUST_COV_UNSET_VAR".to_owned(),
                )])),
                compat: None,
            }]),
            ..ProviderConfigInput::default()
        };
        let failure = resolve_configured_model_headers(
            &model("cov-h", "other"),
            None,
            Some(&unset_extension),
            None,
        )
        .expect_err("an unresolvable header fails");
        assert!(failure.contains("x-unset"));
    }

    /// The per-model overrides and the compat merge apply field by field,
    /// including the one-level-deeper nested merge upstream's mergeCompat
    /// special-cases.
    #[expect(
        clippy::float_cmp,
        reason = "the fixture costs are exact decimal literals the merge copies verbatim"
    )]
    #[test]
    fn model_overrides_and_nested_compat_merge_apply() {
        let config = config_from(
            r#"{"cov-ov": {
                "baseUrl": "https://ov.test/v1", "apiKey": "lit", "api": "openai-completions",
                "compat": {"openRouterRouting": {"order": ["a"], "allow_fallbacks": true},
                           "chatTemplateKwargs": {"thinking.enabled": {"$var": "thinking.enabled", "omitWhenOff": true}}},
                "models": [{"id": "m", "name": "Base", "reasoning": false,
                    "input": ["text"], "cost": {"input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4},
                    "contextWindow": 10, "maxTokens": 5,
                    "compat": {"openRouterRouting": {"ignore": ["b"], "allow_fallbacks": false}}}],
                "modelOverrides": {"m": {"name": "Overridden", "reasoning": true,
                    "input": ["text", "image"],
                    "cost": {"input": 9, "tiers": [{"inputTokensAbove": 1000, "input": 1, "output": 1, "cacheRead": 1, "cacheWrite": 1}]},
                    "contextWindow": 20, "maxTokens": 8,
                    "samplingParams": {"temperature": 0.5},
                    "compat": {"openRouterRouting": {"only": ["c"]}}}}}}"#,
        );
        let provider = composed("cov-ov", Some(&config), None).expect("the provider composes");
        let models = provider.get_models().expect("the models list");
        assert_eq!(models.len(), 1);
        let overridden = &models[0];
        assert_eq!(overridden.name, "Overridden");
        assert!(overridden.reasoning);
        assert_eq!(
            overridden.input,
            vec![pi_ai::types::Modality::Text, pi_ai::types::Modality::Image]
        );
        assert_eq!(overridden.cost.rates.input, 9.0);
        assert_eq!(
            overridden.cost.rates.output, 2.0,
            "unset rates keep the base"
        );
        assert!(overridden.cost.tiers.is_some());
        assert_eq!(overridden.context_window, 20);
        assert_eq!(overridden.max_tokens, 8);
        assert_eq!(
            overridden
                .sampling_params
                .clone()
                .expect("the params merge")
                .get("temperature"),
            Some(&json!(0.5)),
        );

        let compat = overridden.compat.clone().expect("the compat merges");
        let rendered = serde_json::to_value(&compat).expect("the compat serializes");
        let routing = &rendered["openRouterRouting"];
        assert_eq!(routing["order"], json!(["a"]));
        assert_eq!(routing["only"], json!(["c"]));
        assert_eq!(routing["ignore"], json!(["b"]));
        assert_eq!(routing["allow_fallbacks"], json!(false));
        assert_eq!(
            rendered["chatTemplateKwargs"]["thinking.enabled"]["$var"],
            json!("thinking.enabled"),
            "the nested kwarg object merges one level deeper"
        );
    }

    /// The trait-level refresh flow drives the composed state: the
    /// refreshed-list publication, the cancelled-refresh bail, and the OAuth
    /// credential projection.
    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the arms under test share one fixture; splitting them would re-state it"
    )]
    async fn refresh_publications_drive_the_composed_state() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let extension = ProviderConfigInput {
            name: Some("Cov Refresh".to_owned()),
            api_key: Some("sk-ext".to_owned()),
            api: Some(pi_ai::types::Api::from("openai-completions")),
            models: Some(vec![ProviderModelInput {
                id: "first".to_owned(),
                name: "First".to_owned(),
                api: None,
                base_url: Some("https://ext.test/v1".to_owned()),
                reasoning: false,
                thinking_level_map: None,
                input: vec![pi_ai::types::Modality::Text],
                cost: common::model_layer::zero_cost(),
                context_window: 10,
                max_tokens: 5,
                sampling_params: None,
                headers: None,
                compat: None,
            }]),
            ..ProviderConfigInput::default()
        };
        let provider = composed("cov-rf", None, Some(extension)).expect("the provider composes");

        // No refresh phase: the presence check declines and refresh is a no-op.
        provider
            .refresh_models(refresh_context(
                recording_publish(&Arc::new(Mutex::new(Vec::new()))),
                None,
                false,
            ))
            .await
            .expect("the no-op refresh lands");
        assert_eq!(ids(&provider), ["first"]);

        // A refresh closure over a base validates the refreshed list against
        // the base's catalog before publishing.
        let based_extension = ProviderConfigInput {
            name: Some("Cov Based".to_owned()),
            api_key: Some("sk-ext".to_owned()),
            models: Some(vec![ProviderModelInput {
                id: "base-model".to_owned(),
                name: "Rebased".to_owned(),
                api: None,
                base_url: Some("https://ext.test/v1".to_owned()),
                reasoning: false,
                thinking_level_map: None,
                input: vec![pi_ai::types::Modality::Text],
                cost: common::model_layer::zero_cost(),
                context_window: 10,
                max_tokens: 5,
                sampling_params: None,
                headers: None,
                compat: None,
            }]),
            refresh_models: Some(Arc::new(|_context| {
                Box::pin(async { Ok(vec![model("cov-rf3", "based-refresh")]) })
            })),
            ..ProviderConfigInput::default()
        };
        let based = compose_model_provider(
            "cov-rf3",
            Some(Arc::new(RecordingBase::healthy(&log))),
            &ModelConfig::default(),
            Some(based_extension),
        )
        .expect("the based provider composes");
        based
            .refresh_models(refresh_context(
                recording_publish(&Arc::new(Mutex::new(Vec::new()))),
                None,
                false,
            ))
            .await
            .expect("the based refresh lands");
        assert_eq!(ids(&based), ["based-refresh"]);

        // A refresh closure publishes the refreshed list into the shared state.
        let refreshed_extension = ProviderConfigInput {
            name: Some("Cov Refresh".to_owned()),
            api_key: Some("sk-ext".to_owned()),
            api: Some(pi_ai::types::Api::from("openai-completions")),
            models: Some(vec![ProviderModelInput {
                id: "first".to_owned(),
                name: "First".to_owned(),
                api: None,
                base_url: Some("https://ext.test/v1".to_owned()),
                reasoning: false,
                thinking_level_map: None,
                input: vec![pi_ai::types::Modality::Text],
                cost: common::model_layer::zero_cost(),
                context_window: 10,
                max_tokens: 5,
                sampling_params: None,
                headers: None,
                compat: None,
            }]),
            refresh_models: Some(Arc::new(|_context| {
                Box::pin(async { Ok(vec![model("cov-rf", "refreshed"), model("cov-rf", "extra")]) })
            })),
            ..ProviderConfigInput::default()
        };
        let sink: Arc<Mutex<Vec<ModelsStoreEntry>>> = Arc::new(Mutex::new(Vec::new()));
        let dynamic = composed("cov-rf2", None, Some(refreshed_extension))
            .expect("the refreshing provider composes");
        dynamic
            .refresh_models(refresh_context(recording_publish(&sink), None, false))
            .await
            .expect("the refresh lands");
        assert_eq!(ids(&dynamic), ["refreshed", "extra"]);
        assert!(written(&sink).is_empty(), "the refresh omits persistence");

        // A cancelled signal after the closure returns skips the publication.
        let cancelling = ProviderConfigInput {
            name: Some("Cov Cancel".to_owned()),
            api_key: Some("sk-ext".to_owned()),
            api: Some(pi_ai::types::Api::from("openai-completions")),
            models: Some(vec![ProviderModelInput {
                id: "first".to_owned(),
                name: "First".to_owned(),
                api: None,
                base_url: Some("https://ext.test/v1".to_owned()),
                reasoning: false,
                thinking_level_map: None,
                input: vec![pi_ai::types::Modality::Text],
                cost: common::model_layer::zero_cost(),
                context_window: 10,
                max_tokens: 5,
                sampling_params: None,
                headers: None,
                compat: None,
            }]),
            refresh_models: Some(Arc::new(|context: &RefreshModelsContext| {
                let signal = context.signal.clone();
                Box::pin(async move {
                    signal.cancel();
                    Ok(vec![model("cov-cx", "cancelled")])
                })
            })),
            ..ProviderConfigInput::default()
        };
        let cancelled = composed("cov-cx", None, Some(cancelling)).expect("the provider composes");
        cancelled
            .refresh_models(refresh_context(recording_publish(&sink), None, false))
            .await
            .expect("the cancelled refresh lands");
        assert_eq!(
            ids(&cancelled),
            ["first"],
            "the cancelled refresh publishes nothing"
        );

        // An OAuth credential rides the publication and projects the models.
        let projecting = ProviderConfigInput {
            name: Some("Cov Oauth".to_owned()),
            api: Some(pi_ai::types::Api::from("openai-completions")),
            models: Some(vec![ProviderModelInput {
                id: "kept-one".to_owned(),
                name: "Kept".to_owned(),
                api: None,
                base_url: Some("https://ext.test/v1".to_owned()),
                reasoning: false,
                thinking_level_map: None,
                input: vec![pi_ai::types::Modality::Text],
                cost: common::model_layer::zero_cost(),
                context_window: 10,
                max_tokens: 5,
                sampling_params: None,
                headers: None,
                compat: None,
            }]),
            oauth: Some(ExtensionOAuthConfig {
                name: "Cov OAuth".to_owned(),
                is_subscription: Some(true),
                uses_callback_server: None,
                login: Arc::new(|_callbacks| Box::pin(async { Err(unused_failure()) })),
                refresh_token: Arc::new(|_credentials, _signal| {
                    Box::pin(async { Err(unused_failure()) })
                }),
                get_api_key: Arc::new(|_credentials| "oauth-key".to_owned()),
                modify_models: Some(Arc::new(
                    |models: Vec<Model>, _credentials: &OAuthCredentials| {
                        models
                            .into_iter()
                            .filter(|entry| entry.id.starts_with("kept-"))
                            .collect()
                    },
                )),
            }),
            ..ProviderConfigInput::default()
        };
        let oauth_provider =
            composed("cov-oj", None, Some(projecting)).expect("the projecting provider composes");
        oauth_provider
            .refresh_models(refresh_context(
                recording_publish(&sink),
                Some(Credential::OAuth(OAuthCredentials {
                    refresh: "r".to_owned(),
                    access: "a".to_owned(),
                    expires: i64::MAX,
                    extra: BTreeMap::new(),
                })),
                false,
            ))
            .await
            .expect("the oauth refresh lands");
        assert_eq!(ids(&oauth_provider), ["kept-one"]);
        drop(log);
    }

    /// The legacy OAuth bridge walks every callback channel the extension
    /// contract spells: notify, prompt, manual code, and select; the derived
    /// request auth merges the configured headers and bearer form.
    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the arms under test share one fixture; splitting them would re-state it"
    )]
    async fn the_oauth_bridge_walks_the_legacy_callbacks() {
        let seen: Log = Arc::new(Mutex::new(Vec::new()));
        let login: ExtensionOAuthLoginFn = Arc::new({
            let seen = Arc::clone(&seen);
            move |callbacks: OAuthLoginCallbacks| {
                let seen = Arc::clone(&seen);
                Box::pin(async move {
                    (callbacks.notify)(AuthEvent::Info {
                        message: "cov login".to_owned(),
                        links: None,
                    });
                    let text = (callbacks.prompt)(OAuthPrompt {
                        message: "Enter the code".to_owned(),
                        placeholder: None,
                        allow_empty: None,
                    })
                    .await?;
                    let code = (callbacks.manual_code_input)().await?;
                    let picked = (callbacks.select)(OAuthSelectPrompt {
                        message: "Pick one".to_owned(),
                        options: vec![OAuthSelectOption {
                            id: "opt-a".to_owned(),
                            label: "Option A".to_owned(),
                        }],
                    })
                    .await?;
                    record(&seen, format!("picked {picked:?}"));
                    assert_eq!(text, "typed");
                    assert_eq!(code, "pasted");
                    Ok(OAuthCredentials {
                        refresh: "refresh-token".to_owned(),
                        access: "access-token".to_owned(),
                        expires: i64::MAX,
                        extra: BTreeMap::from([("env".to_owned(), json!({"COV_OAUTH": "1"}))]),
                    })
                })
            }
        });
        let refresh: ExtensionOAuthRefreshFn = Arc::new(|credentials, _signal| {
            Box::pin(async move {
                Ok(OAuthCredentials {
                    access: "refreshed-token".to_owned(),
                    ..credentials
                })
            })
        });
        let extension = ProviderConfigInput {
            name: Some("Cov OAuth".to_owned()),
            auth_header: Some(true),
            headers: Some(BTreeMap::from([(
                "x-oauth".to_owned(),
                "$COV_OAUTH".to_owned(),
            )])),
            oauth: Some(ExtensionOAuthConfig {
                name: "Cov OAuth".to_owned(),
                is_subscription: Some(true),
                uses_callback_server: None,
                login,
                refresh_token: refresh,
                get_api_key: Arc::new(|credentials: &OAuthCredentials| credentials.access.clone()),
                modify_models: None,
            }),
            ..ProviderConfigInput::default()
        };
        let provider =
            composed("cov-ob", None, Some(extension)).expect("the oauth provider composes");
        let oauth = provider
            .auth()
            .oauth
            .clone()
            .expect("the oauth method composes");
        assert_eq!(oauth.is_subscription, Some(true));

        let kinds: Log = Arc::new(Mutex::new(Vec::new()));
        let kinds_for_prompt = Arc::clone(&kinds);
        let credential = (oauth.login)(pi_ai::auth::types::ProviderAuthInteraction {
            signal: CancellationToken::new(),
            prompt: Arc::new(move |prompt: AuthPrompt| {
                let kinds = Arc::clone(&kinds_for_prompt);
                Box::pin(async move {
                    let answer = match &prompt.kind {
                        AuthPromptKind::Text { .. } => "typed".to_owned(),
                        AuthPromptKind::ManualCode { .. } => "pasted".to_owned(),
                        AuthPromptKind::Select { .. } => "opt-a".to_owned(),
                        AuthPromptKind::Secret { .. } => "secret".to_owned(),
                    };
                    record(
                        &kinds,
                        format!("{:?}", prompt.kind)
                            .split('(')
                            .next()
                            .map_or_else(String::new, str::to_owned),
                    );
                    Ok(answer)
                })
            }),
            notify: Arc::new(|_event: AuthEvent| {}),
        })
        .await
        .expect("the bridged login runs");
        assert_eq!(credential.access, "access-token");
        let walked = recorded(&kinds);
        assert_eq!(
            walked.len(),
            3,
            "text, manual code, and select rode the bridge"
        );

        let refreshed = (oauth.refresh)(credential.clone(), CancellationToken::new())
            .await
            .expect("the bridged refresh runs");
        assert_eq!(refreshed.access, "refreshed-token");

        let auth = (oauth.to_auth)(refreshed)
            .await
            .expect("the derivation runs");
        assert_eq!(auth.api_key.as_deref(), Some("refreshed-token"));
        let headers = auth.headers.expect("the merged headers ride");
        assert_eq!(
            headers.get("Authorization").map(|value| value.as_deref()),
            Some(Some("Bearer refreshed-token")),
        );
        assert_eq!(
            headers.get("x-oauth").map(|value| value.as_deref()),
            Some(Some("1")),
            "the credential env resolves the configured header template"
        );
    }
}

mod composed_streaming {
    use super::*;

    /// The extension dispatch routes stream and streamSimple through the
    /// registered closure, carrying the request options across.
    #[tokio::test]
    async fn extension_dispatch_streams_through_the_registered_closure() {
        let runtime = cov_runtime().await;
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        runtime
            .register_provider(
                "cov-stream",
                ProviderConfigInput {
                    name: Some("Cov Stream".to_owned()),
                    api_key: Some("sk-ext".to_owned()),
                    api: Some(pi_ai::types::Api::from("openai-completions")),
                    stream_simple: Some(recording_stream_simple(&log, "ext")),
                    models: Some(vec![ProviderModelInput {
                        id: "ext-model".to_owned(),
                        name: "Ext Model".to_owned(),
                        api: None,
                        base_url: Some("https://ext.test/v1".to_owned()),
                        reasoning: false,
                        thinking_level_map: None,
                        input: vec![pi_ai::types::Modality::Text],
                        cost: common::model_layer::zero_cost(),
                        context_window: 10,
                        max_tokens: 5,
                        sampling_params: None,
                        headers: None,
                        compat: None,
                    }]),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");
        let provider = runtime
            .get_provider("cov-stream")
            .expect("the composed provider registers");

        let stream_model = provider
            .get_models()
            .expect("the models list")
            .into_iter()
            .next()
            .expect("the extension model lists");
        let simple = provider
            .stream_simple(
                &stream_model,
                &empty_context(),
                Some(&SimpleStreamOptions {
                    api_key: Some("simple-key".to_owned()),
                    ..SimpleStreamOptions::default()
                }),
            )
            .result()
            .await;
        assert_eq!(simple.stop_reason, StopReason::Error);
        assert_eq!(simple.error_message.as_deref(), Some("ext"));
        assert_eq!(
            recorded(&log),
            ["ext ext-model api_key=Some(\"simple-key\")"]
        );

        let full = provider
            .stream(
                &stream_model,
                &empty_context(),
                Some(&StreamOptions {
                    api_key: Some("full-key".to_owned()),
                    ..StreamOptions::default()
                }),
            )
            .result()
            .await;
        assert_eq!(full.error_message.as_deref(), Some("ext"));
        assert_eq!(
            recorded(&log)[1],
            "ext ext-model api_key=Some(\"full-key\")",
            "the full stream arms the same closure with the carried options"
        );
    }

    /// The api-registry dispatch streams the openai-completions adapter over
    /// the injected client, and an unregistered api fails the dispatch.
    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the arms under test share one fixture; splitting them would re-state it"
    )]
    async fn api_dispatch_streams_the_registry_adapter() {
        let runtime = cov_runtime().await;
        runtime
            .register_provider(
                "cov-api",
                ProviderConfigInput {
                    api_key: Some("sk-api".to_owned()),
                    headers: Some(BTreeMap::from([("x-ext".to_owned(), "v".to_owned())])),
                    models: Some(vec![ProviderModelInput {
                        id: "api-model".to_owned(),
                        name: "Api Model".to_owned(),
                        api: Some(pi_ai::types::Api::from("openai-completions")),
                        base_url: Some("https://api.test/v1".to_owned()),
                        reasoning: false,
                        thinking_level_map: None,
                        input: vec![pi_ai::types::Modality::Text],
                        cost: common::model_layer::zero_cost(),
                        context_window: 10,
                        max_tokens: 5,
                        sampling_params: None,
                        headers: Some(BTreeMap::from([("x-model".to_owned(), "m".to_owned())])),
                        compat: None,
                    }]),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");
        let provider = runtime
            .get_provider("cov-api")
            .expect("the composed provider registers");
        let stream_model = provider
            .get_models()
            .expect("the models list")
            .into_iter()
            .next()
            .expect("the api model lists");

        let mock = MockHttpClient::new();
        mock.on(|request| request.url.contains("/chat/completions"))
            .respond(
                MockResponse::status(200)
                    .with_header("content-type", "text/event-stream")
                    .with_body("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n"),
            );
        let message = provider
            .stream_simple(
                &stream_model,
                &empty_context(),
                Some(&SimpleStreamOptions {
                    api_key: Some("sk-direct".to_owned()),
                    headers: Some(BTreeMap::from([(
                        "x-request".to_owned(),
                        Some("r".to_owned()),
                    )])),
                    transport_options: transport(&mock),
                    ..SimpleStreamOptions::default()
                }),
            )
            .result()
            .await;
        assert_eq!(message.stop_reason, StopReason::Stop, "got {message:?}");
        let recorded_request = &mock.recorded()[0];
        assert!(recorded_request.url.starts_with("https://api.test/v1"));

        let unknown = ProviderModelInput {
            id: "odd-model".to_owned(),
            name: "Odd".to_owned(),
            api: Some(pi_ai::types::Api::from("cov-no-such-api")),
            base_url: Some("https://api.test/v1".to_owned()),
            reasoning: false,
            thinking_level_map: None,
            input: vec![pi_ai::types::Modality::Text],
            cost: common::model_layer::zero_cost(),
            context_window: 10,
            max_tokens: 5,
            sampling_params: None,
            headers: None,
            compat: None,
        };
        runtime
            .register_provider(
                "cov-odd",
                ProviderConfigInput {
                    api_key: Some("sk-api".to_owned()),
                    models: Some(vec![unknown]),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");
        let odd_provider = runtime
            .get_provider("cov-odd")
            .expect("the odd provider registers");
        let odd_model = odd_provider
            .get_models()
            .expect("the odd models list")
            .into_iter()
            .next()
            .expect("the odd model lists");
        let failed = odd_provider
            .stream_simple(&odd_model, &empty_context(), None)
            .result()
            .await;
        assert_eq!(failed.stop_reason, StopReason::Error);
        assert!(
            failed
                .error_message
                .as_deref()
                .is_some_and(|message| message
                    .contains("No API provider registered for api: cov-no-such-api")),
            "got {:?}",
            failed.error_message,
        );
    }

    /// The base dispatch streams the composed base provider when the base
    /// serves the model's api, and falls through otherwise; a failing base
    /// catalog fails the composition itself.
    #[tokio::test]
    async fn base_dispatch_streams_the_base_provider() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let extension = ProviderConfigInput {
            api_key: Some("sk-ext".to_owned()),
            ..ProviderConfigInput::default()
        };
        let provider = compose_model_provider(
            "cov-base",
            Some(Arc::new(RecordingBase::healthy(&log))),
            &ModelConfig::default(),
            Some(extension),
        )
        .expect("the composed provider builds");
        assert_eq!(ids(&provider), ["base-model"]);

        let stream_model = model("cov-base", "base-model");
        let message = provider
            .stream_simple(
                &stream_model,
                &empty_context(),
                Some(&SimpleStreamOptions {
                    api_key: Some("sk-simple".to_owned()),
                    ..SimpleStreamOptions::default()
                }),
            )
            .result()
            .await;
        assert_eq!(message.error_message.as_deref(), Some("base-simple"));
        assert_eq!(
            recorded(&log),
            ["base-simple base-model api_key=Some(\"sk-simple\")"]
        );

        let odd = model("cov-base", "odd-model");
        let mut odd = odd;
        odd.api = pi_ai::types::Api::from("cov-no-such-api");
        let failed = provider
            .stream_simple(&odd, &empty_context(), None)
            .result()
            .await;
        assert!(
            failed
                .error_message
                .as_deref()
                .is_some_and(|message| message.contains("No API provider registered")),
            "the unsupported api falls through to the registry: {:?}",
            failed.error_message,
        );

        let failing = composition_error(compose_model_provider(
            "cov-failing",
            Some(Arc::new(RecordingBase::failing(&log))),
            &ModelConfig::default(),
            None,
        ));
        assert!(failing.contains("base catalog failed"));
    }

    /// The runtime error surface joins the models.json load error and the
    /// per-provider composition errors, and clears when none remain.
    #[tokio::test]
    async fn the_runtime_error_surface_reports_composition_failures() {
        let credentials: Arc<dyn CredentialStore> = empty_auth_storage();
        let dir = tempfile::tempdir().expect("the temp dir creates");
        let schema_path = dir.path().join("schema.json").display().to_string();
        std::fs::write(
            &schema_path,
            json!({"providers": {"cov": {"apiKey": ""}}}).to_string(),
        )
        .expect("the broken models.json writes");
        let broken = ModelRuntime::create(CreateModelRuntimeOptions {
            credentials: Some(Arc::clone(&credentials)),
            models_path: Some(Some(schema_path)),
            models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::default())),
            allow_model_network: false,
            ..CreateModelRuntimeOptions::default()
        })
        .await
        .expect("the schema-invalid snapshot still constructs");
        assert!(
            broken
                .get_error()
                .is_some_and(|error| error.contains("Invalid models.json schema"))
        );

        let compose_path = dir.path().join("compose.json").display().to_string();
        std::fs::write(
            &compose_path,
            json!({"providers": {"cov-broken": {"oauth": "radius"}}}).to_string(),
        )
        .expect("the uncomposable models.json writes");
        let uncomposable = ModelRuntime::create(CreateModelRuntimeOptions {
            credentials: Some(Arc::clone(&credentials)),
            models_path: Some(Some(compose_path)),
            models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::default())),
            allow_model_network: false,
            ..CreateModelRuntimeOptions::default()
        })
        .await
        .expect("the runtime constructs");
        assert!(
            uncomposable
                .get_error()
                .is_some_and(|error| error.contains("Provider \"cov-broken\""))
        );
        assert!(
            uncomposable
                .set_runtime_api_key("cov-broken", "k".to_owned(), None)
                .await
                .expect_err("a broken composition fails the sync")
                .to_string()
                .contains("but local synchronization failed"),
        );

        let clean = cov_runtime().await;
        assert_eq!(clean.get_error(), None);
    }

    /// The auth-status surface reports each source: runtime keys, stored
    /// credentials, extension fallbacks, and the unconfigured default.
    #[tokio::test]
    async fn provider_auth_status_reports_each_source() {
        let runtime = cov_runtime().await;
        runtime
            .register_provider(
                "cov-status",
                ProviderConfigInput {
                    api_key: Some("sk-status".to_owned()),
                    api: Some(pi_ai::types::Api::from("openai-completions")),
                    models: Some(vec![ProviderModelInput {
                        id: "status-model".to_owned(),
                        name: "Status".to_owned(),
                        api: None,
                        base_url: Some("https://status.test/v1".to_owned()),
                        reasoning: false,
                        thinking_level_map: None,
                        input: vec![pi_ai::types::Modality::Text],
                        cost: common::model_layer::zero_cost(),
                        context_window: 10,
                        max_tokens: 5,
                        sampling_params: None,
                        headers: None,
                        compat: None,
                    }]),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");
        let status = runtime.get_provider_auth_status("cov-status");
        assert_eq!(status.source, Some(AuthStatusSource::Fallback));
        assert!(status.configured);

        runtime
            .set_runtime_api_key("cov-status", "sk-runtime".to_owned(), None)
            .await
            .expect("the runtime key commits");
        assert_eq!(
            runtime.get_provider_auth_status("cov-status").source,
            Some(AuthStatusSource::Runtime)
        );
        runtime
            .remove_runtime_api_key("cov-status", None)
            .await
            .expect("the runtime key removes");
        assert_eq!(
            runtime.get_provider_auth_status("cov-status").source,
            Some(AuthStatusSource::Fallback),
            "removal falls back to the configured layer"
        );

        let unset = ProviderConfigInput {
            api_key: Some("$PI_RUST_COV_UNSET_VAR".to_owned()),
            ..ProviderConfigInput::default()
        };
        runtime
            .register_provider("cov-unset", unset)
            .expect("the unset registration validates");
        let status = runtime.get_provider_auth_status("cov-unset");
        assert!(!status.configured);
        assert_eq!(status.source, None);

        let absent = runtime.get_provider_auth_status("cov-never-registered");
        assert!(!absent.configured);
        assert_eq!(absent.source, None);
    }

    /// Auth resolution carries the request overrides and merges the
    /// configured headers; unresolvable headers fail the resolution.
    #[tokio::test]
    async fn auth_resolution_merges_the_configured_headers() {
        let runtime = cov_runtime().await;
        runtime
            .register_provider(
                "cov-auth",
                ProviderConfigInput {
                    api_key: Some("sk-configured".to_owned()),
                    headers: Some(BTreeMap::from([(
                        "x-configured".to_owned(),
                        "v".to_owned(),
                    )])),
                    api: Some(pi_ai::types::Api::from("openai-completions")),
                    models: Some(vec![ProviderModelInput {
                        id: "auth-model".to_owned(),
                        name: "Auth".to_owned(),
                        api: None,
                        base_url: Some("https://auth.test/v1".to_owned()),
                        reasoning: false,
                        thinking_level_map: None,
                        input: vec![pi_ai::types::Modality::Text],
                        cost: common::model_layer::zero_cost(),
                        context_window: 10,
                        max_tokens: 5,
                        sampling_params: None,
                        headers: None,
                        compat: None,
                    }]),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");

        let resolved = runtime
            .get_auth(
                "cov-auth",
                Some(&ModelRuntimeAuthOverrides {
                    api_key: Some("sk-forced".to_owned()),
                    env: Some(BTreeMap::from([("COV_ENV".to_owned(), "1".to_owned())])),
                    min_oauth_validity_ms: Some(1_000),
                    signal: Some(CancellationToken::new()),
                }),
            )
            .await
            .expect("the resolution runs")
            .expect("the configured provider resolves");
        assert_eq!(resolved.auth.api_key.as_deref(), Some("sk-forced"));

        let stream_model = ModelRuntimeCore::get_models(&runtime, Some("cov-auth"))
            .into_iter()
            .next()
            .expect("the auth model lists");
        let merged = runtime
            .get_auth_for_model(&stream_model, None)
            .await
            .expect("the model resolution runs")
            .expect("the model resolves");
        assert_eq!(
            merged
                .auth
                .headers
                .expect("the configured headers merge")
                .get("x-configured")
                .map(|value| value.as_deref()),
            Some(Some("v")),
        );

        runtime
            .register_provider(
                "cov-badheaders",
                ProviderConfigInput {
                    api_key: Some("sk".to_owned()),
                    headers: Some(BTreeMap::from([(
                        "x-unset".to_owned(),
                        "$PI_RUST_COV_UNSET_VAR".to_owned(),
                    )])),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the bad-header registration validates");
        let bad_model = model("cov-badheaders", "m");
        let failure = runtime
            .get_auth_for_model(&bad_model, None)
            .await
            .expect_err("an unresolvable header fails the resolution");
        assert!(failure.to_string().contains("x-unset"), "got {failure:?}");
    }

    /// Availability passes update the snapshot per provider, and the
    /// registered-config surfaces report what the extensions carry.
    #[tokio::test]
    async fn availability_passes_update_the_snapshot() {
        let runtime = cov_runtime().await;
        runtime
            .register_provider(
                "cov-avail",
                ProviderConfigInput {
                    api_key: Some("sk-avail".to_owned()),
                    api: Some(pi_ai::types::Api::from("openai-completions")),
                    models: Some(vec![ProviderModelInput {
                        id: "avail-model".to_owned(),
                        name: "Avail".to_owned(),
                        api: None,
                        base_url: Some("https://avail.test/v1".to_owned()),
                        reasoning: false,
                        thinking_level_map: None,
                        input: vec![pi_ai::types::Modality::Text],
                        cost: common::model_layer::zero_cost(),
                        context_window: 10,
                        max_tokens: 5,
                        sampling_params: None,
                        headers: None,
                        compat: None,
                    }]),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");

        let available = ModelRuntimeCore::get_available(&runtime, Some("cov-avail"), None)
            .await
            .expect("the provider-scoped pass runs");
        assert_eq!(
            available
                .into_iter()
                .map(|entry| entry.id)
                .collect::<Vec<_>>(),
            ["avail-model"]
        );

        let all = ModelRuntimeCore::get_available(&runtime, None, None)
            .await
            .expect("the queued pass runs");
        assert!(
            all.iter().any(|entry| entry.provider.0 == "cov-avail"),
            "the snapshot holds the provider's models"
        );
        assert!(
            runtime
                .get_available_snapshot()
                .iter()
                .any(|entry| entry.provider.0 == "cov-avail")
        );

        runtime
            .refresh(pi_ai::models::ModelsRefreshOptions {
                allow_network: Some(false),
                providers: Some(vec!["cov-avail".to_owned()]),
                ..pi_ai::models::ModelsRefreshOptions::default()
            })
            .await;
        assert!(runtime.has_configured_auth("cov-avail"));
        assert!(!runtime.has_configured_auth("cov-never"));

        let registered = runtime
            .get_registered_provider_config("cov-avail")
            .expect("the registration records its config");
        assert_eq!(registered.api_key.as_deref(), Some("sk-avail"));
        assert_eq!(runtime.get_registered_provider_ids(), ["cov-avail"]);
        assert!(
            runtime
                .get_registered_native_provider("cov-avail")
                .is_none()
        );

        runtime.unregister_provider("cov-avail");
        assert!(runtime.get_provider("cov-avail").is_none());
        assert!(
            runtime
                .get_registered_provider_config("cov-avail")
                .is_none()
        );
    }

    /// Re-registration merges the next config over the previous one,
    /// preserving values the next leaves undefined.
    #[tokio::test]
    async fn re_registration_merges_the_previous_config() {
        let runtime = cov_runtime().await;
        let first = ProviderConfigInput {
            name: Some("Cov First".to_owned()),
            api_key: Some("sk-first".to_owned()),
            base_url: Some("https://first.test/v1".to_owned()),
            auth_header: Some(true),
            ..ProviderConfigInput::default()
        };
        runtime
            .register_provider("cov-merge", first)
            .expect("the first registration validates");
        runtime
            .register_provider(
                "cov-merge",
                ProviderConfigInput {
                    api_key: Some("sk-second".to_owned()),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the second registration validates");
        let merged = runtime
            .get_registered_provider_config("cov-merge")
            .expect("the merged registration records");
        assert_eq!(
            merged.api_key.as_deref(),
            Some("sk-second"),
            "defined values win"
        );
        assert_eq!(
            merged.name.as_deref(),
            Some("Cov First"),
            "undefined values preserve"
        );
        assert_eq!(merged.base_url.as_deref(), Some("https://first.test/v1"));
        assert_eq!(merged.auth_header, Some(true), "the boolean preserves");
    }

    /// `create` runs the signal ladder: the merged caller-and-timeout token,
    /// the caller-only pass-through, the timeout-only token, and the
    /// refresh-skipping flag; the static catalogs survive every arm.
    #[tokio::test]
    async fn create_runs_the_signal_ladder() {
        let credentials: Arc<dyn CredentialStore> = empty_auth_storage();
        let base = CreateModelRuntimeOptions {
            credentials: Some(Arc::clone(&credentials)),
            models_path: Some(None),
            models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::default())),
            catalog_base_url: Some("http://127.0.0.1:9".to_owned()),
            ..CreateModelRuntimeOptions::default()
        };
        let merged = ModelRuntime::create(CreateModelRuntimeOptions {
            allow_model_network: true,
            model_refresh_timeout_ms: Some(25),
            signal: Some(CancellationToken::new()),
            ..base.clone()
        })
        .await
        .expect("the merged-signal create lands");
        assert!(!merged.get_models().is_empty());

        let caller_only = ModelRuntime::create(CreateModelRuntimeOptions {
            signal: Some(CancellationToken::new()),
            ..base.clone()
        })
        .await
        .expect("the caller-only create lands");
        assert!(!caller_only.get_models().is_empty());

        let timeout_only = ModelRuntime::create(CreateModelRuntimeOptions {
            allow_model_network: true,
            model_refresh_timeout_ms: Some(25),
            ..base.clone()
        })
        .await
        .expect("the timeout-only create lands");
        assert!(!timeout_only.get_models().is_empty());

        let skipped = ModelRuntime::create(CreateModelRuntimeOptions {
            refresh_on_create: Some(false),
            ..base
        })
        .await
        .expect("the refresh-skipping create lands");
        assert!(skipped.get_available_snapshot().is_empty());
    }
}

mod resolver_arms {
    use super::*;

    /// The scripted view the resolver tests drive: plain-field reads over
    /// one fixture model set, distinct from the Option-ladder fake the
    /// parity suite uses.
    struct ScriptedView {
        models: Vec<Model>,
        available: Vec<Model>,
        auth_providers: Vec<&'static str>,
    }

    impl ModelRuntimeView for ScriptedView {
        fn get_models(&self) -> Vec<Model> {
            self.models.clone()
        }

        fn get_model(&self, provider: &str, model_id: &str) -> Option<Model> {
            self.models
                .iter()
                .find(|entry| entry.provider.0 == provider && entry.id == model_id)
                .cloned()
        }

        fn has_configured_auth(&self, provider: &str) -> bool {
            self.auth_providers.contains(&provider)
        }

        fn get_available_snapshot(&self) -> Vec<Model> {
            self.available.clone()
        }

        fn get_available(
            &self,
            _options: Option<&pi_ai::auth::types::AuthOptions>,
        ) -> pi_ai::types::BoxedFuture<'_, Result<Vec<Model>, pi_ai::auth::resolve::ModelsFailure>>
        {
            Box::pin(std::future::ready(Ok(self.available.clone())))
        }
    }

    /// The fixture catalog: one authenticated provider with an alias and a
    /// dated model, one unauthenticated provider with a slash-shaped id, and
    /// a case-variant duplicate for the ambiguity arms.
    fn fixture_models() -> Vec<Model> {
        let mut models = vec![
            model("cov-auth", "alias-model"),
            model("cov-auth", "alias-model-20250929"),
            model("cov-auth", "alias-model-20240101"),
            model("cov-open", "slashed/id"),
        ];
        let mut twin = model("COV-AUTH", "twin-model");
        "Twin".clone_into(&mut twin.name);
        models.push(twin);
        models.push(model("cov-auth", "twin-model"));
        models
    }

    /// Pattern parsing walks the exact, fuzzy, suffix, and fallback arms.
    #[test]
    fn pattern_parsing_walks_the_match_ladder() {
        let models = fixture_models();

        let exact = parse_model_pattern("alias-model", &models, None);
        assert_eq!(
            exact.model.expect("the exact match lands").id,
            "alias-model"
        );

        let alias = parse_model_pattern("ALIAS", &models, None);
        assert_eq!(
            alias.model.expect("the alias preference lands").id,
            "alias-model",
            "aliases sort above the dated sibling"
        );
        let dated = parse_model_pattern("alias-model-2", &models, None);
        assert_eq!(
            dated.model.expect("the dated preference lands").id,
            "alias-model-20250929",
            "without aliases the highest dated id wins"
        );
        let two_dated = parse_model_pattern("alias-model-20", &models, None);
        assert_eq!(
            two_dated.model.expect("the dated pair sorts").id,
            "alias-model-20250929",
            "the dated comparator sorts by id descending"
        );

        let level = parse_model_pattern("alias-model:high", &models, None);
        assert_eq!(level.model.expect("the suffix strips").id, "alias-model");
        assert_eq!(level.thinking_level, Some(ThinkingLevel::High));

        let invalid = parse_model_pattern("alias-model:warp", &models, None);
        assert_eq!(
            invalid.model.expect("the fallback still matches").id,
            "alias-model"
        );
        assert!(
            invalid
                .warning
                .as_deref()
                .is_some_and(|warning| warning.contains("Invalid thinking level \"warp\"")),
        );

        let strict = parse_model_pattern(
            "alias-model:warp",
            &models,
            Some(ParseModelPatternOptions {
                allow_invalid_thinking_level_fallback: false,
            }),
        );
        assert!(
            strict.model.is_none(),
            "strict mode keeps the suffix in the id"
        );

        let nested = parse_model_pattern("alias-model:medium:high", &models, None);
        assert_eq!(
            nested.model.expect("the nested suffix strips").id,
            "alias-model",
        );
        assert_eq!(
            nested.thinking_level,
            Some(ThinkingLevel::High),
            "the outermost valid suffix carries"
        );

        let bare = parse_model_pattern("nothing-matches", &models, None);
        assert!(bare.model.is_none());
        assert!(bare.warning.is_none());
    }

    /// Exact-reference matching distinguishes the canonical, bare, and
    /// ambiguous shapes, case-insensitively.
    #[test]
    fn exact_reference_matching_distinguishes_ambiguity() {
        let models = fixture_models();

        assert!(find_exact_model_reference_match("   ", &models).is_none());
        assert_eq!(
            find_exact_model_reference_match("COV-AUTH/ALIAS-MODEL", &models)
                .expect("the canonical reference matches case-insensitively")
                .id,
            "alias-model",
        );
        assert_eq!(
            find_exact_model_reference_match("alias-model", &models)
                .expect("the unique bare id matches")
                .id,
            "alias-model",
        );
        assert!(
            find_exact_model_reference_match("twin-model", &models).is_none(),
            "the bare id is ambiguous across the case-variant providers"
        );
        assert!(
            find_exact_model_reference_match("cov-open/slashed/id", &models).is_some(),
            "the slash-shaped id resolves through the provider split"
        );
        assert!(
            find_exact_model_reference_match("twin", &models).is_none(),
            "the partial id is not an exact reference"
        );
    }

    /// CLI resolution falls back to custom ids over the provider's catalog,
    /// reports the ambiguity hints, and errors on empty catalogs.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the arms under test share one fixture; splitting them would re-state it"
    )]
    fn cli_resolution_falls_back_to_custom_ids() {
        let view = ScriptedView {
            models: fixture_models(),
            available: Vec::new(),
            auth_providers: vec!["cov-auth"],
        };

        let fallback = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: Some("cov-auth"),
            cli_model: Some("my-custom-id:high"),
            cli_thinking: None,
            model_runtime: &view,
        });
        let resolved = fallback.model.expect("the custom id falls back");
        assert_eq!(resolved.id, "my-custom-id");
        assert!(resolved.reasoning, "the thinking suffix marks reasoning");
        assert_eq!(fallback.thinking_level, Some(ThinkingLevel::High));
        assert!(
            fallback
                .warning
                .as_deref()
                .is_some_and(|warning| warning.contains("not found for provider \"cov-auth\"")),
        );

        let first = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: Some("cov-open"),
            cli_model: Some("custom"),
            cli_thinking: None,
            model_runtime: &view,
        });
        assert_eq!(
            first.model.expect("the first model falls back").id,
            "custom",
            "without a default id the first catalog model serves as the base"
        );

        let unknown_provider = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: Some("cov-missing"),
            cli_model: Some("x"),
            cli_thinking: None,
            model_runtime: &view,
        });
        assert!(
            unknown_provider
                .error
                .as_deref()
                .is_some_and(|error| error.contains("Unknown provider \"cov-missing\"")),
        );

        let empty = ScriptedView {
            models: Vec::new(),
            available: Vec::new(),
            auth_providers: vec!["cov-auth"],
        };
        assert!(
            resolve_cli_model(ResolveCliModelOptions {
                cli_provider: None,
                cli_model: Some("x"),
                cli_thinking: None,
                model_runtime: &empty,
            })
            .error
            .as_deref()
            .is_some_and(|error| error.contains("No models available")),
        );

        let unauthenticated = ScriptedView {
            models: vec![model("cov-auth", "shared"), model("cov-open", "shared")],
            available: Vec::new(),
            auth_providers: vec![],
        };
        let ambiguous = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: None,
            cli_model: Some("shared"),
            cli_thinking: None,
            model_runtime: &unauthenticated,
        });
        assert!(
            ambiguous
                .error
                .as_deref()
                .is_some_and(|error| error.contains("No matching provider is authenticated")),
        );
        let authenticated_shared = ScriptedView {
            models: vec![model("cov-auth", "shared"), model("cov-open", "shared")],
            available: Vec::new(),
            auth_providers: vec!["cov-auth"],
        };
        assert_eq!(
            resolve_cli_model(ResolveCliModelOptions {
                cli_provider: None,
                cli_model: Some("shared"),
                cli_thinking: None,
                model_runtime: &authenticated_shared,
            })
            .model
            .expect("the sole authenticated provider resolves the ambiguity")
            .provider
            .0,
            "cov-auth",
        );

        let slash_shaped = ScriptedView {
            models: vec![
                model("cov-open", "cov-auth/inner"),
                model("cov-auth", "inner"),
            ],
            available: Vec::new(),
            auth_providers: vec!["cov-open"],
        };
        let inferred = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: None,
            cli_model: Some("cov-auth/inner"),
            cli_thinking: None,
            model_runtime: &slash_shaped,
        });
        assert_eq!(
            inferred
                .model
                .expect("the authenticated raw-id match serves")
                .provider
                .0,
            "cov-open",
            "the unauthenticated inference loses to the authenticated raw id"
        );

        // The provider/model spelling with --provider strips the prefix
        // before matching.
        let stripped = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: Some("cov-open"),
            cli_model: Some("cov-open/cov-auth/inner"),
            cli_thinking: None,
            model_runtime: &slash_shaped,
        });
        assert_eq!(
            stripped.model.expect("the prefix strips").id,
            "cov-auth/inner",
        );

        // The provider-less miss reports the raw input.
        let unmatched = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: None,
            cli_model: Some("never-matches-anything"),
            cli_thinking: None,
            model_runtime: &view,
        });
        assert!(
            unmatched
                .error
                .as_deref()
                .is_some_and(|error| error.contains("never-matches-anything")),
            "the provider-less resolution reports the raw input"
        );

        // The inferred provider misses, no raw id matches, and the fallback
        // parse misses too: the provider's own fallback builds the custom id.
        let inferred_miss = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: None,
            cli_model: Some("cov-auth/zzz-custom"),
            cli_thinking: None,
            model_runtime: &slash_shaped,
        });
        let built = inferred_miss.model.expect("the provider fallback builds");
        assert_eq!(built.provider.0, "cov-auth");
        assert_eq!(built.id, "zzz-custom");
        assert!(
            inferred_miss
                .warning
                .as_deref()
                .is_some_and(|warning| warning.contains("not found for provider")),
            "the fallback warns: {:?}",
            inferred_miss.warning,
        );
    }

    /// Scope resolution walks the glob ladder and reports the diagnostics the
    /// CLI prints.
    #[test]
    fn scope_resolution_walks_globs_and_diagnostics() {
        let models = fixture_models();
        let result = resolve_model_scope_from_models(
            &[
                "cov-auth/alias-*:high".to_owned(),
                "cov-auth/alias-model".to_owned(),
                "cov-auth/alias-model:high".to_owned(),
                "cov-auth/alias-model:warp".to_owned(),
                "cov-none/*".to_owned(),
                "slashed/id".to_owned(),
            ],
            &models,
        );
        assert_eq!(result.scoped_models.len(), 4, "duplicates skip");
        assert_eq!(
            result.scoped_models[0].thinking_level,
            Some(ThinkingLevel::High),
        );
        let codes: Vec<_> = result
            .diagnostics
            .iter()
            .map(|diagnostic| (diagnostic.code, diagnostic.pattern.clone()))
            .collect();
        assert!(codes.contains(&(
            pi_coding_agent::model_resolver::ModelScopeDiagnosticCode::InvalidThinkingLevel,
            "cov-auth/alias-model:warp".to_owned()
        )));
        assert!(codes.contains(&(
            pi_coding_agent::model_resolver::ModelScopeDiagnosticCode::NoMatch,
            "cov-none/*".to_owned()
        )));

        let warning = format_scope_warning(&result.diagnostics[0]);
        assert!(warning.starts_with("Warning: "));
    }

    /// The real runtime serves the resolver through both view impls, and the
    /// restore ladder substitutes the available fallback.
    #[tokio::test]
    async fn the_real_runtime_view_serves_the_resolver() {
        let credentials: Arc<dyn CredentialStore> = empty_auth_storage();
        let dir = tempfile::tempdir().expect("the temp dir creates");
        let models_path = dir.path().join("models.json").display().to_string();
        std::fs::write(
            &models_path,
            json!({"providers": {"cov-view": {
                "name": "Cov View", "baseUrl": "https://view.test/v1", "apiKey": "lit",
                "api": "openai-completions",
                "models": [{"id": "view-model", "contextWindow": 10, "maxTokens": 5}]}}})
            .to_string(),
        )
        .expect("the models.json writes");
        let registry = create_model_registry(credentials, Some(&models_path)).await;
        let runtime = registry.runtime().clone();

        ModelRuntimeCore::get_available(&runtime, None, None)
            .await
            .expect("the availability pass runs");

        let scoped =
            resolve_model_scope_with_diagnostics(&["view-model".to_owned()], &runtime, None).await;
        assert_eq!(scoped.scoped_models.len(), 1);
        assert!(scoped.diagnostics.is_empty());
        assert!(
            resolve_model_scope(&["cov-none/*".to_owned()], &runtime, None)
                .await
                .is_empty()
        );

        let arc_runtime = Arc::new(runtime.clone());
        let cli = resolve_cli_model(ResolveCliModelOptions {
            cli_provider: Some("cov-view"),
            cli_model: Some("view-model:high"),
            cli_thinking: None,
            model_runtime: &arc_runtime,
        });
        assert_eq!(cli.model.expect("the arc view resolves").id, "view-model");
        assert_eq!(cli.thinking_level, Some(ThinkingLevel::High));

        // The Arc-shared view serves the whole resolver surface: the scoped
        // resolution rides its availability read, the saved default its model
        // and auth reads, and the restore its model lookup.
        let arc_scoped =
            resolve_model_scope_with_diagnostics(&["view-model".to_owned()], &arc_runtime, None)
                .await;
        assert_eq!(arc_scoped.scoped_models.len(), 1);
        let arc_initial = find_initial_model(FindInitialModelOptions {
            cli_provider: None,
            cli_model: None,
            scoped_models: &[],
            is_continuing: false,
            default_provider: Some("cov-view"),
            default_model_id: Some("view-model"),
            default_thinking_level: None,
            model_thinking_levels: None,
            model_runtime: &arc_runtime,
        })
        .expect("the arc view serves the saved default");
        assert_eq!(
            arc_initial.model.expect("the model restores").id,
            "view-model"
        );
        let arc_restore =
            restore_model_from_session("cov-view", "view-model", None, false, &arc_runtime);
        assert_eq!(
            arc_restore.model.expect("the arc view restores").id,
            "view-model",
        );

        // With no CLI, scoped, or saved default, the arc view serves the
        // availability snapshot — a known-provider default when the ambient
        // environment configures one, else the first snapshot model.
        let arc_fallback = find_initial_model(FindInitialModelOptions {
            cli_provider: None,
            cli_model: None,
            scoped_models: &[],
            is_continuing: false,
            default_provider: None,
            default_model_id: None,
            default_thinking_level: None,
            model_thinking_levels: None,
            model_runtime: &arc_runtime,
        })
        .expect("the arc view falls back to the snapshot");
        assert!(
            arc_fallback.model.is_some(),
            "the availability snapshot serves a model"
        );

        let initial = find_initial_model(FindInitialModelOptions {
            cli_provider: None,
            cli_model: None,
            scoped_models: &[],
            is_continuing: false,
            default_provider: Some("cov-view"),
            default_model_id: Some("view-model"),
            default_thinking_level: Some(ThinkingLevel::Low),
            model_thinking_levels: Some(&BTreeMap::from([(
                "cov-view/view-model".to_owned(),
                ThinkingLevel::Max,
            )])),
            model_runtime: &runtime,
        })
        .expect("the saved default resolves");
        assert_eq!(initial.model.expect("the model restores").id, "view-model");
        assert_eq!(
            initial.thinking_level,
            ThinkingLevel::Max,
            "the per-model map wins over the saved default"
        );

        let restored = restore_model_from_session("cov-view", "view-model", None, false, &runtime);
        assert_eq!(
            restored.model.expect("the session model restores").id,
            "view-model"
        );
        assert!(restored.fallback_message.is_none());

        let current = model("cov-view", "view-model");
        let substituted =
            restore_model_from_session("cov-view", "gone-model", Some(&current), false, &runtime);
        assert_eq!(
            substituted.model.expect("the current model serves").id,
            "view-model"
        );
        assert!(
            substituted
                .fallback_message
                .as_deref()
                .is_some_and(|message| message.contains("model no longer exists")),
        );

        // A runtime whose availability pass never ran holds the model but no
        // configured auth: the restore reports the substitution reason and
        // serves the fallback chain.
        let cold_credentials: Arc<dyn CredentialStore> = empty_auth_storage();
        let cold = ModelRuntime::create(CreateModelRuntimeOptions {
            credentials: Some(Arc::clone(&cold_credentials)),
            models_path: Some(Some(models_path.clone())),
            models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::default())),
            allow_model_network: false,
            refresh_on_create: Some(false),
            ..CreateModelRuntimeOptions::default()
        })
        .await
        .expect("the cold runtime constructs");
        let unconfigured =
            restore_model_from_session("cov-view", "view-model", Some(&current), false, &cold);
        assert!(
            unconfigured
                .fallback_message
                .as_deref()
                .is_some_and(|message| message.contains("no auth configured")),
            "got {:?}",
            unconfigured.fallback_message,
        );

        let nothing = restore_model_from_session("cov-gone", "gone", None, false, &runtime);
        assert!(
            nothing
                .model
                .is_some_and(|model| model.provider.0 != "cov-gone"),
            "the available snapshot serves the fallback"
        );
        assert!(
            nothing
                .fallback_message
                .as_deref()
                .is_some_and(|message| message.contains("model no longer exists")),
        );
    }

    /// The options structs render and the initial-model ladder walks the
    /// scoped-models and CLI-error arms over the scripted view.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the arms under test share one fixture; splitting them would re-state it"
    )]
    fn the_initial_model_ladder_walks_scoped_and_error_arms() {
        let view = ScriptedView {
            models: fixture_models(),
            available: vec![model("cov-auth", "alias-model")],
            auth_providers: vec!["cov-auth"],
        };
        let options = format!(
            "{:?}",
            ResolveCliModelOptions {
                cli_provider: Some("cov-auth"),
                cli_model: Some("alias-model"),
                cli_thinking: Some(ThinkingLevel::Low),
                model_runtime: &view,
            }
        );
        assert!(options.contains("ResolveCliModelOptions"));
        let initial_options = format!(
            "{:?}",
            FindInitialModelOptions {
                cli_provider: None,
                cli_model: None,
                scoped_models: &[ScopedModel {
                    model: model("cov-auth", "alias-model"),
                    thinking_level: Some(ThinkingLevel::High),
                }],
                is_continuing: false,
                default_provider: Some("cov-auth"),
                default_model_id: Some("alias-model"),
                default_thinking_level: None,
                model_thinking_levels: Some(&BTreeMap::from([(
                    "cov-auth/alias-model".to_owned(),
                    ThinkingLevel::Max,
                )])),
                model_runtime: &view,
            }
        );
        assert!(initial_options.contains("FindInitialModelOptions"));

        let scoped = find_initial_model(FindInitialModelOptions {
            cli_provider: None,
            cli_model: None,
            scoped_models: &[ScopedModel {
                model: model("cov-auth", "alias-model"),
                thinking_level: Some(ThinkingLevel::High),
            }],
            is_continuing: false,
            default_provider: None,
            default_model_id: None,
            default_thinking_level: None,
            model_thinking_levels: Some(&BTreeMap::from([(
                "cov-auth/alias-model".to_owned(),
                ThinkingLevel::Max,
            )])),
            model_runtime: &view,
        })
        .expect("the scoped model serves");
        assert_eq!(
            scoped.thinking_level,
            ThinkingLevel::High,
            "the scoped level wins over the per-model map"
        );

        let failing = find_initial_model(FindInitialModelOptions {
            cli_provider: Some("cov-missing"),
            cli_model: Some("never-matches"),
            scoped_models: &[],
            is_continuing: false,
            default_provider: None,
            default_model_id: None,
            default_thinking_level: None,
            model_thinking_levels: None,
            model_runtime: &view,
        });
        assert!(
            failing
                .expect_err("the CLI error path errors")
                .0
                .contains("Unknown provider"),
        );

        let continuing = find_initial_model(FindInitialModelOptions {
            cli_provider: None,
            cli_model: None,
            scoped_models: &[ScopedModel {
                model: model("cov-auth", "alias-model"),
                thinking_level: None,
            }],
            is_continuing: true,
            default_provider: None,
            default_model_id: None,
            default_thinking_level: None,
            model_thinking_levels: None,
            model_runtime: &view,
        })
        .expect("the continuing ladder falls through to availability");
        assert_eq!(
            continuing.model.expect("the available snapshot serves").id,
            "alias-model"
        );

        let empty = ScriptedView {
            models: Vec::new(),
            available: Vec::new(),
            auth_providers: vec![],
        };
        let none = find_initial_model(FindInitialModelOptions {
            cli_provider: None,
            cli_model: None,
            scoped_models: &[],
            is_continuing: false,
            default_provider: None,
            default_model_id: None,
            default_thinking_level: None,
            model_thinking_levels: None,
            model_runtime: &empty,
        })
        .expect("the empty ladder returns no model");
        assert!(none.model.is_none());
    }
}

mod models_store_arms {
    use super::*;

    /// The in-memory store round-trips entries, deletes them, and aborts
    /// every operation on a cancelled signal.
    #[tokio::test]
    async fn the_in_memory_store_round_trips_and_aborts() {
        let store = InMemoryCodingAgentModelsStore::default();
        assert!(
            ModelsStore::read(&store, "cov", None)
                .await
                .expect("the read runs")
                .is_none()
        );
        ModelsStore::write(&store, "cov", store_entry(vec![model("cov", "m")]), None)
            .await
            .expect("the write lands");
        let read = ModelsStore::read(&store, "cov", None)
            .await
            .expect("the read runs")
            .expect("the entry round-trips");
        assert_eq!(read.models.len(), 1);
        ModelsStore::delete(&store, "cov", None)
            .await
            .expect("the delete lands");
        assert!(
            ModelsStore::read(&store, "cov", None)
                .await
                .expect("the read runs")
                .is_none()
        );

        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let options = ModelsStoreOptions {
            signal: Some(cancelled),
        };
        assert!(
            ModelsStore::read(&store, "cov", Some(&options))
                .await
                .is_err(),
            "a cancelled signal aborts the read"
        );
        assert!(
            ModelsStore::write(&store, "cov", store_entry(Vec::new()), Some(&options))
                .await
                .is_err(),
            "a cancelled signal aborts the write"
        );
        assert!(
            ModelsStore::delete(&store, "cov", Some(&options))
                .await
                .is_err(),
            "a cancelled signal aborts the delete"
        );

        // The options-carrying write and delete ride their signals through
        // the locked closures.
        let fresh = ModelsStoreOptions {
            signal: Some(CancellationToken::new()),
        };
        ModelsStore::write(
            &store,
            "cov",
            store_entry(vec![model("cov", "m")]),
            Some(&fresh),
        )
        .await
        .expect("the optioned write lands");
        let read = ModelsStore::read(&store, "cov", None)
            .await
            .expect("the read runs")
            .expect("the entry round-trips");
        assert_eq!(read.models.len(), 1);
        ModelsStore::delete(&store, "cov", Some(&fresh))
            .await
            .expect("the optioned delete lands");
    }

    /// The file store round-trips through its on-disk shape, serves the
    /// cached revision, coalesces concurrent readers, and reports the parse
    /// arms for corrupt bodies.
    #[tokio::test]
    async fn the_file_store_round_trips_coalesces_and_reports_corruption() {
        let dir = tempfile::tempdir().expect("the temp dir creates");
        let path = dir.path().join("models-store.json").display().to_string();
        let store = FileModelsStore::new(Some(&path)).expect("the store opens");

        ModelsStore::write(&store, "cov", store_entry(vec![model("cov", "m")]), None)
            .await
            .expect("the write lands");
        let read = ModelsStore::read(&store, "cov", None)
            .await
            .expect("the read runs")
            .expect("the entry round-trips");
        assert_eq!(read.models.len(), 1);
        assert!(
            ModelsStore::read(&store, "absent", None)
                .await
                .expect("the read runs")
                .is_none()
        );
        // The revision cache serves the second read without relocking.
        let cached = ModelsStore::read(&store, "cov", None)
            .await
            .expect("the cached read runs")
            .expect("the cache holds the entry");
        assert_eq!(cached.models.len(), 1);

        let (first, second) = tokio::join!(
            ModelsStore::read(&store, "cov", None),
            ModelsStore::read(&store, "cov", None),
        );
        assert!(
            first.is_ok() && second.is_ok(),
            "concurrent readers coalesce"
        );

        ModelsStore::delete(&store, "cov", None)
            .await
            .expect("the delete lands");
        assert!(
            ModelsStore::read(&store, "cov", None)
                .await
                .expect("the read runs")
                .is_none()
        );

        for body in ["{not json", "[]"] {
            std::fs::write(&path, body).expect("the corrupt body writes");
            assert!(
                ModelsStore::read(&store, "cov", None).await.is_err(),
                "{body} must fail the parse"
            );
        }

        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let options = ModelsStoreOptions {
            signal: Some(cancelled),
        };
        assert!(
            ModelsStore::read(&store, "cov", Some(&options))
                .await
                .is_err(),
            "a cancelled signal aborts the read"
        );
        assert!(
            ModelsStore::write(&store, "cov", store_entry(Vec::new()), Some(&options))
                .await
                .is_err(),
            "a cancelled signal aborts the write"
        );
    }
}

mod remote_catalog_arms {
    use super::*;

    /// The wrapped provider over the recording base and the injected client,
    /// with the publish sink the refresh assertions read.
    struct CatalogRig {
        provider: Arc<dyn Provider>,
        sink: Arc<Mutex<Vec<ModelsStoreEntry>>>,
        mock: MockHttpClient,
    }

    impl CatalogRig {
        /// The rig over `base_url`, serving `responses` in order.
        fn with_responses(base_url: Option<String>, responses: Vec<MockResponse>) -> Self {
            let mock = MockHttpClient::new();
            catalog_route(&mock);
            mock.on(|request| request.url.contains("/api/models/providers/cov-base"))
                .respond_sequence(responses);
            let log: Log = Arc::new(Mutex::new(Vec::new()));
            let sink: Arc<Mutex<Vec<ModelsStoreEntry>>> = Arc::new(Mutex::new(Vec::new()));
            let provider = with_remote_catalog_client(
                Arc::new(RecordingBase::healthy(&log)),
                base_url,
                None,
                Arc::new(mock.clone()),
            );
            Self {
                provider,
                sink,
                mock,
            }
        }

        /// One network-enabled, forced refresh over the rig.
        async fn refresh(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            self.provider
                .refresh_models(refresh_context(
                    recording_publish(&self.sink),
                    Some(Credential::ApiKey(ApiKeyCredential::default())),
                    true,
                ))
                .await
        }

        /// The last entry the refreshes wrote.
        fn last_written(&self) -> ModelsStoreEntry {
            written(&self.sink)
                .pop()
                .expect("the refresh wrote its entry")
        }
    }

    /// Catalog bodies that do not parse fail the refresh without publishing,
    /// and entries without a usable id drop out of the parsed list.
    #[tokio::test]
    async fn unparsable_catalog_bodies_fail_the_refresh() {
        for body in ["42".to_owned(), json!({"models": 7}).to_string()] {
            let rig = CatalogRig::with_responses(
                Some("https://cov.test".to_owned()),
                vec![
                    MockResponse::status(200)
                        .with_header("content-type", "application/json")
                        .with_body(body),
                ],
            );
            let failure = rig
                .refresh()
                .await
                .expect_err("the invalid body fails the refresh");
            assert!(
                failure.to_string().contains("Invalid model catalog"),
                "got {failure:?}"
            );
            assert!(
                written(&rig.sink).is_empty(),
                "the failure publishes nothing"
            );
            assert_eq!(ids(&rig.provider), ["base-model"]);
        }

        let rig = CatalogRig::with_responses(
            Some("https://cov.test".to_owned()),
            vec![json_response(
                200,
                &json!([{"no-id": true}, {"id": 5}, "bare"]),
            )],
        );
        rig.refresh().await.expect("the droppable entries parse");
        assert_eq!(rig.last_written().models, Vec::<Model>::new());
        assert_eq!(ids(&rig.provider), ["base-model"]);
    }

    /// Unparsable Last-Modified dates store zero; a valid date parses, the
    /// etag lands, and the overlay merges by id over the baseline.
    #[tokio::test]
    async fn unparsable_last_modified_dates_store_zero() {
        let malformed = [
            "nonsense",
            "Wed, not-a-day",
            "Wed, 21 Bad 2015 07:28:00 GMT",
            "Wed, 21 Oct yyyy 07:28:00 GMT",
            "Wed, 21 Oct 2015",
            "Wed, 21 Oct 2015 zz:28:00 GMT",
            "Wed, 21 Oct 2015 07:mm:00 GMT",
            "Wed, 21 Oct 2015 07:28:ss GMT",
        ];
        let mut responses = Vec::new();
        for date in malformed {
            responses.push(
                json_response(200, &catalog_body("dynamic")).with_header("last-modified", date),
            );
        }
        responses.push(
            json_response(200, &catalog_body("dynamic"))
                .with_header("last-modified", "Wed, 21 Oct 2015 07:28:00 GMT")
                .with_header("etag", "\"v1\""),
        );
        let rig = CatalogRig::with_responses(Some("https://cov.test".to_owned()), responses);

        for date in malformed {
            rig.refresh()
                .await
                .expect("the malformed-date refresh lands");
            assert_eq!(
                rig.last_written().last_modified,
                Some(0),
                "{date} stores zero"
            );
        }
        rig.refresh().await.expect("the valid-date refresh lands");
        let entry = rig.last_written();
        assert!(entry.last_modified.is_some_and(|ms| ms > 0));
        assert_eq!(entry.etag.as_deref(), Some("\"v1\""));
        assert_eq!(ids(&rig.provider), ["base-model", "dynamic"]);
        assert_eq!(rig.mock.request_count(), malformed.len() + 1);
    }

    /// A 304 without a stored entry has nothing to revalidate: the response
    /// falls through to the transient-failure arm and the refresh reports it.
    #[tokio::test]
    async fn a_304_without_a_stored_entry_reports_the_failure() {
        let rig = CatalogRig::with_responses(
            Some("https://cov.test".to_owned()),
            vec![MockResponse::status(304)],
        );
        let failure = rig.refresh().await.expect_err("the bare 304 fails");
        assert!(
            failure
                .to_string()
                .contains("Model catalog request failed for cov-base: 304"),
            "got {failure:?}"
        );
        let entry = rig.last_written();
        assert!(
            entry.checked_at.is_some(),
            "the transient arm moves the freshness window"
        );
        assert_eq!(entry.models, Vec::<Model>::new(), "no overlay backed it");
    }

    /// A base URL that does not parse fails the refresh before any request.
    #[tokio::test]
    async fn an_unparsable_catalog_base_url_fails_the_refresh() {
        let rig = CatalogRig::with_responses(Some("not a url".to_owned()), Vec::new());
        let failure = rig
            .refresh()
            .await
            .expect_err("the bad base fails before the request");
        assert!(
            failure.to_string().contains("relative URL without a base")
                || failure.to_string().contains("empty host")
        );
        assert_eq!(rig.mock.request_count(), 0);
    }
}

mod inherited_arms {
    use super::*;

    /// The composed api-key method delegates the inherited method's check and
    /// resolve closures: stored credentials, empty credentials, and raw keys
    /// each route through the base's own closures.
    #[tokio::test]
    async fn inherited_api_key_methods_delegate_to_the_base() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let provider = compose_model_provider(
            "cov-inherit",
            Some(Arc::new(RecordingBase::with_inherited_api_key(&log))),
            &ModelConfig::default(),
            Some(ProviderConfigInput {
                api_key: Some("$COV_INHERIT".to_owned()),
                ..ProviderConfigInput::default()
            }),
        )
        .expect("the composed provider builds");
        let api = provider
            .auth()
            .api_key
            .clone()
            .expect("the inherited method composes");
        assert_eq!(api.name, "Inherited Key", "the inherited name wins");
        let check = api.check.expect("the composed check exists");
        let resolve = api.resolve.clone();
        let ctx: Arc<dyn AuthContext> = Arc::new(EnvContext(BTreeMap::new()));

        let stored = check(api_input(
            &ctx,
            Some(ApiKeyCredential {
                key: Some("sk-stored".to_owned()),
                env: None,
            }),
        ))
        .await
        .expect("the check runs")
        .expect("the inherited check answers");
        assert_eq!(stored.source.as_deref(), Some("sk-stored"));

        let empty = check(api_input(&ctx, Some(ApiKeyCredential::default())))
            .await
            .expect("the check runs");
        assert!(empty.is_none(), "the inherited resolve finds nothing");

        let resolved = resolve(api_input(
            &ctx,
            Some(ApiKeyCredential {
                key: Some("sk-resolved".to_owned()),
                env: None,
            }),
        ))
        .await
        .expect("the resolve runs")
        .expect("the inherited resolve answers");
        assert_eq!(resolved.source.as_deref(), Some("inherited"));
    }

    /// The composed provider exposes the base's headers, delegates the base's
    /// refresh phase, and maps a later base-catalog failure through the
    /// composed model list.
    #[tokio::test]
    async fn the_composed_provider_delegates_the_base_surfaces() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let base = RecordingBase::with_refresh_phase(&log);
        let fail_flag = Arc::clone(&base.failing_catalog);
        let provider = compose_model_provider(
            "cov-del",
            Some(Arc::new(base)),
            &ModelConfig::default(),
            Some(ProviderConfigInput {
                api_key: Some("lit".to_owned()),
                ..ProviderConfigInput::default()
            }),
        )
        .expect("the composed provider builds");
        assert!(provider.headers().is_none());

        provider
            .refresh_models(refresh_context(
                recording_publish(&Arc::new(Mutex::new(Vec::new()))),
                None,
                false,
            ))
            .await
            .expect("the base refresh lands");
        assert!(recorded(&log).contains(&"base-refresh".to_owned()));

        fail_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(
            provider.get_models().is_err(),
            "the composed get_models maps the base failure"
        );
    }

    /// The full stream surfaces the dispatch error the same way the simple
    /// stream does, and a broken models.json layer fails validation without
    /// touching the stored configuration.
    #[tokio::test]
    async fn stream_errors_and_validation_arms() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let provider = compose_model_provider(
            "cov-err",
            Some(Arc::new(RecordingBase::healthy(&log))),
            &ModelConfig::default(),
            None,
        )
        .expect("the composed provider builds");
        let mut odd = model("cov-err", "odd");
        odd.api = pi_ai::types::Api::from("cov-no-such-api");
        let failed = provider.stream(&odd, &empty_context(), None).result().await;
        assert!(
            failed
                .error_message
                .as_deref()
                .is_some_and(|message| message.contains("No API provider registered")),
            "got {:?}",
            failed.error_message,
        );

        let broken = config_raw(r#"{"providers": {"cov-broken": {"oauth": "radius"}}}"#);
        let broken_provider = broken.get_provider("cov-broken").cloned();
        let extension = ProviderConfigInput::default();
        assert!(
            validate_extension_provider("cov-v", None, broken_provider.as_ref(), &extension)
                .expect_err("the broken layer fails validation")
                .contains("\"baseUrl\" is required when \"oauth\" is set"),
        );
    }

    /// An extension model without its own api or base URL inherits both from
    /// the exact-id default the composed base catalog supplies.
    #[test]
    fn extension_models_inherit_the_default_sibling_fields() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let provider = compose_model_provider(
            "cov-inheriterit-two",
            Some(Arc::new(RecordingBase::healthy(&log))),
            &ModelConfig::default(),
            Some(ProviderConfigInput {
                api_key: Some("lit".to_owned()),
                models: Some(vec![ProviderModelInput {
                    id: "base-model".to_owned(),
                    name: "Rebased".to_owned(),
                    api: None,
                    base_url: None,
                    reasoning: false,
                    thinking_level_map: None,
                    input: vec![pi_ai::types::Modality::Text],
                    cost: common::model_layer::zero_cost(),
                    context_window: 10,
                    max_tokens: 5,
                    sampling_params: None,
                    headers: None,
                    compat: None,
                }]),
                ..ProviderConfigInput::default()
            }),
        )
        .expect("the composed provider builds");
        let models = provider.get_models().expect("the models list");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "Rebased");
        assert_eq!(
            models[0].base_url, "https://example.test/v1",
            "the default sibling's base URL inherits"
        );
        assert_eq!(
            models[0].api.0, "openai-completions",
            "the default sibling's api inherits"
        );
    }

    /// The composed api-key login surfaces the prompt's failure, and the
    /// composed check routes an empty stored credential through the inherited
    /// resolve-only method.
    #[tokio::test]
    async fn the_login_error_and_the_resolve_only_check_route() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let provider = compose_model_provider(
            "cov-ronly",
            Some(Arc::new(RecordingBase::with_resolve_only_api_key(&log))),
            &ModelConfig::default(),
            None,
        )
        .expect("the composed provider builds");
        let api = provider
            .auth()
            .api_key
            .clone()
            .expect("the method composes");
        let ctx: Arc<dyn AuthContext> = Arc::new(EnvContext(BTreeMap::from([(
            "COV_AMBIENT".to_owned(),
            "sk-ambient".to_owned(),
        )])));

        let empty = api.check.clone().expect("the composed check exists")(api_input(
            &ctx,
            Some(ApiKeyCredential::default()),
        ))
        .await
        .expect("the check runs")
        .expect("the ambient env answers the empty credential");
        assert_eq!(empty.source.as_deref(), Some("resolve-only"));

        let resolved = api.resolve.clone()(api_input(
            &ctx,
            Some(ApiKeyCredential {
                key: Some("sk-only".to_owned()),
                env: None,
            }),
        ))
        .await
        .expect("the resolve runs")
        .expect("the resolve-only method answers");
        assert_eq!(resolved.source.as_deref(), Some("resolve-only"));

        let login = api.login.expect("the default login composes");
        let outcome = login(pi_ai::auth::types::ProviderAuthInteraction {
            signal: CancellationToken::new(),
            prompt: Arc::new(|_prompt: AuthPrompt| Box::pin(async { Err(AbortError) })),
            notify: Arc::new(|_event: AuthEvent| {}),
        })
        .await;
        assert!(outcome.is_err(), "the aborted prompt fails the login");
    }

    /// The raw-key resolve delegates to the inherited resolve with the
    /// resolved key, the arm the layered providers ride.
    #[tokio::test]
    async fn the_raw_key_resolve_delegates_to_the_inherited_method() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let provider = compose_model_provider(
            "cov-rawkey",
            Some(Arc::new(RecordingBase::with_inherited_api_key(&log))),
            &ModelConfig::default(),
            Some(ProviderConfigInput {
                api_key: Some("$COV_RAW".to_owned()),
                ..ProviderConfigInput::default()
            }),
        )
        .expect("the composed provider builds");
        let api = provider
            .auth()
            .api_key
            .clone()
            .expect("the method composes");
        let ctx: Arc<dyn AuthContext> = Arc::new(EnvContext(BTreeMap::from([(
            "COV_RAW".to_owned(),
            "sk-raw".to_owned(),
        )])));
        let resolved = (api.resolve)(api_input(&ctx, None))
            .await
            .expect("the resolve runs")
            .expect("the raw key resolves through the inherited method");
        assert_eq!(resolved.auth.api_key.as_deref(), Some("sk-raw"));
        assert_eq!(resolved.source.as_deref(), Some("inherited"));
    }

    /// Validation surfaces the base catalog's own failure, the arm the
    /// re-registration guard rides.
    #[test]
    fn validation_surfaces_the_base_catalog_failure() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let base: Arc<dyn Provider> = Arc::new(RecordingBase::failing(&log));
        let extension = ProviderConfigInput {
            api_key: Some("lit".to_owned()),
            ..ProviderConfigInput::default()
        };
        assert!(
            validate_extension_provider("cov-vbase", Some(&base), None, &extension)
                .expect_err("the failing base catalog fails validation")
                .contains("base catalog failed"),
        );
    }

    /// The OAuth derivation fails when the configured headers do not resolve,
    /// the error arm upstream's toAuth wrapper carries.
    #[tokio::test]
    async fn oauth_derivation_fails_on_unresolvable_headers() {
        let extension = ProviderConfigInput {
            headers: Some(BTreeMap::from([(
                "x-unset".to_owned(),
                "$PI_RUST_COV_UNSET_VAR".to_owned(),
            )])),
            oauth: Some(ExtensionOAuthConfig {
                name: "Cov OAuth".to_owned(),
                is_subscription: None,
                uses_callback_server: None,
                login: Arc::new(|_callbacks| Box::pin(async { Err(unused_failure()) })),
                refresh_token: Arc::new(|_credentials, _signal| {
                    Box::pin(async { Err(unused_failure()) })
                }),
                get_api_key: Arc::new(|_credentials: &OAuthCredentials| "k".to_owned()),
                modify_models: None,
            }),
            ..ProviderConfigInput::default()
        };
        let provider = composed("cov-obad", None, Some(extension)).expect("the provider composes");
        let oauth = provider
            .auth()
            .oauth
            .clone()
            .expect("the oauth method composes");
        let outcome = (oauth.to_auth)(OAuthCredentials {
            refresh: "r".to_owned(),
            access: "a".to_owned(),
            expires: i64::MAX,
            extra: BTreeMap::new(),
        })
        .await;
        assert!(
            outcome.is_err(),
            "the unresolvable header fails the derivation"
        );
    }
}

mod runtime_arms {
    use super::*;

    /// The runtime renders through its Debug impls, and the snapshot/config
    /// and credential-listing reads answer directly.
    #[tokio::test]
    async fn runtime_reads_render_and_answer() {
        let runtime = cov_runtime().await;
        assert!(format!("{runtime:?}").contains("ModelRuntimeCore"));

        assert!(runtime.get_models_config().get_provider_ids().is_empty());
        assert!(
            runtime
                .list_credentials(None)
                .await
                .expect("the listing reads")
                .is_empty()
        );

        runtime
            .register_provider(
                "cov-defer",
                ProviderConfigInput {
                    api_key: Some("lit".to_owned()),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");
        let handle = pi_ai::types::DeferredHandle {
            provider: "cov-defer".to_owned(),
            model_id: "m".to_owned(),
            api: "openai-completions".to_owned(),
            id: "resp_1".to_owned(),
            expires_at: None,
            poll_after_ms: None,
            data: None,
        };
        let stream_model = model("cov-defer", "m");
        let deferred = runtime.stream_deferred(&stream_model, &handle, None);
        let message = deferred.result().await;
        assert!(
            message
                .error_message
                .as_deref()
                .is_some_and(|message| message.contains("does not support deferred responses")),
            "got {:?}",
            message.error_message,
        );
        let cancelled = runtime
            .cancel_deferred(&stream_model, &handle, None)
            .await
            .expect_err("the unsupported cancellation fails");
        assert!(
            cancelled
                .to_string()
                .contains("does not support deferred responses"),
            "got {cancelled:?}"
        );
    }

    /// `create` takes its own default stores: the file-backed auth store over
    /// `auth_path` and the file-backed catalog store beside `models.json`,
    /// and a radius gateway promotes into the builtin list.
    #[tokio::test]
    async fn create_takes_its_default_stores_and_radius_gateways() {
        let dir = tempfile::tempdir().expect("the temp dir creates");
        let models_path = dir.path().join("models.json").display().to_string();
        std::fs::write(
            &models_path,
            json!({"providers": {"cov-radius": {
                "name": "Cov Radius", "baseUrl": "https://gw.example/v1",
                "oauth": "radius", "api": "openai-completions",
                "models": [{"id": "radius-model", "contextWindow": 10, "maxTokens": 5}]}}})
            .to_string(),
        )
        .expect("the models.json writes");
        let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
            auth_path: Some(dir.path().join("auth.json").display().to_string()),
            models_path: Some(Some(models_path)),
            allow_model_network: false,
            ..CreateModelRuntimeOptions::default()
        })
        .await
        .expect("the default-store create lands");
        assert!(
            ModelRuntimeCore::get_models(&runtime, None)
                .iter()
                .any(|model| model.provider.0 == "cov-radius"),
            "the radius gateway's models compose"
        );
        let provider = runtime
            .get_provider("cov-radius")
            .expect("the radius gateway registers");
        assert_eq!(provider.name(), "Cov Radius");
        assert!(
            dir.path().join("auth.json").exists(),
            "the auth store creates"
        );
    }

    /// The credential lifecycle commits through the login and logout flows:
    /// the prompted login stores the credential, the logout removes it, and
    /// both synchronize the snapshot through their operations.
    #[tokio::test]
    async fn the_credential_lifecycle_commits_login_and_logout() {
        let runtime = cov_runtime().await;
        runtime
            .register_provider(
                "cov-life",
                ProviderConfigInput {
                    api_key: Some("lit".to_owned()),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");

        let credential = runtime
            .login(
                "cov-life",
                pi_ai::auth::types::AuthType::ApiKey,
                AuthInteraction {
                    signal: Some(CancellationToken::new()),
                    prompt: Arc::new(|_prompt: AuthPrompt| {
                        Box::pin(async { Ok("typed-life-key".to_owned()) })
                    }),
                    notify: Arc::new(|_event: AuthEvent| {}),
                },
            )
            .await
            .expect("the login commits");
        assert_eq!(
            credential
                .as_api_key_credential()
                .and_then(|key| key.key.as_deref()),
            Some("typed-life-key"),
        );

        runtime
            .logout(
                "cov-life",
                Some(&pi_ai::auth::types::AuthOptions {
                    signal: Some(CancellationToken::new()),
                }),
            )
            .await
            .expect("the logout commits");
    }

    /// Availability passes record per-provider failures the snapshot carries,
    /// and a cancelled refresh reports its provider errors.
    #[tokio::test]
    async fn availability_failures_record_per_provider() {
        let runtime = cov_runtime().await;
        // The native provider's auth resolution fails, the error the
        // availability pass records for the provider.
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let mut failing_base = RecordingBase::with_inherited_api_key(&log);
        failing_base.auth = ProviderAuth {
            api_key: Some(pi_ai::auth::types::ApiKeyAuth {
                name: "Failing".to_owned(),
                login: None,
                check: None,
                resolve: Arc::new(|_input: ApiKeyAuthInput| {
                    let error: Box<dyn std::error::Error + Send + Sync> =
                        Box::new(std::io::Error::other("resolution failed"));
                    Box::pin(async { Err(error) })
                }),
            }),
            oauth: None,
        };
        runtime.register_native_provider(Arc::new(failing_base));
        assert!(runtime.get_registered_native_provider("cov-base").is_some());

        let failed = runtime
            .refresh(pi_ai::models::ModelsRefreshOptions {
                allow_network: Some(false),
                providers: Some(vec!["cov-base".to_owned()]),
                ..pi_ai::models::ModelsRefreshOptions::default()
            })
            .await;
        assert!(
            failed.errors.contains_key("cov-base"),
            "the failing resolution rides the refresh result: {:?}",
            failed.errors,
        );

        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let aborted_refresh = runtime
            .refresh(pi_ai::models::ModelsRefreshOptions {
                allow_network: Some(false),
                providers: Some(vec!["cov-base".to_owned()]),
                signal: Some(cancelled),
                ..pi_ai::models::ModelsRefreshOptions::default()
            })
            .await;
        assert!(
            aborted_refresh.aborted,
            "the cancelled refresh reports aborted"
        );
    }

    /// The runtime's own stream entries prepare the request and hand the
    /// composed provider the merged options, the path the coding surface
    /// drives.
    #[tokio::test]
    async fn the_runtime_stream_entries_prepare_and_delegate() {
        let runtime = cov_runtime().await;
        runtime
            .register_provider(
                "cov-rt",
                ProviderConfigInput {
                    api_key: Some("lit".to_owned()),
                    api: Some(pi_ai::types::Api::from("openai-completions")),
                    models: Some(vec![ProviderModelInput {
                        id: "rt-model".to_owned(),
                        name: "RT".to_owned(),
                        api: Some(pi_ai::types::Api::from("openai-completions")),
                        base_url: Some("https://rt.test/v1".to_owned()),
                        reasoning: false,
                        thinking_level_map: None,
                        input: vec![pi_ai::types::Modality::Text],
                        cost: common::model_layer::zero_cost(),
                        context_window: 10,
                        max_tokens: 5,
                        sampling_params: None,
                        headers: None,
                        compat: None,
                    }]),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");
        let stream_model = ModelRuntimeCore::get_models(&runtime, Some("cov-rt"))
            .into_iter()
            .next()
            .expect("the model lists");

        let mock = MockHttpClient::new();
        mock.on(|request| request.url.contains("/chat/completions"))
            .respond(
                MockResponse::status(200)
                    .with_header("content-type", "text/event-stream")
                    .with_body("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n"),
            );
        let simple_options = pi_ai::models::ModelsSimpleStreamOptions {
            options: SimpleStreamOptions {
                api_key: Some("sk-rt".to_owned()),
                headers: Some(BTreeMap::from([("x-rt".to_owned(), Some("v".to_owned()))])),
                env: Some(BTreeMap::from([("COV_RT".to_owned(), "1".to_owned())])),
                transport_options: transport(&mock),
                ..SimpleStreamOptions::default()
            },
            transform_headers: None,
        };
        let message = runtime
            .complete_simple(&stream_model, &empty_context(), Some(&simple_options))
            .await;
        assert_eq!(message.stop_reason, StopReason::Stop, "got {message:?}");

        let full_options = pi_ai::models::ModelsStreamOptions {
            options: StreamOptions {
                api_key: Some("sk-rt".to_owned()),
                headers: Some(BTreeMap::from([("x-rt".to_owned(), Some("v".to_owned()))])),
                transport_options: transport(&mock),
                ..StreamOptions::default()
            },
            transform_headers: None,
        };
        let completed = runtime
            .complete(&stream_model, &empty_context(), Some(&full_options))
            .await;
        assert_eq!(completed.stop_reason, StopReason::Stop);
        assert_eq!(mock.recorded().len(), 2, "both requests rode the mock");
    }

    /// The availability reads carry the caller's auth options, and the
    /// failing provider check records the per-provider error.
    #[tokio::test]
    async fn availability_reads_carry_options_and_record_failures() {
        let runtime = cov_runtime().await;
        runtime
            .register_provider(
                "cov-avail2",
                ProviderConfigInput {
                    api_key: Some("lit".to_owned()),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");

        let options = pi_ai::auth::types::AuthOptions {
            signal: Some(CancellationToken::new()),
        };
        let scoped =
            ModelRuntimeCore::get_available(&runtime, Some("cov-avail2"), Some(&options)).await;
        assert!(scoped.is_ok(), "the provider-scoped pass runs: {scoped:?}");
        let all = ModelRuntimeCore::get_available(&runtime, None, Some(&options)).await;
        assert!(all.is_ok(), "the queued pass runs: {all:?}");

        let log: Log = Arc::new(Mutex::new(Vec::new()));
        runtime.register_native_provider(Arc::new(RecordingBase::with_failing_check(&log)));
        let failed = runtime
            .refresh(pi_ai::models::ModelsRefreshOptions {
                allow_network: Some(false),
                providers: Some(vec!["cov-base".to_owned()]),
                ..pi_ai::models::ModelsRefreshOptions::default()
            })
            .await;
        assert!(
            failed.errors.contains_key("cov-base"),
            "the failing check rides the refresh result: {:?}",
            failed.errors,
        );
    }

    /// A radius gateway without a display name takes the provider id, the
    /// promotion arm `configure_radius_providers` carries.
    #[tokio::test]
    async fn a_nameless_radius_gateway_takes_the_provider_id() {
        let dir = tempfile::tempdir().expect("the temp dir creates");
        let models_path = dir.path().join("models.json").display().to_string();
        std::fs::write(
            &models_path,
            json!({"providers": {"cov-rls": {
                "baseUrl": "https://gw.example/v1", "oauth": "radius",
                "api": "openai-completions",
                "models": [{"id": "radius-model", "contextWindow": 10, "maxTokens": 5}]}}})
            .to_string(),
        )
        .expect("the models.json writes");
        let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
            credentials: Some(empty_auth_storage()),
            models_path: Some(Some(models_path)),
            models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::default())),
            allow_model_network: false,
            ..CreateModelRuntimeOptions::default()
        })
        .await
        .expect("the runtime constructs");
        let provider = runtime
            .get_provider("cov-rls")
            .expect("the gateway registers");
        assert_eq!(provider.name(), "cov-rls");
    }

    /// A models.json path that does not normalize fails the create, the arm
    /// the load error rides.
    #[tokio::test]
    async fn an_unnormalizable_models_path_fails_the_create() {
        let outcome = ModelRuntime::create(CreateModelRuntimeOptions {
            credentials: Some(empty_auth_storage()),
            models_path: Some(Some("file://somehost/tmp/x".to_owned())),
            models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::default())),
            allow_model_network: false,
            ..CreateModelRuntimeOptions::default()
        })
        .await;
        assert!(outcome.is_err(), "the bad path fails the create");
    }

    /// Model-scoped auth resolution overlays the caller's env, and the
    /// runtime-key operations read the auth options they carry.
    #[tokio::test]
    async fn model_auth_resolution_carries_the_env_overlay() {
        let runtime = cov_runtime().await;
        runtime
            .register_provider(
                "cov-env",
                ProviderConfigInput {
                    api_key: Some("lit".to_owned()),
                    headers: Some(BTreeMap::from([("x-env".to_owned(), "v".to_owned())])),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");
        let stream_model = model("cov-env", "m");
        let merged = runtime
            .get_auth_for_model(
                &stream_model,
                Some(&ModelRuntimeAuthOverrides {
                    env: Some(BTreeMap::from([("COV_ENV".to_owned(), "1".to_owned())])),
                    ..ModelRuntimeAuthOverrides::default()
                }),
            )
            .await
            .expect("the resolution runs")
            .expect("the configured provider resolves");
        assert_eq!(merged.auth.api_key.as_deref(), Some("lit"));

        runtime
            .set_runtime_api_key(
                "cov-env",
                "sk-runtime".to_owned(),
                Some(&pi_ai::auth::types::AuthOptions {
                    signal: Some(CancellationToken::new()),
                }),
            )
            .await
            .expect("the runtime key commits");
        assert_eq!(
            runtime.get_provider_auth_status("cov-env").source,
            Some(AuthStatusSource::Runtime)
        );
        runtime
            .remove_runtime_api_key(
                "cov-env",
                Some(&pi_ai::auth::types::AuthOptions {
                    signal: Some(CancellationToken::new()),
                }),
            )
            .await
            .expect("the runtime key removes");
    }
}

mod second_pass_arms {
    use super::*;

    /// The remote catalog wrapper answers the deferred-response surfaces the
    /// way the inner provider does: none of the catalog overlay's business.
    #[test]
    fn the_catalog_wrapper_defers_to_the_inner_provider() {
        let log: Log = Arc::new(Mutex::new(Vec::new()));
        let provider = with_remote_catalog_client(
            Arc::new(RecordingBase::healthy(&log)),
            None,
            None,
            Arc::new(MockHttpClient::new()),
        );
        let handle = pi_ai::types::DeferredHandle {
            provider: "cov-base".to_owned(),
            model_id: "base-model".to_owned(),
            api: "openai-completions".to_owned(),
            id: "resp_1".to_owned(),
            expires_at: None,
            poll_after_ms: None,
            data: None,
        };
        let stream_model = model("cov-base", "base-model");
        assert!(
            provider
                .fetch_deferred(&stream_model, &handle, None)
                .is_none()
        );
        assert!(
            provider
                .cancel_deferred(&stream_model, &handle, None)
                .is_none()
        );
        assert!(!provider.supports_fetch_deferred());
        assert!(!provider.supports_cancel_deferred());
    }

    /// A tier threshold that is not a whole token count passes the schema
    /// walk but fails the models.json parse, the arm the snapshot's load
    /// error carries.
    #[test]
    fn a_fractional_tier_threshold_fails_the_snapshot_parse() {
        let dir = tempfile::tempdir().expect("the temp dir creates");
        let path = dir.path().join("models.json").display().to_string();
        let broken = json!({"providers": {"cov-tier": {
            "baseUrl": "https://x", "apiKey": "lit", "api": "openai-completions",
            "models": [{"id": "m", "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                "tiers": [{"inputTokensAbove": 1.5, "input": 1, "output": 1, "cacheRead": 1, "cacheWrite": 1}]}}]}}});
        std::fs::write(&path, broken.to_string()).expect("the body writes");
        let failure = ModelConfig::load(Some(&path))
            .expect_err("the schema-passing body fails the models.json parse");
        assert!(
            failure.contains("Failed to parse models.json"),
            "got {failure}"
        );
    }

    /// The file store refuses paths that do not normalize, both on the
    /// default lock and the replaced lock strategy.
    #[test]
    fn the_file_store_refuses_unnormalizable_paths() {
        let hosted = "file://somehost/tmp/x";
        assert!(FileModelsStore::new(Some(hosted)).is_err());
        assert!(
            FileModelsStore::with_lock_strategy(
                hosted,
                Arc::new(pi_coding_agent::file_lock::MkdirLock)
            )
            .is_err()
        );
    }

    /// A stored value that does not deserialize fails the read, and a corrupt
    /// file fails the delete under the same lock.
    #[tokio::test]
    async fn corrupt_entries_fail_reads_and_deletes() {
        let dir = tempfile::tempdir().expect("the temp dir creates");
        let path = dir.path().join("models-store.json").display().to_string();
        let store = FileModelsStore::new(Some(&path)).expect("the store opens");

        ModelsStore::write(&store, "cov", store_entry(vec![model("cov", "m")]), None)
            .await
            .expect("the write lands");
        let mut raw = std::fs::read_to_string(&path).expect("the store file reads");
        raw = raw.replace(r#""models""#, r#""models": "junk", "ignored""#);
        std::fs::write(&path, raw).expect("the corrupt entry writes");
        assert!(
            ModelsStore::read(&store, "cov", None).await.is_err(),
            "a non-array models value fails the entry parse"
        );

        std::fs::write(&path, "{not json").expect("the corrupt body writes");
        let outcome = ModelsStore::delete(&store, "cov", None).await;
        let message = outcome.as_ref().err().map(ToString::to_string);
        assert!(
            message
                .as_deref()
                .is_some_and(|message| message.contains("key must be a string")),
            "the delete fails inside its parse closure: {message:?}"
        );
    }

    /// A reader departing while a coalesced reload is still pending arms the
    /// reload's cancellation and clears the slot; the surviving reader still
    /// settles.
    #[tokio::test]
    async fn a_departing_reader_cancels_the_pending_reload() {
        use pi_coding_agent::file_lock::{FileLock, FileLockGuard};

        /// The lock double that gates the first acquire, standing in for the
        /// slow disk the race needs.
        struct GatedLock;

        impl FileLock for GatedLock {
            fn lock<'a>(
                &'a self,
                lock_dir: &'a std::path::Path,
                options: &'a pi_coding_agent::file_lock::AsyncLockOptions,
            ) -> pi_ai::types::BoxedFuture<
                'a,
                Result<FileLockGuard, Box<dyn std::error::Error + Send + Sync>>,
            > {
                Box::pin(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(120)).await;
                    let guard = pi_coding_agent::file_lock::MkdirLock
                        .lock(lock_dir, options)
                        .await?;
                    Ok(guard)
                })
            }
        }

        let dir = tempfile::tempdir().expect("the temp dir creates");
        let path = dir.path().join("models-store.json").display().to_string();
        let store = Arc::new(
            FileModelsStore::with_lock_strategy(&path, Arc::new(GatedLock))
                .expect("the gated store opens"),
        );
        ModelsStore::write(&*store, "cov", store_entry(vec![model("cov", "m")]), None)
            .await
            .expect("the write lands");
        // Invalidate the cached revision so the next read spawns a reload.
        std::fs::remove_file(&path).expect("the file removes");

        let reader_signal = CancellationToken::new();
        let reader_store = Arc::clone(&store);
        let reader = tokio::spawn(async move {
            let token = reader_signal.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                token.cancel();
            });
            ModelsStore::read(
                &*reader_store,
                "cov",
                Some(&ModelsStoreOptions {
                    signal: Some(reader_signal),
                }),
            )
            .await
        });
        // The sole reader cancels mid-reload: the last departing reader arms
        // the pending reload's cancellation and clears the slot.
        reader
            .await
            .expect("the reader task joins")
            .expect_err("the cancelled reader aborts");

        // A later reader finds no slot and reloads plainly.
        let settled = ModelsStore::read(&*store, "cov", None)
            .await
            .expect("the later read runs");
        assert_eq!(
            settled, None,
            "the deleted file parses to an empty snapshot"
        );
    }
}

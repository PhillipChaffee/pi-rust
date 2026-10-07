//! Upstream `packages/coding-agent/test/model-runtime-modify-models-compat.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated for
//! `pi_coding_agent::model_runtime` (#121).
//!
//! Porting restatements this suite records:
//!
//! - `getProvider(...).toBe(provider)` restates as `Arc::ptr_eq`: a native
//!   registration with no models.json config and no extension overlay
//!   registers the base provider untouched, so the registry hands back the
//!   registered object itself.
//! - The registration's fire-and-forget refresh (`void this.refresh(...)`)
//!   runs on a spawned task the test cannot await; the refresh-projection
//!   cases observe through repeated refresh passes, which converge once no
//!   scheduled refresh is mid-rebuild, and assert at the observation's own
//!   synchronous point where no task can interleave.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::{Arc, Mutex};

use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::types::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCheckFn, ApiKeyCredential, ApiKeyLoginFn, ApiKeyResolveFn,
    AuthCheck, AuthError, AuthEvent, AuthInteraction, AuthPrompt, AuthPromptKind, AuthResult,
    AuthType, ModelAuth, OAuthCredentials, ProviderAuth, ProviderAuthInteraction,
};
use pi_ai::models::{
    ModelsDeferredCancelOptions, ModelsDeferredFetchOptions, ModelsRefreshOptions, Provider,
    ProviderError, ProviderModelError, RefreshModelsContext, TransformHeadersFn,
};
use pi_ai::models_store::{InMemoryModelsStore, ModelsStore};
use pi_ai::types::{
    Api, AssistantMessage, AssistantMessageEvent, BoxedFuture, Context, DeferredCancelOptions,
    DeferredFetchOptions, DeferredHandle, Model, ProviderHeaders, SimpleStreamOptions, StopReason,
    StreamOptions, Usage,
};
use pi_ai::utils::abort::AbortError;
use pi_ai::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use pi_ai::utils::provider_retry::ProviderRequestError;
use pi_coding_agent::auth_storage::AuthStorageData;
use pi_coding_agent::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use pi_coding_agent::provider_composer::{
    ExtensionOAuthConfig, ExtensionRefreshModelsFn, ProviderConfigInput, ProviderModelInput,
};
use serde_json::json;
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;

/// The shared fixture module compiles whole into every test binary; the
/// config suites' env lookups stay unused in this one.
#[expect(
    dead_code,
    reason = "every test binary recompiles the shared fixture module and consumes only its own helpers"
)]
mod common;

use common::model_layer::{
    create_in_memory_model_registry, empty_auth_storage, in_memory_auth_storage, model,
};

/// The composed wrapper the registry hands back is a different object than
/// the registered double, upstream's `toBe` against a composed provider.
const COMPOSED_NAME: &str = "Extension Native";

/// The poisoned-mutex-tolerant lock the recording cells read through.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The zero-usage message a double's stream settles with, upstream's inline
/// message literal.
fn message_for(model: &Model, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

/// The stream the doubles return for the dispatches no case drives: it
/// settles with the `unused` error, so an accidental dispatch fails the
/// stream instead of hanging.
fn unused_stream(model: &Model) -> AssistantMessageEventStream {
    let mut message = message_for(model, StopReason::Error);
    message.error_message = Some("unused".to_owned());
    let stream = assistant_message_event_stream();
    stream.end(Some(&message));
    stream
}

/// What the deferred double's fetch observed, upstream's `fetchedBaseUrl`
/// and `fetchedOptions` cells.
#[derive(Clone)]
struct FetchRecord {
    base_url: String,
    options: Option<DeferredFetchOptions>,
}

/// What the deferred double's cancel observed, upstream's `cancelledId` and
/// `cancelledOptions` cells.
#[derive(Clone)]
struct CancelRecord {
    id: String,
    options: Option<DeferredCancelOptions>,
}

/// A native provider double: id, name, one catalog model, and the auth
/// method the case drives; the optional recording cells back the deferred
/// methods the overlays case drives.
struct NativeDouble {
    id: &'static str,
    name: &'static str,
    model: Model,
    auth: ProviderAuth,
    fetched: Option<Arc<Mutex<Option<FetchRecord>>>>,
    cancelled: Option<Arc<Mutex<Option<CancelRecord>>>>,
}

impl NativeDouble {
    const fn new(id: &'static str, name: &'static str, model: Model, auth: ProviderAuth) -> Self {
        Self {
            id,
            name,
            model,
            auth,
            fetched: None,
            cancelled: None,
        }
    }

    /// Arm the deferred recording cells, the overlays case's double shape.
    fn with_deferred_recording(
        mut self,
        fetched: Arc<Mutex<Option<FetchRecord>>>,
        cancelled: Arc<Mutex<Option<CancelRecord>>>,
    ) -> Self {
        self.fetched = Some(fetched);
        self.cancelled = Some(cancelled);
        self
    }
}

impl Provider for NativeDouble {
    fn id(&self) -> &str {
        self.id
    }

    fn name(&self) -> &str {
        self.name
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn get_models(&self) -> Result<Vec<Model>, ProviderModelError> {
        Ok(vec![self.model.clone()])
    }

    fn stream(
        &self,
        model: &Model,
        _context: &Context,
        _options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        unused_stream(model)
    }

    fn stream_simple(
        &self,
        model: &Model,
        _context: &Context,
        _options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        unused_stream(model)
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        _handle: &DeferredHandle,
        options: Option<&DeferredFetchOptions>,
    ) -> Option<AssistantMessageEventStream> {
        let fetched = self.fetched.as_ref()?;
        *lock(fetched) = Some(FetchRecord {
            base_url: model.base_url.clone(),
            options: options.cloned(),
        });
        let message = message_for(model, StopReason::Stop);
        let stream = assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Start {
            partial: message.clone(),
        });
        stream.push(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: message.clone(),
        });
        stream.end(Some(&message));
        Some(stream)
    }

    fn cancel_deferred<'a>(
        &'a self,
        _model: &'a Model,
        handle: &'a DeferredHandle,
        options: Option<&'a DeferredCancelOptions>,
    ) -> Option<BoxedFuture<'a, Result<(), ProviderRequestError>>> {
        let cancelled = self.cancelled.as_ref()?;
        *lock(cancelled) = Some(CancelRecord {
            id: handle.id.clone(),
            options: options.cloned(),
        });
        Some(Box::pin(async { Ok(()) }))
    }

    fn supports_fetch_deferred(&self) -> bool {
        self.fetched.is_some()
    }

    fn supports_cancel_deferred(&self) -> bool {
        self.cancelled.is_some()
    }
}

/// The auth method the registration case drives, upstream's `apiKey` block:
/// a secret-prompt login, a key-presence check, and a resolve that overrides
/// the base URL from the stored key.
fn prompt_check_resolve_auth() -> ProviderAuth {
    let login: ApiKeyLoginFn = Arc::new(|interaction: ProviderAuthInteraction| {
        Box::pin(async move {
            let key = (interaction.prompt)(AuthPrompt {
                signal: None,
                kind: AuthPromptKind::Secret {
                    message: "API key".to_owned(),
                    placeholder: None,
                },
            })
            .await
            .map_err(|error| -> AuthError { Box::new(error) })?;
            Ok(ApiKeyCredential {
                key: Some(key),
                env: None,
            })
        })
    });
    let check: ApiKeyCheckFn = Arc::new(|input: ApiKeyAuthInput| {
        Box::pin(async move {
            Ok(input
                .credential
                .as_ref()
                .and_then(|credential| credential.key.as_deref())
                .filter(|key| !key.is_empty())
                .map(|_| AuthCheck {
                    source: Some("stored native key".to_owned()),
                    auth_type: AuthType::ApiKey,
                }))
        })
    });
    let resolve: ApiKeyResolveFn = Arc::new(|input: ApiKeyAuthInput| {
        Box::pin(async move {
            Ok(input
                .credential
                .as_ref()
                .and_then(|credential| credential.key.clone())
                .filter(|key| !key.is_empty())
                .map(|key| AuthResult {
                    auth: ModelAuth {
                        api_key: Some(key),
                        base_url: Some("https://resolved.test/v1".to_owned()),
                        ..ModelAuth::default()
                    },
                    env: None,
                    source: Some("stored native key".to_owned()),
                }))
        })
    });
    ProviderAuth {
        api_key: Some(ApiKeyAuth {
            name: "Native setup".to_owned(),
            login: Some(login),
            check: Some(check),
            resolve,
        }),
        oauth: None,
    }
}

/// The auth method the deferred and overrides cases drive, upstream's
/// `resolve: async () => ({ auth: { apiKey: "key" }, source: "native" })`.
fn static_key_auth() -> ProviderAuth {
    ProviderAuth {
        api_key: Some(ApiKeyAuth {
            name: "Native key".to_owned(),
            login: None,
            check: None,
            resolve: Arc::new(|_input: ApiKeyAuthInput| {
                Box::pin(async move {
                    Ok(Some(AuthResult {
                        auth: ModelAuth {
                            api_key: Some("key".to_owned()),
                            ..ModelAuth::default()
                        },
                        env: None,
                        source: Some("native".to_owned()),
                    }))
                })
            }),
        }),
        oauth: None,
    }
}

/// Write the raw providers JSON, upstream's `writeFileSync(modelsPath, ...)`.
fn write_models_json(path: &std::path::Path, providers: &serde_json::Value) {
    std::fs::write(path, providers.to_string()).expect("the models.json write");
}

/// The runtime over a temp models.json path, the pi-ai in-memory models
/// store, and empty credentials, upstream's temp-dir `ModelRuntime.create`.
async fn temp_models_runtime(models_path: &std::path::Path) -> ModelRuntime {
    let credentials: Arc<dyn CredentialStore> = empty_auth_storage();
    let models_store: Arc<dyn ModelsStore> = Arc::new(InMemoryModelsStore::default());
    ModelRuntime::create(CreateModelRuntimeOptions {
        credentials: Some(credentials),
        models_path: Some(Some(models_path.display().to_string())),
        models_store: Some(models_store),
        allow_model_network: false,
        ..CreateModelRuntimeOptions::default()
    })
    .await
    .expect("the runtime constructs")
}

/// One refresh pass the projection cases repeat, upstream's
/// `await runtime.refresh({ allowNetwork: false })`.
async fn offline_refresh(runtime: &ModelRuntime) {
    runtime
        .refresh(ModelsRefreshOptions {
            allow_network: Some(false),
            ..ModelsRefreshOptions::default()
        })
        .await;
}

/// The registration input a fixture model carries, the inverse of the
/// composer's extension projection.
fn model_input(source: &Model) -> ProviderModelInput {
    ProviderModelInput {
        id: source.id.clone(),
        name: source.name.clone(),
        api: Some(source.api.clone()),
        base_url: Some(source.base_url.clone()),
        reasoning: source.reasoning,
        thinking_level_map: source.thinking_level_map.clone(),
        input: source.input.clone(),
        cost: source.cost.clone(),
        context_window: source.context_window,
        max_tokens: source.max_tokens,
        sampling_params: source.sampling_params.clone(),
        headers: source.headers.clone(),
        compat: source.compat.clone(),
    }
}

/// The model the legacy projection adds, upstream's `model("credential-model")`.
fn credential_model() -> Model {
    model("extension-oauth", "credential-model")
}

mod extension_provider_model_lifecycle {
    use super::*;

    /// Upstream "registers native pi-ai providers with their auth
    /// implementation": the double registers, resolves its auth through the
    /// composed surface after a login, and unregisters.
    #[tokio::test]
    async fn registers_native_pi_ai_providers_with_their_auth_implementation() {
        let registry = create_in_memory_model_registry(empty_auth_storage()).await;
        let runtime = registry.runtime().clone();

        let mut native_model = model("extension-native", "native");
        native_model.base_url = "https://fallback.test/v1".to_owned();
        let provider: Arc<dyn Provider> = Arc::new(NativeDouble::new(
            "extension-native",
            COMPOSED_NAME,
            native_model,
            prompt_check_resolve_auth(),
        ));
        registry.register_provider(Arc::clone(&provider));

        let registered = registry
            .get_registered_native_provider("extension-native")
            .expect("the native provider registers");
        assert!(Arc::ptr_eq(&registered, &provider));
        // No overlays: the registered object itself is what the registry
        // hands back, upstream's identity assertion.
        let registered_provider = registry
            .get_provider("extension-native")
            .expect("the provider registers");
        assert!(Arc::ptr_eq(&registered_provider, &provider));
        assert!(
            registry
                .get_registered_provider_ids()
                .iter()
                .any(|id| id == "extension-native"),
            "got: {:?}",
            registry.get_registered_provider_ids(),
        );
        assert!(registry.find("extension-native", "native").is_some());

        let interaction = AuthInteraction {
            signal: None,
            prompt: Arc::new(|_prompt: AuthPrompt| {
                let entered: BoxedFuture<'static, Result<String, AbortError>> =
                    Box::pin(async { Ok("secret".to_owned()) });
                entered
            }),
            notify: Arc::new(|_event: AuthEvent| {}),
        };
        runtime
            .login("extension-native", AuthType::ApiKey, interaction)
            .await
            .expect("the login commits");

        let resolution = registry
            .get_provider_auth("extension-native")
            .await
            .expect("the auth resolution succeeds")
            .expect("the provider is configured");
        assert_eq!(resolution.auth.api_key.as_deref(), Some("secret"));
        assert_eq!(
            resolution.auth.base_url.as_deref(),
            Some("https://resolved.test/v1"),
        );

        registry.unregister_provider("extension-native");
        assert!(registry.get_provider("extension-native").is_none());
        assert!(
            registry
                .get_registered_native_provider("extension-native")
                .is_none(),
        );
        assert!(registry.find("extension-native", "native").is_none());
    }

    /// Upstream "preserves native deferred methods through provider
    /// overlays": the models.json baseUrl override reaches the double's
    /// fetch, and the auth, wait, and transform headers reach both deferred
    /// methods.
    #[expect(
        clippy::too_many_lines,
        reason = "the 1:1 port carries both deferred calls and their four recorded assertions"
    )]
    #[tokio::test]
    async fn preserves_native_deferred_methods_through_provider_overlays() {
        let temp = tempdir().expect("the models.json temp dir");
        let models_path = temp.path().join("models.json");
        write_models_json(
            &models_path,
            &json!({
                "providers": {
                    "extension-native-deferred": { "baseUrl": "https://overlay.test/v1" },
                },
            }),
        );
        let runtime = temp_models_runtime(&models_path).await;

        let mut native_model = model("extension-native-deferred", "native-deferred");
        native_model.base_url = "https://native.test/v1".to_owned();
        let fetched: Arc<Mutex<Option<FetchRecord>>> = Arc::new(Mutex::new(None));
        let cancelled: Arc<Mutex<Option<CancelRecord>>> = Arc::new(Mutex::new(None));
        let provider: Arc<dyn Provider> = Arc::new(
            NativeDouble::new(
                "extension-native-deferred",
                "Extension Native Deferred",
                native_model,
                static_key_auth(),
            )
            .with_deferred_recording(Arc::clone(&fetched), Arc::clone(&cancelled)),
        );
        runtime.register_native_provider(Arc::clone(&provider));

        let composed = runtime
            .get_model("extension-native-deferred", "native-deferred")
            .expect("the composed model exists");

        let fetch_handle = DeferredHandle {
            provider: "extension-native-deferred".to_owned(),
            model_id: "native-deferred".to_owned(),
            api: "openai-completions".to_owned(),
            id: "fetch-id".to_owned(),
            expires_at: None,
            poll_after_ms: None,
            data: None,
        };
        let transform: TransformHeadersFn = Arc::new(|headers| {
            Box::pin(async move {
                let mut headers = headers;
                headers.insert("X-Transformed".to_owned(), Some("fetch".to_owned()));
                headers
            })
        });
        let fetch_options = ModelsDeferredFetchOptions {
            options: DeferredFetchOptions {
                wait: Some(25),
                headers: Some(ProviderHeaders::from([(
                    "X-Fetch".to_owned(),
                    Some("fetch".to_owned()),
                )])),
                ..DeferredFetchOptions::default()
            },
            transform_headers: Some(transform),
        };
        let message = runtime
            .fetch_deferred(&composed, &fetch_handle, Some(&fetch_options))
            .await;
        assert_eq!(message.stop_reason, StopReason::Stop);

        let cancel_handle = DeferredHandle {
            id: "cancel-id".to_owned(),
            ..fetch_handle.clone()
        };
        let cancel_transform: TransformHeadersFn = Arc::new(|headers| {
            Box::pin(async move {
                let mut headers = headers;
                headers.insert("X-Transformed".to_owned(), Some("cancel".to_owned()));
                headers
            })
        });
        let cancel_options = ModelsDeferredCancelOptions {
            options: DeferredCancelOptions {
                timeout_ms: Some(100),
                ..DeferredCancelOptions::default()
            },
            transform_headers: Some(cancel_transform),
        };
        runtime
            .cancel_deferred(&composed, &cancel_handle, Some(&cancel_options))
            .await
            .expect("the cancellation commits");

        let fetched = lock(&fetched)
            .clone()
            .expect("the fetch reached the provider");
        assert_eq!(fetched.base_url, "https://overlay.test/v1");
        let fetch_options = fetched.options.expect("the fetch carried options");
        assert_eq!(fetch_options.api_key.as_deref(), Some("key"));
        assert_eq!(fetch_options.wait, Some(25));
        let headers = fetch_options.headers.expect("the fetch carried headers");
        assert_eq!(
            headers.get("X-Fetch").map(|value| value.as_deref()),
            Some(Some("fetch")),
        );
        assert_eq!(
            headers.get("X-Transformed").map(|value| value.as_deref()),
            Some(Some("fetch")),
        );

        let cancelled = lock(&cancelled)
            .clone()
            .expect("the cancel reached the provider");
        assert_eq!(cancelled.id, "cancel-id");
        let cancel_options = cancelled.options.expect("the cancel carried options");
        assert_eq!(cancel_options.api_key.as_deref(), Some("key"));
        assert_eq!(cancel_options.timeout_ms, Some(100));
        let headers = cancel_options.headers.expect("the cancel carried headers");
        assert_eq!(
            headers.get("X-Transformed").map(|value| value.as_deref()),
            Some(Some("cancel")),
        );
    }

    /// Upstream "applies models.json overrides above native providers": the
    /// composed model carries the override's context window.
    #[tokio::test]
    async fn applies_models_json_overrides_above_native_providers() {
        let temp = tempdir().expect("the models.json temp dir");
        let models_path = temp.path().join("models.json");
        write_models_json(
            &models_path,
            &json!({
                "providers": {
                    "extension-native": {
                        "modelOverrides": {
                            "native": { "contextWindow": 4242 },
                        },
                    },
                },
            }),
        );
        let runtime = temp_models_runtime(&models_path).await;

        let mut native_model = model("extension-native", "native");
        native_model.base_url = "https://native.test/v1".to_owned();
        runtime.register_native_provider(Arc::new(NativeDouble::new(
            "extension-native",
            COMPOSED_NAME,
            native_model,
            static_key_auth(),
        )));

        assert_eq!(
            runtime
                .get_model("extension-native", "native")
                .expect("the native model registers")
                .context_window,
            4242,
        );
    }

    /// Upstream "publishes refreshModels results without forcing `ModelsStore`
    /// persistence": the refreshed model composes in and the store stays
    /// empty for the provider.
    #[tokio::test]
    async fn publishes_refresh_models_results_without_forcing_models_store_persistence() {
        let models_store: Arc<dyn ModelsStore> = Arc::new(InMemoryModelsStore::default());
        let credentials: Arc<dyn CredentialStore> = empty_auth_storage();
        let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
            credentials: Some(credentials),
            models_path: Some(None),
            models_store: Some(Arc::clone(&models_store)),
            allow_model_network: false,
            ..CreateModelRuntimeOptions::default()
        })
        .await
        .expect("the runtime constructs");

        let mut live = model("extension-dynamic", "live");
        live.base_url = "http://localhost:8080/v1".to_owned();
        let refresh_models: ExtensionRefreshModelsFn =
            Arc::new(move |_context: &RefreshModelsContext| {
                let live = live.clone();
                Box::pin(async move { Ok(vec![live]) })
            });
        runtime
            .register_provider(
                "extension-dynamic",
                ProviderConfigInput {
                    base_url: Some("http://localhost:8080/v1".to_owned()),
                    api_key: Some("local".to_owned()),
                    api: Some(Api::from("openai-completions")),
                    refresh_models: Some(refresh_models),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");

        let mut published = false;
        for _ in 0..64 {
            offline_refresh(&runtime).await;
            if runtime.get_model("extension-dynamic", "live").is_some() {
                published = true;
                break;
            }
        }
        assert!(published, "the refreshed model never published");

        let stored = models_store
            .read("extension-dynamic", None)
            .await
            .expect("the store read");
        assert!(stored.is_none(), "the refresh never persisted to the store");
    }

    /// Upstream "applies legacy OAuth modifyModels after async credential
    /// initialization": the stored OAuth credential projects the extra model
    /// at refresh time and logout drops it again.
    #[tokio::test]
    async fn applies_legacy_oauth_modify_models_after_async_credential_initialization() {
        let seed = json!({
            "extension-oauth": {
                "type": "oauth",
                "access": "access",
                "refresh": "refresh",
                "expires": pi_ai::auth::resolve::now_ms() + 60_000,
            },
        });
        let data: AuthStorageData = seed
            .as_object()
            .cloned()
            .expect("the seed is a credential map");
        let registry = create_in_memory_model_registry(in_memory_auth_storage(&data)).await;
        let runtime = registry.runtime().clone();

        let oauth = ExtensionOAuthConfig {
            name: "Extension OAuth".to_owned(),
            is_subscription: None,
            uses_callback_server: None,
            login: Arc::new(|_callbacks| {
                let refused: ProviderError = Box::new(std::io::Error::other("not used"));
                let outcome: BoxedFuture<'static, Result<OAuthCredentials, ProviderError>> =
                    Box::pin(async move { Err(refused) });
                outcome
            }),
            refresh_token: Arc::new(
                |credentials: OAuthCredentials, _signal: CancellationToken| {
                    Box::pin(async move { Ok(credentials) })
                },
            ),
            get_api_key: Arc::new(|credential: &OAuthCredentials| credential.access.clone()),
            modify_models: Some(Arc::new(
                |models: Vec<Model>, credential: &OAuthCredentials| {
                    if credential.access == "access" {
                        let mut models = models;
                        models.push(credential_model());
                        models
                    } else {
                        models
                    }
                },
            )),
        };
        registry
            .register_provider_config(
                "extension-oauth",
                ProviderConfigInput {
                    base_url: Some("https://example.test/v1".to_owned()),
                    api: Some(Api::from("openai-completions")),
                    models: Some(vec![model_input(&model("extension-oauth", "base"))]),
                    oauth: Some(oauth),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");

        let mut published = false;
        for _ in 0..64 {
            offline_refresh(&runtime).await;
            if runtime
                .get_model("extension-oauth", "credential-model")
                .is_some()
                && runtime.get_model("extension-oauth", "base").is_some()
            {
                published = true;
                break;
            }
        }
        assert!(published, "the credential-projected model never published");

        runtime
            .logout("extension-oauth", None)
            .await
            .expect("the logout commits");

        let mut cleared = 0;
        let mut gone = false;
        for _ in 0..64 {
            offline_refresh(&runtime).await;
            if runtime
                .get_model("extension-oauth", "credential-model")
                .is_none()
            {
                cleared += 1;
                if cleared >= 2 {
                    gone = true;
                    break;
                }
            } else {
                cleared = 0;
            }
        }
        assert!(gone, "the credential-projected model never cleared");
    }
}

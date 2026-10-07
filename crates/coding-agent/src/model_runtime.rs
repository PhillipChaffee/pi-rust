//! The configured pi-ai Models collection used by coding-agent and SDK
//! consumers, upstream's `src/core/model-runtime.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: the runtime's state lives behind one `Arc` the
//! lazy-stream setups capture, the port of upstream's `this` binding inside
//! the promise chains. The per-provider credential-operation chain restates
//! as a tokio mutex per provider id — each operation acquires the tail,
//! aborts when its signal fired before the wait ended, and runs holding the
//! lock, upstream's `await previous.catch(() => {})` +
//! `signal.throwIfAborted()`. The availability snapshot rides one mutex the
//! sequence counters guard: a stale refresh pass never publishes, and the
//! per-provider auth checks walk the provider list sequentially on the
//! availability task (upstream's `Promise.all` concurrency affects latency
//! only; the snapshot publication order is unchanged). The create-time
//! refresh timeout arms a fresh token watched against the caller's, the
//! port of `AbortSignal.any` plus `setTimeout`. The fire-and-forget
//! refreshes after provider registration run on spawned tasks the way
//! upstream's `void this.refresh(...)` did.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::resolve::{ModelsError, ModelsErrorCode, ModelsFailure};
use pi_ai::auth::types::AuthError;
use pi_ai::auth::types::{
    ApiKeyCredential, AuthInteraction, AuthOptions, AuthResult, AuthType, Credential,
    CredentialInfo,
};
use pi_ai::models::{
    ModelsDeferredCancelOptions, ModelsDeferredFetchOptions, ModelsRefreshOptions,
    ModelsRefreshResult, ModelsSimpleStreamOptions, ModelsStreamOptions, Provider, WithTransforms,
    create_models,
};
use pi_ai::models_store::ModelsStore;
use pi_ai::providers::all::builtin_providers;
use pi_ai::providers::catalog::get_builtin_model_data_generated_at;
use pi_ai::providers::radius::{RadiusProviderOptions, radius_provider};
use pi_ai::types::{
    AssistantMessage, BoxedFuture, Context, DeferredHandle, Model, ProviderEnv, ProviderHeaders,
};
use pi_ai::utils::abort::{AbortError, operation_signal};
use pi_ai::utils::event_stream::AssistantMessageEventStream;

use crate::auth_storage::AuthStorage;
use crate::config::get_agent_dir;
use crate::model_config::ModelConfig;
use crate::models_store::{FileModelsStore, InMemoryCodingAgentModelsStore};
use crate::provider_composer::{
    AuthStatus, AuthStatusSource, CompatibilityRequestConfig, ProviderConfigInput,
    compose_model_provider, configured_request_auth_status, resolve_compatibility_request_config,
    resolve_configured_model_headers, validate_extension_provider,
};
use crate::radius::RADIUS_PROVIDER_ID;
use crate::remote_catalog_provider::with_remote_catalog_client;
use crate::runtime_credentials::RuntimeCredentials;

/// The runtime snapshot, upstream's `ModelRuntimeSnapshot`.
#[derive(Debug, Default, Clone)]
struct ModelRuntimeSnapshot {
    all: Vec<Model>,
    available: Vec<Model>,
    configured_providers: BTreeSet<String>,
    stored_providers: BTreeSet<String>,
    auth: BTreeMap<String, Option<pi_ai::auth::types::AuthCheck>>,
}

/// The construction options, upstream's `CreateModelRuntimeOptions`.
#[derive(Clone, Default)]
pub struct CreateModelRuntimeOptions {
    /// Credential storage; defaults to the file store at `auth_path`.
    pub credentials: Option<Arc<dyn CredentialStore>>,
    /// The auth.json path the default credential store uses.
    pub auth_path: Option<String>,
    /// The models.json path: `None` means the default
    /// `<agent-dir>/models.json`, `Some(None)` disables the file entirely.
    pub models_path: Option<Option<String>>,
    /// The persisted-catalog store; defaults to the file beside the
    /// models.json path, or in-memory when models.json is disabled.
    pub models_store: Option<Arc<dyn ModelsStore>>,
    /// The models-store.json path the default file store uses.
    pub models_store_path: Option<String>,
    /// Allow `create` to refresh model catalogs over the network. Defaults
    /// to false.
    pub allow_model_network: bool,
    /// Timeout for the create-time network model refresh.
    pub model_refresh_timeout_ms: Option<u64>,
    /// The pi.dev catalog base URL override.
    pub catalog_base_url: Option<String>,
    /// Optional caller cancellation for initial cache restoration and
    /// availability checks.
    pub signal: Option<CancellationToken>,
    /// Skip initial catalog and availability refresh. Static models remain
    /// available. Defaults to true.
    pub refresh_on_create: Option<bool>,
}

/// The auth overrides a request-time resolution takes, upstream's
/// `ModelRuntimeAuthOverrides`.
#[derive(Clone, Debug, Default)]
pub struct ModelRuntimeAuthOverrides {
    /// Use this API key instead of resolving one.
    pub api_key: Option<String>,
    /// Overlay these provider-scoped environment values over the auth
    /// context.
    pub env: Option<ProviderEnv>,
    /// Require this much remaining OAuth-token validity; defaults to five
    /// minutes.
    pub min_oauth_validity_ms: Option<i64>,
    /// Cancellation for the resolution.
    pub signal: Option<CancellationToken>,
}

/// The credential operations the runtime serializes per provider, upstream's
/// `CredentialSynchronizationOperation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CredentialSynchronizationOperation {
    /// A login flow committed.
    Login,
    /// A logout committed.
    Logout,
    /// A runtime API key was set.
    SetRuntimeApiKey,
    /// A runtime API key was removed.
    RemoveRuntimeApiKey,
}

impl std::fmt::Display for CredentialSynchronizationOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Login => "login",
            Self::Logout => "logout",
            Self::SetRuntimeApiKey => "setRuntimeApiKey",
            Self::RemoveRuntimeApiKey => "removeRuntimeApiKey",
        })
    }
}

/// Credentials changed successfully, but the local model/auth snapshot could
/// not be synchronized, upstream's `CredentialSynchronizationError`.
#[derive(Debug)]
pub struct CredentialSynchronizationError {
    /// The provider whose credential changed.
    pub provider_id: String,
    /// The operation that committed.
    pub operation: CredentialSynchronizationOperation,
    /// The credential that committed, when the operation carried one.
    pub credential: Option<Credential>,
    /// The synchronization failure that aborted the local refresh.
    pub cause: Box<dyn std::error::Error + Send + Sync>,
}

impl std::fmt::Display for CredentialSynchronizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Credential {} committed for {}, but local synchronization failed",
            self.operation, self.provider_id
        )
    }
}

impl std::error::Error for CredentialSynchronizationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause.as_ref())
    }
}

/// Merge header maps case-insensitively, upstream's `mergeHeaders`: an
/// override name replaces every existing name that matches
/// case-insensitively.
fn merge_headers(
    base: Option<&ProviderHeaders>,
    override_headers: Option<&ProviderHeaders>,
) -> Option<ProviderHeaders> {
    let Some(override_headers) = override_headers else {
        return base.cloned();
    };
    let mut merged = base.cloned().unwrap_or_default();
    for (name, value) in override_headers {
        let lower_name = name.to_lowercase();
        let keys: Vec<String> = merged
            .keys()
            .filter(|existing| existing.to_lowercase() == lower_name)
            .cloned()
            .collect();
        for key in keys {
            merged.remove(&key);
        }
        merged.insert(name.clone(), value.clone());
    }
    Some(merged)
}

/// Box one error into the boxed failure the sync operations carry; the
/// unsizing coercion happens in this function's return position, so call
/// sites stay cast-free.
fn boxed_error(
    error: impl std::error::Error + Send + Sync + 'static,
) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(error)
}

/// The `provider\0id` availability key, upstream's map key.
fn model_key(model: &Model) -> String {
    format!("{}\0{}", model.provider.0, model.id)
}

/// The mutable runtime state, upstream's private fields grouped behind one
/// mutex.
struct RuntimeState {
    config: ModelConfig,
    snapshot: ModelRuntimeSnapshot,
    composition_errors: BTreeMap<String, String>,
    availability_refresh_seq: u64,
    availability_error_seq: u64,
    provider_availability_seq: BTreeMap<String, u64>,
    availability_error: Option<String>,
}

/// The provider collections the runtime composes from, upstream's
/// `defaultBuiltins`/`builtins`/`nativeExtensionProviders`/
/// `extensionProviders` maps.
#[derive(Default)]
struct ProviderCollections {
    default_builtins: BTreeMap<String, Arc<dyn Provider>>,
    builtins: BTreeMap<String, Arc<dyn Provider>>,
    native_extension_providers: BTreeMap<String, Arc<dyn Provider>>,
    extension_providers: BTreeMap<String, Arc<ProviderConfigInput>>,
}

/// The per-provider serialized credential chain, upstream's
/// `credentialOperations` promise queue.
struct CredentialChain(tokio::sync::Mutex<()>);

/// The shared runtime core, the `Arc` the lazy-stream setups capture.
pub struct ModelRuntimeCore {
    models: pi_ai::models::Models,
    credentials: Arc<RuntimeCredentials>,
    collections: Mutex<ProviderCollections>,
    state: Mutex<RuntimeState>,
    models_path: Option<String>,
    model_network_enabled: bool,
    credential_chains: Mutex<BTreeMap<String, Arc<CredentialChain>>>,
    /// The owning handle, the seam the spawned stream setups and background
    /// refreshes capture instead of `self`.
    self_weak: std::sync::Weak<Self>,
}

impl std::fmt::Debug for ModelRuntimeCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelRuntimeCore").finish_non_exhaustive()
    }
}

/// The configured pi-ai Models collection, upstream's `ModelRuntime`. State
/// lives behind one `Arc` so spawned stream setups capture it.
#[derive(Clone)]
pub struct ModelRuntime(Arc<ModelRuntimeCore>);

impl std::fmt::Debug for ModelRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::ops::Deref for ModelRuntime {
    type Target = ModelRuntimeCore;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ModelRuntime {
    /// Build the runtime, upstream's `ModelRuntime.create`.
    ///
    /// # Errors
    /// A models.json path that does not normalize or fails to read, or an
    /// auth store the default credential path cannot open.
    #[expect(
        clippy::too_many_lines,
        reason = "the 1:1 port of upstream's create carries the signal-composition ladder inline"
    )]
    pub async fn create(options: CreateModelRuntimeOptions) -> Result<Self, AuthError> {
        let base: Arc<dyn CredentialStore> = if let Some(credentials) = options.credentials {
            credentials
        } else {
            let auth_path = options
                .auth_path
                .clone()
                .unwrap_or_else(|| get_agent_dir().join("auth.json").display().to_string());
            Arc::new(AuthStorage::create(&auth_path)?)
        };
        let runtime_credentials = Arc::new(RuntimeCredentials::new(Arc::clone(&base)));
        let credentials_trait: Arc<dyn CredentialStore> = runtime_credentials.clone();
        let models_path: Option<String> = match &options.models_path {
            Some(None) => None,
            Some(Some(path)) => Some(path.clone()),
            None => Some(get_agent_dir().join("models.json").display().to_string()),
        };
        let config = ModelConfig::load(models_path.as_deref())
            .map_err(|error| AuthError::from(std::io::Error::other(error)))?;
        let models_store: Arc<dyn ModelsStore> = match &options.models_store {
            Some(store) => Arc::clone(store),
            None => match &models_path {
                Some(models_path) => {
                    let store_path = options.models_store_path.clone().unwrap_or_else(|| {
                        std::path::Path::new(models_path)
                            .parent()
                            .unwrap_or_else(|| std::path::Path::new("."))
                            .join("models-store.json")
                            .display()
                            .to_string()
                    });
                    Arc::new(FileModelsStore::new(Some(&store_path))?)
                }
                None => Arc::new(InMemoryCodingAgentModelsStore::default()),
            },
        };
        let builtin_model_data_generated_at = get_builtin_model_data_generated_at();
        let catalog_base_url = options.catalog_base_url.clone();
        let client = pi_ai::http::default_http_client();
        let providers = builtin_providers()
            .into_iter()
            .map(|provider| {
                if provider.id() == RADIUS_PROVIDER_ID {
                    provider
                } else {
                    with_remote_catalog_client(
                        provider,
                        catalog_base_url.clone(),
                        builtin_model_data_generated_at,
                        Arc::clone(&client),
                    )
                }
            })
            .collect::<Vec<_>>();
        let model_network_enabled = std::env::var_os("PI_OFFLINE").is_none();
        let models = create_models(Some(pi_ai::models::CreateModelsOptions {
            credentials: Some(Arc::clone(&credentials_trait)),
            models_store: Some(Arc::clone(&models_store)),
            auth_context: None,
        }));
        let mut default_builtins = BTreeMap::new();
        let mut builtins = BTreeMap::new();
        for provider in providers {
            default_builtins.insert(provider.id().to_owned(), Arc::clone(&provider));
            builtins.insert(provider.id().to_owned(), provider);
        }
        let core = Arc::new_cyclic(|self_weak| ModelRuntimeCore {
            models,
            credentials: Arc::clone(&runtime_credentials),
            collections: Mutex::new(ProviderCollections {
                default_builtins,
                builtins,
                native_extension_providers: BTreeMap::new(),
                extension_providers: BTreeMap::new(),
            }),
            state: Mutex::new(RuntimeState {
                config,
                snapshot: ModelRuntimeSnapshot::default(),
                composition_errors: BTreeMap::new(),
                availability_refresh_seq: 0,
                availability_error_seq: 0,
                provider_availability_seq: BTreeMap::new(),
                availability_error: None,
            }),
            models_path,
            model_network_enabled,
            credential_chains: Mutex::new(BTreeMap::new()),
            self_weak: self_weak.clone(),
        });
        let runtime = Self(core);
        runtime.configure_radius_providers();
        runtime.rebuild_providers();
        let refresh_from_network = runtime.model_network_enabled && options.allow_model_network;
        let signal = match (&options.signal, options.model_refresh_timeout_ms) {
            (Some(caller), Some(timeout_ms)) if refresh_from_network => {
                let timeout_token = CancellationToken::new();
                let watcher = timeout_token.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
                    watcher.cancel();
                });
                let merged = CancellationToken::new();
                let watcher_signal = merged.clone();
                let watcher_caller = caller.clone();
                let watcher_timeout = timeout_token;
                tokio::spawn(async move {
                    tokio::select! {
                        () = watcher_caller.cancelled() => watcher_signal.cancel(),
                        () = watcher_timeout.cancelled() => watcher_signal.cancel(),
                    }
                });
                Some(merged)
            }
            (Some(caller), _) => Some(caller.clone()),
            (None, Some(timeout_ms)) if refresh_from_network => {
                let timeout_token = CancellationToken::new();
                let watcher = timeout_token.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
                    watcher.cancel();
                });
                Some(timeout_token)
            }
            (None, _) => None,
        };
        if options.refresh_on_create.unwrap_or(true) {
            runtime
                .refresh(ModelsRefreshOptions {
                    allow_network: Some(refresh_from_network),
                    signal,
                    ..ModelsRefreshOptions::default()
                })
                .await;
        }
        Ok(runtime)
    }
}

impl ModelRuntimeCore {
    /// Promote configured Radius gateways from models.json to builtin
    /// providers, upstream's `configureRadiusProviders`.
    fn configure_radius_providers(&self) {
        let config = self.models_config();
        let mut collections = self
            .collections
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        collections.builtins.clear();
        let defaults: Vec<(String, Arc<dyn Provider>)> = collections
            .default_builtins
            .iter()
            .map(|(provider_id, provider)| (provider_id.clone(), Arc::clone(provider)))
            .collect();
        for (provider_id, provider) in defaults {
            collections.builtins.insert(provider_id, provider);
        }
        for provider_id in config.get_provider_ids() {
            let Some(config) = config.get_provider(&provider_id) else {
                continue;
            };
            if config.oauth.as_deref() != Some("radius") {
                continue;
            }
            let Some(base_url) = &config.base_url else {
                continue;
            };
            let trimmed = base_url.strip_suffix('/').unwrap_or(base_url);
            let gateway = trimmed.strip_suffix("/v1").unwrap_or(trimmed);
            collections.builtins.insert(
                provider_id.clone(),
                radius_provider(RadiusProviderOptions {
                    id: Some(provider_id.clone()),
                    name: Some(config.name.clone().unwrap_or_else(|| provider_id.clone())),
                    gateway: Some(gateway.to_owned()),
                }),
            );
        }
    }

    /// Every provider id the composition layers know about, upstream's
    /// `providerIds()`.
    fn provider_ids(&self) -> BTreeSet<String> {
        let collections = self
            .collections
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config_ids = self.models_config().get_provider_ids();
        collections
            .builtins
            .keys()
            .cloned()
            .chain(collections.native_extension_providers.keys().cloned())
            .chain(config_ids)
            .chain(collections.extension_providers.keys().cloned())
            .collect()
    }

    /// Recompose one provider from its layers, upstream's
    /// `recomposeProvider`.
    fn recompose_provider(&self, provider_id: &str) {
        let base = {
            let collections = self
                .collections
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            collections
                .native_extension_providers
                .get(provider_id)
                .cloned()
                .or_else(|| collections.builtins.get(provider_id).cloned())
        };
        let extension = {
            let collections = self
                .collections
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            collections.extension_providers.get(provider_id).cloned()
        };
        let config = self.models_config();
        if base.is_none() && config.get_provider(provider_id).is_none() && extension.is_none() {
            self.models.delete_provider(provider_id);
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .composition_errors
                .remove(provider_id);
            return;
        }
        if let Some(base) = &base
            && config.get_provider(provider_id).is_none()
            && extension.is_none()
        {
            // No overlays: the builtin streams untouched so its
            // auth/login/stream behavior is exact.
            self.models.set_provider(Arc::clone(base));
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .composition_errors
                .remove(provider_id);
            return;
        }
        let composed = compose_model_provider(
            provider_id,
            base.clone(),
            &config,
            extension.as_ref().map(|extension| (**extension).clone()),
        );
        match composed {
            Ok(provider) => {
                self.models.set_provider(provider);
                self.state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .composition_errors
                    .remove(provider_id);
            }
            Err(error) => {
                self.state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .composition_errors
                    .insert(provider_id.to_owned(), error);
                match base {
                    Some(base) => self.models.set_provider(base),
                    None => self.models.delete_provider(provider_id),
                }
            }
        }
    }

    /// Recompose every provider, upstream's `rebuildProviders`.
    fn rebuild_providers(&self) {
        self.models.clear_providers();
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .composition_errors
            .clear();
        for provider_id in self.provider_ids() {
            self.recompose_provider(&provider_id);
        }
        self.update_model_snapshot();
    }

    /// Refresh the model list from the current providers, upstream's
    /// `updateModelSnapshot`.
    fn update_model_snapshot(&self) {
        let all = self.models.models(None);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.snapshot.all.clone_from(&all);
        let configured = state.snapshot.configured_providers.clone();
        state.snapshot.available = all
            .into_iter()
            .filter(|model| configured.contains(&model.provider.0))
            .collect();
    }

    /// The models.json snapshot.
    #[must_use]
    pub fn models_config(&self) -> ModelConfig {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .config
            .clone()
    }

    /// The full availability pass, upstream's `runAvailabilityRefresh`.
    async fn run_availability_refresh(&self, seq: u64, error_seq: u64, signal: CancellationToken) {
        let providers = self.models.providers();
        let mut checks = Vec::with_capacity(providers.len());
        for provider in &providers {
            let check = self
                .models
                .check_auth(
                    provider.id(),
                    Some(&AuthOptions {
                        signal: Some(signal.clone()),
                    }),
                )
                .await;
            checks.push((provider.id().to_owned(), check.ok().flatten()));
        }
        let available = self
            .models
            .available(
                None,
                Some(&AuthOptions {
                    signal: Some(signal.clone()),
                }),
            )
            .await;
        let credentials = self
            .credentials
            .list(Some(&AuthOptions {
                signal: Some(signal.clone()),
            }))
            .await;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if seq != state.availability_refresh_seq {
            return;
        }
        let configured_providers: BTreeSet<String> = checks
            .iter()
            .filter(|(_, check)| check.is_some())
            .map(|(provider_id, _)| provider_id.clone())
            .collect();
        state.snapshot = ModelRuntimeSnapshot {
            all: self.models.models(None),
            available: available.unwrap_or_default(),
            configured_providers,
            stored_providers: credentials
                .map(|credentials| {
                    credentials
                        .iter()
                        .map(|entry| entry.provider_id.clone())
                        .collect()
                })
                .unwrap_or_default(),
            auth: checks.into_iter().collect(),
        };
        if error_seq == state.availability_error_seq {
            state.availability_error = None;
        }
    }

    /// Queue an availability pass, upstream's `queueAvailabilityRefresh`.
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the sequence bookkeeping must run under one state lock, upstream's atomic snapshot"
    )]
    fn queue_availability_refresh(
        &self,
        signal: Option<&CancellationToken>,
    ) -> BoxedFuture<'_, Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let effective_signal = operation_signal(signal);
        let (seq, error_seq) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.availability_refresh_seq += 1;
            let seq = state.availability_refresh_seq;
            state.availability_error_seq += 1;
            let error_seq = state.availability_error_seq;
            for provider_seq in state.provider_availability_seq.values_mut() {
                *provider_seq += 1;
            }
            (seq, error_seq)
        };
        Box::pin(async move {
            self.run_availability_refresh(seq, error_seq, effective_signal)
                .await;
            Ok(())
        })
    }

    /// One provider's availability refresh, upstream's
    /// `refreshProviderAvailability`.
    #[expect(
        clippy::too_many_lines,
        reason = "the 1:1 port of upstream's provider-scoped pass carries the sequence bookkeeping inline"
    )]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the sequence bookkeeping must run under one state lock, upstream's atomic snapshot"
    )]
    async fn refresh_provider_availability(
        &self,
        provider_id: &str,
        signal: CancellationToken,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Invalidate any full availability pass that started before this
        // credential change.
        let (provider_seq, error_seq) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.availability_refresh_seq += 1;
            let provider_seq = state
                .provider_availability_seq
                .entry(provider_id.to_owned())
                .or_default();
            *provider_seq += 1;
            let provider_seq = *provider_seq;
            state.availability_error_seq += 1;
            let error_seq = state.availability_error_seq;
            (provider_seq, error_seq)
        };
        let available = self
            .models
            .available(
                Some(provider_id),
                Some(&AuthOptions {
                    signal: Some(signal.clone()),
                }),
            )
            .await;
        let auth = self
            .models
            .check_auth(
                provider_id,
                Some(&AuthOptions {
                    signal: Some(signal.clone()),
                }),
            )
            .await;
        let credential = self
            .credentials
            .read(
                provider_id,
                Some(&AuthOptions {
                    signal: Some(signal.clone()),
                }),
            )
            .await;
        if signal.is_cancelled() {
            return Err(Box::new(AbortError));
        }
        let available: Result<Vec<Model>, Box<dyn std::error::Error + Send + Sync>> =
            available.map_err(boxed_error);
        let auth: Result<
            Option<pi_ai::auth::types::AuthCheck>,
            Box<dyn std::error::Error + Send + Sync>,
        > = auth.map_err(boxed_error);
        let (available, auth, credential) = match (available, auth, credential) {
            (Ok(available), Ok(auth), Ok(credential)) => (available, auth, credential),
            (available, auth, credential) => {
                let error = available
                    .err()
                    .or_else(|| auth.err())
                    .or_else(|| credential.err())
                    .unwrap_or_else(|| Box::new(AbortError));
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if state
                    .provider_availability_seq
                    .get(provider_id)
                    .is_some_and(|current| *current == provider_seq)
                    && error_seq == state.availability_error_seq
                    && !signal.is_cancelled()
                {
                    state.availability_error = Some(error.to_string());
                }
                return Err(error);
            }
        };
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state
                .provider_availability_seq
                .get(provider_id)
                .is_some_and(|current| *current != provider_seq)
            {
                return Ok(());
            }
            let mut configured_providers = state.snapshot.configured_providers.clone();
            let mut stored_providers = state.snapshot.stored_providers.clone();
            let mut auth_by_provider = state.snapshot.auth.clone();
            if let Some(auth) = &auth {
                configured_providers.insert(provider_id.to_owned());
                auth_by_provider.insert(provider_id.to_owned(), Some(auth.clone()));
            } else {
                configured_providers.remove(provider_id);
                auth_by_provider.remove(provider_id);
            }
            match &credential {
                Some(_) => {
                    stored_providers.insert(provider_id.to_owned());
                }
                None => {
                    stored_providers.remove(provider_id);
                }
            }
            let all = self.models.models(None);
            let mut available_by_id: BTreeMap<String, Model> = state
                .snapshot
                .available
                .iter()
                .filter(|model| model.provider.0 != provider_id)
                .map(|model| (model_key(model), model.clone()))
                .collect();
            for model in available {
                available_by_id.insert(model_key(&model), model);
            }
            state.snapshot = ModelRuntimeSnapshot {
                available: all
                    .iter()
                    .filter_map(|model| available_by_id.get(&model_key(model)).cloned())
                    .collect(),
                all,
                configured_providers,
                stored_providers,
                auth: auth_by_provider,
            };
            if error_seq == state.availability_error_seq {
                state.availability_error = None;
            }
        }
        Ok(())
    }

    /// The credential chain for a provider, building one on demand.
    fn credential_chain(&self, provider_id: &str) -> Arc<CredentialChain> {
        self.credential_chains
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(provider_id.to_owned())
            .or_insert_with(|| Arc::new(CredentialChain(tokio::sync::Mutex::new(()))))
            .clone()
    }

    /// Serialize a credential operation behind the provider's chain,
    /// upstream's `enqueueCredentialOperation`.
    fn enqueue_credential_operation<T, F>(
        &self,
        provider_id: &str,
        signal: CancellationToken,
        task: F,
    ) -> BoxedFuture<'static, Result<T, Box<dyn std::error::Error + Send + Sync>>>
    where
        F: FnOnce() -> BoxedFuture<'static, Result<T, Box<dyn std::error::Error + Send + Sync>>>
            + Send
            + 'static,
    {
        let chain = self.credential_chain(provider_id);
        Box::pin(async move {
            let _guard = chain.0.lock().await;
            if signal.is_cancelled() {
                return Err(boxed_error(AbortError));
            }
            task().await
        })
    }

    /// Synchronize the local snapshot after a credential commit, upstream's
    /// `synchronizeCredentialState`.
    async fn synchronize_credential_state(
        &self,
        provider_id: &str,
        operation: CredentialSynchronizationOperation,
        credential: Option<Credential>,
        signal: CancellationToken,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let outcome: Result<(), Box<dyn std::error::Error + Send + Sync>> = async {
            if signal.is_cancelled() {
                return Err(boxed_error(AbortError));
            }
            self.recompose_provider(provider_id);
            let composition_error = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .composition_errors
                .get(provider_id)
                .cloned();
            if let Some(composition_error) = composition_error {
                return Err(boxed_error(std::io::Error::other(composition_error)));
            }
            let result = self
                .models
                .refresh(Some(&ModelsRefreshOptions {
                    allow_network: Some(false),
                    providers: Some(vec![provider_id.to_owned()]),
                    signal: Some(signal.clone()),
                    ..ModelsRefreshOptions::default()
                }))
                .await;
            if result.aborted && signal.is_cancelled() {
                return Err(boxed_error(AbortError));
            }
            if let Some(refresh_error) = result.errors.get(provider_id) {
                return Err(boxed_error(std::io::Error::other(
                    refresh_error.to_string(),
                )));
            }
            self.update_model_snapshot();
            self.refresh_provider_availability(provider_id, signal.clone())
                .await?;
            Ok(())
        }
        .await;
        outcome.map_err(|cause| {
            boxed_error(CredentialSynchronizationError {
                provider_id: provider_id.to_owned(),
                operation,
                credential,
                cause,
            })
        })
    }

    /// The models.json snapshot, for the resolver's runtime views.
    #[must_use]
    pub fn get_models_config(&self) -> ModelConfig {
        self.models_config()
    }

    /// The registered config an extension carries, upstream's
    /// `getRegisteredProviderConfig`.
    #[must_use]
    pub fn get_registered_provider_config(
        &self,
        provider_id: &str,
    ) -> Option<Arc<ProviderConfigInput>> {
        self.collections
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extension_providers
            .get(provider_id)
            .cloned()
    }

    /// The ids extensions registered, upstream's `getRegisteredProviderIds`.
    #[must_use]
    pub fn get_registered_provider_ids(&self) -> Vec<String> {
        let collections = self
            .collections
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        collections
            .extension_providers
            .keys()
            .cloned()
            .chain(collections.native_extension_providers.keys().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// The native provider object an extension registered, upstream's
    /// `getRegisteredNativeProvider`.
    #[must_use]
    pub fn get_registered_native_provider(&self, provider_id: &str) -> Option<Arc<dyn Provider>> {
        self.collections
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .native_extension_providers
            .get(provider_id)
            .cloned()
    }

    /// The compatibility fallback request config, upstream's
    /// `getCompatibilityRequestConfig`.
    ///
    /// # Errors
    /// A header value that fails to resolve.
    pub fn get_compatibility_request_config(
        &self,
        model: &Model,
    ) -> Result<CompatibilityRequestConfig, String> {
        let (config, extension) = self.composition_layers(&model.provider.0);
        resolve_compatibility_request_config(
            model,
            config.as_ref(),
            extension.as_ref().map(AsRef::as_ref),
        )
    }

    /// One provider's models.json config and extension registration, the
    /// pair the composition lookups need.
    fn composition_layers(
        &self,
        provider_id: &str,
    ) -> (
        Option<crate::model_config::ModelsJsonProvider>,
        Option<Arc<ProviderConfigInput>>,
    ) {
        let collections = self
            .collections
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (
            self.models_config().get_provider(provider_id).cloned(),
            collections.extension_providers.get(provider_id).cloned(),
        )
    }

    /// Whether the provider's active credential is OAuth, upstream's
    /// `isUsingOAuth`.
    #[must_use]
    pub fn is_using_oauth(&self, provider_id: &str) -> bool {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state
            .snapshot
            .auth
            .get(provider_id)
            .and_then(Option::as_ref)
            .is_some_and(|check| check.auth_type == AuthType::OAuth)
    }

    /// Whether the provider's OAuth credential is a subscription, upstream's
    /// `isUsingSubscription`.
    #[must_use]
    pub fn is_using_subscription(&self, provider_id: &str) -> bool {
        let oauth = self
            .models
            .provider(provider_id)
            .and_then(|provider| provider.auth().oauth.clone());
        self.is_using_oauth(provider_id)
            && oauth
                .and_then(|oauth| oauth.is_subscription)
                .unwrap_or(false)
    }

    /// Whether the provider has configured auth, upstream's
    /// `hasConfiguredAuth`.
    #[must_use]
    pub fn has_configured_auth(&self, provider_id: &str) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .snapshot
            .configured_providers
            .contains(provider_id)
    }

    /// Resolve provider-scoped auth, upstream's `getAuth(providerId)`.
    ///
    /// # Errors
    /// The abort failure, or the wrapped [`ModelsError`] of the failing step.
    pub fn get_auth(
        &self,
        provider_id: &str,
        overrides: Option<&ModelRuntimeAuthOverrides>,
    ) -> BoxedFuture<'_, Result<Option<AuthResult>, ModelsFailure>> {
        self.models.get_auth(
            provider_id,
            overrides.map(as_auth_resolution_overrides).as_ref(),
        )
    }

    /// Resolve auth for a model and merge the composed layers' configured
    /// headers, upstream's `getAuth(model)`.
    ///
    /// # Errors
    /// The abort failure, or the wrapped [`ModelsError`] of the failing step.
    pub fn get_auth_for_model<'a>(
        &'a self,
        model: &'a Model,
        overrides: Option<&'a ModelRuntimeAuthOverrides>,
    ) -> BoxedFuture<'a, Result<Option<AuthResult>, ModelsFailure>> {
        let merged_env: Option<ProviderEnv> = overrides.and_then(|overrides| {
            let merged = overrides.env.clone().unwrap_or_default();
            (!merged.is_empty()).then_some(merged)
        });
        let resolution = self
            .models
            .get_auth_for_model(model, overrides.map(as_auth_resolution_overrides).as_ref());
        let (config, extension) = self.composition_layers(&model.provider.0);
        Box::pin(async move {
            let Some(resolution) = resolution.await? else {
                return Ok(None);
            };
            let configured_headers = resolve_configured_model_headers(
                model,
                config.as_ref(),
                extension.as_ref().map(AsRef::as_ref),
                merged_env.as_ref(),
            )
            .map_err(|error| ModelsError::new(ModelsErrorCode::Auth, error))?;
            Ok(Some(AuthResult {
                auth: pi_ai::auth::types::ModelAuth {
                    headers: merge_headers(
                        resolution.auth.headers.as_ref(),
                        configured_headers
                            .map(|headers| {
                                headers
                                    .into_iter()
                                    .map(|(name, value)| (name, Some(value)))
                                    .collect::<ProviderHeaders>()
                            })
                            .as_ref(),
                    ),
                    ..resolution.auth
                },
                env: resolution.env,
                source: resolution.source,
            }))
        })
    }

    /// Register a native pi-ai provider object, upstream's
    /// `registerNativeProvider`. An empty id records the upstream throw on
    /// the composition-error surface instead of panicking from a sync entry
    /// point.
    pub fn register_native_provider(&self, provider: Arc<dyn Provider>) {
        if provider.id().trim().is_empty() {
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .composition_errors
                .insert(String::new(), "Provider id must not be empty.".to_owned());
            return;
        }
        let provider_id = provider.id().to_owned();
        {
            let mut collections = self
                .collections
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            collections.extension_providers.remove(&provider_id);
            collections
                .native_extension_providers
                .insert(provider_id.clone(), provider);
        }
        self.recompose_provider(&provider_id);
        self.update_model_snapshot();
        self.spawn_background_refresh();
    }

    /// Register an extension provider configuration, upstream's
    /// `registerProvider(name, config)`.
    ///
    /// # Errors
    /// The registration's own validation failure; a broken re-registration
    /// fails without touching the stored config.
    pub fn register_provider(
        &self,
        provider_id: &str,
        config: ProviderConfigInput,
    ) -> Result<(), String> {
        let (builtin, models_config) = {
            let collections = self
                .collections
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                collections.builtins.get(provider_id).cloned(),
                self.models_config().get_provider(provider_id).cloned(),
            )
        };
        validate_extension_provider(
            provider_id,
            builtin.as_ref(),
            models_config.as_ref(),
            &config,
        )?;
        let effective = {
            let mut collections = self
                .collections
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            collections.native_extension_providers.remove(provider_id);
            // Re-registration merges defined values over the previous
            // registration and preserves undefined ones, matching the legacy
            // ModelRegistry contract.
            let effective = match collections.extension_providers.get(provider_id) {
                Some(previous) => Arc::new(merge_provider_inputs(previous, config)),
                None => Arc::new(config),
            };
            collections
                .extension_providers
                .insert(provider_id.to_owned(), Arc::clone(&effective));
            effective
        };
        self.recompose_provider(provider_id);
        self.update_model_snapshot();
        self.provisional_auth_snapshot(provider_id, &effective);
        self.spawn_background_refresh();
        Ok(())
    }

    /// The provisional availability entry a fresh registration carries until
    /// the async refresh lands, upstream's registerProvider snapshot patch.
    fn provisional_auth_snapshot(&self, provider_id: &str, effective: &ProviderConfigInput) {
        let stored = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .snapshot
            .stored_providers
            .contains(provider_id);
        let (config, extension) = self.composition_layers(provider_id);
        let status = if stored {
            Some(AuthStatus {
                configured: true,
                source: None,
                label: None,
            })
        } else {
            configured_request_auth_status(config.as_ref(), extension.as_ref().map(AsRef::as_ref))
        };
        if !status.is_some_and(|status| status.configured) {
            return;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut configured_providers = state.snapshot.configured_providers.clone();
        configured_providers.insert(provider_id.to_owned());
        let mut auth = state.snapshot.auth.clone();
        // Provisional entry until the async refresh lands; never clobber a
        // real check result.
        auth.entry(provider_id.to_owned()).or_insert_with(|| {
            Some(pi_ai::auth::types::AuthCheck {
                source: Some("configured provider".to_owned()),
                auth_type: if effective.oauth.is_some() && effective.api_key.is_none() {
                    AuthType::OAuth
                } else {
                    AuthType::ApiKey
                },
            })
        });
        let all = state.snapshot.all.clone();
        state.snapshot.auth = auth;
        state
            .snapshot
            .configured_providers
            .clone_from(&configured_providers);
        state.snapshot.available = all
            .into_iter()
            .filter(|model| configured_providers.contains(&model.provider.0))
            .collect();
    }

    /// Remove an extension registration, upstream's `unregisterProvider`.
    pub fn unregister_provider(&self, provider_id: &str) {
        {
            let mut collections = self
                .collections
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            collections.extension_providers.remove(provider_id);
            collections.native_extension_providers.remove(provider_id);
        }
        self.recompose_provider(provider_id);
        self.update_model_snapshot();
        self.spawn_background_refresh();
    }

    /// The fire-and-forget refresh after a registration change, upstream's
    /// `void this.refresh({ allowNetwork: false })`: the refresh runs to
    /// settlement on a spawned task.
    fn spawn_background_refresh(&self) {
        let core = self.self_arc();
        tokio::spawn(async move {
            core.refresh(ModelsRefreshOptions {
                allow_network: Some(false),
                ..ModelsRefreshOptions::default()
            })
            .await;
        });
    }

    /// The `Arc` handle the spawned refresh tasks capture, the owning `Arc` the
    /// constructor built the core through.
    #[expect(
        clippy::expect_used,
        reason = "the runtime core is only constructed through create, which owns the Arc this reads"
    )]
    fn self_arc(&self) -> Arc<Self> {
        self.self_weak
            .upgrade()
            .expect("the runtime core outlives its owning ModelRuntime")
    }

    /// The provider auth status the composed layers and snapshot report,
    /// upstream's `getProviderAuthStatus`.
    #[must_use]
    pub fn get_provider_auth_status(&self, provider_id: &str) -> AuthStatus {
        if self.credentials.has_runtime_api_key(provider_id) {
            return AuthStatus {
                configured: true,
                source: Some(AuthStatusSource::Runtime),
                label: None,
            };
        }
        if self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .snapshot
            .stored_providers
            .contains(provider_id)
        {
            return AuthStatus {
                configured: true,
                source: Some(AuthStatusSource::Stored),
                label: None,
            };
        }
        let (config, extension) = self.composition_layers(provider_id);
        if let Some(configured) =
            configured_request_auth_status(config.as_ref(), extension.as_ref().map(AsRef::as_ref))
        {
            return configured;
        }
        let check = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .snapshot
            .auth
            .get(provider_id)
            .cloned()
            .flatten();
        match check {
            Some(check) => AuthStatus {
                configured: true,
                source: Some(AuthStatusSource::Environment),
                label: check.source,
            },
            None => AuthStatus {
                configured: false,
                source: None,
                label: None,
            },
        }
    }

    /// Reload models.json and recompose, upstream's `refresh`.
    pub async fn refresh(&self, options: ModelsRefreshOptions) -> ModelsRefreshResult {
        if let Ok(config) = ModelConfig::load(self.models_path.as_deref()) {
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .config = config;
        }
        self.configure_radius_providers();
        let selected = options.providers.clone();
        match &selected {
            Some(providers) => {
                for provider_id in providers {
                    self.recompose_provider(provider_id);
                }
                self.update_model_snapshot();
            }
            None => self.rebuild_providers(),
        }
        let mut refresh_options = options.clone();
        refresh_options.allow_network =
            Some(options.allow_network.unwrap_or(self.model_network_enabled));
        let result = self.models.refresh(Some(&refresh_options)).await;
        let aborted = result.aborted;
        let mut errors = result.errors;
        self.update_model_snapshot();
        match &selected {
            Some(providers) => {
                for provider_id in providers {
                    if let Err(error) = self
                        .refresh_provider_availability(
                            provider_id,
                            operation_signal(options.signal.as_ref()),
                        )
                        .await
                        && !options
                            .signal
                            .as_ref()
                            .is_some_and(CancellationToken::is_cancelled)
                    {
                        errors.insert(
                            provider_id.clone(),
                            Box::new(std::io::Error::other(error.to_string())),
                        );
                    }
                }
            }
            None => {
                // Availability errors are recorded by the latest pass;
                // refreshed models remain usable.
                let _ = self
                    .queue_availability_refresh(options.signal.as_ref())
                    .await;
            }
        }
        ModelsRefreshResult {
            aborted: aborted
                || options
                    .signal
                    .as_ref()
                    .is_some_and(CancellationToken::is_cancelled),
            errors,
        }
    }

    /// The request preparation shared by every stream entry point, upstream's
    /// `prepareRequest`.
    async fn prepare_request<T: pi_ai::models::AuthCarrier + Default + Clone>(
        &self,
        model: &Model,
        options: Option<WithTransforms<T>>,
    ) -> Result<(Arc<dyn Provider>, Model, Option<T>), ModelsFailure> {
        let Some(provider) = self.models.provider(&model.provider.0) else {
            return Err(ModelsFailure::Models(ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Unknown provider: {}", model.provider.0),
            )));
        };
        let overrides = ModelRuntimeAuthOverrides {
            api_key: options
                .as_ref()
                .and_then(|wrapper| wrapper.options.api_key_slot().cloned()),
            env: options
                .as_ref()
                .and_then(|wrapper| wrapper.options.env_slot().cloned()),
            min_oauth_validity_ms: None,
            signal: options
                .as_ref()
                .and_then(|wrapper| wrapper.options.transport().signal.clone()),
        };
        let resolution = self.get_auth_for_model(model, Some(&overrides)).await?;
        let Some(resolution) = resolution else {
            return Err(ModelsFailure::Models(ModelsError::new(
                ModelsErrorCode::Auth,
                format!("Provider is not configured: {}", model.provider.0),
            )));
        };
        let transform = options
            .as_ref()
            .and_then(|wrapper| wrapper.transform_headers.clone());
        let mut request_options = options.map_or_else(T::default, |wrapper| wrapper.options);
        let mut headers = merge_headers(
            resolution.auth.headers.as_ref(),
            request_options.headers_slot(),
        );
        if let Some(transform) = transform {
            headers = Some((transform)(headers.unwrap_or_default()).await);
        }
        let env = match (&resolution.env, request_options.env_slot()) {
            (None, None) => None,
            (resolved, explicit) => {
                let mut merged = resolved.clone().unwrap_or_default();
                merged.extend(explicit.cloned().unwrap_or_default());
                Some(merged)
            }
        };
        let mut request_model = model.clone();
        if let Some(base_url) = &resolution.auth.base_url {
            request_model.base_url.clone_from(base_url);
        }
        *request_options.api_key_slot_mut() = request_options
            .api_key_slot()
            .cloned()
            .or_else(|| resolution.auth.api_key.clone());
        *request_options.headers_slot_mut() = headers;
        *request_options.env_slot_mut() = env;
        Ok((provider, request_model, Some(request_options)))
    }

    /// Stream through the configured provider with request-time
    /// authentication, upstream's `stream`.
    #[must_use]
    pub fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&ModelsStreamOptions>,
    ) -> AssistantMessageEventStream {
        let core = self.self_arc();
        let model = model.clone();
        let context = context.clone();
        let options = options.cloned();
        let lazy_model = model.clone();
        pi_ai::api::lazy::lazy_stream(&lazy_model, move || {
            let core = Arc::clone(&core);
            let model = model.clone();
            let context = context.clone();
            Box::pin(async move {
                let (provider, request_model, request_options) =
                    core.prepare_request(&model, options).await?;
                Ok(provider.stream(&request_model, &context, request_options.as_ref()))
            })
        })
    }

    /// Complete a request, upstream's `complete`: the stream's final
    /// assistant message, including error-terminal messages.
    pub async fn complete(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&ModelsStreamOptions>,
    ) -> AssistantMessage {
        self.stream(model, context, options).result().await
    }

    /// Stream a simple request through the configured provider, upstream's
    /// `streamSimple`.
    #[must_use]
    pub fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&ModelsSimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let core = self.self_arc();
        let model = model.clone();
        let context = context.clone();
        let options = options.cloned();
        let lazy_model = model.clone();
        pi_ai::api::lazy::lazy_stream(&lazy_model, move || {
            let core = Arc::clone(&core);
            let model = model.clone();
            let context = context.clone();
            Box::pin(async move {
                let (provider, request_model, request_options) =
                    core.prepare_request(&model, options).await?;
                Ok(provider.stream_simple(&request_model, &context, request_options.as_ref()))
            })
        })
    }

    /// Complete a simple request, upstream's `completeSimple`.
    pub async fn complete_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&ModelsSimpleStreamOptions>,
    ) -> AssistantMessage {
        self.stream_simple(model, context, options).result().await
    }

    /// Ask a capable provider to return a durable handle and continue the
    /// request asynchronously, upstream's `streamDeferred`.
    #[must_use]
    pub fn stream_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: Option<&ModelsDeferredFetchOptions>,
    ) -> AssistantMessageEventStream {
        let core = self.self_arc();
        let model = model.clone();
        let handle = handle.clone();
        let options = options.cloned();
        let lazy_model = model.clone();
        pi_ai::api::lazy::lazy_stream(&lazy_model, move || {
            let core = Arc::clone(&core);
            let model = model.clone();
            Box::pin(async move {
                let missing = || -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(format!(
                        "Provider {} does not support deferred responses",
                        model.provider.0
                    )))
                };
                let (provider, request_model, request_options) =
                    core.prepare_request(&model, options).await?;
                if !provider.supports_fetch_deferred() {
                    return Err(missing());
                }
                provider
                    .fetch_deferred(&request_model, &handle, request_options.as_ref())
                    .ok_or_else(missing)
            })
        })
    }

    /// Fetch a deferred response to completion, upstream's `fetchDeferred`.
    pub async fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: Option<&ModelsDeferredFetchOptions>,
    ) -> AssistantMessage {
        self.stream_deferred(model, handle, options).result().await
    }

    /// Cancel a deferred response, upstream's `cancelDeferred`.
    ///
    /// # Errors
    /// Unknown provider, unsupported operation, auth failure, or the
    /// provider's own failure.
    pub async fn cancel_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: Option<&ModelsDeferredCancelOptions>,
    ) -> Result<(), ModelsFailure> {
        let (provider, request_model, request_options) =
            self.prepare_request(model, options.cloned()).await?;
        if !provider.supports_cancel_deferred() {
            return Err(ModelsFailure::Models(ModelsError::new(
                ModelsErrorCode::Provider,
                format!(
                    "Provider {} does not support deferred responses",
                    model.provider.0
                ),
            )));
        }
        let Some(cancel) =
            provider.cancel_deferred(&request_model, handle, request_options.as_ref())
        else {
            return Err(ModelsFailure::Models(ModelsError::new(
                ModelsErrorCode::Provider,
                format!(
                    "Provider {} does not support deferred responses",
                    model.provider.0
                ),
            )));
        };
        match cancel.await {
            Ok(()) => Ok(()),
            Err(error) => {
                if error.is_abort() {
                    return Err(ModelsFailure::Aborted(AbortError));
                }
                Err(ModelsFailure::Models(ModelsError::with_cause(
                    ModelsErrorCode::Provider,
                    format!("Deferred cancellation failed for {}", model.provider.0),
                    Box::new(error),
                )))
            }
        }
    }

    /// The every-provider listing, upstream's `getProviders`.
    #[must_use]
    pub fn get_providers(&self) -> Vec<Arc<dyn Provider>> {
        self.models.providers()
    }

    /// One provider by id, upstream's `getProvider`.
    #[must_use]
    pub fn get_provider(&self, provider_id: &str) -> Option<Arc<dyn Provider>> {
        self.models.provider(provider_id)
    }

    /// The composed model list, upstream's `getModels`.
    #[must_use]
    pub fn get_models(&self, provider_id: Option<&str>) -> Vec<Model> {
        self.models.models(provider_id)
    }

    /// One model by provider and id, upstream's `getModel`.
    #[must_use]
    pub fn get_model(&self, provider_id: &str, model_id: &str) -> Option<Model> {
        self.models.model(provider_id, model_id)
    }

    /// Whether a provider has complete auth configuration without refreshing
    /// OAuth, upstream's `checkAuth`.
    ///
    /// # Errors
    /// The abort failure, or the wrapped [`ModelsError`] of the failing step.
    pub fn check_auth(
        &self,
        provider_id: &str,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'_, Result<Option<pi_ai::auth::types::AuthCheck>, ModelsFailure>> {
        self.models.check_auth(provider_id, options)
    }

    /// The available models, upstream's `getAvailable`: a provider-scoped
    /// query runs directly, the all-providers query routes through the
    /// serialized availability pass.
    ///
    /// # Errors
    /// The abort failure, or the wrapped [`ModelsError`] of the failing step.
    pub fn get_available(
        &self,
        provider_id: Option<&str>,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'_, Result<Vec<Model>, ModelsFailure>> {
        if let Some(provider_id) = provider_id {
            let error_seq = {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.availability_error_seq += 1;
                state.availability_error_seq
            };
            let available = self.models.available(Some(provider_id), options);
            let signal = options.and_then(|options| options.signal.clone());
            return Box::pin(async move {
                let outcome = available.await;
                let aborted = signal.as_ref().is_some_and(CancellationToken::is_cancelled);
                match outcome {
                    Ok(available) => {
                        let mut state = self
                            .state
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if error_seq == state.availability_error_seq {
                            state.availability_error = None;
                        }
                        drop(state);
                        Ok(available)
                    }
                    // Upstream's catch records the failure on the error
                    // surface before rethrowing; an aborted signal stays
                    // unrecorded.
                    Err(error) => {
                        let mut state = self
                            .state
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if error_seq == state.availability_error_seq && !aborted {
                            state.availability_error = Some(error.to_string());
                        }
                        drop(state);
                        Err(error)
                    }
                }
            });
        }
        let signal = options.and_then(|options| options.signal.clone());
        let queued = self.queue_availability_refresh(signal.as_ref());
        Box::pin(async move {
            queued.await.map_err(|error| {
                ModelsFailure::Models(ModelsError::new(ModelsErrorCode::Auth, error.to_string()))
            })?;
            let state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            Ok(state.snapshot.available.clone())
        })
    }

    /// The last-known available list, upstream's `getAvailableSnapshot`.
    #[must_use]
    pub fn get_available_snapshot(&self) -> Vec<Model> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .snapshot
            .available
            .clone()
    }

    /// The combined error surface: models.json load, provider composition,
    /// and availability refresh failures, upstream's `getError`.
    #[must_use]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the error surface reads one state snapshot under one lock"
    )]
    pub fn get_error(&self) -> Option<String> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut errors: Vec<String> = Vec::new();
        if let Some(config_error) = state.config.get_error() {
            errors.push(config_error.to_owned());
        }
        for (provider_id, error) in &state.composition_errors {
            errors.push(format!("Provider \"{provider_id}\": {error}"));
        }
        if let Some(availability_error) = &state.availability_error {
            errors.push(format!("Availability refresh: {availability_error}"));
        }
        (!errors.is_empty()).then(|| errors.join("\n\n"))
    }

    /// Set a runtime API key and synchronize, upstream's `setRuntimeApiKey`.
    ///
    /// # Errors
    /// The synchronization failure wrapped in
    /// [`CredentialSynchronizationError`], or the abort.
    pub fn set_runtime_api_key(
        &self,
        provider_id: &str,
        api_key: String,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'static, Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let signal = operation_signal(options.and_then(|options| options.signal.as_ref()));
        let core = self.self_arc();
        let provider_id = provider_id.to_owned();
        let task_provider_id = provider_id.clone();
        let task_signal = signal.clone();
        let task = {
            move || {
                let signal = task_signal.clone();
                let core = Arc::clone(&core);
                let provider_id = task_provider_id.clone();
                let api_key = api_key.clone();
                let boxed: BoxedFuture<
                    'static,
                    Result<(), Box<dyn std::error::Error + Send + Sync>>,
                > = Box::pin(async move {
                    core.credentials
                        .set_runtime_api_key(&provider_id, api_key.clone());
                    core.synchronize_credential_state(
                        &provider_id,
                        CredentialSynchronizationOperation::SetRuntimeApiKey,
                        Some(Credential::ApiKey(ApiKeyCredential {
                            key: Some(api_key),
                            env: None,
                        })),
                        signal,
                    )
                    .await
                });
                boxed
            }
        };
        self.enqueue_credential_operation(&provider_id, signal, task)
    }

    /// Remove the runtime API key and synchronize, upstream's
    /// `removeRuntimeApiKey`.
    ///
    /// # Errors
    /// The synchronization failure wrapped in
    /// [`CredentialSynchronizationError`], or the abort.
    pub fn remove_runtime_api_key(
        &self,
        provider_id: &str,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'static, Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let signal = operation_signal(options.and_then(|options| options.signal.as_ref()));
        let core = self.self_arc();
        let provider_id = provider_id.to_owned();
        let task_provider_id = provider_id.clone();
        let task_signal = signal.clone();
        let task = {
            move || {
                let signal = task_signal.clone();
                let core = Arc::clone(&core);
                let provider_id = task_provider_id.clone();
                let boxed: BoxedFuture<
                    'static,
                    Result<(), Box<dyn std::error::Error + Send + Sync>>,
                > = Box::pin(async move {
                    core.credentials.remove_runtime_api_key(&provider_id);
                    core.synchronize_credential_state(
                        &provider_id,
                        CredentialSynchronizationOperation::RemoveRuntimeApiKey,
                        None,
                        signal,
                    )
                    .await
                });
                boxed
            }
        };
        self.enqueue_credential_operation(&provider_id, signal, task)
    }

    /// The credential metadata list, upstream's `listCredentials`.
    ///
    /// # Errors
    /// The store's failure.
    pub async fn list_credentials(
        &self,
        options: Option<&AuthOptions>,
    ) -> Result<Vec<CredentialInfo>, AuthError> {
        self.credentials.list(options).await
    }

    /// Run a provider-owned login flow and synchronize, upstream's `login`.
    ///
    /// # Errors
    /// The login failure, or the post-commit synchronization failure wrapped
    /// in [`CredentialSynchronizationError`].
    pub fn login(
        &self,
        provider_id: &str,
        auth_type: AuthType,
        interaction: AuthInteraction,
    ) -> BoxedFuture<'static, Result<Credential, Box<dyn std::error::Error + Send + Sync>>> {
        let signal = operation_signal(interaction.signal.as_ref());
        let core = self.self_arc();
        let provider_id = provider_id.to_owned();
        let task_provider_id = provider_id.clone();
        let task_signal = signal.clone();
        let task = {
            move || {
                let signal = task_signal.clone();
                let core = Arc::clone(&core);
                let interaction = interaction.clone();
                let provider_id = task_provider_id.clone();
                let boxed: BoxedFuture<
                    'static,
                    Result<Credential, Box<dyn std::error::Error + Send + Sync>>,
                > = Box::pin(async move {
                    let credential = core
                        .models
                        .login(
                            &provider_id,
                            auth_type,
                            AuthInteraction {
                                signal: Some(signal.clone()),
                                ..interaction
                            },
                        )
                        .await
                        .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                            Box::new(error)
                        })?;
                    core.synchronize_credential_state(
                        &provider_id,
                        CredentialSynchronizationOperation::Login,
                        Some(credential.clone()),
                        signal,
                    )
                    .await?;
                    Ok(credential)
                });
                boxed
            }
        };
        self.enqueue_credential_operation(&provider_id, signal, task)
    }

    /// Remove the stored credential and synchronize, upstream's `logout`.
    ///
    /// # Errors
    /// The logout failure, or the post-commit synchronization failure
    /// wrapped in [`CredentialSynchronizationError`].
    pub fn logout(
        &self,
        provider_id: &str,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'static, Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let signal = operation_signal(options.and_then(|options| options.signal.as_ref()));
        let core = self.self_arc();
        let provider_id = provider_id.to_owned();
        let task_provider_id = provider_id.clone();
        let task_signal = signal.clone();
        let task = {
            move || {
                let signal = task_signal.clone();
                let core = Arc::clone(&core);
                let provider_id = task_provider_id.clone();
                let boxed: BoxedFuture<
                    'static,
                    Result<(), Box<dyn std::error::Error + Send + Sync>>,
                > = Box::pin(async move {
                    core.models
                        .logout(
                            &provider_id,
                            Some(&AuthOptions {
                                signal: Some(signal.clone()),
                            }),
                        )
                        .await
                        .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                            Box::new(error)
                        })?;
                    core.synchronize_credential_state(
                        &provider_id,
                        CredentialSynchronizationOperation::Logout,
                        None,
                        signal,
                    )
                    .await
                });
                boxed
            }
        };
        self.enqueue_credential_operation(&provider_id, signal, task)
    }
}

/// The overrides conversion into pi-ai's resolution shape.
fn as_auth_resolution_overrides(
    overrides: &ModelRuntimeAuthOverrides,
) -> pi_ai::auth::resolve::AuthResolutionOverrides {
    pi_ai::auth::resolve::AuthResolutionOverrides {
        api_key: overrides.api_key.clone(),
        env: overrides.env.clone(),
        min_oauth_validity_ms: overrides.min_oauth_validity_ms,
        signal: overrides.signal.clone(),
    }
}

/// The registration merge, upstream's spread over the previous extension
/// config: defined values win, undefined ones preserve the previous.
fn merge_provider_inputs(
    previous: &ProviderConfigInput,
    next: ProviderConfigInput,
) -> ProviderConfigInput {
    ProviderConfigInput {
        name: next.name.or_else(|| previous.name.clone()),
        base_url: next.base_url.or_else(|| previous.base_url.clone()),
        api_key: next.api_key.or_else(|| previous.api_key.clone()),
        api: next.api.or_else(|| previous.api.clone()),
        stream_simple: next
            .stream_simple
            .or_else(|| previous.stream_simple.clone()),
        headers: next.headers.or_else(|| previous.headers.clone()),
        auth_header: next.auth_header.or(previous.auth_header),
        oauth: next.oauth.or_else(|| previous.oauth.clone()),
        models: next.models.or_else(|| previous.models.clone()),
        refresh_models: next
            .refresh_models
            .or_else(|| previous.refresh_models.clone()),
    }
}

impl std::fmt::Debug for CreateModelRuntimeOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateModelRuntimeOptions")
            .field("credentials", &self.credentials.as_ref().map(|_| "set"))
            .field("auth_path", &self.auth_path)
            .field("models_path", &self.models_path)
            .field("models_store", &self.models_store.as_ref().map(|_| "set"))
            .field("models_store_path", &self.models_store_path)
            .field("allow_model_network", &self.allow_model_network)
            .field("model_refresh_timeout_ms", &self.model_refresh_timeout_ms)
            .field("catalog_base_url", &self.catalog_base_url)
            .field("signal", &self.signal)
            .field("refresh_on_create", &self.refresh_on_create)
            .finish()
    }
}

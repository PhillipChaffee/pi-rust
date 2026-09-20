//! The image-generation runtime, ported from
//! `packages/ai/src/images-models.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The image-side counterpart of [`crate::models::Models`]: image providers
//! own id/name metadata, auth, model listing, and generation behavior; the
//! collection resolves auth and delegates each request.
//!
//! Porting restatement: upstream's `ImagesOptions` carries the request
//! signal through `ProviderRequestOptions`; the crate's [`ImagesOptions`]
//! holds no signal, so auth resolution runs on a fresh uncancelled token.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::auth::context::default_provider_auth_context;
use crate::auth::credential_store::CredentialStore;
use crate::auth::credential_store::InMemoryCredentialStore;
use crate::auth::resolve::{
    AuthResolutionOverrides, ModelsError, ModelsErrorCode, ModelsFailure, now_ms,
    resolve_provider_auth,
};
use crate::auth::types::{AuthContext, AuthResult, ProviderAuth};
use crate::models::ProviderError;
use crate::types::{AssistantImages, BoxedFuture, ImagesContext, ImagesModel, ImagesOptions};

/// An image-generation provider, upstream's `ImagesProvider`: the image-side
/// counterpart of `Provider`. Owns id/name metadata, auth, model listing,
/// and generation behavior.
pub trait ImagesProvider: Send + Sync {
    /// The provider id.
    fn id(&self) -> &str;

    /// The display name.
    fn name(&self) -> &str;

    /// Required: at least one of `apiKey`/`oauth`. Same semantics as chat
    /// providers; `ImagesModels::get_auth` resolves `None` when the provider
    /// is unconfigured.
    fn auth(&self) -> &ProviderAuth;

    /// Current known models, sync. Static providers return their catalog;
    /// dynamic providers return the list as of the last `refresh` (empty
    /// before the first). Must not panic; `ImagesModels` treats a failing
    /// implementation as having no models.
    ///
    /// # Errors
    /// The provider's own failure; `ImagesModels` swallows it for listing.
    fn get_models(&self) -> Result<Vec<ImagesModel>, ProviderError>;

    /// Dynamic providers only: fetch and update the model list. Fails on
    /// network errors; on failure the model list stays at its last-known
    /// state and a later call retries.
    ///
    /// # Errors
    /// The refresh failure.
    fn refresh_models(&self) -> BoxedFuture<'_, Result<(), ProviderError>> {
        Box::pin(async { Ok(()) })
    }

    /// Whether this provider has a dynamic catalog.
    #[must_use]
    fn supports_refresh_models(&self) -> bool {
        false
    }

    /// Generate images for the input context.
    ///
    /// # Errors
    /// The provider's own failure; [`ImagesModels::generate_images`] turns
    /// it into an error result.
    fn generate_images<'a>(
        &'a self,
        model: &'a ImagesModel,
        context: &'a ImagesContext,
        options: Option<&'a ImagesOptions>,
    ) -> BoxedFuture<'a, Result<AssistantImages, crate::utils::provider_retry::ProviderRequestError>>;
}

/// The fetch closure of [`CreateImagesProviderOptions`], upstream's
/// `refreshModels`.
pub type ImagesFetchModelsFn =
    Arc<dyn Fn() -> BoxedFuture<'static, Result<Vec<ImagesModel>, ProviderError>> + Send + Sync>;

/// The core state of a provider built by [`create_images_provider`].
struct ImagesProviderCore {
    id: String,
    name: String,
    auth: ProviderAuth,
    api: Arc<dyn crate::types::ProviderImages>,
    models: Mutex<Vec<ImagesModel>>,
    refresh_models_fn: Option<ImagesFetchModelsFn>,
    /// The shared in-flight refresh, upstream's `inflightRefresh ??=`: one
    /// fetch, every concurrent caller waits on it.
    inflight: InflightCell,
}

/// A provider built from parts, upstream's `createImagesProvider` return
/// value.
#[derive(Clone)]
pub struct ImagesProviderImpl(Arc<ImagesProviderCore>);

impl std::fmt::Debug for ImagesProviderImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImagesProviderImpl")
            .field("id", &self.0.id)
            .field("name", &self.0.name)
            .finish_non_exhaustive()
    }
}

impl ImagesProvider for ImagesProviderImpl {
    fn id(&self) -> &str {
        &self.0.id
    }

    fn name(&self) -> &str {
        &self.0.name
    }

    fn auth(&self) -> &ProviderAuth {
        &self.0.auth
    }

    fn get_models(&self) -> Result<Vec<ImagesModel>, ProviderError> {
        Ok(self
            .0
            .models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone())
    }

    fn refresh_models(&self) -> BoxedFuture<'_, Result<(), ProviderError>> {
        let Some(fetch) = self.0.refresh_models_fn.clone() else {
            return Box::pin(async { Ok(()) });
        };
        let core = Arc::clone(&self.0);
        Box::pin(async move {
            // Concurrent callers share one in-flight fetch, upstream's
            // in-flight dedupe.
            let cell = {
                let mut guard = core.inflight.lock().await;
                guard.get_or_insert_with(Arc::default).clone()
            };
            let init_core = Arc::clone(&core);
            let outcome = (*cell
                .get_or_init(|| async move {
                    let outcome = (fetch)().await;
                    if let Ok(models) = &outcome {
                        init_core
                            .models
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .clone_from(models);
                    }
                    outcome.map_or_else(|error| Err(Arc::new(error)), |_| Ok(()))
                })
                .await)
                .clone();
            {
                let mut guard = core.inflight.lock().await;
                if guard
                    .as_ref()
                    .is_some_and(|current| Arc::ptr_eq(current, &cell))
                {
                    *guard = None;
                }
            }
            match outcome {
                Ok(()) => Ok(()),
                Err(error) => Err({
                    let error: ProviderError = Box::new(SharedRefreshError(error));
                    error
                }),
            }
        })
    }

    fn supports_refresh_models(&self) -> bool {
        self.0.refresh_models_fn.is_some()
    }

    fn generate_images<'a>(
        &'a self,
        model: &'a ImagesModel,
        context: &'a ImagesContext,
        options: Option<&'a ImagesOptions>,
    ) -> BoxedFuture<'a, Result<AssistantImages, crate::utils::provider_retry::ProviderRequestError>>
    {
        self.0.api.generate_images(model, context, options)
    }
}

/// The failure a deduplicated refresh reports when the shared fetch failed.
#[derive(Debug)]
struct SharedRefreshError(Arc<ProviderError>);

impl std::fmt::Display for SharedRefreshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for SharedRefreshError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        let boxed: &ProviderError = self.0.as_ref();
        let source: &(dyn std::error::Error + 'static) = &**boxed;
        Some(source)
    }
}

/// The shared in-flight refresh cell of an [`ImagesProviderImpl`].
type InflightCell =
    tokio::sync::Mutex<Option<Arc<tokio::sync::OnceCell<Result<(), Arc<ProviderError>>>>>>;

/// The construction options of [`create_images_provider`], upstream's
/// `CreateImagesProviderOptions`.
#[derive(Clone)]
pub struct CreateImagesProviderOptions {
    /// The provider id.
    pub id: String,
    /// Display name. Default: `id`.
    pub name: Option<String>,
    /// Required — every provider has auth semantics, even ambient/keyless
    /// ones.
    pub auth: ProviderAuth,
    /// Initial model list (empty for purely dynamic providers).
    pub models: Vec<ImagesModel>,
    /// Dynamic providers: fetch the current list. Stored on success;
    /// concurrent calls share one in-flight fetch. Failures leave the stored
    /// list at its last-known state and a later call retries.
    pub refresh_models: Option<ImagesFetchModelsFn>,
    /// The generation implementation.
    pub api: Arc<dyn crate::types::ProviderImages>,
}

impl std::fmt::Debug for CreateImagesProviderOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateImagesProviderOptions")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("models", &self.models.len())
            .field("refresh_models", &self.refresh_models.is_some())
            .finish_non_exhaustive()
    }
}

/// Builds an image-generation provider from parts, upstream's
/// `createImagesProvider`.
#[must_use]
pub fn create_images_provider(input: CreateImagesProviderOptions) -> ImagesProviderImpl {
    let id = input.id;
    ImagesProviderImpl(Arc::new(ImagesProviderCore {
        name: input.name.unwrap_or_else(|| id.clone()),
        id,
        auth: input.auth,
        api: input.api,
        models: Mutex::new(input.models),
        refresh_models_fn: input.refresh_models,
        inflight: tokio::sync::Mutex::new(None),
    }))
}

/// The per-collection shared runtime state of [`ImagesModels`].
struct ImagesModelsState {
    providers: std::sync::RwLock<BTreeMap<String, Arc<dyn ImagesProvider>>>,
    credentials: Arc<dyn CredentialStore>,
    auth_context: Arc<dyn AuthContext>,
}

impl std::fmt::Debug for ImagesModelsState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImagesModelsState").finish_non_exhaustive()
    }
}

/// Runtime collection of image-generation providers plus auth application
/// and generation convenience, upstream's `ImagesModels` (with
/// `MutableImagesModels` collapsed in).
///
/// The image-side counterpart of [`crate::models::Models`].
#[derive(Clone, Debug)]
pub struct ImagesModels {
    state: Arc<ImagesModelsState>,
}

/// The construction options of [`create_images_models`], reusing the chat
/// side's injected stores, upstream's `CreateModelsOptions` reuse.
#[derive(Clone, Default)]
pub struct CreateImagesModelsOptions {
    /// The credential store; defaults to the in-memory store.
    pub credentials: Option<Arc<dyn CredentialStore>>,
    /// The auth context; defaults to the process environment and filesystem.
    pub auth_context: Option<Arc<dyn AuthContext>>,
}

impl std::fmt::Debug for CreateImagesModelsOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateImagesModelsOptions")
            .field("credentials", &self.credentials.as_ref().map(|_| "set"))
            .field("auth_context", &self.auth_context.as_ref().map(|_| "set"))
            .finish()
    }
}

impl ImagesModels {
    /// Upsert/replace by provider id. Provider ids are unique, upstream's
    /// `setProvider`.
    pub fn set_provider(&self, provider: Arc<dyn ImagesProvider>) {
        self.state
            .providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(provider.id().to_owned(), provider);
    }

    /// Remove a provider, upstream's `deleteProvider`.
    pub fn delete_provider(&self, id: &str) {
        self.state
            .providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id);
    }

    /// Remove every provider, upstream's `clearProviders`.
    pub fn clear_providers(&self) {
        self.state
            .providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    /// All registered providers, upstream's `getProviders()`.
    #[must_use]
    pub fn providers(&self) -> Vec<Arc<dyn ImagesProvider>> {
        self.state
            .providers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .cloned()
            .collect()
    }

    /// One provider by id, upstream's `getProvider(id)`.
    #[must_use]
    pub fn provider(&self, id: &str) -> Option<Arc<dyn ImagesProvider>> {
        self.state
            .providers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    /// Sync read of last-known models from one provider or all providers.
    /// Best-effort: a provider whose listing fails yields no models.
    #[must_use]
    pub fn models(&self, provider: Option<&str>) -> Vec<ImagesModel> {
        if let Some(provider_id) = provider {
            let Some(entry) = self.provider(provider_id) else {
                return Vec::new();
            };
            return entry.get_models().unwrap_or_default();
        }
        let mut models = Vec::new();
        for entry in self.providers() {
            // Best-effort: failing providers yield no models.
            if let Ok(provider_models) = entry.get_models() {
                models.extend(provider_models);
            }
        }
        models
    }

    /// Sync runtime model lookup against last-known lists, upstream's
    /// `getModel(provider, id)`.
    #[must_use]
    pub fn model(&self, provider: &str, id: &str) -> Option<ImagesModel> {
        self.models(Some(provider))
            .into_iter()
            .find(|model| model.id == id)
    }

    /// Ask dynamic providers to re-fetch their model lists. With a provider
    /// id, fails with `ModelsError` (`model_source`) on that provider's fetch
    /// failure; without one, refreshes all providers concurrently
    /// best-effort. Static providers are no-ops.
    ///
    /// # Errors
    /// The selected provider's refresh failure, wrapped as `model_source`.
    pub fn refresh(&self, provider: Option<&str>) -> BoxedFuture<'_, Result<(), ModelsError>> {
        let state = Arc::clone(&self.state);
        let provider = provider.map(ToOwned::to_owned);
        Box::pin(async move {
            let Some(provider_id) = provider else {
                // Cannot fail: every provider's outcome is captured, the way
                // upstream's allSettled does.
                let providers = state
                    .providers
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .values()
                    .cloned()
                    .collect::<Vec<_>>();
                for provider in providers {
                    let _ = provider.refresh_models().await;
                }
                return Ok(());
            };
            let Some(entry) = state
                .providers
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(&provider_id)
                .cloned()
            else {
                return Ok(());
            };
            if !entry.supports_refresh_models() {
                return Ok(());
            }
            match entry.refresh_models().await {
                Ok(()) => Ok(()),
                Err(error) => Err(ModelsError::with_cause(
                    ModelsErrorCode::ModelSource,
                    format!("Model refresh failed for {provider_id}"),
                    error,
                )),
            }
        })
    }

    /// Resolve request auth by provider id, upstream's `getAuth(providerId)`.
    ///
    /// # Errors
    /// The abort failure, or the wrapped [`ModelsError`] of the failing step.
    #[must_use]
    pub fn get_auth(
        &self,
        provider_id: &str,
        overrides: Option<&AuthResolutionOverrides>,
    ) -> BoxedFuture<'_, Result<Option<AuthResult>, ModelsFailure>> {
        let Some(provider) = self.provider(provider_id) else {
            return Box::pin(async { Ok(None) });
        };
        resolve_provider_auth(
            provider.id(),
            provider.auth(),
            &self.state.credentials,
            &self.state.auth_context,
            overrides,
        )
    }

    /// Resolve request auth for an image model, upstream's `getAuth(model)`.
    ///
    /// # Errors
    /// The abort failure, or the wrapped [`ModelsError`] of the failing step.
    #[must_use]
    pub fn get_auth_for_model(
        &self,
        model: &ImagesModel,
        overrides: Option<&AuthResolutionOverrides>,
    ) -> BoxedFuture<'_, Result<Option<AuthResult>, ModelsFailure>> {
        let Some(provider) = self.provider(&model.provider.0) else {
            return Box::pin(async { Ok(None) });
        };
        resolve_provider_auth(
            provider.id(),
            provider.auth(),
            &self.state.credentials,
            &self.state.auth_context,
            overrides,
        )
    }

    /// Generate images through the owning provider with auth resolved and
    /// merged (explicit options win per field), upstream's
    /// `generateImages`. Never fails; failures are returned as an
    /// `AssistantImages` with `stopReason: "error"`.
    #[must_use]
    pub fn generate_images(
        &self,
        model: &ImagesModel,
        context: &ImagesContext,
        options: Option<&ImagesOptions>,
    ) -> BoxedFuture<'_, AssistantImages> {
        let state = Arc::clone(&self.state);
        let model = model.clone();
        let context = context.clone();
        let options = options.cloned();
        Box::pin(async move {
            let Some(provider) = state
                .providers
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(&model.provider.0)
                .cloned()
            else {
                return error_images(&model, &format!("Unknown provider: {}", model.provider.0));
            };

            let outcome: Result<AssistantImages, ModelsError> = async {
                let overrides = AuthResolutionOverrides {
                    api_key: options.as_ref().and_then(|options| options.api_key.clone()),
                    env: options.as_ref().and_then(|options| options.env.clone()),
                    min_oauth_validity_ms: None,
                    signal: None,
                };
                let resolution = resolve_provider_auth(
                    provider.id(),
                    provider.auth(),
                    &state.credentials,
                    &state.auth_context,
                    Some(&overrides),
                )
                .await
                .map_err(unwrap_models_failure)?;
                let Some(resolution) = resolution else {
                    // Unconfigured auth still dispatches; the provider
                    // decides what to do.
                    return provider
                        .generate_images(&model, &context, options.as_ref())
                        .await
                        .map_err(|error| {
                            ModelsError::with_cause(
                                ModelsErrorCode::Provider,
                                format!("Image generation failed for {}", model.provider.0),
                                Box::new(error),
                            )
                        });
                };
                let auth = resolution.auth;
                let request_model = auth.base_url.as_ref().map_or_else(
                    || model.clone(),
                    |base_url| {
                        let mut request_model = model.clone();
                        request_model.base_url.clone_from(base_url);
                        request_model
                    },
                );

                // Explicit request options win per-field; headers/env merge
                // per key.
                let api_key = options
                    .as_ref()
                    .and_then(|options| options.api_key.clone())
                    .or(auth.api_key);
                let headers = match (
                    &auth.headers,
                    options
                        .as_ref()
                        .and_then(|options| options.headers.as_ref()),
                ) {
                    (None, None) => None,
                    (auth_headers, request_headers) => {
                        let mut merged = auth_headers.clone().unwrap_or_default();
                        merged.extend(request_headers.cloned().unwrap_or_default());
                        Some(merged)
                    }
                };
                let env = match (
                    &resolution.env,
                    options.as_ref().and_then(|options| options.env.as_ref()),
                ) {
                    (None, None) => None,
                    (resolved, request) => {
                        let mut merged = resolved.clone().unwrap_or_default();
                        merged.extend(request.cloned().unwrap_or_default());
                        Some(merged)
                    }
                };
                let mut request_options = options.clone().unwrap_or_default();
                request_options.api_key = api_key;
                request_options.headers = headers;
                request_options.env = env;
                provider
                    .generate_images(&request_model, &context, Some(&request_options))
                    .await
                    .map_err(|error| {
                        ModelsError::with_cause(
                            ModelsErrorCode::Provider,
                            format!("Image generation failed for {}", model.provider.0),
                            Box::new(error),
                        )
                    })
            }
            .await;
            match outcome {
                Ok(images) => images,
                Err(error) => error_images(&model, &error.to_string()),
            }
        })
    }
}

/// Flatten a [`ModelsFailure`] into the error result the generation catch
/// reports.
fn unwrap_models_failure(failure: ModelsFailure) -> ModelsError {
    match failure {
        ModelsFailure::Models(error) => error,
        ModelsFailure::Aborted(abort) => ModelsError::new(ModelsErrorCode::Auth, abort.to_string()),
    }
}

/// Build the failing [`AssistantImages`] result, upstream's catch return.
fn error_images(model: &ImagesModel, message: &str) -> AssistantImages {
    AssistantImages {
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        output: Vec::new(),
        response_id: None,
        usage: None,
        stop_reason: crate::types::ImagesStopReason::Error,
        error_message: Some(message.to_owned()),
        timestamp: now_ms(),
    }
}

/// An `ImagesModels` collection with the injected stores, upstream's
/// `createImagesModels(options)`.
#[must_use]
pub fn create_images_models(options: Option<CreateImagesModelsOptions>) -> ImagesModels {
    let options = options.unwrap_or_default();
    ImagesModels {
        state: Arc::new(ImagesModelsState {
            providers: std::sync::RwLock::new(BTreeMap::new()),
            credentials: options
                .credentials
                .unwrap_or_else(|| Arc::new(InMemoryCredentialStore::default())),
            auth_context: options
                .auth_context
                .unwrap_or_else(default_provider_auth_context),
        }),
    }
}

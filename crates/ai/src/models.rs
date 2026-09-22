//! The Models runtime, ported from `packages/ai/src/models.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! A provider is the concrete runtime unit: id/name/base metadata, auth
//! methods, model listing, and stream behavior. [`Models`] is the runtime
//! collection that resolves auth and delegates each request to the provider
//! that owns the model.
//!
//! Porting restatements:
//!
//! - Upstream's `Models`/`MutableModels` interface pair collapses into one
//!   [`Models`] struct; a host wanting a read-only view takes `&Models`.
//! - `AbortSignal` ports to [`tokio_util::sync::CancellationToken`]; promise
//!   chains become tokio tasks. A raced operation whose future is dropped
//!   stops running outright instead of lingering as an abandoned promise —
//!   the same user-visible stop with no orphaned work left behind.
//! - `structuredClone` becomes owned clones; publication chains serialize per
//!   provider through tokio mutexes.
//! - The four `Models*Options` intersections share one [`WithTransforms`]
//!   wrapper, and the auth fields the four option shapes carry are reached
//!   through [`AuthCarrier`].

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::sync::{Arc, Mutex};

use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::api::lazy::lazy_stream;
use crate::auth::credential_store::{CredentialStore, InMemoryCredentialStore};
use crate::auth::resolve::{
    AuthResolutionOverrides, ModelsError, ModelsErrorCode, ModelsFailure, now_ms,
    resolve_provider_auth,
};
use crate::auth::types::{
    ApiKeyAuthInput, ApiKeyCredential, AuthCheck, AuthContext, AuthInteraction, AuthOptions,
    AuthResult, AuthType, Credential, CredentialModifyFn, ModelAuth, ProviderAuth,
    ProviderAuthInteraction,
};
use crate::models_store::{InMemoryModelsStore, ModelsStore, ModelsStoreEntry, ModelsStoreOptions};
use crate::types::{
    Api, AssistantMessage, BoxedFuture, Context, DeferredCancelOptions, DeferredFetchOptions,
    DeferredHandle, Model, ModelThinkingLevel, ProviderHeaders, ProviderRequestOptions,
    ProviderStreams, SimpleStreamOptions, StreamOptions, Usage, UsageCost,
};
use crate::utils::abort::{AbortError, RaceError, operation_signal, race_with_abort_signal};
use crate::utils::event_stream::AssistantMessageEventStream;

/// The failure type provider runtime callbacks fail with: its display text is
/// the reason callers surface.
pub type ProviderError = Box<dyn StdError + Send + Sync>;

/// The failure a provider's model listing reports, upstream's thrown value
/// from `getModels()`.
pub type ProviderModelError = Box<dyn StdError + Send + Sync>;

/// The synchronous in-memory catalog update of a [`ModelsPublication`],
/// upstream's `update?: () => void`.
pub type CatalogUpdate = Box<dyn FnOnce() + Send>;

/// Provider-selected persisted catalog publication, upstream's
/// `ModelsPublication`.
#[derive(Default)]
pub struct ModelsPublication {
    /// The persistence side: leave storage unchanged, delete the entry, or
    /// write it. Persistence policy remains provider-owned.
    pub persist: CatalogPersist,
    /// Optional synchronous update of provider-private in-memory catalog
    /// state; runs only after the selected persistence mutation.
    pub update: Option<CatalogUpdate>,
}

/// The persistence side of a [`ModelsPublication`], upstream's
/// `persist?: ModelsStoreEntry | null`.
#[derive(Default)]
pub enum CatalogPersist {
    /// Leave storage unchanged, upstream's omitted `persist`.
    #[default]
    Omit,
    /// Delete the persisted entry, upstream's `persist: null`.
    Delete,
    /// Write this entry, upstream's `persist: entry`.
    Write(ModelsStoreEntry),
}

/// The publication callback of [`RefreshModelsContext`], upstream's
/// `context.publish`.
///
/// Generation-checked: resolves `false` when the publication was rejected
/// (superseded or aborted), and fails with the store's error when persistence
/// failed.
pub type PublishFn = Arc<
    dyn Fn(ModelsPublication) -> BoxedFuture<'static, Result<bool, ProviderError>> + Send + Sync,
>;

/// The context a dynamic provider's refresh phase runs in, upstream's
/// `RefreshModelsContext`.
#[derive(Clone)]
pub struct RefreshModelsContext {
    /// Effective configured credential. OAuth credentials are refreshed
    /// before network access.
    pub credential: Option<Credential>,
    /// Immutable provider-scoped catalog snapshot captured before this
    /// refresh phase.
    pub stored: Option<ModelsStoreEntry>,
    /// Generation-checked publication. Persistence policy remains
    /// provider-owned; the update runs synchronously only after the selected
    /// persistence mutation.
    pub publish: PublishFn,
    /// False during offline/cache-only initialization.
    pub allow_network: bool,
    /// Bypass provider freshness checks and fetch immediately when network
    /// access is allowed.
    pub force: Option<bool>,
    /// Always present, including when the public refresh caller omits its
    /// optional signal.
    pub signal: CancellationToken,
}

impl std::fmt::Debug for RefreshModelsContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefreshModelsContext")
            .field("credential", &self.credential)
            .field("stored", &self.stored)
            .field("allow_network", &self.allow_network)
            .field("force", &self.force)
            .finish_non_exhaustive()
    }
}

/// A provider is the concrete runtime unit, upstream's `Provider`. It owns
/// id/name/base metadata, auth methods, model listing, and stream behavior.
pub trait Provider: Send + Sync {
    /// The provider id.
    fn id(&self) -> &str;

    /// The display name.
    fn name(&self) -> &str;

    /// The API base URL, upstream's optional `baseUrl`.
    fn base_url(&self) -> Option<&str> {
        None
    }

    /// Default headers merged into API requests.
    fn headers(&self) -> Option<&ProviderHeaders> {
        None
    }

    /// Required: at least one of `apiKey`/`oauth`. Every provider has auth
    /// semantics — even providers with only ambient credentials (env vars,
    /// AWS profiles, ADC files) and keyless local servers provide `apiKey`
    /// auth whose `resolve()` reports whether the provider is configured.
    /// [`Models::get_auth`] resolves `None` when the provider is
    /// unconfigured.
    fn auth(&self) -> &ProviderAuth;

    /// Current known models, sync. Static providers return their catalog;
    /// dynamic providers return the list as of the last [`Models::refresh`]
    /// (empty before the first). Must not panic; `Models` treats a failing
    /// implementation as having no models.
    ///
    /// # Errors
    /// The provider's own failure; Models swallows it for listing.
    fn get_models(&self) -> Result<Vec<Model>, ProviderModelError>;

    /// Dynamic providers only: restore `context.stored` and optionally fetch
    /// a newer list using the effective credential. Implementations retain
    /// their previous list on failure, publish persistence and synchronous
    /// state changes through `context.publish`, and honor the shared abort
    /// signal for blocking work.
    ///
    /// # Errors
    /// The refresh failure; Models records it unless the operation aborted.
    fn refresh_models(
        &self,
        _context: RefreshModelsContext,
    ) -> BoxedFuture<'_, Result<(), ProviderError>> {
        Box::pin(async { Ok(()) })
    }

    /// Whether this provider has a dynamic catalog, the port of upstream's
    /// `refreshModels !== undefined` presence check.
    #[must_use]
    fn supports_refresh_models(&self) -> bool {
        false
    }

    /// Optional provider policy for credential-specific model availability.
    /// `get_models` remains the complete synchronous catalog;
    /// [`Models::available`] applies this filter after confirming that
    /// provider auth is configured.
    fn filter_models(&self, models: Vec<Model>, _credential: Option<&Credential>) -> Vec<Model> {
        models
    }

    /// Stream an assistant response for the model and context.
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream;

    /// Stream a simple assistant response for the model and context.
    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream;

    /// Fetch a deferred response by its durable handle; `None` when the
    /// provider does not support deferred responses.
    fn fetch_deferred(
        &self,
        _model: &Model,
        _handle: &DeferredHandle,
        _options: Option<&DeferredFetchOptions>,
    ) -> Option<AssistantMessageEventStream> {
        None
    }

    /// Cancel a deferred response; `None` when the provider does not support
    /// deferred responses.
    ///
    /// # Errors
    /// The returned future resolves to the provider's failure.
    fn cancel_deferred<'a>(
        &'a self,
        _model: &'a Model,
        _handle: &'a DeferredHandle,
        _options: Option<&'a DeferredCancelOptions>,
    ) -> Option<BoxedFuture<'a, Result<(), crate::utils::provider_retry::ProviderRequestError>>>
    {
        None
    }

    /// Whether this provider implements `fetch_deferred`, the port of
    /// upstream's `provider.fetchDeferred !== undefined` presence check.
    #[must_use]
    fn supports_fetch_deferred(&self) -> bool {
        false
    }

    /// Whether this provider implements `cancel_deferred`, the port of
    /// upstream's `provider.cancelDeferred !== undefined` presence check.
    #[must_use]
    fn supports_cancel_deferred(&self) -> bool {
        false
    }
}

/// The result of [`Models::refresh`], upstream's `ModelsRefreshResult`.
#[derive(Debug, Default)]
pub struct ModelsRefreshResult {
    /// Whether the caller's signal aborted the refresh.
    pub aborted: bool,
    /// Per-provider failures, keyed by provider id. Cancellation is not
    /// reported as a provider failure.
    pub errors: BTreeMap<String, ProviderError>,
}

/// Cancellation and selection options for [`Models::refresh`], upstream's
/// `ModelsRefreshOptions`.
#[derive(Clone, Debug, Default)]
pub struct ModelsRefreshOptions {
    /// Whether refresh phases may access the network. Default: true.
    pub allow_network: Option<bool>,
    /// Restrict refresh to these provider IDs. Unknown and static providers
    /// are ignored.
    pub providers: Option<Vec<String>>,
    /// Bypass provider freshness checks and fetch immediately when network
    /// access is allowed.
    pub force: Option<bool>,
    /// Cancellation for the whole refresh.
    pub signal: Option<CancellationToken>,
}

/// The assembled-headers transform, upstream's `transformHeaders`.
pub type TransformHeadersFn =
    Arc<dyn Fn(ProviderHeaders) -> BoxedFuture<'static, ProviderHeaders> + Send + Sync>;

/// Request options with the Models-level header transform, upstream's
/// `ModelsRequestTransforms` intersections.
///
/// The four aliases: `ModelsApiStreamOptions`, `ModelsSimpleStreamOptions`,
/// `ModelsDeferredFetchOptions`, and `ModelsDeferredCancelOptions`.
#[derive(Clone, Default)]
pub struct WithTransforms<T> {
    /// The underlying request options.
    pub options: T,
    /// Transform fully assembled model/auth/request headers before provider
    /// dispatch, upstream's `transformHeaders`.
    pub transform_headers: Option<TransformHeadersFn>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for WithTransforms<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WithTransforms")
            .field("options", &self.options)
            .field(
                "transform_headers",
                &self.transform_headers.as_ref().map(|_| "set"),
            )
            .finish()
    }
}

/// `ModelsApiStreamOptions = ApiStreamOptions & ModelsRequestTransforms`.
pub type ModelsStreamOptions = WithTransforms<StreamOptions>;
/// `ModelsSimpleStreamOptions = SimpleStreamOptions & ModelsRequestTransforms`.
pub type ModelsSimpleStreamOptions = WithTransforms<SimpleStreamOptions>;
/// `ModelsDeferredFetchOptions = DeferredFetchOptions & ModelsRequestTransforms`.
pub type ModelsDeferredFetchOptions = WithTransforms<DeferredFetchOptions>;
/// `ModelsDeferredCancelOptions = DeferredCancelOptions & ModelsRequestTransforms`.
pub type ModelsDeferredCancelOptions = WithTransforms<ProviderRequestOptions>;

/// The auth-carrying fields the four request-option shapes share, upstream's
/// `ProviderRequestOptions & ModelsRequestTransforms` intersection surface
/// `applyAuth` mutates.
pub trait AuthCarrier: Clone {
    /// The API key slot.
    fn api_key_slot(&self) -> Option<&String>;
    /// The API key slot.
    fn api_key_slot_mut(&mut self) -> &mut Option<String>;
    /// The provider-env slot.
    fn env_slot(&self) -> Option<&crate::types::ProviderEnv>;
    /// The provider-env slot.
    fn env_slot_mut(&mut self) -> &mut Option<crate::types::ProviderEnv>;
    /// The headers slot.
    fn headers_slot(&self) -> Option<&ProviderHeaders>;
    /// The headers slot.
    fn headers_slot_mut(&mut self) -> &mut Option<ProviderHeaders>;
    /// The transport seam, which carries the cancellation token.
    fn transport(&self) -> &crate::types::TransportOptions;
}

macro_rules! impl_auth_carrier {
    ($($name:ty),+ $(,)?) => {
        $(
            impl AuthCarrier for $name {
                fn api_key_slot(&self) -> Option<&String> {
                    self.api_key.as_ref()
                }

                fn api_key_slot_mut(&mut self) -> &mut Option<String> {
                    &mut self.api_key
                }

                fn env_slot(&self) -> Option<&crate::types::ProviderEnv> {
                    self.env.as_ref()
                }

                fn env_slot_mut(&mut self) -> &mut Option<crate::types::ProviderEnv> {
                    &mut self.env
                }

                fn headers_slot(&self) -> Option<&ProviderHeaders> {
                    self.headers.as_ref()
                }

                fn headers_slot_mut(&mut self) -> &mut Option<ProviderHeaders> {
                    &mut self.headers
                }

                fn transport(&self) -> &crate::types::TransportOptions {
                    &self.transport_options
                }
            }
        )+
    };
}

impl_auth_carrier!(
    StreamOptions,
    SimpleStreamOptions,
    DeferredFetchOptions,
    ProviderRequestOptions
);

/// Flatten a [`ModelsFailure`] into a [`ModelsError`]; the abort variant's
/// text rides as the message, and the surrounding race reports the real
/// abort when the signal cancelled.
fn failure_into_error(failure: ModelsFailure) -> ModelsError {
    match failure {
        ModelsFailure::Aborted(abort) => {
            ModelsError::new(ModelsErrorCode::Stream, abort.to_string())
        }
        ModelsFailure::Models(error) => error,
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

/// Single implementation, or map keyed by `model.api` for mixed-API
/// providers, upstream's `api: ProviderStreams | Partial<Record<TApi,
/// ProviderStreams>>`.
#[derive(Clone)]
pub enum ProviderApi {
    /// One implementation streams every model.
    Single(Arc<dyn ProviderStreams>),
    /// The implementation dispatches on `model.api`; a model whose api has no
    /// entry produces a stream error.
    ByApi(crate::api::ApiMap),
}

impl From<Arc<dyn ProviderStreams>> for ProviderApi {
    fn from(value: Arc<dyn ProviderStreams>) -> Self {
        Self::Single(value)
    }
}

impl From<crate::api::ApiMap> for ProviderApi {
    fn from(value: crate::api::ApiMap) -> Self {
        Self::ByApi(value)
    }
}

/// The fetch-models closure of [`CreateProviderOptions`], upstream's
/// `fetchModels`: fetches a dynamic model overlay.
pub type FetchModelsFn = Arc<
    dyn Fn(&RefreshModelsContext) -> BoxedFuture<'static, Result<Vec<Model>, ProviderError>>
        + Send
        + Sync,
>;

/// The credential-specific availability filter of [`CreateProviderOptions`],
/// upstream's `filterModels`.
pub type FilterModelsFn = Arc<dyn Fn(Vec<Model>, Option<&Credential>) -> Vec<Model> + Send + Sync>;

/// The construction options of [`create_provider`], upstream's
/// `CreateProviderOptions`.
#[derive(Clone)]
pub struct CreateProviderOptions {
    /// The provider id.
    pub id: String,
    /// Display name. Default: `id`.
    pub name: Option<String>,
    /// The API base URL.
    pub base_url: Option<String>,
    /// Default headers merged into API requests.
    pub headers: Option<ProviderHeaders>,
    /// Required — every provider has auth semantics, even ambient/keyless
    /// ones.
    pub auth: ProviderAuth,
    /// Static baseline model list (empty for purely dynamic providers).
    pub models: Vec<Model>,
    /// Fetch a dynamic model overlay. `create_provider` restores and
    /// publishes it transactionally.
    pub fetch_models: Option<FetchModelsFn>,
    /// Provider policy for credential-specific model availability.
    pub filter_models: Option<FilterModelsFn>,
    /// Single implementation, or map keyed by `model.api` for mixed-API
    /// providers.
    pub api: ProviderApi,
}

impl std::fmt::Debug for CreateProviderOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateProviderOptions")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("base_url", &self.base_url)
            .field("models", &self.models.len())
            .finish_non_exhaustive()
    }
}

/// The core state of a provider built by [`create_provider`], shared behind
/// an [`Arc`] so refresh publications can mutate the dynamic overlay.
struct ProviderCore {
    id: String,
    name: String,
    base_url: Option<String>,
    headers: Option<ProviderHeaders>,
    auth: ProviderAuth,
    baseline_models: Vec<Model>,
    dynamic_models: Mutex<Vec<Model>>,
    fetch_models: Option<FetchModelsFn>,
    filter_models_fn: Option<FilterModelsFn>,
    api: ProviderApi,
}

/// A provider built from parts, upstream's `createProvider` return value.
/// Built-in provider factories and models.json custom providers both go
/// through this.
#[derive(Clone)]
pub struct ProviderImpl(Arc<ProviderCore>);

impl std::fmt::Debug for ProviderImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderImpl")
            .field("id", &self.0.id)
            .field("name", &self.0.name)
            .finish_non_exhaustive()
    }
}

impl ProviderCore {
    /// The baseline list merged with the dynamic overlay, upstream's
    /// `currentModels`: dynamic entries upsert by model id.
    fn current_models(&self) -> Vec<Model> {
        let mut merged = self.baseline_models.clone();
        for model in self
            .dynamic_models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
        {
            match merged.iter_mut().find(|entry| entry.id == model.id) {
                Some(slot) => *slot = model.clone(),
                None => merged.push(model.clone()),
            }
        }
        merged
    }

    /// The streams implementation for a model, upstream's `apiFor`.
    fn api_for(&self, model: &Model) -> Option<Arc<dyn ProviderStreams>> {
        match &self.api {
            ProviderApi::Single(streams) => Some(Arc::clone(streams)),
            ProviderApi::ByApi(map) => map.get(&model.api.0).cloned(),
        }
    }

    /// Every configured streams implementation, for the deferred-response
    /// capability checks.
    fn all_streams(&self) -> Vec<Arc<dyn ProviderStreams>> {
        match &self.api {
            ProviderApi::Single(streams) => vec![Arc::clone(streams)],
            ProviderApi::ByApi(map) => map.values().cloned().collect(),
        }
    }

    /// The stream error a model whose api has no entry produces, upstream's
    /// missing-API dispatch.
    fn missing_api_stream(&self, model: &Model) -> AssistantMessageEventStream {
        let provider_id = self.id.clone();
        let api = model.api.clone();
        lazy_stream(model, move || {
            let message = format!("Provider {provider_id} has no API implementation for \"{api}\"");
            let error: crate::api::lazy::LazyStreamError =
                ModelsError::new(ModelsErrorCode::Stream, message).into();
            Box::pin(async move { Err(error) })
        })
    }
}

impl Provider for ProviderImpl {
    fn id(&self) -> &str {
        &self.0.id
    }

    fn name(&self) -> &str {
        &self.0.name
    }

    fn base_url(&self) -> Option<&str> {
        self.0.base_url.as_deref()
    }

    fn headers(&self) -> Option<&ProviderHeaders> {
        self.0.headers.as_ref()
    }

    fn auth(&self) -> &ProviderAuth {
        &self.0.auth
    }

    fn get_models(&self) -> Result<Vec<Model>, ProviderModelError> {
        Ok(self.0.current_models())
    }

    fn refresh_models(
        &self,
        context: RefreshModelsContext,
    ) -> BoxedFuture<'_, Result<(), ProviderError>> {
        let Some(fetch_models) = self.0.fetch_models.clone() else {
            return Box::pin(async { Ok(()) });
        };
        let core = Arc::clone(&self.0);
        Box::pin(async move {
            if let Some(stored) = context.stored.clone() {
                let restored: Vec<Model> = stored
                    .models
                    .into_iter()
                    .filter(|model| model.provider.0 == core.id)
                    .collect();
                let published = (context.publish)(ModelsPublication {
                    persist: CatalogPersist::Omit,
                    update: Some(Box::new({
                        let core = Arc::clone(&core);
                        move || {
                            core.dynamic_models
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .clone_from(&restored);
                        }
                    })),
                })
                .await?;
                if !published {
                    return Ok(());
                }
            }
            if !context.allow_network || context.signal.is_cancelled() {
                return Ok(());
            }
            let refreshed = fetch_models(&context).await?;
            if context.signal.is_cancelled() {
                return Ok(());
            }
            let entry = ModelsStoreEntry {
                models: refreshed.clone(),
                checked_at: Some(now_ms()),
                ..ModelsStoreEntry::default()
            };
            let published = (context.publish)(ModelsPublication {
                persist: CatalogPersist::Write(entry),
                update: Some(Box::new({
                    let core = Arc::clone(&core);
                    move || {
                        core.dynamic_models
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .clone_from(&refreshed);
                    }
                })),
            })
            .await?;
            let _ = published;
            Ok(())
        })
    }

    fn supports_refresh_models(&self) -> bool {
        self.0.fetch_models.is_some()
    }

    fn filter_models(&self, models: Vec<Model>, credential: Option<&Credential>) -> Vec<Model> {
        match &self.0.filter_models_fn {
            Some(filter) => filter(models, credential),
            None => models,
        }
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        self.0.api_for(model).map_or_else(
            || self.0.missing_api_stream(model),
            |streams| streams.stream(model, context, options),
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        self.0.api_for(model).map_or_else(
            || self.0.missing_api_stream(model),
            |streams| streams.stream_simple(model, context, options),
        )
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: Option<&DeferredFetchOptions>,
    ) -> Option<AssistantMessageEventStream> {
        if !self
            .0
            .all_streams()
            .iter()
            .any(|streams| streams.supports_fetch_deferred())
        {
            return None;
        }
        let core = Arc::clone(&self.0);
        let outer_model = model.clone();
        let model = model.clone();
        let handle = handle.clone();
        let options = options.cloned();
        Some(lazy_stream(&outer_model, move || {
            let core = Arc::clone(&core);
            Box::pin(async move {
                let missing = || {
                    let error: crate::api::lazy::LazyStreamError = ModelsError::new(
                        ModelsErrorCode::Provider,
                        format!(
                            "Provider {} does not support deferred responses for \"{}\"",
                            core.id, model.api
                        ),
                    )
                    .into();
                    error
                };
                let Some(implementation) = core.api_for(&model) else {
                    return Err(missing());
                };
                if !implementation.supports_fetch_deferred() {
                    return Err(missing());
                }
                implementation
                    .fetch_deferred(&model, &handle, options.as_ref())
                    .ok_or_else(missing)
            })
        }))
    }

    fn cancel_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a DeferredHandle,
        options: Option<&'a DeferredCancelOptions>,
    ) -> Option<BoxedFuture<'a, Result<(), crate::utils::provider_retry::ProviderRequestError>>>
    {
        if !self
            .0
            .all_streams()
            .iter()
            .any(|streams| streams.supports_cancel_deferred())
        {
            return None;
        }
        let implementation = self.0.api_for(model)?;
        if !implementation.supports_cancel_deferred() {
            return None;
        }
        Some(Box::pin(async move {
            implementation.cancel_deferred(model, handle, options).await
        }))
    }

    fn supports_fetch_deferred(&self) -> bool {
        self.0
            .all_streams()
            .iter()
            .any(|streams| streams.supports_fetch_deferred())
    }

    fn supports_cancel_deferred(&self) -> bool {
        self.0
            .all_streams()
            .iter()
            .any(|streams| streams.supports_cancel_deferred())
    }
}

/// Builds a provider from parts, upstream's `createProvider`.
///
/// Built-in provider factories and models.json custom providers both go
/// through this. A single `api` streams all models; an `api` map dispatches
/// on `model.api`, and a model whose api has no entry produces a stream
/// error.
#[must_use]
pub fn create_provider(input: CreateProviderOptions) -> ProviderImpl {
    let id = input.id;
    ProviderImpl(Arc::new(ProviderCore {
        name: input.name.unwrap_or_else(|| id.clone()),
        id,
        base_url: input.base_url,
        headers: input.headers,
        auth: input.auth,
        baseline_models: input.models,
        dynamic_models: Mutex::new(Vec::new()),
        fetch_models: input.fetch_models,
        filter_models_fn: input.filter_models,
        api: input.api,
    }))
}

/// The construction options of [`create_models`], upstream's
/// `CreateModelsOptions`.
#[derive(Clone, Default)]
pub struct CreateModelsOptions {
    /// The credential store; defaults to the in-memory store.
    pub credentials: Option<Arc<dyn CredentialStore>>,
    /// The persisted-catalog store; defaults to the in-memory store.
    pub models_store: Option<Arc<dyn ModelsStore>>,
    /// The auth context; defaults to the process environment and filesystem.
    pub auth_context: Option<Arc<dyn AuthContext>>,
}

impl std::fmt::Debug for ModelsPublication {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelsPublication")
            .field("persist", &std::any::type_name::<CatalogPersist>())
            .field("update", &self.update.is_some())
            .finish()
    }
}

impl std::fmt::Debug for CatalogPersist {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Omit => f.write_str("Omit"),
            Self::Delete => f.write_str("Delete"),
            Self::Write(_) => f.write_str("Write(..)"),
        }
    }
}

impl std::fmt::Debug for ProviderApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(_) => f.write_str("Single(..)"),
            Self::ByApi(map) => f.debug_tuple("ByApi").field(&map.len()).finish(),
        }
    }
}

impl std::fmt::Debug for CreateModelsOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateModelsOptions")
            .field("credentials", &self.credentials.as_ref().map(|_| "set"))
            .field("models_store", &self.models_store.as_ref().map(|_| "set"))
            .field("auth_context", &self.auth_context.as_ref().map(|_| "set"))
            .finish()
    }
}

/// The per-provider shared runtime state of [`Models`].
struct ModelsState {
    providers: std::sync::RwLock<BTreeMap<String, Arc<dyn Provider>>>,
    credentials: Arc<dyn CredentialStore>,
    models_store: Arc<dyn ModelsStore>,
    auth_context: Arc<dyn AuthContext>,
    generations: Mutex<BTreeMap<String, u64>>,
    controllers: Mutex<BTreeMap<String, (u64, CancellationToken)>>,
    publication_locks: Mutex<BTreeMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl std::fmt::Debug for ModelsState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelsState").finish_non_exhaustive()
    }
}

impl ModelsState {
    /// The provider owning a model, upstream's `requireProvider`.
    fn require_provider(&self, model: &Model) -> Result<Arc<dyn Provider>, ModelsError> {
        self.provider_of(&model.provider.0).ok_or_else(|| {
            ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Unknown provider: {}", model.provider.0),
            )
        })
    }

    fn providers(&self) -> Vec<Arc<dyn Provider>> {
        self.providers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .cloned()
            .collect()
    }

    fn provider_of(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.providers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    /// Serialize publications per provider id without releasing the chain
    /// before active work settles, upstream's publication chain.
    fn publication_lock(&self, provider_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        self.publication_locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(provider_id.to_owned())
            .or_default()
            .clone()
    }

    /// The recorded refresh generation of a provider; zero when never
    /// refreshed.
    fn generation(&self, provider_id: &str) -> u64 {
        *self
            .generations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(provider_id)
            .unwrap_or(&0)
    }

    /// Bump the generation and abort any in-flight refresh, upstream's
    /// `supersedeProviderRefresh`.
    fn supersede_provider_refresh(&self, provider_id: &str) -> u64 {
        let generation = {
            let mut generations = self
                .generations
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let generation = generations.get(provider_id).copied().unwrap_or(0) + 1;
            generations.insert(provider_id.to_owned(), generation);
            generation
        };
        let controller = self
            .controllers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(provider_id);
        if let Some((_, token)) = controller {
            token.cancel();
        }
        generation
    }

    /// Start a refresh generation for a provider, superseding any previous
    /// one, upstream's `beginProviderRefresh`.
    fn begin_provider_refresh(&self, provider_id: &str) -> (u64, CancellationToken) {
        let generation = self.supersede_provider_refresh(provider_id);
        let token = CancellationToken::new();
        self.controllers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(provider_id.to_owned(), (generation, token.clone()));
        (generation, token)
    }

    /// Release the refresh controller if it is still the current generation,
    /// upstream's refresh finally block.
    fn end_provider_refresh(&self, provider_id: &str, generation: u64) {
        let mut controllers = self
            .controllers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if controllers
            .get(provider_id)
            .is_some_and(|(stored, _)| *stored == generation)
        {
            controllers.remove(provider_id);
        }
    }

    /// Publish a provider's models: serialize behind the provider's
    /// publication chain, re-check generation and abort after the persistence
    /// mutation, then run the synchronous update, upstream's
    /// `publishProviderModels`.
    async fn publish_provider_models(
        &self,
        provider_id: &str,
        generation: u64,
        signal: &CancellationToken,
        publication: ModelsPublication,
    ) -> Result<bool, ProviderError> {
        let chain = self.publication_lock(provider_id);
        let store = Arc::clone(&self.models_store);
        let operation = async {
            let _guard = chain.lock().await;
            if signal.is_cancelled() || self.generation(provider_id) != generation {
                return Ok(false);
            }
            match &publication.persist {
                CatalogPersist::Omit => {}
                CatalogPersist::Delete => {
                    store
                        .delete(
                            provider_id,
                            Some(&ModelsStoreOptions {
                                signal: Some(signal.clone()),
                            }),
                        )
                        .await?;
                }
                CatalogPersist::Write(entry) => {
                    store
                        .write(
                            provider_id,
                            entry.clone(),
                            Some(&ModelsStoreOptions {
                                signal: Some(signal.clone()),
                            }),
                        )
                        .await?;
                }
            }
            if signal.is_cancelled() || self.generation(provider_id) != generation {
                return Ok(false);
            }
            if let Some(update) = publication.update {
                update();
            }
            Ok(true)
        };
        match race_with_abort_signal(operation, signal).await {
            Ok(published) => Ok(published),
            // An aborted publication was not applied; the provider treats it
            // like a rejection.
            Err(RaceError::Aborted(_)) => Ok(false),
            Err(RaceError::Operation(error)) => Err(error),
        }
    }

    /// Resolve the effective credential for a network refresh phase, upstream
    /// `resolveRefreshCredential`.
    async fn resolve_refresh_credential(
        &self,
        provider: &Arc<dyn Provider>,
        stored: Option<Credential>,
        signal: &CancellationToken,
    ) -> Result<Option<Credential>, ProviderError> {
        if let Some(stored) = stored {
            if let Some(oauth) = stored.as_oauth() {
                let Some(oauth_auth) = provider.auth().oauth.as_ref() else {
                    return Ok(None);
                };
                if now_ms() < oauth.expires {
                    return Ok(Some(stored));
                }
                if signal.is_cancelled() {
                    return Ok(None);
                }
                let oauth_auth = oauth_auth.clone();
                let refresh_signal = signal.clone();
                let modify: CredentialModifyFn = Box::new(move |current| {
                    let oauth_auth = oauth_auth.clone();
                    Box::pin(async move {
                        let Some(current) = current else {
                            return Ok(None);
                        };
                        let Some(current) = current.as_oauth() else {
                            return Ok(None);
                        };
                        if now_ms() < current.expires {
                            // Another process/request refreshed.
                            return Ok(None);
                        }
                        let refreshed = (oauth_auth.refresh)(current.clone(), refresh_signal)
                            .await
                            .map(Credential::OAuth)?;
                        Ok(Some(refreshed))
                    })
                });
                let post = self
                    .credentials
                    .modify(
                        provider.id(),
                        modify,
                        Some(&AuthOptions {
                            signal: Some(signal.clone()),
                        }),
                    )
                    .await?;
                return Ok(post.filter(|credential| credential.auth_type() == AuthType::OAuth));
            }

            if stored.auth_type() == AuthType::ApiKey {
                let Some(api_key) = provider.auth().api_key.as_ref() else {
                    return Ok(None);
                };
                let input = ApiKeyAuthInput {
                    ctx: Arc::clone(&self.auth_context),
                    credential: stored.as_api_key_credential().cloned(),
                    signal: signal.clone(),
                };
                let result = (api_key.resolve)(input).await?;
                return Ok(result.map(|resolution| {
                    Credential::ApiKey(ApiKeyCredential {
                        key: resolution.auth.api_key,
                        env: resolution.env,
                    })
                }));
            }
            return Ok(None);
        }

        // Ambient resolution with no stored credential.
        let Some(api_key) = provider.auth().api_key.as_ref() else {
            return Ok(None);
        };
        let input = ApiKeyAuthInput {
            ctx: Arc::clone(&self.auth_context),
            credential: None,
            signal: signal.clone(),
        };
        let result = (api_key.resolve)(input).await?;
        Ok(result.map(|resolution| {
            Credential::ApiKey(ApiKeyCredential {
                key: resolution.auth.api_key,
                env: resolution.env,
            })
        }))
    }
}

/// Runtime collection of providers plus auth application and stream
/// convenience, upstream's `Models` (with `MutableModels` collapsed in).
///
/// Providers own stream behavior; `Models` resolves auth and delegates each
/// request to the provider that owns the model.
#[derive(Clone, Debug)]
pub struct Models {
    state: Arc<ModelsState>,
}

impl Models {
    /// Upsert/replace by provider id. Provider ids are unique; a replacement
    /// supersedes any in-flight refresh of the previous registration,
    /// upstream's `setProvider`.
    pub fn set_provider(&self, provider: Arc<dyn Provider>) {
        self.state.supersede_provider_refresh(provider.id());
        self.state
            .providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(provider.id().to_owned(), provider);
    }

    /// Remove a provider and supersede its refresh, upstream's
    /// `deleteProvider`.
    pub fn delete_provider(&self, id: &str) {
        self.state.supersede_provider_refresh(id);
        self.state
            .providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id);
    }

    /// Remove every provider, superseding all refreshes first, upstream's
    /// `clearProviders`.
    pub fn clear_providers(&self) {
        let ids: Vec<String> = {
            let providers = self
                .state
                .providers
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            providers.keys().cloned().collect()
        };
        for id in ids {
            self.state.supersede_provider_refresh(&id);
        }
        let controller_ids: Vec<String> = self
            .state
            .controllers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keys()
            .cloned()
            .collect();
        for id in controller_ids {
            self.state.supersede_provider_refresh(&id);
        }
        self.state
            .providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    /// All registered providers, upstream's `getProviders()`.
    #[must_use]
    pub fn providers(&self) -> Vec<Arc<dyn Provider>> {
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
    pub fn provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.state.provider_of(id)
    }

    /// Sync read of last-known models from one provider or all providers.
    /// Best-effort: a provider whose listing fails yields no models.
    #[must_use]
    pub fn models(&self, provider: Option<&str>) -> Vec<Model> {
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
    pub fn model(&self, provider: &str, id: &str) -> Option<Model> {
        self.models(Some(provider))
            .into_iter()
            .find(|model| model.id == id)
    }

    /// Resolve provider-scoped auth by provider id, upstream's
    /// `getAuth(providerId)`.
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

    /// Resolve provider auth plus static model headers when passed a model,
    /// upstream's `getAuth(model)`.
    ///
    /// # Errors
    /// The abort failure, or the wrapped [`ModelsError`] of the failing step.
    #[must_use]
    pub fn get_auth_for_model(
        &self,
        model: &Model,
        overrides: Option<&AuthResolutionOverrides>,
    ) -> BoxedFuture<'_, Result<Option<AuthResult>, ModelsFailure>> {
        let state = Arc::clone(&self.state);
        let model = model.clone();
        let overrides = overrides.cloned();
        Box::pin(async move {
            resolve_auth_for_model(
                &state,
                &model,
                overrides
                    .as_ref()
                    .and_then(|overrides| overrides.api_key.clone()),
                overrides
                    .as_ref()
                    .and_then(|overrides| overrides.env.clone()),
                overrides
                    .as_ref()
                    .and_then(|overrides| overrides.signal.clone()),
            )
            .await
        })
    }

    /// Check whether a provider has complete auth configuration without
    /// refreshing OAuth, upstream's `checkAuth`.
    ///
    /// # Errors
    /// The abort failure, or the wrapped [`ModelsError`] of the failing step.
    #[must_use]
    pub fn check_auth(
        &self,
        provider_id: &str,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'_, Result<Option<AuthCheck>, ModelsFailure>> {
        let state = Arc::clone(&self.state);
        let provider_id = provider_id.to_owned();
        let signal = operation_signal(options.and_then(|options| options.signal.as_ref()));
        Box::pin(async move {
            let check = check_auth_operation(&state, &provider_id, &signal);
            race_with_abort_signal(check, &signal)
                .await
                .map_err(ModelsFailure::from)
        })
    }

    /// Return models whose providers have complete auth configuration,
    /// upstream's `getAvailable`.
    ///
    /// # Errors
    /// The abort failure, or the wrapped [`ModelsError`] of the failing step.
    pub fn available(
        &self,
        provider_id: Option<&str>,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'_, Result<Vec<Model>, ModelsFailure>> {
        let state = Arc::clone(&self.state);
        let provider_id = provider_id.map(ToOwned::to_owned);
        let signal = operation_signal(options.and_then(|options| options.signal.as_ref()));
        Box::pin(async move {
            let operation = available_operation(&state, provider_id.as_deref(), &signal);
            race_with_abort_signal(operation, &signal)
                .await
                .map_err(ModelsFailure::from)
        })
    }

    /// Refresh selected configured dynamic providers concurrently (all when
    /// `providers` is omitted). Provider errors and cancellation are returned
    /// without failing; static, unknown, and unconfigured providers are
    /// skipped.
    #[must_use]
    pub fn refresh(
        &self,
        options: Option<&ModelsRefreshOptions>,
    ) -> BoxedFuture<'_, ModelsRefreshResult> {
        let state = Arc::clone(&self.state);
        let allow_network = options
            .and_then(|options| options.allow_network)
            .unwrap_or(true);
        let selected = options.and_then(|options| options.providers.clone());
        let force = options.and_then(|options| options.force);
        let caller_signal = operation_signal(options.and_then(|options| options.signal.as_ref()));
        Box::pin(async move {
            let mut result = ModelsRefreshResult::default();
            if caller_signal.is_cancelled() {
                result.aborted = true;
                return result;
            }
            let refreshable: Vec<Arc<dyn Provider>> = state
                .providers()
                .into_iter()
                .filter(|provider| provider.supports_refresh_models())
                .filter(|provider| {
                    selected
                        .as_ref()
                        .is_none_or(|selected| selected.iter().any(|id| id == provider.id()))
                })
                .collect();

            let mut tasks = JoinSet::new();
            for provider in refreshable {
                let state = Arc::clone(&state);
                let caller_signal = caller_signal.clone();
                tasks.spawn(async move {
                    let provider_id = provider.id().to_owned();
                    let (generation, controller) = state.begin_provider_refresh(&provider_id);
                    // The combined refresh signal: the caller's token and the
                    // supersede token, upstream's `AbortSignal.any`.
                    let signal = CancellationToken::new();
                    let watcher = {
                        let watcher_signal = signal.clone();
                        let watcher_caller = caller_signal.clone();
                        let watcher_controller = controller.clone();
                        tokio::spawn(async move {
                            tokio::select! {
                                () = watcher_caller.cancelled() => watcher_signal.cancel(),
                                () = watcher_controller.cancelled() => watcher_signal.cancel(),
                            }
                        })
                    };
                    let operation = refresh_provider_operation(
                        &state,
                        &provider,
                        allow_network,
                        force,
                        generation,
                        &signal,
                    );
                    let outcome = match race_with_abort_signal(operation, &signal).await {
                        Ok(()) => RefreshOutcome::Done,
                        Err(RaceError::Aborted(_)) => RefreshOutcome::Aborted,
                        Err(RaceError::Operation(error)) => RefreshOutcome::Failed(error),
                    };
                    watcher.abort();
                    state.end_provider_refresh(&provider_id, generation);
                    (provider_id, outcome)
                });
            }

            while !tasks.is_empty() {
                tokio::select! {
                    () = caller_signal.cancelled() => break,
                    Some(joined) = tasks.join_next() => {
                        if let Ok((provider_id, RefreshOutcome::Failed(error))) = joined {
                            result.errors.insert(provider_id, error);
                        }
                    },
                }
            }
            result.aborted = caller_signal.is_cancelled();
            result
        })
    }

    /// Run a provider-owned login flow and persist its returned credential,
    /// upstream's `login`.
    ///
    /// # Errors
    /// Unknown provider, unsupported login type, an aborted flow, or a
    /// credential-store failure.
    #[must_use]
    pub fn login(
        &self,
        provider_id: &str,
        auth_type: AuthType,
        interaction: AuthInteraction,
    ) -> BoxedFuture<'_, Result<Credential, ModelsFailure>> {
        let state = Arc::clone(&self.state);
        let provider_id = provider_id.to_owned();
        let signal = operation_signal(interaction.signal.as_ref());
        Box::pin(async move {
            if signal.is_cancelled() {
                return Err(ModelsFailure::Aborted(AbortError));
            }
            let Some(provider) = state.provider_of(&provider_id) else {
                return Err(ModelsFailure::Models(ModelsError::new(
                    ModelsErrorCode::Provider,
                    format!("Unknown provider: {provider_id}"),
                )));
            };
            let unsupported = || {
                ModelsFailure::Models(ModelsError::new(
                    ModelsErrorCode::Auth,
                    format!("{} does not support {auth_type} login", provider.name()),
                ))
            };
            let login_operation: BoxedFuture<
                'static,
                Result<Credential, crate::auth::types::AuthError>,
            > = match auth_type {
                AuthType::OAuth => {
                    let Some(oauth) = provider.auth().oauth.as_ref() else {
                        return Err(unsupported());
                    };
                    let normalized = ProviderAuthInteraction::from_interaction(
                        interaction.clone(),
                        signal.clone(),
                    );
                    let login = oauth.login.clone();
                    Box::pin(async move { (login)(normalized).await.map(Credential::OAuth) })
                }
                AuthType::ApiKey => {
                    let Some(login) = provider
                        .auth()
                        .api_key
                        .as_ref()
                        .and_then(|key| key.login.clone())
                    else {
                        return Err(unsupported());
                    };
                    let normalized = ProviderAuthInteraction::from_interaction(
                        interaction.clone(),
                        signal.clone(),
                    );
                    Box::pin(async move { (login)(normalized).await.map(Credential::ApiKey) })
                }
            };
            let credential = match race_with_abort_signal(login_operation, &signal).await {
                Ok(credential) => credential,
                Err(RaceError::Aborted(abort)) => return Err(ModelsFailure::Aborted(abort)),
                Err(RaceError::Operation(error)) => {
                    return Err(ModelsFailure::Models(ModelsError::with_cause(
                        ModelsErrorCode::Auth,
                        format!("Login failed for {provider_id}"),
                        error,
                    )));
                }
            };
            persist_login_credential(&state, &provider_id, credential, &signal).await
        })
    }

    /// Remove the stored credential for a provider, upstream's `logout`.
    ///
    /// # Errors
    /// The abort failure, or the wrapped store failure.
    #[must_use]
    pub fn logout(
        &self,
        provider_id: &str,
        options: Option<&AuthOptions>,
    ) -> BoxedFuture<'_, Result<(), ModelsFailure>> {
        let state = Arc::clone(&self.state);
        let provider_id = provider_id.to_owned();
        let signal = operation_signal(options.and_then(|options| options.signal.as_ref()));
        Box::pin(async move {
            if signal.is_cancelled() {
                return Err(ModelsFailure::Aborted(AbortError));
            }
            match state
                .credentials
                .delete(
                    &provider_id,
                    Some(&AuthOptions {
                        signal: Some(signal.clone()),
                    }),
                )
                .await
            {
                Ok(()) => Ok(()),
                Err(error) => {
                    if signal.is_cancelled() {
                        return Err(ModelsFailure::Aborted(AbortError));
                    }
                    Err(ModelsFailure::Models(ModelsError::with_cause(
                        ModelsErrorCode::Auth,
                        format!("Credential store delete failed for {provider_id}"),
                        error,
                    )))
                }
            }
        })
    }

    /// Stream through the owning provider with auth resolved and merged,
    /// upstream's `stream`. Returns the stream synchronously while auth
    /// resolution runs behind it.
    #[must_use]
    pub fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&ModelsStreamOptions>,
    ) -> AssistantMessageEventStream {
        let state = Arc::clone(&self.state);
        let model = model.clone();
        let context = context.clone();
        let options = options.cloned();
        let lazy_model = model.clone();
        lazy_stream(&lazy_model, move || {
            let state = Arc::clone(&state);
            Box::pin(async move {
                let provider = state.require_provider(&model)?;
                let (request_model, request_options) = apply_auth(&state, &model, options).await?;
                Ok(provider.stream(&request_model, &context, request_options.as_ref()))
            })
        })
    }

    /// Complete a request, upstream's `complete`: the stream's final
    /// assistant message, including error-terminal messages.
    #[must_use]
    pub fn complete(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&ModelsStreamOptions>,
    ) -> BoxedFuture<'_, AssistantMessage> {
        let stream = self.stream(model, context, options);
        Box::pin(async move { stream.result().await })
    }

    /// Stream a simple request through the owning provider, upstream's
    /// `streamSimple`.
    #[must_use]
    pub fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&ModelsSimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let state = Arc::clone(&self.state);
        let model = model.clone();
        let context = context.clone();
        let options = options.cloned();
        let lazy_model = model.clone();
        lazy_stream(&lazy_model, move || {
            let state = Arc::clone(&state);
            Box::pin(async move {
                let provider = state.require_provider(&model)?;
                let (request_model, request_options) = apply_auth(&state, &model, options).await?;
                Ok(provider.stream_simple(&request_model, &context, request_options.as_ref()))
            })
        })
    }

    /// Complete a simple request, upstream's `completeSimple`.
    #[must_use]
    pub fn complete_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&ModelsSimpleStreamOptions>,
    ) -> BoxedFuture<'_, AssistantMessage> {
        let stream = self.stream_simple(model, context, options);
        Box::pin(async move { stream.result().await })
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
        let state = Arc::clone(&self.state);
        let model = model.clone();
        let handle = handle.clone();
        let options = options.cloned();
        let lazy_model = model.clone();
        lazy_stream(&lazy_model, move || {
            let state = Arc::clone(&state);
            Box::pin(async move {
                let provider = state.require_provider(&model)?;
                let missing = || {
                    let error: crate::api::lazy::LazyStreamError = ModelsError::new(
                        ModelsErrorCode::Provider,
                        format!(
                            "Provider {} does not support deferred responses",
                            model.provider.0
                        ),
                    )
                    .into();
                    error
                };
                if !provider.supports_fetch_deferred() {
                    return Err(missing());
                }
                let (request_model, request_options) = apply_auth(&state, &model, options)
                    .await
                    .map_err(failure_into_error)?;
                provider
                    .fetch_deferred(&request_model, &handle, request_options.as_ref())
                    .ok_or_else(missing)
            })
        })
    }

    /// Fetch a deferred response to completion, upstream's `fetchDeferred`.
    #[must_use]
    pub fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: Option<&ModelsDeferredFetchOptions>,
    ) -> BoxedFuture<'_, AssistantMessage> {
        let stream = self.stream_deferred(model, handle, options);
        Box::pin(async move { stream.result().await })
    }

    /// Cancel a deferred response, upstream's `cancelDeferred`.
    ///
    /// # Errors
    /// Unknown provider, unsupported operation, auth failure, or the
    /// provider's own failure.
    #[must_use]
    pub fn cancel_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: Option<&ModelsDeferredCancelOptions>,
    ) -> BoxedFuture<'_, Result<(), ModelsFailure>> {
        let state = Arc::clone(&self.state);
        let model = model.clone();
        let handle = handle.clone();
        let options = options.cloned();
        Box::pin(async move {
            let provider = state.require_provider(&model)?;
            if !provider.supports_cancel_deferred() {
                return Err(ModelsFailure::Models(ModelsError::new(
                    ModelsErrorCode::Provider,
                    format!(
                        "Provider {} does not support deferred responses",
                        model.provider.0
                    ),
                )));
            }
            let (request_model, request_options) = apply_auth(&state, &model, options).await?;
            let Some(cancel) =
                provider.cancel_deferred(&request_model, &handle, request_options.as_ref())
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
        })
    }
}

/// Why one provider's refresh settled.
enum RefreshOutcome {
    /// The phase completed.
    Done,
    /// The refresh signal aborted the operation.
    Aborted,
    /// The refresh failed; recorded unless the operation was aborted.
    Failed(ProviderError),
}

/// One provider's refresh operation: restore cached state, then refresh over
/// the network when allowed, upstream's per-provider async body.
async fn refresh_provider_operation(
    state: &Arc<ModelsState>,
    provider: &Arc<dyn Provider>,
    allow_network: bool,
    force: Option<bool>,
    generation: u64,
    signal: &CancellationToken,
) -> Result<(), ProviderError> {
    let read = state
        .credentials
        .read(
            provider.id(),
            Some(&AuthOptions {
                signal: Some(signal.clone()),
            }),
        )
        .await;
    let stored_credential = read.as_ref().ok().cloned().flatten();
    // Restore cached provider state before auth resolution or network access.
    run_provider_refresh_phase(
        state,
        provider,
        stored_credential.clone(),
        false,
        None,
        generation,
        signal,
    )
    .await?;
    let stored_credential = read?;
    if !allow_network || signal.is_cancelled() {
        return Ok(());
    }
    let credential = state
        .resolve_refresh_credential(provider, stored_credential, signal)
        .await?;
    let Some(credential) = credential else {
        return Ok(());
    };
    run_provider_refresh_phase(
        state,
        provider,
        Some(credential),
        true,
        force,
        generation,
        signal,
    )
    .await
}

/// Run one provider refresh phase: restore the persisted catalog and hand the
/// provider its refresh context, upstream's `runProviderRefreshPhase`.
async fn run_provider_refresh_phase(
    state: &Arc<ModelsState>,
    provider: &Arc<dyn Provider>,
    credential: Option<Credential>,
    allow_network: bool,
    force: Option<bool>,
    generation: u64,
    signal: &CancellationToken,
) -> Result<(), ProviderError> {
    let stored = state
        .models_store
        .read(
            provider.id(),
            Some(&ModelsStoreOptions {
                signal: Some(signal.clone()),
            }),
        )
        .await?;
    let publish: PublishFn = {
        let state = Arc::clone(state);
        let provider_id = provider.id().to_owned();
        let signal = signal.clone();
        Arc::new(move |publication| {
            let state = Arc::clone(&state);
            let provider_id = provider_id.clone();
            let signal = signal.clone();
            Box::pin(async move {
                state
                    .publish_provider_models(&provider_id, generation, &signal, publication)
                    .await
            })
        })
    };
    let context = RefreshModelsContext {
        credential,
        stored,
        publish,
        allow_network,
        force: if allow_network { force } else { None },
        signal: signal.clone(),
    };
    provider.refresh_models(context).await
}

/// Check whether one provider has complete auth configuration, upstream's
/// `checkProviderAuth`.
async fn check_provider_auth(
    state: &ModelsState,
    provider: &Arc<dyn Provider>,
    credential: Option<Credential>,
    signal: &CancellationToken,
) -> Result<Option<AuthCheck>, ModelsError> {
    if credential
        .as_ref()
        .is_some_and(|credential| credential.as_oauth().is_some())
    {
        return Ok(provider.auth().oauth.as_ref().map(|_| AuthCheck {
            source: Some("OAuth".to_owned()),
            auth_type: AuthType::OAuth,
        }));
    }
    let Some(api_key) = provider.auth().api_key.as_ref() else {
        return Ok(None);
    };
    if let Some(check) = &api_key.check {
        let input = ApiKeyAuthInput {
            ctx: Arc::clone(&state.auth_context),
            credential: credential
                .as_ref()
                .and_then(|credential| credential.as_api_key_credential())
                .cloned(),
            signal: signal.clone(),
        };
        return (check)(input).await.map_err(|error| {
            ModelsError::with_cause(
                ModelsErrorCode::Auth,
                format!("API key auth check failed for provider {}", provider.id()),
                error,
            )
        });
    }

    let resolution = resolve_provider_auth(
        provider.id(),
        provider.auth(),
        &state.credentials,
        &state.auth_context,
        Some(&AuthResolutionOverrides {
            signal: Some(signal.clone()),
            ..AuthResolutionOverrides::default()
        }),
    )
    .await
    .map_err(failure_into_error)?;
    Ok(resolution.map(|resolution| AuthCheck {
        source: resolution.source,
        auth_type: AuthType::ApiKey,
    }))
}

/// Read a credential for an auth check, wrapping store failures, upstream's
/// inline `readCredential` use.
async fn read_credential_for_check(
    state: &ModelsState,
    provider_id: &str,
    signal: &CancellationToken,
) -> Result<Option<Credential>, ModelsError> {
    state
        .credentials
        .read(
            provider_id,
            Some(&AuthOptions {
                signal: Some(signal.clone()),
            }),
        )
        .await
        .map_err(|error| {
            ModelsError::with_cause(
                ModelsErrorCode::Auth,
                format!("Credential store read failed for {provider_id}"),
                error,
            )
        })
}

/// The `checkAuth` body, raced against the operation signal by
/// [`Models::check_auth`].
async fn check_auth_operation(
    state: &ModelsState,
    provider_id: &str,
    signal: &CancellationToken,
) -> Result<Option<AuthCheck>, ModelsError> {
    if signal.is_cancelled() {
        return Err(ModelsError::new(
            ModelsErrorCode::Stream,
            "operation aborted",
        ));
    }
    let Some(provider) = state.provider_of(provider_id) else {
        return Ok(None);
    };
    let credential = read_credential_for_check(state, provider_id, signal).await?;
    check_provider_auth(state, &provider, credential, signal).await
}

/// The `getAvailable` body: auth-check every selected provider, then collect
/// the models of the configured ones.
async fn available_operation(
    state: &ModelsState,
    provider_id: Option<&str>,
    signal: &CancellationToken,
) -> Result<Vec<Model>, ModelsError> {
    if signal.is_cancelled() {
        return Err(ModelsError::new(
            ModelsErrorCode::Stream,
            "operation aborted",
        ));
    }
    let providers: Vec<Arc<dyn Provider>> = provider_id.map_or_else(
        || state.providers(),
        |provider_id| state.provider_of(provider_id).into_iter().collect(),
    );
    let mut models = Vec::new();
    for provider in providers {
        let credential = read_credential_for_check(state, provider.id(), signal).await?;
        let auth = check_provider_auth(state, &provider, credential.clone(), signal).await?;
        if auth.is_none() {
            continue;
        }
        let Ok(provider_models) = provider.get_models() else {
            continue;
        };
        models.extend(provider.filter_models(provider_models, credential.as_ref()));
    }
    Ok(models)
}

/// Persist a login credential through the store's single write path, upstream
/// `login`'s mutation race: login resolves once the mutation has started, an
/// abort before then rejects, and a failed mutation surfaces as a
/// [`ModelsError`].
async fn persist_login_credential(
    state: &Arc<ModelsState>,
    provider_id: &str,
    credential: Credential,
    signal: &CancellationToken,
) -> Result<Credential, ModelsFailure> {
    let (started_sender, started_receiver) = tokio::sync::oneshot::channel::<()>();
    let mutation_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stored_credential = credential.clone();
    let started_flag = Arc::clone(&mutation_started);
    let modify: CredentialModifyFn = Box::new(move |_current| {
        started_flag.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = started_sender.send(());
        Box::pin(async move { Ok(Some(stored_credential)) })
    });
    let options = AuthOptions {
        signal: Some(signal.clone()),
    };
    let mutation = state
        .credentials
        .modify(provider_id, modify, Some(&options));
    tokio::pin!(mutation);
    let outcome: Result<Option<Credential>, crate::auth::types::AuthError> = tokio::select! {
        () = signal.cancelled() => {
            if !mutation_started.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(ModelsFailure::Aborted(AbortError));
            }
            mutation.await
        }
        started = started_receiver => {
            let _ = started;
            mutation.await
        }
        completed = &mut mutation => completed,
    };
    match outcome {
        _ if signal.is_cancelled() => Err(ModelsFailure::Aborted(AbortError)),
        Ok(_) => Ok(credential),
        Err(error) => Err(ModelsFailure::Models(ModelsError::with_cause(
            ModelsErrorCode::Auth,
            format!("Credential store modify failed for {provider_id}"),
            error,
        ))),
    }
}

/// Resolve provider auth for a model and merge static model headers into the
/// result, upstream's `getAuth(model)` body.
async fn resolve_auth_for_model(
    state: &ModelsState,
    model: &Model,
    explicit_api_key: Option<String>,
    explicit_env: Option<crate::types::ProviderEnv>,
    signal: Option<CancellationToken>,
) -> Result<Option<AuthResult>, ModelsFailure> {
    let Some(provider) = state.provider_of(&model.provider.0) else {
        return Ok(None);
    };
    let overrides = AuthResolutionOverrides {
        api_key: explicit_api_key,
        env: explicit_env,
        min_oauth_validity_ms: None,
        signal,
    };
    let result = resolve_provider_auth(
        provider.id(),
        provider.auth(),
        &state.credentials,
        &state.auth_context,
        Some(&overrides),
    )
    .await?;
    let Some(result) = result else {
        return Ok(None);
    };
    let Some(headers) = &model.headers else {
        return Ok(Some(result));
    };
    let model_headers: ProviderHeaders = headers
        .iter()
        .map(|(name, value)| (name.clone(), Some(value.clone())))
        .collect();
    Ok(Some(AuthResult {
        auth: ModelAuth {
            headers: merge_headers(result.auth.headers.as_ref(), Some(&model_headers)),
            ..result.auth
        },
        env: result.env,
        source: result.source,
    }))
}

/// Resolve provider auth for a model and merge it into the request options,
/// upstream's `applyAuth`: explicit request options win per field; the
/// Models-only transform runs last.
async fn apply_auth<T: AuthCarrier + Default>(
    state: &ModelsState,
    model: &Model,
    options: Option<WithTransforms<T>>,
) -> Result<(Model, Option<T>), ModelsFailure> {
    state
        .require_provider(model)
        .map_err(ModelsFailure::Models)?;
    let (explicit_api_key, explicit_env, explicit_headers, signal) = options.as_ref().map_or_else(
        || (None, None, None, None),
        |wrapper| {
            (
                wrapper.options.api_key_slot().cloned(),
                wrapper.options.env_slot().cloned(),
                wrapper.options.headers_slot().cloned(),
                wrapper.options.transport().signal.clone(),
            )
        },
    );
    let resolution = resolve_auth_for_model(
        state,
        model,
        explicit_api_key.clone(),
        explicit_env.clone(),
        signal,
    )
    .await?;
    let Some(resolution) = resolution else {
        return Err(ModelsFailure::Models(ModelsError::new(
            ModelsErrorCode::Auth,
            format!("Provider is not configured: {}", model.provider.0),
        )));
    };
    let auth = resolution.auth;

    // Explicit request options win per-field; the Models-only transform runs
    // last.
    let api_key = explicit_api_key.or_else(|| auth.api_key.clone());
    let mut headers = merge_headers(auth.headers.as_ref(), explicit_headers.as_ref());
    if let Some(wrapper) = &options
        && let Some(transform) = &wrapper.transform_headers
    {
        headers = Some((transform)(headers.unwrap_or_default()).await);
    }
    let env = match (&resolution.env, &explicit_env) {
        (None, None) => None,
        (resolved, explicit) => {
            let mut merged = resolved.clone().unwrap_or_default();
            merged.extend(explicit.clone().unwrap_or_default());
            Some(merged)
        }
    };
    let request_model = auth.base_url.as_ref().map_or_else(
        || model.clone(),
        |base_url| {
            let mut request_model = model.clone();
            request_model.base_url.clone_from(base_url);
            request_model
        },
    );
    let mut request_options = match options {
        Some(wrapper) => wrapper.options,
        None => T::default(),
    };
    *request_options.api_key_slot_mut() = api_key;
    *request_options.headers_slot_mut() = headers;
    *request_options.env_slot_mut() = env;
    Ok((request_model, Some(request_options)))
}

/// A `Models` collection, upstream's `createModels(options)`.
#[must_use]
pub fn create_models(options: Option<CreateModelsOptions>) -> Models {
    let options = options.unwrap_or_default();
    Models {
        state: Arc::new(ModelsState {
            providers: std::sync::RwLock::new(BTreeMap::new()),
            credentials: options
                .credentials
                .unwrap_or_else(|| Arc::new(InMemoryCredentialStore::default())),
            models_store: options
                .models_store
                .unwrap_or_else(|| Arc::new(InMemoryModelsStore::default())),
            auth_context: options
                .auth_context
                .unwrap_or_else(crate::auth::context::default_provider_auth_context),
            generations: Mutex::new(BTreeMap::new()),
            controllers: Mutex::new(BTreeMap::new()),
            publication_locks: Mutex::new(BTreeMap::new()),
        }),
    }
}

/// Runtime-checked narrowing for dynamically looked-up models, upstream's
/// `hasApi`: whether the model speaks `api`.
#[must_use]
pub fn has_api(model: &Model, api: &Api) -> bool {
    model.api == *api
}

/// Apply a model's pricing to a usage record, upstream's `calculateCost`.
///
/// Tiered pricing applies the highest matching input threshold to the full
/// request; Anthropic charges 2x base input for 1h cache writes. Returns the
/// computed cost, which is also stored into `usage.cost`.
#[expect(
    clippy::cast_precision_loss,
    reason = "token counts are far below the f64 exact-integer range"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "upstream's cost arithmetic is kept verbatim; mul_add rounding would differ"
)]
pub fn calculate_cost(model: &Model, usage: &mut Usage) -> UsageCost {
    let input_tokens = usage
        .input
        .saturating_add(usage.cache_read)
        .saturating_add(usage.cache_write);
    let mut rates = model.cost.rates;
    let mut matched_threshold: Option<u64> = None;
    for tier in model.cost.tiers.iter().flatten() {
        if input_tokens > tier.input_tokens_above
            && matched_threshold.is_none_or(|threshold| tier.input_tokens_above > threshold)
        {
            rates = tier.rates;
            matched_threshold = Some(tier.input_tokens_above);
        }
    }

    // Anthropic charges 2x base input for 1h cache writes.
    let long_write = usage.cache_write_1h.unwrap_or(0);
    let short_write = usage.cache_write.saturating_sub(long_write);
    let mut cost = UsageCost {
        input: rates.input / 1_000_000.0 * usage.input as f64,
        output: rates.output / 1_000_000.0 * usage.output as f64,
        cache_read: rates.cache_read / 1_000_000.0 * usage.cache_read as f64,
        cache_write: (rates.cache_write * short_write as f64
            + rates.input * 2.0 * long_write as f64)
            / 1_000_000.0,
        total: 0.0,
    };
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write;
    usage.cost = cost;
    cost
}

/// The thinking levels a model supports in order, upstream's
/// `EXTENDED_THINKING_LEVELS`.
const EXTENDED_THINKING_LEVELS: [ModelThinkingLevel; 7] = [
    ModelThinkingLevel::Off,
    ModelThinkingLevel::Minimal,
    ModelThinkingLevel::Low,
    ModelThinkingLevel::Medium,
    ModelThinkingLevel::High,
    ModelThinkingLevel::Xhigh,
    ModelThinkingLevel::Max,
];

/// The thinking levels a model supports, upstream's
/// `getSupportedThinkingLevels`.
#[must_use]
pub fn get_supported_thinking_levels(model: &Model) -> Vec<ModelThinkingLevel> {
    if !model.reasoning {
        return vec![ModelThinkingLevel::Off];
    }
    EXTENDED_THINKING_LEVELS
        .iter()
        .filter(|level| {
            let mapped = model
                .thinking_level_map
                .as_ref()
                .and_then(|map| map.get(*level));
            if mapped == Some(&None) {
                return false;
            }
            if matches!(level, ModelThinkingLevel::Xhigh | ModelThinkingLevel::Max) {
                return mapped.is_some();
            }
            true
        })
        .copied()
        .collect()
}

/// Clamp a requested thinking level to one the model supports, upstream's
/// `clampThinkingLevel`: prefer the nearest higher supported level, then the
/// nearest lower one.
#[must_use]
pub fn clamp_thinking_level(model: &Model, level: ModelThinkingLevel) -> ModelThinkingLevel {
    let available_levels = get_supported_thinking_levels(model);
    if available_levels.contains(&level) {
        return level;
    }
    let Some(requested_index) = EXTENDED_THINKING_LEVELS
        .iter()
        .position(|candidate| *candidate == level)
    else {
        return available_levels
            .first()
            .copied()
            .unwrap_or(ModelThinkingLevel::Off);
    };
    for candidate in &EXTENDED_THINKING_LEVELS[requested_index..] {
        if available_levels.contains(candidate) {
            return *candidate;
        }
    }
    for candidate in EXTENDED_THINKING_LEVELS[..requested_index].iter().rev() {
        if available_levels.contains(candidate) {
            return *candidate;
        }
    }
    available_levels
        .first()
        .copied()
        .unwrap_or(ModelThinkingLevel::Off)
}

/// Check if two models are equal by comparing both their id and provider,
/// upstream's `modelsAreEqual`. Returns false if either model is missing.
#[must_use]
pub fn models_are_equal(a: Option<&Model>, b: Option<&Model>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.id == b.id && a.provider == b.provider,
        _ => false,
    }
}

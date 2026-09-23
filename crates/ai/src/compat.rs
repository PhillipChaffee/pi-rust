//! The deprecated global compat API, ported from `packages/ai/src/compat.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream preserves the old global pi-ai surface — api-dispatch
//! [`stream`]/[`complete`] with env-key injection, the api-registry, the
//! generated catalog reads, the per-API lazy factories, and image
//! generation — for apps importing the package's compat entrypoint, and
//! slates the module for deletion with the coding-agent `ModelManager`
//! migration; the port carries it anyway, because parity is parity. The
//! deprecated per-API aliases ride with it as [`legacy_api_aliases`].
//!
//! Porting restatements:
//!
//! - Upstream's re-export block (`export * from "./index.ts"` and friends)
//!   duplicates surfaces the crate root already carries; this module adds
//!   only what compat itself contributes — the registry, the dispatch, the
//!   deprecated catalog reads, the factory re-exports, and the barrel
//!   re-exports of the env-key, image-model, image-generation, and
//!   image-registry surfaces. The faux registration rides here as
//!   [`register_faux_provider`] because upstream defines it in this module.
//!   Upstream's `BuiltinProvider` type union restates as the crate's
//!   string-typed provider ids (`ProviderId`), a type alias with no runtime
//!   presence; the `ApiStreamFunction`/`ApiStreamSimpleFunction` aliases
//!   describe the shape the registry erases, which the
//!   [`ProviderStreams`] trait object carries.
//! - Upstream registers the builtin api implementations at module load and
//!   constructs the compat catalog once; the port registers lazily on first
//!   registry use and builds the catalog on first routing need, and
//!   [`register_api_provider`] still wins because
//!   [`register_built_in_api_providers`] never clobbers an existing entry.
//! - Upstream's `wrapStream`/`wrapStreamSimple` mismatch guard wraps every
//!   registered provider: a registered provider reached with a mismatched
//!   model's api settles its stream as an error carrying upstream's
//!   `Mismatched api` message. The dispatch looks the registry up by the
//!   model's api, so the guard binds the deprecated surface's direct
//!   consumers.
//! - Upstream's synchronous throws (`No API provider registered for api`) —
//!   impossible from a sync Rust fn returning a stream — settle the returned
//!   stream as the error message, the crate's sync-dispatch-failure shape.
//! - The cloudflare Models-runtime detour passes the request options through
//!   a type-level cast upstream; the port wraps them in the Models runtime's
//!   [`WithTransforms`] with no transforms, the same wire shape.
//! - `registerFauxProvider` derives its registration source id from
//!   `Math.random()` upstream; the port draws a process-unique counter, the
//!   same uniqueness the source-scoped unregister needs.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::api::lazy;
use crate::models::{AuthCarrier, Models, Provider, WithTransforms};
use crate::types::{
    Api, AssistantMessage, AssistantMessageEvent, BoxedFuture, Context, Model, ProviderStreams,
    SimpleStreamOptions, StopReason, StreamOptions,
};
use crate::utils::event_stream::AssistantMessageEventStream;

/// The deprecated per-API stream aliases, upstream's `legacy-api-aliases.ts`
/// re-export.
pub use crate::legacy_api_aliases;

/// The env-key surface, upstream's `env-api-keys.ts` re-export.
pub use crate::env_api_keys::{
    ANTHROPIC_API_KEY_ENV, ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_OAUTH_TOKEN_ENV, find_env_keys,
    get_env_api_key,
};
/// The static image-catalog reads, upstream's `image-models.ts` re-export.
pub use crate::image_models::{get_image_model, get_image_models, get_image_providers};
/// The image-generation dispatch, upstream's `images.ts` re-export.
pub use crate::images::generate_images;
/// The image-API registry, upstream's `images-api-registry.ts` re-export.
pub use crate::images_api_registry::{
    ImagesApiProvider, get_images_api_provider, register_images_api_provider,
};
/// The builtin image-API seed, upstream's
/// `providers/images/register-builtins.ts` re-export.
pub use crate::providers::images::{
    generate_images_open_router, register_built_in_images_api_providers,
};

/// The Anthropic Messages wire-API factory, upstream's
/// `anthropicMessagesApi`.
pub use crate::api::anthropic_messages as anthropic_messages_api;
/// The Azure OpenAI Responses wire-API factory, upstream's
/// `azureOpenAIResponsesApi`.
pub use crate::api::azure_openai_responses as azure_openai_responses_api;
/// The Bedrock Converse Stream wire-API factory, upstream's
/// `bedrockConverseStreamApi`.
pub use crate::api::bedrock_converse_stream as bedrock_converse_stream_api;
/// The Google Generative AI wire-API factory, upstream's
/// `googleGenerativeAIApi`.
pub use crate::api::google_generative_ai as google_generative_ai_api;
/// The Google Vertex AI wire-API factory, upstream's `googleVertexApi`.
pub use crate::api::google_vertex as google_vertex_api;
/// The Mistral Conversations wire-API factory, upstream's
/// `mistralConversationsApi`.
pub use crate::api::mistral_conversations as mistral_conversations_api;
/// The OpenAI Codex Responses wire-API factory, upstream's
/// `openAICodexResponsesApi`.
pub use crate::api::openai_codex_responses as openai_codex_responses_api;
/// The OpenAI Completions wire-API factory, upstream's
/// `openAICompletionsApi`.
pub use crate::api::openai_completions as openai_completions_api;
/// The OpenAI Responses wire-API factory, upstream's `openAIResponsesApi`.
pub use crate::api::openai_responses as openai_responses_api;
/// The pi-messages wire-API factory, upstream's `piMessagesApi`.
pub use crate::api::pi_messages as pi_messages_api;

/// A registered api implementation, upstream's `ApiProvider`: the wire-API id
/// and the stream pair that answers it.
pub struct ApiProvider {
    /// The wire-API id this provider serves, upstream's `api`.
    pub api: Api,
    /// The stream pair, upstream's `stream`/`streamSimple` functions.
    pub streams: Arc<dyn ProviderStreams>,
}

impl std::fmt::Debug for ApiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiProvider")
            .field("api", &self.api)
            .field("streams", &"set")
            .finish()
    }
}

/// A registry entry: the registered (api-checked) streams and the source that
/// registered them, upstream's `RegisteredApiProvider`.
struct RegistryEntry {
    streams: Arc<dyn ProviderStreams>,
    source_id: Option<String>,
}

/// The api-registry, upstream's `apiProviderRegistry` map: registrations
/// ordered by api id, where upstream's insertion-ordered map reflects its
/// registration sequence.
static API_PROVIDER_REGISTRY: RwLock<BTreeMap<Api, RegistryEntry>> = RwLock::new(BTreeMap::new());

/// The live builtin registry instances, upstream's
/// `builtinApiProviderInstances` map, rebound on every builtin registration.
static BUILTIN_API_PROVIDER_INSTANCES: RwLock<BTreeMap<Api, Arc<dyn ProviderStreams>>> =
    RwLock::new(BTreeMap::new());

/// The one-time builtin registration flag, upstream's module-load
/// `registerBuiltInApiProviders()` call.
static BUILTINS_REGISTERED: OnceLock<()> = OnceLock::new();

/// The compat catalog, upstream's module-load `builtinModels()` binding.
static COMPAT_MODELS: OnceLock<Models> = OnceLock::new();

/// The api-key value marking ambient auth, upstream's
/// `AMBIENT_AUTH_MARKER`: a provider whose env key resolves to this marker
/// authenticates by its own means and must not receive the literal string.
const AMBIENT_AUTH_MARKER: &str = "<authenticated>";

/// One builtin table entry: the api id and its stream constructor.
type BuiltinApi = (&'static str, fn() -> Arc<dyn ProviderStreams>);

/// The builtin api implementations, upstream's `BUILTIN_APIS` table.
const BUILTIN_APIS: [BuiltinApi; 10] = [
    ("anthropic-messages", crate::api::anthropic_messages),
    ("openai-completions", crate::api::openai_completions),
    ("openai-responses", crate::api::openai_responses),
    ("openai-codex-responses", crate::api::openai_codex_responses),
    ("azure-openai-responses", crate::api::azure_openai_responses),
    ("google-generative-ai", crate::api::google_generative_ai),
    ("google-vertex", crate::api::google_vertex),
    ("mistral-conversations", crate::api::mistral_conversations),
    (
        "bedrock-converse-stream",
        crate::api::bedrock_converse_stream,
    ),
    ("pi-messages", crate::api::pi_messages),
];

/// The builtin api ids and their constructors, upstream's `BUILTIN_APIS`
/// table, exposed so the registration surface can be enumerated.
#[must_use]
pub fn builtin_apis() -> &'static [BuiltinApi] {
    &BUILTIN_APIS
}

/// Deprecated static catalog read. Use
/// [`crate::providers::catalog::get_builtin_model`] or
/// [`crate::models::Models::model`].
#[deprecated(note = "static catalog read; use `pi_ai::providers::catalog::get_builtin_model`")]
#[must_use]
pub fn get_model(provider: &str, model_id: &str) -> Option<Model> {
    crate::providers::catalog::get_builtin_model(provider, model_id)
}

/// Deprecated static catalog read. Use
/// [`crate::providers::all::builtin_models_of`] or
/// [`crate::models::Models::models`].
#[deprecated(note = "static catalog read; use `pi_ai::providers::all::builtin_models_of`")]
#[must_use]
pub fn get_models(provider: &str) -> Vec<Model> {
    crate::providers::all::builtin_models_of(provider)
}

/// Deprecated static catalog read. Use
/// [`crate::providers::all::builtin_catalog_provider_ids`] or
/// [`crate::models::Models::providers`].
#[deprecated(
    note = "static catalog read; use `pi_ai::providers::all::builtin_catalog_provider_ids`"
)]
#[must_use]
pub fn get_providers() -> Vec<String> {
    crate::providers::all::builtin_catalog_provider_ids()
}

/// Register an api implementation, replacing any existing registration for
/// its api id, upstream's `registerApiProvider`.
///
/// The streams are wrapped with the api check at registration, upstream's
/// `wrapStream` pair. The builtin registration runs first — upstream
/// registers builtins at module load, before any override can exist, so a
/// first registration landing after an override would rebind the routing
/// check.
pub fn register_api_provider(provider: ApiProvider, source_id: Option<&str>) {
    ensure_builtins_registered();
    let streams = wrap_streams(&provider.api, provider.streams);
    API_PROVIDER_REGISTRY
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(
            provider.api,
            RegistryEntry {
                streams,
                source_id: source_id.map(str::to_owned),
            },
        );
}

/// The api implementation registered for an api id, upstream's
/// `getApiProvider`.
#[must_use]
pub fn get_api_provider(api: &Api) -> Option<Arc<dyn ProviderStreams>> {
    ensure_builtins_registered();
    API_PROVIDER_REGISTRY
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(api)
        .map(|entry| Arc::clone(&entry.streams))
}

/// Every registered api implementation, upstream's `getApiProviders`.
#[must_use]
pub fn get_api_providers() -> Vec<ApiProvider> {
    ensure_builtins_registered();
    API_PROVIDER_REGISTRY
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .map(|(api, entry)| ApiProvider {
            api: api.clone(),
            streams: Arc::clone(&entry.streams),
        })
        .collect()
}

/// Remove every registration made under `source_id`, upstream's
/// `unregisterApiProviders`.
pub fn unregister_api_providers(source_id: &str) {
    API_PROVIDER_REGISTRY
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .retain(|_, entry| entry.source_id.as_deref() != Some(source_id));
}

/// Drop every registration, upstream's private `clearApiProviders`.
fn clear_api_providers() {
    API_PROVIDER_REGISTRY
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

/// The registration source ids handed out so far, upstream's
/// `Math.random()`-derived source id: process-unique, which is the only
/// property the source-scoped unregister needs.
static FAUX_SOURCE_IDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Register a faux provider into the api-registry, upstream's
/// `registerFauxProvider`.
///
/// The core streams under its api id with a fresh source id, and the
/// returned registration scripts responses and unregisters its own entries.
#[must_use]
pub fn register_faux_provider(
    options: crate::providers::faux::RegisterFauxProviderOptions,
) -> crate::providers::faux::FauxProviderRegistration {
    let core = crate::providers::faux::create_faux_core(options);
    let source_id = format!(
        "faux-provider-{}",
        FAUX_SOURCE_IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
    );
    register_api_provider(
        ApiProvider {
            api: Api::from(core.api().to_owned()),
            streams: Arc::new(core.clone()),
        },
        Some(&source_id),
    );
    crate::providers::faux::FauxProviderRegistration::new(core, source_id)
}

/// Register the builtin api implementations without clobbering existing
/// entries — compat may reach a registry where a test or extension already
/// registered an override for a builtin api id.
///
/// The builtin instances the routing check compares against rebind on every
/// call, upstream's `registerBuiltInApiProviders`.
pub fn register_built_in_api_providers() {
    let instances = {
        let mut registry = API_PROVIDER_REGISTRY
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut instances = BTreeMap::new();
        for (api, constructor) in BUILTIN_APIS {
            let api = Api::from(api);
            let entry = registry
                .entry(api.clone())
                .or_insert_with(|| RegistryEntry {
                    streams: wrap_streams(&api, constructor()),
                    source_id: None,
                });
            instances.insert(api, Arc::clone(&entry.streams));
        }
        drop(registry);
        instances
    };
    *BUILTIN_API_PROVIDER_INSTANCES
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = instances;
}

/// Reset the registry to the builtin api implementations, upstream's
/// `resetApiProviders`.
pub fn reset_api_providers() {
    clear_api_providers();
    register_built_in_api_providers();
}

/// The one-time builtin registration, upstream's module-load call.
fn ensure_builtins_registered() {
    BUILTINS_REGISTERED.get_or_init(register_built_in_api_providers);
}

/// Wrap the registered streams with the api check, upstream's
/// `wrapStream`/`wrapStreamSimple` pair: the registry erases the factory's
/// api typing into the trait object, so the wrapper re-adds the runtime
/// check the open `Api` union needs.
fn wrap_streams(api: &Api, streams: Arc<dyn ProviderStreams>) -> Arc<dyn ProviderStreams> {
    Arc::new(RegistryStreams {
        api: api.clone(),
        inner: streams,
    })
}

/// The registered streams with the api check re-added, upstream's wrapped
/// `ApiProviderInternal.stream`/`streamSimple`.
struct RegistryStreams {
    api: Api,
    inner: Arc<dyn ProviderStreams>,
}

impl ProviderStreams for RegistryStreams {
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        if model.api != self.api {
            return dispatch_failure(
                model,
                format!("Mismatched api: {} expected {}", model.api, self.api),
            );
        }
        self.inner.stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        if model.api != self.api {
            return dispatch_failure(
                model,
                format!("Mismatched api: {} expected {}", model.api, self.api),
            );
        }
        self.inner.stream_simple(model, context, options)
    }
}

/// The failure a synchronous dispatch problem reports, settled the way the
/// crate settles sync dispatch failures: the fresh accumulator as an error
/// event carrying the message, upstream's thrown `Error`.
fn dispatch_failure(model: &Model, message: String) -> AssistantMessageEventStream {
    let stream = crate::utils::event_stream::assistant_message_event_stream();
    let error: lazy::LazyStreamError = Box::new(DispatchError(message));
    let static_error: &(dyn std::error::Error + 'static) = error.as_ref();
    let failing = lazy::setup_error_message(model, static_error);
    stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: failing.clone(),
    });
    stream.end(Some(&failing));
    stream
}

/// The dispatch failure, upstream's thrown `Error`.
#[derive(Debug)]
struct DispatchError(String);

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DispatchError {}

/// The builtin catalog provider a model dispatches through, upstream's
/// `getBuiltinProviderForModel`: `Some` only when the model's api still
/// dispatches through the builtin registry instance — no override stands —
/// and the model's provider is a builtin catalog provider carrying a model
/// of that api.
fn get_builtin_provider_for_model(model: &Model) -> Option<Arc<dyn Provider>> {
    let registered = get_api_provider(&model.api)?;
    let builtin = BUILTIN_API_PROVIDER_INSTANCES
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&model.api)
        .cloned()?;
    if !Arc::ptr_eq(&registered, &builtin) {
        return None;
    }
    let provider = compat_models().provider(&model.provider.0)?;
    // Upstream propagates a getModels throw; every builtin provider's read is
    // a catalog read that cannot fail, so the failure rides as no models.
    let carries_api = provider
        .get_models()
        .unwrap_or_default()
        .iter()
        .any(|candidate| candidate.api == model.api);
    carries_api.then_some(provider)
}

/// The compat catalog, upstream's module-load `builtinModels()` binding,
/// built on first routing need.
fn compat_models() -> &'static Models {
    COMPAT_MODELS.get_or_init(|| crate::providers::all::builtin_models(None))
}

/// Whether the options carry an explicit, non-blank api key, upstream's
/// `hasExplicitApiKey`.
fn has_explicit_api_key<T: AuthCarrier>(options: Option<&T>) -> bool {
    options
        .and_then(|options| options.api_key_slot())
        .is_some_and(|key| !key.trim().is_empty())
}

/// Merge the provider's env-resolved api key into the request options when
/// the caller did not pass one explicitly, upstream's `withEnvApiKey`. An
/// env key equal to the ambient-auth marker is left alone: the provider
/// marked `<authenticated>` resolves credentials itself.
fn with_env_api_key<T: AuthCarrier + Default>(model: &Model, options: Option<&T>) -> Option<T> {
    if has_explicit_api_key(options) {
        return options.cloned();
    }
    let env = options.and_then(|options| options.env_slot());
    match get_env_api_key(&model.provider.0, env) {
        Some(api_key) if api_key != AMBIENT_AUTH_MARKER => {
            let mut resolved = options.cloned().unwrap_or_default();
            *resolved.api_key_slot_mut() = Some(api_key);
            Some(resolved)
        }
        _ => options.cloned(),
    }
}

/// Whether the options already carry cloudflare gateway auth, upstream's
/// `hasResolvedCloudflareAuth`: an explicit api key or a
/// `cf-aig-authorization` header.
fn has_resolved_cloudflare_auth<T: AuthCarrier>(options: Option<&T>) -> bool {
    has_explicit_api_key(options)
        || options
            .and_then(|options| options.headers_slot())
            .and_then(|headers| headers.get("cf-aig-authorization"))
            .is_some_and(Option::is_some)
}

/// The deprecated api-dispatch stream, upstream's `stream`.
///
/// Builtin-provider models route through their builtin catalog provider,
/// with cloudflare gateways lacking resolved auth detouring through the
/// Models runtime for auth resolution; every other model dispatches through
/// the registry. Both paths merge the env-resolved api key into options
/// that lack one.
#[must_use]
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
) -> AssistantMessageEventStream {
    ensure_builtins_registered();
    if let Some(provider) = get_builtin_provider_for_model(model) {
        if model.provider.0.starts_with("cloudflare-") && !has_resolved_cloudflare_auth(options) {
            let wrapped = options.map(|options| WithTransforms {
                options: options.clone(),
                transform_headers: None,
            });
            return compat_models().stream(model, context, wrapped.as_ref());
        }
        let options = with_env_api_key(model, options);
        return provider.stream(model, context, options.as_ref());
    }
    get_api_provider(&model.api).map_or_else(
        || {
            dispatch_failure(
                model,
                format!("No API provider registered for api: {}", model.api),
            )
        },
        |streams| {
            let options = with_env_api_key(model, options);
            streams.stream(model, context, options.as_ref())
        },
    )
}

/// The deprecated api-dispatch completion, upstream's `complete`: the
/// stream's final assistant message, including error-terminal messages.
#[must_use]
pub fn complete<'a>(
    model: &'a Model,
    context: &'a Context,
    options: Option<&'a StreamOptions>,
) -> BoxedFuture<'a, AssistantMessage> {
    let stream = stream(model, context, options);
    Box::pin(async move { stream.result().await })
}

/// The deprecated simple api-dispatch stream, upstream's `streamSimple`.
#[must_use]
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    ensure_builtins_registered();
    if let Some(provider) = get_builtin_provider_for_model(model) {
        if model.provider.0.starts_with("cloudflare-") && !has_resolved_cloudflare_auth(options) {
            let wrapped = options.map(|options| WithTransforms {
                options: options.clone(),
                transform_headers: None,
            });
            return compat_models().stream_simple(model, context, wrapped.as_ref());
        }
        let options = with_env_api_key(model, options);
        return provider.stream_simple(model, context, options.as_ref());
    }
    get_api_provider(&model.api).map_or_else(
        || {
            dispatch_failure(
                model,
                format!("No API provider registered for api: {}", model.api),
            )
        },
        |streams| {
            let options = with_env_api_key(model, options);
            streams.stream_simple(model, context, options.as_ref())
        },
    )
}

/// The deprecated simple api-dispatch completion, upstream's
/// `completeSimple`: the stream's final assistant message.
#[must_use]
pub fn complete_simple<'a>(
    model: &'a Model,
    context: &'a Context,
    options: Option<&'a SimpleStreamOptions>,
) -> BoxedFuture<'a, AssistantMessage> {
    let stream = stream_simple(model, context, options);
    Box::pin(async move { stream.result().await })
}

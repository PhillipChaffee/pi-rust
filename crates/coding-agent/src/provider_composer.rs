//! Composition of built-in, `models.json`, and extension provider layers,
//! upstream's `src/core/provider-composer.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: the composed provider is a [`Provider`] impl over
//! an optional built-in base, one `models.json` snapshot, and one
//! extension registration; the closure-captured refresh state
//! (`refreshedExtensionModels`, `extensionOAuthCredential`) rides a shared
//! state cell the publications mutate. The compat merge runs through a JSON
//! round-trip because upstream's spread (`{ ...base, ...override }`) is a
//! structural merge the typed structs cannot spell; the four nested
//! routing/kwarg objects merge one level deeper the way upstream's
//! `mergeCompat` special-cases them. The extension OAuth callback
//! vocabulary (`OAuthLoginCallbacks` and friends) lives here — pi-ai's
//! canonical flows speak [`AuthInteraction`], and only the extension
//! contract carries the legacy shape; `adaptOAuth` bridges the two.
//!
//! The legacy `OAuthPrompt.allowEmpty` flag has no slot in pi-ai's prompt
//! vocabulary and is dropped by the bridge, like upstream's spread into a
//! prompt whose type does not carry it.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pi_ai::auth::types::{
    ApiKeyAuth, ApiKeyAuthInput, AuthError, AuthEvent, AuthPrompt, AuthPromptKind, Credential,
    ModelAuth, OAuthAuth, OAuthCredentials, OAuthLoginFn, OAuthRefreshFn, OAuthToAuthFn,
    ProviderAuth,
};
use pi_ai::compat::get_api_provider;
use pi_ai::models::{
    CatalogPersist, ModelsPublication, Provider, ProviderError, ProviderModelError,
    RefreshModelsContext,
};
use pi_ai::types::{
    Api, BoxedFuture, Context, Model, ModelCompat, ProviderEnv, ProviderHeaders,
    SimpleStreamOptions,
};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use tokio_util::sync::CancellationToken;

use crate::model_config::{
    ModelConfig, ModelsJsonModel, ModelsJsonModelOverride, ModelsJsonProvider,
};
use crate::resolve_config_value::{
    clear_config_value_cache, get_config_value_env_var_names, is_command_config_value,
    is_config_value_configured, resolve_config_value_or_throw, resolve_headers_or_throw,
};
use crate::utils::abort::AbortError;

/// The display sources [`AuthStatus`] reports, upstream's `AuthStatus.source`
/// string union.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthStatusSource {
    /// A stored credential in auth.json.
    Stored,
    /// A runtime API key.
    Runtime,
    /// The process environment.
    Environment,
    /// An extension-provided key.
    Fallback,
    /// A literal `apiKey` in models.json.
    ModelsJsonKey,
    /// A `!command` apiKey in models.json.
    ModelsJsonCommand,
}

impl std::fmt::Display for AuthStatusSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Stored => "stored",
            Self::Runtime => "runtime",
            Self::Environment => "environment",
            Self::Fallback => "fallback",
            Self::ModelsJsonKey => "models_json_key",
            Self::ModelsJsonCommand => "models_json_command",
        })
    }
}

/// The configured-auth status a provider reports, upstream's `AuthStatus`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthStatus {
    /// Whether auth is configured.
    pub configured: bool,
    /// Where the configuration came from.
    pub source: Option<AuthStatusSource>,
    /// The env-var labels an environment source resolves through.
    pub label: Option<String>,
}

/// Clear the `!cmd` result cache, upstream's `clearApiKeyCache` alias.
pub fn clear_api_key_cache() {
    clear_config_value_cache();
}

/// A legacy extension OAuth prompt, upstream's `OAuthPrompt`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OAuthPrompt {
    /// The prompt message.
    pub message: String,
    /// Optional placeholder.
    pub placeholder: Option<String>,
    /// Whether empty input is accepted; retained for extension source
    /// compatibility, ignored by the bridge.
    pub allow_empty: Option<bool>,
}

/// A legacy extension OAuth authorization link, upstream's `OAuthAuthInfo`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OAuthAuthInfo {
    /// The URL to visit.
    pub url: String,
    /// Optional instructions shown with the URL.
    pub instructions: Option<String>,
}

/// A legacy extension OAuth device-code notification, upstream's
/// `OAuthDeviceCodeInfo`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OAuthDeviceCodeInfo {
    /// The code the user enters.
    pub user_code: String,
    /// The verification URL.
    pub verification_uri: String,
    /// Poll interval in seconds.
    pub interval_seconds: Option<u64>,
    /// Seconds until the device code expires.
    pub expires_in_seconds: Option<u64>,
}

/// A legacy extension OAuth select option, upstream's `OAuthSelectOption`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OAuthSelectOption {
    /// The value selection returns.
    pub id: String,
    /// The displayed label.
    pub label: String,
}

/// A legacy extension OAuth select prompt, upstream's `OAuthSelectPrompt`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OAuthSelectPrompt {
    /// The prompt message.
    pub message: String,
    /// The selectable options.
    pub options: Vec<OAuthSelectOption>,
}

/// The callback surface an extension OAuth login runs in, upstream's
/// `OAuthLoginCallbacks`.
///
/// `on_auth`/`on_device_code`/`on_progress` are the notify channel;
/// `on_prompt`/`on_manual_code_input`/`on_select` the prompt channel.
#[derive(Clone)]
pub struct OAuthLoginCallbacks {
    /// Cancellation for the whole login flow.
    pub signal: CancellationToken,
    /// The notify channel.
    pub notify: Arc<dyn Fn(AuthEvent) + Send + Sync>,
    /// The text/secret prompt channel.
    pub prompt: OAuthPromptFn,
    /// The manual-code prompt channel.
    pub manual_code_input: OAuthManualCodeFn,
    /// The select prompt channel.
    pub select: OAuthSelectFn,
}

impl std::fmt::Debug for OAuthLoginCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthLoginCallbacks")
            .field("signal", &self.signal)
            .finish_non_exhaustive()
    }
}

/// The notify callback type, pi-ai's `NotifyFn` vocabulary.
pub type OAuthNotifyFn = Arc<dyn Fn(AuthEvent) + Send + Sync>;
/// The text prompt callback, upstream's `onPrompt`.
pub type OAuthPromptFn =
    Arc<dyn Fn(OAuthPrompt) -> BoxedFuture<'static, Result<String, AbortError>> + Send + Sync>;
/// The manual-code callback, upstream's `onManualCodeInput`.
pub type OAuthManualCodeFn =
    Arc<dyn Fn() -> BoxedFuture<'static, Result<String, AbortError>> + Send + Sync>;
/// The select callback, upstream's `onSelect`.
pub type OAuthSelectFn = Arc<
    dyn Fn(OAuthSelectPrompt) -> BoxedFuture<'static, Result<Option<String>, AbortError>>
        + Send
        + Sync,
>;

/// A legacy extension OAuth login, upstream's `ExtensionOAuthConfig`.
#[derive(Clone)]
pub struct ExtensionOAuthConfig {
    /// The auth-method display name.
    pub name: String,
    /// Whether access through this auth method is backed by a provider
    /// subscription.
    pub is_subscription: Option<bool>,
    /// Retained for extension source compatibility; ignored by canonical
    /// auth flows.
    pub uses_callback_server: Option<bool>,
    /// Run the interactive login flow.
    pub login: ExtensionOAuthLoginFn,
    /// Exchange the refresh token.
    pub refresh_token: ExtensionOAuthRefreshFn,
    /// Derive the request API key from a credential.
    pub get_api_key: Arc<dyn Fn(&OAuthCredentials) -> String + Send + Sync>,
    /// Project the credential's model catalog over the composed list.
    pub modify_models: Option<ExtensionModifyModelsFn>,
}

/// The extension OAuth login closure, upstream's `login`.
pub type ExtensionOAuthLoginFn = Arc<
    dyn Fn(OAuthLoginCallbacks) -> BoxedFuture<'static, Result<OAuthCredentials, ProviderError>>
        + Send
        + Sync,
>;
/// The extension OAuth refresh closure, upstream's `refreshToken`.
pub type ExtensionOAuthRefreshFn = Arc<
    dyn Fn(
            OAuthCredentials,
            CancellationToken,
        ) -> BoxedFuture<'static, Result<OAuthCredentials, ProviderError>>
        + Send
        + Sync,
>;
/// The extension OAuth model projection, upstream's `modifyModels`.
pub type ExtensionModifyModelsFn =
    Arc<dyn Fn(Vec<Model>, &OAuthCredentials) -> Vec<Model> + Send + Sync>;

impl std::fmt::Debug for ExtensionOAuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionOAuthConfig")
            .field("name", &self.name)
            .field("is_subscription", &self.is_subscription)
            .field("uses_callback_server", &self.uses_callback_server)
            .field("modify_models", &self.modify_models.is_some())
            .finish_non_exhaustive()
    }
}

/// One model definition an extension registration carries, upstream's
/// inline `ProviderConfigInput["models"]` member.
#[derive(Clone, Debug)]
pub struct ProviderModelInput {
    /// The model id.
    pub id: String,
    /// The display name.
    pub name: String,
    /// The wire API; defaults to the registration's, then a built-in
    /// sibling's.
    pub api: Option<Api>,
    /// The API base URL; defaults to the registration's, then a built-in
    /// sibling's.
    pub base_url: Option<String>,
    /// Whether the model supports reasoning.
    pub reasoning: bool,
    /// Maps pi thinking levels to provider/model-specific values.
    pub thinking_level_map: Option<pi_ai::types::ThinkingLevelMap>,
    /// The input modalities.
    pub input: Vec<pi_ai::types::Modality>,
    /// The pricing.
    pub cost: pi_ai::types::ModelCost,
    /// The context window in tokens.
    pub context_window: u64,
    /// The maximum output tokens.
    pub max_tokens: u64,
    /// Default sampling parameters.
    pub sampling_params: Option<BTreeMap<String, serde_json::Value>>,
    /// Custom HTTP headers.
    pub headers: Option<BTreeMap<String, String>>,
    /// Compatibility overrides for OpenAI-compatible APIs.
    pub compat: Option<ModelCompat>,
}

/// The extension stream closure, upstream's `streamSimple`.
pub type ExtensionStreamSimpleFn = Arc<
    dyn Fn(&Model, &Context, Option<&SimpleStreamOptions>) -> AssistantMessageEventStream
        + Send
        + Sync,
>;
/// The extension refresh closure, upstream's `refreshModels`.
pub type ExtensionRefreshModelsFn = Arc<
    dyn Fn(&RefreshModelsContext) -> BoxedFuture<'static, Result<Vec<Model>, ProviderError>>
        + Send
        + Sync,
>;

/// Input type for the extension registerProvider API, upstream's
/// `ProviderConfigInput`.
#[derive(Clone, Default)]
pub struct ProviderConfigInput {
    /// The display name.
    pub name: Option<String>,
    /// The API base URL.
    pub base_url: Option<String>,
    /// A configured API key, a `$VAR`/`${VAR}` template, or a `!command`.
    pub api_key: Option<String>,
    /// The wire API required by `streamSimple` and applied to models
    /// without their own.
    pub api: Option<Api>,
    /// Stream override for the registration's api.
    pub stream_simple: Option<ExtensionStreamSimpleFn>,
    /// Headers merged into every request.
    pub headers: Option<BTreeMap<String, String>>,
    /// Send the resolved API key as `Authorization: Bearer`.
    pub auth_header: Option<bool>,
    /// The OAuth auth method.
    pub oauth: Option<ExtensionOAuthConfig>,
    /// Custom model definitions replacing the composed list.
    pub models: Option<Vec<ProviderModelInput>>,
    /// Fetch a dynamic model overlay.
    pub refresh_models: Option<ExtensionRefreshModelsFn>,
}

impl std::fmt::Debug for ProviderConfigInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConfigInput")
            .field("name", &self.name)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "set"))
            .field("api", &self.api)
            .field("stream_simple", &self.stream_simple.as_ref().map(|_| "set"))
            .field("auth_header", &self.auth_header)
            .field("headers", &self.headers)
            .field("oauth", &self.oauth)
            .field("models", &self.models.as_ref().map(Vec::len))
            .field(
                "refresh_models",
                &self.refresh_models.as_ref().map(|_| "set"),
            )
            .finish()
    }
}

/// The compat merge, upstream's `mergeCompat`: the override's defined fields
/// win, and the four nested routing/kwarg objects merge one level deeper
/// when both sides carry them.
fn merge_compat(
    base: Option<&ModelCompat>,
    override_compat: Option<&ModelCompat>,
) -> Result<Option<ModelCompat>, String> {
    let Some(override_compat) = override_compat else {
        return Ok(base.cloned());
    };
    let mut merged = serde_json::to_value(base.cloned().unwrap_or_default())
        .map_err(|error| error.to_string())?;
    let over = serde_json::to_value(override_compat).map_err(|error| error.to_string())?;
    let (Some(merged), Some(over)) = (merged.as_object_mut(), over.as_object()) else {
        return Ok(Some(override_compat.clone()));
    };
    for (key, value) in over {
        merged.insert(key.clone(), value.clone());
    }
    // The nested routing and kwarg objects spread one level deeper when both
    // sides define them, upstream's mergeCompat special cases.
    for key in [
        "openRouterRouting",
        "vercelGatewayRouting",
        "chatTemplateKwargs",
        "chatTemplateArgs",
    ] {
        let base_nested = serde_json::to_value(base)
            .ok()
            .and_then(|base| base.get(key).cloned());
        let (Some(base_nested), Some(over_nested)) = (base_nested, merged.get(key).cloned()) else {
            continue;
        };
        let (Some(base_map), Some(over_map)) = (
            base_nested.as_object().cloned(),
            over_nested.as_object().cloned(),
        ) else {
            continue;
        };
        let mut combined = base_map;
        for (nested_key, nested_value) in over_map {
            combined.insert(nested_key, nested_value);
        }
        merged.insert(key.to_owned(), serde_json::Value::Object(combined));
    }
    serde_json::from_value(serde_json::Value::Object(std::mem::take(merged)))
        .map(Some)
        .map_err(|error| error.to_string())
}

/// Apply one per-model override, upstream's `applyModelOverride`.
fn apply_model_override(
    model: Model,
    override_model: &ModelsJsonModelOverride,
) -> Result<Model, String> {
    let mut model = model;
    if let Some(name) = &override_model.name {
        model.name.clone_from(name);
    }
    if let Some(reasoning) = override_model.reasoning {
        model.reasoning = reasoning;
    }
    if let Some(map) = &override_model.thinking_level_map {
        let mut merged = model.thinking_level_map.clone().unwrap_or_default();
        for (level, value) in map {
            merged.insert(*level, value.clone());
        }
        model.thinking_level_map = Some(merged);
    }
    if let Some(input) = &override_model.input {
        model.input.clone_from(input);
    }
    if let Some(cost) = &override_model.cost {
        let rates = pi_ai::types::ModelCostRates {
            input: cost.input.unwrap_or(model.cost.rates.input),
            output: cost.output.unwrap_or(model.cost.rates.output),
            cache_read: cost.cache_read.unwrap_or(model.cost.rates.cache_read),
            cache_write: cost.cache_write.unwrap_or(model.cost.rates.cache_write),
        };
        model.cost = pi_ai::types::ModelCost {
            rates,
            tiers: cost.tiers.clone().or_else(|| model.cost.tiers.clone()),
        };
    }
    if let Some(context_window) = override_model.context_window {
        model.context_window = context_window;
    }
    if let Some(max_tokens) = override_model.max_tokens {
        model.max_tokens = max_tokens;
    }
    if let Some(sampling_params) = &override_model.sampling_params {
        let mut merged = model.sampling_params.clone().unwrap_or_default();
        for (key, value) in sampling_params {
            merged.insert(key.clone(), value.clone());
        }
        model.sampling_params = Some(merged);
    }
    if let Some(compat) = &override_model.compat {
        model.compat = merge_compat(model.compat.as_ref(), Some(compat))?;
    }
    Ok(model)
}

/// Build one model from a models.json definition, upstream's `modelFromJson`.
fn model_from_json(
    provider_id: &str,
    definition: &ModelsJsonModel,
    provider_config: &ModelsJsonProvider,
    defaults: Option<&Model>,
) -> Result<Model, String> {
    let api = Api(
        definition
            .api
            .clone()
            .or_else(|| provider_config.api.clone())
            .or_else(|| defaults.map(|defaults| defaults.api.0.clone()))
            .ok_or_else(|| {
                format!(
                    "Provider {provider_id}, model {}: no \"api\" specified. Set at provider or model level.",
                    definition.id
                )
            })?,
    );
    let base_url = definition
        .base_url
        .as_ref()
        .or(provider_config.base_url.as_ref())
        .or_else(|| defaults.map(|defaults| &defaults.base_url))
        .cloned()
        .ok_or_else(|| {
            format!("Provider {provider_id}: \"baseUrl\" is required when defining custom models.")
        })?;
    if definition
        .context_window
        .is_some_and(|context_window| context_window == 0)
    {
        return Err(format!(
            "Provider {provider_id}, model {}: invalid contextWindow",
            definition.id
        ));
    }
    if definition
        .max_tokens
        .is_some_and(|max_tokens| max_tokens == 0)
    {
        return Err(format!(
            "Provider {provider_id}, model {}: invalid maxTokens",
            definition.id
        ));
    }
    Ok(Model {
        id: definition.id.clone(),
        name: definition
            .name
            .clone()
            .unwrap_or_else(|| definition.id.clone()),
        api,
        provider: pi_ai::types::ProviderId(provider_id.to_owned()),
        base_url,
        reasoning: definition.reasoning.unwrap_or(false),
        thinking_level_map: definition.thinking_level_map.clone(),
        input: definition
            .input
            .clone()
            .unwrap_or_else(|| vec![pi_ai::types::Modality::Text]),
        cost: definition.cost.clone().unwrap_or_default(),
        context_window: definition.context_window.unwrap_or(128_000),
        max_tokens: definition.max_tokens.unwrap_or(16_384),
        sampling_params: definition.sampling_params.clone(),
        headers: None,
        compat: merge_compat(provider_config.compat.as_ref(), definition.compat.as_ref())?,
    })
}

/// The defaults an implicit- fields definition inherits, upstream's
/// `findModelDefaults`: an exact id, else the same api, else the first
/// openai-completions model, else the first model.
fn find_model_defaults<'a>(
    models: &'a [Model],
    model_id: &str,
    api: Option<&str>,
) -> Option<&'a Model> {
    models
        .iter()
        .find(|model| model.id == model_id)
        .or_else(|| api.and_then(|api| models.iter().find(|model| model.api.0 == api)))
        .or_else(|| {
            models
                .iter()
                .find(|model| model.api.0 == "openai-completions")
        })
        .or_else(|| models.first())
}

/// Apply the models.json layer, upstream's `applyModelsJson`.
fn apply_models_json(
    provider_id: &str,
    base_models: &[Model],
    config: Option<&ModelsJsonProvider>,
) -> Result<Vec<Model>, String> {
    let Some(config) = config else {
        return Ok(base_models.to_vec());
    };
    if config.oauth.is_some() && config.base_url.is_none() {
        return Err(format!(
            "Provider {provider_id}: \"baseUrl\" is required when \"oauth\" is set."
        ));
    }
    let has_overrides = config
        .model_overrides
        .as_ref()
        .is_some_and(|overrides| !overrides.is_empty());
    if config.models.as_ref().is_none_or(Vec::is_empty)
        && config.base_url.is_none()
        && config.headers.is_none()
        && config.compat.is_none()
        && !has_overrides
        && config.api_key.is_none()
        && config.oauth.is_none()
        && config.auth_header.is_none()
    {
        return Err(format!(
            "Provider {provider_id}: must specify \"baseUrl\", \"headers\", \"compat\", \"modelOverrides\", or \"models\"."
        ));
    }

    let mut models = base_models
        .iter()
        .map(|model| {
            let mut model = model.clone();
            if config.oauth.as_deref() != Some("radius")
                && let Some(base_url) = &config.base_url
            {
                model.base_url.clone_from(base_url);
            }
            model.compat = merge_compat(model.compat.as_ref(), config.compat.as_ref())?;
            Ok(model)
        })
        .collect::<Result<Vec<_>, String>>()?;
    for definition in config.models.as_ref().unwrap_or(&Vec::new()) {
        let existing = models.iter().position(|model| model.id == definition.id);
        let defaults = find_model_defaults(
            &models,
            &definition.id,
            definition.api.as_deref().or(config.api.as_deref()),
        );
        let model = model_from_json(provider_id, definition, config, defaults)?;
        match existing {
            Some(index) => models[index] = model,
            None => models.push(model),
        }
    }
    Ok(models)
}

/// Apply the extension layer, upstream's `applyExtension`.
fn apply_extension(
    provider_id: &str,
    models: &[Model],
    config: Option<&ProviderConfigInput>,
) -> Result<Vec<Model>, String> {
    let Some(config) = config else {
        return Ok(models.to_vec());
    };
    let Some(definitions) = &config.models else {
        return config.base_url.as_ref().map_or_else(
            || Ok(models.to_vec()),
            |base_url| {
                Ok(models
                    .iter()
                    .map(|model| {
                        let mut model = model.clone();
                        model.base_url.clone_from(base_url);
                        model
                    })
                    .collect())
            },
        );
    };
    definitions
        .iter()
        .map(|definition| {
            let defaults = find_model_defaults(
                models,
                &definition.id,
                definition.api.as_deref().or(config.api.as_deref()),
            );
            let api = definition
                .api
                .clone()
                .or_else(|| config.api.clone())
                .or_else(|| defaults.map(|defaults| defaults.api.clone()))
                .ok_or_else(|| {
                            format!(
                            "Provider {provider_id}, model {}: no \"api\" specified. Set at provider or model level.",
                            definition.id
                        )
                    })?;
            let base_url = definition
                .base_url
                .as_ref()
                .or(config.base_url.as_ref())
                .or_else(|| defaults.map(|defaults| &defaults.base_url))
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "Provider {provider_id}: \"baseUrl\" is required when defining custom models."
                    )
                })?;
            Ok(Model {
                id: definition.id.clone(),
                name: definition.name.clone(),
                api,
                provider: pi_ai::types::ProviderId(provider_id.to_owned()),
                base_url,
                reasoning: definition.reasoning,
                thinking_level_map: definition.thinking_level_map.clone(),
                input: definition.input.clone(),
                cost: definition.cost.clone(),
                context_window: definition.context_window,
                max_tokens: definition.max_tokens,
                sampling_params: definition.sampling_params.clone(),
                headers: None,
                compat: definition.compat.clone(),
            })
        })
        .collect()
}

/// Bridge a legacy extension OAuth config onto pi-ai's [`OAuthAuth`],
/// upstream's `adaptOAuth`.
fn adapt_oauth(config: &ExtensionOAuthConfig) -> OAuthAuth {
    let login_name = config.name.clone();
    let login: OAuthLoginFn = Arc::new({
        let config = config.clone();
        move |interaction| {
            let config = config.clone();
            let signal = interaction.signal.clone();
            let prompt = interaction.prompt.clone();
            let notify = interaction.notify.clone();
            Box::pin(async move {
                let callbacks = OAuthLoginCallbacks {
                    signal: signal.clone(),
                    notify: Arc::new(move |event| notify(event)),
                    prompt: Arc::new({
                        let prompt = Arc::clone(&prompt);
                        move |oauth_prompt| {
                            let prompt = Arc::clone(&prompt);
                            Box::pin(async move {
                                prompt(AuthPrompt {
                                    signal: None,
                                    kind: AuthPromptKind::Text {
                                        message: oauth_prompt.message,
                                        placeholder: oauth_prompt.placeholder,
                                    },
                                })
                                .await
                            })
                        }
                    }),
                    manual_code_input: Arc::new({
                        let prompt = Arc::clone(&prompt);
                        move || {
                            let prompt = Arc::clone(&prompt);
                            Box::pin(async move {
                                prompt(AuthPrompt {
                                    signal: None,
                                    kind: AuthPromptKind::ManualCode {
                                        message: "Paste the authorization code".to_owned(),
                                        placeholder: None,
                                    },
                                })
                                .await
                            })
                        }
                    }),
                    select: Arc::new({
                        let prompt = Arc::clone(&prompt);
                        move |select_prompt| {
                            let prompt = Arc::clone(&prompt);
                            Box::pin(async move {
                                prompt(AuthPrompt {
                                    signal: None,
                                    kind: AuthPromptKind::Select {
                                        message: select_prompt.message,
                                        options: select_prompt
                                            .options
                                            .into_iter()
                                            .map(|option| pi_ai::auth::types::AuthPromptOption {
                                                id: option.id,
                                                label: option.label,
                                                description: None,
                                            })
                                            .collect(),
                                    },
                                })
                                .await
                                .map(Some)
                            })
                        }
                    }),
                };
                let credential = (config.login)(callbacks).await?;
                Ok(credential)
            })
        }
    });
    let refresh: OAuthRefreshFn = Arc::new({
        let refresh_token = Arc::clone(&config.refresh_token);
        move |credentials, signal| {
            let refresh_token = Arc::clone(&refresh_token);
            Box::pin(async move { (refresh_token)(credentials, signal).await })
        }
    });
    let to_auth: OAuthToAuthFn = Arc::new({
        let get_api_key = Arc::clone(&config.get_api_key);
        move |credentials| {
            let get_api_key = Arc::clone(&get_api_key);
            Box::pin(async move {
                Ok(ModelAuth {
                    api_key: Some(get_api_key(&credentials)),
                    ..ModelAuth::default()
                })
            })
        }
    });
    OAuthAuth {
        name: login_name,
        is_subscription: config.is_subscription,
        login_label: None,
        login,
        refresh,
        to_auth,
    }
}

/// Merge configured headers into request auth and enforce the
/// `Authorization: Bearer` form, upstream's `withConfiguredAuth`.
fn with_configured_auth(
    auth: ModelAuth,
    headers: Option<&BTreeMap<String, String>>,
    auth_header: bool,
) -> Result<ModelAuth, String> {
    let mut merged_headers: Option<ProviderHeaders> = if auth.headers.is_some() || headers.is_some()
    {
        let mut merged = auth.headers.clone().unwrap_or_default();
        for (name, value) in headers.into_iter().flatten() {
            merged.insert(name.clone(), Some(value.clone()));
        }
        Some(merged)
    } else {
        None
    };
    if auth_header {
        let Some(api_key) = &auth.api_key else {
            return Err("authHeader requires a resolved API key".to_owned());
        };
        merged_headers.get_or_insert_with(BTreeMap::new).insert(
            "Authorization".to_owned(),
            Some(format!("Bearer {api_key}")),
        );
    }
    Ok(ModelAuth {
        headers: merged_headers,
        ..auth
    })
}

/// The raw key the composed layers configure, upstream's `configuredApiKey`.
fn configured_api_key<'a>(
    config: Option<&'a ModelsJsonProvider>,
    extension: Option<&'a ProviderConfigInput>,
) -> Option<&'a str> {
    extension
        .and_then(|extension| extension.api_key.as_deref())
        .or_else(|| config.and_then(|config| config.api_key.as_deref()))
}

/// The raw headers the composed layers configure, upstream's
/// `configuredHeaders`.
fn configured_headers<'a>(
    config: Option<&'a ModelsJsonProvider>,
    extension: Option<&'a ProviderConfigInput>,
) -> Option<BTreeMap<String, String>> {
    if config.and_then(|config| config.headers.as_ref()).is_none()
        && extension
            .and_then(|extension| extension.headers.as_ref())
            .is_none()
    {
        return None;
    }
    let mut merged = config
        .and_then(|config| config.headers.clone())
        .unwrap_or_default();
    for (name, value) in extension
        .and_then(|extension| extension.headers.clone())
        .unwrap_or_default()
    {
        merged.insert(name, value);
    }
    Some(merged)
}

/// Gather the env names the given config values reference plus the explicit
/// overrides, upstream's `configContextEnv`.
fn config_context_env(
    values: &[&str],
    ctx: &Arc<dyn pi_ai::auth::types::AuthContext>,
    explicit: Option<&ProviderEnv>,
) -> Option<ProviderEnv> {
    let mut env = explicit.cloned().unwrap_or_default();
    let names: std::collections::BTreeSet<String> = values
        .iter()
        .flat_map(|value| get_config_value_env_var_names(value))
        .collect();
    for name in names {
        if env.contains_key(&name) {
            continue;
        }
        if let Some(value) = ctx.env(&name) {
            env.insert(name, value);
        }
    }
    (!env.is_empty()).then_some(env)
}

/// Compose the api-key auth method, upstream's `composeApiKeyAuth`.
#[expect(
    clippy::too_many_lines,
    reason = "the 1:1 port of upstream's four-closure auth method reads longer than the lint's slice"
)]
fn compose_api_key_auth(
    provider_id: &str,
    base: Option<&Arc<dyn Provider>>,
    config: Option<&ModelsJsonProvider>,
    extension: Option<&ProviderConfigInput>,
) -> Option<ApiKeyAuth> {
    let inherited = base.and_then(|base| base.auth().api_key.as_ref());
    let raw_key = configured_api_key(config, extension).map(str::to_owned);
    let has_oauth = extension
        .and_then(|extension| extension.oauth.as_ref())
        .is_some()
        || base.and_then(|base| base.auth().oauth.as_ref()).is_some();
    // OAuth-only providers get no fabricated API-key login method.
    if inherited.is_none() && raw_key.is_none() && has_oauth {
        return None;
    }
    let raw_headers = configured_headers(config, extension);
    let auth_header = extension
        .and_then(|extension| extension.auth_header)
        .or_else(|| config.and_then(|config| config.auth_header))
        .unwrap_or(false);

    let name = inherited.map_or_else(|| "API key".to_owned(), |inherited| inherited.name.clone());
    let login = inherited
        .and_then(|inherited| inherited.login.clone())
        .or_else(|| {
            let login_fn: pi_ai::auth::types::ApiKeyLoginFn = Arc::new(
                move |interaction: pi_ai::auth::types::ProviderAuthInteraction| {
                    let prompt = Arc::clone(&interaction.prompt);
                    Box::pin(async move {
                        let key = prompt(AuthPrompt {
                            signal: None,
                            kind: AuthPromptKind::Secret {
                                message: "Enter API key".to_owned(),
                                placeholder: None,
                            },
                        })
                        .await
                        .map_err(|error| -> AuthError { Box::new(error) })?;
                        Ok(pi_ai::auth::types::ApiKeyCredential {
                            key: Some(key),
                            env: None,
                        })
                    })
                },
            );
            Some(login_fn)
        });
    let check: pi_ai::auth::types::ApiKeyCheckFn = Arc::new({
        let inherited = inherited.cloned();
        let raw_key = raw_key.clone();
        move |input| {
            let inherited = inherited.clone();
            let raw_key = raw_key.clone();
            Box::pin(async move {
                if input.credential.is_some() {
                    if let Some(check) = inherited
                        .as_ref()
                        .and_then(|inherited| inherited.check.clone())
                    {
                        return (check)(input).await;
                    }
                    if input
                        .credential
                        .as_ref()
                        .and_then(|credential| credential.key.as_deref())
                        .is_some_and(|key| !key.is_empty())
                    {
                        return Ok(Some(pi_ai::auth::types::AuthCheck {
                            source: Some("stored credential".to_owned()),
                            auth_type: pi_ai::auth::types::AuthType::ApiKey,
                        }));
                    }
                    let resolved = match inherited
                        .as_ref()
                        .map(|inherited| inherited.resolve.clone())
                    {
                        Some(resolve) => (resolve)(input).await?,
                        None => None,
                    };
                    return Ok(resolved.map(|resolved| pi_ai::auth::types::AuthCheck {
                        source: resolved.source,
                        auth_type: pi_ai::auth::types::AuthType::ApiKey,
                    }));
                }
                if let Some(raw_key) = &raw_key {
                    if is_command_config_value(raw_key) {
                        return Ok(Some(pi_ai::auth::types::AuthCheck {
                            source: Some("configured API key".to_owned()),
                            auth_type: pi_ai::auth::types::AuthType::ApiKey,
                        }));
                    }
                    for name in get_config_value_env_var_names(raw_key) {
                        if input.ctx.env(&name).is_none() {
                            return Ok(None);
                        }
                    }
                    return Ok(Some(pi_ai::auth::types::AuthCheck {
                        source: Some("configured API key".to_owned()),
                        auth_type: pi_ai::auth::types::AuthType::ApiKey,
                    }));
                }
                if let Some(check) = inherited
                    .as_ref()
                    .and_then(|inherited| inherited.check.clone())
                {
                    return (check)(input).await;
                }
                let resolved = match inherited
                    .as_ref()
                    .map(|inherited| inherited.resolve.clone())
                {
                    Some(resolve) => (resolve)(input).await?,
                    None => None,
                };
                Ok(resolved.map(|resolved| pi_ai::auth::types::AuthCheck {
                    source: resolved.source,
                    auth_type: pi_ai::auth::types::AuthType::ApiKey,
                }))
            })
        }
    });
    let resolve: pi_ai::auth::types::ApiKeyResolveFn = Arc::new({
        let inherited = inherited.cloned();
        let provider_id = provider_id.to_owned();
        move |input| {
            let inherited = inherited.clone();
            let raw_key = raw_key.clone();
            let raw_headers = raw_headers.clone();
            let provider_id = provider_id.clone();
            Box::pin(async move {
                let result: Option<pi_ai::auth::types::AuthResult> = if input.credential.is_some() {
                    if let Some(resolve) = inherited
                        .as_ref()
                        .map(|inherited| inherited.resolve.clone())
                    {
                        (resolve)(input.clone()).await?
                    } else {
                        let key = input
                            .credential
                            .as_ref()
                            .and_then(|credential| credential.key.as_deref());
                        key.and_then(|key| {
                            (!key.is_empty()).then(|| pi_ai::auth::types::AuthResult {
                                auth: ModelAuth {
                                    api_key: Some(key.to_owned()),
                                    ..ModelAuth::default()
                                },
                                env: input
                                    .credential
                                    .as_ref()
                                    .and_then(|credential| credential.env.clone()),
                                source: Some("stored credential".to_owned()),
                            })
                        })
                    }
                } else if let Some(raw_key) = &raw_key {
                    let env = config_context_env(&[raw_key.as_str()], &input.ctx, None);
                    let key = resolve_config_value_or_throw(
                        raw_key,
                        &format!("API key for provider \"{provider_id}\""),
                        env.as_ref(),
                    )
                    .map_err(|error| -> AuthError { Box::new(std::io::Error::other(error)) })?;
                    match inherited
                        .as_ref()
                        .map(|inherited| inherited.resolve.clone())
                    {
                        Some(resolve) => {
                            let with_credential = ApiKeyAuthInput {
                                ctx: Arc::clone(&input.ctx),
                                credential: Some(pi_ai::auth::types::ApiKeyCredential {
                                    key: Some(key),
                                    env: None,
                                }),
                                signal: input.signal.clone(),
                            };
                            (resolve)(with_credential).await?
                        }
                        None => Some(pi_ai::auth::types::AuthResult {
                            auth: ModelAuth {
                                api_key: Some(key),
                                ..ModelAuth::default()
                            },
                            env: None,
                            source: Some("configured API key".to_owned()),
                        }),
                    }
                } else {
                    match inherited
                        .as_ref()
                        .map(|inherited| inherited.resolve.clone())
                    {
                        Some(resolve) => (resolve)(input.clone()).await?,
                        None => None,
                    }
                };
                let Some(result) = result else {
                    return Ok(None);
                };
                let explicit_env: ProviderEnv = input
                    .credential
                    .as_ref()
                    .and_then(|credential| credential.env.clone())
                    .unwrap_or_default()
                    .into_iter()
                    .chain(result.env.clone().unwrap_or_default())
                    .collect();
                let header_env = config_context_env(
                    &raw_headers
                        .as_ref()
                        .map(|headers| headers.values().map(String::as_str).collect::<Vec<_>>())
                        .unwrap_or_default(),
                    &input.ctx,
                    Some(&explicit_env),
                );
                let headers = resolve_headers_or_throw(
                    raw_headers.as_ref(),
                    &format!("provider \"{provider_id}\""),
                    header_env.as_ref(),
                )
                .map_err(|error| -> AuthError { Box::new(std::io::Error::other(error)) })?;
                Ok(Some(pi_ai::auth::types::AuthResult {
                    auth: with_configured_auth(result.auth, headers.as_ref(), auth_header)?,
                    env: result.env,
                    source: result.source,
                }))
            })
        }
    });
    Some(ApiKeyAuth {
        name,
        login,
        check: Some(check),
        resolve,
    })
}

/// Compose the OAuth auth method, upstream's `composeOAuthAuth`.
fn compose_oauth_auth(
    provider_id: &str,
    base: Option<&Arc<dyn Provider>>,
    config: Option<&ModelsJsonProvider>,
    extension: Option<&ProviderConfigInput>,
) -> Option<OAuthAuth> {
    let oauth = extension
        .and_then(|extension| extension.oauth.as_ref())
        .map_or_else(
            || base.and_then(|base| base.auth().oauth.clone()),
            |extension_oauth| Some(adapt_oauth(extension_oauth)),
        );
    let oauth = oauth?;
    let raw_headers = configured_headers(config, extension);
    let auth_header = extension
        .and_then(|extension| extension.auth_header)
        .or_else(|| config.and_then(|config| config.auth_header))
        .unwrap_or(false);
    let to_auth: OAuthToAuthFn = Arc::new({
        let inner = oauth.to_auth.clone();
        let provider_id = provider_id.to_owned();
        move |credentials| {
            let inner = Arc::clone(&inner);
            let raw_headers = raw_headers.clone();
            let provider_id = provider_id.clone();
            Box::pin(async move {
                let auth = (inner)(credentials.clone()).await?;
                // The env record rides the credential's flattened extras,
                // upstream's index-signature field.
                let env = credentials
                    .extra
                    .get("env")
                    .and_then(|env| serde_json::from_value::<ProviderEnv>(env.clone()).ok());
                let headers = resolve_headers_or_throw(
                    raw_headers.as_ref(),
                    &format!("provider \"{provider_id}\""),
                    env.as_ref(),
                )
                .map_err(|error| -> AuthError { Box::new(std::io::Error::other(error)) })?;
                Ok(with_configured_auth(auth, headers.as_ref(), auth_header)?)
            })
        }
    });
    Some(OAuthAuth { to_auth, ..oauth })
}

/// The per-model headers the composed layers configure, upstream's
/// `rawModelHeaders`.
fn raw_model_headers(
    model: &Model,
    config: Option<&ModelsJsonProvider>,
    extension: Option<&ProviderConfigInput>,
) -> Option<BTreeMap<String, String>> {
    let definition = config
        .and_then(|config| config.models.as_ref())
        .and_then(|models| models.iter().find(|entry| entry.id == model.id))
        .and_then(|entry| entry.headers.as_ref());
    let override_headers = config
        .and_then(|config| config.model_overrides.as_ref())
        .and_then(|overrides| overrides.get(&model.id))
        .and_then(|override_model| override_model.headers.as_ref());
    let extension_model = extension
        .and_then(|extension| extension.models.as_ref())
        .and_then(|models| models.iter().find(|entry| entry.id == model.id))
        .and_then(|entry| entry.headers.as_ref());
    let headers: BTreeMap<String, String> = override_headers
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .chain(definition.cloned().unwrap_or_default())
        .chain(extension_model.cloned().unwrap_or_default())
        .collect();
    (!headers.is_empty()).then_some(headers)
}

/// Validate an extension registration on its own, upstream's
/// `validateExtensionProvider`: a broken re-registration must fail without
/// touching the stored config.
///
/// # Errors
/// The streamSimple-without-api error, or the models.json/extension
/// composition errors.
pub fn validate_extension_provider(
    provider_id: &str,
    base: Option<&Arc<dyn Provider>>,
    models_config: Option<&ModelsJsonProvider>,
    extension: &ProviderConfigInput,
) -> Result<(), String> {
    if extension.stream_simple.is_some() && extension.api.is_none() {
        return Err(format!(
            "Provider {provider_id}: \"api\" is required when registering streamSimple."
        ));
    }
    let base_models = base
        .map(|base| base.get_models())
        .transpose()
        .map_err(|error| error.to_string())?
        .unwrap_or_default();
    apply_extension(
        provider_id,
        &apply_models_json(provider_id, &base_models, models_config)?,
        Some(extension),
    )?;
    Ok(())
}

/// The refresh-time state a composed provider keeps, upstream's closure
/// variables `refreshedExtensionModels` and `extensionOAuthCredential`.
#[derive(Default)]
struct ComposedState {
    refreshed_models: Option<Vec<ProviderModelInput>>,
    oauth_credential: Option<OAuthCredentials>,
}

/// The dispatch a model resolves to, the vocabulary
/// [`ComposedProvider::choose_dispatch`] returns, owned so the lazy stream
/// setup can move it.
enum OwnedDispatch {
    /// The extension's streamSimple closure.
    Extension(ExtensionStreamSimpleFn),
    /// The composed base provider.
    Base(Arc<dyn Provider>),
    /// The api registry's streams implementation.
    Api(Arc<dyn pi_ai::types::ProviderStreams>),
}

/// The composed provider, upstream's object literal `composeModelProvider`
/// returns.
struct ComposedProvider {
    id: String,
    name: String,
    base_url: Option<String>,
    headers: Option<ProviderHeaders>,
    auth: ProviderAuth,
    base: Option<Arc<dyn Provider>>,
    config: Option<ModelsJsonProvider>,
    extension: Option<ProviderConfigInput>,
    state: Arc<Mutex<ComposedState>>,
}

impl std::fmt::Debug for ComposedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComposedProvider")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl ComposedProvider {
    /// The extension as the refresh state sees it, upstream's
    /// `currentExtension()`.
    fn current_extension(&self, state: &ComposedState) -> Option<ProviderConfigInput> {
        self.extension.as_ref().map(|extension| {
            state.refreshed_models.as_ref().map_or_else(
                || extension.clone(),
                |refreshed| ProviderConfigInput {
                    models: Some(refreshed.clone()),
                    ..extension.clone()
                },
            )
        })
    }

    /// The composed model list, upstream's `getModels` closure: models.json
    /// upserts, then extension replacement, then legacy OAuth projection,
    /// then the per-model overrides.
    fn composed_models(&self, state: &ComposedState) -> Result<Vec<Model>, String> {
        let base_models = match &self.base {
            Some(base) => base.get_models().map_err(|error| error.to_string())?,
            None => Vec::new(),
        };
        let mut models = apply_extension(
            &self.id,
            &apply_models_json(&self.id, &base_models, self.config.as_ref())?,
            self.current_extension(state).as_ref(),
        )?;
        if let (Some(oauth_credential), Some(extension)) =
            (&state.oauth_credential, &self.extension)
            && let Some(oauth) = &extension.oauth
            && let Some(modify_models) = &oauth.modify_models
        {
            models = modify_models(models, oauth_credential);
        }
        models
            .into_iter()
            .map(|model| {
                let override_model = self
                    .config
                    .as_ref()
                    .and_then(|config| config.model_overrides.as_ref())
                    .and_then(|overrides| overrides.get(&model.id));
                match override_model {
                    Some(override_model) => apply_model_override(model, override_model),
                    None => Ok(model),
                }
            })
            .collect()
    }

    /// Whether the base can stream a model's api, upstream's
    /// `supportsBaseApi`.
    fn supports_base_api(&self, model: &Model) -> bool {
        self.base
            .as_ref()
            .and_then(|base| base.get_models().ok())
            .is_some_and(|models| models.iter().any(|entry| entry.api == model.api))
    }

    /// The dispatch a model resolves to, upstream's `streamWith` body: the
    /// extension's streamSimple for its api, the base for models it can
    /// serve, else the api registry.
    fn choose_dispatch(&self, model: &Model) -> Result<OwnedDispatch, String> {
        if let Some(extension) = &self.extension
            && let Some(stream_simple) = &extension.stream_simple
            && extension.api.as_ref().is_some_and(|api| *api == model.api)
        {
            return Ok(OwnedDispatch::Extension(Arc::clone(stream_simple)));
        }
        if let Some(base) = &self.base
            && self.supports_base_api(model)
        {
            return Ok(OwnedDispatch::Base(Arc::clone(base)));
        }
        get_api_provider(&model.api).map_or_else(
            || Err(format!("No API provider registered for api: {}", model.api)),
            |api| Ok(OwnedDispatch::Api(api)),
        )
    }

    /// Stream through the composed dispatch, upstream's `stream` arm of
    /// `streamWith`: the extension's streamSimple reads the shared base
    /// fields, the base and the api registry take the options as sent.
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&pi_ai::types::StreamOptions>,
    ) -> AssistantMessageEventStream {
        let model = model.clone();
        let context = context.clone();
        let options = options.cloned();
        let dispatch = self.choose_dispatch(&model);
        let lazy_model = model.clone();
        pi_ai::api::lazy::lazy_stream(&lazy_model, move || {
            let model = model.clone();
            let context = context.clone();
            Box::pin(async move {
                let dispatch =
                    dispatch.map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::other(error))
                    })?;
                Ok(match dispatch {
                    OwnedDispatch::Extension(stream_simple) => {
                        let simple_options = options.as_ref().map(SimpleStreamOptions::from_stream);
                        stream_simple(&model, &context, simple_options.as_ref())
                    }
                    OwnedDispatch::Base(base) => base.stream(&model, &context, options.as_ref()),
                    OwnedDispatch::Api(api) => api.stream(&model, &context, options.as_ref()),
                })
            })
        })
    }

    /// Stream a simple request through the composed dispatch, upstream's
    /// `streamSimple` arm of `streamWith`.
    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let model = model.clone();
        let context = context.clone();
        let options = options.cloned();
        let dispatch = self.choose_dispatch(&model);
        let lazy_model = model.clone();
        pi_ai::api::lazy::lazy_stream(&lazy_model, move || {
            let model = model.clone();
            let context = context.clone();
            Box::pin(async move {
                let dispatch =
                    dispatch.map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                        Box::new(std::io::Error::other(error))
                    })?;
                Ok(match dispatch {
                    OwnedDispatch::Extension(stream_simple) => {
                        stream_simple(&model, &context, options.as_ref())
                    }
                    OwnedDispatch::Base(base) => {
                        base.stream_simple(&model, &context, options.as_ref())
                    }
                    OwnedDispatch::Api(api) => {
                        api.stream_simple(&model, &context, options.as_ref())
                    }
                })
            })
        })
    }

    /// The refresh flow, upstream's `refreshModels` body: the base refreshes,
    /// the extension's closure fetches, then the publication projects both
    /// into the shared state.
    async fn refresh(&self, context: RefreshModelsContext) -> Result<(), ProviderError> {
        if let Some(base) = &self.base
            && base.supports_refresh_models()
        {
            base.refresh_models(context.clone()).await?;
        }
        let extension = self.extension.clone();
        let Some(extension) = extension else {
            return Ok(());
        };
        let refreshed: Option<Vec<ProviderModelInput>> =
            if let Some(extension_refresh) = &extension.refresh_models {
                let models = (extension_refresh)(&context).await?;
                if context.signal.is_cancelled() {
                    return Ok(());
                }
                // Validate before publishing the new synchronous list.
                apply_extension(
                    &self.id,
                    &apply_models_json(
                        &self.id,
                        &self
                            .base
                            .as_ref()
                            .and_then(|base| base.get_models().ok())
                            .unwrap_or_default(),
                        self.config.as_ref(),
                    )?,
                    Some(&ProviderConfigInput {
                        models: Some(models.iter().map(model_to_input).collect()),
                        ..extension.clone()
                    }),
                )?;
                Some(models.iter().map(model_to_input).collect())
            } else {
                None
            };
        let oauth_credential = context
            .credential
            .as_ref()
            .and_then(|credential| credential.as_oauth().cloned());
        (context.publish)(ModelsPublication {
            persist: CatalogPersist::Omit,
            update: Some(Box::new({
                let state = Arc::clone(&self.state);
                move || {
                    let mut state = state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if refreshed.is_some() {
                        state.refreshed_models.clone_from(&refreshed);
                    }
                    state.oauth_credential.clone_from(&oauth_credential);
                }
            })),
        })
        .await?;
        Ok(())
    }
}

impl Provider for ComposedProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    fn headers(&self) -> Option<&ProviderHeaders> {
        self.headers.as_ref()
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn get_models(&self) -> Result<Vec<Model>, ProviderModelError> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.composed_models(&state)
            .map_err(|error| -> ProviderModelError { Box::new(std::io::Error::other(error)) })
    }

    fn refresh_models(
        &self,
        context: RefreshModelsContext,
    ) -> BoxedFuture<'_, Result<(), ProviderError>> {
        let has_refresh = self.has_refresh_models();
        if !has_refresh {
            return Box::pin(async { Ok(()) });
        }
        Box::pin(async move { self.refresh(context).await })
    }

    fn supports_refresh_models(&self) -> bool {
        self.has_refresh_models()
    }

    fn filter_models(&self, models: Vec<Model>, credential: Option<&Credential>) -> Vec<Model> {
        match &self.base {
            Some(base) => base.filter_models(models, credential),
            None => models,
        }
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&pi_ai::types::StreamOptions>,
    ) -> AssistantMessageEventStream {
        Self::stream(self, model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        Self::stream_simple(self, model, context, options)
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &pi_ai::types::DeferredHandle,
        options: Option<&pi_ai::types::DeferredFetchOptions>,
    ) -> Option<AssistantMessageEventStream> {
        self.base
            .as_ref()
            .and_then(|base| base.fetch_deferred(model, handle, options))
    }

    fn cancel_deferred<'a>(
        &'a self,
        model: &'a Model,
        handle: &'a pi_ai::types::DeferredHandle,
        options: Option<&'a pi_ai::types::DeferredCancelOptions>,
    ) -> Option<BoxedFuture<'a, Result<(), pi_ai::utils::provider_retry::ProviderRequestError>>>
    {
        self.base
            .as_ref()
            .and_then(|base| base.cancel_deferred(model, handle, options))
    }

    fn supports_fetch_deferred(&self) -> bool {
        self.base
            .as_ref()
            .is_some_and(|base| base.supports_fetch_deferred())
    }

    fn supports_cancel_deferred(&self) -> bool {
        self.base
            .as_ref()
            .is_some_and(|base| base.supports_cancel_deferred())
    }
}

impl ComposedProvider {
    /// Whether any layer carries a refresh phase, upstream's
    /// `base?.refreshModels || extension?.refreshModels ||
    /// extension?.oauth?.modifyModels` presence check.
    fn has_refresh_models(&self) -> bool {
        self.base
            .as_ref()
            .is_some_and(|base| base.supports_refresh_models())
            || self.extension.as_ref().is_some_and(|extension| {
                extension.refresh_models.is_some()
                    || extension
                        .oauth
                        .as_ref()
                        .is_some_and(|oauth| oauth.modify_models.is_some())
            })
    }
}

/// Convert an extension model back into its registration input, the inverse
/// of `applyExtension`'s projection.
fn model_to_input(model: &Model) -> ProviderModelInput {
    ProviderModelInput {
        id: model.id.clone(),
        name: model.name.clone(),
        api: Some(model.api.clone()),
        base_url: Some(model.base_url.clone()),
        reasoning: model.reasoning,
        thinking_level_map: model.thinking_level_map.clone(),
        input: model.input.clone(),
        cost: model.cost.clone(),
        context_window: model.context_window,
        max_tokens: model.max_tokens,
        sampling_params: model.sampling_params.clone(),
        headers: model.headers.clone(),
        compat: model.compat.clone(),
    }
}

/// Compose built-in, models.json, and extension layers without reading
/// credentials, upstream's `composeModelProvider`.
///
/// # Errors
/// The composition failure: a models.json layer that specifies nothing, a
/// custom model without an api or base URL, or no authentication method.
pub fn compose_model_provider(
    provider_id: &str,
    base: Option<Arc<dyn Provider>>,
    model_config: &ModelConfig,
    extension: Option<ProviderConfigInput>,
) -> Result<Arc<dyn Provider>, String> {
    let config = model_config.get_provider(provider_id).cloned();
    let state = Arc::new(Mutex::new(ComposedState::default()));
    // Validate eagerly so registration/reload reports structural errors
    // immediately, upstream's eager `getModels()` call. The placeholder auth
    // exists only so the probe can build; the real auth is composed below.
    let probe = ComposedProvider {
        id: provider_id.to_owned(),
        name: provider_id.to_owned(),
        base_url: None,
        headers: None,
        auth: ProviderAuth::default(),
        base: base.clone(),
        config: config.clone(),
        extension: extension.clone(),
        state: Arc::clone(&state),
    };
    {
        let state_guard = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        probe.composed_models(&state_guard)?;
    }
    let auth = ProviderAuth {
        api_key: compose_api_key_auth(
            provider_id,
            base.as_ref(),
            config.as_ref(),
            extension.as_ref(),
        ),
        oauth: compose_oauth_auth(
            provider_id,
            base.as_ref(),
            config.as_ref(),
            extension.as_ref(),
        ),
    };
    if auth.api_key.is_none() && auth.oauth.is_none() {
        return Err(format!(
            "Provider {provider_id}: no authentication method configured."
        ));
    }
    let name = extension
        .as_ref()
        .and_then(|extension| extension.name.clone())
        .or_else(|| config.as_ref().and_then(|config| config.name.clone()))
        .or_else(|| base.as_ref().map(|base| base.name().to_owned()))
        .or_else(|| {
            extension
                .as_ref()
                .and_then(|extension| extension.oauth.as_ref())
                .map(|oauth| oauth.name.clone())
        })
        .unwrap_or_else(|| provider_id.to_owned());
    let base_url = extension
        .as_ref()
        .and_then(|extension| extension.base_url.clone())
        .or_else(|| config.as_ref().and_then(|config| config.base_url.clone()))
        .or_else(|| {
            base.as_ref()
                .and_then(|base| base.base_url().map(str::to_owned))
        });
    Ok(Arc::new(ComposedProvider {
        id: provider_id.to_owned(),
        name,
        base_url,
        headers: base.as_ref().and_then(|base| base.headers().cloned()),
        auth,
        base,
        config,
        extension,
        state,
    }))
}

/// The request-time headers a model's composed layers configure, upstream's
/// `resolveConfiguredModelHeaders`.
///
/// # Errors
/// A header value that fails to resolve.
pub fn resolve_configured_model_headers(
    model: &Model,
    config: Option<&ModelsJsonProvider>,
    extension: Option<&ProviderConfigInput>,
    env: Option<&ProviderEnv>,
) -> Result<Option<BTreeMap<String, String>>, String> {
    resolve_headers_or_throw(
        raw_model_headers(model, config, extension).as_ref(),
        &format!("model \"{}/{}\"", model.provider.0, model.id),
        env,
    )
}

/// The compatibility fallback request config [`crate::model_registry`] reads
/// when provider auth is unconfigured, upstream's
/// `resolveCompatibilityRequestConfig`.
///
/// # Errors
/// A header value that fails to resolve.
pub fn resolve_compatibility_request_config(
    model: &Model,
    config: Option<&ModelsJsonProvider>,
    extension: Option<&ProviderConfigInput>,
) -> Result<CompatibilityRequestConfig, String> {
    let merged = configured_headers(config, extension)
        .map(|headers| {
            headers
                .into_iter()
                .chain(raw_model_headers(model, config, extension).unwrap_or_default())
                .collect::<BTreeMap<String, String>>()
        })
        .or_else(|| raw_model_headers(model, config, extension));
    let configured = resolve_headers_or_throw(
        merged.as_ref(),
        &format!("model \"{}/{}\"", model.provider.0, model.id),
        None,
    )?;
    let headers: Option<ProviderHeaders> = if model.headers.is_some() || configured.is_some() {
        let mut merged: ProviderHeaders = model
            .headers
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(|(name, value)| (name, Some(value)))
            .collect();
        for (name, value) in configured.unwrap_or_default() {
            merged.insert(name, Some(value));
        }
        Some(merged)
    } else {
        None
    };
    Ok(CompatibilityRequestConfig {
        headers,
        auth_header: extension
            .and_then(|extension| extension.auth_header)
            .or_else(|| config.and_then(|config| config.auth_header))
            .unwrap_or(false),
    })
}

/// The compatibility fallback's shape, upstream's
/// `CompatibilityRequestConfig`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompatibilityRequestConfig {
    /// The merged static headers.
    pub headers: Option<ProviderHeaders>,
    /// Whether the request must carry `Authorization: Bearer`.
    pub auth_header: bool,
}

/// The configured-auth status the composed layers report, upstream's
/// `configuredRequestAuthStatus`.
#[must_use]
pub fn configured_request_auth_status(
    config: Option<&ModelsJsonProvider>,
    extension: Option<&ProviderConfigInput>,
) -> Option<AuthStatus> {
    let value = configured_api_key(config, extension)?.to_owned();
    if is_command_config_value(&value) {
        return Some(AuthStatus {
            configured: true,
            source: Some(AuthStatusSource::ModelsJsonCommand),
            label: None,
        });
    }
    let names = get_config_value_env_var_names(&value);
    if !names.is_empty() {
        return Some(if is_config_value_configured(&value, None) {
            AuthStatus {
                configured: true,
                source: Some(AuthStatusSource::Environment),
                label: Some(names.join(", ")),
            }
        } else {
            AuthStatus {
                configured: false,
                source: None,
                label: None,
            }
        });
    }
    Some(AuthStatus {
        configured: true,
        source: Some(
            if extension.is_some_and(|extension| extension.api_key.is_some()) {
                AuthStatusSource::Fallback
            } else {
                AuthStatusSource::ModelsJsonKey
            },
        ),
        label: None,
    })
}

//! The synchronous compatibility facade exposed to extensions, upstream's
//! `src/core/model-registry.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Coding-agent internals use [`ModelRuntime`](crate::model_runtime::ModelRuntime)
//! directly.
//!
//! Porting restatement: upstream's `registerProvider` overloads split into
//! [`ModelRegistry::register_provider`] (a native pi-ai provider object) and
//! [`ModelRegistry::register_provider_config`] (a name plus configuration),
//! the two arms of the discriminated overload.

use std::sync::Arc;

use pi_ai::auth::types::AuthResult;
use pi_ai::models::{ModelsSimpleStreamOptions, ModelsStreamOptions, Provider};
use pi_ai::types::{AssistantMessage, BoxedFuture, Context, Model, ProviderHeaders};
use pi_ai::utils::event_stream::AssistantMessageEventStream;

use crate::model_runtime::ModelRuntime;
use crate::provider_composer::{AuthStatus, CompatibilityRequestConfig, ProviderConfigInput};

/// The auth a request resolves to, upstream's `ResolvedRequestAuth`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedRequestAuth {
    /// The request may proceed with this auth.
    Ok {
        /// The resolved API key.
        api_key: Option<String>,
        /// The merged request headers.
        headers: Option<ProviderHeaders>,
        /// The base URL override.
        base_url: Option<String>,
        /// The provider-scoped environment values the resolution carried.
        env: Option<pi_ai::types::ProviderEnv>,
    },
    /// The request must not proceed; the message is user-facing.
    Err {
        /// The error message.
        error: String,
    },
}

impl Default for ResolvedRequestAuth {
    fn default() -> Self {
        Self::Ok {
            api_key: None,
            headers: None,
            base_url: None,
            env: None,
        }
    }
}

impl ResolvedRequestAuth {
    /// Whether the resolution succeeded.
    #[must_use]
    pub const fn ok(&self) -> bool {
        matches!(self, Self::Ok { .. })
    }
}

/// The synchronous compatibility facade, upstream's `ModelRegistry`.
#[derive(Clone)]
pub struct ModelRegistry {
    runtime: ModelRuntime,
}

impl std::fmt::Debug for ModelRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelRegistry").finish_non_exhaustive()
    }
}

impl ModelRegistry {
    /// The facade over a runtime, upstream's constructor.
    #[must_use]
    pub const fn new(runtime: ModelRuntime) -> Self {
        Self { runtime }
    }

    /// The runtime the facade fronts, the seam the coding-agent internals
    /// use.
    #[must_use]
    pub const fn runtime(&self) -> &ModelRuntime {
        &self.runtime
    }

    /// Reload models.json asynchronously. Await before making synchronous
    /// registry reads, upstream's `refresh`.
    pub async fn refresh(
        &self,
        options: Option<pi_ai::models::ModelsRefreshOptions>,
    ) -> pi_ai::models::ModelsRefreshResult {
        self.runtime.refresh(options.unwrap_or_default()).await
    }

    /// The combined load/composition/availability error, upstream's
    /// `getError`.
    #[must_use]
    pub fn get_error(&self) -> Option<String> {
        self.runtime.get_error()
    }

    /// Every composed model, upstream's `getAll`.
    #[must_use]
    pub fn get_all(&self) -> Vec<Model> {
        self.runtime.get_models(None)
    }

    /// The last-known available models, upstream's `getAvailable`.
    #[must_use]
    pub fn get_available(&self) -> Vec<Model> {
        self.runtime.get_available_snapshot()
    }

    /// One model by provider and id, upstream's `find`.
    #[must_use]
    pub fn find(&self, provider: &str, model_id: &str) -> Option<Model> {
        self.runtime.get_model(provider, model_id)
    }

    /// Whether the model's provider has configured auth, upstream's
    /// `hasConfiguredAuth`.
    #[must_use]
    pub fn has_configured_auth(&self, model: &Model) -> bool {
        self.runtime.has_configured_auth(&model.provider.0)
    }

    /// The request auth for a model, upstream's `getApiKeyAndHeaders`: the
    /// unconfigured compatibility fallback carries static headers, and the
    /// legacy missing-key message is preserved.
    pub async fn get_api_key_and_headers(&self, model: &Model) -> ResolvedRequestAuth {
        let resolution = match self.runtime.get_auth_for_model(model, None).await {
            Ok(resolution) => resolution,
            Err(error) => {
                let message = match &error {
                    pi_ai::auth::resolve::ModelsFailure::Models(models_error) => {
                        // Callers surface `error.message` only; the wrapped
                        // reason rides the display text.
                        models_error.to_string()
                    }
                    pi_ai::auth::resolve::ModelsFailure::Aborted(abort) => abort.to_string(),
                };
                let message = if message == "authHeader requires a resolved API key" {
                    format!("No API key found for \"{}\"", model.provider.0)
                } else {
                    message
                };
                return ResolvedRequestAuth::Err { error: message };
            }
        };
        let Some(resolution) = resolution else {
            let Ok(compatibility) = self.runtime.get_compatibility_request_config(model) else {
                return ResolvedRequestAuth::Err {
                    error: format!("No API key found for \"{}\"", model.provider.0),
                };
            };
            if compatibility.auth_header {
                return ResolvedRequestAuth::Err {
                    error: format!("No API key found for \"{}\"", model.provider.0),
                };
            }
            return ResolvedRequestAuth::Ok {
                api_key: None,
                headers: compatibility.headers,
                base_url: None,
                env: None,
            };
        };
        ResolvedRequestAuth::Ok {
            api_key: resolution.auth.api_key,
            headers: resolution.auth.headers,
            base_url: resolution.auth.base_url,
            env: resolution.env,
        }
    }

    /// The configured-auth status a provider reports, upstream's
    /// `getProviderAuthStatus`.
    #[must_use]
    pub fn get_provider_auth_status(&self, provider: &str) -> AuthStatus {
        self.runtime.get_provider_auth_status(provider)
    }

    /// One provider object, upstream's `getProvider`.
    #[must_use]
    pub fn get_provider(&self, provider: &str) -> Option<Arc<dyn Provider>> {
        self.runtime.get_provider(provider)
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
        self.runtime.stream(model, context, options)
    }

    /// Stream with provider-neutral options and request-time authentication,
    /// upstream's `streamSimple`.
    #[must_use]
    pub fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&ModelsSimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        self.runtime.stream_simple(model, context, options)
    }

    /// Complete a request, upstream's `complete`.
    pub async fn complete(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&ModelsStreamOptions>,
    ) -> AssistantMessage {
        self.runtime.complete(model, context, options).await
    }

    /// The display name a provider reports, upstream's
    /// `getProviderDisplayName`.
    #[must_use]
    pub fn get_provider_display_name(&self, provider: &str) -> String {
        self.runtime.get_provider(provider).map_or_else(
            || provider.to_owned(),
            |provider| provider.name().to_owned(),
        )
    }

    /// The resolved auth a provider carries, upstream's `getProviderAuth`.
    ///
    /// # Errors
    /// The resolution failure.
    #[must_use]
    pub fn get_provider_auth(
        &self,
        provider: &str,
    ) -> BoxedFuture<'_, Result<Option<AuthResult>, pi_ai::auth::resolve::ModelsFailure>> {
        self.runtime.get_auth(provider, None)
    }

    /// The resolved API key a provider carries, upstream's
    /// `getApiKeyForProvider`: resolution failures yield `None`.
    pub async fn get_api_key_for_provider(&self, provider: &str) -> Option<String> {
        self.runtime
            .get_auth(provider, None)
            .await
            .ok()
            .flatten()
            .and_then(|resolution| resolution.auth.api_key)
    }

    /// Whether the model's provider resolves through OAuth, upstream's
    /// `isUsingOAuth`.
    #[must_use]
    pub fn is_using_oauth(&self, model: &Model) -> bool {
        self.runtime.is_using_oauth(&model.provider.0)
    }

    /// Register a native pi-ai provider object, upstream's
    /// `registerProvider(provider)`.
    pub fn register_provider(&self, provider: Arc<dyn Provider>) {
        self.runtime.register_native_provider(provider);
    }

    /// Register an extension provider configuration, upstream's
    /// `registerProvider(name, config)`.
    ///
    /// # Errors
    /// The registration's own validation failure.
    pub fn register_provider_config(
        &self,
        provider_name: &str,
        config: ProviderConfigInput,
    ) -> Result<(), String> {
        self.runtime.register_provider(provider_name, config)
    }

    /// Remove an extension registration, upstream's `unregisterProvider`.
    pub fn unregister_provider(&self, provider_name: &str) {
        self.runtime.unregister_provider(provider_name);
    }

    /// The config an extension registered, upstream's
    /// `getRegisteredProviderConfig`.
    #[must_use]
    pub fn get_registered_provider_config(
        &self,
        provider_name: &str,
    ) -> Option<Arc<ProviderConfigInput>> {
        self.runtime.get_registered_provider_config(provider_name)
    }

    /// The native provider object an extension registered, upstream's
    /// `getRegisteredNativeProvider`.
    #[must_use]
    pub fn get_registered_native_provider(&self, provider_name: &str) -> Option<Arc<dyn Provider>> {
        self.runtime.get_registered_native_provider(provider_name)
    }

    /// The ids extensions registered, upstream's `getRegisteredProviderIds`.
    #[must_use]
    pub fn get_registered_provider_ids(&self) -> Vec<String> {
        self.runtime.get_registered_provider_ids()
    }

    /// The runtime the compatibility request config reads, upstream's
    /// `ModelRuntime.getCompatibilityRequestConfig`.
    ///
    /// # Errors
    /// A header value that fails to resolve.
    pub fn get_compatibility_request_config(
        &self,
        model: &Model,
    ) -> Result<CompatibilityRequestConfig, String> {
        self.runtime.get_compatibility_request_config(model)
    }
}

//! The Radius gateway provider, ported from
//! `packages/ai/src/providers/radius.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::{Arc, Mutex};

use crate::auth::helpers::{env_api_key_auth, lazy_oauth};
use crate::auth::types::{Credential, ProviderAuth};
use crate::models::{
    CatalogPersist, ModelsPublication, Provider, ProviderError, RefreshModelsContext,
};
use crate::models_store::ModelsStoreEntry;
use crate::providers::radius_config::{
    DEFAULT_RADIUS_GATEWAY, get_radius_models, get_radius_models_from_config,
    load_radius_gateway_config, normalize_radius_gateway_url,
};

/// The options of [`radius_provider`], upstream's `RadiusProviderOptions`.
#[derive(Clone, Debug, Default)]
pub struct RadiusProviderOptions {
    /// The provider id. Default: "radius".
    pub id: Option<String>,
    /// The display name. Default: "Radius".
    pub name: Option<String>,
    /// The gateway URL. Default: [`DEFAULT_RADIUS_GATEWAY`].
    pub gateway: Option<String>,
}

/// Radius gateway provider with a persisted, dynamically refreshed catalog,
/// upstream's `radiusProvider(options)`.
#[must_use]
pub fn radius_provider(options: RadiusProviderOptions) -> Arc<dyn Provider> {
    let id = options.id.unwrap_or_else(|| "radius".to_owned());
    let name = options.name.unwrap_or_else(|| "Radius".to_owned());
    let gateway = normalize_radius_gateway_url(
        options
            .gateway
            .unwrap_or_else(|| DEFAULT_RADIUS_GATEWAY.to_owned())
            .as_str(),
    );

    // The provider-private in-memory catalog state the publications update,
    // upstream's `let models = getRadiusModels(id, undefined)` closure.
    let models: Arc<Mutex<Vec<crate::types::Model>>> = Arc::new(Mutex::new(Vec::new()));
    let streams = crate::api::pi_messages();

    let core = RadiusCore {
        id: id.clone(),
        models: Arc::clone(&models),
        gateway: Arc::new(gateway.clone()),
    };

    let oauth_name = name.clone();
    let oauth_name_for_load = oauth_name.clone();
    let oauth_gateway = Arc::new(gateway);
    Arc::new(RadiusProvider {
        id,
        name,
        auth: ProviderAuth {
            api_key: Some(env_api_key_auth("Radius API key", &["RADIUS_API_KEY"])),
            oauth: Some(lazy_oauth(crate::auth::helpers::LazyOAuthInput {
                name: oauth_name,
                is_subscription: None,
                login_label: None,
                load: Arc::new(move || {
                    let name = oauth_name_for_load.clone();
                    let gateway = Arc::clone(&oauth_gateway);
                    Box::pin(async move {
                        crate::auth::oauth::load_radius_oauth(
                            &crate::auth::oauth::RadiusOAuthOptions {
                                name,
                                gateway: (*gateway).clone(),
                            },
                        )
                        .await
                    })
                }),
            })),
        },
        core,
        models,
        streams,
    })
}

/// The shared state the radius provider's refresh phases mutate, upstream's
/// closure-captured `models`.
struct RadiusCore {
    id: String,
    models: Arc<Mutex<Vec<crate::types::Model>>>,
    gateway: Arc<String>,
}

impl RadiusCore {
    /// The provider's full refresh flow, upstream's `refreshModels`: restore
    /// the persisted catalog, import legacy credential-cached catalogs, then
    /// fetch the network config when allowed.
    async fn refresh(&self, context: &RefreshModelsContext) -> Result<(), ProviderError> {
        let stored = context.stored.as_ref().map(|entry| {
            entry
                .models
                .iter()
                .filter(|model| model.provider.0 == self.id)
                .cloned()
                .collect::<Vec<_>>()
        });

        // Restore the persisted catalog before anything else, upstream's
        // `context.stored` restore-and-publish.
        if let Some(restored) = &stored {
            let published = (context.publish)(ModelsPublication {
                persist: CatalogPersist::Omit,
                update: Some(Box::new({
                    let models = Arc::clone(&self.models);
                    let restored = restored.clone();
                    move || {
                        models
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

        // Import catalogs cached by the pre-ModelsStore Radius
        // implementation: a stored OAuth credential may carry a gateway
        // config even when nothing was persisted yet.
        if context.stored.is_none()
            && let Some(Credential::OAuth(oauth)) = context.credential.as_ref()
        {
            let legacy = get_radius_models(self.id.as_str(), Some(oauth));
            if !legacy.is_empty() {
                let published = (context.publish)(ModelsPublication {
                    persist: CatalogPersist::Write(ModelsStoreEntry {
                        models: legacy.clone(),
                        checked_at: Some(crate::auth::resolve::now_ms()),
                        ..ModelsStoreEntry::default()
                    }),
                    update: Some(Box::new({
                        let models = Arc::clone(&self.models);
                        move || {
                            models
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .clone_from(&legacy);
                        }
                    })),
                })
                .await?;
                if !published {
                    return Ok(());
                }
            }
        }

        if !context.allow_network || context.signal.is_cancelled() {
            return Ok(());
        }
        let api_key = match context.credential.as_ref() {
            Some(Credential::OAuth(oauth)) => Some(oauth.access.clone()),
            Some(Credential::ApiKey(api_key)) => api_key.key.clone(),
            None => None,
        };
        let config =
            load_radius_gateway_config(self.gateway.as_str(), api_key.as_deref(), &context.signal)
                .await?;
        if context.signal.is_cancelled() {
            return Ok(());
        }
        let refreshed = get_radius_models_from_config(self.id.as_str(), &config);
        (context.publish)(ModelsPublication {
            persist: CatalogPersist::Write(ModelsStoreEntry {
                models: refreshed.clone(),
                checked_at: Some(crate::auth::resolve::now_ms()),
                ..ModelsStoreEntry::default()
            }),
            update: Some(Box::new({
                let models = Arc::clone(&self.models);
                move || {
                    models
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone_from(&refreshed);
                }
            })),
        })
        .await?;
        Ok(())
    }
}

/// The hand-rolled provider object upstream returns from `radiusProvider`:
/// unlike `createProvider`-built providers, its `refreshModels` runs the
/// restore, legacy import, and network phases itself in both refresh phases.
struct RadiusProvider {
    id: String,
    name: String,
    auth: ProviderAuth,
    core: RadiusCore,
    models: Arc<Mutex<Vec<crate::types::Model>>>,
    streams: Arc<dyn crate::types::ProviderStreams>,
}

impl std::fmt::Debug for RadiusProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RadiusProvider")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl Provider for RadiusProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn get_models(&self) -> Result<Vec<crate::types::Model>, crate::models::ProviderModelError> {
        Ok(self
            .models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone())
    }

    fn refresh_models(
        &self,
        context: RefreshModelsContext,
    ) -> crate::types::BoxedFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { self.core.refresh(&context).await })
    }

    fn supports_refresh_models(&self) -> bool {
        true
    }

    fn stream(
        &self,
        model: &crate::types::Model,
        context: &crate::types::Context,
        stream_options: Option<&crate::types::StreamOptions>,
    ) -> crate::utils::event_stream::AssistantMessageEventStream {
        self.streams.stream(model, context, stream_options)
    }

    fn stream_simple(
        &self,
        model: &crate::types::Model,
        context: &crate::types::Context,
        stream_options: Option<&crate::types::SimpleStreamOptions>,
    ) -> crate::utils::event_stream::AssistantMessageEventStream {
        self.streams.stream_simple(model, context, stream_options)
    }
}

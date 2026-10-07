//! Async credential store overlay for non-persistent runtime API keys,
//! upstream's `src/core/runtime-credentials.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::types::{
    ApiKeyCredential, AuthError, AuthOptions, AuthType, Credential, CredentialInfo,
    CredentialModifyFn,
};
use pi_ai::types::BoxedFuture;
use pi_ai::utils::abort::AbortError;

/// The runtime override map, upstream's `overrides`. The ordered map carries
/// the JS `Map` semantics the enumeration and iteration ride: insertion
/// order, an overwrite keeping the original position.
type Overrides = IndexMap<String, String>;

fn signal_check(options: Option<&AuthOptions>) -> Result<(), AuthError> {
    let signal = options.and_then(|options| options.signal.as_ref());
    if signal.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
        return Err(Box::new(AbortError));
    }
    Ok(())
}

/// The runtime credential overlay, upstream's `RuntimeCredentials`.
///
/// Overrides mask the wrapped store's reads and enumeration without
/// persisting, writes and modifications forward, and a successful delete
/// clears the override.
pub struct RuntimeCredentials {
    store: Arc<dyn CredentialStore>,
    overrides: Mutex<Overrides>,
}

impl std::fmt::Debug for RuntimeCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeCredentials").finish_non_exhaustive()
    }
}

impl RuntimeCredentials {
    /// The overlay over `store`, upstream's constructor.
    #[must_use]
    pub fn new(store: Arc<dyn CredentialStore>) -> Self {
        Self {
            store,
            overrides: Mutex::new(IndexMap::new()),
        }
    }

    /// Register a runtime API key for the provider, upstream's
    /// `setRuntimeApiKey`.
    pub fn set_runtime_api_key(&self, provider_id: &str, api_key: String) {
        self.overrides
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(provider_id.to_string(), api_key);
    }

    /// Drop the runtime API key, upstream's `removeRuntimeApiKey`.
    pub fn remove_runtime_api_key(&self, provider_id: &str) {
        self.overrides
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .shift_remove(provider_id);
    }

    /// Whether the provider has a runtime API key, upstream's
    /// `hasRuntimeApiKey`.
    #[must_use]
    pub fn has_runtime_api_key(&self, provider_id: &str) -> bool {
        self.overrides
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(provider_id)
    }
}

impl CredentialStore for RuntimeCredentials {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        Box::pin(async move {
            signal_check(options)?;
            let override_key = self
                .overrides
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(provider_id)
                .cloned();
            if let Some(key) = override_key {
                return Ok(Some(Credential::ApiKey(ApiKeyCredential {
                    key: Some(key),
                    env: None,
                })));
            }
            self.store.read(provider_id, options).await
        })
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        Box::pin(async move {
            let mut entries: IndexMap<String, CredentialInfo> = self
                .store
                .list(options)
                .await?
                .into_iter()
                .map(|entry| (entry.provider_id.clone(), entry))
                .collect();
            signal_check(options)?;
            let overrides = self
                .overrides
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for (provider_id, _) in overrides.iter() {
                entries.insert(
                    provider_id.clone(),
                    CredentialInfo {
                        provider_id: provider_id.clone(),
                        auth_type: AuthType::ApiKey,
                    },
                );
            }
            drop(overrides);
            Ok(entries.into_values().collect())
        })
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: CredentialModifyFn,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        self.store.modify(provider_id, f, options)
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        Box::pin(async move {
            signal_check(options)?;
            self.store.delete(provider_id, options).await?;
            self.overrides
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .shift_remove(provider_id);
            Ok(())
        })
    }
}

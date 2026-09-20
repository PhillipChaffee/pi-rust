//! Shared auth builders, ported from `packages/ai/src/auth/helpers.ts` at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::types::ModelAuth;
use crate::auth::types::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, ApiKeyLoginFn, ApiKeyResolveFn, AuthError,
    AuthPrompt, AuthPromptKind, AuthResult, OAuthAuth, ProviderAuthInteraction,
};
use crate::types::BoxedFuture;
use crate::utils::abort::AbortError;

/// Standard api-key auth, upstream's `envApiKeyAuth`.
///
/// A stored credential key wins, otherwise the first set env var resolves.
/// Includes a `login` that prompts for the key. Providers with
/// non-standard resolution (provider env, ambient files, IAM) write their own
/// [`ApiKeyAuth`].
#[must_use]
pub fn env_api_key_auth(name: &str, env_vars: &[&str]) -> ApiKeyAuth {
    let login_name = name.to_owned();
    let env_vars: Vec<String> = env_vars.iter().map(ToString::to_string).collect();
    let login: ApiKeyLoginFn = Arc::new(move |interaction: ProviderAuthInteraction| {
        let name = login_name.clone();
        Box::pin(async move {
            if interaction.signal.is_cancelled() {
                return Err(AbortError.into());
            }
            let key = (interaction.prompt)(AuthPrompt {
                signal: Some(interaction.signal.clone()),
                kind: AuthPromptKind::Secret {
                    message: format!("Enter {name}"),
                    placeholder: None,
                },
            })
            .await?;
            if interaction.signal.is_cancelled() {
                return Err(AbortError.into());
            }
            Ok(ApiKeyCredential {
                key: Some(key),
                env: None,
            })
        })
    });
    let resolve: ApiKeyResolveFn = Arc::new(move |input: ApiKeyAuthInput| {
        let env_vars = env_vars.clone();
        Box::pin(async move {
            if input.signal.is_cancelled() {
                return Err(AbortError.into());
            }
            if let Some(key) = input
                .credential
                .as_ref()
                .and_then(|credential| credential.key.clone())
            {
                return Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some(key),
                        ..ModelAuth::default()
                    },
                    env: input
                        .credential
                        .as_ref()
                        .and_then(|credential| credential.env.clone()),
                    source: Some("stored credential".to_owned()),
                }));
            }
            for env_var in env_vars {
                if input.signal.is_cancelled() {
                    return Err(AbortError.into());
                }
                let Some(value) = input.ctx.env(&env_var) else {
                    continue;
                };
                return Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some(value),
                        ..ModelAuth::default()
                    },
                    env: None,
                    source: Some(env_var),
                }));
            }
            Ok(None)
        })
    });
    ApiKeyAuth {
        name: name.to_owned(),
        login: Some(login),
        check: None,
        resolve,
    }
}

/// The input of [`lazy_oauth`], upstream's `lazyOAuth` argument object.
pub struct LazyOAuthInput {
    /// Display name.
    pub name: String,
    /// Whether access through this auth method is backed by a provider
    /// subscription.
    pub is_subscription: Option<bool>,
    /// Selector label for the OAuth login option.
    pub login_label: Option<String>,
    /// Load the implementation on first use; concurrent callers share one
    /// load.
    pub load: Arc<dyn Fn() -> BoxedFuture<'static, OAuthAuth> + Send + Sync>,
}

impl std::fmt::Debug for LazyOAuthInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyOAuthInput")
            .field("name", &self.name)
            .field("is_subscription", &self.is_subscription)
            .field("login_label", &self.login_label)
            .finish_non_exhaustive()
    }
}

/// Wraps a lazily loaded [`OAuthAuth`] so provider definitions can advertise
/// OAuth without importing the implementation, upstream's `lazyOAuth`. The
/// flow loads on first `login`/`refresh`/`toAuth` call.
///
/// Porting restatement: upstream caches the load promise even when it fails;
/// the Rust cell only caches successes, so a failed load retries on the next
/// call.
#[must_use]
pub fn lazy_oauth(input: LazyOAuthInput) -> OAuthAuth {
    let cell = Arc::new(tokio::sync::OnceCell::<OAuthAuth>::new());
    let loader = input.load;
    let load: Arc<dyn Fn() -> BoxedFuture<'static, OAuthAuth> + Send + Sync> =
        Arc::new(move || {
            let cell = Arc::clone(&cell);
            let loader = Arc::clone(&loader);
            let future: BoxedFuture<'static, OAuthAuth> =
                Box::pin(async move { cell.get_or_init(|| loader()).await.clone() });
            future
        });
    let login: crate::auth::types::OAuthLoginFn = {
        let load = Arc::clone(&load);
        Arc::new(move |interaction: ProviderAuthInteraction| {
            let load = Arc::clone(&load);
            Box::pin(async move { ((load)().await.login)(interaction).await })
        })
    };
    let refresh: crate::auth::types::OAuthRefreshFn = {
        let load = Arc::clone(&load);
        Arc::new(move |credential, signal| {
            let load = Arc::clone(&load);
            Box::pin(async move { ((load)().await.refresh)(credential, signal).await })
        })
    };
    let to_auth: crate::auth::types::OAuthToAuthFn = {
        let load = Arc::clone(&load);
        Arc::new(move |credential| {
            let load = Arc::clone(&load);
            Box::pin(async move { ((load)().await.to_auth)(credential).await })
        })
    };
    OAuthAuth {
        name: input.name,
        is_subscription: input.is_subscription,
        login_label: input.login_label,
        login,
        refresh,
        to_auth,
    }
}

/// The failure a not-yet-ported auth flow reports when invoked, the shared
/// error shape of the loader stubs in [`crate::auth::oauth`].
#[must_use]
pub fn oauth_stub_error(message: impl Into<String>) -> AuthError {
    Box::new(OAuthStubError(message.into()))
}

/// The failure a not-yet-ported OAuth flow reports when invoked.
#[derive(Debug)]
struct OAuthStubError(String);

impl std::fmt::Display for OAuthStubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for OAuthStubError {}

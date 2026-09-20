//! Auth resolution shared by the `Models` and `ImagesModels` collections,
//! ported from `packages/ai/src/auth/resolve.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio_util::sync::CancellationToken;

use crate::auth::credential_store::CredentialStore;
use crate::auth::types::{
    ApiKeyAuth, ApiKeyCredential, AuthContext, AuthError, AuthOptions, AuthResult, AuthType,
    Credential, CredentialModifyFn, OAuthAuth, OAuthCredentials, ProviderAuth,
};
use crate::types::{BoxedFuture, ProviderEnv};
use crate::utils::abort::{AbortError, RaceError, operation_signal, race_with_abort_signal};
use crate::utils::diagnostics::format_thrown_value;

/// How a Models operation stopped early: the caller's abort, or a wrapped
/// [`ModelsError`]. The abort variant is the `AbortError` rejection upstream
/// surfaces from raced operations.
#[derive(Debug)]
pub enum ModelsFailure {
    /// The signal aborted before the operation settled.
    Aborted(AbortError),
    /// The operation failed with a wrapped [`ModelsError`].
    Models(ModelsError),
}

impl fmt::Display for ModelsFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aborted(abort) => fmt::Display::fmt(abort, f),
            Self::Models(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl std::error::Error for ModelsFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Aborted(abort) => Some(abort),
            Self::Models(error) => Some(error),
        }
    }
}

impl From<ModelsError> for ModelsFailure {
    fn from(value: ModelsError) -> Self {
        Self::Models(value)
    }
}

impl From<RaceError<ModelsError>> for ModelsFailure {
    fn from(value: RaceError<ModelsError>) -> Self {
        match value {
            RaceError::Aborted(abort) => Self::Aborted(abort),
            RaceError::Operation(error) => Self::Models(error),
        }
    }
}

/// The failure codes of [`ModelsError`], upstream's `ModelsErrorCode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelsErrorCode {
    /// Model catalog source failure (refresh, remote catalog).
    ModelSource,
    /// Generated model data failed validation.
    ModelValidation,
    /// Unknown provider or provider does not support the operation.
    Provider,
    /// Stream dispatch failure inside a provider.
    Stream,
    /// Api-key resolution or credential store failure.
    Auth,
    /// OAuth refresh or derivation failure.
    OAuth,
}

impl fmt::Display for ModelsErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.wire())
    }
}

impl ModelsErrorCode {
    /// The wire spelling upstream carries, e.g. `"oauth"`.
    #[must_use]
    pub const fn wire(&self) -> &'static str {
        match self {
            Self::ModelSource => "model_source",
            Self::ModelValidation => "model_validation",
            Self::Provider => "provider",
            Self::Stream => "stream",
            Self::Auth => "auth",
            Self::OAuth => "oauth",
        }
    }
}

/// The error Models collections reject with, upstream's `ModelsError`.
#[derive(Debug)]
pub struct ModelsError {
    code: ModelsErrorCode,
    message: String,
    source: Option<AuthError>,
}

impl ModelsError {
    /// A failure with its code and message.
    #[must_use]
    pub fn new(code: ModelsErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            source: None,
        }
    }

    /// A failure carrying an underlying reason; the reason's display text is
    /// appended to the message per upstream's `withCauseDetail`, because
    /// callers surface `error.message` only.
    #[must_use]
    pub fn with_cause(code: ModelsErrorCode, message: impl Into<String>, cause: AuthError) -> Self {
        let message = with_cause_detail(&message.into(), Some(&cause));
        Self {
            code,
            message,
            source: Some(cause),
        }
    }

    /// The failure code.
    #[must_use]
    pub const fn code(&self) -> ModelsErrorCode {
        self.code
    }
}

impl fmt::Display for ModelsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ModelsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        let source: &(dyn std::error::Error + 'static) = self.source.as_deref()?;
        Some(source)
    }
}

/// Callers surface `error.message` only, so keep the underlying reason in it.
fn with_cause_detail(message: &str, cause: Option<&AuthError>) -> String {
    let Some(cause) = cause else {
        return message.to_owned();
    };
    let detail = format_thrown_value(cause.as_ref()).trim().to_owned();
    if detail.is_empty() || message.contains(&detail) {
        return message.to_owned();
    }
    format!("{message}: {detail}")
}

/// Auth-resolution overrides, upstream's `AuthResolutionOverrides`.
#[derive(Clone, Debug, Default)]
pub struct AuthResolutionOverrides {
    /// Use this API key instead of resolving one.
    pub api_key: Option<String>,
    /// Overlay these provider-scoped environment values over the auth
    /// context.
    pub env: Option<ProviderEnv>,
    /// Require this much remaining OAuth-token validity in milliseconds;
    /// defaults to five minutes.
    pub min_oauth_validity_ms: Option<i64>,
    /// Cancellation for the resolution.
    pub signal: Option<CancellationToken>,
}

/// Unix milliseconds, the port of `Date.now()`.
#[must_use]
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
        })
}

/// Auth resolution shared by the `Models` and `ImagesModels` collections.
///
/// A stored credential owns the provider: ambient/env is consulted only when
/// nothing is stored. No silent env fallback after a failed refresh or for a
/// credential type without a matching handler.
///
/// # Errors
/// The abort failure, or the wrapped [`ModelsError`] of the failing step.
pub fn resolve_provider_auth(
    provider_id: &str,
    auth: &ProviderAuth,
    credentials: &Arc<dyn CredentialStore>,
    auth_context: &Arc<dyn AuthContext>,
    overrides: Option<&AuthResolutionOverrides>,
) -> BoxedFuture<'static, Result<Option<AuthResult>, ModelsFailure>> {
    let signal = operation_signal(overrides.and_then(|overrides| overrides.signal.as_ref()));
    let provider_id = provider_id.to_owned();
    let auth = auth.clone();
    let credentials = Arc::clone(credentials);
    let auth_context = Arc::clone(auth_context);
    let overrides = overrides.cloned();
    Box::pin(async move {
        let operation = resolve_provider_auth_with_signal(
            &provider_id,
            &auth,
            &credentials,
            &auth_context,
            overrides.as_ref(),
            &signal,
        );
        race_with_abort_signal(operation, &signal)
            .await
            .map_err(ModelsFailure::from)
    })
}

async fn resolve_provider_auth_with_signal(
    provider_id: &str,
    auth: &ProviderAuth,
    credentials: &Arc<dyn CredentialStore>,
    auth_context: &Arc<dyn AuthContext>,
    overrides: Option<&AuthResolutionOverrides>,
    signal: &CancellationToken,
) -> Result<Option<AuthResult>, ModelsError> {
    let request_auth_context = match overrides.and_then(|overrides| overrides.env.clone()) {
        Some(env) if !env.is_empty() => {
            let overlay: Arc<dyn AuthContext> = Arc::new(OverlayEnvAuthContext {
                base: Arc::clone(auth_context),
                env,
            });
            overlay
        }
        _ => {
            let context: Arc<dyn AuthContext> = Arc::clone(auth_context);
            context
        }
    };

    if let (Some(api_key_override), Some(api_key)) = (
        overrides.and_then(|overrides| overrides.api_key.clone()),
        &auth.api_key,
    ) {
        return resolve_api_key(
            &request_auth_context,
            api_key,
            provider_id,
            Some(ApiKeyCredential {
                key: Some(api_key_override),
                env: overrides.and_then(|overrides| overrides.env.clone()),
            }),
            signal,
        )
        .await;
    }

    let stored = read_credential(credentials, provider_id, signal).await?;
    let Some(stored) = stored else {
        // Ambient (env vars, AWS profiles, ADC files).
        return match &auth.api_key {
            Some(api_key) => {
                resolve_api_key(&request_auth_context, api_key, provider_id, None, signal).await
            }
            None => Ok(None),
        };
    };

    match stored.auth_type() {
        AuthType::OAuth => {
            let Some(oauth) = &auth.oauth else {
                return Ok(None);
            };
            let Some(stored_oauth) = stored.as_oauth() else {
                return Ok(None);
            };
            resolve_stored_oauth(
                credentials,
                provider_id,
                oauth,
                stored_oauth.clone(),
                signal,
                overrides.and_then(|overrides| overrides.min_oauth_validity_ms),
            )
            .await
        }
        AuthType::ApiKey => {
            let Some(api_key) = &auth.api_key else {
                return Ok(None);
            };
            let Some(mut credential) = stored.as_api_key_credential().cloned() else {
                return Ok(None);
            };
            if let Some(overrides_env) = overrides.and_then(|overrides| overrides.env.clone()) {
                let mut merged = credential.env.take().unwrap_or_default();
                merged.extend(overrides_env);
                credential.env = Some(merged);
            }
            resolve_api_key(
                &request_auth_context,
                api_key,
                provider_id,
                Some(credential),
                signal,
            )
            .await
        }
    }
}

/// The auth context with provider env overlaid, upstream's
/// `overlayEnvAuthContext`.
struct OverlayEnvAuthContext {
    base: Arc<dyn AuthContext>,
    env: ProviderEnv,
}

impl AuthContext for OverlayEnvAuthContext {
    fn env(&self, name: &str) -> Option<String> {
        self.env.get(name).cloned().or_else(|| self.base.env(name))
    }

    fn file_exists(&self, path: &str) -> bool {
        self.base.file_exists(path)
    }
}

/// The default OAuth minimum remaining validity in milliseconds, upstream's
/// `DEFAULT_OAUTH_MINIMUM_VALIDITY_MS`.
const DEFAULT_OAUTH_MINIMUM_VALIDITY_MS: i64 = 5 * 60 * 1000;
/// The OAuth refresh deadline in milliseconds, upstream's
/// `DEFAULT_OAUTH_REFRESH_TIMEOUT_MS`.
const DEFAULT_OAUTH_REFRESH_TIMEOUT_MS: u64 = 15_000;

/// OAuth resolution with double-checked locking: tokens with less than five
/// minutes remaining lock, re-check expiry under the lock, refresh once
/// globally, and persist the rotated credential before release.
#[expect(
    clippy::too_many_lines,
    reason = "the double-checked locking walk mirrors upstream's resolveStoredOAuth one to one"
)]
async fn resolve_stored_oauth(
    credentials: &Arc<dyn CredentialStore>,
    provider_id: &str,
    oauth: &OAuthAuth,
    stored: OAuthCredentials,
    signal: &CancellationToken,
    min_oauth_validity_ms: Option<i64>,
) -> Result<Option<AuthResult>, ModelsError> {
    let minimum_validity_ms =
        DEFAULT_OAUTH_MINIMUM_VALIDITY_MS.max(min_oauth_validity_ms.unwrap_or(0));
    let expires_soon =
        |credential: &OAuthCredentials| now_ms() + minimum_validity_ms >= credential.expires;
    let mut credential = stored;

    if expires_soon(&credential) {
        // Optimistic check said expired; the authoritative check runs under
        // the lock. The refresh itself races a 15-second deadline.
        let oauth = oauth.clone();
        let refresh_signal = signal.clone();
        let refresh_provider_id = provider_id.to_owned();
        let modify: CredentialModifyFn = Box::new(move |current| {
            let oauth = oauth.clone();
            let refresh_signal = refresh_signal.clone();
            let provider_id = refresh_provider_id;
            Box::pin(async move {
                let Some(current) = current else {
                    return Ok(None); // logged out meanwhile
                };
                let Some(current) = current.as_oauth() else {
                    return Ok(None); // not an OAuth credential anymore
                };
                if now_ms() + minimum_validity_ms < current.expires {
                    // The token outlives the freshness window: another
                    // process/request refreshed.
                    return Ok(None);
                }
                match tokio::time::timeout(
                    std::time::Duration::from_millis(DEFAULT_OAUTH_REFRESH_TIMEOUT_MS),
                    (oauth.refresh)(current.clone(), refresh_signal),
                )
                .await
                {
                    Ok(Ok(refreshed)) => Ok(Some(Credential::OAuth(refreshed))),
                    Ok(Err(error)) => {
                        let cause: AuthError = Box::new(ModelsError::with_cause(
                            ModelsErrorCode::OAuth,
                            format!("OAuth refresh failed for {provider_id}"),
                            error,
                        ));
                        Err(cause)
                    }
                    Err(_) => {
                        let cause: AuthError = Box::new(ModelsError::new(
                            ModelsErrorCode::OAuth,
                            format!("OAuth refresh timed out for {provider_id}"),
                        ));
                        Err(cause)
                    }
                }
            })
        });
        let post = credentials
            .modify(
                provider_id,
                modify,
                Some(&AuthOptions {
                    signal: Some(signal.clone()),
                }),
            )
            .await;
        let post = match post {
            Ok(post) => post,
            Err(error) => {
                // A ModelsError from the refresh propagates unwrapped,
                // upstream's `instanceof ModelsError` check.
                if error.is::<ModelsError>() {
                    let unboxed: Box<ModelsError> = error
                        .downcast()
                        .unwrap_or_else(|_| unreachable!("checked above"));
                    return Err(*unboxed);
                }
                return Err(ModelsError::with_cause(
                    ModelsErrorCode::Auth,
                    format!("Credential store modify failed for {provider_id}"),
                    error,
                ));
            }
        };
        let Some(post) = post.and_then(|post| post.as_oauth().cloned()) else {
            return Ok(None); // logged out meanwhile
        };
        credential = post;
        // The normal five-minute window triggers a refresh but does not
        // impose a provider contract. Explicit callers (such as bearer-token
        // export) do require the requested minimum after the refresh.
        if min_oauth_validity_ms.is_some() && expires_soon(&credential) {
            return Err(ModelsError::new(
                ModelsErrorCode::OAuth,
                format!("OAuth refresh returned a token that expires too soon for {provider_id}"),
            ));
        }
    }

    let auth = match (oauth.to_auth)(credential).await {
        Ok(auth) => auth,
        Err(error) => {
            return Err(ModelsError::with_cause(
                ModelsErrorCode::OAuth,
                format!("OAuth auth derivation failed for {provider_id}"),
                error,
            ));
        }
    };
    Ok(Some(AuthResult {
        auth,
        env: None,
        source: Some("OAuth".to_owned()),
    }))
}

async fn resolve_api_key(
    auth_context: &Arc<dyn AuthContext>,
    api_key: &ApiKeyAuth,
    provider_id: &str,
    credential: Option<ApiKeyCredential>,
    signal: &CancellationToken,
) -> Result<Option<AuthResult>, ModelsError> {
    let input = crate::auth::types::ApiKeyAuthInput {
        ctx: Arc::clone(auth_context),
        credential,
        signal: signal.clone(),
    };
    match (api_key.resolve)(input).await {
        Ok(result) => Ok(result),
        Err(error) => Err(ModelsError::with_cause(
            ModelsErrorCode::Auth,
            format!("API key auth failed for provider {provider_id}"),
            error,
        )),
    }
}

async fn read_credential(
    credentials: &Arc<dyn CredentialStore>,
    provider_id: &str,
    signal: &CancellationToken,
) -> Result<Option<Credential>, ModelsError> {
    match credentials
        .read(
            provider_id,
            Some(&AuthOptions {
                signal: Some(signal.clone()),
            }),
        )
        .await
    {
        Ok(credential) => Ok(credential),
        Err(error) => Err(ModelsError::with_cause(
            ModelsErrorCode::Auth,
            format!("Credential store read failed for {provider_id}"),
            error,
        )),
    }
}

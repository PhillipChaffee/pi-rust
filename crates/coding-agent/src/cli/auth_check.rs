//! The `auth check` command, upstream's `src/cli/auth-check.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::types::AuthType;

use crate::cli::auth_command::{
    AuthCommandArgs, AuthCommandError, AuthCommandKind, get_auth_credential,
    validate_auth_command_args,
};
use crate::model_resolver::resolve_cli_model;
use crate::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::models_store::InMemoryCodingAgentModelsStore;

/// The check outcome, upstream's `AuthCheckStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthCheckStatus {
    /// The provider is configured and ready, upstream's `"ready"`.
    Ready,
    /// Credentials are missing or the provider is unknown, upstream's
    /// `"not_ready"`.
    NotReady,
    /// The stored state is unreadable, upstream's `"invalid"`.
    Invalid,
}

impl AuthCheckStatus {
    /// The wire's `snake_case` tag, upstream's union member.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::NotReady => "not_ready",
            Self::Invalid => "invalid",
        }
    }
}

/// Why a check did not reach `ready`, upstream's `AuthCheckReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthCheckReason {
    /// The provider is not registered, upstream's `"provider_not_found"`.
    ProviderNotFound,
    /// No credential is stored, upstream's `"credentials_not_configured"`.
    CredentialsNotConfigured,
    /// The credential exists but did not resolve, upstream's
    /// `"credential_not_available"`.
    CredentialNotAvailable,
    /// The runtime or storage is in an unreadable state, upstream's
    /// `"invalid_state"`.
    InvalidState,
}

impl AuthCheckReason {
    /// The wire's `snake_case` tag, upstream's union member.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProviderNotFound => "provider_not_found",
            Self::CredentialsNotConfigured => "credentials_not_configured",
            Self::CredentialNotAvailable => "credential_not_available",
            Self::InvalidState => "invalid_state",
        }
    }
}

/// The check result, upstream's `AuthCheckResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthCheckResult {
    /// The outcome, upstream's `status`.
    pub status: AuthCheckStatus,
    /// The resolved provider id, upstream's `provider`.
    pub provider: String,
    /// Why the check missed `ready`, upstream's `reason?`.
    pub reason: Option<AuthCheckReason>,
    /// The credential type when ready, upstream's `authType?`.
    pub auth_type: Option<AuthType>,
}

/// The check knobs, upstream's `{ refresh: boolean }`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AuthCheckOptions {
    /// Resolve-and-refresh the credential through the request-auth path,
    /// upstream's `refresh`.
    pub refresh: bool,
}

/// Check one provider's auth state, upstream's `checkProviderAuth`.
///
/// # Errors
/// The [`AuthCommandError`]s `validateAuthCommandArgs` and the model
/// resolution throw; storage failures inside the check degrade to the
/// `invalid_state` result, upstream's catch.
pub async fn check_provider_auth(
    args: &AuthCommandArgs,
    model_runtime: &ModelRuntime,
    options: AuthCheckOptions,
) -> Result<AuthCheckResult, AuthCommandError> {
    let target = validate_auth_command_args(args, AuthCommandKind::Check)?;
    let provider = if let Some(cli_model) = target.model.as_deref() {
        let resolved = resolve_cli_model(crate::model_resolver::ResolveCliModelOptions {
            cli_provider: target.provider.as_deref(),
            cli_model: Some(cli_model),
            cli_thinking: None,
            model_runtime,
        });
        if resolved.error.is_some() || resolved.model.is_none() {
            return Err(AuthCommandError(resolved.error.unwrap_or_else(|| {
                format!("Unable to resolve model \"{cli_model}\"")
            })));
        }
        resolved.model.map(|model| model.provider.0)
    } else {
        target.provider.clone()
    };
    let Some(provider) = provider else {
        return Err(AuthCommandError(
            "Unable to resolve an auth provider".to_string(),
        ));
    };
    if let Some(load_error) = model_runtime.get_error() {
        let _ = load_error;
        return Ok(AuthCheckResult {
            status: AuthCheckStatus::Invalid,
            provider,
            reason: Some(AuthCheckReason::InvalidState),
            auth_type: None,
        });
    }
    if model_runtime.get_provider(&provider).is_none() {
        return Ok(AuthCheckResult {
            status: AuthCheckStatus::NotReady,
            provider,
            reason: Some(AuthCheckReason::ProviderNotFound),
            auth_type: None,
        });
    }
    match model_runtime.check_auth(&provider, None).await {
        Ok(None) => Ok(AuthCheckResult {
            status: AuthCheckStatus::NotReady,
            provider,
            reason: Some(AuthCheckReason::CredentialsNotConfigured),
            auth_type: None,
        }),
        Ok(Some(auth)) => {
            if options.refresh {
                let resolved = model_runtime.get_auth(&provider, None).await;
                let Ok(resolved) = resolved else {
                    return Ok(AuthCheckResult {
                        status: AuthCheckStatus::Invalid,
                        provider,
                        reason: Some(AuthCheckReason::InvalidState),
                        auth_type: None,
                    });
                };
                if resolved.is_none() {
                    return Ok(AuthCheckResult {
                        status: AuthCheckStatus::NotReady,
                        provider,
                        reason: Some(AuthCheckReason::CredentialsNotConfigured),
                        auth_type: None,
                    });
                }
            }
            Ok(AuthCheckResult {
                status: AuthCheckStatus::Ready,
                provider,
                reason: None,
                auth_type: Some(auth.auth_type),
            })
        }
        // Upstream's catch: any storage failure reads as invalid state.
        Err(_) => Ok(AuthCheckResult {
            status: AuthCheckStatus::Invalid,
            provider,
            reason: Some(AuthCheckReason::InvalidState),
            auth_type: None,
        }),
    }
}

/// Read the printable credential for one provider, upstream's
/// `getProviderCredential`: the stored OAuth access token without a
/// refresh, else the request-auth path's credential.
///
/// # Errors
/// The credential store's read failure, or the auth resolution failure —
/// upstream's propagating rejections.
pub async fn get_provider_credential(
    provider_id: &str,
    model_runtime: &ModelRuntime,
    credentials: &dyn CredentialStore,
    options: AuthCheckOptions,
) -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
    let credential = credentials.read(provider_id, None).await?;
    if !options.refresh
        && let Some(pi_ai::auth::types::Credential::OAuth(oauth)) = credential
    {
        return Ok(Some(oauth.access));
    }
    let auth = model_runtime.get_auth(provider_id, None).await;
    match auth {
        Ok(auth) => Ok(get_auth_credential(auth.as_ref())),
        Err(error) => Err(Box::new(error)),
    }
}

/// Build the check's model runtime, upstream's `createAuthCheckModelRuntime`:
/// no models.json, no network, no create-time refresh, the in-memory catalog
/// store.
///
/// # Errors
/// The runtime's create failure (an unreadable `auth_path` or a catalog
/// failure), upstream's rejected promise.
pub async fn create_auth_check_model_runtime(
    credentials: Arc<dyn CredentialStore>,
) -> Result<ModelRuntime, Box<dyn std::error::Error + Send + Sync>> {
    ModelRuntime::create(CreateModelRuntimeOptions {
        credentials: Some(credentials),
        models_path: Some(None),
        models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::default())),
        allow_model_network: false,
        refresh_on_create: Some(false),
        ..CreateModelRuntimeOptions::default()
    })
    .await
}

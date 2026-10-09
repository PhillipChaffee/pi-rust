//! The credential-print commands, upstream's `src/cli/credential-print.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Key material never logs: the resolved value returns to the caller, whose
//! print commands keep upstream's output contract (the house rule forbids
//! printing it anywhere else).

use tokio_util::sync::CancellationToken;

use pi_ai::auth::types::{AuthOptions, AuthType};

use crate::cli::auth_command::{
    AuthCommandArgs, AuthCommandError, AuthCommandKind, get_auth_credential,
    validate_auth_command_args,
};
use crate::model_resolver::resolve_cli_model;
use crate::model_runtime::{ModelRuntime, ModelRuntimeAuthOverrides};

/// The default `--min-expiry`, upstream's
/// `DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS` (30 minutes).
pub const DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS: i64 = 30 * 60_000;

/// The print commands, upstream's `CredentialPrintKind`
/// (`Exclude<AuthCommandKind, "check">`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialPrintKind {
    /// `auth print-api-key`, upstream's `"api_key"`.
    ApiKey,
    /// `auth print-bearer-token`, upstream's `"bearer_token"`.
    BearerToken,
}

impl From<CredentialPrintKind> for AuthCommandKind {
    fn from(kind: CredentialPrintKind) -> Self {
        match kind {
            CredentialPrintKind::ApiKey => Self::ApiKey,
            CredentialPrintKind::BearerToken => Self::BearerToken,
        }
    }
}

/// One provider resolved for the print, upstream's
/// `Array<{ id: string; model?: Model<Api> }>` entry.
struct ProviderTarget {
    id: String,
    model: Option<pi_ai::types::Model>,
}

/// Resolve one configured provider credential.
///
/// This intentionally calls `ModelRuntime`'s auth resolution, which
/// refreshes and persists OAuth credentials with less than five minutes
/// remaining through the normal request-auth path (upstream's
/// `ModelRuntime.getAuth()` note).
///
/// # Errors
/// An [`AuthCommandError`] carrying the upstream messages verbatim: an
/// unknown provider, an unresolvable model, the zero- and multi-match
/// outcomes, and the type-mismatch notices.
#[expect(
    clippy::too_many_lines,
    reason = "the body walks upstream's single resolveCredential stage by stage; the target matching, credential fetching, and message contracts share state a split would thread through parameters"
)]
pub async fn resolve_credential_for_print(
    args: &AuthCommandArgs,
    model_runtime: &ModelRuntime,
    kind: CredentialPrintKind,
    min_expiry_ms: Option<i64>,
    signal: Option<CancellationToken>,
) -> Result<String, AuthCommandError> {
    let command_kind = AuthCommandKind::from(kind);
    let target = validate_auth_command_args(args, command_kind)?;
    let credential_types: indexmap::IndexMap<String, AuthType> = model_runtime
        .list_credentials(Some(&AuthOptions {
            signal: signal.clone(),
        }))
        .await
        .map_err(|_| AuthCommandError("Unable to list the configured credentials".to_string()))?
        .into_iter()
        .map(|credential| (credential.provider_id, credential.auth_type))
        .collect();
    let mut providers: Vec<ProviderTarget> = Vec::new();
    if let Some(cli_provider) = target.provider.as_deref() {
        let Some(provider) = model_runtime.get_provider(cli_provider) else {
            return Err(AuthCommandError(format!(
                "Unknown provider \"{cli_provider}\". Use --list-models to see available providers."
            )));
        };
        if let Some(cli_model) = target.model.as_deref() {
            let resolved = resolve_cli_model(crate::model_resolver::ResolveCliModelOptions {
                cli_provider: Some(provider.id()),
                cli_model: Some(cli_model),
                cli_thinking: None,
                model_runtime,
            });
            if resolved.error.is_some() || resolved.model.is_none() {
                return Err(AuthCommandError(resolved.error.unwrap_or_else(|| {
                    "Unable to resolve the requested provider/model".to_string()
                })));
            }
            providers.push(ProviderTarget {
                id: provider.id().to_string(),
                model: resolved.model,
            });
        } else {
            providers.push(ProviderTarget {
                id: provider.id().to_string(),
                model: None,
            });
        }
    } else {
        let cli_model = target.model.as_deref().unwrap_or_default();
        for provider in model_runtime.get_providers() {
            if !credential_types.contains_key(provider.id()) {
                continue;
            }
            let resolved = resolve_cli_model(crate::model_resolver::ResolveCliModelOptions {
                cli_provider: Some(provider.id()),
                cli_model: Some(cli_model),
                cli_thinking: None,
                model_runtime,
            });
            if resolved.model.is_some()
                && resolved.error.is_none()
                && !resolved
                    .warning
                    .as_ref()
                    .is_some_and(|warning| warning.contains("Using custom model id"))
            {
                providers.push(ProviderTarget {
                    id: provider.id().to_string(),
                    model: resolved.model,
                });
            }
        }
        if providers.is_empty() {
            return Err(AuthCommandError(format!(
                "Model \"{cli_model}\" not found. Use --list-models to see available models."
            )));
        }
    }

    let mut credentials: Vec<(String, String)> = Vec::new();
    for provider in &providers {
        // A provider with no stored credential entry still resolves through
        // the request-auth path, upstream's `credentialTypes.get` miss that
        // falls through to `getAuth` (registration, override, and env keys
        // live outside the store's list).
        let credential_type = credential_types.get(&provider.id).copied();
        if kind == CredentialPrintKind::ApiKey && credential_type == Some(AuthType::OAuth) {
            continue;
        }
        if kind == CredentialPrintKind::BearerToken
            && credential_type.is_none_or(|credential_type| credential_type != AuthType::OAuth)
        {
            continue;
        }
        let overrides = ModelRuntimeAuthOverrides {
            min_oauth_validity_ms: match kind {
                CredentialPrintKind::BearerToken => {
                    Some(min_expiry_ms.unwrap_or(DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS))
                }
                CredentialPrintKind::ApiKey => None,
            },
            signal: signal.clone(),
            ..ModelRuntimeAuthOverrides::default()
        };
        let auth = match &provider.model {
            Some(model) => {
                model_runtime
                    .get_auth_for_model(model, Some(&overrides))
                    .await
            }
            None => model_runtime.get_auth(&provider.id, Some(&overrides)).await,
        };
        // A failed auth resolution propagates, upstream's unhandled promise
        // rejection reaching the command's error printer.
        let value = match auth {
            Ok(auth) => get_auth_credential(auth.as_ref()),
            Err(error) => return Err(AuthCommandError(error.to_string())),
        };
        if let Some(value) = value {
            credentials.push((provider.id.clone(), value));
        }
    }

    if credentials.len() == 1 {
        return Ok(credentials.remove(0).1);
    }
    if credentials.is_empty() {
        let provider_id = providers.first().map(|provider| provider.id.as_str());
        let credential_type = provider_id.and_then(|id| credential_types.get(id).copied());
        if target.provider.is_some()
            && kind == CredentialPrintKind::ApiKey
            && credential_type == Some(AuthType::OAuth)
        {
            return Err(AuthCommandError(format!(
                "Provider \"{}\" is configured with OAuth, not an API key",
                provider_id.unwrap_or_default()
            )));
        }
        if target.provider.is_some()
            && kind == CredentialPrintKind::BearerToken
            && credential_type.is_some_and(|credential_type| credential_type != AuthType::OAuth)
        {
            return Err(AuthCommandError(format!(
                "Provider \"{}\" is not configured with an OAuth bearer token",
                provider_id.unwrap_or_default()
            )));
        }
        let what = match kind {
            CredentialPrintKind::ApiKey => "API key",
            CredentialPrintKind::BearerToken => "OAuth bearer token",
        };
        return Err(AuthCommandError(format!("No usable {what} is configured")));
    }
    let ids = credentials
        .iter()
        .map(|(provider_id, _)| provider_id.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    Err(AuthCommandError(format!(
        "Multiple configured providers matched ({ids}). Specify --provider."
    )))
}

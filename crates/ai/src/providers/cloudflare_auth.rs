//! The Cloudflare auth helpers, ported from
//! `packages/ai/src/providers/cloudflare-auth.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::types::{
    ApiKeyAuth, ApiKeyCredential, AuthContext, AuthError, AuthPrompt, AuthPromptKind, AuthResult,
    ModelAuth, ProviderAuthInteraction,
};
use crate::types::{BoxedFuture, ProviderEnv};
use crate::utils::abort::AbortError;

const CLOUDFLARE_API_KEY: &str = "CLOUDFLARE_API_KEY";
const CLOUDFLARE_ACCOUNT_ID: &str = "CLOUDFLARE_ACCOUNT_ID";
const CLOUDFLARE_GATEWAY_ID: &str = "CLOUDFLARE_GATEWAY_ID";

/// Which Cloudflare surface the auth resolves for, upstream's
/// `CloudflareAuthKind`.
enum CloudflareAuthKind {
    /// Workers AI direct inference.
    WorkersAi,
    /// AI Gateway routing.
    AiGateway,
}

/// Per-field merge, upstream's `resolveValue`: prefer the credential value,
/// fall back to ambient env. A credential carrying only the API key must
/// still pick up the account/gateway id from the environment.
fn resolve_value(
    name: &str,
    ctx: &Arc<dyn AuthContext>,
    credential: Option<&ApiKeyCredential>,
    signal: &tokio_util::sync::CancellationToken,
) -> Result<Option<String>, AuthError> {
    let from_credential = credential.and_then(|credential| {
        if name == CLOUDFLARE_API_KEY {
            credential.key.clone()
        } else {
            credential
                .env
                .as_ref()
                .and_then(|env| env.get(name).cloned())
        }
    });
    if let Some(value) = from_credential {
        return Ok(Some(value));
    }
    if signal.is_cancelled() {
        return Err(AbortError.into());
    }
    let value = ctx.env(name);
    if signal.is_cancelled() {
        return Err(AbortError.into());
    }
    Ok(value)
}

/// Resolve the Cloudflare env, upstream's `resolveCloudflareEnv`: the api
/// key plus the account id, and the gateway id for AI Gateway only.
fn resolve_cloudflare_env(
    kind: &CloudflareAuthKind,
    ctx: &Arc<dyn AuthContext>,
    credential: Option<&ApiKeyCredential>,
    signal: &tokio_util::sync::CancellationToken,
) -> Result<Option<(String, ProviderEnv)>, AuthError> {
    let api_key = resolve_value(CLOUDFLARE_API_KEY, ctx, credential, signal)?;
    let account_id = resolve_value(CLOUDFLARE_ACCOUNT_ID, ctx, credential, signal)?;
    let gateway_id = match kind {
        CloudflareAuthKind::AiGateway => {
            resolve_value(CLOUDFLARE_GATEWAY_ID, ctx, credential, signal)?
        }
        CloudflareAuthKind::WorkersAi => None,
    };

    let Some(api_key) = api_key else {
        return Ok(None);
    };
    let Some(account_id) = account_id else {
        return Ok(None);
    };
    if matches!(kind, CloudflareAuthKind::AiGateway) && gateway_id.is_none() {
        return Ok(None);
    }

    let mut env = ProviderEnv::new();
    env.insert(CLOUDFLARE_ACCOUNT_ID.to_owned(), account_id);
    if let Some(gateway_id) = gateway_id {
        env.insert(CLOUDFLARE_GATEWAY_ID.to_owned(), gateway_id);
    }
    Ok(Some((api_key, env)))
}

/// The source label, upstream's `credential ? "stored credential" : env`.
const fn source_label(has_credential: bool) -> &'static str {
    if has_credential {
        "stored credential"
    } else {
        CLOUDFLARE_API_KEY
    }
}

/// The api key + account id prompts both Cloudflare logins repeat, the
/// shared prompt sequence upstream's builders inline.
async fn prompt_key_and_account(
    interaction: &ProviderAuthInteraction,
) -> Result<(String, String), AuthError> {
    let key = (interaction.prompt)(AuthPrompt {
        signal: Some(interaction.signal.clone()),
        kind: AuthPromptKind::Secret {
            message: "Enter Cloudflare API key".to_owned(),
            placeholder: None,
        },
    })
    .await?;
    let account_id = (interaction.prompt)(AuthPrompt {
        signal: Some(interaction.signal.clone()),
        kind: AuthPromptKind::Text {
            message: "Enter Cloudflare account ID".to_owned(),
            placeholder: None,
        },
    })
    .await?;
    Ok((key, account_id))
}

/// The Workers AI auth, upstream's `cloudflareWorkersAIAuth`.
#[must_use]
pub fn cloudflare_workers_ai_auth() -> ApiKeyAuth {
    let login: crate::auth::types::ApiKeyLoginFn = Arc::new(
        |interaction: ProviderAuthInteraction| -> BoxedFuture<'static, Result<ApiKeyCredential, AuthError>> {
            Box::pin(async move {
                let (key, account_id) = prompt_key_and_account(&interaction).await?;
                Ok(ApiKeyCredential {
                    key: Some(key),
                    env: Some([("CLOUDFLARE_ACCOUNT_ID".to_owned(), account_id)].into()),
                })
            })
        },
    );
    let resolve: crate::auth::types::ApiKeyResolveFn = Arc::new(
        move |input: crate::auth::types::ApiKeyAuthInput| -> BoxedFuture<
            'static,
            Result<Option<AuthResult>, AuthError>,
        > {
            Box::pin(async move {
                let resolved = resolve_cloudflare_env(
                    &CloudflareAuthKind::WorkersAi,
                    &input.ctx,
                    input.credential.as_ref(),
                    &input.signal,
                )?;
                let Some((api_key, env)) = resolved else {
                    return Ok(None);
                };
                Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some(api_key),
                        ..ModelAuth::default()
                    },
                    env: Some(env),
                    source: Some(source_label(input.credential.is_some()).to_owned()),
                }))
            })
        },
    );
    ApiKeyAuth {
        name: "Cloudflare API key".to_owned(),
        login: Some(login),
        check: None,
        resolve,
    }
}

/// The AI Gateway auth, upstream's `cloudflareAIGatewayAuth`: the gateway
/// key rides as `cf-aig-authorization` while the standard auth headers are
/// suppressed.
#[must_use]
pub fn cloudflare_ai_gateway_auth() -> ApiKeyAuth {
    let login: crate::auth::types::ApiKeyLoginFn = Arc::new(
        |interaction: ProviderAuthInteraction| -> BoxedFuture<'static, Result<ApiKeyCredential, AuthError>> {
            Box::pin(async move {
                let (key, account_id) = prompt_key_and_account(&interaction).await?;
                let gateway_id = (interaction.prompt)(AuthPrompt {
                    signal: Some(interaction.signal.clone()),
                    kind: AuthPromptKind::Text {
                        message: "Enter Cloudflare AI Gateway ID".to_owned(),
                        placeholder: None,
                    },
                })
                .await?;
                Ok(ApiKeyCredential {
                    key: Some(key),
                    env: Some(
                        [
                            (CLOUDFLARE_ACCOUNT_ID.to_owned(), account_id),
                            (CLOUDFLARE_GATEWAY_ID.to_owned(), gateway_id),
                        ]
                        .into(),
                    ),
                })
            })
        },
    );
    let resolve: crate::auth::types::ApiKeyResolveFn = Arc::new(
        move |input: crate::auth::types::ApiKeyAuthInput| -> BoxedFuture<
            'static,
            Result<Option<AuthResult>, AuthError>,
        > {
            Box::pin(async move {
                let resolved = resolve_cloudflare_env(
                    &CloudflareAuthKind::AiGateway,
                    &input.ctx,
                    input.credential.as_ref(),
                    &input.signal,
                )?;
                let Some((api_key, env)) = resolved else {
                    return Ok(None);
                };
                Ok(Some(AuthResult {
                    auth: ModelAuth {
                        headers: Some(
                            [
                                (
                                    "cf-aig-authorization".to_owned(),
                                    Some(format!("Bearer {api_key}")),
                                ),
                                ("Authorization".to_owned(), None),
                                ("x-api-key".to_owned(), None),
                            ]
                            .into_iter()
                            .collect(),
                        ),
                        ..ModelAuth::default()
                    },
                    env: Some(env),
                    source: Some(source_label(input.credential.is_some()).to_owned()),
                }))
            })
        },
    );
    ApiKeyAuth {
        name: "Cloudflare API key".to_owned(),
        login: Some(login),
        check: None,
        resolve,
    }
}

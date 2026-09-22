//! The Anthropic provider factory, ported from
//! `packages/ai/src/providers/anthropic.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::helpers::lazy_oauth;
use crate::auth::oauth::load_anthropic_oauth;
use crate::auth::types::AuthResult;
use crate::auth::types::{
    ApiKeyAuth, AuthPrompt, AuthPromptKind, ModelAuth, ProviderAuth, ProviderAuthInteraction,
};
use crate::env_api_keys::{
    ANTHROPIC_API_KEY_ENV, ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_OAUTH_TOKEN_ENV,
};
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;
use crate::utils::abort::AbortError;

/// The Anthropic api-key auth, upstream's `anthropicApiKeyAuth`: the stored
/// credential wins, then `ANTHROPIC_AUTH_TOKEN` (sent as a bearer header),
/// then the OAuth-token and API-key env vars.
fn anthropic_api_key_auth() -> ApiKeyAuth {
    ApiKeyAuth {
        name: "Anthropic API key".to_owned(),
        login: Some(Arc::new(|interaction: ProviderAuthInteraction| {
            Box::pin(async move {
                if interaction.signal.is_cancelled() {
                    return Err(AbortError.into());
                }
                let key = (interaction.prompt)(AuthPrompt {
                    signal: Some(interaction.signal.clone()),
                    kind: AuthPromptKind::Secret {
                        message: "Enter Anthropic API key".to_owned(),
                        placeholder: None,
                    },
                })
                .await?;
                if interaction.signal.is_cancelled() {
                    return Err(AbortError.into());
                }
                Ok(crate::auth::types::ApiKeyCredential {
                    key: Some(key),
                    env: None,
                })
            })
        })),
        check: None,
        resolve: Arc::new(|input| {
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

                let auth_token = input.ctx.env(ANTHROPIC_AUTH_TOKEN_ENV);
                if input.signal.is_cancelled() {
                    return Err(AbortError.into());
                }
                if let Some(auth_token) = auth_token {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth {
                            headers: Some(
                                std::iter::once((
                                    "Authorization".to_owned(),
                                    Some(format!("Bearer {auth_token}")),
                                ))
                                .collect(),
                            ),
                            ..ModelAuth::default()
                        },
                        env: None,
                        source: Some(ANTHROPIC_AUTH_TOKEN_ENV.to_owned()),
                    }));
                }

                for env_var in [ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV] {
                    let api_key = input.ctx.env(env_var);
                    if input.signal.is_cancelled() {
                        return Err(AbortError.into());
                    }
                    if let Some(api_key) = api_key {
                        return Ok(Some(AuthResult {
                            auth: ModelAuth {
                                api_key: Some(api_key),
                                ..ModelAuth::default()
                            },
                            env: None,
                            source: Some(env_var.to_owned()),
                        }));
                    }
                }
                Ok(None)
            })
        }),
    }
}

/// The Anthropic provider, upstream's `anthropicProvider()`.
#[must_use]
pub fn anthropic_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "anthropic".to_owned(),
        name: Some("Anthropic".to_owned()),
        base_url: Some("https://api.anthropic.com".to_owned()),
        auth: ProviderAuth {
            api_key: Some(anthropic_api_key_auth()),
            oauth: Some(lazy_oauth(crate::auth::helpers::LazyOAuthInput {
                name: "Anthropic (Claude Pro/Max)".to_owned(),
                is_subscription: Some(true),
                login_label: None,
                load: Arc::new(|| load_anthropic_oauth()),
            })),
        },
        models: get_builtin_models("anthropic"),
        api: ProviderApi::Single(crate::api::anthropic_messages()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

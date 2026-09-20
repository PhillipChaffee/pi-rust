//! The Amazon Bedrock provider factory, ported from
//! `packages/ai/src/providers/amazon-bedrock.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::types::{
    ApiKeyAuth, ApiKeyLoginFn, ApiKeyResolveFn, AuthPrompt, AuthPromptKind, AuthPromptOption,
    AuthResult, ModelAuth, ProviderAuth, ProviderAuthInteraction,
};
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;
use crate::types::BoxedFuture;
use crate::utils::abort::AbortError;

#[expect(
    clippy::too_many_lines,
    reason = "the login/resolve branches mirror upstream's object literal one to one"
)]
/// Bedrock accepts a bearer token or the AWS SDK's default credential chain,
/// upstream's `bedrockAuth`. The login flow can store a token/profile
/// choice; resolve also detects ambient AWS credentials without copying them
/// into pi's credential store.
fn bedrock_auth() -> ApiKeyAuth {
    let login: ApiKeyLoginFn = Arc::new(|interaction: ProviderAuthInteraction| {
        Box::pin(async move {
            if interaction.signal.is_cancelled() {
                return Err(AbortError.into());
            }
            let method = (interaction.prompt)(AuthPrompt {
                signal: Some(interaction.signal.clone()),
                kind: AuthPromptKind::Select {
                    message: "Select Amazon Bedrock authentication method:".to_owned(),
                    options: vec![
                        AuthPromptOption {
                            id: "bearer-token".to_owned(),
                            label: "Bearer token".to_owned(),
                            description: None,
                        },
                        AuthPromptOption {
                            id: "aws-profile".to_owned(),
                            label: "AWS profile".to_owned(),
                            description: None,
                        },
                        AuthPromptOption {
                            id: "credential-chain".to_owned(),
                            label: "Existing AWS credential chain".to_owned(),
                            description: None,
                        },
                    ],
                },
            })
            .await?;
            if interaction.signal.is_cancelled() {
                return Err(AbortError.into());
            }
            if method == "bearer-token" {
                let token = (interaction.prompt)(AuthPrompt {
                    signal: Some(interaction.signal.clone()),
                    kind: AuthPromptKind::Secret {
                        message: "Enter Amazon Bedrock bearer token".to_owned(),
                        placeholder: None,
                    },
                })
                .await?;
                return Ok(crate::auth::types::ApiKeyCredential {
                    key: Some(token),
                    env: None,
                });
            }
            (interaction.notify)(crate::auth::types::AuthEvent::Info {
                message:
                    "Amazon Bedrock supports AWS profiles, IAM credentials, and role-based credentials."
                        .to_owned(),
                links: Some(vec![crate::auth::types::AuthInfoLink {
                    label: Some("AWS credential provider chain".to_owned()),
                    url: "https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html"
                        .to_owned(),
                }]),
            });
            if method == "aws-profile" {
                let profile = (interaction.prompt)(AuthPrompt {
                    signal: Some(interaction.signal.clone()),
                    kind: AuthPromptKind::Text {
                        message: "Enter AWS profile name".to_owned(),
                        placeholder: None,
                    },
                })
                .await?;
                return Ok(crate::auth::types::ApiKeyCredential {
                    key: None,
                    env: Some([("AWS_PROFILE".to_owned(), profile)].into()),
                });
            }
            if method != "credential-chain" {
                return Err(crate::auth::types::AuthError::from(format!(
                    "Unknown Amazon Bedrock auth method: {method}"
                )));
            }
            (interaction.prompt)(AuthPrompt {
                signal: Some(interaction.signal.clone()),
                kind: AuthPromptKind::Text {
                    message: "Configure AWS credentials, then press Enter to continue".to_owned(),
                    placeholder: None,
                },
            })
            .await?;
            Ok(crate::auth::types::ApiKeyCredential::default())
        })
    });
    let resolve: ApiKeyResolveFn = Arc::new(
        move |input: crate::auth::types::ApiKeyAuthInput| -> BoxedFuture<
            'static,
            Result<Option<AuthResult>, crate::auth::types::AuthError>,
        > {
            Box::pin(async move {
                let env_value = |name: &str| input.ctx.env(name);
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
                if env_value("AWS_BEARER_TOKEN_BEDROCK").is_some() {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth::default(),
                        env: None,
                        source: Some("AWS_BEARER_TOKEN_BEDROCK".to_owned()),
                    }));
                }
                let stored_profile = input
                    .credential
                    .as_ref()
                    .and_then(|credential| credential.env.as_ref())
                    .and_then(|env| env.get("AWS_PROFILE").cloned());
                let has_profile = stored_profile.is_some() || env_value("AWS_PROFILE").is_some();
                if has_profile {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth::default(),
                        env: input
                            .credential
                            .as_ref()
                            .and_then(|credential| credential.env.clone()),
                        source: Some(
                            if input
                                .credential
                                .as_ref()
                                .and_then(|credential| credential.env.as_ref())
                                .is_some_and(|env| env.contains_key("AWS_PROFILE"))
                            {
                                "stored credential"
                            } else {
                                "AWS_PROFILE"
                            }
                            .to_owned(),
                        ),
                    }));
                }
                if env_value("AWS_ACCESS_KEY_ID").is_some()
                    && env_value("AWS_SECRET_ACCESS_KEY").is_some()
                {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth::default(),
                        env: None,
                        source: Some("AWS access keys".to_owned()),
                    }));
                }
                if env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").is_some() {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth::default(),
                        env: None,
                        source: Some("ECS task role".to_owned()),
                    }));
                }
                if env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI").is_some() {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth::default(),
                        env: None,
                        source: Some("ECS task role".to_owned()),
                    }));
                }
                if env_value("AWS_WEB_IDENTITY_TOKEN_FILE").is_some() {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth::default(),
                        env: None,
                        source: Some("web identity token".to_owned()),
                    }));
                }
                Ok(None)
            })
        },
    );
    ApiKeyAuth {
        name: "AWS credentials or bearer token".to_owned(),
        login: Some(login),
        check: None,
        resolve,
    }
}

/// The Amazon Bedrock provider, upstream's `amazonBedrockProvider()`.
#[must_use]
pub fn amazon_bedrock_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "amazon-bedrock".to_owned(),
        name: Some("Amazon Bedrock".to_owned()),
        base_url: None,
        auth: ProviderAuth {
            api_key: Some(bedrock_auth()),
            oauth: None,
        },
        models: get_builtin_models("amazon-bedrock"),
        api: ProviderApi::Single(crate::api::bedrock_converse_stream()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

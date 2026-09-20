//! The Google Vertex AI provider factory, ported from
//! `packages/ai/src/providers/google-vertex.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;

use crate::auth::types::{
    ApiKeyAuth, ApiKeyLoginFn, ApiKeyResolveFn, AuthError, AuthPrompt, AuthPromptKind,
    AuthPromptOption, AuthResult, ModelAuth, ProviderAuth, ProviderAuthInteraction,
};
use crate::models::{CreateProviderOptions, Provider, ProviderApi, create_provider};
use crate::providers::catalog::get_builtin_models;
use crate::types::BoxedFuture;
use crate::utils::abort::AbortError;

/// The ADC path Vertex falls back to, upstream's `VERTEX_ADC_PATH`.
const VERTEX_ADC_PATH: &str = "~/.config/gcloud/application_default_credentials.json";

#[expect(
    clippy::too_many_lines,
    reason = "the login/resolve branches mirror upstream's object literal one to one"
)]
/// Vertex accepts an explicit API key or Application Default Credentials
/// (`gcloud auth application-default login`), upstream's `vertexAuth`. ADC
/// additionally requires project and location env vars, which the
/// implementation reads itself.
fn vertex_auth() -> ApiKeyAuth {
    let login: ApiKeyLoginFn = Arc::new(|interaction: ProviderAuthInteraction| {
        Box::pin(async move {
            if interaction.signal.is_cancelled() {
                return Err(AbortError.into());
            }
            let method = (interaction.prompt)(AuthPrompt {
                signal: Some(interaction.signal.clone()),
                kind: AuthPromptKind::Select {
                    message: "Select Google Vertex AI authentication method:".to_owned(),
                    options: vec![
                        AuthPromptOption {
                            id: "api-key".to_owned(),
                            label: "Google Cloud API key".to_owned(),
                            description: None,
                        },
                        AuthPromptOption {
                            id: "adc".to_owned(),
                            label: "Application Default Credentials".to_owned(),
                            description: None,
                        },
                        AuthPromptOption {
                            id: "service-account".to_owned(),
                            label: "Service account credentials file".to_owned(),
                            description: None,
                        },
                    ],
                },
            })
            .await?;
            if interaction.signal.is_cancelled() {
                return Err(AbortError.into());
            }
            if method == "api-key" {
                let key = (interaction.prompt)(AuthPrompt {
                    signal: Some(interaction.signal.clone()),
                    kind: AuthPromptKind::Secret {
                        message: "Enter Google Cloud API key".to_owned(),
                        placeholder: None,
                    },
                })
                .await?;
                return Ok(crate::auth::types::ApiKeyCredential {
                    key: Some(key),
                    env: None,
                });
            }
            if method != "adc" && method != "service-account" {
                return Err(AuthError::from(format!(
                    "Unknown Google Vertex AI auth method: {method}"
                )));
            }
            (interaction.notify)(crate::auth::types::AuthEvent::Info {
                message: if method == "adc" {
                    "Run `gcloud auth application-default login`, then provide the project and location."
                        .to_owned()
                } else {
                    "Provide a service account credentials file, project, and location.".to_owned()
                },
                links: Some(vec![crate::auth::types::AuthInfoLink {
                    label: Some("Application Default Credentials".to_owned()),
                    url: "https://cloud.google.com/docs/authentication/provide-credentials-adc"
                        .to_owned(),
                }]),
            });
            let project = (interaction.prompt)(AuthPrompt {
                signal: Some(interaction.signal.clone()),
                kind: AuthPromptKind::Text {
                    message: "Enter Google Cloud project ID".to_owned(),
                    placeholder: None,
                },
            })
            .await?;
            let location = (interaction.prompt)(AuthPrompt {
                signal: Some(interaction.signal.clone()),
                kind: AuthPromptKind::Text {
                    message: "Enter Google Cloud location".to_owned(),
                    placeholder: None,
                },
            })
            .await?;
            let credentials_path = if method == "service-account" {
                Some(
                    (interaction.prompt)(AuthPrompt {
                        signal: Some(interaction.signal.clone()),
                        kind: AuthPromptKind::Text {
                            message: "Enter service account credentials file path".to_owned(),
                            placeholder: None,
                        },
                    })
                    .await?,
                )
            } else {
                None
            };
            let mut env = [
                ("GOOGLE_CLOUD_PROJECT".to_owned(), project),
                ("GOOGLE_CLOUD_LOCATION".to_owned(), location),
            ]
            .into_iter()
            .collect::<std::collections::BTreeMap<String, String>>();
            if let Some(credentials_path) = credentials_path {
                env.insert(
                    "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
                    credentials_path,
                );
            }
            Ok(crate::auth::types::ApiKeyCredential {
                key: None,
                env: Some(env),
            })
        })
    });
    let resolve: ApiKeyResolveFn = Arc::new(
        move |input: crate::auth::types::ApiKeyAuthInput| -> BoxedFuture<
            'static,
            Result<Option<AuthResult>, AuthError>,
        > {
            Box::pin(async move {
                let env_value = |name: &str| input.ctx.env(name);
                if input.signal.is_cancelled() {
                    return Err(AbortError.into());
                }
                let key = input
                    .credential
                    .as_ref()
                    .and_then(|credential| credential.key.clone())
                    .or_else(|| env_value("GOOGLE_CLOUD_API_KEY"));
                if let Some(key) = key {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth {
                            api_key: Some(key),
                            ..ModelAuth::default()
                        },
                        env: None,
                        source: Some(if input.credential.is_some() {
                            "stored credential"
                        } else {
                            "GOOGLE_CLOUD_API_KEY"
                        }.to_owned()),
                    }));
                }

                let adc_path = input
                    .credential
                    .as_ref()
                    .and_then(|credential| credential.env.as_ref())
                    .and_then(|env| env.get("GOOGLE_APPLICATION_CREDENTIALS").cloned())
                    .or_else(|| env_value("GOOGLE_APPLICATION_CREDENTIALS"))
                    .unwrap_or_else(|| VERTEX_ADC_PATH.to_owned());
                if input.signal.is_cancelled() {
                    return Err(AbortError.into());
                }
                let has_credentials = input.ctx.file_exists(&adc_path);
                if input.signal.is_cancelled() {
                    return Err(AbortError.into());
                }
                let project = input
                    .credential
                    .as_ref()
                    .and_then(|credential| credential.env.as_ref())
                    .and_then(|env| env.get("GOOGLE_CLOUD_PROJECT").cloned())
                    .or_else(|| env_value("GOOGLE_CLOUD_PROJECT"))
                    .or_else(|| env_value("GCLOUD_PROJECT"));
                let location = input
                    .credential
                    .as_ref()
                    .and_then(|credential| credential.env.as_ref())
                    .and_then(|env| env.get("GOOGLE_CLOUD_LOCATION").cloned())
                    .or_else(|| env_value("GOOGLE_CLOUD_LOCATION"));
                if has_credentials
                    && project.is_some()
                    && location.is_some()
                {
                    return Ok(Some(AuthResult {
                        auth: ModelAuth::default(),
                        env: input
                            .credential
                            .as_ref()
                            .and_then(|credential| credential.env.clone()),
                        source: Some(if input.credential.is_some() {
                            "stored credential"
                        } else {
                            "gcloud application default credentials"
                        }
                        .to_owned()),
                    }));
                }
                Ok(None)
            })
        },
    );
    ApiKeyAuth {
        name: "Google Cloud credentials".to_owned(),
        login: Some(login),
        check: None,
        resolve,
    }
}

/// The Google Vertex AI provider, upstream's `googleVertexProvider()`.
#[must_use]
pub fn google_vertex_provider() -> Arc<dyn Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: "google-vertex".to_owned(),
        name: Some("Google Vertex AI".to_owned()),
        base_url: None,
        auth: ProviderAuth {
            api_key: Some(vertex_auth()),
            oauth: None,
        },
        models: get_builtin_models("google-vertex"),
        api: ProviderApi::Single(crate::api::google_vertex()),
        headers: None,
        fetch_models: None,
        filter_models: None,
    }))
}

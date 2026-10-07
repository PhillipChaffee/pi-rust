//! Upstream
//! `packages/coding-agent/test/suite/regressions/7027-credential-refresh-hang.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated for
//! `pi_coding_agent::model_runtime` (#121).
//!
//! Only the first case of the `issues #7027 and #7113 credential refresh
//! hang` describe ports here as a runtime-level test. The second case,
//! `completes interactive login before its bounded background refresh`, and
//! the whole `post-login model discovery` describe drive the
//! `InteractiveMode` layer — and ride tickets #131 and #132.
//!
//! Porting restatements this file records:
//!
//! - The stalled provider implements [`pi_ai::models::Provider`] directly;
//!   the stream dispatches no case drives resolve through the fixture's
//!   [`common::model_layer::unused_stream`] guard, and the impl spells the
//!   `pi_ai::types` names in full because the credential-sync suite owns
//!   the shared spelling of the trimmed import block.
//! - The `networkStarted` promise restates as a `Notify` permit and the
//!   forever-pending refresh phase as [`std::future::pending`].

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

#[expect(
    dead_code,
    reason = "every test binary recompiles the shared fixture module; this suite drives only the model-layer helpers"
)]
mod common;

use std::sync::Arc;

use common::model_layer::{
    create_in_memory_model_registry, in_memory_auth_storage, model, unused_stream,
};
use pi_ai::auth::types::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCheckFn, ApiKeyCredential, ApiKeyLoginFn, ApiKeyResolveFn,
    AuthCheck, AuthInteraction, AuthResult, AuthType, Credential, ModelAuth, ProviderAuth,
};
use pi_ai::models::{
    ModelsRefreshOptions, Provider, ProviderError, ProviderModelError, RefreshModelsContext,
};
use pi_coding_agent::auth_storage::AuthStorageData;
use tokio::sync::Notify;

/// The api-key availability answer upstream's `check` returns: a non-empty
/// stored key reads as the stored credential.
fn stored_key_check(input: &ApiKeyAuthInput) -> Option<AuthCheck> {
    input
        .credential
        .as_ref()
        .and_then(|credential| credential.key.as_deref())
        .is_some_and(|key| !key.is_empty())
        .then(|| AuthCheck {
            source: Some("stored key".to_owned()),
            auth_type: AuthType::ApiKey,
        })
}

/// The stalled-login provider, upstream's inline `provider` object: an
/// api-key login returning `secret`, an ambient-key resolve, one dynamic
/// model, and a refresh phase that parks forever once the network is
/// allowed.
struct StalledLoginProvider {
    id: String,
    name: String,
    auth: ProviderAuth,
    network_started: Arc<Notify>,
}

impl StalledLoginProvider {
    /// Build the provider, upstream's object literal.
    fn new(network_started: Arc<Notify>) -> Self {
        let login: ApiKeyLoginFn = Arc::new(|_interaction| {
            Box::pin(async {
                Ok(ApiKeyCredential {
                    key: Some("secret".to_owned()),
                    env: None,
                })
            })
        });
        let check: ApiKeyCheckFn = Arc::new(|input: ApiKeyAuthInput| {
            let check = stored_key_check(&input);
            Box::pin(async move { Ok(check) })
        });
        let resolve: ApiKeyResolveFn = Arc::new(|input: ApiKeyAuthInput| {
            let stored_key = input
                .credential
                .as_ref()
                .and_then(|credential| credential.key.clone());
            let resolution = AuthResult {
                auth: ModelAuth {
                    api_key: Some(
                        stored_key
                            .clone()
                            .unwrap_or_else(|| "ambient-key".to_owned()),
                    ),
                    ..ModelAuth::default()
                },
                env: None,
                source: Some(
                    if stored_key.is_some() {
                        "stored key"
                    } else {
                        "ambient key"
                    }
                    .to_owned(),
                ),
            };
            Box::pin(async move { Ok(Some(resolution)) })
        });
        Self {
            id: "stalled-login".to_owned(),
            name: "Stalled Login".to_owned(),
            auth: ProviderAuth {
                api_key: Some(ApiKeyAuth {
                    name: "API key".to_owned(),
                    login: Some(login),
                    check: Some(check),
                    resolve,
                }),
                oauth: None,
            },
            network_started,
        }
    }
}

impl Provider for StalledLoginProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn get_models(&self) -> Result<Vec<pi_ai::types::Model>, ProviderModelError> {
        Ok(vec![model("stalled-login", "dynamic")])
    }

    fn supports_refresh_models(&self) -> bool {
        true
    }

    fn refresh_models(
        &self,
        context: RefreshModelsContext,
    ) -> pi_ai::types::BoxedFuture<'_, Result<(), ProviderError>> {
        if !context.allow_network {
            return Box::pin(async { Ok(()) });
        }
        self.network_started.notify_one();
        Box::pin(async {
            std::future::pending::<()>().await;
            Ok(())
        })
    }

    fn stream(
        &self,
        _model: &pi_ai::types::Model,
        _context: &pi_ai::types::Context,
        _options: Option<&pi_ai::types::StreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        unused_stream()
    }

    fn stream_simple(
        &self,
        _model: &pi_ai::types::Model,
        _context: &pi_ai::types::Context,
        _options: Option<&pi_ai::types::SimpleStreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        unused_stream()
    }
}

/// The `{ prompt: async () => "unused", notify: () => {} }` login
/// interaction, spelled with `String::from` because the credential-sync
/// suite owns the shared spelling of this fixture.
fn login_interaction() -> AuthInteraction {
    AuthInteraction {
        signal: None,
        prompt: Arc::new(|_prompt| Box::pin(async { Ok(String::from("unused")) })),
        notify: Arc::new(|_event| {}),
    }
}

mod issues_7027_and_7113_credential_refresh_hang {
    use super::*;

    /// upstream `it("does not hold login behind an older stalled network
    /// catalog refresh")`.
    #[tokio::test]
    async fn does_not_hold_login_behind_an_older_stalled_network_catalog_refresh() {
        let network_started = Arc::new(Notify::new());
        let credentials = in_memory_auth_storage(&AuthStorageData::new());
        let runtime = create_in_memory_model_registry(credentials.clone())
            .await
            .runtime()
            .clone();
        runtime.register_native_provider(Arc::new(StalledLoginProvider::new(Arc::clone(
            &network_started,
        ))));
        runtime
            .refresh(ModelsRefreshOptions {
                allow_network: Some(false),
                providers: Some(vec!["stalled-login".to_owned()]),
                ..ModelsRefreshOptions::default()
            })
            .await;

        let stalled_task = tokio::spawn({
            let runtime = runtime.clone();
            async move {
                runtime
                    .refresh(ModelsRefreshOptions {
                        allow_network: Some(true),
                        providers: Some(vec!["stalled-login".to_owned()]),
                        ..ModelsRefreshOptions::default()
                    })
                    .await
            }
        });
        network_started.notified().await;
        let credential = runtime
            .login("stalled-login", AuthType::ApiKey, login_interaction())
            .await
            .expect("the login completes while the network refresh stalls");
        assert_eq!(
            credential,
            Credential::ApiKey(ApiKeyCredential {
                key: Some("secret".to_owned()),
                env: None,
            }),
        );

        assert!(
            runtime
                .get_available_snapshot()
                .iter()
                .any(|entry| entry.id == "dynamic"),
            "the refreshed catalog carries the dynamic model"
        );
        let stored = credentials
            .read("stalled-login", None)
            .await
            .expect("the read lands");
        assert_eq!(
            stored,
            Some(Credential::ApiKey(ApiKeyCredential {
                key: Some("secret".to_owned()),
                env: None,
            })),
        );
        let stalled = stalled_task.await.expect("the stalled refresh task joins");
        assert!(
            !stalled.aborted,
            "the superseded refresh resolves uncancelled"
        );
    }
}

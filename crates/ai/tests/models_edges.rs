//! The Models collection's remaining edges: the deferred dispatch's failure
//! paths, login aborts, logout store failures, the option Debug surfaces, and
//! the thinking-level clamp fallbacks. Upstream covers these through the
//! runtime and compat suites at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "test failures panic by design, mirroring expect!'s failure mode"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use common::{ambient_auth, deferred_handle, end_with, fixture_model};
use pi_ai::auth::types::{ApiKeyAuth, ApiKeyAuthInput, ProviderAuth};
use pi_ai::models::{
    CreateModelsOptions, CreateProviderOptions, ModelsDeferredCancelOptions, ProviderApi,
    create_models, create_provider,
};
use pi_ai::types::{
    BoxedFuture, Context, DeferredCancelOptions, DeferredFetchOptions, DeferredHandle, Model,
    ProviderStreams, SimpleStreamOptions, StopReason, StreamOptions,
};

mod common;

/// A credential store whose reads and deletes fail, upstream's inline
/// failing store objects; `list` succeeds with nothing to report.
struct FailingStore;

impl pi_ai::auth::credential_store::CredentialStore for FailingStore {
    fn read<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<
        'a,
        Result<Option<pi_ai::auth::types::Credential>, Box<dyn std::error::Error + Send + Sync>>,
    > {
        let cause: Box<dyn std::error::Error + Send + Sync> =
            Box::new(std::io::Error::other("disk on fire"));
        Box::pin(async move { Err(cause) })
    }

    fn list<'a>(
        &'a self,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<
        'a,
        Result<Vec<pi_ai::auth::types::CredentialInfo>, Box<dyn std::error::Error + Send + Sync>>,
    > {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn modify<'a>(
        &'a self,
        _provider_id: &'a str,
        _f: pi_ai::auth::types::CredentialModifyFn,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<
        'a,
        Result<Option<pi_ai::auth::types::Credential>, Box<dyn std::error::Error + Send + Sync>>,
    > {
        let cause: Box<dyn std::error::Error + Send + Sync> =
            Box::new(std::io::Error::other("disk on fire"));
        Box::pin(async move { Err(cause) })
    }

    fn delete<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        let cause: Box<dyn std::error::Error + Send + Sync> =
            Box::new(std::io::Error::other("disk on fire"));
        Box::pin(async move { Err(cause) })
    }
}

/// A provider whose `cancel_deferred` fails; the Models surface wraps the
/// failure, upstream's cancelDeferred contract.
struct FailingCancelStreams;

impl ProviderStreams for FailingCancelStreams {
    fn stream(
        &self,
        model: &Model,
        _context: &Context,
        _options: Option<&StreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        end_with(model)
    }

    fn stream_simple(
        &self,
        model: &Model,
        _context: &Context,
        _options: Option<&SimpleStreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        end_with(model)
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        _handle: &DeferredHandle,
        _options: Option<&DeferredFetchOptions>,
    ) -> Option<pi_ai::utils::event_stream::AssistantMessageEventStream> {
        Some(end_with(model))
    }

    fn cancel_deferred<'a>(
        &'a self,
        _model: &'a Model,
        _handle: &'a DeferredHandle,
        _options: Option<&'a DeferredCancelOptions>,
    ) -> BoxedFuture<'a, Result<(), pi_ai::utils::provider_retry::ProviderRequestError>> {
        Box::pin(async {
            Err(pi_ai::utils::provider_retry::ProviderRequestError::new(
                Some(500),
                None,
                "cancel rejected",
            ))
        })
    }

    fn supports_fetch_deferred(&self) -> bool {
        true
    }

    fn supports_cancel_deferred(&self) -> bool {
        true
    }
}

/// A Models collection whose single provider streams through `streams`, the
/// shape the deferred dispatch tests route through.
fn models_with_streams(
    auth: ProviderAuth,
    streams: Arc<dyn ProviderStreams>,
) -> pi_ai::models::Models {
    let models = create_models(None);
    models.set_provider(Arc::new(create_provider(CreateProviderOptions {
        id: "p".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth,
        models: vec![fixture_model()],
        api: ProviderApi::Single(streams),
        fetch_models: None,
        filter_models: None,
    })));
    models
}

/// The deferred dispatch when auth resolution fails: the lazy stream settles
/// with the not-configured notice, the port of upstream's applyAuth
/// rejection inside streamDeferred.
#[tokio::test]
async fn the_deferred_stream_reports_unconfigured_providers() {
    let models = models_with_streams(ProviderAuth::default(), Arc::new(FailingCancelStreams));

    let model = fixture_model();
    let handle = deferred_handle();
    let text = models
        .stream_deferred(&model, &handle, None)
        .result()
        .await
        .error_message
        .unwrap_or_default();
    assert!(
        text.contains("Provider is not configured: p"),
        "got: {text}"
    );
}

/// A provider cancel that fails wraps the failure, upstream's
/// `cancelDeferred` rejection path.
#[tokio::test]
async fn the_deferred_cancellation_wraps_provider_failures() {
    let models = models_with_streams(ambient_auth(), Arc::new(FailingCancelStreams));

    let model = fixture_model();
    let handle = deferred_handle();
    let error = models
        .cancel_deferred(
            &model,
            &handle,
            Some(&ModelsDeferredCancelOptions::default()),
        )
        .await
        .expect_err("cancel fails");
    assert!(
        error
            .to_string()
            .contains("Deferred cancellation failed for p"),
        "got: {error}"
    );
}

/// The complete surfaces resolve through their stream wrappers, and a
/// failing credential store surfaces through logout and login.
#[tokio::test]
async fn the_complete_and_auth_store_failure_paths_surface_their_messages() {
    let models = models_with_streams(ambient_auth(), Arc::new(FailingCancelStreams));

    let model = fixture_model();
    // complete/completeSimple route through the streams.
    let done = models.complete(&model, &Context::default(), None).await;
    assert_eq!(done.stop_reason, StopReason::Stop);
    let simple = models
        .complete_simple(&model, &Context::default(), None)
        .await;
    assert_eq!(simple.stop_reason, StopReason::Stop);

    // A failing credential store wraps logout and login failures.
    let failing: Arc<dyn pi_ai::auth::credential_store::CredentialStore> = Arc::new(FailingStore);
    let models = create_models(Some(CreateModelsOptions {
        credentials: Some(Arc::clone(&failing)),
        ..CreateModelsOptions::default()
    }));
    let error = models.logout("p", None).await.expect_err("logout fails");
    assert!(
        error.to_string().contains("Credential store delete failed"),
        "got: {error}"
    );
    let interaction = pi_ai::auth::types::AuthInteraction {
        signal: None,
        prompt: Arc::new(|_prompt: pi_ai::auth::types::AuthPrompt| {
            let entered: BoxedFuture<'static, Result<String, pi_ai::utils::abort::AbortError>> =
                Box::pin(async { Ok("key".to_owned()) });
            entered
        }),
        notify: Arc::new(|_event| {}),
    };
    models.set_provider(Arc::new(create_provider(CreateProviderOptions {
        id: "with-login".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ProviderAuth {
            api_key: Some(ApiKeyAuth {
                name: "Login key".to_owned(),
                login: Some(Arc::new(|_interaction| {
                    let credential: BoxedFuture<
                        'static,
                        Result<
                            pi_ai::auth::types::ApiKeyCredential,
                            Box<dyn std::error::Error + Send + Sync>,
                        >,
                    > = Box::pin(async move {
                        Ok(pi_ai::auth::types::ApiKeyCredential {
                            key: Some("k".to_owned()),
                            env: None,
                        })
                    });
                    credential
                })),
                check: None,
                resolve: Arc::new(|_input: ApiKeyAuthInput| {
                    Box::pin(async move { Ok(Some(pi_ai::auth::types::AuthResult::default())) })
                }),
            }),
            oauth: None,
        },
        models: vec![],
        api: ProviderApi::Single(pi_ai::api::not_ported_streams("test-api")),
        fetch_models: None,
        filter_models: None,
    })));
    let error = models
        .login(
            "with-login",
            pi_ai::auth::types::AuthType::ApiKey,
            interaction,
        )
        .await
        .expect_err("store modify fails");
    assert!(
        error.to_string().contains("Credential store modify failed"),
        "got: {error}"
    );
}

/// The option-shape conversions and Debug impls render their shapes.
#[test]
fn the_provider_option_shapes_render() {
    let streams: Arc<dyn ProviderStreams> = pi_ai::api::not_ported_streams("test-api");
    let single: ProviderApi = streams.into();
    assert!(format!("{single:?}").contains("Single"));
    let by_api: ProviderApi = BTreeMap::new().into();
    assert!(format!("{by_api:?}").contains("ByApi"));

    let options = create_provider(CreateProviderOptions {
        id: "p".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ProviderAuth::default(),
        models: vec![fixture_model()],
        api: ProviderApi::Single(pi_ai::api::not_ported_streams("test-api")),
        fetch_models: None,
        filter_models: None,
    });
    let rendered = format!("{options:?}");
    assert!(rendered.contains("ProviderImpl"), "got: {rendered}");
    let options = CreateProviderOptions {
        id: "p".to_owned(),
        name: Some("Named".to_owned()),
        base_url: Some("https://x.test".to_owned()),
        headers: None,
        auth: ProviderAuth::default(),
        models: vec![fixture_model()],
        api: ProviderApi::Single(pi_ai::api::not_ported_streams("test-api")),
        fetch_models: None,
        filter_models: None,
    };
    assert!(format!("{options:?}").contains("Named"));
}

/// The thinking-level clamp's fallbacks: requesting below every supported
/// level falls back to the first available.
#[test]
fn the_thinking_clamp_falls_back_to_the_first_available_level() {
    let mut model = fixture_model();
    model.reasoning = true;
    model.thinking_level_map = Some(BTreeMap::from([(
        pi_ai::types::ModelThinkingLevel::High,
        Some("high".to_owned()),
    )]));
    // Only high (plus the default-supported levels) is mapped; requesting max
    // falls forward, and the fallback chain lands on the first available.
    assert_eq!(
        pi_ai::models::clamp_thinking_level(&model, pi_ai::types::ModelThinkingLevel::Max),
        pi_ai::types::ModelThinkingLevel::High
    );
}

/// A login whose flow fails wraps the failure, upstream's race rejection.
#[tokio::test]
async fn the_login_failure_wraps_the_underlying_error() {
    let models = create_models(None);
    models.set_provider(Arc::new(create_provider(CreateProviderOptions {
        id: "p".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ProviderAuth {
            api_key: Some(ApiKeyAuth {
                name: "Failing".to_owned(),
                login: Some(Arc::new(|_interaction| {
                    let failing: BoxedFuture<
                        'static,
                        Result<
                            pi_ai::auth::types::ApiKeyCredential,
                            Box<dyn std::error::Error + Send + Sync>,
                        >,
                    > = {
                        let cause: Box<dyn std::error::Error + Send + Sync> =
                            Box::new(std::io::Error::other("flow rejected"));
                        Box::pin(async move { Err(cause) })
                    };
                    failing
                })),
                check: None,
                resolve: Arc::new(|_input: ApiKeyAuthInput| {
                    Box::pin(async move { Ok(Some(pi_ai::auth::types::AuthResult::default())) })
                }),
            }),
            oauth: None,
        },
        models: vec![],
        api: ProviderApi::Single(pi_ai::api::not_ported_streams("test-api")),
        fetch_models: None,
        filter_models: None,
    })));
    let interaction = pi_ai::auth::types::AuthInteraction {
        signal: None,
        prompt: Arc::new(|_prompt: pi_ai::auth::types::AuthPrompt| {
            let entered: BoxedFuture<'static, Result<String, pi_ai::utils::abort::AbortError>> =
                Box::pin(async { Ok("key".to_owned()) });
            entered
        }),
        notify: Arc::new(|_event| {}),
    };
    let outcome = models
        .login("p", pi_ai::auth::types::AuthType::ApiKey, interaction)
        .await;
    let Err(pi_ai::auth::resolve::ModelsFailure::Models(wrapped)) = outcome else {
        panic!("the failed login wraps its failure");
    };
    assert_eq!(wrapped.code(), pi_ai::auth::resolve::ModelsErrorCode::Auth);
    assert!(
        wrapped
            .to_string()
            .contains("Login failed for p: flow rejected")
    );
}

/// The models state debug renders through the collection clone.
#[test]
fn the_models_state_debug_renders() {
    let models = create_models(None);
    let rendered = format!("{models:?}");
    assert!(rendered.contains("Models"), "got: {rendered}");
}

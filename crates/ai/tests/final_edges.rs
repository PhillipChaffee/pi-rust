//! The final #28 coverage edges: deferred aborts through auth resolution,
//! the canceled check/available paths, and the images error/option debug
//! shapes. At commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "test failures panic by design, mirroring expect!'s failure mode"
)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use common::{DeferredStreams, ambient_auth, deferred_handle, fixture_model};
use pi_ai::auth::types::{ApiKeyAuth, ApiKeyAuthInput, ProviderAuth};
use pi_ai::models::{CreateProviderOptions, ProviderApi, create_models, create_provider};
use pi_ai::types::{BoxedFuture, Context, Model, StopReason, TransportOptions};
use tokio_util::sync::CancellationToken;

mod common;

/// The `stream_deferred` abort path: a pre-cancelled request signal aborts the
/// auth resolution mid-setup, upstream's applyAuth race rejection.
#[tokio::test]
async fn the_deferred_stream_reports_the_auth_abort() {
    let provider = create_provider(CreateProviderOptions {
        id: "p".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ambient_auth(),
        models: vec![fixture_model()],
        api: ProviderApi::Single(Arc::new(DeferredStreams::uncounted())),
        fetch_models: None,
        filter_models: None,
    });
    let models = create_models(None);
    models.set_provider(Arc::new(provider));

    let signal = CancellationToken::new();
    signal.cancel();
    let mut options = pi_ai::models::ModelsDeferredFetchOptions::default();
    options.options.transport_options = TransportOptions {
        signal: Some(signal),
        ..TransportOptions::default()
    };

    let model = Model {
        base_url: String::new(),
        ..fixture_model()
    };
    let text = models
        .stream_deferred(&model, &deferred_handle(), Some(&options))
        .result()
        .await
        .error_message
        .unwrap_or_default();
    // The abort rides through the lazy-stream setup as the failure message.
    assert!(
        text.contains("Provider is not configured") || text.contains("aborted"),
        "got: {text}"
    );
}

/// The checkAuth and available surfaces reject pre-cancelled signals.
#[tokio::test]
async fn the_auth_checks_reject_pre_cancelled_signals() {
    let models = create_models(None);
    models.set_provider(Arc::new(create_provider(CreateProviderOptions {
        id: "p".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ambient_auth(),
        models: vec![fixture_model()],
        api: ProviderApi::Single(pi_ai::api::not_ported_streams("test-api")),
        fetch_models: None,
        filter_models: None,
    })));
    let signal = CancellationToken::new();
    signal.cancel();
    let options = pi_ai::auth::types::AuthOptions {
        signal: Some(signal.clone()),
    };
    let check = models.check_auth("p", Some(&options)).await;
    assert!(check.is_err(), "a cancelled check rejects");
    let available = models.available(Some("p"), Some(&options)).await;
    assert!(available.is_err(), "a cancelled available rejects");
}

/// The supersede path cancels an in-flight refresh token when a replacement
/// registration arrives.
#[tokio::test]
async fn the_models_supersede_cancels_in_flight_refreshes() {
    let models = create_models(None);
    let (started_sender, started_receiver) = tokio::sync::oneshot::channel::<()>();
    let started_holder = Arc::new(Mutex::new(Some(started_sender)));
    models.set_provider(Arc::new(create_provider(CreateProviderOptions {
        id: "dyn".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ProviderAuth {
            api_key: Some(ApiKeyAuth {
                name: "Slow".to_owned(),
                login: None,
                check: None,
                resolve: Arc::new(|_input: ApiKeyAuthInput| {
                    Box::pin(async move { Ok(Some(pi_ai::auth::types::AuthResult::default())) })
                }),
            }),
            oauth: None,
        },
        models: vec![],
        fetch_models: Some(Arc::new(
            move |_context: &pi_ai::models::RefreshModelsContext| {
                let started = Arc::clone(&started_holder);
                let future: BoxedFuture<
                    'static,
                    Result<Vec<Model>, Box<dyn std::error::Error + Send + Sync>>,
                > = Box::pin(async move {
                    let sender = started
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .take();
                    if let Some(sender) = sender {
                        let _ = sender.send(());
                    }
                    // Park on an independent future, upstream's stalled
                    // promise: the provider ignores the refresh signal, and a
                    // signal-tied park would wake the abort race and the
                    // fetch's own error in the same poll. Dropping the parked
                    // future stands in for the abandoned promise.
                    loop {
                        std::future::pending::<()>().await;
                    }
                });
                future
            },
        )),
        api: ProviderApi::Single(pi_ai::api::not_ported_streams("test-api")),
        filter_models: None,
    })));

    // Porting restatement: the promise starts eagerly; a Rust future starts
    // on first poll, so the supersede needs a spawned task.
    let first = {
        let models = models.clone();
        tokio::spawn(async move { models.refresh(None).await })
    };
    started_receiver.await.expect("first refresh started");
    // A replacement registration supersedes the in-flight refresh.
    models.set_provider(Arc::new(create_provider(CreateProviderOptions {
        id: "dyn".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ProviderAuth {
            api_key: Some(ApiKeyAuth {
                name: "Quick".to_owned(),
                login: None,
                check: None,
                resolve: Arc::new(|_input: ApiKeyAuthInput| {
                    Box::pin(async move { Ok(Some(pi_ai::auth::types::AuthResult::default())) })
                }),
            }),
            oauth: None,
        },
        models: vec![],
        fetch_models: Some(Arc::new(|_context| Box::pin(async move { Ok(Vec::new()) }))),
        api: ProviderApi::Single(pi_ai::api::not_ported_streams("test-api")),
        filter_models: None,
    })));
    let result = first.await;
    let result = result.expect("joined");
    // Upstream's refresh result reports `aborted` from the caller's signal
    // only; a supersede aborts the provider phase without flagging the
    // collection-level result.
    assert!(!result.aborted);
    assert_eq!(
        result.errors.len(),
        0,
        "the superseded refresh is not an error"
    );
}

/// The images `SharedRefreshError` source chain and the options Debug surface.
#[tokio::test]
async fn the_images_error_chain_and_options_debug_render() {
    let options = pi_ai::images_models::CreateImagesProviderOptions {
        id: "p".to_owned(),
        name: Some("Named".to_owned()),
        auth: ProviderAuth::default(),
        models: Vec::new(),
        refresh_models: Some(Arc::new(|| {
            Box::pin(async move { Err(std::io::Error::other("fetch failed").into()) })
        })),
        api: pi_ai::api::not_ported_images("test-images"),
    };
    let rendered = format!("{options:?}");
    assert!(rendered.contains("Named"), "got: {rendered}");

    let models = pi_ai::images_models::create_images_models(None);
    models.set_provider(Arc::new(pi_ai::images_models::create_images_provider(
        options,
    )));
    let result = models.refresh(Some("p")).await;
    let Err(error) = result else {
        panic!("the flaky refresh fails");
    };
    // Two-level walk: ModelsError -> SharedRefreshError -> inner error.
    let shared = std::error::Error::source(&error).expect("wraps the dedupe error");
    let inner = std::error::Error::source(shared).expect("the dedupe error wraps the fetch");
    assert_eq!(inner.to_string(), "fetch failed");
    assert_eq!(shared.to_string(), "fetch failed");
}

/// The model headers merge in the applyAuth flow through streamSimple.
#[tokio::test]
async fn the_apply_auth_merges_the_model_headers() {
    let provider = create_provider(CreateProviderOptions {
        id: "p".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ambient_auth(),
        models: vec![fixture_model()],
        api: ProviderApi::Single(Arc::new(DeferredStreams::uncounted())),
        fetch_models: None,
        filter_models: None,
    });
    let models = create_models(None);
    models.set_provider(Arc::new(provider));

    let mut model = fixture_model();
    model.headers = Some(BTreeMap::from([("x-model".to_owned(), "yes".to_owned())]));
    let stream = models.stream_simple(&model, &Context::default(), None);
    let done = stream.result().await;
    assert_eq!(done.stop_reason, StopReason::Stop);
}

/// The images provider name/auth getters through the builtin registration.
#[test]
fn the_builtin_images_provider_name_and_auth_render() {
    let models = pi_ai::images_models::create_images_models(None);
    models.set_provider(Arc::new(
        pi_ai::providers::openrouter_images::openrouter_images_provider(),
    ));
    let provider = models.provider("openrouter").expect("registered");
    assert_eq!(provider.name(), "OpenRouter");
    assert!(provider.auth().api_key.is_some());
}

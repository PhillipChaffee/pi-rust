//! The Models collection's deferred-response and login/logout edges plus the
//! `ProviderImpl` deferred dispatch, upstream's `models.ts` deferred contract,
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use common::{DeferredStreams, ambient_auth, block, deferred_handle, fixture_model};
use pi_ai::auth::types::{ApiKeyAuth, ApiKeyAuthInput, ProviderAuth};
use pi_ai::models::{
    CreateProviderOptions, ModelsDeferredCancelOptions, ProviderApi, create_models, create_provider,
};
use pi_ai::types::{Api, BoxedFuture, Context, StopReason, StreamOptions};

mod common;

/// The deferred dispatch: `Models.stream_deferred` routes to the provider's
/// `fetch_deferred`, `fetch_deferred` completes to the handle message, and
/// `cancel_deferred` reaches the provider's cancel.
#[test]
fn the_deferred_dispatch_routes_end_to_end() {
    let (streams, fetches, cancels) = DeferredStreams::new();
    let provider = create_provider(CreateProviderOptions {
        id: "p".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ambient_auth(),
        models: vec![fixture_model()],
        api: ProviderApi::Single(Arc::new(streams)),
        fetch_models: None,
        filter_models: None,
    });
    let models = create_models(None);
    models.set_provider(Arc::new(provider));

    let model = fixture_model();
    let handle = deferred_handle();
    block(async move {
        // stream_deferred returns the deferred stream live.
        let message = models.stream_deferred(&model, &handle, None).result().await;
        assert_eq!(message.stop_reason, StopReason::Stop);
        assert!(message.deferred.is_some());

        // fetch_deferred completes through the same path.
        let fetched_message = models.fetch_deferred(&model, &handle, None).await;
        assert_eq!(fetched_message.stop_reason, StopReason::Stop);

        // cancel_deferred reaches the provider's cancel.
        models
            .cancel_deferred(
                &model,
                &handle,
                Some(&ModelsDeferredCancelOptions::default()),
            )
            .await
            .expect("cancel");
    });
    assert_eq!(
        *fetches
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        2
    );
    assert_eq!(
        *cancels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        1
    );
}

/// A provider whose streams do not report deferred support: the Models
/// surface settles the stream with the not-supported notice.
#[test]
fn the_deferred_surface_reports_unsupported_providers() {
    let provider = create_provider(CreateProviderOptions {
        id: "p".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ambient_auth(),
        models: vec![fixture_model()],
        api: ProviderApi::Single(pi_ai::api::not_ported_streams("test-api")),
        fetch_models: None,
        filter_models: None,
    });
    let models = create_models(None);
    models.set_provider(Arc::new(provider));

    let model = fixture_model();
    let handle = deferred_handle();
    let text = block(async move {
        models
            .stream_deferred(&model, &handle, None)
            .result()
            .await
            .error_message
            .unwrap_or_default()
    });
    assert!(
        text.contains("does not support deferred responses"),
        "got: {text}"
    );
}

/// The login/logout surface: unknown providers and unsupported methods
/// reject; logout resolves for any provider id.
#[tokio::test]
async fn the_login_logout_surface_walks_its_edges() {
    let models = create_models(None);
    let interaction = pi_ai::auth::types::AuthInteraction {
        signal: None,
        prompt: Arc::new(|_prompt: pi_ai::auth::types::AuthPrompt| {
            let entered: BoxedFuture<'static, Result<String, pi_ai::utils::abort::AbortError>> =
                Box::pin(async { Ok("key".to_owned()) });
            entered
        }),
        notify: Arc::new(|_event| {}),
    };
    // Unknown provider login rejects with the provider code.
    let error = models
        .login(
            "ghost",
            pi_ai::auth::types::AuthType::ApiKey,
            interaction.clone(),
        )
        .await
        .expect_err("unknown provider");
    assert!(
        matches!(error, pi_ai::auth::resolve::ModelsFailure::Models(error) if error.code() == pi_ai::auth::resolve::ModelsErrorCode::Provider)
    );

    // A provider whose api-key auth carries no login rejects unsupported, for
    // both auth types.
    models.set_provider(Arc::new(create_provider(CreateProviderOptions {
        id: "keyless".to_owned(),
        name: Some("Keyless".to_owned()),
        base_url: None,
        headers: None,
        auth: ProviderAuth {
            api_key: Some(ApiKeyAuth {
                name: "Ambient".to_owned(),
                login: None,
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
            "keyless",
            pi_ai::auth::types::AuthType::ApiKey,
            interaction.clone(),
        )
        .await
        .expect_err("no login");
    assert!(
        error.to_string().contains("does not support api_key login"),
        "got: {error}"
    );
    let error = models
        .login("keyless", pi_ai::auth::types::AuthType::OAuth, interaction)
        .await
        .expect_err("no oauth");
    assert!(
        error.to_string().contains("does not support oauth login"),
        "got: {error}"
    );

    // logout resolves regardless.
    models.logout("ghost", None).await.expect("logout");
}

/// The mixed-API provider dispatches by model.api and reports the
/// missing-API stream error for unknown apis, upstream's api-map dispatch.
#[tokio::test]
async fn the_api_map_dispatch_serves_each_api_and_reports_missing_ones() {
    use pi_ai::models::Provider;
    let provider = create_provider(CreateProviderOptions {
        id: "mixed".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ambient_auth(),
        models: vec![fixture_model()],
        api: ProviderApi::ByApi(BTreeMap::from([(
            "other-api".to_owned(),
            pi_ai::api::not_ported_streams("other-api"),
        )])),
        fetch_models: None,
        filter_models: None,
    });
    // A model whose api has no entry produces the stream error.
    let message = provider
        .stream(&fixture_model(), &Context::default(), None)
        .result()
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    let text = message.error_message.unwrap_or_default();
    assert!(
        text.contains("has no API implementation for \"test-api\""),
        "got: {text}"
    );
    // The map's own api dispatches.
    let mut other = fixture_model();
    other.api = Api::from("other-api");
    let message = provider
        .stream(&other, &Context::default(), None)
        .result()
        .await;
    assert_eq!(
        message.stop_reason,
        StopReason::Error,
        "the stub settles with its notice"
    );
    let text = message.error_message.unwrap_or_default();
    assert!(
        text.contains("The other-api wire API has not been ported yet"),
        "the other-api dispatch reached the stub"
    );
}

/// The Debug surfaces of the publication/collection option shapes.
#[test]
fn the_models_debug_surfaces_render() {
    use pi_ai::models::{CatalogPersist, CreateModelsOptions, ModelsPublication};
    let publication = ModelsPublication {
        persist: CatalogPersist::Write(pi_ai::models_store::ModelsStoreEntry::default()),
        update: None,
    };
    assert!(format!("{publication:?}").contains("ModelsPublication"));
    let options = CreateModelsOptions {
        credentials: Some(Arc::new(
            pi_ai::auth::credential_store::InMemoryCredentialStore::default(),
        )),
        ..CreateModelsOptions::default()
    };
    assert!(format!("{options:?}").contains("set"));
    let transformed: pi_ai::models::ModelsStreamOptions = pi_ai::models::ModelsStreamOptions {
        options: StreamOptions::default(),
        transform_headers: None,
    };
    assert!(format!("{transformed:?}").contains("WithTransforms"));
    let simple: pi_ai::models::ModelsSimpleStreamOptions =
        pi_ai::models::ModelsSimpleStreamOptions::default();
    assert!(format!("{simple:?}").contains("WithTransforms"));
}

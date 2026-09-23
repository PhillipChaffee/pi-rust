//! The Radius dynamic-catalog network flow: a loopback gateway serves the
//! config JSON and the provider refreshes through it, upstream's
//! `radius.ts` + `loadRadiusGatewayConfig` contract, at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::auth::types::{Credential, OAuthCredentials};
use pi_ai::models::ModelsRefreshOptions;
use pi_ai::models_store::ModelsStore;
use pi_ai::providers::radius::{RadiusProviderOptions, radius_provider};
use tokio::io::AsyncWriteExt;

/// The gateway config JSON the loopback serves, matching the shape the real
/// Radius gateway emits.
const GATEWAY_CONFIG: &str = r#"{
    "baseUrl": "https://gateway.test",
    "models": [
        {
            "id": "m1",
            "name": "M1",
            "reasoning": true,
            "input": ["text"],
            "cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 1000,
            "maxTokens": 100
        }
    ]
}"#;

/// A minimal HTTP/1.1 response over one accepted connection.
async fn serve_one(listener: &tokio::net::TcpListener, body: &'static str) {
    let (mut socket, _) = listener.accept().await.expect("accept");
    let mut buffer = [0u8; 2048];
    let _ = socket.read(&mut buffer).await;
    socket
        .write_all(
            format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .expect("write response");
    socket.shutdown().await.expect("shutdown");
}

use tokio::io::AsyncReadExt;

/// The loopback gateway serves `/v1/config`; the radius provider's network
/// refresh phase loads the config, persists it, and lists the models.
#[tokio::test]
async fn the_radius_provider_refreshes_from_a_loopback_gateway() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();

    let _server = tokio::spawn(async move { serve_one(&listener, GATEWAY_CONFIG).await });

    let provider = radius_provider(RadiusProviderOptions {
        gateway: Some(format!("http://127.0.0.1:{port}")),
        ..Default::default()
    });
    let store = Arc::new(pi_ai::models_store::InMemoryModelsStore::default());
    let credentials: Arc<dyn pi_ai::auth::credential_store::CredentialStore> =
        Arc::new(pi_ai::auth::credential_store::InMemoryCredentialStore::default());
    // The network phase only runs for configured providers, so the refresh
    // carries a stored api key, upstream's resolveRefreshCredential flow.
    store_credential(&credentials).await;
    let models = pi_ai::models::create_models(Some(pi_ai::models::CreateModelsOptions {
        credentials: Some(Arc::clone(&credentials)),
        models_store: Some(store),
        ..pi_ai::models::CreateModelsOptions::default()
    }));
    models.set_provider(provider);

    let result = models.refresh(None).await;
    assert_eq!(result.errors.len(), 0, "errors: {:?}", result.errors);
    // The fetched config materialized as models.
    let fetched = models.model("radius", "m1").expect("m1 listed");
    assert_eq!(fetched.base_url, "https://gateway.test");
    assert_eq!(
        fetched.api.as_known(),
        Some(pi_ai::types::KnownApi::PiMessages)
    );
}

/// The legacy import: a stored OAuth credential with a gatewayConfig seeds
/// the catalog through the legacy-import phase before any network access.
#[tokio::test]
async fn the_radius_provider_imports_the_legacy_gateway_config() {
    let provider = radius_provider(RadiusProviderOptions::default());
    let credentials: Arc<dyn pi_ai::auth::credential_store::CredentialStore> =
        Arc::new(pi_ai::auth::credential_store::InMemoryCredentialStore::default());
    let gateway_oauth: OAuthCredentials = OAuthCredentials {
        refresh: "r".to_owned(),
        access: "a".to_owned(),
        expires: i64::MAX,
        extra: BTreeMap::from([(
            "gatewayConfig".to_owned(),
            serde_json::json!({
                "baseUrl": "https://gateway.test",
                "models": [{
                    "id": "legacy",
                    "name": "Legacy",
                    "reasoning": false,
                    "input": ["text"],
                    "cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 },
                    "contextWindow": 1000,
                    "maxTokens": 100,
                }],
            }),
        )]),
    };
    credentials
        .modify(
            "radius",
            Box::new(move |_current| {
                Box::pin(async move { Ok(Some(Credential::OAuth(gateway_oauth.clone()))) })
            }),
            None,
        )
        .await
        .expect("store");

    let models = pi_ai::models::create_models(Some(pi_ai::models::CreateModelsOptions {
        credentials: Some(Arc::clone(&credentials)),
        ..pi_ai::models::CreateModelsOptions::default()
    }));
    models.set_provider(provider);

    // The offline phase imports the legacy catalog; the network phase is
    // skipped, so no gateway is contacted.
    let result = models
        .refresh(Some(&ModelsRefreshOptions {
            allow_network: Some(false),
            ..ModelsRefreshOptions::default()
        }))
        .await;
    assert_eq!(result.errors.len(), 0);
    assert!(models.model("radius", "legacy").is_some());
}

/// The restore phase: a persisted catalog comes back offline, upstream's
/// `context.stored` restore-and-publish.
#[tokio::test]
async fn the_radius_provider_restores_the_persisted_catalog() {
    let provider = radius_provider(RadiusProviderOptions::default());
    let store = Arc::new(pi_ai::models_store::InMemoryModelsStore::default());
    store
        .write(
            "radius",
            pi_ai::models_store::ModelsStoreEntry {
                models: vec![pi_ai::types::Model {
                    id: "persisted".to_owned(),
                    name: "Persisted".to_owned(),
                    api: pi_ai::types::Api::from(pi_ai::types::KnownApi::PiMessages),
                    provider: pi_ai::types::ProviderId::from("radius"),
                    base_url: "https://old-gateway.test".to_owned(),
                    reasoning: false,
                    thinking_level_map: None,
                    input: vec![pi_ai::types::Modality::Text],
                    cost: pi_ai::types::ModelCost::default(),
                    context_window: 1000,
                    max_tokens: 100,
                    sampling_params: None,
                    headers: None,
                    compat: None,
                }],
                ..pi_ai::models_store::ModelsStoreEntry::default()
            },
            None,
        )
        .await
        .expect("seed store");
    let models = pi_ai::models::create_models(Some(pi_ai::models::CreateModelsOptions {
        models_store: Some(store),
        ..pi_ai::models::CreateModelsOptions::default()
    }));
    models.set_provider(provider);

    let result = models
        .refresh(Some(&ModelsRefreshOptions {
            allow_network: Some(false),
            ..ModelsRefreshOptions::default()
        }))
        .await;
    assert_eq!(result.errors.len(), 0);
    assert!(models.model("radius", "persisted").is_some());
}

/// A gateway failure during the network phase surfaces as a refresh error
/// and leaves the in-memory list at its last-known state.
#[tokio::test]
async fn the_radius_provider_reports_a_failed_gateway_load() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();

    let _server = tokio::spawn(async move { serve_one(&listener, "not-json").await });

    let provider = radius_provider(RadiusProviderOptions {
        gateway: Some(format!("http://127.0.0.1:{port}")),
        ..Default::default()
    });
    let credentials: Arc<dyn pi_ai::auth::credential_store::CredentialStore> =
        Arc::new(pi_ai::auth::credential_store::InMemoryCredentialStore::default());
    store_credential(&credentials).await;
    let models = pi_ai::models::create_models(Some(pi_ai::models::CreateModelsOptions {
        credentials: Some(Arc::clone(&credentials)),
        ..pi_ai::models::CreateModelsOptions::default()
    }));
    models.set_provider(provider);

    let result = models.refresh(None).await;
    assert!(
        result.errors.contains_key("radius"),
        "the failed load is recorded"
    );
    assert!(models.models(Some("radius")).is_empty());
}

/// Store an api-key credential the refresh's network phase carries.
async fn store_credential(credentials: &Arc<dyn pi_ai::auth::credential_store::CredentialStore>) {
    credentials
        .modify(
            "radius",
            Box::new(move |_current| {
                Box::pin(async move {
                    Ok(Some(Credential::ApiKey(
                        pi_ai::auth::types::ApiKeyCredential {
                            key: Some("radius-key".to_owned()),
                            env: None,
                        },
                    )))
                })
            }),
            None,
        )
        .await
        .expect("store");
}

/// A non-2xx gateway response with an over-long body; the loader truncates
/// the body with the ellipsis in the error message.
async fn serve_long_error(listener: &tokio::net::TcpListener, body: String) {
    let (mut socket, _) = listener.accept().await.expect("accept");
    let mut buffer = [0u8; 2048];
    let _ = socket.read(&mut buffer).await;
    socket
        .write_all(
            format!(
                "HTTP/1.1 502 Bad Gateway\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .expect("write response");
    socket.shutdown().await.expect("shutdown");
}

/// The non-2xx gateway error truncates the body at 512 chars with the
/// ellipsis, upstream's `truncateHttpBody`.
#[tokio::test]
async fn the_gateway_error_body_truncates_with_the_ellipsis() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();

    let long_body = "x".repeat(900);
    let server_body = long_body.clone();
    let _server = tokio::spawn(async move { serve_long_error(&listener, server_body).await });

    let gateway = format!("http://127.0.0.1:{port}");
    let error = pi_ai::providers::radius_config::load_radius_gateway_config(
        gateway.as_str(),
        None,
        &tokio_util::sync::CancellationToken::new(),
    )
    .await
    .expect_err("the failed load");

    assert!(error.message.contains(": 502: "), "{error}");
    assert!(
        error.message.ends_with('\u{2026}'),
        "the body truncates with the ellipsis: {}",
        &error.message[error.message.len() - 4..]
    );
    assert!(
        error.message.contains("x".repeat(512).as_str()),
        "512 chars of the body ride"
    );
}

/// The gateway URL normalization, the config sanitization gate, and the
/// credential-config extraction, upstream's radius-config helpers.
#[test]
fn the_radius_config_helpers_pin_their_shapes() {
    use pi_ai::providers::radius_config::{
        get_radius_credential_config, get_radius_models, get_radius_models_from_config,
        normalize_radius_gateway_url, sanitize_radius_gateway_config,
    };

    assert_eq!(
        normalize_radius_gateway_url("gateway.example/"),
        "https://gateway.example",
        "the scheme is added and the trailing slash strips"
    );
    assert_eq!(
        normalize_radius_gateway_url("http://gateway.example//"),
        "http://gateway.example",
        "the plain scheme stays and the trailing slashes strip"
    );

    let config_json = serde_json::json!({
        "baseUrl": "https://gateway.test",
        "models": [{
            "id": "m1",
            "name": "M1",
            "reasoning": true,
            "input": ["text"],
            "cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 1000,
            "maxTokens": 100,
        }],
    });
    let config = sanitize_radius_gateway_config(&config_json).expect("the config sanitizes");
    let models = get_radius_models_from_config("radius", &config);
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].base_url, "https://gateway.test");

    // A shape without the base URL or models arrays sanitizes to nothing.
    assert!(sanitize_radius_gateway_config(&serde_json::json!({ "models": [] })).is_none());
    assert!(
        sanitize_radius_gateway_config(&serde_json::json!({
            "baseUrl": "https://gateway.test",
            "models": [{ "id": "m1" }],
        }))
        .is_some_and(|sanitized| sanitized.models.is_empty()),
        "the invalid model entries drop"
    );

    // The credential-config extraction reads the OAuth extra block.
    let oauth = OAuthCredentials {
        refresh: "r".to_owned(),
        access: "a".to_owned(),
        expires: 1,
        extra: BTreeMap::from([("gatewayConfig".to_owned(), config_json)]),
    };
    assert_eq!(get_radius_credential_config(Some(&oauth)), Some(config));
    assert_eq!(get_radius_models("radius", Some(&oauth)).len(), 1);
    assert!(get_radius_models("radius", None).is_empty());

    // The error formats as its message.
    let error = pi_ai::providers::radius_config::GatewayConfigError {
        message: "Could not load Radius config from https://gateway.test".to_owned(),
    };
    assert_eq!(
        error.to_string(),
        "Could not load Radius config from https://gateway.test"
    );
}

/// The radius provider trait surface: the identity, the auth source, the
/// model list, and the stream delegation, upstream's provider object.
#[tokio::test]
async fn the_radius_provider_reports_its_identity_and_models() {
    let provider = radius_provider(RadiusProviderOptions {
        id: Some("radius-test".to_owned()),
        name: Some("Radius Test".to_owned()),
        ..Default::default()
    });
    assert_eq!(
        pi_ai::models::Provider::id(provider.as_ref()),
        "radius-test"
    );
    assert_eq!(
        pi_ai::models::Provider::name(provider.as_ref()),
        "Radius Test"
    );
    assert!(
        pi_ai::models::Provider::auth(provider.as_ref())
            .api_key
            .is_some()
    );
    assert!(
        pi_ai::models::Provider::get_models(provider.as_ref())
            .expect("models")
            .is_empty(),
        "no credential means no models"
    );
    // The default options carry the default gateway and identity.
    let default = radius_provider(RadiusProviderOptions::default());
    assert_eq!(pi_ai::models::Provider::id(default.as_ref()), "radius");
    assert_eq!(pi_ai::models::Provider::name(default.as_ref()), "Radius");
}

//! Rust-native coverage edges for the merged provider/auth surfaces, driven
//! through the public API: the provider factories' lazy-OAuth loads, the
//! Vertex AI login/resolve branches, the Cloudflare env resolution, the mock
//! WebSocket peer halves, and the auth stub error. Upstream covers these
//! through `Models` integration surfaces (`packages/ai/src/providers/*.ts`,
//! `packages/ai/src/auth/*.ts`) at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`;
//! here each suite pins one contract directly.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected shape panics by design"
)]

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use common::auth_fixtures::{MapAuthContext, oauth_credentials};
use common::auth_interaction::{ScriptedAuthInteraction, provider_interaction};
use pi_ai::auth::context::DefaultAuthContext;
use pi_ai::auth::helpers::oauth_stub_error;
use pi_ai::auth::types::{ApiKeyAuthInput, ApiKeyCredential, OAuthCredentials};
use pi_ai::http::{
    HttpClient as _, HttpMethod, HttpRequest, MockWebSocketTransport, ReqwestHttpClient,
    WebSocketMessage, WebSocketOutbound, WebSocketRequest, WebSocketTransport as _,
};
use pi_ai::images_models::ImagesProvider as _;
use pi_ai::providers::{
    anthropic, cloudflare_auth, cloudflare_stream, github_copilot, google_vertex, kimi_coding,
    openai_codex, openrouter, openrouter_images, xai,
};
use tokio_util::sync::CancellationToken;

fn credential() -> OAuthCredentials {
    oauth_credentials("access-token", "refresh-token", 0)
}

/// Each factory builds its surface, and the OAuth wrapper loads the real flow
/// on first derivation — the load closures the providers register resolve.
#[tokio::test]
async fn the_provider_factories_load_their_oauth_flows_on_first_to_auth() {
    for provider in [
        xai::xai_provider(),
        kimi_coding::kimi_coding_provider(),
        openai_codex::openai_codex_provider(),
        openrouter::openrouter_provider(),
        anthropic::anthropic_provider(),
        github_copilot::github_copilot_provider(),
    ] {
        let oauth = provider
            .auth()
            .oauth
            .clone()
            .expect("the provider advertises OAuth");
        // Each flow derives its own auth shape (bearer header, api key, ...);
        // the flow-specific shapes are pinned in their own suites.
        (oauth.to_auth)(credential())
            .await
            .expect("the flow loads and derives");
    }

    let images = openrouter_images::openrouter_images_provider();
    let oauth = images
        .auth()
        .oauth
        .clone()
        .expect("the images provider advertises OAuth");
    (oauth.to_auth)(credential())
        .await
        .expect("the flow loads and derives");
}

/// The factories keep their upstream ids, and the Radius variant applies its
/// option overrides and Debug shape.
#[tokio::test]
async fn the_provider_factories_report_their_registry_surface() {
    assert_eq!(xai::xai_provider().id(), "xai");
    assert_eq!(kimi_coding::kimi_coding_provider().id(), "kimi-coding");
    assert_eq!(openai_codex::openai_codex_provider().id(), "openai-codex");
    assert_eq!(openrouter::openrouter_provider().id(), "openrouter");
    assert_eq!(anthropic::anthropic_provider().id(), "anthropic");
    assert_eq!(
        github_copilot::github_copilot_provider().id(),
        "github-copilot"
    );
    assert_eq!(
        google_vertex::google_vertex_provider().id(),
        "google-vertex"
    );

    let radius = pi_ai::providers::radius::radius_provider(
        pi_ai::providers::radius::RadiusProviderOptions {
            id: Some("r2".to_owned()),
            name: Some("Radius Two".to_owned()),
            gateway: Some("radius.example/".to_owned()),
        },
    );
    assert_eq!(radius.id(), "r2");
    assert_eq!(radius.name(), "Radius Two");
    assert!(
        radius.supports_refresh_models(),
        "radius refreshes its dynamic catalog"
    );
}

/// The stub error every not-yet-ported OAuth flow reports carries its
/// message through `Display`.
#[test]
fn the_oauth_stub_error_reports_its_message() {
    let error = oauth_stub_error("flow not ported");
    assert_eq!(error.to_string(), "flow not ported");
}

/// The Vertex login walks each selected method: the api-key prompt, the ADC
/// project/location pair, the service-account file path, and the unknown
/// method rejection.
#[tokio::test]
async fn the_vertex_login_walks_each_selected_method() {
    let provider = google_vertex::google_vertex_provider();
    let login = provider
        .auth()
        .api_key
        .as_ref()
        .and_then(|auth| auth.login.clone())
        .expect("vertex advertises a login");
    let signal = CancellationToken::new();

    let scripted = ScriptedAuthInteraction::answering("api-key");
    let credential = login(provider_interaction(&scripted, signal.clone()))
        .await
        .expect("the api-key method resolves");
    assert_eq!(credential.key.as_deref(), Some("api-key"));

    let scripted = ScriptedAuthInteraction::answering("adc");
    let credential = login(provider_interaction(&scripted, signal.clone()))
        .await
        .expect("the adc method resolves");
    assert_eq!(credential.key, None, "adc stores no key");
    let env = credential
        .env
        .expect("the adc method stores project and location");
    assert!(env.contains_key("GOOGLE_CLOUD_PROJECT"));
    assert!(env.contains_key("GOOGLE_CLOUD_LOCATION"));
    assert!(
        !env.contains_key("GOOGLE_APPLICATION_CREDENTIALS"),
        "adc stores no credentials file"
    );

    let scripted = ScriptedAuthInteraction::answering("service-account");
    let credential = login(provider_interaction(&scripted, signal.clone()))
        .await
        .expect("the service-account method resolves");
    let env = credential
        .env
        .expect("the service-account stores its file path");
    assert!(env.contains_key("GOOGLE_APPLICATION_CREDENTIALS"));

    let scripted = ScriptedAuthInteraction::answering("teleport");
    let error = login(provider_interaction(&scripted, signal.clone()))
        .await
        .expect_err("the unknown method rejects");
    assert!(
        error
            .to_string()
            .starts_with("Unknown Google Vertex AI auth method"),
        "{error}"
    );
}

/// The Vertex resolve prefers a stored key, then `GOOGLE_CLOUD_API_KEY`,
/// then ADC credentials with project and location, then nothing.
#[tokio::test]
async fn the_vertex_resolve_prefers_keys_then_adc_then_nothing() {
    let provider = google_vertex::google_vertex_provider();
    let resolve = provider
        .auth()
        .api_key
        .as_ref()
        .map(|auth| auth.resolve.clone())
        .expect("vertex advertises a resolve");
    let signal = CancellationToken::new();

    let stored = resolve(ApiKeyAuthInput {
        ctx: Arc::new(MapAuthContext::default()),
        credential: Some(ApiKeyCredential {
            key: Some("stored-key".to_owned()),
            env: None,
        }),
        signal: signal.clone(),
    })
    .await
    .expect("the stored key resolves")
    .expect("the stored key is present");
    assert_eq!(stored.auth.api_key.as_deref(), Some("stored-key"));
    assert_eq!(stored.source.as_deref(), Some("stored credential"));

    let ambient = resolve(ApiKeyAuthInput {
        ctx: Arc::new(MapAuthContext::new([(
            "GOOGLE_CLOUD_API_KEY",
            "ambient-key",
        )])),
        credential: None,
        signal: signal.clone(),
    })
    .await
    .expect("the env-key resolve runs")
    .expect("the ambient key is present");
    assert_eq!(ambient.auth.api_key.as_deref(), Some("ambient-key"));
    assert_eq!(ambient.source.as_deref(), Some("GOOGLE_CLOUD_API_KEY"));

    let file = sandbox_file("vertex-adc-credentials.json");
    let adc = resolve(ApiKeyAuthInput {
        ctx: Arc::new(DefaultAuthContext),
        credential: Some(ApiKeyCredential {
            key: None,
            env: Some(BTreeMap::from([
                ("GOOGLE_APPLICATION_CREDENTIALS".to_owned(), file),
                ("GOOGLE_CLOUD_PROJECT".to_owned(), "proj".to_owned()),
                ("GOOGLE_CLOUD_LOCATION".to_owned(), "us-central1".to_owned()),
            ])),
        }),
        signal: signal.clone(),
    })
    .await
    .expect("the adc resolve runs")
    .expect("the adc credentials resolve");
    assert!(
        adc.auth.api_key.is_none(),
        "adc carries provider env, not a key"
    );
    assert_eq!(adc.source.as_deref(), Some("stored credential"));

    let absent = resolve(ApiKeyAuthInput {
        ctx: Arc::new(DefaultAuthContext),
        credential: Some(ApiKeyCredential {
            key: None,
            env: None,
        }),
        signal: signal.clone(),
    })
    .await
    .expect("the adc fallback resolve runs");
    assert!(
        absent.is_none(),
        "the sandbox has no ADC file, so the resolve falls through"
    );

    let none = resolve(ApiKeyAuthInput {
        ctx: Arc::new(MapAuthContext::default()),
        credential: None,
        signal: signal.clone(),
    })
    .await
    .expect("the empty resolve runs");
    assert!(none.is_none(), "no credentials resolve to none");
}

/// A zero-byte credentials file the ADC probe treats as present.
fn sandbox_file(name: &str) -> String {
    let path = std::env::temp_dir().join(format!("pi-ai-merged-edges-{name}"));
    std::fs::write(&path, b"{}").expect("the sandbox file writes");
    path.display().to_string()
}

/// The Cloudflare auth resolves each field from the credential first, then
/// the ambient env, and requires the gateway id only for AI Gateway.
#[tokio::test]
async fn the_cloudflare_auth_resolves_from_the_credential_then_the_env() {
    let signal = CancellationToken::new();
    let workers = cloudflare_auth::cloudflare_workers_ai_auth();

    let stored = (workers.resolve)(ApiKeyAuthInput {
        ctx: Arc::new(MapAuthContext::default()),
        credential: Some(ApiKeyCredential {
            key: Some("cf-key".to_owned()),
            env: Some(BTreeMap::from([(
                "CLOUDFLARE_ACCOUNT_ID".to_owned(),
                "acc-from-credential".to_owned(),
            )])),
        }),
        signal: signal.clone(),
    })
    .await
    .expect("the stored credential resolves")
    .expect("the credential is present");
    assert_eq!(stored.auth.api_key.as_deref(), Some("cf-key"));
    assert_eq!(stored.source.as_deref(), Some("stored credential"));
    let env = stored.env.expect("the account id rides the provider env");
    assert_eq!(
        env.get("CLOUDFLARE_ACCOUNT_ID").map(String::as_str),
        Some("acc-from-credential")
    );

    let ambient = (workers.resolve)(ApiKeyAuthInput {
        ctx: Arc::new(MapAuthContext::new([
            ("CLOUDFLARE_API_KEY", "env-key"),
            ("CLOUDFLARE_ACCOUNT_ID", "env-account"),
        ])),
        credential: None,
        signal: signal.clone(),
    })
    .await
    .expect("the env resolution runs")
    .expect("the env values are present");
    assert_eq!(ambient.auth.api_key.as_deref(), Some("env-key"));
    assert_eq!(ambient.source.as_deref(), Some("CLOUDFLARE_API_KEY"));

    let missing = (workers.resolve)(ApiKeyAuthInput {
        ctx: Arc::new(MapAuthContext::default()),
        credential: None,
        signal: signal.clone(),
    })
    .await
    .expect("the empty resolution runs");
    assert!(missing.is_none(), "no values resolve to none");

    let gateway = cloudflare_auth::cloudflare_ai_gateway_auth();
    let gateway_credential = (gateway.resolve)(ApiKeyAuthInput {
        ctx: Arc::new(MapAuthContext::default()),
        credential: Some(ApiKeyCredential {
            key: Some("gateway-key".to_owned()),
            env: Some(BTreeMap::from([
                ("CLOUDFLARE_ACCOUNT_ID".to_owned(), "acc".to_owned()),
                ("CLOUDFLARE_GATEWAY_ID".to_owned(), "gw".to_owned()),
            ])),
        }),
        signal: signal.clone(),
    })
    .await
    .expect("the gateway credential resolves")
    .expect("the gateway credential is present");
    let headers = gateway_credential
        .auth
        .headers
        .expect("the gateway key rides its own header");
    assert!(
        headers
            .get("cf-aig-authorization")
            .and_then(Option::as_deref)
            .is_some_and(|value| value.ends_with("gateway-key")),
        "{headers:?}"
    );
    assert_eq!(
        headers.get("Authorization"),
        Some(&None),
        "the standard authorization header is suppressed"
    );
    assert!(
        headers.get("x-api-key") == Some(&None),
        "the standard api key header is suppressed"
    );

    let gateway_missing = (gateway.resolve)(ApiKeyAuthInput {
        ctx: Arc::new(MapAuthContext::new([
            ("CLOUDFLARE_API_KEY", "env-key"),
            ("CLOUDFLARE_ACCOUNT_ID", "env-account"),
        ])),
        credential: None,
        signal: signal.clone(),
    })
    .await
    .expect("the gateway-less resolution runs");
    assert!(
        gateway_missing.is_none(),
        "ai gateway without a gateway id resolves to none"
    );
}

/// The Cloudflare model endpoint materializes `{CLOUDFLARE_ACCOUNT_ID}` and
/// `{CLOUDFLARE_GATEWAY_ID}` from the resolved provider env; a placeholder
/// without a value stays literal.
#[test]
fn the_cloudflare_model_placeholders_materialize_from_the_provider_env() {
    let mut model = common::fixture_model();
    model.base_url =
        "https://gateway.ai/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}".to_owned();

    let untouched = cloudflare_stream::resolve_cloudflare_model(&model, None);
    assert_eq!(untouched.base_url, model.base_url);

    let env = BTreeMap::from([
        ("CLOUDFLARE_ACCOUNT_ID".to_owned(), "acct".to_owned()),
        ("CLOUDFLARE_GATEWAY_ID".to_owned(), "gwy".to_owned()),
    ]);
    let resolved = cloudflare_stream::resolve_cloudflare_model(&model, Some(&env));
    assert_eq!(resolved.base_url, "https://gateway.ai/acct/gwy");

    let partial = BTreeMap::from([("CLOUDFLARE_ACCOUNT_ID".to_owned(), "acct".to_owned())]);
    let half = cloudflare_stream::resolve_cloudflare_model(&model, Some(&partial));
    assert_eq!(
        half.base_url,
        "https://gateway.ai/acct/{CLOUDFLARE_GATEWAY_ID}"
    );
}

/// The mock WebSocket pair routes messages both ways and reports the
/// peer-initiated close to the reader.
#[tokio::test]
async fn the_mock_websocket_pair_routes_messages_and_the_peer_close() {
    let transport = MockWebSocketTransport::new();
    let request = WebSocketRequest {
        url: "ws://example.test/ws".to_owned(),
        headers: vec![("x-test".to_owned(), "1".to_owned())],
        connect_timeout_ms: None,
        signal: CancellationToken::new(),
    };
    let mut connection = transport
        .connect(request)
        .await
        .expect("the mock transport accepts the connect");
    let mut peer = transport.next_peer().expect("the mock pairs a peer");
    assert_eq!(
        peer.connect_request().url,
        "ws://example.test/ws",
        "the handshake request is recorded"
    );

    connection
        .send(WebSocketMessage::Text("out".to_owned()))
        .await
        .expect("the client send routes to the peer");
    match peer.next_sent().await.expect("the peer reads the send") {
        WebSocketOutbound::Message(WebSocketMessage::Text(text)) => assert_eq!(text, "out"),
        other => panic!("unexpected outbound: {other:?}"),
    }

    peer.send_text("in").expect("the peer pushes a text frame");
    match connection
        .next_message()
        .await
        .expect("the client reads it")
    {
        WebSocketMessage::Text(text) => assert_eq!(text, "in"),
        other @ WebSocketMessage::Binary(_) => panic!("unexpected message: {other:?}"),
    }

    peer.close(1000, "done")
        .expect("the peer closes the stream");
    let error = connection
        .next_message()
        .await
        .expect_err("the close surfaces to the reader");
    assert!(error.to_string().contains("done"), "{error}");
}

/// The default Reqwest transport reports a refused connection as a transport
/// failure rather than a panic.
#[tokio::test]
async fn the_reqwest_transport_reports_a_refused_connection() {
    let client = ReqwestHttpClient::default();
    let request = HttpRequest {
        method: HttpMethod::Get,
        url: "http://127.0.0.1:9/".to_owned(),
        headers: Vec::new(),
        body: None,
        timeout_ms: Some(2_000),
        signal: CancellationToken::new(),
    };
    let error = client
        .execute(request)
        .await
        .expect_err("nothing listens on the discard port");
    assert!(
        matches!(error, pi_ai::http::HttpError::Transport(_)),
        "{error:?}"
    );
}

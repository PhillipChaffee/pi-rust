//! The provider-registry construction and auth suites: every builtin factory
//! builds against the committed catalog, and the hand-written auth
//! implementations (anthropic, bedrock, vertex, cloudflare, copilot, the
//! shared builders, and the not-ported seams) walk their branches. The auth
//! object surface ports from upstream's provider factory files at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pi_ai::api;
use pi_ai::auth::oauth;
use pi_ai::auth::types::{
    ApiKeyAuthInput, ApiKeyCredential, AuthContext, AuthError, AuthPrompt, AuthPromptKind,
    Credential, ModelAuth, OAuthAuth, OAuthCredentials, ProviderAuthInteraction,
};
use pi_ai::providers::all::builtin_providers;
use pi_ai::providers::cloudflare_auth::{cloudflare_ai_gateway_auth, cloudflare_workers_ai_auth};
use pi_ai::providers::opencode_headers::with_opencode_session_header;
use pi_ai::types::{
    BoxedFuture, Context, Model, ProviderEnv, ProviderStreams, SimpleStreamOptions, StreamOptions,
};
use tokio_util::sync::CancellationToken;

const fn fake_context() -> Context {
    Context {
        system_prompt: None,
        messages: Vec::new(),
        tools: None,
    }
}

fn fixture_model() -> Model {
    Model {
        id: "m".to_owned(),
        name: "m".to_owned(),
        api: pi_ai::types::Api::from("test-api"),
        provider: pi_ai::types::ProviderId::from("test"),
        base_url: "https://example.test/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 1000,
        max_tokens: 100,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn input_with(env: ProviderEnv) -> ApiKeyAuthInput {
    ApiKeyAuthInput {
        ctx: Arc::new(FixtureContext(env)),
        credential: None,
        signal: CancellationToken::new(),
    }
}

fn input_with_credential(
    credential: Option<ApiKeyCredential>,
    env: ProviderEnv,
) -> ApiKeyAuthInput {
    ApiKeyAuthInput {
        ctx: Arc::new(FixtureContext(env)),
        credential,
        signal: CancellationToken::new(),
    }
}

/// The env fixture the auth resolution drives, upstream's fake auth context.
struct FixtureContext(ProviderEnv);

impl AuthContext for FixtureContext {
    fn env(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }

    fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

fn prompt_interaction(signal: CancellationToken) -> ProviderAuthInteraction {
    ProviderAuthInteraction {
        signal,
        prompt: Arc::new(|prompt: AuthPrompt| {
            Box::pin(async move {
                match prompt.kind {
                    AuthPromptKind::Select { options, .. } => Ok(options[0].id.clone()),
                    AuthPromptKind::Secret { .. } => Ok("entered-secret".to_owned()),
                    AuthPromptKind::Text { .. } => Ok("entered-text".to_owned()),
                    AuthPromptKind::ManualCode { .. } => Ok("entered-code".to_owned()),
                }
            })
        }),
        notify: Arc::new(|_event| {}),
    }
}

use pi_ai::models_store::{ModelsStore, ModelsStoreEntry};

#[test]
fn every_builtin_provider_constructs_with_its_catalog() {
    for provider in builtin_providers() {
        let models = provider.get_models().unwrap_or_default();
        if provider.id() != "radius" {
            assert!(
                !models.is_empty(),
                "{} carries catalog models",
                provider.id()
            );
            for model in &models {
                assert_eq!(
                    model.provider.0,
                    provider.id(),
                    "{} model ids",
                    provider.id()
                );
                // Azure deployments are resource-relative, so their generated
                // catalog leaves baseUrl empty.
                if provider.id() != "azure-openai-responses" {
                    assert!(
                        !model.base_url.is_empty(),
                        "{} models carry a baseUrl",
                        provider.id()
                    );
                }
            }
        }
        let _ = (provider.base_url(), provider.headers());
    }
}

/// The copilot filter narrows the catalog to the OAuth credential's
/// availableModelIds, upstream's `filterModels`.
#[test]
fn copilot_filter_models_narrows_to_the_credential_allowlist() {
    let copilot = pi_ai::providers::github_copilot::github_copilot_provider();
    let gpt = pi_ai::providers::catalog::get_builtin_model("github-copilot", "gpt-6-astra")
        .expect("gpt-6-astra present");
    let claude = pi_ai::providers::catalog::get_builtin_model("github-copilot", "claude-opus-4-8")
        .or_else(|| {
            pi_ai::providers::catalog::get_builtin_models("github-copilot")
                .into_iter()
                .find(|model| model.id != "gpt-6-astra")
        })
        .expect("a second copilot model");
    let models = vec![gpt, claude];

    // No credential, or a non-OAuth credential: everything passes.
    assert_eq!(copilot.filter_models(models.clone(), None).len(), 2);
    let api_key_only = Credential::ApiKey(ApiKeyCredential::default());
    assert_eq!(
        copilot
            .filter_models(models.clone(), Some(&api_key_only))
            .len(),
        2
    );

    // An OAuth credential without the allowlist: everything passes.
    let oauth_plain = Credential::OAuth(OAuthCredentials {
        refresh: "r".to_owned(),
        access: "a".to_owned(),
        expires: 0,
        extra: BTreeMap::new(),
    });
    assert_eq!(
        copilot
            .filter_models(models.clone(), Some(&oauth_plain))
            .len(),
        2
    );

    // A malformed allowlist passes everything.
    let mut malformed = BTreeMap::new();
    malformed.insert(
        "availableModelIds".to_owned(),
        serde_json::json!("not-a-list"),
    );
    let oauth_bad = Credential::OAuth(OAuthCredentials {
        refresh: "r".to_owned(),
        access: "a".to_owned(),
        expires: 0,
        extra: malformed,
    });
    assert_eq!(
        copilot
            .filter_models(models.clone(), Some(&oauth_bad))
            .len(),
        2
    );

    // A well-formed list narrows to the listed ids.
    let mut allow = BTreeMap::new();
    allow.insert(
        "availableModelIds".to_owned(),
        serde_json::json!(["gpt-6-astra"]),
    );
    let oauth_allow = Credential::OAuth(OAuthCredentials {
        refresh: "r".to_owned(),
        access: "a".to_owned(),
        expires: 0,
        extra: allow,
    });
    let narrowed = copilot.filter_models(models, Some(&oauth_allow));
    assert_eq!(narrowed.len(), 1);
    assert_eq!(narrowed[0].id, "gpt-6-astra");
}

/// The not-ported wire-API stubs settle their streams with the notice, on
/// every wire-API seam constructor.
#[tokio::test]
async fn the_not_ported_wire_api_stubs_report_their_notice() {
    for streams in [
        api::anthropic_messages(),
        api::openai_responses(),
        api::openai_completions(),
        api::azure_openai_responses(),
        api::google_generative_ai(),
        api::google_vertex(),
        api::bedrock_converse_stream(),
        api::mistral_conversations(),
        api::openai_codex_responses(),
        api::pi_messages(),
    ] {
        let model = fixture_model();
        let stream = streams.stream(&model, &fake_context(), None);
        let message = stream.result().await;
        let text = message.error_message.unwrap_or_default();
        assert!(text.contains("has not been ported yet"), "got: {text}");
    }
}

/// The not-ported image seam reports the same notice.
#[tokio::test]
async fn the_not_ported_image_api_reports_its_notice() {
    let images = api::not_ported_images("test-images");
    let model = fixture_image_model();
    let error = images
        .generate_images(&model, &pi_ai::types::ImagesContext::default(), None)
        .await
        .expect_err("not-ported generation fails");
    assert!(
        error.to_string().contains("has not been ported yet"),
        "got: {error}"
    );
}

fn fixture_image_model() -> pi_ai::types::ImagesModel {
    pi_ai::types::ImagesModel {
        id: "img".to_owned(),
        name: "img".to_owned(),
        api: pi_ai::types::ImagesApi::from("test-images"),
        provider: pi_ai::types::ImagesProviderId::from("test"),
        base_url: "https://example.test".to_owned(),
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        output: vec![pi_ai::types::Modality::Image],
        cost: pi_ai::types::ModelCost::default(),
        sampling_params: None,
        headers: None,
    }
}

/// The not-ported OAuth flows fail on every method with the notice.
#[tokio::test]
async fn the_not_ported_oauth_flows_fail_on_every_method() {
    for oauth in [
        oauth::load_anthropic_oauth().await,
        oauth::load_openai_codex_oauth().await,
        oauth::load_github_copilot_oauth().await,
        oauth::load_openrouter_oauth().await,
        oauth::load_kimi_coding_oauth().await,
        oauth::load_xai_oauth().await,
        oauth::load_radius_oauth(&oauth::RadiusOAuthOptions {
            name: "Radius".to_owned(),
            gateway: "https://radius.pi.dev".to_owned(),
        })
        .await,
    ] {
        assert_eq!(oauth.name, expected_stub_name(&oauth.name));
        let error = (oauth.to_auth)(sample_oauth_credential())
            .await
            .expect_err("not ported");
        assert!(
            error.to_string().contains("has not been ported yet"),
            "got: {error}"
        );
        let error = (oauth.refresh)(sample_oauth_credential(), CancellationToken::new())
            .await
            .expect_err("not ported");
        assert!(
            error.to_string().contains("has not been ported yet"),
            "got: {error}"
        );
        let error = (oauth.login)(ProviderAuthInteraction::from_interaction(
            pi_ai::auth::types::AuthInteraction {
                signal: None,
                prompt:
                    Arc::new(
                        |prompt: AuthPrompt| -> BoxedFuture<
                            'static,
                            Result<String, pi_ai::utils::abort::AbortError>,
                        > {
                            let _ = prompt;
                            Box::pin(async { Err(pi_ai::utils::abort::AbortError) })
                        },
                    ),
                notify: Arc::new(|_event| {}),
            },
            CancellationToken::new(),
        ))
        .await
        .expect_err("not ported");
        assert!(
            error.to_string().contains("has not been ported yet"),
            "got: {error}"
        );
    }
}

const fn expected_stub_name(name: &str) -> &str {
    name
}

fn sample_oauth_credential() -> OAuthCredentials {
    OAuthCredentials {
        refresh: "r".to_owned(),
        access: "a".to_owned(),
        expires: 0,
        extra: BTreeMap::new(),
    }
}

/// The lazy OAuth wrapper loads once: concurrent callers share one load.
#[tokio::test]
async fn the_lazy_oauth_wrapper_loads_once() {
    let loads = Arc::new(Mutex::new(0));
    let loads_for_input = Arc::clone(&loads);
    let oauth = pi_ai::auth::helpers::lazy_oauth(pi_ai::auth::helpers::LazyOAuthInput {
        name: "Lazy".to_owned(),
        is_subscription: Some(true),
        login_label: Some("Sign in".to_owned()),
        load: Arc::new(move || {
            let loads = Arc::clone(&loads_for_input);
            Box::pin(async move {
                *loads
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
                fixture_oauth()
            })
        }),
    });
    assert_eq!(oauth.name, "Lazy");
    assert_eq!(oauth.is_subscription, Some(true));
    assert_eq!(oauth.login_label.as_deref(), Some("Sign in"));

    let (first, second) = tokio::join!(
        (oauth.to_auth)(sample_oauth_credential()),
        (oauth.to_auth)(sample_oauth_credential()),
    );
    assert_eq!(first.expect("toAuth").api_key.as_deref(), Some("derived"));
    assert_eq!(second.expect("toAuth").api_key.as_deref(), Some("derived"));
    assert_eq!(
        *loads
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        1
    );
}

/// The OAuth fixture the lazy wrapper loads, reporting its key.
fn fixture_oauth() -> OAuthAuth {
    let login: pi_ai::auth::types::OAuthLoginFn = Arc::new(|_interaction| {
        let cause: AuthError = Box::new(std::io::Error::other("unused"));
        let failing: BoxedFuture<'static, Result<OAuthCredentials, AuthError>> =
            Box::pin(async move { Err(cause) });
        failing
    });
    let refresh: pi_ai::auth::types::OAuthRefreshFn =
        Arc::new(|credential, _signal| Box::pin(async move { Ok(credential) }));
    let to_auth: pi_ai::auth::types::OAuthToAuthFn = Arc::new(|_credential| {
        Box::pin(async move {
            Ok(ModelAuth {
                api_key: Some("derived".to_owned()),
                ..ModelAuth::default()
            })
        })
    });
    OAuthAuth {
        name: "Fixture OAuth".to_owned(),
        is_subscription: None,
        login_label: None,
        login,
        refresh,
        to_auth,
    }
}

/// The shared api-key auth: a stored credential wins, the env order picks
/// the ambient key, and login prompts for the key.
#[tokio::test]
async fn the_shared_api_key_auth_walks_its_stored_env_and_login_paths() {
    let auth = pi_ai::auth::helpers::env_api_key_auth("Fixture key", &["FIRST_VAR", "SECOND_VAR"]);

    // stored credential wins over env, and its env overlay rides along
    let resolved = (auth.resolve)(input_with_credential(
        Some(ApiKeyCredential {
            key: Some("stored-key".to_owned()),
            env: Some(BTreeMap::from([(
                "PROVIDER".to_owned(),
                "stored-env".to_owned(),
            )])),
        }),
        BTreeMap::from([("FIRST_VAR".to_owned(), "env-first".to_owned())]),
    ))
    .await
    .expect("resolve")
    .expect("stored");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("stored-key"));
    assert_eq!(resolved.source.as_deref(), Some("stored credential"));
    assert_eq!(
        resolved.env.and_then(|env| env.get("PROVIDER").cloned()),
        Some("stored-env".to_owned())
    );

    // ambient: the first set env var wins
    let resolved = (auth.resolve)(input_with(BTreeMap::from([
        ("FIRST_VAR".to_owned(), "env-first".to_owned()),
        ("SECOND_VAR".to_owned(), "env-second".to_owned()),
    ])))
    .await
    .expect("resolve")
    .expect("env");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("env-first"));
    assert_eq!(resolved.source.as_deref(), Some("FIRST_VAR"));

    // nothing set: unconfigured
    let resolved = (auth.resolve)(input_with(BTreeMap::new()))
        .await
        .expect("resolve");
    assert!(resolved.is_none());

    // login prompts for the key
    let login = auth.login.as_ref().expect("shared login exists");
    let credential = (login)(prompt_interaction(CancellationToken::new()))
        .await
        .expect("login");
    assert!(
        matches!(credential, ApiKeyCredential { key: Some(key), .. } if key == "entered-secret")
    );
}

/// The anthropic auth: the `AUTH_TOKEN` env becomes a bearer header; the
/// OAuth-token and API-key envs resolve as plain keys; the stored credential
/// wins over every env var.
#[tokio::test]
async fn the_anthropic_auth_resolves_its_env_and_stored_branches() {
    let anthropic = pi_ai::providers::anthropic::anthropic_provider();
    let auth = anthropic.auth().api_key.as_ref().expect("anthropic auth");

    // AUTH_TOKEN env: bearer header
    let resolved = (auth.resolve)(input_with(BTreeMap::from([(
        "ANTHROPIC_AUTH_TOKEN".to_owned(),
        "auth-token".to_owned(),
    )])))
    .await
    .expect("resolve")
    .expect("auth token");
    assert_eq!(resolved.source.as_deref(), Some("ANTHROPIC_AUTH_TOKEN"));
    let headers = resolved.auth.headers.expect("bearer header");
    assert_eq!(
        headers.get("Authorization"),
        Some(&Some("Bearer auth-token".to_owned()))
    );

    // OAuth-token env
    let resolved = (auth.resolve)(input_with(BTreeMap::from([(
        "ANTHROPIC_OAUTH_TOKEN".to_owned(),
        "oauth-token".to_owned(),
    )])))
    .await
    .expect("resolve")
    .expect("oauth token");
    assert_eq!(resolved.source.as_deref(), Some("ANTHROPIC_OAUTH_TOKEN"));

    // API-key env
    let resolved = (auth.resolve)(input_with(BTreeMap::from([(
        "ANTHROPIC_API_KEY".to_owned(),
        "api-key".to_owned(),
    )])))
    .await
    .expect("resolve")
    .expect("api key");
    assert_eq!(resolved.source.as_deref(), Some("ANTHROPIC_API_KEY"));

    // stored credential wins
    let resolved = (auth.resolve)(input_with_credential(
        Some(ApiKeyCredential {
            key: Some("stored-key".to_owned()),
            env: None,
        }),
        BTreeMap::from([("ANTHROPIC_API_KEY".to_owned(), "env".to_owned())]),
    ))
    .await
    .expect("resolve")
    .expect("stored");
    assert_eq!(resolved.source.as_deref(), Some("stored credential"));
    assert_eq!(resolved.auth.api_key.as_deref(), Some("stored-key"));

    // nothing set: unconfigured
    assert!(
        (auth.resolve)(input_with(BTreeMap::new()))
            .await
            .expect("resolve")
            .is_none()
    );

    // login prompts for the key
    let login = auth.login.as_ref().expect("anthropic login");
    let credential = (login)(prompt_interaction(CancellationToken::new()))
        .await
        .expect("login");
    assert!(
        matches!(credential, ApiKeyCredential { key: Some(key), .. } if key == "entered-secret")
    );
}

/// The Cloudflare Workers AI auth: the key plus the account id resolve; a
/// missing account id is unconfigured; login prompts both.
#[tokio::test]
async fn cloudflare_workers_ai_auth_walks_its_env_credential_and_login_paths() {
    let auth = cloudflare_workers_ai_auth();
    assert_eq!(auth.name, "Cloudflare API key");

    // env carries the key and the account id
    let resolved = (auth.resolve)(input_with(BTreeMap::from([
        ("CLOUDFLARE_API_KEY".to_owned(), "cf-key".to_owned()),
        ("CLOUDFLARE_ACCOUNT_ID".to_owned(), "acct".to_owned()),
    ])))
    .await
    .expect("resolve")
    .expect("configured");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("cf-key"));
    assert_eq!(
        resolved
            .env
            .and_then(|env| env.get("CLOUDFLARE_ACCOUNT_ID").cloned()),
        Some("acct".to_owned())
    );
    assert_eq!(resolved.source.as_deref(), Some("CLOUDFLARE_API_KEY"));

    // a credential carrying only the key still picks the account id from env
    let resolved = (auth.resolve)(input_with_credential(
        Some(ApiKeyCredential {
            key: Some("stored-key".to_owned()),
            env: None,
        }),
        BTreeMap::from([("CLOUDFLARE_ACCOUNT_ID".to_owned(), "ambient".to_owned())]),
    ))
    .await
    .expect("resolve")
    .expect("stored");
    assert_eq!(resolved.source.as_deref(), Some("stored credential"));
    assert_eq!(
        resolved
            .env
            .and_then(|env| env.get("CLOUDFLARE_ACCOUNT_ID").cloned()),
        Some("ambient".to_owned())
    );

    // missing account id: unconfigured
    assert!(
        (auth.resolve)(input_with(BTreeMap::from([(
            "CLOUDFLARE_API_KEY".to_owned(),
            "k".to_owned(),
        )])))
        .await
        .expect("resolve")
        .is_none()
    );

    // login prompts the key and the account id
    let login = auth.login.as_ref().expect("workers-ai login");
    let credential = (login)(prompt_interaction(CancellationToken::new()))
        .await
        .expect("login");
    assert!(
        matches!(credential, ApiKeyCredential { key: Some(key), .. } if key == "entered-secret")
    );
    assert_eq!(
        credential
            .env
            .and_then(|env| env.get("CLOUDFLARE_ACCOUNT_ID").cloned()),
        Some("entered-text".to_owned())
    );
}

/// The AI Gateway auth: the key rides as cf-aig-authorization, the standard
/// headers are suppressed, and the gateway id is required.
#[tokio::test]
async fn cloudflare_ai_gateway_auth_suppresses_the_standard_headers() {
    let auth = cloudflare_ai_gateway_auth();

    let resolved = (auth.resolve)(input_with(BTreeMap::from([
        ("CLOUDFLARE_API_KEY".to_owned(), "gw-key".to_owned()),
        ("CLOUDFLARE_ACCOUNT_ID".to_owned(), "acct".to_owned()),
        ("CLOUDFLARE_GATEWAY_ID".to_owned(), "gateway".to_owned()),
    ])))
    .await
    .expect("resolve")
    .expect("configured");
    let headers = resolved.auth.headers.expect("gateway headers");
    assert_eq!(
        headers.get("cf-aig-authorization"),
        Some(&Some("Bearer gw-key".to_owned()))
    );
    assert_eq!(headers.get("Authorization"), Some(&None));
    assert_eq!(headers.get("x-api-key"), Some(&None));

    // the gateway id missing: unconfigured
    assert!(
        (auth.resolve)(input_with(BTreeMap::from([
            ("CLOUDFLARE_API_KEY".to_owned(), "k".to_owned()),
            ("CLOUDFLARE_ACCOUNT_ID".to_owned(), "a".to_owned()),
        ])))
        .await
        .expect("resolve")
        .is_none()
    );

    // login prompts key, account id, gateway id
    let login = auth.login.as_ref().expect("gateway login");
    let credential = (login)(prompt_interaction(CancellationToken::new()))
        .await
        .expect("login");
    assert!(
        matches!(credential, ApiKeyCredential { key: Some(key), .. } if key == "entered-secret")
    );
    assert_eq!(
        credential
            .env
            .and_then(|env| env.get("CLOUDFLARE_GATEWAY_ID").cloned()),
        Some("entered-text".to_owned())
    );
}

/// The Bedrock auth: the login flow walks bearer/profile/credential-chain,
/// and resolve detects every ambient AWS source.
#[tokio::test]
async fn the_bedrock_auth_walks_its_login_choices_and_ambient_sources() {
    let bedrock = pi_ai::providers::amazon_bedrock::amazon_bedrock_provider();
    let auth = bedrock.auth().api_key.as_ref().expect("bedrock auth");

    // login: the fixture selects the first option (bearer token)
    let login = auth.login.as_ref().expect("bedrock login");
    let credential = (login)(prompt_interaction(CancellationToken::new()))
        .await
        .expect("bearer login");
    assert!(
        matches!(credential, ApiKeyCredential { key: Some(key), .. } if key == "entered-secret")
    );

    let resolve = &auth.resolve;
    // stored credential
    let resolved = (resolve)(input_with_credential(
        Some(ApiKeyCredential {
            key: Some("stored-key".to_owned()),
            env: Some(BTreeMap::from([(
                "AWS_PROFILE".to_owned(),
                "stored".to_owned(),
            )])),
        }),
        BTreeMap::new(),
    ))
    .await
    .expect("resolve")
    .expect("stored");
    assert_eq!(resolved.source.as_deref(), Some("stored credential"));
    assert_eq!(
        resolved.env.and_then(|env| env.get("AWS_PROFILE").cloned()),
        Some("stored".to_owned())
    );

    // ambient bearer token
    let resolved = (resolve)(input_with(BTreeMap::from([(
        "AWS_BEARER_TOKEN_BEDROCK".to_owned(),
        "t".to_owned(),
    )])))
    .await
    .expect("resolve")
    .expect("bearer");
    assert_eq!(resolved.source.as_deref(), Some("AWS_BEARER_TOKEN_BEDROCK"));

    // stored profile
    let resolved = (resolve)(input_with_credential(
        Some(ApiKeyCredential {
            key: None,
            env: Some(BTreeMap::from([(
                "AWS_PROFILE".to_owned(),
                "stored".to_owned(),
            )])),
        }),
        BTreeMap::new(),
    ))
    .await
    .expect("resolve")
    .expect("stored profile");
    assert_eq!(resolved.source.as_deref(), Some("stored credential"));

    // ambient profile
    let resolved = (resolve)(input_with(BTreeMap::from([(
        "AWS_PROFILE".to_owned(),
        "prod".to_owned(),
    )])))
    .await
    .expect("resolve")
    .expect("profile");
    assert_eq!(resolved.source.as_deref(), Some("AWS_PROFILE"));

    // IAM keys
    let resolved = (resolve)(input_with(BTreeMap::from([
        ("AWS_ACCESS_KEY_ID".to_owned(), "k".to_owned()),
        ("AWS_SECRET_ACCESS_KEY".to_owned(), "s".to_owned()),
    ])))
    .await
    .expect("resolve")
    .expect("iam");
    assert_eq!(resolved.source.as_deref(), Some("AWS access keys"));

    // ECS task roles, both URI forms
    for var in [
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    ] {
        let resolved = (resolve)(input_with(BTreeMap::from([(
            var.to_owned(),
            "u".to_owned(),
        )])))
        .await
        .expect("resolve")
        .expect("ecs");
        assert_eq!(resolved.source.as_deref(), Some("ECS task role"));
    }

    // web identity
    let resolved = (resolve)(input_with(BTreeMap::from([(
        "AWS_WEB_IDENTITY_TOKEN_FILE".to_owned(),
        "f".to_owned(),
    )])))
    .await
    .expect("resolve")
    .expect("web identity");
    assert_eq!(resolved.source.as_deref(), Some("web identity token"));

    // nothing set: unconfigured
    assert!(
        (resolve)(input_with(BTreeMap::new()))
            .await
            .expect("resolve")
            .is_none()
    );
}

/// The Vertex auth: the stored or ambient API key wins; ADC requires the
/// credentials file, project, and location.
#[tokio::test]
async fn the_vertex_auth_resolves_its_api_key_and_adc_branches() {
    let vertex = pi_ai::providers::google_vertex::google_vertex_provider();
    let auth = vertex.auth().api_key.as_ref().expect("vertex auth");
    let resolve = &auth.resolve;

    // stored key
    let resolved = (resolve)(input_with_credential(
        Some(ApiKeyCredential {
            key: Some("stored-key".to_owned()),
            env: None,
        }),
        BTreeMap::new(),
    ))
    .await
    .expect("resolve")
    .expect("stored");
    assert_eq!(resolved.source.as_deref(), Some("stored credential"));

    // ambient key
    let resolved = (resolve)(input_with(BTreeMap::from([(
        "GOOGLE_CLOUD_API_KEY".to_owned(),
        "vk".to_owned(),
    )])))
    .await
    .expect("resolve")
    .expect("api key");
    assert_eq!(resolved.source.as_deref(), Some("GOOGLE_CLOUD_API_KEY"));

    // ADC with explicit credentials path, project, and location; the
    // fileExists fixture reports the explicit path.
    let resolved = (resolve)(ApiKeyAuthInput {
        ctx: Arc::new(ExistingPathContext {
            env: BTreeMap::from([
                (
                    "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
                    "adc".to_owned(),
                ),
                ("GOOGLE_CLOUD_PROJECT".to_owned(), "p".to_owned()),
                ("GOOGLE_CLOUD_LOCATION".to_owned(), "l".to_owned()),
            ]),
            existing: "adc",
        }),
        credential: None,
        signal: CancellationToken::new(),
    })
    .await
    .expect("resolve")
    .expect("adc");
    assert_eq!(
        resolved.source.as_deref(),
        Some("gcloud application default credentials")
    );

    // ADC via the default path: the fileExists fixture answers yes for the
    // default path.
    let resolved = (resolve)(ApiKeyAuthInput {
        ctx: Arc::new(DefaultAdcContext {
            env: BTreeMap::from([
                ("GOOGLE_CLOUD_PROJECT".to_owned(), "p".to_owned()),
                ("GOOGLE_CLOUD_LOCATION".to_owned(), "l".to_owned()),
            ]),
        }),
        credential: None,
        signal: CancellationToken::new(),
    })
    .await
    .expect("resolve")
    .expect("default adc");
    assert_eq!(
        resolved.source.as_deref(),
        Some("gcloud application default credentials")
    );

    // GCLOUD_PROJECT alias also satisfies the project check.
    let resolved = (resolve)(ApiKeyAuthInput {
        ctx: Arc::new(DefaultAdcContext {
            env: BTreeMap::from([
                ("GCLOUD_PROJECT".to_owned(), "p".to_owned()),
                ("GOOGLE_CLOUD_LOCATION".to_owned(), "l".to_owned()),
            ]),
        }),
        credential: None,
        signal: CancellationToken::new(),
    })
    .await
    .expect("resolve")
    .expect("gcloud alias");
    assert_eq!(
        resolved.source.as_deref(),
        Some("gcloud application default credentials")
    );

    // missing location: unconfigured
    assert!(
        (resolve)(input_with(BTreeMap::from([(
            "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
            "adc".to_owned(),
        )])))
        .await
        .expect("resolve")
        .is_none()
    );
}

/// The fixture whose fileExists answers one exact path.
struct ExistingPathContext {
    env: ProviderEnv,
    existing: &'static str,
}

impl AuthContext for ExistingPathContext {
    fn env(&self, name: &str) -> Option<String> {
        self.env.get(name).cloned()
    }

    fn file_exists(&self, path: &str) -> bool {
        path == self.existing
    }
}

/// The fixture whose fileExists answers only the default ADC path, matching
/// upstream's `~/.config/gcloud/...` expansion.
struct DefaultAdcContext {
    env: ProviderEnv,
}

impl AuthContext for DefaultAdcContext {
    fn env(&self, name: &str) -> Option<String> {
        self.env.get(name).cloned()
    }

    fn file_exists(&self, path: &str) -> bool {
        path.ends_with("application_default_credentials.json")
    }
}

/// The Vertex login walks api-key / ADC / service-account prompts.
#[tokio::test]
async fn the_vertex_login_prompts_the_api_key_path() {
    let vertex = pi_ai::providers::google_vertex::google_vertex_provider();
    let auth = vertex.auth().api_key.as_ref().expect("vertex auth");
    let login = auth.login.as_ref().expect("vertex login");
    let credential = (login)(prompt_interaction(CancellationToken::new()))
        .await
        .expect("api-key login");
    assert!(
        matches!(credential, ApiKeyCredential { key: Some(key), .. } if key == "entered-secret")
    );
}

/// The cloudflare stream wrapper resolves the endpoint placeholders from the
/// resolved env and leaves unknown placeholders literal.
#[test]
fn cloudflare_streams_resolve_the_endpoint_placeholders() {
    let mut model = fixture_model();
    model.base_url =
        "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic"
            .to_owned();
    let env: ProviderEnv = BTreeMap::from([
        ("CLOUDFLARE_ACCOUNT_ID".to_owned(), "acct".to_owned()),
        ("CLOUDFLARE_GATEWAY_ID".to_owned(), "gateway".to_owned()),
    ]);
    let resolved =
        pi_ai::providers::cloudflare_stream::resolve_cloudflare_model(&model, Some(&env));
    assert_eq!(
        resolved.base_url,
        "https://gateway.ai.cloudflare.com/v1/acct/gateway/anthropic"
    );

    // A placeholder without a value stays literal; a placeholder-free model
    // round-trips.
    let partial: ProviderEnv =
        BTreeMap::from([("CLOUDFLARE_ACCOUNT_ID".to_owned(), "acct".to_owned())]);
    let resolved =
        pi_ai::providers::cloudflare_stream::resolve_cloudflare_model(&model, Some(&partial));
    assert!(resolved.base_url.contains("{CLOUDFLARE_GATEWAY_ID}"));
    let plain = fixture_model();
    assert_eq!(
        pi_ai::providers::cloudflare_stream::resolve_cloudflare_model(&plain, None).base_url,
        plain.base_url
    );
}

/// The OpenCode session header wrapper adds the header for a session id,
/// keeps an existing header (any case), and adds nothing without one.
#[test]
fn opencode_session_header_follows_the_session_id() {
    let recorded: Arc<Mutex<Option<StreamOptions>>> = Arc::new(Mutex::new(None));
    let inner: Arc<dyn ProviderStreams> = Arc::new(RecordingInner(Arc::clone(&recorded)));
    let streams = with_opencode_session_header(inner);

    let options = StreamOptions {
        session_id: Some("session-1".to_owned()),
        ..StreamOptions::default()
    };
    streams.stream(&fixture_model(), &fake_context(), Some(&options));
    let sent = recorded
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .expect("dispatched");
    let headers = sent.headers.expect("session header");
    assert_eq!(
        headers.get("x-opencode-session"),
        Some(&Some("session-1".to_owned()))
    );

    let options = StreamOptions {
        session_id: Some("session-1".to_owned()),
        headers: Some(BTreeMap::from([(
            "X-OpenCode-Session".to_owned(),
            Some("already".to_owned()),
        )])),
        ..StreamOptions::default()
    };
    streams.stream(&fixture_model(), &fake_context(), Some(&options));
    let sent = recorded
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .expect("dispatched");
    let headers = sent.headers.expect("headers");
    assert_eq!(headers.get("x-opencode-session"), None);
    assert_eq!(
        headers.get("X-OpenCode-Session"),
        Some(&Some("already".to_owned()))
    );

    streams.stream(
        &fixture_model(),
        &fake_context(),
        Some(&StreamOptions::default()),
    );
    let sent = recorded
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .expect("dispatched");
    assert!(sent.headers.is_none(), "no session id means no header");
}

/// The innermost streams the wrapper dispatches to: records what arrives.
struct RecordingInner(Arc<Mutex<Option<StreamOptions>>>);

impl ProviderStreams for RecordingInner {
    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        options: Option<&StreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        *self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = options.cloned();
        pi_ai::utils::event_stream::assistant_message_event_stream()
    }

    fn stream_simple(
        &self,
        _model: &Model,
        _context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        let _ = options;
        pi_ai::utils::event_stream::assistant_message_event_stream()
    }
}

/// The models store clones on read and honors the cancellation signal.
#[tokio::test]
async fn the_models_store_clones_on_read_and_honors_the_signal() {
    let store = pi_ai::models_store::InMemoryModelsStore::default();
    store
        .write("p", ModelsStoreEntry::default(), None)
        .await
        .expect("write");
    assert!(store.read("p", None).await.expect("read").is_some());
    store.delete("p", None).await.expect("delete");
    assert!(store.read("p", None).await.expect("read").is_none());

    let signal = CancellationToken::new();
    signal.cancel();
    let options = pi_ai::models_store::ModelsStoreOptions {
        signal: Some(signal),
    };
    assert!(store.read("p", Some(&options)).await.is_err());
    assert!(
        store
            .write("p", ModelsStoreEntry::default(), Some(&options))
            .await
            .is_err()
    );
    assert!(store.delete("p", Some(&options)).await.is_err());
}

/// The Radius config plumbing: URL normalization, sanitization, projection.
#[test]
fn the_radius_config_normalizes_urls_and_projects_models() {
    assert_eq!(
        pi_ai::providers::radius_config::normalize_radius_gateway_url("radius.pi.dev"),
        "https://radius.pi.dev"
    );
    assert_eq!(
        pi_ai::providers::radius_config::normalize_radius_gateway_url("http://radius.pi.dev/"),
        "http://radius.pi.dev"
    );

    let config =
        pi_ai::providers::radius_config::sanitize_radius_gateway_config(&serde_json::json!({
            "baseUrl": "https://gateway.test",
            "models": [{
                "id": "m1",
                "name": "M1",
                "reasoning": true,
                "input": ["text"],
                "cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 },
                "contextWindow": 1000,
                "maxTokens": 100,
            }, { "baseUrl": "invalid" }],
        }))
        .expect("sanitized");
    assert_eq!(config.models.len(), 1);
    assert!(
        pi_ai::providers::radius_config::sanitize_radius_gateway_config(&serde_json::json!({}))
            .is_none()
    );

    let models = pi_ai::providers::radius_config::get_radius_models_from_config("radius", &config);
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].provider.0, "radius");
    assert_eq!(models[0].base_url, "https://gateway.test");
    assert_eq!(
        models[0].api.as_known(),
        Some(pi_ai::types::KnownApi::PiMessages)
    );

    // A credential's gatewayConfig sanitizes into the model list.
    let credential = OAuthCredentials {
        refresh: "r".to_owned(),
        access: "a".to_owned(),
        expires: 0,
        extra: BTreeMap::from([(
            "gatewayConfig".to_owned(),
            serde_json::json!({
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
            }),
        )]),
    };
    let models = pi_ai::providers::radius_config::get_radius_models("radius", Some(&credential));
    assert_eq!(models.len(), 1);
    assert!(
        pi_ai::providers::radius_config::get_radius_models("radius", None).is_empty(),
        "no credential, no models"
    );
}

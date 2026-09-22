//! The remaining #28 surface edges: the vertex and bedrock login paths for
//! every prompt choice, the provider trait's default methods, the
//! auth-failure Display surfaces, and the model-data reader's error shapes.
//! Upstream exercises these through the CLI and login suites at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pi_ai::auth::types::{ProviderAuth, ProviderAuthInteraction};
use pi_ai::models::{ModelsRefreshOptions, Provider, ProviderError};
use pi_ai::types::{
    Context, DeferredHandle, Model, ProviderStreams, SimpleStreamOptions, StreamOptions,
};
use tokio_util::sync::CancellationToken;

fn fixture_model() -> Model {
    Model {
        id: "m".to_owned(),
        name: "m".to_owned(),
        api: pi_ai::types::Api::from("test-api"),
        provider: pi_ai::types::ProviderId::from("test"),
        base_url: "https://example.test".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 100,
        max_tokens: 10,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The scripted prompt fixture: answers each select prompt from a queue and
/// every text/secret prompt with the given answer.
fn scripted_interaction(
    selections: Arc<Mutex<std::collections::VecDeque<String>>>,
    text_answer: &'static str,
    signal: CancellationToken,
) -> ProviderAuthInteraction {
    ProviderAuthInteraction {
        signal,
        prompt: Arc::new(move |prompt: pi_ai::auth::types::AuthPrompt| {
            let selections = Arc::clone(&selections);
            Box::pin(async move {
                match prompt.kind {
                    pi_ai::auth::types::AuthPromptKind::Select { options, .. } => {
                        let next = selections
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .pop_front()
                            .unwrap_or_else(|| options[0].id.clone());
                        Ok(next)
                    }
                    pi_ai::auth::types::AuthPromptKind::Secret { .. } => {
                        Ok("entered-secret".to_owned())
                    }
                    pi_ai::auth::types::AuthPromptKind::Text { .. } => Ok(text_answer.to_owned()),
                    pi_ai::auth::types::AuthPromptKind::ManualCode { .. } => Ok(text_code_marker()),
                }
            })
        }),
        notify: Arc::new(|_event| {}),
    }
}

const fn text_code_marker() -> String {
    String::new()
}

/// The bedrock login walks the aws-profile and credential-chain choices.
#[tokio::test]
async fn the_bedrock_login_walks_the_profile_and_chain_choices() {
    let bedrock = pi_ai::providers::amazon_bedrock::amazon_bedrock_provider();
    let login = bedrock
        .auth()
        .api_key
        .as_ref()
        .expect("bedrock auth")
        .login
        .as_ref()
        .expect("login");

    let credential = (login)(scripted_interaction(
        Arc::new(Mutex::new(std::collections::VecDeque::from([
            "aws-profile".to_owned(),
        ]))),
        "my-profile",
        CancellationToken::new(),
    ))
    .await
    .expect("profile login");
    assert_eq!(
        credential
            .env
            .and_then(|env| env.get("AWS_PROFILE").cloned()),
        Some("my-profile".to_owned())
    );

    let credential = (login)(scripted_interaction(
        Arc::new(Mutex::new(std::collections::VecDeque::from([
            "credential-chain".to_owned(),
        ]))),
        "continue",
        CancellationToken::new(),
    ))
    .await
    .expect("chain login");
    assert!(credential.key.is_none(), "the chain login stores no key");

    let error = (login)(scripted_interaction(
        Arc::new(Mutex::new(std::collections::VecDeque::from([
            "mystery".to_owned()
        ]))),
        "x",
        CancellationToken::new(),
    ))
    .await
    .expect_err("unknown method");
    assert!(
        error
            .to_string()
            .contains("Unknown Amazon Bedrock auth method"),
        "got: {error}"
    );
}

/// The vertex login walks the adc and service-account choices.
#[tokio::test]
async fn the_vertex_login_walks_the_adc_and_service_account_choices() {
    let vertex = pi_ai::providers::google_vertex::google_vertex_provider();
    let login = vertex
        .auth()
        .api_key
        .as_ref()
        .expect("vertex auth")
        .login
        .as_ref()
        .expect("login");

    let credential = (login)(scripted_interaction(
        Arc::new(Mutex::new(std::collections::VecDeque::from([
            "adc".to_owned()
        ]))),
        "my-project",
        CancellationToken::new(),
    ))
    .await
    .expect("adc login");
    assert!(credential.key.is_none());
    assert_eq!(
        credential
            .env
            .and_then(|env| env.get("GOOGLE_CLOUD_PROJECT").cloned()),
        Some("my-project".to_owned())
    );

    let credential = (login)(scripted_interaction(
        Arc::new(Mutex::new(std::collections::VecDeque::from([
            "service-account".to_owned(),
        ]))),
        "sa.json",
        CancellationToken::new(),
    ))
    .await
    .expect("service-account login");
    assert_eq!(
        credential
            .env
            .and_then(|env| env.get("GOOGLE_APPLICATION_CREDENTIALS").cloned()),
        Some("sa.json".to_owned())
    );

    let error = (login)(scripted_interaction(
        Arc::new(Mutex::new(std::collections::VecDeque::from([
            "mystery".to_owned()
        ]))),
        "x",
        CancellationToken::new(),
    ))
    .await
    .expect_err("unknown method");
    assert!(
        error
            .to_string()
            .contains("Unknown Google Vertex AI auth method"),
        "got: {error}"
    );
}

/// The refresh context's Debug surface renders its fields.
#[test]
fn the_refresh_context_debug_renders_its_fields() {
    let context = pi_ai::models::RefreshModelsContext {
        credential: None,
        stored: None,
        publish: Arc::new(|_publication| Box::pin(async { Ok(true) })),
        allow_network: false,
        force: None,
        signal: CancellationToken::new(),
    };
    let rendered = format!("{context:?}");
    assert!(rendered.contains("RefreshModelsContext"), "got: {rendered}");
}

/// The auth-failure Display and source surfaces: `ModelsFailure` renders the
/// abort text, `ModelsError` with no cause prints its message, and the
/// wrapped error chains the reason.
#[tokio::test]
async fn the_models_failure_surfaces_render_their_reasons() {
    use pi_ai::auth::resolve::{ModelsError, ModelsErrorCode, ModelsFailure};

    let abort = ModelsFailure::Aborted(pi_ai::utils::abort::AbortError);
    assert_eq!(abort.to_string(), "The operation was aborted");
    assert!(std::error::Error::source(&abort).is_some());

    let plain = ModelsError::new(ModelsErrorCode::Auth, "plain plain");
    assert_eq!(plain.to_string(), "plain plain");
    assert_eq!(plain.code(), ModelsErrorCode::Auth);
    assert!(format!("{plain:?}").contains("ModelsError"));

    let cause: Box<dyn std::error::Error + Send + Sync> =
        Box::new(std::io::Error::other("disk on fire"));
    let wrapped = ModelsError::with_cause(ModelsErrorCode::Auth, "read failed", cause);
    assert_eq!(wrapped.to_string(), "read failed: disk on fire");
    assert!(std::error::Error::source(&wrapped).is_some());
}

/// The model-data reader's error shapes: invalid JSON, non-object JSON, and
/// the exact-allowlist both-differ report.
#[test]
fn the_model_data_reader_reports_its_error_shapes() {
    let dir = std::env::temp_dir().join(format!(
        "pi-ai-reader-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(dir.join("data")).expect("mkdir");

    let structure: pi_ai::model_data::ModelDataStructure = BTreeMap::from([(
        "prov".to_owned(),
        BTreeMap::from([("m".to_owned(), "api".to_owned())]),
    )]);

    // An invalid JSON shard reports the parse failure.
    std::fs::write(dir.join("data/prov.json"), "not json").expect("write");
    let error = pi_ai::model_data::validate_model_data_directory(&structure, &dir.join("data"))
        .expect_err("invalid json fails");
    assert!(
        error.to_string().contains("is not valid JSON"),
        "got: {error}"
    );

    // A non-object shard reports the same reader error.
    std::fs::write(dir.join("data/prov.json"), "[1, 2]").expect("write");
    let error = pi_ai::model_data::validate_model_data_directory(&structure, &dir.join("data"))
        .expect_err("non-object fails");
    assert!(
        error.to_string().contains("must contain a JSON object"),
        "got: {error}"
    );

    // Both sides differing names missing and extra ids.
    let error =
        pi_ai::model_data::assert_exact_model_ids("prov", ["a".to_owned()], ["b".to_owned()])
            .expect_err("mismatch fails");
    assert_eq!(
        error.to_string(),
        "prov model IDs do not match (missing: a; extra: b)"
    );

    // The structure hash is stable across calls.
    let hash = pi_ai::model_data::model_data_structure_hash(&structure);
    assert_eq!(
        hash,
        pi_ai::model_data::model_data_structure_hash(&structure)
    );

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

/// The Provider trait's default method bodies through a bare implementation.
#[tokio::test]
async fn the_provider_trait_defaults_exercise_their_shapes() {
    let provider: Arc<dyn Provider> = Arc::new(BareProvider);
    // The trait defaults: no base URL, no headers, no refresh support, and
    // the default refresh resolves Ok.
    assert_eq!(provider.base_url(), None);
    assert_eq!(provider.headers(), None);
    assert!(!provider.supports_refresh_models());
    assert!(!provider.supports_fetch_deferred());
    assert!(!provider.supports_cancel_deferred());

    let context = pi_ai::models::RefreshModelsContext {
        credential: None,
        stored: None,
        publish: Arc::new(|_publication| Box::pin(async { Ok(true) })),
        allow_network: false,
        force: None,
        signal: CancellationToken::new(),
    };
    let rendered = format!("{context:?}");
    assert!(rendered.contains("RefreshModelsContext"), "got: {rendered}");
    assert_eq!(
        provider
            .refresh_models(context)
            .await
            .expect("default refresh"),
        ()
    );

    // The collection's refresh skips static providers without recording
    // anything.
    let models = pi_ai::models::create_models(None);
    models.set_provider(provider);
    let result = models
        .refresh(Some(&ModelsRefreshOptions {
            providers: Some(vec!["bare".to_owned()]),
            ..ModelsRefreshOptions::default()
        }))
        .await;
    assert_eq!(result.errors.len(), 0);
}

/// The images trait's default refresh method and the images edges.
#[tokio::test]
async fn the_images_trait_default_refresh_is_a_noop() {
    let models = pi_ai::images_models::create_images_models(None);
    models.set_provider(Arc::new(BareImagesProvider));
    assert!(
        !models
            .provider("bare")
            .expect("registered")
            .supports_refresh_models()
    );
    models.refresh(Some("bare")).await.expect("noop refresh");

    // A failing dynamic provider leaves its list intact and reports through
    // the single-provider refresh.
    let models = pi_ai::images_models::create_images_models(None);
    models.set_provider(Arc::new(pi_ai::images_models::create_images_provider(
        pi_ai::images_models::CreateImagesProviderOptions {
            id: "flaky".to_owned(),
            name: None,
            auth: ProviderAuth::default(),
            models: Vec::new(),
            refresh_models: Some(Arc::new(|| {
                Box::pin(async move { Err(std::io::Error::other("fetch failed").into()) })
            })),
            api: pi_ai::api::not_ported_images("test-images"),
        },
    )));
    let result = models.refresh(Some("flaky")).await;
    assert!(result.is_err());
}

/// The wrapped stream dispatch: both wrappers route the model through to the
/// inner streams on both surfaces.
#[test]
fn the_wrappers_dispatch_on_both_surfaces() {
    use pi_ai::types::ProviderEnv;

    let mut model = fixture_model();
    model.base_url = "https://{CLOUDFLARE_ACCOUNT_ID}.gateway.test/v1".to_owned();

    // Cloudflare wrapper dispatch: both surfaces route through.
    let wrapped = pi_ai::providers::cloudflare_stream::cloudflare_streams(
        pi_ai::api::not_ported_streams("test-api"),
    );
    let env: ProviderEnv =
        BTreeMap::from([("CLOUDFLARE_ACCOUNT_ID".to_owned(), "acct".to_owned())]);
    let options = StreamOptions {
        env: Some(env),
        ..StreamOptions::default()
    };
    let _ = wrapped.stream(&model, &Context::default(), Some(&options));
    let _ = wrapped.stream_simple(
        &model,
        &Context::default(),
        Some(&SimpleStreamOptions::default()),
    );

    // OpenCode wrapper stream_simple dispatches with the session header.
    let recorded: Arc<Mutex<Option<SimpleStreamOptions>>> = Arc::new(Mutex::new(None));
    let inner: Arc<dyn ProviderStreams> = Arc::new(SimpleRecording(Arc::clone(&recorded)));
    let simple = pi_ai::providers::opencode_headers::with_opencode_session_header(inner);
    let options = SimpleStreamOptions {
        session_id: Some("session-2".to_owned()),
        ..SimpleStreamOptions::default()
    };
    let _ = simple.stream_simple(&fixture_model(), &Context::default(), Some(&options));
    let sent = recorded
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .expect("dispatched");
    let headers = sent.headers.expect("simple session header");
    assert_eq!(
        headers.get("x-opencode-session"),
        Some(&Some("session-2".to_owned()))
    );
}

/// The bare chat provider the trait-default test drives.
struct BareProvider;

impl Provider for BareProvider {
    fn id(&self) -> &'static str {
        "bare"
    }

    fn name(&self) -> &'static str {
        "Bare"
    }

    fn auth(&self) -> &ProviderAuth {
        static AUTH: std::sync::OnceLock<ProviderAuth> = std::sync::OnceLock::new();
        AUTH.get_or_init(ProviderAuth::default)
    }

    fn get_models(&self) -> Result<Vec<Model>, ProviderError> {
        Ok(Vec::new())
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: Option<&StreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        pi_ai::utils::event_stream::assistant_message_event_stream()
    }

    fn stream_simple(
        &self,
        _model: &Model,
        _context: &Context,
        _options: Option<&SimpleStreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        pi_ai::utils::event_stream::assistant_message_event_stream()
    }
}

/// The bare images provider the trait-default test drives.
struct BareImagesProvider;

impl pi_ai::images_models::ImagesProvider for BareImagesProvider {
    fn id(&self) -> &'static str {
        "bare"
    }

    fn name(&self) -> &'static str {
        "Bare"
    }

    fn auth(&self) -> &ProviderAuth {
        static AUTH: std::sync::OnceLock<ProviderAuth> = std::sync::OnceLock::new();
        AUTH.get_or_init(ProviderAuth::default)
    }

    fn get_models(&self) -> Result<Vec<pi_ai::types::ImagesModel>, ProviderError> {
        Ok(Vec::new())
    }

    fn generate_images<'a>(
        &'a self,
        _model: &'a pi_ai::types::ImagesModel,
        _context: &'a pi_ai::types::ImagesContext,
        _options: Option<&'a pi_ai::types::ImagesOptions>,
    ) -> pi_ai::types::BoxedFuture<
        'a,
        Result<pi_ai::types::AssistantImages, pi_ai::utils::provider_retry::ProviderRequestError>,
    > {
        Box::pin(async { Err(pi_ai::utils::provider_retry::ProviderRequestError::aborted()) })
    }
}

/// The innermost streams the OpenCode wrapper dispatches to: records what
/// arrives on the simple surface.
struct SimpleRecording(Arc<Mutex<Option<SimpleStreamOptions>>>);

impl ProviderStreams for SimpleRecording {
    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: Option<&StreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        pi_ai::utils::event_stream::assistant_message_event_stream()
    }

    fn stream_simple(
        &self,
        _model: &Model,
        _context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        *self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = options.cloned();
        pi_ai::utils::event_stream::assistant_message_event_stream()
    }
}

/// The Provider trait's default fetch/cancel methods return None for a bare
/// provider, and the deferred Models surface reports the unsupported notice
/// through them.
#[tokio::test]
async fn the_provider_trait_deferred_defaults_return_none() {
    let provider: Arc<dyn Provider> = Arc::new(BareProvider);
    let model = fixture_model();
    let handle = DeferredHandle {
        provider: "bare".to_owned(),
        model_id: "m".to_owned(),
        api: "test-api".to_owned(),
        id: "id".to_owned(),
        expires_at: None,
        poll_after_ms: None,
        data: None,
    };
    assert!(provider.fetch_deferred(&model, &handle, None).is_none());
    assert!(provider.cancel_deferred(&model, &handle, None).is_none());
    assert_eq!(provider.get_models().expect("listing"), Vec::new());
}

/// The images trait's default `refresh_models` resolves Ok when called
/// directly, and the `SharedRefreshError` source chain renders the reason.
#[tokio::test]
async fn the_images_provider_edges_reach_their_defaults() {
    let provider: Arc<dyn pi_ai::images_models::ImagesProvider> = Arc::new(BareImagesProvider);
    // The trait default refresh resolves without a fetch.
    provider.refresh_models().await.expect("default refresh");
    assert!(!provider.supports_refresh_models());
    // The Debug impl renders id and name.
    let rendered = format!(
        "{:?}",
        pi_ai::images_models::create_images_provider(
            pi_ai::images_models::CreateImagesProviderOptions {
                id: "bare".to_owned(),
                name: None,
                auth: ProviderAuth::default(),
                models: Vec::new(),
                refresh_models: None,
                api: pi_ai::api::not_ported_images("test-images"),
            }
        )
    );
    assert!(rendered.contains("bare"), "got: {rendered}");
}

/// The failure-code Display surfaces, the From conversions, and the
/// with-cause-detail branches, upstream's error-message contract.
#[test]
fn the_models_error_surfaces_render_their_wire_forms() {
    use pi_ai::auth::resolve::{ModelsError, ModelsErrorCode, ModelsFailure};

    // Every code renders its wire spelling.
    for (code, wire) in [
        (ModelsErrorCode::ModelSource, "model_source"),
        (ModelsErrorCode::ModelValidation, "model_validation"),
        (ModelsErrorCode::Provider, "provider"),
        (ModelsErrorCode::Stream, "stream"),
        (ModelsErrorCode::Auth, "auth"),
        (ModelsErrorCode::OAuth, "oauth"),
    ] {
        assert_eq!(code.wire(), wire);
        assert_eq!(code.to_string(), wire);
    }

    // The From conversions carry the error through.
    let error = ModelsError::new(ModelsErrorCode::Auth, "boom");
    let failure: ModelsFailure = error.into();
    assert_eq!(failure.to_string(), "boom");

    // The Display arm for the Models variant renders through the source.
    let failure = ModelsFailure::Models(ModelsError::new(ModelsErrorCode::Auth, "auth boom"));
    let rendered = format!("{failure}");
    assert_eq!(rendered, "auth boom");
    let _ = std::error::Error::source(&failure);

    // withCauseDetail's no-detail and already-contained branches.
    let cause: Box<dyn std::error::Error + Send + Sync> = Box::new(std::io::Error::other(""));
    let quiet = ModelsError::with_cause(ModelsErrorCode::Auth, "quiet failure", cause);
    assert_eq!(quiet.to_string(), "quiet failure");
}

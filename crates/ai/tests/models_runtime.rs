//! The Models runtime suite, ported from
//! `packages/ai/test/models-runtime.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::float_cmp,
    reason = "the cost assertions pin exact upstream arithmetic outcomes"
)]
#![expect(
    clippy::unwrap_used,
    reason = "the fixtures unwrap only the values the test just placed"
)]
#![expect(
    clippy::panic,
    reason = "test failures panic by design, mirroring expect!'s failure mode"
)]
#![expect(
    clippy::missing_const_for_fn,
    reason = "fixture constructors wrap runtime state that cannot be const"
)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pi_ai::auth::credential_store::{CredentialStore, InMemoryCredentialStore};
use pi_ai::auth::resolve::{AuthResolutionOverrides, ModelsErrorCode, ModelsFailure, now_ms};
use pi_ai::auth::types::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, ApiKeyResolveFn, AuthOptions, AuthPrompt,
    AuthResult, AuthType, Credential, ModelAuth, OAuthAuth, OAuthCredentials, ProviderAuth,
};
use pi_ai::models::{
    CatalogPersist, CreateProviderOptions, ModelsPublication, ModelsRefreshOptions,
    ModelsRefreshResult, Provider, ProviderApi, ProviderError, RefreshModelsContext,
    calculate_cost, create_models, create_provider, has_api,
};
use pi_ai::models_store::{InMemoryModelsStore, ModelsStore, ModelsStoreEntry, ModelsStoreOptions};
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, BoxedFuture, Context, Message, Model, ProviderId,
    SimpleStreamOptions, StopReason, StreamOptions, TextContent, Usage, UsageCost, UserContent,
    UserMessage,
};
use pi_ai::utils::abort::AbortError;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use tokio_util::sync::CancellationToken;

/// The poisoned-mutex-tolerant lock every fixture counter and recording
/// cell reads through.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn test_model(provider: &str, id: &str) -> Model {
    Model {
        id: id.to_owned(),
        name: id.to_owned(),
        api: Api::from("test-api"),
        provider: ProviderId::from(provider),
        base_url: "https://example.test/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: pi_ai::types::ModelCost {
            rates: pi_ai::types::ModelCostRates::default(),
            tiers: None,
        },
        context_window: 10_000,
        max_tokens: 1_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn done_message(model: &Model, text: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 0,
            cost: UsageCost::default(),
        },
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms(),
    }
}

#[derive(Debug, Clone)]
struct ProviderCall {
    model: Model,
    options: Option<StreamOptions>,
}

/// Ambient auth for keyless test providers; reports "configured" with no
/// auth values.
fn ambient_auth() -> ApiKeyAuth {
    ApiKeyAuth {
        name: "Ambient".to_owned(),
        login: None,
        check: None,
        resolve: Arc::new(|_input: ApiKeyAuthInput| {
            Box::pin(async move { Ok(Some(AuthResult::default())) })
        }),
    }
}

fn test_provider(input: TestProviderInput) -> Arc<dyn Provider> {
    let models = input
        .models
        .unwrap_or_else(|| vec![test_model(input.id, "model-a")]);
    let calls = input.calls;
    Arc::new(TestProvider {
        id: input.id,
        models: Mutex::new(models),
        broken_models: input.broken_models,
        get_models_fn: input.get_models,
        auth: input.auth.unwrap_or_else(|| ProviderAuth {
            api_key: Some(ambient_auth()),
            oauth: None,
        }),
        refresh: input.refresh,
        calls,
    })
}

struct TestProviderInput {
    id: &'static str,
    models: Option<Vec<Model>>,
    broken_models: bool,
    auth: Option<ProviderAuth>,
    refresh: Option<RefreshFn>,
    calls: Option<Arc<Mutex<Vec<ProviderCall>>>>,
    get_models: Option<GetModelsFn>,
}

/// The all-default [`TestProviderInput`]: ambient auth, the default
/// listing, no refresh, no recorded calls. Tests override single fields
/// with `..default_input(id)` at the call site.
fn default_input(id: &'static str) -> TestProviderInput {
    TestProviderInput {
        id,
        models: None,
        broken_models: false,
        auth: None,
        refresh: None,
        calls: None,
        get_models: None,
    }
}

/// The custom-listing closure of [`TestProviderInput`], upstream's
/// `getModels: () => list`.
type GetModelsFn = Arc<dyn Fn() -> Result<Vec<Model>, ProviderError> + Send + Sync>;

type RefreshFn = Arc<
    dyn Fn(&RefreshModelsContext) -> BoxedFuture<'static, Result<(), ProviderError>> + Send + Sync,
>;

/// Adapts a test's owned-context refresh body into a [`RefreshFn`]: clones
/// the context per call and hands the body a `'static` future over it, the
/// shape upstream's `async (context) => {}` refresh handlers take.
fn refresh_future(
    refresh: impl Fn(RefreshModelsContext) -> BoxedFuture<'static, Result<(), ProviderError>>
    + Send
    + Sync
    + 'static,
) -> RefreshFn {
    Arc::new(move |context: &RefreshModelsContext| {
        let context = context.clone();
        refresh(context)
    })
}

/// Registers a dynamic provider whose refresh runs `refresh`, the setup
/// every refresh-lifecycle test drives.
fn register_refreshing(models: &pi_ai::models::Models, id: &'static str, refresh: RefreshFn) {
    models.set_provider(test_provider(TestProviderInput {
        refresh: Some(refresh),
        ..default_input(id)
    }));
}

struct TestProvider {
    id: &'static str,
    models: Mutex<Vec<Model>>,
    broken_models: bool,
    get_models_fn: Option<GetModelsFn>,
    auth: ProviderAuth,
    refresh: Option<RefreshFn>,
    calls: Option<Arc<Mutex<Vec<ProviderCall>>>>,
}

impl Provider for TestProvider {
    fn id(&self) -> &str {
        self.id
    }

    fn name(&self) -> &str {
        self.id
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn get_models(&self) -> Result<Vec<Model>, ProviderError> {
        if let Some(get_models) = &self.get_models_fn {
            return get_models();
        }
        if self.broken_models {
            return Err(std::io::Error::other("boom").into());
        }
        Ok(lock(&self.models).clone())
    }

    fn refresh_models(
        &self,
        context: RefreshModelsContext,
    ) -> BoxedFuture<'_, Result<(), ProviderError>> {
        match &self.refresh {
            Some(refresh) => refresh(&context),
            None => Box::pin(async { Ok(()) }),
        }
    }

    fn supports_refresh_models(&self) -> bool {
        self.refresh.is_some()
    }

    fn stream(
        &self,
        model: &Model,
        _context: &Context,
        options: Option<&StreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        if let Some(calls) = &self.calls {
            lock(calls).push(ProviderCall {
                model: model.clone(),
                options: options.cloned(),
            });
        }
        let stream = assistant_message_event_stream();
        let message = done_message(model, "ok");
        stream.push(pi_ai::types::AssistantMessageEvent::Start {
            partial: message.clone(),
        });
        stream.push(pi_ai::types::AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: message.clone(),
        });
        stream.end(Some(&message));
        stream
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        let as_stream = options.map(|options| StreamOptions {
            transport_options: options.transport_options.clone(),
            api_key: options.api_key.clone(),
            telemetry_context: options.telemetry_context.clone(),
            env: options.env.clone(),
            headers: options.headers.clone(),
            timeout_ms: options.timeout_ms,
            max_retries: options.max_retries,
            max_retry_delay_ms: options.max_retry_delay_ms,
            temperature: options.temperature,
            sampling_params: options.sampling_params.clone(),
            max_tokens: options.max_tokens,
            transport: options.transport,
            cache_retention: options.cache_retention,
            session_id: options.session_id.clone(),
            websocket_connect_timeout_ms: options.websocket_connect_timeout_ms,
            metadata: options.metadata.clone(),
        });
        self.stream(model, context, as_stream.as_ref())
    }
}

fn context() -> Context {
    Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("hi".to_owned()),
            timestamp: now_ms(),
        })],
        tools: None,
    }
}

/// Creates the collection over a credential store, the setup every
/// auth-flow test drives through `get_auth`.
fn models_with_credentials(credentials: Arc<dyn CredentialStore>) -> pi_ai::models::Models {
    create_models(Some(pi_ai::models::CreateModelsOptions {
        credentials: Some(credentials),
        ..pi_ai::models::CreateModelsOptions::default()
    }))
}

/// The counting-refresh workspace the refresh-counting tests share: an
/// in-memory store, its collection, and the rotation counter.
fn counting_workspace() -> (
    pi_ai::models::Models,
    Arc<dyn CredentialStore>,
    Arc<Mutex<u32>>,
) {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    let refreshes = Arc::new(Mutex::new(0));
    let models = models_with_credentials(Arc::clone(&credentials));
    (models, credentials, refreshes)
}

/// Creates the collection over a models store, the persistence setup the
/// store-backed tests share.
fn models_with_store(store: Arc<dyn ModelsStore>) -> pi_ai::models::Models {
    create_models(Some(pi_ai::models::CreateModelsOptions {
        models_store: Some(store),
        ..pi_ai::models::CreateModelsOptions::default()
    }))
}

/// Writes one fixed credential through the store's modify path, the setup
/// write every auth-flow test performs; any stored entry is replaced.
async fn store_credential(store: &dyn CredentialStore, provider_id: &str, credential: Credential) {
    store
        .modify(
            provider_id,
            Box::new(move |_current| Box::pin(async move { Ok(Some(credential)) })),
            None,
        )
        .await
        .expect("write");
}

/// The expired stored credential the refresh-lifecycle tests seed, with
/// the remaining validity each scenario drives.
fn old_token_credential(expires: i64) -> OAuthCredentials {
    OAuthCredentials {
        refresh: "r".to_owned(),
        access: "old-token".to_owned(),
        expires,
        extra: BTreeMap::new(),
    }
}

/// The spent stored credential the refresh-failure tests seed, the state
/// the failure paths must preserve.
fn old_credential() -> OAuthCredentials {
    OAuthCredentials {
        refresh: "r".to_owned(),
        access: "old".to_owned(),
        expires: 0,
        extra: BTreeMap::new(),
    }
}

/// Spawns the refresh so it starts polling before the caller cancels: a
/// Rust future starts on first poll, the interleaving the abort tests
/// race against.
fn spawn_refresh(
    models: &pi_ai::models::Models,
    options: ModelsRefreshOptions,
) -> tokio::task::JoinHandle<ModelsRefreshResult> {
    let models = models.clone();
    tokio::spawn(async move { models.refresh(Some(&options)).await })
}

/// Drives the abort race with the given refresh body: registers the
/// dynamic provider, spawns, cancels the controller, joins, and pins the
/// aborted-without-errors outcome, the shape both abort tests check.
async fn expect_aborted_refresh(refresh: RefreshFn) {
    let controller = CancellationToken::new();
    let models = create_models(None);
    register_refreshing(&models, "dynamic", refresh);
    let pending = spawn_refresh(
        &models,
        ModelsRefreshOptions {
            signal: Some(controller.clone()),
            ..ModelsRefreshOptions::default()
        },
    );
    controller.cancel();
    let result = pending.await.expect("joined");
    assert!(result.aborted);
    assert_eq!(result.errors.len(), 0);
}

/// Publishes through the refresh context to completion and discards the
/// accepted flag; the publication tests observe the update side effect,
/// not the generation check.
async fn publish_now(context: &RefreshModelsContext, publication: ModelsPublication) {
    let _ = (context.publish)(publication).await.expect("publish");
}

fn env_key_auth(key: Option<&str>) -> ApiKeyAuth {
    let env_key = key.map(ToOwned::to_owned);
    let resolve: ApiKeyResolveFn = Arc::new(move |input: ApiKeyAuthInput| {
        let resolved = input
            .credential
            .as_ref()
            .and_then(|credential| credential.key.clone())
            .or_else(|| env_key.clone());
        let Some(resolved) = resolved else {
            let none: BoxedFuture<'static, Result<Option<AuthResult>, ProviderError>> =
                Box::pin(async move { Ok(None) });
            return none;
        };
        let source = if input.credential.is_some() {
            "stored"
        } else {
            "env"
        };
        Box::pin(async move {
            Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(resolved),
                    ..ModelAuth::default()
                },
                env: None,
                source: Some(source.to_owned()),
            }))
        })
    });
    ApiKeyAuth {
        name: "Test API key".to_owned(),
        login: None,
        check: None,
        resolve,
    }
}

/// The auth block carrying only an api-key resolve against the ambient
/// env, the setup most credential-path tests register.
fn env_key_auth_only(key: Option<&str>) -> ProviderAuth {
    ProviderAuth {
        api_key: Some(env_key_auth(key)),
        oauth: None,
    }
}

fn test_oauth() -> OAuthAuth {
    test_oauth_with_refresh(None)
}

/// The auth block carrying only the OAuth flow, the shape every
/// oauth-lifecycle test registers.
fn oauth_auth_only(oauth: OAuthAuth) -> ProviderAuth {
    ProviderAuth {
        api_key: None,
        oauth: Some(oauth),
    }
}

/// Registers the p1 provider whose only auth is `oauth`, the registration
/// every oauth-lifecycle test performs.
fn register_oauth_p1(models: &pi_ai::models::Models, oauth: OAuthAuth) {
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(oauth_auth_only(oauth)),
        ..default_input("p1")
    }));
}

/// The boxed auth failure an auth callback reports, matching the error type
/// the trait objects carry.
fn fail_error(message: &str) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(std::io::Error::other(message.to_owned()))
}

/// The custom-refresh closure of [`test_oauth_with_refresh`].
type OAuthRefreshFn = Arc<
    dyn Fn(
            OAuthCredentials,
        ) -> BoxedFuture<
            'static,
            Result<OAuthCredentials, Box<dyn std::error::Error + Send + Sync>>,
        > + Send
        + Sync,
>;

fn test_oauth_with_refresh(refresh: Option<OAuthRefreshFn>) -> OAuthAuth {
    OAuthAuth {
        name: "Test OAuth".to_owned(),
        is_subscription: None,
        login_label: None,
        login: Arc::new(
            |_interaction: pi_ai::auth::types::ProviderAuthInteraction| {
                Box::pin(async move { Err(fail_error("not used")) })
            },
        ),
        refresh: Arc::new(move |credential: OAuthCredentials, _signal| {
            if let Some(refresh) = &refresh {
                refresh(credential)
            } else {
                let passthrough: BoxedFuture<
                    'static,
                    Result<OAuthCredentials, Box<dyn std::error::Error + Send + Sync>>,
                > = Box::pin(async move { Ok(credential) });
                passthrough
            }
        }),
        to_auth: Arc::new(|credential: OAuthCredentials| {
            Box::pin(async move {
                Ok(ModelAuth {
                    api_key: Some(credential.access),
                    ..ModelAuth::default()
                })
            })
        }),
    }
}

/// The OAuth refresh that counts rotations and issues a fresh one-hour
/// token from the credential it receives, the probe the refresh-lifecycle
/// tests assert against.
fn counting_oauth_refresh(count: Arc<Mutex<u32>>) -> OAuthRefreshFn {
    Arc::new(move |credential| {
        let count = Arc::clone(&count);
        Box::pin(async move {
            *lock(&count) += 1;
            Ok(OAuthCredentials {
                refresh: credential.refresh,
                access: "new-token".to_owned(),
                expires: now_ms() + 60 * 60_000,
                extra: credential.extra,
            })
        })
    })
}

/// The counting-refresh setup the refresh-counting tests share: a p1
/// provider whose oauth rotates a fresh one-hour token over an in-memory
/// store, seeded with a credential of the given remaining validity.
async fn setup_counting_oauth(seed_expires: i64) -> (pi_ai::models::Models, Arc<Mutex<u32>>) {
    let (models, credentials, refreshes) = counting_workspace();
    let oauth = test_oauth_with_refresh(Some(counting_oauth_refresh(Arc::clone(&refreshes))));
    register_oauth_p1(&models, oauth);
    store_credential(
        credentials.as_ref(),
        "p1",
        Credential::OAuth(old_token_credential(seed_expires)),
    )
    .await;
    (models, refreshes)
}
#[tokio::test]
async fn enumerates_credential_metadata_without_exposing_secrets() {
    let credentials = InMemoryCredentialStore::default();
    store_credential(
        &credentials,
        "api-provider",
        Credential::ApiKey(ApiKeyCredential {
            key: Some("secret".to_owned()),
            env: None,
        }),
    )
    .await;
    store_credential(
        &credentials,
        "oauth-provider",
        Credential::OAuth(OAuthCredentials {
            refresh: "refresh".to_owned(),
            access: "access".to_owned(),
            expires: now_ms() + 60_000,
            extra: BTreeMap::new(),
        }),
    )
    .await;

    let listed = credentials.list(None).await.expect("list");
    expect_credential_list(
        &listed,
        &[("api-provider", "api_key"), ("oauth-provider", "oauth")],
    );
}

fn expect_credential_list(
    actual: &[pi_ai::auth::types::CredentialInfo],
    expected: &[(&str, &str)],
) {
    let rendered: Vec<String> = actual
        .iter()
        .map(|entry| format!("{}={}", entry.provider_id, entry.auth_type))
        .collect();
    let expected: Vec<String> = expected
        .iter()
        .map(|(provider_id, auth_type)| format!("{provider_id}={auth_type}"))
        .collect();
    assert_eq!(rendered, expected);
}

/// The model ids a listing returns, the shape the listing tests pin.
fn listed_model_ids(listed: &[Model]) -> Vec<String> {
    listed.iter().map(|model| model.id.clone()).collect()
}

/// The api key a provider-level auth resolution carries, the outcome the
/// oauth-lifecycle tests pin.
async fn resolved_api_key(models: &pi_ai::models::Models, provider_id: &str) -> Option<String> {
    models
        .get_auth(provider_id, None)
        .await
        .expect("auth")
        .and_then(|result| result.auth.api_key)
}

/// Asserts the resolution failed with the wrapped auth [`ModelsError`],
/// the outcome the wrapped-store-failure tests pin.
fn expect_wrapped_auth_error(auth: Result<Option<AuthResult>, ModelsFailure>) {
    assert!(
        matches!(auth, Err(ModelsFailure::Models(error)) if error.code() == ModelsErrorCode::Auth)
    );
}

#[tokio::test]
async fn applies_request_wide_pricing_tiers_above_the_configured_input_threshold() {
    let mut model = test_model("openai", "gpt-5.6-sol");
    model.cost = pi_ai::types::ModelCost {
        rates: pi_ai::types::ModelCostRates {
            input: 5.0,
            output: 30.0,
            cache_read: 0.5,
            cache_write: 6.25,
        },
        tiers: Some(vec![pi_ai::types::ModelCostTier {
            rates: pi_ai::types::ModelCostRates {
                input: 10.0,
                output: 45.0,
                cache_read: 1.0,
                cache_write: 12.5,
            },
            input_tokens_above: 272_000,
        }]),
    };
    let create_usage = |cache_write: u64| Usage {
        input: 200_000,
        output: 100_000,
        cache_read: 72_000,
        cache_write,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 372_000 + cache_write,
        cost: UsageCost::default(),
    };

    let mut short_usage = create_usage(0);
    let short = calculate_cost(&model, &mut short_usage);
    assert_eq!(short.input, 1.0);
    assert_eq!(short.output, 3.0);
    assert_eq!(short.cache_read, 0.036);
    assert_eq!(short.cache_write, 0.0);

    let mut long_usage = create_usage(1);
    let long = calculate_cost(&model, &mut long_usage);
    assert_eq!(long.input, 2.0);
    assert_eq!(long.output, 4.5);
    assert_eq!(long.cache_read, 0.072);
    assert_eq!(long.cache_write, 0.000_012_5);
}

#[tokio::test]
async fn registers_replaces_and_deletes_providers() {
    let models = create_models(None);
    models.set_provider(test_provider(default_input("p1")));
    models.set_provider(test_provider(default_input("p2")));
    assert_eq!(
        models
            .providers()
            .iter()
            .map(|provider| provider.id().to_owned())
            .collect::<Vec<_>>(),
        ["p1", "p2"]
    );

    let replacement = test_provider(default_input("p1"));
    models.set_provider(Arc::clone(&replacement));
    assert!(Arc::ptr_eq(
        &replacement,
        &models.provider("p1").expect("registered")
    ));
    assert_eq!(models.providers().len(), 2);

    models.delete_provider("p1");
    assert!(models.provider("p1").is_none());

    models.clear_providers();
    assert_eq!(models.providers().len(), 0);
}

#[tokio::test]
async fn lists_and_finds_models_per_provider() {
    let models = create_models(None);
    models.set_provider(test_provider(TestProviderInput {
        models: Some(vec![test_model("p1", "m1"), test_model("p1", "m2")]),
        ..default_input("p1")
    }));
    models.set_provider(test_provider(TestProviderInput {
        models: Some(vec![test_model("p2", "m3")]),
        ..default_input("p2")
    }));

    assert_eq!(listed_model_ids(&models.models(None)), ["m1", "m2", "m3"]);
    assert_eq!(listed_model_ids(&models.models(Some("p1"))), ["m1", "m2"]);
    assert_eq!(models.models(Some("nope")).len(), 0);
    assert_eq!(
        models.model("p2", "m3").map(|model| model.id),
        Some("m3".to_owned())
    );
    assert!(models.model("p2", "missing").is_none());

    // has_api narrows dynamically looked-up models with a runtime check
    let found = models.model("p2", "m3");
    assert!(
        found
            .as_ref()
            .is_some_and(|model| !has_api(model, &Api::from("openai-completions")))
    );
    assert!(
        found
            .as_ref()
            .is_some_and(|model| has_api(model, &Api::from("test-api")))
    );
}

#[tokio::test]
async fn swallows_provider_source_failures_for_both_listings() {
    let models = create_models(None);
    models.set_provider(test_provider(TestProviderInput {
        broken_models: true,
        ..default_input("broken")
    }));
    models.set_provider(test_provider(TestProviderInput {
        models: Some(vec![test_model("ok", "m1")]),
        ..default_input("ok")
    }));

    assert_eq!(listed_model_ids(&models.models(None)), ["m1"]);
    assert_eq!(models.models(Some("broken")), Vec::<Model>::new());
    // precise failures come from the provider directly
    let error = models.provider("broken").expect("registered").get_models();
    assert_eq!(
        error.expect_err("broken provider fails").to_string(),
        "boom"
    );
}

#[tokio::test]
async fn refresh_updates_dynamic_providers_and_reports_failures() {
    let list = Arc::new(Mutex::new(vec![test_model("dyn", "before")]));
    let refreshes = Arc::new(Mutex::new(0));
    let models = create_models(None);
    let list_for_refresh = Arc::clone(&list);
    let refreshes_for_provider = Arc::clone(&refreshes);
    let list_for_get = Arc::clone(&list);
    models.set_provider(test_provider(TestProviderInput {
        refresh: Some(refresh_future(move |context| {
            let list = Arc::clone(&list_for_refresh);
            let refreshes = Arc::clone(&refreshes_for_provider);
            Box::pin(async move {
                if !context.allow_network {
                    return Ok(());
                }
                *lock(&refreshes) += 1;
                publish_now(
                    &context,
                    ModelsPublication {
                        persist: CatalogPersist::Omit,
                        update: Some(Box::new(move || {
                            *lock(&list) = vec![test_model("dyn", "after")];
                        })),
                    },
                )
                .await;
                Ok(())
            })
        })),
        get_models: Some(Arc::new(move || Ok(lock(&list_for_get).clone()))),
        ..default_input("dyn")
    }));
    models.set_provider(test_provider(TestProviderInput {
        models: Some(vec![test_model("static", "s1")]),
        ..default_input("static")
    }));

    assert!(models.model("dyn", "before").is_some());
    let first = models.refresh(None).await;
    assert_eq!(first.errors.len(), 0);
    assert_eq!(*lock(&refreshes), 1);
    assert!(models.model("dyn", "after").is_some());
    assert!(models.model("dyn", "before").is_none());

    let models = create_models(None);
    register_refreshing(
        &models,
        "flaky",
        refresh_future(|context| {
            Box::pin(async move {
                if context.allow_network {
                    return Err(std::io::Error::other("fetch failed").into());
                }
                Ok(())
            })
        }),
    );
    let second = models.refresh(None).await;
    assert_eq!(
        second.errors.get("flaky").map(ToString::to_string),
        Some("fetch failed".to_owned())
    );
}

#[tokio::test]
async fn restricts_refresh_work_to_selected_providers() {
    let calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let models = create_models(None);
    for id in ["one", "two"] {
        let calls_for_provider = Arc::clone(&calls);
        let id_owned = id;
        register_refreshing(
            &models,
            id_owned,
            refresh_future(move |context| {
                let calls = Arc::clone(&calls_for_provider);
                Box::pin(async move {
                    lock(&calls).push(format!(
                        "{id_owned}:{}",
                        if context.allow_network {
                            "network"
                        } else {
                            "cache"
                        }
                    ));
                    Ok(())
                })
            }),
        );
    }

    let result = models
        .refresh(Some(&ModelsRefreshOptions {
            providers: Some(vec!["two".to_owned(), "unknown".to_owned()]),
            ..ModelsRefreshOptions::default()
        }))
        .await;

    assert_eq!(result.errors.len(), 0);
    assert_eq!(*lock(&calls), ["two:cache", "two:network"]);
}
#[tokio::test]
async fn restores_cached_models_before_waiting_for_network_auth() {
    let store = InMemoryModelsStore::default();
    store
        .write(
            "dynamic",
            ModelsStoreEntry {
                models: vec![test_model("dynamic", "cached")],
                ..ModelsStoreEntry::default()
            },
            None,
        )
        .await
        .expect("write");
    let (auth_started_sender, auth_started_receiver) = tokio::sync::oneshot::channel::<()>();
    let (finish_auth_sender, finish_auth_receiver) = tokio::sync::oneshot::channel::<()>();
    let _ = (&finish_auth_sender, &auth_started_receiver);
    let auth_started = Arc::new(Mutex::new(Some(auth_started_sender)));
    let finish_auth_holder = Arc::new(Mutex::new(Some(finish_auth_receiver)));
    let resolve: ApiKeyResolveFn = Arc::new(move |_input: ApiKeyAuthInput| {
        let auth_started = Arc::clone(&auth_started);
        let finish_auth_holder = Arc::clone(&finish_auth_holder);
        Box::pin(async move {
            let sender = lock(&auth_started).take();
            if let Some(sender) = sender {
                let _ = sender.send(());
            }
            let finish_auth = lock(&finish_auth_holder).take();
            if let Some(finish_auth) = finish_auth {
                let _ = finish_auth.await;
            }
            Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some("key".to_owned()),
                    ..ModelAuth::default()
                },
                env: None,
                source: None,
            }))
        })
    });
    let provider = create_provider(CreateProviderOptions {
        id: "dynamic".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ProviderAuth {
            api_key: Some(ApiKeyAuth {
                name: "Blocked auth".to_owned(),
                login: None,
                check: None,
                resolve,
            }),
            oauth: None,
        },
        models: Vec::new(),
        fetch_models: Some(Arc::new(|_context| {
            Box::pin(async move { Err(std::io::Error::other("must not fetch").into()) })
        })),
        api: ProviderApi::Single(unreachable_streams()),
        filter_models: None,
    });
    let models = models_with_store(Arc::new(store));
    models.set_provider(Arc::new(provider));
    let controller = CancellationToken::new();
    // Porting restatement: upstream's promise starts eagerly; a Rust future
    // starts on first poll, so the interleaving needs a spawned task.
    let pending = spawn_refresh(
        &models,
        ModelsRefreshOptions {
            providers: Some(vec!["dynamic".to_owned()]),
            signal: Some(controller.clone()),
            ..ModelsRefreshOptions::default()
        },
    );
    auth_started_receiver.await.expect("auth started");

    assert!(models.model("dynamic", "cached").is_some());
    controller.cancel();
    assert!(pending.await.expect("joined").aborted);
}

/// Streams that never get called; the fixture only exercises refresh.
fn unreachable_streams() -> Arc<dyn pi_ai::types::ProviderStreams> {
    pi_ai::api::not_ported_streams("test-api")
}

#[tokio::test]
async fn lets_providers_choose_persistent_deletion_and_ephemeral_publication() {
    let entry: Arc<Mutex<Option<ModelsStoreEntry>>> =
        Arc::new(Mutex::new(Some(ModelsStoreEntry {
            models: vec![test_model("dynamic", "stored")],
            ..ModelsStoreEntry::default()
        })));
    let store = shared_store(Arc::clone(&entry));
    let state: Arc<Mutex<String>> = Arc::new(Mutex::new("initial".to_owned()));
    let models = models_with_store(Arc::new(store));
    let state_for_refresh = Arc::clone(&state);
    let entry_for_refresh = Arc::clone(&entry);
    register_refreshing(
        &models,
        "dynamic",
        refresh_future(move |context| {
            let state = Arc::clone(&state_for_refresh);
            let entry = Arc::clone(&entry_for_refresh);
            Box::pin(async move {
                assert_eq!(
                    context
                        .stored
                        .as_ref()
                        .and_then(|stored| stored.models.first())
                        .map(|model| model.id.clone()),
                    Some("stored".to_owned())
                );
                publish_now(
                    &context,
                    ModelsPublication {
                        persist: CatalogPersist::Delete,
                        update: Some(Box::new({
                            let entry = Arc::clone(&entry);
                            let state = Arc::clone(&state);
                            move || {
                                assert!(lock(&entry).is_none());
                                *lock(&state) = "deleted".to_owned();
                            }
                        })),
                    },
                )
                .await;
                publish_now(
                    &context,
                    ModelsPublication {
                        persist: CatalogPersist::Omit,
                        update: Some(Box::new({
                            let state = Arc::clone(&state);
                            move || {
                                *lock(&state) = "ephemeral".to_owned();
                            }
                        })),
                    },
                )
                .await;
                Ok(())
            })
        }),
    );

    let result = models
        .refresh(Some(&ModelsRefreshOptions {
            allow_network: Some(false),
            ..ModelsRefreshOptions::default()
        }))
        .await;

    assert_eq!(result.errors.len(), 0);
    assert!(lock(&entry).is_none());
    assert_eq!(*lock(&state), "ephemeral");
}

/// A [`ModelsStore`] backed by a shared entry cell, upstream's inline
/// `ModelsStore` object.
fn shared_store(entry: Arc<Mutex<Option<ModelsStoreEntry>>>) -> SharedModelsStore {
    SharedModelsStore(entry)
}

struct SharedModelsStore(Arc<Mutex<Option<ModelsStoreEntry>>>);

impl ModelsStore for SharedModelsStore {
    fn read<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<Option<ModelsStoreEntry>, pi_ai::models_store::ModelsStoreError>>
    {
        let entry = Arc::clone(&self.0);
        Box::pin(async move { Ok(lock(&entry).clone()) })
    }

    fn write<'a>(
        &'a self,
        _provider_id: &'a str,
        entry: ModelsStoreEntry,
        _options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<(), pi_ai::models_store::ModelsStoreError>> {
        let slot = Arc::clone(&self.0);
        Box::pin(async move {
            *lock(&slot) = Some(entry);
            Ok(())
        })
    }

    fn delete<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<(), pi_ai::models_store::ModelsStoreError>> {
        let slot = Arc::clone(&self.0);
        Box::pin(async move {
            *lock(&slot) = None;
            Ok(())
        })
    }
}
#[tokio::test]
async fn persists_dynamic_catalogs_and_restores_them_without_network() {
    let credentials = InMemoryCredentialStore::default();
    store_credential(
        &credentials,
        "dynamic",
        Credential::ApiKey(ApiKeyCredential {
            key: Some("key".to_owned()),
            env: None,
        }),
    )
    .await;
    let models_store: Arc<dyn ModelsStore> = Arc::new(InMemoryModelsStore::default());

    let create_dynamic_provider = |fetch: Option<FetchModelsGuard>| -> Arc<dyn Provider> {
        Arc::new(create_provider(CreateProviderOptions {
            id: "dynamic".to_owned(),
            name: None,
            base_url: None,
            headers: None,
            auth: env_key_auth_only(None),
            models: Vec::new(),
            fetch_models: fetch.map(|fetch| fetch.into_fetch_fn()),
            api: ProviderApi::Single(unreachable_streams()),
            filter_models: None,
        }))
    };

    let fetched = FetchModelsGuard::default();
    let online = create_models(Some(pi_ai::models::CreateModelsOptions {
        credentials: Some(Arc::new(credentials)),
        models_store: Some(Arc::clone(&models_store)),
        ..pi_ai::models::CreateModelsOptions::default()
    }));
    online.set_provider(create_dynamic_provider(Some(fetched.clone())));
    let online_result = online.refresh(None).await;
    assert_eq!(online_result.errors.len(), 0);
    assert!(online.model("dynamic", "fetched").is_some());
    assert_eq!(fetched.count(), 1);

    // A fresh collection over the same store restores offline.
    let offline = models_with_store(Arc::clone(&models_store));
    let never = FetchModelsGuard::default();
    offline.set_provider(create_dynamic_provider(Some(never.clone())));
    let offline_result = offline
        .refresh(Some(&ModelsRefreshOptions {
            allow_network: Some(false),
            ..ModelsRefreshOptions::default()
        }))
        .await;
    assert_eq!(offline_result.errors.len(), 0);
    assert!(offline.model("dynamic", "fetched").is_some());
    assert_eq!(never.count(), 0);
}

/// The counted fetch closure of the persistence test.
#[derive(Clone, Default)]
struct FetchModelsGuard(Arc<Mutex<u32>>);

impl FetchModelsGuard {
    fn count(&self) -> u32 {
        *lock(&self.0)
    }

    fn into_fetch_fn(self) -> pi_ai::models::FetchModelsFn {
        Arc::new(move |_context| {
            let guard = Arc::clone(&self.0);
            Box::pin(async move {
                *lock(&guard) += 1;
                Ok(vec![test_model("dynamic", "fetched")])
            })
        })
    }
}
#[tokio::test]
async fn passes_effective_credentials_and_skips_unconfigured_providers() {
    let effective_credential: Arc<Mutex<Option<Credential>>> = Arc::new(Mutex::new(None));
    let force_refresh: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
    let unconfigured_refreshes = Arc::new(Mutex::new(0));
    let models = create_models(None);

    let credential_holder = Arc::clone(&effective_credential);
    let force_holder = Arc::clone(&force_refresh);
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(env_key_auth_only(Some("ambient-key"))),
        refresh: Some(refresh_future(move |context| {
            let credential_holder = Arc::clone(&credential_holder);
            let force_holder = Arc::clone(&force_holder);
            Box::pin(async move {
                if !context.allow_network {
                    return Ok(());
                }
                *lock(&credential_holder) = context.credential.clone();
                *lock(&force_holder) = context.force;
                Ok(())
            })
        })),
        ..default_input("configured")
    }));
    let unconfigured_counter = Arc::clone(&unconfigured_refreshes);
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(env_key_auth_only(None)),
        refresh: Some(refresh_future(move |context| {
            let counter = Arc::clone(&unconfigured_counter);
            Box::pin(async move {
                if context.allow_network {
                    *lock(&counter) += 1;
                }
                Ok(())
            })
        })),
        ..default_input("unconfigured")
    }));

    let result = models
        .refresh(Some(&ModelsRefreshOptions {
            force: Some(true),
            ..ModelsRefreshOptions::default()
        }))
        .await;
    assert_eq!(result.errors.len(), 0);
    let effective = lock(&effective_credential).clone();
    assert!(matches!(
        effective,
        Some(Credential::ApiKey(ApiKeyCredential { key: Some(key), env: None }))
            if key == "ambient-key"
    ));
    assert_eq!(*lock(&force_refresh), Some(true));
    assert_eq!(*lock(&unconfigured_refreshes), 0);
}

#[tokio::test]
async fn refreshes_expired_oauth_before_refreshing_models() {
    let credentials = InMemoryCredentialStore::default();
    store_credential(
        &credentials,
        "oauth-dynamic",
        Credential::OAuth(OAuthCredentials {
            refresh: "refresh".to_owned(),
            access: "expired".to_owned(),
            expires: 0,
            extra: BTreeMap::new(),
        }),
    )
    .await;
    let model_refresh_credential: Arc<Mutex<Option<Credential>>> = Arc::new(Mutex::new(None));
    let credential_holder = Arc::clone(&model_refresh_credential);
    let models = models_with_credentials(Arc::new(credentials));
    let oauth = test_oauth_with_refresh(Some(Arc::new(|_credential| {
        Box::pin(async move {
            Ok(OAuthCredentials {
                refresh: "rotated".to_owned(),
                access: "fresh".to_owned(),
                expires: now_ms() + 60_000,
                extra: BTreeMap::new(),
            })
        })
    })));
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(oauth_auth_only(oauth)),
        refresh: Some(refresh_future(move |context| {
            let holder = Arc::clone(&credential_holder);
            Box::pin(async move {
                if context.allow_network {
                    *lock(&holder) = context.credential.clone();
                }
                Ok(())
            })
        })),
        ..default_input("oauth-dynamic")
    }));

    let result = models.refresh(None).await;
    assert_eq!(result.errors.len(), 0);
    let recorded = lock(&model_refresh_credential).clone();
    let Some(Credential::OAuth(recorded)) = recorded else {
        panic!("expected the refreshed oauth credential to reach the model refresh");
    };
    assert_eq!(recorded.access, "fresh");
    assert_eq!(recorded.refresh, "rotated");
}

#[tokio::test]
async fn always_gives_providers_a_concrete_signal() {
    let received: Arc<Mutex<Option<CancellationToken>>> = Arc::new(Mutex::new(None));
    let holder = Arc::clone(&received);
    let models = create_models(None);
    register_refreshing(
        &models,
        "dynamic",
        refresh_future(move |context| {
            let holder = Arc::clone(&holder);
            Box::pin(async move {
                *lock(&holder) = Some(context.signal.clone());
                Ok(())
            })
        }),
    );

    let result = models.refresh(None).await;
    assert!(!result.aborted);
    let received_signal = lock(&received).clone();
    assert!(received_signal.is_some());
    assert!(!received_signal.unwrap().is_cancelled());
}

#[tokio::test]
async fn binds_model_store_waits_to_the_provider_refresh_signal() {
    let storage_signals: Arc<Mutex<Vec<CancellationToken>>> = Arc::new(Mutex::new(Vec::new()));
    let signals = Arc::clone(&storage_signals);
    let store: Arc<dyn ModelsStore> = Arc::new(RecordingStore {
        signals: signals.clone(),
        inner: Arc::new(InMemoryModelsStore::default()),
    });
    let provider_signal: Arc<Mutex<Option<CancellationToken>>> = Arc::new(Mutex::new(None));
    let signal_holder = Arc::clone(&provider_signal);
    let models = models_with_store(Arc::clone(&store));
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(env_key_auth_only(Some("key"))),
        refresh: Some(refresh_future(move |context| {
            let holder = Arc::clone(&signal_holder);
            Box::pin(async move {
                *lock(&holder) = Some(context.signal.clone());
                if !context.allow_network {
                    return Ok(());
                }
                publish_now(
                    &context,
                    ModelsPublication {
                        persist: CatalogPersist::Write(ModelsStoreEntry {
                            models: vec![test_model("dynamic", "fresh")],
                            ..ModelsStoreEntry::default()
                        }),
                        update: None,
                    },
                )
                .await;
                Ok(())
            })
        })),
        ..default_input("dynamic")
    }));

    let result = models
        .refresh(Some(&ModelsRefreshOptions {
            providers: Some(vec!["dynamic".to_owned()]),
            ..ModelsRefreshOptions::default()
        }))
        .await;

    assert_eq!(result.errors.len(), 0);
    let recorded = lock(&storage_signals).clone();
    assert_eq!(recorded.len(), 3);
    let provider_signal = lock(&provider_signal).clone();
    let Some(provider_signal) = provider_signal else {
        panic!("the provider refresh always carries a signal");
    };
    assert!(recorded.iter().all(|signal| signal == &provider_signal));
}

/// A store that records the signal every operation receives.
struct RecordingStore {
    signals: Arc<Mutex<Vec<CancellationToken>>>,
    inner: Arc<InMemoryModelsStore>,
}

impl RecordingStore {
    /// Records the signal each operation receives; the test asserts all
    /// three ops share the refresh signal.
    fn record(&self, options: Option<&ModelsStoreOptions>) {
        if let Some(options) = options
            && let Some(signal) = &options.signal
        {
            lock(&self.signals).push(signal.clone());
        }
    }
}

impl ModelsStore for RecordingStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<Option<ModelsStoreEntry>, pi_ai::models_store::ModelsStoreError>>
    {
        self.record(options);
        self.inner.read(provider_id, options)
    }

    fn write<'a>(
        &'a self,
        provider_id: &'a str,
        entry: ModelsStoreEntry,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<(), pi_ai::models_store::ModelsStoreError>> {
        self.record(options);
        self.inner.write(provider_id, entry, options)
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a ModelsStoreOptions>,
    ) -> BoxedFuture<'a, Result<(), pi_ai::models_store::ModelsStoreError>> {
        self.record(options);
        self.inner.delete(provider_id, options)
    }
}
#[tokio::test]
async fn returns_aborted_state_without_reporting_cancellation_as_error() {
    expect_aborted_refresh(refresh_future(|context| {
        Box::pin(async move {
            if context.signal.is_cancelled() {
                return Ok(());
            }
            Ok(())
        })
    }))
    .await;
}

#[tokio::test]
async fn stops_waiting_on_abort_when_a_provider_ignores_its_signal() {
    let calls = Arc::new(Mutex::new(0));
    let calls_holder = Arc::clone(&calls);
    expect_aborted_refresh(refresh_future(move |context| {
        let calls = Arc::clone(&calls_holder);
        Box::pin(async move {
            *lock(&calls) += 1;
            // A non-cooperative provider parks on its own work; the
            // runtime's drop-on-abort stops waiting regardless.
            let _ = context;
            Ok(())
        })
    }))
    .await;
}

#[tokio::test]
async fn rejects_late_publication_from_a_superseded_provider() {
    let store = InMemoryModelsStore::default();
    let state: Arc<Mutex<String>> = Arc::new(Mutex::new("initial".to_owned()));
    let models = models_with_store(Arc::new(store));
    let state_for_refresh = Arc::clone(&state);
    let calls = Arc::new(Mutex::new(0));
    let calls_for_provider = Arc::clone(&calls);
    register_refreshing(
        &models,
        "dynamic",
        refresh_future(move |context| {
            let state = Arc::clone(&state_for_refresh);
            let calls = Arc::clone(&calls_for_provider);
            Box::pin(async move {
                if !context.allow_network {
                    return Ok(());
                }
                let current = {
                    *lock(&calls) += 1;
                    *lock(&calls)
                };
                let value = format!("generation-{current}");
                publish_now(
                    &context,
                    ModelsPublication {
                        persist: CatalogPersist::Write(ModelsStoreEntry {
                            models: vec![test_model("dynamic", &value)],
                            ..ModelsStoreEntry::default()
                        }),
                        update: Some(Box::new({
                            let state = Arc::clone(&state);
                            move || {
                                *lock(&state) = value.clone();
                            }
                        })),
                    },
                )
                .await;
                Ok(())
            })
        }),
    );

    let first = spawn_refresh(&models, ModelsRefreshOptions::default());
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    let second = spawn_refresh(&models, ModelsRefreshOptions::default());
    let second = second.await.expect("joined");
    let first = first.await.expect("joined");
    assert_eq!(second.errors.len(), 0);
    assert_eq!(first.errors.len(), 0);

    assert_eq!(*lock(&state), "generation-2");
}

#[tokio::test]
async fn passes_caller_signals_to_provider_auth_callbacks() {
    let controller = CancellationToken::new();
    let received: Arc<Mutex<Vec<CancellationToken>>> = Arc::new(Mutex::new(Vec::new()));
    let holder = Arc::clone(&received);
    let models = create_models(None);
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(ProviderAuth {
            api_key: Some(signal_auth(holder)),
            oauth: None,
        }),
        ..default_input("p1")
    }));

    let options = AuthOptions {
        signal: Some(controller.clone()),
    };
    let _ = models
        .check_auth("p1", Some(&options))
        .await
        .expect("check");
    let _ = models
        .get_auth(
            "p1",
            Some(&AuthResolutionOverrides {
                signal: Some(controller.clone()),
                ..AuthResolutionOverrides::default()
            }),
        )
        .await
        .expect("get auth");
    let interaction = interaction_with_signal(controller.clone());
    let _ = models
        .login("p1", AuthType::ApiKey, interaction)
        .await
        .expect("login");

    let signals = lock(&received).clone();
    assert_eq!(signals.len(), 3);
    assert!(signals.iter().all(|signal| signal == &controller));
}

/// The api-key auth that records every signal it receives.
fn signal_auth(holder: Arc<Mutex<Vec<CancellationToken>>>) -> ApiKeyAuth {
    let login_holder = Arc::clone(&holder);
    ApiKeyAuth {
        name: "Signal auth".to_owned(),
        login: Some(Arc::new(
            move |interaction: pi_ai::auth::types::ProviderAuthInteraction| {
                let holder = Arc::clone(&login_holder);
                Box::pin(async move {
                    lock(&holder).push(interaction.signal.clone());
                    Ok(ApiKeyCredential {
                        key: Some("saved".to_owned()),
                        env: None,
                    })
                })
            },
        )),
        check: Some(Arc::new({
            let holder = Arc::clone(&holder);
            move |input: ApiKeyAuthInput| {
                let holder = Arc::clone(&holder);
                Box::pin(async move {
                    lock(&holder).push(input.signal.clone());
                    Ok(Some(pi_ai::auth::types::AuthCheck {
                        source: None,
                        auth_type: AuthType::ApiKey,
                    }))
                })
            }
        })),
        resolve: Arc::new(move |input: ApiKeyAuthInput| {
            let holder = Arc::clone(&holder);
            Box::pin(async move {
                lock(&holder).push(input.signal.clone());
                Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some("resolved".to_owned()),
                        ..ModelAuth::default()
                    },
                    env: None,
                    source: None,
                }))
            })
        }),
    }
}

/// The interaction fixture of the login flow, upstream's inline object.
fn interaction_with_signal(signal: CancellationToken) -> pi_ai::auth::types::AuthInteraction {
    pi_ai::auth::types::AuthInteraction {
        signal: Some(signal),
        prompt: Arc::new(|_prompt: AuthPrompt| prompt_placeholder()),
        notify: Arc::new(|_event| {}),
    }
}

fn prompt_placeholder() -> BoxedFuture<'static, Result<String, AbortError>> {
    Box::pin(async { Ok("unused".to_owned()) })
}
#[tokio::test]
async fn cancels_queued_credential_mutations_without_running_them_later() {
    let credentials = InMemoryCredentialStore::default();
    let (finish_first_sender, _finish_first_receiver) = tokio::sync::oneshot::channel::<()>();
    let finish_holder = Arc::new(Mutex::new(Some(finish_first_sender)));
    let first = credentials.modify(
        "p1",
        Box::new(move |_current| {
            let finish_holder = Arc::clone(&finish_holder);
            Box::pin(async move {
                let sender = lock(&finish_holder).take();
                if let Some(sender) = sender {
                    let _ = sender.send(());
                }
                Ok(Some(Credential::ApiKey(ApiKeyCredential {
                    key: Some("first".to_owned()),
                    env: None,
                })))
            })
        }),
        None,
    );
    let controller = CancellationToken::new();
    let second_holder = Arc::new(Mutex::new(None::<tokio::sync::oneshot::Sender<bool>>));
    let second_options = AuthOptions {
        signal: Some(controller.clone()),
    };
    let second = credentials.modify(
        "p1",
        Box::new(move |_current| {
            let done = lock(&second_holder).take();
            Box::pin(async move {
                if let Some(done) = done {
                    let _ = done.send(true);
                }
                Ok(Some(Credential::ApiKey(ApiKeyCredential {
                    key: Some("second".to_owned()),
                    env: None,
                })))
            })
        }),
        Some(&second_options),
    );

    controller.cancel();
    let second_result = second.await;
    assert!(second_result.is_err(), "an aborted queued mutation rejects");
    // Release the first mutation; the second must not run afterwards.
    let first_result = first.await;
    assert!(first_result.is_ok());

    // Give any queued task a beat; it must never run.
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    let stored = credentials.read("p1", None).await.expect("read");
    assert!(
        matches!(stored, Some(Credential::ApiKey(ApiKeyCredential { key: Some(key), .. })) if key == "first")
    );
}

#[tokio::test]
async fn passes_cancellation_to_oauth_refresh_and_preserves_the_previous_credential() {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    let previous = OAuthCredentials {
        refresh: "old-refresh".to_owned(),
        access: "old".to_owned(),
        expires: 0,
        extra: BTreeMap::new(),
    };
    store_credential(credentials.as_ref(), "p1", Credential::OAuth(previous)).await;
    let received_signal: Arc<Mutex<Option<CancellationToken>>> = Arc::new(Mutex::new(None));
    let signal_holder = Arc::clone(&received_signal);
    let models = models_with_credentials(Arc::clone(&credentials));
    let oauth = test_oauth_with_refresh(Some(Arc::new(move |_credential| {
        let signal_holder = Arc::clone(&signal_holder);
        Box::pin(async move {
            // The refresh is abandoned by the runtime at abort; it never
            // returns, so nothing rotates.
            let _ = signal_holder;
            pending_refresh().await
        })
    })));
    register_oauth_p1(&models, oauth);
    let controller = CancellationToken::new();
    let pending = {
        let models = models.clone();
        let signal = controller.clone();
        tokio::spawn(async move {
            models
                .get_auth(
                    "p1",
                    Some(&AuthResolutionOverrides {
                        signal: Some(signal),
                        ..AuthResolutionOverrides::default()
                    }),
                )
                .await
        })
    };
    // Wait for the refresh to observe its (still-live) signal, then abort.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    controller.cancel();
    let auth = pending.await.expect("joined");
    assert!(matches!(auth, Err(ModelsFailure::Aborted(_))));

    // The stored credential is preserved for retry / re-login.
    let stored = credentials.read("p1", None).await.expect("read");
    let Some(Credential::OAuth(stored)) = stored else {
        panic!("the oauth credential stays stored");
    };
    assert_eq!(stored.access, "old");
    assert_eq!(stored.refresh, "old-refresh");
}

/// The parked refresh the cancellation test drives; it never settles.
async fn pending_refresh() -> Result<OAuthCredentials, Box<dyn std::error::Error + Send + Sync>> {
    tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
    unreachable!("the parked refresh is cancelled, not completed")
}

#[tokio::test]
async fn resolves_auth_stored_credential_owns_the_provider() {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    let models = models_with_credentials(Arc::clone(&credentials));
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(ProviderAuth {
            oauth: Some(test_oauth()),
            ..env_key_auth_only(Some("env-key"))
        }),
        ..default_input("p1")
    }));
    let model = test_model("p1", "model-a");

    // model and provider-id overloads resolve the same provider-scoped auth
    assert_eq!(
        models
            .get_auth_for_model(&model, None)
            .await
            .expect("auth")
            .and_then(|result| result.auth.api_key),
        Some("env-key".to_owned())
    );
    assert_eq!(
        resolved_api_key(&models, "p1").await,
        Some("env-key".to_owned())
    );
    assert_eq!(
        models
            .get_auth_for_model(
                &model,
                Some(&AuthResolutionOverrides {
                    api_key: Some("explicit-key".to_owned()),
                    ..AuthResolutionOverrides::default()
                })
            )
            .await
            .expect("auth")
            .and_then(|result| result.auth.api_key),
        Some("explicit-key".to_owned())
    );

    // stored oauth credential (persisted via the single write path): beats
    // ambient env
    store_credential(
        credentials.as_ref(),
        "p1",
        Credential::OAuth(OAuthCredentials {
            refresh: "r".to_owned(),
            access: "oauth-token".to_owned(),
            expires: now_ms() + 10 * 60_000,
            extra: BTreeMap::new(),
        }),
    )
    .await;
    assert_eq!(
        resolved_api_key(&models, "p1").await,
        Some("oauth-token".to_owned())
    );

    let models = create_models(None);
    let _ = models;
}
#[tokio::test]
async fn checks_provider_auth_without_refreshing_and_filters_available() {
    let (models, credentials, refreshes) = counting_workspace();
    let refresh_count = Arc::clone(&refreshes);
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(env_key_auth_only(Some("env-key"))),
        ..default_input("ambient")
    }));
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(env_key_auth_only(None)),
        ..default_input("missing")
    }));
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(oauth_auth_only(test_oauth_with_refresh(Some(Arc::new(
            move |credential| {
                let count = Arc::clone(&refresh_count);
                Box::pin(async move {
                    *lock(&count) += 1;
                    Ok(credential)
                })
            },
        ))))),
        ..default_input("oauth")
    }));
    store_credential(
        credentials.as_ref(),
        "oauth",
        Credential::OAuth(OAuthCredentials {
            refresh: "refresh".to_owned(),
            access: "expired".to_owned(),
            expires: 0,
            extra: BTreeMap::new(),
        }),
    )
    .await;

    let ambient = models.check_auth("ambient", None).await.expect("check");
    assert_eq!(
        ambient.map(|check| (check.source, check.auth_type)),
        Some((Some("env".to_owned()), AuthType::ApiKey))
    );
    assert!(
        models
            .check_auth("missing", None)
            .await
            .expect("check")
            .is_none()
    );
    let oauth = models.check_auth("oauth", None).await.expect("check");
    assert_eq!(
        oauth.map(|check| (check.source, check.auth_type)),
        Some((Some("OAuth".to_owned()), AuthType::OAuth))
    );
    assert_eq!(*lock(&refreshes), 0);

    let available = models.available(None, None).await.expect("available");
    assert_eq!(
        available
            .iter()
            .map(|model| model.provider.0.clone())
            .collect::<Vec<_>>(),
        ["ambient", "oauth"]
    );
    let only_ambient = models
        .available(Some("ambient"), None)
        .await
        .expect("available");
    assert_eq!(
        only_ambient
            .iter()
            .map(|model| model.provider.0.clone())
            .collect::<Vec<_>>(),
        ["ambient"]
    );
}

#[tokio::test]
async fn runs_provider_login_and_logout_through_the_credential_store() {
    let credentials = InMemoryCredentialStore::default();
    let login_resolve: ApiKeyResolveFn =
        Arc::new(|_input: ApiKeyAuthInput| Box::pin(async move { Ok(None) }));
    let models = models_with_credentials(Arc::new(credentials));
    let logged_in_key = env_key_auth(None);
    let logged_in = {
        let mut logged_in = logged_in_key.clone();
        logged_in.login = Some(Arc::new(|_interaction| {
            Box::pin(async move {
                Ok(ApiKeyCredential {
                    key: Some("logged-in".to_owned()),
                    env: None,
                })
            })
        }));
        logged_in
    };
    let _ = login_resolve;
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(ProviderAuth {
            api_key: Some(logged_in),
            oauth: None,
        }),
        ..default_input("p1")
    }));

    let credential = models
        .login(
            "p1",
            AuthType::ApiKey,
            interaction_with_signal(CancellationToken::new()),
        )
        .await
        .expect("login");
    assert!(
        matches!(credential, Credential::ApiKey(ApiKeyCredential { key: Some(key), .. }) if key == "logged-in")
    );

    models.logout("p1", None).await.expect("logout");
    // The store no longer holds the credential; get_auth resolves nothing.
    assert!(models.get_auth("p1", None).await.expect("auth").is_none());
}

#[tokio::test]
async fn stored_credential_without_a_matching_handler_blocks_ambient_fallback() {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    let models = models_with_credentials(Arc::clone(&credentials));
    // provider has only apiKey auth, but an oauth credential is stored
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(env_key_auth_only(Some("env-key"))),
        ..default_input("p1")
    }));
    store_credential(
        credentials.as_ref(),
        "p1",
        Credential::OAuth(OAuthCredentials {
            refresh: "r".to_owned(),
            access: "a".to_owned(),
            expires: 0,
            extra: BTreeMap::new(),
        }),
    )
    .await;

    assert!(models.get_auth("p1", None).await.expect("auth").is_none());
}
#[tokio::test]
async fn refreshes_expired_oauth_and_persists_the_rotated_credential() {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    let models = create_models(Some(pi_ai::models::CreateModelsOptions {
        credentials: Some(Arc::clone(&credentials)),
        ..pi_ai::models::CreateModelsOptions::default()
    }));
    let oauth = test_oauth_with_refresh(Some(Arc::new(|credential| {
        Box::pin(async move {
            Ok(OAuthCredentials {
                refresh: credential.refresh,
                access: "new-token".to_owned(),
                expires: now_ms() + 60 * 60_000,
                extra: credential.extra,
            })
        })
    })));
    register_oauth_p1(&models, oauth);
    store_credential(
        credentials.as_ref(),
        "p1",
        Credential::OAuth(old_token_credential(0)),
    )
    .await;

    assert_eq!(
        resolved_api_key(&models, "p1").await,
        Some("new-token".to_owned())
    );
    let stored = credentials.read("p1", None).await.expect("read");
    let Some(Credential::OAuth(stored)) = stored else {
        panic!("the rotated credential persists");
    };
    assert_eq!(stored.access, "new-token");
}

#[tokio::test]
async fn refreshes_oauth_with_less_than_five_minutes_remaining() {
    let (models, refreshes) = setup_counting_oauth(now_ms() + 60_000).await;

    assert_eq!(
        resolved_api_key(&models, "p1").await,
        Some("new-token".to_owned())
    );
    assert_eq!(*lock(&refreshes), 1);
}

#[tokio::test]
async fn honors_a_callers_longer_oauth_minimum_validity() {
    let (models, refreshes) = setup_counting_oauth(now_ms() + 10 * 60_000).await;

    let resolution = models
        .get_auth(
            "p1",
            Some(&AuthResolutionOverrides {
                min_oauth_validity_ms: Some(30 * 60_000),
                ..AuthResolutionOverrides::default()
            }),
        )
        .await
        .expect("auth");
    assert_eq!(
        resolution.and_then(|result| result.auth.api_key),
        Some("new-token".to_owned())
    );
    assert_eq!(*lock(&refreshes), 1);
}

#[tokio::test]
async fn rejects_with_oauth_code_when_refresh_fails_and_preserves_the_credential() {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    let models = models_with_credentials(Arc::clone(&credentials));
    let oauth = test_oauth_with_refresh(Some(Arc::new(|_credential| {
        Box::pin(async move { Err(fail_error("invalid_grant")) })
    })));
    register_oauth_p1(&models, oauth);
    store_credential(
        credentials.as_ref(),
        "p1",
        Credential::OAuth(old_credential()),
    )
    .await;

    let auth = models.get_auth("p1", None).await;
    let Err(ModelsFailure::Models(error)) = auth else {
        panic!("the refresh failure rejects");
    };
    assert_eq!(error.code(), ModelsErrorCode::OAuth);
    // credential preserved for retry / re-login
    let stored = credentials.read("p1", None).await.expect("read");
    let Some(Credential::OAuth(stored)) = stored else {
        panic!("the old credential stays");
    };
    assert_eq!(stored.access, "old");
}

#[tokio::test]
async fn serializes_concurrent_oauth_refreshes_through_store_modify() {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    store_credential(
        credentials.as_ref(),
        "p1",
        Credential::OAuth(OAuthCredentials {
            refresh: "r1".to_owned(),
            access: "old".to_owned(),
            expires: 0,
            extra: BTreeMap::new(),
        }),
    )
    .await;

    let refreshes = Arc::new(Mutex::new(0));
    let models = models_with_credentials(Arc::clone(&credentials));
    let refresh_count = Arc::clone(&refreshes);
    let oauth = test_oauth_with_refresh(Some(Arc::new(move |_credential| {
        let count = Arc::clone(&refresh_count);
        Box::pin(async move {
            *lock(&count) += 1;
            // Mirror upstream's 10ms pause inside the refresh: the second
            // caller must wait under the store lock.
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            Ok(OAuthCredentials {
                refresh: "r2".to_owned(),
                access: format!("new-{}", 1),
                expires: now_ms() + 60 * 60_000,
                extra: BTreeMap::new(),
            })
        })
    })));
    register_oauth_p1(&models, oauth);
    let model = test_model("p1", "model-a");

    let (a, b) = tokio::join!(
        models.get_auth_for_model(&model, None),
        models.get_auth_for_model(&model, None),
    );
    assert_eq!(*lock(&refreshes), 1);
    assert_eq!(
        a.expect("auth").and_then(|result| result.auth.api_key),
        Some("new-1".to_owned())
    );
    assert_eq!(
        b.expect("auth").and_then(|result| result.auth.api_key),
        Some("new-1".to_owned())
    );
}
#[tokio::test]
async fn valid_oauth_tokens_resolve_without_touching_modify() {
    // The setup write lands on the inner store so the counting wrapper sees
    // only the calls the resolution makes, upstream's `base` split.
    let base: Arc<InMemoryCredentialStore> = Arc::new(InMemoryCredentialStore::default());
    store_credential(
        base.as_ref(),
        "p1",
        Credential::OAuth(OAuthCredentials {
            refresh: "r".to_owned(),
            access: "valid".to_owned(),
            expires: now_ms() + 10 * 60_000,
            extra: BTreeMap::new(),
        }),
    )
    .await;
    let modifies = Arc::new(Mutex::new(0));
    let modifies_count = Arc::clone(&modifies);
    let credentials: Arc<dyn CredentialStore> = Arc::new(CountingStore {
        inner: Arc::clone(&base),
        modifies: modifies_count,
    });
    let models = models_with_credentials(Arc::clone(&credentials));
    register_oauth_p1(&models, test_oauth());

    assert_eq!(
        resolved_api_key(&models, "p1").await,
        Some("valid".to_owned())
    );
    assert_eq!(*lock(&modifies), 0);
}

/// The credential store that counts `modify` calls, upstream's inline
/// wrapper object.
struct CountingStore {
    inner: Arc<InMemoryCredentialStore>,
    modifies: Arc<Mutex<u32>>,
}

impl CredentialStore for CountingStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, Box<dyn std::error::Error + Send + Sync>>> {
        self.inner.read(provider_id, options)
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<
        'a,
        Result<Vec<pi_ai::auth::types::CredentialInfo>, Box<dyn std::error::Error + Send + Sync>>,
    > {
        self.inner.list(options)
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: pi_ai::auth::types::CredentialModifyFn,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, Box<dyn std::error::Error + Send + Sync>>> {
        *lock(&self.modifies) += 1;
        self.inner.modify(provider_id, f, options)
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        self.inner.delete(provider_id, options)
    }
}

#[tokio::test]
async fn wraps_credential_store_failures_in_models_error() {
    // read failure
    let models = models_with_credentials(read_fail_store());
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(env_key_auth_only(Some("env-key"))),
        ..default_input("p1")
    }));
    let auth = models.get_auth("p1", None).await;
    expect_wrapped_auth_error(auth);

    // modify failure during refresh
    let oauth_models = models_with_credentials(Arc::new(FailingCredentialStore {
        fail_on: FailureKind::Modify,
    }));
    register_oauth_p1(&oauth_models, test_oauth());
    let auth = oauth_models.get_auth("p1", None).await;
    expect_wrapped_auth_error(auth);
}

fn read_fail_store() -> Arc<dyn CredentialStore> {
    Arc::new(FailingCredentialStore {
        fail_on: FailureKind::Read,
    })
}

/// Which operation the failing store fails on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FailureKind {
    /// `read` always fails.
    Read,
    /// `modify` always fails.
    Modify,
}

/// A credential store whose listed operation always fails.
struct FailingCredentialStore {
    fail_on: FailureKind,
}

impl FailingCredentialStore {
    fn fail(&self, kind: FailureKind) -> Option<Box<dyn std::error::Error + Send + Sync>> {
        if self.fail_on == kind {
            return Some(Box::new(std::io::Error::other("disk on fire")));
        }
        None
    }
}

impl CredentialStore for FailingCredentialStore {
    fn read<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, Box<dyn std::error::Error + Send + Sync>>> {
        let failure = self.fail(FailureKind::Read);
        Box::pin(async move {
            if let Some(error) = failure {
                return Err(error);
            }
            Ok(Some(Credential::OAuth(OAuthCredentials {
                refresh: "r".to_owned(),
                access: "old".to_owned(),
                expires: 0,
                extra: BTreeMap::new(),
            })))
        })
    }

    fn list<'a>(
        &'a self,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<
        'a,
        Result<Vec<pi_ai::auth::types::CredentialInfo>, Box<dyn std::error::Error + Send + Sync>>,
    > {
        Box::pin(async move { Ok(Vec::new()) })
    }

    fn modify<'a>(
        &'a self,
        _provider_id: &'a str,
        _f: pi_ai::auth::types::CredentialModifyFn,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, Box<dyn std::error::Error + Send + Sync>>> {
        let failure = self.fail(FailureKind::Modify);
        Box::pin(async move { failure.map_or_else(|| Ok(None), Err) })
    }

    fn delete<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), Box<dyn std::error::Error + Send + Sync>>> {
        Box::pin(async { Ok(()) })
    }
}
#[tokio::test]
async fn keeps_the_underlying_reason_in_wrapped_oauth_refresh_errors() {
    let credentials: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    store_credential(
        credentials.as_ref(),
        "p1",
        Credential::OAuth(old_credential()),
    )
    .await;
    let models = models_with_credentials(Arc::clone(&credentials));
    let oauth = test_oauth_with_refresh(Some(Arc::new(|_credential| {
        Box::pin(async move { Err(fail_error("token refresh failed (400): invalid_grant")) })
    })));
    register_oauth_p1(&models, oauth);

    let auth = models.get_auth("p1", None).await;
    let Err(ModelsFailure::Models(error)) = auth else {
        panic!("the refresh failure rejects");
    };
    assert_eq!(
        error.to_string(),
        "OAuth refresh failed for p1: token refresh failed (400): invalid_grant"
    );
}

#[tokio::test]
async fn wraps_api_key_auth_failures_in_models_error() {
    let failing = ApiKeyAuth {
        name: "Failing".to_owned(),
        login: None,
        check: None,
        resolve: Arc::new(|_input: ApiKeyAuthInput| {
            Box::pin(async move { Err(fail_error("nope")) })
        }),
    };
    let models = create_models(None);
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(ProviderAuth {
            api_key: Some(failing),
            oauth: None,
        }),
        ..default_input("p1")
    }));
    let auth = models.get_auth("p1", None).await;
    expect_wrapped_auth_error(auth);
}

#[tokio::test]
async fn uses_explicit_request_api_key_and_env_during_resolution() {
    let calls: Arc<Mutex<Vec<ProviderCall>>> = Arc::new(Mutex::new(Vec::new()));
    let calls_holder = Arc::clone(&calls);
    let scoped_resolve: ApiKeyResolveFn = Arc::new(move |input: ApiKeyAuthInput| {
        Box::pin(async move {
            let account = input
                .credential
                .as_ref()
                .and_then(|credential| credential.env.as_ref())
                .and_then(|env| env.get("ACCOUNT_ID").cloned())
                .or_else(|| input.ctx.env("ACCOUNT_ID"));
            let key = input
                .credential
                .as_ref()
                .and_then(|credential| credential.key.clone());
            let (Some(account), Some(key)) = (account, key) else {
                return Ok(None);
            };
            Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key),
                    base_url: Some(format!("https://example.test/{account}")),
                    ..ModelAuth::default()
                },
                env: Some([("ACCOUNT_ID".to_owned(), account)].into()),
                source: None,
            }))
        })
    });
    let models = create_models(None);
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(ProviderAuth {
            api_key: Some(ApiKeyAuth {
                name: "Scoped".to_owned(),
                login: None,
                check: None,
                resolve: scoped_resolve,
            }),
            oauth: None,
        }),
        calls: Some(Arc::clone(&calls_holder)),
        ..default_input("p1")
    }));
    let model = test_model("p1", "model-a");

    let complete = models
        .complete_simple(
            &model,
            &context(),
            Some(&ModelsSimpleStreamOptions {
                options: simple_options_with_env("explicit-key"),
                ..ModelsSimpleStreamOptions::default()
            }),
        )
        .await;
    assert_eq!(complete.stop_reason, StopReason::Stop);

    let recorded = lock(&calls);
    assert_eq!(recorded[0].model.base_url, "https://example.test/acct");
    assert_eq!(
        recorded[0]
            .options
            .as_ref()
            .and_then(|options| options.api_key.clone()),
        Some("explicit-key".to_owned())
    );
    assert_eq!(
        recorded[0]
            .options
            .as_ref()
            .and_then(|options| options.env.clone()),
        Some([("ACCOUNT_ID".to_owned(), "acct".to_owned())].into())
    );
}

#[tokio::test]
async fn merges_resolved_auth_into_stream_options() {
    let calls: Arc<Mutex<Vec<ProviderCall>>> = Arc::new(Mutex::new(Vec::new()));
    let calls_holder = Arc::clone(&calls);
    let resolved_resolve: ApiKeyResolveFn = Arc::new(move |_input: ApiKeyAuthInput| {
        Box::pin(async move {
            Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some("resolved-key".to_owned()),
                    headers: Some(
                        [
                            (
                                "Authorization".to_owned(),
                                Some("Bearer resolved-key".to_owned()),
                            ),
                            ("x-a".to_owned(), Some("auth".to_owned())),
                            ("x-b".to_owned(), Some("auth".to_owned())),
                        ]
                        .into(),
                    ),
                    base_url: Some("https://auth.test/v1".to_owned()),
                },
                env: None,
                source: None,
            }))
        })
    });
    let models = create_models(None);
    models.set_provider(test_provider(TestProviderInput {
        auth: Some(ProviderAuth {
            api_key: Some(ApiKeyAuth {
                name: "Test".to_owned(),
                login: None,
                check: None,
                resolve: resolved_resolve,
            }),
            oauth: None,
        }),
        calls: Some(Arc::clone(&calls_holder)),
        ..default_input("p1")
    }));
    let model = test_model("p1", "model-a");

    let result = models
        .complete_simple(
            &model,
            &context(),
            Some(&ModelsSimpleStreamOptions {
                options: SimpleStreamOptions {
                    api_key: Some("explicit-key".to_owned()),
                    headers: Some(BTreeMap::from([
                        (
                            "authorization".to_owned(),
                            Some("Explicit token".to_owned()),
                        ),
                        ("x-b".to_owned(), Some("explicit".to_owned())),
                    ])),
                    ..SimpleStreamOptions::default()
                },
                transform_headers: None,
            }),
        )
        .await;
    assert_eq!(result.stop_reason, StopReason::Stop);
    let recorded = {
        let recorded = lock(&calls);
        (
            recorded.len(),
            recorded[0]
                .options
                .as_ref()
                .and_then(|options| options.api_key.clone()),
            recorded[0]
                .options
                .as_ref()
                .and_then(|options| options.headers.clone()),
        )
    };
    assert_eq!(recorded.0, 1);
    assert_eq!(recorded.1, Some("explicit-key".to_owned()));
    let headers = recorded.2.expect("merged headers");
    assert_eq!(
        headers.get("authorization"),
        Some(&Some("Explicit token".to_owned()))
    );
    assert_eq!(headers.get("x-a"), Some(&Some("auth".to_owned())));
    assert_eq!(headers.get("x-b"), Some(&Some("explicit".to_owned())));

    // without explicit options, resolved auth applies
    let result2 = models.complete_simple(&model, &context(), None).await;
    assert_eq!(result2.stop_reason, StopReason::Stop);
    let recorded2 = lock(&calls);
    assert_eq!(
        recorded2[1]
            .options
            .as_ref()
            .and_then(|options| options.api_key.clone()),
        Some("resolved-key".to_owned())
    );
}

#[tokio::test]
async fn produces_an_error_stream_for_unknown_providers_instead_of_throwing() {
    let models = create_models(None);
    let result = models
        .complete_simple(&test_model("ghost", "model-a"), &context(), None)
        .await;
    assert_eq!(result.stop_reason, StopReason::Error);
    let message = error_message(&result);
    assert!(
        message.contains("Unknown provider: ghost"),
        "got: {message}"
    );
}

fn error_message(message: &AssistantMessage) -> String {
    message.error_message.clone().unwrap_or_default()
}

#[tokio::test]
async fn streams_through_the_provider() {
    let models = create_models(None);
    models.set_provider(test_provider(default_input("p1")));
    let model = test_model("p1", "model-a");

    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let events_holder = Arc::clone(&events);
    let stream = models.stream_simple(&model, &context(), None);
    let drain = tokio::spawn(async move {
        while let Some(event) = stream.next().await {
            match event {
                pi_ai::types::AssistantMessageEvent::Start { .. } => {
                    lock(&events_holder).push("start".to_owned());
                }
                pi_ai::types::AssistantMessageEvent::Done { .. } => {
                    lock(&events_holder).push("done".to_owned());
                }
                pi_ai::types::AssistantMessageEvent::Error { .. } => {
                    lock(&events_holder).push("error".to_owned());
                }
                _ => {}
            }
        }
    });
    drain.await.expect("drain");
    assert_eq!(*lock(&events), ["start", "done"]);
}

use pi_ai::models::ModelsSimpleStreamOptions;

/// The simple options fixture carrying an api key and the `ACCOUNT_ID` env.
fn simple_options_with_env(api_key: &str) -> SimpleStreamOptions {
    SimpleStreamOptions {
        api_key: Some(api_key.to_owned()),
        env: Some(BTreeMap::from([(
            "ACCOUNT_ID".to_owned(),
            "acct".to_owned(),
        )])),
        ..SimpleStreamOptions::default()
    }
}

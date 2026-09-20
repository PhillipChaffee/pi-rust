//! The remaining #28 edges: provider-core refresh arms, deferred dispatch
//! failures, login/logout aborts, auth-resolution and credential-store
//! failures, the env-key registry fallbacks, and the images error surfaces,
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::helpers::{LazyOAuthInput, env_api_key_auth, lazy_oauth};
use pi_ai::auth::resolve::ModelsFailure;
use pi_ai::auth::resolve::{AuthResolutionOverrides, now_ms};
use pi_ai::auth::types::{
    ApiKeyAuth, ApiKeyAuthInput, AuthInteraction, Credential, ModelAuth, OAuthAuth,
    OAuthCredentials, ProviderAuth, ProviderAuthInteraction,
};
use pi_ai::models::{
    CatalogPersist, CreateProviderOptions, FetchModelsFn, ModelsSimpleStreamOptions, ProviderApi,
    ProviderModelError, PublishFn, RefreshModelsContext, create_models, create_provider,
};
use pi_ai::models_store::ModelsStoreEntry;
use pi_ai::providers::all::{builtin_models_of, builtin_provider_id};
use pi_ai::types::{
    Api, BoxedFuture, Context, Model, ModelThinkingLevel, ProviderHeaders, ProviderId,
    ProviderStreams, SimpleStreamOptions, StopReason, StreamOptions,
};

/// The images generate calls a fixture records: the request model's
/// `base_url` and the merged headers.
type ImagesCalls = Arc<Mutex<Vec<(String, Option<ProviderHeaders>)>>>;

use common::{
    ambient_auth, block, context, deferred_handle, end_with, fixture_model, image_model,
    images_context, ok_images_result,
};

mod common;

fn resolving_auth_with_headers(key: &str) -> ProviderAuth {
    let key = key.to_owned();
    ProviderAuth {
        api_key: Some(ApiKeyAuth {
            name: "Test".to_owned(),
            login: None,
            check: None,
            resolve: Arc::new(move |_input: ApiKeyAuthInput| {
                let key = key.clone();
                Box::pin(async move {
                    Ok(Some(pi_ai::auth::types::AuthResult {
                        auth: ModelAuth {
                            api_key: Some(key),
                            headers: Some(BTreeMap::from([(
                                "x-provider".to_owned(),
                                Some("provider-value".to_owned()),
                            )])),
                            ..ModelAuth::default()
                        },
                        env: None,
                        source: None,
                    }))
                })
            }),
        }),
        oauth: None,
    }
}

/// A plain streams fixture that records the simple-stream options it saw.
struct RecordingStreams {
    simple_options: Arc<Mutex<Vec<Option<SimpleStreamOptions>>>>,
}

impl ProviderStreams for RecordingStreams {
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
        options: Option<&SimpleStreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        self.simple_options
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(options.cloned());
        end_with(model)
    }
}

fn recording_streams() -> Arc<RecordingStreams> {
    Arc::new(RecordingStreams {
        simple_options: Arc::new(Mutex::new(Vec::new())),
    })
}

/// A deferred-capable streams fixture with independently flippable support.
struct SelectiveDeferredStreams {
    fetch_support: bool,
    cancel_support: bool,
}

impl ProviderStreams for SelectiveDeferredStreams {
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

    fn supports_fetch_deferred(&self) -> bool {
        self.fetch_support
    }

    fn supports_cancel_deferred(&self) -> bool {
        self.cancel_support
    }
}

fn no_publish() -> PublishFn {
    // The Models publication runs the synchronous update after persisting;
    // mirror that so provider refreshes apply their fetched lists.
    Arc::new(|publication| {
        Box::pin(async move {
            if let Some(update) = publication.update {
                update();
            }
            Ok(true)
        })
    })
}

fn provider_with(
    id: &str,
    models: Vec<Model>,
    api: ProviderApi,
    fetch_models: Option<FetchModelsFn>,
    auth: ProviderAuth,
) -> pi_ai::models::ProviderImpl {
    create_provider(CreateProviderOptions {
        id: id.to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth,
        models,
        fetch_models,
        api,
        filter_models: None,
    })
}

/// The Debug impls of the publication, images, and auth types render every
/// arm, and the `Credential` accessors report per-variant shapes.
#[test]
fn the_debug_impls_render_every_variant() {
    use pi_ai::models::ModelsPublication;
    let publication = ModelsPublication::default();
    assert!(format!("{publication:?}").contains("CatalogPersist"));

    assert_eq!(format!("{:?}", CatalogPersist::Omit), "Omit");
    assert_eq!(format!("{:?}", CatalogPersist::Delete), "Delete");
    assert_eq!(
        format!("{:?}", CatalogPersist::Write(ModelsStoreEntry::default())),
        "Write(..)"
    );

    let images = pi_ai::images_models::create_images_models(None);
    assert!(format!("{images:?}").contains("ImagesModelsState"));
    let empty_options = pi_ai::images_models::CreateImagesModelsOptions::default();
    assert!(format!("{empty_options:?}").contains("CreateImagesModelsOptions"));
    let injected = injected_images_options();
    let rendered = format!("{injected:?}");
    assert!(rendered.contains("set"), "got: {rendered}");

    let key_auth = ApiKeyAuth {
        name: "Key".to_owned(),
        login: None,
        check: None,
        resolve: Arc::new(|_input: ApiKeyAuthInput| Box::pin(async move { Ok(None) })),
    };
    assert!(format!("{key_auth:?}").contains("ApiKeyAuth"));
    let input = ApiKeyAuthInput {
        ctx: pi_ai::auth::context::default_provider_auth_context(),
        credential: Some(pi_ai::auth::types::ApiKeyCredential {
            key: Some("k".to_owned()),
            env: None,
        }),
        signal: tokio_util::sync::CancellationToken::new(),
    };
    assert!(format!("{input:?}").contains("ApiKeyAuthInput"));
    assert!(!format!("{key_auth:?}").contains("OAuth"));

    let oauth = oauth_auth();
    assert!(format!("{oauth:?}").contains("OAuth"));

    let signal = tokio_util::sync::CancellationToken::new();
    let interaction = AuthInteraction {
        signal: Some(signal.clone()),
        prompt: Arc::new(|_prompt| Box::pin(async move { Ok("entered".to_owned()) })),
        notify: Arc::new(|_event: pi_ai::auth::types::AuthEvent| {}),
    };
    assert!(format!("{interaction:?}").contains("AuthInteraction"));
    let normalized = ProviderAuthInteraction::from_interaction(interaction, signal);
    assert!(format!("{normalized:?}").contains("ProviderAuthInteraction"));

    let lazy_input = LazyOAuthInput {
        name: "Lazy".to_owned(),
        is_subscription: None,
        login_label: None,
        load: Arc::new(|| Box::pin(async move { oauth_auth() })),
    };
    assert!(format!("{lazy_input:?}").contains("Lazy"));

    let api_key_credential = Credential::ApiKey(pi_ai::auth::types::ApiKeyCredential {
        key: Some("k".to_owned()),
        env: None,
    });
    assert_eq!(api_key_credential.api_key(), Some("k"));
    assert!(api_key_credential.as_api_key_credential().is_some());
    assert!(api_key_credential.as_oauth().is_none());
    let oauth_credential = Credential::OAuth(OAuthCredentials {
        refresh: "r".to_owned(),
        access: "a".to_owned(),
        expires: now_ms() + 1_000,
        extra: BTreeMap::new(),
    });
    assert_eq!(oauth_credential.api_key(), None);
    assert!(oauth_credential.as_api_key_credential().is_none());
    assert!(oauth_credential.as_oauth().is_some());
}

/// Auth lookups for unknown providers resolve to `None`, never an error.
#[test]
fn the_auth_queries_answer_unknown_providers_with_none() {
    let models = create_models(None);
    let ghost_auth = block(models.get_auth("ghost", None));
    assert!(ghost_auth.expect("ghost auth").is_none());

    let mut ghost_model = fixture_model();
    ghost_model.provider = ProviderId::from("ghost");
    let ghost_model_auth = block(models.get_auth_for_model(&ghost_model, None));
    assert!(ghost_model_auth.expect("ghost model auth").is_none());
}

/// `applyAuth` merges model-declared headers with resolved provider headers
/// and runs the Models-level header transform last.
#[test]
fn the_models_apply_auth_merges_model_headers_and_transforms() {
    let simple_options = Arc::new(Mutex::new(Vec::new()));
    let models = create_models(None);
    models.set_provider(Arc::new(provider_with(
        "p",
        vec![Model {
            headers: Some(BTreeMap::from([(
                "x-model".to_owned(),
                "model-value".to_owned(),
            )])),
            ..fixture_model()
        }],
        ProviderApi::Single(Arc::new(RecordingStreams {
            simple_options: Arc::clone(&simple_options),
        })),
        None,
        resolving_auth_with_headers("resolved-key"),
    )));

    let model = Model {
        headers: Some(BTreeMap::from([(
            "x-model".to_owned(),
            "model-value".to_owned(),
        )])),
        ..fixture_model()
    };
    let result = block(async {
        models
            .complete_simple(
                &model,
                &context(),
                Some(&ModelsSimpleStreamOptions {
                    options: SimpleStreamOptions::default(),
                    transform_headers: Some(Arc::new(|headers| {
                        Box::pin(async move {
                            let mut transformed = headers;
                            transformed.insert("x-transformed".to_owned(), Some("yes".to_owned()));
                            transformed
                        })
                    })),
                }),
            )
            .await
    });
    assert_eq!(result.stop_reason, StopReason::Stop);
    let recorded = simple_options
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .len();
    assert_eq!(recorded, 1);
}

/// The clamp falls through both neighbors to `Off` when a reasoning model
/// disables every level.
#[test]
fn the_thinking_level_clamp_falls_back_to_off() {
    use pi_ai::models::clamp_thinking_level;
    let model = Model {
        reasoning: true,
        thinking_level_map: Some(BTreeMap::from([
            (ModelThinkingLevel::Off, None),
            (ModelThinkingLevel::Minimal, None),
            (ModelThinkingLevel::Low, None),
            (ModelThinkingLevel::Medium, None),
            (ModelThinkingLevel::High, None),
            (ModelThinkingLevel::Xhigh, None),
            (ModelThinkingLevel::Max, None),
        ])),
        ..fixture_model()
    };
    assert_eq!(
        clamp_thinking_level(&model, ModelThinkingLevel::High),
        ModelThinkingLevel::Off
    );
}

/// The provider-core refresh arms: no fetch is a no-op, a same-id fetch
/// upserts over the baseline, an unpublished restore keeps storage unchanged,
/// and a cancelled signal after the fetch keeps the previous list.
#[test]
fn the_provider_core_refresh_arms_behave() {
    use pi_ai::models::Provider;

    let static_provider = provider_with(
        "static",
        vec![fixture_model()],
        ProviderApi::Single(recording_streams()),
        None,
        ambient_auth(),
    );
    let outcome = block(static_provider.refresh_models(refresh_context()));
    assert!(outcome.is_ok());

    // A fetch returning the baseline id upserts in place.
    let refreshed_name = "refreshed-name".to_owned();
    let upserting = provider_with(
        "dyn",
        vec![fixture_model()],
        ProviderApi::Single(recording_streams()),
        Some(Arc::new(move |_context| {
            let name = refreshed_name.clone();
            Box::pin(async move {
                Ok(vec![Model {
                    name,
                    ..fixture_model()
                }])
            })
        })),
        ambient_auth(),
    );
    let outcome = block(upserting.refresh_models(refresh_context()));
    assert!(outcome.is_ok());
    let listed = upserting.get_models().expect("models listed");
    assert_eq!(listed[0].name, "refreshed-name");

    // A rejected publication ends the restore quietly.
    let restoring = provider_with(
        "restore",
        vec![fixture_model()],
        ProviderApi::Single(recording_streams()),
        None,
        ambient_auth(),
    );
    let outcome = block(restoring.refresh_models(RefreshModelsContext {
        credential: None,
        stored: Some(ModelsStoreEntry {
            models: vec![fixture_model()],
            ..ModelsStoreEntry::default()
        }),
        publish: Arc::new(|_publication| Box::pin(async move { Ok(false) })),
        allow_network: false,
        force: None,
        signal: tokio_util::sync::CancellationToken::new(),
    }));
    assert!(outcome.is_ok());

    // Cancelling the signal during the fetch keeps the previous list.
    let cancelling = provider_with(
        "cancel",
        vec![fixture_model()],
        ProviderApi::Single(recording_streams()),
        Some(Arc::new(|context: &RefreshModelsContext| {
            context.signal.cancel();
            Box::pin(async move { Ok(vec![fixture_model()]) })
        })),
        ambient_auth(),
    );
    let outcome = block(cancelling.refresh_models(refresh_context()));
    assert!(outcome.is_ok());
    let listed = cancelling.get_models().expect("models listed");
    assert_eq!(listed.len(), 1);

    // The ByApi map's every-stream walk reports its own support flags.
    let mut by_api_streams = BTreeMap::<String, Arc<dyn ProviderStreams>>::new();
    by_api_streams.insert("test-api".to_owned(), recording_streams());
    let by_api = provider_with(
        "byapi",
        vec![fixture_model()],
        ProviderApi::ByApi(by_api_streams),
        None,
        ambient_auth(),
    );
    assert!(!by_api.supports_fetch_deferred());
    assert!(!by_api.supports_cancel_deferred());
}

/// The deferred dispatch reports missing implementations per arm: a model
/// whose api has no streams entry, streams without deferred support, and a
/// selective fixture that advertises one capability but not the other.
#[test]
fn the_deferred_dispatch_reports_missing_implementations() {
    use pi_ai::models::Provider;

    let deferred_streams: Arc<dyn ProviderStreams> = Arc::new(SelectiveDeferredStreams {
        fetch_support: true,
        cancel_support: true,
    });
    let plain_streams: Arc<dyn ProviderStreams> = Arc::new(SelectiveDeferredStreams {
        fetch_support: false,
        cancel_support: false,
    });
    let mut dispatch_streams = BTreeMap::<String, Arc<dyn ProviderStreams>>::new();
    dispatch_streams.insert("test-api".to_owned(), Arc::clone(&deferred_streams));
    dispatch_streams.insert("plain-api".to_owned(), Arc::clone(&plain_streams));
    let provider = provider_with(
        "p",
        vec![fixture_model()],
        ProviderApi::ByApi(dispatch_streams),
        None,
        ambient_auth(),
    );

    // A model whose api has no entry streams the missing-API error.
    let mut ghost = fixture_model();
    ghost.api = Api::from("absent-api");
    let streamed = block(async {
        provider
            .stream_simple(&ghost, &context(), None)
            .result()
            .await
    });
    assert!(
        streamed
            .error_message
            .unwrap_or_default()
            .contains("has no API implementation")
    );

    // No deferred support anywhere: the dispatch reports None.
    let plain_provider = provider_with(
        "plain",
        vec![fixture_model()],
        ProviderApi::Single(Arc::clone(&plain_streams)),
        None,
        ambient_auth(),
    );
    // The error arms of the lazy fetch never consult the handle, so the shared
    // fixture handle serves.
    let handle = deferred_handle();
    let plain_model = Model {
        api: Api::from("plain-api"),
        ..fixture_model()
    };
    assert!(
        plain_provider
            .fetch_deferred(&fixture_model(), &handle, None)
            .is_none()
    );
    assert!(
        plain_provider
            .cancel_deferred(&fixture_model(), &handle, None)
            .is_none()
    );

    // The lazy fetch reports the missing implementation per arm: an absent
    // api entry, and an entry whose streams lack the support.
    let fetch_absent = block(async {
        provider
            .fetch_deferred(&ghost, &handle, None)
            .expect("deferred stream")
            .result()
            .await
    });
    assert!(
        fetch_absent
            .error_message
            .unwrap_or_default()
            .contains("does not support deferred responses")
    );
    let fetch_unsupported = block(async {
        provider
            .fetch_deferred(&plain_model, &handle, None)
            .expect("deferred stream")
            .result()
            .await
    });
    assert!(
        fetch_unsupported
            .error_message
            .unwrap_or_default()
            .contains("does not support deferred responses")
    );

    // The cancel dispatch reports None per arm.
    assert!(provider.cancel_deferred(&ghost, &handle, None).is_none());
    assert!(
        provider
            .cancel_deferred(&plain_model, &handle, None)
            .is_none()
    );
}

fn oauth_auth() -> OAuthAuth {
    OAuthAuth {
        name: "OAuth".to_owned(),
        is_subscription: None,
        login_label: None,
        login: Arc::new(|_interaction| {
            Box::pin(async move {
                Ok(OAuthCredentials {
                    refresh: "r".to_owned(),
                    access: "a".to_owned(),
                    expires: now_ms() + 3_600_000,
                    extra: BTreeMap::new(),
                })
            })
        }),
        refresh: Arc::new(|credential, _signal| Box::pin(async move { Ok(credential) })),
        to_auth: Arc::new(|_credential| Box::pin(async move { Ok(ModelAuth::default()) })),
    }
}

/// An OAuth-only provider over the shared oauth fixture.
fn oauth_provider(id: &str) -> Arc<dyn pi_ai::models::Provider> {
    stream_provider(
        id,
        ProviderAuth {
            api_key: None,
            oauth: Some(oauth_auth()),
        },
    )
}

/// A Models collection over a scripted credential store, the seam the
/// resolve-walk tests seed per scenario.
fn models_with_credentials(store: ScriptedCredentialStore) -> pi_ai::models::Models {
    create_models(Some(pi_ai::models::CreateModelsOptions {
        credentials: Some(Arc::new(store)),
        ..Default::default()
    }))
}

fn stream_provider(id: &str, auth: ProviderAuth) -> Arc<dyn pi_ai::models::Provider> {
    Arc::new(create_provider(CreateProviderOptions {
        id: id.to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth,
        models: vec![fixture_model()],
        fetch_models: None,
        api: ProviderApi::Single(recording_streams()),
        filter_models: None,
    }))
}

fn interaction() -> AuthInteraction {
    AuthInteraction {
        signal: None,
        prompt: Arc::new(|_prompt| Box::pin(async move { Ok("x".to_owned()) })),
        notify: Arc::new(|_event: pi_ai::auth::types::AuthEvent| {}),
    }
}

/// Login and logout report pre-cancelled signals, run the OAuth flow, reject
/// unsupported login types, and surface store failures.
#[test]
fn the_login_and_logout_edges_report_aborts_and_failures() {
    use pi_ai::auth::resolve::ModelsErrorCode;

    let models = create_models(None);
    models.set_provider(oauth_provider("oauth"));

    // Pre-cancelled: the login and the logout reject before any work.
    let cancelled = tokio_util::sync::CancellationToken::new();
    cancelled.cancel();
    let outcome = block(models.login(
        "oauth",
        pi_ai::auth::types::AuthType::ApiKey,
        AuthInteraction {
            signal: Some(cancelled.clone()),
            ..interaction()
        },
    ));
    assert!(matches!(outcome, Err(ModelsFailure::Aborted(_))));
    let outcome = block(models.logout(
        "oauth",
        Some(&pi_ai::auth::types::AuthOptions {
            signal: Some(cancelled),
        }),
    ));
    assert!(matches!(outcome, Err(ModelsFailure::Aborted(_))));

    // The OAuth login runs the provider flow.
    let outcome = block(models.login("oauth", pi_ai::auth::types::AuthType::OAuth, interaction()));
    assert!(matches!(outcome, Ok(Credential::OAuth(_))));

    // An api-key login without a login closure is unsupported.
    models.set_provider(stream_provider("nologin", resolving_auth_with_headers("k")));
    let outcome = block(models.login(
        "nologin",
        pi_ai::auth::types::AuthType::ApiKey,
        interaction(),
    ));
    assert!(matches!(
        outcome,
        Err(ModelsFailure::Models(error)) if error.code() == ModelsErrorCode::Auth
    ));

    // A login that cancels its own signal mid-flow reports the abort: the
    // closure cancels the operation token and parks, so the abort branch of
    // the login race is the only one that can settle.
    models.set_provider(stream_provider(
        "aborting",
        ProviderAuth {
            api_key: None,
            oauth: Some(OAuthAuth {
                login: Arc::new(|interaction| {
                    Box::pin(async move {
                        interaction.signal.cancel();
                        loop {
                            std::future::pending::<()>().await;
                        }
                    })
                }),
                ..oauth_auth()
            }),
        },
    ));
    let outcome = block(models.login(
        "aborting",
        pi_ai::auth::types::AuthType::OAuth,
        interaction(),
    ));
    assert!(matches!(outcome, Err(ModelsFailure::Aborted(_))));

    // A failing credential store surfaces through logout.
    let failing = create_models(Some(pi_ai::models::CreateModelsOptions {
        credentials: Some(Arc::new(FailingCredentialStore)),
        ..Default::default()
    }));
    let outcome = block(failing.logout("p", None));
    assert!(matches!(
        outcome,
        Err(ModelsFailure::Models(error)) if error.code() == ModelsErrorCode::Auth
    ));
}

/// A credential store whose every operation fails.
struct FailingCredentialStore;

impl CredentialStore for FailingCredentialStore {
    fn read<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, pi_ai::auth::types::AuthError>> {
        Box::pin(async { Err(std::io::Error::other("store down").into()) })
    }

    fn list<'a>(
        &'a self,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<
        'a,
        Result<Vec<pi_ai::auth::types::CredentialInfo>, pi_ai::auth::types::AuthError>,
    > {
        Box::pin(async { Err(std::io::Error::other("store down").into()) })
    }

    fn modify<'a>(
        &'a self,
        _provider_id: &'a str,
        _f: pi_ai::auth::types::CredentialModifyFn,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, pi_ai::auth::types::AuthError>> {
        Box::pin(async { Err(std::io::Error::other("store down").into()) })
    }

    fn delete<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), pi_ai::auth::types::AuthError>> {
        Box::pin(async { Err(std::io::Error::other("store down").into()) })
    }
}

/// A provider whose model listing fails, and one whose availability check
/// errors; both hand-rolled over the `Provider` trait.
struct StaticStreamsProvider {
    id: &'static str,
    listing_fails: bool,
    auth: ProviderAuth,
}

impl StaticStreamsProvider {
    fn stream_for(model: &Model) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        end_with(model)
    }
}

impl pi_ai::models::Provider for StaticStreamsProvider {
    fn id(&self) -> &str {
        self.id
    }

    fn name(&self) -> &str {
        self.id
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn get_models(&self) -> Result<Vec<Model>, ProviderModelError> {
        if self.listing_fails {
            Err(std::io::Error::other("listing failed").into())
        } else {
            Ok(vec![fixture_model()])
        }
    }

    fn stream(
        &self,
        model: &Model,
        _context: &Context,
        _options: Option<&StreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        Self::stream_for(model)
    }

    fn stream_simple(
        &self,
        model: &Model,
        _context: &Context,
        _options: Option<&SimpleStreamOptions>,
    ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        Self::stream_for(model)
    }
}

/// `available()` skips providers whose listing fails, and a failing
/// availability `check` surfaces as an auth failure.
#[test]
fn the_available_walk_skips_failing_providers_and_surfaces_check_errors() {
    let broken = StaticStreamsProvider {
        id: "broken",
        listing_fails: true,
        auth: ambient_auth(),
    };
    let models = create_models(None);
    models.set_provider(Arc::new(broken));
    let models_listed = block(models.available(None, None));
    assert!(models_listed.expect("available").is_empty());

    let checked = StaticStreamsProvider {
        id: "checked",
        listing_fails: false,
        auth: ProviderAuth {
            api_key: Some(ApiKeyAuth {
                name: "Checked".to_owned(),
                login: None,
                check: Some(Arc::new(|_input: ApiKeyAuthInput| {
                    Box::pin(async move { Err(std::io::Error::other("check failed").into()) })
                })),
                resolve: Arc::new(|_input: ApiKeyAuthInput| {
                    Box::pin(async move { Ok(Some(pi_ai::auth::types::AuthResult::default())) })
                }),
            }),
            oauth: None,
        },
    };
    let models = create_models(None);
    models.set_provider(Arc::new(checked));
    let check = block(models.check_auth("checked", None));
    assert!(
        matches!(check, Err(ModelsFailure::Models(_))),
        "a failing check errors"
    );
}

/// The env-key registry: the anthropic auth-token exclusion, the vertex ADC
/// fallback, and the bedrock ambient-source walk.
#[test]
fn the_env_api_key_registry_edges() {
    use pi_ai::env_api_keys::{find_env_keys, get_env_api_key};
    use pi_ai::types::ProviderEnv;

    // The stored Anthropic auth token participates in discovery but never in
    // request auth.
    let auth_token_only =
        ProviderEnv::from([("ANTHROPIC_AUTH_TOKEN".to_owned(), "token".to_owned())]);
    assert!(find_env_keys("anthropic", Some(&auth_token_only)).is_some());
    assert_eq!(get_env_api_key("anthropic", Some(&auth_token_only)), None);

    let api_key_env = ProviderEnv::from([("ANTHROPIC_API_KEY".to_owned(), "sk-key".to_owned())]);
    assert_eq!(
        get_env_api_key("anthropic", Some(&api_key_env)),
        Some("sk-key".to_owned())
    );

    // Vertex: ADC credentials plus project and location resolve to the
    // sentinel.
    let adc_path = std::env::temp_dir().join(format!("pi-rust-adc-{}.json", std::process::id()));
    std::fs::write(&adc_path, "{}").expect("write the ADC fixture");
    let vertex = ProviderEnv::from([
        (
            "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
            adc_path.to_string_lossy().into_owned(),
        ),
        ("GOOGLE_CLOUD_PROJECT".to_owned(), "proj".to_owned()),
        ("GOOGLE_CLOUD_LOCATION".to_owned(), "us".to_owned()),
    ]);
    assert_eq!(
        get_env_api_key("google-vertex", Some(&vertex)),
        Some("<authenticated>".to_owned())
    );

    // Without any GAC source, the default path is probed. The pin holds only
    // when the machine has no ADC of its own: an ambient gcloud login makes
    // the default path hit.
    let default_adc = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .map(|home| home.join(".config/gcloud/application_default_credentials.json"));
    if default_adc.is_none_or(|path| !path.exists()) {
        assert_eq!(get_env_api_key("google-vertex", None), None);
    }

    // Bedrock's ambient walk: with no ambient source configured it misses,
    // and a named profile hits.
    let aws_sources = [
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ];
    if !aws_sources
        .iter()
        .any(|name| std::env::var_os(name).is_some())
    {
        assert_eq!(
            get_env_api_key("amazon-bedrock", Some(&ProviderEnv::default())),
            None
        );
    }
    let bedrock = ProviderEnv::from([("AWS_PROFILE".to_owned(), "default".to_owned())]);
    assert_eq!(
        get_env_api_key("amazon-bedrock", Some(&bedrock)),
        Some("<authenticated>".to_owned())
    );

    // Unknown providers have no env registry at all.
    assert!(find_env_keys("unknown-provider", None).is_none());
}

/// The provider-env override the resolve-walk tests pass, upstream's
/// `AuthResolutionOverrides.env` overlay.
fn env_overrides() -> AuthResolutionOverrides {
    AuthResolutionOverrides {
        env: Some(BTreeMap::from([("K".to_owned(), "v".to_owned())])),
        ..AuthResolutionOverrides::default()
    }
}

/// The images models options with both injected stores, the Debug shape the
/// suites render.
fn injected_images_options() -> pi_ai::images_models::CreateImagesModelsOptions {
    pi_ai::images_models::CreateImagesModelsOptions {
        credentials: Some(Arc::new(
            pi_ai::auth::credential_store::InMemoryCredentialStore::default(),
        )),
        auth_context: Some(pi_ai::auth::context::default_provider_auth_context()),
    }
}

/// The images surfaces: debug rendering, unknown-provider queries, and the
/// failure wrapping of provider generation failures with auth-derived fields.
#[expect(
    clippy::too_many_lines,
    reason = "each surface pins one behavior; splitting would duplicate the fixtures"
)]
#[test]
fn the_images_edges_render_debug_and_ghost_queries() {
    use pi_ai::images_models::{CreateImagesProviderOptions, create_images_provider};
    use pi_ai::types::ImagesOptions;
    use pi_ai::types::ImagesStopReason;

    let models = pi_ai::images_models::create_images_models(None);
    assert!(format!("{models:?}").contains("ImagesModelsState"));
    let injected = injected_images_options();
    assert!(format!("{injected:?}").contains("set"));

    assert!(models.models(Some("ghost")).is_empty());
    let ghost_refresh = block(models.refresh(Some("ghost")));
    assert!(ghost_refresh.is_ok());
    let ghost_auth = block(models.get_auth("ghost", None));
    assert!(ghost_auth.expect("ghost auth").is_none());
    let ghost_model_auth = block(models.get_auth_for_model(&image_model("ghost", "m"), None));
    assert!(ghost_model_auth.expect("ghost model auth").is_none());

    // A failing generate with unconfigured auth still dispatches, and the
    // provider failure wraps into an error result.
    let models = pi_ai::images_models::create_images_models(None);
    models.set_provider(Arc::new(create_images_provider(
        CreateImagesProviderOptions {
            id: "p".to_owned(),
            name: None,
            auth: ProviderAuth {
                api_key: Some(ApiKeyAuth {
                    name: "Key".to_owned(),
                    login: None,
                    check: None,
                    resolve: Arc::new(|_input: ApiKeyAuthInput| Box::pin(async move { Ok(None) })),
                }),
                oauth: None,
            },
            models: vec![image_model("p", "m")],
            api: Arc::new(FailingImagesApi),
            refresh_models: None,
        },
    )));
    let model = image_model("p", "m");
    let result = block(models.generate_images(&model, &images_context(), None));
    assert_eq!(result.stop_reason, ImagesStopReason::Error);
    let message = result.error_message.unwrap_or_default();
    assert!(
        message.contains("Image generation failed for p"),
        "got: {message}"
    );

    // Resolved auth applies its base_url and merges its headers with the
    // request's explicit headers.
    let calls: ImagesCalls = Arc::new(Mutex::new(Vec::new()));
    let calls_holder = Arc::clone(&calls);
    let models = pi_ai::images_models::create_images_models(None);
    models.set_provider(Arc::new(create_images_provider(
        CreateImagesProviderOptions {
            id: "authed".to_owned(),
            name: None,
            auth: ProviderAuth {
                api_key: Some(ApiKeyAuth {
                    name: "Key".to_owned(),
                    login: None,
                    check: None,
                    resolve: Arc::new(move |_input: ApiKeyAuthInput| {
                        Box::pin(async move {
                            Ok(Some(pi_ai::auth::types::AuthResult {
                                auth: ModelAuth {
                                    api_key: Some("resolved".to_owned()),
                                    headers: Some(BTreeMap::from([(
                                        "x-auth".to_owned(),
                                        Some("auth".to_owned()),
                                    )])),
                                    base_url: Some("https://auth.test/v1".to_owned()),
                                },
                                env: None,
                                source: None,
                            }))
                        })
                    }),
                }),
                oauth: None,
            },
            models: vec![image_model("authed", "m")],
            api: Arc::new(RecordingImagesApi(Arc::clone(&calls_holder))),
            refresh_models: None,
        },
    )));
    let model = image_model("authed", "m");
    let result = block(models.generate_images(
        &model,
        &images_context(),
        Some(&ImagesOptions {
            headers: Some(BTreeMap::from([(
                "x-request".to_owned(),
                Some("request".to_owned()),
            )])),
            ..ImagesOptions::default()
        }),
    ));
    assert_eq!(result.stop_reason, ImagesStopReason::Stop);
    let recorded = calls
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].0, "https://auth.test/v1");
    let headers = recorded[0].1.clone().expect("merged headers");
    assert_eq!(headers.get("x-auth"), Some(&Some("auth".to_owned())));
    assert_eq!(headers.get("x-request"), Some(&Some("request".to_owned())));

    // A generate failure with resolved auth wraps into an error result too.
    let models = pi_ai::images_models::create_images_models(None);
    models.set_provider(Arc::new(create_images_provider(
        CreateImagesProviderOptions {
            id: "failing".to_owned(),
            name: None,
            auth: ProviderAuth {
                api_key: Some(ApiKeyAuth {
                    name: "Key".to_owned(),
                    login: None,
                    check: None,
                    resolve: Arc::new(|_input: ApiKeyAuthInput| {
                        Box::pin(async move {
                            Ok(Some(pi_ai::auth::types::AuthResult {
                                auth: ModelAuth {
                                    api_key: Some("resolved".to_owned()),
                                    ..ModelAuth::default()
                                },
                                env: None,
                                source: None,
                            }))
                        })
                    }),
                }),
                oauth: None,
            },
            models: vec![image_model("failing", "m")],
            api: Arc::new(FailingImagesApi),
            refresh_models: None,
        },
    )));
    let model = image_model("failing", "m");
    let result = block(models.generate_images(&model, &images_context(), None));
    assert_eq!(result.stop_reason, ImagesStopReason::Error);
}

struct FailingImagesApi;

impl pi_ai::types::ProviderImages for FailingImagesApi {
    fn generate_images<'a>(
        &'a self,
        _model: &'a pi_ai::types::ImagesModel,
        _context: &'a pi_ai::types::ImagesContext,
        _options: Option<&'a pi_ai::types::ImagesOptions>,
    ) -> BoxedFuture<
        'a,
        Result<pi_ai::types::AssistantImages, pi_ai::utils::provider_retry::ProviderRequestError>,
    > {
        Box::pin(async move { Err(pi_ai::utils::provider_retry::ProviderRequestError::aborted()) })
    }
}

struct RecordingImagesApi(ImagesCalls);

impl pi_ai::types::ProviderImages for RecordingImagesApi {
    fn generate_images<'a>(
        &'a self,
        model: &'a pi_ai::types::ImagesModel,
        _context: &'a pi_ai::types::ImagesContext,
        options: Option<&'a pi_ai::types::ImagesOptions>,
    ) -> BoxedFuture<
        'a,
        Result<pi_ai::types::AssistantImages, pi_ai::utils::provider_retry::ProviderRequestError>,
    > {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((
                model.base_url.clone(),
                options.and_then(|options| options.headers.clone()),
            ));
        Box::pin(async move { Ok(ok_images_result(model)) })
    }
}

/// The standard refresh context: network allowed, nothing stored, a fresh
/// signal.
fn refresh_context() -> RefreshModelsContext {
    RefreshModelsContext {
        credential: None,
        stored: None,
        publish: no_publish(),
        allow_network: true,
        force: None,
        signal: tokio_util::sync::CancellationToken::new(),
    }
}

/// The builtin registry helpers expose the committed catalog ids.
#[test]
fn the_builtin_helpers_expose_the_catalog() {
    assert!(!builtin_models_of("anthropic").is_empty());
    assert!(builtin_models_of("unknown-provider").is_empty());
    assert_eq!(builtin_provider_id("x"), ProviderId::from("x"));
}

/// A credential store scripted per test: reads return the seeded credential,
/// and modify invokes the callback with a scripted current value.
struct ScriptedCredentialStore {
    stored: Option<Credential>,
    modify_current: Option<Credential>,
}

impl ScriptedCredentialStore {
    const fn with(stored: Option<Credential>, modify_current: Option<Credential>) -> Self {
        Self {
            stored,
            modify_current,
        }
    }
}

/// The api-key credential the stored-credential shapes carry.
fn api_key_credential() -> Credential {
    Credential::ApiKey(pi_ai::auth::types::ApiKeyCredential {
        key: Some("k".to_owned()),
        env: None,
    })
}

/// The OAuth credential the resolve walk reads: `expiring` seeds an expired
/// token so the resolution enters the locked modify path.
fn oauth_credential(expiring: bool) -> Credential {
    Credential::OAuth(OAuthCredentials {
        refresh: "r".to_owned(),
        access: "a".to_owned(),
        expires: now_ms() + if expiring { -1_000 } else { 3_600_000 },
        extra: BTreeMap::new(),
    })
}

impl CredentialStore for ScriptedCredentialStore {
    fn read<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, pi_ai::auth::types::AuthError>> {
        let stored = self.stored.clone();
        Box::pin(async move { Ok(stored) })
    }

    fn list<'a>(
        &'a self,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<
        'a,
        Result<Vec<pi_ai::auth::types::CredentialInfo>, pi_ai::auth::types::AuthError>,
    > {
        Box::pin(async move { Ok(Vec::new()) })
    }

    fn modify<'a>(
        &'a self,
        _provider_id: &'a str,
        f: pi_ai::auth::types::CredentialModifyFn,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, pi_ai::auth::types::AuthError>> {
        let current = self.modify_current.clone();
        Box::pin(async move { (f)(current).await })
    }

    fn delete<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), pi_ai::auth::types::AuthError>> {
        Box::pin(async { Err(std::io::Error::other("delete unsupported").into()) })
    }
}

/// The resolve walk: store-modify arms, OAuth derivation failures, and the
/// stored-credential type mismatches.
#[test]
fn the_resolve_walk_reports_store_and_derivation_failures() {
    use pi_ai::auth::resolve::ModelsErrorCode;

    // A stored OAuth credential expiring soon locks and modifies; the modify
    // closure observes `None` (logged out meanwhile) and resolves to no auth.
    let models = models_with_credentials(ScriptedCredentialStore::with(
        Some(oauth_credential(true)),
        None,
    ));
    models.set_provider(oauth_provider("oauth"));
    let resolved = block(models.get_auth("oauth", None));
    assert!(resolved.expect("resolve").is_none());

    // A modify that observes a non-OAuth credential resolves to no auth.
    let models = models_with_credentials(ScriptedCredentialStore::with(
        Some(oauth_credential(true)),
        Some(api_key_credential()),
    ));
    models.set_provider(oauth_provider("oauth"));
    let resolved = block(models.get_auth("oauth", None));
    assert!(resolved.expect("resolve").is_none());

    // A `to_auth` derivation failure surfaces as an OAuth error.
    let models = models_with_credentials(ScriptedCredentialStore::with(
        Some(oauth_credential(false)),
        None,
    ));
    models.set_provider(stream_provider(
        "oauth",
        ProviderAuth {
            api_key: None,
            oauth: Some(OAuthAuth {
                to_auth: Arc::new(|_credential| {
                    Box::pin(async move { Err(std::io::Error::other("derivation failed").into()) })
                }),
                ..oauth_auth()
            }),
        },
    ));
    let resolved = block(models.get_auth("oauth", None));
    assert!(
        matches!(&resolved, Err(ModelsFailure::Models(error)) if error.code() == ModelsErrorCode::OAuth),
        "got: {resolved:?}"
    );

    // A stored api-key credential on a provider whose api-key auth is absent
    // resolves to no auth, through the overlay auth context.
    let models = models_with_credentials(ScriptedCredentialStore::with(
        Some(api_key_credential()),
        None,
    ));
    models.set_provider(oauth_provider("oauth"));
    let resolved = block(models.get_auth("oauth", Some(&env_overrides())));
    assert!(resolved.expect("resolve").is_none());

    // The overlay context passes file probes through to the base context.
    let models = models_with_credentials(ScriptedCredentialStore::with(
        Some(api_key_credential()),
        None,
    ));
    models.set_provider(stream_provider("overlay", resolving_auth_with_headers("k")));
    let resolved = block(models.get_auth("overlay", Some(&env_overrides())));
    assert!(resolved.expect("resolve").is_some());
}

/// The env-key auth helper and the lazy OAuth wrapper report cancellation and
/// load once per implementation.
#[tokio::test]
async fn the_auth_helpers_report_cancellation_and_load_once() {
    let helper = env_api_key_auth("Test", &["TEST_ENV"]);
    let cancelled = tokio_util::sync::CancellationToken::new();
    cancelled.cancel();
    let aborted_interaction = ProviderAuthInteraction {
        signal: cancelled.clone(),
        prompt: Arc::new(|_prompt| Box::pin(async move { Ok("x".to_owned()) })),
        notify: Arc::new(|_event: pi_ai::auth::types::AuthEvent| {}),
    };
    let login = helper.login.expect("login closure");
    assert!(login(aborted_interaction.clone()).await.is_err());

    // A prompt that cancels mid-flow reports the abort after the prompt.
    let cancelling_prompt = ProviderAuthInteraction {
        signal: cancelled.clone(),
        prompt: {
            let signal = cancelled.clone();
            Arc::new(move |_prompt| {
                let signal = signal.clone();
                Box::pin(async move {
                    signal.cancel();
                    Ok("entered".to_owned())
                })
            })
        },
        notify: Arc::new(|_event: pi_ai::auth::types::AuthEvent| {}),
    };
    assert!(login(cancelling_prompt).await.is_err());

    // A resolve with a cancelled signal reports the abort.
    let input = ApiKeyAuthInput {
        ctx: pi_ai::auth::context::default_provider_auth_context(),
        credential: None,
        signal: cancelled.clone(),
    };
    assert!((helper.resolve)(input).await.is_err());

    // The lazy wrapper loads once and forwards login/refresh/to_auth.
    let calls = Arc::new(Mutex::new(0));
    let calls_holder = Arc::clone(&calls);
    let lazy = lazy_oauth(LazyOAuthInput {
        name: "Lazy".to_owned(),
        is_subscription: None,
        login_label: None,
        load: Arc::new(move || {
            let calls = Arc::clone(&calls_holder);
            Box::pin(async move {
                *calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
                oauth_auth()
            })
        }),
    });
    let fresh_interaction = ProviderAuthInteraction {
        signal: tokio_util::sync::CancellationToken::new(),
        prompt: Arc::new(|_prompt| Box::pin(async move { Ok("x".to_owned()) })),
        notify: Arc::new(|_event: pi_ai::auth::types::AuthEvent| {}),
    };
    let credential = (lazy.login)(fresh_interaction).await.expect("login");
    let credential = (lazy.refresh)(credential, tokio_util::sync::CancellationToken::new())
        .await
        .expect("refresh");
    let _auth = (lazy.to_auth)(credential).await.expect("to_auth");
    assert_eq!(
        *calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        1
    );
}

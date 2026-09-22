//! The `ImagesModels` runtime suite, ported from
//! `packages/ai/test/images-models.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use common::{block, image_model, images_context as context, ok_images_result};
use pi_ai::auth::types::{AuthContext, ModelAuth, ProviderAuth};
use pi_ai::images_models::{
    CreateImagesModelsOptions, CreateImagesProviderOptions, ImagesModels, ImagesProvider,
    create_images_models, create_images_provider,
};

mod common;
use pi_ai::types::{ImagesContext, ImagesModel, ImagesOptions};

fn fake_auth_context(env: BTreeMap<String, String>) -> Arc<dyn AuthContext> {
    Arc::new(FixtureAuthContext(env))
}

struct FixtureAuthContext(BTreeMap<String, String>);

impl AuthContext for FixtureAuthContext {
    fn env(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }

    fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

fn default_images_models() -> ImagesModels {
    create_images_models(None)
}

fn images_models_with(auth_context: Arc<dyn AuthContext>) -> ImagesModels {
    create_images_models(Some(CreateImagesModelsOptions {
        auth_context: Some(auth_context),
        ..CreateImagesModelsOptions::default()
    }))
}

#[derive(Clone, Debug)]
struct GenerateCall {
    options: Option<ImagesOptions>,
}

type GenerateCalls = Arc<Mutex<Vec<GenerateCall>>>;

struct RecordingImagesApi(Option<GenerateCalls>);

impl pi_ai::types::ProviderImages for RecordingImagesApi {
    fn generate_images<'a>(
        &'a self,
        model: &'a ImagesModel,
        _context: &'a ImagesContext,
        options: Option<&'a ImagesOptions>,
    ) -> pi_ai::types::BoxedFuture<
        'a,
        Result<pi_ai::types::AssistantImages, pi_ai::utils::provider_retry::ProviderRequestError>,
    > {
        if let Some(calls) = &self.0 {
            calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(GenerateCall {
                    options: options.cloned(),
                });
        }
        Box::pin(async move { Ok(ok_images_result(model)) })
    }
}

fn ok_images_api(calls: Option<GenerateCalls>) -> Arc<dyn pi_ai::types::ProviderImages> {
    Arc::new(RecordingImagesApi(calls))
}

fn resolve_auth_fixture(env_var: Option<&str>) -> pi_ai::auth::types::ApiKeyResolveFn {
    let env_var = env_var.map(ToOwned::to_owned);
    Arc::new(move |input: pi_ai::auth::types::ApiKeyAuthInput| {
        let env_var = env_var.clone();
        Box::pin(async move {
            let Some(env_var) = env_var else {
                return Ok(Some(pi_ai::auth::types::AuthResult::default()));
            };
            let key = input
                .credential
                .as_ref()
                .and_then(|credential| credential.key.clone())
                .or_else(|| input.ctx.env(&env_var));
            let Some(key) = key else {
                return Ok(None);
            };
            Ok(Some(pi_ai::auth::types::AuthResult {
                auth: ModelAuth {
                    api_key: Some(key),
                    ..ModelAuth::default()
                },
                env: None,
                source: Some(if input.credential.is_some() {
                    "stored".to_owned()
                } else {
                    env_var
                }),
            }))
        })
    })
}

fn test_provider(
    id: &str,
    models: Option<Vec<ImagesModel>>,
    env_var: Option<&str>,
    calls: Option<GenerateCalls>,
) -> Arc<dyn ImagesProvider> {
    Arc::new(create_images_provider(CreateImagesProviderOptions {
        id: id.to_owned(),
        name: None,
        auth: ProviderAuth {
            api_key: Some(pi_ai::auth::types::ApiKeyAuth {
                name: "Test key".to_owned(),
                login: None,
                check: None,
                resolve: resolve_auth_fixture(env_var),
            }),
            oauth: None,
        },
        models: models.unwrap_or_else(|| vec![image_model(id, "model-a")]),
        api: ok_images_api(calls),
        refresh_models: None,
    }))
}

#[test]
fn registers_providers_and_reads_models_synchronously() {
    let models = default_images_models();
    models.set_provider(test_provider(
        "p1",
        Some(vec![image_model("p1", "m1"), image_model("p1", "m2")]),
        None,
        None,
    ));
    models.set_provider(test_provider(
        "p2",
        Some(vec![image_model("p2", "m3")]),
        None,
        None,
    ));

    assert_eq!(
        models
            .providers()
            .iter()
            .map(|provider| provider.id().to_owned())
            .collect::<Vec<_>>(),
        ["p1", "p2"]
    );
    assert_eq!(
        models
            .models(None)
            .iter()
            .map(|model| model.id.clone())
            .collect::<Vec<_>>(),
        ["m1", "m2", "m3"]
    );
    assert_eq!(
        models
            .models(Some("p1"))
            .iter()
            .map(|model| model.id.clone())
            .collect::<Vec<_>>(),
        ["m1", "m2"]
    );
    assert_eq!(
        models.model("p2", "m3").map(|model| model.id),
        Some("m3".to_owned())
    );
    assert!(models.model("p2", "missing").is_none());

    models.delete_provider("p1");
    assert!(models.provider("p1").is_none());
}

#[test]
fn resolves_auth_through_the_provider_and_merges_it_into_requests() {
    let calls: GenerateCalls = Arc::new(Mutex::new(Vec::new()));
    let models = images_models_with(fake_auth_context(BTreeMap::from([(
        "TEST_KEY".to_owned(),
        "env-key".to_owned(),
    )])));
    models.set_provider(test_provider(
        "p1",
        None,
        Some("TEST_KEY"),
        Some(Arc::clone(&calls)),
    ));
    let model = models.model("p1", "model-a").expect("model-a present");

    let auth = block(models.get_auth_for_model(&model, None));
    assert_eq!(
        auth.expect("auth").and_then(|result| result.auth.api_key),
        Some("env-key".to_owned())
    );
    let auth = block(models.get_auth("p1", None));
    assert_eq!(
        auth.expect("auth").and_then(|result| result.auth.api_key),
        Some("env-key".to_owned())
    );
    let auth = block(models.get_auth_for_model(
        &model,
        Some(&pi_ai::auth::resolve::AuthResolutionOverrides {
            api_key: Some("explicit-key".to_owned()),
            ..pi_ai::auth::resolve::AuthResolutionOverrides::default()
        }),
    ));
    assert_eq!(
        auth.expect("auth").and_then(|result| result.auth.api_key),
        Some("explicit-key".to_owned())
    );

    let result = block(models.generate_images(&model, &context(), None));
    assert_eq!(result.stop_reason, pi_ai::types::ImagesStopReason::Stop);
    let recorded = calls
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        recorded[0]
            .options
            .as_ref()
            .and_then(|options| options.api_key.clone()),
        Some("env-key".to_owned())
    );

    let explicit = ImagesOptions {
        api_key: Some("explicit".to_owned()),
        ..ImagesOptions::default()
    };
    let result = block(models.generate_images(&model, &context(), Some(&explicit)));
    assert_eq!(result.stop_reason, pi_ai::types::ImagesStopReason::Stop);
    let recorded = calls
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        recorded[1]
            .options
            .as_ref()
            .and_then(|options| options.api_key.clone()),
        Some("explicit".to_owned())
    );
}

#[test]
fn merges_provider_resolved_env_into_image_options() {
    let calls: GenerateCalls = Arc::new(Mutex::new(Vec::new()));
    let models = default_images_models();
    models.set_provider(Arc::new(create_images_provider(
        CreateImagesProviderOptions {
            id: "p1".to_owned(),
            name: None,
            auth: ProviderAuth {
                api_key: Some(pi_ai::auth::types::ApiKeyAuth {
                    name: "Test key".to_owned(),
                    login: None,
                    check: None,
                    resolve: Arc::new(|_input| {
                        Box::pin(async move {
                            Ok(Some(pi_ai::auth::types::AuthResult {
                                auth: ModelAuth {
                                    api_key: Some("provider-key".to_owned()),
                                    ..ModelAuth::default()
                                },
                                env: Some(BTreeMap::from([
                                    ("PROVIDER_ONLY".to_owned(), "provider".to_owned()),
                                    ("SHARED".to_owned(), "provider".to_owned()),
                                ])),
                                source: None,
                            }))
                        })
                    }),
                }),
                oauth: None,
            },
            models: vec![image_model("p1", "model-a")],
            api: ok_images_api(Some(Arc::clone(&calls))),
            refresh_models: None,
        },
    )));
    let model = models.model("p1", "model-a").expect("model-a present");

    let request_options = ImagesOptions {
        api_key: Some("request-key".to_owned()),
        env: Some(BTreeMap::from([
            ("REQUEST_ONLY".to_owned(), "request".to_owned()),
            ("SHARED".to_owned(), "request".to_owned()),
        ])),
        ..ImagesOptions::default()
    };
    block(models.generate_images(&model, &context(), Some(&request_options)));

    let recorded = calls
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        recorded[0]
            .options
            .as_ref()
            .and_then(|options| options.api_key.clone()),
        Some("request-key".to_owned())
    );
    assert_eq!(
        recorded[0]
            .options
            .as_ref()
            .and_then(|options| options.env.clone()),
        Some(BTreeMap::from([
            ("PROVIDER_ONLY".to_owned(), "provider".to_owned()),
            ("REQUEST_ONLY".to_owned(), "request".to_owned()),
            ("SHARED".to_owned(), "request".to_owned()),
        ]))
    );
}

#[test]
fn returns_an_error_result_for_unknown_providers_and_unconfigured_auth() {
    let models = images_models_with(fake_auth_context(BTreeMap::new()));
    let ghost = block(models.generate_images(&image_model("ghost", "m"), &context(), None));
    assert_eq!(ghost.stop_reason, pi_ai::types::ImagesStopReason::Error);
    let message = ghost.error_message.unwrap_or_default();
    assert!(
        message.contains("Unknown provider: ghost"),
        "got: {message}"
    );

    // unconfigured (resolve -> undefined) still dispatches; provider decides
    let calls: GenerateCalls = Arc::new(Mutex::new(Vec::new()));
    models.set_provider(test_provider(
        "p1",
        None,
        Some("MISSING"),
        Some(Arc::clone(&calls)),
    ));
    let model = models.model("p1", "model-a").expect("model-a present");
    assert!(
        block(models.get_auth_for_model(&model, None))
            .expect("auth")
            .is_none()
    );
    block(models.generate_images(&model, &context(), None));
    let recorded = calls
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        recorded[0]
            .options
            .as_ref()
            .and_then(|options| options.api_key.clone()),
        None
    );
}

#[test]
fn supports_dynamic_providers_via_refresh_with_in_flight_dedupe() {
    let fetches = Arc::new(Mutex::new(0));
    let fetch_count = Arc::clone(&fetches);
    let provider = Arc::new(create_images_provider(CreateImagesProviderOptions {
        id: "dyn".to_owned(),
        name: None,
        auth: ProviderAuth {
            api_key: Some(pi_ai::auth::types::ApiKeyAuth {
                name: "Test".to_owned(),
                login: None,
                check: None,
                resolve: resolve_auth_fixture(None),
            }),
            oauth: None,
        },
        models: Vec::new(),
        refresh_models: Some(Arc::new(move || {
            let count = Arc::clone(&fetch_count);
            Box::pin(async move {
                // Mirror upstream's 5ms pause: the second caller must join
                // the same in-flight fetch, not start another.
                *count
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                Ok(vec![image_model("dyn", "listed")])
            })
        })),
        api: ok_images_api(None),
    }));
    let models = default_images_models();
    models.set_provider(provider);

    assert!(models.models(Some("dyn")).is_empty());
    block(async {
        let refresh_one = models.refresh(Some("dyn"));
        let refresh_two = models.refresh(Some("dyn"));
        let (one, two) = tokio::join!(refresh_one, refresh_two);
        assert_eq!(one.expect("first refresh"), ());
        assert_eq!(two.expect("second refresh"), ());
    });
    assert_eq!(
        *fetches
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        1
    );
    assert!(models.model("dyn", "listed").is_some());

    // failures reject with ModelsError for a single provider
    models.set_provider(Arc::new(create_images_provider(
        CreateImagesProviderOptions {
            id: "flaky".to_owned(),
            name: None,
            auth: ProviderAuth {
                api_key: Some(pi_ai::auth::types::ApiKeyAuth {
                    name: "Test".to_owned(),
                    login: None,
                    check: None,
                    resolve: resolve_auth_fixture(None),
                }),
                oauth: None,
            },
            models: Vec::new(),
            refresh_models: Some(Arc::new(|| {
                Box::pin(async move { Err(std::io::Error::other("fetch failed").into()) })
            })),
            api: ok_images_api(None),
        },
    )));
    let result = block(models.refresh(Some("flaky")));
    assert!(matches!(
        result,
        Err(error) if error.code() == pi_ai::auth::resolve::ModelsErrorCode::ModelSource
    ));
    // the all-providers refresh stays best-effort
    block(models.refresh(None)).expect("all-provider refresh");
}

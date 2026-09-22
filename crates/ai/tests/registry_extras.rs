//! Coverage-belt extras for the #28 surface: the env-key table, the default
//! auth context, the models-store signal paths, the Models collection's
//! deferred/login/logout edges, the images runtime's dedupe and error paths,
//! the catalog registry's parse branches, and the model-data validators'
//! rejection reports. These pin behaviors upstream's suites cover through the
//! compat layer and the CLI, at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use common::{block, fixture_model};

mod common;

use pi_ai::auth::context::DefaultAuthContext;
use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::types::{AuthContext, Credential, ModelAuth, OAuthCredentials, ProviderAuth};
use pi_ai::env_api_keys::{ANTHROPIC_AUTH_TOKEN_ENV, find_env_keys, get_env_api_key};
use pi_ai::models::{ModelsDeferredCancelOptions, calculate_cost};
use pi_ai::models_store::{ModelsStore, ModelsStoreEntry};
use pi_ai::types::{
    Context, DeferredHandle, Model, ModelThinkingLevel, SimpleStreamOptions, StopReason,
    StreamOptions, Usage,
};

/// The env-key table maps every provider id to exactly the vars upstream
/// documents, and the anthropic ordering keeps `AUTH_TOKEN` out of the key.
#[test]
fn the_env_key_table_maps_every_documented_provider() {
    let table: &[(&str, &[&str])] = &[
        ("github-copilot", &["COPILOT_GITHUB_TOKEN"]),
        (
            "anthropic",
            &[
                "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_OAUTH_TOKEN",
                "ANTHROPIC_API_KEY",
            ],
        ),
        ("ant-ling", &["ANT_LING_API_KEY"]),
        ("qwen-token-plan", &["QWEN_TOKEN_PLAN_API_KEY"]),
        ("qwen-token-plan-cn", &["QWEN_TOKEN_PLAN_CN_API_KEY"]),
        ("qwen-token-plan-individual", &["QWEN_TOKEN_PLAN_API_KEY"]),
        ("openai", &["OPENAI_API_KEY"]),
        ("azure-openai-responses", &["AZURE_OPENAI_API_KEY"]),
        ("nvidia", &["NVIDIA_API_KEY"]),
        ("deepseek", &["DEEPSEEK_API_KEY"]),
        ("google", &["GEMINI_API_KEY"]),
        ("google-vertex", &["GOOGLE_CLOUD_API_KEY"]),
        ("groq", &["GROQ_API_KEY"]),
        ("cerebras", &["CEREBRAS_API_KEY"]),
        ("xai", &["XAI_API_KEY"]),
        ("radius", &["RADIUS_API_KEY"]),
        ("openrouter", &["OPENROUTER_API_KEY"]),
        ("vercel-ai-gateway", &["AI_GATEWAY_API_KEY"]),
        ("zai", &["ZAI_API_KEY"]),
        ("zai-coding-cn", &["ZAI_CODING_CN_API_KEY"]),
        ("mistral", &["MISTRAL_API_KEY"]),
        ("minimax", &["MINIMAX_API_KEY"]),
        ("minimax-cn", &["MINIMAX_CN_API_KEY"]),
        ("moonshotai", &["MOONSHOT_API_KEY"]),
        ("moonshotai-cn", &["MOONSHOT_API_KEY"]),
        ("huggingface", &["HF_TOKEN"]),
        ("fireworks", &["FIREWORKS_API_KEY"]),
        ("together", &["TOGETHER_API_KEY"]),
        ("baseten", &["BASETEN_API_KEY"]),
        ("opencode", &["OPENCODE_API_KEY"]),
        ("opencode-go", &["OPENCODE_API_KEY"]),
        ("kimi-coding", &["KIMI_API_KEY"]),
        ("cloudflare-workers-ai", &["CLOUDFLARE_API_KEY"]),
        ("cloudflare-ai-gateway", &["CLOUDFLARE_API_KEY"]),
        ("xiaomi", &["XIAOMI_API_KEY"]),
        ("xiaomi-token-plan-cn", &["XIAOMI_TOKEN_PLAN_CN_API_KEY"]),
        ("xiaomi-token-plan-ams", &["XIAOMI_TOKEN_PLAN_AMS_API_KEY"]),
        ("xiaomi-token-plan-sgp", &["XIAOMI_TOKEN_PLAN_SGP_API_KEY"]),
    ];
    for (provider, vars) in table {
        let env: pi_ai::types::ProviderEnv = vars
            .iter()
            .map(|var| ((*var).to_owned(), "value".to_owned()))
            .collect();
        let found = find_env_keys(provider, Some(&env)).expect("found");
        assert_eq!(
            found.iter().map(String::as_str).collect::<Vec<_>>(),
            *vars,
            "{provider} maps to its documented vars"
        );
        assert_eq!(
            get_env_api_key(provider, Some(&env)),
            Some("value".to_owned())
        );
    }
    // Unknown providers have no table entry.
    assert!(find_env_keys("not-a-provider", None).is_none());
}

#[test]
fn the_anthropic_key_skips_the_auth_token_env() {
    let env: pi_ai::types::ProviderEnv = BTreeMap::from([
        (ANTHROPIC_AUTH_TOKEN_ENV.to_owned(), "auth".to_owned()),
        ("ANTHROPIC_API_KEY".to_owned(), "api".to_owned()),
    ]);
    // find_env_keys reports the two variables the fixture sets; the stored
    // AUTH_TOKEN participates in discovery, and get_env_api_key skips it.
    let found = find_env_keys("anthropic", Some(&env)).expect("found");
    assert_eq!(found.len(), 2, "AUTH_TOKEN is set but OAUTH_TOKEN is not");
    assert_eq!(
        get_env_api_key("anthropic", Some(&env)),
        Some("api".to_owned())
    );
}

/// Vertex resolves through ADC when project, location, and credentials file
/// are all present; the explicit `GOOGLE_APPLICATION_CREDENTIALS` path wins.
#[test]
fn the_vertex_adc_fallback_needs_credentials_project_and_location() {
    let existing = std::env::temp_dir().join("pi-ai-vertex-adc-probe");
    std::fs::write(&existing, b"{}").expect("fixture file");

    let env: pi_ai::types::ProviderEnv = BTreeMap::from([
        ("GOOGLE_CLOUD_PROJECT".to_owned(), "p".to_owned()),
        ("GOOGLE_CLOUD_LOCATION".to_owned(), "l".to_owned()),
        (
            "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
            existing.display().to_string(),
        ),
    ]);
    assert_eq!(
        get_env_api_key("google-vertex", Some(&env)),
        Some("<authenticated>".to_owned())
    );
}

#[test]
fn the_bedrock_sources_report_authenticated() {
    let sources: Vec<pi_ai::types::ProviderEnv> = vec![
        BTreeMap::from([("AWS_PROFILE".to_owned(), "p".to_owned())]),
        BTreeMap::from([
            ("AWS_ACCESS_KEY_ID".to_owned(), "k".to_owned()),
            ("AWS_SECRET_ACCESS_KEY".to_owned(), "s".to_owned()),
        ]),
        BTreeMap::from([("AWS_BEARER_TOKEN_BEDROCK".to_owned(), "t".to_owned())]),
        BTreeMap::from([(
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI".to_owned(),
            "u".to_owned(),
        )]),
        BTreeMap::from([(
            "AWS_CONTAINER_CREDENTIALS_FULL_URI".to_owned(),
            "u".to_owned(),
        )]),
        BTreeMap::from([("AWS_WEB_IDENTITY_TOKEN_FILE".to_owned(), "f".to_owned())]),
    ];
    for env in &sources {
        assert_eq!(
            get_env_api_key("amazon-bedrock", Some(env)),
            Some("<authenticated>".to_owned())
        );
    }
    assert_eq!(
        get_env_api_key("amazon-bedrock", Some(&BTreeMap::new())),
        None
    );
}

#[test]
fn the_default_auth_context_reads_env_blank_and_files() {
    let context = DefaultAuthContext;
    // The process env read: an existing var resolves, blank reads unset.
    assert!(context.env("PATH").is_some());
    assert_eq!(context.env("PI_AI_NO_SUCH_VAR_XYZ"), None);
    // fileExists: the crate manifest exists, a missing file does not.
    assert!(context.file_exists(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml")));
    assert!(!context.file_exists(concat!(env!("CARGO_MANIFEST_DIR"), "/no-such-file")));
    let _ = context;
}

/// The Models collection's stream surface settles through the provider, and
/// the deferred paths report unsupported providers inside the stream.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the stream/deferred walk mirrors the Models surface shape"
)]
fn the_models_stream_surface_routes_through_the_provider() {
    use pi_ai::models::{CreateProviderOptions, create_models};

    fn respond(model: &Model) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
        use pi_ai::utils::event_stream::assistant_message_event_stream;
        let stream = assistant_message_event_stream();
        let message = pi_ai::types::AssistantMessage {
            content: Vec::new(),
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        };
        stream.push(pi_ai::types::AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: message.clone(),
        });
        stream.end(Some(&message));
        stream
    }
    struct RespondingStreams;
    impl pi_ai::types::ProviderStreams for RespondingStreams {
        fn stream(
            &self,
            model: &Model,
            _context: &Context,
            _options: Option<&StreamOptions>,
        ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
            respond(model)
        }

        fn stream_simple(
            &self,
            model: &Model,
            _context: &Context,
            _options: Option<&SimpleStreamOptions>,
        ) -> pi_ai::utils::event_stream::AssistantMessageEventStream {
            self.stream(model, &Context::default(), None)
        }
    }
    let streams: Arc<dyn pi_ai::types::ProviderStreams> = Arc::new(RespondingStreams);
    let provider = pi_ai::models::create_provider(CreateProviderOptions {
        id: "p".to_owned(),
        name: None,
        base_url: None,
        headers: None,
        auth: ProviderAuth {
            api_key: Some(pi_ai::auth::types::ApiKeyAuth {
                name: "Ambient".to_owned(),
                login: None,
                check: None,
                resolve: Arc::new(|_input| {
                    Box::pin(async move { Ok(Some(pi_ai::auth::types::AuthResult::default())) })
                }),
            }),
            oauth: None,
        },
        models: vec![],
        api: pi_ai::models::ProviderApi::Single(streams),
        fetch_models: None,
        filter_models: None,
    });
    let models = create_models(None);
    models.set_provider(Arc::new(provider));

    let model = fixture_model();
    let handle = DeferredHandle {
        provider: "p".to_owned(),
        model_id: "m".to_owned(),
        api: "test-api".to_owned(),
        id: "id".to_owned(),
        expires_at: None,
        poll_after_ms: None,
        data: None,
    };
    block(async move {
        let done = models
            .stream(&model, &Context::default(), None)
            .result()
            .await;
        assert_eq!(done.stop_reason, StopReason::Stop);
        let simple = models
            .stream_simple(&model, &Context::default(), None)
            .result()
            .await;
        assert_eq!(simple.stop_reason, StopReason::Stop);

        // Deferred: the provider's streams report no deferred support, so the
        // lazy stream settles with the not-supported notice.
        let deferred = models.stream_deferred(&model, &handle, None).result().await;
        let text = deferred.error_message.unwrap_or_default();
        assert!(
            text.contains("does not support deferred responses"),
            "got: {text}"
        );

        let cancel = models
            .cancel_deferred(
                &model,
                &handle,
                Some(&ModelsDeferredCancelOptions::default()),
            )
            .await;
        assert!(cancel.is_err());
    });
}

/// `modelsAreEqual`, `hasApi`, the thinking-level clamps, and the usage cost
/// tiers exercise their edges.
#[test]
fn the_model_helpers_pin_their_edges() {
    use pi_ai::models::{clamp_thinking_level, models_are_equal};
    let mut model = fixture_model();
    model.reasoning = true;
    model.thinking_level_map = Some(BTreeMap::from([
        (ModelThinkingLevel::Off, Some("none".to_owned())),
        (ModelThinkingLevel::Low, Some("low".to_owned())),
        (ModelThinkingLevel::Max, Some("max".to_owned())),
    ]));

    // Absent map keys use provider defaults, so only explicit nulls remove a
    // level; xhigh and max require an explicit mapping.
    assert_eq!(
        pi_ai::models::get_supported_thinking_levels(&model),
        vec![
            ModelThinkingLevel::Off,
            ModelThinkingLevel::Minimal,
            ModelThinkingLevel::Low,
            ModelThinkingLevel::Medium,
            ModelThinkingLevel::High,
            ModelThinkingLevel::Max,
        ]
    );
    // A null marks the level unsupported.
    let mut with_null = fixture_model();
    with_null.reasoning = true;
    with_null.thinking_level_map = Some(BTreeMap::from([
        (ModelThinkingLevel::Low, None),
        (ModelThinkingLevel::Max, Some("max".to_owned())),
    ]));
    assert_eq!(
        pi_ai::models::get_supported_thinking_levels(&with_null),
        vec![
            ModelThinkingLevel::Off,
            ModelThinkingLevel::Minimal,
            ModelThinkingLevel::Medium,
            ModelThinkingLevel::High,
            ModelThinkingLevel::Max,
        ]
    );

    // Supported requests clamp to themselves; absent keys use provider
    // defaults, so Minimal is available.
    assert_eq!(
        clamp_thinking_level(&model, ModelThinkingLevel::Low),
        ModelThinkingLevel::Low
    );
    assert_eq!(
        clamp_thinking_level(&model, ModelThinkingLevel::Minimal),
        ModelThinkingLevel::Minimal
    );
    assert_eq!(
        clamp_thinking_level(&model, ModelThinkingLevel::High),
        ModelThinkingLevel::High
    );
    // xhigh is unmapped; the nearest supported higher is max.
    assert_eq!(
        clamp_thinking_level(&model, ModelThinkingLevel::Xhigh),
        ModelThinkingLevel::Max
    );

    // A non-reasoning model supports only off.
    let mut plain = fixture_model();
    plain.reasoning = false;
    assert_eq!(
        clamp_thinking_level(&plain, ModelThinkingLevel::High),
        ModelThinkingLevel::Off
    );

    // modelsAreEqual compares id and provider.
    let mut other = fixture_model();
    assert!(models_are_equal(Some(&model), Some(&other)));
    other.id = "n".to_owned();
    assert!(!models_are_equal(Some(&model), Some(&other)));
    assert!(!models_are_equal(Some(&model), None));
}

#[test]
fn the_usage_cost_applies_the_tier_above_the_threshold() {
    let mut model = fixture_model();
    model.cost.rates = pi_ai::types::ModelCostRates {
        input: 1.0,
        output: 2.0,
        cache_read: 0.0,
        cache_write: 0.0,
    };
    model.cost.tiers = Some(vec![pi_ai::types::ModelCostTier {
        rates: pi_ai::types::ModelCostRates {
            input: 2.0,
            output: 4.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
        input_tokens_above: 1_000,
    }]);
    let mut usage = Usage {
        input: 2_000,
        output: 100,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: Some(10),
        reasoning: None,
        total_tokens: 2_110,
        cost: pi_ai::types::UsageCost::default(),
    };
    let cost = calculate_cost(&model, &mut usage);
    // The tier applies to the whole request: input 2/1e6 x 2000 tokens; the
    // 1h cache write charges 2x base input over its 10 tokens.
    assert!((cost.input - 0.004).abs() < 1e-12);
    assert!((cost.output - 0.0004).abs() < 1e-12);
    assert!((cost.cache_write - 0.00004).abs() < 1e-10);
    // The computed cost is stored on the usage.
    assert!((usage.cost.total - cost.total).abs() < 1e-12);
}

/// The models store's write/delete serialize and the entries round-trip.
#[tokio::test]
async fn the_in_memory_models_store_round_trips_entries() {
    let store = pi_ai::models_store::InMemoryModelsStore::default();
    let entry = ModelsStoreEntry {
        models: vec![fixture_model()],
        last_modified: Some(1),
        checked_at: Some(2),
        etag: Some("\"e\"".to_owned()),
    };
    store.write("p", entry.clone(), None).await.expect("write");
    let read = store.read("p", None).await.expect("read").expect("entry");
    assert_eq!(read, entry);
    // A second write replaces.
    let mut second = entry.clone();
    second.checked_at = Some(1);
    store.write("p", second, None).await.expect("rewrite");
    assert_eq!(
        store
            .read("p", None)
            .await
            .expect("read")
            .expect("entry")
            .checked_at,
        Some(1)
    );
    store.delete("p", None).await.expect("delete");
    assert!(store.read("p", None).await.expect("read").is_none());
}

/// The images collection: clear/delete providers, unknown-provider errors,
/// and the builtin images providers list.
#[tokio::test]
async fn the_images_runtime_covers_the_collection_edges() {
    let models = pi_ai::images_models::create_images_models(None);
    models.set_provider(pi_ai::providers::all::builtin_images_providers().remove(0));
    assert!(!models.models(Some("openrouter")).is_empty());
    models.delete_provider("openrouter");
    assert!(models.provider("openrouter").is_none());
    models.set_provider(Arc::new(
        pi_ai::providers::openrouter_images::openrouter_images_provider(),
    ));
    models.clear_providers();
    assert!(models.providers().is_empty());

    // Unknown provider: an error result with the provider notice.
    let model = fixture_image_model();
    let result = models.generate_images(&model, &pi_ai::types::ImagesContext::default(), None);
    let images = result.await;
    assert_eq!(images.stop_reason, pi_ai::types::ImagesStopReason::Error);
    assert!(
        images
            .error_message
            .unwrap_or_default()
            .contains("Unknown provider")
    );
}

fn fixture_image_model() -> pi_ai::types::ImagesModel {
    pi_ai::types::ImagesModel {
        id: "img".to_owned(),
        name: "img".to_owned(),
        api: pi_ai::types::ImagesApi::from("test-images"),
        provider: pi_ai::types::ImagesProviderId::from("ghost"),
        base_url: "https://example.test".to_owned(),
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        output: vec![pi_ai::types::Modality::Image],
        cost: pi_ai::types::ModelCost::default(),
        sampling_params: None,
        headers: None,
    }
}

/// The image parsing edges: dedup of modalities, non-image outputs skipped,
/// default text input, and per-million price scaling.
#[test]
fn openrouter_image_parsing_covers_the_modality_and_pricing_edges() {
    use pi_ai::image_models::parse_openrouter_image_models;
    let payload = serde_json::json!({
        "data": [
            {
                "id": "a/one",
                "name": "One",
                "architecture": {
                    "input_modalities": ["text", "text", "image"],
                    "output_modalities": ["image", "image"],
                },
                "pricing": { "prompt": "0.000002", "completion": "0.000004" },
            },
            { "id": "a/two", "name": "Two", "architecture": { "output_modalities": ["text"] } },
            { "id": "a/three", "name": "Three", "architecture": { "output_modalities": ["image"] } },
        ],
    });
    let models = parse_openrouter_image_models(&payload, true).expect("parses");
    assert_eq!(models.len(), 2);
    let model = &models[0];
    assert_eq!(model.id, "a/one");
    // Duplicate modalities dedup.
    assert_eq!(
        model.input,
        vec![pi_ai::types::Modality::Text, pi_ai::types::Modality::Image]
    );
    // A missing input list defaults to text.
    assert_eq!(models[1].id, "a/three");
    assert_eq!(models[1].input, vec![pi_ai::types::Modality::Text]);
    // The per-token prices scale to per-million rates.
    assert!((model.cost.rates.input - 2.0).abs() < 1e-12);
    assert!((model.cost.rates.output - 4.0).abs() < 1e-12);
    assert!((model.cost.rates.cache_read - 0.0).abs() < 1e-12);
    // Non-strict mode with an empty list yields nothing.
    assert!(
        parse_openrouter_image_models(&serde_json::json!({ "data": [] }), false)
            .expect("non-strict tolerates empty")
            .is_empty()
    );
}

/// The model-data validator's per-model rejection branches.
#[test]
fn the_model_value_validator_reports_every_field_shape() {
    let dir = std::env::temp_dir().join(format!(
        "pi-ai-value-validator-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let structure: pi_ai::model_data::ModelDataStructure = BTreeMap::from([(
        "prov".to_owned(),
        BTreeMap::from([("m".to_owned(), "api".to_owned())]),
    )]);
    let mut manifest_files = BTreeMap::new();

    let mut build_value = |mutate: &dyn Fn(&mut serde_json::Value)| -> String {
        let mut model = serde_json::json!({
            "id": "m",
            "name": "M",
            "api": "api",
            "provider": "prov",
            "baseUrl": "https://x.test",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 10,
            "maxTokens": 10,
        });
        mutate(&mut model);
        let filename = "prov.json";
        let content = format!("{}\n", serde_json::json!({ "api": { "m": model } }));
        std::fs::write(dir.join(filename), &content).expect("write shard");
        manifest_files.insert(filename.to_owned(), sha_of(&content));
        // report
        let error = pi_ai::model_data::validate_model_data_directory(&structure, &dir)
            .expect_err("mutated model fails");
        error.to_string()
    };

    let report = build_value(&mut |model| {
        model["input"] = serde_json::json!([]);
    });
    assert!(report.contains("invalid input modalities"), "got: {report}");
    let report = build_value(&mut |model| {
        model["contextWindow"] = serde_json::json!(0);
    });
    assert!(report.contains("invalid contextWindow"), "got: {report}");
    let report = build_value(&mut |model| {
        model["maxTokens"] = serde_json::json!(-1);
    });
    assert!(report.contains("invalid maxTokens"), "got: {report}");
    let report = build_value(&mut |model| {
        model["cost"]["cacheWrite"] = serde_json::json!(null);
    });
    assert!(report.contains("invalid cost.cacheWrite"), "got: {report}");
    let report = build_value(&mut |model| {
        model["name"] = serde_json::json!("");
    });
    assert!(report.contains("no model name"), "got: {report}");
    let report = build_value(&mut |model| {
        model["reasoning"] = serde_json::json!("yes");
    });
    assert!(report.contains("no reasoning boolean"), "got: {report}");
    let report = build_value(&mut |model| {
        model["cost"] = serde_json::json!("nope");
    });
    assert!(report.contains("invalid cost metadata"), "got: {report}");
    let report = build_value(&mut |model| {
        *model = serde_json::json!(7);
    });
    assert!(report.contains("must be an object"), "got: {report}");

    std::fs::remove_dir_all(&dir).expect("cleanup");
    let _ = &manifest_files;
}

fn sha_of(content: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(content.as_bytes());
    digest.iter().fold(String::new(), |mut hex, byte| {
        let _ = ::std::fmt::Write::write_fmt(&mut hex, format_args!("{byte:02x}"));
        hex
    })
}

/// The auth-resolution overlay: an env override shadows the base context and
/// a stored api-key credential merges the override env per field.
#[tokio::test]
async fn the_auth_resolution_overlays_the_request_env() {
    use pi_ai::auth::resolve::{AuthResolutionOverrides, resolve_provider_auth};
    use pi_ai::auth::types::{AuthContext, ProviderAuth};

    let auth = pi_ai::auth::helpers::env_api_key_auth("Overlay key", &["BASE_VAR"]);
    let provider_auth = ProviderAuth {
        api_key: Some(auth),
        oauth: None,
    };
    let credentials: Arc<dyn CredentialStore> =
        Arc::new(pi_ai::auth::credential_store::InMemoryCredentialStore::default());
    let context: Arc<dyn AuthContext> = Arc::new(OverlayFixture(BTreeMap::from([(
        "BASE_VAR".to_owned(),
        "base".to_owned(),
    )])));

    // No override: the base env resolves.
    let resolved = resolve_provider_auth("p", &provider_auth, &credentials, &context, None)
        .await
        .expect("resolve")
        .expect("configured");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("base"));

    // An override env shadows the base context.
    let resolved = resolve_provider_auth(
        "p",
        &provider_auth,
        &credentials,
        &context,
        Some(&AuthResolutionOverrides {
            env: Some(BTreeMap::from([(
                "BASE_VAR".to_owned(),
                "override".to_owned(),
            )])),
            ..AuthResolutionOverrides::default()
        }),
    )
    .await
    .expect("resolve")
    .expect("configured");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("override"));

    // A stored api-key credential merges the override env over its own.
    credentials
        .modify(
            "p",
            Box::new(move |_current| {
                Box::pin(async move {
                    Ok(Some(Credential::ApiKey(
                        pi_ai::auth::types::ApiKeyCredential {
                            key: Some("stored".to_owned()),
                            env: Some(BTreeMap::from([("STORED_ONLY".to_owned(), "s".to_owned())])),
                        },
                    )))
                })
            }),
            None,
        )
        .await
        .expect("store");
    let resolved = resolve_provider_auth(
        "p",
        &provider_auth,
        &credentials,
        &context,
        Some(&AuthResolutionOverrides {
            env: Some(BTreeMap::from([(
                "OVERRIDE_ONLY".to_owned(),
                "o".to_owned(),
            )])),
            ..AuthResolutionOverrides::default()
        }),
    )
    .await
    .expect("resolve")
    .expect("configured");
    let env = resolved.env.expect("merged env");
    assert_eq!(env.get("STORED_ONLY"), Some(&"s".to_owned()));
    assert!(env.contains_key("OVERRIDE_ONLY"));
    assert_eq!(resolved.auth.api_key.as_deref(), Some("stored"));
}

struct OverlayFixture(BTreeMap<String, String>);

impl AuthContext for OverlayFixture {
    fn env(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }

    fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

/// The OAuth resolution's "expires too soon" contract: an explicit minimum
/// validity longer than the refreshed token's remaining life rejects.
#[tokio::test]
async fn the_oauth_resolution_rejects_a_token_that_expires_too_soon() {
    use pi_ai::auth::resolve::{AuthResolutionOverrides, resolve_provider_auth};
    use pi_ai::auth::types::ProviderAuth;

    let credentials: Arc<dyn CredentialStore> =
        Arc::new(pi_ai::auth::credential_store::InMemoryCredentialStore::default());
    credentials
        .modify(
            "p",
            Box::new(move |_current| {
                Box::pin(async move {
                    Ok(Some(Credential::OAuth(OAuthCredentials {
                        refresh: "r".to_owned(),
                        access: "old".to_owned(),
                        expires: 0,
                        extra: BTreeMap::new(),
                    })))
                })
            }),
            None,
        )
        .await
        .expect("store");
    let oauth = fixture_oauth_with_refresh(60_000);
    let provider_auth = ProviderAuth {
        api_key: None,
        oauth: Some(oauth),
    };
    let overrides = AuthResolutionOverrides {
        min_oauth_validity_ms: Some(30 * 60_000),
        ..AuthResolutionOverrides::default()
    };
    let context: Arc<dyn AuthContext> = Arc::new(OverlayFixture(BTreeMap::new()));
    let error = resolve_provider_auth(
        "p",
        &provider_auth,
        &credentials,
        &context,
        Some(&overrides),
    )
    .await
    .expect_err("too soon rejects");
    assert!(
        error.to_string().contains("expires too soon"),
        "got: {error}"
    );
}

/// The OAuth fixture that refreshes to a token expiring in `expires_in_ms`.
fn fixture_oauth_with_refresh(expires_in: i64) -> pi_ai::auth::types::OAuthAuth {
    let login: pi_ai::auth::types::OAuthLoginFn = Arc::new(|_interaction| {
        let failing: pi_ai::auth::types::AuthError = Box::new(std::io::Error::other("unused"));
        let future: pi_ai::types::BoxedFuture<
            'static,
            Result<OAuthCredentials, pi_ai::auth::types::AuthError>,
        > = Box::pin(async move { Err(failing) });
        future
    });
    let refresh: pi_ai::auth::types::OAuthRefreshFn = Arc::new(move |credential, _signal| {
        Box::pin(async move {
            Ok(OAuthCredentials {
                refresh: credential.refresh,
                access: "new-token".to_owned(),
                expires: pi_ai::auth::resolve::now_ms() + expires_in,
                extra: credential.extra,
            })
        })
    });
    let to_auth: pi_ai::auth::types::OAuthToAuthFn = Arc::new(|credential| {
        Box::pin(async move {
            Ok(ModelAuth {
                api_key: Some(credential.access),
                ..ModelAuth::default()
            })
        })
    });
    pi_ai::auth::types::OAuthAuth {
        name: "Fixture OAuth".to_owned(),
        is_subscription: None,
        login_label: None,
        login,
        refresh,
        to_auth,
    }
}

/// The catalog registry's manifest stamp parses, and the typed lookup
/// round-trips a real entry, upstream's `getBuiltinModelDataGeneratedAt`.
#[test]
fn the_catalog_registry_parses_the_generated_at_stamp() {
    let stamp = pi_ai::providers::catalog::get_builtin_model_data_generated_at()
        .expect("the committed manifest stamps a timestamp");
    assert!(stamp > 0);
    let model = pi_ai::providers::catalog::get_builtin_models("anthropic")
        .into_iter()
        .find(|model| model.id.contains("opus"))
        .expect("an anthropic opus model");
    assert_eq!(model.provider.0, "anthropic");
}

/// The images runtime's static refresh path and the all-providers refresh.
#[tokio::test]
async fn the_images_refresh_defaults_stay_noops() {
    let models = pi_ai::images_models::create_images_models(None);
    models.set_provider(Arc::new(
        pi_ai::providers::openrouter_images::openrouter_images_provider(),
    ));
    // The builtin openrouter images provider is static: refresh is a noop.
    models.refresh(Some("openrouter")).await.expect("refresh");
    models.refresh(None).await.expect("refresh all");
    let _ = models.provider("openrouter").expect("registered");
}

/// The Provider trait's default methods, exercised through a real builtin.
#[tokio::test]
async fn the_provider_trait_defaults_exercise_their_shapes() {
    let provider = pi_ai::providers::kimi_coding::kimi_coding_provider();
    // A static provider exposes no base-url defaults beyond its own wiring,
    // no refresh support, and no deferred capability.
    assert_eq!(provider.id(), "kimi-coding");
    assert_eq!(provider.base_url(), Some("https://api.kimi.com/coding"));
    assert!(!provider.supports_refresh_models());
    assert!(!provider.supports_fetch_deferred());
    assert!(!provider.supports_cancel_deferred());
    // filter_models defaults to identity.
    let listed = provider.get_models().expect("catalog");
    assert_eq!(provider.filter_models(listed.clone(), None), listed);
}

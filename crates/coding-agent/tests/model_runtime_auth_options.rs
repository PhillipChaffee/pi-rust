//! Upstream `packages/coding-agent/test/model-runtime-auth-options.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated for
//! `pi_coding_agent::model_runtime` (#121).
//!
//! Porting restatements this suite records:
//!
//! - The `authOptions` enumeration helper flattens upstream's
//!   `{ type, provider, method }` members into an [`AuthOption`] record with
//!   the fields the assertions probe; `toMatchObject` probes restate as
//!   field assertions on it.
//! - `AbortController` restates as [`CancellationToken`]; the recorded
//!   `AbortSignal.reason` identity has no counterpart, so the cancellation
//!   case pins signal presence and cancellation only.
//! - The two header-capture cases end their request through an
//!   error-terminated stream: upstream's `streamSimple` throws after
//!   capturing, the port's extension closure records the headers and
//!   returns the `Error`-event stream, and `complete_simple` settles on the
//!   failing assistant message the way awaiting the rejected promise did.
//! - The transform case's `expect(options).not.toHaveProperty(
//!   "transformHeaders")` probe is structural here: the extension closure
//!   receives [`SimpleStreamOptions`], which carries no transform slot — the
//!   transform rides the [`WithTransforms`] wrapper and never reaches the
//!   provider.
//! - The provider-scoped availability failure does not publish the
//!   availability error upstream's `getAvailable(providerId)` catch records
//!   (upstream model-runtime.ts:412-413); the port's provider-scoped
//!   `get_available` drops that record, so the ported case pins the clear
//!   error surface at that point and the gap is reported on #121.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

#[expect(
    dead_code,
    reason = "the fixture module compiles whole into every test binary; this suite drives only its model-layer helpers"
)]
mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use common::model_layer::{
    create_in_memory_model_registry, empty_context, in_memory_auth_storage, zero_cost,
};
use pi_ai::auth::credential_store::{CredentialStore, InMemoryCredentialStore};
use pi_ai::auth::resolve::now_ms;
use pi_ai::auth::types::{
    ApiKeyCredential, AuthError, AuthOptions, AuthType, Credential, CredentialInfo,
    CredentialModifyFn, ModelAuth, OAuthCredentials,
};
use pi_ai::models::TransformHeadersFn;
use pi_ai::types::{
    Api, AssistantMessage, AssistantMessageEvent, BoxedFuture, Modality, ProviderEnv,
    ProviderHeaders, ProviderId, StopReason, Usage,
};
use pi_ai::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use pi_coding_agent::auth_storage::AuthStorageData;
use pi_coding_agent::model_runtime::{ModelRuntime, ModelRuntimeAuthOverrides};
use pi_coding_agent::provider_composer::{
    ExtensionOAuthConfig, ProviderConfigInput, ProviderModelInput,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// The runtime over in-memory credentials, upstream's
/// `ModelRuntime.create({ credentials, modelsPath: null })`.
async fn runtime_with(credentials: Arc<dyn CredentialStore>) -> ModelRuntime {
    create_in_memory_model_registry(credentials)
        .await
        .runtime()
        .clone()
}

/// The empty in-memory store, upstream's `AuthStorage.inMemory()`.
fn empty_store() -> Arc<dyn CredentialStore> {
    in_memory_auth_storage(&AuthStorageData::new())
}

/// An OAuth credential entry, upstream's inline `{ type: "oauth", ... }`
/// seed value.
fn oauth_entry(access: &str, refresh: &str, expires: i64) -> serde_json::Value {
    json!({"type": "oauth", "access": access, "refresh": refresh, "expires": expires})
}

/// The store seeded with one credential entry, upstream's
/// `AuthStorage.inMemory({ [providerId]: entry })`.
fn seeded_store(provider_id: &str, entry: serde_json::Value) -> Arc<dyn CredentialStore> {
    let mut data = AuthStorageData::new();
    data.insert(provider_id.to_owned(), entry);
    in_memory_auth_storage(&data)
}

/// The extension registration's model fixture, upstream's `testModel`: a
/// zero-cost text model over the example base URL.
fn test_model(id: &str) -> ProviderModelInput {
    ProviderModelInput {
        id: id.to_owned(),
        name: id.to_owned(),
        api: None,
        base_url: None,
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: zero_cost(),
        context_window: 10_000,
        max_tokens: 1_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// One enumerated auth-method option, upstream's `{ type, provider, method }`
/// member flattened into the fields the assertions probe.
struct AuthOption {
    auth_type: AuthType,
    provider_id: String,
    provider_name: String,
    method_name: String,
    is_subscription: Option<bool>,
    has_login: bool,
}

/// The `authOptions` helper: every provider's oauth method first, then its
/// api-key method, filtered to `auth_type` when given.
fn auth_options(runtime: &ModelRuntime, auth_type: Option<AuthType>) -> Vec<AuthOption> {
    runtime
        .get_providers()
        .iter()
        .flat_map(|provider| {
            let mut options = Vec::new();
            if auth_type.is_none_or(|selected| selected == AuthType::OAuth)
                && let Some(oauth) = provider.auth().oauth.as_ref()
            {
                options.push(AuthOption {
                    auth_type: AuthType::OAuth,
                    provider_id: provider.id().to_owned(),
                    provider_name: provider.name().to_owned(),
                    method_name: oauth.name.clone(),
                    is_subscription: oauth.is_subscription,
                    has_login: true,
                });
            }
            if auth_type.is_none_or(|selected| selected == AuthType::ApiKey)
                && let Some(api_key) = provider.auth().api_key.as_ref()
            {
                options.push(AuthOption {
                    auth_type: AuthType::ApiKey,
                    provider_id: provider.id().to_owned(),
                    provider_name: provider.name().to_owned(),
                    method_name: api_key.name.clone(),
                    is_subscription: None,
                    has_login: api_key.login.is_some(),
                });
            }
            options
        })
        .collect()
}

/// The error-terminated stream the capture cases return in place of
/// upstream's synchronous `throw`: the headers are already captured when the
/// stream is built, and the stream resolves to the failing assistant
/// message the way the rejected promise did.
fn captured_stream(
    provider_id: &str,
    model_id: &str,
    message: &str,
) -> AssistantMessageEventStream {
    let stream = assistant_message_event_stream();
    let failed = AssistantMessage {
        content: Vec::new(),
        api: Api::from("openai-completions"),
        provider: ProviderId::from(provider_id),
        model: model_id.to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        deferred: None,
        error_message: Some(message.to_owned()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    };
    stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: failed,
    });
    stream
}

/// The one cell a capture closure writes into, poisoned-lock tolerant like
/// every house mutex read.
type CaptureCell<T> = Arc<Mutex<T>>;

fn take_capture<T: Clone>(cell: &CaptureCell<T>) -> T {
    cell.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

fn set_capture<T>(cell: &CaptureCell<T>, value: T) {
    *cell
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = value;
}

/// The store that records provider-scoped reads and can fail them, upstream's
/// inline `credentials` object wrapping the in-memory base.
struct ProbeStore {
    base: Arc<InMemoryCredentialStore>,
    reads: CaptureCell<Vec<String>>,
    fail: Arc<AtomicBool>,
}

impl CredentialStore for ProbeStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        let provider_id = provider_id.to_owned();
        let reads = Arc::clone(&self.reads);
        let fail = Arc::clone(&self.fail);
        let base = Arc::clone(&self.base);
        Box::pin(async move {
            reads
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(provider_id.clone());
            if fail.load(Ordering::SeqCst) {
                let error: AuthError = Box::new(std::io::Error::other(format!(
                    "read failed for {provider_id}"
                )));
                return Err(error);
            }
            base.read(&provider_id, options).await
        })
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        self.base.list(options)
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: CredentialModifyFn,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        self.base.modify(provider_id, f, options)
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        self.base.delete(provider_id, options)
    }
}

/// upstream `it("accepts a pi-ai CredentialStore")`: the runtime reads
/// through the injected store.
#[tokio::test]
async fn accepts_a_pi_ai_credential_store() {
    let credentials = Arc::new(InMemoryCredentialStore::default());
    let seed: CredentialModifyFn = Box::new(|_current| {
        Box::pin(async {
            Ok(Some(Credential::ApiKey(ApiKeyCredential {
                key: Some("stored-key".to_owned()),
                env: None,
            })))
        })
    });
    credentials
        .modify("anthropic", seed, None)
        .await
        .expect("the seed write lands");
    let runtime = runtime_with(credentials).await;

    let auth = runtime
        .get_auth("anthropic", None)
        .await
        .expect("the resolution succeeds")
        .expect("the provider is configured");
    assert_eq!(auth.auth.api_key.as_deref(), Some("stored-key"));
}

/// upstream `it("scopes provider availability reads and records refresh
/// failures")`.
#[tokio::test]
async fn scopes_provider_availability_reads_and_records_refresh_failures() {
    let reads: CaptureCell<Vec<String>> = Arc::new(Mutex::new(Vec::new()));
    let fail = Arc::new(AtomicBool::new(false));
    let runtime = runtime_with(Arc::new(ProbeStore {
        base: Arc::new(InMemoryCredentialStore::default()),
        reads: Arc::clone(&reads),
        fail: Arc::clone(&fail),
    }))
    .await;

    set_capture(&reads, Vec::new());
    runtime
        .get_available(Some("anthropic"), None)
        .await
        .expect("the scoped pass succeeds");
    let scoped_reads: BTreeSet<String> = take_capture(&reads).into_iter().collect();
    assert_eq!(
        scoped_reads,
        BTreeSet::from(["anthropic".to_owned()]),
        "the scoped pass reads only the selected provider"
    );

    fail.store(true, Ordering::SeqCst);
    let failure = runtime
        .get_available(Some("anthropic"), None)
        .await
        .expect_err("the failing read surfaces");
    let message = failure.to_string();
    assert!(
        message.contains("Credential store read failed for anthropic"),
        "unexpected failure message: {message}"
    );
    // upstream model-runtime.ts:412-413 records the provider-scoped failure
    // on the error surface before rethrowing.
    assert!(
        runtime.get_error().is_some_and(|error| error
            .contains("Availability refresh: Credential store read failed for anthropic")),
        "the scoped failure is recorded on the error surface, got: {:?}",
        runtime.get_error()
    );

    fail.store(false, Ordering::SeqCst);
    runtime
        .get_available(None, None)
        .await
        .expect("the full pass succeeds");
    assert!(
        runtime.get_error().is_none(),
        "the full pass keeps the error surface clear"
    );
}

/// upstream `it("projects provider-owned methods, names, and status")`.
#[tokio::test]
async fn projects_provider_owned_methods_names_and_status() {
    let runtime = runtime_with(empty_store()).await;
    let options = auth_options(&runtime, None);

    let contains = |auth_type: AuthType, provider_id: &str, provider_name: &str| {
        options.iter().any(|option| {
            option.auth_type == auth_type
                && option.provider_id == provider_id
                && option.provider_name == provider_name
        })
    };
    let contains_method =
        |auth_type: AuthType, provider_id: &str, provider_name: &str, method_name: &str| {
            contains(auth_type, provider_id, provider_name)
                && options
                    .iter()
                    .any(|option| option.method_name == method_name)
        };
    assert!(
        contains_method(
            AuthType::ApiKey,
            "amazon-bedrock",
            "Amazon Bedrock",
            "AWS credentials or bearer token"
        ),
        "the bedrock api-key method is enumerated"
    );
    assert!(
        contains_method(
            AuthType::ApiKey,
            "google-vertex",
            "Google Vertex AI",
            "Google Cloud credentials"
        ),
        "the vertex api-key method is enumerated"
    );
    assert!(
        options
            .iter()
            .any(|option| option.auth_type == AuthType::OAuth
                && option.provider_id == "anthropic"
                && option.provider_name == "Anthropic"),
        "the anthropic oauth method is enumerated"
    );
    assert!(
        contains(
            AuthType::ApiKey,
            "cloudflare-ai-gateway",
            "Cloudflare AI Gateway"
        ),
        "the gateway api-key method is enumerated"
    );
    assert!(
        contains(
            AuthType::ApiKey,
            "cloudflare-workers-ai",
            "Cloudflare Workers AI"
        ),
        "the workers-ai api-key method is enumerated"
    );
    assert!(
        auth_options(&runtime, Some(AuthType::ApiKey))
            .iter()
            .all(|option| option.auth_type == AuthType::ApiKey),
        "the api-key filter yields only api-key options"
    );
    assert!(
        auth_options(&runtime, Some(AuthType::OAuth))
            .iter()
            .all(|option| option.auth_type == AuthType::OAuth),
        "the oauth filter yields only oauth options"
    );
    assert!(
        !options
            .iter()
            .any(|option| option.provider_id == "openai-codex"
                && option.auth_type == AuthType::ApiKey),
        "the oauth-only codex provider fabricates no api-key method"
    );
}

/// upstream `it("attaches the provider's active auth status to every method
/// option")`.
#[tokio::test]
async fn attaches_the_providers_active_auth_status_to_every_method_option() {
    let runtime = runtime_with(seeded_store(
        "anthropic",
        oauth_entry("access", "refresh", now_ms() + 60_000),
    ))
    .await;

    let options = auth_options(&runtime, None)
        .into_iter()
        .filter(|option| option.provider_id == "anthropic");
    assert_eq!(options.count(), 2, "both anthropic methods enumerate");
    let check = runtime
        .check_auth("anthropic", None)
        .await
        .expect("the check resolves")
        .expect("the seeded credential configures the provider");
    assert_eq!(check.auth_type, AuthType::OAuth);
}

/// upstream `it("distinguishes subscription OAuth from generic OAuth
/// sign-in")`.
#[tokio::test]
async fn distinguishes_subscription_oauth_from_generic_oauth_sign_in() {
    let mut data = AuthStorageData::new();
    data.insert(
        "anthropic".to_owned(),
        oauth_entry(
            "anthropic-access",
            "anthropic-refresh",
            now_ms() + 60 * 60_000,
        ),
    );
    data.insert(
        "openrouter".to_owned(),
        oauth_entry("openrouter-key", "", 9_007_199_254_740_991),
    );
    data.insert(
        "radius".to_owned(),
        oauth_entry("radius-access", "radius-refresh", now_ms() + 60 * 60_000),
    );
    let runtime = runtime_with(in_memory_auth_storage(&data)).await;

    assert!(runtime.is_using_oauth("anthropic"));
    assert!(runtime.is_using_subscription("anthropic"));
    assert!(runtime.is_using_oauth("openrouter"));
    assert!(!runtime.is_using_subscription("openrouter"));
    assert!(runtime.is_using_oauth("radius"));
    assert!(!runtime.is_using_subscription("radius"));
}

/// upstream `it("constructs an API key method for an extension API-key
/// provider")`.
#[tokio::test]
async fn constructs_an_api_key_method_for_an_extension_api_key_provider() {
    let runtime = runtime_with(empty_store()).await;
    runtime
        .register_provider(
            "extension-api-key",
            ProviderConfigInput {
                name: Some("Extension API Key".to_owned()),
                base_url: Some("https://example.test/v1".to_owned()),
                api_key: Some("$EXTENSION_TEST_API_KEY".to_owned()),
                api: Some(Api::from("openai-completions")),
                models: Some(vec![test_model("extension-model")]),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the registration validates");

    let options: Vec<AuthOption> = auth_options(&runtime, None)
        .into_iter()
        .filter(|option| option.provider_id == "extension-api-key")
        .collect();
    assert_eq!(options.len(), 1, "one method enumerates");
    assert_eq!(options[0].auth_type, AuthType::ApiKey);
    assert_eq!(options[0].provider_name, "Extension API Key");
    assert_eq!(options[0].method_name, "API key");
    assert!(
        options[0].has_login,
        "the constructed method carries a login flow"
    );
}

/// upstream `it("resolves configured auth from request-scoped environment
/// overrides")`.
#[tokio::test]
async fn resolves_configured_auth_from_request_scoped_environment_overrides() {
    let runtime = runtime_with(empty_store()).await;
    runtime
        .register_provider(
            "request-env-provider",
            ProviderConfigInput {
                base_url: Some("https://example.test/v1".to_owned()),
                api_key: Some("$REQUEST_SCOPED_API_KEY".to_owned()),
                headers: Some(BTreeMap::from([(
                    "x-request-value".to_owned(),
                    "$REQUEST_SCOPED_HEADER".to_owned(),
                )])),
                api: Some(Api::from("openai-completions")),
                models: Some(vec![test_model("request-env-model")]),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the registration validates");

    let overrides = ModelRuntimeAuthOverrides {
        env: Some(ProviderEnv::from([
            (
                "REQUEST_SCOPED_API_KEY".to_owned(),
                "request-key".to_owned(),
            ),
            (
                "REQUEST_SCOPED_HEADER".to_owned(),
                "request-header".to_owned(),
            ),
        ])),
        ..ModelRuntimeAuthOverrides::default()
    };
    let auth = runtime
        .get_auth("request-env-provider", Some(&overrides))
        .await
        .expect("the resolution succeeds")
        .expect("the request-scoped env configures the provider");
    assert_eq!(
        auth.auth,
        ModelAuth {
            api_key: Some("request-key".to_owned()),
            headers: Some(ProviderHeaders::from([(
                "x-request-value".to_owned(),
                Some("request-header".to_owned()),
            )])),
            base_url: None,
        }
    );
}

/// upstream `it("lets an explicit Authorization header override authHeader
/// case-insensitively")`.
#[tokio::test]
async fn lets_an_explicit_authorization_header_override_auth_header_case_insensitively() {
    let runtime = runtime_with(empty_store()).await;
    let captured: CaptureCell<Option<ProviderHeaders>> = Arc::new(Mutex::new(None));
    runtime
        .register_provider(
            "auth-header-provider",
            ProviderConfigInput {
                base_url: Some("https://example.test/v1".to_owned()),
                api_key: Some("generated-key".to_owned()),
                auth_header: Some(true),
                api: Some(Api::from("openai-completions")),
                stream_simple: Some(Arc::new({
                    let captured = Arc::clone(&captured);
                    move |_model, _context, options| {
                        set_capture(
                            &captured,
                            options.and_then(|options| options.headers.clone()),
                        );
                        captured_stream("auth-header-provider", "auth-header-model", "captured")
                    }
                })),
                models: Some(vec![test_model("auth-header-model")]),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the registration validates");
    let model = runtime
        .get_model("auth-header-provider", "auth-header-model")
        .expect("the registered model resolves");

    let options = pi_ai::models::ModelsSimpleStreamOptions {
        options: pi_ai::types::SimpleStreamOptions {
            headers: Some(ProviderHeaders::from([(
                "authorization".to_owned(),
                Some("Explicit token".to_owned()),
            )])),
            ..pi_ai::types::SimpleStreamOptions::default()
        },
        transform_headers: None,
    };
    let _ = runtime
        .complete_simple(&model, &empty_context(), Some(&options))
        .await;

    assert_eq!(
        take_capture(&captured),
        Some(ProviderHeaders::from([(
            "authorization".to_owned(),
            Some("Explicit token".to_owned()),
        )])),
        "the explicit lowercase name replaced the composed Authorization header"
    );
}

/// upstream `it("transforms fully assembled headers once without forwarding
/// the transform")`.
#[tokio::test]
async fn transforms_fully_assembled_headers_once_without_forwarding_the_transform() {
    use std::sync::atomic::AtomicUsize;

    let runtime = runtime_with(empty_store()).await;
    let captured: CaptureCell<Option<ProviderHeaders>> = Arc::new(Mutex::new(None));
    let transforms = Arc::new(AtomicUsize::new(0));
    runtime
        .register_provider(
            "header-provider",
            ProviderConfigInput {
                base_url: Some("https://example.test/v1".to_owned()),
                api_key: Some("generated-key".to_owned()),
                auth_header: Some(true),
                headers: Some(BTreeMap::from([(
                    "x-provider".to_owned(),
                    "provider".to_owned(),
                )])),
                api: Some(Api::from("openai-completions")),
                stream_simple: Some(Arc::new({
                    let captured = Arc::clone(&captured);
                    move |_model, _context, options| {
                        set_capture(
                            &captured,
                            options.and_then(|options| options.headers.clone()),
                        );
                        captured_stream("header-provider", "header-model", "captured")
                    }
                })),
                models: Some(vec![ProviderModelInput {
                    headers: Some(BTreeMap::from([("x-model".to_owned(), "model".to_owned())])),
                    ..test_model("header-model")
                }]),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the registration validates");
    let model = runtime
        .get_model("header-provider", "header-model")
        .expect("the registered model resolves");

    let assembled: ProviderHeaders = ProviderHeaders::from([
        (
            "Authorization".to_owned(),
            Some("Bearer generated-key".to_owned()),
        ),
        ("x-provider".to_owned(), Some("provider".to_owned())),
        ("x-model".to_owned(), Some("model".to_owned())),
        ("x-explicit".to_owned(), Some("explicit".to_owned())),
    ]);
    let expected_for_closure = assembled.clone();
    let transform: TransformHeadersFn = Arc::new({
        let transforms = Arc::clone(&transforms);
        move |headers: ProviderHeaders| {
            transforms.fetch_add(1, Ordering::SeqCst);
            assert_eq!(
                headers, expected_for_closure,
                "the transform sees the fully assembled headers once"
            );
            let mut transformed = headers;
            transformed.insert("x-transformed".to_owned(), Some("yes".to_owned()));
            Box::pin(async move { transformed })
        }
    });
    let options = pi_ai::models::ModelsSimpleStreamOptions {
        options: pi_ai::types::SimpleStreamOptions {
            headers: Some(ProviderHeaders::from([(
                "x-explicit".to_owned(),
                Some("explicit".to_owned()),
            )])),
            ..pi_ai::types::SimpleStreamOptions::default()
        },
        transform_headers: Some(transform),
    };
    let _ = runtime
        .complete_simple(&model, &empty_context(), Some(&options))
        .await;

    assert_eq!(
        transforms.load(Ordering::SeqCst),
        1,
        "the transform ran exactly once"
    );
    let mut expected = assembled;
    expected.insert("x-transformed".to_owned(), Some("yes".to_owned()));
    assert_eq!(take_capture(&captured), Some(expected));
}

/// upstream `it("forwards cancellation to extension OAuth refresh")`.
#[tokio::test]
async fn forwards_cancellation_to_extension_oauth_refresh() {
    let runtime = runtime_with(seeded_store(
        "extension-oauth",
        oauth_entry("expired", "refresh", 0),
    ))
    .await;
    let recorded: CaptureCell<Option<CancellationToken>> = Arc::new(Mutex::new(None));
    runtime
        .register_provider(
            "extension-oauth",
            ProviderConfigInput {
                name: Some("Extension OAuth".to_owned()),
                base_url: Some("https://example.test/v1".to_owned()),
                api: Some(Api::from("openai-completions")),
                oauth: Some(ExtensionOAuthConfig {
                    name: "Extension subscription".to_owned(),
                    is_subscription: None,
                    uses_callback_server: None,
                    login: Arc::new(|_callbacks| {
                        Box::pin(async {
                            Ok(OAuthCredentials {
                                access: "access".to_owned(),
                                refresh: "refresh".to_owned(),
                                expires: now_ms() + 60_000,
                                extra: BTreeMap::new(),
                            })
                        })
                    }),
                    refresh_token: Arc::new({
                        let recorded = Arc::clone(&recorded);
                        move |credentials, signal| {
                            set_capture(&recorded, Some(signal));
                            let refreshed = OAuthCredentials {
                                expires: now_ms() + 60_000,
                                ..credentials
                            };
                            Box::pin(async move { Ok(refreshed) })
                        }
                    }),
                    get_api_key: Arc::new(|credentials| credentials.access.clone()),
                    modify_models: None,
                }),
                models: Some(vec![test_model("extension-model")]),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the registration validates");

    let token = CancellationToken::new();
    let overrides = ModelRuntimeAuthOverrides {
        signal: Some(token.clone()),
        ..ModelRuntimeAuthOverrides::default()
    };
    let _ = runtime
        .get_auth("extension-oauth", Some(&overrides))
        .await
        .expect("the resolution succeeds");
    let recorded_token = take_capture(&recorded).expect("the refresh received a signal");
    token.cancel();
    assert!(
        recorded_token.is_cancelled(),
        "the caller's cancellation is observable through the recorded signal"
    );
}

/// upstream `it("does not fabricate an API key method for an extension
/// OAuth-only provider")`.
#[tokio::test]
async fn does_not_fabricate_an_api_key_method_for_an_extension_oauth_only_provider() {
    let runtime = runtime_with(empty_store()).await;
    runtime
        .register_provider(
            "extension-oauth",
            ProviderConfigInput {
                name: Some("Extension OAuth".to_owned()),
                base_url: Some("https://example.test/v1".to_owned()),
                api: Some(Api::from("openai-completions")),
                oauth: Some(ExtensionOAuthConfig {
                    name: "Extension subscription".to_owned(),
                    is_subscription: Some(true),
                    uses_callback_server: None,
                    login: Arc::new(|_callbacks| {
                        Box::pin(async {
                            Ok(OAuthCredentials {
                                access: "access".to_owned(),
                                refresh: "refresh".to_owned(),
                                expires: now_ms() + 60_000,
                                extra: BTreeMap::new(),
                            })
                        })
                    }),
                    refresh_token: Arc::new(|credentials, _signal| {
                        Box::pin(async move { Ok(credentials) })
                    }),
                    get_api_key: Arc::new(|credentials| credentials.access.clone()),
                    modify_models: None,
                }),
                models: Some(vec![test_model("extension-model")]),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the registration validates");

    let options: Vec<AuthOption> = auth_options(&runtime, None)
        .into_iter()
        .filter(|option| option.provider_id == "extension-oauth")
        .collect();
    assert_eq!(options.len(), 1, "only the oauth method enumerates");
    assert_eq!(options[0].auth_type, AuthType::OAuth);
    assert_eq!(options[0].provider_name, "Extension OAuth");
    assert_eq!(options[0].method_name, "Extension subscription");
    assert_eq!(options[0].is_subscription, Some(true));
}

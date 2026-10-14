//! The auth command suites, upstream's `test/auth-check.test.ts` and
//! `test/credential-print.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements this suite records:
//!
//! - The built-in openai/openai-codex providers restate to registered
//!   providers over the [`pi_coding_agent::provider_composer`]
//!   registration seam: an API-key provider with a `$VAR` template and an
//!   extension-OAuth provider whose `refresh_token` closure is the spy
//!   upstream installed on `oauth.refresh`.
//! - The `parseArgs([...])` fixtures restate to [`AuthCommandArgs`] views
//!   (the full CLI grammar ports with the CLI-grammar slice).
//! - The `main([...])` unknown-option case rides the CLI-grammar slice's
//!   dispatcher; the grammar-level assertions port through
//!   `parse_auth_command`/`validate_auth_command_args`.
//! - The failure-path cases upstream injects through spies and stores
//!   restate to a [`FailingStore`] credential store whose reads and lists
//!   fail on demand, and to a refresh closure that fails; the wall-clock
//!   fixtures carry absolute `expires` values so no test sleeps.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pi_ai::auth::credential_store::{CredentialStore, InMemoryCredentialStore};
use pi_ai::auth::types::{
    AuthError, AuthOptions, AuthResult, Credential, CredentialInfo, CredentialModifyFn, ModelAuth,
    OAuthCredentials,
};
use pi_ai::types::{Api, BoxedFuture};
use pi_coding_agent::auth_storage::{AuthStorage, InMemoryAuthStorageBackend};
use pi_coding_agent::cli::auth_check::{
    AuthCheckOptions, AuthCheckReason, AuthCheckResult, AuthCheckStatus, check_provider_auth,
    create_auth_check_model_runtime, get_provider_credential,
};
use pi_coding_agent::cli::auth_command::{
    AuthCommandArgs, AuthCommandError, AuthCommandKind, UnknownFlagValue, get_auth_command_name,
    get_auth_command_usage, get_auth_credential, is_auth_command_help, parse_auth_command,
    print_auth_command_help, validate_auth_command_args,
};
use pi_coding_agent::cli::credential_print::{CredentialPrintKind, resolve_credential_for_print};
use pi_coding_agent::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use pi_coding_agent::models_store::InMemoryCodingAgentModelsStore;
use pi_coding_agent::provider_composer::{
    ExtensionOAuthConfig, ExtensionOAuthRefreshFn, ProviderConfigInput,
};

fn args_view(provider: Option<&str>, model: Option<&str>) -> AuthCommandArgs {
    AuthCommandArgs {
        provider: provider.map(str::to_string),
        model: model.map(str::to_string),
        ..AuthCommandArgs::default()
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| i64::try_from(since.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or_default()
}

/// The refresh spy, upstream's `vi.fn(oauth.refresh)` restated to the
/// extension-OAuth closure the registration seam carries.
type RefreshSpy = Arc<Mutex<Vec<OAuthCredentials>>>;

/// The seed data one provider entry carries, upstream's `inMemory({...})`.
fn auth_data(
    entries: &[(&str, serde_json::Value)],
) -> pi_coding_agent::auth_storage::AuthStorageData {
    entries
        .iter()
        .map(|(provider, value)| ((*provider).to_string(), value.clone()))
        .collect()
}

/// The runtime over the given credential store, the suite's create
/// fixture: no model catalog storage, no network, no refresh on create.
async fn runtime_with_credentials(credentials: Arc<dyn CredentialStore>) -> ModelRuntime {
    ModelRuntime::create(CreateModelRuntimeOptions {
        credentials: Some(credentials),
        models_path: Some(None),
        models_store: Some(Arc::new(InMemoryCodingAgentModelsStore::default())),
        allow_model_network: false,
        refresh_on_create: Some(false),
        ..CreateModelRuntimeOptions::default()
    })
    .await
    .expect("runtime creates")
}

/// The registered api-key provider whose key is a template or literal, the
/// registration seam's simplest input.
fn register_api_key_provider(runtime: &ModelRuntime, name: &str, display: &str, api_key: &str) {
    runtime
        .register_provider(
            name,
            ProviderConfigInput {
                name: Some(display.to_string()),
                api: Some(Api::from("openai-completions")),
                api_key: Some(api_key.to_string()),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the registration validates");
}

/// The auth check over the default options, the check cases' shared call.
async fn check_provider(
    runtime: &ModelRuntime,
    provider: Option<&str>,
    model: Option<&str>,
) -> AuthCheckResult {
    check_provider_auth(
        &args_view(provider, model),
        runtime,
        AuthCheckOptions::default(),
    )
    .await
    .expect("check")
}

/// The credential-print resolution over the no-signal default options, the
/// print cases' shared call.
async fn resolve_print(
    runtime: &ModelRuntime,
    provider: Option<&str>,
    model: Option<&str>,
    kind: CredentialPrintKind,
    min_expiry_ms: Option<i64>,
) -> Result<String, AuthCommandError> {
    resolve_credential_for_print(
        &args_view(provider, model),
        runtime,
        kind,
        min_expiry_ms,
        None,
    )
    .await
}

/// The refresh spy's recorded call count, the refresh assertions' probe.
fn refresh_count(spy: &RefreshSpy) -> usize {
    spy.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .len()
}

/// The OAuth credentials the login flow answers, the shared login fixture.
fn login_credentials() -> OAuthCredentials {
    OAuthCredentials {
        access: "login-access".to_string(),
        refresh: "refresh-token".to_string(),
        expires: now_ms() + 60_000,
        extra: BTreeMap::new(),
    }
}

/// The shared OAuth provider registration: the login closure answers
/// `login_credentials`, the refresh closure is the rig's own, and the
/// model catalog carries one `oauth-model` entry.
fn register_oauth_provider(runtime: &ModelRuntime, refresh_token: ExtensionOAuthRefreshFn) {
    runtime
        .register_provider(
            "oauth-provider",
            ProviderConfigInput {
                name: Some("OAuth Provider".to_string()),
                base_url: Some("https://example.test/v1".to_string()),
                api: Some(Api::from("openai-completions")),
                oauth: Some(ExtensionOAuthConfig {
                    name: "OAuth subscription".to_string(),
                    is_subscription: None,
                    uses_callback_server: None,
                    login: Arc::new(|_callbacks| Box::pin(async { Ok(login_credentials()) })),
                    refresh_token,
                    get_api_key: Arc::new(|credentials| credentials.access.clone()),
                    modify_models: None,
                }),
                models: Some(vec![oauth_model()]),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the registration validates");
}

async fn runtime_with_api_key_credentials(key: &str) -> ModelRuntime {
    let storage = AuthStorage::<InMemoryAuthStorageBackend>::in_memory(&auth_data(&[(
        "openai",
        serde_json::json!({ "type": "api_key", "key": key }),
    )]));
    runtime_with_credentials(Arc::new(storage)).await
}

/// The OAuth rig: a credential expiring at `expires_ms` with access token
/// `access`, plus a provider whose refresh spy answers per `outcome` and
/// records the input.
async fn oauth_rig(
    expires_ms: i64,
    access: &str,
    outcome: RefreshOutcome,
) -> (
    ModelRuntime,
    RefreshSpy,
    Arc<AuthStorage<InMemoryAuthStorageBackend>>,
) {
    let storage = Arc::new(AuthStorage::<InMemoryAuthStorageBackend>::in_memory(
        &auth_data(&[(
            "oauth-provider",
            serde_json::json!({
                "type": "oauth",
                "access": access,
                "refresh": "refresh-token",
                "expires": expires_ms,
            }),
        )]),
    ));
    let spy: RefreshSpy = Arc::new(Mutex::new(Vec::new()));
    let credentials: Arc<dyn CredentialStore> = storage.clone();
    let runtime = runtime_with_credentials(credentials).await;

    let spy_for_closure = Arc::clone(&spy);
    register_oauth_provider(
        &runtime,
        Arc::new(move |credentials, _signal| {
            spy_for_closure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(credentials.clone());
            match outcome {
                RefreshOutcome::Fresh => {
                    let refreshed = OAuthCredentials {
                        access: "fresh-token".to_string(),
                        expires: now_ms() + 60 * 60 * 1000,
                        ..credentials
                    };
                    Box::pin(async move { Ok(refreshed) })
                }
                RefreshOutcome::Soon => {
                    let refreshed = OAuthCredentials {
                        access: "soon-token".to_string(),
                        expires: now_ms() + 10 * 60 * 1000,
                        ..credentials
                    };
                    Box::pin(async move { Ok(refreshed) })
                }
                RefreshOutcome::Fails => {
                    let error: Box<dyn std::error::Error + Send + Sync> =
                        Box::new(std::io::Error::other("refresh failed"));
                    Box::pin(async move { Err(error) })
                }
            }
        }),
    );
    (runtime, spy, storage)
}

/// The refresh closure's outcome, upstream's `oauth.refresh` stubs.
#[derive(Clone, Copy)]
enum RefreshOutcome {
    /// Answer a fresh token valid for the next hour.
    Fresh,
    /// Answer a token that still misses a demanding `--min-expiry`.
    Soon,
    /// Fail the refresh, upstream's rejecting stub.
    Fails,
}

/// The OAuth rig over an expired credential whose refresh answers a fresh
/// token, the suite's default fixture.
async fn runtime_with_expired_oauth() -> (
    ModelRuntime,
    RefreshSpy,
    Arc<AuthStorage<InMemoryAuthStorageBackend>>,
) {
    oauth_rig(0, "old-token", RefreshOutcome::Fresh).await
}

/// The one-model input the OAuth rig registers, the restated
/// `test_model("extension-model")` fixture.
fn oauth_model() -> pi_coding_agent::provider_composer::ProviderModelInput {
    use pi_ai::types::{Modality, ModelCost};
    pi_coding_agent::provider_composer::ProviderModelInput {
        id: "oauth-model".to_string(),
        name: "oauth-model".to_string(),
        api: None,
        base_url: None,
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: ModelCost::default(),
        context_window: 10_000,
        max_tokens: 1_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

// =============================================================================
// auth check, upstream's test/auth-check.test.ts
// =============================================================================

#[tokio::test]
async fn reports_a_configured_provider_as_ready() {
    let runtime = runtime_with_api_key_credentials("test-key").await;
    let result = check_provider(&runtime, Some("openai"), None).await;
    assert!(matches!(result.status, AuthCheckStatus::Ready));
    assert_eq!(result.provider, "openai");
    assert!(
        result
            .auth_type
            .is_some_and(|auth_type| auth_type.to_string() == "api_key")
    );
    assert!(result.reason.is_none());
}

#[tokio::test]
async fn resolves_the_provider_from_the_model_flag() {
    let runtime = runtime_with_api_key_credentials("test-key").await;
    let from_model = check_provider(&runtime, None, Some("openai/gpt-5.5")).await;
    assert!(matches!(from_model.status, AuthCheckStatus::Ready));
    assert_eq!(from_model.provider, "openai");

    let from_both = check_provider(&runtime, Some("openai"), Some("gpt-5.5")).await;
    assert!(matches!(from_both.status, AuthCheckStatus::Ready));
    assert_eq!(from_both.provider, "openai");
}

#[tokio::test]
async fn reads_credentials_without_refreshing_oauth_when_requested() {
    let runtime = runtime_with_api_key_credentials("test-key").await;
    let storage = AuthStorage::<InMemoryAuthStorageBackend>::in_memory(&auth_data(&[(
        "openai",
        serde_json::json!({ "type": "api_key", "key": "test-key" }),
    )]));
    let credential = get_provider_credential(
        "openai",
        &runtime,
        &storage,
        AuthCheckOptions { refresh: false },
    )
    .await
    .expect("credential");
    assert_eq!(credential.as_deref(), Some("test-key"));
}

#[tokio::test]
async fn reports_an_unknown_provider_as_not_ready() {
    let runtime = runtime_with_api_key_credentials("test-key").await;
    let result = check_provider(&runtime, Some("not-installed"), None).await;
    assert!(matches!(result.status, AuthCheckStatus::NotReady));
    assert_eq!(result.provider, "not-installed");
    assert!(
        result
            .reason
            .is_some_and(|reason| reason.as_str() == "provider_not_found")
    );
}

#[tokio::test]
async fn does_not_treat_an_unresolved_stored_environment_reference_as_configured() {
    // The unresolved reference restates onto a registered provider whose
    // config template names a variable no ambient environment carries (the
    // openai fixture would ride ambient OPENAI_API_KEY — the pi-ai audit's
    // scrub finding), so the composer's own template check decides.
    let temp = tempfile::tempdir().expect("tempdir");
    let auth_path = temp.path().join("auth.json");
    std::fs::write(
        &auth_path,
        serde_json::json!({ "unresolved-ref": { "type": "api_key", "key": "$MISSING_AUTH_CHECK_KEY" } }).to_string(),
    )
    .expect("auth file");
    let read_only = pi_coding_agent::auth_storage::ReadOnlyAuthStorage::new(
        auth_path.to_string_lossy().as_ref(),
    )
    .expect("read-only store");
    let runtime = runtime_with_credentials(Arc::new(read_only)).await;
    register_api_key_provider(
        &runtime,
        "unresolved-ref",
        "Unresolved Reference",
        "$MISSING_AUTH_CHECK_KEY",
    );

    let result = check_provider(&runtime, Some("unresolved-ref"), None).await;
    assert!(
        matches!(result.status, AuthCheckStatus::NotReady),
        "the unresolved reference reads unconfigured, got {result:?}"
    );
    assert!(
        result
            .reason
            .is_some_and(|reason| reason.as_str() == "credentials_not_configured")
    );
    assert!(
        !temp.path().join("agent").exists(),
        "no parent directory materializes"
    );
}

#[tokio::test]
async fn reports_malformed_auth_state_as_invalid() {
    let temp = tempfile::tempdir().expect("tempdir");
    let auth_path = temp.path().join("auth.json");
    std::fs::write(&auth_path, "{invalid-json").expect("auth file");
    let read_only = pi_coding_agent::auth_storage::ReadOnlyAuthStorage::new(
        auth_path.to_string_lossy().as_ref(),
    )
    .expect("read-only store");
    let runtime = runtime_with_credentials(Arc::new(read_only)).await;

    let result = check_provider(&runtime, Some("openai"), None).await;
    assert!(matches!(result.status, AuthCheckStatus::Invalid));
    assert!(
        result
            .reason
            .is_some_and(|reason| reason.as_str() == "invalid_state")
    );
}

#[tokio::test]
async fn does_not_create_an_auth_file_or_its_parent_directory() {
    // The not-ready provider restates onto the registered template fixture
    // (no ambient env fallback rides it).
    let temp = tempfile::tempdir().expect("tempdir");
    let auth_path = temp.path().join("agent/auth.json");
    let read_only = pi_coding_agent::auth_storage::ReadOnlyAuthStorage::new(
        auth_path.to_string_lossy().as_ref(),
    )
    .expect("read-only store");
    let runtime = runtime_with_credentials(Arc::new(read_only)).await;
    register_api_key_provider(
        &runtime,
        "absent-ref",
        "Absent Reference",
        "$MISSING_AUTH_CHECK_KEY",
    );

    let result = check_provider(&runtime, Some("absent-ref"), None).await;
    assert!(
        matches!(result.status, AuthCheckStatus::NotReady),
        "the absent reference reads unconfigured, got {result:?}"
    );
    assert!(
        result
            .reason
            .is_some_and(|reason| reason.as_str() == "credentials_not_configured")
    );
    assert!(!auth_path.exists());
    assert!(!temp.path().join("agent").exists());
}

#[test]
fn accepts_optional_json_credential_output_and_no_refresh() {
    let check = parse_auth_command(&[
        "auth".to_string(),
        "check".to_string(),
        "--provider".to_string(),
        "openai".to_string(),
    ])
    .expect("parses")
    .expect("auth");
    assert_eq!(check.kind, AuthCommandKind::Check);
    assert_eq!(
        check.args,
        vec!["--provider".to_string(), "openai".to_string()]
    );
    assert!(!check.json && !check.credentials && !check.no_refresh);

    let flagged = parse_auth_command(&[
        "auth".to_string(),
        "check".to_string(),
        "--json".to_string(),
        "--credentials".to_string(),
        "--no-refresh".to_string(),
        "--provider".to_string(),
        "openai".to_string(),
    ])
    .expect("parses")
    .expect("auth");
    assert!(flagged.json && flagged.credentials && flagged.no_refresh);
}

#[tokio::test]
async fn creates_an_auth_check_runtime_without_catalog_storage() {
    let storage = AuthStorage::<InMemoryAuthStorageBackend>::in_memory(&auth_data(&[]));
    let runtime = create_auth_check_model_runtime(Arc::new(storage))
        .await
        .expect("runtime creates");
    assert!(
        runtime.get_provider("openai").is_some(),
        "the builtin providers register"
    );
}

// =============================================================================
// credential print, upstream's test/credential-print.test.ts
// =============================================================================

#[tokio::test]
async fn prints_a_resolved_api_key() {
    let runtime = runtime_with_api_key_credentials("test-api-key").await;
    let resolved = resolve_print(
        &runtime,
        Some("openai"),
        None,
        CredentialPrintKind::ApiKey,
        None,
    )
    .await
    .expect("resolves");
    assert_eq!(resolved, "test-api-key");
}

#[tokio::test]
async fn refreshes_an_expired_oauth_token_before_printing_it() {
    let (runtime, spy, storage) = runtime_with_expired_oauth().await;
    let resolved = resolve_print(
        &runtime,
        Some("oauth-provider"),
        None,
        CredentialPrintKind::BearerToken,
        None,
    )
    .await
    .expect("resolves");
    assert_eq!(resolved, "fresh-token");
    assert_eq!(refresh_count(&spy), 1, "one refresh");
    let stored = storage
        .read("oauth-provider", None)
        .await
        .expect("read")
        .expect("credential");
    match stored {
        Credential::OAuth(oauth) => assert_eq!(oauth.access, "fresh-token"),
        Credential::ApiKey(..) => panic!("expected the oauth credential"),
    }
}

#[tokio::test]
async fn parses_credential_commands_and_rejects_invalid_arguments() {
    let (runtime, _spy, _storage) = runtime_with_expired_oauth().await;

    let print = parse_auth_command(&[
        "auth".to_string(),
        "print-api-key".to_string(),
        "--provider".to_string(),
        "openai".to_string(),
    ])
    .expect("parses")
    .expect("auth");
    assert_eq!(print.kind, AuthCommandKind::ApiKey);

    let bare = parse_auth_command(&["auth".to_string(), "print-bearer-token".to_string()])
        .expect("parses")
        .expect("auth");
    assert_eq!(bare.kind, AuthCommandKind::BearerToken);

    let with_expiry = parse_auth_command(&[
        "auth".to_string(),
        "print-bearer-token".to_string(),
        "--min-expiry".to_string(),
        "30m".to_string(),
    ])
    .expect("parses")
    .expect("auth");
    assert_eq!(with_expiry.min_expiry_ms, Some(30 * 60_000));

    assert_eq!(
        parse_auth_command(&[
            "auth".to_string(),
            "print-api-key".to_string(),
            "--min-expiry".to_string(),
            "30m".to_string(),
        ])
        .expect_err("min-expiry on print-api-key"),
        AuthCommandError("--min-expiry is only supported by print-bearer-token".to_string())
    );

    assert!(is_auth_command_help(&[
        "auth".to_string(),
        "--help".to_string()
    ]));
    assert!(is_auth_command_help(&[
        "auth".to_string(),
        "print-api-key".to_string(),
        "--help".to_string()
    ]));
    assert!(is_auth_command_help(&[
        "auth".to_string(),
        "print-bearer-token".to_string(),
        "-h".to_string()
    ]));
    assert!(is_auth_command_help(&[
        "auth".to_string(),
        "check".to_string(),
        "--help".to_string()
    ]));
    assert!(parse_auth_command(&["auth".to_string(), "unknown".to_string()]).is_err());

    let error = resolve_print(&runtime, None, None, CredentialPrintKind::ApiKey, None)
        .await
        .expect_err("no target");
    assert!(
        error
            .0
            .contains("requires --provider <provider> or --model <model>"),
        "{error}"
    );
}

#[tokio::test]
async fn rejects_a_credential_print_across_the_credential_types() {
    let (runtime, _spy, _storage) = runtime_with_expired_oauth().await;
    let error = resolve_print(
        &runtime,
        Some("oauth-provider"),
        None,
        CredentialPrintKind::ApiKey,
        None,
    )
    .await
    .expect_err("type mismatch");
    assert!(error.0.contains("configured with OAuth"), "{error}");
}

#[test]
fn reports_unknown_auth_options_through_the_validator() {
    let flagged = AuthCommandArgs {
        unknown_flags: indexmap::IndexMap::from([(
            "credentails".to_string(),
            UnknownFlagValue::Flag,
        )]),
        ..AuthCommandArgs::default()
    };
    let error =
        validate_auth_command_args(&flagged, AuthCommandKind::Check).expect_err("unknown flag");
    assert_eq!(
        error,
        AuthCommandError("Unknown option --credentails for \"auth check\".".to_string())
    );
}

// =============================================================================
// auth command surface, upstream's src/cli/auth-command.ts contracts
// =============================================================================

#[test]
fn renders_command_names_usage_and_error_messages() {
    // The wire tags, upstream's union members.
    assert_eq!(AuthCommandKind::Check.as_str(), "check");
    assert_eq!(AuthCommandKind::ApiKey.as_str(), "api_key");
    assert_eq!(AuthCommandKind::BearerToken.as_str(), "bearer_token");
    for (kind, name, tag) in [
        (AuthCommandKind::Check, "auth check", "check"),
        (AuthCommandKind::ApiKey, "auth print-api-key", "api_key"),
        (
            AuthCommandKind::BearerToken,
            "auth print-bearer-token",
            "bearer_token",
        ),
    ] {
        assert_eq!(get_auth_command_name(kind), name);
        assert_eq!(format!("{kind}"), tag);
    }

    assert_eq!(
        get_auth_command_usage(AuthCommandKind::Check),
        "pi auth check --provider <provider> [--json] [--credentials] [--no-refresh]"
    );
    assert_eq!(
        get_auth_command_usage(AuthCommandKind::ApiKey),
        "pi auth print-api-key --provider <provider> [--model <model>]"
    );
    assert_eq!(
        get_auth_command_usage(AuthCommandKind::BearerToken),
        "pi auth print-bearer-token --provider <provider> [--model <model>] [--min-expiry <duration>]"
    );

    // The error's Display, upstream's `error.message` reads.
    let error = AuthCommandError("boom".to_string());
    assert_eq!(format!("{error}"), "boom");
    assert_eq!(error.to_string(), "boom");
}

#[test]
fn prints_the_auth_help_block() {
    print_auth_command_help();
}

#[test]
fn falls_through_for_non_auth_arguments_and_rejects_check_flags_on_prints() {
    // Arguments that do not open with `auth` fall through to the full
    // parse, upstream's `Ok(null)`.
    let fall_through = parse_auth_command(&["install".to_string(), "x".to_string()]);
    assert!(fall_through.expect("parses").is_none());
    // The check-only flags stay exclusive to `auth check`, upstream's
    // flag-scoped errors.
    for flag in ["--json", "--credentials", "--no-refresh"] {
        let error = parse_auth_command(&[
            "auth".to_string(),
            "print-api-key".to_string(),
            flag.to_string(),
        ])
        .expect_err(flag);
        assert_eq!(
            error,
            AuthCommandError(format!("{flag} is only supported by auth check"))
        );
    }
}

#[test]
fn parses_min_expiry_durations_and_rejects_malformed_ones() {
    let parsed = |value: &str| {
        parse_auth_command(&[
            "auth".to_string(),
            "print-bearer-token".to_string(),
            "--min-expiry".to_string(),
            value.to_string(),
        ])
    };
    let parse = |value: &str| parsed(value).expect("parses").expect("auth").min_expiry_ms;
    assert_eq!(parse("500ms"), Some(500));
    assert_eq!(parse("90s"), Some(90_000));
    assert_eq!(parse("90S"), Some(90_000));
    assert_eq!(parse("2h"), Some(7_200_000));
    // The unit grammar is digits then a unit: a sign, a fraction, or a bare
    // number all fail the command, upstream's anchored pattern.
    for value in ["-5m", "1.5h", "30"] {
        assert_eq!(
            parsed(value).expect_err(value),
            AuthCommandError("--min-expiry must use a duration such as 30m or 1h".to_string())
        );
    }
    // A magnitude whose multiplication overflows fails the parse, upstream's
    // `Number.MAX_SAFE_INTEGER`-shaped overflow.
    assert_eq!(
        parsed("9223372036854775807h").expect_err("overflow"),
        AuthCommandError("--min-expiry must use a duration such as 30m or 1h".to_string())
    );
}

#[test]
fn rejects_prompt_and_file_inputs_and_whitespace_targets() {
    let prompted = AuthCommandArgs {
        messages: vec!["hello".to_string()],
        ..AuthCommandArgs::default()
    };
    assert_eq!(
        validate_auth_command_args(&prompted, AuthCommandKind::Check).expect_err("prompt"),
        AuthCommandError("Auth commands only accept --provider and --model".to_string())
    );
    let filed = AuthCommandArgs {
        file_args: vec!["notes.md".to_string()],
        ..AuthCommandArgs::default()
    };
    assert_eq!(
        validate_auth_command_args(&filed, AuthCommandKind::Check).expect_err("file"),
        AuthCommandError("Auth commands only accept --provider and --model".to_string())
    );
    // Whitespace-only targets trim to absent, upstream's `trim() &&` guard.
    let blank_provider = AuthCommandArgs {
        provider: Some("   ".to_string()),
        ..AuthCommandArgs::default()
    };
    assert_eq!(
        validate_auth_command_args(&blank_provider, AuthCommandKind::Check).expect_err("blank"),
        AuthCommandError(
            "Auth checks require --provider <provider> or --model <model>".to_string()
        )
    );
    let blank_model = AuthCommandArgs {
        model: Some("  ".to_string()),
        ..AuthCommandArgs::default()
    };
    assert_eq!(
        validate_auth_command_args(&blank_model, AuthCommandKind::ApiKey).expect_err("blank"),
        AuthCommandError(
            "Credential printing requires --provider <provider> or --model <model>".to_string()
        )
    );
    // The unknown-option message names the command that rejected it.
    for (kind, name) in [
        (AuthCommandKind::Check, "auth check"),
        (AuthCommandKind::ApiKey, "auth print-api-key"),
        (AuthCommandKind::BearerToken, "auth print-bearer-token"),
    ] {
        let flagged = AuthCommandArgs {
            unknown_flags: indexmap::IndexMap::from([(
                "credentails".to_string(),
                UnknownFlagValue::Value("x".to_string()),
            )]),
            ..AuthCommandArgs::default()
        };
        assert_eq!(
            validate_auth_command_args(&flagged, kind).expect_err("flagged"),
            AuthCommandError(format!("Unknown option --credentails for \"{name}\"."))
        );
    }
}

#[test]
fn extracts_credentials_from_header_shapes() {
    // A header block without an authorization header carries no credential.
    let other_headers = AuthResult {
        auth: ModelAuth {
            api_key: None,
            headers: Some(BTreeMap::from([(
                "X-Other".to_string(),
                Some("v".to_string()),
            )])),
            ..ModelAuth::default()
        },
        ..AuthResult::default()
    };
    assert_eq!(get_auth_credential(Some(&other_headers)), None);
    // An authorization header whose value carries no `Bearer ` prefix
    // carries no credential, and an empty header slot neither.
    let non_bearer = AuthResult {
        auth: ModelAuth {
            api_key: None,
            headers: Some(BTreeMap::from([
                ("AUTHORIZATION".to_string(), Some("Basic abc".to_string())),
                ("authorization".to_string(), None),
            ])),
            ..ModelAuth::default()
        },
        ..AuthResult::default()
    };
    assert_eq!(get_auth_credential(Some(&non_bearer)), None);
    // The prefix match is case-insensitive, upstream's `iu` flag.
    let mixed_case = AuthResult {
        auth: ModelAuth {
            api_key: None,
            headers: Some(BTreeMap::from([(
                "authorization".to_string(),
                Some("bEaReR mixed-token".to_string()),
            )])),
            ..ModelAuth::default()
        },
        ..AuthResult::default()
    };
    assert_eq!(
        get_auth_credential(Some(&mixed_case)).as_deref(),
        Some("mixed-token")
    );
}

// =============================================================================
// auth check failure paths, upstream's test/auth-check.test.ts restatements
// =============================================================================

/// The store whose reads or lists fail on demand, the restated spy store
/// the availability tests drive.
struct FailingStore {
    base: Arc<InMemoryCredentialStore>,
    fail_read: AtomicBool,
    fail_list: AtomicBool,
}

impl CredentialStore for FailingStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        let provider_id = provider_id.to_owned();
        let base = Arc::clone(&self.base);
        let fail = self.fail_read.load(Ordering::SeqCst);
        Box::pin(async move {
            if fail {
                return Err(AuthError::from(std::io::Error::other(format!(
                    "read failed for {provider_id}"
                ))));
            }
            base.read(&provider_id, options).await
        })
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        let base = Arc::clone(&self.base);
        let fail = self.fail_list.load(Ordering::SeqCst);
        Box::pin(async move {
            if fail {
                return Err(AuthError::from(std::io::Error::other("list failed")));
            }
            base.list(options).await
        })
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

/// The runtime over a failing store, with the typed handle back so a test
/// flips the failure flags after the create.
async fn failing_store_runtime() -> (ModelRuntime, Arc<FailingStore>) {
    let store = Arc::new(FailingStore {
        base: Arc::new(InMemoryCredentialStore::default()),
        fail_read: AtomicBool::new(false),
        fail_list: AtomicBool::new(false),
    });
    let credentials: Arc<dyn CredentialStore> = store.clone();
    let runtime = runtime_with_credentials(credentials).await;
    (runtime, store)
}

#[test]
fn renders_the_check_status_and_reason_wire_tags() {
    assert_eq!(AuthCheckStatus::Ready.as_str(), "ready");
    assert_eq!(AuthCheckStatus::NotReady.as_str(), "not_ready");
    assert_eq!(AuthCheckStatus::Invalid.as_str(), "invalid");
    assert_eq!(
        AuthCheckReason::ProviderNotFound.as_str(),
        "provider_not_found"
    );
    assert_eq!(
        AuthCheckReason::CredentialsNotConfigured.as_str(),
        "credentials_not_configured"
    );
    assert_eq!(
        AuthCheckReason::CredentialNotAvailable.as_str(),
        "credential_not_available"
    );
    assert_eq!(AuthCheckReason::InvalidState.as_str(), "invalid_state");
}

#[tokio::test]
async fn reads_the_stored_oauth_access_token_without_refreshing() {
    let (runtime, spy, storage) = runtime_with_expired_oauth().await;
    let credential = get_provider_credential(
        "oauth-provider",
        &runtime,
        storage.as_ref(),
        AuthCheckOptions { refresh: false },
    )
    .await
    .expect("credential");
    assert_eq!(credential.as_deref(), Some("old-token"));
    assert_eq!(refresh_count(&spy), 0, "no refresh");
}

#[tokio::test]
async fn refreshes_oauth_when_the_credential_read_requests_it() {
    let (runtime, spy, storage) = runtime_with_expired_oauth().await;
    let credential = get_provider_credential(
        "oauth-provider",
        &runtime,
        storage.as_ref(),
        AuthCheckOptions { refresh: true },
    )
    .await
    .expect("credential");
    assert_eq!(credential.as_deref(), Some("fresh-token"));
    assert_eq!(refresh_count(&spy), 1, "one refresh");
}

#[tokio::test]
async fn propagates_the_credential_store_read_failure() {
    let (runtime, store) = failing_store_runtime().await;
    store.fail_read.store(true, Ordering::SeqCst);
    // The check's own read propagates, upstream's rejecting `readCredential`.
    let error = get_provider_credential(
        "openai",
        &runtime,
        store.as_ref(),
        AuthCheckOptions { refresh: false },
    )
    .await
    .expect_err("read failure");
    assert!(
        error.to_string().contains("read failed for openai"),
        "{error}"
    );
}

#[tokio::test]
async fn propagates_the_auth_resolution_failure_through_the_credential_read() {
    let (runtime, store) = failing_store_runtime().await;
    store.fail_read.store(true, Ordering::SeqCst);
    // With the read intact, the request-auth resolution fails on its own
    // store read and propagates, upstream's rejecting `getAuth`.
    let empty = AuthStorage::<InMemoryAuthStorageBackend>::in_memory(&auth_data(&[]));
    let error = get_provider_credential(
        "openai",
        &runtime,
        &empty,
        AuthCheckOptions { refresh: true },
    )
    .await
    .expect_err("resolution failure");
    assert!(
        error.to_string().contains("read failed for openai"),
        "{error}"
    );
}

#[tokio::test]
async fn reports_a_failing_credential_store_as_invalid() {
    let (runtime, store) = failing_store_runtime().await;
    store.fail_read.store(true, Ordering::SeqCst);
    let result = check_provider(&runtime, Some("openai"), None).await;
    assert!(
        matches!(result.status, AuthCheckStatus::Invalid),
        "{result:?}"
    );
    assert!(
        result
            .reason
            .is_some_and(|reason| reason.as_str() == "invalid_state")
    );
}

#[tokio::test]
async fn reports_a_failing_oauth_refresh_as_invalid() {
    let (runtime, spy, _storage) = oauth_rig(0, "old-token", RefreshOutcome::Fails).await;
    let result = check_provider_auth(
        &args_view(Some("oauth-provider"), None),
        &runtime,
        AuthCheckOptions { refresh: true },
    )
    .await
    .expect("check");
    assert!(
        matches!(result.status, AuthCheckStatus::Invalid),
        "{result:?}"
    );
    assert!(
        result
            .reason
            .is_some_and(|reason| reason.as_str() == "invalid_state")
    );
    assert_eq!(refresh_count(&spy), 1, "the refresh ran");
}

#[tokio::test]
async fn reports_a_recorded_availability_failure_as_invalid() {
    // The error surface the availability pass records reads as invalid
    // state before any credential check, upstream's getError guard.
    let (runtime, store) = failing_store_runtime().await;
    store.fail_read.store(true, Ordering::SeqCst);
    let failure = runtime
        .get_available(Some("openai"), None)
        .await
        .expect_err("the failing read surfaces");
    assert!(
        failure.to_string().contains("read failed for openai"),
        "{failure}"
    );

    let result = check_provider(&runtime, Some("openai"), None).await;
    assert!(
        matches!(result.status, AuthCheckStatus::Invalid),
        "{result:?}"
    );
    assert!(
        result
            .reason
            .is_some_and(|reason| reason.as_str() == "invalid_state")
    );
}

#[tokio::test]
async fn reports_an_unresolvable_model_target_for_the_check() {
    let runtime = runtime_with_credentials(Arc::new(
        AuthStorage::<InMemoryAuthStorageBackend>::in_memory(&auth_data(&[])),
    ))
    .await;
    register_api_key_provider(
        &runtime,
        "nomodels-provider",
        "Nomodels Provider",
        "sk-nomodels",
    );

    let error = check_provider_auth(
        &args_view(Some("nomodels-provider"), Some("whatever")),
        &runtime,
        AuthCheckOptions::default(),
    )
    .await
    .expect_err("unresolvable");
    assert!(
        error.0.contains("Unknown provider \"nomodels-provider\""),
        "{error}"
    );
}

/// The store whose reads still show an OAuth credential the storage no
/// longer holds — the "logged out meanwhile" state between the check's
/// read and the resolution's locked re-read.
struct VanishingOAuthStore {
    stale: OAuthCredentials,
    inner: Arc<InMemoryCredentialStore>,
}

impl CredentialStore for VanishingOAuthStore {
    fn read<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        let stale = self.stale.clone();
        Box::pin(async move { Ok(Some(Credential::OAuth(stale))) })
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        Box::pin(self.inner.list(options))
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: CredentialModifyFn,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        Box::pin(self.inner.modify(provider_id, f, options))
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        Box::pin(self.inner.delete(provider_id, options))
    }
}

#[tokio::test]
async fn reads_not_ready_when_the_credential_vanishes_between_reads() {
    // The check's read still sees an OAuth credential the storage has
    // dropped; the refresh resolution's locked re-read finds nothing and
    // reports not-ready, upstream's "logged out meanwhile" return.
    let stale = OAuthCredentials {
        access: "stale-token".to_string(),
        refresh: "refresh-token".to_string(),
        expires: 0,
        extra: BTreeMap::new(),
    };
    let credentials: Arc<dyn CredentialStore> = Arc::new(VanishingOAuthStore {
        stale: stale.clone(),
        inner: Arc::new(InMemoryCredentialStore::default()),
    });
    let runtime = runtime_with_credentials(credentials).await;
    register_oauth_provider(
        &runtime,
        Arc::new(|_credentials, _signal| {
            let error: Box<dyn std::error::Error + Send + Sync> =
                Box::new(std::io::Error::other("no refresh should run"));
            Box::pin(async move { Err(error) })
        }),
    );

    let result = check_provider_auth(
        &args_view(Some("oauth-provider"), None),
        &runtime,
        AuthCheckOptions { refresh: true },
    )
    .await
    .expect("check");
    assert!(
        matches!(result.status, AuthCheckStatus::NotReady),
        "{result:?}"
    );
    assert!(
        result
            .reason
            .is_some_and(|reason| reason.as_str() == "credentials_not_configured")
    );
    assert_eq!(stale.access, "stale-token");
}

// =============================================================================
// credential print resolution paths, upstream's src/cli/credential-print.ts
// =============================================================================

/// The one-model registration input the print rigs share, the restated
/// `test_model("...")` fixture.
fn print_model(id: &str) -> pi_coding_agent::provider_composer::ProviderModelInput {
    use pi_ai::types::{Modality, ModelCost};
    pi_coding_agent::provider_composer::ProviderModelInput {
        id: id.to_string(),
        name: id.to_string(),
        api: None,
        base_url: None,
        reasoning: false,
        thinking_level_map: None,
        input: vec![Modality::Text],
        cost: ModelCost::default(),
        context_window: 10_000,
        max_tokens: 1_000,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The runtime with two api-key providers registered over one shared model
/// id, each seeded with a stored key, upstream's two-credential fixture.
async fn shared_model_runtime() -> ModelRuntime {
    let storage = AuthStorage::<InMemoryAuthStorageBackend>::in_memory(&auth_data(&[
        (
            "alpha",
            serde_json::json!({ "type": "api_key", "key": "sk-alpha" }),
        ),
        (
            "beta",
            serde_json::json!({ "type": "api_key", "key": "sk-beta" }),
        ),
    ]));
    let runtime = runtime_with_credentials(Arc::new(storage)).await;
    for (provider, key) in [("alpha", "sk-alpha"), ("beta", "sk-beta")] {
        runtime
            .register_provider(
                provider,
                ProviderConfigInput {
                    name: Some(provider.to_string()),
                    base_url: Some("https://example.test/v1".to_string()),
                    api: Some(Api::from("openai-completions")),
                    api_key: Some(key.to_string()),
                    models: Some(vec![print_model("shared-model")]),
                    ..ProviderConfigInput::default()
                },
            )
            .expect("the registration validates");
    }
    runtime
}

/// The runtime with one api-key provider registered and an empty store, the
/// shape whose credential comes from the registration alone.
async fn static_key_runtime() -> ModelRuntime {
    let runtime = runtime_with_credentials(Arc::new(
        AuthStorage::<InMemoryAuthStorageBackend>::in_memory(&auth_data(&[])),
    ))
    .await;
    runtime
        .register_provider(
            "solo-provider",
            ProviderConfigInput {
                name: Some("Solo Provider".to_string()),
                base_url: Some("https://example.test/v1".to_string()),
                api: Some(Api::from("openai-completions")),
                api_key: Some("sk-static".to_string()),
                models: Some(vec![print_model("solo-model")]),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the registration validates");
    runtime
}

#[tokio::test]
async fn reports_an_unknown_provider() {
    let runtime = static_key_runtime().await;
    let error = resolve_print(
        &runtime,
        Some("not-registered"),
        None,
        CredentialPrintKind::ApiKey,
        None,
    )
    .await
    .expect_err("unknown provider");
    assert_eq!(
        error.0,
        "Unknown provider \"not-registered\". Use --list-models to see available providers."
    );
}

#[tokio::test]
async fn prints_through_the_requested_provider_model() {
    // The provider's own model scan falls back to a custom model id for an
    // unknown id; the print resolves the auth through that model.
    let runtime = shared_model_runtime().await;
    let resolved = resolve_print(
        &runtime,
        Some("alpha"),
        Some("no-such-model-xyz"),
        CredentialPrintKind::ApiKey,
        None,
    )
    .await
    .expect("resolves");
    assert_eq!(resolved, "sk-alpha");
}

#[tokio::test]
async fn reports_an_unresolvable_model_for_the_requested_provider() {
    let runtime = runtime_with_credentials(Arc::new(
        AuthStorage::<InMemoryAuthStorageBackend>::in_memory(&auth_data(&[])),
    ))
    .await;
    register_api_key_provider(
        &runtime,
        "nomodels-provider",
        "Nomodels Provider",
        "sk-nomodels",
    );
    let error = resolve_print(
        &runtime,
        Some("nomodels-provider"),
        Some("whatever"),
        CredentialPrintKind::ApiKey,
        None,
    )
    .await
    .expect_err("unresolvable");
    assert!(
        error.0.contains("Unknown provider \"nomodels-provider\""),
        "{error}"
    );
}

#[tokio::test]
async fn prints_an_api_key_resolved_from_the_registration_without_a_stored_credential() {
    let runtime = static_key_runtime().await;
    let resolved = resolve_print(
        &runtime,
        Some("solo-provider"),
        None,
        CredentialPrintKind::ApiKey,
        None,
    )
    .await
    .expect("resolves");
    assert_eq!(resolved, "sk-static");
}

#[tokio::test]
async fn prints_a_bearer_token_from_a_valid_credential_without_refreshing() {
    let (runtime, spy, _storage) = oauth_rig(
        now_ms() + 60 * 60 * 1000,
        "valid-token",
        RefreshOutcome::Fresh,
    )
    .await;
    let resolved = resolve_print(
        &runtime,
        Some("oauth-provider"),
        None,
        CredentialPrintKind::BearerToken,
        None,
    )
    .await
    .expect("resolves");
    assert_eq!(resolved, "valid-token");
    assert_eq!(
        refresh_count(&spy),
        0,
        "the default min-expiry stays inside the token's life"
    );
}

#[tokio::test]
async fn refreshes_when_min_expiry_demands_a_fresher_token() {
    let (runtime, spy, _storage) = oauth_rig(
        now_ms() + 10 * 60 * 1000,
        "soon-token",
        RefreshOutcome::Fresh,
    )
    .await;
    let resolved = resolve_print(
        &runtime,
        Some("oauth-provider"),
        None,
        CredentialPrintKind::BearerToken,
        Some(30 * 60_000),
    )
    .await
    .expect("resolves");
    assert_eq!(resolved, "fresh-token");
    assert_eq!(refresh_count(&spy), 1, "one refresh");
}

#[tokio::test]
async fn rejects_when_the_refreshed_token_misses_the_min_expiry() {
    let (runtime, _spy, _storage) = oauth_rig(0, "old-token", RefreshOutcome::Soon).await;
    let error = resolve_print(
        &runtime,
        Some("oauth-provider"),
        None,
        CredentialPrintKind::BearerToken,
        Some(30 * 60_000),
    )
    .await
    .expect_err("too soon");
    assert!(error.0.contains("expires too soon"), "{error}");
}

#[tokio::test]
async fn rejects_a_bearer_print_for_an_api_key_provider() {
    let runtime = runtime_with_api_key_credentials("test-api-key").await;
    let error = resolve_print(
        &runtime,
        Some("openai"),
        None,
        CredentialPrintKind::BearerToken,
        None,
    )
    .await
    .expect_err("type mismatch");
    assert_eq!(
        error.0,
        "Provider \"openai\" is not configured with an OAuth bearer token"
    );
}

#[tokio::test]
async fn reports_no_usable_credential_for_a_provider_without_auth() {
    // A provider with no auth configuration resolves nothing; each print
    // kind reports its own missing credential.
    let runtime = runtime_with_credentials(Arc::new(
        AuthStorage::<InMemoryAuthStorageBackend>::in_memory(&auth_data(&[])),
    ))
    .await;
    runtime
        .register_provider(
            "authless-provider",
            ProviderConfigInput {
                name: Some("Authless Provider".to_string()),
                api: Some(Api::from("openai-completions")),
                ..ProviderConfigInput::default()
            },
        )
        .expect("the registration validates");

    let error = resolve_print(
        &runtime,
        Some("authless-provider"),
        None,
        CredentialPrintKind::BearerToken,
        None,
    )
    .await
    .expect_err("no bearer");
    assert_eq!(error.0, "No usable OAuth bearer token is configured");

    let error = resolve_print(
        &runtime,
        Some("authless-provider"),
        None,
        CredentialPrintKind::ApiKey,
        None,
    )
    .await
    .expect_err("no key");
    assert_eq!(error.0, "No usable API key is configured");
}

#[tokio::test]
async fn propagates_the_unresolvable_api_key_template() {
    // A template key no ambient environment resolves fails the auth
    // resolution itself, upstream's propagating rejection.
    let temp = tempfile::tempdir().expect("tempdir");
    let auth_path = temp.path().join("auth.json");
    let read_only = pi_coding_agent::auth_storage::ReadOnlyAuthStorage::new(
        auth_path.to_string_lossy().as_ref(),
    )
    .expect("read-only store");
    let runtime = runtime_with_credentials(Arc::new(read_only)).await;
    register_api_key_provider(
        &runtime,
        "tmpl-provider",
        "Template Provider",
        "$MISSING_CREDENTIAL_PRINT_KEY",
    );

    let error = resolve_print(
        &runtime,
        Some("tmpl-provider"),
        None,
        CredentialPrintKind::ApiKey,
        None,
    )
    .await
    .expect_err("unresolved template");
    assert!(
        error
            .0
            .contains("API key auth failed for provider tmpl-provider"),
        "{error}"
    );
}

#[tokio::test]
async fn reports_a_model_not_found_without_a_provider() {
    let runtime = shared_model_runtime().await;
    let error = resolve_print(
        &runtime,
        None,
        Some("no-such-model-xyz"),
        CredentialPrintKind::ApiKey,
        None,
    )
    .await
    .expect_err("not found");
    assert_eq!(
        error.0,
        "Model \"no-such-model-xyz\" not found. Use --list-models to see available models."
    );
}

#[tokio::test]
async fn reports_multiple_matched_providers() {
    let runtime = shared_model_runtime().await;
    let error = resolve_print(
        &runtime,
        None,
        Some("shared-model"),
        CredentialPrintKind::ApiKey,
        None,
    )
    .await
    .expect_err("multiple");
    assert!(
        error
            .0
            .starts_with("Multiple configured providers matched ("),
        "{error}"
    );
    assert!(
        error.0.contains("alpha") && error.0.contains("beta"),
        "{error}"
    );
    assert!(error.0.ends_with("Specify --provider."), "{error}");
}

#[tokio::test]
async fn propagates_the_credential_list_failure() {
    let (runtime, store) = failing_store_runtime().await;
    store.fail_list.store(true, Ordering::SeqCst);
    let error = resolve_print(
        &runtime,
        Some("openai"),
        None,
        CredentialPrintKind::ApiKey,
        None,
    )
    .await
    .expect_err("list failure");
    assert_eq!(error.0, "Unable to list the configured credentials");
}

#[tokio::test]
async fn propagates_the_auth_resolution_failure() {
    let (runtime, store) = failing_store_runtime().await;
    store.fail_read.store(true, Ordering::SeqCst);
    let error = resolve_print(
        &runtime,
        Some("openai"),
        None,
        CredentialPrintKind::ApiKey,
        None,
    )
    .await
    .expect_err("read failure");
    assert!(
        error.0.contains("Credential store read failed for openai"),
        "{error}"
    );
}

#[allow(dead_code, reason = "the fixture path type pins the tempdir lifetime")]
fn keep_temp_alive(_path: PathBuf) {}

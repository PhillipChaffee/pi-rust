//! The environment API key discovery, from `test/env-api-keys.test.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream mutates `process.env` and restores it in `afterEach`; the port
//! drives [`find_env_keys`] and [`get_env_api_key`] with a `ProviderEnv`
//! overlay, which wins over the process environment. The two trailing cases
//! are Rust-native coverage for the ambient `<authenticated>` branches
//! upstream's file leaves to its provider-specific suites. The Adapter
//! cases pass an empty overlay only because the lookup consults the process
//! environment after it: the CI runners these suites run on set none of the
//! discovery variables.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::collections::BTreeMap;

use pi_ai::env_api_keys::{find_env_keys, get_env_api_key};
use pi_ai::types::ProviderEnv;

/// A fixture environment standing in for `process.env`.
fn env_fixture(pairs: &[(&str, &str)]) -> ProviderEnv {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect()
}

/// The discovery path over the fixture environment.
fn find_keys(env: &BTreeMap<String, String>, provider: &str) -> Option<Vec<String>> {
    find_env_keys(provider, Some(env))
}

/// The api-key lookup over the fixture environment.
fn env_api_key(env: &BTreeMap<String, String>, provider: &str) -> Option<String> {
    get_env_api_key(provider, Some(env))
}

#[test]
fn does_not_treat_generic_github_tokens_as_github_copilot_credentials() {
    let env = env_fixture(&[("GH_TOKEN", "gh-token"), ("GITHUB_TOKEN", "github-token")]);

    assert_eq!(find_keys(&env, "github-copilot"), None);
    assert_eq!(env_api_key(&env, "github-copilot"), None);
}

#[test]
fn resolves_github_copilot_credentials_from_copilot_github_token() {
    let env = env_fixture(&[
        ("COPILOT_GITHUB_TOKEN", "copilot-token"),
        ("GH_TOKEN", "gh-token"),
        ("GITHUB_TOKEN", "github-token"),
    ]);

    assert_eq!(
        find_keys(&env, "github-copilot"),
        Some(vec!["COPILOT_GITHUB_TOKEN".to_owned()])
    );
    assert_eq!(
        env_api_key(&env, "github-copilot"),
        Some("copilot-token".to_owned())
    );
}

#[test]
fn resolves_zai_china_coding_plan_credentials_from_zai_coding_cn_api_key() {
    let env = env_fixture(&[("ZAI_CODING_CN_API_KEY", "zai-coding-cn-token")]);

    assert_eq!(
        find_keys(&env, "zai-coding-cn"),
        Some(vec!["ZAI_CODING_CN_API_KEY".to_owned()])
    );
    assert_eq!(
        env_api_key(&env, "zai-coding-cn"),
        Some("zai-coding-cn-token".to_owned())
    );
}

#[test]
fn reports_anthropic_auth_token_but_preserves_oauth_token_api_key_lookup() {
    let env = env_fixture(&[
        ("ANTHROPIC_AUTH_TOKEN", "auth-token"),
        ("ANTHROPIC_OAUTH_TOKEN", "oauth-token"),
        ("ANTHROPIC_API_KEY", "api-key"),
    ]);

    assert_eq!(
        find_keys(&env, "anthropic"),
        Some(vec![
            "ANTHROPIC_AUTH_TOKEN".to_owned(),
            "ANTHROPIC_OAUTH_TOKEN".to_owned(),
            "ANTHROPIC_API_KEY".to_owned(),
        ])
    );
    assert_eq!(
        env_api_key(&env, "anthropic"),
        Some("oauth-token".to_owned())
    );
}

#[test]
fn does_not_return_anthropic_auth_token_as_an_api_key() {
    let env = env_fixture(&[("ANTHROPIC_AUTH_TOKEN", "auth-token")]);

    assert_eq!(
        find_keys(&env, "anthropic"),
        Some(vec!["ANTHROPIC_AUTH_TOKEN".to_owned()])
    );
    assert_eq!(env_api_key(&env, "anthropic"), None);
}

#[test]
fn preserves_anthropic_oauth_token_as_an_api_key() {
    let env = env_fixture(&[("ANTHROPIC_OAUTH_TOKEN", "oauth-token")]);

    assert_eq!(
        find_keys(&env, "anthropic"),
        Some(vec!["ANTHROPIC_OAUTH_TOKEN".to_owned()])
    );
    assert_eq!(
        env_api_key(&env, "anthropic"),
        Some("oauth-token".to_owned())
    );
}

#[test]
fn falls_back_to_anthropic_api_key_for_api_key_lookup() {
    let env = env_fixture(&[("ANTHROPIC_API_KEY", "api-key")]);

    assert_eq!(env_api_key(&env, "anthropic"), Some("api-key".to_owned()));
}

/// Rust-native coverage case for the ambient Vertex branch: an explicit
/// `GOOGLE_APPLICATION_CREDENTIALS` file plus project and location env vars
/// resolve to `<authenticated>` when no api key variable is configured. The
/// project rides `GCLOUD_PROJECT`, covering the fallback spelling of the
/// project check.
#[test]
fn google_vertex_authenticates_through_application_default_credentials() {
    let adc = std::env::temp_dir().join("pi-rust-vertex-adc-fixture.json");
    std::fs::write(&adc, b"{}").expect("the ADC fixture file writes");

    let env = env_fixture(&[
        ("GOOGLE_APPLICATION_CREDENTIALS", &adc.to_string_lossy()),
        ("GCLOUD_PROJECT", "pi-test-project"),
        ("GOOGLE_CLOUD_LOCATION", "us-central1"),
    ]);
    let key = env_api_key(&env, "google-vertex");
    let _ = std::fs::remove_file(&adc);

    assert_eq!(key, Some("<authenticated>".to_owned()));
}

/// Rust-native coverage case for the ambient Bedrock branch: the standard
/// IAM key pair resolves to `<authenticated>` without an api key variable.
#[test]
fn amazon_bedrock_authenticates_through_iam_keys() {
    let env = env_fixture(&[
        ("AWS_ACCESS_KEY_ID", "test-access-key"),
        ("AWS_SECRET_ACCESS_KEY", "test-secret-key"),
    ]);

    assert_eq!(
        env_api_key(&env, "amazon-bedrock"),
        Some("<authenticated>".to_owned())
    );
}

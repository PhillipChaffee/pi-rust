//! The credential-guard helpers and the `auth.json` resolver, ported from
//! `packages/ai/test/azure-utils.ts`, `bedrock-utils.ts`,
//! `cloudflare-utils.ts`, and `test/oauth.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. These helpers serve the
//! wire-API children's live-credential suites; this file pins their behavior
//! hermetically.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use std::collections::BTreeMap;

use common::auth_fixtures::{StubOAuthAuth, oauth_credentials, refresh_returning};
use common::{auth_guards, auth_json};
use pi_ai::auth::types::{ApiKeyCredential, Credential};

fn env_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let map: BTreeMap<String, String> = pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect();
    move |name| map.get(name).cloned()
}

#[test]
fn azure_guard_requires_the_key_and_an_endpoint() {
    assert!(!auth_guards::has_azure_openai_credentials(&env_from(&[])));
    assert!(!auth_guards::has_azure_openai_credentials(&env_from(&[(
        "AZURE_OPENAI_API_KEY",
        "k"
    )])));
    assert!(auth_guards::has_azure_openai_credentials(&env_from(&[
        ("AZURE_OPENAI_API_KEY", "k"),
        ("AZURE_OPENAI_BASE_URL", "https://example"),
    ])));
    assert!(auth_guards::has_azure_openai_credentials(&env_from(&[
        ("AZURE_OPENAI_API_KEY", "k"),
        ("AZURE_OPENAI_RESOURCE_NAME", "res"),
    ])));
}

#[test]
fn azure_deployment_map_resolves_model_ids() {
    assert_eq!(
        auth_guards::resolve_azure_deployment_name("gpt-5", &env_from(&[])),
        None
    );
    let mapped = env_from(&[("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "m1=d1, m2=d2")]);
    assert_eq!(
        auth_guards::resolve_azure_deployment_name("m2", &mapped),
        Some("d2".to_owned())
    );
    assert_eq!(
        auth_guards::resolve_azure_deployment_name("m3", &mapped),
        None
    );
}

#[test]
fn bedrock_guard_covers_each_credential_source() {
    assert!(!auth_guards::has_bedrock_credentials(&env_from(&[])));
    assert!(auth_guards::has_bedrock_credentials(&env_from(&[(
        "AWS_PROFILE",
        "p"
    )])));
    assert!(auth_guards::has_bedrock_credentials(&env_from(&[
        ("AWS_ACCESS_KEY_ID", "id"),
        ("AWS_SECRET_ACCESS_KEY", "secret"),
    ])));
    assert!(auth_guards::has_bedrock_credentials(&env_from(&[(
        "AWS_BEARER_TOKEN_BEDROCK",
        "token"
    )])));
}

#[test]
fn cloudflare_guards_separate_workers_ai_from_ai_gateway() {
    let key_only = env_from(&[("CLOUDFLARE_API_KEY", "k"), ("CLOUDFLARE_ACCOUNT_ID", "a")]);
    assert!(auth_guards::has_cloudflare_workers_ai_credentials(
        &key_only
    ));
    assert!(!auth_guards::has_cloudflare_ai_gateway_credentials(
        &key_only
    ));
    assert!(auth_guards::has_cloudflare_ai_gateway_credentials(
        &env_from(&[
            ("CLOUDFLARE_API_KEY", "k"),
            ("CLOUDFLARE_ACCOUNT_ID", "a"),
            ("CLOUDFLARE_GATEWAY_ID", "g"),
        ])
    ));
}

#[test]
fn the_auth_json_store_round_trips_credentials_with_owner_only_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let dir = std::env::temp_dir().join(format!("pi-ai-auth-json-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let store = auth_json::AuthJsonStore {
        auth_path: dir.join("auth.json"),
    };
    assert!(store.load().is_empty());

    let mut credentials = BTreeMap::new();
    credentials.insert(
        "anthropic".to_owned(),
        Credential::OAuth(oauth_credentials("a", "r", 1_000)),
    );
    store.save(&credentials);
    let mode = std::fs::metadata(&store.auth_path)
        .expect("auth.json written")
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600);
    assert!(matches!(
        store.load().get("anthropic"),
        Some(Credential::OAuth(_))
    ));

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[tokio::test]
async fn the_auth_json_resolver_refreshes_an_expired_oauth_token_and_saves_it_back() {
    let dir = std::env::temp_dir().join(format!("pi-ai-auth-resolve-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let store = auth_json::AuthJsonStore {
        auth_path: dir.join("auth.json"),
    };
    let mut credentials = BTreeMap::new();
    credentials.insert(
        "xai".to_owned(),
        Credential::OAuth(oauth_credentials("stale-access", "old-refresh", 0)),
    );
    store.save(&credentials);

    let refreshed = oauth_credentials("new-access", "new-refresh", 2_000);
    let oauth = StubOAuthAuth::new("xAI (Grok/X subscription)", Ok(refreshed.clone()))
        .with_refresh(refresh_returning(refreshed))
        .auth();
    assert_eq!(
        auth_json::resolve_api_key(&store, "xai", Some(&oauth)).await,
        Some("new-access".to_owned())
    );
    match store.load().get("xai") {
        Some(Credential::OAuth(credential)) => {
            assert_eq!(credential.access, "new-access");
            assert_eq!(credential.refresh, "new-refresh");
        }
        other => assert!(
            matches!(other, Some(Credential::OAuth(credential)) if credential.access == "new-access"),
            "the rotated credential persisted: {other:?}"
        ),
    }

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[tokio::test]
async fn the_auth_json_resolver_serves_api_keys_and_misses_the_rest() {
    let dir = std::env::temp_dir().join(format!("pi-ai-auth-resolve-miss-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let store = auth_json::AuthJsonStore {
        auth_path: dir.join("auth.json"),
    };
    let mut credentials = BTreeMap::new();
    credentials.insert(
        "openai".to_owned(),
        Credential::ApiKey(ApiKeyCredential {
            key: Some("sk-key".to_owned()),
            env: None,
        }),
    );
    store.save(&credentials);
    assert_eq!(
        auth_json::resolve_api_key(&store, "openai", None).await,
        Some("sk-key".to_owned())
    );
    // A provider without a stored credential resolves nothing, the missing
    // entry path.
    assert_eq!(
        auth_json::resolve_api_key(&store, "openrouter", None).await,
        None
    );

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

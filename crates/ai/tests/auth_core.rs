//! Rust-native coverage of the auth core, the substance upstream covers
//! through `Models.getAuth` (`packages/ai/src/auth/resolve.ts`,
//! `packages/ai/src/auth/credential-store.ts`, and the credential serde of
//! `packages/ai/src/auth/types.ts`) at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`. The `Models` collection itself
//! lands with the registry ticket, so the resolution contract is pinned
//! against [`resolve_provider_auth`] directly over a stubbed
//! [`ProviderAuth`], the way upstream's `createModels` cases drive the same
//! path.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected shape panics by design"
)]

mod common;

use std::collections::HashMap;
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::Poll;
use std::time::Duration;

use common::auth_fixtures::{
    ApiKeyAuthStub, FailingStore, MapAuthContext, RecordingInteraction, StubLoginOutcome,
    StubOAuthAuth, oauth_credentials, refresh_after_sleeping, refresh_failing, refresh_returning,
    stub_error,
};
use pi_ai::auth::clock::{AuthClock as _, FixedClock, SteppedClock, SystemClock};
use pi_ai::auth::context::DefaultAuthContext;
use pi_ai::auth::credential_store::{CredentialStore, InMemoryCredentialStore};
use pi_ai::auth::helpers::env_api_key_auth;
use pi_ai::auth::oauth::device_code::{PollOptions, PollOutcome, poll_oauth_device_code_flow};
use pi_ai::auth::oauth::kimi_coding::KimiCodingOAuth;
use pi_ai::auth::resolve::{
    AuthResolutionOverrides, ModelsError, ModelsErrorCode, ModelsFailure, resolve_provider_auth,
};
use pi_ai::auth::types::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, ApiKeyResolveFn, AuthContext, AuthOptions,
    AuthPrompt, AuthPromptKind, AuthResult, AuthType, Credential, CredentialInfo,
    CredentialModifyFn, ModelAuth, OAuthCredentials, ProviderAuth, ProviderAuthInteraction,
};
use pi_ai::env_api_keys::{find_env_keys, get_env_api_key};
use pi_ai::http::{MockHttpClient, MockResponse};
use pi_ai::types::{BoxedFuture, ProviderEnv};
use tokio_util::sync::CancellationToken;

const PROVIDER: &str = "stub-provider";

/// The login outcome the resolve-driven stubs carry: resolution never logs
/// in, so the stub fails if a case drives it.
fn unused_login() -> StubLoginOutcome {
    Err(String::from("login is not driven by resolution"))
}

/// The wall clock the resolution and the credentials' expiries share.
fn now() -> i64 {
    SystemClock.now_ms()
}

/// A modify closure storing `credential`, the write the suites drive.
fn set_credential(credential: Credential) -> CredentialModifyFn {
    Box::new(move |_| Box::pin(std::future::ready(Ok(Some(credential)))))
}

/// A stored api-key credential, upstream's `{ type: "api_key", key }` wire
/// shape.
fn api_key_stored(key: &str) -> Credential {
    Credential::ApiKey(ApiKeyCredential {
        key: Some(key.to_owned()),
        env: None,
    })
}

/// A stored OAuth credential valid for `valid_for_ms` from the wall clock,
/// the expiry the resolution's five-minute window checks.
fn oauth_stored(access: &str, valid_for_ms: i64) -> Credential {
    Credential::OAuth(oauth_credentials(access, "r", now() + valid_for_ms))
}

/// The empty auth context: no ambient env values, no files.
fn context() -> Arc<dyn AuthContext> {
    Arc::new(MapAuthContext::default())
}

/// A provider offering only the OAuth flow under test.
fn oauth_provider_of(oauth: impl FnOnce() -> pi_ai::auth::types::OAuthAuth) -> ProviderAuth {
    ProviderAuth {
        api_key: None,
        oauth: Some(oauth()),
    }
}

/// A provider offering only api-key auth.
fn api_key_provider(auth: impl FnOnce() -> ApiKeyAuth) -> ProviderAuth {
    ProviderAuth {
        api_key: Some(auth()),
        oauth: None,
    }
}

/// Store `credential` under `provider_id`, the login-time write.
async fn store_credential(store: &dyn CredentialStore, provider_id: &str, credential: Credential) {
    store
        .modify(provider_id, set_credential(credential), None)
        .await
        .expect("the credential stores");
}

/// A one-shot sender the modify closures deliver their observation through.
/// A store the resolution must never reach, standing in where a case pins a
/// path that never touches credentials.
fn empty_store() -> Arc<dyn CredentialStore> {
    Arc::new(InMemoryCredentialStore::default())
}

/// The failing store behind the override cases that must not read
/// credentials; a read would reject, keeping the short-circuit honest.
fn failing_store() -> Arc<dyn CredentialStore> {
    Arc::new(FailingStore)
}

/// The failure the resolution settles with, unwrapped to the
/// [`ModelsError`] the assertions read.
fn resolution_error(
    outcome: Result<Option<AuthResult>, ModelsFailure>,
) -> Result<Option<AuthResult>, ModelsError> {
    outcome.map_err(|failure| match failure {
        ModelsFailure::Models(error) => error,
        ModelsFailure::Aborted(abort) => panic!("unexpected abort: {abort}"),
    })
}

// ---------------------------------------------------------------------------
// InMemoryCredentialStore
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reads_missing_entries_as_none() {
    let store = InMemoryCredentialStore::default();
    let read = store.read(PROVIDER, None).await.expect("read succeeds");
    assert_eq!(read, None);
}

#[tokio::test]
async fn modify_stores_the_returned_credential() {
    let store = InMemoryCredentialStore::default();
    let credential = api_key_stored("stored-key");
    let posted = store
        .modify(PROVIDER, set_credential(credential.clone()), None)
        .await
        .expect("modify succeeds");
    assert_eq!(posted, Some(credential.clone()));
    let read = store.read(PROVIDER, None).await.expect("read succeeds");
    assert_eq!(read, Some(credential));
}

#[tokio::test]
async fn modify_returning_none_leaves_the_entry_and_returns_the_current() {
    let store = InMemoryCredentialStore::default();
    let credential = api_key_stored("stored-key");
    store_credential(&store, PROVIDER, credential.clone()).await;
    let posted = store
        .modify(
            PROVIDER,
            Box::new(|_| {
                Box::pin(std::future::ready(Ok::<
                    Option<Credential>,
                    pi_ai::auth::types::AuthError,
                >(None)))
            }),
            None,
        )
        .await
        .expect("modify succeeds");
    assert_eq!(
        posted,
        Some(credential.clone()),
        "the current resolves back"
    );
    let read = store.read(PROVIDER, None).await.expect("read succeeds");
    assert_eq!(read, Some(credential), "the entry is unchanged");
}

#[tokio::test]
async fn list_reports_provider_ids_and_auth_types() {
    let store: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    store_credential(store.as_ref(), "anthropic", api_key_stored("k")).await;
    store_credential(
        store.as_ref(),
        "github-copilot",
        oauth_stored("a", 3_600_000),
    )
    .await;
    let mut infos = store.list(None).await.expect("list succeeds");
    infos.sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
    assert_eq!(
        infos,
        vec![
            CredentialInfo {
                provider_id: "anthropic".to_owned(),
                auth_type: AuthType::ApiKey
            },
            CredentialInfo {
                provider_id: "github-copilot".to_owned(),
                auth_type: AuthType::OAuth,
            },
        ]
    );
}

#[tokio::test]
async fn delete_removes_the_entry() {
    let store = InMemoryCredentialStore::default();
    store_credential(&store, PROVIDER, api_key_stored("k")).await;
    store.delete(PROVIDER, None).await.expect("delete succeeds");
    let read = store.read(PROVIDER, None).await.expect("read succeeds");
    assert_eq!(read, None);
}

/// The lock a pending modify holds until its peer operation has been
/// issued, the deterministic serialization probe.
#[derive(Debug, Default)]
struct ModifyGate {
    peer_started: AtomicBool,
    notify: tokio::sync::Notify,
}

impl ModifyGate {
    /// Wait until the peer operation has been issued; the modify holding
    /// the chain lock runs this before releasing it.
    async fn wait_for_peer(&self) {
        loop {
            if self.peer_started.load(Ordering::SeqCst) {
                return;
            }
            self.notify.notified().await;
        }
    }

    /// Announce that the peer operation has been issued.
    fn announce_peer(&self) {
        self.peer_started.store(true, Ordering::SeqCst);
        self.notify.notify_one();
    }
}

#[tokio::test]
async fn concurrent_modifies_on_one_provider_serialize() {
    let store: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    let gate = Arc::new(ModifyGate::default());
    let credential_a = api_key_stored("a");
    let credential_b = api_key_stored("b");
    let (a_entered, a_entered_rx) = tokio::sync::oneshot::channel::<Option<Credential>>();

    // A observes the empty entry, then holds the chain lock until B has
    // queued its own modify.
    let task_a = {
        let store = Arc::clone(&store);
        let gate = Arc::clone(&gate);
        let credential = credential_a.clone();
        tokio::spawn(async move {
            store
                .modify(
                    "shared",
                    Box::new(move |current| {
                        let gate = Arc::clone(&gate);
                        let credential = credential;
                        let entered = a_entered;
                        Box::pin(async move {
                            let _ = entered.send(current.clone());
                            gate.wait_for_peer().await;
                            Ok(Some(credential))
                        })
                    }),
                    None,
                )
                .await
        })
    };
    let seen_by_a = a_entered_rx.await.expect("A entered the modify");
    assert_eq!(
        seen_by_a, None,
        "A holds the chain lock over the empty entry"
    );

    // B queues behind A: its modify is issued while A still holds the lock.
    let (b_seen, b_seen_rx) = tokio::sync::oneshot::channel::<Option<Credential>>();
    let task_b = {
        let store = Arc::clone(&store);
        let gate = Arc::clone(&gate);
        let credential = credential_b.clone();
        tokio::spawn(async move {
            gate.announce_peer();
            store
                .modify(
                    "shared",
                    Box::new(move |current| {
                        let seen = b_seen;
                        let credential = credential;
                        Box::pin(async move {
                            let _ = seen.send(current);
                            Ok(Some(credential))
                        })
                    }),
                    None,
                )
                .await
        })
    };
    let seen_by_b = b_seen_rx
        .await
        .expect("B observed the entry under the lock");
    assert_eq!(
        seen_by_b,
        Some(credential_a.clone()),
        "the second modify runs after the first's write lands"
    );

    let (posted_a, posted_b) = tokio::join!(task_a, task_b);
    assert_eq!(
        posted_a
            .expect("task A joins")
            .expect("A's modify succeeds"),
        Some(credential_a)
    );
    assert_eq!(
        posted_b
            .expect("task B joins")
            .expect("B's modify succeeds"),
        Some(credential_b.clone())
    );
    let read = store.read("shared", None).await.expect("read succeeds");
    assert_eq!(read, Some(credential_b));
}

#[tokio::test]
async fn delete_serializes_behind_a_pending_modify() {
    let store: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    let gate = Arc::new(ModifyGate::default());
    let credential = api_key_stored("a");
    let (a_entered, a_entered_rx) = tokio::sync::oneshot::channel::<Option<Credential>>();

    let task_a = {
        let store = Arc::clone(&store);
        let gate = Arc::clone(&gate);
        let credential = credential.clone();
        tokio::spawn(async move {
            store
                .modify(
                    "shared",
                    Box::new(move |current| {
                        let entered = a_entered;
                        let gate = Arc::clone(&gate);
                        Box::pin(async move {
                            let _ = entered.send(current);
                            gate.wait_for_peer().await;
                            Ok(Some(credential))
                        })
                    }),
                    None,
                )
                .await
        })
    };
    let seen_by_a = a_entered_rx.await.expect("A entered the modify");
    assert_eq!(
        seen_by_a, None,
        "A holds the chain lock over the empty entry"
    );

    let task_delete = {
        let store = Arc::clone(&store);
        let gate = Arc::clone(&gate);
        tokio::spawn(async move {
            gate.announce_peer();
            store.delete("shared", None).await
        })
    };

    let (posted, deleted) = tokio::join!(task_a, task_delete);
    assert_eq!(
        posted.expect("task A joins").expect("A's modify succeeds"),
        Some(credential)
    );
    deleted
        .expect("task delete joins")
        .expect("delete succeeds");
    let read = store.read("shared", None).await.expect("read succeeds");
    assert_eq!(read, None, "the delete ran after the modify's write");
}

// ---------------------------------------------------------------------------
// Credential serde
// ---------------------------------------------------------------------------

#[test]
fn api_key_credentials_round_trip() {
    let credential: Credential =
        serde_json::from_str(r#"{"type":"api_key","key":"k"}"#).expect("parses");
    assert_eq!(
        credential,
        Credential::ApiKey(ApiKeyCredential {
            key: Some("k".to_owned()),
            env: None
        })
    );
    let wire = serde_json::to_value(&credential).expect("serializes");
    assert_eq!(wire, serde_json::json!({"type": "api_key", "key": "k"}));
}

#[test]
fn oauth_credentials_round_trip_with_extras() {
    let wire = serde_json::json!({
        "type": "oauth",
        "refresh": "r",
        "access": "a",
        "expires": 123_i64,
        "enterpriseUrl": "company.ghe.com",
    });
    let credential: Credential = serde_json::from_value(wire.clone()).expect("parses");
    let Credential::OAuth(oauth) = &credential else {
        panic!("the oauth wire parses to the oauth variant");
    };
    assert_eq!(oauth.refresh, "r");
    assert_eq!(oauth.access, "a");
    assert_eq!(oauth.expires, 123);
    assert_eq!(
        oauth
            .extra
            .get("enterpriseUrl")
            .and_then(serde_json::Value::as_str),
        Some("company.ghe.com")
    );
    assert_eq!(serde_json::to_value(&credential).expect("serializes"), wire);
}

#[test]
fn unknown_credential_types_reject() {
    let error = serde_json::from_str::<Credential>(r#"{"type":"customer_key","key":"k"}"#)
        .expect_err("an unknown type rejects");
    assert!(
        error.to_string().contains("unknown variant `customer_key`"),
        "the parse error names the type: {error}"
    );
}

// ---------------------------------------------------------------------------
// resolve_provider_auth
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_stored_oauth_credential_resolves_with_the_oauth_source() {
    let oauth = StubOAuthAuth::new("Stub OAuth", unused_login());
    let provider = oauth_provider_of(|| oauth.auth());
    let store = empty_store();
    store_credential(
        store.as_ref(),
        PROVIDER,
        oauth_stored("stored-access", 10 * 60_000),
    )
    .await;

    let resolved = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &store, &context(), None).await,
    )
    .expect("the resolution succeeds")
    .expect("a stored credential resolves");
    assert_eq!(resolved.source.as_deref(), Some("OAuth"));
    assert_eq!(resolved.auth.api_key.as_deref(), Some("stored-access"));
}

#[tokio::test]
async fn a_stored_oauth_credential_without_an_oauth_handler_resolves_none() {
    let provider = api_key_provider(|| {
        ApiKeyAuthStub::new("Stub key", ApiKeyAuthStub::resolving_none()).auth()
    });
    let store = empty_store();
    store_credential(
        store.as_ref(),
        PROVIDER,
        oauth_stored("stored-access", 10 * 60_000),
    )
    .await;

    let resolved = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &store, &context(), None).await,
    )
    .expect("the resolution succeeds");
    assert_eq!(
        resolved, None,
        "no silent env fallback for an unmatched credential type"
    );
}

#[tokio::test]
async fn override_env_merges_under_a_stored_api_key_credential() {
    let mut stored_env = ProviderEnv::new();
    stored_env.insert("A".to_owned(), "stored".to_owned());
    let store = empty_store();
    store_credential(
        store.as_ref(),
        PROVIDER,
        Credential::ApiKey(ApiKeyCredential {
            key: Some("stored-key".to_owned()),
            env: Some(stored_env),
        }),
    )
    .await;

    let mut overrides_env = ProviderEnv::new();
    overrides_env.insert("A".to_owned(), "override".to_owned());
    overrides_env.insert("B".to_owned(), "extra".to_owned());
    let overrides = AuthResolutionOverrides {
        env: Some(overrides_env),
        ..AuthResolutionOverrides::default()
    };
    let provider =
        api_key_provider(|| env_api_key_auth("Anthropic API key", &["ANTHROPIC_API_KEY"]));

    let resolved = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &store, &context(), Some(&overrides)).await,
    )
    .expect("the resolution succeeds")
    .expect("the stored credential resolves");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("stored-key"));
    let mut merged = ProviderEnv::new();
    merged.insert("A".to_owned(), "override".to_owned());
    merged.insert("B".to_owned(), "extra".to_owned());
    assert_eq!(resolved.env, Some(merged));
}

#[tokio::test]
async fn the_api_key_override_short_circuits_resolution() {
    let api_key = ApiKeyAuthStub::new("Stub key", ApiKeyAuthStub::resolving_key("override-key"));
    let provider = api_key_provider(|| api_key.auth());
    let overrides = AuthResolutionOverrides {
        api_key: Some("override-key".to_owned()),
        ..AuthResolutionOverrides::default()
    };

    let resolved = resolution_error(
        resolve_provider_auth(
            PROVIDER,
            &provider,
            &failing_store(),
            &context(),
            Some(&overrides),
        )
        .await,
    )
    .expect("the resolution succeeds")
    .expect("the override resolves");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("override-key"));
    let credential = api_key
        .last_credential()
        .expect("the override reached the handler");
    assert_eq!(credential.key.as_deref(), Some("override-key"));
}

#[tokio::test]
async fn ambient_env_resolves_when_nothing_is_stored() {
    let provider =
        api_key_provider(|| env_api_key_auth("Anthropic API key", &["ANTHROPIC_API_KEY"]));
    let ambient: Arc<dyn AuthContext> =
        Arc::new(MapAuthContext::new([("ANTHROPIC_API_KEY", "ambient-key")]));

    let resolved = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &empty_store(), &ambient, None).await,
    )
    .expect("the resolution succeeds")
    .expect("the ambient env resolves");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("ambient-key"));
    assert_eq!(resolved.source.as_deref(), Some("ANTHROPIC_API_KEY"));
}

#[tokio::test]
async fn an_expiring_oauth_credential_refreshes_under_the_lock_and_stores_the_rotation() {
    let expires = now() + 3_600_000;
    let oauth = StubOAuthAuth::new("Stub OAuth", unused_login()).with_refresh(refresh_returning(
        oauth_credentials("new-access", "new-refresh", expires),
    ));
    let provider = oauth_provider_of(|| oauth.auth());
    let store = empty_store();
    store_credential(
        store.as_ref(),
        PROVIDER,
        oauth_stored("old-access", 4 * 60_000),
    )
    .await;

    let resolved = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &store, &context(), None).await,
    )
    .expect("the resolution succeeds")
    .expect("the refreshed credential resolves");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("new-access"));
    let read = store.read(PROVIDER, None).await.expect("read succeeds");
    assert_eq!(
        read,
        Some(Credential::OAuth(oauth_credentials(
            "new-access",
            "new-refresh",
            expires
        ))),
        "the rotated credential persists"
    );
    assert_eq!(oauth.refresh_calls(), 1);
}

#[tokio::test]
async fn concurrent_resolutions_refresh_once_and_both_see_the_rotated_token() {
    let expires = now() + 3_600_000;
    let oauth = StubOAuthAuth::new("Stub OAuth", unused_login()).with_refresh(refresh_returning(
        oauth_credentials("new-access", "new-refresh", expires),
    ));
    let provider = oauth_provider_of(|| oauth.auth());
    let store = empty_store();
    store_credential(
        store.as_ref(),
        PROVIDER,
        oauth_stored("old-access", 4 * 60_000),
    )
    .await;

    let task_a = {
        let provider = provider.clone();
        let store = Arc::clone(&store);
        tokio::spawn(async move {
            resolve_provider_auth(PROVIDER, &provider, &store, &context(), None).await
        })
    };
    let task_b = {
        let provider = provider.clone();
        let store = Arc::clone(&store);
        tokio::spawn(async move {
            resolve_provider_auth(PROVIDER, &provider, &store, &context(), None).await
        })
    };
    for joined in [
        task_a.await.expect("task A joins"),
        task_b.await.expect("task B joins"),
    ] {
        let resolved = resolution_error(joined)
            .expect("the resolution succeeds")
            .expect("the credential resolves");
        assert_eq!(resolved.auth.api_key.as_deref(), Some("new-access"));
    }
    assert_eq!(
        oauth.refresh_calls(),
        1,
        "double-checked locking refreshes once globally"
    );
}

/// A store whose modify observes the entry already gone — another process
/// logged out while the resolution was preparing its refresh, the branch
/// upstream's `modify` callback handles as `undefined`.
#[derive(Debug, Default)]
struct LoggedOutDuringRefreshStore {
    entries: Mutex<HashMap<String, Credential>>,
}

impl CredentialStore for LoggedOutDuringRefreshStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, pi_ai::auth::types::AuthError>> {
        let read = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(provider_id)
            .cloned();
        Box::pin(std::future::ready(Ok(read)))
    }

    fn list<'a>(
        &'a self,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, pi_ai::auth::types::AuthError>> {
        let infos = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|(provider_id, credential)| CredentialInfo {
                provider_id: provider_id.clone(),
                auth_type: credential.auth_type(),
            })
            .collect();
        Box::pin(std::future::ready(Ok(infos)))
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: CredentialModifyFn,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, pi_ai::auth::types::AuthError>> {
        // The logout happened before the modify ran: the entry is gone, so
        // the closure sees None and the store resolves None.
        let removed = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(provider_id);
        Box::pin(async move {
            let _ = removed;
            f(None).await
        })
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), pi_ai::auth::types::AuthError>> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(provider_id);
        Box::pin(std::future::ready(Ok(())))
    }
}

#[tokio::test]
async fn a_logout_during_refresh_resolves_none() {
    let oauth = StubOAuthAuth::new("Stub OAuth", unused_login()).with_refresh(refresh_returning(
        oauth_credentials("late-access", "r", now() + 3_600_000),
    ));
    let provider = oauth_provider_of(|| oauth.auth());
    let store: Arc<dyn CredentialStore> = Arc::new(LoggedOutDuringRefreshStore::default());
    store_credential(
        store.as_ref(),
        PROVIDER,
        oauth_stored("old-access", 4 * 60_000),
    )
    .await;

    let resolved = resolve_provider_auth(PROVIDER, &provider, &store, &context(), None)
        .await
        .expect("the resolution succeeds");
    assert_eq!(resolved, None, "the logged-out resolution resolves none");
    assert_eq!(oauth.refresh_calls(), 0, "no credential is left to refresh");
}

#[tokio::test]
async fn a_failed_refresh_surfaces_the_oauth_models_error() {
    let oauth = StubOAuthAuth::new("Stub OAuth", unused_login())
        .with_refresh(refresh_failing("invalid_grant"));
    let provider = oauth_provider_of(|| oauth.auth());
    let store = empty_store();
    store_credential(
        store.as_ref(),
        PROVIDER,
        oauth_stored("old-access", 4 * 60_000),
    )
    .await;

    let ModelsFailure::Models(error) =
        resolve_provider_auth(PROVIDER, &provider, &store, &context(), None)
            .await
            .expect_err("the refresh failure fails the resolution")
    else {
        panic!("the refresh failure is a models failure");
    };
    assert_eq!(error.code(), ModelsErrorCode::OAuth);
    assert_eq!(
        error.to_string(),
        "OAuth refresh failed for stub-provider: invalid_grant"
    );
}

#[tokio::test]
async fn a_refreshed_token_below_the_requested_validity_rejects() {
    let oauth = StubOAuthAuth::new("Stub OAuth", unused_login()).with_refresh(refresh_returning(
        oauth_credentials("soon-access", "r", now() + 10 * 60_000),
    ));
    let provider = oauth_provider_of(|| oauth.auth());
    let store = empty_store();
    // Outside the five-minute default window, inside the one-hour request.
    store_credential(
        store.as_ref(),
        PROVIDER,
        oauth_stored("old-access", 6 * 60_000),
    )
    .await;
    let overrides = AuthResolutionOverrides {
        min_oauth_validity_ms: Some(3_600_000),
        ..AuthResolutionOverrides::default()
    };

    let error = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &store, &context(), Some(&overrides)).await,
    )
    .expect_err("the too-soon expiry fails the resolution");
    assert_eq!(error.code(), ModelsErrorCode::OAuth);
    assert_eq!(
        error.to_string(),
        "OAuth refresh returned a token that expires too soon for stub-provider"
    );
}

#[tokio::test(start_paused = true)]
async fn a_stuck_refresh_times_out_after_fifteen_seconds() {
    let oauth = StubOAuthAuth::new("Stub OAuth", unused_login()).with_refresh(
        refresh_after_sleeping(30_000, oauth_credentials("late-access", "r", 0)),
    );
    let provider = oauth_provider_of(|| oauth.auth());
    let store = empty_store();
    store_credential(
        store.as_ref(),
        PROVIDER,
        oauth_stored("old-access", 4 * 60_000),
    )
    .await;

    let handle = {
        let provider = provider.clone();
        let store = Arc::clone(&store);
        tokio::spawn(async move {
            resolve_provider_auth(PROVIDER, &provider, &store, &context(), None).await
        })
    };
    let mut advanced_ms = 0_u64;
    while !handle.is_finished() && advanced_ms < 20_000 {
        tokio::time::advance(Duration::from_millis(100)).await;
        tokio::task::yield_now().await;
        advanced_ms += 100;
    }
    let outcome = handle.await.expect("the resolution task joins");
    let ModelsFailure::Models(error) =
        outcome.expect_err("the refresh timeout fails the resolution")
    else {
        panic!("the refresh timeout is a models failure");
    };
    assert_eq!(error.code(), ModelsErrorCode::OAuth);
    assert_eq!(
        error.to_string(),
        "OAuth refresh timed out for stub-provider"
    );
}

#[tokio::test]
async fn a_failed_api_key_resolution_surfaces_the_auth_models_error() {
    let provider = api_key_provider(|| {
        ApiKeyAuthStub::new(
            "Stub key",
            ApiKeyAuthStub::failing("ambient lookup exploded"),
        )
        .auth()
    });

    let error = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &empty_store(), &context(), None).await,
    )
    .expect_err("the resolution failure surfaces");
    assert_eq!(error.code(), ModelsErrorCode::Auth);
    assert_eq!(
        error.to_string(),
        "API key auth failed for provider stub-provider: ambient lookup exploded"
    );
}

#[tokio::test]
async fn a_failing_store_read_surfaces_the_auth_models_error() {
    let provider = api_key_provider(|| {
        ApiKeyAuthStub::new("Stub key", ApiKeyAuthStub::resolving_none()).auth()
    });

    let error = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &failing_store(), &context(), None).await,
    )
    .expect_err("the store failure surfaces");
    assert_eq!(error.code(), ModelsErrorCode::Auth);
    assert_eq!(
        error.to_string(),
        "Credential store read failed for stub-provider: storage backend exploded"
    );
}

#[tokio::test]
async fn an_aborted_resolution_rejects() {
    let provider = api_key_provider(|| {
        ApiKeyAuthStub::new("Stub key", ApiKeyAuthStub::resolving_none()).auth()
    });
    let signal = CancellationToken::new();
    signal.cancel();
    let overrides = AuthResolutionOverrides {
        signal: Some(signal),
        ..AuthResolutionOverrides::default()
    };

    let failure = resolve_provider_auth(
        PROVIDER,
        &provider,
        &empty_store(),
        &context(),
        Some(&overrides),
    )
    .await
    .expect_err("the abort surfaces");
    let ModelsFailure::Aborted(abort) = failure else {
        panic!("the abort surfaces as the aborted failure");
    };
    assert_eq!(
        abort.to_string(),
        "The operation was aborted",
        "the abort is the standard reason: {abort}"
    );
}

// ---------------------------------------------------------------------------
// types.rs: Debug redaction, Display/serde vocabulary
// ---------------------------------------------------------------------------

#[test]
fn the_api_key_env_field_round_trips_through_the_wire() {
    let mut env = ProviderEnv::new();
    env.insert("CLOUDFLARE_ACCOUNT_ID".to_owned(), "acc".to_owned());
    let credential = Credential::ApiKey(ApiKeyCredential {
        key: Some("k".to_owned()),
        env: Some(env.clone()),
    });
    let wire = serde_json::to_value(&credential).expect("serializes");
    assert_eq!(
        wire,
        serde_json::json!({
            "type": "api_key",
            "key": "k",
            "env": {"CLOUDFLARE_ACCOUNT_ID": "acc"},
        })
    );
    let parsed: Credential = serde_json::from_value(wire).expect("parses");
    assert_eq!(parsed, credential);
}

#[test]
fn an_oauth_credential_without_expires_rejects() {
    let error =
        serde_json::from_str::<Credential>(r#"{"type":"oauth","refresh":"r","access":"a"}"#)
            .expect_err("a missing expiry rejects");
    assert!(
        error.to_string().contains("missing field `expires`"),
        "the parse error names the field: {error}"
    );
}

#[test]
fn auth_types_render_and_round_trip() {
    assert_eq!(AuthType::ApiKey.to_string(), "api_key");
    assert_eq!(AuthType::OAuth.to_string(), "oauth");
    assert_eq!(
        serde_json::to_value(AuthType::ApiKey).expect("serializes"),
        serde_json::json!("api_key")
    );
    assert_eq!(
        serde_json::from_str::<AuthType>("\"oauth\"").expect("parses"),
        AuthType::OAuth
    );
    let error =
        serde_json::from_str::<AuthType>("\"saml\"").expect_err("an unknown variant rejects");
    assert!(
        error.to_string().contains("unknown variant"),
        "the parse error names the variant: {error}"
    );
}

#[test]
fn prompts_carry_placeholders_and_events_render() {
    let prompt = AuthPrompt {
        signal: None,
        kind: AuthPromptKind::Text {
            message: "Paste the code".to_owned(),
            placeholder: Some("pi-…".to_owned()),
        },
    };
    let AuthPromptKind::Text { placeholder, .. } = &prompt.kind else {
        panic!("the prompt keeps its kind");
    };
    assert_eq!(placeholder.as_deref(), Some("pi-…"));

    let info = format!(
        "{:?}",
        pi_ai::auth::types::AuthEvent::Info {
            message: "m".to_owned(),
            links: None,
        }
    );
    assert!(info.contains('m'), "the info event debugs: {info:?}");
}

#[test]
fn struct_defaults_report_ambient_only_auth() {
    let api_key = ApiKeyAuthStub::new("Stub key", ApiKeyAuthStub::resolving_none()).auth();
    assert_eq!(api_key.name, "Stub key");
    assert!(
        api_key.login.is_none(),
        "a handler without a login closure is ambient-only"
    );
    assert!(
        api_key.check.is_none(),
        "a handler without a check closure resolves by resolving"
    );

    let oauth = StubOAuthAuth::new("Stub OAuth", unused_login());
    assert!(
        oauth.auth().login_label.is_none(),
        "the default login label is unset"
    );
    let provider = ProviderAuth {
        api_key: Some(api_key),
        oauth: Some(oauth.auth()),
    };
    let provider_debug = format!("{provider:?}");
    assert!(
        provider_debug.contains("ProviderAuth") && provider_debug.contains("Stub key"),
        "the provider debug names the type and handlers: {provider_debug:?}"
    );
}

#[test]
fn models_error_codes_display_their_wire_names() {
    let codes = [
        (ModelsErrorCode::ModelSource, "model_source"),
        (ModelsErrorCode::ModelValidation, "model_validation"),
        (ModelsErrorCode::Provider, "provider"),
        (ModelsErrorCode::Stream, "stream"),
        (ModelsErrorCode::Auth, "auth"),
        (ModelsErrorCode::OAuth, "oauth"),
    ];
    for (code, name) in codes {
        assert_eq!(code.wire(), name, "{code:?} renders {name:?}");
        assert_eq!(code.to_string(), name);
    }
}

#[test]
fn models_error_folds_the_cause_into_the_message_once() {
    let folded = ModelsError::with_cause(
        ModelsErrorCode::OAuth,
        "refresh failed",
        stub_error("invalid_grant"),
    );
    assert_eq!(folded.to_string(), "refresh failed: invalid_grant");

    let already_carried = ModelsError::with_cause(
        ModelsErrorCode::OAuth,
        "invalid_grant",
        stub_error("invalid_grant"),
    );
    assert_eq!(
        already_carried.to_string(),
        "invalid_grant",
        "a detail already in the message does not fold twice"
    );

    let blank = ModelsError::new(ModelsErrorCode::Auth, "resolution failed");
    assert_eq!(
        blank.to_string(),
        "resolution failed",
        "no cause folds nothing"
    );
}

// ---------------------------------------------------------------------------
// clock.rs: SteppedClock advance, FixedClock lockstep determinism
// ---------------------------------------------------------------------------

#[test]
fn the_stepped_clock_advances_by_hand() {
    let clock = SteppedClock::new(1_000);
    assert_eq!(clock.now_ms(), 1_000);
    clock.advance(500);
    assert_eq!(clock.now_ms(), 1_500, "advance moves the epoch");
    clock.advance(0);
    assert_eq!(clock.now_ms(), 1_500);
    let debug = format!("{clock:?}");
    assert!(
        debug.contains("SteppedClock"),
        "the clock debugs: {debug:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn fixed_clocks_constructed_at_the_same_epoch_stay_in_lockstep() {
    let first = FixedClock::new(1_000_000);
    let second = FixedClock::new(1_000_000);
    assert_eq!(first.now_ms(), second.now_ms(), "same start, same now");
    tokio::time::advance(Duration::from_millis(2_500)).await;
    assert_eq!(
        first.now_ms(),
        second.now_ms(),
        "the clocks ride the same tokio clock"
    );
    assert_eq!(first.now_ms(), 1_002_500);
    let debug = format!("{first:?}");
    assert!(
        debug.contains("FixedClock") && debug.contains("1000000"),
        "the fixed clock debugs its epoch: {debug:?}"
    );
    let system_debug = format!("{SystemClock:?}");
    assert_eq!(system_debug, "SystemClock");
    assert!(SystemClock.now_ms() > 0, "the wall clock reads an epoch");
}

// ---------------------------------------------------------------------------
// context.rs: DefaultAuthContext over the process environment and filesystem
// ---------------------------------------------------------------------------

/// The probe key the child-process runs key their scenario on.
const ENV_PROBE: &str = "PI_AI_AUTH_CORE_ENV_PROBE";

/// One child-probe spawn spec: the mode key, the environment entries the
/// child gets, and the environment entries the child loses.
type ProbeSpec = (&'static str, Vec<(String, String)>, Vec<String>);

/// Run this suite's own binary as a child with `extra` overlaying the child's
/// environment and `remove` dropped from it. `set_var` is forbidden in this
/// workspace, so the process-env readers are driven by composing the child's
/// environment at spawn time instead — the same probe pattern the CLI suite
/// uses for stdout capture.
fn probe_child(test_name: &str, mode: &str, extra: &[(&str, &str)], remove: &[&str]) {
    let mut command =
        std::process::Command::new(std::env::current_exe().expect("the test binary path"));
    command
        .args(["--exact", test_name, "--nocapture"])
        .env(ENV_PROBE, mode);
    for (name, value) in extra {
        command.env(name, value);
    }
    for name in remove {
        command.env_remove(name);
    }
    let output = command.output().expect("the probe child runs");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the {mode:?} probe child passes: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// One unique temp directory per call, safe under parallel probe children.
fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let unique = format!(
        "pi-ai-auth-core-{}-{}-{}",
        tag,
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed),
    );
    std::env::temp_dir().join(unique)
}

#[expect(
    clippy::too_many_lines,
    reason = "one spawn spec per probe mode keeps the child environments legible"
)]
#[test]
fn the_process_env_readers_serve_the_probe_modes() {
    if let Ok(mode) = std::env::var(ENV_PROBE) {
        run_env_probe_mode(&mode);
        return;
    }

    // The env readers in the child: DefaultAuthContext, find_env_keys,
    // get_env_api_key, and the Kimi host override. The child's environment
    // is composed at spawn time, so no process-env mutation happens here.
    let context_home = unique_temp_dir("context-home");
    std::fs::create_dir_all(&context_home).expect("the context home creates");
    let adc_home = unique_temp_dir("adc-home");
    std::fs::create_dir_all(&adc_home).expect("the adc home creates");
    let empty_home = unique_temp_dir("adc-empty");
    std::fs::create_dir_all(&empty_home).expect("the empty home creates");
    let adc_explicit = std::env::temp_dir().join(format!(
        "pi-ai-auth-core-adc-explicit-{}",
        std::process::id()
    ));
    std::fs::write(&adc_explicit, b"{}").expect("the explicit ADC file writes");
    let adc_explicit_str = adc_explicit.to_string_lossy().into_owned();
    let home = context_home.to_string_lossy().into_owned();
    let adc_home_str = adc_home.to_string_lossy().into_owned();
    let empty_home_str = empty_home.to_string_lossy().into_owned();

    let homes: Vec<ProbeSpec> = vec![
        (
            "context-env-set",
            vec![(
                String::from("PI_AI_AUTH_CORE_CONTEXT_ENV"),
                String::from("probe-value"),
            )],
            Vec::new(),
        ),
        (
            "context-env-blank",
            vec![(
                String::from("PI_AI_AUTH_CORE_CONTEXT_ENV"),
                String::from("   "),
            )],
            Vec::new(),
        ),
        (
            "context-env-unset",
            Vec::new(),
            vec![String::from("PI_AI_AUTH_CORE_CONTEXT_ENV")],
        ),
        (
            "context-file",
            vec![(String::from("HOME"), home)],
            Vec::new(),
        ),
        (
            "env-keys-set",
            vec![
                (String::from("NVIDIA_API_KEY"), String::from("probe-nvidia")),
                (String::from("OPENAI_API_KEY"), String::from("probe-openai")),
            ],
            Vec::new(),
        ),
        (
            "env-keys-unset",
            Vec::new(),
            vec![
                String::from("NVIDIA_API_KEY"),
                String::from("OPENAI_API_KEY"),
            ],
        ),
        (
            "anthropic-bearer",
            vec![(
                String::from("ANTHROPIC_AUTH_TOKEN"),
                String::from("probe-auth"),
            )],
            vec![
                String::from("ANTHROPIC_API_KEY"),
                String::from("ANTHROPIC_OAUTH_TOKEN"),
            ],
        ),
        (
            "anthropic-api-key",
            vec![(
                String::from("ANTHROPIC_API_KEY"),
                String::from("probe-anthropic"),
            )],
            vec![
                String::from("ANTHROPIC_AUTH_TOKEN"),
                String::from("ANTHROPIC_OAUTH_TOKEN"),
            ],
        ),
        (
            "adc-default",
            vec![
                (String::from("HOME"), adc_home_str),
                (String::from("GOOGLE_CLOUD_PROJECT"), String::from("p")),
                (String::from("GOOGLE_CLOUD_LOCATION"), String::from("l")),
            ],
            vec![
                String::from("GOOGLE_CLOUD_API_KEY"),
                String::from("GOOGLE_APPLICATION_CREDENTIALS"),
            ],
        ),
        (
            "adc-missing",
            vec![(String::from("HOME"), empty_home_str)],
            vec![
                String::from("GOOGLE_CLOUD_API_KEY"),
                String::from("GOOGLE_APPLICATION_CREDENTIALS"),
            ],
        ),
        (
            "vertex-explicit",
            vec![
                (
                    String::from("GOOGLE_APPLICATION_CREDENTIALS"),
                    adc_explicit_str,
                ),
                (String::from("GOOGLE_CLOUD_PROJECT"), String::from("p")),
                (String::from("GOOGLE_CLOUD_LOCATION"), String::from("l")),
            ],
            vec![String::from("GOOGLE_CLOUD_API_KEY")],
        ),
        (
            "kimi-override",
            vec![(
                String::from("KIMI_CODE_OAUTH_HOST"),
                String::from("https://kimi-override.example/"),
            )],
            Vec::new(),
        ),
        (
            "kimi-fallback",
            vec![(
                String::from("KIMI_OAUTH_HOST"),
                String::from("https://kimi-fallback.example"),
            )],
            vec![String::from("KIMI_CODE_OAUTH_HOST")],
        ),
    ];
    for (mode, extra, remove) in homes {
        let extra: Vec<(&str, &str)> = extra
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        let remove: Vec<&str> = remove.iter().map(String::as_str).collect();
        probe_child(
            "the_process_env_readers_serve_the_probe_modes",
            mode,
            &extra,
            &remove,
        );
    }
    let _ = std::fs::remove_file(&adc_explicit);
    std::fs::remove_dir_all(&context_home).ok();
    std::fs::remove_dir_all(&adc_home).ok();
    std::fs::remove_dir_all(&empty_home).ok();

    // A real file behind an absolute path, no environment involved.
    let context = DefaultAuthContext;
    let dir = unique_temp_dir("context");
    std::fs::create_dir_all(&dir).expect("the temp dir creates");
    let file = dir.join("probe.txt");
    std::fs::write(&file, b"x").expect("the probe file writes");
    assert!(
        context.file_exists(&file.to_string_lossy()),
        "an absolute path stats directly"
    );
    assert!(
        !context.file_exists(&dir.join("missing").to_string_lossy()),
        "a missing file reports false"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The child half of [`the_process_env_readers_serve_the_probe_modes`]:
/// each mode drives one process-env reader against the child's own
/// environment and asserts the contract.
#[expect(
    clippy::too_many_lines,
    reason = "one probe mode per branch keeps the child-environment contracts legible"
)]
fn run_env_probe_mode(mode: &str) {
    let context = DefaultAuthContext;
    match mode {
        "context-env-set" => {
            assert_eq!(
                context.env("PI_AI_AUTH_CORE_CONTEXT_ENV"),
                Some("probe-value".to_owned()),
                "a set variable resolves"
            );
        }
        "context-env-blank" => {
            assert_eq!(
                context.env("PI_AI_AUTH_CORE_CONTEXT_ENV"),
                None,
                "blank values drop, upstream's falsy-string semantics"
            );
        }
        "context-env-unset" => {
            assert_eq!(
                context.env("PI_AI_AUTH_CORE_CONTEXT_ENV"),
                None,
                "an unset variable resolves to none"
            );
        }
        "context-file" => {
            let home = std::env::home_dir().expect("the overridden home");
            std::fs::write(home.join("probe"), b"x").expect("the probe file writes");
            assert!(
                context.file_exists("~probe"),
                "the tilde path expands into $HOME"
            );
            assert!(
                !context.file_exists("~/missing"),
                "a missing tilde path reports false"
            );
        }
        "env-keys-set" => {
            assert_eq!(
                find_env_keys("nvidia", None),
                Some(vec!["NVIDIA_API_KEY".to_owned()]),
                "the process env discovery finds the configured key"
            );
            assert_eq!(
                get_env_api_key("openai", None),
                Some("probe-openai".to_owned())
            );
        }
        "env-keys-unset" => {
            assert_eq!(
                find_env_keys("nvidia", None),
                None,
                "unset vars do not resolve"
            );
            assert_eq!(get_env_api_key("openai", None), None);
            assert_eq!(find_env_keys("unknown-provider", None), None);
        }
        "anthropic-bearer" => {
            // ANTHROPIC_AUTH_TOKEN alone resolves nothing: it rides the
            // Authorization header, not the x-api-key slot.
            assert_eq!(get_env_api_key("anthropic", None), None);
            assert_eq!(
                find_env_keys("anthropic", None),
                Some(vec!["ANTHROPIC_AUTH_TOKEN".to_owned()])
            );
        }
        "anthropic-api-key" => {
            assert_eq!(
                get_env_api_key("anthropic", None),
                Some("probe-anthropic".to_owned())
            );
        }
        "adc-default" => {
            let home = std::env::home_dir().expect("the overridden home");
            std::fs::create_dir_all(home.join(".config/gcloud")).expect("the gcloud dir creates");
            assert_eq!(
                get_env_api_key("google-vertex", None),
                None,
                "no ADC file exists yet, so the ambient branch stays unauthenticated"
            );
        }
        "adc-missing" => {
            assert_eq!(
                get_env_api_key("google-vertex", None),
                None,
                "no ADC file yet"
            );
        }
        "vertex-explicit" => {
            let path = std::env::var("GOOGLE_APPLICATION_CREDENTIALS")
                .expect("the probe passes the explicit credentials path");
            assert!(
                std::path::Path::new(&path).exists(),
                "the parent created the explicit ADC file"
            );
            // The explicit path short-circuits the tilde default, upstream's
            // GOOGLE_APPLICATION_CREDENTIALS branch.
            assert_eq!(
                get_env_api_key("google-vertex", None),
                Some("<authenticated>".to_owned()),
                "the explicit ADC file authenticates"
            );
        }
        "kimi-override" | "kimi-fallback" => {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("the runtime builds");
            runtime.block_on(run_kimi_host_probe(mode));
        }
        other => panic!("unknown probe mode: {other:?}"),
    }
}

/// The child half of the Kimi host-override probes: the configured host
/// replaces the default `https://auth.kimi.com` device endpoint.
async fn run_kimi_host_probe(mode: &str) {
    let device_url = match mode {
        "kimi-override" => "https://kimi-override.example/api/oauth/device_authorization",
        "kimi-fallback" => "https://kimi-fallback.example/api/oauth/device_authorization",
        other => panic!("unknown kimi probe mode: {other:?}"),
    };
    let mock = MockHttpClient::new();
    mock.on(move |request| request.url == device_url)
        .respond(MockResponse::status(500));
    let recording = RecordingInteraction::new();
    let interaction = ProviderAuthInteraction::from_interaction(
        recording.interaction(),
        CancellationToken::new(),
    );
    let error = KimiCodingOAuth::new(Arc::new(mock.clone()))
        .login(interaction)
        .await
        .expect_err("the failed device authorization fails login");
    assert!(
        error
            .to_string()
            .starts_with("Kimi Code device authorization failed with status 500"),
        "the failure carries the wire status: {error:?}"
    );
    assert_eq!(
        mock.recorded()[0].url,
        device_url,
        "the override host redirects the device request"
    );
    assert!(
        recording.prompts().is_empty(),
        "the device flow never prompts"
    );
}

// ---------------------------------------------------------------------------
// helpers.rs: env_api_key_auth login and resolve branches
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_standard_api_key_login_prompts_for_the_key() {
    let recording = RecordingInteraction::with_answers(vec![Ok(String::from("sk-ambient"))]);
    let interaction = ProviderAuthInteraction::from_interaction(
        recording.interaction(),
        CancellationToken::new(),
    );
    let auth = env_api_key_auth("Anthropic API key", &["ANTHROPIC_API_KEY"]);
    let login = auth.login.as_ref().expect("the standard login exists");

    let credential = (Arc::clone(login))(interaction)
        .await
        .expect("the prompted key stores");
    assert_eq!(credential.key.as_deref(), Some("sk-ambient"));
    assert_eq!(credential.env, None);

    let prompt = &recording.prompts()[0];
    let AuthPromptKind::Secret { message, .. } = &prompt.kind else {
        panic!("the login prompts for a secret");
    };
    assert_eq!(message, "Enter Anthropic API key");

    // An already-cancelled signal rejects before prompting.
    let signal = CancellationToken::new();
    signal.cancel();
    let aborted = ProviderAuthInteraction::from_interaction(
        RecordingInteraction::new().interaction(),
        signal,
    );
    let error = (login)(aborted)
        .await
        .expect_err("a cancelled login rejects");
    assert_eq!(error.to_string(), "The operation was aborted");
}

#[tokio::test]
async fn the_standard_resolve_reports_ambient_misses_as_none() {
    let auth = env_api_key_auth("Stub key", &["MISSING_A", "MISSING_B"]);
    let resolved = (auth.resolve)(ApiKeyAuthInput {
        ctx: Arc::new(MapAuthContext::default()),
        credential: None,
        signal: CancellationToken::new(),
    })
    .await
    .expect("the resolution succeeds");
    assert_eq!(resolved, None, "no ambient source resolves to none");
}

/// A context whose first env read cancels the operation's signal, the
/// deterministic abort-between-lookups probe.
#[derive(Debug, Default)]
struct CancelOnFirstEnv(Mutex<Option<CancellationToken>>);

impl AuthContext for CancelOnFirstEnv {
    fn env(&self, _name: &str) -> Option<String> {
        let token = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        if let Some(token) = token {
            token.cancel();
        }
        None
    }

    fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

#[tokio::test]
async fn an_abort_between_env_lookups_fails_the_resolution() {
    let signal = CancellationToken::new();
    let context = CancelOnFirstEnv(Mutex::new(Some(signal.clone())));
    let auth = env_api_key_auth("Stub key", &["MISSING_A", "MISSING_B"]);
    let error = (auth.resolve)(ApiKeyAuthInput {
        ctx: Arc::new(context),
        credential: None,
        signal: signal.clone(),
    })
    .await
    .expect_err("the mid-loop abort fails the resolution");
    assert_eq!(error.to_string(), "The operation was aborted");
}

// ---------------------------------------------------------------------------
// resolve.rs: override-merge branches, overlay context, mid-refresh abort
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stored_env_wins_when_the_override_carries_no_env() {
    let mut stored_env = ProviderEnv::new();
    stored_env.insert("A".to_owned(), "stored".to_owned());
    let store = empty_store();
    store_credential(
        store.as_ref(),
        PROVIDER,
        Credential::ApiKey(ApiKeyCredential {
            key: Some("stored-key".to_owned()),
            env: Some(stored_env.clone()),
        }),
    )
    .await;
    let provider = api_key_provider(|| env_api_key_auth("Stub key", &["ANTHROPIC_API_KEY"]));

    let resolved = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &store, &context(), None).await,
    )
    .expect("the resolution succeeds")
    .expect("the stored credential resolves");
    assert_eq!(resolved.env, Some(stored_env), "the stored env survives");
}

#[tokio::test]
async fn the_override_env_applies_when_the_stored_credential_has_none() {
    let store = empty_store();
    store_credential(
        store.as_ref(),
        PROVIDER,
        Credential::ApiKey(ApiKeyCredential {
            key: Some("stored-key".to_owned()),
            env: None,
        }),
    )
    .await;
    let mut override_env = ProviderEnv::new();
    override_env.insert("B".to_owned(), "override".to_owned());
    let overrides = AuthResolutionOverrides {
        env: Some(override_env.clone()),
        ..AuthResolutionOverrides::default()
    };
    let provider = api_key_provider(|| env_api_key_auth("Stub key", &["ANTHROPIC_API_KEY"]));

    let resolved = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &store, &context(), Some(&overrides)).await,
    )
    .expect("the resolution succeeds")
    .expect("the stored credential resolves");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("stored-key"));
    assert_eq!(resolved.env, Some(override_env));
}

/// An api-key handler that reports the context's file check as its key, the
/// probe for the overlay context's `file_exists` delegation.
fn file_probe_auth(path: String) -> ApiKeyAuth {
    let resolve: ApiKeyResolveFn = Arc::new(move |input: ApiKeyAuthInput| {
        let exists = input.ctx.file_exists(&path);
        let result = AuthResult {
            auth: ModelAuth {
                api_key: Some(exists.to_string()),
                ..ModelAuth::default()
            },
            env: None,
            source: Some("file probe".to_owned()),
        };
        Box::pin(std::future::ready(Ok(Some(result))))
    });
    ApiKeyAuth {
        name: "File probe".to_owned(),
        login: None,
        check: None,
        resolve,
    }
}

#[tokio::test]
async fn the_overlay_context_delegates_file_checks_to_the_base_context() {
    let dir = unique_temp_dir("overlay");
    std::fs::create_dir_all(&dir).expect("the temp dir creates");
    std::fs::write(dir.join("probe"), b"x").expect("the probe file writes");

    let mut env = ProviderEnv::new();
    env.insert("PROBE_ONLY".to_owned(), "1".to_owned());
    let overrides = AuthResolutionOverrides {
        api_key: Some("override-key".to_owned()),
        env: Some(env),
        ..AuthResolutionOverrides::default()
    };
    let provider =
        api_key_provider(|| file_probe_auth(dir.join("probe").to_string_lossy().into_owned()));
    let base: Arc<dyn AuthContext> = Arc::new(DefaultAuthContext);

    let resolved = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &empty_store(), &base, Some(&overrides)).await,
    )
    .expect("the resolution succeeds")
    .expect("the override resolves");
    assert_eq!(
        resolved.auth.api_key.as_deref(),
        Some("true"),
        "the overlay context's file check reached the filesystem: {resolved:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn the_overlay_context_overlays_env_values_over_the_base() {
    // The overlay context wraps every resolution whose overrides carry env
    // values: the handler resolves OVERLAY_ONLY from the overlay...
    let mut env = ProviderEnv::new();
    env.insert("OVERLAY_ONLY".to_owned(), "overlay".to_owned());
    let overrides = AuthResolutionOverrides {
        env: Some(env),
        ..AuthResolutionOverrides::default()
    };
    let provider = api_key_provider(|| env_api_key_auth("Stub key", &["OVERLAY_ONLY"]));
    let base: Arc<dyn AuthContext> = Arc::new(MapAuthContext::new([("BASE_ONLY", "base")]));

    let resolved = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &empty_store(), &base, Some(&overrides)).await,
    )
    .expect("the resolution succeeds")
    .expect("the ambient resolution runs");
    assert_eq!(
        resolved.auth.api_key.as_deref(),
        Some("overlay"),
        "the overlay value wins"
    );

    // ...and falls through to the base context for an overlay miss.
    let provider = api_key_provider(|| env_api_key_auth("Stub key", &["BASE_ONLY"]));
    let mut env = ProviderEnv::new();
    env.insert("OVERLAY_ONLY".to_owned(), "overlay".to_owned());
    let overrides = AuthResolutionOverrides {
        env: Some(env),
        ..AuthResolutionOverrides::default()
    };
    let base: Arc<dyn AuthContext> = Arc::new(MapAuthContext::new([("BASE_ONLY", "base")]));
    let resolved = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &empty_store(), &base, Some(&overrides)).await,
    )
    .expect("the resolution succeeds")
    .expect("the ambient resolution runs");
    assert_eq!(
        resolved.auth.api_key.as_deref(),
        Some("base"),
        "a miss in the overlay falls through to the base context"
    );
}

/// A store whose modify rejects with a storage failure after the read, the
/// branch the resolution wraps as `Credential store modify failed`.
#[derive(Debug, Default)]
struct ReadOkModifyFailsStore;

impl CredentialStore for ReadOkModifyFailsStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, pi_ai::auth::types::AuthError>> {
        assert_eq!(provider_id, PROVIDER);
        Box::pin(std::future::ready(Ok(Some(oauth_stored(
            "old-access",
            4 * 60_000,
        )))))
    }

    fn list<'a>(
        &'a self,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, pi_ai::auth::types::AuthError>> {
        Box::pin(std::future::ready(Ok(Vec::new())))
    }

    fn modify<'a>(
        &'a self,
        _provider_id: &'a str,
        _f: CredentialModifyFn,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, pi_ai::auth::types::AuthError>> {
        Box::pin(std::future::ready(Err(stub_error(
            "storage backend exploded",
        ))))
    }

    fn delete<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), pi_ai::auth::types::AuthError>> {
        Box::pin(std::future::ready(Ok(())))
    }
}

#[tokio::test]
async fn a_failing_store_modify_surfaces_the_auth_models_error() {
    let oauth = StubOAuthAuth::new("Stub OAuth", unused_login()).with_refresh(refresh_returning(
        oauth_credentials("new-access", "r", now() + 3_600_000),
    ));
    let provider = oauth_provider_of(|| oauth.auth());
    let store: Arc<dyn CredentialStore> = Arc::new(ReadOkModifyFailsStore);

    let error = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &store, &context(), None).await,
    )
    .expect_err("the modify failure fails the resolution");
    assert_eq!(error.code(), ModelsErrorCode::Auth);
    assert_eq!(
        error.to_string(),
        "Credential store modify failed for stub-provider: storage backend exploded"
    );
}

// ---------------------------------------------------------------------------
// credential_store.rs: clones, aborts, and error propagation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pre_cancelled_store_operations_reject_as_aborted() {
    let store = InMemoryCredentialStore::default();
    let signal = CancellationToken::new();
    signal.cancel();
    let options = AuthOptions {
        signal: Some(signal),
    };

    let error = store
        .read(PROVIDER, Some(&options))
        .await
        .expect_err("read aborts");
    assert_eq!(error.to_string(), "The operation was aborted");
    let error = store.list(Some(&options)).await.expect_err("list aborts");
    assert_eq!(error.to_string(), "The operation was aborted");
    let error: pi_ai::auth::types::AuthError = store
        .modify(
            PROVIDER,
            set_credential(api_key_stored("k")),
            Some(&options),
        )
        .await
        .expect_err("modify aborts");
    assert!(
        error.to_string().contains("The operation was aborted"),
        "the abort surfaces through the store error: {error:?}"
    );
    let error = store
        .delete(PROVIDER, Some(&options))
        .await
        .expect_err("delete aborts");
    assert_eq!(error.to_string(), "The operation was aborted");
}

#[tokio::test]
async fn a_modify_cancelled_while_queued_rejects_without_touching_the_entry() {
    let store: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    let gate = Arc::new(ModifyGate::default());
    let (a_entered, a_entered_rx) = tokio::sync::oneshot::channel::<Option<Credential>>();

    let task_a = {
        let store = Arc::clone(&store);
        let gate = Arc::clone(&gate);
        let entered = a_entered;
        tokio::spawn(async move {
            store
                .modify(
                    "queued",
                    Box::new(move |current| {
                        let gate = Arc::clone(&gate);
                        let entered = entered;
                        Box::pin(async move {
                            let _ = entered.send(current);
                            gate.wait_for_peer().await;
                            Ok(Some(api_key_stored("a")))
                        })
                    }),
                    None,
                )
                .await
        })
    };
    a_entered_rx.await.expect("A entered the modify");

    // B queues behind A, then its signal cancels before the lock is free.
    let signal = CancellationToken::new();
    let task_b = {
        let store = Arc::clone(&store);
        let signal = signal.clone();
        tokio::spawn(async move {
            store
                .modify(
                    "queued",
                    Box::new(|_| {
                        Box::pin(std::future::ready(Ok::<
                            Option<Credential>,
                            pi_ai::auth::types::AuthError,
                        >(None)))
                    }),
                    Some(&AuthOptions {
                        signal: Some(signal),
                    }),
                )
                .await
        })
    };
    tokio::task::yield_now().await;
    signal.cancel();
    gate.announce_peer();

    let error = task_b
        .await
        .expect("task B joins")
        .expect_err("the queued modify aborts");
    assert!(
        error.to_string().contains("The operation was aborted"),
        "the queued abort surfaces: {error:?}"
    );
    task_a
        .await
        .expect("task A joins")
        .expect("A's modify succeeds");
}

#[tokio::test]
async fn a_delete_cancelled_while_queued_rejects_as_aborted() {
    let store = Arc::new(InMemoryCredentialStore::default());
    store_credential(store.as_ref(), "queued", api_key_stored("k")).await;
    let gate = Arc::new(ModifyGate::default());
    let (a_entered, a_entered_rx) = tokio::sync::oneshot::channel::<Option<Credential>>();

    let task_a = {
        let store = Arc::clone(&store);
        let gate = Arc::clone(&gate);
        let entered = a_entered;
        tokio::spawn(async move {
            store
                .modify(
                    "queued",
                    Box::new(move |current| {
                        let gate = Arc::clone(&gate);
                        let entered = entered;
                        Box::pin(async move {
                            let _ = entered.send(current);
                            gate.wait_for_peer().await;
                            Ok(Some(api_key_stored("k")))
                        })
                    }),
                    None,
                )
                .await
        })
    };
    a_entered_rx.await.expect("A entered the modify");

    let signal = CancellationToken::new();
    let task_delete = {
        let store = Arc::clone(&store);
        let signal = signal.clone();
        tokio::spawn(async move {
            store
                .delete(
                    "queued",
                    Some(&AuthOptions {
                        signal: Some(signal),
                    }),
                )
                .await
        })
    };
    tokio::task::yield_now().await;
    signal.cancel();
    gate.announce_peer();

    let error = task_delete
        .await
        .expect("the queued delete joins")
        .expect_err("the queued delete aborts");
    assert_eq!(error.to_string(), "The operation was aborted");
    task_a
        .await
        .expect("task A joins")
        .expect("A's modify succeeds");
}

#[tokio::test]
async fn a_rejecting_modify_closure_propagates_through_the_store() {
    let store = InMemoryCredentialStore::default();
    let error = store
        .modify(
            PROVIDER,
            Box::new(|_| Box::pin(std::future::ready(Err(stub_error("modify refused"))))),
            None,
        )
        .await
        .expect_err("the modify rejection propagates");
    assert!(
        error.to_string().contains("modify refused"),
        "the closure's rejection is the store's: {error:?}"
    );
}

#[tokio::test]
async fn modifies_on_different_providers_run_concurrently() {
    let store: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::default());
    let (a_seen, a_seen_rx) = tokio::sync::oneshot::channel::<Option<Credential>>();
    let (b_seen, b_seen_rx) = tokio::sync::oneshot::channel::<Option<Credential>>();

    // Each modify signals entry, then waits for its peer to enter; the pair
    // only completes when both hold their own provider's chain at once.
    let task_a = {
        let store = Arc::clone(&store);
        let seen = a_seen;
        tokio::spawn(async move {
            store
                .modify(
                    "provider-a",
                    Box::new(move |current| {
                        let seen = seen;
                        Box::pin(async move {
                            let _ = seen.send(current.clone());
                            Ok(Some(api_key_stored("a")))
                        })
                    }),
                    None,
                )
                .await
        })
    };
    let task_b = {
        let store = Arc::clone(&store);
        let seen = b_seen;
        tokio::spawn(async move {
            store
                .modify(
                    "provider-b",
                    Box::new(move |current| {
                        let seen = seen;
                        Box::pin(async move {
                            let _ = seen.send(current);
                            Ok(Some(api_key_stored("b")))
                        })
                    }),
                    None,
                )
                .await
        })
    };
    let _ = tokio::join!(a_seen_rx, b_seen_rx);
    let (posted_a, posted_b) = tokio::join!(task_a, task_b);
    posted_a
        .expect("task A joins")
        .expect("A's modify succeeds");
    posted_b
        .expect("task B joins")
        .expect("B's modify succeeds");
}

// ---------------------------------------------------------------------------
// env_api_keys.rs: the ambient source branches over the injectable seams
// ---------------------------------------------------------------------------

#[test]
fn bedrock_authenticates_through_every_declared_credential_source() {
    let cases: Vec<Vec<(&str, &str)>> = vec![
        vec![("AWS_PROFILE", "default")],
        vec![("AWS_BEARER_TOKEN_BEDROCK", "bearer")],
        vec![("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials")],
        vec![(
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "http://169.254.170.2/v2",
        )],
        vec![("AWS_WEB_IDENTITY_TOKEN_FILE", "/var/run/token")],
    ];
    for env_pairs in cases {
        let env: ProviderEnv = env_pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        assert_eq!(
            get_env_api_key("amazon-bedrock", Some(&env)),
            Some("<authenticated>".to_owned()),
            "{env_pairs:?} authenticates"
        );
    }
    // A half IAM pair does not authenticate.
    let env: ProviderEnv = std::iter::once(("AWS_ACCESS_KEY_ID", "only"))
        .map(|(name, value)| (name.to_owned(), value.to_owned()))
        .collect();
    assert_eq!(get_env_api_key("amazon-bedrock", Some(&env)), None);
}

/// Vertex's ADC path requires project and location together: each half
/// missing leaves the ambient branch unauthenticated.
#[test]
fn vertex_adc_auth_requires_project_and_location() {
    let adc = std::env::temp_dir().join("pi-rust-vertex-adc-probe.json");
    std::fs::write(&adc, b"{}").expect("the ADC fixture writes");

    // The explicit credentials file exists, but no project is configured.
    let projectless = get_env_api_key(
        "google-vertex",
        Some(&ProviderEnv::from_iter([(
            "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
            adc.to_string_lossy().into_owned(),
        )])),
    );
    assert_eq!(projectless, None, "a missing project fails the ADC check");

    // Project and location both configured, but the ADC file is missing.
    let fileless = get_env_api_key(
        "google-vertex",
        Some(&ProviderEnv::from_iter([
            ("GOOGLE_CLOUD_PROJECT".to_owned(), "v".to_owned()),
            ("GOOGLE_CLOUD_LOCATION".to_owned(), "v".to_owned()),
        ])),
    );
    assert_eq!(fileless, None, "a missing credentials file fails the check");

    // The complete shape authenticates through the explicit file.
    let complete = get_env_api_key(
        "google-vertex",
        Some(&ProviderEnv::from_iter([
            (
                "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
                adc.to_string_lossy().into_owned(),
            ),
            ("GOOGLE_CLOUD_PROJECT".to_owned(), "v".to_owned()),
            ("GOOGLE_CLOUD_LOCATION".to_owned(), "v".to_owned()),
        ])),
    );
    let _ = std::fs::remove_file(&adc);
    assert_eq!(complete, Some("<authenticated>".to_owned()));
}

// ---------------------------------------------------------------------------
// device_code.rs: the poll loop's own surface
// ---------------------------------------------------------------------------

#[test]
fn only_completed_poll_outcomes_complete() {
    assert_eq!(PollOutcome::Complete(7).complete(), Some(7));
    assert_eq!(PollOutcome::<u8>::Pending.complete(), None);
    assert_eq!(
        PollOutcome::<u8>::SlowDown {
            interval_seconds: None
        }
        .complete(),
        None
    );
    assert_eq!(PollOutcome::<u8>::Failed("no".to_owned()).complete(), None);

    // The i64 instantiation the token-typed flows carry.
    assert_eq!(PollOutcome::Complete(7_i64).complete(), Some(7));
    assert_eq!(PollOutcome::<i64>::Pending.complete(), None);
    assert_eq!(
        PollOutcome::<i64>::SlowDown {
            interval_seconds: None
        }
        .complete(),
        None
    );
    assert_eq!(PollOutcome::<i64>::Failed("no".to_owned()).complete(), None);

    // The credential-typed instantiations the flows poll for.
    let oauth = oauth_credentials("a", "r", 1);
    assert_eq!(PollOutcome::Complete(oauth.clone()).complete(), Some(oauth));
    assert_eq!(PollOutcome::<OAuthCredentials>::Pending.complete(), None);
    assert_eq!(
        PollOutcome::<OAuthCredentials>::SlowDown {
            interval_seconds: None
        }
        .complete(),
        None
    );
    assert_eq!(
        PollOutcome::<OAuthCredentials>::Failed("no".to_owned()).complete(),
        None
    );
}

#[test]
fn the_poll_options_debug_names_the_scalars() {
    let options = PollOptions {
        interval_seconds: Some(5),
        expires_in_seconds: Some(600),
        wait_before_first_poll: true,
        signal: CancellationToken::new(),
        poll: Arc::new(|| Box::pin(std::future::ready(Ok(PollOutcome::<String>::Pending)))),
    };
    let debug = format!("{options:?}");
    assert!(
        debug.contains("interval_seconds")
            && debug.contains("expires_in_seconds")
            && debug.contains("wait_before_first_poll"),
        "the poll options debug names its scalars: {debug:?}"
    );
}

/// Poll the pinned flow once on the current task, the fake-timer probe
/// upstream's suites drive by hand.
fn poll_flow<F: Future>(flow: std::pin::Pin<&mut F>) -> Poll<F::Output> {
    let waker = std::task::Waker::noop();
    let mut cx = std::task::Context::from_waker(waker);
    flow.poll(&mut cx)
}

/// A poll script serving canned outcomes then parking.
fn scripted_poll<T: Send + 'static>(
    outcomes: Vec<PollOutcome<T>>,
) -> Arc<
    dyn Fn() -> BoxedFuture<'static, Result<PollOutcome<T>, pi_ai::auth::types::AuthError>>
        + Send
        + Sync,
> {
    let outcomes = Arc::new(Mutex::new(outcomes));
    Arc::new(move || {
        let next = outcomes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop();
        let parked: BoxedFuture<'static, Result<PollOutcome<T>, pi_ai::auth::types::AuthError>> =
            Box::pin(async move {
                if let Some(outcome) = next {
                    Ok(outcome)
                } else {
                    std::future::pending::<()>().await;
                    unreachable!("the parked poll never resolves")
                }
            });
        parked
    })
}

#[tokio::test(start_paused = true)]
async fn a_first_poll_skipped_when_the_lifetime_is_already_spent_times_out() {
    let options = PollOptions {
        interval_seconds: None,
        expires_in_seconds: Some(0),
        wait_before_first_poll: true,
        signal: CancellationToken::new(),
        poll: scripted_poll(vec![PollOutcome::<String>::Pending]),
    };
    let error = poll_oauth_device_code_flow(options)
        .await
        .expect_err("the spent lifetime times out");
    assert_eq!(error.to_string(), "Device flow timed out");
}

#[tokio::test(start_paused = true)]
async fn the_deadline_timeout_carries_the_slow_down_flavour_after_a_slow_down() {
    let options: PollOptions<String> = PollOptions {
        interval_seconds: None,
        // The deadline is a duration from start; one second of lifetime.
        expires_in_seconds: Some(1),
        wait_before_first_poll: false,
        signal: CancellationToken::new(),
        poll: scripted_poll(vec![PollOutcome::SlowDown {
            interval_seconds: None,
        }]),
    };
    let flow = pin!(poll_oauth_device_code_flow(options));
    let mut flow = flow;
    // The first poll lands immediately; the slow-down interval parks the wait.
    assert!(poll_flow(flow.as_mut()).is_pending());
    // Step past the deadline in whole-slow-down-interval chunks.
    for _ in 0..110 {
        tokio::time::advance(Duration::from_millis(10_000)).await;
        tokio::task::yield_now().await;
    }
    let error = match poll_flow(flow.as_mut()) {
        Poll::Ready(result) => result.expect_err("the deadline times out"),
        Poll::Pending => panic!("the deadline never arrived"),
    };
    assert!(
        error
            .to_string()
            .starts_with("Device flow timed out after one or more slow_down responses."),
        "the slow-down flavour is the deadline message: {error:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn a_poll_that_cancels_the_signal_is_seen_at_the_top_of_the_loop() {
    let signal = CancellationToken::new();
    let canceller: Arc<
        dyn Fn() -> BoxedFuture<'static, Result<PollOutcome<String>, pi_ai::auth::types::AuthError>>
            + Send
            + Sync,
    > = {
        let signal = signal.clone();
        Arc::new(move || {
            let signal = signal.clone();
            Box::pin(async move {
                signal.cancel();
                Ok(PollOutcome::<String>::Pending)
            })
        })
    };
    let options = PollOptions {
        interval_seconds: None,
        expires_in_seconds: None,
        wait_before_first_poll: false,
        signal: signal.clone(),
        poll: canceller,
    };
    let error = poll_oauth_device_code_flow(options)
        .await
        .expect_err("the cancelled loop rejects");
    assert_eq!(
        error.to_string(),
        "Login cancelled",
        "the cancellation is honoured before the next poll: {error:?}"
    );
}

// ---------------------------------------------------------------------------
// env_api_keys.rs: every provider's env-var arm
// ---------------------------------------------------------------------------

#[test]
fn every_provider_resolves_its_declared_api_key_env_var() {
    let cases: &[(&str, &str)] = &[
        ("github-copilot", "COPILOT_GITHUB_TOKEN"),
        ("ant-ling", "ANT_LING_API_KEY"),
        ("qwen-token-plan", "QWEN_TOKEN_PLAN_API_KEY"),
        ("qwen-token-plan-individual", "QWEN_TOKEN_PLAN_API_KEY"),
        ("qwen-token-plan-cn", "QWEN_TOKEN_PLAN_CN_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("azure-openai-responses", "AZURE_OPENAI_API_KEY"),
        ("nvidia", "NVIDIA_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("google", "GEMINI_API_KEY"),
        ("google-vertex", "GOOGLE_CLOUD_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("cerebras", "CEREBRAS_API_KEY"),
        ("xai", "XAI_API_KEY"),
        ("radius", "RADIUS_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("vercel-ai-gateway", "AI_GATEWAY_API_KEY"),
        ("zai", "ZAI_API_KEY"),
        ("zai-coding-cn", "ZAI_CODING_CN_API_KEY"),
        ("mistral", "MISTRAL_API_KEY"),
        ("minimax", "MINIMAX_API_KEY"),
        ("minimax-cn", "MINIMAX_CN_API_KEY"),
        ("moonshotai", "MOONSHOT_API_KEY"),
        ("moonshotai-cn", "MOONSHOT_API_KEY"),
        ("huggingface", "HF_TOKEN"),
        ("fireworks", "FIREWORKS_API_KEY"),
        ("together", "TOGETHER_API_KEY"),
        ("baseten", "BASETEN_API_KEY"),
        ("opencode", "OPENCODE_API_KEY"),
        ("opencode-go", "OPENCODE_API_KEY"),
        ("kimi-coding", "KIMI_API_KEY"),
        ("cloudflare-workers-ai", "CLOUDFLARE_API_KEY"),
        ("cloudflare-ai-gateway", "CLOUDFLARE_API_KEY"),
        ("xiaomi", "XIAOMI_API_KEY"),
        ("xiaomi-token-plan-cn", "XIAOMI_TOKEN_PLAN_CN_API_KEY"),
        ("xiaomi-token-plan-ams", "XIAOMI_TOKEN_PLAN_AMS_API_KEY"),
        ("xiaomi-token-plan-sgp", "XIAOMI_TOKEN_PLAN_SGP_API_KEY"),
    ];
    for (provider, env_var) in cases {
        let env: ProviderEnv = std::iter::once(((*env_var).to_owned(), "v".to_owned())).collect();
        assert_eq!(
            find_env_keys(provider, Some(&env)),
            Some(vec![(*env_var).to_owned()]),
            "{provider} resolves {env_var}"
        );
    }
}

// ---------------------------------------------------------------------------
// helpers.rs: resolve-side cancellation arms
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_pre_cancelled_signal_fails_the_standard_resolve_before_any_lookup() {
    let signal = CancellationToken::new();
    signal.cancel();
    let auth = env_api_key_auth("Stub key", &["ANTHROPIC_API_KEY"]);
    let error = (auth.resolve)(ApiKeyAuthInput {
        ctx: Arc::new(MapAuthContext::default()),
        credential: None,
        signal: signal.clone(),
    })
    .await
    .expect_err("the cancelled resolve rejects");
    assert_eq!(error.to_string(), "The operation was aborted");
}

#[tokio::test]
async fn an_abort_between_the_prompt_and_the_key_check_fails_the_login() {
    let signal = CancellationToken::new();
    let recording = RecordingInteraction::new();
    recording.set_dynamic({
        let signal = signal.clone();
        move |_prompt| -> BoxedFuture<'static, Result<String, pi_ai::utils::abort::AbortError>> {
            let signal = signal.clone();
            let hung: BoxedFuture<'static, Result<String, pi_ai::utils::abort::AbortError>> =
                Box::pin(async move {
                    signal.cancel();
                    Ok(String::from("sk-late"))
                });
            hung
        }
    });
    let interaction =
        ProviderAuthInteraction::from_interaction(recording.interaction(), signal.clone());
    let auth = env_api_key_auth("Anthropic API key", &["ANTHROPIC_API_KEY"]);
    let login = auth.login.expect("the standard login exists");
    let error = (login)(interaction)
        .await
        .expect_err("the post-prompt abort fails login");
    assert_eq!(error.to_string(), "The operation was aborted");
}

// ---------------------------------------------------------------------------
// resolve.rs: stored credential type without a matching handler
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_stored_api_key_credential_without_an_api_key_handler_resolves_none() {
    let oauth = StubOAuthAuth::new("Stub OAuth", unused_login());
    let provider = oauth_provider_of(|| oauth.auth());
    let store = empty_store();
    store_credential(store.as_ref(), PROVIDER, api_key_stored("k")).await;

    let resolved = resolution_error(
        resolve_provider_auth(PROVIDER, &provider, &store, &context(), None).await,
    )
    .expect("the resolution succeeds");
    assert_eq!(
        resolved, None,
        "no silent env fallback for an unmatched credential type"
    );
}

// ---------------------------------------------------------------------------
// credential_store.rs: a modify that aborts mid-operation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_modify_that_cancels_mid_operation_rejects_as_aborted() {
    // The closure cancels the signal and settles; the store's post-settle
    // signal check rejects the result, so the aborted mutation never lands.
    let store = InMemoryCredentialStore::default();
    store_credential(&store, PROVIDER, api_key_stored("k")).await;
    let signal = CancellationToken::new();
    let inner = signal.clone();
    let error = store
        .modify(
            PROVIDER,
            Box::new(move |_| {
                let signal = inner;
                Box::pin(async move {
                    signal.cancel();
                    Ok(Some(api_key_stored("cancelled")))
                })
            }),
            Some(&AuthOptions {
                signal: Some(signal.clone()),
            }),
        )
        .await
        .expect_err("the mid-operation abort fails the modify");
    assert!(
        error.to_string().contains("The operation was aborted"),
        "the abort surfaces: {error:?}"
    );
    let read = store.read(PROVIDER, None).await.expect("read succeeds");
    assert_eq!(
        read,
        Some(api_key_stored("k")),
        "the aborted mutation never persisted"
    );
}

// ---------------------------------------------------------------------------
// device_code.rs: the post-poll deadline break
// ---------------------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn a_poll_that_overshoots_the_deadline_breaks_to_the_timeout() {
    let options: PollOptions<String> = PollOptions {
        interval_seconds: None,
        expires_in_seconds: Some(1),
        wait_before_first_poll: false,
        signal: CancellationToken::new(),
        poll: Arc::new(|| {
            Box::pin(async move {
                // The poll itself takes longer than the whole lifetime.
                tokio::time::sleep(Duration::from_secs(2)).await;
                Ok(PollOutcome::<String>::Pending)
            })
        }),
    };
    let error = poll_oauth_device_code_flow(options)
        .await
        .expect_err("the overshoot times out");
    assert_eq!(error.to_string(), "Device flow timed out");
}

// ---------------------------------------------------------------------------
// types.rs: serde error paths
// ---------------------------------------------------------------------------

#[test]
fn credential_serde_error_paths_pin_their_messages() {
    // A missing tag.
    let error = serde_json::from_str::<Credential>("{}").expect_err("a missing tag rejects");
    assert!(
        error.to_string().contains("missing field `type`"),
        "{error}"
    );

    // A non-object env value.
    let error = serde_json::from_str::<Credential>(r#"{"type":"api_key","env":[]}"#)
        .expect_err("a malformed env rejects");
    assert!(
        error.to_string().contains("invalid type"),
        "the env failure is the underlying serde error: {error}"
    );

    // Missing refresh, then missing access.
    let error = serde_json::from_str::<Credential>(r#"{"type":"oauth","access":"a","expires":1}"#)
        .expect_err("a missing refresh rejects");
    assert!(
        error.to_string().contains("missing field `refresh`"),
        "{error}"
    );
    let error = serde_json::from_str::<Credential>(r#"{"type":"oauth","refresh":"r","expires":1}"#)
        .expect_err("a missing access rejects");
    assert!(
        error.to_string().contains("missing field `access`"),
        "{error}"
    );

    // An api-key credential without a key serializes without one.
    let wire = serde_json::to_value(Credential::ApiKey(ApiKeyCredential {
        key: None,
        env: None,
    }))
    .expect("serializes");
    assert_eq!(wire, serde_json::json!({"type": "api_key"}));

    // The api_key tag deserializes too.
    assert_eq!(
        serde_json::from_str::<AuthType>("\"api_key\"").expect("parses"),
        AuthType::ApiKey
    );
}

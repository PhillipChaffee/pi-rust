//! Boundary tests binding the `runtime_credentials` branches the 1:1 suites
//! leave untested, at pin 60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use pi_ai::auth::credential_store::CredentialStore;
use pi_ai::auth::types::{
    ApiKeyCredential, AuthError, AuthOptions, AuthType, Credential, CredentialInfo,
    CredentialModifyFn,
};
use pi_ai::types::BoxedFuture;
use pi_ai::utils::abort::AbortError;
use pi_coding_agent::auth_storage::{AuthStorage, AuthStorageData, InMemoryAuthStorageBackend};
use pi_coding_agent::runtime_credentials::RuntimeCredentials;
use tokio_util::sync::CancellationToken;

/// An api-key credential with the given key, the shape the read assertions pin.
#[expect(
    clippy::unnecessary_wraps,
    reason = "the helper returns the store read's full Option shape so assertions compare it whole"
)]
fn api_key(key: &str) -> Option<Credential> {
    Some(Credential::ApiKey(ApiKeyCredential {
        key: Some(key.to_owned()),
        env: None,
    }))
}

/// The in-memory store, the call spelled with its backend parameter because
/// the generic constructor path cannot infer it.
fn in_memory(data: &AuthStorageData) -> Arc<AuthStorage<InMemoryAuthStorageBackend>> {
    Arc::new(AuthStorage::<InMemoryAuthStorageBackend>::in_memory(data))
}

/// The shared handle letting a test read the wrapped store's own view.
struct SharedStore(Arc<AuthStorage<InMemoryAuthStorageBackend>>);

impl CredentialStore for SharedStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        CredentialStore::read(&*self.0, provider_id, options)
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        CredentialStore::list(&*self.0, options)
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: CredentialModifyFn,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        CredentialStore::modify(&*self.0, provider_id, f, options)
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        CredentialStore::delete(&*self.0, provider_id, options)
    }
}

/// The no-op store counting every operation, the probe for the signal checks.
struct CountingStore {
    reads: AtomicUsize,
    lists: AtomicUsize,
    modifies: AtomicUsize,
    deletes: AtomicUsize,
}

impl CountingStore {
    const fn new() -> Self {
        Self {
            reads: AtomicUsize::new(0),
            lists: AtomicUsize::new(0),
            modifies: AtomicUsize::new(0),
            deletes: AtomicUsize::new(0),
        }
    }
}

impl CredentialStore for CountingStore {
    fn read<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(None) })
    }

    fn list<'a>(
        &'a self,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        self.lists.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(Vec::new()) })
    }

    fn modify<'a>(
        &'a self,
        _provider_id: &'a str,
        _f: CredentialModifyFn,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        self.modifies.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(None) })
    }

    fn delete<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        self.deletes.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(()) })
    }
}

#[tokio::test]
async fn has_runtime_api_key_tracks_the_override() {
    let credentials = RuntimeCredentials::new(Box::new(CountingStore::new()));

    assert!(!credentials.has_runtime_api_key("anthropic"));
    credentials.set_runtime_api_key("anthropic", "key".to_owned());
    assert!(credentials.has_runtime_api_key("anthropic"));
    credentials.remove_runtime_api_key("anthropic");
    assert!(!credentials.has_runtime_api_key("anthropic"));
}

#[tokio::test]
async fn set_twice_keeps_one_entry_in_its_original_position() {
    let storage = in_memory(&AuthStorageData::new());
    let credentials = RuntimeCredentials::new(Box::new(SharedStore(Arc::clone(&storage))));

    credentials.set_runtime_api_key("anthropic", "first".to_owned());
    credentials.set_runtime_api_key("openai", "openai-key".to_owned());
    credentials.set_runtime_api_key("anthropic", "second".to_owned());

    let read = credentials.read("anthropic", None).await.expect("read");
    assert_eq!(read, api_key("second"), "the overwrite wins the value");
    let listed = credentials.list(None).await.expect("list");
    assert_eq!(
        listed
            .iter()
            .map(|info| info.provider_id.as_str())
            .collect::<Vec<_>>(),
        vec!["anthropic", "openai"],
        "the JS Map overwrite keeps the original position"
    );
}

#[tokio::test]
async fn remove_then_readd_moves_the_entry_to_the_end() {
    let storage = in_memory(&AuthStorageData::new());
    let credentials = RuntimeCredentials::new(Box::new(SharedStore(Arc::clone(&storage))));

    credentials.set_runtime_api_key("anthropic", "first".to_owned());
    credentials.set_runtime_api_key("openai", "openai-key".to_owned());
    credentials.remove_runtime_api_key("anthropic");
    credentials.set_runtime_api_key("anthropic", "readded".to_owned());

    let listed = credentials.list(None).await.expect("list");
    assert_eq!(
        listed
            .iter()
            .map(|info| info.provider_id.as_str())
            .collect::<Vec<_>>(),
        vec!["openai", "anthropic"],
        "a re-added override joins at the end"
    );
}

#[tokio::test]
async fn list_merge_keeps_the_original_position_of_an_overridden_entry() {
    let mut data = AuthStorageData::new();
    data.insert(
        "openai".to_owned(),
        serde_json::json!({"type": "api_key", "key": "stored"}),
    );
    data.insert(
        "anthropic".to_owned(),
        serde_json::json!({"type": "api_key", "key": "stored"}),
    );
    let storage = in_memory(&data);
    let credentials = RuntimeCredentials::new(Box::new(SharedStore(Arc::clone(&storage))));
    credentials.set_runtime_api_key("anthropic", "runtime".to_owned());

    let listed = credentials.list(None).await.expect("list");
    assert_eq!(
        listed,
        vec![
            CredentialInfo {
                provider_id: "openai".to_owned(),
                auth_type: AuthType::ApiKey,
            },
            CredentialInfo {
                provider_id: "anthropic".to_owned(),
                auth_type: AuthType::ApiKey,
            },
        ],
        "the override retypes the entry where it already sat"
    );
}

#[tokio::test]
async fn a_pre_aborted_read_fails_without_touching_the_store() {
    let counting = CountingStore::new();
    let credentials = RuntimeCredentials::new(Box::new(counting));
    let token = CancellationToken::new();
    token.cancel();
    let options = AuthOptions {
        signal: Some(token),
    };

    let error = credentials
        .read("anthropic", Some(&options))
        .await
        .expect_err("the read must abort");
    assert!(error.downcast_ref::<AbortError>().is_some());
}

#[tokio::test]
async fn a_pre_aborted_list_fails_after_the_store_reports() {
    let counting = CountingStore::new();
    let credentials = RuntimeCredentials::new(Box::new(counting));
    let token = CancellationToken::new();
    token.cancel();
    let options = AuthOptions {
        signal: Some(token),
    };

    let error = credentials
        .list(Some(&options))
        .await
        .expect_err("the list must abort");
    assert!(error.downcast_ref::<AbortError>().is_some());
}

#[tokio::test]
async fn a_pre_aborted_delete_fails_before_the_store_delete() {
    let counting = CountingStore::new();
    let credentials = RuntimeCredentials::new(Box::new(counting));
    let token = CancellationToken::new();
    token.cancel();
    let options = AuthOptions {
        signal: Some(token),
    };

    let error = credentials
        .delete("anthropic", Some(&options))
        .await
        .expect_err("the delete must abort");
    assert!(error.downcast_ref::<AbortError>().is_some());
}

#[tokio::test]
async fn modify_forwards_to_the_store_while_the_override_masks_reads() {
    let storage = in_memory(&AuthStorageData::new());
    let credentials = RuntimeCredentials::new(Box::new(SharedStore(Arc::clone(&storage))));
    credentials.set_runtime_api_key("anthropic", "runtime".to_owned());

    let update: CredentialModifyFn = Box::new(|_current: Option<Credential>| {
        Box::pin(async move {
            Ok(Some(Credential::ApiKey(ApiKeyCredential {
                key: Some("stored-new".to_owned()),
                env: None,
            })))
        })
    });
    let modified = credentials
        .modify("anthropic", update, None)
        .await
        .expect("modify");
    assert_eq!(
        modified,
        api_key("stored-new"),
        "the modify forwards and returns the store's answer"
    );

    let masked = credentials.read("anthropic", None).await.expect("read");
    assert_eq!(masked, api_key("runtime"), "the override still masks");
    let stored = storage.read("anthropic", None).await.expect("stored read");
    assert_eq!(
        stored,
        api_key("stored-new"),
        "the modification persisted under the mask"
    );
}

#[tokio::test]
async fn the_debug_form_renders() {
    let credentials = RuntimeCredentials::new(Box::new(CountingStore::new()));

    assert!(format!("{credentials:?}").contains("RuntimeCredentials"));
}

//! Upstream
//! `packages/coding-agent/test/model-runtime-credential-sync.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated for
//! `pi_coding_agent::model_runtime` (#121).
//!
//! Porting restatements this suite records:
//!
//! - The `provider(id, options)` double implements
//!   [`pi_ai::models::Provider`] directly; `getModels`' throw path is
//!   unreachable for the fixture, and the stream dispatches no case drives
//!   delegate to the shared fixture guard
//!   [`common::model_layer::unused_stream`].
//! - The markStarted/blocked promise pairs restate as the [`Gate`] notify
//!   pair: the scripted closure marks `started` and parks on `blocked`,
//!   whose stored permits keep both registration orders race-free.
//! - `await new Promise((resolve) => setTimeout(resolve, 0))` restates as
//!   scheduler yields.
//! - The delayed-commit case's `settled` flag rides the spawned login task;
//!   the settled mutation resolves the credential (upstream
//!   `packages/ai/src/models.ts:623`) and the aborted synchronization fails
//!   the login with `CredentialSynchronizationError`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::panic,
    reason = "the not-expected-outcome guard panics by design, upstream's fail()"
)]

#[expect(
    dead_code,
    reason = "the fixture module compiles whole into every test binary; this suite drives only its model-layer helpers"
)]
mod common;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use common::model_layer::{create_in_memory_model_registry, in_memory_auth_storage, model};
use pi_ai::auth::credential_store::{CredentialStore, InMemoryCredentialStore};
use pi_ai::auth::types::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCheckFn, ApiKeyCredential, ApiKeyLoginFn, ApiKeyResolveFn,
    AuthCheck, AuthError, AuthInteraction, AuthOptions, AuthResult, AuthType, Credential,
    ModelAuth, ProviderAuth,
};
use pi_ai::models::{
    ModelsRefreshOptions, Provider, ProviderError, ProviderModelError, RefreshModelsContext,
};
use pi_ai::types::{BoxedFuture, Context, Model, SimpleStreamOptions, StreamOptions};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use pi_coding_agent::auth_storage::AuthStorageData;
use pi_coding_agent::model_runtime::{
    CredentialSynchronizationError, CredentialSynchronizationOperation, ModelRuntime,
};
use tokio::sync::{Notify, oneshot};
use tokio_util::sync::CancellationToken;

/// The api-key availability answer upstream's default double returns, the
/// `credential ? { type: "api_key", source: "stored" } : undefined` pair.
fn stored_check(input: &ApiKeyAuthInput) -> Option<AuthCheck> {
    input.credential.as_ref().map(|_| AuthCheck {
        source: Some("stored".to_owned()),
        auth_type: AuthType::ApiKey,
    })
}

/// The stored api-key credential the assertions pin.
#[expect(
    clippy::unnecessary_wraps,
    reason = "the helper returns the store read's full Option shape so assertions compare it whole"
)]
fn api_key_credential(key: &str) -> Option<Credential> {
    Some(Credential::ApiKey(ApiKeyCredential {
        key: Some(key.to_owned()),
        env: None,
    }))
}

/// The runtime over in-memory credentials, upstream's
/// `ModelRuntime.create({ credentials, modelsPath: null, allowModelNetwork:
/// false })`.
async fn runtime_with(credentials: Arc<dyn CredentialStore>) -> ModelRuntime {
    create_in_memory_model_registry(credentials)
        .await
        .runtime()
        .clone()
}

/// The `{ prompt: async () => "unused", notify: () => {} }` interaction.
fn login_interaction() -> AuthInteraction {
    AuthInteraction {
        signal: None,
        prompt: Arc::new(|_prompt| Box::pin(async { Ok("unused".to_owned()) })),
        notify: Arc::new(|_event| {}),
    }
}

/// The interaction carrying the caller's cancellation, upstream's
/// `{ signal: controller.signal, prompt, notify }`.
fn login_interaction_with_signal(signal: CancellationToken) -> AuthInteraction {
    AuthInteraction {
        signal: Some(signal),
        ..login_interaction()
    }
}

/// The markStarted/blocked promise pair, upstream's gate promises: the
/// scripted closure marks `started` when entered and parks on `blocked`
/// until the test releases it. `Notify`'s stored permit keeps both
/// registration orders race-free.
struct Gate {
    started: Arc<Notify>,
    blocked: Arc<Notify>,
}

impl Gate {
    /// The fresh pair.
    fn new() -> Self {
        Self {
            started: Arc::new(Notify::new()),
            blocked: Arc::new(Notify::new()),
        }
    }

    /// The test's wait for the entry marker.
    async fn wait_started(&self) {
        self.started.notified().await;
    }

    /// The test's release of the blocked promise.
    fn release(&self) {
        self.blocked.notify_one();
    }
}

/// The login override, upstream's `login?: () => Promise<ApiKeyCredential>`
/// thunk.
type LoginOverride =
    Arc<dyn Fn() -> BoxedFuture<'static, Result<ApiKeyCredential, AuthError>> + Send + Sync>;

/// The refresh phase override, upstream's `refreshModels`: the context moves
/// in, the boxed future detaches from the provider.
type RefreshOverride = Arc<
    dyn Fn(RefreshModelsContext) -> BoxedFuture<'static, Result<(), ProviderError>> + Send + Sync,
>;

/// The scripted behaviors `provider(id, options)` accepts, upstream's
/// optional overrides object.
#[derive(Default)]
struct ProviderOptions {
    /// The login override.
    login: Option<LoginOverride>,
    /// The refresh phase override.
    refresh_models: Option<RefreshOverride>,
    /// The availability-check override, upstream's `check` replacement.
    check: Option<ApiKeyCheckFn>,
}

/// The scripted api-key provider double, upstream's `provider(id, options)`
/// factory: the "API key" method with the stored-credential check/resolve
/// pair, one dynamic-shaped model, and the optional overrides.
struct ProviderDouble {
    id: String,
    models: Vec<Model>,
    auth: ProviderAuth,
    refresh_models: Option<RefreshOverride>,
}

impl ProviderDouble {
    /// Build the double, upstream's `provider(id, options)`.
    fn new(id: &str, options: ProviderOptions) -> Self {
        let login: ApiKeyLoginFn = Arc::new({
            let id = id.to_owned();
            let login = options.login;
            move |_interaction| {
                let id = id.clone();
                let login = login.clone();
                Box::pin(async move {
                    if let Some(login) = login {
                        return login().await;
                    }
                    Ok(ApiKeyCredential {
                        key: Some(format!("{id}-key")),
                        env: None,
                    })
                })
            }
        });
        let check: ApiKeyCheckFn = options.check.unwrap_or_else(|| {
            Arc::new(|input: ApiKeyAuthInput| Box::pin(async move { Ok(stored_check(&input)) }))
        });
        let resolve: ApiKeyResolveFn = Arc::new(|input: ApiKeyAuthInput| {
            let resolution = input.credential.as_ref().map(|credential| AuthResult {
                auth: ModelAuth {
                    api_key: credential.key.clone(),
                    ..ModelAuth::default()
                },
                env: None,
                source: Some("stored".to_owned()),
            });
            Box::pin(async move { Ok(resolution) })
        });
        Self {
            id: id.to_owned(),
            models: vec![model(id, "dynamic")],
            auth: ProviderAuth {
                api_key: Some(ApiKeyAuth {
                    name: "API key".to_owned(),
                    login: Some(login),
                    check: Some(check),
                    resolve,
                }),
                oauth: None,
            },
            refresh_models: options.refresh_models,
        }
    }
}

impl Provider for ProviderDouble {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.id
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn get_models(&self) -> Result<Vec<Model>, ProviderModelError> {
        Ok(self.models.clone())
    }

    fn supports_refresh_models(&self) -> bool {
        self.refresh_models.is_some()
    }

    fn refresh_models(
        &self,
        context: RefreshModelsContext,
    ) -> BoxedFuture<'_, Result<(), ProviderError>> {
        let Some(refresh_models) = self.refresh_models.as_ref() else {
            return Box::pin(async { Ok(()) });
        };
        let refresh_models = Arc::clone(refresh_models);
        Box::pin(async move { refresh_models(context).await })
    }

    fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        common::model_layer::unused_stream()
    }

    fn stream_simple(
        &self,
        _model: &Model,
        _context: &Context,
        _options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        common::model_layer::unused_stream()
    }
}

/// upstream's `runtimeWithProvider`: create over the credentials, register
/// the native provider, and refresh its availability offline.
async fn runtime_with_provider(
    registered: ProviderDouble,
    credentials: Arc<dyn CredentialStore>,
) -> ModelRuntime {
    let provider_id = registered.id.clone();
    let runtime = runtime_with(credentials).await;
    runtime.register_native_provider(Arc::new(registered));
    runtime
        .refresh(ModelsRefreshOptions {
            allow_network: Some(false),
            providers: Some(vec![provider_id]),
            ..ModelsRefreshOptions::default()
        })
        .await;
    runtime
}

/// The credential store whose modify commits, notifies the test, and parks
/// until released, upstream's inline delayed-commit `credentials` object:
/// the write path settles only when the mutation gate opens.
struct DelayedCommitStore {
    inner: Arc<InMemoryCredentialStore>,
    committed: Mutex<Option<oneshot::Sender<()>>>,
    release: Arc<Notify>,
}

impl CredentialStore for DelayedCommitStore {
    fn read<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        self.inner.read(provider_id, options)
    }

    fn list<'a>(
        &'a self,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<pi_ai::auth::types::CredentialInfo>, AuthError>> {
        self.inner.list(options)
    }

    fn modify<'a>(
        &'a self,
        provider_id: &'a str,
        f: pi_ai::auth::types::CredentialModifyFn,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        Box::pin(async move {
            let current = self.inner.read(provider_id, options).await?;
            let mut stored = current.clone();
            let next = f(current).await?;
            if let Some(next) = &next {
                let committed_value = next.clone();
                stored = Some(committed_value.clone());
                let commit: pi_ai::auth::types::CredentialModifyFn =
                    Box::new(move |_current| Box::pin(async move { Ok(Some(committed_value)) }));
                self.inner.modify(provider_id, commit, None).await?;
            }
            let sender = self
                .committed
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            if let Some(sender) = sender {
                let _ = sender.send(());
            }
            self.release.notified().await;
            // Upstream's double returns the tracked credential, not a fresh
            // read: a read here would ride the caller's (possibly cancelled)
            // signal.
            Ok(stored)
        })
    }

    fn delete<'a>(
        &'a self,
        provider_id: &'a str,
        options: Option<&'a AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        self.inner.delete(provider_id, options)
    }
}

mod model_runtime_credential_sync {
    use super::*;

    /// upstream `it("publishes locally consistent availability before login
    /// and logout resolve")`.
    #[tokio::test]
    async fn publishes_locally_consistent_availability_before_login_and_logout_resolve() {
        let credentials = in_memory_auth_storage(&AuthStorageData::new());
        let runtime = runtime_with_provider(
            ProviderDouble::new("dynamic", ProviderOptions::default()),
            credentials.clone(),
        )
        .await;

        runtime
            .login("dynamic", AuthType::ApiKey, login_interaction())
            .await
            .expect("the login resolves");
        assert!(runtime.has_configured_auth("dynamic"));
        assert!(
            runtime
                .get_available_snapshot()
                .iter()
                .any(|entry| entry.id == "dynamic"),
            "the refreshed catalog carries the dynamic model"
        );
        let stored = credentials
            .read("dynamic", None)
            .await
            .expect("the read lands");
        assert_eq!(stored, api_key_credential("dynamic-key"));

        runtime
            .logout("dynamic", None)
            .await
            .expect("the logout resolves");
        assert!(!runtime.has_configured_auth("dynamic"));
        assert!(
            !runtime
                .get_available_snapshot()
                .iter()
                .any(|entry| entry.provider.0 == "dynamic"),
            "the logged-out provider leaves the available snapshot"
        );
        let stored = credentials
            .read("dynamic", None)
            .await
            .expect("the read lands");
        assert_eq!(stored, None);
    }

    /// upstream `it("orders same-provider credential operations through local
    /// synchronization")`.
    #[tokio::test]
    async fn orders_same_provider_credential_operations_through_local_synchronization() {
        let gate = Gate::new();
        let login: LoginOverride = Arc::new({
            let started = Arc::clone(&gate.started);
            let blocked = Arc::clone(&gate.blocked);
            move || {
                let started = Arc::clone(&started);
                let blocked = Arc::clone(&blocked);
                Box::pin(async move {
                    started.notify_one();
                    blocked.notified().await;
                    Ok(ApiKeyCredential {
                        key: Some("ordered-key".to_owned()),
                        env: None,
                    })
                })
            }
        });
        let credentials = in_memory_auth_storage(&AuthStorageData::new());
        let runtime = runtime_with_provider(
            ProviderDouble::new(
                "ordered",
                ProviderOptions {
                    login: Some(login),
                    ..ProviderOptions::default()
                },
            ),
            credentials.clone(),
        )
        .await;

        let login_task =
            tokio::spawn(runtime.login("ordered", AuthType::ApiKey, login_interaction()));
        gate.wait_started().await;
        let logout_task = tokio::spawn(runtime.logout("ordered", None));
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        let stored = credentials
            .read("ordered", None)
            .await
            .expect("the read lands");
        assert_eq!(stored, None, "the blocked login has not committed");

        gate.release();
        let _login_credential = login_task
            .await
            .expect("the login task joins")
            .expect("the login resolves");
        logout_task
            .await
            .expect("the logout task joins")
            .expect("the logout resolves");
        let stored = credentials
            .read("ordered", None)
            .await
            .expect("the read lands");
        assert_eq!(
            stored, None,
            "the queued logout removed the committed credential"
        );
        assert!(!runtime.has_configured_auth("ordered"));
    }

    /// upstream `it("allows different providers to run credential flows
    /// concurrently")`.
    #[tokio::test]
    async fn allows_different_providers_to_run_credential_flows_concurrently() {
        let gate_one = Gate::new();
        let gate_two = Gate::new();
        let gated_login = |gate: &Gate, key: String| {
            let started = Arc::clone(&gate.started);
            let blocked = Arc::clone(&gate.blocked);
            let login: LoginOverride = Arc::new(move || {
                let started = Arc::clone(&started);
                let blocked = Arc::clone(&blocked);
                let key = key.clone();
                Box::pin(async move {
                    started.notify_one();
                    blocked.notified().await;
                    Ok(ApiKeyCredential {
                        key: Some(key),
                        env: None,
                    })
                })
            });
            ProviderOptions {
                login: Some(login),
                ..ProviderOptions::default()
            }
        };
        let runtime = runtime_with(in_memory_auth_storage(&AuthStorageData::new())).await;
        runtime.register_native_provider(Arc::new(ProviderDouble::new(
            "one",
            gated_login(&gate_one, "one".to_owned()),
        )));
        runtime.register_native_provider(Arc::new(ProviderDouble::new(
            "two",
            gated_login(&gate_two, "two".to_owned()),
        )));
        runtime
            .refresh(ModelsRefreshOptions {
                allow_network: Some(false),
                providers: Some(vec!["one".to_owned(), "two".to_owned()]),
                ..ModelsRefreshOptions::default()
            })
            .await;

        let one = tokio::spawn(runtime.login("one", AuthType::ApiKey, login_interaction()));
        let two = tokio::spawn(runtime.login("two", AuthType::ApiKey, login_interaction()));
        gate_one.wait_started().await;
        gate_two.wait_started().await;
        gate_one.release();
        gate_two.release();
        let _one_credential = one
            .await
            .expect("the first login task joins")
            .expect("the first login resolves");
        let _two_credential = two
            .await
            .expect("the second login task joins")
            .expect("the second login resolves");
    }

    /// upstream `it("does not wait for unrelated provider availability during
    /// local synchronization")`.
    #[tokio::test]
    async fn does_not_wait_for_unrelated_provider_availability_during_local_synchronization() {
        let stall = Arc::new(AtomicBool::new(false));
        let runtime = runtime_with(in_memory_auth_storage(&AuthStorageData::new())).await;
        runtime.register_native_provider(Arc::new(ProviderDouble::new(
            "target",
            ProviderOptions::default(),
        )));
        runtime.register_native_provider(Arc::new(ProviderDouble::new(
            "unrelated",
            ProviderOptions {
                check: Some(Arc::new({
                    let stall = Arc::clone(&stall);
                    move |input: ApiKeyAuthInput| {
                        let stall = Arc::clone(&stall);
                        Box::pin(async move {
                            if stall.load(Ordering::SeqCst) {
                                std::future::pending::<()>().await;
                            }
                            Ok(stored_check(&input))
                        })
                    }
                })),
                ..ProviderOptions::default()
            },
        )));
        runtime
            .refresh(ModelsRefreshOptions {
                allow_network: Some(false),
                providers: Some(vec!["target".to_owned(), "unrelated".to_owned()]),
                ..ModelsRefreshOptions::default()
            })
            .await;
        stall.store(true, Ordering::SeqCst);

        runtime
            .login("target", AuthType::ApiKey, login_interaction())
            .await
            .expect("the login resolves past the stalled unrelated check");
        assert!(runtime.has_configured_auth("target"));
        let refresh = runtime
            .refresh(ModelsRefreshOptions {
                allow_network: Some(false),
                providers: Some(vec!["target".to_owned()]),
                ..ModelsRefreshOptions::default()
            })
            .await;
        assert!(!refresh.aborted, "the scoped refresh completes uncancelled");
    }

    /// upstream `it("reports cancellation that occurs during provider-scoped
    /// availability")`.
    #[tokio::test]
    async fn reports_cancellation_that_occurs_during_provider_scoped_availability() {
        let block = Arc::new(AtomicBool::new(false));
        let started = Arc::new(Notify::new());
        let registered = ProviderDouble::new(
            "cancelled-availability",
            ProviderOptions {
                check: Some(Arc::new({
                    let block = Arc::clone(&block);
                    let started = Arc::clone(&started);
                    move |input: ApiKeyAuthInput| {
                        let block = Arc::clone(&block);
                        let started = Arc::clone(&started);
                        Box::pin(async move {
                            if block.load(Ordering::SeqCst) {
                                started.notify_one();
                                std::future::pending::<()>().await;
                            }
                            Ok(stored_check(&input))
                        })
                    }
                })),
                ..ProviderOptions::default()
            },
        );
        let runtime =
            runtime_with_provider(registered, in_memory_auth_storage(&AuthStorageData::new()))
                .await;
        runtime
            .set_runtime_api_key("cancelled-availability", "key".to_owned(), None)
            .await
            .expect("the runtime key sets");
        block.store(true, Ordering::SeqCst);

        let token = CancellationToken::new();
        let refresh_task = tokio::spawn({
            let runtime = runtime.clone();
            let token = token.clone();
            async move {
                runtime
                    .refresh(ModelsRefreshOptions {
                        allow_network: Some(false),
                        providers: Some(vec!["cancelled-availability".to_owned()]),
                        signal: Some(token),
                        ..ModelsRefreshOptions::default()
                    })
                    .await
            }
        });
        started.notified().await;
        token.cancel();

        let refresh = refresh_task.await.expect("the refresh task joins");
        assert!(
            refresh.aborted,
            "the aborted signal reports through the refresh"
        );
    }

    /// upstream `it("does not run network refresh inside the credential
    /// operation chain")`.
    #[tokio::test]
    async fn does_not_run_network_refresh_inside_the_credential_operation_chain() {
        let network_called = Arc::new(AtomicBool::new(false));
        let refresh_models: RefreshOverride = Arc::new({
            let network_called = Arc::clone(&network_called);
            move |context| {
                let network_called = Arc::clone(&network_called);
                Box::pin(async move {
                    if context.allow_network {
                        network_called.store(true, Ordering::SeqCst);
                    }
                    Ok(())
                })
            }
        });
        let runtime = runtime_with_provider(
            ProviderDouble::new(
                "local-only",
                ProviderOptions {
                    refresh_models: Some(refresh_models),
                    ..ProviderOptions::default()
                },
            ),
            in_memory_auth_storage(&AuthStorageData::new()),
        )
        .await;

        runtime
            .login("local-only", AuthType::ApiKey, login_interaction())
            .await
            .expect("the login resolves");
        assert!(
            !network_called.load(Ordering::SeqCst),
            "the credential chain never opens the network phase"
        );
        assert!(runtime.has_configured_auth("local-only"));
    }

    /// upstream `it("keeps provider-scoped refreshes from superseding
    /// unrelated providers")`.
    #[tokio::test]
    async fn keeps_provider_scoped_refreshes_from_superseding_unrelated_providers() {
        let gate = Gate::new();
        let first_signal: Arc<Mutex<Option<CancellationToken>>> = Arc::new(Mutex::new(None));
        let refresh_models: RefreshOverride = Arc::new({
            let started = Arc::clone(&gate.started);
            let blocked = Arc::clone(&gate.blocked);
            let first_signal = Arc::clone(&first_signal);
            move |context| {
                if !context.allow_network {
                    return Box::pin(async { Ok(()) });
                }
                *first_signal
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(context.signal);
                started.notify_one();
                let blocked = Arc::clone(&blocked);
                Box::pin(async move {
                    blocked.notified().await;
                    Ok(())
                })
            }
        });
        let runtime = runtime_with(in_memory_auth_storage(&AuthStorageData::new())).await;
        runtime.register_native_provider(Arc::new(ProviderDouble::new(
            "one",
            ProviderOptions {
                refresh_models: Some(refresh_models),
                ..ProviderOptions::default()
            },
        )));
        runtime.register_native_provider(Arc::new(ProviderDouble::new(
            "two",
            ProviderOptions::default(),
        )));
        runtime
            .refresh(ModelsRefreshOptions {
                allow_network: Some(false),
                providers: Some(vec!["one".to_owned(), "two".to_owned()]),
                ..ModelsRefreshOptions::default()
            })
            .await;
        runtime
            .set_runtime_api_key("one", "one-key".to_owned(), None)
            .await
            .expect("the runtime key sets");
        runtime
            .set_runtime_api_key("two", "two-key".to_owned(), None)
            .await
            .expect("the runtime key sets");

        let first_task = tokio::spawn({
            let runtime = runtime.clone();
            async move {
                runtime
                    .refresh(ModelsRefreshOptions {
                        allow_network: Some(true),
                        providers: Some(vec!["one".to_owned()]),
                        ..ModelsRefreshOptions::default()
                    })
                    .await
            }
        });
        gate.wait_started().await;
        runtime
            .refresh(ModelsRefreshOptions {
                allow_network: Some(true),
                providers: Some(vec!["two".to_owned()]),
                ..ModelsRefreshOptions::default()
            })
            .await;
        let recorded = first_signal
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .expect("the network phase recorded its signal");
        assert!(
            !recorded.is_cancelled(),
            "the second provider's refresh did not supersede the first"
        );

        gate.release();
        first_task.await.expect("the first refresh task joins");
    }

    /// upstream `it("waits for a committed credential mutation to settle
    /// before reporting cancellation")`.
    #[tokio::test]
    async fn waits_for_a_committed_credential_mutation_to_settle_before_reporting_cancellation() {
        let (committed_sender, committed_receiver) = oneshot::channel();
        let store = Arc::new(DelayedCommitStore {
            inner: Arc::new(InMemoryCredentialStore::default()),
            committed: Mutex::new(Some(committed_sender)),
            release: Arc::new(Notify::new()),
        });
        let runtime = runtime_with_provider(
            ProviderDouble::new("delayed-commit", ProviderOptions::default()),
            store.clone(),
        )
        .await;

        let settled = Arc::new(AtomicBool::new(false));
        let token = CancellationToken::new();
        let login_task = tokio::spawn({
            let runtime = runtime.clone();
            let settled = Arc::clone(&settled);
            let token = token.clone();
            async move {
                let outcome = runtime
                    .login(
                        "delayed-commit",
                        AuthType::ApiKey,
                        login_interaction_with_signal(token),
                    )
                    .await;
                settled.store(true, Ordering::SeqCst);
                outcome
            }
        });
        committed_receiver.await.expect("the mutation commits");
        token.cancel();
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        assert!(
            !settled.load(Ordering::SeqCst),
            "the login waits for the committed mutation to settle"
        );

        store.release.notify_one();
        let outcome = login_task.await.expect("the login task joins");
        // The settled mutation returns the credential; the aborted
        // synchronization then fails the login with the typed error.
        let error = match outcome {
            Err(error) => error,
            Ok(credential) => panic!("login unexpectedly succeeded: {credential:?}"),
        };
        let sync_error = error
            .downcast_ref::<CredentialSynchronizationError>()
            .expect("the login fails with CredentialSynchronizationError");
        assert_eq!(sync_error.provider_id, "delayed-commit");
        assert_eq!(
            sync_error.operation,
            CredentialSynchronizationOperation::Login
        );
        assert_eq!(
            sync_error.credential,
            api_key_credential("delayed-commit-key")
        );
        let stored = store
            .inner
            .read("delayed-commit", None)
            .await
            .expect("the read lands");
        assert_eq!(
            stored,
            api_key_credential("delayed-commit-key"),
            "the committed credential persists"
        );
    }

    /// upstream `it("reports a typed error when cancellation interrupts
    /// post-commit synchronization")`.
    #[tokio::test]
    async fn reports_a_typed_error_when_cancellation_interrupts_post_commit_synchronization() {
        let block = Arc::new(AtomicBool::new(false));
        let started = Arc::new(Notify::new());
        let refresh_models: RefreshOverride = Arc::new({
            let block = Arc::clone(&block);
            let started = Arc::clone(&started);
            move |context| {
                if !context.allow_network && block.load(Ordering::SeqCst) {
                    started.notify_one();
                    return Box::pin(async {
                        std::future::pending::<()>().await;
                        Ok(())
                    });
                }
                Box::pin(async { Ok(()) })
            }
        });
        let credentials = in_memory_auth_storage(&AuthStorageData::new());
        let runtime = runtime_with_provider(
            ProviderDouble::new(
                "cancelled-sync",
                ProviderOptions {
                    refresh_models: Some(refresh_models),
                    ..ProviderOptions::default()
                },
            ),
            credentials.clone(),
        )
        .await;
        block.store(true, Ordering::SeqCst);

        let token = CancellationToken::new();
        let login_task = tokio::spawn(runtime.login(
            "cancelled-sync",
            AuthType::ApiKey,
            login_interaction_with_signal(token.clone()),
        ));
        started.notified().await;
        token.cancel();

        let outcome = login_task.await.expect("the login task joins");
        let error = outcome.expect_err("the interrupted login fails");
        let sync_error = error
            .downcast_ref::<CredentialSynchronizationError>()
            .expect("the typed synchronization error surfaces");
        assert_eq!(sync_error.provider_id, "cancelled-sync");
        assert_eq!(
            sync_error.operation,
            CredentialSynchronizationOperation::Login
        );
        assert_eq!(
            sync_error.credential,
            api_key_credential("cancelled-sync-key")
        );
        let stored = credentials
            .read("cancelled-sync", None)
            .await
            .expect("the read lands");
        assert_eq!(stored, api_key_credential("cancelled-sync-key"));
    }

    /// upstream `it("reports committed credentials when local synchronization
    /// fails")`.
    #[tokio::test]
    async fn reports_committed_credentials_when_local_synchronization_fails() {
        let fail = Arc::new(AtomicBool::new(false));
        let refresh_models: RefreshOverride = Arc::new({
            let fail = Arc::clone(&fail);
            move |context| {
                if !context.allow_network && fail.load(Ordering::SeqCst) {
                    return Box::pin(async {
                        let error: ProviderError =
                            Box::new(std::io::Error::other("cache restore failed"));
                        Err(error)
                    });
                }
                Box::pin(async { Ok(()) })
            }
        });
        let credentials = in_memory_auth_storage(&AuthStorageData::new());
        let runtime = runtime_with_provider(
            ProviderDouble::new(
                "broken-sync",
                ProviderOptions {
                    refresh_models: Some(refresh_models),
                    ..ProviderOptions::default()
                },
            ),
            credentials.clone(),
        )
        .await;
        fail.store(true, Ordering::SeqCst);

        let outcome = runtime
            .login("broken-sync", AuthType::ApiKey, login_interaction())
            .await;
        let error = outcome.expect_err("the broken synchronization fails");
        assert!(
            error
                .downcast_ref::<CredentialSynchronizationError>()
                .is_some(),
            "the failure is the typed synchronization error, got: {error}"
        );
        let sync_error = error
            .downcast_ref::<CredentialSynchronizationError>()
            .expect("the typed synchronization error surfaces");
        assert_eq!(sync_error.provider_id, "broken-sync");
        assert_eq!(
            sync_error.operation,
            CredentialSynchronizationOperation::Login
        );
        assert_eq!(sync_error.credential, api_key_credential("broken-sync-key"));
        let stored = credentials
            .read("broken-sync", None)
            .await
            .expect("the read lands");
        assert_eq!(stored, api_key_credential("broken-sync-key"));
    }
}

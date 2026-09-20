//! Auth test doubles shared by the auth-core, CLI, and manual-flow suites,
//! the shape upstream's vitest suites build inline (`prompt`/`notify`
//! objects in `test/anthropic-oauth.test.ts`, `test/openrouter-oauth.test.ts`,
//! `test/openai-codex-oauth.test.ts`, and `test/kimi-coding-oauth.test.ts`) at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Each double trades upstream's per-test closures for configurable
//! builders: [`RecordingInteraction`] records what a login flow shows and
//! answers prompts from a script (hanging on the prompt's own cancellation
//! signal when the script is empty, the manual-prompt race the browser flows
//! drive), [`StubOAuthAuth`] and [`ApiKeyAuthStub`] script the
//! `refresh`/`to_auth`/`resolve` outcomes with call counters, and
//! [`FailingStore`] reproduces the storage-failure branch `resolve.ts` wraps.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio_util::sync::CancellationToken;

use pi_ai::auth::{
    ApiKeyAuth, ApiKeyCredential, ApiKeyResolveInput, AuthContext, AuthError, AuthEvent,
    AuthInteraction, AuthPrompt, AuthResult, BoxAuthFuture, Credential, CredentialInfo,
    CredentialStore, CredentialStoreError, ModelAuth, OAuthAuth, OAuthCredential,
    ProviderAuthInteraction, StoreError,
};
use pi_ai::types::BoxedFuture;

/// The refresh implementation a [`StubOAuthAuth`] scripts: the credential
/// under refresh and the operation's signal, resolving the replacement.
pub type StubRefreshFn = Arc<
    dyn Fn(&OAuthCredential, CancellationToken) -> BoxAuthFuture<Result<OAuthCredential, AuthError>>
        + Send
        + Sync,
>;

/// A `to_auth` derivation a [`StubOAuthAuth`] scripts, keyed by the stored
/// credential.
pub type StubToAuthFn = Arc<dyn Fn(&OAuthCredential) -> ModelAuth + Send + Sync>;

/// A `resolve` implementation an [`ApiKeyAuthStub`] scripts, resolving from
/// the input upstream's `ApiKeyAuth` handlers receive.
pub type StubResolveFn = Arc<
    dyn Fn(ApiKeyResolveInput) -> BoxAuthFuture<Result<Option<AuthResult>, AuthError>>
        + Send
        + Sync,
>;

/// The dynamic prompt answerer [`RecordingInteraction::with_dynamic`]
/// installs: consulted when the scripted answers are exhausted, so an answer
/// may depend on events the flow has already reported.
type DynamicAnswer =
    Arc<dyn Fn(&AuthPrompt) -> BoxAuthFuture<Result<String, AuthError>> + Send + Sync>;

/// The shared state of [`RecordingInteraction`].
#[derive(Default)]
struct RecordingInner {
    prompts: Mutex<Vec<AuthPrompt>>,
    events: Mutex<Vec<AuthEvent>>,
    answers: Mutex<VecDeque<Result<String, AuthError>>>,
    dynamic: Mutex<Option<DynamicAnswer>>,
}

impl RecordingInner {
    fn lock<T>(guard: &Mutex<T>) -> MutexGuard<'_, T> {
        guard
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// A recording login interaction, upstream's inline `prompt`/`notify` objects.
///
/// `prompt` records the prompt, then serves the next scripted answer; with
/// the script exhausted it consults the dynamic answerer when one is
/// installed, otherwise it waits on the prompt's own cancellation signal —
/// the manual-prompt race the browser flows drive — and fails as aborted
/// when the flow cancels it. A prompt without a signal and without an answer
/// fails as "Login cancelled". `notify` records every event in order.
#[derive(Clone, Default)]
pub struct RecordingInteraction {
    inner: Arc<RecordingInner>,
}

impl std::fmt::Debug for RecordingInteraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingInteraction")
            .field("prompts", &self.prompts().len())
            .field("events", &self.events().len())
            .finish_non_exhaustive()
    }
}

impl RecordingInteraction {
    /// An interaction with no scripted answers and no dynamic answerer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An interaction answering prompts from `answers` in order.
    #[must_use]
    pub fn with_answers<I>(answers: I) -> Self
    where
        I: IntoIterator<Item = Result<String, AuthError>>,
    {
        let interaction = Self::new();
        *RecordingInner::lock(&interaction.inner.answers) = answers.into_iter().collect();
        interaction
    }

    /// Install the dynamic answerer on an existing interaction — the
    /// answer may read the events already reported (a manual-prompt answer
    /// built from the URL an earlier `AuthUrl` event carried).
    pub fn set_dynamic(
        &self,
        answer: impl Fn(&AuthPrompt) -> BoxAuthFuture<Result<String, AuthError>> + Send + Sync + 'static,
    ) {
        *RecordingInner::lock(&self.inner.dynamic) = Some(Arc::new(answer));
    }

    /// An interaction whose unscripted prompts are answered by `answer`,
    /// which may read the events already reported (a manual-prompt answer
    /// built from the URL an earlier `AuthUrl` event carried).
    #[must_use]
    pub fn with_dynamic(
        answer: impl Fn(&AuthPrompt) -> BoxAuthFuture<Result<String, AuthError>> + Send + Sync + 'static,
    ) -> Self {
        let interaction = Self::new();
        *RecordingInner::lock(&interaction.inner.dynamic) = Some(Arc::new(answer));
        interaction
    }

    /// The prompts the flows have shown, in order.
    #[must_use]
    pub fn prompts(&self) -> Vec<AuthPrompt> {
        RecordingInner::lock(&self.inner.prompts).clone()
    }

    /// The events the flows have reported, in order.
    #[must_use]
    pub fn events(&self) -> Vec<AuthEvent> {
        RecordingInner::lock(&self.inner.events).clone()
    }

    /// The URL of the first reported [`AuthEvent::AuthUrl`], `None` before
    /// the flow reports one.
    #[must_use]
    pub fn auth_url(&self) -> Option<String> {
        self.events().into_iter().find_map(|event| match event {
            AuthEvent::AuthUrl { url, .. } => Some(url),
            _ => None,
        })
    }
}

impl AuthInteraction for RecordingInteraction {
    fn prompt(&self, prompt: AuthPrompt) -> BoxAuthFuture<Result<String, AuthError>> {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            RecordingInner::lock(&inner.prompts).push(prompt.clone());
            let scripted = RecordingInner::lock(&inner.answers).pop_front();
            if let Some(answer) = scripted {
                return answer;
            }
            let dynamic = RecordingInner::lock(&inner.dynamic).clone();
            if let Some(answer) = dynamic {
                return answer(&prompt).await;
            }
            match prompt.signal {
                Some(signal) => {
                    signal.cancelled().await;
                    Err(AuthError(
                        pi_ai::utils::abort::AbortError::MESSAGE.to_owned(),
                    ))
                }
                None => Err(AuthError("Login cancelled".to_owned())),
            }
        })
    }

    fn notify(&self, event: AuthEvent) {
        RecordingInner::lock(&self.inner.events).push(event);
    }
}

/// The shared state of [`StubOAuthAuth`].
struct StubOAuthInner {
    name: String,
    login: Result<OAuthCredential, AuthError>,
    refresh: StubRefreshFn,
    refresh_calls: AtomicUsize,
    to_auth: Option<StubToAuthFn>,
}

/// An [`OAuthAuth`] double the suites script: a fixed login outcome, a
/// configurable refresh counted per call, and an optional `to_auth`
/// override (the default derives the api key from the credential's access
/// token).
#[derive(Clone)]
pub struct StubOAuthAuth {
    inner: Arc<StubOAuthInner>,
}

impl std::fmt::Debug for StubOAuthAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StubOAuthAuth")
            .field("name", &self.inner.name)
            .field("refresh_calls", &self.refresh_calls())
            .finish_non_exhaustive()
    }
}

impl StubOAuthAuth {
    /// A flow named `name` whose login resolves `login` and whose refresh
    /// echoes the credential under refresh back unchanged.
    #[must_use]
    pub fn new(name: impl Into<String>, login: Result<OAuthCredential, AuthError>) -> Self {
        Self {
            inner: Arc::new(StubOAuthInner {
                name: name.into(),
                login,
                refresh: refresh_echo(),
                refresh_calls: AtomicUsize::new(0),
                to_auth: None,
            }),
        }
    }

    /// Script the refresh outcome. Builders rebuild the inner state, so
    /// configure the stub before wrapping it in an `Arc<dyn OAuthAuth>`; a
    /// builder call after that keeps the shared copy untouched.
    #[must_use]
    pub fn with_refresh(self, refresh: StubRefreshFn) -> Self {
        Self {
            inner: Arc::new(StubOAuthInner {
                name: self.inner.name.clone(),
                login: self.inner.login.clone(),
                refresh,
                refresh_calls: AtomicUsize::new(self.inner.refresh_calls.load(Ordering::Relaxed)),
                to_auth: self.inner.to_auth.clone(),
            }),
        }
    }

    /// Override `to_auth`'s derivation.
    #[must_use]
    pub fn with_to_auth(self, to_auth: StubToAuthFn) -> Self {
        Self {
            inner: Arc::new(StubOAuthInner {
                name: self.inner.name.clone(),
                login: self.inner.login.clone(),
                refresh: Arc::clone(&self.inner.refresh),
                refresh_calls: AtomicUsize::new(self.inner.refresh_calls.load(Ordering::Relaxed)),
                to_auth: Some(to_auth),
            }),
        }
    }

    /// How many times the flow called `refresh`.
    #[must_use]
    pub fn refresh_calls(&self) -> usize {
        self.inner.refresh_calls.load(Ordering::Relaxed)
    }
}

impl OAuthAuth for StubOAuthAuth {
    fn name(&self) -> &str {
        &self.inner.name
    }

    fn login(
        &self,
        _interaction: ProviderAuthInteraction,
    ) -> BoxAuthFuture<Result<OAuthCredential, AuthError>> {
        let login = self.inner.login.clone();
        Box::pin(std::future::ready(login))
    }

    fn refresh(
        &self,
        credential: &OAuthCredential,
        signal: CancellationToken,
    ) -> BoxAuthFuture<Result<OAuthCredential, AuthError>> {
        self.inner.refresh_calls.fetch_add(1, Ordering::Relaxed);
        (Arc::clone(&self.inner.refresh))(credential, signal)
    }

    fn to_auth(&self, credential: &OAuthCredential) -> ModelAuth {
        self.inner.to_auth.as_ref().map_or_else(
            || ModelAuth::api_key(credential.access.clone()),
            |to_auth| to_auth(credential),
        )
    }
}

/// A refresh that resolves `credential` unchanged, the rotation-free stub
/// the double refresh cases drive.
#[must_use]
pub fn refresh_echo() -> StubRefreshFn {
    refresh_returning_oauth(OAuthCredential::clone)
}

/// A refresh that resolves the OAuth credential `build` derives from the
/// credential under refresh.
#[must_use]
pub fn refresh_returning_oauth(
    build: impl Fn(&OAuthCredential) -> OAuthCredential + Send + Sync + 'static,
) -> StubRefreshFn {
    Arc::new(move |credential, _signal| {
        let refreshed = build(credential);
        Box::pin(std::future::ready(Ok(refreshed)))
    })
}

/// A refresh that resolves the prebuilt `credential`.
#[must_use]
pub fn refresh_returning(credential: OAuthCredential) -> StubRefreshFn {
    refresh_returning_oauth(move |_| credential.clone())
}

/// A refresh that rejects with `message`, the provider-failure branch.
#[must_use]
pub fn refresh_failing(message: impl Into<String>) -> StubRefreshFn {
    let message = message.into();
    Arc::new(move |_credential, _signal| {
        Box::pin(std::future::ready(Err(AuthError(message.clone()))))
    })
}

/// A refresh that sleeps `ms` under the tokio clock before resolving
/// `credential` — the paused-clock timeout cases drive it past the
/// resolution's 15-second refresh timeout.
#[must_use]
pub fn refresh_after_sleeping(ms: u64, credential: OAuthCredential) -> StubRefreshFn {
    Arc::new(move |_credential, _signal| {
        let credential = credential.clone();
        Box::pin(async move {
            tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
            Ok(credential)
        })
    })
}

/// The shared state of [`ApiKeyAuthStub`].
struct ApiKeyStubInner {
    name: String,
    resolve: StubResolveFn,
    last_credential: Mutex<Option<ApiKeyCredential>>,
}

/// An [`ApiKeyAuth`] double the resolve suites script: a `resolve`
/// implementation plus the last credential the resolution passed in, for the
/// stored-credential and merge assertions.
#[derive(Clone)]
pub struct ApiKeyAuthStub {
    inner: Arc<ApiKeyStubInner>,
}

impl std::fmt::Debug for ApiKeyAuthStub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKeyAuthStub")
            .field("name", &self.inner.name)
            .finish_non_exhaustive()
    }
}

impl ApiKeyAuthStub {
    /// A handler named `name` whose resolve runs `resolve`.
    #[must_use]
    pub fn new(name: impl Into<String>, resolve: StubResolveFn) -> Self {
        Self {
            inner: Arc::new(ApiKeyStubInner {
                name: name.into(),
                resolve,
                last_credential: Mutex::new(None),
            }),
        }
    }

    /// A resolve implementation returning the key as the api-key auth, no
    /// env, source "stub".
    #[must_use]
    pub fn resolving_key(key: impl Into<String>) -> StubResolveFn {
        let key = key.into();
        Arc::new(move |_input| {
            let result = AuthResult {
                auth: ModelAuth::api_key(key.clone()),
                env: None,
                source: Some("stub".to_owned()),
            };
            Box::pin(std::future::ready(Ok(Some(result))))
        })
    }

    /// A resolve implementation reporting the provider unconfigured.
    #[must_use]
    pub fn resolving_none() -> StubResolveFn {
        Arc::new(|_input| Box::pin(std::future::ready(Ok(None))))
    }

    /// A resolve implementation rejecting with `message`, the api-key
    /// failure branch.
    #[must_use]
    pub fn failing(message: impl Into<String>) -> StubResolveFn {
        let message = message.into();
        Arc::new(move |_input| Box::pin(std::future::ready(Err(AuthError(message.clone())))))
    }

    /// The credential the last `resolve` call received, `None` before any
    /// call or when the resolution ran without a stored credential.
    #[must_use]
    pub fn last_credential(&self) -> Option<ApiKeyCredential> {
        self.inner
            .last_credential
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl ApiKeyAuth for ApiKeyAuthStub {
    fn name(&self) -> &str {
        &self.inner.name
    }

    fn resolve(
        &self,
        input: ApiKeyResolveInput,
    ) -> BoxAuthFuture<Result<Option<AuthResult>, AuthError>> {
        // Record before the scripted resolve runs; the guard drops with the
        // block so the recorded value never rides the handler call.
        {
            let mut last_credential = self
                .inner
                .last_credential
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            last_credential.clone_from(&input.credential);
        }
        (Arc::clone(&self.inner.resolve))(input)
    }
}

/// A credential store whose every operation rejects with the same storage
/// failure, the store-error branch `resolve.ts` wraps as
/// `Credential store read failed for {provider}`.
#[derive(Debug, Default)]
pub struct FailingStore;

const FAILING_STORE_MESSAGE: &str = "storage backend exploded";

impl CredentialStore for FailingStore {
    fn read(
        &self,
        _provider_id: &str,
        _options: &pi_ai::auth::AuthOperationOptions,
    ) -> BoxedFuture<'static, Result<Option<Credential>, CredentialStoreError>> {
        Box::pin(std::future::ready(Err(CredentialStoreError::Storage(
            FAILING_STORE_MESSAGE.to_owned(),
        ))))
    }

    fn list(
        &self,
        _options: &pi_ai::auth::AuthOperationOptions,
    ) -> BoxedFuture<'static, Result<Vec<CredentialInfo>, CredentialStoreError>> {
        Box::pin(std::future::ready(Err(CredentialStoreError::Storage(
            FAILING_STORE_MESSAGE.to_owned(),
        ))))
    }

    fn modify(
        &self,
        _provider_id: &str,
        _modify: pi_ai::auth::StoreModifyFn,
        _options: &pi_ai::auth::AuthOperationOptions,
    ) -> BoxedFuture<'static, Result<Option<Credential>, StoreError>> {
        let error: StoreError = Box::new(CredentialStoreError::Storage(
            FAILING_STORE_MESSAGE.to_owned(),
        ));
        Box::pin(std::future::ready(Err(error)))
    }

    fn delete(
        &self,
        _provider_id: &str,
        _options: &pi_ai::auth::AuthOperationOptions,
    ) -> BoxedFuture<'static, Result<(), CredentialStoreError>> {
        Box::pin(std::future::ready(Err(CredentialStoreError::Storage(
            FAILING_STORE_MESSAGE.to_owned(),
        ))))
    }
}

/// An [`AuthContext`] over a fixed name→value map, the injectable
/// environment upstream's tests build from inline maps. File existence is
/// always false — these suites resolve from env values only.
#[derive(Debug, Default, Clone)]
pub struct MapAuthContext {
    env: HashMap<String, String>,
}

impl MapAuthContext {
    /// A context resolving the given name/value pairs.
    #[must_use]
    pub fn new<I, K, V>(env: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            env: env
                .into_iter()
                .map(|(key, value)| (key.into(), value.into()))
                .collect(),
        }
    }
}

impl AuthContext for MapAuthContext {
    fn env(&self, name: &str) -> BoxedFuture<'static, Option<String>> {
        let value = self.env.get(name).cloned();
        Box::pin(async move {
            // Blank values fall through, upstream's falsy-string semantics.
            value.filter(|value| !value.trim().is_empty())
        })
    }

    fn file_exists(&self, _path: &str) -> BoxedFuture<'static, bool> {
        Box::pin(std::future::ready(false))
    }
}

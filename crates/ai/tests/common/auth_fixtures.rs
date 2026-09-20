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
//! [`FailingStore`] reproduces the storage-failure branch the resolution wraps.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio_util::sync::CancellationToken;

use pi_ai::auth::types::{
    ApiKeyAuth, ApiKeyAuthInput, ApiKeyCredential, ApiKeyResolveFn, AuthContext, AuthError,
    AuthEvent, AuthInteraction, AuthPrompt, AuthResult, Credential, CredentialInfo,
    CredentialModifyFn, ModelAuth, OAuthAuth, OAuthCredentials, OAuthLoginFn, OAuthRefreshFn,
    OAuthToAuthFn, PromptFn, ProviderAuthInteraction,
};
use pi_ai::types::BoxedFuture;
use pi_ai::utils::abort::AbortError;

/// The failure the scripted doubles reject with: the message is the contract,
/// the shape upstream's thrown `Error`s carry.
#[must_use]
pub fn stub_error(message: impl Into<String>) -> AuthError {
    Box::new(StubError(message.into()))
}

#[derive(Debug)]
struct StubError(String);

impl std::fmt::Display for StubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StubError {}

/// The refresh implementation a [`StubOAuthAuth`] scripts: the credential
/// under refresh and the operation's signal, resolving the replacement.
pub type StubRefreshFn = Arc<
    dyn Fn(
            OAuthCredentials,
            CancellationToken,
        ) -> BoxedFuture<'static, Result<OAuthCredentials, AuthError>>
        + Send
        + Sync,
>;

/// A `to_auth` derivation a [`StubOAuthAuth`] scripts, keyed by the stored
/// credential.
pub type StubToAuthFn = Arc<dyn Fn(&OAuthCredentials) -> ModelAuth + Send + Sync>;

/// A `resolve` implementation an [`ApiKeyAuthStub`] scripts, resolving from
/// the input upstream's `ApiKeyAuth` handlers receive.
pub type StubResolveFn = Arc<
    dyn Fn(ApiKeyAuthInput) -> BoxedFuture<'static, Result<Option<AuthResult>, AuthError>>
        + Send
        + Sync,
>;

/// The dynamic prompt answerer [`RecordingInteraction::with_dynamic`]
/// installs: consulted when the scripted answers are exhausted, so an answer
/// may depend on events the flow has already reported.
type DynamicAnswer =
    Arc<dyn Fn(&AuthPrompt) -> BoxedFuture<'static, Result<String, AbortError>> + Send + Sync>;

/// The shared state of [`RecordingInteraction`].
#[derive(Default)]
struct RecordingInner {
    prompts: Mutex<Vec<AuthPrompt>>,
    events: Mutex<Vec<AuthEvent>>,
    answers: Mutex<VecDeque<Result<String, AbortError>>>,
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
/// fails as aborted too. `notify` records every event in order.
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
        I: IntoIterator<Item = Result<String, AbortError>>,
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
        answer: impl Fn(&AuthPrompt) -> BoxedFuture<'static, Result<String, AbortError>>
        + Send
        + Sync
        + 'static,
    ) {
        *RecordingInner::lock(&self.inner.dynamic) = Some(Arc::new(answer));
    }

    /// An interaction whose unscripted prompts are answered by `answer`,
    /// which may read the events already reported (a manual-prompt answer
    /// built from the URL an earlier `AuthUrl` event carried).
    #[must_use]
    pub fn with_dynamic(
        answer: impl Fn(&AuthPrompt) -> BoxedFuture<'static, Result<String, AbortError>>
        + Send
        + Sync
        + 'static,
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

    /// The recording double as the interaction the flows take: the merged
    /// core's [`AuthInteraction`] with the prompt and notify closures bound
    /// to this recorder.
    #[must_use]
    pub fn interaction(&self) -> AuthInteraction {
        let prompt: PromptFn = {
            let recorder = self.clone();
            Arc::new(move |prompt: AuthPrompt| {
                let recorder = recorder.clone();
                Box::pin(async move { recorder.answer(prompt).await })
            })
        };
        let notify = {
            let recorder = self.clone();
            Arc::new(move |event: AuthEvent| {
                RecordingInner::lock(&recorder.inner.events).push(event);
            })
        };
        AuthInteraction {
            signal: None,
            prompt,
            notify,
        }
    }

    /// Answer one prompt: record, then serve the script, then the dynamic
    /// answerer, then hang on the prompt's own signal.
    async fn answer(&self, prompt: AuthPrompt) -> Result<String, AbortError> {
        RecordingInner::lock(&self.inner.prompts).push(prompt.clone());
        let scripted = RecordingInner::lock(&self.inner.answers).pop_front();
        if let Some(answer) = scripted {
            return answer;
        }
        let dynamic = RecordingInner::lock(&self.inner.dynamic).clone();
        if let Some(answer) = dynamic {
            return answer(&prompt).await;
        }
        match prompt.signal {
            Some(signal) => {
                signal.cancelled().await;
                Err(AbortError)
            }
            None => Err(AbortError),
        }
    }
}

/// Wrap the recording double as the interaction argument the flows take,
/// over a never-cancelled signal — the default interaction the login-task
/// cases drive.
#[must_use]
pub fn provider_interaction(recording: &RecordingInteraction) -> ProviderAuthInteraction {
    ProviderAuthInteraction::from_interaction(recording.interaction(), CancellationToken::new())
}

/// An [`OAuthCredentials`] with the three wire fields set, the shape the
/// suites' fixtures mint.
#[must_use]
pub fn oauth_credentials(
    access: impl Into<String>,
    refresh: impl Into<String>,
    expires: i64,
) -> OAuthCredentials {
    OAuthCredentials {
        access: access.into(),
        refresh: refresh.into(),
        expires,
        extra: BTreeMap::new(),
    }
}

/// The login outcome a [`StubOAuthAuth`] scripts: a credential, or the error
/// message the login rejects with.
pub type StubLoginOutcome = Result<OAuthCredentials, String>;

/// The shared state of [`StubOAuthAuth`].
struct StubOAuthInner {
    name: String,
    login: StubLoginOutcome,
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
            .field("name", &self.name())
            .field("refresh_calls", &self.refresh_calls())
            .finish_non_exhaustive()
    }
}

impl StubOAuthAuth {
    /// A flow named `name` whose login resolves `login` and whose refresh
    /// echoes the credential under refresh back unchanged.
    #[must_use]
    pub fn new(name: impl Into<String>, login: StubLoginOutcome) -> Self {
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
    /// configure the stub before taking `auth()`; a builder call after that
    /// keeps the shared copy untouched.
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

    /// The display name the auth method carries.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// The stub wired into the merged core's callback-based [`OAuthAuth`]:
    /// a fixed login outcome, the counted refresh, and the derivation.
    #[must_use]
    pub fn auth(&self) -> OAuthAuth {
        let login: OAuthLoginFn = {
            let inner = Arc::clone(&self.inner);
            Arc::new(move |_interaction| {
                let outcome = match &inner.login {
                    Ok(credential) => Ok(credential.clone()),
                    Err(message) => Err(stub_error(message.clone())),
                };
                Box::pin(std::future::ready(outcome))
            })
        };
        let refresh: OAuthRefreshFn = {
            let inner = Arc::clone(&self.inner);
            Arc::new(move |credential, signal| {
                inner.refresh_calls.fetch_add(1, Ordering::Relaxed);
                (Arc::clone(&inner.refresh))(credential, signal)
            })
        };
        let to_auth: OAuthToAuthFn = {
            let inner = Arc::clone(&self.inner);
            Arc::new(move |credential| {
                let auth = inner.to_auth.as_ref().map_or_else(
                    || ModelAuth {
                        api_key: Some(credential.access.clone()),
                        ..ModelAuth::default()
                    },
                    |to_auth| to_auth(&credential),
                );
                Box::pin(async move { Ok(auth) })
            })
        };
        OAuthAuth {
            name: self.inner.name.clone(),
            is_subscription: Some(true),
            login_label: None,
            login,
            refresh,
            to_auth,
        }
    }
}

/// A refresh that resolves `credential` unchanged, the rotation-free stub
/// the double refresh cases drive.
#[must_use]
pub fn refresh_echo() -> StubRefreshFn {
    refresh_returning_oauth(OAuthCredentials::clone)
}

/// A refresh that resolves the OAuth credential `build` derives from the
/// credential under refresh.
#[must_use]
pub fn refresh_returning_oauth(
    build: impl Fn(&OAuthCredentials) -> OAuthCredentials + Send + Sync + 'static,
) -> StubRefreshFn {
    Arc::new(move |credential, _signal| {
        let refreshed = build(&credential);
        Box::pin(std::future::ready(Ok(refreshed)))
    })
}

/// A refresh that resolves the prebuilt `credential`.
#[must_use]
pub fn refresh_returning(credential: OAuthCredentials) -> StubRefreshFn {
    refresh_returning_oauth(move |_| credential.clone())
}

/// A refresh that rejects with `message`, the provider-failure branch.
#[must_use]
pub fn refresh_failing(message: impl Into<String>) -> StubRefreshFn {
    let message = message.into();
    Arc::new(move |_credential, _signal| {
        Box::pin(std::future::ready(Err(stub_error(message.clone()))))
    })
}

/// A refresh that sleeps `ms` under the tokio clock before resolving
/// `credential` — the paused-clock timeout cases drive it past the
/// resolution's 15-second refresh timeout.
#[must_use]
pub fn refresh_after_sleeping(ms: u64, credential: OAuthCredentials) -> StubRefreshFn {
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
            .field("name", &self.name())
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
                auth: ModelAuth {
                    api_key: Some(key.clone()),
                    ..ModelAuth::default()
                },
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
        Arc::new(move |_input| Box::pin(std::future::ready(Err(stub_error(message.clone())))))
    }

    /// The display name the auth method carries.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.inner.name
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

    /// The double wired into the merged core's [`ApiKeyAuth`] struct: the
    /// resolve closure records the credential it received, then delegates to
    /// the scripted implementation. No `login` and no `check`, so the
    /// handler reports ambient-only.
    #[must_use]
    pub fn auth(&self) -> ApiKeyAuth {
        let resolve: ApiKeyResolveFn = {
            let inner = Arc::clone(&self.inner);
            Arc::new(move |input: ApiKeyAuthInput| {
                {
                    let mut last_credential = inner
                        .last_credential
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    last_credential.clone_from(&input.credential);
                }
                (Arc::clone(&inner.resolve))(input)
            })
        };
        ApiKeyAuth {
            name: self.inner.name.clone(),
            login: None,
            check: None,
            resolve,
        }
    }
}

/// A credential store whose every operation rejects with the same storage
/// failure, the store-error branch the resolution wraps as
/// `Credential store read failed for {provider}`.
#[derive(Debug, Default)]
pub struct FailingStore;

const FAILING_STORE_MESSAGE: &str = "storage backend exploded";

fn failing_store_error() -> AuthError {
    stub_error(FAILING_STORE_MESSAGE)
}

impl pi_ai::auth::credential_store::CredentialStore for FailingStore {
    fn read<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        Box::pin(std::future::ready(Err(failing_store_error())))
    }

    fn list<'a>(
        &'a self,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<'a, Result<Vec<CredentialInfo>, AuthError>> {
        Box::pin(std::future::ready(Err(failing_store_error())))
    }

    fn modify<'a>(
        &'a self,
        _provider_id: &'a str,
        _f: CredentialModifyFn,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<'a, Result<Option<Credential>, AuthError>> {
        Box::pin(std::future::ready(Err(failing_store_error())))
    }

    fn delete<'a>(
        &'a self,
        _provider_id: &'a str,
        _options: Option<&'a pi_ai::auth::types::AuthOptions>,
    ) -> BoxedFuture<'a, Result<(), AuthError>> {
        Box::pin(std::future::ready(Err(failing_store_error())))
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
    fn env(&self, name: &str) -> Option<String> {
        // Blank values fall through, upstream's falsy-string semantics.
        self.env
            .get(name)
            .filter(|value| !value.trim().is_empty())
            .cloned()
    }

    fn file_exists(&self, _path: &str) -> bool {
        false
    }
}

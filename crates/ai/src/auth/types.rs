//! Auth data shapes and callback contracts, ported from
//! `packages/ai/src/auth/types.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements: callback interfaces (`ApiKeyAuth`, `OAuthAuth`,
//! `AuthInteraction`) hold `Arc`-boxed closures the way the crate's
//! `OnPayload`/`OnResponse` hooks do, so provider factories can assemble auth
//! from closures the way upstream assembles it from object literals; the
//! `AuthContext` environment/file probes are synchronous — the async forms
//! existed for browser bundling, which has no Rust counterpart; auth-layer
//! failures are `Box<dyn Error + Send + Sync>` values whose `Display` carries
//! the reason, because TypeScript lets any thrown value surface.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::types::{BoxedFuture, JsonValue, ProviderEnv, ProviderHeaders};

/// The error type auth callbacks and credential stores fail with.
pub type AuthError = Box<dyn std::error::Error + Send + Sync>;

/// Request auth for a single model request, upstream's `ModelAuth`. If a
/// value cannot be expressed as `apiKey`, `headers`, or `baseUrl`, it is
/// provider config, not auth.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelAuth {
    /// The API key to send.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Custom headers merged into API requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<ProviderHeaders>,
    /// Override of the request base URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

/// Stored api-key credential, upstream's `ApiKeyCredential`. `env` holds
/// provider-scoped environment/config values such as Cloudflare
/// account/gateway ids.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiKeyCredential {
    /// The stored key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Provider-scoped environment/config values stored with the credential.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<ProviderEnv>,
}

/// OAuth token data returned by extension compatibility flows, upstream's
/// `OAuthCredentials`.
///
/// The `extra` map holds arbitrary JSON values, which cannot carry `Eq`
/// (their numbers are floats), so the derived equality stops at `PartialEq`.
#[expect(
    clippy::derive_partial_eq_without_eq,
    reason = "the flattened extra map holds JSON floats, which have no total equality"
)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OAuthCredentials {
    /// The refresh token.
    pub refresh: String,
    /// The access token.
    pub access: String,
    /// Unix-millisecond expiry of `access`.
    pub expires: i64,
    /// Extension-owned extras such as `availableModelIds` or Radius'
    /// `gatewayConfig`, round-tripped verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, JsonValue>,
}

/// One type-tagged credential per provider — the shape of today's auth.json.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Credential {
    /// A stored api key.
    ApiKey(ApiKeyCredential),
    /// A stored OAuth token set.
    OAuth(OAuthCredentials),
}

impl Credential {
    /// The credential's type tag, upstream's `type` field.
    #[must_use]
    pub const fn auth_type(&self) -> AuthType {
        match self {
            Self::ApiKey(_) => AuthType::ApiKey,
            Self::OAuth(_) => AuthType::OAuth,
        }
    }

    /// The api key this credential carries, when it is an api-key credential.
    #[must_use]
    pub fn api_key(&self) -> Option<&str> {
        match self {
            Self::ApiKey(credential) => credential.key.as_deref(),
            Self::OAuth(_) => None,
        }
    }

    /// The OAuth token data, when this credential is OAuth.
    #[must_use]
    pub const fn as_oauth(&self) -> Option<&OAuthCredentials> {
        match self {
            Self::ApiKey(_) => None,
            Self::OAuth(credentials) => Some(credentials),
        }
    }

    /// The stored api-key credential, when this credential is an api key.
    #[must_use]
    pub const fn as_api_key_credential(&self) -> Option<&ApiKeyCredential> {
        match self {
            Self::ApiKey(credential) => Some(credential),
            Self::OAuth(_) => None,
        }
    }
}

/// Non-secret credential metadata for account/status enumeration, upstream's
/// `CredentialInfo`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialInfo {
    /// The provider id the credential is stored under.
    pub provider_id: String,
    /// The credential type.
    #[serde(rename = "type")]
    pub auth_type: AuthType,
}

/// Optional cancellation for public auth and credential operations, upstream's
/// `AuthOperationOptions`.
#[derive(Clone, Debug, Default)]
pub struct AuthOptions {
    /// Cancellation for the operation.
    pub signal: Option<CancellationToken>,
}

/// Environment access for auth resolution, upstream's `AuthContext`.
/// Injectable for tests.
///
/// Porting restatement: the async forms existed so browser bundles could omit
/// the Node implementations; both probes are synchronous here.
pub trait AuthContext: Send + Sync {
    /// Read one environment variable, `None` when unset or blank.
    fn env(&self, name: &str) -> Option<String>;

    /// Check whether a file exists, expanding a leading `~`.
    fn file_exists(&self, path: &str) -> bool;
}

/// Result of resolving auth for a model, upstream's `AuthResult`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthResult {
    /// The request auth.
    pub auth: ModelAuth,
    /// Provider-scoped environment/config values resolved from credentials
    /// and ambient context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<ProviderEnv>,
    /// Human-readable label for status UI: "`ANTHROPIC_API_KEY`", "OAuth",
    /// "~/.aws/credentials".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// The auth status of a provider, upstream's `AuthCheck`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthCheck {
    /// Label for status UI: the credential source or "OAuth".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Which auth method answered.
    #[serde(rename = "type")]
    pub auth_type: AuthType,
}

/// The two auth methods a provider can carry, upstream's `AuthType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthType {
    /// Api-key auth.
    ApiKey,
    /// OAuth auth.
    OAuth,
}

impl std::fmt::Display for AuthType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ApiKey => "api_key",
            Self::OAuth => "oauth",
        })
    }
}

/// One selectable option of a [`AuthPromptKind::Select`] prompt, upstream's
/// select option.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthPromptOption {
    /// The value `prompt` returns when selected.
    pub id: String,
    /// The displayed label.
    pub label: String,
    /// Optional longer description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// The prompt body shown to the user during login, upstream's
/// `AuthPrompt`'s union member.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum AuthPromptKind {
    /// Free-form text entry.
    Text {
        /// The prompt message.
        message: String,
        /// Optional placeholder.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        placeholder: Option<String>,
    },
    /// Secret entry (masked input).
    Secret {
        /// The prompt message.
        message: String,
        /// Optional placeholder.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        placeholder: Option<String>,
    },
    /// Selection from listed options; `prompt` returns the option id.
    Select {
        /// The prompt message.
        message: String,
        /// The selectable options.
        options: Vec<AuthPromptOption>,
    },
    /// A manual-code prompt raced against an out-of-band callback.
    ManualCode {
        /// The prompt message.
        message: String,
        /// Optional placeholder.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        placeholder: Option<String>,
    },
}

/// Prompt shown to the user during login, upstream's `AuthPrompt`. `signal`
/// lets the flow cancel a pending prompt when an out-of-band event resolves
/// the step.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthPrompt {
    /// Per-prompt cancellation.
    #[serde(skip)]
    pub signal: Option<CancellationToken>,
    /// The prompt body.
    #[serde(flatten)]
    pub kind: AuthPromptKind,
}

/// An info link attached to an [`AuthEvent::Info`], upstream's
/// `AuthInfoLink`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthInfoLink {
    /// The link target.
    pub url: String,
    /// Optional display label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Login-flow event pushed to the host UI, upstream's `AuthEvent`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum AuthEvent {
    /// Informational message with optional links.
    Info {
        /// The message.
        message: String,
        /// Optional links.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        links: Option<Vec<AuthInfoLink>>,
    },
    /// A URL the user must visit to continue.
    AuthUrl {
        /// The URL.
        url: String,
        /// Optional instructions shown with the URL.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
    },
    /// Device-code login parameters.
    DeviceCode {
        /// The code the user enters.
        user_code: String,
        /// The verification URL.
        verification_uri: String,
        /// Poll interval in seconds.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        interval_seconds: Option<u64>,
        /// Seconds until the device code expires.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expires_in_seconds: Option<u64>,
    },
    /// Progress message.
    Progress {
        /// The message.
        message: String,
    },
}

/// The prompt callback, upstream's `AuthInteraction.prompt`.
///
/// Returns the entered/selected string (`Select` returns the option id);
/// rejects with [`AbortError`](crate::utils::abort::AbortError) on
/// cancel/abort.
pub type PromptFn = Arc<
    dyn Fn(AuthPrompt) -> BoxedFuture<'static, Result<String, crate::utils::abort::AbortError>>
        + Send
        + Sync,
>;

/// The notify callback, upstream's `AuthInteraction.notify`.
pub type NotifyFn = Arc<dyn Fn(AuthEvent) + Send + Sync>;

/// Login interaction callbacks serving both api-key and OAuth flows, upstream's
/// `AuthInteraction`. `signal` aborts the whole login flow; per-prompt
/// cancellation uses [`AuthPrompt::signal`].
#[derive(Clone)]
pub struct AuthInteraction {
    /// Cancellation for the whole login flow.
    pub signal: Option<CancellationToken>,
    /// The prompt callback.
    pub prompt: PromptFn,
    /// The notify callback.
    pub notify: NotifyFn,
}

impl std::fmt::Debug for AuthInteraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthInteraction")
            .field("signal", &self.signal)
            .finish_non_exhaustive()
    }
}

/// Normalized interaction passed to provider login implementations, upstream's
/// `ProviderAuthInteraction`: the caller's interaction with a guaranteed
/// signal.
#[derive(Clone)]
pub struct ProviderAuthInteraction {
    /// The operation's cancellation token, always present.
    pub signal: CancellationToken,
    /// The prompt callback.
    pub prompt: PromptFn,
    /// The notify callback.
    pub notify: NotifyFn,
}

impl std::fmt::Debug for ProviderAuthInteraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderAuthInteraction")
            .finish_non_exhaustive()
    }
}

impl ProviderAuthInteraction {
    /// Fill in the operation-local token: the interaction's own when present,
    /// the supplied one otherwise.
    #[must_use]
    pub fn from_interaction(interaction: AuthInteraction, signal: CancellationToken) -> Self {
        let _ = (&interaction.signal, &signal);
        Self {
            signal: interaction.signal.unwrap_or(signal),
            prompt: interaction.prompt,
            notify: interaction.notify,
        }
    }
}

/// The shared inputs an api-key auth callback receives, upstream's inline
/// `{ ctx, credential, signal }` argument.
#[derive(Clone)]
pub struct ApiKeyAuthInput {
    /// The environment access context.
    pub ctx: Arc<dyn AuthContext>,
    /// The stored api-key credential, when one is stored.
    pub credential: Option<ApiKeyCredential>,
    /// Cancellation for the operation.
    pub signal: CancellationToken,
}

impl std::fmt::Debug for ApiKeyAuthInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKeyAuthInput")
            .field("credential", &self.credential)
            .finish_non_exhaustive()
    }
}

/// The login closure of [`ApiKeyAuth`].
pub type ApiKeyLoginFn = Arc<
    dyn Fn(ProviderAuthInteraction) -> BoxedFuture<'static, Result<ApiKeyCredential, AuthError>>
        + Send
        + Sync,
>;

/// The check closure of [`ApiKeyAuth`].
pub type ApiKeyCheckFn = Arc<
    dyn Fn(ApiKeyAuthInput) -> BoxedFuture<'static, Result<Option<AuthCheck>, AuthError>>
        + Send
        + Sync,
>;

/// The resolve closure of [`ApiKeyAuth`].
pub type ApiKeyResolveFn = Arc<
    dyn Fn(ApiKeyAuthInput) -> BoxedFuture<'static, Result<Option<AuthResult>, AuthError>>
        + Send
        + Sync,
>;

/// Api-key auth, upstream's `ApiKeyAuth`: stored key/provider env plus ambient
/// sources (env vars, AWS profiles, ADC files). Ambient-only providers omit
/// `login`.
#[derive(Clone)]
pub struct ApiKeyAuth {
    /// Display name, e.g. "Anthropic API key".
    pub name: String,
    /// Interactive setup (prompt for key/provider env). `None` = ambient-only.
    pub login: Option<ApiKeyLoginFn>,
    /// Optional side-effect-free availability check. Use this when `resolve()`
    /// may execute commands or perform other request-time work. `None` means
    /// Models checks availability by resolving auth.
    pub check: Option<ApiKeyCheckFn>,
    /// Resolve auth from the stored credential and/or ambient sources, merging
    /// per field (`credential.key ?? env("...")`,
    /// `credential.env?.NAME ?? env("...")`). `None` = not configured.
    pub resolve: ApiKeyResolveFn,
}

impl std::fmt::Debug for ApiKeyAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKeyAuth")
            .field("name", &self.name)
            .field("login", &self.login.is_some())
            .field("check", &self.check.is_some())
            .finish_non_exhaustive()
    }
}

/// The login closure of [`OAuthAuth`].
pub type OAuthLoginFn = Arc<
    dyn Fn(ProviderAuthInteraction) -> BoxedFuture<'static, Result<OAuthCredentials, AuthError>>
        + Send
        + Sync,
>;

/// The refresh closure of [`OAuthAuth`].
pub type OAuthRefreshFn = Arc<
    dyn Fn(
            OAuthCredentials,
            CancellationToken,
        ) -> BoxedFuture<'static, Result<OAuthCredentials, AuthError>>
        + Send
        + Sync,
>;

/// The toAuth closure of [`OAuthAuth`].
pub type OAuthToAuthFn = Arc<
    dyn Fn(OAuthCredentials) -> BoxedFuture<'static, Result<ModelAuth, AuthError>> + Send + Sync,
>;

/// OAuth auth, upstream's `OAuthAuth`.
///
/// The `refresh`/`toAuth` split lets Models own the locked refresh pattern:
/// `refresh` produces a credential, `toAuth` derives request auth from
/// whatever credential ends up stored.
#[derive(Clone)]
pub struct OAuthAuth {
    /// Display name, e.g. "Anthropic (Claude Pro/Max)".
    pub name: String,
    /// Whether access through this auth method is backed by a provider
    /// subscription.
    pub is_subscription: Option<bool>,
    /// Selector label for the OAuth login option, e.g.
    /// "Sign in with `SuperGrok` or X Premium".
    pub login_label: Option<String>,
    /// Run the interactive login flow.
    pub login: OAuthLoginFn,
    /// Exchange the refresh token. Network call; fails on `invalid_grant`
    /// etc. Models runs this under the store lock.
    pub refresh: OAuthRefreshFn,
    /// Side-effect-free derivation of request auth from a valid credential.
    /// Covers per-credential baseUrl (GitHub Copilot).
    pub to_auth: OAuthToAuthFn,
}

impl std::fmt::Debug for OAuthAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthAuth")
            .field("name", &self.name)
            .field("is_subscription", &self.is_subscription)
            .field("login_label", &self.login_label)
            .finish_non_exhaustive()
    }
}

/// Provider auth, upstream's `ProviderAuth`.
///
/// At least one of `apiKey`/`oauth` must be present: even ambient-credential
/// providers and keyless local servers provide `apiKey` auth whose
/// `resolve()` reports whether the provider is configured.
#[derive(Clone, Debug, Default)]
pub struct ProviderAuth {
    /// The api-key auth method.
    pub api_key: Option<ApiKeyAuth>,
    /// The OAuth auth method.
    pub oauth: Option<OAuthAuth>,
}

/// The serialized write closure of
/// [`CredentialStore::modify`](crate::auth::credential_store::CredentialStore::modify).
///
/// `f` sees the current credential and returns the new one, or `None` to
/// leave the entry unchanged.
pub type CredentialModifyFn = Box<
    dyn FnOnce(Option<Credential>) -> BoxedFuture<'static, Result<Option<Credential>, AuthError>>
        + Send,
>;

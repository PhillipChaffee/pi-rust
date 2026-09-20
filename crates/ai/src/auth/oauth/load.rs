//! The OAuth flow constructors, ported from
//! `packages/ai/src/auth/oauth/load.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements this module records:
//!
//! - Upstream loads each flow module through a variable specifier so
//!   bundlers cannot follow the import into Node-only flow code, and
//!   `registerBundledOAuthFlowLoaders` swaps in statically bundled loaders
//!   for standalone Bun binaries. Rust modules are statically linked, so
//!   both mechanisms collapse into these direct constructors; the
//!   providers' `lazyOAuth` wrappers collapse with them.
//! - Each constructor wires the process-default [`HttpClient`], the wall
//!   clock, and (for GitHub Copilot) the known-model membership from the
//!   registry the registry ticket lands.

use std::sync::Arc;

use crate::auth::oauth::anthropic::AnthropicOAuth;
use crate::auth::oauth::github_copilot::{GitHubCopilotOAuth, KnownModels};
use crate::auth::oauth::kimi_coding::KimiCodingOAuth;
use crate::auth::oauth::openai_codex::OpenAICodexOAuth;
use crate::auth::oauth::openrouter::OpenRouterOAuth;
use crate::auth::oauth::radius::RadiusOAuth;
use crate::auth::oauth::xai::XaiOAuth;
use crate::auth::types::{AuthError, OAuthAuth};
use crate::http::HttpClient;

/// The statically-linked Anthropic flow.
///
/// # Errors
/// Rejects when the PKCE randomness source fails.
pub fn load_anthropic_oauth() -> Result<Arc<dyn OAuthAuth>, AuthError> {
    Ok(Arc::new(AnthropicOAuth::new(
        crate::http::default_http_client(),
        Arc::new(crate::auth::clock::SystemClock),
    )))
}

/// The statically-linked OpenAI Codex flow.
///
/// # Errors
/// Rejects when the PKCE randomness source fails.
pub fn load_openai_codex_oauth() -> Result<Arc<dyn OAuthAuth>, AuthError> {
    Ok(Arc::new(OpenAICodexOAuth::new(
        crate::http::default_http_client(),
        Arc::new(crate::auth::clock::SystemClock),
    )))
}

/// The statically-linked GitHub Copilot flow, with the known-model
/// membership its policy updates check.
///
/// # Errors
/// Rejects when the PKCE randomness source fails.
pub fn load_github_copilot_oauth(
    known_models: KnownModels,
) -> Result<Arc<dyn OAuthAuth>, AuthError> {
    Ok(Arc::new(GitHubCopilotOAuth::new(
        crate::http::default_http_client(),
        Arc::new(crate::auth::clock::SystemClock),
        known_models,
    )))
}

/// The statically-linked OpenRouter flow.
#[must_use]
pub fn load_openrouter_oauth() -> Arc<dyn OAuthAuth> {
    Arc::new(OpenRouterOAuth::new(crate::http::default_http_client()))
}

/// The statically-linked Kimi Code flow.
#[must_use]
pub fn load_kimi_coding_oauth() -> Arc<dyn OAuthAuth> {
    Arc::new(KimiCodingOAuth::new(crate::http::default_http_client()))
}

/// The statically-linked xAI flow.
#[must_use]
pub fn load_xai_oauth() -> Arc<dyn OAuthAuth> {
    Arc::new(XaiOAuth::new(
        crate::http::default_http_client(),
        Arc::new(crate::auth::clock::SystemClock),
    ))
}

/// The statically-linked Radius flow for one gateway.
///
/// # Errors
/// Rejects when the PKCE randomness source fails.
pub fn load_radius_oauth(name: &str, gateway: &str) -> Result<Arc<dyn OAuthAuth>, AuthError> {
    Ok(Arc::new(RadiusOAuth::new(
        name.to_owned(),
        gateway.to_owned(),
        crate::http::default_http_client(),
        Arc::new(crate::auth::clock::SystemClock),
    )))
}

/// The shared constructor wiring every flow carries: the [`HttpClient`]
/// seam and the epoch clock, injectable where the tests stub them.
pub type FlowHttpClient = Arc<dyn HttpClient>;

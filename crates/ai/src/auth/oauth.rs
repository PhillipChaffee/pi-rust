//! The OAuth flow loader seam, ported from
//! `packages/ai/src/auth/oauth/load.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatement: upstream's loaders dynamic-import Node-only flow
//! modules (callback servers, PKCE) on first use; Rust has no bundler to hide
//! from, so the flows land with the auth ticket and these loaders return the
//! not-yet-ported stub whose callbacks fail when invoked. The signatures
//! match the pinned upstream surface so the flows drop in unchanged.

use std::sync::Arc;

use crate::auth::types::{OAuthAuth, ProviderAuthInteraction};
use crate::types::BoxedFuture;

fn not_ported(name: &str, provider: &str) -> OAuthAuth {
    let provider = Arc::new(provider.to_owned());
    let login: crate::auth::types::OAuthLoginFn = {
        let provider = Arc::clone(&provider);
        Arc::new(move |interaction: ProviderAuthInteraction| {
            let message = format!(
                "OAuth login for {provider} has not been ported yet; the flows land with the auth ticket"
            );
            Box::pin(async move {
                let _ = interaction;
                Err(crate::auth::helpers::oauth_stub_error(message))
            })
        })
    };
    let refresh: crate::auth::types::OAuthRefreshFn = {
        let provider = Arc::clone(&provider);
        Arc::new(move |_credential, _signal| {
            let message = format!(
                "OAuth refresh for {provider} has not been ported yet; the flows land with the auth ticket"
            );
            Box::pin(async move { Err(crate::auth::helpers::oauth_stub_error(message)) })
        })
    };
    let to_auth: crate::auth::types::OAuthToAuthFn = {
        let provider = Arc::clone(&provider);
        Arc::new(move |_credential| {
            let message = format!(
                "OAuth auth derivation for {provider} has not been ported yet; the flows land with the auth ticket"
            );
            Box::pin(async move { Err(crate::auth::helpers::oauth_stub_error(message)) })
        })
    };
    OAuthAuth {
        name: name.to_owned(),
        is_subscription: None,
        login_label: None,
        login,
        refresh,
        to_auth,
    }
}

/// Loads the Anthropic (Claude Pro/Max) OAuth flow, upstream's
/// `loadAnthropicOAuth`.
#[must_use]
pub fn load_anthropic_oauth() -> BoxedFuture<'static, OAuthAuth> {
    Box::pin(async move { not_ported("Anthropic (Claude Pro/Max)", "anthropic") })
}

/// Loads the OpenAI (`ChatGPT` Plus/Pro) OAuth flow, upstream's
/// `loadOpenAICodexOAuth`.
#[must_use]
pub fn load_openai_codex_oauth() -> BoxedFuture<'static, OAuthAuth> {
    Box::pin(async move { not_ported("OpenAI (ChatGPT Plus/Pro)", "openai-codex") })
}

/// Loads the GitHub Copilot OAuth flow, upstream's `loadGitHubCopilotOAuth`.
#[must_use]
pub fn load_github_copilot_oauth() -> BoxedFuture<'static, OAuthAuth> {
    Box::pin(async move { not_ported("GitHub Copilot", "github-copilot") })
}

/// Loads the OpenRouter OAuth flow, upstream's `loadOpenRouterOAuth`.
#[must_use]
pub fn load_openrouter_oauth() -> BoxedFuture<'static, OAuthAuth> {
    Box::pin(async move { not_ported("OpenRouter OAuth", "openrouter") })
}

/// Loads the Kimi Code (subscription) OAuth flow, upstream's
/// `loadKimiCodingOAuth`.
#[must_use]
pub fn load_kimi_coding_oauth() -> BoxedFuture<'static, OAuthAuth> {
    Box::pin(async move { not_ported("Kimi Code (subscription)", "kimi-coding") })
}

/// Loads the xAI (Grok/X subscription) OAuth flow, upstream's `loadXaiOAuth`.
#[must_use]
pub fn load_xai_oauth() -> BoxedFuture<'static, OAuthAuth> {
    Box::pin(async move { not_ported("xAI (Grok/X subscription)", "xai") })
}

/// The options of [`load_radius_oauth`], upstream's loadRadiusOAuth argument.
#[derive(Debug)]
pub struct RadiusOAuthOptions {
    /// Display name.
    pub name: String,
    /// Gateway URL the flow authenticates against.
    pub gateway: String,
}

/// Loads the Radius OAuth flow, upstream's `loadRadiusOAuth`.
#[must_use]
pub fn load_radius_oauth(options: &RadiusOAuthOptions) -> BoxedFuture<'static, OAuthAuth> {
    let _ = options;
    Box::pin(async move { not_ported("Radius", "radius") })
}

//! PKCE utilities, ported from
//! `packages/ai/src/auth/oauth/pkce.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The Web Crypto calls port to `getrandom` and `sha2`: a 32-byte verifier
//! and the SHA-256 challenge over its UTF-8 bytes, both encoded
//! base64url without padding.

use sha2::{Digest, Sha256};

use crate::auth::types::AuthError;

/// A PKCE pair: the secret verifier and its S256 challenge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pkce {
    /// The code verifier sent in the token exchange.
    pub verifier: String,
    /// The S256 challenge sent in the authorize URL.
    pub challenge: String,
}

/// Generate a PKCE verifier and its S256 challenge.
///
/// # Errors
/// Rejects when the OS randomness source fails, upstream's `crypto
/// unavailable`.
pub fn generate_pkce() -> Result<Pkce, AuthError> {
    let mut verifier_bytes = [0_u8; 32];
    getrandom::fill(&mut verifier_bytes)
        .map_err(|error| AuthError(format!("getrandom failed: {error}")))?;
    let verifier = base64url_no_pad(&verifier_bytes);
    let challenge = base64url_no_pad(&Sha256::digest(verifier.as_bytes()));
    Ok(Pkce {
        verifier,
        challenge,
    })
}

/// Encode bytes as base64url without padding, upstream's
/// `btoa(binary).replace(+, -).replace(/, _).replace(/=/, "")`.
#[must_use]
pub(crate) fn base64url_no_pad(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// Decode base64url or standard base64, leniently — the port of Node's
/// `atob`, which tolerates the URL-safe alphabet.
pub(crate) fn decode_base64_lenient(text: &str) -> Result<Vec<u8>, AuthError> {
    use base64::Engine as _;
    let standard = base64::engine::general_purpose::STANDARD;
    let url_safe = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    standard
        .decode(text)
        .or_else(|_| url_safe.decode(text.trim_end_matches('=')))
        .map_err(|_| AuthError(format!("invalid base64: {text}")))
}

/// Decode a base64url-or-standard segment and parse it as JSON, the JWT
/// payload path.
pub(crate) fn decode_json_segment<T: serde::de::DeserializeOwned>(
    segment: &str,
) -> Result<T, AuthError> {
    let bytes = decode_base64_lenient(segment)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| AuthError(format!("invalid JSON payload: {error}")))
}

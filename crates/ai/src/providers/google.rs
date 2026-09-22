//! The `google` provider factory, ported from
//! `packages/ai/src/providers/google.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;

env_key_provider!(
    /// The `Google` provider, upstream's `googleProvider()`.
    google_provider,
    "google",
    "Google",
    Some("https://generativelanguage.googleapis.com/v1beta"),
    "Gemini API key",
    ["GEMINI_API_KEY"],
    crate::models::ProviderApi::Single(crate::api::google_generative_ai()),
);

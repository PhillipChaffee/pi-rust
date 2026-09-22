//! The OpenCode Zen provider factory, ported from
//! `packages/ai/src/providers/opencode.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::providers::factory::env_key_provider;
use crate::providers::opencode_headers::with_opencode_session_header;

env_key_provider!(
    /// The OpenCode Zen provider, upstream's `opencodeProvider()`.
    opencode_provider,
    "opencode",
    "OpenCode Zen",
    None,
    "OpenCode API key",
    ["OPENCODE_API_KEY"],
    crate::models::ProviderApi::ByApi(crate::api::ApiMap::from([ ( "anthropic-messages".to_owned(), with_opencode_session_header(crate::api::anthropic_messages()), ), ( "google-generative-ai".to_owned(), with_opencode_session_header(crate::api::google_generative_ai()), ), ( "openai-completions".to_owned(), with_opencode_session_header(crate::api::openai_completions()), ), ( "openai-responses".to_owned(), with_opencode_session_header(crate::api::openai_responses()), ), ])),
);

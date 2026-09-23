//! The deprecated per-API stream aliases, ported from
//! `packages/ai/src/legacy-api-aliases.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatement: upstream's aliases capture each api factory's
//! `stream`/`streamSimple` once at module load; the free functions of the
//! wire-API modules are that same surface, so each alias is a re-export of
//! the matching adapter entry point and carries no code of its own. The
//! bedrock-converse-stream and pi-messages apis have no aliases upstream.

/// Deprecated. Use [`crate::api::anthropic_messages::stream`] or the
/// [`crate::api::anthropic_messages()`] factory's `stream`.
#[deprecated(note = "use `pi_ai::api::anthropic_messages::stream`")]
pub use crate::api::anthropic_messages::stream as stream_anthropic;
/// Deprecated. Use [`crate::api::anthropic_messages::stream_simple`] or the
/// [`crate::api::anthropic_messages()`] factory's `stream_simple`.
#[deprecated(note = "use `pi_ai::api::anthropic_messages::stream_simple`")]
pub use crate::api::anthropic_messages::stream_simple as stream_simple_anthropic;

/// Deprecated. Use [`crate::api::azure_openai_responses::stream`] or the
/// [`crate::api::azure_openai_responses()`] factory's `stream`.
#[deprecated(note = "use `pi_ai::api::azure_openai_responses::stream`")]
pub use crate::api::azure_openai_responses::stream as stream_azure_openai_responses;
/// Deprecated. Use [`crate::api::azure_openai_responses::stream_simple`] or
/// the [`crate::api::azure_openai_responses()`] factory's `stream_simple`.
#[deprecated(note = "use `pi_ai::api::azure_openai_responses::stream_simple`")]
pub use crate::api::azure_openai_responses::stream_simple as stream_simple_azure_openai_responses;

/// Deprecated. Use [`crate::api::google_generative_ai::stream`] or the
/// [`crate::api::google_generative_ai()`] factory's `stream`.
#[deprecated(note = "use `pi_ai::api::google_generative_ai::stream`")]
pub use crate::api::google_generative_ai::stream as stream_google;
/// Deprecated. Use [`crate::api::google_generative_ai::stream_simple`] or the
/// [`crate::api::google_generative_ai()`] factory's `stream_simple`.
#[deprecated(note = "use `pi_ai::api::google_generative_ai::stream_simple`")]
pub use crate::api::google_generative_ai::stream_simple as stream_simple_google;

/// Deprecated. Use [`crate::api::google_vertex::stream`] or the
/// [`crate::api::google_vertex()`] factory's `stream`.
#[deprecated(note = "use `pi_ai::api::google_vertex::stream`")]
pub use crate::api::google_vertex::stream as stream_google_vertex;
/// Deprecated. Use [`crate::api::google_vertex::stream_simple`] or the
/// [`crate::api::google_vertex()`] factory's `stream_simple`.
#[deprecated(note = "use `pi_ai::api::google_vertex::stream_simple`")]
pub use crate::api::google_vertex::stream_simple as stream_simple_google_vertex;

/// Deprecated. Use [`crate::api::mistral_conversations::stream`] or the
/// [`crate::api::mistral_conversations()`] factory's `stream`.
#[deprecated(note = "use `pi_ai::api::mistral_conversations::stream`")]
pub use crate::api::mistral_conversations::stream as stream_mistral;
/// Deprecated. Use [`crate::api::mistral_conversations::stream_simple`] or
/// the [`crate::api::mistral_conversations()`] factory's `stream_simple`.
#[deprecated(note = "use `pi_ai::api::mistral_conversations::stream_simple`")]
pub use crate::api::mistral_conversations::stream_simple as stream_simple_mistral;

/// Deprecated. Use [`crate::api::openai_codex_responses::stream`] or the
/// [`crate::api::openai_codex_responses()`] factory's `stream`.
#[deprecated(note = "use `pi_ai::api::openai_codex_responses::stream`")]
pub use crate::api::openai_codex_responses::stream as stream_openai_codex_responses;
/// Deprecated. Use [`crate::api::openai_codex_responses::stream_simple`] or
/// the [`crate::api::openai_codex_responses()`] factory's `stream_simple`.
#[deprecated(note = "use `pi_ai::api::openai_codex_responses::stream_simple`")]
pub use crate::api::openai_codex_responses::stream_simple as stream_simple_openai_codex_responses;

/// Deprecated. Use [`crate::api::openai_completions::stream`] or the
/// [`crate::api::openai_completions()`] factory's `stream`.
#[deprecated(note = "use `pi_ai::api::openai_completions::stream`")]
pub use crate::api::openai_completions::stream as stream_openai_completions;
/// Deprecated. Use [`crate::api::openai_completions::stream_simple`] or the
/// [`crate::api::openai_completions()`] factory's `stream_simple`.
#[deprecated(note = "use `pi_ai::api::openai_completions::stream_simple`")]
pub use crate::api::openai_completions::stream_simple as stream_simple_openai_completions;

/// Deprecated. Use [`crate::api::openai_responses::stream`] or the
/// [`crate::api::openai_responses()`] factory's `stream`.
#[deprecated(note = "use `pi_ai::api::openai_responses::stream`")]
pub use crate::api::openai_responses::stream as stream_openai_responses;
/// Deprecated. Use [`crate::api::openai_responses::stream_simple`] or the
/// [`crate::api::openai_responses()`] factory's `stream_simple`.
#[deprecated(note = "use `pi_ai::api::openai_responses::stream_simple`")]
pub use crate::api::openai_responses::stream_simple as stream_simple_openai_responses;

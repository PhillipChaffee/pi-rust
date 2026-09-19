//! The utils belt of `packages/ai/src/utils/`, ported one module per file.
//!
//! Modules ported verbatim in behavior: [`event_stream`], [`uuid`],
//! [`json_parse`], [`assistant_message_frame`], [`text`], [`validation`],
//! [`overflow`], [`retry`], [`provider_retry`], [`abort`], [`abort_signals`],
//! [`sleep`], [`estimate`], [`deferred_tools`], [`provider_env`],
//! [`diagnostics`], [`error_body`], [`hash`], [`headers`], [`pi_user_agent`],
//! [`node_http_proxy`], and [`typebox_helpers`].
//!
//! `sanitize-unicode.ts` is not ported: Rust `String` is UTF-8 and cannot
//! hold the unpaired surrogates it removes, and `serde_json` rejects
//! lone-surrogate escapes when reading the wire, so the invariant the
//! function enforces is statically upheld.

pub mod abort;
pub mod abort_signals;
pub mod assistant_message_frame;
pub mod deferred_tools;
pub mod diagnostics;
pub mod error_body;
pub mod estimate;
pub mod event_stream;
pub mod hash;
pub mod headers;
pub mod json_parse;
pub mod node_http_proxy;
pub mod overflow;
pub mod pi_user_agent;
pub mod provider_env;
pub mod provider_retry;
pub mod retry;
pub mod sleep;
pub mod text;
pub mod typebox_helpers;
pub mod uuid;
pub mod validation;

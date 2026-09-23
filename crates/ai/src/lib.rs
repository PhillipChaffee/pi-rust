//! The unified LLM layer, ported from `packages/ai` in earendil-works/pi at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! One `Context`/`Message`/`Tool`/`AssistantMessageEvent` model shared by
//! every wire-protocol implementation, with the data shapes of the provider
//! catalogs, credential resolution, retry/overflow handling, and image
//! generation. The wire-protocol implementations, OAuth flows, provider
//! registry, and the Models runtime land with their own tickets; this module
//! tree holds what they all share.
//!
//! Telemetry crosses crate boundaries as the [`pi_telemetry::TelemetryHandle`]
//! dispatch handle per [ADR 0004]; the raw generic trait stays available for
//! statically-known sites.
//!
//! Porting restatements the child ticket records (map ticket "pi-ai: core
//! types and message model"):
//!
//! - The open string unions (`Api = KnownApi | (string & {})`,
//!   `ProviderId = KnownProvider | string`) become newtype strings over
//!   [`types::KnownApi`]/[`types::KnownProvider`] enums with lookups; no
//!   nominal type parameter survives.
//! - `ProviderRequestOptions`' transport fields — `fetch`, `signal`, the
//!   `onPayload`/`onResponse` callbacks — group into one
//!   [`types::TransportOptions`] bundle carried by the options structs; the
//!   `fetch` field is the [`http::HttpClient`] seam with the reqwest 0.12 +
//!   rustls process default, and `signal` is the
//!   [`tokio_util::sync::CancellationToken`] port of `AbortSignal`.
//! - `ApiOptionsMap`/`ApiStreamOptions` land with the wire-API children; the
//!   stream-contract interfaces (`ProviderStreams`, `ProviderImages`,
//!   `StreamFunction`, `ImagesFunction`) land with the utils child, which owns
//!   `AssistantMessageEventStream`.
//! - The four compat interfaces (`OpenAICompletionsCompat`,
//!   `OpenAIResponsesCompat`, `AnthropicMessagesCompat`, `BedrockCompat`)
//!   collapse into one [`types::ModelCompat`] struct: TypeScript resolves the
//!   field's shape by the model's `api` at the type level, the wire object
//!   carries no discriminator, and the field sets never conflict — one struct
//!   with per-field ownership documented round-trips the same JSON.
//! - `sanitize-unicode.ts` is not ported: Rust strings are UTF-8 and cannot
//!   hold the unpaired surrogates it strips, and `serde_json` rejects
//!   lone-surrogate escapes when reading the wire, so the invariant it
//!   enforces is statically upheld.

#![forbid(unsafe_code)]

pub mod api;
pub mod auth;
pub mod cli;
pub mod env_api_keys;
pub mod http;
pub mod image_models;
pub mod images;
pub mod images_api_registry;
pub mod images_models;
pub mod model_data;
pub mod models;
pub mod models_store;
pub mod providers;
pub mod session_resources;
pub mod types;
pub mod utils;

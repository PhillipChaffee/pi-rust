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
//!   `onPayload`/`onResponse` callbacks — land with the HttpClient-seam child;
//!   the options structs here carry pure data.
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
//!
//! [ADR 0004]: ../../docs/adr/0004-open-telemetry-seam.md

#![forbid(unsafe_code)]

pub mod session_resources;
pub mod types;

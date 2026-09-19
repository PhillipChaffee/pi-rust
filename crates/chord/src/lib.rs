//! Rust port of `packages/chord` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Chord is an application-composition runtime for agentic systems: facets
//! (synchronous setup units) declare and bind typed services, replicated
//! state flows through a delta-tracked JSON primitive, Go-like contexts
//! carry cancellation and invocation-scoped values, and a transport-agnostic
//! remote boundary speaks the `$chord.service` control vocabulary. Upstream
//! is deliberately not a Pi package — it imports no other Pi workspace
//! package — and the port keeps that boundary: this crate declares zero
//! dependencies.
//!
//! Three upstream contracts are restated because JavaScript provides them at
//! runtime and Rust cannot:
//!
//! - Producers mutate a Proxy-tracked state object; owned Rust data has no
//!   dynamic proxies, so producers record mutation intents on an explicit
//!   tracker ([`crate::delta`]) and flush operation batches.
//! - `RemoteServiceContract<T>` and `JsonRepresentation<T>` exist only at
//!   compile time in TypeScript; the Rust mirror enforces the JSON-only wire
//!   contract with owned values ([`crate::types`], [`crate::json`]) instead
//!   of type-level machinery.
//! - Upstream's boundary test structurally pins that chord imports no other
//!   Pi package and ships no files outside its source tree; the empty
//!   `[dependencies]` table is the same guarantee carried by the manifest.
//!
//! [`crate::context`] and [`crate::delta`] mirror upstream's `./context` and
//! `./delta` subpaths; the remaining modules carry the root API surface. The
//! crate root stays the single import point, mirroring upstream's
//! `src/index.ts` re-export list.

#![forbid(unsafe_code)]

pub mod api;
pub mod bundler;
pub mod context;
pub mod delta;
pub mod errors;
pub mod facets;
pub mod json;
pub mod node;
pub mod services;
pub mod types;

#[cfg(test)]
#[allow(
    clippy::redundant_pub_crate,
    reason = "the helpers are crate-visible in test builds; the lint reads the cfg(test) module as private"
)]
pub(crate) mod test_support;

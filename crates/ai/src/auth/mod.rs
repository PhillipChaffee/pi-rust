//! The auth layer of `packages/ai/src/auth/`, ported one module per file at
//! commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Modules: [`types`] (credential and callback contracts), [`context`] (the
//! default environment/filesystem context), [`credential_store`] (the
//! serialized store and its in-memory default), [`resolve`] (provider-scoped
//! auth resolution with the double-checked OAuth refresh), [`helpers`]
//! (shared builders), and [`oauth`] (the flow-loader seam and one flow per
//! provider subscription login). [`clock`] is the epoch seam the flows read
//! epoch milliseconds from — fake timers freeze upstream's `Date.now()`,
//! and [`resolve`]'s own `now_ms()` serves the resolution path.

pub mod clock;
pub mod context;
pub mod credential_store;
pub mod helpers;
pub mod oauth;
pub mod resolve;
pub mod types;
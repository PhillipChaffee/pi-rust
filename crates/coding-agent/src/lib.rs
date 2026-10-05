//! The coding-agent product layer, upstream's `packages/coding-agent`
//! (`@earendil-works/pi-coding-agent`) at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The crate assembles the port. The path/config foundation — the agent-dir
//! derivation ([`config`]), the session-directory cwd encoding, the thinking
//! defaults ([`defaults`]), and the missing-session-cwd checks
//! ([`session_cwd`]) — lands first; the CLI, session manager, tools, and TUI
//! modes land ticket by ticket on the map (ticket "Port pi-coding-agent",
//! #23).
//!
//! Upstream runs on one JavaScript event loop; the port keeps every future
//! single-threaded-contract and carries cancellation on the same token the
//! lower crates' transport options do. Windows is out of scope for this
//! effort (map ticket "Decide the Rust stack").
#![forbid(unsafe_code)]

pub mod config;
pub mod defaults;
pub mod session_cwd;
pub mod utils;

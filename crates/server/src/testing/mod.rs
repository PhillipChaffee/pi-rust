//! The server test doubles, ported from upstream `src/testing/` (the
//! shipped `./testing` subpath export).
//!
//! They carry the wire-level [`ProtocolTestClient`], the [`TestServerHost`]
//! and [`TestHarness`] the conformance cases drive, and
//! [`create_test_server`]'s deterministic defaults.

#![expect(
    clippy::panic,
    reason = "test doubles panic on a broken fixture contract, upstream's node:assert throws"
)]
#![expect(
    clippy::expect_used,
    reason = "the doubles settle results the driving case's assertions would reject"
)]

pub mod client;
pub mod host;
pub mod server;

pub use crate::latch::Deferred;
#[cfg(unix)]
pub use client::connect_unix_test_client;
pub use client::{ProtocolTestClient, WireChannel};
pub use host::{OpenGate, TestHarness, TestServerHost, create_test_server_services};
pub use server::{TestServer, TestServerOptions, create_test_server};

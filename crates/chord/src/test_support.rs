// Shared fixtures for the ported upstream test suites.
//
// Upstream's `test/helpers.ts` is a one-line re-export of the loopback
// transport; this module mirrors that shim and adds the plumbing vitest
// supplies implicitly: `vi.fn()` spy callbacks become [Spy] call
// counters, and cases driven by `Math.random` run on a seeded [StdRng]
// with pinned seeds instead of ambient entropy. The loopback-transport
// fixture joins this module together with `services::loopback`, whose API
// it needs.

#![allow(
    dead_code,
    reason = "these helpers serve the ported test files; only the modules implemented so far reference them"
)]

use std::cell::Cell;

use rand::SeedableRng;
use rand::rngs::StdRng;

/// Seed for the hand-rolled LCG property test in the delta suite, pinned
/// from upstream `delta.test.ts` (`0x5eed1234`).
pub(crate) const LCG_SEED: u32 = 0x5eed_1234;

/// Builds a reproducible generator for cases upstream drives with
/// `Math.random`, whose round-trip property must hold identically on every
/// run and machine.
#[must_use]
pub(crate) fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Call counter standing in for upstream's `vi.fn()` spies: a clonable
/// handle tests capture in closures to record how many times a callback
/// ran.
#[derive(Debug, Default)]
pub(crate) struct Spy {
    calls: Cell<usize>,
}

impl Spy {
    /// A spy that has recorded no calls.
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self {
            calls: Cell::new(0),
        }
    }

    /// Records one invocation.
    pub(crate) fn call(&self) {
        self.calls.set(self.calls.get() + 1);
    }

    /// The number of recorded invocations.
    #[must_use]
    pub(crate) fn count(&self) -> usize {
        self.calls.get()
    }
}

/// Parses a static JSON fixture string, failing the test on malformed
/// fixture data.
#[must_use]
pub(crate) fn parse_json(text: &str) -> serde_json::Value {
    #[allow(
        clippy::expect_used,
        reason = "test fixtures are static strings reviewed beside the tests that use them"
    )]
    {
        serde_json::from_str(text).expect("test fixture must be valid JSON")
    }
}

/// Computes the lowercase hexadecimal digest of `bytes`, the form the
/// `sha256-` integrity prefix carries in bundle manifests.
#[must_use]
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Creates the temporary directory for real-fs tests, deleted on drop.
#[must_use]
pub(crate) fn temp_dir() -> tempfile::TempDir {
    #[allow(
        clippy::expect_used,
        reason = "losing the test filesystem breaks the suite, not the code under test"
    )]
    {
        tempfile::tempdir().expect("temporary directory creation must succeed")
    }
}

/// Builds the current-thread runtime the concurrency-ordered cases run on,
/// keeping event-loop-ordered semantics deterministic.
#[must_use]
pub(crate) fn current_thread_runtime() -> tokio::runtime::Runtime {
    #[allow(
        clippy::expect_used,
        reason = "a failed runtime build breaks the suite, not the code under test"
    )]
    {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime build must succeed")
    }
}

/// The cancellation primitive the context fixtures wire, re-exported so
/// ported suites share one import site.
pub(crate) use tokio_util::sync::CancellationToken;

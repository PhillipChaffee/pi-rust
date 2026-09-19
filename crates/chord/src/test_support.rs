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
use sha2::{Digest, Sha256};

use crate::types::{JsonNumber, JsonObject, JsonValue};

/// Builds an object key path segment.
#[must_use]
pub(crate) fn key(text: &str) -> crate::delta::Seg {
    crate::delta::Seg::Key(text.to_string())
}

/// Builds an array index path segment.
#[must_use]
pub(crate) fn idx(index: usize) -> crate::delta::Seg {
    crate::delta::Seg::Index(index)
}

/// Builds a JSON object from `(key, value)` pairs in iteration order.
#[must_use]
pub(crate) fn jo(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(JsonObject::from_entries(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    ))
}

/// Builds a JSON array.
#[must_use]
pub(crate) fn ja(items: Vec<JsonValue>) -> JsonValue {
    JsonValue::Array(items)
}

/// Builds a JSON string.
#[must_use]
pub(crate) fn js(text: &str) -> JsonValue {
    JsonValue::Str(text.to_string())
}

/// Builds a finite JSON number.
///
/// # Panics
/// Only on a non-finite literal, which no test fixture spells.
#[must_use]
pub(crate) fn jn(value: f64) -> JsonValue {
    #[allow(
        clippy::expect_used,
        reason = "test fixtures spell finite literals; a non-finite one is a fixture bug"
    )]
    {
        JsonValue::Number(JsonNumber::new(value).expect("fixture numbers are finite"))
    }
}

/// Builds a finite JSON number from a count.
///
/// # Panics
/// Never: small integers are finite.
#[must_use]
pub(crate) fn u(value: usize) -> JsonValue {
    #[allow(
        clippy::cast_precision_loss,
        reason = "test fixtures count well under 2^53; the cast is exact there"
    )]
    jn(value as f64)
}

/// Builds a JSON boolean.
#[must_use]
pub(crate) fn jb(flag: bool) -> JsonValue {
    JsonValue::Bool(flag)
}

/// Settles a mutation the case does not expect to fail.
///
/// # Panics
/// When the mutation fails; the case's own `Err` assertions handle those.
pub(crate) fn ok<T, E: std::fmt::Debug>(result: Result<T, E>) -> T {
    #[allow(
        clippy::expect_used,
        reason = "test helpers surface the unexpected rejection at the failing case only"
    )]
    {
        result.expect("the case's mutation succeeds")
    }
}

/// Settles an optional read the case expects to find.
///
/// # Panics
/// When the value is absent; the case's own `None` assertions handle those.
#[must_use]
pub(crate) fn some<T>(option: Option<T>) -> T {
    #[allow(
        clippy::expect_used,
        reason = "test helpers surface the unexpected absence at the failing case only"
    )]
    {
        option.expect("the case's value exists")
    }
}

/// Settles an application, the same contract as [`ok`] for the applier.
pub(crate) fn applied(target: Option<JsonValue>, ops: Vec<crate::delta::Op>) -> JsonValue {
    ok(crate::delta::apply(target, ops))
}

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

/// Call counter standing in for upstream's `vi.fn()` spies: a cloneable
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
    use std::fmt::Write as _;

    let digest = Sha256::digest(bytes);
    let mut text = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(text, "{byte:02x}");
    }
    text
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

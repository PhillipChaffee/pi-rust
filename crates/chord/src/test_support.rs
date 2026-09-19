// Shared fixtures for the ported upstream test suites.
//
// Upstream's `test/helpers.ts` is a one-line re-export of the loopback
// transport; this module mirrors that shim and adds the plumbing vitest
// supplies implicitly: cases driven by `Math.random` run on a seeded
// [StdRng] with pinned seeds instead of ambient entropy. The
// loopback-transport fixture joins this module together with
// `services::loopback`, whose API it needs.

#![allow(
    dead_code,
    reason = "these helpers serve the ported test files; only the modules implemented so far reference them"
)]

use rand::SeedableRng;
use rand::rngs::StdRng;

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

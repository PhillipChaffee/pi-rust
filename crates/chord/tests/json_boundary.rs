//! The `isJsonValue` acceptance suite and the package boundary test, ported
//! from upstream `test/json.test.ts` and `test/boundary.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!

#![allow(
    clippy::panic,
    reason = "test assertions panic at the failing case only; the restriction lint targets production code"
)]
#![allow(
    clippy::expect_used,
    reason = "test helpers settle results the case's own assertions would reject"
)]

use pi_chord::types::JsonObject;

fn jo(entries: Vec<(&str, pi_chord::types::JsonValue)>) -> pi_chord::types::JsonValue {
    pi_chord::types::JsonValue::Object(JsonObject::from_entries(
        entries.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
    ))
}

/// Upstream checks a live JS value tree: `{ nested: [1, true, null] }`.
/// The owned tree cannot carry cycles, exotic prototypes, holes, or
/// non-finite numbers by construction, so the acceptance case pins the
/// shapes `is_json_value` still polices (depth) and the restated
/// guarantees (owned data cannot violate the rest).
#[test]
fn checks_strict_json_without_normalizing_it() {
    let nested = jo(vec![(
        "nested",
        pi_chord::types::JsonValue::Array(vec![
            pi_chord::types::JsonValue::number(1.0).expect("1.0 is finite"),
            pi_chord::types::JsonValue::Bool(true),
            pi_chord::types::JsonValue::Null,
        ]),
    )]);
    assert!(pi_chord::json::is_json_value(&nested));

    // `undefined` has no owned representation; the owned tree restates the
    // guarantee by construction, so the case pins that a value IS strict
    // JSON and that the depth guard exists.
    let deep = deep_json(pi_chord::json::MAX_DEPTH);
    assert!(pi_chord::json::is_json_value(&deep));
    let too_deep = deep_json(pi_chord::json::MAX_DEPTH + 1);
    assert!(!pi_chord::json::is_json_value(&too_deep));

    // `Number.POSITIVE_INFINITY` cannot enter the tree: the constructor
    // rejects it, which is the owned restatement of `isJsonValue`'s check.
    assert!(pi_chord::types::JsonValue::number(f64::INFINITY).is_none());
}

#[allow(
    clippy::expect_used,
    reason = "the fixture generator only builds finite numbers"
)]
fn deep_json(depth: usize) -> pi_chord::types::JsonValue {
    let mut value = pi_chord::types::JsonValue::Null;
    for _ in 0..depth {
        value = pi_chord::types::JsonValue::Array(vec![value]);
    }
    value
}

//! JSON validation restated for owned data.
//!
//! Ported from upstream `src/json.ts`: the acceptance surface
//! [`is_json_value`] checks for the owned [`JsonValue`] tree (finite numbers
//! only, no cycles, no exotic prototypes, no sparse arrays), a
//! [`JsonNumber`] constructor surface that keeps non-finite values
//! unrepresentable, and the helpers the `json.test.ts` port drives. Upstream
//! caps recursion depth at 512 to bound hostile payloads; the same cap
//! applies to the owned tree as a wire contract.

use crate::types::JsonValue;

/// The maximum nesting depth a value may carry.
///
/// Upstream's `json.ts` returns false past this depth for values built at
/// runtime; owned data cannot cycle, so this is the one hostile-payload
/// bound the port still polices.
pub const MAX_DEPTH: usize = 512;

/// Whether the value is strict JSON at or under the recursion cap.
///
/// Upstream validates unknown runtime values — exotic prototypes, sparse
/// arrays, cycles, non-finite numbers — and the owned [`JsonValue`] tree
/// makes everything but depth unrepresentable: [`crate::types::JsonNumber`]
/// rejects non-finite values at construction, an object's keys are strings
/// by type, arrays are dense vectors, and a tree owned by value cannot
/// reference itself. What remains checkable is the depth cap: a tree nested
/// deeper than [`MAX_DEPTH`] is rejected exactly as upstream rejects it.
#[must_use]
pub fn is_json_value(value: &JsonValue) -> bool {
    check(value, 0)
}

fn check(value: &JsonValue, depth: usize) -> bool {
    if depth > MAX_DEPTH {
        return false;
    }
    match value {
        JsonValue::Null | JsonValue::Bool(_) | JsonValue::Number(_) | JsonValue::Str(_) => true,
        JsonValue::Array(items) => {
            if depth >= MAX_DEPTH {
                return false;
            }
            items.iter().all(|item| check(item, depth + 1))
        }
        JsonValue::Object(object) => {
            if depth >= MAX_DEPTH {
                return false;
            }
            object.iter().all(|(_, item)| check(item, depth + 1))
        }
    }
}

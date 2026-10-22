use super::*;
use crate::test_support::ja;
use crate::types::{JsonNumber, JsonObject};

#[test]
fn checks_strict_json_without_normalizing_it() {
    // Nested valid data is accepted...
    let valid = JsonValue::Object(JsonObject::from_entries(vec![(
        "nested".to_string(),
        JsonValue::Array(vec![
            JsonValue::Number(JsonNumber::from(1_i64)),
            JsonValue::Bool(true),
            JsonValue::Null,
        ]),
    )]));
    assert!(is_json_value(&valid));
    // ...and the tree the constructor surface builds is the only JSON:
    // non-finite numbers are unrepresentable at construction,
    assert!(JsonNumber::new(f64::INFINITY).is_none());
    assert!(JsonNumber::new(f64::NAN).is_none());
    // cycles cannot be owned, exotic prototypes and sparse arrays cannot
    // exist, and the one runtime check left is the depth cap.
    let mut deep = JsonValue::Null;
    for _ in 0..600 {
        deep = JsonValue::Array(vec![deep]);
    }
    assert!(!is_json_value(&deep));
    let mut shallow = JsonValue::Null;
    for _ in 0..MAX_DEPTH {
        shallow = JsonValue::Array(vec![shallow]);
    }
    assert!(is_json_value(&shallow));
    // Empty containers are values.
    assert!(is_json_value(&ja(Vec::new())));
    assert!(is_json_value(&JsonValue::Object(JsonObject::default())));
}

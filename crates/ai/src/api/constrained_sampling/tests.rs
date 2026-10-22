use super::*;
use serde_json::json;

/// The non-schema guard the strict conversion cannot reach — the callers
/// convert first, so every walked schema is an object — still reads as
/// "admits no null", upstream's `schemaAllowsNull` guard.
#[test]
fn schema_allows_null_rejects_non_object_schemas() {
    assert![!schema_allows_null(&json!(true))];
    assert![!schema_allows_null(&json!("null"))];
    assert![!schema_allows_null(&json!(0))];
}

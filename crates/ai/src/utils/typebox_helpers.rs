//! Tool-schema helpers for providers that reject `anyOf`/`const` patterns,
//! ported from `packages/ai/src/utils/typebox-helpers.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Tool parameters are JSON-Schema documents in this port (the core-types
//! schema decision), so the string-enum helper emits that document directly:
//! a `type: "string"` schema with an `enum` list, compatible with Google's
//! API and other providers that reject `anyOf`/`const` shapes.

use serde_json::{Map, Value, json};

/// The options of [`string_enum`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StringEnumOptions {
    /// The schema description.
    pub description: Option<String>,
    /// The default member.
    pub default: Option<String>,
}

/// A string-enum schema compatible with Google's API and other providers
/// that do not support `anyOf`/`const` patterns, upstream's `StringEnum`.
#[must_use]
pub fn string_enum(values: &[&str], options: Option<StringEnumOptions>) -> Value {
    let mut schema = Map::new();
    schema.insert(String::from("type"), json!("string"));
    schema.insert(String::from("enum"), json!(values));
    if let Some(description) = options
        .as_ref()
        .and_then(|options| options.description.as_ref())
    {
        schema.insert(String::from("description"), json!(description));
    }
    if let Some(default) = options.and_then(|options| options.default) {
        schema.insert(String::from("default"), json!(default));
    }
    Value::Object(schema)
}

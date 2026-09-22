//! JSON-schema constrained sampling, ported from
//! `packages/ai/src/api/constrained-sampling.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The grammar-constrained half of the upstream module (custom-tool grammar
//! variants for the OpenAI Responses API) ports with the OpenAI-family
//! child; the Anthropic Messages path uses only the JSON-schema subset.

use serde_json::{Map, Value};

use crate::types::{ConstrainedSamplingConfig, ConstrainedSamplingSetting, Strictness, Tool};

/// The schema features provider strict sampling cannot express, upstream's
/// `UNSUPPORTED_STRICT_SCHEMA_KEYS`.
const UNSUPPORTED_STRICT_SCHEMA_KEYS: [&str; 16] = [
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
];

/// A schema strict sampling cannot represent, upstream's
/// `UnsupportedStrictJsonSchemaError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedStrictJsonSchemaError(pub String);

impl std::fmt::Display for UnsupportedStrictJsonSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnsupportedStrictJsonSchemaError {}

fn is_json_schema_object(value: &Value) -> bool {
    value.is_object()
}

/// Whether the schema describes a structured (object or array) shape,
/// upstream's `isStructuredSchema`.
fn is_structured_schema(schema: &Value) -> bool {
    if !is_json_schema_object(schema) {
        return false;
    }
    let types = schema
        .get("type")
        .map_or(Vec::new(), |type_value| match type_value {
            Value::String(name) => vec![name.as_str()],
            Value::Array(names) => names.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        });
    types.contains(&"object")
        || types.contains(&"array")
        || schema.get("properties").is_some_and(|p| !p.is_null())
        || schema.get("items").is_some_and(|items| !items.is_null())
}

fn schema_allows_null(schema: &Value) -> bool {
    if !is_json_schema_object(schema) {
        return false;
    }
    let is_null_type = schema
        .get("type")
        .is_some_and(|type_value| match type_value {
            Value::String(name) => name == "null",
            Value::Array(names) => names.iter().any(|name| name == "null"),
            _ => false,
        });
    if is_null_type {
        return true;
    }
    let const_allows = schema.get("const").is_some_and(Value::is_null);
    let enum_allows = schema
        .get("enum")
        .and_then(Value::as_array)
        .is_some_and(|values| values.iter().any(Value::is_null));
    if const_allows || enum_allows {
        return true;
    }
    schema
        .get("anyOf")
        .and_then(Value::as_array)
        .is_some_and(|variants| variants.iter().any(schema_allows_null))
}

/// Restrict the schema in place to the subset strict sampling expresses,
/// upstream's `makeJsonSchemaNodeStrict`: every property becomes required
/// with an explicit null union, `additionalProperties` closes, and anything
/// outside the subset rejects with the wire-meaningful reason.
///
/// # Errors
/// [`UnsupportedStrictJsonSchemaError`] naming the unsupported construct.
#[allow(
    clippy::expect_used,
    reason = "the navigation targets are object-checked above; a miss is a caller bug the message names"
)]
#[expect(
    clippy::too_many_lines,
    reason = "each rejection branch is one wire-vocabulary pin; splitting would obscure the mapping"
)]
fn make_json_schema_node_strict(
    schema: &mut Value,
) -> Result<(), UnsupportedStrictJsonSchemaError> {
    if !is_json_schema_object(schema) {
        return Err(UnsupportedStrictJsonSchemaError(
            "boolean schemas are unsupported".to_owned(),
        ));
    }
    for key in UNSUPPORTED_STRICT_SCHEMA_KEYS {
        if schema.get(key).is_some() {
            return Err(UnsupportedStrictJsonSchemaError(format!(
                "{key} schemas are unsupported"
            )));
        }
    }

    if let Some(any_of) = schema.get_mut("anyOf") {
        let Some(variants) = any_of
            .as_array_mut()
            .filter(|variants| !variants.is_empty())
        else {
            return Err(UnsupportedStrictJsonSchemaError(
                "anyOf must contain at least one schema".to_owned(),
            ));
        };
        for variant in variants {
            if is_structured_schema(variant) {
                return Err(UnsupportedStrictJsonSchemaError(
                    "object and array unions are unsupported".to_owned(),
                ));
            }
            make_json_schema_node_strict(variant)?;
        }
    }

    if let Some(items) = schema.get_mut("items") {
        if items.is_array() {
            return Err(UnsupportedStrictJsonSchemaError(
                "tuple schemas are unsupported".to_owned(),
            ));
        }
        make_json_schema_node_strict(items)?;
    }

    let is_object_schema = schema.get("type").and_then(Value::as_str) == Some("object");
    if schema.get("properties").is_some() && !is_object_schema {
        return Err(UnsupportedStrictJsonSchemaError(
            "properties require type object".to_owned(),
        ));
    }
    if !is_object_schema {
        return Ok(());
    }
    if schema
        .get("additionalProperties")
        .is_some_and(|value| *value != Value::Bool(false))
    {
        return Err(UnsupportedStrictJsonSchemaError(
            "schema-valued or true additionalProperties is unsupported".to_owned(),
        ));
    }
    if schema
        .get("properties")
        .is_some_and(|value| !value.is_object())
    {
        return Err(UnsupportedStrictJsonSchemaError(
            "object properties must be a schema map".to_owned(),
        ));
    }
    if let Some(required) = schema.get("required")
        && (!required.is_array()
            || required
                .as_array()
                .is_some_and(|keys| keys.iter().any(|key| !key.is_string())))
    {
        return Err(UnsupportedStrictJsonSchemaError(
            "object required must be a string array".to_owned(),
        ));
    }

    let properties: Map<String, Value> = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let property_names: Vec<String> = properties.keys().cloned().collect();
    let required: std::collections::BTreeSet<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|keys| {
            keys.iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if required.iter().any(|key| !properties.contains_key(key)) {
        return Err(UnsupportedStrictJsonSchemaError(
            "required contains an unknown property".to_owned(),
        ));
    }

    let object = schema.as_object_mut().expect("checked object above");
    let live_properties = object
        .entry("properties".to_owned())
        .or_insert_with(|| Value::Object(Map::new()));
    let live = live_properties
        .as_object_mut()
        .expect("checked object above");
    for key in &property_names {
        let Some(property) = live.get_mut(key) else {
            continue;
        };
        make_json_schema_node_strict(property)?;
        let allows_null = schema_allows_null(property);
        let converted = property.clone();
        if !required.contains(key) && !allows_null {
            live.insert(
                key.clone(),
                serde_json::json!({ "anyOf": [converted, { "type": "null" }] }),
            );
        }
    }
    let object = schema.as_object_mut().expect("checked object above");
    object.insert(
        "required".to_owned(),
        Value::Array(property_names.into_iter().map(Value::String).collect()),
    );
    object.insert("additionalProperties".to_owned(), Value::Bool(false));
    Ok(())
}

/// Convert a tool schema to the strict subset provider constrained sampling
/// expresses, upstream's `makeStrictJsonSchema`.
///
/// The input schema is never mutated; the strict form requires every
/// property (null unions where the original allowed absence) and closes the
/// object.
///
/// # Errors
/// [`UnsupportedStrictJsonSchemaError`] when the schema leaves the subset.
pub fn make_strict_json_schema(schema: &Value) -> Result<Value, UnsupportedStrictJsonSchemaError> {
    let mut cloned = schema.clone();
    if !is_json_schema_object(&cloned) {
        return Err(UnsupportedStrictJsonSchemaError(
            "root schema must have type object".to_owned(),
        ));
    }
    make_json_schema_node_strict(&mut cloned)?;
    if cloned.get("type").and_then(Value::as_str) != Some("object") {
        return Err(UnsupportedStrictJsonSchemaError(
            "root schema must have type object".to_owned(),
        ));
    }
    Ok(cloned)
}

/// The parameters a tool's request entry carries, upstream's
/// `getJsonSchemaToolParameters`: the strict schema when constrained, the
/// tool's own schema otherwise.
///
/// # Errors
/// The strict-conversion failure; only reachable when callers skip
/// [`resolve_json_schema_strict_sampling`]'s validation.
pub fn get_json_schema_tool_parameters(
    tool: &Tool,
    strict: Option<bool>,
) -> Result<Value, UnsupportedStrictJsonSchemaError> {
    if strict == Some(true) {
        return make_strict_json_schema(&tool.parameters);
    }
    Ok(tool.parameters.clone())
}

/// Resolve whether a tool's request entry carries `strict: true`, upstream's
/// `resolveJsonSchemaStrictSampling`. `None` means no strict entry; `Some(_)`
/// only follows a successful strict conversion.
///
/// # Errors
/// The rejection message when a `require`-strict tool cannot convert.
pub fn resolve_json_schema_strict_sampling(
    tool: &Tool,
    supports_strict_mode: bool,
) -> Result<Option<bool>, String> {
    let Some(ConstrainedSamplingSetting::Config(ConstrainedSamplingConfig::JsonSchema { strict })) =
        tool.constrained_sampling.as_ref()
    else {
        return Ok(None);
    };

    if supports_strict_mode {
        return match make_strict_json_schema(&tool.parameters) {
            Ok(_) => Ok(Some(true)),
            Err(reason) => {
                if *strict != Strictness::Require {
                    return Ok(None);
                }
                Err(format!(
                    "Tool \"{}\" requires JSON-schema constrained sampling, but {}.",
                    tool.name, reason.0
                ))
            }
        };
    }
    if *strict == Strictness::Require {
        return Err(format!(
            "Tool \"{}\" requires JSON-schema constrained sampling, but strict tools are unsupported.",
            tool.name
        ));
    }
    Ok(None)
}

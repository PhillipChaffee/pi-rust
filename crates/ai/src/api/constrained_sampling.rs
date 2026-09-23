//! JSON-schema constrained sampling, ported from
//! `packages/ai/src/api/constrained-sampling.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The grammar-constrained half of the upstream module (custom-tool grammar
//! variants for the OpenAI Responses API) ports with the OpenAI-family
//! child; the Anthropic Messages path uses only the JSON-schema subset.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::types::{
    ConstrainedSamplingConfig, ConstrainedSamplingSetting, GrammarFormat, Strictness, Tool,
};

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

/// A resolved grammar-constrained tool, upstream's `GrammarConstrainedSampling`:
/// the variant that binds, its definition, and the single string property the
/// grammar's input feeds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrammarConstrainedSampling {
    /// Which OpenAI custom-tool grammar the definition uses.
    pub format: GrammarFormat,
    /// The grammar definition text.
    pub definition: String,
    /// The tool's single required string property the constrained input
    /// streams through.
    pub input_property: String,
}

/// The JSON buffer a grammar tool's streamed input accumulates in, upstream's
/// `GrammarToolInputJsonBuffer`.
///
/// `input` is the raw string value received so far; `started`/`closed` track
/// the wrapper's `{ "prop": "..." ` state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GrammarToolInputJsonBuffer {
    /// The input string received so far.
    pub input: String,
    /// Whether the `{"property":"` prefix was emitted.
    pub started: bool,
    /// Whether the closing `"}` was emitted.
    pub closed: bool,
}

/// The raw input a finished grammar tool call carries, upstream's
/// `getGrammarToolInput`.
///
/// # Errors
/// A string-typed `input_property` is the grammar contract; anything else
/// fails with the wire-meaningful message.
pub fn get_grammar_tool_input(
    tool_name: &str,
    arguments: &Map<String, Value>,
    input_property: &str,
) -> Result<String, String> {
    match arguments.get(input_property) {
        Some(Value::String(input)) => Ok(input.clone()),
        _ => Err(format!(
            "Grammar tool call \"{tool_name}\" requires argument \"{input_property}\" to be a string."
        )),
    }
}

/// Extend the streamed JSON fragment of a grammar tool's arguments with the
/// next input, upstream's `appendGrammarToolInputJsonDelta`. Returns `None`
/// when the update adds nothing observable.
///
/// # Errors
/// A non-monotonic input (the string shrinks or changes) or a change after
/// the property closed: the deltas must be a prefix chain ending once.
pub fn append_grammar_tool_input_json_delta(
    buffer: &mut GrammarToolInputJsonBuffer,
    input_property: &str,
    next_input: &str,
    close: bool,
) -> Result<Option<String>, String> {
    if buffer.closed {
        if close && next_input == buffer.input {
            return Ok(None);
        }
        return Err(format!(
            "grammar tool input for property \"{input_property}\" changed after it was closed"
        ));
    }
    if !next_input.starts_with(&buffer.input) {
        return Err(format!(
            "grammar tool input for property \"{input_property}\" changed non-monotonically"
        ));
    }

    let input_delta = &next_input[buffer.input.len()..];
    if !close && input_delta.is_empty() {
        return Ok(None);
    }

    let mut delta = String::new();
    if !buffer.started {
        delta.push_str("{\"");
        delta.push_str(input_property);
        delta.push_str("\":\"");
        buffer.started = true;
    }
    delta.push_str(&json_string_content(input_delta));
    next_input.clone_into(&mut buffer.input);

    if close {
        delta.push_str("\"}");
        buffer.closed = true;
    }
    Ok(Some(delta))
}

/// The delta's escaped form: the property's JSON-string body without the
/// surrounding quotes, so the emitted fragment stays a valid JSON prefix.
fn json_string_content(value: &str) -> String {
    let quoted = serde_json::to_string(value).unwrap_or_default();
    quoted[1..quoted.len() - 1].to_owned()
}

/// The single string property a grammar tool's input streams through,
/// upstream's `inferGrammarInputProperty`.
///
/// # Errors
/// The wire-meaningful rejection when the schema is not an object with
/// exactly one required string property.
fn infer_grammar_input_property(tool: &Tool) -> Result<String, String> {
    let schema = &tool.parameters;
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err("grammar constrained sampling requires an object parameter schema".to_owned());
    }
    let required = schema.get("required").and_then(Value::as_array);
    let Some(property) =
        required.filter(|required| required.len() == 1 && required[0].as_str().is_some())
    else {
        return Err(
            "grammar constrained sampling requires exactly one required string property".to_owned(),
        );
    };
    let input_property = property[0].as_str().unwrap_or_default().to_owned();
    let property_schema = schema
        .get("properties")
        .and_then(|properties| properties.get(&input_property))
        .ok_or_else(|| {
            format!("grammar constrained sampling requires a properties entry for {input_property}")
        })?;
    if property_schema.get("type").and_then(Value::as_str) != Some("string") {
        return Err(format!(
            "grammar constrained sampling property {input_property} must have type string"
        ));
    }
    Ok(input_property)
}

/// Resolve a tool's grammar-constrained sampling, upstream's
/// `resolveGrammarConstrainedSampling`. `None` when the tool does not opt in
/// or the model cannot host grammar tools.
///
/// # Errors
/// The wire-meaningful rejection when a grammar-constrained tool carries no
/// usable variant or a non-conforming schema.
pub fn resolve_grammar_constrained_sampling(
    tool: &Tool,
    supports_openai_grammar_tools: bool,
) -> Result<Option<GrammarConstrainedSampling>, String> {
    let Some(ConstrainedSamplingSetting::Config(ConstrainedSamplingConfig::Grammar { variants })) =
        tool.constrained_sampling.as_ref()
    else {
        return Ok(None);
    };
    if !supports_openai_grammar_tools {
        return Ok(None);
    }

    let lark = variants.get(&GrammarFormat::OpenaiLark);
    let regex = variants.get(&GrammarFormat::OpenaiRegex);
    let has_lark = lark.is_some_and(|definition| !definition.trim().is_empty());
    let has_regex = regex.is_some_and(|definition| !definition.trim().is_empty());
    if !has_lark && !has_regex {
        return Err(format!(
            "Tool \"{}\" cannot use grammar constrained sampling: no supported grammar variant was provided.",
            tool.name
        ));
    }

    let (format, definition) = if has_lark {
        let definition = lark.unwrap_or(&String::new()).clone();
        (GrammarFormat::OpenaiLark, definition)
    } else {
        let definition = regex.unwrap_or(&String::new()).clone();
        (GrammarFormat::OpenaiRegex, definition)
    };
    let input_property = infer_grammar_input_property(tool).map_err(|message| {
        format!(
            "Tool \"{}\" cannot use grammar constrained sampling: {message}.",
            tool.name
        )
    })?;
    Ok(Some(GrammarConstrainedSampling {
        format,
        definition,
        input_property,
    }))
}

/// The grammar tool-input property per tool name, upstream's
/// `createGrammarToolInputProperties`: the map a stream consults when a tool
/// call's arguments arrive as raw input.
#[must_use]
pub fn create_grammar_tool_input_properties(
    tools: Option<&[Tool]>,
    supports_openai_grammar_tools: bool,
) -> BTreeMap<String, String> {
    let mut properties = BTreeMap::new();
    for tool in tools.unwrap_or_default() {
        if let Ok(Some(grammar)) =
            resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        {
            properties.insert(tool.name.clone(), grammar.input_property);
        }
    }
    properties
}

#[cfg(test)]
mod tests {
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
}

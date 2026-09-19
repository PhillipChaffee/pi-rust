//! Tool-call argument validation, ported from
//! `packages/ai/src/utils/validation.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Tool parameters are JSON-Schema documents in this port (the core-types
//! schema decision), so the three upstream stages collapse to the paths that
//! actually act on plain schemas:
//!
//! - `normalizeOptionalNulls` drops `null` values where the schema forbids
//!   them and the property is not required, recursively.
//! - typebox's `Value.Convert` is a no-op on plain JSON Schema documents —
//!   its conversion dispatches on typebox-internal kind symbols the wire
//!   never carries — so the port runs only the plain-schema coercion
//!   (`coerce_with_json_schema`), which reproduces the AJV-compatible
//!   primitive rules the upstream tests pin.
//! - The compiled validator is the interpreted JSON-Schema checker in this
//!   module. Upstream's generated-code path (`Compile` codegen, exercised by
//!   the CSP test that swaps out the `Function` constructor) has no Rust
//!   counterpart; the interpreted fallback is the only path here, which is
//!   that test's subject.
//!
//! The upstream WeakMap validator cache is keyed on object identity and has
//! no Rust counterpart; schemas are checked per call.

use serde_json::{Map, Value};

use crate::types::{Tool, ToolCall};

/// Why argument validation failed: the formatted multi-line message upstream
/// throws.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationError(pub String);

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ValidationError {}

/// One validation failure with its location and kind, upstream's
/// `TLocalizedValidationError` fields the formatter reads.
struct CheckError {
    /// The JSON pointer segments of the failing value, upstream's
    /// `instancePath`.
    instance_path: Vec<PathSegment>,
    /// The failure description.
    message: String,
    /// The failing keyword, upstream's `error.keyword`.
    keyword: Keyword,
    /// The missing property names for `required` failures, upstream's
    /// `error.params.requiredProperties`.
    required_properties: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Keyword {
    Required,
    Type,
    AnyOf,
    OneOf,
    Enum,
    AdditionalProperties,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PathSegment {
    Key(String),
    Index(usize),
}

/// The JSON-Schema subset the validator checks: `type` (including unions and
/// `null`), `properties`, `required`, `additionalProperties` (boolean and
/// schema), `items` (single and tuple), `allOf`/`anyOf`/`oneOf`, `enum`,
/// `const`, and `#`-pointer `$ref` resolution within the document.
struct Validator {
    root: Value,
}

impl Validator {
    fn new(root: &Value) -> Self {
        Self { root: root.clone() }
    }

    fn check(&self, value: &Value) -> Vec<CheckError> {
        let mut errors = Vec::new();
        self.check_value(value, &self.root, &mut Vec::new(), &mut errors);
        errors
    }

    fn check_value(
        &self,
        value: &Value,
        schema: &Value,
        path: &mut Vec<PathSegment>,
        errors: &mut Vec<CheckError>,
    ) {
        if let Some(pointer) = schema.get("$ref").and_then(Value::as_str) {
            if let Some(target) = resolve_pointer(&self.root, pointer) {
                self.check_value(value, target, path, errors);
            }
            return;
        }

        if let Some(all_of) = schema.get("allOf").and_then(Value::as_array) {
            for member in all_of {
                self.check_value(value, member, path, errors);
            }
        }
        if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array)
            && !any_of
                .iter()
                .any(|member| self.check_value_quiet(value, member))
        {
            errors.push(CheckError {
                instance_path: path.clone(),
                message: String::from("must match a schema of anyOf"),
                keyword: Keyword::AnyOf,
                required_properties: Vec::new(),
            });
        }
        if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
            let matches = one_of
                .iter()
                .filter(|member| self.check_value_quiet(value, member))
                .count();
            if matches != 1 {
                errors.push(CheckError {
                    instance_path: path.clone(),
                    message: String::from("must match exactly one schema of oneOf"),
                    keyword: Keyword::OneOf,
                    required_properties: Vec::new(),
                });
            }
        }

        let schema_types = get_schema_types(schema);
        if !schema_types.is_empty()
            && !schema_types
                .iter()
                .any(|schema_type| matches_json_type(value, schema_type))
        {
            errors.push(CheckError {
                instance_path: path.clone(),
                message: format!("must be {}", schema_types.join(", ")),
                keyword: Keyword::Type,
                required_properties: Vec::new(),
            });
            return;
        }

        if let Some(const_value) = schema.get("const")
            && value != const_value
        {
            errors.push(CheckError {
                instance_path: path.clone(),
                message: format!("must be equal to constant {const_value}"),
                keyword: Keyword::Enum,
                required_properties: Vec::new(),
            });
        }
        if let Some(enum_values) = schema.get("enum").and_then(Value::as_array)
            && !enum_values.contains(value)
        {
            errors.push(CheckError {
                instance_path: path.clone(),
                message: format!(
                    "must be equal to one of the allowed values: {}",
                    enum_value_list(enum_values)
                ),
                keyword: Keyword::Enum,
                required_properties: Vec::new(),
            });
        }

        if let Some(object) = value.as_object() {
            self.check_object(object, schema, path, errors);
        }
        if let Some(array) = value.as_array() {
            self.check_array(array, schema, path, errors);
        }
    }

    fn check_value_quiet(&self, value: &Value, schema: &Value) -> bool {
        let mut errors = Vec::new();
        self.check_value(value, schema, &mut Vec::new(), &mut errors);
        errors.is_empty()
    }

    fn check_object(
        &self,
        object: &Map<String, Value>,
        schema: &Value,
        path: &mut Vec<PathSegment>,
        errors: &mut Vec<CheckError>,
    ) {
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for missing in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(missing) {
                    errors.push(CheckError {
                        instance_path: path.clone(),
                        message: format!("must have required property '{missing}'"),
                        keyword: Keyword::Required,
                        required_properties: vec![missing.to_owned()],
                    });
                }
            }
        }
        let properties = properties_schema(schema).cloned().unwrap_or_default();
        if let Some(property_schemas) = schema.get("properties").and_then(Value::as_object) {
            for (key, property_value) in object {
                if let Some(property_schema) = property_schemas.get(key) {
                    path.push(PathSegment::Key(key.clone()));
                    self.check_value(property_value, property_schema, path, errors);
                    path.pop();
                }
            }
        }
        match schema.get("additionalProperties") {
            Some(Value::Bool(false)) => {
                for key in object.keys() {
                    if !properties.contains_key(key) {
                        path.push(PathSegment::Key(key.clone()));
                        errors.push(CheckError {
                            instance_path: path.clone(),
                            message: String::from("must NOT have additional properties"),
                            keyword: Keyword::AdditionalProperties,
                            required_properties: Vec::new(),
                        });
                        path.pop();
                    }
                }
            }
            Some(additional_schema @ Value::Object(_)) => {
                for (key, property_value) in object {
                    if properties.contains_key(key) {
                        continue;
                    }
                    path.push(PathSegment::Key(key.clone()));
                    self.check_value(property_value, additional_schema, path, errors);
                    path.pop();
                }
            }
            _ => {}
        }
    }

    fn check_array(
        &self,
        array: &[Value],
        schema: &Value,
        path: &mut Vec<PathSegment>,
        errors: &mut Vec<CheckError>,
    ) {
        let Some(items) = schema.get("items") else {
            return;
        };
        match items {
            Value::Array(item_schemas) => {
                for (index, item_schema) in item_schemas.iter().enumerate().take(array.len()) {
                    path.push(PathSegment::Index(index));
                    self.check_value(&array[index], item_schema, path, errors);
                    path.pop();
                }
            }
            item_schema @ Value::Object(_) => {
                for (index, item_value) in array.iter().enumerate() {
                    path.push(PathSegment::Index(index));
                    self.check_value(item_value, item_schema, path, errors);
                    path.pop();
                }
            }
            _ => {}
        }
    }
}

fn properties_schema(schema: &Value) -> Option<&Map<String, Value>> {
    schema.get("properties").and_then(Value::as_object)
}

fn get_schema_types(schema: &Value) -> Vec<String> {
    match schema.get("type") {
        Some(Value::String(single)) => vec![single.clone()],
        Some(Value::Array(types)) => types
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn matches_json_type(value: &Value, schema_type: &str) -> bool {
    match schema_type {
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn enum_value_list(values: &[Value]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The path a validation error formats to, upstream's `formatValidationPath`:
/// dot-joined segments, with `required` failures naming the missing property.
fn format_validation_path(error: &CheckError) -> String {
    let base_path = error
        .instance_path
        .iter()
        .map(|segment| match segment {
            PathSegment::Key(key) => key.clone(),
            PathSegment::Index(index) => index.to_string(),
        })
        .collect::<Vec<_>>()
        .join(".");
    if error.keyword == Keyword::Required
        && let Some(required_property) = error.required_properties.first()
    {
        return if base_path.is_empty() {
            required_property.clone()
        } else {
            format!("{base_path}.{required_property}")
        };
    }
    if base_path.is_empty() {
        String::from("root")
    } else {
        base_path
    }
}

/// Resolve a `#`-pointer `$ref` within the document.
fn resolve_pointer<'a>(root: &'a Value, pointer: &str) -> Option<&'a Value> {
    let pointer = pointer.strip_prefix('#')?;
    if pointer == "/" || pointer.is_empty() {
        return Some(root);
    }
    let mut current = root;
    for raw in pointer.trim_start_matches('/').split('/') {
        let segment = raw.replace("~1", "/").replace("~0", "~");
        current = match current {
            Value::Object(map) => map.get(&segment)?,
            Value::Array(array) => array.get(segment.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(current)
}

fn resolve_schema<'a>(schema: &'a Value, root: &'a Value) -> &'a Value {
    if let Some(pointer) = schema.get("$ref").and_then(Value::as_str)
        && let Some(target) = resolve_pointer(root, pointer)
    {
        return target;
    }
    schema
}

/// Normalize optional nulls out of the arguments: a `null` whose property is
/// not required, whose schema does not reference another definition, and
/// which the schema's validator rejects is omitted entirely. Values under
/// nullable schemas are preserved.
fn normalize_optional_nulls(value: &mut Value, schema: &Value, root: &Value) {
    let schema = resolve_schema(schema, root);
    if let Value::Array(array) = value {
        if let Some(Value::Array(item_schemas)) = schema.get("items") {
            for (index, item) in array.iter_mut().enumerate().take(item_schemas.len()) {
                normalize_optional_nulls(item, &item_schemas[index], root);
            }
            return;
        }
        if let Some(items) = schema.get("items").filter(|items| items.is_object()) {
            for item in array.iter_mut() {
                normalize_optional_nulls(item, items, root);
            }
        }
        return;
    }
    let Some(object) = value.as_object_mut() else {
        return;
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };

    let required: std::collections::BTreeSet<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|required| required.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let validator = Validator::new(root);
    for (key, property_schema) in properties {
        let is_reference = property_schema
            .get("$ref")
            .and_then(Value::as_str)
            .is_some();
        let null_rejected =
            !is_reference && !validator.check_value_quiet(&Value::Null, property_schema);
        if object.get(key).is_some_and(Value::is_null)
            && !required.contains(key.as_str())
            && null_rejected
        {
            object.remove(key);
        } else if let Some(property_value) = object.get_mut(key) {
            normalize_optional_nulls(property_value, property_schema, root);
        }
    }
}

/// Coerce a value to a schema, upstream's `coerceWithJsonSchema`: the
/// AJV-compatible primitive rules plus recursive object, array, and union
/// coercion. JS's `Number()` accepts hex and `Infinity` spellings that Rust's
/// float parser rejects; those inputs stay uncoerced and fail validation.
fn coerce_with_json_schema(value: &Value, schema: &Value, root: &Value) -> Value {
    let schema = resolve_schema(schema, root);
    let mut next_value = value.clone();

    if let Some(all_of) = schema.get("allOf").and_then(Value::as_array) {
        for nested in all_of {
            next_value = coerce_with_json_schema(&next_value, nested, root);
        }
    }

    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
        next_value = coerce_with_union_schema(&next_value, any_of, root);
    }

    if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
        next_value = coerce_with_union_schema(&next_value, one_of, root);
    }

    let schema_types = get_schema_types(schema);
    let matches_union_member = schema_types.len() > 1
        && schema_types
            .iter()
            .any(|schema_type| matches_json_type(&next_value, schema_type));
    if !schema_types.is_empty() && !matches_union_member {
        for schema_type in &schema_types {
            let candidate = coerce_primitive_by_type(&next_value, schema_type);
            if candidate != next_value {
                next_value = candidate;
                break;
            }
        }
    }

    if schema_types
        .iter()
        .any(|schema_type| schema_type == "object")
        && next_value.is_object()
    {
        apply_schema_object_coercion(&mut next_value, schema, root);
    }

    if schema_types
        .iter()
        .any(|schema_type| schema_type == "array")
        && next_value.is_array()
    {
        apply_schema_array_coercion(&mut next_value, schema, root);
    }

    next_value
}

fn coerce_primitive_by_type(value: &Value, schema_type: &str) -> Value {
    match schema_type {
        "number" | "integer" => match value {
            Value::Null | Value::Bool(false) => Value::from(0),
            Value::String(text) if !text.trim().is_empty() => text
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|parsed| {
                    parsed.is_finite()
                        && (!matches!(schema_type, "integer") || parsed.fract() == 0.0)
                })
                .map_or_else(|| value.clone(), number_value),
            Value::Bool(true) => Value::from(1),
            other => other.clone(),
        },
        "boolean" => match value {
            Value::Null => Value::Bool(false),
            Value::String(text) => match text.as_str() {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                _ => value.clone(),
            },
            number @ Value::Number(_) => match number.as_f64() {
                Some(1.0) => Value::Bool(true),
                Some(0.0) => Value::Bool(false),
                _ => value.clone(),
            },
            _ => value.clone(),
        },
        "string" => match value {
            Value::Null => Value::String(String::new()),
            Value::Number(number) => Value::String(number.to_string()),
            Value::Bool(boolean) => Value::String(boolean.to_string()),
            _ => value.clone(),
        },
        "null" => match value {
            Value::String(text) if text.is_empty() => Value::Null,
            number @ Value::Number(_) if number.as_f64() == Some(0.0) => Value::Null,
            Value::Bool(false) => Value::Null,
            _ => value.clone(),
        },
        _ => value.clone(),
    }
}

/// A coerced number: integral values become integer numbers so they compare
/// equal to the wire's integers, upstream's single JS number type.
fn number_value(parsed: f64) -> Value {
    #[expect(
        clippy::cast_precision_loss,
        reason = "the i64 bounds as floats reproduce the JS number conversion range this check ports"
    )]
    let integer_range = (i64::MIN as f64..=i64::MAX as f64).contains(&parsed);
    if parsed.is_finite() && parsed.fract() == 0.0 && integer_range {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the range check bounds the float to the i64 window, upstream's single JS number type"
        )]
        return Value::from(parsed as i64);
    }
    Value::from(parsed)
}

fn apply_schema_object_coercion(value: &mut Value, schema: &Value, root: &Value) {
    let Some(object) = value.as_object_mut() else {
        return;
    };
    let properties = properties_schema(schema).cloned().unwrap_or_default();
    for (key, property_schema) in &properties {
        if let Some(property_value) = object.get_mut(key) {
            *property_value = coerce_with_json_schema(property_value, property_schema, root);
        }
    }

    if let Some(additional_schema @ Value::Object(_)) = schema.get("additionalProperties") {
        for (key, property_value) in object.iter_mut() {
            if properties.contains_key(key) {
                continue;
            }
            *property_value = coerce_with_json_schema(property_value, additional_schema, root);
        }
    }
}

fn apply_schema_array_coercion(value: &mut Value, schema: &Value, root: &Value) {
    let Some(array) = value.as_array_mut() else {
        return;
    };
    match schema.get("items") {
        Some(Value::Array(item_schemas)) => {
            for (index, item_schema) in item_schemas.iter().enumerate().take(array.len()) {
                array[index] = coerce_with_json_schema(&array[index], item_schema, root);
            }
        }
        Some(item_schema @ Value::Object(_)) => {
            for item in array.iter_mut() {
                *item = coerce_with_json_schema(item, item_schema, root);
            }
        }
        _ => {}
    }
}

fn coerce_with_union_schema(value: &Value, schemas: &[Value], root: &Value) -> Value {
    let validator = Validator::new(root);
    for schema in schemas {
        if validator.check_value_quiet(value, schema) {
            return value.clone();
        }
    }

    for schema in schemas {
        let candidate = coerce_with_json_schema(value, schema, root);
        if validator.check_value_quiet(&candidate, schema) {
            return candidate;
        }
    }
    value.clone()
}

/// Find a tool by name and validate the tool call's arguments against its
/// schema.
///
/// # Errors
/// Returns [`ValidationError`] when the tool is not found or validation
/// fails, with the same multi-line message the TypeScript version throws.
pub fn validate_tool_call(tools: &[Tool], tool_call: &ToolCall) -> Result<Value, ValidationError> {
    let Some(tool) = tools.iter().find(|tool| tool.name == tool_call.name) else {
        return Err(ValidationError(format!(
            "Tool \"{}\" not found",
            tool_call.name
        )));
    };
    validate_tool_arguments(tool, tool_call)
}

/// Validate tool-call arguments against the tool's schema, coercing values
/// where the schema permits.
///
/// # Errors
/// Returns [`ValidationError`] with the formatted per-error list and the
/// received arguments when validation fails.
pub fn validate_tool_arguments(
    tool: &Tool,
    tool_call: &ToolCall,
) -> Result<Value, ValidationError> {
    let mut args = Value::Object(tool_call.arguments.clone());
    normalize_optional_nulls(&mut args, &tool.parameters, &tool.parameters);

    let validator = Validator::new(&tool.parameters);
    let coerced = coerce_with_json_schema(&args, &tool.parameters, &tool.parameters);
    let args = if coerced == args
        || (args.is_object() && coerced.is_object())
        || validator.check_value_quiet(&coerced, &tool.parameters)
    {
        coerced
    } else {
        args
    };

    let check_errors = validator.check(&args);
    if check_errors.is_empty() {
        return Ok(args);
    }

    let errors = check_errors
        .iter()
        .map(|error| format!("  - {}: {}", format_validation_path(error), error.message))
        .collect::<Vec<_>>()
        .join("\n");

    let received = serde_json::to_string_pretty(&tool_call.arguments).unwrap_or_default();
    Err(ValidationError(format!(
        "Validation failed for tool \"{}\":\n{errors}\n\nReceived arguments:\n{received}",
        tool_call.name
    )))
}

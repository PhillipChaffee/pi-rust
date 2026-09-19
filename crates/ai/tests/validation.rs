//! The tool-argument validation port, from `test/validation.test.ts`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use pi_ai::types::{Tool, ToolCall};
use pi_ai::utils::validation::{validate_tool_arguments, validate_tool_call};
use serde_json::{Map, json};

fn create_tool_call_with_plain_schema(
    schema: &serde_json::Value,
    value: serde_json::Value,
) -> (Tool, ToolCall) {
    let tool = Tool {
        name: String::from("echo"),
        description: String::from("Echo tool"),
        parameters: json!({
            "type": "object",
            "properties": { "value": schema },
            "required": ["value"],
        }),
        constrained_sampling: None,
    };
    let mut arguments = Map::new();
    arguments.insert(String::from("value"), value);
    let tool_call = ToolCall {
        id: String::from("tool-1"),
        name: String::from("echo"),
        arguments,
        thought_signature: None,
        namespace: None,
    };
    (tool, tool_call)
}

fn validated_value(tool: &Tool, tool_call: &ToolCall) -> serde_json::Value {
    validate_tool_arguments(tool, tool_call).expect("validation passes")
}

#[test]
fn still_validates_without_generated_code() {
    // The CSP test swaps out the Function constructor; the interpreted
    // checker is the only path in this port, so validation works the same
    // way the interpreted fallback does.
    let tool = Tool {
        name: String::from("echo"),
        description: String::from("Echo tool"),
        parameters: json!({
            "type": "object",
            "properties": { "count": { "type": "number" } },
            "required": ["count"],
        }),
        constrained_sampling: None,
    };
    let mut arguments = Map::new();
    arguments.insert(String::from("count"), json!("42"));
    let tool_call = ToolCall {
        id: String::from("tool-1"),
        name: String::from("echo"),
        arguments,
        thought_signature: None,
        namespace: None,
    };

    assert_eq!(validated_value(&tool, &tool_call), json!({"count": 42}));
}

#[test]
fn coerces_serialized_plain_json_schemas_with_ajv_compatible_primitive_rules() {
    let passing_cases: Vec<(serde_json::Value, serde_json::Value, serde_json::Value)> = vec![
        (json!({"type": "number"}), json!("42"), json!(42)),
        (json!({"type": "number"}), json!(true), json!(1)),
        (json!({"type": "number"}), json!(null), json!(0)),
        (json!({"type": "integer"}), json!("42"), json!(42)),
        (json!({"type": "boolean"}), json!("true"), json!(true)),
        (json!({"type": "boolean"}), json!("false"), json!(false)),
        (json!({"type": "boolean"}), json!(1), json!(true)),
        (json!({"type": "boolean"}), json!(0), json!(false)),
        (json!({"type": "string"}), json!(null), json!("")),
        (json!({"type": "string"}), json!(true), json!("true")),
        (json!({"type": "null"}), json!(""), json!(null)),
        (json!({"type": "null"}), json!(0), json!(null)),
        (json!({"type": "null"}), json!(false), json!(null)),
        (
            json!({"type": ["number", "string"]}),
            json!("1"),
            json!("1"),
        ),
        (json!({"type": ["boolean", "number"]}), json!("1"), json!(1)),
    ];

    for (schema, input, expected) in passing_cases {
        let (tool, tool_call) = create_tool_call_with_plain_schema(&schema, input);
        assert_eq!(
            validated_value(&tool, &tool_call),
            json!({"value": expected}),
            "coercion case failed"
        );
    }
}

#[test]
fn treats_null_as_omission_for_optional_non_nullable_properties() {
    let tool = Tool {
        name: String::from("echo"),
        description: String::from("Echo tool"),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "offset": { "type": "number" },
                "nullable": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
                "metadata": {
                    "type": "object",
                    "properties": { "enabled": { "type": "boolean" } }
                },
            },
            "required": ["path"],
        }),
        constrained_sampling: None,
    };
    let mut arguments = Map::new();
    arguments.insert(String::from("path"), json!("file.txt"));
    arguments.insert(String::from("offset"), json!(null));
    arguments.insert(String::from("nullable"), json!(null));
    arguments.insert(String::from("metadata"), json!({"enabled": null}));
    let tool_call = ToolCall {
        id: String::from("tool-1"),
        name: String::from("echo"),
        arguments,
        thought_signature: None,
        namespace: None,
    };

    assert_eq!(
        validated_value(&tool, &tool_call),
        json!({"path": "file.txt", "nullable": null, "metadata": {}})
    );
}

#[test]
fn preserves_optional_nulls_whose_referenced_schema_is_nullable() {
    let tool = Tool {
        name: String::from("echo"),
        description: String::from("Echo tool"),
        parameters: json!({
            "type": "object",
            "properties": { "value": { "$ref": "#/$defs/value" } },
            "$defs": { "value": { "anyOf": [{ "type": "number" }, { "type": "null" }] } },
        }),
        constrained_sampling: None,
    };
    let mut arguments = Map::new();
    arguments.insert(String::from("value"), json!(null));
    let tool_call = ToolCall {
        id: String::from("tool-1"),
        name: String::from("echo"),
        arguments,
        thought_signature: None,
        namespace: None,
    };

    assert_eq!(validated_value(&tool, &tool_call), json!({"value": null}));
}

#[test]
fn preserves_a_value_that_already_matches_a_nullable_union_arm() {
    let tool = Tool {
        name: String::from("echo"),
        description: String::from("Echo tool"),
        parameters: json!({
            "type": "object",
            "properties": { "value": { "anyOf": [{ "type": "number" }, { "type": "null" }] } },
            "required": ["value"],
        }),
        constrained_sampling: None,
    };
    let mut arguments = Map::new();
    arguments.insert(String::from("value"), json!(null));
    let tool_call = ToolCall {
        id: String::from("tool-1"),
        name: String::from("echo"),
        arguments,
        thought_signature: None,
        namespace: None,
    };

    assert_eq!(validated_value(&tool, &tool_call), json!({"value": null}));
}

#[test]
fn preserves_a_value_that_already_matches_a_one_of_nullable_union_arm() {
    let (tool, tool_call) = create_tool_call_with_plain_schema(
        &json!({"oneOf": [{ "type": "number" }, { "type": "null" }]}),
        json!(null),
    );

    assert_eq!(validated_value(&tool, &tool_call), json!({"value": null}));
}

#[test]
fn still_coerces_nullable_unions_when_the_original_value_does_not_match_any_arm() {
    let (tool, tool_call) = create_tool_call_with_plain_schema(
        &json!({"anyOf": [{ "type": "number" }, { "type": "null" }]}),
        json!("42"),
    );

    assert_eq!(validated_value(&tool, &tool_call), json!({"value": 42}));
}

#[test]
fn accepts_null_for_nullable_array_schemas_with_items() {
    let (tool, tool_call) = create_tool_call_with_plain_schema(
        &json!({"type": ["array", "null"], "items": { "type": "string" }}),
        json!(null),
    );

    assert_eq!(validated_value(&tool, &tool_call), json!({"value": null}));
}

#[test]
fn rejects_invalid_coercions_for_serialized_plain_json_schemas() {
    let failing_cases: Vec<(&str, serde_json::Value)> = vec![
        ("boolean", json!("1")),
        ("boolean", json!("0")),
        ("null", json!("null")),
        ("integer", json!("42.1")),
    ];

    for (schema_type, input) in failing_cases {
        let (tool, tool_call) =
            create_tool_call_with_plain_schema(&json!({"type": schema_type}), input);
        let error = validate_tool_arguments(&tool, &tool_call).expect_err("the coercion must fail");
        assert!(
            error.0.contains("Validation failed"),
            "unexpected message: {}",
            error.0
        );
    }
}

#[test]
fn the_rejected_validation_message_lists_errors_and_received_arguments() {
    let (tool, tool_call) =
        create_tool_call_with_plain_schema(&json!({"type": "number"}), json!("not a number"));

    let error = validate_tool_arguments(&tool, &tool_call).expect_err("invalid input");
    assert!(
        error.0.starts_with("Validation failed for tool \"echo\":"),
        "{}",
        error.0
    );
    assert!(error.0.contains("Received arguments:"), "{}", error.0);
}

// --- Rust-native additions: schema keywords, coercion paths, and the
// null normalizer's recursion, on top of the ported suites ---

fn create_tool_call_with_raw_schema(
    schema: serde_json::Value,
    arguments: Map<String, serde_json::Value>,
) -> (Tool, ToolCall) {
    let tool = Tool {
        name: String::from("echo"),
        description: String::from("Echo tool"),
        parameters: schema,
        constrained_sampling: None,
    };
    let tool_call = ToolCall {
        id: String::from("tool-1"),
        name: String::from("echo"),
        arguments,
        thought_signature: None,
        namespace: None,
    };
    (tool, tool_call)
}

fn args_from(value: &serde_json::Value) -> Map<String, serde_json::Value> {
    value.as_object().cloned().expect("an object argument set")
}

#[test]
fn all_of_coerces_through_every_member_schema() {
    let (tool, tool_call) = create_tool_call_with_plain_schema(
        &json!({"allOf": [{"type": "number"}, {"type": "integer"}]}),
        json!("42"),
    );
    assert_eq![validated_value(&tool, &tool_call), json!({"value": 42})];

    let (tool, tool_call) = create_tool_call_with_plain_schema(
        &json!({"allOf": [{"type": "string"}, {"type": "boolean"}]}),
        json!(5),
    );
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("the allOf members disagree");
    assert![error.0.contains("must be boolean"), "{}", error.0];
    assert![
        error.to_string().starts_with("Validation failed for tool"),
        "{}",
        error.0
    ];
}

#[test]
fn a_value_matching_no_any_of_arm_is_rejected() {
    let (tool, tool_call) = create_tool_call_with_plain_schema(
        &json!({"anyOf": [{"type": "number"}, {"type": "boolean"}]}),
        json!({"unexpected": true}),
    );
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("no arm matches");
    assert![
        error.0.contains("must match a schema of anyOf"),
        "{}",
        error.0
    ];
}

#[test]
fn one_of_requires_exactly_one_matching_arm() {
    let (tool, tool_call) = create_tool_call_with_plain_schema(
        &json!({"oneOf": [{"type": "number"}, {"type": "integer"}]}),
        json!(5),
    );
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("two arms match");
    assert![
        error.0.contains("must match exactly one schema of oneOf"),
        "{}",
        error.0
    ];

    let (tool, tool_call) = create_tool_call_with_plain_schema(
        &json!({"oneOf": [{"type": "string"}, {"type": "boolean"}]}),
        json!({"unexpected": true}),
    );
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("no arm matches");
    assert![
        error.0.contains("must match exactly one schema of oneOf"),
        "{}",
        error.0
    ];
}

#[test]
fn const_and_enum_gate_the_exact_value() {
    let (tool, tool_call) = create_tool_call_with_plain_schema(&json!({"const": 7}), json!(8));
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("const mismatch");
    assert![
        error.0.contains("must be equal to constant 7"),
        "{}",
        error.0
    ];
    let (tool, tool_call) = create_tool_call_with_plain_schema(&json!({"const": 7}), json!(7));
    assert_eq![validated_value(&tool, &tool_call), json!({"value": 7})];

    let (tool, tool_call) =
        create_tool_call_with_plain_schema(&json!({"enum": ["a", "b"]}), json!("c"));
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("enum mismatch");
    assert![
        error
            .0
            .contains("must be equal to one of the allowed values: \"a\", \"b\""),
        "{}",
        error.0
    ];
    let (tool, tool_call) =
        create_tool_call_with_plain_schema(&json!({"enum": ["a", "b"]}), json!("a"));
    assert_eq![validated_value(&tool, &tool_call), json!({"value": "a"})];
}

#[test]
fn required_failures_name_the_missing_property_at_its_path() {
    let schema = json!({
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {"deep": {"type": "string"}},
                "required": ["deep"],
            },
        },
        "required": ["nested", "missing"],
    });
    let (tool, tool_call) =
        create_tool_call_with_raw_schema(schema, args_from(&json!({"nested": {}})));
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("required keys are missing");
    assert![
        error
            .0
            .contains("  - missing: must have required property 'missing'"),
        "{}",
        error.0
    ];
    assert![
        error
            .0
            .contains("  - nested.deep: must have required property 'deep'"),
        "{}",
        error.0
    ];
}

#[test]
fn additional_properties_are_rejected_or_checked_by_schema() {
    let schema = json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": false,
    });
    let (tool, tool_call) =
        create_tool_call_with_raw_schema(schema, args_from(&json!({"a": "ok", "extra": 1})));
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("the extra key is rejected");
    assert![
        error
            .0
            .contains("  - extra: must NOT have additional properties"),
        "{}",
        error.0
    ];

    let schema = json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": {"type": "number"},
    });
    let (tool, tool_call) =
        create_tool_call_with_raw_schema(schema, args_from(&json!({"a": "ok", "extra": "5"})));
    assert_eq![
        validated_value(&tool, &tool_call),
        json!({"a": "ok", "extra": 5})
    ];

    let (tool, tool_call) = create_tool_call_with_raw_schema(
        json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": {"type": "number"},
        }),
        args_from(&json!({"a": "ok", "extra": {}})),
    );
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("the extra object fails");
    assert![error.0.contains("  - extra: must be number"), "{}", error.0];
}

#[test]
fn tuple_and_uniform_array_items_coerce_and_check_per_index() {
    let (tool, tool_call) = create_tool_call_with_plain_schema(
        &json!({"type": "array", "items": [{"type": "number"}, {"type": "string"}]}),
        json!([true, 5]),
    );
    assert_eq![
        validated_value(&tool, &tool_call),
        json!({"value": [1, "5"]})
    ];

    let (tool, tool_call) = create_tool_call_with_raw_schema(
        json!({
            "type": "object",
            "properties": {
                "list": {"type": "array", "items": [{"type": "number"}, {"type": "string"}]},
            },
            "required": ["list"],
        }),
        args_from(&json!({"list": [{"bad": 1}, "ok"]})),
    );
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("index 0 fails");
    assert![
        error.0.contains("  - list.0: must be number"),
        "{}",
        error.0
    ];

    let (tool, tool_call) = create_tool_call_with_plain_schema(
        &json!({"type": "array", "items": {"type": "number"}}),
        json!(["1", 2]),
    );
    assert_eq![validated_value(&tool, &tool_call), json!({"value": [1, 2]})];

    let (tool, tool_call) = create_tool_call_with_raw_schema(
        json!({
            "type": "object",
            "properties": {"list": {"type": "array", "items": {"type": "number"}}},
            "required": ["list"],
        }),
        args_from(&json!({"list": [{}]})),
    );
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("the item fails");
    assert![
        error.0.contains("  - list.0: must be number"),
        "{}",
        error.0
    ];
}

#[test]
fn the_null_json_type_matches_null_values() {
    let (tool, tool_call) =
        create_tool_call_with_plain_schema(&json!({"type": ["null", "string"]}), json!(null));
    assert_eq![validated_value(&tool, &tool_call), json!({"value": null})];
    let (tool, tool_call) =
        create_tool_call_with_plain_schema(&json!({"type": "null"}), json!(null));
    assert_eq![validated_value(&tool, &tool_call), json!({"value": null})];
}

#[test]
fn refs_resolve_through_pointers_into_the_document() {
    // The whole-document pointer resolves to the root schema, so the
    // property value must itself be an object.
    let schema = json!({
        "type": "object",
        "properties": {"v": {"$ref": "#/"}},
    });
    let (tool, tool_call) = create_tool_call_with_raw_schema(schema, args_from(&json!({"v": {}})));
    assert_eq![validated_value(&tool, &tool_call), json!({"v": {}})];

    // An array-indexed pointer resolves to the tuple element.
    // An array-indexed pointer resolves to the tuple element.
    let schema = json!({
        "$defs": {"pair": [{"type": "string"}, {"type": "number"}]},
        "type": "object",
        "properties": {"v": {"$ref": "#/$defs/pair/1"}},
        "required": ["v"],
    });
    let (tool, tool_call) = create_tool_call_with_raw_schema(schema, args_from(&json!({"v": "x"})));
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("v must be a number");
    assert![error.0.contains("  - v: must be number"), "{}", error.0];

    // Pointer escapes: ~1 is a literal slash, ~0 a literal tilde.
    let schema = json!({
        "$defs": {"a/b": {"type": "number"}, "a~b": {"type": "number"}},
        "type": "object",
        "properties": {
            "v": {"$ref": "#/$defs/a~1b"},
            "w": {"$ref": "#/$defs/a~0b"},
        },
        "required": ["v", "w"],
    });
    let (tool, tool_call) =
        create_tool_call_with_raw_schema(schema, args_from(&json!({"v": 1, "w": 2})));
    assert_eq![validated_value(&tool, &tool_call), json!({"v": 1, "w": 2})];

    // A pointer that walks into a scalar resolves to nothing; the property
    // is left unchecked and passes.
    let schema = json!({
        "$defs": {"scalar": 5},
        "type": "object",
        "properties": {"v": {"$ref": "#/$defs/scalar/inner"}},
    });
    let (tool, tool_call) = create_tool_call_with_raw_schema(schema, args_from(&json!({"v": 42})));
    assert_eq![validated_value(&tool, &tool_call), json!({"v": 42})];
}

#[test]
fn optional_null_normalization_recurses_through_arrays() {
    // A uniform object-items schema recurses per item.
    let schema = json!({
        "type": "object",
        "properties": {
            "list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"opt": {"type": "string"}},
                },
            },
        },
        "required": ["list"],
    });
    let (tool, tool_call) = create_tool_call_with_raw_schema(
        schema,
        args_from(&json!({"list": [{"opt": null}, {"other": 1}]})),
    );
    assert_eq![
        validated_value(&tool, &tool_call),
        json!({"list": [{}, {"other": 1}]})
    ];

    // A tuple-items schema recurses per index; nullable members keep their
    // nulls while object members shed them.
    let schema = json!({
        "type": "object",
        "properties": {
            "list": {
                "type": "array",
                "items": [
                    {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    {"type": "object", "properties": {"opt": {"type": "string"}}},
                ],
            },
        },
        "required": ["list"],
    });
    let (tool, tool_call) = create_tool_call_with_raw_schema(
        schema,
        args_from(&json!({"list": [null, {"opt": null}]})),
    );
    assert_eq![
        validated_value(&tool, &tool_call),
        json!({"list": [null, {}]})
    ];
}

#[test]
fn primitive_coercion_covers_the_remaining_ajv_rules() {
    let cases: Vec<(serde_json::Value, serde_json::Value, serde_json::Value)> = vec![
        // Numbers accept surrounding whitespace.
        (json!({"type": "number"}), json!(" 42 "), json!(42)),
        // Fractional numbers stay floats.
        (json!({"type": "number"}), json!("42.5"), json!(42.5)),
        // Numbers stringify; booleans stringify to their Rust spelling.
        (json!({"type": "string"}), json!(42), json!("42")),
        (json!({"type": "string"}), json!(true), json!("true")),
        // The boolean zero rule reads float numbers too.
        (json!({"type": "boolean"}), json!(0.0), json!(false)),
        (json!({"type": "boolean"}), json!(1.0), json!(true)),
        (json!({"type": "boolean"}), json!(null), json!(false)),
    ];
    for (schema, input, expected) in cases {
        let (tool, tool_call) = create_tool_call_with_plain_schema(&schema, input);
        assert_eq!(
            validated_value(&tool, &tool_call),
            json!({"value": expected}),
            "coercion case failed"
        );
    }

    // A boolean the coercer cannot move fails validation with its type.
    let (tool, tool_call) =
        create_tool_call_with_plain_schema(&json!({"type": "boolean"}), json!(2));
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("2 is not boolean");
    assert![error.0.contains("must be boolean"), "{}", error.0];
}

#[test]
fn validate_tool_call_finds_the_tool_and_reports_missing_ones() {
    let (tool, tool_call) =
        create_tool_call_with_plain_schema(&json!({"type": "number"}), json!(42));
    let error = validate_tool_call(&[], &tool_call).expect_err("tool is missing");
    assert_eq!(error.0, "Tool \"echo\" not found");

    let validated = validate_tool_call(&[tool], &tool_call).expect("the tool is found");
    assert_eq!(validated, json!({"value": 42}));
}

#[test]
fn arrays_without_item_schemas_and_unknown_type_names_settle_cleanly() {
    // An array schema with no `items` constrains nothing.
    let (tool, tool_call) =
        create_tool_call_with_plain_schema(&json!({"type": "array"}), json!([1, "two", null]));
    assert_eq![
        validated_value(&tool, &tool_call),
        json!({"value": [1, "two", null]})
    ];

    // An `items` keyword that is neither an array nor a schema object
    // constrains nothing.
    let (tool, tool_call) =
        create_tool_call_with_plain_schema(&json!({"type": "array", "items": true}), json!([]));
    assert_eq![validated_value(&tool, &tool_call), json!({"value": []})];

    // An unknown type name matches nothing: a value of that "type" always
    // fails with the type listed.
    let (tool, tool_call) =
        create_tool_call_with_plain_schema(&json!({"type": "widget"}), json!("x"));
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("no value is a widget");
    assert![error.0.contains("must be widget"), "{}", error.0];
}

#[test]
fn a_root_level_enum_failure_reports_the_root_path() {
    let (tool, tool_call) =
        create_tool_call_with_raw_schema(json!({"enum": [{"a": 1}]}), Map::new());
    let error = validate_tool_arguments(&tool, &tool_call).expect_err("the root value mismatches");
    assert![
        error
            .0
            .contains("  - root: must be equal to one of the allowed values: {\"a\":1}"),
        "{}",
        error.0
    ];
}

//! JSON-schema constrained sampling, ported from the module-local parts of
//! `packages/ai/test/constrained-sampling.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (the grammar-variant half ports
//! with the OpenAI Responses wire API).

#![expect(
    clippy::expect_used,
    reason = "the tests pin conversion outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::api::constrained_sampling::{
    make_strict_json_schema, resolve_json_schema_strict_sampling,
};
use pi_ai::types::{ConstrainedSamplingSetting, Strictness, Tool};

fn sample_tool(
    constrained_sampling: Option<ConstrainedSamplingSetting>,
    parameters: serde_json::Value,
) -> Tool {
    Tool {
        name: "sample_tool".to_owned(),
        description: "Sample tool".to_owned(),
        parameters,
        constrained_sampling,
    }
}

const fn json_schema_setting(strict: Strictness) -> ConstrainedSamplingSetting {
    ConstrainedSamplingSetting::Config(pi_ai::types::ConstrainedSamplingConfig::JsonSchema {
        strict,
    })
}

#[test]
fn derives_strict_provider_schemas_without_changing_tool_definitions() {
    let parameters = serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" },
            "offset": { "type": "number" },
            "metadata": {
                "type": "object",
                "properties": { "enabled": { "type": "boolean" } },
            },
            "nullable": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
        },
        "required": ["path", "metadata"],
    });
    let strict = make_strict_json_schema(&parameters).expect("converts");

    assert!(parameters.get("additionalProperties").is_none());
    assert_eq!(
        parameters["required"],
        serde_json::json!(["path", "metadata"])
    );
    assert_eq!(
        strict["additionalProperties"],
        serde_json::Value::Bool(false)
    );
    assert_eq!(
        strict["required"],
        serde_json::json!(["path", "offset", "metadata", "nullable"]),
    );
    assert_eq!(
        strict["properties"]["offset"],
        serde_json::json!({ "anyOf": [{ "type": "number" }, { "type": "null" }] }),
    );
    assert_eq!(
        strict["properties"]["metadata"],
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["enabled"],
            "properties": { "enabled": { "anyOf": [{ "type": "boolean" }, { "type": "null" }] } },
        }),
    );
    assert_eq!(
        strict["properties"]["nullable"],
        serde_json::json!({ "anyOf": [{ "type": "string" }, { "type": "null" }] }),
    );
}

#[test]
fn falls_back_or_rejects_schemas_that_cannot_be_safely_converted() {
    let cases: Vec<(serde_json::Value, &str)> = vec![
        (
            serde_json::json!({
                "type": "object",
                "properties": {
                    "metadata": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": { "type": "string" },
                    },
                },
                "required": ["metadata"],
            }),
            "additionalProperties is unsupported",
        ),
        (
            serde_json::json!({
                "type": "object",
                "allOf": [
                    { "type": "object", "properties": { "a": { "type": "string" } }, "required": ["a"] },
                    { "type": "object", "properties": { "b": { "type": "number" } }, "required": ["b"] },
                ],
            }),
            "allOf schemas are unsupported",
        ),
        (
            serde_json::json!({
                "type": "object",
                "properties": {
                    "value": { "anyOf": [{ "type": "object", "properties": { "nested": { "type": "string" } }, "required": ["nested"] }, { "type": "null" }] },
                },
                "required": ["value"],
            }),
            "object and array unions are unsupported",
        ),
        (
            serde_json::json!({
                "type": "object",
                "properties": { "child": { "$ref": "https://example.com/child.json" } },
                "required": ["child"],
            }),
            "$ref schemas are unsupported",
        ),
    ];

    for (parameters, expected_error) in cases {
        let mut tool = sample_tool(
            Some(json_schema_setting(Strictness::Prefer)),
            parameters.clone(),
        );

        let error = make_strict_json_schema(&parameters)
            .expect_err("the schema leaves the strict subset")
            .0;
        assert!(error.contains(expected_error), "got: {error}");
        assert_eq!(resolve_json_schema_strict_sampling(&tool, true), Ok(None));
        tool.constrained_sampling = Some(json_schema_setting(Strictness::Require));
        assert!(
            resolve_json_schema_strict_sampling(&tool, true)
                .err()
                .is_some_and(|message| message.contains(expected_error)),
            "expected rejection containing {expected_error}"
        );
    }
}

#[test]
fn rejects_require_strict_when_strict_tools_are_unsupported() {
    let tool = sample_tool(
        Some(json_schema_setting(Strictness::Require)),
        serde_json::json!({
            "type": "object",
            "properties": { "payload": { "type": "string" } },
            "required": ["payload"],
        }),
    );
    let error =
        resolve_json_schema_strict_sampling(&tool, false).expect_err("strict tools unsupported");
    assert_eq!(
        error,
        "Tool \"sample_tool\" requires JSON-schema constrained sampling, but strict tools are unsupported.",
    );
}

#[test]
fn resolves_strict_only_when_the_schema_converts() {
    let tool = sample_tool(
        Some(json_schema_setting(Strictness::Prefer)),
        serde_json::json!({
            "type": "object",
            "properties": { "payload": { "type": "string" } },
            "required": ["payload"],
        }),
    );
    assert_eq!(
        resolve_json_schema_strict_sampling(&tool, true),
        Ok(Some(true))
    );
    assert_eq!(resolve_json_schema_strict_sampling(&tool, false), Ok(None));

    let unconstrained = sample_tool(None, serde_json::json!({}));
    assert_eq!(
        resolve_json_schema_strict_sampling(&unconstrained, true),
        Ok(None)
    );

    let grammar_tool = sample_tool(
        Some(ConstrainedSamplingSetting::Config(
            pi_ai::types::ConstrainedSamplingConfig::Grammar {
                variants: pi_ai::types::GrammarVariants::default(),
            },
        )),
        serde_json::json!({}),
    );
    assert_eq!(
        resolve_json_schema_strict_sampling(&grammar_tool, true),
        Ok(None)
    );
}

/// Port-added: the remaining rejection branches, each one wire-vocabulary
/// pin upstream's `makeJsonSchemaNodeStrict` carries.
#[test]
fn rejects_the_remaining_unsupported_schema_shapes() {
    let cases: Vec<(serde_json::Value, &str)> = vec![
        // A nested non-object schema (a boolean schema) fails the walk.
        (
            serde_json::json!({
                "type": "object",
                "properties": { "flag": true },
            }),
            "boolean schemas are unsupported",
        ),
        (
            serde_json::json!({ "type": "object", "anyOf": [] }),
            "anyOf must contain at least one schema",
        ),
        (
            serde_json::json!({
                "type": "object",
                "items": [{ "type": "string" }],
            }),
            "tuple schemas are unsupported",
        ),
        (
            serde_json::json!({ "type": "string", "properties": {} }),
            "properties require type object",
        ),
        (
            serde_json::json!({ "type": "object", "properties": [] }),
            "object properties must be a schema map",
        ),
        (
            serde_json::json!({
                "type": "object",
                "properties": { "a": { "type": "string" } },
                "required": "a",
            }),
            "object required must be a string array",
        ),
        (
            serde_json::json!({
                "type": "object",
                "properties": { "a": { "type": "string" } },
                "required": ["b"],
            }),
            "required contains an unknown property",
        ),
        // A root that is not an object fails before and after the walk.
        (serde_json::json!(true), "root schema must have type object"),
        (
            serde_json::json!({ "type": "string" }),
            "root schema must have type object",
        ),
    ];

    for (schema, expected_error) in cases {
        let error = make_strict_json_schema(&schema)
            .expect_err("the schema leaves the strict subset")
            .0;
        assert!(error.contains(expected_error), "got: {error}");
        // The error renders as its message, upstream's Error subclass.
        let rendered = make_strict_json_schema(&schema)
            .expect_err("the schema leaves the strict subset")
            .to_string();
        assert_eq!(rendered, error);
    }
}

/// Port-added: a strict schema walks into object-valued `items` and
/// requires the element schema; only tuple `items` reject.
#[test]
fn object_valued_items_walk_into_the_element_schema() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "tags": { "type": "object", "items": { "type": "string" } } },
    });

    let strict = make_strict_json_schema(&schema).expect("converts");

    // The element schema survives the walk; the tags property itself gains the
    // null union and closes.
    let tags = &strict["properties"]["tags"]["anyOf"][0];
    assert_eq!(tags["items"], serde_json::json!({ "type": "string" }));
    assert_eq!(tags["additionalProperties"], serde_json::Value::Bool(false));
}

/// Port-added: a type-array union naming null admits null, as do `const`
/// null, enum null, and a null variant nested in `anyOf`.
#[test]
fn null_admitting_properties_keep_their_null() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "typed": { "type": ["string", "null"] },
            "const": { "const": null },
            "enum": { "enum": ["a", null] },
            "union": { "anyOf": [{ "type": "string" }, { "type": "null" }] },
        },
    });

    let strict = make_strict_json_schema(&schema).expect("converts");

    // Every property becomes required; none gains a synthesized null union.
    assert_eq!(
        strict["required"],
        serde_json::json!(["typed", "const", "enum", "union"]),
    );
    for key in ["typed", "const", "enum", "union"] {
        assert!(
            strict["properties"][key].get("anyOf").is_none()
                || strict["properties"][key]["anyOf"]
                    .as_array()
                    .is_some_and(|variants| variants.len() == 2),
            "the {key} property keeps its own null admission",
        );
    }
}

/// Port-added: array-typed union members and boolean variants reject with
/// the union rejection and the boolean-schema rejection respectively.
#[test]
fn array_unions_and_boolean_variants_reject() {
    let array_union = serde_json::json!({
        "type": "object",
        "properties": {
            "value": { "anyOf": [{ "type": ["array", "null"] }] },
        },
    });
    let error = make_strict_json_schema(&array_union)
        .expect_err("array unions are unsupported")
        .0;
    assert!(
        error.contains("object and array unions are unsupported"),
        "got: {error}"
    );

    let boolean_variant = serde_json::json!({
        "type": "object",
        "properties": { "value": { "anyOf": [true] } },
    });
    let error = make_strict_json_schema(&boolean_variant)
        .expect_err("boolean schemas are unsupported")
        .0;
    assert!(
        error.contains("boolean schemas are unsupported"),
        "got: {error}"
    );
}

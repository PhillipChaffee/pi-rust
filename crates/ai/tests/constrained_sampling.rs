//! JSON-schema constrained sampling, ported from the module-local parts of
//! `packages/ai/test/constrained-sampling.test.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (the grammar-variant half ports
//! with the OpenAI Responses wire API).

#![expect(
    clippy::expect_used,
    reason = "the tests pin conversion outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::api::constrained_sampling::{
    get_json_schema_tool_parameters, make_strict_json_schema, resolve_json_schema_strict_sampling,
};
use pi_ai::types::{ConstrainedSamplingConfig, ConstrainedSamplingSetting, Strictness, Tool};

mod common;

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
    ConstrainedSamplingSetting::Config(ConstrainedSamplingConfig::JsonSchema { strict })
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
            ConstrainedSamplingConfig::Grammar {
                variants: GrammarVariants::default(),
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

// ---------------------------------------------------------------------------
// Grammar-constrained sampling (the deferred grammar half of the upstream
// constrained-sampling suite)
// ---------------------------------------------------------------------------

use pi_ai::api::constrained_sampling::{
    GrammarConstrainedSampling, GrammarToolInputJsonBuffer, append_grammar_tool_input_json_delta,
    resolve_grammar_constrained_sampling,
};
use pi_ai::api::openai_responses_shared::{
    ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions, convert_responses_messages,
    convert_responses_tools,
};
use pi_ai::types::{
    AssistantBlock, Context, GrammarFormat, GrammarVariants, Message, Model, StopReason,
    ToolResultBlock,
};
use serde_json::json;

fn grammar_tool(variants: GrammarVariants) -> Tool {
    Tool {
        name: "sample_tool".to_owned(),
        description: "Sample tool".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "payload": { "type": "string" } },
            "required": ["payload"],
            "additionalProperties": false,
        }),
        constrained_sampling: Some(ConstrainedSamplingSetting::Config(
            ConstrainedSamplingConfig::Grammar { variants },
        )),
    }
}

fn lark_tool() -> Tool {
    grammar_tool(
        std::iter::once((GrammarFormat::OpenaiLark, "start: /[a-z]+/".to_owned())).collect(),
    )
}

/// The grammar variant binds to the custom-tool grammar shape and the
/// schema's single required string property feeds its input.
#[test]
fn resolves_the_lark_grammar_variant_with_its_input_property() {
    let resolved = resolve_grammar_constrained_sampling(&lark_tool(), true)
        .expect("resolves")
        .expect("the grammar binds");

    assert_eq!(
        resolved,
        GrammarConstrainedSampling {
            format: GrammarFormat::OpenaiLark,
            definition: "start: /[a-z]+/".to_owned(),
            input_property: "payload".to_owned(),
        }
    );
}

#[test]
fn rejects_grammar_tools_without_a_supported_variant() {
    let empty = grammar_tool(GrammarVariants::default());

    let error = resolve_grammar_constrained_sampling(&empty, true).expect_err("no usable variant");
    assert_eq!(
        error,
        "Tool \"sample_tool\" cannot use grammar constrained sampling: no supported grammar variant was provided."
    );
}

/// A grammar tool without provider support falls back to the function shape
/// without `strict`, and `constrainedSampling: false` matches an
/// unconstrained tool.
#[test]
fn grammar_tools_fall_back_to_function_definitions_without_support() {
    let grammar = lark_tool();
    let fallback = convert_responses_tools(
        std::slice::from_ref(&grammar),
        Some(&ConvertResponsesToolsOptions {
            supports_openai_grammar_tools: Some(false),
            supports_strict_mode: Some(false),
            ..ConvertResponsesToolsOptions::default()
        }),
    )
    .expect("the fallback conversion");
    assert_eq!(fallback[0]["type"], json!("function"));
    assert_eq!(fallback[0]["name"], json!("sample_tool"));
    assert!(fallback[0].get("strict").is_none());

    let mut disabled = grammar;
    disabled.constrained_sampling = Some(ConstrainedSamplingSetting::Disabled(false));
    let converted = convert_responses_tools(
        std::slice::from_ref(&disabled),
        Some(&ConvertResponsesToolsOptions {
            supports_openai_grammar_tools: Some(true),
            ..ConvertResponsesToolsOptions::default()
        }),
    )
    .expect("converts");
    let unconstrained_tool = Tool {
        constrained_sampling: None,
        ..disabled
    };
    let unconstrained = convert_responses_tools(
        std::slice::from_ref(&unconstrained_tool),
        Some(&ConvertResponsesToolsOptions {
            supports_openai_grammar_tools: Some(true),
            ..ConvertResponsesToolsOptions::default()
        }),
    )
    .expect("converts");
    assert_eq!(converted, unconstrained);
}

/// The grammar tool rides the Responses `custom` shape with its lark format.
#[test]
fn converts_grammar_tools_to_the_custom_wire_shape() {
    let converted = convert_responses_tools(
        &[lark_tool()],
        Some(&ConvertResponsesToolsOptions {
            supports_openai_grammar_tools: Some(true),
            ..ConvertResponsesToolsOptions::default()
        }),
    )
    .expect("converts");

    assert_eq!(
        converted[0],
        json!({
            "type": "custom",
            "name": "sample_tool",
            "description": "Sample tool",
            "format": {
                "type": "grammar",
                "syntax": "openai_lark",
                "definition": "start: /[a-z]+/",
            },
        })
    );
}

fn responses_model() -> Model {
    Model {
        id: "gpt-test".to_owned(),
        name: "GPT Test".to_owned(),
        api: pi_ai::types::Api::from("openai-responses"),
        provider: pi_ai::types::ProviderId::from("openai"),
        base_url: "https://api.openai.com/v1".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text, pi_ai::types::Modality::Image],
        cost: pi_ai::types::ModelCost::default(),
        context_window: 128_000,
        max_tokens: 4096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn grammar_replay_context(arguments: serde_json::Map<String, serde_json::Value>) -> Context {
    let timestamp = pi_ai::auth::resolve::now_ms();
    Context {
        messages: vec![
            Message::Assistant(pi_ai::types::AssistantMessage {
                content: vec![AssistantBlock::ToolCall(pi_ai::types::ToolCall {
                    id: "call_1|ctc_1".to_owned(),
                    name: "sample_tool".to_owned(),
                    arguments,
                    thought_signature: None,
                    namespace: None,
                })],
                usage: pi_ai::types::Usage::default(),
                stop_reason: StopReason::ToolUse,
                ..common::assistant_message_with_content(
                    "openai-responses",
                    "openai",
                    "gpt-test",
                    Vec::new(),
                )
            }),
            Message::ToolResult(pi_ai::types::ToolResultMessage {
                tool_call_id: "call_1|ctc_1".to_owned(),
                tool_name: "sample_tool".to_owned(),
                content: vec![ToolResultBlock::Text(pi_ai::types::TextContent {
                    text: "done".to_owned(),
                    text_signature: None,
                })],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: false,
                timestamp,
            }),
        ],
        ..Context::default()
    }
}

fn grammar_replay_options() -> ConvertResponsesMessagesOptions {
    ConvertResponsesMessagesOptions {
        grammar_tool_input_properties: Some(
            std::iter::once(("sample_tool".to_owned(), "payload".to_owned())).collect(),
        ),
        ..ConvertResponsesMessagesOptions::default()
    }
}

/// A replayed custom tool call's arguments must carry the string input its
/// grammar streams through, and the valid call replays as the wire's custom
/// item pair.
#[test]
fn replays_grammar_calls_as_custom_responses_items() {
    let model = responses_model();
    let options = grammar_replay_options();
    let allowed = std::iter::once("openai".to_owned()).collect();

    for invalid_arguments in [
        serde_json::Map::new(),
        serde_json::Map::from_iter([("payload".to_owned(), json!(42))]),
    ] {
        let error = convert_responses_messages(
            &model,
            &grammar_replay_context(invalid_arguments.clone()),
            &allowed,
            Some(&options),
        )
        .expect_err("the grammar input is not a string");
        assert_eq!(
            error,
            "Grammar tool call \"sample_tool\" requires argument \"payload\" to be a string."
        );
    }

    let messages = convert_responses_messages(
        &model,
        &grammar_replay_context(serde_json::Map::from_iter([(
            "payload".to_owned(),
            json!("abc"),
        )])),
        &allowed,
        Some(&options),
    )
    .expect("replays");

    assert!(
        messages.iter().any(|message| *message
            == json!({
                "type": "custom_tool_call",
                "id": "ctc_1",
                "call_id": "call_1",
                "name": "sample_tool",
                "input": "abc",
            })),
        "got: {messages:?}"
    );
    assert!(
        messages.iter().any(|message| *message
            == json!({
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": "done",
            })),
        "messages: {messages:?}"
    );
}

/// The streamed grammar input accumulates append-only through the JSON
/// wrapper buffer.
#[test]
fn keeps_grammar_input_json_deltas_append_only() {
    let mut buffer = GrammarToolInputJsonBuffer::default();
    let first = append_grammar_tool_input_json_delta(&mut buffer, "payload", "a\"", false)
        .expect("the first delta")
        .expect("the first delta adds content");
    let second = append_grammar_tool_input_json_delta(&mut buffer, "payload", "a\"\nb", true)
        .expect("the second delta")
        .expect("the second delta adds content");

    let parsed: serde_json::Value =
        serde_json::from_str(&format!("{first}{second}")).expect("the deltas parse");
    assert_eq!(parsed, json!({ "payload": "a\"\nb" }));
    assert_eq!(
        append_grammar_tool_input_json_delta(&mut buffer, "payload", "a\"\nb", true),
        Ok(None)
    );
    let error = append_grammar_tool_input_json_delta(&mut buffer, "payload", "changed", true)
        .expect_err("the input changed after it closed");
    assert_eq!(
        error,
        "grammar tool input for property \"payload\" changed after it was closed"
    );
}

// ---------------------------------------------------------------------------
// Port-added: the strict-schema and grammar edges the upstream suites reach
// only implicitly
// ---------------------------------------------------------------------------

#[test]
fn the_schema_walkers_read_their_remaining_type_shapes() {
    // A schema whose `type` is neither a string nor a string array is a
    // non-container, so the walk reports the property requirement.
    let tool = sample_tool(
        Some(json_schema_setting(Strictness::Prefer)),
        json!({"type": 42, "properties": {"a": {}}}),
    );
    let error = get_json_schema_tool_parameters(&tool, Some(true)).expect_err("the walk rejects");
    assert![
        error.0.contains("properties require type object"),
        "{}",
        error.0
    ];

    // `schema_allows_null` over a numeric `type` property stays strict.
    let tool = sample_tool(
        Some(json_schema_setting(Strictness::Prefer)),
        json!({
            "type": "object",
            "properties": {"a": {"type": 7}},
            "required": ["a"],
            "additionalProperties": false,
        }),
    );
    let parameters = get_json_schema_tool_parameters(&tool, Some(true)).expect("the walk converts");
    assert_eq!(parameters["required"], json!(["a"]));
}

#[test]
fn the_object_items_recursion_walks_into_array_items() {
    // A non-tuple `items` object walks into the element schema.
    let tool = sample_tool(
        Some(json_schema_setting(Strictness::Prefer)),
        json!({
            "type": "object",
            "properties": {"list": {"type": "array", "items": {"type": "string"}}},
            "required": ["list"],
            "additionalProperties": false,
        }),
    );
    let strict = resolve_json_schema_strict_sampling(&tool, true)
        .expect("the resolution runs")
        .expect("the strict conversion runs");
    let parameters =
        get_json_schema_tool_parameters(&tool, Some(strict)).expect("the walk converts");
    assert_eq!(
        parameters["properties"]["list"]["items"]["type"],
        json!("string")
    );
}

#[test]
fn the_grammar_inference_rejects_the_remaining_schema_shapes() {
    // A non-object parameter schema.
    let tool = grammar_tool([(GrammarFormat::OpenaiLark, "start: /[a-z]+/".to_owned())].into());
    let tool = Tool {
        parameters: json!({"type": "string"}),
        ..tool
    };
    let error = resolve_grammar_constrained_sampling(&tool, true)
        .expect_err("the non-object schema rejects");
    assert_eq!(
        error,
        "Tool \"sample_tool\" cannot use grammar constrained sampling: grammar constrained sampling requires an object parameter schema."
    );

    // No required array / not exactly one required property.
    for parameters in [
        json!({"type": "object", "properties": {"text": {"type": "string"}}}),
        json!({"type": "object", "properties": {"a": {}, "b": {}}, "required": ["a", "b"]}),
        json!({"type": "object", "required": "text"}),
    ] {
        let tool = Tool {
            parameters,
            ..grammar_tool([(GrammarFormat::OpenaiLark, "start: /[a-z]+/".to_owned())].into())
        };
        let error =
            resolve_grammar_constrained_sampling(&tool, true).expect_err("the schema rejects");
        assert![
            error.contains("requires exactly one required string property"),
            "{error}"
        ];
    }

    // A missing properties entry and a non-string property type.
    for parameters in [
        json!({"type": "object", "required": ["text"]}),
        json!({"type": "object", "properties": {"text": {"type": "number"}}, "required": ["text"]}),
    ] {
        let tool = Tool {
            parameters,
            ..grammar_tool([(GrammarFormat::OpenaiLark, "start: /[a-z]+/".to_owned())].into())
        };
        let error =
            resolve_grammar_constrained_sampling(&tool, true).expect_err("the schema rejects");
        assert![
            error.contains("requires a properties entry for text")
                || error.contains("must have type string"),
            "{error}"
        ];
    }
}

#[test]
fn the_regex_variant_and_the_error_wrap_spell_their_wire_forms() {
    // A regex-only variant selects the regex encoding.
    let tool = grammar_tool([(GrammarFormat::OpenaiRegex, "^[a-z]+$".to_owned())].into());
    let grammar = resolve_grammar_constrained_sampling(&tool, true)
        .expect("the resolution runs")
        .expect("the regex variant resolves");
    assert_eq![grammar.format, GrammarFormat::OpenaiRegex];
    assert_eq![grammar.definition, "^[a-z]+$"];

    // The property-schema errors wrap the tool name.
    let tool = Tool {
        parameters: json!({"type": "object", "required": ["text"]}),
        ..grammar_tool([(GrammarFormat::OpenaiLark, "start: /[a-z]+/".to_owned())].into())
    };
    let error = resolve_grammar_constrained_sampling(&tool, true)
        .expect_err("the missing properties entry rejects");
    assert_eq!(
        error,
        "Tool \"sample_tool\" cannot use grammar constrained sampling: grammar constrained sampling requires a properties entry for text."
    );
}

/// Port-added: the structured-union reader treats an object carrying
/// `properties` and one carrying `items` as structured, so each rejects
/// inside `anyOf` with the union rejection, while a non-string `type` names
/// no container — upstream's `isStructuredSchema` match arms.
#[test]
fn the_structured_union_reader_reads_its_implicit_shapes() {
    for variant in [
        serde_json::json!({ "properties": { "a": { "type": "string" } } }),
        serde_json::json!({ "items": { "type": "string" } }),
    ] {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "value": { "anyOf": [variant] } },
        });
        let error = make_strict_json_schema(&schema)
            .expect_err("the structured variant rejects")
            .0;
        assert!(
            error.contains("object and array unions are unsupported"),
            "got: {error}"
        );
    }

    // A `type` of neither spelling names no container: the variant converts
    // as a leaf, upstream's `Vec::new()` arm.
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "value": { "anyOf": [{ "type": 7 }] } },
    });
    let strict = make_strict_json_schema(&schema).expect("the variant converts");
    assert_eq!(
        strict["properties"]["value"]["anyOf"][0]["anyOf"][0]["type"],
        json!(7)
    );
}

/// Port-added: the grammar input buffer rejects a shrinking (or changing)
/// input, and a repeated identical open delta adds nothing,
/// upstream's `appendGrammarToolInputJsonDelta` error arms.
#[test]
fn the_grammar_input_delta_rejects_non_monotonic_and_empty_updates() {
    let mut buffer = GrammarToolInputJsonBuffer::default();
    append_grammar_tool_input_json_delta(&mut buffer, "payload", "ab", false)
        .expect("the first delta")
        .expect("the first delta adds content");

    // A next input that does not extend the accumulated input rejects.
    let error = append_grammar_tool_input_json_delta(&mut buffer, "payload", "xy", false)
        .expect_err("the input changed non-monotonically");
    assert_eq!(
        error,
        "grammar tool input for property \"payload\" changed non-monotonically"
    );

    // Repeating the accumulated input without closing adds nothing.
    assert_eq!(
        append_grammar_tool_input_json_delta(&mut buffer, "payload", "ab", false),
        Ok(None)
    );
}

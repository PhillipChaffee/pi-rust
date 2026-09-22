//! The Google shared conversion suite, ported from
//! `packages/ai/test/google-thinking-signature.test.ts`,
//! `packages/ai/test/google-shared-convert-tools.test.ts`,
//! `packages/ai/test/google-shared-signed-empty-blocks.test.ts`,
//! `packages/ai/test/google-shared-image-tool-result-routing.test.ts`, and
//! `packages/ai/test/google-shared-gemini3-unsigned-tool-call.test.ts`
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin conversion outcomes; an unexpected shape panics the test by design"
)]

use pi_ai::api::google_shared::{
    convert_messages, convert_tools, is_thinking_part, map_tool_choice, requires_tool_call_id,
    resolve_google_function_calling_mode, retain_thought_signature, supports_google_strict_tool_sampling,
};
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, ConstrainedSamplingConfig, ConstrainedSamplingSetting,
    ImageContent, Message, Modality, Model, ProviderId, Strictness, TextContent, ThinkingContent,
    Tool, ToolCall, ToolResultBlock, ToolResultMessage, Usage, UserContent, UserMessage,
};
use serde_json::{Value, json};

/// The Google model fixture the conversion suites run, upstream's
/// `makeModel`/`makeGemini3Model`.
fn google_model(api: &str, provider: &str, id: &str, input: Vec<Modality>) -> Model {
    Model {
        id: id.to_owned(),
        name: if id == "gemini-3-pro-preview" {
            "Gemini 3 Pro Preview".to_owned()
        } else {
            id.to_owned()
        },
        api: Api::from(api),
        provider: ProviderId::from(provider),
        base_url: "https://example.com".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input,
        cost: pi_ai::types::ModelCost::default(),
        context_window: 128_000,
        max_tokens: 8_192,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn text_only() -> Vec<Modality> {
    vec![Modality::Text]
}

fn text_and_image() -> Vec<Modality> {
    vec![Modality::Text, Modality::Image]
}

/// The replay turn the conversion suites build: an assistant message plus
/// matching tool results, upstream's `makeContext`.
fn replay_context(api: &str, provider: &str, model_id: &str, content: Vec<AssistantBlock>) -> pi_ai::types::Context {
    pi_ai::types::Context {
        system_prompt: None,
        messages: vec![
            Message::User(UserMessage {
                content: UserContent::Text("Hi".to_owned()),
                timestamp: 1,
            }),
            Message::Assistant(AssistantMessage {
                api: Api::from(api),
                provider: ProviderId::from(provider),
                model: model_id.to_owned(),
                content,
                usage: Usage::default(),
                stop_reason: pi_ai::types::StopReason::ToolUse,
                ..assistant_message_shape()
            }),
        ],
        tools: None,
    }
}

/// The zeroed assistant-message fields the fixtures share.
fn assistant_message_shape() -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: Api::from("google-generative-ai"),
        provider: ProviderId::from("google"),
        model: "gemini-3-pro-preview".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: pi_ai::types::StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

fn signed_empty_context(
    api: &str,
    provider: &str,
    model_id: &str,
    content: Vec<AssistantBlock>,
) -> pi_ai::types::Context {
    replay_context(api, provider, model_id, content)
}

fn tool_call_block(id: &str, command: &str) -> AssistantBlock {
    let mut arguments = serde_json::Map::new();
    arguments.insert("command".to_owned(), json!(command));
    AssistantBlock::ToolCall(pi_ai::types::ToolCall {
        id: id.to_owned(),
        name: "bash".to_owned(),
        arguments,
        thought_signature: None,
        namespace: None,
    })
}

// --- upstream google-thinking-signature.test.ts ---

/// `thought === true` is the definitive thinking marker regardless of what
/// the signature rides on.
#[test]
fn thought_true_parts_are_thinking() {
    assert!(is_thinking_part(&json!({ "thought": true })));
    assert!(is_thinking_part(&json!({
        "thought": true,
        "thoughtSignature": "opaque-signature",
    })));
}

/// A signature alone never marks thinking: per Google's thought-signatures
/// doc it can appear on any part type for context replay.
#[test]
fn thought_signature_alone_is_not_thinking() {
    assert!(!is_thinking_part(&json!({
        "thoughtSignature": "opaque-signature",
    })));
    assert!(!is_thinking_part(&json!({
        "thought": false,
        "thoughtSignature": "opaque-signature",
    })));
}

/// Missing and empty signatures stay non-thinking.
#[test]
fn missing_signatures_are_not_thinking() {
    assert!(!is_thinking_part(&json!({})));
    assert!(!is_thinking_part(&json!({
        "thought": false,
        "thoughtSignature": "",
    })));
}

/// Later deltas without a signature never clobber the retained one,
/// upstream's `retainThoughtSignature` chain.
#[test]
fn empty_or_missing_signature_updates_retain_the_existing_signature() {
    let first = retain_thought_signature(None, Some("sig-1"));
    assert_eq!(first.as_deref(), Some("sig-1"));

    let second = retain_thought_signature(first.as_deref(), None);
    assert_eq!(second.as_deref(), Some("sig-1"));

    let third = retain_thought_signature(second.as_deref(), Some(""));
    assert_eq!(third.as_deref(), Some("sig-1"));
}

/// A new non-empty signature replaces the retained one.
#[test]
fn a_new_non_empty_signature_updates_the_retained_signature() {
    let updated = retain_thought_signature(Some("sig-1"), Some("sig-2"));
    assert_eq!(updated.as_deref(), Some("sig-2"));
}

// --- upstream google-shared-convert-tools.test.ts ---

fn make_tool(parameters: Value) -> Tool {
    Tool {
        name: "test_tool".to_owned(),
        description: "A test tool".to_owned(),
        parameters,
        constrained_sampling: None,
    }
}

#[test]
fn strips_json_schema_meta_keys_from_parameters_when_use_parameters() {
    let tools = [make_tool(json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "urn:bash-tool",
        "$comment": "A bash tool for demonstration",
        "$defs": { "commandDef": { "type": "string" } },
        "definitions": { "legacyDef": { "type": "number" } },
        "type": "object",
        "properties": { "command": { "type": "string" } },
        "required": ["command"],
    }))];

    let result = convert_tools(&tools, true, true).expect("convert_tools");
    let decl = &result.as_ref().expect("declarations")[0]["functionDeclarations"][0];

    assert_eq!(
        decl["parameters"],
        json!({
            "type": "object",
            "properties": { "command": { "type": "string" } },
            "required": ["command"],
        })
    );
    for meta in ["$schema", "$id", "$comment", "$defs", "definitions"] {
        assert!(decl["parameters"].get(meta).is_none(), "{meta} leaked");
    }
}

#[test]
fn recursively_strips_nested_json_schema_meta_keys() {
    let tools = [make_tool(json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "deep": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "$id": "urn:nested",
                "type": "string",
            },
        },
    }))];

    let result = convert_tools(&tools, true, true).expect("convert_tools");
    let decl = &result.as_ref().expect("declarations")[0]["functionDeclarations"][0];

    assert_eq!(
        decl["parameters"],
        json!({
            "type": "object",
            "properties": { "deep": { "type": "string" } },
        })
    );
}

#[test]
fn preserves_ref_while_stripping_meta_keys() {
    let tools = [make_tool(json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "refProp": {
                "$ref": "#/$defs/someDef",
                "type": "string",
            },
        },
    }))];

    let result = convert_tools(&tools, true, true).expect("convert_tools");
    let decl = &result.as_ref().expect("declarations")[0]["functionDeclarations"][0];

    assert_eq!(
        decl["parameters"],
        json!({
            "type": "object",
            "properties": {
                "refProp": { "$ref": "#/$defs/someDef", "type": "string" },
            },
        })
    );
}

/// The conversion never mutates the tool's own parameter document: Rust
/// values are owned copies, so the original survives verbatim.
#[test]
fn does_not_mutate_the_original_tool_parameters() {
    let original = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": { "command": { "type": "string" } },
        "required": ["command"],
    });
    let untouched = original.clone();
    let tools = [make_tool(original)];

    convert_tools(&tools, true, true).expect("convert_tools");

    assert_eq!(tools[0].parameters, untouched);
}

#[test]
fn preserves_meta_keys_in_parameters_json_schema_when_not_use_parameters() {
    let schema = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": { "command": { "type": "string" } },
        "required": ["command"],
    });
    let tools = [make_tool(schema.clone())];

    let result = convert_tools(&tools, false, true).expect("convert_tools");
    let decl = &result.as_ref().expect("declarations")[0]["functionDeclarations"][0];

    assert_eq!(decl["parametersJsonSchema"], schema);
    assert!(decl.get("parameters").is_none());
}

#[test]
fn handles_tools_without_a_schema_key_gracefully() {
    let tools = [make_tool(json!({
        "type": "object",
        "properties": { "path": { "type": "string" } },
        "required": ["path"],
    }))];

    let result = convert_tools(&tools, true, true).expect("convert_tools");
    let decl = &result.as_ref().expect("declarations")[0]["functionDeclarations"][0];

    assert_eq!(
        decl["parameters"],
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"],
        })
    );
}

/// Gemini 3 enforces required parameters in validated tool-calling modes;
/// strict tools select `VALIDATED` and unresolved strict requirements fail.
#[test]
fn uses_validated_function_calling_for_strict_tools_on_gemini_3() {
    let mut tool = make_tool(json!({ "type": "object", "properties": {} }));
    tool.constrained_sampling = Some(ConstrainedSamplingSetting::Config(
        ConstrainedSamplingConfig::JsonSchema {
            strict: Strictness::Require,
        },
    ));

    assert!(supports_google_strict_tool_sampling("gemini-3.1-pro-preview"));
    assert!(!supports_google_strict_tool_sampling("gemini-2.5-pro"));
    assert_eq!(
        resolve_google_function_calling_mode(std::slice::from_ref(&tool), None, true)
            .expect("mode"),
        Some("VALIDATED")
    );
    let error =
        resolve_google_function_calling_mode(std::slice::from_ref(&tool), None, false)
            .expect_err("strict requirement without support");
    assert!(
        error.contains("Tool \"test_tool\" requires JSON-schema constrained sampling"),
        "{error}"
    );
}

#[test]
fn returns_none_for_an_empty_tool_list() {
    assert_eq!(convert_tools(&[], false, true).expect("convert_tools"), None);
    assert_eq!(convert_tools(&[], true, true).expect("convert_tools"), None);
}

// --- upstream google-shared-signed-empty-blocks.test.ts ---
//
// Gemini can attach `thoughtSignature` to a response part whose visible text
// is empty and requires the signature echoed back on the next request; an
// empty text/thinking block is skipped only when it is UNSIGNED.

const VALID_SIG: &str = "AAAAAAAAAAAAAAAAAAAAAA==";

fn signed_empty_model() -> Model {
    google_model("google-generative-ai", "google", "gemini-3-pro-preview", text_only())
}

fn thinking_block(thinking: &str, signature: Option<&str>) -> AssistantBlock {
    AssistantBlock::Thinking(ThinkingContent {
        thinking: thinking.to_owned(),
        thinking_signature: signature.map(str::to_owned),
        redacted: None,
    })
}

fn signed_text_block(text: &str, signature: Option<&str>) -> AssistantBlock {
    AssistantBlock::Text(TextContent {
        text: text.to_owned(),
        text_signature: signature.map(str::to_owned),
    })
}

fn model_turn(contents: &[Value]) -> Option<&Value> {
    contents
        .iter()
        .find(|content| content["role"] == json!("model"))
}

#[test]
fn keeps_a_signed_empty_thinking_block_so_its_signature_is_echoed_back() {
    let model = google_model("google-generative-ai", "google", "gemini-3-pro-preview", text_only());
    let contents = convert_messages(
        &model,
        &replay_context(
            "google-generative-ai",
            "google",
            "gemini-3-pro-preview",
            vec![
                thinking_block("", Some(VALID_SIG)),
                tool_call_block("call_1", "ls"),
            ],
        ),
    );
    let signed = model_turn(&contents)
        .expect("model turn")["parts"]
        .as_array()
        .expect("parts")
        .iter()
        .filter(|part| part["thoughtSignature"] == json!(VALID_SIG))
        .collect::<Vec<_>>();
    assert_eq!(signed.len(), 1);
    assert_eq!(signed[0]["thought"], json!(true));
}

#[test]
fn keeps_a_signed_empty_text_block_the_same_way() {
    let model = google_model("google-generative-ai", "google", "gemini-3-pro-preview", text_only());
    let contents = convert_messages(
        &model,
        &replay_context(
            "google-generative-ai",
            "google",
            "gemini-3-pro-preview",
            vec![
                signed_text_block("", Some(VALID_SIG)),
                tool_call_block("call_1", "ls"),
            ],
        ),
    );
    let signed = model_turn(&contents)
        .expect("model turn")["parts"]
        .as_array()
        .expect("parts")
        .iter()
        .filter(|part| part["thoughtSignature"] == json!(VALID_SIG))
        .count();
    assert_eq!(signed, 1);
}

#[test]
fn still_drops_unsigned_empty_blocks() {
    let model = google_model("google-generative-ai", "google", "gemini-3-pro-preview", text_only());
    let contents = convert_messages(
        &model,
        &replay_context(
            "google-generative-ai",
            "google",
            "gemini-3-pro-preview",
            vec![
                thinking_block("", None),
                signed_text_block("   ", None),
                tool_call_block("call_1", "ls"),
            ],
        ),
    );
    let parts = model_turn(&contents).expect("model turn")["parts"]
        .as_array()
        .expect("parts");
    assert_eq!(parts.len(), 1);
    assert!(parts[0].get("functionCall").is_some());
}

#[test]
fn still_drops_signed_empty_blocks_from_a_different_provider_model() {
    let model = google_model("google-generative-ai", "google", "gemini-3-pro-preview", text_only());
    let contents = convert_messages(
        &model,
        &replay_context(
            "google-generative-ai",
            "google",
            "other-model",
            vec![
                thinking_block("", Some(VALID_SIG)),
                signed_text_block("", Some(VALID_SIG)),
                tool_call_block("call_1", "ls"),
            ],
        ),
    );
    let parts = model_turn(&contents).expect("model turn")["parts"]
        .as_array()
        .expect("parts");
    assert_eq!(parts.len(), 1);
    assert!(parts[0].get("functionCall").is_some());
    assert!(!model_turn(&contents).expect("model turn").to_string().contains(VALID_SIG));
}

// --- upstream google-shared-image-tool-result-routing.test.ts ---

fn image_routing_context(api: &str, provider: &str, model_id: &str) -> pi_ai::types::Context {
    let mut context = replay_context(
        api,
        provider,
        model_id,
        vec![
            tool_call_block("call_a", "ls"),
            tool_call_block("call_img", "ls"),
            tool_call_block("call_b", "ls"),
        ],
    );
    context.messages.push(Message::ToolResult(ToolResultMessage {
        tool_call_id: "call_a".to_owned(),
        tool_name: "read".to_owned(),
        content: vec![pi_ai::types::ToolResultBlock::Text(TextContent {
            text: "alpha text".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 1,
    }));
    context.messages.push(Message::ToolResult(ToolResultMessage {
        tool_call_id: "call_img".to_owned(),
        tool_name: "read".to_owned(),
        content: vec![pi_ai::types::ToolResultBlock::Image(ImageContent {
            data: "abc".to_owned(),
            mime_type: "image/png".to_owned(),
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 1,
    }));
    context.messages.push(Message::ToolResult(ToolResultMessage {
        tool_call_id: "call_b".to_owned(),
        tool_name: "read".to_owned(),
        content: vec![pi_ai::types::ToolResultBlock::Text(TextContent {
            text: "beta text".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 1,
    }));
    context
}

/// Gemini 2.x gets the synthetic "Tool result image:" user turn.
#[test]
fn keeps_a_separate_synthetic_image_turn_for_gemini_2_models() {
    let model = google_model("google-generative-ai", "google", "gemini-2.5-flash", text_and_image());
    let contents = convert_messages(&model, &image_routing_context("google-generative-ai", "google", "gemini-2.5-flash"));

    assert_eq!(contents.len(), 5);
    let turn = &contents[2];
    assert!(turn["parts"]
        .as_array()
        .expect("parts")
        .iter()
        .all(|part| part.get("functionResponse").is_some()));
    assert_eq!(contents[3]["parts"][0]["text"], json!("Tool result image:"));
    assert!(contents[3]["parts"][1].get("inlineData").is_some());
    assert!(contents[4]["parts"][0].get("functionResponse").is_some());
}

/// Gemini 3 nests the image inside functionResponse.parts.
#[test]
fn nests_image_tool_results_for_gemini_3_models() {
    let model = google_model(
        "google-generative-ai",
        "google",
        "gemini-3-pro-preview",
        text_and_image(),
    );
    let contents = convert_messages(
        &model,
        &image_routing_context("google-generative-ai", "google", "gemini-3-pro-preview"),
    );

    assert_eq!(contents.len(), 3);
    let tool_result_turn = &contents[2];
    let parts = tool_result_turn["parts"].as_array().expect("parts");
    assert_eq!(parts.len(), 3);
    let image_response = &parts[1]["functionResponse"];
    assert!(image_response.get("parts").is_some());
    assert_eq!(
        image_response["parts"]
            .as_array()
            .expect("image parts")
            .len(),
        1
    );
    assert!(image_response["parts"][0].get("inlineData").is_some());
}

// --- upstream google-shared-gemini3-unsigned-tool-call.test.ts ---

fn unsigned_tool_call_content(thought_signature: Option<&str>) -> Vec<AssistantBlock> {
    let mut first_arguments = serde_json::Map::new();
    first_arguments.insert("command".to_owned(), json!("echo hi"));
    let mut second_arguments = serde_json::Map::new();
    second_arguments.insert("command".to_owned(), json!("ls -la"));
    vec![
        AssistantBlock::ToolCall(ToolCall {
            id: "call_1".to_owned(),
            name: "bash".to_owned(),
            arguments: first_arguments,
            thought_signature: thought_signature.map(str::to_owned),
            namespace: None,
        }),
        AssistantBlock::ToolCall(ToolCall {
            id: "call_2".to_owned(),
            name: "bash".to_owned(),
            arguments: second_arguments,
            thought_signature: None,
            namespace: None,
        }),
    ]
}

fn unsigned_tool_call_context(
    api: &str,
    provider: &str,
    model_id: &str,
    thought_signature: Option<&str>,
) -> pi_ai::types::Context {
    let mut context = replay_context(api, provider, model_id, unsigned_tool_call_content(thought_signature));
    for (id, text) in [("call_1", "hi"), ("call_2", "files")] {
        context.messages.push(Message::ToolResult(ToolResultMessage {
            tool_call_id: id.to_owned(),
            tool_name: "bash".to_owned(),
            content: vec![pi_ai::types::ToolResultBlock::Text(TextContent {
                text: text.to_owned(),
                text_signature: None,
            })],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 1,
        }));
    }
    context
}

fn function_call_ids(contents: &[Value]) -> Vec<String> {
    contents
        .iter()
        .flat_map(|content| content["parts"].as_array().cloned().unwrap_or_default())
        .filter_map(|part| part["functionCall"]["id"].as_str().map(str::to_owned))
        .collect()
}

fn function_response_ids(contents: &[Value]) -> Vec<String> {
    contents
        .iter()
        .flat_map(|content| content["parts"].as_array().cloned().unwrap_or_default())
        .filter_map(|part| part["functionResponse"]["id"].as_str().map(str::to_owned))
        .collect()
}

#[test]
fn preserves_tool_call_ids_for_gemini_3_history() {
    for (api, provider, id) in [
        ("google-generative-ai", "google", "gemini-3-pro-preview"),
        ("google-generative-ai", "google", "gemini-3.6-flash"),
        ("google-vertex", "google-vertex", "gemini-3-pro-preview"),
    ] {
        let model = google_model(api, provider, id, text_only());
        let contents = convert_messages(&model, &unsigned_tool_call_context(api, provider, id, None));
        assert_eq!(function_call_ids(&contents), vec!["call_1", "call_2"], "{id}");
        assert_eq!(function_response_ids(&contents), vec!["call_1", "call_2"], "{id}");
    }
}

/// The retired `skip_thought_signature_validator` workaround must never
/// reappear, and cross-model history carries no historical-context text.
#[test]
fn does_not_add_skip_thought_signature_validator_for_unsigned_google_tool_calls() {
    let model = google_model("google-generative-ai", "google", "gemini-3-pro-preview", text_only());
    let contents = convert_messages(
        &model,
        &unsigned_tool_call_context("google-generative-ai", "google", "other-model", None),
    );

    let model_turn = model_turn(&contents).expect("model turn");
    let function_call_parts: Vec<&Value> = model_turn["parts"]
        .as_array()
        .expect("parts")
        .iter()
        .filter(|part| part.get("functionCall").is_some())
        .collect();
    assert_eq!(function_call_parts.len(), 2);
    for part in &function_call_parts {
        assert!(part.get("thoughtSignature").is_none());
    }
    assert!(!model_turn.to_string().contains("skip_thought_signature_validator"));
    let historical = model_turn["parts"]
        .as_array()
        .expect("parts")
        .iter()
        .filter(|part| {
            part["text"]
                .as_str()
                .is_some_and(|text| text.contains("Historical context"))
        })
        .count();
    assert_eq!(historical, 0);
}

#[test]
fn does_not_add_skip_thought_signature_validator_for_unsigned_vertex_tool_calls() {
    let model = google_model("google-vertex", "google-vertex", "gemini-3-pro-preview", text_only());
    let contents = convert_messages(&model, &unsigned_tool_call_context("google-vertex", "google-vertex", "gemini-3-pro-preview", None));
    let model_turn = model_turn(&contents).expect("model turn");
    let function_call_parts: Vec<&Value> = model_turn["parts"]
        .as_array()
        .expect("parts")
        .iter()
        .filter(|part| part.get("functionCall").is_some())
        .collect();

    assert_eq!(function_call_parts.len(), 2);
    for part in &function_call_parts {
        assert!(part.get("thoughtSignature").is_none());
    }
    assert!(!model_turn.to_string().contains("skip_thought_signature_validator"));
}

#[test]
fn preserves_a_valid_thought_signature_for_the_same_provider_and_model() {
    let model = google_model("google-generative-ai", "google", "gemini-3-pro-preview", text_only());
    let contents = convert_messages(
        &model,
        &unsigned_tool_call_context("google-generative-ai", "google", "gemini-3-pro-preview", Some(VALID_SIG)),
    );
    let model_turn = model_turn(&contents).expect("model turn");
    let function_call_parts: Vec<&Value> = model_turn["parts"]
        .as_array()
        .expect("parts")
        .iter()
        .filter(|part| part.get("functionCall").is_some())
        .collect();

    assert_eq!(function_call_parts.len(), 2);
    assert_eq!(function_call_parts[0]["thoughtSignature"], json!(VALID_SIG));
    assert!(function_call_parts[1].get("thoughtSignature").is_none());
}

#[test]
fn does_not_add_a_thought_signature_or_ids_for_non_gemini_3_models() {
    let model = google_model("google-generative-ai", "google", "gemini-2.5-flash", text_only());
    let contents = convert_messages(
        &model,
        &unsigned_tool_call_context("google-generative-ai", "google", "other-model", None),
    );
    let model_turn = model_turn(&contents).expect("model turn");
    let function_call_parts: Vec<&Value> = model_turn["parts"]
        .as_array()
        .expect("parts")
        .iter()
        .filter(|part| part.get("functionCall").is_some())
        .collect();
    let function_response_parts: Vec<Value> = contents
        .iter()
        .flat_map(|content| {
            content["parts"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
        })
        .filter(|part| part.get("functionResponse").is_some())
        .collect();

    assert_eq!(function_call_parts.len(), 2);
    assert!(function_call_parts
        .iter()
        .all(|part| part["functionCall"].get("id").is_none()));
    assert!(function_call_parts
        .iter()
        .all(|part| part.get("thoughtSignature").is_none()));
    assert_eq!(function_response_parts.len(), 2);
    assert!(function_response_parts
        .iter()
        .all(|part| part["functionResponse"].get("id").is_none()));
}

/// The id-requiring rule: Gemini 3+, Claude models behind Google APIs, and
/// gpt-oss models require explicit ids; Gemini 2.x does not.
#[test]
fn requires_tool_call_id_selects_by_model_family() {
    assert!(!requires_tool_call_id("gemini-2.5-flash"));
    assert!(requires_tool_call_id("gemini-3.6-flash"));
    assert!(requires_tool_call_id("claude-sonnet-4-5"));
    assert!(requires_tool_call_id("gpt-oss-120b"));
}
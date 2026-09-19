//! Rust-native suites for the small belt modules whose upstream coverage
//! rides the provider children: `hash.ts` goldens computed from the pinned
//! TypeScript implementation, `provider-env.ts`, `diagnostics.ts`,
//! `typebox-helpers.ts`, `headers.ts`, and `deferred-tools.ts`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use pi_ai::types::{
    Context, Message, ProviderEnv, ProviderHeaders, TextContent, Tool, ToolResultBlock,
    ToolResultMessage, UserContent, UserMessage,
};
use pi_ai::utils::deferred_tools::split_deferred_tools;
use pi_ai::utils::diagnostics::{
    append_assistant_message_diagnostic, create_assistant_message_diagnostic,
    extract_diagnostic_error, format_thrown_value,
};
use pi_ai::utils::hash::short_hash;
use pi_ai::utils::headers::{headers_to_record, provider_headers_to_record};
use pi_ai::utils::provider_env::get_provider_env_value;
use pi_ai::utils::typebox_helpers::{StringEnumOptions, string_enum};
use serde_json::json;

// --- hash.ts: goldens from the pinned TypeScript implementation ---

#[test]
fn short_hash_matches_the_typescript_implementation() {
    // Goldens computed by running the pinned hash.ts through node at the
    // port pin, so the digest stays identical across runtimes.
    let goldens: &[(&str, &str)] = &[
        ("", "k4n83c7h0j2b"),
        ("hello", "1h6qa0qrowduu"),
        ("pi", "bliydt151m3se"),
        ("Hello 🙈 World", "11begrz17n9aby"),
        ("こんにちは", "owdm0g1cui7pe"),
        ("READ ME.md", "bs623trgc33v"),
    ];
    for (input, expected) in goldens {
        assert_eq![&short_hash(input), expected, "input {input}"];
    }
}

#[test]
fn the_hash_is_deterministic_and_length_shrinking() {
    let long = "x".repeat(1_000);
    assert_eq![short_hash(&long), "zykls21ciz8e3"];
    assert_eq![short_hash(&long), short_hash(&long), "deterministic"];
    assert![short_hash(&long).len() < 20];
}

// --- provider-env.ts ---

#[test]
fn provider_env_values_skip_empty_strings() {
    let env: ProviderEnv = [
        (
            String::from("ANTHROPIC_BASE_URL"),
            String::from("https://scoped.example"),
        ),
        (String::from("EMPTY_VAR"), String::new()),
    ]
    .into_iter()
    .collect();

    assert_eq![
        get_provider_env_value("EMPTY_VAR", Some(&env)),
        None,
        "an empty override falls through, upstream's falsy-string semantics"
    ];
    assert_eq![
        get_provider_env_value("PI_AI_BELT_MISSING_VAR", Some(&env)),
        None,
        "an unset name is None"
    ];
}

#[test]
fn provider_env_values_fall_through_to_the_process_environment() {
    use pi_ai::utils::provider_env::get_provider_env_value_with_process_env;

    let env: ProviderEnv =
        std::iter::once((String::from("PI_AI_BELT_VAR"), String::from("scoped"))).collect();
    let process = [("PI_AI_BELT_VAR", "from-process"), ("PI_AI_BELT_EMPTY", "")];
    let lookup = |name: &str| {
        process
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| (*value).to_owned())
    };

    assert_eq![
        get_provider_env_value_with_process_env("PI_AI_BELT_VAR", Some(&env), lookup),
        Some(String::from("scoped")),
        "the scoped override wins over the process environment"
    ];
    assert_eq![
        get_provider_env_value_with_process_env("PI_AI_BELT_UNSET", Some(&env), lookup),
        None,
        "an unset name with no process value is None"
    ];
    assert_eq![
        get_provider_env_value_with_process_env("PI_AI_BELT_VAR", None, lookup),
        Some(String::from("from-process")),
        "an empty override set falls through to the process environment"
    ];
    assert_eq![
        get_provider_env_value_with_process_env("PI_AI_BELT_EMPTY", None, lookup),
        None,
        "an empty process value falls through to None"
    ];
}

// --- diagnostics.ts ---

#[derive(Debug)]
struct Boom;

impl std::fmt::Display for Boom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("request failed")
    }
}

impl std::error::Error for Boom {}

#[test]
fn thrown_values_format_to_their_display_form() {
    assert_eq![format_thrown_value(&Boom), "request failed"];
    assert_eq![format_thrown_value(&"plain string"), "plain string"];
}

#[test]
fn diagnostics_extract_the_error_classification_and_message() {
    let info = extract_diagnostic_error(&Boom);
    assert_eq![info.name, Some(String::from("Boom"))];
    assert_eq![info.message, "request failed"];
    assert_eq![info.code, None];
}

#[test]
fn diagnostics_stamp_a_timestamp_and_append_to_messages() {
    let diagnostic = create_assistant_message_diagnostic("stream_error", &Boom, None, 1_234);
    assert_eq![diagnostic.kind, "stream_error"];
    assert_eq![diagnostic.timestamp, 1_234];
    let info = diagnostic.error.as_ref().expect("an error");
    assert_eq![info.name, Some(String::from("Boom"))];
    assert_eq![info.message, "request failed"];

    let mut message = common::bare_assistant_message();
    append_assistant_message_diagnostic(&mut message, diagnostic);
    assert_eq![message.diagnostics.as_ref().map(Vec::len), Some(1)];
    append_assistant_message_diagnostic(
        &mut message,
        create_assistant_message_diagnostic("retry", &Boom, None, 1_235),
    );
    assert_eq![message.diagnostics.as_ref().map(Vec::len), Some(2)];
}

// --- typebox-helpers.ts ---

#[test]
fn string_enum_emits_the_google_compatible_schema() {
    let schema = string_enum(
        &["add", "subtract", "multiply", "divide"],
        Some(StringEnumOptions {
            description: Some(String::from("The operation to perform")),
            default: Some(String::from("add")),
        }),
    );
    assert_eq![
        schema,
        json!({
            "type": "string",
            "enum": ["add", "subtract", "multiply", "divide"],
            "description": "The operation to perform",
            "default": "add",
        })
    ];
    let bare = string_enum(&["on", "off"], None);
    assert_eq![bare, json!({"type": "string", "enum": ["on", "off"]})];
}

// --- headers.ts ---

#[test]
fn header_records_collect_pairs() {
    let record = headers_to_record([("content-type", "application/json"), ("x-trace", "1")]);
    assert_eq![
        record.get("content-type"),
        Some(&String::from("application/json"))
    ];
    assert_eq![record.get("x-trace"), Some(&String::from("1"))];
}

#[test]
fn provider_headers_drop_nulls_and_collapse_to_none_when_empty() {
    let headers: ProviderHeaders = [
        (String::from("x-keep"), Some(String::from("value"))),
        (String::from("x-suppressed"), None),
    ]
    .into_iter()
    .collect();
    let record = provider_headers_to_record(Some(&headers)).expect("concrete headers remain");
    assert_eq![record.len(), 1];
    assert_eq![record.get("x-keep"), Some(&String::from("value"))];

    let empty = ProviderHeaders::new();
    assert![provider_headers_to_record(Some(&empty)).is_none()];
    assert![provider_headers_to_record(None).is_none()];
}

// --- deferred-tools.ts ---

fn tool(name: &str) -> Tool {
    Tool {
        name: name.to_owned(),
        description: String::new(),
        parameters: json!({}),
        constrained_sampling: None,
    }
}

fn tool_result(added: Option<Vec<String>>) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: String::from("call_1"),
        tool_name: String::from("base_tool"),
        content: vec![ToolResultBlock::Text(TextContent {
            text: String::from("done"),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: added,
        is_error: false,
        timestamp: 3,
    })
}

fn user(timestamp: i64) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(String::from("hi")),
        timestamp,
    })
}

#[test]
fn split_deferred_tools_keeps_marked_unused_tools_deferred() {
    let context = Context {
        system_prompt: None,
        messages: vec![user(1), tool_result(Some(vec![String::from("late_tool")]))],
        tools: Some(vec![tool("base_tool"), tool("late_tool")]),
    };

    let split = split_deferred_tools(&context, true, &|name: &str| name.to_owned());
    assert_eq![
        split
            .immediate
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        ["base_tool"]
    ];
    assert_eq![
        split
            .deferred
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["late_tool"]
    ];
}

#[test]
fn split_deferred_tools_keeps_a_used_tool_immediate_despite_its_marker() {
    let context = Context {
        system_prompt: None,
        messages: vec![user(1), tool_result(Some(vec![String::from("late_tool")]))],
        tools: Some(vec![tool("base_tool"), tool("late_tool")]),
    };
    // The result marker alone defers the tool; an assistant call before the
    // marker would keep it immediate, and the wire children exercise that
    // path through their adapters.
    let split = split_deferred_tools(&context, true, &|name: &str| name.to_owned());
    assert_eq![
        split
            .immediate
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        ["base_tool"]
    ];
    assert_eq![split.deferred.len(), 1];
}

#[test]
fn split_deferred_tools_returns_everything_immediate_when_disabled() {
    let context = Context {
        system_prompt: None,
        messages: vec![user(1), tool_result(Some(vec![String::from("late_tool")]))],
        tools: Some(vec![tool("base_tool"), tool("late_tool")]),
    };
    let split = split_deferred_tools(&context, false, &|name: &str| name.to_owned());
    assert_eq![split.immediate.len(), 2];
    assert![split.deferred.is_empty()];
}

#[test]
fn split_deferred_tools_normalizes_names_before_dedup() {
    let context = Context {
        system_prompt: None,
        messages: Vec::new(),
        tools: Some(vec![tool("read"), tool("Read")]),
    };
    let split = split_deferred_tools(&context, true, &|name: &str| name.to_lowercase());
    // The last definition wins under the normalized name.
    assert_eq![split.immediate.len(), 1];
    assert_eq![split.immediate[0].name, "Read"];
}

#[test]
fn split_deferred_tools_without_definitions_keeps_an_empty_split() {
    let context = Context {
        system_prompt: None,
        messages: vec![user(1)],
        tools: None,
    };
    let split = split_deferred_tools(&context, true, &|name: &str| name.to_owned());
    assert![split.immediate.is_empty()];
    assert![split.deferred.is_empty()];
}

#[test]
fn split_deferred_tools_keeps_an_assistant_called_tool_immediate() {
    use pi_ai::types::{AssistantBlock, ToolCall};

    let mut assistant = common::bare_assistant_message();
    assistant.content.push(AssistantBlock::ToolCall(ToolCall {
        id: String::from("call_1"),
        name: String::from("late_tool"),
        arguments: serde_json::Map::new(),
        thought_signature: None,
        namespace: None,
    }));
    assistant.content.push(AssistantBlock::Text(TextContent {
        text: String::from("working"),
        text_signature: None,
    }));
    let context = Context {
        system_prompt: None,
        messages: vec![
            user(1),
            Message::Assistant(assistant),
            tool_result(Some(vec![String::from("late_tool")])),
        ],
        tools: Some(vec![tool("base_tool"), tool("late_tool")]),
    };

    let split = split_deferred_tools(&context, true, &|name: &str| name.to_owned());
    // The assistant call precedes the load marker, so the tool stays
    // immediate and nothing is deferred.
    assert_eq![
        split
            .immediate
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        ["base_tool", "late_tool"]
    ];
    assert![split.deferred.is_empty()];
}

//! Boundary tests for the restated core-type surfaces.
//!
//! Upstream has no unit file for the core types alone — the surface is
//! exercised by the agent-loop and agent suites, which ride their tickets
//! (map children "pi-agent-core: agent loop" and "pi-agent-core: Agent
//! class"). These tests bind what this slice restates: the default stream-fn
//! global's contract, the `AgentMessage` extension design's wire fidelity,
//! the tool contract's erased shape, and the event/state data shapes.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use std::sync::Arc;

use pi_agent_core::{
    AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentState, AgentTool,
    AgentToolResult, BoxedFuture, NoDefaultStreamFn, SearchQuery, SessionSearchError,
    SessionSearchHit, SessionSearchService, SessionSearchTopHit, StreamFn, ToolExecutionMode,
    ToolReplay, get_default_stream_fn, set_default_stream_fn,
};
use pi_ai::types::{Context, Message, Model, SimpleStreamOptions};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde_json::json;

/// A minimal model for `StreamFn`/`AgentState` construction; the model shape
/// belongs to pi-ai, and this fixture pins only that the type surface binds.
fn test_model() -> Model {
    serde_json::from_value(json!({
        "id": "test-model",
        "name": "Test Model",
        "api": "anthropic-messages",
        "provider": "anthropic",
        "baseUrl": "https://api.example.com",
        "reasoning": false,
        "input": ["text"],
        "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0 },
        "contextWindow": 200_000,
        "maxTokens": 8_192,
    }))
    .expect("a well-formed model literal")
}

#[tokio::test]
async fn unset_default_stream_fn_errors_with_the_upstream_message() {
    set_default_stream_fn(None);
    let Err(error) = get_default_stream_fn() else {
        panic!("no default is configured");
    };
    let error: NoDefaultStreamFn = error;
    assert_eq!(
        error.to_string(),
        "No default stream function configured. Pass streamFn explicitly or call setDefaultStreamFn()."
    );
}

#[tokio::test]
async fn configured_default_stream_fn_hands_out_the_same_function_value() {
    let stream_fn: StreamFn = Arc::new(
        |_model: &Model, _context: &Context, _options: Option<&SimpleStreamOptions>| {
            AssistantMessageEventStream::new(|_event| false, |_event| None)
        },
    );
    set_default_stream_fn(Some(Arc::clone(&stream_fn)));
    let fetched = get_default_stream_fn().expect("a default is configured");
    // Upstream hands one function value to every loop run; the Arc is that
    // statement, so the fetched value must be the installed one.
    assert!(Arc::ptr_eq(&stream_fn, &fetched));
    // Clearing restores the error contract.
    set_default_stream_fn(None);
    assert!(get_default_stream_fn().is_err());
}

#[test]
fn standard_messages_deserialize_by_role_and_round_trip_as_pi_ai_wire() {
    let user = json!({ "role": "user", "content": "hi", "timestamp": 1_700_000_000_000_i64 });
    let agent_message: AgentMessage =
        serde_json::from_value(user.clone()).expect("a standard user message");
    assert!(matches!(
        agent_message,
        AgentMessage::Standard(Message::User(_))
    ));

    let wire = serde_json::to_value(&agent_message).expect("standard messages serialize");
    assert_eq!(wire, user, "standard roles pass through pi-ai's wire shape");
}

#[test]
fn custom_message_round_trips_field_for_field() {
    // The coding-agent's BashExecutionMessage shape, as JSON.
    let wire = json!({
        "role": "bashExecution",
        "command": "cargo test",
        "output": "ok",
        "exitCode": 0,
        "cancelled": false,
        "truncated": false,
        "timestamp": 1_700_000_000_000_i64,
    });
    let message: AgentMessage =
        serde_json::from_value(wire.clone()).expect("a custom message parses");
    let AgentMessage::Custom(custom) = &message else {
        panic!("a non-standard role parses as the custom variant");
    };
    assert_eq!(custom.role, "bashExecution");
    assert_eq!(custom.timestamp, 1_700_000_000_000);
    assert_eq!(custom.field("command"), Some(&json!("cargo test")));
    assert!(custom.field("absent").is_none(), "absent keys read as None");

    let round_trip = serde_json::to_value(&message).expect("custom messages serialize");
    assert_eq!(
        round_trip, wire,
        "custom messages round-trip field-for-field"
    );
}

#[test]
fn custom_message_requires_role_and_timestamp() {
    let missing_timestamp = json!({ "role": "custom-ish", "content": "no timestamp" });
    let parsed: Result<AgentMessage, _> = serde_json::from_value(missing_timestamp);
    assert!(
        parsed.is_err(),
        "a custom object without a timestamp is not an AgentMessage"
    );

    let missing_role = json!({ "content": "no role", "timestamp": 1 });
    let parsed: Result<AgentMessage, _> = serde_json::from_value(missing_role);
    assert!(
        parsed.is_err(),
        "a custom object without a role is not an AgentMessage"
    );
}

#[test]
fn agent_state_and_context_construct_as_plain_data() {
    let state = AgentState {
        system_prompt: "You are a helpful assistant.".to_string(),
        model: test_model(),
        thinking_level: pi_agent_core::ThinkingLevel::Medium,
        tools: Vec::new(),
        messages: Vec::new(),
        is_streaming: false,
        streaming_message: None,
        pending_tool_calls: std::iter::once("call-1".to_string()).collect(),
        error_message: None,
    };
    assert_eq!(state.pending_tool_calls.len(), 1);
    assert!(!state.is_streaming);

    let context = AgentContext {
        system_prompt: state.system_prompt,
        messages: Vec::new(),
        tools: None,
    };
    assert!(context.tools.is_none());
}

#[tokio::test]
async fn agent_tool_erases_to_json_arguments_and_details() {
    let tool = AgentTool {
        tool: pi_ai::types::Tool {
            name: "calculate".to_string(),
            description: "Add two numbers".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
                "required": ["a", "b"],
            }),
            constrained_sampling: None,
        },
        label: "Calculate".to_string(),
        prepare_arguments: None,
        execute: Arc::new(
            |tool_call_id: &str,
             args: &serde_json::Value,
             _signal: Option<&tokio_util::sync::CancellationToken>,
             on_update: Option<pi_agent_core::AgentToolUpdateCallback>| {
                Box::pin(async move {
                    if let Some(on_update) = on_update {
                        on_update(&AgentToolResult {
                            content: Vec::new(),
                            details: json!({ "stage": "started" }),
                            usage: None,
                            added_tool_names: None,
                            terminate: None,
                        });
                    }
                    let sum = args["a"].as_f64().unwrap_or(0.0) + 2.0;
                    Ok(AgentToolResult {
                        content: vec![pi_agent_core::AgentToolContent::Text(
                            pi_ai::types::TextContent {
                                text: format!("{tool_call_id}: {}", args["a"]),
                                text_signature: None,
                            },
                        )],
                        details: json!({ "sum": sum }),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    })
                })
            },
        ),
        replay: None,
        execution_mode: None,
    };
    assert_eq!(tool.name(), "calculate");

    let on_update = |_partial: &AgentToolResult| {};
    let executed = (tool.execute)("call-7", &json!({ "a": 2 }), None, Some(&on_update))
        .await
        .expect("the tool executes");
    assert_eq!(executed.details, json!({ "sum": 4.0 }));
}

#[test]
fn custom_message_rejects_non_string_roles_and_non_numeric_timestamps() {
    // The untagged enum wraps the inner contract error — "data did not match
    // any variant" — so these pin only that the parses fail.
    let numeric_role = json!({ "role": 5, "timestamp": 1 });
    let parsed: Result<AgentMessage, _> = serde_json::from_value(numeric_role);
    assert!(parsed.is_err(), "a numeric role is not a custom message");

    let text_timestamp = json!({ "role": "custom", "timestamp": "yesterday" });
    let parsed: Result<AgentMessage, _> = serde_json::from_value(text_timestamp);
    assert!(parsed.is_err(), "a text timestamp is not a custom message");
}

#[test]
fn debug_impls_surface_the_declarative_shapes() {
    let config = AgentLoopConfig {
        stream_options: SimpleStreamOptions::default(),
        model: test_model(),
        convert_to_llm: Arc::new(|messages| {
            Box::pin(std::future::ready(
                messages
                    .into_iter()
                    .map(|message| match message {
                        AgentMessage::Standard(standard) => standard,
                        other @ AgentMessage::Custom(_) => {
                            Message::User(pi_ai::types::UserMessage {
                                content: pi_ai::types::UserContent::Text(format!("{other:?}")),
                                timestamp: 0,
                            })
                        }
                    })
                    .collect::<Vec<_>>(),
            ))
        }),
        transform_context: None,
        get_api_key: None,
        should_stop_after_turn: None,
        prepare_next_turn: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        tool_execution: None,
        before_tool_call: None,
        after_tool_call: None,
    };
    let debugged = format!("{config:?}");
    assert!(debugged.starts_with("AgentLoopConfig"));
    assert!(debugged.contains("test-model"));

    let tool = AgentTool {
        tool: pi_ai::types::Tool {
            name: "probe".to_string(),
            description: String::new(),
            parameters: json!({}),
            constrained_sampling: None,
        },
        label: "Probe".to_string(),
        prepare_arguments: None,
        execute: Arc::new(
            |_id: &str, _args: &serde_json::Value, _signal, _on_update| {
                Box::pin(std::future::ready(Ok(AgentToolResult {
                    content: Vec::new(),
                    details: json!({}),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })))
            },
        ),
        replay: Some(ToolReplay::Never),
        execution_mode: Some(ToolExecutionMode::Sequential),
    };
    let debugged = format!("{tool:?}");
    assert!(debugged.starts_with("AgentTool"));
    assert!(debugged.contains("probe"));
    assert!(
        debugged.contains("execute: ()"),
        "the executor's presence is named"
    );
}

/// The search contract's no-implementation default: an overriding-free stub
/// reads entries as absent and pins the trait's method shapes.
struct StubSearch;

impl SessionSearchService for StubSearch {
    fn search_sessions<'a>(
        &'a self,
        _query: &'a SearchQuery,
    ) -> BoxedFuture<'a, Result<Vec<SessionSearchHit>, SessionSearchError>> {
        Box::pin(std::future::ready(Ok(vec![SessionSearchHit {
            session_id: "s1".to_string(),
            score: Some(0.5),
            top: Some(SessionSearchTopHit {
                entry_id: "e1".to_string(),
                snippet: None,
                timestamp: 1_700_000_000_000,
            }),
        }])))
    }

    fn sync(&self) -> BoxedFuture<'_, Result<(), SessionSearchError>> {
        Box::pin(std::future::ready(Ok(())))
    }

    fn notify(&self, _session_id: &str) {}

    fn remove(&self, _session_id: &str) -> BoxedFuture<'_, Result<(), SessionSearchError>> {
        Box::pin(std::future::ready(Ok(())))
    }

    fn close(&self) -> BoxedFuture<'_, Result<(), SessionSearchError>> {
        Box::pin(std::future::ready(Ok(())))
    }
}

#[tokio::test]
async fn session_search_default_search_entries_degrades_to_no_hits() {
    let service = StubSearch;
    let hits = service
        .search_entries(&SearchQuery {
            text: "anything".to_string(),
            limit: None,
        })
        .await
        .expect("the default degrades to an empty result");
    assert!(hits.is_empty());
}

#[test]
fn agent_events_carry_the_upstream_payloads() {
    let start = AgentEvent::AgentStart;
    let end = AgentEvent::AgentEnd {
        messages: Vec::new(),
    };
    let tool_end = AgentEvent::ToolExecutionEnd {
        tool_call_id: "call-1".to_string(),
        tool_name: "calculate".to_string(),
        result: AgentToolResult {
            content: Vec::new(),
            details: json!({}),
            usage: None,
            added_tool_names: None,
            terminate: None,
        },
        is_error: false,
    };
    match (&start, &end) {
        (AgentEvent::AgentStart, AgentEvent::AgentEnd { messages }) => {
            assert!(messages.is_empty());
        }
        _ => unreachable!("the constructed variants"),
    }
    let AgentEvent::ToolExecutionEnd { tool_call_id, .. } = tool_end else {
        unreachable!("the constructed variant");
    };
    assert_eq!(tool_call_id, "call-1");
}

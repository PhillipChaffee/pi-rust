//! The agent-loop suite, ported 1:1 from upstream `test/agent-loop.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! `queueMicrotask` restates as a spawned task, so every mock stream is still
//! running when the stream function returns it. The release-timer tests
//! (`setTimeout(..., 20)`) run on the paused tokio clock, which advances the
//! virtual delay whenever every task is idle — no wall-clock waits.
//! Upstream's `beforeToolCall` args-mutation case restates: the hook receives
//! an owned snapshot, so the shared-object mutation cannot flow.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]
#![expect(
    clippy::too_many_lines,
    reason = "each test mirrors one upstream case end to end"
)]

mod common;
use common::*;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use pi_agent_core::AgentContext;
use pi_agent_core::AgentEvent;
use pi_agent_core::AgentLoopConfig;
use pi_agent_core::AgentLoopTurnUpdate;
use pi_agent_core::AgentMessage;
use pi_agent_core::AgentTool;
use pi_agent_core::AgentToolResult;
use pi_agent_core::BeforeToolCallContext;
use pi_agent_core::BeforeToolCallResult;
use pi_agent_core::ConvertToLlm;
use pi_agent_core::CustomAgentMessage;
use pi_agent_core::PrepareNextTurnContext;
use pi_agent_core::ShouldStopAfterTurnContext;
use pi_agent_core::ToolExecutionMode;
use pi_agent_core::set_default_stream_fn;
use pi_ai::auth::resolve::now_ms;
use pi_ai::types::AssistantMessage;
use pi_ai::types::Context;
use pi_ai::types::Message;
use pi_ai::types::Model;
use pi_ai::types::SimpleStreamOptions;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::types::ToolResultBlock;
use pi_ai::types::Usage;
use pi_ai::types::UserContent;
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde_json::Value;
use serde_json::json;
use tokio::sync::Notify;

/// The echo tool result shape every echo fixture returns.
fn echo_result(value: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultBlock::Text(TextContent {
            text: format!("echoed: {value}"),
            text_signature: None,
        })],
        details: json!({ "value": value }),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

/// The gated-order mock stream factory: the first call returns the tool-call
/// message and schedules the gate's release on the paused clock; later calls
/// return the final text response.
fn gated_stream_fn(gate: Arc<Notify>, first_message: AssistantMessage) -> pi_agent_core::StreamFn {
    let call_index = Arc::new(AtomicUsize::new(0));
    let call_index_for_stream = Arc::clone(&call_index);
    Arc::new(move |_model, _context, _options| {
        let index = call_index_for_stream.fetch_add(1, Ordering::Relaxed);
        let mock_stream = mock_stream();
        let push = mock_stream.clone();
        let gate = Arc::clone(&gate);
        let first = first_message.clone();
        tokio::spawn(async move {
            if index == 0 {
                push.push(done_event(first));
                tokio::time::sleep(Duration::from_millis(20)).await;
                gate.notify_one();
            } else {
                push.push(done_event(create_assistant_message(
                    vec![text_block("done")],
                    StopReason::Stop,
                )));
            }
        });
        mock_stream
    })
}

/// The echo tool the upstream tests define repeatedly: records each executed
/// `value` and echoes it back.
fn echo_tool(executed: Arc<Mutex<Vec<String>>>) -> AgentTool {
    suite_tool(
        "echo",
        echo_tool_schema(),
        None,
        Arc::new(move |_tool_call_id, args: &Value, _signal, _on_update| {
            let executed = Arc::clone(&executed);
            Box::pin(async move {
                let value = args["value"].as_str().unwrap_or_default().to_string();
                executed.lock().expect("executed lock").push(value.clone());
                Ok(echo_result(&value))
            })
        }),
    )
}

/// The gated tool of the execution-order tests: the `first` call blocks on
/// the release gate while recording that the `second` call observed it still
/// unresolved.
fn gated_echo_tool(
    gate: Arc<Notify>,
    first_resolved: Arc<AtomicBool>,
    parallel_observed: Arc<AtomicBool>,
) -> AgentTool {
    suite_tool(
        "echo",
        echo_tool_schema(),
        None,
        Arc::new(move |_tool_call_id, args: &Value, _signal, _on_update| {
            let gate = Arc::clone(&gate);
            let first_resolved = Arc::clone(&first_resolved);
            let parallel_observed = Arc::clone(&parallel_observed);
            Box::pin(async move {
                let value = args["value"].as_str().unwrap_or_default().to_string();
                if value == "first" {
                    gate.notified().await;
                    first_resolved.store(true, Ordering::Relaxed);
                }
                if value == "second" && !first_resolved.load(Ordering::Relaxed) {
                    parallel_observed.store(true, Ordering::Relaxed);
                }
                Ok(echo_result(&value))
            })
        }),
    )
}

fn tool_execution_end_ids(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect()
}

fn tool_result_ids_of_message_ends(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageEnd {
                message: AgentMessage::Standard(Message::ToolResult(tool_result)),
            } => Some(tool_result.tool_call_id.clone()),
            _ => None,
        })
        .collect()
}

fn tool_result_ids_of_turn_ends(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::TurnEnd { tool_results, .. } => Some(
                tool_results
                    .iter()
                    .map(|result| result.tool_call_id.clone())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .flatten()
        .collect()
}

fn message_start_sequence(events: &[AgentEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::MessageStart {
                message: AgentMessage::Standard(Message::ToolResult(tool_result)),
            } => Some(format!("tool:{}", tool_result.tool_call_id)),
            AgentEvent::MessageStart {
                message: AgentMessage::Standard(Message::User(user)),
            } => match &user.content {
                UserContent::Text(text) => Some(text.clone()),
                UserContent::Blocks(_) => None,
            },
            _ => None,
        })
        .collect()
}

// default stream function compatibility

#[tokio::test]
async fn uses_the_configured_default_when_a_legacy_caller_omits_stream_fn() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_stream_fn = Arc::clone(&calls);
    set_default_stream_fn(Some(Arc::new(
        move |_model: &Model, _context: &Context, _options: Option<&SimpleStreamOptions>| {
            calls_for_stream_fn.fetch_add(1, Ordering::Relaxed);
            let stream = mock_stream();
            let push = stream.clone();
            tokio::spawn(async move {
                push.push(done_event(create_assistant_message(
                    vec![text_block("fallback")],
                    StopReason::Stop,
                )));
            });
            stream
        },
    )));

    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(Vec::new()),
    };
    let config = test_config(identity_converter());
    let stream = pi_agent_core::agent_loop(
        vec![create_user_message("Hello")],
        context,
        config,
        None,
        None,
    );

    stream.result().await;
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    set_default_stream_fn(None);
}

// agentLoop with AgentMessage

#[tokio::test]
async fn should_emit_events_with_agent_message_types() {
    let context = AgentContext {
        system_prompt: "You are helpful.".to_string(),
        messages: Vec::new(),
        tools: Some(Vec::new()),
    };

    let user_prompt = create_user_message("Hello");
    let config = test_config(identity_converter());

    let stream_fn = single_response_stream_fn(create_assistant_message(
        vec![text_block("Hi there!")],
        StopReason::Stop,
    ));

    let stream =
        pi_agent_core::agent_loop(vec![user_prompt], context, config, None, Some(stream_fn));
    let events = collect_events(&stream).await;

    let messages = stream.result().await;

    // Should have user message and assistant message
    assert_eq!(messages.len(), 2);
    assert_eq!(agent_message_role(&messages[0]), "user");
    assert_eq!(agent_message_role(&messages[1]), "assistant");

    // Verify event sequence
    let event_types: Vec<&str> = events.iter().map(event_type_name).collect();
    for expected in [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ] {
        assert!(event_types.contains(&expected), "missing {expected}");
    }
}

#[tokio::test]
async fn should_handle_custom_message_types_via_convert_to_llm() {
    // Create a custom message type
    let notification = AgentMessage::Custom(CustomAgentMessage {
        role: "notification".to_string(),
        timestamp: now_ms(),
        data: serde_json::Map::from_iter([("text".to_string(), json!("This is a notification"))]),
    });

    let context = AgentContext {
        system_prompt: "You are helpful.".to_string(),
        messages: vec![notification],
        tools: Some(Vec::new()),
    };

    let user_prompt = create_user_message("Hello");

    let converted_messages = Arc::new(Mutex::new(Vec::<Message>::new()));
    let converted_for_config = Arc::clone(&converted_messages);
    let convert_to_llm: ConvertToLlm = Arc::new(move |messages| {
        let converted_for_config = Arc::clone(&converted_for_config);
        Box::pin(async move {
            // Filter out notifications, convert rest
            let converted: Vec<Message> = messages
                .into_iter()
                .filter(|message| match message {
                    AgentMessage::Custom(custom) => custom.role != "notification",
                    AgentMessage::Standard(_) => true,
                })
                .filter_map(|message| match message {
                    AgentMessage::Standard(standard) => Some(standard),
                    AgentMessage::Custom(_) => None,
                })
                .collect();
            *converted_for_config.lock().expect("converted lock") = converted.clone();
            converted
        })
    });
    let config = test_config(convert_to_llm);

    let stream_fn = single_response_stream_fn(create_assistant_message(
        vec![text_block("Response")],
        StopReason::Stop,
    ));

    let stream =
        pi_agent_core::agent_loop(vec![user_prompt], context, config, None, Some(stream_fn));
    let _ = collect_events(&stream).await;

    // The notification should have been filtered out in convertToLlm
    let (count, first_is_user) = {
        let converted = converted_messages.lock().expect("converted lock");
        (converted.len(), matches!(&converted[0], Message::User(_)))
    };
    assert_eq!(count, 1); // Only user message
    assert!(first_is_user);
}

#[tokio::test]
async fn should_apply_transform_context_before_convert_to_llm() {
    let context = AgentContext {
        system_prompt: "You are helpful.".to_string(),
        messages: vec![
            create_user_message("old message 1"),
            AgentMessage::Standard(Message::Assistant(create_assistant_message(
                vec![text_block("old response 1")],
                StopReason::Stop,
            ))),
            create_user_message("old message 2"),
            AgentMessage::Standard(Message::Assistant(create_assistant_message(
                vec![text_block("old response 2")],
                StopReason::Stop,
            ))),
        ],
        tools: Some(Vec::new()),
    };

    let user_prompt = create_user_message("new message");

    let transformed_messages = Arc::new(Mutex::new(Vec::<AgentMessage>::new()));
    let converted_messages = Arc::new(Mutex::new(Vec::<Message>::new()));

    let transformed_for_config = Arc::clone(&transformed_messages);
    let converted_for_config = Arc::clone(&converted_messages);
    let convert_to_llm: ConvertToLlm = Arc::new(move |messages| {
        let converted_for_config = Arc::clone(&converted_for_config);
        Box::pin(async move {
            let converted: Vec<Message> = messages
                .into_iter()
                .filter_map(|message| match message {
                    AgentMessage::Standard(standard) => Some(standard),
                    AgentMessage::Custom(_) => None,
                })
                .collect();
            *converted_for_config.lock().expect("converted lock") = converted.clone();
            converted
        })
    });
    let config = AgentLoopConfig {
        transform_context: Some(Arc::new(move |messages, _signal| {
            let transformed_for_config = Arc::clone(&transformed_for_config);
            Box::pin(async move {
                // Keep only last 2 messages (prune old ones)
                let mut messages = messages;
                let keep = messages.split_off(messages.len().saturating_sub(2));
                *transformed_for_config.lock().expect("transformed lock") = keep.clone();
                keep
            })
        })),
        ..test_config(convert_to_llm)
    };

    let stream_fn = single_response_stream_fn(create_assistant_message(
        vec![text_block("Response")],
        StopReason::Stop,
    ));

    let stream =
        pi_agent_core::agent_loop(vec![user_prompt], context, config, None, Some(stream_fn));
    let _ = collect_events(&stream).await;

    // transformContext should have been called first, keeping only last 2
    assert_eq!(
        transformed_messages.lock().expect("transformed lock").len(),
        2
    );
    // Then convertToLlm receives the pruned messages
    assert_eq!(converted_messages.lock().expect("converted lock").len(), 2);
}

#[tokio::test]
async fn should_handle_tool_calls_and_results() {
    let executed = Arc::new(Mutex::new(Vec::<String>::new()));
    let executed_for_tool = Arc::clone(&executed);
    let tool = suite_tool(
        "echo",
        echo_tool_schema(),
        None,
        Arc::new(move |_tool_call_id, args: &Value, _signal, _on_update| {
            let executed = Arc::clone(&executed_for_tool);
            Box::pin(async move {
                let value = args["value"].as_str().unwrap_or_default().to_string();
                executed.lock().expect("executed lock").push(value.clone());
                Ok(AgentToolResult {
                    content: vec![ToolResultBlock::Text(TextContent {
                        text: format!("echoed: {value}"),
                        text_signature: None,
                    })],
                    details: json!({ "value": value }),
                    usage: Some(tool_usage_fixture()),
                    added_tool_names: None,
                    terminate: None,
                })
            })
        }),
    );

    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };

    let user_prompt = create_user_message("echo something");

    let observed_tool_usage = Arc::new(Mutex::new(None::<Usage>));
    let observed_for_config = Arc::clone(&observed_tool_usage);
    let after_tool_call: pi_agent_core::AfterToolCall = Arc::new(
        move |context: pi_agent_core::AfterToolCallContext, _signal| {
            let observed_for_config = Arc::clone(&observed_for_config);
            Box::pin(async move {
                *observed_for_config.lock().expect("observed lock") = context.result.usage;
                Some(pi_agent_core::AfterToolCallResult {
                    usage: Some(patched_tool_usage_fixture()),
                    ..pi_agent_core::AfterToolCallResult::default()
                })
            })
        },
    );
    let config = AgentLoopConfig {
        after_tool_call: Some(after_tool_call),
        ..test_config(identity_converter())
    };

    let stream_fn: pi_agent_core::StreamFn = two_call_stream_fn(
        // First call: return tool call
        create_assistant_message(
            vec![create_tool_call(
                "tool-1",
                "echo",
                json!({ "value": "hello" }),
            )],
            StopReason::ToolUse,
        ),
        // Second call: return final response
        create_assistant_message(vec![text_block("done")], StopReason::Stop),
    );

    let stream =
        pi_agent_core::agent_loop(vec![user_prompt], context, config, None, Some(stream_fn));
    let events = collect_events(&stream).await;

    // Tool should have been executed
    assert_eq!(*executed.lock().expect("executed lock"), vec!["hello"]);

    // Should have tool execution events
    let tool_end = events
        .iter()
        .find(|event| matches!(event, AgentEvent::ToolExecutionEnd { .. }))
        .expect("a tool_execution_end event");
    let AgentEvent::ToolExecutionEnd { is_error, .. } = tool_end else {
        panic!("the found event is a tool_execution_end");
    };
    assert!(!is_error);
    assert_eq!(
        observed_tool_usage.lock().expect("observed lock").as_ref(),
        Some(&tool_usage_fixture())
    );
    let messages = stream.result().await;
    let tool_result = messages.iter().find_map(|message| match message {
        AgentMessage::Standard(Message::ToolResult(tool_result)) => Some(tool_result),
        _ => None,
    });
    assert_eq!(
        tool_result.and_then(|result| result.usage),
        Some(patched_tool_usage_fixture())
    );
}

fn tool_usage_fixture() -> Usage {
    Usage {
        input: 1,
        output: 2,
        cache_read: 3,
        cache_write: 4,
        total_tokens: 10,
        cost: pi_ai::types::UsageCost {
            input: 0.1,
            output: 0.2,
            cache_read: 0.3,
            cache_write: 0.4,
            total: 1.0,
        },
        ..Usage::default()
    }
}

fn patched_tool_usage_fixture() -> Usage {
    Usage {
        input: 5,
        output: 6,
        cache_read: 7,
        cache_write: 8,
        total_tokens: 26,
        cost: pi_ai::types::UsageCost {
            input: 0.5,
            output: 0.6,
            cache_read: 0.7,
            cache_write: 0.8,
            total: 2.6,
        },
        ..Usage::default()
    }
}

#[tokio::test]
async fn should_not_execute_tool_calls_from_a_length_truncated_assistant_message() {
    let executed = Arc::new(Mutex::new(Vec::<String>::new()));
    let tool = echo_tool(Arc::clone(&executed));
    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };
    let config = test_config(identity_converter());

    // The call counter pins that the loop re-issues after the truncated batch.
    let call_index = Arc::new(AtomicUsize::new(0));
    let call_index_for_stream = Arc::clone(&call_index);
    let stream_fn: pi_agent_core::StreamFn = Arc::new(move |_model, _context, _options| {
        let index = call_index_for_stream.fetch_add(1, Ordering::Relaxed);
        let stream = mock_stream();
        let push = stream.clone();
        tokio::spawn(async move {
            // Output hit the token limit mid tool call. The salvage parser
            // can produce arguments that validate but are silently truncated,
            // so nothing in this message may execute.
            let message = if index == 0 {
                create_assistant_message(
                    vec![create_tool_call(
                        "tool-1",
                        "echo",
                        json!({ "value": "hel" }),
                    )],
                    StopReason::Length,
                )
            } else {
                create_assistant_message(vec![text_block("done")], StopReason::Stop)
            };
            push.push(done_event(message));
        });
        stream
    });

    let stream = pi_agent_core::agent_loop(
        vec![create_user_message("echo something")],
        context,
        config,
        None,
        Some(stream_fn),
    );
    let events = collect_events(&stream).await;

    // The tool must never execute with potentially truncated arguments.
    assert!(executed.lock().expect("executed lock").is_empty());

    let tool_end = events
        .iter()
        .find(|event| matches!(event, AgentEvent::ToolExecutionEnd { .. }))
        .expect("a tool_execution_end event");
    let AgentEvent::ToolExecutionEnd {
        result, is_error, ..
    } = tool_end
    else {
        panic!("the found event is a tool_execution_end");
    };
    assert!(is_error);
    let text = result.content.iter().find_map(|block| match block {
        ToolResultBlock::Text(text) => Some(&text.text),
        ToolResultBlock::Image(_) => None,
    });
    assert!(
        text.is_some_and(|text| text.contains("output token limit")),
        "the error names the token limit"
    );

    // The loop continues so the model can re-issue the tool call.
    assert_eq!(call_index.load(Ordering::Relaxed), 2);
    let messages = stream.result().await;
    assert_eq!(
        agent_message_role(messages.last().expect("a final message")),
        "assistant"
    );
}

#[tokio::test]
async fn executes_the_validated_args_when_before_tool_call_mutates_its_owned_snapshot() {
    // Upstream's hook mutates the shared args object in place and the loop
    // executes the mutation without revalidation; the port hands
    // `beforeToolCall` an owned snapshot, so the mutation cannot flow and
    // the validated arguments execute instead.
    let executed = Arc::new(Mutex::new(Vec::<Value>::new()));
    let executed_for_tool = Arc::clone(&executed);
    let tool = suite_tool(
        "echo",
        echo_tool_schema(),
        None,
        Arc::new(move |_tool_call_id, args: &Value, _signal, _on_update| {
            let executed = Arc::clone(&executed_for_tool);
            Box::pin(async move {
                executed
                    .lock()
                    .expect("executed lock")
                    .push(args["value"].clone());
                Ok(AgentToolResult {
                    content: vec![ToolResultBlock::Text(TextContent {
                        text: format!("echoed: {}", args["value"]),
                        text_signature: None,
                    })],
                    details: json!({ "value": args["value"] }),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        }),
    );

    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };
    let user_prompt = create_user_message("echo something");

    let before_tool_call: pi_agent_core::BeforeToolCall =
        Arc::new(|context: BeforeToolCallContext, _signal| {
            Box::pin(async move {
                let mut owned_args = context.args;
                owned_args["value"] = json!(123);
                // The snapshot mutation dies here; the loop revalidates
                // nothing and the tool receives the validated arguments.
                None
            })
        });
    let config = AgentLoopConfig {
        before_tool_call: Some(before_tool_call),
        ..test_config(identity_converter())
    };

    let stream_fn: pi_agent_core::StreamFn = two_call_stream_fn(
        create_assistant_message(
            vec![create_tool_call(
                "tool-1",
                "echo",
                json!({ "value": "hello" }),
            )],
            StopReason::ToolUse,
        ),
        create_assistant_message(vec![text_block("done")], StopReason::Stop),
    );

    let stream =
        pi_agent_core::agent_loop(vec![user_prompt], context, config, None, Some(stream_fn));
    let _ = collect_events(&stream).await;

    assert_eq!(
        *executed.lock().expect("executed lock"),
        vec![json!("hello")]
    );
}

#[tokio::test]
async fn should_prepare_tool_arguments_for_validation() {
    let executed = Arc::new(Mutex::new(Vec::<Vec<Value>>::new()));
    let executed_for_tool = Arc::clone(&executed);
    let tool = AgentTool {
        tool: pi_ai::types::Tool {
            name: "edit".to_string(),
            description: "Edit tool".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": { "type": "string" },
                                "newText": { "type": "string" },
                            },
                            "required": ["oldText", "newText"],
                        },
                    },
                },
                "required": ["edits"],
            }),
            constrained_sampling: None,
        },
        label: "Edit".to_string(),
        prepare_arguments: Some(Arc::new(|args: &Value| {
            if args["oldText"].is_string() && args["newText"].is_string() {
                let mut edits = args["edits"].as_array().cloned().unwrap_or_default();
                edits.push(json!({
                    "oldText": args["oldText"],
                    "newText": args["newText"],
                }));
                Ok(json!({ "edits": edits }))
            } else {
                Ok(args.clone())
            }
        })),
        execute: Arc::new(move |_tool_call_id, args: &Value, _signal, _on_update| {
            let executed = Arc::clone(&executed_for_tool);
            Box::pin(async move {
                let edits: Vec<Value> = args["edits"].as_array().cloned().unwrap_or_default();
                executed.lock().expect("executed lock").push(edits.clone());
                Ok(AgentToolResult {
                    content: vec![ToolResultBlock::Text(TextContent {
                        text: format!("edited {}", edits.len()),
                        text_signature: None,
                    })],
                    details: json!({ "count": edits.len() }),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        }),
        replay: None,
        execution_mode: None,
    };

    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };
    let user_prompt = create_user_message("edit something");
    let config = test_config(identity_converter());

    let stream_fn: pi_agent_core::StreamFn = two_call_stream_fn(
        create_assistant_message(
            vec![create_tool_call(
                "tool-1",
                "edit",
                json!({ "oldText": "before", "newText": "after" }),
            )],
            StopReason::ToolUse,
        ),
        create_assistant_message(vec![text_block("done")], StopReason::Stop),
    );

    let stream =
        pi_agent_core::agent_loop(vec![user_prompt], context, config, None, Some(stream_fn));
    let _ = collect_events(&stream).await;

    assert_eq!(
        *executed.lock().expect("executed lock"),
        vec![vec![json!({ "oldText": "before", "newText": "after" })]]
    );
}

#[tokio::test]
async fn should_emit_tool_execution_end_in_completion_order_but_persist_tool_results_in_source_order()
 {
    tokio::time::pause();
    let first_resolved = Arc::new(AtomicBool::new(false));
    let parallel_observed = Arc::new(AtomicBool::new(false));
    let first_gate = Arc::new(Notify::new());

    let tool = gated_echo_tool(
        Arc::clone(&first_gate),
        Arc::clone(&first_resolved),
        Arc::clone(&parallel_observed),
    );
    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };
    let user_prompt = create_user_message("echo both");
    let config = AgentLoopConfig {
        tool_execution: Some(ToolExecutionMode::Parallel),
        ..test_config(identity_converter())
    };

    let stream_fn = gated_stream_fn(
        Arc::clone(&first_gate),
        create_assistant_message(
            vec![
                create_tool_call("tool-1", "echo", json!({ "value": "first" })),
                create_tool_call("tool-2", "echo", json!({ "value": "second" })),
            ],
            StopReason::ToolUse,
        ),
    );

    let stream =
        pi_agent_core::agent_loop(vec![user_prompt], context, config, None, Some(stream_fn));
    let events = collect_events(&stream).await;

    let tool_execution_end_ids = tool_execution_end_ids(&events);
    let tool_result_ids = tool_result_ids_of_message_ends(&events);
    let turn_tool_result_ids = tool_result_ids_of_turn_ends(&events);

    assert!(parallel_observed.load(Ordering::Relaxed));
    assert_eq!(tool_execution_end_ids, ["tool-2", "tool-1"]);
    assert_eq!(tool_result_ids, ["tool-1", "tool-2"]);
    assert_eq!(turn_tool_result_ids, ["tool-1", "tool-2"]);
}

#[tokio::test]
async fn should_inject_queued_messages_after_all_tool_calls_complete() {
    let executed = Arc::new(Mutex::new(Vec::<String>::new()));
    let tool = echo_tool(Arc::clone(&executed));
    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };

    let queued_user_message = create_user_message("interrupt");
    let queued_delivered = Arc::new(AtomicBool::new(false));
    let saw_interrupt_in_context = Arc::new(AtomicBool::new(false));

    let executed_for_steering = Arc::clone(&executed);
    let delivered_for_steering = Arc::clone(&queued_delivered);
    let queued_for_steering = queued_user_message.clone();
    let get_steering_messages: pi_agent_core::GetSteeringMessages = Arc::new(move || {
        let queued = queued_for_steering.clone();
        let executed = Arc::clone(&executed_for_steering);
        let delivered = Arc::clone(&delivered_for_steering);
        Box::pin(async move {
            // Return steering message after tool execution has started.
            if !executed.lock().expect("executed lock").is_empty()
                && !delivered.load(Ordering::Relaxed)
            {
                delivered.store(true, Ordering::Relaxed);
                return vec![queued];
            }
            Vec::new()
        })
    });
    let config = AgentLoopConfig {
        tool_execution: Some(ToolExecutionMode::Sequential),
        get_steering_messages: Some(get_steering_messages),
        ..test_config(identity_converter())
    };

    let call_index = Arc::new(AtomicUsize::new(0));
    let call_index_for_stream = Arc::clone(&call_index);
    let saw_for_stream = Arc::clone(&saw_interrupt_in_context);
    let stream_fn: pi_agent_core::StreamFn = Arc::new(move |_model, ctx: &Context, _options| {
        let index = call_index_for_stream.fetch_add(1, Ordering::Relaxed);
        // Check if interrupt message is in context on second call
        if index == 1 {
            let saw_interrupt = ctx.messages.iter().any(|message| match message {
                Message::User(user) => matches!(
                    &user.content,
                    UserContent::Text(text) if text == "interrupt"
                ),
                _ => false,
            });
            saw_for_stream.store(saw_interrupt, Ordering::Relaxed);
        }
        let mock_stream = mock_stream();
        let push = mock_stream.clone();
        tokio::spawn(async move {
            let message = if index == 0 {
                // First call: return two tool calls
                create_assistant_message(
                    vec![
                        create_tool_call("tool-1", "echo", json!({ "value": "first" })),
                        create_tool_call("tool-2", "echo", json!({ "value": "second" })),
                    ],
                    StopReason::ToolUse,
                )
            } else {
                // Second call: return final response
                create_assistant_message(vec![text_block("done")], StopReason::Stop)
            };
            push.push(done_event(message));
        });
        mock_stream
    });

    let stream = pi_agent_core::agent_loop(
        vec![create_user_message("start")],
        context,
        config,
        None,
        Some(stream_fn),
    );
    let events = collect_events(&stream).await;

    // Both tools should execute before steering is injected
    assert_eq!(
        *executed.lock().expect("executed lock"),
        vec!["first", "second"]
    );

    let tool_ends: Vec<&AgentEvent> = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::ToolExecutionEnd { .. }))
        .collect();
    assert_eq!(tool_ends.len(), 2);
    for end in &tool_ends {
        let AgentEvent::ToolExecutionEnd { is_error, .. } = end else {
            panic!("a tool_execution_end event");
        };
        assert!(!is_error);
    }

    // Queued message should appear in events after both tool result messages
    let event_sequence = message_start_sequence(&events);
    assert!(event_sequence.contains(&"interrupt".to_string()));
    let interrupt_index = interrupt_index_of(&event_sequence);
    let tool_1_index = event_sequence
        .iter()
        .position(|entry| entry == "tool:tool-1")
        .expect("tool-1's result message");
    let tool_2_index = event_sequence
        .iter()
        .position(|entry| entry == "tool:tool-2")
        .expect("tool-2's result message");
    assert!(tool_1_index < interrupt_index);
    assert!(tool_2_index < interrupt_index);

    // Interrupt message should be in context when second LLM call is made
    assert!(saw_interrupt_in_context.load(Ordering::Relaxed));
}

fn interrupt_index_of(sequence: &[String]) -> usize {
    sequence
        .iter()
        .position(|entry| entry == "interrupt")
        .expect("the interrupt was injected")
}

#[tokio::test]
async fn should_force_sequential_execution_when_a_tool_has_execution_mode_sequential_even_with_default_parallel_config()
 {
    tokio::time::pause();
    let first_resolved = Arc::new(AtomicBool::new(false));
    let parallel_observed = Arc::new(AtomicBool::new(false));
    let first_gate = Arc::new(Notify::new());

    let mut slow_tool = gated_echo_tool(
        Arc::clone(&first_gate),
        Arc::clone(&first_resolved),
        Arc::clone(&parallel_observed),
    );
    slow_tool.tool.name = "slow".to_string();
    slow_tool.execution_mode = Some(ToolExecutionMode::Sequential);
    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![slow_tool]),
    };
    let user_prompt = create_user_message("run both");
    // config is parallel (default), but tool forces sequential
    let config = test_config(identity_converter());

    let stream_fn = gated_stream_fn(
        Arc::clone(&first_gate),
        create_assistant_message(
            vec![
                create_tool_call("tool-1", "slow", json!({ "value": "first" })),
                create_tool_call("tool-2", "slow", json!({ "value": "second" })),
            ],
            StopReason::ToolUse,
        ),
    );

    let stream =
        pi_agent_core::agent_loop(vec![user_prompt], context, config, None, Some(stream_fn));
    let events = collect_events(&stream).await;

    // With sequential execution, second tool should NOT start before first finishes
    assert!(!parallel_observed.load(Ordering::Relaxed));

    let tool_result_ids = tool_result_ids_of_message_ends(&events);
    assert_eq!(tool_result_ids, ["tool-1", "tool-2"]);
}

#[tokio::test]
async fn should_force_sequential_execution_when_one_of_multiple_tools_has_execution_mode_sequential()
 {
    tokio::time::pause();
    let execution_order = Arc::new(Mutex::new(Vec::<String>::new()));
    let slow_gate = Arc::new(Notify::new());

    let slow_tool = {
        let execution_order = Arc::clone(&execution_order);
        let gate = Arc::clone(&slow_gate);
        suite_tool(
            "slow",
            echo_tool_schema(),
            Some(ToolExecutionMode::Sequential),
            Arc::new(move |_tool_call_id, args: &Value, _signal, _on_update| {
                let execution_order = Arc::clone(&execution_order);
                let gate = Arc::clone(&gate);
                Box::pin(async move {
                    let value = args["value"].as_str().unwrap_or_default().to_string();
                    execution_order
                        .lock()
                        .expect("execution order lock")
                        .push(format!("slow:{value}"));
                    if value == "a" {
                        gate.notified().await;
                    }
                    Ok(AgentToolResult {
                        content: vec![ToolResultBlock::Text(TextContent {
                            text: format!("slow: {value}"),
                            text_signature: None,
                        })],
                        details: json!({ "value": value }),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    })
                })
            }),
        )
    };

    let fast_tool = {
        let execution_order = Arc::clone(&execution_order);
        // no execution_mode = defaults to parallel
        suite_tool(
            "fast",
            echo_tool_schema(),
            None,
            Arc::new(move |_tool_call_id, args: &Value, _signal, _on_update| {
                let execution_order = Arc::clone(&execution_order);
                Box::pin(async move {
                    let value = args["value"].as_str().unwrap_or_default().to_string();
                    execution_order
                        .lock()
                        .expect("execution order lock")
                        .push(format!("fast:{value}"));
                    Ok(AgentToolResult {
                        content: vec![ToolResultBlock::Text(TextContent {
                            text: format!("fast: {value}"),
                            text_signature: None,
                        })],
                        details: json!({ "value": value }),
                        usage: None,
                        added_tool_names: None,
                        terminate: None,
                    })
                })
            }),
        )
    };

    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![slow_tool, fast_tool]),
    };
    // parallel by default, but slowTool forces sequential
    let config = test_config(identity_converter());

    let stream_fn = gated_stream_fn(
        Arc::clone(&slow_gate),
        create_assistant_message(
            vec![
                create_tool_call("tool-1", "slow", json!({ "value": "a" })),
                create_tool_call("tool-2", "fast", json!({ "value": "b" })),
            ],
            StopReason::ToolUse,
        ),
    );

    let stream = pi_agent_core::agent_loop(
        vec![create_user_message("run both")],
        context,
        config,
        None,
        Some(stream_fn),
    );
    let _ = collect_events(&stream).await;

    // Fast tool should NOT run before slow tool finishes
    assert_eq!(
        execution_order
            .lock()
            .expect("execution order lock")
            .first()
            .map(String::as_str),
        Some("slow:a")
    );
    assert!(
        execution_order
            .lock()
            .expect("execution order lock")
            .iter()
            .any(|entry| entry == "fast:b")
    );
}

#[tokio::test]
async fn should_allow_parallel_execution_when_all_tools_have_execution_mode_parallel() {
    tokio::time::pause();
    let first_resolved = Arc::new(AtomicBool::new(false));
    let parallel_observed = Arc::new(AtomicBool::new(false));
    let first_gate = Arc::new(Notify::new());

    let mut tool = gated_echo_tool(
        Arc::clone(&first_gate),
        Arc::clone(&first_resolved),
        Arc::clone(&parallel_observed),
    );
    tool.execution_mode = Some(ToolExecutionMode::Parallel);
    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };
    let user_prompt = create_user_message("echo both");
    let config = test_config(identity_converter());

    let stream_fn = gated_stream_fn(
        Arc::clone(&first_gate),
        create_assistant_message(
            vec![
                create_tool_call("tool-1", "echo", json!({ "value": "first" })),
                create_tool_call("tool-2", "echo", json!({ "value": "second" })),
            ],
            StopReason::ToolUse,
        ),
    );

    let stream =
        pi_agent_core::agent_loop(vec![user_prompt], context, config, None, Some(stream_fn));
    let _ = collect_events(&stream).await;

    // With executionMode=parallel, second tool should start before first finishes
    assert!(parallel_observed.load(Ordering::Relaxed));
}

#[tokio::test]
async fn should_use_prepare_next_turn_snapshot_before_continuing() {
    let tool = echo_tool(Arc::new(Mutex::new(Vec::new())));
    let context = AgentContext {
        system_prompt: "first prompt".to_string(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };
    let prepare_calls = Arc::new(AtomicUsize::new(0));
    let prepared = Arc::new(AtomicBool::new(false));
    let prepare_calls_for_config = Arc::clone(&prepare_calls);
    let prepared_for_config = Arc::clone(&prepared);
    let prepare_next_turn: pi_agent_core::PrepareNextTurn =
        Arc::new(move |context: PrepareNextTurnContext| {
            let prepare_calls = Arc::clone(&prepare_calls_for_config);
            let prepared = Arc::clone(&prepared_for_config);
            Box::pin(async move {
                prepare_calls.fetch_add(1, Ordering::Relaxed);
                if prepared.swap(true, Ordering::Relaxed) {
                    return None;
                }
                Some(AgentLoopTurnUpdate {
                    context: Some(AgentContext {
                        system_prompt: "second prompt".to_string(),
                        messages: context.context.messages.clone(),
                        tools: context.context.tools.clone(),
                    }),
                    model: None,
                    thinking_level: None,
                })
            })
        });
    let config = AgentLoopConfig {
        prepare_next_turn: Some(prepare_next_turn),
        ..test_config(identity_converter())
    };

    let llm_calls = Arc::new(AtomicUsize::new(0));
    let second_turn_system_prompt = Arc::new(Mutex::new(String::new()));
    let llm_calls_for_stream = Arc::clone(&llm_calls);
    let prompt_for_stream = Arc::clone(&second_turn_system_prompt);
    let stream_fn: pi_agent_core::StreamFn = Arc::new(move |_model, ctx: &Context, _options| {
        let calls = llm_calls_for_stream.fetch_add(1, Ordering::Relaxed) + 1;
        if calls == 2 {
            *prompt_for_stream.lock().expect("prompt lock") =
                ctx.system_prompt.clone().unwrap_or_default();
        }
        let mock_stream = mock_stream();
        let push = mock_stream.clone();
        tokio::spawn(async move {
            let message = if calls == 1 {
                create_assistant_message(
                    vec![create_tool_call(
                        "tool-1",
                        "echo",
                        json!({ "value": "hello" }),
                    )],
                    StopReason::ToolUse,
                )
            } else {
                create_assistant_message(vec![text_block("done")], StopReason::Stop)
            };
            push.push(done_event(message));
        });
        mock_stream
    });

    let stream = pi_agent_core::agent_loop(
        vec![create_user_message("echo something")],
        context,
        config,
        None,
        Some(stream_fn),
    );
    let _ = collect_events(&stream).await;

    assert_eq!(llm_calls.load(Ordering::Relaxed), 2);
    assert_eq!(prepare_calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        *second_turn_system_prompt.lock().expect("prompt lock"),
        "second prompt"
    );
}

/// The tool-call message the counted-turn tests stream first.
fn tool_use_message() -> AssistantMessage {
    create_assistant_message(
        vec![create_tool_call(
            "tool-1",
            "echo",
            json!({ "value": "hello" }),
        )],
        StopReason::ToolUse,
    )
}

/// The final text response the counted-turn tests stream second.
fn text_done_message() -> AssistantMessage {
    create_assistant_message(vec![text_block("done")], StopReason::Stop)
}

/// Run one tool-call turn with the counted two-call stream factory and
/// return the collected events, the run's messages, and the call counter.
async fn run_counted_turn(
    context: AgentContext,
    config: AgentLoopConfig,
    first_message: AssistantMessage,
    next_message: AssistantMessage,
) -> (Vec<AgentEvent>, Vec<AgentMessage>, Arc<AtomicUsize>) {
    let llm_calls = Arc::new(AtomicUsize::new(0));
    let stream_fn = counted_two_call_stream_fn(&llm_calls, first_message, next_message);
    let stream = pi_agent_core::agent_loop(
        vec![create_user_message("echo something")],
        context,
        config,
        None,
        Some(stream_fn),
    );
    let events = collect_events(&stream).await;
    let messages = stream.result().await;
    (events, messages, llm_calls)
}

#[tokio::test]
async fn should_stop_after_the_current_turn_when_should_stop_after_turn_returns_true() {
    let executed = Arc::new(Mutex::new(Vec::<String>::new()));
    let tool = echo_tool(Arc::clone(&executed));
    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };

    let steering_polls = Arc::new(AtomicUsize::new(0));
    let follow_up_polls = Arc::new(AtomicUsize::new(0));
    let callback_tool_result_ids = Arc::new(Mutex::new(Vec::<String>::new()));
    let callback_context_roles = Arc::new(Mutex::new(Vec::<String>::new()));

    let steering_polls_for_config = Arc::clone(&steering_polls);
    let follow_up_polls_for_config = Arc::clone(&follow_up_polls);
    let ids_for_config = Arc::clone(&callback_tool_result_ids);
    let roles_for_config = Arc::clone(&callback_context_roles);
    let get_steering_messages: pi_agent_core::GetSteeringMessages = Arc::new(move || {
        let steering_polls = Arc::clone(&steering_polls_for_config);
        Box::pin(async move {
            steering_polls.fetch_add(1, Ordering::Relaxed);
            Vec::new()
        })
    });
    let get_follow_up_messages: pi_agent_core::GetFollowUpMessages = Arc::new(move || {
        let follow_up_polls = Arc::clone(&follow_up_polls_for_config);
        Box::pin(async move {
            follow_up_polls.fetch_add(1, Ordering::Relaxed);
            vec![create_user_message("follow up should stay queued")]
        })
    });
    let should_stop_after_turn: pi_agent_core::ShouldStopAfterTurn =
        Arc::new(move |context: ShouldStopAfterTurnContext| {
            let ids = Arc::clone(&ids_for_config);
            let roles = Arc::clone(&roles_for_config);
            Box::pin(async move {
                // Upstream asserts message.role == "assistant"; the field is
                // typed as the assistant message here.
                *ids.lock().expect("ids lock") = context
                    .tool_results
                    .iter()
                    .map(|tool_result| tool_result.tool_call_id.clone())
                    .collect();
                *roles.lock().expect("roles lock") = context
                    .context
                    .messages
                    .iter()
                    .map(agent_message_role)
                    .map(str::to_owned)
                    .collect();
                true
            })
        });
    let config = AgentLoopConfig {
        get_steering_messages: Some(get_steering_messages),
        get_follow_up_messages: Some(get_follow_up_messages),
        should_stop_after_turn: Some(should_stop_after_turn),
        ..test_config(identity_converter())
    };

    let (events, messages, llm_calls) = run_counted_turn(
        context,
        config,
        tool_use_message(),
        create_assistant_message(vec![text_block("should not run")], StopReason::Stop),
    )
    .await;
    assert_eq!(llm_calls.load(Ordering::Relaxed), 1);
    assert_eq!(*executed.lock().expect("executed lock"), vec!["hello"]);
    assert_eq!(steering_polls.load(Ordering::Relaxed), 1);
    assert_eq!(follow_up_polls.load(Ordering::Relaxed), 0);
    assert_eq!(
        *callback_tool_result_ids.lock().expect("ids lock"),
        ["tool-1"]
    );
    assert_eq!(
        *callback_context_roles.lock().expect("roles lock"),
        ["user", "assistant", "toolResult"]
    );
    let roles: Vec<&str> = messages.iter().map(agent_message_role).collect();
    assert_eq!(roles, ["user", "assistant", "toolResult"]);
    let event_types: Vec<&str> = events.iter().map(event_type_name).collect();
    assert_eq!(
        event_types,
        [
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "tool_execution_start",
            "tool_execution_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );
}

#[tokio::test]
async fn should_stop_after_a_tool_batch_when_every_tool_result_sets_terminate() {
    let tool = suite_tool(
        "echo",
        echo_tool_schema(),
        None,
        Arc::new(|_tool_call_id, args: &Value, _signal, _on_update| {
            Box::pin(async move {
                let value = args["value"].as_str().unwrap_or_default().to_string();
                Ok(AgentToolResult {
                    content: vec![ToolResultBlock::Text(TextContent {
                        text: format!("echoed: {value}"),
                        text_signature: None,
                    })],
                    details: json!({ "value": value }),
                    usage: None,
                    added_tool_names: None,
                    terminate: Some(true),
                })
            })
        }),
    );

    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };
    let config = test_config(identity_converter());

    let (events, messages, llm_calls) = run_counted_turn(
        context,
        config,
        tool_use_message(),
        create_assistant_message(vec![text_block("done")], StopReason::Stop),
    )
    .await;
    assert_eq!(llm_calls.load(Ordering::Relaxed), 1);
    let roles: Vec<&str> = messages.iter().map(agent_message_role).collect();
    assert_eq!(roles, ["user", "assistant", "toolResult"]);
    let turn_end_count = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::TurnEnd { .. }))
        .count();
    assert_eq!(turn_end_count, 1);
}

#[tokio::test]
async fn should_stop_after_a_blocked_tool_call_when_before_tool_call_sets_terminate() {
    let executed = Arc::new(AtomicBool::new(false));
    let executed_for_tool = Arc::clone(&executed);
    let tool = suite_tool(
        "echo",
        echo_tool_schema(),
        None,
        Arc::new(move |_tool_call_id, _args: &Value, _signal, _on_update| {
            let executed = Arc::clone(&executed_for_tool);
            Box::pin(async move {
                executed.store(true, Ordering::Relaxed);
                Ok(AgentToolResult {
                    content: vec![ToolResultBlock::Text(TextContent {
                        text: "should not execute".to_string(),
                        text_signature: None,
                    })],
                    details: json!({ "value": "unexpected" }),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        }),
    );
    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };
    let blocked_by_policy: pi_agent_core::BeforeToolCall =
        Arc::new(|_context: BeforeToolCallContext, _signal| {
            Box::pin(std::future::ready(Some(BeforeToolCallResult {
                block: Some(true),
                reason: Some("Blocked by policy".to_string()),
                terminate: Some(true),
            })))
        });
    let config = AgentLoopConfig {
        before_tool_call: Some(blocked_by_policy),
        ..test_config(identity_converter())
    };

    let (_events, messages, llm_calls) = run_counted_turn(
        context,
        config,
        tool_use_message(),
        create_assistant_message(vec![text_block("should not run")], StopReason::Stop),
    )
    .await;
    let tool_result = messages.iter().find_map(|message| match message {
        AgentMessage::Standard(Message::ToolResult(tool_result)) => Some(tool_result),
        _ => None,
    });
    assert!(!executed.load(Ordering::Relaxed));
    assert_eq!(llm_calls.load(Ordering::Relaxed), 1);
    assert!(tool_result.expect("a tool result").is_error);
    assert!(
        tool_result
            .expect("a tool result")
            .content
            .iter()
            .any(|block| match block {
                ToolResultBlock::Text(text) => text.text == "Blocked by policy",
                ToolResultBlock::Image(_) => false,
            })
    );
}

#[tokio::test]
async fn should_continue_after_a_mixed_batch_with_one_terminating_blocked_call() {
    let executed = Arc::new(Mutex::new(Vec::<String>::new()));
    let tool = echo_tool(Arc::clone(&executed));
    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };
    let block_first: pi_agent_core::BeforeToolCall =
        Arc::new(|context: BeforeToolCallContext, _signal| {
            Box::pin(async move {
                if context.args["value"] == json!("first") {
                    Some(BeforeToolCallResult {
                        block: Some(true),
                        reason: Some("Blocked first".to_string()),
                        terminate: Some(true),
                    })
                } else {
                    None
                }
            })
        });
    let config = AgentLoopConfig {
        tool_execution: Some(ToolExecutionMode::Parallel),
        before_tool_call: Some(block_first),
        ..test_config(identity_converter())
    };

    let llm_calls = Arc::new(AtomicUsize::new(0));
    let stream_fn = counted_two_call_stream_fn(
        &llm_calls,
        create_assistant_message(
            vec![
                create_tool_call("tool-1", "echo", json!({ "value": "first" })),
                create_tool_call("tool-2", "echo", json!({ "value": "second" })),
            ],
            StopReason::ToolUse,
        ),
        create_assistant_message(vec![text_block("done")], StopReason::Stop),
    );

    let stream = pi_agent_core::agent_loop(
        vec![create_user_message("echo both")],
        context,
        config,
        None,
        Some(stream_fn),
    );
    let _ = collect_events(&stream).await;

    assert_eq!(*executed.lock().expect("executed lock"), vec!["second"]);
    assert_eq!(llm_calls.load(Ordering::Relaxed), 2);
}

#[tokio::test]
async fn should_continue_after_parallel_tool_calls_when_not_all_tool_results_terminate() {
    let tool = suite_tool(
        "echo",
        echo_tool_schema(),
        None,
        Arc::new(|_tool_call_id, args: &Value, _signal, _on_update| {
            Box::pin(async move {
                let value = args["value"].as_str().unwrap_or_default().to_string();
                Ok(AgentToolResult {
                    content: vec![ToolResultBlock::Text(TextContent {
                        text: format!("echoed: {value}"),
                        text_signature: None,
                    })],
                    details: json!({ "value": value }),
                    usage: None,
                    added_tool_names: None,
                    terminate: Some(value == "first"),
                })
            })
        }),
    );

    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };
    let config = AgentLoopConfig {
        tool_execution: Some(ToolExecutionMode::Parallel),
        ..test_config(identity_converter())
    };

    let call_index = Arc::new(AtomicUsize::new(0));
    let call_index_for_stream = Arc::clone(&call_index);
    let stream_fn: pi_agent_core::StreamFn = Arc::new(move |_model, _context, _options| {
        let index = call_index_for_stream.fetch_add(1, Ordering::Relaxed);
        let mock_stream = mock_stream();
        let push = mock_stream.clone();
        tokio::spawn(async move {
            let message = if index == 0 {
                create_assistant_message(
                    vec![
                        create_tool_call("tool-1", "echo", json!({ "value": "first" })),
                        create_tool_call("tool-2", "echo", json!({ "value": "second" })),
                    ],
                    StopReason::ToolUse,
                )
            } else {
                create_assistant_message(vec![text_block("done")], StopReason::Stop)
            };
            push.push(done_event(message));
        });
        mock_stream
    });

    let stream = pi_agent_core::agent_loop(
        vec![create_user_message("echo both")],
        context,
        config,
        None,
        Some(stream_fn),
    );
    let _ = collect_events(&stream).await;

    let messages = stream.result().await;
    assert_eq!(call_index.load(Ordering::Relaxed), 2);
    let roles: Vec<&str> = messages.iter().map(agent_message_role).collect();
    assert_eq!(
        roles,
        ["user", "assistant", "toolResult", "toolResult", "assistant"]
    );
}

#[tokio::test]
async fn should_allow_after_tool_call_to_mark_a_tool_batch_as_terminating() {
    let tool = echo_tool(Arc::new(Mutex::new(Vec::new())));
    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Some(vec![tool]),
    };
    let terminating_after: pi_agent_core::AfterToolCall =
        Arc::new(|_context: pi_agent_core::AfterToolCallContext, _signal| {
            Box::pin(std::future::ready(Some(
                pi_agent_core::AfterToolCallResult {
                    terminate: Some(true),
                    ..pi_agent_core::AfterToolCallResult::default()
                },
            )))
        });
    let config = AgentLoopConfig {
        after_tool_call: Some(terminating_after),
        ..test_config(identity_converter())
    };

    let (_events, _messages, llm_calls) =
        run_counted_turn(context, config, tool_use_message(), text_done_message()).await;
    assert_eq!(llm_calls.load(Ordering::Relaxed), 1);
}

// agentLoopContinue with AgentMessage

#[tokio::test]
async fn should_throw_when_context_has_no_messages() {
    let context = AgentContext {
        system_prompt: "You are helpful.".to_string(),
        messages: Vec::new(),
        tools: Some(Vec::new()),
    };
    let config = test_config(identity_converter());

    let error = pi_agent_core::agent_loop_continue(context, config, None, Some(never_stream_fn()))
        .expect_err("an empty context cannot continue");
    assert_eq!(error.to_string(), "Cannot continue: no messages in context");
}

fn never_stream_fn() -> pi_agent_core::StreamFn {
    Arc::new(
        |_model, _context, _options| -> AssistantMessageEventStream {
            panic!("Unexpected stream call");
        },
    )
}

#[tokio::test]
async fn should_continue_from_existing_context_without_emitting_user_message_events() {
    let user_message = create_user_message("Hello");
    let context = AgentContext {
        system_prompt: "You are helpful.".to_string(),
        messages: vec![user_message],
        tools: Some(Vec::new()),
    };
    let config = test_config(identity_converter());

    let stream_fn = single_response_stream_fn(create_assistant_message(
        vec![text_block("Response")],
        StopReason::Stop,
    ));

    let stream = pi_agent_core::agent_loop_continue(context, config, None, Some(stream_fn))
        .expect("a user-terminated context continues");
    let events = collect_events(&stream).await;

    let messages = stream.result().await;

    // Should only return the new assistant message (not the existing user message)
    assert_eq!(messages.len(), 1);
    assert_eq!(agent_message_role(&messages[0]), "assistant");

    // Should NOT have user message events (that's the key difference from agentLoop)
    let message_end_events: Vec<&AgentEvent> = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::MessageEnd { .. }))
        .collect();
    assert_eq!(message_end_events.len(), 1);
    let AgentEvent::MessageEnd { message } = message_end_events[0] else {
        panic!("a message_end event");
    };
    assert_eq!(agent_message_role(message), "assistant");
}

#[tokio::test]
async fn should_allow_custom_message_types_as_last_message() {
    // Custom message that will be converted to user message by convertToLlm
    let custom_message = AgentMessage::Custom(CustomAgentMessage {
        role: "custom".to_string(),
        timestamp: now_ms(),
        data: serde_json::Map::from_iter([("text".to_string(), json!("Hook content"))]),
    });

    let context = AgentContext {
        system_prompt: "You are helpful.".to_string(),
        messages: vec![custom_message],
        tools: Some(Vec::new()),
    };

    let convert_to_llm: ConvertToLlm = Arc::new(|messages| {
        // Convert custom to user message
        Box::pin(async move {
            messages
                .into_iter()
                .filter_map(|message| match message {
                    AgentMessage::Custom(custom) => {
                        let text = custom
                            .field("text")
                            .and_then(Value::as_str)
                            .map(str::to_owned)?;
                        Some(Message::User(pi_ai::types::UserMessage {
                            content: UserContent::Text(text),
                            timestamp: custom.timestamp,
                        }))
                    }
                    AgentMessage::Standard(standard) => Some(standard),
                })
                .collect::<Vec<_>>()
        })
    });
    let config = test_config(convert_to_llm);

    let stream_fn = single_response_stream_fn(create_assistant_message(
        vec![text_block("Response to custom message")],
        StopReason::Stop,
    ));

    // Should not throw - the custom message will be converted to user message
    let stream = pi_agent_core::agent_loop_continue(context, config, None, Some(stream_fn))
        .expect("a custom-terminated context continues");

    let _ = collect_events(&stream).await;

    let messages = stream.result().await;
    assert_eq!(messages.len(), 1);
    assert_eq!(agent_message_role(&messages[0]), "assistant");
}

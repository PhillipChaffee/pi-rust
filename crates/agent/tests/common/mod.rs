//! Shared fixtures for the agent-loop suites, mirroring the shapes upstream's
//! `test/agent-loop.test.ts` builds (the mock assistant stream, the `mock`
//! model, message factories, and the identity converter).

#![expect(
    dead_code,
    reason = "shared fixtures; each test binary uses the subset it needs"
)]
#![expect(
    unreachable_pub,
    reason = "the fixture module is compiled into every integration test binary as a private module"
)]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_agent_core::AgentContext;
use pi_agent_core::AgentEvent;
use pi_agent_core::AgentEventStream;
use pi_agent_core::AgentLoopConfig;
use pi_agent_core::AgentMessage;
use pi_agent_core::ConvertToLlm;
use pi_agent_core::StreamFn;
use pi_ai::auth::resolve::now_ms;
use pi_ai::types::Api;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Message;
use pi_ai::types::Model;
use pi_ai::types::SimpleStreamOptions;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::types::ToolCall;
use pi_ai::types::Usage;
use pi_ai::types::UserContent;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use serde_json::Value;
use serde_json::json;

pub fn mock_stream() -> pi_ai::utils::event_stream::AssistantMessageEventStream {
    assistant_message_event_stream()
}

pub fn create_usage() -> Usage {
    Usage::default()
}

pub fn create_model() -> Model {
    serde_json::from_value(json!({
        "id": "mock",
        "name": "mock",
        "api": "openai-responses",
        "provider": "openai",
        "baseUrl": "https://example.invalid",
        "reasoning": false,
        "input": ["text"],
        "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0 },
        "contextWindow": 8192,
        "maxTokens": 2048,
    }))
    .expect("a well-formed model literal")
}

pub fn create_assistant_message(
    content: Vec<AssistantBlock>,
    stop_reason: StopReason,
) -> AssistantMessage {
    AssistantMessage {
        content,
        api: Api::from("openai-responses"),
        provider: pi_ai::types::ProviderId::from("openai"),
        model: "mock".to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: create_usage(),
        stop_reason,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms(),
    }
}

pub fn create_user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(Message::User(pi_ai::types::UserMessage {
        content: UserContent::Text(text.to_string()),
        timestamp: now_ms(),
    }))
}

pub fn text_block(text: &str) -> AssistantBlock {
    AssistantBlock::Text(TextContent {
        text: text.to_string(),
        text_signature: None,
    })
}

pub fn create_tool_call(id: &str, name: &str, arguments: Value) -> AssistantBlock {
    AssistantBlock::ToolCall(ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments: match arguments {
            Value::Object(map) => map,
            _ => serde_json::Map::new(),
        },
        thought_signature: None,
        namespace: None,
    })
}

pub const fn done_event(message: AssistantMessage) -> AssistantMessageEvent {
    AssistantMessageEvent::Done {
        reason: message.stop_reason,
        message,
    }
}

/// `queueMicrotask`'s analog: the done event lands on a spawned task, so the
/// stream is still running when the stream function returns it.
pub fn push_done(
    stream: &pi_ai::utils::event_stream::AssistantMessageEventStream,
    message: AssistantMessage,
) {
    let stream = stream.clone();
    tokio::spawn(async move {
        stream.push(done_event(message));
    });
}

/// Simple identity converter for tests - just passes through standard messages
pub fn identity_converter() -> ConvertToLlm {
    Arc::new(|messages| {
        Box::pin(async move {
            messages
                .into_iter()
                .filter_map(|message| match message {
                    AgentMessage::Standard(standard) => Some(standard),
                    AgentMessage::Custom(_) => None,
                })
                .collect::<Vec<_>>()
        })
    })
}

pub fn test_config(convert_to_llm: ConvertToLlm) -> AgentLoopConfig {
    AgentLoopConfig {
        stream_options: SimpleStreamOptions::default(),
        model: create_model(),
        convert_to_llm,
        transform_context: None,
        get_api_key: None,
        should_stop_after_turn: None,
        prepare_next_turn: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        tool_execution: None,
        before_tool_call: None,
        after_tool_call: None,
    }
}

pub const fn test_context(
    messages: Vec<AgentMessage>,
    tools: Vec<pi_agent_core::AgentTool>,
) -> AgentContext {
    AgentContext {
        system_prompt: String::new(),
        messages,
        tools: Some(tools),
    }
}

pub fn agent_message_role(message: &AgentMessage) -> &str {
    match message {
        AgentMessage::Standard(Message::User(_)) => "user",
        AgentMessage::Standard(Message::Assistant(_)) => "assistant",
        AgentMessage::Standard(Message::ToolResult(_)) => "toolResult",
        AgentMessage::Custom(custom) => &custom.role,
    }
}

pub const fn event_type_name(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::AgentStart => "agent_start",
        AgentEvent::AgentEnd { .. } => "agent_end",
        AgentEvent::TurnStart => "turn_start",
        AgentEvent::TurnEnd { .. } => "turn_end",
        AgentEvent::MessageStart { .. } => "message_start",
        AgentEvent::MessageUpdate { .. } => "message_update",
        AgentEvent::MessageEnd { .. } => "message_end",
        AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
        AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
        AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
    }
}

/// Collect a run's events by iterating its stream, upstream's `for await`.
pub async fn collect_events(stream: &AgentEventStream) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

/// The two-call mock stream factory most tests share: the first call returns
/// `first_message`, later calls return `next_message`. Each response lands
/// via a spawned task, the queueMicrotask analog. The call count lands in
/// `llm_calls` for tests that assert how many provider calls a run made.
pub fn two_call_stream_fn(
    first_message: AssistantMessage,
    next_message: AssistantMessage,
) -> StreamFn {
    counted_two_call_stream_fn(&Arc::new(AtomicUsize::new(0)), first_message, next_message)
}

pub fn counted_two_call_stream_fn(
    llm_calls: &Arc<AtomicUsize>,
    first_message: AssistantMessage,
    next_message: AssistantMessage,
) -> StreamFn {
    let llm_calls = Arc::clone(llm_calls);
    Arc::new(move |_model, _context, _options| {
        let index = llm_calls.fetch_add(1, Ordering::Relaxed);
        let stream = assistant_message_event_stream();
        let push = stream.clone();
        let first = first_message.clone();
        let next = next_message.clone();
        tokio::spawn(async move {
            let message = if index == 0 { first } else { next };
            push.push(done_event(message));
        });
        stream
    })
}

/// The result shape tools that return no content carry.
pub const fn empty_tool_result(details: Value) -> pi_agent_core::AgentToolResult {
    pi_agent_core::AgentToolResult {
        content: Vec::new(),
        details,
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

/// A tool that counts executions and returns an empty result.
pub fn counting_tool(name: &str, executed: Arc<AtomicUsize>) -> pi_agent_core::AgentTool {
    suite_tool(
        name,
        json!({ "type": "object" }),
        None,
        Arc::new(move |_tool_call_id, _args: &Value, _signal, _on_update| {
            let executed = Arc::clone(&executed);
            Box::pin(async move {
                executed.fetch_add(1, Ordering::Relaxed);
                Ok(empty_tool_result(json!({})))
            })
        }),
    )
}

/// A stream fn that answers every call with one final message.
pub fn single_response_stream_fn(message: AssistantMessage) -> StreamFn {
    Arc::new(move |_model, _context, _options| {
        let stream = assistant_message_event_stream();
        push_done(&stream, message.clone());
        stream
    })
}

/// The tool schema the echo tools validate against.
pub fn echo_tool_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "value": { "type": "string" } },
        "required": ["value"],
    })
}

/// The tool surface a suite tool wraps: name, description, parameter schema.
pub fn suite_tool(
    name: &str,
    parameters: Value,
    execution_mode: Option<pi_agent_core::ToolExecutionMode>,
    execute: Arc<pi_agent_core::AgentToolExecuteFn>,
) -> pi_agent_core::AgentTool {
    pi_agent_core::AgentTool {
        tool: pi_ai::types::Tool {
            name: name.to_string(),
            description: format!("{name} tool"),
            parameters,
            constrained_sampling: None,
        },
        label: name.to_string(),
        prepare_arguments: None,
        execute,
        replay: None,
        execution_mode,
    }
}

/// A record of what ran, for executed-value assertions.
pub type ExecutedRecord = Arc<Mutex<Vec<String>>>;

//! The `Agent`-against-faux-provider end-to-end suite, ported 1:1 from
//! upstream `test/e2e.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The stream function is pi-ai's compat `stream_simple` over a
//! `registerFauxProvider` registration — the same wiring upstream's
//! `streamFn: streamSimple` makes — and the fixture `calculateTool` is
//! transcribed as a test-local custom tool. Upstream's `afterEach`
//! unregistration restates as the registration's drop guard.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod common;

use std::ops::Deref;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use common::event_type_name;
use pi_agent_core::Agent;
use pi_agent_core::AgentEvent;
use pi_agent_core::AgentInitialState;
use pi_agent_core::AgentMessage;
use pi_agent_core::AgentOptions;
use pi_agent_core::AgentTool;
use pi_agent_core::AgentToolError;
use pi_agent_core::AgentToolResult;
use pi_agent_core::StreamFn;
use pi_agent_core::ThinkingLevel;
use pi_ai::auth::resolve::now_ms;
use pi_ai::compat::register_faux_provider;
use pi_ai::compat::stream_simple;
use pi_ai::providers::faux::FauxAssistantMessageOptions;
use pi_ai::providers::faux::FauxContentBlock;
use pi_ai::providers::faux::FauxModelDefinition;
use pi_ai::providers::faux::FauxResponseFactory;
use pi_ai::providers::faux::FauxResponseStep;
use pi_ai::providers::faux::FauxTokenSize;
use pi_ai::providers::faux::RegisterFauxProviderOptions;
use pi_ai::providers::faux::faux_assistant_message;
use pi_ai::providers::faux::faux_text;
use pi_ai::providers::faux::faux_thinking;
use pi_ai::providers::faux::faux_tool_call;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::Context;
use pi_ai::types::Message;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::types::ThinkingContent;
use pi_ai::types::Tool;
use pi_ai::types::ToolCall;
use pi_ai::types::ToolResultBlock;
use pi_ai::types::ToolResultMessage;
use pi_ai::types::Usage;
use pi_ai::types::UserBlock;
use pi_ai::types::UserContent;
use serde_json::Value;
use serde_json::json;

/// The registration bookkeeping upstream's `registrations` array carries,
/// restated as a drop guard: the registration unregisters when the test
/// ends, upstream's `afterEach` pop.
struct RegistrationGuard(pi_ai::providers::faux::FauxProviderRegistration);

impl Drop for RegistrationGuard {
    fn drop(&mut self) {
        self.0.unregister();
    }
}

impl Deref for RegistrationGuard {
    type Target = pi_ai::providers::faux::FauxProviderRegistration;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn create_faux_registration(options: RegisterFauxProviderOptions) -> RegistrationGuard {
    RegistrationGuard(register_faux_provider(options))
}

/// The joined text of the message's text blocks, upstream's
/// `getTextContent`.
fn assistant_text(message: &AssistantMessage) -> String {
    let texts: Vec<&str> = message.content.iter().filter_map(BlockText::text).collect();
    texts.join("\n")
}

/// The joined text of a tool result's text blocks, the tool-result half of
/// upstream's `getTextContent`.
fn tool_result_text(message: &ToolResultMessage) -> String {
    let texts: Vec<&str> = message.content.iter().filter_map(BlockText::text).collect();
    texts.join("\n")
}

/// The text one block carries, the block kinds the fixture's messages hold.
trait BlockText {
    fn text(&self) -> Option<&str>;
}

impl BlockText for AssistantBlock {
    fn text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(&text.text),
            _ => None,
        }
    }
}

impl BlockText for ToolResultBlock {
    fn text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(&text.text),
            Self::Image(_) => None,
        }
    }
}

/// One recorded snapshot of the pending tool-call ids at a tool-execution
/// event, upstream's `{ type, ids }` records.
type PendingSnapshot = (&'static str, Vec<String>);

/// The agent the e2e scenarios run, upstream's repeated
/// `new Agent({ streamFn: streamSimple, initialState: ... })` literal with
/// the scenario's prompt, thinking level, and tools.
fn test_agent(
    model: &pi_ai::types::Model,
    system_prompt: &str,
    thinking_level: ThinkingLevel,
    tools: Vec<AgentTool>,
) -> Agent {
    Agent::new(AgentOptions {
        stream_fn: Some(stream_simple_fn()),
        initial_state: Some(AgentInitialState {
            system_prompt: system_prompt.to_owned(),
            model: Some(model.clone()),
            thinking_level: Some(thinking_level),
            tools,
            messages: Vec::new(),
        }),
        ..AgentOptions::default()
    })
}

/// The assistant-message literal the continue tests seed the transcript
/// with, upstream's inline messages carrying the faux model's wire
/// identity.
fn assistant_message_literal(
    content: &[AssistantBlock],
    stop_reason: StopReason,
    model: &pi_ai::types::Model,
) -> AssistantMessage {
    // The wire literal round-trip restates the field-by-field construction.
    serde_json::from_value(json!({
        "content": content,
        "api": model.api,
        "provider": model.provider,
        "model": model.id,
        "usage": Usage::default(),
        "stopReason": stop_reason,
        "timestamp": now_ms(),
    }))
    .expect("a well-formed assistant message")
}

/// The two-message settled transcript the prompt/continue scenarios end
/// with, asserted and reduced to the final assistant message — upstream's
/// repeated `state` assertions and role guard.
fn settled_exchange(agent: &Agent) -> AssistantMessage {
    let state = agent.state();
    assert!(!state.is_streaming);
    assert_eq!(state.messages.len(), 2);
    assert!(matches!(
        state.messages[0],
        AgentMessage::Standard(Message::User(_))
    ));
    match &state.messages[1] {
        AgentMessage::Standard(Message::Assistant(message)) => message.clone(),
        _ => panic!("Expected assistant message"),
    }
}

/// The stream function upstream passes (`streamFn: streamSimple`): pi-ai's
/// compat simple dispatch, boxed into the agent's stream-fn shape.
fn stream_simple_fn() -> StreamFn {
    Arc::new(|model: &pi_ai::types::Model, context: &Context, options| {
        stream_simple(model, context, options)
    })
}

/// A JS `new Function("return ${expression}")()` for the arithmetic the
/// fixture's callers use: decimal literals, `+ - * /`, parentheses, and
/// unary minus, evaluated in f64 and formatted the way JS prints numbers
/// (integral values print without a decimal part).
fn evaluate_expression(expression: &str) -> Result<f64, String> {
    let chars: Vec<char> = expression.chars().collect();
    let mut position = 0;
    let value = parse_sum(&chars, &mut position)?;
    skip_whitespace(&chars, &mut position);
    if position != chars.len() {
        return Err(format!("Unexpected token at position {position}"));
    }
    Ok(value)
}

const fn skip_whitespace(chars: &[char], position: &mut usize) {
    while *position < chars.len() && chars[*position].is_whitespace() {
        *position += 1;
    }
}

fn parse_sum(chars: &[char], position: &mut usize) -> Result<f64, String> {
    let mut value = parse_product(chars, position)?;
    loop {
        skip_whitespace(chars, position);
        let Some(operator) = chars.get(*position) else {
            return Ok(value);
        };
        match operator {
            '+' => {
                *position += 1;
                value += parse_product(chars, position)?;
            }
            '-' => {
                *position += 1;
                value -= parse_product(chars, position)?;
            }
            _ => return Ok(value),
        }
    }
}

fn parse_product(chars: &[char], position: &mut usize) -> Result<f64, String> {
    let mut value = parse_unary(chars, position)?;
    loop {
        skip_whitespace(chars, position);
        let Some(operator) = chars.get(*position) else {
            return Ok(value);
        };
        match operator {
            '*' => {
                *position += 1;
                value *= parse_unary(chars, position)?;
            }
            '/' => {
                *position += 1;
                let divisor = parse_unary(chars, position)?;
                if divisor == 0.0 {
                    return Err("Division by zero".to_owned());
                }
                value /= divisor;
            }
            _ => return Ok(value),
        }
    }
}

fn parse_unary(chars: &[char], position: &mut usize) -> Result<f64, String> {
    skip_whitespace(chars, position);
    match chars.get(*position) {
        Some('-') => {
            *position += 1;
            Ok(-parse_unary(chars, position)?)
        }
        Some('(') => {
            *position += 1;
            let value = parse_sum(chars, position)?;
            skip_whitespace(chars, position);
            match chars.get(*position) {
                Some(')') => {
                    *position += 1;
                    Ok(value)
                }
                _ => Err(format!("Expected ')' at position {position}")),
            }
        }
        Some(_) => parse_number(chars, position),
        None => Err("Unexpected end of expression".to_owned()),
    }
}

fn parse_number(chars: &[char], position: &mut usize) -> Result<f64, String> {
    skip_whitespace(chars, position);
    let start = *position;
    while *position < chars.len() && (chars[*position].is_ascii_digit() || chars[*position] == '.')
    {
        *position += 1;
    }
    let literal: String = chars[start..*position].iter().collect();
    literal
        .parse::<f64>()
        .map_err(|_| format!("Unexpected token at position {start}"))
}

/// The JS number formatting `${result}` interpolates: integral values print
/// without a decimal part.
fn format_js_number(value: f64) -> String {
    #[expect(
        clippy::float_cmp,
        reason = "the integrality test compares a value against its own truncated form, not two computed floats"
    )]
    if value == value.trunc() && value.abs() < 1.0e15 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the branch guards magnitude and integrality, so the i64 cast carries the value"
        )]
        let integral = value as i64;
        format!("{integral}")
    } else {
        format!("{value}")
    }
}

/// The fixture's `calculateTool`, transcribed as a test-local custom tool.
fn calculate_tool() -> AgentTool {
    AgentTool {
        tool: Tool {
            name: "calculate".to_owned(),
            description: "Evaluate mathematical expressions".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    },
                },
                "required": ["expression"],
            }),
            constrained_sampling: None,
        },
        label: "Calculator".to_owned(),
        prepare_arguments: None,
        execute: Arc::new(|_tool_call_id, args: &Value, _signal, _on_update| {
            let expression = args
                .get("expression")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            Box::pin(async move {
                let result = evaluate_expression(&expression).map_err(AgentToolError::from)?;
                Ok(AgentToolResult {
                    content: vec![ToolResultBlock::Text(TextContent {
                        text: format!("{expression} = {}", format_js_number(result)),
                        text_signature: None,
                    })],
                    details: json!({}),
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        }),
        replay: None,
        execution_mode: None,
    }
}

/// A text block for the fixture factories.
fn text_block(text: &str) -> FauxContentBlock {
    faux_text(text)
}

async fn basic_prompt(model: &pi_ai::types::Model) {
    let agent = test_agent(
        model,
        "You are a helpful assistant. Keep your responses concise.",
        ThinkingLevel::Off,
        Vec::new(),
    );

    agent
        .prompt("What is 2+2? Answer with just the number.")
        .await
        .expect("the prompt settles");

    let assistant_message = settled_exchange(&agent);
    assert!(assistant_text(&assistant_message).contains('4'));
}

async fn tool_execution(model: &pi_ai::types::Model) {
    let agent = test_agent(
        model,
        "You are a helpful assistant. Always use the calculator tool for math.",
        ThinkingLevel::Off,
        vec![calculate_tool()],
    );

    let pending_snapshots: Arc<Mutex<Vec<PendingSnapshot>>> = Arc::new(Mutex::new(Vec::new()));
    let listener_agent = agent.clone();
    let listener_snapshots = Arc::clone(&pending_snapshots);
    let _unsubscribe = agent.subscribe(Arc::new(move |event: AgentEvent, _token| {
        let agent = listener_agent.clone();
        let snapshots = Arc::clone(&listener_snapshots);
        Box::pin(async move {
            let kind = match &event {
                AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
                AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
                _ => return,
            };
            let ids: Vec<String> = agent.state().pending_tool_calls.iter().cloned().collect();
            snapshots.lock().expect("snapshot lock").push((kind, ids));
        })
    }));

    agent
        .prompt("Calculate 123 * 456 using the calculator tool.")
        .await
        .expect("the prompt settles");

    let (tool_result, final_text, pending_empty) = {
        let state = agent.state();
        assert!(!state.is_streaming);
        assert!(state.messages.len() >= 4);
        let tool_result = state
            .messages
            .iter()
            .find_map(|message| match message {
                AgentMessage::Standard(Message::ToolResult(result)) => Some(result.clone()),
                _ => None,
            })
            .expect("a tool result message");
        let final_message = match state.messages.last().expect("a final message") {
            AgentMessage::Standard(Message::Assistant(message)) => message.clone(),
            _ => panic!("Expected final assistant message"),
        };
        (
            tool_result,
            assistant_text(&final_message),
            state.pending_tool_calls.is_empty(),
        )
    };
    assert!(tool_result_text(&tool_result).contains("123 * 456 = 56088"));
    assert!(final_text.contains("56088"));
    assert!(pending_empty);
    assert_eq!(
        *pending_snapshots.lock().expect("snapshot lock"),
        vec![
            ("tool_execution_start", vec!["calc-1".to_owned()]),
            ("tool_execution_end", Vec::new()),
        ]
    );
}

async fn abort_execution(model: &pi_ai::types::Model) {
    let agent = test_agent(
        model,
        "You are a helpful assistant.",
        ThinkingLevel::Off,
        Vec::new(),
    );

    let abort_agent = agent.clone();
    let prompt = agent.prompt("Count slowly from 1 to 20.");
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        abort_agent.abort();
    });

    prompt.await.expect("the prompt settles");

    let state = agent.state();
    assert!(!state.is_streaming);
    assert!(state.messages.len() >= 2);

    let last_message = match state.messages.last().expect("a last message") {
        AgentMessage::Standard(Message::Assistant(message)) => message.clone(),
        _ => panic!("Expected assistant message"),
    };
    assert_eq!(last_message.stop_reason, StopReason::Aborted);
    assert!(last_message.error_message.is_some());
    assert_eq!(state.error_message, last_message.error_message);
}

async fn state_updates(model: &pi_ai::types::Model) {
    let agent = test_agent(
        model,
        "You are a helpful assistant.",
        ThinkingLevel::Off,
        Vec::new(),
    );

    let events: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let listener_events = Arc::clone(&events);
    let _unsubscribe = agent.subscribe(Arc::new(move |event: AgentEvent, _token| {
        let events = Arc::clone(&listener_events);
        Box::pin(async move {
            events
                .lock()
                .expect("event lock")
                .push(event_type_name(&event));
        })
    }));

    agent
        .prompt("Count from 1 to 5.")
        .await
        .expect("the prompt settles");

    let recorded = events.lock().expect("event lock").clone();
    let first_index = |name: &str| {
        recorded
            .iter()
            .position(|event| *event == name)
            .unwrap_or(usize::MAX)
    };
    let last_index = |name: &str| {
        recorded
            .iter()
            .rposition(|event| *event == name)
            .unwrap_or(usize::MAX)
    };
    assert!(recorded.contains(&"agent_start"));
    assert!(recorded.contains(&"turn_start"));
    assert!(recorded.contains(&"message_start"));
    assert!(recorded.contains(&"message_update"));
    assert!(recorded.contains(&"message_end"));
    assert!(recorded.contains(&"turn_end"));
    assert!(recorded.contains(&"agent_end"));
    assert!(first_index("agent_start") < first_index("message_start"));
    assert!(first_index("message_start") < first_index("message_end"));
    assert!(first_index("message_end") < last_index("agent_end"));

    let state = agent.state();
    assert!(!state.is_streaming);
    assert_eq!(state.messages.len(), 2);
}

async fn multi_turn_conversation(model: &pi_ai::types::Model) {
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(stream_simple_fn()),
        initial_state: Some(AgentInitialState {
            system_prompt: "You are a helpful assistant.".to_owned(),
            model: Some(model.clone()),
            thinking_level: Some(ThinkingLevel::Off),
            tools: Vec::new(),
            messages: Vec::new(),
        }),
        ..AgentOptions::default()
    });

    agent
        .prompt("My name is Alice.")
        .await
        .expect("the prompt settles");
    assert_eq!(agent.state().messages.len(), 2);

    agent
        .prompt("What is my name?")
        .await
        .expect("the prompt settles");
    assert_eq!(agent.state().messages.len(), 4);

    let last_message = match agent.state().messages[3] {
        AgentMessage::Standard(Message::Assistant(ref message)) => assistant_text(message),
        _ => panic!("Expected assistant message"),
    };
    assert!(last_message.to_lowercase().contains("alice"));
}

#[tokio::test]
async fn handles_a_basic_text_prompt() {
    let faux = create_faux_registration(RegisterFauxProviderOptions::default());
    faux.set_responses([
        faux_assistant_message("4", FauxAssistantMessageOptions::default()).into(),
    ]);
    basic_prompt(&faux.first_model()).await;
}

#[tokio::test]
async fn executes_tools_and_tracks_pending_tool_calls() {
    let faux = create_faux_registration(RegisterFauxProviderOptions::default());
    faux.set_responses([
        faux_assistant_message(
            vec![
                text_block("Let me calculate that."),
                faux_tool_call(
                    "calculate",
                    serde_json::from_value::<serde_json::Map<String, Value>>(json!({
                        "expression": "123 * 456"
                    }))
                    .expect("an arguments map"),
                    Some("calc-1".to_owned()),
                ),
            ],
            FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..FauxAssistantMessageOptions::default()
            },
        )
        .into(),
        faux_assistant_message(
            "The result is 56088.",
            FauxAssistantMessageOptions::default(),
        )
        .into(),
    ]);
    tool_execution(&faux.first_model()).await;
}

#[tokio::test]
async fn handles_abort_during_streaming() {
    let faux = create_faux_registration(RegisterFauxProviderOptions {
        tokens_per_second: Some(20.0),
        token_size: Some(FauxTokenSize {
            min: Some(2),
            max: Some(2),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    faux.set_responses([faux_assistant_message(
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);
    abort_execution(&faux.first_model()).await;
}

#[tokio::test]
async fn emits_lifecycle_updates_while_streaming() {
    let faux = create_faux_registration(RegisterFauxProviderOptions {
        token_size: Some(FauxTokenSize {
            min: Some(1),
            max: Some(1),
        }),
        ..RegisterFauxProviderOptions::default()
    });
    faux.set_responses([faux_assistant_message(
        "1 2 3 4 5",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);
    state_updates(&faux.first_model()).await;
}

#[tokio::test]
async fn maintains_context_across_multiple_turns() {
    let faux = create_faux_registration(RegisterFauxProviderOptions::default());
    let has_alice_factory: FauxResponseFactory =
        Arc::new(|context: &Context, _options, _state, _model| {
            let has_alice = context.messages.iter().any(|message| match message {
                Message::User(user) => match &user.content {
                    UserContent::Text(text) => text.contains("Alice"),
                    UserContent::Blocks(blocks) => blocks.iter().any(|block| match block {
                        UserBlock::Text(text) => text.text.contains("Alice"),
                        UserBlock::Image(_) => false,
                    }),
                },
                _ => false,
            });
            let response = if has_alice {
                "Your name is Alice."
            } else {
                "I do not know your name."
            };
            Box::pin(async move {
                Ok(faux_assistant_message(
                    response,
                    FauxAssistantMessageOptions::default(),
                ))
            })
        });
    faux.set_responses([
        faux_assistant_message(
            "Nice to meet you, Alice.",
            FauxAssistantMessageOptions::default(),
        )
        .into(),
        FauxResponseStep::Factory(has_alice_factory),
    ]);
    multi_turn_conversation(&faux.first_model()).await;
}

#[tokio::test]
async fn preserves_thinking_content_blocks() {
    let faux = create_faux_registration(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition {
            id: "faux-reasoning".to_owned(),
            reasoning: Some(true),
            ..FauxModelDefinition::default()
        }],
        ..RegisterFauxProviderOptions::default()
    });
    faux.set_responses([faux_assistant_message(
        vec![faux_thinking("step by step"), text_block("4")],
        FauxAssistantMessageOptions::default(),
    )
    .into()]);

    let agent = test_agent(
        &faux.first_model(),
        "You are a helpful assistant.",
        ThinkingLevel::Low,
        Vec::new(),
    );

    agent
        .prompt("What is 2+2?")
        .await
        .expect("the prompt settles");

    let assistant_message = {
        let state = agent.state();
        match &state.messages[1] {
            AgentMessage::Standard(Message::Assistant(message)) => message.clone(),
            _ => panic!("Expected assistant message"),
        }
    };
    assert_eq!(
        assistant_message.content,
        vec![
            AssistantBlock::Thinking(ThinkingContent {
                thinking: "step by step".to_owned(),
                thinking_signature: None,
                redacted: None,
            }),
            AssistantBlock::Text(TextContent {
                text: "4".to_owned(),
                text_signature: None,
            }),
        ]
    );
}

#[tokio::test]
async fn throws_when_no_messages_in_context() {
    let faux = create_faux_registration(RegisterFauxProviderOptions::default());
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(stream_simple_fn()),
        initial_state: Some(AgentInitialState {
            system_prompt: "Test".to_owned(),
            model: Some(faux.first_model()),
            ..AgentInitialState::default()
        }),
        ..AgentOptions::default()
    });

    let error = agent.continue_run().await.expect_err("continue rejects");
    assert_eq!(error.to_string(), "No messages to continue from");
}

#[tokio::test]
async fn throws_when_last_message_is_assistant() {
    let faux = create_faux_registration(RegisterFauxProviderOptions::default());
    let model = faux.first_model();
    let agent = Agent::new(AgentOptions {
        stream_fn: Some(stream_simple_fn()),
        initial_state: Some(AgentInitialState {
            system_prompt: "Test".to_owned(),
            model: Some(model.clone()),
            ..AgentInitialState::default()
        }),
        ..AgentOptions::default()
    });

    let assistant_message = assistant_message_literal(
        &[AssistantBlock::Text(TextContent {
            text: "Hello".to_owned(),
            text_signature: None,
        })],
        StopReason::Stop,
        &model,
    );
    agent.set_messages(&[AgentMessage::Standard(Message::Assistant(
        assistant_message,
    ))]);

    let error = agent.continue_run().await.expect_err("continue rejects");
    assert_eq!(
        error.to_string(),
        "Cannot continue from message role: assistant"
    );
}

#[tokio::test]
async fn continues_and_gets_a_response_when_last_message_is_user() {
    let faux = create_faux_registration(RegisterFauxProviderOptions::default());
    faux.set_responses([faux_assistant_message(
        "HELLO WORLD",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);
    let agent = test_agent(
        &faux.first_model(),
        "You are a helpful assistant. Follow instructions exactly.",
        ThinkingLevel::Off,
        Vec::new(),
    );

    let user_message = common::create_user_message("Say exactly: HELLO WORLD");
    agent.set_messages(&[user_message]);

    agent.continue_run().await.expect("the run settles");

    let assistant_message = settled_exchange(&agent);
    assert!(
        assistant_text(&assistant_message)
            .to_uppercase()
            .contains("HELLO WORLD")
    );
}

#[tokio::test]
async fn continues_and_processes_tool_results() {
    let faux = create_faux_registration(RegisterFauxProviderOptions::default());
    let model = faux.first_model();
    faux.set_responses([faux_assistant_message(
        "The answer is 8.",
        FauxAssistantMessageOptions::default(),
    )
    .into()]);
    let agent = test_agent(
        &model,
        "You are a helpful assistant. After getting a calculation result, state the answer clearly.",
        ThinkingLevel::Off,
        vec![calculate_tool()],
    );

    let user_message = common::create_user_message("What is 5 + 3?");

    let assistant_message = assistant_message_literal(
        &[
            AssistantBlock::Text(TextContent {
                text: "Let me calculate that.".to_owned(),
                text_signature: None,
            }),
            AssistantBlock::ToolCall(ToolCall {
                id: "calc-1".to_owned(),
                name: "calculate".to_owned(),
                arguments: serde_json::from_value::<serde_json::Map<String, Value>>(json!({
                    "expression": "5 + 3"
                }))
                .expect("an arguments map"),
                thought_signature: None,
                namespace: None,
            }),
        ],
        StopReason::ToolUse,
        &model,
    );

    let tool_result = ToolResultMessage {
        tool_call_id: "calc-1".to_owned(),
        tool_name: "calculate".to_owned(),
        content: vec![ToolResultBlock::Text(TextContent {
            text: "5 + 3 = 8".to_owned(),
            text_signature: None,
        })],
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: now_ms(),
    };

    agent.set_messages(&[
        user_message,
        AgentMessage::Standard(Message::Assistant(assistant_message)),
        AgentMessage::Standard(Message::ToolResult(tool_result)),
    ]);

    agent.continue_run().await.expect("the run settles");

    let last_message = {
        let state = agent.state();
        assert!(!state.is_streaming);
        assert!(state.messages.len() >= 4);
        match state.messages.last().expect("a last message") {
            AgentMessage::Standard(Message::Assistant(message)) => assistant_text(message),
            _ => panic!("Expected assistant message"),
        }
    };
    assert!(last_message.contains('8'));
}

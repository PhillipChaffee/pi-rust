#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use super::*;
use crate::types::AgentLoopTurnUpdate;

/// A minimal assistant message for the pure-helper assertions; the
/// provider shapes are arbitrary, only the struct shape matters.
fn unit_assistant_message() -> AssistantMessage {
    serde_json::from_value(serde_json::json!({
        "content": [],
        "api": "google-generative-ai",
        "provider": "google",
        "model": "unit",
        "usage": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 0,
            "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 },
        },
        "stopReason": "pending",
        "timestamp": 0,
    }))
    .expect("a well-formed assistant message literal")
}

fn unit_config() -> AgentLoopConfig {
    AgentLoopConfig {
        stream_options: SimpleStreamOptions {
            api_key: Some("unit-key".to_string()),
            ..SimpleStreamOptions::default()
        },
        model: unit_model("unit"),
        convert_to_llm: Arc::new(|_messages| Box::pin(async move { Vec::new() })),
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

fn unit_model(id: &str) -> pi_ai::types::Model {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "name": id,
        "api": "google-generative-ai",
        "provider": "google",
        "baseUrl": "https://unit.example",
        "reasoning": true,
        "input": ["text", "image"],
        "cost": { "input": 1.0, "output": 2.0, "cacheRead": 0.5, "cacheWrite": 3.0 },
        "contextWindow": 4096,
        "maxTokens": 1024,
    }))
    .expect("a well-formed model literal")
}

#[test]
fn stream_reasoning_maps_every_agent_level_to_the_request_reasoning() {
    // The agent level adds `off`, which the request carries as an absent
    // field; every other level passes through.
    let expected = [
        (ThinkingLevel::Off, None),
        (
            ThinkingLevel::Minimal,
            Some(pi_ai::types::ThinkingLevel::Minimal),
        ),
        (ThinkingLevel::Low, Some(pi_ai::types::ThinkingLevel::Low)),
        (
            ThinkingLevel::Medium,
            Some(pi_ai::types::ThinkingLevel::Medium),
        ),
        (ThinkingLevel::High, Some(pi_ai::types::ThinkingLevel::High)),
        (
            ThinkingLevel::Xhigh,
            Some(pi_ai::types::ThinkingLevel::Xhigh),
        ),
        (ThinkingLevel::Max, Some(pi_ai::types::ThinkingLevel::Max)),
    ];
    for (level, reasoning) in expected {
        assert_eq!(stream_reasoning(level), reasoning);
    }
}

#[test]
fn apply_turn_update_replaces_the_presented_fields_and_keeps_the_absent_ones() {
    let mut context = AgentContext {
        system_prompt: "first".to_string(),
        messages: Vec::new(),
        tools: None,
    };
    let mut config = unit_config();
    let next_context = AgentContext {
        system_prompt: "second".to_string(),
        messages: Vec::new(),
        tools: None,
    };
    apply_turn_update(
        &mut context,
        &mut config,
        AgentLoopTurnUpdate {
            context: Some(next_context),
            model: Some(unit_model("other")),
            thinking_level: Some(ThinkingLevel::Low),
        },
    );
    assert_eq!(context.system_prompt, "second");
    assert_eq!(config.model.id, "other");
    assert_eq!(
        config.stream_options.reasoning,
        Some(pi_ai::types::ThinkingLevel::Low)
    );

    // An all-default update keeps every live value.
    apply_turn_update(&mut context, &mut config, AgentLoopTurnUpdate::default());
    assert_eq!(context.system_prompt, "second");
    assert_eq!(config.model.id, "other");
    assert_eq!(
        config.stream_options.reasoning,
        Some(pi_ai::types::ThinkingLevel::Low)
    );
}

#[test]
fn event_partial_answers_the_live_accumulator_and_none_for_the_terminal_events() {
    let partial = unit_assistant_message();
    assert!(
        event_partial(&AssistantMessageEvent::Start {
            partial: partial.clone(),
        })
        .is_some()
    );
    assert!(
        event_partial(&AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "x".to_string(),
            partial: partial.clone(),
        })
        .is_some()
    );
    assert!(
        event_partial(&AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: partial.clone(),
        })
        .is_none()
    );
    assert!(
        event_partial(&AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: partial,
        })
        .is_none()
    );
}

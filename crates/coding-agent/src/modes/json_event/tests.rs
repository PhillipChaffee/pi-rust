#![expect(
    clippy::unwrap_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use super::*;
use pi_ai::types::Api;
use pi_ai::types::AssistantMessage;
use pi_ai::types::ProviderId;
use pi_ai::types::TextContent;
use pi_ai::types::ToolCall;
use pi_ai::types::Usage;

use crate::compaction::CompactionResult;
use crate::extensions::reasons::CompactionReason;

fn assistant_message(content: Vec<AssistantBlock>, usage: Usage) -> AssistantMessage {
    AssistantMessage {
        content,
        api: Api::from("test"),
        provider: ProviderId("test".to_owned()),
        model: "test-model".to_owned(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage,
        stop_reason: pi_ai::types::StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

fn text_block(text: &str) -> AssistantBlock {
    AssistantBlock::Text(TextContent {
        text: text.to_owned(),
        text_signature: None,
    })
}

fn tool_call_block() -> AssistantBlock {
    AssistantBlock::ToolCall(ToolCall {
        id: "call-1".to_owned(),
        name: "read".to_owned(),
        arguments: serde_json::Map::new(),
        thought_signature: None,
        namespace: None,
    })
}

fn update_message(assistant: AssistantMessage) -> AgentMessage {
    AgentMessage::Standard(Message::Assistant(assistant))
}

#[test]
fn message_update_strips_the_cumulative_snapshot_and_keeps_usage() {
    let usage = Usage {
        input: 3,
        output: 4,
        total_tokens: 7,
        ..Usage::default()
    };
    let partial = assistant_message(vec![text_block("growing")], usage);
    let event = AgentSessionEvent::MessageUpdate {
        message: update_message(partial.clone()),
        assistant_message_event: AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "growing".to_owned(),
            partial,
        },
    };

    let value = to_json_event(&event).unwrap();

    assert_eq!(value["type"], "message_update");
    assert_eq!(value["usage"]["input"], 3);
    assert_eq!(value["usage"]["totalTokens"], 7);
    assert_eq!(value["assistantMessageEvent"]["type"], "text_delta");
    assert_eq!(value["assistantMessageEvent"]["contentIndex"], 0);
    assert!(value["assistantMessageEvent"].get("partial").is_none());
}

#[test]
fn toolcall_start_adds_the_id_and_tool_name() {
    let partial = assistant_message(vec![tool_call_block()], Usage::default());
    let event = AgentSessionEvent::MessageUpdate {
        message: update_message(partial.clone()),
        assistant_message_event: AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial,
        },
    };

    let value = to_json_event(&event).unwrap();
    let delta = &value["assistantMessageEvent"];

    assert_eq!(delta["type"], "toolcall_start");
    assert_eq!(delta["id"], "call-1");
    assert_eq!(delta["toolName"], "read");
    assert!(delta.get("partial").is_none());
}

#[test]
fn toolcall_start_rejects_a_non_tool_call_block() {
    let partial = assistant_message(vec![text_block("not a call")], Usage::default());
    let event = AgentSessionEvent::MessageUpdate {
        message: update_message(partial.clone()),
        assistant_message_event: AssistantMessageEvent::ToolcallStart {
            content_index: 0,
            partial,
        },
    };

    assert_eq!(
        to_json_event(&event),
        Err("toolcall_start content at index 0 is not a tool call".to_owned())
    );
}

#[test]
fn message_update_rejects_a_non_assistant_message() {
    let event = AgentSessionEvent::MessageUpdate {
        message: AgentMessage::Standard(Message::User(pi_ai::types::UserMessage {
            content: pi_ai::types::UserContent::Text("user".to_owned()),
            timestamp: 1,
        })),
        assistant_message_event: AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: String::new(),
            partial: assistant_message(Vec::new(), Usage::default()),
        },
    };

    assert_eq!(
        to_json_event(&event),
        Err("message_update message is not an assistant message".to_owned())
    );
}

#[test]
fn queue_update_uses_the_camel_case_queue_names() {
    let value = to_json_event(&AgentSessionEvent::QueueUpdate {
        steering: vec!["steer".to_owned()],
        follow_up: vec!["later".to_owned()],
    })
    .unwrap();

    assert_eq!(value["type"], "queue_update");
    assert_eq!(value["steering"][0], "steer");
    assert_eq!(value["followUp"][0], "later");
}

#[test]
fn compaction_end_carries_the_result_and_error_message() {
    let value = to_json_event(&AgentSessionEvent::CompactionEnd {
        reason: CompactionReason::Threshold,
        result: Some(CompactionResult {
            summary: "summary".to_owned(),
            first_kept_entry_id: "entry-1".to_owned(),
            tokens_before: 900,
            estimated_tokens_after: Some(120),
            usage: Usage::default(),
            details: None,
        }),
        aborted: false,
        will_retry: true,
        error_message: Some("boom".to_owned()),
    })
    .unwrap();

    assert_eq!(value["type"], "compaction_end");
    assert_eq!(value["reason"], "threshold");
    assert_eq!(value["willRetry"], true);
    assert_eq!(value["result"]["firstKeptEntryId"], "entry-1");
    assert_eq!(value["result"]["tokensBefore"], 900);
    assert_eq!(value["result"]["estimatedTokensAfter"], 120);
    assert_eq!(value["errorMessage"], "boom");
}

#[test]
fn compaction_end_omits_absent_optionals() {
    let value = to_json_event(&AgentSessionEvent::CompactionEnd {
        reason: CompactionReason::Manual,
        result: None,
        aborted: true,
        will_retry: false,
        error_message: None,
    })
    .unwrap();

    assert!(value.get("result").is_none());
    assert!(value.get("errorMessage").is_none());
    assert_eq!(value["aborted"], true);
}

#[test]
fn session_info_changed_omits_a_missing_name() {
    let cleared = to_json_event(&AgentSessionEvent::SessionInfoChanged { name: None }).unwrap();
    assert!(cleared.get("name").is_none());

    let named = to_json_event(&AgentSessionEvent::SessionInfoChanged {
        name: Some("named".to_owned()),
    })
    .unwrap();
    assert_eq!(named["name"], "named");
}

#[test]
fn summarization_attempt_start_splits_branch_and_compaction_sources() {
    let branch = to_json_event(&AgentSessionEvent::SummarizationRetryAttemptStart {
        source: SummarizationSource::BranchSummary,
    })
    .unwrap();
    assert_eq!(branch["source"], "branchSummary");
    assert!(branch.get("reason").is_none());

    let compaction = to_json_event(&AgentSessionEvent::SummarizationRetryAttemptStart {
        source: SummarizationSource::Compaction(CompactionReason::Overflow),
    })
    .unwrap();
    assert_eq!(compaction["source"], "compaction");
    assert_eq!(compaction["reason"], "overflow");
}

#[test]
fn bash_execution_update_omits_a_missing_id() {
    let value = to_json_event(&AgentSessionEvent::BashExecutionUpdate {
        id: None,
        delta: "output".to_owned(),
    })
    .unwrap();

    assert!(value.get("id").is_none());
    assert_eq!(value["delta"], "output");
}

#[expect(
    clippy::too_many_lines,
    reason = "the table is the event union's complete index"
)]
#[test]
fn every_remaining_event_variant_projects_onto_the_wire() {
    use crate::session_manager::MessageEntry;
    use crate::session_manager::SessionEntry;
    use crate::session_manager::SessionEntryBase;
    use pi_agent_core::types::AgentToolResult;
    use pi_agent_core::types::ThinkingLevel;
    use pi_ai::types::ToolResultMessage;
    use pi_ai::types::UserMessage;

    let user = AgentMessage::Standard(Message::User(UserMessage {
        content: pi_ai::types::UserContent::Text("hello".to_owned()),
        timestamp: 1,
    }));
    let tool_result = AgentToolResult {
        content: Vec::new(),
        details: serde_json::json!({}),
        usage: None,
        added_tool_names: None,
        terminate: None,
    };
    let result_message = ToolResultMessage {
        tool_call_id: "call-1".to_owned(),
        tool_name: "read".to_owned(),
        content: Vec::new(),
        details: None,
        usage: None,
        added_tool_names: None,
        is_error: false,
        timestamp: 1,
    };
    let entry = SessionEntry::Message(MessageEntry {
        base: SessionEntryBase {
            id: Some("entry-1".to_owned()),
            parent_id: None,
            timestamp: "2026-01-01T00:00:00Z".to_owned(),
            extras: serde_json::Map::new(),
        },
        message: None,
        extras: serde_json::Map::new(),
    });

    let cases = vec![
        (AgentSessionEvent::AgentStart, "agent_start"),
        (
            AgentSessionEvent::AgentEnd {
                messages: vec![user.clone()],
                will_retry: false,
            },
            "agent_end",
        ),
        (AgentSessionEvent::TurnStart, "turn_start"),
        (
            AgentSessionEvent::TurnEnd {
                message: user.clone(),
                tool_results: vec![result_message],
            },
            "turn_end",
        ),
        (
            AgentSessionEvent::MessageStart {
                message: user.clone(),
            },
            "message_start",
        ),
        (
            AgentSessionEvent::MessageEnd { message: user },
            "message_end",
        ),
        (
            AgentSessionEvent::ToolExecutionStart {
                tool_call_id: "call-1".to_owned(),
                tool_name: "read".to_owned(),
                args: serde_json::json!({ "path": "a" }),
            },
            "tool_execution_start",
        ),
        (
            AgentSessionEvent::ToolExecutionUpdate {
                tool_call_id: "call-1".to_owned(),
                tool_name: "read".to_owned(),
                args: serde_json::json!({}),
                partial_result: tool_result.clone(),
            },
            "tool_execution_update",
        ),
        (
            AgentSessionEvent::ToolExecutionEnd {
                tool_call_id: "call-1".to_owned(),
                tool_name: "read".to_owned(),
                result: tool_result,
                is_error: false,
            },
            "tool_execution_end",
        ),
        (AgentSessionEvent::AgentSettled, "agent_settled"),
        (
            AgentSessionEvent::CompactionStart {
                reason: CompactionReason::Manual,
            },
            "compaction_start",
        ),
        (AgentSessionEvent::EntryAppended { entry }, "entry_appended"),
        (
            AgentSessionEvent::ThinkingLevelChanged {
                level: ThinkingLevel::Low,
            },
            "thinking_level_changed",
        ),
        (
            AgentSessionEvent::AutoRetryStart {
                attempt: 1,
                max_attempts: 3,
                delay_ms: 500,
                error_message: "boom".to_owned(),
            },
            "auto_retry_start",
        ),
        (
            AgentSessionEvent::AutoRetryEnd {
                success: false,
                attempt: 3,
                final_error: Some("boom".to_owned()),
            },
            "auto_retry_end",
        ),
        (
            AgentSessionEvent::SummarizationRetryScheduled {
                attempt: 1,
                max_attempts: 3,
                delay_ms: 500,
                error_message: "boom".to_owned(),
            },
            "summarization_retry_scheduled",
        ),
        (
            AgentSessionEvent::SummarizationRetryFinished,
            "summarization_retry_finished",
        ),
        (
            AgentSessionEvent::BashExecutionUpdate {
                id: Some("bash-1".to_owned()),
                delta: "output".to_owned(),
            },
            "bash_execution_update",
        ),
    ];

    for (event, wire_type) in cases {
        let value = to_json_event(&event).unwrap();
        assert_eq!(value["type"], wire_type);
    }
}

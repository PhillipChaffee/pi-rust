use super::*;
use pi_agent_core::types::ThinkingLevel;
use pi_ai::types::Message;
use pi_ai::types::{Api, ProviderId, StopReason};
use serde_json::json;

use crate::compaction::DEFAULT_COMPACTION_SETTINGS;
use crate::compaction::FileOperations;
use crate::session_manager::{CompactionEntry, SessionEntry, SessionEntryBase};

fn assistant_message() -> AgentMessage {
    AgentMessage::Standard(Message::Assistant(pi_ai::types::AssistantMessage {
        content: Vec::new(),
        api: Api::from("openai"),
        provider: ProviderId::from("openai"),
        model: "gpt-test".to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }))
}

fn tool_result() -> AgentToolResult {
    AgentToolResult {
        content: Vec::new(),
        details: json!({}),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

fn model() -> Model {
    Model {
        id: "gpt-test".to_string(),
        name: "GPT Test".to_string(),
        api: Api::from("openai"),
        provider: ProviderId::from("openai"),
        base_url: "https://example.invalid".to_string(),
        reasoning: false,
        thinking_level_map: None,
        input: Vec::new(),
        cost: pi_ai::types::ModelCost::default(),
        context_window: 0,
        max_tokens: 0,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn compaction_preparation() -> CompactionPreparation {
    CompactionPreparation {
        first_kept_entry_id: "e-1".to_string(),
        messages_to_summarize: vec![assistant_message()],
        turn_prefix_messages: Vec::new(),
        is_split_turn: false,
        tokens_before: 100,
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: DEFAULT_COMPACTION_SETTINGS,
    }
}

fn compaction_entry() -> CompactionEntry {
    CompactionEntry {
        base: SessionEntryBase {
            id: Some("e-2".to_string()),
            parent_id: None,
            timestamp: "2026-01-01T00:00:00.000Z".to_string(),
            extras: serde_json::Map::new(),
        },
        summary: "s".to_string(),
        first_kept_entry_id: Some("e-1".to_string()),
        tokens_before: 100,
        details: None,
        usage: Some(Usage::default()),
        from_hook: None,
        extras: serde_json::Map::new(),
    }
}

fn tree_preparation() -> TreePreparation {
    TreePreparation {
        target_id: "e-3".to_string(),
        old_leaf_id: None,
        common_ancestor_id: None,
        entries_to_summarize: Vec::new(),
        user_wants_summary: false,
        custom_instructions: None,
        replace_instructions: None,
        label: None,
    }
}

fn sample_events() -> Vec<ExtensionEvent> {
    vec![
        ExtensionEvent::ProjectTrust {
            cwd: "/tmp/project".to_string(),
        },
        ExtensionEvent::ResourcesDiscover {
            cwd: "/tmp/project".to_string(),
            reason: SessionStartReason::Startup,
        },
        ExtensionEvent::SessionStart {
            reason: SessionStartReason::Startup,
            previous_session_file: None,
        },
        ExtensionEvent::SessionInfoChanged {
            name: Some("n".to_string()),
        },
        ExtensionEvent::SessionBeforeSwitch {
            reason: SessionShutdownReason::Quit,
            target_session_file: None,
        },
        ExtensionEvent::SessionBeforeFork {
            entry_id: "e-1".to_string(),
            position: ForkPosition::Before,
        },
        ExtensionEvent::SessionBeforeCompact {
            preparation: compaction_preparation(),
            branch_entries: vec![SessionEntry::Message(
                crate::session_manager::MessageEntry {
                    base: SessionEntryBase {
                        id: Some("e-1".to_string()),
                        parent_id: None,
                        timestamp: String::new(),
                        extras: serde_json::Map::new(),
                    },
                    message: Some(assistant_message()),
                    extras: serde_json::Map::new(),
                },
            )],
            custom_instructions: None,
            reason: CompactionReason::Manual,
            will_retry: false,
            signal: None,
        },
        ExtensionEvent::SessionCompact {
            compaction_entry: compaction_entry(),
            from_extension: false,
            reason: CompactionReason::Overflow,
            will_retry: false,
        },
        ExtensionEvent::SessionCompactFailed {
            reason: CompactionReason::Threshold,
            error_message: Some("failed".to_string()),
            aborted: false,
            will_retry: false,
            from_extension: false,
        },
        ExtensionEvent::SessionShutdown {
            reason: SessionShutdownReason::Quit,
            target_session_file: None,
        },
        ExtensionEvent::SessionBeforeTree {
            preparation: tree_preparation(),
            signal: None,
        },
        ExtensionEvent::SessionTree {
            new_leaf_id: Some("e-3".to_string()),
            old_leaf_id: None,
            summary_entry: None,
            from_extension: Some(false),
        },
        ExtensionEvent::Context {
            messages: vec![assistant_message()],
        },
        ExtensionEvent::BeforeProviderRequest {
            payload: json!({ "model": "gpt-test" }),
        },
        ExtensionEvent::BeforeProviderHeaders {
            headers: ProviderHeaders::new(),
        },
    ]
}

#[expect(
    clippy::too_many_lines,
    reason = "the sample carries every variant's payload once; splitting it hides the table shape"
)]
fn more_sample_events() -> Vec<ExtensionEvent> {
    vec![
        ExtensionEvent::AfterProviderResponse {
            status: Some(200),
            headers: ProviderHeaders::new(),
        },
        ExtensionEvent::BeforeAgentStart {
            prompt: "do it".to_string(),
            images: None,
            system_prompt: "sp".to_string(),
            system_prompt_options: BuildSystemPromptOptions::default(),
            signal: None,
        },
        ExtensionEvent::AgentStart,
        ExtensionEvent::AgentEnd {
            messages: vec![assistant_message()],
        },
        ExtensionEvent::AgentSettled,
        ExtensionEvent::TurnStart {
            turn_index: 0,
            timestamp: 1_000,
        },
        ExtensionEvent::TurnEnd {
            turn_index: 0,
            message: assistant_message(),
            tool_results: Vec::new(),
        },
        ExtensionEvent::MessageStart {
            message: assistant_message(),
        },
        ExtensionEvent::MessageUpdate {
            message: assistant_message(),
            assistant_message_event: AssistantMessageEvent::Start {
                partial: pi_ai::types::AssistantMessage {
                    content: Vec::new(),
                    api: Api::from("openai"),
                    provider: ProviderId::from("openai"),
                    model: "gpt-test".to_string(),
                    response_model: None,
                    response_id: None,
                    provider_thinking_level: None,
                    diagnostics: None,
                    usage: Usage::default(),
                    stop_reason: StopReason::Pending,
                    deferred: None,
                    error_message: None,
                    raw_stop_reason: None,
                    end_turn: None,
                    timestamp: 0,
                },
            },
        },
        ExtensionEvent::MessageEnd {
            message: assistant_message(),
        },
        ExtensionEvent::ToolExecutionStart {
            tool_call_id: "call-1".to_string(),
            tool_name: "read".to_string(),
            args: json!({ "path": "/x" }),
        },
        ExtensionEvent::ToolExecutionUpdate {
            tool_call_id: "call-1".to_string(),
            tool_name: "read".to_string(),
            args: json!({ "path": "/x" }),
            partial_result: tool_result(),
        },
        ExtensionEvent::ToolExecutionEnd {
            tool_call_id: "call-1".to_string(),
            tool_name: "read".to_string(),
            result: tool_result(),
            is_error: false,
        },
        ExtensionEvent::ModelSelect {
            model: model(),
            previous_model: None,
            source: ModelSelectSource::Set,
        },
        ExtensionEvent::ThinkingLevelSelect {
            level: ThinkingLevel::Off,
            previous_level: ThinkingLevel::Low,
        },
        ExtensionEvent::UserBash {
            command: "ls".to_string(),
        },
        ExtensionEvent::Input {
            text: "hi".to_string(),
            images: None,
            source: InputSource::Interactive,
            streaming_behavior: Some(StreamingBehavior::FollowUp),
        },
        ExtensionEvent::ToolCall {
            tool_name: "read".to_string(),
            tool_call_id: "call-1".to_string(),
            input: json!({ "path": "/x" }),
        },
        ExtensionEvent::ToolResult {
            tool_name: "read".to_string(),
            tool_call_id: "call-1".to_string(),
            input: json!({ "path": "/x" }),
            content: Vec::new(),
            details: json!({ "ok": true }),
            is_error: false,
            usage: None,
        },
    ]
}

#[test]
fn channels_match_the_wire_discriminators_in_order() {
    let expected: &[&str] = &[
        "project_trust",
        "resources_discover",
        "session_start",
        "session_info_changed",
        "session_before_switch",
        "session_before_fork",
        "session_before_compact",
        "session_compact",
        "session_compact_failed",
        "session_shutdown",
        "session_before_tree",
        "session_tree",
        "context",
        "before_provider_request",
        "before_provider_headers",
        "after_provider_response",
        "before_agent_start",
        "agent_start",
        "agent_end",
        "agent_settled",
        "turn_start",
        "turn_end",
        "message_start",
        "message_update",
        "message_end",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
        "model_select",
        "thinking_level_select",
        "user_bash",
        "input",
        "tool_call",
        "tool_result",
    ];
    let all = [sample_events(), more_sample_events()].concat();
    for (event, channel) in std::iter::zip(all, expected) {
        assert_eq!(event.channel(), *channel);
    }
}

#[test]
fn events_clone_and_debug_every_variant() {
    let all = [sample_events(), more_sample_events()].concat();
    for event in all {
        let clone = event.clone();
        let rendered = format!("{clone:?}");
        assert!(!rendered.is_empty());
    }
}

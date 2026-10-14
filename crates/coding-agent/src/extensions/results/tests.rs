use super::*;
use serde_json::json;

use crate::extensions::reasons::CustomMessagePayload;
use crate::extensions::reasons::text_block;

fn all_session_before_results() -> Vec<SessionBeforeResult> {
    vec![
        SessionBeforeResult::Switch { cancel: true },
        SessionBeforeResult::Fork {
            cancel: false,
            skip_conversation_restore: true,
        },
        SessionBeforeResult::Compact {
            cancel: true,
            compaction: Some(CompactionResult {
                summary: "s".to_string(),
                first_kept_entry_id: "e-1".to_string(),
                tokens_before: 100,
                estimated_tokens_after: Some(50),
                usage: Usage::default(),
                details: None,
            }),
        },
        SessionBeforeResult::Tree {
            cancel: false,
            summary: Some(ExtensionBranchSummary {
                summary: "s".to_string(),
                details: Some(json!({})),
                usage: None,
            }),
            custom_instructions: Some("ci".to_string()),
            replace_instructions: Some(true),
            label: Some("l".to_string()),
        },
        SessionBeforeResult::None,
    ]
}

fn all_handler_results() -> Vec<HandlerResult> {
    let custom_message = CustomMessagePayload {
        custom_type: "t".to_string(),
        content: vec![text_block("text")],
        display: true,
        details: None,
    };
    vec![
        HandlerResult::SessionBefore(SessionBeforeResult::None),
        HandlerResult::MessageEnd(MessageEndResult {
            message: Some(AgentMessage::Standard(pi_ai::types::Message::ToolResult(
                pi_ai::types::ToolResultMessage {
                    tool_call_id: "call-1".to_string(),
                    tool_name: "read".to_string(),
                    content: Vec::new(),
                    details: None,
                    usage: None,
                    added_tool_names: None,
                    is_error: false,
                    timestamp: 0,
                },
            ))),
        }),
        HandlerResult::ToolCall(ToolCallResult {
            block: true,
            reason: Some("bad".to_string()),
            terminate: false,
        }),
        HandlerResult::ToolResult(ToolResultResult {
            content: Some(Vec::new()),
            details: Some(json!({})),
            is_error: Some(true),
            usage: Some(Usage::default()),
        }),
        HandlerResult::Context(ContextResult {
            messages: Some(Vec::new()),
        }),
        HandlerResult::BeforeProviderRequest(json!({})),
        HandlerResult::BeforeProviderHeaders(ProviderHeaders::new()),
        HandlerResult::BeforeAgentStart(BeforeAgentStartResult {
            message: Some(custom_message),
            system_prompt: Some("sp".to_string()),
        }),
        HandlerResult::ResourcesDiscover(ResourcesDiscoverResult {
            skill_paths: vec!["skills".to_string()],
            prompt_paths: vec!["prompts".to_string()],
            theme_paths: Vec::new(),
        }),
        HandlerResult::Input(InputResult::Handled),
        HandlerResult::UserBash(UserBashResult { handled: true }),
        HandlerResult::ProjectTrust(ProjectTrustResult {
            trusted: Some(ProjectTrustDecision::No),
        }),
        HandlerResult::None,
    ]
}

#[test]
fn none_is_the_default_where_default_derives() {
    assert!(matches!(
        SessionBeforeResult::default(),
        SessionBeforeResult::None
    ));
    assert!(matches!(HandlerResult::default(), HandlerResult::None));
    assert!(matches!(
        MessageEndResult::default(),
        MessageEndResult { message: None }
    ));
    assert!(matches!(InputResult::Continue, InputResult::Continue));
    assert_eq!(ProjectTrustResult::default().trusted, None);
    assert!(matches!(
        ProjectTrustDecision::Undecided,
        ProjectTrustDecision::Undecided
    ));
}

#[test]
fn session_before_every_variant_debugs() {
    for result in all_session_before_results() {
        assert!(!format!("{result:?}").is_empty());
    }
}

#[test]
fn handler_result_every_variant_debugs() {
    for result in all_handler_results() {
        assert!(!format!("{result:?}").is_empty());
    }
    for result in all_session_before_results() {
        assert!(!format!("{result:?}").is_empty());
    }
}

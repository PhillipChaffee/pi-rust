#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    reason = "the tests pin outcomes; an unexpected result panics the test by design, the \
              fixture tables are the wire's complete index, and the mapper calls unwrap by design too"
)]

use std::collections::{BTreeMap, HashMap};

use serde_json::json;

use pi_agent_core::harness::context::{
    AbortController, AbortSignal, background_context, with_cancel,
};
use pi_agent_core::types::{
    AgentMessage, AgentToolResult, CustomAgentMessage, ThinkingLevel, ToolExecutionMode,
};
use pi_ai::types::{
    Api, AssistantMessage, ImageContent, Model, ModelCost, ProviderId, StopReason, TextContent,
    ToolResultBlock, ToolResultMessage, UserBlock, UserContent, UserMessage,
};

use crate::compaction::{CompactionPreparation, CompactionResult};
use crate::extensions::events::ExtensionEvent;
use crate::extensions::reasons::{
    CompactionReason, ForkPosition, InputSource, ModelSelectSource, SessionShutdownReason,
    SessionStartReason, StreamingBehavior, TreePreparation,
};
use crate::extensions::results::{
    BeforeAgentStartResult, ContextResult, ExtensionBranchSummary, HandlerResult, InputResult,
    MessageEndResult, ProjectTrustDecision, ProjectTrustResult, SessionBeforeResult,
    ToolCallResult, ToolResultResult, UserBashResult,
};
use crate::session_manager::entries::{
    BranchSummaryEntry, CompactionEntry, MessageEntry, SessionEntry, SessionEntryBase,
};
use crate::skills::Skill;
use crate::source_info::{SourceInfo, SourceOrigin, SourceScope};
use crate::system_prompt::BuildSystemPromptOptions;

use crate::extensions::host::{
    CommandDeclaration, ContextSnapshot, EVENT_CHANNELS, FlagDeclaration, RESULT_CHANNELS,
    ToolDeclaration, command_declaration_from_wire, command_to_wire, context_snapshot_from_wire,
    context_snapshot_to_wire, event_from_wire, event_to_wire, flag_declaration_from_wire,
    flag_to_wire, provider_from_wire, provider_to_wire, result_from_wire, result_to_wire,
    tool_declaration_from_wire, tool_definition_to_wire,
};

// __SAMPLES__

fn sample_model() -> Model {
    Model {
        id: "claude-sonnet-4-20250514".to_string(),
        name: "Claude Sonnet 4".to_string(),
        api: Api("anthropic-messages".to_string()),
        provider: ProviderId("anthropic".to_string()),
        base_url: "https://api.anthropic.com".to_string(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: ModelCost::default(),
        context_window: 200_000,
        max_tokens: 16_384,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn sample_assistant_message() -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: Api("anthropic-messages".to_string()),
        provider: ProviderId("anthropic".to_string()),
        model: "claude-sonnet-4-20250514".to_string(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: sample_usage(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 7,
    }
}

fn sample_usage() -> pi_ai::types::Usage {
    pi_ai::types::Usage {
        input: 3,
        output: 4,
        cache_read: 5,
        cache_write: 6,
        ..pi_ai::types::Usage::default()
    }
}

fn user_message(text: &str) -> AgentMessage {
    AgentMessage::Standard(pi_ai::types::Message::User(UserMessage {
        content: UserContent::Text(text.to_string()),
        timestamp: 7,
    }))
}

fn custom_message() -> AgentMessage {
    let mut data = serde_json::Map::new();
    data.insert("text".to_string(), json!("banner"));
    AgentMessage::Custom(CustomAgentMessage {
        role: "app.banner".to_string(),
        timestamp: 7,
        data,
    })
}

fn sample_image() -> ImageContent {
    ImageContent {
        data: "AAAA".to_string(),
        mime_type: "image/png".to_string(),
    }
}

fn sample_tool_block() -> ToolResultBlock {
    ToolResultBlock::Text(TextContent {
        text: "done".to_string(),
        text_signature: None,
    })
}

fn sample_agent_tool_result() -> AgentToolResult {
    AgentToolResult {
        content: Vec::new(),
        details: json!({}),
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

fn sample_session_entry() -> SessionEntry {
    SessionEntry::Message(MessageEntry {
        base: SessionEntryBase {
            id: Some("e-01".to_string()),
            parent_id: None,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            extras: serde_json::Map::new(),
        },
        message: Some(user_message("hello")),
        extras: serde_json::Map::new(),
    })
}

fn sample_compaction_entry() -> CompactionEntry {
    CompactionEntry {
        base: SessionEntryBase {
            id: Some("e-10".to_string()),
            parent_id: Some("e-09".to_string()),
            timestamp: "2025-01-01T00:01:00Z".to_string(),
            extras: serde_json::Map::new(),
        },
        summary: "kept the hooks".to_string(),
        first_kept_entry_id: Some("e-08".to_string()),
        tokens_before: 120_000,
        details: None,
        usage: Some(sample_usage()),
        from_hook: None,
        extras: serde_json::Map::new(),
    }
}

fn sample_branch_summary_entry() -> BranchSummaryEntry {
    BranchSummaryEntry {
        base: SessionEntryBase {
            id: Some("e-20".to_string()),
            parent_id: Some("e-19".to_string()),
            timestamp: "2025-01-01T00:02:00Z".to_string(),
            extras: serde_json::Map::new(),
        },
        from_id: "e-11".to_string(),
        summary: "left branch summary".to_string(),
        details: None,
        usage: Some(sample_usage()),
        from_hook: None,
        extras: serde_json::Map::new(),
    }
}

fn sample_preparation() -> CompactionPreparation {
    let mut file_ops = crate::compaction::FileOperations::default();
    file_ops.read.insert("src/mod.rs".to_string());
    file_ops.edited.insert("src/lib.rs".to_string());
    CompactionPreparation {
        first_kept_entry_id: "e-08".to_string(),
        messages_to_summarize: vec![user_message("one"), user_message("two")],
        turn_prefix_messages: Vec::new(),
        is_split_turn: true,
        tokens_before: 120_000,
        previous_summary: Some("prior".to_string()),
        file_ops,
        settings: crate::settings_manager::CompactionSettings {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
        },
    }
}

fn sample_compaction_result() -> CompactionResult {
    CompactionResult {
        summary: "kept the hooks".to_string(),
        first_kept_entry_id: "e-08".to_string(),
        tokens_before: 120_000,
        estimated_tokens_after: Some(4_000),
        usage: sample_usage(),
        details: Some(json!({"keep": true})),
    }
}

fn sample_tree_preparation() -> TreePreparation {
    TreePreparation {
        target_id: "e-30".to_string(),
        old_leaf_id: Some("e-19".to_string()),
        common_ancestor_id: Some("e-05".to_string()),
        entries_to_summarize: vec![sample_session_entry()],
        user_wants_summary: false,
        custom_instructions: Some("summarize briefly".to_string()),
        replace_instructions: Some(true),
        label: Some("checkpoint".to_string()),
    }
}

fn sample_source_info() -> SourceInfo {
    SourceInfo {
        path: "/tmp/demo/.pi/skills/demo/SKILL.md".to_string(),
        source: "local".to_string(),
        scope: SourceScope::Project,
        origin: SourceOrigin::TopLevel,
        base_dir: Some("/tmp/demo/.pi".to_string()),
    }
}

fn sample_skill() -> Skill {
    Skill {
        name: "demo".to_string(),
        description: "a demo skill".to_string(),
        file_path: "/tmp/demo/.pi/skills/demo/SKILL.md".to_string(),
        base_dir: "/tmp/demo/.pi/skills/demo".to_string(),
        source_info: sample_source_info(),
        disable_model_invocation: false,
    }
}

fn sample_system_prompt_options() -> BuildSystemPromptOptions {
    let mut tool_snippets = HashMap::new();
    tool_snippets.insert("read".to_string(), "runs reads".to_string());
    BuildSystemPromptOptions {
        custom_prompt: Some("custom body".to_string()),
        selected_tools: Some(vec!["read".to_string()]),
        tool_snippets: Some(tool_snippets),
        prompt_guidelines: Some(vec!["be brief".to_string()]),
        append_system_prompt: Some("appendix".to_string()),
        cwd: "/tmp/demo".to_string(),
        context_files: Some(vec![crate::system_prompt::ContextFile {
            path: "/tmp/demo/AGENTS.md".to_string(),
            content: "context".to_string(),
        }]),
        skills: Some(vec![sample_skill()]),
    }
}

// __FIXTURES__

/// One representative event per variant, keyed by its channel. New events
/// extend this table when the compiler forces the mapping arm; the channel
/// table test below notices a missing fixture.
fn fixtures(signal: &AbortController) -> Vec<(&'static str, ExtensionEvent)> {
    let run_signal = signal.signal().clone();
    vec![
        (
            "project_trust",
            ExtensionEvent::ProjectTrust {
                cwd: "/tmp/demo".to_string(),
            },
        ),
        (
            "resources_discover",
            ExtensionEvent::ResourcesDiscover {
                cwd: "/tmp/demo".to_string(),
                reason: SessionStartReason::Reload,
            },
        ),
        (
            "session_start",
            ExtensionEvent::SessionStart {
                reason: SessionStartReason::New,
                previous_session_file: Some("/tmp/demo/.pi/agent/sessions/old.jsonl".to_string()),
            },
        ),
        (
            "session_info_changed",
            ExtensionEvent::SessionInfoChanged {
                name: Some("demo session".to_string()),
            },
        ),
        (
            "session_before_switch",
            ExtensionEvent::SessionBeforeSwitch {
                reason: SessionShutdownReason::Resume,
                target_session_file: Some("/tmp/demo/.pi/agent/sessions/next.jsonl".to_string()),
            },
        ),
        (
            "session_before_fork",
            ExtensionEvent::SessionBeforeFork {
                entry_id: "e-04".to_string(),
                position: ForkPosition::Before,
            },
        ),
        (
            "session_before_compact",
            ExtensionEvent::SessionBeforeCompact {
                preparation: sample_preparation(),
                branch_entries: vec![sample_session_entry()],
                custom_instructions: Some("keep hooks".to_string()),
                reason: CompactionReason::Threshold,
                will_retry: true,
                signal: Some(run_signal.clone()),
            },
        ),
        (
            "session_compact",
            ExtensionEvent::SessionCompact {
                compaction_entry: sample_compaction_entry(),
                from_extension: false,
                reason: CompactionReason::Manual,
                will_retry: false,
            },
        ),
        (
            "session_compact_failed",
            ExtensionEvent::SessionCompactFailed {
                reason: CompactionReason::Overflow,
                error_message: Some("the summaries stop".to_string()),
                aborted: true,
                will_retry: true,
                from_extension: true,
            },
        ),
        (
            "session_shutdown",
            ExtensionEvent::SessionShutdown {
                reason: SessionShutdownReason::Reload,
                target_session_file: None,
            },
        ),
        (
            "session_before_tree",
            ExtensionEvent::SessionBeforeTree {
                preparation: sample_tree_preparation(),
                signal: Some(run_signal.clone()),
            },
        ),
        (
            "session_tree",
            ExtensionEvent::SessionTree {
                new_leaf_id: Some("e-30".to_string()),
                old_leaf_id: Some("e-19".to_string()),
                summary_entry: Some(sample_branch_summary_entry()),
                from_extension: Some(true),
            },
        ),
        (
            "context",
            ExtensionEvent::Context {
                messages: vec![user_message("hello"), custom_message()],
            },
        ),
        (
            "before_provider_request",
            ExtensionEvent::BeforeProviderRequest {
                payload: json!({"model": "claude-sonnet-4-20250514", "stream": true}),
            },
        ),
        (
            "before_provider_headers",
            ExtensionEvent::BeforeProviderHeaders {
                headers: BTreeMap::from([
                    ("traceparent".to_string(), Some("00-abc".to_string())),
                    ("x-pi-drop".to_string(), None),
                ]),
            },
        ),
        (
            "after_provider_response",
            ExtensionEvent::AfterProviderResponse {
                status: Some(200),
                headers: BTreeMap::from([(
                    "x-ratelimit-remaining".to_string(),
                    Some("12".to_string()),
                )]),
            },
        ),
        (
            "before_agent_start",
            ExtensionEvent::BeforeAgentStart {
                prompt: "fix the bug".to_string(),
                images: Some(vec![sample_image()]),
                system_prompt: "system".to_string(),
                system_prompt_options: sample_system_prompt_options(),
                signal: Some(run_signal),
            },
        ),
        ("agent_start", ExtensionEvent::AgentStart),
        (
            "agent_end",
            ExtensionEvent::AgentEnd {
                messages: vec![user_message("hello"), custom_message()],
            },
        ),
        ("agent_settled", ExtensionEvent::AgentSettled),
        (
            "turn_start",
            ExtensionEvent::TurnStart {
                turn_index: 2,
                timestamp: 1_000,
            },
        ),
        (
            "turn_end",
            ExtensionEvent::TurnEnd {
                turn_index: 2,
                message: AgentMessage::Standard(pi_ai::types::Message::Assistant(
                    sample_assistant_message(),
                )),
                tool_results: vec![ToolResultMessage {
                    tool_call_id: "call_1".to_string(),
                    tool_name: "read".to_string(),
                    content: vec![sample_tool_block()],
                    details: None,
                    usage: None,
                    added_tool_names: None,
                    is_error: false,
                    timestamp: 7,
                }],
            },
        ),
        (
            "message_start",
            ExtensionEvent::MessageStart {
                message: user_message("hello"),
            },
        ),
        (
            "message_update",
            ExtensionEvent::MessageUpdate {
                message: AgentMessage::Standard(pi_ai::types::Message::Assistant(
                    sample_assistant_message(),
                )),
                assistant_message_event: pi_ai::types::AssistantMessageEvent::Start {
                    partial: sample_assistant_message(),
                },
            },
        ),
        (
            "message_end",
            ExtensionEvent::MessageEnd {
                message: AgentMessage::Standard(pi_ai::types::Message::Assistant(
                    sample_assistant_message(),
                )),
            },
        ),
        (
            "tool_execution_start",
            ExtensionEvent::ToolExecutionStart {
                tool_call_id: "call_1".to_string(),
                tool_name: "read".to_string(),
                args: json!({"path": "README.md"}),
            },
        ),
        (
            "tool_execution_update",
            ExtensionEvent::ToolExecutionUpdate {
                tool_call_id: "call_1".to_string(),
                tool_name: "read".to_string(),
                args: json!({"path": "README.md"}),
                partial_result: sample_agent_tool_result(),
            },
        ),
        (
            "tool_execution_end",
            ExtensionEvent::ToolExecutionEnd {
                tool_call_id: "call_1".to_string(),
                tool_name: "read".to_string(),
                result: sample_agent_tool_result(),
                is_error: true,
            },
        ),
        (
            "model_select",
            ExtensionEvent::ModelSelect {
                model: sample_model(),
                previous_model: Some(sample_model()),
                source: ModelSelectSource::Cycle,
            },
        ),
        (
            "thinking_level_select",
            ExtensionEvent::ThinkingLevelSelect {
                level: ThinkingLevel::High,
                previous_level: ThinkingLevel::Off,
            },
        ),
        (
            "user_bash",
            ExtensionEvent::UserBash {
                command: "ls -la".to_string(),
            },
        ),
        (
            "input",
            ExtensionEvent::Input {
                text: "fix it".to_string(),
                images: Some(vec![sample_image()]),
                source: InputSource::Interactive,
                streaming_behavior: Some(StreamingBehavior::Steer),
            },
        ),
        (
            "tool_call",
            ExtensionEvent::ToolCall {
                tool_name: "read".to_string(),
                tool_call_id: "call_1".to_string(),
                input: json!({"path": "README.md"}),
            },
        ),
        (
            "tool_result",
            ExtensionEvent::ToolResult {
                tool_name: "read".to_string(),
                tool_call_id: "call_1".to_string(),
                input: json!({"path": "README.md"}),
                content: vec![sample_tool_block()],
                details: json!({"lines": 3}),
                is_error: false,
                usage: None,
            },
        ),
    ]
}

/// Allocates wire signal ids in registration order, the dispatch's stand-in.
fn signal_allocator() -> impl FnMut(&AbortSignal) -> u64 {
    let mut next = 1u64;
    move |_signal: &AbortSignal| {
        let id = next;
        next += 1;
        id
    }
}

/// Resolves the one id the fixtures can produce.
fn signal_resolver(controller: &AbortController) -> impl FnMut(u64) -> Option<AbortSignal> + '_ {
    move |id: u64| (id == 1).then(|| controller.signal().clone())
}
#[test]
fn every_event_channel_maps_and_round_trips_through_the_wire() {
    let (_, controller) = with_cancel(&background_context());
    let events = fixtures(&controller);
    for (channel, event) in &events {
        assert_eq!(*channel, event.channel(), "the {channel} fixture drifts");
        let wire = event_to_wire(event, &mut signal_allocator())
            .unwrap_or_else(|error| panic!("{channel} does not map: {error}"));
        assert_eq!(
            wire["type"].as_str(),
            Some(*channel),
            "{channel}'s payload carries its discriminator"
        );
        let decoded = event_from_wire(channel, &wire, &mut signal_resolver(&controller))
            .unwrap_or_else(|error| panic!("{channel} does not parse: {error}"));
        let rewire = event_to_wire(&decoded, &mut signal_allocator())
            .unwrap_or_else(|error| panic!("{channel} does not re-map: {error}"));
        assert_eq!(wire, rewire, "{channel} does not round-trip");
    }
}

#[test]
fn the_channel_table_covers_the_fixture_table() {
    let (_, controller) = with_cancel(&background_context());
    let fixtures = fixtures(&controller);
    assert_eq!(
        EVENT_CHANNELS.len(),
        fixtures.len(),
        "the channel table and the fixture table disagree in size"
    );
    for (channel, _) in &fixtures {
        assert!(
            EVENT_CHANNELS.contains(channel),
            "fixture channel {channel} is missing from the channel table"
        );
    }
}

#[test]
fn event_payloads_reject_shape_drift() {
    let (_, controller) = with_cancel(&background_context());
    // A mismatched discriminator refuses to bind.
    let mismatch = json!({"type": "agent_start", "cwd": "/tmp"});
    assert!(
        event_from_wire(
            "project_trust",
            &mismatch,
            &mut signal_resolver(&controller)
        )
        .is_err()
    );
    // Missing discriminator.
    let typeless = json!({"cwd": "/tmp"});
    assert!(
        event_from_wire(
            "project_trust",
            &typeless,
            &mut signal_resolver(&controller)
        )
        .is_err()
    );
    // Unknown channels fail closed.
    let known = json!({"type": "project_trust", "cwd": "/tmp"});
    assert!(event_from_wire("no Such", &known, &mut signal_resolver(&controller)).is_err());
    // A non-string field type rejects.
    let cwd_number = json!({"type": "project_trust", "cwd": 7});
    assert!(
        event_from_wire(
            "project_trust",
            &cwd_number,
            &mut signal_resolver(&controller)
        )
        .is_err()
    );
    // resources_discover narrows the reason vocabulary to its channel.
    let wide = json!({"type": "resources_discover", "cwd": "/tmp", "reason": "new"});
    assert!(
        event_from_wire(
            "resources_discover",
            &wide,
            &mut signal_resolver(&controller)
        )
        .is_err()
    );
    // A signal id without a name in the config cannot resolve.
    let wired = event_to_wire(
        &ExtensionEvent::SessionBeforeTree {
            preparation: sample_tree_preparation(),
            signal: Some(controller.signal().clone()),
        },
        &mut signal_allocator(),
    )
    .unwrap();
    let dead: Option<AbortSignal> = None;
    let mut no_names = |id: u64| id.checked_mul(0).and_then(|_| dead.clone());
    assert!(
        event_from_wire("session_before_tree", &wired, &mut no_names).is_err(),
        "an unnamed signal id must fail the frame"
    );
}

#[test]
fn optional_event_fields_stay_absent_on_the_wire_when_unset() {
    let absent = event_to_wire(
        &ExtensionEvent::SessionInfoChanged { name: None },
        &mut signal_allocator(),
    )
    .unwrap();
    assert_eq!(absent, json!({"type": "session_info_changed"}));

    let nullable = event_to_wire(
        &ExtensionEvent::SessionTree {
            new_leaf_id: None,
            old_leaf_id: None,
            summary_entry: None,
            from_extension: None,
        },
        &mut signal_allocator(),
    )
    .unwrap();
    assert_eq!(
        nullable,
        json!({"type": "session_tree", "newLeafId": null, "oldLeafId": null})
    );

    let status = event_to_wire(
        &ExtensionEvent::AfterProviderResponse {
            status: None,
            headers: BTreeMap::new(),
        },
        &mut signal_allocator(),
    )
    .unwrap();
    assert_eq!(
        status,
        json!({"type": "after_provider_response", "status": null, "headers": {}})
    );

    let bash = event_to_wire(
        &ExtensionEvent::UserBash {
            command: "ls".to_string(),
        },
        &mut signal_allocator(),
    )
    .unwrap();
    assert_eq!(bash, json!({"type": "user_bash", "command": "ls"}));

    assert_eq!(
        event_to_wire(&ExtensionEvent::AgentSettled, &mut signal_allocator()).unwrap(),
        json!({"type": "agent_settled"})
    );
}

/// One result per member of upstream's result-interfaces list, keyed by its
/// channel. New results extend this table when the compiler forces the
/// mapping arm; the result-table test below notices a missing fixture.
fn result_fixtures() -> Vec<(&'static str, HandlerResult)> {
    vec![
        (
            "project_trust",
            HandlerResult::ProjectTrust(ProjectTrustResult {
                trusted: Some(ProjectTrustDecision::Yes),
            }),
        ),
        (
            "resources_discover",
            HandlerResult::ResourcesDiscover(crate::extensions::results::ResourcesDiscoverResult {
                skill_paths: vec!["/tmp/demo/.pi/skills/demo".to_string()],
                prompt_paths: Vec::new(),
                theme_paths: vec!["/tmp/demo/.pi/themes/idea.json".to_string()],
            }),
        ),
        (
            "session_before_switch",
            HandlerResult::SessionBefore(SessionBeforeResult::Switch { cancel: true }),
        ),
        (
            "session_before_fork",
            HandlerResult::SessionBefore(SessionBeforeResult::Fork {
                cancel: false,
                skip_conversation_restore: true,
            }),
        ),
        (
            "session_before_compact",
            HandlerResult::SessionBefore(SessionBeforeResult::Compact {
                cancel: false,
                compaction: Some(sample_compaction_result()),
            }),
        ),
        (
            "session_before_tree",
            HandlerResult::SessionBefore(SessionBeforeResult::Tree {
                cancel: true,
                summary: Some(ExtensionBranchSummary {
                    summary: "the left branch".to_string(),
                    details: Some(json!({"entries": 4})),
                    usage: Some(sample_usage()),
                }),
                custom_instructions: Some("brief".to_string()),
                replace_instructions: Some(true),
                label: Some("checkpoint".to_string()),
            }),
        ),
        (
            "context",
            HandlerResult::Context(ContextResult {
                messages: Some(vec![user_message("replaced")]),
            }),
        ),
        (
            "before_provider_request",
            HandlerResult::BeforeProviderRequest(json!({"model": "replaced"})),
        ),
        (
            "before_provider_headers",
            HandlerResult::BeforeProviderHeaders(BTreeMap::from([(
                "traceparent".to_string(),
                Some("00-def".to_string()),
            )])),
        ),
        (
            "before_agent_start",
            HandlerResult::BeforeAgentStart(BeforeAgentStartResult {
                message: Some(crate::extensions::reasons::CustomMessagePayload {
                    custom_type: "app.note".to_string(),
                    content: vec![UserBlock::Text(TextContent {
                        text: "note".to_string(),
                        text_signature: None,
                    })],
                    display: true,
                    details: None,
                }),
                system_prompt: Some("replaced prompt".to_string()),
            }),
        ),
        (
            "input",
            HandlerResult::Input(InputResult::Transform {
                text: "expanded".to_string(),
                images: None,
            }),
        ),
        (
            "user_bash",
            HandlerResult::UserBash(UserBashResult { handled: true }),
        ),
        (
            "tool_call",
            HandlerResult::ToolCall(ToolCallResult {
                block: true,
                reason: Some("not while streaming".to_string()),
                terminate: true,
            }),
        ),
        (
            "tool_result",
            HandlerResult::ToolResult(ToolResultResult {
                content: Some(vec![sample_tool_block()]),
                details: Some(json!({"lines": 9})),
                is_error: Some(false),
                usage: Some(sample_usage()),
            }),
        ),
        (
            "message_end",
            HandlerResult::MessageEnd(MessageEndResult {
                message: Some(user_message("final")),
            }),
        ),
    ]
}

#[test]
fn every_result_channel_maps_and_round_trips_through_the_wire() {
    for (channel, result) in &result_fixtures() {
        let wire = result_to_wire(result)
            .unwrap_or_else(|error| panic!("{channel} result does not map: {error}"));
        let decoded = result_from_wire(channel, &wire)
            .unwrap_or_else(|error| panic!("{channel} result does not parse: {error}"));
        let rewire = result_to_wire(&decoded)
            .unwrap_or_else(|error| panic!("{channel} result does not re-map: {error}"));
        assert_eq!(wire, rewire, "the {channel} result does not round-trip");
    }
}

#[test]
fn the_result_table_covers_the_result_channels() {
    let fixtures = result_fixtures();
    assert_eq!(
        RESULT_CHANNELS.len(),
        fixtures.len(),
        "the result table and the fixture table disagree in size"
    );
    for (channel, _) in &fixtures {
        assert!(
            RESULT_CHANNELS.contains(channel),
            "fixture channel {channel} is missing from the result table"
        );
    }
}

#[test]
fn handler_absence_is_the_wires_null_and_results_default_to_empty_shapes() {
    for channel in RESULT_CHANNELS {
        let wire = result_to_wire(&HandlerResult::None).unwrap();
        assert!(wire.is_null(), "the {channel} absence must ride null");
        let decoded = result_from_wire(channel, &wire).unwrap();
        assert!(
            matches!(decoded, HandlerResult::None),
            "{channel} reads back absence"
        );
    }
}

#[test]
fn result_shapes_pin_upstream_field_names() {
    let input = result_to_wire(&HandlerResult::Input(InputResult::Continue)).unwrap();
    assert_eq!(input, json!({"action": "continue"}));

    let transform = result_to_wire(&HandlerResult::Input(InputResult::Transform {
        text: "expanded".to_string(),
        images: None,
    }))
    .unwrap();
    assert_eq!(
        transform,
        json!({"action": "transform", "text": "expanded"})
    );

    let undecided = result_to_wire(&HandlerResult::ProjectTrust(ProjectTrustResult {
        trusted: None,
    }))
    .unwrap();
    assert_eq!(undecided, json!({}));

    let undecided_decision = result_to_wire(&HandlerResult::ProjectTrust(ProjectTrustResult {
        trusted: Some(ProjectTrustDecision::Undecided),
    }))
    .unwrap();
    assert_eq!(undecided_decision, json!({"trusted": "undecided"}));

    let tool_call = result_to_wire(&HandlerResult::ToolCall(ToolCallResult {
        block: false,
        reason: None,
        terminate: false,
    }))
    .unwrap();
    assert_eq!(tool_call, json!({"block": false, "terminate": false}));
}

#[test]
fn result_payloads_reject_shape_drift() {
    // A no-result channel refuses a non-null value: a confused extension
    // surfaces as a handler failure, never a silent protocol break.
    let stray = result_from_wire("agent_start", &json!({"unexpected": true}));
    assert!(stray.is_err(), "agent_start must not accept a result");
    assert!(result_from_wire("no-such-channel", &json!({})).is_err());

    let bad_decision = result_from_wire("project_trust", &json!({"trusted": "maybe"}))
        .expect_err("an unknown decision rejects");
    assert!(bad_decision.message().contains("maybe"));

    let bad_action = result_from_wire("input", &json!({"action": "sideline"}))
        .expect_err("an unknown action rejects");
    assert!(bad_action.message().contains("sideline"));

    let bad_flag_array = result_from_wire("resources_discover", &json!({"skillPaths": [1.0]}))
        .expect_err("a non-string path rejects");
    assert!(bad_flag_array.message().contains("skillPaths"));
}

#[test]
fn the_tool_declaration_covers_the_declarative_surface() {
    let definition = crate::extensions::ToolDefinition {
        name: "demo_tool".to_string(),
        label: "Demo tool".to_string(),
        description: "renders the wire".to_string(),
        prompt_snippet: Some("renders demo".to_string()),
        prompt_guidelines: Some(vec!["quote ids".to_string()]),
        parameters: json!({"type": "object"}),
        constrained_sampling: Some(pi_ai::types::ConstrainedSamplingSetting::Disabled(false)),
        render_shell: Some(crate::extensions::RenderShell::SelfRender),
        render_call: None,
        render_result: None,
        prepare_arguments: None,
        execution_mode: Some(ToolExecutionMode::Sequential),
        execute: std::sync::Arc::new(|_, _, _, _, _| {
            Box::pin(async { Ok(sample_agent_tool_result()) })
        }),
    };
    let wire = tool_definition_to_wire(&definition).unwrap();
    assert_eq!(
        wire,
        json!({
            "name": "demo_tool",
            "label": "Demo tool",
            "description": "renders the wire",
            "promptSnippet": "renders demo",
            "promptGuidelines": ["quote ids"],
            "parameters": {"type": "object"},
            "constrainedSampling": false,
            "renderShell": "self",
            "executionMode": "sequential",
        })
    );
    let declaration = ToolDeclaration {
        name: "demo_tool".to_string(),
        label: "Demo tool".to_string(),
        description: "renders the wire".to_string(),
        prompt_snippet: Some("renders demo".to_string()),
        prompt_guidelines: Some(vec!["quote ids".to_string()]),
        parameters: json!({"type": "object"}),
        constrained_sampling: Some(pi_ai::types::ConstrainedSamplingSetting::Disabled(false)),
        render_shell: Some(crate::extensions::RenderShell::SelfRender),
        execution_mode: Some(ToolExecutionMode::Sequential),
    };
    assert_eq!(
        tool_declaration_from_wire(&wire).unwrap(),
        declaration,
        "the declaration parses back without the executor"
    );
}

#[test]
fn command_and_flag_descriptors_port_1_1() {
    let command = CommandDeclaration {
        name: "demo".to_string(),
        description: Some("a demo command".to_string()),
    };
    let wire = command_to_wire(&command).unwrap();
    assert_eq!(
        wire,
        json!({"name": "demo", "description": "a demo command"})
    );
    assert_eq!(command_declaration_from_wire(&wire).unwrap(), command);

    let flag = FlagDeclaration {
        name: "demo-mode".to_string(),
        description: Some("the demo mode".to_string()),
        flag_type: crate::extensions::runtime::FlagType::String,
        default: Some(crate::extensions::runtime::FlagValue::Text(
            "auto".to_string(),
        )),
    };
    let flag_wire = flag_to_wire(&flag).unwrap();
    assert_eq!(
        flag_wire,
        json!({
            "name": "demo-mode",
            "description": "the demo mode",
            "type": "string",
            "default": "auto",
        })
    );
    assert_eq!(flag_declaration_from_wire(&flag_wire).unwrap(), flag);

    let boolean_default_string = json!({
        "name": "demo-mode",
        "type": "boolean",
        "default": "auto",
    });
    assert!(
        flag_declaration_from_wire(&boolean_default_string).is_err(),
        "a boolean flag's string default rejects"
    );
}

#[test]
fn the_provider_registration_carries_its_data_and_leaves_closures_host_side() {
    let config = crate::provider_composer::ProviderConfigInput {
        name: Some("Proxy Claude".to_string()),
        base_url: Some("https://proxy.example.com".to_string()),
        api_key: Some("$PROXY_API_KEY".to_string()),
        api: Some(Api("anthropic-messages".to_string())),
        stream_simple: None,
        headers: Some(BTreeMap::from([("x-proxy".to_string(), "1".to_string())])),
        auth_header: Some(true),
        oauth: None,
        models: Some(vec![crate::provider_composer::ProviderModelInput {
            id: "claude-sonnet-4-20250514".to_string(),
            name: "Claude 4 Sonnet (proxy)".to_string(),
            api: None,
            base_url: None,
            reasoning: false,
            thinking_level_map: None,
            input: vec![pi_ai::types::Modality::Text],
            cost: ModelCost::default(),
            context_window: 200_000,
            max_tokens: 16_384,
            sampling_params: None,
            headers: None,
            compat: None,
        }]),
        refresh_models: None,
    };
    let wire = provider_to_wire("my-proxy", &config).unwrap();
    let (name, parsed) = provider_from_wire(&wire).unwrap();
    assert_eq!(name, "my-proxy");
    assert_eq!(parsed.name.as_deref(), config.name.as_deref());
    assert_eq!(parsed.base_url, config.base_url);
    assert_eq!(parsed.api_key, config.api_key);
    assert_eq!(parsed.api, config.api);
    assert_eq!(parsed.auth_header, config.auth_header);
    assert!(
        parsed.stream_simple.is_none() && parsed.oauth.is_none() && parsed.refresh_models.is_none()
    );
    let models = parsed.models.as_ref().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "claude-sonnet-4-20250514");
    assert_eq!(models[0].cost, ModelCost::default());
    assert_eq!(
        wire["config"]["models"][0]["contextWindow"],
        json!(200_000),
        "the model fields pin upstream's camelCase"
    );
}

#[test]
fn the_context_snapshot_serializes_the_domain_context() {
    let cwd_only = ContextSnapshot::from_extension_context(&crate::extensions::CwdContext {
        cwd: "/tmp/demo".to_string(),
    });
    assert_eq!(
        cwd_only,
        ContextSnapshot {
            cwd: "/tmp/demo".to_string(),
            model: None,
            thinking_level: None,
            session_id: None,
            session_file: None,
        }
    );
    let wire = context_snapshot_to_wire(&cwd_only).unwrap();
    assert_eq!(wire, json!({"cwd": "/tmp/demo"}));
    assert_eq!(context_snapshot_from_wire(&wire).unwrap(), cwd_only);

    let full = ContextSnapshot {
        cwd: "/tmp/demo".to_string(),
        model: Some(sample_model()),
        thinking_level: Some(ThinkingLevel::High),
        session_id: Some("session-1".to_string()),
        session_file: Some("/tmp/demo/.pi/agent/sessions/a.jsonl".to_string()),
    };
    let full_wire = context_snapshot_to_wire(&full).unwrap();
    assert_eq!(
        full_wire,
        json!({
            "cwd": "/tmp/demo",
            "model": serde_json::to_value(sample_model()).unwrap(),
            "thinkingLevel": "high",
            "sessionId": "session-1",
            "sessionFile": "/tmp/demo/.pi/agent/sessions/a.jsonl",
        })
    );
    assert_eq!(context_snapshot_from_wire(&full_wire).unwrap(), full);
}

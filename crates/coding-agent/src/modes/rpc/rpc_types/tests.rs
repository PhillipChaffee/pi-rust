//! Boundary tests for the RPC wire types.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

use pi_agent_core::types::QueueMode;
use pi_agent_core::types::ThinkingLevel;
use serde_json::json;

use super::RpcCommand;
use super::RpcCommandSource;
use super::RpcExtensionUiRequest;
use super::RpcExtensionUiResponse;
use super::RpcNotifyType;
use super::RpcSessionState;
use super::RpcSlashCommand;
use super::RpcUiMethod;
use super::RpcWidgetPlacement;

fn parse(value: serde_json::Value) -> RpcCommand {
    serde_json::from_value(value).expect("the command parses")
}

#[test]
fn every_command_type_parses_and_names_itself() {
    let rows = [
        (json!({"type":"prompt","message":"hi"}), "prompt"),
        (json!({"type":"steer","message":"hi"}), "steer"),
        (json!({"type":"follow_up","message":"hi"}), "follow_up"),
        (json!({"type":"abort"}), "abort"),
        (json!({"type":"clear_queue"}), "clear_queue"),
        (json!({"type":"new_session"}), "new_session"),
        (json!({"type":"get_state"}), "get_state"),
        (
            json!({"type":"set_model","provider":"p","modelId":"m"}),
            "set_model",
        ),
        (json!({"type":"cycle_model"}), "cycle_model"),
        (
            json!({"type":"get_available_models"}),
            "get_available_models",
        ),
        (
            json!({"type":"set_thinking_level","level":"high"}),
            "set_thinking_level",
        ),
        (
            json!({"type":"cycle_thinking_level"}),
            "cycle_thinking_level",
        ),
        (
            json!({"type":"get_available_thinking_levels"}),
            "get_available_thinking_levels",
        ),
        (
            json!({"type":"set_steering_mode","mode":"one-at-a-time"}),
            "set_steering_mode",
        ),
        (
            json!({"type":"set_follow_up_mode","mode":"all"}),
            "set_follow_up_mode",
        ),
        (json!({"type":"compact"}), "compact"),
        (
            json!({"type":"set_auto_compaction","enabled":true}),
            "set_auto_compaction",
        ),
        (
            json!({"type":"set_auto_retry","enabled":false}),
            "set_auto_retry",
        ),
        (json!({"type":"abort_retry"}), "abort_retry"),
        (json!({"type":"bash","command":"ls"}), "bash"),
        (json!({"type":"abort_bash"}), "abort_bash"),
        (json!({"type":"get_session_stats"}), "get_session_stats"),
        (json!({"type":"export_html"}), "export_html"),
        (
            json!({"type":"switch_session","sessionPath":"/s.jsonl"}),
            "switch_session",
        ),
        (json!({"type":"fork","entryId":"e1"}), "fork"),
        (json!({"type":"clone"}), "clone"),
        (json!({"type":"get_fork_messages"}), "get_fork_messages"),
        (json!({"type":"get_entries"}), "get_entries"),
        (json!({"type":"get_tree"}), "get_tree"),
        (
            json!({"type":"get_last_assistant_text"}),
            "get_last_assistant_text",
        ),
        (
            json!({"type":"set_session_name","name":"n"}),
            "set_session_name",
        ),
        (json!({"type":"get_messages"}), "get_messages"),
        (json!({"type":"get_commands"}), "get_commands"),
    ];
    assert_eq!(rows.len(), super::COMMAND_TYPES.len());
    for (value, expected) in rows {
        let command = parse(value);
        assert_eq!(command.type_name(), expected);
        assert_eq!(command.id(), None);
        assert!(super::COMMAND_TYPES.contains(&expected));
    }
}

#[test]
fn prompt_carries_the_camel_case_streaming_behavior_and_id() {
    let command = parse(json!({
        "id": "b1",
        "type": "prompt",
        "message": "hi",
        "streamingBehavior": "followUp",
        "images": [{"data": "abc", "mimeType": "image/png"}]
    }));
    let RpcCommand::Prompt {
        message,
        streaming_behavior,
        images,
        ..
    } = command
    else {
        panic!("the prompt arm parses");
    };
    assert_eq!(message, "hi");
    assert_eq!(
        streaming_behavior,
        Some(crate::extensions::reasons::StreamingBehavior::FollowUp)
    );
    assert_eq!(images.expect("images").len(), 1);
    assert_eq!(parse(json!({"type":"steer","message":"x"})).id(), None);
}

#[test]
fn queue_modes_and_thinking_levels_read_upstreams_wire_names() {
    let RpcCommand::SetSteeringMode { mode, .. } =
        parse(json!({"type":"set_steering_mode","mode":"one-at-a-time"}))
    else {
        panic!("the steering arm parses");
    };
    assert_eq!(mode, QueueMode::OneAtATime);
    let RpcCommand::SetThinkingLevel { level, .. } =
        parse(json!({"type":"set_thinking_level","level":"xhigh"}))
    else {
        panic!("the thinking arm parses");
    };
    assert_eq!(level, ThinkingLevel::Xhigh);
}

#[test]
fn an_unknown_command_type_rejects() {
    assert!(serde_json::from_value::<RpcCommand>(json!({"type":"nope"})).is_err());
}

#[test]
fn the_known_type_table_covers_every_wire_name() {
    assert_eq!(super::COMMAND_TYPES.len(), 33);
    assert!(!super::COMMAND_TYPES.contains(&"foobar"));
}

#[test]
fn ui_requests_serialize_with_the_upstream_body() {
    let select = serde_json::to_value(RpcExtensionUiRequest {
        id: "u1".to_owned(),
        method: RpcUiMethod::Select {
            title: "Pick".to_owned(),
            options: vec!["a".to_owned()],
            timeout: Some(250),
        },
    })
    .expect("the request serializes");
    assert_eq!(
        select,
        json!({
            "type": "extension_ui_request",
            "id": "u1",
            "method": "select",
            "title": "Pick",
            "options": ["a"],
            "timeout": 250
        })
    );

    let status = serde_json::to_value(RpcExtensionUiRequest {
        id: "u2".to_owned(),
        method: RpcUiMethod::SetStatus {
            status_key: "k".to_owned(),
            status_text: None,
        },
    })
    .expect("the status request serializes");
    assert_eq!(status["statusText"], serde_json::Value::Null);

    let notify = serde_json::to_value(RpcExtensionUiRequest {
        id: "u3".to_owned(),
        method: RpcUiMethod::Notify {
            message: "hi".to_owned(),
            notify_type: Some(RpcNotifyType::Warning),
        },
    })
    .expect("the notify request serializes");
    assert_eq!(notify["notifyType"], "warning");

    let widget = serde_json::to_value(RpcExtensionUiRequest {
        id: "u4".to_owned(),
        method: RpcUiMethod::SetWidget {
            widget_key: "w".to_owned(),
            widget_lines: Some(vec!["line".to_owned()]),
            widget_placement: Some(RpcWidgetPlacement::BelowEditor),
        },
    })
    .expect("the widget request serializes");
    assert_eq!(widget["widgetPlacement"], "belowEditor");
    assert_eq!(widget["widgetLines"], json!(["line"]));
}

#[test]
fn ui_responses_read_all_three_arms() {
    let value: RpcExtensionUiResponse =
        serde_json::from_value(json!({"type":"extension_ui_response","id":"a","value":"x"}))
            .expect("the value arm parses");
    assert!(!value.is_cancelled());
    assert_eq!(value.value.as_deref(), Some("x"));

    let confirmed: RpcExtensionUiResponse =
        serde_json::from_value(json!({"type":"extension_ui_response","id":"b","confirmed":true}))
            .expect("the confirm arm parses");
    assert_eq!(confirmed.confirmed, Some(true));

    let cancelled: RpcExtensionUiResponse =
        serde_json::from_value(json!({"type":"extension_ui_response","id":"c","cancelled":true}))
            .expect("the cancel arm parses");
    assert!(cancelled.is_cancelled());
}

#[test]
fn session_state_drops_absent_optionals_and_camel_cases() {
    let state = RpcSessionState {
        model: None,
        thinking_level: ThinkingLevel::Off,
        is_streaming: false,
        is_compacting: false,
        steering_mode: QueueMode::All,
        follow_up_mode: QueueMode::OneAtATime,
        session_file: None,
        session_id: "id".to_owned(),
        session_name: None,
        auto_compaction_enabled: true,
        message_count: 2,
        pending_message_count: 0,
    };
    let value = serde_json::to_value(&state).expect("the state serializes");
    assert!(value.get("model").is_none());
    assert!(value.get("sessionFile").is_none());
    assert!(value.get("sessionName").is_none());
    assert_eq!(value["thinkingLevel"], "off");
    assert_eq!(value["steeringMode"], "all");
    assert_eq!(value["followUpMode"], "one-at-a-time");
    assert_eq!(value["autoCompactionEnabled"], true);
}

#[test]
fn slash_commands_carry_provenance_and_source() {
    let command = RpcSlashCommand {
        name: "skill:demo".to_owned(),
        description: None,
        source: RpcCommandSource::Skill,
        source_info: crate::source_info::create_synthetic_source_info(
            "/skills/demo/SKILL.md",
            &crate::source_info::SyntheticSourceOptions {
                source: "local".to_owned(),
                ..crate::source_info::SyntheticSourceOptions::default()
            },
        ),
    };
    let value = serde_json::to_value(&command).expect("the command serializes");
    assert_eq!(value["source"], "skill");
    assert!(value.get("description").is_none());
    assert_eq!(value["sourceInfo"]["source"], "local");
    assert_eq!(value["sourceInfo"]["scope"], "temporary");
    assert_eq!(value["sourceInfo"]["origin"], "top-level");
}

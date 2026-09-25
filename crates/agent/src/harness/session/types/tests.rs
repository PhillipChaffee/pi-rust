//! The session contract surface's unit tests: the entry accessors, the
//! settled-stop-reason narrowing, and the operation-scope copy helper.
//! Upstream exercises them through the runtime suites, which ride the
//! session-layer child.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use serde_json::json;

use crate::harness::compaction::types::DEFAULT_COMPACTION_SETTINGS;
use crate::harness::session::types::{
    Entry, MessageEntry, OperationScope, OperationState, RunSettings, SessionError,
    SettledStopReason, StopReason, operation_scope_of,
};
use crate::types::{QueueMode, ToolExecutionMode};

fn user_message() -> crate::types::AgentMessage {
    serde_json::from_value(json!({
        "role": "user",
        "content": "hello",
        "timestamp": 1
    }))
    .expect("user message")
}

fn scope_fixture() -> OperationScope {
    OperationScope {
        control: crate::harness::session::types::Control::Running,
        settings: RunSettings {
            compaction: DEFAULT_COMPACTION_SETTINGS,
            steering_mode: QueueMode::All,
            follow_up_mode: QueueMode::OneAtATime,
            tool_execution: ToolExecutionMode::Parallel,
        },
        latest_assistant_entry_id: Some("assistant".to_owned()),
    }
}

/// The settled stop reason narrows over pending and round-trips back.
#[test]
fn the_settled_stop_reason_narrows_over_pending() {
    assert!(SettledStopReason::from_stop_reason(StopReason::Pending).is_none());
    assert_eq!(
        SettledStopReason::from_stop_reason(StopReason::ToolUse)
            .map(SettledStopReason::stop_reason),
        Some(StopReason::ToolUse)
    );
}

/// The entry accessors project the shared base fields.
#[test]
fn the_entry_accessors_project_the_base_fields() {
    let entry = Entry::Message {
        id: "entry".to_owned(),
        parent_id: None,
        seq: 1,
        timestamp: 1,
        body: Box::new(MessageEntry {
            message: user_message(),
            terminate: None,
        }),
    };
    assert_eq!(entry.id(), "entry");
    assert_eq!(entry.parent_id(), None);
    assert!(entry.custom_type().is_none());
    let custom = Entry::Custom {
        id: "custom".to_owned(),
        parent_id: Some("entry".to_owned()),
        seq: 2,
        timestamp: 2,
        body: crate::harness::session::types::CustomEntryBody {
            custom_type: "note".to_owned(),
            data: None,
        },
    };
    assert_eq!(custom.custom_type(), Some("note"));
}

/// The session error renders its message and chains as an error.
#[test]
fn the_session_error_renders_its_message() {
    let error = SessionError("failure".to_owned());
    assert_eq!(error.to_string(), "failure");
    let _: &dyn std::error::Error = &error;
}

/// `operation_scope_of` copies the uniform scope across leaf families.
#[test]
fn operation_scope_of_copies_the_uniform_scope() {
    let state = OperationState::Starting(crate::harness::session::types::StartingOperation {
        scope: scope_fixture(),
    });
    assert_eq!(operation_scope_of(&state), scope_fixture());
}

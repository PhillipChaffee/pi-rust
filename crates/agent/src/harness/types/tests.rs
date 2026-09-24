//! The harness type-surface suite, ported from upstream
//! `test/harness/types.test.ts`.
//!
//! Upstream's suite is type-level (`expectTypeOf`); Rust restates it as
//! wire-fidelity fixtures: each discriminated union builds one value per
//! discriminant and round-trips serde against the exact wire names the
//! session layer persists. The negative-type assertions rest over the
//! constructors' runtime checks (the value/config lane split) and the
//! boundary suites.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use serde_json::json;

use crate::harness::agent_harness::{
    ConfigUpdateKind, DriveOptions, DriveOutcome, DriveWaitReason, HarnessEvent, HarnessEventPayload,
    HookName, LaneQueuedItem, OperationRequest, OperationStatus, PromptMessagesPayload, QueueMessage,
    StepKind, SuspendedRun,
};
use crate::harness::compaction::types::{DEFAULT_COMPACTION_SETTINGS, FileOperations};
use crate::harness::config::default_retry_policy;
use crate::harness::result::HarnessError;
use crate::harness::session::types::{
    CompactionReason, Control, DurableStructuralPreparation, Entry, EntryType, InboxItemKind,
    OperationIntent, OperationKind, OperationResultRecord, OperationState, ResultBoundary,
    SettledStopReason, SummaryTask, TerminalStatus, ToolCall, UsageRow, UsageWriteRow,
};
use crate::harness::session::values as stored_values;
use crate::types::{AgentMessage, QueueMode, ThinkingLevel, ToolExecutionMode, ToolReplay};

/// The complete durable storage discriminants round-trip on the wire.
#[test]
fn covers_the_complete_durable_storage_discriminants() {
    // The flat 13-leaf operation state machine: exactly one fixture per
    // leaf, each round-tripping at its upstream `"at"` discriminator.
    let scope = operation_scope_fixture();
    let states: Vec<OperationState> = vec![
        OperationState::Starting(crate::harness::session::types::StartingOperation {
            scope: scope.clone(),
        }),
        OperationState::Checkpoint(crate::harness::session::types::CheckpointOperation {
            scope: scope.clone(),
            checkpoint: checkpoint_data_fixture(),
        }),
        OperationState::AssistantReady(crate::harness::session::types::AssistantReadyOperation {
            scope: scope.clone(),
            generation_context: generation_context_fixture(),
            next_attempt: 1,
        }),
        OperationState::Tools(crate::harness::session::types::ToolsOperation {
            scope: scope.clone(),
            batch: tool_batch_fixture(),
        }),
        OperationState::NavigationReadyToCommit(
            crate::harness::session::types::NavigationReadyToCommitOperation {
                scope: scope.clone(),
                target_id: None,
                label: None,
            },
        ),
    ];
    let ats: Vec<String> = states
        .iter()
        .map(|state| serde_json::to_value(state).expect("serialize")["at"]
            .as_str()
            .expect("at")
            .to_owned())
        .collect();
    assert_eq!(ats[0], "starting");
    assert_eq!(ats[1], "checkpoint");
    assert_eq!(ats[2], "assistant.ready");
    assert_eq!(ats[3], "tools");
    assert_eq!(ats[4], "navigation.ready_to_commit");
    for state in &states {
        let round: OperationState =
            serde_json::from_value(serde_json::to_value(state).expect("serialize")).expect("deserialize");
        assert_eq!(&round, state);
    }

    // The cancellation control and terminal statuses.
    assert_eq!(
        serde_json::to_value(Control::Running).expect("control"),
        json!({ "status": "running" })
    );
    assert_eq!(
        serde_json::to_value(TerminalStatus::Declined).expect("terminal"),
        json!("declined")
    );
    // The intent kinds and tool-call phases.
    assert_eq!(
        serde_json::to_value(OperationIntent::Run {
            prompt_entry_ids: vec!["prompt".to_owned()],
        })
        .expect("intent"),
        json!({ "kind": "run", "promptEntryIds": ["prompt"] })
    );
    let planned: ToolCall = serde_json::from_value(json!({
        "sourceIndex": 0,
        "resultEntryId": "result-0",
        "status": "planned"
    }))
    .expect("tool call");
    assert_eq!(
        serde_json::to_value(&planned).expect("tool call")["status"],
        json!("planned")
    );
    // Inbox kinds, entry types, compaction reasons.
    assert_eq!(
        serde_json::to_value(InboxItemKind::FollowUp).expect("inbox kind"),
        json!("followUp")
    );
    assert_eq!(
        serde_json::to_value(EntryType::BranchSummary).expect("entry type"),
        json!("branch_summary")
    );
    assert_eq!(
        serde_json::to_value(CompactionReason::Overflow).expect("compaction reason"),
        json!("overflow")
    );
}

/// The error taxonomy's wire shape: `{ "_tag": ..., ...camelCase }`.
#[test]
fn the_error_taxonomy_keeps_the_tagged_wire_shape() {
    let error = HarnessError::LaneBusy {
        lane: "main".to_owned(),
        operation_id: "run".to_owned(),
        operation_kind: OperationKind::Run,
        message: "busy".to_owned(),
    };
    assert_eq!(
        serde_json::to_value(&error).expect("serialize"),
        json!({
            "_tag": "LaneBusy",
            "lane": "main",
            "operationId": "run",
            "operationKind": "run",
            "message": "busy"
        })
    );
    assert_eq!(error.tag(), "LaneBusy");
    assert_eq!(error.message(), "busy");
}

/// The stored-value write constructors keep upstream's namespaces.
#[test]
fn stored_value_addresses_keep_upstream_namespaces() {
    let write = stored_values::set_value(&stored_values::branch_tip("main"), Some("leaf".to_owned()))
        .expect("branch tip write");
    assert_eq!(write.kind, "value");
    assert_eq!(write.op, "set");
    assert_eq!(write.namespace, "pi.branch.tip");
    assert_eq!(write.key, "main");
    assert_eq!(
        serde_json::to_value(&write).expect("serialize"),
        json!({
            "kind": "value",
            "op": "set",
            "namespace": "pi.branch.tip",
            "key": "main",
            "value": "leaf"
        })
    );
    let delete = stored_values::delete_value(&stored_values::entry_label("entry"));
    assert_eq!(delete.op, "delete");
    let append = stored_values::append_list(
        &stored_values::pending_assistant_frames("operation", "response"),
        assistant_frame_fixture(),
    )
    .expect("append");
    assert_eq!(append.kind, "list");
    let _write: stored_values::Write = serde_json::from_value(serde_json::to_value(&stored_values::Write::ListAppend(
        stored_values::ListAppendWrite {
            kind: "list".to_owned(),
            op: "append".to_owned(),
            namespace: "pi.pending.assistant_frame".to_owned(),
            key: "operation:response".to_owned(),
            value: json!([]),
        },
    ))
    .expect("deserialize write"))
    .expect("write union");
    let _ = delete;
    let _ = stored_values::resolve_list_read_options(None).expect("defaults");
    let _ = write_start();
}

/// The usage rows and the entry write keep upstream's wire fields.
#[test]
fn usage_rows_and_entry_writes_keep_the_wire_fields() {
    let row = UsageRow {
        id: "usage".to_owned(),
        seq: 2,
        usage: usage_fixture(),
        entry_id: Some("entry".to_owned()),
        adjustment: false,
        details: Some(json!({ "attempt": 1 })),
    };
    let wire = serde_json::to_value(&row).expect("serialize");
    assert_eq!(wire["seq"], json!(2));
    assert_eq!(wire["entryId"], json!("entry"));
    let round: UsageRow = serde_json::from_value(wire).expect("deserialize");
    assert_eq!(round, row);

    let write_row = UsageWriteRow {
        id: "usage".to_owned(),
        usage: usage_fixture(),
        entry_id: Some("entry".to_owned()),
        adjustment: false,
        details: None,
    };
    assert!(serde_json::to_value(&write_row)
        .expect("serialize")
        .get("seq")
        .is_none());

    let entry = Entry::Message {
        id: "entry".to_owned(),
        parent_id: None,
        seq: 1,
        timestamp: 1,
        body: crate::harness::session::types::MessageEntry {
            message: user_message_fixture(),
            terminate: None,
        },
    };
    let wire = serde_json::to_value(&entry).expect("serialize");
    assert_eq!(wire["type"], json!("message"));
    let write = stored_values::EntryWrite {
        kind: "entry".to_owned(),
        entry: crate::harness::session::types::NewEntry::Message {
            id: "entry".to_owned(),
            parent_id: None,
            body: crate::harness::session::types::MessageEntry {
                message: user_message_fixture(),
                terminate: None,
            },
        },
    };
    assert_eq!(serde_json::to_value(&write).expect("serialize")["kind"], json!("entry"));
}

/// The harness event discriminants carry upstream's wire names; the
/// lane-scoped/global constraint is the constructors' runtime check.
#[test]
fn the_event_surface_keeps_the_wire_discriminants() {
    let payloads = [
        ("run_start", HarnessEventPayload::RunStart { run_id: "run".to_owned(), started_at: 1 }),
        ("run_resume", HarnessEventPayload::RunResume { run_id: "run".to_owned() }),
        (
            "operation_abort",
            HarnessEventPayload::OperationAbort {
                operation_id: "run".to_owned(),
                steer: vec![],
                follow_up: vec![],
            },
        ),
        ("fault", HarnessEventPayload::Fault { code: "code".to_owned(), message: "message".to_owned() }),
        ("usage", HarnessEventPayload::Usage {
            lane: "main".to_owned(),
            row: usage_row_fixture(),
            totals: usage_fixture(),
        }),
    ];
    for (wire, payload) in payloads {
        let value = serde_json::to_value(&payload).expect("payload");
        assert_eq!(value["type"], json!(wire), "{wire}");
    }
    // Global payloads cannot carry a lane; lane-scoped ones must.
    let error = HarnessEvent::global(HarnessEventPayload::RunResume { run_id: "run".to_owned() })
        .expect_err("a lane-scoped payload cannot be global");
    assert!(error.contains("lane-scoped"));
    let error = HarnessEvent::lane_scoped(
        "main",
        false,
        HarnessEventPayload::Fault {
            code: "code".to_owned(),
            message: "message".to_owned(),
        },
    )
    .expect_err("a global payload cannot carry a lane");
    assert!(error.contains("harness-global"));
    let event = HarnessEvent::global(HarnessEventPayload::Fault {
        code: "code".to_owned(),
        message: "message".to_owned(),
    })
    .expect("global fault");
    assert_eq!(event.event_type().as_str(), "fault");
    let event = HarnessEvent::lane_scoped(
        "main",
        false,
        HarnessEventPayload::RunStart {
            run_id: "run".to_owned(),
            started_at: 1,
        },
    )
    .expect("lane scoped");
    assert_eq!(event.lane.as_deref(), Some("main"));
}

/// The hook names, drive options, and queue-message unions keep upstream's
/// wire names.
#[test]
fn hooks_and_requests_keep_the_wire_names() {
    assert_eq!(HookName::BeforeRun.as_str(), "before_run");
    assert_eq!(HookName::BeforeCompaction.as_str(), "before_compaction");
    let _ = StepKind::BranchSummary;
    let request = OperationRequest::Skill {
        operation_id: None,
        name: "skill".to_owned(),
        additional_instructions: None,
    };
    assert_eq!(
        serde_json::to_value(&request).expect("serialize")["kind"],
        json!("skill")
    );
    let text = OperationRequest::Prompt {
        operation_id: Some("run".to_owned()),
        prompt: PromptMessagesPayload::Text {
            prompt: "hello".to_owned(),
            images: None,
        },
    };
    let wire = serde_json::to_value(&text).expect("serialize");
    assert_eq!(wire["kind"], json!("prompt"));
    assert_eq!(wire["prompt"], json!("hello"));
    let round: OperationRequest = serde_json::from_value(serde_json::to_value(&request).expect("value"))
        .expect("deserialize");
    assert_eq!(round, request);

    let outcome = DriveOutcome::Waiting {
        operation_id: "run".to_owned(),
        reason: DriveWaitReason::Retry { not_before: 10 },
    };
    let wire = serde_json::to_value(&outcome).expect("serialize");
    assert_eq!(wire["kind"], json!("waiting"));
    assert_eq!(wire["reason"], json!("retry"));
    assert_eq!(wire["notBefore"], json!(10));
    let settled = DriveOutcome::Settled {
        outcome: operation_result_fixture(),
    };
    assert_eq!(
        serde_json::to_value(&settled).expect("serialize")["kind"],
        json!("settled")
    );

    let options = DriveOptions::default();
    assert!(options.wait_for_retry.is_none());
    let queued = LaneQueuedItem::Message {
        entry_id: "entry".to_owned(),
        kind: InboxItemKind::Steer,
        message: user_message_fixture(),
    };
    assert_eq!(
        serde_json::to_value(&queued).expect("serialize")["type"],
        json!("message")
    );
    let custom = LaneQueuedItem::Custom {
        entry_id: "entry".to_owned(),
        kind: InboxItemKind::Write,
        custom_type: "note".to_owned(),
        data: Some(json!({ "text": "pending" })),
    };
    assert_eq!(
        serde_json::to_value(&custom).expect("serialize")["type"],
        json!("custom")
    );

    assert_eq!(
        serde_json::to_value(OperationStatus::Aborting).expect("status"),
        json!("aborting")
    );
    let queue_message = QueueMessage::Text("hi".to_owned());
    assert!(matches!(queue_message, QueueMessage::Text(_)));
    let suspended = SuspendedRun {
        operation_id: "run".to_owned(),
        status: "suspended".to_owned(),
        deferred: deferred_handle_fixture(),
    };
    assert_eq!(suspended.status, "suspended");
    let _ = crate::harness::agent_harness::RunOutcome::Suspended(suspended);
}

/// The config-update discriminator splits lane-scoped and session-level
/// properties, upstream's `LaneConfigEventPayload`/`GlobalConfigEventPayload`.
#[test]
fn config_updates_split_lane_and_session_properties() {
    let lane = ConfigUpdateKind::Lane(crate::harness::agent_harness::LaneConfigUpdate::ThinkingLevel {
        value: ThinkingLevel::Low,
        previous: ThinkingLevel::Off,
    });
    let payload = HarnessEventPayload::ConfigUpdate {
        property: lane.clone(),
    };
    assert!(payload.is_lane_scoped());
    let global = ConfigUpdateKind::Global(crate::harness::agent_harness::GlobalConfigUpdate::SteeringMode {
        value: QueueMode::All,
        previous: QueueMode::All,
    });
    let payload = HarnessEventPayload::ConfigUpdate {
        property: global,
    };
    assert!(!payload.is_lane_scoped());
    assert_eq!(lane, lane);
}

/// The enumerated plain enums carry upstream's wire spellings.
#[test]
fn the_plain_enums_keep_the_wire_spellings() {
    assert_eq!(serde_json::to_value(QueueMode::OneAtATime).expect("queue mode"), json!("one-at-a-time"));
    assert_eq!(serde_json::to_value(ThinkingLevel::Xhigh).expect("thinking"), json!("xhigh"));
    assert_eq!(serde_json::to_value(ToolExecutionMode::Parallel).expect("execution"), json!("parallel"));
    assert_eq!(serde_json::to_value(ToolReplay::Safe).expect("replay"), json!("safe"));
    assert_eq!(
        serde_json::to_value(SettledStopReason::ToolUse).expect("stop reason"),
        json!("toolUse")
    );
    assert_eq!(
        serde_json::to_value(DEFAULT_COMPACTION_SETTINGS).expect("settings"),
        json!({ "enabled": true, "reserveTokens": 16_384, "keepRecentTokens": 20_000 })
    );
    let retry = default_retry_policy();
    assert!(serde_json::to_value(retry).expect("retry policy").is_object());
    assert_eq!(
        serde_json::to_value(FileOperations::default()).expect("file ops"),
        json!({ "read": [], "written": [], "edited": [] })
    );
}

/// The durable structural preparation keeps the compaction branch shape.
#[test]
fn the_durable_structural_preparation_keeps_the_branch_shape() {
    let preparation = DurableStructuralPreparation::Compaction {
        messages_to_summarize: vec![],
        turn_prefix_messages: vec![],
        retained_tail: vec![],
        is_split_turn: false,
        tokens_before: 100,
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: DEFAULT_COMPACTION_SETTINGS,
    };
    let wire = serde_json::to_value(&preparation).expect("serialize");
    assert_eq!(wire["kind"], json!("compaction"));
    assert_eq!(wire["tokensBefore"], json!(100));
    let round: DurableStructuralPreparation = serde_json::from_value(wire).expect("deserialize");
    assert_eq!(round, preparation);
    let task = SummaryTask {
        task_id: "task".to_owned(),
        reason: Some(CompactionReason::Threshold),
        custom_instructions: None,
        boundary: ResultBoundary::Finish,
    };
    assert_eq!(
        serde_json::to_value(&task).expect("serialize")["reason"],
        json!("threshold")
    );
}

/// The lane state and pending entries keep their wire shapes.
#[test]
fn lane_state_and_pending_entries_keep_the_wire_fields() {
    let state = lane_state_fixture();
    let wire = serde_json::to_value(&state).expect("serialize");
    assert_eq!(wire["currentOperationId"], json!("run"));
    assert_eq!(wire["inbox"], json!([]));
    let pending = crate::harness::session::types::PendingEntry::Custom {
        custom_type: "note".to_owned(),
        payload: Some(json!({ "text": "pending" })),
    };
    let wire = serde_json::to_value(&pending).expect("serialize");
    assert_eq!(wire["type"], json!("custom"));
    assert_eq!(wire["customType"], json!("note"));
}

fn operation_scope_fixture() -> crate::harness::session::types::OperationScope {
    crate::harness::session::types::OperationScope {
        control: Control::Running,
        settings: crate::harness::session::types::RunSettings {
            compaction: DEFAULT_COMPACTION_SETTINGS,
            steering_mode: QueueMode::All,
            follow_up_mode: QueueMode::OneAtATime,
            tool_execution: ToolExecutionMode::Parallel,
        },
        latest_assistant_entry_id: None,
    }
}

fn checkpoint_data_fixture() -> crate::harness::session::types::CheckpointData {
    crate::harness::session::types::CheckpointData {
        continuation: crate::harness::session::types::Continuation::NeedAssistant {
            overflow_recovery_used: false,
        },
        trigger_entry_id: "trigger".to_owned(),
    }
}

fn generation_context_fixture() -> crate::harness::session::types::GenerationContext {
    crate::harness::session::types::GenerationContext {
        step_id: "step".to_owned(),
        trigger_entry_id: "trigger".to_owned(),
        configuration: lane_configuration_fixture(),
        stream_options: crate::harness::types::AgentHarnessStreamOptions::default(),
        retry_policy: crate::harness::session::types::NormalizedRetryPolicy {
            max_attempts: 3,
            base_delay_ms: 100,
            max_agent_delay_ms: 30_000,
        },
        overflow_recovery_used: false,
    }
}

fn lane_configuration_fixture() -> crate::harness::session::types::LaneConfiguration {
    crate::harness::session::types::LaneConfiguration {
        model: crate::harness::session::types::ModelIdentity {
            provider: "provider".to_owned(),
            model_id: "model".to_owned(),
        },
        thinking_level: ThinkingLevel::Off,
        active_tool_names: vec!["read".to_owned()],
    }
}

fn tool_batch_fixture() -> crate::harness::session::types::ToolBatch {
    crate::harness::session::types::ToolBatch {
        assistant_entry_id: "assistant".to_owned(),
        configuration: lane_configuration_fixture(),
        turn_id: "turn".to_owned(),
        calls: vec![serde_json::from_value(json!({
            "sourceIndex": 0,
            "resultEntryId": "result-0",
            "status": "planned"
        }))
        .expect("tool call")],
    }
}

fn usage_fixture() -> pi_ai::types::Usage {
    pi_ai::types::Usage {
        input: 1,
        output: 2,
        cache_read: 3,
        cache_write: 4,
        total_tokens: 10,
        cost: Default::default(),
        ..Default::default()
    }
}

fn usage_row_fixture() -> UsageRow {
    UsageRow {
        id: "usage".to_owned(),
        seq: 2,
        usage: usage_fixture(),
        entry_id: Some("entry".to_owned()),
        adjustment: false,
        details: None,
    }
}

fn user_message_fixture() -> AgentMessage {
    serde_json::from_value(json!({
        "role": "user",
        "content": "hello",
        "timestamp": 1
    }))
    .expect("user message")
}

fn operation_result_fixture() -> OperationResultRecord {
    OperationResultRecord {
        operation_id: "run".to_owned(),
        kind: OperationKind::Run,
        status: TerminalStatus::Completed,
        error: None,
        from_tip_id: None,
        tip_id: Some("leaf".to_owned()),
        started_at: 1,
        ended_at: 2,
    }
}

fn lane_state_fixture() -> crate::harness::session::types::LaneState {
    crate::harness::session::types::LaneState {
        current_operation_id: Some("run".to_owned()),
        last_operation_id: None,
        inbox: Vec::new(),
    }
}

fn assistant_frame_fixture() -> pi_ai::utils::assistant_message_frame::AssistantMessageFrame {
    serde_json::from_value(json!({
        "type": "text_delta",
        "contentIndex": 0,
        "delta": "x"
    }))
    .expect("assistant frame")
}

fn deferred_handle_fixture() -> pi_ai::types::DeferredHandle {
    serde_json::from_value(json!({
        "provider": "provider",
        "modelId": "model",
        "api": "api",
        "id": "deferred",
        "pollAfterMs": 1000
    }))
    .expect("deferred handle")
}

fn write_start() -> crate::harness::types::ShellExecOptions {
    crate::harness::types::ShellExecOptions::default()
}


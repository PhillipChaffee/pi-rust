//! The session contract surface's unit tests: the serde wire-fidelity suite
//! over the entry and new-entry unions, the 13-leaf operation state machine,
//! the scans, and the payload structs, plus the entry accessors, the
//! settled-stop-reason narrowing, and the operation-scope copy helper.
//! Upstream exercises the runtime behavior through the session-layer child;
//! the wire shapes here pin `test/harness/types.test.ts`'s fixtures.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::sync::Arc;

use pi_ai::types::{AssistantMessage, BoxedFuture, StopReason, Usage};
use serde_json::json;

use crate::harness::compaction::types::{DEFAULT_COMPACTION_SETTINGS, FileOperations};
use crate::harness::context::{Context, background_context};
use crate::harness::session::types::{
    AssistantEffectPendingOperation, AssistantReadyOperation, AssistantRetryWaitOperation,
    BranchScan, BranchScanOrder, BranchSummaryEntryBody, CheckpointData, CheckpointOperation,
    CommitResult, CompactionEntryBody, CompactionReason, Continuation, Control, CustomEntryBody,
    DeferredEffectPendingOperation, DeferredScope, DeferredSuspendedOperation,
    DurableStructuralPreparation, Entry, EntryCursor, EntryProjector, EntryQuery, EntryScan,
    EntryScanOrder, EntryStructure, EntryType, GenerationContext, InboxItem, InboxItemKind,
    LaneConfiguration, LaneState, MessageEntry, NavigationReadyToCommitOperation, NewEntry,
    NormalizedRetryPolicy, OperationError, OperationIntent, OperationKind, OperationMeta,
    OperationResultRecord, OperationScope, OperationState, PendingEntry, ResultBoundary, RetryWait,
    RunSettings, SessionError, SessionMetadata, SessionStats, SettledAssistantMessage,
    SettledStopReason, StartingOperation, StorageBranchScan, SummaryContext,
    SummaryDecidingOperation, SummaryEffectPendingOperation, SummaryEffectRequest,
    SummaryGenerationScope, SummaryReadyOperation, SummaryRetryWaitOperation, SummaryTask,
    TerminalStatus, ToolBatch, ToolsOperation, UsageRow, UsageScan, UsageWriteRow,
    operation_scope_of,
};
use crate::harness::types::AgentHarnessStreamOptions;
use crate::types::{AgentMessage, QueueMode, ToolExecutionMode};

fn user_message_json() -> serde_json::Value {
    json!({
        "role": "user",
        "content": "hello",
        "timestamp": 1
    })
}

fn user_message() -> AgentMessage {
    serde_json::from_value(user_message_json()).expect("user message")
}

fn assistant_message_json() -> serde_json::Value {
    json!({
        "content": [{"type": "text", "text": "done"}],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "test-model",
        "timestamp": 2,
        "usage": usage_json(),
        "stopReason": "stop"
    })
}

fn assistant_message() -> AssistantMessage {
    serde_json::from_value(assistant_message_json()).expect("assistant message")
}

fn usage_json() -> serde_json::Value {
    json!({
        "input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4, "totalTokens": 10,
        "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 }
    })
}

fn usage() -> Usage {
    serde_json::from_value(usage_json()).expect("usage")
}

fn lane_configuration_json() -> serde_json::Value {
    json!({
        "model": {"provider": "provider", "modelId": "model"},
        "thinkingLevel": "off",
        "activeToolNames": ["read"]
    })
}

fn lane_configuration() -> LaneConfiguration {
    serde_json::from_value(lane_configuration_json()).expect("lane configuration")
}

fn retry_policy_json() -> serde_json::Value {
    json!({"maxAttempts": 3, "baseDelayMs": 100, "maxAgentDelayMs": 30_000})
}

fn retry_policy() -> NormalizedRetryPolicy {
    serde_json::from_value(retry_policy_json()).expect("retry policy")
}

fn stream_options_json() -> serde_json::Value {
    json!({
        "transport": null, "timeoutMs": null, "maxRetries": null, "maxRetryDelayMs": null,
        "headers": null, "metadata": null, "cacheRetention": null, "deferred": null
    })
}

fn stream_options() -> AgentHarnessStreamOptions {
    AgentHarnessStreamOptions::default()
}

fn generation_context_json() -> serde_json::Value {
    json!({
        "stepId": "step",
        "triggerEntryId": "trigger",
        "configuration": lane_configuration_json(),
        "streamOptions": stream_options_json(),
        "retryPolicy": retry_policy_json(),
        "overflowRecoveryUsed": false
    })
}

fn generation_context() -> GenerationContext {
    serde_json::from_value(generation_context_json()).expect("generation context")
}

fn summary_context_json() -> serde_json::Value {
    json!({
        "resultEntryId": "summary",
        "configuration": lane_configuration_json(),
        "streamOptions": stream_options_json(),
        "retryPolicy": retry_policy_json()
    })
}

fn summary_context() -> SummaryContext {
    serde_json::from_value(summary_context_json()).expect("summary context")
}

fn checkpoint_json() -> serde_json::Value {
    json!({
        "continuation": {"kind": "need_assistant", "overflowRecoveryUsed": false},
        "triggerEntryId": "trigger"
    })
}

fn checkpoint_data() -> CheckpointData {
    serde_json::from_value(checkpoint_json()).expect("checkpoint")
}

fn scope_settings_json() -> serde_json::Value {
    json!({
        "compaction": {"enabled": true, "reserveTokens": 16_384, "keepRecentTokens": 20_000},
        "steeringMode": "all",
        "followUpMode": "one-at-a-time",
        "toolExecution": "parallel"
    })
}

fn scope_json() -> serde_json::Value {
    json!({
        "control": {"status": "running"},
        "settings": scope_settings_json(),
        "latestAssistantEntryId": "assistant"
    })
}

fn scope_fixture() -> OperationScope {
    OperationScope {
        control: Control::Running,
        settings: RunSettings {
            compaction: DEFAULT_COMPACTION_SETTINGS,
            steering_mode: QueueMode::All,
            follow_up_mode: QueueMode::OneAtATime,
            tool_execution: ToolExecutionMode::Parallel,
        },
        latest_assistant_entry_id: Some("assistant".to_owned()),
    }
}

fn retry_wait_json() -> serde_json::Value {
    json!({"nextAttempt": 2, "notBefore": 10, "errorMessage": "retry"})
}

fn retry_wait() -> RetryWait {
    serde_json::from_value(retry_wait_json()).expect("retry wait")
}

fn tool_batch_json() -> serde_json::Value {
    json!({
        "assistantEntryId": "assistant",
        "configuration": lane_configuration_json(),
        "turnId": "turn",
        "calls": [
            {"sourceIndex": 0, "resultEntryId": "result-0", "status": "planned"},
            {"sourceIndex": 1, "resultEntryId": "result-1", "status": "effect_pending", "replay": "safe"},
            {"sourceIndex": 2, "resultEntryId": "result-2", "status": "outcome_ready", "terminate": true},
            {"sourceIndex": 3, "resultEntryId": "result-3", "status": "completed", "terminate": false}
        ]
    })
}

fn tool_batch() -> ToolBatch {
    serde_json::from_value(tool_batch_json()).expect("tool batch")
}

fn summary_task_json(
    boundary: &serde_json::Value,
    reason: Option<&str>,
    custom_instructions: Option<&str>,
) -> serde_json::Value {
    let mut task = json!({"taskId": "task", "boundary": boundary});
    if let Some(reason) = reason {
        task["reason"] = json!(reason);
    }
    if let Some(custom_instructions) = custom_instructions {
        task["customInstructions"] = json!(custom_instructions);
    }
    task
}

fn summary_task(
    boundary: ResultBoundary,
    reason: Option<CompactionReason>,
    custom_instructions: Option<&str>,
) -> SummaryTask {
    SummaryTask {
        task_id: "task".to_owned(),
        reason,
        custom_instructions: custom_instructions.map(str::to_owned),
        boundary,
    }
}

/// Round-trips one payload: the typed value serializes to the pinned wire
/// shape, and the pinned shape deserializes back to the same value.
fn assert_wire_round_trip<T>(value: &T, wire: serde_json::Value)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    assert_eq!(serde_json::to_value(value).expect("serialize"), wire);
    let back: T = serde_json::from_value(wire).expect("deserialize");
    assert_eq!(&back, value);
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

/// The settled stop reason maps every stop reason to its settled pair and
/// back, and the settled assistant message round-trips its flat wire shape.
#[test]
fn the_settled_stop_reason_covers_every_reason() {
    let reasons = [
        (StopReason::Stop, SettledStopReason::Stop, "stop"),
        (StopReason::Length, SettledStopReason::Length, "length"),
        (StopReason::ToolUse, SettledStopReason::ToolUse, "toolUse"),
        (StopReason::Error, SettledStopReason::Error, "error"),
        (StopReason::Aborted, SettledStopReason::Aborted, "aborted"),
        (
            StopReason::Deferred,
            SettledStopReason::Deferred,
            "deferred",
        ),
    ];
    for (reason, settled, wire) in reasons {
        assert_eq!(SettledStopReason::from_stop_reason(reason), Some(settled));
        assert_eq!(settled.stop_reason(), reason);
        assert_eq!(
            serde_json::to_value(settled).expect("serialize"),
            json!(wire)
        );
    }
    let settled = SettledAssistantMessage {
        message: assistant_message(),
        stop_reason: SettledStopReason::Stop,
    };
    assert_wire_round_trip(
        &settled,
        json!({"message": assistant_message_json(), "stopReason": "stop"}),
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
        body: CustomEntryBody {
            custom_type: "note".to_owned(),
            data: None,
        },
    };
    assert_eq!(custom.custom_type(), Some("note"));
}

/// The message and compaction entries round-trip the flat wire shape
/// upstream writes: the `"type"` discriminant with the flattened body.
#[test]
fn the_message_and_compaction_entries_round_trip_their_wire_shape() {
    let message = Entry::Message {
        id: "entry".to_owned(),
        parent_id: None,
        seq: 1,
        timestamp: 1,
        body: Box::new(MessageEntry {
            message: user_message(),
            terminate: Some(true),
        }),
    };
    assert_wire_round_trip(
        &message,
        json!({
            "type": "message",
            "id": "entry",
            "parentId": null,
            "seq": 1,
            "timestamp": 1,
            "message": user_message_json(),
            "terminate": true
        }),
    );
    let compaction = Entry::Compaction {
        id: "compaction".to_owned(),
        parent_id: Some("root".to_owned()),
        seq: 2,
        timestamp: 2,
        body: CompactionEntryBody {
            summary: "summary text".to_owned(),
            retained_tail: vec![user_message()],
            tokens_before: 100,
            details: Some(json!({"reason": "threshold"})),
            usage: Some(usage()),
            from_hook: false,
        },
    };
    assert_wire_round_trip(
        &compaction,
        json!({
            "type": "compaction",
            "id": "compaction",
            "parentId": "root",
            "seq": 2,
            "timestamp": 2,
            "summary": "summary text",
            "retainedTail": [user_message_json()],
            "tokensBefore": 100,
            "details": {"reason": "threshold"},
            "usage": usage_json(),
            "fromHook": false
        }),
    );
}

/// The branch-summary and custom entries round-trip the same flat wire
/// shape, and every entry variant projects the base accessors.
#[test]
fn the_branch_and_custom_entries_round_trip_their_wire_shape() {
    let branch = Entry::BranchSummary {
        id: "branch".to_owned(),
        parent_id: Some("entry".to_owned()),
        seq: 3,
        timestamp: 3,
        body: BranchSummaryEntryBody {
            from_id: Some("entry".to_owned()),
            summary: "branch summary".to_owned(),
            details: None,
            usage: None,
            from_hook: true,
        },
    };
    assert_wire_round_trip(
        &branch,
        json!({
            "type": "branch_summary",
            "id": "branch",
            "parentId": "entry",
            "seq": 3,
            "timestamp": 3,
            "fromId": "entry",
            "summary": "branch summary",
            "fromHook": true
        }),
    );
    let custom = Entry::Custom {
        id: "custom".to_owned(),
        parent_id: None,
        seq: 4,
        timestamp: 4,
        body: CustomEntryBody {
            custom_type: "note".to_owned(),
            data: Some(json!({"x": 1})),
        },
    };
    assert_wire_round_trip(
        &custom,
        json!({
            "type": "custom",
            "id": "custom",
            "parentId": null,
            "seq": 4,
            "timestamp": 4,
            "customType": "note",
            "data": {"x": 1}
        }),
    );
    let entries = [
        message_entry_fixture(),
        compaction_entry_fixture(),
        Entry::BranchSummary {
            id: "branch".to_owned(),
            parent_id: Some("root".to_owned()),
            seq: 9,
            timestamp: 9,
            body: BranchSummaryEntryBody {
                from_id: None,
                summary: "branch summary".to_owned(),
                details: None,
                usage: None,
                from_hook: false,
            },
        },
        Entry::Custom {
            id: "custom".to_owned(),
            parent_id: Some("root".to_owned()),
            seq: 9,
            timestamp: 9,
            body: CustomEntryBody {
                custom_type: "note".to_owned(),
                data: None,
            },
        },
    ];
    let kinds = [
        EntryType::Message,
        EntryType::Compaction,
        EntryType::BranchSummary,
        EntryType::Custom,
    ];
    for (entry, kind) in entries.iter().zip(kinds) {
        assert_eq!(entry.entry_type(), kind);
        assert_eq!(entry.seq(), 9);
        assert_eq!(entry.timestamp(), 9);
        assert_eq!(entry.parent_id(), Some("root"));
    }
    assert_eq!(entries[3].custom_type(), Some("note"));
    assert!(entries[0].custom_type().is_none());
    assert_eq!(EntryType::default(), EntryType::Message);
}

fn message_entry_fixture() -> Entry {
    Entry::Message {
        id: "entry".to_owned(),
        parent_id: Some("root".to_owned()),
        seq: 9,
        timestamp: 9,
        body: Box::new(MessageEntry {
            message: user_message(),
            terminate: None,
        }),
    }
}

fn compaction_entry_fixture() -> Entry {
    Entry::Compaction {
        id: "compaction".to_owned(),
        parent_id: Some("root".to_owned()),
        seq: 9,
        timestamp: 9,
        body: CompactionEntryBody {
            summary: "summary text".to_owned(),
            retained_tail: vec![user_message()],
            tokens_before: 100,
            details: Some(json!({"reason": "threshold"})),
            usage: Some(usage()),
            from_hook: false,
        },
    }
}

/// Every new-entry variant round-trips the entry wire shape minus
/// `seq`/`timestamp`, and materializing stamps both onto the entry.
#[test]
fn every_new_entry_variant_round_trips_and_materializes() {
    let new_entries = [
        NewEntry::Message {
            id: "entry".to_owned(),
            parent_id: None,
            body: Box::new(MessageEntry {
                message: user_message(),
                terminate: None,
            }),
        },
        NewEntry::Compaction {
            id: "compaction".to_owned(),
            parent_id: Some("entry".to_owned()),
            body: CompactionEntryBody {
                summary: "summary text".to_owned(),
                retained_tail: vec![user_message()],
                tokens_before: 100,
                details: None,
                usage: None,
                from_hook: true,
            },
        },
        NewEntry::BranchSummary {
            id: "branch".to_owned(),
            parent_id: Some("entry".to_owned()),
            body: BranchSummaryEntryBody {
                from_id: None,
                summary: "branch summary".to_owned(),
                details: None,
                usage: None,
                from_hook: false,
            },
        },
        NewEntry::Custom {
            id: "custom".to_owned(),
            parent_id: Some("entry".to_owned()),
            body: CustomEntryBody {
                custom_type: "note".to_owned(),
                data: None,
            },
        },
    ];
    let wires = [
        json!({
            "type": "message",
            "id": "entry",
            "parentId": null,
            "message": user_message_json()
        }),
        json!({
            "type": "compaction",
            "id": "compaction",
            "parentId": "entry",
            "summary": "summary text",
            "retainedTail": [user_message_json()],
            "tokensBefore": 100,
            "fromHook": true
        }),
        json!({
            "type": "branch_summary",
            "id": "branch",
            "parentId": "entry",
            "fromId": null,
            "summary": "branch summary",
            "fromHook": false
        }),
        json!({
            "type": "custom",
            "id": "custom",
            "parentId": "entry",
            "customType": "note"
        }),
    ];
    for (new_entry, mut wire) in new_entries.iter().zip(wires) {
        assert_wire_round_trip(new_entry, wire.clone());
        wire["seq"] = json!(5);
        wire["timestamp"] = json!(1_000);
        let entry = new_entry.clone().materialize(5, 1_000);
        assert_eq!(serde_json::to_value(&entry).expect("serialize"), wire);
        assert_eq!(entry.id(), new_entry.id());
        assert_eq!(entry.parent_id(), new_entry.parent_id());
        assert_eq!(entry.seq(), 5);
        assert_eq!(entry.timestamp(), 1_000);
    }
}

/// The operation intents carry their `"kind"` discriminants and optional
/// fields, the metadata wraps one intent, and the cancellation control
/// rides its `"status"` tag.
#[test]
fn operation_intents_and_controls_round_trip_their_wire_shape() {
    let run = OperationIntent::Run {
        prompt_entry_ids: vec!["prompt".to_owned()],
    };
    assert_wire_round_trip(&run, json!({"kind": "run", "promptEntryIds": ["prompt"]}));
    let compacting = OperationIntent::Compaction {
        custom_instructions: Some("compact".to_owned()),
    };
    assert_wire_round_trip(
        &compacting,
        json!({"kind": "compaction", "customInstructions": "compact"}),
    );
    let bare_compacting = OperationIntent::Compaction {
        custom_instructions: None,
    };
    assert_eq!(
        serde_json::to_value(&bare_compacting).expect("serialize"),
        json!({"kind": "compaction"})
    );
    let navigation = OperationIntent::Navigation {
        target_id: Some("target".to_owned()),
        summarize: true,
        label: Some("target".to_owned()),
        custom_instructions: Some("compact".to_owned()),
    };
    assert_wire_round_trip(
        &navigation,
        json!({
            "kind": "navigation",
            "targetId": "target",
            "summarize": true,
            "label": "target",
            "customInstructions": "compact"
        }),
    );
    let bare_navigation = OperationIntent::Navigation {
        target_id: None,
        summarize: false,
        label: None,
        custom_instructions: None,
    };
    assert_wire_round_trip(
        &bare_navigation,
        json!({"kind": "navigation", "targetId": null, "summarize": false}),
    );
    let meta = OperationMeta {
        operation_id: "run".to_owned(),
        lane: "main".to_owned(),
        source_tip_id: None,
        started_at: 1,
        intent: run,
    };
    assert_wire_round_trip(
        &meta,
        json!({
            "operationId": "run",
            "lane": "main",
            "sourceTipId": null,
            "startedAt": 1,
            "intent": {"kind": "run", "promptEntryIds": ["prompt"]}
        }),
    );
    assert_wire_round_trip(&Control::Running, json!({"status": "running"}));
    assert_wire_round_trip(
        &Control::CancelRequested { requested_at: 7 },
        json!({"status": "cancel_requested", "requestedAt": 7}),
    );
}

/// The starting and checkpoint leaves pin their `"at"` discriminants; the
/// checkpoint payload flattens beside the nested uniform scope.
#[test]
fn the_starting_and_checkpoint_leaves_round_trip_their_wire_shape() {
    let scope = scope_fixture();
    let starting = OperationState::Starting(StartingOperation {
        scope: scope.clone(),
    });
    assert_wire_round_trip(&starting, json!({"at": "starting", "scope": scope_json()}));
    assert_eq!(operation_scope_of(&starting), scope);

    let checkpoint = OperationState::Checkpoint(CheckpointOperation {
        scope: scope.clone(),
        checkpoint: checkpoint_data(),
    });
    assert_wire_round_trip(
        &checkpoint,
        json!({
            "at": "checkpoint",
            "scope": scope_json(),
            "continuation": {"kind": "need_assistant", "overflowRecoveryUsed": false},
            "triggerEntryId": "trigger"
        }),
    );
    assert_eq!(operation_scope_of(&checkpoint), scope);
    assert_wire_round_trip(&checkpoint_data(), checkpoint_json());
    assert_wire_round_trip(
        &Continuation::MayFinish {
            include_final_assistant: true,
        },
        json!({"kind": "may_finish", "includeFinalAssistant": true}),
    );
}

/// The assistant leaves pin their `"at"` discriminants with the nested
/// generation context and the flattened retry-wait backoff.
#[test]
fn the_assistant_leaves_round_trip_their_wire_shape() {
    let scope = scope_fixture();
    let ready = OperationState::AssistantReady(AssistantReadyOperation {
        scope: scope.clone(),
        generation_context: generation_context(),
        next_attempt: 1,
    });
    assert_wire_round_trip(
        &ready,
        json!({
            "at": "assistant.ready",
            "scope": scope_json(),
            "generationContext": generation_context_json(),
            "nextAttempt": 1
        }),
    );
    assert_eq!(operation_scope_of(&ready), scope);

    let effect_pending = OperationState::AssistantEffectPending(AssistantEffectPendingOperation {
        scope: scope.clone(),
        generation_context: generation_context(),
        attempt: 1,
        response_entry_id: "response".to_owned(),
        usage_id: "usage".to_owned(),
        intended_output_limit: 4_096,
        context_window: 128_000,
    });
    assert_wire_round_trip(
        &effect_pending,
        json!({
            "at": "assistant.effect_pending",
            "scope": scope_json(),
            "generationContext": generation_context_json(),
            "attempt": 1,
            "responseEntryId": "response",
            "usageId": "usage",
            "intendedOutputLimit": 4_096,
            "contextWindow": 128_000
        }),
    );
    assert_eq!(operation_scope_of(&effect_pending), scope);

    let retry_wait = OperationState::AssistantRetryWait(AssistantRetryWaitOperation {
        scope: scope.clone(),
        generation_context: generation_context(),
        retry_wait: retry_wait(),
    });
    let mut retry_wire = json!({
        "at": "assistant.retry_wait",
        "scope": scope_json(),
        "generationContext": generation_context_json()
    });
    retry_wire["nextAttempt"] = json!(2);
    retry_wire["notBefore"] = json!(10);
    retry_wire["errorMessage"] = json!("retry");
    assert_wire_round_trip(&retry_wait, retry_wire);
    assert_eq!(operation_scope_of(&retry_wait), scope);
    assert_wire_round_trip(&generation_context(), generation_context_json());
}

/// The tools leaf keeps the nested batch with every tool-call phase, and
/// the deferred leaves flatten the whole deferred scope onto the wire.
#[test]
fn the_tools_and_deferred_leaves_round_trip_their_wire_shape() {
    let scope = scope_fixture();
    let tools = OperationState::Tools(ToolsOperation {
        scope: scope.clone(),
        batch: tool_batch(),
    });
    assert_wire_round_trip(
        &tools,
        json!({
            "at": "tools",
            "scope": scope_json(),
            "batch": tool_batch_json()
        }),
    );
    assert_eq!(operation_scope_of(&tools), scope);

    let deferred = DeferredScope {
        scope: scope.clone(),
        step_id: "step".to_owned(),
        source_entry_id: "source".to_owned(),
        poll: 0,
        configuration: lane_configuration(),
        stream_options: stream_options(),
    };
    let suspended = OperationState::DeferredSuspended(DeferredSuspendedOperation {
        deferred: deferred.clone(),
    });
    assert_wire_round_trip(
        &suspended,
        json!({
            "at": "deferred.suspended",
            "control": {"status": "running"},
            "settings": scope_settings_json(),
            "latestAssistantEntryId": "assistant",
            "stepId": "step",
            "sourceEntryId": "source",
            "poll": 0,
            "configuration": lane_configuration_json(),
            "streamOptions": stream_options_json()
        }),
    );
    assert_eq!(operation_scope_of(&suspended), scope);

    let effect_pending = OperationState::DeferredEffectPending(DeferredEffectPendingOperation {
        scope: DeferredScope {
            poll: 1,
            ..deferred
        },
        response_entry_id: "response".to_owned(),
        usage_id: "usage".to_owned(),
    });
    assert_wire_round_trip(
        &effect_pending,
        json!({
            "at": "deferred.effect_pending",
            "control": {"status": "running"},
            "settings": scope_settings_json(),
            "latestAssistantEntryId": "assistant",
            "stepId": "step",
            "sourceEntryId": "source",
            "poll": 1,
            "configuration": lane_configuration_json(),
            "streamOptions": stream_options_json(),
            "responseEntryId": "response",
            "usageId": "usage"
        }),
    );
    assert_eq!(operation_scope_of(&effect_pending), scope);
}

/// The summary deciding and ready leaves pin their `"at"` discriminants
/// with the flattened structural task and summary context.
#[test]
fn the_summary_deciding_and_ready_leaves_round_trip_their_wire_shape() {
    let scope = scope_fixture();
    let deciding = OperationState::SummaryDeciding(SummaryDecidingOperation {
        scope: scope.clone(),
        task: summary_task(
            ResultBoundary::ResumeCheckpoint {
                resume_after: checkpoint_data(),
            },
            Some(CompactionReason::Threshold),
            None,
        ),
    });
    assert_wire_round_trip(
        &deciding,
        json!({
            "at": "summary.deciding",
            "scope": scope_json(),
            "task": summary_task_json(
                &json!({"kind": "resume_checkpoint", "resumeAfter": checkpoint_json()}),
                Some("threshold"),
                None
            )
        }),
    );
    assert_eq!(operation_scope_of(&deciding), scope);

    let ready = OperationState::SummaryReady(SummaryReadyOperation {
        scope: scope.clone(),
        generation: SummaryGenerationScope {
            task: summary_task(
                ResultBoundary::Finish,
                Some(CompactionReason::Manual),
                Some("compact"),
            ),
            summary_context: summary_context(),
        },
        next_attempt: 1,
    });
    assert_wire_round_trip(
        &ready,
        json!({
            "at": "summary.ready",
            "scope": scope_json(),
            "task": summary_task_json(&json!({"kind": "finish"}), Some("manual"), Some("compact")),
            "summaryContext": summary_context_json(),
            "nextAttempt": 1
        }),
    );
    assert_eq!(operation_scope_of(&ready), scope);
}

/// The summary effect-pending and retry-wait leaves pin their wire shapes;
/// the in-flight request and reserved usage rows ride along.
#[test]
fn the_summary_pending_and_retry_leaves_round_trip_their_wire_shape() {
    let scope = scope_fixture();
    let effect_pending = OperationState::SummaryEffectPending(SummaryEffectPendingOperation {
        scope: scope.clone(),
        generation: SummaryGenerationScope {
            task: summary_task(
                ResultBoundary::CommitNavigation {
                    target_id: "target".to_owned(),
                    label: Some("target".to_owned()),
                },
                None,
                None,
            ),
            summary_context: summary_context(),
        },
        attempt: 1,
        request: Some(SummaryEffectRequest {
            index: 0,
            usage_id: "usage".to_owned(),
        }),
        usage_ids: Vec::new(),
    });
    assert_wire_round_trip(
        &effect_pending,
        json!({
            "at": "summary.effect_pending",
            "scope": scope_json(),
            "task": summary_task_json(
                &json!({"kind": "commit_navigation", "targetId": "target", "label": "target"}),
                None,
                None
            ),
            "summaryContext": summary_context_json(),
            "attempt": 1,
            "request": {"index": 0, "usageId": "usage"},
            "usageIds": []
        }),
    );
    assert_eq!(operation_scope_of(&effect_pending), scope);

    let retry_wait = OperationState::SummaryRetryWait(SummaryRetryWaitOperation {
        scope: scope.clone(),
        generation: SummaryGenerationScope {
            task: summary_task(
                ResultBoundary::ResumeCheckpoint {
                    resume_after: checkpoint_data(),
                },
                Some(CompactionReason::Overflow),
                None,
            ),
            summary_context: summary_context(),
        },
        retry_wait: retry_wait(),
    });
    assert_wire_round_trip(
        &retry_wait,
        json!({
            "at": "summary.retry_wait",
            "scope": scope_json(),
            "task": summary_task_json(
                &json!({"kind": "resume_checkpoint", "resumeAfter": checkpoint_json()}),
                Some("overflow"),
                None
            ),
            "summaryContext": summary_context_json(),
            "nextAttempt": 2,
            "notBefore": 10,
            "errorMessage": "retry"
        }),
    );
    assert_eq!(operation_scope_of(&retry_wait), scope);
}

/// The navigation leaf pins its `"at"` discriminant with the nullable
/// target and optional label.
#[test]
fn the_navigation_leaf_round_trips_its_wire_shape() {
    let scope = scope_fixture();
    let root_target = OperationState::NavigationReadyToCommit(NavigationReadyToCommitOperation {
        scope: scope.clone(),
        target_id: None,
        label: None,
    });
    assert_wire_round_trip(
        &root_target,
        json!({"at": "navigation.ready_to_commit", "scope": scope_json(), "targetId": null}),
    );
    assert_eq!(operation_scope_of(&root_target), scope);

    let labeled = OperationState::NavigationReadyToCommit(NavigationReadyToCommitOperation {
        scope: scope.clone(),
        target_id: Some("target".to_owned()),
        label: Some("label".to_owned()),
    });
    assert_wire_round_trip(
        &labeled,
        json!({
            "at": "navigation.ready_to_commit",
            "scope": scope_json(),
            "targetId": "target",
            "label": "label"
        }),
    );
    assert_eq!(operation_scope_of(&labeled), scope);
}

/// The flat scans, cursors, orders, and branch scans round-trip their
/// camelCase wire shapes with the `"kind"` entry-restriction key.
#[test]
fn the_scan_payloads_round_trip_their_wire_shape() {
    assert_eq!(
        serde_json::to_value(EntryScanOrder::Asc).expect("serialize"),
        json!("asc")
    );
    assert_eq!(
        serde_json::to_value(EntryScanOrder::Desc).expect("serialize"),
        json!("desc")
    );
    assert_eq!(EntryScanOrder::default(), EntryScanOrder::Asc);
    assert_wire_round_trip(&EntryCursor { seq: 4 }, json!({"seq": 4}));
    let query = EntryQuery {
        kind: Some(EntryType::Message),
        custom_type: Some("note".to_owned()),
        order: Some(EntryScanOrder::Desc),
        limit: Some(5),
        cursor: Some(EntryCursor { seq: 4 }),
    };
    assert_wire_round_trip(
        &query,
        json!({
            "kind": "message",
            "customType": "note",
            "order": "desc",
            "limit": 5,
            "cursor": {"seq": 4}
        }),
    );
    assert_eq!(
        serde_json::from_value::<EntryQuery>(json!({})).expect("deserialize"),
        EntryQuery::default()
    );
    let scan = EntryScan {
        kind: Some(EntryType::Custom),
        custom_type: Some("note".to_owned()),
        from_seq: Some(1),
        to_seq: Some(9),
        order: Some(EntryScanOrder::Asc),
        limit: Some(50),
    };
    assert_wire_round_trip(
        &scan,
        json!({
            "kind": "custom",
            "customType": "note",
            "fromSeq": 1,
            "toSeq": 9,
            "order": "asc",
            "limit": 50
        }),
    );
    assert_wire_round_trip(
        &UsageScan {
            from_seq: Some(2),
            to_seq: Some(8),
            order: Some(EntryScanOrder::Desc),
            limit: Some(25),
        },
        json!({"fromSeq": 2, "toSeq": 8, "order": "desc", "limit": 25}),
    );
}

/// The branch scans and the entry structure round-trip their camelCase
/// wire shapes with the newest-first and oldest-first orders.
#[test]
fn the_branch_scan_payloads_round_trip_their_wire_shape() {
    assert_eq!(
        serde_json::to_value(BranchScanOrder::NewestFirst).expect("serialize"),
        json!("newestFirst")
    );
    assert_eq!(
        serde_json::to_value(BranchScanOrder::OldestFirst).expect("serialize"),
        json!("oldestFirst")
    );
    let branch_scan = BranchScan {
        start: Some("entry".to_owned()),
        stop_at_type: Some(EntryType::Compaction),
        stop_at_id: Some("stop".to_owned()),
        kind: Some(EntryType::Message),
        custom_type: Some("note".to_owned()),
        order: Some(BranchScanOrder::NewestFirst),
        limit: Some(5),
        cursor: Some(EntryCursor { seq: 2 }),
    };
    assert_wire_round_trip(
        &branch_scan,
        json!({
            "start": "entry",
            "stopAtType": "compaction",
            "stopAtId": "stop",
            "kind": "message",
            "customType": "note",
            "order": "newestFirst",
            "limit": 5,
            "cursor": {"seq": 2}
        }),
    );
    let storage_scan = StorageBranchScan {
        start: "entry".to_owned(),
        stop_at_type: None,
        stop_at_id: None,
        kind: None,
        custom_type: None,
        order: Some(BranchScanOrder::OldestFirst),
        limit: None,
        cursor: None,
    };
    assert_wire_round_trip(
        &storage_scan,
        json!({"start": "entry", "order": "oldestFirst"}),
    );
    let structure = EntryStructure {
        id: "custom".to_owned(),
        parent_id: None,
        seq: 3,
        timestamp: 3,
        kind: EntryType::Custom,
        custom_type: Some("note".to_owned()),
    };
    assert_wire_round_trip(
        &structure,
        json!({
            "id": "custom",
            "parentId": null,
            "seq": 3,
            "timestamp": 3,
            "type": "custom",
            "customType": "note"
        }),
    );
    assert_eq!(EntryType::default(), EntryType::Message);
}

/// The terminal statuses, operation kinds, and result records round-trip
/// their camelCase wire shapes with the optional error omitted.
#[test]
fn the_result_payloads_round_trip_their_wire_shape() {
    let statuses = [
        (TerminalStatus::Completed, "completed"),
        (TerminalStatus::Declined, "declined"),
        (TerminalStatus::Aborted, "aborted"),
        (TerminalStatus::Failed, "failed"),
    ];
    for (status, wire) in statuses {
        assert_eq!(
            serde_json::to_value(status).expect("serialize"),
            json!(wire)
        );
    }
    let kinds = [
        (OperationKind::Run, "run"),
        (OperationKind::Compaction, "compaction"),
        (OperationKind::Navigation, "navigation"),
    ];
    for (kind, wire) in kinds {
        assert_eq!(serde_json::to_value(kind).expect("serialize"), json!(wire));
    }
    let record = OperationResultRecord {
        operation_id: "run".to_owned(),
        kind: OperationKind::Run,
        status: TerminalStatus::Failed,
        error: Some(OperationError {
            code: "provider".to_owned(),
            message: "the request failed".to_owned(),
            details: Some(json!({"attempt": 1})),
        }),
        from_tip_id: None,
        tip_id: Some("leaf".to_owned()),
        started_at: 1,
        ended_at: 2,
    };
    assert_wire_round_trip(
        &record,
        json!({
            "operationId": "run",
            "kind": "run",
            "status": "failed",
            "error": {
                "code": "provider",
                "message": "the request failed",
                "details": {"attempt": 1}
            },
            "fromTipId": null,
            "tipId": "leaf",
            "startedAt": 1,
            "endedAt": 2
        }),
    );
    let clean_record = OperationResultRecord {
        error: None,
        ..record
    };
    assert_wire_round_trip(
        &clean_record,
        json!({
            "operationId": "run",
            "kind": "run",
            "status": "failed",
            "fromTipId": null,
            "tipId": "leaf",
            "startedAt": 1,
            "endedAt": 2
        }),
    );
}

/// The inbox items carry their camelCase queue kinds and the lane state
/// carries them with the nullable operation ids.
#[test]
fn the_inbox_and_lane_state_payloads_round_trip_their_wire_shape() {
    let kinds = [
        (InboxItemKind::Steer, "steer"),
        (InboxItemKind::FollowUp, "followUp"),
        (InboxItemKind::NextRun, "nextRun"),
        (InboxItemKind::Write, "write"),
    ];
    for (kind, wire) in kinds {
        let item = InboxItem {
            entry_id: "inbox-entry".to_owned(),
            kind,
        };
        assert_wire_round_trip(&item, json!({"entryId": "inbox-entry", "kind": wire}));
    }
    let lane_state = LaneState {
        current_operation_id: Some("run".to_owned()),
        last_operation_id: None,
        inbox: vec![InboxItem {
            entry_id: "inbox-entry".to_owned(),
            kind: InboxItemKind::FollowUp,
        }],
    };
    assert_wire_round_trip(
        &lane_state,
        json!({
            "currentOperationId": "run",
            "lastOperationId": null,
            "inbox": [{"entryId": "inbox-entry", "kind": "followUp"}]
        }),
    );
    assert_eq!(
        LaneState::default(),
        LaneState {
            current_operation_id: None,
            last_operation_id: None,
            inbox: Vec::new(),
        }
    );
}

/// The usage rows, session totals, commit results, and session metadata
/// round-trip their camelCase wire shapes with optional fields omitted.
#[test]
fn the_usage_and_metadata_payloads_round_trip_their_wire_shape() {
    let usage_row = UsageRow {
        id: "usage".to_owned(),
        seq: 2,
        usage: usage(),
        entry_id: Some("entry".to_owned()),
        adjustment: false,
        details: Some(json!({"attempt": 1})),
    };
    assert_wire_round_trip(
        &usage_row,
        json!({
            "id": "usage",
            "seq": 2,
            "usage": usage_json(),
            "entryId": "entry",
            "adjustment": false,
            "details": {"attempt": 1}
        }),
    );
    let write_row = UsageWriteRow {
        id: "usage".to_owned(),
        usage: usage(),
        entry_id: None,
        adjustment: true,
        details: None,
    };
    assert_wire_round_trip(
        &write_row,
        json!({"id": "usage", "usage": usage_json(), "adjustment": true}),
    );
    let stats = SessionStats {
        message_count: 3,
        usage: usage(),
    };
    assert_wire_round_trip(&stats, json!({"messageCount": 3, "usage": usage_json()}));
    let commit = CommitResult {
        first_seq: 1,
        seqs: vec![1, 2],
        timestamp: 10,
        stats,
    };
    assert_wire_round_trip(
        &commit,
        json!({
            "firstSeq": 1,
            "seqs": [1, 2],
            "timestamp": 10,
            "stats": {"messageCount": 3, "usage": usage_json()}
        }),
    );
    let metadata = SessionMetadata {
        id: "session".to_owned(),
        created_at: 1,
        storage_version: 3,
        cwd: Some("/tmp".to_owned()),
        parent_session_id: Some("parent".to_owned()),
        legacy_parent_session_path: None,
    };
    assert_wire_round_trip(
        &metadata,
        json!({
            "id": "session",
            "createdAt": 1,
            "storageVersion": 3,
            "cwd": "/tmp",
            "parentSessionId": "parent"
        }),
    );
}

/// The generation payloads, boundaries, and compaction reasons round-trip
/// their camelCase wire shapes.
#[test]
fn the_generation_and_boundary_payloads_round_trip_their_wire_shape() {
    assert_wire_round_trip(&lane_configuration(), lane_configuration_json());
    assert_wire_round_trip(&retry_policy(), retry_policy_json());
    assert_wire_round_trip(&stream_options(), stream_options_json());
    let resume = ResultBoundary::ResumeCheckpoint {
        resume_after: checkpoint_data(),
    };
    assert_wire_round_trip(
        &resume,
        json!({"kind": "resume_checkpoint", "resumeAfter": checkpoint_json()}),
    );
    assert_wire_round_trip(&ResultBoundary::Finish, json!({"kind": "finish"}));
    assert_wire_round_trip(
        &ResultBoundary::CommitNavigation {
            target_id: "target".to_owned(),
            label: None,
        },
        json!({"kind": "commit_navigation", "targetId": "target"}),
    );
    let reasons = [
        (CompactionReason::Manual, "manual"),
        (CompactionReason::Threshold, "threshold"),
        (CompactionReason::Overflow, "overflow"),
    ];
    for (reason, wire) in reasons {
        assert_eq!(
            serde_json::to_value(reason).expect("serialize"),
            json!(wire)
        );
    }
    let task = summary_task(resume, Some(CompactionReason::Threshold), Some("compact"));
    assert_wire_round_trip(
        &task,
        json!({
            "taskId": "task",
            "reason": "threshold",
            "customInstructions": "compact",
            "boundary": {"kind": "resume_checkpoint", "resumeAfter": checkpoint_json()}
        }),
    );
}

/// The durable structural preparations carry their `"kind"` discriminants
/// and the sorted file-operations vectors, omitting the absent previous
/// summary.
#[test]
fn the_durable_structural_preparation_round_trips_its_wire_shape() {
    let compaction = DurableStructuralPreparation::Compaction {
        messages_to_summarize: vec![user_message()],
        turn_prefix_messages: Vec::new(),
        retained_tail: vec![user_message()],
        is_split_turn: true,
        tokens_before: 100,
        previous_summary: Some("previous".to_owned()),
        file_ops: FileOperations {
            read: vec!["a".to_owned()],
            written: vec!["b".to_owned()],
            edited: Vec::new(),
        },
        settings: DEFAULT_COMPACTION_SETTINGS,
    };
    assert_wire_round_trip(
        &compaction,
        json!({
            "kind": "compaction",
            "messagesToSummarize": [user_message_json()],
            "turnPrefixMessages": [],
            "retainedTail": [user_message_json()],
            "isSplitTurn": true,
            "tokensBefore": 100,
            "previousSummary": "previous",
            "fileOps": {"read": ["a"], "written": ["b"], "edited": []},
            "settings": {"enabled": true, "reserveTokens": 16_384, "keepRecentTokens": 20_000}
        }),
    );
    let bare_compaction = DurableStructuralPreparation::Compaction {
        messages_to_summarize: Vec::new(),
        turn_prefix_messages: Vec::new(),
        retained_tail: Vec::new(),
        is_split_turn: false,
        tokens_before: 0,
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: DEFAULT_COMPACTION_SETTINGS,
    };
    assert_wire_round_trip(
        &bare_compaction,
        json!({
            "kind": "compaction",
            "messagesToSummarize": [],
            "turnPrefixMessages": [],
            "retainedTail": [],
            "isSplitTurn": false,
            "tokensBefore": 0,
            "fileOps": {"read": [], "written": [], "edited": []},
            "settings": {"enabled": true, "reserveTokens": 16_384, "keepRecentTokens": 20_000}
        }),
    );
    let branch = DurableStructuralPreparation::BranchSummary {
        messages: vec![user_message()],
        file_ops: FileOperations {
            read: Vec::new(),
            written: Vec::new(),
            edited: vec!["b".to_owned()],
        },
        total_tokens: 42,
    };
    assert_wire_round_trip(
        &branch,
        json!({
            "kind": "branch_summary",
            "messages": [user_message_json()],
            "fileOps": {"read": [], "written": [], "edited": ["b"]},
            "totalTokens": 42
        }),
    );
}

/// The pending entries carry their `"type"` discriminants with the custom
/// payload omitted when absent.
#[test]
fn the_pending_entry_round_trips_its_wire_shape() {
    let message = PendingEntry::Message {
        payload: Box::new(user_message()),
    };
    assert_wire_round_trip(
        &message,
        json!({"type": "message", "payload": user_message_json()}),
    );
    let custom = PendingEntry::Custom {
        custom_type: "note".to_owned(),
        payload: Some(json!({"text": "pending"})),
    };
    assert_wire_round_trip(
        &custom,
        json!({"type": "custom", "customType": "note", "payload": {"text": "pending"}}),
    );
    let bare_custom = PendingEntry::Custom {
        custom_type: "note".to_owned(),
        payload: None,
    };
    assert_eq!(
        serde_json::to_value(&bare_custom).expect("serialize"),
        json!({"type": "custom", "customType": "note"})
    );
    assert_eq!(
        serde_json::from_value::<PendingEntry>(json!({"type": "custom", "customType": "note"}))
            .expect("deserialize"),
        bare_custom
    );
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
    let state = OperationState::Starting(StartingOperation {
        scope: scope_fixture(),
    });
    assert_eq!(operation_scope_of(&state), scope_fixture());
}

/// The entry projector renders as an opaque handle and projects an entry
/// into model context over the harness context seam.
#[tokio::test]
async fn the_entry_projector_renders_debug_and_projects() {
    let projector = EntryProjector(Arc::new(
        |_entry: &Entry, _context: &Context| -> BoxedFuture<'_, Option<Vec<AgentMessage>>> {
            Box::pin(std::future::ready(None))
        },
    ));
    assert_eq!(format!("{projector:?}"), "EntryProjector(..)");
    let entry = message_entry_fixture();
    let projected = (projector.0)(&entry, &background_context()).await;
    assert_eq!(projected, None);
}

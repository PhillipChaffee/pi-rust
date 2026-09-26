//! The agent-harness type surface's unit tests: the event payload's
//! lane-scoped split over every discriminant and the type discriminators'
//! wire names. Upstream exercises them through the runtime suites.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::types::BoxedFuture;

use crate::harness::agent_harness::{
    AgentHarnessOptions, HarnessEvent, HarnessEventPayload, HarnessEventType, HookName,
    LaneSnapshotTool, OperationStatus, Subscription, SystemPromptSource,
};
use crate::harness::context::Context;
use crate::harness::session::types::{
    Branch, Entry, EntryQuery, IdGenerator, SessionError, SessionMetadata, SessionMutation,
    SessionMutationCallback, SessionReader, SessionStats, StorageBranchScan,
};
use crate::harness::session::values::{
    ListAddress, ListElement, ListReadOptions, StoredValue, ValueAddress,
};

fn deferred_handle() -> pi_ai::types::DeferredHandle {
    serde_json::from_value(serde_json::json!({
        "provider": "provider",
        "modelId": "model",
        "api": "api",
        "id": "deferred",
        "pollAfterMs": 1000
    }))
    .expect("deferred handle")
}

fn assistant_message() -> pi_ai::types::AssistantMessage {
    serde_json::from_value(serde_json::json!({
        "content": [{"type": "text", "text": "done"}],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "test-model",
        "timestamp": 2,
        "usage": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2,
            "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 } },
        "stopReason": "stop",
    }))
    .expect("assistant message")
}

fn user_agent_message() -> crate::types::AgentMessage {
    serde_json::from_value(serde_json::json!({
        "role": "user",
        "content": "hello",
        "timestamp": 1
    }))
    .expect("user message")
}

fn assistant_message_event() -> pi_ai::types::AssistantMessageEvent {
    serde_json::from_value(serde_json::json!({
        "type": "text_delta",
        "contentIndex": 0,
        "delta": "x",
        "partial": {
            "content": [{"type": "text", "text": "x"}],
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": "test-model",
            "timestamp": 2,
            "usage": { "input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2,
                "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 } },
            "stopReason": "pending",
        }
    }))
    .expect("assistant message event")
}

fn usage() -> pi_ai::types::Usage {
    serde_json::from_value(serde_json::json!({
        "input": 1, "output": 2, "cacheRead": 3, "cacheWrite": 4, "totalTokens": 10,
        "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 }
    }))
    .expect("usage")
}

/// Every payload discriminant round-trips its upstream wire name and its
/// type discriminator matches, upstream's `HarnessEvent["type"]` table.
#[test]
fn every_payload_discriminant_carries_its_wire_name() {
    let kinds = [
        HarnessEventType::RunStart,
        HarnessEventType::RunResume,
        HarnessEventType::RunSuspend,
        HarnessEventType::OperationAbort,
        HarnessEventType::RunEnd,
        HarnessEventType::Fault,
        HarnessEventType::HandlerError,
        HarnessEventType::TurnStart,
        HarnessEventType::TurnEnd,
        HarnessEventType::RetryScheduled,
        HarnessEventType::RetryStart,
        HarnessEventType::RetryEnd,
        HarnessEventType::MessageStart,
        HarnessEventType::MessageUpdate,
        HarnessEventType::MessageEnd,
        HarnessEventType::ToolStart,
        HarnessEventType::ToolUpdate,
        HarnessEventType::ToolEnd,
        HarnessEventType::EntryAdded,
        HarnessEventType::QueueUpdate,
        HarnessEventType::ValueUpdate,
        HarnessEventType::ConfigUpdate,
        HarnessEventType::CompactionStart,
        HarnessEventType::CompactionEnd,
        HarnessEventType::NavigationStart,
        HarnessEventType::NavigationEnd,
        HarnessEventType::LaneCreated,
        HarnessEventType::Usage,
    ];
    for kind in kinds {
        let payload = payload_for(kind);
        let serialized = serde_json::to_value(&payload).expect("payload serializes");
        let wire = kind.as_str();
        assert_eq!(serialized["type"], serde_json::json!(wire), "{wire}");
        assert_eq!(
            payload.event_type().as_str(),
            wire,
            "the discriminant matches: {wire}"
        );
        let round: HarnessEventPayload = serde_json::from_value(serialized)
            .unwrap_or_else(|error| panic!("{wire} round-trips: {error}"));
        assert_eq!(round, payload, "{wire}");
    }
}

/// The lane-scoped split matches upstream's `LaneEventPayload` membership
/// through the runtime constructors.
#[test]
fn the_lane_split_matches_upstream_membership() {
    for kind in [
        HarnessEventType::RunStart,
        HarnessEventType::RunResume,
        HarnessEventType::RunSuspend,
        HarnessEventType::OperationAbort,
        HarnessEventType::RunEnd,
        HarnessEventType::TurnStart,
        HarnessEventType::TurnEnd,
        HarnessEventType::RetryScheduled,
        HarnessEventType::RetryStart,
        HarnessEventType::RetryEnd,
        HarnessEventType::MessageStart,
        HarnessEventType::MessageUpdate,
        HarnessEventType::MessageEnd,
        HarnessEventType::ToolStart,
        HarnessEventType::ToolUpdate,
        HarnessEventType::ToolEnd,
        HarnessEventType::EntryAdded,
        HarnessEventType::QueueUpdate,
        HarnessEventType::ConfigUpdate,
        HarnessEventType::CompactionStart,
        HarnessEventType::CompactionEnd,
        HarnessEventType::NavigationStart,
        HarnessEventType::NavigationEnd,
        HarnessEventType::LaneCreated,
    ] {
        let payload = payload_for(kind);
        assert!(payload.is_lane_scoped(), "{} is lane-scoped", kind.as_str());
    }
    for kind in [
        HarnessEventType::Fault,
        HarnessEventType::ValueUpdate,
        HarnessEventType::Usage,
    ] {
        let payload = payload_for(kind);
        assert!(
            !payload.is_lane_scoped(),
            "{} is harness-global",
            kind.as_str()
        );
    }
    // A lane-scoped config_update stays lane-scoped; a session-level one
    // does not.
    let lane_update = HarnessEventPayload::ConfigUpdate {
        property: crate::harness::agent_harness::ConfigUpdateKind::Lane(
            crate::harness::agent_harness::LaneConfigUpdate::ThinkingLevel {
                value: crate::types::ThinkingLevel::Low,
                previous: crate::types::ThinkingLevel::Off,
            },
        ),
    };
    assert!(lane_update.is_lane_scoped());
    let event = HarnessEvent::lane_scoped("main", false, lane_update).expect("lane scoped");
    assert_eq!(event.event_type().as_str(), "config_update");
}

/// The status and tool-snapshot unions carry their wire spellings.
#[test]
fn the_status_and_tool_snapshot_unions_keep_the_wire() {
    assert_eq!(
        serde_json::to_value(OperationStatus::Aborting).expect("status"),
        serde_json::json!("aborting")
    );
    let running_tool = LaneSnapshotTool::Running {
        tool_call_id: "call".to_owned(),
        tool_name: "read".to_owned(),
        args: serde_json::json!({}),
        result: None,
    };
    let LaneSnapshotTool::Running {
        tool_call_id,
        tool_name,
        ..
    } = &running_tool
    else {
        panic!("the running shape");
    };
    assert_eq!(tool_call_id, "call");
    assert_eq!(tool_name, "read");
    let subscription = Subscription::new(Arc::new(|| {}));
    subscription.unsubscribe();
    subscription.unsubscribe();
}

/// One fixture payload per event discriminant, upstream's
/// `HarnessEventPayload` table.
#[expect(
    clippy::too_many_lines,
    reason = "the fixture enumerates every payload discriminant; splitting it would hide the table's completeness"
)]
fn payload_for(kind: HarnessEventType) -> HarnessEventPayload {
    use crate::harness::agent_harness::{
        CompactionEndStatus, ConfigUpdateKind, HandlerErrorKind, LaneConfigUpdate,
        NavigationEndStatus, RunEndStatus, ValueUpdateKind,
    };
    use crate::harness::session::types::{CompactionReason, MessageEntry, UsageRow};

    match kind {
        HarnessEventType::RunStart => HarnessEventPayload::RunStart {
            run_id: "run".to_owned(),
            started_at: 1,
        },
        HarnessEventType::RunResume => HarnessEventPayload::RunResume {
            run_id: "run".to_owned(),
        },
        HarnessEventType::RunSuspend => HarnessEventPayload::RunSuspend {
            run_id: "run".to_owned(),
            deferred: deferred_handle(),
            poll: 0,
        },
        HarnessEventType::OperationAbort => HarnessEventPayload::OperationAbort {
            operation_id: "run".to_owned(),
            steer: vec![],
            follow_up: vec![],
        },
        HarnessEventType::RunEnd => HarnessEventPayload::RunEnd {
            run_id: "run".to_owned(),
            from_tip_id: None,
            tip_id: None,
            ended_at: 2,
            status: RunEndStatus::Completed,
        },
        HarnessEventType::Fault => HarnessEventPayload::Fault {
            code: "code".to_owned(),
            message: "message".to_owned(),
        },
        HarnessEventType::HandlerError => HarnessEventPayload::HandlerError {
            error: "error".to_owned(),
            stack: None,
            kind: HandlerErrorKind::Event {
                event: "run_start".to_owned(),
            },
        },
        HarnessEventType::TurnStart => HarnessEventPayload::TurnStart {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
        },
        HarnessEventType::TurnEnd => HarnessEventPayload::TurnEnd {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
            message: assistant_message(),
            tool_results: vec![],
        },
        HarnessEventType::RetryScheduled => HarnessEventPayload::RetryScheduled {
            run_id: "run".to_owned(),
            step: "assistant".to_owned(),
            attempt: 1,
            max_attempts: 3,
            delay_ms: 100,
            not_before: 10,
            error_message: "error".to_owned(),
        },
        HarnessEventType::RetryStart => HarnessEventPayload::RetryStart {
            run_id: "run".to_owned(),
            step: "assistant".to_owned(),
            attempt: 1,
        },
        HarnessEventType::RetryEnd => HarnessEventPayload::RetryEnd {
            run_id: "run".to_owned(),
            step: "assistant".to_owned(),
            attempt: 1,
            success: true,
            final_error: None,
        },
        HarnessEventType::MessageStart => HarnessEventPayload::MessageStart {
            run_id: None,
            message: user_agent_message(),
        },
        HarnessEventType::MessageUpdate => HarnessEventPayload::MessageUpdate {
            run_id: "run".to_owned(),
            message: Box::new(user_agent_message()),
            event: Box::new(assistant_message_event()),
            frame: None,
        },
        HarnessEventType::MessageEnd => HarnessEventPayload::MessageEnd {
            run_id: None,
            message: user_agent_message(),
            entry_id: None,
        },
        HarnessEventType::ToolStart => HarnessEventPayload::ToolStart {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call".to_owned(),
            tool_name: "read".to_owned(),
            args: serde_json::json!({}),
        },
        HarnessEventType::ToolUpdate => HarnessEventPayload::ToolUpdate {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call".to_owned(),
            tool_name: "read".to_owned(),
            partial_result: crate::types::AgentToolResult {
                content: vec![],
                details: serde_json::json!({}),
                usage: None,
                added_tool_names: None,
                terminate: None,
            },
        },
        HarnessEventType::ToolEnd => HarnessEventPayload::ToolEnd {
            run_id: "run".to_owned(),
            turn_id: "turn".to_owned(),
            tool_call_id: "call".to_owned(),
            tool_name: "read".to_owned(),
            result: crate::types::AgentToolResult {
                content: vec![],
                details: serde_json::json!({}),
                usage: None,
                added_tool_names: None,
                terminate: None,
            },
            is_error: false,
            terminate: false,
        },
        HarnessEventType::EntryAdded => HarnessEventPayload::EntryAdded {
            entry: Entry::Message {
                id: "entry".to_owned(),
                parent_id: None,
                seq: 1,
                timestamp: 1,
                body: Box::new(MessageEntry {
                    message: user_agent_message(),
                    terminate: None,
                }),
            },
        },
        HarnessEventType::QueueUpdate => HarnessEventPayload::QueueUpdate { queues: vec![] },
        HarnessEventType::ValueUpdate => HarnessEventPayload::ValueUpdate {
            kind: ValueUpdateKind::SessionName {
                name: Some("session".to_owned()),
            },
        },
        HarnessEventType::ConfigUpdate => HarnessEventPayload::ConfigUpdate {
            property: ConfigUpdateKind::Lane(LaneConfigUpdate::ThinkingLevel {
                value: crate::types::ThinkingLevel::Low,
                previous: crate::types::ThinkingLevel::Off,
            }),
        },
        HarnessEventType::CompactionStart => HarnessEventPayload::CompactionStart {
            run_id: "run".to_owned(),
            reason: CompactionReason::Manual,
            started_at: 1,
        },
        HarnessEventType::CompactionEnd => HarnessEventPayload::CompactionEnd {
            run_id: "run".to_owned(),
            reason: CompactionReason::Manual,
            ended_at: 2,
            status: CompactionEndStatus::Completed {
                entry_id: "entry".to_owned(),
            },
        },
        HarnessEventType::NavigationStart => HarnessEventPayload::NavigationStart {
            run_id: "run".to_owned(),
            target_id: None,
            started_at: 1,
        },
        HarnessEventType::NavigationEnd => HarnessEventPayload::NavigationEnd {
            run_id: "run".to_owned(),
            from_tip_id: None,
            tip_id: None,
            ended_at: 2,
            status: NavigationEndStatus::Completed,
        },
        HarnessEventType::LaneCreated => HarnessEventPayload::LaneCreated { at: None },
        HarnessEventType::Usage => HarnessEventPayload::Usage {
            lane: "main".to_owned(),
            row: UsageRow {
                id: "usage".to_owned(),
                seq: 2,
                usage: usage(),
                entry_id: None,
                adjustment: false,
                details: None,
            },
            totals: usage(),
        },
    }
}

/// The boundary constructors reject cross-scope payloads with the payload's
/// wire name, and build harness-global events lane-less.
#[test]
fn the_boundary_constructors_reject_cross_scope_payloads() {
    let error = HarnessEvent::lane_scoped("main", false, payload_for(HarnessEventType::Fault))
        .expect_err("fault is harness-global");
    assert!(error.contains("fault"), "{error}");
    assert!(error.contains("harness-global"), "{error}");

    let error = HarnessEvent::global(payload_for(HarnessEventType::RunStart))
        .expect_err("run_start is lane-scoped");
    assert!(error.contains("run_start"), "{error}");
    assert!(error.contains("lane-scoped"), "{error}");

    let event = HarnessEvent::global(payload_for(HarnessEventType::Fault)).expect("fault global");
    assert!(event.lane.is_none());
    assert!(!event.recovery);
    assert_eq!(event.event_type().as_str(), "fault");
}

/// The subscription renders an opaque debug shape.
#[test]
fn the_subscription_renders_an_opaque_debug_shape() {
    let subscription = Subscription::new(Arc::new(|| {}));
    assert_eq!(format!("{subscription:?}"), "Subscription(..)");
}

/// Every hook name carries its wire discriminator.
#[test]
fn every_hook_name_carries_its_wire_discriminator() {
    let names = [
        (HookName::BeforeRun, "before_run"),
        (HookName::BeforeDrive, "before_drive"),
        (HookName::BeforeRunEnd, "before_run_end"),
        (HookName::TransformContext, "transform_context"),
        (HookName::BeforeRequest, "before_request"),
        (HookName::BeforePayload, "before_payload"),
        (HookName::AfterResponse, "after_response"),
        (HookName::BeforeTool, "before_tool"),
        (HookName::AfterTool, "after_tool"),
        (HookName::BeforeCompaction, "before_compaction"),
        (HookName::BeforeNavigation, "before_navigation"),
    ];
    for (name, wire) in names {
        assert_eq!(name.as_str(), wire);
    }
}

/// The system-prompt source renders its static prompt inline and its
/// provider opaquely.
#[test]
fn the_system_prompt_source_renders_both_variants() {
    let static_source = SystemPromptSource::Static("be terse".to_owned());
    let debug = format!("{static_source:?}");
    assert!(debug.contains("be terse"), "{debug}");
    let provided_source = SystemPromptSource::Provided(Arc::new(
        |_tool_context: crate::harness::types::ToolContext, _context: &Context| {
            Box::pin(async { "generated".to_owned() })
        },
    ));
    assert_eq!(
        format!("{provided_source:?}"),
        "SystemPromptSource::Provided(..)"
    );
}

/// A session stub that reports nothing and fails every operation; the
/// options' debug shape is the only surface under test.
struct StubSession {
    metadata: SessionMetadata,
    ids: StubIds,
}

impl Default for StubSession {
    fn default() -> Self {
        Self {
            metadata: SessionMetadata {
                id: "session".to_owned(),
                created_at: 1,
                storage_version: 1,
                cwd: None,
                parent_session_id: None,
                legacy_parent_session_path: None,
            },
            ids: StubIds,
        }
    }
}

struct StubIds;

impl IdGenerator for StubIds {
    fn next(&self, _timestamp_ms: Option<i64>) -> String {
        "id".to_owned()
    }
}

fn stub_error<T>() -> BoxedFuture<'static, Result<T, SessionError>> {
    Box::pin(async { Err(SessionError("stub".to_owned())) })
}

impl SessionReader for StubSession {
    fn get_entries(
        &self,
        _ids: Vec<String>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<BTreeMap<String, Entry>, SessionError>> {
        stub_error()
    }

    fn get_stats(&self, _context: &Context) -> BoxedFuture<'_, Result<SessionStats, SessionError>> {
        stub_error()
    }

    fn get_value(
        &self,
        _address: &ValueAddress,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<StoredValue>, SessionError>> {
        stub_error()
    }

    fn scan_values(
        &self,
        _prefix: &ValueAddress,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<StoredValue>, SessionError>> {
        stub_error()
    }

    fn read_list(
        &self,
        _address: &ListAddress,
        _options: Option<ListReadOptions>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<ListElement>, SessionError>> {
        stub_error()
    }

    fn scan_branch(
        &self,
        _query: &StorageBranchScan,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        stub_error()
    }
}

impl crate::harness::session::types::Session for StubSession {
    fn metadata(&self) -> &SessionMetadata {
        &self.metadata
    }

    fn id_generator(&self) -> &dyn IdGenerator {
        &self.ids
    }

    fn get_entry(
        &self,
        _id: &str,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        stub_error()
    }

    fn get_name(
        &self,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        stub_error()
    }

    fn get_label(
        &self,
        _target_id: &str,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<String>, SessionError>> {
        stub_error()
    }

    fn find_entries(
        &self,
        _query: Option<&EntryQuery>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Vec<Entry>, SessionError>> {
        stub_error()
    }

    fn find_entry(
        &self,
        _query: Option<&EntryQuery>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Entry>, SessionError>> {
        stub_error()
    }

    fn branch(
        &self,
        _name: &str,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Option<Box<dyn Branch>>, SessionError>> {
        stub_error()
    }

    fn create_branch(
        &self,
        _name: &str,
        _at: Option<String>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn Branch>, SessionError>> {
        stub_error()
    }

    fn begin_mutation(
        &self,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn SessionMutation>, SessionError>> {
        stub_error()
    }

    fn mutate(
        &self,
        _mutation: SessionMutationCallback,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<Box<dyn std::any::Any + Send>, SessionError>> {
        stub_error()
    }

    fn set_value(
        &self,
        _address: &ValueAddress,
        _next: serde_json::Value,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        stub_error()
    }

    fn delete_value(
        &self,
        _address: &ValueAddress,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        stub_error()
    }

    fn append_list(
        &self,
        _address: &ListAddress,
        _element: serde_json::Value,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        stub_error()
    }

    fn delete_list(
        &self,
        _address: &ListAddress,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        stub_error()
    }

    fn set_name(
        &self,
        _name: Option<String>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        stub_error()
    }

    fn set_label(
        &self,
        _target_id: &str,
        _label: Option<String>,
        _context: &Context,
    ) -> BoxedFuture<'_, Result<(), SessionError>> {
        stub_error()
    }

    fn close(&self, _context: &Context) -> BoxedFuture<'_, Result<(), SessionError>> {
        stub_error()
    }
}

/// A fixture model the options carry.
fn fixture_model() -> pi_ai::types::Model {
    pi_ai::types::Model {
        id: "test-model".to_owned(),
        name: "Test model".to_owned(),
        api: pi_ai::types::Api("anthropic-messages".to_owned()),
        provider: pi_ai::types::ProviderId("anthropic".to_owned()),
        base_url: "https://example.test".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![pi_ai::types::Modality::Text],
        cost: pi_ai::types::ModelCost {
            rates: pi_ai::types::ModelCostRates {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            tiers: None,
        },
        context_window: 1000,
        max_tokens: 100,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// The harness options render their debug shape over a stub session.
#[test]
fn the_harness_options_render_their_debug_shape() {
    let options = AgentHarnessOptions {
        session: Arc::new(StubSession::default()),
        models: Arc::new(pi_ai::models::create_models(None)),
        model: fixture_model(),
        thinking_level: None,
        active_tool_names: None,
        tools: None,
        tool_context: None,
        system_prompt: None,
        resources: None,
        stream_options: None,
        retry: None,
        compaction: None,
        steering_mode: None,
        follow_up_mode: None,
        tool_execution: None,
        to_provider_messages: None,
        entry_projectors: None,
    };
    let debug = format!("{options:?}");
    assert!(debug.contains("AgentHarnessOptions"), "{debug}");
}

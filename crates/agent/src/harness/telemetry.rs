//! The harness telemetry vocabulary, ported from upstream
//! `src/harness/telemetry.ts`.
//!
//! The two schemas restate over the pi-telemetry port's
//! [`define_telemetry_schema!`](pi_telemetry::define_telemetry_schema)
//! macro, which generates the marker vocabulary upstream's TypeScript
//! literal types carry: start attributes are generated structs, closed
//! sets are generated enums, and unknown names or cross-schema attributes
//! are type errors. Two upstream surfaces reshape, recorded here:
//!
//! - The span-name ordering assertion over the schema's key order restates
//!   over the generated [`TelemetrySchema::SPAN_NAMES`](pi_telemetry::schema::TelemetrySchema)
//!   declaration order; the definition map itself sorts alphabetically.
//! - `pi.session.item_kinds`' closed element set (`elementValues`) has no
//!   slot in the macro grammar, so the attribute restates as a plain
//!   string array; the closed element set lives only in the doc
//!   generator, which the port does not carry.
//! - `scripts/generate-telemetry-docs.ts` and the checked-in
//!   `docs/telemetry-schema.md` reference render are not ported — repo
//!   scripts are outside the package port (the map's precedent).
//!
//! The span-as-parent seam: upstream threads the live span through the
//! chord `Context` (`withTelemetryContext(span, context)`); the port
//! carries the dispatch handle per ADR 0004, and spans convert through
//! [`DynSpanHandle::telemetry_handle`](pi_telemetry::DynSpanHandle::telemetry_handle).

use std::future::Future;

use pi_telemetry::DynSpanHandle;
use pi_telemetry::schema::{IntoSpanAttributes, SpanDefinition};
use pi_telemetry::{SpanBodyFailure, SpanOptions};

pub use pi_telemetry::{
    AttributeValue, DynSpanHandle as TelemetrySpanHandle, ExactTelemetryAttributes, SchemaSpan,
    SpanAttributes, SpanOptions as SpanStartOptions, SpanStatus, TelemetryAttributeDefinition,
    TelemetryContext, TelemetryHandle, TelemetrySpan, TypedSpanStarter,
};

use crate::harness::context::{get_telemetry_context, with_telemetry_context, Context};

pi_telemetry::define_telemetry_schema! {
    /// The AI-request vocabulary, upstream's `AI_TELEMETRY_SCHEMA`.
    pub schema ai_schema {
        version: 1,
        spans: {
            /// One logical request to an AI provider, upstream's
            /// `"pi.ai.request"`.
            request => "One logical request to an AI provider" {
                parents: [any],
                start_attributes: {
                    "pi.ai.operation" operation as AiOperation: string required values [
                        Stream: "stream",
                        FetchDeferred: "fetch_deferred",
                        CancelDeferred: "cancel_deferred",
                        GenerateImages: "generate_images",
                    ] description: "Logical provider operation",
                    "pi.ai.provider" provider: string required description: "Selected provider id",
                    "pi.ai.model" model: string required description: "Requested model id",
                    "pi.ai.api" api: string required description: "Provider API id",
                    "pi.ai.streaming" streaming: boolean required description: "Whether this operation returns a stream",
                    "pi.ai.deferred" deferred: boolean optional description: "Whether the operation requests or participates in deferred execution",
                },
                end_attributes: {
                    "pi.ai.response.model" response_model: string optional description: "Concrete response model",
                    "pi.ai.response.id" response_id: string optional description: "Provider response id",
                    "pi.ai.response.stop_reason" response_stop_reason as AiStopReason: string optional values [
                        Stop: "stop", Length: "length", ToolUse: "tool_use", Error: "error",
                        Aborted: "aborted", Deferred: "deferred",
                    ] description: "Normalized terminal response reason",
                    "pi.ai.http.status_code" http_status_code: number optional description: "Final HTTP status",
                    "pi.ai.usage.input_tokens" usage_input_tokens: number optional description: "Reported input tokens",
                    "pi.ai.usage.output_tokens" usage_output_tokens: number optional description: "Reported output tokens",
                    "pi.ai.usage.cache_read_tokens" usage_cache_read_tokens: number optional description: "Reported cache-read tokens",
                    "pi.ai.usage.cache_write_tokens" usage_cache_write_tokens: number optional description: "Reported cache-write tokens",
                    "pi.ai.usage.reasoning_tokens" usage_reasoning_tokens: number optional description: "Reported reasoning tokens",
                    "pi.ai.usage.total_tokens" usage_total_tokens: number optional description: "Reported total tokens",
                    "pi.ai.usage.cost" usage_cost: number optional description: "Reported total cost",
                    "pi.ai.stream.chunk_count" stream_chunk_count: number optional description: "Streamed update chunk count",
                    "pi.ai.stream.time_to_first_chunk_ms" stream_time_to_first_chunk_ms: number optional description: "Elapsed milliseconds to first update chunk",
                    "pi.ai.error.type" error_type: string optional description: "Provider or transport error class",
                },
                status: { default ok, error_when "The operation throws or returns an error result" },
            },
        },
    }
}

pi_telemetry::define_telemetry_schema! {
    /// The harness vocabulary, upstream's `HARNESS_TELEMETRY_SCHEMA`.
    pub schema harness_schema {
        version: 1,
        spans: {
            /// One admitted in-process run invocation, upstream's
            /// `"pi.harness.run"`.
            run => "One admitted in-process run invocation" {
                parents: [root_or_external],
                start_attributes: {
                    "pi.session.id" session_id: string required description: "Session id",
                    "pi.lane.name" lane_name: string required description: "Lane name",
                    "pi.operation.id" operation_id: string required description: "Durable operation id",
                    "pi.operation.recovery" operation_recovery: boolean required description: "Whether this invocation resumes durable work",
                    "pi.operation.kind" operation_kind as RunKind: string required values [Run: "run"] description: "Run operation kind",
                },
                end_attributes: {
                    "pi.operation.outcome" operation_outcome as RunOutcome: string optional values [
                        Completed: "completed", Aborted: "aborted", Failed: "failed", Suspended: "suspended",
                    ] description: "Run invocation outcome",
                    "pi.error.code" error_code: string optional description: "Stable operation error code",
                    "pi.error.type" error_type: string optional description: "Low-cardinality operation error class",
                },
                status: { default ok, error_when "The run fails or throws" },
            },
            /// One admitted in-process manual compaction invocation,
            /// upstream's `"pi.harness.compaction"`.
            compaction => "One admitted in-process manual compaction invocation" {
                parents: [root_or_external],
                start_attributes: {
                    "pi.session.id" session_id: string required description: "Session id",
                    "pi.lane.name" lane_name: string required description: "Lane name",
                    "pi.operation.id" operation_id: string required description: "Durable operation id",
                    "pi.operation.recovery" operation_recovery: boolean required description: "Whether this invocation resumes durable work",
                    "pi.operation.kind" operation_kind as CompactionKind: string required values [Compaction: "compaction"] description: "Compaction operation kind",
                },
                end_attributes: {
                    "pi.operation.outcome" operation_outcome as CompactionOutcome: string optional values [
                        Completed: "completed", Declined: "declined", Aborted: "aborted", Failed: "failed",
                    ] description: "Compaction invocation outcome",
                    "pi.error.code" error_code: string optional description: "Stable operation error code",
                    "pi.error.type" error_type: string optional description: "Low-cardinality operation error class",
                },
                status: { default ok, error_when "The compaction fails or throws" },
            },
            /// One admitted in-process navigation invocation, upstream's
            /// `"pi.harness.navigation"`.
            navigation => "One admitted in-process navigation invocation" {
                parents: [root_or_external],
                start_attributes: {
                    "pi.session.id" session_id: string required description: "Session id",
                    "pi.lane.name" lane_name: string required description: "Lane name",
                    "pi.operation.id" operation_id: string required description: "Durable operation id",
                    "pi.operation.recovery" operation_recovery: boolean required description: "Whether this invocation resumes durable work",
                    "pi.operation.kind" operation_kind as NavigationKind: string required values [Navigation: "navigation"] description: "Navigation operation kind",
                },
                end_attributes: {
                    "pi.operation.outcome" operation_outcome as NavigationOutcome: string optional values [
                        Completed: "completed", Declined: "declined", Aborted: "aborted", Failed: "failed",
                    ] description: "Navigation invocation outcome",
                    "pi.error.code" error_code: string optional description: "Stable operation error code",
                    "pi.error.type" error_type: string optional description: "Low-cardinality operation error class",
                },
                status: { default ok, error_when "The navigation fails or throws" },
            },
            /// One run checkpoint, upstream's `"pi.harness.checkpoint"`.
            checkpoint => "One run checkpoint" {
                parents: [spans "pi.harness.run"],
                start_attributes: {
                    "pi.lane.name" lane_name: string required description: "Lane name",
                    "pi.operation.id" operation_id: string required description: "Durable operation id",
                    "pi.checkpoint.kind" checkpoint_kind as CheckpointKind: string required values [
                        Normal: "normal", AbortReconcile: "abort_reconcile",
                    ] description: "Checkpoint purpose",
                },
                end_attributes: {},
                status: { default ok, error_when "Checkpoint work throws" },
            },
            /// One assistant response and its tool batch, upstream's
            /// `"pi.harness.turn"`.
            turn => "One assistant response and its tool batch" {
                parents: [spans "pi.harness.run"],
                start_attributes: {
                    "pi.lane.name" lane_name: string required description: "Lane name",
                    "pi.operation.id" operation_id: string required description: "Durable operation id",
                    "pi.turn.id" turn_id: string required description: "Invocation-local turn id",
                },
                end_attributes: {},
                status: { default ok, error_when "Turn work throws" },
            },
            /// One durable retry attempt, upstream's `"pi.harness.step"`.
            step => "One durable retry attempt" {
                parents: [spans "pi.harness.turn", "pi.harness.checkpoint", "pi.harness.compaction", "pi.harness.navigation"],
                start_attributes: {
                    "pi.lane.name" lane_name: string required description: "Lane name",
                    "pi.operation.id" operation_id: string required description: "Durable operation id",
                    "pi.step.kind" step_kind as StepKind: string required values [
                        Assistant: "assistant", Compaction: "compaction", BranchSummary: "branch_summary",
                    ] description: "Retryable step kind",
                    "pi.step.attempt" step_attempt: number required description: "One-based durable attempt number",
                    "pi.compaction.reason" compaction_reason as CompactionTrigger: string optional values [
                        Manual: "manual", Threshold: "threshold", Overflow: "overflow",
                    ] description: "Compaction trigger",
                },
                end_attributes: {
                    "pi.step.outcome" step_outcome as StepOutcome: string optional values [
                        Succeeded: "succeeded", Retry: "retry", Failed: "failed", Aborted: "aborted",
                        Deferred: "deferred", Overflow: "overflow",
                    ] description: "Attempt outcome",
                },
                status: { default ok, error_when "The attempt retries, fails, or throws" },
            },
            /// One raw phase-2 tool execution, upstream's
            /// `"pi.harness.tool"`.
            tool => "One raw phase-2 tool execution" {
                parents: [spans "pi.harness.turn", "pi.harness.run"],
                start_attributes: {
                    "pi.lane.name" lane_name: string required description: "Lane name",
                    "pi.operation.id" operation_id: string required description: "Durable operation id",
                    "pi.turn.id" turn_id: string optional description: "Invocation-local live turn id",
                    "pi.tool.name" tool_name: string required description: "Tool name",
                    "pi.tool.call_id" tool_call_id: string required description: "Tool call id",
                    "pi.tool.replay" tool_replay as ToolReplayPolicy: string required values [
                        Never: "never", Safe: "safe",
                    ] description: "Declared replay policy",
                    "pi.tool.recovery" tool_recovery: boolean required description: "Whether this is recovery execution",
                },
                end_attributes: {
                    "pi.tool.is_error" tool_is_error: boolean optional description: "Whether raw phase-2 execution returned an error",
                },
                status: { default ok, error_when "Raw phase-2 execution returns an error" },
            },
            /// One registered hook handler invocation, upstream's
            /// `"pi.harness.hook"`.
            hook => "One registered hook handler invocation" {
                parents: [any],
                start_attributes: {
                    "pi.lane.name" lane_name: string required description: "Lane name",
                    "pi.operation.id" operation_id: string optional description: "Durable operation id when accepted",
                    "pi.hook.name" hook_name as HookNameSpan: string required values [
                        BeforeRun: "before_run", BeforeDrive: "before_drive", BeforeRunEnd: "before_run_end",
                        TransformContext: "transform_context", BeforeRequest: "before_request",
                        BeforePayload: "before_payload", AfterResponse: "after_response",
                        BeforeTool: "before_tool", AfterTool: "after_tool",
                        BeforeCompaction: "before_compaction", BeforeNavigation: "before_navigation",
                    ] description: "Hook name",
                    "pi.hook.registration_id" hook_registration_id: string optional description: "Optional hook registration metadata",
                },
                end_attributes: {
                    "pi.hook.outcome" hook_outcome as HookOutcome: string optional values [
                        Completed: "completed", Skipped: "skipped", Blocked: "blocked", Failed: "failed",
                    ] description: "Handler outcome",
                },
                status: { default ok, error_when "The handler throws" },
            },
            /// One retry delay, upstream's `"pi.harness.sleep"`.
            sleep => "One retry delay" {
                parents: [spans "pi.harness.run", "pi.harness.compaction", "pi.harness.navigation", "pi.harness.turn", "pi.harness.checkpoint"],
                start_attributes: {
                    "pi.operation.id" operation_id: string required description: "Durable operation id",
                    "pi.sleep.delay_ms" sleep_delay_ms: number required description: "Requested delay in milliseconds",
                },
                end_attributes: {
                    "pi.sleep.outcome" sleep_outcome as SleepOutcome: string optional values [
                        Elapsed: "elapsed", Aborted: "aborted",
                    ] description: "Delay outcome",
                },
                status: { default ok, error_when "Sleep work throws" },
            },
            /// One passive event listener invocation, upstream's
            /// `"pi.harness.event_handler"`.
            event_handler => "One passive event listener invocation" {
                parents: [any],
                start_attributes: {
                    "pi.event.type" event_type as HarnessEventSpanType: string required values [
                        RunStart: "run_start", RunResume: "run_resume", RunSuspend: "run_suspend",
                        OperationAbort: "operation_abort", RunEnd: "run_end", Fault: "fault",
                        HandlerError: "handler_error", TurnStart: "turn_start", TurnEnd: "turn_end",
                        RetryScheduled: "retry_scheduled", RetryStart: "retry_start", RetryEnd: "retry_end",
                        MessageStart: "message_start", MessageUpdate: "message_update", MessageEnd: "message_end",
                        ToolStart: "tool_start", ToolUpdate: "tool_update", ToolEnd: "tool_end",
                        EntryAdded: "entry_added", QueueUpdate: "queue_update", ValueUpdate: "value_update",
                        ConfigUpdate: "config_update", CompactionStart: "compaction_start",
                        CompactionEnd: "compaction_end", NavigationStart: "navigation_start",
                        NavigationEnd: "navigation_end", LaneCreated: "lane_created", Usage: "usage",
                    ] description: "Delivered harness event type",
                    "pi.lane.name" lane_name: string optional description: "Lane name for lane-scoped events",
                },
                end_attributes: {},
                status: { default ok, error_when "The listener throws" },
            },
            /// One committed session transaction, upstream's
            /// `"pi.session.write"`.
            session_write => "One committed session transaction" {
                parents: [any],
                start_attributes: {
                    "pi.session.id" session_id: string required description: "Session id",
                    "pi.lane.name" lane_name: string optional description: "Lane name when supplied by the caller",
                    "pi.operation.id" operation_id: string optional description: "Durable operation id when supplied by the caller",
                    "pi.session.item_count" session_item_count: number required description: "Number of writes in the transaction",
                    "pi.session.item_kinds" session_item_kinds: string_array required description: "Distinct write kinds in the transaction",
                },
                end_attributes: {
                    "pi.session.first_seq" session_first_seq: number optional description: "First committed sequence in the transaction",
                    "pi.session.last_seq" session_last_seq: number optional description: "Last committed sequence in the transaction",
                },
                status: { default ok, error_when "Storage rejects the transaction" },
            },
        },
    }
}

/// The combined typed span vocabulary for agent-owned AI-request and
/// harness telemetry, upstream's `AGENT_TELEMETRY_SCHEMAS`.
///
/// The composed typed starter binds both schemas; `create_typed_span_starter!`
/// rejects duplicate span names across them at compile time, the port of
/// upstream's duplicate-schema check.
#[macro_export]
macro_rules! agent_telemetry_starter {
    ($context:expr) => {
        ::pi_telemetry::create_typed_span_starter!(
            $context,
            $crate::harness::telemetry::ai_schema::Schema,
            $crate::harness::telemetry::harness_schema::Schema
        )
    };
}

/// Starts an AI-request span with the typed start attributes, fetching the
/// telemetry parent from the harness context and handing the body the
/// span plus a context whose telemetry parent is the span, upstream's
/// `startAiSpan`.
///
/// # Errors
/// Propagates the body's `Err` value unchanged, after span settlement.
pub fn start_ai_span<N, T, E, Fut, F>(
    name: N,
    attributes: N::Start,
    body: F,
    context: &Context,
) -> impl Future<Output = Result<T, E>> + Send
where
    N: pi_telemetry::schema::SpanDefinition,
    F: FnOnce(TelemetrySpanView, Context) -> Fut + Send + 'static,
    Fut: Future<Output = Result<T, E>> + Send,
    T: Send + 'static,
    E: SpanBodyFailure + Send + 'static,
{
    start_typed_span(
        pi_telemetry::schema::SpanDefinition::NAME,
        attributes.into_span_attributes(),
        body,
        context,
    )
}

/// Starts a harness span, upstream's `startHarnessSpan`.
///
/// # Errors
/// Propagates the body's `Err` value unchanged, after span settlement.
pub fn start_harness_span<N, T, E, Fut, F>(
    name: N,
    attributes: N::Start,
    body: F,
    context: &Context,
) -> impl Future<Output = Result<T, E>> + Send
where
    N: pi_telemetry::schema::SpanDefinition,
    F: FnOnce(TelemetrySpanView, Context) -> Fut + Send + 'static,
    Fut: Future<Output = Result<T, E>> + Send,
    T: Send + 'static,
    E: SpanBodyFailure + Send + 'static,
{
    start_typed_span(
        pi_telemetry::schema::SpanDefinition::NAME,
        attributes.into_span_attributes(),
        body,
        context,
    )
}

/// The typed span view the harness start helpers hand their bodies: the
/// raw erased span handle, which records events and end attributes, plus
/// the child-context half the body threads onward.
pub type TelemetrySpanView = DynSpanHandle;

fn start_typed_span<T, E, Fut, F>(
    name: &str,
    attributes: SpanAttributes,
    body: F,
    context: &Context,
) -> impl Future<Output = Result<T, E>> + Send
where
    F: FnOnce(TelemetrySpanView, Context) -> Fut + Send + 'static,
    Fut: Future<Output = Result<T, E>> + Send,
    T: Send + 'static,
    E: SpanBodyFailure + Send + 'static,
{
    let parent = get_telemetry_context(context);
    let outer_context = context.clone();
    parent.start_span(
        SpanOptions::new(name).with_attributes(attributes),
        move |span| {
            let child_context = with_telemetry_context(span.telemetry_handle(), &outer_context);
            async move { body(span, child_context).await }
        },
    )
}

#[cfg(test)]
mod tests;
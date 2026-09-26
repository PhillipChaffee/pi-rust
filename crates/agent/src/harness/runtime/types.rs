//! The runtime command vocabulary, the drive pass, and the lane's owned
//! state, ported from upstream `src/harness/runtime/types.ts`.
//!
//! Upstream parameterizes `Config<TContext>` over the application context;
//! the port carries the erased tool context (`ToolContext`), the same
//! statement the harness-foundations child made for `AgentHarnessOptions`.
//! `Drive.completion`'s one-shot settlement restates as a watch channel:
//! multiple parties await one drive pass (the claiming caller, an occupied
//! rival, `waitForIdle`), where JS promises are multi-consumer by
//! construction.
//!
//! The drive-pass procedure loop itself (`driveOperation`) rides the drive
//! child; [`Drive`] carries its completion and gate surface so the lane can
//! install and observe passes now.

use std::any::Any;
use std::sync::Arc;

use pi_ai::utils::retry::RetryPolicy;
use pi_chord::context::AbortController;
use pi_chord::context::Context;

use crate::harness::agent_harness::DriveOptions;
use crate::harness::agent_harness::DriveOutcome;
use crate::harness::compaction::types::CompactionSettings;
use crate::harness::context::without_abort_signal;
use crate::harness::gate::Gate;
use crate::harness::gate::GateControl;
use crate::harness::session::types::InboxItem;
use crate::harness::session::types::LaneConfiguration;
use crate::harness::session::types::OperationMeta;
use crate::harness::session::types::OperationResultRecord;
use crate::harness::session::types::OperationState;
use crate::harness::session::values::Write;
use crate::harness::types::AgentHarnessStreamOptions;
use crate::harness::types::AgentHarnessTool;
use crate::harness::types::AgentHarnessToolContextSource;
use crate::types::QueueMode;
use crate::types::ToolExecutionMode;

/// The boxed error the lane's closures carry, upstream's `Error` values —
/// `Arc`-backed so a stored error (the seal, a rejected drive pass)
/// rethrows by clone, which JS error objects are by reference.
pub type LaneError = Arc<dyn std::error::Error + Send + Sync>;

/// Boxes one concrete error as the lane's error type.
#[must_use]
pub fn lane_error(error: impl std::error::Error + Send + Sync + 'static) -> LaneError {
    Arc::new(error)
}

/// Coerces one concrete error's `Arc` to the erased lane error, the
/// unsizing statement the `Arc::from(Box::new(..))` casts made upstream.
#[must_use]
pub fn from_arc<E: std::error::Error + Send + Sync + 'static>(error: Arc<E>) -> LaneError {
    error
}

/// Boxes one payload as the mutation contract's erased value, upstream's
/// `Box<dyn Any + Send>` return.
pub(crate) fn any_payload<T: Send + 'static>(value: T) -> Box<dyn Any + Send> {
    Box::new(value)
}

/// A method or seam a later harness slice implements, upstream's
/// `SliceNotImplemented` in `runtime/types.ts`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceNotImplemented {
    /// The operation the caller reached.
    pub operation: String,
}

impl SliceNotImplemented {
    /// Builds the error for one operation name.
    #[must_use]
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
        }
    }
}

impl std::fmt::Display for SliceNotImplemented {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} is not implemented until its later AgentHarness slice",
            self.operation
        )
    }
}

impl std::error::Error for SliceNotImplemented {}

/// Current process-local harness configuration, upstream's `Config`.
#[derive(Clone)]
pub struct Config {
    /// The harness tools.
    pub tools: Vec<AgentHarnessTool>,
    /// The resources.
    pub resources: crate::harness::agent_harness::Resources,
    /// The curated stream options.
    pub stream_options: AgentHarnessStreamOptions,
    /// The retry policy.
    pub retry_policy: RetryPolicy,
    /// The compaction settings.
    pub compaction: CompactionSettings,
    /// The steering queue mode.
    pub steering_mode: QueueMode,
    /// The follow-up queue mode.
    pub follow_up_mode: QueueMode,
    /// The tool execution mode.
    pub tool_execution: ToolExecutionMode,
    /// The static tool context or provider, upstream's `toolContext`.
    pub tool_context: Option<AgentHarnessToolContextSource>,
    /// The system prompt, static or per-turn, upstream's `systemPrompt`.
    pub system_prompt: Option<crate::harness::agent_harness::SystemPromptSource>,
    /// The transcript-to-provider conversion, upstream's
    /// `toProviderMessages`.
    pub to_provider_messages: crate::harness::agent_harness::ToProviderMessages,
    /// The custom-entry projectors, by custom type, upstream's
    /// `entryProjectors`.
    pub entry_projectors: std::collections::BTreeMap<String, crate::harness::session::types::EntryProjector>,
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("tools", &self.tools.len())
            .field("resources", &self.resources)
            .field("stream_options", &self.stream_options)
            .field("retry_policy", &self.retry_policy)
            .field("compaction", &self.compaction)
            .field("steering_mode", &self.steering_mode)
            .field("follow_up_mode", &self.follow_up_mode)
            .field("tool_execution", &self.tool_execution)
            .finish_non_exhaustive()
    }
}

/// The current durable state owned by one lane, upstream's `LaneState` in
/// `runtime/types.ts` (the session contract's durable lane record restates
/// as [`crate::harness::session::types::LaneState`]).
#[derive(Clone, Debug, PartialEq)]
pub struct LaneState {
    /// The branch tip id, when any entries committed.
    pub tip_id: Option<String>,
    /// The lane configuration.
    pub configuration: LaneConfiguration,
    /// The queued inbox items.
    pub inbox: Vec<InboxItem>,
    /// The last settled operation id, when any.
    pub last_operation_id: Option<String>,
    /// The live operation, when one is admitted.
    pub operation: Option<LiveOperation>,
}

/// One admitted operation's meta plus durable state, upstream's
/// `LaneState["operation"]`.
#[derive(Clone, Debug, PartialEq)]
pub struct LiveOperation {
    /// The durable meta.
    pub meta: OperationMeta,
    /// The durable state leaf.
    pub state: OperationState,
}

/// The lane fields a durable patch replaces, upstream's `LanePatch`.
///
/// `None` leaves the field as-is; `Some` replaces it. The tip id is
/// doubly-optional: the outer `Option` selects replacement, the inner
/// carries the value (a branch root's `None`).
#[derive(Clone, Debug, Default)]
pub struct LanePatch {
    /// Replace the tip id.
    pub tip_id: Option<Option<String>>,
    /// Replace the configuration.
    pub configuration: Option<LaneConfiguration>,
    /// Replace the inbox.
    pub inbox: Option<Vec<InboxItem>>,
}

/// Materializes the caller's result from storage-assigned commit metadata,
/// upstream's `materialize(commit)` — synchronous by construction, the
/// port's type system replacing upstream's runtime thenable guard.
pub type MaterializeFn<T> = Arc<dyn Fn(&crate::harness::session::types::CommitResult) -> T + Send + Sync>;

/// Builds the events a commit publishes, upstream's `events?(commit)`.
pub type EventsFn = Arc<dyn Fn(&crate::harness::session::types::CommitResult) -> Vec<crate::harness::agent_harness::HarnessEvent> + Send + Sync>;

/// One effect-free decision made on a lane's serialized mutation line,
/// upstream's `LaneCommand<TResult>`.
pub enum LaneCommand<T> {
    /// Commit once and materialize the result, upstream's `commit`.
    Commit {
        /// The transaction's writes.
        writes: Vec<Write>,
        /// The lane state published after the commit.
        next: LaneState,
        /// The synchronous result materialization.
        materialize: MaterializeFn<T>,
        /// The commit's events, when any.
        events: Option<EventsFn>,
    },
    /// Return without a commit, upstream's `return`.
    Return {
        /// The caller's result.
        result: T,
    },
    /// Reject outside the mutation/fault boundary as an expected caller
    /// error, upstream's `reject`.
    Reject {
        /// The caller's error.
        error: LaneError,
    },
}

/// The result an operation continuation yields, upstream's
/// `ContinueOperationResult<TResult>`.
pub enum ContinueOperationResult<T> {
    /// Cancellation was requested; the planner never ran.
    CancelRequested,
    /// The planner's materialized result.
    Result {
        /// The materialized value.
        value: T,
    },
}

/// One durable operation transition, upstream's `OperationCommand<TResult>`.
pub enum OperationCommand<T> {
    /// Commit once with the operation-state write, upstream's `commit`.
    Commit {
        /// The transaction's writes.
        writes: Vec<Write>,
        /// The new durable operation state.
        operation_state: OperationState,
        /// The lane fields to patch.
        lane: Option<LanePatch>,
        /// The synchronous result materialization.
        materialize: MaterializeFn<T>,
        /// The commit's events, when any.
        events: Option<EventsFn>,
    },
    /// Finish the operation: record its result, clear it from the lane,
    /// upstream's `finish`.
    Finish {
        /// The transaction's writes.
        writes: Vec<Write>,
        /// The settled result record.
        record: OperationResultRecord,
        /// The lane fields to patch.
        lane: Option<LanePatch>,
        /// The synchronous result materialization.
        materialize: MaterializeFn<T>,
        /// The commit's events, when any.
        events: Option<EventsFn>,
    },
    /// Return without a commit, upstream's `return`.
    Return {
        /// The caller's result.
        result: T,
    },
}

/// The result one drive procedure yields, upstream's `ProcedureResult`.
pub enum ProcedureResult {
    /// The drive continues with the next phase.
    Continue,
    /// The drive waits on a caller decision.
    Waiting {
        /// The waiting outcome.
        outcome: DriveOutcome,
    },
    /// The operation settled.
    Settled {
        /// The settled record.
        outcome: OperationResultRecord,
    },
}

/// The completion state one drive pass settles into, upstream's
/// `Promise<DriveOutcome>` — pending until [`Drive::settle`] or
/// [`Drive::fail`] fires, each no-op after the first.
pub enum DriveCompletion {
    /// No outcome yet.
    Pending,
    /// The pass settled.
    Settled(DriveOutcome),
    /// The pass failed.
    Failed(LaneError),
}

impl Clone for DriveCompletion {
    fn clone(&self) -> Self {
        match self {
            Self::Pending => Self::Pending,
            Self::Settled(outcome) => Self::Settled(outcome.clone()),
            Self::Failed(error) => Self::Failed(Arc::clone(error)),
        }
    }
}

impl std::fmt::Debug for DriveCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => f.write_str("Pending"),
            Self::Settled(outcome) => f.debug_tuple("Settled").field(outcome).finish(),
            Self::Failed(error) => f.debug_tuple("Failed").field(&error.to_string()).finish(),
        }
    }
}

/// One installed process-local drive pass, upstream's `Drive`.
pub struct Drive {
    /// The operation the pass drives, upstream's `operationId`.
    pub operation_id: String,
    /// The pass's invocation context with the abort signal stripped,
    /// upstream's `context` (built through `withoutAbortSignal`).
    pub context: Context,
    /// Whether the pass resolves a retry wait by waiting it out, upstream's
    /// `waitForRetry`.
    pub wait_for_retry: bool,
    /// The deferred polls the pass may resolve, upstream's
    /// `deferredPermits`.
    pub deferred_permits: u32,
    /// The pass's procedure-facing gate, upstream's `gate`.
    pub gate: Gate,
    /// The completion all awaiters share, upstream's `completion`.
    completion: tokio::sync::watch::Receiver<DriveCompletion>,
    completion_tx: tokio::sync::watch::Sender<DriveCompletion>,
    /// The pass's close signal, upstream's `closeSignal`.
    pub close_signal: pi_chord::context::AbortSignal,
    control: GateControl,
    close_controller: AbortController,
}

impl std::fmt::Debug for Drive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Drive")
            .field("operation_id", &self.operation_id)
            .field("wait_for_retry", &self.wait_for_retry)
            .field("deferred_permits", &self.deferred_permits)
            .finish_non_exhaustive()
    }
}

impl Drive {
    /// Builds one drive pass, upstream's `constructor`.
    #[must_use]
    pub fn new(options: &DriveOptions, context: &Context) -> Self {
        let (completion_tx, completion) = tokio::sync::watch::channel(DriveCompletion::Pending);
        let (gate, control) = crate::harness::gate::create_gate();
        let (_, close_controller) = pi_chord::context::with_cancel(&pi_chord::context::background_context());
        Self {
            operation_id: options.operation_id.clone(),
            context: without_abort_signal(context),
            wait_for_retry: options.wait_for_retry.unwrap_or(false),
            deferred_permits: u32::from(options.poll_deferred.unwrap_or(false)),
            gate,
            completion,
            completion_tx,
            close_signal: close_controller.signal().clone(),
            control,
            close_controller,
        }
    }

    /// Settles the pass, upstream's `settle`; a later [`Self::fail`] is a
    /// no-op, matching the promise's one-shot settlement.
    pub fn settle(&self, outcome: DriveOutcome) {
        self.complete(DriveCompletion::Settled(outcome));
    }

    /// Fails the pass, upstream's `fail`; a later [`Self::settle`] is a
    /// no-op.
    pub fn fail(&self, error: LaneError) {
        self.complete(DriveCompletion::Failed(error));
    }

    fn complete(&self, completion: DriveCompletion) {
        let current = self.completion_tx.borrow().clone();
        if matches!(current, DriveCompletion::Pending) {
            let _ = self.completion_tx.send(completion);
        }
    }

    /// Resolves when the pass settles or fails, upstream's awaiting
    /// `completion`.
    pub async fn completion(&self) -> Result<DriveOutcome, Arc<dyn std::error::Error + Send + Sync>> {
        let mut receiver = self.completion.clone();
        loop {
            if let DriveCompletion::Settled(outcome) = receiver.borrow_and_update().clone() {
                return Ok(outcome);
            }
            if let DriveCompletion::Failed(error) = receiver.borrow_and_update().clone() {
                return Err(error);
            }
            if receiver.changed().await.is_err() {
                // The pass outlives every awaiter in practice; a dropped
                // sender cannot carry an outcome, so the wait cannot recover.
                unreachable!("drive completion sender dropped before settlement")
            }
        }
    }

    /// Records the abort's cancellation on the gate, upstream's
    /// `beginAbort`.
    pub fn begin_abort(&self, cancellation: crate::harness::gate::Cancellation) {
        self.control.begin_abort(cancellation);
    }

    /// Fires the gate's signal, upstream's `signalAbort`.
    pub fn signal_abort(&self) {
        self.control.signal_abort();
    }

    /// Closes the gate and the pass, upstream's `closeGate`.
    pub fn close_gate(&self, error: LaneError) {
        self.control.close(error.to_string());
        if !self.close_signal.aborted() {
            self.close_controller.abort(error.to_string());
        }
        self.fail(error);
    }
}

#[cfg(test)]
mod tests;
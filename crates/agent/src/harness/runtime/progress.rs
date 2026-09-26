//! The progress channels an in-flight effect publishes through, ported from
//! upstream `src/harness/runtime/progress.ts`.
//!
//! Upstream's channel chains each write's command onto `latest`, so
//! `drain()` awaits the newest write and the lane's mutation line keeps the
//! order. The port spawns each write's command on write and records the
//! newest write's settlement in a watch channel — multi-consumer awaiting,
//! which JS promises are by construction. Write failures stay retained in
//! the settlement (upstream's `latest` rejection) and surface through
//! `drain`.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use pi_ai::types::BoxedFuture;
use pi_ai::utils::assistant_message_frame::AssistantMessageFrame;

use crate::harness::context::Context;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneCommand;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::LaneState;
use crate::harness::session::types::CommitResult;
use crate::harness::session::types::EntryScanOrder;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::ToolCallStatus;
use crate::harness::session::values::ListCursor;
use crate::harness::session::values::ListElement;
use crate::harness::session::values::ListReadOptions;
use crate::harness::session::values::ToolOutputPayload;
use crate::harness::session::values::Write;
use crate::harness::session::values::append_list;
use crate::harness::session::values::pending_assistant_frames;
use crate::harness::session::values::pending_tool_output;
use crate::harness::session::values::set_value;
use crate::types::AgentToolResult;

/// The error a write's settlement carries — the shared, cloneable restatement
/// of upstream's rejected write promise; the lane error type itself.
pub type WriteFailure = LaneError;

/// The write surface one in-flight effect publishes through, upstream's
/// `ProgressChannel<T>`.
pub struct ProgressChannel<T> {
    write: Arc<dyn Fn(T) + Send + Sync>,
    seal: Arc<dyn Fn() + Send + Sync>,
    drain: Arc<dyn Fn() -> BoxedFuture<'static, Result<(), WriteFailure>> + Send + Sync>,
    _marker: std::marker::PhantomData<fn(T)>,
}

impl<T> std::fmt::Debug for ProgressChannel<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProgressChannel(..)")
    }
}

impl<T> ProgressChannel<T> {
    /// Publishes one item, upstream's `write`; calls after [`Self::seal`]
    /// drop.
    pub fn write(&self, item: T) {
        (self.write)(item);
    }

    /// Stops admission, upstream's `seal`.
    pub fn seal(&self) {
        (self.seal)();
    }

    /// Awaits the newest write's settlement, upstream's `drain`.
    ///
    /// # Errors
    /// The newest write's commit failure, retained from the write.
    pub async fn drain(&self) -> Result<(), WriteFailure> {
        (self.drain)().await
    }
}

/// Reads every frame one response entry holds, upstream's
/// `readAssistantFrames`: paged ascends of up to 1,000 elements.
pub async fn read_assistant_frames(
    reader: &dyn SessionReader,
    operation_id: &str,
    response_entry_id: &str,
    context: &Context,
) -> Result<Vec<AssistantMessageFrame>, SessionError> {
    const PAGE_LIMIT: u64 = 1_000;
    let mut frames: Vec<AssistantMessageFrame> = Vec::new();
    let mut cursor: Option<ListCursor> = None;
    loop {
        let page = reader
            .read_list(
                &pending_assistant_frames(operation_id, response_entry_id).address,
                Some(ListReadOptions {
                    cursor,
                    order: Some(EntryScanOrder::Asc),
                    limit: Some(PAGE_LIMIT),
                }),
                context,
            )
            .await?;
        let page_len = u64::try_from(page.len()).unwrap_or(PAGE_LIMIT);
        let mut page_cursor = None;
        for ListElement { seq, value } in page {
            frames.push(
                serde_json::from_value(value)
                    .map_err(|error| SessionError(format!("Pending assistant frame is malformed: {error}")))?,
            );
            page_cursor = Some(ListCursor { seq });
        }
        if page_len < PAGE_LIMIT {
            return Ok(frames);
        }
        cursor = page_cursor;
    }
}

/// One write's settlement, upstream's `latest` promise state.
type WriteSettlement = Option<Result<(), WriteFailure>>;

/// The shared channel state the write, seal, and drain closures share.
struct ChannelShared {
    sealed: AtomicBool,
    latest: Mutex<Option<tokio::sync::watch::Receiver<WriteSettlement>>>,
}

fn lock_latest(shared: &ChannelShared) -> MutexGuard<'_, Option<tokio::sync::watch::Receiver<WriteSettlement>>> {
    shared
        .latest
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// The shared progress machinery, upstream's `openProgress`.
fn open_progress<T: Send + 'static>(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    commit_write: Arc<dyn Fn(&T) -> Result<Write, SessionError> + Send + Sync>,
    still_owns: Arc<dyn Fn(&LaneState) -> bool + Send + Sync>,
) -> ProgressChannel<T> {
    let shared = Arc::new(ChannelShared {
        sealed: AtomicBool::new(false),
        latest: Mutex::new(None),
    });
    let lane = Arc::clone(lane);
    let drive = Arc::clone(&drive);
    let write_shared = Arc::clone(&shared);
    let write_fn = Arc::new(move |item: T| {
        if write_shared.sealed.load(Ordering::SeqCst) {
            return;
        }
        let lane = Arc::clone(&lane);
        let drive = Arc::clone(&drive);
        let still_owns = Arc::clone(&still_owns);
        let commit_write = Arc::clone(&commit_write);
        let (settlement_tx, settlement_rx) = tokio::sync::watch::channel(None);
        {
            let mut guard = lock_latest(&write_shared);
            *guard = Some(settlement_rx);
        }
        tokio::spawn(async move {
            let outcome = lane
                .command(
                    move |state, _session: Arc<dyn crate::harness::session::types::Session>, _context| {
                        if !still_owns(&state) {
                            return Box::pin(async move {
                                Ok(LaneCommand::Return { result: () })
                            })
                                as BoxedFuture<'static, Result<LaneCommand<()>, LaneError>>;
                        }
                        match commit_write(&item) {
                            Ok(write) => Box::pin(async move {
                                Ok(LaneCommand::Commit {
                                    writes: vec![write],
                                    next: state,
                                    materialize: Arc::new(|_: &CommitResult| ()),
                                    events: None,
                                })
                            }),
                            Err(error) => Box::pin(async move { Err(lane_error(error)) }),
                        }
                    },
                    &drive.context,
                )
                .await;
            let _ = settlement_tx.send(Some(outcome.map_err(Arc::from)));
        });
    });
    let seal_shared = Arc::clone(&shared);
    let seal_fn = Arc::new(move || {
        seal_shared.sealed.store(true, Ordering::SeqCst);
    });
    let drain_shared = Arc::clone(&shared);
    let drain_fn = Arc::new(move || {
        let drain_shared = Arc::clone(&drain_shared);
        Box::pin(async move {
            let receiver = {
                let guard = lock_latest(&drain_shared);
                guard.as_ref().map(tokio::sync::watch::Receiver::clone)
            };
            let Some(mut receiver) = receiver else {
                // No write has been published; upstream's `latest` is the
                // resolved seed promise.
                return Ok(());
            };
            loop {
                {
                    let state = receiver.borrow_and_update();
                    if let Some(outcome) = state.clone() {
                        return outcome;
                    }
                }
                if receiver.changed().await.is_err() {
                    return Ok(());
                }
            }
        }) as BoxedFuture<'static, Result<(), WriteFailure>>
    });
    ProgressChannel {
        write: write_fn,
        seal: seal_fn,
        drain: drain_fn,
        _marker: std::marker::PhantomData,
    }
}

/// The frame progress one assistant response writes through, upstream's
/// `openFrameProgress`.
#[must_use]
pub fn open_frame_progress(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    response_entry_id: &str,
) -> ProgressChannel<AssistantMessageFrame> {
    let address = pending_assistant_frames(&drive.operation_id, response_entry_id);
    let response_entry_id = response_entry_id.to_owned();
    open_progress(
        lane,
        drive,
        Arc::new(move |frame: &AssistantMessageFrame| {
            append_list(&address, frame.clone()).map(Write::ListAppend)
        }),
        Arc::new(move |state| {
            let Some(operation) = &state.operation else {
                return false;
            };
            match &operation.state {
                OperationState::AssistantEffectPending(leaf) => leaf.response_entry_id == response_entry_id,
                OperationState::DeferredEffectPending(leaf) => leaf.response_entry_id == response_entry_id,
                _ => false,
            }
        }),
    )
}

/// The checkpoint progress one tool call writes through, upstream's
/// `openToolProgress`.
#[must_use]
pub fn open_tool_progress(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    turn_id: &str,
    source_index: u64,
    invocation_id: &str,
) -> ProgressChannel<AgentToolResult> {
    let address = pending_tool_output(&drive.operation_id, invocation_id);
    let turn_id = turn_id.to_owned();
    open_progress(
        lane,
        drive,
        Arc::new(move |snapshot: &AgentToolResult| {
            let payload = ToolOutputPayload {
                content: snapshot.content.clone(),
                details: snapshot.details.clone(),
                usage: snapshot.usage.clone(),
                added_tool_names: snapshot.added_tool_names.clone(),
                terminate: snapshot.terminate,
            };
            set_value(&address, payload).map(Write::ValueSet)
        }),
        Arc::new(move |state| {
            let Some(operation) = &state.operation else {
                return false;
            };
            let OperationState::Tools(leaf) = &operation.state else {
                return false;
            };
            leaf.batch.turn_id == turn_id
                && leaf.batch.calls.iter().any(|call| {
                    call.source_index == source_index
                        && call.result_entry_id == invocation_id
                        && matches!(call.status(), ToolCallStatus::EffectPending { .. })
                })
        }),
    )
}
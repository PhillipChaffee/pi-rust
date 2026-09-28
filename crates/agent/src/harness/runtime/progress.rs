//! The progress channels an in-flight effect publishes through, ported from
//! upstream `src/harness/runtime/progress.ts`.
//!
//! Upstream's `write` calls `lane.command` synchronously, so the mutation
//! line enqueues in call order, and replaces `latest` with the newest
//! write's promise, so `drain()` awaits the newest write. The port's
//! enqueue happens when the write's task polls, which a multi-thread
//! runtime schedules in any order, so each write's task first awaits its
//! predecessor's settlement — the chain restores the call-order enqueue on
//! every runtime flavor — and the newest write's settlement rides a watch
//! channel, which serves `drain`'s newest-write await with the
//! multi-consumer semantics JS promises have by construction. Write
//! failures stay retained in the settlement (upstream's `latest`
//! rejection) and surface through `drain`; a vanished predecessor's
//! dropped sender releases the chain, upstream's `.catch(() => {})`
//! keeping the tail alive across any failure.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError, atomic::AtomicBool, atomic::Ordering};

use pi_ai::types::BoxedFuture;
use pi_ai::utils::assistant_message_frame::AssistantMessageFrame;

use crate::harness::context::Context;
use crate::harness::runtime::lane::Lane;
use crate::harness::runtime::types::Drive;
use crate::harness::runtime::types::LaneCommand;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::LaneState;
use crate::harness::runtime::types::lane_error;
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
///
/// # Errors
/// The list read's storage error; a malformed stored frame.
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
            frames.push(serde_json::from_value(value).map_err(|error| {
                SessionError::Message(format!("Pending assistant frame is malformed: {error}"))
            })?);
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

/// The shared plan closure that turns one published item into the write to
/// commit, upstream's `commit` argument to `openProgress`.
type CommitWrite<T> = Arc<dyn Fn(&T) -> Result<Write, SessionError> + Send + Sync>;

/// The shared channel state the write, seal, and drain closures share.
struct ChannelShared {
    sealed: AtomicBool,
    latest: Mutex<Option<tokio::sync::watch::Receiver<WriteSettlement>>>,
}

fn lock_latest(
    shared: &ChannelShared,
) -> MutexGuard<'_, Option<tokio::sync::watch::Receiver<WriteSettlement>>> {
    shared.latest.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The shared progress machinery, upstream's `openProgress`. `T` is `Sync`
/// because the plan closure rides the lane's `Arc`-wrapped mutation
/// contract, which shares it across calls.
fn open_progress<T: Send + Sync + 'static>(
    lane: &Arc<Lane>,
    drive: &Arc<Drive>,
    commit_write: CommitWrite<T>,
    still_owns: Arc<dyn Fn(&LaneState) -> bool + Send + Sync>,
) -> ProgressChannel<T> {
    let shared = Arc::new(ChannelShared {
        sealed: AtomicBool::new(false),
        latest: Mutex::new(None),
    });
    let lane = Arc::clone(lane);
    let drive = Arc::clone(drive);
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
        // The predecessor's receiver and this write's own swap under one
        // lock, so the chain follows the write call order — two rapid
        // writes must not race the swap on a multi-thread runtime,
        // upstream's synchronous read-then-assign of `latest`.
        let previous = {
            let mut guard = lock_latest(&write_shared);
            let previous = guard.take();
            *guard = Some(settlement_rx);
            previous
        };
        tokio::spawn(async move {
            // The enqueue waits on the predecessor's settlement, so the
            // mutation line sees the writes in call order on any runtime
            // flavor, upstream's synchronous `lane.command` enqueue; a
            // vanished predecessor's dropped sender releases the wait,
            // upstream's `.catch(() => {})` keeping the tail alive across
            // any failure.
            if let Some(mut previous) = previous {
                loop {
                    {
                        let settled = previous.borrow_and_update();
                        if settled.is_some() {
                            break;
                        }
                    }
                    if previous.changed().await.is_err() {
                        break;
                    }
                }
            }
            let outcome = lane
                .command(
                    move |state,
                          _session: Arc<dyn crate::harness::session::types::Session>,
                          _context| {
                        if !still_owns(&state) {
                            let skip: BoxedFuture<'static, Result<LaneCommand<()>, LaneError>> =
                                Box::pin(async move { Ok(LaneCommand::Return { result: () }) });
                            return skip;
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
            let _ = settlement_tx.send(Some(outcome));
        });
    });
    let seal_shared = Arc::clone(&shared);
    let seal_fn = Arc::new(move || {
        seal_shared.sealed.store(true, Ordering::SeqCst);
    });
    let drain_shared = Arc::clone(&shared);
    let drain_fn = Arc::new(move || {
        let drain_shared = Arc::clone(&drain_shared);
        let drain: BoxedFuture<'static, Result<(), WriteFailure>> = Box::pin(async move {
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
        });
        drain
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
                OperationState::AssistantEffectPending(leaf) => {
                    leaf.response_entry_id == response_entry_id
                }
                OperationState::DeferredEffectPending(leaf) => {
                    leaf.response_entry_id == response_entry_id
                }
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
    let invocation_id = invocation_id.to_owned();
    open_progress(
        lane,
        drive,
        Arc::new(move |snapshot: &AgentToolResult| {
            let payload = ToolOutputPayload {
                content: snapshot.content.clone(),
                details: snapshot.details.clone(),
                usage: snapshot.usage,
                added_tool_names: snapshot.added_tool_names.clone(),
                terminate: snapshot.terminate,
            };
            set_value(&address, payload).map(Write::ValueSet)
        }),
        Arc::new(move |state: &LaneState| {
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

#[cfg(test)]
mod tests;

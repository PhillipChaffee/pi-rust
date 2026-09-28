//! The terminal-transaction helpers the owning procedures share, ported from
//! upstream `src/harness/runtime/drive/terminal.ts`.
//!
//! `operation_cleanup_writes`' write order fixes the commit's sequence
//! numbers the event builders read, so the four scanned families keep
//! upstream's exact order and `pendingIds` iterates in insertion order
//! (upstream's `Set`), deduplicated.

use crate::harness::context::Context;
use crate::harness::runtime::clock::now_ms;
use crate::harness::runtime::lane::intent_kind_of;
use crate::harness::runtime::types::LaneError;
use crate::harness::runtime::types::lane_error;
use crate::harness::session::types::OperationError;
use crate::harness::session::types::OperationMeta;
use crate::harness::session::types::OperationResultRecord;
use crate::harness::session::types::OperationState;
use crate::harness::session::types::SessionError;
use crate::harness::session::types::SessionReader;
use crate::harness::session::types::TerminalStatus;
use crate::harness::session::types::ToolCallStatus;
use crate::harness::session::values::StoredValue;
use crate::harness::session::values::ValueDeleteWrite;
use crate::harness::session::values::Write;
use crate::harness::session::values::delete_list;
use crate::harness::session::values::delete_value_write;
use crate::harness::session::values::operation_meta;
use crate::harness::session::values::operation_preparation_prefix;
use crate::harness::session::values::operation_state as operation_state_value;
use crate::harness::session::values::operation_tool_args_prefix;
use crate::harness::session::values::operation_tool_memo_prefix;
use crate::harness::session::values::pending_assistant_frames;
use crate::harness::session::values::pending_entry;
use crate::harness::session::values::pending_tool_output_prefix;

/// The value-delete write one scanned stored value removes, upstream's
/// `({ address }) => deleteValue(address)` projection (the stored view
/// carries the address as its `namespace`/`key` fields).
fn delete_stored_value(value: &StoredValue) -> Write {
    Write::ValueDelete(ValueDeleteWrite {
        kind: "value".to_owned(),
        op: "delete".to_owned(),
        namespace: value.namespace.clone(),
        key: value.key.clone(),
    })
}

/// Build the mechanical operation-owned suffix used by an owning procedure's
/// terminal transaction, upstream's `operationCleanupWrites`.
///
/// The four scans run concurrently (upstream's `Promise.all`); only the
/// result order is contractual — the tool-argument, tool-memo, preparation,
/// and tool-output families follow the operation's own meta/state deletes in
/// that order, then the live frame-list delete (effect-pending states only),
/// then the staged tool outcomes' pending-entry deletes (`outcome_ready`
/// calls only, upstream's `Set` insertion order).
///
/// # Errors
/// Any of the four prefix scans' storage errors.
pub(crate) async fn operation_cleanup_writes(
    reader: &dyn SessionReader,
    operation_id: &str,
    state: &OperationState,
    context: &Context,
) -> Result<Vec<Write>, LaneError> {
    let (tool_arguments, tool_memos, preparations, tool_outputs) = tokio::join!(
        reader.scan_values(
            &operation_tool_args_prefix(operation_id, None).address,
            context
        ),
        reader.scan_values(
            &operation_tool_memo_prefix(operation_id, None).address,
            context
        ),
        reader.scan_values(&operation_preparation_prefix(operation_id).address, context),
        reader.scan_values(&pending_tool_output_prefix(operation_id).address, context),
    );
    let tool_arguments = tool_arguments.map_err(lane_error)?;
    let tool_memos = tool_memos.map_err(lane_error)?;
    let preparations = preparations.map_err(lane_error)?;
    let tool_outputs = tool_outputs.map_err(lane_error)?;

    // The staged tool outcomes still under pending entries; completed
    // results keep their entries.
    let mut pending_ids: Vec<String> = Vec::new();
    if let OperationState::Tools(leaf) = state {
        for call in &leaf.batch.calls {
            if matches!(call.status(), ToolCallStatus::OutcomeReady { .. })
                && !pending_ids.contains(&call.result_entry_id)
            {
                pending_ids.push(call.result_entry_id.clone());
            }
        }
    }
    let response_entry_id = match state {
        OperationState::AssistantEffectPending(leaf) => Some(&leaf.response_entry_id),
        OperationState::DeferredEffectPending(leaf) => Some(&leaf.response_entry_id),
        _ => None,
    };
    let frame_delete = response_entry_id.map(|response_entry_id| {
        Write::ListDelete(delete_list(&pending_assistant_frames(
            operation_id,
            response_entry_id,
        )))
    });

    let mut writes = vec![
        delete_value_write(&operation_meta(operation_id)),
        delete_value_write(&operation_state_value(operation_id)),
    ];
    writes.extend(tool_arguments.iter().map(delete_stored_value));
    writes.extend(tool_memos.iter().map(delete_stored_value));
    writes.extend(preparations.iter().map(delete_stored_value));
    writes.extend(tool_outputs.iter().map(delete_stored_value));
    if let Some(frame_delete) = frame_delete {
        writes.push(frame_delete);
    }
    writes.extend(
        pending_ids
            .iter()
            .map(|id| delete_value_write(&pending_entry(id))),
    );
    Ok(writes)
}

/// Construct the immutable observation record for one terminal decision,
/// upstream's `operationResultRecord`.
///
/// `endedAt` reads the clock once, here; the committing procedure reuses the
/// record's field for its `run_end` event so both carry the same instant.
///
/// # Errors
/// The [`SessionError::Invariant`] `"Only a failed operation result may
/// carry an error"` when exactly one of `status === failed` and
/// `error` holds — the invariant binds in both directions.
pub(crate) fn operation_result_record(
    meta: &OperationMeta,
    status: TerminalStatus,
    tip_id: Option<String>,
    error: Option<OperationError>,
) -> Result<OperationResultRecord, LaneError> {
    if matches!(status, TerminalStatus::Failed) != error.is_some() {
        return Err(lane_error(SessionError::Invariant(
            "Only a failed operation result may carry an error".to_owned(),
        )));
    }
    Ok(OperationResultRecord {
        operation_id: meta.operation_id.clone(),
        kind: intent_kind_of(meta),
        status,
        error,
        from_tip_id: meta.source_tip_id.clone(),
        tip_id,
        started_at: meta.started_at,
        ended_at: now_ms(),
    })
}

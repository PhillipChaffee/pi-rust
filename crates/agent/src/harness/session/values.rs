//! Durable value and list addresses, ported from upstream
//! `src/harness/session/values.ts`.
//!
//! Upstream's `Value<T>`/`ValueList<T>` carry a phantom type through a
//! declared symbol; the port keeps the phantom for call-site typing while
//! the durable layer persists payloads as JSON, so the contract traits
//! address values through the erased [`ValueAddress`]/[`ListAddress`] and
//! the constructors here serialize typed payloads. Upstream's
//! `declare const storedValueType` nominal check restates as the phantom
//! type parameter: a `Value<LaneConfiguration>` cannot be passed where a
//! `Value<OperationMeta>` is expected.

use pi_ai::types::{AssistantMessageFrame, Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::harness::compaction::types::FileOperations;
use crate::harness::session::types::{
    DurableStructuralPreparation, LaneConfiguration, LaneState, OperationMeta,
    OperationResultRecord, OperationState, PendingEntry, SessionError,
};
use crate::harness::types::AgentHarnessStreamOptions;
use crate::types::{AgentMessage, AgentToolResult};

/// The base of both address kinds, upstream's `StoredAddressBase`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredAddressBase {
    /// The value namespace.
    pub namespace: String,
    /// The key inside the namespace.
    pub key: String,
}

/// A typed single-value address, upstream's `Value<T>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value<T> {
    /// The erased address the durable layer reads.
    pub address: ValueAddress,
    _value: std::marker::PhantomData<fn(T) -> T>,
}

/// A typed list address, upstream's `ValueList<T>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueList<T> {
    /// The erased address the durable layer reads.
    pub address: ListAddress,
    _value: std::marker::PhantomData<fn(T) -> T>,
}

/// The stored view of one value, upstream's `StoredValue<T>` with the
/// erased payload the durable layer persists.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredValue {
    /// The addressed namespace.
    pub namespace: String,
    /// The addressed key.
    pub key: String,
    /// The stored value.
    pub value: JsonValue,
    /// The value's sequence.
    pub seq: u64,
}

/// One element of a stored list, upstream's `ListElement<T>`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListElement {
    /// The element's sequence.
    pub seq: u64,
    /// The element value.
    pub value: JsonValue,
}

/// A list read cursor, upstream's `ListCursor`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListCursor {
    /// The sequence the cursor sits at.
    pub seq: u64,
}

/// Options for a list read, upstream's `ListReadOptions`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListReadOptions {
    /// Read from this sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<ListCursor>,
    /// The read order. Defaults to ascending.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<crate::harness::session::types::EntryScanOrder>,
    /// The read limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,
}

/// The resolved list-read options, upstream's `ResolvedListReadOptions`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedListReadOptions {
    /// The resolved cursor, when any.
    pub cursor: Option<ListCursor>,
    /// The resolved order.
    pub order: crate::harness::session::types::EntryScanOrder,
    /// The effective limit.
    pub limit: u64,
}

/// A value set write, upstream's `ValueSetWrite`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValueSetWrite {
    /// The write family.
    pub kind: String,
    /// The value operation.
    pub op: String,
    /// The addressed namespace.
    pub namespace: String,
    /// The addressed key.
    pub key: String,
    /// The serialized value.
    pub value: JsonValue,
}

/// A value delete write, upstream's `ValueDeleteWrite`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValueDeleteWrite {
    /// The write family.
    pub kind: String,
    /// The value operation.
    pub op: String,
    /// The addressed namespace.
    pub namespace: String,
    /// The addressed key.
    pub key: String,
}

/// A list append write, upstream's `ListAppendWrite`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListAppendWrite {
    /// The write family.
    pub kind: String,
    /// The list operation.
    pub op: String,
    /// The addressed namespace.
    pub namespace: String,
    /// The addressed key.
    pub key: String,
    /// The serialized element.
    pub value: JsonValue,
}

/// A list delete write, upstream's `ListDeleteWrite`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListDeleteWrite {
    /// The write family.
    pub kind: String,
    /// The list operation.
    pub op: String,
    /// The addressed namespace.
    pub namespace: String,
    /// The addressed key.
    pub key: String,
}

/// One write in a transaction, upstream's `Write` — the four families the
/// durable layer commits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Write {
    /// An entry insert.
    Entry(EntryWrite),
    /// A usage insert.
    Usage(UsageWrite),
    /// A value set.
    ValueSet(ValueSetWrite),
    /// A value delete.
    ValueDelete(ValueDeleteWrite),
    /// A list append.
    ListAppend(ListAppendWrite),
    /// A list delete.
    ListDelete(ListDeleteWrite),
}

/// An entry insert, upstream's `EntryWrite`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryWrite {
    /// The write family.
    pub kind: String,
    /// The entry to insert.
    pub entry: crate::harness::session::types::NewEntry,
}

/// A usage insert, upstream's `UsageWrite`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageWrite {
    /// The write family.
    pub kind: String,
    /// The usage row to insert.
    pub row: crate::harness::session::types::UsageWriteRow,
}

/// Validates one address component pair, upstream's `validateAddress`.
///
/// # Errors
/// A `TypeError`-shaped message when the namespace is empty or either
/// component contains a NUL byte.
pub fn validate_address(namespace: &str, key: &str) -> Result<(), SessionError> {
    if namespace.is_empty() {
        return Err(SessionError("Value namespace must not be empty".to_owned()));
    }
    if namespace.contains('\u{0}') {
        return Err(SessionError("Value namespace must not contain \\u0000".to_owned()));
    }
    if key.contains('\u{0}') {
        return Err(SessionError("Value key must not contain \\u0000".to_owned()));
    }
    Ok(())
}

/// Addresses one single value, upstream's `value<T>(namespace, key)`.
///
/// # Errors
/// Address validation failures; see [`validate_address`].
pub fn value<T>(namespace: &str, key: &str) -> Result<Value<T>, SessionError> {
    validate_address(namespace, key)?;
    Ok(Value {
        address: ValueAddress {
            namespace: namespace.to_owned(),
            key: key.to_owned(),
        },
        _value: std::marker::PhantomData,
    })
}

/// Addresses one list, upstream's `list<T>(namespace, key)`.
///
/// # Errors
/// Address validation failures; see [`validate_address`].
pub fn list<T>(namespace: &str, key: &str) -> Result<ValueList<T>, SessionError> {
    validate_address(namespace, key)?;
    Ok(ValueList {
        address: ListAddress {
            namespace: namespace.to_owned(),
            key: key.to_owned(),
        },
        _value: std::marker::PhantomData,
    })
}

/// Builds a value set write from a typed address, upstream's
/// `setValue<T>(address, next)`.
///
/// # Errors
/// Payload serialization failures.
pub fn set_value<T: serde::Serialize>(
    address: &Value<T>,
    next: T,
) -> Result<ValueSetWrite, SessionError> {
    Ok(ValueSetWrite {
        kind: "value".to_owned(),
        op: "set".to_owned(),
        namespace: address.address.namespace.clone(),
        key: address.address.key.clone(),
        value: serde_json::to_value(next).map_err(|error| {
            SessionError(format!("Value payload serialization failed: {error}"))
        })?,
    })
}

/// Builds a value delete write, upstream's `deleteValue<T>`.
#[must_use]
pub fn delete_value<T>(address: &Value<T>) -> ValueDeleteWrite {
    ValueDeleteWrite {
        kind: "value".to_owned(),
        op: "delete".to_owned(),
        namespace: address.address.namespace.clone(),
        key: address.address.key.clone(),
    }
}

/// Builds a list append write from a typed address, upstream's
/// `appendList<T>(address, element)`.
///
/// # Errors
/// Payload serialization failures.
pub fn append_list<T: serde::Serialize>(
    address: &ValueList<T>,
    element: T,
) -> Result<ListAppendWrite, SessionError> {
    Ok(ListAppendWrite {
        kind: "list".to_owned(),
        op: "append".to_owned(),
        namespace: address.address.namespace.clone(),
        key: address.address.key.clone(),
        value: serde_json::to_value(element).map_err(|error| {
            SessionError(format!("List element serialization failed: {error}"))
        })?,
    })
}

/// Builds a list delete write, upstream's `deleteList<T>`.
#[must_use]
pub fn delete_list<T>(address: &ValueList<T>) -> ListDeleteWrite {
    ListDeleteWrite {
        kind: "list".to_owned(),
        op: "delete".to_owned(),
        namespace: address.address.namespace.clone(),
        key: address.address.key.clone(),
    }
}

/// Resolves list-read defaults, upstream's `resolveListReadOptions`.
///
/// # Errors
/// A `TypeError`-shaped message when the limit is not a positive safe
/// integer.
pub fn resolve_list_read_options(
    options: Option<ListReadOptions>,
) -> Result<ResolvedListReadOptions, SessionError> {
    let options = options.unwrap_or_default();
    let requested_limit = options.limit.unwrap_or(1_000);
    if requested_limit == 0 || requested_limit > 9_007_199_254_740_991 {
        return Err(SessionError(
            "List read limit must be a positive safe integer".to_owned(),
        ));
    }
    Ok(ResolvedListReadOptions {
        cursor: options.cursor,
        order: options
            .order
            .unwrap_or(crate::harness::session::types::EntryScanOrder::Asc),
        limit: requested_limit.min(10_000),
    })
}

/// The branch tip address, upstream's `branchTip(branch)`.
pub fn branch_tip(branch: &str) -> Value<Option<String>> {
    value::<Option<String>>("pi.branch.tip", branch).unwrap_or_else(|_| unreachable_value())
}

/// The branch-tip inventory prefix, upstream's `branchTipInventoryPrefix()`.
pub fn branch_tip_inventory_prefix() -> Value<Option<String>> {
    value::<Option<String>>("pi.branch.tip", "").unwrap_or_else(|_| unreachable_value())
}

/// The lane configuration address, upstream's `laneConfig(lane)`.
pub fn lane_config(lane: &str) -> Value<LaneConfiguration> {
    value::<LaneConfiguration>("pi.lane.config", lane).unwrap_or_else(|_| unreachable_value())
}

/// The lane state address, upstream's `laneState(lane)`.
pub fn lane_state(lane: &str) -> Value<LaneState> {
    value::<LaneState>("pi.lane.state", lane).unwrap_or_else(|_| unreachable_value())
}

/// The operation result address, upstream's `operationResult(operationId)`.
pub fn operation_result(operation_id: &str) -> Value<OperationResultRecord> {
    value::<OperationResultRecord>("pi.result", operation_id).unwrap_or_else(|_| unreachable_value())
}

/// The operation meta address, upstream's `operationMeta(operationId)`.
pub fn operation_meta(operation_id: &str) -> Value<OperationMeta> {
    value::<OperationMeta>("pi.op.meta", operation_id).unwrap_or_else(|_| unreachable_value())
}

/// The operation state address, upstream's `operationState(operationId)`.
pub fn operation_state(operation_id: &str) -> Value<OperationState> {
    value::<OperationState>("pi.op.state", operation_id).unwrap_or_else(|_| unreachable_value())
}

/// The tool-arguments address, upstream's
/// `operationToolArgs(operationId, stepId, sourceIndex)`.
pub fn operation_tool_args(operation_id: &str, step_id: &str, source_index: u64) -> Value<ToolArgs> {
    value::<ToolArgs>(
        "pi.op.tool_args",
        &format!("{operation_id}:{step_id}:{source_index}"),
    )
    .unwrap_or_else(|_| unreachable_value())
}

/// The tool-argument map payload, upstream's `Record<string, JsonValue>`.
pub type ToolArgs = serde_json::Map<String, JsonValue>;

/// The tool memo address, upstream's
/// `operationToolMemo(operationId, invocationId, name)`.
pub fn operation_tool_memo(operation_id: &str, invocation_id: &str, name: &str) -> Value<JsonValue> {
    value::<JsonValue>(
        "pi.op.tool_memo",
        &format!("{operation_id}:{invocation_id}:{name}"),
    )
    .unwrap_or_else(|_| unreachable_value())
}

/// The operation preparation address, upstream's
/// `operationPreparation(operationId, taskId)`.
pub fn operation_preparation(operation_id: &str, task_id: &str) -> Value<DurableStructuralPreparation> {
    value::<DurableStructuralPreparation>("pi.op.preparation", &format!("{operation_id}:{task_id}"))
        .unwrap_or_else(|_| unreachable_value())
}

/// The tool-arguments prefix address, upstream's
/// `operationToolArgsPrefix(operationId, stepId?)`.
pub fn operation_tool_args_prefix(operation_id: &str, step_id: Option<&str>) -> Value<ToolArgs> {
    let key = match step_id {
        Some(step_id) => format!("{operation_id}:{step_id}:"),
        None => format!("{operation_id}:"),
    };
    value::<ToolArgs>("pi.op.tool_args", &key).unwrap_or_else(|_| unreachable_value())
}

/// The tool-memo prefix address, upstream's
/// `operationToolMemoPrefix(operationId, invocationId?)`.
pub fn operation_tool_memo_prefix(operation_id: &str, invocation_id: Option<&str>) -> Value<JsonValue> {
    let key = match invocation_id {
        Some(invocation_id) => format!("{operation_id}:{invocation_id}:"),
        None => format!("{operation_id}:"),
    };
    value::<JsonValue>("pi.op.tool_memo", &key).unwrap_or_else(|_| unreachable_value())
}

/// The preparation prefix address, upstream's
/// `operationPreparationPrefix(operationId)`.
pub fn operation_preparation_prefix(operation_id: &str) -> Value<DurableStructuralPreparation> {
    value::<DurableStructuralPreparation>("pi.op.preparation", &format!("{operation_id}:"))
        .unwrap_or_else(|_| unreachable_value())
}

/// The pending-entry address, upstream's `pendingEntry(entryId)`.
pub fn pending_entry(entry_id: &str) -> Value<PendingEntry> {
    value::<PendingEntry>("pi.pending.entry", entry_id).unwrap_or_else(|_| unreachable_value())
}

/// The pending tool output address, upstream's
/// `pendingToolOutput(operationId, invocationId)`.
pub fn pending_tool_output(operation_id: &str, invocation_id: &str) -> Value<ToolOutputPayload> {
    value::<ToolOutputPayload>(
        "pi.pending.tool_output",
        &format!("{operation_id}:{invocation_id}"),
    )
    .unwrap_or_else(|_| unreachable_value())
}

/// The pending tool output payload, upstream's
/// `AgentToolResult<unknown>` with the erased details.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolOutputPayload {
    /// Text or image content returned to the model.
    pub content: Vec<crate::types::AgentToolContent>,
    /// Structured details.
    pub details: JsonValue,
    /// Usage from the tool execution itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<pi_ai::types::Usage>,
    /// Tools introduced by this result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub added_tool_names: Option<Vec<String>>,
    /// The terminate hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminate: Option<bool>,
}

/// The pending assistant frames list address, upstream's
/// `pendingAssistantFrames(operationId, responseEntryId)`.
pub fn pending_assistant_frames(
    operation_id: &str,
    response_entry_id: &str,
) -> ValueList<AssistantMessageFrame> {
    list::<AssistantMessageFrame>(
        "pi.pending.assistant_frame",
        &format!("{operation_id}:{response_entry_id}"),
    )
    .unwrap_or_else(|_| unreachable_value())
}

/// The pending tool output prefix, upstream's
/// `pendingToolOutputPrefix(operationId)`.
pub fn pending_tool_output_prefix(operation_id: &str) -> Value<ToolOutputPayload> {
    value::<ToolOutputPayload>("pi.pending.tool_output", &format!("{operation_id}:"))
        .unwrap_or_else(|_| unreachable_value())
}

/// The session name address, upstream's `sessionName`.
pub fn session_name() -> Value<String> {
    value::<String>("pi.session.name", "").unwrap_or_else(|_| unreachable_value())
}

/// The entry label address, upstream's `entryLabel(entryId)`.
pub fn entry_label(entry_id: &str) -> Value<String> {
    value::<String>("pi.entry.label", entry_id).unwrap_or_else(|_| unreachable_value())
}

/// A generic value address, upstream's `value<unknown>("test.value", ...)`
/// call sites.
pub fn generic_value(namespace: &str, key: &str) -> Value<JsonValue> {
    value::<JsonValue>(namespace, key).unwrap_or_else(|_| unreachable_value())
}

/// The erased stream options payload the durable preparation carries,
/// upstream's `AgentHarnessStreamOptions` at rest.
pub type StreamOptionsPayload = AgentHarnessStreamOptions;

/// The value-write union the session contract references, upstream's
/// `ValueWrite = ValueSetWrite | ValueDeleteWrite`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ValueWrite {
    /// A set.
    Set(ValueSetWrite),
    /// A delete.
    Delete(ValueDeleteWrite),
}

/// The list-write union, upstream's `ListWrite`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ListWrite {
    /// An append.
    Append(ListAppendWrite),
    /// A delete.
    Delete(ListDeleteWrite),
}

/// The erased single-value address the contract traits read, upstream's
/// `StoredAddressBase` with `kind: "value"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueAddress {
    /// The value namespace.
    pub namespace: String,
    /// The addressed key.
    pub key: String,
}

/// The erased list address the contract traits read, upstream's
/// `StoredAddressBase` with `kind: "list"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListAddress {
    /// The list namespace.
    pub namespace: String,
    /// The list key.
    pub key: String,
}

/// The list cursor, upstream's `ListCursor`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListCursor {
    /// The sequence the cursor sits at.
    pub seq: u64,
}

/// The file operations a durable preparation records, upstream's
/// `DurableFileOperations` — sorted vectors on the wire.
pub type DurableFileOperations = FileOperations;

/// The internal marker the `unreachable_value` helpers feed; the fixed
/// harness namespaces never fail validation, so the fallback never
/// constructs.
fn unreachable_value<T>() -> T {
    panic!("fixed harness value addresses are always valid")
}

#[cfg(test)]
mod tests;
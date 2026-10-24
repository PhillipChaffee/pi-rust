//! The [`ExtensionEvent`] wire mappings, ADR 0010's `call_event` payloads:
//! upstream's event objects as `serde_json::Value`, camelCase per upstream
//! `types.ts`, the `"type"` discriminator pinned to the channel.
//! `AbortSignal` crosses as the wire signal id through the allocator and
//! resolver closures the dispatch slice (#170) binds. Handler results map
//! in [`super::results`].

use serde_json::{Map, Value, json};

use pi_agent_core::harness::context::AbortSignal;

use crate::compaction::{CompactionPreparation, FileOperations};
use crate::extensions::events::ExtensionEvent;
use crate::extensions::reasons::{
    CompactionReason, ForkPosition, InputSource, ModelSelectSource, SessionShutdownReason,
    SessionStartReason, StreamingBehavior, TreePreparation,
};
use crate::session_manager::SessionEntry;
use crate::settings_manager::CompactionSettings;
use crate::system_prompt::BuildSystemPromptOptions;

use super::{
    WireError, fields, optional_bool, optional_string, optional_string_array, payload_error,
    required_bool, required_string, required_string_array, required_u64, required_value,
    serde_from_wire, serde_to_wire,
};

/// The host-side half of the signal bridge: assign the wire signal id for
/// the call scope carrying this event's abort signal.
type SignalIdOf<'a> = &'a mut (dyn FnMut(&AbortSignal) -> u64 + 'a);

/// The child-side half of the signal bridge: resolve a wire signal id to
/// the extension's local abort signal, or `None` when the id is unknown.
type SignalOfId<'a> = &'a mut (dyn FnMut(u64) -> Option<AbortSignal> + 'a);

/// Maps one event into its ADR 0010 `call_event` payload, upstream's
/// `ExtensionEvent` object including its `"type"` discriminator.
///
/// # Errors
///
/// Returns [`super::WireError`] when a serde-backed member stops short of
/// the wire's JSON, or when the closures fail.
#[expect(
    clippy::too_many_lines,
    reason = "the event table is one match, the wire's complete event index"
)]
pub fn event_to_wire(
    event: &ExtensionEvent,
    signal_id: SignalIdOf<'_>,
) -> Result<Value, WireError> {
    match event {
        ExtensionEvent::ProjectTrust { cwd } => Ok(json!({
            "type": "project_trust",
            "cwd": cwd,
        })),
        ExtensionEvent::ResourcesDiscover { cwd, reason } => Ok(json!({
            "type": "resources_discover",
            "cwd": cwd,
            "reason": session_start_reason_to_wire(*reason),
        })),
        ExtensionEvent::SessionStart {
            reason,
            previous_session_file,
        } => {
            let mut payload = json!({
                "type": "session_start",
                "reason": session_start_reason_to_wire(*reason),
            });
            if let Some(previous_session_file) = previous_session_file {
                payload["previousSessionFile"] = json!(previous_session_file);
            }
            Ok(payload)
        }
        ExtensionEvent::SessionInfoChanged { name } => {
            let mut payload = json!({ "type": "session_info_changed" });
            if let Some(name) = name {
                payload["name"] = json!(name);
            }
            Ok(payload)
        }
        ExtensionEvent::SessionBeforeSwitch {
            reason,
            target_session_file,
        } => {
            let mut payload = json!({
                "type": "session_before_switch",
                "reason": session_shutdown_reason_to_wire(*reason),
            });
            if let Some(target_session_file) = target_session_file {
                payload["targetSessionFile"] = json!(target_session_file);
            }
            Ok(payload)
        }
        ExtensionEvent::SessionBeforeFork { entry_id, position } => Ok(json!({
            "type": "session_before_fork",
            "entryId": entry_id,
            "position": fork_position_to_wire(*position),
        })),
        ExtensionEvent::SessionBeforeCompact {
            preparation,
            branch_entries,
            custom_instructions,
            reason,
            will_retry,
            signal,
        } => {
            let mut payload = json!({
                "type": "session_before_compact",
                "preparation": compaction_preparation_to_wire(preparation)?,
                "branchEntries": serde_to_wire(branch_entries, "branchEntries")?,
                "reason": compaction_reason_to_wire(*reason),
                "willRetry": will_retry,
            });
            if let Some(custom_instructions) = custom_instructions {
                payload["customInstructions"] = json!(custom_instructions);
            }
            put_signal(&mut payload, signal.as_ref(), signal_id);
            Ok(payload)
        }
        ExtensionEvent::SessionCompact {
            compaction_entry,
            from_extension,
            reason,
            will_retry,
        } => Ok(json!({
            "type": "session_compact",
            "compactionEntry": serde_to_wire(
                &SessionEntry::Compaction(compaction_entry.clone()),
                "compactionEntry",
            )?,
            "fromExtension": from_extension,
            "reason": compaction_reason_to_wire(*reason),
            "willRetry": will_retry,
        })),
        ExtensionEvent::SessionCompactFailed {
            reason,
            error_message,
            aborted,
            will_retry,
            from_extension,
        } => {
            let mut payload = json!({
                "type": "session_compact_failed",
                "reason": compaction_reason_to_wire(*reason),
                "aborted": aborted,
                "willRetry": will_retry,
                "fromExtension": from_extension,
            });
            if let Some(error_message) = error_message {
                payload["errorMessage"] = json!(error_message);
            }
            Ok(payload)
        }
        ExtensionEvent::SessionShutdown {
            reason,
            target_session_file,
        } => {
            let mut payload = json!({
                "type": "session_shutdown",
                "reason": session_shutdown_reason_to_wire(*reason),
            });
            if let Some(target_session_file) = target_session_file {
                payload["targetSessionFile"] = json!(target_session_file);
            }
            Ok(payload)
        }
        ExtensionEvent::SessionBeforeTree {
            preparation,
            signal,
        } => {
            let mut payload = json!({
                "type": "session_before_tree",
                "preparation": tree_preparation_to_wire(preparation)?,
            });
            put_signal(&mut payload, signal.as_ref(), signal_id);
            Ok(payload)
        }
        ExtensionEvent::SessionTree {
            new_leaf_id,
            old_leaf_id,
            summary_entry,
            from_extension,
        } => {
            let mut payload = json!({
                "type": "session_tree",
                "newLeafId": new_leaf_id.clone().map_or(Value::Null, Value::from),
                "oldLeafId": old_leaf_id.clone().map_or(Value::Null, Value::from),
            });
            if let Some(summary_entry) = summary_entry {
                payload["summaryEntry"] = serde_to_wire(summary_entry, "summaryEntry")?;
            }
            if let Some(from_extension) = from_extension {
                payload["fromExtension"] = json!(from_extension);
            }
            Ok(payload)
        }
        ExtensionEvent::Context { messages } => Ok(json!({
            "type": "context",
            "messages": serde_to_wire(messages, "messages")?,
        })),
        ExtensionEvent::BeforeProviderRequest { payload: request } => Ok(json!({
            "type": "before_provider_request",
            "payload": request,
        })),
        ExtensionEvent::BeforeProviderHeaders { headers } => Ok(json!({
            "type": "before_provider_headers",
            "headers": serde_to_wire(headers, "headers")?,
        })),
        ExtensionEvent::AfterProviderResponse { status, headers } => {
            // Upstream's status is a plain number; the port's Option reads
            // None as the wire's null.
            let status = status.map_or(Value::Null, |status| json!(status));
            Ok(json!({
                "type": "after_provider_response",
                "status": status,
                "headers": serde_to_wire(headers, "headers")?,
            }))
        }
        ExtensionEvent::BeforeAgentStart {
            prompt,
            images,
            system_prompt,
            system_prompt_options,
            signal,
        } => {
            let mut payload = json!({
                "type": "before_agent_start",
                "prompt": prompt,
                "systemPrompt": system_prompt,
                "systemPromptOptions": system_prompt_options_to_wire(system_prompt_options),
            });
            if let Some(images) = images {
                payload["images"] = serde_to_wire(images, "images")?;
            }
            // Upstream delivers the run's signal through ctx; the port's
            // event carries it and crosses as its wire signal id.
            put_signal(&mut payload, signal.as_ref(), signal_id);
            Ok(payload)
        }
        ExtensionEvent::AgentStart => Ok(json!({ "type": "agent_start" })),
        ExtensionEvent::AgentEnd { messages } => Ok(json!({
            "type": "agent_end",
            "messages": serde_to_wire(messages, "messages")?,
        })),
        ExtensionEvent::AgentSettled => Ok(json!({ "type": "agent_settled" })),
        ExtensionEvent::TurnStart {
            turn_index,
            timestamp,
        } => Ok(json!({
            "type": "turn_start",
            "turnIndex": turn_index,
            "timestamp": timestamp,
        })),
        ExtensionEvent::TurnEnd {
            turn_index,
            message,
            tool_results,
        } => Ok(json!({
            "type": "turn_end",
            "turnIndex": turn_index,
            "message": serde_to_wire(message, "message")?,
            "toolResults": serde_to_wire(tool_results, "toolResults")?,
        })),
        ExtensionEvent::MessageStart { message } => Ok(json!({
            "type": "message_start",
            "message": serde_to_wire(message, "message")?,
        })),
        ExtensionEvent::MessageUpdate {
            message,
            assistant_message_event,
        } => Ok(json!({
            "type": "message_update",
            "message": serde_to_wire(message, "message")?,
            "assistantMessageEvent": serde_to_wire(
                assistant_message_event,
                "assistantMessageEvent",
            )?,
        })),
        ExtensionEvent::MessageEnd { message } => Ok(json!({
            "type": "message_end",
            "message": serde_to_wire(message, "message")?,
        })),
        ExtensionEvent::ToolExecutionStart {
            tool_call_id,
            tool_name,
            args,
        } => Ok(json!({
            "type": "tool_execution_start",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "args": args,
        })),
        ExtensionEvent::ToolExecutionUpdate {
            tool_call_id,
            tool_name,
            args,
            partial_result,
        } => Ok(json!({
            "type": "tool_execution_update",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "args": args,
            "partialResult": serde_to_wire(partial_result, "partialResult")?,
        })),
        ExtensionEvent::ToolExecutionEnd {
            tool_call_id,
            tool_name,
            result,
            is_error,
        } => Ok(json!({
            "type": "tool_execution_end",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "result": serde_to_wire(result, "result")?,
            "isError": is_error,
        })),
        ExtensionEvent::ModelSelect {
            model,
            previous_model,
            source,
        } => {
            let mut payload = json!({
                "type": "model_select",
                "model": serde_to_wire(model, "model")?,
                "source": model_select_source_to_wire(*source),
            });
            if let Some(previous_model) = previous_model {
                payload["previousModel"] = serde_to_wire(previous_model, "previousModel")?;
            }
            Ok(payload)
        }
        ExtensionEvent::ThinkingLevelSelect {
            level,
            previous_level,
        } => Ok(json!({
            "type": "thinking_level_select",
            "level": serde_to_wire(level, "level")?,
            "previousLevel": serde_to_wire(previous_level, "previousLevel")?,
        })),
        ExtensionEvent::UserBash { command } => Ok(json!({
            "type": "user_bash",
            "command": command,
        })),
        ExtensionEvent::Input {
            text,
            images,
            source,
            streaming_behavior,
        } => {
            let mut payload = json!({
                "type": "input",
                "text": text,
                "source": input_source_to_wire(*source),
            });
            if let Some(images) = images {
                payload["images"] = serde_to_wire(images, "images")?;
            }
            if let Some(streaming_behavior) = streaming_behavior {
                payload["streamingBehavior"] =
                    json!(streaming_behavior_to_wire(*streaming_behavior));
            }
            Ok(payload)
        }
        ExtensionEvent::ToolCall {
            tool_name,
            tool_call_id,
            input,
        } => Ok(json!({
            "type": "tool_call",
            "toolName": tool_name,
            "toolCallId": tool_call_id,
            "input": input,
        })),
        ExtensionEvent::ToolResult {
            tool_name,
            tool_call_id,
            input,
            content,
            details,
            is_error,
            usage,
        } => {
            let mut payload = json!({
                "type": "tool_result",
                "toolName": tool_name,
                "toolCallId": tool_call_id,
                "input": input,
                "content": serde_to_wire(content, "content")?,
                "details": details,
                "isError": is_error,
            });
            if let Some(usage) = usage {
                payload["usage"] = serde_to_wire(usage, "usage")?;
            }
            Ok(payload)
        }
    }
}

/// Rebuilds the event a `call_event` payload carried.
///
/// # Errors
///
/// Returns [`super::WireError`] when the channel is unknown, the payload's
/// `"type"` disagrees with it, a field is missing or mistyped, or the
/// signal resolver cannot name the payload's signal id.
#[expect(
    clippy::too_many_lines,
    reason = "the event table is one match, the wire's complete event index"
)]
pub fn event_from_wire(
    channel: &str,
    payload: &Value,
    signal: SignalOfId<'_>,
) -> Result<ExtensionEvent, WireError> {
    let fields = fields(payload, channel)?;
    let type_ = fields
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| payload_error("the event payload must carry a string `type`"))?;
    if type_ != channel {
        return Err(payload_error(&format!(
            "`{type_}` payload does not match channel `{channel}`"
        )));
    }
    match channel {
        "project_trust" => Ok(ExtensionEvent::ProjectTrust {
            cwd: required_string(fields, "cwd", channel)?,
        }),
        "resources_discover" => Ok(ExtensionEvent::ResourcesDiscover {
            cwd: required_string(fields, "cwd", channel)?,
            reason: session_start_reason_from_wire(
                &required_string(fields, "reason", channel)?,
                &["startup", "reload"],
            )?,
        }),
        "session_start" => Ok(ExtensionEvent::SessionStart {
            reason: session_start_reason_from_wire(
                &required_string(fields, "reason", channel)?,
                &["startup", "new", "resume", "fork", "reload"],
            )?,
            previous_session_file: optional_string(fields, "previousSessionFile", channel)?,
        }),
        "session_info_changed" => Ok(ExtensionEvent::SessionInfoChanged {
            name: optional_string(fields, "name", channel)?,
        }),
        "session_before_switch" => Ok(ExtensionEvent::SessionBeforeSwitch {
            reason: session_shutdown_reason_from_wire(
                &required_string(fields, "reason", channel)?,
                // Upstream types the channel's reason as new | resume.
                &["new", "resume"],
            )?,
            target_session_file: optional_string(fields, "targetSessionFile", channel)?,
        }),
        "session_before_fork" => Ok(ExtensionEvent::SessionBeforeFork {
            entry_id: required_string(fields, "entryId", channel)?,
            position: fork_position_from_wire(&required_string(fields, "position", channel)?)?,
        }),
        "session_before_compact" => Ok(ExtensionEvent::SessionBeforeCompact {
            preparation: compaction_preparation_from_wire(required_value(
                fields,
                "preparation",
                channel,
            )?)?,
            branch_entries: serde_from_wire(
                required_value(fields, "branchEntries", channel)?.clone(),
                "branchEntries",
            )?,
            custom_instructions: optional_string(fields, "customInstructions", channel)?,
            reason: compaction_reason_from_wire(&required_string(fields, "reason", channel)?)?,
            will_retry: required_bool(fields, "willRetry", channel)?,
            signal: take_signal(fields, signal)?,
        }),
        "session_compact" => {
            let entry: SessionEntry = serde_from_wire(
                required_value(fields, "compactionEntry", channel)?.clone(),
                "compactionEntry",
            )?;
            let SessionEntry::Compaction(compaction_entry) = entry else {
                return Err(payload_error(
                    "session_compact's compactionEntry is not a compaction entry",
                ));
            };
            Ok(ExtensionEvent::SessionCompact {
                compaction_entry,
                from_extension: required_bool(fields, "fromExtension", channel)?,
                reason: compaction_reason_from_wire(&required_string(fields, "reason", channel)?)?,
                will_retry: required_bool(fields, "willRetry", channel)?,
            })
        }
        "session_compact_failed" => Ok(ExtensionEvent::SessionCompactFailed {
            reason: compaction_reason_from_wire(&required_string(fields, "reason", channel)?)?,
            error_message: optional_string(fields, "errorMessage", channel)?,
            aborted: required_bool(fields, "aborted", channel)?,
            will_retry: required_bool(fields, "willRetry", channel)?,
            from_extension: required_bool(fields, "fromExtension", channel)?,
        }),
        "session_shutdown" => Ok(ExtensionEvent::SessionShutdown {
            reason: session_shutdown_reason_from_wire(
                &required_string(fields, "reason", channel)?,
                &["quit", "new", "resume", "fork", "reload"],
            )?,
            target_session_file: optional_string(fields, "targetSessionFile", channel)?,
        }),
        "session_before_tree" => Ok(ExtensionEvent::SessionBeforeTree {
            preparation: tree_preparation_from_wire(required_value(
                fields,
                "preparation",
                channel,
            )?)?,
            signal: take_signal(fields, signal)?,
        }),
        "session_tree" => Ok(ExtensionEvent::SessionTree {
            new_leaf_id: optional_id(fields, "newLeafId", channel)?,
            old_leaf_id: optional_id(fields, "oldLeafId", channel)?,
            summary_entry: match fields.get("summaryEntry") {
                None | Some(Value::Null) => None,
                Some(entry) => Some(serde_from_wire(entry.clone(), "summaryEntry")?),
            },
            from_extension: optional_bool(fields, "fromExtension", channel)?,
        }),
        "context" => Ok(ExtensionEvent::Context {
            messages: serde_from_wire(
                required_value(fields, "messages", channel)?.clone(),
                "messages",
            )?,
        }),
        "before_provider_request" => Ok(ExtensionEvent::BeforeProviderRequest {
            payload: required_value(fields, "payload", channel)?.clone(),
        }),
        "before_provider_headers" => Ok(ExtensionEvent::BeforeProviderHeaders {
            headers: serde_from_wire(
                required_value(fields, "headers", channel)?.clone(),
                "headers",
            )?,
        }),
        "after_provider_response" => Ok(ExtensionEvent::AfterProviderResponse {
            status: match fields.get("status") {
                Some(Value::Number(status)) => {
                    let status = status.as_u64().ok_or_else(|| {
                        payload_error("after_provider_response's status must be an integer")
                    })?;
                    Some(u16::try_from(status).map_err(|_| {
                        payload_error("after_provider_response's status is not a u16")
                    })?)
                }
                Some(Value::Null) | None => None,
                Some(_) => {
                    return Err(payload_error(
                        "after_provider_response's status must be a number",
                    ));
                }
            },
            headers: serde_from_wire(
                required_value(fields, "headers", channel)?.clone(),
                "headers",
            )?,
        }),
        "before_agent_start" => Ok(ExtensionEvent::BeforeAgentStart {
            prompt: required_string(fields, "prompt", channel)?,
            images: match fields.get("images") {
                None | Some(Value::Null) => None,
                Some(images) => Some(serde_from_wire(images.clone(), "images")?),
            },
            system_prompt: required_string(fields, "systemPrompt", channel)?,
            system_prompt_options: system_prompt_options_from_wire(required_value(
                fields,
                "systemPromptOptions",
                channel,
            )?)?,
            signal: take_signal(fields, signal)?,
        }),
        "agent_start" => Ok(ExtensionEvent::AgentStart),
        "agent_end" => Ok(ExtensionEvent::AgentEnd {
            messages: serde_from_wire(
                required_value(fields, "messages", channel)?.clone(),
                "messages",
            )?,
        }),
        "agent_settled" => Ok(ExtensionEvent::AgentSettled),
        "turn_start" => Ok(ExtensionEvent::TurnStart {
            turn_index: required_usize(fields, "turnIndex", channel)?,
            timestamp: required_i64(fields, "timestamp", channel)?,
        }),
        "turn_end" => Ok(ExtensionEvent::TurnEnd {
            turn_index: required_usize(fields, "turnIndex", channel)?,
            message: serde_from_wire(
                required_value(fields, "message", channel)?.clone(),
                "message",
            )?,
            tool_results: serde_from_wire(
                required_value(fields, "toolResults", channel)?.clone(),
                "toolResults",
            )?,
        }),
        "message_start" => Ok(ExtensionEvent::MessageStart {
            message: serde_from_wire(
                required_value(fields, "message", channel)?.clone(),
                "message",
            )?,
        }),
        "message_update" => Ok(ExtensionEvent::MessageUpdate {
            message: serde_from_wire(
                required_value(fields, "message", channel)?.clone(),
                "message",
            )?,
            assistant_message_event: serde_from_wire(
                required_value(fields, "assistantMessageEvent", channel)?.clone(),
                "assistantMessageEvent",
            )?,
        }),
        "message_end" => Ok(ExtensionEvent::MessageEnd {
            message: serde_from_wire(
                required_value(fields, "message", channel)?.clone(),
                "message",
            )?,
        }),
        "tool_execution_start" => Ok(ExtensionEvent::ToolExecutionStart {
            tool_call_id: required_string(fields, "toolCallId", channel)?,
            tool_name: required_string(fields, "toolName", channel)?,
            args: fields.get("args").cloned().unwrap_or(Value::Null),
        }),
        "tool_execution_update" => Ok(ExtensionEvent::ToolExecutionUpdate {
            tool_call_id: required_string(fields, "toolCallId", channel)?,
            tool_name: required_string(fields, "toolName", channel)?,
            args: fields.get("args").cloned().unwrap_or(Value::Null),
            partial_result: serde_from_wire(
                required_value(fields, "partialResult", channel)?.clone(),
                "partialResult",
            )?,
        }),
        "tool_execution_end" => Ok(ExtensionEvent::ToolExecutionEnd {
            tool_call_id: required_string(fields, "toolCallId", channel)?,
            tool_name: required_string(fields, "toolName", channel)?,
            result: serde_from_wire(required_value(fields, "result", channel)?.clone(), "result")?,
            is_error: required_bool(fields, "isError", channel)?,
        }),
        "model_select" => Ok(ExtensionEvent::ModelSelect {
            model: serde_from_wire(required_value(fields, "model", channel)?.clone(), "model")?,
            previous_model: match fields.get("previousModel") {
                None | Some(Value::Null) => None,
                Some(previous_model) => {
                    Some(serde_from_wire(previous_model.clone(), "previousModel")?)
                }
            },
            source: model_select_source_from_wire(&required_string(fields, "source", channel)?)?,
        }),
        "thinking_level_select" => Ok(ExtensionEvent::ThinkingLevelSelect {
            level: serde_from_wire(required_value(fields, "level", channel)?.clone(), "level")?,
            previous_level: serde_from_wire(
                required_value(fields, "previousLevel", channel)?.clone(),
                "previousLevel",
            )?,
        }),
        // Upstream's user_bash also carries excludeFromContext and cwd; the
        // port's domain tracks only the command (#148).
        "user_bash" => Ok(ExtensionEvent::UserBash {
            command: required_string(fields, "command", channel)?,
        }),
        "input" => Ok(ExtensionEvent::Input {
            text: required_string(fields, "text", channel)?,
            images: match fields.get("images") {
                None | Some(Value::Null) => None,
                Some(images) => Some(serde_from_wire(images.clone(), "images")?),
            },
            source: input_source_from_wire(&required_string(fields, "source", channel)?)?,
            streaming_behavior: match fields.get("streamingBehavior") {
                None | Some(Value::Null) => None,
                Some(_) => Some(streaming_behavior_from_wire(&required_string(
                    fields,
                    "streamingBehavior",
                    channel,
                )?)?),
            },
        }),
        "tool_call" => Ok(ExtensionEvent::ToolCall {
            tool_name: required_string(fields, "toolName", channel)?,
            tool_call_id: required_string(fields, "toolCallId", channel)?,
            input: fields.get("input").cloned().unwrap_or(Value::Null),
        }),
        "tool_result" => Ok(ExtensionEvent::ToolResult {
            tool_name: required_string(fields, "toolName", channel)?,
            tool_call_id: required_string(fields, "toolCallId", channel)?,
            input: fields.get("input").cloned().unwrap_or(Value::Null),
            content: serde_from_wire(
                required_value(fields, "content", channel)?.clone(),
                "content",
            )?,
            details: fields.get("details").cloned().unwrap_or(Value::Null),
            is_error: required_bool(fields, "isError", channel)?,
            usage: match fields.get("usage") {
                None | Some(Value::Null) => None,
                Some(usage) => Some(serde_from_wire(usage.clone(), "usage")?),
            },
        }),
        unknown => Err(WireError::new(format!("unknown event channel `{unknown}`"))),
    }
}

/// Every channel the payload mappings cover, in [`ExtensionEvent`] union
/// order.
///
/// The table-driven tests iterate it; a new event extends the
/// compiler-checked `event_to_wire` first, then this table, then the
/// fixture table's round-trip.
pub const EVENT_CHANNELS: [&str; 34] = [
    "project_trust",
    "resources_discover",
    "session_start",
    "session_info_changed",
    "session_before_switch",
    "session_before_fork",
    "session_before_compact",
    "session_compact",
    "session_compact_failed",
    "session_shutdown",
    "session_before_tree",
    "session_tree",
    "context",
    "before_provider_request",
    "before_provider_headers",
    "after_provider_response",
    "before_agent_start",
    "agent_start",
    "agent_end",
    "agent_settled",
    "turn_start",
    "turn_end",
    "message_start",
    "message_update",
    "message_end",
    "tool_execution_start",
    "tool_execution_update",
    "tool_execution_end",
    "model_select",
    "thinking_level_select",
    "user_bash",
    "input",
    "tool_call",
    "tool_result",
];

fn put_signal(payload: &mut Value, signal: Option<&AbortSignal>, signal_id: SignalIdOf<'_>) {
    let Some(signal) = signal else {
        return;
    };
    payload["signal"] = json!(signal_id(signal));
}

fn take_signal(
    fields: &Map<String, Value>,
    resolve: SignalOfId<'_>,
) -> Result<Option<AbortSignal>, WireError> {
    match fields.get("signal") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(id)) => {
            let id = id
                .as_u64()
                .ok_or_else(|| payload_error("a signal id must be a non-negative integer"))?;
            let signal = resolve(id)
                .ok_or_else(|| payload_error(&format!("the config has no signal for id `{id}`")))?;
            Ok(Some(signal))
        }
        Some(_) => Err(payload_error("the `signal` field must carry a signal id")),
    }
}

/// The `before_agent_start` system-prompt options, upstream's
/// `BuildSystemPromptOptions`. Absent optionals stay absent on the wire.
fn system_prompt_options_to_wire(options: &BuildSystemPromptOptions) -> Value {
    let mut wire = json!({
        "cwd": options.cwd,
    });
    if let Some(custom_prompt) = &options.custom_prompt {
        wire["customPrompt"] = json!(custom_prompt);
    }
    if let Some(selected_tools) = &options.selected_tools {
        wire["selectedTools"] = json!(selected_tools);
    }
    if let Some(tool_snippets) = &options.tool_snippets {
        wire["toolSnippets"] = json!(tool_snippets);
    }
    if let Some(prompt_guidelines) = &options.prompt_guidelines {
        wire["promptGuidelines"] = json!(prompt_guidelines);
    }
    if let Some(append_system_prompt) = &options.append_system_prompt {
        wire["appendSystemPrompt"] = json!(append_system_prompt);
    }
    if let Some(context_files) = &options.context_files {
        wire["contextFiles"] = context_files_to_wire(context_files);
    }
    if let Some(skills) = &options.skills {
        wire["skills"] = skills_vec_to_wire(skills);
    }
    wire
}

fn context_files_to_wire(files: &[crate::system_prompt::ContextFile]) -> Value {
    Value::from(
        files
            .iter()
            .map(|file| json!({ "path": file.path, "content": file.content }))
            .collect::<Vec<Value>>(),
    )
}

fn skills_vec_to_wire(skills: &[crate::skills::Skill]) -> Value {
    let mut wire = Vec::new();
    for skill in skills {
        wire.push(json!({
            "name": skill.name,
            "description": skill.description,
            "filePath": skill.file_path,
            "baseDir": skill.base_dir,
            "sourceInfo": super::descriptors::source_info_to_wire(&skill.source_info),
            "disableModelInvocation": skill.disable_model_invocation,
        }));
    }
    Value::from(wire)
}

fn system_prompt_options_from_wire(value: &Value) -> Result<BuildSystemPromptOptions, WireError> {
    let options = fields(value, "system_prompt_options")?;
    Ok(BuildSystemPromptOptions {
        custom_prompt: optional_string(options, "customPrompt", "system_prompt_options")?,
        selected_tools: optional_string_array(options, "selectedTools", "system_prompt_options")?,
        tool_snippets: match options.get("toolSnippets") {
            None | Some(Value::Null) => None,
            Some(Value::Object(snippets)) => {
                let mut resolved = std::collections::HashMap::new();
                for (name, snippet) in snippets {
                    let snippet = snippet.as_str().ok_or_else(|| {
                        payload_error("system_prompt_options' toolSnippets must hold strings")
                    })?;
                    resolved.insert(name.clone(), snippet.to_string());
                }
                Some(resolved)
            }
            Some(_) => {
                return Err(payload_error(
                    "system_prompt_options' toolSnippets must be an object",
                ));
            }
        },
        prompt_guidelines: optional_string_array(
            options,
            "promptGuidelines",
            "system_prompt_options",
        )?,
        append_system_prompt: optional_string(
            options,
            "appendSystemPrompt",
            "system_prompt_options",
        )?,
        cwd: required_string(options, "cwd", "system_prompt_options")?,
        context_files: match options.get("contextFiles") {
            None | Some(Value::Null) => None,
            Some(Value::Array(files)) => {
                let mut resolved = Vec::new();
                for file in files {
                    let file_fields = fields(file, "context_file")?;
                    resolved.push(crate::system_prompt::ContextFile {
                        path: required_string(file_fields, "path", "context_file")?,
                        content: required_string(file_fields, "content", "context_file")?,
                    });
                }
                Some(resolved)
            }
            Some(_) => {
                return Err(payload_error(
                    "system_prompt_options' contextFiles must be an array",
                ));
            }
        },
        skills: skills_from_wire(options.get("skills"))?,
    })
}

fn skills_from_wire(value: Option<&Value>) -> Result<Option<Vec<crate::skills::Skill>>, WireError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(skills)) => {
            let mut resolved = Vec::new();
            for skill in skills {
                let fields = fields(skill, "skill")?;
                let source_info = required_value(fields, "sourceInfo", "skill")?;
                resolved.push(crate::skills::Skill {
                    name: required_string(fields, "name", "skill")?,
                    description: required_string(fields, "description", "skill")?,
                    file_path: required_string(fields, "filePath", "skill")?,
                    base_dir: required_string(fields, "baseDir", "skill")?,
                    source_info: super::descriptors::source_info_from_wire(source_info)?,
                    disable_model_invocation: required_bool(
                        fields,
                        "disableModelInvocation",
                        "skill",
                    )?,
                });
            }
            Ok(Some(resolved))
        }
        Some(_) => Err(payload_error("the skills must be an array")),
    }
}

const fn session_start_reason_to_wire(reason: SessionStartReason) -> &'static str {
    match reason {
        SessionStartReason::Startup => "startup",
        SessionStartReason::Resume => "resume",
        SessionStartReason::New => "new",
        SessionStartReason::Fork => "fork",
        SessionStartReason::Reload => "reload",
    }
}

fn session_start_reason_from_wire(
    text: &str,
    allowed: &[&str],
) -> Result<SessionStartReason, WireError> {
    if !allowed.contains(&text) {
        return Err(payload_error(&format!(
            "reason `{text}` is not one of the channel's reasons"
        )));
    }
    match text {
        "startup" => Ok(SessionStartReason::Startup),
        "resume" => Ok(SessionStartReason::Resume),
        "new" => Ok(SessionStartReason::New),
        "fork" => Ok(SessionStartReason::Fork),
        "reload" => Ok(SessionStartReason::Reload),
        _ => unreachable!("the allowed list above gates this arm"),
    }
}

const fn session_shutdown_reason_to_wire(reason: SessionShutdownReason) -> &'static str {
    match reason {
        SessionShutdownReason::New => "new",
        SessionShutdownReason::Resume => "resume",
        SessionShutdownReason::Fork => "fork",
        SessionShutdownReason::Quit => "quit",
        SessionShutdownReason::Reload => "reload",
    }
}

fn session_shutdown_reason_from_wire(
    text: &str,
    allowed: &[&str],
) -> Result<SessionShutdownReason, WireError> {
    if !allowed.contains(&text) {
        return Err(payload_error(&format!(
            "reason `{text}` is not one of the channel's reasons"
        )));
    }
    match text {
        "new" => Ok(SessionShutdownReason::New),
        "resume" => Ok(SessionShutdownReason::Resume),
        "fork" => Ok(SessionShutdownReason::Fork),
        "quit" => Ok(SessionShutdownReason::Quit),
        "reload" => Ok(SessionShutdownReason::Reload),
        _ => unreachable!("the allowed list above gates this arm"),
    }
}

const fn compaction_reason_to_wire(reason: CompactionReason) -> &'static str {
    match reason {
        CompactionReason::Manual => "manual",
        CompactionReason::Threshold => "threshold",
        CompactionReason::Overflow => "overflow",
    }
}

fn compaction_reason_from_wire(text: &str) -> Result<CompactionReason, WireError> {
    match text {
        "manual" => Ok(CompactionReason::Manual),
        "threshold" => Ok(CompactionReason::Threshold),
        "overflow" => Ok(CompactionReason::Overflow),
        _ => Err(payload_error(&format!(
            "compaction reason `{text}` is unknown"
        ))),
    }
}

const fn model_select_source_to_wire(source: ModelSelectSource) -> &'static str {
    match source {
        ModelSelectSource::Set => "set",
        ModelSelectSource::Cycle => "cycle",
        ModelSelectSource::Restore => "restore",
    }
}

fn model_select_source_from_wire(text: &str) -> Result<ModelSelectSource, WireError> {
    match text {
        "set" => Ok(ModelSelectSource::Set),
        "cycle" => Ok(ModelSelectSource::Cycle),
        "restore" => Ok(ModelSelectSource::Restore),
        _ => Err(payload_error(&format!(
            "model_select source `{text}` is unknown"
        ))),
    }
}

const fn fork_position_to_wire(position: ForkPosition) -> &'static str {
    match position {
        ForkPosition::Before => "before",
        ForkPosition::At => "at",
    }
}

fn fork_position_from_wire(text: &str) -> Result<ForkPosition, WireError> {
    match text {
        "before" => Ok(ForkPosition::Before),
        "at" => Ok(ForkPosition::At),
        _ => Err(payload_error(&format!("fork position `{text}` is unknown"))),
    }
}

const fn input_source_to_wire(source: InputSource) -> &'static str {
    match source {
        InputSource::Interactive => "interactive",
        InputSource::Rpc => "rpc",
        InputSource::Extension => "extension",
    }
}

fn input_source_from_wire(text: &str) -> Result<InputSource, WireError> {
    match text {
        "interactive" => Ok(InputSource::Interactive),
        "rpc" => Ok(InputSource::Rpc),
        "extension" => Ok(InputSource::Extension),
        _ => Err(payload_error(&format!("input source `{text}` is unknown"))),
    }
}

const fn streaming_behavior_to_wire(behavior: StreamingBehavior) -> &'static str {
    match behavior {
        StreamingBehavior::Steer => "steer",
        StreamingBehavior::FollowUp => "followUp",
    }
}

fn streaming_behavior_from_wire(text: &str) -> Result<StreamingBehavior, WireError> {
    match text {
        "steer" => Ok(StreamingBehavior::Steer),
        "followUp" => Ok(StreamingBehavior::FollowUp),
        _ => Err(payload_error(&format!(
            "streaming behavior `{text}` is unknown"
        ))),
    }
}

fn optional_id(
    fields: &Map<String, Value>,
    key: &str,
    channel: &str,
) -> Result<Option<String>, WireError> {
    match fields.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(_)) => optional_string(fields, key, channel),
        Some(_) => Err(payload_error(&format!(
            "{channel} field `{key}` must be a string or null"
        ))),
    }
}

fn required_usize(
    fields: &Map<String, Value>,
    key: &str,
    channel: &str,
) -> Result<usize, WireError> {
    let index = fields.get(key).and_then(Value::as_u64).ok_or_else(|| {
        payload_error(&format!(
            "{channel} field `{key}` must be a non-negative number"
        ))
    })?;
    usize::try_from(index).map_err(|_| {
        payload_error(&format!(
            "{channel} field `{key}` exceeds the range of usize"
        ))
    })
}

fn required_i64(fields: &Map<String, Value>, key: &str, channel: &str) -> Result<i64, WireError> {
    fields.get(key).and_then(Value::as_i64).ok_or_else(|| {
        payload_error(&format!(
            "{channel} field `{key}` must be an integer number"
        ))
    })
}

fn compaction_preparation_to_wire(preparation: &CompactionPreparation) -> Result<Value, WireError> {
    Ok(json!({
        "firstKeptEntryId": preparation.first_kept_entry_id,
        "messagesToSummarize": serde_to_wire(
            &preparation.messages_to_summarize,
            "messagesToSummarize",
        )?,
        "turnPrefixMessages": serde_to_wire(
            &preparation.turn_prefix_messages,
            "turnPrefixMessages",
        )?,
        "isSplitTurn": preparation.is_split_turn,
        "tokensBefore": preparation.tokens_before,
        "fileOps": {
            // Sets render as sorted string arrays; the order is a display of
            // the set, the domain never reads it back as order.
            "read": file_ops_array(&preparation.file_ops.read),
            "written": file_ops_array(&preparation.file_ops.written),
            "edited": file_ops_array(&preparation.file_ops.edited),
        },
        "settings": {
            "enabled": preparation.settings.enabled,
            "reserveTokens": preparation.settings.reserve_tokens,
            "keepRecentTokens": preparation.settings.keep_recent_tokens,
        },
    }))
}

fn file_ops_array(paths: &std::collections::HashSet<String>) -> Value {
    let mut sorted: Vec<String> = paths.iter().cloned().collect();
    sorted.sort();
    Value::from(sorted)
}

fn compaction_preparation_from_wire(value: &Value) -> Result<CompactionPreparation, WireError> {
    let preparation = fields(value, "compaction_preparation")?;
    Ok(CompactionPreparation {
        first_kept_entry_id: required_string(
            preparation,
            "firstKeptEntryId",
            "compaction_preparation",
        )?,
        messages_to_summarize: serde_from_wire(
            required_value(preparation, "messagesToSummarize", "compaction_preparation")?.clone(),
            "messagesToSummarize",
        )?,
        turn_prefix_messages: serde_from_wire(
            required_value(preparation, "turnPrefixMessages", "compaction_preparation")?.clone(),
            "turnPrefixMessages",
        )?,
        is_split_turn: required_bool(preparation, "isSplitTurn", "compaction_preparation")?,
        tokens_before: required_u64(preparation, "tokensBefore", "compaction_preparation")?,
        previous_summary: optional_string(
            preparation,
            "previousSummary",
            "compaction_preparation",
        )?,
        file_ops: file_ops_from_wire(required_value(
            preparation,
            "fileOps",
            "compaction_preparation",
        )?)?,
        settings: compaction_settings_from_wire(required_value(
            preparation,
            "settings",
            "compaction_preparation",
        )?)?,
    })
}

fn file_ops_from_wire(value: &Value) -> Result<FileOperations, WireError> {
    let file_ops = fields(value, "file_ops")?;
    Ok(FileOperations {
        read: required_string_array(file_ops, "read", "file_ops")?
            .into_iter()
            .collect(),
        written: required_string_array(file_ops, "written", "file_ops")?
            .into_iter()
            .collect(),
        edited: required_string_array(file_ops, "edited", "file_ops")?
            .into_iter()
            .collect(),
    })
}

fn compaction_settings_from_wire(value: &Value) -> Result<CompactionSettings, WireError> {
    let settings = fields(value, "compaction_settings")?;
    Ok(CompactionSettings {
        enabled: required_bool(settings, "enabled", "compaction_settings")?,
        reserve_tokens: required_i64(settings, "reserveTokens", "compaction_settings")?,
        keep_recent_tokens: required_i64(settings, "keepRecentTokens", "compaction_settings")?,
    })
}

fn tree_preparation_to_wire(preparation: &TreePreparation) -> Result<Value, WireError> {
    let mut payload = json!({
        "targetId": preparation.target_id,
        "oldLeafId": preparation.old_leaf_id.clone().map_or(Value::Null, Value::from),
        "commonAncestorId": preparation
            .common_ancestor_id
            .clone()
            .map_or(Value::Null, Value::from),
        "entriesToSummarize": serde_to_wire(
            &preparation.entries_to_summarize,
            "entriesToSummarize",
        )?,
        "userWantsSummary": preparation.user_wants_summary,
    });
    if let Some(custom_instructions) = &preparation.custom_instructions {
        payload["customInstructions"] = json!(custom_instructions);
    }
    if let Some(replace_instructions) = preparation.replace_instructions {
        payload["replaceInstructions"] = json!(replace_instructions);
    }
    if let Some(label) = &preparation.label {
        payload["label"] = json!(label);
    }
    Ok(payload)
}

fn tree_preparation_from_wire(value: &Value) -> Result<TreePreparation, WireError> {
    let preparation = fields(value, "tree_preparation")?;
    Ok(TreePreparation {
        target_id: required_string(preparation, "targetId", "tree_preparation")?,
        old_leaf_id: optional_id(preparation, "oldLeafId", "tree_preparation")?,
        common_ancestor_id: optional_id(preparation, "commonAncestorId", "tree_preparation")?,
        entries_to_summarize: serde_from_wire(
            required_value(preparation, "entriesToSummarize", "tree_preparation")?.clone(),
            "entriesToSummarize",
        )?,
        user_wants_summary: required_bool(preparation, "userWantsSummary", "tree_preparation")?,
        custom_instructions: optional_string(
            preparation,
            "customInstructions",
            "tree_preparation",
        )?,
        replace_instructions: optional_bool(
            preparation,
            "replaceInstructions",
            "tree_preparation",
        )?,
        label: optional_string(preparation, "label", "tree_preparation")?,
    })
}

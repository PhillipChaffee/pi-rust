//! Session events on the JSON stdout protocol, upstream's
//! `src/modes/json-event.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! `message_update` stream events. `message_start` seeds the message, deltas
//! rebuild it, and `message_end` carries the final authoritative form.
//! Cumulative usage, tool-call ids, and tool names stay on the wire because
//! their size is constant.

use pi_agent_core::types::AgentMessage;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Message;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;

use crate::core::agent_session::AgentSessionEvent;
use crate::core::agent_session::SummarizationSource;
use crate::extensions::reasons::CompactionReason;

fn to_wire<T: Serialize>(value: &T) -> Result<Value, String> {
    serde_json::to_value(value).map_err(|error| error.to_string())
}

/// Upstream's `"manual" | "threshold" | "overflow"`.
const fn compaction_reason(reason: CompactionReason) -> &'static str {
    match reason {
        CompactionReason::Manual => "manual",
        CompactionReason::Threshold => "threshold",
        CompactionReason::Overflow => "overflow",
    }
}

/// Project one session event onto the JSON and RPC stdout protocols.
///
/// The wire names are upstream's: `snake_case` `type` discriminators with
/// `camelCase` payload fields. Absent optional values stay absent rather than
/// becoming JSON `null`, because upstream's `JSON.stringify` drops
/// `undefined` properties.
///
/// # Errors
/// Upstream's two typing guards: a `message_update` whose message is not an
/// assistant message, and a `toolcall_start` whose content index does not
/// point at a tool call.
#[expect(
    clippy::too_many_lines,
    reason = "the mapper mirrors upstream's event union arm for arm"
)]
pub fn to_json_event(event: &AgentSessionEvent) -> Result<Value, String> {
    match event {
        AgentSessionEvent::AgentStart => Ok(json!({ "type": "agent_start" })),
        AgentSessionEvent::AgentEnd {
            messages,
            will_retry,
        } => Ok(json!({
            "type": "agent_end",
            "messages": to_wire(messages)?,
            "willRetry": will_retry,
        })),
        AgentSessionEvent::TurnStart => Ok(json!({ "type": "turn_start" })),
        AgentSessionEvent::TurnEnd {
            message,
            tool_results,
        } => Ok(json!({
            "type": "turn_end",
            "message": to_wire(message)?,
            "toolResults": to_wire(tool_results)?,
        })),
        AgentSessionEvent::MessageStart { message } => Ok(json!({
            "type": "message_start",
            "message": to_wire(message)?,
        })),
        AgentSessionEvent::MessageUpdate {
            message,
            assistant_message_event,
        } => {
            let AgentMessage::Standard(Message::Assistant(assistant)) = message else {
                return Err("message_update message is not an assistant message".to_owned());
            };
            Ok(json!({
                "type": "message_update",
                "usage": to_wire(&assistant.usage)?,
                "assistantMessageEvent": to_json_assistant_message_event(
                    assistant_message_event,
                )?,
            }))
        }
        AgentSessionEvent::MessageEnd { message } => Ok(json!({
            "type": "message_end",
            "message": to_wire(message)?,
        })),
        AgentSessionEvent::ToolExecutionStart {
            tool_call_id,
            tool_name,
            args,
        } => Ok(json!({
            "type": "tool_execution_start",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "args": args,
        })),
        AgentSessionEvent::ToolExecutionUpdate {
            tool_call_id,
            tool_name,
            args,
            partial_result,
        } => Ok(json!({
            "type": "tool_execution_update",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "args": args,
            "partialResult": to_wire(partial_result)?,
        })),
        AgentSessionEvent::ToolExecutionEnd {
            tool_call_id,
            tool_name,
            result,
            is_error,
        } => Ok(json!({
            "type": "tool_execution_end",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "result": to_wire(result)?,
            "isError": is_error,
        })),
        AgentSessionEvent::AgentSettled => Ok(json!({ "type": "agent_settled" })),
        AgentSessionEvent::QueueUpdate {
            steering,
            follow_up,
        } => Ok(json!({
            "type": "queue_update",
            "steering": steering,
            "followUp": follow_up,
        })),
        AgentSessionEvent::CompactionStart { reason } => Ok(json!({
            "type": "compaction_start",
            "reason": compaction_reason(*reason),
        })),
        AgentSessionEvent::EntryAppended { entry } => Ok(json!({
            "type": "entry_appended",
            "entry": to_wire(entry)?,
        })),
        AgentSessionEvent::SessionInfoChanged { name } => {
            let mut payload = json!({ "type": "session_info_changed" });
            if let Some(name) = name {
                payload["name"] = json!(name);
            }
            Ok(payload)
        }
        AgentSessionEvent::ThinkingLevelChanged { level } => Ok(json!({
            "type": "thinking_level_changed",
            "level": to_wire(level)?,
        })),
        AgentSessionEvent::CompactionEnd {
            reason,
            result,
            aborted,
            will_retry,
            error_message,
        } => {
            let mut payload = json!({
                "type": "compaction_end",
                "reason": compaction_reason(*reason),
                "aborted": aborted,
                "willRetry": will_retry,
            });
            if let Some(result) = result {
                payload["result"] = to_wire(result)?;
            }
            if let Some(error_message) = error_message {
                payload["errorMessage"] = json!(error_message);
            }
            Ok(payload)
        }
        AgentSessionEvent::AutoRetryStart {
            attempt,
            max_attempts,
            delay_ms,
            error_message,
        } => Ok(json!({
            "type": "auto_retry_start",
            "attempt": attempt,
            "maxAttempts": max_attempts,
            "delayMs": delay_ms,
            "errorMessage": error_message,
        })),
        AgentSessionEvent::AutoRetryEnd {
            success,
            attempt,
            final_error,
        } => {
            let mut payload = json!({
                "type": "auto_retry_end",
                "success": success,
                "attempt": attempt,
            });
            if let Some(final_error) = final_error {
                payload["finalError"] = json!(final_error);
            }
            Ok(payload)
        }
        AgentSessionEvent::SummarizationRetryScheduled {
            attempt,
            max_attempts,
            delay_ms,
            error_message,
        } => Ok(json!({
            "type": "summarization_retry_scheduled",
            "attempt": attempt,
            "maxAttempts": max_attempts,
            "delayMs": delay_ms,
            "errorMessage": error_message,
        })),
        AgentSessionEvent::SummarizationRetryAttemptStart { source } => {
            let mut payload = json!({ "type": "summarization_retry_attempt_start" });
            match source {
                SummarizationSource::BranchSummary => {
                    payload["source"] = json!("branchSummary");
                }
                SummarizationSource::Compaction(reason) => {
                    payload["source"] = json!("compaction");
                    payload["reason"] = json!(compaction_reason(*reason));
                }
            }
            Ok(payload)
        }
        AgentSessionEvent::SummarizationRetryFinished => {
            Ok(json!({ "type": "summarization_retry_finished" }))
        }
        AgentSessionEvent::BashExecutionUpdate { id, delta } => {
            let mut payload = json!({
                "type": "bash_execution_update",
                "delta": delta,
            });
            if let Some(id) = id {
                payload["id"] = json!(id);
            }
            Ok(payload)
        }
    }
}

/// Strip the cumulative snapshot from one assistant stream event.
fn to_json_assistant_message_event(event: &AssistantMessageEvent) -> Result<Value, String> {
    if let AssistantMessageEvent::ToolcallStart {
        content_index,
        partial,
    } = event
    {
        let index = usize::try_from(*content_index).unwrap_or(usize::MAX);
        let Some(AssistantBlock::ToolCall(tool_call)) = partial.content.get(index) else {
            return Err(format!(
                "toolcall_start content at index {content_index} is not a tool call"
            ));
        };
        let mut delta = to_wire(event)?;
        if let Value::Object(object) = &mut delta {
            object.remove("partial");
            object.insert("id".to_owned(), json!(tool_call.id));
            object.insert("toolName".to_owned(), json!(tool_call.name));
        }
        return Ok(delta);
    }

    let mut value = to_wire(event)?;
    if let Value::Object(object) = &mut value {
        object.remove("partial");
    }
    Ok(value)
}

#[cfg(test)]
mod tests;

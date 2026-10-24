//! The [`HandlerResult`] wire mappings, upstream's per-event result
//! interfaces as `serde_json::Value`. A handler's absence stays the wire's
//! `null`; shapes the upstream types as optional read arbitrarily as absent
//! or `null` on the way in and write omit-style on the way out.

use serde_json::{Value, json};

use crate::extensions::results::{
    BeforeAgentStartResult, ContextResult, ExtensionBranchSummary, HandlerResult, InputResult,
    MessageEndResult, ProjectTrustDecision, ProjectTrustResult, SessionBeforeResult,
    ToolCallResult, ToolResultResult, UserBashResult,
};

use super::{
    WireError, fields, optional_bool, optional_string, optional_string_array, optional_u64,
    payload_error, required_bool, required_string, required_u64, required_value, serde_from_wire,
    serde_to_wire,
};

/// Every channel whose handlers can return a result, upstream's result
/// interfaces. Channels outside it must read back the wire's `null`; the
/// table-driven tests iterate this table.
pub const RESULT_CHANNELS: [&str; 15] = [
    "project_trust",
    "resources_discover",
    "session_before_switch",
    "session_before_fork",
    "session_before_compact",
    "session_before_tree",
    "context",
    "before_provider_request",
    "before_provider_headers",
    "before_agent_start",
    "input",
    "user_bash",
    "tool_call",
    "tool_result",
    "message_end",
];

/// Maps one handler result into its `call_result` value.
///
/// # Errors
///
/// Returns [`WireError`] when a serde-backed member stops short of the
/// wire's JSON.
#[expect(
    clippy::too_many_lines,
    reason = "the result table is one match, the wire's complete result index"
)]
pub fn result_to_wire(result: &HandlerResult) -> Result<Value, WireError> {
    match result {
        HandlerResult::SessionBefore(SessionBeforeResult::Switch { cancel }) => {
            Ok(json!({ "cancel": cancel }))
        }
        HandlerResult::SessionBefore(SessionBeforeResult::Fork {
            cancel,
            skip_conversation_restore,
        }) => Ok(json!({
            "cancel": cancel,
            "skipConversationRestore": skip_conversation_restore,
        })),
        HandlerResult::SessionBefore(SessionBeforeResult::Compact { cancel, compaction }) => {
            let mut wire = json!({ "cancel": cancel });
            if let Some(compaction) = compaction {
                wire["compaction"] = compaction_result_to_wire(compaction)?;
            }
            Ok(wire)
        }
        HandlerResult::SessionBefore(SessionBeforeResult::Tree {
            cancel,
            summary,
            custom_instructions,
            replace_instructions,
            label,
        }) => {
            let mut wire = json!({ "cancel": cancel });
            if let Some(summary) = summary {
                wire["summary"] = branch_summary_to_wire(summary)?;
            }
            if let Some(custom_instructions) = custom_instructions {
                wire["customInstructions"] = json!(custom_instructions);
            }
            if let Some(replace_instructions) = replace_instructions {
                wire["replaceInstructions"] = json!(replace_instructions);
            }
            if let Some(label) = label {
                wire["label"] = json!(label);
            }
            Ok(wire)
        }
        HandlerResult::MessageEnd(MessageEndResult { message }) => match message {
            Some(message) => Ok(json!({ "message": serde_to_wire(message, "message")? })),
            None => Ok(json!({})),
        },
        HandlerResult::ToolCall(ToolCallResult {
            block,
            reason,
            terminate,
        }) => {
            let mut wire = json!({ "block": block, "terminate": terminate });
            if let Some(reason) = reason {
                wire["reason"] = json!(reason);
            }
            Ok(wire)
        }
        HandlerResult::ToolResult(ToolResultResult {
            content,
            details,
            is_error,
            usage,
        }) => {
            let mut wire = json!({});
            if let Some(content) = content {
                wire["content"] = serde_to_wire(content, "content")?;
            }
            if let Some(details) = details {
                wire["details"] = details.clone();
            }
            if let Some(is_error) = is_error {
                wire["isError"] = json!(is_error);
            }
            if let Some(usage) = usage {
                wire["usage"] = serde_to_wire(usage, "usage")?;
            }
            Ok(wire)
        }
        HandlerResult::Context(ContextResult { messages }) => match messages {
            Some(messages) => Ok(json!({ "messages": serde_to_wire(messages, "messages")? })),
            None => Ok(json!({})),
        },
        HandlerResult::BeforeProviderRequest(payload) => Ok(payload.clone()),
        HandlerResult::BeforeProviderHeaders(headers) => serde_to_wire(headers, "headers"),
        HandlerResult::BeforeAgentStart(BeforeAgentStartResult {
            message,
            system_prompt,
        }) => {
            let mut wire = json!({});
            if let Some(message) = message {
                wire["message"] = custom_message_to_wire(message)?;
            }
            if let Some(system_prompt) = system_prompt {
                wire["systemPrompt"] = json!(system_prompt);
            }
            Ok(wire)
        }
        HandlerResult::ResourcesDiscover(result) => Ok(json!({
            "skillPaths": result.skill_paths,
            "promptPaths": result.prompt_paths,
            "themePaths": result.theme_paths,
        })),
        HandlerResult::Input(InputResult::Handled) => Ok(json!({ "action": "handled" })),
        HandlerResult::Input(InputResult::Transform { text, images }) => {
            let mut wire = json!({ "action": "transform", "text": text });
            if let Some(images) = images {
                wire["images"] = serde_to_wire(images, "images")?;
            }
            Ok(wire)
        }
        HandlerResult::Input(InputResult::Continue) => Ok(json!({ "action": "continue" })),
        // Upstream's operations/result members carry a BashResult the port's
        // seam (#149) owns; the wire crosses the handled decision alone.
        HandlerResult::UserBash(UserBashResult { handled }) => Ok(json!({ "handled": handled })),
        HandlerResult::ProjectTrust(ProjectTrustResult { trusted }) => {
            trusted.as_ref().map_or_else(
                || Ok(json!({})),
                |decision| Ok(json!({ "trusted": decision_to_wire(*decision) })),
            )
        }
        // The generic emit and the session_before absence ride the same
        // wire value: the handler returned nothing.
        HandlerResult::SessionBefore(SessionBeforeResult::None) | HandlerResult::None => {
            Ok(Value::Null)
        }
    }
}

const fn decision_to_wire(decision: ProjectTrustDecision) -> &'static str {
    match decision {
        ProjectTrustDecision::Undecided => "undecided",
        ProjectTrustDecision::Yes => "yes",
        ProjectTrustDecision::No => "no",
    }
}

fn compaction_result_to_wire(
    compaction: &crate::compaction::CompactionResult,
) -> Result<Value, WireError> {
    let mut wire = json!({
        "summary": compaction.summary,
        "firstKeptEntryId": compaction.first_kept_entry_id,
        "tokensBefore": compaction.tokens_before,
        "usage": serde_to_wire(&compaction.usage, "usage")?,
    });
    if let Some(estimated_tokens_after) = compaction.estimated_tokens_after {
        wire["estimatedTokensAfter"] = json!(estimated_tokens_after);
    }
    if let Some(details) = &compaction.details {
        wire["details"] = details.clone();
    }
    Ok(wire)
}

fn branch_summary_to_wire(summary: &ExtensionBranchSummary) -> Result<Value, WireError> {
    let mut wire = json!({ "summary": summary.summary });
    if let Some(details) = &summary.details {
        wire["details"] = details.clone();
    }
    if let Some(usage) = &summary.usage {
        wire["usage"] = serde_to_wire(usage, "usage")?;
    }
    Ok(wire)
}

fn custom_message_to_wire(
    payload: &crate::extensions::reasons::CustomMessagePayload,
) -> Result<Value, WireError> {
    let mut wire = json!({
        "customType": payload.custom_type,
        "content": serde_to_wire(&payload.content, "content")?,
        "display": payload.display,
    });
    if let Some(details) = &payload.details {
        wire["details"] = details.clone();
    }
    Ok(wire)
}

/// Rebuilds the handler result a `call_result` value carried.
///
/// The wire's `null` reads as [`HandlerResult::None`] on every channel. A
/// channel without a result surface rejects any other value, so a confused
/// extension surfaces as a handler failure the caller reads as `None`,
/// never as a silent protocol break.
///
/// # Errors
///
/// Returns [`WireError`] when the channel is unknown, the value is non-null
/// on a channel without a result surface, or a field is missing or
/// mistyped.
#[expect(
    clippy::too_many_lines,
    reason = "the result table is one match, the wire's complete result index"
)]
pub fn result_from_wire(channel: &str, result: &Value) -> Result<HandlerResult, WireError> {
    if result.is_null() {
        return Ok(HandlerResult::None);
    }
    match channel {
        "session_before_switch" => {
            let fields = fields(result, channel)?;
            Ok(HandlerResult::SessionBefore(SessionBeforeResult::Switch {
                cancel: optional_bool(fields, "cancel", channel)?.unwrap_or(false),
            }))
        }
        "session_before_fork" => {
            let fields = fields(result, channel)?;
            Ok(HandlerResult::SessionBefore(SessionBeforeResult::Fork {
                cancel: optional_bool(fields, "cancel", channel)?.unwrap_or(false),
                skip_conversation_restore: optional_bool(
                    fields,
                    "skipConversationRestore",
                    channel,
                )?
                .unwrap_or(false),
            }))
        }
        "session_before_compact" => {
            let fields = fields(result, channel)?;
            Ok(HandlerResult::SessionBefore(SessionBeforeResult::Compact {
                cancel: optional_bool(fields, "cancel", channel)?.unwrap_or(false),
                compaction: match fields.get("compaction") {
                    None | Some(Value::Null) => None,
                    Some(value) => Some(compaction_result_from_wire(value)?),
                },
            }))
        }
        "session_before_tree" => {
            let fields = fields(result, channel)?;
            Ok(HandlerResult::SessionBefore(SessionBeforeResult::Tree {
                cancel: optional_bool(fields, "cancel", channel)?.unwrap_or(false),
                summary: match fields.get("summary") {
                    None | Some(Value::Null) => None,
                    Some(value) => Some(branch_summary_from_wire(value)?),
                },
                custom_instructions: optional_string(fields, "customInstructions", channel)?,
                replace_instructions: optional_bool(fields, "replaceInstructions", channel)?,
                label: optional_string(fields, "label", channel)?,
            }))
        }
        "message_end" => Ok(HandlerResult::MessageEnd(MessageEndResult {
            message: match fields(result, channel)?.get("message") {
                None | Some(Value::Null) => None,
                Some(message) => Some(serde_from_wire(message.clone(), "message")?),
            },
        })),
        "tool_call" => {
            let fields = fields(result, channel)?;
            Ok(HandlerResult::ToolCall(ToolCallResult {
                block: optional_bool(fields, "block", channel)?.unwrap_or(false),
                reason: optional_string(fields, "reason", channel)?,
                terminate: optional_bool(fields, "terminate", channel)?.unwrap_or(false),
            }))
        }
        "tool_result" => {
            let fields = fields(result, channel)?;
            Ok(HandlerResult::ToolResult(ToolResultResult {
                content: match fields.get("content") {
                    None | Some(Value::Null) => None,
                    Some(content) => Some(serde_from_wire(content.clone(), "content")?),
                },
                details: fields.get("details").cloned(),
                is_error: optional_bool(fields, "isError", channel)?,
                usage: match fields.get("usage") {
                    None | Some(Value::Null) => None,
                    Some(usage) => Some(serde_from_wire(usage.clone(), "usage")?),
                },
            }))
        }
        "context" => Ok(HandlerResult::Context(ContextResult {
            messages: match fields(result, channel)?.get("messages") {
                None | Some(Value::Null) => None,
                Some(messages) => Some(serde_from_wire(messages.clone(), "messages")?),
            },
        })),
        "before_provider_request" => Ok(HandlerResult::BeforeProviderRequest(result.clone())),
        "before_provider_headers" => Ok(HandlerResult::BeforeProviderHeaders(serde_from_wire(
            result.clone(),
            "headers",
        )?)),
        "before_agent_start" => {
            let fields = fields(result, channel)?;
            Ok(HandlerResult::BeforeAgentStart(BeforeAgentStartResult {
                message: match fields.get("message") {
                    None | Some(Value::Null) => None,
                    Some(value) => Some(custom_message_from_wire(value)?),
                },
                system_prompt: optional_string(fields, "systemPrompt", channel)?,
            }))
        }
        "resources_discover" => {
            let fields = fields(result, channel)?;
            Ok(HandlerResult::ResourcesDiscover(
                crate::extensions::results::ResourcesDiscoverResult {
                    skill_paths: optional_string_array(fields, "skillPaths", channel)?
                        .unwrap_or_default(),
                    prompt_paths: optional_string_array(fields, "promptPaths", channel)?
                        .unwrap_or_default(),
                    theme_paths: optional_string_array(fields, "themePaths", channel)?
                        .unwrap_or_default(),
                },
            ))
        }
        "input" => {
            let fields = fields(result, channel)?;
            match required_string(fields, "action", channel)?.as_str() {
                "handled" => Ok(HandlerResult::Input(InputResult::Handled)),
                "transform" => Ok(HandlerResult::Input(InputResult::Transform {
                    text: required_string(fields, "text", channel)?,
                    images: match fields.get("images") {
                        None | Some(Value::Null) => None,
                        Some(images) => Some(serde_from_wire(images.clone(), "images")?),
                    },
                })),
                "continue" => Ok(HandlerResult::Input(InputResult::Continue)),
                action => Err(payload_error(&format!(
                    "{channel} action `{action}` is unknown"
                ))),
            }
        }
        // Upstream's operations/result members carry a BashResult the port's
        // seam (#149) owns; the wire crosses the handled decision alone.
        "user_bash" => Ok(HandlerResult::UserBash(UserBashResult {
            handled: required_bool(fields(result, channel)?, "handled", channel)?,
        })),
        "project_trust" => Ok(HandlerResult::ProjectTrust(ProjectTrustResult {
            trusted: match fields(result, channel)?.get("trusted") {
                None | Some(Value::Null) => None,
                Some(decision) => {
                    Some(decision_from_wire(decision.as_str().ok_or_else(|| {
                        payload_error("project_trust's trusted must be a string")
                    })?)?)
                }
            },
        })),
        unknown => Err(WireError::new(format!(
            "unknown result channel `{unknown}`"
        ))),
    }
}

fn decision_from_wire(text: &str) -> Result<ProjectTrustDecision, WireError> {
    match text {
        "undecided" => Ok(ProjectTrustDecision::Undecided),
        "yes" => Ok(ProjectTrustDecision::Yes),
        "no" => Ok(ProjectTrustDecision::No),
        _ => Err(payload_error(&format!(
            "project_trust decision `{text}` is unknown"
        ))),
    }
}

fn compaction_result_from_wire(
    value: &Value,
) -> Result<crate::compaction::CompactionResult, WireError> {
    let fields = fields(value, "compaction_result")?;
    Ok(crate::compaction::CompactionResult {
        summary: required_string(fields, "summary", "compaction_result")?,
        first_kept_entry_id: required_string(fields, "firstKeptEntryId", "compaction_result")?,
        tokens_before: required_u64(fields, "tokensBefore", "compaction_result")?,
        estimated_tokens_after: optional_u64(fields, "estimatedTokensAfter", "compaction_result")?,
        usage: serde_from_wire(
            required_value(fields, "usage", "compaction_result")?.clone(),
            "usage",
        )?,
        details: fields.get("details").cloned(),
    })
}

fn branch_summary_from_wire(value: &Value) -> Result<ExtensionBranchSummary, WireError> {
    let fields = fields(value, "branch_summary")?;
    Ok(ExtensionBranchSummary {
        summary: required_string(fields, "summary", "branch_summary")?,
        details: fields.get("details").cloned(),
        usage: match fields.get("usage") {
            None | Some(Value::Null) => None,
            Some(usage) => Some(serde_from_wire(usage.clone(), "usage")?),
        },
    })
}

fn custom_message_from_wire(
    value: &Value,
) -> Result<crate::extensions::reasons::CustomMessagePayload, WireError> {
    let fields = fields(value, "custom_message")?;
    Ok(crate::extensions::reasons::CustomMessagePayload {
        custom_type: required_string(fields, "customType", "custom_message")?,
        content: serde_from_wire(
            required_value(fields, "content", "custom_message")?.clone(),
            "content",
        )?,
        display: required_bool(fields, "display", "custom_message")?,
        details: fields.get("details").cloned(),
    })
}

//! RPC protocol types, upstream's `src/modes/rpc/rpc-types.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Commands arrive as JSON lines on stdin; responses, session events, and
//! extension UI requests leave as JSON lines on stdout. Every command carries
//! an optional `id` the response echoes for correlation.

use pi_agent_core::types::QueueMode;
use pi_agent_core::types::ThinkingLevel;
use pi_ai::types::ImageContent;
use pi_ai::types::Model;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use crate::extensions::reasons::StreamingBehavior;
use crate::source_info::SourceInfo;

/// One stdin command, upstream's `RpcCommand` union.
#[derive(Clone, Debug, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum RpcCommand {
    /// Start a prompt run; the response follows the preflight result.
    Prompt {
        /// The correlation id.
        id: Option<String>,
        /// The message text.
        message: String,
        /// Images attached to the message.
        images: Option<Vec<ImageContent>>,
        /// How the message joins a running stream.
        streaming_behavior: Option<StreamingBehavior>,
    },
    /// Queue a steering message into the running stream.
    Steer {
        /// The correlation id.
        id: Option<String>,
        /// The message text.
        message: String,
        /// Images attached to the message.
        images: Option<Vec<ImageContent>>,
    },
    /// Queue a follow-up message for after the run settles.
    FollowUp {
        /// The correlation id.
        id: Option<String>,
        /// The message text.
        message: String,
        /// Images attached to the message.
        images: Option<Vec<ImageContent>>,
    },
    /// Abort the running turn.
    Abort {
        /// The correlation id.
        id: Option<String>,
    },
    /// Drop every queued steering and follow-up message.
    ClearQueue {
        /// The correlation id.
        id: Option<String>,
    },
    /// Replace the current session with a fresh one.
    NewSession {
        /// The correlation id.
        id: Option<String>,
        /// The session file the new one tracks as its parent.
        parent_session: Option<String>,
    },
    /// Read the session state.
    GetState {
        /// The correlation id.
        id: Option<String>,
    },
    /// Select a model by provider and id.
    SetModel {
        /// The correlation id.
        id: Option<String>,
        /// The provider id.
        provider: String,
        /// The model id.
        model_id: String,
    },
    /// Cycle to the next model.
    CycleModel {
        /// The correlation id.
        id: Option<String>,
    },
    /// List the available models.
    GetAvailableModels {
        /// The correlation id.
        id: Option<String>,
    },
    /// Set the thinking level.
    SetThinkingLevel {
        /// The correlation id.
        id: Option<String>,
        /// The new level.
        level: ThinkingLevel,
    },
    /// Cycle the thinking level.
    CycleThinkingLevel {
        /// The correlation id.
        id: Option<String>,
    },
    /// List the levels the current model offers.
    GetAvailableThinkingLevels {
        /// The correlation id.
        id: Option<String>,
    },
    /// Set how queued steering messages drain.
    SetSteeringMode {
        /// The correlation id.
        id: Option<String>,
        /// The queue mode.
        mode: QueueMode,
    },
    /// Set how queued follow-up messages drain.
    SetFollowUpMode {
        /// The correlation id.
        id: Option<String>,
        /// The queue mode.
        mode: QueueMode,
    },
    /// Compact the session context.
    Compact {
        /// The correlation id.
        id: Option<String>,
        /// Extra instructions for the summary call.
        custom_instructions: Option<String>,
    },
    /// Toggle threshold compaction.
    SetAutoCompaction {
        /// The correlation id.
        id: Option<String>,
        /// Whether compaction runs automatically.
        enabled: bool,
    },
    /// Toggle automatic retries.
    SetAutoRetry {
        /// The correlation id.
        id: Option<String>,
        /// Whether retries run automatically.
        enabled: bool,
    },
    /// Abort the retry in flight.
    AbortRetry {
        /// The correlation id.
        id: Option<String>,
    },
    /// Run a bash command and record its output.
    Bash {
        /// The correlation id.
        id: Option<String>,
        /// The command line.
        command: String,
        /// Whether the recorded message stays out of the LLM context.
        exclude_from_context: Option<bool>,
    },
    /// Abort the running bash command.
    AbortBash {
        /// The correlation id.
        id: Option<String>,
    },
    /// Read the session statistics.
    GetSessionStats {
        /// The correlation id.
        id: Option<String>,
    },
    /// Export the session to HTML.
    ExportHtml {
        /// The correlation id.
        id: Option<String>,
        /// The output path; generated under the agent dir when absent.
        output_path: Option<String>,
    },
    /// Replace the current session with the one at `sessionPath`.
    SwitchSession {
        /// The correlation id.
        id: Option<String>,
        /// The session file to open.
        session_path: String,
    },
    /// Fork from an entry.
    Fork {
        /// The correlation id.
        id: Option<String>,
        /// The entry to fork from.
        entry_id: String,
    },
    /// Clone the active branch into a new session.
    Clone {
        /// The correlation id.
        id: Option<String>,
    },
    /// List the user messages available for forking.
    GetForkMessages {
        /// The correlation id.
        id: Option<String>,
    },
    /// Read the session entries, optionally after a cursor.
    GetEntries {
        /// The correlation id.
        id: Option<String>,
        /// Return only the entries after the one with this id.
        since: Option<String>,
    },
    /// Read the session entry tree.
    GetTree {
        /// The correlation id.
        id: Option<String>,
    },
    /// Read the last assistant message's text.
    GetLastAssistantText {
        /// The correlation id.
        id: Option<String>,
    },
    /// Set the session display name.
    SetSessionName {
        /// The correlation id.
        id: Option<String>,
        /// The new name; surrounding whitespace is trimmed.
        name: String,
    },
    /// Read every message in the session.
    GetMessages {
        /// The correlation id.
        id: Option<String>,
    },
    /// List the commands a prompt may invoke.
    GetCommands {
        /// The correlation id.
        id: Option<String>,
    },
}

/// The wire `type` every command variant carries.
pub const COMMAND_TYPES: [&str; 33] = [
    "prompt",
    "steer",
    "follow_up",
    "abort",
    "clear_queue",
    "new_session",
    "get_state",
    "set_model",
    "cycle_model",
    "get_available_models",
    "set_thinking_level",
    "cycle_thinking_level",
    "get_available_thinking_levels",
    "set_steering_mode",
    "set_follow_up_mode",
    "compact",
    "set_auto_compaction",
    "set_auto_retry",
    "abort_retry",
    "bash",
    "abort_bash",
    "get_session_stats",
    "export_html",
    "switch_session",
    "fork",
    "clone",
    "get_fork_messages",
    "get_entries",
    "get_tree",
    "get_last_assistant_text",
    "set_session_name",
    "get_messages",
    "get_commands",
];

impl RpcCommand {
    /// The correlation id the response echoes.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        let id = match self {
            Self::Prompt { id, .. }
            | Self::Steer { id, .. }
            | Self::FollowUp { id, .. }
            | Self::Abort { id }
            | Self::ClearQueue { id }
            | Self::NewSession { id, .. }
            | Self::GetState { id }
            | Self::SetModel { id, .. }
            | Self::CycleModel { id }
            | Self::GetAvailableModels { id }
            | Self::SetThinkingLevel { id, .. }
            | Self::CycleThinkingLevel { id }
            | Self::GetAvailableThinkingLevels { id }
            | Self::SetSteeringMode { id, .. }
            | Self::SetFollowUpMode { id, .. }
            | Self::Compact { id, .. }
            | Self::SetAutoCompaction { id, .. }
            | Self::SetAutoRetry { id, .. }
            | Self::AbortRetry { id }
            | Self::Bash { id, .. }
            | Self::AbortBash { id }
            | Self::GetSessionStats { id }
            | Self::ExportHtml { id, .. }
            | Self::SwitchSession { id, .. }
            | Self::Fork { id, .. }
            | Self::Clone { id }
            | Self::GetForkMessages { id }
            | Self::GetEntries { id, .. }
            | Self::GetTree { id }
            | Self::GetLastAssistantText { id }
            | Self::SetSessionName { id, .. }
            | Self::GetMessages { id }
            | Self::GetCommands { id } => id,
        };
        id.as_deref()
    }

    /// The wire `type` the response echoes as `command`.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Prompt { .. } => "prompt",
            Self::Steer { .. } => "steer",
            Self::FollowUp { .. } => "follow_up",
            Self::Abort { .. } => "abort",
            Self::ClearQueue { .. } => "clear_queue",
            Self::NewSession { .. } => "new_session",
            Self::GetState { .. } => "get_state",
            Self::SetModel { .. } => "set_model",
            Self::CycleModel { .. } => "cycle_model",
            Self::GetAvailableModels { .. } => "get_available_models",
            Self::SetThinkingLevel { .. } => "set_thinking_level",
            Self::CycleThinkingLevel { .. } => "cycle_thinking_level",
            Self::GetAvailableThinkingLevels { .. } => "get_available_thinking_levels",
            Self::SetSteeringMode { .. } => "set_steering_mode",
            Self::SetFollowUpMode { .. } => "set_follow_up_mode",
            Self::Compact { .. } => "compact",
            Self::SetAutoCompaction { .. } => "set_auto_compaction",
            Self::SetAutoRetry { .. } => "set_auto_retry",
            Self::AbortRetry { .. } => "abort_retry",
            Self::Bash { .. } => "bash",
            Self::AbortBash { .. } => "abort_bash",
            Self::GetSessionStats { .. } => "get_session_stats",
            Self::ExportHtml { .. } => "export_html",
            Self::SwitchSession { .. } => "switch_session",
            Self::Fork { .. } => "fork",
            Self::Clone { .. } => "clone",
            Self::GetForkMessages { .. } => "get_fork_messages",
            Self::GetEntries { .. } => "get_entries",
            Self::GetTree { .. } => "get_tree",
            Self::GetLastAssistantText { .. } => "get_last_assistant_text",
            Self::SetSessionName { .. } => "set_session_name",
            Self::GetMessages { .. } => "get_messages",
            Self::GetCommands { .. } => "get_commands",
        }
    }
}

/// The session state `get_state` returns, upstream's `RpcSessionState`.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcSessionState {
    /// The current model, absent while none is selected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<Model>,
    /// The thinking level.
    pub thinking_level: ThinkingLevel,
    /// Whether a turn is streaming.
    pub is_streaming: bool,
    /// Whether compaction is running.
    pub is_compacting: bool,
    /// How queued steering messages drain.
    pub steering_mode: QueueMode,
    /// How queued follow-up messages drain.
    pub follow_up_mode: QueueMode,
    /// The session file, absent for in-memory sessions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_file: Option<String>,
    /// The session id.
    pub session_id: String,
    /// The display name, absent when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_name: Option<String>,
    /// Whether threshold compaction runs automatically.
    pub auto_compaction_enabled: bool,
    /// The message count.
    pub message_count: usize,
    /// The queued steering and follow-up count.
    pub pending_message_count: usize,
}

/// Where an invocable command comes from, upstream's `RpcSlashCommand.source`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RpcCommandSource {
    /// A registered extension command.
    Extension,
    /// A prompt template.
    Prompt,
    /// A skill.
    Skill,
}

/// One command a prompt may invoke, upstream's `RpcSlashCommand`.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcSlashCommand {
    /// The command name without the leading slash; skills carry the
    /// `skill:` prefix.
    pub name: String,
    /// The human-readable description, absent when the source has none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// What kind of command this is.
    pub source: RpcCommandSource,
    /// The owning resource's provenance.
    pub source_info: SourceInfo,
}

/// The notification severity, upstream's `notifyType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RpcNotifyType {
    /// An informational notice.
    Info,
    /// A warning notice.
    Warning,
    /// An error notice.
    Error,
}

/// Where an extension widget renders, upstream's `WidgetPlacement`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum RpcWidgetPlacement {
    /// Above the editor.
    AboveEditor,
    /// Below the editor.
    BelowEditor,
}

/// One extension UI request on the stdout wire, upstream's
/// `RpcExtensionUIRequest` union.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RpcExtensionUiRequest {
    /// The request id the response echoes.
    pub id: String,
    /// The request body.
    pub method: RpcUiMethod,
}

/// The request bodies, upstream's `method` discriminated union.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RpcUiMethod {
    /// A selector dialog.
    Select {
        /// The dialog title.
        title: String,
        /// The choices.
        options: Vec<String>,
        /// The auto-dismiss deadline in milliseconds.
        timeout: Option<u64>,
    },
    /// A confirmation dialog.
    Confirm {
        /// The dialog title.
        title: String,
        /// The question.
        message: String,
        /// The auto-dismiss deadline in milliseconds.
        timeout: Option<u64>,
    },
    /// A text input dialog.
    Input {
        /// The dialog title.
        title: String,
        /// The placeholder text.
        placeholder: Option<String>,
        /// The auto-dismiss deadline in milliseconds.
        timeout: Option<u64>,
    },
    /// A multi-line editor dialog.
    Editor {
        /// The dialog title.
        title: String,
        /// The initial text.
        prefill: Option<String>,
    },
    /// A fire-and-forget notice.
    Notify {
        /// The notice text.
        message: String,
        /// The severity, absent for a plain notice.
        notify_type: Option<RpcNotifyType>,
    },
    /// A status-bar entry.
    SetStatus {
        /// The status key.
        status_key: String,
        /// The text; absent clears the entry.
        status_text: Option<String>,
    },
    /// A widget above or below the editor.
    SetWidget {
        /// The widget key.
        widget_key: String,
        /// The rendered lines; absent clears the widget.
        widget_lines: Option<Vec<String>>,
        /// Where the widget renders.
        widget_placement: Option<RpcWidgetPlacement>,
    },
    /// The terminal title.
    SetTitle {
        /// The title.
        title: String,
    },
    /// The editor's text.
    SetEditorText {
        /// The text.
        text: String,
    },
}

impl RpcUiMethod {
    /// The wire `method` name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Select { .. } => "select",
            Self::Confirm { .. } => "confirm",
            Self::Input { .. } => "input",
            Self::Editor { .. } => "editor",
            Self::Notify { .. } => "notify",
            Self::SetStatus { .. } => "setStatus",
            Self::SetWidget { .. } => "setWidget",
            Self::SetTitle { .. } => "setTitle",
            Self::SetEditorText { .. } => "set_editor_text",
        }
    }
}

impl Serialize for RpcExtensionUiRequest {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("type", "extension_ui_request")?;
        map.serialize_entry("id", &self.id)?;
        map.serialize_entry("method", self.method.name())?;
        let mut body = serde_json::Map::new();
        match &self.method {
            RpcUiMethod::Select {
                title,
                options,
                timeout,
            } => {
                body.insert("title".to_owned(), Value::from(title.clone()));
                body.insert("options".to_owned(), Value::from(options.clone()));
                insert_optional(&mut body, "timeout", timeout.as_ref());
            }
            RpcUiMethod::Confirm {
                title,
                message,
                timeout,
            } => {
                body.insert("title".to_owned(), Value::from(title.clone()));
                body.insert("message".to_owned(), Value::from(message.clone()));
                insert_optional(&mut body, "timeout", timeout.as_ref());
            }
            RpcUiMethod::Input {
                title,
                placeholder,
                timeout,
            } => {
                body.insert("title".to_owned(), Value::from(title.clone()));
                insert_optional(&mut body, "placeholder", placeholder.as_ref());
                insert_optional(&mut body, "timeout", timeout.as_ref());
            }
            RpcUiMethod::Editor { title, prefill } => {
                body.insert("title".to_owned(), Value::from(title.clone()));
                insert_optional(&mut body, "prefill", prefill.as_ref());
            }
            RpcUiMethod::Notify {
                message,
                notify_type,
            } => {
                body.insert("message".to_owned(), Value::from(message.clone()));
                insert_optional(&mut body, "notifyType", notify_type.as_ref());
            }
            RpcUiMethod::SetStatus {
                status_key,
                status_text,
            } => {
                body.insert("statusKey".to_owned(), Value::from(status_key.clone()));
                body.insert(
                    "statusText".to_owned(),
                    status_text.clone().map_or(Value::Null, Value::from),
                );
            }
            RpcUiMethod::SetWidget {
                widget_key,
                widget_lines,
                widget_placement,
            } => {
                body.insert("widgetKey".to_owned(), Value::from(widget_key.clone()));
                body.insert(
                    "widgetLines".to_owned(),
                    widget_lines.clone().map_or(Value::Null, Value::from),
                );
                insert_optional(&mut body, "widgetPlacement", widget_placement.as_ref());
            }
            RpcUiMethod::SetTitle { title } => {
                body.insert("title".to_owned(), Value::from(title.clone()));
            }
            RpcUiMethod::SetEditorText { text } => {
                body.insert("text".to_owned(), Value::from(text.clone()));
            }
        }
        for (key, value) in body {
            map.serialize_entry(&key, &value)?;
        }
        map.end()
    }
}

/// Insert an optional body field, dropping the absent case the way
/// `JSON.stringify` drops `undefined`.
fn insert_optional<T: Serialize>(
    map: &mut serde_json::Map<String, Value>,
    key: &str,
    value: Option<&T>,
) {
    if let Some(value) = value
        && let Ok(value) = serde_json::to_value(value)
    {
        map.insert(key.to_owned(), value);
    }
}

/// One stdin reply to an extension UI request, upstream's
/// `RpcExtensionUIResponse` union.
///
/// Upstream discriminates the three arms by presence (`value`, `confirmed`,
/// `cancelled: true`); a struct with optionals reads all three.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcExtensionUiResponse {
    /// The request id the reply answers.
    pub id: String,
    /// The selector/input/editor value.
    pub value: Option<String>,
    /// The confirmation answer.
    pub confirmed: Option<bool>,
    /// Whether the dialog was dismissed.
    pub cancelled: Option<bool>,
}

impl RpcExtensionUiResponse {
    /// Whether the reply dismissed the dialog, upstream's
    /// `"cancelled" in response && response.cancelled`.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled == Some(true)
    }
}

#[cfg(test)]
mod tests;

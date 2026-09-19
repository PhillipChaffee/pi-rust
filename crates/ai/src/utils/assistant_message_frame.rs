//! Compact, replayable assistant-message progress, ported from
//! `packages/ai/src/utils/assistant-message-frame.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`AssistantMessageFrameEncoder`] turns the live assistant-message event
//! stream into small frames that replay to the same partial message through
//! [`reduce_assistant_message_frames`] without replaying deltas an older
//! queued event already covered. Terminal settlement (`done`/`error`) is
//! intentionally excluded and must be persisted separately.
//!
//! Porting restatements: upstream snapshots provider-shaped partials and
//! whitelists their public fields (the `index`, `partialJson`, `streamIndex`
//! scratch fields of the provider adapters never reach a frame). Rust's
//! typed [`crate::types`] structs cannot carry those scratch fields, so the
//! whitelist is statically upheld. The shared-live-partial hazard — a queued
//! event whose `partial` has grown past the deltas it carries — ports
//! directly: events move their partial into the encoder, which tracks
//! per-block offsets instead of replaying covered prefixes.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::types::{
    AssistantMessage, AssistantMessageEvent, TextContent, ThinkingContent, ToolCall,
};
use crate::utils::json_parse::parse_streaming_json;

/// Compact, replayable assistant-message progress. Terminal settlement is
/// intentionally excluded and must be persisted separately.
#[expect(
    clippy::large_enum_variant,
    reason = "the start frame carries the full message snapshot by wire contract; boxing the public field would reshape the enum and its serde form"
)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all_fields = "camelCase")]
pub enum AssistantMessageFrame {
    /// The stream opened; carries the start snapshot.
    #[serde(rename = "start")]
    Start {
        /// The cloned start message.
        partial: AssistantMessage,
    },
    /// A text block opened, wire `"text_start"`.
    #[serde(rename = "text_start")]
    TextStart {
        /// The block's index in `content`.
        content_index: u64,
        /// The cloned text content at start.
        content: TextContent,
    },
    /// Text grew, wire `"text_delta"`.
    #[serde(rename = "text_delta")]
    TextDelta {
        /// The block's index in `content`.
        content_index: u64,
        /// The uncovered appended text.
        delta: String,
    },
    /// A text block closed authoritatively, wire `"text_end"`.
    #[serde(rename = "text_end")]
    TextEnd {
        /// The block's index in `content`.
        content_index: u64,
        /// The authoritative full text.
        content: String,
        /// The authoritative signature, serialized as `textSignature`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text_signature: Option<String>,
    },
    /// A thinking block opened, wire `"thinking_start"`.
    #[serde(rename = "thinking_start")]
    ThinkingStart {
        /// The block's index in `content`.
        content_index: u64,
        /// The cloned thinking content at start.
        content: ThinkingContent,
    },
    /// Thinking grew, wire `"thinking_delta"`.
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        /// The block's index in `content`.
        content_index: u64,
        /// The uncovered appended thinking text.
        delta: String,
    },
    /// A thinking block closed authoritatively, wire `"thinking_end"`.
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        /// The block's index in `content`.
        content_index: u64,
        /// The authoritative full thinking text.
        content: String,
        /// The authoritative thinking signature, serialized as
        /// `thinkingSignature`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thinking_signature: Option<String>,
        /// The authoritative redaction flag, serialized as `redacted`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        redacted: Option<bool>,
    },
    /// A tool call opened, wire `"toolcall_start"`.
    #[serde(rename = "toolcall_start")]
    ToolcallStart {
        /// The block's index in `content`.
        content_index: u64,
        /// The cloned tool call at start.
        tool_call: ToolCall,
    },
    /// The tool-call JSON caught up to the start snapshot, wire
    /// `"toolcall_checkpoint"`.
    #[serde(rename = "toolcall_checkpoint")]
    ToolcallCheckpoint {
        /// The block's index in `content`.
        content_index: u64,
        /// The JSON that replays the arguments up to this point.
        json: String,
    },
    /// Tool-call arguments grew, wire `"toolcall_delta"`.
    #[serde(rename = "toolcall_delta")]
    ToolcallDelta {
        /// The block's index in `content`.
        content_index: u64,
        /// The uncovered appended JSON.
        delta: String,
    },
    /// A tool call closed authoritatively, wire `"toolcall_end"`.
    #[serde(rename = "toolcall_end")]
    ToolcallEnd {
        /// The block's index in `content`.
        content_index: u64,
        /// The authoritative tool-call id.
        id: String,
        /// The authoritative tool name.
        name: String,
        /// The authoritative parsed arguments.
        arguments: Map<String, Value>,
        /// The authoritative thought signature, serialized as
        /// `thoughtSignature`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
        /// The authoritative namespace, serialized as `namespace`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
    },
}

/// Per-block encoder state: how much of the live block the emitted frames
/// already cover.
enum EncoderBlockState {
    Text {
        covered_chars: usize,
        delta_chars: usize,
    },
    Thinking {
        covered_chars: usize,
        delta_chars: usize,
    },
    ToolCall {
        caught_up: bool,
        catchup_json: String,
        snapshot_arguments: String,
    },
}

impl EncoderBlockState {
    const fn kind(&self) -> &'static str {
        match self {
            Self::Text { .. } => "text",
            Self::Thinking { .. } => "thinking",
            Self::ToolCall { .. } => "toolCall",
        }
    }
}

/// Per-block reducer state: what each replayed block has seen so far.
enum ReducerBlockState {
    Text { ended: bool },
    Thinking { ended: bool },
    ToolCall { ended: bool, json: String },
}

fn clone_start_message(message: &AssistantMessage) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: message.api.clone(),
        provider: message.provider.clone(),
        model: message.model.clone(),
        response_model: message.response_model.clone(),
        response_id: message.response_id.clone(),
        provider_thinking_level: message.provider_thinking_level.clone(),
        diagnostics: message.diagnostics.clone(),
        usage: message.usage,
        stop_reason: crate::types::StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: message.timestamp,
    }
}

fn assert_content_index(content_index: u64) -> Result<usize, String> {
    usize::try_from(content_index)
        .map_err(|_| format!("Invalid assistant message frame contentIndex: {content_index}"))
}

fn serialized_arguments(arguments: &Map<String, Value>) -> Result<String, String> {
    serde_json::to_string(&Value::Object(arguments.clone()))
        .map_err(|_| String::from("Tool-call arguments are not JSON-serializable"))
}

/// The parsed arguments of an empty tool-call JSON fragment, upstream's
/// `EMPTY_PARSED_TOOL_ARGUMENTS`.
fn empty_parsed_tool_arguments() -> Result<String, String> {
    let parsed = parse_streaming_json(None);
    parsed.as_object().map_or_else(
        || Ok(String::from("{}")),
        |object| {
            serde_json::to_string(&Value::Object(object.clone()))
                .map_err(|_| String::from("Tool-call arguments are not JSON-serializable"))
        },
    )
}

/// Whether the snapshot value is a JSON prefix of the current value: every
/// string is a prefix of a string extension, every array element and object
/// entry recurses, and everything else compares for equality.
fn is_json_prefix(snapshot: &Value, current: &Value) -> bool {
    match (snapshot, current) {
        (Value::String(snapshot), Value::String(current)) => current.starts_with(snapshot.as_str()),
        (Value::Array(snapshot), Value::Array(current)) => {
            snapshot.len() <= current.len()
                && snapshot
                    .iter()
                    .zip(current.iter())
                    .all(|(snapshot, current)| is_json_prefix(snapshot, current))
        }
        (Value::Object(snapshot), Value::Object(current)) => snapshot.iter().all(|(key, value)| {
            current
                .get(key)
                .is_some_and(|current| is_json_prefix(value, current))
        }),
        _ => snapshot == current,
    }
}

/// Encodes one assistant stream.
///
/// The `partial` of each event is a shared live accumulator; the encoder uses
/// per-block offsets to avoid replaying deltas already visible when an older
/// queued event is consumed.
#[derive(Default)]
pub struct AssistantMessageFrameEncoder {
    started: bool,
    terminal: bool,
    blocks: std::collections::HashMap<usize, EncoderBlockState>,
}

impl std::fmt::Debug for AssistantMessageFrameEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssistantMessageFrameEncoder")
            .field("started", &self.started)
            .field("terminal", &self.terminal)
            .field("blocks", &self.blocks.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl AssistantMessageFrameEncoder {
    /// Encode one assistant-message event into a frame, or [`None`] when the
    /// event produces no frame (covered deltas, terminal events).
    ///
    /// # Errors
    /// Returns the error message upstream throws when the event sequence
    /// violates the stream protocol: events after a terminal event, more
    /// than one start, events before start, content indices pointing at
    /// missing or wrong-kind blocks, and blocks that start more than once.
    pub fn encode(
        &mut self,
        event: AssistantMessageEvent,
    ) -> Result<Option<AssistantMessageFrame>, String> {
        let event_type = event_type_name(&event);
        if self.terminal {
            return Err(format!(
                "Assistant message event {event_type} follows a terminal event"
            ));
        }

        match event {
            AssistantMessageEvent::Start { partial } => {
                if self.started {
                    return Err(String::from(
                        "Assistant message stream contains more than one start event",
                    ));
                }
                self.started = true;
                Ok(Some(AssistantMessageFrame::Start {
                    partial: clone_start_message(&partial),
                }))
            }
            AssistantMessageEvent::Done { .. } => {
                if !self.started {
                    return Err(String::from(
                        "Assistant message done event appears before start",
                    ));
                }
                self.terminal = true;
                Ok(None)
            }
            AssistantMessageEvent::Error { .. } => {
                self.terminal = true;
                Ok(None)
            }
            _ if !self.started => Err(format!(
                "Assistant message {event_type} event appears before start"
            )),
            AssistantMessageEvent::TextStart {
                content_index,
                partial,
            }
            | AssistantMessageEvent::ThinkingStart {
                content_index,
                partial,
            }
            | AssistantMessageEvent::ToolcallStart {
                content_index,
                partial,
            } => self.encode_block_start(event_type, content_index, &partial),
            AssistantMessageEvent::TextDelta {
                content_index,
                delta,
                ..
            } => self.encode_text_delta(content_index, &delta, "text"),
            AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                ..
            } => self.encode_text_delta(content_index, &delta, "thinking"),
            AssistantMessageEvent::ToolcallDelta {
                content_index,
                delta,
                ..
            } => self.encode_toolcall_delta(content_index, &delta),
            AssistantMessageEvent::TextEnd {
                content_index,
                content,
                partial,
            } => self.encode_text_end(content_index, content, &partial),
            AssistantMessageEvent::ThinkingEnd {
                content_index,
                content,
                partial,
            } => self.encode_thinking_end(content_index, content, &partial),
            AssistantMessageEvent::ToolcallEnd {
                content_index,
                tool_call,
                partial,
            } => self.encode_toolcall_end(content_index, tool_call, &partial),
        }
    }

    fn encode_text_end(
        &mut self,
        content_index: u64,
        content: String,
        partial: &AssistantMessage,
    ) -> Result<Option<AssistantMessageFrame>, String> {
        let block = event_block("text_end", content_index, partial)?;
        let crate::types::AssistantBlock::Text(text) = block else {
            return Err(format!(
                "text_end event points to {} block at index {content_index}",
                block_type_name(partial, content_index)
            ));
        };
        let index = assert_content_index(content_index)?;
        self.end_block(index, "text")?;
        Ok(Some(AssistantMessageFrame::TextEnd {
            content_index,
            content,
            text_signature: text.text_signature.clone(),
        }))
    }

    fn encode_thinking_end(
        &mut self,
        content_index: u64,
        content: String,
        partial: &AssistantMessage,
    ) -> Result<Option<AssistantMessageFrame>, String> {
        let block = event_block("thinking_end", content_index, partial)?;
        let crate::types::AssistantBlock::Thinking(thinking) = block else {
            return Err(format!(
                "thinking_end event points to {} block at index {content_index}",
                block_type_name(partial, content_index)
            ));
        };
        let index = assert_content_index(content_index)?;
        self.end_block(index, "thinking")?;
        Ok(Some(AssistantMessageFrame::ThinkingEnd {
            content_index,
            content,
            thinking_signature: thinking.thinking_signature.clone(),
            redacted: thinking.redacted,
        }))
    }

    fn encode_toolcall_end(
        &mut self,
        content_index: u64,
        tool_call: ToolCall,
        partial: &AssistantMessage,
    ) -> Result<Option<AssistantMessageFrame>, String> {
        let block = event_block("toolcall_end", content_index, partial)?;
        if !matches!(block, crate::types::AssistantBlock::ToolCall(_)) {
            return Err(format!(
                "toolcall_end event points to {} block at index {content_index}",
                block_type_name(partial, content_index)
            ));
        }
        let index = assert_content_index(content_index)?;
        self.end_block(index, "toolCall")?;
        Ok(Some(AssistantMessageFrame::ToolcallEnd {
            content_index,
            id: tool_call.id,
            name: tool_call.name,
            arguments: tool_call.arguments,
            thought_signature: tool_call.thought_signature,
            namespace: tool_call.namespace,
        }))
    }

    fn encode_block_start(
        &mut self,
        event_type: &str,
        content_index: u64,
        partial: &AssistantMessage,
    ) -> Result<Option<AssistantMessageFrame>, String> {
        let index = assert_content_index(content_index)?;
        let Some(block) = partial.content.get(index) else {
            return Err(format!(
                "{event_type} event has no content block at index {content_index}"
            ));
        };
        match block {
            crate::types::AssistantBlock::Text(text) => {
                if event_type != "text_start" {
                    return Err(format!(
                        "{event_type} event points to text block at index {index}"
                    ));
                }
                self.start_block(
                    index,
                    EncoderBlockState::Text {
                        covered_chars: text.text.chars().count(),
                        delta_chars: 0,
                    },
                )?;
                Ok(Some(AssistantMessageFrame::TextStart {
                    content_index,
                    content: text.clone(),
                }))
            }
            crate::types::AssistantBlock::Thinking(thinking) => {
                if event_type != "thinking_start" {
                    return Err(format!(
                        "{event_type} event points to thinking block at index {index}"
                    ));
                }
                self.start_block(
                    index,
                    EncoderBlockState::Thinking {
                        covered_chars: thinking.thinking.chars().count(),
                        delta_chars: 0,
                    },
                )?;
                Ok(Some(AssistantMessageFrame::ThinkingStart {
                    content_index,
                    content: thinking.clone(),
                }))
            }
            crate::types::AssistantBlock::ToolCall(tool_call) => {
                if event_type != "toolcall_start" {
                    return Err(format!(
                        "{event_type} event points to toolCall block at index {index}"
                    ));
                }
                let snapshot_arguments = serialized_arguments(&tool_call.arguments)?;
                let caught_up = snapshot_arguments == empty_parsed_tool_arguments()?;
                self.start_block(
                    index,
                    EncoderBlockState::ToolCall {
                        caught_up,
                        catchup_json: String::new(),
                        snapshot_arguments: if caught_up {
                            String::new()
                        } else {
                            snapshot_arguments
                        },
                    },
                )?;
                Ok(Some(AssistantMessageFrame::ToolcallStart {
                    content_index,
                    tool_call: tool_call.clone(),
                }))
            }
        }
    }

    fn encode_toolcall_delta(
        &mut self,
        content_index: u64,
        delta: &str,
    ) -> Result<Option<AssistantMessageFrame>, String> {
        let index = assert_content_index(content_index)?;
        let EncoderBlockState::ToolCall {
            caught_up,
            catchup_json,
            snapshot_arguments,
        } = self.block_state(index, "toolCall")?
        else {
            return Err(String::from("Unreachable tool-call encoder state"));
        };
        if *caught_up {
            return Ok(if delta.is_empty() {
                None
            } else {
                Some(AssistantMessageFrame::ToolcallDelta {
                    content_index,
                    delta: delta.to_owned(),
                })
            });
        }
        catchup_json.push_str(delta);
        let arguments_value = parse_streaming_json(Some(catchup_json));
        let serialized = serde_json::to_string(&arguments_value)
            .map_err(|_| String::from("Tool-call arguments are not JSON-serializable"))?;
        if serialized != *snapshot_arguments {
            // Legacy grammar calls include the initial input in toolcall_start,
            // but their JSON delta stream still begins at an empty input. Its
            // parsed arguments can therefore extend, rather than exactly
            // reproduce, the start snapshot.
            let snapshot_value = parse_streaming_json(Some(snapshot_arguments));
            if !is_json_prefix(&snapshot_value, &arguments_value) {
                return Ok(None);
            }
        }
        *caught_up = true;
        *snapshot_arguments = String::new();
        let json = std::mem::take(catchup_json);
        Ok(if json.is_empty() {
            None
        } else {
            Some(AssistantMessageFrame::ToolcallCheckpoint {
                content_index,
                json,
            })
        })
    }

    fn encode_text_delta(
        &mut self,
        content_index: u64,
        delta: &str,
        kind: &str,
    ) -> Result<Option<AssistantMessageFrame>, String> {
        let index = assert_content_index(content_index)?;
        let state = self.block_state(index, kind)?;
        let (covered_chars, delta_chars) = match state {
            EncoderBlockState::Text {
                covered_chars,
                delta_chars,
            }
            | EncoderBlockState::Thinking {
                covered_chars,
                delta_chars,
            } => (covered_chars, delta_chars),
            EncoderBlockState::ToolCall { .. } => {
                return Err(String::from("Unreachable text encoder state"));
            }
        };
        let delta_start = *delta_chars;
        *delta_chars += delta.chars().count();
        let covered = covered_chars.saturating_sub(delta_start);
        if covered >= delta.chars().count() {
            return Ok(None);
        }
        let uncovered: String = {
            let skip = delta.chars().take(covered).count();
            delta.chars().skip(skip).collect()
        };
        Ok(Some(if kind == "text" {
            AssistantMessageFrame::TextDelta {
                content_index,
                delta: uncovered,
            }
        } else {
            AssistantMessageFrame::ThinkingDelta {
                content_index,
                delta: uncovered,
            }
        }))
    }

    fn start_block(&mut self, index: usize, state: EncoderBlockState) -> Result<(), String> {
        if self.blocks.contains_key(&index) {
            return Err(format!(
                "Assistant message block {index} starts more than once"
            ));
        }
        self.blocks.insert(index, state);
        Ok(())
    }

    fn block_state(&mut self, index: usize, kind: &str) -> Result<&mut EncoderBlockState, String> {
        let Some(state) = self.blocks.get_mut(&index) else {
            return Err(format!(
                "Assistant message {kind} block {index} has not started"
            ));
        };
        if state.kind() != kind {
            return Err(format!(
                "Assistant message block {index} is {}, not {kind}",
                state.kind()
            ));
        }
        Ok(state)
    }

    fn end_block(&mut self, index: usize, kind: &str) -> Result<(), String> {
        self.block_state(index, kind)?;
        self.blocks.remove(&index);
        Ok(())
    }
}

const fn event_type_name(event: &AssistantMessageEvent) -> &'static str {
    match event {
        AssistantMessageEvent::Start { .. } => "start",
        AssistantMessageEvent::TextStart { .. } => "text_start",
        AssistantMessageEvent::TextDelta { .. } => "text_delta",
        AssistantMessageEvent::TextEnd { .. } => "text_end",
        AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
        AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
        AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
        AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start",
        AssistantMessageEvent::ToolcallDelta { .. } => "toolcall_delta",
        AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end",
        AssistantMessageEvent::Done { .. } => "done",
        AssistantMessageEvent::Error { .. } => "error",
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "error-message indexing mirrors the JS caller; out-of-range indices resolve to the missing-block name"
)]
fn block_type_name(partial: &AssistantMessage, content_index: u64) -> &str {
    partial
        .content
        .get(content_index as usize)
        .map_or("missing", |block| match block {
            crate::types::AssistantBlock::Text(_) => "text",
            crate::types::AssistantBlock::Thinking(_) => "thinking",
            crate::types::AssistantBlock::ToolCall(_) => "toolCall",
        })
}

fn event_block<'a>(
    event_type: &str,
    content_index: u64,
    partial: &'a AssistantMessage,
) -> Result<&'a crate::types::AssistantBlock, String> {
    let index = assert_content_index(content_index)?;
    partial
        .content
        .get(index)
        .ok_or_else(|| format!("{event_type} event has no content block at index {content_index}"))
}

/// Replay compact frames into the partial message they encode, without
/// mutating them. Returns [`None`] when the iterable contains no start frame.
///
/// # Errors
/// Returns the error message upstream throws for malformed frame sequences:
/// frames before the start frame, more than one start frame, wrong block
/// kinds, frames after a block's end, and index gaps.
pub fn reduce_assistant_message_frames(
    frames: impl IntoIterator<Item = AssistantMessageFrame>,
) -> Result<Option<AssistantMessage>, String> {
    let mut message: Option<AssistantMessage> = None;
    let mut frame_before_start: Option<&'static str> = None;
    let mut states: std::collections::HashMap<usize, ReducerBlockState> =
        std::collections::HashMap::new();

    for frame in frames {
        let frame_type = frame_type_name(&frame);
        if let AssistantMessageFrame::Start { partial } = frame {
            if message.is_some() {
                return Err(String::from(
                    "Assistant message frame sequence contains more than one start frame",
                ));
            }
            if let Some(before) = frame_before_start {
                return Err(format!("{before} frame appears before the start frame"));
            }
            message = Some(partial);
            continue;
        }
        let Some(current) = message.as_mut() else {
            frame_before_start.get_or_insert(frame_type);
            continue;
        };

        apply_frame(current, &mut states, frame)?;
    }

    let Some(mut message) = message else {
        return Ok(None);
    };
    // Unfinished tool-call blocks replay their accumulated JSON as parsed
    // arguments; visit indexes in order so the parse order is deterministic.
    let mut indexes: Vec<usize> = states.keys().copied().collect();
    indexes.sort_unstable();
    for content_index in indexes {
        let ReducerBlockState::ToolCall { ended, json } = &states[&content_index] else {
            continue;
        };
        if *ended || json.is_empty() {
            continue;
        }
        let Some(crate::types::AssistantBlock::ToolCall(block)) =
            message.content.get_mut(content_index)
        else {
            return Err(String::from("Unreachable tool-call frame state"));
        };
        let parsed = parse_streaming_json(Some(json));
        block.arguments = parsed.as_object().cloned().unwrap_or_default();
    }

    Ok(Some(message))
}

const fn frame_type_name(frame: &AssistantMessageFrame) -> &'static str {
    match frame {
        AssistantMessageFrame::Start { .. } => "start",
        AssistantMessageFrame::TextStart { .. } => "text_start",
        AssistantMessageFrame::TextDelta { .. } => "text_delta",
        AssistantMessageFrame::TextEnd { .. } => "text_end",
        AssistantMessageFrame::ThinkingStart { .. } => "thinking_start",
        AssistantMessageFrame::ThinkingDelta { .. } => "thinking_delta",
        AssistantMessageFrame::ThinkingEnd { .. } => "thinking_end",
        AssistantMessageFrame::ToolcallStart { .. } => "toolcall_start",
        AssistantMessageFrame::ToolcallCheckpoint { .. } => "toolcall_checkpoint",
        AssistantMessageFrame::ToolcallDelta { .. } => "toolcall_delta",
        AssistantMessageFrame::ToolcallEnd { .. } => "toolcall_end",
    }
}

fn apply_frame(
    message: &mut AssistantMessage,
    states: &mut std::collections::HashMap<usize, ReducerBlockState>,
    frame: AssistantMessageFrame,
) -> Result<(), String> {
    match frame {
        AssistantMessageFrame::Start { .. } => Err(String::from(
            "Assistant message frame sequence contains more than one start frame",
        )),
        AssistantMessageFrame::TextStart {
            content_index,
            content,
        } => append_block(
            message,
            states,
            content_index,
            crate::types::AssistantBlock::Text(content),
            ReducerBlockState::Text { ended: false },
        ),
        AssistantMessageFrame::TextDelta {
            content_index,
            delta,
        } => apply_text_delta(message, states, content_index, &delta),
        AssistantMessageFrame::TextEnd {
            content_index,
            content,
            text_signature,
        } => apply_text_end(message, states, content_index, content, text_signature),
        AssistantMessageFrame::ThinkingStart {
            content_index,
            content,
        } => append_block(
            message,
            states,
            content_index,
            crate::types::AssistantBlock::Thinking(content),
            ReducerBlockState::Thinking { ended: false },
        ),
        AssistantMessageFrame::ThinkingDelta {
            content_index,
            delta,
        } => apply_thinking_delta(message, states, content_index, &delta),
        AssistantMessageFrame::ThinkingEnd {
            content_index,
            content,
            thinking_signature,
            redacted,
        } => apply_thinking_end(
            message,
            states,
            content_index,
            content,
            thinking_signature,
            redacted,
        ),
        AssistantMessageFrame::ToolcallStart {
            content_index,
            tool_call,
        } => append_block(
            message,
            states,
            content_index,
            crate::types::AssistantBlock::ToolCall(tool_call),
            ReducerBlockState::ToolCall {
                ended: false,
                json: String::new(),
            },
        ),
        AssistantMessageFrame::ToolcallCheckpoint {
            content_index,
            json,
        } => apply_toolcall_checkpoint(message, states, content_index, &json),
        AssistantMessageFrame::ToolcallDelta {
            content_index,
            delta,
        } => apply_toolcall_delta(message, states, content_index, &delta),
        AssistantMessageFrame::ToolcallEnd {
            content_index,
            id,
            name,
            arguments,
            thought_signature,
            namespace,
        } => apply_toolcall_end(
            message,
            states,
            content_index,
            id,
            name,
            arguments,
            thought_signature,
            namespace,
        ),
    }
}

fn apply_text_delta(
    message: &mut AssistantMessage,
    states: &mut std::collections::HashMap<usize, ReducerBlockState>,
    content_index: u64,
    delta: &str,
) -> Result<(), String> {
    let (block, _) = active_block(message, states, content_index, "text", "text_delta")?;
    if let crate::types::AssistantBlock::Text(text) = block {
        text.text.push_str(delta);
    }
    Ok(())
}

fn apply_text_end(
    message: &mut AssistantMessage,
    states: &mut std::collections::HashMap<usize, ReducerBlockState>,
    content_index: u64,
    content: String,
    text_signature: Option<String>,
) -> Result<(), String> {
    let (block, state) = active_block(message, states, content_index, "text", "text_end")?;
    if let crate::types::AssistantBlock::Text(text) = block {
        text.text = content;
        text.text_signature = text_signature;
    }
    if let ReducerBlockState::Text { ended } = state {
        *ended = true;
    }
    Ok(())
}

fn apply_thinking_delta(
    message: &mut AssistantMessage,
    states: &mut std::collections::HashMap<usize, ReducerBlockState>,
    content_index: u64,
    delta: &str,
) -> Result<(), String> {
    let (block, _) = active_block(message, states, content_index, "thinking", "thinking_delta")?;
    if let crate::types::AssistantBlock::Thinking(thinking) = block {
        thinking.thinking.push_str(delta);
    }
    Ok(())
}

fn apply_thinking_end(
    message: &mut AssistantMessage,
    states: &mut std::collections::HashMap<usize, ReducerBlockState>,
    content_index: u64,
    content: String,
    thinking_signature: Option<String>,
    redacted: Option<bool>,
) -> Result<(), String> {
    let (block, state) = active_block(message, states, content_index, "thinking", "thinking_end")?;
    if let crate::types::AssistantBlock::Thinking(thinking) = block {
        thinking.thinking = content;
        thinking.thinking_signature = thinking_signature;
        thinking.redacted = redacted;
    }
    if let ReducerBlockState::Thinking { ended } = state {
        *ended = true;
    }
    Ok(())
}

fn apply_toolcall_checkpoint(
    message: &mut AssistantMessage,
    states: &mut std::collections::HashMap<usize, ReducerBlockState>,
    content_index: u64,
    json: &str,
) -> Result<(), String> {
    let (block, state) = active_block(
        message,
        states,
        content_index,
        "toolCall",
        "toolcall_checkpoint",
    )?;
    if let (
        crate::types::AssistantBlock::ToolCall(block),
        ReducerBlockState::ToolCall {
            json: state_json, ..
        },
    ) = (block, state)
    {
        state_json.clear();
        state_json.push_str(json);
        let parsed = parse_streaming_json(Some(json));
        block.arguments = parsed.as_object().cloned().unwrap_or_default();
    }
    Ok(())
}

fn apply_toolcall_delta(
    message: &mut AssistantMessage,
    states: &mut std::collections::HashMap<usize, ReducerBlockState>,
    content_index: u64,
    delta: &str,
) -> Result<(), String> {
    let (_, state) = active_block(message, states, content_index, "toolCall", "toolcall_delta")?;
    if let ReducerBlockState::ToolCall { json, .. } = state {
        json.push_str(delta);
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the toolcall_end frame carries six fields that all land on the block"
)]
fn apply_toolcall_end(
    message: &mut AssistantMessage,
    states: &mut std::collections::HashMap<usize, ReducerBlockState>,
    content_index: u64,
    id: String,
    name: String,
    arguments: Map<String, Value>,
    thought_signature: Option<String>,
    namespace: Option<String>,
) -> Result<(), String> {
    let (block, state) = active_block(message, states, content_index, "toolCall", "toolcall_end")?;
    if let (
        crate::types::AssistantBlock::ToolCall(block),
        ReducerBlockState::ToolCall { ended, .. },
    ) = (block, state)
    {
        block.id = id;
        block.name = name;
        block.arguments = arguments;
        block.thought_signature = thought_signature;
        block.namespace = namespace;
        *ended = true;
    }
    Ok(())
}

fn append_block(
    message: &mut AssistantMessage,
    states: &mut std::collections::HashMap<usize, ReducerBlockState>,
    content_index: u64,
    block: crate::types::AssistantBlock,
    state: ReducerBlockState,
) -> Result<(), String> {
    let index = assert_content_index(content_index)?;
    if index != message.content.len() {
        let reason = if content_index < message.content.len() as u64 {
            "already exists"
        } else {
            "would leave a gap"
        };
        return Err(format!(
            "Cannot start assistant message block at index {content_index}: {reason}"
        ));
    }
    states.insert(index, state);
    message.content.push(block);
    Ok(())
}

fn active_block<'a>(
    message: &'a mut AssistantMessage,
    states: &'a mut std::collections::HashMap<usize, ReducerBlockState>,
    content_index: u64,
    expected_kind: &str,
    frame_type: &str,
) -> Result<
    (
        &'a mut crate::types::AssistantBlock,
        &'a mut ReducerBlockState,
    ),
    String,
> {
    let index = assert_content_index(content_index)?;
    let Some(block) = message.content.get_mut(index) else {
        return Err(format!(
            "{frame_type} frame has no started block at index {index}"
        ));
    };
    let Some(state) = states.get_mut(&index) else {
        return Err(format!(
            "{frame_type} frame has no started block at index {index}"
        ));
    };
    if state_kind(state) != expected_kind || block_kind(block) != expected_kind {
        return Err(format!(
            "{frame_type} frame expected {expected_kind} block at index {index}, found {}",
            block_kind(block)
        ));
    }
    if state_ended(state) {
        return Err(format!(
            "{frame_type} frame follows the end of block at index {index}"
        ));
    }
    Ok((block, state))
}

const fn state_kind(state: &ReducerBlockState) -> &'static str {
    match state {
        ReducerBlockState::Text { .. } => "text",
        ReducerBlockState::Thinking { .. } => "thinking",
        ReducerBlockState::ToolCall { .. } => "toolCall",
    }
}

const fn state_ended(state: &ReducerBlockState) -> bool {
    match state {
        ReducerBlockState::Text { ended }
        | ReducerBlockState::Thinking { ended }
        | ReducerBlockState::ToolCall { ended, .. } => *ended,
    }
}

const fn block_kind(block: &crate::types::AssistantBlock) -> &'static str {
    match block {
        crate::types::AssistantBlock::Text(_) => "text",
        crate::types::AssistantBlock::Thinking(_) => "thinking",
        crate::types::AssistantBlock::ToolCall(_) => "toolCall",
    }
}

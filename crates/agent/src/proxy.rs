//! The proxy stream function, upstream's `src/proxy.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`stream_proxy`] is the stream function apps pass when a server routes
//! the LLM calls: the request goes to the proxy server — which owns auth and
//! the provider call — and the server answers with the compact
//! [`ProxyAssistantMessageEvent`] wire, delta events with the partial field
//! stripped to reduce bandwidth. The client rebuilds the partial message
//! from them, so downstream consumers see the same
//! `AssistantMessageEventStream` a direct provider call produces.
//!
//! Porting restatements:
//!
//! - upstream's `ProxyMessageEventStream extends EventStream` carries the
//!   identical is-complete/extractor pair the pi-ai factory builds; the
//!   port returns that factory's [`AssistantMessageEventStream`];
//! - upstream sends the request through ambient `globalThis.fetch`; the
//!   port sends it through the [`pi_ai::http::HttpClient`] seam — the
//!   injected client when the options carry one, the process default
//!   otherwise, the same seam the pi-ai mock harness mounts on;
//! - the streamed tool-call arguments ride an ad-hoc `partialJson` field on
//!   the content object upstream; the accumulator tracks them in a side map
//!   keyed by content index and re-parses the accumulated JSON into
//!   `arguments` on every delta, `parseStreamingJson(..) || {}` folding a
//!   non-object parse into the empty map (the typed `arguments` field
//!   cannot hold one);
//! - the `toolcall_delta` arm reassigns the content slot to trigger JS
//!   reactivity; the Rust mutation is observed directly and the reassign
//!   has no counterpart;
//! - a `contentIndex` past the end fills the gap with empty text blocks —
//!   upstream's sparse array assignment leaves holes a Rust vector cannot
//!   represent, and no conformant server sends gaps (indexes are sequential
//!   by construction);
//! - unknown proxy event types are unreachable: the wire enum is closed and
//!   serde rejects unknown tags where upstream's `default:` arm warned.

use std::collections::BTreeMap;
use std::sync::Arc;

use pi_ai::http::client::read_body_text;
use pi_ai::http::{HttpClient, HttpMethod, HttpRequest, status_reason};
use pi_ai::types::{
    AssistantBlock, AssistantMessage, AssistantMessageEvent, CacheRetention, Context, Model,
    ProviderHeaders, StopReason, TextContent, ThinkingBudgets, ThinkingContent, ThinkingLevel,
    ToolCall, Transport, Usage,
};
use pi_ai::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use pi_ai::utils::json_parse::parse_streaming_json;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

/// The `ProxyMessageEventStream` upstream subclasses.
///
/// An [`AssistantMessageEventStream`] is already the `done`/`error`-completing,
/// message/error-extracting stream the subclass configures, so the port
/// returns that factory directly. Send the prompt through the proxy server
/// instead of calling LLM providers directly. Use this as the `stream_fn`
/// option when creating an [`Agent`](crate::agent::Agent) that needs to go
/// through a proxy.
///
/// # Panics
/// Called outside a tokio runtime context: the request runs on a spawned
/// task, upstream's immediately-invoked async arrow.
#[must_use]
pub fn stream_proxy(
    model: &Model,
    context: &Context,
    options: &ProxyStreamOptions,
) -> AssistantMessageEventStream {
    let stream = assistant_message_event_stream();
    let inner = stream.clone();
    let model = model.clone();
    let context = context.clone();
    let options = options.clone();
    tokio::spawn(async move {
        let signal = options.operation_signal();
        let mut accumulator = ProxyAccumulator::new(&model);
        let mut saw_terminal_event = false;
        let outcome = proxy_fetch_and_read(
            &model,
            &context,
            &options,
            &mut accumulator,
            &mut saw_terminal_event,
            &inner,
        )
        .await;
        match outcome {
            // A clean EOF without a done/error event means the server
            // dropped the response mid-stream. Surface it as an error
            // instead of leaving consumers waiting on a result that never
            // arrives.
            Ok(()) if !saw_terminal_event => {
                accumulator.partial.stop_reason = StopReason::Error;
                accumulator.partial.error_message = Some(
                    "Connection closed by proxy server before the response completed".to_owned(),
                );
                inner.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: accumulator.partial.clone(),
                });
                inner.end(None);
            }
            // The terminal event already settled the final result; `end`
            // without a result keeps it.
            Ok(()) => inner.end(None),
            Err(message) => {
                let reason = if signal.is_cancelled() {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                accumulator.partial.stop_reason = reason;
                accumulator.partial.error_message = Some(message);
                inner.push(AssistantMessageEvent::Error {
                    reason,
                    error: accumulator.partial.clone(),
                });
                inner.end(None);
            }
        }
    });
    stream
}

/// The proxy request the task drives, upstream's async IIFE body up to its
/// `catch`: build and send the request, stream the response lines through
/// the accumulator. `Ok` carries nothing — the caller owns the terminal
/// handling.
///
/// # Errors
/// The request failure, a non-ok status message, a line that fails to parse,
/// a proxy event that updates the wrong block kind, or a JSON body that
/// cannot serialize — each the error message the failure event carries.
async fn proxy_fetch_and_read(
    model: &Model,
    context: &Context,
    options: &ProxyStreamOptions,
    accumulator: &mut ProxyAccumulator,
    saw_terminal_event: &mut bool,
    stream: &AssistantMessageEventStream,
) -> Result<(), String> {
    let body = serde_json::to_vec(&serde_json::json!({
        "model": model,
        "context": context,
        "options": options.serializable(),
    }))
    .map_err(|error| error.to_string())?;
    let request = HttpRequest {
        method: HttpMethod::Post,
        url: format!("{}/api/stream", options.proxy_url),
        headers: vec![
            (
                "Authorization".to_owned(),
                format!("Bearer {}", options.auth_token),
            ),
            ("Content-Type".to_owned(), "application/json".to_owned()),
        ],
        body: Some(body.into()),
        timeout_ms: None,
        signal: options.operation_signal(),
    };
    let client = options.client();
    let mut response = client
        .execute(request)
        .await
        .map_err(|error| error.to_string())?;

    if !(200..300).contains(&response.status) {
        // Couldn't parse error response: the status message stands, exactly
        // like upstream's caught `response.json()`.
        let body_text = read_body_text(response.body).await.unwrap_or_default();
        let body_error = serde_json::from_str::<serde_json::Value>(&body_text)
            .ok()
            .as_ref()
            .and_then(|error_data| error_data.get("error"))
            .and_then(serde_json::Value::as_str)
            .map(|error| format!("Proxy error: {error}"));
        let error_message = body_error.unwrap_or_else(|| {
            format!(
                "Proxy error: {} {}",
                response.status,
                status_reason(response.status)
            )
        });
        return Err(error_message);
    }

    let mut byte_buffer: Vec<u8> = Vec::new();
    loop {
        // The seam races the read against the request's signal, so an abort
        // surfaces as a read error — upstream's reader-cancel abort handler.
        let Some(chunk) = response
            .body
            .next_chunk()
            .await
            .map_err(|error| error.to_string())?
        else {
            break;
        };
        if options
            .signal
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err("Request aborted by user".to_owned());
        }
        // A multi-byte UTF-8 sequence never contains a newline byte, so the
        // line-wise lossy decode cannot split one — the streaming
        // `TextDecoder`'s guarantee restated.
        byte_buffer.extend_from_slice(&chunk);
        while let Some(index) = byte_buffer.iter().position(|byte| *byte == b'\n') {
            let line_bytes = byte_buffer.drain(..=index).collect::<Vec<u8>>();
            let line = String::from_utf8_lossy(&line_bytes[..line_bytes.len() - 1]);
            process_line(&line, accumulator, saw_terminal_event, stream)?;
        }
    }
    if options
        .signal
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
    {
        return Err("Request aborted by user".to_owned());
    }
    // The final event may not be newline-terminated; process whatever is
    // left in the buffer.
    if !byte_buffer.is_empty() {
        let line = String::from_utf8_lossy(&byte_buffer);
        process_line(&line, accumulator, saw_terminal_event, stream)?;
    }
    Ok(())
}

/// One SSE-style `data: ` line of the proxy response, upstream's
/// `processLine`: parse the JSON payload, translate it through the
/// accumulator, push the event. Lines without the prefix and empty payloads
/// are skipped.
///
/// # Errors
/// The payload's JSON parse failure or a proxy event updating the wrong
/// block kind, each the error message the failure event carries.
fn process_line(
    line: &str,
    accumulator: &mut ProxyAccumulator,
    saw_terminal_event: &mut bool,
    stream: &AssistantMessageEventStream,
) -> Result<(), String> {
    let Some(data) = line.strip_prefix("data: ") else {
        return Ok(());
    };
    let data = data.trim();
    if data.is_empty() {
        return Ok(());
    }
    let proxy_event: ProxyAssistantMessageEvent =
        serde_json::from_str(data).map_err(|error| error.to_string())?;
    if let Some(event) = accumulator.process_proxy_event(proxy_event)? {
        if matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        ) {
            *saw_terminal_event = true;
        }
        stream.push(event);
    }
    Ok(())
}

/// The partial-message builder the proxy events update, upstream's `partial`
/// plus the ad-hoc `partialJson` the streamed tool-call arguments ride.
struct ProxyAccumulator {
    /// The live response-so-far, upstream's `partial` initialized to the
    /// pending assistant message.
    partial: AssistantMessage,
    /// The accumulated tool-call argument JSON per content index, upstream's
    /// `partialJson` field on the content object.
    partial_json: BTreeMap<u64, String>,
}

impl ProxyAccumulator {
    /// The pending assistant message a stream starts from, upstream's
    /// `partial` literal: pi-ai's fresh-accumulator constructor — zero
    /// usage, no content, timestamp now — the same literal the wire-API
    /// adapters start their streams from.
    fn new(model: &Model) -> Self {
        Self {
            partial: pi_ai::api::wire_common::initial_output(model),
            partial_json: BTreeMap::new(),
        }
    }

    /// Translate one proxy wire event into the assistant-message event it
    /// reports, updating the partial on the way, upstream's
    /// `processProxyEvent`. `None` drops the event: the `toolcall_end` for a
    /// non-tool-call slot (upstream's silent `return undefined`).
    ///
    /// # Errors
    /// A delta or end event naming the wrong block kind — the messages are
    /// upstream verbatim.
    fn process_proxy_event(
        &mut self,
        proxy_event: ProxyAssistantMessageEvent,
    ) -> Result<Option<AssistantMessageEvent>, String> {
        match proxy_event {
            ProxyAssistantMessageEvent::Start => Ok(Some(self.start_event())),
            ProxyAssistantMessageEvent::TextStart { content_index } => {
                Ok(Some(self.text_start(content_index)))
            }
            ProxyAssistantMessageEvent::TextDelta {
                content_index,
                delta,
            } => Ok(Some(self.text_delta(content_index, &delta)?)),
            ProxyAssistantMessageEvent::TextEnd {
                content_index,
                content_signature,
            } => Ok(Some(self.text_end(content_index, content_signature)?)),
            ProxyAssistantMessageEvent::ThinkingStart { content_index } => {
                Ok(Some(self.thinking_start(content_index)))
            }
            ProxyAssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
            } => Ok(Some(self.thinking_delta(content_index, &delta)?)),
            ProxyAssistantMessageEvent::ThinkingEnd {
                content_index,
                content_signature,
            } => Ok(Some(self.thinking_end(content_index, content_signature)?)),
            ProxyAssistantMessageEvent::ToolcallStart {
                content_index,
                id,
                tool_name,
            } => Ok(Some(self.toolcall_start(content_index, id, tool_name))),
            ProxyAssistantMessageEvent::ToolcallDelta {
                content_index,
                delta,
            } => Ok(Some(self.toolcall_delta(content_index, &delta)?)),
            ProxyAssistantMessageEvent::ToolcallEnd {
                content_index,
                tool_call,
            } => Ok(self.toolcall_end(content_index, tool_call)),
            ProxyAssistantMessageEvent::Done {
                reason,
                usage,
                provider_thinking_level,
            } => Ok(Some(self.done_event(
                reason,
                usage,
                provider_thinking_level,
            ))),
            ProxyAssistantMessageEvent::Error {
                reason,
                error_message,
                usage,
                provider_thinking_level,
            } => Ok(Some(self.error_event(
                reason,
                error_message,
                usage,
                provider_thinking_level,
            ))),
        }
    }

    /// The `start` arm: report the pending accumulator.
    fn start_event(&self) -> AssistantMessageEvent {
        AssistantMessageEvent::Start {
            partial: self.partial.clone(),
        }
    }

    /// Open a block at the index and report its start event, the shape the
    /// three `*_start` arms share — the variant constructor passed as a
    /// closure, the event stamped with the updated accumulator.
    fn open_block(
        &mut self,
        content_index: u64,
        block: AssistantBlock,
        event: impl FnOnce(u64, AssistantMessage) -> AssistantMessageEvent,
    ) -> AssistantMessageEvent {
        set_content(&mut self.partial.content, content_index, block);
        event(content_index, self.partial.clone())
    }

    /// The `text_start` arm: open an empty text block at the index.
    fn text_start(&mut self, content_index: u64) -> AssistantMessageEvent {
        self.open_block(
            content_index,
            AssistantBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            }),
            |content_index, partial| AssistantMessageEvent::TextStart {
                content_index,
                partial,
            },
        )
    }

    /// The `text_delta` arm: grow the text block, upstream's `content.text
    /// += delta`.
    ///
    /// # Errors
    /// The slot does not hold a text block, upstream's throw verbatim.
    fn text_delta(
        &mut self,
        content_index: u64,
        delta: &str,
    ) -> Result<AssistantMessageEvent, String> {
        match block_mut(&mut self.partial.content, content_index) {
            Some(AssistantBlock::Text(text)) => {
                text.text += delta;
                Ok(AssistantMessageEvent::TextDelta {
                    content_index,
                    delta: delta.to_owned(),
                    partial: self.partial.clone(),
                })
            }
            _ => Err("Received text_delta for non-text content".to_owned()),
        }
    }

    /// The `text_end` arm: stamp the text block's signature and report the
    /// authoritative text.
    ///
    /// # Errors
    /// The slot does not hold a text block, upstream's throw verbatim.
    fn text_end(
        &mut self,
        content_index: u64,
        content_signature: Option<String>,
    ) -> Result<AssistantMessageEvent, String> {
        match block_mut(&mut self.partial.content, content_index) {
            Some(AssistantBlock::Text(text)) => {
                text.text_signature = content_signature;
                let content = text.text.clone();
                Ok(AssistantMessageEvent::TextEnd {
                    content_index,
                    content,
                    partial: self.partial.clone(),
                })
            }
            _ => Err("Received text_end for non-text content".to_owned()),
        }
    }

    /// The `thinking_start` arm: open an empty thinking block at the index.
    fn thinking_start(&mut self, content_index: u64) -> AssistantMessageEvent {
        self.open_block(
            content_index,
            AssistantBlock::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: None,
                redacted: None,
            }),
            |content_index, partial| AssistantMessageEvent::ThinkingStart {
                content_index,
                partial,
            },
        )
    }

    /// The `thinking_delta` arm: grow the thinking block.
    ///
    /// # Errors
    /// The slot does not hold a thinking block, upstream's throw verbatim.
    fn thinking_delta(
        &mut self,
        content_index: u64,
        delta: &str,
    ) -> Result<AssistantMessageEvent, String> {
        match block_mut(&mut self.partial.content, content_index) {
            Some(AssistantBlock::Thinking(thinking)) => {
                thinking.thinking += delta;
                Ok(AssistantMessageEvent::ThinkingDelta {
                    content_index,
                    delta: delta.to_owned(),
                    partial: self.partial.clone(),
                })
            }
            _ => Err("Received thinking_delta for non-thinking content".to_owned()),
        }
    }

    /// The `thinking_end` arm: stamp the thinking block's signature and
    /// report the authoritative thinking text.
    ///
    /// # Errors
    /// The slot does not hold a thinking block, upstream's throw verbatim.
    fn thinking_end(
        &mut self,
        content_index: u64,
        content_signature: Option<String>,
    ) -> Result<AssistantMessageEvent, String> {
        match block_mut(&mut self.partial.content, content_index) {
            Some(AssistantBlock::Thinking(thinking)) => {
                thinking.thinking_signature = content_signature;
                let content = thinking.thinking.clone();
                Ok(AssistantMessageEvent::ThinkingEnd {
                    content_index,
                    content,
                    partial: self.partial.clone(),
                })
            }
            _ => Err("Received thinking_end for non-thinking content".to_owned()),
        }
    }

    /// The `toolcall_start` arm: open a tool call with empty arguments and
    /// start the streamed-arguments scratch, upstream's `partialJson: ""`.
    fn toolcall_start(
        &mut self,
        content_index: u64,
        id: String,
        tool_name: String,
    ) -> AssistantMessageEvent {
        let opened = self.open_block(
            content_index,
            AssistantBlock::ToolCall(ToolCall {
                id,
                name: tool_name,
                arguments: serde_json::Map::new(),
                thought_signature: None,
                namespace: None,
            }),
            |content_index, partial| AssistantMessageEvent::ToolcallStart {
                content_index,
                partial,
            },
        );
        self.partial_json.insert(content_index, String::new());
        opened
    }

    /// The `toolcall_delta` arm: grow the streamed argument JSON and
    /// re-parse it into the call's arguments, upstream's `parseStreamingJson
    /// (partialJson) || {}` — the typed arguments field cannot hold a
    /// non-object, so the fold applies to every non-object parse.
    ///
    /// # Errors
    /// The slot does not hold a tool call, upstream's throw verbatim.
    fn toolcall_delta(
        &mut self,
        content_index: u64,
        delta: &str,
    ) -> Result<AssistantMessageEvent, String> {
        match block_mut(&mut self.partial.content, content_index) {
            Some(AssistantBlock::ToolCall(call)) => {
                let json = self.partial_json.entry(content_index).or_default();
                json.push_str(delta);
                call.arguments = match parse_streaming_json(Some(json.as_str())) {
                    serde_json::Value::Object(map) => map,
                    _ => serde_json::Map::new(),
                };
                Ok(AssistantMessageEvent::ToolcallDelta {
                    content_index,
                    delta: delta.to_owned(),
                    partial: self.partial.clone(),
                })
            }
            _ => Err("Received toolcall_delta for non-toolCall content".to_owned()),
        }
    }

    /// The `toolcall_end` arm: the wire tool call replaces the accumulated
    /// one whole, upstream's `Object.assign` plus the `partialJson` delete —
    /// and the side map entry goes with it. `None` when the slot does not
    /// hold a tool call, upstream's silent `return undefined`.
    fn toolcall_end(
        &mut self,
        content_index: u64,
        tool_call: ToolCall,
    ) -> Option<AssistantMessageEvent> {
        match block_mut(&mut self.partial.content, content_index) {
            Some(AssistantBlock::ToolCall(call)) => {
                *call = tool_call.clone();
                self.partial_json.remove(&content_index);
                Some(AssistantMessageEvent::ToolcallEnd {
                    content_index,
                    tool_call,
                    partial: self.partial.clone(),
                })
            }
            _ => None,
        }
    }

    /// The `done` arm: stamp the terminal stop reason, usage, and the
    /// provider thinking level when the server sends one, and settle.
    fn done_event(
        &mut self,
        reason: StopReason,
        usage: Usage,
        provider_thinking_level: Option<String>,
    ) -> AssistantMessageEvent {
        self.partial.stop_reason = reason;
        self.partial.usage = usage;
        self.partial.provider_thinking_level = provider_thinking_level;
        AssistantMessageEvent::Done {
            reason,
            message: self.partial.clone(),
        }
    }

    /// The `error` arm: stamp the failure reason, message, usage, and the
    /// provider thinking level when the server sends one, and settle.
    fn error_event(
        &mut self,
        reason: StopReason,
        error_message: Option<String>,
        usage: Usage,
        provider_thinking_level: Option<String>,
    ) -> AssistantMessageEvent {
        self.partial.stop_reason = reason;
        self.partial.error_message = error_message;
        self.partial.usage = usage;
        self.partial.provider_thinking_level = provider_thinking_level;
        AssistantMessageEvent::Error {
            reason,
            error: self.partial.clone(),
        }
    }
}

/// The block slot a wire `contentIndex` addresses, `None` past the end —
/// upstream's `content?.type` guard.
#[expect(
    clippy::cast_possible_truncation,
    reason = "content indexes address a vec slot the way the JS array index does; conformant streams never exceed the vector length"
)]
fn block_mut(content: &mut [AssistantBlock], content_index: u64) -> Option<&mut AssistantBlock> {
    content.get_mut(content_index as usize)
}

/// Assign the block at `content_index`, filling a past-the-end gap with
/// empty text blocks — upstream's sparse array assignment leaves holes a
/// Rust vector cannot represent.
fn set_content(content: &mut Vec<AssistantBlock>, content_index: u64, block: AssistantBlock) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "content indexes address a vec slot the way the JS array index does; conformant streams never exceed the vector length"
    )]
    let index = content_index as usize;
    if index >= content.len() {
        content.resize_with(index + 1, || {
            AssistantBlock::Text(TextContent {
                text: String::new(),
                text_signature: None,
            })
        });
    }
    content[index] = block;
}

/// The proxy event types.
///
/// The compact wire the server sends with the partial field stripped,
/// upstream's `ProxyAssistantMessageEvent`. The tags and field names are the
/// wire's verbatim (`text_start`, `contentIndex`, ...).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProxyAssistantMessageEvent {
    /// The stream opened.
    #[serde(rename = "start")]
    Start,
    /// A text block opened.
    #[serde(rename = "text_start", rename_all = "camelCase")]
    TextStart {
        /// The block's index in `content`.
        content_index: u64,
    },
    /// Text grew.
    #[serde(rename = "text_delta", rename_all = "camelCase")]
    TextDelta {
        /// The block's index in `content`.
        content_index: u64,
        /// The appended text.
        delta: String,
    },
    /// A text block closed authoritatively.
    #[serde(rename = "text_end", rename_all = "camelCase")]
    TextEnd {
        /// The block's index in `content`.
        content_index: u64,
        /// The provider's text signature, when it sends one.
        content_signature: Option<String>,
    },
    /// A thinking block opened.
    #[serde(rename = "thinking_start", rename_all = "camelCase")]
    ThinkingStart {
        /// The block's index in `content`.
        content_index: u64,
    },
    /// Thinking grew.
    #[serde(rename = "thinking_delta", rename_all = "camelCase")]
    ThinkingDelta {
        /// The block's index in `content`.
        content_index: u64,
        /// The appended thinking text.
        delta: String,
    },
    /// A thinking block closed authoritatively.
    #[serde(rename = "thinking_end", rename_all = "camelCase")]
    ThinkingEnd {
        /// The block's index in `content`.
        content_index: u64,
        /// The provider's thinking signature, when it sends one.
        content_signature: Option<String>,
    },
    /// A tool call opened.
    #[serde(rename = "toolcall_start", rename_all = "camelCase")]
    ToolcallStart {
        /// The block's index in `content`.
        content_index: u64,
        /// The tool call id.
        id: String,
        /// The tool name.
        tool_name: String,
    },
    /// Tool-call arguments grew.
    #[serde(rename = "toolcall_delta", rename_all = "camelCase")]
    ToolcallDelta {
        /// The block's index in `content`.
        content_index: u64,
        /// The appended JSON update.
        delta: String,
    },
    /// A tool call closed authoritatively; the wire call replaces the
    /// accumulated one whole.
    #[serde(rename = "toolcall_end", rename_all = "camelCase")]
    ToolcallEnd {
        /// The block's index in `content`.
        content_index: u64,
        /// The final tool call.
        tool_call: ToolCall,
    },
    /// The stream finished successfully.
    #[serde(rename = "done", rename_all = "camelCase")]
    Done {
        /// Why the model stopped, upstream's `Extract<StopReason, "stop" |
        /// "length" | "toolUse">` — the runtime value flows through
        /// verbatim.
        reason: StopReason,
        /// The provider-reported usage.
        usage: Usage,
        /// The provider's own thinking level, when it reports one.
        provider_thinking_level: Option<String>,
    },
    /// The stream failed or was aborted.
    #[serde(rename = "error", rename_all = "camelCase")]
    Error {
        /// The failure kind, upstream's `Extract<StopReason, "aborted" |
        /// "error">`.
        reason: StopReason,
        /// The failure message, when the server carries one.
        error_message: Option<String>,
        /// The provider-reported usage.
        usage: Usage,
        /// The provider's own thinking level, when it reports one.
        provider_thinking_level: Option<String>,
    },
}

/// The request options the proxy body carries, upstream's
/// `ProxySerializableStreamOptions` — the `SimpleStreamOptions` fields that
/// survive `JSON.stringify`. Absent fields drop from the body the way
/// `undefined` properties do.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProxySerializableStreamOptions {
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    /// Arbitrary sampling parameters merged into the request body as-is.
    #[serde(skip_serializing_if = "Option::is_none")]
    sampling_params: Option<BTreeMap<String, serde_json::Value>>,
    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    /// The pi thinking level for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ThinkingLevel>,
    /// Prompt cache retention preference.
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_retention: Option<CacheRetention>,
    /// Session identifier for providers that support session-based caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    /// Custom HTTP headers merged with provider defaults.
    #[serde(skip_serializing_if = "Option::is_none")]
    headers: Option<ProviderHeaders>,
    /// Optional metadata to include in API requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<BTreeMap<String, serde_json::Value>>,
    /// Preferred transport for providers that support multiple transports.
    #[serde(skip_serializing_if = "Option::is_none")]
    transport: Option<Transport>,
    /// Custom token budgets for thinking levels, token-based providers only.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budgets: Option<ThinkingBudgets>,
    /// Maximum delay in milliseconds to wait for a retry.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_retry_delay_ms: Option<u64>,
}

/// Options for [`stream_proxy`], upstream's `ProxyStreamOptions extends
/// ProxySerializableStreamOptions`: the serializable fields ride in the
/// request body, the rest stay client-side.
///
/// The port adds [`ProxyStreamOptions::http_client`], the `HttpClient` seam
/// upstream's ambient `globalThis.fetch` reads — the field the seam's mock
/// harness mounts on for tests.
#[derive(Clone, Debug, Default)]
pub struct ProxyStreamOptions {
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Arbitrary sampling parameters merged into the request body as-is.
    pub sampling_params: Option<BTreeMap<String, serde_json::Value>>,
    /// Maximum output tokens.
    pub max_tokens: Option<u64>,
    /// The pi thinking level for this request.
    pub reasoning: Option<ThinkingLevel>,
    /// Prompt cache retention preference. Default: `"short"`.
    pub cache_retention: Option<CacheRetention>,
    /// Session identifier for providers that support session-based caching.
    pub session_id: Option<String>,
    /// Custom HTTP headers merged with provider defaults.
    pub headers: Option<ProviderHeaders>,
    /// Optional metadata to include in API requests.
    pub metadata: Option<BTreeMap<String, serde_json::Value>>,
    /// Preferred transport for providers that support multiple transports.
    pub transport: Option<Transport>,
    /// Custom token budgets for thinking levels, token-based providers only.
    pub thinking_budgets: Option<ThinkingBudgets>,
    /// Maximum delay in milliseconds to wait for a retry when the server
    /// requests a long wait.
    pub max_retry_delay_ms: Option<u64>,
    /// Local cancellation for the proxy request, upstream's `signal`.
    pub signal: Option<CancellationToken>,
    /// Auth token for the proxy server.
    pub auth_token: String,
    /// Proxy server URL (e.g., `https://genai.example.com`).
    pub proxy_url: String,
    /// The HTTP client the request sends on, upstream's ambient
    /// `globalThis.fetch`; `None` uses the process default.
    pub http_client: Option<Arc<dyn HttpClient>>,
}

impl ProxyStreamOptions {
    /// Options for one proxy server and token, the required pair upstream's
    /// interface marks; the other fields default.
    pub fn new(auth_token: impl Into<String>, proxy_url: impl Into<String>) -> Self {
        Self {
            auth_token: auth_token.into(),
            proxy_url: proxy_url.into(),
            ..Self::default()
        }
    }

    /// The request body's options, upstream's `buildProxyRequestOptions`:
    /// exactly the serializable pick, field for field.
    fn serializable(&self) -> ProxySerializableStreamOptions {
        ProxySerializableStreamOptions {
            temperature: self.temperature,
            sampling_params: self.sampling_params.clone(),
            max_tokens: self.max_tokens,
            reasoning: self.reasoning,
            cache_retention: self.cache_retention,
            session_id: self.session_id.clone(),
            headers: self.headers.clone(),
            metadata: self.metadata.clone(),
            transport: self.transport,
            thinking_budgets: self.thinking_budgets,
            max_retry_delay_ms: self.max_retry_delay_ms,
        }
    }

    /// The request's cancellation token: the caller's, or a fresh
    /// uncancelled one, pi-ai's `operation_signal`.
    fn operation_signal(&self) -> CancellationToken {
        pi_ai::utils::abort::operation_signal(self.signal.as_ref())
    }

    /// The client the request sends on: the injected one, else the process
    /// default — upstream's ambient fetch.
    fn client(&self) -> Arc<dyn HttpClient> {
        self.http_client
            .clone()
            .unwrap_or_else(pi_ai::http::default_http_client)
    }
}

//! Server-sent event framing, ported from the canonical SSE decoder in
//! `packages/ai/src/api/anthropic-messages.ts` (pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`).
//!
//! Porting restatement: upstream repeats byte-stream + SSE framing loops in
//! every raw-fetch API module (`anthropic-messages.ts` carries the full
//! decoder; `pi-messages.ts`, `mistral-conversations.ts`, and
//! `openai-codex-responses.ts` carry data-line-only variants) — this module
//! is the one shared decoder those copies collapse into; the per-API event
//! parsing rides with each wire-API child. The redundant per-read
//! `signal.aborted` checks the upstream iterators run disappear: aborts race
//! the seam's byte stream itself and surface as [`HttpError::Aborted`].

use crate::http::client::{HttpByteStream, HttpError};

/// One framed server-sent event, upstream's `ServerSentEvent`
/// (`anthropic-messages.ts:310`).
///
/// `event` is the event field name, `data` is the payload with multi-line
/// values joined by newlines, and `raw` holds the raw source lines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerSentEvent {
    /// The `event` field value when the wire sent one.
    pub event: Option<String>,
    /// The `data` field lines joined with `\n`.
    pub data: String,
    /// The event's raw source lines, comments included.
    pub raw: Vec<String>,
}

/// One decoded SSE line's effect: either nothing, or a completed event.
enum LineEvent {
    None,
    Completed(ServerSentEvent),
}

/// The incremental SSE decoder, upstream's `SseDecoderState` plus the line
/// splitting that reads `Response.body` chunk by chunk.
///
/// Feed arbitrary byte chunks — chunk boundaries may cut lines, `\r\n` pairs,
/// or UTF-8 code points; complete lines decode as UTF-8 with Node
/// `TextDecoder`'s default leniency (invalid sequences become U+FFFD), and a
/// truncated tail line only surfaces at [`SseDecoder::finish`].
#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
    state: SseState,
}

/// The event-under-construction state, upstream's `SseDecoderState`.
#[derive(Debug, Default)]
struct SseState {
    event: Option<String>,
    data: Vec<String>,
    raw: Vec<String>,
}

impl SseState {
    /// Complete the event under construction, upstream's `flushSseEvent`.
    fn flush(&mut self) -> Option<ServerSentEvent> {
        if self.event.is_none() && self.data.is_empty() {
            return None;
        }
        let event = ServerSentEvent {
            event: self.event.take(),
            data: std::mem::take(&mut self.data).join("\n"),
            raw: std::mem::take(&mut self.raw),
        };
        Some(event)
    }

    /// Decode one line, upstream's `decodeSseLine`; an empty line flushes the
    /// event under construction, a comment (`:...`) is ignored, and field
    /// values drop one leading space after the colon per the wire grammar.
    fn decode_line(&mut self, line: &str) -> LineEvent {
        if line.is_empty() {
            return self.flush().map_or(LineEvent::None, LineEvent::Completed);
        }

        self.raw.push(line.to_owned());
        if line.starts_with(':') {
            return LineEvent::None;
        }

        let (field_name, value) = line.find(':').map_or_else(
            || (line.to_owned(), ""),
            |index| {
                let value = &line[index + 1..];
                (
                    line[..index].to_owned(),
                    value.strip_prefix(' ').unwrap_or(value),
                )
            },
        );

        if field_name == "event" {
            self.event = Some(value.to_owned());
        } else if field_name == "data" {
            self.data.push(value.to_owned());
        }
        LineEvent::None
    }
}

/// The next line break, preferring the earlier of `\r` and `\n`, upstream's
/// `nextLineBreakIndex`.
fn next_break_index(buffer: &[u8]) -> Option<usize> {
    let carriage = buffer.iter().position(|byte| *byte == b'\r');
    let newline = buffer.iter().position(|byte| *byte == b'\n');
    match (carriage, newline) {
        (Some(c), Some(n)) => Some(c.min(n)),
        (Some(c), None) => Some(c),
        (None, Some(n)) => Some(n),
        (None, None) => None,
    }
}

impl SseDecoder {
    /// Feed one response chunk and drain every line it completes.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<ServerSentEvent> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(index) = next_break_index(&self.buffer) {
            let mut line_end = index + 1;
            if self.buffer[index] == b'\r'
                && line_end < self.buffer.len()
                && self.buffer[line_end] == b'\n'
            {
                line_end += 1;
            }
            let line = String::from_utf8_lossy(&self.buffer[..index]).into_owned();
            self.buffer.drain(..line_end);
            if let LineEvent::Completed(event) = self.state.decode_line(&line) {
                events.push(event);
            }
        }
        events
    }

    /// Flush at end of stream, upstream's post-loop drain: the residual
    /// partial line decodes into the state, then the trailing event flushes.
    ///
    /// The partial line carries no line terminator, so it can never be the
    /// empty line that completes an event — it only contributes its fields.
    pub fn finish(&mut self) -> Vec<ServerSentEvent> {
        if !self.buffer.is_empty() {
            let line = String::from_utf8_lossy(&self.buffer).into_owned();
            self.buffer.clear();
            let _ = self.state.decode_line(&line);
        }
        self.state.flush().into_iter().collect()
    }
}

/// A stream of decoded server-sent events over a seam byte stream,
/// upstream's `iterateSseMessages`.
///
/// The redundant per-read `signal.aborted` checks the upstream iterator runs
/// disappear here: aborts race the body stream itself and surface as
/// [`HttpError::Aborted`].
#[derive(Debug)]
pub struct SseStream {
    body: HttpByteStream,
    decoder: SseDecoder,
    events: std::vec::IntoIter<ServerSentEvent>,
    done: bool,
}

impl SseStream {
    /// Decode server-sent events off a response body stream.
    #[must_use]
    pub fn new(body: HttpByteStream) -> Self {
        Self {
            body,
            decoder: SseDecoder::default(),
            events: Vec::new().into_iter(),
            done: false,
        }
    }

    /// Read the next event, `None` after a clean end of stream.
    ///
    /// # Errors
    /// Returns the body stream's error — [`HttpError::Aborted`] when the
    /// request aborts, [`HttpError::Transport`] when the connection drops —
    /// and terminates the stream; framed events read before the failure
    /// were already yielded.
    pub async fn next(&mut self) -> Result<Option<ServerSentEvent>, HttpError> {
        loop {
            if let Some(event) = self.events.next() {
                return Ok(Some(event));
            }
            if self.done {
                return Ok(None);
            }
            match self.body.next_chunk().await {
                Ok(Some(chunk)) => {
                    self.events = self.decoder.feed(&chunk).into_iter();
                }
                Ok(None) => {
                    self.events = self.decoder.finish().into_iter();
                    self.done = true;
                }
                Err(error) => {
                    self.done = true;
                    return Err(error);
                }
            }
        }
    }

    /// Collect the remaining events.
    ///
    /// # Errors
    /// Returns the first body stream error, mirroring upstream's throwing
    /// generator; events before the failure are discarded.
    pub async fn collect(mut self) -> Result<Vec<ServerSentEvent>, HttpError> {
        let mut events = Vec::new();
        while let Some(event) = self.next().await? {
            events.push(event);
        }
        Ok(events)
    }
}

/// Decode all server-sent events off a response body in one await.
///
/// # Errors
/// Returns the body stream's error; see [`SseStream::next`].
pub async fn collect_sse(body: HttpByteStream) -> Result<Vec<ServerSentEvent>, HttpError> {
    SseStream::new(body).collect().await
}

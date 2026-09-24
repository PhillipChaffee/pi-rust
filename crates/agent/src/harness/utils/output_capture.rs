//! Maintains and publishes one bounded shell-output view, ported from
//! upstream `src/harness/utils/output-capture.ts`.
//!
//! Writes received while publication is rate-limited collapse into the
//! latest view (the adaptive publisher's contract). Small changes remain
//! responsive; complete window turnovers purchase a proportionally longer
//! delay. The first update after idle and an explicit final flush are
//! immediate. The streaming UTF-8 decoder buffers incomplete sequences
//! across chunks, matching Node's `TextDecoder({ stream: true })` and
//! emitting the replacement character for invalid bytes like its
//! non-fatal mode.

use std::sync::Arc;
use std::sync::Mutex;


use crate::harness::context::Context;
use crate::harness::types::{
    ShellOutputCaptureOptions, ShellOutputMetadata, ShellOutputRetention,
    ShellOutputUpdate, ShellOutputView,
};
use crate::harness::utils::adaptive_publisher::{AdaptivePublisher, AdaptivePublisherOptions};
use crate::harness::utils::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_head, truncate_tail, utf8_byte_length,
};

/// The floor on publication spacing, in milliseconds, upstream's
/// `OUTPUT_MIN_EMIT_INTERVAL_MS`.
pub const OUTPUT_MIN_EMIT_INTERVAL_MS: u64 = 100;
/// The publication bytes-per-second budget, upstream's
/// `OUTPUT_TARGET_BYTES_PER_SECOND`.
pub const OUTPUT_TARGET_BYTES_PER_SECOND: u64 = 100 * 1024;

/// The control characters and invisible format marks the sanitized view
/// drops, upstream's `INVALID_SHELL_OUTPUT` regex
/// `[\x00-\x08\x0b-\x1f\ufff9-\ufffb]`.
#[must_use]
pub fn is_invalid_shell_output_char(c: char) -> bool {
    matches!(c, '\u{0}'..='\u{8}' | '\u{b}'..='\u{1f}' | '\u{fff9}'..='\u{fffb}')
}

/// The handlers [`OutputCapture`] reports through, upstream's
/// `OutputCaptureHandlers`.
pub struct OutputCaptureHandlers {
    /// Called with bounded view updates; `None` suppresses publication.
    pub on_update: Option<Arc<dyn Fn(&ShellOutputUpdate, &Context) + Send + Sync>>,
    /// Receives publisher and callback failures; the shell exec maps them
    /// onto its `callback_error` outcome.
    pub on_error: Arc<dyn Fn(String) + Send + Sync>,
}

impl std::fmt::Debug for OutputCaptureHandlers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputCaptureHandlers")
            .field("on_update", &self.on_update.is_some())
            .finish_non_exhaustive()
    }
}

struct StreamingDecoder {
    pending: Vec<u8>,
}

impl StreamingDecoder {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Decodes `chunk`, buffering any incomplete trailing sequence. Invalid
    /// sequences emit the replacement character, Node's non-fatal decoding.
    fn decode(&mut self, chunk: &[u8], final_chunk: bool) -> String {
        self.pending.extend_from_slice(chunk);
        let mut out = String::new();
        while !self.pending.is_empty() {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    out.push_str(text);
                    self.pending.clear();
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    out.push_str(&String::from_utf8_lossy(&self.pending[..valid]).into_owned());
                    match error.error_len() {
                        Some(invalid_len) => {
                            out.push('\u{fffd}');
                            self.pending.drain(..valid + invalid_len);
                        }
                        // Incomplete sequence: emit it as a replacement when
                        // the stream ends, buffer it for the next chunk
                        // otherwise.
                        None => {
                            if final_chunk {
                                out.push('\u{fffd}');
                                self.pending.clear();
                            } else {
                                self.pending.drain(..valid);
                                break;
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

struct CaptureBuffer {
    text: String,
    buffer_bytes: u64,
    total_bytes: u64,
    newlines: u64,
    ends_with_newline: bool,
    current_line_bytes: u64,
    disposed: bool,
}

struct CaptureShared {
    decoder: Mutex<StreamingDecoder>,
    buffer: Mutex<CaptureBuffer>,
    spill_path: Mutex<Option<String>>,
    max_bytes: u64,
    max_lines: u64,
    retain: ShellOutputRetention,
}

/// Maintains and publishes one bounded shell-output view, upstream's
/// `OutputCapture`.
pub struct OutputCapture {
    shared: Arc<CaptureShared>,
    publisher: AdaptivePublisher<ShellOutputView, ShellOutputUpdate>,
}

impl std::fmt::Debug for OutputCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputCapture").finish_non_exhaustive()
    }
}

impl OutputCapture {
    /// Builds a capture over the requested limits.
    ///
    /// # Errors
    /// Returns a `TypeError`-shaped message when `maxBytes` is not a
    /// positive finite number or `maxLines` is not a positive integer,
    /// upstream's constructor throw.
    pub fn new(
        options: Option<&ShellOutputCaptureOptions>,
        context: Context,
        handlers: OutputCaptureHandlers,
    ) -> Result<Self, String> {
        let max_bytes = options.map_or(DEFAULT_MAX_BYTES, |capture| capture.limits.max_bytes);
        let max_lines = options.map_or(DEFAULT_MAX_LINES, |capture| capture.limits.max_lines);
        let retain = options.map_or(ShellOutputRetention::Tail, |capture| {
            capture.limits.retain.unwrap_or_default()
        });
        if max_bytes == 0 {
            return Err("Output maxBytes must be a positive finite number".to_owned());
        }
        if max_lines == 0 {
            return Err("Output maxLines must be a positive integer".to_owned());
        }
        let shared = Arc::new(CaptureShared {
            decoder: Mutex::new(StreamingDecoder::new()),
            buffer: Mutex::new(CaptureBuffer {
                text: String::new(),
                buffer_bytes: 0,
                total_bytes: 0,
                newlines: 0,
                ends_with_newline: true,
                current_line_bytes: 0,
                disposed: false,
            }),
            spill_path: Mutex::new(None),
            max_bytes,
            max_lines,
            retain,
        });
        let snapshot_shared = Arc::clone(&shared);
        let publish_on_update = handlers.on_update.clone();
        let publish_context = context.clone();
        let publisher = AdaptivePublisher::new(AdaptivePublisherOptions {
            snapshot: Arc::new(move || snapshot_of(&snapshot_shared)),
            update: Arc::new(update_from),
            measure: Arc::new(|update: &ShellOutputUpdate| {
                u64::try_from(serde_json::to_string(update).map_or(0, |encoded| encoded.len()))
                    .unwrap_or(u64::MAX)
            }),
            publish: Arc::new(move |update: ShellOutputUpdate| {
                if let Some(on_update) = &publish_on_update {
                    on_update(&update, &publish_context);
                }
            }),
            on_error: handlers.on_error,
            min_interval_ms: Some(OUTPUT_MIN_EMIT_INTERVAL_MS),
            target_bytes_per_second: Some(OUTPUT_TARGET_BYTES_PER_SECOND),
        });
        Ok(Self { shared, publisher })
    }

    /// Whether the captured total crossed either limit, upstream's
    /// `truncated` getter.
    #[must_use]
    pub fn truncated(&self) -> bool {
        let buffer = self.shared.buffer.lock().expect("capture buffer lock");
        buffer.total_bytes > self.shared.max_bytes || total_lines(&buffer) > self.shared.max_lines
    }

    /// Feeds one decoded-text or raw-bytes chunk into the view. A text
    /// chunk first flushes the decoder's pending raw-byte tail, then
    /// appends the text, upstream's string push.
    pub fn push(&mut self, chunk: Chunk<'_>) {
        {
            let buffer = self.shared.buffer.lock().expect("capture buffer lock");
            if buffer.disposed {
                return;
            }
        }
        let text = match chunk {
            Chunk::Text(text) => {
                let flushed = self
                    .shared
                    .decoder
                    .lock()
                    .expect("capture decoder lock")
                    .decode(&[], true);
                if flushed.is_empty() {
                    text.to_owned()
                } else {
                    format!("{flushed}{text}")
                }
            }
            Chunk::Bytes(bytes) => self
                .shared
                .decoder
                .lock()
                .expect("capture decoder lock")
                .decode(bytes, false),
        };
        if !text.is_empty() {
            self.append_text(&text);
        }
    }

    /// Flushes the decoder's pending raw-byte tail into the view.
    pub fn finish(&mut self) {
        {
            let buffer = self.shared.buffer.lock().expect("capture buffer lock");
            if buffer.disposed {
                return;
            }
        }
let pending = self
            .shared
            .decoder
            .lock()
            .expect("capture decoder lock")
            .decode(&[], true);
        if !pending.is_empty() {
            self.append_text(&pending);
        }
    }

    /// Records the spill path in the metadata and republishes immediately.
    pub fn set_spill_path(&mut self, path: &str) {
        {
            let mut spill = self.shared.spill_path.lock().expect("capture spill lock");
            let buffer = self.shared.buffer.lock().expect("capture buffer lock");
            if buffer.disposed || spill.as_deref() == Some(path) {
                return;
            }
            *spill = Some(path.to_owned());
        }
        self.publisher.mark_dirty();
        self.flush();
    }

    /// The complete bounded view, recomputed from the buffer.
    #[must_use]
    pub fn snapshot(&self) -> ShellOutputView {
        snapshot_of(&self.shared)
    }

    /// Publishes the latest view immediately.
    pub fn flush(&mut self) {
        self.publisher.flush(true);
    }

    /// Stops the capture; later pushes and flushes are ignored.
    pub fn dispose(&mut self) {
        self.publisher.dispose();
        self.shared.buffer.lock().expect("capture buffer lock").disposed = true;
    }

    fn append_text(&self, text: &str) {
        if text.is_empty() {
            return;
        }
        let text_bytes = utf8_byte_length(text);
        let mut buffer = self.shared.buffer.lock().expect("capture buffer lock");
        if buffer.disposed {
            return;
        }
        buffer.total_bytes += text_bytes;
        buffer.newlines += u64::try_from(text.matches('\n').count()).unwrap_or(u64::MAX);
        buffer.ends_with_newline = text.ends_with('\n');
        buffer.current_line_bytes = match text.rfind('\n') {
            None => buffer.current_line_bytes + text_bytes,
            Some(last_newline) => utf8_byte_length(&text[last_newline + 1..]),
        };
        buffer.text.push_str(text);
        buffer.buffer_bytes += text_bytes;

        let guard = self.shared.max_bytes.saturating_mul(2);
        if buffer.buffer_bytes > guard.saturating_mul(2) {
            buffer.text = if self.shared.retain == ShellOutputRetention::Tail {
                trim_to_last_utf8_bytes(&buffer.text, guard)
            } else {
                trim_to_first_utf8_bytes(&buffer.text, guard)
            };
            buffer.buffer_bytes = utf8_byte_length(&buffer.text);
        }
        drop(buffer);
        // The publisher's flush re-snapshots the buffer, so the buffer
        // lock must be released before marking dirty.
        self.publisher.mark_dirty();
    }
}

/// The chunk variants [`OutputCapture::push`] accepts, upstream's
/// `string | Uint8Array` parameter.
#[derive(Clone, Debug)]
pub enum Chunk<'a> {
    /// Decoded text appended directly.
    Text(&'a str),
    /// Raw bytes decoded with the streaming decoder.
    Bytes(&'a [u8]),
}

/// Recomputes the bounded view from the shared state; the publisher's
/// snapshot callback and the public [`OutputCapture::snapshot`] share it.
fn snapshot_of(shared: &CaptureShared) -> ShellOutputView {
    let buffer = shared.buffer.lock().expect("capture buffer lock");
    let retained = if shared.retain == ShellOutputRetention::Head {
        truncate_head(
            &buffer.text,
            crate::harness::utils::truncate::TruncationOptions {
                max_bytes: Some(shared.max_bytes),
                max_lines: Some(shared.max_lines),
            },
        )
    } else {
        truncate_tail(
            &buffer.text,
            crate::harness::utils::truncate::TruncationOptions {
                max_bytes: Some(shared.max_bytes),
                max_lines: Some(shared.max_lines),
            },
        )
    };
    let total_lines = total_lines(&buffer);
    let truncated =
        buffer.total_bytes > shared.max_bytes || total_lines > shared.max_lines;
let mut truncation = retained.metadata.clone();
    truncation.truncated = truncated;
    truncation.truncated_by = truncated_by_for(truncated, total_lines, shared.max_lines);
    truncation.total_bytes = buffer.total_bytes;
    truncation.total_lines = total_lines;
    let spill_path = shared.spill_path.lock().expect("capture spill lock").clone();
    let last_line_bytes = if retained.metadata.last_line_partial {
        Some(buffer.current_line_bytes)
    } else {
        None
    };
    ShellOutputView {
        metadata: ShellOutputMetadata {
            truncation,
            spill_path,
            last_line_bytes,
        },
        text: sanitize_shell_output(&retained.content),
    }
}

fn truncated_by_for(
    truncated: bool,
    total_lines: u64,
    max_lines: u64,
) -> Option<crate::harness::types::TruncatedBy> {
    if !truncated {
        return None;
    }
    Some(if total_lines > max_lines {
        crate::harness::types::TruncatedBy::Lines
    } else {
        crate::harness::types::TruncatedBy::Bytes
    })
}

fn total_lines(buffer: &CaptureBuffer) -> u64 {
    buffer.newlines + u64::from(!(buffer.ends_with_newline || buffer.total_bytes == 0))
}

/// The reducer shell-output consumers apply to fold one bounded view into
/// the next, upstream's `applyShellOutputUpdate`.
#[must_use]
pub fn apply_shell_output_update(
    current: Option<&ShellOutputView>,
    update: &ShellOutputUpdate,
) -> ShellOutputView {
    match update {
        ShellOutputUpdate::Replace { output } => output.clone(),
        ShellOutputUpdate::Append { text, metadata } => ShellOutputView {
            text: format!("{}{}", current.map_or(String::new(), |view| view.text.clone()), text),
            metadata: metadata.clone(),
        },
        ShellOutputUpdate::Slide { drop, text, metadata } => {
            let previous = current.map_or(String::new(), |view| view.text.clone());
            let start = (*drop).min(previous.len());
            ShellOutputView {
                text: format!("{}{}", previous.get(start..).unwrap_or_default(), text),
                metadata: metadata.clone(),
            }
        }
        ShellOutputUpdate::Metadata { metadata } => ShellOutputView {
            text: current.map_or(String::new(), |view| view.text.clone()),
            metadata: metadata.clone(),
        },
    }
}

/// Derives the incremental update between two views, upstream's
/// `updateFrom`.
#[must_use]
pub fn update_from(previous: Option<&ShellOutputView>, current: &ShellOutputView) -> Option<ShellOutputUpdate> {
    let Some(previous) = previous else {
        return Some(ShellOutputUpdate::Replace {
            output: current.clone(),
        });
    };
    let metadata = ShellOutputMetadata {
        truncation: current.metadata.truncation.clone(),
        spill_path: current.metadata.spill_path.clone(),
        last_line_bytes: current.metadata.last_line_bytes,
    };
    if current.text == previous.text {
        return Some(ShellOutputUpdate::Metadata { metadata });
    }
    if current.text.len() > previous.text.len()
        && current.text.starts_with(&previous.text)
    {
        return Some(ShellOutputUpdate::Append {
            text: current.text[previous.text.len()..].to_owned(),
            metadata,
        });
    }
    let shared = suffix_prefix_overlap(
        &previous.text,
        &current.text,
        (previous.text.len())
            .min(current.text.len())
            .min(current.metadata.truncation.max_bytes.saturating_mul(2) as usize),
    );
if shared > 0 {
        return Some(ShellOutputUpdate::Slide {
            drop: previous.text.len() - shared,
            text: current.text[shared..].to_owned(),
            metadata,
        });
    }
    Some(ShellOutputUpdate::Replace {
        output: current.clone(),
    })
}

/// The longest suffix of `before` that prefixes `after`, scanned within
/// `scan` bytes, upstream's `suffixPrefixOverlap`.
///
/// The scan runs on raw bytes and the result is rejected (zero) when the
/// overlap would split a UTF-8 character: upstream slices UTF-16 units,
/// which cannot split, and the byte-level port keeps that invariant by
/// falling back to the whole-view replacement.
fn suffix_prefix_overlap(before: &str, after: &str, scan: usize) -> usize {
    if before.is_empty() || after.is_empty() || scan == 0 {
        return 0;
    }
    let before_bytes = before.as_bytes();
    let after_bytes = after.as_bytes();
    let tail = if before_bytes.len() > scan {
        &before_bytes[before_bytes.len() - scan..]
    } else {
        before_bytes
    };
    for probe_length in [after_bytes.len().min(64), 1] {
        let probe = &after_bytes[..probe_length];
        let mut candidates = 0u32;
        let mut index = 0;
        while let Some(found) = find_from(tail, index, probe) {
            candidates += 1;
            if candidates > 8 {
                break;
            }
            let overlap_length = tail.len() - found;
            if overlap_length <= after_bytes.len()
                && &after_bytes[..overlap_length] == &tail[found..]
                && after.is_char_boundary(overlap_length)
                && before.is_char_boundary(before_bytes.len() - overlap_length)
            {
                return overlap_length;
            }
            index = found + 1;
        }
        if probe_length == 1 {
            break;
        }
    }
    0
}

/// The first byte offset at or after `from` where `needle` occurs in
/// `haystack`.
fn find_from(haystack: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || from >= haystack.len() || needle.len() > haystack.len() - from {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|position| position + from)
}

/// Drops control characters and invisible format marks, upstream's
/// `sanitizeShellOutput`.
#[must_use]
pub fn sanitize_shell_output(text: &str) -> String {
    text.chars().filter(|c| !is_invalid_shell_output_char(*c)).collect()
}

fn trim_to_last_utf8_bytes(text: &str, max_bytes: u64) -> String {
    let limit = usize::try_from(max_bytes).unwrap_or(text.len());
    if text.len() <= limit {
        return text.to_owned();
    }
    let mut start = text.len() - limit;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_owned()
}

fn trim_to_first_utf8_bytes(text: &str, max_bytes: u64) -> String {
    let limit = usize::try_from(max_bytes).unwrap_or(text.len());
    if text.len() <= limit {
        return text.to_owned();
    }
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}

#[cfg(test)]
mod tests;
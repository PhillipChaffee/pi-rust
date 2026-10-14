//! Incrementally tracks streaming tool output with bounded memory, ported
//! from upstream `src/core/tools/output-accumulator.ts`.
//!
//! Appends decode chunks with a streaming UTF-8 decoder, keeps only a
//! decoded tail for display snapshots, and opens a temp file when the full
//! output needs to be preserved. The temp file receives the raw bytes (the
//! decoded tail is only the display view); the finish flush completes the
//! decoder the way upstream's argument-less `TextDecoder.decode()` does.

use std::future::Future;
use std::io::Write as _;
use std::sync::Mutex;

use super::truncate::{TruncatedBy, TruncationResult, truncate_tail};

/// The accumulator limits, upstream's `OutputAccumulatorOptions`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct OutputAccumulatorOptions {
    /// Maximum number of lines (default 2000).
    pub max_lines: Option<usize>,
    /// Maximum number of bytes (default 50KB).
    pub max_bytes: Option<usize>,
    /// The temp file name prefix (default `pi-output`).
    pub temp_file_prefix: Option<String>,
}

/// The display snapshot, upstream's `OutputSnapshot`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputSnapshot {
    /// The (possibly truncated) display content.
    pub content: String,
    /// The truncation metadata over the full stream.
    pub truncation: TruncationResult,
    /// The temp file preserving the full output, when one was opened.
    pub full_output_path: Option<String>,
}

/// The streaming UTF-8 decoder, upstream's `new TextDecoder()` driven with
/// `{ stream: true }`: bytes of an incomplete sequence at a chunk boundary
/// hold until the sequence completes, and ill-formed bytes surface one
/// replacement character per maximal invalid subpart.
#[derive(Default)]
pub(crate) struct StreamingDecoder {
    pending: Vec<u8>,
}

impl StreamingDecoder {
    /// Decode one chunk, carrying any incomplete sequence's bytes forward.
    pub(crate) fn decode(&mut self, chunk: &[u8]) -> String {
        self.pending.extend_from_slice(chunk);
        let mut text = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    text.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    if valid_up_to > 0
                        && let Some(valid) = std::str::from_utf8(&self.pending[..valid_up_to]).ok()
                    {
                        text.push_str(valid);
                    }
                    if let Some(invalid_len) = error.error_len() {
                        // One replacement per maximal invalid subpart,
                        // upstream's TextDecoder.
                        let invalid = &self.pending[valid_up_to..valid_up_to + invalid_len];
                        text.push_str(&String::from_utf8_lossy(invalid));
                        self.pending.drain(..valid_up_to + invalid_len);
                    } else {
                        // Incomplete tail: hold for the next chunk.
                        self.pending.drain(..valid_up_to);
                        break;
                    }
                }
            }
        }
        text
    }

    /// Flush at end of stream, upstream's `decoder.decode()` with no
    /// arguments: an incomplete trailing sequence is ill-formed and surfaces
    /// its replacement character.
    pub(crate) fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        text
    }
}

/// The accumulator's mutable state, guarded as one unit the same way the
/// upstream instance carries it.
struct AccumulatorState {
    decoder: StreamingDecoder,
    raw_chunks: Vec<Vec<u8>>,
    tail_text: String,
    tail_bytes: usize,
    tail_starts_at_line_boundary: bool,
    total_raw_bytes: usize,
    total_decoded_bytes: usize,
    completed_lines: usize,
    total_lines: usize,
    current_line_bytes: usize,
    has_open_line: bool,
    finished: bool,
    temp_file_path: Option<std::path::PathBuf>,
    temp_file: Option<std::fs::File>,
}

/// Incrementally tracks streaming output with bounded memory, upstream's
/// `OutputAccumulator`.
pub struct OutputAccumulator {
    max_lines: usize,
    max_bytes: usize,
    max_rolling_bytes: usize,
    temp_file_prefix: String,
    state: Mutex<AccumulatorState>,
}

impl OutputAccumulator {
    /// Build an accumulator, upstream's constructor.
    #[must_use]
    pub fn new(options: OutputAccumulatorOptions) -> Self {
        let max_lines = options
            .max_lines
            .unwrap_or(super::truncate::DEFAULT_MAX_LINES);
        let max_bytes = options
            .max_bytes
            .unwrap_or(super::truncate::DEFAULT_MAX_BYTES);
        Self {
            max_lines,
            max_bytes,
            max_rolling_bytes: std::cmp::max(max_bytes * 2, 1),
            temp_file_prefix: options
                .temp_file_prefix
                .unwrap_or_else(|| "pi-output".to_owned()),
            state: Mutex::new(AccumulatorState {
                decoder: StreamingDecoder::default(),
                raw_chunks: Vec::new(),
                tail_text: String::new(),
                tail_bytes: 0,
                tail_starts_at_line_boundary: true,
                total_raw_bytes: 0,
                total_decoded_bytes: 0,
                completed_lines: 0,
                total_lines: 0,
                current_line_bytes: 0,
                has_open_line: false,
                finished: false,
                temp_file_path: None,
                temp_file: None,
            }),
        }
    }

    /// The default temp file path, upstream's `defaultTempFilePath`: a
    /// process-unique hex id disambiguates concurrent accumulators.
    fn default_temp_file_path(prefix: &str) -> std::path::PathBuf {
        let id = super::random_bytes_hex(8);
        std::env::temp_dir().join(format!("{prefix}-{id}.log"))
    }

    /// Append one raw output chunk, upstream's `append`.
    ///
    /// # Panics
    /// When the accumulator already finished, upstream's throw.
    pub fn append(&self, data: &[u8]) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            !state.finished,
            "Cannot append to a finished output accumulator"
        );

        state.total_raw_bytes += data.len();
        let decoded = state.decoder.decode(data);
        self.append_decoded_text(&mut state, &decoded);

        if state.temp_file.is_some() || self.should_use_temp_file(&state) {
            self.ensure_temp_file(&mut state);
            if let Some(temp_file) = state.temp_file.as_mut() {
                let _written = temp_file.write_all(data);
            }
        } else if !data.is_empty() {
            state.raw_chunks.push(data.to_vec());
        }
    }

    /// Complete the stream, upstream's `finish`: flush the decoder and open
    /// the temp file when the full output must be preserved.
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the finished flag, the decoder flush, and the temp-file open are one finish transaction; splitting them would let a late append interleave"
    )]
    pub fn finish(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.finished {
            return;
        }
        state.finished = true;
        let flushed = state.decoder.flush();
        self.append_decoded_text(&mut state, &flushed);
        if self.should_use_temp_file(&state) {
            self.ensure_temp_file(&mut state);
        }
    }

    /// The display snapshot, upstream's `snapshot`.
    ///
    /// With `persist_if_truncated`, a truncated snapshot opens the temp file
    /// first so the full output is preserved before callers read the path.
    pub fn snapshot(&self, persist_if_truncated: bool) -> OutputSnapshot {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tail_truncation = truncate_tail(
            Self::get_snapshot_text(&state),
            super::truncate::TruncationOptions {
                max_lines: Some(self.max_lines),
                max_bytes: Some(self.max_bytes),
            },
        );
        let truncated =
            state.total_lines > self.max_lines || state.total_decoded_bytes > self.max_bytes;
        let truncated_by = if truncated {
            tail_truncation.truncated_by.or_else(|| {
                if state.total_decoded_bytes > self.max_bytes {
                    Some(TruncatedBy::Bytes)
                } else {
                    Some(TruncatedBy::Lines)
                }
            })
        } else {
            None
        };
        let truncation = TruncationResult {
            truncated,
            truncated_by,
            total_lines: state.total_lines,
            total_bytes: state.total_decoded_bytes,
            max_lines: self.max_lines,
            max_bytes: self.max_bytes,
            ..tail_truncation
        };

        if persist_if_truncated && truncation.truncated {
            self.ensure_temp_file(&mut state);
        }

        OutputSnapshot {
            content: truncation.content.clone(),
            truncation,
            full_output_path: state
                .temp_file_path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
        }
    }

    /// Close the temp file, flushing its writes, upstream's
    /// `closeTempFile` (which awaits the stream's `finish` event).
    ///
    /// # Errors
    /// The temp file's flush failure, upstream's rejected promise.
    pub fn close_temp_file(&self) -> impl Future<Output = Result<(), std::io::Error>> {
        let file = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.temp_file.take()
        };
        async move {
            let Some(mut file) = file else {
                return Ok(());
            };
            file.flush()?;
            file.sync_all()
        }
    }

    /// The size in bytes of the line still being streamed, upstream's
    /// `getLastLineBytes`.
    #[must_use]
    pub fn get_last_line_bytes(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .current_line_bytes
    }

    /// Fold decoded text into the counters and the rolling tail, upstream's
    /// `appendDecodedText`.
    fn append_decoded_text(&self, state: &mut AccumulatorState, text: &str) {
        if text.is_empty() {
            return;
        }

        let bytes = text.len();
        state.total_decoded_bytes += bytes;
        state.tail_text.push_str(text);
        state.tail_bytes += bytes;
        if state.tail_bytes > self.max_rolling_bytes * 2 {
            self.trim_tail(state);
        }

        let mut newlines = 0usize;
        let mut last_newline = None;
        for (index, character) in text.char_indices() {
            if character == '\n' {
                newlines += 1;
                last_newline = Some(index);
            }
        }
        if newlines == 0 {
            state.current_line_bytes += bytes;
            state.has_open_line = true;
        } else {
            state.completed_lines += newlines;
            let last_newline = last_newline.unwrap_or_default() + '\n'.len_utf8();
            let tail = &text[last_newline..];
            state.current_line_bytes = tail.len();
            state.has_open_line = !tail.is_empty();
        }
        state.total_lines = state.completed_lines + usize::from(state.has_open_line);
    }

    /// Drop the tail's front when it exceeds twice the rolling budget,
    /// upstream's `trimTail`, keeping the cut on a UTF-8 boundary.
    fn trim_tail(&self, state: &mut AccumulatorState) {
        let buffer = state.tail_text.clone().into_bytes();
        if buffer.len() <= self.max_rolling_bytes {
            state.tail_bytes = buffer.len();
            return;
        }

        let mut start = buffer.len() - self.max_rolling_bytes;
        while start < buffer.len() && (buffer[start] & 0xc0) == 0x80 {
            start += 1;
        }

        state.tail_starts_at_line_boundary = if start == 0 {
            state.tail_starts_at_line_boundary
        } else {
            buffer[start - 1] == b'\n'
        };
        state.tail_text = String::from_utf8_lossy(&buffer[start..]).into_owned();
        state.tail_bytes = state.tail_text.len();
    }

    /// The text the truncation runs over, upstream's `getSnapshotText`:
    /// when the tail was cut mid-line, the partial first line drops so the
    /// snapshot starts at a line boundary.
    fn get_snapshot_text(state: &'_ AccumulatorState) -> &str {
        if state.tail_starts_at_line_boundary {
            return &state.tail_text;
        }
        match state.tail_text.find('\n') {
            Some(first_newline) => &state.tail_text[first_newline + 1..],
            None => &state.tail_text,
        }
    }

    /// Whether the full output must be preserved, upstream's
    /// `shouldUseTempFile`.
    const fn should_use_temp_file(&self, state: &AccumulatorState) -> bool {
        state.total_raw_bytes > self.max_bytes
            || state.total_decoded_bytes > self.max_bytes
            || state.total_lines > self.max_lines
    }

    /// Open the temp file and backfill the raw chunks, upstream's
    /// `ensureTempFile`.
    fn ensure_temp_file(&self, state: &mut AccumulatorState) {
        if state.temp_file_path.is_some() {
            return;
        }
        let path = Self::default_temp_file_path(&self.temp_file_prefix);
        let Ok(file) = std::fs::File::create(&path) else {
            return;
        };
        state.temp_file_path = Some(path);
        let mut file = file;
        for chunk in state.raw_chunks.drain(..) {
            let _written = file.write_all(&chunk);
        }
        state.temp_file = Some(file);
    }
}

impl std::fmt::Debug for OutputAccumulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputAccumulator")
            .field("max_lines", &self.max_lines)
            .field("max_bytes", &self.max_bytes)
            .field("temp_file_prefix", &self.temp_file_prefix)
            .finish_non_exhaustive()
    }
}

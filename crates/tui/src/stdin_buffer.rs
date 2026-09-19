//! Stdin buffer: reassembles split escape sequences and splits paste batches.
//!
//! Port of `packages/tui/src/stdin-buffer.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#41). Stdin data can arrive in
//! partial chunks, especially for escape sequences like mouse events.
//! Without buffering, partial sequences can be misinterpreted as regular
//! keypresses. For example, the mouse SGR sequence `\x1b[<35;20;5m` might
//! arrive as:
//! - Event 1: `\x1b`
//! - Event 2: `[<35`
//! - Event 3: `;20;5m`
//!
//! The buffer accumulates these until a complete sequence is detected. Call
//! [`StdinBuffer::process`] to feed input data. Based on code from
//! OpenTUI (<https://github.com/anomalyco/opentui>),
//! MIT License - Copyright (c) 2025 opentui.
//!
//! Upstream drives this type with an `EventEmitter` plus `setTimeout`; Rust
//! has no implicit event loop, so the same behavior is a pull interface:
//! [`StdinBuffer::process`] and [`StdinBuffer::poll_flush`] return the
//! emitted [`StdinBufferEvent`]s, and the flush deadline that upstream's
//! timer would fire is exposed as [`StdinBuffer::flush_deadline`] for the
//! owner to pump. Two upstream branches are restated rather than copied:
//!
//! - The `Buffer` argument branch (`data.toString()` plus the single
//!   high-byte → ESC + char conversion) never fires here: the terminal byte
//!   feed decodes UTF-8 incrementally exactly as Node's `setEncoding("utf8")`
//!   did, so strings arrive whole and invalid bytes become U+FFFD before
//!   reaching this type.
//! - JS UTF-16 indexing becomes `char` indexing. Chunks are decoded on code
//!   points, so the only visible difference is that astral-plane characters
//!   survive intact where upstream's surrogate-pair arithmetic would split
//!   them.

use std::time::{Duration, Instant};

use crate::keys::scan_digits;

const ESC: char = '\x1b';
const DEFAULT_SEQUENCE_TIMEOUT_MS: u64 = 50;
const DEFAULT_ESCAPE_TIMEOUT_MS: u64 = 10;
const BRACKETED_PASTE_START: &str = "\x1b[200~";
const BRACKETED_PASTE_END: &str = "\x1b[201~";

/// An event [`StdinBuffer`] emits: a complete sequence (upstream's `data`
/// event) or the content of a complete bracketed paste (upstream's `paste`
/// event).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StdinBufferEvent {
    /// A complete input sequence, upstream's `data` event.
    Data(String),
    /// The content between the bracketed-paste markers, upstream's `paste`
    /// event.
    Paste(String),
}

/// Whether a string is a complete escape sequence or needs more data.
///
/// The caller only enters on ESC (upstream carries the same not-escape arm
/// unreachable), so the port keeps the two reachable answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceStatus {
    Complete,
    Incomplete,
}

/// Check if a string is a complete escape sequence or needs more data.
fn is_complete_sequence(data: &str) -> SequenceStatus {
    if data.chars().count() == 1 {
        return SequenceStatus::Incomplete;
    }

    let after_esc = &data[1..];

    // CSI sequences: ESC [
    if after_esc.starts_with('[') {
        // Check for old-style mouse sequence: ESC[M + 3 bytes
        if after_esc.starts_with("[M") {
            // Old-style mouse needs ESC[M + 3 bytes = 6 total
            return if data.chars().count() >= 6 {
                SequenceStatus::Complete
            } else {
                SequenceStatus::Incomplete
            };
        }
        return is_complete_csi_sequence(data);
    }

    // OSC sequences: ESC ]
    if after_esc.starts_with(']') {
        return is_complete_osc_sequence(data);
    }

    // DCS sequences: ESC P ... ESC \ (includes XTVersion responses)
    if after_esc.starts_with('P') {
        return is_complete_dcs_sequence(data);
    }

    // APC sequences: ESC _ ... ESC \ (includes Kitty graphics responses)
    if after_esc.starts_with('_') {
        return is_complete_apc_sequence(data);
    }

    // SS3 sequences: ESC O
    if after_esc.starts_with('O') {
        // ESC O followed by a single character
        return if after_esc.chars().count() >= 2 {
            SequenceStatus::Complete
        } else {
            SequenceStatus::Incomplete
        };
    }

    // Meta key sequences: ESC followed by a single character
    if after_esc.chars().count() == 1 {
        return SequenceStatus::Complete;
    }

    // Unknown escape sequence - treat as complete
    SequenceStatus::Complete
}

/// Check if CSI sequence is complete.
/// CSI sequences: ESC [ ... followed by a final byte (0x40-0x7E)
fn is_complete_csi_sequence(data: &str) -> SequenceStatus {
    // Need at least ESC [ and one more character
    if data.chars().count() < 3 {
        return SequenceStatus::Incomplete;
    }

    let payload = &data[2..];

    // CSI sequences end with a byte in the range 0x40-0x7E (@-~)
    // This includes all letters and several special characters
    let last_char = payload.chars().last().unwrap_or('?');
    let last_char_code = u32::from(last_char);

    if (0x40..=0x7e).contains(&last_char_code) {
        // Special handling for SGR mouse sequences
        // Format: ESC[<B;X;Ym or ESC[<B;X;YM
        if payload.starts_with('<') {
            // Must have format: <digits;digits;digits[Mm]
            if is_sgr_mouse_payload(payload) {
                return SequenceStatus::Complete;
            }
            // If it ends with M or m but doesn't match the pattern, check the
            // structure digit-group by digit-group.
            if last_char == 'M' || last_char == 'm' {
                let parts = payload[1..payload.len() - 1].split(';').collect::<Vec<_>>();
                if parts.len() == 3
                    && parts.iter().all(|part| {
                        !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit())
                    })
                {
                    return SequenceStatus::Complete;
                }
            }

            return SequenceStatus::Incomplete;
        }

        return SequenceStatus::Complete;
    }

    SequenceStatus::Incomplete
}

/// Restates the SGR-mouse regex `^<\d+;\d+;\d+[Mm]$` over the payload the
/// caller already narrowed to a leading `<` and an `M`/`m` final byte:
/// three non-empty digit runs separated by `;`.
fn is_sgr_mouse_payload(payload: &str) -> bool {
    let bytes = payload.as_bytes();
    let mut pos = 1;
    for group in 0..3 {
        let run_start = pos;
        while pos < bytes.len() - 1 && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos == run_start {
            return false;
        }
        if group < 2 {
            if bytes.get(pos) != Some(&b';') {
                return false;
            }
            pos += 1;
        }
    }
    pos == bytes.len() - 1
}

/// Check if OSC sequence is complete.
/// OSC sequences: ESC ] ... ST (where ST is ESC \ or BEL)
fn is_complete_osc_sequence(data: &str) -> SequenceStatus {
    // OSC sequences end with ST (ESC \) or BEL (\x07)
    if data.ends_with("\x1b\\") || data.ends_with('\x07') {
        return SequenceStatus::Complete;
    }

    SequenceStatus::Incomplete
}

/// Check if DCS (Device Control String) sequence is complete.
/// DCS sequences: ESC P ... ST (where ST is ESC \)
/// Used for XTVersion responses like ESC P >| ... ESC \
fn is_complete_dcs_sequence(data: &str) -> SequenceStatus {
    // DCS sequences end with ST (ESC \)
    if data.ends_with("\x1b\\") {
        return SequenceStatus::Complete;
    }

    SequenceStatus::Incomplete
}

/// Check if APC (Application Program Command) sequence is complete.
/// APC sequences: ESC _ ... ST (where ST is ESC \)
/// Used for Kitty graphics responses like ESC _ G ... ESC \
fn is_complete_apc_sequence(data: &str) -> SequenceStatus {
    // APC sequences end with ST (ESC \)
    if data.ends_with("\x1b\\") {
        return SequenceStatus::Complete;
    }

    SequenceStatus::Incomplete
}

/// Restates the Kitty-printable regex `^\x1b\[(\d+)(?::\d*)?(?::\d+)?u$`: a
/// CSI-u sequence with no modifier segment, whose leading codepoint is
/// printable (>= 32). Absurdly long digit runs saturate to a codepoint no
/// raw character can ever match, the same dead end upstream's `parseInt`
/// reaches.
fn parse_unmodified_kitty_printable_codepoint(sequence: &str) -> Option<u32> {
    let bytes = sequence.as_bytes();
    if bytes.len() < 4 || !bytes.starts_with(b"\x1b[") || bytes[bytes.len() - 1] != b'u' {
        return None;
    }

    let mut pos = 2;
    let codepoint = scan_digits(bytes, &mut pos)?;

    // The first colon slot allows an empty digit run (`\d*`).
    if pos < bytes.len() && bytes[pos] == b':' {
        pos += 1;
        scan_digits(bytes, &mut pos);
    }
    // The second colon slot requires digits.
    if pos < bytes.len() && bytes[pos] == b':' {
        pos += 1;
        scan_digits(bytes, &mut pos)?;
    }
    if pos != bytes.len() - 1 {
        return None;
    }

    let codepoint = u32::try_from(codepoint).ok()?;
    if codepoint >= 32 {
        Some(codepoint)
    } else {
        None
    }
}

/// Split accumulated buffer into complete sequences.
///
/// Returns the complete sequences and the trailing remainder that is still
/// waiting for more data.
fn extract_complete_sequences(buffer: &str) -> (Vec<String>, String) {
    let chars: Vec<char> = buffer.chars().collect();
    let mut sequences: Vec<String> = Vec::new();
    let mut pos = 0;

    while pos < chars.len() {
        if chars[pos] != ESC {
            // Not an escape sequence - take a single character
            sequences.push(chars[pos].to_string());
            pos += 1;
            continue;
        }

        // Find the end of this escape sequence
        let mut seq_end = 1;
        let mut completed = false;
        while seq_end <= chars.len() - pos {
            let candidate: String = chars[pos..pos + seq_end].iter().collect();

            if is_complete_sequence(&candidate) == SequenceStatus::Incomplete {
                seq_end += 1;
                continue;
            }

            // WezTerm with enable_kitty_keyboard sends the Escape key press as a
            // raw '\x1b' byte (simple text path in encode_kitty, ignoring
            // DISAMBIGUATE_ESCAPE_CODES) and the release as a full Kitty CSI-u
            // sequence. These arrive concatenated as '\x1b\x1b[27;...u'.
            // The buffer would normally treat '\x1b\x1b' as a complete meta-key
            // sequence (ESC + single char), leaving '[27;...u' to be typed as
            // plain text. If the character immediately following '\x1b\x1b'
            // would begin a new escape sequence, emit only the first ESC and
            // restart from the second.
            if candidate == "\x1b\x1b"
                && let Some(next_char) = chars.get(pos + seq_end)
                && matches!(next_char, '[' | ']' | 'O' | 'P' | '_')
            {
                sequences.push(ESC.to_string());
                pos += 1;
                completed = true;
                break;
            }

            sequences.push(candidate);
            pos += seq_end;
            completed = true;
            break;
        }

        if !completed {
            return (sequences, chars[pos..].iter().collect());
        }
    }

    (sequences, String::new())
}

/// Buffers stdin input and emits complete sequences.
/// Handles partial escape sequences that arrive across multiple chunks.
///
/// Upstream emitted events through `EventEmitter`; here [`StdinBuffer::process`]
/// and [`StdinBuffer::poll_flush`] return the emitted events in order, and
/// [`StdinBuffer::flush_deadline`] reports the pending timer the owner pumps.
#[derive(Debug)]
pub struct StdinBuffer {
    buffer: String,
    /// Maximum time to wait for an incomplete sequence such as CSI or mouse.
    timeout_ms: u64,
    /// Maximum time to wait after a lone ESC before treating it as Escape.
    /// Increase for high-latency Alt+key input (SSH).
    escape_timeout_ms: u64,
    paste_mode: bool,
    paste_buffer: String,
    pending_kitty_printable_codepoint: Option<u32>,
    flush_at: Option<Instant>,
}

impl Default for StdinBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl StdinBuffer {
    /// A buffer with the default timeouts: 50 ms for incomplete sequences,
    /// 10 ms after a lone ESC.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffer: String::new(),
            timeout_ms: DEFAULT_SEQUENCE_TIMEOUT_MS,
            escape_timeout_ms: DEFAULT_ESCAPE_TIMEOUT_MS,
            paste_mode: false,
            paste_buffer: String::new(),
            pending_kitty_printable_codepoint: None,
            flush_at: None,
        }
    }

    /// Overrides the sequence timeout in milliseconds.
    #[must_use]
    pub const fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Overrides the lone-ESC timeout in milliseconds.
    #[must_use]
    pub const fn with_escape_timeout_ms(mut self, escape_timeout_ms: u64) -> Self {
        self.escape_timeout_ms = escape_timeout_ms;
        self
    }

    /// Feed input data; returns the events emitted by this chunk in order.
    ///
    /// `now` is the time the chunk arrived; an incomplete tail arms the flush
    /// deadline that [`StdinBuffer::poll_flush`] fires.
    pub fn process(&mut self, data: &str, now: Instant) -> Vec<StdinBufferEvent> {
        self.flush_at = None;

        if data.is_empty() && self.buffer.is_empty() {
            // Empty string emits an empty data event, exactly as upstream.
            return self.emit_data_sequence(String::new()).into_iter().collect();
        }

        self.buffer.push_str(data);

        if self.paste_mode {
            self.paste_buffer.push_str(&self.buffer);
            self.buffer.clear();
            return self.complete_paste(now);
        }

        if let Some(start_index) = self.buffer.find(BRACKETED_PASTE_START) {
            let mut events = Vec::new();
            if start_index > 0 {
                let before_paste = self.buffer[..start_index].to_string();
                let (sequences, _) = extract_complete_sequences(&before_paste);
                for sequence in sequences {
                    if let Some(event) = self.emit_data_sequence(sequence) {
                        events.push(event);
                    }
                }
            }

            self.pending_kitty_printable_codepoint = None;
            self.paste_mode = true;
            self.paste_buffer =
                self.buffer[start_index + BRACKETED_PASTE_START.len()..].to_string();
            self.buffer.clear();
            events.extend(self.complete_paste(now));
            return events;
        }

        let (sequences, remainder) = extract_complete_sequences(&self.buffer);
        self.buffer = remainder;

        let mut events = Vec::new();
        for sequence in sequences {
            if let Some(event) = self.emit_data_sequence(sequence) {
                events.push(event);
            }
        }

        if !self.buffer.is_empty() {
            let timeout_ms = if self.buffer == "\x1b" {
                self.escape_timeout_ms
            } else {
                self.timeout_ms
            };
            self.flush_at = Some(now + Duration::from_millis(timeout_ms));
        }

        events
    }

    /// Emits the buffered tail as data once the flush deadline has passed.
    ///
    /// Upstream armed `setTimeout` and let the event loop call `flush()`; the
    /// port hands the deadline to the owner, which calls this with the
    /// current time.
    pub fn poll_flush(&mut self, now: Instant) -> Vec<StdinBufferEvent> {
        let Some(flush_at) = self.flush_at else {
            return Vec::new();
        };
        if now < flush_at {
            return Vec::new();
        }

        self.flush()
            .into_iter()
            .filter_map(|sequence| self.emit_data_sequence(sequence))
            .collect()
    }

    /// The deadline an armed flush fires at, if any.
    #[must_use]
    pub const fn flush_deadline(&self) -> Option<Instant> {
        self.flush_at
    }

    fn complete_paste(&mut self, now: Instant) -> Vec<StdinBufferEvent> {
        let Some(end_index) = self.paste_buffer.find(BRACKETED_PASTE_END) else {
            return Vec::new();
        };

        let pasted_content = self.paste_buffer[..end_index].to_string();
        let remaining = self.paste_buffer[end_index + BRACKETED_PASTE_END.len()..].to_string();

        self.paste_mode = false;
        self.paste_buffer.clear();
        self.pending_kitty_printable_codepoint = None;

        let mut events = vec![StdinBufferEvent::Paste(pasted_content)];
        if !remaining.is_empty() {
            events.extend(self.process(&remaining, now));
        }
        events
    }

    /// Drops the duplicate raw character a terminal echoes after a printable
    /// Kitty CSI-u press, restating upstream's `emitDataSequence`: the
    /// pending codepoint from a parsed unmodified printable sequence swallows
    /// exactly one matching single-character sequence.
    fn emit_data_sequence(&mut self, sequence: String) -> Option<StdinBufferEvent> {
        let raw_codepoint = single_char_codepoint(&sequence);
        if let Some(codepoint) = raw_codepoint
            && self.pending_kitty_printable_codepoint == Some(codepoint)
        {
            self.pending_kitty_printable_codepoint = None;
            return None;
        }

        self.pending_kitty_printable_codepoint =
            parse_unmodified_kitty_printable_codepoint(&sequence);
        Some(StdinBufferEvent::Data(sequence))
    }

    /// Returns and clears the buffered incomplete sequence, if any.
    pub fn flush(&mut self) -> Vec<String> {
        self.flush_at = None;
        if self.buffer.is_empty() {
            return Vec::new();
        }

        self.pending_kitty_printable_codepoint = None;
        vec![std::mem::take(&mut self.buffer)]
    }

    /// Clears buffered content without emitting.
    pub fn clear(&mut self) {
        self.flush_at = None;
        self.buffer.clear();
        self.paste_mode = false;
        self.paste_buffer.clear();
        self.pending_kitty_printable_codepoint = None;
    }

    /// The incomplete sequence currently buffered.
    #[must_use]
    pub fn get_buffer(&self) -> &str {
        &self.buffer
    }
}

/// The codepoint of a one-`char` sequence, restating upstream's
/// `sequence.codePointAt(0)` guard on `sequence.length === 1`.
fn single_char_codepoint(sequence: &str) -> Option<u32> {
    let mut chars = sequence.chars();
    let first = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    Some(u32::from(first))
}

//! Port of `packages/tui/test/stdin-buffer.test.ts` — 1:1 against upstream
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#41).
//!
//! Upstream collected `data`/`paste` events through `EventEmitter` and waited
//! on the runtime's timers; the port collects the event vectors
//! [`StdinBuffer::process`] and [`StdinBuffer::poll_flush`] return and drives
//! the same timers through [`StdinBuffer::flush_deadline`].

use std::time::{Duration, Instant};

use pi_tui::keys::KeyParser;
use pi_tui::stdin_buffer::{StdinBuffer, StdinBufferEvent};

struct Harness {
    buffer: StdinBuffer,
    emitted_sequences: Vec<String>,
    emitted_pastes: Vec<String>,
    /// A synthetic clock, advanced by `wait` without real sleeps — upstream's
    /// tests slept real milliseconds against runtime timers, which races
    /// under a parallel test runner; the port's deadlines are explicit
    /// instants, so the same waits are deterministic.
    now: Instant,
}

impl Harness {
    fn with_timeouts(timeout_ms: u64, escape_timeout_ms: u64) -> Self {
        Self {
            buffer: StdinBuffer::new()
                .with_timeout_ms(timeout_ms)
                .with_escape_timeout_ms(escape_timeout_ms),
            emitted_sequences: Vec::new(),
            emitted_pastes: Vec::new(),
            now: Instant::now(),
        }
    }

    fn process_input(&mut self, data: &str) {
        let events = self.buffer.process(data, self.now);
        self.collect(events);
    }

    /// Advances the synthetic clock past `ms` and fires the flush deadline
    /// the way upstream's timer callback would.
    fn wait(&mut self, ms: u64) {
        self.now += Duration::from_millis(ms);
        let events = self.buffer.poll_flush(self.now);
        self.collect(events);
    }

    fn collect(&mut self, events: Vec<StdinBufferEvent>) {
        for event in events {
            match event {
                StdinBufferEvent::Data(sequence) => self.emitted_sequences.push(sequence),
                StdinBufferEvent::Paste(content) => self.emitted_pastes.push(content),
            }
        }
    }
}

fn matches_key(data: &str, key_id: &str) -> bool {
    KeyParser::new().matches_key(data, key_id)
}

// =============================================================================
// Regular Characters
// =============================================================================

#[test]
fn should_pass_through_regular_characters_immediately() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("a");
    assert_eq!(harness.emitted_sequences, ["a"]);
}

#[test]
fn should_pass_through_multiple_regular_characters() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("abc");
    assert_eq!(harness.emitted_sequences, ["a", "b", "c"]);
}

#[test]
fn should_handle_unicode_characters() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("hello 世界");
    assert_eq!(
        harness.emitted_sequences,
        ["h", "e", "l", "l", "o", " ", "世", "界"]
    );
}

// =============================================================================
// Complete Escape Sequences
// =============================================================================

#[test]
fn should_pass_through_complete_mouse_sgr_sequences() {
    let mut harness = Harness::with_timeouts(10, 10);
    let mouse_seq = "\x1b[<35;20;5m";
    harness.process_input(mouse_seq);
    assert_eq!(harness.emitted_sequences, [mouse_seq]);
}

#[test]
fn should_pass_through_complete_arrow_key_sequences() {
    let mut harness = Harness::with_timeouts(10, 10);
    let up_arrow = "\x1b[A";
    harness.process_input(up_arrow);
    assert_eq!(harness.emitted_sequences, [up_arrow]);
}

#[test]
fn should_pass_through_complete_function_key_sequences() {
    let mut harness = Harness::with_timeouts(10, 10);
    let f1 = "\x1b[11~";
    harness.process_input(f1);
    assert_eq!(harness.emitted_sequences, [f1]);
}

#[test]
fn should_pass_through_meta_key_sequences() {
    let mut harness = Harness::with_timeouts(10, 10);
    let meta_a = "\x1ba";
    harness.process_input(meta_a);
    assert_eq!(harness.emitted_sequences, [meta_a]);
}

#[test]
fn should_pass_through_ss3_sequences() {
    let mut harness = Harness::with_timeouts(10, 10);
    let ss3 = "\x1bOA";
    harness.process_input(ss3);
    assert_eq!(harness.emitted_sequences, [ss3]);
}

// =============================================================================
// Partial Escape Sequences
// =============================================================================

#[test]
fn should_buffer_incomplete_mouse_sgr_sequence() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b");
    assert!(harness.emitted_sequences.is_empty());
    assert_eq!(harness.buffer.get_buffer(), "\x1b");

    harness.process_input("[<35");
    assert!(harness.emitted_sequences.is_empty());
    assert_eq!(harness.buffer.get_buffer(), "\x1b[<35");

    harness.process_input(";20;5m");
    assert_eq!(harness.emitted_sequences, ["\x1b[<35;20;5m"]);
    assert_eq!(harness.buffer.get_buffer(), "");
}

#[test]
fn should_buffer_incomplete_csi_sequence() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[");
    assert!(harness.emitted_sequences.is_empty());

    harness.process_input("1;");
    assert!(harness.emitted_sequences.is_empty());

    harness.process_input("5H");
    assert_eq!(harness.emitted_sequences, ["\x1b[1;5H"]);
}

#[test]
fn should_buffer_split_across_many_chunks() {
    let mut harness = Harness::with_timeouts(10, 10);
    for chunk in ["\x1b", "[", "<", "3", "5", ";", "2", "0", ";", "5", "m"] {
        harness.process_input(chunk);
    }
    assert_eq!(harness.emitted_sequences, ["\x1b[<35;20;5m"]);
}

#[test]
fn should_flush_incomplete_sequence_after_timeout() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<35");
    assert!(harness.emitted_sequences.is_empty());

    // Wait for timeout
    harness.wait(15);

    assert_eq!(harness.emitted_sequences, ["\x1b[<35"]);
}

#[test]
fn should_flush_a_lone_esc_as_escape_when_cr_arrives_after_the_timeout() {
    // Legacy-mode Alt+Enter is ESC + CR; when the terminal/transport splits
    // the bytes further apart than the timeout, ESC is flushed alone and the
    // host sees Escape (interrupt) instead of Alt+Enter. This locks in the
    // behavior so the configurable timeout in ProcessTerminal stays honest.
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b");
    harness.wait(20); // buffer timeout is 10ms in the harness
    harness.process_input("\r");

    assert_eq!(harness.emitted_sequences, ["\x1b", "\r"]);
    assert!(matches_key(&harness.emitted_sequences[0], "escape"));
}

#[test]
fn should_merge_esc_cr_split_across_chunks_within_a_larger_timeout() {
    let mut harness = Harness::with_timeouts(10, 100);

    harness.process_input("\x1b");
    harness.wait(20); // > 10ms default escapeTimeout, < 100ms configured escapeTimeout
    harness.process_input("\r");

    assert_eq!(harness.emitted_sequences, ["\x1b\r"]);
    assert!(matches_key(&harness.emitted_sequences[0], "alt+enter"));
}

#[test]
fn does_not_apply_the_sequence_timeout_to_a_lone_esc() {
    let mut harness = Harness::with_timeouts(100, 10);

    harness.process_input("\x1b");
    harness.wait(20);
    harness.process_input("\r");

    assert_eq!(harness.emitted_sequences, ["\x1b", "\r"]);
    assert!(matches_key(&harness.emitted_sequences[0], "escape"));
}

#[test]
fn keeps_fragmented_mouse_sequences_buffered_across_delayed_chunks_by_default() {
    let mut harness = Harness::with_timeouts(50, 10);
    harness.process_input("\x1b[");
    harness.wait(20);
    assert!(harness.emitted_sequences.is_empty());
    harness.process_input("<65;48;39M");
    assert_eq!(harness.emitted_sequences, ["\x1b[<65;48;39M"]);
}

// =============================================================================
// Mixed Content
// =============================================================================

#[test]
fn should_handle_characters_followed_by_escape_sequence() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("abc\x1b[A");
    assert_eq!(harness.emitted_sequences, ["a", "b", "c", "\x1b[A"]);
}

#[test]
fn should_handle_escape_sequence_followed_by_characters() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[Aabc");
    assert_eq!(harness.emitted_sequences, ["\x1b[A", "a", "b", "c"]);
}

#[test]
fn should_handle_multiple_complete_sequences() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[A\x1b[B\x1b[C");
    assert_eq!(harness.emitted_sequences, ["\x1b[A", "\x1b[B", "\x1b[C"]);
}

#[test]
fn should_handle_partial_sequence_with_preceding_characters() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("abc\x1b[<35");
    assert_eq!(harness.emitted_sequences, ["a", "b", "c"]);
    assert_eq!(harness.buffer.get_buffer(), "\x1b[<35");

    harness.process_input(";20;5m");
    assert_eq!(harness.emitted_sequences, ["a", "b", "c", "\x1b[<35;20;5m"]);
}

// =============================================================================
// Kitty Keyboard Protocol
// =============================================================================

#[test]
fn should_handle_kitty_csi_u_press_events() {
    let mut harness = Harness::with_timeouts(10, 10);
    // Press 'a' in Kitty protocol
    harness.process_input("\x1b[97u");
    assert_eq!(harness.emitted_sequences, ["\x1b[97u"]);
}

#[test]
fn should_handle_kitty_csi_u_release_events() {
    let mut harness = Harness::with_timeouts(10, 10);
    // Release 'a' in Kitty protocol
    harness.process_input("\x1b[97;1:3u");
    assert_eq!(harness.emitted_sequences, ["\x1b[97;1:3u"]);
}

#[test]
fn should_handle_batched_kitty_press_and_release() {
    let mut harness = Harness::with_timeouts(10, 10);
    // Press 'a', release 'a' batched together (common over SSH)
    harness.process_input("\x1b[97u\x1b[97;1:3u");
    assert_eq!(harness.emitted_sequences, ["\x1b[97u", "\x1b[97;1:3u"]);
}

#[test]
fn should_handle_multiple_batched_kitty_events() {
    let mut harness = Harness::with_timeouts(10, 10);
    // Press 'a', release 'a', press 'b', release 'b'
    harness.process_input("\x1b[97u\x1b[97;1:3u\x1b[98u\x1b[98;1:3u");
    assert_eq!(
        harness.emitted_sequences,
        ["\x1b[97u", "\x1b[97;1:3u", "\x1b[98u", "\x1b[98;1:3u"]
    );
}

#[test]
fn should_handle_kitty_arrow_keys_with_event_type() {
    let mut harness = Harness::with_timeouts(10, 10);
    // Up arrow press with event type
    harness.process_input("\x1b[1;1:1A");
    assert_eq!(harness.emitted_sequences, ["\x1b[1;1:1A"]);
}

#[test]
fn should_handle_kitty_functional_keys_with_event_type() {
    let mut harness = Harness::with_timeouts(10, 10);
    // Delete key release
    harness.process_input("\x1b[3;1:3~");
    assert_eq!(harness.emitted_sequences, ["\x1b[3;1:3~"]);
}

#[test]
fn should_split_esc_esc_csi_into_standalone_esc_and_the_csi_sequence() {
    let mut harness = Harness::with_timeouts(10, 10);
    // WezTerm with enable_kitty_keyboard sends Escape key press as raw \x1b
    // and the release as a full Kitty CSI-u sequence, concatenated.
    // The buffer must not treat \x1b\x1b as a complete meta-key when the
    // following byte starts a new escape sequence.
    harness.process_input("\x1b\x1b[27;129:3u");
    assert_eq!(harness.emitted_sequences, ["\x1b", "\x1b[27;129:3u"]);
}

#[test]
fn should_split_esc_esc_csi_with_no_modifier() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b\x1b[27;1:3u");
    assert_eq!(harness.emitted_sequences, ["\x1b", "\x1b[27;1:3u"]);
}

#[test]
fn should_still_emit_esc_esc_as_a_single_sequence_when_not_followed_by_a_new_escape() {
    let mut harness = Harness::with_timeouts(10, 10);
    // \x1b\x1b alone (no following CSI) stays as-is — e.g. ctrl+alt+[
    harness.process_input("\x1b\x1b");
    assert_eq!(harness.emitted_sequences, ["\x1b\x1b"]);
}

#[test]
fn should_handle_plain_characters_mixed_with_kitty_sequences() {
    let mut harness = Harness::with_timeouts(10, 10);
    // Plain 'a' followed by Kitty release
    harness.process_input("a\x1b[97;1:3u");
    assert_eq!(harness.emitted_sequences, ["a", "\x1b[97;1:3u"]);
}

#[test]
fn should_drop_raw_duplicate_character_after_matching_kitty_printable_sequence() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[224uà");
    assert_eq!(harness.emitted_sequences, ["\x1b[224u"]);
}

#[test]
fn should_drop_raw_duplicate_character_after_matching_kitty_printable_sequence_across_chunks() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[64u");
    harness.process_input("@");
    assert_eq!(harness.emitted_sequences, ["\x1b[64u"]);
}

#[test]
fn should_keep_non_matching_plain_character_after_kitty_printable_sequence() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[97ub");
    assert_eq!(harness.emitted_sequences, ["\x1b[97u", "b"]);
}

#[test]
fn should_keep_raw_character_after_modified_kitty_printable_sequence() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[64;3u@");
    assert_eq!(harness.emitted_sequences, ["\x1b[64;3u", "@"]);
}

#[test]
fn should_handle_rapid_typing_simulation_with_kitty_protocol() {
    let mut harness = Harness::with_timeouts(10, 10);
    // Simulates typing "hi" quickly with releases interleaved
    harness.process_input("\x1b[104u\x1b[104;1:3u\x1b[105u\x1b[105;1:3u");
    assert_eq!(
        harness.emitted_sequences,
        ["\x1b[104u", "\x1b[104;1:3u", "\x1b[105u", "\x1b[105;1:3u"]
    );
}

// =============================================================================
// Mouse Events
// =============================================================================

#[test]
fn should_handle_mouse_press_event() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<0;10;5M");
    assert_eq!(harness.emitted_sequences, ["\x1b[<0;10;5M"]);
}

#[test]
fn should_handle_mouse_release_event() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<0;10;5m");
    assert_eq!(harness.emitted_sequences, ["\x1b[<0;10;5m"]);
}

#[test]
fn should_handle_mouse_move_event() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<35;20;5m");
    assert_eq!(harness.emitted_sequences, ["\x1b[<35;20;5m"]);
}

#[test]
fn should_handle_split_mouse_events() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<3");
    harness.process_input("5;1");
    harness.process_input("5;");
    harness.process_input("10m");
    assert_eq!(harness.emitted_sequences, ["\x1b[<35;15;10m"]);
}

#[test]
fn should_handle_multiple_mouse_events() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<35;1;1m\x1b[<35;2;2m\x1b[<35;3;3m");
    assert_eq!(
        harness.emitted_sequences,
        ["\x1b[<35;1;1m", "\x1b[<35;2;2m", "\x1b[<35;3;3m"]
    );
}

#[test]
fn should_handle_old_style_mouse_sequence() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[M abc");
    assert_eq!(harness.emitted_sequences, ["\x1b[M ab", "c"]);
}

#[test]
fn should_buffer_incomplete_old_style_mouse_sequence() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[M");
    assert_eq!(harness.buffer.get_buffer(), "\x1b[M");

    harness.process_input(" a");
    assert_eq!(harness.buffer.get_buffer(), "\x1b[M a");

    harness.process_input("b");
    assert_eq!(harness.emitted_sequences, ["\x1b[M ab"]);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn should_handle_empty_input() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("");
    // Empty string emits an empty data event
    assert_eq!(harness.emitted_sequences, [""]);
}

#[test]
fn should_handle_lone_escape_character_with_timeout() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b");
    assert!(harness.emitted_sequences.is_empty());

    // After timeout, should emit
    harness.wait(15);
    assert_eq!(harness.emitted_sequences, ["\x1b"]);
}

#[test]
fn flushes_a_lone_escape_promptly_with_the_longer_default_sequence_timeout() {
    let mut harness = Harness::with_timeouts(50, 10);
    harness.process_input("\x1b");
    harness.wait(20);
    assert_eq!(harness.emitted_sequences, ["\x1b"]);
}

#[test]
fn should_handle_lone_escape_character_with_explicit_flush() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b");
    assert!(harness.emitted_sequences.is_empty());

    let flushed = harness.buffer.flush();
    assert_eq!(flushed, ["\x1b"]);
}

#[test]
fn should_handle_very_long_sequences() {
    let mut harness = Harness::with_timeouts(10, 10);
    let long_seq = format!("\x1b[{}H", "1;".repeat(50));
    harness.process_input(&long_seq);
    assert_eq!(harness.emitted_sequences, [long_seq]);
}

// =============================================================================
// Flush
// =============================================================================

#[test]
fn should_flush_incomplete_sequences() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<35");
    let flushed = harness.buffer.flush();
    assert_eq!(flushed, ["\x1b[<35"]);
    assert_eq!(harness.buffer.get_buffer(), "");
}

#[test]
fn should_return_empty_array_if_nothing_to_flush() {
    let mut harness = Harness::with_timeouts(10, 10);
    let flushed = harness.buffer.flush();
    assert!(flushed.is_empty());
}

#[test]
fn should_emit_flushed_data_via_timeout() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<35");
    assert!(harness.emitted_sequences.is_empty());

    // Wait for timeout to flush
    harness.wait(15);

    assert_eq!(harness.emitted_sequences, ["\x1b[<35"]);
}

// =============================================================================
// Clear
// =============================================================================

#[test]
fn should_clear_buffered_content_without_emitting() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<35");
    assert_eq!(harness.buffer.get_buffer(), "\x1b[<35");

    harness.buffer.clear();
    assert_eq!(harness.buffer.get_buffer(), "");
    assert!(harness.emitted_sequences.is_empty());
}

// =============================================================================
// Bracketed Paste
// =============================================================================

#[test]
fn should_emit_paste_event_for_complete_bracketed_paste() {
    let mut harness = Harness::with_timeouts(10, 10);
    let paste_start = "\x1b[200~";
    let paste_end = "\x1b[201~";
    let content = "hello world";

    harness.process_input(&format!("{paste_start}{content}{paste_end}"));

    assert_eq!(harness.emitted_pastes, ["hello world"]);
    assert!(harness.emitted_sequences.is_empty()); // No data events during paste
}

#[test]
fn should_handle_paste_arriving_in_chunks() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[200~");
    assert!(harness.emitted_pastes.is_empty());

    harness.process_input("hello ");
    assert!(harness.emitted_pastes.is_empty());

    harness.process_input("world\x1b[201~");
    assert_eq!(harness.emitted_pastes, ["hello world"]);
    assert!(harness.emitted_sequences.is_empty());
}

#[test]
fn should_handle_paste_with_input_before_and_after() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("a");
    harness.process_input("\x1b[200~pasted\x1b[201~");
    harness.process_input("b");

    assert_eq!(harness.emitted_sequences, ["a", "b"]);
    assert_eq!(harness.emitted_pastes, ["pasted"]);
}

#[test]
fn should_handle_paste_with_newlines() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[200~line1\nline2\nline3\x1b[201~");

    assert_eq!(harness.emitted_pastes, ["line1\nline2\nline3"]);
    assert!(harness.emitted_sequences.is_empty());
}

#[test]
fn should_handle_paste_with_unicode() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[200~Hello 世界 🎉\x1b[201~");

    assert_eq!(harness.emitted_pastes, ["Hello 世界 🎉"]);
    assert!(harness.emitted_sequences.is_empty());
}

#[test]
fn should_clear_pending_timeouts_on_destroy() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<35");
    harness.buffer.clear();

    // Wait longer than timeout
    harness.wait(15);

    // Should not have emitted anything
    assert!(harness.emitted_sequences.is_empty());
}

#[test]
fn an_empty_chunk_with_a_pending_tail_rearms_the_deadline_without_emitting() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<35");
    harness.process_input("");
    assert!(harness.emitted_sequences.is_empty());
    assert_eq!(harness.buffer.get_buffer(), "\x1b[<35");
}

// =============================================================================
// Boundary tests: branches upstream leaves untested, so the 95% coverage
// gate binds (#41).
// =============================================================================

/// Boundary: OSC responses (the query replies pi sends) pass through whole,
/// reassemble across chunks, and hold when the string terminator has not
/// arrived.
#[test]
fn osc_sequences_pass_through_and_reassemble() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b]11;#ffffff\x07");
    assert_eq!(harness.emitted_sequences, ["\x1b]11;#ffffff\x07"]);

    let mut split = Harness::with_timeouts(10, 10);
    split.process_input("\x1b]11;#ff");
    assert_eq!(split.buffer.get_buffer(), "\x1b]11;#ff");
    split.process_input("ffff\x1b\\");
    assert_eq!(split.emitted_sequences, ["\x1b]11;#ffffff\x1b\\"]);
}

/// Boundary: DCS responses (XTVersion replies) end at ST only.
#[test]
fn dcs_sequences_pass_through_and_reassemble() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1bP>|xterm(370)\x1b\\");
    assert_eq!(harness.emitted_sequences, ["\x1bP>|xterm(370)\x1b\\"]);

    let mut split = Harness::with_timeouts(10, 10);
    split.process_input("\x1bP>|");
    assert_eq!(split.buffer.get_buffer(), "\x1bP>|");
    split.process_input("xt\x1b\\");
    assert_eq!(split.emitted_sequences, ["\x1bP>|xt\x1b\\"]);
}

/// Boundary: APC responses (Kitty graphics replies) end at ST only.
#[test]
fn apc_sequences_pass_through_and_reassemble() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b_Gi=1;\x1b\\");
    assert_eq!(harness.emitted_sequences, ["\x1b_Gi=1;\x1b\\"]);

    let mut split = Harness::with_timeouts(10, 10);
    split.process_input("\x1b_Gi=1;");
    assert_eq!(split.buffer.get_buffer(), "\x1b_Gi=1;");
    split.process_input("\x1b\\");
    assert_eq!(split.emitted_sequences, ["\x1b_Gi=1;\x1b\\"]);
}

/// Boundary: an SGR payload that ends with the mouse final byte but carries
/// too few groups stays incomplete.
#[test]
fn malformed_sgr_mouse_payloads_stay_buffered() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<1;2");
    assert_eq!(harness.buffer.get_buffer(), "\x1b[<1;2");

    harness.process_input(";5M");
    assert_eq!(harness.emitted_sequences, ["\x1b[<1;2;5M"]);
}

/// Boundary: an SGR payload with empty digit runs never completes.
#[test]
fn sgr_mouse_payloads_with_empty_digit_runs_stay_buffered() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<;2;3m");
    assert_eq!(harness.buffer.get_buffer(), "\x1b[<;2;3m");

    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<1;;3M");
    assert_eq!(harness.buffer.get_buffer(), "\x1b[<1;;3M");
}

/// Boundary: absurdly long codepoint runs saturate instead of matching, the
/// same dead end upstream's `parseInt` reaches.
#[test]
fn overlong_kitty_printable_codepoints_do_not_arm_the_drop() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[99999999999999999999999999u@");
    assert_eq!(
        harness.emitted_sequences,
        ["\x1b[99999999999999999999999999u", "@"]
    );
}
/// Boundary: upstream indexes UTF-16 code units; an astral character pushed
/// one chunk at a time would split into lone surrogates. The port indexes
/// chars, so the pair stays whole.
#[test]
fn astral_plane_characters_survive_intact() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("🎉");
    assert_eq!(harness.emitted_sequences, ["🎉"]);
}

#[test]
fn sub_32_codepoint_kitty_printable_sequences_do_not_arm_the_drop() {
    let mut harness = Harness::with_timeouts(10, 10);
    // \x1b[27u parses as a CSI-u event but codepoint 27 is not printable, so
    // no duplicate is expected and the following ESC flushes normally.
    harness.process_input("\x1b[27u");
    harness.process_input("\x1b");
    harness.wait(15);
    assert_eq!(harness.emitted_sequences, ["\x1b[27u", "\x1b"]);
}

#[test]
fn kitty_printable_dedup_survives_a_non_matching_intermediate() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[64u");
    harness.process_input("b");
    harness.process_input("@");
    assert_eq!(harness.emitted_sequences, ["\x1b[64u", "b", "@"]);
}

#[test]
fn paste_content_with_kitty_release_shaped_runs_stays_in_the_paste() {
    let mut harness = Harness::with_timeouts(10, 10);
    // A bluetooth-MAC-shaped run inside paste content must not leak out as
    // data events; the paste consumes everything between the markers.
    harness.process_input("\x1b[200~90:62:3F:A5\x1b[201~");
    assert_eq!(harness.emitted_pastes, ["90:62:3F:A5"]);
    assert!(harness.emitted_sequences.is_empty());
}

#[test]
fn flush_deadline_tracks_the_pending_tail() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<35");
    assert!(harness.buffer.flush_deadline().is_some());
    harness.buffer.clear();
    assert!(harness.buffer.flush_deadline().is_none());
}

/// Boundary: a Kitty CSI-u with shifted-key and base-layout slots arms the
/// drop on the leading codepoint, exactly as the no-slot form does.
#[test]
fn kitty_printable_slots_still_arm_the_drop() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[97:2u");
    harness.process_input("a");
    assert_eq!(harness.emitted_sequences, ["\x1b[97:2u"]);

    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[97::5u");
    harness.process_input("a");
    assert_eq!(harness.emitted_sequences, ["\x1b[97::5u"]);
}

/// Boundary: a Kitty CSI-u whose base-layout slot is required but empty is
/// not an unmodified printable and does not arm the drop.
#[test]
fn kitty_printable_with_a_second_empty_slot_does_not_arm_the_drop() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[97::u");
    harness.process_input("a");
    assert_eq!(harness.emitted_sequences, ["\x1b[97::u", "a"]);
}

/// Boundary: an SGR payload whose digit run is cut short by a non-`;` byte
/// never completes.
#[test]
fn sgr_mouse_payload_with_a_stray_byte_stays_buffered() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1b[<1X;2;3m");
    assert_eq!(harness.buffer.get_buffer(), "\x1b[<1X;2;3m");
}

/// Boundary: the `Default` impl matches `new`.
#[test]
fn default_matches_new() {
    let default = StdinBuffer::default();
    assert!(default.get_buffer().is_empty());
    assert!(default.flush_deadline().is_none());
}

/// Boundary: the unknown-escape fallthrough treats an ESC-prefixed sequence
/// with an unknown initiator as complete, exactly as upstream.
#[test]
fn meta_key_sequences_complete_without_a_second_chunk() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("\x1ba\x1bb");
    assert_eq!(harness.emitted_sequences, ["\x1ba", "\x1bb"]);
}

/// Boundary: input before the paste marker still flows through the
/// sequence extractor, and a tail after the paste end marker recurses.
#[test]
fn paste_with_preceding_input_in_one_chunk_and_a_trailing_tail() {
    let mut harness = Harness::with_timeouts(10, 10);
    harness.process_input("a\x1b[200~pasted\x1b[201~b");

    assert_eq!(harness.emitted_sequences, ["a", "b"]);
    assert_eq!(harness.emitted_pastes, ["pasted"]);
}

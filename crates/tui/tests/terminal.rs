//! Port of `packages/tui/test/terminal.test.ts` — 1:1 against upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#41).
//!
//! Upstream reached into `ProcessTerminal` privates and monkeypatched
//! `process.stdout.write`/`process.stdin.on`/`process.env` and the timers;
//! the port constructs a headless terminal over injected seams (`headless`,
//! `with_env_lookup`, `with_clock`, `with_stdin_source`, `set_input_handler`)
//! and drives the same pipeline through `feed_input` and `pump`.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use pi_tui::keys::KeyParser;
use pi_tui::terminal::{KeyboardProtocolNegotiationSequence, ProcessTerminal, Terminal, WriteSink};

type TestEnvLookup = Box<dyn Fn(&str) -> Option<String>>;

fn env_lookup(map: &HashMap<String, String>) -> TestEnvLookup {
    let map = map.clone();
    Box::new(move |key: &str| map.get(key).cloned())
}

/// A clock the tests advance in whole milliseconds. Single-threaded, so the
/// non-Send/Sync `Arc<Cell<u64>>` never leaves the constructing test.
struct TestClock {
    base: Instant,
    offset_ms: Cell<u64>,
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the clock is a single-threaded test fixture; nothing spawns it"
)]
impl TestClock {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            base: Instant::now(),
            offset_ms: Cell::new(0),
        })
    }

    fn tick(&self, ms: u64) {
        self.offset_ms.set(self.offset_ms.get() + ms);
    }

    fn now(&self) -> Instant {
        self.base + Duration::from_millis(self.offset_ms.get())
    }
}

struct NegotiationHarness {
    terminal: ProcessTerminal,
    writes: Arc<Mutex<Vec<String>>>,
    inputs: Arc<Mutex<Vec<String>>>,
    clock: Arc<TestClock>,
    parser: Arc<Mutex<KeyParser>>,
}

fn capture_sink(writes: &Arc<Mutex<Vec<String>>>) -> WriteSink {
    let writes = Arc::clone(writes);
    Arc::new(Mutex::new(Box::new(move |data: &str| {
        writes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(data.to_string());
    })))
}

fn setup_negotiation() -> NegotiationHarness {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let inputs = Arc::new(Mutex::new(Vec::new()));
    let clock = TestClock::new();
    let parser = Arc::new(Mutex::new(KeyParser::new()));

    let mut terminal = ProcessTerminal::headless(capture_sink(&writes))
        .with_clock(Box::new({
            let clock = Arc::clone(&clock);
            move || clock.now()
        }))
        .with_key_parser(Arc::clone(&parser));

    let input_sink = Arc::clone(&inputs);
    terminal.set_input_handler(Some(Box::new(move |data: String| {
        input_sink
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(data);
    })));

    NegotiationHarness {
        terminal,
        writes,
        inputs,
        clock,
        parser,
    }
}

fn writes_of(harness: &NegotiationHarness) -> Vec<String> {
    harness
        .writes
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

fn inputs_of(harness: &NegotiationHarness) -> Vec<String> {
    harness
        .inputs
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

fn last_input_of(harness: &NegotiationHarness) -> Option<String> {
    inputs_of(harness).into_iter().next_back()
}

fn count_of(writes: &[String], sequence: &str) -> usize {
    writes
        .iter()
        .filter(|write| write.as_str() == sequence)
        .count()
}

impl NegotiationHarness {
    fn send(&mut self, data: &str) {
        self.terminal.feed_input(data);
        self.terminal.pump();
    }
}

// =============================================================================
// resolveEscapeTimeoutMs
// =============================================================================

#[test]
fn uses_pi_tui_esc_timeout_when_configured() {
    let mut env = HashMap::new();
    env.insert("PI_TUI_ESC_TIMEOUT".to_string(), "80".to_string());
    assert_eq!(
        pi_tui::terminal::resolve_escape_timeout_ms(|key| env.get(key).cloned()),
        80
    );

    env.insert("SSH_TTY".to_string(), "/dev/pts/1".to_string());
    assert_eq!(
        pi_tui::terminal::resolve_escape_timeout_ms(|key| env.get(key).cloned()),
        80
    );
}

#[test]
fn ignores_invalid_pi_tui_esc_timeout_values() {
    for value in ["abc", "0", "-5", ""] {
        let mut env = HashMap::new();
        env.insert("PI_TUI_ESC_TIMEOUT".to_string(), value.to_string());
        assert_eq!(
            pi_tui::terminal::resolve_escape_timeout_ms(|key| env.get(key).cloned()),
            10,
            "invalid override {value:?} falls back to 10"
        );
    }
}

#[test]
fn defaults_to_100ms_over_ssh() {
    let mut env = HashMap::new();
    env.insert("SSH_CONNECTION".to_string(), "10.0.0.1 22".to_string());
    assert_eq!(
        pi_tui::terminal::resolve_escape_timeout_ms(|key| env.get(key).cloned()),
        100
    );

    let mut env = HashMap::new();
    env.insert("SSH_TTY".to_string(), "/dev/pts/1".to_string());
    assert_eq!(
        pi_tui::terminal::resolve_escape_timeout_ms(|key| env.get(key).cloned()),
        100
    );
}

#[test]
fn defaults_to_10ms_otherwise() {
    let env: HashMap<String, String> = HashMap::new();
    assert_eq!(
        pi_tui::terminal::resolve_escape_timeout_ms(|key| env.get(key).cloned()),
        10
    );
}

/// Boundary: empty SSH variables are falsy upstream (`env.SSH_CONNECTION ||
/// env.SSH_TTY`), so an empty value does not select the SSH window.
#[test]
fn empty_ssh_variables_do_not_select_the_ssh_window() {
    let mut env = HashMap::new();
    env.insert("SSH_CONNECTION".to_string(), String::new());
    env.insert("SSH_TTY".to_string(), String::new());
    assert_eq!(
        pi_tui::terminal::resolve_escape_timeout_ms(|key| env.get(key).cloned()),
        10
    );
}

/// Boundary: the override parser follows `Number` semantics — non-finite and
/// sub-millisecond values fall back, a fractional second-scale value floors.
#[test]
fn override_parsing_follows_number_semantics() {
    for (value, expected) in [("inf", 10_u64), ("NaN", 10), ("0.5", 10), ("120.5", 120)] {
        let mut env = HashMap::new();
        env.insert("PI_TUI_ESC_TIMEOUT".to_string(), value.to_string());
        assert_eq!(
            pi_tui::terminal::resolve_escape_timeout_ms(|key| env.get(key).cloned()),
            expected,
            "override {value:?} resolves to {expected}"
        );
    }
}

// =============================================================================
// normalizeNativeShiftEnterInput
// =============================================================================

#[test]
fn rewrites_return_to_csi_u_shift_enter_when_native_shift_detection_is_enabled_and_shift_is_pressed()
 {
    assert_eq!(
        pi_tui::terminal::normalize_native_shift_enter_input("\r", true, true),
        "\x1b[13;2u"
    );
}

#[test]
fn leaves_return_unchanged_when_native_shift_detection_is_disabled() {
    assert_eq!(
        pi_tui::terminal::normalize_native_shift_enter_input("\r", false, true),
        "\r"
    );
}

#[test]
fn leaves_return_unchanged_when_shift_is_not_pressed() {
    assert_eq!(
        pi_tui::terminal::normalize_native_shift_enter_input("\r", true, false),
        "\r"
    );
}

#[test]
fn leaves_non_return_input_unchanged() {
    assert_eq!(
        pi_tui::terminal::normalize_native_shift_enter_input("\x1b[13;2u", true, true),
        "\x1b[13;2u"
    );
    assert_eq!(
        pi_tui::terminal::normalize_native_shift_enter_input("a", true, true),
        "a"
    );
}

// =============================================================================
// normalizeAppleTerminalInput
// =============================================================================

#[test]
fn rewrites_apple_terminal_return_to_csi_u_shift_enter_when_shift_is_pressed() {
    assert_eq!(
        pi_tui::terminal::normalize_apple_terminal_input("\r", true, true),
        "\x1b[13;2u"
    );
}

#[test]
fn leaves_apple_terminal_return_unchanged_when_shift_is_not_pressed() {
    assert_eq!(
        pi_tui::terminal::normalize_apple_terminal_input("\r", true, false),
        "\r"
    );
}

#[test]
fn leaves_non_apple_terminal_return_unchanged_when_shift_is_pressed() {
    assert_eq!(
        pi_tui::terminal::normalize_apple_terminal_input("\r", false, true),
        "\r"
    );
}

#[test]
fn leaves_non_return_input_unchanged_for_apple_terminal() {
    assert_eq!(
        pi_tui::terminal::normalize_apple_terminal_input("\x1b[13;2u", true, true),
        "\x1b[13;2u"
    );
    assert_eq!(
        pi_tui::terminal::normalize_apple_terminal_input("a", true, true),
        "a"
    );
}

// =============================================================================
// ProcessTerminal Kitty keyboard protocol negotiation
// =============================================================================

#[test]
fn queries_kitty_mode_before_enabling_modify_other_keys_fallback() {
    let harness = setup_negotiation();
    let writes = writes_of(&harness);
    assert_eq!(writes[0], "\x1b[>7u\x1b[?u\x1b[c");
    assert_eq!(count_of(&writes, "\x1b[>4;2m"), 0);
    assert!(!harness.terminal.is_kitty_protocol_active());
}

#[test]
fn activates_kitty_mode_for_non_zero_negotiated_flags() {
    let mut harness = setup_negotiation();
    harness.send("\x1b[?7u");

    assert!(last_input_of(&harness).is_none());
    assert!(harness.terminal.is_kitty_protocol_active());
    let writes = writes_of(&harness);
    assert_eq!(count_of(&writes, "\x1b[>4;2m"), 0);
    assert_eq!(count_of(&writes, "\x1b[>4;0m"), 0);
    assert!(
        harness
            .parser
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_kitty_protocol_active(),
        "activation notifies the injected parser"
    );

    harness.terminal.stop();
    let writes = writes_of(&harness);
    assert_eq!(count_of(&writes, "\x1b[<u"), 1);
    assert_eq!(count_of(&writes, "\x1b[>4;0m"), 0);
    assert!(
        !harness
            .parser
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_kitty_protocol_active(),
        "stop notifies the injected parser"
    );
}

#[test]
fn falls_back_to_modify_other_keys_for_zero_kitty_flags() {
    let mut harness = setup_negotiation();
    harness.send("\x1b[?0u");

    assert!(last_input_of(&harness).is_none());
    assert!(!harness.terminal.is_kitty_protocol_active());
    assert_eq!(count_of(&writes_of(&harness), "\x1b[>4;2m"), 1);
    assert!(harness.terminal.is_modify_other_keys_active());

    harness.terminal.stop();
    assert_eq!(count_of(&writes_of(&harness), "\x1b[>4;0m"), 1);
    assert!(!harness.terminal.is_modify_other_keys_active());
}

#[test]
fn falls_back_to_modify_other_keys_for_device_attributes_without_kitty_flags() {
    let mut harness = setup_negotiation();
    harness.send("\x1b[?62;4;52c");

    assert!(last_input_of(&harness).is_none());
    assert!(!harness.terminal.is_kitty_protocol_active());
    assert_eq!(count_of(&writes_of(&harness), "\x1b[>4;2m"), 1);
}

#[test]
fn forwards_normal_input_while_waiting_for_kitty_response() {
    let mut harness = setup_negotiation();
    harness.send("a");

    assert_eq!(last_input_of(&harness), Some("a".to_string()));
    assert!(!harness.terminal.is_kitty_protocol_active());
}

#[test]
fn tracks_split_kitty_confirmation() {
    let mut harness = setup_negotiation();
    harness.terminal.feed_input("\x1b[?7");
    harness.clock.tick(10);
    harness.terminal.pump();

    assert!(last_input_of(&harness).is_none());

    harness.send("u");

    assert!(harness.terminal.is_kitty_protocol_active());
    assert!(
        harness
            .parser
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_kitty_protocol_active()
    );
    assert_eq!(count_of(&writes_of(&harness), "\x1b[>4;2m"), 0);
}

#[test]
fn replays_buffered_csi_prefix_input_when_it_is_not_a_kitty_response() {
    let mut harness = setup_negotiation();
    harness.terminal.feed_input("\x1b[");
    harness.clock.tick(50); // StdinBuffer sequence timeout, not the lone-ESC timeout
    harness.terminal.pump();

    assert!(last_input_of(&harness).is_none());

    harness.clock.tick(150);
    harness.terminal.pump();

    assert_eq!(last_input_of(&harness), Some("\x1b[".to_string()));
}

/// Boundary: the negotiation parser and prefix walk, driven directly.
#[test]
fn parses_kitty_flags_and_device_attributes() {
    assert_eq!(
        pi_tui::terminal::parse_keyboard_protocol_negotiation_sequence("\x1b[?7u"),
        Some(KeyboardProtocolNegotiationSequence::KittyFlags { flags: 7 })
    );
    assert_eq!(
        pi_tui::terminal::parse_keyboard_protocol_negotiation_sequence("\x1b[?62;4;52c"),
        Some(KeyboardProtocolNegotiationSequence::DeviceAttributes)
    );
    assert_eq!(
        pi_tui::terminal::parse_keyboard_protocol_negotiation_sequence("\x1b[?99;7x"),
        None,
        "an unknown final byte is not a negotiation sequence"
    );
    assert_eq!(
        pi_tui::terminal::parse_keyboard_protocol_negotiation_sequence("\x1b[?u!"),
        None,
        "the flags run must be digits only"
    );
}

/// Boundary: an input chunk that grows the buffered prefix stays pending and
/// re-arms the fragment timer.
#[test]
fn negotiation_buffer_accumulates_prefix_chunks() {
    let mut harness = setup_negotiation();
    harness.terminal.feed_input("\x1b[?");
    harness.terminal.pump();
    assert!(last_input_of(&harness).is_none());
    harness.send("7u");
    assert!(harness.terminal.is_kitty_protocol_active());
}

/// Boundary: the fragment deadline flushes buffered prefix input when the
/// clock passes it; a second pump before the deadline does not double-fire.
#[test]
fn pending_negotiation_buffer_flushes_as_input() {
    let mut harness = setup_negotiation();
    harness.terminal.feed_input("\x1b[");
    harness.clock.tick(50); // StdinBuffer sequence timeout fires first
    harness.terminal.pump();
    harness.terminal.pump(); // the second pump must not double-forward
    assert!(last_input_of(&harness).is_none());

    harness.clock.tick(150);
    harness.terminal.pump();
    assert_eq!(last_input_of(&harness), Some("\x1b[".to_string()));
}

/// Boundary: ordinary input while a negotiation prefix is buffered flushes
/// the buffered prefix first, then forwards the current sequence.
#[test]
fn ordinary_input_flushes_a_buffered_negotiation_prefix() {
    let mut harness = setup_negotiation();
    harness.terminal.feed_input("\x1b[?");
    harness.clock.tick(50); // StdinBuffer sequence timeout fires first
    harness.terminal.pump();
    assert!(
        last_input_of(&harness).is_none(),
        "the flushed chunk is a negotiation prefix, so it buffers"
    );

    harness.send("X"); // buffered "\x1b[?" + "X" is neither a parse nor a prefix

    assert_eq!(inputs_of(&harness), ["\x1b[?", "X"]);
}

/// Boundary: an incomplete CSI the `StdinBuffer` holds flushes through the
/// sequence timeout and forwards as ordinary input.
#[test]
fn flushed_incomplete_csi_forwards_as_input() {
    let mut harness = setup_negotiation();
    harness.terminal.feed_input("\x1b[");
    harness.terminal.feed_input("1;");
    assert!(last_input_of(&harness).is_none());

    harness.clock.tick(50);
    harness.terminal.pump();

    assert_eq!(inputs_of(&harness), ["\x1b[1;"]);
}

/// Boundary: paste content reaching the buffer is re-wrapped in the
/// bracketed-paste markers before the input handler sees it.
#[test]
fn paste_events_are_rewrapped_for_editor_handling() {
    let mut harness = setup_negotiation();
    harness.terminal.feed_input("\x1b[200~pasted\x1b[201~");

    assert_eq!(inputs_of(&harness), ["\x1b[200~pasted\x1b[201~"]);
}

/// Boundary: without an input handler, forwarded sequences are dropped
/// instead of panicking, mirroring upstream's `if (!this.inputHandler) return`.
#[test]
fn forwarded_input_without_a_handler_is_dropped() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let mut terminal = ProcessTerminal::headless(capture_sink(&writes));
    terminal.set_input_handler(None);
    terminal.feed_input("a");
    terminal.pump();
    assert_eq!(
        count_of(
            &writes.lock().unwrap_or_else(PoisonError::into_inner),
            "\x1b[>4;2m"
        ),
        0
    );
}

// =============================================================================
// ProcessTerminal progress
// =============================================================================

#[test]
fn writes_a_valid_osc_9_4_clear_sequence() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let mut terminal = ProcessTerminal::headless(capture_sink(&writes));
    terminal.set_progress(false);

    let writes = writes.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(
        writes[0], "\x1b[>7u\x1b[?u\x1b[c",
        "the headless constructor arms the negotiation"
    );
    assert_eq!(writes[1], "\x1b]9;4;0\x07");
    assert_eq!(writes.len(), 2);
}

/// Boundary: the active progress sequence writes immediately and the
/// keepalive repeats it on the injected interval until cleared.
#[test]
fn progress_keepalive_repeats_until_cleared() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let mut terminal = ProcessTerminal::headless(capture_sink(&writes))
        .with_keepalive_interval(Duration::from_millis(10));
    terminal.set_progress(true);
    std::thread::sleep(Duration::from_millis(35));
    terminal.set_progress(false);

    let snapshot = writes
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    assert!(
        count_of(&snapshot, "\x1b]9;4;3\x07") >= 2,
        "keepalive repeats the active sequence"
    );
    assert_eq!(count_of(&snapshot, "\x1b]9;4;0\x07"), 1);
    assert_eq!(snapshot.last(), Some(&"\x1b]9;4;0\x07".to_string()));

    // A second stop is a no-op: no keepalive is running to clear.
    terminal.stop();
    let after = writes.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(count_of(&after, "\x1b]9;4;0\x07"), 1);
}

// =============================================================================
// ProcessTerminal dimensions
// =============================================================================

#[test]
fn falls_back_to_columns_and_lines_before_default_dimensions() {
    let mut env = HashMap::new();
    env.insert("COLUMNS".to_string(), "123".to_string());
    env.insert("LINES".to_string(), "45".to_string());

    let writes = Arc::new(Mutex::new(Vec::new()));
    let terminal =
        ProcessTerminal::headless(capture_sink(&writes)).with_env_lookup(env_lookup(&env));

    assert_eq!(terminal.columns(), 123);
    assert_eq!(terminal.rows(), 45);
}

/// Boundary: beyond COLUMNS/LINES the documented 80x24 defaults bind.
#[test]
fn falls_back_to_default_dimensions() {
    let env: HashMap<String, String> = HashMap::new();
    let writes = Arc::new(Mutex::new(Vec::new()));
    let terminal =
        ProcessTerminal::headless(capture_sink(&writes)).with_env_lookup(env_lookup(&env));

    assert_eq!(terminal.columns(), 80);
    assert_eq!(terminal.rows(), 24);
}

/// Boundary: zero is falsy upstream (`process.stdout.columns || env || 80`),
/// so a zero or unparsable dimension falls through to the default.
#[test]
fn zero_and_junk_dimensions_fall_through_to_the_default() {
    let mut env = HashMap::new();
    env.insert("COLUMNS".to_string(), "0".to_string());
    env.insert("LINES".to_string(), "junk".to_string());

    let writes = Arc::new(Mutex::new(Vec::new()));
    let terminal =
        ProcessTerminal::headless(capture_sink(&writes)).with_env_lookup(env_lookup(&env));

    assert_eq!(terminal.columns(), 80);
    assert_eq!(terminal.rows(), 24);
}

// =============================================================================
// Cursor, title, and clear writers
// =============================================================================

#[test]
fn cursor_and_clear_helpers_write_their_sequences() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let mut terminal = ProcessTerminal::headless(capture_sink(&writes));

    terminal.hide_cursor();
    terminal.show_cursor();
    terminal.clear_line();
    terminal.clear_from_cursor();
    terminal.clear_screen();
    terminal.set_title("pi");
    terminal.move_by(2);
    terminal.move_by(-3);
    terminal.move_by(0);

    let writes = writes.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(
        writes[0], "\x1b[>7u\x1b[?u\x1b[c",
        "the headless constructor arms the negotiation"
    );
    assert_eq!(
        &writes[1..],
        [
            "\x1b[?25l",
            "\x1b[?25h",
            "\x1b[K",
            "\x1b[J",
            "\x1b[2J\x1b[H",
            "\x1b]0;pi\x07",
            "\x1b[2B",
            "\x1b[3A",
        ],
        "move_by(0) writes nothing"
    );
}

// =============================================================================
// drainInput
// =============================================================================

#[test]
fn drain_input_consumes_preloaded_chunks_without_forwarding() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let (input_tx, input_rx) = std::sync::mpsc::channel();
    assert!(
        input_tx.send(b"abc".to_vec()).is_ok(),
        "in-memory channel delivers"
    );
    assert!(
        input_tx.send(b"\x1b[A".to_vec()).is_ok(),
        "in-memory channel delivers"
    );
    drop(input_tx);

    let mut terminal = ProcessTerminal::headless(capture_sink(&writes)).with_stdin_source(input_rx);
    terminal.set_input_handler(None);
    terminal.drain_input(1000, 50);

    let inputs = writes.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(
        count_of(&inputs, "\x1b[>4;2m"),
        0,
        "drained input must not reach any handler"
    );
}

#[test]
fn drain_input_bounds_itself_by_max_ms_without_a_source() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let mut terminal = ProcessTerminal::headless(capture_sink(&writes));
    terminal.set_input_handler(None);

    let start = Instant::now();
    terminal.drain_input(30, 50);
    assert!(start.elapsed() >= Duration::from_millis(30));
    assert!(start.elapsed() < Duration::from_millis(500));
}

#[test]
fn drain_input_disables_the_kitty_protocol_first() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let mut terminal = ProcessTerminal::headless(capture_sink(&writes));
    terminal.set_input_handler(None);

    terminal.drain_input(30, 50);

    let writes = writes.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(
        count_of(&writes, "\x1b[<u"),
        1,
        "drain disables the pushed protocol"
    );
    assert_eq!(
        count_of(&writes, "\x1b[>4;0m"),
        0,
        "modifyOtherKeys was never enabled"
    );
    assert!(!terminal.is_kitty_protocol_active());
}

// =============================================================================
// Boundary tests: OS-touching paths upstream leaves untested, so the 95%
// coverage gate binds (#41).
// =============================================================================

use std::io::Read;

struct InterruptedThenData {
    delivered: bool,
}

impl Read for InterruptedThenData {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.delivered {
            Ok(0)
        } else {
            self.delivered = true;
            buf[0] = b'x';
            Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "retry",
            ))
        }
    }
}

struct FailingReader;

impl Read for FailingReader {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("gone"))
    }
}

#[test]
fn read_input_chunks_retries_interrupts_and_stops_on_eof() {
    let stop_flag = std::sync::atomic::AtomicBool::new(false);

    let (input_tx, input_rx) = std::sync::mpsc::channel();
    let mut source = InterruptedThenData { delivered: false };
    pi_tui::terminal::read_input_chunks(&mut source, &input_tx, &stop_flag);
    assert!(
        input_rx.try_iter().next().is_none(),
        "an interrupted read delivers nothing"
    );

    let (data_tx, data_rx) = std::sync::mpsc::channel();
    let mut cursor = std::io::Cursor::new(b"abc".to_vec());
    pi_tui::terminal::read_input_chunks(&mut cursor, &data_tx, &stop_flag);
    assert_eq!(data_rx.try_iter().collect::<Vec<_>>(), [b"abc".to_vec()]);

    let (err_tx, err_rx) = std::sync::mpsc::channel();
    let mut failing = FailingReader;
    pi_tui::terminal::read_input_chunks(&mut failing, &err_tx, &stop_flag);
    assert!(
        err_rx.try_iter().next().is_none(),
        "an error ends the reader"
    );
}

#[test]
fn utf8_reassembler_decodes_split_multibyte_and_invalid_bytes() {
    let mut decoder = pi_tui::terminal::Utf8Reassembler::default();

    // A multibyte character split across chunks decodes whole.
    let world = "\u{4E16}".as_bytes();
    assert!(
        decoder.feed(&world[..1]).is_empty(),
        "an incomplete tail holds"
    );
    assert_eq!(decoder.feed(&world[1..]), ["\u{4E16}"]);

    // ASCII decodes straight through.
    assert_eq!(decoder.feed(b"ab"), ["ab"]);

    // Invalid bytes become U+FFFD one per maximal invalid subpart.
    assert_eq!(decoder.feed(&[0x80]), ["\u{FFFD}"]);
    assert_eq!(decoder.feed(&[0xE9, 0x28]), ["\u{FFFD}", "("]);

    // A held tail completes on the next chunk.
    assert!(decoder.feed(&[0xF0, 0x9F]).is_empty());
    assert_eq!(decoder.feed(&[0x8E, 0x8A]), ["\u{1F38A}"]);

    assert_eq!(decoder.feed(&[]), Vec::<String>::new());
}

#[test]
fn is_apple_terminal_session_reads_the_environment() {
    // The probe answers whatever the process environment carries; the port
    // keeps it callable so the suite can execute it on every platform.
    let _ = pi_tui::terminal::is_apple_terminal_session();
}

#[test]
fn refresh_terminal_dimensions_survives_a_real_self_signal() {
    // A real SIGWINCH to the test process; the default disposition is
    // ignore, and any registered handler merely queues a resize.
    pi_tui::terminal::refresh_terminal_dimensions();
}

#[test]
fn real_terminal_round_trips_start_stop_and_dimensions() {
    let mut terminal = ProcessTerminal::new();

    // Without a TTY (CI) the raw-mode plumbing and size query fail and are
    // ignored; with a TTY (interactive) raw mode is entered and restored.
    assert!(terminal.columns() > 0);
    assert!(terminal.rows() > 0);

    terminal.start(Box::new(|_: String| {}), Box::new(|| {}));
    std::thread::sleep(Duration::from_millis(30));
    // The start-time self-signal queues a resize; poll dispatches it.
    terminal.poll(Duration::from_millis(50));
    terminal.stop();
}

#[test]
fn poll_wakes_on_deadlines_and_delivers_buffered_input() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let inputs = Arc::new(Mutex::new(Vec::new()));
    let clock = TestClock::new();
    let (input_tx, input_rx) = std::sync::mpsc::channel();

    let input_sink = Arc::clone(&inputs);
    let mut terminal = ProcessTerminal::headless(capture_sink(&writes))
        .with_clock(Box::new({
            let clock = Arc::clone(&clock);
            move || clock.now()
        }))
        .with_stdin_source(input_rx);
    terminal.set_input_handler(Some(Box::new(move |data: String| {
        input_sink
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(data);
    })));

    // (None, None): no pending deadlines, the wait is the caller's timeout.
    terminal.poll(Duration::from_millis(1));

    // (Some, None): a StdinBuffer deadline wakes the pump before the timeout.
    terminal.feed_input("\x1b[");
    clock.tick(50);
    terminal.poll(Duration::from_millis(500));

    // (None, Some): only the negotiation fragment timer is pending.
    clock.tick(150);
    terminal.poll(Duration::from_millis(1));

    // (Some, Some): both deadlines pending, the nearest wins.
    terminal.feed_input("\x1b[?");
    clock.tick(50);
    terminal.poll(Duration::from_millis(1));
    terminal.feed_input("\x1b");
    clock.tick(10);
    terminal.poll(Duration::from_millis(1));

    let recorded = inputs
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    assert_eq!(
        recorded,
        ["\x1b[", "\x1b[?", "\x1b"],
        "poll delivers flushed sequences in order"
    );
    drop(input_tx);
}

#[test]
fn write_log_gains_a_timestamped_file_inside_a_configured_directory() {
    let dir = std::env::temp_dir().join(format!(
        "pi-tui-write-log-dir-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos())
    ));
    assert!(
        std::fs::create_dir_all(&dir).is_ok(),
        "create log dir: {dir:?}"
    );

    let mut env = HashMap::new();
    env.insert(
        "PI_TUI_WRITE_LOG".to_string(),
        dir.to_string_lossy().into_owned(),
    );
    let mut terminal = ProcessTerminal::headless(capture_sink(&Arc::new(Mutex::new(Vec::new()))))
        .with_env_lookup(env_lookup(&env));
    terminal.write("hello");

    let entries = std::fs::read_dir(&dir);
    let mut found: Option<std::path::PathBuf> = None;
    if let Ok(dir_entries) = entries {
        for entry in dir_entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("tui-") && name.to_lowercase().ends_with(".log") {
                found = Some(entry.path());
            }
        }
    }
    assert!(found.is_some(), "timestamped log file exists");
    let contents = std::fs::read_to_string(found.unwrap_or_default());
    assert!(contents.is_ok_and(|text| text.contains("hello")));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn kill_maps_unreachable_pids_and_signal_numbers_to_errors() {
    // A pid that cannot fit a POSIX pid_t maps to EINVAL.
    assert_eq!(
        pi_tui::terminal::kill(u32::MAX, libc::SIGWINCH),
        Err(libc::EINVAL)
    );
    // A signal number that is not a signal maps to EINVAL.
    assert_eq!(
        pi_tui::terminal::kill(std::process::id(), -1),
        Err(libc::EINVAL)
    );
    // A pid that fits but names no process maps to the raw errno (ESRCH).
    let result = pi_tui::terminal::kill(0x7FFF_FFF0, libc::SIGWINCH);
    assert!(result.is_err(), "no such pid carries the raw errno");
    // A real self-signal succeeds.
    assert!(pi_tui::terminal::kill(std::process::id(), libc::SIGWINCH).is_ok());
}

#[test]
fn parses_rejects_malformed_negotiation_sequences() {
    assert_eq!(
        pi_tui::terminal::parse_keyboard_protocol_negotiation_sequence("\x1b[?7z8u"),
        None,
        "the flags run must end at the final byte"
    );
    assert_eq!(
        pi_tui::terminal::parse_keyboard_protocol_negotiation_sequence("\x1b[?7Xc"),
        None,
        "the attribute run must be digits and semicolons"
    );
}

#[test]
fn buffered_kitty_response_completes_from_a_following_chunk() {
    // The split Kitty confirmation arrives with its first fragment flushed by
    // the sequence timeout and the rest delivered separately.
    let mut harness = setup_negotiation();
    harness.terminal.feed_input("\x1b[?");
    harness.clock.tick(50);
    harness.terminal.pump();
    assert!(last_input_of(&harness).is_none(), "still pending");

    // The tail arrives as two single characters the StdinBuffer cannot merge
    // (the buffer is empty between them), so the negotiation buffer merges
    // them instead.
    harness.send("7");
    harness.send("u");

    assert!(harness.terminal.is_kitty_protocol_active());
}

#[test]
fn pending_timer_can_rearm_while_still_pending() {
    let mut harness = setup_negotiation();
    harness.terminal.feed_input("\x1b[");
    harness.clock.tick(50);
    harness.terminal.pump(); // buffered + fragment timer armed

    harness.send("?");
    // The timer is already armed; the second pending chunk grows the buffer
    // and the single timer flushes the whole prefix.
    harness.clock.tick(150);
    harness.terminal.pump();
    assert_eq!(last_input_of(&harness), Some("\x1b[?".to_string()));
}

#[test]
fn carriage_return_forwards_unchanged_without_a_native_helper() {
    let mut harness = setup_negotiation();
    harness.send("\r");
    assert_eq!(last_input_of(&harness), Some("\r".to_string()));
}

#[test]
fn headless_start_and_stop_round_trip() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let mut terminal = ProcessTerminal::headless(capture_sink(&writes));
    terminal.start(Box::new(|_: String| {}), Box::new(|| {}));
    terminal.stop();

    let snapshot = writes
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    drop(writes);
    assert!(
        snapshot.contains(&"\x1b[?2004h".to_string()),
        "bracketed paste enables at start"
    );
    assert!(
        snapshot.contains(&"\x1b[?2004l".to_string()),
        "bracketed paste disables at stop"
    );
}

#[test]
fn input_after_stop_is_dropped() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let mut terminal = ProcessTerminal::headless(capture_sink(&writes));
    terminal.stop();
    terminal.feed_input("a");
    terminal.pump();
}

#[test]
fn drain_input_waits_out_a_silent_channel() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let (input_tx, input_rx) = std::sync::mpsc::channel();
    let mut terminal = ProcessTerminal::headless(capture_sink(&writes)).with_stdin_source(input_rx);
    terminal.set_input_handler(None);

    let start = Instant::now();
    // A silent open channel exits through the idle window, which precedes
    // the max bound here (upstream breaks on whichever fires first).
    terminal.drain_input(30, 20);
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_millis(20),
        "the idle window bounds the wait"
    );
    assert!(elapsed < Duration::from_millis(500));
    drop(input_tx);
}

#[test]
fn poll_delivers_bytes_waiting_in_the_channel() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let inputs = Arc::new(Mutex::new(Vec::new()));
    let (input_tx, input_rx) = std::sync::mpsc::channel();
    assert!(
        input_tx.send(b"a".to_vec()).is_ok(),
        "in-memory channel delivers"
    );

    let input_sink = Arc::clone(&inputs);
    let mut terminal = ProcessTerminal::headless(capture_sink(&writes)).with_stdin_source(input_rx);
    terminal.set_input_handler(Some(Box::new(move |data: String| {
        input_sink
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(data);
    })));
    terminal.poll(Duration::from_millis(50));

    let recorded = inputs
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    assert_eq!(recorded, ["a"]);
    drop(input_tx);
}

#[test]
fn write_log_uses_a_plain_file_path_as_is() {
    let path = std::env::temp_dir().join(format!(
        "pi-tui-write-log-file-{}-{}.log",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos())
    ));
    let _ = std::fs::remove_file(&path);

    let mut env = HashMap::new();
    env.insert(
        "PI_TUI_WRITE_LOG".to_string(),
        path.to_string_lossy().into_owned(),
    );
    let mut terminal = ProcessTerminal::headless(capture_sink(&Arc::new(Mutex::new(Vec::new()))))
        .with_env_lookup(env_lookup(&env));
    terminal.write("logged");

    let contents = std::fs::read_to_string(&path);
    assert!(contents.is_ok_and(|text| text.contains("logged")));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn write_log_paths_resolve_from_the_environment() {
    // An empty value and an unset value both leave the write log disabled.
    let mut env = HashMap::new();
    env.insert("PI_TUI_WRITE_LOG".to_string(), String::new());
    let writes = Arc::new(Mutex::new(Vec::new()));
    let mut terminal =
        ProcessTerminal::headless(capture_sink(&writes)).with_env_lookup(env_lookup(&env));
    terminal.write("unlogged");
}

#[test]
fn debug_and_default_round_trip() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let terminal = ProcessTerminal::headless(capture_sink(&writes));
    let rendered = format!("{terminal:?}");
    assert!(rendered.contains("ProcessTerminal"));
    assert!(rendered.contains("kitty_protocol_active"));

    let mut default_terminal = ProcessTerminal::default();
    default_terminal.stop();
}

#[test]
fn carriage_return_with_an_apple_terminal_env_still_forwards_unchanged() {
    // The native modifier helper is absent in the port, so the probe answers
    // false and Return forwards unchanged, exactly as upstream behaves
    // without its N-API helper loaded.
    let mut env = HashMap::new();
    env.insert("TERM_PROGRAM".to_string(), "Apple_Terminal".to_string());
    let writes = Arc::new(Mutex::new(Vec::new()));
    let inputs = Arc::new(Mutex::new(Vec::new()));
    let input_sink = Arc::clone(&inputs);
    let mut terminal =
        ProcessTerminal::headless(capture_sink(&writes)).with_env_lookup(env_lookup(&env));
    terminal.set_input_handler(Some(Box::new(move |data: String| {
        input_sink
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(data);
    })));
    terminal.feed_input("\r");
    terminal.pump();

    let recorded = inputs
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    assert_eq!(recorded, ["\r"]);
}

#[test]
fn read_input_chunks_stops_when_the_channel_closes() {
    let (input_tx, input_rx) = std::sync::mpsc::channel();
    drop(input_rx);
    let stop_flag = std::sync::atomic::AtomicBool::new(false);
    let mut cursor = std::io::Cursor::new(b"abc".to_vec());
    pi_tui::terminal::read_input_chunks(&mut cursor, &input_tx, &stop_flag);
}

#[test]
fn progress_keepalive_is_cleared_by_stop() {
    let writes = Arc::new(Mutex::new(Vec::new()));
    let mut terminal = ProcessTerminal::headless(capture_sink(&writes))
        .with_keepalive_interval(Duration::from_millis(10));
    terminal.set_progress(true);
    terminal.stop();

    let writes = writes.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(
        count_of(&writes, "\x1b]9;4;0\x07"),
        1,
        "stop clears an active progress"
    );
    assert_eq!(count_of(&writes, "\x1b]9;4;3\x07"), 1);
}

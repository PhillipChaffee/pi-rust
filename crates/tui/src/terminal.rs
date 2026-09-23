//! The `Terminal` interface and the real terminal, ported from
//! `packages/tui/src/terminal.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#41).
//!
//! Raw mode, size, and terminal state ride crossterm as thin plumbing
//! ([ADR 0001](https://github.com/PhillipChaffee/pi-rust/blob/main/docs/adr/0001-tui-renderer-port-not-ratatui.md)):
//! every visible byte still flows through [`Terminal::write`], and the
//! hand-rolled input pipeline — [`StdinBuffer`] reassembly plus the Kitty
//! keyboard-protocol negotiation — stays 1:1, fed by a raw byte reader that
//! replaces the `process.stdin` byte feed behind the same [`Terminal`] seam.
//!
//! Restatements against upstream, forced by Rust having no implicit event
//! loop:
//!
//! - Upstream pushed input through callbacks from the event loop. The port
//!   keeps the callback shape but the owner pumps it: [`Terminal::poll`]
//!   waits for stdin chunks or a pending deadline, then invokes the stored
//!   input and resize handlers. [`ProcessTerminal::feed_input`] and
//!   [`ProcessTerminal::pump`] are the pumps [`Terminal::poll`] drives,
//!   public so headless tests can drive the pipeline without a TTY.
//! - Upstream resolved `process.env` per call and monkeypatched
//!   `process.stdout.write`/`process.stdin.on`/`process.env` in tests. The
//!   port takes an environment lookup, a write sink, and a clock at
//!   construction; the real terminal wires stdout, the process environment,
//!   and `Instant::now`.
//! - `drainInput` awaits a promise; the port drains synchronously.
//! - Upstream restored the previous raw-mode flag on `stop`; crossterm has no
//!   raw-mode query, so the port restores raw-off, which matches pi's
//!   single-session usage.
//! - The Windows VT-input probe (`enableWindowsVTInput`) and the native
//!   Shift+Enter probe (`isNativeModifierPressed` over the N-API platform
//!   helper) are out of scope with Windows ([#1](https://github.com/PhillipChaffee/pi-rust/issues/1)):
//!   the shift probe answers false, exactly as upstream behaves without its
//!   native helper loaded.
//! - Upstream's `process.stdin.pause()` on stop has no counterpart: the
//!   reader thread stops consuming on the stop flag and the channel drops,
//!   so buffered input is never re-interpreted after raw mode is disabled.
//! - Upstream cached dimensions in `process.stdout` between SIGWINCH
//!   deliveries; crossterm's size query is a fresh ioctl per call, so the
//!   start-time self-signal survives only to drive the resize notification.
//! - Upstream's `Buffer` argument to `StdinBuffer.process` never occurs: the
//!   byte feed decodes UTF-8 incrementally exactly as
//!   `process.stdin.setEncoding("utf8")` did ([`Utf8Reassembler`]).
//! - `refreshTerminalDimensions` exists because Node caches dimensions until
//!   SIGWINCH; crossterm's size query is a fresh ioctl per call, so the
//!   self-signal survives only to drive the resize notification at start.

use std::io::{ErrorKind, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::keys::KeyParser;
use crate::stdin_buffer::{StdinBuffer, StdinBufferEvent};

const TERMINAL_PROGRESS_KEEPALIVE_MS: u64 = 1000;
const TERMINAL_PROGRESS_ACTIVE_SEQUENCE: &str = "\x1b]9;4;3\x07";
const TERMINAL_PROGRESS_CLEAR_SEQUENCE: &str = "\x1b]9;4;0\x07";
const NATIVE_SHIFT_ENTER_SEQUENCE: &str = "\x1b[13;2u";
const DESIRED_KITTY_KEYBOARD_PROTOCOL_FLAGS: u32 = 7;
const KEYBOARD_PROTOCOL_RESPONSE_FRAGMENT_TIMEOUT_MS: u64 = 150;
static KITTY_KEYBOARD_PROTOCOL_QUERY: LazyLock<String> =
    LazyLock::new(|| format!("\x1b[>{DESIRED_KITTY_KEYBOARD_PROTOCOL_FLAGS}u\x1b[?u\x1b[c"));
const DEFAULT_ESCAPE_TIMEOUT_MS: u64 = 10;
const DEFAULT_SSH_ESCAPE_TIMEOUT_MS: u64 = 100;
const DEFAULT_COLUMNS: u16 = 80;
const DEFAULT_ROWS: u16 = 24;
const READ_CHUNK_BYTES: usize = 4096;

/// The environment lookup upstream resolved through `process.env`.
///
/// Shared by the terminal and the renderers: Rust cannot mutate the process
/// environment without the `unsafe` this workspace forbids, so tests inject a
/// map-backed lookup and sessions default to the real environment.
pub type EnvLookup = Box<dyn Fn(&str) -> Option<String>>;

/// The process environment, upstream's `process.env` default.
#[must_use]
pub fn default_env_lookup() -> EnvLookup {
    Box::new(|key| std::env::var(key).ok())
}

/// A complete keyboard-protocol response, restating upstream's
/// `KeyboardProtocolNegotiationSequence`: Kitty's `CSI ? <flags> u` report or
/// the device-attributes answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyboardProtocolNegotiationSequence {
    /// `CSI ? <flags> u` — the negotiated Kitty keyboard-protocol flags.
    KittyFlags {
        /// The flags the terminal reports.
        flags: u32,
    },
    /// `CSI ? <attrs> c` — device attributes, the sentinel terminals that do
    /// not know Kitty keyboard protocol answer the trailing query with.
    DeviceAttributes,
}

/// Restates upstream's `parseKeyboardProtocolNegotiationSequence`:
/// `CSI ? <digits> u` for Kitty flags, `CSI ? <digits/semicolons> c` for
/// device attributes.
#[must_use]
pub fn parse_keyboard_protocol_negotiation_sequence(
    sequence: &str,
) -> Option<KeyboardProtocolNegotiationSequence> {
    let bytes = sequence.as_bytes();
    if !bytes.starts_with(b"\x1b[?") {
        return None;
    }
    match bytes.last() {
        Some(&b'u') => {
            let mut pos = 3;
            let flags = crate::keys::scan_digits(bytes, &mut pos)
                .and_then(|flags| u32::try_from(flags).ok());
            if pos == bytes.len() - 1
                && let Some(flags) = flags
            {
                return Some(KeyboardProtocolNegotiationSequence::KittyFlags { flags });
            }
            None
        }
        Some(&b'c') => {
            if bytes[3..bytes.len() - 1]
                .iter()
                .all(|byte| byte.is_ascii_digit() || *byte == b';')
            {
                Some(KeyboardProtocolNegotiationSequence::DeviceAttributes)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Whether a partial response could still grow into a negotiation sequence.
fn is_keyboard_protocol_negotiation_sequence_prefix(sequence: &str) -> bool {
    let bytes = sequence.as_bytes();
    bytes == b"\x1b["
        || (bytes.starts_with(b"\x1b[?")
            && bytes[3..]
                .iter()
                .all(|byte| byte.is_ascii_digit() || *byte == b';'))
}

/// Restates upstream's `isAppleTerminalSession` environment probe: Apple
/// Terminal identifies itself through `TERM_PROGRAM` on darwin.
#[must_use]
pub fn is_apple_terminal_session() -> bool {
    // The probe reads the process environment on every platform, so the
    // function is genuinely effectful everywhere and stays non-const; only
    // the comparison is darwin's.
    let term_program = std::env::var("TERM_PROGRAM").ok();
    #[cfg(target_os = "macos")]
    let apple_terminal = term_program.as_deref() == Some("Apple_Terminal");
    #[cfg(not(target_os = "macos"))]
    let apple_terminal = {
        let _ = &term_program;
        false
    };
    apple_terminal
}

/// Refresh terminal dimensions on POSIX platforms by sending SIGWINCH to
/// this process, best-effort: some environments return EACCES for `kill(2)`
/// and the refresh is skipped rather than crashing.
pub fn refresh_terminal_dimensions() {
    #[cfg(windows)]
    {
        return;
    }
    #[cfg(unix)]
    refresh_terminal_dimensions_with(&kill);
}

/// The `kill(2)`-shaped call [`refresh_terminal_dimensions`] makes: pid,
/// signal, and an error when the signal cannot be delivered.
pub type KillFn<'a> = &'a dyn Fn(u32, i32) -> Result<(), i32>;

/// Best-effort self-signal with the kill call injected, the seam the
/// regression suite drives.
///
/// Every error is ignored: the refresh is best-effort and EACCES (restricted
/// seccomp or LSM policies) and EPERM alike must not crash the caller.
pub fn refresh_terminal_dimensions_with(kill: KillFn<'_>) {
    #[cfg(windows)]
    {
        let _ = kill;
        return;
    }
    #[cfg(unix)]
    {
        let _ = kill(std::process::id(), libc::SIGWINCH);
    }
}

/// Sends a POSIX signal to a process, restating upstream's
/// `process.kill(pid, signal)`. Public so the suite can drive the error
/// mappings with unreachable pids and signal numbers.
///
/// # Errors
///
/// Returns the raw errno when the pid does not fit a POSIX `pid_t`, the
/// signal number is not a signal, or the kernel refuses the delivery —
/// the same errno surface `kill(2)` reports.
#[cfg(unix)]
pub fn kill(pid: u32, signal: i32) -> Result<(), i32> {
    let pid = nix::unistd::Pid::from_raw(i32::try_from(pid).map_err(|_| libc::EINVAL)?);
    let signal = nix::sys::signal::Signal::try_from(signal).map_err(|_| libc::EINVAL)?;
    nix::sys::signal::kill(pid, signal).map_err(|errno| {
        std::io::Error::from(errno)
            .raw_os_error()
            .unwrap_or(libc::EIO)
    })
}

/// Restates upstream's `normalizeNativeShiftEnterInput`.
///
/// On terminals that report Shift through the native helper (Apple Terminal,
/// Windows console), a bare `\r` while Shift is held is rewritten into the
/// CSI-u Shift+Enter sequence the parser recognizes.
#[must_use]
pub fn normalize_native_shift_enter_input(
    data: &str,
    should_detect_native_shift_enter: bool,
    is_shift_pressed: bool,
) -> String {
    if should_detect_native_shift_enter && data == "\r" && is_shift_pressed {
        return NATIVE_SHIFT_ENTER_SEQUENCE.to_string();
    }
    data.to_string()
}

/// Restates upstream's `normalizeAppleTerminalInput`, the Apple Terminal
/// spelling of the native Shift+Enter rewrite.
#[must_use]
pub fn normalize_apple_terminal_input(
    data: &str,
    is_apple_terminal: bool,
    is_shift_pressed: bool,
) -> String {
    normalize_native_shift_enter_input(data, is_apple_terminal, is_shift_pressed)
}

/// Resolve how long to wait for the rest of an escape sequence before
/// dispatching a lone ESC as the Escape key.
///
/// Legacy Alt+key input is ESC plus another byte, so high-latency transports
/// need a longer reassembly window. Reads through `env` (upstream: a
/// `NodeJS.ProcessEnv`, defaulting to `process.env`): `PI_TUI_ESC_TIMEOUT`
/// overrides when it parses to a finite positive number,
/// `SSH_CONNECTION`/`SSH_TTY` select the SSH window, and everything else
/// gets the 10 ms local default. A sub-millisecond override truncates to the
/// default because the port arms whole-millisecond deadlines.
#[must_use]
pub fn resolve_escape_timeout_ms(env: impl Fn(&str) -> Option<String>) -> u64 {
    if let Some(configured) = env("PI_TUI_ESC_TIMEOUT")
        && let Ok(parsed) = configured.parse::<f64>()
        && parsed.is_finite()
        && parsed > 0.0
    {
        let millis = saturating_millis(parsed);
        if millis > 0 {
            return millis;
        }
    }
    if env("SSH_CONNECTION").is_some_and(|value| !value.is_empty())
        || env("SSH_TTY").is_some_and(|value| !value.is_empty())
    {
        return DEFAULT_SSH_ESCAPE_TIMEOUT_MS;
    }
    DEFAULT_ESCAPE_TIMEOUT_MS
}

/// Upstream hands the override to `setTimeout`, which accepts fractional
/// milliseconds; the port arms whole-millisecond deadlines, so the value
/// truncates. Float-to-int casts saturate, bounding absurd overrides.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the override is a wall-clock millisecond budget filtered to finite and positive; truncation below 1ms falls back to the default, which upstream reaches the same way"
)]
const fn saturating_millis(seconds: f64) -> u64 {
    seconds as u64
}

/// The callback that receives each forwarded input sequence.
pub type InputHandler = Box<dyn FnMut(String)>;
/// The callback fired when the terminal size may have changed.
pub type ResizeHandler = Box<dyn Fn()>;
/// Where [`Terminal::write`] delivers bytes: the real terminal's stdout, or
/// a capture sink in tests. Shared so the progress keepalive thread writes
/// through the same path.
pub type WriteSink = Arc<Mutex<Box<dyn FnMut(&str) + Send>>>;

/// Minimal terminal interface for TUI.
///
/// Upstream's `Terminal` interface; the seam every backend implements. The
/// one addition is [`Terminal::poll`]: Rust has no implicit event loop, so
/// the owner's loop pumps it and the backend invokes the stored handlers
/// from there.
pub trait Terminal {
    /// Start the terminal with input and resize handlers.
    fn start(&mut self, on_input: InputHandler, on_resize: ResizeHandler);

    /// Stop the terminal and restore state.
    fn stop(&mut self);

    /// Drain stdin before exiting to prevent Kitty key release events from
    /// leaking to the parent shell over slow SSH connections.
    ///
    /// Upstream awaits this; the port drains synchronously. `max_ms` bounds
    /// the drain (upstream default 1000), `idle_ms` exits early when no input
    /// arrives within this time (upstream default 50).
    fn drain_input(&mut self, max_ms: u64, idle_ms: u64);

    /// Write output to terminal.
    fn write(&mut self, data: &str);

    /// Terminal columns.
    fn columns(&self) -> u16;

    /// Terminal rows.
    fn rows(&self) -> u16;

    /// Whether the Kitty keyboard protocol is active.
    fn is_kitty_protocol_active(&self) -> bool;

    /// Move the cursor up (negative) or down (positive) by N lines.
    fn move_by(&mut self, lines: i32);

    /// Hide the cursor.
    fn hide_cursor(&mut self);

    /// Show the cursor.
    fn show_cursor(&mut self);

    /// Clear the current line.
    fn clear_line(&mut self);

    /// Clear from cursor to end of screen.
    fn clear_from_cursor(&mut self);

    /// Clear the entire screen and move the cursor to (0,0).
    fn clear_screen(&mut self);

    /// Set the terminal window title.
    fn set_title(&mut self, title: &str);

    /// Progress indicator (OSC 9;4).
    fn set_progress(&mut self, active: bool);

    /// Pump the terminal: wait up to `timeout` for input, then deliver
    /// whatever arrived and fire due timers into the stored handlers.
    fn poll(&mut self, timeout: Duration);
}

/// Raw-mode plumbing and size, restating the `process.stdin`/`process.stdout`
/// properties upstream read. Crossterm for the real terminal; a no-op for
/// headless tests. Raw mode restores as raw-off — crossterm has no raw-mode
/// query, which matches pi's single-session usage.
trait TerminalBackend: std::fmt::Debug {
    /// Enter raw mode.
    fn enable_raw(&self);
    /// Restore raw mode to off.
    fn restore_raw(&self);
    /// The current dimensions, when the backend can ask the terminal.
    fn size(&self) -> Option<(u16, u16)>;
}

#[derive(Debug)]
struct CrosstermBackend;

impl TerminalBackend for CrosstermBackend {
    fn enable_raw(&self) {
        let _ = crossterm::terminal::enable_raw_mode();
    }

    fn restore_raw(&self) {
        let _ = crossterm::terminal::disable_raw_mode();
    }

    fn size(&self) -> Option<(u16, u16)> {
        crossterm::terminal::size().ok()
    }
}

#[derive(Debug)]
struct HeadlessBackend;

impl TerminalBackend for HeadlessBackend {
    fn enable_raw(&self) {}

    fn restore_raw(&self) {}

    fn size(&self) -> Option<(u16, u16)> {
        None
    }
}

/// Decodes terminal bytes into strings the way Node's
/// `process.stdin.setEncoding("utf8")` did.
///
/// Complete UTF-8 sequences decode through, incomplete tails buffer until
/// the next chunk, and invalid bytes become U+FFFD one replacement per
/// maximal invalid subpart.
#[derive(Debug, Default)]
pub struct Utf8Reassembler {
    pending: Vec<u8>,
}

impl Utf8Reassembler {
    /// Feed raw bytes; returns the decoded string pieces in order.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(bytes);
        let mut decoded = Vec::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    if !text.is_empty() {
                        decoded.push(text.to_string());
                    }
                    self.pending.clear();
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    if valid_up_to > 0 {
                        // valid_up_to marks a valid UTF-8 prefix.
                        decoded.push(
                            String::from_utf8_lossy(&self.pending[..valid_up_to]).into_owned(),
                        );
                        self.pending.drain(..valid_up_to);
                    }
                    match error.error_len() {
                        Some(length) => {
                            decoded.push("\u{FFFD}".to_string());
                            self.pending.drain(..length);
                        }
                        None => break,
                    }
                }
            }
        }
        decoded
    }
}

/// Real terminal using stdin/stdout.
///
/// Construct with [`ProcessTerminal::new`] for the process's own streams, or
/// [`ProcessTerminal::headless`] with an injected sink, environment, clock,
/// and stdin channel for tests.
pub struct ProcessTerminal {
    backend: Box<dyn TerminalBackend>,
    write_sink: WriteSink,
    env_lookup: EnvLookup,
    clock: Box<dyn Fn() -> Instant>,
    input_handler: Option<InputHandler>,
    resize_handler: Option<ResizeHandler>,
    kitty_protocol_active: bool,
    modify_other_keys_active: bool,
    keyboard_protocol_pushed: bool,
    negotiation_buffer: String,
    negotiation_flush_at: Option<Instant>,
    stdin_buffer: Option<StdinBuffer>,
    utf8: Utf8Reassembler,
    keepalive_interval: Duration,
    progress_keepalive: Option<ProgressKeepalive>,
    key_parser: Option<Arc<Mutex<KeyParser>>>,
    write_log_path: PathBuf,
    input_rx: Option<std::sync::mpsc::Receiver<Vec<u8>>>,
    stop_flag: Arc<AtomicBool>,
    /// The SIGWINCH source the resize notification polls, restating
    /// upstream's `process.stdout.on("resize")`.
    #[cfg(unix)]
    sigwinch: Option<signal_hook::iterator::Signals>,
    /// Windows input is out of scope (ADR 0001): no resize source registers.
    #[cfg(not(unix))]
    sigwinch: (),
}

/// The spawned keepalive writing the OSC 9;4 active sequence until stopped.
struct ProgressKeepalive {
    stop: std::sync::mpsc::Sender<()>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for ProcessTerminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessTerminal")
            .field("kitty_protocol_active", &self.kitty_protocol_active)
            .field("modify_other_keys_active", &self.modify_other_keys_active)
            .field("keyboard_protocol_pushed", &self.keyboard_protocol_pushed)
            .field("stdin_buffer", &self.stdin_buffer)
            .field("write_log_path", &self.write_log_path)
            .finish_non_exhaustive()
    }
}

impl Default for ProcessTerminal {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessTerminal {
    /// The real terminal: stdout writes, the process environment, and a
    /// stdin byte reader. Raw mode, the resize registration, and the byte
    /// feed happen in [`Terminal::start`].
    #[must_use]
    pub fn new() -> Self {
        let sink: Box<dyn FnMut(&str) + Send> = Box::new(|data: &str| {
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(data.as_bytes());
            let _ = out.flush();
        });
        let env_lookup: EnvLookup = default_env_lookup();
        Self {
            backend: Box::new(CrosstermBackend),
            write_sink: Arc::new(Mutex::new(sink)),
            write_log_path: resolve_write_log_path(env_lookup.as_ref()),
            env_lookup,
            clock: Box::new(Instant::now),
            input_handler: None,
            resize_handler: None,
            kitty_protocol_active: false,
            modify_other_keys_active: false,
            keyboard_protocol_pushed: false,
            negotiation_buffer: String::new(),
            negotiation_flush_at: None,
            stdin_buffer: None,
            utf8: Utf8Reassembler::default(),
            keepalive_interval: Duration::from_millis(TERMINAL_PROGRESS_KEEPALIVE_MS),
            progress_keepalive: None,
            key_parser: None,
            input_rx: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
            #[cfg(unix)]
            sigwinch: None,
            #[cfg(not(unix))]
            sigwinch: (),
        }
    }

    /// A terminal without a real TTY: writes land in `write_sink`, the
    /// environment/clock/stdin are overridable, and raw mode is a no-op.
    /// The negotiation pipeline arms itself, exactly as
    /// `queryAndEnableKittyProtocol` runs at upstream's start.
    #[must_use]
    pub fn headless(write_sink: WriteSink) -> Self {
        let mut terminal = Self {
            backend: Box::new(HeadlessBackend),
            write_sink,
            env_lookup: default_env_lookup(),
            clock: Box::new(Instant::now),
            write_log_path: PathBuf::new(),
            input_handler: None,
            resize_handler: None,
            kitty_protocol_active: false,
            modify_other_keys_active: false,
            keyboard_protocol_pushed: false,
            negotiation_buffer: String::new(),
            negotiation_flush_at: None,
            stdin_buffer: None,
            utf8: Utf8Reassembler::default(),
            keepalive_interval: Duration::from_millis(TERMINAL_PROGRESS_KEEPALIVE_MS),
            progress_keepalive: None,
            key_parser: None,
            input_rx: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
            #[cfg(unix)]
            sigwinch: None,
            #[cfg(not(unix))]
            sigwinch: (),
        };
        terminal.query_and_enable_kitty_protocol();
        terminal
    }

    /// Overrides the environment lookup (`process.env` upstream). Re-resolves
    /// the write log path from it, as upstream resolves `writeLogPath` once
    /// at construction.
    #[must_use]
    pub fn with_env_lookup(mut self, env_lookup: EnvLookup) -> Self {
        self.write_log_path = resolve_write_log_path(env_lookup.as_ref());
        self.env_lookup = env_lookup;
        self
    }

    /// Overrides the clock the deadline pumps read.
    #[must_use]
    pub fn with_clock(mut self, clock: Box<dyn Fn() -> Instant>) -> Self {
        self.clock = clock;
        self
    }

    /// Overrides the stdin byte source: chunks arrive through this channel,
    /// restating the `process.stdin` feed [`Terminal::start`] wires up.
    #[must_use]
    pub fn with_stdin_source(mut self, input_rx: std::sync::mpsc::Receiver<Vec<u8>>) -> Self {
        self.input_rx = Some(input_rx);
        self
    }

    /// Overrides the key parsing context the protocol state notifies, so
    /// Kitty activation flows to the session's parser instead of upstream's
    /// module global.
    #[must_use]
    pub fn with_key_parser(mut self, key_parser: Arc<Mutex<KeyParser>>) -> Self {
        self.key_parser = Some(key_parser);
        self
    }

    /// Overrides the progress keepalive interval for tests.
    #[must_use]
    pub const fn with_keepalive_interval(mut self, interval: Duration) -> Self {
        self.keepalive_interval = interval;
        self
    }

    /// Whether the xterm `modifyOtherKeys` fallback is currently active.
    #[must_use]
    pub const fn is_modify_other_keys_active(&self) -> bool {
        self.modify_other_keys_active
    }

    /// Installs or clears the input handler, restating the `onInput`
    /// callback upstream's tests stored on the private field directly.
    pub fn set_input_handler(&mut self, input_handler: Option<InputHandler>) {
        self.input_handler = input_handler;
    }

    /// Feed a decoded input chunk, upstream's `process.stdin` `data` event.
    pub fn feed_input(&mut self, data: &str) {
        let Some(stdin_buffer) = self.stdin_buffer.as_mut() else {
            return;
        };
        let events = stdin_buffer.process(data, (self.clock)());
        self.forward_stdin_events(events);
    }

    /// Feed raw stdin bytes through the UTF-8 decoder into the buffer,
    /// upstream's `setEncoding("utf8")` string delivery.
    pub fn feed_bytes(&mut self, bytes: &[u8]) {
        for chunk in self.utf8.feed(bytes) {
            self.feed_input(&chunk);
        }
    }

    /// Fire every due timer: the stdin buffer's flush deadline, the
    /// negotiation fragment buffer's deadline, and a pending resize.
    pub fn pump(&mut self) {
        let now = (self.clock)();
        let events = self
            .stdin_buffer
            .as_mut()
            .map(|stdin_buffer| stdin_buffer.poll_flush(now))
            .unwrap_or_default();
        self.forward_stdin_events(events);

        if self.negotiation_flush_at.is_some_and(|at| now >= at) {
            self.negotiation_flush_at = None;
            self.flush_negotiation_buffer_as_input();
        }

        self.deliver_resize();
    }

    /// Fires the resize handler when SIGWINCH arrived since the last pump,
    /// restating upstream's `process.stdout.on("resize")` dispatch.
    fn deliver_resize(&mut self) {
        #[cfg(unix)]
        let resize_pending = self
            .sigwinch
            .as_mut()
            .is_some_and(|signals| signals.pending().next().is_some());
        #[cfg(not(unix))]
        let resize_pending = false;
        if resize_pending && let Some(resize_handler) = self.resize_handler.as_mut() {
            resize_handler();
        }
    }

    fn forward_stdin_events(&mut self, events: Vec<StdinBufferEvent>) {
        for event in events {
            match event {
                StdinBufferEvent::Data(sequence) => {
                    match self.read_keyboard_protocol_negotiation_sequence(&sequence) {
                        NegotiationRead::Pending => {
                            // Wait briefly for the rest of a split Kitty response.
                            self.schedule_negotiation_buffer_flush();
                        }
                        NegotiationRead::Sequence(negotiation_sequence) => {
                            if self
                                .handle_keyboard_protocol_negotiation_sequence(negotiation_sequence)
                            {
                                continue;
                            }
                            self.forward_input_sequence(&sequence);
                        }
                    }
                }
                // Re-wrap paste content with bracketed paste markers for
                // existing editor handling.
                StdinBufferEvent::Paste(content) => {
                    if let Some(input_handler) = self.input_handler.as_mut() {
                        input_handler(format!("\x1b[200~{content}\x1b[201~"));
                    }
                }
            }
        }
    }

    /// Forward an input sequence to the input handler, restating upstream's
    /// native Shift+Enter detection for the terminals that probe it.
    fn forward_input_sequence(&mut self, sequence: &str) {
        #[cfg(windows)]
        let windows_session = true;
        #[cfg(not(windows))]
        let windows_session = false;
        let should_detect_native_shift_enter =
            sequence == "\r" && (self.is_apple_terminal_session() || windows_session);
        let is_shift_pressed = should_detect_native_shift_enter && Self::native_shift_pressed();
        let input = normalize_native_shift_enter_input(
            sequence,
            should_detect_native_shift_enter,
            is_shift_pressed,
        );
        if let Some(input_handler) = self.input_handler.as_mut() {
            input_handler(input);
        }
    }

    fn is_apple_terminal_session(&self) -> bool {
        let term_program = (self.env_lookup)("TERM_PROGRAM");
        #[cfg(target_os = "macos")]
        {
            term_program.is_some_and(|value| value == "Apple_Terminal")
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = term_program;
            false
        }
    }

    /// The native modifier probe, upstream `isNativeModifierPressed("shift")`
    /// over the N-API platform helper: the replacement reads the same
    /// `CGEventSourceFlagsState` combined-session state through `readkey` on
    /// macOS (#51); every other platform loads no helper upstream and answers
    /// `false` the same way.
    #[cfg(target_os = "macos")]
    fn native_shift_pressed() -> bool {
        crate::native_modifiers::is_native_modifier_pressed(
            crate::native_modifiers::ModifierKey::Shift,
        )
    }

    #[cfg(not(target_os = "macos"))]
    const fn native_shift_pressed() -> bool {
        false
    }

    /// Query the terminal for Kitty keyboard protocol support and enable it
    /// if available.
    ///
    /// Kitty's progressive enhancement detection requires requesting the
    /// desired flags before querying them. The trailing DA query is a
    /// sentinel supported by terminals that do not know Kitty keyboard
    /// protocol; receiving DA before a Kitty response enables the
    /// modifyOtherKeys fallback without a startup timeout.
    ///
    /// The requested flags are:
    /// - 1 = disambiguate escape codes
    /// - 2 = report event types (press/repeat/release)
    /// - 4 = report alternate keys (shifted key, base layout key)
    fn query_and_enable_kitty_protocol(&mut self) {
        let escape_timeout_ms = {
            let env_lookup = self.env_lookup.as_ref();
            resolve_escape_timeout_ms(move |key| env_lookup(key))
        };
        self.stdin_buffer = Some(StdinBuffer::new().with_escape_timeout_ms(escape_timeout_ms));

        self.keyboard_protocol_pushed = true;
        self.clear_negotiation_buffer();
        self.write(&KITTY_KEYBOARD_PROTOCOL_QUERY);
    }

    /// Handle a parsed negotiation sequence; returns whether it was consumed.
    fn handle_keyboard_protocol_negotiation_sequence(
        &mut self,
        negotiation_sequence: Option<KeyboardProtocolNegotiationSequence>,
    ) -> bool {
        let Some(negotiation_sequence) = negotiation_sequence else {
            return false;
        };
        self.clear_negotiation_buffer();
        match negotiation_sequence {
            KeyboardProtocolNegotiationSequence::KittyFlags { flags } => {
                if flags != 0 {
                    self.disable_modify_other_keys();
                    if !self.kitty_protocol_active {
                        self.set_kitty_protocol_active(true);
                    }
                } else {
                    self.enable_modify_other_keys();
                }
                true
            }
            KeyboardProtocolNegotiationSequence::DeviceAttributes => {
                if !self.kitty_protocol_active {
                    self.enable_modify_other_keys();
                }
                true
            }
        }
    }

    /// Merge the buffered prefix into the current sequence, restating
    /// upstream's `readKeyboardProtocolNegotiationSequence` buffer-or-pending
    /// walk.
    fn read_keyboard_protocol_negotiation_sequence(&mut self, sequence: &str) -> NegotiationRead {
        if !self.negotiation_buffer.is_empty() {
            let buffered_sequence = format!("{}{sequence}", self.negotiation_buffer);
            if let Some(negotiation_sequence) =
                parse_keyboard_protocol_negotiation_sequence(&buffered_sequence)
            {
                self.clear_negotiation_buffer();
                return NegotiationRead::Sequence(Some(negotiation_sequence));
            }
            if is_keyboard_protocol_negotiation_sequence_prefix(&buffered_sequence) {
                self.set_keyboard_protocol_negotiation_buffer(buffered_sequence);
                return NegotiationRead::Pending;
            }
            self.flush_negotiation_buffer_as_input();
        }

        if let Some(negotiation_sequence) = parse_keyboard_protocol_negotiation_sequence(sequence) {
            return NegotiationRead::Sequence(Some(negotiation_sequence));
        }
        if is_keyboard_protocol_negotiation_sequence_prefix(sequence) {
            self.set_keyboard_protocol_negotiation_buffer(sequence.to_string());
            return NegotiationRead::Pending;
        }
        NegotiationRead::Sequence(None)
    }

    fn set_keyboard_protocol_negotiation_buffer(&mut self, sequence: String) {
        self.clear_negotiation_buffer_flush_timer();
        self.negotiation_buffer = sequence;
    }

    fn clear_negotiation_buffer(&mut self) {
        self.clear_negotiation_buffer_flush_timer();
        self.negotiation_buffer.clear();
    }

    /// Forwards the buffered negotiation prefix as ordinary input, upstream's
    /// `flushKeyboardProtocolNegotiationBufferAsInput`.
    fn flush_negotiation_buffer_as_input(&mut self) {
        // The buffer and its flush timer are cleared together, so a due
        // timer always finds a buffered prefix; upstream's empty guard is a
        // race its event loop could hit, the port's pump cannot.
        self.clear_negotiation_buffer_flush_timer();
        let sequence = std::mem::take(&mut self.negotiation_buffer);
        self.forward_input_sequence(&sequence);
    }

    /// Arms the 150 ms fragment flush, upstream's
    /// `scheduleKeyboardProtocolNegotiationBufferFlush`.
    fn schedule_negotiation_buffer_flush(&mut self) {
        if self.negotiation_buffer.is_empty() || self.negotiation_flush_at.is_some() {
            return;
        }
        self.negotiation_flush_at = Some(
            (self.clock)() + Duration::from_millis(KEYBOARD_PROTOCOL_RESPONSE_FRAGMENT_TIMEOUT_MS),
        );
    }

    const fn clear_negotiation_buffer_flush_timer(&mut self) {
        self.negotiation_flush_at = None;
    }

    fn enable_modify_other_keys(&mut self) {
        if self.kitty_protocol_active || self.modify_other_keys_active {
            return;
        }
        self.write("\x1b[>4;2m");
        self.modify_other_keys_active = true;
    }

    fn disable_modify_other_keys(&mut self) {
        if !self.modify_other_keys_active {
            return;
        }
        self.write("\x1b[>4;0m");
        self.modify_other_keys_active = false;
    }

    fn set_kitty_protocol_active(&mut self, active: bool) {
        self.kitty_protocol_active = active;
        if let Some(key_parser) = &self.key_parser {
            let mut key_parser = key_parser.lock().unwrap_or_else(PoisonError::into_inner);
            key_parser.set_kitty_protocol_active(active);
        }
    }

    fn clear_progress_keepalive(&mut self) -> bool {
        let Some(keepalive) = self.progress_keepalive.take() else {
            return false;
        };
        let _ = keepalive.stop.send(());
        if let Some(handle) = keepalive.handle {
            let _ = handle.join();
        }
        true
    }
}

/// How [`ProcessTerminal::forward_stdin_events`] routes a sequence after the
/// negotiation walk.
enum NegotiationRead {
    /// A parsed negotiation sequence, or `None` for ordinary input.
    Sequence(Option<KeyboardProtocolNegotiationSequence>),
    /// A buffered prefix that could still become a negotiation sequence.
    Pending,
}

impl Terminal for ProcessTerminal {
    fn start(&mut self, on_input: InputHandler, on_resize: ResizeHandler) {
        self.input_handler = Some(on_input);
        self.resize_handler = Some(on_resize);

        // Save previous state and enable raw mode.
        self.backend.enable_raw();

        // Enable bracketed paste mode - terminal will wrap pastes in
        // \x1b[200~ ... \x1b[201~
        self.write("\x1b[?2004h");

        // Register the SIGWINCH-driven resize notification, then refresh the
        // dimensions - they may be stale after suspend/resume (SIGWINCH is
        // lost while the process is stopped). POSIX only, best-effort.
        self.attach_resize();
        refresh_terminal_dimensions();

        // Query Kitty keyboard protocol and fall back to modifyOtherKeys when
        // DA confirms no Kitty response.
        // See: https://sw.kovidgoyal.net/kitty/keyboard-protocol/
        self.query_and_enable_kitty_protocol();

        // Spawn the stdin byte feed. The reader exits on EOF, an error, or
        // the stop flag.
        let (input_tx, input_rx) = std::sync::mpsc::channel();
        self.input_rx = Some(input_rx);
        let stop_flag = Arc::clone(&self.stop_flag);
        let mut source: Box<dyn Read + Send> = Box::new(std::io::stdin());
        std::thread::spawn(move || read_input_chunks(&mut source, &input_tx, &stop_flag));
    }

    fn stop(&mut self) {
        if self.clear_progress_keepalive() {
            self.write(TERMINAL_PROGRESS_CLEAR_SEQUENCE);
        }

        // Disable bracketed paste mode
        self.write("\x1b[?2004l");

        let should_disable_kitty_protocol =
            self.keyboard_protocol_pushed || self.kitty_protocol_active;
        self.clear_negotiation_buffer();

        // Disable Kitty keyboard protocol if not already done by drainInput()
        if should_disable_kitty_protocol {
            self.write("\x1b[<u");
            self.keyboard_protocol_pushed = false;
            self.set_kitty_protocol_active(false);
        }
        self.disable_modify_other_keys();

        // Clean up StdinBuffer
        if let Some(mut stdin_buffer) = self.stdin_buffer.take() {
            stdin_buffer.clear();
        }

        // Remove event handlers
        self.input_handler = None;
        self.resize_handler = None;
        // Dropping the SIGWINCH source unregisters it.
        #[cfg(unix)]
        {
            self.sigwinch = None;
        }
        #[cfg(not(unix))]
        {
            self.sigwinch = ();
        }

        // Pause stdin restated: the reader stops consuming on the stop flag
        // and the channel drops, so buffered input (e.g., Ctrl+D) is never
        // re-interpreted after raw mode is disabled.
        self.input_rx = None;
        self.stop_flag.store(true, Ordering::Relaxed);

        // Restore raw mode state
        self.backend.restore_raw();
    }

    fn drain_input(&mut self, max_ms: u64, idle_ms: u64) {
        let should_disable_kitty_protocol =
            self.keyboard_protocol_pushed || self.kitty_protocol_active;
        self.clear_negotiation_buffer();
        if should_disable_kitty_protocol {
            // Disable Kitty keyboard protocol first so any late key releases
            // do not generate new Kitty escape sequences.
            self.write("\x1b[<u");
            self.keyboard_protocol_pushed = false;
            self.set_kitty_protocol_active(false);
        }
        self.disable_modify_other_keys();

        let previous_handler = self.input_handler.take();

        let start = (self.clock)();
        let mut last_data_time = start;
        loop {
            let now = (self.clock)();
            if now.duration_since(start) >= Duration::from_millis(max_ms) {
                break;
            }
            if now.duration_since(last_data_time) >= Duration::from_millis(idle_ms) {
                break;
            }
            let wait = Duration::from_millis(idle_ms)
                .min(Duration::from_millis(max_ms).saturating_sub(now.duration_since(start)));
            match self
                .input_rx
                .as_ref()
                .map(|input_rx| input_rx.recv_timeout(wait))
            {
                Some(Ok(chunk)) => {
                    self.feed_bytes(&chunk);
                    self.pump();
                    last_data_time = (self.clock)();
                }
                Some(Err(std::sync::mpsc::RecvTimeoutError::Timeout)) => {}
                Some(Err(std::sync::mpsc::RecvTimeoutError::Disconnected)) => {
                    self.input_rx = None;
                }
                None => std::thread::sleep(wait),
            }
        }

        self.input_handler = previous_handler;
    }

    fn write(&mut self, data: &str) {
        let mut write_sink = self
            .write_sink
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        write_sink(data);
        drop(write_sink);
        if !self.write_log_path.as_os_str().is_empty()
            && let Ok(mut log) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.write_log_path)
        {
            let _ = log.write_all(data.as_bytes());
        }
    }

    fn columns(&self) -> u16 {
        self.backend
            .size()
            .filter(|&(columns, _)| columns > 0)
            .map(|(columns, _)| columns)
            .or_else(|| self.env_dimension("COLUMNS"))
            .unwrap_or(DEFAULT_COLUMNS)
    }

    fn rows(&self) -> u16 {
        self.backend
            .size()
            .filter(|&(_, rows)| rows > 0)
            .map(|(_, rows)| rows)
            .or_else(|| self.env_dimension("LINES"))
            .unwrap_or(DEFAULT_ROWS)
    }

    fn is_kitty_protocol_active(&self) -> bool {
        self.kitty_protocol_active
    }

    fn move_by(&mut self, lines: i32) {
        if lines > 0 {
            // Move down
            self.write(&format!("\x1b[{lines}B"));
        } else if lines < 0 {
            // Move up
            self.write(&format!("\x1b[{}A", -lines));
        }
        // lines === 0: no movement
    }

    fn hide_cursor(&mut self) {
        self.write("\x1b[?25l");
    }

    fn show_cursor(&mut self) {
        self.write("\x1b[?25h");
    }

    fn clear_line(&mut self) {
        self.write("\x1b[K");
    }

    fn clear_from_cursor(&mut self) {
        self.write("\x1b[J");
    }

    fn clear_screen(&mut self) {
        self.write("\x1b[2J\x1b[H"); // Clear screen and move to home (1,1)
    }

    fn set_title(&mut self, title: &str) {
        // OSC 0;title BEL - set terminal window title
        self.write(&format!("\x1b]0;{title}\x07"));
    }

    fn set_progress(&mut self, active: bool) {
        if active {
            // OSC 9;4;3 - indeterminate progress
            self.write(TERMINAL_PROGRESS_ACTIVE_SEQUENCE);
            if self.progress_keepalive.is_none() {
                self.progress_keepalive = Some(spawn_progress_keepalive(
                    Arc::clone(&self.write_sink),
                    self.keepalive_interval,
                ));
            }
        } else {
            self.clear_progress_keepalive();
            // OSC 9;4;0 - clear progress
            self.write(TERMINAL_PROGRESS_CLEAR_SEQUENCE);
        }
    }

    fn poll(&mut self, timeout: Duration) {
        let now = (self.clock)();
        let wake = match (
            self.stdin_buffer
                .as_ref()
                .and_then(StdinBuffer::flush_deadline),
            self.negotiation_flush_at,
        ) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        let wait = wake.map_or(timeout, |at| at.saturating_duration_since(now).min(timeout));

        if let Some(Ok(chunk)) = self
            .input_rx
            .as_ref()
            .map(|input_rx| input_rx.recv_timeout(wait))
        {
            self.feed_bytes(&chunk);
        }

        self.pump();
    }
}

impl ProcessTerminal {
    /// Registers the SIGWINCH-driven resize notification, restating
    /// upstream's `process.stdout.on("resize")`: the signal lands in the
    /// iterator's queue and [`Terminal::poll`] dispatches it.
    fn attach_resize(&mut self) {
        #[cfg(unix)]
        {
            if let Ok(signals) = signal_hook::iterator::Signals::new([libc::SIGWINCH]) {
                self.sigwinch = Some(signals);
            }
        }
    }

    fn env_dimension(&self, key: &str) -> Option<u16> {
        (self.env_lookup)(key)
            .and_then(|value| value.parse::<u16>().ok())
            .filter(|&dimension| dimension > 0)
    }
}

/// Arms the keepalive writing the OSC 9;4 indeterminate sequence every
/// `interval` until a stop arrives, upstream's `setInterval` in
/// `setProgress`.
fn spawn_progress_keepalive(write_sink: WriteSink, interval: Duration) -> ProgressKeepalive {
    let (stop, stop_rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        loop {
            match stop_rx.recv_timeout(interval) {
                Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    let mut write_sink = write_sink.lock().unwrap_or_else(PoisonError::into_inner);
                    write_sink(TERMINAL_PROGRESS_ACTIVE_SEQUENCE);
                }
            }
        }
    });
    ProgressKeepalive {
        stop,
        handle: Some(handle),
    }
}

/// Resolves `PI_TUI_WRITE_LOG`: an existing directory gains a timestamped log
/// file inside; anything else is used as the file path as-is, restating
/// upstream's `writeLogPath` initializer.
fn resolve_write_log_path(env: &dyn Fn(&str) -> Option<String>) -> PathBuf {
    let Some(configured) = env("PI_TUI_WRITE_LOG") else {
        return PathBuf::new();
    };
    if configured.is_empty() {
        return PathBuf::new();
    }
    let path = PathBuf::from(configured);
    match std::fs::metadata(&path) {
        Ok(metadata) if metadata.is_dir() => path.join(format!(
            "tui-{}-{}.log",
            write_log_timestamp(),
            std::process::id()
        )),
        // Not an existing directory - use as-is (file path)
        _ => path,
    }
}

/// The `YYYY-MM-DD_HH-MM-SS` stamp upstream formats with `new Date()`.
fn write_log_timestamp() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let days = i64::try_from(seconds / 86_400).unwrap_or(0);
    let seconds_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}_{hour:02}-{minute:02}-{second:02}",
        hour = seconds_of_day / 3600,
        minute = (seconds_of_day % 3_600) / 60,
        second = seconds_of_day % 60,
    )
}

/// Days since the Unix epoch to a civil date, Howard Hinnant's algorithm, so
/// no calendar dependency enters the graph for a debug filename.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "month lands in 1..=12 and day in 1..=31 by construction of the algorithm, so the casts never truncate"
)]
const fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 1_531;
    let day = day_of_year - (1_531 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (
        if month <= 2 { year + 1 } else { year },
        month as u32,
        day as u32,
    )
}

/// Reads raw stdin bytes in chunks until EOF, an error, or the stop flag,
/// upstream's `process.stdin` byte feed. Public so the suite can drive it
/// over an in-memory reader.
pub fn read_input_chunks(
    source: &mut dyn Read,
    input_tx: &std::sync::mpsc::Sender<Vec<u8>>,
    stop_flag: &AtomicBool,
) {
    let mut chunk = [0_u8; READ_CHUNK_BYTES];
    while !stop_flag.load(Ordering::Relaxed) {
        match source.read(&mut chunk) {
            Ok(0) => break,
            Ok(length) => {
                if input_tx.send(chunk[..length].to_vec()).is_err() {
                    break;
                }
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
}

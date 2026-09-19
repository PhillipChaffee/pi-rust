//! Terminal UI library with differential rendering, ported from
//! `packages/tui` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The port lands crate-wide in dependency order (map ticket "Port pi-tui",
//! [ADR 0001](https://github.com/PhillipChaffee/pi-rust/blob/main/docs/adr/0001-tui-renderer-port-not-ratatui.md)):
//! stateful components render rows of pre-styled ANSI strings, the renderer
//! diffs whole lines and writes only what changed, and crossterm stays thin
//! plumbing for raw mode, terminal size, and terminal state.
//!
//! This slice carries the ANSI-aware string-width utilities ([`utils`]), the
//! input parsing layer ([#40](https://github.com/PhillipChaffee/pi-rust/issues/40)),
//! and the terminal I/O layer ([#41](https://github.com/PhillipChaffee/pi-rust/issues/41));
//! the renderers and component library land with their own port tickets.
//!
//! - [`utils`] — ANSI-aware width measurement and manipulation.
//! - [`keys`] — the [`keys::Key`] identifier helper, [`keys::KeyParser`]'s
//!   `matches_key`/`parse_key`, legacy CSI/SS3 decoding, Kitty `CSI u` events,
//!   and printable decoding.
//! - [`keybindings`] — the runtime registry of `tui.*` action ids,
//!   [`keybindings::TUI_KEYBINDINGS`] defaults,
//!   [`keybindings::KeybindingsManager`], and the process-wide accessor pair.
//! - [`stdin_buffer`] — [`stdin_buffer::StdinBuffer`], the escape-sequence
//!   reassembly layer (50 ms sequence timeout, 10 ms lone-ESC window,
//!   bracketed-paste batch splitting, the WezTerm ESC+ESC split, and the
//!   duplicate-printable drop).
//! - [`terminal`] — the [`terminal::Terminal`] seam and
//!   [`terminal::ProcessTerminal`]: raw mode, size, and terminal state on
//!   crossterm, the Kitty keyboard-protocol negotiation with its
//!   modifyOtherKeys fallback, the SIGWINCH resize notification, the
//!   lone-ESC timeout resolution (`PI_TUI_ESC_TIMEOUT`, 10/100 ms), and
//!   `drainInput`.
//! - [`terminal_colors`] — the OSC 11 background-color response parser and
//!   the `CSI ? 997 ; n` color-scheme report parser.
//!
//! Two upstream shapes in the input layer are restated rather than copied:
//!
//! - Upstream `keys.ts` kept Kitty protocol state in a mutable module global
//!   (`_kittyProtocolActive`, set by `ProcessTerminal` after protocol
//!   detection). The port moves that state — plus the Windows Terminal
//!   session probe upstream read from `process.env` per call — onto
//!   [`keys::KeyParser`], a context a UI session owns and passes to the
//!   keybinding manager. Rust cannot mutate the process environment without
//!   the `unsafe` this workspace forbids, so the env probe is sampled at
//!   construction and overridable.
//! - Upstream named valid actions through declaration merging on the
//!   `Keybindings` interface. The port keeps a runtime registry keyed by
//!   string action ids ([`keybindings::Keybindings`]) with a builder config:
//!   seed [`keybindings::Keybindings::tui_defaults`], register app actions,
//!   and build the manager.
//!
//! The terminal I/O layer carries its own restatements, recorded in the
//! module docs: the Node event loop becomes an owner-driven
//! [`terminal::Terminal::poll`] pump, the `process.stdin` byte feed becomes a
//! UTF-8-decoding reader over the same [`terminal::Terminal`] seam, and the
//! environment/write-sink/clock seams upstream monkeypatched in tests become
//! constructor parameters.
//!
//! The 1:1 ports of upstream's `node:test` suites for these modules are the
//! acceptance gate; they live in the crate's integration tests, with boundary
//! tests added where upstream left branches untested (event-type queries,
//! modifier table entries, the write log, the UTF-8 decoder, the progress
//! keepalive) so the 95% coverage gate binds.

pub mod keybindings;
pub mod keys;
pub mod stdin_buffer;
pub mod terminal;
pub mod terminal_colors;
pub mod utils;

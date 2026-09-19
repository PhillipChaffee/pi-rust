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
//! This slice carries the ANSI-aware string-width utilities ([`utils`]) and
//! the input parsing layer ([#40](https://github.com/PhillipChaffee/pi-rust/issues/40)):
//! the renderers and component library land with their own port tickets.
//!
//! - [`keys`] — the [`keys::Key`] identifier helper, [`keys::KeyParser`]'s
//!   `matches_key`/`parse_key`, legacy CSI/SS3 decoding, Kitty `CSI u` events,
//!   and printable decoding.
//! - [`keybindings`] — the runtime registry of `tui.*` action ids,
//!   [`keybindings::TUI_KEYBINDINGS`] defaults,
//!   [`keybindings::KeybindingsManager`], and the process-wide accessor pair.
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
//! The 1:1 ports of upstream's `node:test` suites for these modules are the
//! acceptance gate; they live in the crate's integration tests, with boundary
//! tests added where upstream left branches untested (event-type queries,
//! modifier table entries) so the 95% coverage gate binds.

pub mod keybindings;
pub mod keys;
pub mod utils;

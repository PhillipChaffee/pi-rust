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
//! This slice carries the ANSI-aware string-width utilities ([`utils`]); the
//! input parser, renderers, and component library land with their own port
//! tickets.

pub mod utils;

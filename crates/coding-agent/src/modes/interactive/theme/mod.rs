//! The interactive theme system, upstream's
//! `packages/coding-agent/src/modes/interactive/theme/`.
//!
//! - [`theme`] — the `Theme` type, built-in and custom loading, terminal
//!   appearance detection, the process-wide active theme, export colors, and
//!   the TUI theme getters.
//! - [`theme_json`] — theme document validation and the value vocabulary.
//! - [`theme_controller`] — the settings-bound controller that follows
//!   automatic light/dark themes.

#[expect(
    clippy::module_inception,
    reason = "the path mirrors upstream's theme/theme.ts"
)]
pub mod theme;
pub mod theme_controller;
pub mod theme_json;

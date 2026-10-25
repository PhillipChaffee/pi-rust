//! App-level keybindings, upstream's `src/core/keybindings.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The `app.*` actions register over pi-tui's
//! [`Keybindings`](pi_tui::keybindings::Keybindings) registry, and
//! the defaults that upstream switched on `process.platform` /
//! `WSL_DISTRO_NAME` / `WSL_INTEROP` are rebased here as
//! [`keybindings_with`] over an explicit platform and environment lookup, so
//! every arm is testable on one host. [`node_platform`] restates
//! `process.platform` spellings for the process-default [`keybindings`].
//!
//! Recorded restatements:
//!
//! - `loadRawConfig` accepts a JSON object root only. Upstream's `typeof`
//!   check let arrays through, which bound their indices as action names;
//!   no suite pins that artifact.
//! - `getEffectiveConfig` returns the resolved
//!   [`KeybindingsConfig`](pi_tui::keybindings::KeybindingsConfig) with
//!   every action's key list. Upstream's `getResolvedBindings` collapsed a
//!   single key to a scalar for JSON serialization; the Rust config keeps
//!   lists and the serializer owns the scalar shape.

#[path = "keybindings/definition.rs"]
mod definition;
#[path = "keybindings/manager.rs"]
mod manager;

pub use definition::{keybindings, keybindings_with, node_platform, use_windows_keybindings};
pub use manager::{
    AppKeybindingsManager, keybindings_config_from_value, migrate_keybindings_config,
    migrate_keybindings_config_value,
};

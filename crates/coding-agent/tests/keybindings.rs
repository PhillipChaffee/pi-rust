//! The `keybindings.test.ts` suite at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, ported 1:1.
//!
//! Upstream read `process.platform` and `process.env` directly; the port
//! injects both through [`keybindings_with`] and [`use_windows_keybindings`],
//! so the Windows/WSL arms run on any host.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

mod common;

use pi_coding_agent::config::default_env_lookup;
use pi_coding_agent::core::keybindings::{
    keybindings_with, node_platform, use_windows_keybindings,
};
use pi_tui::keybindings::Keybindings;

use common::env_with;

/// The default keys bound to `action`.
fn default_keys(registry: &Keybindings, action: &str) -> Vec<String> {
    registry
        .definition(action)
        .expect("the action is registered")
        .default_keys
        .clone()
}

#[test]
fn uses_windows_keybindings_on_native_windows() {
    assert!(use_windows_keybindings("win32", &env_with(&[])));
}

#[test]
fn uses_windows_keybindings_in_wsl_without_windows_terminal_detection() {
    assert!(use_windows_keybindings(
        "linux",
        &env_with(&[("WSL_DISTRO_NAME", "Ubuntu")])
    ));
    assert!(use_windows_keybindings(
        "linux",
        &env_with(&[("WSL_INTEROP", "/run/WSL/123_interop")])
    ));
}

#[test]
fn does_not_use_windows_keybindings_from_wt_session_alone() {
    assert!(!use_windows_keybindings(
        "linux",
        &env_with(&[("WT_SESSION", "session")])
    ));
}

#[test]
fn keeps_non_windows_defaults_on_other_platforms() {
    assert!(!use_windows_keybindings("linux", &env_with(&[])));
    assert!(!use_windows_keybindings("darwin", &env_with(&[])));
}

#[test]
fn applies_the_detected_defaults_consistently() {
    let env = default_env_lookup();
    let platform = node_platform();
    let windows = use_windows_keybindings(platform, &env);
    let native_windows = platform == "win32";
    let registry = keybindings_with(platform, &env);

    let one = |key: &'static str| vec![key.to_string()];
    assert_eq!(
        default_keys(&registry, "app.clipboard.pasteImage"),
        if windows { one("alt+v") } else { one("ctrl+v") }
    );
    assert_eq!(
        default_keys(&registry, "tui.altScreen.search"),
        if windows {
            one("ctrl+f")
        } else {
            one("ctrl+shift+f")
        }
    );
    assert_eq!(
        default_keys(&registry, "app.message.followUp"),
        if windows {
            one("ctrl+q")
        } else {
            one("alt+enter")
        }
    );
    assert_eq!(
        default_keys(&registry, "app.model.cycleBackward"),
        if windows {
            one("alt+p")
        } else {
            one("shift+ctrl+p")
        }
    );
    assert_eq!(
        default_keys(&registry, "tui.editor.undo"),
        if native_windows {
            one("ctrl+z")
        } else if windows {
            one("alt+z")
        } else {
            one("ctrl+-")
        }
    );
    assert_eq!(
        default_keys(&registry, "tui.altScreen.previousPrompt"),
        if windows {
            one("ctrl+up")
        } else {
            vec!["ctrl+shift+up".to_string(), "ctrl+up".to_string()]
        }
    );
    assert_eq!(
        default_keys(&registry, "tui.altScreen.nextPrompt"),
        if windows {
            one("ctrl+down")
        } else {
            vec!["ctrl+shift+down".to_string(), "ctrl+down".to_string()]
        }
    );
    assert_eq!(
        default_keys(&registry, "app.message.dequeue"),
        if windows { one("alt+q") } else { one("alt+up") }
    );
}

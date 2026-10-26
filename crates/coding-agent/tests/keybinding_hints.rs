//! Active key hints and platform formatting at upstream pin `60e7e76` (#195).

#[path = "common/selectors.rs"]
mod selectors;

use pi_coding_agent::core::keybindings::node_platform;
use pi_coding_agent::modes::interactive::components::keybinding_hints::{
    KeyTextFormatOptions, format_key_text, format_key_text_with, key_display_text, key_hint,
    key_text, raw_key_hint,
};
use pi_coding_agent::utils::ansi::strip_ansi;

use selectors::setup;

#[test]
fn formats_modifiers_alternatives_empty_parts_and_platform_option_key() {
    let lower = KeyTextFormatOptions::default();
    let upper = KeyTextFormatOptions { capitalize: true };
    for (key, linux, darwin) in [
        (
            "ctrl+alt+x/shift+tab",
            "Ctrl+Alt+X/Shift+Tab",
            "Ctrl+Option+X/Shift+Tab",
        ),
        ("ALT+a", "ALT+A", "Option+A"),
        ("", "", ""),
        ("+a/", "+A/", "+A/"),
        ("é+ß", "É+SS", "É+SS"),
    ] {
        assert_eq!(format_key_text_with(key, upper, "linux"), linux);
        assert_eq!(format_key_text_with(key, upper, "darwin"), darwin);
    }
    assert_eq!(format_key_text_with("alt+x", lower, "darwin"), "option+x");
    assert_eq!(format_key_text_with("alt+x", lower, "linux"), "alt+x");
    assert_eq!(
        format_key_text("alt+x", upper),
        format_key_text_with("alt+x", upper, node_platform())
    );
}

#[test]
fn hints_use_all_active_bindings_and_unbound_actions_are_empty() {
    setup(&[
        ("app.thinking.save", &["ctrl+r", "ctrl+s"]),
        ("tui.select.cancel", &[]),
    ]);
    assert_eq!(key_text("app.thinking.save"), "ctrl+r/ctrl+s");
    assert_eq!(key_display_text("app.thinking.save"), "Ctrl+R/Ctrl+S");
    assert_eq!(
        strip_ansi(&key_hint("app.thinking.save", "save")),
        "ctrl+r/ctrl+s save"
    );
    assert_eq!(strip_ansi(&raw_key_hint("↑↓", "navigate")), "↑↓ navigate");
    assert_eq!(key_text("tui.select.cancel"), "");
    assert_eq!(key_display_text("not.registered"), "");
    assert_eq!(
        strip_ansi(&key_hint("tui.select.cancel", "cancel")),
        " cancel"
    );
}

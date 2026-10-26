//! Key hints use the active registry, including user overrides (#195).

use pi_tui::keybindings::get_keybindings;

use crate::core::keybindings::node_platform;
use crate::modes::interactive::theme::theme::theme;

/// Display options for key parts, upstream's `KeyTextFormatOptions`.
#[derive(Debug, Clone, Copy, Default)]
pub struct KeyTextFormatOptions {
    /// Capitalize each modifier and key; defaults to false.
    pub capitalize: bool,
}

/// Formats slash-separated alternatives and plus-separated key parts.
#[must_use]
pub fn format_key_text(key: &str, options: KeyTextFormatOptions) -> String {
    format_key_text_with(key, options, node_platform())
}

/// The platform-injected form; `darwin` displays `alt` as `option`.
#[must_use]
pub fn format_key_text_with(key: &str, options: KeyTextFormatOptions, platform: &str) -> String {
    key.split('/')
        .map(|key| {
            key.split('+')
                .map(|part| {
                    let part = if platform == "darwin" && part.eq_ignore_ascii_case("alt") {
                        "option"
                    } else {
                        part
                    };
                    if options.capitalize {
                        let mut chars = part.chars();
                        chars.next().map_or_else(String::new, |first| {
                            first.to_uppercase().collect::<String>() + chars.as_str()
                        })
                    } else {
                        part.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("+")
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// Lowercase key text from the active bindings; unbound actions yield empty text.
#[must_use]
pub fn key_text(keybinding: &str) -> String {
    let keys = get_keybindings().get_keys(keybinding);
    format_key_text(&keys.join("/"), KeyTextFormatOptions::default())
}

/// Capitalized key text from the active bindings.
#[must_use]
pub fn key_display_text(keybinding: &str) -> String {
    let keys = get_keybindings().get_keys(keybinding);
    format_key_text(&keys.join("/"), KeyTextFormatOptions { capitalize: true })
}

/// A dimmed active binding followed by its muted description.
#[must_use]
pub fn key_hint(keybinding: &str, description: &str) -> String {
    theme().fg("dim", &key_text(keybinding)) + &theme().fg("muted", &format!(" {description}"))
}

/// A literal key hint, for navigation not represented by one registry action.
#[must_use]
pub fn raw_key_hint(key: &str, description: &str) -> String {
    theme().fg(
        "dim",
        &format_key_text(key, KeyTextFormatOptions::default()),
    ) + &theme().fg("muted", &format!(" {description}"))
}

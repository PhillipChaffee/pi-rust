//! Keybinding hint formatting, upstream
//! `src/modes/interactive/components/keybinding-hints.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Hints read the process-wide keybinding registry, so a remapped action's
//! physical keys appear in every row that names it.

use pi_tui::keybindings::get_keybindings;

use crate::core::keybindings::node_platform;
use crate::modes::interactive::theme::theme::theme;

/// Format one key chord, mapping `alt` to `option` on macOS, upstream's
/// `formatKeyPart` plus `formatKeyText`.
#[must_use]
pub fn format_key_text(key: &str, capitalize: bool) -> String {
    key.split('/')
        .map(|chord| {
            chord
                .split('+')
                .map(|part| {
                    let display = if node_platform() == "darwin" && part.eq_ignore_ascii_case("alt")
                    {
                        "option"
                    } else {
                        part
                    };
                    if capitalize {
                        let mut characters = display.chars();
                        characters.next().map_or_else(String::new, |first| {
                            format!("{}{}", first.to_uppercase(), characters.as_str())
                        })
                    } else {
                        display.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("+")
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// The keys bound to `action`, formatted for prose, upstream's `keyText`.
#[must_use]
pub fn key_text(action: &str) -> String {
    let keys = get_keybindings().get_keys(action);
    if keys.is_empty() {
        return String::new();
    }
    format_key_text(&keys.join("/"), false)
}

/// The keys bound to `action`, capitalized for a label, upstream's
/// `keyDisplayText`.
#[must_use]
pub fn key_display_text(action: &str) -> String {
    let keys = get_keybindings().get_keys(action);
    if keys.is_empty() {
        return String::new();
    }
    format_key_text(&keys.join("/"), true)
}

/// A dim key chord followed by a muted description, upstream's `keyHint`.
#[must_use]
pub fn key_hint(action: &str, description: &str) -> String {
    let theme = theme();
    format!(
        "{}{}",
        theme.fg("dim", &key_text(action)),
        theme.fg("muted", &format!(" {description}"))
    )
}

/// [`key_hint`] over a literal key chord, upstream's `rawKeyHint`.
#[must_use]
pub fn raw_key_hint(key: &str, description: &str) -> String {
    let theme = theme();
    format!(
        "{}{}",
        theme.fg("dim", &format_key_text(key, false)),
        theme.fg("muted", &format!(" {description}"))
    )
}

//! Keyboard input handling, ported from `packages/tui/src/keys.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#40).
//!
//! Supports both legacy terminal sequences and the Kitty keyboard protocol
//! (<https://sw.kovidgoyal.net/kitty/keyboard-protocol/>).
//!
//! Symbol keys are also supported, however some ctrl+symbol combos overlap
//! with ASCII codes, e.g. ctrl+[ = ESC
//! (<https://sw.kovidgoyal.net/kitty/keyboard-protocol/#legacy-ctrl-mapping-of-ascii-keys>).
//! Those can still be used for ctrl+shift combos.
//!
//! API:
//! - [`KeyParser::matches_key`] — check if input matches a key identifier
//! - [`KeyParser::parse_key`] — parse input and return the key identifier
//! - [`Key`] — helper for building key identifiers
//! - [`is_key_release`] / [`is_key_repeat`] — query the parsed event type
//! - [`decode_kitty_printable`] / [`decode_printable_key`] — printable decoding
//!
//! A [`KeyParser`] owns the parsing state upstream kept in mutable module
//! globals; see the crate docs for the restatement.

/// A key identifier such as `"ctrl+c"`, `"shift+tab"`, or `"escape"`.
///
/// Upstream validated these at compile time through template-literal types;
/// the port carries them as strings and matches them exactly as upstream's
/// runtime `parseKeyId` does — the [`Key`] helper builds the known
/// spellings.
pub type KeyId = String;

// =============================================================================
// Global Terminal State
// =============================================================================

/// Parsing context for terminal input.
///
/// Upstream kept two pieces of module-global state — `_kittyProtocolActive`
/// (set by `ProcessTerminal` after protocol detection) and the
/// `isWindowsTerminalSession()` environment probe consulted per call. The
/// port moves both onto this context (the crate docs carry the restatement
/// rationale); a UI session constructs one parser and passes it to the
/// keybinding manager.
#[derive(Debug)]
pub struct KeyParser {
    kitty_protocol_active: bool,
    windows_terminal_session: bool,
}

impl Default for KeyParser {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyParser {
    /// A parser with Kitty protocol off and the Windows Terminal session
    /// flag sampled from the process environment.
    ///
    /// Upstream probed `process.env` on every match; the port samples once at
    /// construction, which is equivalent because a session's terminal
    /// environment does not change mid-run. Owners override it through
    /// [`KeyParser::set_windows_terminal_session`] — Rust cannot mutate the
    /// process environment without the `unsafe` this workspace forbids.
    #[must_use]
    pub fn new() -> Self {
        Self {
            kitty_protocol_active: false,
            windows_terminal_session: is_windows_terminal_session(),
        }
    }

    /// Set the Kitty keyboard protocol state.
    /// Called by the terminal backend after detecting protocol support.
    pub const fn set_kitty_protocol_active(&mut self, active: bool) {
        self.kitty_protocol_active = active;
    }

    /// Whether Kitty keyboard protocol is currently active.
    #[must_use]
    pub const fn is_kitty_protocol_active(&self) -> bool {
        self.kitty_protocol_active
    }

    /// Override the Windows Terminal session flag, restating upstream's
    /// `WT_SESSION`/`SSH_*` environment probe (`is_windows_terminal_session`).
    pub const fn set_windows_terminal_session(&mut self, active: bool) {
        self.windows_terminal_session = active;
    }

    /// Whether raw `0x08` bytes mean Ctrl+Backspace (local Windows Terminal)
    /// or plain Backspace (everywhere else).
    ///
    /// This restates upstream's private `isWindowsTerminalSession()` probe:
    /// `WT_SESSION` must be non-empty and every `SSH_*` variable unset or
    /// empty. Kept private; sessions configure the flag through
    /// [`KeyParser::set_windows_terminal_session`].
    const fn windows_terminal_session(&self) -> bool {
        self.windows_terminal_session
    }

    /// Match input data against a key identifier string.
    ///
    /// Supported key identifiers:
    /// - Single keys: `"escape"`, `"tab"`, `"enter"`, `"backspace"`,
    ///   `"delete"`, `"home"`, `"end"`, `"space"`
    /// - Arrow keys: `"up"`, `"down"`, `"left"`, `"right"`
    /// - Ctrl combinations: `"ctrl+c"`, `"ctrl+z"`, etc.
    /// - Shift combinations: `"shift+tab"`, `"shift+enter"`
    /// - Alt combinations: `"alt+enter"`, `"alt+backspace"`
    /// - Super combinations: `"super+k"`, `"super+enter"`
    /// - Combined modifiers: `"shift+ctrl+p"`, `"ctrl+alt+x"`,
    ///   `"ctrl+super+k"`
    ///
    /// Use the [`Key`] helper for spelling: [`Key::ctrl`], [`Key::escape`],
    /// [`Key::ctrl_shift`], [`Key::super_key`].
    #[must_use]
    #[expect(
        clippy::too_many_lines,
        reason = "1:1 port of upstream's one switch statement; splitting it per key would detach the port from the upstream function it mirrors (#40)"
    )]
    pub fn matches_key(&self, data: &str, key_id: &str) -> bool {
        let Some(parsed) = parse_key_id(key_id) else {
            return false;
        };

        let modifier = parsed.modifier;
        let key = parsed.key.as_str();
        match key {
            "escape" | "esc" => {
                if modifier != 0 {
                    return false;
                }
                data == "\x1b"
                    || matches_kitty_sequence(data, CP_ESCAPE, 0)
                    || matches_modify_other_keys(data, CP_ESCAPE, 0)
            }

            "space" => {
                if !self.kitty_protocol_active {
                    if modifier == MOD_CTRL && data == "\x00" {
                        return true;
                    }
                    if modifier == MOD_ALT && data == "\x1b " {
                        return true;
                    }
                }
                if modifier == 0 {
                    return data == " "
                        || matches_kitty_sequence(data, CP_SPACE, 0)
                        || matches_modify_other_keys(data, CP_SPACE, 0);
                }
                matches_kitty_sequence(data, CP_SPACE, modifier)
                    || matches_modify_other_keys(data, CP_SPACE, modifier)
            }

            "tab" => {
                if modifier == MOD_SHIFT {
                    return data == "\x1b[Z"
                        || matches_kitty_sequence(data, CP_TAB, MOD_SHIFT)
                        || matches_modify_other_keys(data, CP_TAB, MOD_SHIFT);
                }
                if modifier == 0 {
                    return data == "\t" || matches_kitty_sequence(data, CP_TAB, 0);
                }
                matches_kitty_sequence(data, CP_TAB, modifier)
                    || matches_modify_other_keys(data, CP_TAB, modifier)
            }

            "enter" | "return" => {
                if modifier == MOD_SHIFT {
                    // CSI u sequences (standard Kitty protocol)
                    if matches_kitty_sequence(data, CP_ENTER, MOD_SHIFT)
                        || matches_kitty_sequence(data, CP_KP_ENTER, MOD_SHIFT)
                    {
                        return true;
                    }
                    // xterm modifyOtherKeys format (fallback when Kitty protocol not enabled)
                    if matches_modify_other_keys(data, CP_ENTER, MOD_SHIFT) {
                        return true;
                    }
                    // When Kitty protocol is active, legacy sequences are custom terminal mappings
                    // \x1b\r = Kitty's "map shift+enter send_text all \e\r"
                    // \n = Ghostty's "keybind = shift+enter=text:\n"
                    if self.kitty_protocol_active {
                        return data == "\x1b\r" || data == "\n";
                    }
                    return false;
                }
                if modifier == MOD_ALT {
                    // CSI u sequences (standard Kitty protocol)
                    if matches_kitty_sequence(data, CP_ENTER, MOD_ALT)
                        || matches_kitty_sequence(data, CP_KP_ENTER, MOD_ALT)
                    {
                        return true;
                    }
                    // xterm modifyOtherKeys format (fallback when Kitty protocol not enabled)
                    if matches_modify_other_keys(data, CP_ENTER, MOD_ALT) {
                        return true;
                    }
                    // \x1b\r is alt+enter only in legacy mode (no Kitty protocol)
                    // When Kitty protocol is active, alt+enter comes as CSI u sequence
                    if !self.kitty_protocol_active {
                        return data == "\x1b\r";
                    }
                    return false;
                }
                if modifier == 0 {
                    return data == "\r"
                        || (!self.kitty_protocol_active && data == "\n")
                        || data == "\x1bOM" // SS3 M (numpad enter in some terminals)
                        || matches_kitty_sequence(data, CP_ENTER, 0)
                        || matches_kitty_sequence(data, CP_KP_ENTER, 0);
                }
                matches_kitty_sequence(data, CP_ENTER, modifier)
                    || matches_kitty_sequence(data, CP_KP_ENTER, modifier)
                    || matches_modify_other_keys(data, CP_ENTER, modifier)
            }

            "backspace" => {
                if modifier == MOD_ALT {
                    if data == "\x1b\x7f" || data == "\x1b\x08" {
                        return true;
                    }
                    return matches_kitty_sequence(data, CP_BACKSPACE, MOD_ALT)
                        || matches_modify_other_keys(data, CP_BACKSPACE, MOD_ALT);
                }
                if modifier == MOD_CTRL {
                    // Legacy raw 0x08 is ambiguous: it can be Ctrl+Backspace on Windows
                    // Terminal or plain Backspace on other terminals, while also
                    // overlapping with Ctrl+H.
                    if matches_raw_backspace(data, MOD_CTRL, self.windows_terminal_session()) {
                        return true;
                    }
                    return matches_kitty_sequence(data, CP_BACKSPACE, MOD_CTRL)
                        || matches_modify_other_keys(data, CP_BACKSPACE, MOD_CTRL);
                }
                if modifier == 0 {
                    return matches_raw_backspace(data, 0, self.windows_terminal_session())
                        || matches_kitty_sequence(data, CP_BACKSPACE, 0)
                        || matches_modify_other_keys(data, CP_BACKSPACE, 0);
                }
                matches_kitty_sequence(data, CP_BACKSPACE, modifier)
                    || matches_modify_other_keys(data, CP_BACKSPACE, modifier)
            }

            "insert" => {
                if modifier == 0 {
                    return legacy_key_sequences("insert").contains(&data)
                        || matches_kitty_sequence(data, FC_INSERT, 0);
                }
                if matches_legacy_modifier_sequence(data, "insert", modifier) {
                    return true;
                }
                matches_kitty_sequence(data, FC_INSERT, modifier)
            }

            "delete" => {
                if modifier == 0 {
                    return legacy_key_sequences("delete").contains(&data)
                        || matches_kitty_sequence(data, FC_DELETE, 0);
                }
                if matches_legacy_modifier_sequence(data, "delete", modifier) {
                    return true;
                }
                matches_kitty_sequence(data, FC_DELETE, modifier)
            }

            "clear" => {
                if modifier == 0 {
                    return legacy_key_sequences("clear").contains(&data);
                }
                matches_legacy_modifier_sequence(data, "clear", modifier)
            }

            "home" => {
                if modifier == 0 {
                    return legacy_key_sequences("home").contains(&data)
                        || matches_kitty_sequence(data, FC_HOME, 0);
                }
                if matches_legacy_modifier_sequence(data, "home", modifier) {
                    return true;
                }
                matches_kitty_sequence(data, FC_HOME, modifier)
            }

            "end" => {
                if modifier == 0 {
                    return legacy_key_sequences("end").contains(&data)
                        || matches_kitty_sequence(data, FC_END, 0);
                }
                if matches_legacy_modifier_sequence(data, "end", modifier) {
                    return true;
                }
                matches_kitty_sequence(data, FC_END, modifier)
            }

            "pageup" => {
                if modifier == 0 {
                    return legacy_key_sequences("pageup").contains(&data)
                        || matches_kitty_sequence(data, FC_PAGE_UP, 0);
                }
                if matches_legacy_modifier_sequence(data, "pageup", modifier) {
                    return true;
                }
                matches_kitty_sequence(data, FC_PAGE_UP, modifier)
            }

            "pagedown" => {
                if modifier == 0 {
                    return legacy_key_sequences("pagedown").contains(&data)
                        || matches_kitty_sequence(data, FC_PAGE_DOWN, 0);
                }
                if matches_legacy_modifier_sequence(data, "pagedown", modifier) {
                    return true;
                }
                matches_kitty_sequence(data, FC_PAGE_DOWN, modifier)
            }

            "up" => {
                if modifier == MOD_ALT {
                    return data == "\x1bp" || matches_kitty_sequence(data, ARROW_UP, MOD_ALT);
                }
                if modifier == 0 {
                    return legacy_key_sequences("up").contains(&data)
                        || matches_kitty_sequence(data, ARROW_UP, 0);
                }
                if matches_legacy_modifier_sequence(data, "up", modifier) {
                    return true;
                }
                matches_kitty_sequence(data, ARROW_UP, modifier)
            }

            "down" => {
                if modifier == MOD_ALT {
                    return data == "\x1bn" || matches_kitty_sequence(data, ARROW_DOWN, MOD_ALT);
                }
                if modifier == 0 {
                    return legacy_key_sequences("down").contains(&data)
                        || matches_kitty_sequence(data, ARROW_DOWN, 0);
                }
                if matches_legacy_modifier_sequence(data, "down", modifier) {
                    return true;
                }
                matches_kitty_sequence(data, ARROW_DOWN, modifier)
            }

            "left" => {
                if modifier == MOD_ALT {
                    return data == "\x1b[1;3D"
                        || (!self.kitty_protocol_active && data == "\x1bB")
                        || data == "\x1bb"
                        || matches_kitty_sequence(data, ARROW_LEFT, MOD_ALT);
                }
                if modifier == MOD_CTRL {
                    return data == "\x1b[1;5D"
                        || matches_legacy_modifier_sequence(data, "left", MOD_CTRL)
                        || matches_kitty_sequence(data, ARROW_LEFT, MOD_CTRL);
                }
                if modifier == 0 {
                    return legacy_key_sequences("left").contains(&data)
                        || matches_kitty_sequence(data, ARROW_LEFT, 0);
                }
                if matches_legacy_modifier_sequence(data, "left", modifier) {
                    return true;
                }
                matches_kitty_sequence(data, ARROW_LEFT, modifier)
            }

            "right" => {
                if modifier == MOD_ALT {
                    return data == "\x1b[1;3C"
                        || (!self.kitty_protocol_active && data == "\x1bF")
                        || data == "\x1bf"
                        || matches_kitty_sequence(data, ARROW_RIGHT, MOD_ALT);
                }
                if modifier == MOD_CTRL {
                    return data == "\x1b[1;5C"
                        || matches_legacy_modifier_sequence(data, "right", MOD_CTRL)
                        || matches_kitty_sequence(data, ARROW_RIGHT, MOD_CTRL);
                }
                if modifier == 0 {
                    return legacy_key_sequences("right").contains(&data)
                        || matches_kitty_sequence(data, ARROW_RIGHT, 0);
                }
                if matches_legacy_modifier_sequence(data, "right", modifier) {
                    return true;
                }
                matches_kitty_sequence(data, ARROW_RIGHT, modifier)
            }

            "f1" | "f2" | "f3" | "f4" | "f5" | "f6" | "f7" | "f8" | "f9" | "f10" | "f11"
            | "f12" => {
                if modifier != 0 {
                    return false;
                }
                legacy_key_sequences(key).contains(&data)
            }

            // Single letter/digit keys and symbols
            _ => {
                let Some(key_char) = printable_key_char(key) else {
                    return false;
                };
                let codepoint = i64::from(key_char as u32);
                let raw_ctrl = raw_ctrl_char(key_char);
                let is_letter = key_char.is_ascii_lowercase();
                let is_digit = key_char.is_ascii_digit();

                if modifier == MOD_CTRL + MOD_ALT
                    && !self.kitty_protocol_active
                    && raw_ctrl.is_some_and(|rc| data == format!("\x1b{rc}"))
                {
                    // Legacy: ctrl+alt+key is ESC followed by the control character.
                    // If that legacy form does not match, continue so CSI-u and
                    // modifyOtherKeys sequences from tmux can still be recognized.
                    return true;
                }

                if modifier == MOD_ALT
                    && !self.kitty_protocol_active
                    && (is_letter || is_digit || SYMBOL_KEYS.contains(&key_char))
                    && data == format!("\x1b{key_char}")
                {
                    // Legacy: alt+printable key is ESC followed by the key
                    return true;
                }

                if modifier == MOD_CTRL {
                    // Legacy: ctrl+key sends the control character
                    if raw_ctrl.is_some_and(|rc| data == rc.to_string()) {
                        return true;
                    }
                    return matches_kitty_sequence(data, codepoint, MOD_CTRL)
                        || matches_printable_modify_other_keys(data, codepoint, MOD_CTRL);
                }

                if modifier == MOD_SHIFT + MOD_CTRL {
                    return matches_kitty_sequence(data, codepoint, MOD_SHIFT + MOD_CTRL)
                        || matches_printable_modify_other_keys(
                            data,
                            codepoint,
                            MOD_SHIFT + MOD_CTRL,
                        );
                }

                if modifier == MOD_SHIFT {
                    // Legacy: shift+letter produces uppercase
                    if is_letter && data == key_char.to_ascii_uppercase().to_string() {
                        return true;
                    }
                    return matches_kitty_sequence(data, codepoint, MOD_SHIFT)
                        || matches_printable_modify_other_keys(data, codepoint, MOD_SHIFT);
                }

                if modifier != 0 {
                    return matches_kitty_sequence(data, codepoint, modifier)
                        || matches_printable_modify_other_keys(data, codepoint, modifier);
                }

                // Check both raw char and Kitty sequence (needed for release events)
                data == key_char.to_string() || matches_kitty_sequence(data, codepoint, 0)
            }
        }
    }

    /// Parse input data and return the key identifier if recognized.
    ///
    /// Returns an identifier such as `"ctrl+c"`, or `None` when the data is
    /// not a recognized key event.
    #[must_use]
    #[expect(
        clippy::too_many_lines,
        reason = "1:1 port of upstream's one ordered if-chain; splitting it per branch would detach the port from the upstream function it mirrors (#40)"
    )]
    pub fn parse_key(&self, data: &str) -> Option<KeyId> {
        if let Some(kitty) = parse_kitty_sequence(data) {
            return format_parsed_key(kitty.codepoint, kitty.modifier, kitty.base_layout_key);
        }

        if let Some((codepoint, modifier)) = parse_modify_other_keys_sequence(data) {
            return format_parsed_key(codepoint, modifier, None);
        }

        // Mode-aware legacy sequences
        // When Kitty protocol is active, ambiguous sequences are interpreted as custom terminal mappings:
        // - \x1b\r = shift+enter (Kitty mapping), not alt+enter
        // - \n = shift+enter (Ghostty mapping)
        if self.kitty_protocol_active && (data == "\x1b\r" || data == "\n") {
            return Some("shift+enter".to_string());
        }

        if let Some(key_id) = legacy_sequence_key_id(data) {
            return Some(key_id.to_string());
        }

        // Legacy sequences (used when Kitty protocol is not active, or for unambiguous sequences)
        if data == "\x1b" {
            return Some("escape".to_string());
        }
        if data == "\x1c" {
            return Some("ctrl+\\".to_string());
        }
        if data == "\x1d" {
            return Some("ctrl+]".to_string());
        }
        if data == "\x1f" {
            return Some("ctrl+-".to_string());
        }
        if data == "\x1b\x1b" {
            return Some("ctrl+alt+[".to_string());
        }
        if data == "\x1b\x1c" {
            return Some("ctrl+alt+\\".to_string());
        }
        if data == "\x1b\x1d" {
            return Some("ctrl+alt+]".to_string());
        }
        if data == "\x1b\x1f" {
            return Some("ctrl+alt+-".to_string());
        }
        if data == "\t" {
            return Some("tab".to_string());
        }
        if data == "\r" || (!self.kitty_protocol_active && data == "\n") || data == "\x1bOM" {
            return Some("enter".to_string());
        }
        if data == "\x00" {
            return Some("ctrl+space".to_string());
        }
        if data == " " {
            return Some("space".to_string());
        }
        if data == "\x7f" {
            return Some("backspace".to_string());
        }
        if data == "\x08" {
            return Some(
                if self.windows_terminal_session() {
                    "ctrl+backspace"
                } else {
                    "backspace"
                }
                .to_string(),
            );
        }
        if data == "\x1b[Z" {
            return Some("shift+tab".to_string());
        }
        if !self.kitty_protocol_active && data == "\x1b\r" {
            return Some("alt+enter".to_string());
        }
        if !self.kitty_protocol_active && data == "\x1b " {
            return Some("alt+space".to_string());
        }
        if data == "\x1b\x7f" || data == "\x1b\x08" {
            return Some("alt+backspace".to_string());
        }
        if !self.kitty_protocol_active && data == "\x1bB" {
            return Some("alt+left".to_string());
        }
        if !self.kitty_protocol_active && data == "\x1bF" {
            return Some("alt+right".to_string());
        }
        if !self.kitty_protocol_active
            && data.chars().count() == 2
            && data.starts_with('\x1b')
            && let Some(second) = data.chars().nth(1)
        {
            let code = u32::from(second);
            if (1..=26).contains(&code) {
                // ESC followed by a control character
                return Some(format!("ctrl+alt+{}", control_to_letter(code)));
            }
            // Legacy alt+letter/digit/symbol (ESC followed by the key)
            if (97..=122).contains(&code)
                || (48..=57).contains(&code)
                || SYMBOL_KEYS.contains(&second)
            {
                return Some(format!("alt+{second}"));
            }
        }
        if data == "\x1b[A" {
            return Some("up".to_string());
        }
        if data == "\x1b[B" {
            return Some("down".to_string());
        }
        if data == "\x1b[C" {
            return Some("right".to_string());
        }
        if data == "\x1b[D" {
            return Some("left".to_string());
        }
        if data == "\x1b[H" || data == "\x1bOH" {
            return Some("home".to_string());
        }
        if data == "\x1b[F" || data == "\x1bOF" {
            return Some("end".to_string());
        }
        // Upstream also names "\x1b[3~"/"\x1b[5~"/"\x1b[6~" here, but those
        // sequences never reach the chain — the functional-key parse above
        // claims them first — so the duplicates are dropped (#40).

        // Raw Ctrl+letter
        if data.chars().count() == 1 {
            let code = u32::from(data.chars().next()?);
            if (1..=26).contains(&code) {
                return Some(format!("ctrl+{}", control_to_letter(code)));
            }
            if (32..=126).contains(&code) {
                return Some(data.to_string());
            }
        }

        None
    }
}

/// Restates upstream's `isWindowsTerminalSession()` environment probe: a local
/// Windows Terminal session carries a non-empty `WT_SESSION` and no SSH
/// variables. The [`KeyParser`] samples this at construction.
fn is_windows_terminal_session() -> bool {
    let wt_session = std::env::var("WT_SESSION").is_ok_and(|value| !value.is_empty());
    let ssh = ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"]
        .iter()
        .any(|name| std::env::var(name).is_ok_and(|value| !value.is_empty()));
    wt_session && !ssh
}

// =============================================================================
// Key Identifiers
// =============================================================================

/// The 31 symbol keys that can carry modifiers, as the wire spells them.
const SYMBOL_KEYS: [char; 31] = [
    '`', '-', '=', '[', ']', '\\', ';', '\'', ',', '.', '/', '!', '@', '#', '$', '%', '^', '&',
    '*', '(', ')', '_', '+', '|', '~', '{', '}', ':', '<', '>', '?',
];

/// Helper for building key identifiers.
///
/// Usage:
/// - [`Key::escape`], [`Key::enter`], [`Key::tab`], ... for special keys
/// - [`Key::backtick`], [`Key::comma`], [`Key::period`], ... for symbol keys
/// - [`Key::ctrl`], [`Key::alt`], [`Key::super_key`] for single modifiers
/// - [`Key::ctrl_shift`], [`Key::ctrl_alt`], [`Key::ctrl_super`] for combined
///   modifiers
#[derive(Debug, Clone, Copy)]
pub struct Key;

impl Key {
    // Special keys
    /// The Escape key identifier.
    #[must_use]
    pub fn escape() -> KeyId {
        "escape".to_string()
    }

    /// Alias identifier for Escape, as upstream's `Key.esc`.
    #[must_use]
    pub fn esc() -> KeyId {
        "esc".to_string()
    }

    /// The Enter key.
    #[must_use]
    pub fn enter() -> KeyId {
        "enter".to_string()
    }

    /// Alias identifier for Enter, as the wire spells it `"return"`.
    #[must_use]
    pub fn r#return() -> KeyId {
        "return".to_string()
    }

    /// The Tab key.
    #[must_use]
    pub fn tab() -> KeyId {
        "tab".to_string()
    }

    /// The Space key.
    #[must_use]
    pub fn space() -> KeyId {
        "space".to_string()
    }

    /// The Backspace key.
    #[must_use]
    pub fn backspace() -> KeyId {
        "backspace".to_string()
    }

    /// The Delete key.
    #[must_use]
    pub fn delete() -> KeyId {
        "delete".to_string()
    }

    /// The Insert key.
    #[must_use]
    pub fn insert() -> KeyId {
        "insert".to_string()
    }

    /// The Clear key (rxvt).
    #[must_use]
    pub fn clear() -> KeyId {
        "clear".to_string()
    }

    /// The Home key.
    #[must_use]
    pub fn home() -> KeyId {
        "home".to_string()
    }

    /// The End key.
    #[must_use]
    pub fn end() -> KeyId {
        "end".to_string()
    }

    /// The Page Up key.
    #[must_use]
    pub fn page_up() -> KeyId {
        "pageUp".to_string()
    }

    /// The Page Down key.
    #[must_use]
    pub fn page_down() -> KeyId {
        "pageDown".to_string()
    }

    /// The Up arrow key.
    #[must_use]
    pub fn up() -> KeyId {
        "up".to_string()
    }

    /// The Down key.
    #[must_use]
    pub fn down() -> KeyId {
        "down".to_string()
    }

    /// The Left key.
    #[must_use]
    pub fn left() -> KeyId {
        "left".to_string()
    }

    /// The Right key.
    #[must_use]
    pub fn right() -> KeyId {
        "right".to_string()
    }

    /// F1.
    #[must_use]
    pub fn f1() -> KeyId {
        "f1".to_string()
    }

    /// F2.
    #[must_use]
    pub fn f2() -> KeyId {
        "f2".to_string()
    }

    /// F3.
    #[must_use]
    pub fn f3() -> KeyId {
        "f3".to_string()
    }

    /// F4.
    #[must_use]
    pub fn f4() -> KeyId {
        "f4".to_string()
    }

    /// F5.
    #[must_use]
    pub fn f5() -> KeyId {
        "f5".to_string()
    }

    /// F6.
    #[must_use]
    pub fn f6() -> KeyId {
        "f6".to_string()
    }

    /// F7.
    #[must_use]
    pub fn f7() -> KeyId {
        "f7".to_string()
    }

    /// F8.
    #[must_use]
    pub fn f8() -> KeyId {
        "f8".to_string()
    }

    /// F9.
    #[must_use]
    pub fn f9() -> KeyId {
        "f9".to_string()
    }

    /// F10.
    #[must_use]
    pub fn f10() -> KeyId {
        "f10".to_string()
    }

    /// F11.
    #[must_use]
    pub fn f11() -> KeyId {
        "f11".to_string()
    }

    /// F12.
    #[must_use]
    pub fn f12() -> KeyId {
        "f12".to_string()
    }

    // Symbol keys
    /// The backtick key.
    #[must_use]
    pub fn backtick() -> KeyId {
        "`".to_string()
    }

    /// The hyphen key.
    #[must_use]
    pub fn hyphen() -> KeyId {
        "-".to_string()
    }

    /// The equals key.
    #[must_use]
    pub fn equals() -> KeyId {
        "=".to_string()
    }

    /// The left bracket key.
    #[must_use]
    pub fn leftbracket() -> KeyId {
        "[".to_string()
    }

    /// The right bracket key.
    #[must_use]
    pub fn rightbracket() -> KeyId {
        "]".to_string()
    }

    /// The backslash key.
    #[must_use]
    pub fn backslash() -> KeyId {
        "\\".to_string()
    }

    /// The semicolon key.
    #[must_use]
    pub fn semicolon() -> KeyId {
        ";".to_string()
    }

    /// The quote key.
    #[must_use]
    pub fn quote() -> KeyId {
        "'".to_string()
    }

    /// The comma key.
    #[must_use]
    pub fn comma() -> KeyId {
        ",".to_string()
    }

    /// The period key.
    #[must_use]
    pub fn period() -> KeyId {
        ".".to_string()
    }

    /// The slash key.
    #[must_use]
    pub fn slash() -> KeyId {
        "/".to_string()
    }

    /// The exclamation key.
    #[must_use]
    pub fn exclamation() -> KeyId {
        "!".to_string()
    }

    /// The at key.
    #[must_use]
    pub fn at() -> KeyId {
        "@".to_string()
    }

    /// The hash key.
    #[must_use]
    pub fn hash() -> KeyId {
        "#".to_string()
    }

    /// The dollar key.
    #[must_use]
    pub fn dollar() -> KeyId {
        "$".to_string()
    }

    /// The percent key.
    #[must_use]
    pub fn percent() -> KeyId {
        "%".to_string()
    }

    /// The caret key.
    #[must_use]
    pub fn caret() -> KeyId {
        "^".to_string()
    }

    /// The ampersand key.
    #[must_use]
    pub fn ampersand() -> KeyId {
        "&".to_string()
    }

    /// The asterisk key.
    #[must_use]
    pub fn asterisk() -> KeyId {
        "*".to_string()
    }

    /// The left parenthesis key.
    #[must_use]
    pub fn leftparen() -> KeyId {
        "(".to_string()
    }

    /// The right paren key.
    #[must_use]
    pub fn rightparen() -> KeyId {
        ")".to_string()
    }

    /// The underscore key.
    #[must_use]
    pub fn underscore() -> KeyId {
        "_".to_string()
    }

    /// The plus key.
    #[must_use]
    pub fn plus() -> KeyId {
        "+".to_string()
    }

    /// The pipe key.
    #[must_use]
    pub fn pipe() -> KeyId {
        "|".to_string()
    }

    /// The tilde key.
    #[must_use]
    pub fn tilde() -> KeyId {
        "~".to_string()
    }

    /// The left brace key.
    #[must_use]
    pub fn leftbrace() -> KeyId {
        "{".to_string()
    }

    /// The right brace key.
    #[must_use]
    pub fn rightbrace() -> KeyId {
        "}".to_string()
    }

    /// The colon key.
    #[must_use]
    pub fn colon() -> KeyId {
        ":".to_string()
    }

    /// The less-than key.
    #[must_use]
    pub fn lessthan() -> KeyId {
        "<".to_string()
    }

    /// The greater-than key.
    #[must_use]
    pub fn greaterthan() -> KeyId {
        ">".to_string()
    }

    /// The question key.
    #[must_use]
    pub fn question() -> KeyId {
        "?".to_string()
    }

    // Single modifiers
    /// `ctrl+<key>`.
    #[must_use]
    pub fn ctrl(key: impl AsRef<str>) -> KeyId {
        format!("ctrl+{}", key.as_ref())
    }

    /// `shift+<key>`.
    #[must_use]
    pub fn shift(key: impl AsRef<str>) -> KeyId {
        format!("shift+{}", key.as_ref())
    }

    /// `alt+<key>`.
    #[must_use]
    pub fn alt(key: impl AsRef<str>) -> KeyId {
        format!("alt+{}", key.as_ref())
    }

    /// `super+<key>`, upstream's `Key.super`.
    #[must_use]
    pub fn super_key(key: impl AsRef<str>) -> KeyId {
        format!("super+{}", key.as_ref())
    }

    // Combined modifiers
    /// `ctrl+shift+<key>`.
    #[must_use]
    pub fn ctrl_shift(key: impl AsRef<str>) -> KeyId {
        format!("ctrl+shift+{}", key.as_ref())
    }

    /// `shift+ctrl+<key>`.
    #[must_use]
    pub fn shift_ctrl(key: impl AsRef<str>) -> KeyId {
        format!("shift+ctrl+{}", key.as_ref())
    }

    /// `ctrl+alt+<key>`.
    #[must_use]
    pub fn ctrl_alt(key: impl AsRef<str>) -> KeyId {
        format!("ctrl+alt+{}", key.as_ref())
    }

    /// `alt+ctrl+<key>`.
    #[must_use]
    pub fn alt_ctrl(key: impl AsRef<str>) -> KeyId {
        format!("alt+ctrl+{}", key.as_ref())
    }

    /// `shift+alt+<key>`.
    #[must_use]
    pub fn shift_alt(key: impl AsRef<str>) -> KeyId {
        format!("shift+alt+{}", key.as_ref())
    }

    /// `alt+shift+<key>`.
    #[must_use]
    pub fn alt_shift(key: impl AsRef<str>) -> KeyId {
        format!("alt+shift+{}", key.as_ref())
    }

    /// `ctrl+super+<key>`.
    #[must_use]
    pub fn ctrl_super(key: impl AsRef<str>) -> KeyId {
        format!("ctrl+super+{}", key.as_ref())
    }

    /// `super+ctrl+<key>`.
    #[must_use]
    pub fn super_ctrl(key: impl AsRef<str>) -> KeyId {
        format!("super+ctrl+{}", key.as_ref())
    }

    /// `shift+super+<key>`.
    #[must_use]
    pub fn shift_super(key: impl AsRef<str>) -> KeyId {
        format!("shift+super+{}", key.as_ref())
    }

    /// `super+shift+<key>`.
    #[must_use]
    pub fn super_shift(key: impl AsRef<str>) -> KeyId {
        format!("super+shift+{}", key.as_ref())
    }

    /// `alt+super+<key>`.
    #[must_use]
    pub fn alt_super(key: impl AsRef<str>) -> KeyId {
        format!("alt+super+{}", key.as_ref())
    }

    /// `super+alt+<key>`.
    #[must_use]
    pub fn super_alt(key: impl AsRef<str>) -> KeyId {
        format!("super+alt+{}", key.as_ref())
    }

    // Triple modifiers
    /// `ctrl+shift+alt+<key>`.
    #[must_use]
    pub fn ctrl_shift_alt(key: impl AsRef<str>) -> KeyId {
        format!("ctrl+shift+alt+{}", key.as_ref())
    }

    /// `ctrl+shift+super+<key>`.
    #[must_use]
    pub fn ctrl_shift_super(key: impl AsRef<str>) -> KeyId {
        format!("ctrl+shift+super+{}", key.as_ref())
    }
}

// =============================================================================
// Constants
// =============================================================================

const MOD_SHIFT: i64 = 1;
const MOD_ALT: i64 = 2;
const MOD_CTRL: i64 = 4;
const MOD_SUPER: i64 = 8;

const LOCK_MASK: i64 = 64 + 128; // Caps Lock + Num Lock

const CP_ESCAPE: i64 = 27;
const CP_TAB: i64 = 9;
const CP_ENTER: i64 = 13;
const CP_SPACE: i64 = 32;
const CP_BACKSPACE: i64 = 127;
const CP_KP_ENTER: i64 = 57414; // Numpad Enter (Kitty protocol)

const ARROW_UP: i64 = -1;
const ARROW_DOWN: i64 = -2;
const ARROW_RIGHT: i64 = -3;
const ARROW_LEFT: i64 = -4;

const FC_DELETE: i64 = -10;
const FC_INSERT: i64 = -11;
const FC_PAGE_UP: i64 = -12;
const FC_PAGE_DOWN: i64 = -13;
const FC_HOME: i64 = -14;
const FC_END: i64 = -15;

/// Kitty's keypad functional keys sent as private codepoints (57399..57426);
/// the port normalizes them to the equivalent main-keyboard codepoint.
const fn normalize_kitty_functional_codepoint(codepoint: i64) -> i64 {
    match codepoint {
        57399 => 48, // KP_0 -> 0
        57400 => 49, // KP_1 -> 1
        57401 => 50, // KP_2 -> 2
        57402 => 51, // KP_3 -> 3
        57403 => 52, // KP_4 -> 4
        57404 => 53, // KP_5 -> 5
        57405 => 54, // KP_6 -> 6
        57406 => 55, // KP_7 -> 7
        57407 => 56, // KP_8 -> 8
        57408 => 57, // KP_9 -> 9
        57409 => 46, // KP_DECIMAL -> .
        57410 => 47, // KP_DIVIDE -> /
        57411 => 42, // KP_MULTIPLY -> *
        57412 => 45, // KP_SUBTRACT -> -
        57413 => 43, // KP_ADD -> +
        57415 => 61, // KP_EQUAL -> =
        57416 => 44, // KP_SEPARATOR -> ,
        57417 => ARROW_LEFT,
        57418 => ARROW_RIGHT,
        57419 => ARROW_UP,
        57420 => ARROW_DOWN,
        57421 => FC_PAGE_UP,
        57422 => FC_PAGE_DOWN,
        57423 => FC_HOME,
        57424 => FC_END,
        57425 => FC_INSERT,
        57426 => FC_DELETE,
        _ => codepoint,
    }
}

/// With Shift held, Kitty reports the uppercase letter identity; key ids are
/// lowercase, so fold `A`..`Z` back to `a`..`z`.
const fn normalize_shifted_letter_identity_codepoint(codepoint: i64, modifier: i64) -> i64 {
    let effective_modifier = modifier & !LOCK_MASK;
    if (effective_modifier & MOD_SHIFT) != 0 && codepoint >= 65 && codepoint <= 90 {
        return codepoint + 32;
    }
    codepoint
}

fn legacy_key_sequences(key: &str) -> &'static [&'static str] {
    match key {
        "up" => &["\x1b[A", "\x1bOA"],
        "down" => &["\x1b[B", "\x1bOB"],
        "right" => &["\x1b[C", "\x1bOC"],
        "left" => &["\x1b[D", "\x1bOD"],
        "home" => &["\x1b[H", "\x1bOH", "\x1b[1~", "\x1b[7~"],
        "end" => &["\x1b[F", "\x1bOF", "\x1b[4~", "\x1b[8~"],
        "insert" => &["\x1b[2~"],
        "delete" => &["\x1b[3~"],
        "pageup" => &["\x1b[5~", "\x1b[[5~"],
        "pagedown" => &["\x1b[6~", "\x1b[[6~"],
        "clear" => &["\x1b[E", "\x1bOE"],
        "f1" => &["\x1bOP", "\x1b[11~", "\x1b[[A"],
        "f2" => &["\x1bOQ", "\x1b[12~", "\x1b[[B"],
        "f3" => &["\x1bOR", "\x1b[13~", "\x1b[[C"],
        "f4" => &["\x1bOS", "\x1b[14~", "\x1b[[D"],
        "f5" => &["\x1b[15~", "\x1b[[E"],
        "f6" => &["\x1b[17~"],
        "f7" => &["\x1b[18~"],
        "f8" => &["\x1b[19~"],
        "f9" => &["\x1b[20~"],
        "f10" => &["\x1b[21~"],
        "f11" => &["\x1b[23~"],
        "f12" => &["\x1b[24~"],
        _ => &[],
    }
}

fn legacy_shift_sequences(key: &str) -> &'static [&'static str] {
    match key {
        "up" => &["\x1b[a"],
        "down" => &["\x1b[b"],
        "right" => &["\x1b[c"],
        "left" => &["\x1b[d"],
        "clear" => &["\x1b[e"],
        "insert" => &["\x1b[2$"],
        "delete" => &["\x1b[3$"],
        "pageup" => &["\x1b[5$"],
        "pagedown" => &["\x1b[6$"],
        "home" => &["\x1b[7$"],
        "end" => &["\x1b[8$"],
        _ => &[],
    }
}

fn legacy_ctrl_sequences(key: &str) -> &'static [&'static str] {
    match key {
        "up" => &["\x1bOa"],
        "down" => &["\x1bOb"],
        "right" => &["\x1bOc"],
        "left" => &["\x1bOd"],
        "clear" => &["\x1bOe"],
        "insert" => &["\x1b[2^"],
        "delete" => &["\x1b[3^"],
        "pageup" => &["\x1b[5^"],
        "pagedown" => &["\x1b[6^"],
        "home" => &["\x1b[7^"],
        "end" => &["\x1b[8^"],
        _ => &[],
    }
}

/// The legacy CSI/SS3 sequences `parse_key` names directly; mirrors upstream's
/// `LEGACY_SEQUENCE_KEY_IDS` table.
fn legacy_sequence_key_id(data: &str) -> Option<&'static str> {
    match data {
        "\x1bOA" => Some("up"),
        "\x1bOB" => Some("down"),
        "\x1bOC" => Some("right"),
        "\x1bOD" => Some("left"),
        "\x1bOH" => Some("home"),
        "\x1bOF" => Some("end"),
        "\x1b[E" | "\x1bOE" => Some("clear"),
        "\x1bOe" => Some("ctrl+clear"),
        "\x1b[e" => Some("shift+clear"),
        "\x1b[2~" => Some("insert"),
        "\x1b[2$" => Some("shift+insert"),
        "\x1b[2^" => Some("ctrl+insert"),
        "\x1b[3$" => Some("shift+delete"),
        "\x1b[3^" => Some("ctrl+delete"),
        "\x1b[[5~" => Some("pageUp"),
        "\x1b[[6~" => Some("pageDown"),
        "\x1b[a" => Some("shift+up"),
        "\x1b[b" => Some("shift+down"),
        "\x1b[c" => Some("shift+right"),
        "\x1b[d" => Some("shift+left"),
        "\x1bOa" => Some("ctrl+up"),
        "\x1bOb" => Some("ctrl+down"),
        "\x1bOc" => Some("ctrl+right"),
        "\x1bOd" => Some("ctrl+left"),
        "\x1b[5$" => Some("shift+pageUp"),
        "\x1b[6$" => Some("shift+pageDown"),
        "\x1b[7$" => Some("shift+home"),
        "\x1b[8$" => Some("shift+end"),
        "\x1b[5^" => Some("ctrl+pageUp"),
        "\x1b[6^" => Some("ctrl+pageDown"),
        "\x1b[7^" => Some("ctrl+home"),
        "\x1b[8^" => Some("ctrl+end"),
        "\x1bOP" | "\x1b[11~" | "\x1b[[A" => Some("f1"),
        "\x1bOQ" | "\x1b[12~" | "\x1b[[B" => Some("f2"),
        "\x1bOR" | "\x1b[13~" | "\x1b[[C" => Some("f3"),
        "\x1bOS" | "\x1b[14~" | "\x1b[[D" => Some("f4"),
        "\x1b[[E" | "\x1b[15~" => Some("f5"),
        "\x1b[17~" => Some("f6"),
        "\x1b[18~" => Some("f7"),
        "\x1b[19~" => Some("f8"),
        "\x1b[20~" => Some("f9"),
        "\x1b[21~" => Some("f10"),
        "\x1b[23~" => Some("f11"),
        "\x1b[24~" => Some("f12"),
        "\x1bb" => Some("alt+left"),
        "\x1bf" => Some("alt+right"),
        "\x1bp" => Some("alt+up"),
        "\x1bn" => Some("alt+down"),
        _ => None,
    }
}

fn matches_legacy_sequence(data: &str, sequences: &[&str]) -> bool {
    sequences.contains(&data)
}

fn matches_legacy_modifier_sequence(data: &str, key: &str, modifier: i64) -> bool {
    if modifier == MOD_SHIFT {
        return matches_legacy_sequence(data, legacy_shift_sequences(key));
    }
    if modifier == MOD_CTRL {
        return matches_legacy_sequence(data, legacy_ctrl_sequences(key));
    }
    false
}

// =============================================================================
// Kitty Protocol Parsing
// =============================================================================

/// Event types from Kitty keyboard protocol (flag 2):
/// 1 = key press, 2 = key repeat, 3 = key release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyEventType {
    /// Key press (Kitty event type 1 or absent).
    Press,
    /// Key repeat (Kitty event type 2).
    Repeat,
    /// Key release (Kitty event type 3).
    Release,
}

/// A parsed Kitty-style escape sequence: a CSI `u` event, a modified arrow or
/// home/end, a functional `~` key, or an xterm `modifyOtherKeys` report.
#[derive(Debug, Clone, Copy)]
struct ParsedKittySequence {
    codepoint: i64,
    /// Shifted version of the key (when shift is pressed).
    shifted_key: Option<i64>,
    /// Key in standard PC-101 layout (for non-Latin layouts).
    base_layout_key: Option<i64>,
    modifier: i64,
    #[allow(
        dead_code,
        reason = "parsed for parity with upstream; consumers read the data patterns"
    )]
    event_type: KeyEventType,
}

/// Check if the parsed key event was a key release.
///
/// Only meaningful when Kitty keyboard protocol with flag 2 is active. Bracketed
/// paste content never counts, even when it carries `:3F`-shaped runs (a
/// bluetooth MAC address like `90:62:3F:A5` inside a paste); the terminal
/// backend re-wraps paste with bracketed-paste markers, so pasted data always
/// contains `\x1b[200~`.
#[must_use]
pub fn is_key_release(data: &str) -> bool {
    // Don't treat bracketed paste content as key release, even if it contains
    // patterns like ":3F" (e.g., bluetooth MAC addresses like "90:62:3F:A5").
    if data.contains("\x1b[200~") {
        return false;
    }

    // Quick check: release events with flag 2 contain ":3"
    // Format: \x1b[<codepoint>;<modifier>:3u
    [":3u", ":3~", ":3A", ":3B", ":3C", ":3D", ":3H", ":3F"]
        .iter()
        .any(|suffix| data.contains(suffix))
}

/// Check if the parsed key event was a key repeat.
///
/// Only meaningful when Kitty keyboard protocol with flag 2 is active. See
/// [`is_key_release`] for the bracketed-paste exclusion.
#[must_use]
pub fn is_key_repeat(data: &str) -> bool {
    if data.contains("\x1b[200~") {
        return false;
    }
    [":2u", ":2~", ":2A", ":2B", ":2C", ":2D", ":2H", ":2F"]
        .iter()
        .any(|suffix| data.contains(suffix))
}

const fn parse_event_type(event_type: Option<i64>) -> KeyEventType {
    match event_type {
        Some(2) => KeyEventType::Repeat,
        Some(3) => KeyEventType::Release,
        _ => KeyEventType::Press,
    }
}

/// Scan a run of ASCII digits, saturating on overflow. Returns `None` when no
/// digit is present. Upstream's `parseInt` yields an out-of-range float for
/// absurdly long runs; the saturating value lands in the same "never matches a
/// real key" bucket.
fn scan_digits(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    let mut value: i64 = 0;
    let mut any = false;
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        any = true;
        value = value
            .saturating_mul(10)
            .saturating_add(i64::from(bytes[*pos] - b'0'));
        *pos += 1;
    }
    if any { Some(value) } else { None }
}

/// Restates upstream's `CSI u` regex
/// `^\x1b\[(\d+)(?::(\d*))?(?::(\d+))?(?:;(\d+))?(?::(\d+))?u$`:
/// `<codepoint>[:<shifted>[:<base>]][;<mod>][:<event>]u`, where the shifted
/// slot allows an empty digit run.
fn parse_csi_u(data: &str) -> Option<ParsedKittySequence> {
    let bytes = data.as_bytes();
    if bytes.len() < 4 || !bytes.starts_with(b"\x1b[") || bytes[bytes.len() - 1] != b'u' {
        return None;
    }
    let mut pos = 2;
    let codepoint = scan_digits(bytes, &mut pos)?;

    let shifted_key = if pos < bytes.len() && bytes[pos] == b':' {
        pos += 1;
        scan_digits(bytes, &mut pos)
    } else {
        None
    };
    let base_layout_key = if pos < bytes.len() && bytes[pos] == b':' {
        pos += 1;
        Some(scan_digits(bytes, &mut pos)?)
    } else {
        None
    };
    let mod_value = if pos < bytes.len() && bytes[pos] == b';' {
        pos += 1;
        scan_digits(bytes, &mut pos)?
    } else {
        1
    };
    let event_type = if pos < bytes.len() && bytes[pos] == b':' {
        pos += 1;
        parse_event_type(Some(scan_digits(bytes, &mut pos)?))
    } else {
        KeyEventType::Press
    };
    if pos != bytes.len() - 1 {
        return None;
    }

    Some(ParsedKittySequence {
        codepoint,
        shifted_key,
        base_layout_key,
        modifier: mod_value - 1,
        event_type,
    })
}

/// Restates upstream's arrow regex `^\x1b\[1;(\d+)(?::(\d+))?([ABCD])$`.
fn parse_kitty_arrow(data: &str) -> Option<(i64, i64, KeyEventType)> {
    let bytes = data.as_bytes();
    if !bytes.starts_with(b"\x1b[1;") {
        return None;
    }
    let mut pos = 4;
    let mod_value = scan_digits(bytes, &mut pos)?;
    let event = if pos < bytes.len() && bytes[pos] == b':' {
        pos += 1;
        parse_event_type(Some(scan_digits(bytes, &mut pos)?))
    } else {
        KeyEventType::Press
    };
    let codepoint = match *bytes.get(pos)? {
        b'A' => ARROW_UP,
        b'B' => ARROW_DOWN,
        b'C' => ARROW_RIGHT,
        b'D' => ARROW_LEFT,
        _ => return None,
    };
    if pos + 1 != bytes.len() {
        return None;
    }
    Some((codepoint, mod_value - 1, event))
}

/// Restates upstream's functional-key regex `^\x1b\[(\d+)(?:;(\d+))?(?::(\d+))?~$`.
fn parse_kitty_functional(data: &str) -> Option<(i64, i64, KeyEventType)> {
    let bytes = data.as_bytes();
    if bytes.len() < 4 || !bytes.starts_with(b"\x1b[") || bytes[bytes.len() - 1] != b'~' {
        return None;
    }
    let mut pos = 2;
    let key_num = scan_digits(bytes, &mut pos)?;
    let mod_value = if pos < bytes.len() && bytes[pos] == b';' {
        pos += 1;
        scan_digits(bytes, &mut pos)?
    } else {
        1
    };
    let event = if pos < bytes.len() && bytes[pos] == b':' {
        pos += 1;
        parse_event_type(Some(scan_digits(bytes, &mut pos)?))
    } else {
        KeyEventType::Press
    };
    if pos + 1 != bytes.len() || bytes[pos] != b'~' {
        return None;
    }
    let codepoint = match key_num {
        2 => FC_INSERT,
        3 => FC_DELETE,
        5 => FC_PAGE_UP,
        6 => FC_PAGE_DOWN,
        7 => FC_HOME,
        8 => FC_END,
        _ => return None,
    };
    Some((codepoint, mod_value - 1, event))
}

/// Restates upstream's home/end regex `^\x1b\[1;(\d+)(?::(\d+))?([HF])$`.
fn parse_kitty_home_end(data: &str) -> Option<(i64, i64, KeyEventType)> {
    let bytes = data.as_bytes();
    if !bytes.starts_with(b"\x1b[1;") {
        return None;
    }
    let mut pos = 4;
    let mod_value = scan_digits(bytes, &mut pos)?;
    let event = if pos < bytes.len() && bytes[pos] == b':' {
        pos += 1;
        parse_event_type(Some(scan_digits(bytes, &mut pos)?))
    } else {
        KeyEventType::Press
    };
    let codepoint = match *bytes.get(pos)? {
        b'H' => FC_HOME,
        b'F' => FC_END,
        _ => return None,
    };
    if pos + 1 != bytes.len() {
        return None;
    }
    Some((codepoint, mod_value - 1, event))
}

fn parse_kitty_sequence(data: &str) -> Option<ParsedKittySequence> {
    if let Some(parsed) = parse_csi_u(data) {
        return Some(parsed);
    }
    if let Some((codepoint, modifier, event_type)) = parse_kitty_arrow(data) {
        return Some(ParsedKittySequence {
            codepoint,
            shifted_key: None,
            base_layout_key: None,
            modifier,
            event_type,
        });
    }
    if let Some((codepoint, modifier, event_type)) = parse_kitty_functional(data) {
        return Some(ParsedKittySequence {
            codepoint,
            shifted_key: None,
            base_layout_key: None,
            modifier,
            event_type,
        });
    }
    if let Some((codepoint, modifier, event_type)) = parse_kitty_home_end(data) {
        return Some(ParsedKittySequence {
            codepoint,
            shifted_key: None,
            base_layout_key: None,
            modifier,
            event_type,
        });
    }
    None
}

fn matches_kitty_sequence(data: &str, expected_codepoint: i64, expected_modifier: i64) -> bool {
    let Some(parsed) = parse_kitty_sequence(data) else {
        return false;
    };
    let actual_mod = parsed.modifier & !LOCK_MASK;
    let expected_mod = expected_modifier & !LOCK_MASK;

    // Check if modifiers match
    if actual_mod != expected_mod {
        return false;
    }

    let normalized_codepoint = normalize_shifted_letter_identity_codepoint(
        normalize_kitty_functional_codepoint(parsed.codepoint),
        parsed.modifier,
    );
    let normalized_expected_codepoint = normalize_shifted_letter_identity_codepoint(
        normalize_kitty_functional_codepoint(expected_codepoint),
        expected_modifier,
    );

    // Primary match: codepoint matches directly after normalizing functional keys
    if normalized_codepoint == normalized_expected_codepoint {
        return true;
    }

    // Alternate match: use base layout key for non-Latin keyboard layouts.
    // This allows Ctrl+С (Cyrillic) to match Ctrl+c (Latin) when terminal reports
    // the base layout key (the key in standard PC-101 layout).
    //
    // Only fall back to base layout key when the codepoint is NOT already a
    // recognized Latin letter (a-z) or symbol (e.g., /, -, [, ;, etc.).
    // When the codepoint is a recognized key, it is authoritative regardless
    // of physical key position. This prevents remapped layouts (Dvorak, Colemak,
    // xremap, etc.) from causing false matches: both letters and symbols move
    // to different physical positions, so Ctrl+K could falsely match Ctrl+V
    // (letter remapping) and Ctrl+/ could falsely match Ctrl+[ (symbol remapping)
    // if the base layout key were always considered.
    if parsed
        .base_layout_key
        .is_some_and(|base| base == expected_codepoint)
    {
        let cp = normalized_codepoint;
        let is_latin_letter = (97..=122).contains(&cp); // a-z
        let is_known_symbol = codepoint_char(cp).is_some_and(|c| SYMBOL_KEYS.contains(&c));
        if !is_latin_letter && !is_known_symbol {
            return true;
        }
    }

    false
}

/// Restates upstream's modifyOtherKeys regex `^\x1b\[27;(\d+);(\d+)~$`.
fn parse_modify_other_keys_sequence(data: &str) -> Option<(i64, i64)> {
    let bytes = data.as_bytes();
    if !bytes.starts_with(b"\x1b[27;") || bytes.last() != Some(&b'~') {
        return None;
    }
    let mut pos = 5;
    let mod_value = scan_digits(bytes, &mut pos)?;
    if bytes.get(pos) != Some(&b';') {
        return None;
    }
    pos += 1;
    let codepoint = scan_digits(bytes, &mut pos)?;
    if pos + 1 != bytes.len() {
        return None;
    }
    Some((codepoint, mod_value - 1))
}

/// Match xterm modifyOtherKeys format: CSI 27 ; modifiers ; keycode ~
/// This is used by terminals when Kitty protocol is not enabled.
/// Modifier values are 1-indexed: 2=shift, 3=alt, 5=ctrl, etc.
fn matches_modify_other_keys(data: &str, expected_keycode: i64, expected_modifier: i64) -> bool {
    let Some((codepoint, modifier)) = parse_modify_other_keys_sequence(data) else {
        return false;
    };
    codepoint == expected_keycode && modifier == expected_modifier
}

/// Raw `0x08` (BS) is ambiguous in legacy terminals.
///
/// - Windows Terminal uses it for Ctrl+Backspace.
/// - Some legacy terminals and tmux setups send it for plain Backspace.
///
/// Prefer explicit Kitty / CSI-u / modifyOtherKeys sequences whenever they are
/// available. Fall back to a Windows Terminal heuristic only for raw BS bytes.
/// The `windows_terminal` flag restates upstream's environment probe.
fn matches_raw_backspace(data: &str, expected_modifier: i64, windows_terminal: bool) -> bool {
    if data == "\x7f" {
        return expected_modifier == 0;
    }
    if data != "\x08" {
        return false;
    }
    if windows_terminal {
        expected_modifier == MOD_CTRL
    } else {
        expected_modifier == 0
    }
}

/// Get the control character for a key.
/// Uses the universal formula: code & 0x1f (mask to lower 5 bits)
///
/// Works for:
/// - Letters a-z → 1-26
/// - Symbols [\]_ → 27, 28, 29, 31
/// - Also maps - to same as _ (same physical key on US keyboards)
fn raw_ctrl_char(key_char: char) -> Option<char> {
    let lowered = key_char.to_ascii_lowercase();
    let code = u32::from(lowered);
    if (97..=122).contains(&code) || matches!(lowered, '[' | '\\' | ']' | '_') {
        return char::from_u32(code & 0x1f);
    }
    // Handle - as _ (same physical key on US keyboards)
    if lowered == '-' {
        return Some('\x1f'); // Same as Ctrl+_
    }
    None
}

/// The single-char key a printable fallthrough handles: one lowercase letter,
/// digit, or symbol from [`SYMBOL_KEYS`].
fn printable_key_char(key: &str) -> Option<char> {
    let mut chars = key.chars();
    let key_char = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    let is_letter = key_char.is_ascii_lowercase();
    let is_digit = key_char.is_ascii_digit();
    let is_symbol = SYMBOL_KEYS.contains(&key_char);
    if is_letter || is_digit || is_symbol {
        Some(key_char)
    } else {
        None
    }
}

fn matches_printable_modify_other_keys(
    data: &str,
    expected_keycode: i64,
    expected_modifier: i64,
) -> bool {
    if expected_modifier == 0 {
        return false;
    }
    let Some((codepoint, modifier)) = parse_modify_other_keys_sequence(data) else {
        return false;
    };
    if modifier != expected_modifier {
        return false;
    }
    normalize_shifted_letter_identity_codepoint(codepoint, modifier)
        == normalize_shifted_letter_identity_codepoint(expected_keycode, expected_modifier)
}

fn format_key_name_with_modifiers(key_name: &str, modifier: i64) -> Option<KeyId> {
    let mut mods: Vec<&str> = Vec::new();
    let effective_mod = modifier & !LOCK_MASK;
    let supported_modifier_mask = MOD_SHIFT | MOD_CTRL | MOD_ALT | MOD_SUPER;
    if (effective_mod & !supported_modifier_mask) != 0 {
        return None;
    }
    if (effective_mod & MOD_SHIFT) != 0 {
        mods.push("shift");
    }
    if (effective_mod & MOD_CTRL) != 0 {
        mods.push("ctrl");
    }
    if (effective_mod & MOD_ALT) != 0 {
        mods.push("alt");
    }
    if (effective_mod & MOD_SUPER) != 0 {
        mods.push("super");
    }
    if mods.is_empty() {
        return Some(key_name.to_string());
    }
    Some(format!("{}+{key_name}", mods.join("+")))
}

/// Upstream's `parseKeyId` result: the key segment and one bool per modifier;
/// the port folds the bools into the wire modifier bitmask.
struct ParsedKeyId {
    key: String,
    modifier: i64,
}

/// Splits a key identifier into its key part and modifier bitmask; upstream's
/// `parseKeyId`, which lowercases and splits on `+` with the key last.
fn parse_key_id(key_id: &str) -> Option<ParsedKeyId> {
    let lowered = key_id.to_lowercase();
    let parts: Vec<&str> = lowered.split('+').collect();
    // split always yields at least one element, so the index is in bounds.
    let key = parts[parts.len() - 1];
    if key.is_empty() {
        return None;
    }
    let mut modifier = 0;
    if parts.contains(&"shift") {
        modifier |= MOD_SHIFT;
    }
    if parts.contains(&"alt") {
        modifier |= MOD_ALT;
    }
    if parts.contains(&"ctrl") {
        modifier |= MOD_CTRL;
    }
    if parts.contains(&"super") {
        modifier |= MOD_SUPER;
    }
    Some(ParsedKeyId {
        key: key.to_string(),
        modifier,
    })
}

fn format_parsed_key(codepoint: i64, modifier: i64, base_layout_key: Option<i64>) -> Option<KeyId> {
    let normalized_codepoint = normalize_kitty_functional_codepoint(codepoint);
    let identity_codepoint =
        normalize_shifted_letter_identity_codepoint(normalized_codepoint, modifier);

    // Use base layout key only when codepoint is not a recognized Latin
    // letter (a-z), digit (0-9), or symbol (/, -, [, ;, etc.). For those,
    // the codepoint is authoritative regardless of physical key position.
    // This prevents remapped layouts (Dvorak, Colemak, xremap, etc.) from
    // reporting the wrong key name based on the QWERTY physical position.
    let is_latin_letter = (97..=122).contains(&identity_codepoint); // a-z
    let is_digit = (48..=57).contains(&identity_codepoint); // 0-9
    let is_known_symbol =
        codepoint_char(identity_codepoint).is_some_and(|c| SYMBOL_KEYS.contains(&c));
    let effective_codepoint = if is_latin_letter || is_digit || is_known_symbol {
        identity_codepoint
    } else {
        base_layout_key.unwrap_or(identity_codepoint)
    };

    let key_name: Option<String> = if effective_codepoint == CP_ESCAPE {
        Some("escape".to_string())
    } else if effective_codepoint == CP_TAB {
        Some("tab".to_string())
    } else if effective_codepoint == CP_ENTER || effective_codepoint == CP_KP_ENTER {
        Some("enter".to_string())
    } else if effective_codepoint == CP_SPACE {
        Some("space".to_string())
    } else if effective_codepoint == CP_BACKSPACE {
        Some("backspace".to_string())
    } else if effective_codepoint == FC_DELETE {
        Some("delete".to_string())
    } else if effective_codepoint == FC_INSERT {
        Some("insert".to_string())
    } else if effective_codepoint == FC_HOME {
        Some("home".to_string())
    } else if effective_codepoint == FC_END {
        Some("end".to_string())
    } else if effective_codepoint == FC_PAGE_UP {
        Some("pageUp".to_string())
    } else if effective_codepoint == FC_PAGE_DOWN {
        Some("pageDown".to_string())
    } else if effective_codepoint == ARROW_UP {
        Some("up".to_string())
    } else if effective_codepoint == ARROW_DOWN {
        Some("down".to_string())
    } else if effective_codepoint == ARROW_LEFT {
        Some("left".to_string())
    } else if effective_codepoint == ARROW_RIGHT {
        Some("right".to_string())
    } else if (48..=57).contains(&effective_codepoint)
        || (97..=122).contains(&effective_codepoint)
        || codepoint_char(effective_codepoint).is_some_and(|c| SYMBOL_KEYS.contains(&c))
    {
        codepoint_char(effective_codepoint).map(|c| c.to_string())
    } else {
        None
    };

    key_name.and_then(|name| format_key_name_with_modifiers(&name, modifier))
}

/// The `char` a wire codepoint denotes, when it is a valid Unicode scalar.
fn codepoint_char(codepoint: i64) -> Option<char> {
    u32::try_from(codepoint).ok().and_then(char::from_u32)
}

/// The letter a legacy control character stands for: `1..=26` maps to
/// `a..=z`, restating upstream's `String.fromCharCode(code + 96)`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "call sites pass 1..=26, so the offset always fits u8"
)]
fn control_to_letter(code: u32) -> char {
    let letter = (code - 1) as u8 + b'a';
    char::from(letter)
}

// =============================================================================
// Kitty CSI-u Printable Decoding
// =============================================================================

const KITTY_PRINTABLE_ALLOWED_MODIFIERS: i64 = MOD_SHIFT | LOCK_MASK;

/// Decode a Kitty CSI-u sequence into a printable character, if applicable.
///
/// When Kitty keyboard protocol flag 1 (disambiguate) is active, terminals send
/// CSI-u sequences for all keys, including plain printable characters. This
/// function extracts the printable character from such sequences.
///
/// Only accepts plain or Shift-modified keys. Rejects Ctrl, Alt, and unsupported
/// modifier combinations (those are handled by keybinding matching instead).
/// Prefers the shifted keycode when Shift is held and a shifted key is reported.
///
/// Returns the printable character, or `None` when the data is not a printable
/// CSI-u sequence.
#[must_use]
pub fn decode_kitty_printable(data: &str) -> Option<String> {
    let parsed = parse_csi_u(data)?;
    let modifier = parsed.modifier;

    // Only accept printable CSI-u input for plain or Shift-modified text keys.
    // Reject unsupported modifier bits (e.g. Super/Meta) to avoid inserting
    // characters from modifier-only terminal events.
    if (modifier & !KITTY_PRINTABLE_ALLOWED_MODIFIERS) != 0 {
        return None;
    }
    if (modifier & (MOD_ALT | MOD_CTRL)) != 0 {
        return None;
    }

    // Prefer the shifted keycode when Shift is held.
    let effective_codepoint = normalize_kitty_functional_codepoint(
        if (modifier & MOD_SHIFT) != 0
            && let Some(shifted) = parsed.shifted_key
        {
            shifted
        } else {
            parsed.codepoint
        },
    );
    // Drop control characters or invalid codepoints.
    if effective_codepoint < 32 {
        return None;
    }

    codepoint_char(effective_codepoint).map(|c| c.to_string())
}

fn decode_modify_other_keys_printable(data: &str) -> Option<String> {
    let (codepoint, modifier) = parse_modify_other_keys_sequence(data)?;
    let modifier = modifier & !LOCK_MASK;
    if (modifier & !MOD_SHIFT) != 0 {
        return None;
    }
    if codepoint < 32 {
        return None;
    }
    codepoint_char(codepoint).map(|c| c.to_string())
}

/// Decode printable text out of either protocol's modified-key report:
/// Kitty CSI-u first, then xterm `modifyOtherKeys`. See
/// [`decode_kitty_printable`] for the accepted CSI-u shape.
#[must_use]
pub fn decode_printable_key(data: &str) -> Option<String> {
    decode_kitty_printable(data).or_else(|| decode_modify_other_keys_printable(data))
}

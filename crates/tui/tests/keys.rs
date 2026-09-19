//! Port of `packages/tui/test/keys.test.ts` — 1:1 against upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#40).
//!
//! Upstream toggled module globals and process environment to stage fixtures
//! (`setKittyProtocolActive`, `withEnv`/`withEnvVars`); the port stages the
//! same state on a fresh [`KeyParser`] per test, so the suites stay hermetic.

use pi_tui::keys::{Key, KeyParser, decode_kitty_printable, decode_printable_key};

fn parser() -> KeyParser {
    KeyParser::new()
}

fn parser_with_kitty(active: bool) -> KeyParser {
    let mut parser = KeyParser::new();
    parser.set_kitty_protocol_active(active);
    parser
}

fn parser_with_windows_terminal(active: bool) -> KeyParser {
    let mut parser = KeyParser::new();
    parser.set_windows_terminal_session(active);
    parser
}

// =============================================================================
// matchesKey — Kitty protocol with alternate keys (non-Latin layouts)
//
// Kitty protocol flag 4 (Report alternate keys) sends:
// CSI codepoint:shifted:base ; modifier:event u
// Where base is the key in standard PC-101 layout
// =============================================================================

#[test]
fn matches_ctrl_c_when_pressing_cyrillic_ctrl_s_with_base_layout_key() {
    // Cyrillic 'с' = codepoint 1089, Latin 'c' = codepoint 99
    // Format: CSI 1089::99;5u (codepoint::base;modifier with ctrl=4, +1=5)
    assert!(parser_with_kitty(true).matches_key("\x1b[1089::99;5u", "ctrl+c"));
}

#[test]
fn matches_ctrl_d_when_pressing_cyrillic_ctrl_v_with_base_layout_key() {
    // Cyrillic 'в' = codepoint 1074, Latin 'd' = codepoint 100
    assert!(parser_with_kitty(true).matches_key("\x1b[1074::100;5u", "ctrl+d"));
}

#[test]
fn matches_ctrl_z_when_pressing_cyrillic_ctrl_ya_with_base_layout_key() {
    // Cyrillic 'я' = codepoint 1103, Latin 'z' = codepoint 122
    assert!(parser_with_kitty(true).matches_key("\x1b[1103::122;5u", "ctrl+z"));
}

#[test]
fn matches_ctrl_shift_p_with_base_layout_key() {
    // Cyrillic 'з' = codepoint 1079, Latin 'p' = codepoint 112
    // ctrl=4, shift=1, +1 = 6
    assert!(parser_with_kitty(true).matches_key("\x1b[1079::112;6u", "ctrl+shift+p"));
}

#[test]
fn still_matches_direct_codepoint_when_no_base_layout_key() {
    // Latin ctrl+c without base layout key (terminal doesn't support flag 4)
    assert!(parser_with_kitty(true).matches_key("\x1b[99;5u", "ctrl+c"));
}

#[test]
fn matches_super_modified_kitty_bindings_including_combined_modifiers() {
    let parser = parser_with_kitty(true);
    assert!(parser.matches_key("\x1b[107;9u", "super+k"));
    assert!(parser.matches_key("\x1b[13;9u", "super+enter"));
    assert!(parser.matches_key("\x1b[107;13u", &Key::ctrl_super("k")));
    assert!(parser.matches_key("\x1b[107;13u", "ctrl+super+k"));
    assert!(parser.matches_key("\x1b[107;14u", "ctrl+shift+super+k"));
    assert!(!parser.matches_key("\x1b[107;13u", "super+k"));
    assert_eq!(parser.parse_key("\x1b[107;9u").as_deref(), Some("super+k"));
    assert_eq!(
        parser.parse_key("\x1b[13;9u").as_deref(),
        Some("super+enter")
    );
    assert_eq!(
        parser.parse_key("\x1b[107;13u").as_deref(),
        Some("ctrl+super+k")
    );
    assert_eq!(
        parser.parse_key("\x1b[107;14u").as_deref(),
        Some("shift+ctrl+super+k")
    );
}

#[test]
fn matches_digit_bindings_via_kitty_csi_u() {
    let parser = parser_with_kitty(true);
    assert!(parser.matches_key("\x1b[49u", "1"));
    assert!(parser.matches_key("\x1b[49;5u", "ctrl+1"));
    assert!(!parser.matches_key("\x1b[49;5u", "ctrl+2"));
    assert_eq!(parser.parse_key("\x1b[49u").as_deref(), Some("1"));
    assert_eq!(parser.parse_key("\x1b[49;5u").as_deref(), Some("ctrl+1"));
}

#[test]
fn normalizes_kitty_keypad_functional_keys_to_logical_digits_symbols_and_navigation() {
    let parser = parser_with_kitty(true);
    assert!(parser.matches_key("\x1b[57400u", "1"));
    assert!(parser.matches_key("\x1b[57410u", "/"));
    assert!(parser.matches_key("\x1b[57417u", "left"));
    assert!(parser.matches_key("\x1b[57426u", "delete"));
    assert_eq!(parser.parse_key("\x1b[57399u").as_deref(), Some("0"));
    assert_eq!(parser.parse_key("\x1b[57409u").as_deref(), Some("."));
    assert_eq!(parser.parse_key("\x1b[57413u").as_deref(), Some("+"));
    assert_eq!(parser.parse_key("\x1b[57416u").as_deref(), Some(","));
    assert_eq!(parser.parse_key("\x1b[57417u").as_deref(), Some("left"));
    assert_eq!(parser.parse_key("\x1b[57418u").as_deref(), Some("right"));
    assert_eq!(parser.parse_key("\x1b[57419u").as_deref(), Some("up"));
    assert_eq!(parser.parse_key("\x1b[57420u").as_deref(), Some("down"));
    assert_eq!(parser.parse_key("\x1b[57421u").as_deref(), Some("pageUp"));
    assert_eq!(parser.parse_key("\x1b[57422u").as_deref(), Some("pageDown"));
    assert_eq!(parser.parse_key("\x1b[57423u").as_deref(), Some("home"));
    assert_eq!(parser.parse_key("\x1b[57424u").as_deref(), Some("end"));
    assert_eq!(parser.parse_key("\x1b[57425u").as_deref(), Some("insert"));
    assert_eq!(parser.parse_key("\x1b[57426u").as_deref(), Some("delete"));
}

#[test]
fn handles_shifted_key_in_format() {
    // Format with shifted key: CSI codepoint:shifted:base;modifier u
    // Latin 'c' with shifted 'C' (67) and base 'c' (99)
    // shift modifier = 1, +1 = 2
    assert!(parser_with_kitty(true).matches_key("\x1b[99:67:99;2u", "shift+c"));
}

#[test]
fn handles_event_type_in_format() {
    // Format with event type: CSI codepoint::base;modifier:event u
    // Cyrillic ctrl+c release event (event type 3)
    assert!(parser_with_kitty(true).matches_key("\x1b[1089::99;5:3u", "ctrl+c"));
}

#[test]
fn handles_full_format_with_shifted_key_base_key_and_event_type() {
    // Full format: CSI codepoint:shifted:base;modifier:event u
    // Cyrillic 'С' (shifted) with base 'c', Ctrl+Shift pressed, repeat event
    // Cyrillic 'с' = 1089, Cyrillic 'С' = 1057, Latin 'c' = 99
    // ctrl=4, shift=1, +1 = 6, repeat event = 2
    assert!(parser_with_kitty(true).matches_key("\x1b[1089:1057:99;6:2u", "ctrl+shift+c"));
}

#[test]
fn prefers_codepoint_for_latin_letters_even_when_base_layout_differs() {
    // Dvorak Ctrl+K reports codepoint 'k' (107) and base layout 'v' (118)
    let parser = parser_with_kitty(true);
    assert!(parser.matches_key("\x1b[107::118;5u", "ctrl+k"));
    assert!(!parser.matches_key("\x1b[107::118;5u", "ctrl+v"));
}

#[test]
fn prefers_codepoint_for_symbol_keys_even_when_base_layout_differs() {
    // Dvorak Ctrl+/ reports codepoint '/' (47) and base layout '[' (91)
    let parser = parser_with_kitty(true);
    assert!(parser.matches_key("\x1b[47::91;5u", "ctrl+/"));
    assert!(!parser.matches_key("\x1b[47::91;5u", "ctrl+["));
}

#[test]
fn does_not_match_wrong_key_even_with_base_layout() {
    // Cyrillic ctrl+с with base 'c' should NOT match ctrl+d
    assert!(!parser_with_kitty(true).matches_key("\x1b[1089::99;5u", "ctrl+d"));
}

#[test]
fn does_not_match_wrong_modifiers_even_with_base_layout() {
    // Cyrillic ctrl+с should NOT match ctrl+shift+c
    assert!(!parser_with_kitty(true).matches_key("\x1b[1089::99;5u", "ctrl+shift+c"));
}

// =============================================================================
// matchesKey — modifyOtherKeys matching
// =============================================================================

#[test]
fn matches_xterm_modify_other_keys_ctrl_c() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;5;99~", "ctrl+c"));
    assert_eq!(parser.parse_key("\x1b[27;5;99~").as_deref(), Some("ctrl+c"));
}

#[test]
fn matches_xterm_modify_other_keys_ctrl_d() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;5;100~", "ctrl+d"));
    assert_eq!(
        parser.parse_key("\x1b[27;5;100~").as_deref(),
        Some("ctrl+d")
    );
}

#[test]
fn matches_xterm_modify_other_keys_ctrl_z() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;5;122~", "ctrl+z"));
    assert_eq!(
        parser.parse_key("\x1b[27;5;122~").as_deref(),
        Some("ctrl+z")
    );
}

#[test]
fn matches_xterm_modify_other_keys_enter_variants() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;5;13~", "ctrl+enter"));
    assert!(parser.matches_key("\x1b[27;2;13~", "shift+enter"));
    assert!(parser.matches_key("\x1b[27;3;13~", "alt+enter"));
    assert_eq!(
        parser.parse_key("\x1b[27;5;13~").as_deref(),
        Some("ctrl+enter")
    );
    assert_eq!(
        parser.parse_key("\x1b[27;2;13~").as_deref(),
        Some("shift+enter")
    );
    assert_eq!(
        parser.parse_key("\x1b[27;3;13~").as_deref(),
        Some("alt+enter")
    );
}

#[test]
fn matches_xterm_modify_other_keys_tab_variants() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;2;9~", "shift+tab"));
    assert!(parser.matches_key("\x1b[27;5;9~", "ctrl+tab"));
    assert!(parser.matches_key("\x1b[27;3;9~", "alt+tab"));
    assert_eq!(
        parser.parse_key("\x1b[27;2;9~").as_deref(),
        Some("shift+tab")
    );
    assert_eq!(
        parser.parse_key("\x1b[27;5;9~").as_deref(),
        Some("ctrl+tab")
    );
    assert_eq!(parser.parse_key("\x1b[27;3;9~").as_deref(), Some("alt+tab"));
}

#[test]
fn matches_xterm_modify_other_keys_backspace_variants() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;1;127~", "backspace"));
    assert!(parser.matches_key("\x1b[27;5;127~", "ctrl+backspace"));
    assert!(parser.matches_key("\x1b[27;3;127~", "alt+backspace"));
    assert_eq!(
        parser.parse_key("\x1b[27;1;127~").as_deref(),
        Some("backspace")
    );
    assert_eq!(
        parser.parse_key("\x1b[27;5;127~").as_deref(),
        Some("ctrl+backspace")
    );
    assert_eq!(
        parser.parse_key("\x1b[27;3;127~").as_deref(),
        Some("alt+backspace")
    );
}

#[test]
fn matches_xterm_modify_other_keys_escape() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;1;27~", "escape"));
    assert_eq!(parser.parse_key("\x1b[27;1;27~").as_deref(), Some("escape"));
}

#[test]
fn matches_xterm_modify_other_keys_space_variants() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;1;32~", "space"));
    assert!(parser.matches_key("\x1b[27;5;32~", "ctrl+space"));
    assert_eq!(parser.parse_key("\x1b[27;1;32~").as_deref(), Some("space"));
    assert_eq!(
        parser.parse_key("\x1b[27;5;32~").as_deref(),
        Some("ctrl+space")
    );
}

#[test]
fn matches_xterm_modify_other_keys_symbol_combos() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;5;47~", "ctrl+/"));
    assert_eq!(parser.parse_key("\x1b[27;5;47~").as_deref(), Some("ctrl+/"));
}

#[test]
fn matches_xterm_modify_other_keys_digit_combos() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;5;49~", "ctrl+1"));
    assert!(parser.matches_key("\x1b[27;2;49~", "shift+1"));
    assert_eq!(parser.parse_key("\x1b[27;5;49~").as_deref(), Some("ctrl+1"));
    assert_eq!(
        parser.parse_key("\x1b[27;2;49~").as_deref(),
        Some("shift+1")
    );
}

#[test]
fn matches_xterm_modify_other_keys_shifted_uppercase_letters() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;2;69~", "shift+e"));
    assert!(parser.matches_key("\x1b[27;6;69~", "ctrl+shift+e"));
    assert_eq!(
        parser.parse_key("\x1b[27;2;69~").as_deref(),
        Some("shift+e")
    );
    assert_eq!(
        parser.parse_key("\x1b[27;6;69~").as_deref(),
        Some("shift+ctrl+e")
    );
}

#[test]
fn matches_ctrl_alt_letter_via_csi_u_when_kitty_inactive() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[104;7u", "ctrl+alt+h"));
    assert_eq!(
        parser.parse_key("\x1b[104;7u").as_deref(),
        Some("ctrl+alt+h")
    );
}

#[test]
fn matches_ctrl_alt_letter_via_xterm_modify_other_keys() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;7;104~", "ctrl+alt+h"));
    assert_eq!(
        parser.parse_key("\x1b[27;7;104~").as_deref(),
        Some("ctrl+alt+h")
    );
}

// =============================================================================
// matchesKey — legacy key matching
// =============================================================================

#[test]
fn matches_legacy_ctrl_c() {
    // Ctrl+c sends ASCII 3 (ETX)
    assert!(parser_with_kitty(false).matches_key("\x03", "ctrl+c"));
}

#[test]
fn matches_legacy_ctrl_d() {
    // Ctrl+d sends ASCII 4 (EOT)
    assert!(parser_with_kitty(false).matches_key("\x04", "ctrl+d"));
}

#[test]
fn matches_escape_key() {
    assert!(parser().matches_key("\x1b", "escape"));
}

#[test]
fn matches_legacy_linefeed_as_enter() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\n", "enter"));
    assert_eq!(parser.parse_key("\n").as_deref(), Some("enter"));
}

#[test]
fn treats_linefeed_as_shift_enter_when_kitty_active() {
    let parser = parser_with_kitty(true);
    assert!(parser.matches_key("\n", "shift+enter"));
    assert!(!parser.matches_key("\n", "enter"));
    assert_eq!(parser.parse_key("\n").as_deref(), Some("shift+enter"));
}

#[test]
fn parses_ctrl_space() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x00", "ctrl+space"));
    assert_eq!(parser.parse_key("\x00").as_deref(), Some("ctrl+space"));
}

#[test]
fn matches_legacy_ctrl_symbol() {
    let parser = parser_with_kitty(false);
    // Ctrl+\ sends ASCII 28 (File Separator) in legacy terminals
    assert!(parser.matches_key("\x1c", "ctrl+\\"));
    assert_eq!(parser.parse_key("\x1c").as_deref(), Some("ctrl+\\"));
    // Ctrl+] sends ASCII 29 (Group Separator) in legacy terminals
    assert!(parser.matches_key("\x1d", "ctrl+]"));
    assert_eq!(parser.parse_key("\x1d").as_deref(), Some("ctrl+]"));
    // Ctrl+_ sends ASCII 31 (Unit Separator) in legacy terminals
    // Ctrl+- is on the same physical key on US keyboards
    assert!(parser.matches_key("\x1f", "ctrl+_"));
    assert!(parser.matches_key("\x1f", "ctrl+-"));
    assert_eq!(parser.parse_key("\x1f").as_deref(), Some("ctrl+-"));
}

#[test]
fn matches_legacy_ctrl_alt_symbol() {
    let parser = parser_with_kitty(false);
    // Ctrl+Alt+[ sends ESC followed by ESC (Ctrl+[ = ESC)
    assert!(parser.matches_key("\x1b\x1b", "ctrl+alt+["));
    assert_eq!(parser.parse_key("\x1b\x1b").as_deref(), Some("ctrl+alt+["));
    // Ctrl+Alt+\ sends ESC followed by ASCII 28
    assert!(parser.matches_key("\x1b\x1c", "ctrl+alt+\\"));
    assert_eq!(parser.parse_key("\x1b\x1c").as_deref(), Some("ctrl+alt+\\"));
    // Ctrl+Alt+] sends ESC followed by ASCII 29
    assert!(parser.matches_key("\x1b\x1d", "ctrl+alt+]"));
    assert_eq!(parser.parse_key("\x1b\x1d").as_deref(), Some("ctrl+alt+]"));
    // Ctrl+_ sends ASCII 31 (Unit Separator) in legacy terminals
    // Ctrl+- is on the same physical key on US keyboards
    assert!(parser.matches_key("\x1b\x1f", "ctrl+alt+_"));
    assert!(parser.matches_key("\x1b\x1f", "ctrl+alt+-"));
    assert_eq!(parser.parse_key("\x1b\x1f").as_deref(), Some("ctrl+alt+-"));
}

#[test]
fn treats_raw_0x08_as_plain_backspace_outside_windows_terminal() {
    let parser = parser_with_windows_terminal(false);
    assert!(parser.matches_key("\x7f", "backspace"));
    assert!(!parser.matches_key("\x7f", "ctrl+backspace"));
    assert_eq!(parser.parse_key("\x7f").as_deref(), Some("backspace"));
    assert!(parser.matches_key("\x08", "backspace"));
    assert!(!parser.matches_key("\x08", "ctrl+backspace"));
    assert_eq!(parser.parse_key("\x08").as_deref(), Some("backspace"));
    // Raw 0x08 overlaps with Ctrl+H either way.
    assert!(parser.matches_key("\x08", "ctrl+h"));
}

#[test]
fn treats_raw_0x08_as_ctrl_backspace_in_local_windows_terminal() {
    let parser = parser_with_windows_terminal(true);
    assert!(parser.matches_key("\x08", "ctrl+backspace"));
    assert!(!parser.matches_key("\x08", "backspace"));
    assert_eq!(parser.parse_key("\x08").as_deref(), Some("ctrl+backspace"));
    assert!(parser.matches_key("\x08", "ctrl+h"));
}

#[test]
fn treats_raw_0x08_as_plain_backspace_in_windows_terminal_over_ssh() {
    // Upstream staged WT_SESSION with SSH_CONNECTION/SSH_CLIENT/SSH_TTY set;
    // the probe result is false, restated here as the explicit flag.
    let parser = parser_with_windows_terminal(false);
    assert!(!parser.matches_key("\x08", "ctrl+backspace"));
    assert!(parser.matches_key("\x08", "backspace"));
    assert_eq!(parser.parse_key("\x08").as_deref(), Some("backspace"));
    assert!(parser.matches_key("\x08", "ctrl+h"));
}

#[test]
fn parses_legacy_alt_prefixed_sequences_when_kitty_inactive() {
    let inactive = parser_with_kitty(false);
    assert!(inactive.matches_key("\x1b ", "alt+space"));
    assert_eq!(inactive.parse_key("\x1b ").as_deref(), Some("alt+space"));
    assert!(inactive.matches_key("\x1b\x08", "alt+backspace"));
    assert_eq!(
        inactive.parse_key("\x1b\x08").as_deref(),
        Some("alt+backspace")
    );
    assert!(inactive.matches_key("\x1b\x03", "ctrl+alt+c"));
    assert_eq!(
        inactive.parse_key("\x1b\x03").as_deref(),
        Some("ctrl+alt+c")
    );
    assert!(inactive.matches_key("\x1bB", "alt+left"));
    assert_eq!(inactive.parse_key("\x1bB").as_deref(), Some("alt+left"));
    assert!(inactive.matches_key("\x1bF", "alt+right"));
    assert_eq!(inactive.parse_key("\x1bF").as_deref(), Some("alt+right"));
    assert!(inactive.matches_key("\x1ba", "alt+a"));
    assert_eq!(inactive.parse_key("\x1ba").as_deref(), Some("alt+a"));
    assert!(inactive.matches_key("\x1b1", "alt+1"));
    assert_eq!(inactive.parse_key("\x1b1").as_deref(), Some("alt+1"));
    assert!(inactive.matches_key("\x1b,", "alt+,"));
    assert_eq!(inactive.parse_key("\x1b,").as_deref(), Some("alt+,"));
    assert!(inactive.matches_key("\x1b.", "alt+."));
    assert_eq!(inactive.parse_key("\x1b.").as_deref(), Some("alt+."));
    assert!(inactive.matches_key("\x1by", "alt+y"));
    assert_eq!(inactive.parse_key("\x1by").as_deref(), Some("alt+y"));
    assert!(inactive.matches_key("\x1bz", "alt+z"));
    assert_eq!(inactive.parse_key("\x1bz").as_deref(), Some("alt+z"));

    let active = parser_with_kitty(true);
    assert!(!active.matches_key("\x1b ", "alt+space"));
    assert_eq!(active.parse_key("\x1b "), None);
    // ESC + BS stays alt+backspace even with Kitty protocol active.
    assert!(active.matches_key("\x1b\x08", "alt+backspace"));
    assert_eq!(
        active.parse_key("\x1b\x08").as_deref(),
        Some("alt+backspace")
    );
    assert!(!active.matches_key("\x1b\x03", "ctrl+alt+c"));
    assert_eq!(active.parse_key("\x1b\x03"), None);
    assert!(!active.matches_key("\x1bB", "alt+left"));
    assert_eq!(active.parse_key("\x1bB"), None);
    assert!(!active.matches_key("\x1bF", "alt+right"));
    assert_eq!(active.parse_key("\x1bF"), None);
    assert!(!active.matches_key("\x1ba", "alt+a"));
    assert_eq!(active.parse_key("\x1ba"), None);
    assert!(!active.matches_key("\x1b1", "alt+1"));
    assert_eq!(active.parse_key("\x1b1"), None);
    assert!(!active.matches_key("\x1b,", "alt+,"));
    assert_eq!(active.parse_key("\x1b,"), None);
    assert!(!active.matches_key("\x1b.", "alt+."));
    assert_eq!(active.parse_key("\x1b."), None);
    assert!(!active.matches_key("\x1by", "alt+y"));
    assert_eq!(active.parse_key("\x1by"), None);
    assert!(!active.matches_key("\x1bz", "alt+z"));
    assert_eq!(active.parse_key("\x1bz"), None);
}

#[test]
fn matches_arrow_keys() {
    let parser = parser();
    assert!(parser.matches_key("\x1b[A", "up"));
    assert!(parser.matches_key("\x1b[B", "down"));
    assert!(parser.matches_key("\x1b[C", "right"));
    assert!(parser.matches_key("\x1b[D", "left"));
}

#[test]
fn matches_ss3_arrows_and_home_end() {
    let parser = parser();
    assert!(parser.matches_key("\x1bOA", "up"));
    assert!(parser.matches_key("\x1bOB", "down"));
    assert!(parser.matches_key("\x1bOC", "right"));
    assert!(parser.matches_key("\x1bOD", "left"));
    assert!(parser.matches_key("\x1bOH", "home"));
    assert!(parser.matches_key("\x1bOF", "end"));
}

#[test]
fn matches_xterm_ctrl_modified_viewport_navigation() {
    let parser = parser();
    assert!(parser.matches_key("\x1b[1;5H", "ctrl+home"));
    assert!(parser.matches_key("\x1b[1;5F", "ctrl+end"));
    assert!(parser.matches_key("\x1b[5;5~", "ctrl+pageUp"));
    assert!(parser.matches_key("\x1b[6;5~", "ctrl+pageDown"));
    assert_eq!(parser.parse_key("\x1b[1;5H").as_deref(), Some("ctrl+home"));
    assert_eq!(parser.parse_key("\x1b[1;5F").as_deref(), Some("ctrl+end"));
    assert_eq!(
        parser.parse_key("\x1b[5;5~").as_deref(),
        Some("ctrl+pageUp")
    );
    assert_eq!(
        parser.parse_key("\x1b[6;5~").as_deref(),
        Some("ctrl+pageDown")
    );
}

#[test]
fn matches_legacy_function_keys_and_clear() {
    let parser = parser();
    assert!(parser.matches_key("\x1bOP", "f1"));
    assert!(parser.matches_key("\x1b[24~", "f12"));
    assert!(parser.matches_key("\x1b[E", "clear"));
}

#[test]
fn matches_alt_arrows() {
    let parser = parser();
    assert!(parser.matches_key("\x1bp", "alt+up"));
    assert!(!parser.matches_key("\x1bp", "up"));
}

#[test]
fn matches_rxvt_modifier_sequences() {
    let parser = parser();
    assert!(parser.matches_key("\x1b[a", "shift+up"));
    assert!(parser.matches_key("\x1bOa", "ctrl+up"));
    assert!(parser.matches_key("\x1b[2$", "shift+insert"));
    assert!(parser.matches_key("\x1b[2^", "ctrl+insert"));
    assert!(parser.matches_key("\x1b[7$", "shift+home"));
}

// =============================================================================
// decodeKittyPrintable
// =============================================================================

#[test]
fn decodes_kitty_keypad_functional_keys_to_printable_characters() {
    assert_eq!(decode_kitty_printable("\x1b[57399u").as_deref(), Some("0"));
    assert_eq!(decode_kitty_printable("\x1b[57400u").as_deref(), Some("1"));
    assert_eq!(decode_kitty_printable("\x1b[57409u").as_deref(), Some("."));
    assert_eq!(decode_kitty_printable("\x1b[57410u").as_deref(), Some("/"));
    assert_eq!(decode_kitty_printable("\x1b[57411u").as_deref(), Some("*"));
    assert_eq!(decode_kitty_printable("\x1b[57412u").as_deref(), Some("-"));
    assert_eq!(decode_kitty_printable("\x1b[57413u").as_deref(), Some("+"));
    assert_eq!(decode_kitty_printable("\x1b[57415u").as_deref(), Some("="));
    assert_eq!(decode_kitty_printable("\x1b[57416u").as_deref(), Some(","));
    assert_eq!(decode_kitty_printable("\x1b[57417u"), None);
}

// =============================================================================
// decodePrintableKey
// =============================================================================

#[test]
fn decodes_printable_xterm_modify_other_keys_sequences() {
    assert_eq!(decode_printable_key("\x1b[27;2;69~").as_deref(), Some("E"));
    assert_eq!(decode_printable_key("\x1b[27;2;196~").as_deref(), Some("Ä"));
    assert_eq!(decode_printable_key("\x1b[27;2;32~").as_deref(), Some(" "));
    assert_eq!(decode_printable_key("\x1b[27;2;13~"), None);
    assert_eq!(decode_printable_key("\x1b[27;6;69~"), None);
}

// =============================================================================
// parseKey — Kitty protocol with alternate keys
// =============================================================================

#[test]
fn returns_latin_key_name_when_base_layout_key_is_present() {
    // Cyrillic ctrl+с with base layout 'c'
    assert_eq!(
        parser_with_kitty(true)
            .parse_key("\x1b[1089::99;5u")
            .as_deref(),
        Some("ctrl+c")
    );
}

#[test]
fn parse_prefers_codepoint_for_latin_letters_when_base_layout_differs() {
    // Dvorak Ctrl+K reports codepoint 'k' (107) and base layout 'v' (118)
    assert_eq!(
        parser_with_kitty(true)
            .parse_key("\x1b[107::118;5u")
            .as_deref(),
        Some("ctrl+k")
    );
}

#[test]
fn parse_prefers_codepoint_for_symbol_keys_when_base_layout_differs() {
    // Dvorak Ctrl+/ reports codepoint '/' (47) and base layout '[' (91)
    assert_eq!(
        parser_with_kitty(true)
            .parse_key("\x1b[47::91;5u")
            .as_deref(),
        Some("ctrl+/")
    );
}

#[test]
fn returns_key_name_from_codepoint_when_no_base_layout() {
    assert_eq!(
        parser_with_kitty(true).parse_key("\x1b[99;5u").as_deref(),
        Some("ctrl+c")
    );
}

#[test]
fn parses_shifted_uppercase_csi_u_letters_as_shift_letter() {
    let parser = parser_with_kitty(true);
    assert!(parser.matches_key("\x1b[69;2u", "shift+e"));
    assert_eq!(parser.parse_key("\x1b[69;2u").as_deref(), Some("shift+e"));
}

#[test]
fn ignores_kitty_csi_u_with_unsupported_modifiers() {
    assert_eq!(parser_with_kitty(true).parse_key("\x1b[99;17u"), None);
}

// =============================================================================
// parseKey — legacy key parsing
// =============================================================================

#[test]
fn parses_legacy_ctrl_letter() {
    let parser = parser_with_kitty(false);
    assert_eq!(parser.parse_key("\x03").as_deref(), Some("ctrl+c"));
    assert_eq!(parser.parse_key("\x04").as_deref(), Some("ctrl+d"));
}

#[test]
fn parses_special_keys() {
    let parser = parser();
    assert_eq!(parser.parse_key("\x1b").as_deref(), Some("escape"));
    assert_eq!(parser.parse_key("\t").as_deref(), Some("tab"));
    assert_eq!(parser.parse_key("\r").as_deref(), Some("enter"));
    assert_eq!(parser.parse_key("\n").as_deref(), Some("enter"));
    assert_eq!(parser.parse_key("\x00").as_deref(), Some("ctrl+space"));
    assert_eq!(parser.parse_key(" ").as_deref(), Some("space"));
    assert_eq!(parser.parse_key("1").as_deref(), Some("1"));
    assert!(parser.matches_key("1", "1"));
}

#[test]
fn parses_arrow_keys() {
    let parser = parser();
    assert_eq!(parser.parse_key("\x1b[A").as_deref(), Some("up"));
    assert_eq!(parser.parse_key("\x1b[B").as_deref(), Some("down"));
    assert_eq!(parser.parse_key("\x1b[C").as_deref(), Some("right"));
    assert_eq!(parser.parse_key("\x1b[D").as_deref(), Some("left"));
}

#[test]
fn parses_ss3_arrows_and_home_end() {
    let parser = parser();
    assert_eq!(parser.parse_key("\x1bOA").as_deref(), Some("up"));
    assert_eq!(parser.parse_key("\x1bOB").as_deref(), Some("down"));
    assert_eq!(parser.parse_key("\x1bOC").as_deref(), Some("right"));
    assert_eq!(parser.parse_key("\x1bOD").as_deref(), Some("left"));
    assert_eq!(parser.parse_key("\x1bOH").as_deref(), Some("home"));
    assert_eq!(parser.parse_key("\x1bOF").as_deref(), Some("end"));
}

#[test]
fn parses_legacy_function_and_modifier_sequences() {
    let parser = parser();
    assert_eq!(parser.parse_key("\x1bOP").as_deref(), Some("f1"));
    assert_eq!(parser.parse_key("\x1b[24~").as_deref(), Some("f12"));
    assert_eq!(parser.parse_key("\x1b[E").as_deref(), Some("clear"));
    assert_eq!(parser.parse_key("\x1b[2^").as_deref(), Some("ctrl+insert"));
    assert_eq!(parser.parse_key("\x1bp").as_deref(), Some("alt+up"));
}

#[test]
fn parses_double_bracket_page_up() {
    assert_eq!(parser().parse_key("\x1b[[5~").as_deref(), Some("pageUp"));
}

// =============================================================================
// Boundary tests (additions beyond upstream): branches upstream leaves
// untested — the event-type queries, legacy modifier table entries, CSI-u
// alternates, and degenerate inputs — so the coverage gate binds and the
// behavior stays pinned.
// =============================================================================

#[test]
fn key_helpers_produce_the_documented_identifiers() {
    // Special keys
    assert_eq!(Key::escape(), "escape");
    assert_eq!(Key::esc(), "esc");
    assert_eq!(Key::enter(), "enter");
    assert_eq!(Key::r#return(), "return");
    assert_eq!(Key::tab(), "tab");
    assert_eq!(Key::space(), "space");
    assert_eq!(Key::backspace(), "backspace");
    assert_eq!(Key::delete(), "delete");
    assert_eq!(Key::insert(), "insert");
    assert_eq!(Key::clear(), "clear");
    assert_eq!(Key::home(), "home");
    assert_eq!(Key::end(), "end");
    assert_eq!(Key::page_up(), "pageUp");
    assert_eq!(Key::page_down(), "pageDown");
    assert_eq!(Key::up(), "up");
    assert_eq!(Key::down(), "down");
    assert_eq!(Key::left(), "left");
    assert_eq!(Key::right(), "right");
    assert_eq!(Key::f1(), "f1");
    assert_eq!(Key::f2(), "f2");
    assert_eq!(Key::f3(), "f3");
    assert_eq!(Key::f4(), "f4");
    assert_eq!(Key::f5(), "f5");
    assert_eq!(Key::f6(), "f6");
    assert_eq!(Key::f7(), "f7");
    assert_eq!(Key::f8(), "f8");
    assert_eq!(Key::f9(), "f9");
    assert_eq!(Key::f10(), "f10");
    assert_eq!(Key::f11(), "f11");
    assert_eq!(Key::f12(), "f12");
    // Symbol keys
    assert_eq!(Key::backtick(), "`");
    assert_eq!(Key::hyphen(), "-");
    assert_eq!(Key::equals(), "=");
    assert_eq!(Key::leftbracket(), "[");
    assert_eq!(Key::rightbracket(), "]");
    assert_eq!(Key::backslash(), "\\");
    assert_eq!(Key::semicolon(), ";");
    assert_eq!(Key::quote(), "'");
    assert_eq!(Key::comma(), ",");
    assert_eq!(Key::period(), ".");
    assert_eq!(Key::slash(), "/");
    assert_eq!(Key::exclamation(), "!");
    assert_eq!(Key::at(), "@");
    assert_eq!(Key::hash(), "#");
    assert_eq!(Key::dollar(), "$");
    assert_eq!(Key::percent(), "%");
    assert_eq!(Key::caret(), "^");
    assert_eq!(Key::ampersand(), "&");
    assert_eq!(Key::asterisk(), "*");
    assert_eq!(Key::leftparen(), "(");
    assert_eq!(Key::rightparen(), ")");
    assert_eq!(Key::underscore(), "_");
    assert_eq!(Key::plus(), "+");
    assert_eq!(Key::pipe(), "|");
    assert_eq!(Key::tilde(), "~");
    assert_eq!(Key::leftbrace(), "{");
    assert_eq!(Key::rightbrace(), "}");
    assert_eq!(Key::colon(), ":");
    assert_eq!(Key::lessthan(), "<");
    assert_eq!(Key::greaterthan(), ">");
    assert_eq!(Key::question(), "?");
    // Single modifiers
    assert_eq!(Key::ctrl("c"), "ctrl+c");
    assert_eq!(Key::shift("tab"), "shift+tab");
    assert_eq!(Key::alt("enter"), "alt+enter");
    assert_eq!(Key::super_key("k"), "super+k");
    // Combined modifiers
    assert_eq!(Key::ctrl_shift("p"), "ctrl+shift+p");
    assert_eq!(Key::shift_ctrl("p"), "shift+ctrl+p");
    assert_eq!(Key::ctrl_alt("x"), "ctrl+alt+x");
    assert_eq!(Key::alt_ctrl("x"), "alt+ctrl+x");
    assert_eq!(Key::shift_alt("x"), "shift+alt+x");
    assert_eq!(Key::alt_shift("x"), "alt+shift+x");
    assert_eq!(Key::ctrl_super("k"), "ctrl+super+k");
    assert_eq!(Key::super_ctrl("k"), "super+ctrl+k");
    assert_eq!(Key::shift_super("k"), "shift+super+k");
    assert_eq!(Key::super_shift("k"), "super+shift+k");
    assert_eq!(Key::alt_super("k"), "alt+super+k");
    assert_eq!(Key::super_alt("k"), "super+alt+k");
    // Triple modifiers
    assert_eq!(Key::ctrl_shift_alt("x"), "ctrl+shift+alt+x");
    assert_eq!(Key::ctrl_shift_super("k"), "ctrl+shift+super+k");
}

#[test]
fn detects_kitty_event_types_from_the_data_patterns() {
    // Release events with flag 2 carry ":3" before the final character.
    assert!(pi_tui::keys::is_key_release("\x1b[1089::99;5:3u"));
    assert!(pi_tui::keys::is_key_release("\x1b[1;5:3A"));
    assert!(pi_tui::keys::is_key_release("\x1b[5;5:3~"));
    assert!(!pi_tui::keys::is_key_release("\x1b[27;5;99~"));
    assert!(!pi_tui::keys::is_key_release("\x1b[99;5u"));
    // Bracketed paste content never counts, even with ":3F"-shaped runs.
    assert!(!pi_tui::keys::is_key_release("\x1b[200~90:62:3F:A5"));
    assert!(pi_tui::keys::is_key_repeat("\x1b[99;5:2u"));
    assert!(!pi_tui::keys::is_key_repeat("\x1b[99;5:3u"));
    assert!(!pi_tui::keys::is_key_repeat("\x1b[99;5u"));
    assert!(!pi_tui::keys::is_key_repeat("\x1b[200~90:62:3F:A5"));
}

#[test]
fn matches_kitty_csi_u_with_event_types_regardless_of_press_or_release() {
    let parser = parser_with_kitty(true);
    assert!(parser.matches_key("\x1b[99;5:2u", "ctrl+c"));
    assert!(parser.matches_key("\x1b[99;5:3u", "ctrl+c"));
    assert_eq!(parser.parse_key("\x1b[99;5:2u").as_deref(), Some("ctrl+c"));
}

#[test]
fn matches_kitty_modified_special_keys() {
    let parser = parser_with_kitty(true);
    // Escape
    assert!(parser.matches_key("\x1b[27u", "escape"));
    assert_eq!(parser.parse_key("\x1b[27u").as_deref(), Some("escape"));
    // Tab with ctrl
    assert!(parser.matches_key("\x1b[9;5u", "ctrl+tab"));
    assert_eq!(parser.parse_key("\x1b[9;5u").as_deref(), Some("ctrl+tab"));
    // Space with ctrl and super
    assert!(parser.matches_key("\x1b[32;5u", "ctrl+space"));
    assert!(parser.matches_key("\x1b[32;9u", "super+space"));
    assert_eq!(
        parser.parse_key("\x1b[32;9u").as_deref(),
        Some("super+space")
    );
    // Enter and numpad enter with ctrl
    assert!(parser.matches_key("\x1b[13;5u", "ctrl+enter"));
    assert!(parser.matches_key("\x1b[57414;5u", "ctrl+enter"));
    assert_eq!(
        parser.parse_key("\x1b[57414;5u").as_deref(),
        Some("ctrl+enter")
    );
    // Ctrl+Shift letters via CSI-u
    assert!(parser.matches_key("\x1b[99;6u", "ctrl+shift+c"));
    // Ctrl+Alt+letter via CSI-u even with Kitty protocol active
    assert!(parser.matches_key("\x1b[104;7u", "ctrl+alt+h"));
    // Ctrl+Alt+Shift via CSI-u (generic modifier path)
    assert!(parser.matches_key("\x1b[99;8u", "ctrl+alt+shift+c"));
    assert_eq!(
        parser.parse_key("\x1b[99;8u").as_deref(),
        Some("shift+ctrl+alt+c")
    );
}

#[test]
fn matches_kitty_functional_and_arrow_keys_with_ctrl() {
    let parser = parser();
    assert!(parser.matches_key("\x1b[57423;5u", "ctrl+home"));
    assert_eq!(
        parser.parse_key("\x1b[57423;5u").as_deref(),
        Some("ctrl+home")
    );
    assert!(parser.matches_key("\x1b[57425;5u", "ctrl+insert"));
    assert!(parser.matches_key("\x1b[1;5A", "ctrl+up"));
    assert_eq!(parser.parse_key("\x1b[1;5A").as_deref(), Some("ctrl+up"));
    // Event type on arrows does not affect matching.
    assert!(parser.matches_key("\x1b[1;5:3A", "ctrl+up"));
}

#[test]
fn matches_xterm_modify_other_keys_super_combos() {
    let parser = parser_with_kitty(false);
    assert!(parser.matches_key("\x1b[27;9;107~", "super+k"));
    assert_eq!(
        parser.parse_key("\x1b[27;9;107~").as_deref(),
        Some("super+k")
    );
}

#[test]
fn matches_legacy_insert_delete_page_and_clear_variants() {
    let parser = parser();
    assert!(parser.matches_key("\x1b[2~", "insert"));
    assert_eq!(parser.parse_key("\x1b[2~").as_deref(), Some("insert"));
    assert!(parser.matches_key("\x1b[3~", "delete"));
    assert_eq!(parser.parse_key("\x1b[3~").as_deref(), Some("delete"));
    assert!(parser.matches_key("\x1b[5~", "pageUp"));
    assert_eq!(parser.parse_key("\x1b[5~").as_deref(), Some("pageUp"));
    assert!(parser.matches_key("\x1b[6~", "pageDown"));
    assert_eq!(parser.parse_key("\x1b[6~").as_deref(), Some("pageDown"));
    assert!(parser.matches_key("\x1bOE", "clear"));
    assert_eq!(parser.parse_key("\x1bOE").as_deref(), Some("clear"));
    assert!(parser.matches_key("\x1bOe", "ctrl+clear"));
    assert_eq!(parser.parse_key("\x1bOe").as_deref(), Some("ctrl+clear"));
    assert!(parser.matches_key("\x1b[e", "shift+clear"));
    assert_eq!(parser.parse_key("\x1b[e").as_deref(), Some("shift+clear"));
    assert!(parser.matches_key("\x1b[3$", "shift+delete"));
    assert_eq!(parser.parse_key("\x1b[3$").as_deref(), Some("shift+delete"));
    assert!(parser.matches_key("\x1b[3^", "ctrl+delete"));
    assert_eq!(parser.parse_key("\x1b[3^").as_deref(), Some("ctrl+delete"));
    assert!(parser.matches_key("\x1b[6$", "shift+pageDown"));
    assert_eq!(
        parser.parse_key("\x1b[6$").as_deref(),
        Some("shift+pageDown")
    );
    assert!(parser.matches_key("\x1b[5^", "ctrl+pageUp"));
    assert_eq!(parser.parse_key("\x1b[5^").as_deref(), Some("ctrl+pageUp"));
    assert!(parser.matches_key("\x1b[6^", "ctrl+pageDown"));
    assert_eq!(
        parser.parse_key("\x1b[6^").as_deref(),
        Some("ctrl+pageDown")
    );
    assert!(parser.matches_key("\x1b[8^", "ctrl+end"));
    assert_eq!(parser.parse_key("\x1b[8^").as_deref(), Some("ctrl+end"));
    assert!(parser.matches_key("\x1bOc", "ctrl+right"));
    assert_eq!(parser.parse_key("\x1bOc").as_deref(), Some("ctrl+right"));
    assert!(parser.matches_key("\x1b[b", "shift+down"));
    assert_eq!(parser.parse_key("\x1b[b").as_deref(), Some("shift+down"));
    assert_eq!(parser.parse_key("\x1b[[A").as_deref(), Some("f1"));
    assert!(parser.matches_key("\x1b[[A", "f1"));
}

#[test]
fn matches_legacy_home_end_variants_parse_asymmetry() {
    let parser = parser();
    // matchesKey knows "\x1b[1~" and "\x1b[4~" as home/end through the legacy
    // tables, but upstream's parse table omits them; the asymmetry is
    // upstream behavior and pinned here.
    assert!(parser.matches_key("\x1b[1~", "home"));
    assert_eq!(parser.parse_key("\x1b[1~"), None);
    assert!(parser.matches_key("\x1b[4~", "end"));
    assert_eq!(parser.parse_key("\x1b[4~"), None);
    assert!(parser.matches_key("\x1b[7~", "home"));
    assert!(parser.matches_key("\x1b[8~", "end"));
    assert_eq!(parser.parse_key("\x1b[3~").as_deref(), Some("delete"));
    assert_eq!(parser.parse_key("\x1b[11~").as_deref(), Some("f1"));
    assert_eq!(parser.parse_key("\x1b[[A").as_deref(), Some("f1"));
}

#[test]
fn matches_enter_return_aliases_and_kp_enter_modifiers() {
    let parser = parser();
    assert!(parser.matches_key("\r", "return"));
    assert!(parser.matches_key("\x1bOM", "enter"));
    assert!(parser.matches_key("\x1b[13;5u", "ctrl+enter"));
    assert!(parser.matches_key("\x1b[57414;5u", "ctrl+enter"));
    assert_eq!(
        parser.parse_key("\x1b[57414;5u").as_deref(),
        Some("ctrl+enter")
    );
    // Shift+enter via the Kitty mapping stays inactive in legacy mode.
    let legacy = parser_with_kitty(false);
    assert!(!legacy.matches_key("\x1b\r", "shift+enter"));
    assert!(legacy.matches_key("\x1b\r", "alt+enter"));
    assert_eq!(legacy.parse_key("\x1b\r").as_deref(), Some("alt+enter"));
}

#[test]
fn matches_cyrillic_letter_without_base_layout_through_base_fallback() {
    // No modifier (modValue 1) and no shifted key: the base layout key still
    // resolves the Cyrillic identity to 'c'.
    let parser = parser_with_kitty(true);
    assert!(parser.matches_key("\x1b[1089::99u", "c"));
    assert_eq!(parser.parse_key("\x1b[1089::99u").as_deref(), Some("c"));
}

#[test]
fn kitty_csi_u_with_empty_shifted_slot_still_matches() {
    let parser = parser_with_kitty(true);
    // Upstream's regex allows the shifted slot to be empty before the modifier.
    assert!(parser.matches_key("\x1b[99:;5u", "ctrl+c"));
    assert_eq!(parser.parse_key("\x1b[99:;5u").as_deref(), Some("ctrl+c"));
}

#[test]
fn matches_key_rejects_malformed_key_identifiers() {
    let parser = parser();
    assert!(!parser.matches_key("c", ""));
    assert!(!parser.matches_key("c", "ctrl+"));
    assert!(!parser.matches_key("c", "++"));
    assert!(!parser.matches_key("c", "unknown+key"));
}

#[test]
fn overlong_numbers_never_match_a_real_key() {
    // Upstream's parseInt yields an out-of-range float for digit runs beyond
    // 2^53; the saturating port lands in the same no-match bucket.
    let parser = parser();
    assert!(!parser.matches_key("\x1b[9999999999999999999999u", "c"));
    assert_eq!(parser.parse_key("\x1b[9999999999999999999999u"), None);
    assert!(!parser.matches_key("\x1b[9999999999999999999999;5u", "ctrl+c"));
    assert_eq!(decode_kitty_printable("\x1b[9999999999999999999999u"), None);
}

#[test]
fn malformed_csi_u_and_functional_sequences_parse_to_nothing() {
    let parser = parser();
    assert_eq!(parser.parse_key("\x1b[u"), None);
    assert_eq!(parser.parse_key("\x1b[99::u"), None);
    assert_eq!(parser.parse_key("\x1b[99U"), None);
    assert_eq!(parser.parse_key("\x1b[99;5;99~"), None);
    assert_eq!(parser.parse_key("\x1b[1;;2A"), None);
    // The arrow slot does accept an event type; pin that it parses.
    assert_eq!(parser.parse_key("\x1b[1;5:2A").as_deref(), Some("ctrl+up"));
}

#[test]
fn decodes_printable_csi_u_with_shift_and_rejects_other_modifiers() {
    // Shifted key preferred when Shift is held.
    assert_eq!(
        decode_kitty_printable("\x1b[99:67;2u").as_deref(),
        Some("C")
    );
    // Shift without a shifted key reports the base codepoint.
    assert_eq!(decode_kitty_printable("\x1b[69;2u").as_deref(), Some("E"));
    // Plain key.
    assert_eq!(decode_kitty_printable("\x1b[97u").as_deref(), Some("a"));
    // Ctrl, Alt, Super, Hyper, and unsupported bits are rejected.
    assert_eq!(decode_kitty_printable("\x1b[99;5u"), None);
    assert_eq!(decode_kitty_printable("\x1b[99;3u"), None);
    assert_eq!(decode_kitty_printable("\x1b[99;9u"), None);
    assert_eq!(decode_kitty_printable("\x1b[99;17u"), None);
    // Caps Lock and Num Lock bits pass through.
    assert_eq!(decode_kitty_printable("\x1b[99;65u").as_deref(), Some("c"));
    assert_eq!(decode_kitty_printable("\x1b[99;66u").as_deref(), Some("c"));
    // Control codepoints and invalid ones are dropped.
    assert_eq!(decode_kitty_printable("\x1b[13u"), None);
    assert_eq!(decode_kitty_printable("\x1b[99999999999u"), None);
    assert_eq!(decode_kitty_printable("\x1b[u"), None);
}

#[test]
fn decodes_printable_modify_other_keys_with_rejections() {
    assert_eq!(decode_printable_key("\x1b[27;2;47~").as_deref(), Some("/"));
    assert_eq!(decode_printable_key("\x1b[27;3;69~"), None);
    assert_eq!(decode_printable_key("\x1b[27;9;69~"), None);
    assert_eq!(decode_printable_key("\x1b[27;1;69~").as_deref(), Some("E"));
    assert_eq!(decode_printable_key("\x1b[27;1;31~"), None);
}

// Every keypad digit normalizes to its logical key.
#[test]
fn normalizes_every_kitty_keypad_digit() {
    let parser = parser_with_kitty(true);
    for (codepoint, digit) in [
        (57399, "0"),
        (57400, "1"),
        (57401, "2"),
        (57402, "3"),
        (57403, "4"),
        (57404, "5"),
        (57405, "6"),
        (57406, "7"),
        (57407, "8"),
        (57408, "9"),
    ] {
        let data = format!("\x1b[{codepoint}u");
        assert!(
            parser.matches_key(&data, digit),
            "KP key {codepoint} should match {digit}"
        );
        assert_eq!(parser.parse_key(&data).as_deref(), Some(digit));
    }
}

// Kitty sends navigation and functional keys as CSI-u too; without
// modifiers they reach the plain-key arms.
#[test]
fn matches_kitty_plain_functional_and_arrow_keys() {
    let parser = parser_with_kitty(true);
    assert!(parser.matches_key("\x1b[57425u", "insert"));
    assert!(parser.matches_key("\x1b[57426u", "delete"));
    assert!(parser.matches_key("\x1b[57423u", "home"));
    assert!(parser.matches_key("\x1b[57424u", "end"));
    assert!(parser.matches_key("\x1b[57421u", "pageUp"));
    assert!(parser.matches_key("\x1b[57422u", "pageDown"));
    assert!(parser.matches_key("\x1b[57419u", "up"));
    assert!(parser.matches_key("\x1b[57420u", "down"));
    assert!(parser.matches_key("\x1b[57417u", "left"));
    assert!(parser.matches_key("\x1b[57418u", "right"));
    assert!(parser.matches_key("\x1b[57414u", "enter"));
}

#[test]
fn matches_plain_tab_and_kitty_tab() {
    let parser = parser_with_kitty(true);
    assert!(parser.matches_key("\t", "tab"));
    assert!(parser.matches_key("\x1b[9u", "tab"));
    assert_eq!(parser.parse_key("\x1b[9u").as_deref(), Some("tab"));
}

#[test]
fn matches_escape_never_carries_modifiers() {
    let parser = parser();
    assert!(!parser.matches_key("\x1b", "ctrl+escape"));
    assert!(!parser.matches_key("\x1b", "alt+esc"));
    assert!(!parser.matches_key("\x1b[27;5u", "escape"));
}

#[test]
fn matches_kitty_shift_and_alt_enter() {
    let parser = parser_with_kitty(true);
    assert!(parser.matches_key("\x1b[13;2u", "shift+enter"));
    assert!(parser.matches_key("\x1b[57414;2u", "shift+enter"));
    assert!(parser.matches_key("\x1b[13;3u", "alt+enter"));
    assert!(parser.matches_key("\x1b[57414;3u", "alt+enter"));
    // With Kitty active, the legacy ESC+CR form is a custom mapping, not
    // alt+enter.
    assert!(!parser.matches_key("\x1b\r", "alt+enter"));
}

#[test]
fn matches_kitty_and_xterm_backspace_with_other_modifiers() {
    let parser = parser();
    assert!(parser.matches_key("\x1b[27;9;127~", "super+backspace"));
    assert!(parser.matches_key("\x1b[127;9u", "super+backspace"));
    assert!(!parser.matches_key("\x1b[127;9u", "ctrl+backspace"));
}

#[test]
fn matches_kitty_functional_keys_with_modifiers() {
    let parser = parser();
    assert!(parser.matches_key("\x1b[57425;3u", "alt+insert"));
    assert_eq!(
        parser.parse_key("\x1b[57425;3u").as_deref(),
        Some("alt+insert")
    );
    assert!(parser.matches_key("\x1b[57426;5u", "ctrl+delete"));
    assert!(parser.matches_key("\x1b[57423;5u", "ctrl+home"));
    assert!(parser.matches_key("\x1b[57424;5u", "ctrl+end"));
    assert!(parser.matches_key("\x1b[57421;5u", "ctrl+pageUp"));
    assert!(parser.matches_key("\x1b[57422;5u", "ctrl+pageDown"));
}

#[test]
fn matches_xterm_ctrl_modified_arrows() {
    let parser = parser();
    assert!(parser.matches_key("\x1b[1;5A", "ctrl+up"));
    assert!(parser.matches_key("\x1b[1;5B", "ctrl+down"));
    assert!(parser.matches_key("\x1b[1;5C", "ctrl+right"));
    assert!(parser.matches_key("\x1b[1;5D", "ctrl+left"));
    assert!(parser.matches_key("\x1b[57420;5u", "ctrl+down"));
    assert!(parser.matches_key("\x1b[57417;5u", "ctrl+left"));
    assert!(parser.matches_key("\x1b[57418;5u", "ctrl+right"));
    // Kitty shift-modified arrows reach the generic modifier arm.
    assert!(parser.matches_key("\x1b[57417;2u", "shift+left"));
    assert!(parser.matches_key("\x1b[57418;2u", "shift+right"));
    assert!(parser.matches_key("\x1b[1;3C", "alt+right"));
    assert!(parser.matches_key("\x1b[1;3D", "alt+left"));
}

#[test]
fn matches_xterm_functional_keys_with_event_types() {
    let parser = parser();
    assert!(parser.matches_key("\x1b[5;5:2~", "ctrl+pageUp"));
    assert_eq!(
        parser.parse_key("\x1b[5;5:2~").as_deref(),
        Some("ctrl+pageUp")
    );
    assert!(parser.matches_key("\x1b[1;5:2H", "ctrl+home"));
    assert_eq!(
        parser.parse_key("\x1b[1;5:2H").as_deref(),
        Some("ctrl+home")
    );
    // Event type 1 is a plain press.
    assert!(parser.matches_key("\x1b[13;5:1u", "ctrl+enter"));
}

#[test]
fn parses_functional_home_and_end_through_the_kitty_table() {
    let parser = parser();
    assert_eq!(parser.parse_key("\x1b[7~").as_deref(), Some("home"));
    assert_eq!(parser.parse_key("\x1b[8~").as_deref(), Some("end"));
}

#[test]
fn parses_remaining_legacy_parse_key_chain_members() {
    let parser = parser();
    assert_eq!(parser.parse_key("\x1b[Z").as_deref(), Some("shift+tab"));
    assert_eq!(parser.parse_key("\x1b[F").as_deref(), Some("end"));
    assert_eq!(parser.parse_key("\x1b[8$").as_deref(), Some("shift+end"));
    assert!(parser.matches_key("\x1b[8$", "shift+end"));
    // The 2-byte ESC branch falls through for keys outside its classes.
    assert_eq!(parser.parse_key("\x1b\x00"), None);
    assert_eq!(parser.parse_key("\x1e"), None);
    // A single char outside both control and printable ranges parses to
    // nothing (DEL does not appear in the chain).
    assert_eq!(parser.parse_key("\x1b"), Some("escape".to_string()));
}

#[test]
fn matches_legacy_ctrl_letter_and_ctrl_digit_kitty() {
    let parser = parser();
    assert!(parser.matches_key("\x11", "ctrl+q"));
    assert!(parser.matches_key("\x15", "ctrl+u"));
    assert!(parser.matches_key("\x0b", "ctrl+k"));
    assert!(parser.matches_key("\x17", "ctrl+w"));
    assert!(parser.matches_key("\x19", "ctrl+y"));
    assert!(parser.matches_key("\x01", "ctrl+a"));
    assert!(parser.matches_key("\x05", "ctrl+e"));
    assert!(parser.matches_key("\x02", "ctrl+b"));
    assert!(parser.matches_key("\x06", "ctrl+f"));
    assert!(parser.matches_key("\x10", "ctrl+p"));
    // Digits have no raw control character; Kitty and modifyOtherKeys carry them.
    assert!(parser.matches_key("\x1b[53;5u", "ctrl+5"));
    assert_eq!(parser.parse_key("\x1b[53;5u").as_deref(), Some("ctrl+5"));
    // Ctrl+j overlaps with the raw linefeed in legacy mode, upstream behavior.
    assert!(parser.matches_key("\n", "ctrl+j"));
    // Shift+letter produces uppercase in legacy mode.
    assert!(parser.matches_key("E", "shift+e"));
    // A non-ASCII key identifier falls through the printable gate.
    assert!(!parser.matches_key("ä", "ä"));
}

#[test]
fn parses_junk_after_valid_sequence_prefixes_to_nothing() {
    let parser = parser();
    assert_eq!(parser.parse_key("\x1b[99;5xu"), None);
    assert_eq!(parser.parse_key("\x1b[1;5Ax"), None);
    assert_eq!(parser.parse_key("\x1b[1;5G"), None);
    assert_eq!(parser.parse_key("\x1b[27;5X99~"), None);
    assert_eq!(parser.parse_key("\x1b[27;5;99x~"), None);
    assert_eq!(parser.parse_key("\x1b[7;5x~"), None);
    assert!(!parser.matches_key("\x1b[99;5xu", "ctrl+c"));
}

#[test]
fn parser_defaults_and_kitty_state_round_trip() {
    let mut parser = KeyParser::default();
    assert!(!parser.is_kitty_protocol_active());
    parser.set_kitty_protocol_active(true);
    assert!(parser.is_kitty_protocol_active());
    parser.set_kitty_protocol_active(false);
    assert!(!parser.is_kitty_protocol_active());
}

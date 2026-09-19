//! Port of the parser suites in `packages/tui/test/terminal-colors.test.ts` —
//! 1:1 against upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#41).
//!
//! The `TUI.queryTerminalBackgroundColor` describe block upstream carries in
//! the same file exercises `TuiMainScreen`, which lands with its own port
//! ticket; the parser suites port here.

use pi_tui::terminal_colors::{
    RgbColor, TerminalColorScheme, is_osc11_background_color_response,
    parse_osc11_background_color, parse_terminal_color_scheme_report,
};

#[test]
fn parses_16_bit_osc_11_rgb_responses() {
    assert_eq!(
        parse_osc11_background_color("\x1b]11;rgb:0000/8000/ffff\x07"),
        Some(RgbColor {
            r: 0,
            g: 128,
            b: 255
        })
    );
}

#[test]
fn parses_osc_11_hex_responses() {
    assert_eq!(
        parse_osc11_background_color("\x1b]11;#ffffff\x1b\\"),
        Some(RgbColor {
            r: 255,
            g: 255,
            b: 255
        })
    );
    assert_eq!(
        parse_osc11_background_color("\x1b]11;#000000\x07"),
        Some(RgbColor { r: 0, g: 0, b: 0 })
    );
}

#[test]
fn rejects_non_strict_osc_11_responses() {
    assert_eq!(parse_osc11_background_color("x\x1b]11;#ffffff\x07"), None);
    assert_eq!(parse_osc11_background_color("\x1b]10;#ffffff\x07"), None);
    assert_eq!(parse_osc11_background_color("\x1b]11;#ffffff\x07x"), None);
}

#[test]
fn parses_color_scheme_reports() {
    assert_eq!(
        parse_terminal_color_scheme_report("\x1b[?997;1n"),
        Some(TerminalColorScheme::Dark)
    );
    assert_eq!(
        parse_terminal_color_scheme_report("\x1b[?997;2n"),
        Some(TerminalColorScheme::Light)
    );
    assert_eq!(
        parse_terminal_color_scheme_report("\x1b[?997;2n\x1b[?997;1n\x1b[?997;1n"),
        Some(TerminalColorScheme::Dark)
    );
    assert_eq!(
        parse_terminal_color_scheme_report("\x1b[?997;1n\x1b[?997;2n\x1b[?997;2n"),
        Some(TerminalColorScheme::Light)
    );
    assert_eq!(parse_terminal_color_scheme_report("\x1b[?997;3n"), None);
    assert_eq!(parse_terminal_color_scheme_report("\x1b[?996n"), None);
    assert_eq!(parse_terminal_color_scheme_report("x\x1b[?997;1n"), None);
}

/// Boundary: the strict-response predicate and the parser agree, and
/// unparsable replies are rejected without tripping the trim.
#[test]
fn osc11_strictness_matches_the_response_gate() {
    assert!(is_osc11_background_color_response("\x1b]11;#ffffff\x1b\\"));
    assert!(is_osc11_background_color_response(
        "\x1b]11;  #ffffff  \x07"
    ));
    assert!(!is_osc11_background_color_response("\x1b]11;#ffffff"));
    assert!(!is_osc11_background_color_response("\x1b]11;\x07x"));

    assert_eq!(
        parse_osc11_background_color("\x1b]11;not-a-color\x07"),
        None,
        "a strict response with an unparsable value resolves nothing"
    );
}

/// Boundary: the `XParseColor` form scales each channel from its own radix.
#[test]
fn parses_osc_11_rgb_channels_of_mixed_widths() {
    assert_eq!(
        parse_osc11_background_color("\x1b]11;rgb:ff/80/0\x07"),
        Some(RgbColor {
            r: 255,
            g: 128,
            b: 0
        })
    );
    assert_eq!(
        parse_osc11_background_color("\x1b]11;rgb:zz/80/ff\x07"),
        None,
        "non-hex channel is rejected"
    );
    assert_eq!(
        parse_osc11_background_color("\x1b]11;rgb:ff/80\x07"),
        None,
        "two channels are not a color"
    );
}

/// Boundary: `rgba:` prefixes are stripped exactly as upstream's anchored
/// replace, uppercase included.
#[test]
fn strips_rgba_prefixes_case_insensitively() {
    assert_eq!(
        parse_osc11_background_color("\x1b]11;RGB:ff/00/ff\x07"),
        Some(RgbColor {
            r: 255,
            g: 0,
            b: 255
        })
    );
    assert_eq!(
        parse_osc11_background_color("\x1b]11;RGBA:ff/00/ff\x07"),
        Some(RgbColor {
            r: 255,
            g: 0,
            b: 255
        })
    );
}

/// Boundary: a 12-digit hex response scales each 16-bit channel, and a
/// non-hex tail after `#` rejects.
#[test]
fn parses_osc_11_sixteen_bit_hex_and_rejects_bad_hex() {
    assert_eq!(
        parse_osc11_background_color("\x1b]11;#ffff8000ffff\x1b\\"),
        Some(RgbColor {
            r: 255,
            g: 128,
            b: 255
        })
    );
    assert_eq!(
        parse_osc11_background_color("\x1b]11;#ffffff0\x07"),
        None,
        "seven digits are neither the 6 nor the 12 digit form"
    );
    assert_eq!(parse_osc11_background_color("\x1b]11;#\x07"), None);
}

/// Boundary: absurdly long channels exceed the radix the port scales from
/// and reject, where upstream's `parseInt` would answer with float garbage.
#[test]
fn overlong_hex_channels_are_rejected() {
    let long_channel = "f".repeat(17);
    assert_eq!(
        parse_osc11_background_color(&format!("\x1b]11;rgb:{long_channel}/0/0\x07")),
        None
    );
    assert_eq!(
        parse_osc11_background_color("\x1b]11;rgb:ffff8000ffff/0/0\x07"),
        Some(RgbColor { r: 255, g: 0, b: 0 }),
        "a 12-digit channel is a valid 16-bit value; upstream ignores any fourth channel"
    );
}

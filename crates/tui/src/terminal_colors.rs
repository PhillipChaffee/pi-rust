//! Terminal color queries and reports, ported from
//! `packages/tui/src/terminal-colors.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#41).
//!
//! Upstream's OSC 11 background-color response parser and the
//! `CSI ? 997 ; n` color-scheme report parser, both 1:1.

use std::sync::LazyLock;

use regex::Regex;

use crate::utils::static_regex;

/// An RGB color the terminal reports back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgbColor {
    /// Red channel, 0-255.
    pub r: u8,
    /// Green channel, 0-255.
    pub g: u8,
    /// Blue channel, 0-255.
    pub b: u8,
}

/// The color scheme a terminal reports through `CSI ? 997 ; <n> n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalColorScheme {
    /// The scheme report `1` names.
    Dark,
    /// The scheme report `2` names.
    Light,
}

fn hex_to_rgb(hex: &str) -> Option<RgbColor> {
    let normalized = hex.strip_prefix('#').unwrap_or(hex);
    let r = u8::from_str_radix(&normalized[0..2], 16).ok()?;
    let g = u8::from_str_radix(&normalized[2..4], 16).ok()?;
    let b = u8::from_str_radix(&normalized[4..6], 16).ok()?;
    Some(RgbColor { r, g, b })
}

/// Restates upstream's `parseOscHexChannel`: a channel is a hex run scaled
/// from its own radix to 0-255, so a 16-bit `ffff` measures 255 and a
/// 16-bit `8000` measures 128. Upstream runs the arithmetic through
/// `parseInt` floats; the port refuses channels wider than a `u64` can
/// measure exactly, which only absurd OSC replies reach.
fn parse_osc_hex_channel(channel: &str) -> Option<u8> {
    if channel.is_empty() || !channel.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let max = u64::checked_pow(16, u32::try_from(channel.len()).ok()?)?.checked_sub(1)?;
    let value = u64::from_str_radix(channel, 16).ok()?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "upstream's parseInt already loses precision beyond 2^53, and the scaled ratio stays exact in f64 for every channel the response grammar can carry"
    )]
    let scaled = (value as f64 / max as f64) * 255.0;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scaled is bounded to 0.0..=255.0 before rounding, so the cast never truncates a value out of u8 range or flips its sign"
    )]
    let rounded = scaled.round() as u8;
    Some(rounded)
}

const OSC11_BACKGROUND_COLOR_RESPONSE_PATTERN: &str =
    r"(?i)^\x1b\]11;([^\x07\x1b]*)(?:\x07|\x1b\\)$";
const COLOR_SCHEME_REPORT_PATTERN: &str = r"^(?:\x1b\[\?997;(1|2)n)+$";
const HEX6_PATTERN: &str = r"(?i)^[0-9a-f]{6}$";
const HEX12_PATTERN: &str = r"(?i)^[0-9a-f]{12}$";

fn osc11_pattern() -> &'static Regex {
    static PATTERN: LazyLock<Regex> =
        LazyLock::new(|| static_regex(OSC11_BACKGROUND_COLOR_RESPONSE_PATTERN));
    &PATTERN
}

fn color_scheme_pattern() -> &'static Regex {
    static PATTERN: LazyLock<Regex> = LazyLock::new(|| static_regex(COLOR_SCHEME_REPORT_PATTERN));
    &PATTERN
}

fn hex_pattern(digits: usize) -> &'static Regex {
    static HEX6: LazyLock<Regex> = LazyLock::new(|| static_regex(HEX6_PATTERN));
    static HEX12: LazyLock<Regex> = LazyLock::new(|| static_regex(HEX12_PATTERN));
    if digits == 6 { &HEX6 } else { &HEX12 }
}

/// Whether the data is a strict OSC 11 background-color response.
#[must_use]
pub fn is_osc11_background_color_response(data: &str) -> bool {
    osc11_pattern().is_match(data)
}

/// Parse an OSC 11 background-color response into RGB.
///
/// Accepts the `#rrggbb` hex form and the `XParseColor`
/// `rgb:<red>/<green>/<blue>` form, where each channel is a hex run scaled
/// from its own radix.
#[must_use]
pub fn parse_osc11_background_color(data: &str) -> Option<RgbColor> {
    let captures = osc11_pattern().captures(data)?;

    let value = captures[1].trim();
    if let Some(hex) = value.strip_prefix('#') {
        if hex_pattern(6).is_match(hex) {
            return hex_to_rgb(value);
        }
        if hex_pattern(12).is_match(hex) {
            return Some(RgbColor {
                r: parse_osc_hex_channel(&hex[0..4])?,
                g: parse_osc_hex_channel(&hex[4..8])?,
                b: parse_osc_hex_channel(&hex[8..12])?,
            });
        }
        return None;
    }

    let rgb_value = strip_rgba_prefix(value);
    let mut channels = rgb_value.split('/');
    let (Some(red), Some(green), Some(blue)) = (channels.next(), channels.next(), channels.next())
    else {
        return None;
    };
    let (Some(r), Some(g), Some(b)) = (
        parse_osc_hex_channel(red),
        parse_osc_hex_channel(green),
        parse_osc_hex_channel(blue),
    ) else {
        return None;
    };
    Some(RgbColor { r, g, b })
}

/// Restates upstream's `value.replace(/^rgba?:/i, "")`.
fn strip_rgba_prefix(value: &str) -> &str {
    let bytes = value.as_bytes();
    let starts_with_rgba =
        bytes.len() >= 5 && bytes[..4].eq_ignore_ascii_case(b"rgba") && bytes[4] == b':';
    let starts_with_rgb =
        bytes.len() >= 4 && bytes[..3].eq_ignore_ascii_case(b"rgb") && bytes[3] == b':';
    if starts_with_rgba {
        &value[5..]
    } else if starts_with_rgb {
        &value[4..]
    } else {
        value
    }
}

/// Parse a `CSI ? 997 ; <n> n` color-scheme report; repeated reports are
/// accepted and the last one wins, exactly as the upstream regex's repeated
/// capture group behaves.
#[must_use]
pub fn parse_terminal_color_scheme_report(data: &str) -> Option<TerminalColorScheme> {
    let captures = color_scheme_pattern().captures(data)?;
    Some(if &captures[1] == "2" {
        TerminalColorScheme::Light
    } else {
        TerminalColorScheme::Dark
    })
}

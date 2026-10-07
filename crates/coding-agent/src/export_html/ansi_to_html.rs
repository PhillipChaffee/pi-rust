//! ANSI escape code to HTML converter, upstream's
//! `src/core/export-html/ansi-to-html.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Converts terminal ANSI color/style codes to HTML with inline styles:
//! standard and bright foreground/background colors, the 256-color palette,
//! RGB true color, the bold/dim/italic/underline text styles, and reset.

use std::fmt::Write as _;

/// Standard ANSI color palette (0-15), upstream's `ANSI_COLORS`.
const ANSI_COLORS: [&str; 16] = [
    "#000000", "#800000", "#008000", "#808000", "#000080", "#800080", "#008080", "#c0c0c0",
    "#808080", "#ff0000", "#00ff00", "#ffff00", "#0000ff", "#ff00ff", "#00ffff", "#ffffff",
];

/// Convert 256-color index to hex, upstream's `color256ToHex`: the 16
/// standard colors, the 6x6x6 cube (16-231), and the 24-step grayscale ramp.
fn color256_to_hex(index: usize) -> String {
    if index < 16 {
        return ANSI_COLORS[index].to_owned();
    }
    if index < 232 {
        let cube_index = index - 16;
        let (r, g, b) = (cube_index / 36, (cube_index % 36) / 6, cube_index % 6);
        let to_hex = |n: usize| format!("{:02x}", if n == 0 { 0 } else { 55 + n * 40 });
        return format!("#{}{}{}", to_hex(r), to_hex(g), to_hex(b));
    }
    let gray = 8 + (index - 232) * 10;
    format!("#{gray:02x}{gray:02x}{gray:02x}")
}

/// Escape HTML special characters, upstream's `escapeHtml`.
pub(crate) fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#039;")
}

/// The SGR style state, upstream's `TextStyle`.
#[derive(Clone, Default, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the four flags restate upstream's TextStyle shape one field per wire flag"
)]
struct TextStyle {
    fg: Option<String>,
    bg: Option<String>,
    bold: bool,
    dim: bool,
    italic: bool,
    underline: bool,
}

/// The inline CSS declaration for a style, upstream's `styleToInlineCSS`.
fn style_to_inline_css(style: &TextStyle) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(fg) = &style.fg {
        parts.push(format!("color:{fg}"));
    }
    if let Some(bg) = &style.bg {
        parts.push(format!("background-color:{bg}"));
    }
    if style.bold {
        parts.push("font-weight:bold".to_owned());
    }
    if style.dim {
        parts.push("opacity:0.6".to_owned());
    }
    if style.italic {
        parts.push("font-style:italic".to_owned());
    }
    if style.underline {
        parts.push("text-decoration:underline".to_owned());
    }
    parts.join(";")
}

const fn has_style(style: &TextStyle) -> bool {
    style.fg.is_some()
        || style.bg.is_some()
        || style.bold
        || style.dim
        || style.italic
        || style.underline
}

/// Parse ANSI SGR (Select Graphic Rendition) codes and update the style,
/// upstream's `applySgrCode`.
fn apply_sgr_code(params: &[usize], style: &mut TextStyle) {
    let mut i = 0;
    while i < params.len() {
        let code = params[i];
        match code {
            0 => {
                *style = TextStyle::default();
            }
            1 => style.bold = true,
            2 => style.dim = true,
            3 => style.italic = true,
            4 => style.underline = true,
            22 => {
                style.bold = false;
                style.dim = false;
            }
            23 => style.italic = false,
            24 => style.underline = false,
            30..=37 => style.fg = Some(ANSI_COLORS[code - 30].to_owned()),
            38 => {
                // Extended foreground color: 38;5;N or 38;2;R;G;B.
                if params.get(i + 1) == Some(&5) && params.len() > i + 2 {
                    style.fg = Some(color256_to_hex(params[i + 2]));
                    i += 2;
                } else if params.get(i + 1) == Some(&2) && params.len() > i + 4 {
                    style.fg = Some(format!(
                        "rgb({},{},{})",
                        params[i + 2],
                        params[i + 3],
                        params[i + 4]
                    ));
                    i += 4;
                }
            }
            39 => style.fg = None,
            40..=47 => style.bg = Some(ANSI_COLORS[code - 40].to_owned()),
            48 => {
                if params.get(i + 1) == Some(&5) && params.len() > i + 2 {
                    style.bg = Some(color256_to_hex(params[i + 2]));
                    i += 2;
                } else if params.get(i + 1) == Some(&2) && params.len() > i + 4 {
                    style.bg = Some(format!(
                        "rgb({},{},{})",
                        params[i + 2],
                        params[i + 3],
                        params[i + 4]
                    ));
                    i += 4;
                }
            }
            49 => style.bg = None,
            90..=97 => style.fg = Some(ANSI_COLORS[code - 90 + 8].to_owned()),
            100..=107 => style.bg = Some(ANSI_COLORS[code - 100 + 8].to_owned()),
            // Ignore unrecognized codes.
            _ => {}
        }
        i += 1;
    }
}

/// Convert ANSI-escaped text to HTML with inline styles, upstream's
/// `ansiToHtml`: escape sequences split the text into styled spans.
#[must_use]
pub fn ansi_to_html(text: &str) -> String {
    let mut style = TextStyle::default();
    let mut result = String::new();
    let mut in_span = false;

    let bytes = text.as_bytes();
    let mut last_index = 0usize;
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        // Match ESC [ params 'm', upstream's /\x1b\[([\d;]*)m/g.
        if bytes[cursor] != 0x1b || bytes.get(cursor + 1) != Some(&b'[') {
            cursor += 1;
            continue;
        }
        let mut end = cursor + 2;
        while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b';') {
            end += 1;
        }
        if bytes.get(end) != Some(&b'm') {
            cursor += 1;
            continue;
        }

        let before_text = &text[last_index..cursor];
        if !before_text.is_empty() {
            result.push_str(&escape_html(before_text));
        }
        if in_span {
            result.push_str("</span>");
            in_span = false;
        }

        let param_str = &text[cursor + 2..end];
        let params: Vec<usize> = if param_str.is_empty() {
            vec![0]
        } else {
            param_str
                .split(';')
                .map(|part| part.parse::<usize>().unwrap_or(0))
                .collect()
        };
        apply_sgr_code(&params, &mut style);

        if has_style(&style) {
            let _ = write!(result, "<span style=\"{}\">", style_to_inline_css(&style));
            in_span = true;
        }

        cursor = end + 1;
        last_index = cursor;
    }

    let remaining_text = &text[last_index..];
    if !remaining_text.is_empty() {
        result.push_str(&escape_html(remaining_text));
    }
    if in_span {
        result.push_str("</span>");
    }

    result
}

/// Convert array of ANSI-escaped lines to HTML, upstream's `ansiLinesToHtml`:
/// each line wraps in a div element, empty renders as `&nbsp;`.
#[must_use]
pub fn ansi_lines_to_html(lines: &[String]) -> String {
    let mut out = String::new();
    for line in lines {
        let rendered = ansi_to_html(line);
        let body = if rendered.is_empty() {
            "&nbsp;"
        } else {
            &rendered
        };
        let _ = write!(out, "<div class=\"ansi-line\">{body}</div>");
    }
    out
}

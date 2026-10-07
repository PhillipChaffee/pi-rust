//! The HTML session export, upstream's `src/core/export-html/` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The five template assets ship embedded byte-verbatim (upstream resolves
//! them from the package tree at runtime; the Rust binary carries them). The
//! placeholder substitution reproduces `String.prototype.replace`'s `$`
//! patterns because upstream's pipeline runs the vendored assets through it:
//! highlight.min.js's `$&` rewrites to the matched placeholder text and the
//! two `$$` sequences collapse to `$`, byte-for-byte, on every export.

pub mod ansi_to_html;
pub mod export;
pub mod tool_renderer;

use std::collections::BTreeMap;

use crate::session_manager::SessionManagerError;

pub use ansi_to_html::{ansi_lines_to_html, ansi_to_html};
pub use export::{
    ExportedTool, RenderedToolHtml, SessionData, export_from_file, export_session_to_html,
    generate_html, pre_render_custom_tools,
};
pub use tool_renderer::{
    CreatedToolHtmlRenderer, RenderedToolResult, ThemeHandle, ToolDefinitionLookup,
    ToolHtmlRenderer, ToolHtmlRendererDeps, ToolRenderCallHook, ToolRenderContext,
    ToolRenderDefinition, ToolRenderResultHook,
};

/// The vendored export template assets, byte-verbatim from the pin.
pub mod assets {
    /// The page skeleton, upstream's `template.html`.
    pub static TEMPLATE_HTML: &str = include_str!("assets/template.html");
    /// The stylesheet, upstream's `template.css`.
    pub static TEMPLATE_CSS: &str = include_str!("assets/template.css");
    /// The client application code, upstream's `template.js`.
    pub static TEMPLATE_JS: &str = include_str!("assets/template.js");
    /// The vendored markdown renderer, upstream's `vendor/marked.min.js`.
    pub static MARKED_JS: &str = include_str!("assets/marked.min.js");
    /// The vendored highlighter, upstream's `vendor/highlight.min.js`.
    pub static HIGHLIGHT_JS: &str = include_str!("assets/highlight.min.js");
}

/// Why an export failed; the message texts are upstream's throws verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportError {
    /// The upstream `Error` message.
    Message(String),
    /// A filesystem operation failed.
    Io(String),
    /// The session manager rejected an operation.
    Session(String),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message(message) | Self::Io(message) | Self::Session(message) => {
                f.write_str(message)
            }
        }
    }
}

impl std::error::Error for ExportError {}

impl From<std::io::Error> for ExportError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

impl From<SessionManagerError> for ExportError {
    fn from(error: SessionManagerError) -> Self {
        Self::Session(error.to_string())
    }
}

/// The export options, upstream's `ExportOptions` (the string overload
/// restates as [`ExportOptions::for_path`]).
#[derive(Default)]
pub struct ExportOptions {
    /// The output file path; defaults to `pi-session-<session basename>.html`.
    pub output_path: Option<String>,
    /// The theme name; the theme source resolves it.
    pub theme_name: Option<String>,
    /// The custom-tool renderer, upstream's `toolRenderer`.
    pub tool_renderer: Option<Box<dyn ToolHtmlRenderer>>,
}

impl std::fmt::Debug for ExportOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The tool renderer is a hook box and does not debug.
        f.debug_struct("ExportOptions")
            .field("output_path", &self.output_path)
            .field("theme_name", &self.theme_name)
            .field("tool_renderer", &self.tool_renderer.is_some())
            .finish()
    }
}

impl ExportOptions {
    /// The upstream string-overload shape: only an output path.
    #[must_use]
    pub fn for_path(output_path: impl Into<String>) -> Self {
        Self {
            output_path: Some(output_path.into()),
            ..Self::default()
        }
    }
}

/// The explicit export colors a theme may carry, upstream's
/// `getThemeExportColors` return.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThemeExportColors {
    /// The page background.
    pub page_bg: Option<String>,
    /// The card background.
    pub card_bg: Option<String>,
    /// The info background.
    pub info_bg: Option<String>,
}

/// The theme color source the HTML generation consumes, upstream's
/// `getResolvedThemeColors`/`getThemeExportColors` reads from the theme
/// system.
///
/// The theme machinery rides its own slice (#132); until then the port
/// supplies a source, and [`NoThemeSource`] stands in with upstream's
/// colorless fallback path.
pub trait ThemeColorsSource {
    /// The resolved CSS color map keyed by theme color name, upstream's
    /// `getResolvedThemeColors`.
    fn resolved_colors(&self, theme_name: Option<&str>) -> BTreeMap<String, String>;

    /// The explicit export-section colors, upstream's `getThemeExportColors`.
    fn export_colors(&self, theme_name: Option<&str>) -> ThemeExportColors;
}

/// The colorless stand-in source: an empty color map with no explicit export
/// colors, which drives generation through upstream's derived-fallback path.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoThemeSource;

impl ThemeColorsSource for NoThemeSource {
    fn resolved_colors(&self, _theme_name: Option<&str>) -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    fn export_colors(&self, _theme_name: Option<&str>) -> ThemeExportColors {
        ThemeExportColors::default()
    }
}

/// Parse a color string to RGB values, upstream's `parseColor`: the `#RRGGBB`
/// and `rgb(r,g,b)` forms.
///
/// Upstream's `rgb` regex allows whitespace after `rgb` and around the
/// channels and accepts out-of-range digits (`\d+` with no bound); the port
/// trims the channels but rejects out-of-range values, degrading such a
/// color to the derived-fallback palette instead of carrying out-of-gamut
/// math into the luminance branch.
fn parse_color(color: &str) -> Option<(u8, u8, u8)> {
    if let Some(hex) = color.strip_prefix('#') {
        if hex.len() == 6 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            let channel = |range: std::ops::Range<usize>| u8::from_str_radix(&hex[range], 16).ok();
            return Some((channel(0..2)?, channel(2..4)?, channel(4..6)?));
        }
        return None;
    }
    let inner = color
        .strip_prefix("rgb")?
        .trim_start()
        .strip_prefix('(')?
        .strip_suffix(')')?;
    let channels: Vec<&str> = inner.split(',').map(str::trim).collect();
    if channels.len() != 3 {
        return None;
    }
    let channel = |value: &str| value.parse::<u8>().ok();
    Some((
        channel(channels[0])?,
        channel(channels[1])?,
        channel(channels[2])?,
    ))
}

/// Calculate relative luminance of a color (0-1, higher = lighter), upstream's
/// `getLuminance`.
#[expect(
    clippy::suboptimal_flops,
    reason = "the weighted sum restates upstream's `0.2126*r + 0.7152*g + 0.0722*b`; mul_add would drift from the JS float math"
)]
fn get_luminance(r: u8, g: u8, b: u8) -> f64 {
    let to_linear = |c: u8| {
        let s = f64::from(c) / 255.0;
        if s <= 0.03928 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * to_linear(r) + 0.7152 * to_linear(g) + 0.0722 * to_linear(b)
}

/// Adjust color brightness (factor > 1 lightens, < 1 darkens), upstream's
/// `adjustBrightness`.
fn adjust_brightness(color: &str, factor: f64) -> String {
    let Some((r, g, b)) = parse_color(color) else {
        return color.to_owned();
    };
    let clamp = |channel: u8| {
        let adjusted = f64::from(channel) * factor;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the channel is clamped to 0-255 before the cast, matching upstream's Math.round ladder"
        )]
        let clamped = adjusted.round().clamp(0.0, 255.0) as u64;
        clamped
    };
    format!("rgb({}, {}, {})", clamp(r), clamp(g), clamp(b))
}

/// The derived export palette, upstream's `deriveExportColors` return with
/// the nullable shape collapsed: every color is set by construction, so the
/// port carries the three strings the CSS block needs.
#[expect(
    clippy::struct_field_names,
    reason = "the fields carry upstream's wire names pageBg/cardBg/infoBg verbatim"
)]
struct DerivedExportColors {
    page_bg: String,
    card_bg: String,
    info_bg: String,
}

/// Derive export background colors from a base color (e.g. userMessageBg),
/// upstream's `deriveExportColors`.
fn derive_export_colors(base_color: &str) -> DerivedExportColors {
    let Some((r, g, b)) = parse_color(base_color) else {
        return DerivedExportColors {
            page_bg: "rgb(24, 24, 30)".to_owned(),
            card_bg: "rgb(30, 30, 36)".to_owned(),
            info_bg: "rgb(60, 55, 40)".to_owned(),
        };
    };

    let luminance = get_luminance(r, g, b);
    let (red, green, blue) = (u64::from(r), u64::from(g), i64::from(b));
    if luminance > 0.5 {
        return DerivedExportColors {
            page_bg: adjust_brightness(base_color, 0.96),
            card_bg: base_color.to_owned(),
            info_bg: format!(
                "rgb({}, {}, {})",
                (red + 10).min(255),
                (green + 5).min(255),
                (blue - 20).max(0)
            ),
        };
    }
    DerivedExportColors {
        page_bg: adjust_brightness(base_color, 0.7),
        card_bg: adjust_brightness(base_color, 0.85),
        info_bg: format!(
            "rgb({}, {}, {})",
            (red + 20).min(255),
            (green + 15).min(255),
            blue
        ),
    }
}

/// Push a `String.prototype.replace` replacement body, upstream's `$`-pattern
/// semantics for a string pattern (no capture groups: `$n`/`$<name>` stay
/// literal; `$$`, `$&`, `` $` ``, `$'` substitute).
fn push_substitution(
    out: &mut String,
    replacement: &str,
    prefix: &str,
    matched: &str,
    suffix: &str,
) {
    let chars: Vec<char> = replacement.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let character = chars[i];
        if character != '$' || i + 1 >= chars.len() {
            out.push(character);
            i += 1;
            continue;
        }
        match chars[i + 1] {
            '$' => {
                out.push('$');
                i += 2;
            }
            '&' => {
                out.push_str(matched);
                i += 2;
            }
            '`' => {
                out.push_str(prefix);
                i += 2;
            }
            '\'' => {
                out.push_str(suffix);
                i += 2;
            }
            _ => {
                out.push('$');
                i += 1;
            }
        }
    }
}

/// The first-occurrence replacement with `String.prototype.replace`
/// semantics, upstream's `template.replace(needle, body)` calls.
fn js_replace(haystack: &str, needle: &str, replacement: &str) -> String {
    let Some(start) = haystack.find(needle) else {
        return haystack.to_owned();
    };
    let end = start + needle.len();
    let mut out = String::with_capacity(haystack.len() + replacement.len());
    let prefix = &haystack[..start];
    let suffix = &haystack[end..];
    out.push_str(prefix);
    push_substitution(&mut out, replacement, prefix, needle, suffix);
    out.push_str(suffix);
    out
}

/// Generate CSS custom property declarations from theme colors, upstream's
/// `generateThemeVars`.
fn generate_theme_vars(theme_name: Option<&str>, source: &dyn ThemeColorsSource) -> String {
    let colors = source.resolved_colors(theme_name);
    let mut lines: Vec<String> = colors
        .iter()
        .map(|(key, value)| format!("--{key}: {value};"))
        .collect();

    // Use explicit theme export colors if available, otherwise derive from
    // userMessageBg.
    let theme_export = source.export_colors(theme_name);
    let user_message_bg = colors
        .get("userMessageBg")
        .map_or("#343541", String::as_str);
    let DerivedExportColors {
        page_bg,
        card_bg,
        info_bg,
    } = derive_export_colors(user_message_bg);

    lines.push(format!(
        "--exportPageBg: {};",
        theme_export.page_bg.unwrap_or(page_bg)
    ));
    lines.push(format!(
        "--exportCardBg: {};",
        theme_export.card_bg.unwrap_or(card_bg)
    ));
    lines.push(format!(
        "--exportInfoBg: {};",
        theme_export.info_bg.unwrap_or(info_bg)
    ));

    lines.join("\n      ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn js_replace_reproduces_the_five_substitution_families() {
        // `$$` collapses to a literal `$`, `$&` rewrites to the match, and the
        // context escapes substitute the surrounding text.
        assert_eq!(js_replace("aXb", "X", "$$"), "a$b");
        assert_eq!(js_replace("aXb", "X", "[$&]"), "a[X]b");
        assert_eq!(
            js_replace("aXb", "X", "[$`]"),
            "a[a]b",
            "the backtick escape substitutes the prefix"
        );
        assert_eq!(
            js_replace("aXb", "X", "[$']"),
            "a[b]b",
            "the quote escape substitutes the suffix"
        );
        assert_eq!(
            js_replace("aXb", "X", "$1$<name>$n"),
            "a$1$<name>$nb",
            "numbered and named groups stay literal for a string pattern"
        );
        assert_eq!(
            js_replace("aXb", "X", "$"),
            "a$b",
            "a trailing dollar rides as text"
        );
        assert_eq!(
            js_replace("abc", "X", "[hit]"),
            "abc",
            "an absent match leaves the haystack alone"
        );
        assert_eq!(
            js_replace("aXYb", "XY", "z"),
            "azb",
            "only the first occurrence replaces"
        );
    }

    #[test]
    fn parse_color_reads_hex_and_rgb_and_rejects_the_rest() {
        assert_eq!(parse_color("#343541"), Some((0x34, 0x35, 0x41)));
        assert_eq!(parse_color("rgb(12, 34, 56)"), Some((12, 34, 56)));
        assert_eq!(parse_color("rgb(12,34,56)"), Some((12, 34, 56)));
        assert_eq!(
            parse_color("rgb (12, 34, 56)"),
            Some((12, 34, 56)),
            "upstream's regex allows the space after rgb"
        );
        assert_eq!(
            parse_color("#343541extra"),
            None,
            "upstream's hex regex is a full match"
        );
        assert_eq!(
            parse_color("rgb(300, 0, 0)"),
            None,
            "the out-of-range degradation"
        );
        assert_eq!(parse_color("#34354"), None, "a short hex rejects");
        assert_eq!(parse_color("#34354G"), None, "a non-hex digit rejects");
        assert_eq!(parse_color("rgb(1,2)"), None, "two channels reject");
        assert_eq!(
            parse_color("rgb(1,2,300)"),
            None,
            "an out-of-range channel rejects"
        );
        assert_eq!(
            parse_color("rgb(1,2,x)"),
            None,
            "a non-numeric channel rejects"
        );
        assert_eq!(parse_color("hsl(1,2,3)"), None, "another function rejects");
        assert_eq!(parse_color(""), None);
    }

    #[test]
    fn adjust_brightness_clamps_and_passes_unparseable_colors_through() {
        assert_eq!(
            adjust_brightness("#ffffff", 2.0),
            "rgb(255, 255, 255)",
            "the ceiling clamps"
        );
        assert_eq!(
            adjust_brightness("#000000", 0.0),
            "rgb(0, 0, 0)",
            "the floor clamps"
        );
        assert_eq!(
            adjust_brightness("not-a-color", 2.0),
            "not-a-color",
            "an unparseable color rides through"
        );
    }

    #[test]
    fn the_export_error_conversions_reprint_their_sources() {
        let io: ExportError = std::io::Error::new(std::io::ErrorKind::Other, "boom").into();
        assert_eq!(io.to_string(), "boom");
        let session: ExportError =
            crate::session_manager::SessionManagerError::Io("gone".to_owned()).into();
        assert_eq!(
            session.to_string(),
            "gone",
            "the session arm reprints the manager error"
        );
    }
}

//! Shared presentation helpers for the built-in tool renderers, upstream
//! `src/core/tools/render-utils.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The helpers read the same wire shapes the tool results carry, degrading a
//! non-string or absent argument the way upstream's `str` does.

use std::rc::Rc;

use pi_ai::types::ToolResultBlock;
use pi_tui::terminal_image::{get_capabilities, get_image_dimensions, hyperlink, image_fallback};
use serde_json::Value;

use crate::config::home_dir;
use crate::extensions::types::{AgentToolResult, ToolRenderContext};
use crate::modes::interactive::theme::theme::Theme;
use crate::utils::ansi::strip_ansi;
use crate::utils::paths::resolve_path;
use crate::utils::shell::sanitize_binary_output;

/// Read an optional argument field, upstream's `str`: strings pass through,
/// absent and `null` read empty, and every other JSON type is invalid.
#[must_use]
pub fn string_arg(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(text)) => Some(text.clone()),
        None | Some(Value::Null) => Some(String::new()),
        Some(_) => None,
    }
}

/// Read one field of an arguments object, upstream's `args?.field`.
#[must_use]
pub fn field<'a>(args: Option<&'a Value>, name: &str) -> Option<&'a Value> {
    args.and_then(|args| args.get(name))
}

/// Read the first present one of several argument names, upstream's
/// `args?.file_path ?? args?.path` ladder.
#[must_use]
pub fn first_string_arg(args: Option<&Value>, names: &[&str]) -> Option<String> {
    for name in names {
        if let Some(value) = field(args, name) {
            return string_arg(Some(value));
        }
    }
    string_arg(None)
}

/// Format a JSON number the way JavaScript string interpolation prints it:
/// integral values without a fractional part.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the guard bounds the value below 2^53, where every integral f64 fits an i64"
)]
pub fn number_text(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 9_007_199_254_740_992.0 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// Reuse the row's previous text component when the last render returned
/// one, upstream's `context.lastComponent as Text`.
#[must_use]
pub fn reuse_text(context: &ToolRenderContext) -> Rc<pi_tui::components::Text> {
    if let Some(last) = &context.last_component {
        let any: Rc<dyn std::any::Any> = last.clone();
        if let Ok(text) = any.downcast::<pi_tui::components::Text>() {
            return text;
        }
    }
    Rc::new(pi_tui::components::Text::with_padding("", 0, 0))
}

/// Shorten an absolute home path to `~`, upstream's `shortenPath`.
#[must_use]
pub fn shorten_path(path: &str) -> String {
    let home = home_dir();
    path.strip_prefix(&home).map_or_else(
        || path.to_string(),
        |rest| {
            let mut shortened = String::from("~");
            shortened.push_str(rest);
            shortened
        },
    )
}

/// Wrap a styled path in a `file:` hyperlink when the terminal supports it,
/// upstream's `linkPath`.
///
/// The URL is the resolved path verbatim; upstream's `pathToFileURL` percent
/// encoding is not restated.
#[must_use]
pub fn link_path(styled_text: &str, raw_path: &str, cwd: &str) -> String {
    if !get_capabilities().hyperlinks {
        return styled_text.to_string();
    }
    let absolute_path = resolve_path(raw_path, cwd, &home_dir());
    hyperlink(styled_text, &format!("file://{absolute_path}"))
}

/// Tabs become three spaces, upstream's `replaceTabs`.
#[must_use]
pub fn replace_tabs(text: &str) -> String {
    text.replace('\t', "   ")
}

/// Drop carriage returns, upstream's `normalizeDisplayText`.
#[must_use]
pub fn normalize_display_text(text: &str) -> String {
    text.replace('\r', "")
}

/// The result's text blocks joined, with image indicators when inline images
/// are unavailable or hidden, upstream's `getTextOutput`.
#[must_use]
pub fn get_text_output(result: Option<&AgentToolResult>, show_images: bool) -> String {
    let Some(result) = result else {
        return String::new();
    };

    let mut output = result
        .content
        .iter()
        .filter_map(|block| match block {
            ToolResultBlock::Text(text) => Some(text.text.as_str()),
            ToolResultBlock::Image(_) => None,
        })
        .map(|text| sanitize_binary_output(&strip_ansi(text)).replace('\r', ""))
        .collect::<Vec<_>>()
        .join("\n");

    let image_blocks: Vec<&pi_ai::types::ImageContent> = result
        .content
        .iter()
        .filter_map(|block| match block {
            ToolResultBlock::Image(image) => Some(image),
            ToolResultBlock::Text(_) => None,
        })
        .collect();

    let caps = get_capabilities();
    if !image_blocks.is_empty() && (caps.images.is_none() || !show_images) {
        let indicators = image_blocks
            .iter()
            .map(|image| {
                let mime_type = if image.mime_type.is_empty() {
                    "image/unknown"
                } else {
                    image.mime_type.as_str()
                };
                let dimensions = get_image_dimensions(&image.data, &image.mime_type);
                image_fallback(mime_type, dimensions, None)
            })
            .collect::<Vec<_>>()
            .join("\n");
        output = if output.is_empty() {
            indicators
        } else {
            format!("{output}\n{indicators}")
        };
    }

    output
}

/// Read a numeric member of the result's details, upstream's
/// `result.details?.name`.
#[must_use]
pub fn details_usize(result: &AgentToolResult, name: &str) -> Option<usize> {
    result
        .details
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

/// Read a boolean member of the result's details.
#[must_use]
pub fn details_bool(result: &AgentToolResult, name: &str) -> Option<bool> {
    result.details.get(name).and_then(Value::as_bool)
}

/// Read the result's truncation metadata when one is present.
#[must_use]
pub fn details_truncation(
    result: &AgentToolResult,
) -> Option<crate::tools::truncate::TruncationResult> {
    result
        .details
        .get("truncation")
        .and_then(|value| serde_json::from_value(value.clone()).ok())
}

/// The invalid-argument marker, upstream's `invalidArgText`.
#[must_use]
pub fn invalid_arg_text(theme: &Theme) -> String {
    theme.fg("error", "[invalid arg]")
}

/// Render a tool path: invalid arguments in the error color, an absent value
/// as `...`, and everything else as a shortened, optionally hyperlinked
/// accent-colored path.
#[must_use]
pub fn render_tool_path(
    raw_path: Option<String>,
    theme: &Theme,
    cwd: &str,
    empty_fallback: Option<&str>,
) -> String {
    let Some(raw_path) = raw_path else {
        return invalid_arg_text(theme);
    };
    let value = if raw_path.is_empty() {
        empty_fallback.unwrap_or("").to_string()
    } else {
        raw_path
    };
    if value.is_empty() {
        return theme.fg("toolOutput", "...");
    }
    link_path(&theme.fg("accent", &shorten_path(&value)), &value, cwd)
}

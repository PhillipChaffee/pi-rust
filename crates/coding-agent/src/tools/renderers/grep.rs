//! Presentation for the grep tool, upstream
//! `src/core/tools/renderers/grep.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::fmt::Write as _;
use std::sync::Arc;

use super::ToolRenderers;
use crate::extensions::types::{
    AgentToolResult, ToolRenderCallFn, ToolRenderResultFn, ToolRenderResultOptions,
};
use crate::modes::interactive::components::keybinding_hints::key_hint;
use crate::modes::interactive::theme::theme::{Theme, bold};
use crate::tools::render_utils::{
    details_bool, details_truncation, details_usize, first_string_arg, get_text_output,
    invalid_arg_text, number_text, reuse_text, shorten_path,
};
use crate::tools::truncate::format_size;

/// The grep call row, upstream's `formatGrepCall`.
fn format_grep_call(args: Option<&serde_json::Value>, theme: &Theme) -> String {
    let pattern = first_string_arg(args, &["pattern"]);
    let raw_path = first_string_arg(args, &["path"]);
    let path = raw_path.map(|path| shorten_path(if path.is_empty() { "." } else { &path }));
    let glob = first_string_arg(args, &["glob"]);
    let limit = args
        .and_then(|args| args.get("limit"))
        .and_then(serde_json::Value::as_f64);
    let invalid_arg = invalid_arg_text(theme);
    let pattern_display = pattern.map_or_else(
        || invalid_arg.clone(),
        |pattern| theme.fg("accent", &format!("/{pattern}/")),
    );
    let path_display = path.map_or(invalid_arg, |path| {
        theme.fg("toolOutput", &format!(" in {path}"))
    });
    let mut text = format!(
        "{} {}{}",
        theme.fg("toolTitle", &bold("grep")),
        pattern_display,
        path_display
    );
    if let Some(glob) = glob.filter(|glob| !glob.is_empty()) {
        text.push_str(&theme.fg("toolOutput", &format!(" ({glob})")));
    }
    if let Some(limit) = limit {
        text.push_str(&theme.fg("toolOutput", &format!(" limit {}", number_text(limit))));
    }
    text
}

/// The grep result body, upstream's `formatGrepResult`.
fn format_grep_result(
    result: &AgentToolResult,
    options: ToolRenderResultOptions,
    theme: &Theme,
    show_images: bool,
) -> String {
    let output = get_text_output(Some(result), show_images)
        .trim()
        .to_string();
    let mut text = String::new();
    if !output.is_empty() {
        let lines: Vec<&str> = output.split('\n').collect();
        let max_lines = if options.expanded { lines.len() } else { 15 };
        let display_lines = lines.iter().take(max_lines).copied().collect::<Vec<_>>();
        let remaining = lines.len().saturating_sub(max_lines);
        let _ = write!(
            text,
            "\n{}",
            display_lines
                .iter()
                .map(|line| theme.fg("toolOutput", line))
                .collect::<Vec<_>>()
                .join("\n")
        );
        if remaining > 0 {
            let more_lines = format!("\n... ({remaining} more lines,");
            let expand_hint = format!(" {}", key_hint("app.tools.expand", "to expand"));
            let _ = write!(
                text,
                "{}{}{}",
                theme.fg("muted", &more_lines),
                expand_hint,
                theme.fg("muted", ")")
            );
        }
    }

    let match_limit = details_usize(result, "matchLimitReached");
    let truncation = details_truncation(result);
    let lines_truncated = details_bool(result, "linesTruncated").unwrap_or(false);
    let truncated = truncation.as_ref().is_some_and(|value| value.truncated);
    if match_limit.is_some() || truncated || lines_truncated {
        let mut warnings = Vec::new();
        if let Some(match_limit) = match_limit {
            warnings.push(format!("{match_limit} matches limit"));
        }
        if let Some(truncation) = truncation.filter(|value| value.truncated) {
            warnings.push(format!("{} limit", format_size(truncation.max_bytes)));
        }
        if lines_truncated {
            warnings.push("some lines truncated".to_string());
        }
        let _ = write!(
            text,
            "\n{}",
            theme.fg("warning", &format!("[Truncated: {}]", warnings.join(", ")))
        );
    }
    text
}

/// The grep tool's renderer pair.
#[must_use]
pub fn grep_renderers() -> ToolRenderers {
    let render_call: Arc<ToolRenderCallFn> = Arc::new(|args, theme, context| {
        let text = reuse_text(context);
        text.set_text(&format_grep_call(Some(args), theme));
        text
    });
    let render_result: Arc<ToolRenderResultFn> = Arc::new(|result, options, theme, context| {
        let text = reuse_text(context);
        text.set_text(&format_grep_result(
            result,
            *options,
            theme,
            context.show_images,
        ));
        text
    });
    ToolRenderers {
        render_call: Some(render_call),
        render_result: Some(render_result),
    }
}

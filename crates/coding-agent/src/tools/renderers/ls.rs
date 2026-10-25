//! Presentation for the ls tool, upstream
//! `src/core/tools/renderers/ls.ts` at pin
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
    details_truncation, details_usize, get_text_output, number_text, render_tool_path, reuse_text,
    string_arg,
};
use crate::tools::truncate::format_size;

/// The ls call row, upstream's `formatLsCall`.
fn format_ls_call(args: Option<&serde_json::Value>, theme: &Theme, cwd: &str) -> String {
    let path = args.and_then(|args| string_arg(args.get("path")));
    let limit = args
        .and_then(|args| args.get("limit"))
        .and_then(serde_json::Value::as_f64);
    let mut text = format!(
        "{} {}",
        theme.fg("toolTitle", &bold("ls")),
        render_tool_path(path, theme, cwd, Some("."))
    );
    if let Some(limit) = limit {
        text.push_str(&theme.fg("toolOutput", &format!(" (limit {})", number_text(limit))));
    }
    text
}

/// The ls result body, upstream's `formatLsResult`.
fn format_ls_result(
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
        let max_lines = if options.expanded { lines.len() } else { 20 };
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

    let entry_limit = details_usize(result, "entryLimitReached");
    let truncation = details_truncation(result);
    if entry_limit.is_some() || truncation.as_ref().is_some_and(|value| value.truncated) {
        let mut warnings = Vec::new();
        if let Some(entry_limit) = entry_limit {
            warnings.push(format!("{entry_limit} entries limit"));
        }
        if let Some(truncation) = truncation.filter(|value| value.truncated) {
            warnings.push(format!("{} limit", format_size(truncation.max_bytes)));
        }
        let _ = write!(
            text,
            "\n{}",
            theme.fg("warning", &format!("[Truncated: {}]", warnings.join(", ")))
        );
    }
    text
}

/// The ls tool's renderer pair.
#[must_use]
pub fn ls_renderers() -> ToolRenderers {
    let render_call: Arc<ToolRenderCallFn> = Arc::new(|args, theme, context| {
        let text = reuse_text(context);
        text.set_text(&format_ls_call(Some(args), theme, context.cwd));
        text
    });
    let render_result: Arc<ToolRenderResultFn> = Arc::new(|result, options, theme, context| {
        let text = reuse_text(context);
        text.set_text(&format_ls_result(
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

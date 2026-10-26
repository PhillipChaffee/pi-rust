//! Presentation for the read tool, upstream
//! `src/core/tools/renderers/read.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Compact rows for skills, project resources, and Pi's own docs expand only
//! when the row is expanded; ordinary reads always render the full row.

use std::fmt::Write as _;
use std::sync::Arc;

use super::ToolRenderers;
use crate::config::get_readme_path;
use crate::extensions::types::{
    AgentToolResult, ToolRenderCallFn, ToolRenderResultFn, ToolRenderResultOptions,
};
use crate::modes::interactive::components::keybinding_hints::{key_hint, key_text};
use crate::modes::interactive::theme::theme::{
    Theme, bold, get_language_from_path, highlight_code,
};
use crate::tools::path_utils::resolve_to_cwd;
use crate::tools::render_utils::{
    first_string_arg, get_text_output, number_text, render_tool_path, replace_tabs, reuse_text,
};
use crate::tools::truncate::{TruncatedBy, TruncationResult, format_size};
use crate::utils::paths::{
    basename_posix, dirname_posix, format_path_relative_to_cwd_or_absolute, is_under_path,
    relative_posix,
};

/// The compact classification upstream's renderer recognizes.
struct CompactReadClassification {
    kind: &'static str,
    label: String,
}

/// The resource file names rendered compactly.
const COMPACT_RESOURCE_FILE_NAMES: [&str; 6] = [
    "AGENTS.override.md",
    "AGENTS.md",
    "AGENTS.MD",
    "CLAUDE.md",
    "CLAUDE.MD",
    "CLAUDE.Md",
];

/// The read call's path argument, upstream's `args?.file_path ?? args?.path`.
fn read_path(args: Option<&serde_json::Value>) -> Option<String> {
    first_string_arg(args, &["file_path", "path"])
}

/// `:start-end` for a ranged read, upstream's `formatReadLineRange`.
#[expect(
    clippy::option_if_let_else,
    reason = "the match keeps the absent-argument arm explicit; the closure form adds function instances the coverage budget counts"
)]
fn numeric_arg(args: Option<&serde_json::Value>, name: &str) -> Option<f64> {
    match args {
        Some(args) => args.get(name).and_then(serde_json::Value::as_f64),
        None => None,
    }
}

fn format_read_line_range(args: Option<&serde_json::Value>, theme: &Theme) -> String {
    let offset = numeric_arg(args, "offset");
    let limit = numeric_arg(args, "limit");
    if offset.is_none() && limit.is_none() {
        return String::new();
    }
    let start_line = offset.unwrap_or(1.0);
    let end_line = limit.map(|limit| start_line + limit - 1.0);
    let range = end_line.map_or_else(
        || number_text(start_line),
        |end| format!("{}-{}", number_text(start_line), number_text(end)),
    );
    theme.fg("warning", &format!(":{range}"))
}

/// The plain read row, upstream's `formatReadCall`.
fn format_read_call(args: Option<&serde_json::Value>, theme: &Theme, cwd: &str) -> String {
    format!(
        "{} {}{}",
        theme.fg("toolTitle", &bold("read")),
        render_tool_path(read_path(args), theme, cwd, None),
        format_read_line_range(args, theme)
    )
}

/// Drop trailing empty lines, upstream's `trimTrailingEmptyLines`.
fn trim_trailing_empty_lines(lines: &mut Vec<String>) {
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
}

/// Classify a read of Pi's own documentation tree, upstream's
/// `getPiDocsClassification`.
fn pi_docs_classification(absolute_path: &str) -> Option<CompactReadClassification> {
    let package_root = dirname_posix(&get_readme_path());
    if absolute_path == package_root || !is_under_path(absolute_path, &package_root) {
        return None;
    }
    let label = relative_posix(&package_root, absolute_path);
    if label == "README.md" || label.starts_with("docs/") || label.starts_with("examples/") {
        return Some(CompactReadClassification {
            kind: "docs",
            label,
        });
    }
    None
}

/// Classify skill/resource/docs reads, upstream's
/// `getCompactReadClassification`.
fn compact_read_classification(
    args: Option<&serde_json::Value>,
    cwd: &str,
) -> Option<CompactReadClassification> {
    let raw_path = read_path(args)?;
    if raw_path.is_empty() {
        return None;
    }
    let absolute_path = resolve_to_cwd(&raw_path, cwd).ok()?;
    let file_name = basename_posix(&absolute_path);
    if file_name == "SKILL.md" {
        let parent = dirname_posix(&absolute_path);
        let label = basename_posix(&parent);
        return Some(CompactReadClassification {
            kind: "skill",
            label: if label.is_empty() { file_name } else { label },
        });
    }
    if let Some(docs) = pi_docs_classification(&absolute_path) {
        return Some(docs);
    }
    if COMPACT_RESOURCE_FILE_NAMES.contains(&file_name.as_str()) {
        return Some(CompactReadClassification {
            kind: "resource",
            label: format_path_relative_to_cwd_or_absolute(&absolute_path, cwd),
        });
    }
    None
}

/// The compact row for a skill/resource/docs read, upstream's
/// `formatCompactReadCall`.
fn format_compact_read_call(
    classification: &CompactReadClassification,
    args: Option<&serde_json::Value>,
    theme: &Theme,
) -> String {
    let expand_hint = theme.fg(
        "dim",
        &format!(" ({} to expand)", key_text("app.tools.expand")),
    );
    if classification.kind == "skill" {
        return format!(
            "{}{}{}{}",
            theme.fg("customMessageLabel", "\u{1b}[1m[skill]\u{1b}[22m "),
            theme.fg("customMessageText", &classification.label),
            format_read_line_range(args, theme),
            expand_hint
        );
    }
    format!(
        "{} {}{}{}",
        theme.fg("toolTitle", &bold(&format!("read {}", classification.kind))),
        theme.fg("accent", &classification.label),
        format_read_line_range(args, theme),
        expand_hint
    )
}

/// The read result body, upstream's `formatReadResult`.
fn format_read_result(
    args: Option<&serde_json::Value>,
    result: &AgentToolResult,
    options: ToolRenderResultOptions,
    theme: &Theme,
    show_images: bool,
    is_error: bool,
) -> String {
    if !options.expanded && !is_error {
        return String::new();
    }

    let raw_path = read_path(args);
    let output = get_text_output(Some(result), show_images);
    let lang = if is_error {
        None
    } else {
        raw_path.as_deref().and_then(get_language_from_path)
    };
    let mut lines: Vec<String> = lang.map_or_else(
        || output.split('\n').map(str::to_string).collect(),
        |lang| highlight_code(&replace_tabs(&output), Some(lang)),
    );
    trim_trailing_empty_lines(&mut lines);
    let max_lines = if options.expanded { lines.len() } else { 10 };
    let display_lines: Vec<String> = lines.iter().take(max_lines).cloned().collect();
    let remaining = lines.len().saturating_sub(max_lines);
    let mut text = format!(
        "\n{}",
        display_lines
            .iter()
            .map(|line| {
                if lang.is_some() {
                    replace_tabs(line)
                } else {
                    theme.fg("toolOutput", &replace_tabs(line))
                }
            })
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
    if let Some(truncation) = result
        .details
        .get("truncation")
        .and_then(|value| serde_json::from_value::<TruncationResult>(value.clone()).ok())
    {
        text.push_str(&format_truncation_warning(&truncation, theme));
    }
    text
}

/// The truncation footer, upstream's three-branch warning ladder.
fn format_truncation_warning(truncation: &TruncationResult, theme: &Theme) -> String {
    if !truncation.truncated {
        return String::new();
    }
    if truncation.first_line_exceeds_limit {
        return format!(
            "\n{}",
            theme.fg(
                "warning",
                &format!(
                    "[First line exceeds {} limit]",
                    format_size(truncation.max_bytes)
                )
            )
        );
    }
    if truncation.truncated_by == Some(TruncatedBy::Lines) {
        return format!(
            "\n{}",
            theme.fg(
                "warning",
                &format!(
                    "[Truncated: showing {} of {} lines ({} line limit)]",
                    truncation.output_lines, truncation.total_lines, truncation.max_lines
                )
            )
        );
    }
    format!(
        "\n{}",
        theme.fg(
            "warning",
            &format!(
                "[Truncated: {} lines shown ({} limit)]",
                truncation.output_lines,
                format_size(truncation.max_bytes)
            )
        )
    )
}

/// The read tool's renderer pair.
#[must_use]
pub fn read_renderers() -> ToolRenderers {
    let render_call: Arc<ToolRenderCallFn> = Arc::new(|args, theme, context| {
        let text = reuse_text(context);
        let classification = if context.expanded {
            None
        } else {
            compact_read_classification(Some(args), context.cwd)
        };
        text.set_text(&classification.map_or_else(
            || format_read_call(Some(args), theme, context.cwd),
            |classification| format_compact_read_call(&classification, Some(args), theme),
        ));
        text
    });
    let render_result: Arc<ToolRenderResultFn> = Arc::new(|result, options, theme, context| {
        let text = reuse_text(context);
        text.set_text(&format_read_result(
            Some(context.args),
            result,
            *options,
            theme,
            context.show_images,
            context.is_error,
        ));
        text
    });
    ToolRenderers {
        render_call: Some(render_call),
        render_result: Some(render_result),
    }
}

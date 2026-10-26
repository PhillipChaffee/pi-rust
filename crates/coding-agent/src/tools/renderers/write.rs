//! Presentation for the write tool, upstream
//! `src/core/tools/renderers/write.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The call row highlights the file content as it streams, re-highlighting
//! only the appended tail unless the write is complete or diverges from the
//! cached prefix.

use std::fmt::Write as _;
use std::sync::Arc;

use super::ToolRenderers;
use crate::extensions::types::{AgentToolResult, ToolRenderCallFn, ToolRenderResultFn};
use crate::modes::interactive::components::keybinding_hints::key_hint;
use crate::modes::interactive::theme::theme::{
    Theme, bold, get_language_from_path, highlight_code,
};
use crate::tools::render_utils::{
    first_string_arg, normalize_display_text, render_tool_path, replace_tabs, reuse_text,
    string_arg,
};

/// The full-highlight prefix ceiling while a write streams.
const WRITE_PARTIAL_FULL_HIGHLIGHT_LINES: usize = 50;

/// The incremental highlight cache, upstream's `WriteHighlightCache`.
struct WriteHighlightCache {
    raw_path: Option<String>,
    lang: &'static str,
    raw_content: String,
    normalized_lines: Vec<String>,
    highlighted_lines: Vec<String>,
}

/// The per-row renderer state.
#[derive(Default)]
struct WriteRenderState {
    cache: Option<WriteHighlightCache>,
}

/// Drop trailing empty lines, upstream's `trimTrailingEmptyLines`.
fn trim_trailing_empty_lines(lines: &mut Vec<String>) {
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
}

/// Highlight one line in isolation, upstream's `highlightSingleLine`.
fn highlight_single_line(line: &str, lang: &str) -> String {
    highlight_code(line, Some(lang))
        .into_iter()
        .next()
        .unwrap_or_default()
}

/// Re-highlight the first lines after an append, upstream's
/// `refreshWriteHighlightPrefix`.
fn refresh_write_highlight_prefix(cache: &mut WriteHighlightCache) {
    let prefix_count = WRITE_PARTIAL_FULL_HIGHLIGHT_LINES.min(cache.normalized_lines.len());
    if prefix_count == 0 {
        return;
    }
    let prefix_source = cache.normalized_lines[..prefix_count].join("\n");
    let prefix_highlighted = highlight_code(&prefix_source, Some(cache.lang));
    for index in 0..prefix_count {
        cache.highlighted_lines[index] = match prefix_highlighted.get(index) {
            Some(line) => line.clone(),
            None => highlight_single_line(&cache.normalized_lines[index], cache.lang),
        };
    }
}

/// Build the cache from the whole file, upstream's
/// `rebuildWriteHighlightCacheFull`.
fn rebuild_write_highlight_cache_full(
    raw_path: Option<&str>,
    file_content: &str,
) -> Option<WriteHighlightCache> {
    let lang = raw_path.and_then(get_language_from_path)?;
    let display_content = normalize_display_text(file_content);
    let normalized = replace_tabs(&display_content);
    Some(WriteHighlightCache {
        raw_path: raw_path.map(str::to_string),
        lang,
        raw_content: file_content.to_string(),
        normalized_lines: normalized.split('\n').map(str::to_string).collect(),
        highlighted_lines: highlight_code(&normalized, Some(lang)),
    })
}

/// Extend the cache with the appended content, upstream's
/// `updateWriteHighlightCacheIncremental`.
fn update_write_highlight_cache_incremental(
    cache: Option<WriteHighlightCache>,
    raw_path: Option<&str>,
    file_content: &str,
) -> Option<WriteHighlightCache> {
    let lang = raw_path.and_then(get_language_from_path)?;
    let Some(mut cache) = cache else {
        return rebuild_write_highlight_cache_full(raw_path, file_content);
    };
    if cache.lang != lang
        || cache.raw_path.as_deref() != raw_path
        || !file_content.starts_with(&cache.raw_content)
    {
        return rebuild_write_highlight_cache_full(raw_path, file_content);
    }
    if file_content.len() == cache.raw_content.len() {
        return Some(cache);
    }

    let delta_raw = &file_content[cache.raw_content.len()..];
    let delta_normalized = replace_tabs(&normalize_display_text(delta_raw));
    cache.raw_content = file_content.to_string();
    if cache.normalized_lines.is_empty() {
        cache.normalized_lines.push(String::new());
        cache.highlighted_lines.push(String::new());
    }
    let segments: Vec<&str> = delta_normalized.split('\n').collect();
    let last_index = cache.normalized_lines.len() - 1;
    cache.normalized_lines[last_index].push_str(segments[0]);
    cache.highlighted_lines[last_index] =
        highlight_single_line(&cache.normalized_lines[last_index], lang);
    for segment in &segments[1..] {
        cache.normalized_lines.push((*segment).to_string());
        cache
            .highlighted_lines
            .push(highlight_single_line(segment, lang));
    }
    refresh_write_highlight_prefix(&mut cache);
    Some(cache)
}

/// The write result body, upstream's `formatWriteResult`.
fn format_write_result(result: &AgentToolResult, is_error: bool, theme: &Theme) -> Option<String> {
    if !is_error {
        return None;
    }
    let output = result
        .content
        .iter()
        .filter_map(|block| match block {
            pi_ai::types::ToolResultBlock::Text(text) => Some(text.text.as_str()),
            pi_ai::types::ToolResultBlock::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if output.is_empty() {
        return None;
    }
    Some(format!("\n{}", theme.fg("error", &output)))
}

/// The write call row, upstream's `formatWriteCall`.
fn format_write_call(
    args: Option<&serde_json::Value>,
    expanded: bool,
    theme: &Theme,
    cache: Option<&WriteHighlightCache>,
    cwd: &str,
) -> String {
    let raw_path = first_string_arg(args, &["file_path", "path"]);
    let file_content = string_arg(args.and_then(|args| args.get("content")));
    let mut text = format!(
        "{} {}",
        theme.fg("toolTitle", &bold("write")),
        render_tool_path(raw_path.clone(), theme, cwd, None)
    );

    match file_content {
        None => {
            let _ = write!(
                text,
                "\n\n{}",
                theme.fg("error", "[invalid content arg - expected string]")
            );
        }
        Some(file_content) if !file_content.is_empty() => {
            let lang = raw_path.as_deref().and_then(get_language_from_path);
            let mut lines = match (lang, cache) {
                (Some(_), Some(cache)) => cache.highlighted_lines.clone(),
                (Some(lang), None) => highlight_code(
                    &replace_tabs(&normalize_display_text(&file_content)),
                    Some(lang),
                ),
                (None, _) => normalize_display_text(&file_content)
                    .split('\n')
                    .map(str::to_string)
                    .collect(),
            };
            trim_trailing_empty_lines(&mut lines);
            let total_lines = lines.len();
            let max_lines = if expanded { lines.len() } else { 10 };
            let display_lines = &lines[..max_lines.min(lines.len())];
            let remaining = lines.len().saturating_sub(max_lines);
            let _ = write!(
                text,
                "\n\n{}",
                display_lines
                    .iter()
                    .map(|line| {
                        if lang.is_some() {
                            line.clone()
                        } else {
                            theme.fg("toolOutput", &replace_tabs(line))
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            if remaining > 0 {
                let more_lines = format!("\n... ({remaining} more lines, {total_lines} total,");
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
        Some(_) => {}
    }
    text
}

/// The write tool's renderer pair.
#[must_use]
pub fn write_renderers() -> ToolRenderers {
    let render_call: Arc<ToolRenderCallFn> = Arc::new(|args, theme, context| {
        let raw_path = first_string_arg(Some(args), &["file_path", "path"]);
        let file_content = string_arg(Some(args).and_then(|args| args.get("content")));
        let text = reuse_text(context);
        let rendered = context
            .state
            .with_state_mut(WriteRenderState::default, |state| {
                if let Some(file_content) = &file_content {
                    state.cache = if context.args_complete {
                        rebuild_write_highlight_cache_full(raw_path.as_deref(), file_content)
                    } else {
                        update_write_highlight_cache_incremental(
                            state.cache.take(),
                            raw_path.as_deref(),
                            file_content,
                        )
                    };
                } else {
                    state.cache = None;
                }
                format_write_call(
                    Some(args),
                    context.expanded,
                    theme,
                    state.cache.as_ref(),
                    context.cwd,
                )
            })
            .unwrap_or_default();
        text.set_text(&rendered);
        text
    });
    let render_result: Arc<ToolRenderResultFn> = Arc::new(|result, _options, theme, context| {
        let text = reuse_text(context);
        text.set_text(&format_write_result(result, context.is_error, theme).unwrap_or_default());
        text
    });
    ToolRenderers {
        render_call: Some(render_call),
        render_result: Some(render_result),
    }
}

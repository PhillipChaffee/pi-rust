//! Presentation for the bash and powershell tools, upstream
//! `src/core/tools/renderers/bash.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatement: upstream schedules a one-second invalidate while a shell
//! result streams so its elapsed label ticks. The port has no timer on the
//! component side; the app frame's render loop invalidates the row instead,
//! and the label recomputes from the same start/end timestamps.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use pi_tui::components::Text;
use pi_tui::tui::{Component, Container};
use pi_tui::utils::truncate_to_width;

use super::ToolRenderers;
use crate::extensions::types::{
    AgentToolResult, ToolRenderCallFn, ToolRenderResultFn, ToolRenderResultOptions,
};
use crate::modes::interactive::components::keybinding_hints::key_hint;
use crate::modes::interactive::components::visual_truncate::truncate_to_visual_lines;
use crate::modes::interactive::theme::theme::{bold, theme};
use crate::tools::render_utils::{
    details_truncation, get_text_output, invalid_arg_text, reuse_text, string_arg,
};
use crate::tools::truncate::format_size;

/// The collapsed preview line count.
const BASH_PREVIEW_LINES: usize = 5;

/// The app frame's streaming-update throttle, upstream's
/// `BASH_UPDATE_THROTTLE_MS`.
pub const BASH_UPDATE_THROTTLE_MS: u64 = 100;

/// The shell call's argument shape.
fn format_duration(milliseconds: f64) -> String {
    format!("{:.1}s", milliseconds / 1000.0)
}

/// Milliseconds since the epoch, upstream's `Date.now()`.
fn now_millis() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |duration| duration.as_secs_f64() * 1000.0)
}

/// The shell call row, upstream's `formatShellCall`.
fn format_shell_call(args: Option<&serde_json::Value>, prompt: &str) -> String {
    let theme = theme();
    let command = string_arg(args.and_then(|args| args.get("command")));
    let timeout = args
        .and_then(|args| args.get("timeout"))
        .and_then(serde_json::Value::as_f64);
    let timeout_suffix = timeout.map_or_else(String::new, |timeout| {
        theme.fg("muted", &format!(" (timeout {}s)", format_number(timeout)))
    });
    let command_display = match command {
        None => invalid_arg_text(&theme),
        Some(command) if command.is_empty() => theme.fg("toolOutput", "..."),
        Some(command) => command,
    };
    format!(
        "{}{}",
        theme.fg("toolTitle", &bold(&format!("{prompt} {command_display}"))),
        timeout_suffix
    )
}

/// Format an integral JSON number without a fractional part.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the integral guard and the f64 source bound the cast to an exactly representable integer"
)]
fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// The preview's width-keyed line cache, upstream's `BashResultRenderState`.
#[derive(Default)]
struct PreviewState {
    width: Option<usize>,
    lines: Option<Vec<String>>,
    skipped: Option<usize>,
}

/// The collapsed output block, upstream's inline render function: it wraps at
/// render-time width and caches until the width changes or the row is
/// invalidated.
struct PreviewComponent {
    styled_output: String,
    state: RefCell<PreviewState>,
}

impl Component for PreviewComponent {
    fn render(&self, width: usize) -> Vec<String> {
        let mut state = self.state.borrow_mut();
        if state.lines.is_none() || state.width != Some(width) {
            let preview =
                truncate_to_visual_lines(&self.styled_output, BASH_PREVIEW_LINES, width, 0);
            state.lines = Some(preview.visual_lines);
            state.skipped = Some(preview.skipped_count);
            state.width = Some(width);
        }
        let skipped = state.skipped.unwrap_or(0);
        let mut lines = vec![String::new()];
        if skipped > 0 {
            let earlier_lines = format!("... ({skipped} earlier lines,");
            let expand_hint = format!(" {}", key_hint("app.tools.expand", "to expand"));
            let hint = format!(
                "{}{}{}",
                theme().fg("muted", &earlier_lines),
                expand_hint,
                theme().fg("muted", ")")
            );
            lines.push(truncate_to_width(&hint, width, "...", true));
        }
        lines.extend(state.lines.clone().unwrap_or_default());
        lines
    }

    fn invalidate(&self) {
        *self.state.borrow_mut() = PreviewState::default();
    }
}

/// The shell result body, upstream's `BashResultRenderComponent`.
pub struct BashResultRenderComponent {
    container: Container,
}

impl std::fmt::Debug for BashResultRenderComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BashResultRenderComponent").finish()
    }
}

impl BashResultRenderComponent {
    /// An empty shell result row.
    #[must_use]
    pub fn new() -> Self {
        Self {
            container: Container::default(),
        }
    }

    /// Clear and rebuild the row, upstream's `rebuildBashResultRenderComponent`.
    pub fn rebuild(
        &self,
        result: &AgentToolResult,
        options: ToolRenderResultOptions,
        show_images: bool,
        started_at: Option<f64>,
        ended_at: Option<f64>,
    ) {
        let theme = theme();
        self.container.clear();

        let mut output = get_text_output(Some(result), show_images)
            .trim()
            .to_string();
        let truncation = details_truncation(result);
        let full_output_path = result
            .details
            .get("fullOutputPath")
            .and_then(serde_json::Value::as_str);
        if !options.is_partial
            && truncation.as_ref().is_some_and(|value| value.truncated)
            && output.ends_with(']')
            && let Some(full_output_path) = full_output_path
            && let Some(footer_start) = output.rfind("\n\n[")
            && output[footer_start..].contains(full_output_path)
        {
            output = output[..footer_start].trim_end().to_string();
        }

        if !output.is_empty() {
            let styled_output = output
                .split('\n')
                .map(|line| theme.fg("toolOutput", line))
                .collect::<Vec<_>>()
                .join("\n");
            if options.expanded {
                self.container.add_child(Rc::new(Text::with_padding(
                    format!("\n{styled_output}"),
                    0,
                    0,
                )));
            } else {
                self.container.add_child(Rc::new(PreviewComponent {
                    styled_output,
                    state: RefCell::new(PreviewState::default()),
                }));
            }
        }

        let truncated = truncation.as_ref().is_some_and(|value| value.truncated);
        if truncated || full_output_path.is_some() {
            let mut warnings = Vec::new();
            if let Some(full_output_path) = full_output_path {
                warnings.push(format!("Full output: {full_output_path}"));
            }
            if let Some(truncation) = truncation.filter(|value| value.truncated) {
                if truncation.truncated_by == Some(crate::tools::truncate::TruncatedBy::Lines) {
                    warnings.push(format!(
                        "Truncated: showing {} of {} lines",
                        truncation.output_lines, truncation.total_lines
                    ));
                } else {
                    warnings.push(format!(
                        "Truncated: {} lines shown ({} limit)",
                        truncation.output_lines,
                        format_size(truncation.max_bytes)
                    ));
                }
            }
            self.container.add_child(Rc::new(Text::with_padding(
                format!(
                    "\n{}",
                    theme.fg("warning", &format!("[{}]", warnings.join(". ")))
                ),
                0,
                0,
            )));
        }

        if let Some(started_at) = started_at {
            let label = if options.is_partial {
                "Elapsed"
            } else {
                "Took"
            };
            let end_time = ended_at.unwrap_or_else(now_millis);
            self.container.add_child(Rc::new(Text::with_padding(
                format!(
                    "\n{}",
                    theme.fg(
                        "muted",
                        &format!("{label} {}", format_duration(end_time - started_at))
                    )
                ),
                0,
                0,
            )));
        }
    }
}

impl Default for BashResultRenderComponent {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for BashResultRenderComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_mouse(
        &self,
        event: &pi_tui::tui::TuiMouseEvent,
    ) -> Option<pi_tui::tui::TuiMouseEventResult> {
        self.container.handle_mouse(event)
    }

    fn invalidate(&self) {
        self.container.invalidate();
    }
}

/// The per-row shell state, upstream's `BashResultRenderState` plus timers.
#[derive(Default)]
struct ShellRenderState {
    started_at: Option<f64>,
    ended_at: Option<f64>,
    result_component: Option<Rc<BashResultRenderComponent>>,
}

/// Shell renderers shared by bash and powershell, which differ only in the
/// prompt they display.
#[must_use]
pub fn create_shell_renderers(prompt: &str) -> ToolRenderers {
    let prompt = prompt.to_string();
    let render_call: Arc<ToolRenderCallFn> = Arc::new(move |args, _theme, context| {
        let prompt = prompt.clone();
        let text = reuse_text(context);
        context
            .state
            .with_state_mut(ShellRenderState::default, |state| {
                if context.execution_started && state.started_at.is_none() {
                    state.started_at = Some(now_millis());
                    state.ended_at = None;
                }
            });
        text.set_text(&format_shell_call(Some(args), &prompt));
        text
    });
    let render_result: Arc<ToolRenderResultFn> = Arc::new(|result, options, _theme, context| {
        let (component, started_at, ended_at) = context
            .state
            .with_state_mut(ShellRenderState::default, |state| {
                if !options.is_partial || context.is_error {
                    state.ended_at.get_or_insert_with(now_millis);
                }
                let existing = state.result_component.clone();
                let component = if let Some(component) = existing {
                    component
                } else {
                    let component = Rc::new(BashResultRenderComponent::new());
                    state.result_component = Some(Rc::clone(&component));
                    component
                };
                (component, state.started_at, state.ended_at)
            })
            .unwrap_or_else(|| (Rc::new(BashResultRenderComponent::new()), None, None));
        component.rebuild(result, *options, context.show_images, started_at, ended_at);
        component.invalidate();
        component
    });
    ToolRenderers {
        render_call: Some(render_call),
        render_result: Some(render_result),
    }
}

//! Presentation for the edit tool, upstream
//! `src/core/tools/renderers/edit.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Restatement: upstream computes the pre-execution diff preview
//! asynchronously and invalidates the row when it settles. Rendering is
//! synchronous here, so the call row paints no preview until the tool's own
//! settled diff arrives in the result; the result path, including the
//! unchanged/error arms, matches upstream.

use std::rc::Rc;
use std::sync::Arc;

use pi_tui::components::{Box, ColorFn, Spacer, Text};
use pi_tui::tui::Component;
use pi_tui::tui::Container;

use super::ToolRenderers;
use crate::extensions::types::{AgentToolResult, ToolRenderCallFn, ToolRenderResultFn};
use crate::modes::interactive::components::diff::render_diff;
use crate::modes::interactive::theme::theme::{Theme, bold, theme};
use crate::tools::render_utils::{first_string_arg, render_tool_path, string_arg};

/// The state upstream keeps on the call component.
#[derive(Default)]
struct EditRenderState {
    call_component: Option<Rc<Box>>,
    preview_diff: Option<String>,
    preview_args_key: Option<String>,
    settled_error: bool,
}

/// The args key that resets a stale preview, upstream's
/// `JSON.stringify({path, edits})`.
fn preview_args_key(args: Option<&serde_json::Value>) -> Option<String> {
    let path = first_string_arg(args, &["path", "file_path"])?;
    if path.is_empty() {
        return None;
    }
    let edits = args
        .and_then(|args| args.get("edits"))
        .filter(|edits| edits.as_array().is_some_and(|edits| !edits.is_empty()))
        .cloned();
    let single = args.and_then(|args| {
        let old_text = string_arg(args.get("oldText"))?;
        let new_text = string_arg(args.get("newText"))?;
        Some(serde_json::json!([{ "oldText": old_text, "newText": new_text }]))
    });
    serde_json::to_string(&serde_json::json!({
        "path": path,
        "edits": edits.or(single),
    }))
    .ok()
}

/// The call header's background, upstream's `getEditHeaderBg`.
fn header_background(state: &EditRenderState) -> ColorFn {
    let color = if state.preview_diff.is_some() {
        "toolSuccessBg"
    } else if state.settled_error {
        "toolErrorBg"
    } else {
        "toolPendingBg"
    };
    let theme = theme();
    Arc::new(move |text| theme.bg(color, text))
}

/// The call header text, upstream's `formatEditCall`.
fn format_edit_call(args: Option<&serde_json::Value>, theme: &Theme, cwd: &str) -> String {
    let path_display = render_tool_path(
        first_string_arg(args, &["file_path", "path"]),
        theme,
        cwd,
        None,
    );
    format!("{} {path_display}", theme.fg("toolTitle", &bold("edit")))
}

/// Rebuild the call box from its state, upstream's `buildEditCallComponent`.
fn build_edit_call_component(
    component: &Box,
    args: Option<&serde_json::Value>,
    theme: &Theme,
    cwd: &str,
    state: &EditRenderState,
) {
    component.set_bg_fn(Some(header_background(state)));
    component.clear();
    component.add_child(Rc::new(Text::with_padding(
        format_edit_call(args, theme, cwd),
        0,
        0,
    )));
    if let Some(preview_diff) = &state.preview_diff {
        component.add_child(Rc::new(Spacer::new()));
        component.add_child(Rc::new(Text::with_padding(render_diff(preview_diff), 0, 0)));
    }
}

/// The reused or fresh call box, upstream's `getEditCallRenderComponent`.
fn get_call_component(state: &mut EditRenderState, last: Option<&Rc<dyn Component>>) -> Rc<Box> {
    if let Some(last) = last {
        let any: Rc<dyn std::any::Any> = last.clone();
        if let Ok(component) = any.downcast::<Box>() {
            state.call_component = Some(Rc::clone(&component));
            return component;
        }
    }
    if let Some(component) = &state.call_component {
        return Rc::clone(component);
    }
    let component = Rc::new(Box::with_padding(1, 1, None));
    state.call_component = Some(Rc::clone(&component));
    component
}

/// The error text of a failed edit result.
fn error_text(result: &AgentToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|block| match block {
            pi_ai::types::ToolResultBlock::Text(text) => Some(text.text.as_str()),
            pi_ai::types::ToolResultBlock::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The edit tool's renderer pair.
#[must_use]
pub fn edit_renderers() -> ToolRenderers {
    let render_call: Arc<ToolRenderCallFn> = Arc::new(|args, theme, context| {
        let args_key = preview_args_key(Some(args));
        context
            .state
            .with_state_mut(EditRenderState::default, |state| {
                let component = get_call_component(state, context.last_component.as_ref());
                if state.preview_args_key != args_key {
                    state.preview_diff = None;
                    state.preview_args_key = args_key;
                    state.settled_error = false;
                }
                build_edit_call_component(&component, Some(args), theme, context.cwd, state);
                component
            })
            .unwrap_or_else(|| Rc::new(Box::with_padding(1, 1, None)))
    });
    let render_result: Arc<ToolRenderResultFn> = Arc::new(|result, _options, theme, context| {
        let result_diff = if context.is_error {
            None
        } else {
            result
                .details
                .get("diff")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        };
        let args_key = preview_args_key(Some(context.args));
        context
            .state
            .with_state_mut(EditRenderState::default, |state| {
                if let Some(result_diff) = &result_diff {
                    state.preview_diff = Some(result_diff.clone());
                    state.preview_args_key = args_key;
                }
                state.settled_error = context.is_error;
                if let Some(component) = state.call_component.clone() {
                    build_edit_call_component(
                        &component,
                        Some(context.args),
                        theme,
                        context.cwd,
                        state,
                    );
                }
            });

        let component = Container::default();
        if context.is_error {
            let text = error_text(result);
            if !text.is_empty() {
                component.add_child(Rc::new(Spacer::new()));
                component.add_child(Rc::new(Text::with_padding(theme.fg("error", &text), 1, 0)));
            }
        }
        Rc::new(component)
    });
    ToolRenderers {
        render_call: Some(render_call),
        render_result: Some(render_result),
    }
}

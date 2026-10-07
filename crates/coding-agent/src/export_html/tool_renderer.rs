//! The tool HTML renderer for custom tools, upstream's
//! `src/core/export-html/tool-renderer.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Renders custom tool calls and results to HTML by invoking their TUI
//! renderers and converting the ANSI output to HTML. The extension-tool
//! surface the lookup serves rides its own slice (#128); until then the
//! source is an injectable lookup and the theme an opaque handle.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use pi_tui::tui::Component;
use serde_json::{Map as JsonMap, Value as JsonValue};

use pi_ai::types::ToolResultBlock;

use super::ansi_to_html::ansi_lines_to_html;

/// The theme handle the render hooks receive, upstream's `Theme` parameter.
///
/// The theme system rides its own slice (#132); until then the handle is
/// opaque — hooks that need colors downcast to their own concrete theme.
pub type ThemeHandle = Arc<dyn Any + Send + Sync>;

/// The call-side render hook, upstream's `toolDef.renderCall(args, theme,
/// context)` — the arguments and theme ride the context.
pub type ToolRenderCallHook =
    Box<dyn for<'a> Fn(&ToolRenderContext<'a>) -> Box<dyn Component> + Send + Sync>;

/// The result-side render hook, upstream's `toolDef.renderResult(result,
/// options, theme, context)`.
pub type ToolRenderResultHook = Box<
    dyn for<'a> Fn(
            &ToolAgentResult<'a>,
            &ToolRenderOptions,
            &ToolRenderContext<'a>,
        ) -> Box<dyn Component>
        + Send
        + Sync,
>;

/// The looked-up tool surface, upstream's `ToolDefinition` render members.
/// The full definition lands with the extensions slice (#128).
pub struct ToolRenderDefinition {
    /// The call-side hook, upstream's `renderCall`.
    pub render_call: Option<ToolRenderCallHook>,
    /// The result-side hook, upstream's `renderResult`.
    pub render_result: Option<ToolRenderResultHook>,
}

impl std::fmt::Debug for ToolRenderDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The hooks do not debug; the declarative surface names their
        // presence.
        f.debug_struct("ToolRenderDefinition")
            .field("render_call", &self.render_call.is_some())
            .field("render_result", &self.render_result.is_some())
            .finish()
    }
}

/// The result payload the result hook receives, upstream's `AgentToolResult`
/// build from the message content.
#[derive(Debug)]
pub struct ToolAgentResult<'a> {
    /// The result content, text and images.
    pub content: &'a [ToolResultBlock],
    /// The structured details from the tool execution.
    pub details: Option<&'a JsonValue>,
    /// Whether the tool execution failed.
    pub is_error: bool,
}

/// The result-hook options, upstream's `{ expanded, isPartial }`.
#[derive(Debug)]
pub struct ToolRenderOptions {
    /// Whether the result view is expanded.
    pub expanded: bool,
    /// Whether the result is partial/streaming.
    pub is_partial: bool,
}

/// The render context upstream's `ToolRenderContext` carries, populated by
/// the renderer per call.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the context restates upstream's ToolRenderContext flag-for-flag; each bool is one upstream field"
)]
pub struct ToolRenderContext<'a> {
    /// Current tool call arguments; shared across call/result renders.
    pub args: Option<&'a JsonMap<String, JsonValue>>,
    /// The unique id for this tool execution.
    pub tool_call_id: &'a str,
    /// Previously returned component for this render slot, if any.
    pub last_component: Option<Arc<dyn Component>>,
    /// Shared renderer state for this tool row.
    pub state: &'a mut JsonMap<String, JsonValue>,
    /// Working directory for this tool execution.
    pub cwd: &'a str,
    /// Whether the tool execution has started.
    pub execution_started: bool,
    /// Whether the tool call arguments are complete.
    pub args_complete: bool,
    /// Whether the result is partial/streaming.
    pub is_partial: bool,
    /// Whether the result view is expanded.
    pub expanded: bool,
    /// Whether inline images are currently shown in the TUI.
    pub show_images: bool,
    /// Whether the current result is an error.
    pub is_error: bool,
    /// The theme for styling, upstream's standalone `theme` parameter.
    pub theme: &'a ThemeHandle,
}

impl std::fmt::Debug for ToolRenderContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRenderContext")
            .field("args", &self.args.map(JsonMap::len))
            .field("tool_call_id", &self.tool_call_id)
            .field("last_component", &self.last_component.is_some())
            .field("state", &self.state.len())
            .field("cwd", &self.cwd)
            .field("execution_started", &self.execution_started)
            .field("args_complete", &self.args_complete)
            .field("is_partial", &self.is_partial)
            .field("expanded", &self.expanded)
            .field("show_images", &self.show_images)
            .field("is_error", &self.is_error)
            .finish_non_exhaustive()
    }
}

/// One render hook's HTML output, upstream's `{ collapsed?, expanded? }`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderedToolResult {
    /// The collapsed-result HTML.
    pub collapsed: Option<String>,
    /// The expanded-result HTML.
    pub expanded: Option<String>,
}

/// The tool HTML renderer seam, upstream's `ToolHtmlRenderer` interface the
/// export consumes.
pub trait ToolHtmlRenderer {
    /// Render a tool call to HTML; absent when the tool has no custom
    /// renderer.
    ///
    /// # Errors
    /// None: hook failures degrade to [`Option::None`], upstream's catch.
    fn render_call(
        &mut self,
        tool_call_id: &str,
        tool_name: &str,
        args: &JsonMap<String, JsonValue>,
    ) -> Option<String>;

    /// Render a tool result to collapsed/expanded HTML; absent when the tool
    /// has no custom renderer.
    ///
    /// # Errors
    /// None: hook failures degrade to [`Option::None`], upstream's catch.
    fn render_result(
        &mut self,
        tool_call_id: &str,
        tool_name: &str,
        result: &[ToolResultBlock],
        details: Option<&JsonValue>,
        is_error: bool,
    ) -> Option<RenderedToolResult>;
}

/// The lookup the renderer consults, upstream's `getToolDefinition`.
pub type ToolDefinitionLookup<'a> = &'a (dyn Fn(&str) -> Option<ToolRenderDefinition> + 'a);

/// The renderer's construction deps, upstream's `ToolHtmlRendererDeps`.
pub struct ToolHtmlRendererDeps<'a> {
    /// The tool-definition lookup by name.
    pub get_tool_definition: ToolDefinitionLookup<'a>,
    /// The theme for styling.
    pub theme: ThemeHandle,
    /// The working directory for the render context.
    pub cwd: String,
    /// The terminal width for rendering (default: 100).
    pub width: usize,
}

/// Strip ANSI SGR escape sequences, upstream's `ANSI_ESCAPE_REGEX` replace.
///
/// The cursor walks byte positions that always land on char boundaries: the
/// loop advances by `char.len_utf8()` after each copied character and by
/// whole escape sequences otherwise.
#[expect(
    clippy::expect_used,
    reason = "the cursor is on a char boundary by construction; the expect names the loop invariant"
)]
fn strip_ansi_escapes(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        if bytes[cursor] == 0x1b && bytes.get(cursor + 1) == Some(&b'[') {
            let mut end = cursor + 2;
            while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b';') {
                end += 1;
            }
            if bytes.get(end) == Some(&b'm') {
                cursor = end + 1;
                continue;
            }
        }
        let character = line[cursor..]
            .chars()
            .next()
            .expect("char boundary at cursor");
        out.push(character);
        cursor += character.len_utf8();
    }
    out
}

fn is_blank_rendered_line(line: &str) -> bool {
    strip_ansi_escapes(line).trim().is_empty()
}

/// Trim TUI spacing lines from rendered output, upstream's
/// `trimRenderedResultLines`.
fn trim_rendered_result_lines(lines: &[String]) -> &[String] {
    let mut start = 0;
    let mut end = lines.len();
    while start < end && is_blank_rendered_line(&lines[start]) {
        start += 1;
    }
    while end > start && is_blank_rendered_line(&lines[end - 1]) {
        end -= 1;
    }
    &lines[start..end]
}

/// The created tool HTML renderer, upstream's `createToolHtmlRenderer`.
pub struct CreatedToolHtmlRenderer<'a> {
    deps_lookup: ToolDefinitionLookup<'a>,
    theme: ThemeHandle,
    cwd: String,
    width: usize,
    rendered_call_components: HashMap<String, Arc<dyn Component>>,
    rendered_result_components: HashMap<String, Arc<dyn Component>>,
    rendered_states: HashMap<String, JsonMap<String, JsonValue>>,
    rendered_args: HashMap<String, JsonMap<String, JsonValue>>,
}

impl std::fmt::Debug for ToolHtmlRendererDeps<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The lookup is a closure and does not debug.
        f.debug_struct("ToolHtmlRendererDeps")
            .field("get_tool_definition", &())
            .field("theme", &"opaque")
            .field("cwd", &self.cwd)
            .field("width", &self.width)
            .finish()
    }
}

impl std::fmt::Debug for CreatedToolHtmlRenderer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The lookup and theme are closures/opaque handles and do not debug;
        // the per-call maps carry their sizes.
        f.debug_struct("CreatedToolHtmlRenderer")
            .field("cwd", &self.cwd)
            .field("width", &self.width)
            .field("rendered_calls", &self.rendered_call_components.len())
            .field("rendered_results", &self.rendered_result_components.len())
            .field("rendered_states", &self.rendered_states.len())
            .field("rendered_args", &self.rendered_args.len())
            .finish_non_exhaustive()
    }
}

impl<'a> CreatedToolHtmlRenderer<'a> {
    /// Create a tool HTML renderer, upstream's `createToolHtmlRenderer`.
    #[must_use]
    pub fn new(deps: ToolHtmlRendererDeps<'a>) -> Self {
        Self {
            deps_lookup: deps.get_tool_definition,
            theme: deps.theme,
            cwd: deps.cwd,
            width: deps.width,
            rendered_call_components: HashMap::new(),
            rendered_result_components: HashMap::new(),
            rendered_states: HashMap::new(),
            rendered_args: HashMap::new(),
        }
    }
}

impl ToolHtmlRenderer for CreatedToolHtmlRenderer<'_> {
    fn render_call(
        &mut self,
        tool_call_id: &str,
        tool_name: &str,
        args: &JsonMap<String, JsonValue>,
    ) -> Option<String> {
        self.rendered_args
            .insert(tool_call_id.to_owned(), args.clone());
        let hook = (self.deps_lookup)(tool_name)?.render_call?;

        // The context borrows disjoint renderer fields; the hook runs inside
        // the catch so a panicking renderer degrades to the structured
        // fallback the way upstream's try/catch does.
        let state = self
            .rendered_states
            .entry(tool_call_id.to_owned())
            .or_default();
        let context = ToolRenderContext {
            args: self.rendered_args.get(tool_call_id),
            tool_call_id,
            last_component: self.rendered_call_components.get(tool_call_id).cloned(),
            state,
            cwd: &self.cwd,
            execution_started: true,
            args_complete: true,
            is_partial: true,
            expanded: false,
            show_images: false,
            is_error: false,
            theme: &self.theme,
        };
        let component =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| hook(&context))).ok()?;
        let component = Arc::from(component);
        self.rendered_call_components
            .insert(tool_call_id.to_owned(), Arc::clone(&component));

        let lines = component.render(self.width);
        Some(ansi_lines_to_html(&lines))
    }

    fn render_result(
        &mut self,
        tool_call_id: &str,
        tool_name: &str,
        result: &[ToolResultBlock],
        details: Option<&JsonValue>,
        is_error: bool,
    ) -> Option<RenderedToolResult> {
        let hook = (self.deps_lookup)(tool_name)?.render_result?;

        // Build AgentToolResult from the content array, upstream's
        // agentToolResult literal.
        let agent_tool_result = ToolAgentResult {
            content: result,
            details,
            is_error,
        };

        // Render collapsed.
        let state = self
            .rendered_states
            .entry(tool_call_id.to_owned())
            .or_default();
        let context = ToolRenderContext {
            args: self.rendered_args.get(tool_call_id),
            tool_call_id,
            last_component: self.rendered_result_components.get(tool_call_id).cloned(),
            state,
            cwd: &self.cwd,
            execution_started: true,
            args_complete: true,
            is_partial: false,
            expanded: false,
            show_images: false,
            is_error,
            theme: &self.theme,
        };
        let collapsed_component = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            hook(
                &agent_tool_result,
                &ToolRenderOptions {
                    expanded: false,
                    is_partial: false,
                },
                &context,
            )
        }))
        .ok()?;
        let collapsed_component = Arc::from(collapsed_component);
        self.rendered_result_components
            .insert(tool_call_id.to_owned(), Arc::clone(&collapsed_component));
        let collapsed = ansi_lines_to_html(trim_rendered_result_lines(
            &collapsed_component.render(self.width),
        ));

        // Render expanded.
        let state = self
            .rendered_states
            .entry(tool_call_id.to_owned())
            .or_default();
        let context = ToolRenderContext {
            args: self.rendered_args.get(tool_call_id),
            tool_call_id,
            last_component: self.rendered_result_components.get(tool_call_id).cloned(),
            state,
            cwd: &self.cwd,
            execution_started: true,
            args_complete: true,
            is_partial: false,
            expanded: true,
            show_images: false,
            is_error,
            theme: &self.theme,
        };
        let expanded_component = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            hook(
                &agent_tool_result,
                &ToolRenderOptions {
                    expanded: true,
                    is_partial: false,
                },
                &context,
            )
        }))
        .ok()?;
        let expanded_component = Arc::from(expanded_component);
        self.rendered_result_components
            .insert(tool_call_id.to_owned(), Arc::clone(&expanded_component));
        let expanded = ansi_lines_to_html(trim_rendered_result_lines(
            &expanded_component.render(self.width),
        ));

        let collapsed = (!collapsed.is_empty() && collapsed != expanded).then_some(collapsed);
        Some(RenderedToolResult {
            collapsed,
            expanded: Some(expanded),
        })
    }
}

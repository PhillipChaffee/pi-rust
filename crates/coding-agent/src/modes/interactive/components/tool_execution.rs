//! The tool execution row, upstream
//! `src/modes/interactive/components/tool-execution.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The row draws one of three shells: a definition's own framing
//! (`renderShell: "self"`), the standard colored box around custom renderers,
//! or the generic text fallback. A click on any rendered row toggles the
//! expanded state.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::rc::Rc;
use std::sync::Arc;

use pi_ai::types::ToolResultBlock;
use pi_tui::components::image::Image;
use pi_tui::components::image::{ImageOptions, ImageTheme};
use pi_tui::components::{Box, ColorFn, MouseRegion, Spacer, Text};
use pi_tui::terminal_image::{ImageProtocol, get_capabilities};
use pi_tui::tui::{
    Component, Container, RenderRequest, TuiMouseButton, TuiMouseEvent, TuiMouseEventResult,
    TuiMouseEventType,
};
use serde_json::Value;

use crate::extensions::types::{
    AgentToolResult, RenderShell, ToolDefinition, ToolRenderContext, ToolRenderResultOptions,
    ToolRenderState,
};
use crate::modes::interactive::components::keybinding_hints::key_hint;
use crate::modes::interactive::theme::theme::{Theme, bold, theme};
use crate::tools::render_utils::get_text_output;
use crate::utils::image_convert::convert_to_png;

/// The generic fallback's collapsed preview lines.
const FALLBACK_PREVIEW_LINES: usize = 10;

/// The row's options.
#[derive(Clone, Copy, Debug, Default)]
pub struct ToolExecutionOptions {
    /// Whether inline images render.
    pub show_images: Option<bool>,
    /// The image width cap in cells.
    pub image_width_cells: Option<usize>,
}

/// The stored result, upstream's `{ content, isError, details }`.
struct RenderResult {
    result: AgentToolResult,
    is_error: bool,
}

/// One tool execution row.
pub struct ToolExecutionComponent {
    root: Container,
    content_box: Rc<Box>,
    content_text: Rc<Text>,
    self_render_container: Rc<Container>,
    self_render_height: Cell<usize>,
    call_renderer_component: RefCell<Option<Rc<dyn Component>>>,
    result_renderer_component: RefCell<Option<Rc<dyn Component>>>,
    renderer_state: Rc<ToolRenderState>,
    image_components: RefCell<Vec<Rc<dyn Component>>>,
    image_spacers: RefCell<Vec<Rc<dyn Component>>>,
    tool_name: String,
    tool_call_id: String,
    args: RefCell<Value>,
    expanded: Rc<Cell<bool>>,
    show_images: Cell<bool>,
    image_width_cells: Cell<usize>,
    is_partial: Cell<bool>,
    tool_definition: Option<ToolDefinition>,
    cwd: String,
    execution_started: Cell<bool>,
    args_complete: Cell<bool>,
    result: RefCell<Option<RenderResult>>,
    converted_images: RefCell<HashMap<usize, (String, String)>>,
    hide_component: Cell<bool>,
    dirty: Rc<Cell<bool>>,
    has_result: Rc<Cell<bool>>,
    invalidate_signal: Rc<dyn Fn()>,
}

impl std::fmt::Debug for ToolExecutionComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutionComponent")
            .field("tool_name", &self.tool_name)
            .field("tool_call_id", &self.tool_call_id)
            .finish_non_exhaustive()
    }
}

impl ToolExecutionComponent {
    /// Build the row, upstream's constructor.
    #[must_use]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the constructor stores the definition; callers keep the definition they pass"
    )]
    pub fn new(
        tool_name: &str,
        tool_call_id: &str,
        args: Value,
        options: ToolExecutionOptions,
        tool_definition: Option<ToolDefinition>,
        render_request: RenderRequest,
        cwd: &str,
    ) -> Self {
        let dirty = Rc::new(Cell::new(false));
        let expanded = Rc::new(Cell::new(false));
        let has_result = Rc::new(Cell::new(false));
        let invalidate_signal: Rc<dyn Fn()> = {
            let dirty = Rc::clone(&dirty);
            let render_request = Arc::clone(&render_request);
            Rc::new(move || {
                dirty.set(true);
                render_request();
            })
        };

        let has_definition = tool_definition.is_some();
        let content_box = Rc::new(Box::with_padding(1, 1, None));
        let content_text = Rc::new(Text::with_background("", 1, 1, None));
        let text_component: Rc<dyn Component> = content_text.clone();
        let content_text_region = result_region(
            text_component,
            Rc::clone(&has_result),
            Rc::clone(&expanded),
            Rc::clone(&dirty),
        );
        let self_render_container = Rc::new(Container::default());
        let root = Container::default();
        root.add_child(Rc::new(Spacer::new()));
        if has_definition {
            let self_shell = tool_definition
                .as_ref()
                .is_some_and(|definition| definition.render_shell == Some(RenderShell::SelfRender));
            if self_shell {
                let self_component: Rc<dyn Component> = self_render_container.clone();
                root.add_child(self_component);
            } else {
                let box_component: Rc<dyn Component> = content_box.clone();
                root.add_child(box_component);
            }
        } else {
            root.add_child(Rc::clone(&content_text_region));
        }

        let component = Self {
            root,
            content_box,
            content_text,
            self_render_container,
            self_render_height: Cell::new(0),
            call_renderer_component: RefCell::new(None),
            result_renderer_component: RefCell::new(None),
            renderer_state: Rc::new(ToolRenderState::new()),
            image_components: RefCell::new(Vec::new()),
            image_spacers: RefCell::new(Vec::new()),
            tool_name: tool_name.to_string(),
            tool_call_id: tool_call_id.to_string(),
            args: RefCell::new(args),
            expanded,
            show_images: Cell::new(options.show_images.unwrap_or(true)),
            image_width_cells: Cell::new(options.image_width_cells.unwrap_or(60)),
            is_partial: Cell::new(true),
            tool_definition,
            cwd: cwd.to_string(),
            execution_started: Cell::new(false),
            args_complete: Cell::new(false),
            result: RefCell::new(None),
            converted_images: RefCell::new(HashMap::new()),
            hide_component: Cell::new(false),
            dirty,
            has_result,
            invalidate_signal,
        };
        component.update_display();
        component
    }
}

/// Wrap a row in the click-to-expand region.
fn result_region(
    component: Rc<dyn Component>,
    has_result: Rc<Cell<bool>>,
    expanded: Rc<Cell<bool>>,
    dirty: Rc<Cell<bool>>,
) -> Rc<dyn Component> {
    let handler = move |event: &TuiMouseEvent| {
        if !has_result.get()
            || event.event_type != TuiMouseEventType::Click
            || event.button != TuiMouseButton::Left
        {
            return None;
        }
        expanded.set(!expanded.get());
        dirty.set(true);
        Some(TuiMouseEventResult {
            handled: true,
            ..TuiMouseEventResult::default()
        })
    };
    Rc::new(MouseRegion::new(component, handler))
}

impl ToolExecutionComponent {
    /// Replace the call arguments and repaint.
    pub fn update_args(&self, args: Value) {
        *self.args.borrow_mut() = args;
        self.update_display();
    }

    /// Mark the execution as started and repaint.
    pub fn mark_execution_started(&self) {
        self.execution_started.set(true);
        self.update_display();
        (self.invalidate_signal)();
    }

    /// Mark the call arguments as complete and repaint.
    pub fn set_args_complete(&self) {
        self.args_complete.set(true);
        self.update_display();
        (self.invalidate_signal)();
    }

    /// Store a result and repaint, upstream's `updateResult`.
    pub fn update_result(&self, result: AgentToolResult, is_error: bool, is_partial: bool) {
        *self.result.borrow_mut() = Some(RenderResult { result, is_error });
        self.is_partial.set(is_partial);
        self.convert_images_for_kitty();
        self.update_display();
    }

    /// Convert non-PNG images to PNG for Kitty terminals, upstream's
    /// `maybeConvertImagesForKitty`.
    fn convert_images_for_kitty(&self) {
        let caps = get_capabilities();
        if caps.images != Some(ImageProtocol::Kitty) {
            return;
        }
        let result = self.result.borrow();
        let Some(result) = result.as_ref() else {
            return;
        };
        let mut converted = self.converted_images.borrow_mut();
        for (index, block) in result.result.content.iter().enumerate() {
            let ToolResultBlock::Image(image) = block else {
                continue;
            };
            if image.mime_type == "image/png" || converted.contains_key(&index) {
                continue;
            }
            if let Some((data, mime_type)) = convert_to_png(&image.data, &image.mime_type) {
                converted.insert(index, (data, mime_type));
            }
        }
    }

    /// Set the expanded state and repaint.
    pub fn set_expanded(&self, expanded: bool) {
        self.expanded.set(expanded);
        self.update_display();
    }

    /// Set whether inline images render and repaint.
    pub fn set_show_images(&self, show: bool) {
        self.show_images.set(show);
        self.update_display();
    }

    /// Set the inline-image width cap and repaint.
    pub fn set_image_width_cells(&self, width: usize) {
        self.image_width_cells.set(width.max(1));
        self.update_display();
    }

    /// Whether a definition supplies the row's rendering.
    const fn has_renderer_definition(&self) -> bool {
        self.tool_definition.is_some()
    }

    /// The shell the definition asks for, upstream's `getRenderShell`.
    fn render_shell(&self) -> RenderShell {
        self.tool_definition
            .as_ref()
            .and_then(|definition| definition.render_shell)
            .unwrap_or(RenderShell::Default)
    }

    /// The render context handed to one slot's hook.
    fn render_context<'a>(
        &'a self,
        args: &'a Value,
        last_component: Option<Rc<dyn Component>>,
    ) -> ToolRenderContext<'a> {
        ToolRenderContext {
            args,
            tool_call_id: &self.tool_call_id,
            invalidate: self.invalidate_signal.as_ref(),
            last_component,
            state: &self.renderer_state,
            cwd: &self.cwd,
            execution_started: self.execution_started.get(),
            args_complete: self.args_complete.get(),
            is_partial: self.is_partial.get(),
            expanded: self.expanded.get(),
            show_images: self.show_images.get(),
            is_error: self
                .result
                .borrow()
                .as_ref()
                .is_some_and(|result| result.is_error),
        }
    }

    /// The generic call fallback, upstream's `createCallFallback`.
    fn create_call_fallback(&self) -> Rc<dyn Component> {
        Rc::new(Text::with_padding(
            theme().fg("toolTitle", &bold(&self.tool_name)),
            0,
            0,
        ))
    }

    /// The generic result fallback, upstream's `createResultFallback`.
    fn create_result_fallback(&self) -> Option<Rc<dyn Component>> {
        let output = self.text_output();
        if output.is_empty() {
            return None;
        }
        let theme = theme();
        let lines: Vec<&str> = output.split('\n').collect();
        let display_lines = if self.expanded.get() {
            lines.clone()
        } else {
            lines.iter().take(FALLBACK_PREVIEW_LINES).copied().collect()
        };
        let remaining = lines.len().saturating_sub(display_lines.len());
        let mut text = display_lines
            .iter()
            .map(|line| theme.fg("toolOutput", line))
            .collect::<Vec<_>>()
            .join("\n");
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
        Some(Rc::new(Text::with_padding(text, 0, 0)))
    }

    /// The result's text for the generic renderer, upstream's `getTextOutput`.
    fn text_output(&self) -> String {
        get_text_output(
            self.result.borrow().as_ref().map(|result| &result.result),
            self.show_images.get(),
        )
    }

    /// The generic row text, upstream's `formatToolExecution`.
    fn format_tool_execution(&self) -> String {
        let theme = theme();
        let mut text = theme.fg("toolTitle", &bold(&self.tool_name));
        if let Ok(args) = serde_json::to_string_pretty(&*self.args.borrow())
            && !args.is_empty()
        {
            let _ = write!(text, "\n\n{args}");
        }
        let output = self.text_output();
        if !output.is_empty() {
            let _ = write!(text, "\n{output}");
        }
        text
    }

    /// Rebuild the shell, renderers, and images, upstream's `updateDisplay`.
    #[expect(
        clippy::too_many_lines,
        reason = "the row's shell, renderer fallback ladder, and image pass are one upstream method"
    )]
    fn update_display(&self) {
        self.dirty.set(false);
        let theme = theme();
        let is_error = self
            .result
            .borrow()
            .as_ref()
            .is_some_and(|result| result.is_error);
        let background: ColorFn = if self.is_partial.get() {
            let theme = Arc::clone(&theme);
            Arc::new(move |text| theme.bg("toolPendingBg", text))
        } else if is_error {
            let theme = Arc::clone(&theme);
            Arc::new(move |text| theme.bg("toolErrorBg", text))
        } else {
            let theme = Arc::clone(&theme);
            Arc::new(move |text| theme.bg("toolSuccessBg", text))
        };

        let mut children: Vec<Rc<dyn Component>> = Vec::new();
        self.hide_component.set(false);
        self.has_result.set(self.result.borrow().is_some());

        if self.has_renderer_definition() {
            let definition = self.tool_definition.as_ref().map(|definition| {
                (
                    definition.render_call.clone(),
                    definition.render_result.clone(),
                )
            });

            let call_component = definition
                .as_ref()
                .and_then(|(render_call, _)| render_call.as_ref())
                .map_or_else(
                    || {
                        self.call_renderer_component.borrow_mut().take();
                        self.create_call_fallback()
                    },
                    |render_call| {
                        let rendered =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                let args = self.args.borrow();
                                let last = self.call_renderer_component.borrow().clone();
                                let context = self.render_context(&args, last);
                                render_call(&args, &theme, &context)
                            }));
                        rendered.unwrap_or_else(|_| {
                            *self.call_renderer_component.borrow_mut() = None;
                            self.create_call_fallback()
                        })
                    },
                );
            *self.call_renderer_component.borrow_mut() = Some(Rc::clone(&call_component));
            children.push(result_region(
                call_component,
                Rc::clone(&self.has_result),
                Rc::clone(&self.expanded),
                Rc::clone(&self.dirty),
            ));

            let result = self.result.borrow();
            if let Some(stored) = result.as_ref() {
                let result_renderer = definition
                    .as_ref()
                    .and_then(|(_, render_result)| render_result.clone());
                let rendered = result_renderer.and_then(|render_result| {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let args = self.args.borrow();
                        let last = self.result_renderer_component.borrow().clone();
                        let context = self.render_context(&args, last);
                        render_result(
                            &stored.result,
                            &ToolRenderResultOptions {
                                expanded: self.expanded.get(),
                                is_partial: self.is_partial.get(),
                            },
                            &theme,
                            &context,
                        )
                    }))
                    .ok()
                });
                if let Some(component) = rendered {
                    *self.result_renderer_component.borrow_mut() = Some(Rc::clone(&component));
                    children.push(result_region(
                        component,
                        Rc::clone(&self.has_result),
                        Rc::clone(&self.expanded),
                        Rc::clone(&self.dirty),
                    ));
                } else {
                    *self.result_renderer_component.borrow_mut() = None;
                    if let Some(component) = self.create_result_fallback() {
                        children.push(result_region(
                            component,
                            Rc::clone(&self.has_result),
                            Rc::clone(&self.expanded),
                            Rc::clone(&self.dirty),
                        ));
                    }
                }
            }
        } else {
            self.content_text
                .set_custom_bg_fn(Some(Arc::clone(&background)));
            self.content_text.set_text(&self.format_tool_execution());
        }

        if self.render_shell() == RenderShell::SelfRender && self.has_renderer_definition() {
            self.self_render_container.clear();
            for child in children {
                self.self_render_container.add_child(child);
            }
        } else if self.has_renderer_definition() {
            self.content_box.set_bg_fn(Some(background));
            self.content_box.clear();
            for child in children {
                self.content_box.add_child(child);
            }
        }

        self.rebuild_images(&theme);
    }

    /// Rebuild the image rows under the shell, upstream's image loop.
    fn rebuild_images(&self, theme: &Arc<Theme>) {
        for component in self.image_components.borrow_mut().drain(..) {
            self.root.remove_child(&component);
        }
        for component in self.image_spacers.borrow_mut().drain(..) {
            self.root.remove_child(&component);
        }

        let result = self.result.borrow();
        let Some(stored) = result.as_ref() else {
            return;
        };
        let caps = get_capabilities();
        for (index, block) in stored.result.content.iter().enumerate() {
            let ToolResultBlock::Image(image) = block else {
                continue;
            };
            if caps.images.is_none() || !self.show_images.get() || image.data.is_empty() {
                continue;
            }
            let converted = self.converted_images.borrow().get(&index).cloned();
            let (data, mime_type) =
                converted.unwrap_or_else(|| (image.data.clone(), image.mime_type.clone()));
            if caps.images == Some(ImageProtocol::Kitty) && mime_type != "image/png" {
                continue;
            }
            let active = Arc::clone(theme);
            let fallback: ColorFn = Arc::new(move |text| active.fg("toolOutput", text));
            let spacer: Rc<dyn Component> = Rc::new(Spacer::new());
            self.root.add_child(Rc::clone(&spacer));
            self.image_spacers.borrow_mut().push(spacer);
            let image_component: Rc<dyn Component> = Rc::new(Image::new(
                &data,
                &mime_type,
                ImageTheme {
                    fallback_color: fallback,
                },
                ImageOptions {
                    max_width_cells: Some(self.image_width_cells.get()),
                    ..ImageOptions::default()
                },
            ));
            self.root.add_child(Rc::clone(&image_component));
            self.image_components.borrow_mut().push(image_component);
        }
    }
}

impl Component for ToolExecutionComponent {
    fn render(&self, width: usize) -> Vec<String> {
        if self.dirty.replace(false) {
            self.update_display();
        }
        if self.hide_component.get() {
            return Vec::new();
        }

        if self.has_renderer_definition() && self.render_shell() == RenderShell::SelfRender {
            let content_lines = self.self_render_container.render(width);
            self.self_render_height.set(content_lines.len());
            let images = self.image_components.borrow();
            if content_lines.is_empty() && images.is_empty() {
                return Vec::new();
            }
            let mut lines = Vec::new();
            if !content_lines.is_empty() {
                lines.push(String::new());
                lines.extend(content_lines);
            }
            let spacers = self.image_spacers.borrow();
            for (index, image) in images.iter().enumerate() {
                if let Some(spacer) = spacers.get(index) {
                    lines.extend(spacer.render(width));
                }
                lines.extend(image.render(width));
            }
            return lines;
        }

        self.root.render(width)
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        if !self.has_renderer_definition() || self.render_shell() != RenderShell::SelfRender {
            return self.root.handle_mouse(event);
        }
        if event.y == 0 || usize::from(event.y) > self.self_render_height.get() {
            return None;
        }
        self.self_render_container.handle_mouse(&TuiMouseEvent {
            y: event.y - 1,
            height: u16::try_from(self.self_render_height.get()).unwrap_or(u16::MAX),
            ..event.clone()
        })
    }

    fn invalidate(&self) {
        self.dirty.set(true);
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.root.children()
    }
}

//! Box component, ported from `packages/tui/src/components/box.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#42).
//!
//! A container that applies padding and background to all children.

use std::cell::RefCell;
use std::rc::Rc;

use crate::components::ColorFn;
use crate::tui::{Component, TuiMouseEvent, TuiMouseEventResult, dispatch_mouse_event};
use crate::utils::{apply_background_to_line, visible_width};

/// Upstream `RenderCache`: the rendered child lines, the width they were
/// produced for, the background sample they were styled against, and the
/// styled result.
struct RenderCache {
    child_lines: Vec<String>,
    width: usize,
    bg_sample: Option<String>,
    lines: Vec<String>,
}

/// Upstream's `mouseLayout`: the child heights the last render measured, so
/// mouse dispatch can map a row onto a child without re-rendering.
struct MouseLayout {
    width: u16,
    children: Vec<(Rc<dyn Component>, u16)>,
}

/// Box component - a container that applies padding and background to all
/// children.
pub struct Box {
    children: RefCell<Vec<Rc<dyn Component>>>,
    padding_x: u16,
    padding_y: u16,
    bg_fn: RefCell<Option<ColorFn>>,
    cache: RefCell<Option<RenderCache>>,
    mouse_layout: RefCell<Option<MouseLayout>>,
}

impl std::fmt::Debug for Box {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Box")
            .field("padding_x", &self.padding_x)
            .field("padding_y", &self.padding_y)
            .finish_non_exhaustive()
    }
}

impl Box {
    /// Upstream's default-argument constructor: one cell of padding on every
    /// side, no background.
    #[must_use]
    pub fn new() -> Self {
        Self::with_padding(1, 1, None)
    }

    /// Upstream's `new Box(paddingX, paddingY, bgFn)`.
    #[must_use]
    pub fn with_padding(padding_x: u16, padding_y: u16, bg_fn: Option<ColorFn>) -> Self {
        Self {
            children: RefCell::new(Vec::new()),
            padding_x,
            padding_y,
            bg_fn: RefCell::new(bg_fn),
            cache: RefCell::new(None),
            mouse_layout: RefCell::new(None),
        }
    }

    /// Upstream `addChild`.
    pub fn add_child(&self, component: Rc<dyn Component>) {
        self.children.borrow_mut().push(component);
        self.invalidate_cache();
    }

    /// Upstream `removeChild`: drops the first child that is the same
    /// component, leaving the rest in order.
    pub fn remove_child(&self, component: &Rc<dyn Component>) {
        let position = self
            .children
            .borrow()
            .iter()
            .position(|child| Rc::ptr_eq(child, component));
        if let Some(index) = position {
            self.children.borrow_mut().remove(index);
            self.invalidate_cache();
        }
    }

    /// Upstream `clear`.
    pub fn clear(&self) {
        self.children.borrow_mut().clear();
        self.invalidate_cache();
    }

    /// Upstream `setBgFn`. Deliberately does not invalidate: a background
    /// change is detected by sampling the new function's output at render
    /// time, upstream's `bgSample`.
    pub fn set_bg_fn(&self, bg_fn: Option<ColorFn>) {
        *self.bg_fn.borrow_mut() = bg_fn;
    }

    fn invalidate_cache(&self) {
        *self.cache.borrow_mut() = None;
    }

    /// Upstream `matchCache`, returning the cached lines on a hit.
    fn match_cache(
        &self,
        width: usize,
        child_lines: &[String],
        bg_sample: Option<&str>,
    ) -> Option<Vec<String>> {
        let cache = self.cache.borrow();
        let cache = cache.as_ref()?;
        (cache.width == width
            && cache.bg_sample.as_deref() == bg_sample
            && cache.child_lines.len() == child_lines.len()
            && cache
                .child_lines
                .iter()
                .zip(child_lines)
                .all(|(cached, fresh)| cached == fresh))
        .then(|| cache.lines.clone())
    }

    fn apply_bg(&self, line: &str, width: usize) -> String {
        let vis_len = visible_width(line);
        let pad_needed = width.saturating_sub(vis_len);
        let padded = line.to_string() + " ".repeat(pad_needed).as_str();

        if let Some(bg_fn) = self.bg_fn.borrow().as_ref() {
            apply_background_to_line(&padded, width, |text| bg_fn(text))
        } else {
            padded
        }
    }
}

impl Default for Box {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for Box {
    fn invalidate(&self) {
        self.invalidate_cache();
        let children = self.children.borrow().clone();
        for child in children {
            child.invalidate();
        }
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        let content_width = event
            .width
            .saturating_sub(self.padding_x.saturating_mul(2))
            .max(1);
        let content_y = event.y.checked_sub(self.padding_y)?;
        let content_x = event.x.checked_sub(self.padding_x)?;
        if content_x >= content_width {
            return None;
        }

        // Cached child heights from the last render, re-measured by
        // rendering when the content width changed. Upstream does not store
        // the re-measured layout here; only `render` refreshes the cache.
        let cached_layout = {
            let layout = self.mouse_layout.borrow();
            layout
                .as_ref()
                .filter(|layout| layout.width == content_width)
                .map(|layout| layout.children.clone())
        };
        let mouse_children = cached_layout.unwrap_or_else(|| {
            self.children
                .borrow()
                .iter()
                .map(|component| {
                    let height = u16::try_from(component.render(content_width as usize).len())
                        .unwrap_or(u16::MAX);
                    (Rc::clone(component), height)
                })
                .collect()
        });

        let mut child_y = 0u16;
        for (child, child_height) in mouse_children {
            if content_y >= child_y && content_y < child_y + child_height {
                return dispatch_mouse_event(
                    &child,
                    &TuiMouseEvent {
                        x: content_x,
                        y: content_y - child_y,
                        width: content_width,
                        height: child_height,
                        ..event.clone()
                    },
                );
            }
            child_y = child_y.saturating_add(child_height);
        }
        None
    }

    fn render(&self, width: usize) -> Vec<String> {
        if self.children.borrow().is_empty() {
            return Vec::new();
        }

        let padding_x = self.padding_x as usize;
        let padding_y = self.padding_y as usize;
        let content_width = width.saturating_sub(padding_x * 2).max(1);
        let left_pad = " ".repeat(padding_x);

        // Render all children
        let mut child_lines: Vec<String> = Vec::new();
        let mut mouse_children: Vec<(Rc<dyn Component>, u16)> = Vec::new();
        let children = self.children.borrow().clone();
        for child in &children {
            let lines = child.render(content_width);
            let height = u16::try_from(lines.len()).unwrap_or(u16::MAX);
            mouse_children.push((Rc::clone(child), height));
            for line in lines {
                child_lines.push(format!("{left_pad}{line}"));
            }
        }
        *self.mouse_layout.borrow_mut() = Some(MouseLayout {
            width: u16::try_from(content_width).unwrap_or(u16::MAX),
            children: mouse_children,
        });

        if child_lines.is_empty() {
            return Vec::new();
        }

        // Check if bgFn output changed by sampling
        let bg_sample = self.bg_fn.borrow().as_ref().map(|bg_fn| bg_fn("test"));

        // Check cache validity
        if let Some(cached) = self.match_cache(width, &child_lines, bg_sample.as_deref()) {
            return cached;
        }

        // Apply background and padding
        let mut result: Vec<String> = Vec::new();

        // Top padding
        for _ in 0..padding_y {
            result.push(self.apply_bg("", width));
        }

        // Content
        for line in &child_lines {
            result.push(self.apply_bg(line, width));
        }

        // Bottom padding
        for _ in 0..padding_y {
            result.push(self.apply_bg("", width));
        }

        // Update cache
        *self.cache.borrow_mut() = Some(RenderCache {
            child_lines,
            width,
            bg_sample,
            lines: result.clone(),
        });

        result
    }
}

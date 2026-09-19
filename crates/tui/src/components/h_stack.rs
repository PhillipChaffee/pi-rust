//! `HStack`, ported from `packages/tui/src/components/h-stack.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#44).
//!
//! A horizontal stack that composites members side by side at their
//! allocated widths, cross-aligned per the stack's align option. Upstream's
//! `class HStack extends Stack` becomes composition over the shared
//! the shared `StackCore` core (see `stack.rs`); the layout node answers
//! [`StackKind::Horizontal`].

use std::rc::Rc;

use crate::layout_node::{LayoutNode, LayoutViewport, StackKind};
use crate::tui::{Component, composite_tui_line};
use crate::utils::visible_width;

use super::stack::{StackChild, StackCore, StackEntryOptions, StackOptions, allocate_stack_sizes};

/// A horizontal stack, upstream `class HStack`.
pub struct HStack {
    core: StackCore,
}

impl std::fmt::Debug for HStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HStack")
            .field("gap", &self.core.gap)
            .field("align", &self.core.align)
            .finish_non_exhaustive()
    }
}

impl HStack {
    /// Upstream `new HStack(children, options)`.
    #[must_use]
    pub fn new(children: Vec<StackChild>, options: StackOptions) -> Rc<Self> {
        let stack = Self {
            core: StackCore::new(StackKind::Horizontal, options),
        };
        for child in children {
            match child {
                StackChild::Component(component) => {
                    stack
                        .core
                        .add_child(component, StackEntryOptions::default());
                }
                StackChild::Entry(entry) => stack.core.add_child(entry.component, entry.options),
            }
        }
        Rc::new(stack)
    }

    /// Upstream `Stack.addChild(component, options?)`.
    pub fn add_child(&self, component: Rc<dyn Component>, options: StackEntryOptions) {
        self.core.add_child(component, options);
    }

    /// Upstream `Stack.removeChild`.
    pub fn remove_child(&self, component: &Rc<dyn Component>) {
        self.core.remove_child(component);
    }

    /// Upstream `Stack.clear`.
    pub fn clear(&self) {
        self.core.clear();
    }
}

impl Component for HStack {
    fn render(&self, width: usize) -> Vec<String> {
        let safe_width = width.max(1);
        let viewport = LayoutViewport {
            width: safe_width,
            height: usize::MAX,
        };
        let entries = self.core.visible_entries(&viewport);
        if entries.is_empty() {
            return Vec::new();
        }

        let intrinsic_widths: Vec<usize> = entries
            .iter()
            .map(|entry| {
                entry
                    .component
                    .render(safe_width)
                    .iter()
                    .map(|line| visible_width(line))
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let widths =
            allocate_stack_sizes(&entries, &intrinsic_widths, Some(safe_width), self.core.gap);
        let rendered: Vec<Vec<String>> = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                if widths[index] == 0 {
                    Vec::new()
                } else {
                    entry.component.render(widths[index])
                }
            })
            .collect();
        let height = rendered.iter().map(Vec::len).max().unwrap_or(0);
        let mut result = vec![String::new(); height];
        let mut x = 0;
        for (index, lines) in rendered.iter().enumerate() {
            let offset = match self.core.align {
                crate::layout_node::StackAlign::Center => (height.saturating_sub(lines.len())) / 2,
                crate::layout_node::StackAlign::End => height.saturating_sub(lines.len()),
                _ => 0,
            };
            for (row, line) in lines.iter().enumerate() {
                let target = row + offset;
                if target >= result.len() {
                    continue;
                }
                result[target] =
                    composite_tui_line(&result[target], line, x, widths[index], safe_width);
            }
            x += widths[index] + self.core.gap;
        }
        result
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.core.children()
    }

    fn handle_mouse(
        &self,
        event: &crate::tui::TuiMouseEvent,
    ) -> Option<crate::tui::TuiMouseEventResult> {
        self.core.handle_mouse(event)
    }

    fn invalidate(&self) {
        self.core.invalidate_children();
    }

    fn layout_node(&self) -> Option<LayoutNode> {
        Some(LayoutNode::Stack(self.core.layout_node()))
    }
}

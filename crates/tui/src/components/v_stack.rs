//! `VStack`, ported from `packages/tui/src/components/v-stack.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#44).
//!
//! A vertical stack that renders without a frame: members stack with the
//! configured gaps, sized by the shared allocator with no available-size
//! constraint. Upstream's `class VStack extends Stack` becomes composition
//! over the shared `StackCore` (see `stack.rs`); the layout node the
//! engine consumes answers [`StackKind::Vertical`].

use std::rc::Rc;

use crate::layout_node::{LayoutNode, LayoutViewport, StackKind};
use crate::tui::Component;

use super::stack::{StackChild, StackCore, StackEntryOptions, StackOptions, allocate_stack_sizes};

/// A vertical stack, upstream `class VStack`.
pub struct VStack {
    core: StackCore,
}

impl std::fmt::Debug for VStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VStack")
            .field("gap", &self.core.gap)
            .field("align", &self.core.align)
            .finish_non_exhaustive()
    }
}

impl VStack {
    /// Upstream `new VStack(children, options)`.
    #[must_use]
    pub fn new(children: Vec<StackChild>, options: StackOptions) -> Rc<Self> {
        let stack = Self {
            core: StackCore::new(StackKind::Vertical, options),
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

impl Component for VStack {
    fn render(&self, width: usize) -> Vec<String> {
        let viewport = LayoutViewport {
            width: width.max(1),
            height: usize::MAX,
        };
        let entries = self.core.visible_entries(&viewport);
        let rendered: Vec<Vec<String>> = entries
            .iter()
            .map(|entry| entry.component.render(viewport.width))
            .collect();
        let sizes = allocate_stack_sizes(
            &entries,
            &rendered.iter().map(Vec::len).collect::<Vec<_>>(),
            None,
            self.core.gap,
        );
        let mut lines: Vec<String> = Vec::new();
        for (index, _entry) in entries.iter().enumerate() {
            if index > 0 {
                for _ in 0..self.core.gap {
                    lines.push(String::new());
                }
            }
            let child_lines: Vec<String> =
                rendered[index].iter().take(sizes[index]).cloned().collect();
            lines.extend(child_lines.iter().cloned());
            for _ in child_lines.len()..sizes[index] {
                lines.push(String::new());
            }
        }
        lines
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

    fn is_stock_mouse_container(&self) -> bool {
        true
    }

    fn invalidate(&self) {
        self.core.invalidate_children();
    }

    fn layout_node(&self) -> Option<LayoutNode> {
        Some(LayoutNode::Stack(self.core.layout_node()))
    }
}

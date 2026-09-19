//! Mouse region, ported from `packages/tui/src/components/mouse-region.ts`
//! in earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#43).
//!
//! Adds mouse handling to an existing component without changing its
//! rendering: the child's own dispatch result wins, and the region's handler
//! answers only when the child did not. Upstream's `MouseRegionHandler`
//! becomes a plain `Fn` the region stores.

use std::rc::Rc;

use crate::tui::{Component, TuiMouseEvent, TuiMouseEventResult, dispatch_mouse_event};

/// The region's mouse handler, upstream `MouseRegionHandler`.
pub type MouseRegionHandler<'a> = dyn Fn(&TuiMouseEvent) -> Option<TuiMouseEventResult> + 'a;

/// Adds mouse handling to an existing component without changing its
/// rendering, upstream `class MouseRegion`.
pub struct MouseRegion<H: Fn(&TuiMouseEvent) -> Option<TuiMouseEventResult>> {
    child: Rc<dyn Component>,
    on_mouse: H,
}

impl std::fmt::Debug for MouseRegion<Box<dyn Fn(&TuiMouseEvent) -> Option<TuiMouseEventResult>>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MouseRegion").finish_non_exhaustive()
    }
}

impl<H: Fn(&TuiMouseEvent) -> Option<TuiMouseEventResult> + 'static> MouseRegion<H> {
    /// Upstream's constructor: `new MouseRegion(child, onMouse)`.
    pub fn new(child: Rc<dyn Component>, on_mouse: H) -> Self {
        Self { child, on_mouse }
    }
}

impl<H: Fn(&TuiMouseEvent) -> Option<TuiMouseEventResult> + 'static> Component for MouseRegion<H> {
    fn render(&self, width: usize) -> Vec<String> {
        self.child.render(width)
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        let child_result = dispatch_mouse_event(&self.child, event);
        child_result.or_else(|| (self.on_mouse)(event))
    }

    fn invalidate(&self) {
        self.child.invalidate();
    }
}

//! The component contract of `packages/tui/src/tui.ts` in earendil-works/pi
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Landed ahead of the full TUI core because the leaf components implement
//! it (#42); the TUI machinery itself — the overlay stack, render
//! scheduling, and input routing — lands with the TUI-core ticket (#43).
//!
//! Restatements against upstream, all surveyed in map ticket "Survey the tui
//! package":
//!
//! - The optional `handleInput?`/`handleMouse?` members and the duck-typed
//!   `Focusable` check become trait methods with defaults plus `Any`
//!   downcasting for focus and mouse dispatch (survey flag 1).
//! - `wantsKeyRelease?: boolean` becomes [`Component::wants_key_release`], a
//!   default method; traits cannot carry a field, and every ported component
//!   keeps the upstream `false` default.
//! - Upstream duck-types two result shapes on `handleMouse` — a plain
//!   `TuiMouseEventResult` or a `TuiMouseDispatchResult` carrying a target.
//!   The port folds them into [`TuiMouseEventResult`] with an optional
//!   `target`: a result whose `target` is already set passes through
//!   [`dispatch_mouse_event`] verbatim, exactly matching upstream's
//!   `"target" in result` branch.
//! - `Loader`'s `ui: TUI` parameter narrows to the one method it calls: the
//!   render request is a [`RenderRequest`] closure seam. The full `TUI`
//!   interface lands with #43; the concrete TUI supplies the closure.

use std::any::Any;
use std::rc::Rc;
use std::sync::Arc;

/// Upstream `TuiMouseEventType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiMouseEventType {
    /// Upstream `"press"`.
    Press,
    /// Upstream `"release"`.
    Release,
    /// Upstream `"move"`.
    Move,
    /// Upstream `"drag"`.
    Drag,
    /// Upstream `"click"`.
    Click,
    /// Upstream `"wheel"`.
    Wheel,
}

/// Upstream `TuiMouseButton`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiMouseButton {
    /// Upstream `"left"`.
    Left,
    /// Upstream `"middle"`.
    Middle,
    /// Upstream `"right"`.
    Right,
    /// Upstream `"none"` — wheel and button-less moves.
    None,
}

/// Normalized cell-based mouse event, upstream `TuiMouseEvent`.
#[derive(Debug, Clone)]
pub struct TuiMouseEvent {
    /// Upstream `type`.
    pub event_type: TuiMouseEventType,
    /// Upstream `button`.
    pub button: TuiMouseButton,
    /// Coordinates local to the receiving component, zero-based.
    pub x: u16,
    /// Coordinates local to the receiving component, zero-based.
    pub y: u16,
    /// Absolute terminal coordinates, zero-based.
    pub screen_x: u16,
    /// Absolute terminal coordinates, zero-based.
    pub screen_y: u16,
    /// Current component bounds.
    pub width: u16,
    /// Current component bounds.
    pub height: u16,
    /// Upstream `shift`.
    pub shift: bool,
    /// Upstream `alt`.
    pub alt: bool,
    /// Upstream `ctrl`.
    pub ctrl: bool,
    /// Logical lines. Negative values scroll up.
    pub wheel_delta: Option<i32>,
    /// Consecutive click count when the event type is [`TuiMouseEventType::Click`].
    pub click_count: Option<u32>,
}

/// Result of dispatching to a concrete component.
///
/// Upstream splits this into `TuiMouseEventResult` and
/// `TuiMouseDispatchResult`; the port folds them into one struct with an
/// optional `target` — see the module docs.
#[derive(Clone, Default)]
pub struct TuiMouseEventResult {
    /// Stop propagation and suppress renderer-level fallback behavior.
    pub handled: bool,
    /// Route subsequent drag/release events to this component. Implies `handled`.
    pub capture: bool,
    /// Give keyboard focus to this component. Implies `handled`.
    pub focus: bool,
    /// Explicitly request or suppress a render. Move and release default to
    /// `false`; press, click, drag, and wheel default to `true`.
    pub render: Option<bool>,
    /// Target metadata [`dispatch_mouse_event`] attaches: the component that
    /// handled the event and the coordinate transform to reach it.
    pub target: Option<TuiMouseDispatchTarget>,
    /// Keyboard focus target, which may be a delegating parent container.
    pub focus_target: Option<Rc<dyn Component>>,
}

impl std::fmt::Debug for TuiMouseEventResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuiMouseEventResult")
            .field("handled", &self.handled)
            .field("capture", &self.capture)
            .field("focus", &self.focus)
            .field("render", &self.render)
            .field("target", &self.target)
            .finish_non_exhaustive()
    }
}

/// Internal target metadata used by containers and alternate-screen dispatch,
/// upstream `TuiMouseDispatchTarget`.
#[derive(Clone)]
pub struct TuiMouseDispatchTarget {
    /// The component the event landed on.
    pub component: Rc<dyn Component>,
    /// Screen coordinate of the component's local origin.
    pub origin_x: u16,
    /// Screen coordinate of the component's local origin.
    pub origin_y: u16,
    /// Bounds at dispatch time.
    pub width: u16,
    /// Bounds at dispatch time.
    pub height: u16,
}

impl std::fmt::Debug for TuiMouseDispatchTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuiMouseDispatchTarget")
            .field("origin_x", &self.origin_x)
            .field("origin_y", &self.origin_y)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

/// The component interface every TUI widget implements. Stateful components
/// mutate through interior mutability; the TUI holds them as
/// `Rc<dyn Component>`.
///
/// Upstream `Component`: `render` is required, `handleInput`/`handleMouse`
/// are optional members, `wantsKeyRelease` is an optional property, and
/// `invalidate` is required but usually a no-op. The port restates the
/// optional members as defaulted trait methods.
pub trait Component: Any {
    /// Render the component to lines for the given viewport width.
    fn render(&self, width: usize) -> Vec<String>;

    /// Handler for keyboard input when the component has focus, upstream
    /// `handleInput`.
    fn handle_input(&self, data: &str) {
        let _ = data;
    }

    /// Normalized mouse handler, upstream `handleMouse`.
    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        let _ = event;
        None
    }

    /// Whether the component receives key release events (Kitty protocol);
    /// upstream `wantsKeyRelease`. Release events are filtered out by default.
    fn wants_key_release(&self) -> bool {
        false
    }

    /// Invalidate any cached rendering state. Called when the theme changes
    /// or the component must re-render from scratch.
    fn invalidate(&self) {}
}

/// Dispatch an event to a component and retain the exact target and
/// coordinate transform. Containers use this when forwarding events to
/// nested children, upstream `dispatchMouseEvent`.
///
/// A component result whose `target` is already set passes through verbatim —
/// upstream's `"target" in result` branch — so nested dispatches keep the
/// innermost target. Otherwise a result that sets none of `handled`,
/// `capture`, or `focus` is dropped, and anything else is re-stamped with
/// `handled: true`, a target built from this component, and a focus target
/// when `focus` was requested.
pub fn dispatch_mouse_event(
    component: &Rc<dyn Component>,
    event: &TuiMouseEvent,
) -> Option<TuiMouseEventResult> {
    let result = component.handle_mouse(event)?;
    if result.target.is_some() {
        return Some(result);
    }
    if !result.handled && !result.capture && !result.focus {
        return None;
    }
    Some(TuiMouseEventResult {
        handled: true,
        target: Some(TuiMouseDispatchTarget {
            component: Rc::clone(component),
            origin_x: event.screen_x.saturating_sub(event.x),
            origin_y: event.screen_y.saturating_sub(event.y),
            width: event.width,
            height: event.height,
        }),
        focus_target: result.focus.then(|| Rc::clone(component)),
        ..result
    })
}

/// The render-request seam [`crate::components::Loader`] uses to ask the TUI
/// to repaint, upstream `TUI.requestRender`.
///
/// The full `TUI` interface lands with the TUI-core ticket (#43); until then
/// the loader takes a closure.
pub type RenderRequest = Arc<dyn Fn() + Send + Sync>;

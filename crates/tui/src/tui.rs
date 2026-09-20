//! The TUI core of `packages/tui/src/tui.ts` in earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` ([#43](https://github.com/PhillipChaffee/pi-rust/issues/43)).
//!
//! It carries the component contract, the mouse event types and dispatch
//! helpers, the `Container`, the overlay stack and its focus-restore
//! machinery, [`composite_tui_line`], the cursor marker, the render
//! scheduler, and the `ViewportTUI` capability.
//!
//! The abstract `TuiBase` carries no rendering strategy of its own: concrete
//! renderers ([`TuiRenderer`]) supply `do_render` and the start/stop hooks,
//! exactly upstream's `TuiMainScreen` ([#45](https://github.com/PhillipChaffee/pi-rust/issues/45)) and
//! `TuiAltScreen` ([#46](https://github.com/PhillipChaffee/pi-rust/issues/46)).
//!
//! The base owns the terminal, the focus state, the input pipeline, the
//! overlay stack, and the scheduler; what upstream declared `protected` is
//! the renderer contract here, with the base passed to the renderer as
//! [`Tui`].
//!
//! Restatements against upstream, all surveyed in map ticket "Survey the tui
//! package":
//!
//! - The optional `handleInput?`/`handleMouse?` members, the duck-typed
//!   `Focusable` check, and `Container`'s `instanceof` walk become trait
//!   methods with defaults plus `Any` downcasting (survey flag 1). Presence
//!   of `handleInput` — which upstream checked on delegating containers and
//!   before routing input — becomes [`Component::wants_input`], a defaulted
//!   query.
//! - `process.nextTick` immediate-render preemption and the 16 ms
//!   `setTimeout` throttle (survey flag 4) become a pump: [`Tui::poll`] is
//!   the owner's event-loop turn — it pumps the terminal, dispatches queued
//!   input, then runs the scheduler.
//!
//!   Input delivered through the [`Terminal::poll`] window is queued and
//!   dispatched once the pump's terminal borrow is released, so a
//!   component's input handler can still write to the terminal, as upstream
//!   did inside the same event-loop turn. The input-preempts-throttle
//!   behavior documented at `tui.ts:971-972` is preserved by running the
//!   immediate path before the throttled deadline and cancelling the armed
//!   deadline.
//!
//!   `requestRender` from the loader's animation thread only raises an
//!   atomic flag ([`Tui::render_request`]); forced renders and `render_now`
//!   stay owner-thread, matching upstream's main-loop-only render calls.
//! - The promise-based terminal queries ([`Tui::query_terminal_background_color`],
//!   [`Tui::query_terminal_color_scheme`]) resolve through `mpsc` channels
//!   whose deadlines fire on [`Tui::poll`], upstream's `setTimeout` under
//!   the same event loop; an unpumped session leaves the receiver pending,
//!   which an owner that never reads queries never observes.
//!
//! - `addInputListener`/`onTerminalColorSchemeChange` return unsubscribe
//!   closures upstream; the port returns [`ListenerId`] handles for
//!   [`Tui::remove_input_listener`] and
//!   [`Tui::remove_terminal_color_scheme_listener`], since a closure cannot
//!   carry the list position it must delete.
//!
//! - `Symbol.for("@earendil-works/pi-tui/viewport")` becomes a capability
//!   the renderer declares ([`TuiRenderer::is_viewport_tui`], survey flag 1);
//!   `SizeValue = number | "50%"` becomes an enum (survey flag 6); the
//!   `getCapabilities().images` gate at `queryCellSize` becomes an injected
//!   probe, since the capability matrix itself is the image ticket's scope
//!   ([#51](https://github.com/PhillipChaffee/pi-rust/issues/51)).
//!
//! - `performance.now()` becomes an injected clock; `Terminal`-side
//!   restatements are the terminal module's own ([`crate::terminal`]).

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::rc::{Rc, Weak};
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::time::{Duration, Instant};

use crate::keys::{KeyParser, is_key_release};
use crate::terminal::{InputHandler, ResizeHandler, Terminal};
use crate::terminal_colors::{
    RgbColor, TerminalColorScheme, is_osc11_background_color_response,
    parse_osc11_background_color, parse_terminal_color_scheme_report,
};
use crate::terminal_image::{CellDimensions, is_image_line, set_cell_dimensions};
use crate::utils::{
    extract_segments, normalize_terminal_output, slice_by_column, slice_with_width, visible_width,
};

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
    /// Consecutive click count when the event type is
    /// [`TuiMouseEventType::Click`].
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

/// Dispatch an event to a component and retain the exact target and
/// coordinate transform. Containers use this when forwarding events to
/// nested children, upstream `dispatchMouseEvent`.
///
/// A component result whose `target` is already set passes through verbatim —
/// upstream's `"target" in result` branch — so nested dispatches keep the
/// innermost target, except that a delegating container ([`Component::wants_input`])
/// rewrites the focus target to itself, upstream's `focusTarget: this` on
/// `Container.handleMouse`. Otherwise a result that sets none of `handled`,
/// `capture`, or `focus` is dropped, and anything else is re-stamped with
/// `handled: true`, a target built from this component, and a focus target
/// when `focus` was requested.
pub fn dispatch_mouse_event(
    component: &Rc<dyn Component>,
    event: &TuiMouseEvent,
) -> Option<TuiMouseEventResult> {
    let result = component.handle_mouse(event)?;
    if result.target.is_some() {
        return Some(if result.focus && component.wants_input() {
            TuiMouseEventResult {
                focus_target: Some(Rc::clone(component)),
                ..result
            }
        } else {
            result
        });
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

/// Recreate local coordinates for a previously dispatched mouse target,
/// upstream `retargetMouseEvent`.
#[must_use]
pub fn retarget_mouse_event(
    event: &TuiMouseEvent,
    target: &TuiMouseDispatchTarget,
) -> TuiMouseEvent {
    let mut event = event.clone();
    event.x = event.screen_x.saturating_sub(target.origin_x);
    event.y = event.screen_y.saturating_sub(target.origin_y);
    event.width = target.width;
    event.height = target.height;
    event
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

    /// Whether the component receives keyboard input at all, upstream's
    /// optional `handleInput?` member: the TUI routes keyboard input only to
    /// focused components that answer `true`, and delegating containers
    /// rewrite nested mouse focus targets to themselves. Traits cannot query
    /// member existence, so receivers declare it.
    fn wants_input(&self) -> bool {
        false
    }

    /// Whether the component receives key release events (Kitty protocol);
    /// upstream `wantsKeyRelease`. Release events are filtered out by default.
    fn wants_key_release(&self) -> bool {
        false
    }

    /// Invalidate any cached rendering state. Called when the theme changes
    /// or the component must re-render from scratch.
    fn invalidate(&self) {}

    /// The child components, when the component is container-shaped —
    /// upstream's duck-typed `Container.children` walk. Non-containers
    /// answer empty.
    fn children(&self) -> Vec<Rc<dyn Component>> {
        Vec::new()
    }

    /// Upstream's duck-typed `Focusable` check: `Some(self)` when the
    /// component implements the cursor-marker focus contract.
    fn as_focusable(&self) -> Option<&dyn Focusable> {
        None
    }

    /// Upstream's `LAYOUT_NODE` capability
    /// (`Symbol.for("@earendil-works/pi-tui/layout-node")`): `Some(node)`
    /// when the component participates in the constrained layout engine
    /// (stacks and scroll views), `None` for plain components (survey
    /// flag 1).
    fn layout_node(&self) -> Option<crate::layout_node::LayoutNode> {
        None
    }
}

/// Interface for components that can receive focus and display a hardware
/// cursor, upstream `Focusable`.
///
/// When focused, the component should emit [`CURSOR_MARKER`] at the cursor
/// position in its render output; the TUI finds the marker and positions
/// the hardware cursor there for proper IME candidate window positioning.
///
/// Upstream duck-types a `focused` property; traits cannot carry a field,
/// so the state lives in the component behind interior mutability and the
/// TUI writes it through these methods.
pub trait Focusable {
    /// Set by the TUI when focus changes. The component should emit
    /// [`CURSOR_MARKER`] when true.
    fn set_focused(&self, focused: bool);

    /// Whether the component is currently focused.
    fn is_focused(&self) -> bool;
}

/// Whether a component implements the `Focusable` contract, upstream
/// `isFocusable`.
#[must_use]
pub fn is_focusable(component: Option<&Rc<dyn Component>>) -> bool {
    component.is_some_and(|component| component.as_focusable().is_some())
}

/// Cursor position marker — APC (Application Program Command) sequence.
///
/// This is a zero-width escape sequence that terminals ignore. Components
/// emit this at the cursor position when focused; the TUI finds and strips
/// the marker, then positions the hardware cursor there for proper IME
/// candidate window positioning.
pub const CURSOR_MARKER: &str = "\x1b_pi:c\x07";

/// The reset appended around composited overlay content, upstream
/// `SEGMENT_RESET`: a full SGR reset plus an OSC 8 terminator, so neither
/// the base's nor the overlay's styling leaks across the boundary.
const SEGMENT_RESET: &str = "\x1b[0m\x1b]8;;\x07";

/// Composite overlay content into a terminal line at a fixed column,
/// upstream `compositeTuiLine`.
///
/// The base line is split around the overlay region in a single
/// [`extract_segments`] pass so the after-segment inherits the styling that
/// was active at the boundary, and a wide grapheme the overlay starts inside
/// is excluded from the before-segment rather than leaking through.
#[must_use]
pub fn composite_tui_line(
    base_line: &str,
    overlay_line: &str,
    start_col: usize,
    overlay_width: usize,
    total_width: usize,
) -> String {
    if is_image_line(base_line) {
        return base_line.to_string();
    }

    let after_start = start_col + overlay_width;
    let base = extract_segments(
        base_line,
        start_col,
        after_start,
        total_width.saturating_sub(after_start),
        true,
    );
    let overlay = slice_with_width(overlay_line, 0, overlay_width, true);
    let before_pad = start_col.saturating_sub(base.before_width);
    let overlay_pad = overlay_width.saturating_sub(overlay.width);
    let actual_before_width = start_col.max(base.before_width);
    let actual_overlay_width = overlay_width.max(overlay.width);
    let after_target = total_width.saturating_sub(actual_before_width + actual_overlay_width);
    let after_pad = after_target.saturating_sub(base.after_width);
    let result = base.before
        + &" ".repeat(before_pad)
        + SEGMENT_RESET
        + &overlay.text
        + &" ".repeat(overlay_pad)
        + SEGMENT_RESET
        + &base.after
        + &" ".repeat(after_pad);

    if visible_width(&result) <= total_width {
        result
    } else {
        slice_by_column(&result, 0, total_width, true)
    }
}

/// Anchor position for overlays, upstream `OverlayAnchor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayAnchor {
    /// Upstream `"center"` — the default anchor.
    Center,
    /// Upstream `"top-left"`.
    TopLeft,
    /// Upstream `"top-right"`.
    TopRight,
    /// Upstream `"bottom-left"`.
    BottomLeft,
    /// Upstream `"bottom-right"`.
    BottomRight,
    /// Upstream `"top-center"`.
    TopCenter,
    /// Upstream `"bottom-center"`.
    BottomCenter,
    /// Upstream `"left-center"`.
    LeftCenter,
    /// Upstream `"right-center"`.
    RightCenter,
}

/// Per-side overlay margins, upstream `interface OverlayMargin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OverlayMarginSides {
    /// Margin above the overlay.
    pub top: i64,
    /// Margin right of the overlay.
    pub right: i64,
    /// Margin below the overlay.
    pub bottom: i64,
    /// Margin left of the overlay.
    pub left: i64,
}

/// Margin configuration, upstream `margin?: OverlayMargin | number`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayMargin {
    /// The same margin on all sides, upstream's number shorthand.
    All(i64),
    /// Independent margins, upstream's `OverlayMargin` object.
    Sides(OverlayMarginSides),
}

/// A size that is either absolute cells or a percentage of the reference
/// dimension, upstream `SizeValue` (survey flag 6).
///
/// Upstream spells the percentage arm as the template literal type
/// `number | "${number}%"`; the port makes it an enum arm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizeValue {
    /// Absolute cells. Row and column positions may be negative and are
    /// clamped to the terminal bounds.
    Cells(i64),
    /// Percentage of the reference dimension, upstream's `"50%"` spelling.
    Percent(f64),
}

/// Options for overlay positioning and sizing, upstream `OverlayOptions`.
/// Values can be absolute cells or percentages of the terminal dimension.
#[derive(Default)]
pub struct OverlayOptions {
    // === Sizing ===
    /// Width in columns, or percentage of terminal width, upstream `width`.
    /// `None` defaults to 80 columns capped by the available width.
    pub width: Option<SizeValue>,
    /// Minimum width in columns, upstream `minWidth`.
    pub min_width: Option<u32>,
    /// Maximum height in rows, or percentage of terminal height, upstream
    /// `maxHeight`.
    pub max_height: Option<SizeValue>,

    // === Positioning - anchor-based ===
    /// Anchor point for positioning, upstream `anchor` (default center).
    pub anchor: Option<OverlayAnchor>,
    /// Horizontal offset from the anchor position, positive right, upstream
    /// `offsetX`.
    pub offset_x: Option<i64>,
    /// Vertical offset from the anchor position, positive down, upstream
    /// `offsetY`.
    pub offset_y: Option<i64>,

    // === Positioning - percentage or absolute ===
    /// Row position: absolute number, or percentage of terminal height,
    /// upstream `row`.
    pub row: Option<SizeValue>,
    /// Column position: absolute number, or percentage of terminal width,
    /// upstream `col`.
    pub col: Option<SizeValue>,

    // === Margin from terminal edges ===
    /// Margin from terminal edges, upstream `margin`.
    pub margin: Option<OverlayMargin>,

    // === Visibility ===
    /// Control overlay visibility based on terminal dimensions, upstream
    /// `visible`. Called each render cycle with the current terminal
    /// dimensions; the overlay renders only while this returns true.
    pub visible: Option<Box<dyn Fn(u16, u16) -> bool>>,
    /// If true, don't capture keyboard focus when shown, upstream
    /// `nonCapturing`.
    pub non_capturing: bool,
}

impl std::fmt::Debug for OverlayOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OverlayOptions")
            .field("width", &self.width)
            .field("min_width", &self.min_width)
            .field("max_height", &self.max_height)
            .field("anchor", &self.anchor)
            .field("offset_x", &self.offset_x)
            .field("offset_y", &self.offset_y)
            .field("row", &self.row)
            .field("col", &self.col)
            .field("margin", &self.margin)
            .field("visible", &self.visible.is_some())
            .field("non_capturing", &self.non_capturing)
            .finish()
    }
}

/// Options for [`OverlayHandle::unfocus`], upstream `OverlayUnfocusOptions`.
#[derive(Clone, Default)]
pub struct OverlayUnfocusOptions {
    /// Explicit target to focus after releasing this overlay; `None` with
    /// the option itself present focuses nothing, upstream's `target: null`.
    pub target: Option<Rc<dyn Component>>,
}

impl std::fmt::Debug for OverlayUnfocusOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OverlayUnfocusOptions")
            .field("has_target", &self.target.is_some())
            .finish()
    }
}

/// Last rendered terminal-relative overlay rectangle, upstream
/// `OverlayBounds`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlayBounds {
    /// Terminal row of the overlay's top edge.
    pub row: usize,
    /// Terminal column of the overlay's left edge.
    pub col: usize,
    /// Overlay width in columns.
    pub width: usize,
    /// Overlay height in rows.
    pub height: usize,
}

/// Handle returned by [`Tui::show_overlay`] for controlling the overlay,
/// upstream `OverlayHandle`.
///
/// Upstream returns closures over the stack entry; the port returns a handle
/// holding the entry's id and the owning TUI, which every method resolves on
/// demand.
#[derive(Clone)]
pub struct OverlayHandle {
    tui: Weak<Tui>,
    entry_id: u64,
}

impl std::fmt::Debug for OverlayHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OverlayHandle")
            .field("entry_id", &self.entry_id)
            .finish_non_exhaustive()
    }
}

impl OverlayHandle {
    fn with_tui<R>(&self, f: impl FnOnce(&Tui) -> R) -> Option<R> {
        let tui = self.tui.upgrade()?;
        Some(f(&tui))
    }

    /// Permanently remove the overlay; it cannot be shown again, upstream
    /// `OverlayHandle.hide`.
    pub fn hide(&self) {
        self.with_tui(|tui| tui.overlay_hide(self.entry_id));
    }

    /// Temporarily hide or show the overlay, upstream
    /// `OverlayHandle.setHidden`.
    pub fn set_hidden(&self, hidden: bool) {
        self.with_tui(|tui| tui.overlay_set_hidden(self.entry_id, hidden));
    }

    /// Check if the overlay is temporarily hidden, upstream `isHidden`.
    #[must_use]
    pub fn is_hidden(&self) -> bool {
        self.with_tui(|tui| {
            tui.overlay_entry(self.entry_id)
                .is_some_and(|entry| entry.hidden)
        })
        .unwrap_or(false)
    }

    /// Focus this overlay and bring it to the visual front, upstream
    /// `OverlayHandle.focus`.
    pub fn focus(&self) {
        self.with_tui(|tui| tui.overlay_focus(self.entry_id));
    }

    /// Release focus to the next visible capturing overlay or previous
    /// target, or to an explicit target when provided, upstream
    /// `OverlayHandle.unfocus`.
    pub fn unfocus(&self, options: Option<OverlayUnfocusOptions>) {
        self.with_tui(|tui| tui.overlay_unfocus(self.entry_id, options));
    }

    /// Check if this overlay currently has focus, upstream `isFocused`.
    #[must_use]
    pub fn is_focused(&self) -> bool {
        self.with_tui(|tui| {
            tui.overlay_entry(self.entry_id).is_some_and(|entry| {
                tui.focused_component
                    .borrow()
                    .as_ref()
                    .is_some_and(|focused| {
                        Rc::as_ptr(focused).cast::<()>()
                            == Rc::as_ptr(&entry.component).cast::<()>()
                    })
            })
        })
        .unwrap_or(false)
    }

    /// Get the most recent rendered bounds for a visible overlay, upstream
    /// `OverlayHandle.getBounds`.
    #[must_use]
    pub fn get_bounds(&self) -> Option<OverlayBounds> {
        self.with_tui(|tui| {
            tui.overlay_entry(self.entry_id)
                .filter(|entry| entry.bounds.is_some() && tui.is_overlay_visible(entry))
                .and_then(|entry| entry.bounds)
        })
        .flatten()
    }
}

/// The unsubscribe handle upstream returned from `addInputListener` and
/// `onTerminalColorSchemeChange`; a closure cannot carry the list position
/// it deletes, so the port hands back ids.
pub type ListenerId = u64;

/// Result of an input listener, upstream `TuiInputListenerResult`: consume
/// the data, rewrite it, both, or neither.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TuiInputListenerResult {
    /// Consume the chunk and stop the input pipeline.
    pub consume: bool,
    /// Replacement data for the rest of the pipeline.
    pub data: Option<String>,
}

/// A registered input listener, upstream `TuiInputListener`. Listeners run
/// in registration order and may rewrite or consume the chunk.
///
/// The port stores them behind `Rc` so the dispatch can iterate a snapshot
/// while a listener mutates the registry, mirroring upstream's `Set`
/// iteration.
pub type TuiInputListener = Rc<dyn Fn(&str) -> Option<TuiInputListenerResult>>;

/// Mode of the TUI, upstream `TuiMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiMode {
    /// Upstream `"regular"` — main-screen scrollback rendering.
    Regular,
    /// Upstream `"fullscreen"` — alternate-screen rendering.
    Fullscreen,
}

/// Options for [`Tui::stop`], upstream `TuiStopOptions`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TuiStopOptions {
    /// Leave renderer output in place for another TUI taking over the same
    /// terminal.
    pub preserve_screen: bool,
}

/// The concrete rendering strategy behind a [`Tui`], restating the abstract
/// `TuiBase` subclass body: the abstract `doRender` plus the start/stop
/// hooks.
///
/// Upstream's `protected` members are the [`Tui`] methods the renderer
/// reaches through the base it is handed.
pub trait TuiRenderer: std::fmt::Debug {
    /// Upstream's abstract `readonly mode`.
    fn mode(&self) -> TuiMode;

    /// Upstream's abstract `doRender`.
    fn do_render(&self, tui: &Tui);

    /// Reset cached render state before a forced render, upstream
    /// `resetRenderState`.
    fn reset_render_state(&self) {}

    /// Hook before the terminal starts, upstream `beforeTerminalStart`.
    fn before_terminal_start(&self) {}

    /// Hook after the terminal starts, upstream `afterTerminalStart`.
    fn after_terminal_start(&self) {}

    /// Hook before the terminal stops, upstream `beforeTerminalStop`.
    fn before_terminal_stop(&self, _options: &TuiStopOptions) {}

    /// Hook after the terminal stops, upstream `afterTerminalStop`.
    fn after_terminal_stop(&self, _options: &TuiStopOptions) {}

    /// The roots the mounted-component walk consults, upstream
    /// `getMountedRoots`. The `ViewportTUI` capability overrides this to
    /// mount the layout root beside the children.
    fn mounted_roots(&self, tui: &Tui) -> Vec<Rc<dyn Component>> {
        Component::children(tui)
    }

    /// Whether this renderer carries the `ViewportTUI` capability, upstream
    /// `isViewportTUI` over `Symbol.for("@earendil-works/pi-tui/viewport")`.
    fn is_viewport_tui(&self) -> bool {
        false
    }

    /// Replace the mounted layout root, upstream
    /// `ViewportTUI.setLayoutRoot`. A no-op for renderers without the
    /// capability.
    fn set_layout_root(&self, _tui: &Tui, _component: Option<Rc<dyn Component>>) {}
}

/// Whether a TUI carries the `ViewportTUI` capability, upstream
/// `isViewportTUI`.
#[must_use]
pub fn is_viewport_tui(tui: &Tui) -> bool {
    tui.is_viewport_tui()
}

/// One overlay on the stack, upstream `OverlayStackEntry` held by reference —
/// the port identifies entries by [`Self::id`].
struct OverlayStackEntry {
    id: u64,
    component: Rc<dyn Component>,
    options: Option<OverlayOptions>,
    pre_focus: Option<Rc<dyn Component>>,
    hidden: bool,
    focus_order: u64,
    bounds: Option<OverlayBounds>,
}

/// Where a visible overlay rendered last frame, upstream
/// `RenderedOverlayLayout`.
#[derive(Clone)]
struct RenderedOverlayLayout {
    /// The overlay the layout belongs to, kept for parity with upstream's
    /// entry reference; the mouse dispatch resolves through the component.
    #[expect(
        dead_code,
        reason = "the renderer contract reads the component; the id rides for #46's selection handling"
    )]
    entry_id: u64,
    component: Rc<dyn Component>,
    row: usize,
    col: usize,
    width: usize,
    height: usize,
}

/// Focus-restore bookkeeping for the overlay stack, upstream
/// `OverlayFocusRestoreState` keyed by entry id.
#[derive(Clone)]
enum OverlayFocusRestoreState {
    /// No overlay focus to restore.
    Inactive,
    /// A visible capturing overlay holds focus and may be restored.
    Eligible { overlay: u64 },
    /// A base component took focus from a visible capturing overlay, upstream
    /// `BlockedOverlayFocusRestoreState`.
    Blocked {
        overlay: u64,
        blocked_by: Rc<dyn Component>,
        resume: OverlayBlockedFocusResume,
    },
}

impl OverlayFocusRestoreState {
    const fn overlay_id(&self) -> Option<u64> {
        match self {
            Self::Inactive => None,
            Self::Eligible { overlay } | Self::Blocked { overlay, .. } => Some(*overlay),
        }
    }
}

/// How a blocked overlay's focus resumes, upstream
/// `OverlayBlockedFocusResume`.
#[derive(Clone)]
enum OverlayBlockedFocusResume {
    /// Restore focus to the overlay itself.
    RestoreOverlay,
    /// Restore focus to an explicit target, upstream `focus-target`.
    FocusTarget { target: Option<Rc<dyn Component>> },
}

/// Whether a focus change clears or preserves the restore state, upstream
/// `OverlayFocusRestorePolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FocusRestorePolicy {
    Clear,
    Preserve,
}

/// A pending OSC 11 background-color query, upstream
/// `PendingOsc11BackgroundQuery` with its timer.
struct PendingOsc11BackgroundQuery {
    settled: bool,
    resolve: Option<std::sync::mpsc::Sender<Option<RgbColor>>>,
    deadline: Option<Instant>,
}

/// A pending color-scheme query waiting on the `CSI ? 996 n` reply.
struct SchemeQuery {
    listener_id: ListenerId,
    settled: bool,
    deadline: Instant,
}

/// A registered color-scheme listener, upstream `(scheme: TerminalColorScheme) => void`.
pub type TerminalColorSchemeListener = Rc<dyn Fn(&TerminalColorScheme)>;

/// The demand flags the scheduler coalesces, upstream's `renderRequested` /
/// `immediateRenderScheduled` pair.
///
/// The loader's animation thread calls `requestRender` from off the owner
/// thread, so the flags are atomics and the only cross-thread surface.
/// Forced renders and `renderNow` touch renderer state and stay
/// owner-thread, as every upstream render call ran on the main loop.
#[derive(Debug, Default)]
struct RenderDemand {
    render_requested: std::sync::atomic::AtomicBool,
    immediate_scheduled: std::sync::atomic::AtomicBool,
}

/// The layout cache `Container` commits on render, upstream's private
/// `mouseLayout`.
struct MouseLayout {
    width: usize,
    children: Vec<(Rc<dyn Component>, usize)>,
}

/// A component that contains other components, upstream `class Container`.
///
/// Rendering concatenates the children's lines and commits their heights for
/// the mouse dispatch. The cache is keyed by the width it was built at, so a
/// same-width dispatch reuses the committed heights and anything else
/// re-measures, exactly upstream's `mouseLayout` width check. The
/// `focusTarget: this` rewrite for delegating containers lives in
/// [`dispatch_mouse_event`], keyed off [`Component::wants_input`].
#[derive(Default)]
pub struct Container {
    /// The child components, upstream's public `children` field.
    pub children: RefCell<Vec<Rc<dyn Component>>>,
    mouse_layout: RefCell<Option<MouseLayout>>,
}

impl std::fmt::Debug for Container {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Container")
            .field("children", &self.children.borrow().len())
            .finish_non_exhaustive()
    }
}

impl Container {
    /// Upstream `addChild`.
    pub fn add_child(&self, component: Rc<dyn Component>) {
        self.children.borrow_mut().push(component);
    }

    /// Upstream `removeChild`.
    pub fn remove_child(&self, component: &Rc<dyn Component>) {
        self.children
            .borrow_mut()
            .retain(|child| !Rc::ptr_eq(child, component));
    }

    /// Upstream `clear`.
    pub fn clear(&self) {
        self.children.borrow_mut().clear();
    }
}

impl Component for Container {
    fn render(&self, width: usize) -> Vec<String> {
        let mut lines = Vec::new();
        let mut mouse_children = Vec::new();
        for child in self.children.borrow().iter() {
            let child_lines = child.render(width);
            mouse_children.push((Rc::clone(child), child_lines.len()));
            lines.extend(child_lines);
        }
        *self.mouse_layout.borrow_mut() = Some(MouseLayout {
            width,
            children: mouse_children,
        });
        lines
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        if event.y >= event.height {
            return None;
        }
        let cached = self
            .mouse_layout
            .borrow()
            .as_ref()
            .filter(|layout| layout.width == usize::from(event.width))
            .map(|layout| layout.children.clone());
        let mouse_children = cached.unwrap_or_else(|| {
            self.children
                .borrow()
                .iter()
                .map(|component| {
                    let height = component.render(usize::from(event.width)).len();
                    (Rc::clone(component), height)
                })
                .collect()
        });
        let mut child_y: usize = 0;
        for (child, child_height) in mouse_children {
            if usize::from(event.y) >= child_y && usize::from(event.y) < child_y + child_height {
                let child_event = TuiMouseEvent {
                    y: event.y - clamp_cell(child_y),
                    height: clamp_cell(child_height),
                    ..event.clone()
                };
                return dispatch_mouse_event(&child, &child_event);
            }
            child_y += child_height;
        }
        None
    }

    fn invalidate(&self) {
        for child in self.children.borrow().iter() {
            child.invalidate();
        }
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.children.borrow().clone()
    }
}

/// Rendered counts and terminal-relative positions ride terminal dimensions,
/// which crossterm keeps in `u16`; only a pathological component could
/// overflow, which clamps rather than panics.
#[expect(
    clippy::cast_possible_truncation,
    reason = "line counts and terminal-relative coordinates ride terminal dimensions, which are u16; the clamp bounds the degenerate case"
)]
const fn clamp_cell(value: usize) -> u16 {
    value as u16
}

/// The TUI core, upstream `TuiBase`. Construct through [`Tui::new`], which
/// returns it inside an `Rc` so the terminal's handlers and overlay handles
/// can reach it.
pub struct Tui {
    self_weak: Weak<Self>,
    terminal: RefCell<Box<dyn Terminal>>,
    renderer: RefCell<Box<dyn TuiRenderer>>,
    container: Container,
    focused_component: RefCell<Option<Rc<dyn Component>>>,
    input_listeners: RefCell<Vec<(ListenerId, TuiInputListener)>>,
    listener_next_id: Cell<ListenerId>,
    on_debug: RefCell<Option<Rc<dyn Fn()>>>,
    demand: Arc<RenderDemand>,
    render_deadline: Cell<Option<Instant>>,
    /// Upstream `lastRenderAt = 0`: `None` means never rendered, and a first
    /// render's throttle delay collapses to zero exactly as the zero
    /// sentinel made upstream's elapsed time unbounded.
    last_render_at: Cell<Option<Instant>>,
    show_hardware_cursor: Cell<bool>,
    clear_on_shrink: Cell<bool>,
    full_redraw_count: Cell<u32>,
    stopped: Cell<bool>,
    scheme_listeners: RefCell<Vec<(ListenerId, TerminalColorSchemeListener)>>,
    scheme_notifications_enabled: Cell<bool>,
    scheme_queries: RefCell<Vec<SchemeQuery>>,
    pending_osc11_replies: Cell<usize>,
    pending_osc11_queries: RefCell<VecDeque<PendingOsc11BackgroundQuery>>,
    /// Directory for debug/crash logs, upstream `protected readonly
    /// logDirectory`. When `None`, debug logging is disabled and crash dumps
    /// fall back to the OS temp directory.
    pub log_directory: Option<PathBuf>,
    focus_order_counter: Cell<u64>,
    overlay_next_id: Cell<u64>,
    overlay_stack: RefCell<Vec<OverlayStackEntry>>,
    rendered_overlay_layouts: RefCell<Vec<RenderedOverlayLayout>>,
    overlay_focus_restore: RefCell<OverlayFocusRestoreState>,
    key_parser: Arc<Mutex<KeyParser>>,
    images_probe: Box<dyn Fn() -> bool>,
    clock: Box<dyn Fn() -> Instant>,
    pending_input: RefCell<VecDeque<String>>,
}

impl std::fmt::Debug for Tui {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tui")
            .field("mode", &self.mode())
            .field("stopped", &self.stopped.get())
            .field("overlays", &self.overlay_stack.borrow().len())
            .field("focused", &self.focused_component.borrow().is_some())
            .finish_non_exhaustive()
    }
}

/// Construction seam for [`Tui`], restating the `TuiBase` constructor's
/// optional parameters and the seams upstream monkeypatched in tests.
#[derive(Default)]
pub struct TuiConfig {
    /// The terminal this TUI renders through, upstream `terminal`.
    pub terminal: Option<Box<dyn Terminal>>,
    /// The concrete rendering strategy, upstream's `TuiBase` subclass body.
    pub renderer: Option<Box<dyn TuiRenderer>>,
    /// Whether the hardware cursor shows when positioned, upstream's
    /// optional `showHardwareCursor` constructor argument.
    pub show_hardware_cursor: Option<bool>,
    /// Directory for debug/crash logs, upstream `logDirectory`.
    pub log_directory: Option<PathBuf>,
    /// The key parsing context the debug key matches through, restating the
    /// module-global parser upstream's `matchesKey` used. A session shares
    /// its parser with its terminal; `None` gives this TUI its own.
    pub key_parser: Option<Arc<Mutex<KeyParser>>>,
    /// The image-support probe behind `queryCellSize`'s
    /// `getCapabilities().images` gate. `None` answers false, so no cell-size
    /// query is sent until the image ticket (#51) supplies the real probe.
    pub images_probe: Option<Box<dyn Fn() -> bool>>,
    /// The clock the 16 ms render deadline and query deadlines read,
    /// upstream `performance.now()`. `None` uses `Instant::now`.
    pub clock: Option<Box<dyn Fn() -> Instant>>,
}

impl std::fmt::Debug for TuiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuiConfig")
            .field("has_terminal", &self.terminal.is_some())
            .field("renderer", &self.renderer)
            .field("show_hardware_cursor", &self.show_hardware_cursor)
            .field("log_directory", &self.log_directory)
            .field("key_parser", &self.key_parser.is_some())
            .field("images_probe", &self.images_probe.is_some())
            .field("clock", &self.clock.is_some())
            .finish_non_exhaustive()
    }
}

impl Tui {
    /// Upstream's `TuiBase` constructor; returns the TUI inside the `Rc` the
    /// terminal's handlers and the overlay handles reference.
    ///
    /// # Panics
    ///
    /// Never: a missing terminal or renderer falls back to inert defaults so
    /// construction cannot fail.
    #[must_use]
    pub fn new(config: TuiConfig) -> Rc<Self> {
        let terminal = config.terminal.unwrap_or_else(|| {
            let sink: crate::terminal::WriteSink = Arc::new(Mutex::new(Box::new(|_| {})));
            Box::new(crate::terminal::ProcessTerminal::headless(sink))
        });
        let renderer: Box<dyn TuiRenderer> =
            config.renderer.unwrap_or_else(|| Box::new(InertRenderer));
        let key_parser = config
            .key_parser
            .unwrap_or_else(|| Arc::new(Mutex::new(KeyParser::new())));
        Rc::new_cyclic(|weak| Self {
            self_weak: weak.clone(),
            terminal: RefCell::new(terminal),
            renderer: RefCell::new(renderer),
            container: Container::default(),
            focused_component: RefCell::new(None),
            input_listeners: RefCell::new(Vec::new()),
            listener_next_id: Cell::new(1),
            on_debug: RefCell::new(None),
            demand: Arc::default(),
            render_deadline: Cell::new(None),
            last_render_at: Cell::new(None),
            show_hardware_cursor: Cell::new(config.show_hardware_cursor.unwrap_or(false)),
            clear_on_shrink: Cell::new(false),
            full_redraw_count: Cell::new(0),
            stopped: Cell::new(false),
            scheme_listeners: RefCell::new(Vec::new()),
            scheme_notifications_enabled: Cell::new(false),
            scheme_queries: RefCell::new(Vec::new()),
            pending_osc11_replies: Cell::new(0),
            pending_osc11_queries: RefCell::new(VecDeque::new()),
            log_directory: config.log_directory,
            focus_order_counter: Cell::new(0),
            overlay_next_id: Cell::new(1),
            overlay_stack: RefCell::new(Vec::new()),
            rendered_overlay_layouts: RefCell::new(Vec::new()),
            overlay_focus_restore: RefCell::new(OverlayFocusRestoreState::Inactive),
            key_parser,
            images_probe: config.images_probe.unwrap_or_else(|| Box::new(|| false)),
            clock: config.clock.unwrap_or_else(|| Box::new(Instant::now)),
            pending_input: RefCell::new(VecDeque::new()),
        })
    }

    /// Install the callback for the debug key (Shift+Ctrl+D), called before
    /// input is forwarded to the focused component, upstream's public
    /// `onDebug` field.
    pub fn set_on_debug(&self, on_debug: Option<Rc<dyn Fn()>>) {
        *self.on_debug.borrow_mut() = on_debug;
    }

    /// The thread-safe render request components hold when they render from
    /// off the owner thread (the loader's animation thread): raising the
    /// flag is the only cross-thread surface, and the pump wakes on it.
    #[must_use]
    pub fn render_request(&self) -> RenderRequest {
        let demand = Arc::clone(&self.demand);
        Arc::new(move || {
            demand
                .render_requested
                .store(true, std::sync::atomic::Ordering::Relaxed);
        })
    }

    /// Whether the session has stopped, upstream's `protected stopped`.
    #[must_use]
    pub const fn is_stopped(&self) -> bool {
        self.stopped.get()
    }

    /// Increment the full-redraw counter, upstream's `protected
    /// fullRedrawCount`, which the concrete renderers bump.
    pub fn bump_full_redraws(&self) {
        self.full_redraw_count.set(self.full_redraw_count.get() + 1);
    }

    /// The full-redraw counter, upstream `TUI.fullRedraws`.
    #[must_use]
    pub const fn full_redraws(&self) -> u32 {
        self.full_redraw_count.get()
    }

    /// Upstream `TUI.mode`.
    #[must_use]
    pub fn mode(&self) -> TuiMode {
        self.renderer.borrow().mode()
    }

    /// Whether this TUI carries the `ViewportTUI` capability, upstream
    /// `isViewportTUI`.
    #[must_use]
    pub fn is_viewport_tui(&self) -> bool {
        self.renderer.borrow().is_viewport_tui()
    }

    /// Replace the mounted layout root, upstream
    /// `ViewportTUI.setLayoutRoot`.
    pub fn set_layout_root(&self, component: Option<Rc<dyn Component>>) {
        self.renderer.borrow().set_layout_root(self, component);
    }

    fn clock_now(&self) -> Instant {
        (self.clock)()
    }

    // === Cursor and shrink behavior ===

    /// Whether the hardware cursor shows when positioned, upstream
    /// `getShowHardwareCursor`.
    #[must_use]
    pub const fn get_show_hardware_cursor(&self) -> bool {
        self.show_hardware_cursor.get()
    }

    /// Toggle the hardware cursor, upstream `setShowHardwareCursor`.
    pub fn set_show_hardware_cursor(&self, enabled: bool) {
        if self.show_hardware_cursor.get() == enabled {
            return;
        }
        self.show_hardware_cursor.set(enabled);
        if !enabled {
            self.terminal.borrow_mut().hide_cursor();
        }
        self.request_render(false);
    }

    /// Whether content shrink clears the empty rows, upstream
    /// `getClearOnShrink`.
    #[must_use]
    pub const fn get_clear_on_shrink(&self) -> bool {
        self.clear_on_shrink.get()
    }

    /// Set whether to trigger a full re-render when content shrinks. When
    /// true, empty rows are cleared when content shrinks. When false
    /// (default), empty rows remain, reducing redraws on slower terminals.
    pub fn set_clear_on_shrink(&self, enabled: bool) {
        self.clear_on_shrink.set(enabled);
    }

    // === Focus ===

    /// The component with keyboard focus, upstream `getFocusedComponent`.
    #[must_use]
    pub fn get_focused_component(&self) -> Option<Rc<dyn Component>> {
        self.focused_component.borrow().clone()
    }

    /// Give keyboard focus to a component, or clear it, upstream
    /// `TUI.setFocus`.
    pub fn set_focus(&self, component: Option<Rc<dyn Component>>) {
        self.set_focus_internal(component, FocusRestorePolicy::Clear);
    }

    fn set_focus_internal(
        &self,
        component: Option<Rc<dyn Component>>,
        overlay_focus_restore: FocusRestorePolicy,
    ) {
        let previous_focus = self.focused_component.borrow().clone();
        let mut next_focus = component;
        let previous_focused_overlay = previous_focus.as_ref().and_then(|previous| {
            self.overlay_stack
                .borrow()
                .iter()
                .find(|entry| {
                    Rc::as_ptr(&entry.component).cast::<()>() == Rc::as_ptr(previous).cast::<()>()
                        && self.is_overlay_visible(entry)
                })
                .map(|entry| entry.id)
        });
        let next_focus_is_overlay = next_focus
            .as_ref()
            .is_some_and(|next| self.overlay_stack_has_component(next));
        let restore_state = self.visible_overlay_focus_restore();
        if next_focus.is_some() && !next_focus_is_overlay {
            if let OverlayFocusRestoreState::Blocked {
                overlay,
                blocked_by,
                resume,
            } = &restore_state
                && previous_focus.as_ref().is_some_and(|previous| {
                    Rc::as_ptr(previous).cast::<()>() == Rc::as_ptr(blocked_by).cast::<()>()
                })
            {
                if matches!(resume, OverlayBlockedFocusResume::FocusTarget { .. })
                    || !self.is_component_mounted(blocked_by)
                {
                    next_focus = self.resolve_blocked_overlay_focus_resume(restore_state);
                } else {
                    let placeholder: Rc<dyn Component> = Rc::new(Container::default());
                    *self.overlay_focus_restore.borrow_mut() = OverlayFocusRestoreState::Blocked {
                        overlay: *overlay,
                        blocked_by: next_focus.clone().unwrap_or(placeholder),
                        resume: resume.clone(),
                    };
                }
            } else if let Some(previous_id) = previous_focused_overlay {
                let restore_matches_previous = match &restore_state {
                    OverlayFocusRestoreState::Eligible { overlay }
                    | OverlayFocusRestoreState::Blocked { overlay, .. } => *overlay == previous_id,
                    OverlayFocusRestoreState::Inactive => false,
                };
                if restore_matches_previous
                    && let Some(next) = next_focus.clone()
                    && !self.is_overlay_focus_ancestor(previous_id, &next)
                {
                    *self.overlay_focus_restore.borrow_mut() = OverlayFocusRestoreState::Blocked {
                        overlay: previous_id,
                        blocked_by: next,
                        resume: OverlayBlockedFocusResume::RestoreOverlay,
                    };
                }
            }
        } else if next_focus.is_none() {
            if let OverlayFocusRestoreState::Blocked { blocked_by, .. } = &restore_state
                && previous_focus.as_ref().is_some_and(|previous| {
                    Rc::as_ptr(previous).cast::<()>() == Rc::as_ptr(blocked_by).cast::<()>()
                })
            {
                next_focus = self.resolve_blocked_overlay_focus_resume(restore_state);
            } else if overlay_focus_restore == FocusRestorePolicy::Clear {
                self.clear_overlay_focus_restore();
            }
        }

        let previous = self.focused_component.borrow().clone();
        if let Some(previous) = previous
            && let Some(focusable) = previous.as_focusable()
        {
            focusable.set_focused(false);
        }

        self.focused_component.borrow_mut().clone_from(&next_focus);

        if let Some(next) = &next_focus
            && let Some(focusable) = next.as_focusable()
        {
            focusable.set_focused(true);
        }

        if let Some(next) = &next_focus {
            let focused_overlay = self
                .overlay_stack
                .borrow()
                .iter()
                .find(|entry| {
                    Rc::as_ptr(&entry.component).cast::<()>() == Rc::as_ptr(next).cast::<()>()
                        && self.is_overlay_visible(entry)
                })
                .map(|entry| entry.id);
            if let Some(overlay) = focused_overlay {
                *self.overlay_focus_restore.borrow_mut() =
                    OverlayFocusRestoreState::Eligible { overlay };
            }
        }
    }

    fn clear_overlay_focus_restore(&self) {
        *self.overlay_focus_restore.borrow_mut() = OverlayFocusRestoreState::Inactive;
    }

    fn clear_overlay_focus_restore_for(&self, entry_id: u64) {
        if self
            .overlay_focus_restore
            .borrow()
            .overlay_id()
            .is_some_and(|id| id == entry_id)
        {
            self.clear_overlay_focus_restore();
        }
    }

    fn resolve_blocked_overlay_focus_resume(
        &self,
        restore_state: OverlayFocusRestoreState,
    ) -> Option<Rc<dyn Component>> {
        let OverlayFocusRestoreState::Blocked {
            overlay, resume, ..
        } = restore_state
        else {
            return None;
        };
        match resume {
            OverlayBlockedFocusResume::RestoreOverlay => self
                .overlay_stack
                .borrow()
                .iter()
                .find(|entry| entry.id == overlay)
                .map(|entry| Rc::clone(&entry.component)),
            OverlayBlockedFocusResume::FocusTarget { target } => {
                self.clear_overlay_focus_restore();
                target
            }
        }
    }

    fn visible_overlay_focus_restore(&self) -> OverlayFocusRestoreState {
        let restore_state = self.overlay_focus_restore.borrow().clone();
        let Some(overlay_id) = restore_state.overlay_id() else {
            return OverlayFocusRestoreState::Inactive;
        };
        let still_visible = self
            .overlay_stack
            .borrow()
            .iter()
            .find(|entry| entry.id == overlay_id)
            .is_some_and(|entry| self.is_overlay_visible(entry));
        if still_visible {
            restore_state
        } else {
            OverlayFocusRestoreState::Inactive
        }
    }

    fn is_overlay_focus_ancestor(&self, entry_id: u64, component: &Rc<dyn Component>) -> bool {
        let mut visited: HashSet<*const ()> = HashSet::new();
        let mut current = self
            .overlay_stack
            .borrow()
            .iter()
            .find(|entry| entry.id == entry_id)
            .and_then(|entry| entry.pre_focus.clone());
        while let Some(current_component) = current {
            if !visited.insert(Rc::as_ptr(&current_component).cast::<()>()) {
                break;
            }
            if Rc::as_ptr(&current_component).cast::<()>() == Rc::as_ptr(component).cast::<()>() {
                return true;
            }
            current = self
                .overlay_stack
                .borrow()
                .iter()
                .find(|overlay| {
                    Rc::as_ptr(&overlay.component).cast::<()>()
                        == Rc::as_ptr(&current_component).cast::<()>()
                })
                .and_then(|overlay| overlay.pre_focus.clone());
        }
        false
    }

    fn retarget_overlay_pre_focus(
        &self,
        removed_id: u64,
        removed_component: &Rc<dyn Component>,
        removed_pre_focus: Option<&Rc<dyn Component>>,
    ) {
        for overlay in self.overlay_stack.borrow_mut().iter_mut() {
            if overlay.id != removed_id
                && overlay.pre_focus.as_ref().is_some_and(|pre| {
                    Rc::as_ptr(pre).cast::<()>() == Rc::as_ptr(removed_component).cast::<()>()
                })
            {
                overlay.pre_focus = removed_pre_focus.cloned();
            }
        }
    }

    /// The roots the mounted-component walk consults, upstream
    /// `getMountedRoots`.
    #[must_use]
    pub fn mounted_roots(&self) -> Vec<Rc<dyn Component>> {
        self.renderer.borrow().mounted_roots(self)
    }

    fn is_component_mounted(&self, component: &Rc<dyn Component>) -> bool {
        self.mounted_roots()
            .iter()
            .any(|root| contains_component(root, component))
    }

    // === Overlays ===

    /// Show an overlay component with configurable positioning and sizing,
    /// upstream `TUI.showOverlay`. Returns a handle to control the
    /// overlay's visibility.
    #[must_use]
    pub fn show_overlay(
        &self,
        component: Rc<dyn Component>,
        options: Option<OverlayOptions>,
    ) -> OverlayHandle {
        let non_capturing = options
            .as_ref()
            .is_some_and(|options| options.non_capturing);
        let component_for_focus = Rc::clone(&component);
        let entry_id = self.overlay_next_id.replace(self.overlay_next_id.get() + 1);
        self.overlay_next_id.set(entry_id + 1);
        self.focus_order_counter
            .set(self.focus_order_counter.get() + 1);
        let entry = OverlayStackEntry {
            id: entry_id,
            component,
            options,
            pre_focus: self.focused_component.borrow().clone(),
            hidden: false,
            focus_order: self.focus_order_counter.get(),
            bounds: None,
        };
        self.overlay_stack.borrow_mut().push(entry);
        // Only focus if overlay is actually visible.
        if !non_capturing {
            let visible = self
                .overlay_stack
                .borrow()
                .iter()
                .find(|entry| entry.id == entry_id)
                .is_some_and(|entry| self.is_overlay_visible(entry));
            if visible {
                self.set_focus(Some(component_for_focus));
            }
        }
        self.terminal.borrow_mut().hide_cursor();
        self.request_render(false);

        OverlayHandle {
            tui: self.self_weak.clone(),
            entry_id,
        }
    }

    /// Hide the topmost overlay and restore previous focus, upstream
    /// `TUI.hideOverlay`.
    pub fn hide_overlay(&self) {
        let overlay = self.overlay_stack.borrow_mut().pop();
        let Some(overlay) = overlay else {
            return;
        };
        self.clear_overlay_focus_restore_for(overlay.id);
        self.retarget_overlay_pre_focus(overlay.id, &overlay.component, overlay.pre_focus.as_ref());
        let had_focus = self
            .focused_component
            .borrow()
            .as_ref()
            .is_some_and(|focused| {
                Rc::as_ptr(focused).cast::<()>() == Rc::as_ptr(&overlay.component).cast::<()>()
            });
        if had_focus {
            // Find topmost visible overlay, or fall back to preFocus.
            let top_visible = self.topmost_visible_overlay();
            self.set_focus(top_visible.or_else(|| overlay.pre_focus.clone()));
        }
        if self.overlay_stack.borrow().is_empty() {
            self.terminal.borrow_mut().hide_cursor();
        }
        self.request_render(false);
    }

    /// Check if there are any visible overlays, upstream `TUI.hasOverlay`.
    #[must_use]
    pub fn has_overlay(&self) -> bool {
        self.overlay_stack
            .borrow()
            .iter()
            .any(|entry| self.is_overlay_visible(entry))
    }

    /// Whether any overlay entries exist regardless of visibility, upstream
    /// `TuiBase.hasOverlayEntries`.
    #[must_use]
    pub fn has_overlay_entries(&self) -> bool {
        !self.overlay_stack.borrow().is_empty()
    }

    /// Check if the focused component is a visible overlay, upstream
    /// `TuiBase.isOverlayFocused`.
    #[must_use]
    pub fn is_overlay_focused(&self) -> bool {
        let focused = self.focused_component.borrow().clone();
        focused.as_ref().is_some_and(|focused| {
            self.overlay_stack.borrow().iter().any(|entry| {
                Rc::as_ptr(&entry.component).cast::<()>() == Rc::as_ptr(focused).cast::<()>()
                    && self.is_overlay_visible(entry)
            })
        })
    }

    /// Keep overlay containers as keyboard focus owners when a nested control
    /// is clicked, upstream `TuiBase.resolveMouseFocusTarget`.
    #[must_use]
    pub fn resolve_mouse_focus_target(&self, component: &Rc<dyn Component>) -> Rc<dyn Component> {
        for overlay in self.overlay_stack.borrow().iter().rev() {
            if self.is_overlay_visible(overlay) && contains_component(&overlay.component, component)
            {
                return Rc::clone(&overlay.component);
            }
        }
        Rc::clone(component)
    }

    /// Dispatch to the visually topmost overlay under the pointer, upstream
    /// `TuiBase.dispatchMouseToOverlay`.
    #[must_use]
    pub fn dispatch_mouse_to_overlay(&self, event: &TuiMouseEvent) -> OverlayMouseDispatch {
        let layouts = self.rendered_overlay_layouts.borrow().clone();
        for layout in layouts.iter().rev() {
            if usize::from(event.screen_x) < layout.col
                || usize::from(event.screen_x) >= layout.col + layout.width
                || usize::from(event.screen_y) < layout.row
                || usize::from(event.screen_y) >= layout.row + layout.height
            {
                continue;
            }
            let overlay_event = TuiMouseEvent {
                x: event.screen_x - clamp_cell(layout.col),
                y: event.screen_y - clamp_cell(layout.row),
                width: clamp_cell(layout.width),
                height: clamp_cell(layout.height),
                ..event.clone()
            };
            let result = dispatch_mouse_event(&layout.component, &overlay_event);
            return OverlayMouseDispatch {
                hit: true,
                result: result.map(|result| {
                    if result.focus {
                        TuiMouseEventResult {
                            focus_target: Some(Rc::clone(&layout.component)),
                            ..result
                        }
                    } else {
                        result
                    }
                }),
            };
        }
        OverlayMouseDispatch {
            hit: false,
            result: None,
        }
    }

    fn is_overlay_visible(&self, entry: &OverlayStackEntry) -> bool {
        if entry.hidden {
            return false;
        }
        if let Some(visible) = entry
            .options
            .as_ref()
            .and_then(|options| options.visible.as_ref())
        {
            let (columns, rows) = {
                let terminal = self.terminal.borrow();
                (terminal.columns(), terminal.rows())
            };
            return visible(columns, rows);
        }
        true
    }

    /// Find the visual-frontmost visible capturing overlay, if any.
    fn topmost_visible_overlay(&self) -> Option<Rc<dyn Component>> {
        let mut topmost: Option<(&OverlayStackEntry, u64)> = None;
        let stack = self.overlay_stack.borrow();
        for overlay in stack.iter() {
            let non_capturing = overlay
                .options
                .as_ref()
                .is_some_and(|options| options.non_capturing);
            if non_capturing || !self.is_overlay_visible(overlay) {
                continue;
            }
            if topmost.is_none_or(|(_, order)| overlay.focus_order > order) {
                topmost = Some((overlay, overlay.focus_order));
            }
        }
        topmost.map(|(entry, _)| Rc::clone(&entry.component))
    }

    fn overlay_entry(&self, entry_id: u64) -> Option<OverlayStackEntry> {
        self.overlay_stack
            .borrow()
            .iter()
            .find(|entry| entry.id == entry_id)
            .map(|entry| OverlayStackEntry {
                id: entry.id,
                component: Rc::clone(&entry.component),
                options: None,
                pre_focus: entry.pre_focus.clone(),
                hidden: entry.hidden,
                focus_order: entry.focus_order,
                bounds: entry.bounds,
            })
    }

    // === Overlay handle backends ===

    fn overlay_hide(&self, entry_id: u64) {
        let removed = {
            let mut stack = self.overlay_stack.borrow_mut();
            let index = stack.iter().position(|entry| entry.id == entry_id);
            index.map(|index| stack.remove(index))
        };
        let Some(entry) = removed else {
            return;
        };
        self.clear_overlay_focus_restore_for(entry_id);
        self.retarget_overlay_pre_focus(entry_id, &entry.component, entry.pre_focus.as_ref());
        // Restore focus if this overlay had focus.
        let had_focus = self
            .focused_component
            .borrow()
            .as_ref()
            .is_some_and(|focused| {
                Rc::as_ptr(focused).cast::<()>() == Rc::as_ptr(&entry.component).cast::<()>()
            });
        if had_focus {
            let top_visible = self.topmost_visible_overlay();
            self.set_focus(top_visible.or_else(|| entry.pre_focus.clone()));
        }
        if self.overlay_stack.borrow().is_empty() {
            self.terminal.borrow_mut().hide_cursor();
        }
        self.request_render(false);
    }

    fn overlay_set_hidden(&self, entry_id: u64, hidden: bool) {
        let state = {
            let mut stack = self.overlay_stack.borrow_mut();
            let Some(entry) = stack.iter_mut().find(|entry| entry.id == entry_id) else {
                return;
            };
            if entry.hidden == hidden {
                return;
            }
            entry.hidden = hidden;
            let component = Rc::clone(&entry.component);
            let non_capturing = entry
                .options
                .as_ref()
                .is_some_and(|options| options.non_capturing);
            let pre_focus = entry.pre_focus.clone();
            (component, non_capturing, pre_focus)
        };
        let (component, non_capturing, pre_focus) = state;
        // Update focus when hiding/showing.
        if hidden {
            self.clear_overlay_focus_restore_for(entry_id);
            // If this overlay had focus, move focus to next visible or preFocus.
            let had_focus = self
                .focused_component
                .borrow()
                .as_ref()
                .is_some_and(|focused| {
                    Rc::as_ptr(focused).cast::<()>() == Rc::as_ptr(&component).cast::<()>()
                });
            if had_focus {
                let top_visible = self.topmost_visible_overlay();
                self.set_focus(top_visible.or_else(|| pre_focus.clone()));
            }
        } else {
            // Restore focus to this overlay when showing (if it's actually
            // visible).
            if !non_capturing {
                let visible = self
                    .overlay_stack
                    .borrow()
                    .iter()
                    .find(|entry| entry.id == entry_id)
                    .is_some_and(|entry| self.is_overlay_visible(entry));
                if visible {
                    self.focus_order_counter
                        .set(self.focus_order_counter.get() + 1);
                    if let Some(entry) = self
                        .overlay_stack
                        .borrow_mut()
                        .iter_mut()
                        .find(|entry| entry.id == entry_id)
                    {
                        entry.focus_order = self.focus_order_counter.get();
                    }
                    self.set_focus(Some(Rc::clone(&component)));
                }
            }
        }
        self.request_render(false);
    }

    fn overlay_focus(&self, entry_id: u64) {
        let component = {
            let stack = self.overlay_stack.borrow();
            let Some(entry) = stack.iter().find(|entry| entry.id == entry_id) else {
                return;
            };
            if !self.is_overlay_visible(entry) {
                return;
            }
            Rc::clone(&entry.component)
        };
        self.focus_order_counter
            .set(self.focus_order_counter.get() + 1);
        if let Some(entry) = self
            .overlay_stack
            .borrow_mut()
            .iter_mut()
            .find(|entry| entry.id == entry_id)
        {
            entry.focus_order = self.focus_order_counter.get();
        }
        self.set_focus(Some(component));
        self.request_render(false);
    }

    fn overlay_unfocus(&self, entry_id: u64, unfocus_options: Option<OverlayUnfocusOptions>) {
        let entry = self.overlay_entry(entry_id);
        let Some(entry) = entry else {
            return;
        };
        let is_focused = self
            .focused_component
            .borrow()
            .as_ref()
            .is_some_and(|focused| {
                Rc::as_ptr(focused).cast::<()>() == Rc::as_ptr(&entry.component).cast::<()>()
            });
        let restore_state = self.overlay_focus_restore.borrow().clone();
        let has_pending_restore = restore_state.overlay_id().is_some_and(|id| id == entry_id);
        if !is_focused && !has_pending_restore {
            return;
        }
        if let OverlayFocusRestoreState::Blocked {
            overlay,
            blocked_by,
            ..
        } = &restore_state
            && *overlay == entry_id
            && self
                .focused_component
                .borrow()
                .as_ref()
                .is_some_and(|focused| {
                    Rc::as_ptr(focused).cast::<()>() == Rc::as_ptr(blocked_by).cast::<()>()
                })
        {
            if let Some(unfocus_options) = unfocus_options {
                *self.overlay_focus_restore.borrow_mut() = OverlayFocusRestoreState::Blocked {
                    overlay: entry_id,
                    blocked_by: Rc::clone(blocked_by),
                    resume: OverlayBlockedFocusResume::FocusTarget {
                        target: unfocus_options.target,
                    },
                };
            } else {
                self.clear_overlay_focus_restore();
            }
            self.request_render(false);
            return;
        }
        self.clear_overlay_focus_restore_for(entry_id);
        if is_focused || unfocus_options.is_some() {
            let top_visible = self.topmost_visible_overlay();
            let fallback_target = match &top_visible {
                Some(top)
                    if Rc::as_ptr(top).cast::<()>()
                        != Rc::as_ptr(&entry.component).cast::<()>() =>
                {
                    Some(Rc::clone(top))
                }
                _ => entry.pre_focus.clone(),
            };
            self.set_focus(
                unfocus_options
                    .and_then(|options| options.target)
                    .or(fallback_target),
            );
        }
        self.request_render(false);
    }

    fn overlay_stack_has_component(&self, component: &Rc<dyn Component>) -> bool {
        self.overlay_stack.borrow().iter().any(|entry| {
            Rc::as_ptr(&entry.component).cast::<()>() == Rc::as_ptr(component).cast::<()>()
        })
    }

    // === Session lifecycle ===

    /// Start the session, upstream `TUI.start`: wire the terminal's input
    /// and resize handlers, hide the cursor, emit the color-scheme
    /// notification enable when configured, query the cell size, and
    /// request a render.
    pub fn start(&self) {
        self.stopped.set(false);
        self.renderer.borrow().before_terminal_start();
        self.terminal.borrow_mut().start(
            self.terminal_input_handler(),
            self.terminal_resize_handler(),
        );
        self.renderer.borrow().after_terminal_start();
        self.terminal.borrow_mut().hide_cursor();
        if self.scheme_notifications_enabled.get() {
            self.terminal.borrow_mut().write("\x1b[?2031h");
        }
        self.query_cell_size();
        self.request_render(false);
    }

    /// Stop the session, upstream `TUI.stop`: disarm the scheduler, emit the
    /// color-scheme notification disable when enabled, and restore the
    /// terminal through the renderer's stop hooks.
    pub fn stop(&self, options: TuiStopOptions) {
        self.stopped.set(true);
        self.cancel_render_deadline();
        self.demand
            .render_requested
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.demand
            .immediate_scheduled
            .store(false, std::sync::atomic::Ordering::Relaxed);
        if self.scheme_notifications_enabled.get() {
            self.terminal.borrow_mut().write("\x1b[?2031l");
        }
        self.renderer.borrow().before_terminal_stop(&options);
        self.terminal.borrow_mut().show_cursor();
        self.terminal.borrow_mut().stop();
        self.renderer.borrow().after_terminal_stop(&options);
    }

    fn terminal_input_handler(&self) -> InputHandler {
        let weak = self.self_weak.clone();
        Box::new(move |data: String| {
            let Some(tui) = weak.upgrade() else {
                return;
            };
            // Inside the terminal's own poll window the terminal is already
            // borrowed, so the chunk queues and the pump dispatches it once
            // the borrow is released — the same event-loop turn ordering
            // upstream's single-threaded loop gave. Outside that window the
            // dispatch runs synchronously, exactly upstream's direct call.
            if tui.terminal.try_borrow().is_ok() {
                tui.handle_terminal_input(&data);
            } else {
                tui.pending_input.borrow_mut().push_back(data);
            }
        })
    }

    fn terminal_resize_handler(&self) -> ResizeHandler {
        let weak = self.self_weak.clone();
        Box::new(move || {
            if let Some(tui) = weak.upgrade() {
                // The resize fires inside the terminal's poll window, where
                // only the atomic demand is safe to touch; the pump schedules.
                tui.demand
                    .render_requested
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        })
    }

    // === Input ===

    /// Register an input listener, upstream `TUI.addInputListener`. Returns
    /// the [`ListenerId`] to pass to [`Tui::remove_input_listener`].
    pub fn add_input_listener(&self, listener: TuiInputListener) -> ListenerId {
        let id = self.listener_next_id.get();
        self.listener_next_id.set(id + 1);
        self.input_listeners.borrow_mut().push((id, listener));
        id
    }

    /// Remove an input listener, upstream's returned unsubscribe closure.
    pub fn remove_input_listener(&self, listener_id: ListenerId) {
        self.input_listeners
            .borrow_mut()
            .retain(|(id, _)| *id != listener_id);
    }

    /// Register a color-scheme listener, upstream
    /// `TUI.onTerminalColorSchemeChange`. Returns the [`ListenerId`] to pass
    /// to [`Tui::remove_terminal_color_scheme_listener`].
    pub fn on_terminal_color_scheme_change(
        &self,
        listener: TerminalColorSchemeListener,
    ) -> ListenerId {
        let id = self.listener_next_id.get();
        self.listener_next_id.set(id + 1);
        self.scheme_listeners.borrow_mut().push((id, listener));
        id
    }

    /// Remove a color-scheme listener, upstream's returned unsubscribe
    /// closure.
    pub fn remove_terminal_color_scheme_listener(&self, listener_id: ListenerId) {
        self.scheme_listeners
            .borrow_mut()
            .retain(|(id, _)| *id != listener_id);
    }

    /// Toggle the `CSI ? 2031` color-scheme notifications, upstream
    /// `setTerminalColorSchemeNotifications`. When the session is stopped the
    /// flag only flips; the next start emits the enable sequence.
    pub fn set_terminal_color_scheme_notifications(&self, enabled: bool) {
        if self.scheme_notifications_enabled.get() == enabled {
            return;
        }
        self.scheme_notifications_enabled.set(enabled);
        if !self.stopped.get() {
            self.terminal.borrow_mut().write(if enabled {
                "\x1b[?2031h"
            } else {
                "\x1b[?2031l"
            });
        }
    }

    fn query_cell_size(&self) {
        // Only query if terminal supports images (cell size is only used for
        // image rendering).
        if !(self.images_probe)() {
            return;
        }
        // Query terminal for cell size in pixels: CSI 16 t.
        // Response format: CSI 6 ; height ; width t
        self.terminal.borrow_mut().write("\x1b[16t");
    }

    fn consume_cell_size_response(&self, data: &str) -> bool {
        // Response format: ESC [ 6 ; height ; width t
        let Some(captures) = CELL_SIZE_RESPONSE.captures(data) else {
            return false;
        };
        let (Ok(height_px), Ok(width_px)) =
            (captures[1].parse::<u64>(), captures[2].parse::<u64>())
        else {
            // Upstream's parseInt never fails on the regex-validated digits;
            // a digit run too long for u64 is consumed and dropped.
            return true;
        };
        if height_px == 0 || width_px == 0 {
            return true;
        }

        set_cell_dimensions(CellDimensions {
            width_px: u32::try_from(width_px).unwrap_or(u32::MAX),
            height_px: u32::try_from(height_px).unwrap_or(u32::MAX),
        });
        // Invalidate all components so images re-render with correct
        // dimensions.
        self.invalidate();
        self.request_render(false);
        true
    }

    /// Route one terminal input chunk through the pipeline, upstream
    /// `TuiBase.handleTerminalInput`: terminal replies first, then the
    /// registered input listeners, then dispatch.
    fn handle_terminal_input(&self, data: &str) {
        if self.consume_osc11_background_response(data) {
            return;
        }
        if self.consume_terminal_color_scheme_report(data) {
            return;
        }

        let listeners = self.input_listeners.borrow().clone();
        let mut current = data.to_string();
        if !listeners.is_empty() {
            for (_, listener) in &listeners {
                if let Some(result) = listener(&current) {
                    if result.consume {
                        return;
                    }
                    if let Some(next) = result.data {
                        current = next;
                    }
                }
            }
            if current.is_empty() {
                return;
            }
        }
        self.dispatch_input(&current);
    }

    fn dispatch_input(&self, data: &str) {
        // Consume terminal cell size responses without blocking unrelated
        // input.
        if self.consume_cell_size_response(data) {
            return;
        }

        // Global debug key handler (Shift+Ctrl+D).
        if self.debug_key_matches(data) {
            let on_debug = self.on_debug.borrow().clone();
            if let Some(on_debug) = on_debug {
                on_debug();
                return;
            }
        }

        // If focused component is an overlay, verify it's still visible
        // (visibility can change due to terminal resize or visible()
        // callback).
        let focused = self.focused_component.borrow().clone();
        let focused_overlay_entry = focused.as_ref().and_then(|focused| {
            self.overlay_stack
                .borrow()
                .iter()
                .find(|entry| {
                    Rc::as_ptr(&entry.component).cast::<()>() == Rc::as_ptr(focused).cast::<()>()
                })
                .map(|entry| (entry.id, entry.pre_focus.clone()))
        });
        if let Some((entry_id, pre_focus)) = &focused_overlay_entry {
            let still_visible = self
                .overlay_stack
                .borrow()
                .iter()
                .find(|entry| entry.id == *entry_id)
                .is_some_and(|entry| self.is_overlay_visible(entry));
            if !still_visible {
                // Focused overlay is no longer visible, redirect to topmost
                // visible overlay.
                let top_visible = self.topmost_visible_overlay();
                if let Some(top_visible) = top_visible {
                    self.set_focus(Some(top_visible));
                } else {
                    self.set_focus_internal(pre_focus.clone(), FocusRestorePolicy::Preserve);
                }
            }
        }

        let focus_is_overlay = focused
            .as_ref()
            .is_some_and(|focused| self.overlay_stack_has_component(focused));
        if !focus_is_overlay {
            let restore_state = self.visible_overlay_focus_restore();
            match &restore_state {
                OverlayFocusRestoreState::Eligible { overlay } => {
                    let component = self
                        .overlay_stack
                        .borrow()
                        .iter()
                        .find(|entry| entry.id == *overlay)
                        .map(|entry| Rc::clone(&entry.component));
                    if let Some(component) = component {
                        self.set_focus(Some(component));
                    }
                }
                OverlayFocusRestoreState::Blocked {
                    overlay,
                    blocked_by,
                    resume,
                } => {
                    let blocked_focuses_currently = focused.as_ref().is_some_and(|focused| {
                        Rc::as_ptr(focused).cast::<()>() == Rc::as_ptr(blocked_by).cast::<()>()
                    });
                    if !blocked_focuses_currently {
                        match resume {
                            OverlayBlockedFocusResume::RestoreOverlay => {
                                let component = self
                                    .overlay_stack
                                    .borrow()
                                    .iter()
                                    .find(|entry| entry.id == *overlay)
                                    .map(|entry| Rc::clone(&entry.component));
                                if let Some(component) = component {
                                    self.set_focus(Some(component));
                                }
                            }
                            OverlayBlockedFocusResume::FocusTarget { target } => {
                                self.clear_overlay_focus_restore();
                                self.set_focus(target.clone());
                            }
                        }
                    }
                }
                OverlayFocusRestoreState::Inactive => {}
            }
        }

        // Pass input to focused component (including Ctrl+C).
        // The focused component can decide how to handle Ctrl+C. Read the
        // field fresh: the redirect above may have moved focus.
        let focused = self.focused_component.borrow().clone();
        if let Some(focused) = &focused {
            // Filter out key release events unless component opts in.
            if focused.wants_input() {
                if is_key_release(data) && !focused.wants_key_release() {
                    return;
                }
                focused.handle_input(data);
                // Keyboard input is latency-sensitive. Avoid the throttled
                // timer path, where even setTimeout(0) can take a full 16 ms
                // tick on Windows.
                self.request_immediate_render();
            }
        }
    }

    fn debug_key_matches(&self, data: &str) -> bool {
        self.key_parser
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .matches_key(data, "shift+ctrl+d")
    }

    fn consume_osc11_background_response(&self, data: &str) -> bool {
        if self.pending_osc11_replies.get() == 0 {
            return false;
        }
        if !is_osc11_background_color_response(data) {
            return false;
        }

        let rgb = parse_osc11_background_color(data);
        self.pending_osc11_replies
            .set(self.pending_osc11_replies.get() - 1);
        if let Some(query) = self.pending_osc11_queries.borrow_mut().pop_front()
            && !query.settled
            && let Some(resolve) = query.resolve
        {
            let _ = resolve.send(rgb);
        }
        true
    }

    fn consume_terminal_color_scheme_report(&self, data: &str) -> bool {
        let Some(scheme) = parse_terminal_color_scheme_report(data) else {
            return false;
        };

        let listeners = self.scheme_listeners.borrow().clone();
        for (_, listener) in &listeners {
            listener(&scheme);
        }
        true
    }

    /// Query the terminal's default background color with OSC 11
    /// (`ESC ] 11 ; ? BEL`), upstream `queryTerminalBackgroundColor`. The
    /// parsed RGB color, or `None` on timeout or parse failure, resolves the
    /// returned channel on [`Tui::poll`].
    #[must_use]
    pub fn query_terminal_background_color(
        &self,
        timeout_ms: u64,
    ) -> std::sync::mpsc::Receiver<Option<RgbColor>> {
        let (resolve, receiver) = std::sync::mpsc::channel();
        let query = PendingOsc11BackgroundQuery {
            settled: false,
            resolve: Some(resolve),
            deadline: Some(self.clock_now() + Duration::from_millis(timeout_ms)),
        };
        self.pending_osc11_queries.borrow_mut().push_back(query);
        self.pending_osc11_replies
            .set(self.pending_osc11_replies.get() + 1);
        self.terminal.borrow_mut().write("\x1b]11;?\x07");
        receiver
    }

    /// Query the terminal's color-scheme preference with DSR
    /// (`CSI ? 996 n`). Terminals that support the color palette
    /// notification protocol reply with `CSI ? 997 ; 1 n` for dark or
    /// `CSI ? 997 ; 2 n` for light.
    #[must_use]
    pub fn query_terminal_color_scheme(
        &self,
        timeout_ms: u64,
    ) -> std::sync::mpsc::Receiver<Option<TerminalColorScheme>> {
        let (settle, receiver) = std::sync::mpsc::channel();
        let settled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let listener = {
            let settled = Arc::clone(&settled);
            move |scheme: &TerminalColorScheme| {
                if !settled.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    let _ = settle.send(Some(*scheme));
                }
            }
        };
        let listener_id = self.on_terminal_color_scheme_change(Rc::new(listener));
        self.scheme_queries.borrow_mut().push(SchemeQuery {
            listener_id,
            settled: false,
            deadline: self.clock_now() + Duration::from_millis(timeout_ms),
        });
        self.terminal.borrow_mut().write("\x1b[?996n");
        receiver
    }

    // === Render scheduling ===

    /// Render immediately, upstream `TUI.renderNow`. A forced render resets
    /// the renderer's cached state first.
    pub fn render_now(&self, force: bool) {
        if force {
            self.renderer.borrow().reset_render_state();
        }
        self.demand
            .render_requested
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.cancel_render_deadline();
        self.last_render_at.set(Some(self.clock_now()));
        self.do_render();
    }

    /// Coalesce a render request, upstream `TUI.requestRender`. A forced
    /// request resets the renderer's cache and takes the immediate path;
    /// otherwise the pump schedules within the 16 ms interval. Forced
    /// renders touch renderer state and stay owner-thread, as every
    /// upstream call ran on the main loop.
    pub fn request_render(&self, force: bool) {
        if force {
            self.renderer.borrow().reset_render_state();
            self.request_immediate_render();
            return;
        }
        if self
            .demand
            .render_requested
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {}
        // Upstream deferred `scheduleRender` onto `process.nextTick`; the
        // pump is that turn.
    }

    fn request_immediate_render(&self) {
        // A previously queued throttled frame must not survive user input;
        // the immediate path cancels the armed deadline, the behavior
        // documented at tui.ts:971-972.
        self.cancel_render_deadline();
        self.demand
            .render_requested
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if self
            .demand
            .immediate_scheduled
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {}
        // The pump runs the scheduled immediate callback.
    }

    fn cancel_render_deadline(&self) {
        self.render_deadline.set(None);
    }

    /// The owner's event-loop turn, restating the `process.nextTick`
    /// immediate-render preemption and the 16 ms throttle: pump the
    /// terminal, dispatch queued input, then run the scheduler.
    pub fn poll(&self, timeout: Duration) {
        let demand_pending = self
            .demand
            .render_requested
            .load(std::sync::atomic::Ordering::Relaxed)
            || self
                .demand
                .immediate_scheduled
                .load(std::sync::atomic::Ordering::Relaxed);
        let wait = if demand_pending {
            Duration::ZERO
        } else {
            self.render_deadline.get().map_or(timeout, |deadline| {
                deadline
                    .saturating_duration_since(self.clock_now())
                    .min(timeout)
            })
        };
        self.terminal.borrow_mut().poll(wait);

        while let Some(data) = self.pending_input.borrow_mut().pop_front() {
            self.handle_terminal_input(&data);
        }

        self.run_scheduler();
    }

    fn run_scheduler(&self) {
        let now = self.clock_now();
        self.fire_query_deadlines(now);

        // The immediate path, upstream's `requestImmediateRender` nextTick:
        // user input preempts a queued throttled frame.
        if self
            .demand
            .immediate_scheduled
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            let requested = self
                .demand
                .render_requested
                .swap(false, std::sync::atomic::Ordering::Relaxed);
            if !self.stopped.get() && requested {
                self.cancel_render_deadline();
                self.last_render_at.set(Some(self.clock_now()));
                self.do_render();
            }
            return;
        }

        // The throttled path, upstream's `scheduleRender` timer.
        if let Some(at) = self.render_deadline.get() {
            let now = self.clock_now();
            if now < at {
                return;
            }
            self.render_deadline.set(None);
            let requested = self
                .demand
                .render_requested
                .swap(false, std::sync::atomic::Ordering::Relaxed);
            if !requested {
                return;
            }
            self.last_render_at.set(Some(now));
            self.do_render();
            self.rearm_after_render();
            return;
        }

        // No timer armed yet: this pump is the nextTick that arms it.
        if !self
            .demand
            .render_requested
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        let delay = self.throttle_delay();
        if delay == 0 {
            // Upstream armed `setTimeout(0)`, which the event loop fires on
            // its next turn; the pump is that turn, so a zero delay renders
            // in this pass.
            self.demand
                .render_requested
                .store(false, std::sync::atomic::Ordering::Relaxed);
            self.last_render_at.set(Some(self.clock_now()));
            self.do_render();
            self.rearm_after_render();
        } else {
            self.render_deadline
                .set(Some(self.clock_now() + Duration::from_millis(delay)));
        }
    }

    fn fire_query_deadlines(&self, now: Instant) {
        self.fire_osc11_deadlines(now);
        self.fire_scheme_deadlines(now);
    }

    fn fire_osc11_deadlines(&self, now: Instant) {
        let due = self
            .pending_osc11_queries
            .borrow_mut()
            .iter_mut()
            .filter(|query| !query.settled)
            .filter(|query| query.deadline.is_some_and(|deadline| now >= deadline))
            .map(|query| {
                query.settled = true;
                query.deadline = None;
                query.resolve.take()
            })
            .collect::<Vec<_>>();
        for resolve in due.into_iter().flatten() {
            let _ = resolve.send(None);
        }
    }

    fn fire_scheme_deadlines(&self, now: Instant) {
        let expired: Vec<ListenerId> = {
            let mut queries = self.scheme_queries.borrow_mut();
            let mut expired = Vec::new();
            for query in queries.iter_mut() {
                if !query.settled && now >= query.deadline {
                    query.settled = true;
                    expired.push(query.listener_id);
                }
            }
            expired
        };
        for listener_id in expired {
            self.remove_terminal_color_scheme_listener(listener_id);
        }
    }

    fn throttle_delay(&self) -> u64 {
        self.last_render_at.get().map_or(0, |last| {
            MIN_RENDER_INTERVAL_MS.saturating_sub(millis_since(self.clock_now(), last))
        })
    }

    fn rearm_after_render(&self) {
        if !self
            .demand
            .render_requested
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        // Upstream re-arms `scheduleRender` after a render that raised new
        // requests; the fresh lastRenderAt makes the delay a full interval.
        let delay = self.throttle_delay();
        self.render_deadline
            .set(Some(self.clock_now() + Duration::from_millis(delay)));
    }

    fn do_render(&self) {
        if self.stopped.get() {
            return;
        }
        self.renderer.borrow().do_render(self);
    }

    // === Renderer contract: the `TuiBase` protected members ===

    /// Upstream `TUI.addChild`.
    pub fn add_child(&self, component: Rc<dyn Component>) {
        self.container.add_child(component);
    }

    /// Upstream `TUI.removeChild`.
    pub fn remove_child(&self, component: &Rc<dyn Component>) {
        self.container.remove_child(component);
    }

    /// Upstream `TUI.clear`.
    pub fn clear(&self) {
        self.container.clear();
    }

    /// Render the mounted children, upstream's `this.render(width)` inside
    /// `doRender`.
    #[must_use]
    pub fn render_children(&self, width: usize) -> Vec<String> {
        Component::render(self, width)
    }

    /// Composite all overlays into content lines, sorted by focus order with
    /// higher on top, upstream `TuiBase.compositeOverlays`.
    #[must_use]
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's compositeOverlays block for block; splitting it would break the 1:1 correspondence"
    )]
    pub fn composite_overlays(
        &self,
        lines: Vec<String>,
        term_width: usize,
        term_height: usize,
    ) -> Vec<String> {
        if self.overlay_stack.borrow().is_empty() {
            self.rendered_overlay_layouts.borrow_mut().clear();
            return lines;
        }
        let mut result = lines;

        for entry in self.overlay_stack.borrow_mut().iter_mut() {
            entry.bounds = None;
        }

        // Pre-render all visible overlays and calculate positions.
        let mut rendered: Vec<RenderedOverlay> = Vec::new();
        let mut min_lines_needed = result.len();

        let visible_entries: Vec<(u64, u64)> = self
            .overlay_stack
            .borrow()
            .iter()
            .filter(|entry| self.is_overlay_visible(entry))
            .map(|entry| (entry.id, entry.focus_order))
            .collect();
        let mut ordered: Vec<&(u64, u64)> = visible_entries.iter().collect();
        // Upstream sorts by focusOrder alone; the id is identity, not order.
        ordered.sort_by_key(|(_, order)| *order);
        for (entry_id, _) in ordered {
            let (component, width, placed, lines) = {
                let stack = self.overlay_stack.borrow();
                let Some(entry) = stack.iter().find(|entry| entry.id == *entry_id) else {
                    continue;
                };
                let options = entry.options.as_ref();

                // Get the layout with height 0 first to determine width and
                // maxHeight (they don't depend on overlay height).
                let layout = resolve_overlay_layout(options, 0, term_width, term_height);

                // Render the component at the calculated width.
                let mut overlay_lines = entry.component.render(layout.width);

                // Apply maxHeight if specified.
                if let Some(max_height) = layout.max_height
                    && overlay_lines.len() > max_height
                {
                    overlay_lines.truncate(max_height);
                }

                // Get the final row/col with the actual overlay height.
                let placed =
                    resolve_overlay_layout(options, overlay_lines.len(), term_width, term_height);
                (
                    Rc::clone(&entry.component),
                    layout.width,
                    placed,
                    overlay_lines,
                )
            };
            let bounds = OverlayBounds {
                row: placed.row,
                col: placed.col,
                width,
                height: lines.len(),
            };
            if let Some(entry) = self
                .overlay_stack
                .borrow_mut()
                .iter_mut()
                .find(|entry| entry.id == *entry_id)
            {
                entry.bounds = Some(bounds);
            }
            min_lines_needed = min_lines_needed.max(placed.row + bounds.height);
            rendered.push(RenderedOverlay {
                entry_id: *entry_id,
                component,
                lines,
                row: placed.row,
                col: placed.col,
                width,
            });
        }
        *self.rendered_overlay_layouts.borrow_mut() = rendered
            .iter()
            .map(|overlay| RenderedOverlayLayout {
                entry_id: overlay.entry_id,
                component: Rc::clone(&overlay.component),
                row: overlay.row,
                col: overlay.col,
                width: overlay.width,
                height: overlay.lines.len(),
            })
            .collect();

        // Pad to at least terminal height so overlays have screen-relative
        // positions. Excludes maxLinesRendered: the historical high-water
        // mark caused self-reinforcing inflation that pushed content into
        // scrollback on terminal widen.
        let working_height = result.len().max(term_height).max(min_lines_needed);

        // Extend with empty lines if content is too short for overlay
        // placement or working area.
        while result.len() < working_height {
            result.push(String::new());
        }

        let viewport_start = working_height.saturating_sub(term_height);

        // Composite each overlay.
        for overlay in &rendered {
            for (i, overlay_line) in overlay.lines.iter().enumerate() {
                let idx = viewport_start + overlay.row + i;
                if idx < result.len() {
                    // Defensive: truncate the overlay line to the declared
                    // width before compositing (components should already
                    // respect the width, but this ensures it).
                    let truncated_overlay_line = if visible_width(overlay_line) > overlay.width {
                        slice_by_column(overlay_line, 0, overlay.width, true)
                    } else {
                        overlay_line.clone()
                    };
                    result[idx] = composite_tui_line(
                        &result[idx],
                        &truncated_overlay_line,
                        overlay.col,
                        overlay.width,
                        term_width,
                    );
                }
            }
        }

        result
    }

    /// Append the terminal-output normalization and the segment reset to
    /// every line, upstream `TuiBase.applyLineResets`. Image lines stay
    /// untouched.
    #[must_use]
    pub fn apply_line_resets(&self, mut lines: Vec<String>) -> Vec<String> {
        for line in &mut lines {
            if !is_image_line(line) {
                let normalized = normalize_terminal_output(line);
                *line = format!("{normalized}{SEGMENT_RESET}");
            }
        }
        lines
    }

    /// Find and extract the cursor position from rendered lines, upstream
    /// `TuiBase.extractCursorPosition`: scan the bottom `height` lines (the
    /// visible viewport) for [`CURSOR_MARKER`], return its row and visual
    /// column, and strip the marker from the output.
    pub fn extract_cursor_position(
        &self,
        lines: &mut [String],
        height: usize,
    ) -> Option<(usize, usize)> {
        // Only scan the bottom `height` lines (visible viewport).
        let viewport_top = lines.len().saturating_sub(height);
        for row in (viewport_top..lines.len()).rev() {
            let line = &lines[row];
            let Some(marker_index) = line.find(CURSOR_MARKER) else {
                continue;
            };
            // Calculate visual column (width of text before marker).
            let col = visible_width(&line[..marker_index]);

            // Strip the marker from the line.
            let marker_len = CURSOR_MARKER.len();
            lines[row] = format!(
                "{}{}",
                &line[..marker_index],
                &line[marker_index + marker_len..]
            );

            return Some((row, col));
        }
        None
    }

    // === Terminal seams for the renderer contract ===

    /// Write to the terminal, upstream's `this.terminal.write` inside the
    /// renderer's `doRender` and stop hooks.
    pub fn terminal_write(&self, data: &str) {
        self.terminal.borrow_mut().write(data);
    }

    /// Terminal columns, upstream's `this.terminal.columns`.
    #[must_use]
    pub fn terminal_columns(&self) -> u16 {
        self.terminal.borrow().columns()
    }

    /// Terminal rows, upstream's `this.terminal.rows`.
    #[must_use]
    pub fn terminal_rows(&self) -> u16 {
        self.terminal.borrow().rows()
    }

    /// Hide the hardware cursor, upstream's `this.terminal.hideCursor`.
    pub fn terminal_hide_cursor(&self) {
        self.terminal.borrow_mut().hide_cursor();
    }

    /// Show the hardware cursor, upstream's `this.terminal.showCursor`.
    pub fn terminal_show_cursor(&self) {
        self.terminal.borrow_mut().show_cursor();
    }

    /// Move the hardware cursor by lines, upstream's `this.terminal.moveBy`.
    pub fn terminal_move_by(&self, lines: i32) {
        self.terminal.borrow_mut().move_by(lines);
    }
}

impl Component for Tui {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.container.handle_mouse(event)
    }

    fn invalidate(&self) {
        for root in self.mounted_roots() {
            root.invalidate();
        }
        for overlay in self.overlay_stack.borrow().iter() {
            overlay.component.invalidate();
        }
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }
}

/// The result of [`Tui::dispatch_mouse_to_overlay`], upstream's
/// `{ hit: boolean; result?: TuiMouseDispatchResult }` shape.
#[derive(Debug, Clone)]
pub struct OverlayMouseDispatch {
    /// Whether any rendered overlay's rectangle contained the pointer.
    pub hit: bool,
    /// The dispatch result when the topmost containing overlay handled the
    /// event.
    pub result: Option<TuiMouseEventResult>,
}

/// A rendered overlay awaiting compositing.
struct RenderedOverlay {
    entry_id: u64,
    component: Rc<dyn Component>,
    lines: Vec<String>,
    row: usize,
    col: usize,
    width: usize,
}

/// A resolved overlay position, upstream's `resolveOverlayLayout` return.
struct OverlayLayout {
    width: usize,
    row: usize,
    col: usize,
    max_height: Option<usize>,
}

/// Resolve overlay layout from options, upstream
/// `TuiBase.resolveOverlayLayout`: width, row, col, and the clamped
/// maxHeight for rendering.
fn resolve_overlay_layout(
    options: Option<&OverlayOptions>,
    overlay_height: usize,
    term_width: usize,
    term_height: usize,
) -> OverlayLayout {
    let Some(options) = options else {
        // Upstream's `opt = {}` default: 80 columns capped to the
        // terminal, centered.
        let width = 80.min(term_width.max(1));
        return OverlayLayout {
            width,
            row: resolve_anchor_row(OverlayAnchor::Center, overlay_height, term_height.max(1), 0),
            col: resolve_anchor_col(OverlayAnchor::Center, width, term_width.max(1), 0),
            max_height: None,
        };
    };

    // Parse margin (clamp to non-negative).
    let margin = match options.margin {
        Some(OverlayMargin::All(all)) => OverlayMarginSides {
            top: all,
            right: all,
            bottom: all,
            left: all,
        },
        Some(OverlayMargin::Sides(sides)) => sides,
        None => OverlayMarginSides::default(),
    };
    let margin_top = margin_cells(margin.top);
    let margin_right = margin_cells(margin.right);
    let margin_bottom = margin_cells(margin.bottom);
    let margin_left = margin_cells(margin.left);

    // Available space after margins.
    let avail_width = term_width.saturating_sub(margin_left + margin_right).max(1);
    let avail_height = term_height
        .saturating_sub(margin_top + margin_bottom)
        .max(1);

    // === Resolve width ===
    let mut width =
        parse_size_value(options.width, term_width).unwrap_or_else(|| 80.min(avail_width));
    // Apply minWidth.
    if let Some(min_width) = options.min_width {
        width = width.max(usize::from(u16::try_from(min_width).unwrap_or(u16::MAX)));
    }
    // Clamp to available space.
    width = width.clamp(1, avail_width);

    // === Resolve maxHeight ===
    let max_height = parse_size_value(options.max_height, term_height)
        .map(|max_height| max_height.clamp(1, avail_height));

    // Effective overlay height (may be clamped by maxHeight).
    let effective_height =
        max_height.map_or(overlay_height, |max_height| overlay_height.min(max_height));

    // === Resolve position ===
    let row = match options.row {
        Some(SizeValue::Percent(percent)) => {
            // 0% = top, 100% = bottom (overlay stays within bounds).
            let max_row = avail_height.saturating_sub(effective_height);
            margin_top + percent_of(percent, max_row)
        }
        Some(SizeValue::Cells(cells)) => cells_usize(cells),
        None => resolve_anchor_row(
            options.anchor.unwrap_or(OverlayAnchor::Center),
            effective_height,
            avail_height,
            margin_top,
        ),
    };

    let col = match options.col {
        Some(SizeValue::Percent(percent)) => {
            // 0% = left, 100% = right (overlay stays within bounds).
            let max_col = avail_width.saturating_sub(width);
            margin_left + percent_of(percent, max_col)
        }
        Some(SizeValue::Cells(cells)) => cells_usize(cells),
        None => resolve_anchor_col(
            options.anchor.unwrap_or(OverlayAnchor::Center),
            width,
            avail_width,
            margin_left,
        ),
    };

    // Apply offsets, then clamp to terminal bounds (respecting margins).
    // The clamp keeps the values at or above the non-negative margins, so
    // the usize conversions never see a negative or out-of-range value.
    let row = usize::try_from(clamp_to_span(
        span_i64(row) + options.offset_y.unwrap_or(0),
        span_i64(margin_top),
        span_i64(term_height) - span_i64(margin_bottom) - span_i64(effective_height),
    ))
    .unwrap_or_default();
    let col = usize::try_from(clamp_to_span(
        span_i64(col) + options.offset_x.unwrap_or(0),
        span_i64(margin_left),
        span_i64(term_width) - span_i64(margin_right) - span_i64(width),
    ))
    .unwrap_or_default();

    OverlayLayout {
        width,
        row,
        col,
        max_height,
    }
}

/// Upstream `TuiBase.MIN_RENDER_INTERVAL_MS`.
const MIN_RENDER_INTERVAL_MS: u64 = 16;

static CELL_SIZE_RESPONSE: LazyLock<regex::Regex> =
    LazyLock::new(|| crate::utils::static_regex(r"^\x1b\[6;(\d+);(\d+)t$"));

/// Milliseconds between an earlier instant and now, saturating at u64:
/// the upstream setTimeout budget truncates to whole milliseconds.
fn millis_since(now: Instant, earlier: Instant) -> u64 {
    u64::try_from(now.saturating_duration_since(earlier).as_millis()).unwrap_or(u64::MAX)
}

/// Parse a `SizeValue` into absolute cells given the reference size,
/// upstream `parseSizeValue`. A negative cell count clamps to zero,
/// upstream's later `Math.max(1, ...)`.
fn parse_size_value(value: Option<SizeValue>, reference: usize) -> Option<usize> {
    match value {
        None => None,
        Some(SizeValue::Cells(cells)) => Some(cells_usize(cells)),
        Some(SizeValue::Percent(percent)) => Some(percent_of(percent, reference)),
    }
}

/// A percentage of the reference dimension, floored, upstream's
/// `Math.floor((referenceSize * pct) / 100)`. The reference is a terminal
/// dimension (u16), so the f64 mantissa never loses precision.
fn percent_of(percent: f64, reference: usize) -> usize {
    let reference = f64::from(u16::try_from(reference).unwrap_or(u16::MAX));
    // The percentage is validated non-negative and the reference is a
    // terminal dimension (u16), so the floor is a small non-negative float.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the floored product of a non-negative percentage and a u16 reference is a small non-negative integer"
    )]
    let cells = (percent * reference / 100.0).floor() as usize;
    cells
}

/// A side margin clamped non-negative, upstream's `Math.max(0, ...)`;
/// terminal-relative margins ride terminal dimensions.
fn margin_cells(value: i64) -> usize {
    cells_usize(value)
}

/// Whether a component tree rooted at `root` contains `target`, upstream's
/// `TuiBase.containsComponent`: pointer identity, then a recursive walk
/// through container-shaped children.
fn contains_component(root: &Rc<dyn Component>, target: &Rc<dyn Component>) -> bool {
    if Rc::as_ptr(root).cast::<()>() == Rc::as_ptr(target).cast::<()>() {
        return true;
    }
    root.children()
        .iter()
        .any(|child| contains_component(child, target))
}

/// i64 into terminal-relative usize: the values ride terminal dimensions,
/// so the saturating fallback bounds pathological inputs.
fn cells_usize(value: i64) -> usize {
    usize::try_from(value.max(0)).unwrap_or(usize::MAX)
}

/// usize into the layout's i64 math: terminal dimensions are u16, so the
/// saturating fallback bounds pathological inputs.
#[expect(
    clippy::cast_possible_wrap,
    reason = "the value rides a terminal dimension (u16); the explicit i64::MAX bound makes the wrap unreachable"
)]
const fn span_i64(value: usize) -> i64 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "i64::MAX truncated to a 32-bit usize is the bound check's upper input; the value rides terminal dimensions (u16), so the branch is unreachable"
    )]
    const I64_MAX_USIZE: usize = i64::MAX as usize;
    if value > I64_MAX_USIZE {
        return i64::MAX;
    }
    value as i64
}

/// Clamp to the placement span, restating upstream's
/// `Math.max(low, Math.min(value, high))`: when the upper bound sits below
/// the lower one, the low bound wins.
fn clamp_to_span(value: i64, low: i64, high: i64) -> i64 {
    if high < low {
        return low;
    }
    value.clamp(low, high)
}

/// Resolve the anchor row, upstream `resolveAnchorRow`.
const fn resolve_anchor_row(
    anchor: OverlayAnchor,
    height: usize,
    avail_height: usize,
    margin_top: usize,
) -> usize {
    match anchor {
        OverlayAnchor::TopLeft | OverlayAnchor::TopCenter | OverlayAnchor::TopRight => margin_top,
        OverlayAnchor::BottomLeft | OverlayAnchor::BottomCenter | OverlayAnchor::BottomRight => {
            margin_top + avail_height.saturating_sub(height)
        }
        OverlayAnchor::LeftCenter | OverlayAnchor::Center | OverlayAnchor::RightCenter => {
            margin_top + (avail_height.saturating_sub(height)) / 2
        }
    }
}

/// Resolve the anchor column, upstream `resolveAnchorCol`.
const fn resolve_anchor_col(
    anchor: OverlayAnchor,
    width: usize,
    avail_width: usize,
    margin_left: usize,
) -> usize {
    match anchor {
        OverlayAnchor::TopLeft | OverlayAnchor::LeftCenter | OverlayAnchor::BottomLeft => {
            margin_left
        }
        OverlayAnchor::TopRight | OverlayAnchor::RightCenter | OverlayAnchor::BottomRight => {
            margin_left + avail_width.saturating_sub(width)
        }
        OverlayAnchor::TopCenter | OverlayAnchor::Center | OverlayAnchor::BottomCenter => {
            margin_left + (avail_width.saturating_sub(width)) / 2
        }
    }
}

/// The no-render stand-in a [`Tui`] falls back to when constructed without a
/// renderer, so `Tui::new` cannot fail.
#[derive(Debug)]
struct InertRenderer;

impl TuiRenderer for InertRenderer {
    fn mode(&self) -> TuiMode {
        TuiMode::Regular
    }

    fn do_render(&self, _tui: &Tui) {}
}

/// The render-request seam [`crate::components::Loader`] holds, upstream
/// `TUI.requestRender`: callable from the loader's animation thread, where it
/// only raises the scheduler's demand flag.
pub type RenderRequest = Arc<dyn Fn() + Send + Sync>;

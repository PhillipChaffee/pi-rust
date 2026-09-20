//! The layout-node capability of `packages/tui/src/layout-node.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#44).
//!
//! It carries the sealed node kinds the constrained layout engine walks.
//!
//! Restatements (survey flags in map ticket "Survey the tui package"):
//!
//! - `Symbol.for("@earendil-works/pi-tui/layout-node")` and the duck-typed
//!   `LayoutComponent` check become [`Component::layout_node`], a defaulted
//!   trait method answering `None` for components outside the engine
//!   (survey flag 1, the same shape the `Focusable` query uses).
//! - The stack entry's `visible` closure rides an `Rc` instead of a `Box`
//!   so an entry stays `Clone`: the stacks hold their entries in a
//!   `RefCell` and the layout node hands a snapshot to the engine per
//!   frame.
//! - `ScrollLayoutState` becomes a trait the scroll view's shared state
//!   implements; the layout engine sees it as an `Arc` handle, which also
//!   carries the box identity `getScrollViewBox` compares.
//! - `SizeValue`-shaped basis stays `auto` or a cell count
//!   ([`Basis`]); sizes are `usize` with `usize::MAX` restating
//!   `Number.MAX_SAFE_INTEGER`.

use std::rc::Rc;
use std::sync::Arc;

use crate::components::ColorFn;
use crate::components::scroll_view::ScrollViewScrollbar;
use crate::tui::{Component, RenderRequest};

/// How a stack member participates in sizing, upstream `basis?: number |
/// "auto"`: an absolute cell count, or `auto` for the component's intrinsic
/// measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// Upstream `"auto"` — measure from the rendered content.
    Auto,
    /// An absolute cell count, upstream's number spelling.
    Cells(i64),
}

/// Cross-axis alignment, upstream `align: "stretch" | "start" | "center" | "end"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StackAlign {
    /// Upstream `"stretch"` — the default.
    #[default]
    Stretch,
    /// Upstream `"start"`.
    Start,
    /// Upstream `"center"`.
    Center,
    /// Upstream `"end"`.
    End,
}

/// Stack axis, upstream `type: "vstack" | "hstack"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackKind {
    /// Upstream `"vstack"`.
    Vertical,
    /// Upstream `"hstack"`.
    Horizontal,
}

/// Scroll chaining behavior at the content edges, upstream
/// `overscroll: "chain" | "contain"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Overscroll {
    /// Upstream `"chain"` — leftover scroll delta propagates outward.
    #[default]
    Chain,
    /// Upstream `"contain"` — leftover delta is absorbed.
    Contain,
}

/// The viewport the layout engine measures against, upstream
/// `LayoutViewport`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutViewport {
    /// Viewport width in columns.
    pub width: usize,
    /// Viewport height in rows.
    pub height: usize,
}

/// One stack member with its sizing options, upstream
/// `StackLayoutEntry`. The `visible` closure rides an `Rc` so the entry
/// stays `Clone` for the per-frame snapshot the engine walks.
#[derive(Clone)]
pub struct StackLayoutEntry {
    /// The member component.
    pub component: Rc<dyn Component>,
    /// Sizing basis, upstream `basis?: number | "auto"`; `None` is auto.
    pub basis: Option<Basis>,
    /// Weight for growing into free space, upstream `grow` (default 0).
    pub grow: u32,
    /// Weight for shrinking under pressure, upstream `shrink` (default 1).
    pub shrink: u32,
    /// Lower size bound in cells, upstream `minSize` (default 0).
    pub min_size: usize,
    /// Upper size bound in cells, upstream `maxSize`
    /// (`usize::MAX` restating `Number.MAX_SAFE_INTEGER`).
    pub max_size: usize,
    /// Visibility gate, upstream `visible?: (viewport) => boolean`.
    pub visible: Option<EntryVisibility>,
}

/// The entry visibility gate, upstream `visible?: (viewport) => boolean`,
/// shared with the stacks' public options.
pub type EntryVisibility = Rc<dyn Fn(&LayoutViewport) -> bool>;

impl std::fmt::Debug for StackLayoutEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StackLayoutEntry")
            .field("component", &"...")
            .field("basis", &self.basis)
            .field("grow", &self.grow)
            .field("shrink", &self.shrink)
            .field("min_size", &self.min_size)
            .field("max_size", &self.max_size)
            .field("visible", &self.visible.is_some())
            .finish()
    }
}

/// A stack's layout node, upstream `StackLayoutNode`.
#[derive(Clone)]
pub struct StackLayoutNode {
    /// The stack axis.
    pub kind: StackKind,
    /// The visible-entry snapshot; the engine re-filters by `visible`.
    pub entries: Vec<StackLayoutEntry>,
    /// Blank rows or columns between members.
    pub gap: usize,
    /// Cross-axis alignment.
    pub align: StackAlign,
}

impl std::fmt::Debug for StackLayoutNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StackLayoutNode")
            .field("kind", &self.kind)
            .field("entries", &self.entries.len())
            .field("gap", &self.gap)
            .field("align", &self.align)
            .finish()
    }
}

/// The scroll state the layout engine drives, upstream
/// `ScrollLayoutState`. The scroll view's shared state implements it; the
/// engine reaches it as an [`Arc`] handle that also carries the box
/// identity.
pub trait ScrollLayoutState {
    /// The current scroll offset in rows, upstream `scrollTop`.
    fn scroll_top(&self) -> usize;

    /// Whether this is the frame's designated scroll view, upstream
    /// `primary`.
    fn is_primary(&self) -> bool;

    /// The overscroll policy, upstream `overscroll`.
    fn overscroll(&self) -> Overscroll;

    /// The last laid-out viewport height, upstream `viewportHeight`.
    fn viewport_height(&self) -> usize;

    /// The content width the child renders at for a given box width,
    /// upstream `getContentWidth`.
    fn content_width(&self, width: usize) -> usize;

    /// Commit one frame's geometry and reconcile the scroll position,
    /// upstream `updateLayout`.
    fn update_layout(
        &self,
        content_height: usize,
        viewport_height: usize,
        request_render: &RenderRequest,
    );

    /// The current scrollbar mode, upstream `scrollbar`.
    fn scrollbar(&self) -> ScrollViewScrollbar;

    /// Whether the scrollbar paints this frame, upstream
    /// `isScrollbarVisible`.
    fn is_scrollbar_visible(&self) -> bool;

    /// Whether the scrollbar is interactively active, upstream
    /// `isScrollbarActive`.
    fn is_scrollbar_active(&self) -> bool;

    /// The track glyph style, upstream `scrollbarTrackStyle`.
    fn scrollbar_track_style(&self) -> ColorFn;

    /// The thumb glyph style, upstream `scrollbarThumbStyle`.
    fn scrollbar_thumb_style(&self) -> ColorFn;
}

/// The scroll view's identity handle in a [`crate::layout::LayoutBox`].
pub type ScrollStateHandle = Arc<dyn ScrollLayoutState>;

/// A scroll view's node, upstream `ScrollLayoutNode`: the content
/// component plus the shared state handle.
#[derive(Clone)]
pub struct ScrollLayoutNode {
    /// The content component.
    pub component: Rc<dyn Component>,
    /// The shared scroll state.
    pub state: ScrollStateHandle,
}

impl std::fmt::Debug for ScrollLayoutNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScrollLayoutNode").finish_non_exhaustive()
    }
}

/// A stack or scroll node, upstream `LayoutNode`.
#[derive(Clone)]
pub enum LayoutNode {
    /// A stack node, upstream `StackLayoutNode`.
    Stack(StackLayoutNode),
    /// A scroll node, upstream `ScrollLayoutNode`.
    Scroll(ScrollLayoutNode),
}

impl std::fmt::Debug for LayoutNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stack(node) => node.fmt(f),
            Self::Scroll(node) => node.fmt(f),
        }
    }
}

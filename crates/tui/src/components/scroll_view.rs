//! `ScrollView`, ported from `packages/tui/src/components/scroll-view.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#44).
//!
//! A single-child container that clips its content to a viewport and
//! carries the scroll state the layout engine drives. The engine reaches
//! the state as a [`ScrollStateHandle`](crate::layout_node::ScrollStateHandle)
//! (`Arc`), which also carries the box identity `getScrollViewBox`
//! compares.
//!
//! Restatements against upstream:
//!
//! - `class ScrollView extends Container` becomes composition: the mutable
//!   state lives in an `Arc`-shared [`ScrollViewState`] implementing
//!   [`ScrollLayoutState`], and the component holds the child and the
//!   immutable options. Upstream's `children.push(component)` with the
//!   throwing `addChild` override becomes a single `child` field, with
//!   [`Component::children`] answering it for the tree walk.
//! - The scrollbar hide timer (`setTimeout` with `.unref()`) becomes a
//!   worker thread per arm with the terminal-port-style stop channel:
//!   `recv_timeout(delay)` is the tick and disconnecting the sender stops
//!   it. The thread only flips the transient-visibility atomic and calls
//!   the captured render request; an in-flight fire after a re-arm may
//!   land one extra request, as upstream's `clearTimeout` timing is also
//!   best-effort. The manual mode
//!   ([`ScrollViewOptions::manual_hide_timer`]) swaps the thread for a
//!   pending flag the harness drives: Node's suites could fake timers
//!   over the `setTimeout`, and the wall-clock thread raced the coverage
//!   gate's parallel load (#78).
//! - The `axis` option's runtime check disappears into the type: upstream
//!   throws for anything but `"vertical"`, and only the vertical axis
//!   exists here.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::components::ColorFn;
use crate::layout_node::{Overscroll, ScrollLayoutState};

use crate::tui::{Component, RenderRequest};

/// The scrollbar mode, upstream `ScrollViewScrollbar`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScrollViewScrollbar {
    /// Upstream `"hidden"` — never painted (the default).
    #[default]
    Hidden,
    /// Upstream `"auto"` — paints while scrolled, hides after the delay.
    Auto,
    /// Upstream `"always"` — paints and reserves a column.
    Always,
}

/// Follow behavior, upstream `follow?: "none" | "end"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FollowMode {
    /// Upstream `"none"`.
    None,
    /// Upstream `"end"` — keep the bottom edge pinned while content grows.
    End,
}

/// Options for [`ScrollView::scroll_to`], upstream
/// `ScrollViewScrollToOptions`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScrollViewScrollToOptions {
    /// Keep follow-end disabled even when the target is the current
    /// content end.
    pub disable_follow: bool,
}

/// Scroll view constructor options, upstream `ScrollViewOptions`.
#[derive(Clone, Default)]
pub struct ScrollViewOptions {
    /// Follow behavior, upstream `follow` (default `"none"`).
    pub follow: Option<FollowMode>,
    /// Whether this is the frame's designated scroll view, upstream
    /// `primary`.
    pub primary: bool,
    /// Overscroll policy, upstream `overscroll` (default `"chain"`).
    pub overscroll: Option<Overscroll>,
    /// Scrollbar mode, upstream `scrollbar` (default `"hidden"`).
    pub scrollbar: Option<ScrollViewScrollbar>,
    /// Track glyph style, upstream `scrollbarTrackStyle`.
    pub scrollbar_track_style: Option<ColorFn>,
    /// Thumb glyph style, upstream `scrollbarThumbStyle`.
    pub scrollbar_thumb_style: Option<ColorFn>,
    /// Transient-hide delay in milliseconds, upstream
    /// `scrollbarHideDelayMs` (default 1000).
    pub scrollbar_hide_delay_ms: Option<u64>,
    /// Arm the hide timer only for a driven fire
    /// ([`ScrollView::fire_scrollbar_hide_timer`]), the fake-timer mode
    /// for tests. Upstream had no equivalent — its `setTimeout` rode the
    /// JS event loop, and Node suites could fake timers over it. Off by
    /// default: the worker thread then fires after the delay, and the
    /// delay is not consulted while this is set.
    pub manual_hide_timer: bool,
}

impl std::fmt::Debug for ScrollViewOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScrollViewOptions")
            .field("follow", &self.follow)
            .field("primary", &self.primary)
            .field("overscroll", &self.overscroll)
            .field("scrollbar", &self.scrollbar)
            .field(
                "scrollbar_track_style",
                &self.scrollbar_track_style.is_some(),
            )
            .field(
                "scrollbar_thumb_style",
                &self.scrollbar_thumb_style.is_some(),
            )
            .field("scrollbar_hide_delay_ms", &self.scrollbar_hide_delay_ms)
            .field("manual_hide_timer", &self.manual_hide_timer)
            .finish()
    }
}

/// The default track style, upstream's fallback `\x1b[90m…\x1b[39m`.
fn default_track_style() -> ColorFn {
    Arc::new(|text| format!("\x1b[90m{text}\x1b[39m"))
}

/// The default thumb style, upstream's fallback `\x1b[37m…\x1b[39m`.
fn default_thumb_style() -> ColorFn {
    Arc::new(|text| format!("\x1b[37m{text}\x1b[39m"))
}

/// The armed scrollbar-hide worker; dropping it disconnects the stop
/// channel and the thread exits on its next wake.
struct HideWorker {
    /// Held purely so its drop disconnects the worker's stop channel.
    #[expect(
        dead_code,
        reason = "the sender's drop is the stop signal; nothing reads it"
    )]
    stop: std::sync::mpsc::Sender<()>,
}

/// The scroll state the layout engine and the component both drive, the
/// [`ScrollLayoutState`] implementation.
///
/// Owner-thread state sits in `Cell`s behind the `Arc`; the hide-timer
/// thread only flips the visibility atomic and calls the captured render
/// request.
pub struct ScrollViewState {
    scroll_top: Cell<usize>,
    content_height: Cell<usize>,
    viewport_height: Cell<usize>,
    following_end: Cell<bool>,
    follow_suppressed_at_end: Cell<bool>,
    scrollbar_mode: Cell<ScrollViewScrollbar>,
    scrollbar_active: Cell<bool>,
    transient_scrollbar_visible: Arc<AtomicBool>,
    hide_worker: std::cell::RefCell<Option<HideWorker>>,
    /// The manual mode's armed timer, upstream's pending `setTimeout`
    /// callback: set by the arm, cleared by every `clearTimeout` path,
    /// fired by [`ScrollView::fire_scrollbar_hide_timer`]. The threaded
    /// mode keeps the armed state in `hide_worker`.
    hide_pending: Cell<bool>,
    request_render_callback: std::cell::RefCell<Option<RenderRequest>>,
    follow_end: bool,
    primary: bool,
    overscroll: Overscroll,
    track_style: ColorFn,
    thumb_style: ColorFn,
    scrollbar_hide_delay_ms: u64,
    manual_hide_timer: bool,
}

impl std::fmt::Debug for ScrollViewState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScrollViewState")
            .field("scroll_top", &self.scroll_top.get())
            .field("content_height", &self.content_height.get())
            .field("viewport_height", &self.viewport_height.get())
            .field("following_end", &self.following_end.get())
            .finish_non_exhaustive()
    }
}

impl ScrollViewState {
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the constructor consumes the options object exactly as upstream's constructor destructures it"
    )]
    fn new(options: ScrollViewOptions) -> Self {
        Self {
            scroll_top: Cell::new(0),
            content_height: Cell::new(0),
            viewport_height: Cell::new(0),
            following_end: Cell::new(
                options
                    .follow
                    .is_some_and(|follow| follow == FollowMode::End),
            ),
            follow_suppressed_at_end: Cell::new(false),
            scrollbar_mode: Cell::new(options.scrollbar.unwrap_or_default()),
            scrollbar_active: Cell::new(false),
            transient_scrollbar_visible: Arc::new(AtomicBool::new(false)),
            hide_worker: std::cell::RefCell::new(None),
            hide_pending: Cell::new(false),
            request_render_callback: std::cell::RefCell::new(None),
            follow_end: options
                .follow
                .is_some_and(|follow| follow == FollowMode::End),
            primary: options.primary,
            overscroll: options.overscroll.unwrap_or_default(),
            track_style: options
                .scrollbar_track_style
                .clone()
                .unwrap_or_else(default_track_style),
            thumb_style: options
                .scrollbar_thumb_style
                .clone()
                .unwrap_or_else(default_thumb_style),
            scrollbar_hide_delay_ms: options.scrollbar_hide_delay_ms.unwrap_or(1000),
            manual_hide_timer: options.manual_hide_timer,
        }
    }

    const fn max_scroll_top(&self) -> usize {
        self.content_height
            .get()
            .saturating_sub(self.viewport_height.get())
    }

    /// Clear the armed hide timer, upstream's `clearTimeout`: the worker
    /// in the threaded mode, the pending flag in the manual mode.
    fn clear_hide_timer(&self) {
        self.hide_worker.take();
        self.hide_pending.set(false);
    }

    /// Mark scrollbar activity and arm the hide timer, upstream
    /// `markScrollbarActivity`.
    fn mark_scrollbar_activity(&self) {
        if self.scrollbar_mode.get() != ScrollViewScrollbar::Auto
            || self.content_height.get() <= self.viewport_height.get()
        {
            return;
        }
        self.transient_scrollbar_visible
            .store(true, Ordering::Relaxed);
        self.clear_hide_timer();
        if self.scrollbar_active.get() {
            return;
        }
        if self.manual_hide_timer {
            self.hide_pending.set(true);
            return;
        }
        let request_render = self.request_render_callback.borrow().clone();
        let transient = Arc::clone(&self.transient_scrollbar_visible);
        let delay = self.scrollbar_hide_delay_ms;
        let (stop, stop_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            match stop_rx.recv_timeout(std::time::Duration::from_millis(delay)) {
                Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    transient.store(false, Ordering::Relaxed);
                    if let Some(request_render) = request_render.as_ref() {
                        request_render();
                    }
                }
            }
        });
        self.hide_worker.replace(Some(HideWorker { stop }));
    }

    /// The manual mode's driven fire, upstream's timer callback running
    /// where a fake clock says the delay elapsed: the pending flag clears,
    /// the visibility flips, and the current render request runs inline on
    /// the calling thread, where the event loop would have run it. A
    /// no-op when nothing is armed.
    fn fire_pending_hide_timer(&self) {
        if !self.hide_pending.replace(false) {
            return;
        }
        self.transient_scrollbar_visible
            .store(false, Ordering::Relaxed);
        self.request_render();
    }

    fn hide_transient_scrollbar(&self) {
        self.transient_scrollbar_visible
            .store(false, Ordering::Relaxed);
        self.clear_hide_timer();
    }

    /// Commit one frame's geometry and reconcile the scroll position, the
    /// body of upstream's `updateLayout`.
    fn reconcile_layout(&self, content_height: usize, viewport_height: usize) {
        self.content_height.set(content_height);
        self.viewport_height.set(viewport_height);
        let max_scroll_top = self.max_scroll_top();
        if self.following_end.get() {
            self.scroll_top.set(max_scroll_top);
        } else {
            self.scroll_top
                .set(self.scroll_top.get().min(max_scroll_top));
        }
        if self.scroll_top.get() < max_scroll_top {
            self.follow_suppressed_at_end.set(false);
        }
        if self.follow_end
            && self.scroll_top.get() == max_scroll_top
            && !self.follow_suppressed_at_end.get()
        {
            self.following_end.set(true);
        }
        if self.content_height.get() <= self.viewport_height.get() {
            self.hide_transient_scrollbar();
        }
    }

    fn request_render(&self) {
        if let Some(request_render) = self.request_render_callback.borrow().as_ref() {
            request_render();
        }
    }

    /// The scrollbar-visible getter, upstream's `isScrollbarVisible`.
    #[must_use]
    pub fn is_scrollbar_visible(&self) -> bool {
        match self.scrollbar_mode.get() {
            ScrollViewScrollbar::Always => self.viewport_height.get() > 0,
            ScrollViewScrollbar::Auto => {
                self.content_height.get() > self.viewport_height.get()
                    && self.transient_scrollbar_visible.load(Ordering::Relaxed)
            }
            ScrollViewScrollbar::Hidden => false,
        }
    }

    fn set_scrollbar(&self, scrollbar: ScrollViewScrollbar) {
        if scrollbar == self.scrollbar_mode.get() {
            return;
        }
        self.scrollbar_mode.set(scrollbar);
        if scrollbar != ScrollViewScrollbar::Auto {
            self.hide_transient_scrollbar();
        } else if self.scrollbar_active.get() {
            self.mark_scrollbar_activity();
        }
        self.request_render();
    }

    fn set_scrollbar_active(&self, active: bool) {
        if active == self.scrollbar_active.get() {
            return;
        }
        self.scrollbar_active.set(active);
        self.mark_scrollbar_activity();
        self.request_render();
    }

    /// Upstream `ScrollView.scrollTo`.
    fn scroll_to(&self, scroll_top: usize, options: ScrollViewScrollToOptions) {
        let max_scroll_top = self.max_scroll_top();
        let next = scroll_top.min(max_scroll_top);
        let next_follow_suppressed_at_end = options.disable_follow && next == max_scroll_top;
        let next_following_end =
            !next_follow_suppressed_at_end && self.follow_end && next == max_scroll_top;
        if next == self.scroll_top.get()
            && next_following_end == self.following_end.get()
            && next_follow_suppressed_at_end == self.follow_suppressed_at_end.get()
        {
            return;
        }
        let moved = next != self.scroll_top.get();
        self.scroll_top.set(next);
        self.following_end.set(next_following_end);
        self.follow_suppressed_at_end
            .set(next_follow_suppressed_at_end);
        if moved {
            self.mark_scrollbar_activity();
        }
        self.request_render();
    }

    /// Upstream `ScrollView.scrollBy`; returns the unused scroll delta.
    fn scroll_by(&self, lines: i64) -> i64 {
        if lines == 0 {
            return 0;
        }
        let max_scroll_top = self.max_scroll_top();
        let start = if self.following_end.get() {
            max_scroll_top
        } else {
            self.scroll_top.get()
        };
        let next = (i64::from(u32::try_from(start).unwrap_or(u32::MAX)) + lines).clamp(
            0,
            i64::from(u32::try_from(max_scroll_top).unwrap_or(u32::MAX)),
        );
        let moved = next - i64::from(u32::try_from(start).unwrap_or(u32::MAX));
        let was_following_end = self.following_end.get();
        self.scroll_top.set(usize::try_from(next).unwrap_or(0));
        self.following_end.set(
            self.follow_end && next == i64::from(u32::try_from(max_scroll_top).unwrap_or(u32::MAX)),
        );
        self.follow_suppressed_at_end.set(false);
        if moved != 0 {
            self.mark_scrollbar_activity();
        }
        if moved != 0 || self.following_end.get() != was_following_end {
            self.request_render();
        }
        lines - moved
    }

    fn scroll_to_start(&self) {
        let changed = self.scroll_top.get() != 0
            || self.following_end.get()
                != (self.follow_end && self.content_height.get() <= self.viewport_height.get());
        self.scroll_top.set(0);
        self.following_end
            .set(self.follow_end && self.content_height.get() <= self.viewport_height.get());
        self.follow_suppressed_at_end.set(false);
        if changed {
            self.mark_scrollbar_activity();
            self.request_render();
        }
    }

    fn scroll_to_end(&self) {
        let next = self.max_scroll_top();
        let changed = self.scroll_top.get() != next || self.following_end.get() != self.follow_end;
        self.scroll_top.set(next);
        self.following_end.set(self.follow_end);
        self.follow_suppressed_at_end.set(false);
        if changed {
            self.mark_scrollbar_activity();
            self.request_render();
        }
    }
}

impl ScrollLayoutState for ScrollViewState {
    fn scroll_top(&self) -> usize {
        self.scroll_top.get()
    }

    fn is_primary(&self) -> bool {
        self.primary
    }

    fn overscroll(&self) -> Overscroll {
        self.overscroll
    }

    fn viewport_height(&self) -> usize {
        self.viewport_height.get()
    }

    fn content_width(&self, width: usize) -> usize {
        // The rule reads the scrollbar mode directly: an always-reserved
        // scrollbar costs one column, everything else renders full width.
        if self.scrollbar_mode.get() == ScrollViewScrollbar::Always && width > 1 {
            width - 1
        } else {
            width
        }
    }

    fn update_layout(
        &self,
        content_height: usize,
        viewport_height: usize,
        request_render: &RenderRequest,
    ) {
        *self.request_render_callback.borrow_mut() = Some(Arc::clone(request_render));
        self.reconcile_layout(content_height, viewport_height);
    }

    fn scrollbar(&self) -> ScrollViewScrollbar {
        self.scrollbar_mode.get()
    }

    fn is_scrollbar_visible(&self) -> bool {
        Self::is_scrollbar_visible(self)
    }

    fn is_scrollbar_active(&self) -> bool {
        self.scrollbar_active.get()
    }

    fn scrollbar_track_style(&self) -> ColorFn {
        self.track_style.clone()
    }

    fn scrollbar_thumb_style(&self) -> ColorFn {
        self.thumb_style.clone()
    }

    fn scroll_by(&self, lines: i64) -> i64 {
        Self::scroll_by(self, lines)
    }

    fn scroll_to(&self, scroll_top: usize, options: ScrollViewScrollToOptions) {
        Self::scroll_to(self, scroll_top, options);
    }

    fn scroll_to_start(&self) {
        Self::scroll_to_start(self);
    }

    fn scroll_to_end(&self) {
        Self::scroll_to_end(self);
    }

    fn follow_end(&self) -> bool {
        self.follow_end
    }

    fn is_following_end(&self) -> bool {
        self.following_end.get()
    }

    fn set_scrollbar_active(&self, active: bool) {
        Self::set_scrollbar_active(self, active);
    }
}

/// A single-child container that clips its content to a viewport and
/// carries the scroll state the layout engine drives, upstream
/// `class ScrollView`.
pub struct ScrollView {
    child: Rc<dyn Component>,
    state: Arc<ScrollViewState>,
}

impl std::fmt::Debug for ScrollView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScrollView")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl ScrollView {
    /// The scroll state handle the layout engine issues, upstream's
    /// `ScrollLayoutNode.state`: the identity the alternate-screen renderer's
    /// selection points and scrollbar drags compare.
    #[must_use]
    pub fn state_handle(&self) -> crate::layout_node::ScrollStateHandle {
        self.state.clone()
    }

    /// Upstream `new ScrollView(component, options)`.
    #[must_use]
    #[expect(
        clippy::arc_with_non_send_sync,
        reason = "the scroll state rides a single-threaded UI executor; the Arc<dyn ScrollLayoutState> handle the layout engine stores deliberately carries no Send/Sync bound"
    )]
    pub fn new(child: Rc<dyn Component>, options: ScrollViewOptions) -> Rc<Self> {
        Rc::new(Self {
            child,
            state: Arc::new(ScrollViewState::new(options)),
        })
    }

    /// The current scroll offset, upstream `scrollTop`.
    #[must_use]
    pub fn scroll_top(&self) -> usize {
        self.state.scroll_top()
    }

    /// Whether the content bottom is pinned, upstream `isFollowingEnd`.
    #[must_use]
    pub fn is_following_end(&self) -> bool {
        self.state.following_end.get()
    }

    /// The last laid-out viewport height, upstream `viewportHeight`.
    #[must_use]
    pub fn viewport_height(&self) -> usize {
        self.state.viewport_height()
    }

    /// The scrollbar mode, upstream `scrollbar`.
    #[must_use]
    pub fn scrollbar(&self) -> ScrollViewScrollbar {
        self.state.scrollbar()
    }

    /// Whether the scrollbar paints this frame, upstream
    /// `isScrollbarVisible`.
    #[must_use]
    pub fn is_scrollbar_visible(&self) -> bool {
        self.state.is_scrollbar_visible()
    }

    /// Whether the scrollbar is interactively active, upstream
    /// `isScrollbarActive`.
    #[must_use]
    pub fn is_scrollbar_active(&self) -> bool {
        self.state.scrollbar_active.get()
    }

    /// The content width the child renders at for a given box width,
    /// upstream `getContentWidth`.
    #[must_use]
    pub fn content_width(&self, width: usize) -> usize {
        self.state.content_width(width)
    }

    /// Toggle the scrollbar mode, upstream `setScrollbar`.
    pub fn set_scrollbar(&self, scrollbar: ScrollViewScrollbar) {
        self.state.set_scrollbar(scrollbar);
    }

    /// Toggle the interactive scrollbar state, upstream
    /// `setScrollbarActive`.
    pub fn set_scrollbar_active(&self, active: bool) {
        self.state.set_scrollbar_active(active);
    }

    /// Fire the armed hide timer now, the drive behind
    /// [`ScrollViewOptions::manual_hide_timer`]: the harness stand-in for
    /// a fake clock advancing past the delay, upstream's fake-timer
    /// advance over the `setTimeout` arm. A no-op in the threaded mode or
    /// when the timer was cleared since the arm; the fire runs the current
    /// render request on the calling thread.
    pub fn fire_scrollbar_hide_timer(&self) {
        self.state.fire_pending_hide_timer();
    }

    /// Scroll to an absolute offset, upstream `ScrollView.scrollTo`.
    pub fn scroll_to(&self, scroll_top: usize, options: ScrollViewScrollToOptions) {
        self.state.scroll_to(scroll_top, options);
    }

    /// Scroll by lines; returns the unused delta, upstream `scrollBy`.
    #[must_use]
    pub fn scroll_by(&self, lines: i64) -> i64 {
        self.state.scroll_by(lines)
    }

    /// Upstream `ScrollView.scrollToStart`.
    pub fn scroll_to_start(&self) {
        self.state.scroll_to_start();
    }

    /// Upstream `ScrollView.scrollToEnd`.
    pub fn scroll_to_end(&self) {
        self.state.scroll_to_end();
    }
}

impl Component for ScrollView {
    fn render(&self, width: usize) -> Vec<String> {
        let content_width = self.state.content_width(width);
        let lines = self.child.render(content_width);
        if content_width == width {
            lines
        } else {
            lines.into_iter().map(|line| format!("{line} ")).collect()
        }
    }

    fn is_stock_mouse_container(&self) -> bool {
        true
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        vec![Rc::clone(&self.child)]
    }

    fn invalidate(&self) {
        self.child.invalidate();
    }

    fn layout_node(&self) -> Option<crate::layout_node::LayoutNode> {
        let state: crate::layout_node::ScrollStateHandle = self.state.clone();
        Some(crate::layout_node::LayoutNode::Scroll(
            crate::layout_node::ScrollLayoutNode {
                component: Rc::clone(&self.child),
                state,
            },
        ))
    }
}

//! The alternate-screen renderer of `packages/tui/src/tui-alt-screen.ts`
//! ([#46](https://github.com/PhillipChaffee/pi-rust/issues/46)).
//!
//! Ported from earendil-works/pi at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`TuiAltScreen`] renders the mounted layout root (or an implicit
//! follow-end scroll view over the mounted children) into a terminal-sized
//! viewport with paint-exactly-`terminal.rows` differential rendering, and
//! owns the mouse: viewport scrolling and wheel chaining, application-owned
//! text selection with OSC 52 copy, scrollbar drags, OSC 8 hyperlink clicks,
//! the transcript search overlay, and the flash stack.
//!
//! Restatements against upstream, all surveyed in map ticket "Survey the tui
//! package" or landed with earlier slices:
//!
//! - `class TuiAltScreen extends TuiBase` becomes [`TuiAltScreen`]
//!   implementing [`TuiRenderer`] over a shared `TuiAltScreenCore`: the
//!   renderer is boxed inside the [`Tui`] while sessions and tests call the
//!   renderer-level API, so the mutable state lives in an `Rc` the handle
//!   clones, and the core reaches the base through the [`Weak`]
//!   [`TuiRenderer::attach_base`] stores.
//! - Upstream's duck-typed `ScrollView` references (selection points,
//!   scrollbar drags, wheel routing) become
//!   [`ScrollStateHandle`]s, the
//!   identity the layout engine issues; the scroll mutations ride the
//!   `ScrollLayoutState` trait methods the handle carries.
//! - The dispatch skip compared `handleMouse ===
//!   Container.prototype.handleMouse` on layout participants; the port
//!   answers [`Component::is_stock_mouse_container`], which the stacks and
//!   the scroll view carry (their walks restate the stock container).
//! - `dispatchMouseToLayout` produced negative local coordinates for boxes
//!   whose rect does not contain the pointer; the port clamps into the
//!   `u16` the event rides — receivers the point sat outside never saw a
//!   meaningful coordinate upstream either.
//! - `Date.now()` click timestamps become epoch milliseconds; the selection
//!   auto-scroll `setInterval` and the flash `setTimeout`s become worker
//!   threads with the terminal-port-style stop channels — the workers only
//!   deliver ticks or expired ids through channels and raise the render
//!   demand, and the renderer drains both on the owner thread at the top of
//!   its frame, exactly upstream's timer callbacks running on the event loop
//!   before the next paint.
//! - The environment reads (`TERM`, `TMUX`, `ZELLIJ`, `STY`, `TERM_PROGRAM`)
//!   resolve through the [`EnvLookup`] seam, and the right-click paste
//!   gate's `process.platform === "win32"` through an injectable Windows
//!   probe: Rust cannot read the process environment (or a runtime platform
//!   identity) without the `unsafe` this workspace forbids, so tests inject
//!   both.
//! - `copySelection`'s `Promise<boolean | string>` becomes a sync handler
//!   answering [`CopySelectionResult`]: every upstream caller resolved in
//!   the same tick, and the coding-agent port supplies a synchronous native
//!   clipboard.
//! - The Kitty upload cache (`Map<number, CachedKittyImage>`) is a
//!   `Vec`-backed insertion-ordered map: the upstream `Map`'s iteration
//!   order is the least-recently-visible eviction sequence, which Rust
//!   `HashMap` does not preserve.
//! - `handleMouse`'s click-count and wheel paths parse SGR coordinates with
//!   saturating `u16` casts; upstream's `parseInt` produced arbitrarily
//!   large numbers the terminal dimensions bounded in practice.
//! - `applySelectionHighlight` walks text by UTF-8 [`char`] rather than
//!   UTF-16 code units (survey flag 7, the same restatement the bounded
//!   writer landed): the character it appends is the whole code point, not
//!   a lone surrogate half.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::rc::{Rc, Weak};
use std::sync::LazyLock;
use std::sync::mpsc::channel;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use regex::Regex;

use crate::alt_screen_search::{
    AltScreenSearchComponent, AltScreenSearchIndex, AltScreenSearchMatch, NavigationButtonStyleFn,
    get_alt_screen_search_match_key,
};
use crate::components::{
    AltScreenFlashContainer, FollowMode, InputStyleFn, ScrollView, ScrollViewOptions,
    ScrollViewScrollToOptions,
};
use crate::keybindings::get_keybindings;
use crate::keys::is_key_release;
use crate::layout::{
    LayoutFrame, ScrollbarGeometry, get_layout_boxes_at, get_scroll_view_box, get_scroll_views_at,
    get_scrollbar_geometry, render_layout_frame,
};
use crate::layout_node::{Overscroll, ScrollStateHandle};
use crate::terminal::{EnvLookup, default_env_lookup};
use crate::terminal_image::{
    ImageProtocol, TerminalCapabilities, delete_all_kitty_images, delete_all_kitty_placements,
    delete_kitty_image, get_capabilities, get_kitty_image_placement, is_image_line,
    set_capabilities,
};
use crate::tui::{
    CURSOR_MARKER, Component, OverlayAnchor, OverlayHandle, OverlayMargin, OverlayOptions,
    SizeValue, Tui, TuiInputListenerResult, TuiMode, TuiMouseButton, TuiMouseDispatchTarget,
    TuiMouseEvent, TuiMouseEventResult, TuiMouseEventType, TuiRenderer, TuiStopOptions,
    composite_tui_line, dispatch_mouse_event, retarget_mouse_event,
};
use crate::utils::{
    extract_ansi_code, get_grapheme_cell_range, get_osc8_link_at_column, is_word_like,
    slice_by_column, static_regex, strip_terminal_sequences, truncate_to_width, visible_width,
    word_segments,
};

const ENTER_ALT_SCREEN: &str = "\x1b[?1049h";
const EXIT_ALT_SCREEN: &str = "\x1b[?1049l";
const DISABLE_AUTOWRAP: &str = "\x1b[?7l";
const ENABLE_AUTOWRAP: &str = "\x1b[?7h";
const ENABLE_BUTTON_MOTION_MOUSE: &str = "\x1b[?1000h\x1b[?1002h\x1b[?1004h\x1b[?1006h";
const ENABLE_ALL_MOTION_MOUSE: &str = "\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1004h\x1b[?1006h";
const DISABLE_MOUSE: &str = "\x1b[?1006l\x1b[?1004l\x1b[?1003l\x1b[?1002l\x1b[?1000l";
const FOCUS_IN: &str = "\x1b[I";
const FOCUS_OUT: &str = "\x1b[O";
const BEGIN_SYNCHRONIZED_OUTPUT: &str = "\x1b[?2026h";
const END_SYNCHRONIZED_OUTPUT: &str = "\x1b[?2026l";
const PAGE_SCROLL_OVERLAP: usize = 4;
const ALT_WHEEL_SCROLL_MULTIPLIER: i64 = 5;
const MAX_CACHED_OFFSCREEN_KITTY_IMAGES: usize = 16;
const MAX_CACHED_OFFSCREEN_KITTY_TRANSMISSION_BYTES: usize = 32 * 1024 * 1024;
const MAX_CACHED_OFFSCREEN_KITTY_DECODED_BYTES: u64 = 64 * 1024 * 1024;
const DOUBLE_CLICK_INTERVAL_MS: u128 = 500;
const COPY_ERROR_FLASH_DURATION_MS: u64 = 5000;
const SELECTION_AUTO_SCROLL_INTERVAL_MS: u64 = 50;

static OSC133_ZONE_PREFIX: LazyLock<Regex> =
    LazyLock::new(|| static_regex(r"^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+"));
static OSC133_PROMPT_START: LazyLock<Regex> =
    LazyLock::new(|| static_regex(r"^\x1b\]133;A(?:\x07|\x1b\\)"));
static SGR_MOUSE: LazyLock<Regex> =
    LazyLock::new(|| static_regex(r"^\x1b\[<(\d+);(\d+);(\d+)([Mm])$"));

/// Word-selection joiners the fullscreen mouse selection keeps whole,
/// upstream `TERMINAL_WORD_SELECTION_JOINERS`: regular mode delegates
/// double-click selection to the terminal emulator, fullscreen owns mouse
/// selection, so mirror common terminal word-selection behavior by keeping
/// paths and kebab-case tokens whole.
const TERMINAL_WORD_SELECTION_JOINERS: [&str; 2] = ["/", "-"];

/// The transmission bookkeeping for one uploaded Kitty image, upstream
/// `CachedKittyImage`.
#[derive(Debug, Clone, Copy)]
struct CachedKittyImage {
    transmission_generation: u64,
    transmission_bytes: usize,
    estimated_decoded_bytes: u64,
}

/// Where a selection begins or ends, upstream `SelectionPoint`.
#[derive(Clone)]
struct SelectionPoint {
    row: usize,
    col: usize,
    scroll_view: Option<ScrollStateHandle>,
    /// Whether this point lies between terminal cells rather than on a cell,
    /// upstream `boundary`.
    boundary: bool,
}

impl std::fmt::Debug for SelectionPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectionPoint")
            .field("row", &self.row)
            .field("col", &self.col)
            .field("boundary", &self.boundary)
            .finish_non_exhaustive()
    }
}

impl SelectionPoint {
    fn screen(row: usize, col: usize) -> Self {
        Self {
            row,
            col,
            scroll_view: None,
            boundary: false,
        }
    }

    /// Upstream's `{ ...point, col }` spread.
    fn with_col(&self, col: usize) -> Self {
        Self {
            col,
            ..self.clone()
        }
    }

    /// Upstream's `{ ...point, boundary: true }` spread.
    fn with_boundary(&self) -> Self {
        Self {
            boundary: true,
            ..self.clone()
        }
    }
}

/// The selected column span on one line, upstream `getSelectionColumns`'s
/// `{ start, end }` object.
#[derive(Debug, Clone, Copy)]
struct SelectionColumns {
    start: usize,
    end: usize,
}

#[derive(Clone)]
struct SelectionRange {
    start: SelectionPoint,
    end: SelectionPoint,
}

impl std::fmt::Debug for SelectionRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectionRange")
            .field("start", &self.start.row)
            .field("end", &self.end.row)
            .finish_non_exhaustive()
    }
}

/// Selection granularity, upstream `SelectionGranularity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SelectionGranularity {
    #[default]
    Character,
    Word,
    Line,
}

/// One component-owned click for repeat detection, upstream's
/// `lastComponentClick` object.
#[derive(Clone)]
struct ComponentClickTarget {
    timestamp: u128,
    count: u32,
    component: Rc<dyn Component>,
    x: u16,
    y: u16,
}

/// One text-selection click for double/triple-click detection, upstream
/// `ClickTarget`.
#[derive(Clone)]
struct ClickTarget {
    timestamp: u128,
    count: u32,
    row: usize,
    scroll_view: Option<ScrollStateHandle>,
    word_start: usize,
    word_end: usize,
}

/// One decoded SGR mouse report, upstream `SgrMouseEvent`.
#[derive(Debug, Clone, Copy)]
struct SgrMouseEvent {
    button: u32,
    x: u16,
    y: u16,
    release: bool,
}

/// One decoded wheel report, upstream `WheelEvent`.
#[derive(Debug, Clone, Copy)]
struct WheelEvent {
    /// `-1` scrolls up, `1` scrolls down, upstream `direction: -1 | 1`.
    direction: i32,
    x: u16,
    y: u16,
    button: u32,
}

/// An active scrollbar drag, upstream `ScrollbarDrag`.
struct ScrollbarDrag {
    scroll_view: ScrollStateHandle,
    grab_offset: usize,
}

/// A scrollbar the pointer can grab, upstream `ScrollbarTarget`.
struct ScrollbarTarget {
    scroll_view: ScrollStateHandle,
    geometry: ScrollbarGeometry,
}

/// The clickable jump-to-end rectangle, upstream `ScrollToEndIndicatorRect`.
#[derive(Debug, Clone, Copy)]
struct ScrollToEndIndicatorRect {
    row: usize,
    column: usize,
    width: usize,
}

/// What the next search refresh selects, upstream `SearchSelectionMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchSelectionMode {
    Query,
    Retain,
    Next,
    Previous,
}

/// One highlight run on one screen row, upstream `SearchHighlightRange`.
#[derive(Debug, Clone, Copy)]
struct SearchHighlightRange {
    start_col: usize,
    end_col: usize,
    current: bool,
}

/// Open an OSC 8 hyperlink, upstream `openUrl`.
pub type OpenUrlCallback = Rc<dyn Fn(&str)>;

/// Handle a secondary-button clipboard paste, upstream `onRightClickPaste`.
pub type PasteCallback = Rc<dyn Fn()>;

/// Copy selected text to the system clipboard, upstream `copySelection`.
pub type CopySelectionCallback = Rc<dyn Fn(&str) -> CopySelectionResult>;

/// Render a clickable jump-to-end label, upstream `scrollToEndIndicator`.
pub type ScrollToEndIndicatorCallback = Rc<dyn Fn() -> String>;

/// The injected clipboard outcome, upstream `copySelection`'s awaited
/// `boolean | string` result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopySelectionResult {
    /// Upstream `true` — the text was copied.
    Copied,
    /// Upstream `false` — a generic failure.
    Failed,
    /// Upstream a string — the error message to flash.
    Message(String),
}

/// The renderer-local construction options, upstream `TuiAltScreenOptions`.
///
/// The environment seam and the Windows probe ride beside the upstream
/// options: the environment reads upstream resolved through `process.env`
/// per event, and the platform gate through `process.platform`, and Rust
/// cannot read either without the `unsafe` this workspace forbids.
#[derive(Default)]
pub struct TuiAltScreenConfig {
    /// Number of logical lines moved for each mouse-wheel event, upstream
    /// `wheelScrollLines`.
    pub wheel_scroll_lines: Option<u32>,
    /// Capture mouse events for viewport scrolling and application-owned
    /// text selection, upstream `mouse` (default true).
    pub mouse: Option<bool>,
    /// Style a non-current transcript search match, upstream
    /// `searchMatchStyle`.
    pub search_match_style: Option<InputStyleFn>,
    /// Style the current transcript search match, upstream
    /// `searchCurrentMatchStyle`.
    pub search_current_match_style: Option<InputStyleFn>,
    /// Style a transcript search navigation button, upstream
    /// `searchNavigationButtonStyle`.
    pub search_navigation_button_style: Option<NavigationButtonStyleFn>,
    /// Render a clickable jump-to-end label, upstream
    /// `scrollToEndIndicator`: centered on the last row of a follow-end
    /// primary scroll view while that view is scrolled away from its end.
    pub scroll_to_end_indicator: Option<ScrollToEndIndicatorCallback>,
    /// Open an OSC 8 hyperlink activated with a primary-button click,
    /// upstream `openUrl`.
    pub open_url: Option<OpenUrlCallback>,
    /// Handle an unmodified secondary-button press for clipboard paste,
    /// upstream `onRightClickPaste`.
    pub on_right_click_paste: Option<PasteCallback>,
    /// Automatically copy selected text to the clipboard on mouse release,
    /// upstream `copyOnSelect` (default true).
    pub copy_on_select: Option<bool>,
    /// Copy selected text to the system clipboard, upstream
    /// `copySelection`. When absent, the selection is copied via an OSC 52
    /// write.
    pub copy_selection: Option<CopySelectionCallback>,
    /// The environment lookup, upstream's `process.env` reads. `None` reads
    /// the real process environment.
    pub env_lookup: Option<EnvLookup>,
    /// The Windows-platform probe behind the right-click paste gate,
    /// upstream's `process.platform !== "win32"` check. `None` answers the
    /// compile-time target.
    pub is_windows: Option<Rc<dyn Fn() -> bool>>,
}

impl std::fmt::Debug for TuiAltScreenConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuiAltScreenConfig")
            .field("wheel_scroll_lines", &self.wheel_scroll_lines)
            .field("mouse", &self.mouse)
            .field("search_match_style", &self.search_match_style.is_some())
            .field(
                "search_current_match_style",
                &self.search_current_match_style.is_some(),
            )
            .field(
                "search_navigation_button_style",
                &self.search_navigation_button_style.is_some(),
            )
            .field(
                "scroll_to_end_indicator",
                &self.scroll_to_end_indicator.is_some(),
            )
            .field("open_url", &self.open_url.is_some())
            .field("on_right_click_paste", &self.on_right_click_paste.is_some())
            .field("copy_on_select", &self.copy_on_select)
            .field("copy_selection", &self.copy_selection.is_some())
            .field("env_lookup", &self.env_lookup.is_some())
            .field("is_windows", &self.is_windows.is_some())
            .finish()
    }
}

/// The armed selection auto-scroll interval, upstream's
/// `selectionAutoScrollTimer`; dropping it stops the worker, upstream's
/// `clearInterval`.
struct AutoScrollWorker {
    /// Held purely so its drop disconnects the worker's stop channel.
    #[expect(
        dead_code,
        reason = "the sender's drop is the stop signal; nothing reads it"
    )]
    stop: std::sync::mpsc::Sender<()>,
}

/// The implicit document, upstream's object literal: the mounted children
/// presented as one component the implicit scroll view wraps.
struct ImplicitDocument {
    tui: RefCell<Weak<Tui>>,
}

impl std::fmt::Debug for ImplicitDocument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImplicitDocument")
            .field("attached", &self.tui.borrow().upgrade().is_some())
            .finish_non_exhaustive()
    }
}

impl Component for ImplicitDocument {
    fn render(&self, width: usize) -> Vec<String> {
        self.tui
            .borrow()
            .upgrade()
            .map_or_else(Vec::new, |tui| tui.render_children(width))
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.tui
            .borrow()
            .upgrade()
            .and_then(|tui| tui.handle_mouse(event))
    }

    fn invalidate(&self) {
        // Upstream walks the base's children only — the layout root and the
        // overlays ride the base's own invalidate.
        if let Some(tui) = self.tui.borrow().upgrade() {
            for child in tui.children() {
                child.invalidate();
            }
        }
    }
}

/// The alternate-screen renderer's mutable state, upstream's `TuiAltScreen`
/// instance fields; the [`TuiAltScreen`] handle delegates to it and
/// sessions hold clones.
struct TuiAltScreenCore {
    /// The core's own handle, for the search component's query callback —
    /// upstream's closures closed over `this`, and the core cannot hold a
    /// strong self-reference.
    self_weak: Weak<Self>,
    base: RefCell<Weak<Tui>>,
    env_lookup: EnvLookup,
    is_windows: Rc<dyn Fn() -> bool>,
    previous_screen: RefCell<Vec<String>>,
    last_document: RefCell<Vec<String>>,
    previous_screen_width: Cell<usize>,
    previous_screen_height: Cell<usize>,
    layout_root: RefCell<Option<Rc<dyn Component>>>,
    current_layout: RefCell<Option<LayoutFrame>>,
    implicit_document: Rc<ImplicitDocument>,
    implicit_scroll_view: Rc<ScrollView>,
    flashes: AltScreenFlashContainer,
    alt_screen_active: Cell<bool>,
    image_protocol: Cell<Option<ImageProtocol>>,
    saved_capabilities: RefCell<Option<TerminalCapabilities>>,
    /// Insertion-ordered upload cache, upstream's `Map<number,
    /// CachedKittyImage>`: the iteration order is the
    /// least-recently-visible eviction sequence.
    uploaded_kitty_images: RefCell<Vec<(u64, CachedKittyImage)>>,
    selection_anchor: RefCell<Option<SelectionPoint>>,
    selection_focus: RefCell<Option<SelectionPoint>>,
    selection_granularity: Cell<SelectionGranularity>,
    selection_initial_range: RefCell<Option<SelectionRange>>,
    last_click: RefCell<Option<ClickTarget>>,
    selection_drag_pointer: RefCell<Option<(usize, usize)>>,
    selection_auto_scroll_direction: Cell<i8>,
    /// The armed auto-scroll interval, upstream's `selectionAutoScrollTimer`;
    /// its worker delivers ticks through [`Self::auto_scroll_rx`].
    auto_scroll_worker: RefCell<Option<AutoScrollWorker>>,
    auto_scroll_rx: RefCell<Option<std::sync::mpsc::Receiver<()>>>,
    selection_press_active: Cell<bool>,
    scrollbar_drag: RefCell<Option<ScrollbarDrag>>,
    scrollbar_hover: RefCell<Option<ScrollStateHandle>>,
    scroll_to_end_indicator_rect: RefCell<Option<ScrollToEndIndicatorRect>>,
    active_search: RefCell<Option<ActiveSearch>>,
    pressed_url: RefCell<Option<String>>,
    selection_dragged: Cell<bool>,
    mouse_capture: RefCell<Option<TuiMouseDispatchTarget>>,
    mouse_press_target: RefCell<Option<TuiMouseDispatchTarget>>,
    mouse_press_point: RefCell<Option<(u16, u16)>>,
    mouse_press_moved: Cell<bool>,
    last_component_click: RefCell<Option<ComponentClickTarget>>,
    wheel_scroll_lines: u32,
    mouse_enabled: bool,
    search_match_style: InputStyleFn,
    search_current_match_style: InputStyleFn,
    search_navigation_button_style: NavigationButtonStyleFn,
    scroll_to_end_indicator: Option<ScrollToEndIndicatorCallback>,
    open_url: Option<OpenUrlCallback>,
    on_right_click_paste: Option<PasteCallback>,
    copy_on_select: Cell<bool>,
    copy_selection: Option<CopySelectionCallback>,
}

/// The live transcript search, upstream `ActiveSearch`.
struct ActiveSearch {
    component: Rc<AltScreenSearchComponent>,
    index: AltScreenSearchIndex,
    overlay: Option<OverlayHandle>,
    query: String,
    matches: Vec<AltScreenSearchMatch>,
    selected_index: i64,
    selected_key: Option<String>,
    anchor_row: usize,
    selection_mode: SearchSelectionMode,
}

impl std::fmt::Debug for TuiAltScreenCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuiAltScreenCore")
            .field("alt_screen_active", &self.alt_screen_active.get())
            .field("wheel_scroll_lines", &self.wheel_scroll_lines)
            .field("mouse_enabled", &self.mouse_enabled)
            .field("copy_on_select", &self.copy_on_select.get())
            .finish_non_exhaustive()
    }
}

/// Alternate-screen TUI with a scrollable, application-owned viewport,
/// upstream `class TuiAltScreen`.
///
/// Construct through [`TuiAltScreen::new`], hand a clone to
/// [`crate::tui::TuiConfig::renderer`], and drive the renderer-level API
/// (`viewport_top`, `flash`, the selection surface) through the clone.
#[derive(Clone)]
pub struct TuiAltScreen {
    core: Rc<TuiAltScreenCore>,
}

impl std::fmt::Debug for TuiAltScreen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuiAltScreen")
            .field("alt_screen_active", &self.core.alt_screen_active.get())
            .finish_non_exhaustive()
    }
}

/// Epoch milliseconds, upstream `Date.now()`.
fn epoch_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

/// usize into the renderer's i64 row arithmetic: line counts ride terminal
/// dimensions and content heights, bounded far below i64.
fn line_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// i64 into a row or column index: negative values clamp to zero, upstream's
/// `Math.max(0, ...)`.
fn span_usize(value: i64) -> usize {
    usize::try_from(value.max(0)).unwrap_or(usize::MAX)
}

/// Whether two optional scroll handles name the same scroll view, upstream's
/// `scrollView === point.scrollView` identity comparisons.
fn same_scroll_view(a: Option<&ScrollStateHandle>, b: Option<&ScrollStateHandle>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

impl TuiAltScreen {
    /// Upstream's `new TuiAltScreen(terminal, showHardwareCursor?,
    /// logDirectory?, options?)` body; the terminal, the cursor toggle, and
    /// the log directory live on the [`Tui`] base, which calls
    /// [`TuiRenderer::attach_base`] with itself once constructed.
    #[must_use]
    pub fn new(config: TuiAltScreenConfig) -> Self {
        let implicit_document = Rc::new(ImplicitDocument {
            tui: RefCell::new(Weak::new()),
        });
        let implicit_document_for_scroll_view: Rc<dyn Component> = implicit_document.clone();
        let implicit_scroll_view = ScrollView::new(
            implicit_document_for_scroll_view,
            ScrollViewOptions {
                follow: Some(FollowMode::End),
                primary: true,
                ..ScrollViewOptions::default()
            },
        );
        let core = Rc::new_cyclic(|core_weak: &Weak<TuiAltScreenCore>| TuiAltScreenCore {
            self_weak: core_weak.clone(),
            base: RefCell::new(Weak::new()),
            env_lookup: config.env_lookup.unwrap_or_else(default_env_lookup),
            is_windows: config
                .is_windows
                .unwrap_or_else(|| Rc::new(|| cfg!(windows))),
            previous_screen: RefCell::new(Vec::new()),
            last_document: RefCell::new(Vec::new()),
            previous_screen_width: Cell::new(0),
            previous_screen_height: Cell::new(0),
            layout_root: RefCell::new(None),
            current_layout: RefCell::new(None),
            implicit_document: Rc::clone(&implicit_document),
            implicit_scroll_view: Rc::clone(&implicit_scroll_view),
            flashes: AltScreenFlashContainer::new(),
            alt_screen_active: Cell::new(false),
            image_protocol: Cell::new(None),
            saved_capabilities: RefCell::new(None),
            uploaded_kitty_images: RefCell::new(Vec::new()),
            selection_anchor: RefCell::new(None),
            selection_focus: RefCell::new(None),
            selection_granularity: Cell::new(SelectionGranularity::Character),
            selection_initial_range: RefCell::new(None),
            last_click: RefCell::new(None),
            selection_drag_pointer: RefCell::new(None),
            selection_auto_scroll_direction: Cell::new(0),
            auto_scroll_worker: RefCell::new(None),
            auto_scroll_rx: RefCell::new(None),
            selection_press_active: Cell::new(false),
            scrollbar_drag: RefCell::new(None),
            scrollbar_hover: RefCell::new(None),
            scroll_to_end_indicator_rect: RefCell::new(None),
            active_search: RefCell::new(None),
            pressed_url: RefCell::new(None),
            selection_dragged: Cell::new(false),
            mouse_capture: RefCell::new(None),
            mouse_press_target: RefCell::new(None),
            mouse_press_point: RefCell::new(None),
            mouse_press_moved: Cell::new(false),
            last_component_click: RefCell::new(None),
            wheel_scroll_lines: config.wheel_scroll_lines.unwrap_or(1).max(1),
            mouse_enabled: config.mouse.unwrap_or(true),
            search_match_style: config
                .search_match_style
                .clone()
                .unwrap_or_else(|| Rc::new(|text| format!("\x1b[4m{text}\x1b[24m"))),
            search_current_match_style: config
                .search_current_match_style
                .clone()
                .unwrap_or_else(|| Rc::new(|text| format!("\x1b[1;7m{text}\x1b[22;27m"))),
            search_navigation_button_style: config
                .search_navigation_button_style
                .clone()
                .unwrap_or_else(|| Rc::new(|text, _hovered| text.to_string())),
            scroll_to_end_indicator: config.scroll_to_end_indicator.clone(),
            open_url: config.open_url.clone(),
            on_right_click_paste: config.on_right_click_paste.clone(),
            copy_on_select: Cell::new(config.copy_on_select.unwrap_or(true)),
            copy_selection: config.copy_selection.clone(),
        });
        Self { core }
    }

    /// Upstream's `get viewportTop`.
    #[must_use]
    pub fn viewport_top(&self) -> usize {
        self.core.primary_scroll_view().scroll_top()
    }

    /// Upstream's `get isFollowingOutput`.
    #[must_use]
    pub fn is_following_output(&self) -> bool {
        self.core.primary_scroll_view().is_following_end()
    }

    /// Upstream `getCopyOnSelect`.
    #[must_use]
    pub fn get_copy_on_select(&self) -> bool {
        self.core.copy_on_select.get()
    }

    /// Upstream `setCopyOnSelect`.
    pub fn set_copy_on_select(&self, enabled: bool) {
        self.core.copy_on_select.set(enabled);
    }

    /// Whether the fullscreen viewport has a non-empty active text
    /// selection, upstream `hasActiveSelection`.
    #[must_use]
    pub fn has_active_selection(&self) -> bool {
        self.core.get_active_selection_text().is_some()
    }

    /// Copy the active fullscreen text selection, if any, using the
    /// configured selection clipboard path, upstream
    /// `copyActiveSelectionToClipboard`.
    #[must_use]
    pub fn copy_active_selection_to_clipboard(&self) -> bool {
        self.core
            .get_active_selection_text()
            .is_some_and(|text| self.core.copy_text_to_clipboard(&text))
    }

    /// Upstream `setLayoutRoot`.
    pub fn set_layout_root(&self, component: Option<Rc<dyn Component>>) {
        self.core.set_layout_root(component);
    }

    /// Upstream `scrollBy`.
    pub fn scroll_by(&self, lines: i64) {
        self.core.scroll_by(lines);
    }

    /// Upstream `scrollToTop`.
    pub fn scroll_to_top(&self) {
        self.core.scroll_to_top();
    }

    /// Upstream `scrollToBottom`.
    pub fn scroll_to_bottom(&self) {
        self.core.scroll_to_bottom();
    }

    /// Show a transient message in the alternate-screen flash stack, upstream
    /// `flash`.
    pub fn flash(&self, message: &str, duration_ms: Option<u64>) {
        self.core.flashes.flash(message, duration_ms);
    }
}
/// The optional fields of a synthesized mouse event, upstream's `extra`
/// object spread over `wheelDelta`/`clickCount`.
#[derive(Debug, Clone, Copy, Default)]
struct MouseEventExtra {
    wheel_delta: Option<i64>,
    click_count: Option<u32>,
}

impl TuiAltScreenCore {
    /// Upstream's `MouseEventExtra`: wheel delta and click count, spelled
    /// positionally upstream and as a struct here.
    const fn mouse_event_extra(
        wheel_delta: Option<i64>,
        click_count: Option<u32>,
    ) -> MouseEventExtra {
        MouseEventExtra {
            wheel_delta,
            click_count,
        }
    }
}

impl TuiAltScreenCore {
    fn base(&self) -> Option<Rc<Tui>> {
        self.base.borrow().upgrade()
    }

    fn request_render(&self) {
        if let Some(tui) = self.base() {
            tui.request_render(false);
        }
    }

    /// The frame's designated scroll view, upstream `getPrimaryScrollView`.
    fn primary_scroll_view(&self) -> ScrollStateHandle {
        self.current_layout
            .borrow()
            .as_ref()
            .and_then(|layout| layout.primary_scroll_view.clone())
            .unwrap_or_else(|| self.implicit_scroll_view.state_handle())
    }

    fn terminal_columns(&self) -> u16 {
        self.base().map_or(0, |tui| tui.terminal_columns())
    }

    fn terminal_rows(&self) -> u16 {
        self.base().map_or(0, |tui| tui.terminal_rows())
    }

    /// Show a transient message in the alternate-screen flash stack, upstream
    /// `flash`.
    fn flash(&self, message: &str, duration_ms: Option<u64>) {
        self.flashes.flash(message, duration_ms);
    }

    /// Upstream `setLayoutRoot`.
    fn set_layout_root(&self, component: Option<Rc<dyn Component>>) {
        let same = match (&*self.layout_root.borrow(), &component) {
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        if same {
            return;
        }
        *self.layout_root.borrow_mut() = component;
        *self.current_layout.borrow_mut() = None;
        self.request_render();
    }

    /// Upstream `scrollBy`.
    fn scroll_by(&self, lines: i64) {
        self.primary_scroll_view().scroll_by(lines);
        self.request_render();
    }

    /// Upstream `scrollToTop`.
    fn scroll_to_top(&self) {
        self.primary_scroll_view().scroll_to_start();
        self.request_render();
    }

    /// Upstream `scrollToBottom`.
    fn scroll_to_bottom(&self) {
        self.primary_scroll_view().scroll_to_end();
        self.request_render();
    }

    fn env(&self, key: &str) -> Option<String> {
        (self.env_lookup)(key)
    }

    fn is_windows(&self) -> bool {
        (self.is_windows)()
    }

    /// The mounted children presented as one component, upstream's
    /// `implicitDocument` — the render target the layout root overrides.
    fn render_document(&self, tui: &Tui, width: usize) -> Vec<String> {
        self.layout_root
            .borrow()
            .as_ref()
            .map_or_else(|| tui.render_children(width), |root| root.render(width))
    }

    // === Lifecycle hooks, upstream's beforeTerminalStart/Stop ===

    fn before_terminal_start(&self) {
        self.stop_selection_auto_scroll();
        self.selection_press_active.set(false);
        self.stop_scrollbar_hover();
        self.stop_scrollbar_drag();
        self.flashes.dispose();
        self.alt_screen_active.set(true);
        let capabilities = get_capabilities();
        self.image_protocol.set(capabilities.images);
        self.uploaded_kitty_images.borrow_mut().clear();
        if capabilities.images == Some(ImageProtocol::Iterm2) {
            *self.saved_capabilities.borrow_mut() = Some(capabilities);
            set_capabilities(TerminalCapabilities {
                images: None,
                ..capabilities
            });
            // Upstream `this.invalidate()`: the mounted roots and overlays.
            if let Some(tui) = self.base() {
                tui.invalidate();
            }
        }
        self.last_document.borrow_mut().clear();
        *self.selection_anchor.borrow_mut() = None;
        *self.selection_focus.borrow_mut() = None;
        self.selection_granularity
            .set(SelectionGranularity::Character);
        *self.selection_initial_range.borrow_mut() = None;
        *self.last_click.borrow_mut() = None;
        *self.pressed_url.borrow_mut() = None;
        self.selection_dragged.set(false);
        self.clear_component_mouse_gesture();
        *self.last_component_click.borrow_mut() = None;
        self.reset_render_state();
        let term = self.env("TERM").unwrap_or_default().to_lowercase();
        // Multiplexers can lag when every pointer movement is forwarded.
        // Button-motion tracking preserves clicks, wheel events, selections,
        // and scrollbar dragging.
        let mouse_sequence = if self.env("TMUX").is_some()
            || self.env("ZELLIJ").is_some()
            || self.env("STY").is_some()
            || term.starts_with("tmux")
            || term.starts_with("screen")
        {
            ENABLE_BUTTON_MOTION_MOUSE
        } else {
            ENABLE_ALL_MOTION_MOUSE
        };
        if let Some(tui) = self.base() {
            tui.terminal_write(&format!(
                "{ENTER_ALT_SCREEN}{DISABLE_AUTOWRAP}{}\x1b[2J\x1b[H\x1b[?25l",
                if self.mouse_enabled {
                    mouse_sequence
                } else {
                    ""
                }
            ));
        }
    }

    fn before_terminal_stop(&self, tui: &Tui, _options: TuiStopOptions) {
        self.close_search();
        self.stop_selection_auto_scroll();
        self.selection_press_active.set(false);
        self.stop_scrollbar_hover();
        self.stop_scrollbar_drag();
        self.clear_component_mouse_gesture();
        self.flashes.dispose();
        if !self.alt_screen_active.get() {
            return;
        }
        tui.terminal_write(&format!(
            "{BEGIN_SYNCHRONIZED_OUTPUT}{}{}{ENABLE_AUTOWRAP}{END_SYNCHRONIZED_OUTPUT}",
            self.delete_kitty_images(),
            if self.mouse_enabled {
                DISABLE_MOUSE
            } else {
                ""
            }
        ));
        self.uploaded_kitty_images.borrow_mut().clear();
    }

    fn after_terminal_stop(&self, options: TuiStopOptions) {
        if !self.alt_screen_active.get() {
            return;
        }
        self.alt_screen_active.set(false);
        let Some(tui) = self.base() else {
            return;
        };
        if options.preserve_screen {
            tui.terminal_write(&format!(
                "{BEGIN_SYNCHRONIZED_OUTPUT}{EXIT_ALT_SCREEN}\x1b[?25h{END_SYNCHRONIZED_OUTPUT}"
            ));
        } else {
            let width = usize::from(tui.terminal_columns()).max(1);
            let document: Vec<String> = tui
                .apply_line_resets(
                    self.render_document(&tui, width)
                        .into_iter()
                        .map(|line| OSC133_ZONE_PREFIX.replace(&line, "").into_owned())
                        .map(|line| line.replace(CURSOR_MARKER, ""))
                        .collect(),
                )
                .into_iter()
                .map(|line| {
                    if is_image_line(&line) || visible_width(&line) <= width {
                        line
                    } else {
                        slice_by_column(&line, 0, width, true)
                    }
                })
                .collect();
            *self.last_document.borrow_mut() = document;
            let last_document = self.last_document.borrow();
            let mut buffer =
                format!("{BEGIN_SYNCHRONIZED_OUTPUT}{EXIT_ALT_SCREEN}{DISABLE_AUTOWRAP}");
            for (row, line) in last_document.iter().enumerate() {
                if row > 0 {
                    buffer.push_str("\r\n");
                }
                buffer.push_str("\r\x1b[2K");
                buffer.push_str(line);
            }
            buffer.push_str("\x1b[0m");
            buffer.push_str(ENABLE_AUTOWRAP);
            buffer.push_str("\r\n\x1b[?25h");
            buffer.push_str(END_SYNCHRONIZED_OUTPUT);
            tui.terminal_write(&buffer);
        }
        if let Some(saved) = self.saved_capabilities.borrow_mut().take() {
            set_capabilities(saved);
        }
    }

    /// The deletion sequence the renderer emits for its Kitty uploads,
    /// upstream `deleteKittyImages`.
    fn delete_kitty_images(&self) -> String {
        if self.image_protocol.get() == Some(ImageProtocol::Kitty) {
            delete_all_kitty_images()
        } else {
            String::new()
        }
    }

    fn reset_render_state(&self) {
        self.previous_screen.borrow_mut().clear();
        self.previous_screen_width.set(0);
        self.previous_screen_height.set(0);
        *self.current_layout.borrow_mut() = None;
    }

    // === Kitty placement cache, upstream's prepareKittyScreen ===

    fn uploaded_get(&self, image_id: u64) -> Option<CachedKittyImage> {
        self.uploaded_kitty_images
            .borrow()
            .iter()
            .find(|(id, _)| *id == image_id)
            .map(|(_, cached)| *cached)
    }

    fn uploaded_set(&self, image_id: u64, cached: CachedKittyImage) {
        let mut cache = self.uploaded_kitty_images.borrow_mut();
        cache.retain(|(id, _)| *id != image_id);
        cache.push((image_id, cached));
    }

    fn uploaded_remove(&self, image_id: u64) -> Option<CachedKittyImage> {
        let mut cache = self.uploaded_kitty_images.borrow_mut();
        let index = cache.iter().position(|(id, _)| *id == image_id)?;
        Some(cache.remove(index).1)
    }

    /// Refresh the upload cache against the next screen, upstream
    /// `prepareKittyScreen`: refresh each visible placement's recency,
    /// replace re-transmitted lines with their placement-only commands, and
    /// evict offscreen uploads while the cache exceeds its three quotas.
    fn prepare_kitty_screen(&self, screen: &[String]) -> (Vec<String>, String) {
        let mut visible_image_ids: HashSet<u64> = HashSet::new();
        let mut lines = Vec::with_capacity(screen.len());
        for line in screen {
            let Some(placement) = get_kitty_image_placement(line) else {
                lines.push(line.clone());
                continue;
            };
            visible_image_ids.insert(placement.image_id);

            let cached = self.uploaded_get(placement.image_id);
            let next_cached = CachedKittyImage {
                transmission_generation: placement.transmission_generation,
                transmission_bytes: placement.transmission_bytes,
                estimated_decoded_bytes: placement.estimated_decoded_bytes,
            };
            self.uploaded_set(placement.image_id, next_cached);

            lines.push(
                if cached.is_some_and(|cached| {
                    cached.transmission_generation == placement.transmission_generation
                }) {
                    placement.replacement_line
                } else {
                    line.clone()
                },
            );
        }

        let mut evicted_image_deletion = String::new();
        loop {
            let (count, transmission_bytes, decoded_bytes) =
                self.offscreen_cache_totals(&visible_image_ids);
            if count <= MAX_CACHED_OFFSCREEN_KITTY_IMAGES
                && transmission_bytes <= MAX_CACHED_OFFSCREEN_KITTY_TRANSMISSION_BYTES
                && decoded_bytes <= MAX_CACHED_OFFSCREEN_KITTY_DECODED_BYTES
            {
                break;
            }
            let Some(image_id) = self
                .uploaded_kitty_images
                .borrow()
                .iter()
                .find(|(image_id, _)| !visible_image_ids.contains(image_id))
                .map(|(image_id, _)| *image_id)
            else {
                break;
            };
            if self.uploaded_remove(image_id).is_none() {
                break;
            }
            evicted_image_deletion.push_str(&delete_kitty_image(image_id));
        }
        (lines, evicted_image_deletion)
    }

    /// The count and byte totals of the cache entries not currently visible,
    /// upstream's `cachedOffscreenImageCount` accumulation.
    fn offscreen_cache_totals(&self, visible_image_ids: &HashSet<u64>) -> (usize, usize, u64) {
        let cache = self.uploaded_kitty_images.borrow();
        let mut count: usize = 0;
        let mut transmission_bytes: usize = 0;
        let mut decoded_bytes: u64 = 0;
        for (image_id, cached) in cache.iter() {
            if visible_image_ids.contains(image_id) {
                continue;
            }
            count += 1;
            transmission_bytes = transmission_bytes.saturating_add(cached.transmission_bytes);
            decoded_bytes = decoded_bytes.saturating_add(cached.estimated_decoded_bytes);
        }
        (count, transmission_bytes, decoded_bytes)
    }

    // === Transcript search, upstream's search members ===

    fn scroll_to_prompt(&self, direction: i8) {
        if self.base().is_none() {
            return;
        }
        let layout_guard = self.current_layout.borrow();
        let Some(layout) = layout_guard.as_ref() else {
            return;
        };
        let scroll_view = self.primary_scroll_view();
        let Some(lines) = get_scroll_view_box(layout, &scroll_view)
            .and_then(|box_| box_.scroll_content_lines.clone())
        else {
            return;
        };
        let mut row = line_i64(scroll_view.scroll_top()) + i64::from(direction);
        while row >= 0 && row < line_i64(lines.len()) {
            let line = lines.get(span_usize(row)).map_or("", String::as_str);
            if !OSC133_PROMPT_START.is_match(line) {
                row += i64::from(direction);
                continue;
            }
            scroll_view.scroll_to(span_usize(row), ScrollViewScrollToOptions::default());
            self.request_render();
            return;
        }
    }

    fn toggle_search(&self) {
        if self.active_search.borrow().is_some() {
            self.close_search();
            return;
        }
        let on_query_change: Rc<dyn Fn(&str)> = {
            let core = self.self_weak.clone();
            Rc::new(move |query: &str| {
                if let Some(core) = core.upgrade() {
                    core.update_search_query(query);
                }
            })
        };
        let component = AltScreenSearchComponent::new(
            on_query_change,
            Some(Rc::clone(&self.search_navigation_button_style)),
        );
        let mut search = ActiveSearch {
            component: Rc::clone(&component),
            index: AltScreenSearchIndex::default(),
            overlay: None,
            query: String::new(),
            matches: Vec::new(),
            selected_index: -1,
            selected_key: None,
            anchor_row: self.primary_scroll_view().scroll_top(),
            selection_mode: SearchSelectionMode::Query,
        };
        search.overlay = self.base().map(|tui| {
            tui.show_overlay(
                component,
                Some(OverlayOptions {
                    anchor: Some(OverlayAnchor::TopRight),
                    width: Some(SizeValue::Percent(40.0)),
                    min_width: Some(32),
                    margin: Some(OverlayMargin::All(1)),
                    ..OverlayOptions::default()
                }),
            )
        });
        *self.active_search.borrow_mut() = Some(search);
    }

    fn close_search(&self) {
        let Some(search) = self.active_search.borrow_mut().take() else {
            return;
        };
        if let Some(overlay) = &search.overlay {
            overlay.hide();
        }
        self.request_render();
    }

    fn update_search_query(&self, query: &str) {
        let mut guard = self.active_search.borrow_mut();
        let Some(search) = guard.as_mut() else {
            return;
        };
        if query == search.query {
            return;
        }
        let selected_row = search
            .matches
            .get(usize::try_from(search.selected_index).unwrap_or(usize::MAX))
            .and_then(|selected| selected.segments.first())
            .map_or_else(
                || self.primary_scroll_view().scroll_top(),
                |segment| segment.row,
            );
        search.anchor_row = selected_row;
        search.query = query.to_string();
        search.selection_mode = SearchSelectionMode::Query;
        search.component.set_result(-1, 0);
        self.request_render();
    }

    fn navigate_search(&self, direction: i8) {
        let mut guard = self.active_search.borrow_mut();
        let Some(search) = guard.as_mut() else {
            return;
        };
        if search.query.is_empty() {
            return;
        }
        search.selection_mode = if direction < 0 {
            SearchSelectionMode::Previous
        } else {
            SearchSelectionMode::Next
        };
        self.request_render();
    }

    fn get_search_navigation_direction_at(&self, x: u16, y: u16) -> Option<i8> {
        let search = self.active_search.borrow();
        let search = search.as_ref()?;
        let bounds = search.overlay.as_ref()?.get_bounds()?;
        let (usize_x, usize_y) = (usize::from(x), usize::from(y));
        if usize_x < bounds.col
            || usize_x >= bounds.col + bounds.width
            || usize_y < bounds.row
            || usize_y >= bounds.row + bounds.height
        {
            return None;
        }
        search
            .component
            .get_navigation_direction_at(usize_y - bounds.row, i64::from(x) - line_i64(bounds.col))
    }

    fn handle_search_mouse_event(&self, tui: &Tui, event: &SgrMouseEvent) -> bool {
        let Some(search) = self
            .active_search
            .borrow()
            .as_ref()
            .map(|search| Rc::clone(&search.component))
        else {
            return false;
        };
        let direction = self.get_search_navigation_direction_at(event.x, event.y);
        if search.set_hovered_navigation_direction(direction) {
            tui.request_render(false);
        }
        if direction.is_none() || event.release || event.button & 32 != 0 || event.button & 3 != 0 {
            return false;
        }
        self.navigate_search(direction.unwrap_or_default());
        true
    }

    /// Reconcile the search against the fresh layout, upstream
    /// `refreshSearch`; answers whether the reveal scrolled the view.
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's refreshSearch block for block; splitting it would break the 1:1 correspondence"
    )]
    fn refresh_search(&self, layout: &LayoutFrame) -> bool {
        let mut guard = self.active_search.borrow_mut();
        let Some(search) = guard.as_mut() else {
            return false;
        };
        let scroll_view = layout
            .primary_scroll_view
            .clone()
            .unwrap_or_else(|| self.implicit_scroll_view.state_handle());
        let lines = get_scroll_view_box(layout, &scroll_view)
            .and_then(|box_| box_.scroll_content_lines.clone());
        if lines
            .as_ref()
            .is_none_or(|_| search.query.trim().is_empty())
        {
            search.matches.clear();
            search.selected_index = -1;
            search.selected_key = None;
            search.selection_mode = SearchSelectionMode::Retain;
            search.component.set_result(-1, 0);
            return false;
        }
        let Some(lines) = lines else {
            return false;
        };

        let should_reveal_selection = search.selection_mode != SearchSelectionMode::Retain;
        let result = search.index.search(&lines, &search.query);
        search.matches.clone_from(&result.matches);
        let matches = result.matches;
        if !result.changed && search.selection_mode == SearchSelectionMode::Retain {
            return false;
        }
        let exact_index = if result.changed {
            search.selected_key.as_ref().map_or(-1, |key| {
                matches
                    .iter()
                    .position(|match_| &get_alt_screen_search_match_key(match_) == key)
                    .map_or(-1, |position| i64::try_from(position).unwrap_or(-1))
            })
        } else {
            search.selected_index
        };
        let mut selected_index: i64 = -1;
        if !matches.is_empty() {
            let match_count = i64::try_from(matches.len()).unwrap_or(i64::MAX);
            match search.selection_mode {
                SearchSelectionMode::Query => {
                    let mut low: usize = 0;
                    let mut high: usize = matches.len();
                    while low < high {
                        let middle = low + (high - low) / 2;
                        let row = matches[middle]
                            .segments
                            .first()
                            .map_or(0, |segment| segment.row);
                        if row < search.anchor_row {
                            low = middle + 1;
                        } else {
                            high = middle;
                        }
                    }
                    selected_index = if low < matches.len() {
                        i64::try_from(low).unwrap_or(0)
                    } else {
                        0
                    };
                }
                SearchSelectionMode::Next => {
                    let base_index = if exact_index >= 0 {
                        exact_index
                    } else {
                        (search.selected_index).min(match_count - 1)
                    };
                    selected_index = if base_index < 0 {
                        0
                    } else {
                        (base_index + 1) % match_count
                    };
                }
                SearchSelectionMode::Previous => {
                    let base_index = if exact_index >= 0 {
                        exact_index
                    } else {
                        (search.selected_index).min(match_count - 1)
                    };
                    selected_index = if base_index < 0 {
                        match_count - 1
                    } else {
                        (base_index - 1 + match_count) % match_count
                    };
                }
                SearchSelectionMode::Retain => {
                    selected_index = if exact_index >= 0 {
                        exact_index
                    } else {
                        search.selected_index.max(0).min(match_count - 1)
                    };
                }
            }
        }

        search.selected_index = selected_index;
        search.selected_key = usize::try_from(selected_index)
            .ok()
            .and_then(|index| matches.get(index).map(get_alt_screen_search_match_key));
        search.selection_mode = SearchSelectionMode::Retain;
        search.component.set_result(selected_index, matches.len());
        if !should_reveal_selection {
            return false;
        }

        let selected = usize::try_from(selected_index)
            .ok()
            .and_then(|index| matches.get(index));
        let Some(selected) = selected else {
            return false;
        };
        if get_scroll_view_box(layout, &scroll_view).is_none() {
            return false;
        }
        let (Some(first_segment), Some(last_segment)) =
            (selected.segments.first(), selected.segments.last())
        else {
            return false;
        };
        if scroll_view.viewport_height() == 0 {
            return false;
        }
        let before = scroll_view.scroll_top();
        let visible_bottom = before + scroll_view.viewport_height().saturating_sub(1);
        let target = if first_segment.row < before || last_segment.row > visible_bottom {
            first_segment
                .row
                .saturating_sub(scroll_view.viewport_height() / 3)
        } else {
            before
        };
        scroll_view.scroll_to(
            target,
            ScrollViewScrollToOptions {
                disable_follow: true,
            },
        );
        scroll_view.scroll_top() != before
    }

    // === Input pipeline, upstream's handleViewportInput ===

    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's handleViewportInput block for block; splitting it would break the 1:1 correspondence"
    )]
    fn handle_viewport_input(&self, tui: &Tui, data: &str) -> Option<TuiInputListenerResult> {
        if data == FOCUS_OUT {
            let had_active_selection = self.selection_press_active.get();
            let had_non_empty_active_selection =
                had_active_selection && self.get_selection_bounds().is_some();
            self.selection_press_active.set(false);
            self.stop_selection_auto_scroll();
            self.stop_scrollbar_hover();
            let hover_changed = self
                .active_search
                .borrow()
                .as_ref()
                .is_some_and(|search| search.component.set_hovered_navigation_direction(None));
            if hover_changed {
                self.request_render();
            }
            self.stop_scrollbar_drag();
            *self.pressed_url.borrow_mut() = None;
            self.selection_dragged.set(false);
            self.clear_component_mouse_gesture();
            *self.last_component_click.borrow_mut() = None;
            if had_active_selection {
                *self.selection_anchor.borrow_mut() = None;
                *self.selection_focus.borrow_mut() = None;
                self.selection_granularity
                    .set(SelectionGranularity::Character);
                *self.selection_initial_range.borrow_mut() = None;
                if had_non_empty_active_selection {
                    self.request_render();
                }
            }
            *self.last_click.borrow_mut() = None;
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if data == FOCUS_IN {
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }

        if let Some(wheel_event) = self.parse_wheel_event(data) {
            let event = self.create_mouse_event(
                tui,
                TuiMouseEventType::Wheel,
                wheel_event.button,
                wheel_event.x,
                wheel_event.y,
                Self::mouse_event_extra(
                    Some(
                        i64::from(wheel_event.direction)
                            * self.get_wheel_scroll_lines(wheel_event.button),
                    ),
                    None,
                ),
            );
            let overlay = tui.dispatch_mouse_to_overlay(&event);
            let result = overlay.result.clone().or_else(|| {
                if overlay.hit {
                    None
                } else {
                    self.dispatch_mouse_to_layout(&event)
                }
            });
            if let Some(result) = result {
                if self.apply_mouse_dispatch_result(tui, &event, &result) {
                    tui.request_render(false);
                }
                return Some(TuiInputListenerResult {
                    consume: true,
                    data: None,
                });
            }
            if self.should_defer_viewport_input_to_overlay() {
                return None;
            }
            self.route_wheel(tui, wheel_event);
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if let Some(mouse_event) = self.parse_sgr_mouse_event(data) {
            self.handle_mouse_event(tui, mouse_event);
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if self.is_mouse_sequence(data) {
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }

        let keybindings = get_keybindings();
        let parser = tui.key_parser();
        let parser = parser
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let is_release = is_key_release(data);
        let matches = |action: &str| keybindings.matches(&parser, data, action);
        if matches("tui.altScreen.search") {
            if !is_release {
                self.toggle_search();
            }
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if self.active_search.borrow().as_ref().is_some_and(|search| {
            search
                .overlay
                .as_ref()
                .is_some_and(OverlayHandle::is_focused)
        }) {
            if matches("tui.altScreen.searchNext") {
                if !is_release {
                    self.navigate_search(1);
                }
                return Some(TuiInputListenerResult {
                    consume: true,
                    data: None,
                });
            }
            if matches("tui.altScreen.searchPrevious") {
                if !is_release {
                    self.navigate_search(-1);
                }
                return Some(TuiInputListenerResult {
                    consume: true,
                    data: None,
                });
            }
            if matches("tui.altScreen.searchClose") {
                if !is_release {
                    self.close_search();
                }
                return Some(TuiInputListenerResult {
                    consume: true,
                    data: None,
                });
            }
        }
        if self.should_defer_viewport_input_to_overlay() {
            return None;
        }
        if matches("tui.altScreen.pageUp") {
            if !is_release {
                let delta = self
                    .primary_scroll_view()
                    .viewport_height()
                    .saturating_sub(PAGE_SCROLL_OVERLAP);
                self.scroll_by(-i64::from(u32::try_from(delta.max(1)).unwrap_or(u32::MAX)));
            }
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if matches("tui.altScreen.pageDown") {
            if !is_release {
                let delta = self
                    .primary_scroll_view()
                    .viewport_height()
                    .saturating_sub(PAGE_SCROLL_OVERLAP);
                self.scroll_by(i64::from(u32::try_from(delta.max(1)).unwrap_or(u32::MAX)));
            }
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if matches("tui.altScreen.halfPageUp") {
            if !is_release {
                let delta = self.primary_scroll_view().viewport_height() / 2;
                self.scroll_by(-i64::from(u32::try_from(delta.max(1)).unwrap_or(u32::MAX)));
            }
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if matches("tui.altScreen.halfPageDown") {
            if !is_release {
                let delta = self.primary_scroll_view().viewport_height() / 2;
                self.scroll_by(i64::from(u32::try_from(delta.max(1)).unwrap_or(u32::MAX)));
            }
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if matches("tui.altScreen.lineUp") {
            if !is_release {
                self.scroll_by(-1);
            }
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if matches("tui.altScreen.lineDown") {
            if !is_release {
                self.scroll_by(1);
            }
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if matches("tui.altScreen.previousPrompt") {
            if !is_release {
                self.scroll_to_prompt(-1);
            }
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if matches("tui.altScreen.nextPrompt") {
            if !is_release {
                self.scroll_to_prompt(1);
            }
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if matches("tui.altScreen.top") {
            if !is_release {
                self.scroll_to_top();
            }
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        if matches("tui.altScreen.bottom") {
            if !is_release {
                self.scroll_to_bottom();
            }
            return Some(TuiInputListenerResult {
                consume: true,
                data: None,
            });
        }
        None
    }

    fn should_defer_viewport_input_to_overlay(&self) -> bool {
        let overlay_focused = self.base().is_some_and(|tui| tui.is_overlay_focused());
        let search_focused = self
            .active_search
            .borrow()
            .as_ref()
            .and_then(|search| search.overlay.as_ref())
            .is_some_and(OverlayHandle::is_focused);
        overlay_focused && !search_focused
    }

    fn clear_component_mouse_gesture(&self) {
        *self.mouse_capture.borrow_mut() = None;
        *self.mouse_press_target.borrow_mut() = None;
        *self.mouse_press_point.borrow_mut() = None;
        self.mouse_press_moved.set(false);
    }

    fn clear_text_selection(&self) {
        self.stop_selection_auto_scroll();
        self.selection_press_active.set(false);
        *self.selection_anchor.borrow_mut() = None;
        *self.selection_focus.borrow_mut() = None;
        self.selection_granularity
            .set(SelectionGranularity::Character);
        *self.selection_initial_range.borrow_mut() = None;
        *self.pressed_url.borrow_mut() = None;
        self.selection_dragged.set(false);
    }

    #[expect(
        clippy::unused_self,
        reason = "upstream's createMouseEvent reads no instance state beyond the base; the port keeps the method shape"
    )]
    fn create_mouse_event(
        &self,
        tui: &Tui,
        event_type: TuiMouseEventType,
        button: u32,
        x: u16,
        y: u16,
        extra: MouseEventExtra,
    ) -> TuiMouseEvent {
        TuiMouseEvent {
            event_type,
            button: if event_type == TuiMouseEventType::Wheel {
                TuiMouseButton::None
            } else {
                decode_mouse_button(button)
            },
            x,
            y,
            screen_x: x,
            screen_y: y,
            width: tui.terminal_columns().max(1),
            height: tui.terminal_rows().max(1),
            shift: button & 4 != 0,
            alt: button & 8 != 0,
            ctrl: button & 16 != 0,
            wheel_delta: extra.wheel_delta.map(|delta| {
                i32::try_from(delta).unwrap_or(if delta < 0 { i32::MIN } else { i32::MAX })
            }),
            click_count: extra.click_count,
        }
    }

    fn dispatch_mouse_to_layout(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        let layout_guard = self.current_layout.borrow();
        let layout = layout_guard.as_ref()?;
        let mut visited: HashSet<usize> = HashSet::new();
        for box_ in
            get_layout_boxes_at(layout, i64::from(event.screen_x), i64::from(event.screen_y))
        {
            let component = &box_.component;
            let ptr = Rc::as_ptr(component).cast::<()>() as usize;
            if visited.contains(&ptr) {
                continue;
            }
            if component.layout_node().is_some() && component.is_stock_mouse_container() {
                continue;
            }
            visited.insert(ptr);
            let local = TuiMouseEvent {
                x: local_coord(i64::from(event.screen_x) - box_.rect.x),
                y: local_coord(i64::from(event.screen_y) - box_.rect.y),
                width: local_coord(box_.rect.width),
                height: local_coord(box_.rect.height),
                ..event.clone()
            };
            if let Some(result) = dispatch_mouse_event(component, &local) {
                return Some(result);
            }
        }
        None
    }

    fn apply_mouse_dispatch_result(
        &self,
        tui: &Tui,
        event: &TuiMouseEvent,
        result: &TuiMouseEventResult,
    ) -> bool {
        let focus_component = result.focus_target.clone().or_else(|| {
            result
                .target
                .as_ref()
                .map(|target| Rc::clone(&target.component))
        });
        let focus_changed = focus_component.as_ref().is_some_and(|focus_component| {
            let focus_target = tui.resolve_mouse_focus_target(focus_component);
            let changed = result.focus
                && !component_matches(tui.get_focused_component().as_ref(), &focus_target);
            if result.focus {
                tui.set_focus(Some(focus_target));
            }
            changed
        });
        if result.capture {
            if let Some(target) = &result.target {
                self.mouse_capture
                    .borrow_mut()
                    .clone_from(&Some(target.clone()));
            } else {
                *self.mouse_capture.borrow_mut() = None;
            }
        }
        result.render.unwrap_or(
            focus_changed
                || matches!(
                    event.event_type,
                    TuiMouseEventType::Press
                        | TuiMouseEventType::Click
                        | TuiMouseEventType::Drag
                        | TuiMouseEventType::Wheel
                ),
        )
    }

    #[expect(
        clippy::unused_self,
        reason = "upstream's dispatchMouseToTarget reads no instance state; the port keeps the method shape"
    )]
    fn dispatch_mouse_to_target(
        &self,
        event: &TuiMouseEvent,
        target: &TuiMouseDispatchTarget,
    ) -> Option<TuiMouseEventResult> {
        dispatch_mouse_event(&target.component, &retarget_mouse_event(event, target))
    }

    fn get_component_click_count(&self, target: &TuiMouseDispatchTarget, x: u16, y: u16) -> u32 {
        let now = epoch_millis();
        let previous = (*self.last_component_click.borrow()).clone();
        let count = previous
            .filter(|previous| {
                now.saturating_sub(previous.timestamp) <= DOUBLE_CLICK_INTERVAL_MS
                    && component_matches(Some(&previous.component), &target.component)
                    && previous.x == x
                    && previous.y == y
            })
            .map_or(1, |previous| (previous.count % 3) + 1);
        *self.last_component_click.borrow_mut() = Some(ComponentClickTarget {
            timestamp: now,
            count,
            component: Rc::clone(&target.component),
            x,
            y,
        });
        count
    }

    // === Mouse handling, upstream's handleMouseEvent ===

    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's handleMouseEvent block for block; splitting it would break the 1:1 correspondence"
    )]
    fn handle_mouse_event(&self, tui: &Tui, raw: SgrMouseEvent) {
        let is_motion = raw.button & 32 != 0;
        let event_type = if raw.release {
            TuiMouseEventType::Release
        } else if is_motion {
            if decode_mouse_button(raw.button) == TuiMouseButton::None {
                TuiMouseEventType::Move
            } else {
                TuiMouseEventType::Drag
            }
        } else {
            TuiMouseEventType::Press
        };
        let event = self.create_mouse_event(
            tui,
            event_type,
            raw.button,
            raw.x,
            raw.y,
            Self::mouse_event_extra(None, None),
        );

        let captured = self
            .mouse_capture
            .borrow()
            .clone()
            .or_else(|| (*self.mouse_press_target.borrow()).clone());
        if let Some(target) = captured {
            let press_point = *self.mouse_press_point.borrow();
            if let Some(point) = press_point
                && (raw.x != point.0 || raw.y != point.1)
            {
                self.mouse_press_moved.set(true);
                *self.last_component_click.borrow_mut() = None;
            }
            let mut render =
                self.dispatch_mouse_to_target(&event, &target)
                    .is_some_and(|target_result| {
                        self.apply_mouse_dispatch_result(tui, &event, &target_result)
                    });
            if raw.release {
                if !self.mouse_press_moved.get() && press_point == Some((raw.x, raw.y)) {
                    let click_event = self.create_mouse_event(
                        tui,
                        TuiMouseEventType::Click,
                        raw.button,
                        raw.x,
                        raw.y,
                        Self::mouse_event_extra(
                            None,
                            Some(self.get_component_click_count(&target, raw.x, raw.y)),
                        ),
                    );
                    if let Some(click_result) = self.dispatch_mouse_to_target(&click_event, &target)
                    {
                        render = self.apply_mouse_dispatch_result(tui, &click_event, &click_result)
                            || render;
                    }
                }
                self.clear_component_mouse_gesture();
            }
            if render {
                tui.request_render(false);
            }
            return;
        }

        if self.handle_search_mouse_event(tui, &raw) {
            return;
        }

        let overlay = tui.dispatch_mouse_to_overlay(&event);
        if overlay.hit {
            self.stop_scrollbar_hover();
        } else {
            if self.handle_scroll_to_end_indicator_mouse_event(&raw) {
                return;
            }
            let scrollbar_handled = self.handle_scrollbar_mouse_event(&raw);
            if self.scrollbar_drag.borrow().is_none() {
                self.update_scrollbar_hover(usize::from(raw.x), usize::from(raw.y));
            }
            if scrollbar_handled {
                return;
            }
        }

        let result = overlay.result.clone().or_else(|| {
            if overlay.hit {
                None
            } else {
                self.dispatch_mouse_to_layout(&event)
            }
        });
        if let Some(result) = &result {
            let render = self.apply_mouse_dispatch_result(tui, &event, result);
            if event_type == TuiMouseEventType::Press {
                self.clear_text_selection();
                self.mouse_press_target
                    .borrow_mut()
                    .clone_from(&result.target);
                *self.mouse_press_point.borrow_mut() = Some((raw.x, raw.y));
                self.mouse_press_moved.set(false);
            }
            if render {
                tui.request_render(false);
            }
            return;
        }

        if self.handle_right_click_paste(&raw) {
            return;
        }
        self.handle_selection_mouse_event(tui, &raw);
    }

    #[expect(
        clippy::unused_self,
        reason = "upstream's parseWheelEvent reads no instance state; the port keeps the method shape"
    )]
    fn parse_wheel_event(&self, data: &str) -> Option<WheelEvent> {
        if let Some(captures) = SGR_MOUSE.captures(data) {
            let button = captures[1].parse::<u32>().ok()?;
            if button & 64 == 0 {
                return None;
            }
            let direction = button & 3;
            if direction != 0 && direction != 1 {
                return None;
            }
            return Some(WheelEvent {
                direction: if direction == 0 { -1 } else { 1 },
                x: mouse_coord(i64::from(parse_u32(&captures[2])) - 1),
                y: mouse_coord(i64::from(parse_u32(&captures[3])) - 1),
                button,
            });
        }
        if data.chars().count() == 6 && data.starts_with("\x1b[M") {
            let chars: Vec<char> = data.chars().collect();
            let button = u32::from(chars[3]) - 32;
            if button & 64 == 0 {
                return None;
            }
            let direction = button & 3;
            if direction != 0 && direction != 1 {
                return None;
            }
            return Some(WheelEvent {
                direction: if direction == 0 { -1 } else { 1 },
                x: mouse_coord(i64::from(u32::from(chars[4])) - 33),
                y: mouse_coord(i64::from(u32::from(chars[5])) - 33),
                button,
            });
        }
        None
    }

    fn get_wheel_scroll_lines(&self, button: u32) -> i64 {
        // SGR mouse button codes use bit 3 (value 8) for the Alt modifier.
        let lines = i64::from(self.wheel_scroll_lines);
        if button & 8 != 0 {
            lines * ALT_WHEEL_SCROLL_MULTIPLIER
        } else {
            lines
        }
    }

    fn route_wheel(&self, tui: &Tui, event: WheelEvent) {
        let mut remaining = i64::from(event.direction) * self.get_wheel_scroll_lines(event.button);
        let mut seen: HashSet<usize> = HashSet::new();
        let scroll_views = self
            .current_layout
            .borrow()
            .as_ref()
            .map_or_else(Vec::new, |layout| {
                get_scroll_views_at(layout, usize::from(event.x), usize::from(event.y))
            });
        for scroll_view in scroll_views {
            seen.insert(std::sync::Arc::as_ptr(&scroll_view).cast::<()>() as usize);
            remaining = scroll_view.scroll_by(remaining);
            if remaining == 0 || scroll_view.overscroll() == Overscroll::Contain {
                break;
            }
        }
        let primary = self.primary_scroll_view();
        if remaining != 0
            && !seen.contains(&(std::sync::Arc::as_ptr(&primary).cast::<()>() as usize))
        {
            primary.scroll_by(remaining);
        }
        self.update_scrollbar_hover(usize::from(event.x), usize::from(event.y));
        tui.request_render(false);
    }

    #[expect(
        clippy::unused_self,
        reason = "upstream's parseSgrMouseEvent reads no instance state; the port keeps the method shape"
    )]
    fn parse_sgr_mouse_event(&self, data: &str) -> Option<SgrMouseEvent> {
        let captures = SGR_MOUSE.captures(data)?;
        Some(SgrMouseEvent {
            button: captures[1].parse().ok()?,
            x: mouse_coord(i64::from(parse_u32(&captures[2])) - 1),
            y: mouse_coord(i64::from(parse_u32(&captures[3])) - 1),
            release: &captures[4] == "m",
        })
    }

    fn handle_right_click_paste(&self, event: &SgrMouseEvent) -> bool {
        if self.on_right_click_paste.is_none()
            || !self.is_windows()
            || self
                .env("TERM_PROGRAM")
                .is_some_and(|program| program.to_lowercase() == "vscode")
            || event.release
            || event.button != 2
        {
            return false;
        }
        if let Some(on_right_click_paste) = &self.on_right_click_paste {
            // Upstream wrapped the handler in try/catch — clipboard paste is
            // best-effort; a Rust handler signals failure through its return
            // instead of throwing.
            on_right_click_paste();
        }
        true
    }

    fn handle_scroll_to_end_indicator_mouse_event(&self, event: &SgrMouseEvent) -> bool {
        let Some(rect) = *self.scroll_to_end_indicator_rect.borrow() else {
            return false;
        };
        if event.release || event.button & 32 != 0 || event.button & 3 != 0 {
            return false;
        }
        if usize::from(event.y) != rect.row
            || usize::from(event.x) < rect.column
            || usize::from(event.x) >= rect.column + rect.width
        {
            return false;
        }
        self.scroll_to_bottom();
        true
    }

    // === Scrollbar handling, upstream's scrollbar members ===

    fn get_scrollbar_target_at(
        &self,
        x: usize,
        y: usize,
        include_hidden_auto: bool,
    ) -> Option<ScrollbarTarget> {
        let overlay = self.base().is_some_and(|tui| tui.has_overlay());
        if overlay {
            return None;
        }
        let layout = self.current_layout.borrow();
        let layout = layout.as_ref()?;
        for scroll_view in get_scroll_views_at(layout, x, y) {
            let geometry = get_scroll_view_box(layout, &scroll_view)
                .and_then(|box_| get_scrollbar_geometry(box_, include_hidden_auto));
            if let Some(geometry) = geometry
                && x == geometry.column
                && y >= geometry.track_top
                && y < geometry.track_top + geometry.track_height
            {
                return Some(ScrollbarTarget {
                    scroll_view,
                    geometry,
                });
            }
        }
        None
    }

    fn set_scrollbar_hover(&self, scroll_view: Option<&ScrollStateHandle>) {
        let current = self.scrollbar_hover.borrow().clone();
        let same = same_scroll_view(current.as_ref(), scroll_view);
        if same {
            return;
        }
        if let Some(previous) = current {
            previous.set_scrollbar_active(false);
        }
        *self.scrollbar_hover.borrow_mut() = scroll_view.cloned();
        if let Some(scroll_view) = scroll_view {
            scroll_view.set_scrollbar_active(true);
        }
    }

    fn update_scrollbar_hover(&self, x: usize, y: usize) {
        let target = self
            .get_scrollbar_target_at(x, y, true)
            .map(|target| target.scroll_view);
        self.set_scrollbar_hover(target.as_ref());
    }

    fn stop_scrollbar_hover(&self) {
        self.set_scrollbar_hover(None);
    }

    #[expect(
        clippy::unused_self,
        reason = "upstream's scrollScrollbarToPointer reads no instance state; the port keeps the method shape"
    )]
    fn scroll_scrollbar_to_pointer(
        &self,
        scroll_view: &ScrollStateHandle,
        geometry: &ScrollbarGeometry,
        pointer_y: usize,
        grab_offset: usize,
    ) {
        let max_thumb_offset = geometry.track_height.saturating_sub(geometry.thumb_height);
        let thumb_offset =
            (line_i64(pointer_y) - line_i64(geometry.track_top) - line_i64(grab_offset))
                .clamp(0, line_i64(max_thumb_offset));
        let scroll_top = if max_thumb_offset == 0 {
            0
        } else {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the proportional scroll offset rounds against a bounded track"
            )]
            let top = (f64::from(u16::try_from(thumb_offset).unwrap_or(u16::MAX))
                / f64::from(u16::try_from(max_thumb_offset).unwrap_or(u16::MAX))
                * f64::from(u16::try_from(geometry.max_scroll_top).unwrap_or(u16::MAX)))
            .round() as usize;
            span_usize(line_i64(top))
        };
        scroll_view.scroll_to(scroll_top, ScrollViewScrollToOptions::default());
    }

    fn handle_scrollbar_mouse_event(&self, event: &SgrMouseEvent) -> bool {
        let drag = self
            .scrollbar_drag
            .borrow()
            .as_ref()
            .map(|drag| (std::sync::Arc::clone(&drag.scroll_view), drag.grab_offset));
        if let Some((scroll_view, grab_offset)) = drag {
            if event.release {
                self.stop_scrollbar_drag();
                return true;
            }
            let geometry = self
                .current_layout
                .borrow()
                .as_ref()
                .and_then(|layout| get_scroll_view_box(layout, &scroll_view))
                .and_then(|box_| get_scrollbar_geometry(box_, false));
            if let Some(geometry) = geometry {
                self.scroll_scrollbar_to_pointer(
                    &scroll_view,
                    &geometry,
                    usize::from(event.y),
                    grab_offset,
                );
            }
            return true;
        }

        if event.release || event.button & 32 != 0 || event.button & 3 != 0 {
            return false;
        }
        let Some(target) =
            self.get_scrollbar_target_at(usize::from(event.x), usize::from(event.y), false)
        else {
            return false;
        };
        self.stop_selection_auto_scroll();
        self.selection_press_active.set(false);
        *self.selection_anchor.borrow_mut() = None;
        *self.selection_focus.borrow_mut() = None;
        self.selection_granularity
            .set(SelectionGranularity::Character);
        *self.selection_initial_range.borrow_mut() = None;
        *self.last_click.borrow_mut() = None;
        *self.pressed_url.borrow_mut() = None;
        self.selection_dragged.set(false);
        self.set_scrollbar_hover(Some(&target.scroll_view));
        let on_thumb = usize::from(event.y) >= target.geometry.thumb_top
            && usize::from(event.y) < target.geometry.thumb_top + target.geometry.thumb_height;
        let grab_offset = if on_thumb {
            usize::from(event.y) - target.geometry.thumb_top
        } else {
            target.geometry.thumb_height / 2
        };
        if !on_thumb {
            self.scroll_scrollbar_to_pointer(
                &target.scroll_view,
                &target.geometry,
                usize::from(event.y),
                grab_offset,
            );
        }
        *self.scrollbar_drag.borrow_mut() = Some(ScrollbarDrag {
            scroll_view: target.scroll_view,
            grab_offset,
        });
        true
    }

    fn stop_scrollbar_drag(&self) {
        *self.scrollbar_drag.borrow_mut() = None;
    }

    // === Text selection, upstream's selection members ===

    fn get_scroll_selection_point(
        &self,
        scroll_view: &ScrollStateHandle,
        x: usize,
        y: usize,
    ) -> Option<SelectionPoint> {
        let layout = self.current_layout.borrow();
        let layout = layout.as_ref()?;
        let box_ = get_scroll_view_box(layout, scroll_view)?;
        if box_.rect.height <= 0 || box_.clip.height <= 0 {
            return None;
        }
        let rows = usize::from(self.terminal_rows());
        let visible_top = span_usize(0.max(box_.rect.y).max(box_.clip.y));
        let visible_bottom = span_usize(
            line_i64(rows.saturating_sub(1))
                .min(box_.rect.y + box_.rect.height - 1)
                .min(box_.clip.y + box_.clip.height - 1),
        );
        if visible_bottom < visible_top {
            return None;
        }
        let pointer_row = (line_i64(y)).clamp(line_i64(visible_top), line_i64(visible_bottom));
        let max_content_row = line_i64(box_.scroll_content_lines.as_ref().map_or(1, Vec::len))
            .saturating_sub(1)
            .max(0);
        Some(SelectionPoint {
            row: span_usize(line_i64(scroll_view.scroll_top()) + pointer_row - box_.rect.y)
                .min(span_usize(max_content_row)),
            col: span_usize(
                (line_i64(x) - box_.rect.x).clamp(0, box_.rect.width.saturating_sub(1).max(0)),
            ),
            scroll_view: Some(std::sync::Arc::clone(scroll_view)),
            boundary: false,
        })
    }

    fn get_selection_point(
        &self,
        event: &SgrMouseEvent,
        scroll_view: Option<&ScrollStateHandle>,
    ) -> SelectionPoint {
        if let Some(scroll_view) = scroll_view
            && let Some(point) = self.get_scroll_selection_point(
                scroll_view,
                usize::from(event.x),
                usize::from(event.y),
            )
        {
            return point;
        }
        SelectionPoint::screen(
            usize::from(self.terminal_rows().saturating_sub(1)).min(usize::from(event.y)),
            usize::from(self.terminal_columns().saturating_sub(1)).min(usize::from(event.x)),
        )
    }

    fn get_selection_source_line(&self, point: &SelectionPoint) -> String {
        if let Some(scroll_view) = point.scroll_view.as_ref() {
            let layout = self.current_layout.borrow();
            if let Some(lines) = layout
                .as_ref()
                .and_then(|layout| get_scroll_view_box(layout, scroll_view))
                .and_then(|box_| box_.scroll_content_lines.clone())
            {
                return lines.get(point.row).cloned().unwrap_or_default();
            }
        }
        self.previous_screen
            .borrow()
            .get(point.row)
            .cloned()
            .unwrap_or_default()
    }

    fn get_word_selection(&self, point: &SelectionPoint) -> Option<SelectionRange> {
        let line = strip_terminal_sequences(&self.get_selection_source_line(point));
        let mut segments: Vec<(usize, usize, bool, bool)> = Vec::new();
        let mut start = 0;
        for segment in word_segments(&line) {
            let end = start + visible_width(segment);
            let joiner = TERMINAL_WORD_SELECTION_JOINERS.contains(&segment);
            segments.push((start, end, is_word_like(segment) || joiner, joiner));
            start = end;
        }
        let clicked_segment_index = segments
            .iter()
            .position(|(start, end, _, _)| point.col >= *start && point.col < *end)?;
        let can_join = |left: &(usize, usize, bool, bool), right: &(usize, usize, bool, bool)| {
            left.2 && right.2 && (left.3 || right.3)
        };
        let mut selection_start = segments[clicked_segment_index].0;
        let mut selection_end = segments[clicked_segment_index].1;
        let mut index = clicked_segment_index;
        while index > 0 && can_join(&segments[index - 1], &segments[index]) {
            selection_start = segments[index - 1].0;
            index -= 1;
        }
        let mut index = clicked_segment_index;
        while index + 1 < segments.len() && can_join(&segments[index], &segments[index + 1]) {
            selection_end = segments[index + 1].1;
            index += 1;
        }
        Some(SelectionRange {
            start: point.with_col(selection_start),
            end: point.with_col(selection_end).with_boundary(),
        })
    }

    fn get_line_selection(&self, point: &SelectionPoint) -> SelectionRange {
        SelectionRange {
            start: point.with_col(0),
            end: point
                .with_col(visible_width(&self.get_selection_source_line(point)))
                .with_boundary(),
        }
    }

    fn update_selection_focus(&self, point: SelectionPoint) {
        if self.selection_granularity.get() == SelectionGranularity::Character
            || self.selection_initial_range.borrow().is_none()
        {
            *self.selection_focus.borrow_mut() = Some(point);
            return;
        }
        let range = match self.selection_granularity.get() {
            SelectionGranularity::Word => self.get_word_selection(&point),
            _ => Some(self.get_line_selection(&point)),
        };
        let Some(range) = range else {
            return;
        };
        let initial = self.selection_initial_range.borrow().clone();
        let Some(initial) = initial else {
            return;
        };
        let target_before_initial = range.start.row < initial.start.row
            || (range.start.row == initial.start.row && range.start.col < initial.start.col);
        if target_before_initial {
            *self.selection_anchor.borrow_mut() = Some(initial.end);
            *self.selection_focus.borrow_mut() = Some(range.start);
        } else {
            *self.selection_anchor.borrow_mut() = Some(initial.start);
            *self.selection_focus.borrow_mut() = Some(range.end);
        }
    }

    fn get_click_count(&self, point: &SelectionPoint, word: Option<&SelectionRange>) -> u32 {
        let now = epoch_millis();
        let previous = (*self.last_click.borrow()).clone();
        let count = word
            .and_then(|word| {
                previous.filter(|previous| {
                    now.saturating_sub(previous.timestamp) <= DOUBLE_CLICK_INTERVAL_MS
                        && previous.row == point.row
                        && same_scroll_view(
                            previous.scroll_view.as_ref(),
                            point.scroll_view.as_ref(),
                        )
                        && previous.word_start == word.start.col
                        && previous.word_end == word.end.col
                })
            })
            .map_or(1, |previous| (previous.count % 3) + 1);
        *self.last_click.borrow_mut() = word.map(|word| ClickTarget {
            timestamp: now,
            count,
            row: point.row,
            scroll_view: point.scroll_view.clone(),
            word_start: word.start.col,
            word_end: word.end.col,
        });
        count
    }

    fn update_selection_auto_scroll(&self, event: &SgrMouseEvent) {
        let Some(scroll_view) = self
            .selection_anchor
            .borrow()
            .as_ref()
            .and_then(|anchor| anchor.scroll_view.clone())
        else {
            self.stop_selection_auto_scroll();
            return;
        };
        let box_geometry = {
            let layout = self.current_layout.borrow();
            layout
                .as_ref()
                .and_then(|layout| get_scroll_view_box(layout, &scroll_view))
                .map(|box_| (box_.rect, box_.clip))
        };
        let Some((rect, clip)) = box_geometry else {
            self.stop_selection_auto_scroll();
            return;
        };
        if rect.height <= 0 || clip.height <= 0 {
            self.stop_selection_auto_scroll();
            return;
        }
        let visible_top = span_usize(0.max(rect.y).max(clip.y));
        let visible_bottom = span_usize(
            line_i64(usize::from(self.terminal_rows()).saturating_sub(1))
                .min(rect.y + rect.height - 1)
                .min(clip.y + clip.height - 1),
        );
        *self.selection_drag_pointer.borrow_mut() =
            Some((usize::from(event.x), usize::from(event.y)));
        let direction: i8 = if usize::from(event.y) <= visible_top {
            -1
        } else {
            i8::from(usize::from(event.y) >= visible_bottom)
        };
        self.selection_auto_scroll_direction.set(direction);
        if direction == 0 {
            self.stop_selection_auto_scroll();
            return;
        }
        if self.auto_scroll_worker.borrow().is_some() {
            return;
        }
        let (stop, stop_rx) = channel();
        let (tick_tx, tick_rx) = channel();
        let request_render = self.base().map(|tui| tui.render_request());
        std::thread::spawn(move || {
            loop {
                match stop_rx.recv_timeout(Duration::from_millis(SELECTION_AUTO_SCROLL_INTERVAL_MS))
                {
                    Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        let _ = tick_tx.send(());
                        if let Some(request_render) = request_render.as_ref() {
                            request_render();
                        }
                    }
                }
            }
        });
        *self.auto_scroll_worker.borrow_mut() = Some(AutoScrollWorker { stop });
        *self.auto_scroll_rx.borrow_mut() = Some(tick_rx);
    }

    /// Drain the armed auto-scroll interval's ticks on the owner thread,
    /// upstream's interval callback running on the event loop.
    fn drain_auto_scroll_ticks(&self) {
        while self
            .auto_scroll_rx
            .borrow()
            .as_ref()
            .and_then(|receiver| receiver.try_recv().ok())
            .ok_or(std::sync::mpsc::TryRecvError::Disconnected)
            == Ok(())
        {
            self.auto_scroll_selection();
        }
    }

    fn auto_scroll_selection(&self) {
        let scroll_view = self
            .selection_anchor
            .borrow()
            .as_ref()
            .and_then(|anchor| anchor.scroll_view.clone());
        let pointer = *self.selection_drag_pointer.borrow();
        let direction = self.selection_auto_scroll_direction.get();
        let (Some(scroll_view), Some(pointer)) = (scroll_view, pointer) else {
            self.stop_selection_auto_scroll();
            return;
        };
        if direction == 0 {
            self.stop_selection_auto_scroll();
            return;
        }
        let remaining = scroll_view.scroll_by(i64::from(direction));
        if remaining == i64::from(direction) {
            self.stop_selection_auto_scroll();
            return;
        }
        if let Some(point) = self.get_scroll_selection_point(&scroll_view, pointer.0, pointer.1) {
            self.update_selection_focus(point);
        }
        self.request_render();
    }

    fn stop_selection_auto_scroll(&self) {
        *self.auto_scroll_worker.borrow_mut() = None;
        *self.auto_scroll_rx.borrow_mut() = None;
        self.selection_auto_scroll_direction.set(0);
        *self.selection_drag_pointer.borrow_mut() = None;
    }

    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's handleSelectionMouseEvent block for block; splitting it would break the 1:1 correspondence"
    )]
    fn handle_selection_mouse_event(&self, tui: &Tui, event: &SgrMouseEvent) {
        let button = event.button & 3;
        if button != 0 && !(event.release && button == 3) {
            return;
        }
        let anchor_scroll_view = self
            .selection_anchor
            .borrow()
            .as_ref()
            .and_then(|anchor| anchor.scroll_view.clone());
        let point = self.get_selection_point(event, anchor_scroll_view.as_ref());
        if event.release {
            if !self.selection_press_active.get() {
                return;
            }
            self.selection_press_active.set(false);
            self.stop_selection_auto_scroll();
            let Some(anchor) = (*self.selection_anchor.borrow()).clone() else {
                return;
            };
            self.update_selection_focus(point.clone());
            let is_click = !self.selection_dragged.get()
                && same_scroll_view(anchor.scroll_view.as_ref(), point.scroll_view.as_ref())
                && anchor.row == point.row
                && anchor.col == point.col;
            let clicked_url = if is_click {
                (*self.pressed_url.borrow()).clone()
            } else {
                None
            };
            *self.pressed_url.borrow_mut() = None;
            if let (Some(url), Some(open_url)) = (clicked_url, &self.open_url) {
                *self.selection_anchor.borrow_mut() = None;
                *self.selection_focus.borrow_mut() = None;
                // Upstream wrapped the handler in try/catch — URL activation
                // is best-effort; a Rust handler signals failure through its
                // return instead of throwing.
                open_url(&url);
                tui.request_render(false);
                return;
            }
            if is_click {
                let click_count = self
                    .last_click
                    .borrow()
                    .as_ref()
                    .map_or(1, |click| click.count);
                let click_event = self.create_mouse_event(
                    tui,
                    TuiMouseEventType::Click,
                    event.button,
                    event.x,
                    event.y,
                    Self::mouse_event_extra(None, Some(click_count)),
                );
                let overlay = tui.dispatch_mouse_to_overlay(&click_event);
                let result = overlay.result.clone().or_else(|| {
                    if overlay.hit {
                        None
                    } else {
                        self.dispatch_mouse_to_layout(&click_event)
                    }
                });
                if let Some(result) = &result {
                    let render = self.apply_mouse_dispatch_result(tui, &click_event, result);
                    self.clear_text_selection();
                    if render {
                        tui.request_render(false);
                    }
                    return;
                }
            }
            if self.copy_on_select.get() {
                self.copy_selection_to_clipboard();
            }
            tui.request_render(false);
            return;
        }
        if event.button & 32 != 0 {
            if !self.selection_press_active.get() || self.selection_anchor.borrow().is_none() {
                return;
            }
            self.selection_dragged.set(true);
            *self.last_click.borrow_mut() = None;
            *self.pressed_url.borrow_mut() = None;
            self.update_selection_focus(point);
            self.update_selection_auto_scroll(event);
            tui.request_render(false);
            return;
        }
        self.stop_selection_auto_scroll();
        self.selection_press_active.set(true);
        let scroll_view = if !self.base().is_some_and(|tui| tui.has_overlay())
            && self.current_layout.borrow().is_some()
        {
            self.current_layout.borrow().as_ref().and_then(|layout| {
                get_scroll_views_at(layout, usize::from(event.x), usize::from(event.y))
                    .first()
                    .cloned()
            })
        } else {
            None
        };
        let anchor = self.get_selection_point(event, scroll_view.as_ref());
        let word = self.get_word_selection(&anchor);
        let click_count = self.get_click_count(&anchor, word.as_ref());
        let range = if click_count == 2 {
            word
        } else if click_count == 3 {
            Some(self.get_line_selection(&anchor))
        } else {
            None
        };
        self.selection_granularity.set(if range.is_some() {
            if click_count == 2 {
                SelectionGranularity::Word
            } else {
                SelectionGranularity::Line
            }
        } else {
            SelectionGranularity::Character
        });
        *self.selection_initial_range.borrow_mut() = range;
        let (anchor_point, focus_point) =
            self.selection_initial_range.borrow().as_ref().map_or_else(
                || (anchor.clone(), anchor),
                |range| (range.start.clone(), range.end.clone()),
            );
        *self.selection_anchor.borrow_mut() = Some(anchor_point);
        *self.selection_focus.borrow_mut() = Some(focus_point);
        self.selection_dragged.set(false);
        *self.pressed_url.borrow_mut() = if self.selection_initial_range.borrow().is_some() {
            None
        } else {
            let rows = usize::from(self.terminal_rows());
            let columns = usize::from(self.terminal_columns());
            let line = self
                .previous_screen
                .borrow()
                .get(usize::from(event.y).min(rows.saturating_sub(1)))
                .cloned()
                .unwrap_or_default();
            get_osc8_link_at_column(&line, usize::from(event.x).min(columns.saturating_sub(1)))
                .map(String::from)
        };
        tui.request_render(false);
    }

    fn get_selection_bounds(&self) -> Option<SelectionRange> {
        let anchor = (*self.selection_anchor.borrow()).clone();
        let focus = (*self.selection_focus.borrow()).clone();
        let (Some(anchor), Some(focus)) = (anchor, focus) else {
            return None;
        };
        if !same_scroll_view(anchor.scroll_view.as_ref(), focus.scroll_view.as_ref()) {
            return None;
        }
        if anchor.row == focus.row && anchor.col == focus.col {
            return None;
        }
        let anchor_before_focus =
            anchor.row < focus.row || (anchor.row == focus.row && anchor.col < focus.col);
        Some(if anchor_before_focus {
            SelectionRange {
                start: anchor,
                end: focus,
            }
        } else {
            SelectionRange {
                start: focus,
                end: anchor,
            }
        })
    }

    #[expect(
        clippy::unused_self,
        reason = "upstream's getSelectionColumns reads no instance state; the port keeps the method shape"
    )]
    fn get_selection_columns(
        &self,
        line: &str,
        row: usize,
        selection: &SelectionRange,
        min_column: usize,
        max_column: usize,
    ) -> SelectionColumns {
        let line_width = visible_width(line);
        let mut start = min_column;
        let mut end = line_width.min(max_column);
        if row == selection.start.row {
            start = get_grapheme_cell_range(line, selection.start.col)
                .map_or_else(|| selection.start.col.min(line_width), |range| range.start);
        }
        if row == selection.end.row {
            end = if selection.end.boundary {
                selection.end.col.min(line_width)
            } else {
                get_grapheme_cell_range(line, selection.end.col).map_or_else(
                    || (selection.end.col + 1).min(line_width),
                    |range| range.end,
                )
            };
        }
        SelectionColumns {
            start: start.max(min_column),
            end: end.min(max_column),
        }
    }

    fn get_active_selection_text(&self) -> Option<String> {
        let selection = self.get_selection_bounds()?;
        let layout_guard = self.current_layout.borrow();
        let previous_screen = self.previous_screen.borrow();
        let source_lines: &[String] = if selection.start.scroll_view.is_some() {
            let layout = layout_guard.as_ref()?;
            let scroll_view = selection.start.scroll_view.as_ref()?;
            let box_ = get_scroll_view_box(layout, scroll_view)?;
            box_.scroll_content_lines.as_ref()?
        } else {
            &previous_screen
        };
        let mut lines = Vec::new();
        for row in selection.start.row..=selection.end.row {
            let line = source_lines.get(row).map_or("", String::as_str);
            let columns = self.get_selection_columns(line, row, &selection, 0, visible_width(line));
            lines.push(
                strip_terminal_sequences(&slice_by_column(
                    line,
                    columns.start,
                    columns.end.saturating_sub(columns.start),
                    true,
                ))
                .trim_end()
                .to_string(),
            );
        }
        let text = lines.join("\n");
        if text.is_empty() { None } else { Some(text) }
    }

    fn copy_selection_to_clipboard(&self) -> bool {
        let Some(text) = self.get_active_selection_text() else {
            return false;
        };
        self.copy_text_to_clipboard(&text)
    }

    /// Copy `text` through the configured selection clipboard path, upstream
    /// `copyTextToClipboard`: an injected handler when provided, otherwise a
    /// bare OSC 52 write — which can show "Copied!" while leaving the system
    /// clipboard untouched, so only the injected path reports verified
    /// success.
    fn copy_text_to_clipboard(&self, text: &str) -> bool {
        if let Some(copy_selection) = &self.copy_selection {
            let result = (copy_selection)(text);
            let ok = result == CopySelectionResult::Copied;
            let message = match &result {
                CopySelectionResult::Copied => "Copied!",
                CopySelectionResult::Message(message) => message.as_str(),
                CopySelectionResult::Failed => "Copy failed",
            };
            self.flash(
                message,
                if ok {
                    None
                } else {
                    Some(COPY_ERROR_FLASH_DURATION_MS)
                },
            );
            return ok;
        }
        if let Some(tui) = self.base() {
            tui.terminal_write(&format!("\x1b]52;c;{}\x07", BASE64.encode(text.as_bytes())));
        }
        self.flash("Copied!", None);
        true
    }

    // === Frame compositing, upstream's highlight/composite members ===

    fn apply_search_text_highlight(&self, text: &str, current: bool) -> String {
        let style = if current {
            &self.search_current_match_style
        } else {
            &self.search_match_style
        };
        let mut result = String::new();
        let mut plain_start = 0;
        let mut index = 0;
        while index < text.len() {
            let Some(ansi) = extract_ansi_code(text, index) else {
                index += 1;
                continue;
            };
            if index > plain_start {
                result.push_str(&style(&text[plain_start..index]));
            }
            result.push_str(ansi.code);
            index += ansi.length;
            plain_start = index;
        }
        if plain_start < text.len() {
            result.push_str(&style(&text[plain_start..]));
        }
        result
    }

    fn apply_search_highlights(&self, screen: &[String], layout: &LayoutFrame) -> Vec<String> {
        let search = self.active_search.borrow();
        let Some(search) = search.as_ref() else {
            return screen.to_vec();
        };
        if search.selected_index < 0 || search.matches.is_empty() {
            return screen.to_vec();
        }
        let scroll_view = layout
            .primary_scroll_view
            .clone()
            .unwrap_or_else(|| self.implicit_scroll_view.state_handle());
        let Some(box_) = get_scroll_view_box(layout, &scroll_view) else {
            return screen.to_vec();
        };

        let mut ranges_by_row: BTreeMap<usize, Vec<SearchHighlightRange>> = BTreeMap::new();
        let scrollbar_column = get_scrollbar_geometry(box_, false).map(|geometry| geometry.column);
        let columns = usize::from(self.terminal_columns());
        let min_row = span_usize(0.max(box_.rect.y).max(box_.clip.y));
        let max_row = line_i64(screen.len())
            .min(box_.rect.y + box_.rect.height)
            .min(box_.clip.y + box_.clip.height);
        let min_column = span_usize(0.max(box_.rect.x).max(box_.clip.x));
        let max_column = columns
            .min(span_usize(box_.rect.x + box_.rect.width))
            .min(span_usize(box_.clip.x + box_.clip.width))
            .min(scrollbar_column.unwrap_or(usize::MAX));
        let scroll_top = line_i64(scroll_view.scroll_top());
        let min_content_row = scroll_top + line_i64(min_row) - box_.rect.y;
        let max_content_row = scroll_top + line_i64(span_usize(max_row)) - box_.rect.y - 1;
        let mut low: usize = 0;
        let mut high: usize = search.matches.len();
        while low < high {
            let middle = low + (high - low) / 2;
            let last_row = search.matches[middle]
                .segments
                .last()
                .map_or(-1, |segment| line_i64(segment.row));
            if last_row < min_content_row {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        for (match_index, match_) in search.matches.iter().enumerate().skip(low) {
            let first_row = match_
                .segments
                .first()
                .map_or(0, |segment| line_i64(segment.row));
            if first_row > max_content_row {
                break;
            }
            for segment in &match_.segments {
                let row = box_.rect.y + line_i64(segment.row) - scroll_top;
                if row < line_i64(min_row) || row >= max_row {
                    continue;
                }
                let start_col =
                    min_column.max(span_usize(box_.rect.x + line_i64(segment.start_col)));
                let end_col = max_column.min(span_usize(box_.rect.x + line_i64(segment.end_col)));
                if end_col <= start_col {
                    continue;
                }
                ranges_by_row
                    .entry(span_usize(row))
                    .or_default()
                    .push(SearchHighlightRange {
                        start_col,
                        end_col,
                        current: match_index
                            == usize::try_from(search.selected_index).unwrap_or(usize::MAX),
                    });
            }
        }

        let mut result = screen.to_vec();
        for (row, ranges) in &ranges_by_row {
            let mut line = result.get(*row).cloned().unwrap_or_default();
            if is_image_line(&line) {
                continue;
            }
            let line_width = visible_width(&line);
            let mut sorted = ranges.clone();
            sorted.sort_by_key(|range| std::cmp::Reverse(range.start_col));
            for range in &sorted {
                let start_col = range.start_col.min(line_width);
                let end_col = range.end_col.min(line_width);
                if end_col <= start_col {
                    continue;
                }
                let before = slice_by_column(&line, 0, start_col, true);
                let highlighted = slice_by_column(&line, start_col, end_col - start_col, true);
                let after =
                    slice_by_column(&line, end_col, line_width.saturating_sub(end_col), true);
                line = format!(
                    "{before}{}{after}",
                    self.apply_search_text_highlight(&highlighted, range.current)
                );
            }
            result[*row] = line;
        }
        result
    }

    #[expect(
        clippy::unused_self,
        reason = "upstream's applySelectionHighlight reads no instance state; the port keeps the method shape"
    )]
    fn apply_selection_highlight(&self, text: &str) -> String {
        let mut result = String::from("\x1b[7m");
        let mut index = 0;
        while index < text.len() {
            let Some(ansi) = extract_ansi_code(text, index) else {
                #[expect(
                    clippy::expect_used,
                    reason = "the loop starts from a byte position inside the text, so the remainder is non-empty"
                )]
                let ch = text[index..]
                    .chars()
                    .next()
                    .expect("the loop index sits on a char boundary inside the text");
                result.push(ch);
                index += ch.len_utf8();
                continue;
            };
            result.push_str(ansi.code);
            if ansi.code.ends_with('m') {
                result.push_str("\x1b[7m");
            }
            index += ansi.length;
        }
        format!("{result}\x1b[27m")
    }

    fn apply_selection(&self, screen: &[String], layout: Option<&LayoutFrame>) -> Vec<String> {
        let Some(selection) = self.get_selection_bounds() else {
            return screen.to_vec();
        };
        let mut screen_selection = selection.clone();
        let mut min_row = 0;
        let mut max_row = line_i64(screen.len()).saturating_sub(1);
        let mut min_column = 0;
        let mut max_column = usize::from(self.terminal_columns());
        if selection.start.scroll_view.is_some() {
            let Some(layout) = layout else {
                return screen.to_vec();
            };
            let Some(scroll_view) = selection.start.scroll_view.as_ref() else {
                return screen.to_vec();
            };
            let Some(box_) = get_scroll_view_box(layout, scroll_view) else {
                return screen.to_vec();
            };
            min_row = span_usize(0.max(box_.rect.y).max(box_.clip.y));
            max_row = line_i64(screen.len())
                .saturating_sub(1)
                .min(box_.rect.y + box_.rect.height - 1)
                .min(box_.clip.y + box_.clip.height - 1);
            min_column = span_usize(0.max(box_.rect.x).max(box_.clip.x));
            max_column = usize::from(self.terminal_columns())
                .min(span_usize(box_.rect.x + box_.rect.width))
                .min(span_usize(box_.clip.x + box_.clip.width));
            screen_selection = SelectionRange {
                start: SelectionPoint {
                    row: span_usize(
                        box_.rect.y + line_i64(selection.start.row)
                            - line_i64(scroll_view.scroll_top()),
                    ),
                    col: span_usize(box_.rect.x + line_i64(selection.start.col)),
                    scroll_view: None,
                    boundary: selection.start.boundary,
                },
                end: SelectionPoint {
                    row: span_usize(
                        box_.rect.y + line_i64(selection.end.row)
                            - line_i64(scroll_view.scroll_top()),
                    ),
                    col: span_usize(box_.rect.x + line_i64(selection.end.col)),
                    scroll_view: None,
                    boundary: selection.end.boundary,
                },
            };
        }
        screen
            .iter()
            .enumerate()
            .map(|(row, line)| {
                if row < min_row
                    || row > span_usize(max_row)
                    || row < screen_selection.start.row
                    || row > screen_selection.end.row
                    || is_image_line(line)
                {
                    return line.clone();
                }
                let line_width = visible_width(line);
                let columns = self.get_selection_columns(
                    line,
                    row,
                    &screen_selection,
                    min_column,
                    max_column,
                );
                if columns.end <= columns.start {
                    return line.clone();
                }
                let before = slice_by_column(line, 0, columns.start, true);
                let selected =
                    slice_by_column(line, columns.start, columns.end - columns.start, true);
                let after = slice_by_column(
                    line,
                    columns.end,
                    line_width.saturating_sub(columns.end),
                    true,
                );
                format!(
                    "{before}{}{after}",
                    self.apply_selection_highlight(&selected)
                )
            })
            .collect()
    }

    #[expect(
        clippy::unused_self,
        reason = "upstream's isMouseSequence reads no instance state; the port keeps the method shape"
    )]
    fn is_mouse_sequence(&self, data: &str) -> bool {
        SGR_MOUSE.is_match(data) || (data.chars().count() == 6 && data.starts_with("\x1b[M"))
    }

    fn composite_scroll_to_end_indicator(
        &self,
        screen: &[String],
        layout: &LayoutFrame,
        width: usize,
    ) -> Vec<String> {
        *self.scroll_to_end_indicator_rect.borrow_mut() = None;
        let scroll_view = layout
            .primary_scroll_view
            .clone()
            .unwrap_or_else(|| self.implicit_scroll_view.state_handle());
        if !scroll_view.follow_end() || scroll_view.is_following_end() {
            return screen.to_vec();
        }
        let Some(box_) = get_scroll_view_box(layout, &scroll_view) else {
            return screen.to_vec();
        };
        let clip = box_.clip;
        if clip.width <= 0 || clip.height <= 0 {
            return screen.to_vec();
        }
        let row = span_usize(clip.y + clip.height - 1);
        if row >= screen.len() || is_image_line(screen.get(row).map_or("", String::as_str)) {
            return screen.to_vec();
        }
        let scrollbar_column = get_scrollbar_geometry(box_, false).map(|geometry| geometry.column);
        let clip_end = span_usize(clip.x + clip.width);
        let available_width = scrollbar_column
            .unwrap_or(clip_end)
            .saturating_sub(span_usize(clip.x));
        let Some(indicator) = &self.scroll_to_end_indicator else {
            return screen.to_vec();
        };
        let text = truncate_to_width(&(indicator)(), available_width, "", false);
        let text_width = visible_width(&text);
        if text_width == 0 {
            return screen.to_vec();
        }
        let column = span_usize(clip.x) + (available_width - text_width) / 2;
        let mut result = screen.to_vec();
        result[row] = composite_tui_line(
            result.get(row).map_or("", String::as_str),
            &text,
            column,
            text_width,
            width,
        );
        *self.scroll_to_end_indicator_rect.borrow_mut() = Some(ScrollToEndIndicatorRect {
            row,
            column,
            width: text_width,
        });
        result
    }

    fn composite_flashes(&self, screen: &[String], width: usize, height: usize) -> Vec<String> {
        let rendered = self.flashes.render(width);
        let flash_lines = &rendered[rendered.len().saturating_sub(height)..];
        if flash_lines.is_empty() {
            return screen.to_vec();
        }
        let mut result = screen.to_vec();
        while result.len() < height {
            result.push(String::new());
        }
        for (row, line) in flash_lines.iter().enumerate() {
            let flash_width = visible_width(line);
            if flash_width == 0 {
                continue;
            }
            result[row] = composite_tui_line(
                result.get(row).map_or("", String::as_str),
                line,
                width - flash_width,
                flash_width,
                width,
            );
        }
        result
    }

    /// The full-screen frame, upstream `doRender`: paint exactly
    /// `terminal.rows` rows with the search highlights, the jump-to-end
    /// indicator, the overlays, the selection, and the flashes composited in
    /// that order, then diff against the previous frame.
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's doRender block for block; splitting it would break the 1:1 correspondence"
    )]
    fn do_render(&self, tui: &Tui) {
        if tui.is_stopped() || !self.alt_screen_active.get() {
            return;
        }
        self.flashes.drain_expired();
        self.drain_auto_scroll_ticks();
        let width = usize::from(tui.terminal_columns()).max(1);
        let height = usize::from(tui.terminal_rows()).max(1);
        let root: Rc<dyn Component> = self.layout_root.borrow().clone().unwrap_or_else(|| {
            let implicit: Rc<dyn Component> = self.implicit_scroll_view.clone();
            implicit
        });
        let request = tui.render_request();
        let mut next_layout = render_layout_frame(&root, width, height, &request);
        if self.refresh_search(&next_layout) {
            next_layout = render_layout_frame(&root, width, height, &request);
        }
        let mut screen: Vec<String> = next_layout
            .lines
            .iter()
            .map(|line| OSC133_ZONE_PREFIX.replace(line, "").into_owned())
            .collect();
        screen = self.apply_search_highlights(&screen, &next_layout);
        screen = self.composite_scroll_to_end_indicator(&screen, &next_layout, width);
        screen = tui.composite_overlays(screen, width, height);
        if screen.len() > height {
            screen = screen.split_off(screen.len() - height);
        }
        screen = self.apply_selection(&screen, Some(&next_layout));
        screen = self.composite_flashes(&screen, width, height);

        let cursor_pos = tui.extract_cursor_position(&mut screen, height);
        let mut screen = tui.apply_line_resets(screen);
        for line in &mut screen {
            if is_image_line(line) || visible_width(line) <= width {
                continue;
            }
            *line = slice_by_column(line, 0, width, true);
        }

        let previous_screen = self.previous_screen.borrow();
        let full_redraw = previous_screen.is_empty()
            || self.previous_screen_width.get() != width
            || self.previous_screen_height.get() != height;
        let images_need_redraw = screen.iter().enumerate().any(|(row, line)| {
            line != previous_screen.get(row).map_or("", String::as_str)
                && (is_image_line(line)
                    || is_image_line(previous_screen.get(row).map_or("", String::as_str)))
        });
        let redraw_images = full_redraw || images_need_redraw;
        let had_uploaded_kitty_images = !self.uploaded_kitty_images.borrow().is_empty();
        let (prepared_lines, evicted_image_deletion) =
            if redraw_images && self.image_protocol.get() == Some(ImageProtocol::Kitty) {
                self.prepare_kitty_screen(&screen)
            } else {
                (screen.clone(), String::new())
            };
        drop(previous_screen);

        let mut buffer = String::from(BEGIN_SYNCHRONIZED_OUTPUT);
        if full_redraw {
            tui.bump_full_redraws();
            let clear_images = if self.image_protocol.get() == Some(ImageProtocol::Kitty)
                && had_uploaded_kitty_images
            {
                delete_all_kitty_placements()
            } else {
                self.delete_kitty_images()
            };
            buffer.push_str(&clear_images);
            buffer.push_str("\x1b[2J");
        } else if images_need_redraw {
            if self.image_protocol.get() == Some(ImageProtocol::Iterm2) {
                buffer.push_str("\x1b[2J");
            } else if self.image_protocol.get() == Some(ImageProtocol::Kitty) {
                buffer.push_str(&delete_all_kitty_placements());
            }
        }
        buffer.push_str(&evicted_image_deletion);

        let previous_screen = self.previous_screen.borrow();
        for row in 0..height {
            if !full_redraw
                && !images_need_redraw
                && screen
                    .get(row)
                    .is_none_or(|line| line == previous_screen.get(row).map_or("", String::as_str))
            {
                continue;
            }
            let _ = write!(buffer, "\x1b[{};1H", row + 1);
            buffer.push_str("\x1b[2K");
            buffer.push_str(prepared_lines.get(row).map_or("", String::as_str));
        }
        drop(previous_screen);

        if let Some((cursor_row, cursor_col)) = cursor_pos {
            let _ = write!(
                buffer,
                "\x1b[{};{}H",
                cursor_row + 1,
                width.min(cursor_col) + 1
            );
            buffer.push_str(if tui.get_show_hardware_cursor() {
                "\x1b[?25h"
            } else {
                "\x1b[?25l"
            });
        } else {
            buffer.push_str("\x1b[?25l");
        }
        buffer.push_str(END_SYNCHRONIZED_OUTPUT);
        tui.terminal_write(&buffer);

        *self.previous_screen.borrow_mut() = screen;
        self.previous_screen_width.set(width);
        self.previous_screen_height.set(height);
        *self.current_layout.borrow_mut() = Some(next_layout);
    }
}

/// Decode an SGR mouse button code, upstream `decodeMouseButton`.
const fn decode_mouse_button(button: u32) -> TuiMouseButton {
    match button & 3 {
        0 => TuiMouseButton::Left,
        1 => TuiMouseButton::Middle,
        2 => TuiMouseButton::Right,
        _ => TuiMouseButton::None,
    }
}

/// i64 into the `u16` the mouse event rides: negative values clamp to zero,
/// oversized ones to `u16::MAX`.
fn local_coord(value: i64) -> u16 {
    u16::try_from(value).unwrap_or(if value < 0 { 0 } else { u16::MAX })
}

/// i64 into the `u16` mouse coordinates ride; same clamp as
/// [`local_coord`].
fn mouse_coord(value: i64) -> u16 {
    local_coord(value)
}

fn parse_u32(text: &str) -> u32 {
    text.parse().unwrap_or(u32::MAX)
}

/// Whether two optional components name the same component, upstream's
/// `!==` identity checks.
fn component_matches(a: Option<&Rc<dyn Component>>, b: &Rc<dyn Component>) -> bool {
    a.is_some_and(|a| Rc::as_ptr(a).cast::<()>() == Rc::as_ptr(b).cast::<()>())
}

impl TuiRenderer for TuiAltScreen {
    fn mode(&self) -> TuiMode {
        TuiMode::Fullscreen
    }

    fn do_render(&self, tui: &Tui) {
        self.core.do_render(tui);
    }

    fn reset_render_state(&self) {
        self.core.reset_render_state();
    }

    fn before_terminal_start(&self) {
        self.core.before_terminal_start();
    }

    fn before_terminal_stop(&self, tui: &Tui, options: &TuiStopOptions) {
        self.core.before_terminal_stop(tui, *options);
    }

    fn after_terminal_stop(&self, options: &TuiStopOptions) {
        self.core.after_terminal_stop(*options);
    }

    fn mounted_roots(&self, tui: &Tui) -> Vec<Rc<dyn Component>> {
        self.core
            .layout_root
            .borrow()
            .as_ref()
            .map_or_else(|| Component::children(tui), |root| vec![Rc::clone(root)])
    }

    fn is_viewport_tui(&self) -> bool {
        true
    }

    fn set_layout_root(&self, _tui: &Tui, component: Option<Rc<dyn Component>>) {
        self.core.set_layout_root(component);
    }

    /// Upstream's constructor registered the input listener and handed the
    /// implicit document its base; `Tui::new` calls this once with the
    /// finished base.
    fn attach_base(&self, tui: &Rc<Tui>) {
        *self.core.base.borrow_mut() = Rc::downgrade(tui);
        *self.core.implicit_document.tui.borrow_mut() = Rc::downgrade(tui);
        self.core.flashes.set_request_render(tui.render_request());
        let core = Rc::clone(&self.core);
        let base = Rc::downgrade(tui);
        tui.add_input_listener(Rc::new(move |data: &str| {
            let tui = base.upgrade()?;
            core.handle_viewport_input(&tui, data)
        }));
    }
}

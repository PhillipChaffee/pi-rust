//! `tui-alt-screen.test.ts` ported 1:1 (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): viewport scrolling, mouse
//! selection and word granularity, hyperlinks, scrollbar drags, the
//! transcript search overlay, Kitty image placement caching, focus routing,
//! and the flash stack.
//!
//! Every suite in this file holds [`CAPABILITIES_LOCK`]: the capability
//! cache is process-global, and `TuiAltScreen`'s start hook disables it for
//! iTerm2 terminals the way upstream's `setCapabilities` did — a write
//! another suite's concurrent `tui.start()` would consume, where upstream's
//! `node:test` ran one process sequentially. The suites that rebind the
//! global keybindings registry additionally hold [`KEYBINDINGS_LOCK`].
//!
//! Deferrals, per the parent ticket's slice rules:
//!
//! - The `Image` component and its capability gate are #51's scope; the
//!   suites that drive it construct placements through a [`TestImage`]
//!   stand-in — a cached, capability-gated render that registers the same
//!   cell metadata the real component's `renderImage` does, so the renderer
//!   sees identical placement lines.
//! - `SelectList` is #49's scope; the vertical-redispatch regression test's
//!   essence is the horizontal container's miss behavior, so a
//!   [`ClickList`] stand-in with the same custom-handleMouse shape drives it
//!   until #49 lands the real list.

#![expect(
    clippy::expect_used,
    reason = "a missing fixture render or unexpected write in a test is an environment failure; expecting keeps the assertions readable"
)]
#![expect(
    clippy::panic,
    reason = "a failed invariant in a test must fail loudly; the panic is the assertion"
)]

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Mutex;
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use pi_tui::alt_screen_search::{
    AltScreenSearchComponent, AltScreenSearchIndex, AltScreenSearchMatch, AltScreenSearchSegment,
    find_alt_screen_search_matches,
};
use pi_tui::components::{
    FollowMode, HStack, MouseRegion, ScrollView, ScrollViewOptions, ScrollViewScrollbar,
    StackChild, StackEntryOptions, StackOptions, Text, VStack,
};
use pi_tui::keybindings::{Keybindings, KeybindingsConfig, KeybindingsManager, set_keybindings};
use pi_tui::layout_node::Basis;
use pi_tui::terminal::{EnvLookup, Terminal};
use pi_tui::terminal_image::{
    EncodeKittyOptions, KittyImageMetadata, encode_kitty, get_capabilities, hyperlink,
    register_kitty_image_metadata, reset_capabilities_cache, set_capabilities,
};
use pi_tui::tui::{
    Component, Tui, TuiConfig, TuiMouseEvent, TuiMouseEventResult, TuiMouseEventType,
    TuiStopOptions,
};
use pi_tui::tui_alt_screen::{CopySelectionResult, TuiAltScreen, TuiAltScreenConfig};
use pi_tui::utils::{strip_terminal_sequences, truncate_to_width};

use tui_support::{VirtualTerminal, render_and_flush, wait_for_render};

const OSC133_ZONE_START: &str = "\x1b]133;A\x07";

/// Serializes the suites that seed the process-global capability cache.
static CAPABILITIES_LOCK: Mutex<()> = Mutex::new(());

/// Serializes the suites that rebind the process-global keybindings
/// registry.
static KEYBINDINGS_LOCK: Mutex<()> = Mutex::new(());

/// The suite's `InputOverlay`: records the input it receives and carries the
/// focusable contract.
struct InputOverlay {
    focused: Cell<bool>,
    inputs: RefCell<Vec<String>>,
}

impl InputOverlay {
    fn new() -> Rc<Self> {
        Rc::new(Self {
            focused: Cell::new(false),
            inputs: RefCell::new(Vec::new()),
        })
    }
}

impl pi_tui::tui::Focusable for InputOverlay {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

impl Component for InputOverlay {
    fn render(&self, _width: usize) -> Vec<String> {
        vec!["overlay".to_string()]
    }

    fn handle_input(&self, data: &str) {
        self.inputs.borrow_mut().push(data.to_string());
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {}

    fn as_focusable(&self) -> Option<&dyn pi_tui::tui::Focusable> {
        Some(self)
    }
}

/// The kind of harness event the `RecordingTerminal` records, upstream
/// `RecordingTerminal.events`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RecordingEvent {
    Write(String),
    Start,
    Stop,
}

/// The suite's `RecordingTerminal`: a [`VirtualTerminal`] that records the
/// start/stop/write order the renderer's sequence assertions need.
struct RecordingTerminal {
    inner: VirtualTerminal,
    events: Rc<RefCell<Vec<RecordingEvent>>>,
}

impl RecordingTerminal {
    fn new(columns: u16, rows: u16) -> Self {
        Self {
            inner: VirtualTerminal::new(columns, rows),
            events: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn writes(&self) -> String {
        self.events
            .borrow()
            .iter()
            .filter_map(|event| match event {
                RecordingEvent::Write(data) => Some(data.as_str()),
                _ => None,
            })
            .collect()
    }

    fn writes_from(&self, offset: usize) -> String {
        self.events
            .borrow()
            .iter()
            .skip(offset)
            .filter_map(|event| match event {
                RecordingEvent::Write(data) => Some(data.as_str()),
                _ => None,
            })
            .collect()
    }

    fn event_count(&self) -> usize {
        self.events.borrow().len()
    }

    fn send_input(&self, data: &str) {
        self.inner.send_input(data);
    }

    fn get_viewport(&self) -> Vec<String> {
        self.inner.get_viewport()
    }
}

impl Clone for RecordingTerminal {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            events: Rc::clone(&self.events),
        }
    }
}

impl Terminal for RecordingTerminal {
    fn start(
        &mut self,
        on_input: pi_tui::terminal::InputHandler,
        on_resize: pi_tui::terminal::ResizeHandler,
    ) {
        self.events.borrow_mut().push(RecordingEvent::Start);
        self.inner.start(on_input, on_resize);
    }

    fn stop(&mut self) {
        self.events.borrow_mut().push(RecordingEvent::Stop);
        self.inner.stop();
    }

    fn drain_input(&mut self, max_ms: u64, idle_ms: u64) {
        self.inner.drain_input(max_ms, idle_ms);
    }

    fn write(&mut self, data: &str) {
        self.events
            .borrow_mut()
            .push(RecordingEvent::Write(data.to_string()));
        self.inner.write(data);
    }

    fn columns(&self) -> u16 {
        self.inner.columns()
    }

    fn rows(&self) -> u16 {
        self.inner.rows()
    }

    fn is_kitty_protocol_active(&self) -> bool {
        self.inner.is_kitty_protocol_active()
    }

    fn move_by(&mut self, lines: i32) {
        self.inner.move_by(lines);
    }

    fn hide_cursor(&mut self) {
        self.inner.hide_cursor();
    }

    fn show_cursor(&mut self) {
        self.inner.show_cursor();
    }

    fn clear_line(&mut self) {
        self.inner.clear_line();
    }

    fn clear_from_cursor(&mut self) {
        self.inner.clear_from_cursor();
    }

    fn clear_screen(&mut self) {
        self.inner.clear_screen();
    }

    fn set_title(&mut self, title: &str) {
        self.inner.set_title(title);
    }

    fn set_progress(&mut self, active: bool) {
        self.inner.set_progress(active);
    }

    fn poll(&mut self, timeout: Duration) {
        self.inner.poll(timeout);
    }
}

/// The `Image` component stand-in for #51's scope: a cached, capability-gated
/// render that registers the same cell metadata the real component's
/// `renderImage` does, so the renderer sees identical placement lines.
struct TestImage {
    base64_data: String,
    columns: usize,
    rows: usize,
    image_id: u64,
    width_px: u64,
    height_px: u64,
    /// The line the real component's fallback renders when the terminal has
    /// no image protocol, upstream `imageFallback`'s output.
    fallback: String,
    cached: RefCell<Option<(usize, Vec<String>)>>,
}

impl TestImage {
    fn new(
        base64_data: &str,
        columns: usize,
        rows: usize,
        image_id: u64,
        width_px: u64,
        height_px: u64,
        fallback: &str,
    ) -> Rc<Self> {
        Rc::new(Self {
            base64_data: base64_data.to_string(),
            columns,
            rows,
            image_id,
            width_px,
            height_px,
            fallback: fallback.to_string(),
            cached: RefCell::new(None),
        })
    }
}

impl Component for TestImage {
    fn render(&self, width: usize) -> Vec<String> {
        if let Some((cached_width, lines)) = self.cached.borrow().as_ref()
            && *cached_width == width
        {
            return lines.clone();
        }
        let lines: Vec<String> = if get_capabilities().images.is_some() {
            register_kitty_image_metadata(KittyImageMetadata {
                image_id: self.image_id,
                columns: self.columns,
                rows: self.rows,
                width_px: self.width_px,
                height_px: self.height_px,
            });
            let mut lines = vec![encode_kitty(
                &self.base64_data,
                EncodeKittyOptions {
                    columns: Some(self.columns),
                    rows: Some(self.rows),
                    image_id: Some(self.image_id),
                    move_cursor: Some(false),
                },
            )];
            for _ in 0..self.rows.saturating_sub(1) {
                lines.push(String::new());
            }
            lines
        } else {
            vec![truncate_to_width(
                &self.fallback,
                width.max(1),
                "...",
                false,
            )]
        };
        *self.cached.borrow_mut() = Some((width, lines.clone()));
        lines
    }

    fn invalidate(&self) {
        *self.cached.borrow_mut() = None;
    }
}

/// The `SelectList` stand-in for #49's scope: a mouse-aware list that fires
/// on clicks within its visible rows — including clicks outside its column,
/// exactly the shape the vertical-redispatch regression guards.
struct ClickList {
    selections: Cell<usize>,
}

impl Component for ClickList {
    fn render(&self, _width: usize) -> Vec<String> {
        vec!["A".to_string(), "B".to_string()]
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        if event.event_type != TuiMouseEventType::Click {
            return None;
        }
        self.selections.set(self.selections.get() + 1);
        Some(TuiMouseEventResult {
            handled: true,
            ..TuiMouseEventResult::default()
        })
    }

    fn invalidate(&self) {}
}

fn env_lookup(map: &HashMap<String, String>) -> EnvLookup {
    let map = map.clone();
    Box::new(move |key| map.get(key).cloned())
}

fn new_tui(terminal: VirtualTerminal, config: TuiAltScreenConfig) -> (Rc<Tui>, TuiAltScreen) {
    let alt = TuiAltScreen::new(config);
    let tui = Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(Box::new(alt.clone())),
        ..TuiConfig::default()
    });
    (tui, alt)
}

fn new_recording_tui(
    terminal: RecordingTerminal,
    config: TuiAltScreenConfig,
) -> (Rc<Tui>, TuiAltScreen) {
    let alt = TuiAltScreen::new(config);
    let tui = Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(Box::new(alt.clone())),
        ..TuiConfig::default()
    });
    (tui, alt)
}

fn text(lines: &[&str]) -> Rc<Text> {
    Rc::new(Text::with_padding(lines.join("\n"), 0, 0))
}

fn numbered_lines(count: usize) -> String {
    (0..count)
        .map(|index| format!("line {}", index + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

fn base64_of(value: &str) -> String {
    BASE64.encode(value.as_bytes())
}

fn osc52(value: &str) -> String {
    format!("\x1b]52;c;{}\x07", base64_of(value))
}

fn char_col(line: &str, byte_index: usize) -> usize {
    line[..byte_index].chars().count()
}

fn trimmed_viewport(terminal: &VirtualTerminal) -> Vec<String> {
    terminal
        .get_viewport()
        .iter()
        .map(|line| line.trim_end().to_string())
        .collect()
}

fn line_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn sleep_ms(ms: u64) {
    std::thread::sleep(Duration::from_millis(ms));
}

/// Pump forced renders until `ready` holds or the attempts run out.
///
/// Upstream's suites drove these workers through node's event loop, which
/// serialized timers with the render; the port's worker threads are real
/// threads, and under parallel test load a single forced flush can race the
/// worker's fire, so the timing assertions retry.
fn flush_until(tui: &Tui, mut ready: impl FnMut() -> bool) {
    for _ in 0..10 {
        render_and_flush(tui);
        if ready() {
            return;
        }
        sleep_ms(30);
    }
    render_and_flush(tui);
}

fn scroll_entry(component: Rc<dyn Component>, basis: i64, grow: u32, min_size: u32) -> StackChild {
    StackChild::entry(
        component,
        StackEntryOptions {
            basis: Some(Basis::Cells(basis)),
            grow: Some(grow),
            min_size: Some(min_size),
            ..StackEntryOptions::default()
        },
    )
}

// === The upstream suites ===
// === TuiAltScreen ===

#[test]
fn renders_a_terminal_height_viewport_and_preserves_manual_scroll_position() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 4);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    let text = text(&[&numbered_lines(10)]);
    tui.add_child(text.clone());
    tui.start();
    wait_for_render(&tui);

    assert_eq!(
        trimmed_viewport(&terminal),
        vec!["line 7", "line 8", "line 9", "line 10"],
    );
    assert!(alt.is_following_output());

    terminal.send_input("\x1b[<64;1;1M");
    wait_for_render(&tui);
    assert_eq!(
        trimmed_viewport(&terminal),
        vec!["line 6", "line 7", "line 8", "line 9"],
    );
    assert_eq!(alt.viewport_top(), 5);
    assert!(!alt.is_following_output());

    text.set_text(&numbered_lines(12));
    tui.request_render(false);
    wait_for_render(&tui);
    assert_eq!(
        trimmed_viewport(&terminal),
        vec!["line 6", "line 7", "line 8", "line 9"],
    );

    tui.stop(TuiStopOptions::default());
}

#[test]
fn shows_a_clickable_jump_to_end_indicator_on_the_transcripts_last_row_while_scrolled_up() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(30, 6);
    let (tui, alt) = new_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            scroll_to_end_indicator: Some(Rc::new(|| "\x1b[7m ↓ Jump to end \x1b[27m".to_string())),
            ..TuiAltScreenConfig::default()
        },
    );
    let transcript = ScrollView::new(
        text(&[&numbered_lines(8)]),
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            primary: true,
            ..ScrollViewOptions::default()
        },
    );
    tui.set_layout_root(Some(VStack::new(
        vec![
            scroll_entry(transcript.clone(), 0, 1, 1),
            StackChild::entry(
                text(&["editor", "footer"]),
                StackEntryOptions {
                    min_size: Some(1),
                    ..StackEntryOptions::default()
                },
            ),
        ],
        StackOptions::default(),
    )));
    tui.start();
    wait_for_render(&tui);
    assert!(
        !terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("Jump to end"))
    );

    terminal.send_input("\x1b[<64;1;1M");
    wait_for_render(&tui);
    assert!(!transcript.is_following_end());
    assert_eq!(terminal.get_viewport()[3], "line 7  ↓ Jump to end         ");
    assert_eq!(terminal.get_viewport()[4].trim_end(), "editor");

    // Pressing next to the label starts a selection instead of jumping.
    terminal.send_input("\x1b[<0;2;4M");
    terminal.send_input("\x1b[<0;2;4m");
    wait_for_render(&tui);
    assert!(!transcript.is_following_end());

    terminal.send_input("\x1b[<0;15;4M");
    terminal.send_input("\x1b[<0;15;4m");
    wait_for_render(&tui);
    assert!(transcript.is_following_end());
    assert_eq!(
        trimmed_viewport(&terminal),
        vec!["line 5", "line 6", "line 7", "line 8", "editor", "footer"],
    );
    let _ = &alt;
    tui.stop(TuiStopOptions::default());
}

#[test]
fn leaves_the_scrollbar_clickable_when_the_jump_to_end_indicator_spans_the_transcript() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(30, 6);
    let (tui, _alt) = new_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            scroll_to_end_indicator: Some(Rc::new(|| "↓".repeat(30))),
            ..TuiAltScreenConfig::default()
        },
    );
    let transcript = ScrollView::new(
        text(&[&numbered_lines(12)]),
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            primary: true,
            scrollbar: Some(ScrollViewScrollbar::Always),
            ..ScrollViewOptions::default()
        },
    );
    tui.set_layout_root(Some(VStack::new(
        vec![
            scroll_entry(transcript.clone(), 0, 1, 1),
            StackChild::entry(
                text(&["editor", "footer"]),
                StackEntryOptions {
                    min_size: Some(1),
                    ..StackEntryOptions::default()
                },
            ),
        ],
        StackOptions::default(),
    )));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<64;1;1M");
    wait_for_render(&tui);
    assert!(!transcript.is_following_end());

    // The indicator must not intercept a press on the scrollbar's last column.
    terminal.send_input("\x1b[<0;30;4M");
    terminal.send_input("\x1b[<0;30;4m");
    wait_for_render(&tui);
    assert!(!transcript.is_following_end());
    tui.stop(TuiStopOptions::default());
}

#[test]
fn never_shows_the_jump_to_end_indicator_for_a_primary_scroll_view_without_follow_end() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(30, 3);
    let (tui, _alt) = new_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            scroll_to_end_indicator: Some(Rc::new(|| " ↓ Jump to end ".to_string())),
            ..TuiAltScreenConfig::default()
        },
    );
    let transcript = ScrollView::new(
        Rc::new(Text::with_padding("one\ntwo\nthree\nfour\nfive", 0, 0)),
        ScrollViewOptions {
            primary: true,
            ..ScrollViewOptions::default()
        },
    );
    tui.set_layout_root(Some(transcript.clone()));
    tui.start();
    wait_for_render(&tui);

    assert!(!transcript.is_following_end());
    assert!(
        !terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("Jump to end"))
    );
    tui.stop(TuiStopOptions::default());
}

#[test]
fn keeps_an_explicit_dock_fixed_while_the_transcript_scrolls() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 6);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    let transcript_text = text(&[&numbered_lines(8)]);
    let transcript = ScrollView::new(
        transcript_text.clone(),
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            primary: true,
            ..ScrollViewOptions::default()
        },
    );
    let dock = VStack::new(
        vec![
            StackChild::component(text(&["editor"])),
            StackChild::component(text(&["footer"])),
        ],
        StackOptions::default(),
    );
    tui.set_layout_root(Some(VStack::new(
        vec![
            scroll_entry(transcript.clone(), 0, 1, 1),
            StackChild::entry(
                dock,
                StackEntryOptions {
                    min_size: Some(1),
                    ..StackEntryOptions::default()
                },
            ),
        ],
        StackOptions::default(),
    )));
    tui.start();
    wait_for_render(&tui);

    assert_eq!(
        trimmed_viewport(&terminal),
        vec!["line 5", "line 6", "line 7", "line 8", "editor", "footer"],
    );

    // Wheel over the dock falls back to the primary transcript scroll view.
    terminal.send_input("\x1b[<64;1;6M");
    wait_for_render(&tui);
    assert_eq!(
        trimmed_viewport(&terminal),
        vec!["line 4", "line 5", "line 6", "line 7", "editor", "footer"],
    );
    assert!(!transcript.is_following_end());

    transcript_text.set_text(&numbered_lines(10));
    tui.request_render(false);
    wait_for_render(&tui);
    assert_eq!(
        trimmed_viewport(&terminal),
        vec!["line 4", "line 5", "line 6", "line 7", "editor", "footer"],
    );

    alt.scroll_to_bottom();
    wait_for_render(&tui);
    assert_eq!(
        trimmed_viewport(&terminal),
        vec!["line 7", "line 8", "line 9", "line 10", "editor", "footer"],
    );
    tui.stop(TuiStopOptions::default());
}

#[test]
fn invalidates_overlays_with_an_explicit_layout_root() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(80, 24);
    let (tui, _alt) = new_tui(terminal, TuiAltScreenConfig::default());
    let overlay = text(&["overlay"]);
    let invalidated = Rc::new(Cell::new(false));
    let probe = Rc::clone(&invalidated);
    let overlay: Rc<dyn Component> = Rc::new(InvalidatingText {
        inner: overlay,
        invalidated: probe,
    });
    let root = text(&["root"]);
    tui.set_layout_root(Some(root));
    let _ = tui.show_overlay(overlay, None);

    tui.invalidate();

    assert!(invalidated.get());
    tui.stop(TuiStopOptions::default());
}

/// A [`Text`] whose `invalidate` records the call, upstream's
/// `overlay.invalidate = () => { invalidated = true }` override.
struct InvalidatingText {
    inner: Rc<Text>,
    invalidated: Rc<Cell<bool>>,
}

impl Component for InvalidatingText {
    fn render(&self, width: usize) -> Vec<String> {
        self.inner.render(width)
    }

    fn invalidate(&self) {
        self.invalidated.set(true);
    }
}

#[test]
fn routes_wheel_input_to_the_scroll_view_under_the_pointer() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 4);
    let (tui, _alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    let left = ScrollView::new(
        Rc::new(Text::with_padding("a1\na2\na3\na4\na5\na6\na7", 0, 0)),
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            primary: true,
            ..ScrollViewOptions::default()
        },
    );
    let right = ScrollView::new(
        Rc::new(Text::with_padding("b1\nb2\nb3\nb4\nb5\nb6\nb7", 0, 0)),
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            ..ScrollViewOptions::default()
        },
    );
    tui.set_layout_root(Some(HStack::new(
        vec![
            StackChild::entry(
                left.clone(),
                StackEntryOptions {
                    basis: Some(Basis::Cells(10)),
                    shrink: Some(0),
                    ..StackEntryOptions::default()
                },
            ),
            StackChild::entry(
                right.clone(),
                StackEntryOptions {
                    basis: Some(Basis::Cells(10)),
                    shrink: Some(0),
                    ..StackEntryOptions::default()
                },
            ),
        ],
        StackOptions::default(),
    )));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<64;15;1M");
    wait_for_render(&tui);
    assert_eq!(left.scroll_top(), 3);
    assert_eq!(right.scroll_top(), 2);
    assert_eq!(
        trimmed_viewport(&terminal),
        vec![
            "a4        b3",
            "a5        b4",
            "a6        b5",
            "a7        b6"
        ],
    );
    tui.stop(TuiStopOptions::default());
}

#[test]
fn scrolls_faster_while_alt_is_held_during_wheel_input() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 4);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&[&numbered_lines(12)]));
    tui.start();
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 8);

    // Alt modifier sets bit 8 on the wheel button (72 = 64 + 8).
    terminal.send_input("\x1b[<72;1;1M");
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 3);
    tui.stop(TuiStopOptions::default());
}

#[test]
fn does_not_vertically_redispatch_misses_through_horizontal_layout_containers() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 2);
    let (tui, _alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    let list = Rc::new(ClickList {
        selections: Cell::new(0),
    });
    tui.set_layout_root(Some(HStack::new(
        vec![
            StackChild::entry(
                list.clone(),
                StackEntryOptions {
                    basis: Some(Basis::Cells(10)),
                    ..StackEntryOptions::default()
                },
            ),
            StackChild::entry(
                text(&["plain"]),
                StackEntryOptions {
                    basis: Some(Basis::Cells(10)),
                    ..StackEntryOptions::default()
                },
            ),
        ],
        StackOptions::default(),
    )));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;15;1M");
    terminal.send_input("\x1b[<0;15;1m");
    wait_for_render(&tui);
    assert_eq!(list.selections.get(), 0);
    tui.stop(TuiStopOptions::default());
}

#[test]
fn uses_button_motion_tracking_inside_terminal_multiplexers() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // The multiplexer probes run per renderer construction through the
    // injected lookup, so a fresh map per case replaces the upstream env
    // saves and restores.
    let direct_terminal = RecordingTerminal::new(80, 24);
    let (direct_tui, _alt) = new_recording_tui(
        direct_terminal.clone(),
        TuiAltScreenConfig {
            env_lookup: Some(env_lookup(&HashMap::from([(
                "TERM".to_string(),
                "xterm-256color".to_string(),
            )]))),
            ..TuiAltScreenConfig::default()
        },
    );
    direct_tui.start();
    let direct_writes = direct_terminal.writes();
    assert!(direct_writes.contains("\x1b[?1003h"));
    direct_tui.stop(TuiStopOptions::default());

    let multiplexers = [
        ("tmux environment", vec![("TMUX", "/tmp/tmux/default,1,0")]),
        ("tmux TERM", vec![("TERM", "tmux-256color")]),
        ("Zellij environment", vec![("ZELLIJ", "0")]),
        ("Screen environment", vec![("STY", "123.session")]),
        ("Screen TERM", vec![("TERM", "screen-256color")]),
    ];
    for (name, environment) in multiplexers {
        let map: HashMap<String, String> = environment
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        let terminal = RecordingTerminal::new(80, 24);
        let (tui, _alt) = new_recording_tui(
            terminal.clone(),
            TuiAltScreenConfig {
                env_lookup: Some(env_lookup(&map)),
                ..TuiAltScreenConfig::default()
            },
        );
        tui.start();
        let recorded = terminal.writes();
        assert!(
            recorded.contains("\x1b[?1002h"),
            "{name} should enable button-motion tracking"
        );
        assert!(
            !recorded.contains("\x1b[?1003h"),
            "{name} should not enable all-motion tracking"
        );
        assert!(
            recorded.contains("\x1b[?1006h"),
            "{name} should enable SGR mouse encoding"
        );
        tui.stop(TuiStopOptions::default());
    }
}

#[test]
fn invokes_the_right_click_paste_handler_only_on_windows_outside_vs_code() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(80, 24);
    let paste_count = Rc::new(Cell::new(0));
    let paste_probe = Rc::clone(&paste_count);
    let is_windows = Rc::new(Cell::new(false));
    let windows_probe = Rc::clone(&is_windows);
    let term_program = Rc::new(RefCell::new(None::<String>));
    let term_lookup = Rc::clone(&term_program);
    let env: EnvLookup = Box::new(move |key| {
        if key == "TERM_PROGRAM" {
            return (*term_lookup.borrow()).clone();
        }
        None
    });
    let (tui, _alt) = new_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            on_right_click_paste: Some(Rc::new(move || paste_probe.set(paste_probe.get() + 1))),
            env_lookup: Some(env),
            is_windows: Some(Rc::new(move || windows_probe.get())),
            ..TuiAltScreenConfig::default()
        },
    );
    is_windows.set(true);
    tui.start();
    terminal.send_input("\x1b[<2;1;1M");
    terminal.send_input("\x1b[<2;1;1m");
    assert_eq!(paste_count.get(), 1);

    *term_program.borrow_mut() = Some("vscode".to_string());
    terminal.send_input("\x1b[<2;1;1M");
    assert_eq!(paste_count.get(), 1);

    is_windows.set(false);
    *term_program.borrow_mut() = None;
    terminal.send_input("\x1b[<2;1;1M");
    assert_eq!(paste_count.get(), 1);

    tui.stop(TuiStopOptions::default());
}

#[test]
fn reveals_an_auto_scrollbar_when_the_pointer_enters_its_hidden_track() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(10, 5);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    let scroll_view = ScrollView::new(
        text(&[&numbered_lines(20)]),
        ScrollViewOptions {
            primary: true,
            scrollbar: Some(ScrollViewScrollbar::Auto),
            scrollbar_hide_delay_ms: Some(20),
            ..ScrollViewOptions::default()
        },
    );
    tui.set_layout_root(Some(scroll_view.clone()));
    tui.start();
    wait_for_render(&tui);
    assert!(!scroll_view.is_scrollbar_visible());

    terminal.send_input("\x1b[<35;10;3M");
    // A forced flush: the reveal frame must land before the 20 ms hide
    // worker fires, which the throttled wait's sleep races.
    render_and_flush(&tui);
    assert!(scroll_view.is_scrollbar_visible());
    assert!(scroll_view.is_scrollbar_active());
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains('│') || line.contains('█'))
    );

    terminal.send_input("\x1b[<35;9;3M");
    flush_until(&tui, || !scroll_view.is_scrollbar_visible());
    tui.stop(TuiStopOptions::default());
}

#[test]
fn jumps_to_a_scrollbar_track_position_and_continues_dragging_from_there() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(10, 10);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    let scroll_view = ScrollView::new(
        text(&[&numbered_lines(50)]),
        ScrollViewOptions {
            primary: true,
            scrollbar: Some(ScrollViewScrollbar::Always),
            ..ScrollViewOptions::default()
        },
    );
    tui.set_layout_root(Some(scroll_view.clone()));
    tui.start();
    wait_for_render(&tui);
    assert_eq!(scroll_view.scroll_top(), 0);

    terminal.send_input("\x1b[<0;10;6M");
    wait_for_render(&tui);
    assert_eq!(scroll_view.scroll_top(), 20);

    terminal.send_input("\x1b[<32;10;10M");
    wait_for_render(&tui);
    assert_eq!(scroll_view.scroll_top(), 40);

    terminal.send_input("\x1b[<0;10;10m");
    wait_for_render(&tui);
    assert!(
        terminal
            .writes()
            .lines()
            .all(|line| !line.contains("\x1b]52;c;"))
    );
    tui.stop(TuiStopOptions::default());
}

#[test]
fn chains_unused_wheel_delta_to_an_outer_scroll_view() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 4);
    let (tui, _alt) = new_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            wheel_scroll_lines: Some(3),
            ..TuiAltScreenConfig::default()
        },
    );
    let inner = ScrollView::new(
        Rc::new(Text::with_padding("i1\ni2\ni3\ni4\ni5\ni6", 0, 0)),
        ScrollViewOptions::default(),
    );
    let outer = ScrollView::new(
        VStack::new(
            vec![
                StackChild::entry(
                    inner.clone(),
                    StackEntryOptions {
                        basis: Some(Basis::Cells(2)),
                        ..StackEntryOptions::default()
                    },
                ),
                StackChild::component(text(&["tail1", "tail2", "tail3", "tail4", "tail5"])),
            ],
            StackOptions::default(),
        ),
        ScrollViewOptions {
            primary: true,
            ..ScrollViewOptions::default()
        },
    );
    tui.set_layout_root(Some(outer.clone()));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<65;1;1M");
    wait_for_render(&tui);
    assert_eq!(inner.scroll_top(), 3);
    assert_eq!(outer.scroll_top(), 0);

    terminal.send_input("\x1b[<65;1;1M");
    wait_for_render(&tui);
    assert_eq!(inner.scroll_top(), 4);
    assert_eq!(outer.scroll_top(), 2);
    tui.stop(TuiStopOptions::default());
}

#[test]
fn supports_configurable_keyboard_viewport_navigation_with_four_rows_of_page_overlap() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 8);
    let (tui, _alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&[&numbered_lines(12)]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[57421u");
    terminal.send_input("\x1b[57421;1:3u");
    wait_for_render(&tui);
    assert_eq!(
        trimmed_viewport(&terminal),
        vec![
            "line 1", "line 2", "line 3", "line 4", "line 5", "line 6", "line 7", "line 8",
        ],
    );

    terminal.send_input("\x1b[57422u");
    terminal.send_input("\x1b[57422;1:3u");
    wait_for_render(&tui);
    assert_eq!(
        trimmed_viewport(&terminal),
        vec![
            "line 5", "line 6", "line 7", "line 8", "line 9", "line 10", "line 11", "line 12",
        ],
    );

    terminal.send_input("\x1bOH");
    wait_for_render(&tui);
    assert_eq!(
        trimmed_viewport(&terminal),
        vec![
            "line 1", "line 2", "line 3", "line 4", "line 5", "line 6", "line 7", "line 8",
        ],
    );

    terminal.send_input("\x1bOF");
    wait_for_render(&tui);
    assert_eq!(
        trimmed_viewport(&terminal),
        vec![
            "line 5", "line 6", "line 7", "line 8", "line 9", "line 10", "line 11", "line 12",
        ],
    );

    tui.stop(TuiStopOptions::default());
}

// === Transcript search ===

#[test]
fn searches_normalized_rendered_transcript_text_across_rows() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(
        find_alt_screen_search_matches(
            &["alpha QUICK".to_string(), "brown fox".to_string()],
            "quick brown",
        ),
        vec![AltScreenSearchMatch {
            segments: vec![
                AltScreenSearchSegment {
                    row: 0,
                    start_col: 6,
                    end_col: 11
                },
                AltScreenSearchSegment {
                    row: 1,
                    start_col: 0,
                    end_col: 5
                },
            ],
        }],
    );
}

#[test]
fn maps_normalized_ascii_and_unicode_search_matches_back_to_rendered_columns() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(
        find_alt_screen_search_matches(
            &[
                "\x1b[31mfoo  bar\x1b[0m".to_string(),
                "A\u{754c}\u{1f642}e\u{301}Z".to_string(),
            ],
            "oo   bar\nA\u{754c}\u{1f642}e\u{301}",
        ),
        vec![AltScreenSearchMatch {
            segments: vec![
                AltScreenSearchSegment {
                    row: 0,
                    start_col: 1,
                    end_col: 3
                },
                AltScreenSearchSegment {
                    row: 0,
                    start_col: 5,
                    end_col: 8
                },
                AltScreenSearchSegment {
                    row: 1,
                    start_col: 0,
                    end_col: 6
                },
            ],
        }],
    );
}

#[test]
fn reuses_indexed_transcript_matches_until_the_query_or_rendered_lines_change() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut index = AltScreenSearchIndex::default();
    let initial = index.search(&["alpha needle".to_string(), "omega".to_string()], "needle");
    assert!(initial.changed);
    assert_eq!(initial.matches.len(), 1);

    let cached = index.search(&["alpha needle".to_string(), "omega".to_string()], "needle");
    assert!(!cached.changed);
    assert_eq!(cached.matches, initial.matches);

    let changed_query = index.search(&["alpha needle".to_string(), "omega".to_string()], "omega");
    assert!(changed_query.changed);
    assert_ne!(changed_query.matches, initial.matches);
    assert_eq!(
        changed_query
            .matches
            .first()
            .map(|match_| match_.segments.as_slice()),
        Some(
            &[AltScreenSearchSegment {
                row: 1,
                start_col: 0,
                end_col: 5
            }][..]
        ),
    );

    let changed_lines = index.search(
        &["alpha needle".to_string(), "no match".to_string()],
        "omega",
    );
    assert!(changed_lines.changed);
    assert!(changed_lines.matches.is_empty());
}

#[test]
fn renders_transcript_search_with_a_muted_placeholder_and_right_aligned_controls() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let component = AltScreenSearchComponent::new(Rc::new(|_query: &str| {}), None);
    let rendered = component.render(48);
    let lines: Vec<String> = rendered
        .iter()
        .map(|line| strip_terminal_sequences(line))
        .collect();

    assert_eq!(lines.len(), 3);
    assert!(
        lines
            .iter()
            .all(|line| pi_tui::utils::visible_width(line) == 48)
    );
    assert!(lines[0].starts_with('┌') && lines[0].ends_with('┐'));
    assert!(lines[1].starts_with("│ Find in transcript"));
    assert!(lines[1].ends_with("│"));
    assert!(rendered[1].contains("\x1b[2m"));
    let bottom = &lines[2];
    assert!(bottom.starts_with("└") && bottom.ends_with("┘"));
    assert!(bottom.contains(" ↑ Shift+Enter · ↓ Enter "));
    let arrow_up = char_col(bottom, bottom.find('↑').expect("a previous button"));
    let shifted = char_col(
        bottom,
        bottom.find("Shift+Enter").expect("a previous key label"),
    ) + 5;
    let separator = char_col(bottom, bottom.find('·').expect("a separator"));
    let down = char_col(bottom, bottom.find('↓').expect("a next button"));
    let enter_tail = char_col(bottom, bottom.rfind("Enter").expect("a next key label")) + 2;
    assert_eq!(
        component.get_navigation_direction_at(2, line_i64(arrow_up)),
        Some(-1)
    );
    assert_eq!(
        component.get_navigation_direction_at(2, line_i64(shifted)),
        Some(-1)
    );
    assert_eq!(
        component.get_navigation_direction_at(2, line_i64(separator)),
        None
    );
    assert_eq!(
        component.get_navigation_direction_at(2, line_i64(down)),
        Some(1)
    );
    assert_eq!(
        component.get_navigation_direction_at(2, line_i64(enter_tail)),
        Some(1)
    );

    component.handle_input("n");
    component.set_result(0, 2);
    let populated_render = component.render(48);
    let populated: Vec<String> = populated_render
        .iter()
        .map(|line| strip_terminal_sequences(line))
        .collect();
    assert!(populated[1].contains('n'));
    assert!(populated[1].contains("1/2"));
    assert!(populated_render[1].contains("\x1b[2m 1/2 \x1b[22m"));
    assert!(
        !populated
            .iter()
            .any(|line| line.contains("Find in transcript"))
    );
}

#[test]
fn navigates_transcript_search_with_hoverable_arrow_buttons_and_toggles_it_with_its_shortcut() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(120, 6);
    let (tui, _alt) = new_recording_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            search_navigation_button_style: Some(Rc::new(|text: &str, hovered: bool| {
                format!(
                    "{}{text}\x1b[49m",
                    if hovered { "\x1b[45m" } else { "\x1b[44m" }
                )
            })),
            ..TuiAltScreenConfig::default()
        },
    );
    tui.add_child(text(&["needle one", "middle", "needle two", "end"]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[102;6u");
    terminal.send_input("needle");
    wait_for_render(&tui);
    let viewport = terminal.get_viewport();
    assert!(viewport.iter().any(|line| line.contains("1/2")));
    assert!(
        viewport
            .iter()
            .any(|line| line.contains("↑ Shift+Enter · ↓ Enter"))
    );

    let arrow_row = viewport
        .iter()
        .position(|line| line.contains('↑') && line.contains('↓'))
        .expect("a navigation row");
    let arrow_column = char_col(
        &viewport[arrow_row],
        viewport[arrow_row].rfind("Enter").expect("an enter label"),
    );
    let hover_event_count = terminal.event_count();
    terminal.send_input(&format!("\x1b[<35;{};{}M", arrow_column + 1, arrow_row + 1));
    wait_for_render(&tui);
    assert!(
        terminal
            .writes_from(hover_event_count)
            .contains("\x1b[45m↓ Enter\x1b[49m")
    );
    terminal.send_input(&format!("\x1b[<0;{};{}M", arrow_column + 1, arrow_row + 1));
    wait_for_render(&tui);
    let viewport = terminal.get_viewport();
    assert!(viewport.iter().any(|line| line.contains("2/2")));
    assert!(
        viewport
            .iter()
            .any(|line| line.contains("↑ Shift+Enter · ↓ Enter"))
    );

    let arrow_row = viewport
        .iter()
        .position(|line| line.contains('↑') && line.contains('↓'))
        .expect("a navigation row");
    let arrow_column = char_col(
        &viewport[arrow_row],
        viewport[arrow_row]
            .find("Shift+Enter")
            .expect("a previous key label"),
    ) + 3;
    terminal.send_input(&format!("\x1b[<0;{};{}M", arrow_column + 1, arrow_row + 1));
    wait_for_render(&tui);
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("1/2"))
    );
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("↑ Shift+Enter · ↓ Enter"))
    );

    terminal.send_input("\x1b[102;6u");
    wait_for_render(&tui);
    assert!(
        !terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("↑ Shift+Enter · ↓ Enter"))
    );
    tui.stop(TuiStopOptions::default());
}

#[test]
fn does_not_treat_transcript_box_drawing_as_search_navigation_buttons() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(80, 10);
    let (tui, _alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&[
        "needle one",
        "middle",
        "needle two",
        "filler",
        "┌────────────────────────────────────────┐",
        "│ box                                    │",
        "└────────────────────────────────────────┘",
        "end",
    ]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[102;6u");
    terminal.send_input("needle");
    wait_for_render(&tui);
    let viewport = terminal.get_viewport();
    assert!(viewport.iter().any(|line| line.contains("1/2")));
    assert!(!viewport.iter().any(|line| line.contains("2/2")));

    let box_bottom_row = viewport
        .iter()
        .position(|line| line.starts_with('└'))
        .expect("a box bottom row");
    terminal.send_input(&format!("\x1b[<0;24;{}M", box_bottom_row + 1));
    wait_for_render(&tui);

    let viewport = terminal.get_viewport();
    assert!(viewport.iter().any(|line| line.contains("1/2")));
    assert!(!viewport.iter().any(|line| line.contains("2/2")));
    tui.stop(TuiStopOptions::default());
}

#[test]
fn uses_configured_styles_for_current_and_non_current_search_matches() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(60, 4);
    let (tui, _alt) = new_recording_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            search_match_style: Some(Rc::new(|text: &str| format!("\x1b[41m{text}\x1b[49m"))),
            search_current_match_style: Some(Rc::new(|text: &str| {
                format!("\x1b[42m{text}\x1b[49m")
            })),
            ..TuiAltScreenConfig::default()
        },
    );
    tui.add_child(text(&["needle first", "middle", "needle second", "end"]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[102;6u");
    terminal.send_input("needle");
    wait_for_render(&tui);

    assert!(terminal.writes().contains("\x1b[42mneedle\x1b[49m"));
    assert!(terminal.writes().contains("\x1b[41mneedle\x1b[49m"));
    tui.stop(TuiStopOptions::default());
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "mirrors upstream's longest search-suite fixture step for step"
)]
fn searches_the_transcript_with_ctrl_shift_f_and_restores_editor_focus_on_close() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(60, 8);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    let transcript_lines: Vec<String> = (0..12)
        .map(|index| {
            if index == 4 {
                "line 5 needle one".to_string()
            } else if index == 9 {
                "line 10 needle two".to_string()
            } else {
                format!("line {}", index + 1)
            }
        })
        .collect();
    let transcript = ScrollView::new(
        Rc::new(Text::with_padding(transcript_lines.join("\n"), 0, 0)),
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            primary: true,
            ..ScrollViewOptions::default()
        },
    );
    let editor_inputs: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let editor = Rc::new(RecordingComponent {
        lines: RefCell::new(vec!["editor".to_string()]),
        inputs: Rc::clone(&editor_inputs),
        focused: Cell::new(false),
    });
    tui.set_layout_root(Some(VStack::new(
        vec![
            scroll_entry(transcript.clone(), 0, 1, 1),
            StackChild::entry(
                editor.clone(),
                StackEntryOptions {
                    basis: Some(Basis::Cells(1)),
                    shrink: Some(0),
                    ..StackEntryOptions::default()
                },
            ),
        ],
        StackOptions::default(),
    )));
    tui.set_focus(Some(editor));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[102;6u");
    terminal.send_input("needle");
    wait_for_render(&tui);
    assert!(!transcript.is_following_end());
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("2/2"))
    );
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("↑ Shift+Enter · ↓ Enter"))
    );
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("line 10 needle two"))
    );
    assert!(editor_inputs.borrow().is_empty());
    assert!(terminal.writes().contains("\x1b[1;7mneedle\x1b[22;27m"));

    for _ in 0..6 {
        terminal.send_input("\x1b[<64;1;4M");
    }
    wait_for_render(&tui);
    assert_eq!(transcript.scroll_top(), 0);
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("needle") && line.contains("2/2"))
    );

    terminal.send_input("\x07");
    wait_for_render(&tui);
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("1/2"))
    );
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("line 5 needle one"))
    );

    terminal.send_input("\x1b[103;6u");
    wait_for_render(&tui);
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("2/2"))
    );
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("line 10 needle two"))
    );

    terminal.send_input("\x1b");
    terminal.send_input("x");
    wait_for_render(&tui);
    assert!(
        !terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("↑ Shift+Enter · ↓ Enter"))
    );
    assert_eq!(*editor_inputs.borrow(), vec!["x".to_string()]);

    tui.stop(TuiStopOptions::default());
}

/// The editor stand-in the focus suites use: fixed lines beside an input
/// recorder, upstream's object literal.
struct RecordingComponent {
    lines: RefCell<Vec<String>>,
    inputs: Rc<RefCell<Vec<String>>>,
    focused: Cell<bool>,
}

impl pi_tui::tui::Focusable for RecordingComponent {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

impl Component for RecordingComponent {
    fn render(&self, _width: usize) -> Vec<String> {
        self.lines.borrow().clone()
    }

    fn handle_input(&self, data: &str) {
        self.inputs.borrow_mut().push(data.to_string());
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {}

    fn as_focusable(&self) -> Option<&dyn pi_tui::tui::Focusable> {
        Some(self)
    }
}

#[test]
fn scrolls_the_transcript_by_half_a_page_with_custom_bindings() {
    let _guard = KEYBINDINGS_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 10);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    set_keybindings(KeybindingsManager::with_user_bindings(
        Keybindings::tui_defaults(),
        KeybindingsConfig::new()
            .bind("tui.altScreen.halfPageUp", ["ctrl+u"])
            .bind("tui.altScreen.halfPageDown", ["ctrl+d"]),
    ));
    tui.add_child(text(&[&numbered_lines(30)]));
    tui.start();
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 20);

    terminal.send_input("\x15");
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 15);

    terminal.send_input("\x04");
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 20);

    tui.stop(TuiStopOptions::default());
    set_keybindings(KeybindingsManager::new(Keybindings::tui_defaults()));
}

#[test]
fn scrolls_the_transcript_by_one_line_with_custom_bindings() {
    let _guard = KEYBINDINGS_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 10);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    set_keybindings(KeybindingsManager::with_user_bindings(
        Keybindings::tui_defaults(),
        KeybindingsConfig::new()
            .bind("tui.altScreen.lineUp", ["ctrl+y"])
            .bind("tui.altScreen.lineDown", ["ctrl+e"]),
    ));
    tui.add_child(text(&[&numbered_lines(30)]));
    tui.start();
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 20);

    terminal.send_input("\x19");
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 19);

    terminal.send_input("\x05");
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 20);

    tui.stop(TuiStopOptions::default());
    set_keybindings(KeybindingsManager::new(Keybindings::tui_defaults()));
}

#[test]
fn routes_ctrl_modified_viewport_navigation_to_the_focused_component() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 6);
    let (tui, _alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    let transcript = ScrollView::new(
        text(&[&numbered_lines(12)]),
        ScrollViewOptions {
            follow: Some(FollowMode::End),
            primary: true,
            ..ScrollViewOptions::default()
        },
    );
    let editor_inputs: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let editor = Rc::new(RecordingComponent {
        lines: RefCell::new(vec!["editor".to_string()]),
        inputs: Rc::clone(&editor_inputs),
        focused: Cell::new(false),
    });
    tui.set_layout_root(Some(VStack::new(
        vec![
            scroll_entry(transcript.clone(), 0, 1, 1),
            StackChild::entry(
                editor.clone(),
                StackEntryOptions {
                    basis: Some(Basis::Cells(1)),
                    shrink: Some(0),
                    ..StackEntryOptions::default()
                },
            ),
        ],
        StackOptions::default(),
    )));
    tui.set_focus(Some(editor));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1bOH");
    wait_for_render(&tui);
    assert_eq!(transcript.scroll_top(), 0);
    assert!(editor_inputs.borrow().is_empty());

    let modified_inputs = [
        "\x1b[1;5H",
        "\x1b[1;5F",
        "\x1b[5;5~",
        "\x1b[6;5~",
        "\x1b[57423;5u",
    ];
    for input in modified_inputs {
        terminal.send_input(input);
    }
    terminal.send_input("\x1b[57423;5:3u");
    wait_for_render(&tui);
    assert_eq!(transcript.scroll_top(), 0);
    assert_eq!(
        *editor_inputs.borrow(),
        modified_inputs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
    );

    terminal.send_input("\x1b[6~");
    wait_for_render(&tui);
    assert_eq!(transcript.scroll_top(), 1);
    assert_eq!(
        *editor_inputs.borrow(),
        modified_inputs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
    );

    tui.stop(TuiStopOptions::default());
}

#[test]
fn jumps_between_osc_133_semantic_prompt_markers() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 3);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    let message_lines: Vec<String> = [1, 2, 3, 4]
        .iter()
        .flat_map(|message| {
            vec![
                format!("{OSC133_ZONE_START}message {message}"),
                "detail".to_string(),
            ]
        })
        .collect();
    tui.add_child(Rc::new(Text::with_padding(message_lines.join("\n"), 0, 0)));
    tui.start();
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 5);

    terminal.send_input("\x1b[57419;6u");
    terminal.send_input("\x1b[57419;6:3u");
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 4);
    assert_eq!(terminal.get_viewport()[0].trim_end(), "message 3");

    terminal.send_input("\x1b[1;6A");
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 2);
    assert_eq!(terminal.get_viewport()[0].trim_end(), "message 2");

    terminal.send_input("\x1b[57420;6u");
    terminal.send_input("\x1b[57420;6:3u");
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 4);
    assert_eq!(terminal.get_viewport()[0].trim_end(), "message 3");

    terminal.send_input("\x1b[1;6B");
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 5);
    assert_eq!(terminal.get_viewport()[1].trim_end(), "message 4");
    assert!(alt.is_following_output());

    tui.stop(TuiStopOptions::default());
}

#[test]
fn does_not_emit_kitty_graphics_commands_or_osc_133_zones_in_iterm2() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    set_capabilities(pi_tui::terminal_image::TerminalCapabilities {
        images: Some(pi_tui::terminal_image::ImageProtocol::Iterm2),
        true_color: true,
        hyperlinks: true,
    });
    let terminal = RecordingTerminal::new(20, 3);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(Rc::new(FixtureComponent {
        lines: RefCell::new(vec![
            "\x1b]133;B\x07\x1b]133;C\x07\x1b]133;A\x07content".to_string(),
        ]),
    }));
    tui.add_child(TestImage::new(
        "AAAA",
        8,
        4,
        1,
        10,
        10,
        "[Image: example.png image/png 10x10]",
    ));
    tui.start();
    wait_for_render(&tui);
    tui.stop(TuiStopOptions::default());
    assert!(!terminal.writes().contains("\x1b_G"));
    assert!(!terminal.writes().contains("\x1b]133;"));
    assert!(!terminal.writes().contains("\x1b]1337;File="));
    assert!(terminal.writes().contains("[Image:"));
    reset_capabilities_cache();
}

/// A fixed-line component the fixtures use in place of an inline object
/// literal, upstream `{ render: () => [...], invalidate: () => {} }`.
struct FixtureComponent {
    lines: RefCell<Vec<String>>,
}

impl Component for FixtureComponent {
    fn render(&self, _width: usize) -> Vec<String> {
        self.lines.borrow().clone()
    }

    fn invalidate(&self) {}
}

#[test]
fn clears_stale_iterm2_image_placements_when_they_leave_the_viewport() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    set_capabilities(pi_tui::terminal_image::TerminalCapabilities {
        images: Some(pi_tui::terminal_image::ImageProtocol::Iterm2),
        true_color: true,
        hyperlinks: true,
    });
    let terminal = RecordingTerminal::new(20, 3);
    let (tui, alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    let image_line = "\x1b]1337;File=inline=1;width=2;height=auto:AAAA\x07".to_string();
    tui.add_child(Rc::new(FixtureComponent {
        lines: RefCell::new(vec![
            image_line,
            String::new(),
            String::new(),
            "after".to_string(),
            "more".to_string(),
            "end".to_string(),
        ]),
    }));
    tui.start();
    wait_for_render(&tui);
    alt.scroll_to_top();
    // A forced flush: the top=0 frame must be on the terminal before the
    // marker, or scrollBy(1) coalesces with it and the image never crosses
    // the viewport edge under parallel test load.
    render_and_flush(&tui);
    let event_count = terminal.event_count();

    alt.scroll_by(1);
    wait_for_render(&tui);
    let tail = terminal.writes_from(event_count);
    assert!(
        tail.contains("\x1b[2J"),
        "no 2J after scroll-by; tail: {tail}"
    );
    tui.stop(TuiStopOptions::default());
    reset_capabilities_cache();
}

#[test]
fn crops_a_kitty_image_whose_first_line_is_above_the_viewport() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 3);
    let (tui, alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    let image_id = 123;
    let image_line = encode_kitty(
        "AAAA",
        EncodeKittyOptions {
            columns: Some(2),
            rows: Some(3),
            image_id: Some(image_id),
            move_cursor: Some(false),
        },
    );
    register_kitty_image_metadata(KittyImageMetadata {
        image_id,
        columns: 2,
        rows: 3,
        width_px: 100,
        height_px: 100,
    });
    tui.add_child(Rc::new(FixtureComponent {
        lines: RefCell::new(vec![
            "before".to_string(),
            image_line,
            String::new(),
            String::new(),
            "after".to_string(),
            "end".to_string(),
        ]),
    }));
    tui.start();
    wait_for_render(&tui);

    assert_eq!(alt.viewport_top(), 3);
    let writes = terminal.writes();
    assert!(writes.contains("i=123") && writes.contains("y=66,h=34,r=1"));

    tui.stop(TuiStopOptions::default());
    reset_capabilities_cache();
}

#[test]
fn reuses_moved_kitty_images_without_dropping_hstack_siblings() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    set_capabilities(pi_tui::terminal_image::TerminalCapabilities {
        images: Some(pi_tui::terminal_image::ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: true,
    });
    let terminal = RecordingTerminal::new(20, 6);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    let label = text(&["left"]);
    let image = TestImage::new(
        &"A".repeat(8192),
        8,
        4,
        1,
        100,
        100,
        "[Image: image/png 100x100]",
    );
    let header = text(&["header"]);
    let row = HStack::new(
        vec![
            StackChild::entry(
                label.clone(),
                StackEntryOptions {
                    basis: Some(Basis::Cells(10)),
                    ..StackEntryOptions::default()
                },
            ),
            StackChild::entry(
                image,
                StackEntryOptions {
                    basis: Some(Basis::Cells(10)),
                    ..StackEntryOptions::default()
                },
            ),
        ],
        StackOptions::default(),
    );
    tui.set_layout_root(Some(VStack::new(
        vec![
            StackChild::entry(header.clone(), StackEntryOptions::default()),
            StackChild::entry(
                row,
                StackEntryOptions {
                    basis: Some(Basis::Cells(4)),
                    ..StackEntryOptions::default()
                },
            ),
        ],
        StackOptions::default(),
    )));
    tui.start();
    wait_for_render(&tui);
    assert!(terminal.writes().contains("\x1b_Ga=T"));

    let event_count = terminal.event_count();
    label.set_text("changed");
    header.set_text("header\nsecond");
    tui.request_render(false);
    wait_for_render(&tui);
    let redraw_writes = terminal.writes_from(event_count);
    let placement_index = redraw_writes.find("\x1b_Ga=p,q=2").expect("a placement");
    assert!(redraw_writes.contains("\x1b_Ga=d,d=a,q=2\x1b\\"));
    assert!(placement_index > redraw_writes.find("changed").expect("a label write"));
    assert!(!redraw_writes.contains("\x1b_Ga=T"));
    assert!(
        redraw_writes.len() < 2000,
        "expected placement-only redraw, got {} bytes",
        redraw_writes.len()
    );
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.trim_end() == "changed")
    );
    tui.stop(TuiStopOptions::default());
    reset_capabilities_cache();
}

#[test]
fn retains_recently_offscreen_kitty_images_for_placement_only_reuse() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    set_capabilities(pi_tui::terminal_image::TerminalCapabilities {
        images: Some(pi_tui::terminal_image::ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: true,
    });
    let terminal = RecordingTerminal::new(20, 1);
    let (tui, alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    let image_id = 321;
    let image_line = encode_kitty(
        "AAAA",
        EncodeKittyOptions {
            columns: Some(2),
            rows: Some(1),
            image_id: Some(image_id),
            move_cursor: Some(false),
        },
    );
    register_kitty_image_metadata(KittyImageMetadata {
        image_id,
        columns: 2,
        rows: 1,
        width_px: 100,
        height_px: 50,
    });
    tui.set_layout_root(Some(ScrollView::new(
        Rc::new(FixtureComponent {
            lines: RefCell::new(vec![image_line, "after".to_string()]),
        }),
        ScrollViewOptions {
            primary: true,
            ..ScrollViewOptions::default()
        },
    )));
    tui.start();
    wait_for_render(&tui);
    assert!(terminal.writes().contains("\x1b_Ga=T"));

    let event_count = terminal.event_count();
    alt.scroll_by(1);
    wait_for_render(&tui);
    alt.scroll_by(-1);
    wait_for_render(&tui);
    let reentry_writes = terminal.writes_from(event_count);
    assert!(reentry_writes.contains("\x1b_Ga=p,q=2"));
    assert!(!reentry_writes.contains("\x1b_Ga=T"));
    assert!(!reentry_writes.contains(&format!("\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\")));
    tui.stop(TuiStopOptions::default());
    reset_capabilities_cache();
}

#[test]
fn evicts_the_least_recently_visible_kitty_image_when_the_cache_is_full() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    set_capabilities(pi_tui::terminal_image::TerminalCapabilities {
        images: Some(pi_tui::terminal_image::ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: true,
    });
    let terminal = RecordingTerminal::new(20, 1);
    let (tui, alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    let first_image_id = 500;
    let image_lines: Vec<String> = (0..18)
        .map(|index| {
            let image_id = first_image_id + u64::try_from(index).unwrap_or(u64::MAX);
            register_kitty_image_metadata(KittyImageMetadata {
                image_id,
                columns: 2,
                rows: 1,
                width_px: 100,
                height_px: 50,
            });
            encode_kitty(
                "AAAA",
                EncodeKittyOptions {
                    columns: Some(2),
                    rows: Some(1),
                    image_id: Some(image_id),
                    move_cursor: Some(false),
                },
            )
        })
        .collect();
    tui.set_layout_root(Some(ScrollView::new(
        Rc::new(FixtureComponent {
            lines: RefCell::new(image_lines),
        }),
        ScrollViewOptions {
            primary: true,
            ..ScrollViewOptions::default()
        },
    )));
    tui.start();
    wait_for_render(&tui);
    for _ in 1..18 {
        alt.scroll_by(1);
        wait_for_render(&tui);
    }
    assert!(
        terminal
            .writes()
            .contains(&format!("\x1b_Ga=d,d=I,i={first_image_id},q=2\x1b\\"))
    );

    let event_count = terminal.event_count();
    alt.scroll_to_top();
    wait_for_render(&tui);
    let reentry_writes = terminal.writes_from(event_count);
    assert!(reentry_writes.contains("\x1b_Ga=T"));
    tui.stop(TuiStopOptions::default());
    reset_capabilities_cache();
}

#[test]
fn evicts_offscreen_kitty_images_when_decoded_raster_memory_exceeds_the_cache_quota() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    set_capabilities(pi_tui::terminal_image::TerminalCapabilities {
        images: Some(pi_tui::terminal_image::ImageProtocol::Kitty),
        true_color: true,
        hyperlinks: true,
    });
    let terminal = RecordingTerminal::new(20, 1);
    let (tui, alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    let first_image_id = 600;
    let image_lines: Vec<String> = (0..4)
        .map(|index| {
            let image_id = first_image_id + u64::try_from(index).unwrap_or(u64::MAX);
            register_kitty_image_metadata(KittyImageMetadata {
                image_id,
                columns: 2,
                rows: 1,
                width_px: 3840,
                height_px: 2160,
            });
            encode_kitty(
                "AAAA",
                EncodeKittyOptions {
                    columns: Some(2),
                    rows: Some(1),
                    image_id: Some(image_id),
                    move_cursor: Some(false),
                },
            )
        })
        .collect();
    tui.set_layout_root(Some(ScrollView::new(
        Rc::new(FixtureComponent {
            lines: RefCell::new(image_lines),
        }),
        ScrollViewOptions {
            primary: true,
            ..ScrollViewOptions::default()
        },
    )));
    tui.start();
    wait_for_render(&tui);
    for _ in 1..4 {
        alt.scroll_by(1);
        wait_for_render(&tui);
    }
    assert!(
        terminal
            .writes()
            .contains(&format!("\x1b_Ga=d,d=I,i={first_image_id},q=2\x1b\\"))
    );
    tui.stop(TuiStopOptions::default());
    reset_capabilities_cache();
}

#[test]
fn opens_an_osc_8_hyperlink_with_specific_or_generic_release_codes_but_not_on_drag() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 3);
    let opened_urls: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let opened_probe = Rc::clone(&opened_urls);
    let (tui, _alt) = new_recording_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            open_url: Some(Rc::new(move |url: &str| {
                opened_probe.borrow_mut().push(url.to_string());
            })),
            ..TuiAltScreenConfig::default()
        },
    );
    let url = "https://example.com/path?q=1";
    let bel_url = "https://example.com/bel";
    let emoji_url = "https://example.com/emoji";
    tui.add_child(Rc::new(Text::with_padding(
        format!(
            "{}\n\x1b]8;;{bel_url}\x07link\x1b]8;;\x07\n{}",
            hyperlink("link", url),
            hyperlink("🙂", emoji_url)
        ),
        0,
        0,
    )));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;2;1M");
    terminal.send_input("\x1b[<3;2;1m");
    wait_for_render(&tui);
    assert_eq!(*opened_urls.borrow(), vec![url.to_string()]);

    terminal.send_input("\x1b[<0;2;2M");
    terminal.send_input("\x1b[<0;2;2m");
    wait_for_render(&tui);
    assert_eq!(
        *opened_urls.borrow(),
        vec![url.to_string(), bel_url.to_string()]
    );

    terminal.send_input("\x1b[<0;2;3M");
    terminal.send_input("\x1b[<0;2;3m");
    wait_for_render(&tui);
    assert_eq!(
        *opened_urls.borrow(),
        vec![url.to_string(), bel_url.to_string(), emoji_url.to_string()],
    );

    terminal.send_input("\x1b[<0;2;1M");
    terminal.send_input("\x1b[<32;4;1M");
    terminal.send_input("\x1b[<0;4;1m");
    wait_for_render(&tui);
    assert_eq!(
        *opened_urls.borrow(),
        vec![url.to_string(), bel_url.to_string(), emoji_url.to_string()],
    );

    tui.stop(TuiStopOptions::default());
}

#[test]
fn selects_visible_text_with_the_mouse_and_copies_it_with_osc_52_after_a_generic_release() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 4);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(Rc::new(Text::with_padding(
        "\x1b[1mal\x1b[0mpha\nbeta\ngamma\ndelta",
        0,
        0,
    )));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<32;4;2M");
    terminal.send_input("\x1b[<3;4;2m");
    wait_for_render(&tui);

    let expected_clipboard_sequence = osc52("alpha\nbeta");
    assert!(
        clipboard_writes_contain(&terminal, "\x1b]52;c;"),
        "{}",
        clipboard_writes_containing(&terminal, "\x1b]52;c;"),
    );
    assert!(clipboard_writes_contain(
        &terminal,
        &expected_clipboard_sequence
    ));
    assert!(terminal.writes().contains("\x1b[7m"));
    assert!(
        terminal.writes().contains("al\x1b[0m\x1b[7mpha"),
        "selection inverse must be reapplied after a reset inside the selection",
    );
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("Copied!"))
    );

    tui.stop(TuiStopOptions::default());
}

fn clipboard_writes_contain(terminal: &RecordingTerminal, needle: &str) -> bool {
    terminal
        .events
        .borrow()
        .iter()
        .any(|event| matches!(event, RecordingEvent::Write(data) if data.contains(needle)))
}

fn clipboard_writes_containing(terminal: &RecordingTerminal, needle: &str) -> String {
    terminal
        .events
        .borrow()
        .iter()
        .filter_map(|event| match event {
            RecordingEvent::Write(data) if data.contains(needle) => Some(data.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

#[test]
fn uses_an_injected_copy_selection_handler_instead_of_osc_52_and_reports_success() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 4);
    let copied: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let copied_probe = Rc::clone(&copied);
    let (tui, _alt) = new_recording_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            copy_selection: Some(Rc::new(move |text: &str| {
                copied_probe.borrow_mut().push(text.to_string());
                CopySelectionResult::Copied
            })),
            ..TuiAltScreenConfig::default()
        },
    );
    tui.add_child(text(&["alpha", "beta", "gamma", "delta"]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<32;4;2M");
    terminal.send_input("\x1b[<0;4;2m");
    wait_for_render(&tui);

    assert_eq!(*copied.borrow(), vec!["alpha\nbeta".to_string()]);
    assert!(
        !clipboard_writes_contain(&terminal, "\x1b]52;c;"),
        "must not emit OSC 52 when a copySelection handler is provided",
    );
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("Copied!"))
    );

    tui.stop(TuiStopOptions::default());
}

#[test]
fn leaves_selections_visible_without_copying_when_copy_on_select_is_disabled() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 4);
    let copied: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let copied_probe = Rc::clone(&copied);
    let (tui, alt) = new_recording_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            copy_on_select: Some(false),
            copy_selection: Some(Rc::new(move |text: &str| {
                copied_probe.borrow_mut().push(text.to_string());
                CopySelectionResult::Copied
            })),
            ..TuiAltScreenConfig::default()
        },
    );
    tui.add_child(text(&["alpha", "beta", "gamma", "delta"]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<32;4;2M");
    terminal.send_input("\x1b[<0;4;2m");
    wait_for_render(&tui);

    assert!(copied.borrow().is_empty());
    assert!(alt.has_active_selection());
    assert!(terminal.writes().contains("\x1b[7m"));
    assert!(
        terminal
            .get_viewport()
            .iter()
            .all(|line| !line.contains("Copied!"))
    );

    tui.stop(TuiStopOptions::default());
}

#[test]
fn copies_an_active_selection_programmatically() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 4);
    let copied: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let copied_probe = Rc::clone(&copied);
    let (tui, alt) = new_recording_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            copy_selection: Some(Rc::new(move |text: &str| {
                copied_probe.borrow_mut().push(text.to_string());
                CopySelectionResult::Copied
            })),
            ..TuiAltScreenConfig::default()
        },
    );
    tui.add_child(text(&["alpha", "beta", "gamma", "delta"]));
    tui.start();
    wait_for_render(&tui);

    assert!(!alt.has_active_selection());
    assert!(!alt.copy_active_selection_to_clipboard());

    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<32;4;2M");
    terminal.send_input("\x1b[<0;4;2m");
    wait_for_render(&tui);

    assert_eq!(*copied.borrow(), vec!["alpha\nbeta".to_string()]);
    assert!(alt.has_active_selection());

    copied.borrow_mut().clear();
    assert!(alt.copy_active_selection_to_clipboard());
    wait_for_render(&tui);

    assert_eq!(*copied.borrow(), vec!["alpha\nbeta".to_string()]);
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("Copied!"))
    );

    tui.stop(TuiStopOptions::default());
}

#[test]
fn flashes_an_error_when_the_injected_copy_selection_handler_fails() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 4);
    let (tui, _alt) = new_recording_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            copy_selection: Some(Rc::new(|_text: &str| CopySelectionResult::Failed)),
            ..TuiAltScreenConfig::default()
        },
    );
    tui.add_child(text(&["alpha", "beta", "gamma", "delta"]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<32;4;2M");
    terminal.send_input("\x1b[<0;4;2m");
    wait_for_render(&tui);

    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("Copy failed"))
    );
    assert!(
        !clipboard_writes_contain(&terminal, "\x1b]52;c;"),
        "must not emit OSC 52 when a copySelection handler is provided",
    );

    tui.stop(TuiStopOptions::default());
}

#[test]
fn flashes_a_specific_error_returned_by_the_injected_copy_selection_handler() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Regression test for #9618. Upstream intercepted `tui.flash` to pin the
    // 5000 ms error duration; the port pins it behaviorally — the message
    // outlives the one-second default flash.
    let terminal = RecordingTerminal::new(80, 4);
    let (tui, alt) = new_recording_tui(
        terminal.clone(),
        TuiAltScreenConfig {
            copy_on_select: Some(false),
            copy_selection: Some(Rc::new(|_text: &str| {
                CopySelectionResult::Message(
                    "Clipboard unavailable: install wl-clipboard".to_string(),
                )
            })),
            ..TuiAltScreenConfig::default()
        },
    );
    tui.add_child(text(&["alpha", "beta", "gamma", "delta"]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<32;4;2M");
    terminal.send_input("\x1b[<0;4;2m");
    wait_for_render(&tui);
    assert!(!alt.copy_active_selection_to_clipboard());
    wait_for_render(&tui);
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("Clipboard unavailable: install wl-clipboard"))
    );
    assert!(
        terminal
            .get_viewport()
            .iter()
            .all(|line| !line.contains("Copy failed"))
    );
    sleep_ms(1200);
    wait_for_render(&tui);
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("Clipboard unavailable: install wl-clipboard")),
        "the copy-error flash must outlive the one-second default",
    );

    tui.stop(TuiStopOptions::default());
}

#[test]
fn does_not_append_whitespace_to_double_click_word_highlighting() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 1);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&["foo  bar"]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<0;1;1m");
    terminal.send_input("\x1b[<0;3;1M");
    wait_for_render(&tui);

    assert!(terminal.writes().contains("foo\x1b[27m"));
    tui.stop(TuiStopOptions::default());
}

#[test]
fn coalesces_slash_and_hyphen_separated_segments_for_double_click_word_selection() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for (line, needle) in [
        ("extensions/starline/fixed-editor/compositor.ts", "starline"),
        ("earendil-works/pi-tui", "works"),
    ] {
        let copied: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let copied_probe = Rc::clone(&copied);
        let terminal = RecordingTerminal::new(80, 1);
        let (tui, _alt) = new_recording_tui(
            terminal.clone(),
            TuiAltScreenConfig {
                copy_selection: Some(Rc::new(move |text: &str| {
                    copied_probe.borrow_mut().push(text.to_string());
                    CopySelectionResult::Copied
                })),
                ..TuiAltScreenConfig::default()
            },
        );
        tui.add_child(text(&[line]));
        tui.start();
        wait_for_render(&tui);

        let one_based_click_column = line.find(needle).expect("the needle") + 1;
        for _ in 0..2 {
            terminal.send_input(&format!("\x1b[<0;{one_based_click_column};1M"));
            terminal.send_input(&format!("\x1b[<0;{one_based_click_column};1m"));
        }
        wait_for_render(&tui);

        assert_eq!(*copied.borrow(), vec![line.to_string()]);
        tui.stop(TuiStopOptions::default());
    }
}

#[test]
fn highlights_a_complete_whitespace_segment_during_a_word_drag() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 1);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&["foo  bar"]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<0;1;1m");
    terminal.send_input("\x1b[<0;2;1M");
    terminal.send_input("\x1b[<32;4;1M");
    terminal.send_input("\x1b[<0;4;1m");
    wait_for_render(&tui);

    assert!(terminal.writes().contains("foo  \x1b[27m"));
    tui.stop(TuiStopOptions::default());
}

#[test]
fn selects_whole_words_on_double_click_extends_word_drags_and_selects_lines_on_triple_click() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 2);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&["zero alpha beta", "gamma delta"]));
    tui.start();
    wait_for_render(&tui);

    // The second click lands on a different character in alpha.
    terminal.send_input("\x1b[<0;6;1M");
    terminal.send_input("\x1b[<0;6;1m");
    terminal.send_input("\x1b[<0;10;1M");
    terminal.send_input("\x1b[<0;10;1m");
    wait_for_render(&tui);
    assert!(clipboard_writes_contain(&terminal, &osc52("alpha")));

    // A double-click drag includes each word touched, rather than partial words.
    terminal.send_input("\x1b[<0;12;1M");
    terminal.send_input("\x1b[<0;12;1m");
    terminal.send_input("\x1b[<0;14;1M");
    terminal.send_input("\x1b[<32;3;2M");
    terminal.send_input("\x1b[<0;3;2m");
    wait_for_render(&tui);
    assert!(clipboard_writes_contain(&terminal, &osc52("beta\ngamma")));

    terminal.send_input("\x1b[<0;7;2M");
    terminal.send_input("\x1b[<0;7;2m");
    terminal.send_input("\x1b[<0;9;2M");
    terminal.send_input("\x1b[<0;9;2m");
    terminal.send_input("\x1b[<0;11;2M");
    terminal.send_input("\x1b[<0;11;2m");
    wait_for_render(&tui);
    assert!(clipboard_writes_contain(&terminal, &osc52("gamma delta")));

    tui.stop(TuiStopOptions::default());
}

#[test]
fn does_not_repaint_idle_or_zero_width_selections_on_focus_loss() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 4);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&["alpha", "beta", "gamma", "delta"]));
    tui.start();
    wait_for_render(&tui);

    let write_count = |terminal: &RecordingTerminal| {
        terminal
            .events
            .borrow()
            .iter()
            .filter(|event| matches!(event, RecordingEvent::Write(_)))
            .count()
    };
    let clipboard_write_count = |terminal: &RecordingTerminal| {
        terminal
            .events
            .borrow()
            .iter()
            .filter(
                |event| matches!(event, RecordingEvent::Write(data) if data.contains("\x1b]52;c;")),
            )
            .count()
    };

    let idle_write_count = write_count(&terminal);
    terminal.send_input("\x1b[O");
    terminal.send_input("\x1b[I");
    wait_for_render(&tui);
    assert_eq!(write_count(&terminal), idle_write_count);

    // A completed click leaves a zero-width anchor, but later orphaned
    // drag/release events must not extend it.
    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<0;1;1m");
    terminal.send_input("\x1b[<32;4;2M");
    terminal.send_input("\x1b[<0;4;2m");
    wait_for_render(&tui);
    assert_eq!(clipboard_write_count(&terminal), 0);

    // Losing focus after a press without a drag cancels the press without
    // repainting.
    terminal.send_input("\x1b[<0;1;3M");
    wait_for_render(&tui);
    let pressed_write_count = write_count(&terminal);
    terminal.send_input("\x1b[O");
    terminal.send_input("\x1b[I");
    wait_for_render(&tui);
    assert_eq!(write_count(&terminal), pressed_write_count);
    terminal.send_input("\x1b[<32;4;2M");
    terminal.send_input("\x1b[<0;4;2m");
    wait_for_render(&tui);
    assert_eq!(clipboard_write_count(&terminal), 0);
    assert!(terminal.writes().contains("\x1b[?1004h"));

    tui.stop(TuiStopOptions::default());
    assert!(terminal.writes().contains("\x1b[?1004l"));
}

#[test]
fn clears_an_active_visible_selection_on_focus_loss_and_ignores_orphan_events() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 4);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&["alpha", "beta", "gamma", "delta"]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<32;4;2M");
    wait_for_render(&tui);
    let focus_loss_event_count = terminal.event_count();
    terminal.send_input("\x1b[O");
    terminal.send_input("\x1b[I");
    wait_for_render(&tui);
    let focus_loss_writes = terminal.writes_from(focus_loss_event_count);
    assert!(focus_loss_writes.contains("alpha"));
    assert!(focus_loss_writes.contains("beta"));
    assert!(!focus_loss_writes.contains("\x1b[7m"));

    terminal.send_input("\x1b[<32;4;2M");
    terminal.send_input("\x1b[<0;4;2m");
    wait_for_render(&tui);
    assert!(!clipboard_writes_contain(&terminal, "\x1b]52;c;"));
    tui.stop(TuiStopOptions::default());
}

#[test]
fn retains_a_completed_visible_selection_across_focus_changes() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 4);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&["alpha", "beta", "gamma", "delta"]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<32;4;2M");
    terminal.send_input("\x1b[<0;4;2m");
    wait_for_render(&tui);
    let completed_write_count = terminal
        .events
        .borrow()
        .iter()
        .filter(|event| matches!(event, RecordingEvent::Write(_)))
        .count();
    terminal.send_input("\x1b[O");
    terminal.send_input("\x1b[I");
    wait_for_render(&tui);
    assert_eq!(
        terminal
            .events
            .borrow()
            .iter()
            .filter(|event| matches!(event, RecordingEvent::Write(_)))
            .count(),
        completed_write_count,
    );

    let redraw_event_count = terminal.event_count();
    tui.render_now(true);
    let redraw_writes = terminal.writes_from(redraw_event_count);
    assert!(redraw_writes.contains("alpha"));
    assert!(redraw_writes.contains("beta"));
    assert!(redraw_writes.contains("\x1b[7m"));
    tui.stop(TuiStopOptions::default());
}

#[test]
fn stacks_flash_messages_and_collapses_them_as_they_expire() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 4);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&["one", "two", "three", "four"]));
    tui.start();
    wait_for_render(&tui);

    alt.flash("First", Some(80));
    alt.flash("Second", Some(500));
    render_and_flush(&tui);
    let viewport = terminal.get_viewport();
    assert!(viewport[0].ends_with(" First "));
    assert!(viewport[1].ends_with(" Second "));

    flush_until(&tui, || {
        !terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("First"))
    });
    let viewport = terminal.get_viewport();
    assert!(viewport[0].ends_with(" Second "));
    assert!(!viewport.iter().any(|line| line.contains("First")));

    tui.stop(TuiStopOptions::default());
}

#[test]
fn auto_scrolls_and_extends_a_drag_selection_held_at_the_viewport_edge() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 4);
    let (tui, alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&[&numbered_lines(10)]));
    tui.start();
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 6);

    terminal.send_input("\x1b[<0;1;3M");
    terminal.send_input("\x1b[<32;1;1M");
    sleep_ms(130);
    flush_until(&tui, || alt.viewport_top() < 6);

    let selection_top = alt.viewport_top();
    assert!(
        selection_top < 6,
        "expected auto-scroll above row 6, got {selection_top}"
    );
    terminal.send_input("\x1b[<0;1;1m");
    wait_for_render(&tui);

    let mut selected_lines: Vec<String> = (selection_top..8)
        .map(|index| format!("line {}", index + 1))
        .collect();
    selected_lines.push("l".to_string());
    let expected_clipboard_sequence = osc52(&selected_lines.join("\n"));
    assert!(
        clipboard_writes_contain(&terminal, &expected_clipboard_sequence),
        "{}",
        clipboard_writes_containing(&terminal, "\x1b]52;c;"),
    );
    tui.stop(TuiStopOptions::default());
}

#[test]
fn snaps_mouse_selection_to_cjk_emoji_and_combining_grapheme_boundaries() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 2);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(Rc::new(Text::with_padding(
        "A\u{754c}\u{1f642}e\u{301}Z",
        0,
        0,
    )));
    tui.start();
    wait_for_render(&tui);

    let wide_selection = osc52("\u{754c}\u{1f642}");
    terminal.send_input("\x1b[<0;3;1M");
    terminal.send_input("\x1b[<32;4;1M");
    terminal.send_input("\x1b[<0;4;1m");
    wait_for_render(&tui);
    assert_eq!(
        terminal
            .events
            .borrow()
            .iter()
            .filter(|event| matches!(event, RecordingEvent::Write(data) if data.contains(&wide_selection)))
            .count(),
        1,
    );

    terminal.send_input("\x1b[<0;5;1M");
    terminal.send_input("\x1b[<32;2;1M");
    terminal.send_input("\x1b[<0;2;1m");
    wait_for_render(&tui);
    assert_eq!(
        terminal
            .events
            .borrow()
            .iter()
            .filter(|event| matches!(event, RecordingEvent::Write(data) if data.contains(&wide_selection)))
            .count(),
        2,
    );

    let combining_selection = osc52("e\u{301}Z");
    terminal.send_input("\x1b[<0;6;1M");
    terminal.send_input("\x1b[<32;7;1M");
    terminal.send_input("\x1b[<0;7;1m");
    wait_for_render(&tui);
    assert!(clipboard_writes_contain(&terminal, &combining_selection));

    tui.stop(TuiStopOptions::default());
}

#[test]
fn ignores_horizontal_trackpad_wheel_events() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 4);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&[&numbered_lines(8)]));
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<66;1;1M");
    terminal.send_input("\x1b[<67;1;1M");
    wait_for_render(&tui);
    assert_eq!(alt.viewport_top(), 4);
    assert_eq!(
        trimmed_viewport(&terminal),
        vec!["line 5", "line 6", "line 7", "line 8"],
    );

    tui.stop(TuiStopOptions::default());
}

#[test]
fn dispatches_clicks_to_nested_mouse_regions_without_breaking_drag_selection() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 2);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    let clicks = Rc::new(Cell::new(0));
    let clicks_probe = Rc::clone(&clicks);
    let region = Rc::new(MouseRegion::new(
        text(&["clickable", "selectable"]),
        move |event: &TuiMouseEvent| {
            if event.event_type != TuiMouseEventType::Click {
                return None;
            }
            clicks_probe.set(clicks_probe.get() + 1);
            Some(TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            })
        },
    ));
    tui.add_child(region);
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;2;1M");
    terminal.send_input("\x1b[<0;2;1m");
    wait_for_render(&tui);
    assert_eq!(clicks.get(), 1);

    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<32;4;2M");
    terminal.send_input("\x1b[<0;4;2m");
    wait_for_render(&tui);
    assert_eq!(clicks.get(), 1);
    assert!(clipboard_writes_contain(&terminal, "\x1b]52;c;"));
    tui.stop(TuiStopOptions::default());
}

#[test]
fn focuses_and_captures_drag_gestures_for_mouse_aware_components() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 2);
    let (tui, _alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    let events: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let events_probe = Rc::clone(&events);
    let component = Rc::new(MouseAwareControl {
        lines: RefCell::new(vec!["control".to_string()]),
        events: Rc::clone(&events_probe),
    });
    tui.add_child(component.clone());
    tui.start();
    wait_for_render(&tui);

    terminal.send_input("\x1b[<0;1;1M");
    terminal.send_input("\x1b[<32;5;2M");
    terminal.send_input("\x1b[<0;5;2m");
    wait_for_render(&tui);

    assert_eq!(*events.borrow(), vec!["press", "drag", "release"]);
    let focused = tui.get_focused_component().expect("a focused component");
    let component: Rc<dyn Component> = component;
    assert!(Rc::ptr_eq(&focused, &component));
    tui.stop(TuiStopOptions::default());
}

/// A mouse-aware control the gesture suites use, upstream's object literal
/// with a `handleMouse`.
struct MouseAwareControl {
    lines: RefCell<Vec<String>>,
    events: Rc<RefCell<Vec<String>>>,
}

impl MouseAwareControl {
    #[expect(
        clippy::unnecessary_wraps,
        reason = "mirrors upstream's handleMouse, whose every branch answers a result object"
    )]
    fn record(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.events.borrow_mut().push(
            match event.event_type {
                TuiMouseEventType::Press => "press",
                TuiMouseEventType::Release => "release",
                TuiMouseEventType::Move => "move",
                TuiMouseEventType::Drag => "drag",
                TuiMouseEventType::Click => "click",
                TuiMouseEventType::Wheel => "wheel",
            }
            .to_string(),
        );
        match event.event_type {
            TuiMouseEventType::Press => Some(TuiMouseEventResult {
                handled: true,
                capture: true,
                focus: true,
                ..TuiMouseEventResult::default()
            }),
            _ => Some(TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            }),
        }
    }
}

impl Component for MouseAwareControl {
    fn render(&self, _width: usize) -> Vec<String> {
        self.lines.borrow().clone()
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.record(event)
    }

    fn invalidate(&self) {}
}

#[test]
fn reports_consecutive_click_counts_to_component_owned_controls() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 1);
    let (tui, _alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    let click_counts: Rc<RefCell<Vec<u32>>> = Rc::new(RefCell::new(Vec::new()));
    let counts_probe = Rc::clone(&click_counts);
    let component = Rc::new(ClickCountingControl {
        counts: counts_probe,
    });
    tui.add_child(component);
    tui.start();
    wait_for_render(&tui);

    for _ in 0..3 {
        terminal.send_input("\x1b[<0;1;1M");
        terminal.send_input("\x1b[<0;1;1m");
    }
    wait_for_render(&tui);
    assert_eq!(*click_counts.borrow(), vec![1, 2, 3]);
    tui.stop(TuiStopOptions::default());
}

/// A control that reports the click counts it receives, upstream's
/// `handleMouse` click-count object literal.
struct ClickCountingControl {
    counts: Rc<RefCell<Vec<u32>>>,
}

impl Component for ClickCountingControl {
    fn render(&self, _width: usize) -> Vec<String> {
        vec!["control".to_string()]
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        if event.event_type == TuiMouseEventType::Press {
            return Some(TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            });
        }
        if event.event_type == TuiMouseEventType::Click {
            self.counts
                .borrow_mut()
                .push(event.click_count.unwrap_or(0));
            return Some(TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            });
        }
        None
    }

    fn invalidate(&self) {}
}

#[test]
fn does_not_rerender_for_handled_no_op_pointer_motion() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 2);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    let render_count = Rc::new(Cell::new(0));
    let render_probe = Rc::clone(&render_count);
    let component = Rc::new(MoveTargetingControl {
        lines: RefCell::new(vec!["hover target".to_string()]),
        render_count: Rc::clone(&render_probe),
    });
    tui.add_child(component);
    tui.start();
    wait_for_render(&tui);
    let rendered_before_motion = render_count.get();
    let writes_before_motion = terminal
        .events
        .borrow()
        .iter()
        .filter(|event| matches!(event, RecordingEvent::Write(_)))
        .count();

    terminal.send_input("\x1b[<35;1;1M");
    wait_for_render(&tui);
    assert_eq!(render_count.get(), rendered_before_motion);
    assert_eq!(
        terminal
            .events
            .borrow()
            .iter()
            .filter(|event| matches!(event, RecordingEvent::Write(_)))
            .count(),
        writes_before_motion,
    );
    tui.stop(TuiStopOptions::default());
}

/// A control that counts its renders and consumes handled moves, upstream's
/// render-counting object literal.
struct MoveTargetingControl {
    lines: RefCell<Vec<String>>,
    render_count: Rc<Cell<usize>>,
}

impl Component for MoveTargetingControl {
    fn render(&self, _width: usize) -> Vec<String> {
        self.render_count.set(self.render_count.get() + 1);
        self.lines.borrow().clone()
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        if event.event_type == TuiMouseEventType::Move {
            return Some(TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            });
        }
        None
    }

    fn invalidate(&self) {}
}

#[test]
fn lets_mouse_aware_components_consume_wheel_events_before_viewport_scrolling() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 3);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    let wheel_events = Rc::new(Cell::new(0));
    let wheel_probe = Rc::clone(&wheel_events);
    let region = Rc::new(MouseRegion::new(
        text(&[&numbered_lines(8)]),
        move |event: &TuiMouseEvent| {
            if event.event_type != TuiMouseEventType::Wheel {
                return None;
            }
            wheel_probe.set(wheel_probe.get() + 1);
            Some(TuiMouseEventResult {
                handled: true,
                ..TuiMouseEventResult::default()
            })
        },
    ));
    tui.add_child(region);
    tui.start();
    wait_for_render(&tui);
    let viewport_top = alt.viewport_top();

    terminal.send_input("\x1b[<64;1;1M");
    wait_for_render(&tui);
    assert_eq!(wheel_events.get(), 1);
    assert_eq!(alt.viewport_top(), viewport_top);
    tui.stop(TuiStopOptions::default());
}

#[test]
fn restores_keyboard_state_before_leaving_alt_mode_and_prints_the_full_document() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = RecordingTerminal::new(20, 3);
    let (tui, _alt) = new_recording_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&[
        "first", "second", "third", "fourth", "fifth", "sixth",
    ]));
    tui.start();
    wait_for_render(&tui);
    tui.stop(TuiStopOptions::default());

    let events = terminal.events.borrow().clone();
    let start_index = events
        .iter()
        .position(|event| event == &RecordingEvent::Start)
        .expect("a start event");
    let alt_screen_enter_index = events
        .iter()
        .position(
            |event| matches!(event, RecordingEvent::Write(data) if data.contains("\x1b[?1049h")),
        )
        .expect("an alt-screen enter write");
    let stop_index = events
        .iter()
        .position(|event| event == &RecordingEvent::Stop)
        .expect("a stop event");
    let mouse_disable_index = events
        .iter()
        .position(
            |event| matches!(event, RecordingEvent::Write(data) if data.contains("\x1b[?1006l")),
        )
        .expect("a mouse disable write");
    let main_screen_restore_index = events
        .iter()
        .position(
            |event| matches!(event, RecordingEvent::Write(data) if data.contains("\x1b[?1049l")),
        )
        .expect("a main-screen restore write");
    assert!(alt_screen_enter_index < start_index);
    assert!(mouse_disable_index < stop_index);
    assert!(main_screen_restore_index > stop_index);

    let Some(RecordingEvent::Write(restore_event)) = events.get(main_screen_restore_index) else {
        panic!("the restore index holds a write");
    };
    for line in ["first", "second", "third", "fourth", "fifth", "sixth"] {
        assert!(restore_event.contains(line));
    }
    assert!(
        restore_event.find("first").expect("a first line")
            < restore_event.find("sixth").expect("a sixth line")
    );
}

#[test]
fn gives_wheel_and_viewport_keys_to_a_focused_overlay() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 6);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&[&numbered_lines(12)]));
    let overlay = InputOverlay::new();
    tui.start();
    wait_for_render(&tui);
    let top_before = alt.viewport_top();
    let handle = tui.show_overlay(overlay.clone(), None);
    wait_for_render(&tui);
    assert!(overlay.focused.get());

    let wheel = "\x1b[<64;10;3M";
    let keys = ["\x1b[5~", "\x1b[6~", "\x1bOH", "\x1bOF", wheel];
    for key in keys {
        terminal.send_input(key);
    }
    wait_for_render(&tui);

    assert_eq!(
        *overlay.inputs.borrow(),
        keys.iter().map(ToString::to_string).collect::<Vec<_>>(),
    );
    assert_eq!(alt.viewport_top(), top_before);

    handle.hide();
    wait_for_render(&tui);
    terminal.send_input("\x1b[5~");
    wait_for_render(&tui);
    assert!(alt.viewport_top() < top_before);
    tui.stop(TuiStopOptions::default());
}

#[test]
fn keeps_viewport_scrolling_when_an_overlay_is_not_focused() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 6);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    let editor = InputOverlay::new();
    tui.add_child(text(&[&numbered_lines(12)]));
    tui.set_focus(Some(editor));
    tui.start();
    wait_for_render(&tui);
    let top_before = alt.viewport_top();

    let hidden = tui.show_overlay(InputOverlay::new(), None);
    hidden.set_hidden(true);
    let non_capturing_overlay = InputOverlay::new();
    let _ = tui.show_overlay(
        non_capturing_overlay.clone(),
        Some(pi_tui::tui::OverlayOptions {
            non_capturing: true,
            ..pi_tui::tui::OverlayOptions::default()
        }),
    );
    let unfocused_overlay = InputOverlay::new();
    let unfocused_handle = tui.show_overlay(unfocused_overlay.clone(), None);
    unfocused_handle.unfocus(None);
    wait_for_render(&tui);
    assert!(!non_capturing_overlay.focused.get());
    assert!(!unfocused_overlay.focused.get());

    terminal.send_input("\x1b[5~");
    terminal.send_input("\x1b[<64;10;3M");
    wait_for_render(&tui);
    assert!(alt.viewport_top() < top_before);
    assert!(non_capturing_overlay.inputs.borrow().is_empty());
    assert!(unfocused_overlay.inputs.borrow().is_empty());
    tui.stop(TuiStopOptions::default());
}

#[test]
fn keeps_viewport_scrolling_while_transcript_search_is_focused() {
    let _guard = CAPABILITIES_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let terminal = VirtualTerminal::new(20, 6);
    let (tui, alt) = new_tui(terminal.clone(), TuiAltScreenConfig::default());
    tui.add_child(text(&[&numbered_lines(12)]));
    tui.start();
    wait_for_render(&tui);
    let top_before = alt.viewport_top();

    terminal.send_input("\x1b[102;6u");
    wait_for_render(&tui);
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("↑ ↓"))
    );

    terminal.send_input("\x1b[5~");
    terminal.send_input("\x1b[<64;1;4M");
    wait_for_render(&tui);
    assert!(alt.viewport_top() < top_before);
    assert!(
        terminal
            .get_viewport()
            .iter()
            .any(|line| line.contains("↑ ↓"))
    );
    tui.stop(TuiStopOptions::default());
}

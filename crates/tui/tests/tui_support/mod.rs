//! Shared test harness for the TUI suites, the port of upstream's
//! `test/virtual-terminal.ts` ([#45](https://github.com/PhillipChaffee/pi-rust/issues/45),
//! upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`).
//!
//! - [`VirtualTerminal`] — the `@xterm/headless` stand-in: a `vte`-driven
//!   headless terminal emulator with upstream's `sendInput`/`resize`/
//!   `getViewport`/`getScrollBuffer`/`waitForRender` surface, plus per-cell
//!   italic reads for the style-leak suite and the raw write log.
//! - [`new_test_tui`] / [`new_test_tui_with_images`] — the `Tui` construction
//!   upstream's suites perform, with [`pi_tui::tui_main_screen::TuiMainScreen`]
//!   as the renderer and a quiet environment lookup so the suites cannot trip
//!   the Termux or debug-log switches.
//!
//! The emulator's restatements, against `@xterm/headless`:
//!
//! - Row resize follows xterm's `Buffer.resize`: shrinking pops viewport rows
//!   below the cursor or trims the top into scrollback, growing pulls
//!   scrollback back into the viewport or pads blank rows, and the cursor
//!   rides the pull. Column resize does not reflow wrapped lines — no suite
//!   in the ported slices depends on reflow, and the harness grows that
//!   behavior when one does.
//! - The scrollback buffer is unbounded; xterm caps it at 1000 lines, which
//!   no ported suite reaches.
//! - Print past the last column drops rather than wraps; the renderer throws
//!   on over-wide lines before they reach the terminal, so only the exact-
//!   width full-render writes reach the emulator, and those fit.
//! - Upstream's `waitForRender` (nextTick + 20 ms + flush) collapses into
//!   [`wait_for_render`], two pump turns around the 20 ms sleep — the port's
//!   event-loop turns — because writes land synchronously in the emulator
//!   instead of through xterm's async writer.

#![allow(
    unreachable_pub,
    reason = "the support module compiles into every suite binary through #[path]; its items are crate-local by construction"
)]
#![allow(
    dead_code,
    reason = "each suite uses only the fixtures it needs; the shared module carries the rest for its siblings, and the unused set differs per binary"
)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::Duration;

use unicode_width::UnicodeWidthChar;

use pi_tui::terminal::{InputHandler, ResizeHandler, Terminal};
use pi_tui::tui::{
    Component, Focusable, Tui, TuiConfig, TuiMouseButton, TuiMouseEvent, TuiMouseEventType,
    TuiRenderer,
};

/// One screen cell: the grapheme it starts, its visible width (zero for
/// wide-char placeholders), whether italic is active, and whether anything
/// was ever written into it — upstream xterm's `translateToString(true)`
/// trims only unwritten trailing cells, so a printed space survives.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ScreenCell {
    ch: char,
    width: u8,
    italic: bool,
    written: bool,
}

impl ScreenCell {
    const BLANK: Self = Self {
        ch: ' ',
        width: 1,
        italic: false,
        written: false,
    };
}

/// A row of the emulated buffer.
type ScreenRow = Vec<ScreenCell>;

fn blank_row(columns: usize) -> ScreenRow {
    vec![ScreenCell::BLANK; columns]
}

/// The visible screen upstream's xterm buffer exposed, driven by `vte`: the
/// grid is the viewport, `scrollback` holds the lines scrolled above it, and
/// row resizes follow xterm's pull-from-scrollback or pad-blank semantics.
struct Screen {
    grid: Vec<ScreenRow>,
    scrollback: Vec<ScreenRow>,
    cursor_row: usize,
    cursor_col: usize,
    italic: bool,
}

impl Screen {
    fn new(columns: usize, rows: usize) -> Self {
        Self {
            grid: vec![blank_row(columns); rows],
            scrollback: Vec::new(),
            cursor_row: 0,
            cursor_col: 0,
            italic: false,
        }
    }

    fn resize(&mut self, columns: usize, rows: usize) {
        for row in &mut self.grid {
            row.resize(columns, ScreenCell::BLANK);
        }
        let old_rows = self.grid.len();
        if rows < old_rows {
            // Shrink: xterm pops viewport rows below the cursor and trims the
            // top into scrollback otherwise, so the last content stays put.
            for _ in rows..old_rows {
                if self.grid.len() > self.cursor_row + 1 {
                    self.grid.pop();
                } else {
                    let top = self.grid.remove(0);
                    self.scrollback.push(top);
                }
            }
            self.cursor_row = self.cursor_row.min(rows.saturating_sub(1));
        } else if rows > old_rows {
            // Grow: pull scrollback back into the viewport when the buffer is
            // too short to fill it, pad blank rows otherwise; the cursor
            // rides the pull.
            let mut pulled = 0;
            for _ in old_rows..rows {
                if !self.scrollback.is_empty() && self.grid.len() <= self.cursor_row + pulled + 1 {
                    self.grid
                        .insert(0, self.scrollback.pop().unwrap_or_default());
                    pulled += 1;
                } else {
                    self.grid.push(blank_row(columns));
                }
            }
            self.cursor_row = self.cursor_row.min(rows - 1) + pulled;
        }
        self.cursor_row = self.cursor_row.min(rows.saturating_sub(1));
        self.cursor_col = self.cursor_col.min(columns.saturating_sub(1));
    }

    fn scroll_up(&mut self) {
        let columns = self.grid.first().map_or(1, Vec::len);
        self.scrollback.push(self.grid.remove(0));
        self.grid.push(blank_row(columns));
    }

    fn put(&mut self, c: char) {
        let Some(width) = c.width().filter(|width| *width > 0) else {
            return;
        };
        let columns = self.grid.first().map_or(1, Vec::len);
        if self.cursor_col + width > columns {
            return;
        }
        let cell = ScreenCell {
            ch: c,
            width: u8::try_from(width).unwrap_or(u8::MAX),
            italic: self.italic,
            written: true,
        };
        let (row, col) = (self.cursor_row, self.cursor_col);
        self.grid[row][col] = cell;
        for offset in 1..width {
            self.grid[row][col + offset] = ScreenCell {
                ch: '\u{0}',
                width: 0,
                italic: cell.italic,
                written: true,
            };
        }
        self.cursor_col += width;
    }

    fn clear_row(&mut self, row: usize, from: usize) {
        let columns = self.grid[row].len();
        for cell in &mut self.grid[row][from.min(columns)..columns] {
            *cell = ScreenCell::BLANK;
        }
    }

    /// The visible rows, trailing whitespace trimmed the way xterm's
    /// `translateToString(true)` is.
    fn viewport(&self) -> Vec<String> {
        self.grid.iter().map(|row| render_row(row)).collect()
    }

    /// Every buffer line including the scrollback, upstream
    /// `VirtualTerminal.getScrollBuffer` over xterm's `buffer.active`.
    fn scroll_buffer(&self) -> Vec<String> {
        self.scrollback
            .iter()
            .chain(self.grid.iter())
            .map(|row| render_row(row))
            .collect()
    }

    fn italic_at(&self, row: usize, col: usize) -> bool {
        self.grid
            .get(row)
            .and_then(|line| line.get(col))
            .is_some_and(|cell| cell.italic)
    }
}

fn render_row(row: &[ScreenCell]) -> String {
    let last_written = row
        .iter()
        .rposition(|cell| cell.written)
        .map_or(0, |position| position + 1);
    row[..last_written]
        .iter()
        .filter(|cell| cell.width > 0)
        .map(|cell| cell.ch)
        .collect()
}

/// The cell-size suite's SGR tracking with the extended-color groups
/// skipped, so `38;2;r;g;b` never trips the italic flag.
fn apply_sgr(screen: &mut Screen, params: &vte::Params) {
    let groups: Vec<Vec<u16>> = params.iter().map(<[u16]>::to_vec).collect();
    let mut index = 0;
    while index < groups.len() {
        let group = &groups[index];
        let param = group.first().copied().unwrap_or(0);
        match param {
            38 | 48 | 58 | 59 => {
                if group.len() > 1 {
                    // Colon form: the whole spec lives in this group.
                    index += 1;
                } else {
                    match groups.get(index + 1).and_then(|next| next.first()).copied() {
                        Some(5) => index += 3,
                        Some(2) => index += 5,
                        _ => index += 1,
                    }
                }
            }
            3 => screen.italic = true,
            23 | 0 => screen.italic = false,
            _ => {}
        }
        index += 1;
    }
}

impl vte::Perform for Screen {
    fn print(&mut self, c: char) {
        self.put(c);
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            b'\r' => self.cursor_col = 0,
            b'\n' => {
                let rows = self.grid.len().max(1);
                if self.cursor_row + 1 >= rows {
                    self.scroll_up();
                    self.cursor_row = rows.saturating_sub(1);
                } else {
                    self.cursor_row += 1;
                }
            }
            _ => {}
        }
    }

    fn csi_dispatch(
        &mut self,
        params: &vte::Params,
        intermediates: &[u8],
        _ignore: bool,
        action: char,
    ) {
        if intermediates.contains(&b'?') {
            // Private modes (cursor visibility, synchronized output, paste
            // mode) carry no content.
            return;
        }
        let param = |index: usize, default: u16| -> u16 {
            params
                .iter()
                .nth(index)
                .and_then(|sub| sub.first().copied())
                .filter(|value| *value > 0)
                .unwrap_or(default)
        };
        let columns = self.grid.first().map_or(1, Vec::len);
        let rows = self.grid.len().max(1);
        match action {
            'A' => {
                self.cursor_row = self.cursor_row.saturating_sub(param(0, 1) as usize);
            }
            'B' => {
                self.cursor_row =
                    (self.cursor_row + param(0, 1) as usize).min(rows.saturating_sub(1));
            }
            'C' => {
                self.cursor_col =
                    (self.cursor_col + param(0, 1) as usize).min(columns.saturating_sub(1));
            }
            'D' => {
                self.cursor_col = self.cursor_col.saturating_sub(param(0, 1) as usize);
            }
            'G' => {
                self.cursor_col =
                    (param(0, 1).saturating_sub(1) as usize).min(columns.saturating_sub(1));
            }
            'H' => {
                self.cursor_row =
                    (param(0, 1).saturating_sub(1) as usize).min(rows.saturating_sub(1));
                self.cursor_col =
                    (param(1, 1).saturating_sub(1) as usize).min(columns.saturating_sub(1));
            }
            'J' => match param(0, 0) {
                0 => {
                    for row in self.cursor_row..self.grid.len() {
                        self.clear_row(row, 0);
                    }
                }
                2 => {
                    for row in 0..self.grid.len() {
                        self.clear_row(row, 0);
                    }
                }
                // Clear scrollback, xterm's `eraseInDisplay` case 3: the
                // viewport lines stay, everything above them is dropped.
                3 => self.scrollback.clear(),
                _ => {}
            },
            'K' => {
                let row = self.cursor_row.min(self.grid.len().saturating_sub(1));
                match param(0, 0) {
                    0 => self.clear_row(row, self.cursor_col),
                    2 => self.clear_row(row, 0),
                    _ => {}
                }
            }
            'm' => apply_sgr(self, params),
            _ => {}
        }
    }
}

/// The test terminal, upstream's `VirtualTerminal`: a fixed-size emulator
/// that captures the TUI's writes and dispatches injected input. Tests keep
/// a clone for the test surface; the TUI takes the terminal itself by value.
#[derive(Clone)]
pub struct VirtualTerminal {
    shared: Rc<RefCell<VirtualShared>>,
}

struct VirtualShared {
    screen: Screen,
    columns: u16,
    rows: u16,
    input_handler: Option<InputHandler>,
    resize_handler: Option<ResizeHandler>,
    writes: String,
    queued_input: VecDeque<String>,
}

impl VirtualTerminal {
    /// A fresh `VirtualTerminal(columns, rows)`.
    #[must_use]
    pub(crate) fn new(columns: u16, rows: u16) -> Self {
        Self {
            shared: Rc::new(RefCell::new(VirtualShared {
                screen: Screen::new(usize::from(columns), usize::from(rows)),
                columns,
                rows,
                input_handler: None,
                resize_handler: None,
                writes: String::new(),
                queued_input: VecDeque::new(),
            })),
        }
    }

    /// Queue a chunk for the next pump turn, restating the real terminal's
    /// stdin feed: [`Terminal::poll`] delivers it through the stored handler
    /// while the TUI's pump window holds the terminal borrow, which is the
    /// queue-and-drain path the input pipeline restates.
    pub(crate) fn queue_input(&self, data: &str) {
        self.shared
            .borrow_mut()
            .queued_input
            .push_back(data.to_string());
    }

    /// The raw bytes every [`Terminal::write`] delivered, for assertions the
    /// emulator cannot make (cursor visibility, synchronized output, image
    /// deletions), upstream `LoggingVirtualTerminal.getWrites`.
    #[must_use]
    pub(crate) fn write_log(&self) -> String {
        self.shared.borrow().writes.clone()
    }

    /// Drop the write log, upstream `LoggingVirtualTerminal.clearWrites`.
    pub(crate) fn clear_writes(&self) {
        self.shared.borrow_mut().writes.clear();
    }

    /// Simulate keyboard input, upstream `VirtualTerminal.sendInput`: the
    /// stored handler runs synchronously, exactly as the xterm harness
    /// invoked it.
    pub(crate) fn send_input(&self, data: &str) {
        let mut handler = self.shared.borrow_mut().input_handler.take();
        if let Some(handler) = handler.as_mut() {
            handler(data.to_string());
        }
        self.shared.borrow_mut().input_handler = handler;
    }

    /// Resize the terminal, upstream `VirtualTerminal.resize`.
    pub(crate) fn resize(&self, columns: u16, rows: u16) {
        {
            let mut shared = self.shared.borrow_mut();
            shared.columns = columns;
            shared.rows = rows;
            shared
                .screen
                .resize(usize::from(columns), usize::from(rows));
        }
        let mut handler = self.shared.borrow_mut().resize_handler.take();
        if let Some(handler) = handler.as_mut() {
            handler();
        }
        self.shared.borrow_mut().resize_handler = handler;
    }

    /// The visible viewport, upstream `VirtualTerminal.getViewport`.
    #[must_use]
    pub(crate) fn get_viewport(&self) -> Vec<String> {
        self.shared.borrow().screen.viewport()
    }

    /// The entire scroll buffer including scrollback, upstream
    /// `VirtualTerminal.getScrollBuffer`.
    #[must_use]
    pub(crate) fn get_scroll_buffer(&self) -> Vec<String> {
        self.shared.borrow().screen.scroll_buffer()
    }

    /// The italic attribute of one cell, upstream's `getCellItalic` helper.
    #[must_use]
    pub(crate) fn is_italic(&self, row: usize, col: usize) -> bool {
        self.shared.borrow().screen.italic_at(row, col)
    }
}

impl Terminal for VirtualTerminal {
    fn start(&mut self, on_input: InputHandler, on_resize: ResizeHandler) {
        let mut shared = self.shared.borrow_mut();
        shared.input_handler = Some(on_input);
        shared.resize_handler = Some(on_resize);
        drop(shared);
        // Enable bracketed paste mode for consistency with ProcessTerminal.
        Terminal::write(self, "\x1b[?2004h");
    }

    fn stop(&mut self) {
        // Disable bracketed paste mode.
        Terminal::write(self, "\x1b[?2004l");
        let mut shared = self.shared.borrow_mut();
        shared.input_handler = None;
        shared.resize_handler = None;
        shared.queued_input.clear();
    }

    fn drain_input(&mut self, _max_ms: u64, _idle_ms: u64) {
        // No-op for the virtual terminal - no stdin to drain.
    }

    fn write(&mut self, data: &str) {
        let mut parser = vte::Parser::new();
        let mut shared = self.shared.borrow_mut();
        shared.writes.push_str(data);
        parser.advance(&mut shared.screen, data.as_bytes());
    }

    fn columns(&self) -> u16 {
        self.shared.borrow().columns
    }

    fn rows(&self) -> u16 {
        self.shared.borrow().rows
    }

    fn is_kitty_protocol_active(&self) -> bool {
        // The virtual terminal always reports Kitty protocol as active for
        // testing, upstream's answer.
        true
    }

    fn move_by(&mut self, lines: i32) {
        if lines > 0 {
            // Move down
            Terminal::write(self, &format!("\x1b[{lines}B"));
        } else if lines < 0 {
            // Move up
            Terminal::write(self, &format!("\x1b[{}A", -lines));
        }
        // lines === 0: no movement
    }

    fn hide_cursor(&mut self) {
        Terminal::write(self, "\x1b[?25l");
    }

    fn show_cursor(&mut self) {
        Terminal::write(self, "\x1b[?25h");
    }

    fn clear_line(&mut self) {
        Terminal::write(self, "\x1b[K");
    }

    fn clear_from_cursor(&mut self) {
        Terminal::write(self, "\x1b[J");
    }

    fn clear_screen(&mut self) {
        Terminal::write(self, "\x1b[2J\x1b[H"); // Clear screen and move to home (1,1)
    }

    fn set_title(&mut self, title: &str) {
        // OSC 0;title BEL - set terminal window title
        Terminal::write(self, &format!("\x1b]0;{title}\u{7}"));
    }

    fn set_progress(&mut self, _active: bool) {}

    fn poll(&mut self, _timeout: Duration) {
        // Deliver queued stdin chunks through the stored handler, upstream's
        // data event inside the pump window: the handler sees the terminal
        // already borrowed and queues for the pump's drain.
        loop {
            let data = self.shared.borrow_mut().queued_input.pop_front();
            let Some(data) = data else {
                break;
            };
            let mut handler = self.shared.borrow_mut().input_handler.take();
            if let Some(handler) = handler.as_mut() {
                handler(data);
            }
            self.shared.borrow_mut().input_handler = handler;
        }
    }
}

/// The renderer the harness hands every TUI: [`TuiMainScreen`], upstream's
/// `new TuiMainScreen(terminal)`. The environment lookup answers nothing, so
/// the suites cannot trip the Termux or debug-log switches a developer's
/// shell might export.
#[must_use]
pub fn test_renderer() -> Box<dyn TuiRenderer> {
    Box::new(pi_tui::tui_main_screen::TuiMainScreen::new(
        pi_tui::tui_main_screen::TuiMainScreenConfig {
            env_lookup: Some(Box::new(|_| None)),
        },
    ))
}

/// Build the TUI the suites drive, upstream `new TuiMainScreen(terminal)`.
#[must_use]
pub fn new_test_tui(terminal: VirtualTerminal) -> Rc<Tui> {
    Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(test_renderer()),
        ..TuiConfig::default()
    })
}

/// Build the TUI with the image-capable probe injected, restating the
/// cell-size suite's `withImageTerminal` env setup.
#[must_use]
pub fn new_test_tui_with_images(terminal: VirtualTerminal) -> Rc<Tui> {
    Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(test_renderer()),
        images_probe: Some(Box::new(|| true)),
        ..TuiConfig::default()
    })
}

/// Upstream's `renderAndFlush`: a forced render request plus one pump turn,
/// the port's stand-in for `requestRender(true)` + nextTick + flush.
pub fn render_and_flush(tui: &Tui) {
    tui.request_render(true);
    tui.poll(Duration::ZERO);
}

/// Upstream's `waitForRender`, for the suites that render through the
/// request `start()` made or an armed throttled deadline: the first pump
/// turn arms the schedule, the 20 ms sleep lets the deadline pass, and the
/// second turn is the flush.
pub fn wait_for_render(tui: &Tui) {
    tui.poll(Duration::ZERO);
    std::thread::sleep(Duration::from_millis(20));
    tui.poll(Duration::ZERO);
}

/// Upstream's `tui.stop()` with the default stop options.
pub fn stop(tui: &Tui) {
    tui.stop(pi_tui::tui::TuiStopOptions::default());
}

/// The overlay-options `StaticOverlay`: renders fixed lines and records the
/// width it was asked to render at.
pub struct StaticOverlay {
    lines: Vec<String>,
    requested_width: std::cell::Cell<Option<usize>>,
}

impl StaticOverlay {
    /// Upstream `new StaticOverlay(lines, requestedWidth?)`.
    #[must_use]
    pub(crate) fn new(lines: Vec<&str>) -> Rc<Self> {
        Rc::new(Self {
            lines: lines.into_iter().map(String::from).collect(),
            requested_width: std::cell::Cell::new(None),
        })
    }

    /// The width of the last render, upstream `requestedWidth`.
    #[must_use]
    pub(crate) const fn requested_width(&self) -> Option<usize> {
        self.requested_width.get()
    }
}

impl Component for StaticOverlay {
    fn render(&self, width: usize) -> Vec<String> {
        // Store the width we were asked to render at for verification.
        self.requested_width.set(Some(width));
        self.lines.clone()
    }

    fn invalidate(&self) {}
}

/// The overlay suites' `EmptyContent`.
#[derive(Debug, Default)]
pub struct EmptyContent;

impl Component for EmptyContent {
    fn render(&self, _width: usize) -> Vec<String> {
        Vec::new()
    }

    fn invalidate(&self) {}
}

/// The overlay-non-capturing `FocusableOverlay`: a focus-aware input
/// recorder that renders fixed lines.
/// The installed `handleInput` body, upstream's inline function property.
type InputHook = Box<dyn Fn(&str)>;

pub struct FocusableOverlay {
    focused: std::cell::Cell<bool>,
    inputs: RefCell<Vec<String>>,
    lines: Vec<String>,
    on_input: RefCell<InputHook>,
}

impl FocusableOverlay {
    /// Upstream `new FocusableOverlay(lines)`.
    #[must_use]
    pub(crate) fn new(lines: &[&str]) -> Rc<Self> {
        Rc::new(Self {
            focused: std::cell::Cell::new(false),
            inputs: RefCell::new(Vec::new()),
            lines: lines.iter().map(|line| (*line).to_string()).collect(),
            on_input: RefCell::new(Box::new(|_| {})),
        })
    }

    /// The recorded inputs, upstream `inputs`.
    #[must_use]
    pub(crate) fn inputs(&self) -> Vec<String> {
        self.inputs.borrow().clone()
    }

    /// Install the `handleInput` body, upstream's
    /// `component.handleInput = (data) => { ... }` rewires.
    pub(crate) fn set_on_input(&self, on_input: impl Fn(&str) + 'static) {
        *self.on_input.borrow_mut() = Box::new(on_input);
    }
}

impl Focusable for FocusableOverlay {
    fn set_focused(&self, focused: bool) {
        self.focused.set(focused);
    }

    fn is_focused(&self) -> bool {
        self.focused.get()
    }
}

impl Component for FocusableOverlay {
    fn render(&self, _width: usize) -> Vec<String> {
        self.lines.clone()
    }

    fn handle_input(&self, data: &str) {
        self.inputs.borrow_mut().push(data.to_string());
        (self.on_input.borrow())(data);
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {}

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }
}

/// The focus assertions upstream read off `component.focused` directly.
#[must_use]
pub fn is_focused<C: Component + ?Sized>(component: &Rc<C>) -> bool {
    component.as_focusable().is_some_and(Focusable::is_focused)
}

/// A mouse event with screen coordinates mirroring the local ones, upstream
/// `mouse(type, x, y, width, height)` in the mouse suites.
#[must_use]
pub const fn mouse_event(
    event_type: TuiMouseEventType,
    x: u16,
    y: u16,
    width: u16,
    height: u16,
) -> TuiMouseEvent {
    TuiMouseEvent {
        event_type,
        button: TuiMouseButton::Left,
        x,
        y,
        screen_x: x,
        screen_y: y,
        width,
        height,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: None,
    }
}

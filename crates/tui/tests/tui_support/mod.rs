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
#![expect(
    clippy::expect_used,
    reason = "the SGR-strip regex pattern is a compile-time constant; a bad pattern is a programmer error that must surface"
)]

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use unicode_width::UnicodeWidthChar;

use pi_tui::components::ColorFn;
use pi_tui::components::{Editor, EditorTheme, MarkdownTheme, SelectListTheme};
use pi_tui::terminal::{EnvLookup, InputHandler, ResizeHandler, Terminal};
use pi_tui::terminal_image::{EncodeKittyOptions, KittyImageMetadata, encode_kitty};
use pi_tui::tui::{
    Component, Focusable, Tui, TuiConfig, TuiMouseButton, TuiMouseEvent, TuiMouseEventType,
    TuiRenderer,
};
use pi_tui::tui_alt_screen::{TuiAltScreen, TuiAltScreenConfig};
use pi_tui::tui_main_screen::{TuiMainScreen, TuiMainScreenConfig};

/// One screen cell: the grapheme it starts, its visible width (zero for
/// wide-char placeholders), the SGR attributes it was printed under, and
/// whether anything was ever written into it — upstream xterm's
/// `translateToString(true)` trims only unwritten trailing cells, so a
/// printed space survives.
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per tracked SGR attribute (italic, bold, dim, underline) plus the written marker, mirroring xterm's cell model"
)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct ScreenCell {
    ch: char,
    width: u8,
    italic: bool,
    bold: bool,
    dim: bool,
    underline: bool,
    fg: Option<FgColor>,
    written: bool,
}

/// A foreground color, upstream xterm's cell color model: default (no SGR
/// 30-38 seen), an indexed 0-255 color, or a 24-bit RGB triple.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FgColor {
    Indexed(u16),
    Rgb(u8, u8, u8),
}

impl FgColor {
    /// The packed number xterm's `getFgColor` reports: the index itself, or
    /// an RGB triple packed `(r << 16) | (g << 8) | b`.
    const fn packed(self) -> u32 {
        match self {
            Self::Indexed(index) => index as u32,
            Self::Rgb(r, g, b) => ((r as u32) << 16) | ((g as u32) << 8) | (b as u32),
        }
    }
}

impl ScreenCell {
    const BLANK: Self = Self {
        ch: ' ',
        width: 1,
        italic: false,
        bold: false,
        dim: false,
        underline: false,
        fg: None,
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
#[expect(
    clippy::struct_excessive_bools,
    reason = "the cursor-position SGR state is one flag per tracked attribute, exactly what xterm's buffer carries"
)]
struct Screen {
    grid: Vec<ScreenRow>,
    scrollback: Vec<ScreenRow>,
    cursor_row: usize,
    cursor_col: usize,
    italic: bool,
    bold: bool,
    dim: bool,
    underline: bool,
    fg: Option<FgColor>,
}

impl Screen {
    fn new(columns: usize, rows: usize) -> Self {
        Self {
            grid: vec![blank_row(columns); rows],
            scrollback: Vec::new(),
            cursor_row: 0,
            cursor_col: 0,
            italic: false,
            bold: false,
            dim: false,
            underline: false,
            fg: None,
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
            bold: self.bold,
            dim: self.dim,
            underline: self.underline,
            fg: self.fg,
            written: true,
        };
        let (row, col) = (self.cursor_row, self.cursor_col);
        self.grid[row][col] = cell;
        for offset in 1..width {
            self.grid[row][col + offset] = ScreenCell {
                ch: '\u{0}',
                width: 0,
                italic: cell.italic,
                bold: cell.bold,
                dim: cell.dim,
                underline: cell.underline,
                fg: cell.fg,
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

    /// The cell attribute reads the style suites assert on: upstream's
    /// `isBold`/`isDim`/`isUnderline`/`isFgDefault`/`getFgColor` per cell.
    fn attr_at(&self, row: usize, col: usize) -> Option<ScreenCell> {
        self.grid.get(row).and_then(|line| line.get(col)).copied()
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

/// The cell-style suites' SGR tracking: italic, bold, dim, underline, and
/// the foreground color model, with the extended-color group syntax
/// (`38;2;r;g;b`, `38;5;n`) consumed so it never trips the flags.
fn apply_sgr(screen: &mut Screen, params: &vte::Params) {
    let groups: Vec<Vec<u16>> = params.iter().map(<[u16]>::to_vec).collect();
    let mut index = 0;
    while index < groups.len() {
        let group = &groups[index];
        let param = group.first().copied().unwrap_or(0);
        match param {
            38 | 48 | 58 | 59 if group.len() > 1 => {
                // Colon form: the whole spec lives in this group, and no
                // style suite asserts on it — consume and move on.
                index += 1;
            }
            38 => {
                let spec = groups
                    .get(index + 1)
                    .and_then(|next| next.first())
                    .copied()
                    .unwrap_or(0);
                match spec {
                    5 => {
                        screen.fg = groups
                            .get(index + 2)
                            .and_then(|g| g.first())
                            .map(|value| FgColor::Indexed(*value));
                        index += 3;
                    }
                    2 => {
                        let channel = |offset: usize| -> u8 {
                            groups
                                .get(index + offset)
                                .and_then(|g| g.first())
                                .and_then(|value| u8::try_from(*value).ok())
                                .unwrap_or(0)
                        };
                        screen.fg = Some(FgColor::Rgb(channel(2), channel(3), channel(4)));
                        index += 5;
                    }
                    _ => index += 1,
                }
            }
            48 | 58 | 59 => {
                let spec = groups
                    .get(index + 1)
                    .and_then(|next| next.first())
                    .copied()
                    .unwrap_or(0);
                index += match spec {
                    5 => 3,
                    2 => 5,
                    _ => 1,
                };
            }
            0 => {
                screen.italic = false;
                screen.bold = false;
                screen.dim = false;
                screen.underline = false;
                screen.fg = None;
            }
            1 => screen.bold = true,
            2 => screen.dim = true,
            3 => screen.italic = true,
            4 => screen.underline = true,
            22 => {
                screen.bold = false;
                screen.dim = false;
            }
            23 => screen.italic = false,
            24 => screen.underline = false,
            30..=37 => screen.fg = Some(FgColor::Indexed(param - 30)),
            39 => screen.fg = None,
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

    /// The bold attribute of one cell, upstream `getCell(...).isBold()`.
    #[must_use]
    pub(crate) fn is_bold(&self, row: usize, col: usize) -> bool {
        self.shared
            .borrow()
            .screen
            .attr_at(row, col)
            .is_some_and(|cell| cell.bold)
    }

    /// The dim attribute of one cell, upstream `getCell(...).isDim()`.
    #[must_use]
    pub(crate) fn is_dim(&self, row: usize, col: usize) -> bool {
        self.shared
            .borrow()
            .screen
            .attr_at(row, col)
            .is_some_and(|cell| cell.dim)
    }

    /// The underline attribute of one cell, upstream `getCell(...).isUnderline()`.
    #[must_use]
    pub(crate) fn is_underline(&self, row: usize, col: usize) -> bool {
        self.shared
            .borrow()
            .screen
            .attr_at(row, col)
            .is_some_and(|cell| cell.underline)
    }

    /// Whether one cell keeps the default foreground, upstream
    /// `getCell(...).isFgDefault()`: no foreground SGR preceded the print.
    #[must_use]
    pub(crate) fn is_fg_default(&self, row: usize, col: usize) -> bool {
        self.shared
            .borrow()
            .screen
            .attr_at(row, col)
            .is_none_or(|cell| cell.fg.is_none())
    }

    /// The packed foreground color of one cell, upstream
    /// `getCell(...).getFgColor()`: `None` for the default foreground, the
    /// palette index for indexed colors, and `(r << 16) | (g << 8) | b` for
    /// RGB triples.
    #[must_use]
    pub(crate) fn fg_color(&self, row: usize, col: usize) -> Option<u32> {
        self.shared
            .borrow()
            .screen
            .attr_at(row, col)
            .and_then(|cell| cell.fg)
            .map(FgColor::packed)
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
    Box::new(TuiMainScreen::new(TuiMainScreenConfig {
        env_lookup: Some(Box::new(|_| None)),
    }))
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

/// The select/settings suites' move event: `mouse("move", x, y, 80, 10)`
/// with no button held.
#[must_use]
pub const fn mouse_move(x: u16, y: u16) -> TuiMouseEvent {
    TuiMouseEvent {
        event_type: TuiMouseEventType::Move,
        button: TuiMouseButton::None,
        x,
        y,
        screen_x: x,
        screen_y: y,
        width: 80,
        height: 10,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: None,
        click_count: None,
    }
}

/// The select/settings suites' wheel event: `mouse("wheel", x, y, 80, 10)`
/// carrying the wheel delta.
#[must_use]
pub const fn mouse_wheel(x: u16, y: u16, wheel_delta: i32) -> TuiMouseEvent {
    TuiMouseEvent {
        event_type: TuiMouseEventType::Wheel,
        button: TuiMouseButton::Left,
        x,
        y,
        screen_x: x,
        screen_y: y,
        width: 80,
        height: 10,
        shift: false,
        alt: false,
        ctrl: false,
        wheel_delta: Some(wheel_delta),
        click_count: None,
    }
}

/// The crop tests' 2×3-cell placement fixture: the `AAAA` Kitty line and
/// its 100×100-pixel metadata, the shape the layout/alt-screen/layout
/// crop sites share.
#[must_use]
pub fn kitty_fixture(image_id: u64) -> (String, KittyImageMetadata) {
    (
        encode_kitty(
            "AAAA",
            EncodeKittyOptions {
                columns: Some(2),
                rows: Some(3),
                image_id: Some(image_id),
                move_cursor: Some(false),
            },
        ),
        KittyImageMetadata {
            image_id,
            columns: 2,
            rows: 3,
            width_px: 100,
            height_px: 100,
        },
    )
}

/// The suites' environment closure, `map.get(key)` closed over a clone.
#[must_use]
pub fn env_lookup(map: &HashMap<String, String>) -> EnvLookup {
    let map = map.clone();
    Box::new(move |key| map.get(key).cloned())
}

/// The render suites' TUI construction: the main-screen renderer over a
/// quiet-map environment lookup.
#[must_use]
pub fn new_main_screen_tui(terminal: VirtualTerminal, env: &HashMap<String, String>) -> Rc<Tui> {
    Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(Box::new(TuiMainScreen::new(TuiMainScreenConfig {
            env_lookup: Some(env_lookup(env)),
        }))),
        ..TuiConfig::default()
    })
}

/// The alt-screen suites' TUI construction: the alternate-screen renderer
/// over the given configuration, with a handle kept for the scroll and
/// selection surfaces.
#[must_use]
pub fn new_alt_screen_tui(
    terminal: VirtualTerminal,
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

/// The image suites' Kitty placement builder: `encodeKitty` with the same
/// cell geometry the renderer reads, so the placements drive the identical
/// branches the `Image` component would.
#[must_use]
pub fn kitty_image(base64: &str, columns: usize, rows: usize, image_id: u64) -> String {
    encode_kitty(
        base64,
        EncodeKittyOptions {
            columns: Some(columns),
            rows: Some(rows),
            image_id: Some(image_id),
            move_cursor: Some(false),
        },
    )
}

/// The editor suites' TUI construction, upstream test `createTestTUI`: the
/// main-screen renderer over a virtual terminal with default geometry,
/// under a quiet-map environment lookup.
#[must_use]
pub fn new_editor_test_tui(columns: u16, rows: u16) -> Rc<Tui> {
    new_main_screen_tui(VirtualTerminal::new(columns, rows), &HashMap::new())
}

/// The editor suites' theme, upstream test-themes.ts `defaultEditorTheme`:
/// the border color is chalk's dim, and the autocomplete dropdown theme is
/// upstream test-themes.ts `defaultSelectListTheme`.
#[must_use]
pub fn default_editor_theme() -> EditorTheme {
    EditorTheme {
        border_color: Rc::new(|text| format!("\x1b[2m{text}\x1b[22m")),
        select_list: default_select_list_theme(),
    }
}

/// Let the editor's autocomplete worker settle and apply its result,
/// upstream `flushAutocomplete` (microtask + setImmediate): the port's
/// request runs on a worker thread, so the flush spins until the pipeline
/// is idle and then drains.
///
/// # Panics
/// Panics if the pipeline does not settle within five seconds — a hung
/// provider or a lost request.
pub fn flush_autocomplete(editor: &Editor) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !editor.autocomplete_idle() {
        assert!(
            std::time::Instant::now() <= deadline,
            "autocomplete pipeline did not settle"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    editor.drain_autocomplete();
}

/// The suites' select-list theme, upstream test-themes.ts
/// `defaultSelectListTheme` (chalk, level 3): the selected prefix blue,
/// the selected text bold, the description, scroll info, and no-match
/// lines dim.
#[must_use]
pub fn default_select_list_theme() -> SelectListTheme {
    let chalked = |style: &'static ChalkStyle| {
        let styled = chalk_one(style);
        Rc::new(move |text: &str| styled(text))
    };
    SelectListTheme {
        selected_prefix: chalked(&CHALK_BLUE),
        selected_text: chalked(&CHALK_BOLD),
        description: chalked(&CHALK_DIM),
        scroll_info: chalked(&CHALK_DIM),
        no_match: chalked(&CHALK_DIM),
    }
}

/// The suites' `BoundedWriteTerminal`: captures every write without
/// emulating anything, so the chunking asserts on raw writes.
#[derive(Clone, Default)]
pub struct BoundedWriteTerminal {
    writes: Rc<RefCell<Vec<String>>>,
}

impl BoundedWriteTerminal {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn writes(&self) -> Vec<String> {
        self.writes.borrow().clone()
    }

    pub fn clear_writes(&self) {
        self.writes.borrow_mut().clear();
    }
}

impl Terminal for BoundedWriteTerminal {
    fn start(&mut self, _on_input: InputHandler, _on_resize: ResizeHandler) {}
    fn stop(&mut self) {}
    fn drain_input(&mut self, _max_ms: u64, _idle_ms: u64) {}
    fn write(&mut self, data: &str) {
        self.writes.borrow_mut().push(data.to_string());
    }
    fn columns(&self) -> u16 {
        80
    }
    fn rows(&self) -> u16 {
        24
    }
    fn is_kitty_protocol_active(&self) -> bool {
        false
    }
    fn move_by(&mut self, _lines: i32) {}
    fn hide_cursor(&mut self) {}
    fn show_cursor(&mut self) {}
    fn clear_line(&mut self) {}
    fn clear_from_cursor(&mut self) {}
    fn clear_screen(&mut self) {}
    fn set_title(&mut self, _title: &str) {}
    fn set_progress(&mut self, _active: bool) {}
    fn poll(&mut self, _timeout: Duration) {}
}

// --- markdown suite fixtures (#48), upstream test-themes.ts -----------------

/// One chalk style: the byte-exact SGR open and close pair.
struct ChalkStyle {
    open: &'static str,
    close: &'static str,
}

const CHALK_BOLD: ChalkStyle = ChalkStyle {
    open: "\x1b[1m",
    close: "\x1b[22m",
};
const CHALK_DIM: ChalkStyle = ChalkStyle {
    open: "\x1b[2m",
    close: "\x1b[22m",
};
const CHALK_ITALIC: ChalkStyle = ChalkStyle {
    open: "\x1b[3m",
    close: "\x1b[23m",
};
const CHALK_UNDERLINE: ChalkStyle = ChalkStyle {
    open: "\x1b[4m",
    close: "\x1b[24m",
};
const CHALK_STRIKETHROUGH: ChalkStyle = ChalkStyle {
    open: "\x1b[9m",
    close: "\x1b[29m",
};
const CHALK_BLUE: ChalkStyle = ChalkStyle {
    open: "\x1b[34m",
    close: "\x1b[39m",
};
const CHALK_CYAN: ChalkStyle = ChalkStyle {
    open: "\x1b[36m",
    close: "\x1b[39m",
};
const CHALK_GREEN: ChalkStyle = ChalkStyle {
    open: "\x1b[32m",
    close: "\x1b[39m",
};
const CHALK_YELLOW: ChalkStyle = ChalkStyle {
    open: "\x1b[33m",
    close: "\x1b[39m",
};
const CHALK_GRAY: ChalkStyle = ChalkStyle {
    open: "\x1b[90m",
    close: "\x1b[39m",
};
const CHALK_MAGENTA: ChalkStyle = ChalkStyle {
    open: "\x1b[35m",
    close: "\x1b[39m",
};

/// A chalk-builder closure over one style chain, `styles` ordered innermost
/// first: chalk's level-3 wrapping is `openAll + text + closeAll`, with each
/// style's open code re-inserted after every occurrence of its close code
/// when the text carries ANSI (so nested resets re-open the outer style),
/// and each line break closed and re-opened so styles never bleed across
/// lines (chalk#92).
fn chalk(styles: &[ChalkStyle], text: &str) -> String {
    let open_all: String = styles.iter().rev().map(|style| style.open).collect();
    let close_all: String = styles.iter().map(|style| style.close).collect();
    let mut wrapped = text.to_string();
    if wrapped.contains('\u{1b}') {
        for style in styles {
            wrapped = insert_after_all(&wrapped, style.close, style.open);
        }
    }
    if wrapped.contains('\n') {
        wrapped = close_reopen_at_lfs(&wrapped, &close_all, &open_all);
    }
    format!("{open_all}{wrapped}{close_all}")
}

/// `stringReplaceAll(string, substring, substring + postfix)` — each match
/// survives and the postfix lands after it.
fn insert_after_all(text: &str, substring: &str, postfix: &str) -> String {
    let mut result = String::new();
    let mut cursor = 0;
    while let Some(index) = text[cursor..].find(substring) {
        let index = cursor + index;
        result.push_str(&text[cursor..index + substring.len()]);
        result.push_str(postfix);
        cursor = index + substring.len();
    }
    result.push_str(&text[cursor..]);
    result
}

/// Close the styling before every line break and reopen after it.
fn close_reopen_at_lfs(text: &str, close_all: &str, open_all: &str) -> String {
    let mut result = String::new();
    let mut cursor = 0;
    while let Some(index) = text[cursor..].find('\n') {
        let index = cursor + index;
        let cr = index > 0 && text.as_bytes()[index - 1] == b'\r';
        let line_end = if cr { index - 1 } else { index };
        result.push_str(&text[cursor..line_end]);
        result.push_str(close_all);
        result.push_str(if cr { "\r\n" } else { "\n" });
        result.push_str(open_all);
        cursor = index + 1;
    }
    result.push_str(&text[cursor..]);
    result
}

/// A single-style chalk closure, the fixture shape most theme entries use.
fn chalk_one(style: &'static ChalkStyle) -> ColorFn {
    Arc::new(move |text| chalk(std::slice::from_ref(style), text))
}

/// The markdown suites' theme, upstream test-themes.ts
/// `defaultMarkdownTheme` (chalk, level 3): heading is bold cyan, links are
/// blue, link URLs dim, code yellow, code blocks green, the code block
/// border and quote border dim, quotes italic, rules dim, list bullets
/// cyan, and the plain decorations map to chalk's own.
#[must_use]
pub fn default_markdown_theme() -> MarkdownTheme {
    MarkdownTheme {
        heading: Arc::new(|text| chalk(&[CHALK_CYAN, CHALK_BOLD], text)),
        link: Arc::new(|text| chalk(&[CHALK_BLUE], text)),
        link_url: Arc::new(|text| chalk(&[CHALK_DIM], text)),
        code: Arc::new(|text| chalk(&[CHALK_YELLOW], text)),
        code_block: Arc::new(|text| chalk(&[CHALK_GREEN], text)),
        code_block_border: Arc::new(|text| chalk(&[CHALK_DIM], text)),
        quote: Arc::new(|text| chalk(&[CHALK_ITALIC], text)),
        quote_border: Arc::new(|text| chalk(&[CHALK_DIM], text)),
        hr: Arc::new(|text| chalk(&[CHALK_DIM], text)),
        list_bullet: Arc::new(|text| chalk(&[CHALK_CYAN], text)),
        bold: Arc::new(|text| chalk(&[CHALK_BOLD], text)),
        italic: Arc::new(|text| chalk(&[CHALK_ITALIC], text)),
        strikethrough: Arc::new(|text| chalk(&[CHALK_STRIKETHROUGH], text)),
        underline: Arc::new(|text| chalk(&[CHALK_UNDERLINE], text)),
        highlight_code: None,
        code_block_indent: None,
    }
}

/// The default text styles the thinking-trace tests style with, upstream's
/// inline `chalk.gray` / `chalk.magenta` / `chalk.cyan` / `chalk.yellow`
/// color callbacks.
#[must_use]
pub fn chalk_gray() -> ColorFn {
    Arc::new(|text| chalk(&[CHALK_GRAY], text))
}

#[must_use]
pub fn chalk_magenta() -> ColorFn {
    Arc::new(|text| chalk(&[CHALK_MAGENTA], text))
}

#[must_use]
pub fn chalk_cyan() -> ColorFn {
    Arc::new(|text| chalk(&[CHALK_CYAN], text))
}

#[must_use]
pub fn chalk_yellow() -> ColorFn {
    Arc::new(|text| chalk(&[CHALK_YELLOW], text))
}

/// The markdown suites' SGR-only strip, upstream `stripAnsi`: drops SGR
/// sequences but keeps OSC 8 payloads, so hyperlink asserts see the URL.
#[must_use]
pub fn strip_ansi(line: &str) -> String {
    static SGR_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    SGR_RE
        .get_or_init(|| regex::Regex::new(r"\x1b\[[0-9;]*m").expect("static pattern"))
        .replace_all(line, "")
        .into_owned()
}

/// Serialize the tests that override the process-wide capability cache:
/// upstream's vitest suite runs sequentially, while cargo runs test
/// functions on parallel threads, and the cache is module-global.
pub fn capabilities_lock() -> std::sync::MutexGuard<'static, ()> {
    static TEST_LOCK: std::sync::LazyLock<std::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(()));
    TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

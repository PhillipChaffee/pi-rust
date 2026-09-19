//! Shared harness for the #43 TUI-core suites, restating upstream's
//! `test/virtual-terminal.ts` until the harness ticket (#45) formalizes the
//! port.
//!
//! - [`TestTerminal`] — the `@xterm/headless` `VirtualTerminal` stand-in: a
//!   `vte`-driven grid emulator with `send_input`/`resize`/`get_viewport`,
//!   plus per-cell italic reads for the style-leak suite.
//! - [`TestRenderer`] — the concrete `TuiBase` subclass the overlay suites
//!   stand in for upstream's `TuiMainScreen`. It renders the mounted
//!   children, composites overlays, extracts the cursor marker, applies the
//!   line resets, and clears-then-rewrites the screen. The real three-
//!   strategy differential renderer lands with #45, at which point these
//!   suites re-point to it; the compositing contract under test lives in the
//!   base either way.
//!
//! The restatements the harness carries: upstream's `waitForRender`
//! (nextTick + 20 ms + flush) collapses into [`wait_for_render`], one pump
//! turn — the port's event-loop turn — because writes land synchronously in
//! the emulator instead of through xterm's async writer.

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
    Component, Focusable, Tui, TuiConfig, TuiMode, TuiMouseButton, TuiMouseEvent,
    TuiMouseEventType, TuiRenderer,
};

/// One screen cell: the grapheme it starts, its visible width (zero for
/// wide-char placeholders), and whether italic is active.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ScreenCell {
    ch: char,
    width: u8,
    italic: bool,
}

impl ScreenCell {
    const BLANK: Self = Self {
        ch: ' ',
        width: 1,
        italic: false,
    };
}

/// The visible screen upstream's xterm buffer exposed, driven by `vte`.
struct Screen {
    grid: Vec<Vec<ScreenCell>>,
    cursor_row: usize,
    cursor_col: usize,
    italic: bool,
}

impl Screen {
    fn new(columns: usize, rows: usize) -> Self {
        Self {
            grid: vec![vec![ScreenCell::BLANK; columns]; rows],
            cursor_row: 0,
            cursor_col: 0,
            italic: false,
        }
    }

    fn resize(&mut self, columns: usize, rows: usize) {
        self.grid.resize(rows, vec![ScreenCell::BLANK; columns]);
        for row in &mut self.grid {
            row.resize(columns, ScreenCell::BLANK);
        }
        self.cursor_row = self.cursor_row.min(rows.saturating_sub(1));
        self.cursor_col = self.cursor_col.min(columns.saturating_sub(1));
    }

    fn scroll_up(&mut self) {
        let columns = self.grid.first().map_or(1, Vec::len);
        self.grid.remove(0);
        self.grid.push(vec![ScreenCell::BLANK; columns]);
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
        };
        let (row, col) = (self.cursor_row, self.cursor_col);
        self.grid[row][col] = cell;
        for offset in 1..width {
            self.grid[row][col + offset] = ScreenCell {
                ch: '\u{0}',
                width: 0,
                italic: cell.italic,
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
        self.grid
            .iter()
            .map(|row| {
                row.iter()
                    .filter(|cell| cell.width > 0)
                    .map(|cell| cell.ch)
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    fn italic_at(&self, row: usize, col: usize) -> bool {
        self.grid
            .get(row)
            .and_then(|line| line.get(col))
            .is_some_and(|cell| cell.italic)
    }
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
pub struct TestTerminal {
    shared: Rc<RefCell<TestShared>>,
}

struct TestShared {
    screen: Screen,
    columns: u16,
    rows: u16,
    input_handler: Option<InputHandler>,
    resize_handler: Option<ResizeHandler>,
    writes: String,
    queued_input: VecDeque<String>,
}

impl TestTerminal {
    /// A fresh `VirtualTerminal(columns, rows)`.
    #[must_use]
    pub(crate) fn new(columns: u16, rows: u16) -> Self {
        Self {
            shared: Rc::new(RefCell::new(TestShared {
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
    /// emulator cannot make (cursor visibility, synchronized output).
    #[must_use]
    pub(crate) fn write_log(&self) -> String {
        self.shared.borrow().writes.clone()
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

    /// The italic attribute of one cell, upstream's `getCellItalic` helper.
    #[must_use]
    pub(crate) fn is_italic(&self, row: usize, col: usize) -> bool {
        self.shared.borrow().screen.italic_at(row, col)
    }
}

impl Terminal for TestTerminal {
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

/// The concrete render strategy the overlay suites stand in for upstream's
/// `TuiMainScreen` until the renderer ticket (#45) lands: renders children,
/// composites overlays, extracts the cursor marker, applies the line resets,
/// then clears and rewrites the screen each frame.
#[derive(Debug)]
pub struct TestRenderer;

impl TuiRenderer for TestRenderer {
    fn mode(&self) -> TuiMode {
        TuiMode::Regular
    }

    fn do_render(&self, tui: &Tui) {
        if tui.is_stopped() {
            return;
        }
        let width = usize::from(tui.terminal_columns());
        let height = usize::from(tui.terminal_rows());
        let mut lines = tui.render_children(width);
        if tui.has_overlay_entries() {
            lines = tui.composite_overlays(lines, width, height);
        }
        let cursor = tui.extract_cursor_position(&mut lines, height);
        let lines = tui.apply_line_resets(lines);
        let mut out = String::from("\x1b[?2026h\x1b[2J\x1b[H");
        for (i, line) in lines.iter().enumerate() {
            if i > 0 {
                out.push_str("\r\n");
            }
            out.push_str(line);
        }
        out.push_str("\x1b[?2026l");
        tui.terminal_write(&out);
        if cursor.is_none() {
            tui.terminal_hide_cursor();
        }
    }
}

/// Build the TUI the suites drive, upstream `new TuiMainScreen(terminal)`
/// over the [`TestRenderer`] stand-in.
#[must_use]
pub fn new_test_tui(terminal: TestTerminal) -> Rc<Tui> {
    Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(Box::new(TestRenderer)),
        ..TuiConfig::default()
    })
}

/// Build the TUI with the image-capable probe injected, restating the
/// cell-size suite's `withImageTerminal` env setup.
#[must_use]
pub fn new_test_tui_with_images(terminal: TestTerminal) -> Rc<Tui> {
    Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(Box::new(TestRenderer)),
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
/// request `start()` made: one pump turn runs the armed deadline.
pub fn wait_for_render(tui: &Tui) {
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

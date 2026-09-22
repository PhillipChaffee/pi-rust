//! The main-screen renderer of `packages/tui/src/tui-main-screen.ts` in pi
//! at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! ([#45](https://github.com/PhillipChaffee/pi-rust/issues/45)).
//!
//! [`TuiMainScreen`] renders into the terminal's main screen and scrollback
//! with the three strategies upstream's `doRender` picks between: a first
//! render writes every line into scrollback without clearing; a width change
//! or a change above the viewport triggers a synchronized clear-screen full
//! redraw; and the normal update moves the cursor to the first changed line
//! and rewrites through to the last. Overwritten regions delete their Kitty
//! image placements before redrawing, and every frame flows through
//! `BoundedTerminalWriter`, which keeps a full render from forming one
//! oversized write.
//! Restatements against upstream, all surveyed in map ticket "Survey the tui
//! package":
//!
//! - `BoundedTerminalWriter`'s UTF-16 surrogate-safe split (survey flag 7)
//!   becomes UTF-8-native chunking: the 1 MiB budget is bytes and a chunk
//!   boundary backs up to the nearest [`char`] boundary, the same
//!   never-split-a-code-point contract without the surrogate arithmetic.
//! - The environment reads (`TERMUX_VERSION`, `PI_TUI_DEBUG_REDRAW`,
//!   `PI_TUI_DEBUG`) resolve through the [`crate::terminal::EnvLookup`] seam:
//!   upstream consulted `process.env` per render, and Rust cannot mutate the
//!   process environment without the `unsafe` this workspace forbids, so
//!   tests inject a map-backed lookup.
//! - Upstream threw an `Error` when a component emitted a line wider than the
//!   terminal; the port panics with the same message after stopping the
//!   session, since there is no caller contract to return an error through —
//!   upstream's throw was uncaught and crashed the process.
//! - The redraw log's ISO timestamp becomes epoch milliseconds and the
//!   `PI_TUI_DEBUG` render dump's `JSON.stringify` becomes Rust debug
//!   formatting, so neither gains a serialization dependency.
//! - Upstream's `KittyImageHeader` parse accepted `Number()`-parsable
//!   integers, including exponent forms; the port's `u64` parse covers the
//!   plain integers Kitty emits and rejects the rest.

use std::cell::{Cell, RefCell};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::terminal::{EnvLookup, default_env_lookup};
use crate::terminal_image::{delete_kitty_image, is_image_line};
use crate::tui::{Tui, TuiMode, TuiRenderer, TuiStopOptions};
use crate::utils::visible_width;

const KITTY_SEQUENCE_PREFIX: &str = "\x1b_G";
const MAX_RENDER_WRITE_CHARS: usize = 1024 * 1024;

/// The deletion sequence for a batch of image ids, upstream
/// `TuiMainScreen.deleteKittyImages`.
/// The insertion-ordered deduplicated ids of every Kitty placement in the
/// lines, upstream `TuiMainScreen.collectKittyImageIds`.
fn collect_kitty_image_ids(lines: &[String]) -> Vec<u64> {
    let mut ids = Vec::new();
    for line in lines {
        for id in extract_kitty_image_ids(line) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

fn delete_kitty_images(ids: &[u64]) -> String {
    let mut buffer = String::new();
    for id in ids {
        buffer.push_str(&delete_kitty_image(*id));
    }
    buffer
}

/// usize into the renderer's i64 cursor arithmetic: line counts ride terminal
/// dimensions and content heights, bounded far below i64.
fn line_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// i64 into a row index: negative values clamp to zero, upstream's
/// `Math.max(0, ...)`.
fn row_usize(value: i64) -> usize {
    usize::try_from(value.max(0)).unwrap_or(usize::MAX)
}

/// A wall-clock stamp for debug artifacts, upstream's ISO timestamp: the
/// epoch-millis restatement keeps the dependency set unchanged.
fn epoch_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

/// The crash or debug dump destination, upstream's
/// `path.join(this.logDirectory ?? os.tmpdir(), ...)`.
fn log_directory_or_temp(tui: &Tui) -> PathBuf {
    tui.log_directory.clone().unwrap_or_else(std::env::temp_dir)
}

fn ensure_parent_dir(path: &Path) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
}

/// The rows a Kitty placement at `index` reserves below its own line, upstream
/// `getKittyImageReservedRows`: the placement's declared rows clamped to the
/// blank, image-free rows that actually follow it.
fn get_kitty_image_reserved_rows(
    lines: &[String],
    index: usize,
    max_index: Option<usize>,
) -> usize {
    let line = lines.get(index).map_or("", String::as_str);
    let rows = extract_kitty_image_rows(line);
    let Some(rows) = usize::try_from(rows).ok().filter(|rows| *rows > 1) else {
        return 1;
    };

    let max_rows = rows
        .min(
            max_index
                .unwrap_or_else(|| lines.len().saturating_sub(1))
                .saturating_sub(index)
                .saturating_add(1),
        )
        .min(lines.len().saturating_sub(index));
    let mut reserved_rows = 1;
    while reserved_rows < max_rows {
        let line = lines.get(index + reserved_rows).map_or("", String::as_str);
        if is_image_line(line) || visible_width(line) > 0 {
            break;
        }
        reserved_rows += 1;
    }
    reserved_rows
}

/// Streams terminal output in 1 MiB chunks so a full render never forms one
/// write large enough to trip a terminal's line-length limits in a single
/// batch.
///
/// [`Self::append`] fills the current chunk and flushes it when full.
/// Oversized input is split at chunk boundaries, backing up to the nearest
/// UTF-8 [`char`] boundary so each write stays valid UTF-8 — upstream's
/// surrogate-pair guard (survey flag 7). Callers append synchronized-output
/// begin/end sequences themselves; the final [`Self::flush`] writes any
/// remainder, including the end sequence.
struct BoundedTerminalWriter<'w> {
    write: &'w mut dyn FnMut(&str),
    buffer: String,
    written_bytes: usize,
}

impl<'w> BoundedTerminalWriter<'w> {
    fn new(write: &'w mut dyn FnMut(&str)) -> Self {
        Self {
            write,
            buffer: String::new(),
            written_bytes: 0,
        }
    }

    /// Append terminal data, flushing full chunks as needed. Callers must
    /// call [`Self::flush`] after the final append.
    fn append(&mut self, value: &str) {
        let mut offset = 0;
        while offset < value.len() {
            let capacity = MAX_RENDER_WRITE_CHARS - self.buffer.len();
            if capacity == 0 {
                self.flush();
                continue;
            }

            let mut end = value.len().min(offset + capacity);
            while end < value.len() && !value.is_char_boundary(end) {
                end -= 1;
            }
            if end == offset {
                self.flush();
                continue;
            }

            self.buffer.push_str(&value[offset..end]);
            offset = end;
            if self.buffer.len() == MAX_RENDER_WRITE_CHARS {
                self.flush();
            }
        }
    }

    /// Write the current chunk, if any, and retain only its byte count for
    /// debug output.
    fn flush(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let chunk = std::mem::take(&mut self.buffer);
        self.written_bytes = self.written_bytes.saturating_add(chunk.len());
        (self.write)(chunk.as_str());
    }

    /// Total bytes appended so far, written and buffered, upstream `length`.
    const fn length(&self) -> usize {
        self.written_bytes + self.buffer.len()
    }
}

struct KittyImageHeader {
    ids: Vec<u64>,
    rows: u64,
}

fn parse_kitty_image_header(line: &str) -> Option<KittyImageHeader> {
    let sequence_start = line.find(KITTY_SEQUENCE_PREFIX)?;
    let params_start = sequence_start + KITTY_SEQUENCE_PREFIX.len();
    let params_end = params_start + line[params_start..].find(';')?;

    let mut ids = Vec::new();
    let mut rows = 1;
    for param in line[params_start..params_end].split(',') {
        let Some((key, value)) = param.split_once('=') else {
            continue;
        };
        // Upstream accepted integers in (0, 0xffffffff]; anything else —
        // non-numeric, negative, fractional, oversized — is skipped.
        let Ok(number_value) = value.parse::<u64>() else {
            continue;
        };
        if number_value == 0 || number_value > 0xffff_ffff {
            continue;
        }
        if key == "i" {
            ids.push(number_value);
        } else if key == "r" {
            rows = number_value;
        }
    }
    Some(KittyImageHeader { ids, rows })
}

fn extract_kitty_image_ids(line: &str) -> Vec<u64> {
    parse_kitty_image_header(line).map_or_else(Vec::new, |header| header.ids)
}

fn extract_kitty_image_rows(line: &str) -> u64 {
    parse_kitty_image_header(line).map_or(1, |header| header.rows)
}

/// Construction seam for [`TuiMainScreen`], restating the environment reads
/// upstream resolved through `process.env` per render.
#[derive(Default)]
pub struct TuiMainScreenConfig {
    /// The environment lookup the renderer consults each frame: the Termux
    /// session probe and the two debug-log switches. `None` reads the real
    /// process environment.
    pub env_lookup: Option<EnvLookup>,
}

impl std::fmt::Debug for TuiMainScreenConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuiMainScreenConfig")
            .field("env_lookup", &self.env_lookup.is_some())
            .finish()
    }
}

/// The renderer state a caller can snapshot around overlay swaps, upstream
/// `TuiMainScreenRenderState`.
#[derive(Debug, Clone)]
pub struct TuiMainScreenRenderState {
    /// The lines the last frame left on the terminal, after compositing and
    /// line resets, upstream `previousLines`.
    pub previous_lines: Vec<String>,
    /// The width the last frame rendered at; `-1` after a reset, upstream
    /// `previousWidth`.
    pub previous_width: i64,
    /// The height the last frame rendered at; `-1` after a reset, upstream
    /// `previousHeight`.
    pub previous_height: i64,
    /// The last content row, upstream `cursorRow`.
    pub cursor_row: usize,
    /// The row the hardware cursor sits on, upstream `hardwareCursorRow`.
    pub hardware_cursor_row: usize,
    /// The working-area high-water mark, upstream `maxLinesRendered`.
    pub max_lines_rendered: usize,
    /// The first buffer line of the last viewport, upstream
    /// `previousViewportTop`.
    pub previous_viewport_top: usize,
}

/// TUI implementation that renders into the terminal's main screen and
/// scrollback, upstream `class TuiMainScreen`.
///
/// Construct through [`TuiMainScreen::new`] and hand to
/// [`crate::tui::TuiConfig::renderer`].
pub struct TuiMainScreen {
    env_lookup: EnvLookup,
    previous_lines: RefCell<Vec<String>>,
    /// Insertion-ordered deduplicated ids of the last frame's Kitty
    /// placements, upstream `previousKittyImageIds`: a `Set` upstream, so the
    /// deletion writes keep its iteration order.
    previous_kitty_image_ids: RefCell<Vec<u64>>,
    previous_width: Cell<i64>,
    previous_height: Cell<i64>,
    cursor_row: Cell<usize>,
    hardware_cursor_row: Cell<usize>,
    max_lines_rendered: Cell<usize>,
    previous_viewport_top: Cell<usize>,
}

impl std::fmt::Debug for TuiMainScreen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuiMainScreen")
            .field("previous_lines", &self.previous_lines.borrow().len())
            .field("previous_width", &self.previous_width.get())
            .field("previous_height", &self.previous_height.get())
            .field("cursor_row", &self.cursor_row.get())
            .field("hardware_cursor_row", &self.hardware_cursor_row.get())
            .field("max_lines_rendered", &self.max_lines_rendered.get())
            .field("previous_viewport_top", &self.previous_viewport_top.get())
            .finish_non_exhaustive()
    }
}

impl TuiMainScreen {
    /// Upstream's `new TuiMainScreen(terminal)` body; the terminal, the
    /// cursor toggle, and the log directory live on the [`Tui`] base, so the
    /// only renderer-local construction input is the environment seam.
    #[must_use]
    pub fn new(config: TuiMainScreenConfig) -> Self {
        Self {
            env_lookup: config.env_lookup.unwrap_or_else(default_env_lookup),
            previous_lines: RefCell::new(Vec::new()),
            previous_kitty_image_ids: RefCell::new(Vec::new()),
            previous_width: Cell::new(0),
            previous_height: Cell::new(0),
            cursor_row: Cell::new(0),
            hardware_cursor_row: Cell::new(0),
            max_lines_rendered: Cell::new(0),
            previous_viewport_top: Cell::new(0),
        }
    }

    fn env_flag(&self, key: &str) -> bool {
        (self.env_lookup)(key).is_some_and(|value| value == "1")
    }

    fn is_termux_session(&self) -> bool {
        // Upstream `Boolean(process.env.TERMUX_VERSION)`: presence of a
        // non-empty value, not the `=1` debug-switch shape.
        (self.env_lookup)("TERMUX_VERSION").is_some_and(|value| !value.is_empty())
    }

    /// Snapshot the render state, upstream `captureRenderState`.
    #[must_use]
    pub fn capture_render_state(&self) -> TuiMainScreenRenderState {
        TuiMainScreenRenderState {
            previous_lines: self.previous_lines.borrow().clone(),
            previous_width: self.previous_width.get(),
            previous_height: self.previous_height.get(),
            cursor_row: self.cursor_row.get(),
            hardware_cursor_row: self.hardware_cursor_row.get(),
            max_lines_rendered: self.max_lines_rendered.get(),
            previous_viewport_top: self.previous_viewport_top.get(),
        }
    }

    /// Restore a snapshot; image lines blank out because their placements
    /// belong to the discarded frame, upstream `restoreRenderState`.
    pub fn restore_render_state(&self, state: &TuiMainScreenRenderState) {
        *self.previous_lines.borrow_mut() = state
            .previous_lines
            .iter()
            .map(|line| {
                if is_image_line(line) {
                    String::new()
                } else {
                    line.clone()
                }
            })
            .collect();
        *self.previous_kitty_image_ids.borrow_mut() = Vec::new();
        self.previous_width.set(state.previous_width);
        self.previous_height.set(state.previous_height);
        self.cursor_row.set(state.cursor_row);
        self.hardware_cursor_row.set(state.hardware_cursor_row);
        self.max_lines_rendered.set(state.max_lines_rendered);
        self.previous_viewport_top.set(state.previous_viewport_top);
    }

    /// Widen the changed range so a touched image block re-renders whole,
    /// upstream `expandChangedRangeForKittyImages`.
    fn expand_changed_range_for_kitty_images(
        &self,
        first_changed: usize,
        last_changed: usize,
        new_lines: &[String],
    ) -> (usize, usize) {
        let mut expanded_first_changed = first_changed;
        let mut expanded_last_changed = last_changed;
        let mut expand_for_lines = |lines: &[String]| {
            for (i, line) in lines.iter().enumerate() {
                if extract_kitty_image_ids(line).is_empty() {
                    continue;
                }
                let block_end = i + get_kitty_image_reserved_rows(lines, i, None) - 1;
                if i >= first_changed || (i <= last_changed && block_end >= first_changed) {
                    expanded_first_changed = expanded_first_changed.min(i);
                    expanded_last_changed = expanded_last_changed.max(block_end);
                }
            }
        };

        expand_for_lines(&self.previous_lines.borrow());
        expand_for_lines(new_lines);
        (expanded_first_changed, expanded_last_changed)
    }

    /// The deletion sequence for every image id in the previous lines the
    /// update overwrites, upstream `deleteChangedKittyImages`.
    fn delete_changed_kitty_images(&self, first_changed: usize, last_changed: usize) -> String {
        if last_changed < first_changed {
            return String::new();
        }

        let previous_lines = self.previous_lines.borrow();
        let mut ids = Vec::new();
        let max_line = last_changed.min(previous_lines.len().saturating_sub(1));
        for i in first_changed..=max_line {
            let line = previous_lines.get(i).map_or("", String::as_str);
            for id in extract_kitty_image_ids(line) {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        delete_kitty_images(&ids)
    }

    /// The clear-and-rewrite strategy, upstream's `fullRender` closure: a
    /// synchronized full redraw of every line, optionally deleting the
    /// previous frame's images and clearing the screen and scrollback first.
    fn full_render(
        &self,
        tui: &Tui,
        new_lines: &[String],
        cursor_pos: Option<(usize, usize)>,
        clear: bool,
        width: usize,
        height: usize,
    ) {
        tui.bump_full_redraws();
        let mut write = |data: &str| tui.terminal_write(data);
        let mut output = BoundedTerminalWriter::new(&mut write);
        output.append("\x1b[?2026h"); // Begin synchronized output
        if clear {
            let ids = self.previous_kitty_image_ids.borrow().clone();
            output.append(&delete_kitty_images(&ids));
            output.append("\x1b[2J\x1b[H\x1b[3J"); // Clear screen, home, then clear scrollback
        }
        let mut index = 0;
        while index < new_lines.len() {
            if index > 0 {
                output.append("\r\n");
            }
            let line = &new_lines[index];
            let is_image = is_image_line(line);
            let image_reserved_rows = if is_image {
                get_kitty_image_reserved_rows(new_lines, index, None)
            } else {
                1
            };
            if image_reserved_rows > 1 && image_reserved_rows <= height {
                for _ in 1..image_reserved_rows {
                    output.append("\r\n");
                }
                output.append(&format!("\x1b[{}A", image_reserved_rows - 1));
                output.append(line);
                output.append(&format!("\x1b[{}B", image_reserved_rows - 1));
                index += image_reserved_rows;
                continue;
            }
            output.append(line);
            index += 1;
        }
        output.append("\x1b[?2026l"); // End synchronized output
        output.flush();
        self.cursor_row.set(new_lines.len().saturating_sub(1));
        self.hardware_cursor_row.set(self.cursor_row.get());
        // Reset max lines when clearing, otherwise track growth
        if clear {
            self.max_lines_rendered.set(new_lines.len());
        } else {
            self.max_lines_rendered
                .set(self.max_lines_rendered.get().max(new_lines.len()));
        }
        let buffer_length = height.max(new_lines.len());
        self.previous_viewport_top
            .set(buffer_length.saturating_sub(height));
        self.position_hardware_cursor(tui, cursor_pos, new_lines.len());
        *self.previous_lines.borrow_mut() = new_lines.to_vec();
        *self.previous_kitty_image_ids.borrow_mut() = collect_kitty_image_ids(new_lines);
        self.previous_width
            .set(i64::from(u16::try_from(width).unwrap_or(u16::MAX)));
        self.previous_height
            .set(i64::from(u16::try_from(height).unwrap_or(u16::MAX)));
    }

    /// The redraw debug log, upstream `logRedraw`: one appended line per full
    /// redraw into `logDirectory/pi-tui-debug.log` when
    /// `PI_TUI_DEBUG_REDRAW=1`.
    fn log_redraw(&self, tui: &Tui, reason: &str, new_lines_len: usize, height: usize) {
        if !self.env_flag("PI_TUI_DEBUG_REDRAW") {
            return;
        }
        let Some(log_directory) = &tui.log_directory else {
            return;
        };
        let log_path = log_directory.join("pi-tui-debug.log");
        let message = format!(
            "[{}] fullRender: {reason} (prev={}, new={new_lines_len}, height={height})\n",
            epoch_millis(),
            self.previous_lines.borrow().len()
        );
        ensure_parent_dir(&log_path);
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            let _ = file.write_all(message.as_bytes());
        }
    }

    /// Position the hardware cursor for the IME candidate window, upstream
    /// `TuiMainScreen.positionHardwareCursor`: hide the cursor when no marker
    /// was found or there is nothing to position on, otherwise move from the
    /// tracked hardware row to the marker's row and column.
    fn position_hardware_cursor(
        &self,
        tui: &Tui,
        cursor_pos: Option<(usize, usize)>,
        total_lines: usize,
    ) {
        let Some((cursor_row, cursor_col)) = cursor_pos.filter(|_| total_lines > 0) else {
            tui.terminal_hide_cursor();
            return;
        };

        // Clamp cursor position to valid range
        let target_row = cursor_row.min(total_lines - 1);
        let target_col = cursor_col;

        // Move cursor from current position to target
        let row_delta = line_i64(target_row) - line_i64(self.hardware_cursor_row.get());
        let mut buffer = String::new();
        if row_delta > 0 {
            let _ = write!(buffer, "\x1b[{row_delta}B"); // Move down
        } else if row_delta < 0 {
            let _ = write!(buffer, "\x1b[{}A", -row_delta); // Move up
        }
        // Move to absolute column (1-indexed)
        let _ = write!(buffer, "\x1b[{}G", target_col + 1);

        if !buffer.is_empty() {
            tui.terminal_write(&buffer);
        }

        self.hardware_cursor_row.set(target_row);
        if tui.get_show_hardware_cursor() {
            tui.terminal_show_cursor();
        } else {
            tui.terminal_hide_cursor();
        }
    }

    /// The three-strategy frame, upstream `TuiMainScreen.doRender`.
    ///
    /// # Panics
    ///
    /// Through [`Self::crash_on_oversized_line`], when a component emits a
    /// line wider than the terminal on the differential path.
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's three-strategy doRender block for block; splitting it would break the 1:1 correspondence"
    )]
    fn render_frame(&self, tui: &Tui) {
        if tui.is_stopped() {
            return;
        }
        let width = usize::from(tui.terminal_columns());
        let height = usize::from(tui.terminal_rows());
        let width_changed = self.previous_width.get() != 0
            && self.previous_width.get() != i64::from(tui.terminal_columns());
        let height_changed = self.previous_height.get() != 0
            && self.previous_height.get() != i64::from(tui.terminal_rows());
        let previous_buffer_length = if self.previous_height.get() > 0 {
            line_i64(self.previous_viewport_top.get()) + self.previous_height.get()
        } else {
            i64::from(tui.terminal_rows())
        };
        let mut prev_viewport_top: usize = if height_changed {
            row_usize(previous_buffer_length - i64::from(tui.terminal_rows()))
        } else {
            self.previous_viewport_top.get()
        };
        let mut viewport_top = prev_viewport_top;
        let mut hardware_cursor_row = self.hardware_cursor_row.get();

        // Render all components to get new lines
        let mut new_lines = tui.render_children(width);

        // Composite overlays into the rendered lines (before differential compare)
        if tui.has_overlay_entries() {
            new_lines = tui.composite_overlays(new_lines, width, height);
        }

        // Extract cursor position before applying line resets (marker must be found first)
        let cursor_pos = tui.extract_cursor_position(&mut new_lines, height);

        let new_lines = tui.apply_line_resets(new_lines);

        // First render - just output everything without clearing (assumes clean screen)
        if self.previous_lines.borrow().is_empty() && !width_changed && !height_changed {
            self.log_redraw(tui, "first render", new_lines.len(), height);
            self.full_render(tui, &new_lines, cursor_pos, false, width, height);
            return;
        }

        // Width changes always need a full re-render because wrapping changes.
        if width_changed {
            self.log_redraw(
                tui,
                &format!(
                    "terminal width changed ({} -> {width})",
                    self.previous_width.get()
                ),
                new_lines.len(),
                height,
            );
            self.full_render(tui, &new_lines, cursor_pos, true, width, height);
            return;
        }

        // Height changes normally need a full re-render to keep the visible viewport aligned,
        // but Termux changes height when the software keyboard shows or hides.
        // In that environment, a full redraw causes the entire history to replay on every toggle.
        if height_changed && !self.is_termux_session() {
            self.log_redraw(
                tui,
                &format!(
                    "terminal height changed ({} -> {height})",
                    self.previous_height.get()
                ),
                new_lines.len(),
                height,
            );
            self.full_render(tui, &new_lines, cursor_pos, true, width, height);
            return;
        }

        // Content shrunk below the working area and no overlays - re-render to clear empty rows
        // (overlays need the padding, so only do this when no overlays are active)
        // Configurable via setClearOnShrink()
        if tui.get_clear_on_shrink()
            && new_lines.len() < self.max_lines_rendered.get()
            && !tui.has_overlay_entries()
        {
            self.log_redraw(
                tui,
                &format!(
                    "clearOnShrink (maxLinesRendered={})",
                    self.max_lines_rendered.get()
                ),
                new_lines.len(),
                height,
            );
            self.full_render(tui, &new_lines, cursor_pos, true, width, height);
            return;
        }

        // Find first and last changed lines
        let mut first_changed: Option<usize> = None;
        let mut last_changed: usize = 0;
        let max_lines = new_lines.len().max(self.previous_lines.borrow().len());
        let previous_lines = self.previous_lines.borrow();
        for i in 0..max_lines {
            let old_line = previous_lines.get(i).map_or("", String::as_str);
            let new_line = new_lines.get(i).map_or("", String::as_str);

            if old_line != new_line {
                if first_changed.is_none() {
                    first_changed = Some(i);
                }
                last_changed = i;
            }
        }
        drop(previous_lines);
        let appended_lines = new_lines.len() > self.previous_lines.borrow().len();
        if appended_lines {
            first_changed = Some(first_changed.unwrap_or(self.previous_lines.borrow().len()));
            last_changed = new_lines.len() - 1;
        }
        if let Some(first) = first_changed {
            let (first, last) =
                self.expand_changed_range_for_kitty_images(first, last_changed, &new_lines);
            first_changed = Some(first);
            last_changed = last;
        }
        let append_start = appended_lines
            && first_changed
                .is_some_and(|first| first == self.previous_lines.borrow().len() && first > 0);

        // No changes - but still need to update hardware cursor position if it moved
        let Some(first_changed) = first_changed else {
            self.position_hardware_cursor(tui, cursor_pos, new_lines.len());
            self.previous_viewport_top.set(prev_viewport_top);
            self.previous_height.set(i64::from(tui.terminal_rows()));
            return;
        };

        // All changes are in deleted lines (nothing to render, just clear)
        if first_changed >= new_lines.len() {
            if self.previous_lines.borrow().len() > new_lines.len() {
                let mut write = |data: &str| tui.terminal_write(data);
                let mut output = BoundedTerminalWriter::new(&mut write);
                output.append("\x1b[?2026h");
                output.append(&self.delete_changed_kitty_images(first_changed, last_changed));
                // Move to end of new content (clamp to 0 for empty content)
                let target_row = new_lines.len().saturating_sub(1);
                if target_row < prev_viewport_top {
                    self.log_redraw(
                        tui,
                        &format!(
                            "deleted lines moved viewport up ({target_row} < {prev_viewport_top})"
                        ),
                        new_lines.len(),
                        height,
                    );
                    self.full_render(tui, &new_lines, cursor_pos, true, width, height);
                    return;
                }
                let current_screen_row =
                    line_i64(hardware_cursor_row) - line_i64(prev_viewport_top);
                let target_screen_row = line_i64(target_row) - line_i64(viewport_top);
                let line_diff = target_screen_row - current_screen_row;
                if line_diff > 0 {
                    output.append(&format!("\x1b[{line_diff}B"));
                } else if line_diff < 0 {
                    output.append(&format!("\x1b[{}A", -line_diff));
                }
                output.append("\r");
                // Clear extra lines without scrolling
                let extra_lines = self.previous_lines.borrow().len() - new_lines.len();
                if extra_lines > height {
                    self.log_redraw(
                        tui,
                        &format!("extraLines > height ({extra_lines} > {height})"),
                        new_lines.len(),
                        height,
                    );
                    self.full_render(tui, &new_lines, cursor_pos, true, width, height);
                    return;
                }
                let clear_start_offset = usize::from(!new_lines.is_empty());
                if extra_lines > 0 && clear_start_offset > 0 {
                    output.append(&format!("\x1b[{clear_start_offset}B"));
                }
                for i in 0..extra_lines {
                    output.append("\r\x1b[2K");
                    if i < extra_lines - 1 {
                        output.append("\x1b[1B");
                    }
                }
                let move_back = extra_lines.saturating_sub(1) + clear_start_offset;
                if move_back > 0 {
                    output.append(&format!("\x1b[{move_back}A"));
                }
                output.append("\x1b[?2026l");
                output.flush();
                self.cursor_row.set(target_row);
                self.hardware_cursor_row.set(target_row);
            }
            self.position_hardware_cursor(tui, cursor_pos, new_lines.len());
            let kitty_ids = collect_kitty_image_ids(&new_lines);
            *self.previous_lines.borrow_mut() = new_lines;
            *self.previous_kitty_image_ids.borrow_mut() = kitty_ids;
            self.previous_width.set(i64::from(tui.terminal_columns()));
            self.previous_height.set(i64::from(tui.terminal_rows()));
            self.previous_viewport_top.set(prev_viewport_top);
            return;
        }

        // Differential rendering can only touch what was actually visible.
        // If the first changed line is above the previous viewport, we need a full redraw.
        if first_changed < prev_viewport_top {
            self.log_redraw(
                tui,
                &format!("firstChanged < viewportTop ({first_changed} < {prev_viewport_top})"),
                new_lines.len(),
                height,
            );
            self.full_render(tui, &new_lines, cursor_pos, true, width, height);
            return;
        }

        // Render from first changed line to end
        // Keep updates wrapped in synchronized output while writing bounded chunks.
        let mut write = |data: &str| tui.terminal_write(data);
        let mut output = BoundedTerminalWriter::new(&mut write);
        output.append("\x1b[?2026h"); // Begin synchronized output
        output.append(&self.delete_changed_kitty_images(first_changed, last_changed));
        let prev_viewport_bottom = line_i64(prev_viewport_top) + line_i64(height) - 1;
        let move_target_row = if append_start {
            first_changed - 1
        } else {
            first_changed
        };
        if line_i64(move_target_row) > prev_viewport_bottom {
            let current_screen_row = (line_i64(hardware_cursor_row) - line_i64(prev_viewport_top))
                .min(line_i64(height) - 1)
                .max(0);
            let move_to_bottom = line_i64(height) - 1 - current_screen_row;
            if move_to_bottom > 0 {
                output.append(&format!("\x1b[{move_to_bottom}B"));
            }
            let scroll = usize::try_from(line_i64(move_target_row) - prev_viewport_bottom)
                .unwrap_or_default();
            output.append(&"\r\n".repeat(scroll));
            prev_viewport_top += scroll;
            viewport_top += scroll;
            hardware_cursor_row = move_target_row;
        }

        // Move cursor to first changed line (use hardwareCursorRow for actual position)
        let current_screen_row = line_i64(hardware_cursor_row) - line_i64(prev_viewport_top);
        let target_screen_row = line_i64(move_target_row) - line_i64(viewport_top);
        let line_diff = target_screen_row - current_screen_row;
        if line_diff > 0 {
            output.append(&format!("\x1b[{line_diff}B")); // Move down
        } else if line_diff < 0 {
            output.append(&format!("\x1b[{}A", -line_diff)); // Move up
        }

        output.append(if append_start { "\r\n" } else { "\r" }); // Move to column 0

        // Only render changed lines (firstChanged to lastChanged), not all lines to end
        // This reduces flicker when only a single line changes (e.g., spinner animation)
        let render_end = last_changed.min(new_lines.len() - 1);
        let mut index = first_changed;
        while index <= render_end {
            if index > first_changed {
                output.append("\r\n");
            }
            let line = &new_lines[index];
            let is_image = is_image_line(line);
            let image_reserved_rows = if is_image {
                get_kitty_image_reserved_rows(&new_lines, index, Some(render_end))
            } else {
                1
            };
            if image_reserved_rows > 1 {
                let image_start_screen_row = line_i64(index) - line_i64(viewport_top);
                if image_start_screen_row < 0
                    || image_start_screen_row + line_i64(image_reserved_rows) > line_i64(height)
                {
                    self.log_redraw(
                        tui,
                        &format!(
                            "kitty image pre-clear would scroll ({image_start_screen_row} + \
                             {image_reserved_rows} > {height})"
                        ),
                        new_lines.len(),
                        height,
                    );
                    self.full_render(tui, &new_lines, cursor_pos, true, width, height);
                    return;
                }

                output.append("\x1b[2K");
                for _ in 1..image_reserved_rows {
                    output.append("\r\n\x1b[2K");
                }
                output.append(&format!("\x1b[{}A", image_reserved_rows - 1));
                output.append(line);
                output.append(&format!("\x1b[{}B", image_reserved_rows - 1));
                index += image_reserved_rows;
                continue;
            }

            output.append("\x1b[2K"); // Clear current line
            if !is_image && visible_width(line) > width {
                crash_on_oversized_line(tui, &new_lines, index, line, width);
                return;
            }
            output.append(line);
            index += 1;
        }

        // Track where cursor ended up after rendering
        let mut final_cursor_row = render_end;

        // If we had more lines before, clear them and move cursor back
        if self.previous_lines.borrow().len() > new_lines.len() {
            // Move to end of new content first if we stopped before it
            if render_end < new_lines.len() - 1 {
                let move_down = new_lines.len() - 1 - render_end;
                output.append(&format!("\x1b[{move_down}B"));
                final_cursor_row = new_lines.len() - 1;
            }
            let extra_lines = self.previous_lines.borrow().len() - new_lines.len();
            for _ in 0..extra_lines {
                output.append("\r\n\x1b[2K");
            }
            // Move cursor back to end of new content
            output.append(&format!("\x1b[{extra_lines}A"));
        }

        output.append("\x1b[?2026l"); // End synchronized output

        if self.env_flag("PI_TUI_DEBUG") {
            let previous_lines = self.previous_lines.borrow();
            let context = DebugDumpContext {
                first_changed,
                viewport_top,
                cursor_row: self.cursor_row.get(),
                height,
                line_diff,
                hardware_cursor_row,
                render_end,
                final_cursor_row,
                cursor_pos,
                new_lines_len: new_lines.len(),
                previous_lines_len: previous_lines.len(),
                new_lines: &new_lines,
                previous_lines: &previous_lines,
            };
            write_render_debug_dump(&output, &context);
        }

        output.flush();

        // Track cursor position for next render
        // cursorRow tracks end of content (for viewport calculation)
        // hardwareCursorRow tracks actual terminal cursor position (for movement)
        self.cursor_row.set(new_lines.len().saturating_sub(1));
        self.hardware_cursor_row.set(final_cursor_row);
        // Track terminal's working area (grows but doesn't shrink unless cleared)
        self.max_lines_rendered
            .set(self.max_lines_rendered.get().max(new_lines.len()));
        self.previous_viewport_top.set(
            prev_viewport_top.max(row_usize(line_i64(final_cursor_row) - line_i64(height) + 1)),
        );

        // Position hardware cursor for IME
        self.position_hardware_cursor(tui, cursor_pos, new_lines.len());

        let kitty_ids = collect_kitty_image_ids(&new_lines);
        *self.previous_lines.borrow_mut() = new_lines;
        *self.previous_kitty_image_ids.borrow_mut() = kitty_ids;
        self.previous_width.set(i64::from(tui.terminal_columns()));
        self.previous_height.set(i64::from(tui.terminal_rows()));
    }
}

/// Upstream's crash path: dump every rendered line to the crash log, stop
/// the session to restore terminal state, then surface the error.
///
/// # Panics
///
/// Always: a component emitting a line wider than the terminal is a contract
/// violation upstream crashed the process with, and the panic message carries
/// the crash-log path the crash-dump suite asserts on.
#[expect(
    clippy::panic,
    reason = "upstream throws when a component emits a line wider than the terminal; there is no caller contract to return an error through, so the panic is the contract-violation surface the crash-dump suite asserts on"
)]
fn crash_on_oversized_line(
    tui: &Tui,
    new_lines: &[String],
    index: usize,
    line: &str,
    width: usize,
) {
    let crash_log_path = log_directory_or_temp(tui).join("pi-tui-crash.log");
    let mut crash_data = format!(
        "Crash at {}\nTerminal width: {width}\nLine {index} visible width: {}\n\n=== All rendered lines ===\n",
        epoch_millis(),
        visible_width(line)
    );
    for (idx, rendered) in new_lines.iter().enumerate() {
        let _ = writeln!(
            crash_data,
            "[{idx}] (w={}) {rendered}",
            visible_width(rendered)
        );
    }
    ensure_parent_dir(&crash_log_path);
    let _ = std::fs::write(&crash_log_path, &crash_data);

    // Clean up terminal state before throwing
    tui.stop(TuiStopOptions::default());

    let error_message = format!(
        "Rendered line {index} exceeds terminal width ({} > {width}).\n\n\
         This is likely caused by a custom TUI component not truncating its output.\n\
         Use visibleWidth() to measure and truncateToWidth() to truncate lines.\n\n\
         Debug log written to: {}",
        visible_width(line),
        crash_log_path.display()
    );
    panic!("{error_message}");
}

/// The values the `PI_TUI_DEBUG` render dump carries, upstream's
/// `debugData` array.
struct DebugDumpContext<'a> {
    first_changed: usize,
    viewport_top: usize,
    cursor_row: usize,
    height: usize,
    line_diff: i64,
    hardware_cursor_row: usize,
    render_end: usize,
    final_cursor_row: usize,
    cursor_pos: Option<(usize, usize)>,
    new_lines_len: usize,
    previous_lines_len: usize,
    new_lines: &'a [String],
    previous_lines: &'a [String],
}

/// The `PI_TUI_DEBUG` render dump, upstream's `debugData` block: one unique
/// file per differential frame under `/tmp/tui`, with the diff coordinates
/// and both line arrays.
fn write_render_debug_dump(output: &BoundedTerminalWriter<'_>, context: &DebugDumpContext<'_>) {
    static NEXT_DUMP_ID: AtomicU64 = AtomicU64::new(1);
    let dump_id = NEXT_DUMP_ID.fetch_add(1, Ordering::Relaxed);
    let debug_dir = Path::new("/tmp/tui");
    let _ = std::fs::create_dir_all(debug_dir);
    let debug_path = debug_dir.join(format!("render-{}-{dump_id}.log", epoch_millis()));
    let mut debug_data = format!(
        "firstChanged: {}\nviewportTop: {}\ncursorRow: {}\nheight: {}\nlineDiff: {}\n\
         hardwareCursorRow: {}\nrenderEnd: {}\nfinalCursorRow: {}\ncursorPos: {:?}\n\
         newLines.length: {}\npreviousLines.length: {}\n",
        context.first_changed,
        context.viewport_top,
        context.cursor_row,
        context.height,
        context.line_diff,
        context.hardware_cursor_row,
        context.render_end,
        context.final_cursor_row,
        context.cursor_pos,
        context.new_lines_len,
        context.previous_lines_len,
    );
    debug_data.push_str("\n=== newLines ===\n");
    let _ = writeln!(debug_data, "{:?}", context.new_lines);
    debug_data.push_str("\n=== previousLines ===\n");
    let _ = writeln!(debug_data, "{:?}", context.previous_lines);
    debug_data.push_str("\n=== buffer ===\n");
    let _ = write!(
        debug_data,
        "[{} bytes written in bounded chunks]",
        output.length()
    );
    let _ = std::fs::write(debug_path, debug_data);
}

impl TuiRenderer for TuiMainScreen {
    fn mode(&self) -> TuiMode {
        TuiMode::Regular
    }

    fn do_render(&self, tui: &Tui) {
        self.render_frame(tui);
    }

    fn reset_render_state(&self) {
        *self.previous_lines.borrow_mut() = Vec::new();
        self.previous_width.set(-1);
        self.previous_height.set(-1);
        self.cursor_row.set(0);
        self.hardware_cursor_row.set(0);
        self.max_lines_rendered.set(0);
        self.previous_viewport_top.set(0);
    }

    fn before_terminal_stop(&self, tui: &Tui, options: &TuiStopOptions) {
        if options.preserve_screen || self.previous_lines.borrow().is_empty() {
            return;
        }
        tui.terminal_write(" ");
        let target_row = self.previous_lines.borrow().len();
        let line_diff = line_i64(target_row) - line_i64(self.hardware_cursor_row.get());
        if line_diff > 0 {
            tui.terminal_write(&format!("\x1b[{line_diff}B"));
        } else if line_diff < 0 {
            tui.terminal_write(&format!("\x1b[{}A", -line_diff));
        }
        tui.terminal_write("\r\n");
    }
}

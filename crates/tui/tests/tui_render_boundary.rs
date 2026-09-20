//! Boundary tests for the #45 renderer slice, added where the ported
//! upstream suites leave branches untested: the render-state capture and
//! restore, the bounded writer's UTF-8 char-boundary chunking (the survey
//! flag 7 restatement, which upstream's surrogate test never exercised), the
//! `PI_TUI_DEBUG` render dump, the hardware-cursor marker positioning, the
//! malformed Kitty header params, and the defensive fall-backs crafted
//! through the restore seam. These bind the 95% coverage gate alongside the
//! ported suites.

#![expect(
    clippy::expect_used,
    reason = "a missing fixture render or unexpected write in a test is an environment failure; expecting keeps the assertions readable"
)]

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use pi_tui::terminal::{EnvLookup, InputHandler, ResizeHandler, Terminal};
use pi_tui::terminal_image::{EncodeKittyOptions, encode_kitty};
use pi_tui::tui::{CURSOR_MARKER, Component, Tui, TuiConfig, TuiMode, TuiRenderer, TuiStopOptions};
use pi_tui::tui_main_screen::{TuiMainScreen, TuiMainScreenConfig, TuiMainScreenRenderState};

use tui_support::{VirtualTerminal, wait_for_render};

const MAX_RENDER_WRITE_CHARS: usize = 1024 * 1024;

/// A component whose render output the test swaps through an `Rc` handle,
/// optionally carrying the cursor marker on a chosen line.
struct TestComponent {
    lines: RefCell<Vec<String>>,
    marker_line: std::cell::Cell<Option<usize>>,
}

impl TestComponent {
    fn set_lines(&self, lines: Vec<&str>) {
        *self.lines.borrow_mut() = lines.into_iter().map(String::from).collect();
    }

    fn set_marker_line(&self, line: Option<usize>) {
        self.marker_line.set(line);
    }
}

impl Component for TestComponent {
    fn render(&self, _width: usize) -> Vec<String> {
        self.lines
            .borrow()
            .iter()
            .enumerate()
            .map(|(i, line)| {
                if self.marker_line.get().is_some_and(|marker| marker == i) {
                    format!("{line}{CURSOR_MARKER}")
                } else {
                    line.clone()
                }
            })
            .collect()
    }

    fn invalidate(&self) {}
}

/// Keeps a handle on the `TuiMainScreen` the TUI owns, so the tests can call
/// the capture/restore surface directly, upstream's typed `TuiMainScreen`
/// reference.
#[derive(Debug)]
struct MainScreenHandle(Rc<TuiMainScreen>);

impl TuiRenderer for MainScreenHandle {
    fn mode(&self) -> TuiMode {
        self.0.mode()
    }

    fn do_render(&self, tui: &Tui) {
        self.0.do_render(tui);
    }

    fn reset_render_state(&self) {
        TuiRenderer::reset_render_state(&*self.0);
    }

    fn before_terminal_stop(&self, tui: &Tui, options: &TuiStopOptions) {
        TuiRenderer::before_terminal_stop(&*self.0, tui, options);
    }
}

fn env_lookup(map: &HashMap<String, String>) -> EnvLookup {
    let map = map.clone();
    Box::new(move |key| map.get(key).cloned())
}

/// Build the TUI and a handle on its renderer, upstream's typed
/// `new TuiMainScreen(terminal)` the suites keep a reference to.
fn new_tui_with_handle(
    terminal: VirtualTerminal,
    env: &HashMap<String, String>,
) -> (Rc<Tui>, Rc<TuiMainScreen>) {
    let renderer = Rc::new(TuiMainScreen::new(TuiMainScreenConfig {
        env_lookup: Some(env_lookup(env)),
    }));
    let tui = Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(Box::new(MainScreenHandle(Rc::clone(&renderer)))),
        ..TuiConfig::default()
    });
    (tui, renderer)
}

fn new_tui(terminal: VirtualTerminal, env: &HashMap<String, String>) -> Rc<Tui> {
    Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(Box::new(TuiMainScreen::new(TuiMainScreenConfig {
            env_lookup: Some(env_lookup(env)),
        }))),
        ..TuiConfig::default()
    })
}

fn new_tui_showing_cursor(terminal: VirtualTerminal) -> Rc<Tui> {
    Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(Box::new(TuiMainScreen::new(TuiMainScreenConfig::default()))),
        show_hardware_cursor: Some(true),
        ..TuiConfig::default()
    })
}

fn kitty_image(base64: &str, columns: usize, rows: usize, image_id: u64) -> String {
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

fn stop(tui: &Tui) {
    tui.stop(TuiStopOptions::default());
}

fn segment_reset() -> String {
    "\x1b[0m\x1b]8;;\u{7}".to_string()
}

#[test]
fn render_state_capture_and_restore_round_trip() {
    let terminal = VirtualTerminal::new(40, 10);
    let (tui, renderer) = new_tui_with_handle(terminal.clone(), &HashMap::new());
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
        marker_line: std::cell::Cell::new(None),
    });
    tui.add_child(component.clone());

    component.set_lines(vec!["hello"]);
    tui.start();
    wait_for_render(&tui);

    let state = renderer.capture_render_state();
    assert_eq!(
        state.previous_lines,
        vec![format!("hello{}", segment_reset())]
    );
    assert_eq!(state.previous_width, 40);
    assert_eq!(state.previous_height, 10);
    assert_eq!(state.cursor_row, 0);
    assert_eq!(state.hardware_cursor_row, 0);
    assert_eq!(state.max_lines_rendered, 1);
    assert_eq!(state.previous_viewport_top, 0);

    // A state whose previous lines carry an image placement restores that
    // line blanked, so the next differential frame re-renders the placement.
    let image = kitty_image("AAAA", 2, 2, 42);
    let captured = TuiMainScreenRenderState {
        previous_lines: vec![image.clone(), "text".to_string()],
        previous_width: 40,
        previous_height: 10,
        cursor_row: 1,
        hardware_cursor_row: 1,
        max_lines_rendered: 2,
        previous_viewport_top: 0,
    };
    renderer.restore_render_state(&captured);

    terminal.clear_writes();
    component.set_lines(vec![&image, "text"]);
    tui.request_render(false);
    wait_for_render(&tui);
    let writes = terminal.write_log();
    assert!(
        writes.contains(&image),
        "the restored image line blanked out, so the placement re-renders: {writes:?}"
    );
    assert!(
        !writes.contains("\x1b[2J"),
        "the restore path stays differential"
    );

    stop(&tui);
}

#[test]
fn bounded_writer_splits_multi_byte_lines_at_char_boundaries() {
    let terminal = BoundedWriteTerminal::new();
    let tui = Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal.clone())),
        renderer: Some(Box::new(TuiMainScreen::new(TuiMainScreenConfig::default()))),
        ..TuiConfig::default()
    });
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
        marker_line: std::cell::Cell::new(None),
    });
    tui.add_child(component.clone());

    component.set_lines(vec!["before"]);
    tui.render_now(false);
    terminal.clear_writes();

    // A differential frame whose changed lines are 2-byte code points: the
    // 1 MiB byte boundary lands inside a character, and the writer backs up
    // to the char boundary instead of splitting it (survey flag 7).
    let payload = "é".repeat(700_000);
    let line = format!("\x1b_Ga=T,f=100;{payload}\x1b\\");
    component.set_lines(vec!["before", &line, &line]);
    tui.render_now(false);

    let writes = terminal.writes();
    assert!(writes.len() > 2, "the multi-byte line splits across writes");
    assert!(
        writes
            .iter()
            .all(|write| write.len() <= MAX_RENDER_WRITE_CHARS),
        "each write stays under the byte budget"
    );
    assert_eq!(
        writes.concat(),
        format!("\x1b[?2026h\r\n\x1b[2K{line}\r\n\x1b[2K{line}\x1b[?2026l")
    );
    stop(&tui);
}

/// The ported suite's `BoundedWriteTerminal` shape, local to the boundary
/// file's chunking test.
#[derive(Clone)]
struct BoundedWriteTerminal {
    writes: Rc<RefCell<Vec<String>>>,
}

impl BoundedWriteTerminal {
    fn new() -> Self {
        Self {
            writes: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn writes(&self) -> Vec<String> {
        self.writes.borrow().clone()
    }

    fn clear_writes(&self) {
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

#[test]
fn render_debug_dump_writes_the_frame_coordinates() {
    let mut env = HashMap::new();
    env.insert("PI_TUI_DEBUG".to_string(), "1".to_string());
    let terminal = VirtualTerminal::new(40, 10);
    let tui = new_tui(terminal, &env);
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
        marker_line: std::cell::Cell::new(None),
    });
    tui.add_child(component.clone());

    component.set_lines(vec!["a"]);
    tui.start();
    wait_for_render(&tui);

    let debug_dir = std::path::Path::new("/tmp/tui");
    let before: HashSet<PathBuf> = std::fs::read_dir(debug_dir)
        .map(|entries| entries.flatten().map(|entry| entry.path()).collect())
        .unwrap_or_default();

    // The dump rides the differential path, upstream's block placement.
    component.set_lines(vec!["a", "b"]);
    tui.request_render(false);
    wait_for_render(&tui);

    let after: HashSet<PathBuf> = std::fs::read_dir(debug_dir)
        .map(|entries| entries.flatten().map(|entry| entry.path()).collect())
        .expect("the debug directory exists after the dump");
    let new_files: Vec<PathBuf> = after.difference(&before).cloned().collect();
    assert_eq!(
        new_files.len(),
        1,
        "one dump per differential frame: {new_files:?}"
    );
    let dump_path = new_files.first().expect("one dump");
    let dump = std::fs::read_to_string(dump_path).expect("dump readable");
    assert!(dump.contains("firstChanged: 1"));
    assert!(dump.contains("=== newLines ==="));
    assert!(dump.contains("bytes written in bounded chunks"));
    let _ = std::fs::remove_file(dump_path);

    stop(&tui);
}

#[test]
fn differential_append_scrolls_and_moves_to_the_viewport_bottom() {
    let terminal = VirtualTerminal::new(40, 6);
    let tui = new_tui_showing_cursor(terminal.clone());
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
        marker_line: std::cell::Cell::new(None),
    });
    tui.add_child(component.clone());

    // The marker parks the hardware cursor on row 1, above the bottom row.
    let lines: Vec<String> = (0..7).map(|i| format!("Line {i}")).collect();
    let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    component.set_marker_line(Some(1));
    component.set_lines(line_refs);
    tui.start();
    wait_for_render(&tui);
    let writes = terminal.write_log();
    assert!(
        writes.contains("\x1b[7G") && writes.contains("\x1b[?25h"),
        "the marker positions the hardware cursor and shows it: {writes:?}"
    );
    terminal.clear_writes();

    // Appending past the viewport bottom keeps the differential path and the
    // content lands below the scrolled lines.
    let redraws_before = tui.full_redraws();
    let mut more = lines.clone();
    more.extend([
        "Appended 0".to_string(),
        "Appended 1".to_string(),
        "Appended 2".to_string(),
    ]);
    let more_refs: Vec<&str> = more.iter().map(String::as_str).collect();
    component.set_marker_line(None);
    component.set_lines(more_refs);
    tui.request_render(false);
    wait_for_render(&tui);

    assert_eq!(
        tui.full_redraws(),
        redraws_before,
        "the append stays differential"
    );
    let writes = terminal.write_log();
    assert!(
        writes.contains("\x1b[5B"),
        "the cursor moves down to the first changed line: {writes:?}"
    );
    let viewport = terminal.get_viewport();
    assert_eq!(
        viewport,
        vec![
            "Line 4",
            "Line 5",
            "Line 6",
            "Appended 0",
            "Appended 1",
            "Appended 2"
        ]
    );

    stop(&tui);
}

#[test]
fn kitty_header_skips_malformed_params_and_adjacent_images_stop_reservation() {
    let terminal = VirtualTerminal::new(40, 10);
    let tui = new_tui(terminal.clone(), &HashMap::new());
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
        marker_line: std::cell::Cell::new(None),
    });
    tui.add_child(component.clone());

    component.set_lines(vec!["top"]);
    tui.start();
    wait_for_render(&tui);
    terminal.clear_writes();

    // A header with a non-numeric value, an id of zero, an oversized id, a
    // param without `=`, and an oversized `r=`: all skipped on the parse.
    // The adjacent image line also stops the reserved-row walk.
    let malformed = "\x1b_Ga=T,i=x,i=0,i=99999999999,bogus,r=99,q=2;AAAA\x1b\\";
    let adjacent = kitty_image("BBBB", 2, 2, 43);
    component.set_lines(vec!["top", malformed, &adjacent, "text"]);
    tui.request_render(false);
    wait_for_render(&tui);

    let writes = terminal.write_log();
    assert!(
        !writes.contains("\x1b_Ga=d"),
        "no image ids parsed, so nothing is deleted: {writes:?}"
    );
    let viewport = terminal.get_viewport();
    assert!(
        viewport.iter().any(|line| line.contains("text")),
        "content renders after the malformed headers: {viewport:?}"
    );

    stop(&tui);
}

#[test]
fn oversized_extra_lines_fall_back_to_a_full_redraw() {
    let terminal = VirtualTerminal::new(40, 5);
    let (tui, renderer) = new_tui_with_handle(terminal.clone(), &HashMap::new());
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
        marker_line: std::cell::Cell::new(None),
    });
    tui.add_child(component.clone());
    component.set_lines(vec!["x"]);
    tui.start();
    wait_for_render(&tui);

    // Craft the previous state through the restore seam: ten lines with a
    // viewport top of zero, which the deleted-lines branch alone can reach.
    // The restore input carries the segment resets the renderer stores, and
    // the swapped-in lines stay raw because the renderer appends the resets.
    let lines: Vec<String> = (0..10).map(|i| format!("Stale {i}")).collect();
    let crafted = TuiMainScreenRenderState {
        previous_lines: lines
            .iter()
            .map(|line| format!("{line}{}", segment_reset()))
            .collect(),
        previous_width: 40,
        previous_height: 5,
        cursor_row: 9,
        hardware_cursor_row: 9,
        max_lines_rendered: 10,
        previous_viewport_top: 0,
    };
    renderer.restore_render_state(&crafted);

    let redraws_before = tui.full_redraws();
    // Shrink to the identical first two lines: a pure deletion whose
    // extra-line count exceeds the height.
    let raw_refs: Vec<&str> = lines.iter().take(2).map(String::as_str).collect();
    component.set_lines(raw_refs);
    tui.request_render(false);
    wait_for_render(&tui);

    assert!(
        tui.full_redraws() > redraws_before,
        "extra lines beyond the height force a full redraw"
    );
    let viewport = terminal.get_viewport();
    assert!(
        viewport
            .first()
            .is_some_and(|line| line.contains("Stale 0")),
        "content preserved after the fallback: {viewport:?}"
    );

    stop(&tui);
}

#[test]
fn scrolling_past_the_viewport_bottom_stays_differential() {
    let terminal = VirtualTerminal::new(40, 6);
    let (tui, renderer) = new_tui_with_handle(terminal.clone(), &HashMap::new());
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
        marker_line: std::cell::Cell::new(None),
    });
    tui.add_child(component.clone());
    component.set_lines(vec!["seed"]);
    tui.start();
    wait_for_render(&tui);

    // Craft the previous state through the restore seam: twenty lines with a
    // viewport top of zero, so the first changed line sits past the previous
    // viewport bottom and the scroll branch fires. The restore input carries
    // the segment resets the renderer stores, and the swapped-in lines stay
    // raw because the renderer appends the resets.
    let previous: Vec<String> = (0..20).map(|i| format!("Old {i}")).collect();
    let crafted = TuiMainScreenRenderState {
        previous_lines: previous
            .iter()
            .map(|line| format!("{line}{}", segment_reset()))
            .collect(),
        previous_width: 40,
        previous_height: 6,
        cursor_row: 19,
        hardware_cursor_row: 19,
        max_lines_rendered: 20,
        previous_viewport_top: 0,
    };
    renderer.restore_render_state(&crafted);

    let redraws_before = tui.full_redraws();
    let mut new_lines = previous;
    new_lines[10] = "CHANGED".to_string();
    let new_refs: Vec<&str> = new_lines.iter().map(String::as_str).collect();
    component.set_lines(new_refs);
    tui.request_render(false);
    wait_for_render(&tui);

    assert_eq!(
        tui.full_redraws(),
        redraws_before,
        "the scroll branch stays on the differential path"
    );
    let writes = terminal.write_log();
    assert!(
        writes.contains("\r\n\r\n\r\n\r\n\r\n"),
        "the frame scrolls to bring the changed line into the viewport: {writes:?}"
    );
    assert!(!writes.contains("\x1b[2J"), "no full redraw");

    stop(&tui);
}

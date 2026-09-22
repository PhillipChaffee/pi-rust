//! `tui-render.test.ts` ported 1:1 (upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`): render scheduling, debug
//! logging, bounded render output, the crash dump, Kitty image cleanup,
//! resize handling, content shrinkage, and differential rendering.
//!
//! The image placements are constructed with [`pi_tui::terminal_image::encode_kitty`]
//! directly: the `Image` component and its capability gate are the image
//! ticket's scope (#51), and the renderer under test only reads the `r=`/`i=`
//! header fields of the line, so `encodeKitty` with the same cell geometry
//! the `Image` component computes for those inputs drives the identical
//! branches.

#![expect(
    clippy::expect_used,
    reason = "a missing fixture render or unexpected write in a test is an environment failure; expecting keeps the assertions readable"
)]

#[path = "tui_support/mod.rs"]
mod tui_support;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use pi_tui::terminal::Terminal;
use pi_tui::terminal_image::delete_kitty_image;
use pi_tui::tui::{Component, Tui, TuiConfig};
use pi_tui::tui_main_screen::{TuiMainScreen, TuiMainScreenConfig};

use tui_support::{
    BoundedWriteTerminal, VirtualTerminal, kitty_image, new_main_screen_tui, stop, wait_for_render,
};

const MAX_RENDER_WRITE_CHARS: usize = 1024 * 1024;

/// The suite's `TestComponent`.
struct TestComponent {
    lines: RefCell<Vec<String>>,
}

impl TestComponent {
    fn set_lines(&self, lines: Vec<&str>) {
        *self.lines.borrow_mut() = lines.into_iter().map(String::from).collect();
    }
}

impl Component for TestComponent {
    fn render(&self, _width: usize) -> Vec<String> {
        self.lines.borrow().clone()
    }

    fn invalidate(&self) {}
}

/// The suite's `InputComponent`: records render calls and replaces its lines
/// with the data it receives.
struct InputComponent {
    lines: RefCell<Vec<String>>,
    render_count: Cell<usize>,
}

impl InputComponent {
    fn set_lines(&self, lines: Vec<&str>) {
        *self.lines.borrow_mut() = lines.into_iter().map(String::from).collect();
    }
}

impl Component for InputComponent {
    fn render(&self, _width: usize) -> Vec<String> {
        self.render_count.set(self.render_count.get() + 1);
        self.lines.borrow().clone()
    }

    fn handle_input(&self, data: &str) {
        *self.lines.borrow_mut() = vec![data.to_string()];
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {}
}

fn new_tui_with_log_dir(
    terminal: VirtualTerminal,
    env: &HashMap<String, String>,
    log_dir: PathBuf,
) -> Rc<Tui> {
    Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        renderer: Some(Box::new(TuiMainScreen::new(TuiMainScreenConfig {
            env_lookup: Some(tui_support::env_lookup(env)),
        }))),
        log_directory: Some(log_dir),
        ..TuiConfig::default()
    })
}

fn temp_dir(prefix: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "{prefix}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos())
    ))
}

/// The suites' most common fixture: a started TUI carrying one fresh
/// [`TestComponent`], with handles on the terminal and the component.
fn started_component(width: u16, height: u16) -> (Rc<Tui>, VirtualTerminal, Rc<TestComponent>) {
    let terminal = VirtualTerminal::new(width, height);
    let tui = new_main_screen_tui(terminal.clone(), &HashMap::new());
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
    });
    tui.add_child(component.clone());
    (tui, terminal, component)
}

// === TUI render scheduling ===

#[test]
fn renders_keyboard_input_without_waiting_for_a_throttled_frame() {
    let terminal = VirtualTerminal::new(40, 10);
    let tui = new_main_screen_tui(terminal.clone(), &HashMap::new());
    let component = Rc::new(InputComponent {
        lines: RefCell::new(vec!["initial".to_string()]),
        render_count: Cell::new(0),
    });
    tui.add_child(component.clone());
    tui.set_focus(Some(component.clone()));
    tui.start();
    tui.render_now(false);
    let render_count_before_input = component.render_count.get();

    // Queue a normal throttled render first. Keyboard input should preempt it.
    component.set_lines(vec!["pending"]);
    tui.request_render(false);
    terminal.send_input("first");
    terminal.send_input("second");
    terminal.send_input("typed");
    // Upstream awaits one `process.nextTick`; the pump is that turn.
    tui.poll(Duration::ZERO);

    assert_eq!(
        component.render_count.get(),
        render_count_before_input + 1,
        "input renders immediately, preempting the throttled frame"
    );
    assert_eq!(*component.lines.borrow(), vec!["typed".to_string()]);
    stop(&tui);
}

// === TUI debug logging ===

#[test]
fn writes_redraw_logs_to_the_provided_directory() {
    let log_dir = temp_dir("pi-tui-log");
    assert!(
        std::fs::create_dir_all(&log_dir).is_ok(),
        "create log dir: {log_dir:?}"
    );

    let mut env = HashMap::new();
    env.insert("PI_TUI_DEBUG_REDRAW".to_string(), "1".to_string());
    let terminal = VirtualTerminal::new(40, 10);
    let tui = new_tui_with_log_dir(terminal, &env, log_dir.clone());
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
    });
    tui.add_child(component.clone());
    component.set_lines(vec!["test"]);
    tui.start();
    wait_for_render(&tui);

    let log = std::fs::read_to_string(log_dir.join("pi-tui-debug.log"))
        .expect("redraw log written to the provided directory");
    assert!(log.contains("fullRender: first render"));
    stop(&tui);
    let _ = std::fs::remove_dir_all(&log_dir);
}

// === TUI bounded render output ===

#[test]
fn splits_a_large_full_render_without_changing_its_output() {
    let terminal = BoundedWriteTerminal::new();
    let tui = Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal.clone())),
        renderer: Some(Box::new(TuiMainScreen::new(TuiMainScreenConfig::default()))),
        ..TuiConfig::default()
    });
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
    });
    let kitty_line = format!("\x1b_Ga=T,f=100;{}\x1b\\", "A".repeat(1_200_000));
    component.set_lines(vec![&kitty_line, &kitty_line]);
    tui.add_child(component);

    tui.render_now(false);

    let writes = terminal.writes();
    assert!(
        writes.len() > 2,
        "large output should be split across terminal writes"
    );
    assert!(
        writes
            .iter()
            .all(|write| write.len() <= MAX_RENDER_WRITE_CHARS),
        "each terminal write should stay below the configured limit"
    );
    assert_eq!(
        writes.concat(),
        format!("\x1b[?2026h{kitty_line}\r\n{kitty_line}\x1b[?2026l"),
        "chunking must preserve the synchronized render output"
    );
}

#[test]
fn splits_large_differential_updates_without_a_full_redraw() {
    let terminal = BoundedWriteTerminal::new();
    let tui = Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal.clone())),
        renderer: Some(Box::new(TuiMainScreen::new(TuiMainScreenConfig::default()))),
        ..TuiConfig::default()
    });
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
    });
    tui.add_child(component.clone());
    component.set_lines(vec!["before"]);
    tui.render_now(false);
    terminal.clear_writes();

    let kitty_line = format!("\x1b_Ga=T,f=100;{}\x1b\\", "A".repeat(1_200_000));
    component.set_lines(vec!["before", &kitty_line, &kitty_line]);
    tui.render_now(false);

    let writes = terminal.writes();
    assert!(
        writes.len() > 2,
        "large output should be split across terminal writes"
    );
    assert!(
        writes
            .iter()
            .all(|write| write.len() <= MAX_RENDER_WRITE_CHARS),
        "each terminal write should stay below the configured limit"
    );
    let output = writes.concat();
    assert!(output.starts_with("\x1b[?2026h"));
    assert!(output.ends_with("\x1b[?2026l"));
    assert!(
        !output.contains("\x1b[2J"),
        "the update should stay on the differential render path"
    );
    stop(&tui);
}

// === TUI crash dump without configured log directory ===

#[test]
fn writes_the_crash_dump_to_the_os_temp_directory_instead_of_a_home_directory_default() {
    let terminal = VirtualTerminal::new(40, 10);
    let tui = new_main_screen_tui(terminal, &HashMap::new());
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
    });
    tui.add_child(component.clone());
    component.set_lines(vec!["ok"]);
    tui.start();
    wait_for_render(&tui);

    // Width overflow is detected in the differential render path
    let crash_log_path = std::env::temp_dir().join("pi-tui-crash.log");
    let _ = std::fs::remove_file(&crash_log_path);
    let long_line = "x".repeat(60);
    component.set_lines(vec!["ok", &long_line]);
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| tui.render_now(false)));
    assert!(result.is_err(), "the width overflow must crash the render");
    let message = result
        .expect_err("the width overflow must panic")
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(
        message.contains(crash_log_path.to_string_lossy().as_ref()),
        "error message should reference {crash_log_path:?}: {message}"
    );
    let crash_log = std::fs::read_to_string(&crash_log_path).expect("crash log written");
    assert!(crash_log.contains("Terminal width: 40"));
    let _ = std::fs::remove_file(&crash_log_path);
}

// === TUI Kitty image cleanup ===

#[test]
fn clears_reserved_kitty_image_rows_before_drawing_appended_image_placements() {
    let (tui, terminal, component) = started_component(40, 10);

    component.set_lines(vec!["before"]);
    tui.start();
    wait_for_render(&tui);
    terminal.clear_writes();

    // The `Image` component for a 20x20 px image at a 10 px cell renders one
    // placement line carrying r=2 plus one reserved blank row.
    let image_lines = [kitty_image("AAAA", 2, 2, 42), String::new()];
    let image_sequence = image_lines[0].clone();
    let mut lines = vec!["before"];
    lines.extend(image_lines.iter().map(String::as_str));
    lines.push("after");
    component.set_lines(lines);
    tui.request_render(false);
    wait_for_render(&tui);

    let writes = terminal.write_log();
    assert!(
        writes.contains(&format!("\x1b[2K\r\n\x1b[2K\x1b[1A{image_sequence}\x1b[1B")),
        "reserved rows should be cleared before the image placement is drawn: {writes:?}"
    );
    assert!(
        !writes.contains(&format!("{image_sequence}\r\n\x1b[2K")),
        "reserved row clears must not run after the image placement is drawn"
    );

    stop(&tui);
}

#[test]
fn falls_back_to_full_redraw_when_kitty_image_pre_clear_would_scroll() {
    let (tui, terminal, component) = started_component(40, 2);

    component.set_lines(vec!["before"]);
    tui.start();
    wait_for_render(&tui);
    let redraws_before_image = tui.full_redraws();
    terminal.clear_writes();

    // 30x30 px at a 10 px cell renders r=3, one row past the 2-row terminal.
    let image_lines = [kitty_image("AAAA", 3, 3, 42), String::new(), String::new()];
    let mut lines = vec!["before"];
    lines.extend(image_lines.iter().map(String::as_str));
    lines.push("after");
    component.set_lines(lines);
    tui.request_render(false);
    wait_for_render(&tui);

    assert!(
        tui.full_redraws() > redraws_before_image,
        "unsafe image pre-clear should force a full redraw"
    );
    assert!(
        terminal.write_log().contains("\x1b[2J"),
        "fallback should clear and fully redraw"
    );

    stop(&tui);
}

#[test]
fn reserves_kitty_image_rows_before_drawing_during_full_redraw_fallbacks() {
    let (tui, terminal, component) = started_component(40, 5);

    component.set_lines(vec!["l0", "l1", "l2", "l3", "l4"]);
    tui.start();
    wait_for_render(&tui);
    let redraws_before_image = tui.full_redraws();
    terminal.clear_writes();

    let image_lines = [kitty_image("AAAA", 3, 3, 42), String::new(), String::new()];
    let image_sequence = image_lines[0].clone();
    let mut lines = vec!["l0", "l1", "l2", "l3", "l4"];
    lines.extend(image_lines.iter().map(String::as_str));
    lines.push("after");
    component.set_lines(lines);
    tui.request_render(false);
    wait_for_render(&tui);

    let writes = terminal.write_log();
    assert!(
        tui.full_redraws() > redraws_before_image,
        "scrolling image append should force a full redraw"
    );
    assert!(
        writes.contains(&format!("\r\n\r\n\x1b[2A{image_sequence}\x1b[2B")),
        "full redraw should reserve visible image rows before drawing the placement: {writes:?}"
    );
    assert!(
        !writes.contains(&format!("{image_sequence}\r\n\x1b[0m")),
        "full redraw must not write reserved padding rows after drawing the placement"
    );

    stop(&tui);
}

#[test]
fn does_not_use_cursor_up_placement_for_kitty_images_taller_than_the_viewport() {
    let (tui, terminal, component) = started_component(40, 5);

    component.set_lines(vec!["before"]);
    tui.start();
    wait_for_render(&tui);
    terminal.clear_writes();

    // 60x60 px at a 10 px cell renders r=6 with five reserved blank rows.
    let image_lines = [
        kitty_image("AAAA", 6, 6, 42),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    ];
    let image_sequence = image_lines[0].clone();
    assert!(
        image_lines.len() > usize::from(terminal.rows()),
        "test image should exceed the viewport height"
    );

    let mut lines = vec!["before"];
    lines.extend(image_lines.iter().map(String::as_str));
    lines.push("after");
    component.set_lines(lines);
    tui.request_render(true);
    wait_for_render(&tui);

    let writes = terminal.write_log();
    assert!(
        writes.contains(&image_sequence),
        "image placement should be drawn"
    );
    assert!(
        !writes.contains(&format!("\x1b[{}A{image_sequence}", image_lines.len() - 1)),
        "taller-than-viewport images must keep the #4461 first-row placement path"
    );

    stop(&tui);
}

#[test]
fn deletes_changed_image_ids_before_drawing_moved_placements() {
    let (tui, terminal, component) = started_component(40, 10);

    let old_image = kitty_image("AAAA", 2, 2, 42);
    component.set_lines(vec!["top", &old_image]);
    tui.start();
    wait_for_render(&tui);
    terminal.clear_writes();

    let new_image = kitty_image("BBBB", 2, 1, 42);
    component.set_lines(vec![&new_image, ""]);
    tui.request_render(false);
    wait_for_render(&tui);

    let writes = terminal.write_log();
    let deletion = delete_kitty_image(42);
    let delete_index = writes.find(&deletion);
    let draw_index = writes.find(&new_image);
    assert!(
        delete_index.is_some(),
        "changed old image should be deleted"
    );
    assert!(draw_index.is_some(), "new image should be drawn");
    assert!(
        delete_index.unwrap_or(0) < draw_index.unwrap_or(usize::MAX),
        "old image must be deleted before the new placement is drawn"
    );

    stop(&tui);
}

#[test]
fn redraws_image_lines_when_an_earlier_reserved_image_row_changes() {
    let (tui, terminal, component) = started_component(40, 10);

    let image = kitty_image("AAAA", 2, 2, 88);
    component.set_lines(vec!["", &image]);
    tui.start();
    wait_for_render(&tui);
    terminal.clear_writes();

    component.set_lines(vec!["covered", &image]);
    tui.request_render(false);
    wait_for_render(&tui);

    let writes = terminal.write_log();
    let deletion = delete_kitty_image(88);
    let delete_index = writes.find(&deletion);
    let draw_index = writes.find(&image);
    assert!(
        delete_index.is_some(),
        "image should be deleted when a reserved row changes"
    );
    assert!(
        draw_index.is_some(),
        "unchanged image line should be redrawn after deleting the placement"
    );
    assert!(
        delete_index.unwrap_or(0) < draw_index.unwrap_or(usize::MAX),
        "old placement must be deleted before the image line is redrawn"
    );
    assert!(
        !writes.contains("\x1b[2J"),
        "reserved row changes should not force a full redraw"
    );

    stop(&tui);
}

#[test]
fn deletes_previously_rendered_image_ids_during_full_redraws() {
    let (tui, terminal, component) = started_component(40, 10);

    let image = kitty_image("AAAA", 2, 2, 77);
    component.set_lines(vec![&image]);
    tui.start();
    wait_for_render(&tui);
    terminal.clear_writes();

    component.set_lines(vec!["plain text"]);
    tui.request_render(true);
    wait_for_render(&tui);

    let writes = terminal.write_log();
    let deletion = delete_kitty_image(77);
    let delete_index = writes.find(&deletion);
    let clear_index = writes.find("\x1b[2J");
    assert!(
        delete_index.is_some(),
        "previous image should be deleted during full redraw"
    );
    assert!(clear_index.is_some(), "full redraw should clear the screen");
    assert!(
        delete_index.unwrap_or(0) < clear_index.unwrap_or(usize::MAX),
        "old image should be deleted before the screen is cleared"
    );

    stop(&tui);
}

// === TUI resize handling ===

#[test]
fn triggers_full_re_render_when_terminal_height_changes() {
    // The injected environment lookup answers nothing, so the session is not
    // a Termux session, upstream's `withEnv({ TERMUX_VERSION: undefined })`.
    let (tui, terminal, component) = started_component(40, 10);

    component.set_lines(vec!["Line 0", "Line 1", "Line 2"]);
    tui.start();
    wait_for_render(&tui);

    let initial_redraws = tui.full_redraws();

    // Resize height
    terminal.resize(40, 15);
    wait_for_render(&tui);

    // Should have triggered a full redraw
    assert!(
        tui.full_redraws() > initial_redraws,
        "Height change should trigger full redraw"
    );

    let viewport = terminal.get_viewport();
    assert!(
        viewport.first().is_some_and(|line| line.contains("Line 0")),
        "Content preserved after height change"
    );

    stop(&tui);
}

#[test]
fn skips_full_re_render_on_height_changes_in_termux() {
    let mut env = HashMap::new();
    env.insert("TERMUX_VERSION".to_string(), "1".to_string());
    let terminal = VirtualTerminal::new(40, 10);
    let tui = new_main_screen_tui(terminal.clone(), &env);
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
    });
    tui.add_child(component.clone());

    let lines: Vec<String> = (0..20).map(|i| format!("Line {i}")).collect();
    let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    component.set_lines(line_refs);
    tui.start();
    wait_for_render(&tui);
    terminal.clear_writes();

    let initial_redraws = tui.full_redraws();
    for height in [15, 8, 14, 11] {
        terminal.resize(40, height);
        wait_for_render(&tui);
    }

    assert_eq!(
        tui.full_redraws(),
        initial_redraws,
        "Height change should not trigger full redraw"
    );
    assert!(
        !terminal.write_log().contains("\x1b[2J"),
        "Height change should not clear the screen"
    );
    assert!(
        !terminal.write_log().contains("\x1b[3J"),
        "Height change should not clear scrollback"
    );

    let viewport = terminal.get_viewport();
    assert!(
        viewport.join("\n").contains("Line 19"),
        "Latest content remains visible after resize: {viewport:?}"
    );

    stop(&tui);
}

#[test]
fn triggers_full_re_render_when_terminal_width_changes() {
    let (tui, terminal, component) = started_component(40, 10);

    component.set_lines(vec!["Line 0", "Line 1", "Line 2"]);
    tui.start();
    wait_for_render(&tui);

    let initial_redraws = tui.full_redraws();

    // Resize width
    terminal.resize(60, 10);
    wait_for_render(&tui);

    // Should have triggered a full redraw
    assert!(
        tui.full_redraws() > initial_redraws,
        "Width change should trigger full redraw"
    );

    stop(&tui);
}

// === TUI content shrinkage ===

#[test]
fn clears_empty_rows_when_content_shrinks_significantly() {
    let terminal = VirtualTerminal::new(40, 10);
    let tui = new_main_screen_tui(terminal.clone(), &HashMap::new());
    tui.set_clear_on_shrink(true); // Explicitly enable (may be disabled via env var)
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
    });
    tui.add_child(component.clone());

    // Start with many lines
    component.set_lines(vec![
        "Line 0", "Line 1", "Line 2", "Line 3", "Line 4", "Line 5",
    ]);
    tui.start();
    wait_for_render(&tui);

    let initial_redraws = tui.full_redraws();

    // Shrink to fewer lines
    component.set_lines(vec!["Line 0", "Line 1"]);
    tui.request_render(false);
    wait_for_render(&tui);

    // Should have triggered a full redraw to clear empty rows
    assert!(
        tui.full_redraws() > initial_redraws,
        "Content shrinkage should trigger full redraw"
    );

    let viewport = terminal.get_viewport();
    assert!(
        viewport.first().is_some_and(|line| line.contains("Line 0")),
        "First line preserved"
    );
    assert!(
        viewport.get(1).is_some_and(|line| line.contains("Line 1")),
        "Second line preserved"
    );
    // Lines below should be empty (cleared)
    assert!(
        viewport.get(2).is_some_and(|line| line.trim().is_empty()),
        "Line 2 should be cleared"
    );
    assert!(
        viewport.get(3).is_some_and(|line| line.trim().is_empty()),
        "Line 3 should be cleared"
    );

    stop(&tui);
}

#[test]
fn handles_shrink_to_single_line() {
    let terminal = VirtualTerminal::new(40, 10);
    let tui = new_main_screen_tui(terminal.clone(), &HashMap::new());
    tui.set_clear_on_shrink(true);
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
    });
    tui.add_child(component.clone());

    component.set_lines(vec!["Line 0", "Line 1", "Line 2", "Line 3"]);
    tui.start();
    wait_for_render(&tui);

    // Shrink to single line
    component.set_lines(vec!["Only line"]);
    tui.request_render(false);
    wait_for_render(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport
            .first()
            .is_some_and(|line| line.contains("Only line")),
        "Single line rendered"
    );
    assert!(
        viewport.get(1).is_some_and(|line| line.trim().is_empty()),
        "Line 1 should be cleared"
    );

    stop(&tui);
}

#[test]
fn handles_shrink_to_empty() {
    let terminal = VirtualTerminal::new(40, 10);
    let tui = new_main_screen_tui(terminal.clone(), &HashMap::new());
    tui.set_clear_on_shrink(true);
    let component = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
    });
    tui.add_child(component.clone());

    component.set_lines(vec!["Line 0", "Line 1", "Line 2"]);
    tui.start();
    wait_for_render(&tui);

    // Shrink to empty
    component.set_lines(vec![]);
    tui.request_render(false);
    wait_for_render(&tui);

    let viewport = terminal.get_viewport();
    // All lines should be empty
    assert!(
        viewport.first().is_some_and(|line| line.trim().is_empty()),
        "Line 0 should be cleared"
    );
    assert!(
        viewport.get(1).is_some_and(|line| line.trim().is_empty()),
        "Line 1 should be cleared"
    );

    stop(&tui);
}

// === TUI differential rendering ===

#[test]
fn tracks_cursor_correctly_when_content_shrinks_with_unchanged_remaining_lines() {
    let (tui, terminal, component) = started_component(40, 10);

    // Initial render: 5 identical lines
    component.set_lines(vec!["Line 0", "Line 1", "Line 2", "Line 3", "Line 4"]);
    tui.start();
    wait_for_render(&tui);

    // Shrink to 3 lines, all identical to before (no content changes in remaining lines)
    component.set_lines(vec!["Line 0", "Line 1", "Line 2"]);
    tui.request_render(false);
    wait_for_render(&tui);

    // Verify by doing another render with a change on line 1
    component.set_lines(vec!["Line 0", "CHANGED", "Line 2"]);
    tui.request_render(false);
    wait_for_render(&tui);

    let viewport = terminal.get_viewport();
    // Line 1 should show "CHANGED", proving cursor tracking was correct
    assert!(
        viewport.get(1).is_some_and(|line| line.contains("CHANGED")),
        "Expected \"CHANGED\" on line 1, got: {viewport:?}"
    );

    stop(&tui);
}

#[test]
fn renders_correctly_when_only_a_middle_line_changes_spinner_case() {
    let (tui, terminal, component) = started_component(40, 10);

    // Initial render
    component.set_lines(vec!["Header", "Working...", "Footer"]);
    tui.start();
    wait_for_render(&tui);

    // Simulate spinner animation - only middle line changes
    for frame in ["|", "/", "-", "\\"] {
        component.set_lines(vec!["Header", &format!("Working {frame}"), "Footer"]);
        tui.request_render(false);
        wait_for_render(&tui);

        let viewport = terminal.get_viewport();
        assert!(
            viewport.first().is_some_and(|line| line.contains("Header")),
            "Header preserved: {viewport:?}"
        );
        assert!(
            viewport
                .get(1)
                .is_some_and(|line| line.contains(&format!("Working {frame}"))),
            "Spinner updated: {viewport:?}"
        );
        assert!(
            viewport.get(2).is_some_and(|line| line.contains("Footer")),
            "Footer preserved: {viewport:?}"
        );
    }

    stop(&tui);
}

#[test]
fn resets_styles_after_each_rendered_line() {
    let (tui, terminal, component) = started_component(20, 6);

    component.set_lines(vec!["\x1b[3mItalic", "Plain"]);
    tui.start();
    wait_for_render(&tui);

    assert!(
        !terminal.is_italic(1, 0),
        "the reset after line 0 must not leak onto line 1"
    );
    stop(&tui);
}

#[test]
fn renders_correctly_when_first_line_changes_but_rest_stays_same() {
    let (tui, terminal, component) = started_component(40, 10);

    component.set_lines(vec!["Line 0", "Line 1", "Line 2", "Line 3"]);
    tui.start();
    wait_for_render(&tui);

    // Change only first line
    component.set_lines(vec!["CHANGED", "Line 1", "Line 2", "Line 3"]);
    tui.request_render(false);
    wait_for_render(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport
            .first()
            .is_some_and(|line| line.contains("CHANGED")),
        "First line changed: {viewport:?}"
    );
    assert!(
        viewport.get(1).is_some_and(|line| line.contains("Line 1")),
        "Line 1 preserved: {viewport:?}"
    );
    assert!(
        viewport.get(2).is_some_and(|line| line.contains("Line 2")),
        "Line 2 preserved: {viewport:?}"
    );
    assert!(
        viewport.get(3).is_some_and(|line| line.contains("Line 3")),
        "Line 3 preserved: {viewport:?}"
    );

    stop(&tui);
}

#[test]
fn renders_correctly_when_last_line_changes_but_rest_stays_same() {
    let (tui, terminal, component) = started_component(40, 10);

    component.set_lines(vec!["Line 0", "Line 1", "Line 2", "Line 3"]);
    tui.start();
    wait_for_render(&tui);

    // Change only last line
    component.set_lines(vec!["Line 0", "Line 1", "Line 2", "CHANGED"]);
    tui.request_render(false);
    wait_for_render(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport.first().is_some_and(|line| line.contains("Line 0")),
        "Line 0 preserved: {viewport:?}"
    );
    assert!(
        viewport.get(1).is_some_and(|line| line.contains("Line 1")),
        "Line 1 preserved: {viewport:?}"
    );
    assert!(
        viewport.get(2).is_some_and(|line| line.contains("Line 2")),
        "Line 2 preserved: {viewport:?}"
    );
    assert!(
        viewport.get(3).is_some_and(|line| line.contains("CHANGED")),
        "Last line changed: {viewport:?}"
    );

    stop(&tui);
}

#[test]
fn renders_correctly_when_multiple_non_adjacent_lines_change() {
    let (tui, terminal, component) = started_component(40, 10);

    component.set_lines(vec!["Line 0", "Line 1", "Line 2", "Line 3", "Line 4"]);
    tui.start();
    wait_for_render(&tui);

    // Change lines 1 and 3, keep 0, 2, 4 the same
    component.set_lines(vec!["Line 0", "CHANGED 1", "Line 2", "CHANGED 3", "Line 4"]);
    tui.request_render(false);
    wait_for_render(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport.first().is_some_and(|line| line.contains("Line 0")),
        "Line 0 preserved: {viewport:?}"
    );
    assert!(
        viewport
            .get(1)
            .is_some_and(|line| line.contains("CHANGED 1")),
        "Line 1 changed: {viewport:?}"
    );
    assert!(
        viewport.get(2).is_some_and(|line| line.contains("Line 2")),
        "Line 2 preserved: {viewport:?}"
    );
    assert!(
        viewport
            .get(3)
            .is_some_and(|line| line.contains("CHANGED 3")),
        "Line 3 changed: {viewport:?}"
    );
    assert!(
        viewport.get(4).is_some_and(|line| line.contains("Line 4")),
        "Line 4 preserved: {viewport:?}"
    );

    stop(&tui);
}

#[test]
fn handles_transition_from_content_to_empty_and_back_to_content() {
    let (tui, terminal, component) = started_component(40, 10);

    // Start with content
    component.set_lines(vec!["Line 0", "Line 1", "Line 2"]);
    tui.start();
    wait_for_render(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport.first().is_some_and(|line| line.contains("Line 0")),
        "Initial content rendered"
    );

    // Clear to empty
    component.set_lines(vec![]);
    tui.request_render(false);
    wait_for_render(&tui);

    // Add content back - this should work correctly even after empty state
    component.set_lines(vec!["New Line 0", "New Line 1"]);
    tui.request_render(false);
    wait_for_render(&tui);

    let viewport = terminal.get_viewport();
    assert!(
        viewport
            .first()
            .is_some_and(|line| line.contains("New Line 0")),
        "New content rendered: {viewport:?}"
    );
    assert!(
        viewport
            .get(1)
            .is_some_and(|line| line.contains("New Line 1")),
        "New content line 1: {viewport:?}"
    );

    stop(&tui);
}

#[test]
fn full_re_renders_when_deleted_lines_move_the_viewport_upward() {
    let (tui, terminal, component) = started_component(20, 5);

    let lines: Vec<String> = (0..12).map(|i| format!("Line {i}")).collect();
    let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    component.set_lines(line_refs);
    tui.start();
    wait_for_render(&tui);

    let initial_redraws = tui.full_redraws();

    let shrunk: Vec<String> = (0..7).map(|i| format!("Line {i}")).collect();
    let shrunk_refs: Vec<&str> = shrunk.iter().map(String::as_str).collect();
    component.set_lines(shrunk_refs);
    tui.request_render(false);
    wait_for_render(&tui);

    assert!(
        tui.full_redraws() > initial_redraws,
        "Shrink should trigger a full redraw"
    );
    assert_eq!(
        terminal.get_viewport(),
        vec!["Line 2", "Line 3", "Line 4", "Line 5", "Line 6"]
    );

    stop(&tui);
}

#[test]
fn appends_after_a_shrink_without_another_full_redraw_once_the_viewport_is_reset() {
    let (tui, terminal, component) = started_component(20, 5);

    let lines: Vec<String> = (0..8).map(|i| format!("Line {i}")).collect();
    let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    component.set_lines(line_refs);
    tui.start();
    wait_for_render(&tui);

    let initial_redraws = tui.full_redraws();

    component.set_lines(vec!["Line 0", "Line 1"]);
    tui.request_render(false);
    wait_for_render(&tui);

    assert!(
        tui.full_redraws() > initial_redraws,
        "Shrink should reset the viewport with a full redraw"
    );
    let redraws_after_shrink = tui.full_redraws();

    component.set_lines(vec!["Line 0", "Line 1", "Line 2"]);
    tui.request_render(false);
    wait_for_render(&tui);

    assert_eq!(
        tui.full_redraws(),
        redraws_after_shrink,
        "Append should stay on the differential path"
    );
    assert_eq!(
        terminal.get_viewport(),
        vec!["Line 0", "Line 1", "Line 2", "", ""]
    );

    stop(&tui);
}

#[test]
fn clears_stale_content_when_max_lines_rendered_was_inflated_by_a_transient_component() {
    let terminal = VirtualTerminal::new(40, 10);
    let tui = new_main_screen_tui(terminal.clone(), &HashMap::new());
    let chat = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
    });
    let editor = Rc::new(TestComponent {
        lines: RefCell::new(Vec::new()),
    });
    tui.add_child(chat.clone());
    tui.add_child(editor.clone());

    let long_chat: Vec<String> = (0..15).map(|i| format!("Chat {i}")).collect();
    let short_chat: Vec<String> = (0..12).map(|i| format!("Chat {i}")).collect();
    let editor_lines = vec!["Editor 0", "Editor 1", "Editor 2"];
    let selector_lines: Vec<String> = (0..8).map(|i| format!("Selector {i}")).collect();

    let long_chat_refs: Vec<&str> = long_chat.iter().map(String::as_str).collect();
    chat.set_lines(long_chat_refs);
    editor.set_lines(editor_lines.clone());
    tui.start();
    wait_for_render(&tui);

    let selector_refs: Vec<&str> = selector_lines.iter().map(String::as_str).collect();
    editor.set_lines(selector_refs);
    tui.request_render(false);
    wait_for_render(&tui);

    editor.set_lines(editor_lines);
    tui.request_render(false);
    wait_for_render(&tui);
    tui.request_render(false);
    wait_for_render(&tui);

    let redraws_before_switch = tui.full_redraws();
    let short_chat_refs: Vec<&str> = short_chat.iter().map(String::as_str).collect();
    chat.set_lines(short_chat_refs);
    tui.request_render(false);
    wait_for_render(&tui);

    assert!(
        tui.full_redraws() > redraws_before_switch,
        "Branch switch should trigger a full redraw"
    );

    let viewport = terminal.get_viewport();
    for (i, line) in viewport.iter().enumerate().take(10) {
        assert!(
            !line.contains("Chat 12"),
            "Stale \"Chat 12\" at viewport row {i}"
        );
        assert!(
            !line.contains("Chat 13"),
            "Stale \"Chat 13\" at viewport row {i}"
        );
        assert!(
            !line.contains("Chat 14"),
            "Stale \"Chat 14\" at viewport row {i}"
        );
    }

    assert_eq!(
        viewport,
        vec![
            "Chat 5", "Chat 6", "Chat 7", "Chat 8", "Chat 9", "Chat 10", "Chat 11", "Editor 0",
            "Editor 1", "Editor 2",
        ]
    );

    stop(&tui);
}

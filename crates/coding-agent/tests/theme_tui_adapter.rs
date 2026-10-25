//! The real-TUI adapter behind the theme controller (#187): queries pump the
//! TUI poll loop, and appearance subscriptions reach the controller's
//! listener shape.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::Duration;

use pi_coding_agent::modes::interactive::theme::theme::{
    TerminalAutoThemeDetector, TerminalBackgroundThemeDetector, TerminalTheme,
};
use pi_coding_agent::modes::interactive::theme::theme_controller::{ThemeUi, TuiThemeUi};
use pi_tui::terminal::Terminal;
use pi_tui::terminal_colors::RgbColor;
use pi_tui::tui::{Tui, TuiConfig};

/// A terminal that answers the color queries and replays scripted input on
/// the next poll.
struct QueryTerminal {
    on_input: Option<Box<dyn FnMut(String)>>,
    pending: VecDeque<String>,
    answer_queries: bool,
    written: Rc<RefCell<String>>,
}

impl QueryTerminal {
    fn new(answer_queries: bool) -> Self {
        Self {
            on_input: None,
            pending: VecDeque::new(),
            answer_queries,
            written: Rc::new(RefCell::new(String::new())),
        }
    }

    fn queue(&mut self, data: &str) {
        self.pending.push_back(data.to_string());
    }
}

impl Terminal for QueryTerminal {
    fn start(&mut self, on_input: Box<dyn FnMut(String)>, _on_resize: Box<dyn Fn()>) {
        self.on_input = Some(on_input);
    }

    fn stop(&mut self) {}

    fn drain_input(&mut self, _max_ms: u64, _idle_ms: u64) {}

    fn write(&mut self, data: &str) {
        self.written.borrow_mut().push_str(data);
        if !self.answer_queries {
            return;
        }
        if data.contains("]11;?") {
            self.queue("\x1b]11;#ffffff\x07");
        }
        if data.contains("[?996n") {
            self.queue("\x1b[?997;2n");
        }
        if data.contains("[?2031h") {
            self.queue("\x1b[?997;1n");
        }
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

    fn poll(&mut self, _timeout: Duration) {
        let pending: Vec<String> = self.pending.drain(..).collect();
        if let Some(on_input) = &mut self.on_input {
            for data in pending {
                on_input(data);
            }
        }
    }
}

fn tui(terminal: QueryTerminal) -> Rc<Tui> {
    Tui::new(TuiConfig {
        terminal: Some(Box::new(terminal)),
        ..TuiConfig::default()
    })
}

#[test]
fn queries_the_terminal_through_the_poll_loop() {
    let tui = tui(QueryTerminal::new(true));
    tui.start();
    let adapter = TuiThemeUi::new(Rc::clone(&tui));
    assert_eq!(
        adapter.query_terminal_background_color(1_000),
        Ok(Some(RgbColor {
            r: 255,
            g: 255,
            b: 255
        }))
    );
    assert_eq!(
        adapter.query_terminal_color_scheme(1_000),
        Ok(Some(TerminalTheme::Light))
    );
}

#[test]
fn times_out_when_the_terminal_does_not_answer() {
    let tui = tui(QueryTerminal::new(false));
    tui.start();
    let adapter = TuiThemeUi::new(Rc::clone(&tui));
    assert_eq!(adapter.query_terminal_background_color(10), Ok(None));
    assert_eq!(adapter.query_terminal_color_scheme(10), Ok(None));
}

#[test]
fn forwards_appearance_reports_to_the_listener() {
    let tui = tui(QueryTerminal::new(true));
    tui.start();
    let adapter = TuiThemeUi::new(Rc::clone(&tui));
    let seen = Rc::new(RefCell::new(Vec::<TerminalTheme>::new()));
    let listener = {
        let seen = Rc::clone(&seen);
        Rc::new(move |theme: TerminalTheme| seen.borrow_mut().push(theme))
    };
    let listener_id = adapter.on_terminal_color_scheme_change(listener);

    adapter.set_terminal_color_scheme_notifications(true);
    tui.poll(Duration::from_millis(1));
    assert_eq!(*seen.borrow(), vec![TerminalTheme::Dark]);

    adapter.remove_terminal_color_scheme_listener(listener_id);
    adapter.invalidate();
    adapter.request_render();
    adapter.set_terminal_color_scheme_notifications(false);
}

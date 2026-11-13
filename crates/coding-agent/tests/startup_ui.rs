//! The startup-UI suite: upstream's `startup-ui.ts` at pin `60e7e76` (#131),
//! driving the selector, first-run setup, and text prompt through a scripted
//! terminal. The predicate cases live beside the predicate in
//! `cli/startup_ui/tests.rs`.
//!
//! Every case locks the process-wide theme, keybinding, and capability state;
//! the crate's theme tests share the same globals.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::await_holding_lock,
    reason = "each case serializes the process-wide theme and keybinding state on a current-thread runtime, so the guard never crosses a thread"
)]

use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use pi_coding_agent::cli::startup_ui::{
    StartupTuiOptions, create_startup_tui_with, show_first_time_setup_with,
    show_startup_input_with, show_startup_selector_with,
};
use pi_coding_agent::modes::interactive::theme::theme::current_theme_name;
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, Settings, SettingsManager, SettingsManagerCreateOptions,
};
use pi_tui::terminal::{InputHandler, ResizeHandler, Terminal};
use pi_tui::tui::Tui;
use serde_json::json;

/// Serializes the cases: the theme, keybindings, and image capabilities are
/// process-wide.
static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A terminal that replays queued input on each poll and answers the
/// color-scheme query when configured.
struct ScriptedTerminal {
    on_input: Option<InputHandler>,
    /// Query replies, delivered on the next poll.
    replies: VecDeque<String>,
    /// Scripted keystrokes, held until the color-scheme query settles so the
    /// detection poll cannot consume them before the dialog mounts.
    inputs: VecDeque<String>,
    inputs_armed: bool,
    scheme_reply: Option<&'static str>,
}

impl ScriptedTerminal {
    fn new(scheme_reply: Option<&'static str>, inputs: &[&str]) -> Self {
        Self {
            on_input: None,
            replies: VecDeque::new(),
            inputs: inputs.iter().map(|input| (*input).to_string()).collect(),
            inputs_armed: scheme_reply.is_none(),
            scheme_reply,
        }
    }
}

impl Terminal for ScriptedTerminal {
    fn start(&mut self, on_input: InputHandler, _on_resize: ResizeHandler) {
        self.on_input = Some(on_input);
    }

    fn stop(&mut self) {}

    fn drain_input(&mut self, _max_ms: u64, _idle_ms: u64) {}

    fn write(&mut self, data: &str) {
        if data.contains("[?996n")
            && let Some(reply) = self.scheme_reply
        {
            self.replies.push_back(reply.to_string());
            self.inputs_armed = true;
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
        let mut pending: Vec<String> = self.replies.drain(..).collect();
        if self.inputs_armed {
            pending.extend(self.inputs.drain(..));
        }
        if let Some(on_input) = &mut self.on_input {
            for data in pending {
                on_input(data);
            }
        }
    }
}

fn manager(theme: &str) -> SettingsManager<InMemorySettingsStorage> {
    let mut settings = Settings::new();
    settings.insert("theme".to_string(), json!(theme));
    SettingsManager::in_memory(
        &settings,
        SettingsManagerCreateOptions {
            project_trusted: Some(true),
        },
    )
}

async fn startup_tui(
    manager: &SettingsManager<InMemorySettingsStorage>,
    terminal: ScriptedTerminal,
) -> Rc<Tui> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_string_lossy().into_owned();
    create_startup_tui_with(
        manager,
        StartupTuiOptions {
            terminal: Some(Box::new(terminal)),
            cwd: path.clone(),
            agent_dir: path,
        },
    )
    .await
}

#[tokio::test]
async fn create_resolves_the_settings_theme_and_terminal_flags() {
    let _guard = lock();
    let manager = manager("light");
    let ui = startup_tui(&manager, ScriptedTerminal::new(None, &[])).await;

    assert_eq!(current_theme_name().as_deref(), Some("light"));
    assert_eq!(ui.get_clear_on_shrink(), manager.get_clear_on_shrink());
    assert_eq!(
        ui.get_show_hardware_cursor(),
        manager.get_show_hardware_cursor()
    );
    assert_eq!(ui.terminal_rows(), 24);
}

#[tokio::test]
async fn selector_returns_the_confirmed_option() {
    let _guard = lock();
    let manager = manager("dark");
    let ui = startup_tui(&manager, ScriptedTerminal::new(None, &["\r"])).await;

    let selection = show_startup_selector_with(
        &ui,
        &manager,
        "Pick a value",
        vec![("Alpha".to_string(), 1), ("Beta".to_string(), 2)],
    );
    assert_eq!(selection, Some(1));
}

#[tokio::test]
async fn selector_cancel_returns_none() {
    let _guard = lock();
    let manager = manager("dark");
    let ui = startup_tui(&manager, ScriptedTerminal::new(None, &["\x1b"])).await;

    let selection =
        show_startup_selector_with(&ui, &manager, "Pick a value", vec![("Only".to_string(), 7)]);
    assert_eq!(selection, None);
}

#[tokio::test]
async fn input_prompt_returns_the_submitted_text() {
    let _guard = lock();
    let manager = manager("dark");
    let ui = startup_tui(&manager, ScriptedTerminal::new(None, &["hello", "\r"])).await;

    let value = show_startup_input_with(&ui, &manager, "Name", None);
    assert_eq!(value.as_deref(), Some("hello"));
}

#[tokio::test]
async fn input_prompt_cancel_returns_none() {
    let _guard = lock();
    let manager = manager("dark");
    let ui = startup_tui(&manager, ScriptedTerminal::new(None, &["\x1b"])).await;

    assert_eq!(show_startup_input_with(&ui, &manager, "Name", None), None);
}

#[tokio::test]
async fn first_time_setup_persists_the_theme_and_analytics_choice() {
    let _guard = lock();
    let mut manager = manager("dark");
    let ui = startup_tui(
        &manager,
        ScriptedTerminal::new(Some("\x1b[?997;2n"), &["\r", "\r"]),
    )
    .await;

    show_first_time_setup_with(&ui, &mut manager).await;
    assert_eq!(manager.get_theme().as_deref(), Some("light"));
    assert!(manager.get_enable_analytics());
    assert!(manager.get_tracking_id().is_some());
}

#[tokio::test]
async fn first_time_setup_cancel_keeps_the_settings() {
    let _guard = lock();
    let mut manager = manager("dark");
    let ui = startup_tui(
        &manager,
        ScriptedTerminal::new(Some("\x1b[?997;2n"), &["\x1b"]),
    )
    .await;

    show_first_time_setup_with(&ui, &mut manager).await;
    assert_eq!(manager.get_theme().as_deref(), Some("dark"));
    assert!(!manager.get_enable_analytics());
}

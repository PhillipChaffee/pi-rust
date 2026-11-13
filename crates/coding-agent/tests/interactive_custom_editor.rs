//! The custom editor suite — upstream's
//! `test/custom-editor-history-keybindings.test.ts` and the portable parts of
//! `test/status-indicator.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` — plus boundary cases.
//!
//! Restatements against upstream: the tests share one process-wide keybinding
//! registry, so every test takes a lock and reinstalls the defaults on drop
//! (vitest isolates the module graph instead). The status-border assertions
//! check the embedded label and exact width rather than one spinner frame,
//! because the port's loader animates on a worker thread.

#![expect(
    clippy::expect_used,
    reason = "tests pin outcomes; an unexpected result panics by design"
)]

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Mutex, MutexGuard, PoisonError};

use pi_coding_agent::core::keybindings::AppKeybindingsManager;
use pi_coding_agent::modes::interactive::components::custom_editor::{
    CustomEditor, CustomEditorOptions, EmbeddedStatusIndicator,
};
use pi_coding_agent::modes::interactive::theme::theme::{get_editor_theme, init_theme};
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::components::editor::EditorOptions;
use pi_tui::keybindings::{Keybindings, KeybindingsConfig, KeybindingsManager, set_keybindings};
use pi_tui::tui::{Component, Focusable, Tui, TuiConfig};
use pi_tui::utils::visible_width;
use serde_json::json;
use tempfile::TempDir;

static KEYBINDINGS_LOCK: Mutex<()> = Mutex::new(());

fn lock_keybindings() -> MutexGuard<'static, ()> {
    KEYBINDINGS_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Upstream's `afterEach` restore.
struct RestoreDefaults;

impl Drop for RestoreDefaults {
    fn drop(&mut self) {
        set_keybindings(KeybindingsManager::new(Keybindings::tui_defaults()));
    }
}

fn install_global(bindings: &[(&str, &str)]) -> RestoreDefaults {
    let mut config = KeybindingsConfig::new();
    for (action, key) in bindings {
        config = config.bind(*action, [*key]);
    }
    set_keybindings(KeybindingsManager::with_user_bindings(
        Keybindings::tui_defaults(),
        config,
    ));
    RestoreDefaults
}

fn app_manager(dir: &TempDir, bindings: &[(&str, &str)]) -> Rc<AppKeybindingsManager> {
    let mut config = serde_json::Map::new();
    for (action, key) in bindings {
        config.insert((*action).to_string(), json!(key));
    }
    std::fs::write(
        dir.path().join("keybindings.json"),
        serde_json::to_string(&config).expect("keybindings JSON"),
    )
    .expect("write keybindings.json");
    Rc::new(AppKeybindingsManager::create(dir.path()))
}

fn new_editor(dir: &TempDir, bindings: &[(&str, &str)]) -> Rc<CustomEditor> {
    init_theme(None, false);
    let tui: Rc<Tui> = Tui::new(TuiConfig::default());
    Rc::new(CustomEditor::new(
        &tui,
        get_editor_theme(),
        app_manager(dir, bindings),
    ))
}

/// The two-method status surface the sibling status-indicator unit supplies.
struct FakeIndicator;

impl EmbeddedStatusIndicator for FakeIndicator {
    fn render_in_border(&self, width: usize) -> String {
        "Working".chars().take(width).collect()
    }

    fn render_spinner_in_border(&self, width: usize) -> String {
        "⠋".chars().take(width).collect()
    }
}

fn working_indicator() -> Rc<dyn EmbeddedStatusIndicator> {
    Rc::new(FakeIndicator)
}

#[test]
fn explicit_history_binding_beats_model_cycling() {
    let _guard = lock_keybindings();
    let _restore = install_global(&[
        ("tui.editor.historyPrevious", "ctrl+p"),
        ("tui.editor.historyNext", "ctrl+n"),
    ]);
    let dir = TempDir::new().expect("temp dir");
    let editor = new_editor(
        &dir,
        &[
            ("tui.editor.historyPrevious", "ctrl+p"),
            ("tui.editor.historyNext", "ctrl+n"),
        ],
    );
    let model_cycles = Rc::new(Cell::new(0));
    {
        let model_cycles = Rc::clone(&model_cycles);
        editor.on_action(
            "app.model.cycleForward",
            Rc::new(move || model_cycles.set(model_cycles.get() + 1)),
        );
    }
    editor.add_to_history("previous prompt");
    editor.set_text("draft");

    editor.handle_input("\x10"); // Ctrl+P
    assert_eq!(editor.get_text(), "previous prompt");
    assert_eq!(model_cycles.get(), 0);

    editor.handle_input("\x0e"); // Ctrl+N
    assert_eq!(editor.get_text(), "draft");
    assert_eq!(model_cycles.get(), 0);
}

#[test]
fn registered_app_action_dispatches() {
    let _guard = lock_keybindings();
    let _restore = install_global(&[]);
    let dir = TempDir::new().expect("temp dir");
    let editor = new_editor(&dir, &[("app.model.cycleForward", "ctrl+p")]);
    let cycles = Rc::new(Cell::new(0));
    {
        let cycles = Rc::clone(&cycles);
        editor.on_action(
            "app.model.cycleForward",
            Rc::new(move || cycles.set(cycles.get() + 1)),
        );
    }

    editor.handle_input("\x10");
    assert_eq!(cycles.get(), 1);
    assert!(editor.get_text().is_empty());
}

#[test]
fn extension_shortcuts_precede_app_actions() {
    let _guard = lock_keybindings();
    let _restore = install_global(&[]);
    let dir = TempDir::new().expect("temp dir");
    let editor = new_editor(&dir, &[("app.model.cycleForward", "ctrl+p")]);
    let cycles = Rc::new(Cell::new(0));
    {
        let cycles = Rc::clone(&cycles);
        editor.on_action(
            "app.model.cycleForward",
            Rc::new(move || cycles.set(cycles.get() + 1)),
        );
    }
    let seen = Rc::new(RefCell::new(Vec::new()));
    {
        let seen = Rc::clone(&seen);
        editor.set_on_extension_shortcut(Some(Rc::new(move |data: &str| {
            seen.borrow_mut().push(data.to_string());
            true
        })));
    }

    editor.handle_input("\x10");
    assert_eq!(seen.borrow().as_slice(), ["\x10"]);
    assert_eq!(cycles.get(), 0);
}

#[test]
fn interrupt_and_exit_hooks_fire_only_on_their_conditions() {
    let _guard = lock_keybindings();
    let _restore = install_global(&[]);
    let dir = TempDir::new().expect("temp dir");
    let editor = new_editor(&dir, &[("app.interrupt", "escape"), ("app.exit", "ctrl+d")]);
    let escapes = Rc::new(Cell::new(0));
    let exits = Rc::new(Cell::new(0));
    {
        let escapes = Rc::clone(&escapes);
        editor.set_on_escape(Some(Rc::new(move || escapes.set(escapes.get() + 1))));
    }
    {
        let exits = Rc::clone(&exits);
        editor.set_on_ctrl_d(Some(Rc::new(move || exits.set(exits.get() + 1))));
    }

    editor.handle_input("\x1b");
    assert_eq!(escapes.get(), 1);

    editor.handle_input("\x04"); // Ctrl+D on an empty editor
    assert_eq!(exits.get(), 1);

    editor.set_text("text");
    editor.handle_input("\x04"); // Ctrl+D deletes forward instead
    assert_eq!(exits.get(), 1);
}

#[test]
fn paste_image_binding_runs_its_handler() {
    let _guard = lock_keybindings();
    let _restore = install_global(&[]);
    let dir = TempDir::new().expect("temp dir");
    let editor = new_editor(&dir, &[("app.clipboard.pasteImage", "ctrl+v")]);
    let pastes = Rc::new(Cell::new(0));
    {
        let pastes = Rc::clone(&pastes);
        editor.set_on_paste_image(Some(Rc::new(move || pastes.set(pastes.get() + 1))));
    }

    editor.handle_input("\x16");
    assert_eq!(pastes.get(), 1);
    assert!(editor.get_text().is_empty());
}

#[test]
fn top_border_stays_plain_unless_embedding_is_enabled() {
    let _guard = lock_keybindings();
    let _restore = install_global(&[]);
    init_theme(None, false);
    let dir = TempDir::new().expect("temp dir");
    let tui: Rc<Tui> = Tui::new(TuiConfig::default());
    let editor = CustomEditor::new(&tui, get_editor_theme(), app_manager(&dir, &[]));
    let indicator = working_indicator();
    editor.set_working_status_indicator(Some(Rc::clone(&indicator)));

    assert_eq!(strip_ansi(&editor.render(20)[0]), "─".repeat(20));
}

#[test]
fn embedded_status_fills_the_top_border_at_every_width() {
    let _guard = lock_keybindings();
    let _restore = install_global(&[]);
    init_theme(None, false);
    let dir = TempDir::new().expect("temp dir");
    let tui: Rc<Tui> = Tui::new(TuiConfig::default());
    let editor = CustomEditor::with_options(
        &tui,
        get_editor_theme(),
        app_manager(&dir, &[]),
        CustomEditorOptions {
            editor: EditorOptions::default(),
            embed_working_status: true,
        },
    );
    assert!(editor.embed_working_status);
    let indicator = working_indicator();
    editor.set_working_status_indicator(Some(Rc::clone(&indicator)));

    let top = strip_ansi(&editor.render(20)[0]);
    assert!(top.starts_with("── "), "top: {top:?}");
    assert!(top.contains("Working"), "top: {top:?}");
    assert_eq!(visible_width(&editor.render(20)[0]), 20);

    for width in [1, 4, 10, 20, 80, 120] {
        let line = &editor.render(width)[0];
        assert_eq!(visible_width(line), width, "width {width}");
    }

    editor.set_working_status_indicator(None);
    assert_eq!(strip_ansi(&editor.render(120)[0]), "─".repeat(120));
}

#[test]
fn a_custom_border_color_reaches_both_borders() {
    let _guard = lock_keybindings();
    let _restore = install_global(&[]);
    let dir = TempDir::new().expect("temp dir");
    let editor = new_editor(&dir, &[]);
    editor.set_border_color(Rc::new(|text: &str| format!("<{text}>")));

    let lines = editor.render(12);
    assert!(lines[0].starts_with('<') && lines[0].ends_with('>'));
    assert!(lines.last().expect("bottom border").starts_with('<'));
}

#[test]
fn focus_reaches_the_wrapped_editor() {
    let _guard = lock_keybindings();
    let _restore = install_global(&[]);
    let dir = TempDir::new().expect("temp dir");
    let editor = new_editor(&dir, &[]);

    editor.set_focused(true);
    assert!(editor.is_focused());
    assert!(editor.editor().is_focused());
    editor.set_focused(false);
    assert!(!editor.editor().is_focused());
}

#[test]
fn embedded_status_keeps_the_scroll_indicator_label() {
    let _guard = lock_keybindings();
    let _restore = install_global(&[]);
    init_theme(None, false);
    let dir = TempDir::new().expect("temp dir");
    let tui: Rc<Tui> = Tui::new(TuiConfig::default());
    let editor = CustomEditor::with_options(
        &tui,
        get_editor_theme(),
        app_manager(&dir, &[]),
        CustomEditorOptions {
            editor: EditorOptions::default(),
            embed_working_status: true,
        },
    );
    editor.set_text(
        &(0..40)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    editor.set_working_status_indicator(Some(working_indicator()));

    let top = strip_ansi(&editor.render(60)[0]);
    assert!(top.contains("↑"), "top: {top:?}");
    assert!(top.contains("more"), "top: {top:?}");
    assert_eq!(visible_width(&editor.render(60)[0]), 60);
}

//! Port of `test/theme-controller.test.ts` at pin `60e7e76`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(
    clippy::type_complexity,
    reason = "the fake UI mirrors the trait surface it stands in for"
)]
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use pi_coding_agent::modes::interactive::theme::theme::{
    TerminalAutoThemeDetector, TerminalBackgroundThemeDetector, TerminalTheme,
    set_theme_dirs_override, theme,
};
use pi_coding_agent::modes::interactive::theme::theme_controller::{
    InteractiveThemeController, ThemeControllerOptions, ThemeUi,
};
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, Settings, SettingsManager, SettingsManagerCreateOptions,
};
use pi_tui::terminal_colors::RgbColor;
use pi_tui::tui::ListenerId;

type Manager = Arc<Mutex<SettingsManager<InMemorySettingsStorage>>>;

struct FakeUi {
    invalidated: Cell<u32>,
    renders: Cell<u32>,
    notifications: RefCell<Vec<bool>>,
    listeners: RefCell<Vec<(ListenerId, Rc<dyn Fn(TerminalTheme)>)>>,
    next_listener: Cell<ListenerId>,
    removed: RefCell<Vec<ListenerId>>,
    background: RefCell<Result<Option<RgbColor>, String>>,
    scheme: RefCell<Result<Option<TerminalTheme>, String>>,
    background_queries: Cell<u32>,
    scheme_queries: Cell<u32>,
}

impl FakeUi {
    fn new() -> Rc<Self> {
        Rc::new(Self {
            invalidated: Cell::new(0),
            renders: Cell::new(0),
            notifications: RefCell::new(Vec::new()),
            listeners: RefCell::new(Vec::new()),
            next_listener: Cell::new(1),
            removed: RefCell::new(Vec::new()),
            background: RefCell::new(Ok(None)),
            scheme: RefCell::new(Ok(None)),
            background_queries: Cell::new(0),
            scheme_queries: Cell::new(0),
        })
    }

    fn emit(&self, terminal_theme: TerminalTheme) {
        let listeners = self.listeners.borrow().clone();
        for (_, listener) in listeners {
            listener(terminal_theme);
        }
    }
}

impl TerminalBackgroundThemeDetector for FakeUi {
    fn query_terminal_background_color(
        &self,
        _timeout_ms: u64,
    ) -> Result<Option<RgbColor>, String> {
        self.background_queries
            .set(self.background_queries.get() + 1);
        self.background.borrow().clone()
    }
}

impl TerminalAutoThemeDetector for FakeUi {
    fn query_terminal_color_scheme(
        &self,
        _timeout_ms: u64,
    ) -> Result<Option<TerminalTheme>, String> {
        self.scheme_queries.set(self.scheme_queries.get() + 1);
        self.scheme.borrow().clone()
    }
}

impl ThemeUi for FakeUi {
    fn invalidate(&self) {
        self.invalidated.set(self.invalidated.get() + 1);
    }

    fn request_render(&self) {
        self.renders.set(self.renders.get() + 1);
    }

    fn set_terminal_color_scheme_notifications(&self, enabled: bool) {
        self.notifications.borrow_mut().push(enabled);
    }

    fn on_terminal_color_scheme_change(&self, listener: Rc<dyn Fn(TerminalTheme)>) -> ListenerId {
        let id = self.next_listener.get();
        self.next_listener.set(id + 1);
        self.listeners.borrow_mut().push((id, listener));
        id
    }

    fn remove_terminal_color_scheme_listener(&self, listener_id: ListenerId) {
        self.removed.borrow_mut().push(listener_id);
        self.listeners
            .borrow_mut()
            .retain(|(id, _)| *id != listener_id);
    }
}

fn settings_with(theme_setting: Option<&str>) -> Settings {
    let mut settings = Settings::new();
    if let Some(theme_setting) = theme_setting {
        settings.insert(
            "theme".to_string(),
            serde_json::Value::String(theme_setting.to_string()),
        );
    }
    settings
}

fn manager(theme_setting: Option<&str>) -> Manager {
    Arc::new(Mutex::new(SettingsManager::in_memory(
        &settings_with(theme_setting),
        SettingsManagerCreateOptions::default(),
    )))
}

struct Harness {
    ui: Rc<FakeUi>,
    manager: Rc<RefCell<Manager>>,
    errors: Rc<RefCell<Vec<String>>>,
    changes: Rc<Cell<u32>>,
    _root: tempfile::TempDir,
}

fn harness() -> Harness {
    let root = tempfile::TempDir::new().expect("temp");
    let builtin = root.path().join("builtin");
    let custom = root.path().join("custom");
    std::fs::create_dir_all(&builtin).expect("builtin");
    std::fs::create_dir_all(&custom).expect("custom");
    set_theme_dirs_override(Some((builtin, custom)));
    Harness {
        ui: FakeUi::new(),
        manager: Rc::new(RefCell::new(manager(None))),
        errors: Rc::new(RefCell::new(Vec::new())),
        changes: Rc::new(Cell::new(0)),
        _root: root,
    }
}

impl Harness {
    fn controller(
        &self,
        initial_theme_setting: Option<&str>,
    ) -> InteractiveThemeController<InMemorySettingsStorage> {
        let manager = Rc::clone(&self.manager);
        let errors = Rc::clone(&self.errors);
        let changes = Rc::clone(&self.changes);
        let ui: Rc<dyn ThemeUi> = self.ui.clone();
        InteractiveThemeController::new(
            ui,
            ThemeControllerOptions {
                get_settings_manager: Rc::new(move || {
                    let manager: Manager = manager.borrow().clone();
                    manager
                }),
                show_error: Rc::new(move |message: &str| {
                    errors.borrow_mut().push(message.to_string());
                }),
                on_changed: Rc::new(move || changes.set(changes.get() + 1)),
                initial_theme_setting: initial_theme_setting.map(str::to_string),
            },
        )
    }

    fn set_theme_setting(&self, theme_setting: Option<&str>) {
        let next = manager(theme_setting);
        *self.manager.borrow_mut() = next;
    }
}

fn manager_theme(manager: &Manager) -> Option<String> {
    manager.lock().expect("manager").get_theme_setting()
}

#[test]
fn uses_the_initial_theme_without_persisting_it() {
    let harness = harness();
    *harness.manager.borrow_mut() = manager(Some("dark"));
    let controller = harness.controller(Some("light"));
    assert_eq!(theme().name(), Some("light"));
    assert_eq!(controller.get_theme_selection(), Some("light".to_string()));
    controller.apply_from_settings();
    assert_eq!(harness.ui.background_queries.get(), 0);
    assert_eq!(harness.ui.scheme_queries.get(), 0);
    assert_eq!(
        manager_theme(&harness.manager.borrow()),
        Some("dark".to_string())
    );
}

#[test]
fn resolves_a_theme_pair_and_follows_terminal_changes() {
    let harness = harness();
    *harness.manager.borrow_mut() = manager(Some("dark/light"));
    *harness.ui.scheme.borrow_mut() = Ok(Some(TerminalTheme::Light));
    let controller = harness.controller(Some("light/dark"));
    controller.apply_from_settings();
    assert_eq!(theme().name(), Some("light"));
    assert_eq!(
        harness.ui.notifications.borrow().last().copied(),
        Some(true)
    );
    harness.ui.emit(TerminalTheme::Dark);
    assert_eq!(theme().name(), Some("dark"));
}

#[test]
fn disposes_of_terminal_appearance_updates() {
    let harness = harness();
    *harness.manager.borrow_mut() = manager(Some("light/dark"));
    *harness.ui.scheme.borrow_mut() = Ok(Some(TerminalTheme::Light));
    let controller = harness.controller(None);
    controller.apply_from_settings();
    controller.dispose();
    assert_eq!(
        harness.ui.notifications.borrow().last().copied(),
        Some(false)
    );
    assert_eq!(harness.ui.removed.borrow().len(), 1);
    assert!(harness.ui.listeners.borrow().is_empty());
}

#[test]
fn detects_the_current_appearance_when_selecting_a_pair() {
    let harness = harness();
    *harness.manager.borrow_mut() = manager(Some("dark"));
    *harness.ui.scheme.borrow_mut() = Ok(Some(TerminalTheme::Light));
    let controller = harness.controller(None);
    controller.set_theme_setting("light/dark");
    assert_eq!(theme().name(), Some("light"));
    assert_eq!(harness.ui.scheme_queries.get(), 1);
}

#[test]
fn explicit_selections_keep_precedence_over_settings() {
    let harness = harness();
    *harness.manager.borrow_mut() = manager(Some("dark"));
    let controller = harness.controller(Some("light"));
    controller.apply_from_settings();
    assert!(controller.set_theme_name("dark", false).success);
    harness.set_theme_setting(Some("light"));
    controller.apply_from_settings();
    assert_eq!(controller.get_theme_selection(), Some("dark".to_string()));
    assert_eq!(theme().name(), Some("dark"));
}

#[test]
fn reloads_settings_when_no_initial_theme_was_supplied() {
    let harness = harness();
    *harness.manager.borrow_mut() = manager(Some("dark"));
    let controller = harness.controller(None);
    controller.apply_from_settings();
    assert_eq!(theme().name(), Some("dark"));
    harness
        .manager
        .borrow()
        .lock()
        .expect("manager")
        .apply_overrides(&settings_with(Some("light")));
    controller.apply_from_settings();
    assert_eq!(theme().name(), Some("light"));
    harness.set_theme_setting(Some("dark"));
    controller.apply_from_settings();
    assert_eq!(theme().name(), Some("dark"));
}

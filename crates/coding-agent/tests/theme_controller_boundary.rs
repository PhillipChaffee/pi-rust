//! Boundary cases for the interactive theme controller (#187): previews,
//! persistence, re-binding, and the failure paths.

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
    ColorMode, TerminalAutoThemeDetector, TerminalBackgroundThemeDetector, TerminalTheme, Theme,
    ThemeOptions, set_theme_dirs_override, theme,
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
        self.background.borrow().clone()
    }
}

impl TerminalAutoThemeDetector for FakeUi {
    fn query_terminal_color_scheme(
        &self,
        _timeout_ms: u64,
    ) -> Result<Option<TerminalTheme>, String> {
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
    manager: Manager,
    errors: Rc<RefCell<Vec<String>>>,
    changes: Rc<Cell<u32>>,
    _root: tempfile::TempDir,
}

fn harness(theme_setting: Option<&str>) -> Harness {
    let root = tempfile::TempDir::new().expect("temp");
    let builtin = root.path().join("builtin");
    let custom = root.path().join("custom");
    std::fs::create_dir_all(&builtin).expect("builtin");
    std::fs::create_dir_all(&custom).expect("custom");
    set_theme_dirs_override(Some((builtin, custom)));
    Harness {
        ui: FakeUi::new(),
        manager: manager(theme_setting),
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
        let manager = Arc::clone(&self.manager);
        let errors = Rc::clone(&self.errors);
        let changes = Rc::clone(&self.changes);
        let ui: Rc<dyn ThemeUi> = self.ui.clone();
        InteractiveThemeController::new(
            ui,
            ThemeControllerOptions {
                get_settings_manager: Rc::new(move || Arc::clone(&manager)),
                show_error: Rc::new(move |message: &str| {
                    errors.borrow_mut().push(message.to_string());
                }),
                on_changed: Rc::new(move || changes.set(changes.get() + 1)),
                initial_theme_setting: initial_theme_setting.map(str::to_string),
            },
        )
    }
}

fn manager_theme(manager: &Manager) -> Option<String> {
    manager.lock().expect("manager").get_theme_setting()
}

#[test]
fn reports_a_failed_explicit_switch() {
    let harness = harness(None);
    let controller = harness.controller(None);
    let result = controller.set_theme_name("missing-theme", true);
    assert!(!result.success);
    assert_eq!(theme().name(), Some("dark"));
    let errors = harness.errors.borrow();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("Failed to load theme \"missing-theme\""));
    assert!(errors[0].contains("Fell back to dark theme."));
}

#[test]
fn keeps_failed_switches_silent_when_asked() {
    let harness = harness(None);
    let controller = harness.controller(None);
    let result = controller.set_theme_name("missing-theme", false);
    assert!(!result.success);
    assert!(harness.errors.borrow().is_empty());
}

#[test]
fn persists_a_confidently_detected_appearance() {
    let harness = harness(None);
    *harness.ui.background.borrow_mut() = Ok(Some(RgbColor {
        r: 250,
        g: 250,
        b: 250,
    }));
    let controller = harness.controller(None);
    controller.apply_from_settings();
    assert_eq!(theme().name(), Some("light"));
    assert_eq!(manager_theme(&harness.manager), Some("light".to_string()));
}

#[test]
fn does_not_persist_the_low_confidence_fallback() {
    let harness = harness(None);
    let controller = harness.controller(None);
    controller.apply_from_settings();
    assert!(matches!(theme().name(), Some("dark" | "light")));
    if std::env::var_os("COLORFGBG").is_none() {
        assert_eq!(manager_theme(&harness.manager), None);
    }
}

#[test]
fn previews_valid_themes_and_ignores_unknown_ones() {
    let harness = harness(None);
    let controller = harness.controller(None);
    controller.preview("light");
    assert_eq!(theme().name(), Some("light"));
    assert!(harness.ui.invalidated.get() > 0);
    assert!(harness.ui.renders.get() > 0);

    let renders_before = harness.ui.renders.get();
    controller.preview("missing-theme");
    assert_eq!(harness.ui.renders.get(), renders_before);
}

#[test]
fn installs_in_memory_themes_and_reports_the_appearance() {
    let harness = harness(None);
    let controller = harness.controller(None);
    let instance = Theme::new(
        indexmap::IndexMap::new(),
        indexmap::IndexMap::new(),
        ColorMode::Truecolor,
        ThemeOptions {
            name: Some("memory".to_string()),
            source_path: None,
            source_info: None,
        },
    );
    assert!(controller.set_theme_instance(Arc::new(instance)).success);
    assert_eq!(
        controller.get_theme_selection(),
        Some("<in-memory>".to_string())
    );
    assert!(matches!(
        controller.get_terminal_theme(),
        TerminalTheme::Dark | TerminalTheme::Light
    ));
    controller.disable_auto_sync();
    assert!(!harness.ui.notifications.borrow().contains(&true));
}

#[test]
fn rebinds_the_listener_to_a_new_tui() {
    let harness = harness(None);
    let controller = harness.controller(None);
    let replacement = FakeUi::new();
    let replacement_ui: Rc<dyn ThemeUi> = replacement.clone();
    controller.rebind_tui(replacement_ui);

    assert_eq!(harness.ui.removed.borrow().len(), 1);
    assert_eq!(replacement.listeners.borrow().len(), 1);
    assert!(harness.ui.listeners.borrow().is_empty());

    *replacement.scheme.borrow_mut() = Ok(Some(TerminalTheme::Light));
    controller.set_theme_setting("light/dark");
    replacement.emit(TerminalTheme::Dark);
    assert_eq!(theme().name(), Some("dark"));
    assert!(harness.changes.get() > 0);
}

#[test]
fn reports_a_malformed_setting_as_written() {
    let harness = harness(Some("light/dark/extra"));
    let controller = harness.controller(None);
    assert_eq!(
        controller.get_theme_selection(),
        Some("light/dark/extra".to_string())
    );
}

#[test]
fn ignores_reports_while_auto_sync_is_off() {
    let harness = harness(None);
    let controller = harness.controller(Some("light"));
    harness.ui.emit(TerminalTheme::Dark);
    assert_eq!(theme().name(), Some("light"));
    assert!(
        !harness.ui.notifications.borrow().contains(&false)
            || controller.get_theme_selection().is_some()
    );
}

#[test]
fn stops_auto_sync_when_the_setting_stops_being_a_pair() {
    let harness = harness(None);
    harness
        .manager
        .lock()
        .expect("manager")
        .apply_overrides(&settings_with(Some("light/dark")));
    let controller = harness.controller(None);
    controller.apply_from_settings();
    assert_eq!(
        harness.ui.notifications.borrow().last().copied(),
        Some(true)
    );

    harness
        .manager
        .lock()
        .expect("manager")
        .apply_overrides(&settings_with(Some("light/dark/extra")));
    harness.ui.emit(TerminalTheme::Dark);
    assert_eq!(
        harness.ui.notifications.borrow().last().copied(),
        Some(false)
    );
}

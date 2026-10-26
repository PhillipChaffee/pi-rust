//! Interactive theme controller, upstream's
//! `src/modes/interactive/theme/theme-controller.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The controller owns the theme setting, follows automatic light/dark
//! pairs as the terminal reports appearance changes, and persists detected
//! appearances. Upstream binds a closure to the TUI and awaits the query
//! promises; the port's queries block, so every method is synchronous and
//! [`ThemeUi`] is the TUI surface it drives.
//!
//! `flush()` is omitted at the persistence points: settings writes are
//! synchronous in the port, so the await upstream keeps has nothing to wait
//! for.

use std::cell::RefCell;
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pi_tui::terminal_colors::{RgbColor, TerminalColorScheme};
use pi_tui::tui::{Component, ListenerId, Tui};

use super::theme::{
    TerminalAutoThemeDetector, TerminalBackgroundThemeDetector, TerminalTheme, Theme, ThemeResult,
    detect_terminal_background_from_env, detect_terminal_background_theme,
    detect_terminal_theme_for_auto, init_theme, parse_auto_theme_setting, resolve_theme_setting,
    set_theme,
};
use crate::settings_manager::{SettingsManager, SettingsStorage};

/// The TUI surface the controller drives, upstream's `TUI` argument.
///
/// The supertrait carries the terminal-appearance queries the controller
/// awaits; [`TuiThemeUi`] adapts the real TUI.
pub trait ThemeUi: TerminalAutoThemeDetector {
    /// Invalidate the mounted components, upstream `ui.invalidate()`.
    fn invalidate(&self);
    /// Schedule a render, upstream `ui.requestRender()`.
    fn request_render(&self);
    /// Toggle `CSI ? 2031` appearance notifications.
    fn set_terminal_color_scheme_notifications(&self, enabled: bool);
    /// Subscribe to appearance reports.
    fn on_terminal_color_scheme_change(&self, listener: Rc<dyn Fn(TerminalTheme)>) -> ListenerId;
    /// Drop an appearance subscription.
    fn remove_terminal_color_scheme_listener(&self, listener_id: ListenerId);
}

/// The real TUI adapter: queries pump the TUI's own poll loop so the
/// terminal replies are consumed on the calling thread.
#[derive(Clone)]
pub struct TuiThemeUi {
    tui: Rc<Tui>,
}

impl std::fmt::Debug for TuiThemeUi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TuiThemeUi").finish_non_exhaustive()
    }
}

impl TuiThemeUi {
    /// Wrap the interactive TUI.
    #[must_use]
    pub const fn new(tui: Rc<Tui>) -> Self {
        Self { tui }
    }
}

/// Wait for a TUI query reply, pumping the render/input loop until it
/// arrives or the deadline passes.
fn wait_for_query<T>(
    tui: &Tui,
    receiver: &std::sync::mpsc::Receiver<T>,
    timeout_ms: u64,
) -> Option<T> {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms.saturating_add(100));
    loop {
        match receiver.try_recv() {
            Ok(value) => return Some(value),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => return None,
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
        if Instant::now() >= deadline {
            return None;
        }
        tui.poll(Duration::from_millis(5));
    }
}

impl TerminalBackgroundThemeDetector for TuiThemeUi {
    fn query_terminal_background_color(&self, timeout_ms: u64) -> Result<Option<RgbColor>, String> {
        let receiver = self.tui.query_terminal_background_color(timeout_ms);
        Ok(wait_for_query(&self.tui, &receiver, timeout_ms).flatten())
    }
}

impl TerminalAutoThemeDetector for TuiThemeUi {
    fn query_terminal_color_scheme(
        &self,
        timeout_ms: u64,
    ) -> Result<Option<TerminalTheme>, String> {
        let receiver = self.tui.query_terminal_color_scheme(timeout_ms);
        Ok(wait_for_query(&self.tui, &receiver, timeout_ms)
            .flatten()
            .map(|scheme| match scheme {
                TerminalColorScheme::Dark => TerminalTheme::Dark,
                TerminalColorScheme::Light => TerminalTheme::Light,
            }))
    }
}

impl ThemeUi for TuiThemeUi {
    fn invalidate(&self) {
        self.tui.invalidate();
    }

    fn request_render(&self) {
        self.tui.request_render(false);
    }

    fn set_terminal_color_scheme_notifications(&self, enabled: bool) {
        self.tui.set_terminal_color_scheme_notifications(enabled);
    }

    fn on_terminal_color_scheme_change(&self, listener: Rc<dyn Fn(TerminalTheme)>) -> ListenerId {
        self.tui
            .on_terminal_color_scheme_change(Rc::new(move |scheme: &TerminalColorScheme| {
                listener(match scheme {
                    TerminalColorScheme::Dark => TerminalTheme::Dark,
                    TerminalColorScheme::Light => TerminalTheme::Light,
                });
            }))
    }

    fn remove_terminal_color_scheme_listener(&self, listener_id: ListenerId) {
        self.tui.remove_terminal_color_scheme_listener(listener_id);
    }
}

/// Construction options, upstream's `InteractiveThemeController` options.
pub struct ThemeControllerOptions<S: SettingsStorage> {
    /// The current settings manager; re-read on every use, upstream's
    /// `getSettingsManager`.
    pub get_settings_manager: Rc<dyn Fn() -> Arc<Mutex<SettingsManager<S>>>>,
    /// The error reporter, upstream `showError`.
    pub show_error: Rc<dyn Fn(&str)>,
    /// The change notifier, upstream `onChanged`.
    pub on_changed: Rc<dyn Fn()>,
    /// A theme setting supplied by the caller, upstream
    /// `initialThemeSetting`.
    pub initial_theme_setting: Option<String>,
}

impl<S: SettingsStorage> std::fmt::Debug for ThemeControllerOptions<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThemeControllerOptions")
            .field("initial_theme_setting", &self.initial_theme_setting)
            .finish_non_exhaustive()
    }
}

/// The controller's mutable state; `Rc<RefCell<..>>` so the TUI listener
/// can reach it without a self-referential struct.
struct ControllerState<S: SettingsStorage> {
    ui: Rc<dyn ThemeUi>,
    get_settings_manager: Rc<dyn Fn() -> Arc<Mutex<SettingsManager<S>>>>,
    show_error: Rc<dyn Fn(&str)>,
    on_changed: Rc<dyn Fn()>,
    current_theme_setting: Option<String>,
    terminal_theme: TerminalTheme,
    active_theme_name: Option<String>,
    auto_sync_enabled: bool,
    terminal_color_scheme_listener: Option<ListenerId>,
}

/// The interactive theme controller, upstream's `InteractiveThemeController`.
pub struct InteractiveThemeController<S: SettingsStorage> {
    state: Rc<RefCell<ControllerState<S>>>,
}

impl<S: SettingsStorage> std::fmt::Debug for InteractiveThemeController<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractiveThemeController")
            .finish_non_exhaustive()
    }
}

impl<S: SettingsStorage + 'static> InteractiveThemeController<S> {
    /// Build the controller and install the resolved theme, upstream's
    /// constructor.
    #[must_use]
    pub fn new(ui: Rc<dyn ThemeUi>, options: ThemeControllerOptions<S>) -> Self {
        let terminal_theme = detect_terminal_background_from_env().theme;
        let initial_setting = options.initial_theme_setting.clone();
        let settings_setting = {
            let settings = (options.get_settings_manager)();
            let guard = settings
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.get_theme_setting()
        };
        let active_theme_name = resolve_theme_setting(
            initial_setting.as_deref().or(settings_setting.as_deref()),
            terminal_theme,
        );
        init_theme(active_theme_name.as_deref(), true);
        let state = Rc::new(RefCell::new(ControllerState {
            ui,
            get_settings_manager: options.get_settings_manager,
            show_error: options.show_error,
            on_changed: options.on_changed,
            current_theme_setting: options.initial_theme_setting,
            terminal_theme,
            active_theme_name,
            auto_sync_enabled: false,
            terminal_color_scheme_listener: None,
        }));
        bind_terminal_color_scheme_listener(&state);
        Self { state }
    }
}

/// Register the appearance listener against the current TUI.
fn bind_terminal_color_scheme_listener<S: SettingsStorage + 'static>(
    state: &Rc<RefCell<ControllerState<S>>>,
) {
    let weak: Weak<RefCell<ControllerState<S>>> = Rc::downgrade(state);
    let listener = Rc::new(move |terminal_theme: TerminalTheme| {
        if let Some(state) = weak.upgrade() {
            apply_terminal_theme(&state, terminal_theme);
        }
    });
    let listener_id = state.borrow().ui.on_terminal_color_scheme_change(listener);
    state.borrow_mut().terminal_color_scheme_listener = Some(listener_id);
}

/// The current settings manager, upstream's `getSettingsManager()`.
fn settings_handle<S: SettingsStorage>(
    state: &Rc<RefCell<ControllerState<S>>>,
) -> Arc<Mutex<SettingsManager<S>>> {
    (state.borrow().get_settings_manager)()
}

/// The theme setting in force: the caller's override, then the settings
/// file, upstream's `currentThemeSetting ?? getThemeSetting()`.
fn current_setting<S: SettingsStorage>(state: &Rc<RefCell<ControllerState<S>>>) -> Option<String> {
    let current = state.borrow().current_theme_setting.clone();
    current.or_else(|| {
        let settings = settings_handle(state);
        let guard = settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.get_theme_setting()
    })
}

/// Apply one theme name and report it, upstream's private `applyThemeName`.
fn apply_theme_name<S: SettingsStorage>(
    state: &Rc<RefCell<ControllerState<S>>>,
    theme_name: &str,
    show_error: bool,
) -> ThemeResult {
    let (ui, error_reporter, on_changed) = {
        let state = state.borrow();
        (
            Rc::clone(&state.ui),
            Rc::clone(&state.show_error),
            Rc::clone(&state.on_changed),
        )
    };
    let result = set_theme(theme_name, true);
    state.borrow_mut().active_theme_name = Some(if result.success {
        theme_name.to_string()
    } else {
        "dark".to_string()
    });
    ui.invalidate();
    on_changed();
    if !result.success
        && show_error
        && let Some(error) = &result.error
    {
        error_reporter(&format!(
            "Failed to load theme \"{theme_name}\": {error}\nFell back to dark theme."
        ));
    }
    result
}

/// Toggle the appearance notifications, upstream's `setAutoSync`.
fn set_auto_sync<S: SettingsStorage>(state: &Rc<RefCell<ControllerState<S>>>, enabled: bool) {
    let ui = {
        let mut state = state.borrow_mut();
        if state.auto_sync_enabled == enabled {
            return;
        }
        state.auto_sync_enabled = enabled;
        Rc::clone(&state.ui)
    };
    ui.set_terminal_color_scheme_notifications(enabled);
}

/// Follow a reported appearance change, upstream's `applyTerminalTheme`.
fn apply_terminal_theme<S: SettingsStorage>(
    state: &Rc<RefCell<ControllerState<S>>>,
    terminal_theme: TerminalTheme,
) {
    let auto_sync_enabled = {
        let mut state = state.borrow_mut();
        state.terminal_theme = terminal_theme;
        state.auto_sync_enabled
    };
    if !auto_sync_enabled {
        return;
    }
    let Some(auto_theme) = parse_auto_theme_setting(current_setting(state).as_deref()) else {
        set_auto_sync(state, false);
        return;
    };
    let theme_name = match terminal_theme {
        TerminalTheme::Light => auto_theme.light_theme,
        TerminalTheme::Dark => auto_theme.dark_theme,
    };
    let active = state.borrow().active_theme_name.clone();
    if active.as_deref() != Some(theme_name.as_str()) {
        apply_theme_name(state, &theme_name, false);
    }
}

impl<S: SettingsStorage + 'static> InteractiveThemeController<S> {
    /// Re-bind the TUI after a renderer swap, upstream's `rebindTui`.
    pub fn rebind_tui(&self, ui: Rc<dyn ThemeUi>) {
        let (old_ui, listener_id) = {
            let mut state = self.state.borrow_mut();
            (
                Rc::clone(&state.ui),
                state.terminal_color_scheme_listener.take(),
            )
        };
        if let Some(listener_id) = listener_id {
            old_ui.remove_terminal_color_scheme_listener(listener_id);
        }
        self.state.borrow_mut().ui = ui;
        bind_terminal_color_scheme_listener(&self.state);
    }

    /// Apply the settings' theme choice, upstream's `applyFromSettings`.
    pub fn apply_from_settings(&self) {
        let setting = current_setting(&self.state);
        if let Some(auto_theme) = parse_auto_theme_setting(setting.as_deref()) {
            let terminal_theme = {
                let ui = Rc::clone(&self.state.borrow().ui);
                detect_terminal_theme_for_auto(ui.as_ref(), 100)
            };
            self.state.borrow_mut().terminal_theme = terminal_theme;
            set_auto_sync(&self.state, true);
            let theme_name = match terminal_theme {
                TerminalTheme::Light => auto_theme.light_theme,
                TerminalTheme::Dark => auto_theme.dark_theme,
            };
            apply_theme_name(&self.state, &theme_name, true);
            return;
        }
        set_auto_sync(&self.state, false);
        if let Some(setting) = setting {
            apply_theme_name(&self.state, &setting, true);
            return;
        }
        let detection = {
            let ui = Rc::clone(&self.state.borrow().ui);
            detect_terminal_background_theme(ui.as_ref(), 100)
        };
        self.state.borrow_mut().terminal_theme = detection.theme;
        let result = apply_theme_name(&self.state, detection.theme.theme_name(), false);
        if result.success && detection.confidence == super::theme::DetectionConfidence::High {
            let settings = settings_handle(&self.state);
            settings
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .set_theme(detection.theme.theme_name());
        }
    }

    /// The setting the selector shows, upstream's `getThemeSelection`.
    #[must_use]
    pub fn get_theme_selection(&self) -> Option<String> {
        current_setting(&self.state).or_else(|| self.state.borrow().active_theme_name.clone())
    }

    /// Switch to a named theme, upstream's `setThemeName`; a successful
    /// switch replaces the supplied setting.
    #[must_use]
    pub fn set_theme_name(&self, theme_name: &str, show_error: bool) -> ThemeResult {
        set_auto_sync(&self.state, false);
        let result = apply_theme_name(&self.state, theme_name, show_error);
        if result.success {
            self.state.borrow_mut().current_theme_setting = Some(theme_name.to_string());
        }
        result
    }

    /// Apply a new setting value, upstream's `setThemeSetting`.
    pub fn set_theme_setting(&self, theme_setting: &str) {
        self.state.borrow_mut().current_theme_setting = Some(theme_setting.to_string());
        self.apply_from_settings();
    }

    /// Install a constructed theme, upstream's `setThemeInstance`.
    #[must_use]
    pub fn set_theme_instance(&self, theme_instance: Arc<Theme>) -> ThemeResult {
        set_auto_sync(&self.state, false);
        super::theme::set_theme_instance(theme_instance);
        let (ui, on_changed) = {
            let mut state = self.state.borrow_mut();
            state.active_theme_name = Some("<in-memory>".to_string());
            (Rc::clone(&state.ui), Rc::clone(&state.on_changed))
        };
        ui.invalidate();
        on_changed();
        ThemeResult::success()
    }

    /// Preview a theme without persisting it, upstream's `preview`.
    pub fn preview(&self, theme_setting_or_name: &str) {
        let terminal_theme = self.state.borrow().terminal_theme;
        let theme_name = resolve_theme_setting(Some(theme_setting_or_name), terminal_theme)
            .or_else(|| self.state.borrow().active_theme_name.clone());
        let Some(theme_name) = theme_name else {
            return;
        };
        if set_theme(&theme_name, true).success {
            let ui = Rc::clone(&self.state.borrow().ui);
            ui.invalidate();
            ui.request_render();
        }
    }

    /// Stop following terminal appearance changes, upstream's
    /// `disableAutoSync`.
    pub fn disable_auto_sync(&self) {
        set_auto_sync(&self.state, false);
    }

    /// Unsubscribe and stop following appearance changes, upstream's
    /// `dispose`.
    pub fn dispose(&self) {
        set_auto_sync(&self.state, false);
        let (ui, listener_id) = {
            let mut state = self.state.borrow_mut();
            (
                Rc::clone(&state.ui),
                state.terminal_color_scheme_listener.take(),
            )
        };
        if let Some(listener_id) = listener_id {
            ui.remove_terminal_color_scheme_listener(listener_id);
        }
    }

    /// The terminal appearance the controller last detected.
    #[must_use]
    pub fn get_terminal_theme(&self) -> TerminalTheme {
        self.state.borrow().terminal_theme
    }
}

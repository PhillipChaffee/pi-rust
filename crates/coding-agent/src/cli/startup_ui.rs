//! The startup dialogs, upstream's `src/cli/startup-ui.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! [`should_run_first_time_setup`] gates startup before any TUI exists; the
//! rest of the module builds a throwaway main-screen TUI for the prompts that
//! run ahead of the interactive frame — a generic selector, the first-run
//! setup dialog, and an extension text prompt — and tears it down when the
//! prompt settles.
//!
//! Recorded restatements:
//!
//! - `Tui` is a concrete renderer the host pumps, so each prompt runs its own
//!   poll loop to completion; upstream's promise resolved from the TUI's
//!   internal event loop.
//! - [`start_startup_tui`] runs the automatic-theme probe before it returns.
//!   Upstream fired it and let the first frame paint while the query was in
//!   flight; the blocking port has no frame to hand back to.
//! - A package-resolve failure leaves the registered theme set empty.
//!   Upstream rejected the startup TUI; a broken package source must not fail
//!   a first-run prompt.
//! - A theme whose name carries `/` is skipped while loading. The port's
//!   registry reserves that spelling for automatic light/dark settings;
//!   upstream registered the name.
//! - The global keybindings manager is rebuilt over the app registry from the
//!   app manager's loaded user bindings: pi-tui's `set_keybindings` takes
//!   ownership and `AppKeybindingsManager` has no move-out accessor.
//! - The selector's `{ label, value }` rows restate as `(label, value)` pairs.
//! - `clear_startup_tui` sleeps its 25 ms drain inline, blocking the caller;
//!   upstream awaited the same delay.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pi_tui::keybindings::{KeybindingsManager, set_keybindings};
use pi_tui::terminal::{ProcessTerminal, Terminal};
use pi_tui::terminal_image::set_capability_overrides;
use pi_tui::tui::{Component, Tui, TuiConfig, TuiStopOptions};
use pi_tui::tui_main_screen::{TuiMainScreen, TuiMainScreenConfig};

use crate::config::{
    APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR, EnvLookup, default_env_lookup, get_agent_dir,
    get_settings_path,
};
use crate::core::experimental::are_experimental_features_enabled_with;
use crate::core::keybindings::{AppKeybindingsManager, keybindings};
use crate::modes::interactive::components::extension_input::{
    ExtensionInputCancel, ExtensionInputComponent, ExtensionInputOptions, ExtensionInputSubmit,
};
use crate::modes::interactive::components::extension_selector::{
    ExtensionCancelCallback, ExtensionSelectCallback, ExtensionSelectorComponent,
    ExtensionSelectorOptions,
};
use crate::modes::interactive::components::first_time_setup::{
    FirstTimeSetupComponent, FirstTimeSetupOptions, FirstTimeSetupResult,
};
use crate::modes::interactive::theme::theme::{
    TerminalTheme, Theme, assert_theme_name_is_valid, detect_terminal_background_from_env,
    detect_terminal_theme_for_auto, init_theme, load_theme_from_path, parse_auto_theme_setting,
    resolve_theme_setting, set_registered_themes, set_theme,
};
use crate::modes::interactive::theme::theme_controller::TuiThemeUi;
use crate::package_manager::{
    DefaultPackageManager, MissingSourceAction, PackageManagerOptions, ResolveMissingCallback,
    ResolvedResource,
};
use crate::settings_manager::{
    InMemorySettingsStorage, SettingsManager, SettingsManagerCreateOptions, SettingsStorage,
};

/// Upstream's `OFFICIAL_PACKAGE_NAME`. The Rust build pins the official
/// identity; upstream's `piConfig` fork-rename mechanism has no runtime
/// counterpart (recorded in `config.rs`).
const OFFICIAL_PACKAGE_NAME: &str = "@earendil-works/pi-coding-agent";

/// Upstream's `isOfficialDistribution` metadata comparison.
fn is_official_distribution(package_name: &str, app_name: &str, config_dir_name: &str) -> bool {
    package_name == OFFICIAL_PACKAGE_NAME
        && app_name == APP_NAME
        && config_dir_name == CONFIG_DIR_NAME
}

/// Whether the first-time setup runs before the interactive session, upstream's
/// `shouldRunFirstTimeSetup`.
///
/// Setup runs only for the official distribution, with experimental features
/// enabled, without an agent-dir override, and before `settings.json` exists.
#[must_use]
pub fn should_run_first_time_setup() -> bool {
    should_run_first_time_setup_with(&default_env_lookup(), &get_settings_path())
}

/// [`should_run_first_time_setup`] over an injected environment and settings
/// path, the test seam for the override and completed-setup arms.
#[must_use]
pub fn should_run_first_time_setup_with(env: &EnvLookup, settings_path: &Path) -> bool {
    if !is_official_distribution(OFFICIAL_PACKAGE_NAME, APP_NAME, CONFIG_DIR_NAME) {
        return false;
    }
    if !are_experimental_features_enabled_with(env) {
        return false;
    }
    if env(ENV_AGENT_DIR).is_some() {
        return false;
    }
    !settings_path.exists()
}

/// The construction seam for [`create_startup_tui`], restating the process
/// reads upstream made directly: the terminal, the working directory, and the
/// agent directory.
pub struct StartupTuiOptions {
    /// The terminal; `None` builds the process terminal.
    pub terminal: Option<Box<dyn Terminal>>,
    /// The directory the package manager resolves project resources against,
    /// upstream's `process.cwd()`.
    pub cwd: String,
    /// The directory themes and keybindings load from, upstream's
    /// `getAgentDir()`.
    pub agent_dir: String,
}

impl Default for StartupTuiOptions {
    fn default() -> Self {
        Self {
            terminal: None,
            cwd: process_cwd(),
            agent_dir: get_agent_dir().to_string_lossy().into_owned(),
        }
    }
}

impl std::fmt::Debug for StartupTuiOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StartupTuiOptions")
            .field("terminal", &self.terminal.is_some())
            .field("cwd", &self.cwd)
            .field("agent_dir", &self.agent_dir)
            .finish()
    }
}

/// Upstream's `process.cwd()`.
fn process_cwd() -> String {
    std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// Build the startup TUI with the process's terminal and directories,
/// upstream's `createStartupTui`.
pub async fn create_startup_tui<S: SettingsStorage>(
    settings_manager: &SettingsManager<S>,
) -> Rc<Tui> {
    create_startup_tui_with(settings_manager, StartupTuiOptions::default()).await
}

/// [`create_startup_tui`] over the injected terminal and directories.
pub async fn create_startup_tui_with<S: SettingsStorage>(
    settings_manager: &SettingsManager<S>,
    options: StartupTuiOptions,
) -> Rc<Tui> {
    set_capability_overrides(settings_manager.get_terminal_capability_overrides());
    let themes = load_startup_themes(settings_manager, &options.cwd, &options.agent_dir).await;
    let _ = set_registered_themes(themes);

    let terminal_theme = detect_terminal_background_from_env().theme;
    let theme_name = resolve_theme_setting(
        settings_manager.get_theme_setting().as_deref(),
        terminal_theme,
    )
    .unwrap_or_else(|| terminal_theme.theme_name().to_string());
    init_theme(Some(&theme_name), false);

    let app_bindings = AppKeybindingsManager::create(Path::new(&options.agent_dir));
    set_keybindings(KeybindingsManager::with_user_bindings(
        keybindings(),
        app_bindings.get_user_bindings(),
    ));

    let ui = Tui::new(TuiConfig {
        terminal: options
            .terminal
            .or_else(|| Some(Box::new(ProcessTerminal::new()))),
        renderer: Some(Box::new(TuiMainScreen::new(TuiMainScreenConfig::default()))),
        show_hardware_cursor: Some(settings_manager.get_show_hardware_cursor()),
        log_directory: Some(PathBuf::from(&options.agent_dir)),
        ..TuiConfig::default()
    });
    ui.set_clear_on_shrink(settings_manager.get_clear_on_shrink());
    ui
}

/// Start the startup TUI, upstream's `startStartupTui`: the frame starts and
/// the automatic-theme probe runs before this returns.
pub fn start_startup_tui<S: SettingsStorage>(ui: &Rc<Tui>, settings_manager: &SettingsManager<S>) {
    ui.start();
    apply_detected_startup_theme(ui, settings_manager);
}

/// Apply the probed terminal appearance when the setting is automatic or
/// unset, upstream's `applyDetectedStartupTheme`.
fn apply_detected_startup_theme<S: SettingsStorage>(
    ui: &Rc<Tui>,
    settings_manager: &SettingsManager<S>,
) {
    let theme_setting = settings_manager.get_theme_setting();
    if let Some(setting) = theme_setting.as_deref()
        && parse_auto_theme_setting(Some(setting)).is_none()
    {
        return;
    }

    let terminal_theme = detect_terminal_theme_for_auto(&TuiThemeUi::new(Rc::clone(ui)), 100);
    let theme_name = resolve_theme_setting(theme_setting.as_deref(), terminal_theme)
        .unwrap_or_else(|| terminal_theme.theme_name().to_string());
    let _ = set_theme(&theme_name, false);
    ui.invalidate();
    ui.request_render(false);
}

/// Clear the frame and give the terminal 25 ms to drain, upstream's
/// `clearStartupTui`.
fn clear_startup_tui(ui: &Tui) {
    ui.clear();
    ui.request_render(false);
    std::thread::sleep(Duration::from_millis(25));
}

/// Load the enabled themes from resolved resources, upstream's `loadThemes`.
fn load_themes(resources: &[ResolvedResource]) -> Vec<Arc<Theme>> {
    let mut themes = Vec::new();
    let mut seen = HashSet::new();
    for resource in resources {
        if !resource.enabled {
            continue;
        }
        let Ok(theme) = load_theme_from_path(Path::new(&resource.path), None) else {
            continue;
        };
        if let Some(name) = theme.name()
            && (assert_theme_name_is_valid(name).is_err() || !seen.insert(name.to_string()))
        {
            continue;
        }
        themes.push(Arc::new(theme));
    }
    themes
}

/// Resolve the global settings' themes for the startup TUI, upstream's
/// `loadStartupThemes`.
async fn load_startup_themes<S: SettingsStorage>(
    settings_manager: &SettingsManager<S>,
    cwd: &str,
    agent_dir: &str,
) -> Vec<Arc<Theme>> {
    let global_manager = SettingsManager::in_memory(
        &settings_manager.get_global_settings(),
        SettingsManagerCreateOptions {
            project_trusted: Some(false),
        },
    );
    let package_manager: DefaultPackageManager<InMemorySettingsStorage> =
        DefaultPackageManager::new(PackageManagerOptions {
            cwd: cwd.to_string(),
            agent_dir: agent_dir.to_string(),
            settings: Arc::new(Mutex::new(global_manager)),
            command_runner: None,
            env: None,
            http_client: None,
        });
    let on_missing: ResolveMissingCallback = Box::new(|_| Ok(MissingSourceAction::Skip));
    match package_manager.resolve(Some(on_missing)).await {
        Ok(paths) => load_themes(&paths.themes),
        Err(_) => Vec::new(),
    }
}

/// The deferred result a startup prompt's callback stores, upstream's resolve
/// capture.
type StartupResultSlot<T> = Rc<RefCell<Option<T>>>;

/// The one-shot settlement callback, upstream's `finish`.
type StartupFinish<T> = Rc<dyn Fn(T)>;

/// The shared settlement: a one-shot latch and the deferred result, upstream's
/// per-prompt `finish` closure body.
fn startup_finisher<T: 'static>(ui: &Rc<Tui>) -> (StartupResultSlot<T>, StartupFinish<T>) {
    let settled = Rc::new(Cell::new(false));
    let result = Rc::new(RefCell::new(None));
    let weak = Rc::downgrade(ui);
    let finish = {
        let settled = Rc::clone(&settled);
        let result = Rc::clone(&result);
        Rc::new(move |value: T| {
            if settled.replace(true) {
                return;
            }
            if let Some(ui) = weak.upgrade() {
                clear_startup_tui(&ui);
                ui.stop(TuiStopOptions::default());
            }
            *result.borrow_mut() = Some(value);
        })
    };
    (result, finish)
}

/// Pump the startup TUI's loop until the prompt settles, upstream's promise
/// kept by the TUI's own event loop.
fn pump_startup_result<T>(ui: &Tui, result: &RefCell<Option<T>>) -> T {
    loop {
        if let Some(value) = result.borrow_mut().take() {
            return value;
        }
        ui.poll(Duration::from_millis(16));
    }
}

/// Show a string-option selector and return the chosen value, upstream's
/// `showStartupSelector`.
pub async fn show_startup_selector<S, T>(
    settings_manager: &SettingsManager<S>,
    title: &str,
    options: Vec<(String, T)>,
) -> Option<T>
where
    S: SettingsStorage,
    T: 'static,
{
    let ui = create_startup_tui(settings_manager).await;
    show_startup_selector_with(&ui, settings_manager, title, options)
}

/// [`show_startup_selector`] mounted on a caller-built TUI.
pub fn show_startup_selector_with<S, T>(
    ui: &Rc<Tui>,
    settings_manager: &SettingsManager<S>,
    title: &str,
    options: Vec<(String, T)>,
) -> Option<T>
where
    S: SettingsStorage,
    T: 'static,
{
    let (result, finish) = startup_finisher::<Option<T>>(ui);
    let labels = options.iter().map(|(label, _)| label.clone()).collect();
    let entries = Rc::new(RefCell::new(Some(options)));
    let on_select: ExtensionSelectCallback = Rc::new({
        let entries = Rc::clone(&entries);
        let finish = Rc::clone(&finish);
        move |label: &str| {
            let value = entries.borrow_mut().as_mut().and_then(|options| {
                let index = options.iter().position(|(name, _)| name == label)?;
                Some(options.remove(index).1)
            });
            finish(value);
        }
    });
    let on_cancel: ExtensionCancelCallback = Rc::new({
        let finish = Rc::clone(&finish);
        move || finish(None)
    });

    let selector = Rc::new(ExtensionSelectorComponent::new(
        title,
        labels,
        on_select,
        on_cancel,
        ExtensionSelectorOptions::default(),
    ));
    ui.add_child({
        let child: Rc<dyn Component> = selector.clone();
        child
    });
    ui.set_focus(Some({
        let focused: Rc<dyn Component> = selector;
        focused
    }));
    start_startup_tui(ui, settings_manager);

    pump_startup_result(ui, &result)
}

/// Show the first-run setup dialog and persist its decision, upstream's
/// `showFirstTimeSetup`.
pub async fn show_first_time_setup<S: SettingsStorage>(settings_manager: &mut SettingsManager<S>) {
    let ui = create_startup_tui(settings_manager).await;
    show_first_time_setup_with(&ui, settings_manager).await;
}

/// [`show_first_time_setup`] mounted on a caller-built TUI.
///
/// The decision persists after the frame tears down: upstream wrote and
/// flushed inside its async `finish`, while the port's callback surface is
/// synchronous.
pub async fn show_first_time_setup_with<S: SettingsStorage>(
    ui: &Rc<Tui>,
    settings_manager: &mut SettingsManager<S>,
) {
    let (result, finish) = startup_finisher::<Option<FirstTimeSetupResult>>(ui);

    ui.start();
    let detected_theme = detect_terminal_theme_for_auto(&TuiThemeUi::new(Rc::clone(ui)), 100);
    let _ = set_theme(detected_theme.theme_name(), false);

    let component = Rc::new(FirstTimeSetupComponent::new(FirstTimeSetupOptions {
        detected_theme,
        on_theme_preview: Rc::new({
            let ui = Rc::downgrade(ui);
            move |theme: TerminalTheme| {
                let _ = set_theme(theme.theme_name(), false);
                if let Some(ui) = ui.upgrade() {
                    ui.request_render(false);
                }
            }
        }),
        on_submit: Rc::new({
            let finish = Rc::clone(&finish);
            move |result| finish(Some(result))
        }),
        on_cancel: Rc::new(move || finish(None)),
    }));
    ui.add_child({
        let child: Rc<dyn Component> = component.clone();
        child
    });
    ui.set_focus(Some({
        let focused: Rc<dyn Component> = component;
        focused
    }));
    ui.request_render(false);

    if let Some(result) = pump_startup_result(ui, &result) {
        settings_manager.set_theme(result.theme.theme_name());
        settings_manager.set_enable_analytics(result.share_analytics);
        settings_manager.flush().await;
    }
}

/// Show a one-line text prompt and return the submitted value, upstream's
/// `showStartupInput`.
pub async fn show_startup_input<S: SettingsStorage>(
    settings_manager: &SettingsManager<S>,
    title: &str,
    placeholder: Option<&str>,
) -> Option<String> {
    let ui = create_startup_tui(settings_manager).await;
    show_startup_input_with(&ui, settings_manager, title, placeholder)
}

/// [`show_startup_input`] mounted on a caller-built TUI.
pub fn show_startup_input_with<S: SettingsStorage>(
    ui: &Rc<Tui>,
    settings_manager: &SettingsManager<S>,
    title: &str,
    placeholder: Option<&str>,
) -> Option<String> {
    let (result, finish) = startup_finisher::<Option<String>>(ui);
    let on_submit: ExtensionInputSubmit = Rc::new({
        let finish = Rc::clone(&finish);
        move |value| finish(Some(value))
    });
    let on_cancel: ExtensionInputCancel = Rc::new(move || finish(None));
    let input = Rc::new(ExtensionInputComponent::new(
        title,
        placeholder,
        on_submit,
        on_cancel,
        ExtensionInputOptions::default(),
    ));
    ui.add_child({
        let child: Rc<dyn Component> = input.clone();
        child
    });
    ui.set_focus(Some({
        let focused: Rc<dyn Component> = input.clone();
        focused
    }));
    start_startup_tui(ui, settings_manager);

    let value = pump_startup_result(ui, &result);
    input.dispose();
    value
}

#[cfg(test)]
mod tests;

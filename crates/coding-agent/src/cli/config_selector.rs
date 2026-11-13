//! The `pi config` TUI entry point, upstream's `src/cli/config-selector.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! [`select_config`] initializes the theme with its watcher, mounts a
//! [`ConfigSelectorComponent`] on a main-screen TUI, and returns when the
//! selector closes. Dismissing stops the TUI and the watcher; the exit chord
//! exits the process, as upstream does.
//!
//! Restatements: `Tui` is a concrete renderer the host pumps, not an
//! interface with its own event loop, so the port runs the poll loop to
//! completion on the calling thread where upstream returned a promise
//! resolved from inside its loop.

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use pi_tui::terminal::ProcessTerminal;
use pi_tui::tui::{Component, Tui, TuiConfig, TuiStopOptions};
use pi_tui::tui_main_screen::{TuiMainScreen, TuiMainScreenConfig};

use crate::modes::interactive::components::config_selector::{
    ConfigSelectorComponent, ConfigSelectorOptions as ComponentOptions, ConfigWriteScope,
    ScopedResolvedPaths,
};
use crate::modes::interactive::theme::theme::{init_theme, stop_theme_watcher};
use crate::settings_manager::{SettingsManager, SettingsStorage};

/// The `pi config` inputs, upstream's `ConfigSelectorOptions`.
pub struct ConfigSelectorOptions<S: SettingsStorage> {
    /// Both scopes' resolved resources, upstream's `resolvedPaths`.
    pub resolved_paths: ScopedResolvedPaths,
    /// The manager both scopes read and write, upstream's
    /// `settingsManager`.
    pub settings_manager: Arc<Mutex<SettingsManager<S>>>,
    /// The project directory the selector labels rows with, upstream `cwd`.
    pub cwd: String,
    /// The agent directory the TUI logs to, upstream `agentDir`.
    pub agent_dir: String,
    /// The scope the selector opens in, upstream's `writeScope`.
    pub write_scope: ConfigWriteScope,
    /// Whether the scope switch is offered, upstream's
    /// `projectModeAvailable`.
    pub project_mode_available: bool,
}

impl<S: SettingsStorage> std::fmt::Debug for ConfigSelectorOptions<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigSelectorOptions")
            .field("cwd", &self.cwd)
            .field("agent_dir", &self.agent_dir)
            .field("write_scope", &self.write_scope)
            .field("project_mode_available", &self.project_mode_available)
            .finish_non_exhaustive()
    }
}

/// Show the config selector and return when it closes, upstream's
/// `selectConfig`.
pub fn select_config<S: SettingsStorage + 'static>(options: ConfigSelectorOptions<S>) {
    let (theme_name, show_hardware_cursor, clear_on_shrink) = {
        let manager = options
            .settings_manager
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        (
            manager.get_theme(),
            manager.get_show_hardware_cursor(),
            manager.get_clear_on_shrink(),
        )
    };
    init_theme(theme_name.as_deref(), true);

    let ui = Tui::new(TuiConfig {
        terminal: Some(Box::new(ProcessTerminal::new())),
        renderer: Some(Box::new(TuiMainScreen::new(TuiMainScreenConfig::default()))),
        show_hardware_cursor: Some(show_hardware_cursor),
        log_directory: Some(PathBuf::from(&options.agent_dir)),
        ..TuiConfig::default()
    });
    ui.set_clear_on_shrink(clear_on_shrink);

    let closed = Rc::new(Cell::new(false));
    let weak: Weak<Tui> = Rc::downgrade(&ui);
    let close = Rc::clone(&closed);
    let on_close: Rc<dyn Fn()> = Rc::new(move || {
        if close.replace(true) {
            return;
        }
        if let Some(ui) = weak.upgrade() {
            ui.stop(TuiStopOptions::default());
        }
        stop_theme_watcher();
    });
    let weak = Rc::downgrade(&ui);
    let on_exit: Rc<dyn Fn()> = Rc::new(move || {
        if let Some(ui) = weak.upgrade() {
            ui.stop(TuiStopOptions::default());
        }
        stop_theme_watcher();
        std::process::exit(0);
    });
    let weak = Rc::downgrade(&ui);
    let request_render: Rc<dyn Fn()> = Rc::new(move || {
        if let Some(ui) = weak.upgrade() {
            ui.request_render(false);
        }
    });

    let selector = Rc::new(ConfigSelectorComponent::new(
        &options.resolved_paths,
        Arc::clone(&options.settings_manager),
        options.cwd,
        options.agent_dir,
        on_close,
        on_exit,
        request_render,
        ComponentOptions {
            terminal_height: Some(usize::from(ui.terminal_rows())),
            write_scope: options.write_scope,
            project_mode_available: options.project_mode_available,
        },
    ));
    ui.add_child({
        let child: Rc<dyn Component> = selector.clone();
        child
    });
    ui.set_focus(Some({
        let focused: Rc<dyn Component> = selector.resource_list();
        focused
    }));
    ui.start();

    while !closed.get() {
        ui.poll(Duration::from_millis(16));
    }
}

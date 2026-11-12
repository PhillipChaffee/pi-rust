//! The config selector's exit binding (#213): upstream's `onExit` branch is
//! reachable only when the cancel binding no longer claims `ctrl+c`, so this
//! suite rebinds cancel to Escape. It lives in its own test binary because
//! keybindings are process-global and the sibling suite's tests install the
//! defaults concurrently.

#[path = "common/selectors.rs"]
mod selectors;

use std::rc::Rc;
use std::sync::{Arc, Mutex};

use pi_coding_agent::modes::interactive::components::config_selector::{
    ConfigSelectorComponent, ConfigSelectorOptions, ConfigWriteScope, ScopedResolvedPaths,
};
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, SettingsManager, SettingsManagerCreateOptions,
};
use pi_tui::tui::Component;
use selectors::{cancellation, setup};

#[test]
fn ctrl_c_reaches_exit_when_cancel_is_rebound() {
    setup(&[("tui.select.cancel", &["escape"])]);
    let manager = Arc::new(Mutex::new(SettingsManager::from_storage(
        InMemorySettingsStorage::default(),
        SettingsManagerCreateOptions {
            project_trusted: Some(true),
        },
    )));
    let (exit_count, on_exit) = cancellation();
    let selector = ConfigSelectorComponent::new(
        &ScopedResolvedPaths::default(),
        manager,
        "/home/user/project".to_string(),
        "/home/user/.pi/agent".to_string(),
        Rc::new(|| {}),
        on_exit,
        Rc::new(|| {}),
        ConfigSelectorOptions {
            terminal_height: Some(24),
            write_scope: ConfigWriteScope::Global,
            project_mode_available: true,
        },
    );

    selector.resource_list().handle_input("\x03");
    assert_eq!(exit_count.get(), 1);
}

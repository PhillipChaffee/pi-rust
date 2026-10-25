//! The first-time-setup predicate, upstream's `src/cli/startup-ui.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Only the predicate lands here: `shouldRunFirstTimeSetup` gates startup
//! before any TUI exists. The module's TUI half — the startup theme load, the
//! selector/input components, and `showFirstTimeSetup` — rides the
//! interactive-shell slice (#131).

use std::path::Path;

use crate::config::{
    APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR, EnvLookup, default_env_lookup, get_settings_path,
};
use crate::core::experimental::are_experimental_features_enabled_with;

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

#[cfg(test)]
mod tests;

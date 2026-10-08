//! Install telemetry gating, upstream `src/core/telemetry.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::config::EnvLookup;
use crate::settings_manager::{SettingsManager, SettingsStorage};

/// The truthy env-flag ladder, upstream's `isTruthyEnvFlag`: `1`, `true`,
/// and `yes` in any case.
fn is_truthy_env_flag(value: Option<&str>) -> bool {
    matches!(
        value,
        Some(value) if value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
    )
}

/// Whether install telemetry sends, upstream's `isInstallTelemetryEnabled`:
/// the `PI_TELEMETRY` env override wins, else the setting.
#[must_use]
pub fn is_install_telemetry_enabled<S: SettingsStorage>(
    settings_manager: &SettingsManager<S>,
    env: &EnvLookup,
) -> bool {
    env("PI_TELEMETRY").map_or_else(
        || settings_manager.get_enable_install_telemetry(),
        |value| is_truthy_env_flag(Some(&value)),
    )
}

/// [`is_install_telemetry_enabled`] with the env override injected, the test
/// seam for upstream's `process.env` rig.
#[must_use]
pub fn is_install_telemetry_enabled_with<S: SettingsStorage>(
    settings_manager: &SettingsManager<S>,
    telemetry_env: Option<&str>,
) -> bool {
    telemetry_env.map_or_else(
        || settings_manager.get_enable_install_telemetry(),
        |value| is_truthy_env_flag(Some(value)),
    )
}

//! Settings diagnostics, upstream's
//! `packages/coding-agent/src/core/settings-diagnostics.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The drained errors map to warning diagnostics; the deduplication keeps
//! first occurrences where upstream's Set-guarded filter does. The
//! `AgentSessionRuntimeDiagnostic` type upstream imports lands with the
//! session-services slice, so the diagnostic shape this module produces —
//! the `{ type: "warning", message }` pair — rides here as
//! [`SettingsDiagnostic`] until that slice names it.

use crate::settings_manager::SettingsManager;

/// One runtime diagnostic, upstream's `AgentSessionRuntimeDiagnostic`
/// narrowed to the warning shape this module produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsDiagnostic {
    /// The diagnostic kind, upstream's `type`.
    pub diagnostic_type: String,
    /// The human-readable message, upstream's `message`.
    pub message: String,
}

/// Collect the manager's drained errors as warning diagnostics, upstream's
/// `collectSettingsDiagnostics`: a file path names the file, a pathless
/// scope names the scope.
#[must_use]
pub fn collect_settings_diagnostics<S: crate::settings_manager::SettingsStorage>(
    settings_manager: &mut SettingsManager<S>,
) -> Vec<SettingsDiagnostic> {
    settings_manager
        .drain_errors()
        .into_iter()
        .map(|error| SettingsDiagnostic {
            diagnostic_type: "warning".to_string(),
            message: match error.path {
                Some(path) => format!("Invalid settings file {path}: {}", error.message),
                None => format!(
                    "Invalid {} settings: {}",
                    scope_name(error.scope),
                    error.message
                ),
            },
        })
        .collect()
}

const fn scope_name(scope: crate::settings_manager::SettingsScope) -> &'static str {
    match scope {
        crate::settings_manager::SettingsScope::Global => "global",
        crate::settings_manager::SettingsScope::Project => "project",
    }
}

/// Remove duplicate type/message diagnostics while preserving their first
/// occurrence, upstream's `deduplicateDiagnostics`: startup and runtime
/// settings managers can report the same file error.
#[must_use]
pub fn deduplicate_diagnostics(diagnostics: Vec<SettingsDiagnostic>) -> Vec<SettingsDiagnostic> {
    let mut seen = std::collections::HashSet::new();
    diagnostics
        .into_iter()
        .filter(|diagnostic| {
            let key = format!("{}\u{0}{}", diagnostic.diagnostic_type, diagnostic.message);
            seen.insert(key)
        })
        .collect()
}

//! The settings-diagnostics suite, upstream's
//! `packages/coding-agent/test/settings-diagnostics.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated 1:1.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use pi_coding_agent::settings_diagnostics::{
    SettingsDiagnostic, collect_settings_diagnostics, deduplicate_diagnostics,
};
use pi_coding_agent::settings_manager::{
    FileSettingsStorage, SettingsLockFn, SettingsManager, SettingsManagerCreateOptions,
    SettingsScope, SettingsStorage,
};

/// The storage upstream's second case spells as an inline object literal:
/// the global lock fails, the project lock reads empty and drops any write.
struct FailingGlobalStorage;

impl SettingsStorage for FailingGlobalStorage {
    fn with_lock(&self, scope: SettingsScope, f: SettingsLockFn<'_>) -> Result<(), String> {
        if scope == SettingsScope::Global {
            return Err("backend failed".to_string());
        }
        f(None).map(|_| ())
    }
}

#[test]
fn includes_the_settings_file_path_for_file_backed_storage() {
    let temp = tempfile::tempdir().expect("scratch root");
    let agent_dir = temp.path().join("agent");
    let settings_path = agent_dir.join("settings.json");
    std::fs::create_dir(&agent_dir).expect("agent dir");
    std::fs::write(&settings_path, "{").expect("invalid settings file");

    let mut manager = SettingsManager::<FileSettingsStorage>::create(
        &temp.path().to_string_lossy(),
        &agent_dir.to_string_lossy(),
        SettingsManagerCreateOptions::default(),
    );
    let diagnostics = collect_settings_diagnostics(&mut manager);

    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].diagnostic_type, "warning");
    assert!(
        diagnostics[0].message.contains(&format!(
            "Invalid settings file {}:",
            settings_path.to_string_lossy()
        )),
        "{}",
        diagnostics[0].message
    );
}

#[test]
fn falls_back_to_the_settings_scope_for_storage_without_file_paths() {
    let mut manager = SettingsManager::from_storage(
        FailingGlobalStorage,
        SettingsManagerCreateOptions::default(),
    );
    let diagnostics = collect_settings_diagnostics(&mut manager);

    assert_eq!(
        diagnostics,
        vec![SettingsDiagnostic {
            diagnostic_type: "warning".to_string(),
            message: "Invalid global settings: backend failed".to_string(),
        }]
    );
}

#[test]
fn deduplicates_diagnostics_by_type_and_message() {
    let warning = SettingsDiagnostic {
        diagnostic_type: "warning".to_string(),
        message: "Invalid settings file /tmp/settings.json".to_string(),
    };
    let error = SettingsDiagnostic {
        diagnostic_type: "error".to_string(),
        message: warning.message.clone(),
    };

    assert_eq!(
        deduplicate_diagnostics(vec![warning.clone(), warning.clone(), error.clone()]),
        vec![warning, error]
    );
}

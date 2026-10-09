//! The settings leg: TS pi's deep-merge inputs migrated into the Rust pi's
//! settings files.
//!
//! The inputs are the global `settings.json` under the agent dir and, with
//! `--project`, the project's `<cwd>/.pi/settings.json`.
//!
//! The Rust settings manager owns the format: the leg seeds the manager's
//! in-memory storage with the source content and reads the migrated map
//! back through it, so the legacy-key migrations (queueMode → steeringMode,
//! the websockets boolean → transport, the skills object → array,
//! retry.maxDelayMs → retry.provider.maxRetryDelayMs) run exactly where the
//! Rust pi runs them — at load. The report names the pairs merged and the
//! keys dropped; nothing else leaves the map, unknown keys ride.
//!
//! The write normalizes the file to the migrated form when it differs —
//! the Rust pi would hold that map in memory anyway, and TS pi's own load
//! migration is idempotent, so both runtimes agree on the rewritten file.
//! One key is removed on purpose: when the credentials leg folded
//! `settings.json#apiKeys` into auth.json, the migrated settings drop the
//! key, the file shape upstream's own migration leaves behind.

use std::path::Path;

use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, SettingsManager, SettingsManagerCreateOptions, SettingsPaths,
    SettingsScope,
};
use serde_json::{Map, Value};

use crate::discovery::Discovery;
use crate::plan::{OpTarget, PlannedOp};
use crate::report::{ImportReport, SettingsItem, SettingsOutcome};

/// Run the settings leg: the global scope always, the project scope when
/// `--project` was given. `api_keys_consumed` is the credentials leg's
/// fold flag.
pub fn run_settings(
    discovery: &Discovery,
    report: &mut ImportReport,
    api_keys_consumed: bool,
) -> Vec<PlannedOp> {
    let mut ops = Vec::new();
    let global_source = discovery.source.join("settings.json");
    let global_target = discovery.target.join("settings.json");
    run_scope(
        discovery,
        report,
        &mut ops,
        SettingsScope::Global,
        &global_source,
        &global_target,
        api_keys_consumed,
    );
    if let Some(project) = &discovery.project {
        let project_source = project
            .join(pi_coding_agent::config::CONFIG_DIR_NAME)
            .join("settings.json");
        // The project settings live in the project dir both runtimes read;
        // the migration is in place wherever the file sits.
        run_scope(
            discovery,
            report,
            &mut ops,
            SettingsScope::Project,
            &project_source,
            &project_source,
            api_keys_consumed,
        );
    }
    ops
}

/// Run one settings scope: read, migrate through the manager, report the
/// merged and dropped keys, and plan the write when the content changes.
fn run_scope(
    discovery: &Discovery,
    report: &mut ImportReport,
    ops: &mut Vec<PlannedOp>,
    scope: SettingsScope,
    source_path: &Path,
    target_path: &Path,
    api_keys_consumed: bool,
) {
    let index = report.settings.len();
    let scope_name = match scope {
        SettingsScope::Global => "global",
        SettingsScope::Project => "project",
    };
    report.settings.push(SettingsItem {
        scope: scope_name.to_string(),
        path: source_path.display().to_string(),
        outcome: SettingsOutcome::Unchanged,
        merged: Vec::new(),
        dropped: Vec::new(),
        carried_keys: 0,
    });
    if !source_path.exists() {
        report.settings[index].outcome = SettingsOutcome::Skipped {
            reason: "no settings file".to_string(),
        };
        return;
    }
    let content = match std::fs::read_to_string(source_path) {
        Ok(content) => content,
        Err(error) => {
            let reason = error.to_string();
            report.settings[index].outcome = SettingsOutcome::Failed {
                reason: reason.clone(),
            };
            report.push_failure(source_path.display().to_string(), reason);
            return;
        }
    };
    let raw_map = match parse_object(&content) {
        Ok(map) => map,
        Err(reason) => {
            let reason = format!("Failed to parse settings: {reason}");
            report.settings[index].outcome = SettingsOutcome::Failed {
                reason: reason.clone(),
            };
            report.push_failure(source_path.display().to_string(), reason);
            return;
        }
    };

    // The migration runs where the Rust pi runs it: at the manager's load.
    // The manager cannot fail here: the leg's parse gate already accepted
    // the same bytes the manager re-parses, and the in-memory storage has
    // no lock to lose.
    let migrated = migrate_through_manager(&content, scope);

    let (merged, mut dropped) = migration_notes(&raw_map);
    let mut final_map = migrated;
    if api_keys_consumed && scope == SettingsScope::Global && final_map.contains_key("apiKeys") {
        final_map.shift_remove("apiKeys");
        dropped.push("apiKeys (folded into auth.json)".to_string());
    }
    dropped.sort();
    report.settings[index].merged = merged;
    report.settings[index].dropped = dropped;
    report.settings[index].carried_keys = final_map.len() as u64;

    // The write: in place when the migrated form differs, cross-dir into a
    // target without one.
    let write_path = if discovery.in_place {
        source_path
    } else {
        target_path
    };
    if !discovery.in_place && target_path.exists() {
        report.settings[index].outcome = SettingsOutcome::Skipped {
            reason: "target already has settings.json".to_string(),
        };
        return;
    }
    let unchanged = discovery.in_place && final_map == raw_map;
    if unchanged {
        report.settings[index].outcome = SettingsOutcome::Unchanged;
        return;
    }
    // Serializing a JSON map cannot fail; the default stands in for the
    // unreachable arm.
    let next = serde_json::to_string_pretty(&final_map).unwrap_or_default();
    report.settings[index].outcome = SettingsOutcome::Written;
    report.settings[index].path = write_path.display().to_string();
    ops.push(PlannedOp::WriteFile {
        path: write_path.to_path_buf(),
        content: next,
        mode: None,
        targets: vec![OpTarget::Settings(index)],
    });
}

/// The migrated map the Rust settings manager produces from the seeded
/// content. The leg's parse gate accepted the same bytes the manager
/// re-parses and the in-memory storage cannot fail, so the load has no
/// error arm to serve.
fn migrate_through_manager(content: &str, scope: SettingsScope) -> Map<String, Value> {
    let storage = InMemorySettingsStorage::default();
    storage.seed(scope, content.to_string());
    let manager = SettingsManager::from_storage_with_paths(
        storage,
        SettingsManagerCreateOptions {
            project_trusted: Some(true),
        },
        SettingsPaths {
            global: None,
            project: None,
        },
    );
    match scope {
        SettingsScope::Global => manager.get_global_settings(),
        SettingsScope::Project => manager.get_project_settings(),
    }
}

/// The legacy-key migration notes: the pairs applied and the keys dropped
/// without a replacement. The checks mirror `migrate_settings`'s own
/// conditions — the four rules are the whole deletion contract at the pin —
/// so a note names exactly what the manager did.
fn migration_notes(raw: &Map<String, Value>) -> (Vec<String>, Vec<String>) {
    let mut merged = Vec::new();
    let mut dropped = Vec::new();

    if raw.contains_key("queueMode") && !raw.contains_key("steeringMode") {
        merged.push("queueMode -> steeringMode".to_string());
    }
    if raw.get("websockets").is_some_and(Value::is_boolean) && !raw.contains_key("transport") {
        merged.push("websockets -> transport".to_string());
    }
    if raw
        .get("skills")
        .is_some_and(|skills| skills.is_object() && !skills.is_array())
    {
        let custom = raw
            .get("skills")
            .and_then(Value::as_object)
            .and_then(|skills| skills.get("customDirectories"))
            .and_then(Value::as_array);
        match custom {
            Some(custom) if !custom.is_empty() => {
                merged.push("skills.customDirectories -> skills".to_string());
            }
            _ => dropped.push("skills (empty customDirectories)".to_string()),
        }
    }
    let retry_override = raw
        .get("retry")
        .and_then(Value::as_object)
        .and_then(|retry| retry.get("provider"))
        .and_then(Value::as_object)
        .and_then(|provider| provider.get("maxRetryDelayMs"))
        .is_some_and(|value| !value.is_null());
    if raw
        .get("retry")
        .and_then(Value::as_object)
        .is_some_and(|retry| retry.contains_key("maxDelayMs"))
    {
        if retry_override {
            dropped.push("retry.maxDelayMs (provider override already set)".to_string());
        } else {
            merged.push("retry.maxDelayMs -> retry.provider.maxRetryDelayMs".to_string());
        }
    }
    (merged, dropped)
}

/// The BOM-stripped JSON object parse the raw diff reads.
#[doc(hidden)]
pub fn parse_object(content: &str) -> Result<Map<String, Value>, String> {
    let value: Value = serde_json::from_str(pi_coding_agent::utils::text::strip_bom(content))
        .map_err(|error| error.to_string())?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| "not a JSON object".to_string())
}

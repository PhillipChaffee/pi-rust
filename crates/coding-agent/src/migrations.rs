//! One-time migrations that run on startup, upstream's
//! `packages/coding-agent/src/migrations.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Porting restatements this module records:
//!
//! - `migrateKeybindingsConfigFile` stages behind the keybindings ticket:
//!   the migration machinery lives in upstream's `core/keybindings.ts` and
//!   lands with the interactive shell (map ticket "pi-coding-agent:
//!   interactive mode shell and selectors"), so the file migration runs
//!   through the [`KeybindingsMigrator`] seam — the real one arrives with
//!   that ticket, and until then the file passes through untouched.
//! - `showDeprecationWarnings` rides the CLI entry: its colored console
//!   output and raw-mode stdin wait are startup-UI surface, upstream's
//!   `main.ts` caller, and no suite covers it.
//! - The colored `console.log` lines carry through a printer seam
//!   ([`run_migrations_with`]); the plain [`run_migrations`] prints them
//!   uncolored, the chalk dependency riding the CLI's runtime styling.
//! - The sweep sources its agent dir through `getAgentDir()`, upstream's
//!   `process.env[ENV_AGENT_DIR]` test rig rides the `_with` forms that
//!   inject an environment lookup, the same seam the auth and session
//!   migrations carry.

use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use serde_json::Value;

use crate::config::{
    CONFIG_DIR_NAME, EnvLookup, default_env_lookup, encode_session_cwd, get_agent_dir_with,
};
use crate::utils::text::strip_bom;

/// The extension-migration guide, upstream's `MIGRATION_GUIDE_URL`.
pub const MIGRATION_GUIDE_URL: &str = "https://github.com/earendil-works/pi/blob/main/packages/coding-agent/CHANGELOG.md#extensions-migration";
/// The extensions documentation, upstream's `EXTENSIONS_DOC_URL`.
pub const EXTENSIONS_DOC_URL: &str =
    "https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/extensions.md";

/// The printer seam, upstream's `console.log`.
pub type Printer<'a> = &'a mut dyn FnMut(&str);

/// The keybindings migration seam, upstream's `migrateKeybindingsConfig`
/// until the interactive-shell ticket lands the real table.
pub type KeybindingsMigrator<'a> =
    &'a mut dyn FnMut(
        &serde_json::Map<String, Value>,
    ) -> Option<(serde_json::Map<String, Value>, bool)>;

/// Migrate legacy oauth.json and settings.json apiKeys to auth.json,
/// upstream's `migrateAuthToAuthJson`.
///
/// The provider names that were migrated return, `auth.json`'s existence
/// skips the whole migration, and every failure inside a source file skips
/// that source silently.
///
/// # Errors
/// The filesystem failures the agent-dir operations raise.
pub fn migrate_auth_to_auth_json() -> Result<Vec<String>, String> {
    migrate_auth_to_auth_json_with(&default_env_lookup())
}

/// [`migrate_auth_to_auth_json`] over an injected environment lookup,
/// upstream's `process.env[ENV_AGENT_DIR]` test rig.
///
/// # Errors
/// The filesystem failures the agent-dir operations raise.
pub fn migrate_auth_to_auth_json_with(env: &EnvLookup) -> Result<Vec<String>, String> {
    migrate_auth_to_auth_json_in(&get_agent_dir_with(env))
}

/// [`migrate_auth_to_auth_json`] against an
/// explicit agent directory, the test form.
///
/// # Errors
/// The filesystem failures the agent-dir operations raise.
pub fn migrate_auth_to_auth_json_in(agent_dir: &Path) -> Result<Vec<String>, String> {
    let agent_dir = agent_dir.to_string_lossy().into_owned();
    let auth_path = format!("{agent_dir}/auth.json");
    let oauth_path = format!("{agent_dir}/oauth.json");
    let settings_path = format!("{agent_dir}/settings.json");

    // Skip if auth.json already exists.
    if Path::new(&auth_path).exists() {
        return Ok(Vec::new());
    }

    let mut migrated = serde_json::Map::new();
    let mut providers: Vec<String> = Vec::new();

    // Migrate oauth.json.
    if Path::new(&oauth_path).exists() {
        let outcome = (|| {
            let content =
                std::fs::read_to_string(&oauth_path).map_err(|error| error.to_string())?;
            let oauth: Value =
                serde_json::from_str(strip_bom(&content)).map_err(|error| error.to_string())?;
            let Some(entries) = oauth.as_object() else {
                return Ok(());
            };
            for (provider, cred) in entries {
                let mut credential = match cred {
                    Value::Object(object) => object.clone(),
                    _ => serde_json::Map::new(),
                };
                credential.insert("type".to_string(), Value::String("oauth".to_string()));
                migrated.insert(provider.clone(), Value::Object(credential));
                providers.push(provider.clone());
            }
            std::fs::rename(&oauth_path, format!("{oauth_path}.migrated"))
                .map_err(|error| error.to_string())
        })();
        // Skip on error.
        let _ = outcome;
    }

    // Migrate settings.json apiKeys.
    if Path::new(&settings_path).exists() {
        let outcome = (|| {
            let content =
                std::fs::read_to_string(&settings_path).map_err(|error| error.to_string())?;
            let mut settings: serde_json::Map<String, Value> =
                serde_json::from_str(strip_bom(&content)).map_err(|error| error.to_string())?;
            let Some(api_keys) = settings.get("apiKeys").and_then(Value::as_object) else {
                return Ok(());
            };
            for (provider, key) in api_keys {
                if !migrated.contains_key(provider)
                    && let Some(key) = key.as_str()
                {
                    let mut credential = serde_json::Map::new();
                    credential.insert("type".to_string(), Value::String("api_key".to_string()));
                    credential.insert("key".to_string(), Value::String(key.to_string()));
                    migrated.insert(provider.clone(), Value::Object(credential));
                    providers.push(provider.clone());
                }
            }
            settings.shift_remove("apiKeys");
            std::fs::write(
                &settings_path,
                serde_json::to_string_pretty(&settings).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())
        })();
        // Skip on error.
        let _ = outcome;
    }

    if !migrated.is_empty() {
        if let Some(parent) = Path::new(&auth_path).parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&auth_path)
            .map_err(|error| error.to_string())?
            .write_all(
                serde_json::to_string_pretty(&migrated)
                    .map_err(|error| error.to_string())?
                    .as_bytes(),
            )
            .map_err(|error| error.to_string())?;
    }

    Ok(providers)
}

/// Migrate sessions from `~/.pi/agent/*.jsonl` to their per-cwd session
/// directories, upstream's `migrateSessionsFromAgentRoot`.
///
/// The v0.30.0 bug saved sessions next to the agent dir, and this moves
/// them by the cwd in their session header (upstream issue 320). Files
/// that cannot migrate skip silently.
pub fn migrate_sessions_from_agent_root() {
    migrate_sessions_from_agent_root_with(&default_env_lookup());
}

/// [`migrate_sessions_from_agent_root`] over an injected environment
/// lookup, upstream's `process.env[ENV_AGENT_DIR]` test rig.
pub fn migrate_sessions_from_agent_root_with(env: &EnvLookup) {
    migrate_sessions_from_agent_root_in(&get_agent_dir_with(env));
}

/// [`migrate_sessions_from_agent_root`] against an explicit agent
/// directory, the test form.
pub fn migrate_sessions_from_agent_root_in(agent_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(agent_dir) else {
        return;
    };
    let files: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension == "jsonl")
        })
        .collect();
    if files.is_empty() {
        return;
    }

    for file in files {
        let outcome = (|| -> Result<(), String> {
            let content = std::fs::read_to_string(&file).map_err(|error| error.to_string())?;
            let first_line = content.split('\n').next().unwrap_or_default();
            if first_line.trim().is_empty() {
                return Ok(());
            }
            let header: Value =
                serde_json::from_str(first_line).map_err(|error| error.to_string())?;
            if header.get("type").and_then(Value::as_str) != Some("session") {
                return Ok(());
            }
            let Some(cwd) = header.get("cwd").and_then(Value::as_str) else {
                return Ok(());
            };

            // The session-dir encoding, upstream's session-manager form.
            let correct_dir = agent_dir.join("sessions").join(encode_session_cwd(cwd));

            if !correct_dir.exists() {
                std::fs::create_dir_all(&correct_dir).map_err(|error| error.to_string())?;
            }
            let file_name = file
                .file_name()
                .ok_or("unnamed session file")?
                .to_string_lossy()
                .into_owned();
            let new_path = correct_dir.join(file_name);
            if new_path.exists() {
                // Skip if target exists.
                return Ok(());
            }
            std::fs::rename(&file, &new_path).map_err(|error| error.to_string())
        })();
        // Skip files that can't be migrated.
        let _ = outcome;
    }
}

/// Migrate `commands/` to `prompts/` when needed, upstream's
/// `migrateCommandsToPrompts`: regular directories and symlinks both move.
fn migrate_commands_to_prompts(base_dir: &str, label: &str, print: Printer<'_>) -> bool {
    let commands_dir = Path::new(base_dir).join("commands");
    let prompts_dir = Path::new(base_dir).join("prompts");

    if commands_dir.exists() && !prompts_dir.exists() {
        match std::fs::rename(&commands_dir, &prompts_dir) {
            Ok(()) => {
                print(&format!("Migrated {label} commands/ → prompts/"));
                return true;
            }
            Err(error) => {
                print(&format!(
                    "Warning: Could not migrate {label} commands/ to prompts/: {error}"
                ));
            }
        }
    }
    false
}

/// Migrate `keybindings.json` through the migrator seam, upstream's
/// `migrateKeybindingsConfigFile`. Malformed files ignore silently.
fn migrate_keybindings_config_file_in(agent_dir: &Path, migrate: KeybindingsMigrator<'_>) {
    let config_path = agent_dir.join("keybindings.json");
    if !config_path.exists() {
        return;
    }
    let outcome = (|| -> Result<(), String> {
        let content = std::fs::read_to_string(&config_path).map_err(|error| error.to_string())?;
        let parsed: Value =
            serde_json::from_str(strip_bom(&content)).map_err(|error| error.to_string())?;
        let Some(config) = parsed.as_object() else {
            return Ok(());
        };
        let Some((migrated_config, migrated)) = migrate(config) else {
            return Ok(());
        };
        if !migrated {
            return Ok(());
        }
        std::fs::write(
            &config_path,
            format!(
                "{}\n",
                serde_json::to_string_pretty(&migrated_config).map_err(|error| error.to_string())?
            ),
        )
        .map_err(|error| error.to_string())
    })();
    // Ignore malformed files during migration.
    let _ = outcome;
}

/// Move fd/rg binaries from `tools/` to `bin/`, upstream's
/// `migrateToolsToBin`. The bin dir derives from the same agent dir,
/// upstream's `getBinDir()` over the env-resolved agent dir.
fn migrate_tools_to_bin_in(agent_dir: &Path, print: Printer<'_>) {
    let tools_dir = agent_dir.join("tools");
    let bin_dir = agent_dir.join("bin");

    if !tools_dir.exists() {
        return;
    }

    let binaries = ["fd", "rg", "fd.exe", "rg.exe"];
    let mut moved_any = false;

    for bin in binaries {
        let old_path = tools_dir.join(bin);
        let new_path = bin_dir.join(bin);

        if old_path.exists() {
            if !bin_dir.exists() {
                let _ = std::fs::create_dir_all(&bin_dir);
            }
            if new_path.exists() {
                // Target exists, just delete the old one.
                let _ = std::fs::remove_file(&old_path);
            } else if std::fs::rename(&old_path, &new_path).is_ok() {
                moved_any = true;
            }
        }
    }

    if moved_any {
        print("Migrated managed binaries tools/ → bin/");
    }
}

/// Check for deprecated `hooks/` and `tools/` directories, upstream's
/// `checkDeprecatedExtensionDirs`: `tools/` may carry the auto-extracted
/// fd/rg binaries, so only other files warn, and hidden files never do.
fn check_deprecated_extension_dirs(base_dir: &str, label: &str) -> Vec<String> {
    let hooks_dir = Path::new(base_dir).join("hooks");
    let tools_dir = Path::new(base_dir).join("tools");
    let mut warnings = Vec::new();

    if hooks_dir.exists() {
        warnings.push(format!(
            "{label} hooks/ directory found. Hooks have been renamed to extensions."
        ));
    }

    if tools_dir.exists() {
        let custom_tools: Vec<String> = std::fs::read_dir(&tools_dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .filter(|name| {
                        let lower = name.to_lowercase();
                        lower != "fd"
                            && lower != "rg"
                            && lower != "fd.exe"
                            && lower != "rg.exe"
                            && !name.starts_with('.')
                    })
                    .collect()
            })
            .unwrap_or_default();
        if !custom_tools.is_empty() {
            warnings.push(format!(
                "{label} tools/ directory contains custom tools. Custom tools have been merged into extensions."
            ));
        }
    }

    warnings
}

/// Run the extension-system migrations and collect the deprecation
/// warnings, upstream's `migrateExtensionSystem`.
fn migrate_extension_system_in(cwd: &str, agent_dir: &Path, print: Printer<'_>) -> Vec<String> {
    let project_dir = Path::new(cwd).join(CONFIG_DIR_NAME);

    migrate_commands_to_prompts(&agent_dir.to_string_lossy(), "Global", print);
    migrate_commands_to_prompts(&project_dir.to_string_lossy(), "Project", print);

    let mut warnings = check_deprecated_extension_dirs(&agent_dir.to_string_lossy(), "Global");
    warnings.extend(check_deprecated_extension_dirs(
        &project_dir.to_string_lossy(),
        "Project",
    ));
    warnings
}

/// The default migration printer, upstream's unoverridden `console.log`.
#[doc(hidden)]
#[expect(
    clippy::print_stdout,
    reason = "the migration report prints to stdout, upstream's console.log surface"
)]
pub fn print_migration_line(line: &str) {
    println!("{line}");
}

/// The default keybindings migrator: the sweep rides the
/// [`KeybindingsMigrator`] seam until the interactive-shell ticket lands the
/// real table, so the default passes the config through untouched.
#[doc(hidden)]
#[must_use]
pub const fn no_keybindings_migration(
    _config: &serde_json::Map<String, Value>,
) -> Option<(serde_json::Map<String, Value>, bool)> {
    None
}

/// Run all migrations, upstream's `runMigrations`: the migrated auth
/// providers plus the deprecation warnings.
#[must_use]
pub fn run_migrations(cwd: &str) -> MigrationReport {
    run_migrations_with(
        &default_env_lookup(),
        cwd,
        &mut print_migration_line,
        &mut no_keybindings_migration,
    )
}

/// [`run_migrations`] over an injected environment lookup, an explicit
/// printer, and a keybindings migrator.
///
/// The seams upstream threads through `process.env` and optional call
/// arguments.
pub fn run_migrations_with(
    env: &EnvLookup,
    cwd: &str,
    print: Printer<'_>,
    migrate_keybindings: KeybindingsMigrator<'_>,
) -> MigrationReport {
    run_migrations_with_in(cwd, &get_agent_dir_with(env), print, migrate_keybindings)
}

/// [`run_migrations_with`] against an explicit agent
/// directory.
///
/// The env-independent test seam in place of upstream's
/// `process.env[ENV_AGENT_DIR]` — the `_in` counterpart the auth and session
/// migrations already carry.
pub fn run_migrations_with_in(
    cwd: &str,
    agent_dir: &Path,
    print: Printer<'_>,
    migrate_keybindings: KeybindingsMigrator<'_>,
) -> MigrationReport {
    let migrated_auth_providers = migrate_auth_to_auth_json_in(agent_dir).unwrap_or_default();
    migrate_sessions_from_agent_root_in(agent_dir);
    migrate_tools_to_bin_in(agent_dir, print);
    migrate_keybindings_config_file_in(agent_dir, migrate_keybindings);
    let deprecation_warnings = migrate_extension_system_in(cwd, agent_dir, print);
    MigrationReport {
        migrated_auth_providers,
        deprecation_warnings,
    }
}

/// The migration report, upstream's `{ migratedAuthProviders,
/// deprecationWarnings }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationReport {
    /// The provider names migrated into `auth.json`.
    pub migrated_auth_providers: Vec<String>,
    /// The deprecation warnings the extension-system sweep collected.
    pub deprecation_warnings: Vec<String>,
}

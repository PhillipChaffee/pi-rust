//! Boundary tests binding the migrations branches the 1:1 suites leave
//! untested, at pin 60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759.
//!
//! The process-agent-dir entry points (`run_migrations`,
//! `migrate_auth_to_auth_json`, `migrate_sessions_from_agent_root`) read the
//! process agent dir and would touch the developer's real `~/.pi`; the
//! workspace forbids mutating the process environment, so every test here
//! rides the `_in` seams or the `_with` env seam over its own temp directory.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::cell::Cell;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use pi_coding_agent::config::{CONFIG_DIR_NAME, EnvLookup, encode_session_cwd};
use pi_coding_agent::migrations::{
    MigrationReport, migrate_auth_to_auth_json_in, migrate_auth_to_auth_json_with,
    migrate_sessions_from_agent_root_in, migrate_sessions_from_agent_root_with,
    no_keybindings_migration, print_migration_line, run_migrations_with, run_migrations_with_in,
};
use serde_json::{Map, Value, json};

/// A scratch directory with a per-test prefix, unique across the suite.
fn temp_dir(prefix: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(prefix)
        .tempdir()
        .expect("temp dir")
}

/// The environment upstream's test rig builds by setting
/// `process.env[ENV_AGENT_DIR]`: one entry, the scratch agent dir.
fn agent_dir_env(agent_dir: &str) -> EnvLookup {
    let owned = agent_dir.to_string();
    Box::new(move |key| (key == "PI_CODING_AGENT_DIR").then(|| owned.clone()))
}

/// Restore a directory's mode on drop, so a failing assertion cannot leave a
/// read-only tree behind for the temp-dir cleanup to trip over.
struct RestorePerms<'a> {
    path: &'a Path,
    mode: u32,
}

impl Drop for RestorePerms<'_> {
    fn drop(&mut self) {
        let _ = fs::set_permissions(self.path, fs::Permissions::from_mode(self.mode | 0o755));
    }
}

/// Run the full migration sweep over `agent_dir` (its own cwd, upstream's
/// `runMigrations(agentDir)`), capturing the printed lines.
fn sweep(agent_dir: &Path) -> (MigrationReport, Vec<String>) {
    sweep_over(agent_dir, agent_dir)
}

/// [`sweep`] with the project cwd distinct from the agent dir.
fn sweep_over(agent_dir: &Path, cwd: &Path) -> (MigrationReport, Vec<String>) {
    let mut lines: Vec<String> = Vec::new();
    let report = {
        let mut print_line = |line: &str| lines.push(line.to_string());
        let mut no_keybindings =
            |_config: &Map<String, Value>| -> Option<(Map<String, Value>, bool)> { None };
        run_migrations_with_in(
            &cwd.to_string_lossy(),
            agent_dir,
            &mut print_line,
            &mut no_keybindings,
        )
    };
    (report, lines)
}

// =============================================================================
// migrate_auth_to_auth_json_in
// =============================================================================

#[test]
fn migrate_auth_skips_everything_when_auth_json_exists() {
    let temp = temp_dir("pi-mig-auth-skip-");
    fs::write(temp.path().join("auth.json"), "{}").expect("auth write");
    fs::write(
        temp.path().join("oauth.json"),
        r#"{"anthropic":{"access":"a"}}"#,
    )
    .expect("oauth write");
    fs::write(temp.path().join("settings.json"), r#"{"apiKeys":{}}"#).expect("settings write");

    let providers = migrate_auth_to_auth_json_in(temp.path()).expect("migration");

    assert!(providers.is_empty(), "auth.json skips the whole migration");
    assert!(
        temp.path().join("oauth.json").exists(),
        "the oauth source is never touched"
    );
    assert_eq!(
        fs::read_to_string(temp.path().join("settings.json")).expect("settings read"),
        r#"{"apiKeys":{}}"#,
        "the settings source is never touched"
    );
}

#[test]
fn migrate_auth_moves_oauth_entries_and_renames_the_source() {
    let temp = temp_dir("pi-mig-auth-oauth-");
    // A BOM rides the oauth source; the strip-bom pass must see through it.
    fs::write(
        temp.path().join("oauth.json"),
        format!(
            "\u{FEFF}{}",
            json!({
                "anthropic": {"access": "access-token", "refresh": "refresh-token", "expires": 1},
                "broken": "not-an-object",
            })
        ),
    )
    .expect("oauth write");

    let providers = migrate_auth_to_auth_json_in(temp.path()).expect("migration");

    assert_eq!(providers, vec!["anthropic", "broken"]);
    let auth: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("auth.json")).expect("auth read"),
    )
    .expect("auth parse");
    assert_eq!(
        auth["anthropic"],
        json!({"access": "access-token", "refresh": "refresh-token", "expires": 1, "type": "oauth"}),
        "the oauth entry keeps its fields and gains the type tag"
    );
    // A non-object credential migrates as the type tag alone, upstream's
    // empty-record spread.
    assert_eq!(auth["broken"], json!({"type": "oauth"}));
    assert!(
        temp.path().join("oauth.json.migrated").exists(),
        "the migrated source is renamed"
    );
    assert!(!temp.path().join("oauth.json").exists());

    let mode = fs::metadata(temp.path().join("auth.json"))
        .expect("auth metadata")
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600, "the migrated store is owner-only");
}

#[test]
fn migrate_auth_skips_a_malformed_oauth_file_silently() {
    let temp = temp_dir("pi-mig-auth-bad-oauth-");
    fs::write(temp.path().join("oauth.json"), "{invalid-json").expect("oauth write");

    let providers = migrate_auth_to_auth_json_in(temp.path()).expect("migration");

    assert!(
        providers.is_empty(),
        "the failure skips that source silently"
    );
    assert!(
        !temp.path().join("auth.json").exists(),
        "nothing is written from a malformed source"
    );
    assert!(
        temp.path().join("oauth.json").exists(),
        "a malformed source is never renamed"
    );
}

#[test]
fn migrate_auth_skips_unreadable_and_non_object_oauth_sources() {
    // A non-object root parses but yields no entries.
    let temp = temp_dir("pi-mig-auth-oauth-array-");
    fs::write(temp.path().join("oauth.json"), "[1,2]").expect("oauth write");
    let providers = migrate_auth_to_auth_json_in(temp.path()).expect("migration");
    assert!(providers.is_empty(), "a non-object root migrates nothing");
    assert!(
        temp.path().join("oauth.json").exists(),
        "an empty migration never renames the source"
    );

    // An unreadable source fails the read and skips silently.
    let temp = temp_dir("pi-mig-auth-oauth-dir-");
    fs::create_dir(temp.path().join("oauth.json")).expect("a directory as the source");
    let providers = migrate_auth_to_auth_json_in(temp.path()).expect("migration");
    assert!(providers.is_empty(), "the read failure skips silently");
}

#[test]
fn migrate_auth_skips_unreadable_and_malformed_settings_sources() {
    // A malformed settings file fails the parse and skips silently.
    let temp = temp_dir("pi-mig-auth-bad-settings-");
    fs::write(temp.path().join("settings.json"), "{invalid").expect("settings write");
    let providers = migrate_auth_to_auth_json_in(temp.path()).expect("migration");
    assert!(providers.is_empty());
    assert_eq!(
        fs::read_to_string(temp.path().join("settings.json")).expect("settings read"),
        "{invalid",
        "a malformed settings file is never rewritten"
    );

    // A directory as the settings source fails the read and skips silently.
    let temp = temp_dir("pi-mig-auth-settings-dir-");
    fs::create_dir(temp.path().join("settings.json")).expect("a directory as the source");
    let providers = migrate_auth_to_auth_json_in(temp.path()).expect("migration");
    assert!(providers.is_empty());
    assert!(!temp.path().join("auth.json").exists());

    // A rewrite failure (read-only file) swallows silently, but the providers
    // were already collected and auth.json still writes.
    let temp = temp_dir("pi-mig-auth-settings-ro-");
    fs::write(
        temp.path().join("settings.json"),
        json!({"apiKeys": {"openai": "sk-openai"}}).to_string(),
    )
    .expect("settings write");
    fs::set_permissions(
        temp.path().join("settings.json"),
        fs::Permissions::from_mode(0o444),
    )
    .expect("read-only the settings file");
    let providers = migrate_auth_to_auth_json_in(temp.path()).expect("migration");
    assert_eq!(
        providers,
        vec!["openai"],
        "the collected providers survive the swallowed rewrite failure"
    );
    let auth: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("auth.json")).expect("auth read"),
    )
    .expect("auth parse");
    assert_eq!(auth["openai"]["type"], "api_key");
    let settings: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("settings.json")).expect("settings read"),
    )
    .expect("settings parse");
    assert_eq!(
        settings["apiKeys"]["openai"], "sk-openai",
        "the failed rewrite left apiKeys in place"
    );
    fs::set_permissions(
        temp.path().join("settings.json"),
        fs::Permissions::from_mode(0o644),
    )
    .expect("restore the settings file");
}

#[test]
fn migrate_auth_merges_settings_api_keys_under_the_oauth_winners() {
    let temp = temp_dir("pi-mig-auth-merge-");
    fs::write(
        temp.path().join("oauth.json"),
        json!({"anthropic": {"access": "a", "refresh": "r", "expires": 1}}).to_string(),
    )
    .expect("oauth write");
    fs::write(
        temp.path().join("settings.json"),
        json!({"theme": "dark", "apiKeys": {
            "openai": "sk-openai",
            "anthropic": "sk-anthropic-loses",
            "numbered": 42,
        }})
        .to_string(),
    )
    .expect("settings write");

    let providers = migrate_auth_to_auth_json_in(temp.path()).expect("migration");

    // oauth.json runs first, its providers win, and non-string keys skip.
    assert_eq!(providers, vec!["anthropic", "openai"]);
    let auth: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("auth.json")).expect("auth read"),
    )
    .expect("auth parse");
    assert_eq!(auth["anthropic"]["type"], "oauth");
    assert_eq!(
        auth["openai"],
        json!({"type": "api_key", "key": "sk-openai"}),
    );
    assert!(
        auth.get("numbered").is_none(),
        "a non-string api key never migrates"
    );

    // The settings file is rewritten pretty with apiKeys removed.
    let settings: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join("settings.json")).expect("settings read"),
    )
    .expect("settings parse");
    assert_eq!(settings, json!({"theme": "dark"}));
    let expected =
        serde_json::to_string_pretty(&json!({"theme": "dark"})).expect("settings serialize");
    assert_eq!(
        fs::read_to_string(temp.path().join("settings.json")).expect("settings read"),
        expected,
        "the rewrite is serde's pretty form"
    );
}

#[test]
fn migrate_auth_leaves_settings_untouched_without_an_api_keys_object() {
    let temp = temp_dir("pi-mig-auth-nokeys-");
    fs::write(
        temp.path().join("settings.json"),
        json!({"theme": "dark"}).to_string(),
    )
    .expect("settings write");

    let providers = migrate_auth_to_auth_json_in(temp.path()).expect("migration");

    assert!(providers.is_empty());
    assert!(!temp.path().join("auth.json").exists());
    assert_eq!(
        fs::read_to_string(temp.path().join("settings.json")).expect("settings read"),
        json!({"theme": "dark"}).to_string(),
        "the early return happens before any rewrite"
    );
}

#[test]
fn migrate_auth_writes_nothing_when_neither_source_exists() {
    let temp = temp_dir("pi-mig-auth-empty-");

    let providers = migrate_auth_to_auth_json_in(temp.path()).expect("migration");

    assert!(providers.is_empty());
    assert!(!temp.path().join("auth.json").exists());
}

#[test]
fn migrate_auth_reports_an_error_when_the_auth_file_cannot_be_written() {
    let temp = temp_dir("pi-mig-auth-ro-");
    fs::write(
        temp.path().join("oauth.json"),
        json!({"anthropic": {"access": "a", "refresh": "r", "expires": 1}}).to_string(),
    )
    .expect("oauth write");
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o555))
        .expect("read-only the agent dir");
    let _restore = RestorePerms {
        path: temp.path(),
        mode: 0o555,
    };

    let outcome = migrate_auth_to_auth_json_in(temp.path());

    let error = outcome.expect_err("the write failure propagates");
    assert!(
        error.contains("Permission denied"),
        "the filesystem failure carries its message, got: {error}"
    );
}

// =============================================================================
// migrate_sessions_from_agent_root_in
// =============================================================================

fn write_session(path: &Path, first_line: &str) {
    fs::write(path, format!("{first_line}\nrest of the session\n")).expect("session write");
}

#[test]
fn migrate_sessions_moves_each_session_by_its_cwd_encoding() {
    let temp = temp_dir("pi-mig-sessions-move-");
    write_session(
        &temp.path().join("abc.jsonl"),
        r#"{"type":"session","cwd":"/Users/foo/bar"}"#,
    );
    write_session(
        &temp.path().join("def.jsonl"),
        r#"{"type":"session","cwd":"/tmp/a b"}"#,
    );

    migrate_sessions_from_agent_root_in(temp.path());

    let foo_dir = temp
        .path()
        .join("sessions")
        .join(encode_session_cwd("/Users/foo/bar"));
    let tmp_dir = temp
        .path()
        .join("sessions")
        .join(encode_session_cwd("/tmp/a b"));
    assert!(
        foo_dir.join("abc.jsonl").exists(),
        "the file lands in the encoded cwd dir"
    );
    assert!(tmp_dir.join("def.jsonl").exists());
    assert!(!temp.path().join("abc.jsonl").exists());
}

#[test]
fn migrate_sessions_skips_files_that_cannot_migrate() {
    let temp = temp_dir("pi-mig-sessions-skip-");
    write_session(
        &temp.path().join("other-type.jsonl"),
        r#"{"type":"message"}"#,
    );
    write_session(&temp.path().join("no-cwd.jsonl"), r#"{"type":"session"}"#);
    write_session(&temp.path().join("malformed.jsonl"), "{not-json");
    fs::write(
        temp.path().join("blank-first.jsonl"),
        "\n{\"type\":\"session\"}",
    )
    .expect("blank write");
    // A correct target already holding the name skips the move.
    let target = temp
        .path()
        .join("sessions")
        .join(encode_session_cwd("/Users/foo/bar"));
    fs::create_dir_all(&target).expect("target dir");
    fs::write(target.join("taken.jsonl"), "already there").expect("target write");
    write_session(
        &temp.path().join("taken.jsonl"),
        r#"{"type":"session","cwd":"/Users/foo/bar"}"#,
    );

    migrate_sessions_from_agent_root_in(temp.path());

    for name in [
        "other-type.jsonl",
        "no-cwd.jsonl",
        "malformed.jsonl",
        "blank-first.jsonl",
    ] {
        assert!(
            temp.path().join(name).exists(),
            "{name} stays in the agent root"
        );
    }
    assert!(
        target.join("taken.jsonl").exists(),
        "the pre-existing target survives"
    );
    assert_eq!(
        fs::read_to_string(target.join("taken.jsonl")).expect("target read"),
        "already there",
        "the target is never overwritten"
    );
    assert!(
        temp.path().join("taken.jsonl").exists(),
        "the skip keeps the source"
    );
}

#[test]
fn migrate_sessions_ignores_subdirectories_and_non_jsonl_files() {
    let temp = temp_dir("pi-mig-sessions-nested-");
    fs::create_dir_all(temp.path().join("sessions")).expect("sessions dir");
    write_session(
        &temp.path().join("sessions/nested.jsonl"),
        r#"{"type":"session","cwd":"/Users/foo/bar"}"#,
    );
    write_session(&temp.path().join("notes.txt"), "not a session");
    fs::create_dir(temp.path().join("dir.jsonl")).expect("a directory named like a session");

    migrate_sessions_from_agent_root_in(temp.path());

    assert!(
        temp.path().join("sessions/nested.jsonl").exists(),
        "only the agent root's own files scan"
    );
    assert!(temp.path().join("notes.txt").exists());
    assert!(temp.path().join("dir.jsonl").exists());
}

#[test]
fn migrate_sessions_returns_when_the_agent_dir_is_missing() {
    let missing = temp_dir("pi-mig-sessions-missing-").path().join("nope");

    migrate_sessions_from_agent_root_in(&missing);
}

#[test]
fn migrate_sessions_skips_files_that_fail_to_read_or_move() {
    // An unreadable session file fails the read and skips silently.
    let temp = temp_dir("pi-mig-sessions-noperm-");
    let locked = temp.path().join("locked.jsonl");
    write_session(&locked, r#"{"type":"session","cwd":"/Users/foo/bar"}"#);
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000))
        .expect("strip the session file's permissions");

    migrate_sessions_from_agent_root_in(temp.path());
    assert!(
        temp.path().join("locked.jsonl").exists(),
        "the unreadable file stays in place"
    );
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o644))
        .expect("restore the session file's permissions");

    // A read-only agent dir fails the target's mkdir and skips silently.
    let temp = temp_dir("pi-mig-sessions-ro-");
    write_session(
        &temp.path().join("abc.jsonl"),
        r#"{"type":"session","cwd":"/Users/foo/bar"}"#,
    );
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o555))
        .expect("read-only the agent dir");
    let _restore = RestorePerms {
        path: temp.path(),
        mode: 0o555,
    };
    migrate_sessions_from_agent_root_in(temp.path());
    assert!(
        temp.path().join("abc.jsonl").exists(),
        "the mkdir failure keeps the source"
    );
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o755))
        .expect("restore the agent dir");

    // A held target dir fails the rename and skips silently.
    let temp = temp_dir("pi-mig-sessions-renamer-");
    let target = temp
        .path()
        .join("sessions")
        .join(encode_session_cwd("/Users/foo/bar"));
    fs::create_dir_all(&target).expect("pre-create the target dir");
    write_session(
        &temp.path().join("abc.jsonl"),
        r#"{"type":"session","cwd":"/Users/foo/bar"}"#,
    );
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o555))
        .expect("read-only the agent dir");
    let _restore = RestorePerms {
        path: temp.path(),
        mode: 0o555,
    };
    migrate_sessions_from_agent_root_in(temp.path());
    assert!(
        temp.path().join("abc.jsonl").exists(),
        "the rename failure keeps the source"
    );
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o755))
        .expect("restore the agent dir");
}

// =============================================================================
// commands/ → prompts/ and the deprecation sweep
// =============================================================================

#[test]
fn sweep_renames_commands_to_prompts_and_prints_the_migration() {
    let project = temp_dir("pi-mig-cmds-move-");
    let agent = project.path().join("agent");
    fs::create_dir_all(agent.join("commands")).expect("global commands");
    fs::write(agent.join("commands/review.md"), "prompt").expect("command write");
    fs::create_dir_all(project.path().join(CONFIG_DIR_NAME).join("commands"))
        .expect("project commands");
    fs::write(
        project.path().join(CONFIG_DIR_NAME).join("commands/fix.md"),
        "prompt",
    )
    .expect("command write");

    let (_report, lines) = sweep_over(&agent, project.path());

    assert!(
        agent.join("prompts/review.md").exists(),
        "global commands move"
    );
    assert!(
        project
            .path()
            .join(CONFIG_DIR_NAME)
            .join("prompts/fix.md")
            .exists(),
        "project commands move"
    );
    assert!(!agent.join("commands").exists());
    assert_eq!(
        lines,
        vec![
            "Migrated Global commands/ → prompts/".to_string(),
            "Migrated Project commands/ → prompts/".to_string(),
        ],
        "both migrations print, in order"
    );
}

#[test]
fn sweep_keeps_commands_when_prompts_already_exists() {
    let temp = temp_dir("pi-mig-cmds-both-");
    fs::create_dir_all(temp.path().join("commands")).expect("commands dir");
    fs::write(temp.path().join("commands/review.md"), "prompt").expect("command write");
    fs::create_dir_all(temp.path().join("prompts")).expect("prompts dir");

    let (_report, lines) = sweep(temp.path());

    assert!(temp.path().join("commands").exists(), "nothing is renamed");
    assert!(lines.is_empty(), "no migration prints when both exist");
}

#[test]
fn sweep_warns_when_the_commands_rename_fails() {
    let temp = temp_dir("pi-mig-cmds-ro-");
    fs::create_dir_all(temp.path().join("commands")).expect("commands dir");
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o555))
        .expect("read-only the agent dir");
    let _restore = RestorePerms {
        path: temp.path(),
        mode: 0o555,
    };

    let (_report, lines) = sweep(temp.path());

    assert_eq!(lines.len(), 1, "the failure prints one warning");
    assert!(
        lines[0].starts_with("Warning: Could not migrate Global commands/ to prompts/: "),
        "the warning names the pair and the error, got: {}",
        lines[0]
    );
}

#[test]
fn sweep_warns_on_hooks_and_custom_tools_but_not_managed_or_hidden_ones() {
    let project = temp_dir("pi-mig-deprecated-");
    let agent = project.path().join("agent");
    fs::create_dir_all(agent.join("hooks")).expect("global hooks");
    fs::create_dir_all(agent.join("tools")).expect("global tools");
    // Hidden files never warn, and a managed binary whose target already
    // exists is deleted without moving or warning.
    fs::write(agent.join("tools/.hidden.js"), "hidden").expect("hidden write");
    fs::create_dir_all(agent.join("bin")).expect("bin dir");
    fs::write(agent.join("tools/fd"), "old fd").expect("old fd write");
    fs::write(agent.join("bin/fd"), "newer fd").expect("target fd write");
    let project_config = project.path().join(CONFIG_DIR_NAME);
    fs::create_dir_all(project_config.join("tools")).expect("project tools");
    fs::write(project_config.join("tools/my-tool.js"), "custom").expect("custom write");
    // The managed filter lowercases, so this never warns — and the exact-name
    // move list never touches it.
    fs::write(project_config.join("tools/RG.EXE"), "binary").expect("managed exe write");

    let (report, lines) = sweep_over(&agent, project.path());

    assert_eq!(
        report.deprecation_warnings,
        vec![
            "Global hooks/ directory found. Hooks have been renamed to extensions.".to_string(),
            "Project tools/ directory contains custom tools. Custom tools have been merged into extensions.".to_string(),
        ],
        "managed (fd/rg in any case) and hidden files never warn"
    );
    assert!(lines.is_empty(), "nothing printed: nothing moved");
    assert!(
        !agent.join("tools/fd").exists(),
        "the stale managed binary is deleted"
    );
    assert_eq!(
        fs::read_to_string(agent.join("bin/fd")).expect("target read"),
        "newer fd",
    );
}

// =============================================================================
// tools/ → bin/
// =============================================================================

#[test]
fn sweep_moves_managed_binaries_into_bin_and_prints_once() {
    let temp = temp_dir("pi-mig-bin-move-");
    fs::create_dir_all(temp.path().join("tools")).expect("tools dir");
    for name in ["fd", "rg", "fd.exe"] {
        fs::write(temp.path().join("tools").join(name), "binary").expect("binary write");
    }

    let (_report, lines) = sweep(temp.path());

    for name in ["fd", "rg", "fd.exe"] {
        assert!(
            temp.path().join("bin").join(name).exists(),
            "{name} lands in bin/"
        );
        assert!(!temp.path().join("tools").join(name).exists());
    }
    assert_eq!(
        lines,
        vec!["Migrated managed binaries tools/ → bin/".to_string()],
        "one print covers the whole move"
    );
}

#[test]
fn sweep_deletes_moved_binaries_when_the_target_already_exists() {
    let temp = temp_dir("pi-mig-bin-exists-");
    fs::create_dir_all(temp.path().join("tools")).expect("tools dir");
    fs::create_dir_all(temp.path().join("bin")).expect("bin dir");
    fs::write(temp.path().join("tools/fd"), "old fd").expect("old fd write");
    fs::write(temp.path().join("tools/rg"), "old rg").expect("old rg write");
    fs::write(temp.path().join("bin/fd"), "newer fd").expect("target fd write");

    let (_report, lines) = sweep(temp.path());

    assert_eq!(
        fs::read_to_string(temp.path().join("bin/fd")).expect("target read"),
        "newer fd",
        "the pre-installed binary wins"
    );
    assert!(
        !temp.path().join("tools/fd").exists(),
        "the stale copy is deleted"
    );
    assert_eq!(
        fs::read_to_string(temp.path().join("bin/rg")).expect("moved read"),
        "old rg",
        "the uncontested binary still moves"
    );
    assert_eq!(
        lines,
        vec!["Migrated managed binaries tools/ → bin/".to_string()],
        "the print rides any moved binary, not every one"
    );
}

#[test]
fn sweep_stays_silent_when_no_binary_moves() {
    let temp = temp_dir("pi-mig-bin-silent-");
    fs::create_dir_all(temp.path().join("tools")).expect("tools dir");
    fs::create_dir_all(temp.path().join("bin")).expect("bin dir");
    for name in ["fd.exe", "rg.exe"] {
        fs::write(temp.path().join("tools").join(name), "old").expect("old write");
        fs::write(temp.path().join("bin").join(name), "newer").expect("target write");
    }

    let (_report, lines) = sweep(temp.path());

    for name in ["fd.exe", "rg.exe"] {
        assert!(
            !temp.path().join("tools").join(name).exists(),
            "{name} deleted"
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("bin").join(name)).expect("target read"),
            "newer",
        );
    }
    assert!(lines.is_empty(), "nothing moved, so nothing prints");

    // A missing tools/ dir takes the early return without touching bin/.
    let empty = temp_dir("pi-mig-bin-empty-");
    let (_report, lines) = sweep(empty.path());
    assert!(lines.is_empty());
    assert!(!empty.path().join("bin").exists());
}

// =============================================================================
// keybindings.json through the migrator seam
// =============================================================================

/// Run the sweep with a counting migrator that answers `answer`.
fn sweep_with_keybindings(
    agent_dir: &Path,
    answer: Option<&(Map<String, Value>, bool)>,
    calls: &Cell<usize>,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut print_line = |line: &str| lines.push(line.to_string());
    let mut migrate = |_config: &Map<String, Value>| -> Option<(Map<String, Value>, bool)> {
        calls.set(calls.get() + 1);
        answer.cloned()
    };
    let _report = run_migrations_with_in(
        &agent_dir.to_string_lossy(),
        agent_dir,
        &mut print_line,
        &mut migrate,
    );
    lines
}

#[test]
fn keybindings_migration_writes_the_migrated_config_with_a_trailing_newline() {
    let temp = temp_dir("pi-mig-kb-write-");
    fs::write(
        temp.path().join("keybindings.json"),
        json!({"a": 1}).to_string(),
    )
    .expect("keybindings write");
    let calls = Cell::new(0);

    let lines = sweep_with_keybindings(
        temp.path(),
        Some(&(json!({"a": 2}).as_object().expect("object").clone(), true)),
        &calls,
    );

    assert_eq!(calls.get(), 1, "the migrator sees the parsed config");
    let expected = format!(
        "{}\n",
        serde_json::to_string_pretty(&json!({"a": 2})).expect("serialize")
    );
    assert_eq!(
        fs::read_to_string(temp.path().join("keybindings.json")).expect("keybindings read"),
        expected,
        "the write is pretty with the trailing newline"
    );
    assert!(lines.is_empty(), "the keybindings migration never prints");
}

#[test]
fn keybindings_migration_skips_the_write_when_the_migrator_reports_no_change() {
    let temp = temp_dir("pi-mig-kb-nochange-");
    let original = json!({"a": 1}).to_string();
    fs::write(temp.path().join("keybindings.json"), &original).expect("keybindings write");
    let calls = Cell::new(0);

    let migrated_false = Some(&(json!({"a": 2}).as_object().expect("object").clone(), false));
    sweep_with_keybindings(temp.path(), migrated_false, &calls);
    assert_eq!(
        fs::read_to_string(temp.path().join("keybindings.json")).expect("keybindings read"),
        original,
        "migrated=false leaves the file untouched"
    );

    sweep_with_keybindings(temp.path(), None, &calls);
    assert_eq!(
        fs::read_to_string(temp.path().join("keybindings.json")).expect("keybindings read"),
        original,
        "a None answer leaves the file untouched"
    );
}

#[test]
fn keybindings_migration_ignores_missing_malformed_and_non_object_files() {
    let calls = Cell::new(0);

    // Missing file: the migrator is never consulted.
    let missing = temp_dir("pi-mig-kb-missing-");
    sweep_with_keybindings(missing.path(), None, &calls);
    assert_eq!(calls.get(), 0, "no file, no consult");

    // Malformed file: ignored silently.
    let malformed = temp_dir("pi-mig-kb-malformed-");
    fs::write(malformed.path().join("keybindings.json"), "{invalid").expect("write");
    sweep_with_keybindings(malformed.path(), None, &calls);
    assert_eq!(
        calls.get(),
        0,
        "a malformed file never reaches the migrator"
    );
    assert_eq!(
        fs::read_to_string(malformed.path().join("keybindings.json")).expect("read"),
        "{invalid",
        "the malformed file is never rewritten"
    );

    // A non-object root: parsed but not an object, so the migrator skips.
    let array = temp_dir("pi-mig-kb-array-");
    fs::write(array.path().join("keybindings.json"), "[1,2]").expect("write");
    sweep_with_keybindings(array.path(), None, &calls);
    assert_eq!(
        calls.get(),
        0,
        "a non-object root never reaches the migrator"
    );
}

#[test]
fn keybindings_migration_swallows_a_failed_write() {
    let temp = temp_dir("pi-mig-kb-ro-");
    let config = temp.path().join("keybindings.json");
    fs::write(&config, json!({"a": 1}).to_string()).expect("keybindings write");
    fs::set_permissions(&config, fs::Permissions::from_mode(0o444))
        .expect("read-only the keybindings file");
    let calls = Cell::new(0);

    let lines = sweep_with_keybindings(
        temp.path(),
        Some(&(json!({"a": 2}).as_object().expect("object").clone(), true)),
        &calls,
    );

    assert_eq!(calls.get(), 1, "the migrator ran");
    assert!(lines.is_empty(), "the write failure never prints");
    assert_eq!(
        fs::read_to_string(&config).expect("keybindings read"),
        json!({"a": 1}).to_string(),
        "the failed write left the file untouched"
    );
    fs::set_permissions(&config, fs::Permissions::from_mode(0o644))
        .expect("restore the keybindings file");
}

// =============================================================================
// report aggregation
// =============================================================================

#[test]
fn sweep_aggregates_the_auth_providers_and_deprecation_warnings_into_the_report() {
    let temp = temp_dir("pi-mig-report-");
    fs::write(
        temp.path().join("oauth.json"),
        json!({"anthropic": {"access": "a", "refresh": "r", "expires": 1}}).to_string(),
    )
    .expect("oauth write");
    fs::create_dir_all(temp.path().join("hooks")).expect("hooks dir");

    let (report, _lines) = sweep(temp.path());

    assert_eq!(report.migrated_auth_providers, vec!["anthropic"]);
    assert_eq!(
        report.deprecation_warnings,
        vec!["Global hooks/ directory found. Hooks have been renamed to extensions.".to_string(),],
    );
}

// =============================================================================
// process-agent-dir entry points over the injected environment
// =============================================================================

#[test]
fn the_with_forms_drive_the_process_entries_through_the_injected_environment() {
    // upstream's test rig points `process.env[ENV_AGENT_DIR]` at a scratch
    // directory; the workspace cannot mutate the process environment, so the
    // `_with` forms inject the same override through the env seam.
    let temp = temp_dir("pi-mig-with-");
    let agent_dir = temp.path().to_string_lossy().into_owned();
    let env = agent_dir_env(&agent_dir);
    fs::write(
        temp.path().join("oauth.json"),
        json!({"anthropic": {"access": "a", "refresh": "r", "expires": 1}}).to_string(),
    )
    .expect("oauth write");
    fs::write(
        temp.path().join("settings.json"),
        json!({"apiKeys": {"openai": "sk-1"}}).to_string(),
    )
    .expect("settings write");
    fs::write(
        temp.path().join("session.jsonl"),
        json!({"type": "session", "cwd": "/tmp/pi-mig-with-cwd"}).to_string(),
    )
    .expect("session write");

    let providers = migrate_auth_to_auth_json_with(&env).expect("auth migration");
    migrate_sessions_from_agent_root_with(&env);
    let mut lines: Vec<String> = Vec::new();
    let report = {
        let mut print_line = |line: &str| lines.push(line.to_string());
        let mut no_keybindings =
            |_config: &Map<String, Value>| -> Option<(Map<String, Value>, bool)> { None };
        run_migrations_with(&env, "", &mut print_line, &mut no_keybindings)
    };

    assert_eq!(providers, vec!["anthropic", "openai"]);
    assert!(
        temp.path().join("auth.json").exists(),
        "the auth migration wrote the merged credentials"
    );
    assert_eq!(
        report.migrated_auth_providers,
        Vec::<String>::new(),
        "the sweep's auth migration sees the fresh auth.json and skips"
    );
    let session_dir = temp
        .path()
        .join("sessions")
        .join(encode_session_cwd("/tmp/pi-mig-with-cwd"));
    assert!(
        session_dir.join("session.jsonl").exists(),
        "the session migrated under the injected agent dir"
    );
    assert!(lines.is_empty(), "nothing in the sweep prints");
}

#[test]
fn the_default_printer_prints_and_the_default_keybindings_migrator_passes_through() {
    // upstream's unoverridden defaults: console.log to stdout, and the
    // keybindings config untouched until the interactive shell lands the
    // real migration table.
    print_migration_line("migrations boundary print");
    let config = json!({"a": 1});
    let outcome = no_keybindings_migration(config.as_object().expect("object"));

    assert_eq!(outcome, None);
}

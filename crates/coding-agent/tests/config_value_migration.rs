//! Upstream `test/config-value-migration.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`, restated.
//!
//! Upstream drives the sweep through `process.env[ENV_AGENT_DIR]`; the port
//! drives `run_migrations_with_in`, the env-independent sweep over an
//! explicit agent directory, with a line-capturing printer in place of the
//! stubbed `console.log` and a no-op keybindings migrator (the real table
//! rides the interactive-shell ticket). Upstream passes the agent dir as the
//! sweep's cwd, mirrored.
//!
//! Deferred to the model-registry ticket, which lands that slice: the
//! `createModelRegistry` halves of the malformed/blank `models.json` case
//! (`registry.getError()` naming "Failed to parse models.json" and the file
//! path) and of the uppercase-values case (`registry.find`,
//! `getApiKeyForProvider`, `getApiKeyAndHeaders`), plus the uppercase case's
//! env-var setup (`CUSTOM_API_KEY` and friends), which only the registry
//! half reads.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::fs;
use std::path::Path;

use pi_coding_agent::migrations::{MigrationReport, run_migrations_with_in};
use serde_json::{Map, Value, json};

/// Run the full migration sweep against `agent_dir` (its own cwd, upstream's
/// `runMigrations(agentDir)`), capturing the printed lines the printer seam
/// carries.
fn sweep(agent_dir: &Path) -> (MigrationReport, Vec<String>) {
    let mut lines: Vec<String> = Vec::new();
    let report = {
        let mut print_line = |line: &str| lines.push(line.to_string());
        let mut no_keybindings =
            |_config: &Map<String, Value>| -> Option<(Map<String, Value>, bool)> { None };
        run_migrations_with_in(
            &agent_dir.to_string_lossy(),
            agent_dir,
            &mut print_line,
            &mut no_keybindings,
        )
    };
    (report, lines)
}

/// Write `content` to `models.json` in a fresh agent dir and run the sweep;
/// the temp dir lives until the caller drops it.
fn sweep_with_models_json(content: &str) -> tempfile::TempDir {
    let temp = tempfile::Builder::new()
        .prefix("pi-config-value-migration-test-")
        .tempdir()
        .expect("agent dir");
    fs::write(temp.path().join("models.json"), content).expect("models write");
    sweep(temp.path());
    temp
}

#[test]
fn leaves_uppercase_auth_json_api_key_values_unchanged() {
    let temp = tempfile::Builder::new()
        .prefix("pi-config-value-migration-test-")
        .tempdir()
        .expect("agent dir");
    let auth_path = temp.path().join("auth.json");
    fs::write(
        &auth_path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&json!({
                "anthropic": { "type": "api_key", "key": "ANTHROPIC_API_KEY" },
                "openai": { "type": "api_key", "key": "$OPENAI_API_KEY" },
                "opencode": { "type": "api_key", "key": "public" },
                "github": { "type": "oauth", "access": "ACCESS_TOKEN", "refresh": "REFRESH_TOKEN", "expires": 1 },
            }))
            .expect("auth json serialize")
        ),
    )
    .expect("auth write");

    let (report, lines) = sweep(temp.path());

    // auth.json exists, so the whole auth migration skips it untouched
    assert!(report.migrated_auth_providers.is_empty());
    let migrated: Value = serde_json::from_str(&fs::read_to_string(&auth_path).expect("auth read"))
        .expect("auth parse");
    assert_eq!(migrated["anthropic"]["key"], "ANTHROPIC_API_KEY");
    assert_eq!(migrated["openai"]["key"], "$OPENAI_API_KEY");
    assert_eq!(migrated["opencode"]["key"], "public");
    assert_eq!(migrated["github"]["access"], "ACCESS_TOKEN");
    assert!(lines.is_empty());
}

#[test]
fn does_not_throw_on_malformed_models_json_during_migrations() {
    let temp = sweep_with_models_json("{\n  \"providers\": {\n");
    let models_path = temp.path().join("models.json");

    assert_eq!(
        fs::read_to_string(&models_path).expect("models read"),
        "{\n  \"providers\": {\n"
    );
    // registry.getError() naming "Failed to parse models.json" and the file
    // path: deferred to the model-registry ticket
}

#[test]
fn does_not_throw_on_blank_models_json_during_migrations() {
    let temp = sweep_with_models_json("");
    let models_path = temp.path().join("models.json");

    assert_eq!(fs::read_to_string(&models_path).expect("models read"), "");
    // registry.getError() naming "Failed to parse models.json" and the file
    // path: deferred to the model-registry ticket
}

#[test]
fn leaves_uppercase_models_json_api_key_and_header_values_unchanged() {
    let temp = tempfile::Builder::new()
        .prefix("pi-config-value-migration-test-")
        .tempdir()
        .expect("agent dir");
    let models_path = temp.path().join("models.json");
    fs::write(
        &models_path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&json!({
                "providers": {
                    "custom-provider": {
                        "baseUrl": "https://example.com/v1",
                        "apiKey": "CUSTOM_API_KEY",
                        "api": "openai-completions",
                        "headers": {
                            "x-api-key": "HEADER_API_KEY",
                            "x-literal": "literal",
                        },
                        "models": [
                            { "id": "model-a", "headers": { "x-model-key": "MODEL_API_KEY" } },
                        ],
                        "modelOverrides": {
                            "model-b": { "headers": { "x-override-key": "OVERRIDE_API_KEY" } },
                        },
                    },
                },
            }))
            .expect("models json serialize")
        ),
    )
    .expect("models write");

    let (_report, lines) = sweep(temp.path());

    let migrated: Value =
        serde_json::from_str(&fs::read_to_string(&models_path).expect("models read"))
            .expect("models parse");
    let provider = &migrated["providers"]["custom-provider"];
    assert_eq!(provider["apiKey"], "CUSTOM_API_KEY");
    assert_eq!(provider["headers"]["x-api-key"], "HEADER_API_KEY");
    assert_eq!(provider["headers"]["x-literal"], "literal");
    assert_eq!(
        provider["models"][0]["headers"]["x-model-key"],
        "MODEL_API_KEY"
    );
    assert_eq!(
        provider["modelOverrides"]["model-b"]["headers"]["x-override-key"],
        "OVERRIDE_API_KEY"
    );
    assert!(lines.is_empty());
    // registry.find/getApiKeyForProvider/getApiKeyAndHeaders: deferred to the
    // model-registry ticket
}

//! The TS-pi to Rust-pi migration tool, the one workspace crate with no
//! upstream counterpart (ADR 0003 addendum; decided by the import-tool
//! ticket #13).
//!
//! [`run_import`] discovers the source and target agent dirs, runs the four
//! migration legs — sessions, credentials, settings, trust — against the
//! Rust formats as the port tickets define them, enumerates the TS
//! extensions and npm packages the Rust pi cannot carry as skipped items,
//! applies the planned writes, and returns the [`report::ImportReport`] the
//! CLI renders. `--dry-run` plans every write and applies none, so the
//! report's outcomes and the writes come from one pass.
//!
//! The tool never executes `!cmd`/`${VAR}` indirections and never prints
//! key material; the house rules in AGENTS.md bind every leg.

pub mod cli;
pub mod credentials;
pub mod discovery;
pub mod plan;
pub mod report;
pub mod sessions;
pub mod settings;
pub mod trust;

use serde_json::Value;

use plan::{OpTarget, PlannedOp};
use report::{ImportReport, SkippedItem};

/// Run the migration: discover, run the legs, apply the planned writes
/// (unless dry-run), and report.
///
/// # Errors
/// The discovery refusals — a missing or unrecognized source dir, a missing
/// project dir — message-only for the CLI to print.
pub fn run_import(options: &discovery::ImportOptions) -> Result<ImportReport, String> {
    let target =
        pi_coding_agent::config::get_agent_dir_with(&pi_coding_agent::config::default_env_lookup());
    run_import_with_target(options, &target)
}

/// [`run_import`] against an explicit target agent dir, the test form the
/// env-derived default cannot pin down.
///
/// # Errors
/// The same refusal messages.
#[doc(hidden)]
pub fn run_import_with_target(
    options: &discovery::ImportOptions,
    target: &std::path::Path,
) -> Result<ImportReport, String> {
    let discovery = discovery::discover_with_target(options, target)?;
    let mut report = ImportReport {
        source: discovery.source.display().to_string(),
        target: discovery.target.display().to_string(),
        project: discovery
            .project
            .as_ref()
            .map(|path| path.display().to_string()),
        dry_run: options.dry_run,
        ..ImportReport::default()
    };

    let mut ops = sessions::run_sessions(&discovery, &mut report);

    let credentials = credentials::run_credentials(&discovery, &mut report);
    if let Some(plan) = &credentials.write {
        ops.push(credentials::auth_write_op(plan, &credentials));
    }
    ops.extend(settings::run_settings(
        &discovery,
        &mut report,
        credentials.settings_api_keys_consumed,
    ));
    ops.extend(trust::run_trust(&discovery, &mut report));

    report.skipped = enumerate_skipped(&discovery);
    apply_ops(&mut report, &ops, options.dry_run);
    Ok(report)
}

/// Apply the planned operations, downgrading each operation's report items
/// to failures when its filesystem work fails.
fn apply_ops(report: &mut ImportReport, ops: &[PlannedOp], dry_run: bool) {
    for op in ops {
        if let Err(reason) = op.apply(dry_run) {
            for target in op.targets() {
                match target {
                    OpTarget::Session(index) => report.fail_session(*index, &reason),
                    OpTarget::Credential(index) => report.fail_credential(*index, &reason),
                    OpTarget::Settings(index) => report.fail_settings(*index, &reason),
                    OpTarget::Trust => report.fail_trust(&reason),
                }
            }
        }
    }
}

/// The TS artifacts the Rust pi cannot carry: the source extensions dir's
/// contents, then the settings' extension paths and npm package entries,
/// reported as skipped per the import-tool decision and ADR 0007.
fn enumerate_skipped(discovery: &discovery::Discovery) -> Vec<SkippedItem> {
    let mut skipped = Vec::new();
    let extensions_dir = discovery.source.join("extensions");
    if extensions_dir.is_dir() {
        let mut children: Vec<_> = std::fs::read_dir(&extensions_dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .collect();
        children.sort();
        for child in children {
            skipped.push(SkippedItem {
                source: child.display().to_string(),
                kind: "extension".to_string(),
                reason: "TS extensions do not execute on the Rust pi".to_string(),
            });
        }
    }
    let Ok(content) = std::fs::read_to_string(discovery.source.join("settings.json")) else {
        return skipped;
    };
    let Ok(raw) = settings::parse_object(&content) else {
        return skipped;
    };
    if let Some(extensions) = raw.get("extensions").and_then(Value::as_array) {
        for extension in extensions {
            if let Some(path) = extension.as_str() {
                skipped.push(SkippedItem {
                    source: path.to_string(),
                    kind: "extension".to_string(),
                    reason: "TS extension files do not execute on the Rust pi".to_string(),
                });
            }
        }
    }
    if let Some(packages) = raw.get("packages").and_then(Value::as_array) {
        for package in packages {
            let label = package.as_str().map_or_else(
                || serde_json::to_string(package).unwrap_or_default(),
                str::to_string,
            );
            skipped.push(SkippedItem {
                source: label,
                kind: "package".to_string(),
                reason: "npm packages have no Rust equivalent".to_string(),
            });
        }
    }
    skipped
}

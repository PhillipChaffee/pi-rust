//! The command line, the report-UX decision's surface (import-tool ticket
//! #13): argv parsing, the text and JSON renders, and the exit codes.
//!
//! The exit is `0` when nothing failed, `1` when the report itemizes
//! failures, `2` for the refusals and usage errors.

#![expect(
    clippy::print_stdout,
    reason = "the tool's job is printing the migration report to stdout, the report-UX decision's surface"
)]
#![expect(
    clippy::print_stderr,
    reason = "refusals and usage errors print to stderr so stdout stays the report"
)]

use std::fmt::Write as _;

use crate::discovery::ImportOptions;
use crate::report::{
    CredentialOutcome, SessionOutcome, SettingsItem, SettingsOutcome, TrustOutcome,
};

/// The usage text, `--help`'s output.
const USAGE: &str = "pi-import: migrate TS-pi artifacts into the Rust pi's formats

usage: pi-import [--source <path>] [--project <path>] [--json] [--dry-run]

  --source <path>   the TS-pi agent dir (default: PI_CODING_AGENT_DIR, else ~/.pi/agent)
  --project <path>  migrate the project's .pi/settings.json too (default: off)
  --json            render the report as JSON
  --dry-run         report without writing
  -h, --help        this text

The exit is 0 when nothing failed, 1 when the report itemizes failures,
2 for a refused source or a usage error.";

/// Parse the argument vector, run the migration, render the report, and
/// return the process exit code.
#[must_use]
pub fn run_cli(args: Vec<String>) -> i32 {
    let target =
        pi_coding_agent::config::get_agent_dir_with(&pi_coding_agent::config::default_env_lookup());
    run_cli_with_target(args, &target)
}

/// [`run_cli`] against an explicit target agent dir, the test form the
/// env-derived default cannot pin down.
#[doc(hidden)]
#[must_use]
pub fn run_cli_with_target(args: Vec<String>, target: &std::path::Path) -> i32 {
    let options = match parse_args(args) {
        ParseOutcome::Options(options) => options,
        ParseOutcome::Help => {
            println!("{USAGE}");
            return 0;
        }
        ParseOutcome::Error(message) => {
            eprintln!("{message}");
            eprintln!("run `pi-import --help` for usage");
            return 2;
        }
    };
    match crate::run_import_with_target(&options, target) {
        Ok(report) => {
            if options.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".to_string())
                );
            } else {
                println!("{}", render_text(&report));
            }
            i32::from(report.has_failures())
        }
        Err(message) => {
            eprintln!("pi-import: {message}");
            2
        }
    }
}

/// The parse result: options, the help request, or a usage error.
enum ParseOutcome {
    Options(ImportOptions),
    Help,
    Error(String),
}

/// Parse the flags, rejecting unknown ones and missing values.
fn parse_args(args: Vec<String>) -> ParseOutcome {
    let mut options = ImportOptions::default();
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => return ParseOutcome::Help,
            "--json" => options.json = true,
            "--dry-run" => options.dry_run = true,
            "--source" | "--project" => {
                let Some(value) = iter.next() else {
                    return ParseOutcome::Error(format!("flag {arg} requires a path"));
                };
                if value.starts_with("--") {
                    return ParseOutcome::Error(format!("flag {arg} requires a path, got {value}"));
                }
                let path = std::path::PathBuf::from(value);
                if arg == "--source" {
                    options.source = Some(path);
                } else {
                    options.project = Some(path);
                }
            }
            other => {
                return ParseOutcome::Error(format!("unknown flag: {other}"));
            }
        }
    }
    ParseOutcome::Options(options)
}

/// The human-readable report, the default render.
#[must_use]
pub fn render_text(report: &crate::report::ImportReport) -> String {
    let mut text = String::new();
    let _ = writeln!(text, "pi-import: TS-pi -> Rust-pi migration");
    let _ = writeln!(text, "  source: {}", report.source);
    let _ = writeln!(text, "  target: {}", report.target);
    let _ = writeln!(
        text,
        "  project: {}",
        report.project.as_deref().unwrap_or("none")
    );
    if report.dry_run {
        let _ = writeln!(text, "  mode: dry-run (no writes)");
    } else {
        let _ = writeln!(text, "  mode: write");
    }
    render_sessions(report, &mut text);
    render_credentials(report, &mut text);
    render_settings(report, &mut text);
    render_trust(report, &mut text);
    render_skipped(report, &mut text);
    render_failures(report, &mut text);
    text
}

/// The sessions section, per-artifact lines and the outcome summary.
fn render_sessions(report: &crate::report::ImportReport, text: &mut String) {
    if report.sessions.is_empty() {
        let _ = writeln!(text, "sessions: none");
        return;
    }
    let _ = writeln!(text, "sessions");
    for item in &report.sessions {
        let line = match &item.outcome {
            SessionOutcome::Copied { placed_to } => {
                format!("  copied   {} -> {placed_to}", item.path)
            }
            SessionOutcome::Present => format!("  present  {}", item.path),
            SessionOutcome::Skipped { reason } => {
                format!("  skipped  {} ({reason})", item.path)
            }
            SessionOutcome::Failed { reason } => {
                format!("  failed   {} ({reason})", item.path)
            }
        };
        match item.version {
            Some(version) => {
                let _ = writeln!(text, "{line} (v{version})");
            }
            None => {
                let _ = writeln!(text, "{line}");
            }
        }
    }
    let (copied, present, skipped, failed) = report.sessions.iter().fold(
        (0u64, 0u64, 0u64, 0u64),
        |(copied, present, skipped, failed), item| match item.outcome {
            SessionOutcome::Copied { .. } => (copied + 1, present, skipped, failed),
            SessionOutcome::Present => (copied, present + 1, skipped, failed),
            SessionOutcome::Skipped { .. } => (copied, present, skipped + 1, failed),
            SessionOutcome::Failed { .. } => (copied, present, skipped, failed + 1),
        },
    );
    let _ = writeln!(
        text,
        "  summary: {copied} copied, {present} present, {skipped} skipped, {failed} failed"
    );
}

/// The credentials section, provider ids and kinds only.
fn render_credentials(report: &crate::report::ImportReport, text: &mut String) {
    if report.credentials.is_empty() {
        let _ = writeln!(text, "credentials: none");
        return;
    }
    let _ = writeln!(
        text,
        "credentials (from {})",
        report.credential_sources.join(", ")
    );
    for item in &report.credentials {
        let kind = item.kind.as_deref().unwrap_or("unknown");
        let line = match &item.outcome {
            CredentialOutcome::Imported => format!("  imported   {} ({kind})", item.provider),
            CredentialOutcome::Normalized => format!(
                "  normalized {} ({kind}, expires floored to an integer)",
                item.provider
            ),
            CredentialOutcome::Kept => format!("  kept       {}", item.provider),
            CredentialOutcome::Skipped { reason } => {
                format!("  skipped    {} ({reason})", item.provider)
            }
            CredentialOutcome::Failed { reason } => {
                format!("  failed     {} ({reason})", item.provider)
            }
        };
        let _ = writeln!(text, "{line}");
    }
    let (imported, normalized, kept, failed) = report.credentials.iter().fold(
        (0u64, 0u64, 0u64, 0u64),
        |(imported, normalized, kept, failed), item| match item.outcome {
            CredentialOutcome::Imported => (imported + 1, normalized, kept, failed),
            CredentialOutcome::Normalized => (imported, normalized + 1, kept, failed),
            CredentialOutcome::Kept => (imported, normalized, kept + 1, failed),
            CredentialOutcome::Skipped { .. } => (imported, normalized, kept, failed),
            CredentialOutcome::Failed { .. } => (imported, normalized, kept, failed + 1),
        },
    );
    let _ = writeln!(
        text,
        "  summary: {imported} imported ({normalized} normalized), {kept} kept, {failed} failed"
    );
}

/// The settings section, per-scope lines with the migration notes.
fn render_settings(report: &crate::report::ImportReport, text: &mut String) {
    let _ = writeln!(text, "settings");
    for item in &report.settings {
        let notes = settings_notes(item);
        let line = match &item.outcome {
            SettingsOutcome::Written => format!("  {} {}: written{notes}", item.scope, item.path),
            SettingsOutcome::Unchanged => {
                format!("  {} {}: unchanged{notes}", item.scope, item.path)
            }
            SettingsOutcome::Skipped { reason } => {
                format!("  {} {}: skipped ({reason})", item.scope, item.path)
            }
            SettingsOutcome::Failed { reason } => {
                format!("  {} {}: failed ({reason})", item.scope, item.path)
            }
        };
        let _ = writeln!(text, "{line}");
    }
}

/// The migration notes suffix, the merged/dropped lists and key count.
fn settings_notes(item: &SettingsItem) -> String {
    let mut notes = String::new();
    if !item.merged.is_empty() {
        notes.push_str(" (merged: ");
        notes.push_str(&item.merged.join("; "));
        notes.push(')');
    }
    if !item.dropped.is_empty() {
        notes.push_str(" (dropped: ");
        notes.push_str(&item.dropped.join("; "));
        notes.push(')');
    }
    notes.push_str(" (");
    let _ = write!(notes, "{}", item.carried_keys);
    notes.push_str(" keys)");
    notes
}

/// The trust section, the one store's outcome.
fn render_trust(report: &crate::report::ImportReport, text: &mut String) {
    let Some(item) = &report.trust else {
        return;
    };
    let line = match &item.outcome {
        TrustOutcome::Written { entries } => {
            format!("trust\n  {}: copied ({entries} entries)", item.path)
        }
        TrustOutcome::Unchanged { entries } => {
            format!("trust\n  {}: unchanged ({entries} entries)", item.path)
        }
        TrustOutcome::Skipped { reason } => {
            format!("trust\n  {}: skipped ({reason})", item.path)
        }
        TrustOutcome::Failed { reason } => {
            format!("trust\n  {}: failed ({reason})", item.path)
        }
    };
    let _ = writeln!(text, "{line}");
}

/// The skipped extensions and packages, the decision's enumeration.
fn render_skipped(report: &crate::report::ImportReport, text: &mut String) {
    if report.skipped.is_empty() {
        return;
    }
    let _ = writeln!(
        text,
        "skipped TS extensions and packages (a Rust pi cannot execute them)"
    );
    for item in &report.skipped {
        let _ = writeln!(text, "  {:9} {}", item.kind, item.source);
    }
}

/// The itemized failures, the re-run targets; omitted when none.
fn render_failures(report: &crate::report::ImportReport, text: &mut String) {
    if report.failures().is_empty() {
        return;
    }
    let _ = writeln!(text, "failures");
    for failure in report.failures() {
        let _ = writeln!(text, "  {}: {}", failure.artifact, failure.detail);
    }
}

//! Input discovery, the import-tool decision's first question.
//!
//! The default source honors TS pi's own override — `PI_CODING_AGENT_DIR`
//! when set, else `~/.pi/agent` — `--source` overrides explicitly, and a
//! source that does not look like a TS-pi dir is refused with an
//! explanation, not guessed at. The target is always the Rust pi's own
//! agent dir, derived the same way; project-side artifacts migrate only
//! with `--project`.

use std::path::{Path, PathBuf};

use pi_coding_agent::config::{default_env_lookup, get_agent_dir_with, home_dir};
use pi_coding_agent::utils::paths::resolve_path;

/// The options the CLI hands the run.
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// The source agent dir override, `--source`.
    pub source: Option<PathBuf>,
    /// The project dir, `--project`; project-side artifacts migrate only
    /// when this is set.
    pub project: Option<PathBuf>,
    /// Whether the report renders as JSON.
    pub json: bool,
    /// Whether the run reports without writing.
    pub dry_run: bool,
}

/// The resolved inputs a run works from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Discovery {
    /// The source agent dir.
    pub source: PathBuf,
    /// The target agent dir, the Rust pi's own derivation.
    pub target: PathBuf,
    /// The project dir when `--project` was given.
    pub project: Option<PathBuf>,
    /// Whether source and target resolve to the same directory, the
    /// in-place normalization mode.
    pub in_place: bool,
}

/// The artifacts a TS-pi dir is recognized by, the refusal check's set.
const RECOGNIZED_ARTIFACTS: [&str; 9] = [
    "auth.json",
    "oauth.json",
    "settings.json",
    "trust.json",
    "models.json",
    "sessions",
    "extensions",
    "prompts",
    "themes",
];

/// Resolve the source, target, and project dirs and validate the source as
/// a TS-pi dir.
///
/// # Errors
/// The refusal messages: a missing or unrecognized source, a missing
/// project dir.
pub fn discover(options: &ImportOptions) -> Result<Discovery, String> {
    let target = get_agent_dir_with(&default_env_lookup());
    discover_with_target(options, &target)
}

/// [`discover`] against an explicit target agent dir, the test form the
/// env-derived default cannot pin down.
///
/// # Errors
/// The same refusal messages.
#[doc(hidden)]
pub fn discover_with_target(options: &ImportOptions, target: &Path) -> Result<Discovery, String> {
    let source = options
        .source
        .as_ref()
        .map_or_else(|| target.to_path_buf(), Clone::clone);
    if !source.is_dir() {
        return Err(format!(
            "source directory does not exist: {}",
            source.display()
        ));
    }
    if !looks_like_ts_agent_dir(&source) {
        return Err(format!(
            "source directory does not look like a TS-pi agent dir: {}\nlooked for {}",
            source.display(),
            RECOGNIZED_ARTIFACTS.join(", ")
        ));
    }
    if let Some(project) = &options.project
        && !project.is_dir()
    {
        return Err(format!(
            "project directory does not exist: {}",
            project.display()
        ));
    }
    let project = options.project.clone();
    let in_place = resolve(&source) == resolve(target);
    Ok(Discovery {
        source,
        target: target.to_path_buf(),
        project,
        in_place,
    })
}

/// Whether the directory carries at least one artifact TS pi writes, the
/// recognition rule the refusal turns on.
fn looks_like_ts_agent_dir(source: &Path) -> bool {
    RECOGNIZED_ARTIFACTS
        .iter()
        .any(|artifact| source.join(artifact).exists())
}

/// The resolver the in-place check runs through, the same one the agent-dir
/// derivation uses.
fn resolve(path: &Path) -> String {
    resolve_path(&path.display().to_string(), &process_cwd(), &home_dir())
}

/// The process cwd, upstream's `process.cwd()`.
fn process_cwd() -> String {
    std::env::current_dir()
        .map(|cwd| cwd.display().to_string())
        .unwrap_or_default()
}

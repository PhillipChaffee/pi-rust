//! Project trust decisions on disk, upstream's
//! `packages/coding-agent/src/core/trust-manager.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The store maps canonicalized absolute paths to a boolean decision, lookups
//! climb parent directories until an entry lands or the root stops the climb
//! (upstream's `null` stop marker is the file's way to record "decided
//! nothing here" — a `null` value entry is rejected on read and a `null`
//! decision deletes), and every read-modify-write serializes behind the
//! `trust.json.lock` directory. The canonical-path map is
//! [`crate::utils::paths::canonicalize_path`] over the default-options
//! resolver, upstream's `normalizeCwd`.

use std::collections::BTreeMap;
use std::path::Path;

use pi_ai::types::BoxedFuture;
use serde_json::Value;

use crate::config::CONFIG_DIR_NAME;
use crate::file_lock::{acquire_sync_retrying, lock_dir_for};
use crate::utils::paths::{canonicalize_path, resolve_path};
use crate::utils::text::strip_bom;

/// One trust decision, upstream's `ProjectTrustDecision`: `None` deletes an
/// entry, a value records it.
pub type ProjectTrustDecision = Option<bool>;

/// One stored entry, upstream's `ProjectTrustStoreEntry`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectTrustStoreEntry {
    /// The canonicalized path the decision is stored under.
    pub path: String,
    /// The recorded decision.
    pub decision: bool,
}

/// One recorded decision, upstream's `ProjectTrustUpdate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectTrustUpdate {
    /// The path to decide.
    pub path: String,
    /// The decision: `None` deletes the entry.
    pub decision: ProjectTrustDecision,
}

/// One trust prompt option, upstream's `ProjectTrustOption`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectTrustOption {
    /// The label the selector shows.
    pub label: String,
    /// Whether choosing this option trusts the project.
    pub trusted: bool,
    /// The store updates choosing applies.
    pub updates: Vec<ProjectTrustUpdate>,
    /// The path the decision persists under, when it persists.
    pub saved_path: Option<String>,
}

/// The project-local resources that must sit behind project trust, upstream's
/// `TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES`.
const TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES: [&str; 7] = [
    "settings.json",
    "extensions",
    "skills",
    "prompts",
    "themes",
    "SYSTEM.md",
    "APPEND_SYSTEM.md",
];

/// The stored file shape, upstream's `TrustFile`: path-keyed decisions where
/// a `null` value is a legal stored entry the lookups climb past.
type TrustFile = BTreeMap<String, Option<bool>>;

/// One trust-store failure, carrying the upstream message verbatim.
#[derive(Debug)]
pub struct TrustError(String);

impl std::fmt::Display for TrustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TrustError {}

fn trust_error(message: impl Into<String>) -> TrustError {
    TrustError(message.into())
}

/// Canonicalize a cwd, upstream's `normalizeCwd`.
fn normalize_cwd(cwd: &str) -> String {
    canonicalize_path(&resolve_cwd_default(cwd))
}

/// Upstream's one-argument `resolvePath(input)`: the belt resolver over the
/// process cwd as the base.
fn resolve_cwd_default(input: &str) -> String {
    let base = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    resolve_path(input, &base, &crate::config::home_dir())
}

/// The parent directory of `dir`, or `None` at the filesystem root, the
/// ascent bound upstream's `dirname(currentDir) === currentDir` check.
fn parent_dir(dir: &str) -> Option<String> {
    let parent = Path::new(dir).parent()?.to_string_lossy().into_owned();
    if parent == dir { None } else { Some(parent) }
}

/// The nearest recorded decision for a cwd, upstream's
/// `findNearestTrustEntry`: climb parents until a boolean entry lands or the
/// root stops the climb. A stored `null` entry climbs past, the same as a
/// missing key.
fn find_nearest_trust_entry(data: &TrustFile, cwd: &str) -> Option<ProjectTrustStoreEntry> {
    let mut current_dir = normalize_cwd(cwd);
    loop {
        if let Some(Some(decision)) = data.get(&current_dir) {
            return Some(ProjectTrustStoreEntry {
                path: current_dir,
                decision: *decision,
            });
        }
        current_dir = parent_dir(&current_dir)?;
    }
}

/// The parent directory a "trust parent folder" decision would record,
/// upstream's `getProjectTrustParentPath`.
#[must_use]
pub fn get_project_trust_parent_path(cwd: &str) -> Option<String> {
    let trust_path = normalize_cwd(cwd);
    parent_dir(&trust_path)
}

/// The options the trust prompt offers, upstream's `getProjectTrustOptions`.
///
/// Trust here; trust the parent (recording a `None` for this path so the
/// parent decision governs); the session-only variants; and refuse.
#[must_use]
pub fn get_project_trust_options(cwd: &str, include_session_only: bool) -> Vec<ProjectTrustOption> {
    let trust_path = normalize_cwd(cwd);
    let mut trust_options = vec![ProjectTrustOption {
        label: "Trust".to_string(),
        trusted: true,
        updates: vec![ProjectTrustUpdate {
            path: trust_path.clone(),
            decision: Some(true),
        }],
        saved_path: Some(trust_path.clone()),
    }];
    if let Some(parent_path) = get_project_trust_parent_path(cwd) {
        trust_options.push(ProjectTrustOption {
            label: format!("Trust parent folder ({parent_path})"),
            trusted: true,
            updates: vec![
                ProjectTrustUpdate {
                    path: parent_path.clone(),
                    decision: Some(true),
                },
                ProjectTrustUpdate {
                    path: trust_path.clone(),
                    decision: None,
                },
            ],
            saved_path: Some(parent_path),
        });
    }
    if include_session_only {
        trust_options.push(ProjectTrustOption {
            label: "Trust (this session only)".to_string(),
            trusted: true,
            updates: Vec::new(),
            saved_path: None,
        });
    }
    trust_options.push(ProjectTrustOption {
        label: "Do not trust".to_string(),
        trusted: false,
        updates: vec![ProjectTrustUpdate {
            path: trust_path.clone(),
            decision: Some(false),
        }],
        saved_path: Some(trust_path),
    });
    if include_session_only {
        trust_options.push(ProjectTrustOption {
            label: "Do not trust (this session only)".to_string(),
            trusted: false,
            updates: Vec::new(),
            saved_path: None,
        });
    }
    trust_options
}

/// Read the store file, upstream's `readTrustFile`: a missing file is the
/// empty map, every value must be a boolean or `null`.
fn read_trust_file(path: &str) -> Result<TrustFile, TrustError> {
    if !Path::new(path).exists() {
        return Ok(TrustFile::new());
    }
    let content = std::fs::read_to_string(path).map_err(|error| trust_error(error.to_string()))?;
    let parsed: Value = serde_json::from_str(strip_bom(&content))
        .map_err(|error| trust_error(format!("Failed to read trust store {path}: {error}")))?;
    let Value::Object(entries) = parsed else {
        return Err(trust_error(format!(
            "Invalid trust store {path}: expected an object"
        )));
    };
    let mut data = TrustFile::new();
    for (key, value) in entries {
        let decision = match value {
            Value::Bool(decision) => Some(decision),
            Value::Null => None,
            _other => {
                return Err(trust_error(format!(
                    "Invalid trust store {path}: value for {} must be true, false, or null",
                    serde_json::to_string(&key).unwrap_or_default()
                )));
            }
        };
        data.insert(key, decision);
    }
    Ok(data)
}

/// Write the store file, upstream's `writeTrustFile`: keys sorted, `null`
/// entries preserved, a trailing newline, the parent directory created as
/// needed.
fn write_trust_file(path: &str, data: &TrustFile) -> Result<(), TrustError> {
    let sorted: serde_json::Map<String, Value> = data
        .iter()
        .map(|(key, decision)| {
            let value = decision.map_or_else(|| Value::Null, Value::Bool);
            (key.clone(), value)
        })
        .collect();
    #[expect(
        clippy::expect_used,
        reason = "a serde_json map of parsed JSON values serializes; only non-self-describing formats fail"
    )]
    let content = format!(
        "{}\n",
        serde_json::to_string_pretty(&sorted).expect("trust decisions serialize")
    );
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(|error| trust_error(error.to_string()))?;
    }
    std::fs::write(path, content).map_err(|error| trust_error(error.to_string()))
}

/// The trust store, upstream's `ProjectTrustStore`: decisions under
/// `<agentDir>/trust.json`, every access behind the file lock.
pub struct ProjectTrustStore {
    trust_path: String,
}

impl std::fmt::Debug for ProjectTrustStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectTrustStore")
            .field("trust_path", &self.trust_path)
            .finish()
    }
}

impl ProjectTrustStore {
    /// The store under `agent_dir`, upstream's constructor.
    #[must_use]
    pub fn new(agent_dir: &str) -> Self {
        Self {
            trust_path: format!("{}/trust.json", resolve_cwd_default(agent_dir)),
        }
    }

    /// The decision governing `cwd`, climbing parents; `None` when nothing
    /// up the tree decided, upstream's `get`.
    ///
    /// # Errors
    /// A malformed store file.
    pub fn get(&self, cwd: &str) -> Result<ProjectTrustDecision, TrustError> {
        Ok(self.get_entry(cwd)?.map(|entry| entry.decision))
    }

    /// The nearest governing entry, upstream's `getEntry`.
    ///
    /// # Errors
    /// A malformed store file or a lock failure.
    pub fn get_entry(&self, cwd: &str) -> Result<Option<ProjectTrustStoreEntry>, TrustError> {
        let lock_dir = lock_dir_for(&self.trust_path);
        std::fs::create_dir_all(
            Path::new(&self.trust_path)
                .parent()
                .unwrap_or_else(|| Path::new(".")),
        )
        .map_err(|error| trust_error(error.to_string()))?;
        let guard =
            acquire_sync_retrying(&lock_dir).map_err(|error| trust_error(error.to_string()))?;
        // The read outcome is computed before the release so a malformed file
        // does not leave the lock directory behind, upstream's `finally`.
        let outcome =
            read_trust_file(&self.trust_path).map(|data| find_nearest_trust_entry(&data, cwd));
        guard
            .release()
            .map_err(|error| trust_error(error.to_string()))?;
        outcome
    }

    /// Record one decision, upstream's `set`.
    ///
    /// # Errors
    /// A malformed store file, a lock failure, or a write failure.
    pub fn set(&self, cwd: &str, decision: ProjectTrustDecision) -> Result<(), TrustError> {
        self.set_many(&[ProjectTrustUpdate {
            path: cwd.to_string(),
            decision,
        }])
    }

    /// Record several decisions in one locked write, upstream's `setMany`.
    ///
    /// # Errors
    /// A malformed store file, a lock failure, or a write failure.
    pub fn set_many(&self, decisions: &[ProjectTrustUpdate]) -> Result<(), TrustError> {
        let lock_dir = lock_dir_for(&self.trust_path);
        std::fs::create_dir_all(
            Path::new(&self.trust_path)
                .parent()
                .unwrap_or_else(|| Path::new(".")),
        )
        .map_err(|error| trust_error(error.to_string()))?;
        let guard =
            acquire_sync_retrying(&lock_dir).map_err(|error| trust_error(error.to_string()))?;
        let outcome = (|| {
            let mut data = read_trust_file(&self.trust_path)?;
            for update in decisions {
                let key = normalize_cwd(&update.path);
                match update.decision {
                    None => {
                        data.remove(&key);
                    }
                    Some(decision) => {
                        data.insert(key, Some(decision));
                    }
                }
            }
            write_trust_file(&self.trust_path, &data)
        })();
        guard
            .release()
            .map_err(|error| trust_error(error.to_string()))?;
        outcome
    }
}

/// Whether `cwd` has project-local resources that must be gated by project
/// trust, upstream's `hasTrustRequiringProjectResources`.
///
/// Trust-requiring entries under `cwd/.pi`, or `.agents/skills` in `cwd` or
/// one of its ancestors, count. The user's `~/.agents/skills` directory is
/// always a trusted user resource and never counts, even when `cwd` is the
/// home directory.
#[must_use]
pub fn has_trust_requiring_project_resources(cwd: &str) -> bool {
    has_trust_requiring_project_resources_with_home(cwd, &crate::config::home_dir())
}

/// [`has_trust_requiring_project_resources`] over an injected home, the test
/// seam for upstream's `process.env.HOME || homedir()`.
#[must_use]
pub fn has_trust_requiring_project_resources_with_home(cwd: &str, home: &str) -> bool {
    let home_dir = canonicalize_path(&resolve_cwd_default(home));
    let user_agents_skills_dir = Path::new(&home_dir).join(".agents").join("skills");
    let mut current_dir = canonicalize_path(&resolve_cwd_default(cwd));

    let config_dir = Path::new(&current_dir).join(CONFIG_DIR_NAME);
    if TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES
        .iter()
        .any(|entry| config_dir.join(entry).exists())
    {
        return true;
    }

    loop {
        let agents_skills_dir = Path::new(&current_dir).join(".agents").join("skills");
        if agents_skills_dir != user_agents_skills_dir && agents_skills_dir.exists() {
            return true;
        }
        let Some(parent) = parent_dir(&current_dir) else {
            return false;
        };
        current_dir = parent;
    }
}

/// The trust context the resolution flow needs from its host, upstream's
/// `ProjectTrustContext` narrowed to the surface this slice consumes.
///
/// The full context — the extension emission and the interactive UI — lands
/// with the extension system and interactive mode.
pub trait ProjectTrustSelector: Send + Sync {
    /// Offer the labels and resolve the chosen one, upstream's
    /// `ctx.ui.select`.
    fn select(&self, prompt: String, options: Vec<String>) -> BoxedFuture<'static, Option<String>>;

    /// Whether a UI can ask, upstream's `ctx.hasUI`.
    fn has_ui(&self) -> bool;
}
